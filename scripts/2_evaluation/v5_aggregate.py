"""v5 비교 집계(재설정 문서 6절). v5_compare.py 의 예측 파일을 읽어 지표·짝지은 차이를 만든다.

- 씨앗 평균 예측(앙상블)으로 지표를 계산하고, 씨앗별 RMSE 의 표준편차도 적는다.
- 지표: RMSE·편향·MAE·r·R²·블록 등가중 RMSE, 연간 편차 지표(scheme year·block_year·random·block 모두 계산, 셀에 2개 이상 연도가 있는 행만).
- 짝지은 ΔRMSE(블록 부트스트랩 95 % CI): 각 방법 대 기준(P1, P0), 같은 (region, scheme, forcing, featset) 안에서 같은 cell_year_id 끼리.
- 강제력 판 비교: 같은 방법의 year 판 − clim 판(같은 행).
출력: <run>/summary_metrics.csv, <run>/deltas_vs_ref.csv, <run>/deltas_forcing.csv
실행: python3 scripts/2_evaluation/v5_aggregate.py --run data/processed/v5/runs/<tag> [--nboot 2000]
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

KEY = ["region", "scheme", "forcing", "featset", "method", "learner", "lam"]


def load_preds(run, ref_run=None, ref_methods=("P0", "P1")):
    """실행 폴더의 예측 파일. ref_run 을 주면 그 실행의 기준 방법(P0·P1) 파일도 더한다(학습기 축처럼 기준이 없는 실행용)."""
    fs = sorted(glob.glob(str(Path(run) / "preds" / "*.csv.gz")))
    if ref_run:
        have = {Path(f).name for f in fs}
        for m in ref_methods:
            fs += [f for f in sorted(glob.glob(str(Path(ref_run) / "preds" / f"*__{m}__*.csv.gz"))) if Path(f).name not in have]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def ensemble(df):
    """씨앗 평균 예측. 반환 (앙상블 표, 씨앗별 RMSE 표준편차 표)."""
    g = df.groupby(KEY + ["cell_year_id"], as_index=False).agg(pred=("pred", "mean"), y=("y", "first"), block=("block", "first"),
                                                                 loc_id=("loc_id", "first"), year=("year", "first"), n_seed=("seed", "nunique"))
    per_seed = df.groupby(KEY + ["seed"]).apply(lambda s: T.rmse(s.y.values, s.pred.values)).rename("rmse").reset_index()
    sd = per_seed.groupby(KEY).rmse.agg(rmse_seed_sd="std", n_seed="size").reset_index()
    return g, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--refs", default="P1,P0")
    ap.add_argument("--conditions", action="store_true", help="TDD 구간별 조건 분석도 만든다")
    ap.add_argument("--ref-run", default="", help="기준 방법(P0·P1) 예측을 가져올 다른 실행 폴더")
    ap.add_argument("--forcing-pairs", default="year:clim", help="강제력 판 비교 쌍(첫째 − 둘째), 쉼표로 여러 개. 예 year_obs:year")
    a = ap.parse_args()
    run = ROOT / a.run
    df = load_preds(run, ROOT / a.ref_run if a.ref_run else None)
    if df.empty:
        raise SystemExit("예측 파일 없음")
    df["lam"] = df.lam.astype(float)
    ens, sd = ensemble(df)
    rows = []
    for k, s in ens.groupby(KEY):
        m = T.metrics(s.y.values, s.pred.values, s.block.values)
        m.update(T.anomaly_metrics(s.y.values, s.pred.values, s.loc_id.values))
        m.update(dict(zip(KEY, k)), n_blocks=int(s.block.nunique()), n_cells=int(s.loc_id.nunique()))
        rows.append(m)
    summ = pd.DataFrame(rows).merge(sd, on=KEY, how="left")
    summ.to_csv(run / "summary_metrics.csv", index=False)

    # 기준 대비 짝지은 차이
    drows = []
    grp = ["region", "scheme", "forcing", "featset"]
    for ref in a.refs.split(","):
        for g, s in ens.groupby(grp):
            r = s[(s.method == ref)]
            if r.empty:
                continue
            r = r.drop_duplicates("cell_year_id").set_index("cell_year_id")
            for k, t in s.groupby(["method", "learner", "lam"]):
                if k[0] == ref:
                    continue
                t = t.drop_duplicates("cell_year_id").set_index("cell_year_id")
                idx = t.index.intersection(r.index)
                if len(idx) < 10:
                    continue
                bd = T.boot_delta(t.y.loc[idx].values, t.pred.loc[idx].values, r.pred.loc[idx].values, t.block.loc[idx].values, a.nboot)
                drows.append(dict(zip(grp, g)) | dict(method=k[0], learner=k[1], lam=k[2], ref=ref) | bd)
    pd.DataFrame(drows).to_csv(run / "deltas_vs_ref.csv", index=False)

    # 강제력 판 쌍 비교(첫째 − 둘째)
    frows = []
    g2 = ["region", "scheme", "featset", "method", "learner", "lam"]
    for pair in a.forcing_pairs.split(","):
        fa, fb = pair.split(":")
        for k, s in ens.groupby(g2):
            y_ = s[s.forcing == fa].drop_duplicates("cell_year_id").set_index("cell_year_id")
            c_ = s[s.forcing == fb].drop_duplicates("cell_year_id").set_index("cell_year_id")
            idx = y_.index.intersection(c_.index)
            if len(idx) < 10:
                continue
            bd = T.boot_delta(y_.y.loc[idx].values, y_.pred.loc[idx].values, c_.pred.loc[idx].values, y_.block.loc[idx].values, a.nboot)
            ay = T.anomaly_metrics(y_.y.loc[idx].values, y_.pred.loc[idx].values, y_.loc_id.loc[idx].values)
            ac = T.anomaly_metrics(c_.y.loc[idx].values, c_.pred.loc[idx].values, c_.loc_id.loc[idx].values)
            frows.append(dict(zip(g2, k)) | dict(pair=pair) | bd | dict(r_anom_year=ay["r_anom"], r_anom_clim=ac["r_anom"],
                                                                     skill_anom_year=ay["skill_anom"], skill_anom_clim=ac["skill_anom"]))
    pd.DataFrame(frows).to_csv(run / "deltas_forcing.csv", index=False)
    show = summ[["region", "scheme", "forcing", "featset", "method", "learner", "lam", "n", "rmse", "bias", "r", "rmse_beq", "r_anom", "skill_anom"]]
    print(show.round(3).to_string(index=False))
    if drows:
        d = pd.DataFrame(drows)
        print(d[d.ref == "P1"][["region", "scheme", "forcing", "method", "learner", "lam", "delta", "lo", "hi", "delta_beq", "lo_beq", "hi_beq"]].round(2).to_string(index=False))
    if a.conditions:
        c = conditions(a.run, max(200, a.nboot // 2))
        print(c[["region", "scheme", "forcing", "method", "lam", "tdd_bin", "delta", "lo", "hi"]].round(2).to_string(index=False)[:4000])



# ================================================================ 조건 분석(재설정 문서 Q3): TDD 구간·지역별 ΔRMSE(방법 − P1)
def conditions(run, nboot=1000):
    """예측 표에 셀-연도의 TDD(연도 정합)를 붙여 지역별 TDD 3분위 구간에서 짝지은 ΔRMSE(방법 − P1)를 낸다."""
    run = ROOT / run
    df = load_preds(run); df["lam"] = df.lam.astype(float)
    ens, _ = ensemble(df)
    base = T.load_v5(ROOT / "data" / "processed" / "v5", products=False)[["cell_year_id", "y_e5_tdd", "loc_id"]].rename(columns={"y_e5_tdd": "tdd"})
    ens = ens.merge(base[["cell_year_id", "tdd"]], on="cell_year_id", how="left")
    rows = []
    grp = ["region", "scheme", "forcing", "featset"]
    for g, s in ens.groupby(grp):
        r = s[s.method == "P1"].drop_duplicates("cell_year_id").set_index("cell_year_id")
        if r.empty:
            continue
        q = np.nanpercentile(r.tdd, [33.3, 66.7])
        r["tdd_bin"] = np.digitize(r.tdd, q)
        for k, t in s.groupby(["method", "learner", "lam"]):
            if k[0] == "P1":
                continue
            t = t.drop_duplicates("cell_year_id").set_index("cell_year_id")
            for b in (0, 1, 2):
                idx = t.index.intersection(r.index[r.tdd_bin == b])
                if len(idx) < 30:
                    continue
                bd = T.boot_delta(t.y.loc[idx].values, t.pred.loc[idx].values, r.pred.loc[idx].values, t.block.loc[idx].values, nboot)
                rows.append(dict(zip(grp, g)) | dict(method=k[0], learner=k[1], lam=k[2], tdd_bin=b, tdd_lo=float(np.nanmin(r.tdd.loc[idx])),
                                                     tdd_hi=float(np.nanmax(r.tdd.loc[idx])), rmse_P1=T.rmse(r.y.loc[idx].values, r.pred.loc[idx].values)) | bd)
    out = pd.DataFrame(rows)
    out.to_csv(run / "conditions_tdd.csv", index=False)
    return out

if __name__ == "__main__":
    main()
