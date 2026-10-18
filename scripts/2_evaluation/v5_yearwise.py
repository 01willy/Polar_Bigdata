"""v5 연도별 평가(사용자 질문 3). 블록 CV OOF·연도 홀드아웃 예측과 제품 값을 연도별·다년 셀별로 채점하고 편차의 원인을 본다.

출력(<run>/yearwise/):
  yearwise_metrics.csv     지역 × 연도 × 계열(ours:P1·D0·R1 블록 OOF, ours LOYO, 제품) RMSE·편향·n
  multiyear_changes.csv    2개 연도 이상 관측된 셀에서 연속 관측 연도 쌍의 Δobs 대 Δpred(계열별): r, 부호 일치율, RMSE(Δ), n
  bias_vs_tdd.csv          지역·계열별 연도 편향 대 그 해 셀 평균 TDD 편차(연도 TDD − 2011–2024 기후값)의 기울기·r
  outlier_years.csv        편향 |bias| > 8 cm 인 지역·연도의 구성(셀 수, 출처, 관측 월, 2014년 셀과의 거리, TDD 편차)
실행: python3 scripts/2_evaluation/v5_yearwise.py --run data/processed/v5/runs/core_cb
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

PRODS = ["cci4y", "cci5y", "wei", "liu", "yk", "aalto"]


def load_series(run, scheme, forcing, featset, spec):
    m, lam = (spec.split(":") + ["-1"])[:2]
    fs = glob.glob(str(run / "preds" / f"*__{scheme}__{forcing}__{featset}__{m}__*__lam{float(lam):g}.csv.gz"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs]).groupby("cell_year_id").pred.mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--ours", default="P1,D0,R1:0.5"); ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--forcing", default="year"); ap.add_argument("--featset", default="F8TS"); ap.add_argument("--no-products", action="store_true")
    a = ap.parse_args(); run = ROOT / a.run; out = run / f"yearwise_{a.forcing}_{a.featset}" if (a.forcing != "year" or a.featset != "F8TS") else run / "yearwise"; out.mkdir(exist_ok=True)
    base = T.load_v5(ROOT / "data" / "processed" / "v5")
    base["tdd_anom"] = base.y_e5_tdd - base.c_e5_tdd
    series = {}
    for spec in a.ours.split(","):
        for scheme, tag in (("block", "blockCV"), ("year", "LOYO")):
            p = load_series(run, scheme, a.forcing, a.featset, spec)
            if p is not None:
                series[f"ours:{spec}:{tag}"] = base.cell_year_id.map(p).values
    for p in ([] if a.no_products else PRODS):
        if f"prod_{p}" in base:
            series[f"prod:{p}"] = base[f"prod_{p}"].values
    y = base.alt_cm.values
    # 1. 연도별 지표
    rows = []
    for r in T.REGIONS:
        for yr, idx in base[base.region == r].groupby("year").groups.items():
            idx = np.asarray(idx)
            for name, v in series.items():
                m = np.isfinite(v[idx]) & np.isfinite(y[idx])
                if m.sum() < a.min_n:
                    continue
                e = v[idx][m] - y[idx][m]
                rows.append(dict(region=r, year=int(yr), series=name, n=int(m.sum()), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()),
                                 obs_mean=float(y[idx][m].mean()), tdd_anom_mean=float(np.nanmean(base.tdd_anom.values[idx][m])),
                                 obs_month_mode=int(base.obs_month_last.values[idx][m].astype(int).min())))
    ym = pd.DataFrame(rows); ym.to_csv(out / "yearwise_metrics.csv", index=False)
    # 2. 다년 셀의 연도 간 변화
    ch = []
    for r in T.REGIONS:
        s = base[base.region == r]
        multi = s.groupby("loc_id").year.nunique(); multi = multi[multi >= 2].index
        s = s[s.loc_id.isin(multi)].sort_values(["loc_id", "year"])
        for name, v in series.items():
            vv = pd.Series(v, index=base.index).loc[s.index].values
            d_obs, d_pred, d_tdd = [], [], []
            for loc, g in s.assign(pred=vv).groupby("loc_id"):
                g = g.dropna(subset=["pred", "alt_cm"])
                if len(g) < 2:
                    continue
                d_obs.extend(np.diff(g.alt_cm.values)); d_pred.extend(np.diff(g.pred.values)); d_tdd.extend(np.diff(g.y_e5_tdd.values))
            d_obs, d_pred, d_tdd = np.array(d_obs), np.array(d_pred), np.array(d_tdd)
            if len(d_obs) < 10:
                continue
            ch.append(dict(region=r, series=name, n_pairs=int(len(d_obs)), r_delta=float(np.corrcoef(d_obs, d_pred)[0, 1]) if d_pred.std() > 0 else np.nan,
                           sign_agree=float(np.mean(np.sign(d_obs) == np.sign(d_pred))), rmse_delta=float(np.sqrt(np.mean((d_pred - d_obs) ** 2))),
                           sd_delta_obs=float(d_obs.std()), sd_delta_pred=float(d_pred.std()), r_dobs_dtdd=float(np.corrcoef(d_obs, d_tdd)[0, 1]),
                           r_dpred_dtdd=float(np.corrcoef(d_pred, d_tdd)[0, 1]) if d_pred.std() > 0 else np.nan))
    pd.DataFrame(ch).to_csv(out / "multiyear_changes.csv", index=False)
    # 3. 연도 편향 대 TDD 편차
    bt = []
    for (r, name), g in ym.groupby(["region", "series"]):
        g = g[g.n >= 50]
        if len(g) >= 4 and g.tdd_anom_mean.std() > 0:
            b, a0 = np.polyfit(g.tdd_anom_mean, g.bias, 1)
            bt.append(dict(region=r, series=name, n_years=len(g), slope_cm_per_100tdd=float(b * 100), r=float(np.corrcoef(g.tdd_anom_mean, g.bias)[0, 1]),
                           bias_sd_across_years=float(g.bias.std())))
    pd.DataFrame(bt).to_csv(out / "bias_vs_tdd.csv", index=False)
    # 4. 이상 연도 구성
    oy = []
    ref = base[(base.region == "Alaska") & (base.year == 2014)][["lat", "lon"]].values
    for _, row in ym[(ym.series == "ours:R1:0.5:blockCV") & (ym.bias.abs() > 8)].iterrows():
        s = base[(base.region == row.region) & (base.year == row.year)]
        d = np.nan
        if row.region == "Alaska" and len(ref):
            from sklearn.neighbors import BallTree
            t = BallTree(np.radians(ref), metric="haversine")
            d = float(np.median(t.query(np.radians(s[["lat", "lon"]].values), k=1)[0][:, 0] * 6371))
        oy.append(dict(region=row.region, year=int(row.year), n=len(s), bias_R1=row.bias, sources=s.source.value_counts().to_dict(),
                       obs_month=s.obs_month_last.value_counts().to_dict(), median_km_to_2014_cells=d, tdd_anom_mean=float(s.tdd_anom.mean()),
                       obs_mean=float(s.alt_cm.mean()), n_blocks=int(s.block.nunique())))
    pd.DataFrame(oy).to_csv(out / "outlier_years.csv", index=False)
    show = ym[ym.series.isin(["ours:P1:blockCV", "ours:R1:0.5:blockCV", "ours:R1:0.5:LOYO", "prod:wei", "prod:liu"])]
    print(show.pivot_table(index=["region", "year"], columns="series", values="bias").round(1).to_string())
    print(pd.DataFrame(ch)[["region", "series", "n_pairs", "r_delta", "sign_agree", "r_dobs_dtdd", "r_dpred_dtdd"]].round(2).to_string(index=False))
    print(pd.DataFrame(bt).round(2).to_string(index=False))
    if oy:
        print(pd.DataFrame(oy).drop(columns=["sources", "obs_month"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
