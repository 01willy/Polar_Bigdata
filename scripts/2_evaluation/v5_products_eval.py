"""v5 제품 비교(재설정 문서 Q4, 단계 F). 같은 셀·같은 해에서 기존 제품 값과 우리 방법의 OOF 예측을 비교한다.

- 우리 방법: <run>/preds 의 block 분할(V-B)·연도 정합 강제력·F8TS·씨앗 평균 예측(P0, P1, Ku, D0, R1 λ, S 등 지정).
- 제품: products_cell_year_v5.csv 의 연도 정합 값(prod_<p>), 다년 평균 값(prod_<p>_mean). 두 정의를 모두 채점한다.
- 공통 셀-연도: 지정 제품과 우리 예측이 모두 유한한 행(제품마다 다르므로 제품별로 공통 집합을 따로 잡고 n 을 적는다).
- 지표: RMSE·편향·r·블록 등가중 RMSE, 짝지은 ΔRMSE(우리 − 제품) 블록 부트스트랩 CI. 알래스카는 연도별 편향도 적는다.
출력: <run>/products_eval.csv, <run>/products_bias_by_year.csv
실행: python3 scripts/2_evaluation/v5_products_eval.py --run data/processed/v5/runs/core_cb [--ours P1,R1:0.5,D0]
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

PRODUCTS = ["cci4y", "cci5y", "wei", "liu", "yk", "aalto"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scheme", default="block")
    ap.add_argument("--forcing", default="year")
    ap.add_argument("--featset", default="F8TS")
    ap.add_argument("--ours", default="P0,P1,Ku,D0,R1:0.5")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--mask-km", type=float, default=0.0, help="제품 학습 지점(CALM·GTN-P 대리, XG xg_leak_cells wei 거리)에서 이 거리 안의 셀을 뺀다")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    run = ROOT / a.run
    base = T.load_v5(ROOT / "data" / "processed" / "v5")
    pcols = [c for c in base.columns if c.startswith("prod_")]
    base = base[["cell_year_id", "region", "loc_id", "year", "block", "alt_cm"] + pcols]
    n_before = len(base)
    if a.mask_km > 0:
        lk = pd.read_csv(ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv")
        lk = lk[lk["product"] == "wei"].groupby("loc_id").dist_km.min()
        d = base.loc_id.map(lk)
        base = base[~(d <= a.mask_km)].reset_index(drop=True)          # 거리 정보가 없는 셀은 남긴다
        print(f"[mask] {a.mask_km} km: {n_before:,} → {len(base):,} 셀-연도", flush=True)
    ours = {}
    for spec in a.ours.split(","):
        m, lam = (spec.split(":") + ["-1"])[:2]
        fs = glob.glob(str(run / "preds" / f"*__{a.scheme}__{a.forcing}__{a.featset}__{m}__*__lam{float(lam):g}.csv.gz"))
        if not fs:
            print(f"[skip] {spec}: 예측 없음"); continue
        p = pd.concat([pd.read_csv(f) for f in fs]).groupby("cell_year_id").pred.mean()
        ours[spec] = base.cell_year_id.map(p).values
    rows, byyear = [], []
    for r in T.REGIONS:
        s = base[base.region == r]
        for spec, pv in ours.items():                                   # 우리 방법, 지역의 모든 셀-연도
            mm = np.isfinite(pv[s.index]) & np.isfinite(s.alt_cm.values)
            if mm.sum() >= 30:
                rows.append(dict(region=r, series=f"ours:{spec}", definition="all cells", **T.metrics(s.alt_cm.values[mm], pv[s.index][mm], s.block.values[mm])))
        for p in PRODUCTS:
            for defn, col in (("year-matched", f"prod_{p}"), ("period mean", f"prod_{p}_mean" if f"prod_{p}_mean" in s else f"prod_{p}")):
                if col not in s:
                    continue
                v = s[col].values.astype(float)
                m = np.isfinite(v) & np.isfinite(s.alt_cm.values)
                if m.sum() < 30:
                    continue
                met = T.metrics(s.alt_cm.values[m], v[m], s.block.values[m])
                rows.append(dict(region=r, series=p, definition=defn, **met))
                for spec, pv in ours.items():
                    mm = m & np.isfinite(pv[s.index])
                    if mm.sum() < 30:
                        continue
                    mo = T.metrics(s.alt_cm.values[mm], pv[s.index][mm], s.block.values[mm])
                    bd = T.boot_delta(s.alt_cm.values[mm], pv[s.index][mm], v[mm], s.block.values[mm], a.nboot)
                    rows.append(dict(region=r, series=f"ours:{spec}", definition=f"vs {p} ({defn})", **mo, delta_vs_product=bd["delta"],
                                     delta_lo=bd["lo"], delta_hi=bd["hi"], delta_beq=bd["delta_beq"], delta_beq_lo=bd["lo_beq"], delta_beq_hi=bd["hi_beq"]))
                if defn == "year-matched":
                    for y, sy in s[m].groupby("year"):
                        if len(sy) >= 20:
                            byyear.append(dict(region=r, series=p, year=int(y), n=len(sy), bias=float((sy[col] - sy.alt_cm).mean()),
                                               rmse=T.rmse(sy.alt_cm.values, sy[col].values)))
        for spec, pv in ours.items():
            for y, sy in s.groupby("year"):
                mm = np.isfinite(pv[sy.index])
                if mm.sum() >= 20:
                    byyear.append(dict(region=r, series=f"ours:{spec}", year=int(y), n=int(mm.sum()), bias=float((pv[sy.index][mm] - sy.alt_cm.values[mm]).mean()),
                                       rmse=T.rmse(sy.alt_cm.values[mm], pv[sy.index][mm])))
    out = pd.DataFrame(rows); out["mask_km"] = a.mask_km; out.to_csv(run / f"products_eval{a.suffix}.csv", index=False)
    pd.DataFrame(byyear).to_csv(run / f"products_bias_by_year{a.suffix}.csv", index=False)
    show = out[["region", "series", "definition", "n", "rmse", "bias", "r", "rmse_beq"] + [c for c in ("delta_vs_product", "delta_lo", "delta_hi") if c in out]]
    print(show.round(2).to_string(index=False)[:6000])


if __name__ == "__main__":
    main()
