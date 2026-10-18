"""공통 셀-연도 비교(가장 직관적인 제품 비교). 지역마다 Wei·Liu·CCI v4 가 모두 값을 가진 셀-연도만 골라 우리 방법(블록 CV OOF)과 제품의 RMSE·편향·r 을
같은 셀에서 계산한다. 누설 마스크(제품 학습 지점 25 km 밖) 판도 낸다. 출력: data/processed/v5/runs/core_cb/common_cells_eval.csv
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

RUNS = ROOT / "data" / "processed" / "v5" / "runs"
_a = sys.argv[1:]
RUN = _a[_a.index("--run") + 1] if "--run" in _a else "core_cb"
FORC = _a[_a.index("--forcing") + 1] if "--forcing" in _a else "year"
ENS = _a[_a.index("--ens") + 1] if "--ens" in _a else "ens"
base = T.load_v5(ROOT / "data" / "processed" / "v5")
lk = pd.read_csv(ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv")
lk = lk[lk["product"] == "wei"].groupby("loc_id").dist_km.min()
base["dist_train_km"] = base.loc_id.map(lk)


def preds(run, m, lam, scheme="block"):
    fs = glob.glob(str(RUNS / run / "preds" / f"*__{scheme}__{FORC}__F8TS__{m}__*__lam{lam:g}.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in fs]).groupby("cell_year_id").pred.mean() if fs else None


ours = {"P1 (Stefan, recal.)": preds(RUN, "P1", -1), "R1 (CatBoost)": preds(RUN, "R1", 0.5), "D0 (CatBoost)": preds(RUN, "D0", -1),
        "R1 (5-learner ens.)": preds(ENS, "R1", 0.5), "D0 (5-learner ens.)": preds(ENS, "D0", -1)}
ours = {k: v for k, v in ours.items() if v is not None}
prods = {"Wei 2026": "prod_wei", "Liu 2024": "prod_liu", "ESA CCI v4": "prod_cci4y", "ESA CCI v5": "prod_cci5y", "Aalto 2018": "prod_aalto"}
rows = []
for r in T.REGIONS:
    s = base[base.region == r]
    for mask in (0, 25):
        q = s if mask == 0 else s[~(s.dist_train_km <= mask)]
        common = q[np.isfinite(q.prod_wei) & np.isfinite(q.prod_liu) & np.isfinite(q.prod_cci4y) & np.isfinite(q.alt_cm)]
        if len(common) < 30:
            continue
        for name, p in list(ours.items()):
            if p is None:
                continue
            v = common.cell_year_id.map(p).values
            m = np.isfinite(v)
            met = T.metrics(common.alt_cm.values[m], v[m], common.block.values[m])
            rows.append(dict(region=r, mask_km=mask, series=name, kind="ours", **met))
        for name, c in prods.items():
            v = common[c].values; m = np.isfinite(v)
            if m.sum() < 30:
                continue
            met = T.metrics(common.alt_cm.values[m], v[m], common.block.values[m])
            rows.append(dict(region=r, mask_km=mask, series=name, kind="product", **met))
out = pd.DataFrame(rows); out.to_csv(RUNS / RUN / "common_cells_eval.csv", index=False)
print(out.pivot_table(index=["region", "mask_km", "series"], values=["n", "rmse", "bias", "r"]).round(2).to_string())
