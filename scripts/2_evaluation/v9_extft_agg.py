# -*- coding: utf-8 -*-
"""v9_extft_agg.py: 외부 지역 소수 라벨 적응(EXTFT) 집계. 씨앗 평균 예측, 소지역별·전체 RMSE, 제품과 같은 행 비교, 블록 부트스트랩 ΔRMSE 95% 구간.
입력 <run>/preds_EXTFT_*.csv.gz. 출력 <run>/extft_metrics.csv(scope, n_label, series, rows, n, rmse), <run>/extft_deltas.csv.
실행: python3 scripts/2_evaluation/v9_extft_agg.py --run data/processed/v5/runs/extra_v9 [--nboot 2000]
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--nboot", type=int, default=2000); a = ap.parse_args()
R = Path(a.run)
p = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "preds_EXTFT_*.csv.gz")))], ignore_index=True)
p["n_label"] = p.n_label.astype(str)
pcols = [c for c in p.columns if c.startswith("prod_")]
first = p.drop_duplicates("cell_year_id").set_index("cell_year_id")[["y", "block", "subregion"] + pcols]
NS = ["0", "5", "10", "25", "50", "all"]
met, dl = [], []
for nl in NS:
    w = p[p.n_label == nl].groupby(["cell_year_id", "method"]).pred.mean().unstack().join(first)
    for scope, q in [("all", w)] + [(r, g) for r, g in w.groupby("subregion")]:
        for m in ("P0", "P1n", "D0S", "R1S"):
            met.append(dict(scope=scope, n_label=nl, series=m, rows="all", n=int(len(q)), rmse=T.rmse(q.y.values, q[m].values)))
        for prod, col in (("Wei 2026", "prod_wei"), ("Liu 2024", "prod_liu"), ("ESA CCI v5", "prod_cci5y")):
            if col not in q:
                continue
            k = q[q[col].notna()]
            if len(k) < 10:
                continue
            met.append(dict(scope=scope, n_label=nl, series=prod, rows=prod, n=int(len(k)), rmse=T.rmse(k.y.values, k[col].values)))
            for m in ("P1n", "R1S", "D0S"):
                met.append(dict(scope=scope, n_label=nl, series=m, rows=prod, n=int(len(k)), rmse=T.rmse(k.y.values, k[m].values)))
            if scope == "all":
                for m in ("P1n", "R1S", "D0S"):
                    r_ = T.boot_delta(k.y.values, k[m].values, k[col].values, k.block.values, nboot=a.nboot)
                    dl.append(dict(n_label=nl, A=m, B=prod, delta=r_["delta"], lo=r_["lo"], hi=r_["hi"], n=r_["n"], n_blocks=r_["n_blocks"]))
        if scope == "all":
            for A, B in (("R1S", "P1n"), ("R1S", "D0S"), ("P1n", "P0"), ("R1S", "P0")):
                r_ = T.boot_delta(q.y.values, q[A].values, q[B].values, q.block.values, nboot=a.nboot)
                dl.append(dict(n_label=nl, A=A, B=B, delta=r_["delta"], lo=r_["lo"], hi=r_["hi"], n=r_["n"], n_blocks=r_["n_blocks"]))
m_ = pd.DataFrame(met); d_ = pd.DataFrame(dl)
m_.to_csv(R / "extft_metrics.csv", index=False); d_.to_csv(R / "extft_deltas.csv", index=False)
print(m_[(m_.scope == "all") & (m_.rows == "all")].pivot_table(index="series", columns="n_label", values="rmse").reindex(columns=NS).round(1).to_string())
print(m_[(m_.scope == "all") & (m_.rows != "all")].pivot_table(index=["rows", "series"], columns="n_label", values="rmse").reindex(columns=NS).round(1).to_string())
print(m_[(m_.rows == "all") & (m_.series == "R1S")].pivot_table(index="scope", columns="n_label", values="rmse").reindex(columns=NS).round(1).to_string())
d_["txt"] = d_.apply(lambda r: f"{r.delta:+.1f} [{r.lo:+.1f},{r.hi:+.1f}]", axis=1)
print(d_.pivot_table(index=["A", "B"], columns="n_label", values="txt", aggfunc="first").reindex(columns=NS).to_string())
