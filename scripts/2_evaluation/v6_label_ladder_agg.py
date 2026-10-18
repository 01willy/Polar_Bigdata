# -*- coding: utf-8 -*-
"""v6_label_ladder_agg.py: 라벨 수 사다리(LL·TL) 집계. 씨앗 평균 예측으로 풀링 RMSE, 기준 대비 짝지은 ΔRMSE 와 블록 부트스트랩 95% 구간.
입력: <run>/preds_{LL,TL}_<Region>.csv.gz. 출력: <run>/ladder_metrics.csv(exp, region, n_label, method, rmse, rmse_sd_seed, n),
      <run>/ladder_deltas.csv(exp, region, n_label, method, ref, delta, lo, hi). LL 기준 P1(같은 n), TL 기준 P1n(같은 n)과 P0.
실행: python3 scripts/2_evaluation/v6_label_ladder_agg.py --run data/processed/v5/runs/ladder_v6 [--nboot 1000]
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--nboot", type=int, default=1000); a = ap.parse_args()
R = Path(a.run); rng = np.random.RandomState(0)


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def boot_delta(e1, e2, blocks, nboot):
    """짝지은 ΔRMSE = rmse(e1) − rmse(e2); 블록 재추출."""
    d0 = rmse(e1) - rmse(e2); ub = np.unique(blocks); idx = {b: np.where(blocks == b)[0] for b in ub}
    out = np.empty(nboot)
    for i in range(nboot):
        bs = rng.choice(ub, len(ub), replace=True); j = np.concatenate([idx[b] for b in bs])
        out[i] = rmse(e1[j]) - rmse(e2[j])
    return d0, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


met, dl = [], []
for f in sorted(glob.glob(str(R / "preds_*_*.csv.gz"))):
    p = pd.read_csv(f); exp = p.exp.iloc[0]; region = p.region.iloc[0]
    p["n_label"] = p.n_label.astype(str)
    for (nl, m), g in p.groupby(["n_label", "method"]):
        per_seed = g.groupby("seed").apply(lambda q: rmse(q.pred.values - q.y.values))
        mean = g.groupby("cell_year_id").agg(y=("y", "first"), pred=("pred", "mean"), block=("block", "first"))
        met.append(dict(exp=exp, region=region, n_label=nl, n=int(g.n.iloc[0]), method=m, rmse=rmse(mean.pred.values - mean.y.values), rmse_seed_mean=float(per_seed.mean()), rmse_seed_sd=float(per_seed.std()), n_test=int(len(mean))))
    refs = {"LL": ["P1", "P0"], "TL": ["P1n", "P0"]}[exp]
    for nl, g in p.groupby("n_label"):
        wide = g.groupby(["cell_year_id", "method"]).agg(y=("y", "first"), pred=("pred", "mean"), block=("block", "first")).reset_index()
        pv = wide.pivot(index="cell_year_id", columns="method", values="pred"); yy = wide.groupby("cell_year_id").y.first().reindex(pv.index); bl = wide.groupby("cell_year_id").block.first().reindex(pv.index).values
        for ref in refs:
            if ref not in pv:
                continue
            for m in pv.columns:
                if m == ref:
                    continue
                ok = pv[m].notna() & pv[ref].notna()
                d0, lo, hi = boot_delta((pv[m] - yy).values[ok], (pv[ref] - yy).values[ok], bl[ok], a.nboot)
                dl.append(dict(exp=exp, region=region, n_label=nl, n=int(g.n.iloc[0]), method=m, ref=ref, delta=d0, lo=lo, hi=hi, n_test=int(ok.sum())))
    print(f"[agg] {exp} {region} rows {len(p):,}", flush=True)
pd.DataFrame(met).to_csv(R / "ladder_metrics.csv", index=False); pd.DataFrame(dl).to_csv(R / "ladder_deltas.csv", index=False)
print(pd.DataFrame(dl).query("ref in ('P1','P1n') and method in ('R1','D1','D0','R1S','D0S','D1S')").assign(txt=lambda t: t.apply(lambda r: f"{r.delta:+.1f} [{r.lo:+.1f},{r.hi:+.1f}]", axis=1)).pivot_table(index=["exp", "region", "method"], columns="n_label", values="txt", aggfunc="first").to_string())
