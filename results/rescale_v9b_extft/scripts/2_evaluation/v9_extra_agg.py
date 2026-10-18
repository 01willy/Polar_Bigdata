# -*- coding: utf-8 -*-
"""v9_extra_agg.py: v9 추가 실험(PA·EXT·SHAP) 집계. 씨앗 평균 예측, RMSE·편향, 블록 부트스트랩 짝지은 ΔRMSE 95% 구간(2000회).
입력 <run>/preds_PA_<region>_<product>.csv.gz, preds_EXT.csv.gz, shap_summary.csv.
출력 <run>/pa_metrics.csv, pa_deltas.csv, ext_metrics.csv, ext_deltas.csv, shap_groups.csv
실행: python3 scripts/2_evaluation/v9_extra_agg.py --run data/processed/v5/runs/extra_v9
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402
from polar.fidelity import TERRAIN, SOIL  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--nboot", type=int, default=2000); a = ap.parse_args()
R = Path(a.run)


def wide(p, extra=()):
    g = p.groupby(["cell_year_id", "method"]).agg(pred=("pred", "mean")).reset_index().pivot(index="cell_year_id", columns="method", values="pred")
    first = p.drop_duplicates("cell_year_id").set_index("cell_year_id")[["y", "block", "loc_id", "year", "region"] + list(extra)]
    return g.join(first)


def met(y, p, b):
    m = T.metrics(y, p, b); return dict(n=m["n"], rmse=m["rmse"], bias=m["bias"], r=m["r"])


def delta(w, A, B):
    ok = w[A].notna() & w[B].notna()
    r = T.boot_delta(w.y[ok].values, w[A][ok].values, w[B][ok].values, w.block[ok].values, nboot=a.nboot)
    return dict(A=A, B=B, delta=r["delta"], lo=r["lo"], hi=r["hi"], n=r["n"], n_blocks=r["n_blocks"])


# ---------------------------------------------------------------- PA
pm, pdl = [], []
for f in sorted(glob.glob(str(R / "preds_PA_*.csv.gz"))):
    p = pd.read_csv(f); region = p.region.iloc[0]; prod = p["product"].iloc[0]; w = wide(p)
    for m in [c for c in ("Praw", "Pcal", "P1", "R1", "RP", "RP1", "R1x") if c in w]:
        pm.append(dict(region=region, product=prod, method=m, **met(w.y.values, w[m].values, w.block.values)))
    for A, B in (("R1", "P1"), ("RP", "Pcal"), ("R1", "RP"), ("R1", "Pcal"), ("Pcal", "Praw"), ("R1", "Praw"), ("P1", "Praw"), ("R1x", "R1"), ("RP1", "RP")):
        if A in w and B in w:
            pdl.append(dict(region=region, product=prod, **delta(w, A, B)))
    print(f"[PA] {region} {prod} rows {len(w):,}", flush=True)
if pm:
    pd.DataFrame(pm).to_csv(R / "pa_metrics.csv", index=False); pd.DataFrame(pdl).to_csv(R / "pa_deltas.csv", index=False)
    d = pd.DataFrame(pdl); d["txt"] = d.apply(lambda r: f"{r.delta:+.1f} [{r.lo:+.1f},{r.hi:+.1f}]", axis=1)
    print(d.pivot_table(index=["region", "product"], columns=["A", "B"], values="txt", aggfunc="first").to_string())

# ---------------------------------------------------------------- EXT
fe = R / "preds_EXT.csv.gz"
if fe.exists():
    p = pd.read_csv(fe); pcols = [c for c in p.columns if c.startswith("prod_")]
    w = wide(p, pcols)
    cell_mean = w.groupby("loc_id").y.transform("mean"); w["in_range"] = cell_mean <= 300.0      # 학습 라벨 범위(0–300 cm) 안 셀
    ours = [m for m in ("P0", "P1loc", "D0", "F1", "R1", "R1loc") if m in w]
    em, edl = [], []
    for sub_name, sub in [("all", w), ("in_range", w[w.in_range])] + [(f"in_range|{r}", s) for r, s in w[w.in_range].groupby("region")]:
        for s_name, col in [(m, m) for m in ours] + [(c.replace("prod_", ""), c) for c in pcols]:
            em.append(dict(subset=sub_name, series=s_name, rows="own", **met(sub.y.values, sub[col].values, sub.block.values)))
        wei_ok = sub[sub["prod_wei"].notna()] if "prod_wei" in sub else sub.iloc[0:0]
        for s_name, col in [(m, m) for m in ours] + [("wei", "prod_wei"), ("liu", "prod_liu")]:
            q = wei_ok if s_name != "liu" else wei_ok[wei_ok["prod_liu"].notna()]
            if len(q) >= 10:
                em.append(dict(subset=sub_name, series=s_name, rows="wei_common" if s_name != "liu" else "wei_liu_common", **met(q.y.values, q[col].values, q.block.values)))
        if sub_name in ("all", "in_range"):
            ww = sub.rename(columns={"prod_wei": "Wei", "prod_liu": "Liu"})
            for A, B in (("R1", "P0"), ("D0", "P0"), ("F1", "P0"), ("R1loc", "P1loc"), ("P1loc", "P0"), ("R1loc", "R1"), ("R1", "Wei"), ("R1loc", "Wei"), ("P0", "Wei"),
                         ("R1", "Liu"), ("R1loc", "Liu")):
                if A in ww and B in ww and (ww[A].notna() & ww[B].notna()).sum() >= 10:
                    edl.append(dict(subset=sub_name, **delta(ww, A, B)))
    pd.DataFrame(em).to_csv(R / "ext_metrics.csv", index=False); pd.DataFrame(edl).to_csv(R / "ext_deltas.csv", index=False)
    e = pd.DataFrame(em); print(e[e.rows == "own"].pivot_table(index="subset", columns="series", values="rmse").round(1).to_string())
    print(e[e.rows != "own"].pivot_table(index=["subset", "rows"], columns="series", values="rmse").round(1).to_string())
    d = pd.DataFrame(edl); d["txt"] = d.apply(lambda r: f"{r.delta:+.1f} [{r.lo:+.1f},{r.hi:+.1f}] n{r.n}", axis=1); print(d[["subset", "A", "B", "txt"]].to_string())

# ---------------------------------------------------------------- SHAP
fs = R / "shap_summary.csv"
if fs.exists():
    s = pd.read_csv(fs); grp = {f: ("climate" if f.startswith("e5_") else "terrain" if f in TERRAIN else "soil" if f in SOIL else "other") for f in s.feature.unique()}
    s["group"] = s.feature.map(grp)
    g = s.groupby(["region", "kind", "group"]).mean_abs.sum().reset_index(); g["share"] = g.mean_abs / g.groupby(["region", "kind"]).mean_abs.transform("sum")
    g.to_csv(R / "shap_groups.csv", index=False); print(g.pivot_table(index=["region", "kind"], columns="group", values="share").round(2).to_string())
    top = s.sort_values("mean_abs", ascending=False).groupby(["region", "kind"]).head(5); print(top[["region", "kind", "feature", "mean_abs"]].to_string(index=False))
