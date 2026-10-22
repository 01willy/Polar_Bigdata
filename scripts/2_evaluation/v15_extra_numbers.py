"""v15 extra manuscript macros (2026-10-21): Kudryavtsev and physics-ensemble anchors, lambda sensitivity, learner ranges, residuals by observation source,
TDD-anomaly agreement of the yearly maps. Output: paper/manuscript/shared_v15/numbers_v15_extra.tex (+ Supplementary Table S8 body table_s8_source_{en,ko}.tex)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; RUNS = ROOT / "data/processed/v5/runs"; SH = ROOT / "paper/manuscript/shared_v15"; SRC = ROOT / "outputs/figures/paper/v15/source_data"
RK = {"Alaska": "Ak", "Lena": "Le", "Canada": "Ca"}; out = []
def m(k, v): out.append(r"\newcommand{\n%s}{%s}" % (k, v))
f1 = lambda v: f"{v:.1f}"; sg = lambda v: f"{v:+.1f}".replace("-", "−")
sm = pd.read_csv(RUNS / "core_v6/summary_metrics.csv"); blk = sm[(sm.scheme == "block") & (sm.forcing == "year_obs") & (sm.featset == "F8TS")]
for r, k in RK.items():
    q = blk[blk.region == r]
    m(f"ku{k}blk", f1(q[q.method == "Ku"].rmse.iloc[0])); m(f"pe{k}blk", f1(q[q.method == "PE"].rmse.iloc[0]))
    for lam, tag in ((0.25, "Qtr"), (1.0, "One")):
        m(f"lam{tag}{k}blk", f1(q[(q.method == "R1") & np.isclose(q.lam, lam)].rmse.iloc[0]))
d = pd.read_csv(RUNS / "core_v6/deltas_vs_ref.csv"); dd = d[(d.ref == "P1") & (d.method == "R1") & (d.forcing == "year_obs") & (d.featset == "F8TS")]
for r, k in RK.items():
    for sch, st in (("block", "blk"), ("region", "reg")):
        for lam, tag in ((0.25, "Qtr"), (1.0, "One")):
            z = dd[(dd.region == r) & (dd.scheme == sch) & np.isclose(dd.lam, lam)].iloc[0]
            m(f"dlam{tag}{k}{st}", sg(z.delta)); m(f"dlam{tag}{k}{st}ci", f"[{sg(z.lo)}, {sg(z.hi)}]")
# learner ranges (block, year_obs): CatBoost (core), RF, MLP, TabM, FT-Transformer
rows = []
for tag in ("core_v6", "rf_v6", "nn_v6"):
    s = pd.read_csv(RUNS / f"{tag}/summary_metrics.csv"); s = s[(s.scheme == "block") & (s.forcing == "year_obs") & (s.featset == "F8TS") & (s.method.isin(["R1", "D0"]))]
    s = s[(s.method == "D0") | np.isclose(s.lam, 0.5)]; rows.append(s[["region", "method", "learner", "rmse"]])
L = pd.concat(rows)
for r, k in RK.items():
    for meth, mk in (("R1", "Ro"), ("D0", "Dz")):
        q = L[(L.region == r) & (L.method == meth)]; m(f"lrn{mk}{k}lo", f1(q.rmse.min())); m(f"lrn{mk}{k}hi", f1(q.rmse.max())); m(f"lrn{mk}{k}n", str(q.learner.nunique()))
        best = q.sort_values("rmse").iloc[0]; m(f"lrn{mk}{k}best", {"catboost": "CatBoost", "rf": "random forest", "mlp": "MLP", "tabm": "TabM", "ftt": "FT-Transformer"}[best.learner])
# residuals by observation source (block OOF, site-years)
P = RUNS / "core_v6/preds"; lab = pd.read_csv(ROOT / "data/processed/v5/labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
key = lab.set_index(["loc_id", "year"])[["alt_cm", "source", "method", "region"]]; SRCN = {"ABoVE_ds2369": "ABoVE", "CALM": "CALM", "ALLENA_PANGAEA_973813": "ALLena"}
PAT = {"P1": "P1__-__lam-1", "R1": "R1__catboost__lam0.5", "D0": "D0__catboost__lam-1"}; tab = []
for r, k in RK.items():
    preds = {mm: pd.read_csv(P / f"{r}__block__year_obs__F8TS__{pat}.csv.gz", usecols=["loc_id", "year", "pred", "seed"]).groupby(["loc_id", "year"]).pred.mean() for mm, pat in PAT.items()}
    base = preds["R1"].to_frame().join(key, how="inner")
    for src, g in base.groupby("source"):
        sk = SRCN[src]; idx = g.index; row = dict(region=r, source=sk, n=len(g), obs_mean=float(g.alt_cm.mean()), method=g.method.iloc[0])
        for mm in PAT:
            e = preds[mm].reindex(idx).values - g.alt_cm.values; row[f"rmse_{mm}"] = float(np.sqrt(np.mean(e ** 2))); row[f"bias_{mm}"] = float(np.mean(e))
        tab.append(row); kk = f"src{k}{sk}"
        m(f"{kk}N", f"{len(g):,}"); m(f"{kk}Obs", f1(row["obs_mean"])); m(f"{kk}Rmse", f1(row["rmse_R1"])); m(f"{kk}Bias", sg(row["bias_R1"])); m(f"{kk}RmseP", f1(row["rmse_P1"]))
T = pd.DataFrame(tab); T.to_csv(SRC / "TableS8_residuals_by_source.csv", index=False)
REN = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}; RKO = {"Alaska": "알래스카", "Lena": "레나 델타", "Canada": "캐나다(북서부)"}
MEN = {"probe": "probe transects", "probe_grid": "CALM grid means"}; MKO = {"probe": "탐침 측선", "probe_grid": "CALM 격자 평균"}
for lang, RN_, MN_ in (("en", REN, MEN), ("ko", RKO, MKO)):
    lines = []
    for _, z in T.iterrows():
        lines.append(f"{RN_[z.region]} & {z.source} & {MN_.get(z.method, z.method)} & {z.n:,} & {z.obs_mean:.1f} & {z.rmse_P1:.1f} / {sg(z.bias_P1)} & {z.rmse_R1:.1f} / {sg(z.bias_R1)} & {z.rmse_D0:.1f} / {sg(z.bias_D0)} \\\\")
    head = {"en": r"Region & Programme & Protocol & $n$ & Mean ALT (cm) & Stefan (recalibrated) & Physics + residual ML & Direct ML \\",
            "ko": r"지역 & 프로그램 & 규약 & $n$ & 평균 ALT (cm) & Stefan(재보정) & 물리 잔차 ML & 직접 ML \\"}[lang]
    (SH / f"table_s8_source_{lang}.tex").write_text("\\begin{tabular}{@{}lllrrlll@{}}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
# TDD-anomaly agreement of the yearly maps
# v15.1 (2026-10-22): read from the Fig 7 source data (panel d: regional TDD anomaly x map anomaly, computed over the mapped cells of each year)
f7d = json.loads((ROOT / "outputs/figures/paper/v15/source_data/Fig7_maps.json").read_text())["d"]["series"]
for r, k in RK.items():
    z = f7d[r]; yrs = sorted(z["tdd_anom"]); ta = np.array([z["tdd_anom"][y] for y in yrs]); ma = np.array([z["map_anom"][y] for y in yrs]); tdd = z["mean_tdd_degd"] + ta
    m(f"tddanom{k}", f"{np.corrcoef(ta, ma)[0, 1]:.2f}")
    m(f"tddmax{k}", yrs[int(np.argmax(ta))]); m(f"tddmin{k}", yrs[int(np.argmin(ta))])
    m(f"tddrange{k}", f"{int(round(tdd.min())):,}--{int(round(tdd.max())):,}")

# derived macros used by the abstract: |region hold-out delta of the physics-residual model| and the gain from 25 target labels (Lena Delta)
# (dref*regRoabs are defined in numbers_v6.tex)
lm = pd.read_csv(RUNS / "ladder_v6/ladder_metrics.csv"); lm["n_label"] = lm.n_label.astype(str)
for r, k in RK.items():
    q = lm[(lm.exp == "TL") & (lm.region == r) & (lm.method == "R1S")].set_index("n_label").rmse
    m(f"tlgain{k}Rstwfive", f1(float(q["0"] - q["25"])))

# largest in-region (spatial-block) advantage of direct ML over the physics-residual model across the three regions
m("drefRoDzblkmax", f1(max(float(blk[(blk.region == r) & (blk.method == "R1") & np.isclose(blk.lam, 0.5)].rmse.iloc[0] - blk[(blk.region == r) & (blk.method == "D0")].rmse.iloc[0]) for r in RK)))
(SH / "numbers_v15_extra.tex").write_text("\n".join(out) + "\n"); print("\n".join(out)); print(T.round(1).to_string())
