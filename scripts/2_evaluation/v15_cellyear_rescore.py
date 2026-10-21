"""v14 (2026-10-21): rescoring of the main comparisons on 1 km cell-year means (Supplementary Table S7).

Why: the label unit is the site (loc_id = coordinates rounded to 0.0001 deg); several sites can lie in one 1 km cell, so a site-year score weights
cells by their number of sites. Here the existing out-of-fold predictions are averaged per (1 km cell, year) together with the observations and
scored again. No model is refitted. Cell = the 1 km grid of the maps (CELL = 0.009 deg, same binning as fig6_product_maps.py / c13_maps.F6).

Inputs : data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1); data/processed/v5/runs/core_v6/preds/{Region}__{design}__year_obs__F8TS__*.csv.gz
         (block, random: per region; region hold-out: ALL file split by target region). Methods P0, P1, R1 (catboost, lambda 0.5), D0 (catboost).
Outputs: outputs/figures/paper/v15/source_data/TableS7_cellyear_rescore.csv (absolute RMSE), TableS7_cellyear_deltas.csv (paired differences + CIs),
         paper/manuscript/shared_v15/table_s7_cellyear_{en,ko}.tex (tabular bodies), paper/manuscript/shared_v15/numbers_v14_cellyear.tex (macros).
Run    : OMP_NUM_THREADS=2 nice -n 10 python3 scripts/2_evaluation/v14_cellyear_rescore.py
"""
from __future__ import annotations
import glob, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"; P = PROC / "v5/runs/core_v6/preds"
OUT_SRC = ROOT / "outputs/figures/paper/v15/source_data"; SH = ROOT / "paper/manuscript/shared_v15"
CELL = 0.009; B = 2000; SEED = 0
REGIONS = ("Alaska", "Lena", "Canada"); RK = {"Alaska": "Ak", "Lena": "Le", "Canada": "Ca"}
REGION_EN = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}; REGION_KO = {"Alaska": "알래스카", "Lena": "레나 델타", "Canada": "캐나다(북서부)"}
METHODS = (("P0", "P0__-__lam-1"), ("P1", "P1__-__lam-1"), ("R1", "R1__catboost__lam0.5"), ("D0", "D0__catboost__lam-1"))
MK = {"P0": "Pz", "P1": "Po", "R1": "Ro", "D0": "Dz"}
DESIGNS = (("block", "Spatial block", "공간 블록"), ("region", "Region hold-out", "지역 홀드아웃"), ("random", "Random", "무작위"))
PAIRS = {"block": (("R1", "P1"), ("D0", "P0")), "region": (("R1", "P0"), ("D0", "P0")), "random": (("D0", "P0"),)}
NAME_EN = {"P0": "Stefan (source coeff.)", "P1": "Stefan (recalibrated)", "R1": "Physics + residual ML", "D0": "Direct ML"}
NAME_KO = {"P0": "Stefan(원천 계수)", "P1": "Stefan(재보정)", "R1": "물리 잔차 ML", "D0": "직접 ML"}
rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))

t0 = time.time()
lab = pd.read_csv(PROC / "v5/labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1].copy()
ky = np.floor(lab.lat / CELL).astype(int); dx = CELL / np.cos(np.radians((ky + 0.5) * CELL)); kx = np.floor(lab.lon / dx).astype(int)
lab["cell"] = ky.astype(str) + "_" + kx.astype(str)
key = lab.set_index(["loc_id", "year"])[["cell", "block", "alt_cm", "region"]]

def load(region, design, pat):
    if design == "region":
        f = P / f"ALL__region__year_obs__F8TS__{pat}.csv.gz"
        d = pd.read_csv(f).groupby(["loc_id", "year"]).pred.mean().to_frame().join(key, how="inner"); return d[d.region == region]
    fs = sorted(glob.glob(str(P / f"{region}__{design}__year_obs__F8TS__{pat}.csv.gz")))
    assert fs, (region, design, pat)
    return pd.read_csv(fs[0]).groupby(["loc_id", "year"]).pred.mean().to_frame().join(key, how="inner")

def to_cells(d):
    return d.groupby(["cell", "year"]).agg(pred=("pred", "mean"), obs=("alt_cm", "mean"), block=("block", "first"))

def boot(j, pred_a, pred_b, obs):
    """paired block bootstrap of RMSE(a) - RMSE(b); j has a 'block' column"""
    blocks = j.block.unique(); rng = np.random.default_rng(SEED); g = {b: np.where(j.block.values == b)[0] for b in blocks}; out = np.empty(B)
    pa, pb, ob = j[pred_a].values, j[pred_b].values, j[obs].values
    for i in range(B):
        idx = np.concatenate([g[b] for b in rng.choice(blocks, len(blocks))]); out[i] = rmse(pa[idx] - ob[idx]) - rmse(pb[idx] - ob[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

abs_rows, d_rows, macros = [], [], []
within = {}
for region in REGIONS:
    g = lab[lab.region == region]; cy = g.groupby(["cell", "year"]).alt_cm
    w = g.alt_cm - cy.transform("mean"); within[region] = dict(pooled_within_cell_year_sd_cm=float(np.sqrt((w ** 2).sum() / max(1, len(g) - cy.ngroups))),
                                                             within_share_of_variance=float((w ** 2).sum() / ((g.alt_cm - g.alt_cm.mean()) ** 2).sum()),
                                                             sites_per_cell_median=float(g.groupby("cell").loc_id.nunique().median()),
                                                             sites_per_cell_max=int(g.groupby("cell").loc_id.nunique().max()),
                                                             n_site_years=int(len(g)), n_cells=int(g.cell.nunique()), n_cell_years=int(cy.ngroups))
    macros.append(r"\newcommand{\ncyWithinSd%s}{%.1f}" % (RK[region], within[region]["pooled_within_cell_year_sd_cm"]))
    macros.append(r"\newcommand{\ncySitesMed%s}{%.0f}" % (RK[region], within[region]["sites_per_cell_median"]))
    macros.append(r"\newcommand{\ncySitesMax%s}{%d}" % (RK[region], within[region]["sites_per_cell_max"]))
    for design, den, dko in DESIGNS:
        site, cells = {}, {}
        for m, pat in METHODS:
            d = load(region, design, pat); site[m] = d; cells[m] = to_cells(d)
            abs_rows.append(dict(region=region, design=design, method=m, n_site_years=len(d), n_cell_years=len(cells[m]),
                                 rmse_site_year_cm=round(rmse(d.pred - d.alt_cm), 2), rmse_cell_year_cm=round(rmse(cells[m].pred - cells[m].obs), 2)))
            macros.append(r"\newcommand{\ncy%s%s%s}{%.1f}" % (RK[region], {"block": "blk", "region": "reg", "random": "rnd"}[design], MK[m], rmse(cells[m].pred - cells[m].obs)))
        for a, b in PAIRS[design]:
            js = site[a][["pred", "alt_cm", "block"]].join(site[b][["pred"]], rsuffix="_b", how="inner")
            jc = cells[a][["pred", "obs", "block"]].join(cells[b][["pred"]], rsuffix="_b", how="inner")
            ds = rmse(js.pred - js.alt_cm) - rmse(js.pred_b - js.alt_cm); dc = rmse(jc.pred - jc.obs) - rmse(jc.pred_b - jc.obs)
            slo, shi = boot(js, "pred", "pred_b", "alt_cm"); clo, chi = boot(jc, "pred", "pred_b", "obs")
            d_rows.append(dict(region=region, design=design, pair=f"{a}-{b}", n_site_years=len(js), n_cell_years=len(jc),
                               delta_site_year_cm=ds, ci_site_lo=slo, ci_site_hi=shi, delta_cell_year_cm=dc, ci_cell_lo=clo, ci_cell_hi=chi,
                               resolved_site=bool(slo > 0 or shi < 0), resolved_cell=bool(clo > 0 or chi < 0), B=B))
            tag = "%s%s%s%s" % (RK[region], {"block": "blk", "region": "reg", "random": "rnd"}[design], MK[a], MK[b])
            tm = lambda v: f"{v:+.1f}".replace("-", "−")
            macros.append(r"\newcommand{\ncyd%s}{%s}" % (tag, tm(dc))); macros.append(r"\newcommand{\ncydci%s}{[%s, %s]}" % (tag, tm(clo), tm(chi)))
            macros.append(r"\newcommand{\nsyd%s}{%s}" % (tag, tm(ds))); macros.append(r"\newcommand{\nsydci%s}{[%s, %s]}" % (tag, tm(slo), tm(shi)))
        print(f"{region} {design}: {time.time() - t0:.0f} s")

A = pd.DataFrame(abs_rows); D = pd.DataFrame(d_rows)
OUT_SRC.mkdir(parents=True, exist_ok=True)
A.to_csv(OUT_SRC / "TableS7_cellyear_rescore.csv", index=False); D.round(3).to_csv(OUT_SRC / "TableS7_cellyear_deltas.csv", index=False)   # tables and macros format the raw values; the CSV keeps 3 decimals
(OUT_SRC / "TableS7_cellyear_meta.json").write_text(json.dumps(dict(script="scripts/2_evaluation/v14_cellyear_rescore.py", cell_deg=CELL, B=B, seed=SEED,
    definition="cell-year = mean of the observations and of the out-of-fold predictions over the label sites of one 1 km cell in one year; "
               "site-year values recomputed from the same prediction files (point estimates equal the manuscript values; intervals re-drawn)",
    within_cell=within), indent=1, ensure_ascii=False))

def fmt_d(r):  # "+1.2 [−0.3, +2.6]" with typographic minus
    f = lambda v: f"{v:+.1f}".replace("-", "−")
    return f"{f(r.delta_site_year_cm)} [{f(r.ci_site_lo)}, {f(r.ci_site_hi)}]", f"{f(r.delta_cell_year_cm)} [{f(r.ci_cell_lo)}, {f(r.ci_cell_hi)}]"

for lang, RN, NM, DN in (("en", REGION_EN, NAME_EN, {d: e for d, e, _ in DESIGNS}), ("ko", REGION_KO, NAME_KO, {d: k for d, _, k in DESIGNS})):
    L = []
    # (a) absolute RMSE: rows region x design, columns methods x unit
    L.append("% (a) RMSE (cm) on site-years and on cell-years; generated by scripts/2_evaluation/v14_cellyear_rescore.py")
    for region in REGIONS:
        for design, _, _ in DESIGNS:
            q = A[(A.region == region) & (A.design == design)].set_index("method")
            n = f"{q.n_site_years.iloc[0]:,} / {q.n_cell_years.iloc[0]:,}"
            cells_ = " & ".join(f"{q.loc[m, 'rmse_site_year_cm']:.1f} / {q.loc[m, 'rmse_cell_year_cm']:.1f}" for m, _ in METHODS)
            L.append(f"{RN[region]} & {DN[design]} & {n} & {cells_} \\\\")
    body_a = "\n".join(L)
    L = ["% (b) paired differences with 95% block-bootstrap intervals (B = 2000), site-years and cell-years"]
    for region in REGIONS:
        for design, _, _ in DESIGNS:
            for a, b in PAIRS[design]:
                r = D[(D.region == region) & (D.design == design) & (D.pair == f"{a}-{b}")].iloc[0]; s, c = fmt_d(r)
                comp = f"{NM[a]} − {NM[b]}"
                L.append(f"{RN[region]} & {DN[design]} & {comp} & {s} & {c} \\\\")
    body_b = "\n".join(L)
    HA = {"en": r"Region & Design & $n$ & Stefan (source) & Stefan (recalibrated) & Physics + residual ML & Direct ML \\",
          "ko": r"지역 & 설계 & $n$ & Stefan(원천 계수) & Stefan(재보정) & 물리 잔차 ML & 직접 ML \\"}[lang]
    HB = {"en": r"Region & Design & Comparison & Site-years: $\Delta$RMSE [95\% CI] & Cell-years: $\Delta$RMSE [95\% CI] \\",
          "ko": r"지역 & 설계 & 비교 & 지점-연도: $\Delta$RMSE [95\% 구간] & 셀-연도: $\Delta$RMSE [95\% 구간] \\"}[lang]
    tab_a = "\\begin{tabular}{@{}llcrrrr@{}}\n\\toprule\n" + HA + "\n\\midrule\n" + body_a + "\n\\bottomrule\n\\end{tabular}\n"
    tab_b = "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n" + HB + "\n\\midrule\n" + body_b + "\n\\bottomrule\n\\end{tabular}\n"
    (SH / f"table_s7a_cellyear_{lang}.tex").write_text(tab_a); (SH / f"table_s7b_cellyear_{lang}.tex").write_text(tab_b)
    (SH / f"table_s7_cellyear_{lang}.tex").unlink(missing_ok=True)
(SH / "numbers_v14_cellyear.tex").write_text("\n".join(macros) + "\n")
print(A.to_string(index=False)); print(D[["region", "design", "pair", "delta_site_year_cm", "ci_site_lo", "ci_site_hi", "delta_cell_year_cm", "ci_cell_lo", "ci_cell_hi", "resolved_site", "resolved_cell"]].to_string(index=False))
print(json.dumps(within, indent=0)); print(f"done {time.time() - t0:.0f} s")
