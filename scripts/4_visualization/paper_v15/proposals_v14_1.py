"""Figure design proposals, round 2 (2026-10-21). Real data, proposal-grade QA (text checks printed, not enforced).
Outputs: outputs/figures/paper/v15_proposals/P*_*.{pdf,svg,png}. Run from this folder with POLAR_PALETTE=paper13.
Designs: P3_D validation-strictness slope chart; P3_E observed-vs-predicted density under spatial-block CV; P3_F three-question layout
(when / with what / where); P4_E label-budget dot plot; P4_F transfer-gain heat table; P5_D observed-vs-predicted matrix on the common site-years;
P5_F error distributions per series; P7_D anomaly agreement scatter."""
from __future__ import annotations
import glob, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = C.ROOT; RUNS = ROOT / "data" / "processed" / "v5" / "runs"; P = RUNS / "core_v6" / "preds"
OUT = ROOT / "outputs" / "figures" / "paper" / "v15_proposals"; OUT.mkdir(parents=True, exist_ok=True)
REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}; RC = {r: C.REGION[r][1] for r in REGIONS}
SCHEMES = ["random", "block", "year", "block_year", "region"]; SCHEME_LB = ["Random", "Spatial block", "Year", "Block × year", "Region"]
PAT = {"P0": "P0__-__lam-1", "P1": "P1__-__lam-1", "R1": "R1__catboost__lam0.5", "D0": "D0__catboost__lam-1"}
MN = C.METHOD_NAME; MC = C.METHOD
NEG, POS = "#1A7AB3", "#B03A2E"
CM_DIV = LinearSegmentedColormap.from_list("div", [NEG, "#FFFFFF", POS])
_lab = None


def labels():
    global _lab
    if _lab is None:
        l = pd.read_csv(ROOT / "data/processed/v5/labels_cell_year_v5.csv"); l = l[l.in_forcing == 1]
        _lab = l.set_index(["loc_id", "year"])[["alt_cm", "region", "block"]]
    return _lab


def oof(region, scheme, m):
    """seed-mean out-of-fold prediction joined with the observation"""
    if scheme == "region":
        d = pd.read_csv(P / f"ALL__region__year_obs__F8TS__{PAT[m]}.csv.gz", usecols=["loc_id", "year", "pred", "seed", "region"])
        d = d[d.region == region]
    else:
        d = pd.read_csv(P / f"{region}__{scheme}__year_obs__F8TS__{PAT[m]}.csv.gz", usecols=["loc_id", "year", "pred", "seed"])
    d = d.groupby(["loc_id", "year"]).pred.mean().to_frame().join(labels(), how="inner")
    return d


rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])


def ramp(hex_):
    return LinearSegmentedColormap.from_list("r_" + hex_, ["#FFFFFF", hex_])


def density(ax, x, y, col, top, n_hex=2000):
    if len(x) >= n_hex:
        ax.hexbin(x, y, gridsize=30, extent=(0, top, 0, top), cmap=ramp(col), norm=LogNorm(vmin=1), mincnt=1, linewidths=0.1, edgecolors="none")
    else:
        ax.scatter(x, y, s=2.2 ** 2, facecolor=col, edgecolor="white", lw=0.2, alpha=0.85)
    ax.plot([0, top], [0, top], color=C.ZERO, lw=C.LW_REF, zorder=1)
    ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal")
    step = 40 if top > 120 else 20
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step)); ax.yaxis.set_major_locator(mticker.MultipleLocator(step))


def legend_row(fig, y_top, items, x0=12.0):
    hs = [Line2D([], [], color=c, lw=C.LW_MAIN, ls=ls, marker=mk, ms=C.MS, mfc=(c if filled else "white"), mec=c, mew=0.7) for _, c, mk, ls, filled in items]
    fig.legend(hs, [n for n, *_ in items], loc="upper left", bbox_to_anchor=(x0 / fig.get_figwidth() / 25.4, 1 - y_top / fig.get_figheight() / 25.4),
               ncol=len(items), fontsize=C.FS, frameon=False, handlelength=2.4, columnspacing=1.6, handletextpad=0.6, borderaxespad=0.0)


def finish(fig, stem, meta):
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(stem, rep)
    paths = C.export(fig, stem, OUT); plt.close(fig)
    meta.update(dict(outputs=C.file_info(paths), qa=rep)); C.write_json(OUT / f"{stem}.json", meta)
    for p in paths: print("  ", p.name, p.stat().st_size)


# ================================================================ P3_D slope chart: RMSE across validation designs ordered by strictness
def p3_d():
    sm = pd.read_csv(RUNS / "core_v6/summary_metrics.csv"); sm = sm[(sm.forcing == "year_obs") & (sm.featset == "F8TS")]

    def rm(region, scheme, m):
        q = sm[(sm.region == region) & (sm.scheme == scheme) & (sm.method == m)]
        if m == "R1": q = q[(q.learner == "catboost") & (np.isclose(q.lam, 0.5))]
        if m == "D0": q = q[q.learner == "catboost"]
        return float(q.rmse.iloc[0])
    W, H = 180.0, 66.0; fig = C.figure(W, H)
    items = [(MN["P0"], MC["P0"], "s", "-", True), (MN["P1"], MC["P1"], "s", "-", True), (MN["R1"], MC["R1"], "o", "-", True), (MN["D0"], MC["D0"], "o", C.DASH, False)]
    legend_row(fig, 1.0, items)
    vals = {}
    for i, r in enumerate(REGIONS):
        ax = C.ax_mm(fig, 14 + i * 57.5, 10.5, 46, 40); C.clean_axes(ax)
        ys = {m: [rm(r, s, m) for s in SCHEMES] for m in ("P0", "P1", "R1", "D0")}; vals[r] = ys
        allv = [v for m in ("P0", "P1", "R1") for v in ys[m]] + [v for v in ys["D0"] if v < 35]
        lo, hi = np.floor(min(allv) / 2) * 2 - 1, np.ceil(max(allv) / 2) * 2 + 1
        for m, (_, c, mk, ls, filled) in zip(("P0", "P1", "R1", "D0"), items):
            y = np.array(ys[m]); yy = np.where(y > hi, hi, y)
            ax.plot(range(5), yy, color=c, lw=C.LW_MAIN, ls=ls, marker=mk, ms=C.MS, mfc=(c if filled else "white"), mec=c, mew=0.7, zorder=3 if m == "R1" else 2)
            for k, v in enumerate(y):
                if v > hi: ax.annotate(f"{v:.1f}", (k, hi), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=C.FS_SMALL, color=c)
        ax.set_xlim(-0.4, 4.4); ax.set_ylim(lo, hi + (2.2 if r == "Lena" else 0)); ax.set_xticks(range(5)); ax.set_xticklabels(SCHEME_LB, rotation=35, ha="right", rotation_mode="anchor")
        for k in range(5): ax.axvline(k, color="#E6E6E6", lw=0.4, zorder=0)
        if i == 0: ax.set_ylabel("RMSE (cm)")
        C.letter(fig, 14 + i * 57.5 - 9.5, 7.5, "abc"[i]); C.text_mm(fig, 14 + i * 57.5, 7.5, RN[r], fontsize=C.FS, ha="left", va="top")
    C.text_mm(fig, 14 + 57.5 + 23, 62.5, "Validation design, ordered from least to most demanding →", fontsize=C.FS, ha="center", va="bottom")
    finish(fig, "P3_D_strictness_slope", dict(design="RMSE of four methods across five validation designs ordered by strictness (lines = same method)",
                                            source="data/processed/v5/runs/core_v6/summary_metrics.csv (forcing year_obs, F8TS)", values=vals))


# ================================================================ P3_E observed vs predicted density under spatial-block CV
def p3_e():
    W = 180.0; s = 48.0; g = 6.0; x0 = 14.0; y0 = 9.0; H = y0 + 3 * s + 2 * g + 10
    fig = C.figure(W, H); meta = {}
    for i, r in enumerate(REGIONS):
        d = {m: oof(r, "block", m) for m in ("P1", "R1", "D0")}
        allv = np.concatenate([d["P1"].alt_cm.values] + [d[m].pred.values for m in d]); top = float(np.ceil(np.percentile(allv, 99.5) / 20) * 20)
        for j, m in enumerate(("P1", "R1", "D0")):
            ax = C.ax_mm(fig, x0 + j * (s + g), y0 + i * (s + g), s, s); C.clean_axes(ax)
            q = d[m]; density(ax, q.alt_cm.values, q.pred.values, MC[m], top)
            e = rmse(q.pred - q.alt_cm); rr = corr(q.alt_cm, q.pred)
            ax.text(0.04, 0.96, f"RMSE {e:.1f} cm\nr {rr:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
            meta[f"{r}_{m}"] = dict(n=int(len(q)), rmse=round(e, 2), r=round(rr, 3), axis_top=top)
            if i == 0: C.text_mm(fig, x0 + j * (s + g), 6.3, MN[m], fontsize=C.FS, ha="left", va="top"); C.letter(fig, x0 + j * (s + g) - 3.5, 2.5, "abc"[j])
            if j == 0: ax.set_ylabel(f"{RN[r]}\nPredicted ALT (cm)")
            if i == 2: ax.set_xlabel("Observed ALT (cm)")
    finish(fig, "P3_E_obs_pred_density", dict(design="observed vs predicted on all site-years under spatial-block CV; hexbin log density (n >= 2000) or points",
                                            source="core_v6/preds/<Region>__block__year_obs__F8TS__*.csv.gz (seed mean) + labels", values=meta))


# ================================================================ P3_F three questions: when (design) / with what (covariates) / where (terciles)
def p3_f():
    dv = pd.read_csv(RUNS / "core_v6/deltas_vs_ref.csv")
    a = dv[(dv.ref == "P1") & (dv.method == "R1") & (dv.featset == "F8TS") & (dv.forcing == "year_obs") & (dv.learner == "catboost") & np.isclose(dv.lam, 0.5)]
    import fig3_accuracy as F3
    lad = F3.data_c(); ter = pd.read_csv(ROOT / "outputs/figures/paper/v7/source_data/F4e_gain_conditions.csv")
    ter = ter[(ter.tercile >= 0)]
    W, H = 180.0, 92.0; fig = C.figure(W, H)
    # (a) when
    ax = C.ax_mm(fig, 26, 12, 36, 62); C.clean_axes(ax)
    for i, r in enumerate(REGIONS):
        for k, sch in enumerate(SCHEMES):
            q = a[(a.region == r) & (a.scheme == sch)]
            if q.empty: continue
            y = k + (i - 1) * 0.24
            ax.plot([q.lo.iloc[0], q.hi.iloc[0]], [y, y], color=RC[r], lw=C.LW_AUX); ax.plot(q.delta.iloc[0], y, marker=C.REGION[r][2], ms=C.MS, color=RC[r], mec=RC[r], ls="none")
    ax.axvline(0, color=C.ZERO, lw=C.LW_REF); ax.set_yticks(range(5)); ax.set_yticklabels(SCHEME_LB); ax.invert_yaxis(); ax.set_ylim(4.6, -0.6)
    ax.set_xlabel("ΔRMSE, physics + residual ML\n− recalibrated Stefan (cm)")
    C.letter(fig, 2.5, 3.0, "a"); C.text_mm(fig, 7.5, 3.3, "When does the learner help? Validation design", fontsize=C.FS, ha="left", va="top")
    # (b) with what
    ax = C.ax_mm(fig, 94, 12, 28, 62); C.clean_axes(ax)
    feats = [f for f in F3.FEATS if f in set(lad.featset)] if "featset" in lad.columns else []
    for i, r in enumerate(REGIONS):
        for k, f in enumerate(feats):
            q = lad[(lad.region == r) & (lad.featset == f)]
            if q.empty: continue
            y = k + (i - 1) * 0.24
            ax.plot([q.lo.iloc[0], q.hi.iloc[0]], [y, y], color=RC[r], lw=C.LW_AUX); ax.plot(q.delta.iloc[0], y, marker=C.REGION[r][2], ms=C.MS, color=RC[r], mec=RC[r], ls="none")
    ax.axvline(0, color=C.ZERO, lw=C.LW_REF); ax.set_yticks(range(len(feats))); SHORT = {"F8": "Climate only", "F8T": "+ Terrain", "F8TS": "+ Soil (main)", "F8TS_M": "+ MODIS", "F8TS_X": "+ Hi-res soil/cover", "F8TS_ALL": "All"}
    ax.set_yticklabels([SHORT.get(f, f) for f in feats]); ax.invert_yaxis()
    ax.set_xlabel("ΔRMSE vs recalibrated Stefan (cm)")
    C.letter(fig, 66.5, 3.0, "b"); C.text_mm(fig, 71.5, 3.3, "With which inputs? Covariate set", fontsize=C.FS, ha="left", va="top")
    hs = [Line2D([], [], ls="none", marker=C.REGION[r][2], color=RC[r], ms=C.MS) for r in REGIONS]
    ax.legend(hs, [RN[r] for r in REGIONS], loc="lower right", fontsize=C.FS_SMALL, frameon=False, handletextpad=0.4, borderaxespad=0.2)
    # (c) where: terciles heat table
    covs = [c for c in ["tdd", "fdd", "swe", "soc", "clay", "slope", "elev", "ndvi"] if c in set(ter.covariate)]
    lbl = {"tdd": "TDD", "fdd": "FDD", "swe": "SWE", "soc": "SOC", "clay": "Clay", "slope": "Slope", "elev": "Elev.", "ndvi": "NDVI"}
    ax = C.ax_mm(fig, 137, 12, 41, 62); ax.set_axis_off(); vmax = 7.0; norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    nrow, ncol = len(covs), 9
    for i, c in enumerate(covs):
        for j, r in enumerate(REGIONS):
            for t in range(3):
                q = ter[(ter.covariate == c) & (ter.region == r) & (ter.tercile == t)]
                x, y = j * 3 + t, nrow - 1 - i
                if q.empty or not bool(q.shown.iloc[0]):
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor="white", edgecolor="#BFBFBF", hatch="////", lw=0.3)); continue
                v = float(q.delta.iloc[0]); ex = bool(q.excludes_0.iloc[0])
                ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, facecolor=CM_DIV(norm(v)), edgecolor=(C.TEXT if ex else "none"), lw=0.6))
                ax.text(x + 0.5, y + 0.5, f"{v:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=5.0, color=("white" if abs(v) > 4 else C.TEXT))
        ax.text(-0.15, nrow - 1 - i + 0.5, lbl[c], ha="right", va="center", fontsize=C.FS_SMALL)
    for j, r in enumerate(REGIONS):
        ax.text(j * 3 + 1.5, nrow + 0.15, RN[r], ha="center", va="bottom", fontsize=C.FS_SMALL)
        for t, tl in enumerate(("L", "M", "H")): ax.text(j * 3 + t + 0.5, -0.15, tl, ha="center", va="top", fontsize=C.FS_SMALL)
    ax.set_xlim(-0.2, ncol + 0.2); ax.set_ylim(-0.9, nrow + 0.9); ax.set_aspect("equal")
    C.letter(fig, 125.5, 3.0, "c"); C.text_mm(fig, 130.5, 3.3, "Where? Terciles of covariates", fontsize=C.FS, ha="left", va="top")
    C.text_mm(fig, 178.0, 83.5, "L / M / H = low, mid, high tercile within the region\noutlined: 95% interval excludes 0; hatched: insufficient data", fontsize=C.FS_SMALL, ha="right", va="top", color=C.AUX)
    finish(fig, "P3_F_three_questions", dict(design="question-driven layout: validation design (a), covariate set (b), covariate terciles (c); all ΔRMSE vs recalibrated Stefan with 95% intervals",
                                            source="core_v6/deltas_vs_ref.csv; featsets runs (fig3_accuracy.data_c); v7/source_data/F4e_gain_conditions.csv"))


# ================================================================ P4_E label budget: smallest n reaching within 1 / 2 cm of the full-label error (LL) and beating the source Stefan (TL)
def p4_e():
    m = pd.read_csv(RUNS / "ladder_v6/ladder_metrics.csv"); m["n_label"] = m.n_label.astype(str)
    order = ["25", "50", "100", "250", "500", "1000", "2500", "all"]; xpos = {n: i for i, n in enumerate(order)}
    W, H = 180.0, 78.0; fig = C.figure(W, H); meta = {"LL": {}, "TL": {}}
    # (a) within region
    ax = C.ax_mm(fig, 42, 12, 50, 56); C.clean_axes(ax)
    rows = []
    for r in REGIONS:
        for mm in ("P1", "R1", "D0"):
            q = m[(m.exp == "LL") & (m.region == r) & (m.method == mm)].set_index("n_label").rmse
            if "all" not in q.index: continue
            full = q["all"]; ns = [n for n in order if n in q.index and n != "all"]
            n1 = next((n for n in ns if q[n] <= full + 1.0), "all"); n2 = next((n for n in ns if q[n] <= full + 2.0), "all")
            rows.append((r, mm, n2, n1, full)); meta["LL"][f"{r}_{mm}"] = dict(n_within_2cm=n2, n_within_1cm=n1, full_rmse=round(float(full), 2))
    for k, (r, mm, n2, n1, full) in enumerate(rows):
        y = len(rows) - 1 - k; c = MC[mm]
        ax.plot([xpos[n2], xpos[n1]], [y, y], color=c, lw=C.LW_MAIN); ax.plot(xpos[n2], y, marker="o", ms=C.MS, mfc="white", mec=c, mew=0.8, ls="none")
        ax.plot(xpos[n1], y, marker="o", ms=C.MS, color=c, ls="none"); ax.text(len(order) + 0.2, y, f"{full:.1f}", ha="left", va="center", fontsize=C.FS_SMALL, color=C.AUX)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{RN[r]} · {MN[mm]}" for r, mm, *_ in rows][::-1]); ax.set_xticks(range(len(order))); ax.set_xticklabels([o if o in ("25", "100", "500", "2500", "all") else "" for o in order])
    ax.set_xlim(-0.5, len(order) + 1.9); ax.set_ylim(-0.7, len(rows) - 0.3); ax.set_xlabel("Training labels in the region (site-years)")
    ax.text(len(order) + 0.2, len(rows) - 0.25, "Full-label\nRMSE (cm)", ha="left", va="bottom", fontsize=C.FS_SMALL, color=C.AUX)
    for k in range(0, len(rows), 3): ax.axhline(k - 0.5, color="#E6E6E6", lw=0.4, zorder=0)
    C.letter(fig, 2.5, 3.0, "a"); C.text_mm(fig, 7.5, 3.3, "Labels needed to come within 2 cm (open) or 1 cm (filled)\nof the full-label error of the same method", fontsize=C.FS, ha="left", va="top")
    # (b) transfer: smallest target-label number beating the source-coefficient Stefan
    ax = C.ax_mm(fig, 150, 12, 26, 56); C.clean_axes(ax)
    torder = ["0", "25", "50", "100", "250", "500"]; tx = {n: i for i, n in enumerate(torder)}; trows = []
    for r in REGIONS:
        p0 = m[(m.exp == "TL") & (m.region == r) & (m.method == "P0")].rmse.iloc[0]
        for mm, key in (("P1", "P1n"), ("R1", "R1S"), ("D0", "D0S")):
            q = m[(m.exp == "TL") & (m.region == r) & (m.method == key)].set_index("n_label").rmse
            nb = next((n for n in torder if n in q.index and q[n] < p0 - 0.5), None); trows.append((r, mm, nb, p0)); meta["TL"][f"{r}_{mm}"] = dict(n_beats_source_stefan_by_0_5cm=nb, source_stefan_rmse=round(float(p0), 2))
    for k, (r, mm, nb, p0) in enumerate(trows):
        y = len(trows) - 1 - k; c = MC[mm]
        if nb is None: ax.text(len(torder) - 0.6, y, "none ≤ 500", ha="left", va="center", fontsize=C.FS_SMALL, color=c)
        else: ax.plot(tx[nb], y, marker="o", ms=C.MS, color=c, ls="none")
    ax.set_yticks(range(len(trows))); ax.set_yticklabels([f"{RN[r]} · {MN[mm]}" for r, mm, *_ in trows][::-1]); ax.set_xticks(range(len(torder))); ax.set_xticklabels([o if o in ("0", "50", "250", "500") else "" for o in torder])
    ax.set_xlim(-0.5, len(torder) + 2.4); ax.set_ylim(-0.7, len(trows) - 0.3); ax.set_xlabel("Target-region labels added")
    for k in range(0, len(trows), 3): ax.axhline(k - 0.5, color="#E6E6E6", lw=0.4, zorder=0)
    C.letter(fig, 112.5, 3.0, "b"); C.text_mm(fig, 117.5, 3.3, "Target labels needed to beat the source\nStefan model by 0.5 cm in a new region", fontsize=C.FS, ha="left", va="top")
    finish(fig, "P4_E_label_budget", dict(design="label budget summary: within-region ladder (a) and transfer ladder (b)", source="ladder_v6/ladder_metrics.csv", values=meta))


# ================================================================ P4_F transfer-gain heat table: RMSE by target labels, colour = change from the no-label Stefan model
def p4_f():
    m = pd.read_csv(RUNS / "ladder_v6/ladder_metrics.csv"); m["n_label"] = m.n_label.astype(str)
    ext = pd.read_csv(ROOT / "outputs/figures/paper/v7/source_data/F10_extft.csv"); ext["n_label"] = ext.n_label.astype(str)
    blocks = []   # (title, rows=[(label, {n: rmse})], ns, baseline per row)
    tn = ["0", "25", "50", "100", "250", "500"]; en = ["0", "5", "10", "25", "50", "all"]
    for mm, key in (("R1", "R1S"), ("D0", "D0S")):
        rows = []
        for r in REGIONS:
            q = m[(m.exp == "TL") & (m.region == r) & (m.method == key)].set_index("n_label").rmse; p0 = float(m[(m.exp == "TL") & (m.region == r) & (m.method == "P0")].rmse.iloc[0])
            rows.append((RN[r], {n: float(q[n]) for n in tn if n in q.index}, p0))
        blocks.append((MN[mm], rows, tn))
    scopes = [s for s in ["all", "W Siberia", "C Siberia", "European Russia", "NE Siberia", "Chukotka", "Svalbard", "Kamchatka"] if s in set(ext.scope)]
    for mm, key in (("R1", "R1S"), ("D0", "D0S")):
        rows = []
        for sc in scopes:
            q = ext[(ext.scope == sc) & (ext.series == key)].set_index("n_label").rmse; b = ext[(ext.scope == sc) & (ext.series == "P1n") & (ext.n_label == "0")].rmse
            if q.empty or b.empty: continue
            rows.append((("All external" if sc == "all" else sc), {n: float(q[n]) for n in en if n in q.index}, float(b.iloc[0])))
        blocks.append((MN[mm], rows, en))
    W = 180.0; cw = 7.6; rh = 5.2; H = 110.0; fig = C.figure(W, H); norm = TwoSlopeNorm(vcenter=0, vmin=-20, vmax=20)
    x0 = [36, 36 + 7 * cw + 8, 36, 36 + 7 * cw + 8]; y0 = [12, 12, 44, 44]; titles = ["a  Study regions (transfer from the other two regions)", "", "b  External CALM regions (transfer from the three study regions)", ""]
    meta = {}
    for b, (title, rows, ns) in enumerate(blocks):
        ax = C.ax_mm(fig, x0[b], y0[b], (len(ns) + 1) * cw, (len(rows) + 1) * rh); ax.set_axis_off()
        ax.set_xlim(0, len(ns) + 1); ax.set_ylim(0, len(rows) + 1)
        ax.text(0.5, len(rows) + 0.5, "Stefan\n(no labels)", ha="center", va="center", fontsize=5.5, color=C.AUX)
        for j, n in enumerate(ns): ax.text(j + 1.5, len(rows) + 0.5, n, ha="center", va="center", fontsize=C.FS_SMALL)
        for i, (lab, vals, base) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.add_patch(Rectangle((0.05, y + 0.08), 0.9, 0.84, facecolor="#F2F2F2", edgecolor="none")); ax.text(0.5, y + 0.5, f"{base:.0f}", ha="center", va="center", fontsize=C.FS_SMALL, color=C.TEXT)
            if b % 2 == 0: ax.text(-0.15, y + 0.5, lab, ha="right", va="center", fontsize=C.FS_SMALL)
            for j, n in enumerate(ns):
                if n not in vals: continue
                v = vals[n]; d = v - base
                ax.add_patch(Rectangle((j + 1.05, y + 0.08), 0.9, 0.84, facecolor=CM_DIV(norm(d)), edgecolor="none"))
                ax.text(j + 1.5, y + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=C.FS_SMALL, color=("white" if abs(d) > 12 else C.TEXT))
            meta[f"{b}_{lab}"] = dict(baseline=round(base, 2), rmse={n: round(v, 2) for n, v in vals.items()})
        C.text_mm(fig, x0[b], y0[b] - 1.2, ("Physics + residual ML" if b % 2 == 0 else "Direct ML"), fontsize=C.FS, ha="left", va="bottom")
        if titles[b]: C.letter(fig, 3.0, y0[b] - 8.6, titles[b][0]); C.text_mm(fig, 8.0, y0[b] - 8.3, titles[b][3:], fontsize=C.FS, ha="left", va="top")
    # colour key
    cax = C.ax_mm(fig, 60, 97, 60, 2.2); import matplotlib as mpl
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CM_DIV, norm=norm, orientation="horizontal", ticks=[-20, -10, 0, 10, 20]); cb.outline.set_linewidth(C.LW_AX)
    cax.tick_params(labelsize=C.FS_TICK, length=1.5, width=0.5); cax.set_xlabel("Colour: RMSE − RMSE of the Stefan model without target labels (cm). Cell number: RMSE (cm)", fontsize=C.FS_SMALL, labelpad=1.5)
    finish(fig, "P4_F_transfer_heat_table", dict(design="heat table of RMSE by number of target-region labels for the physics-residual and direct models; colour = change from the no-label Stefan model",
                                               source="ladder_v6/ladder_metrics.csv (TL), v7/source_data/F10_extft.csv", values=meta))


# ================================================================ P5_D observed vs predicted matrix on the common site-years (ours, Wei, Liu)
def common_rows():
    lab = labels().reset_index(); pr = pd.read_csv(ROOT / "data/processed/v5/products_cell_year_v5.csv")
    idx = pd.MultiIndex.from_arrays([lab.loc_id, lab.year])
    for p in ("wei", "liu", "cci4y"):
        s = pr[pr["product"] == p].set_index(["loc_id", "year"]).value_cm; lab[p] = s.reindex(idx).values
    ours = {m: oof_all(m) for m in ("P1", "R1", "D0")}
    for m in ours: lab[m] = ours[m].reindex(idx).values
    c = lab[np.isfinite(lab.wei) & np.isfinite(lab.liu) & np.isfinite(lab.cci4y) & np.isfinite(lab.R1)].copy()
    return c


def oof_all(m):
    out = []
    for r in REGIONS:
        d = pd.read_csv(P / f"{r}__block__year_obs__F8TS__{PAT[m]}.csv.gz", usecols=["loc_id", "year", "pred", "seed"]); out.append(d.groupby(["loc_id", "year"]).pred.mean())
    return pd.concat(out)


def p5_d():
    c = common_rows(); W = 180.0; s = 46.0; g = 7.0; x0 = 14.0; y0 = 9.0; H = y0 + 3 * s + 2 * g + 10
    fig = C.figure(W, H); meta = {}
    series = [("R1", MN["R1"], MC["R1"]), ("wei", "Wei 2026", C.PRODUCT["wei"][1]), ("liu", "Liu 2024", C.PRODUCT["liu"][1])]
    for i, r in enumerate(REGIONS):
        q = c[c.region == r]; top = float(np.ceil(np.percentile(np.concatenate([q.alt_cm.values] + [q[k].values for k, _, _ in series]), 99.5) / 20) * 20)
        for j, (k, name, col) in enumerate(series):
            ax = C.ax_mm(fig, x0 + j * (s + g), y0 + i * (s + g), s, s); C.clean_axes(ax)
            density(ax, q.alt_cm.values, q[k].values, col, top)
            e = rmse(q[k] - q.alt_cm); rr = corr(q.alt_cm, q[k]); bias = float((q[k] - q.alt_cm).mean())
            ax.text(0.04, 0.96, f"RMSE {e:.1f} cm\nbias {bias:+.1f} cm\nr {rr:.2f}".replace("-", "−"), transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
            meta[f"{r}_{k}"] = dict(n=int(len(q)), rmse=round(e, 2), bias=round(bias, 2), r=round(rr, 3))
            if i == 0: C.text_mm(fig, x0 + j * (s + g), 6.3, name, fontsize=C.FS, ha="left", va="top"); C.letter(fig, x0 + j * (s + g) - 3.5, 2.5, "abc"[j])
            if j == 0: ax.set_ylabel(f"{RN[r]} (n = {len(q):,})\nPredicted ALT (cm)")
            if i == 2: ax.set_xlabel("Observed ALT (cm)")
    finish(fig, "P5_D_obs_pred_matrix", dict(design="observed vs predicted on the common site-years (Wei, Liu, CCI v4 and our block-CV prediction all present): regions x (ours, Wei, Liu)",
                                           source="labels + products_cell_year_v5.csv + core_v6 block OOF (R1)", values=meta))


# ================================================================ P5_F error distributions per series and region (violins, median and IQR)
def p5_f():
    c = common_rows(); series = [("P1", MN["P1"], MC["P1"]), ("R1", MN["R1"], MC["R1"]), ("D0", MN["D0"], MC["D0"]),
                                 ("wei", "Wei 2026", C.PRODUCT["wei"][1]), ("liu", "Liu 2024", C.PRODUCT["liu"][1]), ("cci4y", "ESA CCI v4", C.PRODUCT["cci4y"][1])]
    W, H = 180.0, 82.0; fig = C.figure(W, H); meta = {}; LIM = 80.0
    SHORT = {"P1": "Stefan (recal.)", "R1": "Physics + res. ML", "D0": "Direct ML", "wei": "Wei 2026", "liu": "Liu 2024", "cci4y": "ESA CCI v4"}
    for i, r in enumerate(REGIONS):
        q = c[c.region == r]; ax = C.ax_mm(fig, 14 + i * 57.5, 10, 50, 48); C.clean_axes(ax)
        for j, (k, name, col) in enumerate(series):
            e = (q[k] - q.alt_cm).values; e = e[np.isfinite(e)]; ec = np.clip(e, -LIM, LIM)
            vp = ax.violinplot([ec], positions=[j], widths=0.8, showextrema=False, showmedians=False, points=200)
            for b in vp["bodies"]: b.set_facecolor(C.tint(col, 0.45)); b.set_edgecolor(col); b.set_linewidth(0.5); b.set_alpha(1.0)
            q1, md, q3 = np.percentile(e, [25, 50, 75]); ax.plot([j, j], [max(q1, -LIM), min(q3, LIM)], color=col, lw=1.4); ax.plot(j, md, marker="o", ms=2.6, color="white", mec=col, mew=0.8)
            beyond = float((np.abs(e) > LIM).mean())
            if beyond > 0.005: ax.text(j, LIM + 2, f"{beyond * 100:.0f}%", ha="center", va="bottom", fontsize=C.FS_SMALL, color=C.AUX)
            meta[f"{r}_{k}"] = dict(n=int(len(e)), median=round(float(md), 2), q1=round(float(q1), 2), q3=round(float(q3), 2), share_beyond_80cm=round(beyond, 4))
        ax.axhline(0, color=C.ZERO, lw=C.LW_REF); ax.set_xticks(range(len(series))); ax.set_xticklabels([SHORT[k] for k, _, _ in series], rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylim(-LIM - 4, LIM + 12); ax.set_yticks([-80, -40, 0, 40, 80])
        if i == 0: ax.set_ylabel("Predicted − observed ALT (cm)")
        C.letter(fig, 14 + i * 57.5 - 9.5, 4.5, "abc"[i]); C.text_mm(fig, 14 + i * 57.5, 4.8, f"{RN[r]} (n = {len(q):,} site-years)", fontsize=C.FS, ha="left", va="top")
    C.text_mm(fig, 90, 81.0, "Violins clipped at ±80 cm (share beyond printed above); white dot = median, bar = interquartile range; our methods are spatial-block out-of-fold values", fontsize=C.FS_SMALL, ha="center", va="bottom", color=C.AUX)
    finish(fig, "P5_F_error_distributions", dict(design="error distributions (predicted − observed) of our three methods and three products on the common site-years", source="as P5_D", values=meta))


# ================================================================ P7_D anomaly agreement: observed vs mapped regional anomalies by year
def p7_d():
    d = json.load(open(ROOT / "outputs/figures/paper/v15/source_data/Fig7_maps.json"))["d"]["series"]
    W, H = 180.0, 62.0; fig = C.figure(W, H); meta = {}; keys = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}
    for i, r in enumerate(REGIONS):
        s = d.get(r) or d.get(keys[r]); pr, ob = s["predicted"], s["observed"]
        yrs = [y for y in sorted(pr) if y in ob and ob[y]["n"] >= 50]; x = np.array([ob[y]["mean"] for y in yrs]); y = np.array([pr[y] for y in yrs])
        ax = C.ax_mm(fig, 14 + i * 57.5, 10, 46, 42); C.clean_axes(ax)
        lim = float(np.ceil(max(np.abs(np.concatenate([x, y])).max(), 5) / 5) * 5)
        ax.plot([-lim, lim], [-lim, lim], color=C.ZERO, lw=C.LW_REF); ax.axhline(0, color="#E6E6E6", lw=0.4); ax.axvline(0, color="#E6E6E6", lw=0.4)
        ax.scatter(x, y, s=C.MS ** 2, color=RC[r], zorder=3)
        offs = [(3, 2), (3, -7), (-11, 2), (-11, -7)]
        for k_, (xx, yy, yr) in enumerate(sorted(zip(x, y, yrs), key=lambda t: (round(t[0] / 2), t[1]))):
            ax.annotate(yr[2:], (xx, yy), xytext=offs[k_ % 4], textcoords="offset points", fontsize=C.FS_SMALL, color=C.AUX)
        rr = corr(x, y) if len(x) >= 3 else float("nan"); ax.text(0.04, 0.96, f"r {rr:.2f}, n = {len(x)} years", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xlabel("Observed anomaly (cm)")
        if i == 0: ax.set_ylabel("Mapped anomaly (cm)")
        C.letter(fig, 14 + i * 57.5 - 9.5, 4.5, "abc"[i]); C.text_mm(fig, 14 + i * 57.5, 4.8, RN[r], fontsize=C.FS, ha="left", va="top")
        meta[r] = dict(years=yrs, r=round(rr, 3))
    C.text_mm(fig, 90, 61.0, "Anomaly from the 2011–2024 mean; observed anomaly = mean over observed cells of years with at least 50 site-years; mapped = regional mean of the yearly map", fontsize=C.FS_SMALL, ha="center", va="bottom", color=C.AUX)
    finish(fig, "P7_D_anomaly_agreement", dict(design="observed vs mapped regional ALT anomaly per year (alternative to Fig. 7d time series)", source="v14/source_data/Fig7_maps.json (d)", values=meta))


if __name__ == "__main__":
    C.setup(); t0 = time.time(); sel = sys.argv[1:] or ["p3_d", "p3_e", "p3_f", "p4_e", "p4_f", "p5_d", "p5_f", "p7_d"]
    for name in sel:
        try:
            print(f"== {name}"); globals()[name](); print(f"   {time.time() - t0:.0f} s")
        except Exception:
            traceback.print_exc()
