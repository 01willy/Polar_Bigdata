# -*- coding: utf-8 -*-
"""Paper v11 Fig. 4: label budget and new regions (Scientific Reports, 180 mm).

Copied from paper_v10/fig4_labels.py (data functions unchanged) and restyled for v11 (fig34_common12.py, POLAR_PALETTE=paper12).
a  In-region label ladder (spatial block folds; the training fold is subsampled to n labels): dRMSE = RMSE(method) - RMSE(Stefan recalibrated
   on the same n labels), Physics + residual ML (line + light 95 % block-bootstrap band) and Direct ML; n = 25 ... all on a categorical
   axis; n values present only in some folds are dropped (n_test < 0.95 x n_test of 'all'), as scripts/4_visualization/v8_f9_label_ladder.load.
b  New-region transfer: RMSE in the held-out region vs target-region labels added (0, 25, 50, 100, 250, 500): Stefan (source coeff.),
   Stefan (recalibrated on the n labels), Direct ML (source + n), Physics + residual ML (source + n). Out-of-range values
   (> 1.3 x the largest value at n >= 25 in the panel) are drawn on the top edge with the value printed.
c  External CALM regions (data/processed/v9_ext): RMSE vs labels from the same external region (0, 5, 10, 25, 50, all), seed-mean predictions
   recomputed from runs/extra_v9/preds_EXTFT_*.csv.gz on the cell-years that have Wei 2026, Liu 2024 and ESA CCI v5 values; Stefan
   (recalibrated on n) is drawn as a wide light-blue line under Physics + residual ML (they nearly coincide; keyed inside the panel);
   horizontal reference lines for the three products in their v11 colours (ESA CCI v5 dashed; Wei 2026 and Liu 2024 dotted, trained on these sites).
Inputs: data/processed/v5/runs/ladder_v6/{ladder_metrics,ladder_deltas}.csv (+ preds_LL_*.csv.gz for the 'all' fold sizes),
        data/processed/v5/runs/extra_v9/preds_EXTFT_*.csv.gz (check: outputs/figures/paper/v7/source_data/F10_extft.csv).
Output: outputs/figures/paper/v12/Fig4_labels.{pdf,svg,png} and source_data/Fig4_labels.json.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper12 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v12/fig4_labels.py
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig34_common12 as C  # noqa: E402
import polar.style_v5 as S  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402

ROOT = C.ROOT; RUN = ROOT / "data" / "processed" / "v5" / "runs" / "ladder_v6"; EXTRA = ROOT / "data" / "processed" / "v5" / "runs" / "extra_v9"
F10 = ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F10_extft.csv"
STEM = "Fig4_labels"
REG = C.REG
TL_KEY = {"P0": "P0", "P1": "P1n", "D0": "D0S", "R1": "R1S"}       # style key -> TL method
TL_ORDER = ["P0", "P1", "D0", "R1"]                                  # drawing order (Physics + residual ML on top)
FT_NS = ["0", "5", "10", "25", "50", "all"]
OVER = 1.3
DOT = (0, (1.0, 1.4)); REF_DASH = (0, (3.0, 1.8))
PROD_REF = [("cci5y", "ESA CCI v5", REF_DASH), ("liu", "Liu 2024", DOT), ("wei", "Wei 2026", DOT)]   # product reference lines (Fig. 4c)
LW_PROD = 0.9
DELTA_LB = "ΔRMSE vs recalibrated Stefan (cm)"

# ---------------------------------------------------------------- layout (mm; y from the top edge)
H = 122.0
Y_LEG = 1.4
Y1, H1 = 11.5, 42.5
Y2, H2 = 71.5, 42.5
DY2 = 1.3                           # facet-label offset above the axes (same in all rows of Figs 3 and 4)
X0, GAP, R_EDGE = 11.5, 7.2, 179.0
C_GAP0, C_LAB = 11.5, 14.6          # gap b -> c axes, room for the reference labels right of c


def cats(labels):
    num = sorted({int(v) for v in labels if v != "all"}); return [str(v) for v in num] + (["all"] if "all" in set(labels) else [])


def cat_label(c):
    return "All" if c == "all" else str(int(c))                      # four-digit numbers without a comma (journal style)


# ---------------------------------------------------------------- data (unchanged from v10)
def load_ladder():
    """As v8_f9_label_ladder.load: drop LL rows whose n_test is below 95 % of the region's 'all' n_test (n present only in some folds)."""
    m = pd.read_csv(RUN / "ladder_metrics.csv"); d = pd.read_csv(RUN / "ladder_deltas.csv")
    for t in (m, d):
        t["n_label"] = t.n_label.astype(str)
    full = m[(m.exp == "LL") & (m.n_label == "all")].groupby("region").n_test.max()
    dropped = sorted({(r, n) for t in (m, d) for r, n, nt, e in zip(t.region, t.n_label, t.n_test, t.exp) if e == "LL" and nt < 0.95 * full[r]})
    for t in (m, d):
        keep = (t.exp != "LL") | (t.n_test >= 0.95 * t.region.map(full))
        t.drop(index=t.index[~keep], inplace=True)
    return m, d, dropped


def ll_fold_sizes():
    """Training-fold sizes for n = 'all' (from the LL prediction files) and label years."""
    out = {}
    for r in REG:
        p = pd.read_csv(RUN / f"preds_LL_{r}.csv.gz", usecols=["fold", "n", "n_label", "method", "seed", "year", "cell_year_id"], dtype={"n_label": str})
        q = p[(p.method == "P1") & (p.seed == 0)]
        a = q[q.n_label == "all"].groupby("fold").n.first()
        out[r] = dict(all_training_labels_per_fold={k: int(v) for k, v in a.items()}, years=f"{int(q.year.min())}-{int(q.year.max())}",
                      test_cell_years=int(q[q.n_label == 'all'].cell_year_id.nunique()))
    return out


def ft_table():
    """Seed-mean predictions -> RMSE on the rows with all three product values (Wei 2026, Liu 2024, ESA CCI v5), as v9_f10_extra.ft_table."""
    p = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(EXTRA / "preds_EXTFT_*.csv.gz")))], ignore_index=True); p["n_label"] = p.n_label.astype(str)
    first = p.drop_duplicates("cell_year_id").set_index("cell_year_id")[["y", "year", "loc_id", "block", "subregion", "prod_wei", "prod_liu", "prod_cci5y"]]
    rows = []
    for nl in FT_NS:
        w = p[p.n_label == nl].groupby(["cell_year_id", "method"]).pred.mean().unstack().join(first)
        w = w[w.prod_wei.notna() & w.prod_liu.notna() & w.prod_cci5y.notna()]
        for m in ("P0", "P1n", "R1S", "D0S"):
            rows.append(dict(n_label=nl, series=m, rmse=float(np.sqrt(np.mean((w[m] - w.y) ** 2))), n=len(w)))
        for ser, col in (("wei", "prod_wei"), ("liu", "prod_liu"), ("cci5y", "prod_cci5y")):
            rows.append(dict(n_label=nl, series=ser, rmse=float(np.sqrt(np.mean((w[col] - w.y) ** 2))), n=len(w)))
    t = pd.DataFrame(rows)
    rows_used = first[first.prod_wei.notna() & first.prod_liu.notna() & first.prod_cci5y.notna()]
    info = dict(cell_years=int(len(rows_used)), cells=int(rows_used.loc_id.nunique()), blocks=int(rows_used.block.nunique()),
                years=f"{int(rows_used.year.min())}-{int(rows_used.year.max())}", subregions=rows_used.subregion.value_counts().to_dict(),
                all_external_cell_years=int(len(first)), all_external_years=f"{int(first.year.min())}-{int(first.year.max())}",
                all_external_subregions=sorted(first.subregion.unique().tolist()),
                n_label_all_pool_sizes=sorted({int(v) for v in p[p.n_label == "all"].n.unique()}))
    # check against the earlier aggregate (same definition)
    old = pd.read_csv(F10); old["n_label"] = old.n_label.astype(str); old = old[old.scope == "all"]
    for _, z in old.iterrows():
        v = t[(t.n_label == z.n_label) & (t.series == z.series)].rmse
        assert len(v) == 1 and abs(float(v.iloc[0]) - z.rmse) < 1e-6, (z.n_label, z.series)
    return t, info


# ---------------------------------------------------------------- drawing
def widths(ncat, x0, x1, gap):
    u = (x1 - x0 - gap * (len(ncat) - 1)) / sum(ncat)
    xs = np.cumsum([x0] + [n * u + gap for n in ncat[:-1]])
    return [(x, n * u) for x, n in zip(xs, ncat)], u


def draw_a(fig, d):
    q = d[(d.exp == "LL") & (d.ref == "P1")]
    ncat = [len(cats(q[q.region == r].n_label.unique())) for r in REG]
    boxes, u = widths(ncat, X0, R_EDGE, GAP)
    axes, vals = [], {}
    for (x, w), r in zip(boxes, REG):
        ax = C.ax_mm(fig, x, Y1, w, H1); axes.append(ax)
        C.grid(ax, "y")
        qr = q[q.region == r]; cs = cats(qr.n_label.unique()); pos = {c: i for i, c in enumerate(cs)}
        s1 = qr[qr.method == "R1"].assign(x=lambda z: z.n_label.map(pos)).sort_values("x")
        s0 = qr[qr.method == "D0"].assign(x=lambda z: z.n_label.map(pos)).sort_values("x")
        dx = 0.10                                    # small dodge so coinciding values (Lena n = 25, Canada n = 100) stay visible
        ax.fill_between(s1.x - dx, s1.lo, s1.hi, color=C.BAND, lw=0, zorder=1)
        C.zero_line(ax)
        C.plot_method(ax, s0.x.values + dx, s0.delta.values, "D0", zorder=3)
        C.plot_method(ax, s1.x.values - dx, s1.delta.values, "R1", zorder=4)
        lo = min(s1.lo.min(), s0.delta.min(), 0); hi = max(s1.hi.max(), s0.delta.max(), 0); pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad); ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlim(-0.5, len(cs) - 0.5); ax.set_xticks(range(len(cs))); ax.set_xticklabels([cat_label(c) for c in cs])
        C.facet(ax, C.REG_EN[r])
        vals[C.REG_EN[r]] = {"categories": [cat_label(c) for c in cs],
                             "Physics + residual ML": {cat_label(c): [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)] for c, z in zip(s1.n_label, s1.itertuples())},
                             "Direct ML": {cat_label(c): [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)] for c, z in zip(s0.n_label, s0.itertuples())}}
    axes[0].set_ylabel(DELTA_LB)
    return axes, vals, u


def draw_b(fig, m, boxes):
    q = m[m.exp == "TL"]
    dodge = {"P0": -0.24, "P1": -0.08, "R1": 0.08, "D0": 0.24}
    axes, vals, clipped = [], {}, []
    for (x, w), r in zip(boxes, REG):
        ax = C.ax_mm(fig, x, Y2, w, H2); axes.append(ax)
        C.grid(ax, "y")
        qr = q[q.region == r]; cs = cats(qr.n_label.unique()); pos = {c: i for i, c in enumerate(cs)}
        ser = {k: qr[qr.method == TL_KEY[k]].assign(x=lambda z: z.n_label.map(pos)).sort_values("x") for k in TL_ORDER}
        allv = np.concatenate([s.rmse.values for s in ser.values()])
        rest = np.concatenate([s[s.n_label != "0"].rmse.values for s in ser.values()]); thr = OVER * rest.max()
        over_any = (allv > thr).any(); yin = allv[allv <= thr]
        span = yin.max() - yin.min(); ylo = yin.min() - 0.07 * span; yhi = yin.max() + (0.16 if over_any else 0.07) * span
        ax.set_ylim(ylo, yhi)
        for k in TL_ORDER:
            s = ser[k]; v = s.rmse.values; xx = s.x.values + dodge[k]; ov = v > thr; vc = np.where(ov, yhi, v); st = C.METH[k]
            z = 2 + TL_ORDER.index(k) * 0.2
            ax.plot(xx, vc, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=z, clip_on=False)
            ax.plot(xx[~ov], v[~ov], ls="none", marker=st["marker"], ms=st["ms"], mfc=C.WHITE if st["open"] else st["color"], mec=st["color"],
                    mew=C.MEW if st["open"] else 0.0, zorder=z + 0.1)
            for i in np.where(ov)[0]:
                ax.plot([xx[i]], [yhi], ls="none", marker=st["marker"], ms=st["ms"], mfc=C.WHITE, mec=st["color"], mew=C.MEW, zorder=5, clip_on=False)
                # value printed right of and just below its top-edge marker (inside the empty corner; the facet label stays at the top left)
                ax.annotate(C.num(v[i]), (xx[i], yhi), xytext=(3.2, -0.6), textcoords="offset points", ha="left", va="top",
                            fontsize=C.FS_TICK, color=C.TEXT, annotation_clip=False)
                clipped.append(dict(region=C.REG_EN[r], method=C.METH[k]["label"], n_label=s.n_label.values[i], value=round(float(v[i]), 2),
                                    printed=C.num(v[i]), threshold_cm=round(float(thr), 2), axis_top_cm=round(float(yhi), 2)))
        ax.set_xlim(-0.5, len(cs) - 0.5); ax.set_xticks(range(len(cs))); ax.set_xticklabels([cat_label(c) for c in cs])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=True))
        C.facet(ax, C.REG_EN[r], dy_mm=DY2)
        vals[C.REG_EN[r]] = {C.METH[k]["label"]: {cat_label(c): round(float(v), 2) for c, v in zip(ser[k].n_label, ser[k].rmse)} for k in TL_ORDER}
    axes[0].set_ylabel("RMSE (cm)")
    return axes, vals, clipped


def draw_c(fig, t, x, w):
    ax = C.ax_mm(fig, x, Y2, w, H2)
    C.grid(ax, "y")
    xs = np.arange(len(FT_NS))
    get = lambda s: np.array([t[(t.series == s) & (t.n_label == n)].rmse.iloc[0] for n in FT_NS])
    dx = 0.10                                                                                 # same dodge as panel a
    ax.plot(xs - dx, get("P1n"), color=C.HALO, lw=3.0, solid_capstyle="round", zorder=2)     # Stefan (recalibrated on n), under R1
    C.plot_method(ax, xs + dx, get("D0S"), "D0", zorder=3)
    C.plot_method(ax, xs - dx, get("R1S"), "R1", zorder=4)
    ylo, yhi = 12.0, 66.0
    ax.set_ylim(ylo, yhi); ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlim(-0.5, len(FT_NS) - 0.5); ax.set_xticks(xs); ax.set_xticklabels([cat_label(n) for n in FT_NS])
    rv = {}
    for key, lab, ls in PROD_REF:
        v = float(t[(t.series == key) & (t.n_label == "0")].rmse.iloc[0]); rv[lab] = v
        ax.axhline(v, color=S.PRODUCT_HEX[key], lw=LW_PROD, ls=ls, zorder=1.5)
        ax.annotate(lab, (1.0, v), xycoords=("axes fraction", "data"), xytext=(2.5, 0), textcoords="offset points", ha="left", va="center_baseline",
                    fontsize=C.FS_TICK, color=C.TEXT, annotation_clip=False)
    ax.plot([-0.40, -0.40], [rv["Wei 2026"], rv["Liu 2024"]], color=C.ZERO, lw=0.6, solid_capstyle="butt", zorder=1.6)   # bracket: the note covers both lines
    ax.text(-0.31, 0.5 * (rv["Liu 2024"] + rv["Wei 2026"]), "Trained on these sites", ha="left", va="center_baseline", fontsize=C.FS_TICK, color=C.TEXT,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=1.8)   # white backing hides the light grid line behind the note
    C.facet(ax, "External CALM regions", dy_mm=DY2)
    ax.set_ylabel("RMSE (cm)")
    ax.legend([Line2D([], [], color=C.HALO, lw=3.0, solid_capstyle="round")], ["Stefan (recalibrated)"], loc="center right",
              bbox_to_anchor=(1.0, (31.5 - ylo) / (yhi - ylo)), frameon=False, fontsize=C.FS_TICK, handlelength=1.8, handletextpad=0.5,
              borderaxespad=0.3, borderpad=0)
    vals = {lab: {cat_label(n): round(float(v), 2) for n, v in zip(FT_NS, get(s))} for s, lab in (("R1S", "Physics + residual ML"), ("P1n", "Stefan (recalibrated on n)"),
                                                                                             ("D0S", "Direct ML"), ("P0", "Stefan (source coeff.)"))}
    vals.update({f"{lab} (reference line)": round(v, 2) for lab, v in rv.items()})
    return ax, vals


def main():
    C.setup()
    m, d, dropped = load_ladder(); fold_info = ll_fold_sizes(); ft, ft_info = ft_table()
    fig = C.new_fig(H)
    a_axes, a_vals, u_a = draw_a(fig, d)
    # row 2: b (3 panels, 6 categories) and c (6 categories) share one category width
    ncb = [len(cats(m[(m.exp == "TL") & (m.region == r)].n_label.unique())) for r in REG]
    tot = R_EDGE - C_LAB - X0 - 2 * GAP - C_GAP0
    u_b = tot / (sum(ncb) + len(FT_NS)); boxes = []
    x = X0
    for n in ncb:
        boxes.append((x, n * u_b)); x += n * u_b + GAP
    b_axes, b_vals, clipped = draw_b(fig, m, boxes)
    c_x = boxes[-1][0] + boxes[-1][1] + C_GAP0
    ax_c, c_vals = draw_c(fig, ft, c_x, len(FT_NS) * u_b)
    # x-axis labels below the tick labels of each group
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); px = 25.4 / fig.dpi
    def below(axes):
        return H - min(lb.get_window_extent(rd).y0 for a in axes for lb in a.get_xticklabels()) * px + 1.0
    mid = lambda axes: (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2 * C.W
    C.text_mm(fig, mid(a_axes), below(a_axes), "Training labels in the region (cell-years)", ha="center", va="top")
    C.text_mm(fig, mid(b_axes), below(b_axes), "Target-region labels added (cell-years)", ha="center", va="top")
    C.text_mm(fig, mid([ax_c]), below([ax_c]), "External-region labels (cell-years)", ha="center", va="top")
    # legend (one row, top)
    hs = [C.method_handle(k) for k in ("P0", "P1", "R1")] + [Patch(facecolor=C.BAND, edgecolor="none"), C.method_handle("D0")]
    labs = [C.METH[k]["label"] for k in ("P0", "P1", "R1")] + ["95% interval", C.METH["D0"]["label"]]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(X0 / C.W, 1 - Y_LEG / H), ncol=5, frameon=False, fontsize=C.FS, handlelength=2.4,
               handletextpad=0.5, columnspacing=1.8, borderaxespad=0, borderpad=0)
    C.letter(fig, 0.8, Y1 - 1.3, "a"); C.letter(fig, 0.8, Y2 - DY2, "b"); C.letter(fig, c_x - 10.5, Y2 - DY2, "c")
    probs = C.qa_report(fig, STEM)
    margins = C.ink_margins_mm(fig); print("ink margins (mm):", margins, "| category width a/b,c (mm):", round(u_a, 2), round(u_b, 2))
    outs = C.export(fig, STEM)
    yrs = {C.REG_EN[r]: fold_info[r]["years"] for r in REG}
    src = {
        "figure": "Fig. 4", "version": "v11 (2026-10-16): v11 palette restyle of v10 Fig. 4; numbers unchanged from v10",
        "files": [str(p.relative_to(ROOT)) for p in outs], "script": "scripts/4_visualization/paper_v12/fig4_labels.py",
        "size_mm": [C.W, H], "qa_problems": probs, "ink_margins_mm": margins,
        "style": {"palette": "design/style_tokens_v4.json color.paper_v12 (POLAR_PALETTE=paper12)", "text": C.TEXT, "axes": C.AXIS, "grid": C.GRID,
                  "methods": {C.METH[k]["label"]: C.METH[k]["color"] for k in TL_ORDER}, "band_95": C.BAND, "stefan_halo_c": C.HALO,
                  "product_reference_lines_c": {lab: [S.PRODUCT_HEX[k], "dashed" if ls == REF_DASH else "dotted"] for k, lab, ls in PROD_REF}},
        "main_setting": "climate input year_obs (ERA5-Land of the label year, TDD to the observation day), featset F8TS, CatBoost, Physics + residual ML lambda 0.5, 3 seeds (seed-mean predictions)",
        "panels": {
            "a": {
                "quantity": "dRMSE = RMSE(method) - RMSE(Stefan recalibrated on the same n labels) (cm); band = 95 % block-bootstrap interval of Physics + residual ML (1000 resamples)",
                "input_files": ["data/processed/v5/runs/ladder_v6/ladder_deltas.csv", "data/processed/v5/runs/ladder_v6/ladder_metrics.csv (n_test filter)",
                                "data/processed/v5/runs/ladder_v6/preds_LL_<region>.csv.gz (fold sizes for 'All')"],
                "row_filter": "exp == LL, ref == P1, method in {R1, D0}; rows with n_test < 0.95 x n_test of 'all' dropped",
                "dropped_region_n": [f"{C.REG_EN[r]} n={n}" for r, n in dropped],
                "design": "spatial block 5-fold CV; in each fold the training fold is subsampled to n labels (seeds 0, 1, 2) and the whole test fold is scored",
                "label_years": yrs, "climate_input_years": "year_obs for each label year",
                "all_category_training_labels_per_fold": {C.REG_EN[r]: fold_info[r]["all_training_labels_per_fold"] for r in REG},
                "plotted_delta_cm_95ci": a_vals, "printed_numbers": "none (tick labels only)"},
            "b": {
                "quantity": "pooled RMSE (cm) over the held-out region's block test folds",
                "input_files": ["data/processed/v5/runs/ladder_v6/ladder_metrics.csv"],
                "row_filter": "exp == TL, method in {P0, P1n, D0S, R1S}",
                "design": "target region = test region; training = all labels of the two other regions (source) + n labels drawn from target blocks outside the test fold; n = 0 means source only (Stefan recalibrated = source coefficient)",
                "label_years": yrs, "source_label_years": "the other two regions, all years (Alaska 2010-2024, Lena Delta 2010-2022, Canada (NW) 2010-2024)",
                "climate_input_years": "year_obs for each label year",
                "out_of_range_rule": f"values > {OVER} x the largest value at n >= 25 in that panel are drawn on the top edge with the value printed",
                "plotted_rmse_cm": b_vals, "printed_numbers": clipped},
            "c": {
                "quantity": "RMSE (cm) on external CALM cell-years that have Wei 2026, Liu 2024 and ESA CCI v5 values",
                "input_files": ["data/processed/v5/runs/extra_v9/preds_EXTFT_*.csv.gz (recomputed here; equal to outputs/figures/paper/v7/source_data/F10_extft.csv scope 'all' within 1e-6)"],
                "row_filter": "seed-mean prediction per cell-year and n; rows with prod_wei, prod_liu and prod_cci5y all present",
                "design": "training = all labels of the three study regions (2010-2024) + n labels from other blocks of the same external sub-region (one block held out per fold; Greenland one cell); cells with mean ALT > 300 cm excluded",
                "rows": ft_info,
                "label_years": f"external CALM observations {ft_info['all_external_years']}; comparison rows {ft_info['years']}",
                "climate_input_years": "year_obs for each label year",
                "note": "Stefan (recalibrated on n) is the wide light-blue line under Physics + residual ML (keyed inside the panel); the two differ by at most 0.55 cm; a thin grey bracket joins the Liu 2024 and Wei 2026 lines next to the note 'trained on these sites'",
                "plotted_rmse_cm": c_vals,
                "printed_numbers": {"reference labels": [lab for _, lab, _ in PROD_REF], "text": "Trained on these sites"}},
        },
    }
    C.write_json(C.SRC / f"{STEM}.json", src)
    for p in outs:
        print(p, p.stat().st_size)
    return probs


if __name__ == "__main__":
    main()
