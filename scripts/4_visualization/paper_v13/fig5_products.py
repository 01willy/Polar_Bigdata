"""Fig. 5 (v13 manuscript): comparison with published ALT products on the same cells and years (a RMSE, b observed vs predicted, c bias by ALT class, d year-wise).
Copied from scripts/4_visualization/paper_v10/fig5_products.py and edited for v11 (colours per series, dark-grey text, thin lines, small markers).

a  RMSE on common cell-years (cell-years where Wei 2026, Liu 2024 and ESA CCI v4 all have a value of the observation year; 2010-2020).
   Rows: Stefan (recalibrated), Physics + residual ML, Direct ML | Wei 2026, Liu 2024, ESA CCI v4, ESA CCI v5, Aalto 2018. One panel per region.
   Filled marker = all common cell-years, open marker = cells > 25 km from product (Wei 2026) training sites. Values beyond 45 cm are drawn at the
   axis end and printed. ESA CCI v5 and Aalto 2018 are scored on the common cell-years where they have a value.
b  RMSE per observation year (years with >= 30 cell-years) for Physics + residual ML, Stefan (recalibrated), Wei 2026 and Liu 2024 (to 2020),
   same cell-years within each year; grey bars = cell-years per year (log scale).
Inputs: data/processed/v5/runs/core_v6/common_cells_eval.csv (v5_common_cells.py --run core_v6 --forcing year_obs --ens none),
        outputs/figures/paper/v7/source_data/F5_yearwise_rmse.csv (v7_f5_products_years.yearwise_rmse; read only),
        data/processed/v5 labels/products via polar.testbed_v5.load_v5 (year coverage of the common cell-years, for the source-data JSON).
Outputs: outputs/figures/paper/v13/Fig5_products.{pdf,svg,png}, outputs/figures/paper/v13/source_data/Fig5_products.json
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v13/fig5_products.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = C.ROOT
CC_CSV = ROOT / "data" / "processed" / "v5" / "runs" / "core_v6" / "common_cells_eval.csv"
YW_CSV = ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F5_yearwise_rmse.csv"
LEAK_CSV = ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv"
STEM = "Fig5_products"
REG = C.REGIONS
# rows of panel a: (series code in common_cells_eval.csv, label, face colour, edge colour, marker)
ROWS = [("P1 (Stefan, recal.)", C.METHOD_NAME["P1"], C.METHOD["P1"], C.METHOD["P1"], "s"),
        ("R1 (CatBoost)", C.METHOD_NAME["R1"], C.METHOD["R1"], C.METHOD["R1"], "o"),
        ("D0 (CatBoost)", C.METHOD_NAME["D0"], C.METHOD["D0"], C.METHOD["D0"], "o")]
for _code, _key in (("Wei 2026", "wei"), ("Liu 2024", "liu"), ("ESA CCI v4", "cci4y"), ("ESA CCI v5", "cci5y"), ("Aalto 2018", "aalto")):
    _n, _f, _e, _m = C.product_style(_key)
    ROWS.append((_code, _n, _f, _e, _m))
N_OURS = 3
X_A = (10.0, 45.0)                       # a: main x range (cm)
X_S = (45.0, 130.0); X_S_TICKS = (60, 100)   # a: compressed right segment for ESA CCI v5 and Aalto 2018 (broken axis)
GAP_ROWS = 0.35                          # extra space (rows) at the rule between our methods and products
DY = 0.2                                 # a: sub-row offset (filled above, open below)
MS_A = 3.3
MIN_N_YEAR = 30
YW_SER = [("r1", C.METHOD_NAME["R1"], C.METHOD["R1"], C.METHOD["R1"], "o", "-", C.LW_MAIN, 2.8),
          ("p1", C.METHOD_NAME["P1"], C.METHOD["P1"], C.METHOD["P1"], "s", "-", C.LW_MAIN, 2.4)]
for _k, _ls in (("wei", "-"), ("liu", C.DASH)):
    _n, _f, _e, _m = C.product_style(_k)
    YW_SER.append((_k, _n, _f, _e, _m, _ls, C.LW_AUX, 2.8 if _m != "D" else 2.5))
NBAR = "#D4D4D4"


def row_y(i):
    return i + (GAP_ROWS if i >= N_OURS else 0.0)


def common_years():
    """Year coverage of the common cell-years (same rule as scripts/2_evaluation/v5_common_cells.py)."""
    import polar.testbed_v5 as T
    base = T.load_v5(ROOT / "data" / "processed" / "v5")
    lk = pd.read_csv(LEAK_CSV); lk = lk[lk["product"] == "wei"].groupby("loc_id").dist_km.min(); base["dist"] = base.loc_id.map(lk)
    out = {}
    for r in REG:
        s = base[base.region == r]; out[r] = {}
        for mask in (0, 25):
            q = s if mask == 0 else s[~(s.dist <= mask)]
            c = q[np.isfinite(q.prod_wei) & np.isfinite(q.prod_liu) & np.isfinite(q.prod_cci4y) & np.isfinite(q.alt_cm)]
            out[r][str(mask)] = dict(n=int(len(c)), n_cells=int(c.loc_id.nunique()), years=[int(c.year.min()), int(c.year.max())],
                                     n_by_year={str(int(k)): int(v) for k, v in c.year.value_counts().sort_index().items()},
                                     n_cci5y=int(np.isfinite(c.prod_cci5y).sum()), n_aalto=int(np.isfinite(c.prod_aalto).sum()))
    return out


def draw_a(fig, axes, sides, cc, src):
    """Dot plot of RMSE per region. Values up to X_A[1] on the main axes; larger values (ESA CCI v5, Aalto et al.) on a compressed right
    segment (X_S) separated by a broken axis, so that no value is printed as a number."""
    ny = len(ROWS); ylim = (row_y(ny - 1) + 0.62, -0.62)
    beyond = []
    for j, (ax, axs, r) in enumerate(zip(axes, sides, REG)):
        C.clean_axes(ax, left=(j == 0)); C.clean_axes(axs, left=False)
        ax.set_xlim(*X_A); ax.set_ylim(*ylim); axs.set_xlim(*X_S); axs.set_ylim(*ylim)
        for a_ in (ax, axs):
            for i in range(ny):
                a_.axhline(row_y(i), color=C.GRID, lw=C.LW_THIN, zorder=0)
            a_.axhline((row_y(N_OURS - 1) + row_y(N_OURS)) / 2, color=C.ZERO, lw=C.LW_THIN, zorder=0.5)
        vals = {}
        for i, (code, name, face, edge, mk) in enumerate(ROWS):
            for mask, dy, filled in ((0, -DY, True), (25, DY, False)):
                q = cc[(cc.region == r) & (cc.mask_km == mask) & (cc.series == code)]
                if not len(q):
                    continue
                v = float(q.rmse.iloc[0]); n = int(q.n.iloc[0]); vals[f"{name}|{mask}"] = dict(rmse_cm=round(v, 3), n=n)
                y = row_y(i) + dy; ms = MS_A if mk not in ("D", "v") else MS_A - 0.3
                tgt = ax if v <= X_A[1] else axs
                tgt.plot([v], [y], ls="none", marker=mk, ms=ms, mfc=face if filled else "white", mec=edge, mew=0.6, zorder=3, clip_on=False)
                if v > X_A[1]:
                    beyond.append(dict(region=r, series=name, subset="all" if mask == 0 else "> 25 km", rmse_cm=round(v, 1)))
        ax.set_yticks([row_y(i) for i in range(ny)])
        ax.set_yticklabels([n for _, n, *_ in ROWS] if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=2.0)
        ax.set_xticks([10, 20, 30, 40]); axs.set_xticks(list(X_S_TICKS)); axs.set_yticks([])
        # broken-axis marks at the junction of the two segments
        wmm = ax.get_position().width * fig.get_figwidth() * 25.4; hmm = ax.get_position().height * fig.get_figheight() * 25.4
        dx, dy_ = 0.7 / wmm, 0.9 / hmm; kw = dict(color=C.AXIS, lw=C.LW_AX, clip_on=False, zorder=5)
        ax.plot((1 - dx, 1 + dx), (-dy_, dy_), transform=ax.transAxes, **kw)
        wmm_s = axs.get_position().width * fig.get_figwidth() * 25.4
        dxs = 0.7 / wmm_s
        axs.plot((-dxs, dxs), (-dy_, dy_), transform=axs.transAxes, **kw)
        n0 = int(cc[(cc.region == r) & (cc.mask_km == 0) & (cc.series == "R1 (CatBoost)")].n.iloc[0])
        n25 = int(cc[(cc.region == r) & (cc.mask_km == 25) & (cc.series == "R1 (CatBoost)")].n.iloc[0])
        ax.text(0.0, 1.0, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        axs.text(1.0, 1.0, f"n = {n0:,} / {n25:,}", transform=axs.transAxes, ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.AUX)
        if j == 1:
            ax.set_xlabel("RMSE on common site-years (cm)")
        src["a"]["values"][r] = vals; src["a"]["n_label"][r] = dict(all=n0, gt25km=n25)
    src["a"]["right_segment"] = dict(range_cm=list(X_S), ticks=list(X_S_TICKS), values=beyond, note="values above %g cm drawn on the compressed segment" % X_A[1])


def draw_b(fig, axes, strips, yw, src):
    q0 = yw[(yw.mask_km == 0) & yw.shown]
    NL = (8.0, 1.2e4)
    for j, (ax, axn, r) in enumerate(zip(axes, strips, REG)):
        C.clean_axes(ax); C.clean_axes(axn)
        ax.set_axisbelow(True)
        q = q0[q0.region == r].sort_values("year")
        for key, name, face, edge, mk, ls, lw, ms in YW_SER[::-1]:
            z = q[np.isfinite(q[f"rmse_{key}"])]
            yrs, v = z.year.values, z[f"rmse_{key}"].values
            seg = np.split(np.arange(len(yrs)), np.where(np.diff(yrs) > 1)[0] + 1) if len(yrs) else []
            main = key in ("r1", "p1")
            for sg in seg:
                if len(sg) > 1:
                    ax.plot(yrs[sg], v[sg], ls=ls, color=face, lw=lw, zorder=3.5 if key == "r1" else (3.2 if key == "p1" else 3))
            ax.plot(yrs, v, ls="none", marker=mk, ms=ms, mfc=face, mec="white" if main else edge, mew=0.4 if main else 0.4,
                    zorder=4.5 if key == "r1" else (4.2 if key == "p1" else 4))
        vmax = np.nanmax(q[[f"rmse_{k}" for k, *_ in YW_SER]].values)
        top = float(np.ceil(1.08 * vmax / 10.0) * 10.0)
        ax.set_xlim(2009.3, 2024.7); ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20 if top > 60 else 10))
        ax.set_xticks([2010, 2014, 2018, 2022]); ax.tick_params(axis="x", labelbottom=False)
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1)); ax.tick_params(axis="x", which="minor", length=1.0, width=0.4)
        hb_mm = ax.get_position().height * fig.get_figheight() * 25.4
        ax.text(0.0, 1.0 + 1.5 / hb_mm, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        axn.bar(q.year, q.n, width=0.62, color=NBAR, lw=0, zorder=2)
        axn.set_yscale("log"); axn.set_ylim(*NL); axn.set_xlim(*ax.get_xlim())
        axn.set_xticks([2010, 2014, 2018, 2022]); axn.set_yticks([10, 100, 1000]); axn.yaxis.set_minor_locator(mticker.NullLocator())
        axn.xaxis.set_minor_locator(mticker.MultipleLocator(1)); axn.tick_params(axis="x", which="minor", length=1.0, width=0.4)
        axn.set_yticklabels(["10", "100", "1000"])                      # same log scale in all three strips
        if j == 0:
            ax.set_ylabel("RMSE (cm)")
            axn.set_ylabel("n", rotation=0, ha="right", va="center", labelpad=3)
        if j == 1:
            axn.set_xlabel("Observation year")
        src["b"]["years"][r] = dict(
            years_shown=[int(y) for y in q.year], n_cell_years={str(int(y)): int(n) for y, n in zip(q.year, q.n)},
            liu_included={str(int(y)): bool(b) for y, b in zip(q.year, q.liu_included)},
            rmse_cm={k: {str(int(y)): (None if not np.isfinite(v) else round(float(v), 2)) for y, v in zip(q.year, q[f"rmse_{k}"])} for k, *_ in YW_SER},
            ylim_cm=[0, top])




# ================================================================ v13 (2026-10-18 night): a RMSE on common cell-years (as before), b observed against predicted
# ALT in the Lena Delta for our model and the two ML products, c mean bias by observed-ALT class (Alaska, Lena Delta), d year-wise RMSE (as before).
import glob  # noqa: E402

R1C = C.METHOD["R1"]
_wn, _wf, _we, _wm = C.product_style("wei"); _ln, _lf, _le, _lm = C.product_style("liu"); _cn, _cf, _ce, _cm = C.product_style("cci5y"); _c4n, _c4f, _c4e, _c4m = C.product_style("cci4y")
B_SER = [("pred_r1", "Physics + residual ML", R1C, ""), ("prod_wei", "Wei 2026", _wf, ""), ("prod_liu", "Liu 2024", _lf, "")]
C_SER = [("pred_r1", "Physics + residual ML", R1C, R1C, "o", "-"), ("prod_wei", "Wei 2026", _wf, _we, "D", "-"), ("prod_liu", "Liu 2024", _lf, _le, "v", C.DASH),
         ("prod_cci4y", "ESA CCI v4", _c4f, _c4e, "s", "-"), ("prod_cci5y", "ESA CCI v5", _cf, _ce, "s", C.DOT)]
BINS = [0.0, 30.0, 45.0, 60.0, 1000.0]; BIN_LB = ["< 30", "30–45", "45–60", "≥ 60"]
MIN_BIN = 10
LIM_B = (0.0, 120.0)


def num(v, nd=1, plus=False):
    t = f"{abs(v):.{nd}f}"
    if float(t) == 0:
        return t
    return ("−" if v < 0 else ("+" if plus else "")) + t


def common_rows():
    """Row-level table behind panel a (same rule as scripts/2_evaluation/v5_common_cells.py): label cell-years with Wei 2026, Liu 2024 and ESA CCI v4
    values of the observation year, plus the spatial-block out-of-fold prediction of Physics + residual ML (core_v6, seed mean)."""
    import polar.testbed_v5 as T
    base = T.load_v5(ROOT / "data" / "processed" / "v5")
    lk = pd.read_csv(LEAK_CSV); lk = lk[lk["product"] == "wei"].groupby("loc_id").dist_km.min(); base["dist"] = base.loc_id.map(lk)
    fs = sorted(glob.glob(str(ROOT / "data" / "processed" / "v5" / "runs" / "core_v6" / "preds" / "*__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz")))
    p = pd.concat([pd.read_csv(f) for f in fs]).groupby("cell_year_id").pred.mean()
    base["pred_r1"] = base.cell_year_id.map(p)
    c = base[np.isfinite(base.prod_wei) & np.isfinite(base.prod_liu) & np.isfinite(base.prod_cci4y) & np.isfinite(base.alt_cm) & np.isfinite(base.pred_r1)].copy()
    return c, [str(Path(f).relative_to(ROOT)) for f in fs]


def draw_scatter(fig, axes, c, cc, src):
    q = c[c.region == "Lena"]
    vals = {}
    for j, (ax, (col, name, colr, note)) in enumerate(zip(axes, B_SER)):
        C.clean_axes(ax)
        ax.plot(LIM_B, LIM_B, color=C.ZERO, lw=C.LW_REF, zorder=1)
        ax.scatter(q.alt_cm, q[col], s=2.6, color=colr, alpha=0.6, lw=0, zorder=3, rasterized=True)
        e = q[col] - q.alt_cm; rmse = float(np.sqrt(np.mean(e ** 2))); bias = float(np.mean(e)); r = float(np.corrcoef(q.alt_cm, q[col])[0, 1])
        hmm = ax.get_position().height * fig.get_figheight() * 25.4
        ax.text(0.0, 1.0 + 4.4 / hmm, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        ax.text(0.0, 1.0 + 1.6 / hmm, f"r = {r:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS_SMALL, color=C.AUX)
        ax.set_xlim(*LIM_B); ax.set_ylim(*LIM_B); ax.set_xticks([0, 40, 80, 120]); ax.set_yticks([0, 40, 80, 120]); ax.set_aspect("equal")
        if j == 0:
            ax.set_ylabel("Predicted ALT (cm)")
        if j == 1:
            ax.set_xlabel("Observed ALT (cm)")
        vals[name] = dict(rmse_cm=round(rmse, 2), bias_cm=round(bias, 2), r=round(r, 3), n=int(len(q)))
    ref = cc[(cc.region == "Lena") & (cc.mask_km == 0) & (cc.series == "R1 (CatBoost)")]
    src["b"] = dict(definition="observed against predicted ALT on the common cell-years of the Lena Delta (same rows as panel a, all cells); Physics + residual ML = spatial-block "
                               "out-of-fold predictions (seed mean); products = value of the observation year; 1:1 line; Pearson r printed above each panel (RMSE and mean bias are given in the text and recorded here)",
                    n=int(len(q)), axis_range_cm=list(LIM_B), values=vals,
                    check_against_panel_a=dict(rmse_r1_common_cells_eval=round(float(ref.rmse.iloc[0]), 2), n_common_cells_eval=int(ref.n.iloc[0])))


def draw_bias(fig, axes, c, src):
    vals = {}
    for j, (ax, r) in enumerate(zip(axes, ("Alaska", "Lena"))):
        C.clean_axes(ax)
        q = c[c.region == r].copy(); q["bin"] = pd.cut(q.alt_cm, BINS, labels=False, right=False)
        C.zero_line(ax) if hasattr(C, "zero_line") else ax.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=1)
        vals[r] = {}; series_xy = []
        n_bin = q.groupby("bin").size().reindex(range(4)).fillna(0).astype(int)
        for k, (col, name, face, edge, mk, ls) in enumerate(C_SER):
            e = (q[col] - q.alt_cm); ok = np.isfinite(e)
            g = e[ok].groupby(q.bin[ok]).agg(["mean", "size"]).reindex(range(4))
            m = g["mean"].where(g["size"] >= MIN_BIN)
            xs = np.arange(4) + (k - 2) * 0.09
            vals[r][name] = {BIN_LB[i]: (None if not np.isfinite(v) else round(float(v), 2)) for i, v in enumerate(m.values)}
            vals[r][name + " (n)"] = {BIN_LB[i]: int(v) for i, v in enumerate(g["size"].fillna(0).values)}
            series_xy.append((col, name, face, edge, mk, ls, xs, m.values))
        allv = np.array([v for _, _, _, _, _, _, _, m in series_xy for v in m if np.isfinite(v)])
        srt = np.sort(np.abs(allv)); thr = 1.6 * srt[-2] if len(srt) > 1 else np.inf      # one outlying value is drawn at the edge and printed
        inside = allv[np.abs(allv) <= thr]
        lo = min(inside.min(), 0.0); hi = max(inside.max(), 0.0); pad = 0.12 * (hi - lo); ylo, yhi = lo - pad, hi + pad
        ax.set_ylim(ylo, yhi); ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        clipped = []
        for col, name, face, edge, mk, ls, xs, m in series_xy:
            main = col == "pred_r1"; mv = np.array(m, float); over = np.abs(mv) > thr; mc = np.where(over, np.where(mv > 0, yhi, ylo), mv)
            ax.plot(xs, mc, color=face, ls=ls, lw=C.LW_MAIN if main else C.LW_AUX, zorder=4 if main else 3, clip_on=False)
            ax.plot(xs[~over], mv[~over], ls="none", marker=mk, ms=C.MS if mk not in ("D", "v") else C.MS - 0.3, mfc=face, mec="white" if main else edge, mew=0.4,
                    zorder=4.5 if main else 3.5)
            for i in np.where(over)[0]:
                ax.plot([xs[i]], [mc[i]], ls="none", marker=mk, ms=C.MS, mfc="white", mec=edge, mew=0.6, zorder=5, clip_on=False)
                ax.annotate(num(mv[i], plus=True), (xs[i], mc[i]), xytext=(2.5, -3.0), textcoords="offset points", ha="left", va="top", fontsize=C.FS_SMALL,
                            color=C.AUX, annotation_clip=False)
                clipped.append(dict(region=r, series=name, cls=BIN_LB[i], value=round(float(mv[i]), 2)))
        vals[r]["drawn_at_axis_edge"] = clipped
        ax.set_xlim(-0.6, 3.6); ax.set_xticks(range(4)); ax.set_xticklabels(BIN_LB)
        for i in range(4):
            ax.annotate(f"{int(n_bin.iloc[i]):,}", (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -9.5), textcoords="offset points", ha="center", va="top",
                        fontsize=C.FS_SMALL, color=C.AUX, annotation_clip=False)
        hmm = ax.get_position().height * fig.get_figheight() * 25.4
        ax.text(0.0, 1.0 + 1.3 / hmm, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        if j == 0:
            ax.set_ylabel("Bias (cm)")
        vals[r]["cell_years_per_class"] = {BIN_LB[i]: int(n_bin.iloc[i]) for i in range(4)}
    src["c"] = dict(definition="mean bias (predicted minus observed) within classes of observed ALT on the common cell-years (all cells); classes with fewer than %d cell-years "
                               "are not drawn; ESA CCI v5 scored where it has a value; numbers under the axis = cell-years per class" % MIN_BIN,
                    classes_cm=BINS, values=vals)


def main():
    C.setup()
    cc = pd.read_csv(CC_CSV); yw = pd.read_csv(YW_CSV)
    c, pred_files = common_rows()
    for r in REG:                                                   # consistency with panel a (same rule, same predictions)
        n_a = int(cc[(cc.region == r) & (cc.mask_km == 0) & (cc.series == "R1 (CatBoost)")].n.iloc[0]); n_c = int((c.region == r).sum())
        print(f"common cell-years {r}: panel a {n_a}, recomputed {n_c}")
        assert n_a == n_c, (r, n_a, n_c)
    src = dict(figure="Fig. 5", script="scripts/4_visualization/paper_v13/fig5_products.py",
               inputs=dict(common_cells=str(CC_CSV.relative_to(ROOT)), yearwise=str(YW_CSV.relative_to(ROOT)), predictions=pred_files,
                           training_site_distance=str(LEAK_CSV.relative_to(ROOT)) + " (product == 'wei', nearest site per label cell)"),
               a=dict(definition="RMSE on common cell-years: label cell-years where Wei 2026, Liu 2024 and ESA CCI v4 all have a value of the observation "
                                 "year; ESA CCI v5 and Aalto 2018 are scored on the common cell-years where they have a value. Our methods: block "
                                 "cross-validation out-of-fold predictions (core_v6, climate input to the observation day, covariate set F8TS; Physics "
                                 "+ residual ML = CatBoost lambda 0.5, Direct ML = CatBoost, seeds averaged).",
                      filled="all common cell-years", open="cells more than 25 km from the nearest Wei 2026 training site", x_range_cm=list(X_A),
                      values={}, n_label={}),
               d=dict(definition="per observation year: labelled cell-years where Physics + residual ML, Stefan (recalibrated) and Wei 2026 all have "
                                 "values; Liu 2024 (to 2020) is included when it leaves >= 30 cell-years, otherwise the year is shown without Liu; "
                                 "years with < 30 cell-years are not shown. All cell-years (no distance mask).", min_cell_years=MIN_N_YEAR, years={}))
    src["a"]["common_cell_years"] = common_years()
    W, H = 180.0, 146.0
    fig = C.figure(W, H)
    # ---------------- a
    xa0, gap_a, xr = 27.5, 11.0, 9.5
    wa = (W - xa0 - 2 * gap_a - xr) / 3.0; ya, ha = 9.0, 30.0
    ws = 8.0; gs_ = 1.6                                                 # compressed right segment (broken axis) per region
    axes_a = [C.ax_mm(fig, xa0 + j * (wa + gap_a), ya, wa - ws - gs_, ha) for j in range(3)]
    sides_a = [C.ax_mm(fig, xa0 + j * (wa + gap_a) + wa - ws, ya, ws, ha) for j in range(3)]
    draw_a(fig, axes_a, sides_a, cc, src)
    C.letter(fig, 0.6, 0.6, "a")
    hs = [Line2D([], [], ls="none", marker="o", ms=MS_A, mfc=C.DARK, mec=C.DARK, mew=0.6),
          Line2D([], [], ls="none", marker="o", ms=MS_A, mfc="white", mec=C.DARK, mew=0.6)]
    fig.legend(hs, ["All common site-years", "Sites > 25 km from product training sites"], loc="upper left",
               bbox_to_anchor=(xa0 / W, 1 - 0.9 / H), ncol=2, handlelength=0.9, handletextpad=0.4, columnspacing=1.6, fontsize=C.FS_TICK)
    # ---------------- b: observed vs predicted (Lena Delta), three squares; titles and statistics above the squares
    yb0 = 50.0; sq = 26.0; yb_ax = yb0 + 7.0; xb = [11.0, 41.0, 71.0]
    axes_b = [C.ax_mm(fig, x, yb_ax, sq, sq) for x in xb]
    draw_scatter(fig, axes_b, c, cc, src)
    C.letter(fig, 0.6, yb0 + 0.3, "b")
    # ---------------- c: bias by observed-ALT class (Alaska, Lena Delta)
    xc = [110.0, 148.0]; wc = 30.0
    axes_c = [C.ax_mm(fig, x, yb_ax, wc, sq) for x in xc]
    draw_bias(fig, axes_c, c, src)
    C.letter(fig, xc[0] - 10.0, yb0 + 0.3, "c")
    C.text_mm(fig, xc[0] + wc + 4.0, yb_ax + sq + 4.0 + 2.2, "Observed ALT (cm)", ha="center", va="top")
    # ---------------- one key for b-d (methods and products share colours and markers in b, c and d)
    key = [Line2D([], [], color=f, ls=ls, lw=lw, marker=mk, ms=ms, mfc=f, mec=("white" if k in ("r1", "p1") else e), mew=0.4)
           for k, n, f, e, mk, ls, lw, ms in YW_SER]
    labs = [n for _, n, *_ in YW_SER]
    for col, n, f, e, mk, ls in C_SER[3:]:
        key.append(Line2D([], [], color=f, ls=ls, lw=C.LW_AUX, marker=mk, ms=C.MS_SQ if hasattr(C, "MS_SQ") else C.MS - 0.3, mfc=f, mec=e, mew=0.4)); labs.append(n)
    key.append(plt.Rectangle((0, 0), 1, 1, facecolor=NBAR, edgecolor="none")); labs.append("Site-years (n)")
    y_key = yb_ax + sq + 9.2
    fig.legend(key, labs, loc="upper left", bbox_to_anchor=(4.0 / W, 1 - y_key / H), ncol=7, handlelength=1.8, handletextpad=0.4,
               columnspacing=1.1, fontsize=C.FS_TICK, borderaxespad=0, borderpad=0)                    # one row, keys b, c and d
    # ---------------- d: year-wise RMSE
    yd0 = y_key + 5.5
    xb0, gap_b, xbr = 12.0, 10.0, 2.0
    wb = (W - xb0 - 2 * gap_b - xbr) / 3.0; yb, hb, hs_, gs = yd0 + 4.0, 23.5, 8.5, 1.3
    axes_d = [C.ax_mm(fig, xb0 + j * (wb + gap_b), yb, wb, hb) for j in range(3)]
    strips = [C.ax_mm(fig, xb0 + j * (wb + gap_b), yb + hb + gs, wb, hs_) for j in range(3)]
    src_d = {"b": src.pop("d")}; src_d["b"]["years"] = {}
    draw_b(fig, axes_d, strips, yw, src_d)
    src["d"] = src_d["b"]
    C.letter(fig, 0.6, yd0 - 1.2, "d")
    rep = C.qa_figure(fig, ignore_ink_for=["+77"], allow_overlap=())
    C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN)
    plt.close(fig)
    cy = src["a"]["common_cell_years"]
    src["years_used"] = dict(
        a={r: dict(all=f"{cy[r]['0']['years'][0]}-{cy[r]['0']['years'][1]}", gt25km=f"{cy[r]['25']['years'][0]}-{cy[r]['25']['years'][1]}") for r in REG},
        d={r: dict(years_shown=src["d"]["years"][r]["years_shown"], liu_included=[int(y) for y, v in src["d"]["years"][r]["liu_included"].items() if v]) for r in REG})
    src["outputs"] = C.file_info(paths); src["canvas_mm"] = [W, H]; src["pdf_page_mm"] = C.pdf_page_mm(paths[0]); src["pdf_fonts"] = C.pdf_fonts(paths[0])
    src["qa"] = rep
    src["style"] = dict(font="FreeSans 7 pt labels, 6.5 pt ticks and notes, panel letters 8 pt bold; text black, ticks #333333, notes #595959; axes 0.6 pt",
                        colours="design/style_tokens_v4.json color.paper_v13 (POLAR_PALETTE=paper13)", key="one key for panels b-d below panel c",
                        markers={n: mk for _, n, _, _, mk in ROWS}, series_colours={n: f for _, n, f, _, _ in ROWS}, n_bars=NBAR)
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
