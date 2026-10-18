"""Fig. 5 (v10 manuscript): comparison with published ALT products on the same cells and years.

a  RMSE on common cell-years (cell-years where Wei 2026, Liu 2024 and ESA CCI v4 all have a value of the observation year; 2010-2020).
   Rows: Stefan (recalibrated), Physics + residual ML, Direct ML | Wei 2026, Liu 2024, ESA CCI v4, ESA CCI v5, Aalto 2018. One panel per region.
   Filled marker = all common cell-years, open marker = cells > 25 km from product (Wei 2026) training sites. Values beyond 45 cm are drawn at the
   axis end and printed. ESA CCI v5 and Aalto 2018 are scored on the common cell-years where they have a value.
b  RMSE per observation year (years with >= 30 cell-years) for Physics + residual ML, Stefan (recalibrated), Wei 2026 and Liu 2024 (to 2020),
   same cell-years within each year; grey bars = cell-years per year (log scale).
Inputs: data/processed/v5/runs/core_v6/common_cells_eval.csv (v5_common_cells.py --run core_v6 --forcing year_obs --ens none),
        outputs/figures/paper/v7/source_data/F5_yearwise_rmse.csv (v7_f5_products_years.yearwise_rmse; read only),
        data/processed/v5 labels/products via polar.testbed_v5.load_v5 (year coverage of the common cell-years, for the source-data JSON).
Outputs: outputs/figures/paper/v10/Fig5_products.{pdf,svg,png}, outputs/figures/paper/v10/source_data/Fig5_products.json
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/fig5_products.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
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
# rows of panel a: (series code in common_cells_eval.csv, label, colour, edge, marker)
ROWS = [("P1 (Stefan, recal.)", C.METHOD_NAME["P1"], C.METHOD["P1"], C.METHOD["P1"], "s"),
        ("R1 (CatBoost)", C.METHOD_NAME["R1"], C.METHOD["R1"], C.METHOD["R1"], "o"),
        ("D0 (CatBoost)", C.METHOD_NAME["D0"], C.METHOD["D0"], C.METHOD["D0"], "o")]
for _code, _key in (("Wei 2026", "wei"), ("Liu 2024", "liu"), ("ESA CCI v4", "cci4y"), ("ESA CCI v5", "cci5y"), ("Aalto 2018", "aalto")):
    _n, _f, _e, _m = C.product_style(_key)
    ROWS.append((_code, _n, _f, _e, _m))
N_OURS = 3
X_A = (10.0, 45.0)                       # a: x range (cm); larger values drawn at the axis end and printed
GAP_ROWS = 0.35                          # extra space (rows) at the rule between our methods and products
DY = 0.2                                 # a: sub-row offset (filled above, open below)
MIN_N_YEAR = 30
YW_SER = [("r1", C.METHOD_NAME["R1"], C.METHOD["R1"], C.METHOD["R1"], "o", "-", C.LW_MAIN, 3.0),
          ("p1", C.METHOD_NAME["P1"], C.METHOD["P1"], C.METHOD["P1"], "s", "-", C.LW_MAIN, 2.6)]
for _k, _ls in (("wei", "-"), ("liu", C.DASH)):
    _n, _f, _e, _m = C.product_style(_k)
    YW_SER.append((_k, _n, _f, _e, _m, _ls, C.LW_AUX, 3.0 if _m != "D" else 2.7))


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


def draw_a(fig, axes, cc, src):
    ny = len(ROWS); ylim = (row_y(ny - 1) + 0.62, -0.62)
    printed = []
    for j, (ax, r) in enumerate(zip(axes, REG)):
        C.clean_axes(ax, left=(j == 0))
        ax.set_xlim(*X_A); ax.set_ylim(*ylim)
        for i in range(ny):
            ax.axhline(row_y(i), color="#EDEDED", lw=C.LW_THIN, zorder=0)
        ax.axhline((row_y(N_OURS - 1) + row_y(N_OURS)) / 2, color="#8C8C8C", lw=C.LW_THIN, zorder=0.5)
        vals = {}
        for i, (code, name, face, edge, mk) in enumerate(ROWS):
            over = []
            for mask, dy, filled in ((0, -DY, True), (25, DY, False)):
                q = cc[(cc.region == r) & (cc.mask_km == mask) & (cc.series == code)]
                if not len(q):
                    continue
                v = float(q.rmse.iloc[0]); n = int(q.n.iloc[0]); vals[f"{name}|{mask}"] = dict(rmse_cm=round(v, 3), n=n)
                x = min(v, X_A[1]); y = row_y(i) + dy
                ms = 3.5 if mk != "D" else 3.1
                ax.plot([x], [y], ls="none", marker=mk, ms=ms, mfc=face if filled else "white", mec=edge, mew=0.6, zorder=3, clip_on=False)
                if v > X_A[1]:
                    over.append((mask, v))
                    printed.append(dict(region=r, series=name, subset="all" if mask == 0 else "> 25 km", rmse_cm=round(v, 1)))
            if over:                               # values beyond the axis: markers at the axis end, numbers 'all / > 25 km' right of them (row centre)
                txt = " / ".join(f"{v:.0f}" for _, v in over) if len(over) == 2 else f"{over[0][1]:.0f}"
                yy = row_y(i) if len(over) == 2 else row_y(i) + (-DY if over[0][0] == 0 else DY)
                ax.annotate(txt, (X_A[1], yy), xytext=(3.5 / 2 + 1.8, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=C.FS_SMALL, color=C.TEXT, annotation_clip=False)
        ax.set_yticks([row_y(i) for i in range(ny)])
        ax.set_yticklabels([n for _, n, *_ in ROWS] if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=2.0)
        ax.set_xticks([10, 20, 30, 40])
        n0 = int(cc[(cc.region == r) & (cc.mask_km == 0) & (cc.series == "R1 (CatBoost)")].n.iloc[0])
        n25 = int(cc[(cc.region == r) & (cc.mask_km == 25) & (cc.series == "R1 (CatBoost)")].n.iloc[0])
        ax.text(0.0, 1.0, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        ax.text(1.0, 1.0, f"n = {n0:,} / {n25:,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT)
        if j == 1:
            ax.set_xlabel("RMSE on common cell-years (cm)")
        src["a"]["values"][r] = vals; src["a"]["n_label"][r] = dict(all=n0, gt25km=n25)
    src["a"]["printed_beyond_axis"] = printed


def draw_b(fig, axes, strips, yw, src):
    q0 = yw[(yw.mask_km == 0) & yw.shown]
    NL = (8.0, 1.2e4)
    for j, (ax, axn, r) in enumerate(zip(axes, strips, REG)):
        C.clean_axes(ax); C.clean_axes(axn)
        q = q0[q0.region == r].sort_values("year")
        for key, name, face, edge, mk, ls, lw, ms in YW_SER[::-1]:
            z = q[np.isfinite(q[f"rmse_{key}"])]
            yrs, v = z.year.values, z[f"rmse_{key}"].values
            seg = np.split(np.arange(len(yrs)), np.where(np.diff(yrs) > 1)[0] + 1) if len(yrs) else []
            main = key in ("r1", "p1")
            for sg in seg:
                if len(sg) > 1:
                    ax.plot(yrs[sg], v[sg], ls=ls, color=face if main else edge, lw=lw, zorder=3.5 if key == "r1" else (3.2 if key == "p1" else 3))
            ax.plot(yrs, v, ls="none", marker=mk, ms=ms, mfc=face, mec=edge if not main else "white", mew=0.4 if main else 0.5,
                    zorder=4.5 if key == "r1" else (4.2 if key == "p1" else 4))
        vmax = np.nanmax(q[[f"rmse_{k}" for k, *_ in YW_SER]].values)
        top = float(np.ceil(1.08 * vmax / 10.0) * 10.0)
        ax.set_xlim(2009.3, 2024.7); ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20 if top > 60 else 10))
        ax.set_xticks([2010, 2014, 2018, 2022]); ax.tick_params(axis="x", labelbottom=False)
        hb_mm = ax.get_position().height * fig.get_figheight() * 25.4
        ax.text(0.0, 1.0 + 1.5 / hb_mm, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)   # clear of the top y tick label
        axn.bar(q.year, q.n, width=0.62, color="#CFCFCF", lw=0, zorder=2)
        axn.set_yscale("log"); axn.set_ylim(*NL); axn.set_xlim(*ax.get_xlim())
        axn.set_xticks([2010, 2014, 2018, 2022]); axn.set_yticks([10, 100, 1000]); axn.yaxis.set_minor_locator(mticker.NullLocator())
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


def main():
    C.setup()
    cc = pd.read_csv(CC_CSV); yw = pd.read_csv(YW_CSV)
    src = dict(figure="Fig. 5", script="scripts/4_visualization/paper_v10/fig5_products.py",
               inputs=dict(common_cells=str(CC_CSV.relative_to(ROOT)), yearwise=str(YW_CSV.relative_to(ROOT)),
                           training_site_distance=str(LEAK_CSV.relative_to(ROOT)) + " (product == 'wei', nearest site per label cell)"),
               a=dict(definition="RMSE on common cell-years: label cell-years where Wei 2026, Liu 2024 and ESA CCI v4 all have a value of the observation "
                                 "year; ESA CCI v5 and Aalto 2018 are scored on the common cell-years where they have a value. Our methods: block "
                                 "cross-validation out-of-fold predictions (core_v6, climate input to the observation day, covariate set F8TS; Physics "
                                 "+ residual ML = CatBoost lambda 0.5, Direct ML = CatBoost, seeds averaged).",
                      filled="all common cell-years", open="cells more than 25 km from the nearest Wei 2026 training site", x_range_cm=list(X_A),
                      values={}, n_label={}),
               b=dict(definition="per observation year: labelled cell-years where Physics + residual ML, Stefan (recalibrated) and Wei 2026 all have "
                                 "values; Liu 2024 (to 2020) is included when it leaves >= 30 cell-years, otherwise the year is shown without Liu; "
                                 "years with < 30 cell-years are not shown. All cell-years (no distance mask).", min_cell_years=MIN_N_YEAR, years={}))
    src["a"]["common_cell_years"] = common_years()
    W, H = 180.0, 108.0
    fig = C.figure(W, H)
    # ---------------- a
    xa0, gap_a, xr = 27.5, 11.0, 9.5
    wa = (W - xa0 - 2 * gap_a - xr) / 3.0; ya, ha = 9.0, 35.0
    axes_a = [C.ax_mm(fig, xa0 + j * (wa + gap_a), ya, wa, ha) for j in range(3)]
    draw_a(fig, axes_a, cc, src)
    C.letter(fig, 0.6, 0.6, "a")
    hs = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc="black", mec="black", mew=0.6),
          Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec="black", mew=0.6)]
    fig.legend(hs, ["All common cell-years", "Cells > 25 km from product training sites"], loc="upper left",
               bbox_to_anchor=(xa0 / W, 1 - 0.9 / H), ncol=2, handlelength=0.9, handletextpad=0.4, columnspacing=1.6, fontsize=C.FS_TICK)
    # ---------------- b
    yb0 = 56.5
    xb0, gap_b, xbr = 12.0, 10.0, 2.0
    wb = (W - xb0 - 2 * gap_b - xbr) / 3.0; yb, hb, hs_, gs = yb0 + 9.0, 24.5, 9.0, 1.3
    axes_b = [C.ax_mm(fig, xb0 + j * (wb + gap_b), yb, wb, hb) for j in range(3)]
    strips = [C.ax_mm(fig, xb0 + j * (wb + gap_b), yb + hb + gs, wb, hs_) for j in range(3)]
    draw_b(fig, axes_b, strips, yw, src)
    C.letter(fig, 0.6, yb0 + 0.6, "b")
    hb_ = [Line2D([], [], color=(f if k in ("r1", "p1") else e), ls=ls, lw=lw, marker=mk, ms=ms, mfc=f, mec=("white" if k in ("r1", "p1") else e),
                  mew=0.4 if k in ("r1", "p1") else 0.5) for k, n, f, e, mk, ls, lw, ms in YW_SER]
    fig.legend(hb_, [n for _, n, *_ in YW_SER], loc="upper left", bbox_to_anchor=(xb0 / W, 1 - (yb0 + 0.9) / H), ncol=4, handlelength=2.0,
               handletextpad=0.5, columnspacing=1.8, fontsize=C.FS_TICK)
    rep = C.qa_figure(fig)
    C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN)
    plt.close(fig)
    cy = src["a"]["common_cell_years"]
    src["years_used"] = dict(
        a={r: dict(all=f"{cy[r]['0']['years'][0]}-{cy[r]['0']['years'][1]}", gt25km=f"{cy[r]['25']['years'][0]}-{cy[r]['25']['years'][1]}") for r in REG},
        b={r: dict(years_shown=src["b"]["years"][r]["years_shown"], liu_included=[int(y) for y, v in src["b"]["years"][r]["liu_included"].items() if v]) for r in REG})
    src["outputs"] = C.file_info(paths); src["canvas_mm"] = [W, H]; src["pdf_page_mm"] = C.pdf_page_mm(paths[0]); src["qa"] = rep
    src["style"] = dict(font="FreeSans 7 pt (ticks 6.5 pt, small numbers 6 pt), panel letters 8 pt bold", colours="paper_v10 tokens",
                        markers={n: mk for _, n, _, _, mk in ROWS})
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
