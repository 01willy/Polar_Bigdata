"""Supplementary Fig. S6 (v11 numbering; S4 in v10): product-anchored residual learning (content of v9_f10_extra.py PA), v11 style.
Copied from scripts/4_visualization/paper_v10/si_s04_product_anchor.py and edited (v11 colours, dark-grey text, output v11_si).

RMSE on the same cell-years (label cell-years where Wei 2026 has a value of the observation year) for: Wei 2026 as published, Wei 2026 recalibrated
(linear, training folds), Wei 2026 + residual ML (residual learner on the product anchor), Stefan (recalibrated) and Physics + residual ML (Stefan anchor).
Spatial-block CV (5 folds, 3 seeds). One panel per region, common x axis.
Inputs: data/processed/v5/runs/extra_v9/pa_metrics.csv (product == 'wei'), preds_PA_<region>_wei.csv.gz (years, n).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper11 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v11/si_s06_product_anchor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c11_common as C  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STEM = "FigS6_product_anchor"
RUN = C.ROOT / "data" / "processed" / "v5" / "runs" / "extra_v9"
REG = C.REGIONS
WEI = C.PRODUCT["wei"][1]
ROWS = [("Praw", "Wei 2026", WEI, WEI, "D", True), ("Pcal", "Wei 2026, recalibrated", WEI, WEI, "D", False),
        ("RP", "Wei 2026 + residual ML", WEI, WEI, "o", True),
        ("P1", C.METHOD_NAME["P1"], C.METHOD["P1"], C.METHOD["P1"], "s", True), ("R1", C.METHOD_NAME["R1"], C.METHOD["R1"], C.METHOD["R1"], "o", True)]
X = (15.0, 32.0)


def main():
    C.setup()
    m = pd.read_csv(RUN / "pa_metrics.csv"); m = m[m["product"] == "wei"]
    t = m.pivot_table(index="region", columns="method", values="rmse"); nn = m.groupby("region").n.first()
    yrs = {}
    for r in REG:
        p = pd.read_csv(RUN / f"preds_PA_{r}_wei.csv.gz", usecols=["cell_year_id", "year"]).drop_duplicates("cell_year_id")
        yrs[r] = dict(years=[int(p.year.min()), int(p.year.max())], n_cell_years=int(len(p)))
    W, H = 180.0, 47.0
    fig = C.figure(W, H)
    x0, gap, xr = 36.0, 7.0, 3.0; w = (W - x0 - 2 * gap - xr) / 3.0
    axes = [C.ax_mm(fig, x0 + j * (w + gap), 5.0, w, 31.0) for j in range(3)]
    ypos = [0, 1, 2, 3.35, 4.35]; vals = {}
    for j, (ax, r) in enumerate(zip(axes, REG)):
        C.clean_axes(ax, left=(j == 0))
        for y in ypos:
            ax.axhline(y, color=C.GRID, lw=C.LW_THIN, zorder=0)
        ax.axhline(2.675, color=C.ZERO, lw=C.LW_THIN, zorder=0.5)
        for y, (k, name, face, edge, mk, filled) in zip(ypos, ROWS):
            v = float(t.loc[r, k])
            ax.plot([v], [y], ls="none", marker=mk, ms=3.3 if mk != "D" else 3.0, mfc=face if filled else "white", mec=C.darker(edge, 0.9), mew=0.6,
                    zorder=3)
            vals.setdefault(r, {})[name] = round(v, 3)
        ax.set_xlim(*X); ax.set_ylim(ypos[-1] + 0.6, -0.6); ax.set_yticks(ypos)
        ax.set_yticklabels([n for _, n, *_ in ROWS] if j == 0 else []); ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xticks([15, 20, 25, 30])
        ax.text(0.0, 1.0, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        ax.text(1.0, 1.0, f"n = {int(nn[r]):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT)
        if j == 1:
            ax.set_xlabel("RMSE on the same cell-years (cm)")
    rep = C.qa_figure(fig); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S6", dict(
        title="Product-anchored residual learning", stem=STEM, script="scripts/4_visualization/paper_v11/si_s06_product_anchor.py",
        v10_number="S4 (scripts/4_visualization/paper_v10/si_s04_product_anchor.py)",
        inputs=["data/processed/v5/runs/extra_v9/pa_metrics.csv (product == 'wei')", "data/processed/v5/runs/extra_v9/preds_PA_<region>_wei.csv.gz"],
        years=yrs, definition="label cell-years where Wei 2026 has a value of the observation year; spatial-block CV; Wei 2026, recalibrated = linear "
                              "recalibration fitted on the training folds; + residual ML = CatBoost residual learner (lambda 0.5) on the anchor",
        values_rmse_cm=vals, numbers_printed={r: f"n = {int(nn[r]):,}" for r in REG},
        colours=dict(wei_rows=WEI, stefan=C.METHOD["P1"], physics_residual_ml=C.METHOD["R1"]),
        legend_draft="Fig. S6 | Product-anchored residual learning. RMSE on the same cell-years (label cell-years with a Wei 2026 value of the "
                     "observation year; spatial-block cross-validation) for Wei 2026 as published, recalibrated, and with a residual learner, "
                     "compared with Stefan (recalibrated) and Physics + residual ML, per region.",
        outputs=C.file_info(paths), size_mm=[W, H], pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
