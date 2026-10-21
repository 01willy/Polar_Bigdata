"""Supplementary Fig. S5 (v15, 2026-10-21): (a) effect of the year-specific climate input (block RMSE change, year_obs minus 2011-2024 mean input) for
the recalibrated Stefan model, the physics-residual model and direct ML; (b) year-wise RMSE (years with >= 30 site-years) of the physics-residual
model, the recalibrated Stefan model, Wei et al. and Liu et al. Reuses _f3_v14_parts.data_ab (a) and fig5_products.draw_b (b)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import _f3_v14_parts as F3  # noqa: E402
import fig5_products as F5  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STEM = "FigS5_yearwise_input"; REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}; MN = C.METHOD_NAME; MC = C.METHOD


def main():
    C.setup(); W, H = 180.0, 74.0; fig = C.figure(W, H)
    t, dl = F3.data_ab()
    # ---------------- a: forest, rows = regions, series = P1 / R1 / D0
    ax = C.ax_mm(fig, 24.0, 8.0, 44.0, 36.0); C.clean_axes(ax)
    for i, r in enumerate(REGIONS): ax.axhline(i, color="#EDEDED", lw=0.4, zorder=0)
    vals = {}
    for m, dy, filled in (("P1", -0.25, True), ("R1", 0.0, True), ("D0", 0.25, False)):
        for i, r in enumerate(REGIONS):
            z = dl[(dl.region == r) & (dl.method == m)].iloc[0]; c = MC[m]; y = i + dy
            ax.plot([z.lo, z.hi], [y, y], color=c, lw=C.LW_AUX, zorder=2); ax.plot(z.delta, y, marker=("s" if m == "P1" else "o"), ms=C.MS, mfc=(c if filled else "white"), mec=c, mew=0.7, ls="none", zorder=3)
            vals[f"{r}_{m}"] = [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)]
    ax.axvline(0, color=C.ZERO, lw=C.LW_REF, zorder=1); ax.set_yticks(range(3)); ax.set_yticklabels([RN[r] for r in REGIONS]); ax.set_ylim(2.5, -0.5)
    ax.set_xlabel("ΔRMSE, year-specific − mean climate input (cm)")
    hs = [Line2D([], [], ls="none", marker="s", ms=C.MS, color=MC["P1"]), Line2D([], [], ls="none", marker="o", ms=C.MS, color=MC["R1"]), Line2D([], [], ls="none", marker="o", ms=C.MS, mfc="white", mec=MC["D0"], mew=0.7)]
    fig.legend(hs, [MN["P1"], MN["R1"], MN["D0"]], loc="upper left", bbox_to_anchor=(24.0 / W, 1 - 58.0 / H), ncol=3, fontsize=C.FS_SMALL, frameon=False, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.0)
    C.letter(fig, 2.0, 3.0, "a")
    # ---------------- b: year-wise RMSE (fig5_products panel d)
    yw = pd.read_csv(F5.YW_CSV)
    xb0, gap_b, xbr = 84.0, 9.0, 2.0; wb = (W - xb0 - 2 * gap_b - xbr) / 3.0; yb, hb, hs_, gs = 8.0, 24.0, 8.0, 1.3
    axes_d = [C.ax_mm(fig, xb0 + j * (wb + gap_b), yb, wb, hb) for j in range(3)]
    strips = [C.ax_mm(fig, xb0 + j * (wb + gap_b), yb + hb + gs, wb, hs_) for j in range(3)]
    src_d = {"b": dict(definition="per observation year: labelled site-years where the physics-residual model, the recalibrated Stefan model and Wei et al. all have values; Liu et al. to 2020", min_cell_years=F5.MIN_N_YEAR, years={})}
    F5.draw_b(fig, axes_d, strips, yw, src_d)
    C.letter(fig, xb0 - 12.0, 3.0, "b")
    key = [Line2D([], [], color=f, ls=ls, lw=lw, marker=mk, ms=ms, mfc=f, mec=("white" if k in ("r1", "p1") else e), mew=0.4) for k, n, f, e, mk, ls, lw, ms in F5.YW_SER]
    labs = [n for _, n, *_ in F5.YW_SER]
    key.append(plt.Rectangle((0, 0), 1, 1, facecolor=F5.NBAR, edgecolor="none")); labs.append("Site-years (n)")
    fig.legend(key, labs, loc="upper left", bbox_to_anchor=(24.0 / W, 1 - 64.0 / H), ncol=5, handlelength=1.8, handletextpad=0.4, columnspacing=1.1, fontsize=C.FS_SMALL, frameon=False, borderaxespad=0.0)
    rep = C.qa_figure(fig, ignore_ink_for=[], allow_overlap=()); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S5", dict(figure="Supplementary Fig. S5", script="scripts/4_visualization/paper_v15/si_s05_yearwise_input.py",
                                a=dict(definition="block RMSE(year-specific input to the observation day) - RMSE(2011-2024 mean input), 95 % paired block-bootstrap interval; core_v6/deltas_forcing.csv", values=vals),
                                b=src_d["b"], outputs=C.file_info(paths), canvas_mm=[W, H], qa=rep))
    for p in paths: print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
