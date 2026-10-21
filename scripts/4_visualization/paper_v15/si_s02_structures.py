"""Supplementary Fig. S2 (v11): model structures and learners (content of deck R03b / v7 F4 e-f).

Copied from paper_v10/si_s02_structures.py (data unchanged) and restyled for v11 (si13_common.py, POLAR_PALETTE=paper13).
a  Delta RMSE (method - Stefan (recalibrated), same cell-years, spatial-block CV, climate input to the observation day, covariates F8TS, CatBoost)
   for four model structures: Direct ML, Physics-input ML, Physics pseudo-labels, Physics + residual ML; points = regions (marker shape), bars = 95 %
   block-bootstrap intervals. Colour = model structure (tokens color.paper_v13.methods).
b  Same Delta RMSE for five learners (CatBoost, RF, MLP, TabM, FT-T): Physics + residual ML (filled) and Direct ML (open), v11 method colours.
Inputs: data/processed/v5/runs/{core_v6,rf_v6,nn_v6}/deltas_vs_ref.csv via v7_f4_results_core.load() (read only).
Output: outputs/figures/paper/v15_si/FigS2_structures.{pdf,png}, entry S2 of outputs/figures/paper/v15_si/SI_figures.json.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v15/si_s02_structures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import si13_common as C  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import v7_f4_results_core as F4  # noqa: E402

STEM = "FigS2_structures"
REG = C.REGIONS
STRUCT = ["D0", "F1", "D1", "R1"]
STRUCT_LB = {"D0": "Direct ML", "F1": "Physics-\ninput ML", "D1": "Physics\npseudo-labels", "R1": "Physics +\nresidual ML"}
LEARN = ["catboost", "rf", "mlp", "tabm", "ftt"]; LEARN_LB = ["CatBoost", "RF", "MLP", "TabM", "FT-T"]
OFF = {"Alaska": -0.24, "Lena": 0.0, "Canada": 0.24}
YLAB = "ΔRMSE vs recalibrated Stefan (cm)"


def point(ax, x, d, lo, hi, col, r, filled, lw=0.8):
    mk = C.REGION[r][2]; ms = C.REGION_MS[r]
    ax.vlines(x, lo, hi, color=col, lw=lw, zorder=2)
    ax.plot([x], [d], ls="none", marker=mk, ms=ms, mfc=col if filled else C.WHITE, mec=col if not filled else C.WHITE, mew=C.MEW if not filled else 0.35,
            zorder=3)


def main():
    C.setup()
    df, sm, ct, lad, st, lr = F4.load()
    W, H = 180.0, 66.0
    fig = C.figure(W, H)
    axa = C.ax_mm(fig, 14.0, 8.0, 62.0, 46.0); axb = C.ax_mm(fig, 92.0, 8.0, 86.0, 46.0)
    vals = {"a": {}, "b": {}}
    for i, m in enumerate(STRUCT):
        for r in REG:
            q = st[(st.region == r) & (st.method == m)]
            if not len(q):
                continue
            d, lo, hi = float(q.delta.iloc[0]), float(q.lo.iloc[0]), float(q.hi.iloc[0])
            point(axa, i + OFF[r], d, lo, hi, C.METHOD[m], r, True)
            vals["a"].setdefault(C.METHOD_NAME[m], {})[r] = dict(delta_cm=round(d, 3), ci95=[round(lo, 3), round(hi, 3)], n=int(q.n.iloc[0]))
    for i, l in enumerate(LEARN):
        for r in REG:
            for m, sub, filled in (("R1", -0.07, True), ("D0", 0.07, False)):
                q = lr[(lr.region == r) & (lr.method == m) & (lr.learner == l)]
                if not len(q):
                    continue
                d, lo, hi = float(q.delta.iloc[0]), float(q.lo.iloc[0]), float(q.hi.iloc[0])
                point(axb, i + OFF[r] + sub, d, lo, hi, C.METHOD[m], r, filled, lw=0.8 if m == "R1" else 0.6)
                vals["b"].setdefault(LEARN_LB[i], {}).setdefault(C.METHOD_NAME[m], {})[r] = dict(delta_cm=round(d, 3), ci95=[round(lo, 3), round(hi, 3)])
    ylo = min(st.lo.min(), lr.lo.min()); yhi = max(st.hi.max(), lr.hi.max())
    for ax, n, lbs in ((axa, len(STRUCT), [STRUCT_LB[m] for m in STRUCT]), (axb, len(LEARN), LEARN_LB)):
        C.clean_axes(ax); C.grid(ax, "y")
        ax.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=1)
        ax.set_xlim(-0.6, n - 0.4); ax.set_xticks(range(n)); ax.set_xticklabels(lbs); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_ylim(np.floor(ylo) - 0.5, np.ceil(yhi) + 0.5); ax.yaxis.set_major_locator(mticker.MultipleLocator(4))
    axa.set_ylabel(YLAB)
    axa.set_xlabel("Model structure"); axb.set_xlabel("Learner")
    for ax in (axa, axb):                                    # same baseline for both x labels (a has two-line tick labels)
        ax.xaxis.set_label_coords(0.5, -0.175)
    C.letter(fig, 0.5, 0.5, "a"); C.letter(fig, 80.0, 0.5, "b")
    hr = [Line2D([], [], ls="none", marker=C.REGION[r][2], ms=C.REGION_MS[r], mfc=C.TEXT, mec=C.TEXT, mew=0) for r in REG]
    fig.legend(hr, [C.REGION[r][0] for r in REG], loc="upper left", bbox_to_anchor=(16.0 / W, 1 - 1.0 / H), ncol=3, handlelength=0.8,
               columnspacing=1.4, fontsize=C.FS_TICK)
    hm = [Line2D([], [], ls="none", marker="o", ms=C.MS, mfc=C.METHOD["R1"], mec=C.WHITE, mew=0.35),
          Line2D([], [], ls="none", marker="o", ms=C.MS, mfc=C.WHITE, mec=C.METHOD["D0"], mew=C.MEW)]
    fig.legend(hm, [C.METHOD_NAME["R1"], C.METHOD_NAME["D0"]], loc="upper left", bbox_to_anchor=(94.0 / W, 1 - 1.0 / H), ncol=2,
               handlelength=0.8, columnspacing=1.4, fontsize=C.FS_TICK)
    rep = C.qa_figure(fig); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S2", dict(
        title="Model structures and learners", stem=STEM, script="scripts/4_visualization/paper_v15/si_s02_structures.py",
        version="v11 (2026-10-16): v11 palette restyle of v10 FigS2_structures; numbers unchanged",
        inputs=["data/processed/v5/runs/core_v6/deltas_vs_ref.csv", "data/processed/v5/runs/rf_v6/deltas_vs_ref.csv", "data/processed/v5/runs/nn_v6/deltas_vs_ref.csv"],
        years="all label cell-years: Alaska 2010-2024, Lena Delta 2010-2022, Canada (NW) 2010-2024 (labels_cell_year_v5, in_forcing == 1)",
        definition="Delta RMSE = method - Stefan (recalibrated) on the same cell-years, spatial-block CV, climate input to the observation day, covariates F8TS; "
                   "a: CatBoost; b: Physics + residual ML (lambda 0.5) and Direct ML for five learners; bars = 95 % block-bootstrap intervals",
        colours={C.METHOD_NAME[m]: C.METHOD[m] for m in STRUCT}, region_markers={C.REGION[r][0]: C.REGION[r][2] for r in REG},
        label_year_ranges=C.label_year_ranges(), values=vals, numbers_printed="none (values in this JSON)",
        outputs=C.file_info(paths), size_mm=[W, H], qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
