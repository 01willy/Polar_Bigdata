"""Supplementary Fig. S10 (v11; v10 Fig. S11): label ladder with all model structures (content of v8_f9_label_ladder.py paper figure).

Copied from paper_v10/si_s11_label_ladder.py (data unchanged) and restyled for v11 (si11_common.py, POLAR_PALETTE=paper11): v11 method colours,
markers 3 pt, light grid lines, Physics + residual ML band as a solid light tint.
a  Within-region label ladder: Delta RMSE (method - Stefan (recalibrated) at the same number of training labels) against the number of training
   labels (cell-years; ordered categories, 'all' = all training-fold labels of the region), for Direct ML, Physics pseudo-labels, Physics-input ML and
   Physics + residual ML (band = 95 % block-bootstrap interval of Physics + residual ML). Spatial-block CV.
b  New-region transfer: RMSE in the held-out region against the number of target-region labels added to the training data, for Stefan (source coeff.),
   Stefan (recalibrated on the target labels), Direct ML, Physics pseudo-labels, Physics + residual ML and Physics + residual ML trained on the target
   labels only.
Inputs: data/processed/v5/runs/ladder_v6/ladder_metrics.csv, ladder_deltas.csv via v8_f9_label_ladder.load() (rows present in only some folds removed).
Output: outputs/figures/paper/v11_si/FigS10_label_ladder.{pdf,png}, entry S10 of outputs/figures/paper/v11_si/SI_figures.json.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper11 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v11/si_s10_label_ladder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import si11_common as C  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import v8_f9_label_ladder as F9  # noqa: E402

STEM = "FigS10_label_ladder"
REG = C.REGIONS
DOT = (0, (1.0, 1.3)); DASHDOT = (0, (3.0, 1.2, 1.0, 1.2)); DASH = (0, (3.4, 1.7))
LL = {"D0": (C.METHOD_NAME["D0"], C.METHOD["D0"], DASH), "D1": (C.METHOD_NAME["D1"], C.METHOD["D1"], DOT),
      "F1": (C.METHOD_NAME["F1"], C.METHOD["F1"], DASHDOT), "R1": (C.METHOD_NAME["R1"], C.METHOD["R1"], "-")}
MK = {"D0": ("o", False), "D0S": ("o", False), "P0": ("s", True), "P1n": ("s", True), "R1T": ("o", False)}   # (marker, filled); others filled circles
TL = {"P0": (C.METHOD_NAME["P0"], C.METHOD["P0"], "-"), "P1n": ("Stefan (recalibrated on target labels)", C.METHOD["P1"], "-"),
      "D0S": (C.METHOD_NAME["D0"], C.METHOD["D0"], DASH), "D1S": (C.METHOD_NAME["D1"], C.METHOD["D1"], DOT),
      "R1S": (C.METHOD_NAME["R1"], C.METHOD["R1"], "-"), "R1T": ("Physics + residual ML, target labels only", C.METHOD["R1"], DASH)}
BAND = C.tint(C.METHOD["R1"], 0.20)
DODGE_LL = {"R1": -0.15, "D1": -0.05, "F1": 0.05, "D0": 0.15}                                   # category units (~0.7 mm per 0.1)
DODGE_TL = {"P0": -0.20, "P1n": -0.12, "D1S": -0.04, "R1S": 0.04, "R1T": 0.12, "D0S": 0.20}
MS_C, MS_S = 3.0, 2.7                           # circle diameter, square side (equal area)


def mk(k, col):
    m, f = MK.get(k, ("o", True))
    return dict(marker=m, ms=MS_C if m == "o" else MS_S, mfc=col if f else C.WHITE, mec=col, mew=C.MEW if not f else 0.0)


def lw(k):
    return C.LW_MAIN if k.startswith("R1") else 0.85


def cat_label(c):
    return "All" if c == "all" else f"{int(c)}"


def row_axes(fig, n_cats, y, h, W):
    x0, gap, xr = 13.0, 12.0, 2.0
    u = (W - x0 - 2 * gap - xr) / sum(k - 0.2 for k in n_cats); xs = np.cumsum([x0] + [(k - 0.2) * u + gap for k in n_cats[:-1]])
    return [C.ax_mm(fig, x, y, (k - 0.2) * u, h) for x, k in zip(xs, n_cats)]


def main():
    C.setup()
    m, d = F9.load()
    W, H = 180.0, 116.0
    fig = C.figure(W, H); src = {"a": {}, "b": {}}
    ql = d[(d.exp == "LL") & (d.ref == "P1")]; qt = m[m.exp == "TL"]
    ncl = [len(F9.cats(ql[ql.region == r].n_label.unique())[0]) for r in REG]; nct = [len(F9.cats(qt[qt.region == r].n_label.unique())[0]) for r in REG]
    axa = row_axes(fig, ncl, 12.0, 34.0, W); axb = row_axes(fig, nct, 70.0, 34.0, W)
    for ax, r in zip(axa, REG):
        C.clean_axes(ax); C.grid(ax, "y"); qr = ql[ql.region == r]; cs, pos = F9.cats(qr.n_label.unique())
        for k, (name, col, ls) in LL.items():
            s = qr[qr.method == k].copy(); s["x"] = s.n_label.map(pos) + DODGE_LL[k]; s = s.sort_values("x")
            if not len(s):
                continue
            if k == "R1":
                ax.fill_between(s.x, s.lo, s.hi, color=BAND, lw=0, zorder=1)
            ax.plot(s.x, s.delta, ls=ls, lw=lw(k), color=col, zorder=3 + (k == "R1"), **mk(k, col))
            src["a"].setdefault(r, {})[name] = {c: round(float(v), 3) for c, v in s.set_index("n_label").delta.items()}
        ax.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=2)
        ax.set_xticks(range(len(cs))); ax.set_xticklabels([cat_label(c) for c in cs]); ax.set_xlim(-0.4, len(cs) - 0.6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.text(0.0, 1.0 + 1.3 / 34.0, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        n_all = qr[qr.n_label == "all"].n.max(); src["a"].setdefault(r, {})["n_all_training_labels"] = int(n_all) if np.isfinite(n_all) else None
    axa[0].set_ylabel("ΔRMSE vs recalibrated\nStefan (cm)"); axa[1].set_xlabel("Training labels in the region (cell-years)")
    for ax, r in zip(axb, REG):
        C.clean_axes(ax); C.grid(ax, "y"); qr = qt[qt.region == r]; cs, pos = F9.cats(qr.n_label.unique())
        for k, (name, col, ls) in TL.items():
            s = qr[qr.method == k].copy(); s["x"] = s.n_label.map(pos) + DODGE_TL[k]; s = s.sort_values("x")
            if not len(s):
                continue
            ax.plot(s.x, s.rmse, ls=ls, lw=lw(k), color=col, zorder=3 + k.startswith("R1"), **mk(k, col))
            src["b"].setdefault(r, {})[name] = {c: round(float(v), 3) for c, v in s.set_index("n_label").rmse.items()}
        ax.set_xticks(range(len(cs))); ax.set_xticklabels([cat_label(c) for c in cs]); ax.set_xlim(-0.4, len(cs) - 0.6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.text(0.0, 1.0 + 1.3 / 34.0, C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
    axb[0].set_ylabel("RMSE (cm)"); axb[1].set_xlabel("Target-region labels added (cell-years)")
    ha = [Line2D([], [], color=C.ZERO, lw=C.LW_REF)] + [Line2D([], [], ls=ls, lw=lw(k), color=col, **mk(k, col))
                                                       for k, (n, col, ls) in LL.items()] + [Patch(facecolor=BAND, lw=0)]
    fig.legend(ha, [C.METHOD_NAME["P1"]] + [v[0] for v in LL.values()] + ["95% interval"], loc="upper left", bbox_to_anchor=(13.0 / W, 1 - 0.6 / H),
               ncol=3, handlelength=2.4, columnspacing=1.6, labelspacing=0.3, fontsize=C.FS_TICK)
    hb = [Line2D([], [], ls=ls, lw=lw(k), color=col, **mk(k, col)) for k, (n, col, ls) in TL.items()]
    fig.legend(hb, [v[0] for v in TL.values()], loc="upper left", bbox_to_anchor=(13.0 / W, 1 - 58.6 / H), ncol=3, handlelength=2.4,
               columnspacing=1.6, labelspacing=0.3, fontsize=C.FS_TICK)
    C.letter(fig, 0.5, 0.5, "a"); C.letter(fig, 0.5, 58.5, "b")
    rep = C.qa_figure(fig); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S10", dict(
        title="Label ladder with all model structures", stem=STEM, script="scripts/4_visualization/paper_v11/si_s10_label_ladder.py",
        version="v11 (2026-10-16): v11 palette restyle of v10 FigS11_label_ladder (renumbered S10); numbers unchanged",
        inputs=["data/processed/v5/runs/ladder_v6/ladder_metrics.csv", "data/processed/v5/runs/ladder_v6/ladder_deltas.csv"],
        years="label cell-years of each region pooled (Alaska 2010-2024, Lena Delta 2010-2022, Canada (NW) 2010-2024); climate input to the observation day",
        definition=dict(a="Delta RMSE vs Stefan (recalibrated) at the same number of training labels, spatial-block CV, band = 95 % block bootstrap of "
                          "Physics + residual ML; x = ordered label-number categories ('all' = all training-fold labels)",
                        b="region hold-out RMSE against target-region labels added (0, 25, 50, 100, 250, 500)"),
        colours={v[0]: v[1] for v in list(LL.values()) + list(TL.values())},
        values=src, numbers_printed="none", outputs=C.file_info(paths), size_mm=[W, H], qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
