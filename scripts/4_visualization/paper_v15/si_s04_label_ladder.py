"""Supplementary Fig. S4 (v15.2, 2026-10-22): within-region label ladder, three methods only.

Delta RMSE against the Stefan model recalibrated on the same training labels, for the physics-residual model and direct ML, against the number of
training labels drawn at random from the training folds of a spatial-block cross-validation (25 ... 2,500 site-years and all; three draws per
number, averaged); bands = 95 % block-bootstrap intervals. The transfer ladder (former panel b) is Fig. 4c of the main text; the two other
model structures (physics-input ML, physics pseudo-labels) are in the deposited ladder tables.
Inputs: data/processed/v5/runs/ladder_v6/ladder_metrics.csv, ladder_deltas.csv via v8_f9_label_ladder.load().
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper15 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 python3 scripts/4_visualization/paper_v15/si_s04_label_ladder.py
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
from matplotlib.patches import Patch  # noqa: E402
import v8_f9_label_ladder as F9  # noqa: E402

STEM = "FigS4_label_ladder"; REG = C.REGIONS; DASH = (0, (3.4, 1.7))
SER = {"R1": (C.METHOD_NAME["R1"], C.METHOD["R1"], "-", "o", True), "D0": (C.METHOD_NAME["D0"], C.METHOD["D0"], DASH, "o", False)}
DODGE = {"R1": -0.08, "D0": 0.08}


def cat_label(c): return "All" if c == "all" else f"{int(c)}"


def main():
    C.setup(); m, d = F9.load(); W, H = 180.0, 60.0; fig = C.figure(W, H); src = {}
    ql = d[(d.exp == "LL") & (d.ref == "P1")]
    ncl = [len(F9.cats(ql[ql.region == r].n_label.unique())[0]) for r in REG]
    x0, gap, xr = 14.0, 11.0, 3.0; u = (W - x0 - 2 * gap - xr) / sum(k - 0.2 for k in ncl); xs = np.cumsum([x0] + [(k - 0.2) * u + gap for k in ncl[:-1]])
    axes = [C.ax_mm(fig, x, 14.0, (k - 0.2) * u, 36.0) for x, k in zip(xs, ncl)]
    for ax, r in zip(axes, REG):
        C.clean_axes(ax); qr = ql[ql.region == r]; cs, pos = F9.cats(qr.n_label.unique())
        for k, (name, col, ls, mk, filled) in SER.items():
            s = qr[qr.method == k].copy(); s["x"] = s.n_label.map(pos) + DODGE[k]; s = s.sort_values("x")
            if not len(s): continue
            ax.fill_between(s.x, s.lo, s.hi, color=C.tint(col, 0.18), lw=0, zorder=1)
            ax.plot(s.x, s.delta, ls=ls, lw=C.LW_MAIN, color=col, marker=mk, ms=3.0, mfc=col if filled else C.WHITE, mec=col, mew=0.7, zorder=3 + (k == "R1"))
            src.setdefault(r, {})[name] = {c: [round(float(v), 3), round(float(lo), 3), round(float(hi), 3)] for c, v, lo, hi in zip(s.n_label, s.delta, s.lo, s.hi)}
        ax.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=2)
        ax.set_xticks(range(len(cs))); ax.set_xticklabels([cat_label(c) for c in cs]); ax.set_xlim(-0.4, len(cs) - 0.6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        n_all = qr[qr.n_label == "all"].n.max(); src.setdefault(r, {})["n_all_training_labels"] = int(n_all) if np.isfinite(n_all) else None
        ax.text(0.0, 1.0 + 1.3 / 36.0, f"{C.REGION[r][0]} ({int(n_all):,} training labels in all)" if np.isfinite(n_all) else C.REGION[r][0], transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
    axes[0].set_ylabel("ΔRMSE against Stefan recalibrated\non the same labels (cm)"); axes[1].set_xlabel("Training labels drawn from the region (site-years)")
    hs = [Line2D([], [], color=C.ZERO, lw=C.LW_REF)] + [Line2D([], [], ls=ls, lw=C.LW_MAIN, color=col, marker=mk, ms=3.0, mfc=col if f else C.WHITE, mec=col, mew=0.7) for _, col, ls, mk, f in SER.values()] + [Patch(facecolor="#DDDDDD", lw=0)]
    fig.legend(hs, [C.METHOD_NAME["P1"] + " (reference)"] + [v[0] for v in SER.values()] + ["95% block-bootstrap interval"], loc="upper left", bbox_to_anchor=(x0 / W, 1 - 1.0 / H),
               ncol=4, handlelength=2.4, columnspacing=1.6, labelspacing=0.3, fontsize=C.FS_TICK, frameon=False, borderaxespad=0)
    C.letter(fig, 0.5, 0.5, "a")
    rep = C.qa_figure(fig); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S4", dict(title="Within-region label ladder", stem=STEM, script="scripts/4_visualization/paper_v15/si_s04_label_ladder.py", version="v15.2 (2026-10-22): two learners, bands for both",
                                inputs=["data/processed/v5/runs/ladder_v6/ladder_metrics.csv", "data/processed/v5/runs/ladder_v6/ladder_deltas.csv"],
                                definition="Delta RMSE vs Stefan recalibrated on the same labels, spatial-block CV, labels drawn at random from the training folds (3 draws), 95 % block bootstrap",
                                values=src, outputs=C.file_info(paths), size_mm=[W, H], qa=rep))
    for p in paths: print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
