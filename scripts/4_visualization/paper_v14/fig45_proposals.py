# -*- coding: utf-8 -*-
"""Fig. 4 and Fig. 5 design proposals (docs/FIGURE_PROPOSALS_v14_2026-10-20.md), rendered with the real result tables for the user's choice.
Fig. 4 proposal C  Absolute RMSE small multiples: rows = regions, columns = within-region label ladder (training labels n) and transfer ladder
                   (target-region labels n added to the other regions); the same y axis per region, so that the two experiments are read on one scale;
                   the Stefan baselines as horizontal references; shaded seed spread (3 draws) for the physics-residual model.
Fig. 5 proposal B  Target diagram (Jolliff et al. 2009): x = mean bias, y = centred RMSE (standard deviation of the error), distance from the origin =
                   RMSE, for the three methods and five products on the common site-years of each region (all cells and cells > 25 km from Wei 2026
                   training sites). One panel per region; RMSE isolines as quarter circles.
Inputs: data/processed/v5/runs/ladder_v6/ladder_metrics.csv, data/processed/v5/runs/core_v6/common_cells_eval.csv
Outputs: outputs/figures/paper/v14_proposals/Fig4_C_absolute_ladders.{png,pdf}, Fig5_B_target_diagram.{png,pdf}
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 python3 scripts/4_visualization/paper_v14/fig45_proposals.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig34_common13 as C  # noqa: E402
import c13_common as C13  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = C.ROOT; RUNS = ROOT / "data" / "processed" / "v5" / "runs"
OUT = ROOT / "outputs" / "figures" / "paper" / "v14_proposals"; OUT.mkdir(parents=True, exist_ok=True)
REG = C.REG


def nkey(s):
    return 10 ** 9 if s == "all" else int(s)


def fig4_c():
    m = pd.read_csv(RUNS / "ladder_v6" / "ladder_metrics.csv"); m["n_label"] = m.n_label.astype(str)
    full = m[(m.exp == "LL") & (m.n_label == "all")].groupby("region").n_test.max()
    m = m[(m.exp != "LL") | (m.n_test >= 0.95 * m.region.map(full))]
    W, H = 180.0, 150.0
    fig = C.new_fig(H)
    xs0 = {"LL": 14.0, "TL": 100.0}; w = 72.0; h = 38.0; gy = 9.0
    meth = {"LL": ["P1", "R1", "D0"], "TL": ["P0", "P1n", "R1S", "D0S"]}
    mstyle = {"P1n": "P1", "R1S": "R1", "D0S": "D0", "P0": "P0", "P1": "P1", "R1": "R1", "D0": "D0"}
    for i, r in enumerate(REG):
        y0 = 8.0 + i * (h + gy)
        ylo, yhi = np.inf, -np.inf
        axes = []
        for exp in ("LL", "TL"):
            ax = C.ax_mm(fig, xs0[exp], y0, w, h); axes.append(ax)
            q = m[(m.exp == exp) & (m.region == r)]
            ns = sorted(q.n_label.unique(), key=nkey)
            xpos = {n: k for k, n in enumerate(ns)}
            for mm in meth[exp]:
                s = q[q.method == mm].set_index("n_label").reindex(ns)
                st = C.METH[mstyle[mm]]
                x = np.array([xpos[n] for n in ns]); y = s.rmse.values
                if mm in ("R1", "R1S") and "rmse_seed_sd" in s:
                    ax.fill_between(x, y - s.rmse_seed_sd.values, y + s.rmse_seed_sd.values, color=st["color"], alpha=0.18, lw=0, zorder=1)
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=2)
                ax.plot(x, y, ls="none", marker=st["marker"], ms=st["ms"], mfc=C.WHITE if st["open"] else st["color"], mec=st["color"], mew=C.MEW if st["open"] else 0.0, zorder=3)
                ylo = min(ylo, np.nanmin(y)); yhi = max(yhi, np.nanmax(y[np.isfinite(y)] if np.isfinite(y).any() else [yhi]))
            ax.set_xticks(range(len(ns))); ax.set_xticklabels([n if n != "all" else "All" for n in ns])
            ax.set_xlabel("Training labels in the region (site-years)" if exp == "LL" else "Target-region labels added (site-years)")
            C.facet(ax, f"{C.REG_EN[r]}: " + ("within region" if exp == "LL" else "transfer from the other regions"))
        for ax in axes:
            ax.set_ylim(np.floor(ylo / 2) * 2 - 1, min(np.ceil(yhi / 2) * 2 + 1, np.floor(ylo / 2) * 2 + 24))
            ax.yaxis.set_major_locator(MultipleLocator(4))
        axes[0].set_ylabel("RMSE (cm)")
    hs = [C.method_handle(mm) for mm in ("P0", "P1", "R1", "D0")]
    fig.legend(hs, [C.METH[mm]["label"] for mm in ("P0", "P1", "R1", "D0")], loc="upper left", bbox_to_anchor=(40.0 / W, 1 - 1.0 / H), ncol=4, frameon=False,
               fontsize=C.FS, handlelength=2.4, columnspacing=2.0, borderaxespad=0)
    C.text_mm(fig, 1.0, 1.5, "Fig. 4 proposal C", fontsize=C.FS_LETTER, fontweight="bold", ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig4_C_absolute_ladders.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def fig5_b():
    cc = pd.read_csv(RUNS / "core_v6" / "common_cells_eval.csv")
    series = [("P1 (Stefan, recal.)", "P1"), ("R1 (CatBoost)", "R1"), ("D0 (CatBoost)", "D0"), ("Wei 2026", "wei"), ("Liu 2024", "liu"), ("ESA CCI v4", "cci4y"),
              ("ESA CCI v5", "cci5y"), ("Aalto 2018", "aalto")]
    W, H = 180.0, 70.0
    fig = C.new_fig(H)
    w = 46.0; h = 46.0; x0 = 14.0; g = 13.0
    lim = {"Alaska": 40.0, "Lena": 40.0, "Canada": 40.0}
    for j, r in enumerate(REG):
        ax = C.ax_mm(fig, x0 + j * (w + g), 10.0, w, h)
        L = lim[r]
        for rr in (10, 20, 30, 40):
            th = np.linspace(0, np.pi, 200); ax.plot(rr * np.cos(th), rr * np.sin(th), color="#E0E0E0", lw=0.5, zorder=1)
            if rr <= L:
                ax.text(rr * 0.72, rr * 0.72, f"{rr}", fontsize=5.5, color=C.AUX, ha="center", va="center", zorder=1)
        ax.axvline(0, color=C.ZERO, lw=C.LW_REF, zorder=1)
        for code, key in series:
            for mask, open_ in ((0, False), (25, True)):
                q = cc[(cc.region == r) & (cc.mask_km == mask) & (cc.series == code)]
                if not len(q):
                    continue
                q = q.iloc[0]; rmse, bias = float(q.rmse), float(q.bias); crmse = np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))
                if key in ("P1", "R1", "D0"):
                    name, fc, ec, mk = C13.METHOD_NAME[key], C13.METHOD[key], C13.METHOD[key], {"P1": "s", "R1": "o", "D0": "o"}[key]
                else:
                    name, fc, ec, mk = C13.product_style(key)
                over = rmse > L
                xx, yy = (bias, crmse) if not over else (bias * L / rmse, crmse * L / rmse)
                ax.plot([xx], [yy], ls="none", marker=mk, ms=3.6 if not over else 3.0, mfc=C.WHITE if open_ else fc, mec=ec, mew=0.7, zorder=3 + (0 if open_ else 1))
                if over and not open_:
                    ax.annotate(f"{rmse:.0f}", (xx, yy), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5.5, color=C.AUX)
        ax.set_xlim(-L, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(20)); ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlabel("Mean bias (cm)"); C.facet(ax, C.REG_EN[r])
        if j == 0:
            ax.set_ylabel("Centred RMSE (cm)")
    hs, labs = [], []
    for code, key in series:
        if key in ("P1", "R1", "D0"):
            name, fc, ec, mk = C13.METHOD_NAME[key], C13.METHOD[key], C13.METHOD[key], {"P1": "s", "R1": "o", "D0": "o"}[key]
        else:
            name, fc, ec, mk = C13.product_style(key)
        hs.append(Line2D([], [], ls="none", marker=mk, ms=3.6, mfc=fc, mec=ec, mew=0.7)); labs.append(name)
    hs.append(Line2D([], [], ls="none", marker="o", ms=3.6, mfc=C.WHITE, mec=C.TEXT, mew=0.7)); labs.append("open: > 25 km from Wei training sites")
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(34.0 / W, 1 - 1.0 / H), ncol=5, frameon=False, fontsize=C.FS_TICK, handlelength=1.2, columnspacing=1.4, borderaxespad=0)
    C.text_mm(fig, 1.0, 1.5, "Fig. 5 proposal B", fontsize=C.FS_LETTER, fontweight="bold", ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig5_B_target_diagram.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    C.setup(); fig4_c(); fig5_b(); print(sorted(p.name for p in OUT.glob("Fig[45]_*")))
