# -*- coding: utf-8 -*-
"""Fig. 3 design proposals B and C (docs/FIGURE_PROPOSALS_v14_2026-10-20.md), rendered with the real result tables for the user's choice.
B  Specification curve of the physics-residual gain: dRMSE = RMSE(Physics + residual ML) - RMSE(Stefan recalibrated) for every specification run
   (validation design, climate input, lambda, block size, learner, covariate set), sorted within region, with 95 % block-bootstrap intervals and an
   indicator matrix of the specification below (Simonsohn et al. 2020 style). One panel answers "when does ML beyond the physics help".
C  Transfer-penalty plot: RMSE under spatial-block cross-validation (x) against RMSE under region hold-out (y) for the four methods in the three
   regions; the diagonal marks no penalty; the distance above the diagonal is the loss of accuracy when the region has no labels.
Inputs: data/processed/v5/runs/{core_v6,featsets_v6,modis_v6,nn_v6,rf_v6,blk025_v6,blk100_v6}/deltas_vs_ref.csv, core_v6/summary_metrics.csv.
Outputs: outputs/figures/paper/v14_proposals/Fig3_B_specification_curve.{png,pdf}, Fig3_C_transfer_penalty.{png,pdf}
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 python3 scripts/4_visualization/paper_v14/fig3_proposals.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig34_common13 as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = C.ROOT; RUNS = ROOT / "data" / "processed" / "v5" / "runs"
OUT = ROOT / "outputs" / "figures" / "paper" / "v14_proposals"; OUT.mkdir(parents=True, exist_ok=True)
REG = C.REG
SCHEME_LB = {"random": "Random", "block": "Spatial block", "year": "Year", "block_year": "Block × year", "region": "Region"}
FORC_LB = {"clim": "Mean 2011–2024", "year": "Full season", "year_obs": "To obs. day"}
FEAT_LB = {"F8": "Climate", "F8T": "+ Terrain", "F8TS": "+ Soil", "F8TS_M": "+ MODIS", "F8TS_X": "+ Hi-res soil", "F8TS_L": "+ Hi-res soil", "F8TS_ALL": "All"}
LRN_LB = {"catboost": "CatBoost", "rf": "Random forest", "mlp": "MLP", "tabm": "TabM", "ftt": "FT-Transformer"}


def spec_rows():
    """All (region, dRMSE R1 - P1, lo, hi) estimates with their specification; one row per run."""
    rows = []

    def add(d, axis, label_fn, **fixed):
        for q in d.itertuples():
            rows.append(dict(region=q.region, delta=float(q.delta), lo=float(q.lo), hi=float(q.hi), n=int(q.n), axis=axis, label=label_fn(q),
                             design=SCHEME_LB[q.scheme], input=FORC_LB.get(q.forcing, q.forcing), lam=f"λ {q.lam:g}", block=fixed.get("block", "0.5°"),
                             learner=LRN_LB.get(q.learner, str(q.learner)), feats=FEAT_LB.get(q.featset, q.featset)))

    core = pd.read_csv(RUNS / "core_v6" / "deltas_vs_ref.csv"); core = core[(core.ref == "P1") & (core.method == "R1") & (core.featset == "F8TS")]
    base = core[(core.learner == "catboost") & (core.lam == 0.5) & (core.forcing == "year_obs")]
    add(base, "design", lambda q: SCHEME_LB[q.scheme])                                                   # 5 designs
    lam = core[(core.learner == "catboost") & (core.scheme == "block") & (core.forcing == "year_obs") & (core.lam != 0.5)]
    add(lam, "lambda", lambda q: f"λ {q.lam:g}")
    forc = core[(core.learner == "catboost") & (core.scheme == "block") & (core.lam == 0.5) & (core.forcing != "year_obs")]
    add(forc, "input", lambda q: FORC_LB[q.forcing])
    for tag, blk in (("blk025_v6", "0.25°"), ("blk100_v6", "1.0°")):
        d = pd.read_csv(RUNS / tag / "deltas_vs_ref.csv"); d = d[(d.ref == "P1") & (d.method == "R1") & (d.lam == 0.5)]
        add(d, "block", lambda q, b=blk: b, block=blk)
    for tag in ("nn_v6", "rf_v6"):
        d = pd.read_csv(RUNS / tag / "deltas_vs_ref.csv")
        d = d[(d.ref == "P1") & (d.method == "R1") & (d.scheme == "block") & (d.forcing == "year_obs") & (d.lam == 0.5) & (d.learner != "catboost")]
        add(d, "learner", lambda q: LRN_LB.get(q.learner, q.learner))
    for tag in ("featsets_v6", "modis_v6"):
        d = pd.read_csv(RUNS / tag / "deltas_vs_ref.csv")
        d = d[(d.ref == "P1") & (d.method == "R1") & (d.scheme == "block") & (d.forcing == "year_obs") & (d.lam == 0.5) & (d.featset != "F8TS")]
        add(d, "feats", lambda q: FEAT_LB.get(q.featset, q.featset))
    df = pd.DataFrame(rows).drop_duplicates(["region", "axis", "label"])
    return df


def draw_b(df):
    W, H = 180.0, 118.0
    fig = C.new_fig(H)
    axes_cols = ["design", "input", "lam", "block", "learner", "feats"]
    keys = {"design": list(SCHEME_LB.values()), "input": list(FORC_LB.values()), "lam": ["λ 0.25", "λ 0.5", "λ 1"], "block": ["0.25°", "0.5°", "1.0°"],
            "learner": list(LRN_LB.values()), "feats": ["Climate", "+ Terrain", "+ Soil", "+ MODIS", "+ Hi-res soil", "All"]}
    # fill the specification of each row with the main setting where the axis is not varied
    main = dict(design="Spatial block", input="To obs. day", lam="λ 0.5", block="0.5°", learner="CatBoost", feats="+ Soil")
    for c in axes_cols:
        df[c] = df[c].where(df.axis == {"design": "design", "input": "input", "lam": "lambda", "block": "block", "learner": "learner", "feats": "feats"}[c], main[c]) \
            if c != "lam" else df["lam"]
    df.loc[df.axis != "lambda", "lam"] = "λ 0.5"
    df.loc[df.axis != "input", "input"] = "To obs. day"; df.loc[df.axis != "design", "design"] = "Spatial block"
    df.loc[df.axis != "learner", "learner"] = "CatBoost"; df.loc[df.axis != "feats", "feats"] = "+ Soil"
    # x positions: regions side by side, sorted by delta within region
    X0, XW, GAP = 24.0, 124.0, 4.0                 # room for the specification labels at the right (about 28 mm)
    n_tot = len(df); x = 0; xs = {}
    order = []
    for r in REG:
        q = df[df.region == r].sort_values("delta")
        for i in q.index:
            xs[i] = x; x += 1
        order.extend(q.index.tolist()); x += 2
    nx = x - 2
    dx = (XW) / nx
    ax = C.ax_mm(fig, X0, 8.0, XW, 44.0)
    for i in order:
        q = df.loc[i]; col = C.REG_HEX[q.region]; xx = xs[i] * dx + dx / 2
        resolved = q.hi < 0 or q.lo > 0
        ax.vlines(xx, q.lo, q.hi, color=col, lw=0.6, alpha=0.9, zorder=2)
        ax.plot([xx], [q.delta], ls="none", marker=C.REG_MK[q.region], ms=C.REG_MS[q.region] * 0.85, mfc=col if resolved else C.WHITE, mec=col, mew=0.6, zorder=3)
    C.zero_line(ax)
    ax.set_xlim(0, XW); ax.set_xticks([])
    ax.set_ylabel("ΔRMSE, physics-residual − Stefan (cm)")
    ax.spines["bottom"].set_visible(False)
    for r in REG:
        q = df[df.region == r]; xm = np.mean([xs[i] for i in q.index]) * dx + dx / 2
        ax.text(xm, 1.0, C.REG_EN[r], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=C.FS, color=C.TEXT)
    # indicator matrix
    rows = [(c, v) for c in axes_cols for v in keys[c]]
    hrow = 2.3; y0 = 8.0 + 44.0 + 3.0
    axm = C.ax_mm(fig, X0, y0, XW, hrow * len(rows)); axm.set_xlim(0, XW); axm.set_ylim(len(rows), 0); axm.axis("off")
    grp_lab = {"design": "Validation design", "input": "Climate input", "lam": "Weight λ", "block": "Block size", "learner": "Learner", "feats": "Covariate set"}
    prev = None
    for j, (c, v) in enumerate(rows):
        if c != prev:
            axm.text(-1.5, j + 0.5, grp_lab[c], ha="right", va="center", fontsize=C.FS_TICK, color=C.TEXT, fontweight="bold")
            prev = c
        axm.text(XW + 1.5, j + 0.5, v, ha="left", va="center", fontsize=C.FS_TICK, color=C.TEXT)
        if j % 2 == 0:
            axm.axhspan(j, j + 1, color="#F2F2F2", lw=0, zorder=0)
        for i in order:
            q = df.loc[i]
            if q[c] == v:
                axm.plot([xs[i] * dx + dx / 2], [j + 0.5], ls="none", marker="o", ms=1.6, mfc=C.REG_HEX[q.region], mec="none", zorder=2)
    hs = [Line2D([], [], ls="none", marker="o", ms=3.0, mfc=C.TEXT, mec=C.TEXT), Line2D([], [], ls="none", marker="o", ms=3.0, mfc=C.WHITE, mec=C.TEXT, mew=0.6)]
    fig.legend(hs, ["95% interval excludes 0", "includes 0"], loc="upper left", bbox_to_anchor=(X0 / W, 1 - 1.0 / H), ncol=2, frameon=False, fontsize=C.FS_TICK,
               handlelength=1.2, columnspacing=1.5, borderaxespad=0)
    C.text_mm(fig, 1.0, 1.5, "B", fontsize=C.FS_LETTER, fontweight="bold", ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig3_B_specification_curve.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    return len(df)


def draw_c():
    sm = pd.read_csv(RUNS / "core_v6" / "summary_metrics.csv"); sm = sm[(sm.featset == "F8TS") & (sm.forcing == "year_obs")]
    W, H = 180.0, 80.0
    fig = C.new_fig(H)
    ax = C.ax_mm(fig, 14.0, 6.0, 66.0, 62.0)
    pts = []
    for r in REG:
        for m in ("P0", "P1", "R1", "D0"):
            q = sm[(sm.region == r) & (sm.method == m)]
            if m == "R1":
                q = q[(q.lam == 0.5) & (q.learner == "catboost")]
            elif m == "D0":
                q = q[q.learner == "catboost"]
            b = q[q.scheme == "block"].rmse.iloc[0]; g = q[q.scheme == "region"].rmse.iloc[0]
            st = C.METH[m]
            ax.plot([b], [g], ls="none", marker=st["marker"], ms=st["ms"] + 0.6, mfc=C.WHITE if st["open"] else st["color"], mec=st["color"], mew=C.MEW if st["open"] else 0.0, zorder=3)
            ab = {"Alaska": "AK", "Lena": "LD", "Canada": "CA"}[r]; off = {"P0": (3, 3), "P1": (3, -4), "R1": (-3, 3), "D0": (-3, -4)}[m]
            ax.annotate(ab, (b, g), xytext=off, textcoords="offset points", fontsize=5.5, color=C.AUX, va="center", ha="left" if off[0] > 0 else "right")
            pts.append((r, m, b, g))
    lo = 10.0; hi = 42.0
    ax.plot([lo, hi], [lo, hi], color=C.ZERO, lw=C.LW_REF, zorder=1)
    ax.set_xlim(lo, 32); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("RMSE, spatial-block CV (cm)"); ax.set_ylabel("RMSE, region hold-out (cm)")
    ax.xaxis.set_major_locator(MultipleLocator(5)); ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.text(0.03, 0.97, "Above the diagonal:\naccuracy lost without\ntarget-region labels", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_TICK, color=C.AUX)
    hs = [C.method_handle(m) for m in ("P0", "P1", "R1", "D0")]
    fig.legend(hs, [C.METH[m]["label"] for m in ("P0", "P1", "R1", "D0")], loc="upper left", bbox_to_anchor=(92.0 / W, 1 - 8.0 / H), ncol=1, frameon=False,
               fontsize=C.FS_TICK, handlelength=1.8, borderaxespad=0)
    C.text_mm(fig, 92.0, 30.0, "AK Alaska, LD Lena Delta, CA Canada (NW)", fontsize=C.FS_TICK, color=C.AUX, ha="left", va="top")
    C.text_mm(fig, 1.0, 1.5, "C", fontsize=C.FS_LETTER, fontweight="bold", ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Fig3_C_transfer_penalty.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    return pts


if __name__ == "__main__":
    C.setup()
    df = spec_rows(); print("specifications:", len(df), df.groupby(["region", "axis"]).size().to_dict())
    n = draw_b(df); pts = draw_c()
    print("B rows", n, "| C points", len(pts)); print(sorted(OUT.glob("Fig3_*")))
