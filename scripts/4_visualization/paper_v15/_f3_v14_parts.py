# -*- coding: utf-8 -*-
"""Paper v14 Fig. 3 (2026-10-20, design A of docs/FIGURE_PROPOSALS_v14_2026-10-20.md): accuracy of physics + residual ML and the conditions of its gain.

Changes from v13: (a) the effect of the climate input is shown as the paired difference with its 95 % block-bootstrap interval (forest plot) instead of
two connected points; (b) the validation designs are categorical, so the four methods are drawn as points without connecting lines, and a strip below
each region panel gives the difference physics-residual minus recalibrated Stefan with its interval for every design (the gain and whether it is
resolved); (c) and (d) as in v13.
a  dRMSE = RMSE(year-specific input) - RMSE(2011-2024 mean input), spatial block, for Stefan (recalibrated), Physics + residual ML and Direct ML in
   each region; 95 % paired block-bootstrap intervals; negative favours the year-specific input.
b  Upper: RMSE by validation design (random, spatial block, year, block x year, region hold-out), four methods, year-specific input; values above
   1.3 x the largest value of the four non-region designs are drawn on the top edge (open marker) with the value printed.
   Lower: dRMSE = RMSE(Physics + residual ML) - RMSE(Stefan, recalibrated) per design with its 95 % interval.
c  dRMSE (Physics + residual ML - Stefan recalibrated) for six covariate sets, spatial block, 95 % intervals; regions by colour and marker shape.
d  The same dRMSE within regional terciles of eight covariates (heat map; dark outline = interval excludes 0; hatching = fewer than 30 site-years or
   5 blocks).
Inputs: data/processed/v5/runs/core_v6/{summary_metrics,deltas_forcing,deltas_vs_ref}.csv, runs/{modis_v6,featsets_v6}/deltas_vs_ref.csv,
outputs/figures/paper/v7/source_data/F4e_gain_conditions.csv, label years from runs/core_v6/preds (P1 out-of-fold files).
Main setting: forcing year_obs, featset F8TS, learner catboost, R1 lam 0.5, scheme block.
Output: outputs/figures/paper/v15/Fig3_accuracy.{pdf,svg,png} and source_data/Fig3_accuracy.json.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v15/fig3_accuracy.py
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
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator, MaxNLocator  # noqa: E402

ROOT = C.ROOT; RUNS = ROOT / "data" / "processed" / "v5" / "runs"
F4E = ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F4e_gain_conditions.csv"
STEM = "Fig3_accuracy"
REG = C.REG
MAIN = dict(forcing="year_obs", featset="F8TS", learner="catboost", lam=0.5, scheme="block")
A_METH = ["P1", "R1", "D0"]                              # a: rows per region (Stefan recalibrated, Physics + residual ML, Direct ML)
B_METH = ["P0", "P1", "D0", "R1"]
SCHEMES = ["random", "block", "year", "block_year", "region"]
SCHEME_LB = ["Random", "Spatial block", "Year", "Block × year", "Region"]
OVER = 1.3
FEATS = ["F8", "F8T", "F8TS", "F8TS_M", "F8TS_X", "F8TS_ALL"]
FEAT_LB = ["Climate only", "+ Terrain", "+ Soil (main set)", "+ MODIS", "+ High-resolution soil\nand land cover", "All"]
FEAT_Y = [0.0, 1.0, 2.0, 3.45, 4.45, 5.45]
REG_DY = {"Alaska": -0.25, "Lena": 0.0, "Canada": 0.25}
COVS = [("tdd", "TDD", "climate"), ("fdd", "FDD", "climate"), ("swe", "Snow (SWE)", "climate"), ("soc", "SOC 0–5 cm", "soil"),
        ("clay", "Clay", "soil"), ("slope", "Slope", "terrain"), ("elev", "Elevation", "terrain"), ("ndvi", "NDVI", "vegetation")]
TERC = ["Low", "Mid", "High"]
DELTA_LB = "ΔRMSE vs recalibrated Stefan (cm)"

# ---------------------------------------------------------------- layout (mm; y from the top edge)
H = 150.0
Y_LEG = 1.4                          # top of the method legend row
Y1 = 11.5                            # row 1 axes top
A_X0, A_W, H1 = 19.0, 41.0, 50.0     # a: forest plot left (room for the region labels), width, height
B_X0, B_G, R_EDGE = 76.0, 7.0, 179.0 # b: first panel left, gap, right edge
HB_UP, HB_GAP, HB_LO = 32.0, 5.0, 13.0   # b: upper (RMSE) height, gap, lower (delta strip) height
Y2, H2 = 88.0, 48.0                  # row 2 axes top, height
C_X0, C_W = 27.0, 45.5
D_X0 = 95.0
D_GREG, D_GGRP = 2.2, 1.6
CB_GAP, CB_W, CB_RIGHT = 3.0, 2.2, 13.0
VMAX = 7.0
CELL_GAP, OUT_LW = 0.5, 0.8
HATCH_EDGE = "#B3B3B3"
GUIDE = "#DADADA"                    # vertical guide lines of the categorical design axis


def method_rows(df, m):
    d = df[df.method == m]
    if m == "R1":
        d = d[(d.lam == MAIN["lam"]) & (d.learner == MAIN["learner"])]
    elif m in ("D0", "D1", "F1"):
        d = d[d.learner == MAIN["learner"]]
    return d


# ---------------------------------------------------------------- data (processing, no plotting)
def data_ab():
    sm = pd.read_csv(RUNS / "core_v6" / "summary_metrics.csv")
    sm = sm[sm.featset == MAIN["featset"]]
    rows = []
    for r in REG:
        for sch in SCHEMES:
            for f in ("clim", "year_obs"):
                for m in B_METH:
                    q = method_rows(sm[(sm.region == r) & (sm.scheme == sch) & (sm.forcing == f)], m)
                    assert len(q) == 1, (r, sch, f, m, len(q))
                    q = q.iloc[0]
                    rows.append(dict(region=r, scheme=sch, forcing=f, method=m, rmse=float(q.rmse), n=int(q.n), n_blocks=int(q.n_blocks),
                                     n_cells=int(q.n_cells), n_seed=int(q.n_seed)))
    t = pd.DataFrame(rows)
    df = pd.read_csv(RUNS / "core_v6" / "deltas_forcing.csv")
    df = df[(df.scheme == "block") & (df.featset == MAIN["featset"]) & (df.pair == "year_obs:clim")]
    dl = []
    for r in REG:
        for m in A_METH:
            q = method_rows(df[df.region == r], m); assert len(q) == 1
            q = q.iloc[0]
            a_ = t[(t.region == r) & (t.scheme == "block") & (t.method == m)].set_index("forcing").rmse
            assert abs((a_["year_obs"] - a_["clim"]) - q.delta) < 1e-6, (r, m)
            dl.append(dict(region=r, method=m, delta=float(q.delta), lo=float(q.lo), hi=float(q.hi), n=int(q.n), n_blocks=int(q.n_blocks)))
    return t, pd.DataFrame(dl)


def data_b_delta():
    """dRMSE(R1 - P1) with 95 % paired block-bootstrap intervals per validation design (year-specific input, main setting)."""
    d = pd.read_csv(RUNS / "core_v6" / "deltas_vs_ref.csv")
    d = d[(d.featset == MAIN["featset"]) & (d.forcing == MAIN["forcing"]) & (d.ref == "P1")]
    d = method_rows(d, "R1")
    rows = []
    for r in REG:
        for sch in SCHEMES:
            q = d[(d.region == r) & (d.scheme == sch)]; assert len(q) == 1, (r, sch, len(q))
            q = q.iloc[0]; rows.append(dict(region=r, scheme=sch, delta=float(q.delta), lo=float(q.lo), hi=float(q.hi), n=int(q.n), n_blocks=int(q.n_blocks)))
    return pd.DataFrame(rows)


def data_c():
    parts = []
    for tag in ("modis_v6", "featsets_v6"):
        d = pd.read_csv(RUNS / tag / "deltas_vs_ref.csv")
        d = d[(d.scheme == "block") & (d.forcing == MAIN["forcing"]) & (d.ref == "P1")]
        parts.append(method_rows(d, "R1").assign(source=f"data/processed/v5/runs/{tag}/deltas_vs_ref.csv"))
    lad = pd.concat(parts).drop_duplicates(["region", "featset"], keep="first")
    lad = lad[lad.featset.isin(FEATS)][["region", "featset", "delta", "lo", "hi", "n", "n_blocks", "source"]]
    assert len(lad) == len(REG) * len(FEATS)
    return lad


def data_d():
    return pd.read_csv(F4E)


def label_years():
    P = RUNS / "core_v6" / "preds"; out = {}
    for r in REG:
        out[r] = {}
        for sch in ("random", "block", "year", "block_year"):
            d = pd.read_csv(P / f"{r}__{sch}__year_obs__F8TS__P1__-__lam-1.csv.gz", usecols=["cell_year_id", "loc_id", "year", "block"]).drop_duplicates("cell_year_id")
            out[r][sch] = dict(first=int(d.year.min()), last=int(d.year.max()), n_years=int(d.year.nunique()), site_years=int(len(d)),
                               sites=int(d.loc_id.nunique()), blocks=int(d.block.nunique()))
    d = pd.read_csv(P / "ALL__region__year_obs__F8TS__P1__-__lam-1.csv.gz", usecols=["region", "cell_year_id", "loc_id", "year"]).drop_duplicates("cell_year_id")
    for r in REG:
        q = d[d.region == r]
        out[r]["region"] = dict(first=int(q.year.min()), last=int(q.year.max()), n_years=int(q.year.nunique()), site_years=int(len(q)), sites=int(q.loc_id.nunique()))
    return out


# ---------------------------------------------------------------- drawing
def draw_a(fig, dforc):
    """Forest plot: three regions (blocks) x three methods, dRMSE year-specific minus mean input with intervals."""
    ax = C.ax_mm(fig, A_X0, Y1, A_W, H1)
    ys = {}; y = 0.0
    for r in REG:
        for m in A_METH:
            ys[(r, m)] = y; y += 1.0
        y += 0.9
    for r in REG:
        for m in A_METH:
            q = dforc[(dforc.region == r) & (dforc.method == m)].iloc[0]; st = C.METH[m]; yy = ys[(r, m)]
            ax.hlines(yy, q.lo, q.hi, color=st["color"], lw=C.LW_CI, zorder=2)
            ax.plot([q.delta], [yy], ls="none", marker=st["marker"], ms=st["ms"], mfc=C.WHITE if st["open"] else st["color"], mec=st["color"],
                    mew=C.MEW if st["open"] else 0.0, zorder=3)
    C.zero_line(ax, vertical=True)
    ytick = [np.mean([ys[(r, m)] for m in A_METH]) for r in REG]
    ax.set_yticks(ytick); ax.set_yticklabels([C.REG_EN[r] for r in REG]); ax.tick_params(axis="y", length=0, pad=2.0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y - 0.9 - 0.45, -0.55)
    lo, hi = float(dforc.lo.min()), float(dforc.hi.max())
    ax.set_xlim(np.floor(lo / 2) * 2 - 0.3, np.ceil(hi / 2) * 2 + 0.3); ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("ΔRMSE, year-specific − mean climate input (cm)")
    return ax


def draw_b(fig, t, dlt):
    q = t[t.forcing == MAIN["forcing"]]
    wb = (R_EDGE - B_X0 - 2 * B_G) / 3
    dodge = {"P0": -0.27, "P1": -0.09, "R1": 0.09, "D0": 0.27}
    axes, axes_lo, clipped = [], [], []
    for j, r in enumerate(REG):
        x0 = B_X0 + j * (wb + B_G)
        ax = C.ax_mm(fig, x0, Y1, wb, HB_UP); axes.append(ax)
        vals = {m: q[(q.region == r) & (q.method == m)].set_index("scheme").reindex(SCHEMES).rmse.values for m in B_METH}
        allv = np.array([vals[m] for m in B_METH]); thr = OVER * np.nanmax(allv[:, :4]); over = allv > thr
        ylo_d, yin = np.nanmin(allv), np.nanmax(allv[~over]); span = yin - ylo_d
        yhi = yin + (0.18 if over.any() else 0.08) * span; ylo = ylo_d - 0.08 * span
        ax.set_ylim(ylo, yhi)
        for xi in range(len(SCHEMES)):                        # categorical guides (no connecting lines between designs)
            ax.axvline(xi, color=GUIDE, lw=0.4, zorder=1)
        for m in B_METH:
            v = vals[m]; ov = v > thr; x = np.arange(len(SCHEMES)) + dodge[m]; st = C.METH[m]; z = 2 + B_METH.index(m) * 0.2
            ax.plot(x[~ov], v[~ov], ls="none", marker=st["marker"], ms=st["ms"], mfc=C.WHITE if st["open"] else st["color"], mec=st["color"],
                    mew=C.MEW if st["open"] else 0.0, zorder=z + 0.1)
            for xi in np.where(ov)[0]:
                ax.plot([x[xi]], [yhi], ls="none", marker=st["marker"], ms=st["ms"], mfc=C.WHITE, mec=st["color"], mew=C.MEW, zorder=5, clip_on=False)
                tx = ax.annotate(C.num(v[xi]), (x[xi], yhi), xytext=(0, 2.4), textcoords="offset points", ha="center", va="bottom",
                                 fontsize=C.FS_TICK, color=C.TEXT, annotation_clip=False)
                tx._qa_skip = True
                clipped.append(dict(region=C.REG_EN[r], method=C.METH[m]["label"], scheme=SCHEME_LB[xi], value=round(float(v[xi]), 2), printed=C.num(v[xi]),
                                    threshold_cm=round(float(thr), 2), axis_top_cm=round(float(yhi), 2)))
        ax.set_xlim(-0.6, len(SCHEMES) - 0.4); ax.set_xticks(range(len(SCHEMES))); ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=True))
        C.facet(ax, C.REG_EN[r])
        # lower strip: dRMSE(R1 - P1) per design with intervals
        axl = C.ax_mm(fig, x0, Y1 + HB_UP + HB_GAP, wb, HB_LO); axes_lo.append(axl)
        d = dlt[dlt.region == r].set_index("scheme").reindex(SCHEMES); st = C.METH["R1"]
        for xi in range(len(SCHEMES)):
            axl.axvline(xi, color=GUIDE, lw=0.4, zorder=1)
        C.zero_line(axl)
        axl.vlines(np.arange(len(SCHEMES)), d.lo.values, d.hi.values, color=st["color"], lw=C.LW_CI, zorder=2)
        axl.plot(np.arange(len(SCHEMES)), d.delta.values, ls="none", marker=st["marker"], ms=st["ms"], mfc=st["color"], mec=st["color"], mew=0.0, zorder=3)
        lo_, hi_ = float(d.lo.min()), float(d.hi.max()); pad = 0.08 * (hi_ - lo_)
        axl.set_ylim(min(lo_ - pad, -0.5), max(hi_ + pad, 0.5)); axl.set_xlim(-0.6, len(SCHEMES) - 0.4)
        axl.set_xticks(range(len(SCHEMES))); axl.set_xticklabels(SCHEME_LB, rotation=45, ha="right", rotation_mode="anchor")
        axl.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], integer=True))
    axes[0].set_ylabel("RMSE (cm)")
    axes_lo[0].set_ylabel("ΔRMSE (cm)", labelpad=2.0)
    return axes, axes_lo, clipped, wb


def draw_c(fig, lad):
    ax = C.ax_mm(fig, C_X0, Y2, C_W, H2)
    for r in REG:
        s = lad[lad.region == r].set_index("featset").reindex(FEATS)
        y = np.array(FEAT_Y) + REG_DY[r]; col = C.REG_HEX[r]
        ax.hlines(y, s.lo.values, s.hi.values, color=col, lw=C.LW_CI, zorder=2)
        ax.plot(s.delta.values, y, ls="none", marker=C.REG_MK[r], ms=C.REG_MS[r], mfc=col, mec=C.WHITE, mew=0.35, zorder=3)
    C.zero_line(ax, vertical=True)
    ax.set_yticks(FEAT_Y); ax.set_yticklabels(FEAT_LB, linespacing=1.05); ax.tick_params(axis="y", length=0, pad=2.0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(FEAT_Y[-1] + 0.55, FEAT_Y[0] - 0.55)
    lo, hi = float(lad.lo.min()), float(lad.hi.max())
    ax.set_xlim(np.floor(lo / 2) * 2 - 0.5, np.ceil(hi / 2) * 2 + 0.5); ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlabel(DELTA_LB)
    return ax


def hatch_rect(ax, x, y, w, h, zorder=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=HATCH_EDGE, hatch="//////", lw=0, zorder=zorder))


def draw_d(fig, t):
    from cmcrameri import cm as cmc
    cmap = cmc.vik; norm = plt.Normalize(-VMAX, VMAX)
    hm_w = R_EDGE - CB_RIGHT - D_X0
    wc = (hm_w - 2 * D_GREG) / 9
    groups = [g for _, _, g in COVS]; ngap = sum(1 for i in range(1, len(COVS)) if groups[i] != groups[i - 1])
    hc = (H2 - ngap * D_GGRP) / len(COVS)
    ax = C.ax_mm(fig, D_X0, Y2, hm_w, H2); ax.set_xlim(0, hm_w); ax.set_ylim(H2, 0); ax.axis("off")
    ys, y = [], 0.0
    for i in range(len(COVS)):
        if i > 0 and groups[i] != groups[i - 1]:
            y += D_GGRP
        ys.append(y); y += hc
    printed, outlined, hatched = [], [], []
    g = CELL_GAP / 2
    for j, r in enumerate(REG):
        xr = j * (3 * wc + D_GREG)
        for i, (key, lab, _) in enumerate(COVS):
            for k in range(3):
                z = t[(t.region == r) & (t.covariate == key) & (t.tercile == k)].iloc[0]
                x = xr + k * wc; y0 = ys[i]
                if bool(z.shown):
                    v = float(z.delta); col = cmap(norm(v))
                    ax.add_patch(Rectangle((x + g, y0 + g), wc - 2 * g, hc - 2 * g, facecolor=col, edgecolor="none", lw=0, zorder=1))
                    s = C.num(v, plus=True)
                    ax.text(x + wc / 2, y0 + hc / 2, s, ha="center", va="center_baseline", fontsize=C.FS_CELL, color=C.on_colour(col), zorder=3)
                    printed.append(dict(region=C.REG_EN[r], covariate=lab, tercile=TERC[k], value=round(v, 3), printed=s, lo=round(float(z.lo), 3),
                                        hi=round(float(z.hi), 3), n=int(z.n), blocks=int(z.n_blocks), excludes_0=bool(z.excludes_0)))
                    if bool(z.excludes_0):
                        ax.add_patch(Rectangle((x + g, y0 + g), wc - 2 * g, hc - 2 * g, facecolor="none", edgecolor=C.TEXT, lw=OUT_LW, zorder=2, joinstyle="miter"))
                        outlined.append(f"{C.REG_EN[r]} {lab} {TERC[k]}")
                else:
                    hatch_rect(ax, x + g, y0 + g, wc - 2 * g, hc - 2 * g)
                    hatched.append(dict(region=C.REG_EN[r], covariate=lab, tercile=TERC[k], n=int(z.n), blocks=int(z.n_blocks)))
        ax.text(xr + 1.5 * wc, -1.3, C.REG_EN[r], ha="center", va="baseline", fontsize=C.FS, color=C.TEXT, clip_on=False)
        for k in range(3):
            ax.text(xr + (k + 0.5) * wc, H2 + 1.0, TERC[k], ha="center", va="top", fontsize=C.FS_TICK, color=C.TEXT, clip_on=False)
    for i, (_, lab, _) in enumerate(COVS):
        ax.text(-1.2, ys[i] + hc / 2, lab, ha="right", va="center_baseline", fontsize=C.FS_TICK, color=C.TEXT, clip_on=False)
    ax.text(hm_w / 2, H2 + 4.6, "Tercile within region", ha="center", va="top", fontsize=C.FS, color=C.TEXT, clip_on=False)
    xb = D_X0 + hm_w + CB_GAP
    cax = C.ax_mm(fig, xb, Y2, CB_W, H2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    tk = np.arange(-6, 6.01, 3); cb.set_ticks(tk); cb.set_ticklabels([C.num(v, 0, plus=True) for v in tk]); cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=C.FS_TICK, length=2.0, width=C.LW_AXIS, pad=1.5, color=C.AXIS, labelcolor=C.TEXT)
    cb.outline.set_linewidth(C.LW_AXIS); cb.outline.set_edgecolor(C.AXIS)
    cb.set_label("ΔRMSE (cm)", fontsize=C.FS, labelpad=2.0, color=C.TEXT)
    ky = Y2 + H2 + 9.6; kw_, kh_ = 4.2, 2.6
    kax = C.ax_mm(fig, D_X0, ky, hm_w, kh_); kax.set_xlim(0, hm_w); kax.set_ylim(kh_, 0); kax.axis("off")
    kax.add_patch(Rectangle((0, 0), kw_, kh_, facecolor=cmap(norm(0.0)), edgecolor=C.TEXT, lw=OUT_LW, joinstyle="miter"))
    t1 = kax.text(kw_ + 1.2, kh_ / 2, "95% interval excludes 0", ha="left", va="center_baseline", fontsize=C.FS_TICK, color=C.TEXT)
    fig.canvas.draw(); bb = t1.get_window_extent(fig.canvas.get_renderer()); x2 = kw_ + 1.2 + bb.width * 25.4 / fig.dpi + 4.5
    hatch_rect(kax, x2, 0.0, kw_, kh_)
    kax.text(x2 + kw_ + 1.2, kh_ / 2, "Insufficient data", ha="left", va="center_baseline", fontsize=C.FS_TICK, color=C.TEXT)
    return ax, cax, dict(printed=printed, outlined=outlined, hatched=hatched, cell_mm=[round(wc, 2), round(hc, 2)], vmax=VMAX)


def main():
    C.setup()
    t, dforc = data_ab(); dlt = data_b_delta(); lad = data_c(); tg = data_d(); years = label_years()
    sh = tg[tg.shown & (tg.covariate != "all")]
    assert np.nanmax(np.abs(sh.delta)) <= VMAX, "colour limit below the data range"
    fig = C.new_fig(H)
    ax_a = draw_a(fig, dforc)
    b_axes, b_lo, clipped, wb = draw_b(fig, t, dlt)
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); px = 25.4 / fig.dpi
    b_mid = B_X0 + (3 * wb + 2 * B_G) / 2
    yb = H - min(lb.get_window_extent(rd).y0 for a in b_lo for lb in a.get_xticklabels()) * px + 1.0
    C.text_mm(fig, b_mid, yb, "Validation design", ha="center", va="top")
    hs = [C.method_handle(m) for m in ("P0", "P1", "R1", "D0")]
    fig.legend(hs, [C.METH[m]["label"] for m in ("P0", "P1", "R1", "D0")], loc="upper left", bbox_to_anchor=(A_X0 / C.W, 1 - Y_LEG / H),
               ncol=4, frameon=False, fontsize=C.FS, handlelength=2.6, handletextpad=0.6, columnspacing=2.2, borderaxespad=0, borderpad=0)
    ax_c = draw_c(fig, lad)
    ax_d, cax, dinfo = draw_d(fig, tg)
    rh = [Line2D([], [], ls="none", marker=C.REG_MK[r], ms=C.REG_MS[r], mfc=C.REG_HEX[r], mec=C.WHITE, mew=0.35) for r in REG]
    leg = ax_c.legend(rh, [C.REG_EN[r] for r in REG], loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1, frameon=True, fontsize=C.FS,
                      handlelength=0.8, handletextpad=0.5, labelspacing=0.45, borderaxespad=0.3, borderpad=0.35)
    leg.get_frame().set(facecolor="white", edgecolor="none", alpha=1.0, linewidth=0)
    yb1 = Y1 - 1.3; yb2 = Y2 - 1.3
    C.letter(fig, 0.8, yb1, "a"); C.letter(fig, B_X0 - 11.5, yb1, "b"); C.letter(fig, 0.8, yb2, "c"); C.letter(fig, D_X0 - 15.0, yb2, "d")
    probs = C.qa_report(fig, STEM)
    margins = C.ink_margins_mm(fig); print("ink margins (mm):", margins)
    outs = C.export(fig, STEM)
    blk = t[(t.scheme == "block")]
    src = {
        "figure": "Fig. 3", "version": "v14 (2026-10-20): design A; a forest plot of the input effect, b points per design with a dRMSE strip; numbers as v13",
        "files": [str(p.relative_to(ROOT)) for p in outs], "script": "scripts/4_visualization/paper_v15/fig3_accuracy.py",
        "size_mm": [C.W, H], "fonts_pt": {"axis_labels_legends": C.FS, "ticks_direct_labels": C.FS_TICK, "heat_map_values": C.FS_CELL, "panel_letters_bold": C.FS_LETTER},
        "style": {"palette": "design/style_tokens_v4.json color.paper_v13 (POLAR_PALETTE=paper13)", "text": C.TEXT, "axes": C.AXIS,
                  "methods": {C.METH[m]["label"]: C.METH[m]["color"] for m in B_METH}, "regions": {C.REG_EN[r]: [C.REG_HEX[r], C.REG_MK[r]] for r in REG}},
        "qa_problems": probs, "ink_margins_mm": margins,
        "main_setting": "spatial block CV (0.5 degree blocks, 5 folds), climate input year_obs (ERA5-Land thaw degree-days accumulated to the observation day), "
                        "featset F8TS (8 climate + 6 terrain + 9 soil), CatBoost, R1 lambda 0.5, 3 seeds averaged; intervals = 95 % paired block bootstrap (2000 resamples)",
        "label_years_by_region_and_split": {C.REG_EN[r]: v for r, v in years.items()},
        "panels": {
            "a": {"quantity": "dRMSE = RMSE(year_obs) - RMSE(clim), spatial block, with 95 % paired block-bootstrap interval",
                  "input_files": ["data/processed/v5/runs/core_v6/deltas_forcing.csv", "data/processed/v5/runs/core_v6/summary_metrics.csv (consistency check)"],
                  "rows": [dict(region=C.REG_EN[q.region], method=C.METH[q.method]["label"], delta=round(q.delta, 3), lo=round(q.lo, 3), hi=round(q.hi, 3), n=int(q.n),
                                blocks=int(q.n_blocks)) for q in dforc.itertuples()],
                  "block_rmse_cm": [dict(region=C.REG_EN[q.region], method=C.METH[q.method]["label"], forcing=q.forcing, rmse=round(q.rmse, 3), n=int(q.n)) for q in blk.itertuples()]},
            "b": {"quantity_upper": "pooled out-of-fold RMSE (cm) per validation design, year-specific input",
                  "quantity_lower": "dRMSE = RMSE(Physics + residual ML) - RMSE(Stefan recalibrated) per design with 95 % interval",
                  "input_files": ["data/processed/v5/runs/core_v6/summary_metrics.csv", "data/processed/v5/runs/core_v6/deltas_vs_ref.csv"],
                  "rmse": [dict(region=C.REG_EN[q.region], scheme=q.scheme, method=C.METH[q.method]["label"], rmse=round(q.rmse, 3), n=int(q.n), blocks=int(q.n_blocks))
                           for q in t[t.forcing == MAIN["forcing"]].itertuples()],
                  "delta": [dict(region=C.REG_EN[q.region], scheme=q.scheme, delta=round(q.delta, 3), lo=round(q.lo, 3), hi=round(q.hi, 3), n=int(q.n)) for q in dlt.itertuples()],
                  "values_beyond_axis": clipped, "over_threshold_rule": f"value > {OVER} x max of the four non-region designs"},
            "c": {"quantity": "dRMSE (R1 - P1) per covariate set, spatial block, 95 % intervals",
                  "input_files": sorted(set(lad.source)), "rows": [dict(region=C.REG_EN[q.region], featset=q.featset, delta=round(q.delta, 3), lo=round(q.lo, 3),
                                                                        hi=round(q.hi, 3), n=int(q.n), blocks=int(q.n_blocks)) for q in lad.itertuples()]},
            "d": {"quantity": "dRMSE (R1 - P1) within regional terciles of eight covariates", "input_files": [str(F4E.relative_to(ROOT))], **dinfo},
        },
    }
    C.write_json(C.SRC / f"{STEM}.json", src)
    print("done", [str(p) for p in outs])


if __name__ == "__main__":
    main()
