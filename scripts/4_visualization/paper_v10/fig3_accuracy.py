# -*- coding: utf-8 -*-
"""Paper v10 Fig. 3: accuracy of physics + residual ML and the conditions of its gain (Scientific Reports, 180 mm).

a  Spatial-block RMSE with mean climate input (2011-2024 ERA5-Land mean, open markers) and year-specific climate input
   (thaw degree-days to the observation day, filled markers), Stefan (recalibrated), Physics + residual ML, Direct ML; one panel per region.
   Same y span (cm) in the three panels so that the changes can be compared.
b  RMSE by validation split (random, spatial block, year, block x year, region hold-out), four methods, year-specific climate input.
   Values above 1.3 x the largest value of the four non-region splits are drawn on the top edge with the value printed.
c  dRMSE = RMSE(Physics + residual ML) - RMSE(Stefan, recalibrated) for six covariate sets, spatial block, 95 % block-bootstrap intervals.
d  The same dRMSE within regional terciles of eight covariates (heat map, cmcrameri vik symmetric about 0; black outline = 95 % interval
   excludes 0; light hatching = fewer than 30 cell-years or 5 blocks).
Inputs (numbers only from files): data/processed/v5/runs/core_v6/{summary_metrics,deltas_forcing}.csv, runs/{modis_v6,featsets_v6}/deltas_vs_ref.csv,
outputs/figures/paper/v7/source_data/F4e_gain_conditions.csv, label years from runs/core_v6/preds (P1 out-of-fold files).
Main setting: forcing year_obs, featset F8TS, learner catboost, R1 lam 0.5, scheme block.
Output: outputs/figures/paper/v10/Fig3_accuracy.{pdf,svg,png} and source_data/Fig3_accuracy.json.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/fig3_accuracy.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig34_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator, MaxNLocator, FixedLocator  # noqa: E402

ROOT = C.ROOT; RUNS = ROOT / "data" / "processed" / "v5" / "runs"
F4E = ROOT / "outputs" / "figures" / "paper" / "v7" / "source_data" / "F4e_gain_conditions.csv"
STEM = "Fig3_accuracy"
REG = C.REG
MAIN = dict(forcing="year_obs", featset="F8TS", learner="catboost", lam=0.5, scheme="block")
A_METH = ["P1", "R1", "D0"]
B_METH = ["P0", "P1", "D0", "R1"]                        # drawing order (Physics + residual ML on top)
SCHEMES = ["random", "block", "year", "block_year", "region"]
SCHEME_LB = ["Random", "Spatial block", "Year", "Block × year", "Region"]
OVER = 1.3
FEATS = ["F8", "F8T", "F8TS", "F8TS_M", "F8TS_X", "F8TS_ALL"]
FEAT_LB = ["Climate only", "+ Terrain", "+ Soil (main set)", "+ MODIS", "+ High-resolution soil\nand land cover", "All"]
FEAT_Y = [0.0, 1.0, 2.0, 3.45, 4.45, 5.45]                # gap between the cumulative ladder and the additions to the main set
REG_DY = {"Alaska": -0.25, "Lena": 0.0, "Canada": 0.25}
COVS = [("tdd", "TDD", "climate"), ("fdd", "FDD", "climate"), ("swe", "Snow (SWE)", "climate"), ("soc", "SOC 0–5 cm", "soil"),
        ("clay", "Clay", "soil"), ("slope", "Slope", "terrain"), ("elev", "Elevation", "terrain"), ("ndvi", "NDVI", "vegetation")]
TERC = ["Low", "Mid", "High"]

# ---------------------------------------------------------------- layout (mm; y from the top edge)
H = 142.0
Y_LEG = 1.2                          # top of the method legend row
Y1, H1 = 11.5, 46.0                  # row 1 axes top, height
A_X0, A_W, A_G = 10.0, 18.5, 6.0     # a: first axes left, width, gap
B_GAP0, B_G, R_EDGE = 12.0, 7.0, 179.0
Y2, H2 = 80.0, 48.0                  # row 2 axes top, height (c axes = heat-map cell block)
C_X0, C_W = 27.0, 45.5               # c axes left, width (row labels to the left)
D_X0 = 95.0                          # heat-map left (row labels to the left)
D_GREG, D_GGRP = 2.2, 1.6            # gaps between region blocks, between covariate groups
CB_GAP, CB_W, CB_RIGHT = 3.0, 2.2, 13.0   # colour bar gap, width; room for colour bar + ticks + label to R_EDGE
VMAX = 7.0                           # colour limit (cm), symmetric; checked against the data below
CELL_GAP, OUT_LW = 0.4, 0.8          # white gap between heat-map cells (mm), outline width (pt)


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
    t = pd.read_csv(F4E)
    return t


def label_years():
    """Label years per region and split from the out-of-fold Stefan (recalibrated) prediction files of the main run."""
    P = RUNS / "core_v6" / "preds"; out = {}
    for r in REG:
        out[r] = {}
        for sch in ("random", "block", "year", "block_year"):
            d = pd.read_csv(P / f"{r}__{sch}__year_obs__F8TS__P1__-__lam-1.csv.gz", usecols=["cell_year_id", "loc_id", "year", "block"]).drop_duplicates("cell_year_id")
            out[r][sch] = dict(first=int(d.year.min()), last=int(d.year.max()), n_years=int(d.year.nunique()), cell_years=int(len(d)),
                               cells=int(d.loc_id.nunique()), blocks=int(d.block.nunique()))
    d = pd.read_csv(P / "ALL__region__year_obs__F8TS__P1__-__lam-1.csv.gz", usecols=["region", "cell_year_id", "loc_id", "year"]).drop_duplicates("cell_year_id")
    for r in REG:
        q = d[d.region == r]
        out[r]["region"] = dict(first=int(q.year.min()), last=int(q.year.max()), n_years=int(q.year.nunique()), cell_years=int(len(q)), cells=int(q.loc_id.nunique()))
    return out


# ---------------------------------------------------------------- drawing
def nice_span(vals, factor=1.15, step=2.0):
    return step * np.ceil(factor * vals / step)


def draw_a(fig, t):
    q = t[(t.scheme == "block")]
    rng = {r: (q[q.region == r].pipe(lambda z: z[z.method.isin(A_METH)]).rmse.min(), q[q.region == r].pipe(lambda z: z[z.method.isin(A_METH)]).rmse.max()) for r in REG}
    span = float(nice_span(max(hi - lo for lo, hi in rng.values())))
    dodge = {"P1": -0.10, "R1": 0.0, "D0": 0.10}
    axes = []
    for j, r in enumerate(REG):
        ax = C.ax_mm(fig, A_X0 + j * (A_W + A_G), Y1, A_W, H1); axes.append(ax)
        for m in A_METH:
            v = q[(q.region == r) & (q.method == m)].set_index("forcing").rmse
            x = np.array([0.0, 1.0]) + dodge[m]; y = np.array([v["clim"], v["year_obs"]])
            C.plot_method(ax, x, y, m, marker=False, zorder=2 + A_METH.index(m) * 0.1)
            st = C.METH[m]
            ax.plot(x[:1], y[:1], ls="none", marker=st["marker"], ms=st["ms"], mfc="white", mec=st["color"], mew=C.MEW, zorder=4)
            ax.plot(x[1:], y[1:], ls="none", marker=st["marker"], ms=st["ms"], mfc=st["color"], mec=st["color"], mew=C.MEW, zorder=4)
        lo, hi = rng[r]; c = 0.5 * (lo + hi); ylo = np.floor((c - span / 2) * 2) / 2
        ax.set_ylim(ylo, ylo + span); ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlim(-0.42, 1.42); ax.set_xticks([0, 1]); ax.set_xticklabels(["2011–2024\nmean", "Obs.\nyear"], linespacing=1.0)
        C.facet(ax, C.REG_EN[r])
    axes[0].set_ylabel("RMSE, spatial block (cm)")
    return axes, span


def draw_b(fig, t, x0):
    q = t[t.forcing == MAIN["forcing"]]
    wb = (R_EDGE - x0 - 2 * B_G) / 3
    dodge = {"P0": -0.24, "P1": -0.08, "R1": 0.08, "D0": 0.24}
    axes, clipped = [], []
    for j, r in enumerate(REG):
        ax = C.ax_mm(fig, x0 + j * (wb + B_G), Y1, wb, H1); axes.append(ax)
        vals = {m: q[(q.region == r) & (q.method == m)].set_index("scheme").reindex(SCHEMES).rmse.values for m in B_METH}
        allv = np.array([vals[m] for m in B_METH]); thr = OVER * np.nanmax(allv[:, :4]); over = allv > thr
        ylo_d, yin = np.nanmin(allv), np.nanmax(allv[~over]); span = yin - ylo_d
        yhi = yin + (0.16 if over.any() else 0.07) * span; ylo = ylo_d - 0.07 * span
        ax.set_ylim(ylo, yhi)
        for m in B_METH:
            v = vals[m]; ov = v > thr; vc = np.where(ov, yhi, v); x = np.arange(len(SCHEMES)) + dodge[m]
            st = C.METH[m]; z = 2 + B_METH.index(m) * 0.2
            ax.plot(x, vc, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=z, clip_on=False)
            ax.plot(x[~ov], v[~ov], ls="none", marker=st["marker"], ms=st["ms"], mfc="white" if st["open"] else st["color"], mec=st["color"],
                    mew=C.MEW if st["open"] else 0.0, zorder=z + 0.1)
            for xi in np.where(ov)[0]:
                ax.plot([x[xi]], [yhi], ls="none", marker=st["marker"], ms=st["ms"], mfc="white", mec=st["color"], mew=C.MEW, zorder=5, clip_on=False)
                tx = ax.annotate(C.num(v[xi]), (x[xi], yhi), xytext=(0, 2.4), textcoords="offset points", ha="center", va="bottom",
                                 fontsize=C.FS_TICK, color=C.BLACK, annotation_clip=False)
                tx._qa_skip = True        # sits on its own marker by design
                clipped.append(dict(region=C.REG_EN[r], method=C.METH[m]["label"], scheme=SCHEME_LB[xi], value=round(float(v[xi]), 2), printed=C.num(v[xi]),
                                    threshold_cm=round(float(thr), 2), axis_top_cm=round(float(yhi), 2)))
        ax.set_xlim(-0.55, len(SCHEMES) - 0.45); ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels(SCHEME_LB, rotation=45, ha="right", rotation_mode="anchor")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=True))
        C.facet(ax, C.REG_EN[r])
    axes[0].set_ylabel("RMSE (cm)")
    return axes, clipped, wb


def draw_c(fig, lad):
    ax = C.ax_mm(fig, C_X0, Y2, C_W, H2)
    for r in REG:
        s = lad[lad.region == r].set_index("featset").reindex(FEATS)
        y = np.array(FEAT_Y) + REG_DY[r]; col = C.REG_HEX[r]
        ax.hlines(y, s.lo.values, s.hi.values, color=col, lw=C.LW_CI, zorder=2)
        ax.plot(s.delta.values, y, ls="none", marker=C.REG_MK[r], ms=C.REG_MS[r], mfc=col, mec=col, mew=0, zorder=3)
    C.zero_line(ax, vertical=True)
    ax.set_yticks(FEAT_Y); ax.set_yticklabels(FEAT_LB, linespacing=1.05); ax.tick_params(axis="y", length=0, pad=2.0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(FEAT_Y[-1] + 0.55, FEAT_Y[0] - 0.55)
    lo, hi = float(lad.lo.min()), float(lad.hi.max())
    ax.set_xlim(np.floor(lo / 2) * 2 - 0.5, np.ceil(hi / 2) * 2 + 0.5); ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlabel("ΔRMSE vs Stefan, recal. (cm)")
    return ax


def hatch_rect(ax, x, y, w, h, zorder=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#BDBDBD", hatch="//////", lw=0, zorder=zorder))


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
    for j, r in enumerate(REG):
        xr = j * (3 * wc + D_GREG)
        for i, (key, lab, _) in enumerate(COVS):
            for k in range(3):
                z = t[(t.region == r) & (t.covariate == key) & (t.tercile == k)].iloc[0]
                x = xr + k * wc; y0 = ys[i]
                g = CELL_GAP / 2
                if bool(z.shown):
                    v = float(z.delta); col = cmap(norm(v))
                    ax.add_patch(Rectangle((x + g, y0 + g), wc - 2 * g, hc - 2 * g, facecolor=col, edgecolor="none", lw=0, zorder=1))
                    lum = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
                    s = C.num(v, plus=True)
                    ax.text(x + wc / 2, y0 + hc / 2, s, ha="center", va="center_baseline", fontsize=C.FS_CELL,
                            color="white" if lum < 0.42 else C.BLACK, zorder=3)
                    printed.append(dict(region=C.REG_EN[r], covariate=lab, tercile=TERC[k], value=round(v, 3), printed=s, lo=round(float(z.lo), 3),
                                        hi=round(float(z.hi), 3), n=int(z.n), blocks=int(z.n_blocks), excludes_0=bool(z.excludes_0)))
                    if bool(z.excludes_0):                       # outline just inside the white gap (visible on dark cells too)
                        e = g + OUT_LW * 25.4 / 72 / 2
                        ax.add_patch(Rectangle((x + e, y0 + e), wc - 2 * e, hc - 2 * e, facecolor="none", edgecolor=C.BLACK, lw=OUT_LW, zorder=2,
                                               joinstyle="miter"))
                        outlined.append(f"{C.REG_EN[r]} {lab} {TERC[k]}")
                else:
                    hatch_rect(ax, x + g, y0 + g, wc - 2 * g, hc - 2 * g)
                    hatched.append(dict(region=C.REG_EN[r], covariate=lab, tercile=TERC[k], n=int(z.n), blocks=int(z.n_blocks)))
        ax.text(xr + 1.5 * wc, -1.3, C.REG_EN[r], ha="center", va="baseline", fontsize=C.FS, color=C.BLACK, clip_on=False)
        for k in range(3):
            ax.text(xr + (k + 0.5) * wc, H2 + 1.0, TERC[k], ha="center", va="top", fontsize=C.FS_TICK, color=C.BLACK, clip_on=False)
    for i, (_, lab, _) in enumerate(COVS):
        ax.text(-1.2, ys[i] + hc / 2, lab, ha="right", va="center_baseline", fontsize=C.FS_TICK, color=C.BLACK, clip_on=False)
    ax.text(hm_w / 2 - 0 * D_GREG, H2 + 4.6, "Tercile within region", ha="center", va="top", fontsize=C.FS, color=C.BLACK, clip_on=False)
    # colour bar
    xb = D_X0 + hm_w + CB_GAP
    cax = C.ax_mm(fig, xb, Y2, CB_W, H2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    tk = np.arange(-6, 6.01, 3); cb.set_ticks(tk); cb.set_ticklabels([C.num(v, 0, plus=True) for v in tk]); cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=C.FS_TICK, length=2.0, width=C.LW_AXIS, pad=1.5); cb.outline.set_linewidth(C.LW_AXIS)
    cb.set_label("ΔRMSE (cm)", fontsize=C.FS, labelpad=2.0)
    # key row: outlined cell, hatched cell
    ky = Y2 + H2 + 9.6; kw_, kh_ = 4.2, 2.6
    kax = C.ax_mm(fig, D_X0, ky, hm_w, kh_); kax.set_xlim(0, hm_w); kax.set_ylim(kh_, 0); kax.axis("off")
    kax.add_patch(Rectangle((0, 0), kw_, kh_, facecolor=cmap(norm(-3.0)), edgecolor="none"))
    e = OUT_LW * 25.4 / 72 / 2
    kax.add_patch(Rectangle((e, e), kw_ - 2 * e, kh_ - 2 * e, facecolor="none", edgecolor=C.BLACK, lw=OUT_LW, joinstyle="miter"))
    t1 = kax.text(kw_ + 1.2, kh_ / 2, "95% CI excludes 0", ha="left", va="center_baseline", fontsize=C.FS_TICK, color=C.BLACK)
    fig.canvas.draw(); bb = t1.get_window_extent(fig.canvas.get_renderer()); x2 = kw_ + 1.2 + bb.width * 25.4 / fig.dpi + 4.5
    hatch_rect(kax, x2, 0.0, kw_, kh_)
    kax.add_patch(Rectangle((x2, 0.0), kw_, kh_, facecolor="none", edgecolor="#BDBDBD", lw=0.4))
    kax.text(x2 + kw_ + 1.2, kh_ / 2, "Insufficient data", ha="left", va="center_baseline", fontsize=C.FS_TICK, color=C.BLACK)
    return ax, cax, dict(printed=printed, outlined=outlined, hatched=hatched, cell_mm=[round(wc, 2), round(hc, 2)], vmax=VMAX)


def main():
    C.setup()
    t, dforc = data_ab(); lad = data_c(); tg = data_d(); years = label_years()
    sh = tg[tg.shown & (tg.covariate != "all")]
    assert np.nanmax(np.abs(sh.delta)) <= VMAX, "colour limit below the data range"
    fig = C.new_fig(H)
    # ---- row 1
    a_axes, span_a = draw_a(fig, t)
    b_x0 = A_X0 + 3 * A_W + 2 * A_G + B_GAP0
    b_axes, clipped, wb = draw_b(fig, t, b_x0)
    # x-axis labels of a and b on one line below the lowest tick label of b
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); px = 25.4 / fig.dpi
    def below(axes):        # top of an x-axis label 1.0 mm below the lowest tick label of the panel group (mm from the top)
        return H - min(lb.get_window_extent(rd).y0 for a in axes for lb in a.get_xticklabels()) * px + 1.0
    a_mid = A_X0 + (3 * A_W + 2 * A_G) / 2; b_mid = b_x0 + (3 * wb + 2 * B_G) / 2
    C.text_mm(fig, a_mid, below(a_axes), "Climate input", ha="center", va="top")
    C.text_mm(fig, b_mid, below(b_axes), "Validation split", ha="center", va="top")
    # method legend (one row, top)
    hs = [C.method_handle(m) for m in ("P0", "P1", "R1", "D0")]
    fig.legend(hs, [C.METH[m]["label"] for m in ("P0", "P1", "R1", "D0")], loc="upper left", bbox_to_anchor=(A_X0 / C.W, 1 - Y_LEG / H),
               ncol=4, frameon=False, fontsize=C.FS, handlelength=2.4, handletextpad=0.5, columnspacing=1.8, borderaxespad=0, borderpad=0)
    # ---- row 2
    ax_c = draw_c(fig, lad)
    ax_d, cax, dinfo = draw_d(fig, tg)
    rh = [Line2D([], [], ls="none", marker=C.REG_MK[r], ms=C.REG_MS[r], mfc=C.REG_HEX[r], mec=C.REG_HEX[r], mew=0) for r in REG]
    ax_c.legend(rh, [C.REG_EN[r] for r in REG], loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1, frameon=False, fontsize=C.FS,
                handlelength=0.8, handletextpad=0.5, labelspacing=0.45, borderaxespad=0.6, borderpad=0)
    # ---- panel letters (baseline = facet-label baseline)
    yb1 = Y1 - 1.3; yb2 = Y2 - 1.3
    C.letter(fig, 0.8, yb1, "a"); C.letter(fig, b_x0 - 11.5, yb1, "b"); C.letter(fig, 0.8, yb2, "c"); C.letter(fig, D_X0 - 15.0, yb2, "d")
    probs = C.qa_report(fig, STEM)
    margins = C.ink_margins_mm(fig); print("ink margins (mm):", margins)
    outs = C.export(fig, STEM)
    # ---- source data
    blk = t[(t.scheme == "block")]
    src = {
        "figure": "Fig. 3", "files": [str(p.relative_to(ROOT)) for p in outs], "script": "scripts/4_visualization/paper_v10/fig3_accuracy.py",
        "size_mm": [C.W, H], "fonts_pt": {"axis_labels_legends": C.FS, "ticks_direct_labels": C.FS_TICK, "heat_map_values": C.FS_CELL, "panel_letters_bold": C.FS_LETTER},
        "qa_problems": probs, "ink_margins_mm": margins,
        "main_setting": "spatial block CV (0.5 degree blocks, 5 folds), climate input year_obs (ERA5-Land thaw degree-days accumulated to the observation day of the label year), featset F8TS (8 climate + terrain + soil covariates), CatBoost, Physics + residual ML lambda 0.5, 3 seeds averaged",
        "label_years_by_region_and_split": {C.REG_EN[r]: v for r, v in years.items()},
        "panels": {
            "a": {
                "quantity": "pooled out-of-fold RMSE (cm), spatial block CV",
                "input_files": ["data/processed/v5/runs/core_v6/summary_metrics.csv", "data/processed/v5/runs/core_v6/deltas_forcing.csv (delta check, not printed)"],
                "row_filter": "scheme == block, featset == F8TS, forcing in {clim, year_obs}; P1 (learner '-'), R1 (catboost, lam 0.5), D0 (catboost)",
                "label_years": {C.REG_EN[r]: f"{years[r]['block']['first']}-{years[r]['block']['last']}" for r in REG},
                "climate_input_years": {"clim (open markers, '2011-2024 mean')": "ERA5-Land 2011-2024 mean of each covariate per cell",
                                        "year_obs (filled markers, 'Obs. year')": "ERA5-Land of the label year, TDD accumulated to the observation day"},
                "y_span_cm_all_panels": span_a,
                "plotted_rmse_cm": {C.REG_EN[r]: {C.METH[m]["label"]: {"mean_2011_2024": round(float(blk[(blk.region == r) & (blk.method == m) & (blk.forcing == "clim")].rmse.iloc[0]), 2),
                                                                     "obs_year": round(float(blk[(blk.region == r) & (blk.method == m) & (blk.forcing == "year_obs")].rmse.iloc[0]), 2)}
                                                  for m in A_METH} for r in REG},
                "change_obs_year_minus_mean_cm_95ci_not_printed": {C.REG_EN[r]: {C.METH[m]["label"]: [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)]
                                                                                 for m in A_METH for z in [dforc[(dforc.region == r) & (dforc.method == m)].iloc[0]]} for r in REG},
                "n_cell_years": {C.REG_EN[r]: int(blk[(blk.region == r)].n.iloc[0]) for r in REG},
                "printed_numbers": "none (tick labels only)"},
            "b": {
                "quantity": "pooled out-of-fold RMSE (cm) by validation split, climate input year_obs",
                "input_files": ["data/processed/v5/runs/core_v6/summary_metrics.csv"],
                "row_filter": "forcing == year_obs, featset == F8TS, scheme in {random, block, year, block_year, region}; P0, P1 (learner '-'), R1 (catboost, lam 0.5), D0 (catboost)",
                "label_years": {C.REG_EN[r]: {s: f"{years[r][s]['first']}-{years[r][s]['last']} ({years[r][s]['n_years']} years, {years[r][s]['cell_years']} cell-years)" for s in SCHEMES} for r in REG},
                "climate_input_years": "ERA5-Land of the label year, TDD to the observation day",
                "note": "region hold-out: no in-region labels to recalibrate, so Stefan (recalibrated) uses the source coefficient and equals Stefan (source coeff.)",
                "plotted_rmse_cm": {C.REG_EN[r]: {C.METH[m]["label"]: {s: round(float(t[(t.region == r) & (t.method == m) & (t.scheme == s) & (t.forcing == "year_obs")].rmse.iloc[0]), 2) for s in SCHEMES}
                                                  for m in B_METH} for r in REG},
                "n_cell_years": {C.REG_EN[r]: {s: int(t[(t.region == r) & (t.scheme == s) & (t.forcing == "year_obs")].n.iloc[0]) for s in SCHEMES} for r in REG},
                "out_of_range_rule": f"values > {OVER} x the largest value of the random/block/year/block x year splits in that panel are drawn on the top edge (open marker) with the value printed",
                "printed_numbers": clipped},
            "c": {
                "quantity": "dRMSE = RMSE(Physics + residual ML) - RMSE(Stefan, recalibrated) (cm), spatial block, 95 % block-bootstrap interval",
                "input_files": ["data/processed/v5/runs/modis_v6/deltas_vs_ref.csv", "data/processed/v5/runs/featsets_v6/deltas_vs_ref.csv"],
                "row_filter": "scheme == block, forcing == year_obs, ref == P1, method R1, learner catboost, lam 0.5; modis_v6 (MODIS 2010-2024) preferred where both files have the featset (F8TS, F8TS_M, F8TS_ALL), featsets_v6 for F8, F8T, F8TS_X",
                "featsets": dict(zip(FEATS, [f.replace("\n", " ") for f in FEAT_LB])),
                "featset_definitions": {"F8": "8 ERA5-Land climate covariates", "F8T": "F8 + terrain", "F8TS": "F8 + terrain + SoilGrids (main set)",
                                        "F8TS_M": "main set + MODIS (summer NDVI, EVI, snow-off day, snow days; 2010-2024)", "F8TS_X": "main set + NCSCD SOC, CAVM class, WorldCover fractions",
                                        "F8TS_ALL": "main set + MODIS + previous-year climate + hi-res soil and land cover"},
                "label_years": {C.REG_EN[r]: f"{years[r]['block']['first']}-{years[r]['block']['last']}" for r in REG},
                "climate_input_years": "year_obs (label year, TDD to the observation day)",
                "plotted_delta_cm_95ci": {C.REG_EN[r]: {FEAT_LB[i].replace("\n", " "): [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)]
                                                        for i, f in enumerate(FEATS) for z in [lad[(lad.region == r) & (lad.featset == f)].iloc[0]]} for r in REG},
                "row_source": {C.REG_EN[r]: {f: lad[(lad.region == r) & (lad.featset == f)].source.iloc[0] for f in FEATS} for r in REG},
                "printed_numbers": "none (tick labels only)"},
            "d": {
                "quantity": "dRMSE = RMSE(Physics + residual ML) - RMSE(Stefan, recalibrated) (cm) within regional terciles of each covariate, spatial block out-of-fold predictions",
                "input_files": ["outputs/figures/paper/v7/source_data/F4e_gain_conditions.csv (computed by scripts/4_visualization/v9_f4e_gain_conditions.py from data/processed/v5/runs/core_v6/preds)"],
                "row_filter": "covariate != all; cell shown when n >= 30 cell-years and >= 5 blocks; outline when the 95 % block-bootstrap interval (2000 resamples) excludes 0",
                "covariate_definitions": {"TDD": "thaw degree-days of the label year, full season (ERA5-Land)", "FDD": "freezing degree-days of the preceding winter, Oct-Apr (ERA5-Land)",
                                          "Snow (SWE)": "mean snow water equivalent of the hydrological year, Oct-Sep (ERA5-Land)", "SOC 0-5 cm": "SoilGrids", "Clay": "SoilGrids 5-15 cm",
                                          "Slope": "DEM", "Elevation": "DEM", "NDVI": "MODIS summer NDVI, multi-year mean per cell"},
                "terciles": "within-region 33.3 / 66.7 percentiles over the labelled cell-years",
                "label_years": {C.REG_EN[r]: f"{years[r]['block']['first']}-{years[r]['block']['last']}" for r in REG},
                "colour": f"cmcrameri vik, limits -{VMAX} to +{VMAX} cm (symmetric)",
                "region_wide_delta_cm_95ci": {C.REG_EN[r]: [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)] for r in REG
                                              for z in [tg[(tg.region == r) & (tg.covariate == "all")].iloc[0]]},
                "printed_numbers": dinfo["printed"], "outlined_cells": dinfo["outlined"], "hatched_cells": dinfo["hatched"], "cell_mm": dinfo["cell_mm"]},
        },
    }
    C.write_json(C.SRC / f"{STEM}.json", src)
    for p in outs:
        print(p)
    return probs


if __name__ == "__main__":
    main()
