"""Round-4 figure proposals (2026-10-22), part B: groups Fig 4-5 (conditions, products) and Fig 6-7 (maps). Real data, paper_v15 palette, one message each.

Fig 4-5 group
  P45_1  block_gain_vs_labels   per 0.5 degree block: RMSE change of the physics-residual model against recalibrated Stefan (block CV) against the block's
                                number of site-years; grey binned means; shows where and with how many labels the learner helps
  P45_2  site_difference_maps   (Koppa 2022 Fig 3 grammar) observed cells coloured by |our error| - |Wei 2026 error| on the common site-years, three regional
                                maps with grey land, plus the share of cells where our map is closer for Wei 2026 and Liu 2024
  P45_3  binned_comparison      (Araza 2022 Fig 4 grammar) predicted against observed by 10 cm bins of observed ALT: bin mean + interquartile whisker,
                                symbol size = n, for our map, Wei 2026 and Liu 2024; shows the compression of the products
Fig 6-7 group
  P67_1  hovmoller              latitude band x year ALT anomaly of the yearly maps (one matrix per region, same diverging scale), band mean ALT at the right
  P67_2  latitude_profiles      median observed ALT and the medians of our map, Wei 2026 and Liu 2024 by latitude band at the common site-years (n per band)
  P67_3  map_minus_product      our map minus Wei 2026 and minus Liu 2024 for the Fig 6 year of each region (diverging scale, grey masks), mean difference printed

Run   cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper15 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 \
        python3 scripts/4_visualization/paper_v15/proposals_v15_2b.py [P45_1 ...]
"""
from __future__ import annotations
import gc, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import c13_maps as MP  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import proposals_v15_2a as A  # noqa: E402

F6 = MP.F6; ROOT = A.ROOT; MAPS = A.MAPS; OUT = A.OUT; REGIONS = A.REGIONS; KEY = A.KEY; RN = A.RN; RC = A.RC; MC = A.MC; MN = A.MN; YEARS = A.YEARS
PRODUCT = {"wei": ("Wei 2026", C.PRODUCT["wei"][1]), "liu": ("Liu 2024", C.PRODUCT["liu"][1])}
CM_DIV = A.CM_DIV


def common():
    import fig5_products_v15 as G5
    return G5.common_rows()


# ================================================================ Fig 4-5 group
def P45_1():
    stem = "P45_1_block_gain_vs_labels"; t0 = time.time(); W, H = 180.0, 64.0; fig = C.figure(W, H); meta = {}
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 46.0, 38.0, 15.0
    for j, r in enumerate(REGIONS):
        P1 = A.preds(r, "P1").rename(columns={"pred": "p1"}); R1 = A.preds(r, "R1").rename(columns={"pred": "r1"}); d = P1.merge(R1[["cell_year_id", "r1"]], on="cell_year_id")
        g = d.groupby("block").apply(lambda q: pd.Series(dict(n=len(q), delta=A.rmse(q.r1 - q.y) - A.rmse(q.p1 - q.y)))).reset_index()
        ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax)
        ax.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=1)
        ax.scatter(g.n, g.delta, s=2.2 ** 2, facecolor=RC[r], edgecolor="white", linewidths=0.25, alpha=0.9, zorder=3)
        edges = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000]); mids, means, ns = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            q = g[(g.n >= a) & (g.n < b)]
            if len(q) >= 3: mids.append(np.sqrt(a * b)); means.append(float(np.average(q.delta, weights=q.n))); ns.append(int(len(q)))
        ax.plot(mids, means, color=C.TEXT, lw=C.LW_MAIN, marker="_", ms=6, mew=1.0, zorder=4)
        ax.set_xscale("log"); ax.set_xlim(0.8, 12000); ax.set_xlabel("Site-years in the block"); ax.set_ylim(-25, 25)
        if j == 0: ax.set_ylabel("RMSE change per block,\nphysics + residual ML − Stefan (cm)")
        share = float((g.delta < 0).mean()); ax.text(0.03, 0.96, f"{len(g)} blocks; lower RMSE in {share * 100:.0f} %", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
        C.text_mm(fig, xs[j], y0 - 1.2, RN[r], fontsize=C.FS, ha="left", va="bottom")
        meta[r] = dict(blocks=int(len(g)), share_gain=round(share, 3), binned=dict(mid=[round(float(m), 1) for m in mids], weighted_mean_delta=[round(m, 2) for m in means], n_blocks=ns),
                       clipped=int((g.delta.abs() > 25).sum()))
    hs = [Line2D([], [], ls="none", marker="o", ms=3, mfc=C.DARK, mec="white", mew=0.3), Line2D([], [], color=C.TEXT, lw=C.LW_MAIN, marker="_", ms=6, mew=1.0)]
    fig.legend(hs, ["One 0.5° block (spatial-block CV)", "Mean weighted by site-years, per label-count bin"], loc="upper left", bbox_to_anchor=(xs[0] / W, 1 - 6.8 / H), ncol=2, frameon=False,
               fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    C.letter(fig, 2.0, 2.4, "a"); C.text_mm(fig, 7.0, 2.7, "Gain of the residual learner by block against the block's number of labels", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P45_1", message="The gain is not a matter of label count alone: dense Alaska blocks gain little, sparse Lena and Canada blocks gain", canvas_mm=[W, H]))
    A.finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


def P45_2():
    stem = "P45_2_site_difference_maps"; t0 = time.time(); M = MP.Med(); c = common()
    R = {r: A.Region(KEY[r]) for r in REGIONS}; rgbs = {r: R[r].land_rgb() for r in REGIONS}
    W = 180.0; w = 50.0; y0 = 6.5; hmax = max(w * R[r].asp for r in REGIONS); yb = y0 + hmax + 12.0; hb = 24.0; H = yb + hb + 9.0; fig = C.figure(W, H)
    axes, _ = A.three_maps(fig, y0, w, R, M, rgbs, [RN[r] for r in REGIONS], ["a", "b", "c"]); norm = TwoSlopeNorm(vcenter=0, vmin=-20, vmax=20); meta = {}
    for j, r in enumerate(REGIONS):
        q = c[c.region == r].copy(); q["cell"] = A.cell_ids(q.lat.values, q.lon.values)
        q["d_wei"] = (q.R1 - q.alt_cm).abs() - (q.wei - q.alt_cm).abs(); q["d_liu"] = (q.R1 - q.alt_cm).abs() - (q.liu - q.alt_cm).abs()
        g = q.groupby("cell").agg(lat=("lat", "mean"), lon=("lon", "mean"), d_wei=("d_wei", "mean"), d_liu=("d_liu", "mean"), n=("alt_cm", "size"))
        x, y = R[r].tf.transform(g.lon.values, g.lat.values); o = np.argsort(-np.abs(g.d_wei.values))[::-1]
        axes[j].scatter(x[o], y[o], c=g.d_wei.values[o], cmap=CM_DIV, norm=norm, s=2.4 ** 2, edgecolors=C.TEXT, linewidths=0.3, zorder=6)
        meta[r] = dict(cells=int(len(g)), site_years=int(len(q)), share_closer_wei=round(float((g.d_wei < 0).mean()), 3), share_closer_liu=round(float((g.d_liu < 0).mean()), 3),
                       mean_abs_err=dict(ours=round(float((q.R1 - q.alt_cm).abs().mean()), 2), wei=round(float((q.wei - q.alt_cm).abs().mean()), 2), liu=round(float((q.liu - q.alt_cm).abs().mean()), 2)))
    cax = C.ax_mm(fig, 22.0, y0 + hmax + 4.0, 56.0, 1.8); cb = mpl.colorbar.ColorbarBase(cax, cmap=CM_DIV, norm=norm, orientation="horizontal", ticks=[-20, -10, 0, 10, 20], extend="both")
    cb.outline.set_linewidth(C.LW_AX); cax.tick_params(labelsize=C.FS_TICK, length=1.5, width=0.5, pad=1.0)
    cax.set_xlabel("|error of our map| − |error of Wei 2026| per observed cell (cm)\nblue: our map closer to the observation", fontsize=C.FS_SMALL, labelpad=1.5)
    # d: share of cells where our map is closer (Wei, Liu), three regions
    ax = C.ax_mm(fig, 128.5, yb, 46.0, hb); C.clean_axes(ax); yy = np.arange(3)
    for k, (p, dy) in enumerate((("wei", -0.18), ("liu", 0.18))):
        v = [meta[r][f"share_closer_{p}"] * 100 for r in REGIONS]
        ax.barh(yy + dy, v, height=0.34, color=PRODUCT[p][1], edgecolor="none"); [ax.text(val + 1, yv + dy, f"{val:.0f} %", ha="left", va="center", fontsize=C.FS_SMALL, color=C.TEXT) for val, yv in zip(v, yy)]
    ax.axvline(50, color=C.ZERO, lw=C.LW_REF); ax.set_xlim(0, 100); ax.set_yticks(yy); ax.set_yticklabels([RN[r] for r in REGIONS]); ax.set_ylim(2.6, -0.6); ax.set_xlabel("Observed cells where our map is closer (%)")
    fig.legend([Line2D([], [], color=PRODUCT[p][1], lw=4) for p in ("wei", "liu")], [f"Against {PRODUCT[p][0]}" for p in ("wei", "liu")], loc="upper left", bbox_to_anchor=(128.5 / W, 1 - (yb - 4.6) / H), ncol=2, fontsize=C.FS_SMALL, frameon=False, borderaxespad=0, handlelength=1.2, columnspacing=1.4)
    C.letter(fig, 112.0, yb - 4.1, "d")
    meta.update(dict(proposal="P45_2", message="Cell by cell, our map is closer to the observation than the products in most of the Lena Delta and Canada and in about half of Alaska", canvas_mm=[W, round(H, 1)],
                     definition="common site-years (Wei, Liu, CCI v4 and our block-CV prediction present), averaged per 1 km cell"))
    A.finish(fig, stem, meta, ignore=["km"]); print(f"  {stem} {time.time() - t0:.0f} s")


def P45_3():
    stem = "P45_3_binned_comparison"; t0 = time.time(); c = common(); W, H = 180.0, 68.0; fig = C.figure(W, H); meta = {}
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 44.0, 44.0, 15.0
    series = [("R1", MN["R1"], MC["R1"], 0.0), ("wei", "Wei 2026", PRODUCT["wei"][1], -2.5), ("liu", "Liu 2024", PRODUCT["liu"][1], 2.5)]
    for j, r in enumerate(REGIONS):
        q = c[c.region == r]; top = {"Alaska": 120, "Lena": 120, "Canada": 160}[r]; edges = np.arange(0, top + 1, 10); ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax)
        ax.plot([0, top], [0, top], color=C.ZERO, lw=C.LW_REF, zorder=1)
        k = np.clip(np.digitize(q.alt_cm.values, edges) - 1, 0, len(edges) - 2)
        for key, name, col, dx in series:
            mids, means, q1s, q3s, ns = [], [], [], [], []
            for i in range(len(edges) - 1):
                v = q[key].values[k == i]
                if len(v) < 5: continue
                mids.append(edges[i] + 5 + dx); means.append(float(np.mean(v))); q1s.append(float(np.percentile(v, 25))); q3s.append(float(np.percentile(v, 75))); ns.append(int(len(v)))
            mids, means, q1s, q3s, ns = map(np.array, (mids, means, q1s, q3s, ns))
            ax.vlines(mids, q1s, q3s, color=col, lw=0.6, zorder=3); ax.scatter(mids, means, s=np.clip(6 + 10 * np.log10(ns), 6, 40), facecolor=col, edgecolor="white", linewidths=0.4, zorder=4)
            meta[f"{r}_{key}"] = dict(bin_mid=[round(float(m - dx), 0) for m in mids], mean=[round(float(m), 1) for m in means], q1=[round(float(v), 1) for v in q1s], q3=[round(float(v), 1) for v in q3s], n=[int(v) for v in ns])
        ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal"); ax.set_xlabel("Observed ALT (cm), 10 cm bins")
        if j == 0: ax.set_ylabel("Predicted (cm): bin mean, interquartile whisker")
        C.text_mm(fig, xs[j], y0 - 1.2, f"{RN[r]}, n = {len(q):,} common site-years", fontsize=C.FS, ha="left", va="bottom")
    hs = [Line2D([], [], ls="none", marker="o", ms=4, mfc=col, mec="white", mew=0.4) for _, _, col, _ in series] + [Line2D([], [], ls="none", marker="o", ms=s, mfc=C.DARK, mec="white", mew=0.3) for s in (2.5, 5.5)]
    fig.legend(hs, [n for _, n, _, _ in series] + ["n = 10", "n = 1,000"], loc="upper left", bbox_to_anchor=(xs[0] / W, 1 - 6.8 / H), ncol=5, frameon=False, fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    C.letter(fig, 2.0, 2.4, "a"); C.text_mm(fig, 7.0, 2.7, "Predicted against observed by bins of observed ALT on the common site-years", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P45_3", message="The products predict a narrow range regardless of the observation; our map follows the observations in the Lena Delta and Canada", canvas_mm=[W, H]))
    A.finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


# ================================================================ Fig 6-7 group
def P67_1():
    stem = "P67_1_hovmoller"; t0 = time.time(); W, H = 180.0, 82.0; fig = C.figure(W, H); meta = {}
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 40.0, 48.0, 11.0; norm = Normalize(-10, 10); band = {"Alaska": 1.0, "Lena": 0.25, "Canada": 1.0}
    for j, r in enumerate(REGIONS):
        mean = pd.read_csv(MAPS / r / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1", "lat"]).set_index("cell_id")
        b = np.floor(mean.lat.values / band[r]) * band[r]; bands = np.unique(b); mat = np.full((len(bands), len(YEARS)), np.nan); ncell = np.array([(b == bb).sum() for bb in bands])
        keep = ncell >= 200
        for jj, y in enumerate(YEARS):
            p = pd.read_csv(MAPS / r / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1.reindex(mean.index).values - mean.r1.values
            mat[:, jj] = [np.nanmean(p[b == bb]) if keep[i] else np.nan for i, bb in enumerate(bands)]
        bands, mat, ncell = bands[keep], mat[keep], ncell[keep]; bm = np.array([float(np.nanmean(mean.r1.values[b == bb])) for bb in bands])
        ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax)
        ax.pcolormesh(np.array(YEARS + [YEARS[-1] + 1]) - 0.5, np.r_[bands, bands[-1] + band[r]], mat, cmap=MP.CM_ANOM, norm=norm, rasterized=True, edgecolors="none")
        ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 0.5); ax.set_xticks([2012, 2016, 2020, 2024]); ax.set_xlabel("Year")
        ax.set_ylabel("Latitude (°N)" if j == 0 else ""); ax.yaxis.set_major_locator(mticker.MaxNLocator(5)); ax.set_ylim(bands[0], bands[-1] + band[r])
        axr = C.ax_mm(fig, xs[j] + w + 1.5, y0, 7.0, h); C.clean_axes(axr, left=False); axr.barh(bands + band[r] / 2, bm, height=band[r] * 0.9, color="#BDBDBD", edgecolor="none")
        axr.set_ylim(ax.get_ylim()); axr.set_yticks([]); axr.spines["left"].set_visible(False); axr.set_xlabel("Mean\nALT (cm)", fontsize=C.FS_SMALL); axr.set_xlim(0, 1.15 * bm.max()); axr.set_xticks([int(round(bm.max() / 10.0) * 10)]); axr.tick_params(labelsize=C.FS_SMALL)
        C.text_mm(fig, xs[j], y0 - 1.2, f"{RN[r]}, {band[r]:g}° bands", fontsize=C.FS, ha="left", va="bottom")
        meta[r] = dict(bands=[float(v) for v in bands], band_deg=band[r], mean_alt=[round(float(v), 1) for v in bm], matrix=[[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in mat], cells_per_band=[int(v) for v in ncell])
    cax = C.ax_mm(fig, 70.0, y0 + h + 10.5, 40.0, 1.8); MP.colorbar(fig, cax, MP.CM_ANOM, norm, "ALT anomaly of the year from the 2011–2024 mean (cm)", [-10, -5, 0, 5, 10], extend="both")
    C.letter(fig, 2.0, 2.4, "a"); C.text_mm(fig, 7.0, 2.7, "Yearly ALT anomaly of the maps by latitude band (mean over the mapped cells of the band)", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P67_1", message="Fourteen yearly maps in one view: warm and cold years are regional, the amplitude differs by latitude", canvas_mm=[W, H]))
    A.finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


def P67_2():
    stem = "P67_2_latitude_profiles"; t0 = time.time(); c = common(); W, H = 180.0, 66.0; fig = C.figure(W, H); meta = {}
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 40.0, 38.0, 15.0; band = {"Alaska": 1.0, "Lena": 0.25, "Canada": 1.0}
    series = [("alt_cm", "Observed", C.TEXT), ("R1", MN["R1"], MC["R1"]), ("wei", "Wei 2026", PRODUCT["wei"][1]), ("liu", "Liu 2024", PRODUCT["liu"][1])]
    for j, r in enumerate(REGIONS):
        q = c[c.region == r].copy(); q["band"] = np.floor(q.lat / band[r]) * band[r]; g = q.groupby("band"); ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax)
        bands = [b for b, d in g if len(d) >= 10]; meta[r] = {"bands": [float(b) for b in bands], "n": [int(len(g.get_group(b))) for b in bands]}
        for key, name, col in series:
            med = [float(g.get_group(b)[key].median()) for b in bands]
            if key == "alt_cm":
                q1 = [float(g.get_group(b)[key].quantile(0.25)) for b in bands]; q3 = [float(g.get_group(b)[key].quantile(0.75)) for b in bands]
                ax.fill_betweenx(np.array(bands) + band[r] / 2, q1, q3, color="#E6E6E6", zorder=1); ax.plot(med, np.array(bands) + band[r] / 2, color=col, lw=C.LW_MAIN, marker="o", ms=C.MS, mfc="white", mec=col, mew=0.7, zorder=5)
            else:
                ax.plot(med, np.array(bands) + band[r] / 2, color=col, lw=C.LW_MAIN, marker="o", ms=C.MS * 0.8, zorder=4 if key == "R1" else 3)
            meta[r][key] = [round(v, 1) for v in med]
        for b, n in zip(bands, meta[r]["n"]): ax.text(1.03, b + band[r] / 2, f"{n:,}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=C.FS_SMALL, color=C.AUX, clip_on=False)
        ax.set_xlabel("ALT (cm), median per band"); ax.set_ylabel("Latitude (°N)" if j == 0 else ""); ax.set_ylim(bands[0], bands[-1] + band[r])
        C.text_mm(fig, xs[j], y0 - 1.2, f"{RN[r]}, {band[r]:g}° bands, n per band at the right", fontsize=C.FS, ha="left", va="bottom")
    hs = [Line2D([], [], color=C.TEXT, lw=C.LW_MAIN, marker="o", ms=C.MS, mfc="white", mew=0.7)] + [Line2D([], [], color=col, lw=C.LW_MAIN, marker="o", ms=C.MS * 0.8) for _, _, col in series[1:]] + [mpl.patches.Patch(facecolor="#E6E6E6")]
    fig.legend(hs, [n for _, n, _ in series] + ["Observed interquartile range"], loc="upper left", bbox_to_anchor=(xs[0] / W, 1 - 6.8 / H), ncol=5, frameon=False, fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    C.letter(fig, 2.0, 2.4, "a"); C.text_mm(fig, 7.0, 2.7, "Observed and predicted ALT by latitude band on the common site-years", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P67_2", message="Along latitude the observations and our map move together; the products stay flat or offset", canvas_mm=[W, H]))
    A.finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


def P67_3():
    stem = "P67_3_map_minus_product"; t0 = time.time(); M = MP.Med()
    import fig6_product_maps as G6
    W = 180.0; w = 50.0; cols = [9.5, 69.0, 128.5]; norm = Normalize(-30, 30); meta = {}; y = 6.5; fig = None; rows = []
    Ds = [G6.load_row(dict(r, step=4)) for r in G6.ROWS]; hs_ = [w * D["asp"] for D in Ds]; hmax = max(hs_)
    H = 6.5 + 2 * (hmax + 7.0) + 14.0; fig = C.figure(W, H); letters = "abcdef"
    for i, p in enumerate(("wei", "liu")):
        yrow = 6.5 + i * (hmax + 7.0)
        for j, D in enumerate(Ds):
            diff = np.where(D["drawn"] & D["pm"][p], D["px_r1"] - D["prod"][p], np.nan); drawn = np.isfinite(diff)
            rgb = G6.render_px(D["base"], diff, MP.CM_ANOM, norm, drawn, w_shade=0.0); ax = C.ax_mm(fig, cols[j], yrow, w, hs_[j]); MP.frame(ax, rgb, D["base"].extent, D["lim"]); del rgb
            MP.coast(ax, D["key"], D["tf"], M, base=D["base"], cells=D["cells"], tol_m=2000.0); MP.graticule(ax, D["key"], D["tf"], D["lim"], M, lon_lab=D["lon_lab"] if i == 1 else (), lat_lab=D["lat_lab"] if j == 0 else ())
            ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
            if i == 0 and j == 0: MP.scalebar(ax, D["lim"], D["scale_km"], M, fx=0.66, fy=0.06)
            v = diff[drawn]; txt = f"mean {np.mean(v):+.1f} cm, {np.mean(np.abs(v) <= 10) * 100:.0f} % within 10 cm".replace("-", "−")
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT, zorder=9, bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
            C.letter(fig, cols[j] - 7.5, yrow - 4.1, letters[i * 3 + j]); C.text_mm(fig, cols[j] - 4.1, yrow - 3.95, f"{RN[D['region']]} {D['year']}: our map − {PRODUCT[p][0]}", fontsize=C.FS, ha="left", va="top")
            meta[f"{D['region']}_{p}"] = dict(year=int(D["year"]), mean=round(float(np.mean(v)), 2), p10=round(float(np.percentile(v, 10)), 1), p90=round(float(np.percentile(v, 90)), 1), within10=round(float(np.mean(np.abs(v) <= 10)), 3), pixels=int(drawn.sum()))
            gc.collect()
    cax = C.ax_mm(fig, 70.0, H - 8.5, 40.0, 1.8); MP.colorbar(fig, cax, MP.CM_ANOM, norm, "Our map − product (cm); grey: product or our map without a value", [-30, -15, 0, 15, 30], extend="both")
    meta.update(dict(proposal="P67_3", message="Where the products are deeper than our map and by how much: the Lena Delta products are 15-30 cm deeper almost everywhere", canvas_mm=[W, round(H, 1)],
                     definition="pixel-level difference at the Fig 6 year of each region; products masked to the drawn cells of our map"))
    A.finish(fig, stem, meta, ignore=["km", "mean"]); print(f"  {stem} {time.time() - t0:.0f} s")


REG = {"P45_1": P45_1, "P45_2": P45_2, "P45_3": P45_3, "P67_1": P67_1, "P67_2": P67_2, "P67_3": P67_3}

if __name__ == "__main__":
    C.setup(); sel = sys.argv[1:] or list(REG)
    for k in sel:
        try:
            REG[k]()
        except Exception:
            print(f"[FAIL] {k}"); traceback.print_exc()
        gc.collect()
