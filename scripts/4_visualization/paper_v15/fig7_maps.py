"""Fig. 7 (v14 manuscript, 2026-10-20): year-labelled 1 km ALT maps 2011-2024 of the physics-residual model in the three study regions.
a  Alaska mean ALT 2011-2024 (mean of the yearly maps), 250 m display interpolation and subtle hillshade, observed cells filled with the mean
   observed ALT of their 2011-2024 labels (v14), outside the area of
   applicability lighter; two 50 km windows (North Slope near Toolik; Interior near Fairbanks) drawn as boxes and enlarged at the right (1 km cells).
b  Lena Delta mean ALT 2011-2024, same drawing; one 24 km window (Samoylov-Kurungnakh) enlarged at the right.
c  Canada (NW) mean ALT 2011-2024, same drawing; two 50 km windows (Old Crow Flats; Mackenzie Delta near Inuvik) enlarged at the right.
   a-c share one colour range (1st-99th percentile of the pooled mean maps, 5 cm steps) and one colour bar.
d  Regional mean anomaly of the predicted maps and mean anomaly of the observed cells (years with >= 50 site-years; labels matched to the nearest
   map cell <= 1.5 km, observed ALT minus the cell's 2011-2024 map mean, centred on the predicted anomaly mean of the same years), three regions.
e  Linear trend of the yearly maps 2011-2024 (ordinary least squares per 1 km cell, cm per decade), three regions, one symmetric colour range;
   outside the area of applicability lighter. The trend describes the modelled maps driven by the ERA5-Land climate of each year.
f  ALT anomaly of each year 2011-2024 relative to the 2011-2024 mean (Alaska), 14 small maps, cmcrameri vik, symmetric +-15 cm.
Inputs: data/processed/v5/maps/{Alaska,Lena,Canada}/{pred_<year>,mean_2011_2024}.csv.gz (TDD accumulated to 1 September), data/processed/map_<region>/
        (grid, hillshade), data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1), caches outputs/figures/paper/v15/_cache.
Outputs: outputs/figures/paper/v15/Fig7_maps.{pdf,svg,png}, outputs/figures/paper/v15/source_data/Fig7_maps.json
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 \\
       python3 scripts/4_visualization/paper_v15/fig7_maps.py
     FIG7_REGIONS=alaska,lena (environment) restricts the regions for a test render (the delivered figure has all three).
"""
from __future__ import annotations
import gc
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import c13_maps as MP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import Normalize, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import cmcrameri.cm as cmc  # noqa: E402

F6 = MP.F6
import v7_f7_series_significance as F7  # noqa: E402  (observed_anomaly: same definition as the deck R10)

ROOT = C.ROOT
PROC = ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"
STEM = "Fig7_maps"
YEARS = list(range(2011, 2025))
ANOM = 15.0                                   # symmetric anomaly range (cm)
W_SHADE = 0.30
CM_TREND = ListedColormap(cmc.broc(np.linspace(0.0, 1.0, 256)))     # trend: diverging (olive-brown = thickening, blue = thinning), symmetric
REGS = [dict(key="alaska", reg="Alaska", scale_km=200, lon_lab=(-165, -155, -145), lat_lab=(60, 65, 70),
             zooms=[("North Slope, Toolik", (-149.55, 68.75), 25e3), ("Interior, Fairbanks", (-147.85, 64.85), 25e3)], zoom_scale_km=10),
        dict(key="lena", reg="Lena", scale_km=50, lon_lab=(124, 128), lat_lab=(72, 73),
             zooms=[("Samoylov–Kurungnakh", (126.4, 72.36), 12e3)], zoom_scale_km=5),
        dict(key="canada", reg="Canada", scale_km=200, lon_lab=(-135, -125, -115), lat_lab=(60, 65),
             zooms=[("Old Crow Flats", (-139.7, 67.9), 25e3), ("Mackenzie Delta, Inuvik", (-133.5, 68.3), 25e3)], zoom_scale_km=10)]
_sel = os.environ.get("FIG7_REGIONS")
if _sel:
    REGS = [r for r in REGS if r["key"] in _sel.split(",")]


def load_region(key, reg):
    cells, lut, ky0, kx0 = F6.cell_table(key)
    mean = pd.read_csv(MAPS / reg / "mean_2011_2024.csv.gz", dtype={"cell_id": str})
    m = mean.set_index("cell_id")
    vmean = F6.cell_values(key, cells, m.r1); aoa = np.nan_to_num(F6.cell_values(key, cells, m.aoa_in.astype(float)), nan=0.0)
    anom, reg_mean = {}, {}
    for y in YEARS:
        p = pd.read_csv(MAPS / reg / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
        reg_mean[y] = float(p.reindex(mean.cell_id).mean())
        anom[y] = F6.cell_values(key, cells, p) - vmean
    t = np.array(YEARS, float); w = (t - t.mean()) / np.sum((t - t.mean()) ** 2)            # OLS slope weights (cm per year)
    trend = np.zeros_like(vmean)
    for y, wy in zip(YEARS, w):
        trend = trend + wy * np.nan_to_num(anom[y], nan=0.0)
    trend = np.where(np.isfinite(vmean), trend * 10.0, np.nan)                               # cm per decade
    if key != "alaska":                                                                      # the yearly anomaly maps (f) are drawn for Alaska only
        anom = {}
    return dict(cells=cells, lut=lut, ky0=ky0, kx0=kx0, mean=mean, vmean=vmean, aoa=aoa, anom=anom, reg_mean=reg_mean, trend=trend)


def load():
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    return dict(lab=lab, R={r["key"]: load_region(r["key"], r["reg"]) for r in REGS})


def obs_cells_mean(lab, region, years=(2011, 2024)):
    """Observed 1 km cells of the region with the mean observed ALT of the labels of `years` (v14: the cells are drawn filled with this value so that the
    map can be compared with the observations on the same colour scale). Cell arithmetic as F6.obs_cells."""
    d = lab[(lab.region == region) & (lab.year >= years[0]) & (lab.year <= years[1])]
    ky = np.floor(d.lat.values / F6.CELL).astype(np.int64); dx = F6.CELL / np.cos(np.radians((ky + 0.5) * F6.CELL)); kx = np.floor(d.lon.values / dx).astype(np.int64)
    g = pd.DataFrame({"ky": ky, "kx": kx, "alt": d.alt_cm.values}).groupby(["ky", "kx"]).alt.agg(["mean", "size"]).reset_index()
    lat = (g.ky.values + 0.5) * F6.CELL; lon = (g.kx.values + 0.5) * F6.CELL / np.cos(np.radians(lat))
    return lon, lat, g["mean"].values.astype(float), g["size"].values.astype(int), int(d.loc_id.nunique())


def obs_marks_filled(ax, ox, oy, vals, lim, norm, s, lw, pad_frac=0.0):
    """Observed cells as circles filled with the observed mean ALT (map colour scale), black edge; cells pad_frac inside the panel only."""
    px = pad_frac * (lim[1] - lim[0]); py = pad_frac * (lim[3] - lim[2])
    m = (ox >= lim[0] + px) & (ox <= lim[1] - px) & (oy >= lim[2] + py) & (oy <= lim[3] - py)
    o = np.argsort(-oy[m], kind="stable")
    ax.scatter(ox[m][o], oy[m][o], c=vals[m][o], cmap=MP.CM_ALT, norm=norm, s=s, edgecolors=C.TEXT, linewidths=lw, zorder=6)
    return int(m.sum())


def series(D):
    out = {}
    for r in REGS:
        R = D["R"][r["key"]]; rm = R["reg_mean"]; mu = float(np.mean(list(rm.values()))); pred = {y: rm[y] - mu for y in YEARS}
        obs = F7.observed_anomaly(D["lab"], r["reg"], R["mean"], pred, min_n=50, max_km=1.5)
        out[r["reg"]] = dict(mean_cm=mu, pred=pred, obs=obs)
    return out


def draw_series(ax, reg, S_, show_xlabel):
    C.clean_axes(ax)
    s = S_[reg]; pred = np.array([s["pred"][y] for y in YEARS]); o = s["obs"]
    name, col, mk = C.REGION[reg]
    ax.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=1)
    ax.plot(YEARS, pred, color=col, lw=C.LW_MAIN, zorder=3, solid_capstyle="round", solid_joinstyle="round")
    if len(o):
        ax.plot(o.index.values, o["mean"].values, ls="none", marker="o", ms=3.1, mfc="white", mec=col, mew=0.7, zorder=4)
    omin = float(o["mean"].min()) if len(o) else 0.0; omax = float(o["mean"].max()) if len(o) else 0.0
    lo = -5.0 * np.ceil((max(float(np.abs(pred).max()), -omin) + 1.0) / 5.0)
    hi = 5.0 * np.ceil((max(float(pred.max()), omax) + 1.5) / 5.0)
    ax.set_ylim(min(lo, -10), max(hi, 10)); ax.set_xlim(2010.4, 2024.6)
    ax.set_xticks([2012, 2016, 2020, 2024]); ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.tick_params(which="minor", length=1.0, width=0.4)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20 if (hi - lo) > 45 else (10 if (hi - lo) > 18 else 5)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−")))
    ax.text(0.02, 0.95, name, transform=ax.transAxes, ha="left", va="top", fontsize=C.FS, color=C.TEXT)
    labelled = []
    for yr, row in o.iterrows():                                     # observed points far outside the predicted range (Lena Delta 2020; Canada 2023-2024)
        if abs(row["mean"]) > 2.0 * float(np.abs(pred).max()) and abs(row["mean"]) > 15:
            labelled.append(dict(year=int(yr), anomaly_cm=round(float(row["mean"]), 2), n=int(row["size"])))
    groups, cur = [], []                                             # consecutive years share one label at the first point ("n = 80, 80")
    for L in labelled:
        if cur and L["year"] == cur[-1]["year"] + 1:
            cur.append(L)
        else:
            cur = [L]; groups.append(cur)
    for grp in groups:
        ax.annotate("n = " + ", ".join(str(L["n"]) for L in grp), (grp[0]["year"], grp[0]["anomaly_cm"]), xytext=(-4.5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=C.FS_SMALL, color=C.AUX)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("Year")
    return labelled, [float(v) for v in ax.get_ylim()]


def mean_map(fig, M, D, r, R, x0, y0, w, h, norm, xz, wz, lon_lab, lat_lab, letter, title, key_text=None):
    """One mean-ALT map with its windows (stacked at xz). Returns (observed cells drawn, observed sites, window info)."""
    t1 = time.time(); key = r["key"]
    b1 = F6.Base(key, R["cells"], R["lut"], R["ky0"], R["kx0"], step=1, tag="full"); tf = b1.tf; lim = F6.region_limits(key, tf)
    big = key != "lena"
    rgb, _ = MP.render(b1, R["vmean"], MP.CM_ALT, norm, aoa=R["aoa"], smooth=True, fill_max=F6.FILL_MAX, w_shade=W_SHADE,
                       feather=4.0 if big else 1.5, aoa_sigma=6.0 if big else 4.0)
    ax = C.ax_mm(fig, x0, y0, w, h); MP.frame(ax, rgb, b1.extent, lim); del rgb; gc.collect()
    MP.coast(ax, key, tf, M, base=b1, cells=R["cells"], tol_m=500.0); MP.graticule(ax, key, tf, lim, M, lon_lab=lon_lab, lat_lab=lat_lab)
    ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
    olon, olat, oval, ocnt, nloc = obs_cells_mean(D["lab"], r["reg"]); ox, oy = tf.transform(olon, olat)
    n_obs = obs_marks_filled(ax, ox, oy, oval, lim, norm, s=2.3 ** 2, lw=0.3)
    MP.scalebar(ax, lim, r["scale_km"], M, fx=0.05, fy=0.05)
    zinfo = {}
    for j, (name, cen, hx) in enumerate(r["zooms"]):
        zl = F6.zoom_lim(tf, cen, hx, hx)
        MP.box(ax, zl, lw=0.6)
        pad = 2e3 if big else 1e3
        bz = F6.Base(key, R["cells"], R["lut"], R["ky0"], R["kx0"], step=1, window=[zl[0] - pad, zl[1] + pad, zl[2] - pad, zl[3] + pad], tag=f"zoom{j}")
        rgbz, _ = MP.render(bz, R["vmean"], MP.CM_ALT, norm, aoa=R["aoa"], smooth=False, w_shade=W_SHADE)
        axz = C.ax_mm(fig, xz, y0 + j * (wz + 1.5), wz, wz); MP.frame(axz, rgbz, bz.extent, zl); del rgbz
        obs_marks_filled(axz, ox, oy, oval, zl, norm, s=M.obs_zoom_s(wz / ((zl[1] - zl[0]) / 1e3)) * 1.3, lw=0.4, pad_frac=0.012)
        MP.scalebar(axz, zl, r["zoom_scale_km"], M, fx=0.06 if j == 0 else 0.56, fy=0.07); MP.corner_label(axz, name, M, fs=C.FS_SMALL)
        zinfo[name] = dict(centre_lonlat=list(cen), width_km=2 * hx / 1e3, cells_outside_aoa_in_window=F6.zoom_outside(bz, R["vmean"], R["aoa"], zl))
    del b1; gc.collect()
    lx = 0.4 if x0 <= 8.0 else x0 - 4.2                          # letter: far left of the canvas, or just left of the map (above its latitude labels)
    C.letter(fig, lx, y0 - 4.1, letter); C.text_mm(fig, lx + 3.4, y0 - 3.95, title, fontsize=C.FS, ha="left", va="top")
    print(f"  {letter} rendered {time.time() - t1:.0f} s")
    return n_obs, int(nloc), zinfo, tf, lim


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    D = load(); R = D["R"]; print(f"  data {time.time() - t0:.0f} s")
    keys = [r["key"] for r in REGS]; byreg = {r["key"]: r for r in REGS}
    src = dict(figure="Fig. 7", script="scripts/4_visualization/paper_v15/fig7_maps.py", years=[YEARS[0], YEARS[-1]],
               inputs=dict(maps="data/processed/v5/maps/{Alaska,Lena,Canada}/pred_<year>.csv.gz and mean_2011_2024.csv.gz (column r1 = Physics + residual ML; "
                                "TDD accumulated to 1 September)",
                           grid="data/processed/map_{alaska,lena,canada}/*_grid_x25_v1.csv.gz; hillshade data/processed/map_<region>/hillshade_v1/shade_250m.tif",
                           labels="data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)",
                           projection="EPSG:3338 (Alaska Albers); Lena Delta and Canada (NW) as v7_f6_maps / maps_alt_v3"))
    # ---------------- geometry: aspect of each region box at step 4
    B4, TF, LIM, ASP = {}, {}, {}, {}
    for k in keys:
        B4[k] = F6.Base(k, R[k]["cells"], R[k]["lut"], R[k]["ky0"], R[k]["kx0"], step=4, tag="full"); TF[k] = B4[k].tf
        LIM[k] = F6.region_limits(k, TF[k]); ASP[k] = (LIM[k][3] - LIM[k][2]) / (LIM[k][1] - LIM[k][0])
    aspA, aspL, aspC = ASP.get("alaska", 0.78), ASP.get("lena", 0.95), ASP.get("canada", 0.83)
    W = 180.0
    # row 1: a (Alaska mean) + two stacked windows | b (Lena mean) + one window; widths follow the two map aspects so that the row spans 178 mm
    y1 = 4.6; xa = 8.0; ha = 43.0                            # row 1 map height (mm); the right margin holds the vertical colour bar and the key
    wa = ha / aspA; wz = (ha - 1.5) / 2.0; xz = xa + wa + 1.5
    xb = xz + wz + 7.0; hb = ha; wb = hb / aspL; xzl = xb + wb + 1.5; wzl = wz
    xk = xzl + wzl + 4.5                                     # colour bar of a-c (vertical) and key
    print(f"  row 1: aspA {aspA:.3f} aspL {aspL:.3f} aspC {aspC:.3f} wa {wa:.1f} ha {ha:.1f} wz {wz:.1f} wb {wb:.1f} key x {xk:.1f}")
    assert xk + 25.0 <= W, ("row 1 too wide", xk)
    # row 2: c (Canada mean) + two stacked windows | d (series, three stacked)
    y2 = y1 + ha + 6.0 + 4.6; hc = 40.0; wc = hc / aspC; wz2 = (hc - 1.5) / 2.0; xz2 = xa + wc + 1.5
    xe = xz2 + wz2 + 15.0; we = W - xe - 1.5; he = (hc - 2 * 2.4) / 3.0
    y_key2 = y2 + hc + 7.0                                   # key of d (one line) below the x labels of the series
    # row 3: e trend maps (three, one height, no longitude labels) + vertical colour bar
    y3 = y_key2 + 3.0 + 4.6; ht = 30.0; xt = {}; x = 8.0
    for k in keys:
        xt[k] = x; x += ht / ASP[k] + 3.0
    xcb = x - 1.0
    assert xcb + 23.0 <= W, ("row 3 too wide", xcb)
    # row 4: f anomaly small multiples (Alaska), 7 x 2
    y4 = y3 + ht + 2.5 + 4.6; gd = 1.0; xd = 8.0; wd = (W - xd - 3.0 - 6 * gd) / 7.0; hd = wd * aspA
    y_cb4 = y4 + 2 * hd + gd + 2.4
    H = y_cb4 + 8.5
    print(f"  canvas {W} x {H:.1f} mm (row tops {y1}, {y2:.1f}, {y3:.1f}, {y4:.1f})")
    fig = C.figure(W, H)
    # ---------------- a-c: mean maps, one colour range
    allm = np.concatenate([R[k]["mean"].r1.values for k in keys]); loM, hiM = F6.MV.pct_range(allm, step=5.0); nM = Normalize(loM, hiM)
    eM = F6.MV.extend_of(allm, loM, hiM); tick = 20.0 if hiM - loM >= 80 else 10.0
    ticksM = list(np.arange(tick * np.ceil(loM / tick), hiM + 1e-6, tick))
    info = {}
    if "alaska" in keys:
        info["alaska"] = mean_map(fig, M, D, byreg["alaska"], R["alaska"], xa, y1, wa, ha, nM, xz, wz, byreg["alaska"]["lon_lab"], byreg["alaska"]["lat_lab"],
                                  "a", "Alaska, mean ALT 2011–2024")
    if "lena" in keys:
        info["lena"] = mean_map(fig, M, D, byreg["lena"], R["lena"], xb, y1, wb, hb, nM, xzl, wzl, byreg["lena"]["lon_lab"], byreg["lena"]["lat_lab"],
                                "b", "Lena Delta, mean ALT 2011–2024")
    if "canada" in keys:
        info["canada"] = mean_map(fig, M, D, byreg["canada"], R["canada"], xa, y2, wc, hc, nM, xz2, wz2, byreg["canada"]["lon_lab"], byreg["canada"]["lat_lab"],
                                  "c", "Canada (NW), mean ALT 2011–2024")
    cax = C.ax_mm(fig, xk, y1 + 1.0, 1.9, 24.0)
    MP.colorbar(fig, cax, MP.CM_ALT, nM, "ALT, mean 2011–2024 (cm)", ticksM, extend=eM, orientation="vertical")
    hs = [Line2D([], [], ls="none", marker="o", ms=2.4, mfc=MP.CM_ALT(0.55), mec=C.TEXT, mew=0.35), MP.FadedRamp(MP.CM_ALT),
          Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3), Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3)]
    fig.legend(hs, ["Observed cell\n(observed mean ALT)", "Outside area of\napplicability", "Not mapped", "Water"], loc="upper left",
               bbox_to_anchor=((xk - 1.5) / W, 1 - (y1 + 30.5) / H), ncol=1, handlelength=1.6, handleheight=0.9, handletextpad=0.45,
               labelspacing=0.35, fontsize=C.FS_SMALL, handler_map={MP.FadedRamp: MP.RampHandler()}, borderaxespad=0, borderpad=0)
    # ---------------- d: series (stacked, three regions)
    S_ = series(D)
    axe = [C.ax_mm(fig, xe, y2 + j * (he + 2.4), we, he) for j in range(len(keys))]
    lab_e = {}; ylims = {}
    for j, k in enumerate(keys):
        lab_e[byreg[k]["reg"]], ylims[byreg[k]["reg"]] = draw_series(axe[j], byreg[k]["reg"], S_, j == len(keys) - 1)
    fig.text((xe - 8.6) / W, 1 - (y2 + hc / 2) / H, "ALT anomaly (cm)", rotation=90, ha="center", va="center", fontsize=C.FS, color=C.TEXT)
    C.letter(fig, xe - 8.6, y2 - 4.1, "d"); C.text_mm(fig, xe - 5.2, y2 - 3.95, "Regional mean ALT anomaly", fontsize=C.FS, ha="left", va="top")
    he_ = [Line2D([], [], color=C.DARK, lw=C.LW_MAIN), Line2D([], [], ls="none", marker="o", ms=3.1, mfc="white", mec=C.DARK, mew=0.7)]
    fig.legend(he_, ["Mapped cells, regional mean", "Observed cells (≥ 50 site-years)"], loc="upper left", bbox_to_anchor=(xe / W, 1 - y_key2 / H),
               ncol=2, handlelength=1.6, handletextpad=0.5, columnspacing=1.4, fontsize=C.FS_SMALL, borderaxespad=0, borderpad=0)
    # ---------------- e: trend maps (three regions, step 4 display)
    t1 = time.time()
    allt = np.concatenate([R[k]["trend"][np.isfinite(R[k]["trend"])] for k in keys])
    vt = float(min(15.0, 2.5 * np.ceil(np.nanpercentile(np.abs(allt), 98) / 2.5))); nT = Normalize(-vt, vt)
    eT = F6.MV.extend_of(allt, -vt, vt)
    Mt = MP.Med(); Mt.lw_coast = 0.35
    for k in keys:
        rgb, _ = MP.render(B4[k], R[k]["trend"], CM_TREND, nT, aoa=R[k]["aoa"], smooth=False, w_shade=0.0)
        ax = C.ax_mm(fig, xt[k], y3, ht / ASP[k], ht); MP.frame(ax, rgb, B4[k].extent, LIM[k]); del rgb
        MP.coast(ax, k, TF[k], Mt, base=B4[k], cells=R[k]["cells"], tol_m=800.0); MP.graticule(ax, k, TF[k], LIM[k], Mt)
        ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT); MP.scalebar(ax, LIM[k], byreg[k]["scale_km"], M, fx=0.05, fy=0.05)
        MP.corner_label(ax, C.REGION[byreg[k]["reg"]][0], M)
    cax = C.ax_mm(fig, xcb, y3 + 3.0, 1.9, ht - 6.0)
    MP.colorbar(fig, cax, CM_TREND, nT, "Trend (cm per decade)", [-vt, -vt / 2, 0, vt / 2, vt], extend=eT, orientation="vertical")
    print(f"  e rendered {time.time() - t1:.0f} s")
    C.letter(fig, 0.4, y3 - 4.1, "e"); C.text_mm(fig, 3.8, y3 - 3.95, "Linear trend of the yearly maps 2011–2024", fontsize=C.FS, ha="left", va="top")
    trend_stats = {}
    for k in keys:
        tv = R[k]["trend"]; ok = np.isfinite(tv); ina = ok & (R[k]["aoa"] > 0.5)
        trend_stats[C.REGION[byreg[k]["reg"]][0]] = dict(mean_cm_per_decade=round(float(np.nanmean(tv[ok])), 3), mean_inside_aoa=round(float(np.nanmean(tv[ina])), 3),
                                                       p05=round(float(np.nanpercentile(tv[ok], 5)), 2), p95=round(float(np.nanpercentile(tv[ok], 95)), 2),
                                                       frac_positive=round(float((tv[ok] > 0).mean()), 4), n_cells=int(ok.sum()))
    # ---------------- f: anomaly small multiples (Alaska)
    t1 = time.time(); stats_f = {}; eF = "neither"
    if "alaska" in keys:
        A = R["alaska"]; bA4 = B4["alaska"]; limA = LIM["alaska"]; tfA = TF["alaska"]
        normF = Normalize(-ANOM, ANOM); allv = np.concatenate([v[np.isfinite(v)] for v in A["anom"].values()]); eF = F6.MV.extend_of(allv, -ANOM, ANOM)
        Md = MP.Med(); Md.lw_coast = 0.25
        for n_, y in enumerate(YEARS):
            rr, cc = divmod(n_, 7)
            ax = C.ax_mm(fig, xd + cc * (wd + gd), y4 + rr * (hd + gd), wd, hd)
            rgb, _ = MP.render(bA4, A["anom"][y], MP.CM_ANOM, normF, aoa=A["aoa"], smooth=False, w_shade=0.0)
            MP.frame(ax, rgb, bA4.extent, limA); del rgb
            MP.coast(ax, "alaska", tfA, Md, tol_m=2000.0); MP.corner_label(ax, str(y), M, x=0.04, y=0.955, fs=C.FS_SMALL)
            a = A["anom"][y]; f = np.isfinite(a)
            stats_f[str(y)] = dict(mean_cm=round(float(np.nanmean(a)), 3), p01_cm=round(float(np.nanpercentile(a, 1)), 2), p99_cm=round(float(np.nanpercentile(a, 99)), 2),
                                   frac_beyond_15cm=round(float((np.abs(a[f]) > ANOM).mean()), 4))
        C.letter(fig, 0.4, y4 - 4.1, "f"); C.text_mm(fig, 3.8, y4 - 3.95, "Alaska, ALT anomaly from the 2011–2024 mean", fontsize=C.FS, ha="left", va="top")
        cax = C.ax_mm(fig, xd + 2.5 * (wd + gd), y_cb4, 2 * wd + gd, 1.8)
        MP.colorbar(fig, cax, MP.CM_ANOM, normF, "ALT anomaly (cm)", [-15, -10, -5, 0, 5, 10, 15], extend=eF)
    print(f"  f rendered {time.time() - t1:.0f} s")
    # ---------------- QA + export
    rep = C.qa_figure(fig, ink_axes=axe, ignore_ink_for=[str(y) for y in YEARS] + ["km", "Toolik", "Fairbanks", "Samoylov", "Old Crow", "Mackenzie",
                                                                                   "Alaska", "Lena Delta", "Canada (NW)", "n = "])
    C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN)
    plt.close(fig)
    names = {k: C.REGION[byreg[k]["reg"]][0] for k in keys}
    src.update(
        a_b_c=dict(variable="mean ALT 2011-2024 of Physics + residual ML (mean_2011_2024.csv.gz, r1)", colormap=f"cmcrameri oslo_r section {MP.ALT_LUT[0]}-{MP.ALT_LUT[1]}",
                   range_cm=[loM, hiM], extend=eM, ticks_cm=[float(t) for t in ticksM],
                   display="1 km cell centres bilinearly interpolated to the 250 m hillshade grid; gaps of <= 2 cells filled; hillshade weight 0.30",
                   regions={names[k]: dict(panel="abc"[keys.index(k)], n_cells_mapped=int(np.isfinite(R[k]["vmean"]).sum()),
                                           frac_inside_aoa=round(float(R[k]["mean"].aoa_in.mean()), 4), observed_cells_drawn=info[k][0], observed_sites=info[k][1],
                                           windows=info[k][2], scale_bar_km=byreg[k]["scale_km"], window_scale_bar_km=byreg[k]["zoom_scale_km"]) for k in keys}),
        d=dict(definition="mapped cells: mean of the year map over all mapped cells minus the 2011-2024 mean of these yearly means; observed cells: labels (in_forcing == 1) "
                          "matched to the nearest map cell within 1.5 km, observed ALT minus the cell's 2011-2024 map mean, averaged per year (years with >= 50 site-years), "
                          "centred to the predicted anomaly mean of the same years",
               series={byreg[k]["reg"]: dict(mean_alt_cm=round(S_[byreg[k]["reg"]]["mean_cm"], 2), predicted={str(y): round(S_[byreg[k]["reg"]]["pred"][y], 3) for y in YEARS},
                                            observed={str(int(y)): dict(mean=round(float(r_["mean"]), 3), n=int(r_["size"])) for y, r_ in S_[byreg[k]["reg"]]["obs"].iterrows()},
                                            labelled=lab_e[byreg[k]["reg"]], ylim=ylims[byreg[k]["reg"]]) for k in keys}),
        e=dict(variable="ordinary least-squares slope of the yearly maps 2011-2024 per 1 km cell, cm per decade", colormap="cmcrameri broc (olive-brown = thickening, blue = thinning)",
               range_cm_per_decade=[-vt, vt], extend=eT, stats=trend_stats, note="trend of modelled maps driven by the ERA5-Land climate of each year; not an observed trend"),
        f=dict(variable="ALT(year) - mean ALT 2011-2024, per 1 km cell (Alaska)", colormap="cmcrameri vik", range_cm=[-ANOM, ANOM], extend=eF, years=YEARS, per_year=stats_f),
        canvas_mm=[W, round(H, 1)], outputs=C.file_info(paths), pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep,
        style="design/style_tokens_v4.json color.paper_v13 (POLAR_PALETTE=paper13); FreeSans 7 pt labels, 6.5 pt ticks; letters 8 pt bold")
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
