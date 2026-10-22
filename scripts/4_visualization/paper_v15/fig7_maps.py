"""Fig. 7 (v14 manuscript, 2026-10-20): year-labelled 1 km ALT maps 2011-2024 of the physics-residual model in the three study regions.
a  Alaska mean ALT 2011-2024 (mean of the yearly maps), 250 m display interpolation and subtle hillshade, observed cells filled with the mean
   observed ALT of their 2011-2024 labels (v14), outside the area of
   applicability lighter; two 50 km windows (North Slope near Toolik; Interior near Fairbanks) drawn as boxes and enlarged at the right (1 km cells).
b  Lena Delta mean ALT 2011-2024, same drawing; one 24 km window (Samoylov-Kurungnakh) enlarged at the right.
c  Canada (NW) mean ALT 2011-2024, same drawing; two 50 km windows (Old Crow Flats; Mackenzie Delta near Inuvik) enlarged at the right.
   a-c share one colour range (1st-99th percentile of the pooled mean maps, 5 cm steps) and one colour bar.
d  Regional mean ALT anomaly of the yearly maps against the regional mean TDD anomaly of the same year (both relative to 2011-2024; one point per year,
   least-squares line, Pearson r and slope printed; the warmest and coldest thaw seasons labelled), three regions. The observed-cell series of v15.0
   (campaign composition changes between years) was dropped as unclear; site-level observed and mapped series are in Supplementary Fig. S10.
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
             zooms=[("Old Crow Flats", (-139.7, 67.9), 25e3), ("Mackenzie Delta", (-133.5, 68.3), 25e3)], zoom_scale_km=10)]
_sel = os.environ.get("FIG7_REGIONS")
if _sel:
    REGS = [r for r in REGS if r["key"] in _sel.split(",")]


def load_region(key, reg):
    cells, lut, ky0, kx0 = F6.cell_table(key)
    mean = pd.read_csv(MAPS / reg / "mean_2011_2024.csv.gz", dtype={"cell_id": str})
    m = mean.set_index("cell_id")
    vmean = F6.cell_values(key, cells, m.r1); aoa = np.nan_to_num(F6.cell_values(key, cells, m.aoa_in.astype(float)), nan=0.0)
    anom, reg_mean, tdd_mean = {}, {}, {}
    for y in YEARS:
        py = pd.read_csv(MAPS / reg / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1", "e5_tdd"]).set_index("cell_id").reindex(mean.cell_id)
        p = py.r1; reg_mean[y] = float(p.mean()); tdd_mean[y] = float(py.e5_tdd.mean())              # regional means over the mapped cells
        anom[y] = F6.cell_values(key, cells, p) - vmean
    t = np.array(YEARS, float); w = (t - t.mean()) / np.sum((t - t.mean()) ** 2)            # OLS slope weights (cm per year)
    trend = np.zeros_like(vmean)
    for y, wy in zip(YEARS, w):
        trend = trend + wy * np.nan_to_num(anom[y], nan=0.0)
    trend = np.where(np.isfinite(vmean), trend * 10.0, np.nan)                               # cm per decade
    if key != "alaska":                                                                      # the yearly anomaly maps (f) are drawn for Alaska only
        anom = {}
    return dict(cells=cells, lut=lut, ky0=ky0, kx0=kx0, mean=mean, vmean=vmean, aoa=aoa, anom=anom, reg_mean=reg_mean, tdd_mean=tdd_mean, trend=trend)


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
        R = D["R"][r["key"]]; rm = R["reg_mean"]; tm = R["tdd_mean"]
        mu = float(np.mean([rm[y] for y in YEARS])); tmu = float(np.mean([tm[y] for y in YEARS]))
        ya = np.array([rm[y] - mu for y in YEARS]); xa = np.array([tm[y] - tmu for y in YEARS])
        slope, icpt = np.polyfit(xa, ya, 1); rr = float(np.corrcoef(xa, ya)[0, 1])
        out[r["reg"]] = dict(mean_cm=mu, tdd_mean=tmu, tdd_anom={y: float(x) for y, x in zip(YEARS, xa)}, map_anom={y: float(v) for y, v in zip(YEARS, ya)},
                             slope_cm_per_100degd=float(slope * 100.0), intercept_cm=float(icpt), r=rr)
    return out


def draw_series(ax, reg, S_, show_ylabel):
    C.clean_axes(ax)
    s = S_[reg]; xa = np.array([s["tdd_anom"][y] for y in YEARS]); ya = np.array([s["map_anom"][y] for y in YEARS])
    name, col, mk = C.REGION[reg]
    ax.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=1)
    xl = 1.3 * float(np.abs(xa).max()); yl = 1.35 * float(np.abs(ya).max()) + 0.5
    xx = np.array([-xl, xl]); ax.plot(xx, s["intercept_cm"] + s["slope_cm_per_100degd"] / 100.0 * xx, color=C.AUX, lw=C.LW_AUX, zorder=2)
    ax.plot(xa, ya, ls="none", marker="o", ms=3.1, mfc=col, mec="white", mew=0.5, zorder=4)
    for y in (int(YEARS[int(np.argmax(xa))]), int(YEARS[int(np.argmin(xa))])):                 # warmest and coldest thaw seasons
        k = YEARS.index(y); warm = xa[k] > 0                                                 # warm year: label below the point; cold year: above
        ax.text(xa[k], ya[k] - (0.09 * yl if warm else -0.09 * yl), str(y), ha="center", va="top" if warm else "bottom", fontsize=C.FS_SMALL, color=C.AUX)
    ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])); ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")); ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.text(0.03, 0.96, name, transform=ax.transAxes, ha="left", va="top", fontsize=C.FS, color=C.TEXT)
    ax.text(0.97, 0.04, f"r = {s['r']:.2f}\n{s['slope_cm_per_100degd']:+.1f} cm per 100 °C d".replace("-", "−"), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=C.FS_SMALL, color=C.TEXT)
    ax.set_xlabel("TDD anomaly of the year (°C d)")
    if show_ylabel: ax.set_ylabel("Map ALT anomaly (cm)")
    return [float(v) for v in ax.get_xlim()], [float(v) for v in ax.get_ylim()]


def mean_map(fig, M, D, r, R, x0, y0, w, h, norm, wins, wz, lon_lab, lat_lab, letter, title, key_text=None):
    """One mean-ALT map with its windows (placed at wins = [(x, y_top), ...]; labels above the windows). Returns (observed cells drawn, observed sites, window info)."""
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
        wx, wy = wins[j]; axz = C.ax_mm(fig, wx, wy, wz, wz); MP.frame(axz, rgbz, bz.extent, zl); del rgbz
        obs_marks_filled(axz, ox, oy, oval, zl, norm, s=M.obs_zoom_s(wz / ((zl[1] - zl[0]) / 1e3)) * 1.3, lw=0.4, pad_frac=0.012)
        MP.scalebar(axz, zl, r["zoom_scale_km"], M, fx=0.06, fy=0.07); C.text_mm(fig, wx, wy - 0.8, name, fontsize=C.FS_SMALL, ha="left", va="bottom")
        zinfo[name] = dict(centre_lonlat=list(cen), width_km=2 * hx / 1e3, cells_outside_aoa_in_window=F6.zoom_outside(bz, R["vmean"], R["aoa"], zl))
    del b1; gc.collect()
    lx = x0 - 7.5                                                # letter: left of the map, above its latitude labels
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
    W = 180.0; wmap = 46.0; cols = {k: x for k, x in zip(keys, [9.5, 69.0, 128.5])}       # three equal-width columns used by rows a-c, d and e
    hmap = {k: wmap * ASP[k] for k in keys}; hmax = max(hmap.values())
    y1 = 6.5; wz = 20.0; y_win = y1 + hmax + 6.0                                          # windows below the maps, labels above the windows
    y_cb1 = y_win + wz + 3.5
    y2 = y_cb1 + 13.5; hd = 21.0                                                          # d: three scatter panels side by side
    y3 = y2 + hd + 11.5; y_cb3 = y3 + hmax + 3.0                                      # e: trend maps in the same columns
    y4 = y_cb3 + 10.5; gd = 1.0; xd = 12.0; wd = (W - xd - 7.0 - 6 * gd) / 7.0; hd4 = wd * aspA
    y_cb4 = y4 + 2 * hd4 + gd + 2.4
    H = y_cb4 + 8.0
    print(f"  canvas {W} x {H:.1f} mm (rows {y1}, {y2:.1f}, {y3:.1f}, {y4:.1f}; map heights {({k: round(v, 1) for k, v in hmap.items()})})")
    fig = C.figure(W, H)
    # ---------------- a-c: mean maps, one colour range; windows below each map
    allm = np.concatenate([R[k]["mean"].r1.values for k in keys]); loM, hiM = F6.MV.pct_range(allm, step=5.0); nM = Normalize(loM, hiM)
    eM = F6.MV.extend_of(allm, loM, hiM); tick = 20.0 if hiM - loM >= 80 else 10.0
    ticksM = list(np.arange(tick * np.ceil(loM / tick), hiM + 1e-6, tick))
    info = {}
    for n_, k in enumerate(keys):
        wins = [(cols[k] + jj * (wz + 2.0), y_win) for jj in range(len(byreg[k]["zooms"]))]
        info[k] = mean_map(fig, M, D, byreg[k], R[k], cols[k], y1, wmap, hmap[k], nM, wins, wz, byreg[k]["lon_lab"], byreg[k]["lat_lab"],
                           "abc"[n_], f"{C.REGION[byreg[k]['reg']][0]}, mean ALT 2011–2024")
    cax = C.ax_mm(fig, cols[keys[0]], y_cb1, 56.0, 1.8)
    MP.colorbar(fig, cax, MP.CM_ALT, nM, "ALT, mean 2011–2024 (cm)", ticksM, extend=eM)
    hs = [Line2D([], [], ls="none", marker="o", ms=2.4, mfc=MP.CM_ALT(0.55), mec=C.TEXT, mew=0.35), MP.FadedRamp(MP.CM_ALT),
          Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3), Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3)]
    fig.legend(hs, ["Observed cell (observed mean ALT)", "Outside area of applicability", "Not mapped", "Water"], loc="upper left",
               bbox_to_anchor=((cols[keys[0]] + 62.0) / W, 1 - (y_cb1 - 1.4) / H), ncol=2, handlelength=1.6, handleheight=0.9, handletextpad=0.45, labelspacing=0.35,
               columnspacing=1.2, fontsize=C.FS_SMALL, handler_map={MP.FadedRamp: MP.RampHandler()}, borderaxespad=0, borderpad=0)
    # ---------------- d: series (three panels side by side)
    S_ = series(D)
    axe = [C.ax_mm(fig, cols[k], y2, wmap, hd) for k in keys]
    lims_d = {}
    for jj, k in enumerate(keys):
        lims_d[byreg[k]["reg"]] = draw_series(axe[jj], byreg[k]["reg"], S_, jj == 0)
    C.letter(fig, 2.0, y2 - 4.1, "d"); C.text_mm(fig, 5.4, y2 - 3.95, "Regional mean ALT anomaly of the yearly maps against the TDD anomaly of the same year (one point per year, 2011–2024)",
                                                   fontsize=C.FS, ha="left", va="top")
    # ---------------- e: trend maps (three regions, same columns and widths as a-c)
    t1 = time.time()
    allt = np.concatenate([R[k]["trend"][np.isfinite(R[k]["trend"])] for k in keys])
    vt = float(min(15.0, 2.5 * np.ceil(np.nanpercentile(np.abs(allt), 98) / 2.5))); nT = Normalize(-vt, vt)
    eT = F6.MV.extend_of(allt, -vt, vt)
    Mt = MP.Med(); Mt.lw_coast = 0.35
    for k in keys:
        rgb, _ = MP.render(B4[k], R[k]["trend"], CM_TREND, nT, aoa=R[k]["aoa"], smooth=False, w_shade=0.0)
        ax = C.ax_mm(fig, cols[k], y3, wmap, hmap[k]); MP.frame(ax, rgb, B4[k].extent, LIM[k]); del rgb
        MP.coast(ax, k, TF[k], Mt, base=B4[k], cells=R[k]["cells"], tol_m=800.0); MP.graticule(ax, k, TF[k], LIM[k], Mt)
        ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT); MP.scalebar(ax, LIM[k], byreg[k]["scale_km"], M, fx=0.05, fy=0.05)
        MP.corner_label(ax, C.REGION[byreg[k]["reg"]][0], M)
    cax = C.ax_mm(fig, cols[keys[1]] - 3.0, y_cb3, 56.0, 1.8)
    MP.colorbar(fig, cax, CM_TREND, nT, "Trend of the yearly maps (cm per decade)", [-vt, -vt / 2, 0, vt / 2, vt], extend=eT)
    print(f"  e rendered {time.time() - t1:.0f} s")
    C.letter(fig, 2.0, y3 - 4.1, "e"); C.text_mm(fig, 5.4, y3 - 3.95, "Linear trend of the yearly maps 2011–2024 (modelled response to the climate of each year)", fontsize=C.FS, ha="left", va="top")
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
            ax = C.ax_mm(fig, xd + cc * (wd + gd), y4 + rr * (hd4 + gd), wd, hd4)
            rgb, _ = MP.render(bA4, A["anom"][y], MP.CM_ANOM, normF, aoa=A["aoa"], smooth=False, w_shade=0.0)
            MP.frame(ax, rgb, bA4.extent, limA); del rgb
            MP.coast(ax, "alaska", tfA, Md, tol_m=2000.0); MP.corner_label(ax, str(y), M, x=0.04, y=0.955, fs=C.FS_SMALL)
            a = A["anom"][y]; f = np.isfinite(a)
            stats_f[str(y)] = dict(mean_cm=round(float(np.nanmean(a)), 3), p01_cm=round(float(np.nanpercentile(a, 1)), 2), p99_cm=round(float(np.nanpercentile(a, 99)), 2),
                                   frac_beyond_15cm=round(float((np.abs(a[f]) > ANOM).mean()), 4))
        C.letter(fig, 2.0, y4 - 4.1, "f"); C.text_mm(fig, 5.4, y4 - 3.95, "Alaska, ALT anomaly of each year from the 2011–2024 mean", fontsize=C.FS, ha="left", va="top")
        cax = C.ax_mm(fig, xd + 2.5 * (wd + gd), y_cb4, 2 * wd + gd, 1.8)
        MP.colorbar(fig, cax, MP.CM_ANOM, normF, "ALT anomaly (cm)", [-15, -10, -5, 0, 5, 10, 15], extend=eF)
    print(f"  f rendered {time.time() - t1:.0f} s")
    # ---------------- QA + export
    rep = C.qa_figure(fig, ink_axes=axe, ignore_ink_for=[str(y) for y in YEARS] + ["km", "Toolik", "Fairbanks", "Samoylov", "Old Crow", "Mackenzie",
                                                                                   "Alaska", "Lena Delta", "Canada (NW)", "r = "])
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
        d=dict(definition="x: regional mean of the ERA5-Land TDD to 1 September over the mapped cells of the year minus its 2011-2024 mean (degree C d); "
                          "y: regional mean of the year map (column r1) minus the 2011-2024 mean of these yearly means (cm); least-squares line, Pearson r; "
                          "labelled points: warmest and coldest thaw seasons",
               series={byreg[k]["reg"]: dict(mean_alt_cm=round(S_[byreg[k]["reg"]]["mean_cm"], 2), mean_tdd_degd=round(S_[byreg[k]["reg"]]["tdd_mean"], 1),
                                            tdd_anom={str(y): round(S_[byreg[k]["reg"]]["tdd_anom"][y], 2) for y in YEARS},
                                            map_anom={str(y): round(S_[byreg[k]["reg"]]["map_anom"][y], 3) for y in YEARS},
                                            r=round(S_[byreg[k]["reg"]]["r"], 3), slope_cm_per_100degd=round(S_[byreg[k]["reg"]]["slope_cm_per_100degd"], 3),
                                            xlim=lims_d[byreg[k]["reg"]][0], ylim=lims_d[byreg[k]["reg"]][1]) for k in keys}),
        e=dict(variable="ordinary least-squares slope of the yearly maps 2011-2024 per 1 km cell, cm per decade", colormap="cmcrameri broc (olive-brown = thickening, blue = thinning)",
               range_cm_per_decade=[-vt, vt], extend=eT, stats=trend_stats, note="trend of modelled maps driven by the ERA5-Land climate of each year; not an observed trend"),
        f=dict(variable="ALT(year) - mean ALT 2011-2024, per 1 km cell (Alaska)", colormap="cmcrameri vik", range_cm=[-ANOM, ANOM], extend=eF, years=YEARS, per_year=stats_f),
        canvas_mm=[W, round(H, 1)], outputs=C.file_info(paths), pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep,
        style="design/style_tokens_v4.json color.paper_v15 (POLAR_PALETTE=paper15); FreeSans 7 pt labels, 6.5 pt ticks; letters 8 pt bold")
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
