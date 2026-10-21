"""Fig. 6 (v14 manuscript, 2026-10-20): our ALT map of 2013 against two published products in the three study regions, with the observed
values drawn on the maps and an observed-versus-predicted panel per region.
v14 changes (user items 1 and 8 of 2026-10-20): (i) the observed cells are drawn as filled circles coloured by the observed ALT on the map colour
scale, so that map and truth can be compared directly; (ii) the product maps are masked to the cells of our map (same area in the three maps);
(iii) each region has its own colour range from the pooled values of the three maps inside the area of applicability (5th-95th percentile,
5 cm steps), so that the maps are compared on the same scale without the mountain extremes of the products; (iv) the error beeswarm is replaced
by an observed-versus-predicted scatter (1:1 line, r and RMSE per series).

Rows: Alaska (a-d), Lena Delta (e-h), Canada (NW) (i-l); columns: our map, Wei 2026, Liu 2024, observed versus predicted.
  maps     Physics + residual ML 1 km map of 2013 (model fitted to all labels, TDD accumulated to 1 September,
           data/processed/v5/maps/<Region>/pred_2013.csv.gz, column r1), display interpolation to the hillshade grid (250 m Lena Delta, 500 m
           Alaska and Canada), subtle hillshade, cells outside the area of applicability lighter (mean_2011_2024.csv.gz, aoa_in).
           Wei 2026 (data/raw/wei2026_alt_v2/extracted/ALT_2013.tif, EPSG:3995, 1 km) and Liu 2024 (data/raw/liu2024_alt/ALT_2013.tif, 1/120 deg)
           bilinearly reprojected to the same display grid and masked to the drawn cells of our map.
  scatter  Observed ALT (cell mean of the 2013 labels, labels_cell_year_v5.csv in_forcing == 1) against the prediction of each map at the common
           cells (cells where our map, Wei 2026 and Liu 2024 all have a value). Our series = spatial-block out-of-fold predictions
           (core_v6, three seeds averaged per label, then the cell mean), not the in-sample map; product series = product value of 2013 at the
           label sites (products_cell_year_v5.csv), cell mean. r = Pearson correlation, RMSE over the common cells.
Outputs: outputs/figures/paper/v15/Fig6_product_maps.{pdf,svg,png}, outputs/figures/paper/v15/source_data/Fig6_product_maps.json
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 \\
       python3 scripts/4_visualization/paper_v15/fig6_product_maps.py        (FIG6_ROWS=alaska,lena restricts the rows for a test render)
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
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import v8_f8_product_maps as F8  # noqa: E402

F6 = MP.F6
MV = F6.MV
STEM = "Fig6_product_maps"
PROC = C.ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"
CELL = F6.CELL
W_SHADE = 0.30
YEARS = {"alaska": 2013, "lena": 2016, "canada": 2013}   # v14.1: per region, the year with the most observed 1 km cells that have a value in our map and in both products
WHY_YEAR = ("per region, the year with the most observed 1 km cells with a value in our map, Wei 2026 and Liu 2024 (Liu 2024 ends in 2020); "
            "Alaska 2013 (126 common cells; next 2018 and 2014), Lena Delta 2016 (14; next 2014), Canada 2013 (33; other years at most 12)")
FAR_KM = 25.0                                   # scatter: cells farther than this from the nearest Wei 2026 training site are drawn open (as Fig. 5a)
ROWS = [dict(key="alaska", region="Alaska", lon_lab=(-160, -150), lat_lab=(60, 65, 70), scale_km=200, letters="abcd", step=2,
             lat_edges=(59, 64, 66, 68, 70, 72)),
        dict(key="lena", region="Lena", lon_lab=(124, 128), lat_lab=(72, 73), scale_km=50, letters="efgh", step=1,
             lat_edges=(71, 72, 73, 74)),
        dict(key="canada", region="Canada", lon_lab=(-135, -125, -115), lat_lab=(60, 65), scale_km=200, letters="ijkl", step=2,
             lat_edges=(59, 64, 66, 68, 70, 72))]
for _r in ROWS:
    _r["year"] = YEARS[_r["key"]]
_sel = os.environ.get("FIG6_ROWS")
if _sel:
    ROWS = [r for r in ROWS if r["key"] in _sel.split(",")]
RANGE_PCT = (5.0, 95.0)                         # per-row colour range: pooled values of the three maps inside the area of applicability
RANGE_STEP = 5.0
ELEV_SPLIT = 600.0                              # band statistics: elevation split (m)
SERIES = [("oof", "Physics + residual ML", C.METHOD["R1"]), ("wei", "Wei 2026", C.PRODUCT["wei"][1]), ("liu", "Liu 2024", C.PRODUCT["liu"][1])]
MAP_HEAD = {"ours": C.METHOD_NAME["R1"], "wei": "Wei 2026", "liu": "Liu 2024"}
OOF = "{region}__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz"
LEAK = C.ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv"
OBS_PT = 2.7                                    # observed-cell circles on the maps (filled with the observed ALT colour): diameter (pt)
SC_PT = 2.5                                     # scatter markers (pt)


# ================================================================ data
def render_px(base, vpx, cmap, norm, drawn, aoa=None, w_shade=W_SHADE):
    """Pixel values -> RGB: LAND outside `drawn` on land, WATER on water, colours (faded outside the area of applicability) x hillshade on drawn pixels."""
    water = base.water_cell | (~base.land & ~drawn)
    rgb = np.empty(base.idx.shape + (3,), np.float32); rgb[:] = MP.LAND; rgb[water] = MP.WATER
    v = np.where(drawn & np.isfinite(vpx), vpx, float(np.nanmean(vpx[drawn])) if drawn.any() else 0.0)
    col = np.empty(base.idx.shape + (3,), np.float32)
    for r0 in range(0, v.shape[0], 512):
        col[r0:r0 + 512] = cmap(norm(v[r0:r0 + 512]))[..., :3]
    if aoa is not None:
        f = (MP.FADE * (1.0 - np.clip(aoa, 0.0, 1.0)))[..., None]; col = col * (1.0 - f) + f * MP.LAND
    if w_shade > 0:
        col = col * ((1 - w_shade) + w_shade * np.where(base.valid, base.shade, base.shade_fill))[..., None]
    rgb[drawn] = col[drawn]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def error_table(row, map_r1, tf):
    """Observed cells of the year: label rows -> 1 km cells (cell means). Columns obs, oof (block CV, seeds averaged), wei, liu, mapv (r1 map value, in-sample)."""
    region, yr = row["region"], row["year"]
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    pr = pd.read_csv(PROC / "v5" / "products_cell_year_v5.csv")
    l = lab[(lab.region == region) & (lab.year == yr)].copy()
    ky = np.floor(l.lat.values / CELL).astype(np.int64); dx = CELL / np.cos(np.radians((ky + 0.5) * CELL)); kx = np.floor(l.lon.values / dx).astype(np.int64)
    l["ky"], l["kx"] = ky, kx; l["cell_id"] = [f"{a}_{b}" for a, b in zip(ky, kx)]
    for p in ("wei", "liu"):
        l[p] = pr[(pr["product"] == p) & (pr.year == yr)].set_index("loc_id").value_cm.reindex(l.loc_id).values
    o = pd.read_csv(PROC / "v5" / "runs" / "core_v6" / "preds" / OOF.format(region=region), usecols=["loc_id", "year", "pred", "seed"])
    o = o[o.year == yr]
    n_seed = o.groupby("loc_id").seed.nunique()
    l["oof"] = o.groupby("loc_id").pred.mean().reindex(l.loc_id).values
    l["mapv"] = map_r1.reindex(l.cell_id).values
    lk = pd.read_csv(LEAK); lk = lk[lk["product"] == "wei"].groupby("loc_id").dist_km.min()
    l["dist_wei_km"] = l.loc_id.map(lk).values
    c = l.groupby("cell_id").agg(ky=("ky", "first"), kx=("kx", "first"), n_lab=("alt_cm", "size"), obs=("alt_cm", "mean"), oof=("oof", "mean"),
                                 wei=("wei", "mean"), liu=("liu", "mean"), mapv=("mapv", "mean"), dist_wei_km=("dist_wei_km", "min"))
    lat = (c.ky.values + 0.5) * CELL; lon = (c.kx.values + 0.5) * CELL / np.cos(np.radians(lat))     # 1 km cell centre (as F6.obs_cells)
    c["x"], c["y"] = tf.transform(lon, lat); c["lat_c"], c["lon_c"] = lat, lon
    for m in ("oof", "wei", "liu", "mapv"):
        c[f"e_{m}"] = c[m] - c.obs
    common = c[c[["mapv", "wei", "liu"]].notna().all(axis=1)].copy()
    rmse = lambda q, m: float(np.sqrt(np.mean(q[f"e_{m}"] ** 2))) if len(q) else float("nan")
    corr = lambda q, m: (float(np.corrcoef(q.obs.values, q[m].values)[0, 1]) if len(q) >= 3 else float("nan"))
    far = common[common.dist_wei_km > FAR_KM]; near = common[common.dist_wei_km <= FAR_KM]
    out = dict(cells=c, common=common, n_labels=int(len(l)), n_cells=int(len(c)), n_common=int(len(common)),
               n_seeds=sorted(int(v) for v in n_seed.unique()),
               rmse={m: rmse(common, m) for m in ("oof", "wei", "liu", "mapv")}, r={m: corr(common, m) for m in ("oof", "wei", "liu", "mapv")},
               bias={m: float(common[f"e_{m}"].mean()) for m in ("oof", "wei", "liu", "mapv")},
               n_with_value={m: int(c[m].notna().sum()) for m in ("oof", "wei", "liu", "mapv")},
               training_sites=dict(note=f"distances are to the nearest Wei 2026 training site (the {FAR_KM:.0f} km mask of Fig. 5a); scatter markers: filled = within, open = beyond",
                                   distance_source=str(LEAK.relative_to(C.ROOT)) + " (product == 'wei', nearest Wei 2026 training site per label; cell value = minimum over its labels)",
                                   n_common_within_25km=int(len(near)), n_common_gt25km=int(len(far)),
                                   rmse_common_gt25km_cm=({m: round(rmse(far, m), 2) for m in ("oof", "wei", "liu")} if len(far) else None),
                                   r_common_gt25km=({m: round(corr(far, m), 3) for m in ("oof", "wei", "liu")} if len(far) >= 3 else None),
                                   rmse_common_within25km_cm=({m: round(rmse(near, m), 2) for m in ("oof", "wei", "liu")} if len(near) else None),
                                   r_common_within25km=({m: round(corr(near, m), 3) for m in ("oof", "wei", "liu")} if len(near) >= 3 else None)),
               far=far, near=near)
    return out


def product_cell_means(base, vpx, mask, n_cells):
    """Product pixel values -> 1 km cell means (NaN where the cell has no product pixel)."""
    m = (base.idx >= 0) & mask & np.isfinite(vpx); idx = base.idx[m]
    s = np.bincount(idx, weights=vpx[m].astype(np.float64), minlength=n_cells); n = np.bincount(idx, minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def band_stats(row, cells, r1, prod_cell, elev):
    """Mean difference product - our map by latitude band and elevation class over the map cells with a value in both (cell level)."""
    lat = (cells.ky.values + 0.5) * CELL; ok = np.isfinite(r1) & np.isfinite(elev)
    out = []
    edges = row["lat_edges"]
    for a, b in zip(edges[:-1], edges[1:]):
        for mtn, sel in ((f"elev<={ELEV_SPLIT:.0f}m", elev <= ELEV_SPLIT), (f"elev>{ELEV_SPLIT:.0f}m", elev > ELEV_SPLIT)):
            q = ok & sel & (lat > a) & (lat <= b)
            rec = dict(band=f"({a}, {b}]", mtn=mtn, n_cells=int(q.sum()), ours_mean_cm=(round(float(np.nanmean(r1[q])), 2) if q.any() else None))
            for p in ("wei", "liu"):
                qq = q & np.isfinite(prod_cell[p])
                rec[f"{p}_minus_ours_cm"] = round(float(np.mean(prod_cell[p][qq] - r1[qq])), 2) if qq.any() else None
                rec[f"n_cells_{p}"] = int(qq.sum())
            out.append(rec)
    return out


def load_row(row):
    t0 = time.time(); key, region, yr = row["key"], row["region"], row["year"]
    cells, lut, ky0, kx0 = F6.cell_table(key)
    pred = pd.read_csv(MAPS / region / f"pred_{yr}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id")
    mean = pd.read_csv(MAPS / region / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"]).set_index("cell_id")
    elev = pd.read_csv(PROC / f"map_{key}" / f"{key}_grid_x25_v1.csv.gz", usecols=["cell_id", "dem_elev"], dtype={"cell_id": str}).set_index("cell_id")
    r1 = F6.cell_values(key, cells, pred.r1)
    aoa = np.nan_to_num(F6.cell_values(key, cells, mean.aoa_in.astype(float)), nan=0.0)
    elev = elev.dem_elev.reindex(cells.cell_id.values).values.astype(np.float32)
    base = F6.Base(key, cells, lut, ky0, kx0, step=row["step"], tag="full"); tf = base.tf; lim = F6.region_limits(key, tf)
    base.lat = base.lon = None
    geo = F8.geo_index(base, ky0, kx0, "full"); G = lambda v: F8.grid2d(cells, lut.shape, ky0, kx0, v)
    v_r1, drawn, _ = F8.nearest_px(base, r1)
    px_r1 = F8.interp_px(G(r1), geo); px_r1 = np.where(np.isfinite(px_r1), px_r1, v_r1)
    px_aoa = np.nan_to_num(F8.interp_px(G(aoa), geo), nan=0.0)
    del geo, v_r1; gc.collect()
    prod = {p: F8.product_px(base, p, yr) for p in ("wei", "liu")}
    pm = {p: F8.product_masks(base, prod[p])[0] & drawn for p in ("wei", "liu")}          # v14: products masked to the drawn cells of our map
    prod_cell = {p: product_cell_means(base, prod[p], pm[p], len(cells)) for p in prod}
    in_aoa = drawn & (px_aoa > 0.5)
    pooled = np.concatenate([px_r1[in_aoa]] + [prod[p][pm[p] & in_aoa] for p in prod])
    lo = RANGE_STEP * np.floor(np.percentile(pooled, RANGE_PCT[0]) / RANGE_STEP); hi = RANGE_STEP * np.ceil(np.percentile(pooled, RANGE_PCT[1]) / RANGE_STEP)
    allv = np.concatenate([px_r1[drawn]] + [prod[p][pm[p]] for p in prod]); ext = MV.extend_of(allv, lo, hi)
    tick = 20.0 if hi - lo >= 70 else 10.0; ticks = [t for t in np.arange(tick * np.ceil(lo / tick), hi + 1e-6, tick)]
    fin = r1[np.isfinite(r1)]
    err = error_table(row, pred.r1, tf)
    bands = band_stats(row, cells, r1, prod_cell, elev)
    D = dict(row, year=yr, cells=cells, base=base, tf=tf, lim=lim, asp=(lim[3] - lim[2]) / (lim[1] - lim[0]), px_r1=px_r1, px_aoa=px_aoa, drawn=drawn,
             prod=prod, pm=pm, lo=float(lo), hi=float(hi), norm=Normalize(lo, hi), ticks=ticks, ext=ext, err=err, bands=bands,
             n_cells_drawn=int(np.isfinite(r1).sum()),
             pooled_pct={f"p{int(q)}": round(float(np.percentile(pooled, q)), 1) for q in (2, 5, 50, 95, 98)},
             series_pct={k: [round(float(np.percentile(v, 5)), 1), round(float(np.percentile(v, 50)), 1), round(float(np.percentile(v, 95)), 1)]
                         for k, v in (("ours_in_aoa", px_r1[in_aoa]),) + tuple((f"{p}_in_aoa", prod[p][pm[p] & in_aoa]) for p in prod)},
             frac_beyond={"ours": [round(float((px_r1[drawn] < lo).mean()), 4), round(float((px_r1[drawn] > hi).mean()), 4)]}
             | {p: [round(float((prod[p][pm[p]] < lo).mean()), 4), round(float((prod[p][pm[p]] > hi).mean()), 4)] for p in prod},
             prod_cells_with_value={p: int(np.isfinite(prod_cell[p][np.isfinite(r1)]).sum()) for p in prod})
    print(f"  load {region} {yr}: range {lo}-{hi} ({ext}), pooled pct {D['pooled_pct']}, series p5/p50/p95 {D['series_pct']}, labels {err['n_labels']} -> "
          f"cells {err['n_cells']}, common {err['n_common']}, RMSE {({k: round(v, 2) for k, v in err['rmse'].items()})}, r {({k: round(v, 2) for k, v in err['r'].items()})}, "
          f"{time.time() - t0:.0f} s")
    return D


# ================================================================ panels
def draw_map(ax, Dk, kind, M, lon_lab=(), lat_lab=(), scale=False):
    b = Dk["base"]; norm = Dk["norm"]
    if kind == "ours":
        rgb = render_px(b, Dk["px_r1"], MP.CM_ALT, norm, Dk["drawn"], aoa=Dk["px_aoa"])
    else:
        rgb = render_px(b, Dk["prod"][kind], MP.CM_ALT, norm, Dk["pm"][kind])
    MP.frame(ax, rgb, b.extent, Dk["lim"]); del rgb; gc.collect()
    MP.coast(ax, Dk["key"], Dk["tf"], M, base=Dk["base"], cells=Dk["cells"], tol_m=600.0)
    MP.graticule(ax, Dk["key"], Dk["tf"], Dk["lim"], M, lon_lab=lon_lab, lat_lab=lat_lab)
    ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
    cm = Dk["err"]["common"]                                                   # observed cells: filled with the observed ALT colour (same scale)
    ax.scatter(cm.x.values, cm.y.values, c=cm.obs.values, cmap=MP.CM_ALT, norm=norm, s=OBS_PT ** 2, edgecolors=C.TEXT, linewidths=0.35, zorder=6)
    if scale:
        MP.scalebar(ax, Dk["lim"], Dk["scale_km"], M)


def draw_scatter(fig, ax, Dk, bottom):
    C.clean_axes(ax)
    cm = Dk["err"]["common"]; e = Dk["err"]
    top = 20.0 * np.ceil((max(float(cm.obs.max()), float(cm[["oof", "wei", "liu"]].max().max())) * 1.04) / 20.0)
    ax.plot([0, top], [0, top], color=C.ZERO, lw=C.LW_REF, zorder=1)
    near = cm.dist_wei_km <= FAR_KM; far = ~near
    for j, (m, name, col) in enumerate(SERIES):
        if near.any():
            ax.scatter(cm.obs.values[near], cm[m].values[near], s=SC_PT ** 2, facecolor=col, edgecolor="white", lw=0.25, alpha=0.9, zorder=3 + j)
        if far.any():
            ax.scatter(cm.obs.values[far], cm[m].values[far], s=SC_PT ** 2, facecolor="white", edgecolor=col, lw=0.5, alpha=0.95, zorder=3 + j)
    ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal")
    loc = mticker.MultipleLocator(40.0 if top > 100 else 20.0); ax.xaxis.set_major_locator(loc); ax.yaxis.set_major_locator(mticker.MultipleLocator(40.0 if top > 100 else 20.0))
    ax.set_ylabel("Predicted ALT (cm)")
    ax.set_xlabel(f"Observed ALT (cm), n = {e['n_common']}", labelpad=1.5)
    ts = e["training_sites"]; two = ts["rmse_common_gt25km_cm"] is not None and ts["n_common_gt25km"] >= 3 and ts["n_common_within_25km"] >= 3
    # statistics table below the panel (in the band that holds the colour bar of the maps): header + one line per series, nothing inside the axes
    head = f"RMSE (cm), all | >{FAR_KM:.0f} km" if two else "RMSE (cm)"
    vals = [(f"{e['rmse'][m]:.1f} | {ts['rmse_common_gt25km_cm'][m]:.1f}" if two else f"{e['rmse'][m]:.1f}", name, col) for m, name, col in SERIES]
    Dk["stats_lines"] = [(head, C.TEXT)] + [(f"{name}: {v}", col) for v, name, col in vals]
    Dk["stats_corner"] = "below"; Dk["stats_marker_hits"] = 0
    return top


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    D = {r["key"]: load_row(r) for r in ROWS}
    print(f"  data {time.time() - t0:.0f} s")
    keys = [r["key"] for r in ROWS]
    # ---------------- layout (mm)
    W = 180.0
    xm0, gm = 11.0, 1.6                        # first map left edge, gap between maps
    ws, ml_s, mr = 32.0, 14.5, 3.4             # scatter panel width, its left margin (ticks + y label + separation from the maps), right margin
    wm = (W - xm0 - 2 * gm - ml_s - ws - mr) / 3.0
    xs = W - mr - ws
    y_head = 7.4                               # top band: letters and column titles 6.9 mm above the maps
    band = 18.5                                # band under each row: colour bar (maps) and the statistics table (scatter)
    gap_row = 4.6                              # letters of the next row
    ys, hs = [], []
    y = y_head
    for k in keys:
        h = wm * D[k]["asp"]; ys.append(y); hs.append(h); y += h + band + gap_row
    y_key = y - gap_row + 1.0
    H = y_key + 8.2                            # two legend rows
    fig = C.figure(W, H)
    out = {}
    for i, k in enumerate(keys):
        Dk = D[k]; y0, hrow = ys[i], hs[i]; lt = Dk["letters"]; name = C.REGION[Dk["region"]][0]
        for j, kind in enumerate(("ours", "wei", "liu")):
            xm = xm0 + j * (wm + gm)
            ax = C.ax_mm(fig, xm, y0, wm, hrow)
            draw_map(ax, Dk, kind, M, lon_lab=Dk["lon_lab"] if j == 0 else (), lat_lab=Dk["lat_lab"] if j == 0 else (), scale=(j == 0))
            C.letter(fig, xm, y0 - (6.9 if i == 0 else 4.4), lt[j])
            if i == 0:
                C.text_mm(fig, xm + 3.3, y0 - 6.75, MAP_HEAD[kind], fontsize=C.FS, ha="left", va="top")
            gc.collect()
        C.text_mm(fig, 2.0, y0 + hrow / 2, name, rotation=90, ha="center", va="center", fontsize=C.FS)
        # colour bar of the row under the Wei and Liu maps
        xc0 = xm0 + wm + gm; wcb = 2 * wm + gm
        cax = C.ax_mm(fig, xc0 + 0.18 * wcb, y0 + hrow + 2.6, 0.64 * wcb, 1.7)
        MP.colorbar(fig, cax, MP.CM_ALT, Dk["norm"], f"ALT {Dk['year']} (cm)", Dk["ticks"], extend=Dk["ext"])
        # observed versus predicted
        axs = C.ax_mm(fig, xs, y0, ws, hrow)
        top = draw_scatter(fig, axs, Dk, bottom=(i == len(keys) - 1))
        for q_, (txt, col) in enumerate(Dk["stats_lines"]):                     # statistics under the x-axis label, left aligned with the panel
            C.text_mm(fig, xs, y0 + hrow + 7.6 + q_ * 2.5, txt, fontsize=C.FS_SMALL, ha="left", va="top", color=col)
        C.letter(fig, xs - ml_s + 0.2, y0 - (6.9 if i == 0 else 4.4), lt[3])
        if i == 0:
            C.text_mm(fig, xs - ml_s + 3.5, y0 - 6.75, "Observed vs predicted", fontsize=C.FS, ha="left", va="top")
        e = Dk["err"]
        out[Dk["region"]] = dict(year=Dk["year"], letters=lt, colour_range_cm=[Dk["lo"], Dk["hi"]], extend=Dk["ext"], ticks_cm=[float(t) for t in Dk["ticks"]],
                                 pooled_percentiles_inside_aoa_cm=Dk["pooled_pct"], series_p5_p50_p95_inside_aoa_cm=Dk["series_pct"],
                                 share_of_drawn_pixels_beyond_range=Dk["frac_beyond"], n_map_cells_drawn=Dk["n_cells_drawn"],
                                 n_map_cells_with_product_value=Dk["prod_cells_with_value"], display_pixel_m=250 * Dk["step"],
                                 n_labels=e["n_labels"], n_observed_cells=e["n_cells"], n_cells_with_value=e["n_with_value"], n_common_cells=e["n_common"],
                                 oof_seeds_per_label=e["n_seeds"], scatter_axis_top_cm=top, stats_text_corner=Dk.get("stats_corner"), stats_text_marker_hits=Dk.get("stats_marker_hits"),
                                 rmse_common_cm=dict(physics_residual_ml_block_cv=round(e["rmse"]["oof"], 2), wei_2026=round(e["rmse"]["wei"], 2),
                                                     liu_2024=round(e["rmse"]["liu"], 2), our_map_in_sample_not_shown=round(e["rmse"]["mapv"], 2)),
                                 r_common=dict(physics_residual_ml_block_cv=round(e["r"]["oof"], 3), wei_2026=round(e["r"]["wei"], 3), liu_2024=round(e["r"]["liu"], 3),
                                               our_map_in_sample_not_shown=round(e["r"]["mapv"], 3)),
                                 bias_common_cm=dict(physics_residual_ml_block_cv=round(e["bias"]["oof"], 3), wei_2026=round(e["bias"]["wei"], 3),
                                                     liu_2024=round(e["bias"]["liu"], 3)),
                                 product_training_sites=e["training_sites"],
                                 band_differences=dict(definition=f"mean (product − our map) over the 1 km map cells of the latitude band with a value in both, "
                                                                  f"split at {ELEV_SPLIT:.0f} m elevation (grid dem_elev); product cell value = mean of its display pixels",
                                                       rows=Dk["bands"]),
                                 cells=[dict(cell_id=str(ci), lat=round(float(r.lat_c), 4), lon=round(float(r.lon_c), 4), n_labels=int(r.n_lab),
                                             observed_cm=round(float(r.obs), 2), physics_residual_ml_block_cv_cm=round(float(r.oof), 2),
                                             wei_2026_cm=round(float(r.wei), 2), liu_2024_cm=round(float(r.liu), 2), our_map_cm=round(float(r["mapv"]), 2),
                                             dist_wei_training_site_km=(None if not np.isfinite(r.dist_wei_km) else round(float(r.dist_wei_km), 2)))
                                        for ci, r in e["common"].iterrows()])
    # ---------------- key (one line)
    hs_ = [Line2D([], [], ls="none", marker="o", ms=OBS_PT + 0.3, mfc=MP.CM_ALT(0.55), mec=C.TEXT, mew=0.4), MP.FadedRamp(MP.CM_ALT),
           Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3), Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3),
           Line2D([], [], ls="none", marker="o", ms=SC_PT + 0.3, mfc=C.AXIS, mec="white", mew=0.3),
           Line2D([], [], ls="none", marker="o", ms=SC_PT + 0.3, mfc="white", mec=C.AXIS, mew=0.6)]
    labs = ["Observed cell, filled with the observed ALT", "Outside area of applicability (lighter)", "Not mapped (products masked to our cells)", "Water",
            f"Cell within {FAR_KM:.0f} km of a Wei 2026 training site", f"Cell beyond {FAR_KM:.0f} km (d, h, l; statistics for both)"]
    fig.legend(hs_, labs, loc="upper left", bbox_to_anchor=(xm0 / W, 1 - y_key / H), ncol=3, fontsize=C.FS_SMALL, handlelength=2.2, handleheight=0.9,
               handletextpad=0.5, columnspacing=1.5, borderaxespad=0.0, handler_map={MP.FadedRamp: MP.RampHandler()})
    rep = C.qa_figure(fig, text_ink=False, allow_colors=tuple(c for _, _, c in SERIES))
    C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN)
    plt.close(fig)
    src = dict(figure="Fig. 6", script="scripts/4_visualization/paper_v15/fig6_product_maps.py", version="v14.1 (2026-10-21): per-region year, training-site distance split",
               inputs=dict(our_map="data/processed/v5/maps/<Region>/pred_<year>.csv.gz (column r1; model fitted to all labels, TDD to 1 September); mean_2011_2024.csv.gz (aoa_in)",
                           products="data/raw/wei2026_alt_v2/extracted/ALT_<year>.tif, data/raw/liu2024_alt/ALT_<year>.tif",
                           labels="data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)",
                           product_values_at_labels="data/processed/v5/products_cell_year_v5.csv (product wei / liu, year of the figure)",
                           our_errors="data/processed/v5/runs/core_v6/preds/" + OOF.format(region="<Region>") + " (spatial-block CV, 3 seeds averaged)",
                           elevation="data/processed/map_<region>/<region>_grid_x25_v1.csv.gz (dem_elev) for the band differences"),
               year_used={C.REGION[D[k]["region"]][0]: D[k]["year"] for k in keys}, rows=[C.REGION[D[k]["region"]][0] for k in keys],
               why_this_year=WHY_YEAR,
               definition=dict(colour_range=f"per region: {RANGE_PCT[0]:.0f}th-{RANGE_PCT[1]:.0f}th percentile of the pooled display pixels of our map, Wei 2026 and Liu 2024 "
                                            f"inside our area of applicability, rounded outwards to {RANGE_STEP:.0f} cm; values beyond are drawn at the end colours",
                               product_mask="product pixels are drawn only where our map has a value (same area in the three maps)",
                               observed_cells="common cells (our map, Wei 2026 and Liu 2024 all have a value) drawn as filled circles coloured by the observed ALT on the same scale",
                               scatter="observed cell mean against the prediction of each series at the common cells; our series = block-CV out-of-fold predictions "
                                       "(seeds averaged per label, then cell mean), not the in-sample map; r = Pearson correlation; RMSE over the common cells; "
                                       f"markers filled = cell within {FAR_KM:.0f} km of a Wei 2026 training site, open = beyond; the panel prints the RMSE for all cells and for the open cells"),
               values=out, numbers_printed={r: {"r": [f"{v['r_common']['physics_residual_ml_block_cv']:.2f}", f"{v['r_common']['wei_2026']:.2f}", f"{v['r_common']['liu_2024']:.2f}"],
                                                "RMSE (cm)": [f"{v['rmse_common_cm']['physics_residual_ml_block_cv']:.1f}", f"{v['rmse_common_cm']['wei_2026']:.1f}",
                                                              f"{v['rmse_common_cm']['liu_2024']:.1f}"], "n": v["n_common_cells"]} for r, v in out.items()},
               outputs=C.file_info(paths), canvas_mm=[W, round(H, 1)], pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep,
               layout_mm=dict(map_width=round(wm, 2), map_heights=[round(h, 2) for h in hs], scatter_panel=[round(xs, 2), ws]),
               style="cmcrameri oslo_r section %s for ALT; series colours: Physics + residual ML %s, Wei 2026 %s, Liu 2024 %s" %
                     (MP.ALT_LUT, C.METHOD["R1"], C.PRODUCT["wei"][1], C.PRODUCT["liu"][1]))
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
