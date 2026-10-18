"""Fig. 6 (v11 manuscript, main text): our yearly ALT map against two published products in the same year, and the error at the observed cells.
Copied from scripts/4_visualization/paper_v10/si_s06_s07_product_maps.py (Supplementary Fig. S6 of v10) and rebuilt for the main text.

Rows: Alaska 2014 (a-d) and Lena Delta 2016 (e-h).
  a, e  Physics + residual ML 1 km map of that year (full-data model, data/processed/v5/maps/<Region>/pred_<year>.csv.gz, column r1), 250 m display
        interpolation, subtle hillshade, cells outside the area of applicability lighter.
  b, f  Wei 2026 ALT of the same year (data/raw/wei2026_alt_v2/extracted/ALT_<year>.tif, EPSG:3995, 1 km), bilinear reprojection to the same grid.
  c, g  Liu 2024 ALT of the same year (data/raw/liu2024_alt/ALT_<year>.tif, 1/120 deg), same reprojection.
        a-c (e-g) share one colour range: 1st-99th percentile of our map (5 or 10 cm steps), cmcrameri oslo_r; values beyond are drawn at the
        end colours (extend arrows).
  d, h  Error at the observed cells of that year (predicted - observed, one dot per 1 km cell): labels (labels_cell_year_v5.csv, in_forcing == 1)
        aggregated to 1 km map cells (cell mean of observed ALT). Our series uses the spatial-block cross-validation out-of-fold predictions
        (data/processed/v5/runs/core_v6/preds/<Region>__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz, three seeds averaged per label,
        then the cell mean) and not the map values (the map is trained on all labels, so map values at observed cells are in-sample).
        Wei 2026 and Liu 2024: product values of that year at the label sites (data/processed/v5/products_cell_year_v5.csv), cell mean.
        Common cells = observed cells where our map, Wei 2026 and Liu 2024 all have a value (the cells marked on the maps). RMSE over the common
        cells is printed above each column. Values beyond the axis are drawn at the axis end and printed.
Inputs: the files above, data/processed/v5/maps/<Region>/mean_2011_2024.csv.gz (aoa_in), data/processed/map_<region>/ (grid, hillshade);
        reprojection caches in outputs/figures/paper/v7/_cache (read through symlinks in outputs/figures/paper/v11/_cache).
Outputs: outputs/figures/paper/v11/Fig6_product_maps.{pdf,svg,png}, outputs/figures/paper/v11/source_data/Fig6_product_maps.json
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper11 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v11/fig6_product_maps.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c11_common as C  # noqa: E402
import c11_maps as MP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
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
ROWS = [dict(key="alaska", region="Alaska", year=2014, lon_lab=(-160, -150), lat_lab=(60, 65, 70), scale_km=200, alt_step=5.0, letters="abcd",
             err_lim=(-55.0, 45.0), err_step=20.0),
        dict(key="lena", region="Lena", year=2016, lon_lab=(124, 128), lat_lab=(72, 73), scale_km=50, alt_step=10.0, letters="efgh",
             err_lim=(-20.0, 55.0), err_step=20.0)]
SERIES = [("oof", "Physics +\nresidual ML\n(block CV)", C.METHOD["R1"]), ("wei", "Wei 2026", C.PRODUCT["wei"][1]),
          ("liu", "Liu 2024", C.PRODUCT["liu"][1])]
MAP_HEAD = {"ours": C.METHOD_NAME["R1"], "wei": "Wei 2026", "liu": "Liu 2024"}
OOF = "{region}__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz"
LEAK = C.ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv"
DOT_PT = 2.1                                   # error dots: diameter (pt)
OBS_PT = 1.9                                   # observed-cell circles on the maps: diameter (pt)
OUT_GREY = "#6B6B6B"                            # values beyond the error axis (printed above the open markers on the spine)


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
    alt = c[c[["oof", "wei", "liu"]].notna().all(axis=1)]
    rmse = lambda q, m: float(np.sqrt(np.mean(q[f"e_{m}"] ** 2)))
    far = common[common.dist_wei_km > 25.0]
    out = dict(cells=c, common=common, n_labels=int(len(l)), n_cells=int(len(c)), n_common=int(len(common)),
               n_seeds=sorted(int(v) for v in n_seed.unique()),
               rmse={m: rmse(common, m) for m in ("oof", "wei", "liu", "mapv")},
               bias={m: float(common[f"e_{m}"].mean()) for m in ("oof", "wei", "liu", "mapv")},
               n_with_value={m: int(c[m].notna().sum()) for m in ("oof", "wei", "liu", "mapv")},
               alt_common=dict(definition="cells where the block-CV prediction, Wei 2026 and Liu 2024 all have a value (no condition on our map)",
                               n=int(len(alt)), rmse_cm={m: round(rmse(alt, m), 2) for m in ("oof", "wei", "liu")}),
               label_level={m: dict(rmse=round(float(np.sqrt(np.nanmean((l[m] - l.alt_cm) ** 2))), 2), n=int(l[m].notna().sum()))
                            for m in ("oof", "wei", "liu", "mapv")},
               training_sites=dict(note="no distance mask in this figure: Wei 2026 and Liu 2024 values at all common cells are used, including cells at "
                                        "or near Wei 2026 training sites; distances are to Wei 2026 training sites (supplement table), the same "
                                        "distance used for the 25 km mask of Fig. 5",
                                   distance_source=str(LEAK.relative_to(C.ROOT)) + " (product == 'wei', nearest Wei 2026 training site per label; "
                                                   "cell value = minimum over its labels)",
                                   n_common_within_25km=int((common.dist_wei_km <= 25.0).sum()), n_common_within_1km=int((common.dist_wei_km <= 1.0).sum()),
                                   n_common_gt25km=int(len(far)),
                                   rmse_common_gt25km_cm_not_shown=({m: round(rmse(far, m), 2) for m in ("oof", "wei", "liu")} if len(far) else None)))
    return out


def load_row(row):
    t0 = time.time(); key, region, yr = row["key"], row["region"], row["year"]
    cells, lut, ky0, kx0 = F6.cell_table(key)
    pred = pd.read_csv(MAPS / region / f"pred_{yr}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id")
    mean = pd.read_csv(MAPS / region / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"]).set_index("cell_id")
    r1 = F6.cell_values(key, cells, pred.r1)
    aoa = np.nan_to_num(F6.cell_values(key, cells, mean.aoa_in.astype(float)), nan=0.0)
    base = F6.Base(key, cells, lut, ky0, kx0, step=1, tag="full"); tf = base.tf; lim = F6.region_limits(key, tf)
    geo = F8.geo_index(base, ky0, kx0, "full"); G = lambda v: F8.grid2d(cells, lut.shape, ky0, kx0, v)
    v_r1, drawn, _ = F8.nearest_px(base, r1)
    px_r1 = F8.interp_px(G(r1), geo); px_r1 = np.where(np.isfinite(px_r1), px_r1, v_r1)
    px_aoa = np.nan_to_num(F8.interp_px(G(aoa), geo), nan=0.0)
    del geo; gc.collect()
    prod = {p: F8.product_px(base, p, yr) for p in ("wei", "liu")}
    pm = {p: F8.product_masks(base, prod[p])[0] for p in ("wei", "liu")}
    fin = r1[np.isfinite(r1)]; lo, hi = MV.pct_range(fin, step=row["alt_step"]); norm = Normalize(lo, hi)
    ticks = [t for t in np.arange(10.0 * np.ceil(lo / 10.0), hi + 1e-6, 10.0)]
    allv = np.concatenate([fin] + [prod[p][pm[p]] for p in ("wei", "liu")]); ext = MV.extend_of(allv, lo, hi)
    err = error_table(row, pred.r1, tf)
    D = dict(row, cells=cells, base=base, tf=tf, lim=lim, asp=(lim[3] - lim[2]) / (lim[1] - lim[0]), px_r1=px_r1, px_aoa=px_aoa, drawn=drawn,
             prod=prod, pm=pm, lo=lo, hi=hi, norm=norm, ticks=ticks, ext=ext, err=err, n_cells_drawn=int(np.isfinite(r1).sum()),
             frac_beyond={p: dict(below=round(float((prod[p][pm[p]] < lo).mean()), 4), above=round(float((prod[p][pm[p]] > hi).mean()), 4)) for p in prod}
             | {"ours": dict(below=round(float((fin < lo).mean()), 4), above=round(float((fin > hi).mean()), 4))},
             prod_range={p: [round(float(np.nanpercentile(prod[p][pm[p]], 1)), 1), round(float(np.nanpercentile(prod[p][pm[p]], 99)), 1)] for p in prod})
    print(f"  load {region} {yr}: ALT range {lo}–{hi} ({ext}), labels {err['n_labels']} -> cells {err['n_cells']}, common {err['n_common']}, "
          f"RMSE {({k: round(v, 2) for k, v in err['rmse'].items()})}, {time.time() - t0:.0f} s")
    return D


# ================================================================ panels
def draw_map(ax, Dk, kind, M, lon_lab=(), lat_lab=(), scale=False):
    b = Dk["base"]
    if kind == "ours":
        rgb = render_px(b, Dk["px_r1"], MP.CM_ALT, Dk["norm"], Dk["drawn"], aoa=Dk["px_aoa"])
    else:
        rgb = render_px(b, Dk["prod"][kind], MP.CM_ALT, Dk["norm"], Dk["pm"][kind])
    MP.frame(ax, rgb, b.extent, Dk["lim"]); del rgb; gc.collect()
    F6.coast(ax, Dk["key"], Dk["tf"], M)
    F6.graticule(ax, Dk["key"], Dk["tf"], Dk["lim"], M, lon_lab=lon_lab, lat_lab=lat_lab)
    ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
    cm = Dk["err"]["common"]
    ax.plot(cm.x.values, cm.y.values, ls="none", marker="o", ms=OBS_PT, mfc="none", mec=C.TEXT, mew=0.35, zorder=6,
            path_effects=[pe.withStroke(linewidth=0.35 + 0.5, foreground="white")])
    if scale:
        MP.scalebar(ax, Dk["lim"], Dk["scale_km"], M)


def swarm_x(y_mm, d_mm):
    """Beeswarm offsets (mm) for values y_mm (mm): each point takes the position closest to the column centre that keeps a distance >= d_mm to the
    points already placed (points placed in increasing y)."""
    order = np.argsort(y_mm, kind="stable"); xs = np.zeros(len(y_mm)); placed = []
    for i in order:
        yi = y_mm[i]; bad = []
        for (xj, yj) in placed:
            dy = abs(yi - yj)
            if dy < d_mm:
                h = np.sqrt(d_mm ** 2 - dy ** 2); bad.append((xj - h, xj + h))
        cand = [0.0] + [b for iv in bad for b in iv]
        cand = sorted(cand, key=lambda v: (abs(v), v))
        for x in cand:
            if all(not (a + 1e-9 < x < b - 1e-9) for a, b in bad):
                xs[i] = x; break
        placed.append((xs[i], yi))
    return xs


def draw_err(fig, ax, Dk, w_mm, h_mm, show_ylabel=True):
    C.clean_axes(ax)
    lo, hi = Dk["err_lim"]; ax.set_ylim(lo, hi); ax.set_xlim(-0.55, 2.55)
    ax.axhline(0.0, color=C.ZERO, lw=C.LW_REF, zorder=1)
    cm = Dk["err"]["common"]; printed = {}
    d_mm = DOT_PT * C.PT_MM + 0.06; mm_per_unit_x = w_mm / 3.1; mm_per_cm = h_mm / (hi - lo)
    any_out = False
    for j, (m, name, col) in enumerate(SERIES):
        e = cm[f"e_{m}"].values; inside = (e >= lo) & (e <= hi)
        xs = swarm_x(e[inside] * mm_per_cm, d_mm) / mm_per_unit_x
        ax.scatter(j + xs, e[inside], s=DOT_PT ** 2, facecolor=col, edgecolor="white", lw=0.25, zorder=3, clip_on=False)
        out = np.sort(e[~inside])
        if len(out):                                         # beyond the axis: open markers on the axis end, values printed just above them
            any_out = True
            yy = lo if out[0] < lo else hi
            xo = j + (np.arange(len(out)) - (len(out) - 1) / 2) * (d_mm * 1.25 / mm_per_unit_x)
            ax.scatter(xo, np.full(len(out), yy), s=DOT_PT ** 2, facecolor="white", edgecolor=col, lw=0.6, zorder=4, clip_on=False)
            txt = ", ".join(f"{v:.0f}".replace("-", "−") for v in out)
            ax.annotate(txt, (j, yy), xytext=(0, (DOT_PT / 2 + 1.0) * (1 if yy == lo else -1)), textcoords="offset points", ha="center",
                        va="bottom" if yy == lo else "top", fontsize=C.FS_SMALL, color=OUT_GREY, annotation_clip=False)
            printed[m] = [round(float(v), 1) for v in out]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([s[1] for s in SERIES], fontsize=C.FS_SMALL, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(Dk["err_step"])); ax.yaxis.set_minor_locator(mticker.MultipleLocator(Dk["err_step"] / 2))
    ax.tick_params(axis="y", which="minor", length=1.0, width=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−")))
    if show_ylabel:
        ax.set_ylabel("Predicted − observed (cm)")
    ax.set_xlabel(f"Observed cells (n = {Dk['err']['n_common']})", labelpad=1.5)
    return printed


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    D = {r["key"]: load_row(r) for r in ROWS}
    print(f"  data {time.time() - t0:.0f} s")
    A, L = D["alaska"], D["lena"]
    # ---------------- layout (mm)
    W = 180.0
    xm0, gm = 11.0, 1.6                       # first map left edge, gap between maps
    we, ml_e, mr = 38.0, 11.2, 0.6            # error panel width, its left margin (ticks + y label), right margin
    wm = (W - xm0 - 2 * gm - ml_e - we - mr) / 3.0
    xe = W - mr - we
    y_head = 0.5; band = 13.2; gap_row = 4.6
    y1 = 4.6; h1 = wm * A["asp"]
    y2 = y1 + h1 + band + gap_row; h2 = wm * L["asp"]
    H = y2 + h2 + band + 4.6
    fig = C.figure(W, H)
    out = {}
    for Dk, y0, hrow in ((A, y1, h1), (L, y2, h2)):
        lt = Dk["letters"]; name = C.REGION[Dk["region"]][0]
        for j, kind in enumerate(("ours", "wei", "liu")):
            xm = xm0 + j * (wm + gm)
            ax = C.ax_mm(fig, xm, y0, wm, hrow)
            draw_map(ax, Dk, kind, M, lon_lab=Dk["lon_lab"] if j == 0 else (), lat_lab=Dk["lat_lab"] if j == 0 else (), scale=(j == 0))
            C.letter(fig, xm, y0 - 4.1, lt[j])
            C.text_mm(fig, xm + 3.3, y0 - 3.95, MAP_HEAD[kind], fontsize=C.FS, ha="left", va="top")
            gc.collect()
        # row label (rotated, far left)
        C.text_mm(fig, 2.0, y0 + hrow / 2, f"{name}, {Dk['year']}", rotation=90, ha="center", va="center", fontsize=C.FS)
        # colour bar under the Wei and Liu maps
        xc0 = xm0 + wm + gm; wcb = 2 * wm + gm
        cax = C.ax_mm(fig, xc0 + 0.18 * wcb, y0 + hrow + 2.6, 0.64 * wcb, 1.7)
        MP.colorbar(fig, cax, MP.CM_ALT, Dk["norm"], f"ALT {Dk['year']} (cm)", Dk["ticks"], extend=Dk["ext"])
        # error panel
        axe = C.ax_mm(fig, xe, y0, we, hrow)
        printed = draw_err(fig, axe, Dk, we, hrow)
        C.letter(fig, xe - ml_e + 0.2, y0 - 4.1, lt[3])
        rm = Dk["err"]["rmse"]
        C.text_mm(fig, xe - 1.2, y0 - 2.6, "RMSE", fontsize=C.FS_SMALL, ha="right", va="center")
        for j, (m, _, _) in enumerate(SERIES):
            xx = xe + (j + 0.55) / 3.1 * we
            C.text_mm(fig, xx, y0 - 2.6, f"{rm[m]:.1f} cm", fontsize=C.FS_SMALL, ha="center", va="center")
        e = Dk["err"]
        out[Dk["region"]] = dict(year=Dk["year"], alt_range_cm=[Dk["lo"], Dk["hi"]], extend=Dk["ext"], colour_bar_ticks_cm=[float(t) for t in Dk["ticks"]],
                                 share_of_drawn_pixels_beyond_range=Dk["frac_beyond"], product_value_range_p1_p99_cm=Dk["prod_range"],
                                 n_map_cells_drawn=Dk["n_cells_drawn"], n_labels=e["n_labels"], n_observed_cells=e["n_cells"],
                                 n_cells_with_value=e["n_with_value"], n_common_cells=e["n_common"], oof_seeds_per_label=e["n_seeds"],
                                 rmse_common_cm=dict(physics_residual_ml_block_cv=round(e["rmse"]["oof"], 2), wei_2026=round(e["rmse"]["wei"], 2),
                                                     liu_2024=round(e["rmse"]["liu"], 2),
                                                     our_map_in_sample_not_shown=round(e["rmse"]["mapv"], 2)),
                                 bias_common_cm=dict(physics_residual_ml_block_cv=round(e["bias"]["oof"], 2), wei_2026=round(e["bias"]["wei"], 2),
                                                     liu_2024=round(e["bias"]["liu"], 2)),
                                 error_axis_cm=list(Dk["err_lim"]), values_beyond_error_axis_cm=printed,
                                 alternative_common_cells=e["alt_common"], label_level_rmse_not_shown=e["label_level"],
                                 product_training_sites=e["training_sites"],
                                 cells=[dict(cell_id=str(i), lat=round(float(r.lat_c), 4), lon=round(float(r.lon_c), 4), n_labels=int(r.n_lab),
                                             observed_cm=round(float(r.obs), 2), physics_residual_ml_block_cv_cm=round(float(r.oof), 2),
                                             wei_2026_cm=round(float(r.wei), 2), liu_2024_cm=round(float(r.liu), 2), our_map_cm=round(float(r["mapv"]), 2),
                                             dist_wei_training_site_km=(None if not np.isfinite(r.dist_wei_km) else round(float(r.dist_wei_km), 2)))
                                        for i, r in e["common"].iterrows()])
    # ---------------- key (bottom, one line)
    hs = [Line2D([], [], ls="none", marker="o", ms=OBS_PT + 0.4, mfc="none", mec=C.TEXT, mew=0.4),
          MP.FadedRamp(MP.CM_ALT), Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3),
          Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3)]
    labs = ["Observed cell", "Outside area of applicability (lighter colours)", "Not mapped or no value", "Water"]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(xm0 / W, 1 - (H - 3.9) / H), ncol=4, fontsize=C.FS_SMALL, handlelength=2.2, handleheight=0.9,
               handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0, handler_map={MP.FadedRamp: MP.RampHandler()})
    rep = C.qa_figure(fig, text_ink=False, allow_colors=(OUT_GREY,))
    C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN)
    plt.close(fig)
    src = dict(figure="Fig. 6", script="scripts/4_visualization/paper_v11/fig6_product_maps.py",
               inputs=dict(our_map="data/processed/v5/maps/Alaska/pred_2014.csv.gz and data/processed/v5/maps/Lena/pred_2016.csv.gz (column r1); "
                                   "mean_2011_2024.csv.gz (aoa_in)",
                           products="data/raw/wei2026_alt_v2/extracted/ALT_{2014,2016}.tif, data/raw/liu2024_alt/ALT_{2014,2016}.tif",
                           labels="data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)",
                           product_values_at_labels="data/processed/v5/products_cell_year_v5.csv (product wei / liu, year of the row)",
                           our_errors="data/processed/v5/runs/core_v6/preds/" + OOF.format(region="<Region>") + " (spatial-block CV, 3 seeds averaged)"),
               years_used=dict(alaska=2014, lena_delta=2016),
               definition=dict(maps="a-c and e-g share one colour range per row (1st-99th percentile of our map in 5 cm (Alaska) or 10 cm (Lena Delta) "
                                    "steps); product rasters bilinearly reprojected to the 250 m display grid; our map interpolated from 1 km cell centres "
                                    "for display; cells outside the area of applicability blended 40 % towards the land grey",
                               errors="labels of the year aggregated to 1 km map cells (cell means); our series = block-CV out-of-fold predictions "
                                      "(seeds averaged per label, then cell mean), not the in-sample map; product series = product value of the year at "
                                      "the label sites (cell mean); common cells = cells where our map, Wei 2026 and Liu 2024 all have a value; "
                                      "RMSE over the common cells (cell-level, equal weight per cell)",
                               dots="beeswarm per series (x offsets only for legibility); values beyond the axis drawn as open markers on the lower spine with the values printed just above them (6 pt grey)"),
               values=out, numbers_printed={r: {"RMSE (cm)": [f"{v['rmse_common_cm']['physics_residual_ml_block_cv']:.1f}",
                                                             f"{v['rmse_common_cm']['wei_2026']:.1f}", f"{v['rmse_common_cm']['liu_2024']:.1f}"],
                                                "n": v["n_common_cells"], "beyond_axis": v["values_beyond_error_axis_cm"]} for r, v in out.items()},
               outputs=C.file_info(paths), canvas_mm=[W, round(H, 1)], pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep,
               layout_mm=dict(map_width=round(wm, 2), map_heights=[round(h1, 2), round(h2, 2)], error_panel=[round(xe, 2), we]),
               style="cmcrameri oslo_r section %s for ALT; series colours: Physics + residual ML %s, Wei 2026 %s, Liu 2024 %s" %
                     (MP.ALT_LUT, C.METHOD["R1"], C.PRODUCT["wei"][1], C.PRODUCT["liu"][1]))
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
