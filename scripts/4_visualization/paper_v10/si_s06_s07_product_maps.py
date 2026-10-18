"""Supplementary Figs S6 and S7 (content of v8_f8_product_maps.py rows 1-2 and deck R13), paper palette. One run makes both figures.

S6  Product maps against our map, same year: Alaska 2014 (a our map, b Wei 2026, c Liu 2024, d residuals at observed cells) and Lena Delta 2016 (e-h).
    a, e: Physics + residual ML 1 km map (full-data model), 250 m display interpolation, cells outside the area of applicability lighter.
    b, c, f, g: Wei 2026 (EPSG:3995, 1 km) and Liu 2024 (1/120 deg) ALT of the same year, bilinear reprojection to the same 250 m grid.
    Same colour range per row (1st-99th percentile of our map, 5 or 10 cm steps). d, h: observed cells of that year grouped into site clusters; colour =
    mean (map value - observed) of the cluster, area ~ number of cells with a value, x = no value; RMSE on the cells where all three maps have a value.
S7  Residual ML correction (Physics + residual ML - Stefan (recalibrated)): a Alaska mean 2011-2024 (mean_2011_2024 r1 - p1), b Lena Delta 2016
    (r1 - p1 of 2016); symmetric common range (larger 99th percentile of |correction|, 5 cm steps), cmcrameri vik.
Inputs: data/processed/v5/maps/<Region>/{pred_<year>,mean_2011_2024}.csv.gz, data/raw/wei2026_alt_v2/extracted/ALT_<year>.tif,
        data/raw/liu2024_alt/ALT_<year>.tif, data/processed/v5/{labels_cell_year_v5,products_cell_year_v5}.csv, runs/core_v6/preds (OOF, JSON only);
        reprojection caches in outputs/figures/paper/v7/_cache (read through outputs/figures/paper/v10/_cache).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/si_s06_s07_product_maps.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import maps10 as MP  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import v8_f8_product_maps as F8  # noqa: E402

F6 = MP.F6
F8.MV.CM_DIFF = MP.CM_ANOM                       # residual bubbles / correction maps use vik (paper palette)
W_SHADE = 0.30
EDGE = "#4D4D4D"


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


def draw_map(ax, Dk, kind, D, M, lon_lab=(), lat_lab=(), scale=False, label=None):
    b = Dk["base"]
    if kind == "ours":
        rgb = render_px(b, Dk["px"]["r1"], MP.CM_ALT, Dk["norm"], Dk["drawn"], aoa=Dk["px"]["aoa"])
    elif kind == "corr":
        rgb = render_px(b, Dk["px"]["corr"], MP.CM_ANOM, D["normC"], Dk["drawn_c"], aoa=Dk["px"]["aoa"], w_shade=0.0)   # diverging: no hillshade (as Fig. 6b)
    else:
        dr, _ = Dk["pm"][kind]; rgb = render_px(b, Dk["prod"][kind], MP.CM_ALT, Dk["norm"], dr)
    MP.frame(ax, rgb, b.extent, Dk["lim"]); del rgb
    F6.coast(ax, Dk["key"], Dk["tf"], M); F6.graticule(ax, Dk["key"], Dk["tf"], Dk["lim"], M, lon_lab=lon_lab, lat_lab=lat_lab)
    if scale:
        F6.scalebar(ax, Dk["lim"], Dk["scale_km"], M)
    if label:
        MP.corner_label(ax, label, M)


def draw_mini(ax, Dk, m, name, D, M):
    bm = Dk["base_mini"]; MP.frame(ax, Dk["rgb_mini"], bm.extent, Dk["lim"]); F6.coast(ax, Dk["key"], Dk["tf"], M)
    g = Dk["res"]["clusters"]
    for _, r in g.sort_values("n", ascending=False).iterrows():
        n = int(r[f"n_{m}"]); val = r[f"d_{m}"]
        if n > 0 and np.isfinite(val):
            ax.scatter(r.x, r.y, s=(1.6 * np.sqrt(n)) ** 2, facecolor=MP.CM_ANOM(D["normR"](val)), edgecolor=EDGE, lw=0.3, zorder=6)
        else:
            ax.scatter(r.x, r.y, s=(1.7 * 1.6) ** 2, marker="x", c=EDGE, lw=0.45, zorder=6)
    ax.text(0.5, 1.03, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT, linespacing=1.0)
    ax.text(0.5, -0.04, f"RMSE {Dk['res']['rmse'][m]:.1f} cm", transform=ax.transAxes, ha="center", va="top", fontsize=C.FS_SMALL, color=C.TEXT)


def hcbar(fig, rect_mm, cmap, norm, label, ticks, extend):
    cax = C.ax_mm(fig, *rect_mm); MP.colorbar(fig, cax, cmap, norm, label, list(ticks), extend=extend); return cax


def fig_s6(D, M):
    A, L = D["alaska"], D["lena"]
    W = 180.0; xa, mw, gm = 8.0, 33.5, 2.2; xr = xa + 3 * mw + 2 * gm + 6.0; wr = W - xr - 1.0; gmini = 1.6; mini_w = (wr - 2 * gmini) / 3.0
    blk = 19.0; y1 = 4.5; y2 = y1 + mw * A["asp"] + blk; H = y2 + mw * L["asp"] + blk + 2.0
    fig = C.figure(W, H); out = {}
    for Dk, y0, lt in ((A, y1, "abcd"), (L, y2, "efgh")):
        mh = mw * Dk["asp"]; name = C.REGION[Dk["region"]][0]
        labels = {"ours": f"{name}, {Dk['year']}\nPhysics + residual ML", "wei": "Wei 2026", "liu": "Liu 2024"}
        for j, kind in enumerate(("ours", "wei", "liu")):
            ax = C.ax_mm(fig, xa + j * (mw + gm), y0, mw, mh)
            draw_map(ax, Dk, kind, D, M, lon_lab=Dk["lon_lab"], lat_lab=Dk["lat_lab"] if j == 0 else (), scale=(j == 0), label=labels[kind])
            C.letter(fig, xa + j * (mw + gm) - (6.6 if j == 0 else 0.0), y0 - 3.9, lt[j]) if j == 0 else C.letter(fig, xa + j * (mw + gm), y0 - 3.9, lt[j])
        hcbar(fig, (xa + mw + gm + (2 * mw + gm) / 2 - 25.0, y0 + mh + 5.5, 50.0, 2.0), MP.CM_ALT, Dk["norm"], "ALT (cm)", Dk["ticks"], Dk["ext"])
        hmini = mini_w * Dk["asp"]; yi = y0 + 4.0
        for j, (m, nm) in enumerate((("ours", "Physics +\nresidual ML"), ("wei", "Wei 2026"), ("liu", "Liu 2024"))):
            ax = C.ax_mm(fig, xr + j * (mini_w + gmini), yi, mini_w, hmini); draw_mini(ax, Dk, m, nm, D, M)
        C.letter(fig, xr - 4.0, y0 - 3.9, lt[3])
        ycb = yi + hmini + 6.0
        hcbar(fig, (xr + 6.0, ycb, wr - 12.0, 2.0), MP.CM_ANOM, D["normR"], "Map value − observed (cm)", D["ticksR"], "both")
        # size key + note
        kax = C.ax_mm(fig, xr, ycb + 8.6, wr, 3.0); kax.axis("off"); kax.set_xlim(0, wr); kax.set_ylim(0, 3.0)
        cx = 0.5
        for n in (1, 20):
            d_pt = 1.6 * np.sqrt(n); d_mm = d_pt * C.PT_MM
            kax.scatter(cx + d_mm / 2, 1.5, s=d_pt ** 2, facecolor="white", edgecolor=EDGE, lw=0.3, clip_on=False)
            t = kax.text(cx + d_mm + 0.8, 1.5, f"{n} cell" + ("s" if n > 1 else ""), va="center", ha="left", fontsize=C.FS_SMALL, color=C.TEXT)
            cx += d_mm + 0.8 + (2.2 if n == 1 else 3.6) * 2.0 + 2.0
        kax.scatter(cx + 1.0, 1.5, s=(1.7 * 1.6) ** 2, marker="x", c=EDGE, lw=0.45, clip_on=False)
        kax.text(cx + 2.4, 1.5, "no value", va="center", ha="left", fontsize=C.FS_SMALL, color=C.TEXT)
        kax.text(0.5, -2.2, f"RMSE on {Dk['res']['n_common']} common cells", va="center", ha="left", fontsize=C.FS_SMALL, color=C.TEXT)
        out[Dk["region"]] = dict(year=Dk["year"], alt_range_cm=[Dk["lo"], Dk["hi"]], extend=Dk["ext"], n_cells_drawn=Dk["n_cells_drawn"],
                                 n_labels=Dk["res"]["n_labels"], n_label_cells=int(len(Dk["res"]["cells"])), n_clusters=int(len(Dk["res"]["clusters"])),
                                 n_common_cells=Dk["res"]["n_common"], rmse_common_cm={k: round(v, 2) for k, v in Dk["res"]["rmse"].items()},
                                 rmse_own_cells={k: dict(rmse=round(v["rmse"], 2), n=v["n"]) for k, v in Dk["res"]["rmse_own"].items()},
                                 product_value_range_p1_p99=Dk["prod_range"])
    # shared legend (bottom left, under e)
    mid = np.array(MP.CM_ALT(0.55)[:3], np.float32); faded = mid * (1 - MP.FADE) + MP.FADE * MP.LAND
    fig.legend([Patch(facecolor=tuple(faded.tolist()), edgecolor="#9A9A9A", lw=0.3), Patch(facecolor=MP.LAND_HEX, edgecolor="#9A9A9A", lw=0.3),
                Patch(facecolor=MP.WATER_HEX, edgecolor="#9A9A9A", lw=0.3)],
               ["Outside area of applicability", "Not mapped / no value", "Water"], loc="upper left",
               bbox_to_anchor=(1.0 / W, 1 - (y2 + mw * L["asp"] + 10.4) / H), ncol=1, fontsize=C.FS_SMALL, handlelength=1.3, handleheight=0.9, labelspacing=0.3)
    return fig, W, H, out


def fig_s7(D, M):
    A, L = D["alaska"], D["lena"]
    W = 180.0; xa = 8.0; g = 9.0; cbw = 16.0
    h = (W - xa - g - cbw - 1.0) / (1.0 / A["asp"] + 1.0 / L["asp"]); wA, wL = h / A["asp"], h / L["asp"]; xL = xa + wA + g
    H = 4.5 + h + 6.5
    fig = C.figure(W, H)
    ax = C.ax_mm(fig, xa, 4.5, wA, h); draw_map(ax, A, "corr", D, M, lon_lab=(-165, -155, -145), lat_lab=A["lat_lab"], scale=True, label="Alaska, mean 2011–2024")
    C.letter(fig, 0.5, 0.6, "a")
    ax = C.ax_mm(fig, xL, 4.5, wL, h); draw_map(ax, L, "corr", D, M, lon_lab=(124, 126, 128, 130), lat_lab=L["lat_lab"], scale=True, label="Lena Delta, 2016")
    C.letter(fig, xL - 7.5, 0.6, "b")
    cax = C.ax_mm(fig, xL + wL + 3.0, 4.5 + (h - 40.0) / 2.0, 2.0, 40.0)
    cb = fig.colorbar(ScalarMappable(norm=D["normC"], cmap=MP.CM_ANOM), cax=cax, orientation="vertical", extend="both", ticks=list(D["ticksC"]), extendfrac=0.05)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=C.FS_SMALL, width=0.4, length=1.6, pad=1.2)
    cb.ax.set_yticklabels([f"{t:g}".replace("-", "−") for t in D["ticksC"]])
    cb.set_label("Residual ML correction (cm)", fontsize=C.FS_TICK, labelpad=2.0)
    return fig, W, H


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    D = F8.load_all(); print(f"  data {time.time() - t0:.0f} s")
    fig, W, H, out6 = fig_s6(D, M)
    rep = C.qa_figure(fig, text_ink=False); C.print_qa("FigS6_product_maps", rep)
    p6 = C.export_si(fig, "FigS6_product_maps"); plt.close(fig); gc.collect()
    C.update_si_json("S6", dict(
        title="Product maps against our map in the same year", stem="FigS6_product_maps", script="scripts/4_visualization/paper_v10/si_s06_s07_product_maps.py",
        inputs=["data/processed/v5/maps/Alaska/pred_2014.csv.gz", "data/processed/v5/maps/Lena/pred_2016.csv.gz", "data/processed/v5/maps/<Region>/mean_2011_2024.csv.gz (aoa_in)",
                "data/raw/wei2026_alt_v2/extracted/ALT_{2014,2016}.tif", "data/raw/liu2024_alt/ALT_{2014,2016}.tif",
                "data/processed/v5/labels_cell_year_v5.csv", "data/processed/v5/products_cell_year_v5.csv"],
        years=dict(alaska=2014, lena_delta=2016),
        definition=dict(maps="our map = Physics + residual ML trained on all labels (values at observed cells are in-sample; block-CV out-of-fold RMSE in "
                             "rmse_common_cm['oof'])", residuals="observed cells of the year aggregated to 1 km cells, clusters = single-linkage within 3 % "
                             "of the map width; RMSE on the cells where all three maps have a value"),
        values=out6, numbers_printed={r: {k: f"RMSE {v:.1f} cm" for k, v in out6[r]["rmse_common_cm"].items() if k != "oof"} for r in out6},
        outputs=C.file_info(p6), size_mm=[W, round(H, 1)], qa=rep))
    fig, W, H = fig_s7(D, M)
    rep = C.qa_figure(fig, text_ink=False); C.print_qa("FigS7_correction_maps", rep)
    p7 = C.export_si(fig, "FigS7_correction_maps"); plt.close(fig)
    C.update_si_json("S7", dict(
        title="Residual ML correction maps", stem="FigS7_correction_maps", script="scripts/4_visualization/paper_v10/si_s06_s07_product_maps.py",
        inputs=["data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz (r1 - p1)", "data/processed/v5/maps/Lena/pred_2016.csv.gz (r1 - p1)"],
        years=dict(alaska="mean of the yearly maps 2011-2024", lena_delta=2016),
        definition="Physics + residual ML minus Stefan (recalibrated) per 1 km cell; common symmetric range = larger of the two regions' 99th percentile of "
                   "|correction| rounded up to 5 cm; cells outside the area of applicability lighter",
        values=dict(range_cm=[-D["vC"], D["vC"]], corr_sym_cm={"Alaska": D["alaska"]["corr_sym"], "Lena": D["lena"]["corr_sym"]}),
        numbers_printed="scale bars (200 km, 50 km)", outputs=C.file_info(p7), size_mm=[W, round(H, 1)], qa=rep))
    for p in p6 + p7:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
