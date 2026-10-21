"""Supplementary Fig. S9 (v13.2, 2026-10-20): residual ML correction maps of the three study regions, mean 2011-2024.
Rebuilt from the v13.1 S9 (Alaska mean 2011-2024, Lena Delta 2016) so that the three regions are treated alike (same quantity, same period).

Residual ML correction (Physics + residual ML - Stefan (recalibrated)) per 1 km cell, mean of the yearly maps 2011-2024
(data/processed/v5/maps/<Region>/mean_2011_2024.csv.gz, r1 - p1): a Alaska, b Lena Delta, c Canada (NW).
Symmetric common range (largest of the regions' 99th percentile of |correction|, rounded up to 5 cm), cmcrameri vik, values beyond drawn at the end
colours; display interpolation of the 1 km cell values (250 m Lena Delta, 500 m Alaska and Canada); cells outside the area of applicability lighter;
no hillshade (diverging map).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 \\
       python3 scripts/4_visualization/paper_v15/si_s09_correction_maps.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import c13_maps as MP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import v8_f8_product_maps as F8  # noqa: E402

F6 = MP.F6
MV = F6.MV
STEM = "FigS9_correction_maps"
PROC = C.ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"
SRC = "mean_2011_2024"
ROWS = [dict(key="alaska", region="Alaska", lon_lab=(-160, -150), lat_lab=(60, 65, 70), scale_km=200, step=2),
        dict(key="lena", region="Lena", lon_lab=(124, 128), lat_lab=(72, 73), scale_km=50, step=1),
        dict(key="canada", region="Canada", lon_lab=(-135, -125, -115), lat_lab=(60, 65), scale_km=200, step=2)]
_sel = __import__("os").environ.get("FIGS9_REGIONS")
if _sel:                                                   # test renders before all regions exist
    ROWS = [r for r in ROWS if r["key"] in _sel.split(",")]


def render_px(base, vpx, cmap, norm, drawn, aoa=None):
    """Pixel values -> RGB: LAND outside `drawn` on land, WATER on water, colours (faded outside the area of applicability) on drawn pixels."""
    water = base.water_cell | (~base.land & ~drawn)
    rgb = np.empty(base.idx.shape + (3,), np.float32); rgb[:] = MP.LAND; rgb[water] = MP.WATER
    v = np.where(drawn & np.isfinite(vpx), vpx, 0.0)
    col = np.empty(base.idx.shape + (3,), np.float32)
    for r0 in range(0, v.shape[0], 512):
        col[r0:r0 + 512] = cmap(norm(v[r0:r0 + 512]))[..., :3]
    if aoa is not None:
        f = (MP.FADE * (1.0 - np.clip(aoa, 0.0, 1.0)))[..., None]; col = col * (1.0 - f) + f * MP.LAND
    rgb[drawn] = col[drawn]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def load_row(row):
    t0 = time.time(); key, region = row["key"], row["region"]
    cells, lut, ky0, kx0 = F6.cell_table(key)
    t = pd.read_csv(MAPS / region / f"{SRC}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1", "p1", "aoa_in"]).set_index("cell_id")
    corr = F6.cell_values(key, cells, t.r1 - t.p1)
    aoa = np.nan_to_num(F6.cell_values(key, cells, t.aoa_in.astype(float)), nan=0.0)
    base = F6.Base(key, cells, lut, ky0, kx0, step=row["step"], tag="full"); tf = base.tf; lim = F6.region_limits(key, tf)
    base.lat = base.lon = None
    geo = F8.geo_index(base, ky0, kx0, "full"); G = lambda v: F8.grid2d(cells, lut.shape, ky0, kx0, v)
    v_c, drawn, _ = F8.nearest_px(base, corr)
    px = F8.interp_px(G(corr), geo); px = np.where(np.isfinite(px), px, v_c)
    px_aoa = np.nan_to_num(F8.interp_px(G(aoa), geo), nan=0.0)
    del geo, v_c; gc.collect()
    fin = corr[np.isfinite(corr)]
    D = dict(row, base=base, cells=cells, tf=tf, lim=lim, asp=(lim[3] - lim[2]) / (lim[1] - lim[0]), px=px, px_aoa=px_aoa, drawn=drawn, sym=MV.sym_max(fin),
             stats=dict(n_cells=int(len(fin)), mean_cm=round(float(fin.mean()), 3), median_cm=round(float(np.median(fin)), 3),
                        p01_cm=round(float(np.percentile(fin, 1)), 2), p99_cm=round(float(np.percentile(fin, 99)), 2),
                        abs_p99_cm=round(float(np.percentile(np.abs(fin), 99)), 2), share_negative=round(float((fin < 0).mean()), 4)), fin=fin)
    print(f"  load {region} {SRC}: cells {len(fin)}, |corr| p99 {D['stats']['abs_p99_cm']}, {time.time() - t0:.0f} s")
    return D


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    D = [load_row(r) for r in ROWS]
    v = max(d["sym"] for d in D); norm = Normalize(-v, v); ticks = list(np.arange(-v, v + 1e-6, 5.0 if v <= 20 else 10.0))
    ext = MV.extend_of(np.concatenate([d["fin"] for d in D]), -v, v)
    W = 180.0; xa = 9.0; g = 9.0; cbw = 16.0; y1 = 4.5
    h = (W - xa - (len(D) - 1) * g - cbw - 1.0) / sum(1.0 / d["asp"] for d in D)
    xs = []; x = xa
    for d in D:
        xs.append(x); x += h / d["asp"] + g
    H = y1 + h + 6.5 + 4.6
    fig = C.figure(W, H)
    for i, (Dk, x0) in enumerate(zip(D, xs)):
        wk = h / Dk["asp"]; ax = C.ax_mm(fig, x0, y1, wk, h)
        rgb = render_px(Dk["base"], Dk["px"], MP.CM_ANOM, norm, Dk["drawn"], aoa=Dk["px_aoa"])
        MP.frame(ax, rgb, Dk["base"].extent, Dk["lim"]); del rgb; gc.collect()
        MP.coast(ax, Dk["key"], Dk["tf"], M, base=Dk["base"], cells=Dk["cells"], tol_m=600.0); MP.graticule(ax, Dk["key"], Dk["tf"], Dk["lim"], M, lon_lab=Dk["lon_lab"], lat_lab=Dk["lat_lab"])
        ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
        MP.scalebar(ax, Dk["lim"], Dk["scale_km"], M); MP.corner_label(ax, C.REGION[Dk["region"]][0], M)
        C.letter(fig, x0 - 8.0 if i else 0.5, 0.6, "abc"[i])
        Dk["base"] = None; gc.collect()
    cax = C.ax_mm(fig, x - g + 3.0, y1 + (h - 36.0) / 2.0, 1.9, 36.0)
    MP.colorbar(fig, cax, MP.CM_ANOM, norm, "Residual ML correction (cm)", ticks, extend=ext, orientation="vertical")
    fig.legend([MP.FadedRamp(MP.CM_ANOM, lo=0.0, hi=1.0), Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3),
                Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3)],
               ["Outside area of applicability (lighter colours)", "Not mapped", "Water"], loc="upper left",
               bbox_to_anchor=(xa / W, 1 - (y1 + h + 6.6) / H), ncol=3, fontsize=C.FS_SMALL, handlelength=2.2,
               handleheight=0.9, columnspacing=1.6, handler_map={MP.FadedRamp: MP.RampHandler()}, borderaxespad=0, borderpad=0)
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    names = [C.REGION[d["region"]][0] for d in D]
    C.update_si_json("S9", dict(
        title="Residual ML correction maps, mean 2011-2024, three regions", stem=STEM, script="scripts/4_visualization/paper_v15/si_s09_correction_maps.py",
        inputs=[f"data/processed/v5/maps/{{Alaska,Lena,Canada}}/{SRC}.csv.gz (r1 - p1, aoa_in)"],
        years="mean of the yearly maps 2011-2024 (all regions)",
        definition="Physics + residual ML minus Stefan (recalibrated) per 1 km cell; common symmetric range = largest of the regions' 99th percentile of "
                   "|correction| rounded up to 5 cm; cells outside the area of applicability lighter; display interpolation (250 m Lena Delta, 500 m Alaska and Canada); no hillshade",
        values=dict(range_cm=[-v, v], extend=ext, colour_bar_ticks_cm=[float(t) for t in ticks],
                    corr_sym_cm={n: d["sym"] for n, d in zip(names, D)}, stats={n: d["stats"] for n, d in zip(names, D)}),
        numbers_printed="scale bars (200, 50, 200 km), colour bar ticks",
        legend_draft="Fig. S9 | Residual ML correction. Physics + residual ML minus Stefan (recalibrated) per 1 km cell, mean of the yearly maps "
                     "2011–2024: a, Alaska; b, Lena Delta; c, Canada (NW). Negative values: the residual learner lowers the Stefan estimate.",
        outputs=C.file_info(paths), size_mm=[W, round(H, 1)], pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
