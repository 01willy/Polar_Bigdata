"""Supplementary Fig. S8 (v11 numbering; S7 in v10): residual ML correction maps, v11 style.
Copied from scripts/4_visualization/paper_v10/si_s06_s07_product_maps.py (the S7 part only; the S6 part became the main-text Fig. 6 of v11,
scripts/4_visualization/paper_v11/fig6_product_maps.py) and edited (light loader without product rasters, v11 map palette, output v11_si).

Residual ML correction (Physics + residual ML - Stefan (recalibrated)) per 1 km cell:
  a  Alaska, mean 2011-2024 (data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz, r1 - p1)
  b  Lena Delta, 2016 (data/processed/v5/maps/Lena/pred_2016.csv.gz, r1 - p1)
Symmetric common range (larger of the two regions' 99th percentile of |correction|, rounded up to 5 cm), cmcrameri vik, values beyond drawn at the
end colours; 250 m display interpolation of the 1 km cell values; cells outside the area of applicability lighter; no hillshade (diverging map).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper11 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v11/si_s08_correction_maps.py
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
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import v8_f8_product_maps as F8  # noqa: E402

F6 = MP.F6
MV = F6.MV
STEM = "FigS8_correction_maps"
PROC = C.ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"
ROWS = [dict(key="alaska", region="Alaska", src="mean_2011_2024", label="Alaska, mean 2011–2024", lon_lab=(-165, -155, -145), lat_lab=(60, 65, 70),
             scale_km=200),
        dict(key="lena", region="Lena", src="pred_2016", label="Lena Delta, 2016", lon_lab=(124, 126, 128, 130), lat_lab=(72, 73), scale_km=50)]


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
    t = pd.read_csv(MAPS / region / f"{row['src']}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1", "p1"]).set_index("cell_id")
    mean = pd.read_csv(MAPS / region / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"]).set_index("cell_id")
    corr = F6.cell_values(key, cells, t.r1 - t.p1)
    aoa = np.nan_to_num(F6.cell_values(key, cells, mean.aoa_in.astype(float)), nan=0.0)
    base = F6.Base(key, cells, lut, ky0, kx0, step=1, tag="full"); tf = base.tf; lim = F6.region_limits(key, tf)
    geo = F8.geo_index(base, ky0, kx0, "full"); G = lambda v: F8.grid2d(cells, lut.shape, ky0, kx0, v)
    v_c, drawn, _ = F8.nearest_px(base, corr)
    px = F8.interp_px(G(corr), geo); px = np.where(np.isfinite(px), px, v_c)
    px_aoa = np.nan_to_num(F8.interp_px(G(aoa), geo), nan=0.0)
    del geo; gc.collect()
    fin = corr[np.isfinite(corr)]
    D = dict(row, base=base, tf=tf, lim=lim, asp=(lim[3] - lim[2]) / (lim[1] - lim[0]), px=px, px_aoa=px_aoa, drawn=drawn, sym=MV.sym_max(fin),
             stats=dict(n_cells=int(len(fin)), mean_cm=round(float(fin.mean()), 3), median_cm=round(float(np.median(fin)), 3),
                        p01_cm=round(float(np.percentile(fin, 1)), 2), p99_cm=round(float(np.percentile(fin, 99)), 2),
                        abs_p99_cm=round(float(np.percentile(np.abs(fin), 99)), 2), share_negative=round(float((fin < 0).mean()), 4)), fin=fin)
    print(f"  load {region} {row['src']}: cells {len(fin)}, |corr| p99 {D['stats']['abs_p99_cm']}, {time.time() - t0:.0f} s")
    return D


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    A, L = [load_row(r) for r in ROWS]
    v = max(A["sym"], L["sym"]); norm = Normalize(-v, v); ticks = list(np.arange(-v, v + 1e-6, 5.0 if v <= 20 else 10.0))
    ext = MV.extend_of(np.concatenate([A["fin"], L["fin"]]), -v, v)
    W = 180.0; xa = 8.0; g = 9.0; cbw = 16.0
    h = (W - xa - g - cbw - 1.0) / (1.0 / A["asp"] + 1.0 / L["asp"]); wA, wL = h / A["asp"], h / L["asp"]; xL = xa + wA + g
    H = 4.5 + h + 6.5 + 4.6
    fig = C.figure(W, H)
    for Dk, x0, wk, lt, lx in ((A, xa, wA, "a", 0.5), (L, xL, wL, "b", xL - 7.5)):
        ax = C.ax_mm(fig, x0, 4.5, wk, h)
        rgb = render_px(Dk["base"], Dk["px"], MP.CM_ANOM, norm, Dk["drawn"], aoa=Dk["px_aoa"])
        MP.frame(ax, rgb, Dk["base"].extent, Dk["lim"]); del rgb; gc.collect()
        F6.coast(ax, Dk["key"], Dk["tf"], M); F6.graticule(ax, Dk["key"], Dk["tf"], Dk["lim"], M, lon_lab=Dk["lon_lab"], lat_lab=Dk["lat_lab"])
        ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
        MP.scalebar(ax, Dk["lim"], Dk["scale_km"], M); MP.corner_label(ax, Dk["label"], M)
        C.letter(fig, lx, 0.6, lt)
    cax = C.ax_mm(fig, xL + wL + 3.0, 4.5 + (h - 40.0) / 2.0, 1.9, 40.0)
    MP.colorbar(fig, cax, MP.CM_ANOM, norm, "Residual ML correction (cm)", ticks, extend=ext, orientation="vertical")
    fig.legend([MP.FadedRamp(MP.CM_ANOM, lo=0.0, hi=1.0), Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3),
                Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3)],
               ["Outside area of applicability (lighter colours)", "Not mapped", "Water"], loc="upper left",
               bbox_to_anchor=(xa / W, 1 - (4.5 + h + 6.6) / H), ncol=3, fontsize=C.FS_SMALL, handlelength=2.2,
               handleheight=0.9, columnspacing=1.6, handler_map={MP.FadedRamp: MP.RampHandler()})
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S8", dict(
        title="Residual ML correction maps", stem=STEM, script="scripts/4_visualization/paper_v11/si_s08_correction_maps.py",
        v10_number="S7 (S7 part of scripts/4_visualization/paper_v10/si_s06_s07_product_maps.py)",
        inputs=["data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz (r1 - p1)", "data/processed/v5/maps/Lena/pred_2016.csv.gz (r1 - p1)",
                "data/processed/v5/maps/<Region>/mean_2011_2024.csv.gz (aoa_in)"],
        years=dict(alaska="mean of the yearly maps 2011-2024", lena_delta=2016),
        definition="Physics + residual ML minus Stefan (recalibrated) per 1 km cell; common symmetric range = larger of the two regions' 99th percentile of "
                   "|correction| rounded up to 5 cm; cells outside the area of applicability lighter; 250 m display interpolation; no hillshade",
        values=dict(range_cm=[-v, v], extend=ext, colour_bar_ticks_cm=[float(t) for t in ticks],
                    corr_sym_cm={"Alaska": A["sym"], "Lena": L["sym"]}, stats={"Alaska": A["stats"], "Lena Delta": L["stats"]}),
        numbers_printed="scale bars (200 km, 50 km), colour bar ticks",
        legend_draft="Fig. S8 | Residual ML correction. Physics + residual ML minus Stefan (recalibrated) per 1 km cell: a, Alaska, mean of the yearly "
                     "maps 2011–2024; b, Lena Delta, 2016. Negative values: the residual learner lowers the Stefan estimate.",
        outputs=C.file_info(paths), size_mm=[W, round(H, 1)], pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
