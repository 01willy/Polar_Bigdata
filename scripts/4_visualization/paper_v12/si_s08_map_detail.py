"""Supplementary Fig. S7 (v11 numbering; S9 in v10): 1 km detail, area of applicability and the Lena Delta 2016 map (content of v7_f6_maps.py deck
R08/R09), v11 style. Copied from scripts/4_visualization/paper_v10/si_s09_map_detail.py and edited (v11 map palette, dark-grey text and frames,
area-of-applicability colours taken from the ALT colormap, faded-ramp key, output v11_si).

a, b  Alaska mean ALT 2011-2024 (Physics + residual ML) in two 50 km windows (North Slope near Toolik; Interior near Fairbanks), 1 km cells drawn as
      cells (no display interpolation) on a subtle 250 m hillshade; observed cells as open circles; cells outside the area of applicability lighter.
c     Area of applicability of the Alaska map (inside / outside; static, from the 2011-2024 covariates); boxes = windows of a and b.
d     Lena Delta ALT 2016 (Physics + residual ML), 250 m display interpolation, observed cells, box = window of e.
e     Samoylov-Kurungnakh window (24 km), 1 km cells, observed cells.
Inputs: data/processed/v5/maps/{Alaska,Lena}/{mean_2011_2024,pred_2016}.csv.gz, data/processed/map_<region>/..., labels_cell_year_v5.csv (in_forcing == 1).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper12 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v12/si_s07_map_detail.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c12_common as C  # noqa: E402
import c12_maps as MP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, ListedColormap, to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

F6 = MP.F6
STEM = "FigS8_map_detail"
PROC = C.ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"
AOA_IN = to_hex(MP.CM_ALT(0.50))                                  # inside: mid ALT blue
AOA_OUT = to_hex(MP.faded(MP.CM_ALT, 0.10))                       # outside: light blue-grey (low end of the ALT colours blended 40 % towards the land grey)
W_SHADE = 0.30


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    # ---------------- Alaska data
    cA, lutA, ky0A, kx0A = F6.cell_table("alaska")
    mA = pd.read_csv(MAPS / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    vA = F6.cell_values("alaska", cA, mA.r1); aA = np.nan_to_num(F6.cell_values("alaska", cA, mA.aoa_in.astype(float)), nan=0.0)
    loA, hiA = F6.MV.pct_range(mA.r1.values); nA = Normalize(loA, hiA); eA = F6.MV.extend_of(mA.r1.values, loA, hiA)
    bA = F6.Base("alaska", cA, lutA, ky0A, kx0A, step=2, tag="full"); tfA = bA.tf; limA = F6.region_limits("alaska", tfA); aspA = (limA[3] - limA[2]) / (limA[1] - limA[0])
    olon, olat, nlocA = F6.obs_cells(lab, "Alaska"); oxA, oyA = tfA.transform(olon, olat)
    # ---------------- Lena data
    cL, lutL, ky0L, kx0L = F6.cell_table("lena")
    mL = pd.read_csv(MAPS / "Lena" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    l16 = pd.read_csv(MAPS / "Lena" / "pred_2016.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
    vL = F6.cell_values("lena", cL, l16); aL = np.nan_to_num(F6.cell_values("lena", cL, mL.aoa_in.astype(float)), nan=0.0)
    loL, hiL = F6.MV.pct_range(l16.values, step=10.0); nL = Normalize(loL, hiL); eL = F6.MV.extend_of(l16.values, loL, hiL)
    bL = F6.Base("lena", cL, lutL, ky0L, kx0L, step=1, tag="full"); tfL = bL.tf; limL = F6.region_limits("lena", tfL); aspL = (limL[3] - limL[2]) / (limL[1] - limL[0])
    olonL, olatL, nlocL = F6.obs_cells(lab, "Lena"); oxL, oyL = tfL.transform(olonL, olatL)
    print(f"  data {time.time() - t0:.0f} s")
    # ---------------- layout (mm)
    W = 180.0; g = 4.0
    w1 = (W - 1.0 - 9.0 - 2 * g) / 3.0; h1 = w1 * aspA
    y1 = 4.3; xs1 = [1.0, 1.0 + w1 + g, W - 1.0 - w1]
    y2 = y1 + h1 + 16.0; h2 = 66.0; wd = h2 / aspL; we = h2
    xd = 9.0; xe = xd + wd + 6.0
    H = y2 + h2 + 7.5
    fig = C.figure(W, H); src = {}
    zooms = [("North Slope, Toolik", (-149.55, 68.75), 25e3), ("Interior, Fairbanks", (-147.85, 64.85), 25e3)]
    out_cells = {}
    for j, (name, cen, hx) in enumerate(zooms):
        zl = F6.zoom_lim(tfA, cen, hx, hx * aspA)
        bz = F6.Base("alaska", cA, lutA, ky0A, kx0A, step=1, window=[zl[0] - 2e3, zl[1] + 2e3, zl[2] - 2e3, zl[3] + 2e3], tag=f"zoom{j}")
        rgb, _ = MP.render(bz, vA, MP.CM_ALT, nA, aoa=aA, smooth=False, w_shade=W_SHADE)
        ax = C.ax_mm(fig, xs1[j], y1, w1, h1); MP.frame(ax, rgb, bz.extent, zl)
        F6.obs_marks(ax, oxA, oyA, zl, s=M.obs_zoom_s(w1 / ((zl[1] - zl[0]) / 1e3)), lw=0.45, case=0.3, pad_frac=0.012)
        MP.scalebar(ax, zl, 10, M, fx=0.05 if j == 0 else 0.62); MP.corner_label(ax, name, M)   # b: bottom right (observed cell at bottom left)
        out_cells[name] = F6.zoom_outside(bz, vA, aA, zl)
        src.setdefault("a_b", {})[name] = dict(centre_lonlat=list(cen), width_km=2 * hx / 1e3, cells_outside_aoa_in_window=out_cells[name])
        zooms[j] = (name, cen, hx, zl)
    C.letter(fig, xs1[0], 0.6, "a"); C.letter(fig, xs1[1], 0.6, "b")
    # colour bar under a-b
    cax = C.ax_mm(fig, xs1[0] + 12.0, y1 + h1 + 4.0, 2 * w1 + g - 24.0, 1.9)
    MP.colorbar(fig, cax, MP.CM_ALT, nA, "ALT, mean 2011–2024 (cm)", list(np.arange(loA, hiA + 1e-6, 10.0)), extend=eA)
    # c: area of applicability
    ax = C.ax_mm(fig, xs1[2], y1, w1, h1)
    rgb, _ = MP.render(bA, np.where(np.isfinite(vA), aA, np.nan), ListedColormap([AOA_OUT, AOA_IN]), Normalize(-0.5, 1.5), smooth=False, w_shade=W_SHADE)
    MP.frame(ax, rgb, bA.extent, limA); del rgb
    F6.coast(ax, "alaska", tfA, M); F6.graticule(ax, "alaska", tfA, limA, M, lon_lab=(-160, -150), lat_lab=(60, 65, 70))
    ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
    for _, _, _, zl in zooms:
        MP.box(ax, zl, lw=0.6)
    MP.scalebar(ax, limA, 200, M)
    C.letter(fig, xs1[2] - 8.0, 0.6, "c")
    fig.legend([Patch(facecolor=AOA_IN, edgecolor="none"), Patch(facecolor=AOA_OUT, edgecolor="none"), Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3),
                Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3)],
               ["Inside area of applicability", "Outside", "Not mapped", "Water"], loc="upper left",
               bbox_to_anchor=(xs1[2] / W, 1 - (y1 + h1 + 4.6) / H), ncol=2, fontsize=C.FS_SMALL, handlelength=1.3, handleheight=0.9, columnspacing=1.0,
               labelspacing=0.35)
    src["c"] = dict(frac_inside_aoa=round(float(mA.aoa_in.mean()), 4), colours=dict(inside=AOA_IN, outside=AOA_OUT))
    del bA; gc.collect()
    # ---------------- d: Lena 2016
    name, cen, hz = "Samoylov–Kurungnakh", (126.4, 72.36), 12e3
    zlL = F6.zoom_lim(tfL, cen, hz, hz)
    ax = C.ax_mm(fig, xd, y2, wd, h2)
    rgb, _ = MP.render(bL, vL, MP.CM_ALT, nL, aoa=aL, smooth=True, fill_max=F6.FILL_MAX, w_shade=W_SHADE, feather=1.5, aoa_sigma=4.0)
    MP.frame(ax, rgb, bL.extent, limL); del rgb
    F6.coast(ax, "lena", tfL, M); F6.graticule(ax, "lena", tfL, limL, M, lon_lab=(124, 126, 128, 130), lat_lab=(72, 73))
    ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
    nobsL = F6.obs_marks(ax, oxL, oyL, limL, s=1.7 ** 2, lw=0.3, case=0.2)
    MP.box(ax, zlL, lw=0.6)
    MP.scalebar(ax, limL, 50, M)
    C.letter(fig, 0.3, y2 - 3.7, "d")
    # e: zoom
    bz = F6.Base("lena", cL, lutL, ky0L, kx0L, step=1, window=[zlL[0] - 1e3, zlL[1] + 1e3, zlL[2] - 1e3, zlL[3] + 1e3], tag="zoom0")
    rgb, _ = MP.render(bz, vL, MP.CM_ALT, nL, aoa=aL, smooth=False, w_shade=W_SHADE)
    ax = C.ax_mm(fig, xe, y2, we, h2); MP.frame(ax, rgb, bz.extent, zlL)
    F6.obs_marks(ax, oxL, oyL, zlL, s=M.obs_zoom_s(we / ((zlL[1] - zlL[0]) / 1e3)), lw=0.45, case=0.3, pad_frac=0.012)
    MP.scalebar(ax, zlL, 5, M); MP.corner_label(ax, name, M)
    outL = F6.zoom_outside(bz, vL, aL, zlL)
    C.letter(fig, xe, y2 - 3.7, "e")
    # Lena colour bar (vertical, right of e) and key
    xk = xe + we + 5.0
    cax = C.ax_mm(fig, xk, y2 + 4.0, 1.9, 30.0)
    MP.colorbar(fig, cax, MP.CM_ALT, nL, "ALT 2016 (cm)", list(np.arange(loL, hiL + 1e-6, 10.0)), extend=eL, orientation="vertical")
    fig.legend([Line2D([], [], ls="none", marker="o", ms=2.6, mfc="none", mec=C.TEXT, mew=0.45), MP.FadedRamp(MP.CM_ALT)],
               ["Observed cell", "Outside area of\napplicability"], loc="upper left", bbox_to_anchor=(xk / W, 1 - (y2 + 40.0) / H), ncol=1,
               fontsize=C.FS_SMALL, handlelength=1.8, handleheight=0.9, labelspacing=0.5, handler_map={MP.FadedRamp: MP.RampHandler()})
    src["d_e"] = dict(year=2016, range_cm=[loL, hiL], extend=eL, observed_cells_drawn_d=nobsL, zoom_width_km=2 * hz / 1e3, cells_outside_aoa_in_zoom=outL,
                      frac_inside_aoa=round(float(mL.aoa_in.mean()), 4))
    src["a_b_range_cm"] = [loA, hiA]
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S8", dict(
        title="1 km detail, area of applicability and the Lena Delta 2016 map", stem=STEM, script="scripts/4_visualization/paper_v12/si_s07_map_detail.py",
        v10_number="S9 (scripts/4_visualization/paper_v10/si_s09_map_detail.py)",
        inputs=["data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz", "data/processed/v5/maps/Lena/{mean_2011_2024,pred_2016}.csv.gz",
                "data/processed/map_{alaska,lena}/{*_grid_x25_v1.csv.gz, hillshade_v1/shade_250m.tif}", "data/processed/v5/labels_cell_year_v5.csv"],
        years=dict(a_b="mean of the yearly maps 2011-2024", c="static (covariate dissimilarity to the training labels, 2011-2024 mean map table)", d_e="2016"),
        values=src, numbers_printed="scale bars (10, 200, 50, 5 km)",
        legend_draft="Fig. S7 | 1 km detail and area of applicability. a, b, Alaska mean ALT 2011–2024 in two 50 km windows (North Slope near Toolik; "
                     "Interior near Fairbanks) drawn as 1 km cells. c, area of applicability of the Alaska map with the two windows. d, Lena Delta ALT "
                     "2016. e, Samoylov–Kurungnakh window (24 km).",
        outputs=C.file_info(paths), size_mm=[W, round(H, 1)], pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
