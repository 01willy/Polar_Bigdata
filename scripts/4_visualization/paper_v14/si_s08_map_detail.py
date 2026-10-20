"""Supplementary Fig. S8 (v13.2, 2026-10-20): 1 km detail and area of applicability in the three study regions.
Rebuilt from the v13.1 S8 (Alaska windows + Alaska area of applicability + Lena Delta 2016 map) so that the three regions are treated alike.

a-d  Mean ALT 2011-2024 (Physics + residual ML) in four windows drawn as 1 km cells (no display interpolation) on a subtle 250 m hillshade:
     a North Slope near Toolik (50 km), b Interior near Fairbanks (50 km), c Samoylov-Kurungnakh, Lena Delta (24 km), d Mackenzie Delta near Inuvik
     (50 km). Observed cells as open circles; cells outside the area of applicability lighter. One colour range for the four windows (as Fig. 7 a-c:
     1st-99th percentile of the pooled mean maps of the three regions, 5 cm steps).
e-g  Area of applicability (inside / outside; static, from the covariates of the 2011-2024 mean map table) of the Alaska, Lena Delta and Canada (NW)
     maps; boxes = windows of a-d.
Inputs: data/processed/v5/maps/{Alaska,Lena,Canada}/mean_2011_2024.csv.gz (r1, aoa_in), data/processed/map_<region>/ (grid, hillshade),
        data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 \\
       python3 scripts/4_visualization/paper_v14/si_s08_map_detail.py
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
from matplotlib.colors import Normalize, ListedColormap, to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

F6 = MP.F6
STEM = "FigS8_map_detail"
PROC = C.ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"
AOA_IN = to_hex(MP.CM_ALT(0.50))                                  # inside: mid ALT blue
AOA_OUT = to_hex(MP.faded(MP.CM_ALT, 0.10))                       # outside: light blue-grey (low end of the ALT colours blended 40 % towards the land grey)
W_SHADE = 0.30
REGS = [dict(key="alaska", reg="Alaska", lon_lab=(-160, -150), lat_lab=(60, 65, 70), scale_km=200, step=2),
        dict(key="lena", reg="Lena", lon_lab=(124, 128), lat_lab=(72, 73), scale_km=50, step=1),
        dict(key="canada", reg="Canada", lon_lab=(-135, -125, -115), lat_lab=(60, 65), scale_km=200, step=2)]
WIN = [dict(key="alaska", name="North Slope, Toolik", cen=(-149.55, 68.75), hx=25e3, scale_km=10, tag="zoom0"),
       dict(key="alaska", name="Interior, Fairbanks", cen=(-147.85, 64.85), hx=25e3, scale_km=10, tag="zoom1"),
       dict(key="lena", name="Samoylov–Kurungnakh", cen=(126.4, 72.36), hx=12e3, scale_km=5, tag="zoom0"),
       dict(key="canada", name="Mackenzie Delta, Inuvik", cen=(-133.5, 68.3), hx=25e3, scale_km=10, tag="zoom1")]
_sel = __import__("os").environ.get("FIGS8_REGIONS")
if _sel:                                                   # test renders before all regions exist
    REGS = [r for r in REGS if r["key"] in _sel.split(",")]; WIN = [w for w in WIN if w["key"] in _sel.split(",")]


def load_region(r):
    key = r["key"]; cells, lut, ky0, kx0 = F6.cell_table(key)
    m = pd.read_csv(MAPS / r["reg"] / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    v = F6.cell_values(key, cells, m.r1); a = np.nan_to_num(F6.cell_values(key, cells, m.aoa_in.astype(float)), nan=0.0)
    b = F6.Base(key, cells, lut, ky0, kx0, step=r["step"], tag="full"); tf = b.tf; lim = F6.region_limits(key, tf)
    return dict(r, cells=cells, lut=lut, ky0=ky0, kx0=kx0, mean=m, v=v, a=a, base=b, tf=tf, lim=lim, asp=(lim[3] - lim[2]) / (lim[1] - lim[0]))


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    R = {r["key"]: load_region(r) for r in REGS}
    for k, Rk in R.items():
        olon, olat, nloc = F6.obs_cells(lab, Rk["reg"]); Rk["ox"], Rk["oy"] = Rk["tf"].transform(olon, olat); Rk["nloc"] = int(nloc)
    allm = np.concatenate([R[k]["mean"].r1.values for k in R]); lo, hi = F6.MV.pct_range(allm, step=5.0); norm = Normalize(lo, hi)
    ext = F6.MV.extend_of(allm, lo, hi); tick = 20.0 if hi - lo >= 80 else 10.0; ticks = list(np.arange(tick * np.ceil(lo / tick), hi + 1e-6, tick))
    print(f"  data {time.time() - t0:.0f} s; colour range {lo}-{hi} ({ext})")
    # ---------------- layout (mm)
    W = 180.0; g1 = 3.0
    nw = len(WIN); w1 = (W - 1.0 - 1.0 - 3 * g1) / 4.0; y1 = 4.3; xs1 = [1.0 + j * (w1 + g1) for j in range(nw)]
    y_cb = y1 + w1 + 3.0                                   # colour bar line below the windows (no tick labels under the windows)
    y2 = y_cb + 13.0; h2 = 40.0; g2 = 9.0
    xs2 = {}; x = 9.0
    for r in REGS:
        xs2[r["key"]] = x; x += h2 / R[r["key"]]["asp"] + g2
    assert x - g2 <= W - 0.5, ("row 2 too wide", x - g2)
    y_key2 = y2 + h2 + 6.5
    H = y_key2 + 4.5
    fig = C.figure(W, H); src = {"windows": {}, "aoa": {}}
    # ---------------- a-d: windows (1 km cells)
    zl_by_key = {k: [] for k in R}
    for j, w in enumerate(WIN):
        Rk = R[w["key"]]; tf = Rk["tf"]; big = w["key"] != "lena"
        zl = F6.zoom_lim(tf, w["cen"], w["hx"], w["hx"]); zl_by_key[w["key"]].append(zl)
        pad = 2e3 if big else 1e3
        bz = F6.Base(w["key"], Rk["cells"], Rk["lut"], Rk["ky0"], Rk["kx0"], step=1, window=[zl[0] - pad, zl[1] + pad, zl[2] - pad, zl[3] + pad], tag=w["tag"])
        rgb, _ = MP.render(bz, Rk["v"], MP.CM_ALT, norm, aoa=Rk["a"], smooth=False, w_shade=W_SHADE)
        ax = C.ax_mm(fig, xs1[j], y1, w1, w1); MP.frame(ax, rgb, bz.extent, zl); del rgb
        F6.obs_marks(ax, Rk["ox"], Rk["oy"], zl, s=M.obs_zoom_s(w1 / ((zl[1] - zl[0]) / 1e3)), lw=0.45, case=0.3, pad_frac=0.012)
        MP.scalebar(ax, zl, w["scale_km"], M, fx=0.05 if j != 1 else 0.62); MP.corner_label(ax, w["name"], M)   # b: scale bar bottom right (observed cells at the left)
        out_cells = F6.zoom_outside(bz, Rk["v"], Rk["a"], zl)
        src["windows"][w["name"]] = dict(panel="abcd"[j], region=C.REGION[Rk["reg"]][0], centre_lonlat=list(w["cen"]), width_km=2 * w["hx"] / 1e3,
                                         cells_outside_aoa_in_window=out_cells, scale_bar_km=w["scale_km"])
        C.letter(fig, xs1[j], 0.6, "abcd"[j])
        del bz; gc.collect()
    cax = C.ax_mm(fig, xs1[0] + 12.0, y_cb, w1 + g1 + w1 - 24.0, 1.9)                 # colour bar under windows a-b, key under c-d
    MP.colorbar(fig, cax, MP.CM_ALT, norm, "ALT, mean 2011–2024 (cm)", ticks, extend=ext)
    fig.legend([Line2D([], [], ls="none", marker="o", ms=2.6, mfc="none", mec=C.TEXT, mew=0.45), MP.FadedRamp(MP.CM_ALT)],
               ["Observed cell", "Outside area of applicability (lighter colours)"], loc="upper left", bbox_to_anchor=((xs1[min(2, nw - 1)] + 2.0) / W, 1 - (y_cb - 0.4) / H), ncol=1,
               fontsize=C.FS_SMALL, handlelength=1.8, handleheight=0.9, labelspacing=0.4, handler_map={MP.FadedRamp: MP.RampHandler()}, borderaxespad=0, borderpad=0)
    # ---------------- e-g: area of applicability
    for i, r in enumerate(REGS):
        Rk = R[r["key"]]; b = Rk["base"]; w2 = h2 / Rk["asp"]
        ax = C.ax_mm(fig, xs2[r["key"]], y2, w2, h2)
        rgb, _ = MP.render(b, np.where(np.isfinite(Rk["v"]), Rk["a"], np.nan), ListedColormap([AOA_OUT, AOA_IN]), Normalize(-0.5, 1.5), smooth=False, w_shade=W_SHADE)
        MP.frame(ax, rgb, b.extent, Rk["lim"]); del rgb
        MP.coast(ax, r["key"], Rk["tf"], M, base=b, cells=Rk["cells"], tol_m=600.0); MP.graticule(ax, r["key"], Rk["tf"], Rk["lim"], M, lon_lab=r["lon_lab"], lat_lab=r["lat_lab"])
        ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
        for zl in zl_by_key[r["key"]]:
            MP.box(ax, zl, lw=0.6)
        MP.scalebar(ax, Rk["lim"], r["scale_km"], M); MP.corner_label(ax, C.REGION[Rk["reg"]][0], M)
        C.letter(fig, xs2[r["key"]] - 8.0, y2 - 3.7, "efg"[i])
        src["aoa"][C.REGION[Rk["reg"]][0]] = dict(panel="efg"[i], frac_inside_aoa=round(float(Rk["mean"].aoa_in.mean()), 4), n_cells_mapped=int(np.isfinite(Rk["v"]).sum()),
                                                  observed_sites=Rk["nloc"])
        Rk["base"] = None; gc.collect()
    fig.legend([Patch(facecolor=AOA_IN, edgecolor="none"), Patch(facecolor=AOA_OUT, edgecolor="none"), Patch(facecolor=MP.LAND_HEX, edgecolor="#B5B5B5", lw=0.3),
                Patch(facecolor=MP.WATER_HEX, edgecolor="#B5B5B5", lw=0.3)],
               ["Inside area of applicability", "Outside", "Not mapped", "Water"], loc="upper left",
               bbox_to_anchor=(9.0 / W, 1 - y_key2 / H), ncol=4, fontsize=C.FS_SMALL, handlelength=1.3, handleheight=0.9, columnspacing=1.4, borderaxespad=0, borderpad=0)
    src["colour_range_cm"] = [lo, hi]; src["colours"] = dict(inside=AOA_IN, outside=AOA_OUT)
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S8", dict(
        title="1 km detail and area of applicability in the three regions", stem=STEM, script="scripts/4_visualization/paper_v14/si_s08_map_detail.py",
        inputs=["data/processed/v5/maps/{Alaska,Lena,Canada}/mean_2011_2024.csv.gz (r1, aoa_in)",
                "data/processed/map_{alaska,lena,canada}/{*_grid_x25_v1.csv.gz, hillshade_v1/shade_250m.tif}", "data/processed/v5/labels_cell_year_v5.csv"],
        years=dict(a_d="mean of the yearly maps 2011-2024", e_g="static (covariate dissimilarity to the training labels, 2011-2024 mean map table)"),
        values=src, numbers_printed="scale bars (10, 10, 5, 10 km; 200, 50, 200 km), colour bar ticks",
        legend_draft="Fig. S8 | 1 km detail and area of applicability. a–d, Mean ALT 2011–2024 in four windows drawn as 1 km cells: North Slope near "
                     "Toolik (a) and Interior near Fairbanks (b), Alaska, 50 km; Samoylov–Kurungnakh, Lena Delta (c), 24 km; Mackenzie Delta near Inuvik, "
                     "Canada (d), 50 km. e–g, Area of applicability of the Alaska, Lena Delta and Canada (NW) maps with the windows of a–d.",
        outputs=C.file_info(paths), size_mm=[W, round(H, 1)], pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
