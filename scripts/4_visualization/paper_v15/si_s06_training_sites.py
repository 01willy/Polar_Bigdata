"""Supplementary Fig. S5 (v11 numbering; S10 in v10): product training-site distances and latitude-band bias (content of v9_f11_qa.py a, b, d), v11 style.
Copied from scripts/4_visualization/paper_v10/si_s10_training_sites.py and edited (v11 colours: Wei 2026 sites in the Wei colour, label cells and
distance curves in region colours, product bars in product colours; dark-grey text; output v11_si).

a  Wei 2026 training sites (supplement table ALT_variables.xlsx) and the label cells of the three regions (north polar stereographic).
b  Cumulative share of label cells against the distance to the nearest Wei 2026 training site (log axis; dashed line 25 km).
c  Alaska 2013: mean difference product map - our map (Physics + residual ML, 2013, TDD to 1 September) by latitude band and elevation class
   (<= 600 m, > 600 m) for Wei 2026 and Liu 2024 (all 1 km cells with a value in both maps; outputs/figures/paper/v15/source_data/Fig6_product_maps.json).
Inputs: data/raw/qtp_alt_open_2026-09-29/essd-2026-447_supplement/ALT_variables.xlsx, data/processed/xbatch/XG_product_comparison/xg_leak_cells_v1.csv,
        data/processed/v5/labels_cell_year_v5.csv, data/processed/v9_qa/map_band_alaska2014_by_elev.csv.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v15/si_s05_training_sites.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

STEM = "FigS6_training_sites"
ROOT = C.ROOT
Q = ROOT / "data" / "processed" / "v9_qa"
XLSX = ROOT / "data" / "raw" / "qtp_alt_open_2026-09-29" / "essd-2026-447_supplement" / "ALT_variables.xlsx"
LEAK = ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv"
REG = C.REGIONS
LS = {"Alaska": "-", "Lena": C.DASH, "Canada": (0, (1.0, 1.2))}
BAND = {"(59, 64]": "59–64°N", "(64, 66]": "64–66°N", "(66, 68]": "66–68°N", "(68, 70]": "68–70°N", "(70, 72]": "70–72°N"}


def band_table():
    """Panel c table (band, mtn, n, ours, wei_minus, liu_minus): v14 reads the Alaska band differences of the current Fig. 6 (2013 map, TDD to
    1 September, all 1 km cells with a value in both maps) from outputs/figures/paper/v15/source_data/Fig6_product_maps.json; the old
    data/processed/v9_qa/map_band_alaska2014_by_elev.csv (full-season 2014 map, every tenth cell) is used only if the JSON is missing."""
    import json
    j = ROOT / "outputs" / "figures" / "paper" / "v15" / "source_data" / "Fig6_product_maps.json"
    if j.exists():
        rows = json.loads(j.read_text())["values"]["Alaska"]["band_differences"]["rows"]
        return pd.DataFrame([dict(band=r["band"], mtn=r["mtn"], n=r["n_cells"], ours=r["ours_mean_cm"], wei_minus=r["wei_minus_ours_cm"], liu_minus=r["liu_minus_ours_cm"])
                             for r in rows if r["n_cells"] > 0])
    return pd.read_csv(Q / "map_band_alaska2014_by_elev.csv")


def main():
    C.setup()
    plt.rcParams["hatch.linewidth"] = 0.45
    W, H = 180.0, 86.0
    fig = C.figure(W, H); src = {}
    # ---------------- a: map
    axm = fig.add_axes([0.5 / W, 1 - (4.0 + 66.0) / H, 66.0 / W, 66.0 / H], projection=ccrs.NorthPolarStereo(central_longitude=-100))
    PC = ccrs.PlateCarree(); axm.set_extent([-180, 180, 50, 90], PC)
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#E6E6E6", edgecolor="none", zorder=0)
    axm.coastlines("50m", linewidth=0.3, color="#9A9A9A", zorder=1)
    axm.gridlines(crs=PC, linewidth=0.3, color="#D0D0D0", xlocs=range(-180, 180, 60), ylocs=[60, 70, 80], zorder=1)
    axm.spines["geo"].set_linewidth(0.4); axm.spines["geo"].set_edgecolor(C.AXIS)
    lab = pd.read_csv(ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv", usecols=["region", "lat", "lon"]).drop_duplicates()
    for r in REG:
        q = lab[lab.region == r]; nm, col, mk = C.REGION[r]
        axm.scatter(q.lon, q.lat, s=4.2, marker=mk, color=col, transform=PC, lw=0, zorder=3)
    w = pd.read_excel(XLSX, usecols=["LAT", "LONG"]).drop_duplicates()
    WEI = C.PRODUCT["wei"][1]; WEI_D = C.darker(WEI, 0.8)
    axm.scatter(w.LONG, w.LAT, s=5.0, marker="x", color=WEI_D, lw=0.4, transform=PC, zorder=4)
    hs = [Line2D([], [], ls="none", marker="x", color=WEI_D, ms=3.2, mew=0.6)] + \
         [Line2D([], [], ls="none", marker=C.REGION[r][2], ms=2.8, mfc=C.REGION[r][1], mec="none") for r in REG]
    fig.legend(hs, ["Wei 2026 training sites"] + [f"{C.REGION[r][0]} label cells" for r in REG], loc="upper left",
               bbox_to_anchor=(1.0 / W, 1 - 71.5 / H), ncol=2, fontsize=C.FS_SMALL, handlelength=0.9, labelspacing=0.35, columnspacing=1.2)
    src["a"] = dict(n_wei_sites=int(len(w)), colours=dict(wei_sites=WEI_D, regions={C.REGION[r][0]: C.REGION[r][1] for r in REG}))
    # ---------------- b: distance distribution
    axb = C.ax_mm(fig, 86.0, 5.0, 90.0, 27.0); C.clean_axes(axb)
    lk = pd.read_csv(LEAK); lk = lk[(lk["product"] == "wei") & lk.target.str.endswith("|r")]; lk["region"] = lk.target.str.split("|").str[0]
    for r in REG:
        d = np.sort(lk[lk.region == r].dist_km.values)
        axb.plot(np.maximum(d, 0.5), np.arange(1, len(d) + 1) / len(d), color=C.REGION[r][1], lw=C.LW_MAIN, ls=LS[r])
        src.setdefault("b", {})[r] = dict(cells=int(len(d)), median_km=round(float(np.median(d)), 2), share_le25km=round(float((d <= 25).mean()), 4),
                                          share_le100km=round(float((d <= 100).mean()), 4))
    axb.axvline(25, color=C.ZERO, lw=0.5, ls=C.DASH, zorder=0)
    axb.set_xscale("log"); axb.set_xlim(0.5, 2000); axb.set_xticks([1, 10, 25, 100, 1000]); axb.set_xticklabels(["1", "10", "25", "100", "1000"])
    axb.minorticks_off(); axb.set_ylim(0, 1.02); axb.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    axb.set_xlabel("Distance to the nearest Wei 2026 training site (km)"); axb.set_ylabel("Share of label cells")
    axb.legend([Line2D([], [], color=C.REGION[r][1], lw=C.LW_MAIN, ls=LS[r]) for r in REG], [C.REGION[r][0] for r in REG], loc="lower right",
               bbox_to_anchor=(1.0, 0.02), fontsize=C.FS_SMALL, handlelength=2.2)
    # ---------------- c: latitude band x elevation, product - our map (Alaska 2013, v14 Fig. 6 source data)
    axc = C.ax_mm(fig, 86.0, 47.0, 90.0, 31.0); C.clean_axes(axc)
    e = band_table(); bands = list(dict.fromkeys(e.band))
    _, wf, we, _ = C.product_style("wei"); _, lf, le, _ = C.product_style("liu")
    HT = {"wei_minus": "//////", "liu_minus": "\\\\\\\\\\\\"}           # > 600 m: hatched (Wei and Liu in opposite directions)
    for j, mtn in enumerate(("elev<=600m", "elev>600m")):
        for k, (face, edge, col) in enumerate(((wf, we, "wei_minus"), (lf, le, "liu_minus"))):
            q = e[e.mtn == mtn].set_index("band").reindex(bands)
            x = np.arange(len(bands)) + (j * 2 + k - 1.5) * 0.19
            hatch = None if j == 0 else HT[col]
            axc.bar(x, q[col].values, width=0.17, color=face if hatch is None else "white", edgecolor=edge, hatch=hatch, lw=0.5, zorder=2)
            src.setdefault("c", {}).setdefault(mtn, {})[col.split("_")[0]] = {BAND[b]: (None if not np.isfinite(v) else round(float(v), 2))
                                                                                 for b, v in zip(bands, q[col].values)}
        src["c"].setdefault("n_cells", {})[mtn] = {BAND[b]: (None if not np.isfinite(v) else int(v)) for b, v in zip(bands, e[e.mtn == mtn].set_index("band").reindex(bands).n.values)}
    axc.axhline(0, color=C.ZERO, lw=C.LW_REF, zorder=1)
    axc.set_xticks(range(len(bands))); axc.set_xticklabels([BAND[b] for b in bands]); axc.tick_params(axis="x", length=0, pad=3)
    axc.set_xlim(-0.6, len(bands) - 0.4); axc.yaxis.set_major_locator(mticker.MultipleLocator(25)); axc.set_ylim(-15, 140)
    axc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−")))
    axc.set_ylabel("Product − our map (cm)")
    hs = [Patch(facecolor=wf, edgecolor=we, lw=0.5), Patch(facecolor="white", edgecolor=we, hatch=HT["wei_minus"], lw=0.5),
          Patch(facecolor=lf, edgecolor=le, lw=0.5), Patch(facecolor="white", edgecolor=le, hatch=HT["liu_minus"], lw=0.5)]
    axc.legend(hs, ["Wei 2026, elevation ≤ 600 m", "Wei 2026, elevation > 600 m", "Liu 2024, elevation ≤ 600 m", "Liu 2024, elevation > 600 m"],
               loc="upper right", bbox_to_anchor=(1.0, 1.03), ncol=2, fontsize=C.FS_SMALL, handlelength=1.4, handleheight=0.9, columnspacing=1.0)
    C.letter(fig, 0.5, 0.5, "a"); C.letter(fig, 72.0, 0.5, "b"); C.letter(fig, 72.0, 42.5, "c")
    rep = C.qa_figure(fig); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S6", dict(
        title="Product training-site distances and latitude-band bias", stem=STEM, script="scripts/4_visualization/paper_v15/si_s05_training_sites.py",
        v10_number="S10 (scripts/4_visualization/paper_v10/si_s10_training_sites.py)",
        inputs=[str(XLSX.relative_to(ROOT)), str(LEAK.relative_to(ROOT)) + " (product == 'wei', label cells '|r')", "data/processed/v5/labels_cell_year_v5.csv",
                "data/processed/v9_qa/map_band_alaska2014_by_elev.csv"],
        years=dict(a_b="label cells of all years (Alaska 2010-2024, Lena Delta 2010-2022, Canada (NW) 2010-2024)", c="2013 (Alaska map with the TDD to 1 September, all 1 km cells with a value in both maps; v14 Fig. 6 source data)"),
        definition=dict(c="mean of (product - Physics + residual ML map) over every 10th 1 km map cell of Alaska 2013, by latitude band and elevation class"),
        values=src, numbers_printed="none (axis ticks only)", omitted="v9_f11_qa.py panel c (Lena Delta transfer RMSE vs products) is not part of this figure",
        legend_draft="Fig. S5 | Distance to product training sites and latitude-band differences. a, Wei 2026 training sites and the label cells of the "
                     "three regions. b, cumulative share of label cells against the distance to the nearest Wei 2026 training site (dashed line: "
                     "25 km). c, Alaska 2013: mean difference between product and our map by latitude band and elevation class.",
        outputs=C.file_info(paths), size_mm=[W, H], pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
