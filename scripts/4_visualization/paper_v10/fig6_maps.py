"""Fig. 6 (v10 manuscript): year-labelled 1 km ALT maps (Physics + residual ML).

a  Alaska mean ALT 2011-2024 (mean of the yearly maps), 1 km cells drawn with 250 m display interpolation and a subtle hillshade; observed cells
   (1 km map cells containing a label site, in_forcing == 1) as small open circles; cells outside the area of applicability blended 40 % towards the
   land grey. Colour bar ALT (cm), cmcrameri oslo_r.
b  ALT anomaly of each year 2011-2024 relative to the 2011-2024 mean (year map - mean map, per cell), 14 small maps, cmcrameri vik, symmetric
   +-15 cm (values beyond are drawn at the end colours), outside the area of applicability lighter (same rule as a). No hillshade.
c  Regional mean anomaly 2011-2024 (mean over all mapped cells of the year map minus the 2011-2024 mean of these yearly means) for Alaska and the
   Lena Delta (lines), and the mean anomaly of observed cells (labels matched to the nearest map cell <= 1.5 km; observed ALT minus the cell's
   2011-2024 map mean, averaged per year, centred to the predicted anomaly mean of the same years) for years with >= 50 cell-years (open circles).
Inputs: data/processed/v5/maps/{Alaska,Lena}/{pred_<year>,mean_2011_2024}.csv.gz, data/processed/map_alaska/{alaska_grid_x25_v1.csv.gz,
        hillshade_v1/{shade,dem}_250m.tif}, data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1), caches outputs/figures/paper/v7/_cache (read).
Outputs: outputs/figures/paper/v10/Fig6_maps.{pdf,svg,png}, outputs/figures/paper/v10/source_data/Fig6_maps.json
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/fig6_maps.py
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
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

F6 = MP.F6
import v7_f7_series_significance as F7  # noqa: E402  (observed_anomaly: same definition as the deck R10)

ROOT = C.ROOT
PROC = ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"
STEM = "Fig6_maps"
YEARS = list(range(2011, 2025))
ANOM = 15.0                                   # symmetric anomaly range (cm)
STEP_A, STEP_B = 1, 4                         # a: 250 m display grid; b: 1 km grid (no interpolation)
W_SHADE_A = 0.30


def load():
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    cells, lut, ky0, kx0 = F6.cell_table("alaska")
    mean = pd.read_csv(MAPS / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str})
    m = mean.set_index("cell_id")
    vmean = F6.cell_values("alaska", cells, m.r1); aoa = np.nan_to_num(F6.cell_values("alaska", cells, m.aoa_in.astype(float)), nan=0.0)
    anom, reg_mean = {}, {}
    for y in YEARS:
        p = pd.read_csv(MAPS / "Alaska" / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
        reg_mean[y] = float(p.reindex(mean.cell_id).mean())
        anom[y] = F6.cell_values("alaska", cells, p) - vmean
    reg_mean_L = {y: float(pd.read_csv(MAPS / "Lena" / f"pred_{y}.csv.gz", usecols=["r1"]).r1.mean()) for y in YEARS}
    meanL = pd.read_csv(MAPS / "Lena" / "mean_2011_2024.csv.gz", dtype={"cell_id": str})
    return dict(lab=lab, cells=cells, lut=lut, ky0=ky0, kx0=kx0, mean=mean, vmean=vmean, aoa=aoa, anom=anom, reg_mean=reg_mean,
                reg_mean_L=reg_mean_L, meanL=meanL)


def series(D):
    out = {}
    for reg, rm, cellt in (("Alaska", D["reg_mean"], D["mean"]), ("Lena", D["reg_mean_L"], D["meanL"])):
        mu = float(np.mean(list(rm.values()))); pred = {y: rm[y] - mu for y in YEARS}
        obs = F7.observed_anomaly(D["lab"], reg, cellt, pred, min_n=50, max_km=1.5)
        out[reg] = dict(mean_cm=mu, pred=pred, obs=obs)
    return out


def draw_series(ax, reg, S_, show_ylabel, M):
    C.clean_axes(ax)
    s = S_[reg]; pred = np.array([s["pred"][y] for y in YEARS]); o = s["obs"]
    name, col, mk = C.REGION[reg]
    ax.axhline(0, color="#9A9A9A", lw=C.LW_REF, zorder=1)
    ax.plot(YEARS, pred, color=col, lw=C.LW_MAIN, zorder=3, solid_capstyle="round")
    ax.plot(o.index.values, o["mean"].values, ls="none", marker="o", ms=3.4, mfc="white", mec="black", mew=0.6, zorder=4)
    vmax = max(float(np.abs(pred).max()), float(np.abs(o["mean"]).max()))
    lo = -5.0 * np.ceil((max(float(np.abs(pred).max()), float(-o["mean"].min())) + 1.0) / 5.0)
    hi = 5.0 * np.ceil((float(max(pred.max(), o["mean"].max())) + 1.5) / 5.0)
    ax.set_ylim(min(lo, -10), max(hi, 10)); ax.set_xlim(2010.4, 2024.6)
    ax.set_xticks([2012, 2016, 2020, 2024]); ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.tick_params(which="minor", length=1.0, width=0.4)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20 if (hi - lo) > 45 else (10 if (hi - lo) > 18 else 5)))
    hmm = ax.get_position().height * ax.figure.get_figheight() * 25.4
    ax.text(0.0, 1.0 + 1.2 / hmm, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
    labelled = []
    for yr, row in o.iterrows():                                     # label the observed point far outside the predicted range (Lena 2020)
        if abs(row["mean"]) > 2.0 * float(np.abs(pred).max()) and abs(row["mean"]) > 15:
            ax.annotate(f"n = {int(row['size'])}", (yr, row["mean"]), xytext=(4.5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=C.FS_SMALL, color=C.TEXT)
            labelled.append(dict(year=int(yr), anomaly_cm=round(float(row["mean"]), 2), n=int(row["size"])))
    if show_ylabel:
        ax.set_ylabel("ALT anomaly (cm)")
    ax.set_xlabel("Year")
    return labelled, [float(v) for v in ax.get_ylim()]


def main():
    t0 = time.time(); C.setup(); M = MP.Med()
    D = load(); print(f"  data {time.time() - t0:.0f} s")
    cells, lut, ky0, kx0 = D["cells"], D["lut"], D["ky0"], D["kx0"]
    src = dict(figure="Fig. 6", script="scripts/4_visualization/paper_v10/fig6_maps.py", years=[YEARS[0], YEARS[-1]],
               inputs=dict(maps="data/processed/v5/maps/{Alaska,Lena}/pred_<year>.csv.gz and mean_2011_2024.csv.gz (column r1 = Physics + residual ML)",
                           grid="data/processed/map_alaska/alaska_grid_x25_v1.csv.gz; hillshade data/processed/map_alaska/hillshade_v1/shade_250m.tif",
                           labels="data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)", projection="EPSG:3338 (Alaska Albers)"))
    # ---------------- layout (mm)
    W = 180.0; lim = None
    base_b = F6.Base("alaska", cells, lut, ky0, kx0, step=STEP_B, tag="full"); tf = base_b.tf
    lim = F6.region_limits("alaska", tf); asp = (lim[3] - lim[2]) / (lim[1] - lim[0])
    xl, g, gab, xr = 7.6, 1.0, 6.5, 0.8
    w = (W - xl - gab - xr - 3 * g - 3 * g * 1.0 / asp / 1.0 * 0 - 4 * g) / 8.0 if False else None
    # a height = 4 small-map rows: wa*asp = 4*w*asp + 3*g  ->  wa = 4w + 3g/asp ; width: xl + wa + gab + 4w + 3g + xr = W
    w = (W - xl - gab - xr - 3 * g - 3 * g / asp) / 8.0; wa = 4 * w + 3 * g / asp; ha = wa * asp; hb = w * asp
    ya = 4.2; xa = xl; xb = xa + wa + gab
    H = 142.0
    fig = C.figure(W, H)
    # ---------------- a
    axa = C.ax_mm(fig, xa, ya, wa, ha)
    t1 = time.time()
    base_a = F6.Base("alaska", cells, lut, ky0, kx0, step=STEP_A, tag="full")
    lo, hi = F6.MV.pct_range(D["mean"].r1.values); normA = Normalize(lo, hi); extA = F6.MV.extend_of(D["mean"].r1.values, lo, hi)
    fe = 2.0 * 2.0 / base_a.step; sg = 3.0 * 2.0 / base_a.step
    rgb, n_drawn_px = MP.render(base_a, D["vmean"], MP.CM_ALT, normA, aoa=D["aoa"], smooth=True, fill_max=F6.FILL_MAX, w_shade=W_SHADE_A,
                                feather=fe, aoa_sigma=sg)
    MP.frame(axa, rgb, base_a.extent, lim); del rgb; gc.collect()
    shade_fill = base_a.shade_fill
    print(f"  a rendered {time.time() - t1:.0f} s")
    F6.coast(axa, "alaska", tf, M)
    F6.graticule(axa, "alaska", tf, lim, M, lon_lab=(-165, -155, -145), lat_lab=(60, 65, 70))
    olon, olat, nloc = F6.obs_cells(D["lab"], "Alaska"); ox, oy = tf.transform(olon, olat)
    n_obs = F6.obs_marks(axa, ox, oy, lim, s=1.6 ** 2, lw=0.28, case=0.18)
    F6.scalebar(axa, lim, 200, M, fx=0.05, fy=0.05)
    del base_a; gc.collect()
    C.letter(fig, 0.4, 0.4, "a")
    # colour bar + legend under a
    y_cb = ya + ha + 6.6
    cax = C.ax_mm(fig, xa + 1.0, y_cb, 34.0, 2.0)
    MP.colorbar(fig, cax, MP.CM_ALT, normA, "ALT (cm)", list(np.arange(lo, hi + 1e-6, 10.0)), extend=extA)
    mid = np.array(MP.CM_ALT(0.55)[:3], np.float32); faded = mid * (1 - MP.FADE) + MP.FADE * MP.LAND
    hs = [Line2D([], [], ls="none", marker="o", ms=2.2, mfc="none", mec="black", mew=0.35),
          Patch(facecolor=tuple(faded.tolist()), edgecolor="#9A9A9A", lw=0.3),
          Patch(facecolor=MP.LAND_HEX, edgecolor="#9A9A9A", lw=0.3), Patch(facecolor=MP.WATER_HEX, edgecolor="#9A9A9A", lw=0.3)]
    labs = ["Observed cell", "Outside area of applicability", "Not mapped", "Water"]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=((xa + 40.0) / W, 1 - (y_cb - 1.6) / H), ncol=2, handlelength=1.3, handleheight=0.9,
               handletextpad=0.45, columnspacing=1.0, labelspacing=0.45, fontsize=C.FS_SMALL)
    # ---------------- b
    normB = Normalize(-ANOM, ANOM); allv = np.concatenate([v[np.isfinite(v)] for v in D["anom"].values()])
    extB = F6.MV.extend_of(allv, -ANOM, ANOM)
    frac_beyond = float((np.abs(allv) > ANOM).mean())
    axes_b = []; t1 = time.time(); stats_b = {}
    Mb = MP.Med(); Mb.lw_coast = 0.25
    for k, y in enumerate(YEARS):
        r, c = divmod(k, 4)
        ax = C.ax_mm(fig, xb + c * (w + g), ya + r * (hb + g), w, hb); axes_b.append(ax)
        rgb, _ = MP.render(base_b, D["anom"][y], MP.CM_ANOM, normB, aoa=D["aoa"], smooth=False, w_shade=0.0)
        MP.frame(ax, rgb, base_b.extent, lim); del rgb
        F6.coast(ax, "alaska", tf, Mb)
        MP.corner_label(ax, str(y), M, x=0.04, y=0.955)
        a = D["anom"][y]; f = np.isfinite(a)
        stats_b[str(y)] = dict(mean_cm=round(float(np.nanmean(a)), 3), p01_cm=round(float(np.nanpercentile(a, 1)), 2), p99_cm=round(float(np.nanpercentile(a, 99)), 2),
                               frac_beyond_15cm=round(float((np.abs(a[f]) > ANOM).mean()), 4))
    print(f"  b rendered {time.time() - t1:.0f} s")
    C.letter(fig, xb - 4.6, 0.4, "b")
    # anomaly colour bar in the two free slots (row 4, columns 3-4)
    xk = xb + 2 * (w + g); yk = ya + 3 * (hb + g)
    cbx = C.ax_mm(fig, xk + 3.0, yk + hb * 0.42, 2 * w + g - 6.0, 2.0)
    MP.colorbar(fig, cbx, MP.CM_ANOM, normB, "ALT anomaly (cm)", [-15, -10, -5, 0, 5, 10, 15], extend=extB)
    # ---------------- c
    S_ = series(D)
    yc = 108.0; hc = 24.0; xc0 = 15.0; gapc = 16.0; wc = (W - xc0 - gapc - 2.0) / 2.0
    axc = [C.ax_mm(fig, xc0 + j * (wc + gapc), yc, wc, hc) for j in range(2)]
    lab_c = {}; ylims = {}
    for j, reg in enumerate(("Alaska", "Lena")):
        lab_c[reg], ylims[reg] = draw_series(axc[j], reg, S_, j == 0, M)
    C.letter(fig, 0.4, yc - 7.4, "c")
    hc_ = [Line2D([], [], color="#1A1A1A", lw=C.LW_MAIN), Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec="black", mew=0.6)]
    fig.legend(hc_, ["Mapped cells", "Observed cells (years with ≥ 50 cell-years)"], loc="upper left",
               bbox_to_anchor=(xc0 / W, 1 - (yc - 7.0) / H), ncol=2, handlelength=1.8, columnspacing=1.8, fontsize=C.FS_TICK)
    # ---------------- QA + export
    rep = C.qa_figure(fig, ink_axes=axc, ignore_ink_for=[str(y) for y in YEARS] + ["km"])
    C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN)
    plt.close(fig)
    src.update(
        a=dict(variable="mean ALT 2011-2024 of Physics + residual ML (mean_2011_2024.csv.gz, r1)", colormap="cmcrameri oslo_r truncated 0.12-0.90",
               range_cm=[lo, hi], extend=extA, display="1 km cell centres bilinearly interpolated to the 250 m hillshade grid (v7_f6_maps.interp_cells); "
               "gaps of <= 2 cells filled for display; hillshade weight 0.30 on mapped cells only",
               n_cells_mapped=int(np.isfinite(D["vmean"]).sum()), frac_inside_aoa=round(float(D["mean"].aoa_in.mean()), 4),
               outside_aoa=f"colour blended {int(MP.FADE * 100)} % towards the land grey {MP.LAND_HEX}", observed_cells_drawn=n_obs,
               observed_sites=int(nloc), land=MP.LAND_HEX, water=MP.WATER_HEX, scale_bar_km=200),
        b=dict(variable="ALT(year) - mean ALT 2011-2024, per 1 km cell", colormap="cmcrameri vik", range_cm=[-ANOM, ANOM], extend=extB,
               frac_cells_beyond_range_all_years=round(frac_beyond, 4), outside_aoa="same blending as a (lighter)", years=YEARS, per_year=stats_b),
        c=dict(definition="mapped cells: mean of the year map over all mapped cells minus the 2011-2024 mean of these yearly means; observed cells: "
                          "labels (in_forcing == 1) matched to the nearest map cell within 1.5 km, observed ALT minus the cell's 2011-2024 map mean, "
                          "averaged per year for years with >= 50 cell-years, then centred to the mean predicted anomaly of the same years "
                          "(v7_f7_series_significance.observed_anomaly)",
               regions={reg: dict(mean_2011_2024_cm=round(S_[reg]["mean_cm"], 3), predicted_anomaly_cm={str(y): round(S_[reg]["pred"][y], 3) for y in YEARS},
                                  observed_anomaly_cm={str(int(y)): round(float(v), 3) for y, v in S_[reg]["obs"]["mean"].items()},
                                  observed_n_cell_years={str(int(y)): int(v) for y, v in S_[reg]["obs"]["size"].items()},
                                  observed_years=[int(y) for y in S_[reg]["obs"].index], labelled_points=lab_c[reg], ylim_cm=ylims[reg])
                        for reg in ("Alaska", "Lena")}),
        years_used=dict(a="2011-2024 (mean of the 14 yearly maps)", b=f"{YEARS[0]}-{YEARS[-1]} (one map per year)",
                        c=dict(mapped_cells=f"{YEARS[0]}-{YEARS[-1]}", observed_cells={reg: [int(y) for y in S_[reg]["obs"].index] for reg in ("Alaska", "Lena")})),
        outputs=C.file_info(paths), canvas_mm=[W, H], pdf_page_mm=C.pdf_page_mm(paths[0]), qa=rep,
        layout_mm=dict(a=[xa, ya, round(wa, 2), round(ha, 2)], b_map=[round(w, 2), round(hb, 2)], c=[xc0, yc, round(wc, 2), hc]))
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths:
        print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
