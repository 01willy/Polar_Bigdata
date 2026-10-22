"""Round-4 figure proposals (2026-10-22), part A: groups Fig 1 (data), Fig 2 (workflow) and Fig 3 (accuracy). Three designs per group, real data,
paper_v15 palette, FreeSans 6.5/7/8 pt, no icons, no equations inside the figures. Each design is one message.

Fig 1 group
  P1_1  record_length_maps   three regional maps, sites marked by the number of observed years (1 / 2-4 / >= 5) with counts; shows where repeat observations exist
  P1_2  label_characteristics observed ALT distributions by region and source (ECDF), observation day of year, and the block-CV residual by source
  P1_3  climate_space        sites against the mapped domain in climate-terrain space (TDD of the observation year x elevation), map cells as grey density
Fig 2 group
  P2_1  equation_strip       Alaska 2014: Stefan term map + learned residual map = ALT map, each with its value histogram; words only, no boxes or arrows
  P2_2  site_example         one long-record site per region: observed ALT by year with the Stefan and physics-residual out-of-fold predictions and the residual
  P2_3  method_spectrum      Stefan (source coeff.) -> Stefan (recalibrated) -> physics + residual ML -> direct ML: what each learns from the labels,
                             observed against predicted (Lena Delta, spatial-block CV) and RMSE in and out of the region
Fig 3 group
  P3_1  design_ladder        RMSE against the validation design (least to most demanding), one panel per region, three methods as lines
  P3_2  block_gain_maps      spatial-block CV: each 0.5 degree block coloured by the RMSE change of the physics-residual model against recalibrated Stefan
  P3_3  bias_by_decile       mean error by decile of observed ALT (block CV) for the three methods: compression of extremes, one panel per region

Run   cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper15 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 \
        python3 scripts/4_visualization/paper_v15/proposals_v15_2a.py [P1_1 P1_2 ...]
Out   outputs/figures/paper/v15_proposals/<id>_<name>.{pdf,svg,png,json}
"""
from __future__ import annotations
import gc, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import c13_maps as MP  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon, Patch  # noqa: E402

F6 = MP.F6
ROOT = C.ROOT; PROC = ROOT / "data/processed"; RUNS = PROC / "v5/runs/core_v6"; PREDS = RUNS / "preds"; MAPS = PROC / "v5/maps"
OUT = ROOT / "outputs/figures/paper/v15_proposals"; OUT.mkdir(parents=True, exist_ok=True)
REGIONS = ["Alaska", "Lena", "Canada"]; KEY = {"Alaska": "alaska", "Lena": "lena", "Canada": "canada"}; RN = {r: C.REGION[r][0] for r in REGIONS}
RC = {r: C.REGION[r][1] for r in REGIONS}; MC = C.METHOD; MN = C.METHOD_NAME
YEARS = list(range(2011, 2025))
SRC_NAME = {"ABoVE_ds2369": "ABoVE synthesis", "CALM": "CALM", "ALLENA_PANGAEA_973813": "Lena Delta (PANGAEA)"}
SHORT_SRC = {"ABoVE_ds2369": "ABoVE", "CALM": "CALM", "ALLENA_PANGAEA_973813": "PANGAEA"}
GEO = {"alaska": dict(lon_lab=(-160, -150), lat_lab=(60, 65, 70), scale_km=200), "lena": dict(lon_lab=(124, 128), lat_lab=(72, 73), scale_km=50),
       "canada": dict(lon_lab=(-125, -115), lat_lab=(60, 65), scale_km=200)}
NEG, POS = "#3C78A8", "#B5534A"; CM_DIV = LinearSegmentedColormap.from_list("div15", [NEG, "#FFFFFF", POS])


# ---------------------------------------------------------------- data
def labels():
    lab = pd.read_csv(PROC / "v5/labels_cell_year_v5.csv"); return lab[lab.in_forcing == 1].copy()


def preds(region, method, scheme="block"):
    """Seed-mean out-of-fold prediction of core_v6 (forcing year_obs, F8TS). Columns: cell_year_id, loc_id, year, block, y, pred."""
    learner, lam = {"R1": ("catboost", "0.5"), "D0": ("catboost", "-1"), "P1": ("-", "-1"), "P0": ("-", "-1")}[method]
    f = PREDS / f"{region}__{scheme}__year_obs__F8TS__{method}__{learner}__lam{lam}.csv.gz"                  # R1 files: lam0.5; others: lam-1
    d = pd.read_csv(f); d = d.groupby(["cell_year_id", "loc_id", "year", "block"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
    return d


def rmse(e): return float(np.sqrt(np.mean(np.square(np.asarray(e, float)))))


def rmse_table():
    sm = pd.read_csv(RUNS / "summary_metrics.csv"); sm = sm[(sm.forcing == "year_obs") & (sm.featset == "F8TS")]
    def val(region, scheme, m):
        q = sm[(sm.region == region) & (sm.scheme == scheme) & (sm.method == m)]
        if m == "R1": q = q[(q.learner == "catboost") & np.isclose(q.lam, 0.5)]
        if m == "D0": q = q[q.learner == "catboost"]
        return float(q.rmse.iloc[0])
    return val


def cell_ids(lat, lon):
    ky = np.floor(lat / F6.CELL).astype(np.int64); dx = F6.CELL / np.cos(np.radians((ky + 0.5) * F6.CELL)); kx = np.floor(lon / dx).astype(np.int64)
    return np.char.add(np.char.add(ky.astype(str), "_"), kx.astype(str))


class Region:
    """Projection, limits and a land/water base image of one region (map grid at step 4)."""
    def __init__(self, key, step=4):
        self.key = key; self.cells, self.lut, self.ky0, self.kx0 = F6.cell_table(key)
        self.base = F6.Base(key, self.cells, self.lut, self.ky0, self.kx0, step=step, tag="full"); self.tf = self.base.tf; self.lim = F6.region_limits(key, self.tf)
        self.asp = (self.lim[3] - self.lim[2]) / (self.lim[1] - self.lim[0])
    def values(self, series): return F6.cell_values(self.key, self.cells, series)
    def land_rgb(self):
        rgb, _ = MP.render(self.base, np.full(len(self.cells), np.nan, np.float32), MP.CM_ALT, Normalize(0, 1), smooth=False, w_shade=0.0); return rgb
    def frame(self, ax, rgb, M, grat=True, scale=True, lon_lab=None, lat_lab=None):
        g = GEO[self.key]; MP.frame(ax, rgb, self.base.extent, self.lim); MP.coast(ax, self.key, self.tf, M, base=self.base, cells=self.cells, tol_m=2000.0)
        if grat: MP.graticule(ax, self.key, self.tf, self.lim, M, lon_lab=g["lon_lab"] if lon_lab is None else lon_lab, lat_lab=g["lat_lab"] if lat_lab is None else lat_lab)
        ax.tick_params(colors=C.AXIS, labelcolor=C.TEXT)
        if scale: MP.scalebar(ax, self.lim, g["scale_km"], M, fx=0.05, fy=0.05)


def finish(fig, stem, meta, ink_axes=None, ignore=()):
    rep = C.qa_figure(fig, text_ink=False, ink_axes=ink_axes, ignore_ink_for=list(ignore)); C.print_qa(stem, rep)
    paths = C.export(fig, stem, OUT); plt.close(fig)
    meta.update(dict(outputs=C.file_info(paths), canvas_mm=meta.get("canvas_mm"), pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    C.write_json(OUT / f"{stem}.json", meta)
    for p in paths: print(" ", p.name, p.stat().st_size)


def three_maps(fig, y0, w, R, M, rgbs, titles, letters, x0=9.5, gap=None):
    """Three region maps in one row at equal widths (heights follow the region aspect); returns axes and the row height."""
    gap = gap if gap is not None else (180.0 - x0 - 3 * w - 3.5) / 2; axes = []; hmax = 0.0
    for j, r in enumerate(REGIONS):
        Rk = R[r]; h = w * Rk.asp; hmax = max(hmax, h); x = x0 + j * (w + gap)
        ax = C.ax_mm(fig, x, y0, w, h); Rk.frame(ax, rgbs[r], M); axes.append(ax)
        C.letter(fig, x - 7.5, y0 - 4.1, letters[j]); C.text_mm(fig, x - 4.1, y0 - 3.95, titles[j], fontsize=C.FS, ha="left", va="top")
    return axes, hmax


# ================================================================ Fig 1 group
def P1_1():
    """Record length of the observation network: sites by number of observed years, three regional maps + counts."""
    stem = "P1_1_record_length_maps"; t0 = time.time(); lab = labels(); M = MP.Med()
    R = {r: Region(KEY[r]) for r in REGIONS}; rgbs = {r: R[r].land_rgb() for r in REGIONS}
    W = 180.0; w = 50.0; y0 = 6.5; fig = C.figure(W, 100.0)
    axes, hmax = three_maps(fig, y0, w, R, M, rgbs, [RN[r] for r in REGIONS], ["a", "b", "c"])
    classes = [("1 year", 1, 1, "#8C8C8C", 1.4), ("2–4 years", 2, 4, C.tint("#1A7AB3", 0.55), 2.0), ("5 years or more", 5, 99, "#1A3F6B", 2.6)]
    meta = {"classes": {}, "sites": {}}
    for j, r in enumerate(REGIONS):
        d = lab[lab.region == r]; g = d.groupby("loc_id").agg(lat=("lat", "first"), lon=("lon", "first"), ny=("year", "nunique"), n=("year", "size"))
        x, y = R[r].tf.transform(g.lon.values, g.lat.values); meta["sites"][r] = int(len(g))
        for name, lo, hi, col, ms in classes:
            m = (g.ny.values >= lo) & (g.ny.values <= hi); meta["classes"].setdefault(r, {})[name] = int(m.sum())
            axes[j].scatter(x[m], y[m], s=ms ** 2, facecolor=col, edgecolor="white" if lo > 1 else "none", linewidths=0.25, zorder=6 + (lo > 1) + (lo > 4))
        cnt = meta["classes"][r]; txt = "\n".join(f"{cnt[n]:,} sites, {n}" for n, *_ in classes)
        axes[j].text(0.97, 0.03, txt, transform=axes[j].transAxes, ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT, zorder=9,
                     bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    hs = [Line2D([], [], ls="none", marker="o", ms=ms * 1.2, mfc=col, mec="white" if lo > 1 else col, mew=0.3) for _, lo, _, col, ms in classes]
    fig.legend(hs, [f"Observed in {n}" for n, *_ in classes], loc="upper left", bbox_to_anchor=(9.5 / W, 1 - (y0 + hmax + 3.0) / 100.0), ncol=3, frameon=False,
               fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    H = y0 + hmax + 9.5; fig.set_size_inches(W / 25.4, H / 25.4); fig._h_mm = H
    # the figure was created at 100 mm; re-create at the final height so that mm coordinates hold
    plt.close(fig); fig = C.figure(W, H)
    axes, hmax = three_maps(fig, y0, w, R, M, rgbs, [RN[r] for r in REGIONS], ["a", "b", "c"])
    for j, r in enumerate(REGIONS):
        d = lab[lab.region == r]; g = d.groupby("loc_id").agg(lat=("lat", "first"), lon=("lon", "first"), ny=("year", "nunique"))
        x, y = R[r].tf.transform(g.lon.values, g.lat.values)
        for name, lo, hi, col, ms in classes:
            m = (g.ny.values >= lo) & (g.ny.values <= hi)
            axes[j].scatter(x[m], y[m], s=ms ** 2, facecolor=col, edgecolor="white" if lo > 1 else "none", linewidths=0.25, zorder=6 + (lo > 1) + (lo > 4))
        cnt = meta["classes"][r]; txt = "\n".join(f"{cnt[n]:,} sites, {n}" for n, *_ in classes)
        axes[j].text(0.97, 0.03, txt, transform=axes[j].transAxes, ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT, zorder=9,
                     bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    fig.legend(hs, [f"Observed in {n}" for n, *_ in classes], loc="upper left", bbox_to_anchor=(9.5 / W, 1 - (y0 + hmax + 3.0) / H), ncol=3, frameon=False,
               fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    meta.update(dict(proposal="P1_1", message="Repeat observations exist at few sites; most sites were measured once", canvas_mm=[W, round(H, 1)],
                     inputs="data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1); number of distinct years per loc_id"))
    finish(fig, stem, meta, ignore=["sites", "km"]); print(f"  {stem} {time.time() - t0:.0f} s")


def P1_2():
    """What the labels are: ALT distributions by region and source, observation timing, and the block-CV residual by source."""
    stem = "P1_2_label_characteristics"; t0 = time.time(); lab = labels()
    W, H = 180.0, 62.0; fig = C.figure(W, H); meta = {"a": {}, "b": {}, "c": {}}
    xs = [12.0, 66.0, 138.0]; w, h, y0 = 44.0, 40.0, 10.0; wb, wc = 40.0, 38.0
    srcs = sorted(lab.source.unique(), key=lambda s: list(SRC_NAME).index(s)); SC = {"ABoVE_ds2369": "#1A3F6B", "CALM": "#C9671D", "ALLENA_PANGAEA_973813": "#5F8FB0"}
    # a: ECDF of observed ALT by region and source
    ax = C.ax_mm(fig, xs[0], y0, w, h); C.clean_axes(ax)
    for r in REGIONS:
        for s in srcs:
            d = lab[(lab.region == r) & (lab.source == s)]
            if len(d) == 0: continue
            v = np.sort(d.alt_cm.values); ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=SC[s], lw=C.LW_MAIN, ls={"Alaska": "-", "Lena": C.DASH, "Canada": C.DOT}[r], zorder=3)
            meta["a"][f"{r}_{s}"] = dict(n=int(len(d)), median=round(float(np.median(v)), 1), q10=round(float(np.percentile(v, 10)), 1), q90=round(float(np.percentile(v, 90)), 1))
    ax.set_xlim(0, 200); ax.set_ylim(0, 1); ax.set_xlabel("Observed ALT (cm)"); ax.set_ylabel("Cumulative share of site-years"); ax.set_yticks([0, 0.5, 1])
    hs = [Line2D([], [], color=SC[s], lw=C.LW_MAIN) for s in srcs] + [Line2D([], [], color=C.DARK, lw=C.LW_MAIN, ls=ls) for ls in ("-", C.DASH, C.DOT)]
    ax.legend(hs, [SRC_NAME[s] for s in srcs] + [RN[r] for r in REGIONS], loc="lower right", fontsize=C.FS_SMALL, frameon=False, handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)
    C.letter(fig, 2.0, 3.0, "a"); C.text_mm(fig, 7.0, 3.3, "Observed ALT by region and source", fontsize=C.FS, ha="left", va="top")
    # b: observation day of year by region (histograms, share)
    ax = C.ax_mm(fig, xs[1], y0, wb, h); C.clean_axes(ax); bins = np.arange(150, 320, 7)
    for r in REGIONS:
        d = lab[lab.region == r].obs_doy.dropna().values; hcount, _ = np.histogram(d, bins=bins); share = hcount / hcount.sum()
        ax.step(bins[:-1], share, where="post", color=RC[r], lw=C.LW_MAIN, zorder=3)
        meta["b"][r] = dict(n=int(len(d)), median_doy=round(float(np.median(d)), 1), q10=round(float(np.percentile(d, 10)), 1), q90=round(float(np.percentile(d, 90)), 1))
    for doy, name in ((213, "1 Aug"), (244, "1 Sep"), (274, "1 Oct")):
        ax.axvline(doy, color=C.ZERO, lw=C.LW_REF, zorder=1); ax.text(doy, ax.get_ylim()[1] * 0.98, name, ha="center", va="top", fontsize=C.FS_SMALL, color=C.AUX)
    ax.set_xlim(150, 320); ax.set_xlabel("Observation day of year"); ax.set_ylabel("Share of site-years per week")
    ax.legend([Line2D([], [], color=RC[r], lw=C.LW_MAIN) for r in REGIONS], [RN[r] for r in REGIONS], loc="center right", fontsize=C.FS_SMALL, frameon=False, borderaxespad=0.2, labelspacing=0.3)
    C.letter(fig, xs[1] - 8.0, 3.0, "b"); C.text_mm(fig, xs[1] - 3.0, 3.3, "Timing of the observations", fontsize=C.FS, ha="left", va="top")
    # c: block-CV mean residual (physics-residual model) by region and source with 95 % block-bootstrap interval
    ax = C.ax_mm(fig, xs[2], y0, wc, h); C.clean_axes(ax); rows = []
    rng = np.random.default_rng(0)
    for r in REGIONS:
        p = preds(r, "R1").merge(lab[["loc_id", "year", "source"]], on=["loc_id", "year"], how="left")
        for s in srcs:
            q = p[p.source == s]
            if len(q) < 20: continue
            e = (q.pred - q.y).values; blocks = q.block.values; ub = np.unique(blocks); bl = [e[blocks == b] for b in ub]
            bs = [np.mean(np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])) for _ in range(1000)]
            rows.append((r, s, float(e.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(q))))
    for i, (r, s, m, lo, hi, n) in enumerate(rows):
        y = len(rows) - 1 - i; ax.plot([lo, hi], [y, y], color=SC[s], lw=C.LW_AUX); ax.plot(m, y, marker=C.REGION[r][2], ms=C.MS, color=SC[s], ls="none")
        ax.text(ax.get_xlim()[1], y, f"n = {n:,}", ha="right", va="center", fontsize=C.FS_SMALL, color=C.AUX)
        meta["c"][f"{r}_{s}"] = dict(mean_error=round(m, 2), lo=round(lo, 2), hi=round(hi, 2), n=n)
    ax.axvline(0, color=C.ZERO, lw=C.LW_REF); ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{RN[r]} · {SHORT_SRC[s]}" for r, s, *_ in rows][::-1]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Mean error, predicted − observed (cm)")
    xl = ax.get_xlim(); ax.set_xlim(xl[0], xl[1] + 0.35 * (xl[1] - xl[0]))
    for t in ax.texts: t.set_x(ax.get_xlim()[1])
    C.letter(fig, xs[2] - 26.0, 3.0, "c"); C.text_mm(fig, xs[2] - 21.0, 3.3, "Mean error by source, physics + residual ML (block CV)", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P1_2", message="Labels differ by source in level and timing; the residual by source is the label-bias check", canvas_mm=[W, H]))
    finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


def P1_3():
    """Representativeness: sites against the mapped domain in TDD x elevation space (map cells as grey density)."""
    stem = "P1_3_climate_space"; t0 = time.time(); lab = labels(); W, H = 180.0, 69.0; fig = C.figure(W, H); meta = {}
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 44.0, 44.0, 12.0
    for j, r in enumerate(REGIONS):
        key = KEY[r]; grid = pd.read_csv(PROC / f"map_{key}" / f"{key}_grid_x25_v1.csv.gz", usecols=["cell_id", "dem_elev"], dtype={"cell_id": str}).set_index("cell_id")
        tdd = pd.concat([pd.read_csv(MAPS / r / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "e5_tdd"]).set_index("cell_id").e5_tdd for y in (2013, 2016, 2019, 2022)], axis=1).mean(axis=1)
        cells = pd.DataFrame({"tdd": tdd, "elev": grid.dem_elev.reindex(tdd.index)}).dropna()
        d = lab[lab.region == r].copy(); d["cell_id"] = cell_ids(d.lat.values, d.lon.values)
        py = {y: pd.read_csv(MAPS / r / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "e5_tdd"]).set_index("cell_id").e5_tdd for y in sorted(d.year.unique()) if YEARS[0] <= y <= YEARS[-1]}
        d["tdd"] = [py[y].get(c, np.nan) if y in py else np.nan for c, y in zip(d.cell_id, d.year)]; d["elev"] = grid.dem_elev.reindex(d.cell_id).values; d = d.dropna(subset=["tdd", "elev"])
        ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax)
        ex = np.linspace(0, min(1500, np.percentile(cells.elev, 99.5)), 46); tx = np.linspace(cells.tdd.min(), cells.tdd.max(), 46)
        Hc, _, _ = np.histogram2d(cells.tdd.values, cells.elev.values, bins=[tx, ex])
        ax.pcolormesh(tx, ex, np.where(Hc.T > 0, Hc.T, np.nan), cmap=LinearSegmentedColormap.from_list("g", ["#F2F2F2", "#8C8C8C"]), norm=LogNorm(vmin=1, vmax=Hc.max()), rasterized=True, zorder=1)
        ax.scatter(d.tdd, d.elev, s=1.3 ** 2, facecolor=RC[r], edgecolor="none", alpha=0.6, zorder=3, rasterized=True)
        ax.set_xlim(tx[0], tx[-1]); ax.set_ylim(0, ex[-1]); ax.set_xlabel("TDD to 1 September of the year (°C d)")
        if j == 0: ax.set_ylabel("Elevation (m)")
        inside = float(((d.tdd >= np.percentile(cells.tdd, 1)) & (d.tdd <= np.percentile(cells.tdd, 99)) & (d.elev <= np.percentile(cells.elev, 99))).mean())
        ax.text(0.97, 0.96, f"{len(d):,} site-years\n{len(cells):,} map cells", transform=ax.transAxes, ha="right", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
        C.text_mm(fig, xs[j], y0 - 1.2, RN[r], fontsize=C.FS, ha="left", va="bottom")
        meta[r] = dict(site_years=int(len(d)), map_cells=int(len(cells)), tdd_cells_p1_p99=[round(float(np.percentile(cells.tdd, 1)), 0), round(float(np.percentile(cells.tdd, 99)), 0)],
                       tdd_sites_p1_p99=[round(float(np.percentile(d.tdd, 1)), 0), round(float(np.percentile(d.tdd, 99)), 0)], elev_sites_p99=round(float(np.percentile(d.elev, 99)), 0),
                       elev_cells_p99=round(float(np.percentile(cells.elev, 99)), 0), share_sites_inside_cell_range=round(inside, 3))
    C.letter(fig, 2.0, 3.0, "a"); C.text_mm(fig, 7.0, 3.3, "Observed site-years (colour) against all mapped cells (grey density; TDD mean of 2013, 2016, 2019, 2022) in climate–terrain space", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P1_3", message="The observation network covers the climate range of the maps but not the high terrain; basis of the area of applicability", canvas_mm=[W, H]))
    finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


# ================================================================ Fig 2 group
def P2_1():
    """Equation strip with real maps: Stefan term + learned residual = ALT map (Alaska 2014), value histograms below; no boxes, no arrows."""
    stem = "P2_1_equation_strip"; t0 = time.time(); M = MP.Med(); R = Region("alaska", step=4)
    p = pd.read_csv(MAPS / "Alaska/pred_2014.csv.gz", dtype={"cell_id": str}).set_index("cell_id"); mean = pd.read_csv(MAPS / "Alaska/mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"]).set_index("cell_id")
    v_p1 = R.values(p.p1); v_r1 = R.values(p.r1); v_res = v_r1 - v_p1; aoa = np.nan_to_num(R.values(mean.aoa_in.astype(float)), nan=0.0)
    W = 180.0; w = 46.0; h = w * R.asp; y0 = 11.5; H = y0 + h + 30.0; fig = C.figure(W, H)
    lo, hi = 20.0, 100.0; nA = Normalize(lo, hi); nR = Normalize(-25, 25)
    cols = [9.5, 69.0, 128.5]; items = [("Stefan term", v_p1, MP.CM_ALT, nA), ("Learned residual, weight 0.5", v_res, MP.CM_ANOM, nR), ("ALT map 2014", v_r1, MP.CM_ALT, nA)]
    meta = {}
    for j, (name, v, cm, nm) in enumerate(items):
        rgb, _ = MP.render(R.base, v, cm, nm, aoa=aoa, smooth=False, w_shade=0.0)
        ax = C.ax_mm(fig, cols[j], y0, w, h); R.frame(ax, rgb, M, grat=False, scale=(j == 0)); del rgb
        C.text_mm(fig, cols[j], y0 - 1.3, name, fontsize=C.FS, ha="left", va="bottom")
        axh = C.ax_mm(fig, cols[j] + 4.0, y0 + h + 6.0, w - 8.0, 12.0); C.clean_axes(axh)
        f = np.isfinite(v); vals = v[f]; edges = np.linspace(nm.vmin, nm.vmax, 41); cnt, _ = np.histogram(np.clip(vals, nm.vmin, nm.vmax), bins=edges)
        for a, b, c_ in zip(edges[:-1], edges[1:], cnt): axh.add_patch(plt.Rectangle((a, 0), b - a, c_ / cnt.max(), facecolor=cm(nm(0.5 * (a + b))), edgecolor="none"))
        axh.set_xlim(nm.vmin, nm.vmax); axh.set_ylim(0, 1.05); axh.set_yticks([]); axh.spines["left"].set_visible(False)
        axh.set_xlabel("cm"); axh.xaxis.set_major_locator(mticker.MaxNLocator(5))
        axh.text(0.0, 1.12, f"median {np.median(vals):.0f} cm, 5–95 % {np.percentile(vals, 5):.0f} to {np.percentile(vals, 95):.0f} cm".replace("-", "−"), transform=axh.transAxes, ha="left", va="bottom", fontsize=C.FS_SMALL, color=C.AUX)
        meta[name] = dict(median=round(float(np.median(vals)), 2), p5=round(float(np.percentile(vals, 5)), 2), p95=round(float(np.percentile(vals, 95)), 2), n_cells=int(f.sum()))
    for j, glyph in ((0, "+"), (1, "=")):
        C.text_mm(fig, cols[j] + w + (cols[j + 1] - cols[j] - w) / 2, y0 + h / 2, glyph, fontsize=14, ha="center", va="center", color=C.TEXT)
    C.letter(fig, 2.0, 3.0, "a"); C.text_mm(fig, 7.0, 3.3, "Alaska, 2014: the map is the Stefan term of the year plus a learned residual (lighter colours: outside the area of applicability)", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P2_1", message="One line explains the model: physics term plus a learned, spatially structured correction", canvas_mm=[W, round(H, 1)],
                     inputs="data/processed/v5/maps/Alaska/pred_2014.csv.gz (p1, r1); residual = r1 - p1"))
    finish(fig, stem, meta, ignore=["km"]); print(f"  {stem} {time.time() - t0:.0f} s")


def P2_2():
    """One long-record site per region: observed ALT by year, Stefan (recalibrated) and physics-residual out-of-fold predictions; residual drawn as bars."""
    stem = "P2_2_site_example"; t0 = time.time(); lab = labels(); W, H = 180.0, 66.0; fig = C.figure(W, H); meta = {}
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 46.0, 36.0, 16.0
    for j, r in enumerate(REGIONS):
        P1 = preds(r, "P1").rename(columns={"pred": "p1"}); R1 = preds(r, "R1").rename(columns={"pred": "r1"})
        d = P1.merge(R1[["cell_year_id", "r1"]], on="cell_year_id")
        g = d.groupby("loc_id").year.nunique(); n_obs = lab[lab.region == r].groupby("loc_id").size()
        best = g[g == g.max()].index; best = n_obs.reindex(best).idxmax() if len(best) > 1 else best[0]
        s = d[d.loc_id == best].sort_values("year"); site = lab[(lab.region == r) & (lab.loc_id == best)].iloc[0]
        ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax)
        ax.plot(s.year, s.p1, color=MC["P1"], lw=C.LW_MAIN, marker="s", ms=C.MS * 0.85, zorder=3)
        ax.plot(s.year, s.r1, color=MC["R1"], lw=C.LW_MAIN, marker="o", ms=C.MS * 0.85, zorder=4)
        for yy, a, b in zip(s.year, s.p1, s.r1): ax.plot([yy, yy], [a, b], color=MC["R1"], lw=0.6, alpha=0.6, zorder=2)
        ax.plot(s.year, s.y, ls="none", marker="o", ms=C.MS, mfc="white", mec=C.TEXT, mew=0.7, zorder=5)
        ax.set_xlim(2010.4, 2024.6); ax.set_xticks([2012, 2016, 2020, 2024]); ax.set_xlabel("Year")
        if j == 0: ax.set_ylabel("ALT (cm)")
        e1, e2 = rmse(s.p1 - s.y), rmse(s.r1 - s.y)
        C.text_mm(fig, xs[j], y0 - 4.0, f"{RN[r]}, {abs(site.lat):.2f}°{'N' if site.lat > 0 else 'S'} {abs(site.lon):.2f}°{'E' if site.lon > 0 else 'W'}", fontsize=C.FS, ha="left", va="bottom")
        C.text_mm(fig, xs[j], y0 - 1.2, f"{len(s)} years; RMSE {e1:.1f} → {e2:.1f} cm (Stefan → physics + residual ML)".replace(" (Stefan → physics + residual ML)", ""), fontsize=C.FS_SMALL, ha="left", va="bottom", color=C.AUX)
        meta[r] = dict(loc_id=int(best), lat=round(float(site.lat), 4), lon=round(float(site.lon), 4), years=[int(v) for v in s.year], observed=[round(float(v), 1) for v in s.y],
                       stefan=[round(float(v), 1) for v in s.p1], physics_residual=[round(float(v), 1) for v in s.r1], rmse_stefan=round(e1, 2), rmse_r1=round(e2, 2))
    hs = [Line2D([], [], ls="none", marker="o", ms=C.MS, mfc="white", mec=C.TEXT, mew=0.7), Line2D([], [], color=MC["P1"], lw=C.LW_MAIN, marker="s", ms=C.MS * 0.85),
          Line2D([], [], color=MC["R1"], lw=C.LW_MAIN, marker="o", ms=C.MS * 0.85), Line2D([], [], color=MC["R1"], lw=0.6, alpha=0.6)]
    fig.legend(hs, ["Observed", MN["P1"] + ", out of fold", MN["R1"] + ", out of fold", "Learned residual"], loc="upper left", bbox_to_anchor=(xs[0] / W, 1 - 6.6 / H), ncol=4, frameon=False,
               fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    C.letter(fig, 2.0, 3.0, "a"); C.text_mm(fig, 7.0, 3.3, "The site with the most observed years in each region (spatial-block CV, the site's block held out); RMSE: Stefan → physics + residual ML", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P2_2", message="At one site: the Stefan term follows the climate of each year; the residual learned from other blocks shifts the level, which helps at some sites and not at others", canvas_mm=[W, H]))
    finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


def P2_3():
    """Method spectrum: four methods from physics to data, Lena Delta observed-vs-predicted thumbnails (block CV) and RMSE in / out of the region."""
    stem = "P2_3_method_spectrum"; t0 = time.time(); val = rmse_table(); W, H = 180.0, 70.0; fig = C.figure(W, H); meta = {}
    items = [("P0", "Stefan, source coefficient", "Nothing learned here.\nCoefficient from another\nregion", "P0"),
             ("P1", "Stefan, recalibrated", "One coefficient fitted\nto the region's labels", "P1"),
             ("R1", "Physics + residual ML", "Stefan term, plus a\nlearned correction from\nterrain, soil and climate", "R1"),
             ("D0", "Direct ML", "ALT learned from the\ninputs alone; physics\nonly in the training data", "D0")]
    x0 = 9.0; w = 36.0; gap = (W - x0 - 4 * w - 4.0) / 3; y_map = 20.0; r = "Lena"
    for j, (m, name, desc, mk) in enumerate(items):
        x = x0 + j * (w + gap); C.text_mm(fig, x, 7.0, name, fontsize=C.FS, ha="left", va="top", color=MC[m], fontweight="bold")
        C.text_mm(fig, x, 10.5, desc, fontsize=C.FS_SMALL, ha="left", va="top", color=C.AUX, linespacing=1.15)
        d = preds(r, m); ax = C.ax_mm(fig, x, y_map + 2.0, w - 8.0, w - 8.0); C.clean_axes(ax); top = 120.0
        ax.scatter(d.y, d.pred, s=1.6 ** 2, facecolor=MC[m], edgecolor="none", alpha=0.45, rasterized=True, zorder=3)
        ax.plot([0, top], [0, top], color=C.ZERO, lw=C.LW_REF); ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal"); ax.set_xticks([0, 40, 80, 120]); ax.set_yticks([0, 40, 80, 120])
        ax.set_xlabel("Observed ALT (cm)")
        if j == 0: ax.set_ylabel("Predicted (cm)")
        e_in = val(r, "block", m); e_out = val(r, "region", m)
        ax.text(0.04, 0.96, f"RMSE {e_in:.1f} cm", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
        C.text_mm(fig, x, y_map + w - 8.0 + 11.0, f"In the region (block CV): {e_in:.1f} cm\nRegion withheld: {e_out:.1f} cm", fontsize=C.FS_SMALL, ha="left", va="top", color=C.TEXT, linespacing=1.15)
        meta[m] = dict(rmse_block=round(e_in, 2), rmse_region=round(e_out, 2), n=int(len(d)))
        if j < 3: C.text_mm(fig, x + w + gap / 2 - 1.5, 8.0, "→", fontsize=9, ha="center", va="top", color=C.AUX)
    C.letter(fig, 2.0, 3.0, "a"); C.text_mm(fig, 7.0, 3.3, "From physics to data: what each method takes from the labels (Lena Delta, spatial-block cross-validation, all site-years)", fontsize=C.FS, ha="left", va="top")
    C.text_mm(fig, 2.0, H - 2.5, "Region withheld: trained on Alaska and Canada (NW) only; for the source-coefficient Stefan model both numbers use the coefficient of the other regions.", fontsize=C.FS_SMALL, ha="left", va="bottom", color=C.AUX)
    meta.update(dict(proposal="P2_3", message="The methods form a spectrum; the physics term is what survives the region hold-out", canvas_mm=[W, H]))
    finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


# ================================================================ Fig 3 group
def P3_1():
    """RMSE against the validation design, one panel per region, three methods as lines (random to region hold-out)."""
    stem = "P3_1_design_ladder"; t0 = time.time(); val = rmse_table(); W, H = 180.0, 64.0; fig = C.figure(W, H); meta = {}
    schemes = ["random", "block", "year", "block_year", "region"]; lbl = ["Random", "Spatial\nblock", "Year", "Block-\nyear", "Region\nheld out"]
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 46.0, 36.0, 15.0
    for j, r in enumerate(REGIONS):
        ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax)
        for m, mk, ls, filled in (("P1", "s", "-", True), ("R1", "o", "-", True), ("D0", "o", C.DASH, False)):
            v = [val(r, s, "P0" if (s == "region" and m == "P1") else m) for s in schemes]; c = MC[m]
            ax.plot(range(5), v, color=c, lw=C.LW_MAIN, ls=ls, marker=mk, ms=C.MS, mfc=c if filled else "white", mec=c, mew=0.7, zorder=3 if m == "R1" else 2)
            meta[f"{r}_{m}"] = [round(x, 2) for x in v]
        ax.set_xticks(range(5)); ax.set_xticklabels(lbl); ax.set_xlim(-0.4, 4.4); ax.tick_params(axis="x", labelsize=C.FS_SMALL)
        if j == 0: ax.set_ylabel("RMSE (cm)")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        C.text_mm(fig, xs[j], y0 - 1.2, RN[r], fontsize=C.FS, ha="left", va="bottom")
    hs = [Line2D([], [], color=MC["P1"], lw=C.LW_MAIN, marker="s", ms=C.MS), Line2D([], [], color=MC["R1"], lw=C.LW_MAIN, marker="o", ms=C.MS),
          Line2D([], [], color=MC["D0"], lw=C.LW_MAIN, ls=C.DASH, marker="o", ms=C.MS, mfc="white", mew=0.7)]
    fig.legend(hs, [MN["P1"] + " (source coefficient in the region hold-out)", MN["R1"], MN["D0"]], loc="upper left", bbox_to_anchor=(xs[0] / W, 1 - 6.8 / H), ncol=3, frameon=False,
               fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    C.letter(fig, 2.0, 2.4, "a"); C.text_mm(fig, 7.0, 2.7, "RMSE by validation design, from the least to the most demanding", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P3_1", message="Random splits flatter the learners; the ranking changes with the design, the physics term degrades least", canvas_mm=[W, H]))
    finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


def P3_2():
    """Where the learner helps: each 0.5-degree block coloured by RMSE(physics-residual) - RMSE(recalibrated Stefan) in spatial-block CV, three regional maps."""
    stem = "P3_2_block_gain_maps"; t0 = time.time(); lab = labels(); M = MP.Med(); R = {r: Region(KEY[r]) for r in REGIONS}; rgbs = {r: R[r].land_rgb() for r in REGIONS}
    W = 180.0; w = 50.0; y0 = 6.5; hmax = max(w * R[r].asp for r in REGIONS); H = y0 + hmax + 16.0; fig = C.figure(W, H)
    axes, _ = three_maps(fig, y0, w, R, M, rgbs, [RN[r] for r in REGIONS], ["a", "b", "c"]); norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10); meta = {}
    for j, r in enumerate(REGIONS):
        P1 = preds(r, "P1").rename(columns={"pred": "p1"}); R1 = preds(r, "R1").rename(columns={"pred": "r1"}); d = P1.merge(R1[["cell_year_id", "r1"]], on="cell_year_id")
        d = d.merge(lab[["loc_id", "year", "lat", "lon"]], on=["loc_id", "year"], how="left")
        g = d.groupby("block").agg(n=("y", "size"), lat=("lat", "mean"), lon=("lon", "mean"), e1=("p1", lambda s: 0), )
        rows = []
        for b, q in d.groupby("block"):
            rows.append(dict(block=b, n=len(q), lat0=np.floor(q.lat.mean() / 0.5) * 0.5, lon0=np.floor(q.lon.mean() / 0.5) * 0.5, d=rmse(q.r1 - q.y) - rmse(q.p1 - q.y)))
        g = pd.DataFrame(rows); meta[r] = dict(blocks=int(len(g)), blocks_gain=int((g.d < 0).sum()), blocks_loss=int((g.d > 0).sum()), median_delta=round(float(g.d.median()), 2),
                                               site_year_weighted_delta=round(float(np.average(g.d, weights=g.n)), 2))
        for _, b in g.iterrows():
            lon = np.array([b.lon0, b.lon0 + 0.5, b.lon0 + 0.5, b.lon0]); lat = np.array([b.lat0, b.lat0, b.lat0 + 0.5, b.lat0 + 0.5])
            x, y = R[r].tf.transform(lon, lat); axes[j].add_patch(Polygon(np.c_[x, y], closed=True, facecolor=CM_DIV(norm(b.d)), edgecolor=C.TEXT, lw=0.3, zorder=6))
        axes[j].text(0.97, 0.03, f"{meta[r]['blocks']} blocks; gain in {meta[r]['blocks_gain']}, loss in {meta[r]['blocks_loss']}", transform=axes[j].transAxes, ha="right", va="bottom",
                     fontsize=C.FS_SMALL, color=C.TEXT, zorder=9, bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    import matplotlib as mpl
    cax = C.ax_mm(fig, 65.0, y0 + hmax + 5.0, 50.0, 1.8); cb = mpl.colorbar.ColorbarBase(cax, cmap=CM_DIV, norm=norm, orientation="horizontal", ticks=[-10, -5, 0, 5, 10], extend="both")
    cb.outline.set_linewidth(C.LW_AX); cax.tick_params(labelsize=C.FS_TICK, length=1.5, width=0.5, pad=1.0); cax.set_xlabel("RMSE change of the physics-residual model against recalibrated Stefan, per block (cm)", fontsize=C.FS_SMALL, labelpad=1.5)
    meta.update(dict(proposal="P3_2", message="The gain is spatially structured: most blocks improve in the Lena Delta and Canada, Alaska is mixed", canvas_mm=[W, round(H, 1)],
                     definition="per 0.5 degree block (block CV fold assignment), RMSE over its site-years, seed-mean out-of-fold predictions"))
    finish(fig, stem, meta, ignore=["km", "blocks"]); print(f"  {stem} {time.time() - t0:.0f} s")


def P3_3():
    """Mean error by decile of observed ALT (block CV), three methods, one panel per region: compression of the extremes."""
    stem = "P3_3_bias_by_decile"; t0 = time.time(); W, H = 180.0, 64.0; fig = C.figure(W, H); meta = {}
    xs = [12.0, 70.0, 128.0]; w, h, y0 = 46.0, 36.0, 15.0
    for j, r in enumerate(REGIONS):
        ax = C.ax_mm(fig, xs[j], y0, w, h); C.clean_axes(ax); P = {m: preds(r, m) for m in ("P1", "R1", "D0")}
        edges = np.percentile(P["P1"].y, np.linspace(0, 100, 11)); mids = 0.5 * (edges[:-1] + edges[1:])
        for m, mk, ls, filled in (("P1", "s", "-", True), ("R1", "o", "-", True), ("D0", "o", C.DASH, False)):
            d = P[m]; k = np.clip(np.searchsorted(edges, d.y.values, side="right") - 1, 0, 9); e = d.pred.values - d.y.values
            mb = np.array([e[k == i].mean() for i in range(10)]); c = MC[m]
            ax.plot(mids, mb, color=c, lw=C.LW_MAIN, ls=ls, marker=mk, ms=C.MS * 0.85, mfc=c if filled else "white", mec=c, mew=0.7, zorder=3 if m == "R1" else 2)
            meta[f"{r}_{m}"] = dict(decile_mid=[round(float(v), 1) for v in mids], mean_error=[round(float(v), 2) for v in mb])
        ax.axhline(0, color=C.ZERO, lw=C.LW_REF); ax.set_xlabel("Observed ALT, decile midpoint (cm)")
        if j == 0: ax.set_ylabel("Mean error, predicted − observed (cm)")
        C.text_mm(fig, xs[j], y0 - 1.2, f"{RN[r]}, n = {len(P['P1']):,}", fontsize=C.FS, ha="left", va="bottom")
    hs = [Line2D([], [], color=MC["P1"], lw=C.LW_MAIN, marker="s", ms=C.MS), Line2D([], [], color=MC["R1"], lw=C.LW_MAIN, marker="o", ms=C.MS),
          Line2D([], [], color=MC["D0"], lw=C.LW_MAIN, ls=C.DASH, marker="o", ms=C.MS, mfc="white", mew=0.7)]
    fig.legend(hs, [MN["P1"], MN["R1"], MN["D0"]], loc="upper left", bbox_to_anchor=(xs[0] / W, 1 - 6.8 / H), ncol=3, frameon=False, fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    C.letter(fig, 2.0, 2.4, "a"); C.text_mm(fig, 7.0, 2.7, "Error by decile of the observed ALT, spatial-block cross-validation", fontsize=C.FS, ha="left", va="top")
    meta.update(dict(proposal="P3_3", message="All methods over-predict shallow and under-predict deep sites; the residual learner reduces this compression most where labels are dense", canvas_mm=[W, H]))
    finish(fig, stem, meta); print(f"  {stem} {time.time() - t0:.0f} s")


REG = {"P1_1": P1_1, "P1_2": P1_2, "P1_3": P1_3, "P2_1": P2_1, "P2_2": P2_2, "P2_3": P2_3, "P3_1": P3_1, "P3_2": P3_2, "P3_3": P3_3}

if __name__ == "__main__":
    C.setup(); sel = sys.argv[1:] or list(REG)
    for k in sel:
        try:
            REG[k]()
        except Exception:
            print(f"[FAIL] {k}"); traceback.print_exc()
        gc.collect()
