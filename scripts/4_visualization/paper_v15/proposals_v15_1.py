"""Figure design proposals, round 3 (v15, 2026-10-21). Real data, proposal-grade QA.
P3_G  error against the distance to the nearest training site (spatial-block CV; Meyer & Pebesma 2022 / Ploton 2020 logic), three regions x three methods,
      with the region hold-out error as the far end.
P7_E  ALT anomaly of selected years in Alaska (0.1 deg boxes) with the regional TDD anomaly printed (Wei 2026 Fig. 4 style small multiples).
P1_B  data overview alternative: site-years per region and year as a heat matrix, and the distribution of sites per 1 km cell."""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import TwoSlopeNorm, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = C.ROOT; P = ROOT / "data/processed/v5/runs/core_v6/preds"; OUT = ROOT / "outputs/figures/paper/v15_proposals"; OUT.mkdir(parents=True, exist_ok=True)
REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}; RC = {r: C.REGION[r][1] for r in REGIONS}; MN = C.METHOD_NAME; MC = C.METHOD
PAT = {"P0": "P0__-__lam-1", "P1": "P1__-__lam-1", "R1": "R1__catboost__lam0.5", "D0": "D0__catboost__lam-1"}
rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))


def labels():
    l = pd.read_csv(ROOT / "data/processed/v5/labels_cell_year_v5.csv"); return l[l.in_forcing == 1]


def finish(fig, stem, meta):
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(stem, rep); paths = C.export(fig, stem, OUT); plt.close(fig)
    meta.update(dict(outputs=C.file_info(paths), qa=rep)); C.write_json(OUT / f"{stem}.json", meta)
    for p in paths: print("  ", p.name, p.stat().st_size)


def xy_km(lat, lon, lat0):
    return np.column_stack([(lon - 0.0) * 111.32 * np.cos(np.radians(lat0)), lat * 110.57])


# ================================================================ P3_G distance to the nearest training site
def p3_g():
    lab = labels(); key = lab.set_index(["loc_id", "year"])[["alt_cm", "lat", "lon", "block"]]
    bins = [0, 5, 25, 100, 1e9]; blab = ["< 5", "5–25", "25–100", "> 100"]
    W, H = 180.0, 66.0; fig = C.figure(W, H); meta = {}
    for i, r in enumerate(REGIONS):
        d = {}
        for m in ("P1", "R1", "D0"):
            q = pd.read_csv(P / f"{r}__block__year_obs__F8TS__{PAT[m]}.csv.gz", usecols=["loc_id", "year", "fold", "pred", "seed"])
            q = q.groupby(["loc_id", "year", "fold"]).pred.mean().reset_index().set_index(["loc_id", "year"]).join(key, how="inner"); d[m] = q
        base = d["P1"]; lat0 = float(base.lat.mean()); xy = xy_km(base.lat.values, base.lon.values, lat0)
        dist = np.full(len(base), np.nan)
        sites = base.reset_index().drop_duplicates("loc_id"); sxy = xy_km(sites.lat.values, sites.lon.values, lat0); sfold = sites.fold.values
        for f in np.unique(base.fold.values):
            tr = cKDTree(sxy[sfold != f]); te = base.fold.values == f; dist[te], _ = tr.query(xy[te], k=1)
        base = base.assign(dist=dist); cat = pd.cut(base.dist, bins, labels=blab, right=False)
        # region hold-out as the far end
        reg = {m: pd.read_csv(P / f"ALL__region__year_obs__F8TS__{PAT[m]}.csv.gz", usecols=["loc_id", "year", "pred", "region"]) for m in ("P0", "R1", "D0")}
        ax = C.ax_mm(fig, 16 + i * 57, 11, 46, 40); C.clean_axes(ax); vals = {}
        for m, mk, ls, filled in (("P1", "s", "-", True), ("R1", "o", "-", True), ("D0", "o", C.DASH, False)):
            pr = d[m].pred.reindex(base.index).values; e = pr - base.alt_cm.values
            ys = [rmse(e[(cat == b).values]) if (cat == b).sum() >= 30 else np.nan for b in blab]
            ns = [int((cat == b).sum()) for b in blab]
            rr = reg["P0" if m == "P1" else m]; rr = rr[rr.region == r].groupby(["loc_id", "year"]).pred.mean().to_frame().join(key, how="inner"); yreg = rmse(rr.pred - rr.alt_cm)
            xs = list(range(len(blab))) + [len(blab) + 0.6]; yy = ys + [yreg]
            ax.plot(xs, yy, color=MC[m], ls=ls, lw=C.LW_MAIN, marker=mk, ms=C.MS, mfc=(MC[m] if filled else "white"), mec=MC[m], mew=0.7)
            vals[m] = dict(bins=blab, rmse=[None if np.isnan(v) else round(v, 2) for v in ys], n=ns, region_holdout=round(yreg, 2))
        ax.set_xticks(list(range(len(blab))) + [len(blab) + 0.6]); ax.set_xticklabels(blab + ["Region\nheld out"]); ax.axvline(len(blab) - 0.2, color="#DDDDDD", lw=0.5)
        if i == 0: ax.set_ylabel("RMSE (cm)")
        if i == 1: ax.set_xlabel("Distance from the test site to the nearest training site (km)")
        C.letter(fig, 16 + i * 57 - 11, 5.0, "abc"[i]); C.text_mm(fig, 16 + i * 57, 5.3, f"{RN[r]} (n per bin: " + ", ".join(f"{n:,}" for n in vals['R1']['n']) + ")", fontsize=C.FS_SMALL, ha="left", va="top")
        meta[r] = vals
    hs = [Line2D([], [], color=MC["P1"], marker="s", ms=C.MS), Line2D([], [], color=MC["R1"], marker="o", ms=C.MS), Line2D([], [], color=MC["D0"], ls=C.DASH, marker="o", ms=C.MS, mfc="white", mew=0.7)]
    fig.legend(hs, [MN["P1"] + " (source coeff. when the region is held out)", MN["R1"], MN["D0"]], loc="upper left", bbox_to_anchor=(16 / W, 1 - 0.8 / H), ncol=3, fontsize=C.FS_SMALL, frameon=False, handlelength=2.4, columnspacing=1.5, borderaxespad=0.0)
    finish(fig, "P3_G_error_vs_distance", dict(design="spatial-block OOF RMSE by distance from each test site to the nearest training site of its fold (bins with >= 30 site-years), plus the region hold-out RMSE", values=meta))


# ================================================================ P7_E selected-year anomaly multiples (Alaska, 0.1 deg boxes)
def p7_e():
    years = [2012, 2016, 2019, 2023]; tdd = pd.DataFrame(__import__("json").load(open(ROOT / "outputs/figures/paper/v14/source_data/Fig7_tdd_anomaly_by_year.json"))["Alaska"]).set_index("year")
    mean = pd.read_csv(ROOT / "data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "lat", "lon", "r1"]).set_index("cell_id")
    W, H = 180.0, 58.0; fig = C.figure(W, H); norm = TwoSlopeNorm(0, -15, 15); import cmcrameri.cm as cmc; meta = {}
    for k, y in enumerate(years):
        pr = pd.read_csv(ROOT / f"data/processed/v5/maps/Alaska/pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
        a = (pr - mean.r1).dropna(); la = mean.lat.reindex(a.index).values; lo = mean.lon.reindex(a.index).values
        bx = np.floor(lo / 0.1) * 0.1; by = np.floor(la / 0.1) * 0.1; g = pd.DataFrame(dict(x=bx, y=by, a=a.values)).groupby(["x", "y"]).a.mean().reset_index()
        xs = np.sort(g.x.unique()); ys = np.sort(g.y.unique()); Z = np.full((len(ys), len(xs)), np.nan); ix = {v: i for i, v in enumerate(xs)}; iy = {v: i for i, v in enumerate(ys)}
        for x, yy_, v in zip(g.x, g.y, g.a): Z[iy[yy_], ix[x]] = v
        ax = C.ax_mm(fig, 6 + k * 41.5, 10, 38, 36); ax.set_axis_off()
        ax.pcolormesh(xs, ys, np.ma.masked_invalid(Z), cmap=cmc.vik, norm=norm, shading="nearest"); ax.set_aspect(1 / np.cos(np.radians(65)))
        ax.text(0.02, 0.98, f"{y}", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS, color=C.TEXT)
        ax.text(0.02, 0.86, f"TDD {tdd.loc[y, 'tdd_anom']:+.0f} °C d\nmap {tdd.loc[y, 'r1_anom']:+.1f} cm".replace("-", "−"), transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.AUX)
        meta[str(y)] = dict(tdd_anom=round(float(tdd.loc[y, "tdd_anom"]), 1), map_anom=round(float(tdd.loc[y, "r1_anom"]), 2), boxes=int(np.isfinite(Z).sum()))
    import matplotlib as mpl
    cax = C.ax_mm(fig, 172, 14, 2.2, 28); cb = mpl.colorbar.ColorbarBase(cax, cmap=cmc.vik, norm=norm, ticks=[-15, 0, 15]); cb.outline.set_linewidth(C.LW_AX); cax.tick_params(labelsize=C.FS_TICK, length=1.5, width=0.5)
    C.text_mm(fig, 173, 47.0, "ALT anomaly\n(cm)", fontsize=C.FS_SMALL, ha="center", va="top")
    C.text_mm(fig, 6, 3.0, "Alaska, ALT anomaly of selected years relative to the 2011–2024 mean (0.1° boxes); regional TDD anomaly and mapped mean anomaly printed", fontsize=C.FS, ha="left", va="top")
    finish(fig, "P7_E_anomaly_selected_years", dict(design="four selected years (two cold, two warm) instead of 14 yearly panels; 0.1 deg box means of the 1 km anomaly", years=years, values=meta))


# ================================================================ P1_B coverage matrix + sites per cell
def p1_b():
    lab = labels(); CELL = 0.009
    ky = np.floor(lab.lat / CELL).astype(int); dx = CELL / np.cos(np.radians((ky + 0.5) * CELL)); kx = np.floor(lab.lon / dx).astype(int); lab = lab.assign(cell=ky.astype(str) + "_" + kx.astype(str))
    years = list(range(2010, 2025)); W, H = 180.0, 60.0; fig = C.figure(W, H); meta = {}
    ax = C.ax_mm(fig, 24, 10, 100, 24); ax.set_axis_off(); M = np.zeros((3, len(years)))
    for i, r in enumerate(REGIONS):
        c = lab[lab.region == r].groupby("year").size()
        for j, y in enumerate(years): M[i, j] = c.get(y, 0)
    import cmcrameri.cm as cmc
    ax.imshow(np.where(M > 0, M, np.nan), cmap=cmc.grayC_r, norm=LogNorm(vmin=1, vmax=M.max()), aspect="auto", interpolation="nearest")
    for i in range(3):
        for j in range(len(years)):
            v = int(M[i, j]);
            if v: ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=5.5, color=("white" if v > 1500 else C.TEXT))
    ax.set_xlim(-0.5, len(years) - 0.5); ax.set_ylim(2.5, -0.5)
    for j, y in enumerate(years): ax.text(j, 2.7, str(y)[2:], ha="center", va="top", fontsize=C.FS_SMALL)
    for i, r in enumerate(REGIONS): ax.text(-0.7, i, RN[r], ha="right", va="center", fontsize=C.FS_SMALL)
    C.letter(fig, 2, 3, "a"); C.text_mm(fig, 7, 3.3, "Site-years by region and observation year", fontsize=C.FS, ha="left", va="top")
    meta["matrix"] = {r: {str(y): int(M[i, j]) for j, y in enumerate(years)} for i, r in enumerate(REGIONS)}
    ax = C.ax_mm(fig, 140, 10, 34, 36); C.clean_axes(ax)
    for r in REGIONS:
        spc = lab[lab.region == r].groupby("cell").loc_id.nunique().values; bins = [1, 2, 3, 5, 10, 25, 100, 1000]
        h, _ = np.histogram(spc, bins); ax.plot(range(len(h)), h / h.sum(), color=RC[r], marker=C.REGION[r][2], ms=C.MS, lw=C.LW_MAIN, label=RN[r]); meta[f"sites_per_cell_{r}"] = [int(v) for v in h]
    ax.set_xticks(range(len(bins) - 1)); ax.set_xticklabels(["1", "2", "3–4", "5–9", "10–24", "25–99", "≥100"], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlabel("Sites per 1 km cell"); ax.set_ylabel("Share of cells")
    ax.legend(fontsize=C.FS_SMALL, frameon=False, handletextpad=0.4, loc="upper right", bbox_to_anchor=(1.02, 0.98))
    C.letter(fig, 128, 3, "b"); C.text_mm(fig, 133, 3.3, "Clustering of sites", fontsize=C.FS, ha="left", va="top")
    finish(fig, "P1_B_coverage_matrix", dict(design="coverage matrix (site-years by region and year) and distribution of sites per 1 km cell, as a compact replacement for Fig. 1b-c", values=meta))


if __name__ == "__main__":
    C.setup(); t0 = time.time()
    for name in (sys.argv[1:] or ["p3_g", "p7_e", "p1_b"]):
        try:
            print("==", name); globals()[name](); print(f"   {time.time() - t0:.0f} s")
        except Exception:
            traceback.print_exc()
