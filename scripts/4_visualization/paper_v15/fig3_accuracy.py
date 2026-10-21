"""Fig. 3 (v15, 2026-10-21): accuracy under spatial-block validation and across validation designs.
a  observed vs predicted on all site-years under spatial-block cross-validation (rows = regions, columns = recalibrated Stefan, physics-residual, direct ML);
   hexagonal log-density for n >= 2000, points otherwise; RMSE, r and n printed.
b  difference in RMSE from the recalibrated Stefan model (physics-residual and direct ML) under five validation designs, per region, with 95 % paired
   block-bootstrap intervals; in the region hold-out the recalibrated Stefan model equals the source-coefficient model.
Inputs: core_v6 preds (seed mean), labels, core_v6/deltas_vs_ref.csv. Run with POLAR_PALETTE=paper13."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = C.ROOT; RUNS = ROOT / "data" / "processed" / "v5" / "runs"; P = RUNS / "core_v6" / "preds"
STEM = "Fig3_accuracy"
REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}
PAT = {"P0": "P0__-__lam-1", "P1": "P1__-__lam-1", "R1": "R1__catboost__lam0.5", "D0": "D0__catboost__lam-1"}
MN = C.METHOD_NAME; MC = C.METHOD
SCHEMES = ["random", "block", "year", "block_year", "region"]; SCHEME_LB = ["Random", "Spatial block", "Year", "Block-year", "Region"]
A_METH = ["P1", "R1", "D0"]
_lab = None


def labels():
    global _lab
    if _lab is None:
        l = pd.read_csv(ROOT / "data/processed/v5/labels_cell_year_v5.csv"); l = l[l.in_forcing == 1]
        _lab = l.set_index(["loc_id", "year"])[["alt_cm", "region", "block"]]
    return _lab


def oof(region, m):
    d = pd.read_csv(P / f"{region}__block__year_obs__F8TS__{PAT[m]}.csv.gz", usecols=["loc_id", "year", "pred", "seed"])
    return d.groupby(["loc_id", "year"]).pred.mean().to_frame().join(labels(), how="inner")


rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))
ramp = lambda hex_: LinearSegmentedColormap.from_list("r_" + hex_, ["#FFFFFF", hex_])


def density(ax, x, y, col, top):
    if len(x) >= 2000:
        ax.hexbin(x, y, gridsize=28, extent=(0, top, 0, top), cmap=ramp(col), norm=LogNorm(vmin=1), mincnt=1, linewidths=0.1, edgecolors="none")
    else:
        ax.scatter(x, y, s=2.0 ** 2, facecolor=col, edgecolor="white", lw=0.2, alpha=0.85)
    ax.plot([0, top], [0, top], color=C.ZERO, lw=C.LW_REF, zorder=1)
    ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal")
    step = 40 if top > 120 else 20
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step)); ax.yaxis.set_major_locator(mticker.MultipleLocator(step))


def data_b():
    d = pd.read_csv(RUNS / "core_v6/deltas_vs_ref.csv")
    d = d[(d.ref == "P1") & (d.forcing == "year_obs") & (d.featset == "F8TS")]
    r1 = d[(d.method == "R1") & (d.learner == "catboost") & np.isclose(d.lam, 0.5)]; d0 = d[(d.method == "D0") & (d.learner == "catboost")]
    return {"R1": r1, "D0": d0}


def main():
    t0 = time.time(); C.setup()
    W = 180.0; s = 35.0; g = 6.0; x0 = 15.0; ya = 11.0
    yb = ya + 3 * s + 2 * g + 18.0; hb = 40.0; H = yb + hb + 11.0
    fig = C.figure(W, H); meta = {"a": {}, "b": {}}
    # ---------------- a: density matrix (left two thirds) ... use three columns of 35 mm, leaving the right part for b? no: b spans the full width below
    for i, r in enumerate(REGIONS):
        d = {m: oof(r, m) for m in A_METH}
        allv = np.concatenate([d["P1"].alt_cm.values] + [d[m].pred.values for m in d]); top = float(np.ceil(np.percentile(allv, 99.5) / 20) * 20)
        for j, m in enumerate(A_METH):
            ax = C.ax_mm(fig, x0 + j * (s + g), ya + i * (s + g), s, s); C.clean_axes(ax)
            q = d[m]; density(ax, q.alt_cm.values, q.pred.values, MC[m], top)
            e = rmse(q.pred - q.alt_cm); rr = float(np.corrcoef(q.alt_cm, q.pred)[0, 1])
            ax.text(0.04, 0.96, f"RMSE {e:.1f} cm\nr {rr:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
            meta["a"][f"{r}_{m}"] = dict(n=int(len(q)), rmse=round(e, 2), r=round(rr, 3), axis_top=top)
            if i == 0: C.text_mm(fig, x0 + j * (s + g), ya - 1.2, MN[m], fontsize=C.FS, ha="left", va="bottom")
            if j == 0: ax.set_ylabel(f"{RN[r]}, n = {len(q):,}\nPredicted ALT (cm)")
            if i == 2: ax.set_xlabel("Observed ALT (cm)")
            else: ax.set_xticklabels([])
    C.letter(fig, 2.0, 2.5, "a"); C.text_mm(fig, 7.0, 2.8, "Spatial-block cross-validation, all site-years", fontsize=C.FS, ha="left", va="top")
    # ---------------- b: delta vs recalibrated Stefan by design (one panel per region, rows = designs, two series)
    xb0 = 24.0; wb = (W - xb0 - 2 * 9.0 - 4.0) / 3.0
    db = data_b(); hs = []
    for j, r in enumerate(REGIONS):
        ax = C.ax_mm(fig, xb0 + j * (wb + 9.0), yb, wb, hb); C.clean_axes(ax, left=(j == 0))
        for k, sch in enumerate(SCHEMES): ax.axhline(k, color="#EDEDED", lw=0.4, zorder=0)
        for m, dy, filled in (("R1", -0.17, True), ("D0", 0.17, False)):
            q = db[m][db[m].region == r].set_index("scheme")
            for k, sch in enumerate(SCHEMES):
                if sch not in q.index: continue
                z = q.loc[sch]; y = k + dy; c = MC[m]
                ax.plot([z.lo, z.hi], [y, y], color=c, lw=C.LW_AUX, zorder=2)
                ax.plot(z.delta, y, marker="o", ms=C.MS, mfc=(c if filled else "white"), mec=c, mew=0.7, ls="none", zorder=3)
                meta["b"][f"{r}_{m}_{sch}"] = [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)]
        ax.axvline(0, color=C.ZERO, lw=C.LW_REF, zorder=1); ax.set_ylim(len(SCHEMES) - 0.5, -0.5); ax.set_yticks(range(len(SCHEMES)))
        ax.set_yticklabels(SCHEME_LB if j == 0 else []); ax.tick_params(axis="y", length=(C.TICK_LEN if j == 0 else 0))
        lo = min(min(db[m][db[m].region == r].lo.min(), 0) for m in db); hi = max(max(db[m][db[m].region == r].hi.max(), 0) for m in db)
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)); ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel("ΔRMSE vs recalibrated Stefan (cm)" if j == 1 else " ")
        C.text_mm(fig, xb0 + j * (wb + 9.0), yb - 1.2, RN[r], fontsize=C.FS, ha="left", va="bottom")
    hs = [Line2D([], [], ls="none", marker="o", ms=C.MS, mfc=MC["R1"], mec=MC["R1"], mew=0.7), Line2D([], [], ls="none", marker="o", ms=C.MS, mfc="white", mec=MC["D0"], mew=0.7)]
    fig.legend(hs, [MN["R1"], MN["D0"]], loc="upper left", bbox_to_anchor=(0.55, 1 - (yb - 9.0) / H), ncol=2, fontsize=C.FS_SMALL, frameon=False, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0)
    C.letter(fig, 2.0, yb - 9.0, "b"); C.text_mm(fig, 7.0, yb - 8.7, "Validation design", fontsize=C.FS, ha="left", va="top")
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN); plt.close(fig)
    src = dict(figure="Fig. 3", script="scripts/4_visualization/paper_v15/fig3_accuracy.py", version="v15 (2026-10-21)",
               inputs=dict(predictions="data/processed/v5/runs/core_v6/preds/<Region>__block__year_obs__F8TS__{P1,R1,D0}*.csv.gz (seed mean)",
                           labels="data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)", deltas="data/processed/v5/runs/core_v6/deltas_vs_ref.csv (ref P1, forcing year_obs, F8TS)"),
               a=dict(definition="observed vs block-CV out-of-fold prediction on all site-years; hexagonal bins with log density (n >= 2000) or points; 1:1 line; RMSE and Pearson r", values=meta["a"]),
               b=dict(definition="RMSE(method) - RMSE(recalibrated Stefan) on the same site-years under each design; 95 % paired block-bootstrap interval; region hold-out: reference = source-coefficient Stefan", designs=SCHEMES, values=meta["b"]),
               outputs=C.file_info(paths), canvas_mm=[W, round(H, 1)], pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep)
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths: print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
