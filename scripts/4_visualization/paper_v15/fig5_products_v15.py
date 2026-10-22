"""Fig. 5 (v15, 2026-10-21): comparison with published products on the same site-years.
a  RMSE of the recalibrated Stefan model, the physics-residual model and five products on the common site-years (all / > 25 km from Wei training sites)
   (panel a of fig5_products.py without direct ML).
b  observed vs predicted on the common site-years: rows = regions, columns = physics-residual model, Wei et al., Liu et al. (hexagonal log density / points).
c  distributions of predicted - observed on the common site-years for the two models and three products (violins clipped at +-80 cm, median, IQR)."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import fig5_products as F5  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

STEM = "Fig5_products"; ROOT = C.ROOT; P = ROOT / "data/processed/v5/runs/core_v6/preds"
REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}; MN = C.METHOD_NAME; MC = C.METHOD
PAT = {"P1": "P1__-__lam-1", "R1": "R1__catboost__lam0.5", "D0": "D0__catboost__lam-1"}
rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))
ramp = lambda hex_: LinearSegmentedColormap.from_list("r_" + hex_, ["#FFFFFF", hex_])


def oof_all(m):
    out = []
    for r in REGIONS:
        d = pd.read_csv(P / f"{r}__block__year_obs__F8TS__{PAT[m]}.csv.gz", usecols=["loc_id", "year", "pred", "seed"]); out.append(d.groupby(["loc_id", "year"]).pred.mean())
    return pd.concat(out)


def common_rows():
    lab = pd.read_csv(ROOT / "data/processed/v5/labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1].copy()
    pr = pd.read_csv(ROOT / "data/processed/v5/products_cell_year_v5.csv"); idx = pd.MultiIndex.from_arrays([lab.loc_id, lab.year])
    for p in ("wei", "liu", "cci4y"):
        lab[p] = pr[pr["product"] == p].set_index(["loc_id", "year"]).value_cm.reindex(idx).values
    for m in ("P1", "R1"):
        lab[m] = oof_all(m).reindex(idx).values
    return lab[np.isfinite(lab.wei) & np.isfinite(lab.liu) & np.isfinite(lab.cci4y) & np.isfinite(lab.R1)].copy()


def density(ax, x, y, col, top):
    if len(x) >= 2000:
        ax.hexbin(x, y, gridsize=26, extent=(0, top, 0, top), cmap=ramp(col), norm=LogNorm(vmin=1), mincnt=1, linewidths=0.1, edgecolors="none")
    else:
        ax.scatter(x, y, s=2.0 ** 2, facecolor=col, edgecolor="white", lw=0.2, alpha=0.85)
    ax.plot([0, top], [0, top], color=C.ZERO, lw=C.LW_REF, zorder=1); ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal")
    step = 40 if top > 120 else 20; ax.xaxis.set_major_locator(mticker.MultipleLocator(step)); ax.yaxis.set_major_locator(mticker.MultipleLocator(step))


def main():
    t0 = time.time(); C.setup()
    F5.ROWS = [r for r in F5.ROWS if r[1] != MN["D0"]]; F5.N_OURS = 2                           # panel a without direct ML
    cc = pd.read_csv(F5.CC_CSV); c = common_rows()
    for r in REGIONS:
        n_a = int(cc[(cc.region == r) & (cc.mask_km == 0) & (cc.series == "R1 (CatBoost)")].n.iloc[0]); assert n_a == int((c.region == r).sum()), (r, n_a)
    W = 180.0; s = 36.0; g = 5.0; xa0 = 30.5; gap_col = (W - xa0 - 9.5 - 3 * s) / 2; cols = [xa0 + j * (s + gap_col) for j in range(3)]   # one 3-column grid for a, b, c
    ya, ha = 9.0, 24.0; yb = ya + ha + 14.0; yc = yb + 3 * s + 2 * g + 21.0; hc = s; H = yc + hc + 16.0
    fig = C.figure(W, H); src = {"a": {"values": {}, "n_label": {}}}
    # ---------------- a (fig5_products panel a)
    wa = s; ws = 7.5; gs_ = 1.5
    axes_a = [C.ax_mm(fig, cols[j], ya, wa - ws - gs_, ha) for j in range(3)]
    sides_a = [C.ax_mm(fig, cols[j] + wa - ws, ya, ws, ha) for j in range(3)]
    F5.draw_a(fig, axes_a, sides_a, cc, src)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker="o", ms=C.MS, mfc=C.AUX, mec=C.AUX), Line2D([], [], ls="none", marker="o", ms=C.MS, mfc="white", mec=C.AUX, mew=0.7)]
    fig.legend(hs, ["All common site-years", "Sites > 25 km from product training sites"], loc="upper left", bbox_to_anchor=(cols[0] / W, 1 - 1.0 / H), ncol=2, frameon=False,
               fontsize=C.FS_SMALL, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0)
    C.letter(fig, 2.0, ya - 4.5, "a")
    # ---------------- b: observed vs predicted matrix
    series = [("R1", MN["R1"], MC["R1"]), ("wei", "Wei 2026", C.PRODUCT["wei"][1]), ("liu", "Liu 2024", C.PRODUCT["liu"][1])]
    meta_b = {}
    for i, r in enumerate(REGIONS):
        q = c[c.region == r]; top = float(np.ceil(np.percentile(np.concatenate([q.alt_cm.values] + [q[k].values for k, _, _ in series]), 99.5) / 20) * 20)
        for j, (k, name, col) in enumerate(series):
            ax = C.ax_mm(fig, cols[j], yb + i * (s + g), s, s); C.clean_axes(ax)
            density(ax, q.alt_cm.values, q[k].values, col, top)
            e = rmse(q[k] - q.alt_cm); rr = float(np.corrcoef(q.alt_cm, q[k])[0, 1]); bias = float((q[k] - q.alt_cm).mean())
            ax.text(0.04, 0.96, f"RMSE {e:.1f}\nbias {bias:+.1f}\nr {rr:.2f}".replace("-", "−"), transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
            meta_b[f"{r}_{k}"] = dict(n=int(len(q)), rmse=round(e, 2), bias=round(bias, 2), r=round(rr, 3), axis_top=top)
            if i == 0: C.text_mm(fig, cols[j], yb - 1.2, name, fontsize=C.FS, ha="left", va="bottom")
            if j == 0: ax.set_ylabel(f"{RN[r]}, n = {len(q):,}\nPredicted (cm)")
            if i == 2: ax.set_xlabel("Observed ALT (cm)")
            else: ax.set_xticklabels([])
    C.letter(fig, 2.0, yb - 8.5, "b"); C.text_mm(fig, 7.0, yb - 8.2, "Observed against predicted on the common site-years", fontsize=C.FS, ha="left", va="top")
    # ---------------- c: error distributions
    vs = [("P1", "Stefan (recal.)", MC["P1"]), ("R1", "Physics + ML", MC["R1"]), ("wei", "Wei 2026", C.PRODUCT["wei"][1]), ("liu", "Liu 2024", C.PRODUCT["liu"][1]), ("cci4y", "ESA CCI v4", C.PRODUCT["cci4y"][1])]
    LIM = 80.0; meta_c = {}; wc = s
    for i, r in enumerate(REGIONS):
        q = c[c.region == r]; ax = C.ax_mm(fig, cols[i], yc, wc, hc); C.clean_axes(ax)
        for j, (k, name, col) in enumerate(vs):
            e = (q[k] - q.alt_cm).values; e = e[np.isfinite(e)]; ec = np.clip(e, -LIM, LIM)
            vp = ax.violinplot([ec], positions=[j], widths=0.8, showextrema=False, showmedians=False, points=200)
            for b in vp["bodies"]: b.set_facecolor(C.tint(col, 0.45)); b.set_edgecolor(col); b.set_linewidth(0.5); b.set_alpha(1.0)
            q1, md, q3 = np.percentile(e, [25, 50, 75]); ax.plot([j, j], [max(q1, -LIM), min(q3, LIM)], color=col, lw=1.4); ax.plot(j, md, marker="o", ms=2.6, color="white", mec=col, mew=0.8)
            beyond = float((np.abs(e) > LIM).mean())
            if beyond > 0.005: ax.text(j, LIM + 2, f"{beyond * 100:.0f}%", ha="center", va="bottom", fontsize=C.FS_SMALL, color=C.AUX)
            meta_c[f"{r}_{k}"] = dict(n=int(len(e)), median=round(float(md), 2), q1=round(float(q1), 2), q3=round(float(q3), 2), share_beyond_80cm=round(beyond, 4))
        ax.axhline(0, color=C.ZERO, lw=C.LW_REF); ax.set_xticks(range(len(vs))); ax.set_xticklabels([n for _, n, _ in vs], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylim(-LIM - 4, LIM + 12); ax.set_yticks([-80, -40, 0, 40, 80])
        if i == 0: ax.set_ylabel("Predicted − observed (cm)")
        C.text_mm(fig, cols[i], yc - 1.2, RN[r], fontsize=C.FS, ha="left", va="bottom")
    C.letter(fig, 2.0, yc - 8.5, "c"); C.text_mm(fig, 7.0, yc - 8.2, "Error distributions on the common site-years (clipped at ±80 cm; white dot = median, bar = interquartile range)", fontsize=C.FS, ha="left", va="top")
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN); plt.close(fig)
    src.update(dict(figure="Fig. 5", script="scripts/4_visualization/paper_v15/fig5_products_v15.py", version="v15 (2026-10-21)",
                    inputs=dict(common_cells=str(F5.CC_CSV.relative_to(ROOT)), labels="data/processed/v5/labels_cell_year_v5.csv", products="data/processed/v5/products_cell_year_v5.csv",
                                predictions="core_v6/preds/<Region>__block__year_obs__F8TS__{P1,R1}*.csv.gz (seed mean)"),
                    b=dict(definition="common site-years (Wei, Liu, CCI v4 and our block-CV prediction all present); hexagonal log density (n >= 2000) or points; RMSE, mean bias, Pearson r", values=meta_b),
                    c=dict(definition="predicted - observed on the same rows; violins clipped at +-80 cm (share beyond printed); white dot median, bar interquartile range", values=meta_c),
                    outputs=C.file_info(paths), canvas_mm=[W, round(H, 1)], pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths: print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
