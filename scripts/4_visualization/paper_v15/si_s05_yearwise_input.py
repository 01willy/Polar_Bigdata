"""Supplementary Fig. S5 (v15.2, 2026-10-22): RMSE per observation year as a heat table (replaces the dot-line panels of v15.0/15.1).

Rows = observation years with at least 30 site-years in the region; columns = recalibrated Stefan, physics + residual ML (both spatial-block
out-of-fold, climate input of the observation year to the observation day), Wei et al. 2026 and Liu et al. 2024 (product value of the observation
year); one column group per region. Cell text = RMSE (cm); cell colour = RMSE minus the RMSE of the physics-residual model in the same year
(blue, lower; red, higher; +-10 cm). Row set per year = site-years with a Wei et al. value for that year (Liu et al. on the subset with a Liu value,
2010-2020; n printed as n_wei / n_liu). The effect of the year-specific climate input (former panel a) is Fig. 3a of the main text.
Inputs: data/processed/v5/labels_cell_year_v5.csv, products_cell_year_v5.csv, core_v6 out-of-fold predictions (fig5_products_v15.oof_all).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper15 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 python3 scripts/4_visualization/paper_v15/si_s05_yearwise_input.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import fig5_products_v15 as G5  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

STEM = "FigS5_yearwise"; REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}; MN = C.METHOD_NAME; MC = C.METHOD
NEG, POS = "#3C78A8", "#B5534A"; CM_DIV = LinearSegmentedColormap.from_list("div15", [NEG, "#FFFFFF", POS]); VMAX = 10.0; MIN_N = 30
COLS = [("P1", "Stefan\n(recal.)", MC["P1"]), ("R1", "Physics\n+ ML", MC["R1"]), ("wei", "Wei et al.\n2026", C.PRODUCT["wei"][1]), ("liu", "Liu et al.\n2024", C.PRODUCT["liu"][1])]


def rmse(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]; return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def rows():
    lab = pd.read_csv(C.ROOT / "data/processed/v5/labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1].copy()
    pr = pd.read_csv(C.ROOT / "data/processed/v5/products_cell_year_v5.csv"); idx = pd.MultiIndex.from_arrays([lab.loc_id, lab.year])
    for p in ("wei", "liu"):
        lab[p] = pr[pr["product"] == p].set_index(["loc_id", "year"]).value_cm.reindex(idx).values
    for m in ("P1", "R1"):
        lab[m] = G5.oof_all(m).reindex(idx).values
    return lab[np.isfinite(lab.wei) & np.isfinite(lab.R1) & np.isfinite(lab.P1)].copy()


def main():
    C.setup(); d = rows(); tab = {}
    for r in REGIONS:
        q = d[d.region == r]; cnt = q.groupby("year").size(); yrs = [int(y) for y in cnt[cnt >= MIN_N].index]
        rec = []
        for y in yrs:
            z = q[q.year == y]; zl = z[np.isfinite(z.liu)]
            rec.append(dict(year=y, n=int(len(z)), n_liu=int(len(zl)), P1=rmse(z.P1 - z.alt_cm), R1=rmse(z.R1 - z.alt_cm), wei=rmse(z.wei - z.alt_cm), liu=rmse(zl.liu - zl.alt_cm) if len(zl) >= MIN_N else np.nan))
        tab[r] = pd.DataFrame(rec)
    nrow = max(len(t) for t in tab.values()); cw, rh = 9.0, 5.6; gap_g = 3.5; x0 = 8.0; y0 = 21.0; xn = 15.5; xc = 17.0
    W = 180.0; H = y0 + nrow * rh + 14.0; fig = C.figure(W, H); norm = TwoSlopeNorm(vcenter=0, vmin=-VMAX, vmax=VMAX); meta = {}
    gw = 4 * cw + xc                                                                   # group width: year column + n column + 4 cells
    for gi, r in enumerate(REGIONS):
        t = tab[r]; gx = x0 + gi * (gw + gap_g); ax = C.ax_mm(fig, gx, y0, gw, nrow * rh); ax.set_axis_off(); ax.set_xlim(0, gw); ax.set_ylim(nrow * rh, 0)
        ax.text(xc + 2 * cw, -rh * 2.1, RN[r], ha="center", va="bottom", fontsize=C.FS, color=C.TEXT)
        ax.text(0.0, -0.25 * rh, "Year", ha="left", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT); ax.text(xn, -0.25 * rh, "n", ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT)
        for ci, (k, name, col) in enumerate(COLS):
            ax.text(xc + (ci + 0.5) * cw, -0.22 * rh, name, ha="center", va="bottom", fontsize=C.FS_SMALL, color=col, linespacing=1.05)
        for i, z in t.iterrows():
            ax.text(0.0, (i + 0.5) * rh, str(int(z.year)), ha="left", va="center", fontsize=C.FS_SMALL, color=C.TEXT)
            ax.text(xn, (i + 0.5) * rh, f"{int(z.n):,}", ha="right", va="center", fontsize=C.FS_SMALL, color=C.TEXT_TICK)
            for ci, (k, name, col) in enumerate(COLS):
                v = z[k]; xx = xc + ci * cw
                if not np.isfinite(v):
                    ax.text(xx + 0.5 * cw, (i + 0.5) * rh, "–", ha="center", va="center", fontsize=C.FS_SMALL, color=C.AUX); continue
                dv = v - z.R1; face = "#F4F4F4" if k == "R1" else CM_DIV(norm(dv))
                ax.add_patch(Rectangle((xx + 0.08, i * rh + 0.08), cw - 0.16, rh - 0.16, facecolor=face, edgecolor="none"))
                ax.text(xx + 0.5 * cw, (i + 0.5) * rh, f"{v:.1f}", ha="center", va="center", fontsize=C.FS_SMALL, color=("white" if abs(dv) > 7.0 else C.TEXT))
        meta[r] = t.round(2).to_dict(orient="records")
    cax = C.ax_mm(fig, 70.0, y0 + nrow * rh + 4.0, 40.0, 1.6); cb = mpl.colorbar.ColorbarBase(cax, cmap=CM_DIV, norm=norm, orientation="horizontal", ticks=[-10, -5, 0, 5, 10], extend="both")
    cb.outline.set_linewidth(C.LW_AX); cax.tick_params(labelsize=C.FS_TICK, length=1.5, width=0.5, pad=1.0); cax.set_xlabel("RMSE − RMSE of physics + residual ML, same year (cm)", fontsize=C.FS_SMALL, labelpad=1.5)
    C.letter(fig, 2.0, 2.5, "a"); C.text_mm(fig, 7.0, 2.8, "RMSE (cm) per observation year on the site-years with a Wei et al. value; Liu et al. on its subset (to 2020)", fontsize=C.FS, ha="left", va="top")
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S5", dict(figure="Supplementary Fig. S5", script="scripts/4_visualization/paper_v15/si_s05_yearwise_input.py", version="v15.2 (2026-10-22) heat table",
                                definition=f"per observation year with >= {MIN_N} site-years: RMSE of recalibrated Stefan and physics-residual (spatial-block out-of-fold, seed mean, year-specific input), "
                                           "Wei et al. 2026 and Liu et al. 2024 (product value of the year) on the site-years with a Wei value; Liu on the subset with a Liu value; colour = RMSE - RMSE(physics-residual)",
                                values=meta, outputs=C.file_info(paths), canvas_mm=[W, round(H, 1)], qa=rep))
    for p in paths: print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
