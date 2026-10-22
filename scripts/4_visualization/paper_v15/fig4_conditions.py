"""Fig. 4 (v15, 2026-10-21): conditions of the physics-residual gain and the number of labels.
a  covariate sets: RMSE difference from the recalibrated Stefan model under spatial-block validation (six nested sets, three regions).
b  terciles of eight covariates within each region: the same difference (heat table; outlined = interval excludes 0; hatched = insufficient data).
c  within-region label ladder: difference from a Stefan model recalibrated on the same n labels (physics-residual with 95 % band, direct ML).
d  transfer ladder: RMSE in each region for models trained on the other two regions plus n target labels.
Reuses the ladder panels of fig4_labels.py (draw_a, draw_b) and the data loaders of the v14 Fig. 3 (_f3_v14_parts.py)."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import _f3_v14_parts as F3  # noqa: E402
import fig4_labels as F4  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

STEM = "Fig4_conditions"; REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}; RC = {r: C.REGION[r][1] for r in REGIONS}
NEG, POS = "#1A7AB3", "#B03A2E"; CM_DIV = LinearSegmentedColormap.from_list("div", [NEG, "#FFFFFF", POS]); VMAX = 7.0
FEAT_SHORT = {"F8": "Climate only", "F8T": "+ Terrain", "F8TS": "+ Soil (main set)", "F8TS_M": "+ MODIS", "F8TS_X": "+ Hi-res soil, cover", "F8TS_ALL": "All"}
COV_SHORT = {"tdd": "TDD", "fdd": "FDD", "swe": "Snow (SWE)", "soc": "SOC", "clay": "Clay", "slope": "Slope", "elev": "Elevation", "ndvi": "NDVI"}


def main():
    t0 = time.time(); C.setup()
    W, H = 180.0, 128.0; fig = F4.C.new_fig(H); meta = {}                                  # fig34_common13 figure (its ax_mm needs fig._h_mm); c13_common.ax_mm also works on it
    # ---------------- a: covariate sets (heat table, v15.3: replaces the forest plot)
    lad = F3.data_c(); feats = [f for f in F3.FEATS if f in set(lad.featset)]
    xa0, ya0, cwa, rha = 33.0, 15.0, 15.0, 6.2; norm_a = TwoSlopeNorm(vcenter=0, vmin=-VMAX, vmax=VMAX)
    ax = C.ax_mm(fig, xa0, ya0, 3 * cwa, len(feats) * rha); ax.set_axis_off(); ax.set_xlim(0, 3 * cwa); ax.set_ylim(len(feats) * rha, 0)
    for j_, r in enumerate(REGIONS):
        ax.text((j_ + 0.5) * cwa, -0.25 * rha, RN[r], ha="center", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT)
    for k, f in enumerate(feats):
        ax.text(-0.8, (k + 0.5) * rha, FEAT_SHORT.get(f, f), ha="right", va="center", fontsize=C.FS_SMALL, color=C.TEXT)
        for j_, r in enumerate(REGIONS):
            q = lad[(lad.region == r) & (lad.featset == f)]
            if q.empty: continue
            v, lo, hi = float(q.delta.iloc[0]), float(q.lo.iloc[0]), float(q.hi.iloc[0]); ex = (lo > 0) or (hi < 0)
            ax.add_patch(Rectangle((j_ * cwa + 0.08, k * rha + 0.08), cwa - 0.16, rha - 0.16, facecolor=CM_DIV(norm_a(v)), edgecolor=(C.TEXT if ex else "none"), lw=0.6))
            ax.text((j_ + 0.5) * cwa, (k + 0.5) * rha, f"{v:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=C.FS_SMALL, color=("white" if abs(v) > 4 else C.TEXT))
            meta[f"a_{r}_{f}"] = [round(v, 2), round(lo, 2), round(hi, 2)]
    C.text_mm(fig, xa0 - 20.0, ya0 + len(feats) * rha + 2.0, "ΔRMSE vs recalibrated Stefan (cm); outlined: 95% interval excludes 0", fontsize=C.FS_SMALL, ha="left", va="top", color=C.AUX)
    C.letter(fig, 2.0, 3.0, "a"); C.text_mm(fig, 7.0, 3.3, "Covariate set given to the residual learner (spatial-block validation)", fontsize=C.FS, ha="left", va="top")
    # ---------------- b: terciles heat table
    ter = F3.data_d(); ter = ter[ter.tercile >= 0]; covs = [c for c, _, _ in F3.COVS if c in set(ter.covariate)]
    ax = C.ax_mm(fig, 108.0, 11.0, 70.0, 50.0); ax.set_axis_off(); norm = TwoSlopeNorm(vcenter=0, vmin=-VMAX, vmax=VMAX)
    nrow, ncol = len(covs), 9
    for i, c in enumerate(covs):
        for j, r in enumerate(REGIONS):
            for t in range(3):
                q = ter[(ter.covariate == c) & (ter.region == r) & (ter.tercile == t)]; x, y = j * 3 + t, nrow - 1 - i
                if q.empty or not bool(q.shown.iloc[0]):
                    ax.add_patch(Rectangle((x + 0.05, y + 0.07), 0.9, 0.86, facecolor="white", edgecolor="#BFBFBF", hatch="////", lw=0.3)); continue
                v = float(q.delta.iloc[0]); ex = bool(q.excludes_0.iloc[0])
                ax.add_patch(Rectangle((x + 0.05, y + 0.07), 0.9, 0.86, facecolor=CM_DIV(norm(v)), edgecolor=(C.TEXT if ex else "none"), lw=0.6))
                ax.text(x + 0.5, y + 0.5, f"{v:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=C.FS_SMALL, color=("white" if abs(v) > 4 else C.TEXT))
                meta[f"b_{r}_{c}_{t}"] = [round(v, 2), ex]
        ax.text(-0.15, nrow - 1 - i + 0.5, COV_SHORT.get(c, c), ha="right", va="center", fontsize=C.FS_SMALL)
    for j, r in enumerate(REGIONS):
        ax.text(j * 3 + 1.5, nrow + 0.95, RN[r], ha="center", va="bottom", fontsize=C.FS_SMALL)
        for t, tl in enumerate(("Low", "Mid", "High")): ax.text(j * 3 + t + 0.5, nrow + 0.12, tl, ha="center", va="bottom", fontsize=5.8 if False else C.FS_SMALL, rotation=0)
    ax.set_xlim(-0.2, ncol + 0.2); ax.set_ylim(-0.3, nrow + 1.9); ax.set_aspect("auto")
    C.letter(fig, 88.0, 3.0, "b"); C.text_mm(fig, 93.0, 3.3, "Terciles of covariates within each region", fontsize=C.FS, ha="left", va="top")
    C.text_mm(fig, 178.0, 64.5, "Outlined: 95% interval excludes 0; hatched: fewer than 30 site-years or five blocks", fontsize=C.FS_SMALL, ha="right", va="top", color=C.AUX)
    # ---------------- c: transfer ladder (fig4_labels panel b, repositioned); the within-region ladder is Supplementary Fig. S4
    F4.Y2, F4.H2 = 86.0, 34.0
    F4.COL_W = (F4.R_EDGE - F4.X0 - 2 * F4.GAP) / 3.0; F4.COL_X = [F4.X0 + j * (F4.COL_W + F4.GAP) for j in range(3)]
    m, d, dropped = F4.load_ladder()
    b_axes, b_vals, clipped = F4.draw_b(fig, m, [(x, F4.COL_W) for x in F4.COL_X])
    b_axes[1].set_xlabel("Target-region labels added (site-years)")
    meta["c"] = b_vals; meta["c_clipped"] = clipped
    # method legend above c (four methods, as in fig4_labels)
    hs = [F4.C.method_handle(k) for k in ("P0", "P1", "R1", "D0")]; labs = [F4.C.METH[k]["label"] for k in ("P0", "P1", "R1", "D0")]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(F4.X0 / W, 1 - 73.0 / H), ncol=4, frameon=False,
               fontsize=C.FS_SMALL, handlelength=2.2, columnspacing=1.6, handletextpad=0.6, borderaxespad=0.0)
    C.letter(fig, 2.0, F4.Y2 - 6.5, "c"); C.text_mm(fig, 7.0, F4.Y2 - 6.2, "Transfer to a region: trained on the other two regions plus n labels of the target region", fontsize=C.FS, ha="left", va="top")
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN); plt.close(fig)
    src = dict(figure="Fig. 4", script="scripts/4_visualization/paper_v15/fig4_conditions.py", version="v15 (2026-10-21)",
               inputs=dict(covariate_sets="runs/{core_v6,featsets_v6,modis_v6}/deltas_vs_ref.csv via _f3_v14_parts.data_c", terciles="outputs/figures/paper/v7/source_data/F4e_gain_conditions.csv",
                           ladders="data/processed/v5/runs/ladder_v6/ladder_{metrics,deltas}.csv (LL within region, TL transfer)"),
               definitions=dict(a="RMSE(physics-residual) - RMSE(recalibrated Stefan), spatial-block CV, 95 % paired block-bootstrap interval",
                                b="same difference within regional terciles; cells with < 30 site-years or < 5 blocks hatched",
                                c="RMSE for models trained on the other two regions plus n target labels; values beyond the axis printed (within-region ladder: Supplementary Fig. S4)"),
               values=meta, dropped_ladder_rows=dropped, outputs=C.file_info(paths), canvas_mm=[W, H], pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep)
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths: print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
