"""Fig. 3 (v15.1, 2026-10-22): climate input, validation design and the in-region / out-of-region contrast.
a  change in spatial-block RMSE when the 2011-2024 mean climate input is replaced by the input of the observation year (three regions x three methods, 95 % intervals).
b  RMSE of three methods under five validation designs (rows, ordered by strictness) in three regions (column groups): number = RMSE (cm), colour = difference
   from the recalibrated Stefan model of the same design (blue = lower, red = higher); region hold-out uses the source-coefficient Stefan model.
c  the same methods when the region's own labels are used (spatial-block CV) and when the region is withheld (trained on the other two regions only):
   RMSE connected per method, values printed. The observed-vs-predicted density matrix moved to Supplementary Fig. S11."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import _f3_v14_parts as F3  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = C.ROOT; RUNS = ROOT / "data" / "processed" / "v5" / "runs"
STEM = "Fig3_accuracy"; REGIONS = C.REGIONS; RN = {r: C.REGION[r][0] for r in REGIONS}; MN = C.METHOD_NAME; MC = C.METHOD
SCHEMES = ["random", "block", "year", "block_year", "region"]; SCHEME_LB = ["Random", "Spatial block", "Year", "Block-year", "Region held out"]
METHS = ["P1", "R1", "D0"]; SHORT = {"P1": "Stefan\n(recal.)", "R1": "Physics\n+ ML", "D0": "Direct\nML"}
NEG, POS = "#3C78A8", "#B5534A"; CM_DIV = LinearSegmentedColormap.from_list("div15", [NEG, "#FFFFFF", POS]); VMAX = 8.0


def rmse_table():
    sm = pd.read_csv(RUNS / "core_v6/summary_metrics.csv"); sm = sm[(sm.forcing == "year_obs") & (sm.featset == "F8TS")]
    def val(region, scheme, m):
        q = sm[(sm.region == region) & (sm.scheme == scheme) & (sm.method == m)]
        if m == "R1": q = q[(q.learner == "catboost") & np.isclose(q.lam, 0.5)]
        if m == "D0": q = q[q.learner == "catboost"]
        return float(q.rmse.iloc[0])
    return val


def main():
    t0 = time.time(); C.setup(); W, H = 180.0, 106.0; fig = C.figure(W, H); meta = {}
    # ---------------- a: climate input effect (forest)
    t, dl = F3.data_ab()
    ax = C.ax_mm(fig, 24.0, 12.0, 46.0, 36.0); C.clean_axes(ax)
    for i in range(3): ax.axhline(i, color="#EDEDED", lw=0.4, zorder=0)
    for m, dy, filled, mk in (("P1", -0.25, True, "s"), ("R1", 0.0, True, "o"), ("D0", 0.25, False, "o")):
        for i, r in enumerate(REGIONS):
            z = dl[(dl.region == r) & (dl.method == m)].iloc[0]; c = MC[m]; y = i + dy
            ax.plot([z.lo, z.hi], [y, y], color=c, lw=C.LW_AUX, zorder=2); ax.plot(z.delta, y, marker=mk, ms=C.MS, mfc=(c if filled else "white"), mec=c, mew=0.7, ls="none", zorder=3)
            meta[f"a_{r}_{m}"] = [round(float(z.delta), 2), round(float(z.lo), 2), round(float(z.hi), 2)]
    ax.axvline(0, color=C.ZERO, lw=C.LW_REF, zorder=1); ax.set_yticks(range(3)); ax.set_yticklabels([RN[r] for r in REGIONS]); ax.set_ylim(2.5, -0.5)
    ax.set_xlabel("ΔRMSE, year-specific − mean climate input (cm)")
    hs = [Line2D([], [], ls="none", marker="s", ms=C.MS, color=MC["P1"]), Line2D([], [], ls="none", marker="o", ms=C.MS, color=MC["R1"]), Line2D([], [], ls="none", marker="o", ms=C.MS, mfc="white", mec=MC["D0"], mew=0.7)]
    fig.legend(hs, [MN["P1"], MN["R1"], MN["D0"]], loc="upper left", bbox_to_anchor=(7.0 / W, 1 - 6.3 / H), ncol=3, fontsize=C.FS_SMALL, frameon=False, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.0)
    C.letter(fig, 2.0, 3.0, "a"); C.text_mm(fig, 7.0, 3.3, "Year-specific climate input (spatial-block CV)", fontsize=C.FS, ha="left", va="top")
    # ---------------- b: heat table designs x (region, method)
    val = rmse_table(); x0, y0 = 90.0, 19.0; cw, rh = 9.2, 6.0; gap_g = 2.2
    ax = C.ax_mm(fig, x0, y0, 3 * 3 * cw + 2 * gap_g, 5 * rh); ax.set_axis_off(); norm = TwoSlopeNorm(vcenter=0, vmin=-VMAX, vmax=VMAX)
    ax.set_xlim(0, 9 * cw + 2 * gap_g); ax.set_ylim(5 * rh, 0)
    for gi, r in enumerate(REGIONS):
        gx = gi * (3 * cw + gap_g)
        ax.text(gx + 1.5 * cw, -rh * 1.45, RN[r], ha="center", va="bottom", fontsize=C.FS, color=C.TEXT)
        for mi, m in enumerate(METHS):
            ax.text(gx + (mi + 0.5) * cw, -0.22 * rh, SHORT[m], ha="center", va="bottom", fontsize=C.FS_SMALL, color=MC[m], linespacing=1.05)
            for si, sch in enumerate(SCHEMES):
                ref = "P0" if sch == "region" else "P1"
                v = val(r, sch, m if not (sch == "region" and m == "P1") else "P0"); d = v - val(r, sch, ref)
                face = "#F4F4F4" if m == "P1" else CM_DIV(norm(d))
                ax.add_patch(Rectangle((gx + mi * cw + 0.08, si * rh + 0.08), cw - 0.16, rh - 0.16, facecolor=face, edgecolor="none"))
                ax.text(gx + (mi + 0.5) * cw, (si + 0.5) * rh, f"{v:.1f}", ha="center", va="center", fontsize=C.FS_SMALL, color=("white" if abs(d) > 5.5 else C.TEXT))
                meta[f"b_{r}_{sch}_{m}"] = round(v, 2)
    for si, sch in enumerate(SCHEME_LB): ax.text(-0.6, (si + 0.5) * rh, sch, ha="right", va="center", fontsize=C.FS_SMALL, color=C.TEXT)
    import matplotlib as mpl
    cax = C.ax_mm(fig, x0 + 3 * 3 * cw + 2 * gap_g - 40.0, y0 + 5 * rh + 2.2, 40.0, 1.5)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CM_DIV, norm=norm, orientation="horizontal", ticks=[-8, -4, 0, 4, 8]); cb.outline.set_linewidth(C.LW_AX)
    cax.tick_params(labelsize=C.FS_TICK, length=1.5, width=0.5, pad=1.0); cax.set_xlabel("RMSE − Stefan RMSE, same design (cm)", fontsize=C.FS_SMALL, labelpad=1.2)
    C.letter(fig, 76.0, 3.0, "b"); C.text_mm(fig, 81.0, 3.3, "RMSE (cm) by validation design, from least to most demanding", fontsize=C.FS, ha="left", va="top")
    # ---------------- c: in-region vs region withheld (slope per method)
    yc = 69.0; hc = 26.0; wc = 46.0; gc = 11.0; xc0 = 24.0
    for i, r in enumerate(REGIONS):
        ax = C.ax_mm(fig, xc0 + i * (wc + gc), yc, wc, hc); C.clean_axes(ax)
        styles = (("P1", "s", True, "-"), ("R1", "o", True, "-"), ("D0", "o", False, C.DASH))
        vals = {m: (val(r, "block", m), val(r, "region", "P0" if m == "P1" else m)) for m, _, _, _ in styles}
        meta[f"c_{r}"] = {m: [round(a, 2), round(b, 2)] for m, (a, b) in vals.items()}
        allv = [v for ab in vals.values() for v in ab]; lo, hi = min(allv), max(allv); pad = 0.12 * (hi - lo) + 0.5
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1]); ax.set_xticklabels(["Trained in\nthe region", "Region\nwithheld"])
        min_gap = 2.5 * (hi - lo + 2 * pad) / hc                                               # 2.5 mm between printed values, in data units
        def spread(items):                                                                      # items: [(value, key)] -> {key: label position}
            items = sorted(items); pos = [v for v, _ in items]
            for _ in range(30):
                for k in range(1, len(pos)):
                    if pos[k] - pos[k - 1] < min_gap: mid = 0.5 * (pos[k] + pos[k - 1]); pos[k - 1] = mid - 0.5 * min_gap; pos[k] = mid + 0.5 * min_gap
            return {key: p for (_, key), p in zip(items, pos)}
        lab_in = spread([(v[0], m) for m, v in vals.items()]); lab_out = spread([(v[1], m) for m, v in vals.items()])
        for m, mk, filled, ls in styles:
            v_in, v_out = vals[m]; c = MC[m]
            ax.plot([0, 1], [v_in, v_out], color=c, lw=C.LW_MAIN, ls=ls, marker=mk, ms=C.MS, mfc=(c if filled else "white"), mec=c, mew=0.7, zorder=3 if m == "R1" else 2)
            ax.text(-0.07, lab_in[m], f"{v_in:.1f}", ha="right", va="center", fontsize=C.FS_SMALL, color=c); ax.text(1.07, lab_out[m], f"{v_out:.1f}", ha="left", va="center", fontsize=C.FS_SMALL, color=c)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        if i == 0: ax.set_ylabel("RMSE (cm)")
        C.text_mm(fig, xc0 + i * (wc + gc), yc - 1.2, RN[r], fontsize=C.FS, ha="left", va="bottom")
    C.letter(fig, 2.0, yc - 8.0, "c"); C.text_mm(fig, 7.0, yc - 7.7, "Trained in the region (spatial-block CV) against trained on the other two regions only", fontsize=C.FS, ha="left", va="top")
    rep = C.qa_figure(fig, text_ink=False); C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, C.OUT_MAIN); plt.close(fig)
    src = dict(figure="Fig. 3", script="scripts/4_visualization/paper_v15/fig3_accuracy.py", version="v15.1 (2026-10-22)",
               inputs=dict(summary="data/processed/v5/runs/core_v6/summary_metrics.csv (forcing year_obs, F8TS; R1 lambda 0.5 CatBoost)", deltas_forcing="core_v6/deltas_forcing.csv (year_obs:clim, block)"),
               definitions=dict(a="RMSE(year-specific input) - RMSE(2011-2024 mean input), spatial-block CV, 95 % paired block bootstrap",
                                b="RMSE under each design; colour = RMSE - RMSE(recalibrated Stefan, same design; source-coefficient Stefan for the region hold-out)",
                                c="block-CV RMSE (own labels) and region hold-out RMSE (other two regions only) per method; Stefan: recalibrated vs source coefficient"),
               designs=SCHEMES, values=meta, outputs=C.file_info(paths), canvas_mm=[W, H], pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), qa=rep)
    C.write_json(C.SRC_MAIN / f"{STEM}.json", src)
    for p in paths: print(p, p.stat().st_size)
    print(f"  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
