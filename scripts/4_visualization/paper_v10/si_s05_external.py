"""Supplementary Fig. S5: external CALM regions without local labels, and adaptation with five local labels (content of v9_f10_extra.py EXT and
the right panel of the FT figure), paper palette.

a  External CALM sites (open circles; numbered sub-regions) and the three training regions (grey markers), north polar stereographic.
b  RMSE per external sub-region without local labels (sites inside the covariate range of the training data): Physics + residual ML trained on the
   three source regions; the same with Stefan recalibrated on the local labels; Wei 2026 and Liu 2024 (trained on these sites); ESA CCI v5 (no ALT
   training). Each series is scored on the rows where it has a value (n per series in the JSON). Log x axis.
c  Physics + residual ML with no local labels and with five local labels (labels from the same sub-region), and ESA CCI v5, on the rows where Wei 2026,
   Liu 2024 and ESA CCI v5 all have a value (seed-averaged predictions). Same rows as b where available.
Inputs: data/processed/v5/runs/extra_v9/ext_metrics.csv (rows == 'own', subset in_range), preds_EXTFT_*.csv.gz (via v9_f10_extra.ft_table),
        data/processed/v9_ext/labels_cell_year_v5.csv, data/processed/v5/labels_cell_year_v5.csv.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/si_s05_external.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import v9_f10_extra as F10  # noqa: E402

STEM = "FigS5_external"
EXT = C.ROOT / "data" / "processed" / "v9_ext"
SUBS = F10.SUB_ORDER
R1C = C.METHOD["R1"]
_wn, _wf, _we, _wm = C.product_style("wei"); _ln, _lf, _le, _lm = C.product_style("liu"); _cn, _cf, _ce, _cm = C.product_style("cci5y")
SER = [("R1", "Physics + residual ML, source regions only", R1C, R1C, "o", True),
       ("R1loc", "Physics + residual ML, Stefan recalibrated locally", R1C, R1C, "o", False),
       ("wei", "Wei 2026 (trained on these sites)", _wf, _we, _wm, True),
       ("liu", "Liu 2024 (trained on these sites)", _lf, _le, _lm, True),
       ("cci5y", "ESA CCI v5 (no ALT training)", _cf, _ce, _cm, True)]
XL = (4.0, 200.0)


def logx(ax):
    ax.set_xscale("log"); ax.set_xlim(*XL); ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(["5", "10", "20", "50", "100"]); ax.minorticks_off()


def main():
    C.setup()
    t, n = F10.ext_tables(); ft = F10.ft_table()
    subs = [s for s in SUBS if s in t.index] + ["All sites"]; ny = len(subs)
    W, H = 180.0, 80.0
    fig = C.figure(W, H)
    # ---------------- a: map
    axm = fig.add_axes([0.5 / W, 1 - (12.0 + 58.0) / H, 58.0 / W, 58.0 / H], projection=ccrs.NorthPolarStereo(central_longitude=90))
    PC = ccrs.PlateCarree(); axm.set_extent([-180, 180, 55, 90], PC)
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#E6E6E6", edgecolor="none", zorder=0)
    axm.coastlines("50m", linewidth=0.3, color="#9A9A9A", zorder=1)
    axm.gridlines(crs=PC, draw_labels=False, linewidth=0.3, color="#D0D0D0", xlocs=range(-180, 180, 60), ylocs=[60, 70, 80], zorder=1)
    axm.spines["geo"].set_linewidth(0.4); axm.spines["geo"].set_edgecolor("black")
    lab = pd.read_csv(C.ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv", usecols=["region", "lat", "lon"]).drop_duplicates()
    for r in C.REGIONS:
        q = lab[lab.region == r]; nm, col, mk = C.REGION[r]
        axm.scatter(q.lon, q.lat, s=3.0, marker=mk, color=col, transform=PC, zorder=3, lw=0)
    ex = pd.read_csv(EXT / "labels_cell_year_v5.csv")
    exs = ex.drop_duplicates("loc_id")
    axm.scatter(exs.lon, exs.lat, s=4.5, facecolor="white", edgecolor="black", lw=0.45, transform=PC, zorder=4)
    for k, name in enumerate([s for s in SUBS if s in set(exs.region)], start=1):
        q = exs[exs.region == name]; xy = axm.projection.transform_points(PC, q.lon.values, q.lat.values)[:, :2]
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean(); dx, dy = F10.NUM_OFF.get(name, (12, 0))
        axm.annotate(str(k), (cx, cy), xytext=(dx * 0.62, dy * 0.62), textcoords="offset points", fontsize=C.FS_SMALL, color="black", ha="center",
                     va="center", zorder=6, bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="black", lw=0.4))
    hs = [Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec="black", mew=0.45)]
    hs += [Line2D([], [], ls="none", marker=C.REGION[r][2], ms=2.6, mfc=C.REGION[r][1], mec="none") for r in C.REGIONS]
    fig.legend(hs, ["External CALM sites"] + [f"{C.REGION[r][0]} (training)" for r in C.REGIONS], loc="upper left",
               bbox_to_anchor=(1.0 / W, 1 - 71.5 / H), ncol=2, fontsize=C.FS_SMALL, handlelength=0.8, columnspacing=1.0, labelspacing=0.3)
    # ---------------- b: no local labels
    xb, wb = 92.0, 39.0; yt, hh = 12.0, 58.0
    axb = C.ax_mm(fig, xb, yt, wb, hh); C.clean_axes(axb, left=False)
    vals_b = {}
    for j, (k, name, face, edge, mk, filled) in enumerate(SER):
        off = (j - (len(SER) - 1) / 2) * 0.14
        for i, s in enumerate(subs):
            v = t.loc[s, k] if k in t.columns else np.nan
            if not np.isfinite(v):
                continue
            axb.plot([v], [i + off], ls="none", marker=mk, ms=2.9 if mk != "D" else 2.6, mfc=face if filled else "white", mec=edge, mew=0.5, zorder=3)
            vals_b.setdefault(s, {})[name] = dict(rmse_cm=round(float(v), 2), n=int(n.loc[s, k]))
    for i in range(ny):
        axb.axhline(i, color="#EDEDED", lw=C.LW_THIN, zorder=0)
    axb.axhline(ny - 1.5, color="#8C8C8C", lw=C.LW_THIN, zorder=0.5)
    axb.set_yticks(range(ny))
    axb.set_yticklabels([(f"{i}  {s} ({int(n.loc[s, 'R1'])})" if s != "All sites" else f"All sites ({int(n.loc[s, 'R1'])})") for i, s in enumerate(subs, start=1)])
    axb.tick_params(axis="y", length=0, pad=2); axb.set_ylim(ny - 0.5, -0.5); logx(axb)
    axb.set_xlabel("RMSE (cm), no local labels")
    # ---------------- c: five local labels (FT right panel)
    xc = xb + wb + 6.0; wc = W - xc - 2.0
    axc = C.ax_mm(fig, xc, yt, wc, hh); C.clean_axes(axc, left=False)
    vals_c = {}
    for i, s in enumerate(subs):
        q = ft[ft.scope == s]
        if s not in F10.FT_SUBS or not len(q):                 # sub-regions of the original FT panel (Kamchatka, Greenland: too few rows)
            continue
        v0 = float(q[(q.series == "R1S") & (q.n_label == "0")].rmse.iloc[0]); v5 = float(q[(q.series == "R1S") & (q.n_label == "5")].rmse.iloc[0])
        vc = float(q[(q.series == "cci5y") & (q.n_label == "0")].rmse.iloc[0]); nn = int(q[(q.series == "R1S") & (q.n_label == "0")].n.iloc[0])
        axc.plot([v0, v5], [i, i], color="#C8C8C8", lw=1.0, zorder=1)
        axc.plot([v0], [i - 0.12], ls="none", marker="o", ms=3.1, mfc="white", mec=R1C, mew=0.6, zorder=3)
        axc.plot([v5], [i - 0.12], ls="none", marker="o", ms=3.1, mfc=R1C, mec=R1C, zorder=3)
        axc.plot([vc], [i + 0.16], ls="none", marker=_cm, ms=2.9, mfc=_cf, mec=_ce, mew=0.5, zorder=2)
        vals_c[s] = dict(no_local_labels=round(v0, 2), five_local_labels=round(v5, 2), esa_cci_v5=round(vc, 2), n_rows=nn)
    for i, s in enumerate(subs):                              # row guides only where c has values (Kamchatka, Greenland, all sites: no five-label run)
        if s in vals_c:
            axc.axhline(i, color="#EDEDED", lw=C.LW_THIN, zorder=0)
    axc.set_yticks(range(ny)); axc.set_yticklabels([]); axc.tick_params(axis="y", length=0); axc.set_ylim(ny - 0.5, -0.5); logx(axc)
    axc.set_xlabel("RMSE (cm)")
    # legends
    hb = [Line2D([], [], ls="none", marker=mk, ms=3.0, mfc=face if filled else "white", mec=edge, mew=0.5) for _, _, face, edge, mk, filled in SER]
    fig.legend(hb, [s[1] for s in SER], loc="upper left", bbox_to_anchor=(62.0 / W, 1 - 0.6 / H), ncol=2, fontsize=C.FS_SMALL, handlelength=0.8,
               columnspacing=1.0, labelspacing=0.25)
    hc = [Line2D([], [], ls="none", marker="o", ms=3.1, mfc="white", mec=R1C, mew=0.6), Line2D([], [], ls="none", marker="o", ms=3.1, mfc=R1C, mec=R1C),
          Line2D([], [], ls="none", marker=_cm, ms=2.9, mfc=_cf, mec=_ce, mew=0.5)]
    C.letter(fig, 0.5, 0.5, "a"); C.letter(fig, 60.0, 9.0, "b"); C.letter(fig, xc - 1.0, 9.0, "c")
    axc.legend(hc, ["0 local labels", "5 local labels", "ESA CCI v5"], loc="lower left", bbox_to_anchor=(0.03, 0.02), ncol=1, fontsize=C.FS_SMALL,
               handlelength=0.8, labelspacing=0.3)
    rep = C.qa_figure(fig, ignore_ink_for=[str(k) for k in range(1, 9)]); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S5", dict(
        title="External CALM regions without local labels and adaptation by sub-region", stem=STEM, script="scripts/4_visualization/paper_v10/si_s05_external.py",
        inputs=["data/processed/v5/runs/extra_v9/ext_metrics.csv", "data/processed/v5/runs/extra_v9/preds_EXTFT_*.csv.gz", "data/processed/v9_ext/labels_cell_year_v5.csv"],
        years=dict(external_labels=[int(ex.year.min()), int(ex.year.max())], c_rows="rows where Wei 2026, Liu 2024 (to 2020) and ESA CCI v5 all have a value of the observation year",
                   training_labels="Alaska 2010-2024, Lena Delta 2010-2022, Canada (NW) 2010-2024"),
        definition=dict(b="sites inside the covariate range of the training data (subset in_range); each series scored on its own rows (n per series)",
                        c="seed-averaged predictions; 5 local labels drawn from the same sub-region; ESA CCI v5 on the same rows"),
        n_external_sites=int(exs.loc_id.nunique()), n_external_cell_years=int(len(ex)), values_b=vals_b, values_c=vals_c,
        numbers_printed=dict(b_row_labels="sub-region number, name and n (Physics + residual ML rows)"), outputs=C.file_info(paths), size_mm=[W, H], qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
