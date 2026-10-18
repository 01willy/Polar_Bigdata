"""Supplementary Fig. S8: long-record sites (content of deck R06 / v7_f5_products_years.draw_site), paper palette.

For the label cell with the most observed years in Alaska and in the Lena Delta (ties: most probe records, then lowest loc_id; v7_f5.long_sites):
observed ALT (open circles), Physics + residual ML at the observation day (squares) and at the end of the season (line), Stefan (recalibrated) at the
end of the season (line), Wei 2026 and Liu 2024 values of each year (grey lines; none at the Lena site). Predictions are spatial-block out-of-fold
predictions for every year 2010-2024 (scripts/2_evaluation/v5_site_series.py, climate input to the observation day / full season).
Inputs: data/processed/v5/site_series/site_series_year_obs.csv.gz, data/processed/v5/products_cell_year_v5.csv, data/processed/v5/labels_cell_year_v5.csv.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/si_s08_site_series.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import v7_f5_products_years as F5  # noqa: E402

STEM = "FigS8_site_series"
PROC = C.ROOT / "data" / "processed" / "v5"
YR = (2010, 2024)


def fmt_coord(lat, lon):
    return f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"


def main():
    C.setup()
    st = pd.read_csv(PROC / "site_series" / "site_series_year_obs.csv.gz")
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv", usecols=["loc_id", "region", "lat", "lon", "year", "in_forcing", "n_obs"]); lab = lab[lab.in_forcing == 1]
    pr = pd.read_csv(PROC / "products_cell_year_v5.csv", usecols=["loc_id", "product", "year", "value_cm"])
    picks = F5.long_sites(st, lab)
    W, H = 180.0, 66.0
    fig = C.figure(W, H)
    axes = [C.ax_mm(fig, 13.0, 15.0, 74.0, 41.0), C.ax_mm(fig, 103.0, 15.0, 74.0, 41.0)]
    wn, wf, we, wm = C.product_style("wei"); ln_, lf, le, lm = C.product_style("liu")
    info = {}
    for k, (ax, r) in enumerate(zip(axes, ("Alaska", "Lena"))):
        C.clean_axes(ax); lid = picks[r]
        g = st[(st.region == r) & (st.loc_id == lid) & st.year.between(*YR)].sort_values("year"); o = g[g.observed == 1]
        ax.plot(g.year, g.p1_eos, color=C.METHOD["P1"], lw=C.LW_MAIN, zorder=3)
        ax.plot(g.year, g.r1_eos, color=C.METHOD["R1"], lw=C.LW_MAIN, zorder=4)
        drawn = [o.y.values, o.r1_obs.values, g.p1_eos.values, g.r1_eos.values]; have = []
        for p, col, ls in (("wei", we, "-"), ("liu", le, C.DASH)):
            sp = pr[(pr.loc_id == lid) & (pr["product"] == p) & pr.year.between(*YR)].dropna(subset=["value_cm"]).sort_values("year")
            if len(sp):
                ax.plot(sp.year, sp.value_cm, color=col, lw=C.LW_AUX, ls=ls, zorder=2); drawn.append(sp.value_cm.values); have.append(p)
                info.setdefault(r, {})[f"{p}_years"] = [int(sp.year.min()), int(sp.year.max())]
        ax.plot(o.year, o.r1_obs, ls="none", marker="s", ms=2.6, mfc=C.METHOD["R1"], mec=C.METHOD["R1"], zorder=5)
        ax.plot(o.year, o.y, ls="none", marker="o", ms=3.6, mfc="white", mec="black", mew=0.6, zorder=6)
        ya = np.concatenate(drawn); lo, hi = np.nanmin(ya), np.nanmax(ya); pad = 0.08 * (hi - lo)
        ax.set_ylim(np.floor((lo - pad) / 5) * 5, np.ceil((hi + pad) / 5) * 5); ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.set_xlim(YR[0] - 0.5, YR[1] + 0.5); ax.set_xticks([2010, 2014, 2018, 2022]); ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.tick_params(which="minor", length=1.0, width=0.4)
        ax.set_xlabel("Year")
        if k == 0:
            ax.set_ylabel("ALT (cm)")
        la = float(lab[lab.loc_id == lid].lat.iloc[0]); lo_ = float(lab[lab.loc_id == lid].lon.iloc[0])
        hmm = ax.get_position().height * H
        ax.text(0.0, 1.0 + 1.2 / hmm, f"{C.REGION[r][0]}, {fmt_coord(la, lo_)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=C.FS, color=C.TEXT)
        if not have:
            ax.text(1.0, 1.0 + 1.2 / hmm, "no Wei 2026 or Liu 2024 value", transform=ax.transAxes, ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.TEXT)
        info.setdefault(r, {}).update(loc_id=int(lid), lat=round(la, 4), lon=round(lo_, 4), observed_years=[int(y) for y in o.year],
                                      n_observed_years=int(len(o)), products_drawn=have,
                                      observed_cm={str(int(y)): round(float(v), 1) for y, v in zip(o.year, o.y)},
                                      r1_obs_cm={str(int(y)): round(float(v), 1) for y, v in zip(o.year, o.r1_obs)})
        C.letter(fig, 0.5 if k == 0 else 91.0, 8.6, "ab"[k])
    hs = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec="black", mew=0.6),
          Line2D([], [], ls="none", marker="s", ms=2.6, mfc=C.METHOD["R1"], mec=C.METHOD["R1"]),
          Line2D([], [], color=C.METHOD["R1"], lw=C.LW_MAIN), Line2D([], [], color=C.METHOD["P1"], lw=C.LW_MAIN),
          Line2D([], [], color=we, lw=C.LW_AUX), Line2D([], [], color=le, lw=C.LW_AUX, ls=C.DASH)]
    labs = ["Observed", "Physics + residual ML, observation day", "Physics + residual ML, end of season", "Stefan (recalibrated), end of season",
            "Wei 2026", "Liu 2024"]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(13.0 / W, 1 - 0.8 / H), ncol=3, handlelength=1.8, columnspacing=1.6, labelspacing=0.3,
               fontsize=C.FS_TICK)
    rep = C.qa_figure(fig); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S8", dict(
        title="Long-record sites", stem=STEM, script="scripts/4_visualization/paper_v10/si_s08_site_series.py",
        inputs=["data/processed/v5/site_series/site_series_year_obs.csv.gz", "data/processed/v5/products_cell_year_v5.csv", "data/processed/v5/labels_cell_year_v5.csv"],
        years=f"{YR[0]}-{YR[1]} (predictions every year; observed years listed per site)",
        definition="cell with the most observed years per region (ties: most probe records, lowest loc_id); predictions = spatial-block out-of-fold "
                   "(v5_site_series.py, 3 seeds): observation day = climate input to the observation day, end of season = full-season TDD",
        sites=info, numbers_printed="site coordinates", outputs=C.file_info(paths), size_mm=[W, H], qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
