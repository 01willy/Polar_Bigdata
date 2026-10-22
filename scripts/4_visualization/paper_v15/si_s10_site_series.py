"""Supplementary Fig. S10 (v15.2, 2026-10-22): yearly series at the site with the most observed years in each of the three regions.

Per region: observed ALT (open circles joined by a thin line), physics + residual ML and recalibrated Stefan with the TDD to 1 September (lines, as
the yearly maps), physics + residual ML at the observation day (squares), and the Wei et al. 2026 and Liu et al. 2024 values of each year where the
product has a value at the cell. The warmest and coldest thaw seasons of the region (regional mean TDD, Fig. 7d) are shaded so that the spikes of
the observed series can be read against the climate of the year. Predictions are spatial-block out-of-fold (v5_site_series.py, 3 seeds).
Inputs: data/processed/v5/site_series/site_series_year_obs_d244.csv.gz, products_cell_year_v5.csv, labels_cell_year_v5.csv, Fig7_maps.json (d).
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper15 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 python3 scripts/4_visualization/paper_v15/si_s10_site_series.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c13_common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

STEM = "FigS10_site_series"; PROC = C.ROOT / "data" / "processed" / "v5"; YR = (2010, 2024); REGIONS = ["Alaska", "Lena", "Canada"]
PN = {"wei": "Wei et al. 2026", "liu": "Liu et al. 2024"}


def fmt_coord(lat, lon): return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'} {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


def main():
    C.setup()
    st = pd.read_csv(PROC / "site_series" / "site_series_year_obs_d244.csv.gz")
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv", usecols=["loc_id", "region", "lat", "lon", "year", "in_forcing", "n_obs"]); lab = lab[lab.in_forcing == 1]
    pr = pd.read_csv(PROC / "products_cell_year_v5.csv", usecols=["loc_id", "product", "year", "value_cm"])
    f7 = json.loads((C.ROOT / "outputs/figures/paper/v15/source_data/Fig7_maps.json").read_text())["d"]["series"]
    o_ = st[st.observed == 1]; ny = o_.groupby(["region", "loc_id"]).year.nunique().rename("n_years").reset_index()
    ny = ny.merge(lab.groupby("loc_id").n_obs.sum().rename("n_obs"), left_on="loc_id", right_index=True, how="left").fillna({"n_obs": 0})
    picks = {r: int(ny[ny.region == r].sort_values(["n_years", "n_obs", "loc_id"], ascending=[False, False, True]).loc_id.iloc[0]) for r in REGIONS}
    W, H = 180.0, 68.0; fig = C.figure(W, H); xs = [13.0, 71.0, 129.0]; w, h, y0 = 46.0, 38.0, 18.0
    R1, P1 = C.METHOD["R1"], C.METHOD["P1"]; PC = {"wei": C.PRODUCT["wei"][1], "liu": C.PRODUCT["liu"][1]}; info = {}
    for k, r in enumerate(REGIONS):
        ax = C.ax_mm(fig, xs[k], y0, w, h); C.clean_axes(ax); lid = picks[r]
        g = st[(st.region == r) & (st.loc_id == lid) & st.year.between(*YR)].sort_values("year"); o = g[g.observed == 1]
        ta = f7[r]["tdd_anom"]; warm = int(max(ta, key=ta.get)); cold = int(min(ta, key=ta.get))
        for yy, lab_ in ((warm, "warmest"), (cold, "coldest")):
            ax.axvspan(yy - 0.5, yy + 0.5, color="#EFEFEF", lw=0, zorder=0); ax.text(yy, 0.985, lab_, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=C.FS_SMALL, color=C.AUX, rotation=90)
        drawn = [o.y.values, o.r1_obs.values, g.p1_eos.values, g.r1_eos.values]; have = []
        for p in ("wei", "liu"):
            sp = pr[(pr.loc_id == lid) & (pr["product"] == p) & pr.year.between(*YR)].dropna(subset=["value_cm"]).sort_values("year")
            if len(sp):
                ax.plot(sp.year, sp.value_cm, color=PC[p], lw=C.LW_MAIN, ls=("-" if p == "wei" else C.DASH), marker="o", ms=1.8, zorder=2); drawn.append(sp.value_cm.values); have.append(p)
                info.setdefault(r, {})[f"{p}_years"] = [int(sp.year.min()), int(sp.year.max())]
        ax.plot(g.year, g.p1_eos, color=P1, lw=C.LW_MAIN, zorder=3); ax.plot(g.year, g.r1_eos, color=R1, lw=C.LW_MAIN, zorder=4)
        ax.plot(o.year, o.r1_obs, ls="none", marker="s", ms=2.6, mfc=R1, mec="white", mew=0.4, zorder=5)
        ax.plot(o.year, o.y, color=C.TEXT, lw=0.6, zorder=5); ax.plot(o.year, o.y, ls="none", marker="o", ms=3.4, mfc="white", mec=C.TEXT, mew=0.7, zorder=6)
        ya = np.concatenate(drawn); lo, hi = np.nanmin(ya), np.nanmax(ya); pad = 0.08 * (hi - lo) + 1.0
        ax.set_ylim(np.floor((lo - pad) / 5) * 5, np.ceil((hi + pad) / 5) * 5); ax.yaxis.set_major_locator(mticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlim(YR[0] - 0.6, YR[1] + 0.6); ax.set_xticks([2010, 2014, 2018, 2022]); ax.xaxis.set_minor_locator(mticker.MultipleLocator(1)); ax.tick_params(which="minor", length=1.0, width=0.4)
        ax.set_xlabel("Year")
        if k == 0: ax.set_ylabel("ALT (cm)")
        la = float(lab[lab.loc_id == lid].lat.iloc[0]); lo_ = float(lab[lab.loc_id == lid].lon.iloc[0])
        C.text_mm(fig, xs[k], y0 - 4.6, f"{C.REGION[r][0]}, {fmt_coord(la, lo_)}", fontsize=C.FS, ha="left", va="bottom")
        C.text_mm(fig, xs[k], y0 - 1.6, f"{len(o)} observed years" + ("; no product value at this cell" if not have else ""), fontsize=C.FS_SMALL, ha="left", va="bottom", color=C.AUX)
        info.setdefault(r, {}).update(loc_id=int(lid), lat=round(la, 4), lon=round(lo_, 4), observed_years=[int(y) for y in o.year], n_observed_years=int(len(o)), products_drawn=have,
                                      warmest_year=warm, coldest_year=cold, observed_cm={str(int(y)): round(float(v), 1) for y, v in zip(o.year, o.y)},
                                      r1_obs_cm={str(int(y)): round(float(v), 1) for y, v in zip(o.year, o.r1_obs)}, r1_eos_cm={str(int(y)): round(float(v), 1) for y, v in zip(g.year, g.r1_eos)})
        C.letter(fig, xs[k] - 11.0, y0 - 4.9, "abc"[k])
    hs = [Line2D([], [], color=C.TEXT, lw=0.6, marker="o", ms=3.4, mfc="white", mec=C.TEXT, mew=0.7), Line2D([], [], ls="none", marker="s", ms=2.6, mfc=R1, mec="white", mew=0.4),
          Line2D([], [], color=R1, lw=C.LW_MAIN), Line2D([], [], color=P1, lw=C.LW_MAIN), Line2D([], [], color=PC["wei"], lw=C.LW_MAIN, marker="o", ms=1.8),
          Line2D([], [], color=PC["liu"], lw=C.LW_MAIN, ls=C.DASH, marker="o", ms=1.8), Patch(facecolor="#EFEFEF")]
    labs = ["Observed", "Physics + residual ML, observation day", "Physics + residual ML, 1 September", "Stefan (recalibrated), 1 September", PN["wei"], PN["liu"], "Warmest / coldest thaw season of the region"]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(xs[0] / W, 1 - 1.0 / H), ncol=4, handlelength=1.8, columnspacing=1.4, labelspacing=0.3, fontsize=C.FS_TICK, frameon=False, borderaxespad=0)
    rep = C.qa_figure(fig, ignore_ink_for=["warmest", "coldest"]); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S10", dict(title="Long-record sites", stem=STEM, script="scripts/4_visualization/paper_v15/si_s10_site_series.py", version="v15.2 (2026-10-22): three regions, observed joined, warm/cold seasons shaded",
                                 inputs=["data/processed/v5/site_series/site_series_year_obs_d244.csv.gz", "data/processed/v5/products_cell_year_v5.csv", "data/processed/v5/labels_cell_year_v5.csv", "Fig7_maps.json d"],
                                 definition="site with the most observed years per region (ties: most probe records, lowest loc_id); predictions = spatial-block out-of-fold (3 seeds): observation day = climate input to the observation day, "
                                            "1 September = TDD to day 244 (as the yearly maps); shaded years = warmest and coldest regional mean TDD of 2011-2024",
                                 sites=info, outputs=C.file_info(paths), size_mm=[W, H], pdf_fonts=C.pdf_fonts(paths[0]), qa=rep))
    for p in paths: print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
