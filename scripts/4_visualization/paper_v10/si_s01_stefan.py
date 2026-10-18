"""Supplementary Fig. S1: the Stefan relation within a season and among years (content of v7_f2_problem_time.py c-d and v8_g5_stefan.py), paper palette.

a  Samoylov 2015: cumulative thawing degree-days (ERA5-Land monthly means, daily linear interpolation) and probed thaw depth (ALLena, PANGAEA 973813;
   daily mean +- s.d.) with the Stefan curve E sqrt(TDD(t)) (E = origin-through least squares of all probe values against sqrt(TDD) on the probe day).
b  Thaw depth against sqrt(TDD to the observation day) for Samoylov 2015 and Kurungnakh 2020 with origin-through fits (E in the legend).
c  Among-year deviations in cells observed in >= 4 years: (ALT - cell mean) against (sqrt(TDD) - cell mean), TDD to the observation day; Alaska as a
   density (hexagons), Lena Delta and Canada (NW) as points; origin-through regression lines per region (slopes in the legend).
d  Stefan coefficient E per region: source coefficient (least squares on the other two regions' labels) and recalibrated (training-fold labels, mean of
   the five spatial-block folds).
Inputs: data/raw/allena/PANGAEA_973813_ALLena_main.txt, data/raw/era5land/nh_monthly_2010-2024.nc, data/processed/v5 (labels, forcing via
        polar.testbed_v5.with_forcing(..., 'year_obs')), data/processed/v5/runs/core_v6/meta/<Region>__block__year_obs__F8TS__P1__-__lam-1.json.
Run: cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/si_s01_stefan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import v7_f2_problem_time as F2  # noqa: E402
import polar.testbed_v5 as T  # noqa: E402

STEM = "FigS1_stefan"
PROC = C.ROOT / "data" / "processed" / "v5"; META = PROC / "runs" / "core_v6" / "meta"
REG = C.REGIONS
SQ_TDD = r"$\sqrt{\mathrm{TDD}}$"; SQ_UNIT = r"$\sqrt{{}^{\circ}\mathrm{C{\cdot}d}}$"; E_UNIT = "cm/" + SQ_UNIT
SITE = {"Samoylov": ("#000000", "o", True), "Kurungnakh": ("#7F7F7F", "s", False)}
MONTH_DOY = {152: "Jun", 182: "Jul", 213: "Aug", 244: "Sep", 274: "Oct"}
X_DOY = (140.0, 278.0)
D_XLIM = (-8.0, 6.0); D_YLIM = 45.0


def site_series():
    out = []
    for name, lat, lon, yr, doy_s, z in F2._allena_series():
        dd, cum = F2._daily_tdd(lat, lon, yr); tdd = np.interp(doy_s, dd, cum)
        g = pd.DataFrame({"doy": doy_s, "tdd": tdd, "z": z}).groupby("doy").agg(tdd=("tdd", "first"), z=("z", "mean"), sd=("z", "std"), n=("z", "size"))
        E = float((np.sqrt(tdd) @ z) / (np.sqrt(tdd) @ np.sqrt(tdd)))
        out.append(dict(name=name, lat=lat, lon=lon, year=yr, doy=dd, cum=cum, g=g, E=E, n_probe=int(len(z))))
    return out


def e_table():
    rows = []
    for r in REG:
        m = json.loads((META / f"{r}__block__year_obs__F8TS__P1__-__lam-1.json").read_text())
        folds = {k: float(v["E"]) for k, v in m["coefs"].items()}
        rows.append(dict(region=r, E_source=float(m["E0"][r]), E_recal=float(np.mean(list(folds.values()))), E_recal_folds=folds, n=int(m["n"])))
    return rows


def main():
    C.setup()
    ser = site_series(); rows = e_table()
    d = T.with_forcing(T.load_v5(PROC, products=False), "year_obs")
    W, H = 180.0, 107.0
    fig = C.figure(W, H); src = {}
    # ---------------- a: Samoylov 2015, same time axis
    samo = [s for s in ser if s["name"] == "Samoylov"][0]; g = samo["g"]; doy, cum = samo["doy"], samo["cum"]; E = samo["E"]
    au = C.ax_mm(fig, 14.0, 6.0, 62.0, 13.0); al = C.ax_mm(fig, 14.0, 21.0, 62.0, 22.5)
    sel = (doy >= X_DOY[0]) & (doy <= X_DOY[1])
    au.plot(doy[sel], cum[sel], color="black", lw=C.LW_MAIN)
    cmax = float(cum[sel].max()); au.set_ylim(0, cmax * 1.12); au.set_yticks([0, 400, 800] if cmax > 700 else [0, 250, 500])
    au.set_ylabel("TDD (°C·d)")
    sd = g.sd.fillna(0)
    al.fill_between(g.index.values, g.z - sd, g.z + sd, color="#D9D9D9", lw=0, zorder=1)
    al.plot(doy[sel], E * np.sqrt(cum[sel]), color=C.METHOD["P1"], lw=C.LW_MAIN, zorder=3)
    al.plot(g.index.values, g.z, ls="none", marker="o", ms=2.0, mfc="black", mec="black", mew=0, zorder=4)
    zmax = max(float((g.z + sd).max()), float(E * np.sqrt(cmax)))
    al.set_ylim(np.ceil(zmax * 1.10 / 10) * 10, 0); al.set_yticks(np.arange(0, zmax * 1.10, 10))
    al.set_ylabel("Thaw depth (cm)")
    for a_ in (au, al):
        C.clean_axes(a_); a_.set_xlim(*X_DOY); a_.set_xticks(list(MONTH_DOY))
    au.set_xticklabels([]); al.set_xticklabels(list(MONTH_DOY.values())); al.set_xlabel("2015")
    al.legend([Line2D([], [], ls="none", marker="o", ms=2.0, mfc="black", mec="black"), Patch(facecolor="#D9D9D9", edgecolor="none"),
               Line2D([], [], color=C.METHOD["P1"], lw=C.LW_MAIN)],
              ["Probed thaw depth, daily mean", "± 1 s.d.", r"$E\sqrt{\mathrm{TDD}}$"], loc="upper right", bbox_to_anchor=(1.0, 1.0),
              fontsize=C.FS_SMALL, handlelength=1.4, labelspacing=0.3)
    au.text(0.02, 0.97, "Samoylov", transform=au.transAxes, ha="left", va="top", fontsize=C.FS_SMALL, color=C.TEXT)
    src["a"] = dict(site="Samoylov", year=2015, lat=samo["lat"], lon=samo["lon"], E=round(E, 4), n_probe=samo["n_probe"], n_days=int(len(g)),
                    doy_range=[int(g.index.min()), int(g.index.max())], tdd_max_window=round(cmax, 1))
    # ---------------- b: thaw depth vs sqrt(TDD)
    ab = C.ax_mm(fig, 96.0, 6.0, 80.0, 37.5); C.clean_axes(ab)
    xmax = 0.0; hs, labs = [], []
    for s_ in ser:
        g_ = s_["g"]; col, mk, filled = SITE[s_["name"]]; sd_ = g_.sd.fillna(0)
        ab.fill_between(np.sqrt(g_.tdd), g_.z - sd_, g_.z + sd_, color="#E3E3E3" if filled else "#EEEEEE", lw=0, zorder=1)
        ab.plot(np.sqrt(g_.tdd), g_.z, ls="none", marker=mk, ms=2.4, mfc=col if filled else "white", mec=col, mew=0.5, zorder=4)
        xmax = max(xmax, float(np.sqrt(g_.tdd.max())))
    xe = xmax * 1.12; xx = np.linspace(0, xe, 20)
    for s_ in ser:
        col, mk, filled = SITE[s_["name"]]
        ab.plot(xx, s_["E"] * xx, color=col, lw=C.LW_AUX + 0.2, ls="-" if filled else C.DASH, zorder=3)
        hs.append(Line2D([], [], color=col, lw=C.LW_AUX + 0.2, ls="-" if filled else C.DASH, marker=mk, ms=2.4, mfc=col if filled else "white", mec=col, mew=0.5))
        labs.append(f"{s_['name']} {s_['year']}, $E$ = {s_['E']:.2f} " + E_UNIT)
        src.setdefault("b", {})[f"{s_['name']}_{s_['year']}"] = dict(E=round(s_["E"], 4), n_probe=s_["n_probe"], n_days=int(len(s_["g"])))
    xt = np.ceil(xe / 10) * 10; ab.set_xlim(0, xt); ab.set_xticks(np.arange(0, xt + 1e-9, 10))
    ab.set_ylim(0, max(s_["E"] for s_ in ser) * xe * 1.10); ab.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ab.set_xlabel(f"{SQ_TDD} to observation day ({SQ_UNIT})"); ab.set_ylabel("Thaw depth (cm)")
    ab.legend(hs, labs, loc="upper left", fontsize=C.FS_SMALL, handlelength=2.0, labelspacing=0.35, bbox_to_anchor=(0.02, 1.0))
    # ---------------- c: among-year deviations (cells with >= 4 observed years)
    ac = C.ax_mm(fig, 14.0, 60.0, 82.0, 37.0); C.clean_axes(ac)
    n = d.groupby("loc_id").year.nunique(); m = d[d.loc_id.isin(n[n >= 4].index)].copy()
    gg = m.groupby("loc_id"); m["dx"] = m.e5_sqrt_tdd - gg.e5_sqrt_tdd.transform("mean"); m["dy"] = m.alt_cm - gg.alt_cm.transform("mean")
    inside = m.dx.between(*D_XLIM) & m.dy.between(-D_YLIM, D_YLIM)
    ak = m[(m.region == "Alaska") & inside]
    cmap = LinearSegmentedColormap.from_list("ak_grey", ["#EFEFEF", "#A8A8A8"])
    hb = ac.hexbin(ak.dx, ak.dy, gridsize=(24, 14), extent=(*D_XLIM, -D_YLIM, D_YLIM), cmap=cmap, norm=LogNorm(vmin=1), mincnt=1, linewidths=0.2,
                   edgecolors="face", zorder=1)
    ac.axhline(0, color="#BDBDBD", lw=0.4, zorder=2); ac.axvline(0, color="#BDBDBD", lw=0.4, zorder=2)
    for r in ("Canada", "Lena"):
        sr = m[(m.region == r) & inside]; name, col, mk = C.REGION[r]
        ac.scatter(sr.dx, sr.dy, s=5.5, marker=mk, facecolor=col, edgecolor="white", linewidths=0.25, zorder=4 + (r == "Lena"))
    fits = {}
    for r in REG:
        sr = m[m.region == r]; b = float((sr.dx @ sr.dy) / (sr.dx @ sr.dx))
        x0, x1 = float(np.percentile(sr.dx, 1)), float(np.percentile(sr.dx, 99)); fits[r] = (b, x0, x1)
        src.setdefault("c", {})[r] = dict(n_cells=int(sr.loc_id.nunique()), n_cell_years=int(len(sr)), n_shown=int(inside[sr.index].sum()), slope=round(b, 4),
                                          r=round(float(np.corrcoef(sr.dx, sr.dy)[0, 1]), 3), years=[int(sr.year.min()), int(sr.year.max())])
    for r in ("Canada", "Lena", "Alaska"):
        b, x0, x1 = fits[r]; ac.plot([x0, x1], [b * x0, b * x1], color=C.REGION[r][1], lw=1.1, zorder=8, solid_capstyle="round")
    ac.set_xlim(*D_XLIM); ac.set_ylim(-D_YLIM, D_YLIM + 30); ac.spines["left"].set_bounds(-D_YLIM, D_YLIM)
    ac.set_xticks(np.arange(-8, 6.1, 2)); ac.set_yticks(np.arange(-40, 40.1, 20))
    ac.set_xlabel(f"{SQ_TDD} − cell mean ({SQ_UNIT})"); ac.set_ylabel("ALT − cell mean (cm)")
    hs = [Line2D([], [], color=C.REGION["Alaska"][1], lw=1.1, marker="h", ms=4.0, mfc="#C8C8C8", mec="#C8C8C8")]
    hs += [Line2D([], [], color=C.REGION[r][1], lw=1.1, marker=C.REGION[r][2], ms=2.6, mfc=C.REGION[r][1], mec="white", mew=0.25) for r in ("Lena", "Canada")]
    leg = ac.legend(hs, [f"{C.REGION[r][0]}, slope {fits[r][0]:.2f}".replace("-", "−") for r in REG], loc="upper left", bbox_to_anchor=(0.01, 1.0),
                    fontsize=C.FS_SMALL, handlelength=2.0, labelspacing=0.3)
    cax = C.ax_mm(fig, 70.0, 61.5, 22.0, 1.6)
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.4); cb.minorticks_off()
    cmx = int(hb.get_array().max()); tk = [t for t in (1, 3, 10, 30, 100) if t <= cmx]; cb.set_ticks(tk); cb.set_ticklabels([str(t) for t in tk])
    cb.ax.tick_params(labelsize=C.FS_SMALL, length=1.4, width=0.4, pad=1.0)
    cax.set_title("Alaska cell-years", fontsize=C.FS_SMALL, pad=1.5)
    src["c_def"] = dict(min_observed_years=4, x="sqrt(TDD to the observation day) minus the cell mean", y="ALT minus the cell mean",
                        n_outside_axes=int((~inside).sum()), xlim=list(D_XLIM), ylim=[-D_YLIM, D_YLIM], slope_unit="cm per sqrt(degC d)")
    # ---------------- d: E per region
    ad = C.ax_mm(fig, 124.0, 60.0, 52.0, 30.0); C.clean_axes(ad, left=False)
    ys = [0, 1, 2]
    for yy, rr in zip(ys, rows):
        ad.plot([rr["E_source"], rr["E_recal"]], [yy, yy], color="#C8C8C8", lw=1.0, zorder=1)
        ad.plot([rr["E_source"]], [yy], ls="none", marker="o", ms=3.6, mfc=C.METHOD["P0"], mec=C.METHOD["P0"], zorder=3)
        ad.plot([rr["E_recal"]], [yy], ls="none", marker="o", ms=3.6, mfc=C.METHOD["P1"], mec=C.METHOD["P1"], zorder=3)
    ad.set_yticks(ys); ad.set_yticklabels([C.REGION[r["region"]][0] for r in rows]); ad.set_ylim(2.6, -0.6); ad.tick_params(axis="y", length=0, pad=2)
    vals = [v for r in rows for v in (r["E_source"], r["E_recal"])]
    ad.set_xlim(np.floor(min(vals) * 10 - 1) / 10, np.ceil(max(vals) * 10 + 1) / 10); ad.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ad.set_xlabel(r"$E$ (" + E_UNIT + ")")
    ad.legend([Line2D([], [], ls="none", marker="o", ms=3.6, mfc=C.METHOD["P0"], mec=C.METHOD["P0"]),
               Line2D([], [], ls="none", marker="o", ms=3.6, mfc=C.METHOD["P1"], mec=C.METHOD["P1"])], ["Source coefficient", "Recalibrated"],
              loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=C.FS_SMALL, handlelength=0.8, columnspacing=1.2)
    src["d"] = rows
    C.letter(fig, 0.5, 0.5, "a"); C.letter(fig, 84.0, 0.5, "b"); C.letter(fig, 0.5, 54.0, "c"); C.letter(fig, 102.0, 54.0, "d")
    rep = C.qa_figure(fig, ignore_ink_for=["Samoylov"]); C.print_qa(STEM, rep)
    paths = C.export_si(fig, STEM); plt.close(fig)
    C.update_si_json("S1", dict(
        title="Stefan relation within a season and among years", stem=STEM, script="scripts/4_visualization/paper_v10/si_s01_stefan.py",
        inputs=["data/raw/allena/PANGAEA_973813_ALLena_main.txt", "data/raw/era5land/nh_monthly_2010-2024.nc", "data/processed/v5/labels_cell_year_v5.csv",
                "data/processed/v5/forcing_year.csv (via polar.testbed_v5.with_forcing year_obs)", "data/processed/v5/runs/core_v6/meta/<Region>__block__year_obs__F8TS__P1__-__lam-1.json"],
        years=dict(a="2015 (Samoylov)", b="Samoylov 2015, Kurungnakh 2020", c="label years of cells observed in >= 4 years (per region in values.c)",
                   d="all label cell-years (Alaska 2010-2024, Lena Delta 2010-2022, Canada (NW) 2010-2024)"),
        values=src, numbers_printed=dict(b={k: f"E = {v['E']:.2f}" for k, v in src["b"].items()}, c={r: f"slope {fits[r][0]:.2f}" for r in REG}),
        outputs=C.file_info(paths), size_mm=[W, H], qa=rep))
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
