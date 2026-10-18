"""v6 논문 그림(figures/figure_spec_v6.json). 실행: python3 scripts/4_visualization/v6_figs.py --fig 1 [--slide]"""
from __future__ import annotations
import argparse, calendar, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe
PROC = ROOT / "data" / "processed" / "v5"; RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "figures" / "paper" / "v6"
REG = ["Alaska", "Lena", "Canada"]
BOX = {"Alaska": (-168, -141, 59, 71.5), "Lena": (123.3, 130.1, 71.5, 73.6), "Canada": (-123, -112, 60.5, 69.5)}
RUN = PROC / "runs" / "core_v6"


def export(fig, stem):
    return S.export(fig, stem, outdir=OUT)


# ================================================================ F1 연구 지역·관측·기간
def f1():
    import cartopy.crs as ccrs, cartopy.feature as cfeature, xarray as xr
    from matplotlib.colors import LinearSegmentedColormap
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    lab["obs_month_last"] = lab.obs_months.astype(str).map(lambda v: 9 if v == "eos" else max(int(x) for x in v.split(";") if x.strip().isdigit()))
    cells = lab.groupby("loc_id").agg(lat=("lat", "first"), lon=("lon", "first"), region=("region", "first"), n_years=("year", "nunique")).reset_index()
    pfr = xr.open_dataset(ROOT / "data" / "processed" / "cci_pfr_mean_1997_2021.nc"); P = pfr[[v for v in pfr.data_vars][0]]
    latn, lonn = [d for d in P.dims if "lat" in d][0], [d for d in P.dims if "lon" in d][0]
    Ps = P.isel({latn: slice(None, None, 4), lonn: slice(None, None, 4)})
    H = 128.0; fig = S.figure(S.W2, H)
    pcm = LinearSegmentedColormap.from_list("pfr", ["#ffffff", "#e3eaf3", "#b6c8de", "#7f9fc4", "#4a6fa0", "#24456f"])
    # a: 북극 지도(왼쪽, 폭 36 %)
    proj = ccrs.NorthPolarStereo(central_longitude=-100)
    axm = fig.add_axes([0.035, 0.40, 0.33, 0.57], projection=proj); axm.set_extent([-180, 180, 52, 90], ccrs.PlateCarree())
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f4f4", edgecolor="none", zorder=0)
    im = axm.pcolormesh(Ps[lonn].values, Ps[latn].values, np.clip(Ps.values, 0, 100), transform=ccrs.PlateCarree(), cmap=pcm, vmin=0, vmax=100, shading="auto", zorder=1, rasterized=True)
    axm.coastlines("50m", linewidth=0.35, color="#8c8c8c", zorder=2)
    axm.gridlines(draw_labels=False, linewidth=0.3, color="#d8d8d8", xlocs=range(-180, 181, 60), ylocs=[60, 70, 80])
    for r in REG:
        x0, x1, y0, y1 = BOX[r]
        lons = np.r_[np.linspace(x0, x1, 40), np.full(40, x1), np.linspace(x1, x0, 40), np.full(40, x0)]
        lats = np.r_[np.full(40, y0), np.linspace(y0, y1, 40), np.full(40, y1), np.linspace(y1, y0, 40)]
        axm.plot(lons, lats, transform=ccrs.PlateCarree(), color=S.REGION_HEX[r], lw=1.1, zorder=4)
    for r, (lon, lat, ha) in {"Alaska": (-160, 54.5, "center"), "Lena": (112, 67.5, "center"), "Canada": (-98, 57.5, "center")}.items():
        axm.text(lon, lat, f"{S.REGION_EN[r]}\n{int((cells.region == r).sum()):,} cells", transform=ccrs.PlateCarree(), ha=ha, va="top", fontsize=6.5, color=S.REGION_HEX[r],
                 path_effects=[pe.withStroke(linewidth=1.8, foreground="white")], zorder=5)
    axm.spines["geo"].set_linewidth(0.5); axm.spines["geo"].set_edgecolor("#8c8c8c")
    cax = fig.add_axes([0.07, 0.385, 0.26, 0.012]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Permafrost probability, ESA CCI 1997–2021 mean (%)", fontsize=6); cb.ax.tick_params(labelsize=6, length=2)
    S.panel_letter(axm, "a", x=0.0, y=1.0); S.register("a", cax)
    # b–d: 지역 지도(같은 높이, 폭은 종횡비에 맞춤)
    EXT = {"Alaska": (-168.5, -140.5, 58.8, 71.6), "Lena": (122.6, 130.6, 71.3, 74.0), "Canada": (-128, -108, 60.3, 69.6)}
    def w_of(e): return (e[1] - e[0]) * np.cos(np.radians((e[2] + e[3]) / 2)) / (e[3] - e[2])
    ws = np.array([w_of(EXT[r]) for r in REG]); avail = 0.535; gap = 0.06; hh = 0.29
    widths = ws / ws.sum() * (avail - 2 * gap)
    x = 0.455
    for j, r in enumerate(REG):
        e = EXT[r]; w = float(widths[j])
        ax = fig.add_axes([x, 0.64, w, hh], projection=ccrs.PlateCarree()); ax.set_extent(list(e), ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f4f4f4", edgecolor="none"); ax.coastlines("10m", linewidth=0.35, color="#8c8c8c")
        ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="#bcd2e8", linewidth=0.3); ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#dbe7f3", edgecolor="none")
        c = cells[cells.region == r]
        ax.scatter(c.lon, c.lat, s=0.5, c=S.REGION_HEX[r], lw=0, transform=ccrs.PlateCarree(), rasterized=True, zorder=3, alpha=0.85)
        m = c[c.n_years >= 4]
        ax.scatter(m.lon, m.lat, s=11, marker="*", c="#D55E00", lw=0.3, edgecolor="black", transform=ccrs.PlateCarree(), zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#d8d8d8", xlocs=np.arange(-180, 181, 5 if r != "Lena" else 2), ylocs=np.arange(50, 90, 2 if r != "Lena" else 1))
        gl.top_labels = gl.right_labels = False; gl.xlabel_style = {"size": 5.5}; gl.ylabel_style = {"size": 5.5}
        ax.set_title(f"{S.REGION_EN[r]}: {len(c):,} cells\n{int((c.n_years >= 4).sum())} multi-year sites (stars)", fontsize=5.5, loc="left", pad=2)
        ax.spines["geo"].set_linewidth(0.5)
        S.panel_letter(ax, "bcd"[j], x=-0.06, y=1.02)
        x += w + gap
    # e: 연도별 셀-연도 수
    x = 0.465; w3 = (0.525 - 2 * 0.05) / 3
    for j, r in enumerate(REG):
        ax = fig.add_axes([x + j * (w3 + 0.05), 0.40, w3, 0.19])
        years = np.arange(2010, 2025); cnt = lab[lab.region == r].groupby("year").size().reindex(years, fill_value=0)
        ax.bar(years, cnt.values, width=0.75, color=S.REGION_HEX[r], lw=0)
        ax.set_xlim(2009.3, 2024.7); ax.set_xticks([2010, 2014, 2018, 2022]); ax.set_ylim(0, cnt.max() * 1.3)
        if j == 0:
            ax.set_ylabel("Cell-years", labelpad=2)
        mo = lab[lab.region == r].obs_month_last.value_counts(normalize=True).reindex([8, 9, 10], fill_value=0)
        ax.text(0.03, 0.96, f"{S.REGION_EN[r]}\nobs. month Aug {mo[8]:.0%}, Sep {mo[9]:.0%}, Oct {mo[10]:.0%}", transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=S.NEUTRAL["text_aux"])
        if j == 0:
            S.panel_letter(ax, "e")
    # f: 기간 띠
    ax = fig.add_axes([0.20, 0.07, 0.78, 0.24])
    rows = [("Labels: Alaska / Lena Delta / Canada (NW)", 2010, 2024, S.STATUS["label"]), ("ERA5-Land forcing (yearly; TDD to observation day)", 2010, 2024, S.STATUS["climate"]),
            ("MODIS vegetation and snow", 2015, 2021, S.STATUS["satellite"]), ("ABoVE SAR ALT (Alaska, 2017)", 2017, 2017, S.STATUS["satellite"]),
            ("ESA CCI v4 / v5", 1997, 2023, S.STATUS["product"]), ("Wei 2026", 2000, 2024, S.STATUS["product"]), ("Liu 2024", 2003, 2020, S.STATUS["product"]),
            ("Yi–Kimball 2019 (Alaska)", 2001, 2015, S.STATUS["product"]), ("Aalto 2018 (2000–2014 mean)", 2000, 2014, S.STATUS["product"]), ("Terrain, soil, land cover (static)", 1995, 2025, S.STATUS["terrain"])]
    for j, (name, y0, y1, col) in enumerate(rows[::-1]):
        if name.startswith("Terrain"):
            ax.hlines(j, y0, y1 + 1, color=col, lw=1.0, ls=(0, (2, 2)))
        elif y0 == y1:
            ax.plot([y0 + 0.5], [j], marker="s", ms=3, color=col, lw=0)
        else:
            ax.barh(j, y1 - y0 + 1, left=y0, height=0.55, color=col, lw=0)
    ax.axvspan(2010, 2025, color=S.NEUTRAL["equiv_band"], zorder=0, lw=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_xlim(1995, 2026); ax.set_xlabel("Year"); ax.text(2017.5, len(rows) - 0.35, "analysis window 2010–2024", ha="center", va="bottom", fontsize=6, color=S.NEUTRAL["text_aux"])
    ax.set_ylim(-0.7, len(rows) + 0.5); S.panel_letter(ax, "f", x=-0.26)
    S.register("ae", *S.PANELS.get("a", []), *S.PANELS.get("e", []))
    S.register("bcdf", *[a for k in "bcdf" for a in S.PANELS.get(k, [])])
    return export(fig, "F1_study_area")


# ================================================================ F2 실자료 융해 진행 + 설계
_GRID = None


def _grid_of(lat, lon):
    """라벨 셀 표(forcing_year)에서 가장 가까운 셀의 ERA5-Land 사용 격자(육지 폴백 포함) 좌표."""
    global _GRID
    if _GRID is None:
        _GRID = pd.read_csv(PROC / "forcing_clim.csv", usecols=["loc_id", "e5_grid_lat", "e5_grid_lon"]).merge(
            pd.read_csv(PROC / "static_v5.csv", usecols=["loc_id", "lat", "lon"]), on="loc_id")
    j = ((_GRID.lat - lat) ** 2 + (_GRID.lon - lon) ** 2).idxmin()
    return float(_GRID.e5_grid_lat[j]), float(_GRID.e5_grid_lon[j])


def _daily_tdd(lat, lon, year):
    """ERA5-Land 월평균에서 일별 누적 TDD(월 누적값의 선형 보간). 격자는 라벨 셀과 같은 육지 폴백 규칙. 반환 (doy, 누적 TDD)."""
    import xarray as xr
    ds = xr.open_dataset(RAW / "era5land" / "nh_monthly_2010-2024.nc")
    tn = "valid_time" if "valid_time" in ds.coords else "time"
    glat, glon = _grid_of(lat, lon)
    t = ds["t2m"].sel(latitude=glat, longitude=glon, method="nearest")
    yrs = t[tn].dt.year.values; mos = t[tn].dt.month.values
    T = t.values[yrs == year] - 273.15
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    cum = np.cumsum(np.clip(T, 0, None) * days); starts = np.r_[0, np.cumsum(days)]
    doy = np.arange(1, starts[-1] + 1); out = np.interp(doy, np.r_[0, starts[1:]], np.r_[0, cum])
    return doy, out


def f2():
    fig = S.figure(S.W2, 92)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38, left=0.06, right=0.99, top=0.9, bottom=0.17, width_ratios=[1.1, 1.0, 1.0])
    # a 계절 안 융해 깊이 대 관측일 누적 TDD(실자료: 레나 사모일로프 2015, 알래스카 MegaSpot 2016·Rolling Hills 2017·Lins Lake 2019)
    ax = fig.add_subplot(gs[0, 0])
    series = []
    al = pd.read_csv(RAW / "allena" / "PANGAEA_973813_ALLena_main.txt", sep="\t", skiprows=437, low_memory=False)
    col = lambda p: [c for c in al.columns if c.startswith(p)][0]
    al["dt"] = pd.to_datetime(al[col("Date/Time (")]); al = al[(al[col("QF TD")] != 2) & al[col("Thaw depth mean [cm]")].notna()]
    al["year"] = al.dt.dt.year; al["doy"] = al.dt.dt.dayofyear
    s = al[(al.year == 2015) & (al[col("Latitude (")].round(3) == 72.374) & (al[col("Longitude (")].round(3) == 126.497)]
    series.append(("Lena Delta, Samoylov 2015", 72.374, 126.497, 2015, s.doy.values, s[col("Thaw depth mean [cm]")].values, S.REGION_HEX["Lena"]))
    ab = pd.read_csv(RAW / "above" / "ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv", usecols=["site_name", "latitude", "longitude", "date", "ALT", "ALT_instrument"], low_memory=False)
    ab = ab[(ab.ALT > 0) & (ab.ALT < 300) & (ab.ALT_instrument == "Probe")]; ab["dt"] = pd.to_datetime(ab.date, errors="coerce"); ab = ab.dropna(subset=["dt"]); ab["year"] = ab.dt.dt.year; ab["doy"] = ab.dt.dt.dayofyear
    ab["month"] = ab.dt.dt.month
    g2 = ab[ab.year.between(2010, 2024)].groupby(["site_name", "year"]).agg(n=("doy", "size"), nm=("month", "nunique"), lat=("latitude", "mean"), lon=("longitude", "mean")).reset_index()
    g2 = g2[(g2.nm >= 3) & (g2.n >= 20)].sort_values("n", ascending=False).head(2)
    # 레나 2020 'channel' 캠페인(6–9월, 4개월)도 더한다
    s20 = al[(al.year == 2020) & (al[col("Latitude (")].round(3) == 72.373) & (al[col("Longitude (")].round(3) == 126.495)]
    if len(s20) >= 6:
        series.append(("Lena Delta, Kurungnakh 2020", 72.373, 126.495, 2020, s20.doy.values, s20[col("Thaw depth mean [cm]")].values, "#2E7BBF"))
    for (site, yr), colr in zip(zip(g2.site_name, g2.year), (S.REGION_HEX["Alaska"], "#6A5ACD", "#2E8B57")):
        s = ab[(ab.site_name == site) & (ab.year == yr)]
        series.append((f"Alaska, {site.replace('_', ' ')} {yr}", float(s.latitude.mean()), float(s.longitude.mean()), yr, s.doy.values, s.ALT.values, colr))
    for name, lat, lon, yr, doy, z, colr in series:
        d, cum = _daily_tdd(lat, lon, yr)
        tdd = np.interp(doy, d, cum)
        g = pd.DataFrame({"tdd": tdd, "z": z}).groupby(pd.cut(pd.Series(tdd), 14), observed=True).agg(tdd=("tdd", "mean"), z=("z", "mean"), n=("z", "size"), sd=("z", "std")).dropna()
        ax.errorbar(np.sqrt(g.tdd), g.z, yerr=g.sd.fillna(0), fmt="o", ms=2.6, lw=0.6, color=colr, capsize=0, label=name, alpha=0.95)
        E = float((np.sqrt(tdd) @ z) / (np.sqrt(tdd) @ np.sqrt(tdd)))
        xx = np.linspace(0, np.sqrt(tdd.max()) * 1.05, 50); ax.plot(xx, E * xx, color=colr, lw=0.9, alpha=0.9)
        ax.text(xx[-1], E * xx[-1], f" E {E:.2f}", fontsize=5.5, color=colr, va="center")
    ax.set_xlabel("√(cumulative TDD to observation day) (√°C·d)"); ax.set_ylabel("Observed thaw depth (cm)")
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.0); S.panel_letter(ax, "a")
    # b 다년 지점: 관측 ALT 대 √TDD(관측일), 지점별 색·선
    ax = fig.add_subplot(gs[0, 1])
    import polar.testbed_v5 as T
    df = T.load_v5(PROC, products=False); d = T.with_forcing(df, "year_obs")
    multi = d.groupby("loc_id").year.nunique(); top = {}
    for r in REG:
        ids = multi[multi.index.isin(d[d.region == r].loc_id.unique())]; ids = ids[ids >= 6].index
        top[r] = list(ids[:4])
    k = 0
    for r in REG:
        for i, lid in enumerate(top[r]):
            s = d[d.loc_id == lid].sort_values("year")
            ax.plot(s.e5_sqrt_tdd, s.alt_cm, "-o", ms=2.2, lw=0.6, color=S.REGION_HEX[r], alpha=0.75 - 0.12 * i, label=S.REGION_EN[r] if i == 0 else None)
            k += 1
    ax.set_xlabel("√TDD to observation day, that year (√°C·d)"); ax.set_ylabel("Observed ALT (cm)")
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.2); ax.text(0.98, 0.03, "lines connect years at one cell", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=S.NEUTRAL["text_aux"])
    S.panel_letter(ax, "b")
    # c 검증 사다리 도식(알래스카 실제 셀): 무작위·블록·연도 겹 색
    import cartopy.crs as ccrs
    sub = gs[0, 2].subgridspec(3, 1, hspace=0.12)
    ak = d[d.region == "Alaska"].drop_duplicates("loc_id")
    rng = np.random.RandomState(0); f_rand = rng.permutation(len(ak)) % 5
    f_blk = T._balanced_group_folds(ak.block.values, 5, 0)
    yr_f = (ak.merge(d.groupby("loc_id").year.agg(lambda s: s.mode().iloc[0]).rename("ymode"), left_on="loc_id", right_index=True).ymode.values)
    cm = plt.get_cmap("tab10")
    for i, (name, fold) in enumerate((("Random split", f_rand), ("Spatial block (0.5°)", f_blk), ("Year hold-out (cell's main year)", yr_f))):
        ax = fig.add_subplot(sub[i, 0], projection=ccrs.PlateCarree()); ax.set_extent([-168, -141, 59, 71.5])
        ax.coastlines("50m", linewidth=0.3, color="#9a9a9a")
        if name.startswith("Year"):
            vals = (fold - 2010) / 14.0; ax.scatter(ak.lon, ak.lat, s=0.5, c=plt.get_cmap("viridis")(vals), lw=0, rasterized=True)
        else:
            ax.scatter(ak.lon, ak.lat, s=0.5, c=cm(fold % 10), lw=0, rasterized=True)
        ax.text(0.02, 0.95, name, transform=ax.transAxes, va="top", fontsize=5.5, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        ax.spines["geo"].set_linewidth(0.4)
        if i == 0:
            S.panel_letter(ax, "c", x=-0.02, y=1.05)
    S.register("ab", *S.PANELS.get("a", []), *S.PANELS.get("b", []))
    S.register("cd", *S.PANELS.get("c", []), *S.PANELS.get("d", []))
    return export(fig, "F2_thaw_and_design")


FIGS = {1: f1, 2: f2}


# ================================================================ F3 시간 정합 + 검증 사다리 + 연도 변화
def _run():
    for t in ("core_v6", "core_cb"):
        if (PROC / "runs" / t / "summary_metrics.csv").exists():
            return PROC / "runs" / t, ("year_obs" if t == "core_v6" else "year")
    raise SystemExit("no run")


METH = ["P0", "P1", "Ku", "D0", "F1", "R1"]
METH_EN = {"P0": "Stefan, source coeff.", "P1": "Stefan, recalibrated", "Ku": "Kudryavtsev", "D0": "Direct ML", "F1": "Physics-input ML", "D1": "Physics pseudo-labels", "R1": "Physics + residual ML"}


def f3():
    run, forcing = _run()
    sm = pd.read_csv(run / "summary_metrics.csv"); sm = sm[(sm.featset == "F8TS") & ((sm.method != "R1") | (sm.lam == 0.5)) & (sm.learner.isin(["catboost", "-"]))]
    df = pd.read_csv(run / "deltas_forcing.csv"); df = df[(df.featset == "F8TS") & ((df.method != "R1") | (df.lam == 0.5)) & (df.learner.isin(["catboost", "-"]))]
    pair = f"{forcing}:clim" if "pair" in df else None
    if pair:
        df = df[df.pair == pair]
    fig = S.figure(S.W2, 112)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.55, left=0.06, right=0.99, top=0.88, bottom=0.12, width_ratios=[1, 1.55, 0.9])
    # a 강제력 효과(블록 CV)
    ax = fig.add_subplot(gs[0, 0])
    d = df[df.scheme == "block"]
    for i, r in enumerate(REG):
        s = d[d.region == r].set_index("method").reindex(METH).dropna(subset=["delta"])
        x = np.array([METH.index(m) for m in s.index]) + (i - 1) * 0.24
        ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0, label=S.REGION_EN[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_xticks(range(len(METH))); ax.set_xticklabels(METH)
    ax.set_ylabel("RMSE change (cm)\nyear-specific − climatological forcing"); ax.legend(loc="upper right", handlelength=1.0, fontsize=6); S.panel_letter(ax, "a")
    # b 검증 사다리(세 지역 소패널)
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.55)
    schemes = ["random", "block", "year", "block_year", "region"]
    for j, r in enumerate(REG):
        ax = fig.add_subplot(sub[0, j])
        vals = {m: sm[(sm.region == r) & (sm.forcing == forcing) & (sm.method == m)].set_index("scheme").reindex(schemes).rmse.values for m in METH}
        base = np.nanmax([vals[m][:4] for m in METH]); cap = base * 1.25
        nclip = {}
        for m in METH:
            v = vals[m]; vc = np.minimum(v, cap)
            ax.plot(range(len(schemes)), vc, marker="o", ms=2.3, lw=0.9, ls=S.METHOD_LS.get(m, "-"), color=S.METHOD_HEX[m], label=METH_EN[m], clip_on=False)
            for xi, (a_, b_) in enumerate(zip(v, vc)):
                if np.isfinite(a_) and a_ > b_:
                    k_ = nclip.get(xi, 0); nclip[xi] = k_ + 1
                    ax.annotate(f"{a_:.0f}", (xi, b_), xytext=(-6 + 12 * k_, 3), textcoords="offset points", ha="center", va="bottom", fontsize=5, color=S.METHOD_HEX[m])
        ax.set_ylim(None, cap * 1.08)
        ax.set_xticks(range(len(schemes))); ax.set_xticklabels(["rand", "block", "year", "blk×yr", "region"], rotation=40, ha="right", fontsize=5.5)
        ax.text(0.03, 0.97, S.REGION_EN[r], transform=ax.transAxes, va="top", ha="left", fontsize=6)
        if j == 0:
            ax.set_ylabel("RMSE (cm)"); S.panel_letter(ax, "b")
        else:
            S.register("b", ax)
        if j == 1:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), fontsize=5.5, handlelength=1.4, ncol=3, columnspacing=0.8, frameon=False)
    # c 연간 편차 상관(연도 홀드아웃)
    ax = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(REG):
        s = sm[(sm.region == r) & (sm.forcing == forcing) & (sm.scheme == "year")].set_index("method").reindex(METH)
        ax.plot(np.arange(len(METH)) + (i - 1) * 0.24, s.r_anom.values, "o", ms=3.0, color=S.REGION_HEX[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_xticks(range(len(METH))); ax.set_xticklabels(METH)
    ax.set_ylabel("Interannual anomaly correlation\n(year hold-out)"); S.panel_letter(ax, "c")
    # d 다년 셀 Δpred 대 Δobs(R1, 세 지역 산점)
    yw = run / "yearwise"
    if (yw / "multiyear_changes.csv").exists():
        ch = pd.read_csv(yw / "multiyear_changes.csv")
    ss = ROOT / "data" / "processed" / "v5" / "site_series" / f"site_series_{forcing}.csv.gz"
    ax = fig.add_subplot(gs[1, 0:2])
    if ss.exists():
        st = pd.read_csv(ss); st = st[st.observed == 1].sort_values(["loc_id", "year"])
        for i, r in enumerate(REG):
            g = st[st.region == r]
            dobs, dpred = [], []
            for lid, q in g.groupby("loc_id"):
                if len(q) >= 2:
                    dobs.extend(np.diff(q.y.values)); dpred.extend(np.diff(q.r1_obs.values))
            dobs, dpred = np.array(dobs), np.array(dpred)
            rr = np.corrcoef(dobs, dpred)[0, 1] if len(dobs) > 5 else np.nan
            ax.scatter(dobs, dpred, s=4, color=S.REGION_HEX[r], alpha=0.5, lw=0, label=f"{S.REGION_EN[r]}: r {rr:.2f}, n {len(dobs):,}", rasterized=True)
        lim = 60; ax.plot([-lim, lim], [-lim, lim], color=S.NEUTRAL["zero_line"], lw=0.6); ax.set_xlim(-lim, lim); ax.set_ylim(-lim / 2, lim / 2)
        ax.set_xlabel("Observed year-to-year change at a cell (cm)"); ax.set_ylabel("Predicted change, R1 (cm)"); ax.legend(loc="upper left", fontsize=6, handlelength=1.0)
    S.panel_letter(ax, "d")
    # e 연도별 편향(R1 블록 CV), 세 지역
    ax = fig.add_subplot(gs[1, 2])
    if (yw / "yearwise_metrics.csv").exists():
        ym = pd.read_csv(yw / "yearwise_metrics.csv"); ym = ym[(ym.series == "ours:R1:0.5:blockCV") & (ym.n >= 50)]
        for r in REG:
            s = ym[ym.region == r].sort_values("year"); ax.plot(s.year, s.bias, "-o", ms=2.4, lw=0.8, color=S.REGION_HEX[r])
        ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylim(-45, 25)
        ax.set_xlabel("Observation year"); ax.set_ylabel("Bias of R1, block CV (cm)")
    S.panel_letter(ax, "e")
    S.register("ce", *S.PANELS.get("c", []), *S.PANELS.get("e", []))
    return export(fig, "F3_time_validation")


# ================================================================ F4 이득의 조건 + 제품 비교
def f4():
    run, forcing = _run()
    fig = S.figure(S.W2, 112)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.45, hspace=0.6, left=0.06, right=0.99, top=0.95, bottom=0.14)
    # a 공변량 사다리
    ax = fig.add_subplot(gs[0, 0])
    tags = ["modis_v6", "featsets_v6"] if (PROC / "runs" / "featsets_v6" / "deltas_vs_ref.csv").exists() else ["modis_v6", "featsets", "featsets2", "featsets3"]
    d = pd.concat([pd.read_csv(PROC / "runs" / t / "deltas_vs_ref.csv").assign(_tag=t) for t in tags if (PROC / "runs" / t / "deltas_vs_ref.csv").exists()])
    d = d[(d.scheme == "block") & (d.ref == "P1") & (d.method == "R1") & (d.lam == 0.5)]
    if "forcing" in d:   # 관측일 정합 판이 있으면 그것만(옛 실행은 year)
        want = "year_obs" if (d.forcing == "year_obs").any() else "year"; d = d[(d.forcing == want) | (~d._tag.isin(["modis_v6", "featsets_v6"]))]
    order = ["F8", "F8T", "F8TS", "F8TS_M", "F8TS_X", "F8TS_ALL"]; labels = ["climate", "+terrain", "+soil", "+MODIS", "+hi-res", "+all"]
    for i, r in enumerate(REG):
        s = d[d.region == r].drop_duplicates("featset").set_index("featset").reindex(order).dropna(subset=["delta"])
        x = np.array([order.index(f) for f in s.index]) + (i - 1) * 0.24
        ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0, label=S.REGION_EN[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE change, physics + residual ML\n− recalibrated Stefan (cm)"); ax.legend(loc="upper right", fontsize=6, handlelength=1.0); S.panel_letter(ax, "a")
    # b TDD 3분위
    ax = fig.add_subplot(gs[0, 1])
    c = pd.read_csv(run / "conditions_tdd.csv"); c = c[(c.scheme == "block") & (c.forcing == forcing) & (c.method == "R1") & (c.lam == 0.5)]
    for i, r in enumerate(REG):
        s = c[c.region == r].sort_values("tdd_bin"); x = s.tdd_bin.values + (i - 1) * 0.24
        ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["cold", "mid", "warm"]); ax.set_xlabel("Thawing degree-day tercile within region")
    ax.set_ylabel("RMSE change (cm)"); S.panel_letter(ax, "b")
    # c 학습기
    ax = fig.add_subplot(gs[0, 2])
    parts = []
    for t in ("core_v6", "rf_v6", "nn_v6", "core_cb", "core_rf", "core_nn"):
        f = PROC / "runs" / t / "deltas_vs_ref.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    d = pd.concat(parts); d = d[(d.scheme == "block") & (d.ref == "P1") & (d.featset == "F8TS") & (d.method == "R1") & (d.lam == 0.5)]
    if "forcing" in d:
        d = d[d.forcing == forcing]
    order = ["catboost", "rf", "mlp", "tabm", "ftt"]
    for i, r in enumerate(REG):
        s = d[d.region == r].drop_duplicates("learner").set_index("learner").reindex(order).dropna(subset=["delta"])
        x = np.array([order.index(l) for l in s.index]) + (i - 1) * 0.24
        ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_xticks(range(len(order))); ax.set_xticklabels(["CatBoost", "RF", "MLP", "TabM", "FT-T"], rotation=35, ha="right")
    ax.set_ylabel("RMSE change (cm)"); S.panel_letter(ax, "c")
    # d 공통 셀 비교
    ax = fig.add_subplot(gs[1, 0:2])
    cc = pd.read_csv(run / "common_cells_eval.csv") if (run / "common_cells_eval.csv").exists() else pd.read_csv(PROC / "runs" / "core_cb" / "common_cells_eval.csv")
    order = ["P1 (Stefan, recal.)", "R1 (CatBoost)", "D0 (CatBoost)", "Wei 2026", "Liu 2024", "ESA CCI v4", "ESA CCI v5", "Aalto 2018"]
    short = ["Stefan\nrecal.", "physics +\nresidual ML", "direct ML", "Wei 2026", "Liu 2024", "CCI v4", "CCI v5", "Aalto 2018"]
    for i, r in enumerate(REG):
        for mask, mfc in ((0, S.REGION_HEX[r]), (25, "white")):
            s = cc[(cc.region == r) & (cc.mask_km == mask)].set_index("series").reindex(order)
            x = np.arange(len(order)) + (i - 1) * 0.24
            v = s.rmse.values.copy(); v = np.minimum(v, 45)
            ax.plot(x, v, "o", ms=3.0, color=S.REGION_HEX[r], mfc=mfc, mew=0.8, label=S.REGION_EN[r] if mask == 0 else None)
        s0 = cc[(cc.region == r) & (cc.mask_km == 0)].set_index("series").reindex(order).rmse.values; s25 = cc[(cc.region == r) & (cc.mask_km == 25)].set_index("series").reindex(order).rmse.values
        for xi, (a_, b_) in enumerate(zip(s0, s25)):
            if np.nanmax([a_, b_]) > 45:
                ax.text(xi + (i - 1) * 0.24, 45.6 + i * 0.9, f"{a_:.0f}/{b_:.0f}" if np.isfinite(b_) else f"{a_:.0f}", ha="center", va="bottom", fontsize=4.8, color=S.REGION_HEX[r])
    ax.axvline(2.5, color=S.NEUTRAL["grid"], lw=0.6); ax.set_ylim(10, 50); ax.set_xticks(range(len(order))); ax.set_xticklabels(short, fontsize=6)
    ax.set_ylabel("RMSE on common cell-years (cm)"); ax.legend(loc="upper left", fontsize=6, handlelength=1.0)
    ax.text(0.99, 0.03, "filled: all common cells; open: > 25 km from product training sites\nours = block-CV out-of-fold; products = year-matched values", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=S.NEUTRAL["text_aux"])
    S.panel_letter(ax, "d")
    # e 관측 대 예측(R1, 세 지역 밀도 산점)
    ax = fig.add_subplot(gs[1, 2])
    import glob
    fs = glob.glob(str(run / "preds" / f"*__block__{forcing}__F8TS__R1__catboost__lam0.5.csv.gz"))
    if fs:
        p = pd.concat([pd.read_csv(f) for f in fs]).groupby("cell_year_id").agg(y=("y", "first"), pred=("pred", "mean"), region=("region", "first"))
        for r in REG:
            q = p[p.region == r].copy(); q["bin"] = pd.qcut(q.y, 10, labels=False, duplicates="drop")
            g = q.groupby("bin").agg(x=("y", "mean"), m=("pred", "mean"), lo=("pred", lambda v: np.percentile(v, 25)), hi=("pred", lambda v: np.percentile(v, 75)))
            ax.fill_between(g.x, g.lo, g.hi, color=S.REGION_HEX[r], alpha=0.18, lw=0)
            ax.plot(g.x, g.m, "-o", ms=2.6, lw=1.0, color=S.REGION_HEX[r], label=f"{S.REGION_EN[r]}: r {np.corrcoef(q.y, q.pred)[0,1]:.2f}")
        ax.plot([0, 150], [0, 150], color=S.NEUTRAL["zero_line"], lw=0.6); ax.set_xlim(0, 150); ax.set_ylim(0, 150)
        ax.set_xlabel("Observed ALT, decile mean (cm)"); ax.set_ylabel("Predicted ALT, physics + residual ML (cm)")
        ax.legend(loc="upper left", fontsize=5.5, handlelength=1.2); ax.text(0.98, 0.03, "line: decile mean; band: interquartile", transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color=S.NEUTRAL["text_aux"])
    S.panel_letter(ax, "e")
    S.register("ab", *S.PANELS.get("a", []), *S.PANELS.get("b", []))
    S.register("de", *S.PANELS.get("d", []), *S.PANELS.get("e", []))
    return export(fig, "F4_gain_products")


# ================================================================ F5 지점 시계열
def f5():
    run, forcing = _run()
    ss = ROOT / "data" / "processed" / "v5" / "site_series" / f"site_series_{forcing}.csv.gz"
    if not ss.exists():
        ss = ROOT / "data" / "processed" / "v5" / "site_series" / "site_series_year_obs.csv.gz"
    st = pd.read_csv(ss)
    import polar.testbed_v5 as T
    base = T.load_v5(PROC)[["loc_id", "year", "prod_wei", "prod_liu", "prod_cci5y", "lat", "lon"]]
    obs = st[st.observed == 1]
    nyr = obs.groupby(["region", "loc_id"]).year.nunique()
    picks = []
    for r in REG:
        q = nyr.loc[r].sort_values(ascending=False)
        picks += [(r, lid) for lid in q.index[:2]]
    fig = S.figure(S.W2, 150)
    gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.35, hspace=0.65, left=0.06, right=0.99, top=0.93, bottom=0.07, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 1.1])
    handles = None
    for k, (r, lid) in enumerate(picks):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        g = st[(st.region == r) & (st.loc_id == lid)].sort_values("year")
        ax.plot(g.year, g.p1_eos, "-", lw=0.9, color=S.METHOD_HEX["P1"], label="Stefan, recalibrated")
        ax.plot(g.year, g.r1_eos, "-", lw=1.1, color=S.METHOD_HEX["R1"], label="physics + residual ML")
        pr = base[base.loc_id == lid].set_index("year").reindex(g.year)
        ax.plot(g.year, pr.prod_wei.values, "-", lw=0.7, color=S.PRODUCT_HEX["wei"], alpha=0.9, label="Wei 2026")
        ax.plot(g.year, pr.prod_liu.values, "-", lw=0.7, color=S.PRODUCT_HEX["liu"], alpha=0.9, label="Liu 2024")
        o = g[g.observed == 1]
        ax.plot(o.year, o.y, "o", ms=3.4, color="black", mfc="white", mew=0.9, label="observed", zorder=5)
        ax.plot(o.year, o.r1_obs, "s", ms=2.4, color=S.METHOD_HEX["R1"], zorder=6, label="R1 at observation date")
        la, lo = float(base[base.loc_id == lid].lat.iloc[0]), float(base[base.loc_id == lid].lon.iloc[0])
        ax.text(0.02, 0.97, f"{S.REGION_EN[r]} site {k % 2 + 1} ({la:.2f}°, {lo:.2f}°), {len(o)} years", transform=ax.transAxes, va="top", ha="left", fontsize=6)
        ax.set_xlim(2009.5, 2024.5); ax.set_xticks([2010, 2014, 2018, 2022])
        if k % 2 == 0:
            ax.set_ylabel("ALT (cm)")
        if k == 0:
            handles = ax.get_legend_handles_labels()
        S.panel_letter(ax, "abcdef"[k])
    fig.legend(*handles, loc="upper center", bbox_to_anchor=(0.36, 0.995), ncol=6, fontsize=5.5, handlelength=1.3, columnspacing=0.9, frameon=False)
    # g 안정성: 연도별 예측 변동 폭 대 관측 변동 폭(다년 지점)
    ax = fig.add_subplot(gs[0:3, 2])
    rows = []
    for (r, lid), g in obs.groupby(["region", "loc_id"]):
        if len(g) >= 4:
            rows.append(dict(region=r, sd_obs=g.y.std(), sd_r1=g.r1_obs.std(), sd_p1=g.p1_obs.std()))
    rr = pd.DataFrame(rows)
    for r in REG:
        q = rr[rr.region == r]; ax.scatter(q.sd_obs, q.sd_r1, s=8, color=S.REGION_HEX[r], alpha=0.7, lw=0, label=f"{S.REGION_EN[r]} ({len(q)} sites)")
    ax.plot([0, 40], [0, 40], color=S.NEUTRAL["zero_line"], lw=0.6); ax.set_xlim(0, 40); ax.set_ylim(0, 40)
    ax.set_xlabel("s.d. of observed ALT across years (cm)"); ax.set_ylabel("s.d. of predicted ALT across years, R1 (cm)"); ax.legend(loc="upper left", fontsize=6, handlelength=1.0)
    S.panel_letter(ax, "g")
    # h 전 지점 평균 시계열: 예측(모든 셀, 전 연도) 지역 평균과 관측 평균
    ax = fig.add_subplot(gs[3, :]); _p = ax.get_position(); ax.set_position([_p.x0, _p.y0 - 0.015, _p.width, _p.height - 0.02])   # g 의 x 라벨과 간격
    for r in REG:
        g = st[st.region == r]
        m = g.groupby("year").r1_eos.mean(); ax.plot(m.index, m.values - m.mean(), "-", lw=1.0, color=S.REGION_HEX[r], label=f"{S.REGION_EN[r]}: predicted (all cells)")
        o = g[g.observed == 1].groupby("year").agg(y=("y", "mean"), n=("y", "size")); o = o[o.n >= 50]
        ax.plot(o.index, o.y - o.y.mean(), "o", ms=3, color=S.REGION_HEX[r], mfc="white", mew=0.9, label=f"{S.REGION_EN[r]}: observed (years with ≥ 50 cells)")
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6); ax.set_xlim(2009.5, 2024.5); ax.set_xlabel("Year"); ax.set_ylabel("Regional mean ALT anomaly (cm)")
    ax.set_ylim(-35, 50); ax.legend(loc="upper left", fontsize=5.5, ncol=3, handlelength=1.2, columnspacing=0.8); S.panel_letter(ax, "h")
    S.register("abcdef", *[a for k in "abcdef" for a in S.PANELS.get(k, [])])
    S.register("gh", *S.PANELS.get("g", []), *S.PANELS.get("h", []))
    return export(fig, "F5_site_series")


FIGS.update({3: f3, 4: f4, 5: f5})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", type=int, action="append", default=None); ap.add_argument("--slide", action="store_true")
    a = ap.parse_args(); S.setup(scale=1.69 if a.slide else 1.0)
    for k in (a.fig or sorted(FIGS)):
        FIGS[k](); print(f"[F{k}] done")
