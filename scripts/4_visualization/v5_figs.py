"""v5 논문 그림(figures/figure_spec_v5.json). 실행: python3 scripts/4_visualization/v5_figs.py --fig 1 [--fig 2 ...]"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

PROC = ROOT / "data" / "processed" / "v5"
REG = ["Alaska", "Lena", "Canada"]


def fig1():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv")
    lab = lab[lab.in_forcing == 1]
    cells = lab.drop_duplicates("loc_id")
    fig = S.figure(S.W2, 95)
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1.0, 1.25], height_ratios=[1, 1, 1], wspace=0.35, hspace=0.55,
                           left=0.05, right=0.99, top=0.95, bottom=0.08)
    # (a) 지도
    proj = ccrs.NorthPolarStereo(central_longitude=-100)
    axm = fig.add_subplot(gs[:, 0], projection=proj)
    axm.set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor=S.NEUTRAL["land"], edgecolor="none", zorder=0)
    axm.coastlines("50m", linewidth=S.L["map_coast_pt"], color=S.NEUTRAL["coast"], zorder=1)
    gl = axm.gridlines(draw_labels=False, linewidth=S.L["graticule_pt"], color=S.NEUTRAL["grid"], xlocs=range(-180, 181, 60), ylocs=[60, 70, 80])
    for r in REG:
        c = cells[cells.region == r]
        axm.scatter(c.lon, c.lat, s=1.2, c=S.REGION_HEX[r], transform=ccrs.PlateCarree(), lw=0, zorder=3, rasterized=True)
    for r, (lon, lat, ha, va) in {"Alaska": (-152, 57.0, "center", "top"), "Lena": (100, 70.5, "right", "center"),
                                  "Canada": (-116, 55.5, "center", "top")}.items():
        n = int((cells.region == r).sum())
        axm.text(lon, lat, f"{S.REGION_EN[r]}\n{n:,} cells", transform=ccrs.PlateCarree(), ha=ha, va=va, fontsize=S.F["paper"]["text_pt"] - 0.5,
                 color=S.REGION_HEX[r])
    axm.spines["geo"].set_linewidth(S.L["axis_pt"]); axm.spines["geo"].set_edgecolor(S.NEUTRAL["coast"])
    S.panel_letter(axm, "a", x=0.0, y=1.0)
    # (b) 연도별 셀-연도 수
    years = np.arange(2010, 2025)
    for i, r in enumerate(REG):
        ax = fig.add_subplot(gs[i, 1])
        cnt = lab[lab.region == r].groupby("year").size().reindex(years, fill_value=0)
        ax.bar(years, cnt.values, width=0.75, color=S.REGION_HEX[r], lw=0)
        ax.set_xlim(2009.3, 2024.7); ax.set_xticks([2010, 2014, 2018, 2022])
        ax.set_ylabel("Cell-years")
        ymax = cnt.max(); ax.set_ylim(0, ymax * 1.18)
        ax.text(0.02, 0.95, f"{S.REGION_EN[r]}: {int(cnt.sum()):,} cell-years, {lab[lab.region == r].loc_id.nunique():,} cells",
                transform=ax.transAxes, va="top", ha="left", fontsize=S.F["paper"]["text_pt"] - 0.5)
        if i == 2:
            ax.set_xlabel("Observation year")
        else:
            ax.set_xticklabels([])
        if i == 0:
            S.panel_letter(ax, "b")
    S.export(fig, "Fig1_data_time_ab")

    # (c) 기간 띠: 별도 그림(같은 Fig 1 의 아래 띠로 합칠 때는 벡터 편집 없이 두 파일을 쓴다)
    rows = [
        ("Labels: Alaska (ABoVE, CALM)", 2010, 2024, STATUS_("label")), ("Labels: Lena Delta (ALLENA)", 2010, 2022, STATUS_("label")),
        ("Labels: Canada (ABoVE, CALM)", 2010, 2024, STATUS_("label")),
        ("ERA5-Land forcing (yearly)", 2010, 2024, STATUS_("climate")),
        ("MODIS NDVI/EVI (MOD13Q1)", 2015, 2020, STATUS_("satellite")), ("MODIS snow (MOD10A2)", 2015, 2021, STATUS_("satellite")),
        ("ABoVE SAR ALT (Alaska)", 2017, 2017, STATUS_("satellite")),
        ("ESA CCI ALT v4", 1997, 2021, STATUS_("product")), ("ESA CCI ALT v5", 1997, 2023, STATUS_("product")), ("Wei 2026", 2000, 2024, STATUS_("product")),
        ("Liu 2024", 2003, 2020, STATUS_("product")), ("Yi–Kimball 2019 (Alaska)", 2001, 2015, STATUS_("product")),
        ("Aalto 2018 (2000–2014 mean)", 2000, 2014, STATUS_("product")),
        ("Terrain (DEM), soil (SoilGrids)", 1995, 2025, STATUS_("terrain")),
    ]
    fig2 = S.figure(S.W2, 58)
    ax = fig2.add_axes([0.30, 0.10, 0.68, 0.86])
    for j, (name, y0, y1, col) in enumerate(rows[::-1]):
        if y0 == y1:
            ax.plot([y0 + 0.5], [j], marker="s", ms=3.5, color=col, lw=0)
        elif name.startswith("Terrain"):
            ax.hlines(j, y0, y1 + 1, color=col, lw=1.0, ls=(0, (2, 2)))
        else:
            ax.barh(j, y1 - y0 + 1, left=y0, height=0.55, color=col, lw=0)
    ax.axvspan(2010, 2025, color=S.NEUTRAL["equiv_band"], zorder=0, lw=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(1995, 2026); ax.set_xticks([1995, 2000, 2005, 2010, 2015, 2020, 2025]); ax.set_xlabel("Year")
    ax.tick_params(axis="y", length=0)
    ax.text(2017.5, len(rows) - 0.35, "analysis window 2010–2024", ha="center", va="bottom", fontsize=S.F["paper"]["text_pt"] - 0.5, color=S.NEUTRAL["text_aux"])
    ax.set_ylim(-0.7, len(rows) + 0.3)
    S.panel_letter(ax, "c", x=-0.42, y=0.995)
    S.export(fig2, "Fig1_data_time_c")


def STATUS_(k):
    return S.STATUS[k]


FIGS = {1: fig1}


# ================================================================ Fig 2·3 (집계 표 기반)
RUN = PROC / "runs" / "core_cb"
METH_ORDER = ["P0", "P1", "P1a", "Ku", "D0", "F1", "D1", "R1", "S"]


def _label(m, learner="-", lam=-1.0):
    if m == "R1":
        return f"R1 (λ {lam:g})"
    return m


def fig2():
    """연도 정합 − 기후값 강제력: 짝지은 ΔRMSE 와 연간 편차 상관."""
    d = pd.read_csv(RUN / "deltas_forcing.csv")
    d = d[(d.featset == "F8TS") & d.scheme.isin(["block", "year"])]
    d = d[(d.method != "R1") | (d.lam == 0.5)]
    fig = S.figure(S.W2, 66)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45, left=0.07, right=0.99, top=0.9, bottom=0.26, width_ratios=[1, 1, 0.9])
    for j, scheme in enumerate(["block", "year"]):
        ax = fig.add_subplot(gs[0, j])
        for i, r in enumerate(REG):
            s = d[(d.scheme == scheme) & (d.region == r)].set_index("method").reindex(METH_ORDER).dropna(subset=["delta"])
            x = np.array([METH_ORDER.index(m) for m in s.index]) + (i - 1) * 0.25
            ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.2, color=S.REGION_HEX[r], lw=0.8, capsize=0,
                        label=S.REGION_EN[r])
        ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0)
        ax.set_xticks(range(len(METH_ORDER))); ax.set_xticklabels(METH_ORDER)
        ax.set_ylabel("RMSE change (cm)\nyear-matched − climatological")
        ax.text(0.02, 0.97, S.SCHEME_EN[scheme], transform=ax.transAxes, va="top", ha="left")
        S.panel_letter(ax, "ab"[j])
        if j == 0:
            h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h2, l2, loc="lower center", ncol=3, handlelength=1.0, columnspacing=1.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    ax = fig.add_subplot(gs[0, 2])
    sm = pd.read_csv(RUN / "summary_metrics.csv")
    sm = sm[(sm.featset == "F8TS") & (sm.scheme == "year") & ((sm.method != "R1") | (sm.lam == 0.5))]
    for i, r in enumerate(REG):
        s = sm[(sm.region == r) & (sm.forcing == "year")].set_index("method").reindex(METH_ORDER)
        x = np.arange(len(METH_ORDER)) + (i - 1) * 0.25
        ax.plot(x, s.r_anom.values, "o", ms=3.0, color=S.REGION_HEX[r], label=S.REGION_EN[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0)
    ax.set_xticks(range(len(METH_ORDER))); ax.set_xticklabels(METH_ORDER)
    ax.set_ylabel("Interannual anomaly correlation\n(year hold-out, year-matched forcing)")
    ax.text(0.98, 0.97, "climatological forcing:\nanomaly 0 by construction", transform=ax.transAxes, ha="right", va="top",
            fontsize=S.F["paper"]["text_pt"] - 1.5, color=S.NEUTRAL["text_aux"])
    S.panel_letter(ax, "c")
    return S.export(fig, "Fig2_forcing_effect")


def fig3():
    """검증 사다리: 지역별 패널, 검증 단 × 방법 RMSE(연도 정합 강제력)."""
    sm = pd.read_csv(RUN / "summary_metrics.csv")
    sm = sm[(sm.featset == "F8TS") & (sm.forcing == "year") & ((sm.method != "R1") | (sm.lam == 0.5))]
    dl = pd.read_csv(RUN / "deltas_vs_ref.csv")
    dl = dl[(dl.featset == "F8TS") & (dl.forcing == "year") & (dl.ref == "P1") & ((dl.method != "R1") | (dl.lam == 0.5))]
    schemes = ["random", "block", "year", "block_year", "region"]
    fig = S.figure(S.W2, 84)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32, left=0.06, right=0.99, top=0.92, bottom=0.30)
    for j, r in enumerate(REG):
        ax = fig.add_subplot(gs[0, j])
        for m in METH_ORDER:
            s = sm[(sm.region == r) & (sm.method == m)].set_index("scheme").reindex(schemes)
            x = np.arange(len(schemes)) + (METH_ORDER.index(m) - 4) * 0.08
            ax.plot(x, s.rmse.values, marker="o", ms=2.6, lw=0.9, ls=S.METHOD_LS.get(m, "-"), color=S.METHOD_HEX[m], label=S.METHOD_EN.get(m, m))
        ax.set_xticks(range(len(schemes))); ax.set_xticklabels(["Random", "Block", "Year", "Block×year", "Region"], rotation=25, ha="right")
        ax.set_ylabel("RMSE (cm)" if j == 0 else "")
        ax.text(0.02, 0.97, S.REGION_EN[r], transform=ax.transAxes, va="top", ha="left")
        S.panel_letter(ax, "abc"[j])
        if j == 0:
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=S.F["paper"]["text_pt"] - 1, handlelength=1.8, columnspacing=1.2, frameon=False,
               bbox_to_anchor=(0.52, 0.0))
    return S.export(fig, "Fig3_method_ladder")


FIGS.update({2: fig2, 3: fig3})



# ================================================================ Fig 4·5
def fig4():
    """조건: TDD 구간별 ΔRMSE(R1 − P1), 공변량 집합별, 학습기별."""
    fig = S.figure(S.W2, 65)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4, left=0.07, right=0.99, top=0.9, bottom=0.25)
    # (a) TDD 구간
    ax = fig.add_subplot(gs[0, 0])
    c = pd.read_csv(RUN / "conditions_tdd.csv")
    c = c[(c.scheme == "block") & (c.forcing == "year") & (c.method == "R1") & (c.lam == 0.5)]
    for i, r in enumerate(REG):
        s = c[c.region == r].sort_values("tdd_bin")
        x = s.tdd_bin.values + (i - 1) * 0.22
        ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.2, color=S.REGION_HEX[r], lw=0.8, capsize=0, label=S.REGION_EN[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Low", "Mid", "High"]); ax.set_xlabel("Thawing degree-day tercile (within region)")
    ax.set_ylabel("RMSE change, R1 − P1 (cm)"); ax.legend(loc="best", handlelength=1.0); S.panel_letter(ax, "a")
    # (b) 공변량 집합
    ax = fig.add_subplot(gs[0, 1])
    parts = [pd.read_csv(PROC / "runs" / t / "deltas_vs_ref.csv") for t in ("featsets", "featsets2", "featsets3") if (PROC / "runs" / t / "deltas_vs_ref.csv").exists()]
    if parts:
        d = pd.concat(parts)
        d = d[(d.scheme == "block") & (d.forcing == "year") & (d.ref == "P1") & (d.method == "R1") & (d.lam == 0.5)]
        order = ["F8", "F8T", "F8TS", "F8TS_L", "F8TS_M", "F8TS_X", "F8TS_ALL", "F8TS_SAR"]
        labels = ["Climate", "+terrain", "+soil", "+prev. year", "+MODIS", "+hi-res", "+all", "+SAR"]
        for i, r in enumerate(REG):
            s = d[d.region == r].drop_duplicates("featset").set_index("featset").reindex(order).dropna(subset=["delta"])
            x = np.array([order.index(f) for f in s.index]) + (i - 1) * 0.22
            ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylabel("RMSE change, R1 − P1 (cm)"); S.panel_letter(ax, "b")
    # (c) 학습기
    ax = fig.add_subplot(gs[0, 2])
    parts = []
    for tag in ("core_cb", "core_nn", "core_rf"):
        f = PROC / "runs" / tag / "deltas_vs_ref.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    if parts:
        d = pd.concat(parts)
        d = d[(d.scheme == "block") & (d.forcing == "year") & (d.ref == "P1") & (d.featset == "F8TS") & d.method.isin(["D0", "R1"]) & ((d.method != "R1") | (d.lam == 0.5))]
        order = ["catboost", "rf", "mlp", "tabm", "ftt"]
        for i, r in enumerate(REG):
            for m, mk in (("R1", "o"), ("D0", "s")):
                s = d[(d.region == r) & (d.method == m)].set_index("learner").reindex(order).dropna(subset=["delta"])
                x = np.array([order.index(l) for l in s.index]) + (i - 1) * 0.22
                ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt=mk, ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0,
                            mfc=S.REGION_HEX[r] if m == "R1" else "white", mew=0.8)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(["CatBoost", "RF", "MLP", "TabM", "FT-T"], rotation=25, ha="right")
        ax.text(0.98, 0.03, "filled: R1 (λ 0.5), open: D0", transform=ax.transAxes, ha="right", va="bottom", fontsize=S.F["paper"]["text_pt"] - 1)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylabel("RMSE change vs P1 (cm)"); S.panel_letter(ax, "c")
    return S.export(fig, "Fig4_conditions")


def fig5():
    """제품 비교: (a) 같은 셀·같은 해 짝지은 ΔRMSE(R1 − 제품), (b) 편향, (c) 알래스카 연도별 편향."""
    pe = pd.read_csv(RUN / "products_eval.csv"); by = pd.read_csv(RUN / "products_bias_by_year.csv")
    prods = ["cci4y", "cci5y", "wei", "liu", "yk", "aalto"]
    fig = S.figure(S.W2, 78)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42, left=0.07, right=0.99, top=0.92, bottom=0.34, width_ratios=[1.1, 1.1, 1.3])
    # (a) 짝지은 ΔRMSE
    ax = fig.add_subplot(gs[0, 0])
    YL = (-30, 10)
    pm = pd.read_csv(RUN / "products_eval_mask25.csv") if (RUN / "products_eval_mask25.csv").exists() else None
    for i, r in enumerate(REG):
        for k, p in enumerate(prods):
            for j, (tab, mk, dx) in enumerate(((pe, "o", -0.08), (pm, "o", 0.08))):
                if tab is None:
                    continue
                q = tab[(tab.region == r) & (tab.series == "ours:R1:0.5") & (tab.definition == f"vs {p} (year-matched)")]
                if q.empty:
                    continue
                d, lo, hi = float(q.delta_vs_product.iloc[0]), float(q.delta_lo.iloc[0]), float(q.delta_hi.iloc[0])
                x = k + (i - 1) * 0.26 + dx
                if d < YL[0]:
                    ax.text(x, YL[0] + 0.5, f"{d:+.0f}", ha="center", va="bottom", fontsize=S.F["paper"]["text_pt"] - 1.5, color=S.REGION_HEX[r], rotation=90)
                else:
                    ax.errorbar(x, d, yerr=[[d - max(lo, YL[0])], [min(hi, YL[1]) - d]], fmt=mk, ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0,
                                mfc=S.REGION_HEX[r] if j == 0 else "white", mew=0.8, label=S.REGION_EN[r] if (k == 0 and j == 0) else None)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylim(*YL)
    ax.set_xticks(range(len(prods))); ax.set_xticklabels([S.PRODUCT_EN[p] for p in prods], rotation=35, ha="right")
    ax.set_ylabel("RMSE change, R1 − product (cm)\nsame cells and years, block CV")
    ax.text(0.02, 0.97, "filled: all cells\nopen: cells > 25 km from\nproduct training sites", transform=ax.transAxes, ha="left", va="top",
            fontsize=S.F["paper"]["text_pt"] - 1.5, color=S.NEUTRAL["text_aux"])
    ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0.3); S.panel_letter(ax, "a")
    # (b) 편향
    ax = fig.add_subplot(gs[0, 1])
    YB = (-25, 25)
    for i, r in enumerate(REG):
        s = pe[(pe.region == r) & (pe.definition == "year-matched") & pe.series.isin(prods)].set_index("series").reindex(prods)
        x = np.arange(len(prods)) + (i - 1) * 0.22
        v = s.bias.values
        inr = np.isfinite(v) & (v >= YB[0]) & (v <= YB[1])
        ax.plot(x[inr], v[inr], "o", ms=3.0, color=S.REGION_HEX[r])
        for xx, vv in zip(x[np.isfinite(v) & ~inr], v[np.isfinite(v) & ~inr]):
            ax.text(xx, YB[1] - 0.5 if vv > 0 else YB[0] + 0.5, f"{vv:+.0f}", ha="center", va="top" if vv > 0 else "bottom",
                    fontsize=S.F["paper"]["text_pt"] - 1.5, color=S.REGION_HEX[r], rotation=90)
        o = pe[(pe.region == r) & (pe.definition == "all cells") & pe.series.isin(["ours:P1", "ours:R1:0.5"])].set_index("series").reindex(["ours:P1", "ours:R1:0.5"])
        ax.plot(len(prods) + np.arange(2) + 0.3 + (i - 1) * 0.22, o.bias.values, "D", ms=2.8, color=S.REGION_HEX[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylim(*YB)
    ax.set_xticks(list(range(len(prods))) + list(len(prods) + np.arange(2) + 0.3))
    ax.set_xticklabels([S.PRODUCT_EN[p] for p in prods] + ["P1 (ours)", "R1 (ours)"], rotation=35, ha="right")
    ax.set_ylabel("Bias, prediction − observed (cm)"); S.panel_letter(ax, "b")
    # (c) 알래스카 연도별 편향
    ax = fig.add_subplot(gs[0, 2])
    b = by[(by.region == "Alaska") & (by.n >= 100)]
    YC = (-40, 40)
    handles = []
    for p in prods + ["ours:P1", "ours:R1:0.5"]:
        s = b[b.series == p].sort_values("year")
        if s.empty:
            continue
        col = S.PRODUCT_HEX.get(p, S.METHOD_HEX["R1"] if "R1" in p else S.METHOD_HEX["P1"])
        v = np.clip(s.bias.values, *YC)
        ln, = ax.plot(s.year, v, marker="o", ms=2.4, lw=0.9, color=col, label=S.PRODUCT_EN.get(p, p.replace("ours:", "").replace(":0.5", "") + " (ours)"))
        handles.append(ln)
        for yy, vv in zip(s.year.values[(s.bias.values < YC[0]) | (s.bias.values > YC[1])], s.bias.values[(s.bias.values < YC[0]) | (s.bias.values > YC[1])]):
            ax.text(yy, YC[1] if vv > 0 else YC[0], f"{vv:+.0f}", ha="center", va="bottom" if vv > 0 else "top", fontsize=S.F["paper"]["text_pt"] - 2, color=col)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylim(YC[0] - 6, YC[1] + 6)
    ax.set_xlabel("Observation year (Alaska, years with ≥ 100 cells)"); ax.set_ylabel("Bias, prediction − observed (cm)")
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=S.F["paper"]["text_pt"] - 1, handlelength=1.6, columnspacing=1.0, frameon=False,
               bbox_to_anchor=(0.52, 0.0))
    S.panel_letter(ax, "c")
    return S.export(fig, "Fig5_products")


FIGS.update({4: fig4, 5: fig5})


# ================================================================ Fig 6·7 (지도)
def _mv():
    sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
    import maps_alt_v3 as MV
    return MV


def _map_setup(MV, key):
    import cartopy.crs as ccrs
    cells_f = {"alaska": PROC.parent / "map_alaska" / "alaska_grid_cells_v1.csv", "lena": PROC.parent / "map_lena" / "lena_grid_cells_v1.csv"}[key]
    cells = pd.read_csv(cells_f, dtype={"cell_id": str})
    d = dict(key=key, proj=MV.projection(key), **MV.REG[key])
    d["ext"] = MV.box_extent(d["proj"], d["box"], d["pad_m"])
    d["cells"] = cells
    return d


def _draw_panel(fig, rect, MV, d, cell_ids, values, cmap, norm, M, left, bottom, label):
    """격자 셀 값(values, cell_ids 순)을 QuadMesh 로 그린다."""
    c = d["cells"].set_index("cell_id").loc[cell_ids]
    mesh = MV.build_mesh(d["proj"], c.ky.values, c.kx.values)
    ax = fig.add_axes(rect, projection=d["proj"])
    ax.set_extent(d["ext"], crs=d["proj"]); ax.set_facecolor("#ffffff")
    rgba = cmap(norm(np.asarray(values, float)))
    rgba[~np.isfinite(values), :] = (0, 0, 0, 0)
    MV.add_cells(ax, mesh, rgba)
    ne = MV.ne_layers(d["key"])
    ax.add_geometries(ne["coastline"], crs=MV.PC, facecolor="none", edgecolor=S.NEUTRAL["coast"], linewidth=M.lw_coast, zorder=4)
    MV.graticule(ax, d["xlocs"] if d["key"] == "alaska" else d["xlocs"][::2], d["ylocs"], M, left=left, bottom=bottom)
    ax.spines["geo"].set_linewidth(M.lw); ax.spines["geo"].set_edgecolor("#000000")
    ax.text(0.02, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=S.F["paper"]["text_pt"],
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    return ax


def fig6():
    """알래스카 연도별 ALT(R1, 연도 정합 강제력), 2011–2024 평균, 선택 연도 편차."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    MV = _mv(); M = MV.Med("paper"); MV.use_medium("paper"); S.setup(scale=S.SCALE)
    d = _map_setup(MV, "alaska")
    MD = PROC / "maps" / "Alaska"
    meta = pd.read_json(MD / "meta.json", typ="series")
    mean = pd.read_csv(MD / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    years = [2014, 2016, 2018, 2022]
    preds = {y: pd.read_csv(MD / f"pred_{y}.csv.gz", dtype={"cell_id": str}).set_index("cell_id") for y in years}
    ids = mean.index
    lo, hi = MV.pct_range(mean.r1.values)
    anom = preds[2014].r1.reindex(ids) - mean.r1
    v = MV.sym_max(anom.values)
    fig = S.figure(S.W2, 118)
    asp = (d["ext"][3] - d["ext"][2]) / (d["ext"][1] - d["ext"][0])
    pw = 0.30; ph = pw * (S.W2 / 118) * asp
    xs = [0.035, 0.355, 0.675]; ys = [0.53, 0.06]
    panels = [(2014, "a"), (2016, "b"), (2018, "c"), (2022, "d")]
    axes = []
    for i, (y, L) in enumerate(panels):
        r, c = divmod(i, 3)
        vals = preds[y].r1.reindex(ids).values
        ax = _draw_panel(fig, [xs[c], ys[r], pw, ph], MV, d, ids, vals, MV.CM_ALT, Normalize(lo, hi), M, left=(c == 0), bottom=(r == 1),
                         label=f"ALT {y} (R1; ERA5-Land {y})")
        S.panel_letter(ax, L, x=-0.02, y=1.02); axes.append(ax)
    ax = _draw_panel(fig, [xs[1], ys[1], pw, ph], MV, d, ids, mean.r1.values, MV.CM_ALT, Normalize(lo, hi), M, left=False, bottom=True,
                     label="Mean ALT 2011–2024 (R1)")
    S.panel_letter(ax, "e", x=-0.02, y=1.02)
    ax = _draw_panel(fig, [xs[2], ys[1], pw, ph], MV, d, ids, anom.values, MV.CM_DIFF, TwoSlopeNorm(0.0, -v, v), M, left=False, bottom=True,
                     label="ALT 2014 − mean 2011–2024")
    S.panel_letter(ax, "f", x=-0.02, y=1.02)
    cax = fig.add_axes([0.035, 0.015, 0.28, 0.018])
    MV.colorbar(fig, cax, MV.CM_ALT, Normalize(lo, hi), "ALT (cm)", MV.ticks_between(lo, hi), M, MV.extend_of(mean.r1.values, lo, hi), orientation="horizontal")
    cax = fig.add_axes([0.675, 0.015, 0.28, 0.018])
    MV.colorbar(fig, cax, MV.CM_DIFF, TwoSlopeNorm(0.0, -v, v), "ALT difference (cm)", MV.ticks_between(-v, v), M, MV.extend_of(anom.values, -v, v),
                orientation="horizontal")
    return S.export(fig, "Fig6_maps_alaska")


def fig7():
    """알래스카 적용 가능 영역(AOA)·잔차 보정량 지도(위), 레나 2016 ALT·편차(아래)."""
    from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap
    MV = _mv(); M = MV.Med("paper"); MV.use_medium("paper"); S.setup(scale=S.SCALE)
    H = 150.0
    fig = S.figure(S.W2, H)
    d = _map_setup(MV, "alaska"); MD = PROC / "maps" / "Alaska"
    mean = pd.read_csv(MD / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    ids = mean.index
    asp = (d["ext"][3] - d["ext"][2]) / (d["ext"][1] - d["ext"][0])
    pw = 0.40; ph = pw * (S.W2 / H) * asp
    y_top = 0.97 - ph
    aoa_cm = ListedColormap([S.NEUTRAL["no_data"], "#6BAED6"])
    ax = _draw_panel(fig, [0.06, y_top, pw, ph], MV, d, ids, mean.aoa_in.values.astype(float), aoa_cm, Normalize(-0.5, 1.5), M, True, True,
                     label="Area of applicability (blue = inside)")
    S.panel_letter(ax, "a", x=-0.02, y=1.02)
    corr = (mean.r1 - mean.p1).values; v = MV.sym_max(corr)
    ax = _draw_panel(fig, [0.56, y_top, pw, ph], MV, d, ids, corr, MV.CM_DIFF, TwoSlopeNorm(0.0, -v, v), M, False, True,
                     label="Residual correction R1 − P1, mean 2011–2024")
    S.panel_letter(ax, "b", x=-0.02, y=1.02)
    cax = fig.add_axes([0.62, y_top - 0.075, 0.28, 0.014])
    MV.colorbar(fig, cax, MV.CM_DIFF, TwoSlopeNorm(0.0, -v, v), "ALT difference (cm)", MV.ticks_between(-v, v), M, MV.extend_of(corr, -v, v), orientation="horizontal")
    dl = _map_setup(MV, "lena"); ML = PROC / "maps" / "Lena"
    if (ML / "mean_2011_2024.csv.gz").exists():
        lm = pd.read_csv(ML / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
        p16 = pd.read_csv(ML / "pred_2016.csv.gz", dtype={"cell_id": str}).set_index("cell_id").r1.reindex(lm.index)
        lo, hi = MV.pct_range(lm.r1.values)
        aspl = (dl["ext"][3] - dl["ext"][2]) / (dl["ext"][1] - dl["ext"][0])
        phl = y_top - 0.075 - 0.07 - 0.09
        pwl = phl / (S.W2 / H) / aspl
        y_bot = 0.09
        ax = _draw_panel(fig, [0.06, y_bot, pwl, phl], MV, dl, lm.index, p16.values, MV.CM_ALT, Normalize(lo, hi), M, True, True,
                         label="Lena Delta ALT 2016 (R1; ERA5-Land 2016)")
        S.panel_letter(ax, "c", x=-0.02, y=1.02)
        an = (p16 - lm.r1).values; vl = MV.sym_max(an)
        ax = _draw_panel(fig, [0.56, y_bot, pwl, phl], MV, dl, lm.index, an, MV.CM_DIFF, TwoSlopeNorm(0.0, -vl, vl), M, False, True,
                         label="ALT 2016 − mean 2011–2024")
        S.panel_letter(ax, "d", x=-0.02, y=1.02)
        cax = fig.add_axes([0.10, 0.025, 0.28, 0.014])
        MV.colorbar(fig, cax, MV.CM_ALT, Normalize(lo, hi), "ALT (cm)", MV.ticks_between(lo, hi), M, MV.extend_of(lm.r1.values, lo, hi), orientation="horizontal")
        cax = fig.add_axes([0.60, 0.025, 0.28, 0.014])
        MV.colorbar(fig, cax, MV.CM_DIFF, TwoSlopeNorm(0.0, -vl, vl), "ALT difference (cm)", MV.ticks_between(-vl, vl), M, MV.extend_of(an, -vl, vl),
                    orientation="horizontal")
    return S.export(fig, "Fig7_aoa_lena")


FIGS.update({6: fig6, 7: fig7})


# ================================================================ Fig 8 (연도별 평가)
def fig8():
    """연도별 평가: (a) 알래스카 연도별 편향(우리 블록 CV·연도 홀드아웃 대 Wei·Liu), (b) 다년 셀의 연도 간 변화 Δpred 대 Δobs 상관, (c) 연도 편향 대 TDD 편차."""
    yw = pd.read_csv(RUN / "yearwise" / "yearwise_metrics.csv"); ch = pd.read_csv(RUN / "yearwise" / "multiyear_changes.csv")
    fig = S.figure(S.W2, 70)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42, left=0.07, right=0.99, top=0.9, bottom=0.3, width_ratios=[1.3, 1.0, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    keys = [("ours:R1:0.5:blockCV", S.METHOD_HEX["R1"], "-", "R1, block CV"), ("ours:R1:0.5:LOYO", S.METHOD_HEX["R1"], (0, (3, 1.5)), "R1, year hold-out"),
            ("ours:P1:blockCV", S.METHOD_HEX["P1"], "-", "P1, block CV"), ("prod:wei", S.PRODUCT_HEX["wei"], "-", "Wei 2026"), ("prod:liu", S.PRODUCT_HEX["liu"], "-", "Liu 2024")]
    a = yw[(yw.region == "Alaska") & (yw.n >= 100)]
    for k, col, ls, lab in keys:
        s = a[a.series == k].sort_values("year")
        ax.plot(s.year, s.bias, marker="o", ms=2.4, lw=0.9, ls=ls, color=col, label=lab)
    for _, r in a[a.series == "ours:R1:0.5:blockCV"].iterrows():
        ax.text(r.year, 19.5, f"{int(r.n)}", ha="center", va="bottom", fontsize=S.F["paper"]["text_pt"] - 2.5, color=S.NEUTRAL["text_aux"], rotation=90)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylim(-26, 27); ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_xlabel("Observation year (Alaska; cell-years per year above)"); ax.set_ylabel("Bias, prediction − observed (cm)")
    ax.legend(loc="lower left", fontsize=S.F["paper"]["text_pt"] - 1.5, ncol=2, handlelength=1.6, columnspacing=0.8); S.panel_letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    order = ["ours:P1:blockCV", "ours:R1:0.5:blockCV", "ours:D0:blockCV", "prod:wei", "prod:liu", "prod:cci5y"]
    labels = ["P1", "R1", "D0", "Wei", "Liu", "CCI v5"]
    for i, r in enumerate(REG):
        s = ch[ch.region == r].set_index("series").reindex(order)
        ax.plot(np.arange(len(order)) + (i - 1) * 0.22, s.r_delta.values, "o", ms=3.2, color=S.REGION_HEX[r], label=S.REGION_EN[r])
        ax.plot(np.arange(len(order)) + (i - 1) * 0.22, s.r_dobs_dtdd.values, "_", ms=7, mew=1.2, color=S.REGION_HEX[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=25, ha="right"); ax.set_ylabel("Correlation of year-to-year change\n(Δpred vs Δobs at multi-year cells)")
    ax.text(0.98, 0.97, "dash: corr(Δobs, ΔTDD)", transform=ax.transAxes, ha="right", va="top", fontsize=S.F["paper"]["text_pt"] - 1.5, color=S.NEUTRAL["text_aux"])
    ax.legend(loc="lower left", handlelength=1.0, fontsize=S.F["paper"]["text_pt"] - 1.5); S.panel_letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(REG):
        s = yw[(yw.region == r) & (yw.series == "ours:P1:blockCV") & (yw.n >= 50)]
        ax.plot(s.tdd_anom_mean, s.bias, "o", ms=3.0, color=S.REGION_HEX[r], label=S.REGION_EN[r])
        for _, q in s.iterrows():
            ax.text(q.tdd_anom_mean, q.bias, str(int(q.year))[2:], fontsize=S.F["paper"]["text_pt"] - 3, ha="left", va="bottom", color=S.REGION_HEX[r])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.axvline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0)
    ax.set_xlabel("Year TDD anomaly (°C·d)"); ax.set_ylabel("P1 bias in that year (cm)"); S.panel_letter(ax, "c")
    return S.export(fig, "Fig8_yearwise")


FIGS.update({8: fig8})


# ================================================================ Fig 9 (보강 실험: 강제력 정합·진단·민감도·새 방법)
def _load(path, **flt):
    p = Path(path)
    if not p.exists():
        return None
    d = pd.read_csv(p)
    for k, v in flt.items():
        d = d[d[k] == v]
    return d


def fig9():
    fig = S.figure(S.W2, 70)
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.5, left=0.06, right=0.99, top=0.9, bottom=0.3)
    # (a) 관측 월 정합 − 연도 정합(블록 CV), 방법별
    ax = fig.add_subplot(gs[0, 0])
    d = _load(PROC / "runs" / "forcing_obs" / "deltas_forcing.csv")
    if d is not None:
        d = d[(d.pair == "year_obs:year") & (d.scheme == "block") & ((d.method != "R1") | (d.lam == 0.5))]
        order = ["P0", "P1", "Ku", "D0", "R1"]
        for i, r in enumerate(REG):
            s = d[d.region == r].set_index("method").reindex(order).dropna(subset=["delta"])
            x = np.array([order.index(m) for m in s.index]) + (i - 1) * 0.22
            ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0, label=S.REGION_EN[r])
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0)
    ax.set_ylabel("RMSE change (cm)\nTDD to observation month − full year"); ax.legend(loc="best", handlelength=1.0, fontsize=S.F["paper"]["text_pt"] - 1.5)
    S.panel_letter(ax, "a")
    # (b) 알래스카 진단: 2014 제외 시 연도 정합 − 기후값
    ax = fig.add_subplot(gs[0, 1])
    base = _load(PROC / "runs" / "core_cb" / "deltas_forcing.csv"); dx = _load(PROC / "runs" / "ak_no2014" / "deltas_forcing.csv")
    order = ["P0", "P1", "R1"]
    for j, (d, lab, mk) in enumerate(((base, "all years", "o"), (dx, "2014 excluded", "s"))):
        if d is None:
            continue
        d = d[(d.region == "Alaska") & (d.scheme == "block") & ((d.method != "R1") | (d.lam == 0.5))]
        if "pair" in d:
            d = d[d.pair == "year:clim"]
        s = d.set_index("method").reindex(order).dropna(subset=["delta"])
        x = np.array([order.index(m) for m in s.index]) + (j - 0.5) * 0.25
        ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt=mk, ms=3.2, color=S.REGION_HEX["Alaska"], mfc=S.REGION_HEX["Alaska"] if j == 0 else "white",
                    lw=0.8, capsize=0, label=lab)
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("RMSE change (cm)\nyear-matched − climatological, Alaska"); ax.legend(loc="best", handlelength=1.0, fontsize=S.F["paper"]["text_pt"] - 1.5)
    S.panel_letter(ax, "b")
    # (c) 블록 크기 민감도: R1 − P1
    ax = fig.add_subplot(gs[0, 2])
    pts = []
    for deg, tag in ((0.25, "blk025"), (0.5, "core_cb"), (1.0, "blk100")):
        d = _load(PROC / "runs" / tag / "deltas_vs_ref.csv")
        if d is None:
            continue
        d = d[(d.ref == "P1") & (d.scheme == "block") & (d.method == "R1") & (d.lam == 0.5) & (d.forcing == "year") & (d.featset == "F8TS")]
        for _, r in d.iterrows():
            pts.append(dict(deg=deg, region=r.region, delta=r.delta, lo=r.lo, hi=r.hi, nb=r.n_blocks))
    if pts:
        p = pd.DataFrame(pts)
        for i, r in enumerate(REG):
            s = p[p.region == r].sort_values("deg")
            x = np.log2(s.deg / 0.5) + (i - 1) * 0.12
            ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o-", ms=3.0, lw=0.8, color=S.REGION_HEX[r], capsize=0, label=S.REGION_EN[r])
        ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["0.25°", "0.5°", "1.0°"])
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_xlabel("Spatial block size"); ax.set_ylabel("RMSE change, R1 − P1 (cm)")
    S.panel_letter(ax, "c")
    # (d) 새 방법·학습기 변형: Ed, PE, R1cv, R1 연도가중, R1 앙상블 대 P1(블록 CV)
    ax = fig.add_subplot(gs[0, 3])
    rows = []
    for tag, meths, label in (("methods2", ["Ed", "PE", "R1cv"], None), ("yearweight", ["R1"], "R1 yr-wt"), ("core_cb", ["R1"], "R1")):
        d = _load(PROC / "runs" / tag / "deltas_vs_ref.csv")
        if d is None:
            continue
        d = d[(d.ref == "P1") & (d.scheme == "block") & d.method.isin(meths) & ((d.method != "R1") | (d.lam == 0.5))]
        if "forcing" in d:
            d = d[d.forcing == "year"]
        for _, r in d.iterrows():
            rows.append(dict(region=r.region, name=label or r.method, delta=r.delta, lo=r.lo, hi=r.hi))
    if rows:
        p = pd.DataFrame(rows); order = ["Ed", "PE", "R1", "R1cv", "R1 yr-wt"]
        for i, r in enumerate(REG):
            s = p[p.region == r].set_index("name").reindex(order).dropna(subset=["delta"])
            x = np.array([order.index(n) for n in s.index]) + (i - 1) * 0.22
            ax.errorbar(x, s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="o", ms=3.0, color=S.REGION_HEX[r], lw=0.8, capsize=0)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25, ha="right")
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=0.6, zorder=0); ax.set_ylabel("RMSE change vs P1 (cm), block CV")
    S.panel_letter(ax, "d")
    return S.export(fig, "Fig9_supplementary")


FIGS.update({9: fig9})


# ================================================================ Fig 10 (공통 셀 직관 비교)
def fig10():
    """지역마다 Wei·Liu·CCI v4 가 모두 값을 가진 같은 셀-연도에서 우리 방법과 제품의 RMSE. 채움 = 전체, 빈 마커 = 제품 학습 지점 25 km 밖."""
    d = pd.read_csv(RUN / "common_cells_eval.csv")
    order = ["P1 (Stefan, recal.)", "R1 (CatBoost)", "R1 (5-learner ens.)", "D0 (CatBoost)", "D0 (5-learner ens.)", "Wei 2026", "Liu 2024", "ESA CCI v4", "ESA CCI v5", "Aalto 2018"]
    short = ["P1", "R1", "R1 ens.", "D0", "D0 ens.", "Wei 2026", "Liu 2024", "CCI v4", "CCI v5", "Aalto 2018"]
    col = {"P1 (Stefan, recal.)": S.METHOD_HEX["P1"], "R1 (CatBoost)": S.METHOD_HEX["R1"], "R1 (5-learner ens.)": S.METHOD_HEX["R1"], "D0 (CatBoost)": S.METHOD_HEX["D0"],
           "D0 (5-learner ens.)": S.METHOD_HEX["D0"], "Wei 2026": S.PRODUCT_HEX["wei"], "Liu 2024": S.PRODUCT_HEX["liu"], "ESA CCI v4": S.PRODUCT_HEX["cci4y"],
           "ESA CCI v5": S.PRODUCT_HEX["cci5y"], "Aalto 2018": S.PRODUCT_HEX["aalto"]}
    fig = S.figure(S.W2, 72)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3, left=0.06, right=0.99, top=0.9, bottom=0.3)
    YMAX = {"Alaska": 40, "Lena": 40, "Canada": 40}
    for j, r in enumerate(REG):
        ax = fig.add_subplot(gs[0, j])
        for k, name in enumerate(order):
            for mask, mk in ((0, "o"), (25, "o")):
                q = d[(d.region == r) & (d.series == name) & (d.mask_km == mask)]
                if q.empty:
                    continue
                v = float(q.rmse.iloc[0]); x = k + (-0.15 if mask == 0 else 0.15)
                if v > YMAX[r]:
                    ax.text(x, YMAX[r] - 0.5, f"{v:.0f}", ha="center", va="top", fontsize=S.F["paper"]["text_pt"] - 2, color=col[name], rotation=90)
                else:
                    ax.plot(x, v, mk, ms=3.6, color=col[name], mfc=col[name] if mask == 0 else "white", mew=0.9)
        n0 = int(d[(d.region == r) & (d.mask_km == 0)].n.iloc[0]); n25 = d[(d.region == r) & (d.mask_km == 25)].n
        ax.text(0.02, 0.97, f"{S.REGION_EN[r]}\ncommon cell-years {n0:,} (all), {int(n25.iloc[0]) if len(n25) else 0:,} (> 25 km)", transform=ax.transAxes, va="top", ha="left",
                fontsize=S.F["paper"]["text_pt"] - 1)
        ax.axvline(4.5, color=S.NEUTRAL["grid"], lw=0.6, zorder=0)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(short, rotation=40, ha="right"); ax.set_ylim(10, YMAX[r])
        if j == 0:
            ax.set_ylabel("RMSE on common cell-years (cm)")
        S.panel_letter(ax, "abc"[j])
    fig.text(0.52, 0.02, "filled: all common cells; open: cells > 25 km from product training sites. Ours = block-CV out-of-fold; products = year-matched extraction.",
             ha="center", va="bottom", fontsize=S.F["paper"]["text_pt"] - 1, color=S.NEUTRAL["text_aux"])
    return S.export(fig, "Fig10_common_cells")


FIGS.update({10: fig10})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", type=int, action="append", default=None); ap.add_argument("--slide", action="store_true")
    a = ap.parse_args(); S.setup(scale=1.69 if a.slide else 1.0)
    for k in (a.fig or sorted(FIGS)):
        FIGS[k](); print(f"[fig{k}] done")
