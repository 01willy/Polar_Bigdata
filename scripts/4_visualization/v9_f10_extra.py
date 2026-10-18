# -*- coding: utf-8 -*-
"""v9_f10_extra.py: F10 추가 실험 그림(2026-10-15). 입력 data/processed/v5/runs/extra_v9/{pa_metrics,pa_deltas,ext_metrics,shap_summary,shap_groups}.csv,
data/processed/v9_ext/labels_cell_year_v5.csv, data/processed/v5/labels_cell_year_v5.csv.
덱판(12 × 4.65 in, 300 dpi): F10_extra__deck_PA.png(제품 기준값 대 Stefan 기준값의 잔차 결합, 같은 셀-연도 RMSE),
  __deck_EXT.png(외부 지역 무라벨 시험: 극 지도 + 소지역별 RMSE), __deck_SHAP.png(잔차 학습기 입력 기여).
논문판 F10_extra.{pdf,svg,png}(180 × 150 mm, a PA · b EXT · c SHAP).
실행: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v9_f10_extra.py [--deck] [--paper]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S  # noqa: E402
RUN = ROOT / "data" / "processed" / "v5" / "runs" / "extra_v9"; EXT = ROOT / "data" / "processed" / "v9_ext"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; DECK = OUT / "deck"; STEM = "F10_extra"
REG = ["Alaska", "Lena", "Canada"]
GRAY = S.NEUTRAL.get("text_aux", "#6b6b6b") if isinstance(S.NEUTRAL, dict) else "#6b6b6b"
ZERO = S.NEUTRAL.get("zero_line", "#9a9a9a") if isinstance(S.NEUTRAL, dict) else "#9a9a9a"
C_W, C_L, C_C5 = S.PRODUCT_HEX["wei"], S.PRODUCT_HEX["liu"], S.PRODUCT_HEX["cci5y"]
C_P1, C_R1 = S.METHOD_HEX["P1"], S.METHOD_HEX["R1"]
FEAT = {"e5_twarm": "Warmest-month T", "e5_tcold": "Coldest-month T", "e5_maat": "Mean annual T", "e5_tdd": "TDD", "e5_fdd": "FDD",
        "e5_sqrt_tdd": r"$\sqrt{\mathrm{TDD}}$", "e5_swe": "Snow water eq.", "e5_stl1": "Soil T (0–7 cm)", "dem_elev": "Elevation", "dem_slope": "Slope",
        "dem_aspect_sin": "Aspect (E–W)", "dem_aspect_cos": "Aspect (N–S)", "dem_tpi": "Topographic position", "dem_rough": "Roughness",
        "sg_clay_5_15": "Clay", "sg_sand_5_15": "Sand", "sg_silt_5_15": "Silt", "sg_bdod_5_15": "Bulk density", "sg_cfvo_5_15": "Coarse fragments",
        "sg_phh2o_5_15": "Soil pH", "sg_soc_0_5": "SOC 0–5 cm", "sg_soc_5_15": "SOC 5–15 cm", "sg_soc_15_30": "SOC 15–30 cm"}
GROUP_HEX = {"climate": "#4F7CAC", "terrain": "#8C8C5A", "soil": "#A8744F"}
NUM_OFF = {"W Siberia": (0, -26), "C Siberia": (18, -14), "European Russia": (-22, -10), "NE Siberia": (-20, 14), "Chukotka": (-18, 10),
           "Kamchatka": (-16, -8), "Svalbard": (14, 12), "Greenland": (-14, 12)}
SUB_ORDER = ["W Siberia", "C Siberia", "European Russia", "NE Siberia", "Chukotka", "Kamchatka", "Svalbard", "Greenland"]


def pa_rows():
    m = pd.read_csv(RUN / "pa_metrics.csv"); return m[m["product"] == "wei"].pivot_table(index="region", columns="method", values="rmse")


PA_ROWS = [("Praw", "Wei 2026", C_W, "D", True), ("Pcal", "Wei 2026, recalibrated", C_W, "D", False), ("RP", "Wei 2026 + residual ML", C_W, "o", True),
           ("P1", "Stefan, recalibrated", C_P1, "s", False), ("R1", "Stefan + residual ML", C_R1, "o", True)]


def draw_pa(axes, M):
    t = pa_rows()
    for ax, r in zip(axes, REG):
        for i, (k, name, col, mk, filled) in enumerate(PA_ROWS):
            y = len(PA_ROWS) - 1 - i
            ax.plot([t.loc[r, k]], [y], ls="none", marker=mk, ms=M["ms"], mfc=col if filled else "white", mec=col, mew=M["mew"], zorder=3)
        ax.axhline(1.5, color=ZERO, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(range(len(PA_ROWS))); ax.set_yticklabels([n for _, n, *_ in PA_ROWS][::-1] if ax is axes[0] else [])
        ax.tick_params(axis="y", length=0, labelsize=M["tick"]); ax.tick_params(axis="x", labelsize=M["tick"])
        lo, hi = np.nanmin(t.loc[r].values), np.nanmax(t.loc[r].values); pad = max(1.5, 0.12 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.6, len(PA_ROWS) - 0.4)
        ax.set_title(S.REGION_EN[r], loc="left", fontsize=M["lab"], color=S.REGION_HEX[r], pad=6)
        if ax is axes[1]:
            ax.set_xlabel("RMSE on the same cell-years (cm)", fontsize=M["lab"])
        ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
        ax.spines["left"].set_visible(False)


def ext_tables():
    e = pd.read_csv(RUN / "ext_metrics.csv"); e = e[e.rows == "own"]
    e = e[e.subset.str.startswith("in_range|") | (e.subset == "in_range")].copy()
    e["sub"] = e.subset.str.replace("in_range|", "", regex=False).replace({"in_range": "All sites"})
    return e.pivot_table(index="sub", columns="series", values="rmse"), e.pivot_table(index="sub", columns="series", values="n")


EXT_SER = [("R1", "Ours, source regions only", C_R1, "o", True), ("R1loc", "Ours, Stefan recalibrated locally", C_R1, "o", False),
           ("wei", "Wei 2026, trained on these sites", C_W, "D", True), ("liu", "Liu 2024, trained on these sites", C_L, "D", False),
           ("cci5y", "ESA CCI v5, no ALT training", C_C5, "s", True)]


def draw_ext_map(ax, M):
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    PC = ccrs.PlateCarree()
    ax.set_extent([-180, 180, 55, 90], PC)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f2f2", edgecolor="none", zorder=0)
    ax.coastlines("50m", linewidth=0.4, color="#9a9a9a", zorder=1)
    ax.gridlines(crs=PC, draw_labels=False, linewidth=0.4, color="#d0d0d0", xlocs=range(-180, 180, 60), ylocs=[60, 70, 80], zorder=1)
    lab = pd.read_csv(ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv", usecols=["region", "lat", "lon"]).drop_duplicates()
    for r in REG:
        q = lab[lab.region == r]; ax.scatter(q.lon, q.lat, s=M["dot"] * 0.5, color=S.REGION_HEX[r], transform=PC, zorder=3, lw=0)
    ex = pd.read_csv(EXT / "labels_cell_year_v5.csv").drop_duplicates("loc_id")
    ax.scatter(ex.lon, ex.lat, s=M["dot"], facecolor="white", edgecolor="#222222", lw=0.9, transform=PC, zorder=4)
    ex_all = pd.read_csv(EXT / "labels_cell_year_v5.csv").drop_duplicates("loc_id")
    for k, name in enumerate([n for n in SUB_ORDER if n in set(ex_all.region)], start=1):
        q = ex_all[ex_all.region == name]; xy = ax.projection.transform_points(PC, q.lon.values, q.lat.values)[:, :2]
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean(); dx, dy = NUM_OFF.get(name, (12, 0))
        ax.annotate(str(k), (cx, cy), xytext=(dx, dy), textcoords="offset points", fontsize=M["leg"], fontweight="bold", color="#222222",
                    ha="center", va="center", zorder=6, bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#222222", lw=0.6))
    hs = [Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec="#222222", label="External CALM sites")] + \
         [Line2D([], [], ls="none", marker="o", ms=5, mfc=S.REGION_HEX[r], mec="none", label=f"{S.REGION_EN[r]} (training)") for r in REG]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.02, -0.01), fontsize=M["leg"], frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)


def draw_ext_dots(ax, M, xmin=4, xmax=160):
    t, n = ext_tables(); subs = [s for s in SUB_ORDER if s in t.index] + ["All sites"]
    for j, (k, name, col, mk, filled) in enumerate(EXT_SER):
        off = (j - (len(EXT_SER) - 1) / 2) * 0.13
        for i, s in enumerate(subs):
            v = t.loc[s, k] if k in t else np.nan
            if not np.isfinite(v):
                continue
            y = len(subs) - 1 - i + off
            ax.plot([v], [y], ls="none", marker=mk, ms=M["ms"] * 0.8, mfc=col if filled else "white", mec=col, mew=M["mew"], zorder=3)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([(f"{i}  {s} ({int(n.loc[s, 'R1'])})" if s != "All sites" else f"All sites ({int(n.loc[s, 'R1'])})") for i, s in enumerate(subs, start=1)][::-1], fontsize=M["tick"])
    ax.axhline(0.5, color=ZERO, lw=0.8, ls=(0, (4, 3)))
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax); ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.minorticks_off(); ax.set_ylim(-0.6, len(subs) - 0.4); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=M["tick"])
    ax.set_xlabel("RMSE (cm, log scale)", fontsize=M["lab"]); ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0); ax.spines["left"].set_visible(False)
    hs = [Line2D([], [], ls="none", marker=mk, ms=7, mfc=col if f else "white", mec=col, mew=M["mew"], label=nm) for _, nm, col, mk, f in EXT_SER]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=M["leg"], frameon=False, handletextpad=0.3, labelspacing=0.9)


def draw_shap(ax_bar, axes_top, M):
    g = pd.read_csv(RUN / "shap_groups.csv"); s = pd.read_csv(RUN / "shap_summary.csv")
    ylab = []
    for i, r in enumerate(REG):
        for j, kind in enumerate(("residual", "direct")):
            y = (len(REG) - 1 - i) * 2.6 + (1 - j) * 1.0; left = 0.0
            q = g[(g.region == r) & (g.kind == kind)].set_index("group").share
            for grp in ("climate", "terrain", "soil"):
                v = float(q.get(grp, 0.0)); ax_bar.barh(y, v, left=left, color=GROUP_HEX[grp], height=0.8, edgecolor="white", lw=0.8)
                if v >= 0.12:
                    ax_bar.text(left + v / 2, y, f"{v * 100:.0f}%", ha="center", va="center", fontsize=M["leg"] - 1, color="white")
                left += v
            ylab.append((y, f"{S.REGION_EN[r]}, {'residual ML' if kind == 'residual' else 'direct ML'}", r))
    ax_bar.set_yticks([y for y, _, _ in ylab]); ax_bar.set_yticklabels([t for _, t, _ in ylab], fontsize=M["tick"])
    for tl, (_, _, r) in zip(ax_bar.get_yticklabels(), ylab):
        tl.set_color(S.REGION_HEX[r])
    ax_bar.set_xlim(0, 1); ax_bar.set_xticks([0, 0.5, 1]); ax_bar.set_xticklabels(["0", "50", "100"], fontsize=M["tick"])
    ax_bar.set_xlabel("Share of mean |SHAP| (%)", fontsize=M["lab"]); ax_bar.tick_params(axis="y", length=0); ax_bar.spines["left"].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=GROUP_HEX[k], label=k.capitalize()) for k in ("climate", "terrain", "soil")]
    ax_bar.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=M["leg"], frameon=False, handlelength=1.0, columnspacing=1.0)
    grp = {f: ("climate" if f.startswith("e5_") else "terrain" if f.startswith("dem_") else "soil") for f in s.feature.unique()}
    xmax_top = float(np.ceil(s[s.kind == "residual"].mean_abs.max()))
    for ax, r in zip(axes_top, REG):
        q = s[(s.region == r) & (s.kind == "residual")].sort_values("mean_abs", ascending=False).head(4)[::-1]
        ax.barh(range(len(q)), q.mean_abs, color=[GROUP_HEX[grp[f]] for f in q.feature], height=0.65)
        ax.set_yticks(range(len(q))); ax.set_yticklabels([FEAT.get(f, f) for f in q.feature], fontsize=M["tick"])
        ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=M["tick"]); ax.spines["left"].set_visible(False)
        ax.set_title(S.REGION_EN[r], loc="left", fontsize=M["lab"], color=S.REGION_HEX[r], pad=4)
        ax.set_xlim(0, xmax_top); ax.set_ylim(-0.6, len(q) - 0.4)
        if ax is axes_top[-1]:
            ax.set_xlabel("Mean |SHAP| of the residual learner (cm)", fontsize=M["lab"])
        else:
            ax.set_xticklabels([])


FT_SUBS = ["W Siberia", "C Siberia", "European Russia", "NE Siberia", "Chukotka", "Svalbard"]
FT_NS = ["0", "5", "10", "25", "50", "all"]


def ft_table():
    """외부 지역 소수 라벨 적응: 씨앗 평균 예측 → 세 제품(Wei · Liu · ESA CCI v5) 값이 모두 있는 같은 행에서 방법별 RMSE(전체 · 소지역)와 제품 RMSE."""
    import glob
    p = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RUN / "preds_EXTFT_*.csv.gz")))], ignore_index=True); p["n_label"] = p.n_label.astype(str)
    first = p.drop_duplicates("cell_year_id").set_index("cell_year_id")[["y", "subregion", "prod_wei", "prod_liu", "prod_cci5y"]]
    rows = []
    for nl in FT_NS:
        w = p[p.n_label == nl].groupby(["cell_year_id", "method"]).pred.mean().unstack().join(first)
        w = w[w.prod_wei.notna() & w.prod_liu.notna() & w.prod_cci5y.notna()]
        for scope, q in [("all", w)] + [(r, g) for r, g in w.groupby("subregion")]:
            for m in ("P1n", "R1S", "D0S"):
                rows.append(dict(scope=scope, n_label=nl, series=m, rmse=float(np.sqrt(np.mean((q[m] - q.y) ** 2))), n=len(q)))
            for ser, col in (("wei", "prod_wei"), ("liu", "prod_liu"), ("cci5y", "prod_cci5y")):
                rows.append(dict(scope=scope, n_label=nl, series=ser, rmse=float(np.sqrt(np.mean((q[col] - q.y) ** 2))), n=len(q)))
    return pd.DataFrame(rows)


def draw_ft(axa, axb, M):
    t = ft_table(); a = t[t.scope == "all"]; xs = np.arange(len(FT_NS))
    for m, nm, col, ls, mk, filled, lw, ms, z in (("P1n", "Stefan, recalibrated on n", C_P1, "-", "s", False, M["lw"] * 2.4, M["ms"] * 1.05, 2),
                                                  ("R1S", "Physics + residual ML", C_R1, "-", "o", True, M["lw"] * 0.9, M["ms"] * 0.55, 3),
                                                  ("D0S", "Direct ML", S.METHOD_HEX["D0"], (0, (3.5, 1.8)), "o", False, M["lw"], M["ms"] * 0.7, 3)):
        v = [a[(a.series == m) & (a.n_label == n)].rmse.iloc[0] for n in FT_NS]
        axa.plot(xs, v, ls=ls, color=col, lw=lw, marker=mk, ms=ms, mfc=col if filled else "white", mec=col, mew=M["mew"], label=nm, zorder=z,
                 alpha=0.55 if m == "P1n" else 1.0)
    for ser, nm, col in (("cci5y", "ESA CCI v5, independent", C_C5), ("wei", "Wei 2026, trained on these sites", C_W), ("liu", "Liu 2024, trained on these sites", C_L)):
        v = a[(a.series == ser) & (a.n_label == "0")].rmse.iloc[0]
        axa.axhline(v, color=col, lw=M["lw"] * 0.8, ls=(0, (5, 2)), label=nm)
    axa.set_xticks(xs); axa.set_xticklabels(["0", "5", "10", "25", "50", "all"], fontsize=M["tick"]); axa.tick_params(axis="y", labelsize=M["tick"])
    axa.set_ylim(0, 70); axa.set_xlabel("Labels from the same external region", fontsize=M["lab"]); axa.set_ylabel("RMSE (cm)", fontsize=M["lab"])
    axa.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=M["leg"], frameon=False, handlelength=1.8, labelspacing=0.35)
    axa.grid(axis="y", color="#eeeeee", lw=0.6)
    subs = [r for r in FT_SUBS if r in set(t.scope)]
    for i, r in enumerate(subs):
        q = t[t.scope == r]; y = len(subs) - 1 - i
        v0 = q[(q.series == "R1S") & (q.n_label == "0")].rmse.iloc[0]; v5 = q[(q.series == "R1S") & (q.n_label == "5")].rmse.iloc[0]
        vc = q[(q.series == "cci5y") & (q.n_label == "0")].rmse.iloc[0]
        axb.plot([v0, v5], [y, y], color="#cccccc", lw=1.4, zorder=1)
        axb.plot([v0], [y], "o", ms=M["ms"] * 0.75, mfc="white", mec=C_R1, mew=M["mew"], zorder=3)
        axb.plot([v5], [y], "o", ms=M["ms"] * 0.75, color=C_R1, zorder=3)
        axb.plot([vc], [y], "s", ms=M["ms"] * 0.6, color=C_C5, zorder=2)
    axb.set_yticks(range(len(subs))); axb.set_yticklabels(subs[::-1], fontsize=M["tick"]); axb.tick_params(axis="y", length=0); axb.tick_params(axis="x", labelsize=M["tick"])
    axb.set_xscale("log"); axb.set_xticks([5, 10, 20, 50, 100]); axb.set_xticklabels(["5", "10", "20", "50", "100"]); axb.minorticks_off(); axb.set_xlim(4, 200)
    axb.set_xlabel("RMSE (cm, log scale)", fontsize=M["lab"]); axb.spines["left"].set_visible(False); axb.grid(axis="x", color="#eeeeee", lw=0.6)
    hs = [Line2D([], [], ls="none", marker="o", ms=8, mfc="white", mec=C_R1, mew=M["mew"], label="No local labels"),
          Line2D([], [], ls="none", marker="o", ms=8, color=C_R1, label="5 local labels"),
          Line2D([], [], ls="none", marker="s", ms=7, color=C_C5, label="ESA CCI v5")]
    axb.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.585, 0.865), bbox_transform=axb.figure.transFigure, fontsize=M["leg"], frameon=False, ncol=3,
               handletextpad=0.2, columnspacing=0.8)
    return t


def save_deck(fig, key):
    from PIL import Image
    DECK.mkdir(parents=True, exist_ok=True); p = DECK / f"{STEM}__deck_{key}.png"
    fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    with Image.open(p) as im:
        im.convert("RGB").save(p, dpi=(300, 300)); print(f"[deck {key}] {p} {im.size}")


def deck_ft():
    S.setup(scale=1.69); plt.rcParams.update({"font.size": 13})
    M = dict(ms=11, mew=1.6, tick=13, lab=14, leg=12, dot=36, lw=2.2)
    fig = plt.figure(figsize=(12.0, 4.65)); fig.patch.set_facecolor("white")
    axa = fig.add_axes([0.07, 0.15, 0.30, 0.80]); axb = fig.add_axes([0.725, 0.15, 0.255, 0.70])
    t = draw_ft(axa, axb, M); t.to_csv(OUT / "source_data" / "F10_extft.csv", index=False); save_deck(fig, "FT")


def deck():
    import cartopy.crs as ccrs
    S.setup(scale=1.69); plt.rcParams.update({"font.size": 13})
    M = dict(ms=11, mew=1.6, tick=13, lab=14, leg=12, dot=36)
    fig = plt.figure(figsize=(12.0, 4.65)); fig.patch.set_facecolor("white")
    axes = [fig.add_axes([0.205 + j * 0.27, 0.17, 0.215, 0.72]) for j in range(3)]
    draw_pa(axes, M); save_deck(fig, "PA")
    fig = plt.figure(figsize=(12.0, 4.65)); fig.patch.set_facecolor("white")
    axm = fig.add_axes([0.0, 0.20, 0.335, 0.78], projection=ccrs.NorthPolarStereo(central_longitude=90))
    draw_ext_map(axm, M)
    axd = fig.add_axes([0.525, 0.14, 0.185, 0.82]); draw_ext_dots(axd, M); save_deck(fig, "EXT")
    fig = plt.figure(figsize=(12.0, 4.65)); fig.patch.set_facecolor("white")
    axb = fig.add_axes([0.20, 0.15, 0.22, 0.70]); axt = [fig.add_axes([0.66, 0.70 - j * 0.265, 0.30, 0.185]) for j in range(3)]
    draw_shap(axb, axt, M); save_deck(fig, "SHAP")


def paper():
    import cartopy.crs as ccrs
    S.setup(scale=1.0); M = dict(ms=4.2, mew=0.8, tick=6.5, lab=7, leg=6, dot=6)
    fig = S.figure(S.W2, 190)
    axa = [fig.add_axes([0.20 + j * 0.27, 0.775, 0.21, 0.195]) for j in range(3)]; draw_pa(axa, M); S.panel_letter(axa[0], "a", x=-0.85); S.register("a", *axa[1:])
    axm = fig.add_axes([0.0, 0.43, 0.38, 0.29], projection=ccrs.NorthPolarStereo(central_longitude=90)); draw_ext_map(axm, M); S.panel_letter(axm, "b", x=0.0, y=1.0)
    axd = fig.add_axes([0.56, 0.45, 0.19, 0.26]); draw_ext_dots(axd, M); S.register("b", axd)
    axb = fig.add_axes([0.20, 0.035, 0.22, 0.215]); axt = [fig.add_axes([0.64, 0.205 - j * 0.09, 0.30, 0.058]) for j in range(3)]
    draw_shap(axb, axt, M); S.panel_letter(axb, "c", x=-0.85, y=1.12); S.register("c", *axt)
    return S.export(fig, STEM, outdir=OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--deck", action="store_true"); ap.add_argument("--paper", action="store_true"); ap.add_argument("--ft", action="store_true")
    a = ap.parse_args()
    if a.ft:
        deck_ft(); raise SystemExit
    if a.deck or not a.paper:
        deck()
    if a.paper:
        print(paper())
