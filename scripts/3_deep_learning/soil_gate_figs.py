"""W2.1 SoilGrids 게이트 시각화 4종.

1. 토양 지도(알래스카+레나, sg_clay·sg_soc): IGH tif 를 WGS84 산점으로 렌더(냉색 순차).
2. 피처셋별 skill 막대(공간블록 vs LORO).
3. sg_ vs ALT 산점·상관(알래스카 vs 레나: 왜 전이서 무너지나).
4. 지역별 sg_ 유효율 막대.

시각 규약: polar.plotstyle(냉색 순차, 붉은/rainbow 금지, 축 단위 필수). PNG+PDF.
실행: /home/anaconda3/bin/python scripts/3_deep_learning/soil_gate_figs.py
"""
import os
import sys
import glob
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import spearmanr
from polar.plotstyle import (use_polar, CMAP, add_cbar, style_geo, despine,
                             lon_formatter, lat_formatter, FROZEN, THAWED)
from polar.outputs import figpath, mappath

plt = use_polar()
PROC = "data/processed"
RAW = "data/raw/soilgrids_wcs"
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
INV = Transformer.from_crs(IGH, "EPSG:4326", always_xy=True)  # IGH->WGS84

df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell_v3_soil.csv"), low_memory=False)
gate = pd.read_csv(os.path.join(PROC, "soil_ablation_gate.csv"))
loro = pd.read_csv(os.path.join(PROC, "soil_ablation_gate_loro.csv"))
valid = pd.read_csv(os.path.join(PROC, "soilgrids_wcs_valid_by_region.csv"))
SOIL = sorted([c for c in df.columns if c.startswith("sg_")])

FACTOR = {"clay": 10.0, "sand": 10.0, "silt": 10.0, "soc": 10.0, "bdod": 100.0,
          "cfvo": 10.0, "phh2o": 10.0}


def tif_to_wgs84_points(path, prop, stride=1):
    """IGH tif → (lon, lat, val_physical) 산점(유효 픽셀만). 표시용 다운샘플 stride."""
    with rasterio.open(path) as src:
        b = src.read(1).astype(float)
        b = np.where(b <= 0, np.nan, b)  # nodata
        h, w = b.shape
        rr, cc = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing="ij")
        rr = rr.ravel(); cc = cc.ravel()
        xs, ys = rasterio.transform.xy(src.transform, rr, cc)
        vals = b[rr, cc]
    lon, lat = INV.transform(np.asarray(xs), np.asarray(ys))
    m = np.isfinite(vals)
    return np.asarray(lon)[m], np.asarray(lat)[m], vals[m] / FACTOR[prop]


# ============ 그림 1: 토양 지도(알래스카·레나 clay·soc) ============
def fig_soil_maps():
    panels = [
        ("namerica", "clay", "clay_5-15cm_mean.tif", "점토 함량 (%)", "알래스카·서캐나다"),
        ("namerica", "soc", "soc_5-15cm_mean.tif", "토양 유기탄소 (g/kg)", "알래스카·서캐나다"),
        ("lena", "clay", "clay_5-15cm_mean.tif", "점토 함량 (%)", "레나 삼각주"),
        ("lena", "soc", "soc_5-15cm_mean.tif", "토양 유기탄소 (g/kg)", "레나 삼각주"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for ax, (win, prop, tif, lab, place) in zip(axes.ravel(), panels):
        path = os.path.join(RAW, win, tif)
        lon, lat, val = tif_to_wgs84_points(path, prop, stride=1)
        vlo, vhi = np.nanpercentile(val, [2, 98])
        sc = ax.scatter(lon, lat, c=val, s=4, cmap=CMAP.alt, vmin=vlo, vmax=vhi,
                        marker="s", linewidths=0, rasterized=True)
        # 관측 셀 위치 오버레이
        reg_cells = df[df.region.isin(
            ["ABoVE_AK", "ABoVE_CA", "United States (Alaska)", "Canada", "GTNPenv_US"]
            if win == "namerica" else ["Lena_RU"])]
        ax.scatter(reg_cells.lon, reg_cells.lat, s=2, c="k", alpha=0.25,
                   linewidths=0, label="관측 셀")
        add_cbar(fig, sc, ax, lab)
        style_geo(ax, title=f"{place} · SoilGrids {prop}")
        ax.xaxis.set_major_formatter(lon_formatter())
        ax.yaxis.set_major_formatter(lat_formatter())
        despine(ax)
    fig.suptitle("SoilGrids 토양 공변량(5-15cm): 알래스카(학습) 대 레나(전이 대상)",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(mappath("soilgrids_ak_lena", ext))
    plt.close(fig)
    print("[fig1] soilgrids_ak_lena  ->", mappath("soilgrids_ak_lena", "png"))


# ============ 그림 2: 피처셋별 skill 막대(공간블록 vs LORO) ============
def fig_skill_bars():
    order = ["M_clim", "M_climterr", "M_soil", "M_soilonly"]
    labels = {"M_clim": "기후8", "M_climterr": "기후+지형\n(baseline)",
              "M_soil": "기후+지형+토양", "M_soilonly": "기후+토양"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, cv, title in [(a1, "spatial_block", "공간블록 CV (내삽)"),
                          (a2, "LORO", "LORO (지역 전이)")]:
        sub = gate[gate.cv_type == cv].set_index("config")
        sk = [sub.loc[c, "skill_over_mean"] * 100 for c in order]
        rm = [sub.loc[c, "rmse_cm"] for c in order]
        cols = [CMAP.count(0.55) if c != "M_climterr" else "#888888" for c in order]
        # 토양 포함 셋 강조
        cols = ["#9ecae1", "#888888", FROZEN, "#6baed6"]
        x = np.arange(len(order))
        bars = ax.bar(x, sk, color=cols, edgecolor="#333", linewidth=0.7, width=0.66)
        for xi, (s, r) in enumerate(zip(sk, rm)):
            ax.text(xi, s + (1.2 if s >= 0 else -1.2), f"{s:.1f}%\n{r:.1f}cm",
                    ha="center", va="bottom" if s >= 0 else "top", fontsize=8.6)
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in order], fontsize=9)
        ax.set_ylabel("skill-over-mean (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.4)
        despine(ax)
        ax.margins(y=0.18)
    fig.suptitle("피처셋별 예측력: 토양은 내삽서 이득, 전이서 손실",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(figpath("09_soilgrids", "soil_skill_bars", ext))
    plt.close(fig)
    print("[fig2] soil_skill_bars ->", figpath("09_soilgrids", "soil_skill_bars", "png"))


# ============ 그림 3: sg_ vs ALT 산점·상관(AK vs Lena) ============
def fig_soil_alt_scatter():
    cols = ["sg_soc_5_15", "sg_clay_5_15", "sg_bdod_5_15"]
    labs = {"sg_soc_5_15": "토양 유기탄소 (g/kg)", "sg_clay_5_15": "점토 함량 (%)",
            "sg_bdod_5_15": "용적밀도 (kg/dm³)"}
    ak = df[df.region == "ABoVE_AK"]
    le = df[df.region == "Lena_RU"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, c in zip(axes, cols):
        for g, color, name in [(ak, FROZEN, "알래스카(학습)"), (le, "#e08214", "레나(전이)")]:
            x = g[c].values; y = g.alt_cm.values
            m = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[m], y[m], s=6, c=color, alpha=0.28, linewidths=0)
            rho = spearmanr(x[m], y[m]).correlation
            ax.scatter([], [], c=color, s=24, label=f"{name}  ρ={rho:+.2f}")
        ax.set_xlabel(labs[c])
        ax.set_ylabel("활성층 두께 ALT (cm)")
        ax.set_ylim(0, 150)
        ax.legend(fontsize=8.4, loc="upper right")
        ax.grid(alpha=0.35)
        despine(ax)
    fig.suptitle("토양·ALT 관계의 지역 불일치: 알래스카서 학습된 관계가 레나서 소멸",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(figpath("09_soilgrids", "soil_alt_scatter", ext))
    plt.close(fig)
    print("[fig3] soil_alt_scatter ->", figpath("09_soilgrids", "soil_alt_scatter", "png"))


# ============ 그림 4: 지역별 sg_ 유효율 막대 ============
def fig_valid_bars():
    v = valid.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    y = np.arange(len(v))
    vals = v.sg_valid_avg_pct.values
    cols = [CMAP.count(0.35 + 0.55 * (x / 100)) for x in vals]
    ax.barh(y, vals, color=cols, edgecolor="#333", linewidth=0.6)
    for yi, (val, n) in enumerate(zip(vals, v.n.values)):
        ax.text(min(val + 1.5, 82), yi, f"{val:.0f}%  (n={n})", va="center", fontsize=8.6)
    ax.set_yticks(y)
    ax.set_yticklabels(v.region.values, fontsize=9)
    ax.set_xlabel("sg_ 평균 유효율 (%)")
    ax.set_xlim(0, 118)
    ax.set_title("지역별 SoilGrids 공변량 유효율(전지구 커버, 남극 제외 완전)")
    ax.grid(axis="x", alpha=0.4)
    despine(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figpath("09_soilgrids", "soil_valid_by_region", ext))
    plt.close(fig)
    print("[fig4] soil_valid_by_region ->", figpath("09_soilgrids", "soil_valid_by_region", "png"))


if __name__ == "__main__":
    fig_soil_maps()
    fig_skill_bars()
    fig_soil_alt_scatter()
    fig_valid_bars()
    print("\n[done] 4 figs (PNG+PDF)")
