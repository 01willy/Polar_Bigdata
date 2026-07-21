"""W2.1 SoilGrids 게이트 시각화 — 토양 지도·피처셋 skill·토양vsALT·지역별 유효율.

산출:
  outputs/figures/09_soilgrids/soil_covariate_maps.{png,pdf}   토양 공변량 지도(알래스카·레나)
  outputs/figures/09_soilgrids/soil_gate_skill.{png,pdf}       피처셋별 skill(공간블록 vs LORO)
  outputs/figures/09_soilgrids/soil_vs_alt.{png,pdf}           토양 vs ALT 산점·상관
  outputs/figures/09_soilgrids/soil_valid_by_region.{png,pdf}  지역별 sg_ 유효율

시각 규약: 냉색 순차(토양=밀도맵 계열), diff 발산은 skill Δ, 축 단위 필수, PNG+PDF.
"""
import os, sys, glob
sys.path.insert(0, os.path.join("/home/willy010313/Polar_Bigdata", "src"))
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as tr_from_bounds
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo, BAD
from polar.outputs import figpath, mappath
plt = use_polar()

ROOT = "/home/willy010313/Polar_Bigdata"
PROC = os.path.join(ROOT, "data/processed")
RAW = os.path.join(ROOT, "data/raw/soilgrids_multi")
FIGDIR = "09_soilgrids"


def save(fig, name):
    fig.savefig(figpath(FIGDIR, name, "png"))
    fig.savefig(figpath(FIGDIR, name, "pdf"))
    plt.close(fig)
    print("저장:", figpath(FIGDIR, name, "png"))


def mosaic_layer(layer_base, lo0, lo1, la0, la1, scale, npix=900):
    """지역 bbox 내 모든 bin tif 를 WGS84 격자로 재투영 병합(최근접). 물리단위 반환."""
    H = W = npix
    dst_tr = tr_from_bounds(lo0, la0, lo1, la1, W, H)
    acc = np.full((H, W), np.nan, np.float32)
    tifs = sorted(glob.glob(os.path.join(RAW, "*", f"{layer_base}.tif")))
    for tif in tifs:
        with rasterio.open(tif) as src:
            tmp = np.full((H, W), np.nan, np.float32)
            reproject(source=rasterio.band(src, 1), destination=tmp,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_tr, dst_crs="EPSG:4326",
                      resampling=Resampling.nearest,
                      src_nodata=src.nodata, dst_nodata=np.nan)
            m = np.isnan(acc) & np.isfinite(tmp)
            acc[m] = tmp[m]
    acc = acc / scale
    return acc, (lo0, lo1, la0, la1)


# ---------- Fig 1: 토양 공변량 지도(알래스카·레나) ----------
def fig_soil_maps():
    regions = [("알래스카", -167, -141, 60.5, 71.5), ("레나델타", 122, 131, 71, 74)]
    layers = [("clay_5-15cm_mean", "점토 5-15cm (%)", 10.0),
              ("soc_5-15cm_mean", "토양 유기탄소 5-15cm (g/kg)", 10.0)]
    fig, axes = plt.subplots(len(layers), len(regions), figsize=(12.5, 9.2))
    for i, (lb, label, sc) in enumerate(layers):
        # 지역 공통 색범위(두 지역 동일 스케일로 비교 가능)
        arrs = []
        for _, lo0, lo1, la0, la1 in regions:
            a, _ = mosaic_layer(lb, lo0, lo1, la0, la1, sc)
            arrs.append(a)
        allv = np.concatenate([a[np.isfinite(a)] for a in arrs])
        vmin, vmax = np.percentile(allv, [3, 97]) if allv.size else (0, 1)
        for j, (rname, lo0, lo1, la0, la1) in enumerate(regions):
            ax = axes[i, j]
            a = arrs[j]
            im = ax.imshow(a, extent=[lo0, lo1, la0, la1], origin="upper",
                           cmap=CMAP.count, vmin=vmin, vmax=vmax, aspect="auto")
            style_geo(ax, title=f"{rname} · {label}")
            add_cbar(fig, im, ax, label)
    fig.suptitle("SoilGrids 250m 토양 공변량 — 알래스카·레나델타", fontsize=13.5, y=1.005)
    fig.tight_layout()
    save(fig, "soil_covariate_maps")


# ---------- Fig 2: 피처셋별 skill(공간블록 vs LORO) ----------
def fig_gate_skill():
    res = pd.read_csv(os.path.join(PROC, "soil_ablation_gate.csv"))
    order = ["M_clim", "M_climterr", "M_soil", "M_soilonly"]
    labels = {"M_clim": "기후8", "M_climterr": "기후+지형\n(baseline)",
              "M_soil": "+토양\n(sg_*)", "M_soilonly": "기후+토양\n(지형 대신)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, cv, title in [(a1, "spatial_block", "공간블록 6-fold (알래스카 위주)"),
                          (a2, "LORO", "LORO (지역 간 전이)")]:
        sub = res[res.cv_type == cv].set_index("config")
        sk = [sub.loc[c, "skill_over_mean"] * 100 for c in order]
        rm = [sub.loc[c, "rmse_cm"] for c in order]
        x = np.arange(len(order))
        base = sub.loc["M_climterr", "skill_over_mean"] * 100
        cols = [CMAP.count(0.55) if c != "M_soil" else CMAP.diff(0.78) for c in order]
        bars = ax.bar(x, sk, color=cols, edgecolor="#333", linewidth=0.6, width=0.66)
        ax.axhline(base, color="#888", ls="--", lw=0.9, zorder=0)
        for xi, (s, r) in enumerate(zip(sk, rm)):
            ax.text(xi, s + 0.4, f"{s:.1f}%\n{r:.1f}cm", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels([labels[c] for c in order], fontsize=9)
        ax.set_ylabel("skill over mean (%)  =  1 − RMSE/SD")
        ax.set_title(title)
        ax.set_ylim(0, max(sk) * 1.18)
        ax.margins(x=0.04)
    fig.suptitle("피처셋별 예측력 — 토양 공변량 추가 효과(누설통제)", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, "soil_gate_skill")


# ---------- Fig 3: 토양 vs ALT 산점·상관 ----------
def fig_soil_vs_alt():
    df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell_v3_soil.csv"), low_memory=False)
    sg = sorted([c for c in df.columns if c.startswith("sg_")])
    # 상관(스피어만) 막대 + 대표 2개 산점
    from scipy.stats import spearmanr
    cors = []
    for c in sg:
        m = np.isfinite(df[c]) & np.isfinite(df["alt_cm"])
        if m.sum() > 100:
            rho, _ = spearmanr(df.loc[m, c], df.loc[m, "alt_cm"])
            cors.append((c, rho, int(m.sum())))
    cors.sort(key=lambda t: abs(t[1]), reverse=True)
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    names = [c.replace("sg_", "") for c, _, _ in cors]
    rhos = [r for _, r, _ in cors]
    y = np.arange(len(names))[::-1]
    cols = [CMAP.diff(0.85) if r > 0 else CMAP.diff(0.15) for r in rhos]
    ax0.barh(y, rhos, color=cols, edgecolor="#333", linewidth=0.5)
    ax0.axvline(0, color="#444", lw=0.8)
    ax0.set_yticks(y); ax0.set_yticklabels(names, fontsize=8.5)
    ax0.set_xlabel("Spearman ρ (토양 vs ALT)")
    ax0.set_title("토양 공변량 · ALT 상관")
    # 두 대표 산점
    for k, ax in enumerate([fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]):
        if k >= len(cors):
            break
        c = cors[k][0]
        m = np.isfinite(df[c]) & np.isfinite(df["alt_cm"])
        ax.scatter(df.loc[m, c], df.loc[m, "alt_cm"], s=5, alpha=0.18,
                   color=CMAP.count(0.6), edgecolors="none")
        ax.set_xlabel(c.replace("sg_", "") + " (물리단위)")
        ax.set_ylabel("ALT (cm)")
        ax.set_title(f"{c.replace('sg_','')}  ρ={cors[k][1]:.2f}")
        ax.set_ylim(0, np.percentile(df.loc[m, "alt_cm"], 99))
    fig.suptitle("토양 공변량과 활성층 두께(ALT)의 물리 상관", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, "soil_vs_alt")


# ---------- Fig 4: 지역별 sg_ 유효율 ----------
def fig_valid_by_region():
    v = pd.read_csv(os.path.join(PROC, "soilgrids_valid_by_region.csv"))
    v = v.sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(v))
    bars = ax.bar(x, v["sg_valid_avg_pct"], color=CMAP.count(0.6),
                  edgecolor="#333", linewidth=0.6)
    for xi, (val, n) in enumerate(zip(v["sg_valid_avg_pct"], v["n"])):
        ax.text(xi, val + 0.8, f"{val:.0f}%\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(v["region"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("sg_ 평균 유효율 (%)")
    ax.set_ylim(0, 108)
    ax.axhline(100, color="#888", ls=":", lw=0.8)
    ax.set_title("지역별 SoilGrids 공변량 유효율 — 전지구 커버라 결측 적음")
    fig.tight_layout()
    save(fig, "soil_valid_by_region")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    todo = {"maps": fig_soil_maps, "skill": fig_gate_skill,
            "scatter": fig_soil_vs_alt, "valid": fig_valid_by_region}
    if args.only == "all":
        for fn in todo.values():
            try:
                fn()
            except Exception as e:
                print("FIG ERR", fn.__name__, str(e)[:160])
    else:
        todo[args.only]()
