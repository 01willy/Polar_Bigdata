"""
시각화 모음 — 크리깅 3D 결과/원시 데이터를 직관적 이미지로.

depth_slices()    : 깊이별 크리깅 MAGT '지도'(3D 보간 결과를 가장 직관적으로)
uncertainty_map() : 크리깅 표준편차(어디를 믿을 수 있나)
active_layer()    : ALT(활성층) — CALM 사이트 지도 + 격자 구조 + 분포
profiles_gallery(): borehole 수직 온도 프로파일 갤러리
"""
import json
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from . import config as C
from . import geo

for _fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False

CLIM = (-10.0, 2.0)
CMAP = "RdBu_r"


def _load_grid():
    d = np.load(C.PROCESSED / "volume_grid.npz")
    gx, gy, levels, vol = d["gx"], d["gy"], d["levels"], d["vol"]
    var = d["var"] if "var" in d else None
    GX, GY = np.meshgrid(gx, gy)                  # (NY,NX) in EPSG:3413
    LON, LAT = geo.to_lonlat(GX, GY)
    return LON, LAT, levels, vol, var


def depth_slices():
    LON, LAT, levels, vol, _ = _load_grid()
    summ = pd.read_csv(C.SUMMARY_CSV)
    targets = [1, 5, 10, 20, 40, 70]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
    for ax, td in zip(axes.ravel(), targets):
        li = int(np.argmin(np.abs(levels - td)))
        sl = vol[li]
        m = ax.pcolormesh(LON, LAT, np.clip(sl, *CLIM), cmap=CMAP,
                          vmin=CLIM[0], vmax=CLIM[1], shading="auto")
        if np.isfinite(sl).sum() > 20:
            try:
                cs = ax.contour(LON, LAT, sl, levels=[0.0], colors="k", linewidths=1.6)
                ax.clabel(cs, fmt="0°C", fontsize=8)
            except Exception:
                pass
        # 이 깊이에 데이터가 닿는 borehole만 강조
        deep_enough = summ[summ["max_depth"] >= levels[li]]
        ax.scatter(deep_enough["lon"], deep_enough["lat"], s=22, c="k",
                   marker="^", edgecolor="w", linewidth=0.4, zorder=3)
        ax.set_title(f"깊이 {levels[li]:.0f} m  (borehole {len(deep_enough)}개 도달)",
                     fontsize=13)
        ax.set_xlabel("경도"); ax.set_ylabel("위도")
        ax.set_facecolor("#f3f3f3")
    fig.suptitle("Alaska 영구동토 3D 크리깅 — 깊이별 평균 지중온도 지도",
                 fontsize=18, fontweight="bold", y=0.99)
    cb = fig.colorbar(m, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("MAGT (°C)   ←차가움(영구동토)  ·  따뜻함→", fontsize=12)
    out = C.FIGURES / "02_depth_slices.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def uncertainty_map():
    LON, LAT, levels, vol, var = _load_grid()
    if var is None:
        print("  (분산 없음)"); return
    std = np.sqrt(np.clip(var, 0, None))
    summ = pd.read_csv(C.SUMMARY_CSV)
    targets = [5, 20, 50]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    vmax = np.nanpercentile(std, 92)
    for ax, td in zip(axes, targets):
        li = int(np.argmin(np.abs(levels - td)))
        m = ax.pcolormesh(LON, LAT, std[li], cmap="magma_r", vmin=0, vmax=vmax, shading="auto")
        ax.scatter(summ["lon"], summ["lat"], s=14, c="cyan", edgecolor="k", linewidth=0.3, zorder=3)
        ax.set_title(f"깊이 {levels[li]:.0f} m", fontsize=13)
        ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.set_facecolor("#111")
    fig.suptitle("크리깅 불확실성(표준편차) — borehole에서 멀수록 큼 (밝을수록 불확실)",
                 fontsize=16, fontweight="bold")
    cb = fig.colorbar(m, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("MAGT 표준편차 (°C)", fontsize=12)
    out = C.FIGURES / "05_kriging_uncertainty.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _alt_site_coords():
    sites = json.load(open(C.SITES_JSON))
    idx = {}
    for s in sites:
        loc = s.get("location") or {}
        for a in s.get("activelayers", []):
            idx[a["id"]] = (a.get("latitude", loc.get("latitude_avg")),
                            a.get("longitude", loc.get("longitude_avg")), a.get("name", s.get("name")))
    return idx


def active_layer():
    files = sorted(glob.glob(str(C.ALT_CSV_DIR / "alt_dataset_*.csv")))
    if not files:
        print("  (ALT 없음)"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["alt"])
    idx = _alt_site_coords()
    per = df.groupby("activelayer_id")["alt"].agg(["mean", "count"]).reset_index()
    per["lat"] = per["activelayer_id"].map(lambda i: (idx.get(i) or (None,))[0])
    per["lon"] = per["activelayer_id"].map(lambda i: (idx.get(i) or (None, None))[1])
    per = per.dropna(subset=["lat", "lon"])

    fig = plt.figure(figsize=(17, 6))
    # (a) 지도
    ax = fig.add_subplot(1, 3, 1)
    sc = ax.scatter(per["lon"], per["lat"], c=per["mean"], s=40 + per["count"] / 20,
                    cmap="YlGnBu_r", edgecolor="k", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="평균 활성층 두께 (cm)")
    ax.set_title(f"(a) Alaska 활성층(CALM) 사이트 {len(per)}개", fontsize=13)
    ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.grid(alpha=0.3)
    # (b) 한 사이트 격자 구조
    ax = fig.add_subplot(1, 3, 2)
    big = df.groupby("dataset_id").size().idxmax()
    one = df[df["dataset_id"] == big]
    latest = one[one["date"] == one["date"].max()]
    sc = ax.scatter(latest["offset_x"], latest["offset_y"], c=latest["alt"],
                    cmap="YlGnBu_r", s=120, marker="s", edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="활성층 두께 (cm)")
    ax.set_title(f"(b) CALM 격자 구조 1개 사이트\n(100 m 간격 측정, {latest['date'].iloc[0]})", fontsize=12)
    ax.set_xlabel("offset_x (m)"); ax.set_ylabel("offset_y (m)"); ax.set_aspect("equal")
    # (c) 위도별 분포
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(per["lat"], per["mean"], s=40, c=per["mean"], cmap="YlGnBu_r", edgecolor="k", linewidth=0.3)
    ax.set_title("(c) 위도별 평균 활성층 두께", fontsize=13)
    ax.set_xlabel("위도 (°N)"); ax.set_ylabel("평균 ALT (cm)"); ax.grid(alpha=0.3)
    ax.invert_xaxis()
    fig.suptitle("활성층 두께 (Active Layer Thickness) — 여름철 해빙 깊이", fontsize=17, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURES / "03_active_layer.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def profiles_gallery():
    prof = pd.read_csv(C.PROFILES_CSV).dropna(subset=["magt", "depth"])
    summ = pd.read_csv(C.SUMMARY_CSV).sort_values("lat", ascending=False)
    sel = summ.nlargest(12, "max_depth").sort_values("lat", ascending=False)["borehole_id"].tolist()
    fig, axes = plt.subplots(2, 6, figsize=(19, 8), sharey=True)
    for ax, bid in zip(axes.ravel(), sel):
        g = prof[prof["borehole_id"] == bid].sort_values("depth")
        s = summ[summ["borehole_id"] == bid].iloc[0]
        ax.scatter(g["magt"], g["depth"], c=np.clip(g["magt"], *CLIM), cmap=CMAP,
                   vmin=CLIM[0], vmax=CLIM[1], s=16, edgecolor="k", linewidth=0.2, zorder=3)
        ax.plot(g["magt"], g["depth"], "-", color="#888", lw=0.8, zorder=2)
        ax.axvline(0, color="b", ls="--", lw=1)
        ax.set_title(f"{s['site']}\n{s['lat']:.1f}°N, {s['max_depth']:.0f} m", fontsize=9.5)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("깊이 (m)")
    for ax in axes[1, :]:
        ax.set_xlabel("MAGT (°C)")
    axes[0, 0].invert_yaxis()
    fig.suptitle("borehole 수직 온도 프로파일 (위도 높은 순) — 점선=0 °C", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURES / "04_profiles_gallery.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def stage1_chart():
    """Stage1 — 도일 피처 추가 효과 + 변수 중요도."""
    res = pd.read_csv(C.PROCESSED / "stage1_results.csv")
    imp = pd.read_csv(C.PROCESSED / "stage1_feature_importance.csv", index_col=0).iloc[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cvs = ["공간블록", "LORO(전이)"]
    configs = ["GBM 기본(공변량5)", "GBM +도일피처", "RF +도일피처"]
    colors = ["#9bb", "#2a9d4a", "#c0697a"]
    ax = axes[0]; x = np.arange(len(cvs)); w = 0.25
    for i, cfg in enumerate(configs):
        vals = [res[(res.cv == cv) & (res.config == cfg)]["rmse"].values[0] for cv in cvs]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=cfg, color=colors[i], edgecolor="k", linewidth=0.4)
        ax.bar_label(bars, fmt="%.0f", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cvs, fontsize=12)
    ax.set_ylabel("ALT RMSE (cm)  ↓낮을수록 정확", fontsize=12)
    ax.set_title("(a) 도일 피처 추가 효과 (정직한 CV)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.annotate("전이(LORO)에선 GBM이\nRF보다 안정(과적합 적음)", xy=(1.0, 97), xytext=(0.3, 120),
                fontsize=10, color="#b22", arrowprops=dict(arrowstyle="->", color="#b22"))

    ax = axes[1]
    imp = imp.sort_values()
    ax.barh(range(len(imp)), imp.values, color="#3a7abd", edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=9)
    ax.set_title("(b) 변수 중요도 (RF) — 고도가 지배(지역 대리값 우려)", fontsize=13)
    ax.set_xlabel("중요도"); ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=8)
    fig.suptitle("Stage 1 — 물리 피처(도일·Stefan) 추가 + 변수 중요도", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURES / "12_stage1_features.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def cv_leakage_chart():
    """CV 설계별 baseline RMSE — 무작위 CV의 과대평가(누설)를 한눈에."""
    res = pd.read_csv(C.PROCESSED / "cv_leakage_table.csv")
    order_cv = [c for c in ["무작위 K-fold", "site-disjoint"] if c in res.cv.unique()]
    order_cv += [c for c in res.cv.unique() if c.startswith("공간블록")]
    order_cv += [c for c in ["LORO(지역전이)"] if c in res.cv.unique()]
    methods = ["지역평균", "IDW", "Kriging", "GBM(공변량)"]
    colors = {"지역평균": "#bbb", "IDW": "#e8a33d", "Kriging": "#7aa6c2", "GBM(공변량)": "#2a9d4a"}

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(order_cv)); w = 0.2
    for i, m in enumerate(methods):
        vals = [res[(res.cv == cv) & (res.method == m)]["rmse"].values[0] for cv in order_cv]
        vals = [min(v, 250) for v in vals]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[m], edgecolor="k", linewidth=0.4)
        ax.bar_label(bars, fmt="%.0f", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(order_cv, fontsize=11)
    ax.set_ylabel("ALT 예측 오차 RMSE (cm)  ↓낮을수록 정확", fontsize=12)
    ax.set_title("CV 설계별 baseline 정확도 — 왼쪽일수록 '낙관적(누설)', 오른쪽이 '정직(전이)'",
                 fontsize=14, fontweight="bold")
    ax.legend(title="방법", fontsize=10, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("무작위 CV는 공간 자기상관으로\n성능을 크게 과대평가(IDW 28→111cm)",
                xy=(0.15, 30), xytext=(0.6, 175), fontsize=10.5, color="#b22",
                arrowprops=dict(arrowstyle="->", color="#b22"))
    ax.text(len(order_cv) - 1, 5, "전이 난도 ↑", ha="center", fontsize=10, color="#444")
    fig.tight_layout()
    out = C.FIGURES / "11_cv_leakage.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def global_alt_coverage():
    """전 지구 CALM ALT 데이터 커버리지 — 다지역 학습데이터 현황."""
    df = pd.read_csv(C.PROCESSED / "alt_global.csv")
    per = df.groupby(["site", "lat", "lon", "country"]).agg(
        alt_mean=("alt_cm", "mean"), n=("year", "nunique")).reset_index()

    fig = plt.figure(figsize=(18, 7))
    # (a) NH 지도
    ax = fig.add_subplot(1, 3, (1, 2))
    sc = ax.scatter(per["lon"], per["lat"], c=np.clip(per["alt_mean"], 20, 200),
                    s=15 + per["n"] * 2.2, cmap="YlGnBu_r", alpha=0.85,
                    edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, label="평균 활성층 두께 ALT (cm)")
    ax.axhline(66.56, color="gray", ls=":", lw=1)
    ax.text(-178, 67.2, "북극권(66.5°N)", fontsize=9, color="gray")
    ax.set(title=f"(a) 전 지구 CALM 활성층 관측망 — {len(per)}개 사이트 / {df.country.nunique()}개국",
           xlabel="경도", ylabel="위도", xlim=(-180, 180), ylim=(28, 82))
    ax.grid(alpha=0.25); ax.set_facecolor("#eef2f5")
    # (b) 국가별
    ax = fig.add_subplot(1, 3, 3)
    sc2 = per.groupby("country")["site"].nunique().sort_values()
    ax.barh(range(len(sc2)), sc2.values, color="#3a7abd")
    ax.set_yticks(range(len(sc2))); ax.set_yticklabels(sc2.index, fontsize=8)
    ax.set(title="(b) 국가별 사이트 수", xlabel="사이트 수")
    ax.bar_label(ax.containers[0], fontsize=8)
    fig.suptitle("전 지구 활성층(ALT) 학습 데이터 — 다지역 (모델 일반화·전이 검증용)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURES / "10_global_alt_coverage.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def research_concept():
    """연구 목표 개념도 — 드문 well → 3D 영구동토(활성층 윗면 + base) 예측."""
    import matplotlib.patches as mp
    x = np.linspace(0, 10, 400)
    surf = 0.0 * x
    alt_base = 0.4 + 0.07 * x                       # 활성층 두께(남쪽=오른쪽이 두꺼움)
    perm_base = np.clip(9.2 - 0.85 * x, alt_base, 10)  # 영구동토 base(북=왼쪽이 깊음)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.fill_between(x, surf, alt_base, color="#e3c79a", zorder=1)            # 활성층
    ax.fill_between(x, alt_base, perm_base, color="#bfe0f2", zorder=1)       # 영구동토
    ax.fill_between(x, perm_base, 10, color="#efe7d6", zorder=1)            # 비동결
    ax.plot(x, alt_base, color="#7a5a1e", lw=2.2)
    ax.plot(x, perm_base, color="#1f6699", lw=2.4, ls="--")

    # 드문드문한 well (수직선 + 온도 점)
    rng_depths = np.array([0.5, 1.5, 3, 4.5, 6])
    for wx in [1.8, 4.5, 7.5]:
        wb = float(np.interp(wx, x, perm_base))
        dd = rng_depths[rng_depths < min(wb, 5)]
        cval = np.linspace(-8 + wx, 0.5, len(dd))   # 대략적 온도(왼쪽 차가움)
        ax.plot([wx, wx], [0, dd.max()], color="#333", lw=3, zorder=3)
        ax.scatter([wx] * len(dd), dd, c=np.clip(cval, *CLIM), cmap=CMAP,
                   vmin=CLIM[0], vmax=CLIM[1], s=90, edgecolor="k", linewidth=0.6, zorder=4)
        ax.text(wx, -0.35, "well", ha="center", fontsize=10, color="#333")

    ax.annotate("활성층(여름에 녹는 층) 밑면\n= 영구동토 table  ← 활성층 데이터(CALM)가 제약",
                xy=(6.5, float(np.interp(6.5, x, alt_base))), xytext=(6.6, 2.0),
                fontsize=11, color="#7a5a1e",
                arrowprops=dict(arrowstyle="->", color="#7a5a1e"))
    ax.annotate("영구동토 base (0 °C)\n← 대부분 well이 못 닿음 → 물리로 외삽",
                xy=(2.5, float(np.interp(2.5, x, perm_base))), xytext=(3.2, 8.3),
                fontsize=11, color="#1f6699",
                arrowprops=dict(arrowstyle="->", color="#1f6699"))
    ax.text(0.15, 0.3, "북(추움)", fontsize=11, color="#1f6699", fontweight="bold")
    ax.text(8.8, 0.3, "남(따뜻)", fontsize=11, color="#c0392b", fontweight="bold")
    ax.text(5, 5.2, "영구동토 (연중 ≤ 0 °C)", fontsize=12, color="#1f6699", ha="center")

    # 화살표: 입력 → 모델 → 출력
    ax.text(5, -1.15, "입력: 드문드문한 well 온도(점) + 활성층 측정 + 격자 공변량(기온·적설·고도)",
            ha="center", fontsize=11.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff7e6", ec="#caa"))
    ax.text(5, 10.9, "딥러닝(보간+물리 외삽) → 출력: 3D 영구동토 = 활성층 윗면 + base + 내부 온도장 + 불확실성",
            ha="center", fontsize=11.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f4ec", ec="#9c9"))

    ax.set_xlim(0, 10); ax.set_ylim(-1.6, 11.4); ax.invert_yaxis()
    ax.set_xlabel("공간(거리) →", fontsize=12); ax.set_ylabel("깊이 (m, 모식)", fontsize=12)
    ax.set_title("연구 목표 한눈에 — well 온도로 3D 영구동토(활성층~base)를 예측",
                 fontsize=15, fontweight="bold")
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    out = C.FIGURES / "09_research_concept.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def covariate_maps():
    """WorldClim 격자 공변량을 Alaska 지도로 — 무엇으로 예측하나."""
    from . import covariates as cov
    d = np.load(C.PROCESSED / "volume_grid.npz")
    GX, GY = np.meshgrid(d["gx"], d["gy"])
    LON, LAT = geo.to_lonlat(GX, GY)
    gwc = cov.sample_worldclim(LON.ravel(), LAT.ravel())
    summ = pd.read_csv(C.SUMMARY_CSV)
    panels = [("bio1", "연평균기온 (°C)", "RdBu_r"), ("elev", "고도 (m)", "terrain"),
              ("bio4", "기온 계절성", "viridis"), ("bio12", "연강수량 (mm)", "YlGnBu")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))
    for ax, (var, label, cmap) in zip(axes.ravel(), panels):
        g = gwc[var].reshape(LON.shape)
        m = ax.pcolormesh(LON, LAT, g, cmap=cmap, shading="auto")
        ax.scatter(summ["lon"], summ["lat"], s=14, c="k", marker="^",
                   edgecolor="w", linewidth=0.4, zorder=3)
        plt.colorbar(m, ax=ax, fraction=0.04, pad=0.02, label=label)
        ax.set_title(f"WorldClim {var} — {label}", fontsize=12)
        ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.set_facecolor("#eee")
    fig.suptitle("격자 공변량(무계정 WorldClim) — 이것으로 빈 공간의 지중온도를 예측",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURES / "07_covariates.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def model_compare():
    """크리깅 vs 공변량 RF 비교 + 교차검증 RMSE."""
    dk = np.load(C.PROCESSED / "volume_grid.npz")
    dr = np.load(C.PROCESSED / "volume_grid_rf.npz")
    GX, GY = np.meshgrid(dk["gx"], dk["gy"])
    LON, LAT = geo.to_lonlat(GX, GY)
    lev = dk["levels"]; li = int(np.argmin(np.abs(lev - 10)))
    summ = pd.read_csv(C.SUMMARY_CSV)

    fig = plt.figure(figsize=(17, 5.4))
    for j, (vol, ttl) in enumerate([(dk["vol"][li], "크리깅 (거리기반)"),
                                    (dr["vol"][li], "RF (공변량 기반)")]):
        ax = fig.add_subplot(1, 3, j + 1)
        m = ax.pcolormesh(LON, LAT, np.clip(vol, *CLIM), cmap=CMAP,
                          vmin=CLIM[0], vmax=CLIM[1], shading="auto")
        ax.scatter(summ["lon"], summ["lat"], s=14, c="k", marker="^", edgecolor="w", linewidth=0.4)
        ax.set_title(f"{ttl}  —  깊이 {lev[li]:.0f} m", fontsize=12)
        ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.set_facecolor("#eee")
        plt.colorbar(m, ax=ax, fraction=0.045, pad=0.02, label="MAGT (°C)")
    # CV 막대
    ax = fig.add_subplot(1, 3, 3)
    try:
        cv = pd.read_csv(C.PROCESSED / "cv_results.csv", index_col=0)
        bars = ax.bar(cv.index, cv["rmse"], color=["#888", "#2a9d4a", "#3a7abd"])
        ax.bar_label(bars, fmt="%.2f", fontsize=11)
        ax.set_ylabel("RMSE (°C)  ↓낮을수록 좋음"); ax.set_title("교차검증(LOBO) 정확도", fontsize=12)
        ax.tick_params(axis="x", labelrotation=12)
    except FileNotFoundError:
        ax.axis("off")
    fig.suptitle("거리 보간 vs 공변량 모델 — 공변량(기후)이 더 정확 (RMSE 2.17→1.53 °C)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = C.FIGURES / "08_kriging_vs_rf.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def schematic_alt_pt():
    """개념도 — ALT(활성층)와 PT(지중온도)가 어떻게 연결되는지(trumpet curve)."""
    z = np.linspace(0, 20, 500)
    magt0, grad, A0, damp = -6.0, 0.04, 9.0, 4.3        # 대표값(개념 설명용)
    MAGT = magt0 + grad * z
    A = A0 * np.exp(-z / damp)
    Tmax, Tmin = MAGT + A, MAGT - A
    alt = z[np.where(Tmax <= 0)[0][0]]                   # 여름최대=0 깊이=영구동토 table
    dzaa = z[np.where(A < 0.1)[0][0]]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.fill_betweenx([0, alt], -20, 12, color="#e9d3b0", alpha=0.6, zorder=0)   # 활성층
    ax.fill_betweenx([alt, 20], -20, 12, color="#cfe6f5", alpha=0.6, zorder=0)  # 영구동토
    ax.plot(Tmin, z, color="#2b6cb0", lw=2, label="겨울 최저온도  T_min(z)")
    ax.plot(Tmax, z, color="#c0392b", lw=2, label="여름 최고온도  T_max(z)")
    ax.plot(MAGT, z, color="k", lw=2.4, ls="--", label="연평균 MAGT(z)  ← PT가 주는 값")
    ax.axvline(0, color="#333", lw=1)
    ax.axhline(alt, color="#6b4f1d", lw=1.6, ls=":")
    ax.axhline(dzaa, color="#555", lw=1, ls=":")

    ax.annotate(f"영구동토 table = 활성층 두께(ALT)\n= {alt:.2f} m\n(여기서 여름최대온도 = 0 °C)",
                xy=(0, alt), xytext=(3.5, alt + 2.2), fontsize=11, color="#6b4f1d",
                arrowprops=dict(arrowstyle="->", color="#6b4f1d"))
    ax.text(-18, alt / 2, "활성층\n(여름에 해빙)", fontsize=11, va="center", color="#7a5a1e")
    ax.text(-18, (alt + 20) / 2, "영구동토\n(연중 ≤ 0 °C)", fontsize=11, va="center", color="#1f6699")
    ax.text(6.5, dzaa + 0.4, f"DZAA ≈ {dzaa:.0f} m\n(계절변동 소멸)", fontsize=9.5, color="#555")
    ax.annotate("ALT(CALM)이 주는 값\n= 영구동토 상부 경계 깊이", xy=(0, alt), xytext=(-12, 17),
                fontsize=11, color="#2b6cb0",
                arrowprops=dict(arrowstyle="->", color="#2b6cb0"))

    ax.set_xlim(-20, 12); ax.set_ylim(0, 20); ax.invert_yaxis()
    ax.set_xlabel("온도 (°C)", fontsize=12); ax.set_ylabel("깊이 (m)", fontsize=12)
    ax.set_title("ALT(활성층)와 PT(지중온도)의 연결 — 개념도\n"
                 "PT=깊이별 온도 / ALT=영구동토 table 깊이 / 둘은 0 °C 경계에서 만남",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.005, "※ 대표값으로 그린 개념도. 실제 모델에선 ALT가 frozen body의 윗면(상부 경계조건),"
             " PT가 내부 온도장을 담당.", ha="center", fontsize=9, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = C.FIGURES / "06_concept_alt_pt.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def run():
    C.ensure_dirs()
    print("Rendering concept ALT-PT ...");     schematic_alt_pt()
    print("Rendering depth slices ...");       depth_slices()
    print("Rendering uncertainty ...");        uncertainty_map()
    print("Rendering active layer ...");       active_layer()
    print("Rendering profiles gallery ...");   profiles_gallery()
    try:
        print("Rendering covariates ...");     covariate_maps()
        print("Rendering kriging vs RF ...");  model_compare()
    except FileNotFoundError:
        print("  (공변량/RF 결과 없음 — scripts/04,05 먼저 실행)")
    print("Done -> outputs/figures/")
