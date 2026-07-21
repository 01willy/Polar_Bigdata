"""KPDC 현장 검증·물리 forcing 사례연구 시각화.

산출(PNG+PDF, 냉색·단위 표기):
  outputs/figures/10_kpdc/kpdc_era5_scatter        지점 실측 vs ERA5(MAAT·√TDD) 산점, 1:1선·bias
  outputs/figures/10_kpdc/kpdc_era5_bias_bar       ERA5 편의(지점 대비) 막대 요약
  outputs/figures/10_kpdc/kpdc_forcing_scatter     forcing별 Stefan ALT vs 관측 ALT 산점
  outputs/maps/kpdc_council_location               콘슬 위치(알래스카 내)·ALT 셀·관측소 마커

실행: /home/anaconda3/bin/python scripts/4_visualization/kpdc_figs.py
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = "/home/willy010313/Polar_Bigdata"
sys.path.insert(0, os.path.join(ROOT, "src"))
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo, BAD
from polar.outputs import figpath, mappath

plt = use_polar()
PROC = os.path.join(ROOT, "data/processed")

val = pd.read_csv(os.path.join(PROC, "kpdc_era5_validation.csv"))
forcing = pd.read_csv(os.path.join(PROC, "kpdc_council_forcing.csv"))
cell = pd.read_csv(os.path.join(PROC, "dl_dataset_cell.csv"))

COLD = "#1f5f8b"       # 냉색 강조(청)
COLD2 = "#3d8fb0"
ACCENT = "#5b3a86"     # 보조(자주, 붉은계열 회피)
MK = dict(Council="o", c1="s")


def _save(fig, path_png):
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------- 그림 1: 지점 실측 vs ERA5 산점(MAAT·√TDD) ----------
def fig_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    # MAAT
    ax = axes[0]
    sub = val.dropna(subset=["st_maat", "e5_maat_masked"])
    lo = min(sub.st_maat.min(), sub.e5_maat_masked.min()) - 1.5
    hi = max(sub.st_maat.max(), sub.e5_maat_masked.max()) + 1.5
    ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1, zorder=1, label="1:1")
    for site in sub.site.unique():
        s = sub[sub.site == site]
        ax.scatter(s.st_maat, s.e5_maat_masked, s=70, marker=MK.get(site, "o"),
                   color=COLD, edgecolor="white", lw=0.8, zorder=3, label=site)
        for _, r in s.iterrows():
            ax.annotate(f"{int(r.year)}", (r.st_maat, r.e5_maat_masked),
                        fontsize=7, color="#555", xytext=(4, 3),
                        textcoords="offset points")
    bias = (sub.e5_maat_masked - sub.st_maat).mean()
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("지점 실측 MAAT (°C)")
    ax.set_ylabel("ERA5-Land MAAT (°C)")
    ax.set_title(f"연평균 기온 (bias {bias:+.2f}°C)")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, loc="upper left", frameon=True)

    # √TDD
    ax = axes[1]
    sub = val.dropna(subset=["st_sqrt_tdd_mon", "e5_sqrt_tdd_masked"])
    lo = min(sub.st_sqrt_tdd_mon.min(), sub.e5_sqrt_tdd_masked.min()) - 2
    hi = max(sub.st_sqrt_tdd_mon.max(), sub.e5_sqrt_tdd_masked.max()) + 2
    ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1, zorder=1, label="1:1")
    for site in sub.site.unique():
        s = sub[sub.site == site]
        ax.scatter(s.st_sqrt_tdd_mon, s.e5_sqrt_tdd_masked, s=70,
                   marker=MK.get(site, "o"), color=COLD2, edgecolor="white",
                   lw=0.8, zorder=3, label=site)
        for _, r in s.iterrows():
            ax.annotate(f"{int(r.year)}", (r.st_sqrt_tdd_mon, r.e5_sqrt_tdd_masked),
                        fontsize=7, color="#555", xytext=(4, 3),
                        textcoords="offset points")
    bias = (sub.e5_sqrt_tdd_masked - sub.st_sqrt_tdd_mon).mean()
    corr = np.corrcoef(sub.st_sqrt_tdd_mon, sub.e5_sqrt_tdd_masked)[0, 1]
    # 이상점(Council 2019, 성장기 부분값) 제외 시 상관 급변 → 소표본 취약성 병기
    m3 = sub[~((sub.site == "Council") & (sub.year == 2019))]
    corr3 = np.corrcoef(m3.st_sqrt_tdd_mon, m3.e5_sqrt_tdd_masked)[0, 1] if len(m3) >= 3 else np.nan
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("지점 실측 √TDD (√°C·d)")
    ax.set_ylabel("ERA5-Land √TDD (√°C·d)")
    ax.set_title(f"√융해도일 (bias {bias:+.2f}, r={corr:.2f}, n={len(sub)})")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, loc="upper left", frameon=True)
    # 정직 주석: r은 단일 이상점 의존. Council 2019 제외 시 r 값 병기.
    ax.text(0.97, 0.03,
            f"Council 2019 제외 시 r={corr3:.2f}\n(소표본·단일점 의존)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#555", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                    ec="#ccc", lw=0.5))

    fig.suptitle("KPDC 지점 실측 대비 ERA5-Land 공변량 정합 (관측월 정합, n=4 사례연구)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, figpath("10_kpdc", "kpdc_era5_scatter"))


# ---------- 그림 2: ERA5 편의 막대(지점 대비) ----------
def fig_bias_bar():
    sub = val.copy()
    sub["label"] = sub.site + " " + sub.year.astype(int).astype(str)
    metrics = [("bias_maat", "MAAT (°C)", COLD),
               ("bias_sqrt_tdd", "√TDD (√°C·d)", COLD2)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=False)
    for ax, (col, lab, c) in zip(axes, metrics):
        s = sub.dropna(subset=[col])
        y = np.arange(len(s))
        ax.barh(y, s[col].values, color=c, edgecolor="white", height=0.6)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(s.label.values, fontsize=8)
        ax.set_xlabel(f"ERA5 - 지점 편의: {lab}")
        ax.grid(axis="x", alpha=0.3, lw=0.4)
        # 라벨은 항상 막대 끝 바깥쪽(0에서 멀어지는 방향)에 두어 축라벨과 충돌 방지
        for yi, v in zip(y, s[col].values):
            off = 6 if v >= 0 else -6
            ax.annotate(f"{v:+.2f}", (v, yi), fontsize=7.5, color="#333",
                        va="center", xytext=(off, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right")
        # x축 여백 확보(라벨 잘림 방지)
        xmin, xmax = ax.get_xlim()
        pad = 0.18 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)
        rmse = np.sqrt(np.mean(s[col].values ** 2))
        ax.set_title(f"{lab}  (RMSE {rmse:.2f}, n={len(s)})", fontsize=10)
    fig.suptitle("ERA5-Land 공변량 편의 (지점 실측 기준, 관측월 정합)",
                 fontsize=12, y=1.02)
    # 정직 주석: c1 2018 관측월(1-10) 정합 후 MAAT 냉편의는 소멸(bias -0.3°C)
    fig.text(0.5, -0.03,
             "관측월은 실측 자료에서 유도(c1 2018=1-10월). MAAT 편의는 -0.3°C 수준으로 "
             "체계적 냉편의 없음. 소표본(n<=4) 사례연구.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    _save(fig, figpath("10_kpdc", "kpdc_era5_bias_bar"))


# ---------- 그림 3: forcing별 Stefan ALT vs 관측 ALT ----------
def fig_forcing_scatter():
    obs = forcing.alt_cm.values
    variants = [
        ("pred_stefan_era5_pt", "ERA5 지점 √TDD forcing", COLD),
        ("pred_stefan_st", "KPDC 지점 실측 √TDD forcing", ACCENT),
    ]
    lo = 0
    hi = max(obs.max(), forcing[[v[0] for v in variants]].values.max()) + 8
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharex=True, sharey=True)
    for ax, (col, title, c) in zip(axes, variants):
        pred = forcing[col].values
        ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1, label="1:1", zorder=1)
        ax.scatter(obs, pred, s=42, color=c, edgecolor="white", lw=0.6,
                   alpha=0.9, zorder=3)
        e = pred - obs
        rmse = np.sqrt(np.mean(e ** 2)); bias = e.mean()
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("관측 ALT (cm)")
        ax.set_title(f"{title}\nRMSE {rmse:.1f}cm · bias {bias:+.1f}cm", fontsize=10)
        ax.grid(alpha=0.3, lw=0.4)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Stefan 예측 ALT (cm)")
    fig.suptitle("콘슬 Stefan forcing 사례연구: forcing별 예측 vs 관측 ALT (44셀)",
                 fontsize=12, y=1.02)
    # 정직 주석: 국소 평균 하한
    cmean = obs.mean()
    fig.text(0.5, -0.04,
             f"콘슬 국소 평균 ALT {cmean:.0f}cm(RMSE 13.4cm)를 두 forcing 모두 못 이김. "
             f"단일 기후셀 내 √TDD 변이 미미(37.7-38.0) → forcing이 국소 ALT변이 미설명. "
             f"소표본·사례연구.", ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    _save(fig, figpath("10_kpdc", "kpdc_forcing_scatter"))


# ---------- 그림 4: 콘슬 위치 지도(알래스카) ----------
def fig_location_map():
    # 알래스카 ALT 셀 배경 + 콘슬 셀 강조 + 관측소 마커
    ak = cell[(cell.region.isin(["ABoVE_AK", "United States (Alaska)"]))].copy()
    ak = ak.dropna(subset=["lat", "lon", "alt_cm"])
    box = (ak.lat.between(64.83, 64.87)) & (ak.lon.between(-163.73, -163.68))
    council = ak[box]

    fig = plt.figure(figsize=(10.5, 4.8))
    # 왼쪽: 알래스카 전역 맥락
    ax0 = fig.add_subplot(1, 2, 1)
    sc = ax0.scatter(ak.lon, ak.lat, c=ak.alt_cm, cmap=CMAP.alt, s=8,
                     vmin=20, vmax=110, edgecolor="none", alpha=0.85)
    ax0.scatter([-163.70], [64.85], marker="*", s=320, color=ACCENT,
                edgecolor="white", lw=1.2, zorder=5, label="콘슬 관측소")
    ax0.set_xlim(ak.lon.min() - 1, ak.lon.max() + 1)
    ax0.set_ylim(ak.lat.min() - 0.5, ak.lat.max() + 0.5)
    add_cbar(fig, sc, ax0, "관측 ALT (cm)")
    style_geo(ax0, title="알래스카 ALT 관측 셀 · 콘슬 위치")
    ax0.legend(fontsize=8, loc="lower left", frameon=True)

    # 오른쪽: 콘슬 근접(셀별 ALT)
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(council.lon, council.lat, c=council.alt_cm, cmap=CMAP.alt,
                      s=140, vmin=council.alt_cm.min(), vmax=council.alt_cm.max(),
                      edgecolor="#333", lw=0.5, zorder=3)
    ax1.scatter([-163.70], [64.85], marker="*", s=320, color=ACCENT,
                edgecolor="white", lw=1.2, zorder=5, label="콘슬 관측소")
    add_cbar(fig, sc1, ax1, "관측 ALT (cm)")
    style_geo(ax1, title=f"콘슬 근접 ALT 셀 ({len(council)}개)")
    ax1.legend(fontsize=8, loc="lower left", frameon=True)
    ax1.ticklabel_format(useOffset=False)

    fig.suptitle("콘슬(Council) 관측소·ALT 셀 위치 (알래스카)", fontsize=12, y=1.02)
    fig.tight_layout()
    p = mappath("kpdc_council_location")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


fig_scatter()
fig_bias_bar()
fig_forcing_scatter()
fig_location_map()
print("저장 완료:")
print(" -", figpath("10_kpdc", "kpdc_era5_scatter"))
print(" -", figpath("10_kpdc", "kpdc_era5_bias_bar"))
print(" -", figpath("10_kpdc", "kpdc_forcing_scatter"))
print(" -", mappath("kpdc_council_location"))
