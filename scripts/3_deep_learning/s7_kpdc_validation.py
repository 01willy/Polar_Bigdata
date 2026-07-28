"""S7: KPDC Council 유도 ALT vs 실측·물리·학습모델 정합 검증 (CPU 전용).

입력(s7_parse_kpdc_council.py 선행):
- data/processed/s7_council_alt_derived.csv  0°C 등온선 유도 ALT(지점×시즌, 검열 포함)
- data/processed/s7_council_daily_temp.csv   QC 후 일별 온도(단면 그림용)
- kpdc/council/core_alt/AK_core_sample_2022.xlsx  2022 코어 18개(길이 72-88cm)
- data/processed/fidelity_base.csv           최근접 셀 기후(e5_sqrt_tdd)·ABoVE 인근 실측
- data/processed/alt_era5_temporal.csv       연도별 √TDD(2010-2024)
- data/processed/s2_physics_meta.json        Stefan E_global=1.571
- data/processed/s1_baseline_oof.csv         S1 학습모델 OOF 예측(최근접 셀)

비교 3종: (a) 유도 ALT vs 2022 코어, (b) vs Stefan(E_global·√TDD),
(c) vs S1 모델 OOF. Council은 알래스카 in-domain 점검증이다. 전이 근거로 쓰지 않는다.
헤드라인 = 2025 시즌·비검열·피크 커버리지(peak_days>=40) 프로파일의 중앙값
+ 프로파일 부트스트랩 95% CI. 검열 행은 하한으로만 집계(구간검열 정량표).

코어 해석 주의: AK_core_sample_2022의 "core sample 26~112 (86cm)"에서 README 규약은
길이(86)=ALT 약라벨. 다만 코어 상단(19-36cm)·하단(104-120cm)도 각각 동결면 해석의
대안 후보이므로 셋 다 기록하고 판단은 meta에 명시한다.

산출: data/processed/s7_kpdc_results.csv · s7_kpdc_meta.json,
      outputs/figures/s7_kpdc/*.png
실행: /home/anaconda3/bin/python scripts/3_deep_learning/s7_kpdc_validation.py  (ROOT)
      SMOKE=1 이면 그림 저해상도·부트스트랩 축소.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from polar.plotstyle import use_polar, CMAP, tnorm, FROZEN

# 검열·강조 냉색 팔레트(규약: 붉은계열·주황 금지, 냉색 톤 + 마커/해칭으로 범주 구분).
CENSOR = "#3b1f5e"   # 검열 하한 강조(짙은 자주, acton 계열과 정합, 냉색 톤)
from polar import geomap

SMOKE = os.environ.get("SMOKE", "0") == "1"
NBOOT = 1000 if SMOKE else 10000
RNG = np.random.default_rng(20260727)
DPI = 90 if SMOKE else 300
LAT, LON = 64.85, -163.70          # Council 대표점(kpdc/README, 파일 내 좌표 부재)
NEAR_KM = 3.0                       # ABoVE 인근 실측·모델 셀 반경
PROC = ROOT / "data/processed"
FIGD = ROOT / "outputs/figures/s7_kpdc"
FIGD.mkdir(parents=True, exist_ok=True)
LOCAL_EXT = ("콘슬 일대", -164.1, -163.3, 64.68, 65.02)


def dist_km(lat, lon):
    return np.hypot((lat - LAT) * 111.0, (lon - LON) * 111.0 * np.cos(np.deg2rad(LAT)))


def boot_ci(vals, stat=np.median, n=NBOOT):
    """소표본 부트스트랩 95% CI(프로파일/코어 단위 재표집)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    s = np.array([stat(RNG.choice(v, len(v), replace=True)) for _ in range(n)])
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


# ---------- 1) 입력 적재 ----------
alt = pd.read_csv(PROC / "s7_council_alt_derived.csv")
daily = pd.read_csv(PROC / "s7_council_daily_temp.csv", parse_dates=["date"])
parse_meta = json.load(open(PROC / "s7_parse_meta.json"))
E_GLOBAL = json.load(open(PROC / "s2_physics_meta.json"))["E_global"]

# 코어 2022
raw = pd.read_excel(ROOT / "kpdc/council/core_alt/AK_core_sample_2022.xlsx", header=None)
core_rows, site = [], None
for _, r in raw.iloc[3:].iterrows():
    if isinstance(r[0], str) and r[0].strip():
        site = r[0].strip().replace(" ", "")
    m = re.search(r"(\d+)\s*~\s*(\d+)\s*\((\d+)\s*cm\)", str(r[5]))
    if site and m:
        core_rows.append((f"{site}_{str(r[1]).strip()}", int(m.group(1)),
                          int(m.group(2)), int(m.group(3))))
cores = pd.DataFrame(core_rows, columns=["core_id", "core_top", "core_bot", "core_len"])
assert len(cores) == 18, "2022 코어는 18개여야 한다"

# 최근접 기후 셀(Stefan) + ABoVE 인근 실측
fb = pd.read_csv(PROC / "fidelity_base.csv",
                 usecols=["loc_id", "lat", "lon", "region", "source_id", "alt_cm", "e5_sqrt_tdd"])
fb["dist_km"] = dist_km(fb.lat, fb.lon)
cell = fb.nsmallest(1, "dist_km").iloc[0]
near_obs = fb[(fb.dist_km <= NEAR_KM) & (fb.source_id == "F4_direct")]
stefan_clim = float(E_GLOBAL * cell.e5_sqrt_tdd)

# 연도별 Stefan(ERA5 연 √TDD, 2010-2024)
tmp = pd.read_csv(PROC / "alt_era5_temporal.csv", usecols=["lat", "lon", "year", "e5t_sqrt_tdd"])
tmp = tmp[dist_km(tmp.lat, tmp.lon) < 2.0]
sqtdd_by_year = tmp.groupby("year")["e5t_sqrt_tdd"].mean()
stefan_year = {int(y): float(E_GLOBAL * v) for y, v in sqtdd_by_year.items()}

# S1 학습모델 OOF(인근 셀 평균)
oof = pd.read_csv(PROC / "s1_baseline_oof.csv")
oof["dist_km"] = dist_km(oof.lat, oof.lon)
near_oof = oof[oof.dist_km <= NEAR_KM]
pred_cols = [c for c in oof.columns if c.startswith("pred_")]
s1_mean = {c.replace("pred_", ""): float(near_oof[c].mean()) for c in pred_cols}
s1_ens = float(near_oof[pred_cols].mean(axis=1).mean())

# ---------- 2) 헤드라인·검열 정량표 ----------
head = alt[(alt.year == 2025) & (alt.right_censored == 0) & (alt.peak_days >= 40)]
head_vals = head.alt_cm.values
head_med = float(np.median(head_vals))
head_ci = boot_ci(head_vals)
cens_tab = (alt.groupby(["deployment", "depth_max_cm", "right_censored"])
            .size().rename("n").reset_index())

core_mean = float(cores.core_len.mean())
core_ci = boot_ci(cores.core_len, stat=np.mean)
near_mean = float(near_obs.alt_cm.mean())
near_ci = boot_ci(near_obs.alt_cm, stat=np.mean)

# ---------- 3) 통합 결과 CSV ----------
res = alt.copy()
res.insert(0, "method", "isotherm_0C")
core_res = pd.DataFrame(dict(
    method="core_2022", site_key=cores.core_id, profile=cores.core_id.str[:3],
    deployment="core", year=2022, alt_cm=cores.core_len.astype(float),
    right_censored=0, alt_lo=cores.core_len.astype(float), alt_hi=cores.core_len.astype(float),
    coverage="core", depth_max_cm=np.nan, n_layers=np.nan, depth_reversed=0))
core_res["core_top"] = cores.core_top
core_res["core_bot"] = cores.core_bot
res = pd.concat([res, core_res], ignore_index=True)
res["stefan_clim_cm"] = round(stefan_clim, 1)
res["stefan_year_cm"] = res.year.map(lambda y: round(stefan_year[y], 1) if y in stefan_year else np.nan)
res["s1_catboost_cm"] = round(s1_mean["catboost"], 1)
res["s1_ftt_cm"] = round(s1_mean["ftt"], 1)
res["s1_ens7_cm"] = round(s1_ens, 1)
res["above_near_obs_cm"] = round(near_mean, 1)
res["cell_dist_km"] = round(float(cell.dist_km), 2)
res.to_csv(PROC / "s7_kpdc_results.csv", index=False)

# ---------- 4) 그림 ----------
use_polar()


def savefig(fig, stem, **kw):
    """논문용 벡터(PDF) + 고해상도 PNG 동시 출력. stem은 확장자 없는 파일명."""
    fig.savefig(FIGD / f"{stem}.png", dpi=DPI, **kw)
    fig.savefig(FIGD / f"{stem}.pdf", **kw)   # 벡터: DPI 무관, 텍스트 편집 가능


def section_fig(keys, stem, note):
    """시간-깊이 온도 단면 2x2 (vik, 0°C 등온선)."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=True, constrained_layout=True)
    norm = tnorm(-16, 16)
    pc = None
    for ax, key in zip(axes.ravel(), keys):
        g = daily[daily.site_key == key]
        piv = g.pivot_table(index="depth_cm", columns="date", values="temp_c")
        piv = piv.reindex(columns=pd.date_range(piv.columns.min(), piv.columns.max()))
        pc = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap=CMAP.temp,
                           norm=norm, shading="nearest")
        try:
            # 0°C 등온선: 인라인 라벨 제거(패널마다 다수 교차로 클러터 유발).
            # 선은 굵게 유지하고 의미는 그림 범례로 한 번만 표기한다.
            ax.contour(piv.columns, piv.index, piv.values, levels=[0.0],
                       colors="k", linewidths=1.2)
        except Exception:
            pass
        for _, r in alt[(alt.site_key == key)].iterrows():
            if not r.right_censored:
                ax.plot(pd.Timestamp(r.peak_day), r.alt_cm, marker="v", ms=7,
                        mfc="w", mec="k", zorder=5)
                ax.annotate(f"ALT {r.alt_cm:.0f}cm", (pd.Timestamp(r.peak_day), r.alt_cm),
                            xytext=(4, -10), textcoords="offset points", fontsize=8)
            else:
                # 검열 주석은 규약상 붉은 계열 금지. 냉색(동결 청색)으로 표기한다.
                ax.annotate(f"{int(r.year)}: 전층 양수, ALT > {r.alt_cm:.0f}cm(검열)",
                            (0.02, 0.05 if r.year % 2 else 0.14), xycoords="axes fraction",
                            fontsize=7.5, color=FROZEN)
        ax.set_ylim(daily[daily.site_key == key].depth_cm.max() + 8, 0)
        ax.set_title(key, fontsize=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    for ax in axes[:, 0]:
        ax.set_ylabel("깊이 (cm)")
    for ax in axes[1, :]:
        ax.set_xlabel("연-월")
    cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("지중온도 (°C)")
    # 0°C 등온선 의미는 그림 전체에 한 번만 표기(패널 인라인 라벨 대체)
    iso_h = plt.Line2D([], [], color="k", lw=1.2, label="0°C 등온선(융해면)")
    alt_h = plt.Line2D([], [], marker="v", ms=7, mfc="w", mec="k", lw=0, label="유도 ALT(비검열)")
    fig.legend(handles=[iso_h, alt_h], loc="upper right", fontsize=8.5,
               ncol=2, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"KPDC 콘슬 지중온도 단면과 0°C 등온선: {note}", fontsize=12)
    savefig(fig, stem)
    plt.close(fig)


section_fig(["ID20_B", "ID21_B", "ID22_B", "ID24_B"], "s7_section_16layer_B",
            "배치 B(2024-09~2025-09, 1.6m 16층)")
section_fig(["ID21_A", "ID22_A", "ID23_A", "ID24_A"], "s7_section_16layer_A",
            "배치 A(2023-09~2024-08, 깊이 역순 교정, 전 시즌 검열)")

# --- 비교 그림: 유도 ALT vs 코어·물리·모델 ---
fig, ax = plt.subplots(figsize=(9.6, 6.0), constrained_layout=True)
x = 0
labels = []
# (1) 유도 ALT 2025 헤드라인
u = head.sort_values("alt_cm")
ax.errorbar(np.full(len(u), x) + np.linspace(-0.18, 0.18, len(u)), u.alt_cm,
            yerr=[u.alt_cm - u.alt_lo, u.alt_hi - u.alt_cm], fmt="o", ms=6,
            color="#1f4e79", ecolor="#1f4e79", capsize=2, lw=1)
ax.errorbar([x + 0.32], [head_med], yerr=[[head_med - head_ci[0]], [head_ci[1] - head_med]],
            fmt="D", ms=8, color="k", capsize=4)
labels.append(f"유도 ALT 2025\n(비검열 n={len(u)})")
x += 1
# (2) 검열 하한(구간검열). 정방향 축이므로 화살표는 위(더 큰 ALT)로 향한다.
# 규약상 붉은계열 금지 → 냉색(CENSOR, 짙은 자주)으로 표기한다.
for dmax, dy in [(80.0, 0), (160.0, 0)]:
    sub = alt[(alt.right_censored == 1) & (alt.depth_max_cm == dmax)]
    if len(sub):
        ax.annotate("", xy=(x, dmax + 22), xytext=(x, dmax),
                    arrowprops=dict(arrowstyle="-|>", color=CENSOR, lw=2))
        ax.plot([x - 0.16, x + 0.16], [dmax, dmax], color=CENSOR, lw=2)
        ax.annotate(f"n={len(sub)}", (x + 0.05, dmax - 12), fontsize=8, color=CENSOR)
ax.plot([], [], color=CENSOR, lw=2, label="검열 하한(전층 양수): ALT > 최심센서")
labels.append("검열 시즌\n(>80/>160cm)")
x += 1
# (3) 코어 2022
bp = ax.boxplot([cores.core_len], positions=[x], widths=0.42, patch_artist=True,
                medianprops=dict(color="k"))
bp["boxes"][0].set(facecolor="#c9d7e4")
ax.scatter(np.full(18, x) + RNG.uniform(-0.1, 0.1, 18), cores.core_len, s=12,
           color="#4a6b8a", zorder=4)
ax.scatter(np.full(18, x) + RNG.uniform(-0.1, 0.1, 18), cores.core_top, s=12,
           facecolor="none", edgecolor="#8a8a8a", zorder=4)
ax.annotate("코어 상단(대안 해석)", (x + 0.18, cores.core_top.mean()), fontsize=7.5, color="#6a6a6a")
labels.append("코어 2022\n길이(n=18)")
x += 1
# (4) ABoVE 인근 실측 셀
ax.scatter(np.full(len(near_obs), x) + RNG.uniform(-0.15, 0.15, len(near_obs)),
           near_obs.alt_cm, s=14, color="#2e7d63")
ax.errorbar([x + 0.3], [near_mean], yerr=[[near_mean - near_ci[0]], [near_ci[1] - near_mean]],
            fmt="D", ms=7, color="#185c44", capsize=4)
labels.append(f"ABoVE 실측\n(≤{NEAR_KM:.0f}km, n={len(near_obs)})")
x += 1
# (5) Stefan 물리
yrs = sorted(y for y in stefan_year if 2022 <= y <= 2024)
ax.scatter([x - 0.15] * len(yrs), [stefan_year[y] for y in yrs], marker="s", s=30,
           color="#7a5c99")
ax.scatter([x + 0.2], [stefan_clim], marker="s", s=60, color="#4b2d73")
cap = " / ".join(f"{y} {stefan_year[y]:.0f}" for y in yrs) + f"\n다년평균 {stefan_clim:.0f}cm"
ax.annotate(cap, (x, 14), ha="center", fontsize=7.5, color="#4b2d73")
labels.append("Stefan\nE·√TDD")
x += 1
# (6) S1 학습모델. 규약상 주황·갈색 금지 → 냉색(청록)으로, 범주 구분은 삼각 마커로 유지.
mods = ["catboost", "ftt", "histgbm"]
ax.scatter([x - 0.15] * 3, [s1_mean[m] for m in mods], marker="^", s=34, color="#2c7fb8")
ax.scatter([x + 0.25], [s1_ens], marker="^", s=64, color="#193f5c")
cap = (f"GBM {s1_mean['histgbm']:.0f} / CatB {s1_mean['catboost']:.0f} / "
       f"FT-T {s1_mean['ftt']:.0f}\n7모델 평균 {s1_ens:.0f}cm")
ax.annotate(cap, (x, 14), ha="center", fontsize=7.5, color="#193f5c")
labels.append("S1 모델 OOF\n(최근접 셀)")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
# ALT는 지표로부터의 깊이가 아니라 활동층 '두께'이므로 정방향 축(0 아래, 큰 값 위).
# 반전축(0 위)은 깊이장으로 오독될 수 있어 사용하지 않는다.
ax.set_ylabel("활동층 두께 ALT (cm)")
ax.set_ylim(0, 200)
ax.set_title("콘슬 ALT 정합: KPDC 유도(2023-2025) vs 코어·실측·물리·학습모델 (알래스카 in-domain)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(axis="y", alpha=0.3)
savefig(fig, "s7_alt_comparison")
plt.close(fig)

# --- 위치 지도(알래스카 + 로컬, inset locator, mask_ocean) ---
fig = plt.figure(figsize=(11.5, 5.4))
ax1 = fig.add_subplot(1, 2, 1, projection=geomap._proj(geomap.ALASKA))
geomap.make_ax(geomap.ALASKA, ax=ax1, fig=fig, title="콘슬 위치(알래스카)")
xs, ys = geomap.to_proj(ax1, [LON], [LAT])
# 규약상 붉은 별 금지. 별 모양(마커)으로 대표점을 구분하고 색은 중립(흑/백 테두리)으로 대비 확보.
ax1.plot(xs, ys, marker="*", ms=17, mfc="k", mec="w", mew=1.0, lw=0)
ax1.annotate("Council", (xs[0], ys[0]), xytext=(8, 6), textcoords="offset points", fontsize=10)
geomap.mask_ocean(ax1)
geomap.add_inset_locator(fig, ax1, geomap.ALASKA)
zoom = ("콘슬 실측 셀", float(near_obs.lon.min()) - 0.012, float(near_obs.lon.max()) + 0.012,
        float(near_obs.lat.min()) - 0.006, float(near_obs.lat.max()) + 0.006)
ax2 = fig.add_subplot(1, 2, 2, projection=geomap._proj(zoom))
geomap.make_ax(zoom, ax=ax2, fig=fig, title="콘슬 일대 ABoVE 실측 셀(약 2km)")
# 컬러바 범위를 실측 분포(약 10-70cm)에 맞춘다. 이전 vmax=120은 실제 최댓값(69cm)의
# 두 배여서 44개 점이 컬러맵 하단(옅은 청색)에 몰려 셀 간 대비가 사라졌다.
alt_vals = near_obs.alt_cm.values
vlo = float(np.floor(np.percentile(alt_vals, 2) / 5) * 5)   # 약 10cm
vhi = float(np.ceil(np.percentile(alt_vals, 98) / 5) * 5)   # 약 65cm
sc = geomap.scatter_map(ax2, near_obs.lon.values, near_obs.lat.values, alt_vals,
                        cmap=CMAP.alt, vmin=vlo, vmax=vhi, s=64, edge=True)
xs, ys = geomap.to_proj(ax2, [LON], [LAT])
# 컬러 점(oslo 청색) 위에서도 대비되도록 흑색 별 + 백색 테두리(붉은색 미사용).
ax2.plot(xs, ys, marker="*", ms=17, mfc="k", mec="w", mew=1.0, lw=0, zorder=6)
ax2.annotate("Council 대표점\n(KPDC 프로파일 좌표 미상)", (xs[0], ys[0]),
             xytext=(10, -18), textcoords="offset points", fontsize=8)
# 지리적 군집(위도 0.85N 기준 남·북 2군 + 대표점 부근 1점)별 실측 평균을 병기해
# 뭉친 점의 값을 숫자로 읽을 수 있게 한다.
near_g = near_obs.assign(_grp=(near_obs.lat > 0.5 * (near_obs.lat.min() + near_obs.lat.max())))
for gv, gsub in near_g.groupby("_grp"):
    gx, gy = geomap.to_proj(ax2, [gsub.lon.mean()], [gsub.lat.mean()])
    ax2.annotate(f"평균 {gsub.alt_cm.mean():.0f}cm (n={len(gsub)})",
                 (gx[0], gy[0]), xytext=(0, 12 if gv else -16),
                 textcoords="offset points", ha="center", fontsize=8, color="#1f3b57",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
geomap.mask_ocean(ax2)
geomap.add_scalebar(ax2, length_km=1)
geomap.add_colorbar(fig, sc, ax2, "ABoVE 실측 ALT (cm)")
savefig(fig, "s7_council_map", bbox_inches="tight")
plt.close(fig)

# ---------- 5) meta ----------
meta = dict(
    purpose="S7 KPDC 콘슬 0°C 등온선 유도 ALT vs 코어·물리·학습모델 정합(대회 KPDC 주 활용)",
    framing="Council=알래스카 in-domain 점검증. 전이(LORO) 근거로 사용 금지.",
    council=dict(lat=LAT, lon=LON, coord_source="kpdc/README(파일 내 좌표 부재)"),
    headline=dict(
        derived_2025_median_cm=round(head_med, 1),
        derived_2025_note=("2025 헤드라인 중앙값은 비검열 부분집합(2025 시즌·우측검열 제외·"
                           "피크 커버리지 peak_days>=40, n=%d) 프로파일의 조건부 추정이다. "
                           "검열 프로파일을 배제한 심부 16층 하위집단 조건부 값이므로 "
                           "콘슬 전체 대표 ALT로 일반화하지 않는다." % int(len(head))),
        derived_2025_ci95=[round(v, 1) for v in head_ci],
        derived_2025_n=int(len(head)),
        core2022_len_mean_cm=round(core_mean, 1),
        core2022_ci95=[round(v, 1) for v in core_ci],
        core2022_top_mean_cm=round(float(cores.core_top.mean()), 1),
        core2022_bot_mean_cm=round(float(cores.core_bot.mean()), 1),
        above_near_obs_mean_cm=round(near_mean, 1),
        above_near_obs_ci95=[round(v, 1) for v in near_ci],
        above_near_obs_n=int(len(near_obs)),
        stefan_clim_cm=round(stefan_clim, 1),
        stefan_year_cm={y: round(stefan_year[y], 1) for y in yrs},
        s1_models_cm={k: round(v, 1) for k, v in s1_mean.items()},
        s1_ens7_cm=round(s1_ens, 1),
        nearest_cell_dist_km=round(float(cell.dist_km), 2),
    ),
    censoring_table=cens_tab.to_dict("records"),
    n_profiles=parse_meta.get("n_profiles"),
    n_profiles_by_deployment=parse_meta.get("n_profiles_by_deployment"),
    n_profiles_note=parse_meta.get("n_profiles_note"),
    n_rows=int(len(alt)), n_censored=int(alt.right_censored.sum()),
    e_global=E_GLOBAL, e_source="s2_physics_meta.json",
    sqrt_tdd_clim=float(cell.e5_sqrt_tdd),
    depth_provenance=parse_meta["depth_provenance"],
    qc=parse_meta["qc"],
    core_interpretation=("README 규약=코어 길이(72-88cm)를 ALT 약라벨로 사용. "
                         "대안 해석(코어 상단 19-36cm=동결면, 하단 104-120cm)도 기록. "
                         "상단은 ABoVE 인근 실측(10-46cm)과, 하단은 16층 유도 ALT(116-153cm)와 "
                         "정합해 해석 불확실성이 큼을 명시."),
    caveats=[
        "KPDC 프로파일·코어의 개별 좌표 미상 → 사이트 대표점(64.85N,-163.70W) 비교",
        "배치 A(2023-24) 깊이 역순 교정은 물리 추론(겨울 단조성) 근거",
        "배치 A 전 시즌 우측 검열(전층 양수) = 심부 미동결 구간(탈릭 의심) 포함 가능",
        "유도 ALT는 연도별(2023-2025), Stefan 다년평균·S1 OOF는 정적 다년평균 대상",
        "확률 모델 없음(결정론적 파싱·유도) → seed 반복 비적용, CI는 부트스트랩",
    ],
    nboot=NBOOT, smoke=SMOKE, runtime_s=round(time.time() - T0, 1),
)
with open(PROC / "s7_kpdc_meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(json.dumps(meta["headline"], ensure_ascii=False, indent=1))
print("censoring:", cens_tab.to_string(index=False))
print(f"[done] figures -> {FIGD}, runtime {time.time() - T0:.1f}s")
