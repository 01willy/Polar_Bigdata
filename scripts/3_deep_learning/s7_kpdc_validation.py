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
      FIGS_ONLY=1 이면 기존 산출물(s7_kpdc_results.csv·s7_kpdc_meta.json)에서 수치를
      읽어 그림만 재생성한다(엑셀 파싱·ERA5 연도별 적재·부트스트랩·결과 저장 생략).
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
FIGS_ONLY = os.environ.get("FIGS_ONLY", "0") == "1"
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

# ABoVE 인근 실측(지도·산점 그림에 개별 점 필요. 기존 processed CSV 필터, 재계산 아님)
fb = pd.read_csv(PROC / "fidelity_base.csv",
                 usecols=["loc_id", "lat", "lon", "region", "source_id", "alt_cm", "e5_sqrt_tdd"])
fb["dist_km"] = dist_km(fb.lat, fb.lon)
cell = fb.nsmallest(1, "dist_km").iloc[0]
near_obs = fb[(fb.dist_km <= NEAR_KM) & (fb.source_id == "F4_direct")]

# 헤드라인 부분집합(그림 공통)
head = alt[(alt.year == 2025) & (alt.right_censored == 0) & (alt.peak_days >= 40)]
head_vals = head.alt_cm.values

if FIGS_ONLY:
    # 그림 전용 분기: 기존 산출물에서 수치만 적재(엑셀·ERA5 연도별·OOF·부트스트랩 생략)
    hl = json.load(open(PROC / "s7_kpdc_meta.json"))["headline"]
    head_med, head_ci = hl["derived_2025_median_cm"], tuple(hl["derived_2025_ci95"])
    core_mean, core_ci = hl["core2022_len_mean_cm"], tuple(hl["core2022_ci95"])
    near_mean, near_ci = hl["above_near_obs_mean_cm"], tuple(hl["above_near_obs_ci95"])
    stefan_clim = hl["stefan_clim_cm"]
    stefan_year = {int(y): float(v) for y, v in hl["stefan_year_cm"].items()}
    s1_mean, s1_ens = hl["s1_models_cm"], hl["s1_ens7_cm"]
    crows = pd.read_csv(PROC / "s7_kpdc_results.csv").query("method == 'core_2022'")
    cores = pd.DataFrame(dict(core_id=crows.site_key.values,
                              core_top=crows.core_top.values,
                              core_bot=crows.core_bot.values,
                              core_len=crows.alt_cm.values))
    assert len(cores) == 18, "2022 코어는 18개여야 한다"
else:
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

    # 최근접 기후 셀(Stefan)
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
    head_med = float(np.median(head_vals))
    head_ci = boot_ci(head_vals)
    core_mean = float(cores.core_len.mean())
    core_ci = boot_ci(cores.core_len, stat=np.mean)
    near_mean = float(near_obs.alt_cm.mean())
    near_ci = boot_ci(near_obs.alt_cm, stat=np.mean)

cens_tab = (alt.groupby(["deployment", "depth_max_cm", "right_censored"])
            .size().rename("n").reset_index())

# ---------- 3) 통합 결과 CSV ----------
if not FIGS_ONLY:
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


def section_fig(keys, stem):
    """시간-깊이 온도 단면 2x2 (vik, 0°C 등온선). 내부 제목 없음(캡션이 대신)."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=True, constrained_layout=True)
    norm = tnorm(-16, 16)
    pc = None
    for i, (ax, key) in enumerate(zip(axes.ravel(), keys)):
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
                            xytext=(4, -10), textcoords="offset points", fontsize=9)
            else:
                # 검열 주석은 규약상 붉은 계열 금지. 냉색(동결 청색)으로 표기한다.
                ax.annotate(f"{int(r.year)}: 전층 양수, ALT > {r.alt_cm:.0f}cm(검열)",
                            (0.02, 0.05 if r.year % 2 else 0.15), xycoords="axes fraction",
                            fontsize=8.5, color=FROZEN)
        ax.set_ylim(daily[daily.site_key == key].depth_cm.max() + 8, 0)
        ax.set_title(f"({chr(97 + i)}) {key}", fontsize=11, loc="left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    for ax in axes[:, 0]:
        ax.set_ylabel("깊이 (cm)")
    for ax in axes[1, :]:
        ax.set_xlabel("연-월")
    cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("지중온도 (°C)")
    # 0°C 등온선 의미는 그림 전체에 한 번만 표기(패널 인라인 라벨 대체).
    # loc="outside ..."로 constrained_layout이 범례 공간을 예약해 패널 라벨과 겹치지 않는다.
    iso_h = plt.Line2D([], [], color="k", lw=1.2, label="0°C 등온선(융해면)")
    alt_h = plt.Line2D([], [], marker="v", ms=7, mfc="w", mec="k", lw=0, label="유도 ALT(비검열)")
    fig.legend(handles=[iso_h, alt_h], loc="outside upper right", fontsize=9.5,
               ncol=2, frameon=False)
    savefig(fig, stem)
    plt.close(fig)


# 배치 B(2024-09~2025-09, 1.6m 16층) / 배치 A(2023-09~2024-08, 깊이 역순 교정, 전 시즌 검열)
# 위 배치 설명은 보고서 캡션에 기재한다(그림 내부 제목 없음).
section_fig(["ID20_B", "ID21_B", "ID22_B", "ID24_B"], "s7_section_16layer_B")
section_fig(["ID21_A", "ID22_A", "ID23_A", "ID24_A"], "s7_section_16layer_A")

# --- 비교 그림: 0°C 등온선 ALT vs 코어·물리·학습모델 ---
# 색: 저채도 냉색 4단(진청·청록·청회·연청회). 범주 구분은 위치·마커가 담당하고
# 색은 관측 계열(진청)·물리/모델 계열(청록)만 구분한다. 값 라벨은 작은 회색.
C_OBS, C_MOD, C_MUT, C_PALE = "#2f4b6e", "#4a7c8c", "#8296a8", "#b8c4d0"
C_TXT = "#444444"
fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
x = 0
labels = []
# (1) 0°C 등온선 ALT 2025 헤드라인
# 개별 점의 세로선 = 0°C 교차를 감싸는 인접 센서 깊이 구간 [alt_lo, alt_hi](센서 간격 전파).
# 측정 불확실성이 아니라 깊이 이산화 구간이므로 범례에 정의를 명시한다.
u = head.sort_values("alt_cm")
h_pt = ax.errorbar(np.full(len(u), x) + np.linspace(-0.18, 0.18, len(u)), u.alt_cm,
                   yerr=[u.alt_cm - u.alt_lo, u.alt_hi - u.alt_cm], fmt="o", ms=5,
                   color=C_OBS, ecolor=C_MUT, capsize=1.8, lw=0.9, mew=0)
ax.errorbar([x + 0.34], [head_med], yerr=[[head_med - head_ci[0]], [head_ci[1] - head_med]],
            fmt="D", ms=6, color="#33414f", ecolor="#33414f", capsize=3, lw=1.0)
labels.append(f"0°C 등온선 ALT\n2025 (n={len(u)})")
x += 1
# (2) 검열 하한(구간검열). 정방향 축이므로 화살표는 위(더 큰 ALT)로 향한다.
for dmax in (80.0, 160.0):
    sub = alt[(alt.right_censored == 1) & (alt.depth_max_cm == dmax)]
    if len(sub):
        ax.annotate("", xy=(x, dmax + 20), xytext=(x, dmax),
                    arrowprops=dict(arrowstyle="-|>", color=C_MUT, lw=1.2,
                                    mutation_scale=10))
        ax.plot([x - 0.17, x + 0.17], [dmax, dmax], color=C_MUT, lw=1.4)
        ax.annotate(f"n={len(sub)}", (x + 0.06, dmax - 12), fontsize=9, color=C_TXT)
labels.append("검열 시즌 하한\n(>80 · >160 cm)")
x += 1
# (3) 코어 2022
bp = ax.boxplot([cores.core_len], positions=[x], widths=0.44, patch_artist=True,
                medianprops=dict(color="#33414f", lw=1.0),
                boxprops=dict(lw=0.6, edgecolor=C_MUT),
                whiskerprops=dict(lw=0.6, color=C_MUT),
                capprops=dict(lw=0.6, color=C_MUT),
                flierprops=dict(marker="o", ms=3, mfc="none", mec=C_MUT, mew=0.5))
bp["boxes"][0].set(facecolor=C_PALE, alpha=0.55)
ax.scatter(np.full(18, x) + RNG.uniform(-0.1, 0.1, 18), cores.core_len, s=11,
           color=C_OBS, alpha=0.85, lw=0, zorder=4)
ax.scatter(np.full(18, x) + RNG.uniform(-0.1, 0.1, 18), cores.core_top, s=11,
           facecolor="none", edgecolor=C_MUT, lw=0.6, zorder=4)
# 우측(ABoVE 점군 방향) 대신 좌측 여백에 배치해 산점과의 겹침을 피한다.
ax.annotate("코어 상단\n(대안 해석)", (x - 0.24, cores.core_top.mean()), ha="right",
            va="center", fontsize=8.5, color=C_TXT, linespacing=1.25)
labels.append("코어 길이 2022\n(n=18)")
x += 1
# (4) ABoVE 인근 실측 셀
ax.scatter(np.full(len(near_obs), x) + RNG.uniform(-0.15, 0.15, len(near_obs)),
           near_obs.alt_cm, s=11, color=C_OBS, alpha=0.7, lw=0)
ax.errorbar([x + 0.32], [near_mean], yerr=[[near_mean - near_ci[0]], [near_ci[1] - near_mean]],
            fmt="D", ms=6, color="#33414f", ecolor="#33414f", capsize=3, lw=1.0)
labels.append(f"ABoVE 실측\n(≤{NEAR_KM:.0f} km, n={len(near_obs)})")
x += 1
# (5) Stefan 물리. 값은 표식 위치로만 읽는다(숫자 목록 중복 표기 제거).
yrs = sorted(y for y in stefan_year if 2022 <= y <= 2024)
ax.scatter([x - 0.16] * len(yrs), [stefan_year[y] for y in yrs], marker="s", s=22,
           color=C_MOD, alpha=0.85, lw=0)
ax.scatter([x + 0.22], [stefan_clim], marker="s", s=46, color=C_MOD, lw=0)
ax.annotate("다년평균", (x + 0.22, stefan_clim), xytext=(0, 9),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8, color=C_TXT)
labels.append("Stefan 물리\nE·√TDD (2022–2024)")
x += 1
# (6) 학습모델(교차검증 예측). 범주 구분은 삼각 마커로 유지.
# 모델명은 값 없이 이름만 병기해 표식-텍스트 이중 부호화를 없앤다.
mods = ["catboost", "ftt", "histgbm"]
ax.scatter([x - 0.16] * 3, [s1_mean[m] for m in mods], marker="^", s=24,
           color=C_MOD, alpha=0.85, lw=0)
ax.scatter([x + 0.24], [s1_ens], marker="^", s=50, color=C_MOD, lw=0)
ax.annotate("앙상블 평균", (x + 0.24, s1_ens), xytext=(0, 9),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8, color=C_TXT)
ax.annotate("GBM · CatBoost\nFT-Transformer", (x - 0.16, 4), ha="center",
            va="bottom", fontsize=8, color=C_TXT, linespacing=1.35)
labels.append("학습모델 예측\n(교차검증)")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlim(-0.55, len(labels) - 0.4)
# ALT는 지표로부터의 깊이가 아니라 활동층 '두께'이므로 정방향 축(0 아래, 큰 값 위).
# 반전축(0 위)은 깊이장으로 오독될 수 있어 사용하지 않는다.
ax.set_ylabel("활동층 두께 ALT (cm)", fontsize=10.5)
ax.set_ylim(0, 200)
ax.tick_params(labelsize=9.5, length=3)
# 내부 제목 없음(캡션 대체). 범례는 좌상단 점군과 겹치지 않게 우상단 여백으로.
# 통계 표기(세로선·상자·수염·요약 표식)는 모두 범례에서 정의한다.
import matplotlib.patches as mpatches
h_cens = plt.Line2D([], [], color=C_MUT, lw=1.4, marker="^", ms=5, mfc=C_MUT, mec=C_MUT)
h_summ = plt.Line2D([], [], color="#33414f", lw=1.0, marker="D", ms=6)
h_box = mpatches.Patch(facecolor=C_PALE, edgecolor=C_MUT, lw=0.6, alpha=0.55)
ax.legend([h_pt, h_cens, h_box, h_summ],
          ["개별 프로파일 · 세로선 = 인접 센서 깊이 구간",
           "검열 하한: ALT > 최심 센서 깊이",
           "상자 = 사분위수(Q1–Q3) · 수염 = 1.5×IQR",
           "중앙값 · 평균 [95% CI]"],
          loc="upper right", fontsize=9, frameon=False, handlelength=1.6,
          labelspacing=0.45, borderaxespad=0.2)
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
savefig(fig, "s7_alt_comparison")
plt.close(fig)

# --- 위치 지도(광역 + 로컬, inset locator, mask_ocean) ---
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import Gridliner as _GL


def _extent_aspect(extent):
    """투영 좌표에서의 폭/높이 비. 두 패널의 렌더 높이를 맞추는 데 쓴다."""
    _, lo0, lo1, la0, la1 = extent
    lo = np.linspace(lo0, lo1, 40)
    la = np.linspace(la0, la1, 40)
    LO, LA = np.meshgrid(lo, la)
    p = geomap._proj(extent).transform_points(geomap.ccrs.PlateCarree(),
                                              LO.ravel(), LA.ravel())
    return float((p[:, 0].max() - p[:, 0].min()) / (p[:, 1].max() - p[:, 1].min()))


def _tune_gl(ax, nx=4, ny=4, xlocs=None):
    """격자 라벨: 작은 회색·수평·개수 축소(내부 제목 없이 축만으로 위치 전달)."""
    for gl in [a for a in ax.artists if isinstance(a, _GL)]:
        gl.xlabel_style = {"size": 8.5, "color": "0.4"}
        gl.ylabel_style = {"size": 8.5, "color": "0.4"}
        gl.rotate_labels = False
        gl.xlocator = mticker.FixedLocator(xlocs) if xlocs else mticker.MaxNLocator(nx)
        gl.ylocator = mticker.MaxNLocator(ny)


zlo0, zlo1 = float(near_obs.lon.min()) - 0.012, float(near_obs.lon.max()) + 0.012
zla0, zla1 = float(near_obs.lat.min()) - 0.006, float(near_obs.lat.max()) + 0.006
# 확대 패널의 경도 여백을 넓혀 투영 종횡비를 1에 맞춘다. 두 패널 높이를 같게 두면서도
# 폭 비를 1.2:1 수준으로 좁혀 좌우 비대칭과 패널 간 과대 여백을 함께 줄인다.
_need_lon = (zla1 - zla0) / np.cos(np.deg2rad(LAT))
_pad = max(0.0, 0.5 * (_need_lon - (zlo1 - zlo0)))
zoom = ("현장 실측 셀", zlo0 - _pad, zlo1 + _pad, zla0, zla1)
aspA, aspB = _extent_aspect(geomap.ALASKA), _extent_aspect(zoom)
# 패널 폭을 각 투영 종횡비에 비례시켜 두 지도의 높이를 동일하게 만든다(비대칭 해소).
fig = plt.figure(figsize=(9.8, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[aspA, aspB], left=0.055, right=0.895,
                      top=0.93, bottom=0.14, wspace=0.18)
ax1 = fig.add_subplot(gs[0, 0], projection=geomap._proj(geomap.ALASKA))
geomap.make_ax(geomap.ALASKA, ax=ax1, fig=fig)   # 내부 제목 없음, 패널 라벨만
ax1.set_title("(a)", loc="left", fontsize=11, fontweight="bold")
_tune_gl(ax1, ny=4, xlocs=[-165, -160, -155, -150, -145])
xs, ys = geomap.to_proj(ax1, [LON], [LAT])
# 규약상 붉은 별 금지. 별 모양(마커)으로 대표점을 구분하고 색은 중립(진회/백테두리).
ax1.plot(xs, ys, marker="*", ms=11, mfc="#33414f", mec="w", mew=0.8, lw=0, zorder=6)
ax1.annotate("Council", (xs[0], ys[0]), xytext=(8, 5), textcoords="offset points",
             fontsize=9.5, color="#333333")
geomap.mask_ocean(ax1)
geomap.add_inset_locator(fig, ax1, geomap.ALASKA)

ax2 = fig.add_subplot(gs[0, 1], projection=geomap._proj(zoom))
geomap.make_ax(zoom, ax=ax2, fig=fig)            # 내부 제목 없음, 패널 라벨만
ax2.set_title("(b)", loc="left", fontsize=11, fontweight="bold")
_tune_gl(ax2, nx=3, ny=4)
# 컬러바 범위는 실측 분포 5-95 분위수로 잡아 과포화를 막는다.
alt_vals = near_obs.alt_cm.values
vlo = float(np.floor(np.percentile(alt_vals, 5) / 5) * 5)
vhi = float(np.ceil(np.percentile(alt_vals, 95) / 5) * 5)
sc = geomap.scatter_map(ax2, near_obs.lon.values, near_obs.lat.values, alt_vals,
                        cmap=CMAP.alt, vmin=vlo, vmax=vhi, s=44, edge=True)
xs, ys = geomap.to_proj(ax2, [LON], [LAT])
ax2.plot(xs, ys, marker="*", ms=11, mfc="#33414f", mec="w", mew=0.8, lw=0, zorder=6)
# GeoAxes는 축 경계로 텍스트를 클리핑하므로 우측(컬러바 쪽) 대신 별 좌측에 배치해 잘림 방지.
ax2.annotate("Council 관측지", (xs[0], ys[0]), xytext=(-9, 0),
             textcoords="offset points", ha="right", va="center",
             fontsize=9, color="#333333")
# 지리적 군집별 실측 평균·개수를 병기해 뭉친 점의 값을 숫자로 읽게 한다.
# 위도 중앙 분할은 고립점을 남쪽 군집에 흡수시켜 라벨 없는 점을 만들었다.
# 단일연결(200 m 임계) 군집화로 바꿔 고립점(n=1)도 반드시 라벨이 붙게 한다.
_pt = near_obs[["lat", "lon"]].to_numpy(float)
_kx = 111.0 * np.cos(np.deg2rad(LAT))
_d = np.hypot((_pt[:, None, 0] - _pt[None, :, 0]) * 111.0,
              (_pt[:, None, 1] - _pt[None, :, 1]) * _kx)
_cid = np.full(len(_pt), -1, int)
_c = 0
for _i in range(len(_pt)):
    if _cid[_i] >= 0:
        continue
    _cid[_i] = _c
    _stack = [_i]
    while _stack:
        _j = _stack.pop()
        for _k in np.where((_d[_j] <= 0.2) & (_cid < 0))[0]:
            _cid[_k] = _c
            _stack.append(_k)
    _c += 1
_latmid = 0.5 * (near_obs.lat.min() + near_obs.lat.max())
near_g = near_obs.assign(_grp=_cid)
for gv, gsub in near_g.groupby("_grp"):
    gx, gy = geomap.to_proj(ax2, [gsub.lon.mean()], [gsub.lat.mean()])
    if len(gsub) == 1:
        # 단일 관측은 평균이 아니라 관측값 그대로임을 표기하고, 군집 라벨과 겹치지
        # 않도록 점 우측에 둔다.
        ax2.annotate(f"{gsub.alt_cm.iloc[0]:.0f} cm (n=1)", (gx[0], gy[0]),
                     xytext=(9, 0), textcoords="offset points", ha="left",
                     va="center", fontsize=8.5, color="#444444")
    else:
        ax2.annotate(f"평균 {gsub.alt_cm.mean():.0f} cm (n={len(gsub)})",
                     (gx[0], gy[0]),
                     xytext=(0, 11 if gsub.lat.mean() > _latmid else -17),
                     textcoords="offset points", ha="center", fontsize=8.5,
                     color="#444444")
geomap.mask_ocean(ax2)
geomap.add_scalebar(ax2, length_km=1)
# 컬러바를 패널 밖 inset으로 붙여 (b) 축 박스 크기를 유지한다(두 패널 높이 동일 유지).
cax = ax2.inset_axes([1.045, 0.0, 0.038, 1.0])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("ABoVE 실측 ALT (cm)", fontsize=9.5)
cb.ax.tick_params(labelsize=8.5, length=2.5)
cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor("#888888")
savefig(fig, "s7_council_map")
plt.close(fig)

# ---------- 5) meta ----------
if FIGS_ONLY:
    print("[figs-only] 그림만 재생성, 결과 CSV·meta 미갱신")
    print(f"[done] figures -> {FIGD}, runtime {time.time() - T0:.1f}s")
    sys.exit(0)

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
