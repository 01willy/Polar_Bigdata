"""S9 마감: 연도별 ALT timelapse 재렌더 + 시간정합 문서화 (재학습 없음, CPU 전용).

기존 결과 재사용
- timelapse_alaska_meta.json : 연도별 fold-safe Stefan (a, E). 각 연도의 (a,E)는 해당
  연도를 제외한 나머지 연도에서 적합된 값이므로 dense 예측도 leave-one-year-out 정합이다.
- timelapse_alaska_oof.csv   : temporal holdout OOF. 헤드라인 수치의 블록 부트스트랩 CI 산출.
- alt_era5_temporal.csv      : 연도별 ERA5-Land forcing(2010-2024, 197,648 위치 완전 패널).
  기존 프레임은 그해 관측 위치(21~64,114개, 연도별 급변)만 그려 커버리지 깜빡임이 있었다.
  본 렌더는 전 위치 dense 예측으로 커버리지를 고정한 진짜 연별 지도를 만든다.

렌더 규약
- cartopy AlbersEqualArea(ABoVE 영역: 알래스카+서부 캐나다), hexbin 셀통계(median),
  mask_ocean, 고정 색축(전 연도 공통 vmin/vmax), CMAP.alt(oslo_r 절단), 컬러바 단위 cm.
- 프레임 픽셀 크기 고정(bbox_inches tight 미사용) 후 PIL 공통 adaptive palette(median-cut)
  + Floyd-Steinberg dither로 GIF 생성(밴딩 최소화). 기존 GIF는 덮지 않고 _v2로 저장.
- 정직 각주: within-site 연 anomaly는 예측 불가(corr 0.06)임을 프레임과 meta에 명시.

실행: SMOKE=1 python scripts/4_visualization/s9_timelapse_figs.py  (검증 후 SMOKE 없이 본 실행)
      PANELS=1 → 대표 4연도 패널 그림만 재생성(프레임·GIF·부트스트랩·메타 생략, figs-only)

산출물 게재 구분
- frames/frame_YYYY.png, frames/anom_frame_YYYY.png 은 GIF 전용 중간산출물이다.
  연도 스탬프·방법론 제목·각주는 애니메이션 판독을 위한 것으로, 보고서에 직접 게재하지
  않는다(게재용 정지 그림은 alt_year_panels_v2·alt_anom_year_panels_v1 패널만 사용).
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP
from polar.geomap import (make_ax, hexbin_map, add_colorbar, add_scalebar,
                          add_inset_locator, mask_ocean, _proj)

try:
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import Gridliner
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False

# 본토 경도 범위 168~141°W. 자동 격자가 120°W 한 개만 라벨하는 문제를
# 명시 눈금(160·150·140°W)으로 교체한다. make_ax의 격자를 덮어씀.
LON_TICKS = [-160, -150, -140]
LAT_TICKS = [60, 63, 66, 69, 72]


def fix_graticule(ax, label_size=8, left_labels=True, bottom_labels=True):
    """make_ax가 그린 자동 경도 격자(120°W 하나만 라벨) 대신 160·150·140°W를 명시.
    make_ax가 이미 만든 gridliner의 눈금을 고정 locator로 덮어써서 120°W 단일 라벨
    문제를 없앤다(gridliner를 추가하지 않고 기존 것을 갱신). cartopy 0.23에서 gridliner는
    ax.get_children()에 존재한다(ax._gridliners 아님).
    left_labels/bottom_labels: 다패널 그림에서 왼쪽 열·아래 행에만 라벨을 남겨
    큰 폰트(12pt)에서도 패널 간 라벨 충돌을 막는다."""
    if not (_HAS_CARTOPY and getattr(ax, "_is_geo", False)):
        return
    gls = [c for c in ax.get_children() if isinstance(c, Gridliner)]
    for gl in gls:
        gl.xlocator = mticker.FixedLocator(LON_TICKS)
        gl.ylocator = mticker.FixedLocator(LAT_TICKS)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlabel_style = gl.ylabel_style = {"size": label_size, "color": "0.3"}

def add_panel_scalebar(ax, font_size=9):
    """패널 전용 스케일바(geomap.add_scalebar와 동일 규약, 축소 게재용 폰트 확대).
    공유 모듈을 바꾸지 않고 이 그림에서만 폰트를 키운다. 실패 시 조용히 스킵."""
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        if getattr(ax, "_is_geo", False):
            ax.add_artist(ScaleBar(1, units="m", location="lower right",
                                   length_fraction=0.25, box_alpha=0.7, color="0.2",
                                   font_properties={"size": font_size}))
    except Exception:
        pass


SMOKE = os.environ.get("SMOKE", "0") == "1"
# figs-only 분기: 대표 패널(정적 그림)만 재생성. 프레임·GIF·부트스트랩·meta는 기존 산출 유지.
PANELS_ONLY = os.environ.get("PANELS", "0") == "1"
t0 = time.time()
rng = np.random.default_rng(0)
use_polar()

PROC = C.PROCESSED
REPO = Path(__file__).resolve().parents[2]
if SMOKE:
    FIGDIR = Path("/tmp/s9_smoke/figures")
    ANIMDIR = Path("/tmp/s9_smoke/animations")
    META_OUT = Path("/tmp/s9_smoke/s9_timelapse_meta.json")
else:
    FIGDIR = C.FIGURES / "s9_timelapse"
    ANIMDIR = REPO / "outputs" / "animations"
    META_OUT = PROC / "s9_timelapse_meta.json"
FRAMEDIR = FIGDIR / "frames"
FRAMEDIR.mkdir(parents=True, exist_ok=True)
ANIMDIR.mkdir(parents=True, exist_ok=True)

# ABoVE 관측역(알래스카+서부 캐나다). 데이터 범위 lat 60.4-71.3, lon -166.0~-113.0
ABOVE = ("알래스카·서부 캐나다 (ABoVE)", -166.8, -112.5, 59.8, 72.1)
GRIDSIZE = 46  # hexbin 셀 크기(관측 클러스터가 보이도록 과소분할 방지)
PANEL_GRIDSIZE = 30  # 4패널 정적 그림 전용: 축소 게재 시 희소 hex 판독성을 위해 셀 확대

# =====================================================================
# 1. 기존 결과 로드(재학습 없음)
# =====================================================================
with open(PROC / "timelapse_alaska_meta.json") as f:
    meta0 = json.load(f)
YEARS = [int(y) for y in meta0["years"]]
EA = {int(r["holdout_year"]): (float(r["a"]), float(r["E"]))
      for r in meta0["E_fits_by_holdout_year"]}
print(f"[load] 연도 {YEARS[0]}-{YEARS[-1]}  headline={meta0['headline_model']}  "
      f"holdout RMSE={meta0['temporal_holdout']['STEFAN']['rmse_cm']}cm")

era = pd.read_csv(PROC / "alt_era5_temporal.csv",
                  usecols=["lat", "lon", "year", "e5t_sqrt_tdd"])
era["year"] = era["year"].astype(int)
era = era.dropna(subset=["e5t_sqrt_tdd"])
print(f"[load] alt_era5_temporal {len(era):,}행  위치/연 "
      f"{era.groupby('year').size().iloc[0]:,}")

oof = pd.read_csv(PROC / "timelapse_alaska_oof.csv")
obs_per_year = oof.groupby("year").size().to_dict()

# dense 예측: 그해 (a,E)에 그해 sqrt(TDD). fold-safe(해당 연도 미포함 적합값).
era["pred_cm"] = np.nan
for yr in YEARS:
    a, E = EA[yr]
    m = era["year"].values == yr
    era.loc[m, "pred_cm"] = np.clip(a + E * era.loc[m, "e5t_sqrt_tdd"].values, 1.0, 600.0)

# 고정 색축: 전 연도 dense 예측의 p2-p98을 5cm 단위로 라운딩
vlo, vhi = np.nanpercentile(era["pred_cm"].values, [2, 98])
vlo = float(np.floor(vlo / 5) * 5)
vhi = float(np.ceil(vhi / 5) * 5)
print(f"[color] 고정 색축 {vlo:.0f}-{vhi:.0f} cm (전 연도 공통)")

RENDER_YEARS = [YEARS[0], YEARS[-1]] if SMOKE else YEARS

# =====================================================================
# 2. 프레임 렌더(픽셀 크기 고정: bbox tight 미사용)
#    GIF 전용 중간산출물: 제목·각주·연도 스탬프는 애니메이션 판독용이며
#    보고서 게재용 정지 그림(섹션 4 패널)에는 적용되지 않는다.
# =====================================================================
FOOT = ("주: 연도 간 차이는 그해 ERA5-Land forcing 반영이다. within-site 연 anomaly "
        f"예측력은 없음(corr {meta0['anom_corr']['STEFAN']:+.2f}). "
        f"연도 홀드아웃 $R^2$={meta0['temporal_holdout']['STEFAN']['r2']:.2f}.")
proj = _proj(ABOVE)


def render_frame(yr):
    fig = plt.figure(figsize=(7.6, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    make_ax(ABOVE, ax=ax, fig=fig,
            title="연도별 예측 ALT · STEFAN(a+E·√TDD, 그해 ERA5-Land, fold-safe E)")
    fix_graticule(ax)
    sub = era[era["year"] == yr]
    hb = hexbin_map(ax, sub["lon"].values, sub["lat"].values, sub["pred_cm"].values,
                    gridsize=GRIDSIZE, cmap=CMAP.alt, vmin=vlo, vmax=vhi)
    mask_ocean(ax)
    add_scalebar(ax, length_km=200)
    add_colorbar(fig, hb, ax, "예측 ALT (cm)", shrink=0.85)
    # 연도 스탬프 + 그해 실측 수
    ax.text(0.975, 0.965, f"{yr}", transform=ax.transAxes, ha="right", va="top",
            fontsize=24, fontweight="bold", color="#16324f", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.85))
    ax.text(0.975, 0.845, f"실측 ALT n={obs_per_year.get(yr, 0):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="0.25",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaa", alpha=0.8))
    fig.text(0.01, 0.008, FOOT, fontsize=6.8, color="0.35", ha="left", va="bottom")
    fp = FRAMEDIR / f"frame_{yr}.png"
    fig.savefig(fp, dpi=140)  # bbox 고정(프레임 크기 동일)
    plt.close(fig)
    return fp


frame_paths = []
if not PANELS_ONLY:
    for yr in RENDER_YEARS:
        t1 = time.time()
        frame_paths.append(render_frame(yr))
        print(f"[frame] {yr}  ({time.time()-t1:.1f}s)")

# =====================================================================
# 3. GIF v2: 공통 adaptive palette + FS dither(밴딩 최소화)
# =====================================================================
from PIL import Image

if not PANELS_ONLY:
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    assert len({im.size for im in imgs}) == 1, "프레임 픽셀 크기 불일치"
    # 공통 팔레트: 전 프레임 축소 몽타주에서 median-cut 256색
    tw, th = imgs[0].width // 4, imgs[0].height // 4
    mont = Image.new("RGB", (tw, th * len(imgs)))
    for i, im in enumerate(imgs):
        mont.paste(im.resize((tw, th)), (0, i * th))
    pal = mont.quantize(colors=256, method=Image.MEDIANCUT)
    qs = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in imgs]
    durations = [800] * len(qs)
    durations[-1] = 1600  # 마지막 프레임 홀드
    gif_path = ANIMDIR / ("timelapse_alt_alaska_v2_smoke.gif" if SMOKE
                          else "timelapse_alt_alaska_v2.gif")
    qs[0].save(gif_path, save_all=True, append_images=qs[1:], duration=durations,
               loop=0, optimize=False, disposal=2)
    print(f"[gif] {gif_path}  ({gif_path.stat().st_size/1e6:.2f} MB, {len(qs)}프레임)")

# =====================================================================
# 3b. anomaly 프레임 + GIF: 연도-다년평균 편차(broc, 0중심). forcing 연변동 가시화.
#     절대 ALT의 연간 변화(0.68%)는 사실상 비가시이므로 편차 렌더를 추가한다.
#     기존 절대값 v2 GIF는 유지하고 별도 산출물로 저장.
# =====================================================================
from matplotlib.colors import Normalize

# 위치별 다년평균 예측을 빼 연도 편차(cm)를 만든다. 같은 위치 패널이므로 forcing 연변동만 남는다.
era["pred_mean"] = era.groupby(["lat", "lon"])["pred_cm"].transform("mean")
era["pred_anom"] = era["pred_cm"] - era["pred_mean"]
# 대칭 색축: 전 연도 편차 |p2|,|p98| 최대를 5cm 단위 라운딩(0이 중앙)
alo, ahi = np.nanpercentile(era["pred_anom"].values, [2, 98])
amax = float(np.ceil(max(abs(alo), abs(ahi)) / 5) * 5)
anorm = Normalize(vmin=-amax, vmax=amax)
print(f"[anom] 대칭 색축 ±{amax:.0f} cm (전 연도 공통, 0중심 broc)")

FOOT_A = ("주: 색은 그해 예측 ALT에서 위치별 다년평균을 뺀 편차이며 그해 ERA5-Land forcing의 "
          "연변동을 나타낸다. 절대 ALT의 연간 변화는 평균 0.7%로 절대값 지도에서는 비가시이다. "
          f"within-site 연 anomaly 예측력은 없음(corr {meta0['anom_corr']['STEFAN']:+.2f}).")


def render_anom_frame(yr):
    fig = plt.figure(figsize=(7.6, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    make_ax(ABOVE, ax=ax, fig=fig,
            title="연도별 ALT 편차(그해값 빼기 다년평균) · STEFAN(그해 ERA5-Land forcing)")
    fix_graticule(ax)
    sub = era[era["year"] == yr]
    hb = hexbin_map(ax, sub["lon"].values, sub["lat"].values, sub["pred_anom"].values,
                    gridsize=GRIDSIZE, cmap=CMAP.diff, norm=anorm)
    mask_ocean(ax)
    add_scalebar(ax, length_km=200)
    add_colorbar(fig, hb, ax, "ALT 편차 (cm, 그해값 빼기 다년평균)", shrink=0.85)
    ax.text(0.975, 0.965, f"{yr}", transform=ax.transAxes, ha="right", va="top",
            fontsize=24, fontweight="bold", color="#16324f", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", alpha=0.85))
    ax.text(0.975, 0.845, f"실측 ALT n={obs_per_year.get(yr, 0):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="0.25",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaa", alpha=0.8))
    fig.text(0.01, 0.008, FOOT_A, fontsize=6.8, color="0.35", ha="left", va="bottom")
    fp = FRAMEDIR / f"anom_frame_{yr}.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)
    return fp


anom_paths = []
if not PANELS_ONLY:
    for yr in RENDER_YEARS:
        t1 = time.time()
        anom_paths.append(render_anom_frame(yr))
        print(f"[anom-frame] {yr}  ({time.time()-t1:.1f}s)")

    aimgs = [Image.open(p).convert("RGB") for p in anom_paths]
    assert len({im.size for im in aimgs}) == 1, "anomaly 프레임 픽셀 크기 불일치"
    atw, ath = aimgs[0].width // 4, aimgs[0].height // 4
    amont = Image.new("RGB", (atw, ath * len(aimgs)))
    for i, im in enumerate(aimgs):
        amont.paste(im.resize((atw, ath)), (0, i * ath))
    apal = amont.quantize(colors=256, method=Image.MEDIANCUT)
    aqs = [im.quantize(palette=apal, dither=Image.FLOYDSTEINBERG) for im in aimgs]
    adurations = [800] * len(aqs)
    adurations[-1] = 1600  # 마지막 프레임 홀드
    anom_gif_path = ANIMDIR / ("timelapse_alt_anomaly_v1_smoke.gif" if SMOKE
                               else "timelapse_alt_anomaly_v1.gif")
    aqs[0].save(anom_gif_path, save_all=True, append_images=aqs[1:], duration=adurations,
                loop=0, optimize=False, disposal=2)
    print(f"[anom-gif] {anom_gif_path}  ({anom_gif_path.stat().st_size/1e6:.2f} MB, "
          f"{len(aqs)}프레임)")

# =====================================================================
# 4. 대표 4연도 패널(공통 컬러바) : 절대값(4)·편차(4b)를 같은 함수로 렌더
#    - 축소 게재(인쇄폭 8~17cm) 대비 격자·컬러바 틱·n= 라벨 12pt 상향.
#    - 두 그림의 픽셀 치수를 동일하게 맞추기 위해 고정 캔버스로 저장한다
#      (bbox_inches tight 미사용, 컬러바는 명시 cax). 라벨 길이와 무관하게
#      figsize x dpi = 3540x2280 px로 두 그림이 일치한다.
#    - 격자 라벨은 왼쪽 열(위도)·아래 행(경도)에만 남겨 12pt에서 충돌을 방지.
# =====================================================================
rep_years = [int(y) for y in meta0.get("rep_years_panel", RENDER_YEARS[:4])]
if SMOKE:
    rep_years = RENDER_YEARS[:2] * 2


def render_panels(value_col, cmap, cbar_label, out_stem, vmin=None, vmax=None, norm=None):
    fig = plt.figure(figsize=(11.8, 7.6))
    fig.subplots_adjust(top=0.955, bottom=0.075, left=0.075, right=0.90,
                        hspace=0.14, wspace=0.08)
    hb = None
    for i, yr in enumerate(rep_years):
        ax = fig.add_subplot(2, 2, i + 1, projection=proj)
        make_ax(ABOVE, ax=ax, fig=fig)
        ax.set_title(f"({chr(97+i)}) {yr}년", fontsize=13)
        fix_graticule(ax, label_size=12, left_labels=(i % 2 == 0), bottom_labels=(i >= 2))
        sub = era[era["year"] == yr]
        ckw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
        hb = hexbin_map(ax, sub["lon"].values, sub["lat"].values, sub[value_col].values,
                        gridsize=PANEL_GRIDSIZE, cmap=cmap, **ckw)
        mask_ocean(ax)
        if i == 0:
            axins = add_inset_locator(fig, ax, ABOVE)
            if axins is not None:  # 위치 박스를 진청 계열로(난색 강조 금지)
                for p_ in axins.patches:
                    p_.set_edgecolor("#2b4a6f")
            add_panel_scalebar(ax, font_size=11)
        ax.text(0.975, 0.955, f"실측 n={obs_per_year.get(yr, 0):,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=12, color="0.25",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaa", alpha=0.8))
    cax = fig.add_axes([0.915, 0.17, 0.016, 0.64])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label(cbar_label, fontsize=13)
    cb.ax.tick_params(labelsize=12)
    # use_polar()의 전역 savefig.bbox='tight'를 해제해 고정 캔버스로 저장한다.
    # tight crop은 컬러바 라벨 길이에 따라 두 그림의 픽셀 치수를 다르게 만든다.
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("png", "pdf"):
            fig.savefig(FIGDIR / f"{out_stem}.{ext}", dpi=300)  # 고정 캔버스(치수 동일)
    plt.close(fig)
    print(f"[fig] {FIGDIR}/{out_stem}.png+pdf")


render_panels("pred_cm", CMAP.alt, "예측 ALT (cm)", "alt_year_panels_v2",
              vmin=vlo, vmax=vhi)
# 4b. 편차 패널: 절대값 패널과 짝을 이뤄 "연도 차이=그해 forcing 반영"을 정지 그림으로 보인다.
render_panels("pred_anom", CMAP.diff, "ALT 편차 (cm, 그해값 빼기 다년평균)",
              "alt_anom_year_panels_v1", norm=anorm)

# figs-only 종료: 프레임·GIF는 기존 산출 유지, 부트스트랩·meta 재계산 생략
if PANELS_ONLY:
    print(f"[done] 패널만 재생성  {time.time()-t0:.0f}s  (PANELS=1)")
    sys.exit(0)

# =====================================================================
# 5. 헤드라인 수치 블록 부트스트랩 CI(위치 블록, OOF 재사용)
# =====================================================================
N_BOOT = 100 if SMOKE else 1000
key_codes, _ = pd.factorize(oof["key"].values)
err = (oof["pred_STEFAN"].values - oof["alt_cm"].values).astype(float)
n_keys = key_codes.max() + 1
# 위치별 SSE·n을 집계해 두면 부트스트랩은 키 재표집 후 합산만으로 계산된다
sse_k = np.bincount(key_codes, weights=err ** 2, minlength=n_keys)
cnt_k = np.bincount(key_codes, minlength=n_keys).astype(float)

boot_rmse = np.empty(N_BOOT)
for b in range(N_BOOT):
    pick = rng.integers(0, n_keys, n_keys)
    boot_rmse[b] = np.sqrt(sse_k[pick].sum() / cnt_k[pick].sum())
rmse_ci = np.percentile(boot_rmse, [2.5, 97.5])

# within-site anomaly corr CI: 다년 위치만, 위치 블록 재표집
yc = oof.groupby("key")["year"].transform("nunique")
multi = oof[yc >= 2].copy()
multi["oa"] = multi["alt_cm"] - multi.groupby("key")["alt_cm"].transform("mean")
multi["pa"] = multi["pred_STEFAN"] - multi.groupby("key")["pred_STEFAN"].transform("mean")
mk, _ = pd.factorize(multi["key"].values)
n_mk = mk.max() + 1
oa, pa = multi["oa"].values, multi["pa"].values
S = {name: np.bincount(mk, weights=w, minlength=n_mk)
     for name, w in [("oa", oa), ("pa", pa), ("oa2", oa ** 2), ("pa2", pa ** 2),
                     ("op", oa * pa), ("n", np.ones_like(oa))]}

def corr_from_sums(idx):
    n = S["n"][idx].sum()
    so, sp = S["oa"][idx].sum(), S["pa"][idx].sum()
    so2, sp2, sop = S["oa2"][idx].sum(), S["pa2"][idx].sum(), S["op"][idx].sum()
    cov = sop / n - so * sp / n ** 2
    vo = so2 / n - (so / n) ** 2
    vp = sp2 / n - (sp / n) ** 2
    return cov / np.sqrt(vo * vp) if vo > 0 and vp > 0 else np.nan

boot_corr = np.empty(N_BOOT)
for b in range(N_BOOT):
    boot_corr[b] = corr_from_sums(rng.integers(0, n_mk, n_mk))
corr_ci = np.nanpercentile(boot_corr, [2.5, 97.5])
anom_corr_pt = corr_from_sums(np.arange(n_mk))
print(f"[boot] holdout RMSE 95% CI [{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}] cm  "
      f"anomaly corr {anom_corr_pt:+.3f} 95% CI [{corr_ci[0]:+.3f}, {corr_ci[1]:+.3f}]")

# =====================================================================
# 6. s9 meta 저장: 수치 + 시간정합 문서화 + 정직 각주
# =====================================================================
meta = {
    "stage": "S9",
    "date": "2026-07-27",
    "smoke": SMOKE,
    "period": f"{YEARS[0]}-{YEARS[-1]}",
    "data": {
        "forcing": "data/processed/alt_era5_temporal.csv (연도별 ERA5-Land, 197,648 위치 x 15년)",
        "labels": "alt_above_pointlevel.csv 유래 (위치,연도) 패널 99,931행/95,793위치 "
                  "(timelapse_alaska.py, 2026-07-20)",
        "reused_results": ["data/processed/timelapse_alaska_results.csv",
                           "data/processed/timelapse_alaska_meta.json",
                           "data/processed/timelapse_alaska_oof.csv"],
    },
    "headline": {
        "model": "STEFAN (a + E·√TDD, fold-safe E)",
        "temporal_holdout_rmse_cm": meta0["temporal_holdout"]["STEFAN"]["rmse_cm"],
        "temporal_holdout_rmse_ci95_cm": [round(float(x), 2) for x in rmse_ci],
        "temporal_holdout_r2": meta0["temporal_holdout"]["STEFAN"]["r2"],
        "temporal_holdout_skill": meta0["temporal_holdout"]["STEFAN"]["skill"],
        "anomaly_corr": round(float(anom_corr_pt), 4),
        "anomaly_corr_ci95": [round(float(x), 3) for x in corr_ci],
        "anomaly_skill": meta0["within_site_anomaly"]["STEFAN"]["skill"],
        "trend_alt_cm_per_yr": meta0["trend_alt_slope_cm_per_yr"],
        "trend_alt_r2": meta0["trend_alt_r2"],
        "trend_tdd_per_yr": meta0["trend_tdd_slope_per_yr"],
        "bootstrap": f"위치 블록 부트스트랩 {N_BOOT}회 (seed 0)",
    },
    "temporal_alignment": (
        "기후 시간 불일치 해결: 학습·평가 모두 연도별 ERA5-Land forcing(e5t_*, 그해 값)과 "
        "연도별 ALT 라벨을 (위치 key, year)로 정확 병합한 시간 정합 설계이다. "
        "leave-one-year-out으로 각 연도의 (a,E)를 그 연도 제외 데이터에서 적합했고, "
        "본 렌더의 dense 예측도 같은 fold-safe (a,E)를 사용한다. "
        "정적 지도 산출물(fidelity_base.csv의 alt_cm=다년평균 x 다년평균 기후)과 구분되는 "
        "연별 제품이다."
    ),
    "honest_caveats": [
        "within-site 연 anomaly는 예측 불가: corr 0.06(95% CI 병기), anomaly skill -0.003. "
        "연도 간 지도 차이는 그해 forcing의 반영일 뿐 위치별 연 변동 예측력을 의미하지 않는다.",
        "temporal holdout R² 0.338의 대부분은 공간 신호이다(시간 분산 비율 0.40, "
        "within-site corr(ΔTDD,ΔALT)=0.07).",
        "관측 연평균 추세 +1.29 cm/yr는 R² 0.07로 약하고 연도별 관측 위치 구성 변화의 "
        "영향을 받는다(2014년 64,114행 등 표본 불균형).",
        "프레임의 dense 커버리지는 forcing 가용 위치 기준이며 실측 라벨 수는 연 21~64,114개로 "
        "불균등하다(각 프레임에 n 표기).",
    ],
    "render": {
        "extent": list(ABOVE),
        "projection": "AlbersEqualArea",
        "cmap": "cmcrameri oslo_r (CMAP.alt 절단)",
        "fixed_clim_cm": [vlo, vhi],
        "hexbin_gridsize": GRIDSIZE,
        "mask_ocean": True,
        "graticule": ("경도 눈금 160·150·140°W 명시(본토 범위 168~141°W 정합). "
                      "자동 격자가 120°W 하나만 라벨하던 문제를 고정 locator로 교체."),
        "frames": [str(p.relative_to(REPO)) if not SMOKE else str(p) for p in frame_paths],
        "gif": str(gif_path.relative_to(REPO)) if not SMOKE else str(gif_path),
        "gif_palette": "PIL median-cut 256색 공통 팔레트 + Floyd-Steinberg dither",
        "frame_duration_ms": 800,
        "frame_duration_note": ("프레임당 800 ms. 단, 마지막 프레임은 1600 ms 홀드(정지 시각 확보)로 "
                                "frame_duration_ms=800의 예외이다."),
        "anomaly": {
            "note": ("연도-다년평균 ALT 편차 렌더(broc 0중심, 대칭 색축). 절대 ALT의 연간 변화가 "
                     "평균 0.7%로 절대값 지도에서 비가시라, forcing 연변동을 편차로 가시화한 추가 산출물이다. "
                     "절대값 v2 산출물은 그대로 유지."),
            "cmap": "cmcrameri broc (CMAP.diff, 0중심)",
            "sym_clim_cm": [-amax, amax],
            "frames": [str(p.relative_to(REPO)) if not SMOKE else str(p) for p in anom_paths],
            "gif": str(anom_gif_path.relative_to(REPO)) if not SMOKE else str(anom_gif_path),
            "frame_duration_ms": 800,
            "frame_duration_note": "프레임당 800 ms, 마지막 프레임 1600 ms 홀드(예외).",
        },
    },
    "ci_note": ("headline의 temporal_holdout_rmse_ci95_cm 및 anomaly_corr_ci95는 위치 블록 부트스트랩이다: "
                "관측 위치 key를 재표집 단위로 삼아 위치 내 상관을 보존한다. 0.5° 공간 블록이 아니라 "
                "개별 관측 위치(key) 단위 블록이다."),
    "elapsed_s": round(time.time() - t0, 1),
}
with open(META_OUT, "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[meta] {META_OUT}")
print(f"[done] {time.time()-t0:.0f}s  (SMOKE={SMOKE})")
