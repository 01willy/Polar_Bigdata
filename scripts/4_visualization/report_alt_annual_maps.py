"""연별 ALT 지도(조밀 격자). 연도 홀드아웃에서 채택된 2모수 회귀에 그해 ERA5-Land
기후를 대입해 연속장을 산출한다.

기존 그림은 관측 지점에만 예측을 얹어 성글고 연도 차이가 보이지 않았다. 기후 구동장은
격자 전역에서 계산되므로 조밀 격자로 그리면 그해 기후의 공간 구조가 드러난다.
아래 행에는 기간 평균 대비 편차를 그려 연도 간 신호를 직접 보인다.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.path as mpath
import cartopy.crs as ccrs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.plotstyle import use_polar                      # noqa: E402
from polar.fidelity import macro_region, fit_stefan_E      # noqa: E402
from polar.geomap import make_ax, mask_ocean, map_projection   # noqa: E402
import pandas as pd                                        # noqa: E402
from cmcrameri import cm as cmc                            # noqa: E402

use_polar()
# 축 사각형을 인치로 직접 배치하므로 tight 크롭을 끈다(최종 인쇄 크기를 확정하기 위함).
plt.rcParams["savefig.bbox"] = None
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2010-2024.nc"
OUTD = ROOT / "outputs" / "figures" / "s9_timelapse"
EXT = ("alaska", -168.5, -138.0, 58.5, 71.5)   # (이름, 경도, 위도) 창
YEARS = [2013, 2016, 2019, 2022]
RES = 0.05
GREY = "#5b6670"

# ---------------------------------------------------------------- 모델
# 연별 실험 재설계(scripts/3_deep_learning/s14_annual_alt_redesign.py) 결과, 연도 홀드아웃
# 오차는 물리 잔차 결합 14.06 < CatBoost 14.24 < 물리식 단독 14.97 cm 다. 학습 모형 둘이
# 물리식을 앞선다. 지도에는 이 가운데 CatBoost 를 쓴다. 능형 잔차는 격자에서 학습 분포를
# 크게 벗어난 조합에 선형 외삽해 -1,200 cm 급 값을 내지만(진단 확인), 트리 모형은 학습
# 라벨 범위 밖으로 외삽하지 않기 때문이다. 4.1절 광역 지도(그림 5)와 같은 근거다.
# 입력은 격자에서 전부 채울 수 있는 그해 기후 9종이다.
YEAR9 = ["e5t_maat", "e5t_tdd", "e5t_fdd", "e5t_sqrt_tdd", "e5t_twarm",
         "e5t_tcold", "e5t_stl1", "e5t_swe", "e5t_swe_prevwinter"]
FEATS = YEAR9

_pt = pd.read_csv(ROOT / "data" / "processed" / "alt_above_pointlevel.csv")
_pt = _pt.dropna(subset=["lat", "lon", "year", "alt_cm"]).copy()
_pt["year"] = _pt["year"].astype(int)
_pt["key"] = _pt["lat"].round(5).astype(str) + "_" + _pt["lon"].round(5).astype(str)
_panel = (_pt.groupby(["key", "year"])
          .agg(alt_cm=("alt_cm", "mean"), lat=("lat", "mean"), lon=("lon", "mean"))
          .reset_index())
_era = pd.read_csv(ROOT / "data" / "processed" / "alt_era5_temporal.csv")
_era["year"] = _era["year"].astype(int)
_era["key"] = _era["lat"].round(5).astype(str) + "_" + _era["lon"].round(5).astype(str)
_era = _era.drop_duplicates(["key", "year"])[["key", "year"] + YEAR9]
_pan = _panel.merge(_era, on=["key", "year"], how="inner")

_pan = _pan.dropna(subset=["alt_cm", "e5t_sqrt_tdd"]).reset_index(drop=True)
_y = _pan["alt_cm"].to_numpy(float)
_X = _pan[FEATS].to_numpy("float32")

sys.path.insert(0, str(ROOT / "src"))
from polar.tab_models import fit_predict                       # noqa: E402
print(f"CatBoost(그해 기후 {len(FEATS)}종) · 패널 {len(_pan):,}행 · "
      f"{_pan['key'].nunique():,}위치 전량 적합 · seed 3회 앙상블")

# ---------------------------------------------------------------- 연별 격자 공변량
ds = xr.open_dataset(NC)
lat_name = "latitude" if "latitude" in ds.coords else "lat"
lon_name = "longitude" if "longitude" in ds.coords else "lon"
tname = "valid_time" if "valid_time" in ds.coords else "time"
lon = ds[lon_name].values
lon180 = np.where(lon > 180, lon - 360, lon)
sel_lon = (lon180 >= EXT[1] - 1) & (lon180 <= EXT[2] + 1)
sel_lat = (ds[lat_name].values >= EXT[3] - 1) & (ds[lat_name].values <= EXT[4] + 1)
sub = ds.isel({lon_name: np.where(sel_lon)[0], lat_name: np.where(sel_lat)[0]}).load()

ALL_Y = list(range(2010, 2025))
_days = np.array([pd.Period(f"2019-{m:02d}", freq="M").days_in_month
                  for m in range(1, 13)], dtype="float64")


def _cube(var, off=0.0):
    a = sub[var].values.astype("float64") - off      # (180, lat, lon)
    return a.reshape(len(ALL_Y), 12, a.shape[1], a.shape[2])


_t2m, _stl, _sd = _cube("t2m", 273.15), _cube("stl1", 273.15), _cube("sd")
glat0 = sub[lat_name].values
glon0 = np.sort(lon180[sel_lon])
order = np.argsort(lon180[sel_lon])
flip = glat0[0] > glat0[-1]
if flip:
    glat0 = glat0[::-1]
glat = np.arange(EXT[3], EXT[4] + RES, RES)
glon = np.arange(EXT[1], EXT[2] + RES, RES)
NY, NX = len(glat), len(glon)


def _fine(a2d):
    """0.1도 원장을 목표 격자로 이중선형 세분(경도 정렬·위도 방향 통일 포함)."""
    a = a2d[:, order]
    if flip:
        a = a[::-1]
    return (xr.DataArray(a, coords={"lat": glat0, "lon": glon0}, dims=["lat", "lon"])
            .interp(lat=glat, lon=glon, method="linear").values)


def year_climate(year):
    """그해 기후 9종을 목표 격자에서 산출(alt_era5_temporal.csv 와 동일 정의)."""
    yi = ALL_Y.index(year)
    tm, gm, sm = _t2m[yi], _stl[yi], _sd[yi]
    tdd = np.nansum(np.clip(tm, 0, None) * _days[:, None, None], axis=0)
    fdd = np.nansum(np.clip(-tm, 0, None) * _days[:, None, None], axis=0)
    prev = (np.concatenate([_sd[yi - 1][9:12], _sd[yi][0:4]], axis=0) if yi > 0
            else _sd[yi][0:4])
    raw = {"e5t_maat": np.nanmean(tm, axis=0), "e5t_tdd": tdd, "e5t_fdd": fdd,
           "e5t_sqrt_tdd": np.sqrt(np.clip(tdd, 0, None)),
           "e5t_twarm": np.nanmax(tm, axis=0), "e5t_tcold": np.nanmin(tm, axis=0),
           "e5t_stl1": np.nanmean(gm, axis=0), "e5t_swe": np.nanmean(sm, axis=0),
           "e5t_swe_prevwinter": np.nanmean(prev, axis=0)}
    return {k: _fine(v) for k, v in raw.items()}


print(f"[격자] {NX} x {NY} = {NX*NY:,} 셀")


def field(year):
    """그해 기후 9종을 격자에 채우고 CatBoost 3-seed 앙상블로 예측한다."""
    G = year_climate(year)
    Xg = np.column_stack([G[c].ravel() for c in FEATS]).astype("float32")
    p = np.mean([np.expm1(fit_predict("catboost", _X, np.log1p(_y), Xg, seed=sd)["pred"])
                 for sd in (0, 1, 2)], axis=0)
    return np.clip(p, 1.0, 400.0).reshape(NY, NX)


alts = {y: field(y) for y in YEARS}
base = np.mean([field(y) for y in ALL_Y], axis=0)
print(f"격자 {NY}x{NX} · 기간평균 ALT {np.nanmean(base):.1f} cm")

maat_all = np.nanmean([_fine(np.nanmean(_t2m[k], axis=0)) for k in range(len(ALL_Y))], axis=0)

# 영구동토 마스크: 기간평균 연평균기온 < 0 인 곳만
pf = maat_all < 0.0
for y in YEARS:
    alts[y] = np.where(pf, alts[y], np.nan)
base = np.where(pf, base, np.nan)

# ---------------------------------------------------------------- 색 단계
vals = np.concatenate([alts[y][np.isfinite(alts[y])] for y in YEARS])
vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
lv = np.arange(np.floor(vmin / 5) * 5, np.ceil(vmax / 5) * 5 + 5, 5)
amax = max(abs(np.nanpercentile(alts[y] - base, [2, 98])).max() for y in YEARS)
amax = np.ceil(amax / 2) * 2                    # 눈금이 정수가 되도록 반올림
lva = np.arange(-amax, amax + 1e-9, amax / 5)   # 0 중심 대칭 단계(발산형)


def _cbar_ticks(levels, nmax=5):
    """단계가 촘촘한 컬러바에서 눈금만 솎아낸다(28 mm 높이에서 라벨 겹침 방지)."""
    k = int(np.ceil((len(levels) - 1) / nmax))
    return levels[::k]


def _seamless(cs):
    """등치면 사이 흰 이음새 제거. 래스터화는 축의 rasterization_zorder가 담당한다
    (cartopy GeoContourSet은 artist 단위 set_rasterized를 무시한다)."""
    try:
        cs.set_edgecolor("face")
    except AttributeError:                       # mpl < 3.8
        for c in cs.collections:
            c.set_edgecolor("face")


def _domain_path(ext, n=240):
    """계산 격자 도메인(경위도 사각형)의 경계 경로. 정적도법에서는 사다리꼴이 된다."""
    _, lo0, lo1, la0, la1 = ext
    la = np.linspace(la0, la1, n)
    lo = np.linspace(lo0, lo1, n)
    lons = np.concatenate([lo, np.full(n, lo1), lo[::-1], np.full(n, lo0)])
    lats = np.concatenate([np.full(n, la0), la, np.full(n, la1), la[::-1]])
    return mpath.Path(np.column_stack([lons, lats]), closed=True)


def domain_box(ext):
    """도메인 경계를 투영한 외접 사각형 (x0, x1, y0, y1)과 가로/세로 비."""
    p = _domain_path(ext)
    xy = map_projection(ext).transform_points(ccrs.PlateCarree(),
                                              p.vertices[:, 0], p.vertices[:, 1])
    x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
    y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
    return (x0, x1, y0, y1), (x1 - x0) / (y1 - y0)


def fit_domain(ax, ext):
    """축 범위를 도메인 외접 사각형에, 축 경계를 도메인 자체에 맞춘다.

    cartopy 기본 set_extent는 경위도 창의 외접 사각형을 쓰므로 (a) 남쪽 가장자리가
    잘리고 (b) 도메인 밖 모서리에 값 없는 배경이 남아 직선 절단처럼 보인다.
    """
    (x0, x1, y0, y1), _ = domain_box(ext)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_boundary(_domain_path(ext), transform=ccrs.PlateCarree())
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_edgecolor("#8e8e8e")


# ---------------------------------------------------------------- 레이아웃
# 지도 투영 후 종횡비를 먼저 구해 축 사각형을 인치로 직접 배치한다.
# 그래야 8개 패널 크기가 정확히 같고, 컬러바 높이가 지도 높이와 일치한다.
# 폭은 보고서 삽입 크기(0.94 x textwidth = 165.44 mm)와 같게 잡아 1:1로 렌더한다.
FIG_W = 165.44 / 25.4                      # in
M_L, M_R = 0.21, 0.03                      # 좌(행 라벨)·우 여백
M_T, M_B = 0.20, 0.30                      # 상(연도 제목)·하(주석) 여백
COL_GAP, ROW_GAP = 0.045, 0.075            # 패널 간격
CB_GAP, CB_W, CB_LAB = 0.055, 0.085, 0.40  # 컬러바 블록(간격·폭·라벨 영역)

_, ASP = domain_box(EXT)                   # 투영 후 지도 가로/세로
NC_ = len(YEARS)
CB_BLOCK = CB_GAP + CB_W + CB_LAB
MAP_W = (FIG_W - M_L - M_R - (NC_ - 1) * COL_GAP - CB_BLOCK) / NC_
MAP_H = MAP_W / ASP
FIG_H = M_T + 2 * MAP_H + ROW_GAP + M_B
Y_ROW = [M_B + MAP_H + ROW_GAP, M_B]       # 위 행(절대값), 아래 행(편차)
X_CB = M_L + NC_ * MAP_W + (NC_ - 1) * COL_GAP + CB_GAP


def fr(x, y, w, h):
    """인치 → figure 좌표."""
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


print(f"패널 종횡비 {ASP:.3f} · 지도 {MAP_W*25.4:.1f} x {MAP_H*25.4:.1f} mm · "
      f"전체 {FIG_W*25.4:.1f} x {FIG_H*25.4:.1f} mm")

# ---------------------------------------------------------------- 그림
proj = map_projection(EXT)
fig = plt.figure(figsize=(FIG_W, FIG_H))
rows = [
    dict(data=lambda y: alts[y], levels=lv, cmap=cmc.oslo_r, ticks=_cbar_ticks(lv),
         label="ALT (cm)", rowlab="예측 ALT"),
    # 편차는 0 중심 대칭 단계 → 중립색이 편차 0에 정렬. 눈금도 0을 포함해 대칭으로.
    dict(data=lambda y: alts[y] - base, levels=lva, cmap=cmc.broc_r,
         ticks=np.array([-amax, -amax / 2, 0.0, amax / 2, amax]),
         label="편차 (cm)", rowlab="기간평균 대비 편차"),
]
for i, r in enumerate(rows):
    for j, y in enumerate(YEARS):
        ax = fig.add_axes(fr(M_L + j * (MAP_W + COL_GAP), Y_ROW[i], MAP_W, MAP_H),
                          projection=proj)
        make_ax(EXT, ax=ax, fig=fig, grid=False)
        fit_domain(ax, EXT)
        ax.set_rasterization_zorder(1.6)         # 배경·등치면만 래스터, 해안선은 벡터 유지
        cs = ax.contourf(glon, glat, r["data"](y), levels=r["levels"], cmap=r["cmap"],
                         extend="both", transform=ccrs.PlateCarree(), zorder=1.5)
        _seamless(cs)
        mask_ocean(ax, zorder=3)
        if i == 0:
            ax.set_title(f"{y}년", fontsize=9.5, pad=3, color="#222")
    # 행 공유 컬러바 — 높이를 지도 패널과 같게 두어 패널 폭을 빼앗지 않는다.
    cax = fig.add_axes(fr(X_CB, Y_ROW[i], CB_W, MAP_H))
    cb = fig.colorbar(cs, cax=cax)
    cb.set_ticks(r["ticks"])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
    cb.set_label(r["label"], fontsize=8.5, color=GREY, labelpad=3)
    cb.ax.tick_params(labelsize=7.6, labelcolor=GREY, length=1.8, pad=1.6)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("#666")
    fig.text((M_L - 0.075) / FIG_W, (Y_ROW[i] + MAP_H / 2) / FIG_H, r["rowlab"],
             rotation=90, va="center", ha="center", fontsize=8.0, color=GREY)

fig.text(M_L / FIG_W, (M_B - 0.135) / FIG_H,
         f"격자 {RES}° · CatBoost(그해 ERA5-Land 기후 {len(FEATS)}종 입력, 연도 홀드아웃 채택) · "
         f"기간평균은 2010–2024년 · 연평균기온 0 °C 이상 육지는 제외",
         fontsize=7.0, color="#7a7a7a", ha="left", va="center")
for ext in ("png", "pdf"):
    fig.savefig(OUTD / f"alt_annual_fields.{ext}", dpi=300)
print(f"saved: {OUTD}/alt_annual_fields.png")
