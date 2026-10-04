"""XE 점 규모 공변량의 계산 함수(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.5절, 개정 1, T0 = git 2678100).

배열·시계열만 받는 순수 함수다. 파일 입출력과 명령행은 xe_point_covariates.py 에 있다. 라벨 값은 받지 않는다.

래스터 창 규약(북쪽 위 격자, 변환 a > 0, e < 0)
  점 화소 (r, c) = (floor((f − y)/|e|), floor((x − c0)/a)). k × k 창은 점 화소를 가운데 두고, 격자 밖 칸은 결측으로 둔다.
  창 평균은 유한 칸의 평균이다(모두 결측이면 NaN). 등급 비율의 분모는 자료가 있는 칸 수다(WorldCover 0 = 자료 없음).
지형(x25 의 terrain_features_dem.py 와 같은 규약)
  경사 = degrees(arctan(|∇z|)), ∇z 는 np.gradient(z, dy, dx)(중앙 차분). TPI = z(점) − 창 평균(가운데 포함). 거칠기 = 창의 표준편차.
  TWI = ln(a / max(tanβ, 1e-3)), a = (누적 칸 수 + 1)·칸 면적/등고선 폭(칸 변 평균), 누적은 pysheds D8(함몰 채움, 평탄 해소 뒤).
  곡률(M 군) = 라플라시안 ∂²z/∂x² + ∂²z/∂y²(1/m).
MODIS(계획 2.5 표와 '2단계 취득의 마감과 대체 경로' 표)
  MOD13Q1: 합성 시작일(AppEEARS Date)이 2015–2020 의 6월 1일–8월 31일인 16일 합성, pixel_reliability ∈ {0, 1}.
    ndvi250_jja·evi250_jja·ndwi250_jja = 그 합성 값의 평균(NDWI = (NIR − MIR)/(NIR + MIR)),
    ndvi250_max = 합성 시작 일차별 해 평균(기후값) 가운데 최댓값(H19 ndvi_max 의 월 기후값 최댓값과 같은 구성).
  MOD10A2(대체 경로, 등록 정의): scd500 = 수문년 안 적설 표시 합성 수 × 8, snowoff·snowon = 첫·마지막 무적설 전이 합성의 시작일.
    적설 표시 = Maximum_Snow_Extent 200, 무적설 = 25, 그 밖의 부호는 판정 없음(건너뛴다).
  MOD10A1(주 경로): 일 NDSI_Snow_Cover 10–100 = 적설, 0–9 = 무적설, 그 밖은 판정 없음. 판정 없는 날은 시간상 가장 가까운 판정 날의 상태로
    채운다(같은 거리면 앞 날). scd500 = 수문년 안 적설 일수, snowoff·snowon 은 MOD10A2 와 같은 전이 규칙을 일 단위로 적용한다.
  전이 규칙: 판정 있는 칸만 시간 순으로 보아 적설 → 무적설로 바뀐 칸이 소멸 전이, 무적설 → 적설로 바뀐 칸이 시작 전이다. 달력 연도 Y 의
    소멸일 = 그해 첫 소멸 전이 칸의 시작 일차, 시작일 = 그해 마지막 시작 전이 칸의 시작 일차. 수문년은 9월–8월이고 끝 해로 이름 붙인다
    (2015 수문년 = 2014-09-01 – 2015-08-31). 값은 2015–2020 수문년(적설 일수)과 2015–2020 달력 연도(일차)의 평균이다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

M_PER_DEG = 111320.0
SNOW_HY = tuple(range(2015, 2021))                                          # 수문년(끝 해 표기)
SNOW_CY = tuple(range(2015, 2021))                                          # 달력 연도(일차)
VEG_YEARS = tuple(range(2015, 2021))
VEG_MONTHS = (6, 7, 8)
MOD10A2_SNOW, MOD10A2_NOSNOW = 200, 25
MOD10A1_SNOW_MIN = 10                                                       # NDSI 0.1(C6 적설 판정 하한, MOD10A2 의 적설 판정과 같은 기준)
MOD13Q1_REL_OK = (0, 1)                                                     # 좋음, 보통


# ================================================================ 래스터 창
def pixel_rc(transform, x, y):
    """점 좌표 → (행, 열) 정수 배열. transform = affine(a, b, c, d, e, f) 또는 (a, c, e, f) 의 북쪽 위 격자."""
    a, c0, e, f = (transform.a, transform.c, transform.e, transform.f) if hasattr(transform, "a") else transform
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = np.floor((f - y) / abs(e)).astype(np.int64)
    c = np.floor((x - c0) / a).astype(np.int64)
    return r, c


def window(arr, r, c, half):
    """(2·half + 1)² 창. 격자 밖 칸은 NaN 이다."""
    arr = np.asarray(arr)
    H, W = arr.shape
    k = 2 * int(half) + 1
    out = np.full((k, k), np.nan)
    r0, c0 = int(r) - int(half), int(c) - int(half)
    rr0, rr1 = max(0, r0), min(H, r0 + k)
    cc0, cc1 = max(0, c0), min(W, c0 + k)
    if rr0 < rr1 and cc0 < cc1:
        out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = arr[rr0:rr1, cc0:cc1]
    return out


def px_value(arr, r, c):
    H, W = np.asarray(arr).shape
    return float(arr[r, c]) if (0 <= r < H and 0 <= c < W) else float("nan")


def win_mean(arr, r, c, half):
    w = window(arr, r, c, half)
    return float(np.nanmean(w)) if np.isfinite(w).any() else float("nan")


def win_std(arr, r, c, half):
    w = window(arr, r, c, half)
    return float(np.nanstd(w)) if np.isfinite(w).any() else float("nan")


def tpi_at(z, r, c, half):
    """TPI = z(점 화소) − 창 평균(가운데 포함)."""
    zc = px_value(z, r, c)
    m = win_mean(z, r, c, half)
    return zc - m if np.isfinite(zc) and np.isfinite(m) else float("nan")


def class_fractions(arr, r, c, half, codes, nodata=0):
    """창 안의 등급 비율(분모 = 자료가 있는 칸). codes = {열 이름: 등급 부호}. 자료가 있는 칸이 없으면 모두 NaN."""
    w = window(arr, r, c, half)
    ok = np.isfinite(w) & (w != nodata)
    n = int(ok.sum())
    if n == 0:
        return {k: float("nan") for k in codes}
    return {k: float(np.sum(w[ok] == v)) / n for k, v in codes.items()}


def class_onehot(arr, r, c, codes, nodata=0):
    """점 화소의 등급 원핫(민감도 열). 자료가 없으면 NaN."""
    v = px_value(arr, r, c)
    if not np.isfinite(v) or v == nodata:
        return {k: float("nan") for k in codes}
    return {k: float(v == code) for k, code in codes.items()}


def geo_spacing(transform, lat_c):
    """위경도 격자 칸의 (dy, dx) 미터 근사(창 가운데 위도의 cos 로 경도 간격을 줄인다. 기존 build_covariates_ext 와 같다)."""
    a, e = (transform.a, transform.e) if hasattr(transform, "a") else (transform[0], transform[2])
    return abs(e) * M_PER_DEG, abs(a) * M_PER_DEG * math.cos(math.radians(float(lat_c)))


def slope_deg_grid(z, dy, dx):
    gy, gx = np.gradient(np.asarray(z, float), dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def tan_slope_grid(z, dy, dx):
    gy, gx = np.gradient(np.asarray(z, float), dy, dx)
    return np.hypot(gx, gy)


def laplacian_grid(z, dy, dx):
    z = np.asarray(z, float)
    gy, gx = np.gradient(z, dy, dx)
    return np.gradient(gx, dx, axis=1) + np.gradient(gy, dy, axis=0)


def twi_from_acc(acc, tanb, dy, dx):
    """TWI = ln(a / max(tanβ, 1e-3)), a = (acc + 1)·dx·dy / ((dx + dy)/2)."""
    a_spec = (np.asarray(acc, float) + 1.0) * (dx * dy) / (0.5 * (dx + dy))
    return np.log(a_spec / np.maximum(np.asarray(tanb, float), 1e-3))


def d8_accumulation(z, transform, crs="epsg:4326"):
    """pysheds D8 누적(함몰 채움, 평탄 해소 뒤). 결측은 창 중앙값으로 채운다(기존 build_covariates_ext.terrain_tile 과 같다). 칸 수."""
    import pyproj
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder
    z = np.asarray(z, float)
    zf = np.where(np.isfinite(z), z, np.nanmedian(z))
    vf = ViewFinder(affine=transform, shape=zf.shape, crs=pyproj.Proj(crs), nodata=np.float64(np.nan))
    grid = Grid(viewfinder=vf)
    dem = Raster(zf, viewfinder=vf)
    pit = grid.fill_pits(dem)
    dep = grid.fill_depressions(pit)
    infl = grid.resolve_flats(dep)
    fdir = grid.flowdir(infl)
    return np.asarray(grid.accumulation(fdir), float)


def terrain_points(z, transform, rows, cols, dy, dx, acc=None):
    """T2 군(DEM 부분)과 점 화소 민감도 열: twi_pt, tpi_90(3 × 3), tpi_270(9 × 9), slope_90(3 × 3 경사 평균), slope_px."""
    z = np.asarray(z, float)
    tanb = tan_slope_grid(z, dy, dx)
    slope = np.degrees(np.arctan(tanb))
    twi = twi_from_acc(acc, tanb, dy, dx) if acc is not None else None
    out = {k: np.full(len(rows), np.nan) for k in ("twi_pt", "tpi_90", "tpi_270", "slope_90", "slope_px")}
    for i, (r, c) in enumerate(zip(rows, cols)):
        if not np.isfinite(px_value(z, r, c)):
            continue
        out["twi_pt"][i] = px_value(twi, r, c) if twi is not None else np.nan
        out["tpi_90"][i] = tpi_at(z, r, c, 1)
        out["tpi_270"][i] = tpi_at(z, r, c, 4)
        out["slope_90"][i] = win_mean(slope, r, c, 1)
        out["slope_px"][i] = px_value(slope, r, c)
    return out


def half_for(length_m, res_m):
    """미터 지원 규모 → 창 반폭(화소). 10 m 에서 30 → 1(3 × 3), 50 → 2(5 × 5), 150 → 7(15 × 15). 최소 1."""
    return max(1, int(round((float(length_m) / float(res_m) - 1.0) / 2.0)))


def arcticdem_points(z, res, rows, cols):
    """M 군: ad_slope_30(30 m 창 경사 평균), ad_curv_30(30 m 창 라플라시안 평균, 1/m), ad_tpi_50, ad_tpi_150, ad_rough_50(50 m 창 표준편차),
    민감도 ad_slope_px, ad_curv_px. 투영 좌표(EPSG:3413, 미터)라 dy = dx = res."""
    z = np.asarray(z, float)
    slope = slope_deg_grid(z, res, res)
    lap = laplacian_grid(z, res, res)
    h30, h50, h150 = half_for(30, res), half_for(50, res), half_for(150, res)
    keys = ("ad_slope_30", "ad_curv_30", "ad_tpi_50", "ad_tpi_150", "ad_rough_50", "ad_slope_px", "ad_curv_px")
    out = {k: np.full(len(rows), np.nan) for k in keys}
    for i, (r, c) in enumerate(zip(rows, cols)):
        if not np.isfinite(px_value(z, r, c)):
            continue
        out["ad_slope_30"][i] = win_mean(slope, r, c, h30)
        out["ad_curv_30"][i] = win_mean(lap, r, c, h30)
        out["ad_tpi_50"][i] = tpi_at(z, r, c, h50)
        out["ad_tpi_150"][i] = tpi_at(z, r, c, h150)
        out["ad_rough_50"][i] = win_std(z, r, c, h50)
        out["ad_slope_px"][i] = px_value(slope, r, c)
        out["ad_curv_px"][i] = px_value(lap, r, c)
    return out


# ================================================================ MODIS
def hydro_year(dates):
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.where(d.month >= 9, d.year + 1, d.year)


def mod10a2_state(codes):
    """Maximum_Snow_Extent → 1(적설, 200), 0(무적설, 25), NaN(그 밖: 0 자료 없음, 1 판정 없음, 11 밤, 37 호수, 39 바다, 50 구름, 100 호수 얼음,
    254 포화, 255 채움)."""
    v = np.asarray(codes, float)
    return np.where(v == MOD10A2_SNOW, 1.0, np.where(v == MOD10A2_NOSNOW, 0.0, np.nan))


def mod10a1_state(ndsi):
    """NDSI_Snow_Cover → 1(10–100), 0(0–9), NaN(200 이상의 부호: 결측, 판정 없음, 밤, 내륙 수면, 바다, 구름, 포화, 채움)."""
    v = np.asarray(ndsi, float)
    return np.where((v >= MOD10A1_SNOW_MIN) & (v <= 100), 1.0, np.where((v >= 0) & (v < MOD10A1_SNOW_MIN), 0.0, np.nan))


def fill_nearest(states):
    """판정 없는 칸을 시간상 가장 가까운 판정 칸의 상태로 채운다(같은 거리면 앞 칸). 판정 칸이 없으면 그대로."""
    s = np.asarray(states, float).copy()
    ok = np.isfinite(s)
    if not ok.any() or ok.all():
        return s
    idx = np.arange(len(s))
    prev = np.where(ok, idx, -1)
    prev = np.maximum.accumulate(prev)
    nxt = np.where(ok, idx, len(s))
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    for i in np.where(~ok)[0]:
        p, q = prev[i], nxt[i]
        if p < 0:
            s[i] = s[q]
        elif q >= len(s):
            s[i] = s[p]
        else:
            s[i] = s[p] if (i - p) <= (q - i) else s[q]
    return s


def snow_metrics(dates, states, step_days, hy_years=SNOW_HY, cy_years=SNOW_CY):
    """적설 일수(수문년), 소멸일·시작일(달력 연도) 의 해 평균. states: 1 적설, 0 무적설, NaN 판정 없음. 칸의 날짜는 합성(또는 일)의 시작일.
    반환 dict(scd500, snowoff_doy500, snowon_doy500, n_hy, n_off, n_on, by_year)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    s = np.asarray(states, float)
    o = np.argsort(d.values, kind="stable")
    d, s = d[o], s[o]
    hy = np.where(d.month >= 9, d.year + 1, d.year)
    scd = {}
    for Y in hy_years:
        v = s[hy == Y]
        v = v[np.isfinite(v)]
        scd[Y] = float(step_days * np.sum(v == 1.0)) if len(v) else float("nan")
    ok = np.isfinite(s)
    dv, sv = d[ok], s[ok]
    off, on = {}, {}
    for i in range(1, len(sv)):
        if sv[i] == sv[i - 1]:
            continue
        Y, doy = int(dv[i].year), int(dv[i].dayofyear)
        if sv[i - 1] == 1.0 and sv[i] == 0.0 and Y not in off:
            off[Y] = doy                                                    # 그해 첫 소멸 전이
        elif sv[i - 1] == 0.0 and sv[i] == 1.0:
            on[Y] = doy                                                     # 그해 마지막 시작 전이(덮어쓴다)
    sc = [scd[Y] for Y in hy_years if np.isfinite(scd[Y])]
    fo = [off[Y] for Y in cy_years if Y in off]
    fn = [on[Y] for Y in cy_years if Y in on]
    return dict(scd500=float(np.mean(sc)) if sc else float("nan"), snowoff_doy500=float(np.mean(fo)) if fo else float("nan"),
                snowon_doy500=float(np.mean(fn)) if fn else float("nan"), n_hy=len(sc), n_off=len(fo), n_on=len(fn),
                by_year=dict(scd={int(k): v for k, v in scd.items()}, off={int(k): v for k, v in off.items()}, on={int(k): v for k, v in on.items()}))


def mod10a2_metrics(dates, codes):
    """대체 경로(MOD10A2, 등록 정의): 적설 표시 합성 수 × 8, 첫·마지막 무적설 전이 합성의 시작일."""
    return snow_metrics(dates, mod10a2_state(codes), 8)


def mod10a1_metrics(dates, ndsi):
    """주 경로(MOD10A1): 일 상태를 가장 가까운 판정 날로 채운 뒤 적설 일수와 전이 일차."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    o = np.argsort(d.values, kind="stable")
    st = fill_nearest(mod10a1_state(np.asarray(ndsi, float)[o]))
    return snow_metrics(d[o], st, 1)


def mod13q1_metrics(dates, ndvi, evi, nir, mir, reliability, years=VEG_YEARS, months=VEG_MONTHS):
    """V 군의 MOD13Q1 4열. 입력은 AppEEARS 점 결과의 척도 적용 값(NDVI 등은 −0.2–1, 반사도 0–1)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    rel = np.asarray(reliability, float)
    keep = np.isin(d.year, years) & np.isin(d.month, months) & np.isin(rel, MOD13Q1_REL_OK)
    nd, ev = np.asarray(ndvi, float), np.asarray(evi, float)
    ni, mi = np.asarray(nir, float), np.asarray(mir, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        nw = np.where((ni + mi) > 0, (ni - mi) / (ni + mi), np.nan)
    keep &= np.isfinite(nd)
    out = dict(ndvi250_jja=float("nan"), evi250_jja=float("nan"), ndwi250_jja=float("nan"), ndvi250_max=float("nan"), n_comp=int(keep.sum()))
    if not keep.any():
        return out
    out["ndvi250_jja"] = float(np.nanmean(nd[keep]))
    out["evi250_jja"] = float(np.nanmean(ev[keep])) if np.isfinite(ev[keep]).any() else float("nan")
    out["ndwi250_jja"] = float(np.nanmean(nw[keep])) if np.isfinite(nw[keep]).any() else float("nan")
    clim = pd.Series(nd[keep]).groupby(np.asarray(d.dayofyear)[keep]).mean()
    out["ndvi250_max"] = float(clim.max())
    return out
