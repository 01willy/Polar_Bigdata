"""활성층 두께(ALT) 예측장의 해상도 비교 — 기후 단독 vs 30 m 지형 공변량 포함.

목적
    기존 광역 ALT 지도는 ERA5-Land 0.1°(약 11 km) 기후 강제력만을 정보원으로 쓴다.
    격자를 세분해도 새 정보가 없으므로 장은 매끄럽기만 하다. 여기서는 Copernicus DEM
    30 m에서 지형 공변량 6종을 실제로 산출해 약 250 m 격자에서 ALT를 재예측하고,
    지형이 더한 세부 구조를 같은 창에서 직접 비교한다.

방법
    1) 창: 브룩스 산맥 북사면~북극 연안평야 (경도 -152.0~-148.0°, 위도 68.2~70.0°).
       창 안 실측 셀은 3,509개이며, 0.01° 격자로 집계하면 146개 위치가 된다.
    2) 지형: Copernicus DEM 30 m 타일 8매를 모자이크해 원해상도(위도 1", 경도 2")에서
       표고·경사·사면방위(sin·cos)·TPI·거칠기를 계산한다. 계산식·창크기(33x33 px)는
       학습 공변량 생성 스크립트(scripts/1_data_prep/terrain_features_dem.py)와 동일하다.
       원해상도 화소값을 8(위도) x 11(경도) 블록평균해 목표 격자로 집계한다.
    3) 기후: ERA5-Land 월 기후값(2015-2020, 0.1°) 8종을 목표 격자로 이중선형 보간한다.
    4) 토양: SoilGrids(약 5 km, IGH) 9종을 목표 격자에 샘플링한다.
    5) 예측: 알래스카 관측 13,606셀(보고서 표본과 동일한 지역 정의)로 CatBoost(CPU)를
       학습해 목표 격자에 적용한다.
       (a) 지형 6종을 창 중앙값으로 고정 → 기후·토양만의 기여(정보량 약 11 km).
       (b) 지형 6종에 실제 격자값 사용 → 30 m 지형정보 반영.
       (c) (b) - (a) = 지형 기여.
       지도 산출용이며 일반화 성능 주장이 아니므로 별도 홀드아웃을 두지 않는다.

산출
    outputs/maps/alt_downscale_demo.png (dpi 300)
    outputs/maps/alt_downscale_demo.pdf
    보고서 삽입 폭(0.78 x textwidth = 137.3 mm)과 같은 figsize 로 생성하므로 축소 배율은
    1.0 이고, 스크립트에 지정한 pt 값이 그대로 인쇄 pt 가 된다(최소 7.0 pt).

실행
    PYTHONPATH=/home/willy010313/Polar_Bigdata/src \
        python scripts/4_visualization/report_alt_downscale_demo.py
    옵션: --agg-lat 8 --agg-lon 11 (기본, 약 247 x 244 m) / --agg-lat 16 --agg-lon 22 (약 500 m)
"""
from __future__ import annotations

import argparse
import calendar
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/willy010313/Polar_Bigdata"
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from polar.plotstyle import use_polar  # noqa: E402
from polar.geomap import make_ax, add_scalebar  # noqa: E402
from polar.outputs import mappath  # noqa: E402
from polar.fidelity import TERRAIN, CLIMATE, SOIL  # noqa: E402

plt = use_polar()
plt.rcParams["axes.unicode_minus"] = False   # 한글 폰트에 U+2212 글리프 없음
# 축 사각형을 인치로 직접 배치하므로 tight 크롭을 끈다(인쇄 폭을 figsize 로 확정).
plt.rcParams["savefig.bbox"] = None

DEM_DIR = os.path.join(ROOT, "data/raw/dem")
NC = os.path.join(ROOT, "data/raw/era5land/nh_monthly_2015-2020.nc")
SG_DIR = os.path.join(ROOT, "data/raw/soilgrids_wcs/namerica")
TRAIN_CSV = os.path.join(ROOT, "data/processed/fidelity_base.csv")

# 알래스카 지역 정의는 보고서 표본(13,606셀)과 같게 둔다. 다른 보고서 지도 스크립트
# (report_alt_map_hires.py)와 동일한 정의이며, 학습 표본이 표·본문 수치와 어긋나지 않는다.
AK_REGIONS = ("ABoVE_AK", "United States (Alaska)")
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"

# 창 안 실측 셀(3,509개)은 그대로 찍으면 서로 겹치므로 0.01° 격자로 집계해 표시한다.
# 범례의 n은 관측 지점 수가 아니라 집계 위치 수다(그림 4와 같은 표기 규칙, 간격만 다름).
DEDUP_DEG = 0.01

# 창(경도0, 경도1, 위도0, 위도1). 약 159 x 200 km. DEM 타일 8매 보유 구간.
WIN = (-152.0, -148.0, 68.2, 70.0)
WIN_NAME = "브룩스 산맥 북사면"

# Copernicus DEM(위도 60~70° 대역) 화소 간격
DLAT_PX = 1.0 / 3600.0     # 1" ≈ 30.9 m
DLON_PX = 1.0 / 1800.0     # 2" ≈ 22.3 m (위도 69°)
PATCH = 33                 # 학습 공변량과 동일한 국소창(≈1 km)

# SoilGrids 정수 스케일 → 물리단위 (enrich_soilgrids_wcs.py SCALE 과 동일)
SG_LAYERS = {
    "sg_clay_5_15": ("clay_5-15cm_mean.tif", 10.0),
    "sg_sand_5_15": ("sand_5-15cm_mean.tif", 10.0),
    "sg_silt_5_15": ("silt_5-15cm_mean.tif", 10.0),
    "sg_bdod_5_15": ("bdod_5-15cm_mean.tif", 100.0),
    "sg_cfvo_5_15": ("cfvo_5-15cm_mean.tif", 10.0),
    "sg_phh2o_5_15": ("phh2o_5-15cm_mean.tif", 10.0),
    "sg_soc_0_5": ("soc_0-5cm_mean.tif", 10.0),
    "sg_soc_5_15": ("soc_5-15cm_mean.tif", 10.0),
    "sg_soc_15_30": ("soc_15-30cm_mean.tif", 10.0),
}

GREY_TXT = "#444444"

# ---- 인쇄 크기 ----------------------------------------------------------
# main.tex 는 이 그림을 width=0.78\textwidth 로 넣는다(textwidth 176 mm → 137.3 mm).
# 그림을 그 폭 그대로 생성하고 tight 크롭을 끄면 축소 배율이 1.0 이 되어, 아래 pt 값이
# 인쇄 pt 와 같아진다. 모든 글자를 7 pt 이상으로 두어 6.5 pt 하한을 만족시킨다.
FIG_W_MM = 109.1      # main.tex \includegraphics[width=0.62\textwidth] 와 동일(인쇄 1:1)
FS_TAG = 7.5        # 패널 라벨
FS_CB_LAB = 7.5     # 컬러바 라벨
FS_CB_TICK = 7.0    # 컬러바 눈금
FS_GRID = 7.0       # 경위도 눈금
FS_LEG = 7.0        # 범례
FS_SCALE = 7.0      # 축척막대

# 축 사각형을 인치로 직접 배치한다(단위 in). 세 패널 폭이 정확히 같아지고 컬러바가
# 패널 폭에 정렬된다.
M_L, M_R = 0.38, 0.02       # 좌(위도 눈금)·우 여백. 7자 라벨(69.5°N) 폭 확보
M_T = 0.16                  # 상(패널 라벨) 여백
COL_GAP = 0.05              # 패널 간격
FOOT_H = 0.17               # 하단 범례 줄
CB_LAB_H, CB_TICK_H = 0.115, 0.105   # 컬러바 라벨·눈금 영역
CB_H = 0.075                # 컬러바 두께
LON_LAB_H = 0.145           # 지도 아래 경도 눈금 영역
CB_Y = FOOT_H + CB_LAB_H + CB_TICK_H
M_B = CB_Y + CB_H + LON_LAB_H


# ----------------------------------------------------------------------
# 1) 지형 공변량 (Copernicus DEM 30 m → 목표 격자)
# ----------------------------------------------------------------------
def _tile_name(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM.tif"


def dem_mosaic(win):
    """창을 덮는 DEM 타일을 원해상도 모자이크로 합친다. 반환 (arr, lat_c, lon_c)."""
    import rasterio

    lo0, lo1, la0, la1 = win
    nrow = int(round((la1 - la0) / DLAT_PX))
    ncol = int(round((lo1 - lo0) / DLON_PX))
    arr = np.full((nrow, ncol), np.nan, dtype=np.float32)
    lat_c = la1 - np.arange(nrow) * DLAT_PX          # 위→아래
    lon_c = lo0 + np.arange(ncol) * DLON_PX

    tlats = range(int(np.floor(la0)), int(np.ceil(la1)))
    tlons = range(int(np.floor(lo0)), int(np.ceil(lo1)))
    n_ok = 0
    for tlat in tlats:
        for tlon in tlons:
            path = os.path.join(DEM_DIR, _tile_name(tlat, tlon))
            if not os.path.exists(path):
                print(f"[DEM] 타일 없음 {os.path.basename(path)}")
                continue
            with rasterio.open(path) as ds:
                t = ds.read(1).astype(np.float32)
            t[t < -1000] = np.nan                     # nodata
            r0 = int(round((la1 - (tlat + 1)) / DLAT_PX))   # 타일 첫 행의 모자이크 행
            c0 = int(round((tlon - lo0) / DLON_PX))
            # 타일에서 잘라 쓸 구간
            tr0, tr1 = max(0, -r0), min(t.shape[0], nrow - r0)
            tc0, tc1 = max(0, -c0), min(t.shape[1], ncol - c0)
            if tr1 <= tr0 or tc1 <= tc0:
                continue
            arr[r0 + tr0:r0 + tr1, c0 + tc0:c0 + tc1] = t[tr0:tr1, tc0:tc1]
            n_ok += 1
            del t
    print(f"[DEM] 모자이크 {arr.shape[1]} x {arr.shape[0]} px (타일 {n_ok}매), "
          f"유효 {np.isfinite(arr).mean()*100:.1f}%, "
          f"표고 {np.nanmin(arr):.0f}~{np.nanmax(arr):.0f} m")
    return arr, lat_c, lon_c


def terrain_grid(win, agg_lat, agg_lon, stripe=1200):
    """DEM 원해상도에서 지형 6종을 계산해 (agg_lat x agg_lon) 블록평균으로 집계.

    계산식은 terrain_features_dem.py 와 동일:
      slope  = atan(|grad z|) [deg]         (중앙차분, m 단위 간격)
      aspect = atan2(-dz/dy, dz/dx)         (sin·cos 로 분해)
      tpi    = z - mean(33x33 창)
      rough  = std(33x33 창)
    """
    from scipy.ndimage import uniform_filter

    arr, lat_c, lon_c = dem_mosaic(win)
    nrow, ncol = arr.shape
    nlat = nrow // agg_lat
    nlon = ncol // agg_lon
    mperdeg = 111320.0
    dy = mperdeg * DLAT_PX

    acc = np.zeros((6, nlat, nlon), dtype=np.float64)
    halo = PATCH // 2 + 2
    filled = np.where(np.isfinite(arr), arr, np.nanmean(arr)).astype(np.float32)
    del arr

    for m0 in range(0, nlat, max(1, stripe // agg_lat)):
        m1 = min(nlat, m0 + max(1, stripe // agg_lat))
        r0, r1 = m0 * agg_lat, m1 * agg_lat
        h0, h1 = max(0, r0 - halo), min(nrow, r1 + halo)
        z = filled[h0:h1]
        lat_mid = float(np.mean(lat_c[r0:r1]))
        dx = mperdeg * DLON_PX * np.cos(np.radians(lat_mid))

        gy, gx = np.gradient(z, dy, dx)
        slope = np.degrees(np.arctan(np.hypot(gy, gx)))
        asp = np.arctan2(-gy, gx)
        del gy, gx
        mu = uniform_filter(z, size=PATCH, mode="nearest")
        mu2 = uniform_filter(z.astype(np.float64) ** 2, size=PATCH, mode="nearest")
        rough = np.sqrt(np.clip(mu2 - mu.astype(np.float64) ** 2, 0, None))
        del mu2

        a, b = r0 - h0, r1 - h0                      # 실제 사용할 구간
        feats = [z[a:b], slope[a:b], np.sin(asp[a:b]), np.cos(asp[a:b]),
                 (z[a:b] - mu[a:b]), rough[a:b]]
        for k, f in enumerate(feats):
            f = np.asarray(f, dtype=np.float64)[:, :nlon * agg_lon]
            acc[k, m0:m1] = f.reshape(m1 - m0, agg_lat, nlon, agg_lon).mean(axis=(1, 3))
        del z, slope, asp, mu, rough, feats

    del filled
    lat_t = np.array([lat_c[m * agg_lat:(m + 1) * agg_lat].mean() for m in range(nlat)])
    lon_t = np.array([lon_c[k * agg_lon:(k + 1) * agg_lon].mean() for k in range(nlon)])
    # 남→북 오름차순으로 뒤집어 반환(contourf·ERA5 좌표 규약과 일치)
    T = {name: acc[k][::-1] for k, name in enumerate(TERRAIN)}
    lat_t = lat_t[::-1]
    km_lat = DLAT_PX * agg_lat * mperdeg / 1000.0
    km_lon = DLON_PX * agg_lon * mperdeg * np.cos(np.radians(lat_t.mean())) / 1000.0
    print(f"[지형] 목표 격자 {nlon} x {nlat} = {nlon*nlat:,} 셀 "
          f"({km_lon*1000:.0f} m x {km_lat*1000:.0f} m)")
    for k in TERRAIN:
        v = T[k]
        print(f"        {k:16s} {np.nanmin(v):9.2f} ~ {np.nanmax(v):9.2f} "
              f"(중앙 {np.nanmedian(v):.2f})")
    return T, lat_t, lon_t, (km_lon * 1000, km_lat * 1000)


# ----------------------------------------------------------------------
# 2) 기후 공변량 (ERA5-Land 0.1°)
# ----------------------------------------------------------------------
def _nearest_fill(arr2d, valid):
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr2d[tuple(idx)]


def climate_grid(lat_t, lon_t, pad=0.6):
    """ERA5-Land 월 기후값 → 목표 격자 도일·기온·적설(이중선형)."""
    import xarray as xr

    ds = xr.open_dataset(NC)
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    win = ds[["t2m", "stl1", "sd"]].sel(
        latitude=slice(lat_t.max() + pad, lat_t.min() - pad),
        longitude=slice(lon_t.min() - pad, lon_t.max() + pad)).load()
    clim = win.assign_coords(month=win[tname].dt.month).groupby("month").mean(tname)

    land = np.isfinite(clim["t2m"].values.astype("float32")).all(axis=0)
    src = {}
    for k in ("t2m", "stl1", "sd"):
        a = clim[k].values.astype("float32")
        src[k] = np.stack([_nearest_fill(a[m], land) for m in range(a.shape[0])])
    filled = xr.Dataset(
        {k: (("month", "latitude", "longitude"), v) for k, v in src.items()},
        coords={"month": clim["month"].values,
                "latitude": clim["latitude"].values,
                "longitude": clim["longitude"].values})
    fine = filled.interp(latitude=lat_t, longitude=lon_t, method="linear")

    days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)],
                    dtype="float32")[:, None, None]
    t = fine["t2m"].values - 273.15
    stl = fine["stl1"].values - 273.15
    sd = fine["sd"].values
    tdd = np.nansum(np.clip(t, 0, None) * days, axis=0)
    fdd = np.nansum(np.clip(-t, 0, None) * days, axis=0)
    G = {"e5_maat": np.nanmean(t, axis=0), "e5_tdd": tdd, "e5_fdd": fdd,
         "e5_sqrt_tdd": np.sqrt(np.clip(tdd, 0, None)),
         "e5_twarm": np.nanmax(t, axis=0), "e5_tcold": np.nanmin(t, axis=0),
         "e5_stl1": np.nanmean(stl, axis=0), "e5_swe": np.nanmean(sd, axis=0)}
    ds.close()
    n_src = int(np.sum((clim["latitude"].values >= lat_t.min() - 0.05)
                       & (clim["latitude"].values <= lat_t.max() + 0.05))) * \
        int(np.sum((clim["longitude"].values >= lon_t.min() - 0.05)
                   & (clim["longitude"].values <= lon_t.max() + 0.05)))
    print(f"[기후] ERA5-Land 원격자 창 내 {n_src} 셀(0.1°) → 목표 격자 보간. "
          f"TDD {np.nanmin(tdd):.0f}~{np.nanmax(tdd):.0f} °C·d")
    return G


# ----------------------------------------------------------------------
# 3) 토양 공변량 (SoilGrids ~5 km)
# ----------------------------------------------------------------------
def soil_grid(lat_t, lon_t):
    """SoilGrids(IGH) 창에서 목표 격자점 이중선형 샘플. nodata는 최근접 유효값으로 채움."""
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt, map_coordinates

    lon2d, lat2d = np.meshgrid(lon_t, lat_t)
    tr = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)
    x, y = tr.transform(lon2d.ravel(), lat2d.ravel())
    out = {}
    for col, (fname, factor) in SG_LAYERS.items():
        with rasterio.open(os.path.join(SG_DIR, fname)) as src:
            b = src.read(1).astype("float64")
            if src.nodata is not None:
                b[b == src.nodata] = np.nan
            b[b <= 0] = np.nan
            valid = np.isfinite(b)
            idx = distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
            fc, fr = (~src.transform) * (x, y)
            v = map_coordinates(b[tuple(idx)], [fr - 0.5, fc - 0.5],
                                order=1, mode="nearest")
        out[col] = (v / factor).reshape(lat2d.shape)
    print("[토양] SoilGrids 9종 샘플: " + ", ".join(
        f"{c.replace('sg_','')}={np.nanmedian(v):.2f}" for c, v in list(out.items())[:4]))
    return out


# ----------------------------------------------------------------------
# 4) 모델 학습·예측
# ----------------------------------------------------------------------
FEATS = TERRAIN + CLIMATE + SOIL


def fit_model(seed=0):
    from catboost import CatBoostRegressor

    d = pd.read_csv(TRAIN_CSV, low_memory=False)
    ak = d[d.region.isin(AK_REGIONS)].dropna(subset=FEATS + ["alt_cm"]).copy()
    X, y = ak[FEATS].values.astype("float64"), ak["alt_cm"].values.astype("float64")
    m = CatBoostRegressor(iterations=1200, depth=7, learning_rate=0.05,
                          loss_function="RMSE", random_seed=seed,
                          task_type="CPU", thread_count=8, verbose=False)
    m.fit(X, y)
    imp = pd.Series(m.get_feature_importance(), index=FEATS).sort_values(ascending=False)
    r = y - m.predict(X)
    print(f"[학습] 알래스카 {len(ak):,}셀 · 특징 {len(FEATS)} · 재현 RMSE "
          f"{np.sqrt((r**2).mean()):.1f} cm")
    print("[중요도] " + " | ".join(f"{k} {v:.1f}" for k, v in imp.head(8).items()))
    terr_share = imp[TERRAIN].sum()
    print(f"[중요도] 지형 6종 합 {terr_share:.1f}% / 기후 {imp[CLIMATE].sum():.1f}% "
          f"/ 토양 {imp[SOIL].sum():.1f}%")
    return m, ak


def predict_fields(model, T, C, S, lat_t, lon_t):
    """(a) 지형 고정 · (b) 지형 실측 두 장을 같은 모델로 예측.

    고정값은 창의 중앙값을 쓴다. 평균은 창 남단 산지(표고·거칠기 상위 꼬리)에 끌려
    창의 대부분을 차지하는 연안평야를 대표하지 못하고, 그 결과 (b)-(a) 차이가 한쪽으로
    치우친 상수 편의를 얻는다.
    """
    ny, nx = len(lat_t), len(lon_t)
    n = ny * nx
    base = {k: np.asarray(v, dtype="float64").ravel() for k, v in
            list(C.items()) + list(S.items())}
    Xb = np.empty((n, len(FEATS)))
    Xa = np.empty((n, len(FEATS)))
    const = {}
    for j, f in enumerate(FEATS):
        if f in TERRAIN:
            v = np.asarray(T[f], dtype="float64").ravel()
            const[f] = float(np.nanmedian(v))
            Xb[:, j] = v
            Xa[:, j] = const[f]
        else:
            Xb[:, j] = base[f]
            Xa[:, j] = base[f]
    print("[대조군] 지형 고정값 " + " ".join(f"{k.replace('dem_','')}={v:.2f}"
                                            for k, v in const.items()))
    fa = model.predict(Xa).reshape(ny, nx)
    fb = model.predict(Xb).reshape(ny, nx)
    d = fb - fa
    print(f"[예측] (a) {np.nanpercentile(fa,1):.0f}~{np.nanpercentile(fa,99):.0f} cm "
          f"(표준편차 {np.nanstd(fa):.1f}) | "
          f"(b) {np.nanpercentile(fb,1):.0f}~{np.nanpercentile(fb,99):.0f} cm "
          f"(표준편차 {np.nanstd(fb):.1f})")
    print(f"[차이] (b)-(a) {np.nanpercentile(d,1):+.1f}~{np.nanpercentile(d,99):+.1f} cm, "
          f"|차이| 중앙 {np.nanmedian(np.abs(d)):.1f} cm")
    # 세부구조 정량: 목표격자 3x3 라플라시안 에너지 비(고주파 성분)
    def hf(a):
        return float(np.nanstd(a[1:-1, 1:-1] - 0.25 * (a[:-2, 1:-1] + a[2:, 1:-1]
                                                       + a[1:-1, :-2] + a[1:-1, 2:])))
    print(f"[세부] 고주파 표준편차 (a) {hf(fa):.3f} cm vs (b) {hf(fb):.3f} cm "
          f"(비 {hf(fb)/max(hf(fa),1e-9):.0f}배)")
    return fa, fb, d


def load_obs(win, dedup_deg=DEDUP_DEG):
    lo0, lo1, la0, la1 = win
    d = pd.read_csv(TRAIN_CSV, low_memory=False,
                    usecols=["lat", "lon", "region", "alt_cm"])
    d = d[d.region.isin(AK_REGIONS)]
    d = d[d.lat.between(la0, la1) & d.lon.between(lo0, lo1)]
    g = (d.assign(_la=(d.lat / dedup_deg).round(), _lo=(d.lon / dedup_deg).round())
           .groupby(["_la", "_lo"], as_index=False)
           .agg(lat=("lat", "mean"), lon=("lon", "mean"), alt_cm=("alt_cm", "mean")))
    print(f"[관측] 창 내 원 셀 {len(d):,} → {dedup_deg}° 집계 {len(g):,} 위치")
    return g


# ----------------------------------------------------------------------
# 5) 렌더
# ----------------------------------------------------------------------
def alt_cmap():
    from cmcrameri import cm as cmc
    from matplotlib.colors import LinearSegmentedColormap
    cols = cmc.oslo_r(np.linspace(0.20, 0.68, 256))
    cm = LinearSegmentedColormap.from_list("alt_hires", cols)
    cm.set_bad(alpha=0.0)
    return cm


def diff_cmap():
    from cmcrameri import cm as cmc
    from matplotlib.colors import LinearSegmentedColormap
    cols = cmc.broc(np.linspace(0.08, 0.92, 256))
    cm = LinearSegmentedColormap.from_list("alt_diff", cols)
    cm.set_bad(alpha=0.0)
    return cm


def _panel(ax, lon_t, lat_t, F, cmap, norm):
    """등치대(이산 색단) 렌더.

    색 이산화는 BoundaryNorm이 담당하므로 등치대 외관은 contourf와 같다. 다만 목표
    격자가 53만 셀이라 contourf는 등치선 폴리곤 수십만 개를 투영변환해야 해 실사용
    시간(1시간 이상)을 넘긴다. 셀 크기가 인쇄 시 1픽셀 미만이므로 동일 BoundaryNorm을
    쓴 pcolormesh로 대체한다(외관 동일, 수십 초). 등치선 오버레이는 조밀 격자에서
    선 잡음이 되므로 두지 않는다.
    """
    import cartopy.crs as ccrs
    A = np.ma.masked_invalid(F)
    im = ax.pcolormesh(lon_t, lat_t, A, cmap=cmap, norm=norm, shading="nearest",
                       transform=ccrs.PlateCarree(), zorder=1.5, rasterized=True)
    return im


def inscribed_box(proj, extent):
    """자료 사각형의 내접 직사각형 (x0, x1, y0, y1)을 투영좌표로 반환한다.

    원뿔투영에서 경위도 창은 곡선 사변형이 되므로 set_extent 의 외접 사각형에는
    자료가 없는 모서리 띠가 남는다. 네 변을 투영해 내접 직사각형을 취하면 예측장이
    축 전체를 채운다. 이 상자의 가로/세로 비로 축 사각형을 잡아야 등축척(equal aspect)
    조정이 축 크기를 다시 줄이지 않는다.
    """
    import cartopy.crs as ccrs
    _, lo0, lo1, la0, la1 = extent
    n = 201
    lo = np.linspace(lo0, lo1, n)
    la = np.linspace(la0, la1, n)

    def pr(x, y):
        p = proj.transform_points(ccrs.PlateCarree(), np.asarray(x), np.asarray(y))
        return p[:, 0], p[:, 1]
    xl, _ = pr(np.full(n, lo0), la)
    xr, _ = pr(np.full(n, lo1), la)
    _, yb = pr(lo, np.full(n, la0))
    _, yt = pr(lo, np.full(n, la1))
    return float(xl.max()), float(xr.min()), float(yb.max()), float(yt.min())


def _check_print_size(fig, tags, map_w):
    """인쇄 1:1 가정에서 글자 크기·라벨 폭을 실측 검증한다(축소 배율 1.0)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    seen = []
    for t in fig.findobj(Text):
        s = (t.get_text() or "").strip()
        if s:
            seen.append((float(t.get_fontsize()), s))
    # 축척막대 문자는 매 draw 마다 새 OffsetBox 로 만들어져 findobj 로 잡히지 않는다.
    seen.append((float(FS_SCALE), "축척막대"))
    seen.sort(key=lambda x: x[0])
    lo = [f"{v:.1f} pt '{s[:22]}'" for v, s in seen[:3]]
    print(f"[QA] 그림 내 글자 {len(seen)}개 · 최소 " + " | ".join(lo))
    if seen[0][0] < 6.5:
        print("[QA] 경고: 6.5 pt 미만 글자가 있다")
    for t in tags:
        w = t.get_window_extent(renderer=r).width / fig.dpi
        flag = "" if w <= map_w else "  <- 패널 폭 초과"
        print(f"[QA] 라벨 폭 {w:.2f} in / 패널 {map_w:.2f} in  '{t.get_text()}'{flag}")


def render(lon_t, lat_t, fa, fb, d, obs, cell_m, out="alt_downscale_demo"):
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from matplotlib.colors import BoundaryNorm
    from polar.geomap import map_projection

    # 표시 창을 격자 셀 중심의 외곽에 맞춘다(등치대가 축 가장자리까지 채워 배경 띠 제거).
    extent = (WIN_NAME, float(lon_t[0]), float(lon_t[-1]),
              float(lat_t[0]), float(lat_t[-1]))
    proj = map_projection(extent)
    x0, x1, y0, y1 = inscribed_box(proj, extent)
    asp = (x1 - x0) / (y1 - y0)

    v = np.concatenate([fa[np.isfinite(fa)], fb[np.isfinite(fb)]])
    vmin = float(np.floor(np.percentile(v, 1) / 2) * 2)
    vmax = float(np.ceil(np.percentile(v, 99) / 2) * 2)
    levels = np.linspace(vmin, vmax, 25)
    cmap = alt_cmap()
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend="both")

    dv = float(np.nanpercentile(np.abs(d), 99))
    dv = max(np.ceil(dv / 2.0) * 2.0, 2.0)
    dlev = np.linspace(-dv, dv, 21)
    dcmap = diff_cmap()
    dnorm = BoundaryNorm(dlev, ncolors=dcmap.N, extend="both")
    print(f"[색] ALT {vmin:.0f}~{vmax:.0f} cm ({len(levels)-1}단) | 차이 ±{dv:.0f} cm")

    fig_w = FIG_W_MM / 25.4
    map_w = (fig_w - M_L - M_R - 2 * COL_GAP) / 3
    map_h = map_w / asp
    fig_h = M_B + map_h + M_T
    print(f"[배치] 패널 종횡비 {asp:.3f} · 지도 {map_w*25.4:.1f} x {map_h*25.4:.1f} mm · "
          f"전체 {fig_w*25.4:.1f} x {fig_h*25.4:.1f} mm (인쇄 1:1)")

    def fr(x, y, w, h):
        """인치 → figure 좌표."""
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i in range(3):
        ax = fig.add_axes(fr(M_L + i * (map_w + COL_GAP), M_B, map_w, map_h),
                          projection=proj)
        make_ax(extent, ax=ax, fig=fig, grid=True, grid_labels=True,
                grid_kw=dict(linewidth=0.25, color="0.72", alpha=0.5,
                             linestyle=":"))
        gl = ax._gl
        if gl is not None:
            gl.rotate_labels = False
            gl.left_labels = (i == 0)          # 위도 눈금은 왼쪽 패널에만
            gl.xpadding = gl.ypadding = 2      # 기본 5 pt 는 좁은 여백에서 과하다
            gl.xlocator = mticker.FixedLocator(np.arange(-151.0, -148.4, 1.0))
            gl.ylocator = mticker.FixedLocator(np.arange(68.5, 70.0, 0.5))
            gl.xlabel_style = gl.ylabel_style = {"size": FS_GRID, "color": "0.38"}
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_edgecolor("#8e8e8e")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        axes.append(ax)

    im_a = _panel(axes[0], lon_t, lat_t, fa, cmap, norm)
    _panel(axes[1], lon_t, lat_t, fb, cmap, norm)
    im_c = _panel(axes[2], lon_t, lat_t, d, dcmap, dnorm)

    for ax in axes:
        ax.scatter(obs.lon.values, obs.lat.values, s=2.0, c="#111111",
                   linewidths=0, alpha=0.6, transform=ccrs.PlateCarree(),
                   zorder=4)
    # 세 패널이 같은 창·같은 투영이므로 축척막대는 하나만 둔다.
    add_scalebar(axes[0], fontsize=FS_SCALE)

    tags = ["(a) 기후·토양만",
            f"(b) 지형 추가 · {cell_m[0]:.0f} m",
            "(c) (b) $-$ (a)"]
    dy = 0.02 / map_h                          # 축 위 2 mm 미만 간격(축 좌표 환산)
    tag_txt = [ax.text(0.0, 1.0 + dy, t, transform=ax.transAxes,
                       fontsize=FS_TAG, color=GREY_TXT, ha="left", va="bottom")
               for ax, t in zip(axes, tags)]

    cax1 = fig.add_axes(fr(M_L, CB_Y, 2 * map_w + COL_GAP, CB_H))
    cb1 = fig.colorbar(im_a, cax=cax1, orientation="horizontal")
    cb1.set_label("활성층 두께 ALT (cm)", fontsize=FS_CB_LAB, color=GREY_TXT,
                  labelpad=1.5)
    cb1.set_ticks(np.arange(np.ceil(vmin / 5) * 5, vmax + 0.1, 5))
    cax2 = fig.add_axes(fr(M_L + 2 * (map_w + COL_GAP), CB_Y, map_w, CB_H))
    cb2 = fig.colorbar(im_c, cax=cax2, orientation="horizontal")
    cb2.set_label("ALT 차이 (cm)", fontsize=FS_CB_LAB, color=GREY_TXT, labelpad=1.5)
    cb2.set_ticks(np.arange(-dv, dv + 0.1, max(2.0, round(dv / 2))))
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=FS_CB_TICK, length=2.0, pad=1.4,
                          color=GREY_TXT, labelcolor=GREY_TXT)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor("#9a9a9a")

    # 관측점 범례는 지도 위 상자 대신 하단 한 줄로 둔다(좁은 패널을 가리지 않는다).
    h = plt.Line2D([], [], marker="o", ls="none", ms=2.4, mfc="#111111", mec="none",
                   label=f"실측 셀 {DEDUP_DEG:g}° 집계 위치 (n={len(obs):,})")
    leg = fig.legend(handles=[h], loc="lower left", frameon=False,
                     bbox_to_anchor=(M_L / fig_w, 0.02 / fig_h),
                     fontsize=FS_LEG, handletextpad=0.4, borderpad=0.0,
                     borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(GREY_TXT)

    _check_print_size(fig, tag_txt, map_w)
    png, pdf = mappath(out), mappath(out, ext="pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, dpi=300)
    plt.close(fig)
    print("saved", png)
    print("saved", pdf)


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-lat", type=int, default=8, help="DEM 화소 집계 배수(위도)")
    ap.add_argument("--agg-lon", type=int, default=11, help="DEM 화소 집계 배수(경도)")
    a = ap.parse_args()

    t0 = time.time()

    def lap(tag):
        print(f"[시간] {tag} {time.time()-lap.t:.1f}s (누적 {time.time()-t0:.1f}s)",
              flush=True)
        lap.t = time.time()
    lap.t = t0

    T, lat_t, lon_t, cell_m = terrain_grid(WIN, a.agg_lat, a.agg_lon)
    lap("지형")
    C = climate_grid(lat_t, lon_t)
    lap("기후")
    S = soil_grid(lat_t, lon_t)
    lap("토양")
    model, _ = fit_model()
    lap("학습")
    fa, fb, d = predict_fields(model, T, C, S, lat_t, lon_t)
    lap("예측")
    obs = load_obs(WIN)
    render(lon_t, lat_t, fa, fb, d, obs, cell_m)
    lap("렌더")


if __name__ == "__main__":
    main()
