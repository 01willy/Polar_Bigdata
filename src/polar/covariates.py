"""
공변량(covariate) 취득.

(A) 지점 샘플링 — 무인증 공개 API로 borehole 위치의 예측변수 즉시 취득:
    - 고도(DEM): USGS EPQS (미국/Alaska 전용, 무인증)
    - 토양: SoilGrids 2.0 REST (전지구, 무인증; clay/sand/soc/bdod, 깊이별)
  -> data/processed/borehole_covariates.csv

(B) 격자(gridded) 소스 — 예측 격자 전체용. 계정/인증 필요:
    - ESA CCI Permafrost (CEDA), ERA5-Land (Copernicus CDS), MODIS LST (GEE),
      ArcticDEM (PGC/AWS, 일부 무인증). check_gridded_sources()가 설정 상태를 진단.
"""
import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

# --- WorldClim 2.1 (무계정 오픈) — 기후·고도 격자 공변량 ---
WC_DIR = C.DATA / "raw" / "covariates" / "worldclim"
WC_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/"
WC_RES = "10m"          # 10 arc-min(~18km). 더 고해상도는 2.5m/30s(대용량)
# 사용할 변수: 고도 + 핵심 생기후(BIO1=연평균기온, BIO4=기온계절성, BIO7=연교차, BIO12=연강수)
WC_VARS = {"elev": "wc2.1_{r}_elev.tif",
           "bio1": "wc2.1_{r}_bio_1.tif", "bio4": "wc2.1_{r}_bio_4.tif",
           "bio7": "wc2.1_{r}_bio_7.tif", "bio12": "wc2.1_{r}_bio_12.tif"}
WC_LABEL = {"elev": "고도(m)", "bio1": "연평균기온(°C)", "bio4": "기온계절성",
            "bio7": "연기온교차(°C)", "bio12": "연강수량(mm)"}

EPQS = "https://epqs.nationalmap.gov/v1/json"
SOILGRIDS = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPS = ["clay", "sand", "soc", "bdod"]     # 점토/모래/유기탄소/용적밀도
SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]


def usgs_elevation(lat, lon):
    try:
        r = requests.get(EPQS, params=dict(x=lon, y=lat, units="Meters",
                                           wkid=4326, includeDate="false"), timeout=30)
        if r.status_code == 200:
            return float(r.json()["value"])
    except (requests.RequestException, KeyError, ValueError, TypeError):
        pass
    return None


def soilgrids_point(lat, lon, retries=4):
    """SoilGrids 평균값(여러 property×depth). rate-limit(429) 백오프."""
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    params += [("property", p) for p in SOIL_PROPS]
    params += [("depth", d) for d in SOIL_DEPTHS]
    for attempt in range(retries):
        try:
            r = requests.get(SOILGRIDS, params=params, timeout=40)
            if r.status_code == 200:
                out = {}
                for layer in r.json().get("properties", {}).get("layers", []):
                    name = layer["name"]
                    for dd in layer.get("depths", []):
                        v = dd.get("values", {}).get("mean")
                        # 단위 스케일(d_factor)로 나눠 물리값으로
                        df = layer.get("unit_measure", {}).get("d_factor", 1) or 1
                        if v is not None:
                            out[f"{name}_{dd['label']}"] = v / df
                return out
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
        except requests.RequestException:
            time.sleep(5)
    return {}


def enrich_boreholes():
    """borehole_summary.csv의 각 지점에 고도+토양 공변량을 붙여 저장."""
    C.ensure_dirs()
    summ = pd.read_csv(C.SUMMARY_CSV)
    rows = []
    n = len(summ)
    for i, r in summ.iterrows():
        rec = dict(borehole_id=int(r["borehole_id"]), lat=r["lat"], lon=r["lon"])
        rec["elev_dem"] = usgs_elevation(r["lat"], r["lon"])
        rec.update(soilgrids_point(r["lat"], r["lon"]))
        rows.append(rec)
        print(f"  [{i + 1}/{n}] bh {rec['borehole_id']}: elev={rec['elev_dem']}, "
              f"soil_keys={len(rec) - 4}")
        time.sleep(1.5)            # 공개 API 예의상 간격
    cov = pd.DataFrame(rows)
    out = C.PROCESSED / "borehole_covariates.csv"
    cov.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(cov)} boreholes, {cov.shape[1]} cols)")
    return cov


def download_worldclim():
    """WorldClim 2.1 elev+bio 다운로드(무계정). 이미 있으면 건너뜀."""
    WC_DIR.mkdir(parents=True, exist_ok=True)
    needed = {v.format(r=WC_RES) for v in WC_VARS.values()}
    have = {p.name for p in WC_DIR.glob("*.tif")}
    if needed <= have:
        print(f"[WorldClim] 이미 보유: {sorted(needed)}")
        return
    for zname in [f"wc2.1_{WC_RES}_elev.zip", f"wc2.1_{WC_RES}_bio.zip"]:
        print(f"[WorldClim] 다운로드 {zname} ...")
        r = requests.get(WC_BASE + zname, timeout=300)
        if r.status_code != 200:
            print(f"  실패 {r.status_code}"); continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for n in z.namelist():
                if n.endswith(".tif") and Path(n).name in needed:
                    (WC_DIR / Path(n).name).write_bytes(z.read(n))
    print(f"[WorldClim] 저장 -> {WC_DIR}  ({len(list(WC_DIR.glob('*.tif')))} tif)")


def _wc_read(var):
    import tifffile
    arr = tifffile.imread(WC_DIR / WC_VARS[var].format(r=WC_RES)).astype("float32")
    arr[arr < -9999] = np.nan                       # nodata(-3.4e38 float / -32768 int 모두)
    return arr


# --- WorldClim 월별(tavg/prec) → 도일(degree-day) 물리 피처 (무계정) ---
_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype="float32")


def download_worldclim_monthly():
    """WorldClim 월별 평균기온(tavg)·강수(prec) 12밴드 다운로드(무계정)."""
    WC_DIR.mkdir(parents=True, exist_ok=True)
    for var in ["tavg", "prec"]:
        need = {f"wc2.1_{WC_RES}_{var}_{m:02d}.tif" for m in range(1, 13)}
        if need <= {p.name for p in WC_DIR.glob("*.tif")}:
            print(f"[WorldClim] 월별 {var} 보유"); continue
        zname = f"wc2.1_{WC_RES}_{var}.zip"
        print(f"[WorldClim] 다운로드 {zname} ...")
        r = requests.get(WC_BASE + zname, timeout=400)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for n in z.namelist():
                if n.endswith(".tif") and Path(n).name in need:
                    (WC_DIR / Path(n).name).write_bytes(z.read(n))
    print(f"[WorldClim] 월별 저장 -> {WC_DIR}")


def _read_month(var, m):
    import tifffile
    a = tifffile.imread(WC_DIR / f"wc2.1_{WC_RES}_{var}_{m:02d}.tif").astype("float32")
    a[a < -9990] = np.nan
    return a


def sample_degree_days(lons, lats):
    """경위도 -> 도일 기반 물리 피처. ALT의 물리적 예측력 강화.
    tdd=융해도일, fdd=동결도일, sqrt_tdd=Stefan ALT proxy, thaw_months=해빙월수,
    summer_t=여름평균기온, winter_prec=겨울강수(적설 proxy)."""
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    tavg = np.stack([_read_month("tavg", m) for m in range(1, 13)])   # (12,H,W)
    prec = np.stack([_read_month("prec", m) for m in range(1, 13)])
    H, W = tavg.shape[1:]
    res = 360.0 / W
    col = np.clip(((lons + 180) / res).astype(int), 0, W - 1)
    row = np.clip(((90 - lats) / res).astype(int), 0, H - 1)
    T = tavg[:, row, col]                                            # (12, N) °C
    P = prec[:, row, col]                                            # (12, N) mm
    d = _DAYS[:, None]
    tdd = np.nansum(np.clip(T, 0, None) * d, axis=0)                 # 융해 도일
    fdd = np.nansum(np.clip(-T, 0, None) * d, axis=0)                # 동결 도일
    thaw_months = np.nansum(T > 0, axis=0).astype("float32")
    summer_t = np.nanmax(T, axis=0)                                  # 가장 따뜻한 달
    winter_prec = np.nansum(np.where(T < 0, P, 0.0), axis=0)         # 결빙기 강수=적설 proxy
    return dict(tdd=tdd, fdd=fdd, sqrt_tdd=np.sqrt(np.clip(tdd, 0, None)),
                thaw_months=thaw_months, summer_t=summer_t, winter_prec=winter_prec)


def sample_worldclim(lons, lats):
    """경위도 배열 -> {var: 값배열} (최근접 픽셀). WorldClim 전지구 등간격 격자."""
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    out = {}
    for var in WC_VARS:
        a = _wc_read(var)
        nrow, ncol = a.shape
        res = 360.0 / ncol                          # 전지구 -180..180
        col = np.clip(((lons + 180.0) / res).astype(int), 0, ncol - 1)
        row = np.clip(((90.0 - lats) / res).astype(int), 0, nrow - 1)
        out[var] = a[row, col]
    return out


def enrich_with_worldclim():
    """borehole_covariates.csv에 WorldClim 변수 추가."""
    download_worldclim()
    cov = pd.read_csv(C.PROCESSED / "borehole_covariates.csv")
    wc = sample_worldclim(cov["lon"].to_numpy(), cov["lat"].to_numpy())
    for var, vals in wc.items():
        cov[f"wc_{var}"] = vals
    cov.to_csv(C.PROCESSED / "borehole_covariates.csv", index=False)
    print(f"[WorldClim] borehole 공변량에 {len(wc)}개 추가 -> borehole_covariates.csv")
    print(cov[["borehole_id"] + [f"wc_{v}" for v in WC_VARS]].head(8).to_string(index=False))
    return cov


def check_gridded_sources():
    """격자 공변량 소스의 인증/도구 상태 진단 + 안내."""
    print("=== 격자 공변량 소스 상태 ===")
    have_cds = os.path.exists(os.path.expanduser("~/.cdsapirc"))
    have_gee = os.path.exists(os.path.expanduser("~/.config/earthengine/credentials"))
    print(f"[ERA5-Land] Copernicus CDS: {'준비됨' if have_cds else '미설정'} "
          "(pip install cdsapi + ~/.cdsapirc 필요; https://cds.climate.copernicus.eu)")
    print(f"[MODIS LST] Google Earth Engine: {'준비됨' if have_gee else '미설정'} "
          "(pip install earthengine-api + earthengine authenticate 필요)")
    print("[ESA CCI Permafrost] CEDA: 계정 필요 "
          "(https://catalogue.ceda.ac.uk/uuid/5675b0be944f45a8af0e7ddbeb47a011, NetCDF; "
          "표층/1/2/5/10 m 별도 변수 GST/T1m/T2m/T5m/T10m)")
    print("[ArcticDEM] PGC/AWS: 일부 무인증 (10/32 m mosaic, s3://pgc-opendata-dems)")
    print("필요 라이브러리(격자 처리): pip install cdsapi earthengine-api rasterio rioxarray netCDF4")


def download_era5_land(year_start=2005, year_end=2012):
    """ERA5-Land(2m기온·토양온도·적설) Alaska 월평균 다운로드. CDS 계정 필요.
    준비: pip install cdsapi ; ~/.cdsapirc 에 url/key (https://cds.climate.copernicus.eu)."""
    try:
        import cdsapi
    except ImportError:
        print("[ERA5] cdsapi 미설치. `pip install cdsapi` 후 ~/.cdsapirc 설정 필요. (건너뜀)")
        return None
    if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
        print("[ERA5] ~/.cdsapirc 없음. https://cds.climate.copernicus.eu 가입 후 키 설정. (건너뜀)")
        return None
    out_dir = C.DATA / "raw" / "covariates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"era5_land_alaska_{year_start}_{year_end}.nc"
    bb = C.ALASKA_BBOX
    cdsapi.Client().retrieve(
        "reanalysis-era5-land-monthly-means",
        {"product_type": "monthly_averaged_reanalysis",
         "variable": ["2m_temperature", "soil_temperature_level_1", "soil_temperature_level_4",
                      "snow_depth_water_equivalent"],
         "year": [str(y) for y in range(year_start, year_end + 1)],
         "month": [f"{m:02d}" for m in range(1, 13)], "time": "00:00",
         "area": [bb["lat_max"], bb["lon_min"], bb["lat_min"], bb["lon_max"]],
         "format": "netcdf"}, str(out))
    print(f"[ERA5] saved {out}")
    return out


def run():
    """③ 지점 enrich (고도+토양) + 격자소스 진단."""
    enrich_boreholes()
    print()
    check_gridded_sources()


def run_gridded():
    """④ 격자 공변량 다운로드. 무계정(WorldClim) 즉시 + 계정 필요(ERA5) 시도."""
    print("[무계정] WorldClim 기후·고도")
    enrich_with_worldclim()
    print()
    check_gridded_sources()
    print("\n[계정 필요 — 가능 시]")
    download_era5_land()
