"""H18·H19 외부 공변량 원자료 취득 (계획 docs/EXPERIMENT_PLAN_NEXT_2026-09-22.md §2 A·B, 설계 docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md).

  --modis : MOD11C3(Terra 월별 LST 0.05°, 주·야)·MOD13C2(월별 NDVI/EVI/반사율 0.05°) 2015–2020 → data/raw/modis_cmg/ (earthaccess, ~/.netrc)
  --tiles : JRC Global Surface Water occurrence(30 m, 10° 타일)·Hansen GFC treecover2000(30 m, 10° 타일) 라벨 셀이 있는 타일만 → data/raw/{gsw,hansen}/
  --era5  : ERA5-Land 월평균 확장 변수(토양수분 1·2층, 지표면 온도 skt, LAI 고·저, 적설 밀도·깊이, 강수) 2015–2020 NH 50–85°N → data/raw/era5land/nh_monthly_ext_2015-2020.nc (cdsapi)
실행(ROOT): python3 scripts/1_data_prep/fetch_ext_covariates.py --modis --tiles --era5
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--modis", action="store_true"); ap.add_argument("--tiles", action="store_true"); ap.add_argument("--era5", action="store_true")
ap.add_argument("--aqua", action="store_true", help="MYD11C3(Aqua)도 받기")
args = ap.parse_args()

df = pd.read_csv(ROOT / "data/processed/fidelity_base_v3.csv", usecols=["lat", "lon"])

if args.modis:
    import earthaccess
    earthaccess.login(strategy="netrc")
    out = ROOT / "data/raw/modis_cmg"; out.mkdir(exist_ok=True)
    prods = ["MOD11C3", "MOD13C2"] + (["MYD11C3"] if args.aqua else [])
    for sn in prods:
        g = earthaccess.search_data(short_name=sn, version="061", temporal=("2015-01-01", "2020-12-31"))
        print(f"[modis] {sn}: {len(g)} granules", flush=True)
        t0 = time.time()
        earthaccess.download(g, str(out), threads=4)
        print(f"[modis] {sn} done {time.time()-t0:.0f}s", flush=True)

if args.tiles:
    import urllib.request
    def tile_corners(lat, lon):
        """10° 타일 좌상단(lat: 위쪽 경계, lon: 왼쪽 경계)."""
        return int(np.floor(lat / 10) * 10 + 10), int(np.floor(lon / 10) * 10)
    need = sorted(set(tile_corners(a, b) for a, b in zip(df.lat, df.lon)))
    print("[tiles] 필요 타일", need, flush=True)
    for top, left in need:
        ns = f"{abs(top)}{'N' if top >= 0 else 'S'}"; ew = f"{abs(left):03d}{'E' if left >= 0 else 'W'}"
        ew_g = f"{abs(left)}{'E' if left >= 0 else 'W'}"
        jobs = [
            (f"https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{ew_g}_{ns}v1_4_2021.tif",
             ROOT / f"data/raw/gsw/occurrence_{ew_g}_{ns}v1_4_2021.tif"),
            (f"https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_treecover2000_{ns}_{ew}.tif",
             ROOT / f"data/raw/hansen/Hansen_GFC-2023-v1.11_treecover2000_{ns}_{ew}.tif"),
        ]
        for url, dst in jobs:
            if dst.exists() and dst.stat().st_size > 1e6:
                continue
            try:
                t0 = time.time(); urllib.request.urlretrieve(url, dst)
                print(f"[tiles] {dst.name} {dst.stat().st_size/1e6:.0f}MB {time.time()-t0:.0f}s", flush=True)
            except Exception as e:                        # noqa: BLE001
                print(f"[tiles] FAIL {url}: {e}", flush=True)

if args.era5:
    import cdsapi
    dst = ROOT / "data/raw/era5land/nh_monthly_ext_2015-2020.nc"
    if not dst.exists():
        c = cdsapi.Client()
        c.retrieve("reanalysis-era5-land-monthly-means", {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "skin_temperature",
                         "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation",
                         "snow_density", "snow_depth", "total_precipitation"],
            "year": [str(y) for y in range(2015, 2021)], "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"], "data_format": "netcdf", "download_format": "unarchived",
            "area": [85, -180, 50, 180]}, str(dst))
        print(f"[era5] saved {dst} {dst.stat().st_size/1e9:.2f}GB", flush=True)
