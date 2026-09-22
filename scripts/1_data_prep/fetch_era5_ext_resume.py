"""ERA5-Land 확장 변수 월평균(2015–2020, NH 50–85°N) — CDS 결과 URL을 받아 curl 재개 다운로드(연결 끊김 대응)."""
import subprocess, sys, time
from pathlib import Path
import cdsapi
ROOT = Path(__file__).resolve().parents[2]
dst = ROOT / "data/raw/era5land/nh_monthly_ext_2015-2020.nc"
c = cdsapi.Client()
req = {"product_type": ["monthly_averaged_reanalysis"],
       "variable": ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "skin_temperature",
                    "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation", "snow_density", "snow_depth", "total_precipitation"],
       "year": [str(y) for y in range(2015, 2021)], "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00"],
       "data_format": "netcdf", "download_format": "unarchived", "area": [85, -180, 50, 180]}
r = c.retrieve("reanalysis-era5-land-monthly-means", req)
url = getattr(r, "location", None) or r.url
size = getattr(r, "content_length", None)
print("url", url, "size", size, flush=True)
for k in range(60):
    p = subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", str(dst), url])
    have = dst.stat().st_size if dst.exists() else 0
    print(f"try {k}: {have/1e6:.1f} MB", flush=True)
    if size and have >= int(size):
        break
    if not size and p.returncode == 0:
        break
    time.sleep(5)
print("done", dst.stat().st_size)
