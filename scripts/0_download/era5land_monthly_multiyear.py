"""ERA5-Land 월별 평균 다년(2010-2024) 다운로드 — 시간정합(㉠)용.

기존 nh_monthly_2015-2020.nc(정적 평년값 근사)의 한계 극복:
라벨의 70%가 2010-2014(특히 2014=59%)라 그 해 기후로 정합하려면 다년 필요.
→ 각 (위치,연도) 라벨에 '그 해' TDD/FDD/여름온도/적설을 붙여 연도차를 신호로.

실행(백그라운드): CUDA 불필요(네트워크). CDS 큐라 수십분~수시간 소요 가능.
  python3 scripts/0_download/era5land_monthly_multiyear.py
"""
import os
import cdsapi

OUT = "data/raw/era5land"
os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "nh_monthly_2010-2024.nc")

if os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
    print("이미 있음:", dst, os.path.getsize(dst), "bytes"); raise SystemExit

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-land-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "2m_temperature",
            "snow_depth_water_equivalent",
            "soil_temperature_level_1",
        ],
        "year": [str(y) for y in range(2010, 2025)],   # 2010-2024 (라벨 99.4% 커버)
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": [84, -180, 25, 180],   # N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    },
    dst,
)
print("DONE", dst, os.path.getsize(dst), "bytes")
