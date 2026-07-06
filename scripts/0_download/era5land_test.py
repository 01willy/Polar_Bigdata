"""ERA5-Land 시험 다운로드: 알래스카 소구역, 1일, 소수 변수로 라이선스·파이프라인 확인."""
import os
import cdsapi

OUT = "data/raw/era5land"
os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "test_alaska_2020-07-15.nc")

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-land",
    {
        "variable": [
            "soil_temperature_level_1",   # 0~7cm 토양온도
            "soil_temperature_level_4",   # 100~289cm (심부)
            "snow_depth_water_equivalent",
        ],
        "year": "2020",
        "month": "07",
        "day": "15",
        "time": ["00:00", "12:00"],
        # N, W, S, E  (알래스카 일부)
        "area": [72, -165, 60, -140],
        "data_format": "netcdf",
        "download_format": "unarchived",
    },
    dst,
)
print("DONE", dst, os.path.getsize(dst), "bytes")
