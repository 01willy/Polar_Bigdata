"""ERA5-Land 월별 평균(북반구) 다운로드 → 실측 월기온으로 도일(TDD/FDD)·적설·토양온도 공변량 산출용.
WorldClim(18km 평년값)의 정밀 대체. 우선 최근 6년(2015-2020) 월별로 기후값 근사.
"""
import os
import cdsapi

OUT = "data/raw/era5land"
os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, "nh_monthly_2015-2020.nc")

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-land-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "2m_temperature",                # 도일(TDD/FDD)·MAAT
            "snow_depth_water_equivalent",   # 적설(단열)
            "soil_temperature_level_1",      # 지중온도 상부경계
        ],
        "year": ["2015", "2016", "2017", "2018", "2019", "2020"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        # N, W, S, E  (북반구 영구동토+고산 대부분 포함; 남반구 소수 사이트 제외)
        "area": [84, -180, 25, 180],
        "data_format": "netcdf",
        "download_format": "unarchived",
    },
    dst,
)
print("DONE", dst, os.path.getsize(dst), "bytes")
