#!/usr/bin/env python3
"""④ 격자 공변량 다운로드 (CCI/ERA5/MODIS/ArcticDEM).

대부분 계정/인증 필요:
  - ERA5-Land: pip install cdsapi + ~/.cdsapirc  (https://cds.climate.copernicus.eu)
  - MODIS LST: pip install earthengine-api + earthengine authenticate
  - ESA CCI Permafrost: CEDA 계정 (https://catalogue.ceda.ac.uk)
이 스크립트는 자격증명이 있으면 ERA5-Land를 실제 받고, 없으면 설정법을 안내한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polar import covariates  # noqa: E402

if __name__ == "__main__":
    covariates.run_gridded()
