#!/usr/bin/env python3
"""③ 공변량 enrich: borehole 지점에 고도(USGS)+토양(SoilGrids) 샘플 + 격자소스 진단."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polar import covariates  # noqa: E402

if __name__ == "__main__":
    covariates.run()
