#!/usr/bin/env python3
"""① GTN-P 데이터 다운로드 (기본: Alaska 지중온도, Open).

예:
  python scripts/01_download_gtnp.py                 # Alaska 지중온도 + ALT
  python scripts/01_download_gtnp.py --region all    # 전 지구
  python scripts/01_download_gtnp.py --manifest-only # 매니페스트만
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polar import acquire  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="alaska", choices=["alaska", "all"])
    ap.add_argument("--variable", default="Ground Temperature",
                    help="PT 변수 필터 ('all'이면 전체)")
    ap.add_argument("--all-policies", action="store_true", help="Restricted 포함")
    ap.add_argument("--no-alt", action="store_true", help="ALT 제외")
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()
    acquire.run(region=a.region,
                variable=None if a.variable == "all" else a.variable,
                open_only=not a.all_policies,
                include_alt=not a.no_alt,
                manifest_only=a.manifest_only)
