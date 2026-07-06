#!/usr/bin/env python3
"""② 전처리: 원본 CSV → long table + MAGT 프로파일 + borehole 요약."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polar import preprocess  # noqa: E402

if __name__ == "__main__":
    preprocess.run()
