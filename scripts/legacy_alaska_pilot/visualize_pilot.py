#!/usr/bin/env python3
"""⑥ 시각화: 개요 패널(matplotlib) + 세련된 3D hero 렌더(PyVista 합성)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polar import visualize, viz_suite, viz3d  # noqa: E402

if __name__ == "__main__":
    visualize.run()      # 01 개요
    viz_suite.run()      # 02 깊이슬라이스 / 03 활성층 / 04 프로파일 / 05 불확실성
    viz3d.run()          # 위도-깊이 단면 / hero 3D
