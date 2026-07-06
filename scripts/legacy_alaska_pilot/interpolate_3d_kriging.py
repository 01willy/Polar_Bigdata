#!/usr/bin/env python3
"""⑤ Baseline 3D 보간: borehole MAGT → 3D 체적 + 0 °C 등온면 + frozen body mesh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polar import interpolate, model  # noqa: E402

if __name__ == "__main__":
    interpolate.run()   # 크리깅 baseline 체적
    model.run()         # 공변량 RF + leave-one-borehole-out CV
