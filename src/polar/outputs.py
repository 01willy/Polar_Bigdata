"""산출물 경로 규칙 — 모든 시각화/결과 스크립트는 이 헬퍼로 저장 경로를 얻는다.

폴더 규칙 (docs/VISUALIZATION.md 참조):
  outputs/figures/<카테고리>/   분석 그래프(bar/hist/scatter/CV 등)
  outputs/maps/                 2D 공간 레이어(예측 ALT 지도 등) PNG/GeoTIFF
  outputs/volumes_3d/           3D 지중 열구조 렌더/메시
  outputs/animations/           GIF/MP4 (깊이 슬라이스, 계절 변화 등)
  outputs/models/               DL 체크포인트

사용:
  from polar.outputs import figpath, mappath, volpath, animpath, modelpath
  figpath("covariate_upgrade", "era5_rescore") -> outputs/figures/03_covariate_upgrade/era5_rescore.png
  mappath("alt_alaska_pred")                   -> outputs/maps/alt_alaska_pred.png
"""
import os

ROOT = "outputs"
FIG_CATS = {
    "concept": "00_concept",
    "data": "01_data",
    "eval": "02_evaluation",
    "covariate_upgrade": "03_covariate_upgrade",
    "ground_temp": "04_ground_temp",
    "alaska_pilot": "05_alaska_pilot",
    "spatial_dl": "06_deep_learning",
}


def _ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def figpath(category, name, ext="png"):
    """분석 그래프. category는 FIG_CATS 키."""
    sub = FIG_CATS.get(category, category)
    return _ensure(os.path.join(ROOT, "figures", sub, f"{name}.{ext}"))


def mappath(name, ext="png"):
    """2D 공간 레이어(예측 지도 등)."""
    return _ensure(os.path.join(ROOT, "maps", f"{name}.{ext}"))


def volpath(name, ext="png"):
    """3D 지중 열구조 렌더/메시."""
    return _ensure(os.path.join(ROOT, "volumes_3d", f"{name}.{ext}"))


def animpath(name, ext="gif"):
    """애니메이션(GIF/MP4)."""
    return _ensure(os.path.join(ROOT, "animations", f"{name}.{ext}"))


def modelpath(name, ext="pt"):
    """DL 체크포인트."""
    return _ensure(os.path.join(ROOT, "models", f"{name}.{ext}"))
