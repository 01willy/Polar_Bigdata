"""Fold-safe 전처리 — S1/S2/S3 등 공유 (코드 중복·회귀 위험 제거, 전체 리뷰 must_fix).

핵심 규약: 모든 통계(median·mean·std)는 train에서만 산출해 test에 적용(transductive 누설 방지).
tests/test_preprocessing.py가 'test 통계를 바꿔도 변환 결과 불변'을 강제한다.
"""
from __future__ import annotations
import numpy as np


def fold_prep(Xtr, Xte, nan_native: bool):
    """NaN-native 모델(GBM류)은 raw+NaN 유지. 신경망은 train median 대체 + z-score 표준화.

    Args:
        Xtr, Xte: (n, d) float 배열 (NaN 가능)
        nan_native: True면 그대로 반환(LightGBM/XGBoost/CatBoost/HistGBM)
    Returns:
        (Xtr', Xte') — nan_native면 원본, 아니면 fold-safe 대체·표준화
    """
    if nan_native:
        return Xtr, Xte
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6            # fold-safe: train 통계만
    return ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)


def train_median_fill(values, train_mask):
    """train_mask 내 유효값의 median으로 결측 대치(fold-safe). physics 공변량용."""
    v = np.asarray(values, float).copy()
    m = np.isfinite(v) & np.asarray(train_mask, bool)
    fill = np.median(v[m]) if m.any() else 0.0
    v[~np.isfinite(v)] = fill
    return v
