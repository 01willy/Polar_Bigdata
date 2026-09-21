"""M1 확장(S-D 상호작용·개정 09-21 §F) — 자기훈련 대조·편향 물리 용량-반응·거리 버퍼·표본 가중.

실행 중인 하네스(m1_core.py의 축 목록)를 건드리지 않기 위해 별도 모듈로 둔다. m1_master_factorial.py는
--configs JSON에 아래 키가 있을 때만 이 모듈을 쓴다.
  pseudo = "self"            : 알래스카(학습 실측) 적합 모델의 A블록 예측을 유사라벨로(Lee 2013 자기훈련 대조, H16)
  pseudo = "self_ridge14"    : 지형·기후 14종 ridge 예측(저용량 자기훈련)
  pseudo = "stefan_biased"   : ỹ = c·E√TDD + ε, 구성 키 bias_c(배율)·bias_sigma(cm)
  buffer_km                  : A 풀에서 어떤 B(평가)셀과도 buffer_km 이내인 셀 제외(C1-E 공간 보간 성분 분리)
  pseudo_mode = "weight"     : 복제 대신 고유 A행 1회 + 표본 가중 w = r·n_src/|A| (감사: 복제≠가중 검증)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from polar.fidelity import TERRAIN, CLIMATE, TARGET
from polar.preprocessing import fold_prep

EXT_PSEUDOS = {"self", "self_ridge14", "stefan_biased"}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def buffer_filter(df: pd.DataFrame, A_idx: np.ndarray, B_idx: np.ndarray, d_km: float) -> np.ndarray:
    """A 셀 중 어떤 B 셀과도 d_km 이내인 셀을 제외한 인덱스(라벨 미사용, 좌표만)."""
    if d_km <= 0 or len(B_idx) == 0:
        return A_idx
    la, lo = df.lat.values[A_idx], df.lon.values[A_idx]
    lb, lob = df.lat.values[B_idx], df.lon.values[B_idx]
    keep = np.ones(len(A_idx), bool)
    for i in range(len(A_idx)):
        d = haversine_km(la[i], lo[i], lb, lob)
        if d.min() < d_km:
            keep[i] = False
    return A_idx[keep]


def self_pseudo(kind: str, train_real: pd.DataFrame, pool: pd.DataFrame, feats: list, seed: int) -> np.ndarray:
    """자기훈련 유사라벨: 학습 실측만으로 모델을 1회 적합해 풀(A블록) 셀을 예측. 라벨 미사용."""
    from polar.m1_core import fit_model
    if kind == "self_ridge14":
        f14 = list(TERRAIN + CLIMATE)
        Xtr, Xp = fold_prep(train_real[f14].values.astype(np.float32), pool[f14].values.astype(np.float32), nan_native=False)
        return fit_model("ridge", Xtr, train_real[TARGET].values.astype(float), Xp, seed)["pred"]
    Xtr, Xp = fold_prep(train_real[feats].values.astype(np.float32), pool[feats].values.astype(np.float32), nan_native=True)
    return fit_model("catboost_lo", Xtr, train_real[TARGET].values.astype(float), Xp, seed)["pred"]


def biased_pseudo(pool: pd.DataFrame, k: dict, c: float, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """편향 물리 용량-반응: ỹ = c·E√TDD + N(0, σ²). c=1, σ=0 이면 현행 Stefan 유사라벨."""
    st = k["E"] * pool["e5_sqrt_tdd"].values.astype(float)
    return c * st + (rng.normal(0.0, sigma, len(st)) if sigma > 0 else 0.0)


def weighted_rows(pool: pd.DataFrame, r: float, n_src: int):
    """복제 대신 가중: 고유 A행 1회, 가중 r·n_src/|A| (총가중은 복제와 동일)."""
    w = float(r * n_src / max(len(pool), 1))
    return pool, np.full(len(pool), w)
