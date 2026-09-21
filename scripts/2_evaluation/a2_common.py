"""A2 감사 대안 설명 검증 공용 유틸(CPU 전용). 신규 스크립트 a2_*.py가 공유한다.

정보 없음 전이 규약(scripts/3_deep_learning/m1_master_factorial.py와 동일):
  학습 = 알래스카 실측(dset=alaska)만, 평가 = 대상 지역 전체 셀 중 y·CCI·토양 도일 유효 셀(eval_mask).
  계수(E, CCI 보정 등)는 알래스카 실측에서만 적합. 예측 = anchor + λ·g(x); anchor=none 이면 직접 회귀(λ=1).
  ridge: 알래스카 중앙값 대치 + z-score(fold_prep) / catboost_lo: 결측 원값 유지. seed 0·1·2 평균.
학습 자료가 대상 지역에 의존하지 않으므로(알래스카만) 모델은 (모델, 앵커, seed)당 한 번만 적합하고 전 대상 셀을 예측한다.
CatBoost thread_count=4(하네스는 6). 재현 검증은 repro_check()로 m1_gate noinfo 값과 대조한다.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import (TARGET, TRANSFER_MAIN, TRANSFER_DEEP, SHARED_CORE, TERRAIN, CLIMATE, SOIL,  # noqa: E402
                            spatial_block_splits)
from polar.preprocessing import fold_prep                                                                # noqa: E402
from polar.m1_core import INPUT_SETS, NAN_NATIVE, load_base, eval_mask, fit_coefs, anchor_pred          # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "m1"
REGIONS = list(TRANSFER_MAIN) + list(TRANSFER_DEEP)
SEEDS = [0, 1, 2]
NBOOT = 400
SMALL_N = 10          # 평가 셀 수가 이보다 작으면 참고 값(지역 수준 주장 불가)으로 표기
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


# ---------------------------------------------------------------- 자료
class Base:
    """fidelity_base_v3 + 토양 도일. 알래스카 학습 집합·계수·평가 셀 마스크."""

    def __init__(self, sources=("F4_direct", "F4_calm_temp"), eval_all: bool = False, min_cells: int = 3,
                 regions=None, main_direct_only: bool = True):
        self.sources = tuple(sources)
        self.df = load_base(PROC, sources=self.sources)
        self.ev = self.df[TARGET].notna().values if eval_all else eval_mask(self.df)
        if main_direct_only:   # 주 전이 6지역은 하네스 규약(--sources F4_direct)대로 탐침 셀만 평가(캐나다 747 = m1_gate)
            self.ev &= ~(np.isin(self.df.macro.values, list(TRANSFER_MAIN)) & (self.df.source_id.values != "F4_direct"))
        self.ak = np.where(self.df.macro.values == "Alaska")[0]
        self.train = self.df.iloc[self.ak]
        self.k = fit_coefs(self.train)
        cand = regions or REGIONS
        self.regions = [r for r in cand if len(self.eval_idx(r)) >= min_cells]
        self.skipped = [r for r in cand if r not in self.regions]

    def eval_idx(self, tg: str) -> np.ndarray:
        return np.where((self.df.macro.values == tg) & self.ev)[0]

    def te(self, tg: str) -> pd.DataFrame:
        return self.df.iloc[self.eval_idx(tg)]

    def y(self, tg: str) -> np.ndarray:
        return self.te(tg)[TARGET].values.astype(float)

    def blocks(self, tg: str) -> np.ndarray:
        return self.te(tg).block.values

    def anchor(self, kind: str, tg: str) -> np.ndarray:
        return np.asarray(anchor_pred(kind, self.te(tg), self.k), float)

    def describe(self) -> str:
        return (f"[data] {len(self.df):,}셀 · 알래스카 학습 {len(self.ak):,} · E={self.k['E']:.4f} · "
                f"sources={self.sources} · 대상 {self.regions} · 제외(셀 부족) {self.skipped}")


# ---------------------------------------------------------------- 지표
def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else np.nan


def bias(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(p[m] - y[m])) if m.any() else np.nan


def paired_boot(y, pA, pB, blocks, nboot: int = NBOOT, seed: int = 0):
    """ΔRMSE = RMSE(A) − RMSE(B) 점추정 + 0.5° 블록 짝지은 부트스트랩 95% CI (e1_analysis 규약).
    반환 (d0, lo, hi, n_cells, n_blocks). 블록 3개 미만이면 CI NaN."""
    y, pA, pB, blocks = (np.asarray(v) for v in (y, pA, pB, blocks))
    m = np.isfinite(y) & np.isfinite(pA.astype(float)) & np.isfinite(pB.astype(float))
    y, pA, pB, blocks = y[m].astype(float), pA[m].astype(float), pB[m].astype(float), blocks[m]
    if len(y) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    d0 = rmse(y, pA) - rmse(y, pB)
    ub = np.unique(blocks)
    if len(ub) < 3:
        return d0, np.nan, np.nan, int(len(y)), int(len(ub))
    rng = np.random.RandomState(seed)
    where = {b: np.where(blocks == b)[0] for b in ub}
    out = np.empty(nboot)
    for i in range(nboot):
        pick = rng.choice(ub, len(ub), replace=True)
        idx = np.concatenate([where[b] for b in pick])
        out[i] = rmse(y[idx], pA[idx]) - rmse(y[idx], pB[idx])
    return d0, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), int(len(y)), int(len(ub))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl, dp = np.radians(np.asarray(lon2) - np.asarray(lon1)), p2 - p1
    return 2 * R * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


# ---------------------------------------------------------------- 모델 (하네스 fit_model과 동일 하이퍼파라미터, 스레드 4)
def _fit(name: str, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0).fit(np.nan_to_num(Xtr), ytr).predict(np.nan_to_num(Xte))
    if name == "catboost_lo":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                              verbose=0, allow_writing_files=False, thread_count=4)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte))
    raise ValueError(name)


class NoinfoPredictor:
    """정보 없음 조건 예측기. 알래스카 학습 1회로 전 대상 평가 셀을 예측하고 지역별로 잘라 준다."""

    def __init__(self, base: Base, seeds=SEEDS):
        self.base, self.seeds = base, list(seeds)
        self.idx_all = np.concatenate([base.eval_idx(r) for r in base.regions])
        self.slices, s = {}, 0
        for r in base.regions:
            n = len(base.eval_idx(r)); self.slices[r] = slice(s, s + n); s += n
        self.te_all = base.df.iloc[self.idx_all]
        self._g = {}

    def g(self, model: str, anchor: str = "none", xset: str = "x25") -> np.ndarray:
        """seed 평균 g(x) (anchor=none이면 직접 예측). 전 대상 셀 순서."""
        key = (model, anchor, xset)
        if key in self._g:
            return self._g[key]
        feats = INPUT_SETS[xset]
        tr = self.base.train
        Xtr = tr[feats].values.astype(np.float32)
        Xte = self.te_all[feats].values.astype(np.float32)
        ytr = tr[TARGET].values.astype(float)
        if anchor == "none":
            target = ytr
        else:
            target = ytr - np.asarray(anchor_pred(anchor, tr, self.base.k), float)
        Xtr2, Xte2 = fold_prep(Xtr, Xte, nan_native=model in NAN_NATIVE)
        ok = np.isfinite(target)
        g = np.mean([_fit(model, Xtr2[ok], target[ok], Xte2, s) for s in self.seeds], axis=0)
        self._g[key] = g.astype(float)
        return self._g[key]

    def predict(self, tg: str, model: str, anchor: str = "none", lam: float = 1.0, xset: str = "x25") -> np.ndarray:
        g = self.g(model, anchor, xset)[self.slices[tg]]
        if anchor == "none":
            return g
        return self.base.anchor(anchor, tg) + lam * g


# ---------------------------------------------------------------- E1-e3(v2) mlp 예측 재사용 (loc_id 대응)
class E1Preds:
    """data/processed/e1_factorial_e3_shard*_preds.npz 의 정보 없음 예측(v2 자료, 알래스카 학습·같은 E)을 loc_id로 대응.
    키: g::cond|target|fold|seed|anchor|pseudo|r|resid, anchor::cond|target|fold|anchor, eval::cond|target|fold::loc_id."""

    def __init__(self):
        self.G, self.A, self.L = {}, {}, {}
        for f in sorted(glob.glob(str(PROC / "e1_factorial_e3_shard*_preds.npz"))):
            z = np.load(f, allow_pickle=False)
            for k in z.files:
                kind, rest = k.split("::", 1)
                if kind == "g":
                    self.G[rest] = z[k]
                elif kind == "anchor":
                    self.A[rest] = z[k]
                elif kind == "eval" and rest.endswith("::loc_id"):
                    self.L[rest.rsplit("::", 1)[0]] = z[k]
        self.ok = bool(self.G)

    def g(self, tg: str, model: str, anchor: str, loc_ids: np.ndarray):
        """seed 평균 g를 요청 loc_id 순서로 반환(없는 셀은 NaN). 예측이 전혀 없으면 None."""
        tk = f"noinfo|{tg}|0"
        if tk not in self.L:
            return None
        ks = [f"{tk}|{s}|{anchor}|none|0.0|{model}" for s in SEEDS if f"{tk}|{s}|{anchor}|none|0.0|{model}" in self.G]
        if not ks:
            return None
        gm = np.mean([self.G[k].astype(float) for k in ks], axis=0)
        pos = {l: i for i, l in enumerate(self.L[tk])}
        out = np.full(len(loc_ids), np.nan)
        for i, l in enumerate(loc_ids):
            if l in pos:
                out[i] = gm[pos[l]]
        return out

    def anchor(self, tg: str, anchor: str, loc_ids: np.ndarray):
        tk = f"noinfo|{tg}|0"
        k = f"{tk}|{anchor}"
        if k not in self.A:
            return None
        pos = {l: i for i, l in enumerate(self.L[tk])}
        a = self.A[k].astype(float)
        return np.array([a[pos[l]] if l in pos else np.nan for l in loc_ids])


# ---------------------------------------------------------------- 공변량 이동 도구
def standardize_on_alaska(base: Base, feats):
    """알래스카 통계로 표준화(결측은 알래스카 중앙값 대치). 반환 Z(전 셀), mu, sd."""
    X = base.df[feats].values.astype(float)
    Xa = X[base.ak]
    med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[base.ak].mean(0), X[base.ak].std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def aoa_di(base: Base, feats, n_folds: int = 6):
    """Meyer & Pebesma(2021) 비유사도 지수 DI(가중 없음).
    표준화 공변량(알래스카 평균·표준편차)에서 학습(알래스카) 최근접 유클리드 거리 / 정규화 상수.
    학습 DI는 6-fold 0.5° 공간블록 CV(다른 fold의 최근접), 정규화 상수 = 학습 CV 최근접 거리 평균,
    임계 = 학습 DI의 Q3 + 1.5·IQR. 반환 (di 전 셀, thr, di_train)."""
    from sklearn.neighbors import NearestNeighbors
    Z, _, _ = standardize_on_alaska(base, feats)
    Za = Z[base.ak]
    pos = {g: i for i, g in enumerate(base.ak)}
    d_cv = np.full(len(base.ak), np.nan)
    for tr, te in spatial_block_splits(base.df, n_splits=n_folds, sub_idx=base.ak):
        nn = NearestNeighbors(n_neighbors=1).fit(Z[tr])
        d, _ = nn.kneighbors(Z[te])
        d_cv[[pos[i] for i in te]] = d[:, 0]
    dbar = float(np.nanmean(d_cv))
    di_tr = d_cv / dbar
    q75, q25 = np.nanpercentile(di_tr, [75, 25])
    thr = float(q75 + 1.5 * (q75 - q25))
    nn_all = NearestNeighbors(n_neighbors=1).fit(Za)
    d_all, _ = nn_all.kneighbors(Z)
    di = d_all[:, 0] / dbar
    di[base.ak] = di_tr
    return di, thr, di_tr


def out_of_range(base: Base, feats) -> pd.DataFrame:
    """대상 셀 공변량이 알래스카 학습 범위[min, max] 밖인지(결측은 False). 전 셀 × feats 불리언 프레임."""
    Xa = base.train[feats].values.astype(float)
    lo, hi = np.nanmin(Xa, 0), np.nanmax(Xa, 0)
    X = base.df[feats].values.astype(float)
    with np.errstate(invalid="ignore"):
        oor = (X < lo) | (X > hi)
    oor[np.isnan(X)] = False
    return pd.DataFrame(oor, columns=feats, index=base.df.index)


# ---------------------------------------------------------------- 재현 검증 (m1_gate noinfo)
def repro_check(base: Base, pred: NoinfoPredictor, tag: str = "m1_gate"):
    """m1_gate_shard0.csv 정보 없음 행(Lena·Canada, F4_direct 평가 셀)과 seed 평균 RMSE를 같은 loc_id 집합에서 대조."""
    f = OUT / f"{tag}_shard0.csv"
    z = OUT / f"{tag}_shard0_preds.npz"
    if not f.exists() or not z.exists():
        print(f"[repro] {f.name} 없음 → 생략"); return None
    zz = np.load(z, allow_pickle=False)
    r = pd.read_csv(f); r = r[(r.cond == "noinfo") & (r.xset == "x25") & (r.pseudo == "none")]
    rows = []
    for tg in sorted(set(r.target) & set(base.regions)):
        lk = f"eval::noinfo|{tg}|0::loc_id"
        if lk not in zz.files:
            continue
        keep = np.isin(base.te(tg).loc_id.values, zz[lk])
        y = base.y(tg)[keep]
        for (anchor, model, lam), sub in r[r.target == tg].groupby(["anchor", "resid", "lam"]):
            if model not in ("ridge", "catboost_lo"):
                continue
            mine = rmse(y, pred.predict(tg, model, anchor, lam)[keep])
            rows.append(dict(target=tg, anchor=anchor, model=model, lam=lam, n=int(keep.sum()),
                             rmse_m1gate=round(sub.rmse_cm.mean(), 3), rmse_a2=round(mine, 3),
                             diff=round(mine - sub.rmse_cm.mean(), 3)))
    t = pd.DataFrame(rows)
    if len(t):
        print("[repro] m1_gate noinfo 대조 (seed 평균 RMSE, cm)")
        print(t.to_string(index=False))
        print(f"[repro] 최대 |차| = {t['diff'].abs().max():.3f} cm")
    return t


def flag_small(n: int) -> str:
    return "참고(n<10)" if n < SMALL_N else ""
