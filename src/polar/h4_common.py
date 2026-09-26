"""H4(논문 완성 통합 실험, docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md) 공용 통계·저장 모듈.

규약(사전 고정, 결과 열람 전)
  1. 지역 안 Δ(방법 A − 방법 B, 음수 = A 개선)의 95 % CI 는 블록 부트스트랩으로 구한다.
     각 분할 안에서 채점 블록 B 개를 복원 추출하고, 같은 재표집 인덱스를 방법 A·B 와 모든 반복·seed 에 공유한다(짝지음).
     반복 RMSE = sqrt(Σ SSE / Σ 셀 수), 반복·seed 평균 후 A − B, 분할별 분포를 같은 부트스트랩 번호끼리 평균해 결합한다.
     기존 (분할, 반복) 행 재표집 CI 는 rep_boot_ci 로 계산하여 'ci_rep' 열로 보조 병기만 한다.
  2. 모든 스크립트는 실행 단위(대상, 분할, 방법, n, 반복, seed, …)마다 채점 블록별 SSE·셀 수 벡터를 BlockStore 로 저장한다.
  3. 최소 n(주 정의): 블록 부트스트랩 CI 상한 < 0, 분할 승률(분할별 평균 Δ < 0 비율) ≥ 2/3, 반복 승률 ≥ 0.75 를 모두 만족하는 최소 n.
     미달성은 censored=True 와 n_max 를 기록한다. 순위 통계는 절단 대상을 최대 순위 동률로 처리한다.
  4. 회복률 = (물리식 − 방법) / (물리식 − 라벨 전량). 분모 ≤ 0 이면 NaN.
  5. log E 오프셋 최대우도(계층 수축)는 offset_mle_prior / offset_mle_estimate 만 사용한다.
     z = log y − 0.5 log TDD = log y − log √TDD. σ² = 지역 내 합동 분산, τ² = 지역 평균의 분산 − 평균 표본 잡음(적률, 하한 1e-4).
     원천 지역 포함 기준은 셀 ≥ 3(주), 민감도 분석은 셀 ≥ 30.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

TAU2_FLOOR = 1e-4
SPLIT_WIN_MIN = 2.0 / 3.0
REP_WIN_MIN = 0.75
MIN_BLOCKS_FLAG = 5          # 분할 안 채점 블록이 이보다 적으면 ci_flag 에 표시(계산은 수행)


def seed_of(*parts) -> int:
    """문자열 결합 CRC32 기반 재현 가능 seed(m1_stats.seed_of 와 같은 규칙)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


# ---------------------------------------------------------------- 1) 블록 SSE
def block_sse(y, pred, codes, nb):
    """채점 블록별 SSE·셀 수. y·pred·codes 는 같은 길이, codes ∈ [0, nb). 비유한 셀은 제외한다.

    반환: (sse[nb] float64, cnt[nb] int64)
    """
    y = np.asarray(y, float); pred = np.asarray(pred, float); codes = np.asarray(codes, np.int64)
    m = np.isfinite(y) & np.isfinite(pred)
    e2 = (pred[m] - y[m]) ** 2
    sse = np.bincount(codes[m], weights=e2, minlength=nb).astype(float)
    cnt = np.bincount(codes[m], minlength=nb).astype(np.int64)
    return sse, cnt


# ---------------------------------------------------------------- 2) 저장 도우미
def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def norm_key(key) -> tuple:
    """키를 JSON 직렬화 가능한 파이썬 튜플로 정규화한다(numpy 스칼라 → int/float)."""
    if not isinstance(key, (tuple, list)):
        key = (key,)
    return tuple(_py(v) for v in key)


class BlockStore:
    """작업 단위(대상, 분할) 하나의 채점 블록별 SSE·셀 수 저장소.

    block_ids: 채점 셀의 블록 식별자(셀 순서). 내부에서 np.unique 로 codes·nb 를 만든다.
    add(key, y, pred): key = (방법 문자열, n, rep, seed, lam, …) 튜플. 같은 key 를 다시 넣으면 덮어쓴다.
    """

    def __init__(self, target, split, block_ids, meta: dict | None = None):
        self.target = str(target); self.split = int(split)
        ub, codes = np.unique(np.asarray(block_ids).astype(str), return_inverse=True)
        self.blocks = ub; self.codes = codes.astype(np.int64); self.nb = int(len(ub))
        self.ncell = np.bincount(self.codes, minlength=self.nb).astype(np.int64)
        self.meta = dict(meta or {})
        self.keys: list[tuple] = []; self._idx: dict[tuple, int] = {}
        self._sse: list[np.ndarray] = []; self._cnt: list[np.ndarray] = []

    # -- 입력
    def add(self, key, y, pred):
        if self.codes is None:
            raise RuntimeError("파일에서 적재한 BlockStore 는 셀 순서가 없다. add_sse() 를 쓴다")
        sse, cnt = block_sse(y, pred, self.codes, self.nb)
        self.add_sse(key, sse, cnt)

    def add_sse(self, key, sse, cnt):
        key = norm_key(key)
        sse = np.asarray(sse, float); cnt = np.asarray(cnt, np.int64)
        assert sse.shape == (self.nb,) and cnt.shape == (self.nb,), "블록 수 불일치"
        if key in self._idx:
            i = self._idx[key]; self._sse[i] = sse; self._cnt[i] = cnt
        else:
            self._idx[key] = len(self.keys); self.keys.append(key); self._sse.append(sse); self._cnt.append(cnt)

    # -- 조회
    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return norm_key(key) in self._idx

    def get(self, key):
        i = self._idx[norm_key(key)]
        return self._sse[i], self._cnt[i]

    def select(self, key_fn) -> list[tuple]:
        """key_fn: 술어 함수, 키 목록, 또는 단일 키. 선택된 키 목록(입력 순서)."""
        return select_keys(self.keys, key_fn)

    def matrices(self, keys):
        idx = [self._idx[norm_key(k)] for k in keys]
        if not idx:
            return np.zeros((0, self.nb)), np.zeros((0, self.nb), np.int64)
        return np.stack([self._sse[i] for i in idx]), np.stack([self._cnt[i] for i in idx])

    def rmse(self, key):
        s, c = self.get(key)
        return float(np.sqrt(s.sum() / c.sum())) if c.sum() > 0 else np.nan

    def merge(self, other: "BlockStore"):
        assert (self.target, self.split) == (other.target, other.split), "다른 작업 단위는 병합할 수 없다"
        assert np.array_equal(self.blocks, other.blocks) and np.array_equal(self.ncell, other.ncell), "채점 블록 집합 불일치"
        for k in other.keys:
            s, c = other.get(k); self.add_sse(k, s, c)
        self.meta.update(other.meta)
        return self

    @classmethod
    def _from_arrays(cls, target, split, blocks, ncell, keys, sse, cnt, meta):
        st = cls.__new__(cls)
        st.target = str(target); st.split = int(split); st.blocks = np.asarray(blocks).astype(str); st.nb = int(len(st.blocks))
        st.ncell = np.asarray(ncell, np.int64); st.codes = None            # 적재본은 셀 순서가 없으므로 add() 불가, add_sse() 만 허용
        st.meta = dict(meta or {}); st.keys = []; st._idx = {}; st._sse = []; st._cnt = []
        for k, s, c in zip(keys, sse, cnt):
            st.add_sse(k, s, c)
        return st


def select_keys(keys, key_fn) -> list[tuple]:
    if callable(key_fn):
        return [k for k in keys if key_fn(k)]
    if isinstance(key_fn, (list, set)) and key_fn and isinstance(next(iter(key_fn)), (tuple, list)):
        want = {norm_key(k) for k in key_fn}
        return [k for k in keys if k in want]
    want = norm_key(key_fn)
    return [k for k in keys if k == want]


def save_stores(stores: Iterable[BlockStore], path) -> Path:
    """여러 작업 단위를 한 npz 로 저장(savez_compressed, pickle 미사용).

    형식: meta = JSON 문자열 {format:'h4_blocksse_v1', units:[{target, split, nb, n_keys, meta}]}
          u{i}_keys  = 키 JSON 문자열 배열(K,)
          u{i}_sse   = float64 (K, nb)       u{i}_cnt = int32 (K, nb)  (키별 유효 셀 수)
          u{i}_blocks = 블록 식별자 문자열 (nb,)   u{i}_ncell = int64 (nb,)  (블록별 채점 셀 수)
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stores = list(stores); arrs = {}; units = []
    for i, st in enumerate(stores):
        S, C = st.matrices(st.keys)
        arrs[f"u{i}_keys"] = np.array([json.dumps(list(k)) for k in st.keys], dtype=str)
        arrs[f"u{i}_sse"] = S.astype(np.float64); arrs[f"u{i}_cnt"] = C.astype(np.int32)
        arrs[f"u{i}_blocks"] = st.blocks.astype(str); arrs[f"u{i}_ncell"] = st.ncell
        units.append(dict(target=st.target, split=st.split, nb=st.nb, n_keys=len(st), meta=st.meta))
    arrs["meta"] = np.array(json.dumps(dict(format="h4_blocksse_v1", units=units), ensure_ascii=False, default=_py))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.replace(path)
    return path


def load_stores(paths) -> dict[tuple, BlockStore]:
    """npz 하나 또는 여러 개를 읽어 {(target, split): BlockStore} 로 병합한다(같은 단위는 키 합집합)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: dict[tuple, BlockStore] = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            for i, u in enumerate(meta["units"]):
                keys = [tuple(json.loads(s)) for s in z[f"u{i}_keys"]]
                st = BlockStore._from_arrays(u["target"], u["split"], z[f"u{i}_blocks"], z[f"u{i}_ncell"], keys,
                                             z[f"u{i}_sse"], z[f"u{i}_cnt"], u.get("meta"))
                k = (st.target, st.split)
                out[k] = out[k].merge(st) if k in out else st
    return out


def merge_stores(stores: Iterable[BlockStore]) -> dict[tuple, BlockStore]:
    out: dict[tuple, BlockStore] = {}
    for st in stores:
        k = (st.target, st.split)
        out[k] = out[k].merge(st) if k in out else st
    return out


def stores_for_target(stores: dict[tuple, BlockStore], target) -> dict[int, BlockStore]:
    """{(target, split): store} → 한 대상의 {split: store}(분할 순서 정렬)."""
    return {sp: st for (t, sp), st in sorted(stores.items(), key=lambda kv: kv[0][1]) if t == str(target)}


# ---------------------------------------------------------------- 3) 블록 부트스트랩 Δ
def boot_weights(nb, nboot, seed):
    """블록 복원 추출 다중도 행렬 W (nboot, nb). W[b, j] = b번째 재표집에서 블록 j 가 뽑힌 횟수."""
    pick = np.random.RandomState(seed).randint(0, nb, size=(nboot, nb))
    W = np.zeros((nboot, nb))
    np.add.at(W, (np.repeat(np.arange(nboot), nb), pick.ravel()), 1.0)
    return W


def _rmse_rows(S, C):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(S.sum(1) / C.sum(1))


def _beq_rows(S, C):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sqrt(S / C)
    return np.nanmean(np.where(C > 0, r, np.nan), axis=1)


def _group_mean(keys, vals, rep_fn):
    g: dict = {}
    for k, v in zip(keys, vals):
        g.setdefault(rep_fn(k), []).append(v)
    return {r: float(np.nanmean(v)) for r, v in g.items()}


def boot_delta_blocks(store_by_split: dict, key_fn_A, key_fn_B, nboot: int = 1000, seed: int = 0,
                      rep_fn: Callable | None = None, return_dist: bool = False) -> dict:
    """분할 안 블록 부트스트랩으로 Δ = RMSE(A) − RMSE(B) 의 점 추정·95 % CI 를 구한다.

    store_by_split: {split: BlockStore} (한 대상). key_fn_A/B: 방법 A/B 에 속하는 키 선택(술어·키 목록·단일 키).
    rep_fn: 키 → 반복 식별자(같은 식별자의 키 = seed 들, 먼저 평균). 기본은 키 자체.
            A 반복과 같은 식별자의 B 반복이 있으면 반복 승률을 짝지어 계산하고, 없으면 B 평균과 비교한다.
    각 분할 s 에서 W_s = boot_weights(nb_s, nboot, seed_of(seed, s)) 를 A·B 모든 키에 공유한다.
    반환 dict
      delta        = 분할 평균( 반복·seed 평균 RMSE_A − 반복·seed 평균 RMSE_B ), 셀 가중
      ci_lo, ci_hi = 분할별 부트스트랩 분포를 같은 번호끼리 평균한 분포의 2.5·97.5 백분위
      p_boot       = 양측 부트스트랩 p
      delta_split  = {split: Δ_s},  split_win = mean(Δ_s < 0),  rep_win = mean(반복별 Δ < 0)(분할 전체 반복 풀링)
      delta_beq, ci_lo_beq, ci_hi_beq = 블록 등가중 RMSE 기반 같은 계산
      rmse_A, rmse_B, n_splits, n_blocks(분할별 목록), n_keys_A, n_keys_B, ci_flag
      dist, dist_beq (return_dist=True 일 때)
    """
    rep_fn = rep_fn or (lambda k: k)
    d_split, db_split, dists, dists_b, rA, rB, nbs, rep_d = {}, {}, [], [], [], [], [], []
    nkA = nkB = 0
    for sp, st in sorted(store_by_split.items()):
        kA, kB = st.select(key_fn_A), st.select(key_fn_B)
        if not kA or not kB:
            continue
        SA, CA = st.matrices(kA); SB, CB = st.matrices(kB)
        nkA += len(kA); nkB += len(kB); nbs.append(st.nb)
        a_full, b_full = _rmse_rows(SA, CA), _rmse_rows(SB, CB)
        d_split[sp] = float(np.nanmean(a_full) - np.nanmean(b_full))
        db_split[sp] = float(np.nanmean(_beq_rows(SA, CA)) - np.nanmean(_beq_rows(SB, CB)))
        rA.append(float(np.nanmean(a_full))); rB.append(float(np.nanmean(b_full)))
        # 반복 승률
        ga, gb = _group_mean(kA, a_full, rep_fn), _group_mean(kB, b_full, rep_fn)
        b_mean = float(np.nanmean(b_full))
        for r, v in ga.items():
            rep_d.append(v - gb.get(r, b_mean))
        if nboot > 0:
            W = boot_weights(st.nb, nboot, seed_of(seed, sp))
            with np.errstate(invalid="ignore", divide="ignore"):
                bA = np.sqrt((SA @ W.T) / (CA @ W.T)); bB = np.sqrt((SB @ W.T) / (CB @ W.T))      # (K, nboot)
                dists.append(np.nanmean(bA, 0) - np.nanmean(bB, 0))
                vA = np.where(CA > 0, np.sqrt(SA / np.where(CA > 0, CA, 1)), 0.0); mA = (CA > 0).astype(float)
                vB = np.where(CB > 0, np.sqrt(SB / np.where(CB > 0, CB, 1)), 0.0); mB = (CB > 0).astype(float)
                eA = (vA @ W.T) / (mA @ W.T); eB = (vB @ W.T) / (mB @ W.T)
                dists_b.append(np.nanmean(eA, 0) - np.nanmean(eB, 0))
    if not d_split:
        return dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan, delta_split={}, split_win=np.nan, rep_win=np.nan,
                    delta_beq=np.nan, ci_lo_beq=np.nan, ci_hi_beq=np.nan, rmse_A=np.nan, rmse_B=np.nan, n_splits=0,
                    n_blocks=[], n_keys_A=0, n_keys_B=0, ci_flag="no keys")
    dv = np.array(list(d_split.values()))
    out = dict(delta=float(dv.mean()), delta_split=d_split, split_win=float(np.mean(dv < 0)),
               rep_win=float(np.mean(np.array(rep_d) < 0)) if rep_d else np.nan,
               delta_beq=float(np.mean(list(db_split.values()))), rmse_A=float(np.mean(rA)), rmse_B=float(np.mean(rB)),
               n_splits=len(d_split), n_blocks=nbs, n_keys_A=nkA, n_keys_B=nkB,
               ci_flag=f"blocks<{MIN_BLOCKS_FLAG}" if min(nbs) < MIN_BLOCKS_FLAG else "")
    if dists:
        dist = np.mean(dists, 0); dist_b = np.mean(dists_b, 0)
        out.update(ci_lo=float(np.nanpercentile(dist, 2.5)), ci_hi=float(np.nanpercentile(dist, 97.5)), p_boot=boot_p(dist),
                   ci_lo_beq=float(np.nanpercentile(dist_b, 2.5)), ci_hi_beq=float(np.nanpercentile(dist_b, 97.5)))
        if return_dist:
            out.update(dist=dist, dist_beq=dist_b)
    else:
        out.update(ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan, ci_lo_beq=np.nan, ci_hi_beq=np.nan)
    return out


def boot_p(dist) -> float:
    dist = np.asarray(dist, float); dist = dist[np.isfinite(dist)]
    if not len(dist):
        return np.nan
    p = 2.0 * min(np.mean(dist <= 0), np.mean(dist >= 0))
    return float(min(1.0, max(p, 1.0 / len(dist))))


def rep_boot_ci(d, nboot: int = 1000, seed: int = 1):
    """보조 'ci_rep': (분할, 반복) 행 Δ 의 평균을 행 재표집한 95 % CI(h25 방식). 행 < 3 이면 NaN."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 3:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    bs = d[rng.randint(0, len(d), size=(nboot, len(d)))].mean(1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def recovery(phys, method, full) -> float:
    """회복률 = (물리식 − 방법)/(물리식 − 라벨 전량). 분모 ≤ 0 이면 NaN."""
    den = float(phys) - float(full)
    return float((float(phys) - float(method)) / den) if np.isfinite(den) and den > 0 else np.nan


# ---------------------------------------------------------------- 4) 최소 n
def min_n(curve_df: pd.DataFrame, group_cols=None, n_col="n", ci_hi_col="ci_hi", split_win_col="split_win",
          rep_win_col="rep_win", split_win_min=SPLIT_WIN_MIN, rep_win_min=REP_WIN_MIN, n_max=None):
    """주 정의 최소 n: ci_hi < 0 & split_win ≥ 2/3 & rep_win ≥ 0.75 를 만족하는 최소 n.

    group_cols 가 없으면 dict(n_star, censored, n_max, n_tested) 하나, 있으면 그룹별 DataFrame.
    미달성: n_star = NaN, censored = True, n_max = 검사한 최대 n(또는 인자 n_max).
    """
    def one(sub):
        sub = sub.sort_values(n_col)
        ok = (sub[ci_hi_col] < 0) & (sub[split_win_col] >= split_win_min - 1e-12) & (sub[rep_win_col] >= rep_win_min - 1e-12)
        nm = int(n_max) if n_max is not None else int(sub[n_col].max())
        if ok.any():
            return dict(n_star=float(sub.loc[ok, n_col].iloc[0]), censored=False, n_max=nm, n_tested=int(len(sub)))
        return dict(n_star=np.nan, censored=True, n_max=nm, n_tested=int(len(sub)))
    if not group_cols:
        return one(curve_df)
    rows = []
    for g, sub in curve_df.groupby(list(group_cols), dropna=False):
        g = g if isinstance(g, tuple) else (g,)
        rows.append({**dict(zip(group_cols, g)), **one(sub)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5) 절단 순위 통계
def rank_with_censoring(values, censored=None):
    """평균 순위. 절단(censored=True) 값은 관측값 전부보다 큰 것으로 보고 서로 최대 순위 동률로 둔다. NaN(비절단)은 NaN."""
    from scipy.stats import rankdata
    v = np.asarray(values, float).copy()
    c = np.zeros(len(v), bool) if censored is None else np.asarray(censored, bool)
    v[c] = np.inf
    ok = np.isfinite(v) | c
    r = np.full(len(v), np.nan)
    r[ok] = rankdata(v[ok], method="average")
    return r


def _cens_pair(x, y, censored_x, censored_y):
    rx, ry = rank_with_censoring(x, censored_x), rank_with_censoring(y, censored_y)
    m = np.isfinite(rx) & np.isfinite(ry)
    return rx[m], ry[m]


def spearman_censored(x, y, censored_y=None, censored_x=None):
    """절단 반영 Spearman ρ(순위의 Pearson, 동률 평균 순위). 반환 (rho, p, n)."""
    from scipy.stats import spearmanr
    rx, ry = _cens_pair(x, y, censored_x, censored_y)
    if len(rx) < 3:
        return (np.nan, np.nan, int(len(rx)))
    rho, p = spearmanr(rx, ry)
    return (float(rho), float(p), int(len(rx)))


def kendall_censored(x, y, censored_y=None, censored_x=None):
    """절단 반영 Kendall τ-b(절단 대상끼리 동률). 반환 (tau, p, n)."""
    from scipy.stats import kendalltau
    rx, ry = _cens_pair(x, y, censored_x, censored_y)
    if len(rx) < 3:
        return (np.nan, np.nan, int(len(rx)))
    tau, p = kendalltau(rx, ry)
    return (float(tau), float(p), int(len(rx)))


def _logit_irls(x, yb, iters=100, tol=1e-10):
    X = np.c_[np.ones(len(x)), x]; b = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30); p = 1.0 / (1.0 + np.exp(-eta)); w = p * (1 - p) + 1e-12
        H = X.T @ (X * w[:, None]); g = X.T @ (yb - p)
        step = np.linalg.solve(H, g); b = b + step
        if np.max(np.abs(step)) < tol:
            break
    eta = np.clip(X @ b, -30, 30); p = 1.0 / (1.0 + np.exp(-eta)); w = p * (1 - p) + 1e-12
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    return b, np.sqrt(np.diag(cov))


def achieved_test(achieved, x) -> dict:
    """달성(bool) 여부와 연속 지표 x 의 관계.

    Mann-Whitney U(달성 군 x 대 미달성 군 x, 양측; 동률 없고 소표본이면 정확, 아니면 근사)와
    로지스틱 회귀 계수(x 표준화 전 원척도, Wald p). 완전 분리이면 separation=True, 계수 NaN.
    """
    from scipy.stats import mannwhitneyu, norm
    a = np.asarray(achieved, bool); x = np.asarray(x, float)
    m = np.isfinite(x); a, x = a[m], x[m]
    x1, x0 = x[a], x[~a]
    out = dict(n=int(len(x)), n_achieved=int(a.sum()), median_achieved=float(np.median(x1)) if len(x1) else np.nan,
               median_not=float(np.median(x0)) if len(x0) else np.nan, U=np.nan, p_mw=np.nan, mw_method="",
               logit_coef=np.nan, logit_se=np.nan, logit_p=np.nan, separation=False)
    if len(x1) and len(x0):
        ties = len(np.unique(x)) < len(x)
        meth = "exact" if (not ties and len(x) <= 40) else "asymptotic"
        res = mannwhitneyu(x1, x0, alternative="two-sided", method=meth)
        out.update(U=float(res.statistic), p_mw=float(res.pvalue), mw_method=meth)
        sep = (x1.min() > x0.max()) or (x1.max() < x0.min())
        if sep:
            out["separation"] = True
        else:
            try:
                b, se = _logit_irls(x, a.astype(float))
                out.update(logit_coef=float(b[1]), logit_se=float(se[1]), logit_p=float(2 * norm.sf(abs(b[1] / se[1]))))
            except np.linalg.LinAlgError:
                out["separation"] = True
    return out


# ---------------------------------------------------------------- 6) log E 오프셋 최대우도(계층 수축)
def _z(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.log(y) - np.log(s)
    return z[np.isfinite(z)]


def offset_z(df: pd.DataFrame, y_col="y", s_col="s"):
    """z = log y − log √TDD (= log y − 0.5 log TDD). s_col 은 √TDD 열(없으면 e5_sqrt_tdd)."""
    s_col = s_col if s_col in df.columns else "e5_sqrt_tdd"
    return _z(df[y_col].values, df[s_col].values)


def offset_mle_prior(src_df: pd.DataFrame, region_col="macro", min_cells=3, y_col="y", s_col="s", logE0=None) -> dict:
    """원천 라벨 셀로 log E 계층 사전을 적률 추정한다.

    지역 j(셀 ≥ min_cells): z̄_j, s²_j(ddof=1), n_j.
    σ² = Σ(n_j−1)s²_j / Σ(n_j−1) (지역 내 합동 분산).
    τ² = Var_j(z̄_j)(ddof=1) − mean_j(σ²/n_j), 하한 TAU2_FLOOR.
    logE0 = 지역 평균 z̄_j 의 비가중 평균(인자로 주면 그 값, 예: log 원천 최소제곱 E0).
    주 분석 min_cells=3, 민감도 min_cells=30. 지역 < 2 이면 ValueError.
    """
    s_col_ = s_col if s_col in src_df.columns else "e5_sqrt_tdd"
    regs, zb, s2, nn = [], [], [], []
    for r, sub in src_df.groupby(region_col):
        z = _z(sub[y_col].values, sub[s_col_].values)
        if len(z) >= max(int(min_cells), 2):
            regs.append(str(r)); zb.append(float(z.mean())); s2.append(float(z.var(ddof=1))); nn.append(len(z))
    if len(regs) < 2:
        raise ValueError(f"offset_mle_prior: 셀 ≥ {min_cells} 지역이 {len(regs)}개로 τ² 추정 불가")
    zb, s2, nn = np.array(zb), np.array(s2), np.array(nn, float)
    sigma2 = float(((nn - 1) * s2).sum() / (nn - 1).sum())
    tau2_raw = float(zb.var(ddof=1) - np.mean(sigma2 / nn))
    tau2 = max(tau2_raw, TAU2_FLOOR)
    m0 = float(zb.mean()) if logE0 is None else float(logE0)
    return dict(logE0=m0, tau2=tau2, tau2_raw=tau2_raw, sigma2=sigma2, n_regions=len(regs), regions=regs,
                region_means=dict(zip(regs, zb.tolist())), region_n=dict(zip(regs, nn.astype(int).tolist())), min_cells=int(min_cells))


def offset_mle_estimate(z_target_n, prior: dict) -> dict:
    """대상 라벨 n 개의 z 로 log E 사후(정규-정규 켤레).

    정밀도 = 1/τ² + n/σ², 사후 평균 = (logE0/τ² + Σz/σ²)/정밀도, 수축 가중 w = n/(n + σ²/τ²).
    반환: logE_post, logE_sd(사후 sd), E(= exp(사후 평균), 사후 중앙값), E_mean(= exp(m + v/2)), E_sd(로그정규 sd), w, n.
    n = 0 이면 사전(logE0, √τ²) 그대로.
    """
    z = np.asarray(z_target_n, float); z = z[np.isfinite(z)]; n = len(z)
    t2, s2, m0 = float(prior["tau2"]), float(prior["sigma2"]), float(prior["logE0"])
    prec = 1.0 / t2 + n / s2
    m = (m0 / t2 + z.sum() / s2) / prec; v = 1.0 / prec
    E_mean = float(np.exp(m + v / 2)); E_sd = float(E_mean * np.sqrt(np.expm1(v)))
    return dict(logE_post=float(m), logE_sd=float(np.sqrt(v)), E=float(np.exp(m)), E_mean=E_mean, E_sd=E_sd,
                w=float(n / (n + s2 / t2)), n=int(n))


def n_theory(prior: dict, eps: float) -> float:
    """계획서 B2 의 이론 필요 라벨 수 n_theory = σ²/(τ²·ε) (ε = 허용 log 오차)."""
    return float(prior["sigma2"] / (prior["tau2"] * eps))
