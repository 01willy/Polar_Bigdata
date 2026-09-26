"""H31 · Track A2 잔차 학습의 최소 라벨 수(확인적). CPU.

목적
  H25 S3 는 같은 n개 라벨로 E 를 수축한 뒤 잔차를 학습했으므로 "잔차에 필요한 n" 과 "E 를 건드린 해악" 이 교락되어 있었다.
  본 실험은 E 처리와 라벨 블록 배치를 요인으로 분리하여, 잔차 CatBoost 가 물리식(원천 E0)을 넘는 최소 라벨 수 n* 를 대상별로 판정한다.
  계획서 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §3(공통 설계)·§4 Track A2·§5 F1–F3·§8(산출물). 공용 규약 src/polar/h4_common.py.

요인
  E 처리(4, CatBoost 잔차 포함)
    E0_fixed     원천 최소제곱 E0 고정. 대상 라벨은 잔차 학습에만 쓴다.
    E_own_fixed  대상 A 전체 최소제곱 E_own 고정(상한 참조). 잔차 학습에는 n개만 쓴다.
    shrink_k10   κ=10 수축 E = (n·E_ls + κ·E0)/(n + κ) (h25 S1 과 같다).
    offset_mle   log E 오프셋 최대우도(계층 수축). h4_common.offset_mle_prior(원천, macro 지역 셀 ≥ 3, logE0 = log E0)
                 와 offset_mle_estimate(대상 n개의 z = log y − log √TDD) 만 쓴다. E = exp(사후 평균).
  민감도(해석적 anchor 행만, CatBoost 없음)
    offset_mle30 같은 방식, 원천 지역 포함 기준 셀 ≥ 30.
  라벨 블록 분산(3)
    concentrated 1 블록에서 n개. 셀 ≥ n 인 블록 중 하나를 무작위로 고른다.
                 셀 ≥ n 인 블록이 없으면(1블록으로 n개를 못 채우면) 셀 수 최대 블록을 전부 쓰고, 나머지를 셀 수 내림차순 블록에서
                 순차 보충한다(사용 블록 수 최소화). 이 경우 n_blocks_used > 1 이며, curve 의 conc_overflow_rate(반복 중 비율)와
                 meta 의 conc_overflow 표로 기록한다. F2(분산 − 집중) 해석은 overflow 비율을 함께 보고, 비율 0 인 (대상, n) 만의
                 민감도 결과를 병기한다.
    spread5      ⌈n/5⌉ 블록을 무작위로 골라 순환 균등 배정(n ≤ 5 이면 1블록). 선택 블록이 소진되면 나머지 블록에서 셀 수 순 보충.
    all_blocks   A 의 모든 블록을 무작위 순서로 순환 배정.
  n 격자: {3,5,10,20,40,80,160,320} 중 |A| 미만 + allA(A 전체, 키 n = −1, 실제 라벨 수는 n_lab). n0(원천 잔차만)·physics 행을 함께 저장.
  잔차 모델: CatBoost(h25 cb_fit_predict 설정), 학습 = 원천 잔차(E0 앵커) ∪ 대상 n개 잔차(해당 E 처리 앵커), α = 1, λ ∈ {0.25, 0.5}.
  예측 = E_used·√TDD + λ·g. 잔차 없음 행(stage = anchor) = E_used·√TDD.
  반복: 라벨 추출 20회(anchor 행, 해석적) 중 앞 10회에서 CatBoost(계획서 사양 "라벨 추출 10회, seed 2").
        본 실행 예산 근거: 적합 수 --count-only 산정 × 6워커 동시 적합당 0.55 s(2026-09-26 측정, load ≈ 35) ≈ 2.0–2.3 h < 3 h.

대상
  h25 와 같은 15개(주 4지역 + 하위 지역 11; 블록 중심 k-means, 라벨 미사용) + Alaska_f0(알래스카 블록 GroupKFold 2분할 fold 0,
  원천 = 알래스카 밖 전체 라벨, A = 알래스카 절반, 대규모 n 참조용; 분할 1개만). 분석 대상 14 = 15 − CA-1(|A| = 3, n 격자 없음).
  원천(LORO) = 대상 셀을 뺀 전체 라벨 셀에서 대상 경계 --buffer-km 이내 셀 제외. 분할 = A/B 블록 2분할(split 0..K−1), 채점 = B 의 eval_mask 셀.

채점(h4_common 규약)
  Δ = 방법 − 물리식(E0), 음수 = 개선. 실행 단위(대상, 분할, stage, E 처리, 분산, scope, n, rep, seed, λ)마다 채점 블록별 SSE·셀 수를
  <tag>_blocksse.npz(BlockStore) 로 저장한다. 지역 안 95 % CI = h4_common.boot_delta_blocks(분할 안 채점 블록 재표집 1,000회,
  모든 방법·반복·seed 에 같은 재표집 인덱스, seed 평균 → 반복 평균, 분할별 분포를 같은 번호끼리 평균). (분할, 반복) 행 재표집 CI 는 ci_rep
  (d_phys_lo_rep·d_phys_hi_rep) 로 보조 병기. 셀 가중 RMSE(주)·블록 등가중 RMSE(보조, _beq) 병기.
  잔차 순가치 d_noresid = resid − 같은 (E 처리, 분산, scope, n) 의 anchor 행, 같은 블록 부트스트랩. anchor 키는 resid 가 가진 반복
  (rep < reps_cb)으로 제한해 같은 라벨 추출끼리 짝짓는다. 잔차 분해(F3)용으로 anchor 를 rep < reps_cb 로 제한한 Δ 를
  d_phys_mean_repcb·d_phys_lo_repcb·d_phys_hi_repcb 로 병기한다(resid 행은 주 값과 같다).
  최소 n*(주) = h4_common.min_n: 블록 CI 상한 < 0 & 분할 승률 ≥ 2/3 & 반복 승률 ≥ 0.75 인 최소 n. 미달성은 censored = True, n_max 기록.
  회복률(보조) = h4_common.recovery(분모 ≤ 0 이면 NaN). 분산 대 집중 대비(F2)는 <tag>_spread.csv(대상별 + 분석 14 대상 평균).

입력  data/processed/fidelity_base_v3.csv + e5_soil_tdd_v3.csv(load_base 병합), 입력 x25, 열 e5_sqrt_tdd.
출력  data/processed/h4/<tag>_runs.csv(행 단위), <tag>_blocksse.npz(블록 SSE), <tag>_curve.csv(집계·CI), <tag>_minn.csv(최소 n*),
      <tag>_spread.csv(F2 대비), <tag>_targets.csv(대상 지표), <tag>_meta.json. 스모크는 <tag>_smoke_* 로 저장.
실행(ROOT, CPU)
  스모크: python3 scripts/3_deep_learning/h31_min_labels_residual.py --smoke --workers 2
  본 실행: python3 scripts/3_deep_learning/h31_min_labels_residual.py --workers 6 --tag a2
  재집계: python3 scripts/3_deep_learning/h31_min_labels_residual.py --summarize-only --tag a2
  중단 후 이어서: python3 scripts/3_deep_learning/h31_min_labels_residual.py --workers 6 --tag a2 --resume
  적합 수 산정: python3 scripts/3_deep_learning/h31_min_labels_residual.py --count-only
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import functools
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET                                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402
import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import (BlockStore, save_stores, load_stores, stores_for_target,  # noqa: E402
                             boot_delta_blocks, rep_boot_ci, min_n, recovery, offset_mle_prior, offset_mle_estimate, seed_of)

# 같은 (nb, nboot, seed) 의 재표집 행렬을 재사용한다(값은 동일, 계산만 줄인다).
H4.boot_weights = functools.lru_cache(maxsize=1024)(H4.boot_weights)

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="", help="쉼표 목록(기본: 주 4지역 + 하위 지역 11 + Alaska_f0)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,5,10,20,40,80,160,320")
ap.add_argument("--reps", type=int, default=20, help="라벨 추출 반복(해석적 anchor 행)")
ap.add_argument("--reps-cb", type=int, default=10, help="CatBoost 잔차 반복(앞에서부터; 계획서 사양 10)")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--lams", default="0.25,0.5")
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
ap.add_argument("--min-cells-prior", type=int, default=3, help="offset_mle 사전의 원천 macro 지역 최소 셀 수(주, h4_common 규약)")
ap.add_argument("--min-cells-sens", type=int, default=30, help="offset_mle30 민감도 사전의 최소 셀 수")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="a2")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--resume", action="store_true", help="저장된 runs·blocksse 의 완료 작업 단위를 건너뛴다")
ap.add_argument("--ckpt-every", type=int, default=6, help="완료 작업 단위 몇 개마다 중간 저장할지")
ap.add_argument("--count-only", action="store_true", help="CatBoost 를 돌리지 않고 적합 수만 센다")
ap.add_argument("--summarize-only", action="store_true", help="저장된 runs·blocksse 로 집계만 다시")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
SPLITS = list(range(args.splits)) if not args.smoke else [0, 1]
N_GRID = [int(v) for v in args.n_grid.split(",")] if not args.smoke else [3, 20]
REPS = args.reps if not args.smoke else 3
REPS_CB = min(args.reps_cb, REPS) if not args.smoke else 2
SEEDS = list(range(args.seeds)) if not args.smoke else [0, 1]
LAMS = [float(v) for v in args.lams.split(",")]
NBOOT = args.nboot
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
ALASKA_FOLD = "Alaska_f0"
E_TREATS = ["E0_fixed", "E_own_fixed", "shrink_k10", "offset_mle"]
E_SENS = ["offset_mle30"]                      # 해석적 anchor 행만
SPREADS = ["concentrated", "spread5", "all_blocks"]
TAG = args.tag + ("_smoke" if args.smoke else "")
# 키 = (stage, e_treat, spread, scope, n, rep, seed, lam)
KI = dict(stage=0, e_treat=1, spread=2, scope=3, n=4, rep=5, seed=6, lam=7)
PHYS_KEY = ("physics", "E0_fixed", "none", "n0", 0, -1, -1, 0.0)
REP_FN = lambda k: k[5]                        # noqa: E731  반복 식별자(같은 rep 의 seed 는 먼저 평균)

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)
# z = log y − log √TDD (h4_common.offset_z 와 같은 정의; 비유한 값은 offset_mle_estimate 에서 제외)
with np.errstate(invalid="ignore", divide="ignore"):
    DF["z"] = np.log(DF.y.values) - np.log(DF.s.values)


# ---------------------------------------------------------------- 하위 지역(h25 와 동일 정의: 블록 중심 좌표 k-means, 라벨 미사용)
def make_subregions(df):
    from sklearn.cluster import KMeans
    sub = np.array(["" for _ in range(len(df))], dtype=object)
    rows = []
    for spec in args.k_sub.split(","):
        reg, k = spec.split(":"); k = int(k)
        idx = np.where(df.macro.values == reg)[0]
        bt = df.iloc[idx].groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size")).reset_index()
        X = np.c_[bt.lat.values, bt.lon.values * np.cos(np.radians(bt.lat.values))]
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=np.sqrt(bt.n.values))
        lab = km.labels_
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])
        name_of = {c: f"{reg[:2].upper()}-{r + 1}" for r, c in enumerate(order)}
        b2s = dict(zip(bt.block.values, [name_of[c] for c in lab]))
        for i in idx:
            sub[i] = b2s[df.block.values[i]]
        for c in range(k):
            m = lab == c
            rows.append(dict(subregion=name_of[c], parent=reg, n_blocks=int(m.sum()), n_cells=int(bt.n.values[m].sum()),
                             lat=float(bt.lat.values[m].mean()), lon=float(bt.lon.values[m].mean())))
    return sub, pd.DataFrame(rows)


DF["sub"], SUBS = make_subregions(DF)
_ref = PROC / "h3" / "subregions.csv"
if _ref.exists():                                          # h25 정의와 일치 확인(다른 담당 파일은 수정하지 않음)
    _r = pd.read_csv(_ref).set_index("subregion").n_cells
    _m = SUBS.set_index("subregion").n_cells
    if not _r.reindex(_m.index).equals(_m):
        print("[warn] 하위 지역 정의가 h3/subregions.csv 와 다르다", flush=True)
TARGETS = args.targets.split(",") if args.targets else MAIN4 + sorted(SUBS.subregion.tolist()) + [ALASKA_FOLD]
if args.smoke and not args.targets:
    TARGETS = ["Russia_W", "AL-3", ALASKA_FOLD]
EXCLUDE_ANALYSIS = {"CA-1", ALASKA_FOLD}                    # CA-1: |A| = 3(n 격자 없음), Alaska_f0: 대규모 n 참조
if __name__ == "__main__":
    print(f"[data] {len(DF):,}셀 · 대상 {len(TARGETS)}: {TARGETS} · 분할 {SPLITS} · n {N_GRID} · reps {REPS}/{REPS_CB} · seeds {SEEDS}", flush=True)


def target_idx(t):
    if t == ALASKA_FOLD:
        return np.where(DF.macro.values == "Alaska")[0]
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def parent_of(t):
    if t == ALASKA_FOLD:
        return "Alaska"
    return t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])


def splits_of(t):
    return [0] if t == ALASKA_FOLD else SPLITS


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def cb_fit_predict(Xtr, ytr, Xte, seed, w=None):
    if args.count_only:
        return np.zeros(len(Xte))
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr, sample_weight=w)
    return np.asarray(m.predict(Xte), float)


# ---------------------------------------------------------------- 라벨 블록 분산 규칙
def select_spread(spread, n, rng, pools_by_block):
    """pools_by_block: 블록 → A 안 국소 인덱스 배열. 반환 (선택 인덱스, 사용 블록 수).
    concentrated: 셀 ≥ n 인 블록 중 하나를 무작위 선택. 없으면 셀 수 최대 블록을 전부 쓰고 나머지 블록을 셀 수 내림차순으로 순차 보충.
    spread5: ⌈n/5⌉ 블록을 무작위 선택 후 순환 배정(모두 소진되면 나머지 블록을 셀 수 순으로 보충).
    all_blocks: 모든 블록을 무작위 순서로 순환 배정."""
    ub = list(pools_by_block.keys()); sizes = np.array([len(pools_by_block[b]) for b in ub]); nb = len(ub)
    by_size = [int(j) for j in np.argsort(-sizes, kind="stable")]
    if spread == "concentrated":
        elig = np.where(sizes >= n)[0]
        primary = [int(rng.choice(elig))] if len(elig) else [by_size[0]]
    elif spread == "spread5":
        k = min(int(np.ceil(n / 5)), nb)
        primary = [int(j) for j in rng.choice(nb, k, replace=False)]
    elif spread == "all_blocks":
        primary = [int(j) for j in rng.permutation(nb)]
    else:
        raise ValueError(spread)
    fallback = [j for j in by_size if j not in primary]
    pools = {j: list(rng.permutation(pools_by_block[ub[j]])) for j in range(nb)}
    sel, owner = [], []
    active = list(primary)
    while len(sel) < n and active:                          # 순환 배정
        nxt = []
        for j in active:
            if len(sel) >= n:
                break
            if pools[j]:
                sel.append(int(pools[j].pop())); owner.append(j); nxt.append(j)
        active = nxt
    for j in fallback:                                       # 순차 보충(사용 블록 수 최소화)
        while len(sel) < n and pools[j]:
            sel.append(int(pools[j].pop())); owner.append(j)
        if len(sel) >= n:
            break
    return np.array(sel[:n], int), int(len(set(owner[:n])))


def hetero_metrics(t_idx, src_idx):
    """라벨 미사용 이질성 지표 + 라벨 기반 진단 지표(h25 와 동일)."""
    Xt = DF.iloc[t_idx][FEATS].values.astype(float); Xs = DF.iloc[src_idx][FEATS].values.astype(float)
    mt, ms = np.nanmean(Xt, 0), np.nanmean(Xs, 0); sd_s = np.nanstd(Xs, 0) + 1e-9
    smd = float(np.nanmean(np.abs(mt - ms) / sd_s))
    d = DF.iloc[t_idx]; y, s = d.y.values, d.s.values
    ratio = y / np.where(s > 0, s, np.nan)
    g = pd.DataFrame(dict(b=d.block.values, r=ratio)).groupby("b").r
    Eb = g.mean(); nb = g.size()
    return dict(n_cells=int(len(t_idx)), n_blocks=int(d.block.nunique()), smd_x25=smd, E_own_all=ls_E(y, s),
                E_block_cv=float(np.nanstd(Eb[nb >= 3]) / np.nanmean(Eb[nb >= 3])) if (nb >= 3).sum() >= 2 else np.nan,
                y_sd=float(np.nanstd(y)), y_mean=float(np.nanmean(y)))


def source_of(t_idx):
    src_idx = np.where(np.isfinite(DF.y.values))[0]
    src_idx = src_idx[~np.isin(src_idx, t_idx)]
    if args.buffer_km > 0:                                    # 대상 경계 버퍼 제외(좌표만)
        la, lo = DF.lat.values, DF.lon.values
        keep = np.ones(len(src_idx), bool); tl, tn = la[t_idx], lo[t_idx]
        for j, i in enumerate(src_idx):
            if abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                keep[j] = False
        src_idx = src_idx[keep]
    return src_idx


def priors_of(src, E0):
    """offset_mle(주, 셀 ≥ 3)·offset_mle30(민감도, 셀 ≥ 30) 사전. 사전 평균 = log E0(원천 최소제곱). 지역 < 2 이면 None."""
    out = {}
    for name, mc in (("offset_mle", args.min_cells_prior), ("offset_mle30", args.min_cells_sens)):
        try:
            out[name] = offset_mle_prior(src, region_col="macro", min_cells=mc, y_col="y", s_col="s", logE0=float(np.log(E0)))
        except ValueError:
            out[name] = None
    return out


# ---------------------------------------------------------------- 작업 단위(대상 × 분할)
def run_task(t, sp):
    t0 = time.time()
    t_idx = target_idx(t); parent = parent_of(t)
    src_idx = source_of(t_idx); src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    pri = priors_of(src, E0)
    p1, p30 = pri["offset_mle"], pri["offset_mle30"]
    X_src = src[FEATS].values.astype(np.float32); y_src = src.y.values; s_src = src.s.values
    r_src = y_src - E0 * s_src; ok = np.isfinite(r_src); XR_src, R_src = X_src[ok], r_src[ok]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]
    nA = len(A)
    yA, sA, zA, XA = A.y.values, A.s.values, A.z.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    E_own = ls_E(yA, sA)
    bA = A.block.values
    pools_by_block = {b: np.where(bA == b)[0] for b in np.unique(bA)}
    max_block_A = int(max(len(v) for v in pools_by_block.values()))
    st = BlockStore(t, sp, B.block.values, meta=dict(parent=parent, n_A=nA, n_blocks_A=len(pools_by_block), E0=E0, E_own=E_own))
    nbB = st.nb
    rows, n_fit = [], 0
    base = dict(target=t, parent=parent, split=sp, n_A=nA, n_blocks_A=int(len(pools_by_block)), n_eval=len(evB), n_blocks_eval=nbB,
                n_src=int(len(src_idx)), E0=E0, E_own=E_own)
    conc_over = {}

    def add(stage, e_treat, spread, scope, n, n_lab, rep, seed, lam, pred, E_used, nb_used):
        key = (stage, e_treat, spread, scope, int(n), int(rep), int(seed), float(lam))
        st.add(key, yB, pred)
        sse, cnt = st.get(key)
        with np.errstate(invalid="ignore", divide="ignore"):
            beq = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan)))
        rows.append(dict(**base, stage=stage, e_treat=e_treat, spread=spread, scope=scope, n=int(n), n_lab=int(n_lab), rep=int(rep),
                         seed=int(seed), lam=float(lam), rmse_cm=float(np.sqrt(sse.sum() / cnt.sum())), rmse_beq_cm=beq,
                         bias_cm=float(np.nanmean(pred - yB)), E_used=float(E_used), n_blocks_used=int(nb_used)))

    def E_of(e_treat, sel):
        n = len(sel)
        if e_treat == "E0_fixed":
            return E0
        if e_treat == "E_own_fixed":
            return E_own
        if e_treat == "shrink_k10":
            E_ls = ls_E(yA[sel], sA[sel]) if n else np.nan
            return (n * E_ls + args.kappa * E0) / (n + args.kappa) if np.isfinite(E_ls) else E0
        if e_treat in ("offset_mle", "offset_mle30"):
            prior = p1 if e_treat == "offset_mle" else p30
            return offset_mle_estimate(zA[sel], prior)["E"] if prior is not None else E0
        raise ValueError(e_treat)

    def resid_fit(sel, E_used, seed):
        if len(sel) == 0:
            return cb_fit_predict(XR_src, R_src, XB, seed)
        r_n = yA[sel] - E_used * sA[sel]
        return cb_fit_predict(np.vstack([XR_src, XA[sel]]), np.concatenate([R_src, r_n]), XB, seed)

    # n = 0: 물리식·원천 잔차만(E0 앵커, E_own 앵커)
    add(*PHYS_KEY[:5], 0, -1, -1, 0.0, E0 * sB, E0, 0)
    g0 = {}
    for seed in SEEDS:
        g0[seed] = resid_fit(np.array([], int), E0, seed); n_fit += 1
    for e_treat, E_used in (("E0_fixed", E0), ("E_own_fixed", E_own)):
        add("anchor", e_treat, "none", "n0", 0, 0, -1, -1, 0.0, E_used * sB, E_used, 0)
        for seed in SEEDS:
            for lam in LAMS:
                add("resid", e_treat, "none", "n0", 0, 0, -1, seed, lam, E_used * sB + lam * g0[seed], E_used, 0)

    def evaluate(sel, nb_used, n_key, rep, spread, scope, do_cb):
        nonlocal n_fit
        for e_treat in E_TREATS + E_SENS:
            E_used = E_of(e_treat, sel)
            add("anchor", e_treat, spread, scope, n_key, len(sel), rep, -1, 0.0, E_used * sB, E_used, nb_used)
            if not do_cb or e_treat in E_SENS:
                continue
            for seed in SEEDS:
                g = resid_fit(sel, E_used, seed); n_fit += 1
                for lam in LAMS:
                    add("resid", e_treat, spread, scope, n_key, len(sel), rep, seed, lam, E_used * sB + lam * g, E_used, nb_used)

    evaluate(np.arange(nA), len(pools_by_block), -1, -1, "all", "allA", do_cb=True)
    for n in N_GRID:
        if n >= nA:
            continue
        for spread in SPREADS:
            for rep in range(REPS):
                rng = np.random.RandomState(seed_of("h31", t, sp, n, spread, rep))
                sel, nb_used = select_spread(spread, n, rng, pools_by_block)
                if spread == "concentrated":
                    conc_over.setdefault(int(n), []).append(int(nb_used > 1))
                evaluate(sel, nb_used, n, rep, spread, "n", do_cb=rep < REPS_CB)
    het = hetero_metrics(t_idx, src_idx)
    meta = dict(target=t, parent=parent, split=sp, n_A=nA, n_blocks_A=int(len(pools_by_block)), max_block_A=max_block_A,
                n_eval=len(evB), n_blocks_eval=nbB, n_src=int(len(src_idx)), E0=E0, E_own=E_own,
                tau2=p1["tau2"] if p1 else np.nan, tau2_raw=p1["tau2_raw"] if p1 else np.nan, sigma2=p1["sigma2"] if p1 else np.nan,
                n_regions_prior=p1["n_regions"] if p1 else 0, prior_regions="|".join(p1["regions"]) if p1 else "",
                tau2_30=p30["tau2"] if p30 else np.nan, sigma2_30=p30["sigma2"] if p30 else np.nan, n_regions_prior30=p30["n_regions"] if p30 else 0,
                logE_ratio_own=float(np.log(E_own / E0)) if E_own > 0 else np.nan,
                conc_overflow=json.dumps({k: float(np.mean(v)) for k, v in sorted(conc_over.items())}),
                n_fit=n_fit, elapsed_s=round(time.time() - t0, 1), **het)
    st.meta.update(dict(n_fit=n_fit))
    return rows, meta, st


# ---------------------------------------------------------------- 집계
def _group_index(st):
    """저장소 키 → 곡선 그룹 (e_treat, spread, scope, n, lam, stage) 별 키 목록."""
    idx = {}
    for k in st.keys:
        if k[0] == "physics":
            continue
        idx.setdefault((k[1], k[2], k[3], k[4], k[7], k[0]), []).append(k)
    return idx


def _union(by_split_idx, g):
    out, seen = [], set()
    for idx in by_split_idx.values():
        for k in idx.get(g, []):
            if k not in seen:
                seen.add(k); out.append(k)
    return out


def summarize(runs, stores, metas):
    r = runs.copy()
    phys = r[r.stage == "physics"].set_index(["target", "split"])
    r = r[r.stage != "physics"].copy()
    key_ts = list(zip(r.target, r.split))
    r["d_phys"] = r.rmse_cm.values - phys.rmse_cm.loc[key_ts].values
    r["win"] = (r.d_phys < 0).astype(float)
    r["multi_block"] = (r.n_blocks_used > 1).astype(float)
    K = ["target", "parent", "split", "e_treat", "spread", "scope", "n", "rep"]
    g = r.groupby(K + ["lam", "stage"], as_index=False).agg(                        # seed 평균(짝지음)
        rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"), d_phys=("d_phys", "mean"),
        win=("win", "mean"), E_used=("E_used", "mean"), n_blocks_used=("n_blocks_used", "mean"), multi_block=("multi_block", "mean"),
        n_lab=("n_lab", "mean"), n_seed=("seed", "nunique"))
    eo = g[g.stage == "anchor"].set_index(K).rmse_cm
    g["d_noresid_rep"] = g.rmse_cm.values - eo.reindex(pd.MultiIndex.from_frame(g[K])).values
    g.loc[g.stage == "anchor", "d_noresid_rep"] = 0.0
    C = ["target", "parent", "e_treat", "spread", "scope", "n", "lam", "stage"]
    curve = g.groupby(C, as_index=False).agg(
        rmse_mean_rows=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), d_phys_rows=("d_phys", "mean"), win_rate=("win", "mean"),
        E_used_mean=("E_used", "mean"), E_used_sd=("E_used", "std"), n_blocks_used_mean=("n_blocks_used", "mean"),
        conc_overflow_rate=("multi_block", "mean"), n_lab_mean=("n_lab", "mean"), n_runs=("rmse_cm", "size"), n_splits=("split", "nunique"),
        n_reps=("rep", "nunique"))
    curve.loc[curve.spread != "concentrated", "conc_overflow_rate"] = np.nan
    n_eval = runs.groupby(["target", "split"]).n_eval.first().groupby("target").mean()
    curve["n_eval_mean"] = curve.target.map(n_eval).values
    grp_rows = {k: sub for k, sub in g.groupby(C)}
    res = []
    spread_rows, spread_dists = [], {}
    for t in curve.target.unique():
        by_split = stores_for_target(stores, t)
        bidx = {sp: _group_index(st) for sp, st in by_split.items()}
        bseed = seed_of("a2boot", t)
        for row in curve[curve.target == t].itertuples(index=False):
            gkey = (row.e_treat, row.spread, row.scope, int(row.n), float(row.lam), row.stage)
            kA = _union(bidx, gkey)
            rp = boot_delta_blocks(by_split, kA, [PHYS_KEY], nboot=NBOOT, seed=bseed, rep_fn=REP_FN)
            sub = grp_rows[tuple(getattr(row, c) for c in C)]
            lo_r, hi_r = rep_boot_ci(sub.d_phys.values, nboot=NBOOT, seed=seed_of("a2rep", *gkey, t))
            out = dict(rmse_mean=rp["rmse_A"], rmse_phys=rp["rmse_B"], d_phys_mean=rp["delta"], d_phys_lo=rp["ci_lo"], d_phys_hi=rp["ci_hi"],
                       p_boot=rp["p_boot"], split_win=rp["split_win"], rep_win=rp["rep_win"], d_phys_beq_mean=rp["delta_beq"],
                       d_phys_beq_lo=rp["ci_lo_beq"], d_phys_beq_hi=rp["ci_hi_beq"], ci_flag=rp["ci_flag"],
                       n_blocks_eval_min=int(min(rp["n_blocks"])) if rp["n_blocks"] else 0,
                       d_phys_split=json.dumps({str(k): round(v, 4) for k, v in rp["delta_split"].items()}),
                       d_phys_lo_rep=lo_r, d_phys_hi_rep=hi_r)
            # 잔차 분해용 보조: CatBoost 가 돈 반복(rep < REPS_CB)만 쓴 Δ(anchor 20 반복 중 앞 REPS_CB 개; resid 는 주 값과 같다)
            kA_cb = [k for k in kA if k[KI["rep"]] < REPS_CB]
            rc = rp if len(kA_cb) == len(kA) else (boot_delta_blocks(by_split, kA_cb, [PHYS_KEY], nboot=NBOOT, seed=bseed, rep_fn=REP_FN)
                                                   if kA_cb else None)
            out.update(d_phys_mean_repcb=rc["delta"] if rc else np.nan, d_phys_lo_repcb=rc["ci_lo"] if rc else np.nan,
                       d_phys_hi_repcb=rc["ci_hi"] if rc else np.nan, rmse_mean_repcb=rc["rmse_A"] if rc else np.nan,
                       n_reps_repcb=len({k[KI["rep"]] for k in kA_cb}))
            if row.stage == "resid":
                # 반복 짝지음: anchor 키를 resid 가 가진 반복 집합으로 제한한다(라벨 추출 의존 E 처리에서 다른 추출이 섞이지 않게)
                reps_A = {k[KI["rep"]] for k in kA}
                kB = [k for k in _union(bidx, (row.e_treat, row.spread, row.scope, int(row.n), 0.0, "anchor")) if k[KI["rep"]] in reps_A]
                rn = boot_delta_blocks(by_split, kA, kB, nboot=NBOOT, seed=bseed, rep_fn=REP_FN)
                lo_n, hi_n = rep_boot_ci(sub.d_noresid_rep.values, nboot=NBOOT, seed=seed_of("a2repnr", *gkey, t))
                out.update(d_noresid_mean=rn["delta"], d_noresid_lo=rn["ci_lo"], d_noresid_hi=rn["ci_hi"], split_win_nr=rn["split_win"],
                           rep_win_nr=rn["rep_win"], d_noresid_lo_rep=lo_n, d_noresid_hi_rep=hi_n)
            else:
                out.update(d_noresid_mean=0.0, d_noresid_lo=np.nan, d_noresid_hi=np.nan, split_win_nr=np.nan, rep_win_nr=np.nan,
                           d_noresid_lo_rep=np.nan, d_noresid_hi_rep=np.nan)
            res.append(out)
        # F2 대비: 같은 (E 처리, n, λ, stage) 에서 분산 − 집중(블록 부트스트랩, 반복 번호 짝지음은 독립 추출이므로 임의 짝)
        ct = curve[(curve.target == t) & (curve.scope == "n") & (curve.spread == "concentrated")]
        n_sp_t = len(by_split)
        for row in ct.itertuples(index=False):
            kB = _union(bidx, (row.e_treat, "concentrated", "n", int(row.n), float(row.lam), row.stage))
            for alt in ("spread5", "all_blocks"):
                kA = _union(bidx, (row.e_treat, alt, "n", int(row.n), float(row.lam), row.stage))
                if not kA:
                    continue
                rs = boot_delta_blocks(by_split, kA, kB, nboot=NBOOT, seed=bseed, rep_fn=REP_FN, return_dist=True)
                full = rs["n_splits"] == n_sp_t
                spread_rows.append(dict(target=t, parent=row.parent, contrast=f"{alt}-concentrated", e_treat=row.e_treat, n=int(row.n),
                                        lam=float(row.lam), stage=row.stage, delta=rs["delta"], ci_lo=rs["ci_lo"], ci_hi=rs["ci_hi"],
                                        p_boot=rs["p_boot"], split_win=rs["split_win"], rep_win=rs["rep_win"], delta_beq=rs["delta_beq"],
                                        n_splits=rs["n_splits"], all_splits=bool(full), conc_overflow_rate=float(row.conc_overflow_rate),
                                        ci_flag=rs["ci_flag"]))
                if full and t not in EXCLUDE_ANALYSIS and "dist" in rs:
                    spread_dists.setdefault((f"{alt}-concentrated", row.e_treat, int(row.n), float(row.lam), row.stage), []).append(
                        (t, rs["delta"], rs["dist"], float(row.conc_overflow_rate)))
    curve = pd.concat([curve.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    # 회복률(보조): 같은 (target, e_treat, lam, stage) 의 allA 대비, 분모 ≤ 0 이면 NaN
    allA = curve[curve.scope == "allA"].set_index(["target", "e_treat", "lam", "stage"]).rmse_mean
    curve["recovery"] = [recovery(rw.rmse_phys, rw.rmse_mean, allA.get((rw.target, rw.e_treat, rw.lam, rw.stage), np.nan))
                         for rw in curve.itertuples(index=False)]
    curve["etreat"] = curve.e_treat                                                   # h4_analysis 호환 별칭
    # 분석 14 대상 평균(F2): 대상별 블록 부트스트랩 분포를 같은 번호끼리 평균(대상 층화)
    for (con, e_treat, n, lam, stage), lst in spread_dists.items():
        dist = np.mean([d for _, _, d, _ in lst], 0)
        clean = [x for x in lst if x[3] == 0]
        row = dict(target="MEAN_ANALYSIS", parent="", contrast=con, e_treat=e_treat, n=n, lam=lam, stage=stage,
                   delta=float(np.mean([x[1] for x in lst])), ci_lo=float(np.nanpercentile(dist, 2.5)), ci_hi=float(np.nanpercentile(dist, 97.5)),
                   p_boot=H4.boot_p(dist), split_win=np.nan, rep_win=np.nan, target_win=float(np.mean([x[1] < 0 for x in lst])), delta_beq=np.nan,
                   n_splits=np.nan, all_splits=True, conc_overflow_rate=float(np.mean([x[3] for x in lst])), ci_flag="", n_targets=len(lst),
                   targets="|".join(x[0] for x in lst))
        if len(clean) >= 2:                                                           # overflow 0 대상만(민감도)
            dc = np.mean([d for _, _, d, _ in clean], 0)
            row.update(delta_nooverflow=float(np.mean([x[1] for x in clean])), ci_lo_nooverflow=float(np.nanpercentile(dc, 2.5)),
                       ci_hi_nooverflow=float(np.nanpercentile(dc, 97.5)), n_targets_nooverflow=len(clean))
        spread_rows.append(row)
    spread = pd.DataFrame(spread_rows)
    if "n_targets" in spread:
        spread.loc[spread.target != "MEAN_ANALYSIS", "n_targets"] = 1
    # 최소 n*: 주 정의(h4_common.min_n), 분할 일부에만 있는 n 은 제외
    G = ["target", "parent", "e_treat", "spread", "lam", "stage"]
    cn = curve[curve.scope == "n"].copy()
    cn = cn[cn.n_splits == cn.groupby(G).n_splits.transform("max")]
    mn = min_n(cn, group_cols=G, n_col="n", ci_hi_col="d_phys_hi", split_win_col="split_win", rep_win_col="rep_win")
    cr = cn[cn.stage == "resid"]
    if len(cr):
        mnr = min_n(cr, group_cols=G, n_col="n", ci_hi_col="d_noresid_hi", split_win_col="split_win_nr", rep_win_col="rep_win_nr")
        mnr = mnr.rename(columns=dict(n_star="n_star_noresid", censored="censored_noresid"))[G + ["n_star_noresid", "censored_noresid"]]
        mn = mn.merge(mnr, on=G, how="left")
    aux = []
    for key, sub in cn.groupby(G):
        sub = sub.sort_values("n")
        first = lambda m: float(sub.n[m].iloc[0]) if m.any() else np.nan            # noqa: E731
        t, e_treat, lam, stage = key[0], key[2], key[4], key[5]
        fa = curve[(curve.scope == "allA") & (curve.target == t) & (curve.e_treat == e_treat) & (curve.lam == lam) & (curve.stage == stage)]
        aux.append(dict(zip(G, key), minn_ci_rep=first(sub.d_phys_hi_rep < 0), minn_ci_only=first(sub.d_phys_hi < 0),
                        minn_rec50=first((sub.recovery >= 0.5) & (sub.win_rate >= 0.75)), minn_point=first(sub.d_phys_mean < 0),
                        n_splits=int(sub.n_splits.max()), best_n=int(sub.loc[sub.rmse_mean.idxmin(), "n"]), best_d_phys=float(sub.d_phys_mean.min()),
                        d_phys_at_nmax=float(sub.d_phys_mean.iloc[-1]), d_phys_hi_at_nmax=float(sub.d_phys_hi.iloc[-1]),
                        d_phys_allA=float(fa.d_phys_mean.iloc[0]) if len(fa) else np.nan,
                        d_phys_hi_allA=float(fa.d_phys_hi.iloc[0]) if len(fa) else np.nan))
    if len(mn):
        mn = mn.merge(pd.DataFrame(aux), on=G, how="left")
        tm = pd.DataFrame(metas).groupby("target").agg(logE_ratio_own=("logE_ratio_own", "mean"), n_A_mean=("n_A", "mean"))
        mn["logE_ratio_own"] = mn.target.map(tm.logE_ratio_own); mn["abs_logE_ratio"] = mn.logE_ratio_own.abs()
        mn["struct_region"] = mn.abs_logE_ratio < 0.15                                 # F1 구조 오차 지역 정의(|log E비| < 0.15)
        mn["analysis"] = ~mn.target.isin(EXCLUDE_ANALYSIS)
    return curve, mn, spread


def write_outputs(runs, stores, metas, elapsed, extra):
    curve, mn, spread = summarize(runs, stores, metas)
    curve.to_csv(OUT / f"{TAG}_curve.csv", index=False); mn.to_csv(OUT / f"{TAG}_minn.csv", index=False)
    spread.to_csv(OUT / f"{TAG}_spread.csv", index=False)
    pd.DataFrame(metas).to_csv(OUT / f"{TAG}_targets.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        commit = "NA"
    mt = pd.DataFrame(metas)
    conc = {f"{r.target}|{r.split}": json.loads(r.conc_overflow) for r in mt.itertuples()} if "conc_overflow" in mt else {}
    meta = dict(stage="H31/A2", plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 Track A2", common="src/polar/h4_common.py",
                args=vars(args), tag=TAG, targets=TARGETS, exclude_analysis=sorted(EXCLUDE_ANALYSIS),
                splits=SPLITS, n_grid=N_GRID, reps=REPS, reps_cb=REPS_CB, seeds=SEEDS, lams=LAMS, e_treats=E_TREATS, e_sens=E_SENS,
                spreads=SPREADS, kappa=args.kappa, buffer_km=args.buffer_km, nboot=NBOOT, key_fields=list(KI), phys_key=list(PHYS_KEY),
                ci_rule="h4_common.boot_delta_blocks(분할 안 채점 블록 재표집, 모든 방법·반복·seed 같은 인덱스, seed→반복 평균, 분할 분포 평균)",
                minn_rule="h4_common.min_n: d_phys_hi < 0 & split_win ≥ 2/3 & rep_win ≥ 0.75, 미달성 censored",
                offset_prior=f"h4_common.offset_mle_prior(macro, min_cells={args.min_cells_prior}, logE0=log E0), 민감도 min_cells={args.min_cells_sens}",
                conc_rule="셀 ≥ n 블록 무작위 1개, 없으면 최대 블록 + 셀 수 내림차순 순차 보충(n_blocks_used > 1 로 기록)",
                noresid_rule="d_noresid: anchor 키를 resid 의 반복 집합(rep < reps_cb)으로 제한해 짝지음. F3 분해는 *_repcb 열 사용",
                conc_overflow=conc, rep_budget=("reps_cb=10(계획서 사양). 적합 수(--count-only) × 0.55 s(6워커 동시, load≈35 측정) / 6 "
                                                "≈ 2.0–2.3 h < 3 h. 예산 초과 시 --reps-cb 8 로 낮춘다"),
                git_commit=commit, elapsed_s=round(elapsed, 1), n_rows=int(len(runs)), n_curve=int(len(curve)), n_minn=int(len(mn)),
                n_spread=int(len(spread)), n_fit_total=int(sum(m["n_fit"] for m in metas)), **extra)
    (OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    show = mn[(mn.stage == "resid") & (mn.lam == LAMS[0])] if len(mn) else mn
    if len(show):
        print(show[["target", "e_treat", "spread", "n_star", "censored", "n_max", "n_star_noresid", "minn_ci_rep", "best_n", "best_d_phys",
                    "d_phys_allA"]].to_string(index=False))
    print(f"saved {TAG}_* · runs {len(runs)} · curve {len(curve)} · minn {len(mn)} · spread {len(spread)} · {elapsed:.0f}s", flush=True)


def _checkpoint(frames, stores, metas):
    pd.concat(frames, ignore_index=True).to_csv(OUT / f"{TAG}_runs.csv", index=False)
    save_stores(list(stores.values()), OUT / f"{TAG}_blocksse.npz")
    pd.DataFrame(metas).to_csv(OUT / f"{TAG}_targets.csv", index=False)


def _task_cost(t, sp):
    """작업 비용 추정(긴 작업 먼저 배정): |A| 미만 n 격자 수."""
    A_idx, _ = half_split_blocks(DF, target_idx(t), sp)
    return sum(n < len(A_idx) for n in N_GRID)


def main():
    t0 = time.time()
    tasks = [(t, sp) for t in TARGETS for sp in splits_of(t)]
    if args.summarize_only:
        runs = pd.read_csv(OUT / f"{TAG}_runs.csv")
        stores = load_stores(OUT / f"{TAG}_blocksse.npz")
        metas = pd.read_csv(OUT / f"{TAG}_targets.csv").to_dict("records")
        write_outputs(runs, stores, metas, time.time() - t0, dict(summarize_only=True))
        return
    frames, metas, stores = [], [], {}
    if args.resume and (OUT / f"{TAG}_runs.csv").exists() and (OUT / f"{TAG}_blocksse.npz").exists():
        old = pd.read_csv(OUT / f"{TAG}_runs.csv"); stores = load_stores(OUT / f"{TAG}_blocksse.npz")
        done = set(stores) & set(zip(old.target.astype(str), old.split.astype(int)))
        old = old[[(a, b) in done for a, b in zip(old.target.astype(str), old.split.astype(int))]]
        stores = {k: v for k, v in stores.items() if k in done}
        frames.append(old)
        if (OUT / f"{TAG}_targets.csv").exists():
            metas = [m for m in pd.read_csv(OUT / f"{TAG}_targets.csv").to_dict("records") if (str(m["target"]), int(m["split"])) in done]
        tasks = [(t, sp) for t, sp in tasks if (t, sp) not in done]
        print(f"[resume] 완료 {len(done)} 작업 단위 적재, 남은 {len(tasks)}", flush=True)
    tasks.sort(key=lambda ts: -_task_cost(*ts))
    n_done = [0]

    def _done(r, m, st):
        frames.append(pd.DataFrame(r)); metas.append(m); stores[(st.target, st.split)] = st
        n_done[0] += 1
        print(f"  [{m['target']}|{m['split']}] 적합 {m['n_fit']} · A {m['n_A']}/{m['n_blocks_A']}블록(최대 {m['max_block_A']}) · 채점 {m['n_eval']}"
              f"/{m['n_blocks_eval']}블록 · 원천 {m['n_src']} · E0 {m['E0']:.3f} · E_own {m['E_own']:.3f} · τ² {m['tau2']:.4f} σ² {m['sigma2']:.4f}"
              f"(지역 {m['n_regions_prior']}) · 집중 overflow {m['conc_overflow']} · {m['elapsed_s']}s · 완료 {n_done[0]}/{len(tasks)} · 누적 {time.time()-t0:.0f}s",
              flush=True)
        if not args.count_only and n_done[0] % max(args.ckpt_every, 1) == 0:
            _checkpoint(frames, stores, metas)
    if args.workers <= 1 or args.count_only:
        for t, sp in tasks:
            _done(*run_task(t, sp))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:   # fork 후 OpenMP 충돌 회피
            futs = {ex.submit(run_task, t, sp): (t, sp) for t, sp in tasks}
            for f in as_completed(futs):
                _done(*f.result())
    if args.count_only:
        tot = sum(m["n_fit"] for m in metas)
        print(f"[count-only] 작업 {len(tasks)} · 총 적합 {tot:,} · 6워커·적합당 0.55/0.7/1.0 s 가정 시 "
              f"{tot*0.55/6/3600:.2f}/{tot*0.7/6/3600:.2f}/{tot/6/3600:.2f} h")
        pd.DataFrame(metas).to_csv(OUT / f"{TAG}_count_targets.csv", index=False)
        return
    _checkpoint(frames, stores, metas)
    runs = pd.concat(frames, ignore_index=True)
    write_outputs(runs, stores, metas, time.time() - t0, dict(count_only=False, resumed=bool(args.resume)))


if __name__ == "__main__":
    main()
