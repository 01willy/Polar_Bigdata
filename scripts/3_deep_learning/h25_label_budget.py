"""H25·H26 라벨 예산 프로토콜. 대상 15(주 4 + 하위 11), 분석 14(CA-1 제외) × 라벨 n × 선택 규칙 × 단계(E 수축, 증강, 잔차 ML, 대상 행 가중 α). CPU.

계획 docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md §2–§3, 재실행(감사 반영) docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §3.
전신 m1_sparse_label_curve.py(S-F)·h24_obs_design.py(H24).

대상: 주 4지역(레나·캐나다·러시아 W·E) + 하위 지역(알래스카 k=6·캐나다 k=3·레나 k=2; 블록 중심 k-means, 라벨 미사용) → data/processed/h3/subregions.csv
원천(LORO): 대상 셀을 뺀 전체 F4 라벨 셀에서 대상 경계 --buffer-km 이내 셀 제외. 앵커 E0 = 원천 최소제곱.
      --exclude-parent: 하위 지역 대상은 같은 상위 macro 지역 셀을 원천에서 전부 제외한다(지역 간 전이 민감도, 기본 대상 = 하위 지역만).
      대상별 원천 구성(상위 지역 셀 수·비율, 버퍼 제외 수)은 targets CSV 에 기록한다.
분할: 대상 블록 A/B 2분할(split 0..K-1). 풀 = A, 채점 = B(실측·CCI·토양 도일 유효).
규칙: random · kmedoid(표준화 x25 k-means 중심 최근접 셀).  n 격자: |A| 미만 값만.
단계(누적): S1 E 수축(κ=10, n=0 → E0) · S2 S1 + 유사라벨 증강 직접 CatBoost(A 풀에 E_n·√TDD 유사라벨 r=10, 라벨 셀은 실측 대체)
          · S3 S1 앵커 + 잔차 CatBoost(원천 잔차(E0 앵커) ∪ 대상 n개 잔차(E_n 앵커), 대상 행 가중 α, λ)
n=0: physics(E0), S2(순수 증강 = 공변량만 레시피), S3(원천 잔차만). 라벨 전량(allA): 같은 단계를 A 전체로, n = -1('all')로 통일.
--no-s2: S2 적합 생략(S1·S1refit·S3 만).
통계(h4_common 규약): 실행 단위(대상, 분할, 단계, 규칙, 범위, n, 반복, seed, λ, α)마다 채점 블록별 SSE·셀 수를 <tag>_blocksse.npz 에 저장한다.
  d_phys_lo/hi = 분할 안 채점 블록 부트스트랩(모든 방법·반복 짝지음) 95 % CI, d_phys_lo_rep/hi_rep = 기존 (분할, 반복) 행 재표집 CI(보조).
  split_win = 분할별 평균 Δ < 0 비율, rep_win = 반복별 Δ < 0 비율.
  손익분기 주 정의 breakeven_n_ci = 블록 CI 상한 < 0 · split_win ≥ 2/3 · rep_win ≥ 0.75 인 최소 n(h4_common.min_n).
  미달성은 -1 과 censored=True, n_max 로 기록한다. 보조 breakeven_n_rec50 = 회복률 ≥ 0.5 · 승률 ≥ 0.75(회복률 분모 ≤ 0 이면 NaN).
분할 구조 점검(split_structure): 블록 수가 적은 대상은 분할이 서로 같거나(A 블록 집합 동일) 채점 블록이 1개일 수 있다.
  중복 분할(dup_of ≥ 0)은 모든 집계(곡선·블록 부트스트랩)에서 한 번만 쓴다(첫 분할 유지). 거울 분할(A·B 교환)은 풀·채점 셀이 달라 유지한다.
  블록 부트스트랩 주 통계는 유효 분할(중복 아님 · 채점 블록 ≥ 2)만 결합한다(채점 블록 1개면 분할 안 부트스트랩 분산이 0 이 되어 CI 가 좁아짐).
  유효 분할이 2개 미만인 대상은 breakeven_n_ci 를 절단(-1, censored, be_flag='valid_splits<2')으로 둔다.
  민감도: 중복 제거만 한 전체 분할 결합 결과를 *_all 열(d_phys_hi_all, split_win_all, breakeven_n_ci_all)로 병기한다.
산출 data/processed/h3/<tag>_{runs,curve,breakeven,targets}.csv, <tag>_blocksse.npz, <tag>_meta.json
실행(ROOT, CPU): python3 scripts/3_deep_learning/h25_label_budget.py --workers 5 --tag h25b --alphas 1 [--smoke]
               python3 scripts/3_deep_learning/h25_label_budget.py --workers 3 --tag h25x --exclude-parent --alphas 1 --no-s2
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
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
from polar.h4_common import (BlockStore, save_stores, load_stores, merge_stores, stores_for_target,  # noqa: E402
                             boot_delta_blocks, rep_boot_ci, recovery, min_n)

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="", help="쉼표 목록(기본: 주 4지역 + 하위 지역 전부)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,5,10,20,40,80,160,320")
ap.add_argument("--reps", type=int, default=20, help="해석적 단계 반복")
ap.add_argument("--reps-cb", type=int, default=5, help="CatBoost 단계 반복(앞에서부터)")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--lams", default="0.25,0.5")
ap.add_argument("--alphas", default="1,10,100")
ap.add_argument("--r", type=float, default=10.0)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
ap.add_argument("--rules", default="random,kmedoid")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="h25")
ap.add_argument("--exclude-parent", action="store_true", help="하위 지역 대상의 원천에서 같은 상위 macro 지역 셀 전부 제외")
ap.add_argument("--no-s2", action="store_true", help="S2(증강 직접 CatBoost) 생략")
ap.add_argument("--nboot", type=int, default=1000, help="블록 부트스트랩 반복 수")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--summarize-only", action="store_true", help="저장된 runs 로 집계만 다시(대상 지표는 경량 재계산)")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h3"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
N_GRID = [int(v) for v in args.n_grid.split(",")] if not args.smoke else [3, 20]
REPS = args.reps if not args.smoke else 2
REPS_CB = args.reps_cb if not args.smoke else 1
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
LAMS = [float(v) for v in args.lams.split(",")]
ALPHAS = [float(v) for v in args.alphas.split(",")]
RULES = args.rules.split(",")
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)


# ---------------------------------------------------------------- 하위 지역(라벨 미사용: 블록 중심 좌표 k-means)
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
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])           # 북→남 이름 순서
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
SUBS.to_csv(OUT / "subregions.csv", index=False)
SUB_NAMES = sorted(SUBS.subregion.tolist())
if args.targets:
    TARGETS = args.targets.split(",")
else:                                                          # --exclude-parent 는 주 4지역에 영향이 없으므로 하위 지역만
    TARGETS = SUB_NAMES if args.exclude_parent else MAIN4 + SUB_NAMES
if args.smoke:
    TARGETS = ["LE-2", "AL-4", "CA-1"] if args.exclude_parent else ["Lena", "AL-1", "Russia_W"]
print(f"[data] {len(DF):,}셀 · 대상 {len(TARGETS)}: {TARGETS}", flush=True)


def target_idx(t):
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def parent_of(t):
    return t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])


def source_idx(t):
    """대상 t 의 원천 셀 색인과 구성 기록. 대상 셀 제외 → (--exclude-parent) 상위 지역 셀 제외 → 경계 버퍼 제외."""
    t_idx = target_idx(t); parent = parent_of(t)
    src = np.where(np.isfinite(DF.y.values))[0]
    src = src[~np.isin(src, t_idx)]
    n_par_ex = 0
    if args.exclude_parent and t not in MAIN4:
        m = DF.macro.values[src] == parent
        n_par_ex = int(m.sum()); src = src[~m]
    n_before_buf = len(src)
    if args.buffer_km > 0:                                    # 대상 경계 버퍼 제외(좌표만)
        la, lo = DF.lat.values, DF.lon.values
        keep = np.ones(len(src), bool)
        tl, tn = la[t_idx], lo[t_idx]
        for j, i in enumerate(src):
            if abs(la[i] - tl).min() < 1.0:                   # 위도 1° 안에서만 거리 계산(속도)
                if haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                    keep[j] = False
        src = src[keep]
    n_par = int((DF.macro.values[src] == parent).sum())
    comp = dict(exclude_parent=bool(args.exclude_parent and t not in MAIN4), n_src=int(len(src)), n_src_parent=n_par,
                frac_src_parent=float(n_par / max(len(src), 1)), n_parent_excluded=n_par_ex, n_buffer_excluded=int(n_before_buf - len(src)))
    return t_idx, parent, src, comp


SPLIT_INFO: dict = {}


def split_structure(t):
    """대상 t 의 분할별 구조. dup_of = A 블록 집합이 같은 앞선 분할(없으면 -1), mirror_of = A·B 가 바뀐 앞선 분할,
    nb_eval = 채점 블록 수, valid = 중복 아님 · nb_eval ≥ 2. CatBoost 없이 색인만 계산한다."""
    if t in SPLIT_INFO:
        return SPLIT_INFO[t]
    t_idx = target_idx(t); allb = frozenset(DF.block.values[t_idx]); seen, info = [], {}
    for sp in SPLITS:
        A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
        a = frozenset(DF.block.values[A_idx]); evB = B_idx[eval_mask(DF.iloc[B_idx])]
        dup = next((q for q, aq in seen if aq == a), -1)
        mir = next((q for q, aq in seen if aq == allb - a), -1)
        nb = int(DF.iloc[evB].block.nunique())
        info[sp] = dict(dup_of=int(dup), mirror_of=int(mir), nb_eval=nb, nb_A=len(a), valid=bool(dup < 0 and nb >= 2))
        seen.append((sp, a))
    n_unique = sum(v["dup_of"] < 0 for v in info.values()); n_valid = sum(v["valid"] for v in info.values())
    for v in info.values():
        v.update(n_unique_splits=int(n_unique), n_valid_splits=int(n_valid))
    SPLIT_INFO[t] = info
    return info


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def cb_fit_predict(Xtr, ytr, Xte, seed, w=None):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr, sample_weight=w)
    return np.asarray(m.predict(Xte), float)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2; s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


def select(rule, n, rng, Z):
    if rule == "random":
        return rng.choice(len(Z), n, replace=False)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)
    sel = []
    for c in km.cluster_centers_:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


def hetero_metrics(t_idx, src_idx):
    """라벨 미사용 이질성 지표 + 라벨 기반 진단 지표."""
    Xt = DF.iloc[t_idx][FEATS].values.astype(float); Xs = DF.iloc[src_idx][FEATS].values.astype(float)
    mt, ms = np.nanmean(Xt, 0), np.nanmean(Xs, 0); sd_s = np.nanstd(Xs, 0) + 1e-9
    smd = float(np.nanmean(np.abs(mt - ms) / sd_s))
    var_ratio = float(np.nanmean(np.nanvar(Xt, 0) / (np.nanvar(Xs, 0) + 1e-9)))
    d = DF.iloc[t_idx]; y, s = d.y.values, d.s.values
    ratio = y / np.where(s > 0, s, np.nan)
    g = pd.DataFrame(dict(b=d.block.values, r=ratio)).groupby("b").r
    Eb = g.mean(); nb = g.size()
    within = float(np.nanmean(g.std()[nb >= 3])) if (nb >= 3).any() else np.nan
    return dict(n_cells=int(len(t_idx)), n_blocks=int(d.block.nunique()), smd_x25=smd, var_ratio_x25=var_ratio,
                E_own=ls_E(y, s), E_block_cv=float(np.nanstd(Eb[nb >= 3]) / np.nanmean(Eb[nb >= 3])) if (nb >= 3).sum() >= 2 else np.nan,
                within_block_sd_ratio=within, y_sd=float(np.nanstd(y)), y_mean=float(np.nanmean(y)))


# ---------------------------------------------------------------- 작업 단위(대상 × 분할)
def run_task(t, sp):
    t0 = time.time()
    t_idx, parent, src_idx, comp = source_idx(t)
    src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    X_src = src[FEATS].values.astype(np.float32); y_src = src.y.values; s_src = src.s.values
    r_src = y_src - E0 * s_src; ok = np.isfinite(r_src); XR_src, R_src = X_src[ok], r_src[ok]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]
    nA = len(A)
    yA, sA, XA = A.y.values, A.s.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
    rows, n_fit = [], 0
    st = BlockStore(t, sp, B.block.values, meta=dict(n_A=int(nA), n_eval=int(len(evB)), E0=float(E0), parent=parent, **comp))
    base = dict(target=t, parent=parent, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB,
                n_src=int(len(src_idx)), E0=E0)

    def add(rule, stage, n, rep, seed, lam, alpha, pred, E_used, scope="n"):
        st.add((stage, rule, scope, int(n), int(rep), int(seed), float(lam), float(alpha)), yB, pred)   # 키 순서 = KEY_COLS
        rows.append(dict(**base, rule=rule, stage=stage, scope=scope, n=int(n), rep=int(rep), seed=int(seed), lam=float(lam), alpha=float(alpha),
                         rmse_cm=rmse(yB, pred), rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)), E_used=float(E_used)))

    n_ps = int(round(args.r * len(y_src)))
    PS_SEL = {} if args.no_s2 else {seed: np.random.RandomState(seed).choice(nA, n_ps, replace=True) for seed in SEEDS}

    def stage2(sel, E_n, seed):
        ps = PS_SEL[seed]; yps = E_n * sA[ps]
        if sel is not None and len(sel):
            m = np.isin(ps, sel); yps = yps.copy(); yps[m] = yA[ps[m]]
        return cb_fit_predict(np.vstack([X_src, XA[ps]]), np.concatenate([y_src, yps]), XB, seed)

    def stage3(sel, E_n, seed, alpha):
        if sel is None or len(sel) == 0:
            return cb_fit_predict(XR_src, R_src, XB, seed)
        r_n = yA[sel] - E_n * sA[sel]
        Xtr = np.vstack([XR_src, XA[sel]]); ytr = np.concatenate([R_src, r_n])
        w = np.concatenate([np.ones(len(R_src)), np.full(len(sel), alpha)])
        return cb_fit_predict(Xtr, ytr, XB, seed, w=w)

    # n = 0
    add("none", "physics", 0, -1, -1, 0.0, 1.0, E0 * sB, E0)
    for seed in SEEDS:
        if not args.no_s2:
            p2 = stage2(None, E0, seed); n_fit += 1; add("none", "S2", 0, -1, seed, 1.0, 1.0, p2, E0)
        g = stage3(None, E0, seed, 1.0); n_fit += 1
        for lam in LAMS:
            add("none", "S3", 0, -1, seed, lam, 1.0, E0 * sB + lam * g, E0)

    def evaluate(sel, n, rep, rule, scope, do_cb):
        nonlocal n_fit
        m_ = len(sel)                                                   # 수축에 쓰는 실제 라벨 수(allA 는 n = -1 이므로 len(sel))
        E_ls = ls_E(yA[sel], sA[sel]); E_n = (m_ * E_ls + args.kappa * E0) / (m_ + args.kappa)
        add(rule, "S1", n, rep, -1, 0.0, 1.0, E_n * sB, E_n, scope)
        add(rule, "S1refit", n, rep, -1, 0.0, 1.0, E_ls * sB, E_ls, scope)
        if not do_cb:
            return
        for seed in SEEDS:
            if not args.no_s2:
                p2 = stage2(sel, E_n, seed); n_fit += 1; add(rule, "S2", n, rep, seed, 1.0, 1.0, p2, E_n, scope)
            for alpha in ALPHAS:
                g = stage3(sel, E_n, seed, alpha); n_fit += 1
                for lam in LAMS:
                    add(rule, "S3", n, rep, seed, lam, alpha, E_n * sB + lam * g, E_n, scope)

    evaluate(np.arange(nA), -1, -1, "all", "allA", do_cb=True)          # 라벨 전량: 분할마다 |A| 가 달라도 n = -1('all')로 통일
    for n in N_GRID:
        if n >= nA:
            continue
        for rule in RULES:
            for rep in range(REPS):
                rng = np.random.RandomState(7_919 * sp + 1_009 * n + 31 * RULES.index(rule) + rep)
                sel = select(rule, n, rng, Z)
                evaluate(sel, n, rep, rule, "n", do_cb=rep < REPS_CB)
    het = hetero_metrics(t_idx, src_idx)
    return rows, dict(target=t, parent=parent, split=sp, n_A=nA, n_eval=len(evB), n_blocks_eval=nbB, E0=E0, n_fit=n_fit,
                      elapsed_s=round(time.time() - t0, 1), **comp, **het), st


# ---------------------------------------------------------------- 집계
KEY_COLS = ["stage", "rule", "scope", "n", "rep", "seed", "lam", "alpha"]      # BlockStore 키 순서
GRP_COLS = ["stage", "rule", "scope", "n", "lam", "alpha"]                     # 곡선 한 행 = 키에서 rep·seed 를 뺀 것
PHYS_KEY = ("physics", "none", "n", 0, -1, -1, 0.0, 1.0)


def _grp_of(k):
    return (k[0], k[1], k[2], int(k[3]), float(k[6]), float(k[7]))


def block_stats(stores):
    """대상별로 곡선 그룹(단계·규칙·범위·n·λ·α)마다 물리식 대비 블록 부트스트랩 Δ 를 계산한다.
    주 통계 = 유효 분할(중복 아님 · 채점 블록 ≥ 2)만 결합, 민감도(*_all) = 중복만 뺀 전체 분할 결합."""
    out = []
    targets = sorted({t for t, _ in stores})
    for t in targets:
        info = split_structure(t)
        by_all = {sp: st for sp, st in stores_for_target(stores, t).items() if info.get(int(sp), {}).get("dup_of", -1) < 0}
        by_split = {sp: st for sp, st in by_all.items() if info.get(int(sp), {}).get("valid", True)}
        groups: dict = {}
        for sp, st in by_all.items():
            for k in st.keys:
                if k[0] != "physics":
                    groups.setdefault(_grp_of(k), []).append(k)
        for grp, keys in groups.items():
            ra = boot_delta_blocks(by_all, keys, [PHYS_KEY], nboot=args.nboot, seed=0, rep_fn=lambda k: k[4])
            ks = set(keys)
            sub = {sp: st for sp, st in by_split.items() if ks & set(st.keys)}
            r = boot_delta_blocks(sub, keys, [PHYS_KEY], nboot=args.nboot, seed=0, rep_fn=lambda k: k[4]) if sub else \
                dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan, split_win=np.nan, rep_win=np.nan, delta_beq=np.nan,
                     ci_lo_beq=np.nan, ci_hi_beq=np.nan, rmse_A=np.nan, rmse_B=np.nan, n_splits=0, n_blocks=[], ci_flag="no valid split",
                     delta_split={})
            out.append(dict(target=t, **dict(zip(GRP_COLS, grp)), d_phys_blk=r["delta"], d_phys_lo=r["ci_lo"], d_phys_hi=r["ci_hi"],
                            p_boot=r["p_boot"], split_win=r["split_win"], rep_win=r["rep_win"],
                            d_phys_beq=r["delta_beq"], d_phys_beq_lo=r["ci_lo_beq"], d_phys_beq_hi=r["ci_hi_beq"],
                            rmse_blk=r["rmse_A"], phys_blk=r["rmse_B"], n_splits_blk=r["n_splits"],
                            n_blocks_min=int(min(r["n_blocks"])) if r["n_blocks"] else 0, ci_flag=r["ci_flag"],
                            d_phys_splits=json.dumps({str(k_): round(v, 4) for k_, v in r["delta_split"].items()}),
                            n_valid_splits=int(info[SPLITS[0]]["n_valid_splits"]) if info else 0,
                            n_unique_splits=int(info[SPLITS[0]]["n_unique_splits"]) if info else 0,
                            d_phys_blk_all=ra["delta"], d_phys_lo_all=ra["ci_lo"], d_phys_hi_all=ra["ci_hi"], split_win_all=ra["split_win"],
                            rep_win_all=ra["rep_win"], n_splits_blk_all=ra["n_splits"],
                            n_blocks_min_all=int(min(ra["n_blocks"])) if ra["n_blocks"] else 0, ci_flag_all=ra["ci_flag"]))
    return pd.DataFrame(out)


def summarize(runs, stores):
    dup = np.array([split_structure(t)[int(sp)]["dup_of"] >= 0 for t, sp in zip(runs.target, runs.split)], bool)
    runs = runs[~dup]                                                         # 중복 분할은 한 번만(첫 분할 유지)
    r = runs[runs.stage != "physics"].copy()
    phys = runs[runs.stage == "physics"].set_index(["target", "split"]).rmse_cm
    r["phys"] = [phys.loc[(t, s)] for t, s in zip(r.target, r.split)]
    r["d_phys"] = r.rmse_cm - r.phys; r["win"] = (r.rmse_cm < r.phys).astype(float)
    g = r.groupby(["target", "parent", "split", "rule", "stage", "scope", "lam", "alpha", "n", "rep"], as_index=False).agg(
        rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"), d_phys=("d_phys", "mean"), win=("win", "mean"))
    ck = ["target", "parent", "rule", "stage", "scope", "lam", "alpha", "n"]
    curve = g.groupby(ck, as_index=False).agg(
        rmse_mean=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), rmse_beq_mean=("rmse_beq_cm", "mean"),
        d_phys_mean=("d_phys", "mean"), win_rate=("win", "mean"), n_runs=("rmse_cm", "size"), n_splits=("split", "nunique"))
    # 보조: 기존 (분할, 반복) 행 재표집 CI
    ci_rep = {key: rep_boot_ci(sub.d_phys.values, nboot=1000, seed=1) for key, sub in g.groupby(ck)}
    curve["d_phys_lo_rep"] = [ci_rep[tuple(v)][0] for v in curve[ck].itertuples(index=False)]
    curve["d_phys_hi_rep"] = [ci_rep[tuple(v)][1] for v in curve[ck].itertuples(index=False)]
    # 주: 채점 블록 부트스트랩 CI·분할 승률·반복 승률
    bs = block_stats(stores)
    curve = curve.merge(bs, on=["target"] + GRP_COLS, how="left", validate="one_to_one")
    # 회복률(보조): 같은 단계·λ·α 의 라벨 전량(n = -1) 값 기준, 분모 ≤ 0 이면 NaN
    full = curve[curve.scope == "allA"].set_index(["target", "stage", "lam", "alpha"]).rmse_mean
    def rec(row):
        key = (row.target, row.stage, row.lam, row.alpha)
        if key not in full.index:
            return np.nan
        return recovery(row.rmse_mean - row.d_phys_mean, row.rmse_mean, full.loc[key])
    curve["recovery"] = curve.apply(rec, axis=1)
    be = []
    cn = curve[(curve.scope == "n") & curve.rule.isin(RULES)]
    for (t, rule, stage, lam, alpha), sub in cn.groupby(["target", "rule", "stage", "lam", "alpha"]):
        sub = sub[sub.n_splits == sub.n_splits.max()].sort_values("n")            # 분할 일부에만 있는 n(|A| 불균형 대상)은 제외
        mn = min_n(sub, n_col="n", ci_hi_col="d_phys_hi", split_win_col="split_win", rep_win_col="rep_win")
        mn_all = min_n(sub, n_col="n", ci_hi_col="d_phys_hi_all", split_win_col="split_win_all", rep_win_col="rep_win_all")
        n_valid = int(sub.n_valid_splits.iloc[0]); flags = []
        if n_valid < 2:
            flags.append("valid_splits<2")
        if int(sub.n_unique_splits.iloc[0]) < len(SPLITS):
            flags.append("dup_splits")
        if (sub.n_blocks_min < 5).any():
            flags.append("blocks<5")
        cens = bool(mn["censored"]) or n_valid < 2                          # 유효 분할 2개 미만은 주 정의 절단
        ok = sub[(sub.recovery >= 0.5) & (sub.win_rate >= 0.75)]
        ok_rep = sub[sub.d_phys_hi_rep < 0]
        be.append(dict(target=t, parent=sub.parent.iloc[0], rule=rule, stage=stage, lam=lam, alpha=alpha,
                       breakeven_n_ci=int(mn["n_star"]) if not cens else -1, censored=cens, be_flag=";".join(flags),
                       breakeven_n_ci_all=int(mn_all["n_star"]) if not mn_all["censored"] else -1, censored_all=bool(mn_all["censored"]),
                       n_valid_splits=n_valid, n_unique_splits=int(sub.n_unique_splits.iloc[0]),
                       n_blocks_min=int(sub.n_blocks_min.min()), ci_flag=";".join(sorted({f for f in sub.ci_flag.fillna("") if f})),
                       breakeven_n_rec50=int(ok.n.iloc[0]) if len(ok) else -1, censored_rec50=bool(len(ok) == 0),
                       breakeven_n_ci_rep=int(ok_rep.n.iloc[0]) if len(ok_rep) else -1,
                       n_max=int(mn["n_max"]), n_tested=int(mn["n_tested"]), n_splits=int(sub.n_splits.max()),
                       best_n=int(sub.loc[sub.rmse_mean.idxmin(), "n"]), best_d_phys=float(sub.d_phys_mean.min())))
    return curve, pd.DataFrame(be)


def light_meta(t, sp):
    """CatBoost 없이 대상 지표만(요약 재계산용)."""
    t_idx, parent, src_idx, comp = source_idx(t)
    src = DF.iloc[src_idx]; A_idx, B_idx = half_split_blocks(DF, t_idx, sp); evB = B_idx[eval_mask(DF.iloc[B_idx])]
    return dict(target=t, parent=parent, split=sp, n_A=len(A_idx), n_eval=len(evB), n_blocks_eval=int(DF.iloc[evB].block.nunique()),
                E0=ls_E(src.y.values, src.s.values), n_fit=0, elapsed_s=0.0, **comp, **hetero_metrics(t_idx, src_idx))


def meta_dict(**extra):
    return dict(stage="H25/H26 재실행(감사 반영)", plan=["docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md", "docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md"],
                targets=TARGETS, splits=SPLITS, n_grid=N_GRID, reps=REPS, reps_cb=REPS_CB, seeds=SEEDS, lams=LAMS, alphas=ALPHAS,
                r=args.r, kappa=args.kappa, buffer_km=args.buffer_km, k_sub=args.k_sub, rules=RULES,
                exclude_parent=bool(args.exclude_parent), no_s2=bool(args.no_s2), nboot=args.nboot, allA_n=-1,
                ci_main="h4_common.boot_delta_blocks: 분할 안 채점 블록 재표집, 방법·반복·seed 짝지음, 분할 분포 평균(d_phys_lo/hi)",
                ci_aux="(분할, 반복) 행 재표집(d_phys_lo_rep/hi_rep)",
                breakeven_main="블록 CI 상한 < 0 · split_win ≥ 2/3 · rep_win ≥ 0.75 인 최소 n, 미달성 -1·censored",
                breakeven_aux="회복률 ≥ 0.5 · 승률 ≥ 0.75(분모 ≤ 0 → 회복률 NaN), 회복률 기준 = 같은 단계 allA",
                split_rule="중복 분할(A 블록 집합 동일)은 모든 집계에서 첫 분할만 사용, 거울 분할은 유지. 블록 CI 주 통계는 유효 분할"
                           "(중복 아님 · 채점 블록 ≥ 2)만 결합, 유효 분할 < 2 인 대상은 breakeven_n_ci 절단(be_flag). 민감도 *_all = 중복만 제거한 전체 분할",
                split_info={t: {str(sp): v for sp, v in split_structure(t).items()} for t in TARGETS},
                smoke=bool(args.smoke), **extra)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        return "NA"


def write_outputs(tag, runs, stores, metas, **extra):
    curve, be = summarize(runs, stores)
    curve.to_csv(OUT / f"{tag}_curve.csv", index=False); be.to_csv(OUT / f"{tag}_breakeven.csv", index=False)
    tg = pd.DataFrame(metas)
    sinfo = pd.DataFrame([dict(target=t, split=sp, **{k: v for k, v in split_structure(t)[sp].items() if k != "nb_eval"})
                          for t in tg.target.unique() for sp in SPLITS])
    tg = tg.merge(sinfo, on=["target", "split"], how="left")
    tg.sort_values(["target", "split"]).to_csv(OUT / f"{tag}_targets.csv", index=False)
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta_dict(git_commit=git_commit(), n_rows=int(len(runs)), **extra),
                                                     ensure_ascii=False, indent=1, default=float))
    show = be[(be.rule == "kmedoid") & (be.lam.isin([0.0, 0.25])) & (be.alpha == 1.0)]
    print(show[["target", "stage", "breakeven_n_ci", "censored", "breakeven_n_rec50", "breakeven_n_ci_rep", "n_max", "best_n", "best_d_phys"]].to_string(index=False))


def main():
    t0 = time.time()
    tasks = [(t, sp) for t in TARGETS for sp in SPLITS]
    tag = args.tag + ("_smoke" if args.smoke else "")
    rows, metas, stores = [], [], []
    if args.summarize_only:
        runs = pd.read_csv(OUT / f"{tag}_runs.csv")
        st = load_stores(OUT / f"{tag}_blocksse.npz")
        metas = [light_meta(t, sp) for t, sp in tasks if ((runs.target == t) & (runs.split == sp)).any()]
        write_outputs(tag, runs, st, metas, summarize_only=True)
        print(f"summarized · 행 {len(runs)}")
        return
    print(f"[plan] 작업 {len(tasks)} · 워커 {args.workers} · n {N_GRID} · reps {REPS}/{REPS_CB} · seeds {SEEDS} · λ {LAMS} · α {ALPHAS} · "
          f"exclude_parent={args.exclude_parent} · no_s2={args.no_s2}", flush=True)
    def _done(r, m, st):
        rows.extend(r); metas.append(m); stores.append(st)
        print(f"  [{m['target']}|{m['split']}] 적합 {m['n_fit']} · A {m['n_A']} · 원천 {m['n_src']}(상위 {m['n_src_parent']}, 제외 {m['n_parent_excluded']}) · "
              f"E0 {m['E0']:.3f} · {m['elapsed_s']}s · 누적 {time.time()-t0:.0f}s · 완료 {len(metas)}/{len(tasks)}", flush=True)
    if args.workers <= 1:
        for t, sp in tasks:
            _done(*run_task(t, sp))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:   # fork 후 OpenMP 충돌 회피
            futs = {ex.submit(run_task, t, sp): (t, sp) for t, sp in tasks}
            for f in as_completed(futs):
                _done(*f.result())
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / f"{tag}_runs.csv", index=False)
    save_stores(stores, OUT / f"{tag}_blocksse.npz")
    write_outputs(tag, runs, merge_stores(stores), metas, elapsed_s=round(time.time() - t0, 1))
    print(f"saved {tag}_* · 행 {len(runs)} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
