"""M1 S-F: 희소 라벨 곡선. 대상 지역 라벨 n개가 전이 오차를 얼마나 줄이며, 어느 층(물리 계수 재적합 · 잔차 학습 ·
유사라벨 증강)이 라벨을 더 잘 쓰는가.

계획 docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md §3 F1–F3, docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md §3 S-F(개정 09-21 §B: A/B 성립 4지역).
결정 규칙 B의 두 번째 의의("라벨이 조금이라도 생기면 ML이 물리식을 넘는다")의 근거 실험.
E3(scripts/3_deep_learning/e3_adaptive_E.py: 단일 A/B 분할, E 재적합만, data/processed/e3_adaptive_E.csv)를 감사 지적
(단일 분할의 블록 이질성 교락, 수축 추정 부재)에 따라 재설계한다.

설계
  대상 = 레나 · 캐나다 · 러시아 W · 러시아 E. 학습 실측 = 알래스카 13,606셀(F4_direct). 입력 x25(SHARED_CORE).
  분할: 대상 셀을 0.5° 블록 단위로 A/B 2분할, split_seed 0..9(0 = E1·S3 결정적 GroupKFold, 1.. = 무작위 블록 균형 분할;
        m1_core.half_split_blocks). 평가 = B블록 셀 중 y·CCI·토양 도일 유효(m1_core.eval_mask).
  라벨: 각 분할에서 n ∈ {0, 3, 5, 10, 20, 40, 80}개 셀을 A블록에서 무작위 추출(n ≥ |A|는 생략). 반복 20회.
        같은 추출을 F1·F3·F2가 공유한다(짝지음). 러시아 W/E는 |A|≈15라 n ≤ 10.
  채점: 셀 가중 RMSE(주) + 블록 등가중 RMSE(보조, 09-21 개정 §A). 승률 = 같은 분할의 물리식(n=0) RMSE보다 낮은 실행 비율.
  F1 계수 적응(해석적, 반복 20회):
        refit      최소제곱 E_n = Σ s y / Σ s²
        shrink     (n·Ê_n + κ·E_AK)/(n + κ). κ = 알래스카 0.5° 블록별 E의 일원 확률효과 적률 추정(경험 베이즈):
                   σ²(블록 내 y/√TDD 분산, 합동) / τ²(블록 간 E 분산 − 표본 잡음 σ²·mean(1/n_b)). 메타에 기록.
        shrink_k10 κ = 10 고정(민감도, 표만)
        locscale   a + E_n√TDD 최소제곱(k=2, n ≥ 5)
        × 앵커 {stefan: E·√TDD, stefan_cci: ½(E·√TDD + CCI)}. E만 적응하고 CCI는 고정(E3 규약).
  F3 잔차 학습 대 계수 적응(catboost_lo 초모수 = m1_core.fit_model, thread_count 4; seed 3, 반복 10회, λ ∈ {0.25, 0.5}):
        resid      앵커 E_AK 고정. 알래스카 잔차 ∪ 대상 라벨 n개 잔차로 g 학습(라벨 있음 조건의 축소판, 가중 없음).
        both       앵커 E_n(shrink) 적응 + 잔차 g(대상 n개 행의 잔차는 E_n 기준으로 계산, 별도 적합).
        both_ls    both 의 g 에 E_n(refit) 앵커를 사후 결합(근사, 표만).
  F2 라벨 + 유사라벨(직접 catboost_lo, 반복 5회, seed 3):
        direct_pseudo  학습 = 알래스카 실측 ∪ A풀 Stefan 유사라벨(r = 10 × 13,606행, 복원 추출, E_AK·√TDD) ∪ 라벨 n개.
                   라벨이 있는 셀은 유사라벨 풀의 해당 행을 실측으로 대체한다(유사라벨은 결측 라벨의 대용이므로).
                   n = 0 이 S-C 공변량만 레시피(Stefan 유사라벨 직접 회귀)와 같다.
  상한(scope = allA): A 전체 라벨로 같은 방법을 적용한 값. 기준선(scope = n, n = 0): 앵커 E_AK 물리식.
누설 규약: 계수·유사라벨·잔차 학습은 알래스카 실측과 추출된 대상 라벨 n개만 사용. B 라벨은 채점 전용.

산출 data/processed/m1/
  sparse_label_runs.csv    실행 단위 장형(target, split, anchor, method, scope, n, rep, seed, lam, rmse_cm, rmse_beq_cm, bias_cm, E_used)
  sparse_label_curve.csv   집계(평균·SD·중앙값·10–90%·블록 등가중 평균·Δ(물리식 대비, 분할 짝지음)·승률·실행 수)
  sparse_label_table.csv   지역 × 앵커 × 방법 × λ 행, n 열(평균 RMSE) + 물리식 기준 + A 전체 상한 + 물리식을 넘는 최소 n(평균 기준·승률 90% 기준)
  sparse_label_meta.json   E_AK·κ 추정 내역·분할별 |A|·|B|·구성·계산량·git 해시
  outputs/figures/m1/sparse_label_curve.{png,pdf}             주 그림(수축 계수 · 잔차 · 계수+잔차 · 직접+유사라벨)
  outputs/figures/m1/sparse_label_curve_estimators.{png,pdf}  보조 그림(F1 추정량 3종)
실행(CPU 전용. GPU 5–9는 다른 실험이 점유 중이므로 사용 금지. 프로세스 4 × 4스레드, 약 70분)
  python3 scripts/3_deep_learning/m1_sparse_label_curve.py [--workers 4]
  python3 scripts/3_deep_learning/m1_sparse_label_curve.py --smoke      # 축소 실행(스크래치 폴더)
  python3 scripts/3_deep_learning/m1_sparse_label_curve.py --figure-only
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # CPU 전용
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
from polar.fidelity import TARGET, spatial_block_splits                                   # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks, fit_coefs   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="fidelity_base_v3.csv")
ap.add_argument("--soil", default="e5_soil_tdd_v3.csv")
ap.add_argument("--targets", default="Lena,Canada,Russia_W,Russia_E")
ap.add_argument("--splits", type=int, default=10, help="A/B 분할 반복 수(split_seed 0..K-1)")
ap.add_argument("--n-grid", default="0,3,5,10,20,40,80")
ap.add_argument("--reps-f1", type=int, default=20, help="라벨 추출 반복(계수 적응)")
ap.add_argument("--reps-f3", type=int, default=10, help="잔차 학습에 쓰는 반복 수(앞에서부터)")
ap.add_argument("--reps-f2", type=int, default=5, help="직접+유사라벨에 쓰는 반복 수(앞에서부터)")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--lams", default="0.25,0.5")
ap.add_argument("--r", type=float, default=10.0, help="유사라벨 배율(알래스카 실측 수 대비)")
ap.add_argument("--workers", type=int, default=4, help="프로세스 수(각 --threads 스레드)")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--min-cells", type=int, default=6)
ap.add_argument("--out-dir", default="")
ap.add_argument("--fig-dir", default="")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--figure-only", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"
TAG = "sparse_label"
SCRATCH = Path(os.environ.get("CLAUDE_SCRATCHPAD", "/tmp/claude-1025/-home-willy010313-Polar-Bigdata/5c7a18f6-a36f-445b-9b70-49574eb28aac/scratchpad"))
if args.smoke:
    OUT = Path(args.out_dir) if args.out_dir else SCRATCH / "sparse_label_smoke"
    FIG = Path(args.fig_dir) if args.fig_dir else OUT
    args.targets, args.splits, args.reps_f1, args.reps_f3, args.reps_f2, args.seeds = "Lena,Russia_E", 2, 2, 1, 1, 1
    args.workers = min(args.workers, 2)
else:
    OUT = Path(args.out_dir) if args.out_dir else PROC / "m1"
    FIG = Path(args.fig_dir) if args.fig_dir else ROOT / "outputs" / "figures" / "m1"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

FEATS = INPUT_SETS["x25"]
ANCHORS = ["stefan", "stefan_cci"]
TARGETS = [t for t in args.targets.split(",") if t]
SPLITS = list(range(args.splits))
N_GRID = [int(v) for v in args.n_grid.split(",")]
SEEDS = list(range(args.seeds))
LAMS = [float(v) for v in args.lams.split(",")]
REPS_F1, REPS_F3, REPS_F2 = args.reps_f1, args.reps_f3, args.reps_f2
LAM_FIG = 0.5                                              # 그림에 쓰는 λ(표에는 전부)
KAPPA_FIXED = 10.0


# ---------------------------------------------------------------- 수식 유틸
def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def affine_E(y, s, min_n=5):
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() < min_n:
        return np.nan, np.nan
    b, a = np.polyfit(s[m], y[m], 1)
    return float(a), float(b)


def anchor_vals(kind, E, s, c, a=0.0):
    st = a + E * s
    if kind == "stefan":
        return st
    if kind == "stefan_cci":
        return 0.5 * (st + c)
    raise ValueError(kind)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    """블록별 RMSE 후 블록 등가중 평균(09-21 개정 §A 보조 채점)."""
    e2 = (y - p) ** 2
    s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


def cb_fit_predict(Xtr, ytr, Xte, seed):
    """catboost_lo (m1_core.fit_model 과 동일 초모수, thread_count 만 args.threads)."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float)


def estimate_kappa(ak: pd.DataFrame, min_cells: int = 3) -> dict:
    """알래스카 0.5° 블록별 E 의 일원 확률효과(비율 모형 r_bi = y/√TDD = E_b + e, E_b ~ N(μ, τ²), e ~ N(0, σ²)) 적률 추정.
    σ² = 합동 블록 내 분산, τ² = Var(Ē_b) − σ²·mean(1/n_b), κ = σ²/τ² (수축 가중, 셀 수 단위)."""
    d = ak[np.isfinite(ak[TARGET].values) & (ak.e5_sqrt_tdd.values > 0)]
    rows = []
    for b, sub in d.groupby("block"):
        if len(sub) < min_cells:
            continue
        y, s = sub[TARGET].values.astype(float), sub.e5_sqrt_tdd.values.astype(float)
        r = y / s
        rows.append(dict(block=int(b), n=len(sub), E_ratio=float(r.mean()), E_ls=float((s @ y) / (s @ s)), s2=float(r.var(ddof=1))))
    bt = pd.DataFrame(rows)
    w = bt.n.values - 1
    sigma2_pooled = float((w * bt.s2.values).sum() / w.sum())
    sigma2_unw = float(bt.s2.mean())
    var_E = float(bt.E_ratio.var(ddof=1))
    noise = sigma2_pooled * float(np.mean(1.0 / bt.n.values))
    tau2 = max(var_E - noise, 1e-6)
    kappa = sigma2_pooled / tau2
    # 참고: 6-fold(공간블록 GroupKFold) 학습 집합별 E 의 산포
    folds = spatial_block_splits(ak.reset_index(drop=True), n_splits=6)
    E_folds = [ls_E(ak[TARGET].values[tr], ak.e5_sqrt_tdd.values[tr]) for tr, _ in folds]
    E_folds_te = [ls_E(ak[TARGET].values[te], ak.e5_sqrt_tdd.values[te]) for _, te in folds]
    return dict(kappa=float(kappa), sigma2_pooled=sigma2_pooled, sigma2_unweighted=sigma2_unw, tau2=float(tau2),
                var_E_blocks=var_E, sampling_noise=float(noise), n_blocks=int(len(bt)), min_cells=min_cells,
                E_block_ratio_mean=float(bt.E_ratio.mean()), E_block_ls_sd=float(bt.E_ls.std(ddof=1)),
                E_block_ratio_sd=float(np.sqrt(var_E)), block_n_median=float(bt.n.median()),
                kappa_unweighted_sigma2=float(sigma2_unw / max(var_E - sigma2_unw * float(np.mean(1.0 / bt.n.values)), 1e-6)),
                E_fold_train=[float(v) for v in E_folds], E_fold_test=[float(v) for v in E_folds_te],
                E_fold_test_sd=float(np.std(E_folds_te, ddof=1)))


# ---------------------------------------------------------------- 자료 (fork 전에 전역으로 적재)
if not args.figure_only:
    DF = load_base(PROC, base=args.base, soil=args.soil)
    AK = DF[DF.macro == "Alaska"]
    K_AK = fit_coefs(AK)
    E_AK = float(K_AK["E"])
    X_AK = AK[FEATS].values.astype(np.float32)
    y_AK = AK[TARGET].values.astype(float)
    s_AK = AK.e5_sqrt_tdd.values.astype(float)
    c_AK = AK.cci_alt.values.astype(float)
    KAPPA_INFO = estimate_kappa(AK)
    KAPPA = KAPPA_INFO["kappa"]
    print(f"[data] {args.base} · {len(DF):,}셀 · 알래스카 {len(AK):,} · E_AK {E_AK:.4f} · κ {KAPPA:.2f} "
          f"(σ² {KAPPA_INFO['sigma2_pooled']:.3f}, τ² {KAPPA_INFO['tau2']:.3f}, 블록 {KAPPA_INFO['n_blocks']}) · 대상 {TARGETS} · 분할 {SPLITS}", flush=True)


# ---------------------------------------------------------------- 작업 단위 (대상 × 분할)
def run_task(tg: str, sp: int):
    t0 = time.time()
    t_idx = np.where(DF.macro.values == tg)[0]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]
    nA = len(A_idx)
    yA, sA, cA = A[TARGET].values.astype(float), A.e5_sqrt_tdd.values.astype(float), A.cci_alt.values.astype(float)
    XA = A[FEATS].values.astype(np.float32)
    yB, sB, cB = B[TARGET].values.astype(float), B.e5_sqrt_tdd.values.astype(float), B.cci_alt.values.astype(float)
    XB = B[FEATS].values.astype(np.float32)
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    base = dict(target=tg, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB)
    rows, n_fit = [], 0

    def add(anchor, method, n, rep, seed, lam, pred, E_used, scope="n", n_real=-1):
        rows.append(dict(**base, anchor=anchor, method=method, scope=scope, n=int(n), rep=int(rep), seed=int(seed), lam=float(lam),
                         rmse_cm=rmse(yB, pred), rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)),
                         E_used=float(E_used) if np.isfinite(E_used) else np.nan, n_real_rows=int(n_real)))

    # 알래스카 잔차(앵커별, E_AK 고정)와 물리식 기준선
    R_AK, XR_AK, ANC_B = {}, {}, {}
    for anc in ANCHORS:
        r = y_AK - anchor_vals(anc, E_AK, s_AK, c_AK); ok = np.isfinite(r)
        R_AK[anc], XR_AK[anc] = r[ok], X_AK[ok]
        ANC_B[anc] = anchor_vals(anc, E_AK, sB, cB)
        add(anc, "physics", 0, -1, -1, 0.0, ANC_B[anc], E_AK)
    # n = 0 잔차 학습(알래스카만)
    for anc in ANCHORS:
        for seed in SEEDS:
            g = cb_fit_predict(XR_AK[anc], R_AK[anc], XB, seed); n_fit += 1
            for lam in LAMS:
                add(anc, "resid", 0, -1, seed, lam, ANC_B[anc] + lam * g, E_AK)
    # F2 유사라벨 풀(seed 별 고정 추출 → n 에 걸쳐 짝지음)
    n_ps = int(round(args.r * len(y_AK)))
    PS_SEL = {seed: np.random.RandomState(seed).choice(nA, n_ps, replace=True) for seed in SEEDS}

    def f2_fit(sel_lab, seed):
        ps = PS_SEL[seed]
        yps = E_AK * sA[ps]
        n_real = 0
        if sel_lab is not None and len(sel_lab):
            m = np.isin(ps, sel_lab); yps = yps.copy(); yps[m] = yA[ps[m]]; n_real = int(m.sum())
        Xtr = np.vstack([X_AK, XA[ps]]); ytr = np.concatenate([y_AK, yps])
        return cb_fit_predict(Xtr, ytr, XB, seed), n_real

    for seed in SEEDS:
        p, nr = f2_fit(None, seed); n_fit += 1
        add("none", "direct_pseudo", 0, -1, seed, 1.0, p, E_AK, n_real=nr)

    def evaluate_labels(sel, n, rep, scope, do_f3, do_f2):
        nonlocal n_fit
        y_n, s_n, c_n, X_n = yA[sel], sA[sel], cA[sel], XA[sel]
        E_ls = ls_E(y_n, s_n)
        E_sh = (n * E_ls + KAPPA * E_AK) / (n + KAPPA)
        E_k10 = (n * E_ls + KAPPA_FIXED * E_AK) / (n + KAPPA_FIXED)
        a_n, E_aff = affine_E(y_n, s_n)
        for anc in ANCHORS:
            add(anc, "refit", n, rep, -1, 0.0, anchor_vals(anc, E_ls, sB, cB), E_ls, scope)
            add(anc, "shrink", n, rep, -1, 0.0, anchor_vals(anc, E_sh, sB, cB), E_sh, scope)
            add(anc, "shrink_k10", n, rep, -1, 0.0, anchor_vals(anc, E_k10, sB, cB), E_k10, scope)
            if np.isfinite(E_aff):
                add(anc, "locscale", n, rep, -1, 0.0, anchor_vals(anc, E_aff, sB, cB, a=a_n), E_aff, scope)
        if do_f3:
            for anc in ANCHORS:
                r_fix = y_n - anchor_vals(anc, E_AK, s_n, c_n); ok = np.isfinite(r_fix)     # A 셀의 CCI 결측 행 제외
                r_ad = y_n - anchor_vals(anc, E_sh, s_n, c_n)
                Xtr = np.vstack([XR_AK[anc], X_n[ok]])
                ytr_fix = np.concatenate([R_AK[anc], r_fix[ok]]); ytr_ad = np.concatenate([R_AK[anc], r_ad[ok]])
                ancB_sh, ancB_ls = anchor_vals(anc, E_sh, sB, cB), anchor_vals(anc, E_ls, sB, cB)
                for seed in SEEDS:
                    g = cb_fit_predict(Xtr, ytr_fix, XB, seed)
                    g2 = cb_fit_predict(Xtr, ytr_ad, XB, seed); n_fit += 2
                    for lam in LAMS:
                        add(anc, "resid", n, rep, seed, lam, ANC_B[anc] + lam * g, E_AK, scope)
                        add(anc, "both", n, rep, seed, lam, ancB_sh + lam * g2, E_sh, scope)
                        add(anc, "both_ls", n, rep, seed, lam, ancB_ls + lam * g2, E_ls, scope)
        if do_f2:
            for seed in SEEDS:
                p, nr = f2_fit(sel, seed); n_fit += 1
                add("none", "direct_pseudo", n, rep, seed, 1.0, p, E_AK, scope, n_real=nr)

    # 상한: A 전체 라벨
    evaluate_labels(np.arange(nA), nA, -1, "allA", do_f3=True, do_f2=True)
    # n 격자
    for n in N_GRID:
        if n <= 0 or n >= nA:
            continue
        for rep in range(REPS_F1):
            sel = np.random.RandomState(1_000_003 * sp + 1_009 * n + rep).choice(nA, n, replace=False)
            evaluate_labels(sel, n, rep, "n", do_f3=rep < REPS_F3, do_f2=rep < REPS_F2)
    return rows, dict(target=tg, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB,
                      n_fit=n_fit, elapsed_s=round(time.time() - t0, 1))


# ---------------------------------------------------------------- 집계
def summarize(runs: pd.DataFrame):
    runs = runs.copy()
    phys = runs[runs.method == "physics"].set_index(["target", "anchor", "split"]).rmse_cm.to_dict()
    ref_anchor = np.where(runs.anchor.values == "none", "stefan", runs.anchor.values)
    runs["ref_rmse_cm"] = [phys.get((t, a, s), np.nan) for t, a, s in zip(runs.target, ref_anchor, runs.split)]
    runs["delta_cm"] = runs.rmse_cm - runs.ref_rmse_cm
    keys = ["target", "anchor", "method", "lam", "scope", "n"]
    cur = runs.groupby(keys).agg(
        rmse_mean=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), rmse_med=("rmse_cm", "median"),
        rmse_p10=("rmse_cm", lambda v: float(np.quantile(v, 0.10))), rmse_p90=("rmse_cm", lambda v: float(np.quantile(v, 0.90))),
        rmse_beq_mean=("rmse_beq_cm", "mean"), bias_mean=("bias_cm", "mean"), delta_mean=("delta_cm", "mean"),
        win_frac=("delta_cm", lambda v: float(np.mean(v < 0))), n_runs=("rmse_cm", "size"), n_splits=("split", "nunique"),
        E_mean=("E_used", "mean"), E_sd=("E_used", "std")).reset_index()
    return runs, cur


def make_table(cur: pd.DataFrame, grid):
    """지역 × 앵커 × 방법 × λ 행, n 열(평균 RMSE). n=0 열: F1 추정량은 물리식, both/both_ls 는 resid n=0."""
    out = []
    phys = {(r.target, r.anchor): r.rmse_mean for r in cur[cur.method == "physics"].itertuples()}
    for (t, a, m, lam), sub in cur[(cur.scope == "n") & (cur.method != "physics")].groupby(["target", "anchor", "method", "lam"]):
        ref_a = "stefan" if a == "none" else a
        p0 = phys.get((t, ref_a), np.nan)
        row = dict(target=t, anchor=a, method=m, lam=lam, physics_rmse=p0)
        vals = {int(r.n): r for r in sub.itertuples()}
        if 0 not in vals:
            if m in ("both", "both_ls"):
                r0 = cur[(cur.target == t) & (cur.anchor == a) & (cur.method == "resid") & (cur.lam == lam) & (cur.n == 0)]
                n0 = float(r0.rmse_mean.iloc[0]) if len(r0) else np.nan
            else:
                n0 = p0
        else:
            n0 = float(vals[0].rmse_mean)
        row["n0"] = n0
        for n in grid:
            if n == 0:
                continue
            row[f"n{n}"] = float(vals[n].rmse_mean) if n in vals else np.nan
            row[f"n{n}_win"] = float(vals[n].win_frac) if n in vals else np.nan
        ub = cur[(cur.target == t) & (cur.anchor == a) & (cur.method == m) & (cur.lam == lam) & (cur.scope == "allA")]
        row["rmse_allA"] = float(ub.rmse_mean.iloc[0]) if len(ub) else np.nan
        row["n_allA"] = float(ub.n.mean()) if len(ub) else np.nan
        pos = sub[sub.n > 0].sort_values("n")
        ok_mean = pos[pos.rmse_mean < p0]
        ok_win = pos[pos.win_frac >= 0.9]
        row["n_min_mean"] = int(ok_mean.n.min()) if len(ok_mean) else -1
        row["n_min_win90"] = int(ok_win.n.min()) if len(ok_win) else -1
        row["beats_physics_all_larger_n"] = bool(len(ok_mean)) and bool((pos[pos.n >= ok_mean.n.min()].rmse_mean < p0).all())
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 그림
REG_ORDER = ["Lena", "Canada", "Russia_W", "Russia_E"]
REG_NAME = {"Lena": "레나", "Canada": "캐나다", "Russia_W": "러시아 W", "Russia_E": "러시아 E"}
ANC_NAME = {"stefan": "앵커 Stefan", "stefan_cci": "앵커 Stefan + CCI"}
INK, MUTED, REF = "#333333", "#6b6b6b", "#7a7a7a"
# 계열 색: dataviz validate_palette(light, all-pairs) 통과. 붉은 계열 없음(갈색은 brand_tokens categorical.warn 계열 재단계).
MAIN_SERIES = [  # (method, lam, anchor_mode, label, color, marker, linestyle)
    ("shrink", 0.0, "same", "계수 적응(수축 E_n)", "#2a78d6", "o", "-"),
    ("resid", LAM_FIG, "same", f"잔차 학습(E 고정, λ={LAM_FIG:g})", "#c0692a", "s", "-"),
    ("both", LAM_FIG, "same", f"계수 적응 + 잔차(λ={LAM_FIG:g})", "#4a3aa7", "D", "-"),
    ("direct_pseudo", 1.0, "none", "직접 회귀 + Stefan 유사라벨(r=10) + 라벨", "#1baf7a", "^", (0, (4, 1.5))),
]
EST_SERIES = [
    ("refit", 0.0, "same", "최소제곱 E_n", "#008300", "s", "-"),
    ("shrink", 0.0, "same", "수축 E_n (κ 경험 베이즈)", "#2a78d6", "o", "-"),
    ("locscale", 0.0, "same", "위치·척도 a + E_n√TDD (n ≥ 5)", "#d55181", "^", (0, (4, 1.5))),
]


def _series_points(cur, t, anc, m, lam, grid, phys_val):
    a_ = "none" if m == "direct_pseudo" else anc
    sub = cur[(cur.target == t) & (cur.anchor == a_) & (cur.method == m) & np.isclose(cur.lam, lam) & (cur.scope == "n")]
    d = {int(r.n): r for r in sub.itertuples()}
    if 0 not in d:
        if m in ("both", "both_ls"):
            r0 = cur[(cur.target == t) & (cur.anchor == a_) & (cur.method == "resid") & np.isclose(cur.lam, lam) & (cur.scope == "n") & (cur.n == 0)]
            if len(r0):
                d[0] = r0.iloc[0]
        elif m in ("refit", "shrink", "shrink_k10", "locscale"):
            d[0] = None                                     # 물리식 값(분할 평균), 범위 없음
    xs, ys, lo, hi = [], [], [], []
    for k, n in enumerate(grid):
        if n not in d:
            continue
        r = d[n]
        xs.append(k)
        if r is None:
            ys.append(phys_val); lo.append(np.nan); hi.append(np.nan)
        else:
            ys.append(float(r.rmse_mean)); lo.append(float(r.rmse_p10)); hi.append(float(r.rmse_p90))
    return np.array(xs, float), np.array(ys), np.array(lo), np.array(hi)


def plot_curves(cur, meta, series, fname, footnote):
    from polar.plotstyle import use_polar
    from matplotlib.lines import Line2D
    plt = use_polar(); plt.rcParams["pdf.fonttype"] = 42
    targets = [t for t in REG_ORDER if t in set(cur.target)]
    grids = {t: sorted(set(cur[(cur.target == t) & (cur.scope == "n")].n.astype(int).tolist()) | {0}) for t in targets}
    widths = [len(grids[t]) for t in targets]
    fig, axes = plt.subplots(2, len(targets), figsize=(13.5, 6.9), sharey="col", squeeze=False,
                             gridspec_kw=dict(width_ratios=widths, wspace=0.16, hspace=0.30))
    split_info = {t: [d for d in meta["tasks"] if d["target"] == t] for t in targets}
    labels_done = False
    for i, anc in enumerate(ANCHORS):
        for j, t in enumerate(targets):
            ax = axes[i, j]; grid = grids[t]
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_linewidth(0.7); ax.spines[s_].set_color(REF)
            ax.grid(True, axis="y", color="#aab3bd", lw=0.5, alpha=0.35); ax.grid(False, axis="x"); ax.set_axisbelow(True)
            ax.tick_params(labelsize=9, length=2.5, width=0.7, color="#8a8a8a", labelcolor=INK)
            phys = float(cur[(cur.target == t) & (cur.anchor == anc) & (cur.method == "physics")].rmse_mean.iloc[0])
            ub_rows = cur[(cur.target == t) & (cur.scope == "allA") & (cur.anchor.isin([anc, "none"]))]
            ub_rows = ub_rows[[any((r.method == m and np.isclose(r.lam, lam)) for m, lam, *_ in series) for r in ub_rows.itertuples()]]
            ub = float(ub_rows.rmse_mean.min()) if len(ub_rows) else np.nan
            ax.axhline(phys, color=REF, lw=1.0, ls="-", zorder=1)
            if np.isfinite(ub):
                ax.axhline(ub, color=REF, lw=1.0, ls=(0, (3, 2)), zorder=1)
            for si, (m, lam, amode, label, col, mk, ls) in enumerate(series):
                xs, ys, lo, hi = _series_points(cur, t, anc, m, lam, grid, phys)
                if len(xs) == 0:
                    continue
                dodge = (si - (len(series) - 1) / 2) * 0.09
                ax.plot(xs + dodge, ys, color=col, lw=1.8, ls=ls, marker=mk, ms=4.6, mec="white", mew=0.6, zorder=4,
                        label=label if not labels_done else None)
                ok = np.isfinite(lo)
                ax.vlines(xs[ok] + dodge, lo[ok], hi[ok], color=col, lw=0.9, alpha=0.7, zorder=3)
            labels_done = True
            ax.set_xticks(range(len(grid))); ax.set_xticklabels([str(n) for n in grid])
            ax.set_xlim(-0.5, len(grid) - 0.5)
            if i == 0:
                nA = np.mean([d["n_A"] for d in split_info[t]]); nB = np.mean([d["n_eval"] for d in split_info[t]])
                ax.set_title(f"{REG_NAME[t]}  (|A| ≈ {nA:.0f}, B {nB:.0f}셀)", fontsize=10.5, color=INK, pad=6)
            if j == 0:
                ax.set_ylabel("RMSE (cm)", fontsize=10.5, color=INK)
            if i == 1:
                ax.set_xlabel("대상 지역 라벨 수 n", fontsize=10.5, color=INK)
            ax.text(0.02, 0.97, "(" + "abcdefgh"[i * len(targets) + j] + ")", transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color=INK, va="top", ha="left")
            # 기준선 값 라벨(오른쪽 끝)
            ax.text(len(grid) - 0.55, phys, f"물리식 n=0: {phys:.1f}", fontsize=7.2, color=MUTED, va="bottom", ha="right")
            if np.isfinite(ub):
                ax.text(len(grid) - 0.55, ub, f"A 전체 라벨: {ub:.1f}", fontsize=7.2, color=MUTED, va="top", ha="right")
    for i, anc in enumerate(ANCHORS):
        y0 = axes[i, 0].get_position().y0; y1 = axes[i, 0].get_position().y1
        fig.text(0.012, (y0 + y1) / 2, ANC_NAME[anc], rotation=90, fontsize=10.5, color=INK, va="center", ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color=REF, lw=1.0, ls="-"), Line2D([], [], color=REF, lw=1.0, ls=(0, (3, 2)))]
    labels += ["물리식 n=0 (E_AK, 분할 평균)", "A 전체 라벨 상한(그림 계열 중 최소)"]
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), fontsize=8.3, frameon=False,
               bbox_to_anchor=(0.5, 0.035), handlelength=2.2, columnspacing=1.4)
    fig.text(0.5, 0.005, footnote, fontsize=7.8, color=MUTED, ha="center", va="bottom")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.17)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=300)
    plt.close(fig)
    print("saved", FIG / f"{fname}.png", flush=True)


def make_figures():
    cur = pd.read_csv(OUT / f"{TAG}_curve.csv")
    meta = json.loads((OUT / f"{TAG}_meta.json").read_text())
    cfg = meta["config"]
    foot = (f"선: 분할 {cfg['splits']} × 라벨 추출 반복(계수 {cfg['reps_f1']} · 잔차 {cfg['reps_f3']} · 유사라벨 {cfg['reps_f2']}) "
            f"× seed {cfg['seeds']}(잔차·유사라벨) 평균 · 세로 막대: 10–90% 범위 · 평가: B블록 셀(CCI·토양 도일 유효) · "
            f"수축 κ = {meta['kappa']['kappa']:.2f}(알래스카 블록 경험 베이즈) · 러시아 W/E는 |A| ≈ 15라 n ≤ 10")
    plot_curves(cur, meta, MAIN_SERIES, f"{TAG}_curve", foot)
    foot2 = (f"선: 분할 {cfg['splits']} × 라벨 추출 반복 {cfg['reps_f1']} 평균 · 세로 막대: 10–90% 범위 · 평가: B블록 셀 · "
             f"수축 κ = {meta['kappa']['kappa']:.2f}(σ² {meta['kappa']['sigma2_pooled']:.3f} / τ² {meta['kappa']['tau2']:.3f}, "
             f"블록 {meta['kappa']['n_blocks']}) · 위치·척도는 n ≥ 5")
    plot_curves(cur, meta, EST_SERIES, f"{TAG}_curve_estimators", foot2)


# ---------------------------------------------------------------- 실행
def main():
    t_start = time.time()
    tasks = []
    for tg in TARGETS:
        n_cells = int((DF.macro.values == tg).sum())
        if n_cells < args.min_cells:
            print(f"[skip] {tg}: 셀 {n_cells}", flush=True); continue
        for sp in SPLITS:
            tasks.append((tg, sp, n_cells))
    tasks.sort(key=lambda x: -x[2])                            # 큰 지역부터(부하 균형)
    rows, task_meta = [], []
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_task, tg, sp): (tg, sp) for tg, sp, _ in tasks}
        done = 0
        for f in as_completed(futs):
            tg, sp = futs[f]
            r, m = f.result(); rows.extend(r); task_meta.append(m); done += 1
            print(f"  [{tg}|split {sp}] 행 {len(r):,} · 적합 {m['n_fit']} · {m['elapsed_s']}s · 완료 {done}/{len(tasks)} · 경과 {time.time()-t_start:.0f}s", flush=True)
    runs = pd.DataFrame(rows)
    runs, cur = summarize(runs)
    table = make_table(cur, N_GRID)
    runs.to_csv(OUT / f"{TAG}_runs.csv", index=False)
    cur.to_csv(OUT / f"{TAG}_curve.csv", index=False)
    table.to_csv(OUT / f"{TAG}_table.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                          # noqa: BLE001
        commit = "NA"
    n_min = {}
    for r in table.itertuples():
        n_min[f"{r.target}|{r.anchor}|{r.method}|{r.lam:g}"] = dict(n_min_mean=int(r.n_min_mean), n_min_win90=int(r.n_min_win90),
                                                                   physics=round(float(r.physics_rmse), 3), allA=round(float(r.rmse_allA), 3))
    meta = dict(
        stage="M1-S-F", tag=TAG, base=args.base, soil=args.soil, sources="F4_direct", features="x25 (SHARED_CORE)",
        config=dict(targets=TARGETS, splits=args.splits, n_grid=N_GRID, reps_f1=REPS_F1, reps_f3=REPS_F3, reps_f2=REPS_F2,
                    seeds=SEEDS, lams=LAMS, r=args.r, kappa_fixed=KAPPA_FIXED, workers=args.workers, threads=args.threads, smoke=args.smoke),
        E_alaska=E_AK, E_alaska_k2=dict(a=float(K_AK["st_a"]), E=float(K_AK["st_E"])), n_alaska=int(len(AK)),
        kappa=KAPPA_INFO, tasks=sorted(task_meta, key=lambda d: (d["target"], d["split"])),
        n_fit_total=int(sum(d["n_fit"] for d in task_meta)), n_rows=int(len(runs)), n_min=n_min,
        methods=dict(physics="앵커 E_AK(n=0 기준선)", refit="최소제곱 E_n", shrink="(n·Ê_n + κ·E_AK)/(n+κ), κ 경험 베이즈",
                     shrink_k10="κ=10 고정(민감도)", locscale="a + E_n√TDD 최소제곱(n ≥ 5)",
                     resid="앵커 E_AK 고정 + catboost_lo 잔차 g(알래스카 잔차 ∪ 대상 라벨 n개 잔차, 가중 없음), λ 스윕",
                     both="앵커 E_n(shrink) + 잔차 g(대상 n개 행의 잔차는 E_n 기준, 별도 적합)",
                     both_ls="both 의 g 에 E_n(refit) 앵커를 사후 결합(근사)",
                     direct_pseudo="직접 catboost_lo: 알래스카 실측 ∪ A풀 Stefan 유사라벨(r=10, 복원 추출) ∪ 라벨 n개(라벨 셀의 유사라벨 행을 실측으로 대체)"),
        scoring="셀 가중 RMSE(주), 블록 등가중 RMSE(보조), 승률 = 같은 분할 물리식(n=0) 대비 낮은 실행 비율. 10–90% 범위는 분할×반복×seed 합동 분포",
        label_draw="RandomState(1000003·split + 1009·n + rep).choice(|A|, n, replace=False); F1·F3·F2 공유(짝지음)",
        catboost_lo="iterations=200, lr=0.05, depth=3, l2=3, thread_count=4 (m1_core.fit_model 과 동일 초모수)",
        plan="docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md §3 F1–F3; docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md §3 S-F, 개정 09-21 §A·§B",
        predecessor="scripts/3_deep_learning/e3_adaptive_E.py (단일 분할, E 재적합만, 반복 30회)",
        env=dict(device="cpu", cuda_visible_devices="", git_commit=commit, elapsed_s=round(time.time() - t_start, 1)))
    (OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 250)
    for anc in ANCHORS + ["none"]:
        sub = table[(table.anchor == anc) & (table.lam.isin([0.0, LAM_FIG, 1.0]))]
        if len(sub):
            cols = ["target", "method", "lam", "n0"] + [f"n{n}" for n in N_GRID if n > 0] + ["rmse_allA", "n_min_mean", "n_min_win90"]
            print(f"\n[{anc}]"); print(sub[cols].round(2).to_string(index=False))
    print(f"\nsaved: {OUT / (TAG + '_runs.csv')} ({len(runs):,}행) · 적합 {meta['n_fit_total']:,} · {time.time()-t_start:.0f}s", flush=True)
    make_figures()


if __name__ == "__main__":
    if args.figure_only:
        make_figures()
    else:
        main()
