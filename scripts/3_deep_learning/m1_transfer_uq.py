"""M1 전이 UQ (H17): 알래스카 보정 CQR 구간의 전이 지역 커버리지와 회복 방법.

계획 docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md 개정 2026-09-21 §F H17. S11(scripts/3_deep_learning/s11_conformal_uq.py)
프로토콜(CatBoost MultiQuantile 11분위, CQR, 보정 블록 25%, 3 seed)을 전이 지역(TRANSFER_MAIN 6)으로 연장한다.
기여 C3(보정 예측구간)은 알래스카 지역 내 커버리지(S11: 93.4%)만 보고했고, 공변량 이동 하에서는 교환가능성이 깨져
보장이 없다. 여기서는 (i) 알래스카 보정량을 그대로 옮겼을 때의 대상 지역 커버리지, (ii) 밀도비 가중 등순응·
대상 유사라벨 재보정이 명목 90%를 회복하는지를 본다.

구간 종류 (α=0.1, 명목 90%)
  raw            분위 예측 [q05, q95] 무보정
  cqr_ak         알래스카 보정 블록 CQR (Romano 2019)                                    (a)
  cqr_iw         밀도비 가중 CQR (Tibshirani 2019; w(x)=p_t/p_s, 로지스틱, 클립 10)         (b)
  anchor_ak      Stefan+CCI 앵커 ± Q, Q = 알래스카 보정 블록 |y−anchor| 의 90% 순서통계 분위   (c)
  anchor_iw      앵커 구간의 밀도비 가중판                                                (c)
  cqr_pseudo     대상 A블록 Stefan 유사라벨(ỹ=E·√TDD, E는 알래스카)로 CQR 재보정 → B블록 채점  (d, 라벨 미사용)
  anchor_pseudo  앵커 ± Q(|ỹ−anchor| = 0.5·|Stefan−CCI|; 공식 혼합, 계획 §E 표기 규칙)      (d, 라벨 미사용)
  cqr_real       대상 A블록 실측으로 CQR 재보정 → B블록 채점                               (e, 라벨 있음, 실용 상한)
  anchor_real    앵커 ± Q(A블록 실측 |y−anchor|)                                        (e)
조건
  noinfo    대상 전체 평가 셀(y·CCI·토양 도일 유효), 주 6지역
  covonly   half_split_blocks(df, idx, split_seed)의 B블록(분할 0·1·2) 채점, A블록은 보정 전용.
            4지역(레나·캐나다·러시아 W·러시아 E; 계획 개정 09-21 §B. 러시아 C 7셀·그린란드 3셀은 noinfo만)
참조
  Alaska/in_domain  같은 규약(x25, fidelity_base_v3)의 알래스카 6-fold 공간블록 OOF(각 fold train 블록 25% 보정)
  Alaska_S11        S11 헤드라인(x34, fidelity_base v1, 93.4%) — s11_conformal_meta.json·e4_interval_score.csv에서 읽음
누설 규약
  계수(E·CCI 회귀)·분위 모델은 알래스카 proper-train 라벨에서만. 밀도비·DI 표준화는 공변량만 사용.
  대상 라벨은 (e)와 채점에만 쓴다. 가중 분위는 검정 셀의 w(x)까지 포함한 Tibshirani(2019) 정의를 셀별로 계산한다.
  w≡1이면 ceil((n+1)(1−α)) 번째 순서통계, 즉 S11 CQR의 유한표본 규약과 같다(S11은 같은 수준의 보간 분위).
지표
  커버리지·평균 폭(cm)·interval score(Winkler, α=0.1)와 분해, 무한 구간 비율. 지역별 seed(·분할) 평균과
  0.5° 블록 부트스트랩 95% CI(400회, 통계량 = seed·분할 평균; 블록 8 미만 지역은 CI 미산출·플래그).
  조건부: AOA DI(Meyer 2021, x25 표준화 최근접 거리, 임계 = 학습 DI Q75+1.5·IQR) 안/밖, 예측 중앙값 5분위.

산출  data/processed/m1/transfer_uq_results.csv      region × cond × split × method × seed 지표
      data/processed/m1/transfer_uq_summary.csv      region × cond × method (seed·분할 평균, 블록 부트스트랩 CI, 참조 행 포함)
      data/processed/m1/transfer_uq_conditional.csv  AOA DI 안/밖·예측 5분위별 커버리지
      data/processed/m1/transfer_uq_weights.csv      밀도비 가중 진단(AUC·ESS·클립 비율·Q_w 대 Q)
      data/processed/m1/transfer_uq_cells.csv        noinfo 셀별 seed 평균 구간(지도용)
      data/processed/m1/transfer_uq_meta.json
      outputs/figures/m1/transfer_uq_coverage.{png,pdf}     지역별 커버리지 점·CI(방법별)
      outputs/figures/m1/transfer_uq_width_coverage.{png,pdf} 폭 대 커버리지
실행(ROOT, CPU 전용, GPU 노출 차단): python3 scripts/3_deep_learning/m1_transfer_uq.py  [--smoke] [--figure-only]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # CPU 전용. GPU 5–9는 타 실험 점유(사용 금지)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import SHARED_CORE, TARGET, TRANSFER_MAIN, spatial_block_splits          # noqa: E402
from polar.m1_core import (load_base, eval_mask, half_split_blocks, fit_coefs, anchor_pred,   # noqa: E402
                           importance_weights)
# 입력 전처리: CatBoost는 NaN-native(polar.preprocessing.fold_prep(nan_native=True)와 동일하게 원본 유지)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--nboot", type=int, default=400)
ap.add_argument("--iters", type=int, default=600)
ap.add_argument("--no-alaska-ref", action="store_true", help="알래스카 지역 내 6-fold 참조(x25) 생략")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--figure-only", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"
OUT = PROC / "m1"; OUT.mkdir(exist_ok=True)
FIG = ROOT / "outputs" / "figures" / "m1"; FIG.mkdir(parents=True, exist_ok=True)
TAG = "transfer_uq" + ("_smoke" if args.smoke else "")
ALPHA = 0.10
LEVEL = 1.0 - ALPHA
CAL_FRAC = 0.25                                                # S11
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95]   # S11 분위 11개
AIDX = {a: i for i, a in enumerate(ALPHAS)}
ALPHA_STR = ",".join(str(a) for a in ALPHAS)
SEEDS = [0] if args.smoke else list(range(args.seeds))
SPLITS = [0] if args.smoke else list(range(args.splits))
ITERS = 150 if args.smoke else args.iters
NBOOT = 50 if args.smoke else args.nboot
N_REF_FOLDS = 6
COVONLY_REGIONS = ["Lena", "Canada", "Russia_W", "Russia_E"]  # 계획 개정 2026-09-21 §B
FEATS = list(SHARED_CORE)                                      # x25
ANCHOR = "stefan_cci"
MIN_BLOCKS_CI = 8
IW_CLIP = 10.0
METHODS_NOINFO = ["raw", "cqr_ak", "cqr_iw", "anchor_ak", "anchor_iw"]
METHODS_COVONLY = METHODS_NOINFO + ["cqr_pseudo", "cqr_real", "anchor_pseudo", "anchor_real"]
METHODS_REF = ["raw", "cqr_ak", "anchor_ak"]


# ---------------------------------------------------------------- 도구
def fit_quantiles(Xtr, ytr, seed):
    """S11 하이퍼파라미터의 MultiQuantile CatBoost(11분위). thread 4 고정(공유 서버)."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={ALPHA_STR}", iterations=ITERS, learning_rate=0.03,
                          depth=6, l2_leaf_reg=3.0, random_seed=seed, verbose=0, thread_count=4,
                          allow_writing_files=False)
    m.fit(Xtr, ytr)
    return m


def predict_q(m, X):
    return np.sort(np.asarray(m.predict(X)), axis=1)         # rearrangement: 분위 단조 강제(S11)


def wquantile(scores, w, w_test, level):
    """Tibshirani(2019) 가중 등순응 분위. 검정 셀별 Q = inf{q : Σ_{i: E_i≤q} w_i / (Σ_j w_j + w_test) ≥ level},
    누적 가중이 level에 못 미치면 +inf(무한 구간). w≡1, w_test=1 이면 ceil((n+1)·level) 번째 순서통계."""
    scores = np.asarray(scores, float); w = np.asarray(w, float)
    ok = np.isfinite(scores) & np.isfinite(w)
    scores, w = scores[ok], w[ok]
    o = np.argsort(scores, kind="mergesort"); Es = scores[o]; C = np.cumsum(w[o]); S = C[-1]
    tau = level * (S + np.asarray(w_test, float)) - 1e-9
    k = np.searchsorted(C, tau, side="left")
    return np.where(k < len(Es), Es[np.minimum(k, len(Es) - 1)], np.inf)


def q_plain(scores, level=LEVEL):
    scores = np.asarray(scores, float); scores = scores[np.isfinite(scores)]
    return float(wquantile(scores, np.ones(len(scores)), np.ones(1), level)[0])


def density_ratio(Xsrc, Xtgt, seed=0, clip=IW_CLIP):
    """m1_core.importance_weights(T3)와 같은 분류기(중앙값 대치·표준화·balanced 로지스틱)로 소스·대상 양쪽의
    w(x)=p_t(x)/p_s(x)를 얻는다(소스 평균 1 정규화 후 [1/clip, clip] 클립). 소스 쪽은 importance_weights와
    동일함을 assert로 확인한다. 분류기 AUC(분리 가능성)도 반환."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Xsrc = np.asarray(Xsrc, float); Xtgt = np.asarray(Xtgt, float)
    X = np.vstack([Xsrc, Xtgt]); yy = np.r_[np.zeros(len(Xsrc)), np.ones(len(Xtgt))]
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0.0); X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed).fit(X, yy)
    p = clf.predict_proba(X)[:, 1]
    w = p / np.clip(1 - p, 1e-6, None)
    w = np.clip(w / np.mean(w[:len(Xsrc)]), 1.0 / clip, clip)
    w_src, w_tgt = w[:len(Xsrc)], w[len(Xsrc):]
    chk = importance_weights(Xsrc, Xtgt, seed=seed, clip=clip)
    assert np.allclose(w_src, chk, rtol=1e-5, atol=1e-7), "밀도비 소스 가중이 importance_weights와 불일치"
    auc = float(roc_auc_score(yy, p))
    return w_src, w_tgt, auc


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def compute_di(X_ref, X_q, ref_folds, seed=0, n_pair=4000):
    """Meyer & Pebesma(2021) DI(CAST 규약, 변수 가중 없음). 표준화(참조 통계) 공변량 공간에서
      d̄ = 학습점 간 평균 쌍거리(고유 벡터 n_pair개 표본으로 근사),
      DI(새 점) = 최근접 학습점 거리 / d̄,
      학습 DI = 자기 CV fold 밖 학습점까지의 최근접 거리 / d̄ (fold = 알래스카 6-fold 공간블록, 지역 내 CV 규약),
      AOA 임계 = 학습 DI의 Q75 + 1.5·IQR.
    (최근접 거리 기반 d̄ 근사는 조밀·중복 표본에서 d̄→0이 되어 전 대상이 AOA 밖으로 퇴화하므로 쓰지 않는다.)"""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import pairwise_distances
    X_ref = np.asarray(X_ref, float); X_q = np.asarray(X_q, float)
    med = np.nanmedian(X_ref, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xr = np.where(np.isnan(X_ref), med, X_ref); Xq = np.where(np.isnan(X_q), med, X_q)
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-6; Xr = (Xr - mu) / sd; Xq = (Xq - mu) / sd
    Xr_u = np.unique(Xr, axis=0)
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(Xr_u), min(n_pair, len(Xr_u)), replace=False)
    D = pairwise_distances(Xr_u[sub]); d_bar = float(D[np.triu_indices(len(sub), 1)].mean())
    di_ref = np.full(len(Xr), np.nan)
    for tr, te in ref_folds:                                  # fold 밖 최근접(학습 DI)
        nn_f = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(Xr[tr])
        di_ref[te] = nn_f.kneighbors(Xr[te])[0][:, 0] / d_bar
    q75, q25 = np.nanquantile(di_ref, [0.75, 0.25]); thr = float(q75 + 1.5 * (q75 - q25))
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(Xr_u)
    d_q, _ = nn.kneighbors(Xq)
    print(f"[DI] 참조 고유 벡터 {len(Xr_u):,}/{len(Xr):,} · d̄(평균 쌍거리, n={len(sub):,}) {d_bar:.3f} · "
          f"학습 DI 중앙값 {np.nanmedian(di_ref):.3f} · Q75 {q75:.3f} · 임계 {thr:.3f}", flush=True)
    return d_q[:, 0] / d_bar, thr, di_ref


def interval_metrics(y, lo, hi):
    """셀별 커버리지·폭·interval score(α=0.1)·무한 구간 표시. 무한 구간은 포함(커버)으로 세고 폭·IS는 NaN."""
    fin = np.isfinite(lo) & np.isfinite(hi)
    cov = np.where(fin, (y >= lo) & (y <= hi), True).astype(float)
    width = np.where(fin, hi - lo, np.nan)
    under = (2 / ALPHA) * np.clip(np.where(fin, lo - y, 0.0), 0, None)
    over = (2 / ALPHA) * np.clip(np.where(fin, y - hi, 0.0), 0, None)
    IS = np.where(fin, width + under + over, np.nan)
    return cov, width, IS, under, over, (~fin).astype(float)


def summarize_cells(y, lo, hi):
    cov, width, IS, under, over, inf = interval_metrics(y, lo, hi)
    return dict(n=int(len(y)), coverage=float(cov.mean()), width_mean=float(np.nanmean(width)) if np.isfinite(width).any() else np.nan,
                interval_score=float(np.nanmean(IS)) if np.isfinite(IS).any() else np.nan,
                is_under=float(np.nanmean(np.where(np.isfinite(IS), under, np.nan))) if np.isfinite(IS).any() else np.nan,
                is_over=float(np.nanmean(np.where(np.isfinite(IS), over, np.nan))) if np.isfinite(IS).any() else np.nan,
                frac_inf=float(inf.mean())), (cov, width, IS)


STORE = {}     # (region, cond, split, method) -> dict(idx, cov[S,n], width[S,n], IS[S,n], lo[S,n], hi[S,n])


def store(region, cond, split, method, idx, cov, width, IS, lo, hi):
    d = STORE.setdefault((region, cond, split, method), dict(idx=np.asarray(idx), cov=[], width=[], IS=[], lo=[], hi=[]))
    assert np.array_equal(d["idx"], np.asarray(idx))
    d["cov"].append(cov); d["width"].append(width); d["IS"].append(IS); d["lo"].append(lo); d["hi"].append(hi)


def block_boot(stat_by_split, blocks_by_split, ub, nboot, seed=0):
    """지역 블록 재표집 부트스트랩. 통계량 = 분할 평균( 분할 내 셀 평균(seed 평균 셀값) ). 반환 (lo, hi)."""
    rng = np.random.RandomState(seed)
    nb = len(ub); pos = {b: i for i, b in enumerate(ub)}
    M = np.stack([np.bincount(rng.randint(0, nb, nb), minlength=nb) for _ in range(nboot)]).astype(float)
    stats = np.full((nboot, len(stat_by_split)), np.nan)
    for s, (v, blk) in enumerate(zip(stat_by_split, blocks_by_split)):
        fin = np.isfinite(v)
        bi = np.array([pos[b] for b in blk])
        S = np.bincount(bi[fin], weights=v[fin], minlength=nb); N = np.bincount(bi[fin], minlength=nb).astype(float)
        num, den = M @ S, M @ N
        stats[:, s] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    st = np.nanmean(stats, axis=1)
    return float(np.nanpercentile(st, 2.5)), float(np.nanpercentile(st, 97.5))


# ---------------------------------------------------------------- 본 실험
def run():
    t0 = time.time()
    df = load_base(PROC)
    em = eval_mask(df)
    X_all = df[FEATS].values.astype(np.float32)
    y_all = df[TARGET].values.astype(float)
    blk_all = df.block.values
    macro = df.macro.values
    ak_idx = np.where(macro == "Alaska")[0]
    targets = [t for t in TRANSFER_MAIN if (macro == t).any()]
    n_ak = len(ak_idx)
    print(f"[data] {len(df):,}셀 · 알래스카 {n_ak:,}(블록 {np.unique(blk_all[ak_idx]).size}) · 대상 {targets} · "
          f"seeds {SEEDS} · splits {SPLITS} · iters {ITERS} · smoke={args.smoke}", flush=True)

    # 밀도비 가중(공변량만, seed 무관: lbfgs 결정적) · DI(알래스카 전체 참조)
    W, wrows = {}, []
    for tg in targets:
        t_idx = np.where(macro == tg)[0]
        w_src, w_tgt, auc = density_ratio(X_all[ak_idx], X_all[t_idx], seed=0)
        W[tg] = (w_src, w_tgt)
        wrows.append(dict(region=tg, n_src=n_ak, n_tgt=len(t_idx), clf_auc=auc, ess_src_all=ess(w_src),
                          ess_src_frac=ess(w_src) / n_ak, w_src_frac_at_min=float(np.mean(w_src <= 1 / IW_CLIP + 1e-9)),
                          w_src_frac_at_max=float(np.mean(w_src >= IW_CLIP - 1e-9)),
                          w_tgt_median=float(np.median(w_tgt)), w_tgt_frac_at_max=float(np.mean(w_tgt >= IW_CLIP - 1e-9))))
    di_all = np.full(len(df), np.nan)
    tgt_all = np.where(macro != "Alaska")[0]
    ak_folds = spatial_block_splits(df, n_splits=N_REF_FOLDS, sub_idx=ak_idx)
    ak_pos = {c: i for i, c in enumerate(ak_idx)}
    ref_folds = [(np.array([ak_pos[c] for c in tr]), np.array([ak_pos[c] for c in te])) for tr, te in ak_folds]
    di_all[tgt_all], DI_THR, di_ref = compute_di(X_all[ak_idx], X_all[tgt_all], ref_folds)
    aoa_in = di_all <= DI_THR
    print(f"[DI] 대상 DI 중앙값 " + ", ".join(f"{tg} {np.median(di_all[macro == tg]):.2f}" for tg in targets)
          + " · AOA 안 비율 " + ", ".join(f"{tg} {100*np.mean(aoa_in[macro == tg]):.0f}%" for tg in targets), flush=True)

    rows, qlog, coef_log = [], [], []
    pred_med = np.zeros((len(SEEDS), len(df)))
    for si, seed in enumerate(SEEDS):
        ts = time.time()
        rng = np.random.default_rng(1000 + seed)              # S11 규약: 보정 블록 무작위 선택
        ub = np.unique(blk_all[ak_idx]); rng.shuffle(ub)
        n_cb = max(1, int(round(CAL_FRAC * len(ub))))
        cal_b = set(ub[:n_cb].tolist())
        cal_m = np.array([b in cal_b for b in blk_all[ak_idx]])
        prop, cal = ak_idx[~cal_m], ak_idx[cal_m]
        assert not (set(prop.tolist()) & set(cal.tolist()))
        k = fit_coefs(df.iloc[prop])                          # 계수는 proper-train 라벨에서만
        coef_log.append(dict(seed=seed, E=k["E"], cci_a=k["cci_a"], cci_b=k["cci_b"], n_prop=len(prop), n_cal=len(cal),
                             n_cal_blocks=n_cb))
        model = fit_quantiles(X_all[prop], y_all[prop], seed)
        q = predict_q(model, X_all)                           # 전 셀 예측(알래스카 proper 값은 미사용)
        lo, hi, med = q[:, AIDX[0.05]], q[:, AIDX[0.95]], q[:, AIDX[0.50]]
        pred_med[si] = med
        anc = anchor_pred(ANCHOR, df, k)                      # Stefan+CCI (CCI 결측 셀 NaN → 평가 마스크로 배제)
        st = k["E"] * df.e5_sqrt_tdd.values.astype(float)     # Stefan 유사라벨(라벨 미사용)
        E_cal = np.maximum(lo[cal] - y_all[cal], y_all[cal] - hi[cal])
        R_cal = np.abs(y_all[cal] - anc[cal])
        Q_ak_cqr, Q_ak_anc = q_plain(E_cal), q_plain(R_cal)
        qlog.append(dict(seed=seed, region="Alaska", cond="calibration", Q_cqr=Q_ak_cqr, Q_anchor=Q_ak_anc, n_cal=len(cal)))
        print(f"[seed {seed}] proper {len(prop):,} / cal {len(cal):,}({n_cb}블록) · E={k['E']:.2f} · "
              f"Q_cqr={Q_ak_cqr:.1f} cm · Q_anchor={Q_ak_anc:.1f} cm · 적합 {time.time()-ts:.0f}s", flush=True)

        def add(region, cond, split, method, cells, lo_i, hi_i, extra):
            summ, (cov, width, IS) = summarize_cells(y_all[cells], lo_i, hi_i)
            rows.append(dict(region=region, cond=cond, split=split, method=method, seed=seed,
                             n_blocks=int(np.unique(blk_all[cells]).size), **summ, **extra))
            store(region, cond, split, method, cells, cov, width, IS, lo_i, hi_i)

        for tg in targets:
            t_idx = np.where(macro == tg)[0]
            w_src, w_tgt = W[tg]
            w_cal = w_src[cal_m]
            Qiw_cqr = wquantile(E_cal, w_cal, w_tgt, LEVEL)   # 검정 셀별(대상 셀 순서 = t_idx)
            Qiw_anc = wquantile(R_cal, w_cal, w_tgt, LEVEL)
            wrows.append(dict(region=tg, seed=seed, cond="calibration", ess_cal=ess(w_cal), ess_cal_frac=ess(w_cal) / len(cal),
                              n_cal=len(cal), Q_cqr_ak=Q_ak_cqr, Q_cqr_iw_median=float(np.median(Qiw_cqr)),
                              Q_cqr_iw_frac_inf=float(np.mean(~np.isfinite(Qiw_cqr))),
                              Q_anchor_ak=Q_ak_anc, Q_anchor_iw_median=float(np.median(Qiw_anc))))

            def base_intervals(cells):
                pos = np.searchsorted(t_idx, cells)
                assert np.array_equal(t_idx[pos], cells)
                return {
                    "raw": (lo[cells], hi[cells]),
                    "cqr_ak": (lo[cells] - Q_ak_cqr, hi[cells] + Q_ak_cqr),
                    "cqr_iw": (lo[cells] - Qiw_cqr[pos], hi[cells] + Qiw_cqr[pos]),
                    "anchor_ak": (anc[cells] - Q_ak_anc, anc[cells] + Q_ak_anc),
                    "anchor_iw": (anc[cells] - Qiw_anc[pos], anc[cells] + Qiw_anc[pos]),
                }

            # noinfo: 대상 전체 평가 셀
            ev = t_idx[em[t_idx]]
            if len(ev) == 0:
                continue
            iv = base_intervals(ev)
            for mth in METHODS_NOINFO:
                add(tg, "noinfo", 0, mth, ev, *iv[mth], dict(n_cal=len(cal), Q=Q_ak_cqr if mth == "cqr_ak" else
                    Q_ak_anc if mth == "anchor_ak" else np.nan))
            # covonly: A블록 보정(유사라벨 또는 실측) → B블록 채점
            if tg not in COVONLY_REGIONS:
                continue
            for sp in SPLITS:
                A, B = half_split_blocks(df, t_idx, sp)
                evA, evB = A[em[A]], B[em[B]]
                if len(evA) < 3 or len(evB) == 0:
                    continue
                Qp_cqr = q_plain(np.maximum(lo[evA] - st[evA], st[evA] - hi[evA]))
                Qp_anc = q_plain(np.abs(st[evA] - anc[evA]))
                Qr_cqr = q_plain(np.maximum(lo[evA] - y_all[evA], y_all[evA] - hi[evA]))
                Qr_anc = q_plain(np.abs(y_all[evA] - anc[evA]))
                qlog.append(dict(seed=seed, region=tg, cond=f"covonly_A_split{sp}", n_A=len(evA), n_B=len(evB),
                                 Q_cqr_ak=Q_ak_cqr, Q_cqr_pseudo=Qp_cqr, Q_cqr_real=Qr_cqr,
                                 Q_anchor_ak=Q_ak_anc, Q_anchor_pseudo=Qp_anc, Q_anchor_real=Qr_anc))
                iv = base_intervals(evB)
                iv["cqr_pseudo"] = (lo[evB] - Qp_cqr, hi[evB] + Qp_cqr)
                iv["cqr_real"] = (lo[evB] - Qr_cqr, hi[evB] + Qr_cqr)
                iv["anchor_pseudo"] = (anc[evB] - Qp_anc, anc[evB] + Qp_anc)
                iv["anchor_real"] = (anc[evB] - Qr_anc, anc[evB] + Qr_anc)
                Qmap = dict(cqr_ak=Q_ak_cqr, anchor_ak=Q_ak_anc, cqr_pseudo=Qp_cqr, cqr_real=Qr_cqr,
                            anchor_pseudo=Qp_anc, anchor_real=Qr_anc)
                for mth in METHODS_COVONLY:
                    add(tg, "covonly", sp, mth, evB, *iv[mth], dict(n_cal=len(evA) if mth.endswith(("pseudo", "real")) else len(cal),
                                                                   Q=Qmap.get(mth, np.nan)))

        # 알래스카 지역 내 참조(x25, 같은 규약): 6-fold 공간블록, fold train 블록 25% 보정
        if not args.no_alaska_ref:
            rng_ref = np.random.default_rng(2000 + seed)
            folds = spatial_block_splits(df, n_splits=N_REF_FOLDS, sub_idx=ak_idx)
            if args.smoke:
                folds = folds[:2]
            oof = {m: (np.full(len(df), np.nan), np.full(len(df), np.nan)) for m in METHODS_REF}
            te_all = []
            for fi, (tr, te) in enumerate(folds):
                ubf = np.unique(blk_all[tr]); rng_ref.shuffle(ubf)
                ncb = max(1, int(round(CAL_FRAC * len(ubf))))
                cbf = set(ubf[:ncb].tolist())
                cmf = np.array([b in cbf for b in blk_all[tr]])
                pf, cf = tr[~cmf], tr[cmf]
                assert not (set(cf.tolist()) & set(te.tolist())) and not (set(pf.tolist()) & set(te.tolist()))
                kf = fit_coefs(df.iloc[pf])
                mf = fit_quantiles(X_all[pf], y_all[pf], seed)
                qc, qt = predict_q(mf, X_all[cf]), predict_q(mf, X_all[te])
                Qc = q_plain(np.maximum(qc[:, AIDX[0.05]] - y_all[cf], y_all[cf] - qc[:, AIDX[0.95]]))
                anc_c, anc_t = anchor_pred(ANCHOR, df.iloc[cf], kf), anchor_pred(ANCHOR, df.iloc[te], kf)
                Qa = q_plain(np.abs(y_all[cf] - anc_c))
                oof["raw"][0][te], oof["raw"][1][te] = qt[:, AIDX[0.05]], qt[:, AIDX[0.95]]
                oof["cqr_ak"][0][te], oof["cqr_ak"][1][te] = qt[:, AIDX[0.05]] - Qc, qt[:, AIDX[0.95]] + Qc
                oof["anchor_ak"][0][te], oof["anchor_ak"][1][te] = anc_t - Qa, anc_t + Qa
                te_all.append(te)
                qlog.append(dict(seed=seed, region="Alaska", cond=f"in_domain_fold{fi}", n_cal=len(cf), n_test=len(te),
                                 Q_cqr_ak=Qc, Q_anchor_ak=Qa))
            te_all = np.sort(np.concatenate(te_all))
            for mth in METHODS_REF:
                add("Alaska", "in_domain", 0, mth, te_all, oof[mth][0][te_all], oof[mth][1][te_all], dict(n_cal=np.nan, Q=np.nan))
            print(f"[seed {seed}] 알래스카 지역 내 참조 {len(folds)} fold 완료 · 누적 {time.time()-t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{TAG}_results.csv", index=False)

    # ------------------------------------------------ 요약: seed(·분할) 평균 + 블록 부트스트랩 CI
    srows = []
    keys = sorted(set((r, c, m) for (r, c, _s, m) in STORE), key=lambda t: (t[0], t[1], METHODS_COVONLY.index(t[2])))
    for (region, cond, mth) in keys:
        splits = sorted(s for (r, c, s, m) in STORE if (r, c, m) == (region, cond, mth))
        cov_s, w_s, is_s, blk_s, ninf_s = [], [], [], [], []
        for s in splits:
            d = STORE[(region, cond, s, mth)]
            cov_s.append(np.mean(np.stack(d["cov"]), axis=0)); w_s.append(np.nanmean(np.stack(d["width"]), axis=0))
            is_s.append(np.nanmean(np.stack(d["IS"]), axis=0)); blk_s.append(blk_all[d["idx"]])
            ninf_s.append(float(np.mean(~np.isfinite(np.stack(d["width"])))))
        all_idx = np.concatenate([STORE[(region, cond, s, mth)]["idx"] for s in splits])
        if cond == "covonly":                                 # 지역 블록 전체(A∪B)를 재표집 단위로
            ub = np.unique(blk_all[np.where(macro == region)[0]])
        else:
            ub = np.unique(blk_all[all_idx])
        nb_eval = int(np.unique(blk_all[all_idx]).size)
        point = dict(coverage=float(np.mean([np.nanmean(c) for c in cov_s])),
                     width_mean=float(np.nanmean([np.nanmean(w) for w in w_s])),
                     interval_score=float(np.nanmean([np.nanmean(v) for v in is_s])))
        sub = res[(res.region == region) & (res.cond == cond) & (res.method == mth)]
        row = dict(region=region, cond=cond, method=mth, n_splits=len(splits), n_seeds=len(SEEDS),
                   n_cells=int(round(np.mean([len(c) for c in cov_s]))), n_blocks=nb_eval,
                   coverage=point["coverage"], coverage_sd_seed=float(sub.groupby("seed").coverage.mean().std(ddof=0)) if len(SEEDS) > 1 else np.nan,
                   width_mean=point["width_mean"], interval_score=point["interval_score"],
                   is_under=float(sub.is_under.mean()), is_over=float(sub.is_over.mean()), frac_inf=float(np.mean(ninf_s)),
                   Q_mean=float(sub.Q.mean()) if sub.Q.notna().any() else np.nan)
        if nb_eval >= MIN_BLOCKS_CI:
            row["cov_ci_lo"], row["cov_ci_hi"] = block_boot(cov_s, blk_s, ub, NBOOT, seed=0)
            row["is_ci_lo"], row["is_ci_hi"] = block_boot(is_s, blk_s, ub, NBOOT, seed=1)
            row["ci_flag"] = ""
        else:
            row.update(cov_ci_lo=np.nan, cov_ci_hi=np.nan, is_ci_lo=np.nan, is_ci_hi=np.nan, ci_flag=f"blocks<{MIN_BLOCKS_CI}")
        srows.append(row)
    # S11 참조 행(x34, fidelity_base v1)
    try:
        s11 = json.loads((PROC / "s11_conformal_meta.json").read_text())["headline"]
        e4 = pd.read_csv(PROC / "e4_interval_score.csv").set_index("interval")
        for mth, key in (("raw", "raw"), ("cqr_ak", "cqr")):
            srows.append(dict(region="Alaska_S11", cond="in_domain_x34", method=mth, n_splits=6, n_seeds=3, n_cells=13606, n_blocks=74,
                              coverage=s11[f"cov90_{key}"], cov_ci_lo=s11[f"cov90_{key}_ci"][0], cov_ci_hi=s11[f"cov90_{key}_ci"][1],
                              width_mean=s11[f"width90_{key}_cm"], interval_score=float(e4.loc[key, "interval_score"]),
                              is_ci_lo=float(e4.loc[key, "is_ci_lo"]), is_ci_hi=float(e4.loc[key, "is_ci_hi"]),
                              is_under=float(e4.loc[key, "is_under_part"]), is_over=float(e4.loc[key, "is_over_part"]),
                              ci_flag="S11 x34 v1 참조(파일)"))
    except Exception as e:                                    # noqa: BLE001
        print("[warn] S11 참조 행 생략:", e)
    summ = pd.DataFrame(srows)
    summ.to_csv(OUT / f"{TAG}_summary.csv", index=False)

    # ------------------------------------------------ 조건부 커버리지(noinfo, 셀별 seed 평균)
    crows = []
    cell_rows = []
    for tg in targets:
        for mth in METHODS_NOINFO:
            d = STORE.get((tg, "noinfo", 0, mth))
            if d is None:
                continue
            idx = d["idx"]; c = np.mean(np.stack(d["cov"]), 0); w = np.nanmean(np.stack(d["width"]), 0)
            inside = aoa_in[idx]
            for lab, m in (("inside", inside), ("outside", ~inside)):
                if m.sum():
                    crows.append(dict(axis="aoa_di", region=tg, method=mth, bin=lab, n=int(m.sum()), coverage=float(c[m].mean()),
                                      width_mean=float(np.nanmean(w[m]))))
    # 풀링(주 6지역)
    for mth in METHODS_NOINFO:
        parts = [(STORE[(tg, "noinfo", 0, mth)]) for tg in targets if (tg, "noinfo", 0, mth) in STORE]
        idx = np.concatenate([p["idx"] for p in parts])
        c = np.concatenate([np.mean(np.stack(p["cov"]), 0) for p in parts])
        w = np.concatenate([np.nanmean(np.stack(p["width"]), 0) for p in parts])
        inside = aoa_in[idx]
        for lab, m in (("inside", inside), ("outside", ~inside)):
            if m.sum():
                crows.append(dict(axis="aoa_di", region="pooled_main6", method=mth, bin=lab, n=int(m.sum()),
                                  coverage=float(c[m].mean()), width_mean=float(np.nanmean(w[m]))))
        pm = pred_med.mean(0)[idx]
        qb = pd.qcut(pm, 5, labels=[f"P{i}" for i in range(1, 6)], duplicates="drop")
        for b in qb.categories:
            m = np.asarray(qb == b)
            crows.append(dict(axis="pred_quintile", region="pooled_main6", method=mth, bin=str(b), n=int(m.sum()),
                              coverage=float(c[m].mean()), width_mean=float(np.nanmean(w[m])),
                              pred_lo=float(pm[m].min()), pred_hi=float(pm[m].max()), y_mean=float(y_all[idx][m].mean())))
        for tg in ("Lena", "Canada"):
            p = STORE.get((tg, "noinfo", 0, mth))
            if p is None or len(p["idx"]) < 100:
                continue
            pmr = pred_med.mean(0)[p["idx"]]; cr = np.mean(np.stack(p["cov"]), 0); wr = np.nanmean(np.stack(p["width"]), 0)
            qb = pd.qcut(pmr, 5, labels=[f"P{i}" for i in range(1, 6)], duplicates="drop")
            for b in qb.categories:
                m = np.asarray(qb == b)
                crows.append(dict(axis="pred_quintile", region=tg, method=mth, bin=str(b), n=int(m.sum()),
                                  coverage=float(cr[m].mean()), width_mean=float(np.nanmean(wr[m])),
                                  pred_lo=float(pmr[m].min()), pred_hi=float(pmr[m].max()), y_mean=float(y_all[p["idx"]][m].mean())))
    pd.DataFrame(crows).to_csv(OUT / f"{TAG}_conditional.csv", index=False)

    # 셀별(noinfo) seed 평균 구간: 지도용
    for tg in targets:
        d0 = STORE.get((tg, "noinfo", 0, "raw"))
        if d0 is None:
            continue
        idx = d0["idx"]
        cd = df.iloc[idx][["loc_id", "lat", "lon", "region", "macro", "block", TARGET]].copy()
        cd["pred_med"] = pred_med.mean(0)[idx]; cd["di"] = di_all[idx]; cd["aoa_inside"] = aoa_in[idx].astype(int)
        for mth in METHODS_NOINFO:
            d = STORE[(tg, "noinfo", 0, mth)]
            cd[f"lo90_{mth}"] = np.mean(np.stack(d["lo"]), 0); cd[f"hi90_{mth}"] = np.mean(np.stack(d["hi"]), 0)
            cd[f"cov90_{mth}"] = np.mean(np.stack(d["cov"]), 0)
        cell_rows.append(cd)
    pd.concat(cell_rows).to_csv(OUT / f"{TAG}_cells.csv", index=False)
    pd.DataFrame(wrows).to_csv(OUT / f"{TAG}_weights.csv", index=False)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                         # noqa: BLE001
        commit = "NA"
    meta = dict(
        stage="M1-UQ", hypothesis="H17", plan="docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md (개정 2026-09-21 §F)",
        protocol=dict(base="fidelity_base_v3.csv + e5_soil_tdd_v3.csv (F4_direct)", features=f"x25 SHARED_CORE ({len(FEATS)})",
                      model=f"CatBoost MultiQuantile alpha={ALPHA_STR}, iters={ITERS}, lr=0.03, depth=6, l2=3, thread 4",
                      alaska_split=f"블록 {CAL_FRAC:.0%}를 보정(np.random.default_rng(1000+seed) 셔플, S11 규약), 나머지 proper-train",
                      quantile_rule="순서통계 ceil((n+1)(1−α)); 가중판은 Tibshirani 2019 셀별 Q(검정 셀 w 포함), 미달 시 +inf",
                      density_ratio=f"로지스틱(balanced, C=1) 알래스카 전체 대 대상 전체 셀, 소스 평균 1 정규화, 클립 [1/{IW_CLIP:g}, {IW_CLIP:g}]",
                      anchor=ANCHOR, pseudo="stefan (E = 알래스카 proper-train 최소제곱)", alpha=ALPHA,
                      covonly_regions=COVONLY_REGIONS, splits=SPLITS, seeds=SEEDS, nboot=NBOOT, min_blocks_ci=MIN_BLOCKS_CI,
                      di="Meyer 2021 DI, x25 표준화(알래스카 통계), 임계 Q75+1.5IQR", di_threshold=DI_THR,
                      alaska_ref=(not args.no_alaska_ref), alaska_ref_folds=N_REF_FOLDS if not args.smoke else 2),
        coefs=coef_log, q_log=qlog, weights_diag=[r for r in wrows if "seed" not in r],
        aoa_inside_frac={tg: float(np.mean(aoa_in[macro == tg])) for tg in targets},
        n_regions={tg: dict(n=int((macro == tg).sum()), n_eval=int(em[macro == tg].sum()),
                            n_blocks=int(np.unique(blk_all[macro == tg]).size)) for tg in targets},
        caveats=[
            "전이 지역의 교환가능성 가정은 성립하지 않으므로 어떤 방법도 유한표본 보장이 없다. 커버리지는 경험적 값이다.",
            "러시아 W·E는 셀 30 안팎, 러시아 C 7셀, 그린란드 평가 1셀(토양 도일 결측): 지역 수준 판정 불가, 블록 8 미만은 CI 미산출.",
            "밀도비 가중은 분류기가 지역을 거의 완전히 분리하면(AUC≈1) 가중이 클립 경계로 몰려 ESS가 작아진다(weights.csv 참조).",
            "anchor_pseudo의 보정 잔차는 0.5|Stefan−CCI|로 공변량의 결정적 함수(공식 혼합)이다.",
            "알래스카 S11 참조는 x34·fidelity_base v1 규약이라 수치 직접 비교는 x25 지역 내 참조 행(Alaska/in_domain)으로 한다.",
        ],
        git_commit=commit, gpu="미사용(CUDA_VISIBLE_DEVICES='')", runtime_min=round((time.time() - t0) / 60, 1),
    )
    (OUT / f"{TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    pd.set_option("display.width", 220)
    cols = ["region", "cond", "method", "n_cells", "n_blocks", "coverage", "cov_ci_lo", "cov_ci_hi", "width_mean", "interval_score", "frac_inf", "ci_flag"]
    print("\n=== 요약(seed·분할 평균, 블록 부트스트랩 95% CI) ===")
    print(summ[cols].round(3).to_string(index=False))
    print(f"\nsaved: {TAG}_{{results,summary,conditional,weights,cells}}.csv · {TAG}_meta.json ({meta['runtime_min']}분)")


# ---------------------------------------------------------------- 그림
INK, MUTED = "#1f2933", "#5b6470"
# 보정 출처별 색. 유채색 4색은 dataviz validate_palette(light, --pairs all) 전 항목 통과(CVD 최악쌍 ΔE 9.1, 정상시 16.3).
# 회색은 무보정 기준선(의도적 중립). 난색(황)은 대상 실측 라벨을 쓰는 상한에만 배정(한색 = 라벨 미사용 방법).
SRC_COLOR = {"raw": "#8a8f98", "ak": "#2a78d6", "iw": "#1baf7a", "pseudo": "#4a3aa7", "real": "#eda100"}
FAM_MARKER = {"cqr": "o", "anchor": "s"}
METHOD_ORDER = ["raw", "cqr_ak", "cqr_iw", "cqr_pseudo", "cqr_real", "anchor_ak", "anchor_iw", "anchor_pseudo", "anchor_real"]
METHOD_LABEL = {"raw": "무보정 분위", "cqr_ak": "CQR·알래스카 보정", "cqr_iw": "CQR·밀도비 가중",
                "cqr_pseudo": "CQR·대상 유사라벨 재보정", "cqr_real": "CQR·대상 실측 재보정(상한)",
                "anchor_ak": "앵커·알래스카 보정", "anchor_iw": "앵커·밀도비 가중",
                "anchor_pseudo": "앵커·대상 유사라벨 재보정", "anchor_real": "앵커·대상 실측 재보정(상한)"}
REGION_LABEL = {"Alaska_S11": "알래스카 지역 내(S11, x34)", "Alaska": "알래스카 지역 내(x25)", "Lena": "레나", "Canada": "캐나다",
                "Russia_W": "러시아 W", "Russia_C": "러시아 C", "Russia_E": "러시아 E", "Greenland": "그린란드"}


def _style(mth):
    fam = "cqr" if mth.startswith("cqr") or mth == "raw" else "anchor"
    src = "raw" if mth == "raw" else mth.split("_", 1)[1]
    return SRC_COLOR[src], FAM_MARKER[fam]


def make_figures():
    from polar.plotstyle import use_polar
    from matplotlib.lines import Line2D
    plt = use_polar()
    summ = pd.read_csv(OUT / f"{TAG}_summary.csv")
    meta = json.loads((OUT / f"{TAG}_meta.json").read_text())
    summ["ci_flag"] = summ.ci_flag.fillna("")

    def prep_ax(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.7); ax.spines[s].set_color("#7a7a7a")
        ax.grid(False)
        ax.tick_params(labelsize=9, length=2.5, width=0.7, color="#8a8a8a", labelcolor=INK)

    # ---------- 그림 1: 지역별 커버리지 점·CI
    panels = [("noinfo", ["Alaska_S11", "Alaska", "Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"],
               ["raw", "cqr_ak", "cqr_iw", "anchor_ak", "anchor_iw"], "정보 없음(대상 전체 셀)"),
              ("covonly", ["Lena", "Canada", "Russia_W", "Russia_E"],
               ["cqr_ak", "cqr_iw", "cqr_pseudo", "cqr_real", "anchor_ak", "anchor_iw", "anchor_pseudo", "anchor_real"],
               "공변량만(B블록 채점, A블록 보정)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.5))
    fig.subplots_adjust(left=0.13, right=0.80, top=0.90, bottom=0.12)
    for ax, (cond, regions, methods, title) in zip(axes, panels):
        prep_ax(ax)
        regions = [r for r in regions if ((summ.region == r) & (summ.cond.str.startswith(cond if r not in ("Alaska", "Alaska_S11") else "in_domain"))).any()]
        nm = len(methods); step = 0.8 / nm
        yt, ylab = [], []
        for ri, reg in enumerate(regions):
            sub = summ[(summ.region == reg) & (summ.cond.str.startswith(cond if reg not in ("Alaska", "Alaska_S11") else "in_domain"))]
            y0 = -ri
            if ri % 2 == 0:
                ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f2f4f7", zorder=0, lw=0)
            for mi, mth in enumerate(methods):
                r = sub[sub.method == mth]
                if r.empty:
                    continue
                r = r.iloc[0]
                yy = y0 + 0.4 - (mi + 0.5) * step
                col, mk = _style(mth)
                ms = 5.2 if mk == "o" else 4.6
                if np.isfinite(r.cov_ci_lo):
                    ax.plot([r.cov_ci_lo, r.cov_ci_hi], [yy, yy], color=col, lw=1.3, solid_capstyle="butt", zorder=2, alpha=0.9, clip_on=False)
                    ax.plot(r.coverage, yy, marker=mk, ms=ms, color=col, mec="white", mew=0.6, zorder=3, ls="none", clip_on=False)
                else:                                          # 블록 8 미만: 빈 마커, CI 없음
                    ax.plot(r.coverage, yy, marker=mk, ms=ms, mfc="white", mec=col, mew=1.0, zorder=3, ls="none", clip_on=False)
            n = int(sub.n_cells.iloc[0]); nb = int(sub.n_blocks.iloc[0])
            yt.append(y0); ylab.append(f"{REGION_LABEL.get(reg, reg)}\n" + f"n={n:,} · {nb}블록" + ("" if nb >= MIN_BLOCKS_CI else " · CI 없음"))
        ax.axvline(0.9, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.text(0.9, 0.6, "명목 90%", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_yticks(yt); ax.set_yticklabels(ylab, fontsize=8.6, color=INK)
        ax.set_ylim(-len(regions) + 0.45, 0.6)
        ax.set_xlim(0, 1.0); ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
        ax.set_xlabel("90% 구간 경험 커버리지", fontsize=10.5, color=INK)
        ax.grid(True, axis="x", color="#aab3bd", lw=0.5, alpha=0.35); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    handles = [Line2D([0], [0], marker=_style(m)[1], color=_style(m)[0], ls="none", ms=5.2, mec="white", mew=0.6, label=METHOD_LABEL[m])
               for m in METHOD_ORDER]
    handles.append(Line2D([0], [0], marker="o", mfc="white", mec=INK, color="none", ls="none", ms=5.2, mew=1.0, label="빈 마커: 블록 8 미만(CI 미산출)"))
    lg = fig.legend(handles=handles, loc="center left", ncol=1, fontsize=8.2, frameon=False, bbox_to_anchor=(0.805, 0.52),
                    handletextpad=0.6, labelspacing=0.75, title="방법(지역 내 위→아래 순서)", title_fontsize=8.4, alignment="left")
    lg.get_title().set_color(INK)
    for ax, lab in zip(axes, ("(a)", "(b)")):
        ax.text(-0.40, 1.03, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
    nseeds, nsplits = len(meta["protocol"]["seeds"]), len(meta["protocol"]["splits"])
    fig.text(0.465, 0.015, f"점: seed(n={nseeds}){'·분할(n=%d)' % nsplits if nsplits > 1 else ''} 평균 커버리지 · 가로선: 0.5° 블록 부트스트랩 95% CI({meta['protocol']['nboot']}회) · "
             f"원=분위 CatBoost(CQR) 계열, 사각=Stefan+CCI 앵커 계열\nx25 공변량, F4 직접 라벨, α=0.1 · 알래스카 S11 참조는 x34·fidelity_base v1 규약",
             ha="center", va="top", fontsize=7.8, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{TAG}_coverage.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- 그림 2: 폭 대 커버리지(공변량만 B블록, 4지역 소다중)
    regs = ["Lena", "Canada", "Russia_W", "Russia_E"]
    methods = METHOD_ORDER
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    wmax = float(summ[(summ.cond == "covonly")].width_mean.max()) * 1.12
    for ax, reg in zip(axes, regs):
        prep_ax(ax)
        sub = summ[(summ.region == reg) & (summ.cond == "covonly")]
        ax.axhline(0.9, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
        for mth in methods:
            r = sub[sub.method == mth]
            if r.empty:
                continue
            r = r.iloc[0]; col, mk = _style(mth)
            ms = 6.2 if mk == "o" else 5.5
            if np.isfinite(r.cov_ci_lo):
                ax.plot([r.width_mean, r.width_mean], [r.cov_ci_lo, r.cov_ci_hi], color=col, lw=1.1, alpha=0.8, zorder=2, clip_on=False)
                ax.plot(r.width_mean, r.coverage, marker=mk, ms=ms, color=col, mec="white", mew=0.6, ls="none", zorder=3, clip_on=False)
            else:
                ax.plot(r.width_mean, r.coverage, marker=mk, ms=ms, mfc="white", mec=col, mew=1.0, ls="none", zorder=3, clip_on=False)
        n = int(sub.n_cells.iloc[0]); nb = int(sub.n_blocks.iloc[0])
        ax.set_title(f"{REGION_LABEL[reg]}  (B블록 n≈{n:,} · {nb}블록)", fontsize=9.8, color=INK, loc="left", pad=5)
        ax.set_xlim(0, wmax); ax.set_ylim(0, 1.02)
        ax.set_xlabel("평균 구간 폭 (cm)", fontsize=10.5, color=INK)
        ax.grid(True, color="#aab3bd", lw=0.5, alpha=0.3); ax.set_axisbelow(True)
    axes[0].set_ylabel("90% 구간 경험 커버리지", fontsize=10.5, color=INK)
    axes[0].set_yticks(np.arange(0, 1.01, 0.2)); axes[0].set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
    axes[0].text(wmax * 0.98, 0.905, "명목 90%", ha="right", va="bottom", fontsize=8, color=INK)
    handles = [Line2D([0], [0], marker=_style(m)[1], color=_style(m)[0], ls="none", ms=6, mec="white", mew=0.6, label=METHOD_LABEL[m]) for m in methods]
    handles.append(Line2D([0], [0], marker="o", mfc="white", mec=INK, color="none", ls="none", ms=6, mew=1.0, label="빈 마커: 블록 8 미만(CI 미산출)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.12), handletextpad=0.5, columnspacing=1.2)
    for ax, lab in zip(axes, ("(a)", "(b)", "(c)", "(d)")):
        ax.text(-0.04, 1.06, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
    fig.text(0.5, -0.24, "세로선: 커버리지 블록 부트스트랩 95% CI · 원=CQR 계열, 사각=앵커 계열 · 유사라벨·실측 재보정은 같은 지역의 A블록에서 Q를 다시 구한 것 · "
             "점: seed·분할 평균, x25 공변량, α=0.1", ha="center", va="top", fontsize=7.8, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{TAG}_width_coverage.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", FIG / f"{TAG}_coverage.png", FIG / f"{TAG}_width_coverage.png")


if __name__ == "__main__":
    if not args.figure_only:
        run()
    make_figures()
