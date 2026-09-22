"""H29 블록 라벨 가치 — 대상 A블록의 블록 하나를 통째로 라벨링했을 때 B 오차가 얼마나 줄어드는가(캐나다·레나·알래스카 하위 지역). CPU.

계획 docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md §3 H29, §4 A5.
값 = 물리식(E0) RMSE − 방법 RMSE (B 셀). 방법: S1 E 수축(κ=10) · S3 잔차 CatBoost(λ=.25, α=1, seed 2).
블록 지표(라벨 미사용): 셀 수, 원천 대비 표준화 평균차(SMD), 공변량 분산. 라벨 기반(진단): 블록 내 y/√TDD 표준편차, 블록 E − E0.
k-중심 규칙이 고르는 블록: n=10, 반복 20 에서 블록별 선택 빈도 → 선택 블록의 평균 가치 대 전체 평균.
산출 data/processed/h3/h29_block_value.csv, h29_summary.csv
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET                                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="Canada,Lena")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--lam", type=float, default=0.25)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--n-rule", type=int, default=10)
ap.add_argument("--reps", type=int, default=20)
args = ap.parse_args()
PROC = ROOT / "data" / "processed"; OUT = PROC / "h3"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
DF = load_base(PROC); DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def cb(Xtr, ytr, Xte, seed):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed, verbose=0, allow_writing_files=False, thread_count=4)
    m.fit(Xtr, ytr); return np.asarray(m.predict(Xte), float)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


rows = []
t0 = time.time()
for t in args.targets.split(","):
    t_idx = np.where(DF.macro.values == t)[0]
    src = DF.iloc[np.setdiff1d(np.where(np.isfinite(DF.y.values))[0], t_idx)]
    E0 = ls_E(src.y.values, src.s.values)
    X_src = src[FEATS].values.astype(np.float32); R_src = src.y.values - E0 * src.s.values
    Xs_mean, Xs_sd = np.nanmean(X_src, 0), np.nanstd(X_src, 0) + 1e-9
    for sp in range(args.splits):
        A_idx, B_idx = half_split_blocks(DF, t_idx, sp); evB = B_idx[eval_mask(DF.iloc[B_idx])]
        A, B = DF.iloc[A_idx], DF.iloc[evB]
        XA, yA, sA = A[FEATS].values.astype(np.float32), A.y.values, A.s.values
        XB, yB, sB = B[FEATS].values.astype(np.float32), B.y.values, B.s.values
        phys = rmse(yB, E0 * sB)
        # k-중심 선택 빈도(n=10, 반복 20)
        from sklearn.cluster import KMeans
        Xa = XA.astype(float); med = np.nanmedian(Xa, 0); Xa = np.where(np.isnan(Xa), np.where(np.isfinite(med), med, 0.0), Xa)
        Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
        freq = pd.Series(0.0, index=np.unique(A.block.values))
        for rep in range(args.reps):
            km = KMeans(n_clusters=min(args.n_rule, len(Z)), n_init=3, random_state=rep).fit(Z); sel = []
            for c in km.cluster_centers_:
                d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
            for b in A.block.values[sel]:
                freq[b] += 1.0 / args.reps
        for b in np.unique(A.block.values):
            m = A.block.values == b
            n_b = int(m.sum())
            E_b = ls_E(yA[m], sA[m]); E_n = (n_b * E_b + args.kappa * E0) / (n_b + args.kappa)
            v1 = phys - rmse(yB, E_n * sB)
            g = np.mean([cb(np.vstack([X_src, XA[m]]), np.concatenate([R_src, yA[m] - E_n * sA[m]]), XB, sd) for sd in range(args.seeds)], 0)
            v3 = phys - rmse(yB, E_n * sB + args.lam * g)
            Xb = XA[m].astype(float)
            rows.append(dict(target=t, split=sp, block=int(b), n_cells=n_b, lat=float(A.lat.values[m].mean()), lon=float(A.lon.values[m].mean()),
                             value_S1=v1, value_S3=v3, E_block=E_b, E_block_minus_E0=E_b - E0, within_sd_ratio=float(np.nanstd(yA[m] / sA[m])) if n_b >= 2 else np.nan,
                             smd_src=float(np.nanmean(np.abs(np.nanmean(Xb, 0) - Xs_mean) / Xs_sd)), cov_var=float(np.nanmean(np.nanvar(Xb, 0) / Xs_sd ** 2)) if n_b >= 2 else np.nan,
                             kmedoid_freq=float(freq[b]), phys_rmse=phys, E0=E0))
        print(f"  [{t}|{sp}] 블록 {A.block.nunique()} · {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "h29_block_value.csv", index=False)
summ = []
for t, sub in R.groupby("target"):
    ok = sub.dropna(subset=["within_sd_ratio"])
    rho1, p1 = spearmanr(ok.within_sd_ratio, ok.value_S3); rho2, p2 = spearmanr(sub.n_cells, sub.value_S3); rho3, p3 = spearmanr(sub.smd_src, sub.value_S3)
    rho4, p4 = spearmanr(sub.E_block_minus_E0.abs(), sub.value_S1)
    w = sub.kmedoid_freq / sub.kmedoid_freq.sum() if sub.kmedoid_freq.sum() > 0 else None
    summ.append(dict(target=t, n_blocks=len(sub), value_S1_mean=sub.value_S1.mean(), value_S3_mean=sub.value_S3.mean(), frac_harm_S1=float((sub.value_S1 < 0).mean()),
                     frac_harm_S3=float((sub.value_S3 < 0).mean()), rho_withinSD_S3=rho1, p_withinSD=p1, rho_ncells_S3=rho2, rho_smd_S3=rho3, rho_absdE_S1=rho4,
                     value_S3_kmedoid_weighted=float((sub.value_S3 * w).sum()) if w is not None else np.nan, value_S1_kmedoid_weighted=float((sub.value_S1 * w).sum()) if w is not None else np.nan))
S = pd.DataFrame(summ); S.to_csv(OUT / "h29_summary.csv", index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string(index=False)); print(f"done {time.time()-t0:.0f}s")
