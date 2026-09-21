"""A2-1 공변량 이동 진단 — 알래스카(학습) 대 대상 지역(정보 없음 전이 평가 셀).

지역별(주 6 + 심부 5)로 네 가지를 산출한다.
  (i)   도메인 분류기 AUC: catboost_lo 급 분류기(200 iter, depth 3, 균형 가중), 알래스카=0 대 대상=1, x25, 층화 5-fold OOF.
        proxy A-distance = 2(1 − 2ε), ε = OOF 균형 오류율(임계 0.5). x14도 병기.
  (ii)  표준화 x25(알래스카 평균·표준편차, 결측은 알래스카 중앙값)의 선형·RBF MMD²(알래스카 부표본 2,000, 대상 ≤2,000).
        RBF 대역폭 = 합동 표본 쌍 거리 중앙값. RBF는 라벨 순열 p값(200회) 병기.
  (iii) AOA 비유사도 DI(Meyer & Pebesma 2021, 가중 없음): 학습 최근접 거리 / 학습 CV 최근접 거리 평균,
        임계 = 알래스카 6-fold 공간블록 OOF DI의 Q3 + 1.5·IQR. x25·x14 각각 중앙값·Q3·AOA 밖 비율.
  (iv)  공변량별 표준화 평균 차 SMD = (평균_대상 − 평균_알래스카)/표준편차_알래스카, |SMD| 상위 5.
        추가: 알래스카 학습 범위[min, max] 밖 셀 비율(SWE 단일·x25 중 하나라도).
산출: data/processed/m1/a2_shift_diagnostics.csv (지역 1행), a2_shift_smd.csv (지역×공변량), a2_shift_cells.csv (셀 수준 DI·범위 플래그)
실행: python3 scripts/2_evaluation/a2_shift_diagnostics.py [--sources F4_direct,F4_calm_temp] [--nsub 2000] [--nperm 200]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import (Base, OUT, INPUT_SETS, SOIL, aoa_di, out_of_range, standardize_on_alaska,   # noqa: E402
                       flag_small, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="F4_direct,F4_calm_temp")
ap.add_argument("--nsub", type=int, default=2000)
ap.add_argument("--nperm", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-all", action="store_true")
args = ap.parse_args()
t0 = time.time()
base = Base(sources=tuple(args.sources.split(",")), eval_all=args.eval_all)
print(base.describe(), flush=True)
rng = np.random.RandomState(args.seed)


# ---------------------------------------------------------------- (i) 도메인 분류기
def domain_auc(Xa, Xt, seed=0):
    """OOF AUC·균형 오류율·proxy A-distance. 대상 셀이 5 미만이면 fold 수를 대상 셀 수로 줄인다(2 이상)."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    X = np.vstack([Xa, Xt]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xt))]
    k = int(min(5, len(Xt)))
    if k < 2:
        return np.nan, np.nan, np.nan, k
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y):
        clf = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                                 verbose=0, allow_writing_files=False, thread_count=4, auto_class_weights="Balanced")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    eps = 1.0 - float(balanced_accuracy_score(y, (oof >= 0.5).astype(int)))
    return auc, eps, 2.0 * (1.0 - 2.0 * eps), k


# ---------------------------------------------------------------- (ii) MMD
def mmd_stats(Zs, Zt, nperm=200, seed=0):
    """선형 MMD²(평균 차 제곱 노름)·RBF MMD²(편향 추정, 중앙값 대역폭)·RBF 순열 p값."""
    from scipy.spatial.distance import cdist
    lin = float(np.sum((Zs.mean(0) - Zt.mean(0)) ** 2))
    Z = np.vstack([Zs, Zt]); ns = len(Zs)
    D2 = cdist(Z, Z, "sqeuclidean")
    med = np.median(np.sqrt(D2[np.triu_indices(len(Z), 1)]))
    K = np.exp(-D2 / (2 * med ** 2))

    def mmd2(idx_s, idx_t):
        return float(K[np.ix_(idx_s, idx_s)].mean() + K[np.ix_(idx_t, idx_t)].mean() - 2 * K[np.ix_(idx_s, idx_t)].mean())
    obs = mmd2(np.arange(ns), np.arange(ns, len(Z)))
    r = np.random.RandomState(seed); cnt = 0
    for _ in range(nperm):
        p = r.permutation(len(Z))
        if mmd2(p[:ns], p[ns:]) >= obs:
            cnt += 1
    return lin, obs, (cnt + 1) / (nperm + 1), float(med)


# ---------------------------------------------------------------- (iii) DI, (iv) SMD·범위
feats25, feats14 = INPUT_SETS["x25"], INPUT_SETS["x14"]
di25, thr25, _ = aoa_di(base, feats25)
di14, thr14, _ = aoa_di(base, feats14)
oor25 = out_of_range(base, feats25)
Z25, mu25, sd25 = standardize_on_alaska(base, feats25)
Xraw = base.df[feats25].values.astype(float)
ak_mean, ak_sd = np.nanmean(Xraw[base.ak], 0), np.nanstd(Xraw[base.ak], 0) + 1e-9
ak_sub = rng.choice(base.ak, min(args.nsub, len(base.ak)), replace=False)

rows, smd_rows, cell_rows = [], [], []
for tg in base.regions:
    idx = base.eval_idx(tg)
    te = base.df.iloc[idx]
    n = len(idx)
    # (i)
    auc25, eps25, ad25, k25 = domain_auc(Xraw[base.ak], Xraw[idx], seed=args.seed)
    X14 = base.df[feats14].values.astype(float)
    auc14, eps14, ad14, _ = domain_auc(X14[base.ak], X14[idx], seed=args.seed)
    # (ii)
    t_sub = idx if n <= args.nsub else rng.choice(idx, args.nsub, replace=False)
    lin, rbf, p_rbf, bw = mmd_stats(Z25[ak_sub], Z25[t_sub], nperm=args.nperm, seed=args.seed)
    # (iii)
    d25, d14 = di25[idx], di14[idx]
    # (iv)
    smd = (np.nanmean(Xraw[idx], 0) - ak_mean) / ak_sd
    order = np.argsort(-np.abs(smd))
    top5 = "; ".join(f"{feats25[j]}:{smd[j]:+.2f}" for j in order[:5])
    for j, f in enumerate(feats25):
        smd_rows.append(dict(target=tg, covariate=f, smd=round(float(smd[j]), 4), mean_target=float(np.nanmean(Xraw[idx, j])),
                             mean_alaska=float(ak_mean[j]), sd_alaska=float(ak_sd[j]),
                             frac_out_of_range=round(float(oor25.iloc[idx, j].mean()), 4)))
    o = oor25.iloc[idx]
    n_oor = o.sum(1).values
    top_oor = o.mean(0).sort_values(ascending=False)
    soil_missing = te[list(SOIL)].isna().any(axis=1).values
    rows.append(dict(target=tg, n_cells=n, n_blocks=int(te.block.nunique()), flag=flag_small(n),
                     auc_x25=round(auc25, 3), eps_bal_x25=round(eps25, 3), a_dist_x25=round(ad25, 3), cv_folds=k25,
                     auc_x14=round(auc14, 3), a_dist_x14=round(ad14, 3),
                     mmd2_linear=round(lin, 3), mmd2_rbf=round(rbf, 4), p_perm_rbf=round(p_rbf, 3), rbf_bandwidth=round(bw, 3),
                     di_x25_median=round(float(np.median(d25)), 3), di_x25_q3=round(float(np.percentile(d25, 75)), 3),
                     out_aoa_x25=round(float(np.mean(d25 > thr25)), 3),
                     di_x14_median=round(float(np.median(d14)), 3), out_aoa_x14=round(float(np.mean(d14 > thr14)), 3),
                     smd_top5=top5, swe_out_of_range=round(float(o["e5_swe"].mean()), 3),
                     any_x25_out_of_range=round(float((n_oor > 0).mean()), 3), mean_n_out_of_range=round(float(n_oor.mean()), 2),
                     top_out_of_range=f"{top_oor.index[0]}:{top_oor.iloc[0]:.2f}", soil_missing=round(float(soil_missing.mean()), 3)))
    for i, gi in enumerate(idx):
        cell_rows.append(dict(loc_id=int(te.loc_id.values[i]), target=tg, block=int(te.block.values[i]),
                              di_x25=round(float(d25[i]), 4), in_aoa_x25=bool(d25[i] <= thr25),
                              di_x14=round(float(d14[i]), 4), in_aoa_x14=bool(d14[i] <= thr14),
                              swe_oor=bool(o["e5_swe"].values[i]), n_oor_x25=int(n_oor[i]), soil_missing=bool(soil_missing[i])))
    print(f"  [{tg}] n={n} AUC x25={auc25:.3f} (A={ad25:.2f}) x14={auc14:.3f} · MMD lin={lin:.2f} rbf={rbf:.4f} p={p_rbf:.3f} · "
          f"DI med={np.median(d25):.2f} AOA밖={np.mean(d25 > thr25):.2f} · SWE범위밖={o['e5_swe'].mean():.2f} · {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT / "a2_shift_diagnostics.csv", index=False)
pd.DataFrame(smd_rows).to_csv(OUT / "a2_shift_smd.csv", index=False)
pd.DataFrame(cell_rows).to_csv(OUT / "a2_shift_cells.csv", index=False)
(OUT / "a2_shift_diagnostics_meta.json").write_text(json.dumps(dict(
    sources=args.sources, nsub=args.nsub, nperm=args.nperm, seed=args.seed, eval_all=args.eval_all,
    di_threshold_x25=thr25, di_threshold_x14=thr14, n_alaska=int(len(base.ak)), E_alaska=base.k["E"],
    regions=base.regions, skipped=base.skipped, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1))
print("\n=== A2-1 공변량 이동 진단 (알래스카 학습 → 대상 평가 셀) ===")
print(out[["target", "n_cells", "flag", "auc_x25", "a_dist_x25", "auc_x14", "mmd2_linear", "mmd2_rbf", "p_perm_rbf",
           "di_x25_median", "out_aoa_x25", "out_aoa_x14", "swe_out_of_range", "any_x25_out_of_range", "soil_missing"]].to_string(index=False))
print("\nSMD 상위 5:")
for _, r in out.iterrows():
    print(f"  {r.target:15s} {r.smd_top5}")
print(f"\nDI 임계 x25={thr25:.3f} x14={thr14:.3f} · 저장: {OUT/'a2_shift_diagnostics.csv'} · {time.time()-t0:.0f}s")
