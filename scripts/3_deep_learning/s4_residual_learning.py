"""S4: Stefan 앵커 + 저용량 잔차학습 fallback (게이트식, A4·H2 재검증).

`RESEARCH_PLAN_multifidelity` S4. 핵심 질문: 과거 "잔차학습 무익(48cm)"은 covariate shift
심한 토양 입력 조건의 판정이었다. shift-robust 입력(지형6+기후8)·저용량 모델·shrinkage λ로
잔차학습이 물리 앵커(LORO 게이트 22.24cm, S2)를 넘는지 게이트식 재검증.

설계:
 - 예측 = E_train·√TDD + λ·g(x). E는 train fold 내부만(fold-safe), g는 train 잔차 학습.
 - 잔차 모델 용량 사다리: ridge(선형) < catboost_lo(depth3·200iter) < catboost(depth6·600iter).
 - 입력 2종: shift14(지형+기후, shift-robust) vs shared25(+토양·CCI, W3 실패 조건 대조).
 - λ ∈ {0, 0.25, 0.5, 0.75, 1.0} 스윕(사후 곡선) + inner 공간블록 CV로 λ 자동선택(정직 배포판).
 - 평가: LORO(매크로 Alaska·Lena·Canada, 비가중평균 게이트) + in-domain AK 공간블록.
 - 게이트: LORO 비가중평균 ≤ S2 p1_stefan 22.24cm. 미달 시 negative 확정(P2 재확인).

산출: data/processed/s4_residual_{results,oof}.csv, s4_residual_meta.json
실행: GPU=6 python scripts/3_deep_learning/s4_residual_learning.py
"""
import os
GPU = os.environ.get("GPU", "4")
assert GPU in {"2", "3", "4", "5", "6", "7", "8", "9"}, f"GPU는 2-9만 허용(요청 {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (TERRAIN, CLIMATE, SHARED_CORE, TARGET, add_group_keys,
                            spatial_block_splits, loro_splits, macro_region,
                            fit_stefan_E, assert_fold_safe_E)
from polar.preprocessing import fold_prep as prep
from polar.eval_metrics import all_metrics

PROC = C.PROCESSED
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FEATSETS = {"shift14": TERRAIN + CLIMATE, "shared25": SHARED_CORE}
MODELS = ["ridge", "catboost_lo", "catboost"]
SEEDS = {"ridge": [0], "catboost_lo": [0, 1, 2], "catboost": [0, 1, 2]}
NBOOT = 300
ALASKA = ["ABoVE_AK", "United States (Alaska)"]

df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
df["macro"] = macro_region(df)
y = df[TARGET].values.astype(float)
sqtdd = df["e5_sqrt_tdd"].values.astype(float)
s2_gate = json.loads((PROC / "s2_physics_meta.json").read_text())["p1_stefan_LORO_gate"]
print(f"[S4] n={len(df)} · 게이트(S2 p1_stefan LORO 비가중평균) = {s2_gate:.2f}cm")


def fit_residual(model, Xtr, rtr, Xte, seed):
    """잔차 g(x) 학습·예측. ridge는 fold-safe 표준화, catboost는 NaN native."""
    if model == "ridge":
        from sklearn.linear_model import Ridge
        Xtr2, Xte2 = prep(Xtr, Xte, nan_native=False)
        m = Ridge(alpha=10.0)
        m.fit(Xtr2, rtr)
        return np.asarray(m.predict(Xte2))
    from catboost import CatBoostRegressor
    p = (dict(iterations=200, learning_rate=0.05, depth=3) if model == "catboost_lo"
         else dict(iterations=600, learning_rate=0.03, depth=6))
    m = CatBoostRegressor(**p, l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                          allow_writing_files=False)
    m.fit(Xtr, rtr)
    return np.asarray(m.predict(Xte))


rng = np.random.RandomState(42)


def block_bootstrap_delta(yv, base_pred, aug_pred, blocks, nboot=NBOOT):
    """블록 리샘플 ΔRMSE(base−aug) CI. 양수=잔차가 Stefan-only 대비 개선."""
    ublk = np.unique(blocks)
    deltas = []
    for _ in range(nboot):
        samp = np.concatenate([np.where(blocks == k)[0]
                               for k in rng.choice(ublk, len(ublk), replace=True)])
        rb = np.sqrt(np.nanmean((yv[samp] - base_pred[samp]) ** 2))
        ra = np.sqrt(np.nanmean((yv[samp] - aug_pred[samp]) ** 2))
        deltas.append(rb - ra)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


rows = []
# ============ Part A: LORO 전이 (게이트) ============
loro = loro_splits(df, min_test=100)
loro_regions = [r for r, _, _ in loro]
print(f"[Part A] LORO 지역: {loro_regions}")
# 예측 저장: preds[(region, model, fs, seed)] = g_te / stefan[(region)] = stefan_te
g_store, stefan_store, te_store = {}, {}, {}
for r, tr, te in loro:
    assert_fold_safe_E(tr, te, df)
    E = fit_stefan_E(y[tr], sqtdd[tr])
    stefan_te = E * sqtdd[te]
    stefan_store[r], te_store[r] = stefan_te, te
    resid_tr = y[tr] - E * sqtdd[tr]
    ok = np.isfinite(resid_tr)
    m0 = all_metrics(y[te], stefan_te)
    print(f"  [{r}] E={E:.2f} Stefan-only RMSE {m0['rmse_cm']:.2f} (n={m0['n']})")
    for fs, cols in FEATSETS.items():
        X = df[cols].values.astype(np.float32)
        for model in MODELS:
            for seed in SEEDS[model]:
                g_te = fit_residual(model, X[tr][ok], resid_tr[ok], X[te], seed)
                g_store[(r, model, fs, seed)] = g_te
                for lam in LAMBDAS:
                    m = all_metrics(y[te], stefan_te + lam * g_te)
                    rows.append(dict(part="A_loro", cv="LORO", region=r, model=model,
                                     featset=fs, lam=lam, seed=seed, **m))

# 게이트 집계: (model, fs, λ)별 지역 seed-mean RMSE의 비가중평균
res_a = pd.DataFrame([x for x in rows if x["part"] == "A_loro"])
for (model, fs, lam), g in res_a.groupby(["model", "featset", "lam"]):
    per_reg = g.groupby("region").rmse_cm.mean()
    rows.append(dict(part="A_loro", cv="LORO_gate", region="UNWEIGHTED_MEAN", model=model,
                     featset=fs, lam=lam, rmse_cm=float(per_reg.mean()), n=len(per_reg)))
gate_tbl = pd.DataFrame([x for x in rows if x["cv"] == "LORO_gate"])
print("\n[게이트 곡선] LORO 비가중평균 RMSE (λ):")
for (model, fs), g in gate_tbl.groupby(["model", "featset"]):
    s = " ".join(f"λ{lam:g}={v:.2f}" for lam, v in zip(g.lam, g.rmse_cm))
    print(f"  {model:12s} {fs:8s} {s}")

# ============ Part B: λ 자동선택 (inner 공간블록 CV, 정직 배포판) ============
print("\n[Part B] inner 공간블록 CV로 λ 자동선택 (test 라벨 미사용)")
for r, tr, te in loro:
    tr_df = df.iloc[tr].reset_index(drop=True)
    ytr_all, sq_tr = y[tr], sqtdd[tr]
    inner = spatial_block_splits(tr_df, n_splits=3)
    for fs, cols in FEATSETS.items():
        X = df[cols].values.astype(np.float32)
        Xtr_all = X[tr]
        for model in MODELS:
            lam_rmse = {lam: [] for lam in LAMBDAS}
            for itr, ite in inner:
                E_i = fit_stefan_E(ytr_all[itr], sq_tr[itr])
                resid_i = ytr_all[itr] - E_i * sq_tr[itr]
                ok_i = np.isfinite(resid_i)
                g_i = fit_residual(model, Xtr_all[itr][ok_i], resid_i[ok_i], Xtr_all[ite], 0)
                stef_i = E_i * sq_tr[ite]
                for lam in LAMBDAS:
                    e = ytr_all[ite] - (stef_i + lam * g_i)
                    lam_rmse[lam].append(float(np.sqrt(np.nanmean(e ** 2))))
            lam_star = min(LAMBDAS, key=lambda l: np.mean(lam_rmse[l]))
            # seed-mean 예측으로 test 평가
            g_te = np.mean([g_store[(r, model, fs, s)] for s in SEEDS[model]], axis=0)
            m = all_metrics(y[te], stefan_store[r] + lam_star * g_te)
            rows.append(dict(part="B_autolam", cv="LORO_autolam", region=r, model=model,
                             featset=fs, lam=lam_star, **m))
res_b = pd.DataFrame([x for x in rows if x["part"] == "B_autolam"])
for (model, fs), g in res_b.groupby(["model", "featset"]):
    rows.append(dict(part="B_autolam", cv="LORO_autolam_gate", region="UNWEIGHTED_MEAN",
                     model=model, featset=fs, lam=float(g.lam.mean()),
                     rmse_cm=float(g.rmse_cm.mean()), n=len(g)))
    print(f"  {model:12s} {fs:8s} λ*={list(g.lam)} → 게이트 {g.rmse_cm.mean():.2f}")

# ============ Part C: 블록부트스트랩 CI (지역별, λ별 vs Stefan-only) ============
print("\n[Part C] 블록부트스트랩 CI (catboost_lo·shift14, seed-mean)")
for r, _, te in loro:
    blocks = df["block"].values[te]
    g_te = np.mean([g_store[(r, "catboost_lo", "shift14", s)] for s in SEEDS["catboost_lo"]], axis=0)
    for lam in LAMBDAS[1:]:
        d, lo, hi = block_bootstrap_delta(y[te], stefan_store[r], stefan_store[r] + lam * g_te, blocks)
        sig = "유의" if lo > 0 else ("악화유의" if hi < 0 else "CI 0 포함")
        rows.append(dict(part="C_boot", cv="LORO_boot", region=r, model="catboost_lo",
                         featset="shift14", lam=lam, delta_rmse=d, ci_lo=lo, ci_hi=hi))
        print(f"  [{r}] λ={lam:g} ΔRMSE {d:+.2f} [{lo:+.2f}, {hi:+.2f}] {sig}")

# ============ Part D: in-domain AK 공간블록 (참조) ============
print("\n[Part D] in-domain 알래스카 공간블록")
ak = df[df.region.isin(ALASKA)].reset_index(drop=True)
folds_ak = spatial_block_splits(ak, n_splits=6)
yak, sqak = ak[TARGET].values.astype(float), ak["e5_sqrt_tdd"].values.astype(float)
for fs, cols in FEATSETS.items():
    Xak = ak[cols].values.astype(np.float32)
    for model in MODELS:
        for seed in SEEDS[model]:
            oof = {lam: np.full(len(ak), np.nan) for lam in LAMBDAS}
            for tr, te in folds_ak:
                assert_fold_safe_E(tr, te, ak)
                E = fit_stefan_E(yak[tr], sqak[tr])
                resid_tr = yak[tr] - E * sqak[tr]
                ok = np.isfinite(resid_tr)
                g_te = fit_residual(model, Xak[tr][ok], resid_tr[ok], Xak[te], seed)
                for lam in LAMBDAS:
                    oof[lam][te] = E * sqak[te] + lam * g_te
            for lam in LAMBDAS:
                m = all_metrics(yak, oof[lam])
                rows.append(dict(part="D_indomain", cv="spatial_block_AK", region="Alaska",
                                 model=model, featset=fs, lam=lam, seed=seed, **m))
res_d = pd.DataFrame([x for x in rows if x["part"] == "D_indomain"])
for (model, fs), g in res_d.groupby(["model", "featset"]):
    s = " ".join(f"λ{lam:g}={v:.2f}" for lam, v in g.groupby("lam").rmse_cm.mean().items())
    print(f"  {model:12s} {fs:8s} {s}")

# ============ 저장 + 게이트 판정 ============
res = pd.DataFrame(rows)
res.to_csv(PROC / "s4_residual_results.csv", index=False)

# OOF (LORO, catboost_lo·shift14 seed-mean): 지도용
oof_df = df[["loc_id", "lat", "lon", "region", "macro", "block", TARGET]].copy()
oof_df["stefan_pred"] = np.nan
for lam in LAMBDAS[1:]:
    oof_df[f"pred_lam{lam:g}"] = np.nan
for r, _, te in loro:
    oof_df.iloc[te, oof_df.columns.get_loc("stefan_pred")] = stefan_store[r]
    g_te = np.mean([g_store[(r, "catboost_lo", "shift14", s)] for s in SEEDS["catboost_lo"]], axis=0)
    for lam in LAMBDAS[1:]:
        oof_df.iloc[te, oof_df.columns.get_loc(f"pred_lam{lam:g}")] = stefan_store[r] + lam * g_te
oof_df.to_csv(PROC / "s4_residual_oof.csv", index=False)

best = gate_tbl.loc[gate_tbl.rmse_cm.idxmin()]
auto_gate = pd.DataFrame([x for x in rows if x["cv"] == "LORO_autolam_gate"])
best_auto = auto_gate.loc[auto_gate.rmse_cm.idxmin()]
gate_pass = bool((best.rmse_cm <= s2_gate) and (best.lam > 0))
import subprocess
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(C.ROOT)).decode().strip()
except Exception:
    commit = "NA"
meta = dict(purpose="S4 Stefan 앵커 + 저용량 잔차 shrinkage 게이트", s2_gate_cm=float(s2_gate),
            lambdas=LAMBDAS, featsets={k: len(v) for k, v in FEATSETS.items()}, models=MODELS,
            loro_regions=loro_regions,
            best_posthoc=dict(model=best.model, featset=best.featset, lam=float(best.lam),
                              gate_rmse=float(best.rmse_cm)),
            best_autolam=dict(model=best_auto.model, featset=best_auto.featset,
                              gate_rmse=float(best_auto.rmse_cm)),
            gate_pass_posthoc=gate_pass, gpu=GPU, git_commit=commit,
            note="예측=E_train·√TDD+λ·g(x). 사후 λ 곡선은 진단용, 배포 판정은 autolam(정직).")
(PROC / "s4_residual_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"\n[done] 게이트 {s2_gate:.2f} | 사후최선 {best.model}/{best.featset}/λ{best.lam:g} = "
      f"{best.rmse_cm:.2f} ({'통과' if gate_pass else '미달'}) | "
      f"자동λ최선 {best_auto.model}/{best_auto.featset} = {best_auto.rmse_cm:.2f}")
