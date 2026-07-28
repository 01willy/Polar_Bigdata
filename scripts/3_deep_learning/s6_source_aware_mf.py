"""S6: source-aware multi-fidelity (A5) — 구조적 증분 실증 게이트.

`RESEARCH_PLAN_multifidelity` S6. 관측모델 y_s = z(x) + b_s + ε_s, ε_s~N(0, σ_s(x)²).
공유 MLP 인코더 + source별 스칼라 b_s(direct=0 고정) + source별 log σ_s(x) 헤드, Gaussian NLL.
clean 소스는 Stefan pseudo(fold-safe E·√TDD)·CCI 2개(overlap 100%·99.5%). 둘 다 이미
feature로 쓸 수 있으므로 "Stefan-feature+CCI-feature baseline 대비 구조적 증분"을 실증한다.

비교군(동일 공변량·동일 fold 강제):
 (a) direct_catboost / direct_mlp  — 실측-only(S1 재현 수준)
 (b) pool_catboost / pool_mlp      — naive pooling(모든 소스 라벨 동일 취급)
 (c) fixaug_catboost               — 고정가중 증강(S3 최적 stefan·r=10, 표본가중 등가 구현)
 (d) physfeat_catboost             — Stefan-feature 추가(CCI는 SHARED/FULL feature에 이미 포함)
 (ref) stefan_anchor               — E_train·√TDD(최소제곱 E, S4에서 LORO 21.26 앵커)
 (A5) sa_z / sa_fusion             — 인코더 z(x) 단독 / test 셀 보조관측 정밀도가중 융합

정보량 공정 regime: 보조소스(Stefan·CCI)는 test 셀에서도 가용으로 취급한다. 기존 baseline이
cci_alt·e5_sqrt_tdd를 test 셀 feature로 이미 사용하므로 전 방법 동일 정보량이다(직접 라벨은
어떤 방법도 test 셀 것을 쓰지 않음. Stefan pseudo=E_train×√TDD는 test 라벨 무관).

평가: (1) 알래스카 in-domain 공간블록 6-fold RMSE, (2) LORO 매크로 게이트(Alaska/Lena/Canada
비가중평균; 기준 S2 Stefan 22.24·S4 최고 21.26), (3) cov90(σ 헤드 vs quantile CatBoost).
3 seeds + 블록 부트스트랩 CI. 판정: LORO 정확도 우위 또는 cov90 보정 우위 중 하나라도
실증되면 partial/adopted, 아니면 negative(단 b_s·σ_s 진단은 UQ 부산물로 산출).

산출: data/processed/s6_source_aware_{results.csv,meta.json}
실행: GPU=3 python scripts/3_deep_learning/s6_source_aware_mf.py   (SMOKE=1 빠른 점검)
"""
import os
# ★ GPU 고정은 어떤 import(특히 torch)보다 먼저. 오늘(2026-07-27)은 3,4,5만 허용.
GPU = os.environ.get("GPU", "3")
assert GPU in {"3", "4", "5"}, f"GPU는 3,4,5만 허용(요청 {GPU}, 2026-07-27 지시)"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import sys, json, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (TERRAIN, CLIMATE, SOIL, COVARIATE_CORE, SHARED_CORE, TARGET,
                            add_group_keys, spatial_block_splits, loro_splits, macro_region,
                            fit_stefan_E, assert_fold_safe_E)
from polar.preprocessing import fold_prep as prep
from polar.eval_metrics import all_metrics
from polar.tab_models import fit_predict, set_device
from polar.source_aware import fit_source_aware, predict_source_aware, fuse_posterior

import torch
set_device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"[GPU guard] CUDA_VISIBLE_DEVICES={GPU} → 논리0 = 물리 GPU "
          f"(uuid …{str(torch.cuda.get_device_properties(0).uuid)[-8:]})")

T0 = time.time()
SMOKE = os.environ.get("SMOKE", "0") == "1"
PROC = C.PROCESSED
SEEDS = [0] if SMOKE else [0, 1, 2]
NBOOT = 30 if SMOKE else 300
SA_EPOCHS = 25 if SMOKE else 150
MLP_EPOCHS = 20 if SMOKE else 120
CB_ITERS = 200 if SMOKE else 600
R_FIX = 10.0                       # S3 최적 증강비율(고정가중 비교군)
Z90 = 1.6449                       # 90% 양측 구간 z-값
S2_GATE, S4_GATE = 22.24, 21.26    # 기준: S2 Stefan 앵커 / S4 최소제곱 E 앵커
ENC_LORO = TERRAIN + CLIMATE + SOIL                                # 23 (CCI는 관측으로만)
FEAT_LORO = SHARED_CORE                                            # 25 (CCI feature 포함)
ENC_IND = [c for c in COVARIATE_CORE if c not in ("cci_alt", "cci_valid")]   # 32
FEAT_IND = COVARIATE_CORE                                          # 34
SOURCES = ["direct", "stefan", "cci"]
BOOT_PAIRS = [("direct_catboost", "sa_fusion"), ("physfeat_catboost", "sa_fusion"),
              ("pool_catboost", "sa_fusion"), ("fixaug_catboost", "sa_fusion"),
              ("stefan_anchor", "sa_fusion"), ("stefan_anchor", "sa_z"),
              ("direct_catboost", "sa_z"), ("direct_mlp", "sa_z"), ("pool_mlp", "sa_z")]

df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
df["macro"] = macro_region(df)
if SMOKE:
    ak_s = df[df.macro == "Alaska"].sample(n=3000, random_state=0)
    df = pd.concat([ak_s, df[df.macro != "Alaska"]]).reset_index(drop=True)
print(f"[S6] n={len(df)} · SMOKE={SMOKE} · seeds={SEEDS}")

rng_boot = np.random.RandomState(42)


def fit_catboost(Xtr, ytr, Xte, seed, loss=None, weights=None):
    from catboost import CatBoostRegressor
    kw = dict(loss_function=loss) if loss else {}
    m = CatBoostRegressor(iterations=CB_ITERS, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                          random_seed=seed, verbose=0, allow_writing_files=False,
                          thread_count=8, **kw)
    m.fit(Xtr, ytr, sample_weight=weights)
    return np.asarray(m.predict(Xte))


def cb_with_q(Xtr, ytr, Xte, seed, weights=None):
    """CatBoost 점예측 + quantile 5/95% 구간(비교군 cov90 산출)."""
    pred = fit_catboost(Xtr, ytr, Xte, seed, weights=weights)
    lo = fit_catboost(Xtr, ytr, Xte, seed, loss="Quantile:alpha=0.05", weights=weights)
    hi = fit_catboost(Xtr, ytr, Xte, seed, loss="Quantile:alpha=0.95", weights=weights)
    return pred, lo, hi


def block_bootstrap_delta(yv, base_pred, comp_pred, blocks, nboot=NBOOT):
    """블록 리샘플 ΔRMSE(base−comp) CI. 양수=comp(sa) 개선."""
    ublk = np.unique(blocks)
    deltas = []
    for _ in range(nboot):
        samp = np.concatenate([np.where(blocks == k)[0]
                               for k in rng_boot.choice(ublk, len(ublk), replace=True)])
        rb = np.sqrt(np.nanmean((yv[samp] - base_pred[samp]) ** 2))
        rc = np.sqrt(np.nanmean((yv[samp] - comp_pred[samp]) ** 2))
        deltas.append(rb - rc)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run_fold(sub, tr, te, feat_cols, enc_cols, axis, region, rows, diag_rows):
    """한 fold의 전 방법 학습·예측. 반환: preds/lo/hi dict — {method: {seed: array}}."""
    y = sub[TARGET].values.astype(float)
    sqtdd = sub["e5_sqrt_tdd"].values.astype(float)
    cci = sub["cci_alt"].values.astype(float)
    cci_ok = np.isfinite(cci)
    assert_fold_safe_E(tr, te, sub)
    E = fit_stefan_E(y[tr], sqtdd[tr])
    stefan_all = E * sqtdd            # 전 셀. E는 train 라벨만(fold-safe)
    stef_ok = np.isfinite(stefan_all)  # √TDD 결측 셀(예: GTNPenv_AQ 1개) 제외
    Xb = sub[feat_cols].values.astype(np.float32)
    Xe = sub[enc_cols].values.astype(np.float32)
    aux = np.concatenate([tr, te])    # 보조소스 관측 셀(공정 regime: test 셀 포함)
    aux_stef = aux[stef_ok[aux]]
    aux_cci = aux[cci_ok[aux]]
    P, L, H = {}, {}, {}

    def put(method, seed, pred, lo=None, hi=None):
        m = all_metrics(y[te], pred, lower=lo, upper=hi)
        rows.append(dict(axis=axis, region=region, method=method, seed=seed, E_train=round(E, 4), **m))
        P.setdefault(method, {})[seed] = pred
        if lo is not None:
            L.setdefault(method, {})[seed] = lo
            H.setdefault(method, {})[seed] = hi

    # (ref) Stefan 앵커
    put("stefan_anchor", 0, stefan_all[te])

    for seed in SEEDS:
        # (a) direct-only
        pred, lo, hi = cb_with_q(Xb[tr], y[tr], Xb[te], seed)
        put("direct_catboost", seed, pred, lo, hi)
        Xtr2, Xte2 = prep(Xb[tr], Xb[te], nan_native=False)
        put("direct_mlp", seed, fit_predict("mlp", Xtr2, y[tr], Xte2, seed=seed,
                                            epochs=MLP_EPOCHS)["pred"])
        # (d) Stefan-feature 추가(CCI는 이미 feature)
        Xp = np.column_stack([Xb, stefan_all.astype(np.float32)])
        pred, lo, hi = cb_with_q(Xp[tr], y[tr], Xp[te], seed)
        put("physfeat_catboost", seed, pred, lo, hi)
        # (b) naive pooling: direct + stefan(aux) + cci(aux) 라벨 동일 취급
        Xpool = np.vstack([Xb[tr], Xb[aux_stef], Xb[aux_cci]])
        ypool = np.concatenate([y[tr], stefan_all[aux_stef], cci[aux_cci]])
        pred, lo, hi = cb_with_q(Xpool, ypool, Xb[te], seed)
        put("pool_catboost", seed, pred, lo, hi)
        Xtr2, Xte2 = prep(Xpool, Xb[te], nan_native=False)
        put("pool_mlp", seed, fit_predict("mlp", Xtr2, ypool, Xte2, seed=seed,
                                          epochs=MLP_EPOCHS)["pred"])
        # (c) 고정가중 증강(S3 최적: stefan·r=10, target 셀). 물리 오버샘플링 대신
        #     표본가중 w = r·n_tr/n_te 등가 구현(기대손실 동일, 실행시간 절감).
        te_st = te[stef_ok[te]]
        w_ps = R_FIX * len(tr) / max(len(te_st), 1)
        Xfx = np.vstack([Xb[tr], Xb[te_st]])
        yfx = np.concatenate([y[tr], stefan_all[te_st]])
        wfx = np.concatenate([np.ones(len(tr)), np.full(len(te_st), w_ps)])
        pred, lo, hi = cb_with_q(Xfx, yfx, Xb[te], seed, weights=wfx)
        put("fixaug_catboost", seed, pred, lo, hi)
        # (A5) source-aware: 관측 = direct(tr) + stefan(aux) + cci(aux)
        Xobs = np.vstack([Xe[tr], Xe[aux_stef], Xe[aux_cci]])
        yobs = np.concatenate([y[tr], stefan_all[aux_stef], cci[aux_cci]])
        sobs = np.concatenate([np.zeros(len(tr)), np.ones(len(aux_stef)),
                               np.full(len(aux_cci), 2)]).astype(int)
        Xobs2, Xte_e = prep(Xobs, Xe[te], nan_native=False)
        fit = fit_source_aware(Xobs2, yobs, sobs, seed=seed, epochs=SA_EPOCHS)
        out = predict_source_aware(fit, Xte_e)
        z, sig, b_cm = out["z"], out["sig"], out["b_cm"]
        put("sa_z", seed, z, z - Z90 * sig[:, 0], z + Z90 * sig[:, 0])
        fz, fs = fuse_posterior(z, sig[:, 0],
                                [(stefan_all[te], b_cm[1], sig[:, 1], stef_ok[te]),
                                 (cci[te], b_cm[2], sig[:, 2], cci_ok[te])])
        put("sa_fusion", seed, fz, fz - Z90 * fs, fz + Z90 * fs)
        # 진단: 학습 b_s·σ_s vs 실제 소스 bias(train은 학습가능 정보, test는 진단 전용)
        for si, sname in enumerate(SOURCES):
            ys_tr = {"direct": y, "stefan": stefan_all, "cci": cci}[sname]
            m_tr = np.isfinite(ys_tr[tr]) & np.isfinite(y[tr])
            m_te = np.isfinite(ys_tr[te]) & np.isfinite(y[te])
            diag_rows.append(dict(
                axis=axis, region=region, seed=seed, source=sname,
                b_learned_cm=float(b_cm[si]),
                sig_learned_te_cm=float(np.nanmean(sig[:, si])),
                b_emp_train_cm=float(np.mean((ys_tr[tr] - y[tr])[m_tr])),
                sig_emp_train_cm=float(np.std((ys_tr[tr] - y[tr])[m_tr])),
                b_emp_test_cm=float(np.mean((ys_tr[te] - y[te])[m_te])),
                sig_emp_test_cm=float(np.std((ys_tr[te] - y[te])[m_te]))))
    return P, L, H


def seed_mean(d):
    return {m: np.mean([v for v in sd.values()], axis=0) for m, sd in d.items()}


rows, diag_rows, boot_rows = [], [], []

# ============ Part A: LORO 전이(게이트) ============
print("\n[Part A] LORO 매크로 전이 (Alaska·Lena·Canada)")
loro = loro_splits(df, min_test=100)
for r, tr, te in loro:
    t0 = time.time()
    P, L, H = run_fold(df, tr, te, FEAT_LORO, ENC_LORO, "LORO", r, rows, diag_rows)
    Pm = seed_mean(P)
    yv = df[TARGET].values.astype(float)[te]
    blocks = df["block"].values[te]
    for base, comp in BOOT_PAIRS:
        if base in Pm and comp in Pm:
            d, lo, hi = block_bootstrap_delta(yv, Pm[base], Pm[comp], blocks)
            boot_rows.append(dict(axis="LORO", region=r, base=base, comp=comp,
                                  delta_rmse=d, ci_lo=lo, ci_hi=hi))
    msg = " ".join(f"{m}={np.sqrt(np.nanmean((yv - p) ** 2)):.1f}" for m, p in sorted(Pm.items()))
    print(f"  [{r}] ({time.time() - t0:.0f}s) {msg}")

# 게이트 집계: 방법별 지역 seed-mean RMSE 비가중평균
res_a = pd.DataFrame([x for x in rows if x["axis"] == "LORO"])
for method, g in res_a.groupby("method"):
    per_reg = g.groupby("region").rmse_cm.mean()
    cov = g.groupby("region").coverage_90.mean()
    wid = g.groupby("region").width_90.mean()
    rows.append(dict(axis="LORO_gate", region="UNWEIGHTED_MEAN", method=method, seed=-1,
                     rmse_cm=float(per_reg.mean()),
                     coverage_90=(float(cov.mean()) if cov.notna().all() else np.nan),
                     width_90=(float(wid.mean()) if wid.notna().all() else np.nan),
                     n=int(len(per_reg))))
gate_tbl = pd.DataFrame([x for x in rows if x["axis"] == "LORO_gate"]).sort_values("rmse_cm")
print("\n[게이트] LORO 비가중평균 RMSE(cm)·cov90:")
for _, x in gate_tbl.iterrows():
    c = f" cov90={x.coverage_90:.3f}" if np.isfinite(x.coverage_90) else ""
    print(f"  {x.method:18s} {x.rmse_cm:6.2f}{c}")

# ============ Part B: 알래스카 in-domain 공간블록 ============
print("\n[Part B] 알래스카 in-domain 공간블록")
ak = df[df.macro == "Alaska"].reset_index(drop=True)
folds_ak = spatial_block_splits(ak, n_splits=3 if SMOKE else 6)
oofP, oofL, oofH = {}, {}, {}
for fi, (tr, te) in enumerate(folds_ak):
    P, L, H = run_fold(ak, tr, te, FEAT_IND, ENC_IND, "AK_fold", f"fold{fi}", rows, diag_rows)
    for m, sd in P.items():
        for s, p in sd.items():
            oofP.setdefault((m, s), np.full(len(ak), np.nan))[te] = p
    for store, src in [(oofL, L), (oofH, H)]:
        for m, sd in src.items():
            for s, p in sd.items():
                store.setdefault((m, s), np.full(len(ak), np.nan))[te] = p
yak = ak[TARGET].values.astype(float)
methods_ind = sorted({m for m, _ in oofP})
for m in methods_ind:
    seeds_m = sorted({s for mm, s in oofP if mm == m})
    for s in seeds_m:
        lo = oofL.get((m, s)); hi = oofH.get((m, s))
        mm = all_metrics(yak, oofP[(m, s)], lower=lo, upper=hi)
        rows.append(dict(axis="spatial_block_AK", region="Alaska", method=m, seed=s, **mm))
    sm = np.mean([oofP[(m, s)] for s in seeds_m], axis=0)
    rows.append(dict(axis="spatial_block_AK_seedmean", region="Alaska", method=m, seed=-1,
                     **all_metrics(yak, sm)))
oofPm = {m: np.mean([oofP[(m, s)] for mm, s in oofP if mm == m], axis=0) for m in methods_ind}
for base, comp in BOOT_PAIRS:
    if base in oofPm and comp in oofPm:
        d, lo, hi = block_bootstrap_delta(yak, oofPm[base], oofPm[comp], ak["block"].values)
        boot_rows.append(dict(axis="spatial_block_AK", region="Alaska", base=base, comp=comp,
                              delta_rmse=d, ci_lo=lo, ci_hi=hi))
ind_tbl = pd.DataFrame([x for x in rows if x["axis"] == "spatial_block_AK_seedmean"]).sort_values("rmse_cm")
print("\n[in-domain] AK 공간블록 OOF(seed-mean) RMSE(cm):")
for _, x in ind_tbl.iterrows():
    print(f"  {x.method:18s} {x.rmse_cm:6.2f}")

# ============ 저장 + 판정 ============
res = pd.DataFrame(rows)
for br in boot_rows:
    res = pd.concat([res, pd.DataFrame([dict(axis=f"boot_{br['axis']}", region=br["region"],
                                             method=f"{br['comp']}_vs_{br['base']}", seed=-1,
                                             delta_rmse=br["delta_rmse"], ci_lo=br["ci_lo"],
                                             ci_hi=br["ci_hi"])])], ignore_index=True)
diag = pd.DataFrame(diag_rows)
diag["method"] = "sa_diag"
diag["axis_kind"] = "diagnostic"
res = pd.concat([res, diag.rename(columns={"axis": "axis"})], ignore_index=True)
res.to_csv(PROC / "s6_source_aware_results.csv", index=False)

sa_gate = float(gate_tbl[gate_tbl.method.isin(["sa_z", "sa_fusion"])].rmse_cm.min())
base_methods = ["stefan_anchor", "direct_catboost", "direct_mlp", "physfeat_catboost",
                "pool_catboost", "pool_mlp", "fixaug_catboost"]
base_gate = float(gate_tbl[gate_tbl.method.isin(base_methods)].rmse_cm.min())
base_best = gate_tbl[gate_tbl.method.isin(base_methods)].iloc[0].method
cov_tbl = gate_tbl.dropna(subset=["coverage_90"])
# cov90 판정에 구간폭 건전성 강제: 보정된 가우시안 90% 폭은 3.29·σ ≈ 3.29·RMSE이므로,
# 폭 > 4·RMSE 는 무정보(발산) 구간으로 간주하고 보정 우위 후보에서 제외한다.
cov_err, width_sane = {}, {}
for _, x in cov_tbl.iterrows():
    cov_err[x.method] = abs(x.coverage_90 - 0.90)
    width_sane[x.method] = bool(np.isfinite(x.width_90) and x.width_90 <= 4.0 * x.rmse_cm)
sa_cov = min((v for k, v in cov_err.items() if k.startswith("sa_") and width_sane[k]),
             default=np.nan)
bl_cov = min((v for k, v in cov_err.items() if not k.startswith("sa_") and width_sane[k]),
             default=np.nan)
acc_win = sa_gate < base_gate
cov_win = np.isfinite(sa_cov) and np.isfinite(bl_cov) and sa_cov < bl_cov
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=str(C.ROOT)).decode().strip()
except Exception:
    commit = "NA"
meta = dict(
    purpose="S6 source-aware multi-fidelity(A5): 공유 인코더 + source별 (b_s, log σ_s), Gaussian NLL",
    obs_model="y_s = z(x) + b_s + eps_s, eps_s~N(0, sigma_s(x)^2), b_direct=0 고정",
    sources=SOURCES, seeds=SEEDS, nboot=NBOOT, smoke=SMOKE, gpu=GPU, git_commit=commit,
    enc_features={"LORO": len(ENC_LORO), "AK": len(ENC_IND)},
    baseline_features={"LORO": len(FEAT_LORO), "AK": len(FEAT_IND)},
    fair_regime=("보조소스(Stefan pseudo=E_train·√TDD, CCI)는 test 셀에서도 가용 취급. "
                 "baseline도 cci_alt·e5_sqrt_tdd를 test 셀 feature로 사용하므로 동일 정보량. "
                 "직접 라벨은 어떤 방법도 test 것 미사용(fold-safe E, assert 배선)."),
    fixaug_note="S3 최적(stefan·r=10)을 표본가중 w=r·n_tr/n_te로 등가 구현(기대손실 동일)",
    gates_ref=dict(s2_stefan=S2_GATE, s4_anchor=S4_GATE),
    loro_gate={m: float(v) for m, v in zip(gate_tbl.method, gate_tbl.rmse_cm)},
    loro_cov90={x.method: float(x.coverage_90) for _, x in cov_tbl.iterrows()},
    loro_width90={x.method: float(x.width_90) for _, x in cov_tbl.iterrows()},
    width_sane=width_sane,
    cov90_note="cov90 우위 판정은 폭 건전성(width_90 ≤ 4·RMSE) 통과 방법만 대상. "
               "과대폭 구간의 커버리지 1.0은 무정보(공허)로 배제한다.",
    sa_best_gate=sa_gate, baseline_best_gate=base_gate, baseline_best_method=base_best,
    accuracy_win=bool(acc_win), cov90_win=bool(cov_win),
    verdict=("partial/adopted 후보" if (acc_win or cov_win) else "negative(진단 산출물만 채택)"),
    runtime_min=round((time.time() - T0) / 60, 1),
)
(PROC / "s6_source_aware_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"\n[done] sa 최고 게이트 {sa_gate:.2f} vs baseline 최고 {base_gate:.2f}({base_best}) · "
      f"정확도 우위={acc_win} · cov90 우위={cov_win} · {meta['runtime_min']}분")
