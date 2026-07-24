"""S2: 물리식 5종 앙상블 baseline + physics-as-feature (A2).

`RESEARCH_PLAN_multifidelity` S2. 목적:
 Part A — 물리 5종(Stefan기본·edaphic·TTOP·Kudryavtsev·λ) 각각의 in-domain·LORO 성능.
          "어느 물리가 어느 지역서 강한가". fold-safe E 역산. LORO 게이트 프로토콜 고정.
 Part B — physics 예측을 ML 입력 feature로(A2): BASE vs +physics, 여러 모델.
게이트 프로토콜(적대검증 반영): LORO 대상=min_test≥100 지역(AK·Lena·CA), 집계=비가중 평균 + 지역별 병기.

산출: data/processed/s2_physics_results.csv, s2_physics_oof.csv(물리 예측·mask), s2_physics_meta.json
실행: python scripts/3_deep_learning/s2_physics_baseline.py   (GPU=6 기본)
"""
import os
GPU = os.environ.get("GPU", "6")
assert GPU in {"6", "7", "8", "9"}, f"GPU는 6,7,8,9만 허용(요청 {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (COVARIATE_CORE, SHARED_CORE, TARGET, add_group_keys,
                            spatial_block_splits, loro_splits, assert_fold_safe_E)
from polar.physics import physics_ensemble, fit_E, PHYSICS_MEMBERS
from polar.tab_models import available_models, fit_predict, NAN_NATIVE, set_device
from polar.preprocessing import fold_prep as prep
from polar.eval_metrics import all_metrics
import torch
set_device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"[GPU guard] CUDA_VISIBLE_DEVICES={GPU} → uuid …{str(torch.cuda.get_device_properties(0).uuid)[-8:]}")

PROC = C.PROCESSED
ALASKA = ["ABoVE_AK", "United States (Alaska)"]
df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
y = df[TARGET].values.astype(float)
sqtdd = df["e5_sqrt_tdd"].values

# 물리 5종. p2-p5·ttop·mask는 E무관. p1(라벨의존)은 시각화·feature용으로 global E 사용
# (평가는 아래 fold별 E로 별도 계산). E=1.0 스케일 버그 수정(must_fix).
E_GLOBAL = fit_E(y, sqtdd)
phys0 = physics_ensemble(df, E=E_GLOBAL)
rows = []

# p1_stefan을 OOF에 저장할 때 쓸 fold-safe E OOF (E=1.0 스케일 버그 수정, must_fix)
p1_oof_ak = np.full(len(df), np.nan)   # 알래스카 셀만 채움

# ================= Part A: 물리 5종 baseline =================
print("[Part A] 물리 5종 baseline (fold-safe E)")

# in-domain 알래스카 공간블록: 각 물리 멤버 + phys_mean
ak = df[df.region.isin(ALASKA)].reset_index(drop=True)
ak_idx = df.index[df.region.isin(ALASKA)].values
folds_ak = spatial_block_splits(ak, n_splits=6)
for member in PHYSICS_MEMBERS + ["phys_mean"]:
    oof = np.full(len(ak), np.nan)
    for tr, te in folds_ak:
        if member == "p1_stefan":
            assert_fold_safe_E(tr, te, ak)   # 프로덕션 배선: train/test 교집합 없음 보증
            E = fit_E(ak[TARGET].values[tr], ak["e5_sqrt_tdd"].values[tr])
            oof[te] = E * ak["e5_sqrt_tdd"].values[te]
        else:
            glob = phys0[member][ak_idx]
            oof[te] = glob[te]
    if member == "p1_stefan":
        p1_oof_ak[ak_idx] = oof   # fold-safe p1 OOF 보존
    m = all_metrics(ak[TARGET].values, oof)
    rows.append(dict(part="A_physics", cv="spatial_block_AK", model=member, **m))
    print(f"  {member:16s} in-domain RMSE {m['rmse_cm']:.2f} bias {m['bias_cm']:+.2f}")

# LORO 전이: 물리 멤버 (fold-safe E from train regions)
loro = loro_splits(df, min_test=100)
loro_regions = [r for r, _, _ in loro]
print(f"[Part A] LORO 지역: {loro_regions}")
for member in PHYSICS_MEMBERS + ["phys_mean"]:
    reg_rmse = {}
    for r, tr, te in loro:
        if member == "p1_stefan":
            assert_fold_safe_E(tr, te, df)
            E = fit_E(y[tr], sqtdd[tr])
            pred = E * sqtdd[te]
        else:
            pred = phys0[member][te]
        m = all_metrics(y[te], pred)
        rows.append(dict(part="A_physics", cv="LORO", model=member, region=r, **m))
        reg_rmse[r] = m["rmse_cm"]
    unweighted = np.mean([v for v in reg_rmse.values() if np.isfinite(v)])
    rows.append(dict(part="A_physics", cv="LORO_gate", model=member, region="UNWEIGHTED_MEAN",
                     rmse_cm=unweighted, n=len(loro)))
    print(f"  {member:16s} LORO " + " ".join(f"{k[:8]}={v:.1f}" for k, v in reg_rmse.items())
          + f" | 게이트(비가중평균) {unweighted:.2f}")

# ================= Part B: physics-as-feature (A2) =================
print("[Part B] physics feature 효과 (BASE vs +physics)")
# physics feature = E-무관 물리 예측 6개(p2·p4·ttop·perm_mask·phys_std·phys_mean).
# p1(라벨의존)은 fold-safe E 필요하므로 feature에서 제외(누설 방지).


def build_phys_features(sub_df, sub_idx):
    """E-무관 물리 feature(누설 없음). p1(라벨의존)은 제외."""
    return np.column_stack([
        phys0["p2_edaphic"][sub_idx], phys0["p4_kudryavtsev"][sub_idx],
        phys0["ttop"][sub_idx], phys0["perm_mask"][sub_idx],
        phys0["phys_std"][sub_idx], phys0["phys_mean"][sub_idx]])


models_B = [m for m in ["catboost", "mlp", "lightgbm"] if m in available_models()]
SEEDS = [0, 1, 2]
# in-domain AK: BASE(FULL34) vs +physics
Xak_full = ak[COVARIATE_CORE].values.astype(np.float32)
phys_ak = build_phys_features(ak, ak_idx)
for model in models_B:
    for tag, X in [("BASE", Xak_full), ("+physics", np.column_stack([Xak_full, phys_ak]))]:
        for seed in (SEEDS if model not in NAN_NATIVE else [0]):
            oof = np.full(len(ak), np.nan)
            for tr, te in folds_ak:
                Xtr, Xte = prep(X[tr], X[te], model in NAN_NATIVE)
                oof[te] = fit_predict(model, Xtr, ak[TARGET].values[tr], Xte, seed=seed, epochs=120)["pred"]
            m = all_metrics(ak[TARGET].values, oof)
            rows.append(dict(part="B_feature", cv="spatial_block_AK", model=f"{model}_{tag}", seed=seed, **m))
    b = np.mean([r["rmse_cm"] for r in rows if r.get("model") == f"{model}_BASE" and r["cv"] == "spatial_block_AK"])
    p = np.mean([r["rmse_cm"] for r in rows if r.get("model") == f"{model}_+physics" and r["cv"] == "spatial_block_AK"])
    print(f"  {model:10s} in-domain: BASE {b:.2f} → +physics {p:.2f}  (Δ {p-b:+.2f})")

# LORO: BASE(SHARED25) vs +physics
Xall_sh = df[SHARED_CORE].values.astype(np.float32)
phys_all = build_phys_features(df, df.index.values)
for model in models_B:
    for tag, X in [("BASE", Xall_sh), ("+physics", np.column_stack([Xall_sh, phys_all]))]:
        for seed in (SEEDS if model not in NAN_NATIVE else [0]):
            for r, tr, te in loro:
                Xtr, Xte = prep(X[tr], X[te], model in NAN_NATIVE)
                pred = fit_predict(model, Xtr, y[tr], Xte, seed=seed, epochs=120)["pred"]
                m = all_metrics(y[te], pred)
                rows.append(dict(part="B_feature", cv="LORO", model=f"{model}_{tag}", seed=seed, region=r, **m))

# ================= 저장 =================
res = pd.DataFrame(rows)
res.to_csv(PROC / "s2_physics_results.csv", index=False)
# 물리 예측 OOF (시각화·source-aware용)
oof_df = df[["loc_id", "lat", "lon", "region", "block", TARGET]].copy()
for m in PHYSICS_MEMBERS:
    oof_df[m] = phys0[m]
oof_df["phys_mean"] = phys0["phys_mean"]; oof_df["phys_std"] = phys0["phys_std"]
oof_df["ttop"] = phys0["ttop"]; oof_df["perm_mask"] = phys0["perm_mask"]
oof_df["p1_stefan_calib_ak"] = p1_oof_ak   # fold-safe OOF(알래스카만); 나머지는 global E
oof_df.to_csv(PROC / "s2_physics_oof.csv", index=False)

gate = res[(res.cv == "LORO_gate") & (res.model == "p1_stefan")]["rmse_cm"].iloc[0]
import subprocess
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(C.ROOT)).decode().strip()
except Exception:
    commit = "NA"
meta = dict(purpose="S2 물리 5종 baseline + physics feature", E_global=float(E_GLOBAL),
            loro_regions=loro_regions, gate_protocol="LORO 비가중평균(macro: Alaska·Lena·Canada)",
            p1_stefan_LORO_gate=float(gate), models_B=models_B, gpu=GPU, git_commit=commit,
            note="p1 시각화·feature=global E; 평가=fold별 E. OOF E=1.0 버그 수정됨.")
(PROC / "s2_physics_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"\n[done] LORO 하한 게이트 p1_stefan(비가중평균) = {gate:.2f}cm. 물리 baseline·feature 저장.")
