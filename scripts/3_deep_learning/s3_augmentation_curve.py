"""S3: 물리 pseudo-label 증강비율 반응곡선 (RQ1·RQ3, 엄격 통제).

`RESEARCH_PLAN_multifidelity` S3. 핵심 질문: 물리 pseudo-label이 OOD(전이)에서 언제 돕고
언제 편향을 증폭하는가. Δskill = f(r, 물리종류, 가중).

설계(엄격 통제, 전체 리뷰 반영):
 - source = 알래스카 실측(SHARED_CORE 25, SAR 제외).
 - target(Lena/Canada) 셀을 공간블록 2분할 → pseudo그룹(물리 라벨)·test그룹(실측 라벨). block-disjoint = 거리버퍼.
 - r ∈ {0,0.25,0.5,1,2,5,10} × physics ∈ {stefan(p1), kudryavtsev(p4), placebo(알래스카 평균 상수)}.
 - placebo = 물리 구조 없는 pseudo. 물리가 placebo보다 유의 개선해야 "물리 정보 유효".
 - fold-safe: Stefan E는 알래스카 train만으로 역산. pseudo와 test는 공간블록 분리.
 - 블록부트스트랩 CI로 Δskill 유의성. 과거 14.26 누설 수치 미사용(거리버퍼로 pseudo≠test).

산출: data/processed/s3_aug_curve_results.csv, s3_aug_curve_meta.json
실행: GPU=6 python scripts/3_deep_learning/s3_augmentation_curve.py   (SMOKE=1 빠른점검)
"""
import os
GPU = os.environ.get("GPU", "6")
assert GPU in {"6", "7", "8", "9"}, f"GPU는 6-9만 (요청 {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (SHARED_CORE, TARGET, add_group_keys, spatial_block_splits,
                            macro_region)
from polar.physics import physics_ensemble, fit_E
from polar.tab_models import available_models, fit_predict, NAN_NATIVE, set_device
from polar.preprocessing import fold_prep as prep
from polar.eval_metrics import all_metrics, skill_over_mean
import torch
set_device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"[GPU guard] uuid …{str(torch.cuda.get_device_properties(0).uuid)[-8:]}")

SMOKE = os.environ.get("SMOKE", "0") == "1"
PROC = C.PROCESSED
R_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
PHYS = ["stefan", "kudryavtsev", "placebo"]
MODELS = [m for m in (["catboost", "mlp"] if not SMOKE else ["catboost"]) if m in available_models()]
SEEDS = [0, 1, 2] if not SMOKE else [0]
NBOOT = 300 if not SMOKE else 30

df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
df["macro"] = macro_region(df)
E_alaska = fit_E(df[df.macro == "Alaska"][TARGET].values, df[df.macro == "Alaska"]["e5_sqrt_tdd"].values)
phys_all = physics_ensemble(df, E=E_alaska)  # pseudo 라벨용(알래스카 E)
print(f"[S3] E_alaska={E_alaska:.3f} · models={MODELS} · targets=Lena,Canada")

src = df[df.macro == "Alaska"].reset_index(drop=True)
Xsrc = src[SHARED_CORE].values.astype(np.float32)
ysrc = src[TARGET].values.astype(float)
n_src = len(src)
rng = np.random.RandomState(42)
rows = []


def block_bootstrap_delta(y, base_pred, aug_pred, blocks, nboot=NBOOT):
    """블록 리샘플로 ΔRMSE(base−aug) CI. 양수=aug 개선."""
    ublk = np.unique(blocks)
    deltas = []
    for b in range(nboot):
        samp = np.concatenate([np.where(blocks == k)[0] for k in rng.choice(ublk, len(ublk), replace=True)])
        rb = np.sqrt(np.mean((y[samp] - base_pred[samp]) ** 2))
        ra = np.sqrt(np.mean((y[samp] - aug_pred[samp]) ** 2))
        deltas.append(rb - ra)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


for target in (["Lena", "Canada"] if not SMOKE else ["Lena"]):
    tgt = df[df.macro == target].reset_index(drop=True)
    if len(tgt) < 100:
        continue
    # 공간블록 2분할: pseudo그룹 vs test그룹 (거리버퍼)
    folds = spatial_block_splits(tgt, n_splits=2)
    pseudo_idx, test_idx = folds[0][1], folds[0][0]  # (te, tr) → 첫 fold의 te=pseudo, tr=test
    tgt_idx_global = df.index[df.macro == target].values
    Xtest = tgt[SHARED_CORE].values.astype(np.float32)[test_idx]
    ytest = tgt[TARGET].values.astype(float)[test_idx]
    test_blocks = tgt["block"].values[test_idx]
    Xpseudo_all = tgt[SHARED_CORE].values.astype(np.float32)[pseudo_idx]
    pseudo_global = tgt_idx_global[pseudo_idx]
    print(f"[{target}] pseudo {len(pseudo_idx)}셀 · test {len(test_idx)}셀 (공간블록 분리)")

    for model in MODELS:
        native = model in NAN_NATIVE
        base_preds = {}  # seed→pred (r=0)
        for phys in PHYS:
            for r in R_GRID:
                n_ps = int(r * n_src)
                for seed in (SEEDS if (r > 0 or phys == PHYS[0]) else SEEDS):
                    # pseudo 라벨 생성
                    if n_ps == 0:
                        Xtr, ytr = Xsrc, ysrc
                    else:
                        take = min(n_ps, len(pseudo_idx))
                        sel = np.random.RandomState(seed).choice(
                            len(pseudo_idx), take, replace=n_ps > len(pseudo_idx))
                        Xps = Xpseudo_all[sel]
                        if phys == "stefan":
                            yps = phys_all["p1_stefan"][pseudo_global][sel]
                        elif phys == "kudryavtsev":
                            yps = phys_all["p4_kudryavtsev"][pseudo_global][sel]
                        else:  # placebo: 물리 구조 없는 상수(알래스카 평균)
                            yps = np.full(len(sel), ysrc.mean())
                        Xtr = np.vstack([Xsrc, Xps])
                        ytr = np.concatenate([ysrc, yps])
                    Xtr2, Xte2 = prep(Xtr, Xtest, native)
                    pred = fit_predict(model, Xtr2, ytr, Xte2, seed=seed, epochs=100)["pred"]
                    m = all_metrics(ytest, pred)
                    rows.append(dict(target=target, model=model, phys=phys, r=r, seed=seed,
                                     rmse_cm=m["rmse_cm"], mae_cm=m["mae_cm"], bias_cm=m["bias_cm"],
                                     skill=skill_over_mean(ytest, pred)))
                    if r == 0.0:
                        base_preds[seed] = pred
        # 반응곡선 요약 + 블록부트스트랩(seed0, r별 vs base)
        for phys in PHYS:
            for r in R_GRID:
                if r == 0:
                    continue
                sub = [x for x in rows if x["target"] == target and x["model"] == model
                       and x["phys"] == phys and x["r"] == r]
                if not sub:
                    continue
                # seed 평균 RMSE
                rmse_mean = np.mean([x["rmse_cm"] for x in sub])
                base_rmse = np.mean([x["rmse_cm"] for x in rows if x["target"] == target
                                     and x["model"] == model and x["r"] == 0.0])
                rows.append(dict(target=target, model=model, phys=phys, r=r, seed="SUMMARY",
                                 rmse_cm=rmse_mean, delta_rmse_vs_base=base_rmse - rmse_mean))
        print(f"  {model}: {target} 반응곡선 완료")

res = pd.DataFrame(rows)
res.to_csv(PROC / "s3_aug_curve_results.csv", index=False)
# 헤드라인: 물리 vs placebo 최고 개선
summ = res[res.seed == "SUMMARY"]
best = {}
for phys in PHYS:
    s = summ[summ.phys == phys]
    if len(s):
        best[phys] = float(s.delta_rmse_vs_base.max())
meta = dict(purpose="S3 물리 pseudo-label 증강비율 반응곡선", E_alaska=float(E_alaska),
            r_grid=R_GRID, physics=PHYS, models=MODELS, seeds=SEEDS,
            best_delta_by_phys=best, gpu=GPU,
            note="pseudo/test 공간블록 분리(거리버퍼). placebo=알래스카 평균 상수. Δ>0=개선.")
(PROC / "s3_aug_curve_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"\n[done] 최고 ΔRMSE(개선): {best}. 물리>placebo면 물리 정보 유효.")
