"""S1: 실측-only baseline — 여러 모델 병렬 비교 (한 모델로 단정 금지).

`docs/RESEARCH_PLAN_multifidelity_2026-07-22.md` S1. 모델군: GBM3(LightGBM·XGBoost·CatBoost)
+ HistGBM + RealMLP + torch(MLP·FT-T·TabM) + TabPFN(설치 시). 더 나은 모델은 자동 채택.
평가 2축: (1) 알래스카 in-domain 공간블록 6-fold, FULL 공변량34. (2) LORO 전이, 공유 코어25(SAR 제외).
누설통제: fold-safe median 대체(torch/realmlp/tabpfn), GBM은 NaN native. tests/test_leakage.py 통과 전제.

산출: data/processed/s1_baseline_results.csv (모델×cv×seed metrics)
      data/processed/s1_baseline_oof.csv         (알래스카 OOF 예측; 앙상블·시각화용)
      data/processed/s1_baseline_meta.json
실행: python scripts/3_deep_learning/s1_baseline_tournament.py   (SMOKE=1 로 빠른 점검)
"""
import os
# ★ GPU 고정은 어떤 import(특히 torch)보다 먼저. 6,7,8,9만 허용.
GPU = os.environ.get("GPU", "6")
assert GPU in {"6", "7", "8", "9"}, f"GPU는 6,7,8,9만 허용(요청: {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (COVARIATE_CORE, SHARED_CORE, TARGET, add_group_keys,
                            spatial_block_splits, loro_splits)
from polar.tab_models import available_models, fit_predict, NAN_NATIVE, set_device
from polar.preprocessing import fold_prep as prep
from polar.eval_metrics import all_metrics

SMOKE = os.environ.get("SMOKE", "0") == "1"
import torch
set_device("cuda:0" if torch.cuda.is_available() else "cpu")
# 안전검증: CUDA_VISIBLE_DEVICES 매핑으로 논리 0이 물리 GPU {GPU}인지 UUID 대조
if torch.cuda.is_available():
    used_uuid = torch.cuda.get_device_properties(0).uuid
    print(f"[GPU guard] CUDA_VISIBLE_DEVICES={GPU} → 논리0 = 물리 GPU (uuid …{str(used_uuid)[-8:]})")
PROC = C.PROCESSED
ALASKA = ["ABoVE_AK", "United States (Alaska)"]
SEEDS = [0] if SMOKE else [0, 1, 2]
EPOCHS = 20 if SMOKE else 120


df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
models = available_models()
print(f"[S1] device GPU={GPU} · models={models} · SMOKE={SMOKE}")

rows, oof_store = [], {}

# ============ Part 1: 알래스카 in-domain 공간블록 6-fold (FULL 34) ============
ak = df[df.region.isin(ALASKA)].reset_index(drop=True)
if SMOKE:
    ak = ak.sample(n=min(3000, len(ak)), random_state=0).reset_index(drop=True)
Xak = ak[COVARIATE_CORE].values.astype(np.float32)
yak = ak[TARGET].values.astype(float)
folds = spatial_block_splits(ak, n_splits=6)
print(f"[Part1] 알래스카 in-domain {len(ak):,}셀 · FULL {len(COVARIATE_CORE)}feature · 6 spatial blocks")

usable, skipped = [], []
for model in models:
    t0 = time.time()
    try:
        oof_seeds = []
        for seed in SEEDS:
            oof = np.full(len(ak), np.nan)
            for tr, te in folds:
                Xtr, Xte = prep(Xak[tr], Xak[te], model in NAN_NATIVE)
                oof[te] = fit_predict(model, Xtr, yak[tr], Xte, seed=seed, epochs=EPOCHS)["pred"]
            oof_seeds.append(oof)
            m = all_metrics(yak, oof)
            rows.append(dict(cv="spatial_block_AK", model=model, seed=seed, **m))
    except Exception as e:
        print(f"  {model:10s} SKIP: {str(e)[:80]}")
        skipped.append(model)
        continue
    usable.append(model)
    oof_store[model] = np.mean(oof_seeds, 0)  # seed 평균 (앙상블·시각화용)
    mm = all_metrics(yak, oof_store[model])
    print(f"  {model:10s} RMSE {mm['rmse_cm']:.2f} MAE {mm['mae_cm']:.2f} bias {mm['bias_cm']:+.2f} "
          f"R2 {mm['r2']:.3f} | {time.time()-t0:.0f}s")
models = usable  # 이후 단계는 성공한 모델만
if skipped:
    print(f"[skip] 사용 불가(라이선스/의존성): {skipped}")

# ============ Part 2: LORO 전이 (공유 코어25, SAR 제외) ============
print(f"[Part2] LORO 전이 · SHARED {len(SHARED_CORE)}feature (SAR 제외)")
dfx = df if not SMOKE else df.groupby("region", group_keys=False).apply(
    lambda g: g.sample(n=min(500, len(g)), random_state=0)).reset_index(drop=True)
Xall = dfx[SHARED_CORE].values.astype(np.float32)
yall = dfx[TARGET].values.astype(float)
for model in models:
    for seed in SEEDS if model not in NAN_NATIVE else [0]:
        for r, tr, te in loro_splits(dfx, min_test=100):
            Xtr, Xte = prep(Xall[tr], Xall[te], model in NAN_NATIVE)
            pred = fit_predict(model, Xtr, yall[tr], Xte, seed=seed, epochs=EPOCHS)["pred"]
            m = all_metrics(yall[te], pred)
            rows.append(dict(cv="LORO", model=model, seed=seed, region=r, **m))
    # 요약 출력(seed0)
    sub = [x for x in rows if x["cv"] == "LORO" and x["model"] == model and x.get("seed") == (SEEDS[0] if model not in NAN_NATIVE else 0)]
    if sub:
        print(f"  {model:10s} LORO: " + " ".join(f"{x['region'][:8]}={x['rmse_cm']:.1f}" for x in sub))

# ============ 저장 ============
res = pd.DataFrame(rows)
res.to_csv(PROC / "s1_baseline_results.csv", index=False)
oof_df = ak[["loc_id", "lat", "lon", "region", "block", TARGET]].copy()
for model, o in oof_store.items():
    oof_df[f"pred_{model}"] = o
oof_df.to_csv(PROC / "s1_baseline_oof.csv", index=False)

# in-domain 요약(seed 평균 기준)
summ = (res[res.cv == "spatial_block_AK"].groupby("model")
        .agg(rmse=("rmse_cm", "mean"), mae=("mae_cm", "mean"), r2=("r2", "mean"))
        .sort_values("rmse").round(3))
print("\n=== 알래스카 in-domain 순위(공간블록, seed평균 metric) ===")
print(summ.to_string())

meta = dict(purpose="S1 실측-only baseline 다모델", models=models, seeds=SEEDS,
            n_alaska=int(len(ak)), full_features=COVARIATE_CORE, shared_features=SHARED_CORE,
            smoke=SMOKE, gpu=GPU,
            best_indomain=summ.index[0], best_rmse=float(summ.rmse.iloc[0]))
(PROC / "s1_baseline_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"\n[done] 저장. in-domain 최선: {summ.index[0]} ({summ.rmse.iloc[0]:.2f}cm). "
      f"단정 금지 — 전 모델 CI 비교는 S11에서.")
