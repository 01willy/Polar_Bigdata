"""지도 산출 모델 게이트 — 격자에서 산출 가능한 공변량만으로 모델을 비교한다.

배경
    보고서 표 4의 지역 내 최상위 모델(다층 퍼셉트론 14.37 cm)은 입력 34종을 쓰며 그중
    SAR 8종은 관측 지점에만 존재해 광역 격자에서 채울 수 없다. 따라서 광역 연속장을
    산출할 모델은 "격자에서 전부 채울 수 있는 공변량"만으로 다시 검증해야 한다.

    격자 산출 가능 집합 = 기후 8종(ERA5-Land) + 토양 9종(SoilGrids) = 17종.
    지형 6종은 Copernicus DEM 30 m 타일이 알래스카 전역에 확보되지 않아 제외한다.

평가
    알래스카 실측 13,606셀, 0.5° 공간블록 6-fold, seed 3회. 표 4와 같은 프로토콜이며
    입력 집합만 다르다. 물리식 단독(1모수·2모수)을 같은 fold에서 함께 적합해 대조한다.

산출
    data/processed/map_model_gate_results.csv
    data/processed/map_model_gate_meta.json
실행
    GPU=9 PYTHONPATH=src python scripts/2_evaluation/map_model_gate.py
"""
import os

GPU = os.environ.get("GPU", "9")
assert GPU in {"6", "7", "8", "9"}, f"GPU는 6,7,8,9만 허용(요청: {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C                                        # noqa: E402
from polar.fidelity import TARGET, add_group_keys, spatial_block_splits  # noqa: E402
from polar.tab_models import available_models, fit_predict, NAN_NATIVE, set_device  # noqa: E402
from polar.preprocessing import fold_prep as prep                    # noqa: E402
from polar.eval_metrics import all_metrics                           # noqa: E402

import torch                                                          # noqa: E402
set_device("cuda:0" if torch.cuda.is_available() else "cpu")

PROC = C.PROCESSED
ALASKA = ["ABoVE_AK", "United States (Alaska)"]
SEEDS = [0, 1, 2]

# 격자에서 전부 채울 수 있는 공변량(지형 제외)
CLIM = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd",
        "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
SOIL = ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15",
        "sg_cfvo_5_15", "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15",
        "sg_soc_15_30"]
GRIDDABLE = CLIM + SOIL

df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
ak = df[df.region.isin(ALASKA)].reset_index(drop=True)
missing = [c for c in GRIDDABLE if c not in ak.columns]
assert not missing, f"공변량 결측: {missing}"

y = ak[TARGET].values.astype(float)
s = ak["e5_sqrt_tdd"].values.astype(float)
X = ak[GRIDDABLE].values.astype(np.float32)
folds = spatial_block_splits(ak, n_splits=6)
print(f"[gate] 알래스카 {len(ak):,}셀 · 격자산출가능 {len(GRIDDABLE)}feature · "
      f"블록 {ak['block'].nunique()}개 · 6 fold")

rows, oof = [], {}


def fit_stefan(y_tr, s_tr, two_param: bool):
    """물리식 계수 적합. two_param=True 면 절편 포함(a + E·s)."""
    m = np.isfinite(y_tr) & np.isfinite(s_tr)
    yt, st = y_tr[m], s_tr[m]
    if not two_param:
        return 0.0, float((st * yt).sum() / (st * st).sum())
    A = np.c_[np.ones_like(st), st]
    coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
    return float(coef[0]), float(coef[1])


# ---------- 물리식 단독(확률 성분 없음, seed 1회) ----------
for two in (False, True):
    pred = np.full(len(ak), np.nan)
    fits = []
    for tr, te in folds:
        a, E = fit_stefan(y[tr], s[tr], two)
        pred[te] = a + E * s[te]
        fits.append({"a": a, "E": E})
    name = "Stefan 2모수" if two else "Stefan 1모수"
    rows.append(dict(model=name, seed=-1, **all_metrics(y, pred)))
    oof[name] = pred
    print(f"  {name:12s} RMSE {rows[-1]['rmse_cm']:.3f}  "
          f"(a {np.mean([f['a'] for f in fits]):+.2f}, E {np.mean([f['E'] for f in fits]):.4f})")

# ---------- 학습 모델 ----------
WANT = ["mlp", "tabm", "catboost", "histgbm"]
models = [m for m in available_models() if m in WANT]
print(f"[gate] 학습 모델: {models}")

for model in models:
    for seed in SEEDS:
        pred = np.full(len(ak), np.nan)
        for tr, te in folds:
            Xtr, Xte = prep(X[tr], X[te], model in NAN_NATIVE)
            pred[te] = fit_predict(model, Xtr, y[tr], Xte, seed=seed)["pred"]
        rows.append(dict(model=model, seed=seed, **all_metrics(y, pred)))
        oof.setdefault(model, []).append(pred)
        print(f"  {model:12s} seed{seed} RMSE {rows[-1]['rmse_cm']:.3f}")

res = pd.DataFrame(rows)
res.to_csv(PROC / "map_model_gate_results.csv", index=False)

summ = (res[res.model.isin(models)].groupby("model")["rmse_cm"]
        .agg(["mean", "std"]).round(3))
phys = res[~res.model.isin(models)].set_index("model")["rmse_cm"].round(3)
print("\n[요약] 격자 산출 가능 17종 · 알래스카 공간블록 6-fold")
print(summ.to_string())
print(phys.to_string())

# 3-seed 앙상블 예측(표 4와 같은 규약)
ens = {m: np.mean(np.vstack(v), axis=0) for m, v in oof.items() if isinstance(v, list)}
ens_rmse = {m: all_metrics(y, p)["rmse_cm"] for m, p in ens.items()}
print("\n[3-seed 앙상블 예측 기준]")
for m, v in sorted(ens_rmse.items(), key=lambda kv: kv[1]):
    print(f"  {m:12s} {v:.3f}")

best = min(ens_rmse, key=ens_rmse.get)
json.dump({
    "featset": GRIDDABLE,
    "n_features": len(GRIDDABLE),
    "n_cells": int(len(ak)),
    "n_blocks": int(ak["block"].nunique()),
    "protocol": "알래스카 0.5도 공간블록 6-fold, seed 3회, 3-seed 앙상블 예측",
    "note": "지형 6종은 알래스카 전역 DEM 미확보로 제외. SAR 8종은 격자 산출 불가로 제외.",
    "physics_only": {k: float(v) for k, v in phys.items()},
    "ensemble_rmse_cm": {k: float(v) for k, v in ens_rmse.items()},
    "per_seed_mean": {k: float(v) for k, v in summ["mean"].items()},
    "best_learned_model": best,
    "best_learned_rmse_cm": float(ens_rmse[best]),
}, open(PROC / "map_model_gate_meta.json", "w"), ensure_ascii=False, indent=2)

pd.DataFrame({"loc_id": ak.loc_id, "lat": ak.lat, "lon": ak.lon,
              "block": ak.block, TARGET: y,
              **{f"pred_{m}": p for m, p in {**{k: v for k, v in oof.items()
                                                if not isinstance(v, list)}, **ens}.items()}}
             ).to_csv(PROC / "map_model_gate_oof.csv", index=False)
print("\n저장: map_model_gate_{results.csv,meta.json,oof.csv}")
