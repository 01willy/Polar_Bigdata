"""SoilGrids 토양 공변량 누설통제 게이트.

overnight_cell_experiments.py 의 CV 재사용:
- 공간블록: block=floor(lat/0.5)*100000+floor(lon/0.5), GroupKFold 6
- LORO: leave-one-region-out, region test>=100
표준지표(polar.eval_metrics.all_metrics). skill_over_mean 병기.

모델: GBM(HistGradientBoostingRegressor), log1p 타깃.
결측처리: train fold 중앙값 대체 + 결측 플래그(sg_ 만 결측 가능; clim/terr 무결측).

피처셋:
  M_clim     = 기후8
  M_climterr = 기후8+지형6 (현 baseline)
  M_soil     = 기후8+지형6+토양(sg_*)
  M_soilonly = 기후8+토양(지형 대신)

게이트: 공간블록+LORO 둘 다에서 M_climterr 대비 개선해야 채택. 개선분·악화분 명시.
물리 주석: 토양수분·유기탄소는 Stefan 의 땅 열특성 E 와 직결 → W3 물리 파라미터 예측 입력 후보.

산출: data/processed/soil_ablation_gate.csv          (피처셋×CV 표준지표)
      data/processed/soil_ablation_gate_loro.csv     (지역별 LORO 상세)
      data/processed/soil_ablation_gate_meta.json    (게이트 판정)
"""
import os, sys, json, time
sys.path.insert(0, os.path.join("/home/willy010313/Polar_Bigdata", "src"))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from polar.eval_metrics import all_metrics

ROOT = "/home/willy010313/Polar_Bigdata"
PROC = os.path.join(ROOT, "data/processed")
CLIP = (np.log1p(1.0), np.log1p(600.0))
t0 = time.time()

CLIM = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
TERR = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]

df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell_v3_soil.csv"), low_memory=False)
SOIL = sorted([c for c in df.columns if c.startswith("sg_")])
print(f"[load] cells={len(df)}  soil cols={len(SOIL)}: {SOIL}")

y_cm = df["alt_cm"].values.astype(float)
ylog = np.log1p(y_cm)
df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int))
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                         l2_regularization=1.0, early_stopping=True, random_state=0)


def oof_spatial(feats, soil_in_set, splits):
    o = np.full(len(df), np.nan)
    Xbase = df[feats].values.astype(float)
    soil_idx = [i for i, f in enumerate(feats) if f in soil_in_set]
    for tr, te in splits:
        Xtr = Xbase[tr].copy(); Xte = Xbase[te].copy()
        # train fold 중앙값 대체(sg_ 만 결측). 플래그는 별도 열로 append.
        if soil_idx:
            med = np.nanmedian(Xtr[:, soil_idx], axis=0)
            for k, ci in enumerate(soil_idx):
                mtr = np.isnan(Xtr[:, ci]); mte = np.isnan(Xte[:, ci])
                Xtr[mtr, ci] = med[k]; Xte[mte, ci] = med[k]
            # 결측 플래그 append
            ftr = np.isnan(Xbase[tr][:, soil_idx]).astype(float)
            fte = np.isnan(Xbase[te][:, soil_idx]).astype(float)
            Xtr = np.hstack([Xtr, ftr]); Xte = np.hstack([Xte, fte])
        m = gbm().fit(Xtr, ylog[tr])
        o[te] = to_cm(m.predict(Xte))
    return o


def run_cv(feats, soil_in_set, cv_name, splits):
    o = oof_spatial(feats, soil_in_set, splits)
    m = all_metrics(y_cm, o)
    m.update({"cv_type": cv_name, "nfeat": len(feats)})
    return m, o


def spatial_splits():
    return list(GroupKFold(n_splits=6).split(df, ylog, groups=df.block.values))


def loro_splits():
    reg = df.region.values; out = []
    for r in pd.unique(reg):
        te = np.where(reg == r)[0]; tr = np.where(reg != r)[0]
        if len(te) >= 100:
            out.append((r, tr, te))
    return out


FEATSETS = {
    "M_clim":     (CLIM, set()),
    "M_climterr": (CLIM + TERR, set()),
    "M_soil":     (CLIM + TERR + SOIL, set(SOIL)),
    "M_soilonly": (CLIM + SOIL, set(SOIL)),
}

SB = spatial_splits()
LO = loro_splits()
print(f"[cv] spatial folds={len(SB)}  LORO regions={[r for r,_,_ in LO]}")

rows = []
oof_store = {}
# --- 공간블록 ---
print("\n=== 공간블록 6-fold ===")
for name, (feats, soilset) in FEATSETS.items():
    t = time.time()
    m, o = run_cv(feats, soilset, "spatial_block", [(tr, te) for tr, te in SB])
    m["config"] = name
    rows.append(m); oof_store[("spatial_block", name)] = o
    print(f"  {name:12s} rmse={m['rmse_cm']:6.2f}  r2={m['r2']:.3f}  skill={m['skill_over_mean']*100:5.1f}%  ({time.time()-t:.0f}s)")

# --- LORO(전체 통합) + 지역별 ---
print("\n=== LORO(leave-one-region-out) ===")
loro_region_rows = []
for name, (feats, soilset) in FEATSETS.items():
    t = time.time()
    o = np.full(len(df), np.nan)
    for r, tr, te in LO:
        oo = oof_spatial(feats, soilset, [(tr, te)])
        o[te] = oo[te]
        mr = all_metrics(y_cm[te], o[te])
        loro_region_rows.append({"config": name, "region": r, "n": len(te),
                                 "rmse_cm": mr["rmse_cm"], "skill_over_mean": mr["skill_over_mean"],
                                 "r2": mr["r2"], "bias_cm": mr["bias_cm"]})
    m = all_metrics(y_cm, o); m.update({"cv_type": "LORO", "nfeat": len(feats), "config": name})
    rows.append(m); oof_store[("LORO", name)] = o
    print(f"  {name:12s} rmse={m['rmse_cm']:6.2f}  r2={m['r2']:.3f}  skill={m['skill_over_mean']*100:5.1f}%  ({time.time()-t:.0f}s)")

res = pd.DataFrame(rows)[["cv_type", "config", "nfeat", "n", "rmse_cm", "mae_cm", "bias_cm",
                          "r2", "target_sd_cm", "skill_over_mean"]]
res.to_csv(os.path.join(PROC, "soil_ablation_gate.csv"), index=False)
pd.DataFrame(loro_region_rows).to_csv(os.path.join(PROC, "soil_ablation_gate_loro.csv"), index=False)

# --- 게이트 판정: M_soil vs M_climterr, 두 CV 모두 개선? ---
def get(cv, cfg, col):
    row = res[(res.cv_type == cv) & (res.config == cfg)]
    return float(row[col].iloc[0])

verdict = {}
for cv in ["spatial_block", "LORO"]:
    base_r = get(cv, "M_climterr", "rmse_cm"); base_s = get(cv, "M_climterr", "skill_over_mean")
    soil_r = get(cv, "M_soil", "rmse_cm"); soil_s = get(cv, "M_soil", "skill_over_mean")
    verdict[cv] = {
        "climterr_rmse": round(base_r, 2), "soil_rmse": round(soil_r, 2),
        "d_rmse": round(soil_r - base_r, 2),        # 음수=개선
        "climterr_skill": round(base_s, 4), "soil_skill": round(soil_s, 4),
        "d_skill": round(soil_s - base_s, 4),       # 양수=개선
        "improved": bool(soil_r < base_r and soil_s > base_s),
    }

# LORO 레나 전이 개별(핵심): 토양이 전지구커버라 결측 라우팅 없음 → 순수 정보 검정
lena_rows = pd.DataFrame(loro_region_rows)
lena = lena_rows[lena_rows.region == "Lena_RU"] if "Lena_RU" in lena_rows.region.values else None
lena_delta = None
if lena is not None:
    lc = lena[lena.config == "M_climterr"]["rmse_cm"]; ls = lena[lena.config == "M_soil"]["rmse_cm"]
    lena_delta = {"lena_climterr_rmse": round(float(lc.iloc[0]), 2),
                  "lena_soil_rmse": round(float(ls.iloc[0]), 2),
                  "lena_d_rmse": round(float(ls.iloc[0]) - float(lc.iloc[0]), 2)}

adopt = verdict["spatial_block"]["improved"] and verdict["LORO"]["improved"]
meta = {
    "gate": "adopt" if adopt else "reject",
    "rule": "공간블록+LORO 둘 다 M_climterr 대비 rmse 감소 && skill 증가 시에만 채택",
    "verdict": verdict,
    "lena_transfer": lena_delta,
    "n_soil_features": len(SOIL),
    "soil_features": SOIL,
    "elapsed_s": round(time.time() - t0, 1),
}
with open(os.path.join(PROC, "soil_ablation_gate_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n=== 게이트 결과 ===")
print(res.to_string(index=False))
print("\n판정:", meta["gate"])
print(json.dumps(verdict, ensure_ascii=False, indent=2))
if lena_delta:
    print("레나 전이:", json.dumps(lena_delta, ensure_ascii=False))
print(f"\n[done] {time.time()-t0:.0f}s")
