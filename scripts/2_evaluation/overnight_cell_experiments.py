"""Overnight — 정직한 cell-level 기준선 위 다중모달 ablation + AOA + conformal UQ.

GPT 계획(2026-07-09) P2/P3 반영. 기준선 = dl_dataset_cell.csv (loc당 1행, 셀평균 ALT target).
- 모달리티 그룹은 컬럼 존재 기반으로 동적 구성 → 확충 데이터(soilgrids sg_*, cci cci_*, 식생 veg_*)가
  셀 CSV에 추가되면 자동으로 ablation에 편입.
- 평가: 공간블록(0.5° GroupKFold) + LORO(leave-one-region-out). 표준지표(eval_metrics).
- AOA(Meyer 2021 DI) + Conformal(CQR, Romano 2019)로 적용범위·보정 90% 구간.

실행: python3 scripts/2_evaluation/overnight_cell_experiments.py
출력 CSV: alt_ablation_cell_results / _delta / alt_conformal_cell_results /
          alt_aoa_cell_transfer / alt_cell_best_oof
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from polar.eval_metrics import all_metrics, coverage, interval_width

PROC = "data/processed"
CLIP = (np.log1p(1.0), np.log1p(600.0))
ALPHA = 0.10
RNG = np.random.default_rng(0)
t_start = time.time()

df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell.csv"))
print(f"[load] cell rows={len(df)}  cols={df.shape[1]}")

# ---------- 모달리티 그룹(컬럼 존재 기반 동적) ----------
def present(cols):
    return [c for c in cols if c in df.columns]

GROUPS = {
    "TERRAIN": present(["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]),
    "CLIMATE": present(["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]),
    "INSAR":   present(["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]),
    "POLSAR":  present(["polsar_alt", "polsar_std", "polsar_valid"]),
    "SOIL":    present([c for c in df.columns if c.startswith("sg_") or c.startswith("soil_")]),
    "VEG":     present([c for c in df.columns if c.startswith("veg_") or c.startswith("ndvi") or c.startswith("lc_")]),
    "CCI":     present([c for c in df.columns if c.startswith("cci_")]),
    "LOC":     present(["lat", "lon"]),
}
GROUPS = {k: v for k, v in GROUPS.items() if v}
print("[groups]", {k: len(v) for k, v in GROUPS.items()})

def cfg(*names):
    feats = []
    for n in names:
        feats += GROUPS.get(n, [])
    return feats

CONFIGS = [("M0 지역평균", None), ("M1 기후", cfg("CLIMATE")), ("M2 지형", cfg("TERRAIN")),
           ("M3 기후+지형", cfg("CLIMATE", "TERRAIN"))]
if "INSAR" in GROUPS:  CONFIGS.append(("M4 +InSAR", cfg("CLIMATE", "TERRAIN", "INSAR")))
if "POLSAR" in GROUPS: CONFIGS.append(("M5 +PolSAR", cfg("CLIMATE", "TERRAIN", "POLSAR")))
if "SOIL" in GROUPS:   CONFIGS.append(("M6 +토양(SoilGrids)", cfg("CLIMATE", "TERRAIN", "SOIL")))
if "VEG" in GROUPS:    CONFIGS.append(("M7 +식생", cfg("CLIMATE", "TERRAIN", "VEG")))
if "CCI" in GROUPS:    CONFIGS.append(("M8 +CCI prior", cfg("CLIMATE", "TERRAIN", "CCI")))
ALLGRP = [g for g in ["CLIMATE", "TERRAIN", "INSAR", "POLSAR", "SOIL", "VEG", "CCI"] if g in GROUPS]
CONFIGS.append(("M9 전체", cfg(*ALLGRP)))
CONFIGS.append(("Mloc 위치만(대조)", cfg("LOC")))

y_cm = df["alt_cm"].values.astype(float)
ylog = np.log1p(y_cm)
df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int))
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))

def gbm(q=None):
    if q is None:
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                             l2_regularization=1.0, early_stopping=True, random_state=0)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=63, l2_regularization=1.0, random_state=0)

def oof(feats, splits):
    o = np.full(len(df), np.nan)
    if feats is None:
        for tr, te in splits:
            reg = df.iloc[tr].groupby("region")["alt_cm"].mean(); gm = df.iloc[tr]["alt_cm"].mean()
            o[te] = df.iloc[te]["region"].map(reg).fillna(gm).values
        return o
    X = df[feats].values
    for tr, te in splits:
        o[te] = to_cm(gbm().fit(X[tr], ylog[tr]).predict(X[te]))
    return o

def spatial_splits():
    return list(GroupKFold(n_splits=6).split(df, ylog, groups=df.block.values))

def loro_splits():
    reg = df.region.values; out = []
    for r in pd.unique(reg):
        te = np.where(reg == r)[0]; tr = np.where(reg != r)[0]
        if len(te) >= 100:
            out.append((tr, te))
    return out

# ================= Part 1: cell-level ablation =================
print("\n=== Part 1: cell-level 다중모달 ablation ===")
rows = []; best_oof = None; best_rmse = 1e9
SB = spatial_splits(); LO = loro_splits()
for cv_name, splits in [("spatial_block", SB), ("LORO", LO)]:
    print(f"-- {cv_name} ({len(splits)} folds)")
    for name, feats in CONFIGS:
        t0 = time.time(); o = oof(feats, splits); m = all_metrics(y_cm, o)
        m.update({"config": name, "cv_type": cv_name, "nfeat": 0 if feats is None else len(feats)})
        rows.append(m)
        if cv_name == "spatial_block" and feats is not None and m["rmse_cm"] < best_rmse \
           and not name.startswith("Mloc"):
            best_rmse = m["rmse_cm"]; best_oof = o.copy(); best_name = name; best_feats = feats
        print(f"   {name:20s} rmse={m['rmse_cm']:6.2f}  r2={m['r2']:.3f}  skill={m['skill_over_mean']*100:5.1f}%  ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows)[["cv_type", "config", "nfeat", "n", "rmse_cm", "mae_cm", "bias_cm",
                         "r2", "target_sd_cm", "skill_over_mean"]]
res.to_csv(os.path.join(PROC, "alt_ablation_cell_results.csv"), index=False)
print("저장: alt_ablation_cell_results.csv  | best(spatial) =", best_name, f"{best_rmse:.2f}cm")

# per-group Δskill (M3 대비 각 모달리티 추가 효과, 두 CV)
delta_rows = []
for cv_name in ["spatial_block", "LORO"]:
    sub = res[res.cv_type == cv_name].set_index("config")
    base3 = sub.loc["M3 기후+지형"] if "M3 기후+지형" in sub.index else None
    for name, _ in CONFIGS:
        if name in sub.index and base3 is not None:
            r = sub.loc[name]
            delta_rows.append({"cv_type": cv_name, "config": name,
                               "rmse_cm": r.rmse_cm, "skill": r.skill_over_mean,
                               "d_rmse_vs_M3": round(r.rmse_cm - base3.rmse_cm, 3),
                               "d_skill_vs_M3": round(r.skill_over_mean - base3.skill_over_mean, 4)})
pd.DataFrame(delta_rows).to_csv(os.path.join(PROC, "alt_ablation_cell_delta.csv"), index=False)
print("저장: alt_ablation_cell_delta.csv")

# ================= Part 2: within-domain conformal (CQR) =================
print("\n=== Part 2: within-domain 보정 UQ (CQR) ===")
X = df[best_feats].values
oof_pt = np.full(len(df), np.nan); oof_lo = np.full(len(df), np.nan); oof_hi = np.full(len(df), np.nan)
for tr, te in SB:
    oof_pt[te] = to_cm(gbm().fit(X[tr], ylog[tr]).predict(X[te]))
    oof_lo[te] = to_cm(gbm(ALPHA / 2).fit(X[tr], ylog[tr]).predict(X[te]))
    oof_hi[te] = to_cm(gbm(1 - ALPHA / 2).fit(X[tr], ylog[tr]).predict(X[te]))
blocks = df.block.values; ub = pd.unique(blocks).copy(); RNG.shuffle(ub)
cal_b = set(ub[: len(ub) // 2]); calm = np.array([b in cal_b for b in blocks])
E = np.maximum(oof_lo - y_cm, y_cm - oof_hi)
n_cal = int(calm.sum()); qlevel = min(1.0, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
Q = np.quantile(E[calm], qlevel); lo_c = oof_lo - Q; hi_c = oof_hi + Q; tst = ~calm
raw_cov = coverage(y_cm[tst], oof_lo[tst], oof_hi[tst]); cqr_cov = coverage(y_cm[tst], lo_c[tst], hi_c[tst])
raw_w = interval_width(oof_lo[tst], oof_hi[tst]); cqr_w = interval_width(lo_c[tst], hi_c[tst])
print(f"   raw quantile-GBM  coverage={raw_cov*100:.1f}%  width={raw_w:.1f}cm")
print(f"   CQR 보정          coverage={cqr_cov*100:.1f}%  width={cqr_w:.1f}cm")
pd.DataFrame([
    {"setting": "raw quantile-GBM", "coverage_pct": round(raw_cov*100, 1), "width_cm": round(raw_w, 1)},
    {"setting": "CQR 보정", "coverage_pct": round(cqr_cov*100, 1), "width_cm": round(cqr_w, 1)},
]).to_csv(os.path.join(PROC, "alt_conformal_cell_results.csv"), index=False)

# ================= Part 3: transfer AOA (LORO, DI decile + inside/outside) =================
print("\n=== Part 3: transfer AOA (LORO) ===")
mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
Xs = np.nan_to_num((X - mu) / sd)
reg = df.region.values
di_all = np.full(len(df), np.nan); aoa_all = np.full(len(df), np.nan)
predT = np.full(len(df), np.nan); loT = np.full(len(df), np.nan); hiT = np.full(len(df), np.nan)
inout_rows = []
for r in pd.unique(reg):
    te = np.where(reg == r)[0]; tr = np.where(reg != r)[0]
    if len(te) < 100:
        continue
    nn = NearestNeighbors(n_neighbors=2).fit(Xs[tr])
    d_tr, _ = nn.kneighbors(Xs[tr]); d_bar = d_tr[:, 1].mean()
    q75, q25 = np.quantile(d_tr[:, 1] / d_bar, [0.75, 0.25]); thr = q75 + 1.5 * (q75 - q25)
    d_te, _ = nn.kneighbors(Xs[te], n_neighbors=1); di_te = d_te[:, 0] / d_bar
    inside = di_te <= thr
    di_all[te] = di_te; aoa_all[te] = inside.astype(float)
    pt = to_cm(gbm().fit(X[tr], ylog[tr]).predict(X[te]))
    lo = to_cm(gbm(ALPHA/2).fit(X[tr], ylog[tr]).predict(X[te]))
    hi = to_cm(gbm(1-ALPHA/2).fit(X[tr], ylog[tr]).predict(X[te]))
    predT[te] = pt; loT[te] = lo; hiT[te] = hi
    if inside.sum() > 20 and (~inside).sum() > 20:
        inout_rows.append({"region": r, "n": len(te), "pct_inside": round(100*inside.mean(), 1),
                           "rmse_in": all_metrics(y_cm[te][inside], pt[inside])["rmse_cm"],
                           "rmse_out": all_metrics(y_cm[te][~inside], pt[~inside])["rmse_cm"],
                           "cov_in": round(100*coverage(y_cm[te][inside], lo[inside], hi[inside]), 1),
                           "cov_out": round(100*coverage(y_cm[te][~inside], lo[~inside], hi[~inside]), 1)})
# DI-decile 그래디언트(전 LORO test 통합)
mask = np.isfinite(di_all) & np.isfinite(predT)
di = di_all[mask]; yy = y_cm[mask]; pp = predT[mask]; ll = loT[mask]; hh = hiT[mask]
dec = pd.qcut(di, 10, labels=False, duplicates="drop")
dec_rows = []
for d in np.unique(dec):
    s = dec == d
    dec_rows.append({"di_decile": int(d)+1, "di_mean": round(float(np.mean(di[s])), 2), "n": int(s.sum()),
                     "rmse_cm": all_metrics(yy[s], pp[s])["rmse_cm"],
                     "coverage_pct": round(100*coverage(yy[s], ll[s], hh[s]), 1)})
dec_df = pd.DataFrame(dec_rows)
inout = pd.DataFrame(inout_rows)
dec_df.to_csv(os.path.join(PROC, "alt_aoa_cell_transfer.csv"), index=False)
if len(inout):
    inout.to_csv(os.path.join(PROC, "alt_aoa_cell_inout.csv"), index=False)
    print(f"   AOA안 RMSE {inout.rmse_in.mean():.1f} < 밖 {inout.rmse_out.mean():.1f}cm | "
          f"cov 안 {inout.cov_in.mean():.0f}% > 밖 {inout.cov_out.mean():.0f}%")
print(dec_df.to_string(index=False))

# best OOF 저장(그림용)
pd.DataFrame({"lat": df.lat, "lon": df.lon, "region": df.region, "alt_cm": y_cm, "alt_sd": df.get("alt_sd", np.nan),
              "pred": oof_pt, "lo_raw": oof_lo, "hi_raw": oof_hi, "lo_cqr": lo_c, "hi_cqr": hi_c,
              "di": di_all, "aoa_inside": aoa_all, "pred_loro": predT}).to_csv(
    os.path.join(PROC, "alt_cell_best_oof.csv"), index=False)

meta = {"best_config": best_name, "best_feats": best_feats, "n_cells": len(df),
        "groups": {k: len(v) for k, v in GROUPS.items()},
        "raw_cov": round(raw_cov*100, 1), "cqr_cov": round(cqr_cov*100, 1),
        "cqr_width": round(cqr_w, 1), "elapsed_s": round(time.time()-t_start, 1)}
with open(os.path.join(PROC, "overnight_cell_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("\n[done]", json.dumps(meta, ensure_ascii=False), f"  ({time.time()-t_start:.0f}s)")
