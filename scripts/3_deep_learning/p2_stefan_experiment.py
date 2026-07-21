"""트랙 β — Stefan 물리 잔차학습 견고성 검증.

목표: Stefan 근사(ALT = E·sqrt(TDD), E=edaphic factor)를 물리 base로 두고 ML이 잔차를
보정하는 방식(RESIDUAL)이 순수 ML(ML_only)·물리단독(PHYS_only) 대비 공간블록·특히
LORO 전이에서 견고한 개선을 주는지 검증한다.

방법(모두 공간블록 6-fold GroupKFold + LORO region, test>=100):
  - PHYS_only : train fold에서 E 최소제곱 적합(절편 포함/미포함 둘 다 시도, train RMSE
                낮은 쪽 채택). test에 Stefan 적용. 누설 방지 위해 fold마다 재적합.
  - ML_only   : GBM으로 log1p(ALT) 직접 예측. P1과 동일 baseline. 전 공변량.
  - RESIDUAL  : r = ALT - Stefan(E·sqrt(TDD))(원 cm 잔차)를 전 공변량 GBM으로 학습.
                예측 = Stefan + GBM(잔차). E는 train fold에서만 적합.
  - PHYS_strat: (개선 물리 슬롯) E를 전역 상수 대신 토양수분(e5_swe)·기후로 층화한 선형
                회귀로 모델링. E(x) = g0 + g1·swe + g2·maat 형태를 train fold에서 적합,
                Stefan_strat = E(x)·sqrt(TDD).

전 공변량: 지형6(dem_*)·기후8(e5_*)·InSAR5(insar_*)·PolSAR3(polsar_*)·CCI2(cci_*)+insar_miss.
결측 공변량은 train fold 중앙값 대체+플래그(NaN 네이티브 라우팅 아티팩트 회피, P1 교훈).

타깃 log1p(ALT) 후 expm1+clip[log1p(1),log1p(600)] (ML_only). 잔차는 원 cm.

출력:
  data/processed/p2b_stefan_results.csv  (method,cv_type,n,rmse_cm,mae_cm,bias_cm,r2,skill_over_mean)
  data/processed/p2b_stefan_oof.csv      (loc_id,lat,lon,region,alt_cm + 세 방식 OOF, 두 CV)
  data/processed/p2b_stefan_meta.json    (적합 E값 등 메타)

실행: python3 scripts/3_deep_learning/p2_stefan_experiment.py
GPU 불필요(sklearn CPU).
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from polar.eval_metrics import all_metrics

PROC = "data/processed"
CLIP = (np.log1p(1.0), np.log1p(600.0))
t_start = time.time()

# ---------- 데이터 ----------
df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell_v2.csv"), low_memory=False)
print(f"[load] cell rows={len(df)}  cols={df.shape[1]}")

# 전 공변량(명시 투입). insar_miss 플래그 포함.
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
INSAR = ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]
POLSAR = ["polsar_alt", "polsar_std", "polsar_valid"]
CCI = ["cci_alt", "cci_valid"]
FLAGS = ["insar_miss"]
FEATS = [c for c in (TERRAIN + CLIMATE + INSAR + POLSAR + CCI + FLAGS) if c in df.columns]
missing_cols = [c for c in (TERRAIN + CLIMATE + INSAR + POLSAR + CCI + FLAGS) if c not in df.columns]
print(f"[feats] n={len(FEATS)}  누락컬럼={missing_cols if missing_cols else '없음'}")

# 숫자 강제(mixed-type 컬럼 방어)
for c in FEATS + ["alt_cm", "e5_tdd", "e5_swe", "e5_maat", "lat", "lon"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

y_cm = df["alt_cm"].values.astype(float)
ylog = np.log1p(y_cm)
tdd = df["e5_tdd"].values.astype(float)
sqrt_tdd = np.sqrt(np.clip(tdd, 0, None))
loc_id = df["loc_id"].values if "loc_id" in df.columns else np.arange(len(df))

# 공간블록/LORO
df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int))
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))


def gbm():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, random_state=0)


def spatial_splits():
    return list(GroupKFold(n_splits=6).split(df, ylog, groups=df.block.values))


def loro_splits():
    reg = df.region.values
    out = []
    for r in pd.unique(reg):
        te = np.where(reg == r)[0]
        tr = np.where(reg != r)[0]
        if len(te) >= 100:
            out.append((r, tr, te))
    return out


# ---------- 결측 대체(train fold 중앙값)+플래그 ----------
def impute_train(X_tr, X_te, feats):
    """train fold 중앙값으로 대체. 원 결측 위치 플래그 컬럼 추가."""
    Xtr = X_tr.copy()
    Xte = X_te.copy()
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)  # 전부 결측인 컬럼 방어
    flags_tr = []
    flags_te = []
    for j in range(Xtr.shape[1]):
        m_tr = ~np.isfinite(Xtr[:, j])
        m_te = ~np.isfinite(Xte[:, j])
        if m_tr.any() or m_te.any():
            Xtr[m_tr, j] = med[j]
            Xte[m_te, j] = med[j]
            flags_tr.append(m_tr.astype(float))
            flags_te.append(m_te.astype(float))
    if flags_tr:
        Xtr = np.c_[Xtr, np.array(flags_tr).T]
        Xte = np.c_[Xte, np.array(flags_te).T]
    return Xtr, Xte


# ---------- Stefan E 적합(train fold) ----------
def fit_E(y, s):
    """train fold에서 E 적합. 절편 포함/미포함 둘 다 시도, train RMSE 낮은 쪽 채택.
    반환: dict(mode, E, a, train_rmse)."""
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    # 절편 미포함(through origin): E = sum(s*y)/sum(s*s)
    denom = float((s * s).sum())
    E0 = float((s * y).sum() / denom) if denom > 0 else 0.0
    rmse0 = float(np.sqrt(np.mean((y - E0 * s) ** 2)))
    # 절편 포함: y = a + E*s
    A = np.c_[np.ones_like(s), s]
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a1, E1 = float(coef[0]), float(coef[1])
    rmse1 = float(np.sqrt(np.mean((y - (a1 + E1 * s)) ** 2)))
    if rmse1 < rmse0:
        return {"mode": "intercept", "E": E1, "a": a1, "train_rmse": rmse1}
    return {"mode": "origin", "E": E0, "a": 0.0, "train_rmse": rmse0}


def stefan_pred(fit, s):
    return fit["a"] + fit["E"] * s


# ---------- PHYS_strat: E를 swe·maat로 층화한 선형회귀 ----------
def fit_E_strat(y, s, swe, maat):
    """ALT = E(x)·sqrt(TDD), E(x) = g0 + g1·swe + g2·maat.
    => ALT = g0·s + g1·(swe·s) + g2·(maat·s). 절편 없는 선형회귀(물리적으로 s→0이면 ALT→0).
    train 중앙값으로 swe·maat 결측 대체. 반환 계수 dict."""
    swe = swe.copy()
    maat = maat.copy()
    swe_med = np.nanmedian(swe[np.isfinite(y) & np.isfinite(s)])
    maat_med = np.nanmedian(maat[np.isfinite(y) & np.isfinite(s)])
    swe = np.where(np.isfinite(swe), swe, swe_med)
    maat = np.where(np.isfinite(maat), maat, maat_med)
    m = np.isfinite(y) & np.isfinite(s)
    ym, sm, swem, maatm = y[m], s[m], swe[m], maat[m]
    # 설계행렬: [s, swe*s, maat*s]
    A = np.c_[sm, swem * sm, maatm * sm]
    coef, _, _, _ = np.linalg.lstsq(A, ym, rcond=None)
    g0, g1, g2 = map(float, coef)
    return {"g0": g0, "g1": g1, "g2": g2, "swe_med": float(swe_med), "maat_med": float(maat_med)}


def strat_pred(fit, s, swe, maat):
    swe = np.where(np.isfinite(swe), swe, fit["swe_med"])
    maat = np.where(np.isfinite(maat), maat, fit["maat_med"])
    E = fit["g0"] + fit["g1"] * swe + fit["g2"] * maat
    return E * s


# ---------- OOF 실행 ----------
swe = df["e5_swe"].values.astype(float) if "e5_swe" in df.columns else np.full(len(df), np.nan)
maat = df["e5_maat"].values.astype(float) if "e5_maat" in df.columns else np.full(len(df), np.nan)
Xall = df[FEATS].values.astype(float)

CVS = [("spatial_block", [(None, tr, te) for tr, te in spatial_splits()]),
       ("LORO", loro_splits())]

# OOF 배열(CV별 분리 저장)
oof = {cv: {m: np.full(len(df), np.nan) for m in ["PHYS_only", "ML_only", "RESIDUAL", "PHYS_strat"]}
       for cv, _ in CVS}
E_fits = {cv: [] for cv, _ in CVS}

for cv_name, splits in CVS:
    print(f"\n-- {cv_name} ({len(splits)} folds)")
    for fold_i, (rlabel, tr, te) in enumerate(splits):
        t0 = time.time()
        # sqrt(TDD) 결측(v2에 1행 e5_tdd NaN)은 train fold 중앙값으로 대체(물리 base NaN 회피).
        s_med = np.nanmedian(sqrt_tdd[tr])
        s_tr = np.where(np.isfinite(sqrt_tdd[tr]), sqrt_tdd[tr], s_med)
        s_te = np.where(np.isfinite(sqrt_tdd[te]), sqrt_tdd[te], s_med)
        # --- PHYS_only ---
        pf = fit_E(y_cm[tr], s_tr)
        oof[cv_name]["PHYS_only"][te] = np.clip(stefan_pred(pf, s_te), 1.0, 600.0)
        # --- PHYS_strat ---
        sf = fit_E_strat(y_cm[tr], s_tr, swe[tr], maat[tr])
        oof[cv_name]["PHYS_strat"][te] = np.clip(strat_pred(sf, s_te, swe[te], maat[te]), 1.0, 600.0)
        # --- ML_only (log1p 직접) ---
        Xtr, Xte = impute_train(Xall[tr], Xall[te], FEATS)
        oof[cv_name]["ML_only"][te] = to_cm(gbm().fit(Xtr, ylog[tr]).predict(Xte))
        # --- RESIDUAL (원 cm 잔차 = ALT - Stefan) ---
        base_tr = stefan_pred(pf, s_tr)
        base_te = stefan_pred(pf, s_te)
        r_tr = y_cm[tr] - base_tr  # 원 cm 잔차
        resid_pred = gbm().fit(Xtr, r_tr).predict(Xte)
        oof[cv_name]["RESIDUAL"][te] = np.clip(base_te + resid_pred, 1.0, 600.0)
        E_fits[cv_name].append({"fold": fold_i, "region": rlabel,
                                "E_mode": pf["mode"], "E": round(pf["E"], 4),
                                "a": round(pf["a"], 4), "train_rmse": round(pf["train_rmse"], 3),
                                "strat_g0": round(sf["g0"], 4), "strat_g1_swe": round(sf["g1"], 4),
                                "strat_g2_maat": round(sf["g2"], 4),
                                "n_tr": int(len(tr)), "n_te": int(len(te))})
        lbl = rlabel if rlabel else f"fold{fold_i}"
        print(f"   {lbl:16s} E={pf['E']:.3f}({pf['mode']:9s}) resid_std={np.std(r_tr):6.2f}  ({time.time()-t0:.0f}s)")

# ---------- 지표 집계 ----------
rows = []
for cv_name, _ in CVS:
    for method in ["PHYS_only", "ML_only", "RESIDUAL", "PHYS_strat"]:
        o = oof[cv_name][method]
        m = all_metrics(y_cm, o)
        rows.append({"method": method, "cv_type": cv_name, "n": m["n"],
                     "rmse_cm": m["rmse_cm"], "mae_cm": m["mae_cm"], "bias_cm": m["bias_cm"],
                     "r2": m["r2"], "skill_over_mean": m["skill_over_mean"]})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(PROC, "p2b_stefan_results.csv"), index=False)
print("\n=== 결과 ===")
print(res.to_string(index=False))

# ---------- OOF 저장 ----------
oof_df = pd.DataFrame({"loc_id": loc_id, "lat": df.lat.values, "lon": df.lon.values,
                       "region": df.region.values, "alt_cm": y_cm})
for cv_name, _ in CVS:
    for method in ["PHYS_only", "ML_only", "RESIDUAL", "PHYS_strat"]:
        oof_df[f"{cv_name}__{method}"] = oof[cv_name][method]
oof_df.to_csv(os.path.join(PROC, "p2b_stefan_oof.csv"), index=False)
print("저장: p2b_stefan_oof.csv")

# ---------- 게이트 판정 ----------
def get(cv, meth, col="rmse_cm"):
    return float(res[(res.cv_type == cv) & (res.method == meth)][col].values[0])

gate = {}
for cv in ["spatial_block", "LORO"]:
    r_res = get(cv, "RESIDUAL")
    r_ml = get(cv, "ML_only")
    r_phys = get(cv, "PHYS_only")
    improves = (r_res < r_ml) and (r_res < r_phys)
    gate[cv] = {"rmse_RESIDUAL": r_res, "rmse_ML_only": r_ml, "rmse_PHYS_only": r_phys,
                "d_vs_ML": round(r_res - r_ml, 3), "d_vs_PHYS": round(r_res - r_phys, 3),
                "RESIDUAL_improves_both": bool(improves)}

verdict = "ADOPT" if gate["LORO"]["RESIDUAL_improves_both"] and gate["spatial_block"]["RESIDUAL_improves_both"] \
    else ("PARTIAL" if gate["LORO"]["RESIDUAL_improves_both"] or gate["spatial_block"]["RESIDUAL_improves_both"]
          else "REJECT")

meta = {"n_cells": int(len(df)), "n_feats": len(FEATS), "feats": FEATS,
        "missing_cols": missing_cols,
        "target_sd_cm": round(float(np.std(y_cm)), 3),
        "E_fits": E_fits, "gate": gate, "verdict": verdict,
        "elapsed_s": round(time.time() - t_start, 1)}
with open(os.path.join(PROC, "p2b_stefan_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n=== 게이트 ===")
for cv in ["spatial_block", "LORO"]:
    g = gate[cv]
    print(f"[{cv}] RESIDUAL {g['rmse_RESIDUAL']:.2f} vs ML {g['rmse_ML_only']:.2f} "
          f"(Δ{g['d_vs_ML']:+.2f}) vs PHYS {g['rmse_PHYS_only']:.2f} (Δ{g['d_vs_PHYS']:+.2f}) "
          f"→ 개선={g['RESIDUAL_improves_both']}")
print(f"판정: {verdict}")
print(f"\n[done] ({time.time()-t_start:.0f}s)")
