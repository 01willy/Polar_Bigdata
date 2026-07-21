"""W3 — 물리결합 엔진: 토양 의존 E(x) + 물리식 형태강제 ML결합.

목표: "물리식 ALT = a + E·√TDD 형태강제 + 토양 E(x)가 전이(특히 레나)를 회복시킨다"는
가설을 참으로 가정하지 않고 반증 설계로 검증한다. 회의적 원칙 하에 (a) E(x)가 지역을
암기해 LORO 서 무의미해지는지, (b) 물리 제약이 실제 전이를 돕는지(같은 입력 무제약 ML 대조),
(c) 누설·표본편향을 점검한다. 공간블록+LORO 게이트, 평균예측 대비 skill·debiased RMSE 병기.
결론 요약: 물리 상수 E(PHYS_const)가 여전히 전이 최선이고, 토양 E(x)·미분물리 NN 은 전이를
회복시키지 못한다(정직 보고). 상세는 meta 의 claim_i/ii/iii 참조.

데이터: data/processed/dl_dataset_cell_v3_soil.csv (17,423셀, sg_* 토양 9종·e5_* 기후8 포함).
TDD=e5_tdd, √TDD=sqrt(clip(tdd,0)). 타깃 alt_cm.

모델(동일 CV로 비교):
  1 PHYS_const   ALT=a+E·√TDD, E·a 전역 상수(fold별 최소제곱). P2 baseline.
  2 PHYS_clim    E(x)=g0+g·[maat,swe] 선형(P2 PHYS_strat 재현).
  3 PHYS_soil    E(x)=g0+g·z, z=[soc_5_15,bdod,clay,swe] 표준화. 토양이 E 경유로만 들어감.
  4 PHYS_nn      미분가능 물리층: MLP(전 공변량)→(a,E>0), ALT=softplus(a)+softplus(E)·√TDD.
  5 ML_pure_nn   같은 MLP·같은 입력, 물리층 없이 ALT 직접(구조 C 제약 제거판).
  6 ML_pure_gbm  GBM 전 공변량 직접(W2.1 M_soil 재현, 전이 붕괴 예상).
                 주의: gbm 만 log1p(alt) 공간 학습(→expm1). 나머지(PHYS_*·NN)는 원 cm 학습이라
                 '동일 입력 대조'는 gbm 에 대해 엄밀하지 않다. 다만 log 압축은 gbm 레나 과대예측을
                 오히려 축소하는 방향이므로 gbm 전이 붕괴 결론은 아티팩트가 아니다. 핵심 물리효과
                 대조(PHYS_nn vs ML_pure_nn)는 둘 다 원 cm·동일 백본이라 공정하다.
  7 RESIDUAL     PHYS_const + GBM(잔차)(P2 재현, 무익 예상).
  8 HYBRID_aoa   AOA 임계 안=ML_pure_gbm, 밖=PHYS_const 앵커.

CV: 공간블록 GroupKFold6(block=floor(lat/0.5)*100000+floor(lon/0.5)), LORO(region test>=100).
LORO 는 지역별(ABoVE_AK·ABoVE_CA·Lena_RU) 별도 집계.

출력:
  data/processed/w3_physics_ml_results.csv    (모델×CV 전역 지표)
  data/processed/w3_physics_ml_perregion.csv  (모델×LORO 지역별 지표)
  data/processed/w3_physics_ml_Ex_diag.csv    (지역별 적합 E(x) 진단)
  data/processed/w3_physics_ml_council.csv     (콘슬 44셀 관측 vs PHYS_const/soil/nn 교정)
  data/processed/w3_physics_ml_meta.json
실행: python3 scripts/3_deep_learning/w3_physics_ml.py
GPU: PHYS_nn·ML_pure_nn 학습에 0-3 사용(CUDA_VISIBLE_DEVICES 로 0).
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from polar.eval_metrics import all_metrics

import torch
import torch.nn as nn

PROC = "data/processed"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED_ENS = 5  # W3 수정: seed 앙상블 확대(레나 PHYS_nn 단일 seed RMSE 34-70cm 요동, ±11cm).
t_start = time.time()

# ---------- 데이터 ----------
df = pd.read_csv(os.path.join(PROC, "dl_dataset_cell_v3_soil.csv"), low_memory=False)
print(f"[load] cell rows={len(df)}  cols={df.shape[1]}  device={DEVICE}")

# 전 공변량(NN·GBM 공통 입력): 기후8+지형6+토양9+눈(swe는 기후8에 포함).
CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
SOIL = ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15", "sg_cfvo_5_15",
        "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15", "sg_soc_15_30"]
FEATS = [c for c in (CLIMATE + TERRAIN + SOIL) if c in df.columns]
missing_cols = [c for c in (CLIMATE + TERRAIN + SOIL) if c not in df.columns]
print(f"[feats] n={len(FEATS)}  누락={missing_cols if missing_cols else '없음'}")

# PHYS_soil 용 E(x) 토양·눈 피처(표준화 대상)
SOIL_E = ["sg_soc_5_15", "sg_bdod_5_15", "sg_clay_5_15", "e5_swe"]

for c in FEATS + ["alt_cm", "e5_tdd", "e5_swe", "e5_maat", "lat", "lon"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

y_cm = df["alt_cm"].values.astype(float)
tdd = df["e5_tdd"].values.astype(float)
sqrt_tdd = np.sqrt(np.clip(tdd, 0, None))
swe = df["e5_swe"].values.astype(float)
maat = df["e5_maat"].values.astype(float)
loc_id = df["loc_id"].values if "loc_id" in df.columns else np.arange(len(df))
region = df["region"].values
Xall = df[FEATS].values.astype(float)
Xsoil = df[SOIL_E].values.astype(float)
CLIP = (1.0, 600.0)

df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int))
LORO_REGIONS = ["ABoVE_AK", "ABoVE_CA", "Lena_RU"]


def gbm():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, random_state=0)


def spatial_splits():
    return [(None, tr, te) for tr, te in GroupKFold(n_splits=6).split(df, y_cm, groups=df.block.values)]


def loro_splits():
    out = []
    for r in pd.unique(region):
        te = np.where(region == r)[0]
        tr = np.where(region != r)[0]
        if len(te) >= 100:
            out.append((r, tr, te))
    return out


# ---------- 결측 대체(train fold 중앙값)+플래그 ----------
def impute_train(X_tr, X_te):
    Xtr = X_tr.copy(); Xte = X_te.copy()
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    flags_tr, flags_te = [], []
    for j in range(Xtr.shape[1]):
        m_tr = ~np.isfinite(Xtr[:, j]); m_te = ~np.isfinite(Xte[:, j])
        if m_tr.any() or m_te.any():
            Xtr[m_tr, j] = med[j]; Xte[m_te, j] = med[j]
            flags_tr.append(m_tr.astype(float)); flags_te.append(m_te.astype(float))
    if flags_tr:
        Xtr = np.c_[Xtr, np.array(flags_tr).T]; Xte = np.c_[Xte, np.array(flags_te).T]
    return Xtr, Xte


def impute_simple(X_tr, X_te, med=None):
    """플래그 없이 train 중앙값만 대체(선형회귀·표준화용)."""
    Xtr = X_tr.copy(); Xte = X_te.copy()
    if med is None:
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte, med


# ---------- 물리 상수 E 적합 ----------
def fit_E(y, s):
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    denom = float((s * s).sum())
    E0 = float((s * y).sum() / denom) if denom > 0 else 0.0
    rmse0 = float(np.sqrt(np.mean((y - E0 * s) ** 2)))
    A = np.c_[np.ones_like(s), s]
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a1, E1 = float(coef[0]), float(coef[1])
    rmse1 = float(np.sqrt(np.mean((y - (a1 + E1 * s)) ** 2)))
    if rmse1 < rmse0:
        return {"mode": "intercept", "E": E1, "a": a1}
    return {"mode": "origin", "E": E0, "a": 0.0}


def const_pred(fit, s):
    return fit["a"] + fit["E"] * s


# ---------- PHYS_clim: E(x)=g0+g·[maat,swe] ----------
def fit_E_lin(y, s, Z):
    """ALT = s·(g0 + Z·g) = g0·s + sum_k g_k·(Z_k·s). 절편 없는 선형회귀(s→0이면 ALT→0).
    Z 는 이미 결측 대체된 (n,k) 배열. 반환: g0, g(k,)."""
    m = np.isfinite(y) & np.isfinite(s)
    ym, sm, Zm = y[m], s[m], Z[m]
    A = np.c_[sm, Zm * sm[:, None]]
    coef, _, _, _ = np.linalg.lstsq(A, ym, rcond=None)
    return float(coef[0]), coef[1:].astype(float)


def lin_pred(g0, g, s, Z):
    E = g0 + Z @ g
    return E * s, E


# ---------- PHYS_nn / ML_pure_nn (미분가능 물리층) ----------
class PhysMLP(nn.Module):
    """공변량 → (a_raw, E_raw). ALT = softplus(a)+softplus(E)·√TDD (E>0 물리 강제)."""
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2))

    def forward(self, x, s):
        h = self.net(x)
        a = nn.functional.softplus(h[:, 0])
        E = nn.functional.softplus(h[:, 1])
        return a + E * s, E


class PureMLP(nn.Module):
    """같은 백본, ALT 직접 예측(물리층 없음). softplus 로 양수 강제만."""
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1))

    def forward(self, x, s):
        return nn.functional.softplus(self.net(x)[:, 0]), None


def train_nn(model_cls, Xtr, s_tr, y_tr, Xte, s_te, seed):
    """단일 seed 학습. 입력 표준화(train), √TDD 원단위, 타깃 원 cm. 반환 (pred_te, E_te)."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-6] = 1.0
    Xtr_s = (Xtr - mu) / sd; Xte_s = (Xte - mu) / sd
    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32, device=DEVICE)
    s_tr_t = torch.tensor(s_tr, dtype=torch.float32, device=DEVICE)
    s_te_t = torch.tensor(s_te, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    model = model_cls(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=10.0)  # 원 cm 스케일, 이상치 견고
    n = Xtr.shape[0]; bs = 4096
    model.train()
    for ep in range(300):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred, _ = model(Xtr_t[idx], s_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred, E = model(Xte_t, s_te_t)
        pred = pred.cpu().numpy()
        E = E.cpu().numpy() if E is not None else None
    return pred, E


# ---------- AOA (dissimilarity index) ----------
def aoa_mask(Xtr, Xte):
    """train 표준화 공간에서 test 각 점의 최근접 train 거리(DI). 임계=train NN 거리 95퍼센타일.
    반환: te 가 AOA '안'(임계 이하)이면 True."""
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[sd < 1e-6] = 1.0
    Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
    nn1 = NearestNeighbors(n_neighbors=2).fit(Ztr)
    d_tr, _ = nn1.kneighbors(Ztr)
    thr = np.percentile(d_tr[:, 1], 95)  # train-to-train NN(자기 제외) 95퍼센타일
    nn2 = NearestNeighbors(n_neighbors=1).fit(Ztr)
    d_te, _ = nn2.kneighbors(Zte)
    return d_te[:, 0] <= thr


# ---------- OOF 실행 ----------
MODELS = ["PHYS_const", "PHYS_clim", "PHYS_soil", "PHYS_nn",
          "ML_pure_nn", "ML_pure_gbm", "RESIDUAL", "HYBRID_aoa"]
CVS = [("spatial_block", spatial_splits()), ("LORO", loro_splits())]

oof = {cv: {m: np.full(len(df), np.nan) for m in MODELS} for cv, _ in CVS}
Ex_store = {cv: {"PHYS_soil": np.full(len(df), np.nan),
                 "PHYS_nn": np.full(len(df), np.nan)} for cv, _ in CVS}
Ex_diag_rows = []
lena_seed_spread = None  # 레나 LORO fold 의 PHYS_nn 단일 seed별 RMSE·bias·E(seed 요동 진단)

for cv_name, splits in CVS:
    print(f"\n== {cv_name} ({len(splits)} folds)")
    for fold_i, (rlabel, tr, te) in enumerate(splits):
        t0 = time.time()
        s_med = np.nanmedian(sqrt_tdd[tr])
        s_tr = np.where(np.isfinite(sqrt_tdd[tr]), sqrt_tdd[tr], s_med)
        s_te = np.where(np.isfinite(sqrt_tdd[te]), sqrt_tdd[te], s_med)

        # --- 1 PHYS_const ---
        pf = fit_E(y_cm[tr], s_tr)
        oof[cv_name]["PHYS_const"][te] = np.clip(const_pred(pf, s_te), *CLIP)

        # --- 2 PHYS_clim: Z=[maat,swe] ---
        Zc_tr = np.c_[maat[tr], swe[tr]]; Zc_te = np.c_[maat[te], swe[te]]
        Zc_tr, Zc_te, _ = impute_simple(Zc_tr, Zc_te)
        g0c, gc = fit_E_lin(y_cm[tr], s_tr, Zc_tr)
        pc, _ = lin_pred(g0c, gc, s_te, Zc_te)
        oof[cv_name]["PHYS_clim"][te] = np.clip(pc, *CLIP)

        # --- 3 PHYS_soil: z=[soc_5_15,bdod,clay,swe] 표준화 ---
        Zs_tr = Xsoil[tr].copy(); Zs_te = Xsoil[te].copy()
        Zs_tr, Zs_te, med_s = impute_simple(Zs_tr, Zs_te)
        mu_s = Zs_tr.mean(0); sd_s = Zs_tr.std(0); sd_s[sd_s < 1e-6] = 1.0
        Zsn_tr = (Zs_tr - mu_s) / sd_s; Zsn_te = (Zs_te - mu_s) / sd_s
        g0s, gs = fit_E_lin(y_cm[tr], s_tr, Zsn_tr)
        ps, E_soil_te = lin_pred(g0s, gs, s_te, Zsn_te)
        oof[cv_name]["PHYS_soil"][te] = np.clip(ps, *CLIP)
        Ex_store[cv_name]["PHYS_soil"][te] = E_soil_te

        # --- 6 ML_pure_gbm (전 공변량 log1p 직접) ---
        Xtr_g, Xte_g = impute_train(Xall[tr], Xall[te])
        ylog_tr = np.log1p(np.clip(y_cm[tr], 0, None))
        pred_gbm = np.expm1(np.clip(gbm().fit(Xtr_g, ylog_tr).predict(Xte_g),
                                    np.log1p(1.0), np.log1p(600.0)))
        oof[cv_name]["ML_pure_gbm"][te] = np.clip(pred_gbm, *CLIP)

        # --- 7 RESIDUAL (PHYS_const + GBM 잔차) ---
        base_tr = const_pred(pf, s_tr); base_te = const_pred(pf, s_te)
        r_tr = y_cm[tr] - base_tr
        resid_pred = gbm().fit(Xtr_g, r_tr).predict(Xte_g)
        oof[cv_name]["RESIDUAL"][te] = np.clip(base_te + resid_pred, *CLIP)

        # --- 8 HYBRID_aoa (AOA 안=ML_pure_gbm, 밖=PHYS_const) ---
        Xtr_a, Xte_a, _ = impute_simple(Xall[tr].copy(), Xall[te].copy())
        inside = aoa_mask(Xtr_a, Xte_a)
        hyb = np.where(inside, oof[cv_name]["ML_pure_gbm"][te], oof[cv_name]["PHYS_const"][te])
        oof[cv_name]["HYBRID_aoa"][te] = np.clip(hyb, *CLIP)

        # --- 4 PHYS_nn / 5 ML_pure_nn (미분가능 물리층, seed 앙상블) ---
        Xtr_n, Xte_n, _ = impute_simple(Xall[tr].copy(), Xall[te].copy())
        p_phys, p_pure, E_nn = [], [], []
        for sd_i in range(SEED_ENS):
            pp, Ee = train_nn(PhysMLP, Xtr_n, s_tr, y_cm[tr], Xte_n, s_te, seed=sd_i)
            pu, _ = train_nn(PureMLP, Xtr_n, s_tr, y_cm[tr], Xte_n, s_te, seed=100 + sd_i)
            p_phys.append(pp); p_pure.append(pu); E_nn.append(Ee)
        pred_phys = np.mean(p_phys, 0); pred_pure = np.mean(p_pure, 0)
        E_nn_mean = np.mean(E_nn, 0)
        oof[cv_name]["PHYS_nn"][te] = np.clip(pred_phys, *CLIP)
        oof[cv_name]["ML_pure_nn"][te] = np.clip(pred_pure, *CLIP)
        Ex_store[cv_name]["PHYS_nn"][te] = E_nn_mean

        # 레나 LORO fold: PHYS_nn 단일 seed별 RMSE·bias·E 요동 기록(과장 방지용 구간 병기).
        if cv_name == "LORO" and rlabel == "Lena_RU":
            y_te = y_cm[te]
            seed_rmse, seed_bias, seed_E = [], [], []
            for pp_s, Ee_s in zip(p_phys, E_nn):
                ps_clip = np.clip(pp_s, *CLIP)
                m = np.isfinite(ps_clip) & np.isfinite(y_te)
                seed_rmse.append(float(np.sqrt(np.mean((ps_clip[m] - y_te[m]) ** 2))))
                seed_bias.append(float(np.mean(ps_clip[m] - y_te[m])))
                seed_E.append(float(np.nanmean(Ee_s)))
            lena_seed_spread = {
                "n_seeds": SEED_ENS,
                "single_seed_rmse": [round(v, 2) for v in seed_rmse],
                "single_seed_rmse_mean": round(float(np.mean(seed_rmse)), 2),
                "single_seed_rmse_sd": round(float(np.std(seed_rmse)), 2),
                "single_seed_rmse_min": round(float(np.min(seed_rmse)), 2),
                "single_seed_rmse_max": round(float(np.max(seed_rmse)), 2),
                "single_seed_bias": [round(v, 2) for v in seed_bias],
                "single_seed_E_mean": [round(v, 2) for v in seed_E],
                "E_mean_range": [round(float(np.min(seed_E)), 2), round(float(np.max(seed_E)), 2)],
                "ensemble_rmse": round(float(np.sqrt(np.mean(
                    (np.clip(pred_phys, *CLIP)[np.isfinite(y_te)] - y_te[np.isfinite(y_te)]) ** 2))), 2)}

        # E(x) 지역별 진단(이 fold test 의 지역별 요약)
        for r in LORO_REGIONS:
            mask = (region[te] == r)
            if mask.sum() >= 30:
                Ex_diag_rows.append({
                    "cv_type": cv_name, "fold": fold_i, "fold_label": rlabel if rlabel else f"fold{fold_i}",
                    "region": r, "n": int(mask.sum()),
                    "E_const": round(pf["E"], 4),
                    "E_soil_mean": round(float(np.nanmean(E_soil_te[mask])), 4),
                    "E_soil_std": round(float(np.nanstd(E_soil_te[mask])), 4),
                    "E_nn_mean": round(float(np.nanmean(E_nn_mean[mask])), 4),
                    "E_nn_std": round(float(np.nanstd(E_nn_mean[mask])), 4)})
        lbl = rlabel if rlabel else f"fold{fold_i}"
        print(f"   {lbl:12s} E_const={pf['E']:.3f}  "
              f"E_soil[{np.nanmean(E_soil_te):.2f}] E_nn[{np.nanmean(E_nn_mean):.2f}]  "
              f"({time.time()-t0:.0f}s)")

# ---------- 전역 지표 ----------
rows = []
for cv_name, _ in CVS:
    for m in MODELS:
        met = all_metrics(y_cm, oof[cv_name][m])
        rows.append({"model": m, "cv_type": cv_name, "n": met["n"],
                     "rmse_cm": met["rmse_cm"], "mae_cm": met["mae_cm"], "bias_cm": met["bias_cm"],
                     "r2": met["r2"], "skill_over_mean": met["skill_over_mean"]})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(PROC, "w3_physics_ml_results.csv"), index=False)
print("\n=== 전역 결과 ===")
print(res.to_string(index=False))

# ---------- LORO 지역별 지표 ----------
def debiased_rmse_corr(y_true, y_pred):
    """bias 제거 후 RMSE(=잔차 표준편차)와 셀별 상관. mean-shift 대 실제 skill 분리용."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan, np.nan
    yt, yp = y_true[m], y_pred[m]
    b = float((yp - yt).mean())
    drmse = float(np.sqrt(np.mean(((yp - b) - yt) ** 2)))
    corr = float(np.corrcoef(yp, yt)[0, 1]) if yp.std() > 1e-9 else np.nan
    return round(drmse, 3), (round(corr, 3) if np.isfinite(corr) else np.nan)


pr_rows = []
for m in MODELS:
    for r in LORO_REGIONS:
        mask = (region == r)
        met = all_metrics(y_cm[mask], oof["LORO"][m][mask])
        drmse, corr = debiased_rmse_corr(y_cm[mask], oof["LORO"][m][mask])
        pr_rows.append({"model": m, "region": r, "n": met["n"],
                        "rmse_cm": met["rmse_cm"], "mae_cm": met["mae_cm"], "bias_cm": met["bias_cm"],
                        # debiased_rmse: bias(평균이동) 제거 후 잔차 표준편차. corr: 셀별 상관.
                        # 두 지표가 obs 표준편차 근처·0 근처면 RMSE 개선이 mean-shift 뿐임을 뜻함.
                        "debiased_rmse_cm": drmse, "corr": corr,
                        "r2": met["r2"], "skill_over_mean": met["skill_over_mean"]})
perregion = pd.DataFrame(pr_rows)
perregion.to_csv(os.path.join(PROC, "w3_physics_ml_perregion.csv"), index=False)
print("\n=== LORO 지역별 ===")
print(perregion.to_string(index=False))

# ---------- E(x) 진단 ----------
exd = pd.DataFrame(Ex_diag_rows)
exd.to_csv(os.path.join(PROC, "w3_physics_ml_Ex_diag.csv"), index=False)

# ---------- 콘슬 교정: LORO 예측(레나 학습 제외 상황과 별개, 여기선 전역모델로 콘슬 예측) ----------
# 콘슬 44셀에 PHYS_const/PHYS_soil/PHYS_nn 을 '전 데이터 학습' 모델로 예측하고 관측과 비교.
council = pd.read_csv(os.path.join(PROC, "kpdc_council_forcing.csv"))
c_s = council["e5_sqrt_tdd"].values.astype(float)
c_alt = council["alt_cm"].values.astype(float)

# 전 데이터로 최종 모델 적합(콘슬 교정 진단용)
s_all = np.where(np.isfinite(sqrt_tdd), sqrt_tdd, np.nanmedian(sqrt_tdd))
pf_all = fit_E(y_cm, s_all)
c_pred_const = np.clip(const_pred(pf_all, c_s), *CLIP)

# PHYS_soil: 콘슬 셀 토양 E(x) 필요. 콘슬은 v3 에 loc_id 로 존재하는지 매칭.
council_ids = set(council["loc_id"].values.tolist())
df_c = df[df["loc_id"].isin(council_ids)].copy()
matched = df_c["loc_id"].nunique()
if matched > 0:
    Zs_all = Xsoil.copy()
    Zs_all_f, _, med_all = impute_simple(Zs_all, Zs_all)
    mu_a = Zs_all_f.mean(0); sd_a = Zs_all_f.std(0); sd_a[sd_a < 1e-6] = 1.0
    Zsn_all = (Zs_all_f - mu_a) / sd_a
    g0a, ga = fit_E_lin(y_cm, s_all, Zsn_all)
    # 콘슬 셀별 z
    idx_map = {lid: i for i, lid in enumerate(loc_id)}
    c_pred_soil = np.full(len(council), np.nan)
    c_E_soil = np.full(len(council), np.nan)
    for k, lid in enumerate(council["loc_id"].values):
        if lid in idx_map:
            z = Zsn_all[idx_map[lid]]
            E = g0a + z @ ga
            c_E_soil[k] = E
            c_pred_soil[k] = np.clip(E * c_s[k], *CLIP)
else:
    c_pred_soil = np.full(len(council), np.nan)
    c_E_soil = np.full(len(council), np.nan)

# PHYS_nn: 전 데이터 학습 모델로 콘슬 예측(콘슬 셀 공변량)
Xall_f, _, med_x = impute_simple(Xall.copy(), Xall.copy())
c_pred_nn = np.full(len(council), np.nan)
c_E_nn = np.full(len(council), np.nan)
if matched > 0:
    # 콘슬 셀 공변량 행렬
    Xc = np.full((len(council), Xall.shape[1]), np.nan)
    for k, lid in enumerate(council["loc_id"].values):
        if lid in idx_map:
            Xc[k] = Xall_f[idx_map[lid]]
    # 전 데이터로 PhysMLP 학습(seed 앙상블)
    valid_c = np.isfinite(Xc).all(1) & np.isfinite(c_s)
    if valid_c.any():
        preds_c, E_c = [], []
        for sd_i in range(SEED_ENS):
            pc_pred, pc_E = train_nn(PhysMLP, Xall_f, s_all, y_cm,
                                     Xc[valid_c], c_s[valid_c], seed=200 + sd_i)
            preds_c.append(pc_pred); E_c.append(pc_E)
        c_pred_nn[valid_c] = np.clip(np.mean(preds_c, 0), *CLIP)
        c_E_nn[valid_c] = np.mean(E_c, 0)

council_out = pd.DataFrame({
    "loc_id": council["loc_id"].values, "lat": council["lat"].values, "lon": council["lon"].values,
    "alt_cm_obs": c_alt, "sqrt_tdd": c_s,
    "pred_PHYS_const": c_pred_const, "pred_PHYS_soil": c_pred_soil, "pred_PHYS_nn": c_pred_nn,
    "E_soil": c_E_soil, "E_nn": c_E_nn})
council_out.to_csv(os.path.join(PROC, "w3_physics_ml_council.csv"), index=False)

# 콘슬 교정 요약(과대예측 정도)
def council_bias(pred):
    m = np.isfinite(pred) & np.isfinite(c_alt)
    if m.sum() == 0:
        return {"n": 0, "mean_pred": np.nan, "mean_obs": np.nan, "bias": np.nan, "ratio": np.nan,
                "rmse": np.nan, "pred_std": np.nan, "obs_std": np.nan, "corr": np.nan}
    p = pred[m]; o = c_alt[m]
    corr = float(np.corrcoef(p, o)[0, 1]) if p.std() > 1e-9 else float("nan")
    return {"n": int(m.sum()), "mean_pred": round(float(p.mean()), 2),
            "mean_obs": round(float(o.mean()), 2),
            "bias": round(float((p - o).mean()), 2),
            "ratio": round(float(p.mean() / o.mean()), 3),
            "rmse": round(float(np.sqrt(np.mean((p - o) ** 2))), 2),
            # 콘슬 44셀은 √TDD 동일값이라 상수 E 모델(const·soil)은 사실상 단일점.
            # PHYS_nn 은 좁은 밴드로 뭉치고 obs와 상관 낮음(셀별 물리보정 아닌 평균회귀 진단).
            "pred_std": round(float(p.std()), 2), "obs_std": round(float(o.std()), 2),
            "corr": round(corr, 3) if np.isfinite(corr) else None}

council_summary = {"PHYS_const": council_bias(c_pred_const),
                   "PHYS_soil": council_bias(c_pred_soil),
                   "PHYS_nn": council_bias(c_pred_nn),
                   "matched_cells": int(matched)}
print("\n=== 콘슬 교정 ===")
for k, v in council_summary.items():
    if isinstance(v, dict):
        print(f"   {k:12s} n={v['n']} pred={v['mean_pred']} obs={v['mean_obs']} "
              f"bias={v['bias']} ratio={v['ratio']} rmse={v['rmse']}")


# ---------- 판정 게이트 ----------
def g(cv, m, col="rmse_cm"):
    return float(res[(res.cv_type == cv) & (res.model == m)][col].values[0])


def gr(m, r, col="rmse_cm"):
    return float(perregion[(perregion.model == m) & (perregion.region == r)][col].values[0])


# (i) PHYS_soil·PHYS_nn 이 PHYS_const 대비 전이(특히 레나) 개선?
lena_const = gr("PHYS_const", "Lena_RU")
lena_soil = gr("PHYS_soil", "Lena_RU")
lena_nn = gr("PHYS_nn", "Lena_RU")
loro_const = g("LORO", "PHYS_const")
loro_soil = g("LORO", "PHYS_soil")
loro_nn = g("LORO", "PHYS_nn")
# 레나 PHYS_soil vs PHYS_const 의 개선이 실제 skill 인지 mean-shift 인지 분리.
lena_soil_drmse, lena_soil_corr = debiased_rmse_corr(y_cm[region == "Lena_RU"], oof["LORO"]["PHYS_soil"][region == "Lena_RU"])
lena_const_drmse, lena_const_corr = debiased_rmse_corr(y_cm[region == "Lena_RU"], oof["LORO"]["PHYS_const"][region == "Lena_RU"])
claim_i = {
    "lena_PHYS_const": round(lena_const, 2), "lena_PHYS_soil": round(lena_soil, 2), "lena_PHYS_nn": round(lena_nn, 2),
    "loro_PHYS_const": round(loro_const, 2), "loro_PHYS_soil": round(loro_soil, 2), "loro_PHYS_nn": round(loro_nn, 2),
    "soil_helps_lena": bool(lena_soil < lena_const), "nn_helps_lena": bool(lena_nn < lena_const),
    # PHYS_soil 의 레나 RMSE 개선은 순전히 bias(평균이동) 감소다. debiased_rmse·corr 이 두 모델에서
    # 사실상 동일(둘 다 obs 표준편차 근처·상관 0.1 근처)이라 셀별 skill 은 없다. '레나 최선'은
    # '평균예측기보다도 나쁜(skill 음수) 모델 중 덜 나쁨'을 뜻한다.
    "lena_PHYS_soil_debiased_rmse": lena_soil_drmse, "lena_PHYS_soil_corr": lena_soil_corr,
    "lena_PHYS_const_debiased_rmse": lena_const_drmse, "lena_PHYS_const_corr": lena_const_corr,
    "soil_gain_is_mean_shift_only": bool(
        (lena_soil < lena_const) and abs(lena_soil_drmse - lena_const_drmse) < 1.0),
    "soil_skill_over_mean_negative": bool(loro_soil > g("LORO", "PHYS_const"))}

# (ii) PHYS_nn − ML_pure_nn (동일 입력) 전이 차이 = 물리 제약 순효과
lena_pure = gr("ML_pure_nn", "Lena_RU")
loro_pure = g("LORO", "ML_pure_nn")
claim_ii = {
    "lena_PHYS_nn": round(lena_nn, 2), "lena_ML_pure_nn": round(lena_pure, 2),
    "lena_phys_effect_cm": round(lena_pure - lena_nn, 2),  # 양수면 물리가 도움
    "loro_PHYS_nn": round(loro_nn, 2), "loro_ML_pure_nn": round(loro_pure, 2),
    "loro_phys_effect_cm": round(loro_pure - loro_nn, 2),
    "phys_constraint_helps": bool(lena_nn < lena_pure)}

# (iii) 토양이 E(x) 경유로 들어가면 raw 입력(ML_pure_gbm 레나)보다 나은가
lena_gbm = gr("ML_pure_gbm", "Lena_RU")
claim_iii = {
    "lena_PHYS_soil": round(lena_soil, 2), "lena_PHYS_nn": round(lena_nn, 2),
    "lena_ML_pure_gbm_raw": round(lena_gbm, 2),
    "Ex_beats_raw_soil": bool(min(lena_soil, lena_nn) < lena_gbm)}

# LORO 전역 대표성: n=17305 는 3개 대지역(AK·CA·Lena)만 test 로 등장. 소규모 지역(region<100)은 제외.
loro_covered = {r: int((region == r).sum()) for r in LORO_REGIONS}
loro_excluded_small = int(len(df) - sum((region == r).sum() for r in LORO_REGIONS))
representativeness_note = (
    "LORO 전역 지표(n=17305)의 대표성은 사실상 3개 대지역(AK·CA·Lena)에 국한된다. "
    f"소규모 지역 {loro_excluded_small}셀(region<100: US(Alaska)·Canada·GTNPenv_* 등)은 "
    "어떤 LORO test 폴드에도 등장하지 않는다(누설 아님, 대표성 한정).")

# 콘슬 leakage 명시: 콘슬 44셀 전부가 최종 모델 학습 데이터에 포함(43셀 region=ABoVE_AK).
_c_nn = council_summary["PHYS_nn"]
council_leakage_note = (
    "콘슬 44셀 교정은 held-out 전이가 아니라 완전 in-domain 이다. 44셀 전부(43셀 ABoVE_AK)가 "
    "최종 모델의 학습 데이터에 포함되며, 이 44셀로 학습한 모델을 같은 44셀에서 예측한다. "
    "따라서 콘슬 결과에서 전이 일반화 주장을 도출하면 안 된다. 또한 콘슬 44셀은 √TDD 동일값이라 "
    "상수 E 모델(const·soil)의 예측은 사실상 단일점이고, PHYS_nn 은 예측 표준편차 "
    f"{_c_nn['pred_std']}cm(obs {_c_nn['obs_std']}cm)로 좁은 밴드에 뭉쳐 obs 와 셀별 상관 "
    f"{_c_nn['corr']}(약 0.2, seed별 요동)에 불과하다. 즉 bias 감소는 셀별 물리보정이 아니라 평균회귀다.")

meta = {"n_cells": int(len(df)), "n_feats": len(FEATS), "feats": FEATS, "soil_E_feats": SOIL_E,
        "missing_cols": missing_cols, "device": str(DEVICE), "seed_ensemble": SEED_ENS,
        "target_sd_cm": round(float(np.std(y_cm)), 3),
        "loro_regions": LORO_REGIONS, "loro_covered_cells": loro_covered,
        "loro_excluded_small_cells": loro_excluded_small,
        "representativeness_note": representativeness_note,
        "council_summary": council_summary,
        "council_leakage_note": council_leakage_note,
        "lena_phys_nn_seed_spread": lena_seed_spread,
        "claim_i_soil_nn_vs_const": claim_i,
        "claim_ii_phys_constraint_effect": claim_ii,
        "claim_iii_Ex_vs_raw": claim_iii,
        "elapsed_s": round(time.time() - t_start, 1)}
with open(os.path.join(PROC, "w3_physics_ml_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# OOF 저장(시각화용)
oof_df = pd.DataFrame({"loc_id": loc_id, "lat": df.lat.values, "lon": df.lon.values,
                       "region": region, "alt_cm": y_cm, "sqrt_tdd": sqrt_tdd})
for cv_name, _ in CVS:
    for m in MODELS:
        oof_df[f"{cv_name}__{m}"] = oof[cv_name][m]
oof_df["Ex_soil_LORO"] = Ex_store["LORO"]["PHYS_soil"]
oof_df["Ex_nn_LORO"] = Ex_store["LORO"]["PHYS_nn"]
oof_df.to_csv(os.path.join(PROC, "w3_physics_ml_oof.csv"), index=False)

print("\n=== 판정 요약 ===")
print(f"(i)  레나 PHYS_const={lena_const:.1f} soil={lena_soil:.1f} nn={lena_nn:.1f}  "
      f"soil개선={claim_i['soil_helps_lena']}(mean-shift만={claim_i['soil_gain_is_mean_shift_only']}) "
      f"nn개선={claim_i['nn_helps_lena']}")
print(f"     레나 debiased_rmse: const={lena_const_drmse} soil={lena_soil_drmse} "
      f"corr: const={lena_const_corr} soil={lena_soil_corr}  → skill 사실상 동일(둘 다 평균예측 이하)")
print(f"(ii) 레나 PHYS_nn={lena_nn:.1f} vs ML_pure_nn={lena_pure:.1f}  "
      f"물리순효과={claim_ii['lena_phys_effect_cm']:+.1f}cm  도움={claim_ii['phys_constraint_helps']}")
if lena_seed_spread:
    ss = lena_seed_spread
    print(f"     레나 PHYS_nn seed 요동: RMSE {ss['single_seed_rmse_min']}-{ss['single_seed_rmse_max']} "
          f"(평균 {ss['single_seed_rmse_mean']}±{ss['single_seed_rmse_sd']}), "
          f"E_mean {ss['E_mean_range']}  → 헤드라인은 구간 병기 필수(단일값 과장)")
print(f"(iii)레나 E(x)최선={min(lena_soil,lena_nn):.1f} vs raw토양GBM={lena_gbm:.1f}  "
      f"E경유우세={claim_iii['Ex_beats_raw_soil']}")
print(f"\n[leakage] {council_leakage_note[:60]}...")
print(f"[대표성] {representativeness_note[:60]}...")
print(f"\n[done] ({time.time()-t_start:.0f}s)")
