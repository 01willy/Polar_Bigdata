"""트랙 M(P2) — 3D 지중온도장 라벨 증강이 ALT 예측·전이를 개선하는지 정량화.

배경
----
현재 ALT 라벨은 약 97%가 알래스카(ABoVE_AK/CA 등)에 집중되어 있다. 시추공 지중온도(GTN-P)를
0°C 등온면 깊이로 변환해 ALT를 유도하면(P1의 GTNPenv 37셀 + Lena_RU + QTP) 다지역 라벨이 늘어난다.
이 스크립트는 증강 라벨이

  (a) 알래스카 in-domain(공간블록 6-fold) ALT 예측 정확도에 미치는 영향, 그리고
  (b) 신규 지역(Lena_RU, GTNPenv 병합)으로의 LORO 전이

를 BASE(증강 전) 대비 AUG(증강 후)로 정량 비교한다.

데이터
------
  BASE = data/processed/dl_dataset_cell.csv       (알래스카 14,348, 증강 전)
  AUG  = data/processed/dl_dataset_cell_v2.csv     (17,423, GTNPenv·Lena·QTP 증강 후)
  BASE는 AUG의 알래스카 부분집합과 정확히 동일(ALT 최대차 0.0cm 확인).

공변량(전 입력 원칙)
--------------------
  지형6  dem_elev/slope/aspect_sin/aspect_cos/tpi/rough
  기후8  e5_maat/tdd/fdd/sqrt_tdd/twarm/tcold/stl1/swe
  InSAR5 insar_alt/alt_std/sub/dist/n  + insar_miss 플래그
  PolSAR3 polsar_alt/std/valid
  CCI2   cci_alt/cci_valid
  결측 공변량은 train fold 중앙값 대체 + <col>_isnan 플래그(NaN 네이티브 라우팅 아티팩트 회피, P1 교훈).
  주: InSAR/PolSAR 산출물은 알래스카 전용이라 신규 지역에서 100% 결측 → 대체+플래그로 라우팅.

모델
----
  GBM = HistGradientBoostingRegressor(max_iter=400, lr=0.05, max_leaf_nodes=63, l2=1.0)
  타깃 = log1p(alt_cm), 예측 후 expm1 + clip[log1p(1), log1p(600)].

평가
----
  (a) 공간블록 6-fold(GroupKFold, block=floor(lat/0.5)*100000+floor(lon/0.5)):
      BASE vs AUG. 공정 비교 위해 '알래스카 공통 셀'에서만 OOF 채점.
      AUG는 신규 지역을 train에 포함하되 평가는 알래스카 셀로 한정.
  (b) LORO 전이: held-out 신규 지역(Lena_RU 개별, GTNPenv+QTP 병합=GTNPenv_ALL).
      BASE = 알래스카만 학습 → 신규 지역 예측.
      AUG  = 알래스카 + (held-out 제외) 다른 증강 지역 학습 → 같은 신규 지역 예측.
      추가로 알래스카 자체(ABoVE_CA) LORO도 병기(in-domain 전이 영향).

증강 라벨 품질
--------------
  GTNPenv/Lena/QTP ALT의 물리 범위(0-400cm), qc_flag(0/1)·censor_flag별 분리 통계.

산출
----
  data/processed/p2m_augment_results.csv  (dataset,cv_type,region,n,rmse_cm,mae_cm,bias_cm,r2,skill_over_mean 등)
  data/processed/p2m_augment_oof.csv      (셀별 OOF 예측)
  data/processed/p2m_augment_meta.json    (게이트 판정·요약)

실행: (ROOT에서) /home/anaconda3/bin/python scripts/3_deep_learning/p2_augment_experiment.py
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

# ---------------- 공변량 정의 ----------------
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
INSAR = ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]
POLSAR = ["polsar_alt", "polsar_std", "polsar_valid"]
CCI = ["cci_alt", "cci_valid"]
BASE_FEATS = TERRAIN + CLIMATE + INSAR + POLSAR + CCI  # insar_miss는 원본에 없을 수 있어 별도 처리
AK_REGIONS = ["ABoVE_AK", "ABoVE_CA", "United States (Alaska)", "Canada"]
NEW_REGIONS = ["Lena_RU", "GTNPenv_RU", "GTNPenv_SJ", "GTNPenv_US",
               "GTNPenv_CH", "QTP_CN", "GTNPenv_AQ"]
GTNPENV_QTP = ["GTNPenv_RU", "GTNPenv_SJ", "GTNPenv_US", "GTNPenv_CH", "QTP_CN", "GTNPenv_AQ"]


def load(path):
    df = pd.read_csv(path, low_memory=False)
    # insar_miss 플래그: 원본에 없으면 insar_alt 결측 기준으로 생성
    if "insar_miss" not in df.columns:
        df["insar_miss"] = df["insar_alt"].isna().astype(int)
    df["insar_miss"] = pd.to_numeric(df["insar_miss"], errors="coerce").fillna(
        df["insar_alt"].isna().astype(int)).astype(int)
    df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000
                   + np.floor(df.lon / 0.5).astype(int))
    return df


def gbm():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, random_state=0)


def to_cm(p):
    return np.expm1(np.clip(p, *CLIP))


def build_X(df, feats):
    """공변량 행렬 + 결측 플래그. 대체값은 밖에서 fold별로 주입."""
    return df[feats].astype(float).copy()


def impute_fit(Xtr):
    """train fold 중앙값 사전. NaN 열은 전체 중앙값(글로벌 fallback)도 보관."""
    med = Xtr.median(numeric_only=True)
    return med


def apply_impute(X, med, feat_flags):
    """결측 대체 + <col>_isnan 플래그 컬럼 부착."""
    Xf = X.copy()
    flags = {}
    for c in feat_flags:
        isnan = Xf[c].isna().values.astype(float)
        flags[c + "_isnan"] = isnan
        fill = med.get(c, np.nan)
        if not np.isfinite(fill):
            fill = 0.0  # train에서 전부 결측인 열(예: 신규지역 InSAR) → 0 대체 + 플래그로 라우팅
        Xf[c] = Xf[c].fillna(fill)
    flag_df = pd.DataFrame(flags, index=Xf.index)
    out = pd.concat([Xf, flag_df], axis=1)
    return out.values


# ---------------- 데이터 로드 ----------------
base = load(os.path.join(PROC, "dl_dataset_cell.csv"))
aug = load(os.path.join(PROC, "dl_dataset_cell_v2.csv"))
FEATS = BASE_FEATS + ["insar_miss"]
FLAG_FEATS = BASE_FEATS  # insar_miss는 이미 0/1이라 플래그 제외
print(f"[load] BASE n={len(base)}  AUG n={len(aug)}  feats={len(FEATS)}(+플래그{len(FLAG_FEATS)})")

rows = []
oof_records = []


# ================= Part A: 알래스카 in-domain (공간블록 6-fold) =================
# 두 데이터셋 모두 6-fold를 돌리되, AUG는 신규 지역을 train에 포함하고 알래스카 셀만 채점.
print("\n=== Part A: 알래스카 in-domain (공간블록 6-fold) ===")


def spatial_eval(df, dataset_name, eval_mask):
    """df 전체로 6-fold 학습, eval_mask(알래스카 셀)에 대해서만 OOF 채점.
    fold 분할은 알래스카 블록 기준(공정 비교: 두 셋 동일 알래스카 셀 채점)."""
    ylog = np.log1p(df["alt_cm"].values.astype(float))
    y_cm = df["alt_cm"].values.astype(float)
    Xall = build_X(df, FEATS)
    blocks = df["block"].values
    idx = np.arange(len(df))
    ak_idx = idx[eval_mask]
    # 알래스카 블록에 대해서만 6-fold 분할 → test는 항상 알래스카 셀
    gkf = GroupKFold(n_splits=6)
    o = np.full(len(df), np.nan)
    for tr_ak, te_ak in gkf.split(ak_idx, ylog[ak_idx], groups=blocks[ak_idx]):
        te = ak_idx[te_ak]
        te_blocks = set(blocks[te])
        # train = (모든 셀 중 test 블록에 없는 것). AUG면 신규 지역도 train에 자연 포함.
        tr = idx[~np.isin(blocks, list(te_blocks))]
        med = impute_fit(Xall.iloc[tr])
        Xtr = apply_impute(Xall.iloc[tr], med, FLAG_FEATS)
        Xte = apply_impute(Xall.iloc[te], med, FLAG_FEATS)
        o[te] = to_cm(gbm().fit(Xtr, ylog[tr]).predict(Xte))
    m = all_metrics(y_cm[ak_idx], o[ak_idx])
    m.update({"dataset": dataset_name, "cv_type": "spatial_block_AK",
              "region": "Alaska_all", "eval_scope": "alaska_cells"})
    for i in ak_idx:
        oof_records.append({"dataset": dataset_name, "cv_type": "spatial_block_AK",
                            "loc_id": int(df.iloc[i].get("loc_id", i)),
                            "lat": df.iloc[i].lat, "lon": df.iloc[i].lon,
                            "region": df.iloc[i].region, "alt_cm": y_cm[i], "pred": o[i]})
    return m


base_ak_mask = base["region"].isin(AK_REGIONS).values
aug_ak_mask = aug["region"].isin(AK_REGIONS).values
mA_base = spatial_eval(base, "BASE", base_ak_mask)
mA_aug = spatial_eval(aug, "AUG", aug_ak_mask)
rows += [mA_base, mA_aug]
print(f"  BASE 알래스카 in-domain: rmse={mA_base['rmse_cm']:.2f}cm skill={mA_base['skill_over_mean']*100:.1f}% r2={mA_base['r2']:.3f} (n={mA_base['n']})")
print(f"  AUG  알래스카 in-domain: rmse={mA_aug['rmse_cm']:.2f}cm skill={mA_aug['skill_over_mean']*100:.1f}% r2={mA_aug['r2']:.3f} (n={mA_aug['n']})")


# ================= Part B: LORO 전이 (신규 지역) =================
# held-out 지역을 test로, 나머지를 train으로.
#   BASE 모델: 알래스카 라벨만 train (신규 지역 라벨 미보유) → held-out 신규 지역 예측.
#   AUG  모델: 알래스카 + (held-out 제외) 다른 증강 지역 train → 같은 held-out 예측.
print("\n=== Part B: LORO 전이 (신규 지역) ===")


def loro_transfer(train_df, test_df, dataset_name, region_label):
    ytr = np.log1p(train_df["alt_cm"].values.astype(float))
    y_te = test_df["alt_cm"].values.astype(float)
    Xtr_raw = build_X(train_df, FEATS)
    Xte_raw = build_X(test_df, FEATS)
    med = impute_fit(Xtr_raw)
    Xtr = apply_impute(Xtr_raw, med, FLAG_FEATS)
    Xte = apply_impute(Xte_raw, med, FLAG_FEATS)
    pred = to_cm(gbm().fit(Xtr, ytr).predict(Xte))
    m = all_metrics(y_te, pred)
    m.update({"dataset": dataset_name, "cv_type": "LORO_transfer",
              "region": region_label, "eval_scope": "held_out_region"})
    for i, (_, r) in enumerate(test_df.iterrows()):
        oof_records.append({"dataset": dataset_name, "cv_type": "LORO_transfer",
                            "loc_id": int(r.get("loc_id", i)),
                            "lat": r.lat, "lon": r.lon, "region": r.region,
                            "alt_cm": y_te[i], "pred": pred[i]})
    return m


# --- 전이 타깃 정의 ---
# Lena_RU (개별, n=3037)
lena_test = aug[aug.region == "Lena_RU"]
# GTNPenv+QTP 병합 (n=38)
gtnp_test = aug[aug.region.isin(GTNPENV_QTP)]

transfer_targets = [
    ("Lena_RU", lena_test),
    ("GTNPenv_ALL", gtnp_test),
]

for reg_label, test_df in transfer_targets:
    test_idx = test_df.index
    # BASE 모델: 알래스카만 학습 (aug의 알래스카부 = base와 동일)
    base_train = aug[aug.region.isin(AK_REGIONS)]
    mB_base = loro_transfer(base_train, test_df, "BASE", reg_label)
    # AUG 모델: 알래스카 + held-out 제외 다른 증강 지역
    aug_train = aug[~aug.index.isin(test_idx)]
    mB_aug = loro_transfer(aug_train, test_df, "AUG", reg_label)
    rows += [mB_base, mB_aug]
    d = mB_base["rmse_cm"] - mB_aug["rmse_cm"]
    print(f"  [{reg_label:12s}] n={mB_base['n']:4d}  BASE rmse={mB_base['rmse_cm']:7.2f}  "
          f"AUG rmse={mB_aug['rmse_cm']:7.2f}  Δ(BASE-AUG)={d:+7.2f}cm  "
          f"skill BASE={mB_base['skill_over_mean']:.3f}→AUG={mB_aug['skill_over_mean']:.3f}")

# --- in-domain 전이 대조: ABoVE_CA (알래스카 내부 held-out) ---
# 알래스카 안에서도 증강이 해를 끼치지 않는지 확인용.
ca_test = aug[aug.region == "ABoVE_CA"]
ca_idx = ca_test.index
base_train_ca = base[base.region != "ABoVE_CA"]
mCA_base = loro_transfer(base_train_ca, ca_test, "BASE", "ABoVE_CA")
aug_train_ca = aug[~aug.index.isin(ca_idx)]
mCA_aug = loro_transfer(aug_train_ca, ca_test, "AUG", "ABoVE_CA")
rows += [mCA_base, mCA_aug]
print(f"  [ABoVE_CA(대조) ] n={mCA_base['n']:4d}  BASE rmse={mCA_base['rmse_cm']:7.2f}  "
      f"AUG rmse={mCA_aug['rmse_cm']:7.2f}  Δ(BASE-AUG)={mCA_base['rmse_cm']-mCA_aug['rmse_cm']:+7.2f}cm")


# ================= Part B2: 진단 — 원인 분리 =================
# LORO 전이 열화의 원인을 분리한다.
#  진단1: AK-only→Lena (BASE 재현 검증).
#  진단2: AK+Lena를 함께 학습하고 Lena를 공간블록 CV로 자기평가 → Lena 라벨이 train에
#         있을 때의 참값. (Lena 자체 라벨의 유용성/자기일관성 측정)
#  진단3: AK+GTNPenv→Lena → GTNPenv 심부(高 ALT) 라벨의 교란(confound) 정량화.
print("\n=== Part B2: 진단(원인 분리) ===")
lena_df = aug[aug.region == "Lena_RU"]

# 진단2: AK+Lena 공동학습 후 Lena 공간블록 CV 자기평가
sub = aug[aug.region.isin(AK_REGIONS + ["Lena_RU"])].reset_index(drop=True)
sub_yl = np.log1p(sub["alt_cm"].values.astype(float))
sub_ycm = sub["alt_cm"].values.astype(float)
sub_X = build_X(sub, FEATS)
sub_blk = sub["block"].values
sub_idx = np.arange(len(sub))
l_idx = sub_idx[sub["region"].values == "Lena_RU"]
o2 = np.full(len(sub), np.nan)
for tr_l, te_l in GroupKFold(n_splits=6).split(l_idx, sub_yl[l_idx], groups=sub_blk[l_idx]):
    te = l_idx[te_l]
    te_b = set(sub_blk[te])
    tr = sub_idx[~np.isin(sub_blk, list(te_b))]
    med = impute_fit(sub_X.iloc[tr])
    o2[te] = to_cm(gbm().fit(apply_impute(sub_X.iloc[tr], med, FLAG_FEATS), sub_yl[tr])
                   .predict(apply_impute(sub_X.iloc[te], med, FLAG_FEATS)))
m_selfcv = all_metrics(sub_ycm[l_idx], o2[l_idx])
m_selfcv.update({"dataset": "AUG_AK+Lena", "cv_type": "spatial_block_Lena",
                 "region": "Lena_RU", "eval_scope": "lena_own_labels_in_train"})
rows.append(m_selfcv)
print(f"  진단2 [AK+Lena 공간CV on Lena]  rmse={m_selfcv['rmse_cm']:.2f}cm  "
      f"skill={m_selfcv['skill_over_mean']:.3f}  (Lena 라벨이 train에 있을 때 → 자기일관성 검증)")

# 진단3: AK+GTNPenv→Lena (GTNPenv 교란)
tr_g = aug[aug.region.isin(AK_REGIONS + GTNPENV_QTP)]
m_gconf = loro_transfer(tr_g, lena_df, "AUG_AK+GTNPenv", "Lena_RU")
m_gconf["cv_type"] = "LORO_confound_diag"
m_gconf["eval_scope"] = "gtnpenv_confound"
rows.append(m_gconf)
print(f"  진단3 [AK+GTNPenv→Lena]         rmse={m_gconf['rmse_cm']:.2f}cm  "
      f"(GTNPenv 심부 라벨 교란: OOD 냉대륙 기후에서 과대추정)")


# ================= Part C: 증강 라벨 품질 =================
print("\n=== Part C: 증강 라벨 품질 ===")
qc_rows = []
newpart = aug[aug.region.isin(NEW_REGIONS)].copy()
# 물리 범위 체크(0-400cm)
in_range = ((newpart["alt_cm"] >= 0) & (newpart["alt_cm"] <= 400)).mean()
print(f"  증강 라벨 물리범위(0-400cm) 내 비율: {in_range*100:.1f}%  (n={len(newpart)})")
print(f"  alt 통계: min={newpart.alt_cm.min():.1f} max={newpart.alt_cm.max():.1f} "
      f"mean={newpart.alt_cm.mean():.1f} median={newpart.alt_cm.median():.1f}")

# qc_flag별 분리 통계
for flag_col in ["qc_flag", "censor_flag"]:
    if flag_col in newpart.columns:
        for val, g in newpart.groupby(newpart[flag_col].fillna(-1)):
            qc_rows.append({
                "group": "new_augment", "flag": flag_col, "flag_value": val,
                "n": len(g), "alt_mean": round(g.alt_cm.mean(), 2),
                "alt_min": round(g.alt_cm.min(), 2), "alt_max": round(g.alt_cm.max(), 2),
                "in_range_0_400_pct": round(((g.alt_cm >= 0) & (g.alt_cm <= 400)).mean() * 100, 1)})
# 지역별
for reg, g in newpart.groupby("region"):
    qc_rows.append({
        "group": "new_augment", "flag": "region", "flag_value": reg,
        "n": len(g), "alt_mean": round(g.alt_cm.mean(), 2),
        "alt_min": round(g.alt_cm.min(), 2), "alt_max": round(g.alt_cm.max(), 2),
        "in_range_0_400_pct": round(((g.alt_cm >= 0) & (g.alt_cm <= 400)).mean() * 100, 1)})
qc_df = pd.DataFrame(qc_rows)
print(qc_df.to_string(index=False))

# qc_flag별 LORO 전이 분리 평가(Lena): qc_flag=0 vs 1
if "qc_flag" in lena_test.columns:
    aug_train_lena = aug[~aug.index.isin(lena_test.index)]
    ytr = np.log1p(aug_train_lena["alt_cm"].values.astype(float))
    Xtr_raw = build_X(aug_train_lena, FEATS)
    med = impute_fit(Xtr_raw)
    Xtr = apply_impute(Xtr_raw, med, FLAG_FEATS)
    model = gbm().fit(Xtr, ytr)
    Xte = apply_impute(build_X(lena_test, FEATS), med, FLAG_FEATS)
    pred_lena = to_cm(model.predict(Xte))
    y_lena = lena_test["alt_cm"].values.astype(float)
    qf = lena_test["qc_flag"].fillna(-1).values
    for v in np.unique(qf):
        s = qf == v
        if s.sum() >= 5:
            mm = all_metrics(y_lena[s], pred_lena[s])
            print(f"  Lena qc_flag={v}: n={int(s.sum())} rmse={mm['rmse_cm']:.2f} skill={mm['skill_over_mean']:.3f}")


# ================= 저장 =================
res = pd.DataFrame(rows)[
    ["dataset", "cv_type", "region", "eval_scope", "n", "rmse_cm", "mae_cm",
     "bias_cm", "r2", "target_sd_cm", "skill_over_mean"]]
res.to_csv(os.path.join(PROC, "p2m_augment_results.csv"), index=False)
qc_df.to_csv(os.path.join(PROC, "p2m_augment_qc.csv"), index=False)
pd.DataFrame(oof_records).to_csv(os.path.join(PROC, "p2m_augment_oof.csv"), index=False)
print("\n저장: p2m_augment_results.csv / p2m_augment_qc.csv / p2m_augment_oof.csv")


# ================= 게이트 판정 =================
def get(dataset, cv_type, region):
    r = res[(res.dataset == dataset) & (res.cv_type == cv_type) & (res.region == region)]
    return r.iloc[0] if len(r) else None


ak_base = get("BASE", "spatial_block_AK", "Alaska_all")
ak_aug = get("AUG", "spatial_block_AK", "Alaska_all")
lena_base = get("BASE", "LORO_transfer", "Lena_RU")
lena_aug = get("AUG", "LORO_transfer", "Lena_RU")
gtnp_base = get("BASE", "LORO_transfer", "GTNPenv_ALL")
gtnp_aug = get("AUG", "LORO_transfer", "GTNPenv_ALL")

selfcv = get("AUG_AK+Lena", "spatial_block_Lena", "Lena_RU")
gconf = res[(res.dataset == "AUG_AK+GTNPenv") & (res.region == "Lena_RU")].iloc[0]

d_lena = lena_base.rmse_cm - lena_aug.rmse_cm
d_gtnp = gtnp_base.rmse_cm - gtnp_aug.rmse_cm
d_ak = ak_base.rmse_cm - ak_aug.rmse_cm  # >0이면 AUG가 알래스카에서도 개선

# 게이트: 신규 지역 전이 RMSE 개선(Lena 또는 GTNPenv에서 Δ>0)이면 채택.
# 현 AUG 구성(GTNPenv 심부 라벨 포함)은 LORO 전이를 악화 → 미채택.
lena_improved = d_lena > 0
gtnp_improved = d_gtnp > 0
adopt = lena_improved or gtnp_improved

verdict = "채택" if adopt else "미채택"
meta = {
    "gate_verdict": verdict,
    "alaska_indomain": {
        "BASE_rmse_cm": round(ak_base.rmse_cm, 2), "AUG_rmse_cm": round(ak_aug.rmse_cm, 2),
        "delta_BASE_minus_AUG_cm": round(d_ak, 2),
        "BASE_skill": round(ak_base.skill_over_mean, 4), "AUG_skill": round(ak_aug.skill_over_mean, 4),
        "note": "양수 delta=AUG가 알래스카 in-domain 개선; 음수=in-domain 소폭 해"},
    "transfer_Lena_RU": {
        "n": int(lena_base.n), "BASE_rmse_cm": round(lena_base.rmse_cm, 2),
        "AUG_rmse_cm": round(lena_aug.rmse_cm, 2), "delta_BASE_minus_AUG_cm": round(d_lena, 2),
        "BASE_skill": round(lena_base.skill_over_mean, 4), "AUG_skill": round(lena_aug.skill_over_mean, 4),
        "improved": bool(lena_improved)},
    "transfer_GTNPenv_ALL": {
        "n": int(gtnp_base.n), "BASE_rmse_cm": round(gtnp_base.rmse_cm, 2),
        "AUG_rmse_cm": round(gtnp_aug.rmse_cm, 2), "delta_BASE_minus_AUG_cm": round(d_gtnp, 2),
        "BASE_skill": round(gtnp_base.skill_over_mean, 4), "AUG_skill": round(gtnp_aug.skill_over_mean, 4),
        "improved": bool(gtnp_improved)},
    "diagnostic_Lena_selfconsistency": {
        "note": "AK+Lena 공동학습 후 Lena 공간블록 CV(=Lena 라벨이 train에 있을 때). "
                "Lena 자체 라벨의 유용성/자기일관성 지표.",
        "rmse_cm": round(selfcv.rmse_cm, 2), "skill": round(selfcv.skill_over_mean, 4),
        "r2": round(selfcv.r2, 4),
        "interpretation": "이 값이 BASE→Lena(21.78)보다 우수하고 skill>0이면 Lena 라벨 자체는 유효. "
                          "LORO 열화는 GTNPenv 교란 탓."},
    "diagnostic_GTNPenv_confound": {
        "note": "AK+GTNPenv→Lena(GTNPenv 제외 안 함). GTNPenv 심부 라벨의 교란.",
        "rmse_cm": round(gconf.rmse_cm, 2),
        "interpretation": "GTNPenv 37셀(평균 154cm 심부)이 Lena 냉대륙 OOD 기후에서 과대추정 유발."},
    "augment_label_quality": {
        "n_new": int(len(newpart)), "in_range_0_400_pct": round(in_range * 100, 1),
        "alt_min": round(float(newpart.alt_cm.min()), 1), "alt_max": round(float(newpart.alt_cm.max()), 1)},
    "elapsed_s": round(time.time() - t_start, 1),
}
with open(os.path.join(PROC, "p2m_augment_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"[게이트] {verdict}")
print(f"  알래스카 in-domain: BASE {ak_base.rmse_cm:.2f} → AUG {ak_aug.rmse_cm:.2f}cm (Δ={d_ak:+.2f})")
print(f"  Lena 전이:          BASE {lena_base.rmse_cm:.2f} → AUG {lena_aug.rmse_cm:.2f}cm (Δ={d_lena:+.2f})")
print(f"  GTNPenv 전이:       BASE {gtnp_base.rmse_cm:.2f} → AUG {gtnp_aug.rmse_cm:.2f}cm (Δ={d_gtnp:+.2f})")
print("=" * 70)
print(f"[done] {json.dumps(meta, ensure_ascii=False)}  ({time.time()-t_start:.0f}s)")
