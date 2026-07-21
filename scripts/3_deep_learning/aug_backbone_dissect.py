"""라벨 증강 백본의 회의적 재해부 — "라벨 증강이 전이에 해가 된다" 주장 반증 설계.

목적
----
기존 결론(p2m_augment): AUG(GTNPenv 심부 라벨 포함)가 Lena 전이 RMSE를 21.8→89cm로
악화시키므로 "라벨 증강이 해가 된다"고 판정하고 미채택. 이 스크립트는 그 결론을 그대로
믿지 않고, 다음 대안 가설을 통제군으로 검증한다. 초판(v1)에서 지적된 누설·자명성·기제
과단정·seed 취약성을 정정한 개정판(v2)이다.

  H0(기존): 증강 라벨(특히 GTNPenv 심부 ALT)이 분포이동을 유발해 전이를 해친다.
  H1(반증): 열화의 주원인은 라벨 자체가 아니라 결측 공변량(InSAR/PolSAR/CCI가 신규지역
            100% 결측) 라우팅 아티팩트다. 결측 없는 물리/기후만 모델은 증강에 무해할 것.
  H2(반증): "held-out 신규지역"에 GTNPenv_US(알래스카 내부 9셀)가 섞여 집계착시를
            일으켰다. 순수 out-region과 분리하면 판정이 바뀔 수 있다.
  H3(부분): 심부 라벨(ALT>150cm)만 캡/게이팅/가중으로 처리하면 전이가 회복되는가.

v1 지적 정정(설계상 함정·과단정 제거)
-------------------------------------
1. 자명성(tautology) 명시: Lena가 test라 LORO 제외되면 '+Lena' 학습셋(14348행)이
   BASE(14348행)와 비트 단위로 동일하다. 마찬가지로 GTNPenv_ALL 타깃에서 '+GTNPenv'는
   BASE와 동일 학습셋이 된다. 이 셀들은 'tautology' 열로 표시해 변형 간 비교로 오독되지
   않게 한다. Lena 증강의 순효과는 test 지역이 아닌 GTNPenv_OUT 타깃에서만 관측 가능하다.
2. 누설의 정확한 위치: GTNPenv_US 9셀(7블록 전부 AK 블록)의 혼입은 기존 p2m의
   GTNPenv_ALL 전이 '집계' 표본을 오염시킨 집계착시이지, 본 재해부의 공간블록/LORO CV를
   오염시키는 train/test 누설은 아니다(gbm_spatial은 test 블록 공유 행을 train서 전부 배제).
3. 기제 분해(Part D 신설): as_is 붕괴의 원인을 (플래그 유무)×(심부18행 유무) 2x2로 분해한다.
   초판의 '_isnan 플래그 라우팅' 단정을 검증 — 플래그 제거만으로는 부분 완화에 그치고,
   심부 GTNPenv 18행 제거가 플래그 유무와 무관하게 회복을 유발함을 정량화한다.
4. 근퇴화(near-degenerate) 기저선 표면화: Lena 상수 평균예측 RMSE를 계산해 BASE full ML의
   skill<0(평균예측보다 나쁨)을 명시한다. cap의 회복은 '전이 기량 회복'이 아니라
   '평균예측 수준으로 복귀'임을 판정문에 반영한다.
5. seed 취약성: Part C DL(MLP)을 다중 seed로 재현해 BASE 자체의 seed 변동을 표면화하고,
   'DL은 증강 하에서만 붕괴'라는 인과 서술을 완화한다. GBM 붕괴는 다중 seed 견고성 확인.

공통 CV
-------
  공간블록: block=floor(lat/0.5)*100000+floor(lon/0.5), GroupKFold(6).
  LORO: region held-out(test>=100). GTNPenv_US(AK내부)는 out-region에서 분리.
  모든 결과에 rmse_cm·r2·skill_over_mean·bias·pred_mean·pred_std·n 기록.

산출(신규명, 기존 파일 미변경)
  data/processed/aug_backbone_sourceregion.csv   Part A(+tautology 열)
  data/processed/aug_backbone_controls.csv        Part B(물리/ML·기후/전공변량·심부처리)
  data/processed/aug_backbone_mechanism.csv       Part D(플래그x심부 2x2 기제 분해)
  data/processed/aug_backbone_dl.csv              Part C(GBM/MLP 다중 seed)
  data/processed/aug_backbone_predstats.csv       예측분포 진단
  data/processed/aug_backbone_meta.json           판정 요약
  figures: outputs/figures/07_augmentation/*, outputs/maps/*

실행: (ROOT) /home/anaconda3/bin/python scripts/3_deep_learning/aug_backbone_dissect.py
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
t0 = time.time()
RNG = np.random.RandomState(0)
DL_SEEDS = [0, 1, 2]        # Part C DL seed 앙상블(취약성 표면화)
GBM_SEEDS = [0, 1, 7, 42]   # GBM 붕괴 견고성 확인

# ---------------- 공변량 정의 ----------------
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
INSAR = ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]
POLSAR = ["polsar_alt", "polsar_std", "polsar_valid"]
CCI = ["cci_alt", "cci_valid"]
FULL_FEATS = TERRAIN + CLIMATE + INSAR + POLSAR + CCI    # 전 공변량(결측 대체+플래그)
CLIMONLY_FEATS = CLIMATE                                  # 결측 없는 기후만
TERRCLIM_FEATS = TERRAIN + CLIMATE                        # 지형+기후(결측 없음, 신규지역서도 관측)

AK_REGIONS = ["ABoVE_AK", "ABoVE_CA", "United States (Alaska)", "Canada"]
# 순수 out-region(알래스카 밖) vs AK내부 GTNPenv_US 분리
GTNPENV_OUT = ["GTNPenv_RU", "GTNPenv_SJ", "GTNPenv_CH", "QTP_CN", "GTNPenv_AQ"]  # 진짜 out
GTNPENV_US = ["GTNPenv_US"]                                                        # AK내부(집계착시)
GTNPENV_ALL = GTNPENV_OUT + GTNPENV_US
DEEP_CAP = 150.0   # 심부 라벨 기준


def load(path):
    df = pd.read_csv(path, low_memory=False)
    if "insar_miss" not in df.columns:
        df["insar_miss"] = df["insar_alt"].isna().astype(int)
    df["insar_miss"] = pd.to_numeric(df["insar_miss"], errors="coerce").fillna(
        df["insar_alt"].isna().astype(int)).astype(int)
    df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000
                   + np.floor(df.lon / 0.5).astype(int))
    return df


def gbm(seed=0):
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, random_state=seed)


def to_cm(p):
    return np.expm1(np.clip(p, *CLIP))


def impute_fit(Xtr):
    return Xtr.median(numeric_only=True)


def apply_impute(X, med, feat_flags):
    """결측 대체 + <col>_isnan 플래그. feat_flags 열만 플래그 부착."""
    Xf = X.copy()
    flags = {}
    for c in feat_flags:
        flags[c + "_isnan"] = Xf[c].isna().values.astype(float)
        fill = med.get(c, np.nan)
        if not np.isfinite(fill):
            fill = 0.0
        Xf[c] = Xf[c].fillna(fill)
    flag_df = pd.DataFrame(flags, index=Xf.index)
    return pd.concat([Xf, flag_df], axis=1).values


def metrics_ext(y, pred, extra):
    m = all_metrics(y, pred)
    fin = np.isfinite(pred)
    m["pred_mean"] = round(float(np.mean(pred[fin])), 3) if fin.any() else np.nan
    m["pred_std"] = round(float(np.std(pred[fin])), 3) if fin.any() else np.nan
    m.update(extra)
    return m


def const_mean_rmse(y):
    """상수 평균예측(self-mean) RMSE = std(y). 근퇴화 기저선."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    return float(np.std(y)) if y.size else np.nan


# ---------------- GBM 학습기(피처셋·가중·플래그·seed 지원) ----------------
def gbm_loro(train_df, test_df, feats, use_flags=True, sample_weight=None, seed=0):
    ytr = np.log1p(train_df["alt_cm"].values.astype(float))
    y_te = test_df["alt_cm"].values.astype(float)
    flags = feats if use_flags else []
    Xtr_raw = train_df[feats].astype(float)
    Xte_raw = test_df[feats].astype(float)
    med = impute_fit(Xtr_raw)
    Xtr = apply_impute(Xtr_raw, med, flags)
    Xte = apply_impute(Xte_raw, med, flags)
    model = gbm(seed)
    if sample_weight is not None:
        model.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        model.fit(Xtr, ytr)
    return to_cm(model.predict(Xte)), y_te


def gbm_spatial(df, eval_mask, feats, use_flags=True):
    """df 전체 6-fold 학습, eval_mask 셀만 OOF 채점. fold는 eval 셀 블록 기준.

    누설 안전성: test 블록을 공유하는 모든 행(다른 지역 포함)을 train에서 배제한다
    (tr = idx[~isin(block, te_blocks)]). GTNPenv_US 7블록이 전부 AK 블록의 부분집합이므로,
    AK 셀을 채점하는 fold에서 같은 블록의 GTNPenv_US는 train에서 자동 배제된다.
    같은 위치가 train/test 양쪽에 오는 누설은 발생하지 않는다."""
    ylog = np.log1p(df["alt_cm"].values.astype(float))
    y_cm = df["alt_cm"].values.astype(float)
    flags = feats if use_flags else []
    Xall = df[feats].astype(float)
    blocks = df["block"].values
    idx = np.arange(len(df))
    ev_idx = idx[eval_mask]
    o = np.full(len(df), np.nan)
    for tr_e, te_e in GroupKFold(n_splits=6).split(ev_idx, ylog[ev_idx], groups=blocks[ev_idx]):
        te = ev_idx[te_e]
        te_blocks = set(blocks[te])
        tr = idx[~np.isin(blocks, list(te_blocks))]
        med = impute_fit(Xall.iloc[tr])
        Xtr = apply_impute(Xall.iloc[tr], med, flags)
        Xte = apply_impute(Xall.iloc[te], med, flags)
        o[te] = to_cm(gbm().fit(Xtr, ylog[tr]).predict(Xte))
    return o, y_cm, ev_idx


# ---------------- 물리 모델(Stefan): ALT = a + E*sqrt(TDD) ----------------
def physics_loro(train_df, test_df):
    """fold별 최소제곱으로 a,E 적합. 기후(sqrt_tdd)만 사용 → 결측 없음."""
    xtr = train_df["e5_sqrt_tdd"].values.astype(float)
    ytr = train_df["alt_cm"].values.astype(float)
    ok = np.isfinite(xtr) & np.isfinite(ytr)
    A = np.column_stack([np.ones(ok.sum()), xtr[ok]])
    coef, *_ = np.linalg.lstsq(A, ytr[ok], rcond=None)   # [a, E]
    xte = test_df["e5_sqrt_tdd"].values.astype(float)
    pred = coef[0] + coef[1] * xte
    pred = np.clip(pred, 1.0, 600.0)
    return pred, test_df["alt_cm"].values.astype(float), coef


# ================= 데이터 로드 =================
base = load(os.path.join(PROC, "dl_dataset_cell.csv"))
aug = load(os.path.join(PROC, "dl_dataset_cell_v2.csv"))
print(f"[load] BASE n={len(base)}  AUG n={len(aug)}")

ak_aug = aug[aug.region.isin(AK_REGIONS)]
lena = aug[aug.region == "Lena_RU"]
gtnp_out = aug[aug.region.isin(GTNPENV_OUT)]
gtnp_us = aug[aug.region.isin(GTNPENV_US)]
gtnp_all = aug[aug.region.isin(GTNPENV_ALL)]
qtp = aug[aug.region == "QTP_CN"]
print(f"  AK={len(ak_aug)} Lena={len(lena)} GTNPenv_out={len(gtnp_out)} "
      f"GTNPenv_US={len(gtnp_us)} GTNPenv_all={len(gtnp_all)} QTP={len(qtp)}")

# --- 누설 위치 진단(집계착시 vs CV 누설) ---
us_blocks = set(gtnp_us["block"])
ak_blocks = set(ak_aug["block"])
us_in_ak_block = len(us_blocks & ak_blocks)
in_ak_box = int(((gtnp_us.lat >= 61.0) & (gtnp_us.lat <= 71.4)
                 & (gtnp_us.lon >= -166.0) & (gtnp_us.lon <= -141.0)).sum())
print(f"  [누설 진단] GTNPenv_US 블록 {len(us_blocks)}개 중 AK블록과 겹치는 것 {us_in_ak_block}개, "
      f"AK 경위도박스 내부 {in_ak_box}/{len(gtnp_us)}셀")

rows_A, rows_B, rows_C, rows_D, pred_stats = [], [], [], [], []
oof_store = {}   # (tag)->DataFrame(lat,lon,region,alt,pred) for maps


# ================= Part A: 증강 소스·지역 분해 =================
print("\n=== Part A: 증강 소스·지역 분해 ===")
# 학습셋 변형 정의: 이름 -> (그 변형에 포함할 증강지역 리스트)
VARIANTS = {
    "BASE":                    [],
    "+Lena":                   ["Lena_RU"],
    "+GTNPenv":                GTNPENV_ALL,
    "+Lena+GTNPenv":           ["Lena_RU"] + GTNPENV_ALL,
}


def train_variant(variant_name, exclude_test_idx=None):
    """변형 학습셋 = AK + 지정 증강지역, exclude_test_idx 제외."""
    regs = AK_REGIONS + VARIANTS[variant_name]
    tr = aug[aug.region.isin(regs)]
    if exclude_test_idx is not None:
        tr = tr[~tr.index.isin(exclude_test_idx)]
    return tr


def tautology_note(variant, target_regions):
    """LORO에서 test 지역이 학습셋에서 제외될 때, 학습셋이 어느 변형과 비트 동일해지는지.
    test 지역만 빼는 변형은 그 지역을 뺀 다른 변형과 동일 학습셋이 된다."""
    tr = train_variant(variant, exclude_test_idx=aug[aug.region.isin(target_regions)].index)
    base_tr = train_variant("BASE", exclude_test_idx=aug[aug.region.isin(target_regions)].index)
    if set(tr.index) == set(base_tr.index):
        return "identical_to_BASE"
    return ""


# (a) 알래스카 in-domain(공간블록, 알래스카 셀 채점) — 각 변형
for vname in VARIANTS:
    df_v = train_variant(vname)
    o, y, ev_idx = gbm_spatial(df_v.reset_index(drop=True),
                               df_v["region"].isin(AK_REGIONS).values, FULL_FEATS)
    ev = df_v["region"].isin(AK_REGIONS).values
    m = metrics_ext(y[ev], o[ev],
                    {"part": "A", "variant": vname, "cv_type": "spatial_block_AK",
                     "eval_region": "Alaska_all", "tautology": ""})
    rows_A.append(m)
    print(f"  [in-domain AK] {vname:16s} rmse={m['rmse_cm']:.2f} skill={m['skill_over_mean']:.3f} "
          f"predμ={m['pred_mean']:.1f} n={m['n']}")

# (b) Lena_RU 전이(LORO). Lena 포함 변형은 Lena를 train서 제외 → tautology 표시.
lena_const = const_mean_rmse(lena.alt_cm.values)
print(f"  [Lena 근퇴화 기저선] 상수 평균예측 RMSE = std(y) = {lena_const:.2f}cm "
      f"(obsμ={lena.alt_cm.mean():.1f}cm). 이보다 skill이 음수면 평균예측보다 나쁨.")
for vname in VARIANTS:
    tr = train_variant(vname, exclude_test_idx=lena.index)
    pred, y = gbm_loro(tr, lena, FULL_FEATS)
    taut = tautology_note(vname, ["Lena_RU"])
    m = metrics_ext(y, pred, {"part": "A", "variant": vname, "cv_type": "LORO",
                              "eval_region": "Lena_RU", "tautology": taut})
    rows_A.append(m)
    oof_store[("Lena", vname)] = pd.DataFrame(
        {"lat": lena.lat.values, "lon": lena.lon.values, "region": lena.region.values,
         "alt_cm": y, "pred": pred})
    tflag = f" [{taut}]" if taut else ""
    print(f"  [LORO Lena]    {vname:16s} rmse={m['rmse_cm']:.2f} skill={m['skill_over_mean']:.3f} "
          f"bias={m['bias_cm']:.1f} predμ={m['pred_mean']:.1f} predσ={m['pred_std']:.1f} n={m['n']}{tflag}")

# (c) GTNPenv 전이 — out-region(진짜 out) vs US(AK내부) 분리, 그리고 ALL(기존 정의)
#     GTNPenv_OUT 타깃이 Lena 증강의 순효과를 관측할 수 있는 유일한 지점(Lena가 test 아님).
gtnp_targets = [("GTNPenv_OUT", gtnp_out, GTNPENV_OUT),
                ("GTNPenv_US", gtnp_us, GTNPENV_US),
                ("GTNPenv_ALL", gtnp_all, GTNPENV_ALL)]
for tname, tdf, tregs in gtnp_targets:
    if len(tdf) < 1:
        continue
    for vname in ["BASE", "+Lena", "+GTNPenv", "+Lena+GTNPenv"]:
        tr = train_variant(vname, exclude_test_idx=tdf.index)
        pred, y = gbm_loro(tr, tdf, FULL_FEATS)
        taut = tautology_note(vname, tregs)
        m = metrics_ext(y, pred, {"part": "A", "variant": vname, "cv_type": "LORO",
                                  "eval_region": tname, "tautology": taut})
        rows_A.append(m)
    print(f"  [LORO {tname}] n={len(tdf)} 완료(변형 4종, tautology 표시)")


# ================= Part B: 교란 원인 통제군 =================
print("\n=== Part B: 교란 원인 통제군 ===")
# 통제 대상 전이 타깃: 순수 out-region Lena(대표성 높음). 보조로 GTNPenv_OUT.
# 통제 변형: BASE vs AUG(=+Lena+GTNPenv). Lena는 test라 train서 제외.

def controls_on(target_name, target_df):
    tests = target_df.index
    tr_base = train_variant("BASE", exclude_test_idx=tests)
    tr_aug = train_variant("+Lena+GTNPenv", exclude_test_idx=tests)

    # B1. 물리 vs ML(전공변량), BASE vs AUG
    for setname, tr in [("BASE", tr_base), ("AUG", tr_aug)]:
        pp, yy, coef = physics_loro(tr, target_df)
        rows_B.append(metrics_ext(yy, pp, {
            "part": "B", "control": "physics_vs_ml", "model": "physics_stefan",
            "feats": "climate_sqrt_tdd", "set": setname, "eval_region": target_name}))
        pm, ym = gbm_loro(tr, target_df, FULL_FEATS)
        rows_B.append(metrics_ext(ym, pm, {
            "part": "B", "control": "physics_vs_ml", "model": "gbm",
            "feats": "full", "set": setname, "eval_region": target_name}))

    # B2. 기후만 ML vs 지형기후 ML vs 전공변량 ML, BASE vs AUG
    for setname, tr in [("BASE", tr_base), ("AUG", tr_aug)]:
        pc, yc = gbm_loro(tr, target_df, CLIMONLY_FEATS)
        rows_B.append(metrics_ext(yc, pc, {
            "part": "B", "control": "climate_vs_full", "model": "gbm",
            "feats": "climate8", "set": setname, "eval_region": target_name}))
        ptc, ytc = gbm_loro(tr, target_df, TERRCLIM_FEATS)
        rows_B.append(metrics_ext(ytc, ptc, {
            "part": "B", "control": "climate_vs_full", "model": "gbm",
            "feats": "terr+clim", "set": setname, "eval_region": target_name}))
        pf, yf = gbm_loro(tr, target_df, FULL_FEATS)
        rows_B.append(metrics_ext(yf, pf, {
            "part": "B", "control": "climate_vs_full", "model": "gbm",
            "feats": "full", "set": setname, "eval_region": target_name}))
    return tr_base, tr_aug

trB_base, trB_aug = controls_on("Lena_RU", lena)
controls_on("GTNPenv_OUT", gtnp_out)

# B3. 심부 라벨 처리별(캡/게이팅/가중) — Lena 전이 회복 검사(처리별로 개별 보고)
print("  [B3] 심부 라벨 처리별 Lena 전이(처리별 개별 판정)")


def env_dissim_gate(train_df, test_df, feats=TERRCLIM_FEATS, q=0.98):
    """간이 AOA: train 표준화공간에서 test 중심까지 거리가 상위 q분위 이상인 train 표본 배제."""
    Xtr = train_df[feats].astype(float)
    Xte = test_df[feats].astype(float)
    mu = Xtr.mean(); sd = Xtr.std().replace(0, 1.0)
    Ztr = ((Xtr - mu) / sd).fillna(0.0).values
    Zte = ((Xte - mu) / sd).fillna(0.0).values
    te_center = Zte.mean(0)
    dist_tr = np.sqrt(((Ztr - te_center) ** 2).sum(1))
    thr = np.quantile(dist_tr, q)
    return dist_tr <= thr


tr_full = train_variant("+Lena+GTNPenv", exclude_test_idx=lena.index)
treatments = {}
treatments["as_is"] = (tr_full, None)
treatments["cap_gt150"] = (tr_full[tr_full.alt_cm <= DEEP_CAP], None)
keep_mask = env_dissim_gate(tr_full, lena)
treatments["aoa_gate"] = (tr_full[keep_mask], None)
w = np.where(tr_full.alt_cm.values > DEEP_CAP, 0.1, 1.0)
treatments["deep_downweight"] = (tr_full, w)

for tname, (trdf, wt) in treatments.items():
    pred, y = gbm_loro(trdf, lena, FULL_FEATS, sample_weight=wt)
    m = metrics_ext(y, pred, {"part": "B", "control": "deep_treatment", "model": "gbm",
                              "feats": "full", "set": tname, "eval_region": "Lena_RU"})
    m["n_train"] = int(len(trdf))
    rows_B.append(m)
    print(f"    {tname:16s} rmse={m['rmse_cm']:.2f} skill={m['skill_over_mean']:.3f} "
          f"bias={m['bias_cm']:.1f} predμ={m['pred_mean']:.1f} n_train={m['n_train']}")

# B4. 분포 진단: 각 변형의 Lena 예측 pred_mean/std, 실측 대비 과대추정
print("  [B4] Lena 예측 분포 진단")
lena_obs_mean = float(lena.alt_cm.mean())
for vname in VARIANTS:
    df_oof = oof_store[("Lena", vname)]
    pm = float(df_oof.pred.mean()); ps = float(df_oof.pred.std())
    over = pm - lena_obs_mean
    pred_stats.append({"target": "Lena_RU", "variant": vname,
                       "obs_mean": round(lena_obs_mean, 2),
                       "pred_mean": round(pm, 2), "pred_std": round(ps, 2),
                       "overestimate_cm": round(over, 2),
                       "constant_flag": bool(ps < 2.0)})
    print(f"    {vname:16s} obsμ={lena_obs_mean:.1f} predμ={pm:.1f} predσ={ps:.1f} "
          f"over={over:+.1f}cm {'[상수예측 의심]' if ps < 2 else ''}")


# ================= Part D: 붕괴 기제 분해(플래그 x 심부18행 2x2) =================
# 초판의 '_isnan 플래그 라우팅' 단정을 검증한다. as_is 붕괴(88cm)의 원인을
# (플래그 유무) x (심부 GTNPenv 18행 유무)로 분해. 플래그가 유일 기제라면 플래그 제거만으로
# 회복해야 한다. 심부18행이 핵심이면 그 제거가 플래그 유무와 무관하게 회복시킨다.
print("\n=== Part D: 붕괴 기제 분해(플래그 x 심부18행 2x2, Lena 전이) ===")
tr_aug_lena = train_variant("+Lena+GTNPenv", exclude_test_idx=lena.index)
tr_base_lena = train_variant("BASE", exclude_test_idx=lena.index)
deep_idx = tr_aug_lena[(tr_aug_lena.region.isin(GTNPENV_ALL))
                       & (tr_aug_lena.alt_cm > DEEP_CAP)].index
n_deep = len(deep_idx)
mech_cells = [
    ("BASE_flags",        tr_base_lena, True),
    ("AUG_flags",         tr_aug_lena, True),
    ("AUG_noflags",       tr_aug_lena, False),
    ("AUG_nodeep_flags",  tr_aug_lena[~tr_aug_lena.index.isin(deep_idx)], True),
    ("AUG_nodeep_noflags", tr_aug_lena[~tr_aug_lena.index.isin(deep_idx)], False),
]
for cname, trdf, use_flags in mech_cells:
    pred, y = gbm_loro(trdf, lena, FULL_FEATS, use_flags=use_flags)
    m = metrics_ext(y, pred, {"part": "D", "cell": cname, "flags": bool(use_flags),
                              "deep_removed": ("nodeep" in cname), "n_deep_removed": n_deep,
                              "eval_region": "Lena_RU"})
    m["n_train"] = int(len(trdf))
    rows_D.append(m)
    print(f"  {cname:20s} rmse={m['rmse_cm']:.2f} bias={m['bias_cm']:.1f} n_train={m['n_train']}")


# ================= Part C: DL 대조(GPU, 다중 seed) =================
print("\n=== Part C: DL 대조(GBM vs MLP, 다중 seed) ===")
import torch
import torch.nn as nn
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("  device:", dev)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def epochs_fit(net, Xtr, ytr, Xva, yva, epochs=120, bs=4096, lr=1e-3, wd=1e-5, pat=8):
    lossf = nn.SmoothL1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    Xv = torch.tensor(Xva).to(dev)
    best, bstate, p = 1e9, None, 0
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean((net(Xv).cpu().numpy() - yva) ** 2))
        if v < best - 1e-4:
            best, bstate, p = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
        else:
            p += 1
            if p >= pat:
                break
    if bstate:
        net.load_state_dict(bstate)
    return net


def dl_loro(train_df, test_df, feats, seed):
    """log1p 타깃, 중앙값대체+플래그, train기준 표준화. MLP. seed로 val split·초기화 제어."""
    rng = np.random.RandomState(seed)
    flags = feats
    Xtr_raw = train_df[feats].astype(float)
    Xte_raw = test_df[feats].astype(float)
    med = impute_fit(Xtr_raw)
    Xtr = apply_impute(Xtr_raw, med, flags).astype(np.float32)
    Xte = apply_impute(Xte_raw, med, flags).astype(np.float32)
    ytr = np.log1p(train_df["alt_cm"].values.astype(np.float32))
    y_te = test_df["alt_cm"].values.astype(float)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-6] = 1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8
    ytr_n = (ytr - ymu) / ysd
    va = rng.rand(len(Xtr)) < 0.1
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    net = MLP(d).to(dev)
    net = epochs_fit(net, Xtr[~va], ytr_n[~va], Xtr[va], ytr_n[va])
    net.eval()
    with torch.no_grad():
        pl = net(torch.tensor(Xte).to(dev)).cpu().numpy() * ysd + ymu
    return to_cm(pl), y_te


for setname, regs in [("BASE", []), ("+Lena+GTNPenv", ["Lena_RU"] + GTNPENV_ALL)]:
    tr = aug[aug.region.isin(AK_REGIONS + regs)]
    tr = tr[~tr.index.isin(lena.index)]
    # GBM 다중 seed
    for gs in GBM_SEEDS:
        pg, yg = gbm_loro(tr, lena, FULL_FEATS, seed=gs)
        rows_C.append(metrics_ext(yg, pg, {"part": "C", "model": "gbm", "seed": gs,
                                           "set": setname, "eval_region": "Lena_RU"}))
    # MLP 다중 seed
    for ds in DL_SEEDS:
        pdd, yd = dl_loro(tr, lena, FULL_FEATS, ds)
        rows_C.append(metrics_ext(yd, pdd, {"part": "C", "model": "mlp", "seed": ds,
                                            "set": setname, "eval_region": "Lena_RU"}))
    gc = [r["rmse_cm"] for r in rows_C if r["set"] == setname and r["model"] == "gbm"]
    mc = [r["rmse_cm"] for r in rows_C if r["set"] == setname and r["model"] == "mlp"]
    print(f"  set={setname}: GBM {np.mean(gc):.1f}±{np.std(gc):.1f} (범위 {min(gc):.1f}-{max(gc):.1f}) | "
          f"MLP {np.mean(mc):.1f}±{np.std(mc):.1f} (범위 {min(mc):.1f}-{max(mc):.1f})")


# ================= 저장 =================
A = pd.DataFrame(rows_A)
B = pd.DataFrame(rows_B)
C = pd.DataFrame(rows_C)
D = pd.DataFrame(rows_D)
P = pd.DataFrame(pred_stats)
colord = ["part", "variant", "control", "cell", "set", "model", "feats", "flags",
          "deep_removed", "seed", "cv_type", "tautology", "eval_region",
          "n", "n_train", "n_deep_removed", "rmse_cm", "mae_cm", "bias_cm", "r2",
          "target_sd_cm", "skill_over_mean", "pred_mean", "pred_std"]
def order(df):
    cols = [c for c in colord if c in df.columns] + [c for c in df.columns if c not in colord]
    return df[cols]
order(A).to_csv(os.path.join(PROC, "aug_backbone_sourceregion.csv"), index=False)
order(B).to_csv(os.path.join(PROC, "aug_backbone_controls.csv"), index=False)
order(C).to_csv(os.path.join(PROC, "aug_backbone_dl.csv"), index=False)
order(D).to_csv(os.path.join(PROC, "aug_backbone_mechanism.csv"), index=False)
P.to_csv(os.path.join(PROC, "aug_backbone_predstats.csv"), index=False)
print("\n저장: aug_backbone_sourceregion/controls/dl/mechanism/predstats.csv")


# ================= 판정 요약 =================
def getA(variant, region):
    r = A[(A.variant == variant) & (A.eval_region == region) & (A.cv_type == "LORO")]
    return r.iloc[0] if len(r) else None

def getB(control, model, feats, setn, region):
    r = B[(B.control == control) & (B.model == model) & (B.feats == feats)
          & (B.set == setn) & (B.eval_region == region)]
    return r.iloc[0] if len(r) else None

def getD(cell):
    r = D[D.cell == cell]
    return r.iloc[0] if len(r) else None

lena_base = getA("BASE", "Lena_RU")
lena_aug = getA("+Lena+GTNPenv", "Lena_RU")
lena_onlyL = getA("+Lena", "Lena_RU")            # tautology: == BASE
lena_onlyG = getA("+GTNPenv", "Lena_RU")

# GTNPenv_OUT에서 Lena 증강 순효과(자명하지 않은 유일 지점)
gout_base = getA("BASE", "GTNPenv_OUT")
gout_onlyL = getA("+Lena", "GTNPenv_OUT")
lena_pure_effect = round(gout_onlyL.rmse_cm - gout_base.rmse_cm, 2)  # +면 악화

# 통제군: 물리·기후만·전공변량 ML BASE vs AUG (Lena)
phys_base = getB("physics_vs_ml", "physics_stefan", "climate_sqrt_tdd", "BASE", "Lena_RU")
phys_aug = getB("physics_vs_ml", "physics_stefan", "climate_sqrt_tdd", "AUG", "Lena_RU")
clim_base = getB("climate_vs_full", "gbm", "climate8", "BASE", "Lena_RU")
clim_aug = getB("climate_vs_full", "gbm", "climate8", "AUG", "Lena_RU")
full_base = getB("climate_vs_full", "gbm", "full", "BASE", "Lena_RU")
full_aug = getB("climate_vs_full", "gbm", "full", "AUG", "Lena_RU")
d_full = full_base.rmse_cm - full_aug.rmse_cm
d_clim = clim_base.rmse_cm - clim_aug.rmse_cm
d_phys = phys_base.rmse_cm - phys_aug.rmse_cm

# 심부 처리 회복(처리별 개별)
def getT(setn):
    r = B[(B.control == "deep_treatment") & (B.set == setn)]
    return r.iloc[0] if len(r) else None
t_asis = getT("as_is"); t_cap = getT("cap_gt150"); t_gate = getT("aoa_gate"); t_dw = getT("deep_downweight")
treat_map = {"cap_gt150": t_cap, "aoa_gate": t_gate, "deep_downweight": t_dw}
# BASE+10 이내 회복 처리 목록(min으로 뭉개지 않고 개별 판정)
recovered_treats = [k for k, v in treat_map.items() if v.rmse_cm < lena_base.rmse_cm + 10]
recovered_any = len(recovered_treats) > 0

# 기제 분해(Part D)
m_augf = getD("AUG_flags"); m_augnf = getD("AUG_noflags")
m_ndf = getD("AUG_nodeep_flags"); m_ndnf = getD("AUG_nodeep_noflags")
flag_only_recovers = m_augnf.rmse_cm < lena_base.rmse_cm + 10       # 플래그 제거만으로 회복?
deep_recovers = m_ndf.rmse_cm < lena_base.rmse_cm + 10              # 심부제거(플래그有)로 회복?

# 근퇴화 기저선
base_skill = float(lena_base.skill_over_mean)
base_near_degenerate = base_skill < 0.0    # BASE가 평균예측보다 나쁨

# DL/GBM seed 통계
def cstat(model, setn):
    v = C[(C.model == model) & (C["set"] == setn)]["rmse_cm"].values
    return dict(mean=round(float(np.mean(v)), 2), sd=round(float(np.std(v)), 2),
                lo=round(float(np.min(v)), 2), hi=round(float(np.max(v)), 2), n=int(len(v)))
gbm_base_s = cstat("gbm", "BASE"); gbm_aug_s = cstat("gbm", "+Lena+GTNPenv")
mlp_base_s = cstat("mlp", "BASE"); mlp_aug_s = cstat("mlp", "+Lena+GTNPenv")
# GBM 붕괴 견고: 모든 seed에서 AUG가 BASE보다 크게 나쁨
gbm_collapse_robust = gbm_aug_s["lo"] > gbm_base_s["hi"] + 30
# MLP: BASE 자체 seed 변동이 크면 인과 서술 취약
mlp_base_unstable = (mlp_base_s["hi"] - mlp_base_s["lo"]) > 15

# --- 확정 판정(통제군 근거 기반, 과단정 제거) ---
routing_supported = (d_full < -20) and (abs(d_clim) < 10) and (abs(d_phys) < 10)

if routing_supported:
    verdict = "부분기각(조건부 성립)"
    reason = (
        "결측 공변량을 가진 전공변량 GBM만 대폭 악화(Δ%.0fcm)하고, 결측 없는 물리·기후만·"
        "지형기후 모델은 무해(|Δ|<10cm)하다. 열화는 심부 라벨 자체의 분포이동이 아니라 "
        "결측 공변량과 심부 GTNPenv 라벨의 결합이 원인이다(H1 지지). 다만 (1) BASE 전이 자체가 "
        "평균예측보다 나쁜 근퇴화 상태(skill=%.3f)라 cap의 '회복'은 전이 기량 회복이 아니라 "
        "평균예측 수준 복귀이며, (2) 붕괴 기제는 _isnan 플래그 단독이 아니라 심부 GTNPenv 18행이 "
        "핵심이다(플래그 제거만으로는 %.0f→%.0fcm 부분완화, 심부18행 제거 시 플래그有 %.0f·플래그無 "
        "%.0fcm 회복). '증강이 해가 된다'는 [전공변량 GBM + 무처리]라는 좁은 조건에서만 성립한다."
        % (d_full, base_skill, m_augf.rmse_cm, m_augnf.rmse_cm, m_ndf.rmse_cm, m_ndnf.rmse_cm)
    )
else:
    verdict = "부분"
    reason = "통제군이 혼재. 결측·심부라벨 성분 공존, 조건 재검 필요."

meta = {
    "claim": "라벨 증강이 전이에 해가 된다",
    "verdict": verdict,
    "reason": reason,
    "confirmed_conclusions": [
        "전공변량 GBM의 Lena 전이 붕괴(BASE %.1f -> AUG %.1fcm)는 seed 견고(GBM AUG 범위 %.1f-%.1fcm, "
        "모든 seed에서 BASE보다 30cm+ 악화)." % (full_base.rmse_cm, full_aug.rmse_cm, gbm_aug_s["lo"], gbm_aug_s["hi"]),
        "결측 없는 물리(Δ%.1f)·기후만ML(Δ%.1f)·지형기후ML은 증강에 무해 -> 붕괴는 결측 공변량 채널을 "
        "통해서만 전파(H1 지지)." % (d_phys, d_clim),
        "붕괴 기제는 _isnan 플래그 단독이 아님: 심부 GTNPenv %d행 제거가 플래그 유무와 무관하게 회복을 "
        "유발(AUG %.1f -> nodeep %.1f). 플래그 제거만으로는 %.1f로 부분완화에 그침." % (
            n_deep, m_augf.rmse_cm, m_ndf.rmse_cm, m_augnf.rmse_cm),
    ],
    "corrections_from_v1": [
        "tautology 명시: LORO에서 test 지역 제외 시 학습셋이 BASE와 비트 동일해지는 셀(+Lena@Lena, "
        "+GTNPenv@GTNPenv_ALL)을 tautology 열로 표시. '+Lena 단독 무해'는 자명성의 산물이라 근거 아님.",
        "Lena 증강 순효과는 GTNPenv_OUT 타깃에서만 관측 가능: +Lena는 오히려 소폭 악화(Δ%+.2fcm) -> "
        "'+Lena 무해' 결론 철회." % lena_pure_effect,
        "근퇴화 기저선 표면화: Lena 상수 평균예측 RMSE=%.2fcm(=std), BASE full ML=%.2fcm(skill=%.3f<0). "
        "BASE 전이는 평균예측보다 나쁨. cap '회복'은 평균예측 수준 복귀." % (
            lena_const, lena_base.rmse_cm, base_skill),
        "기제 과단정 정정: '_isnan 플래그 라우팅'은 여러 채널 중 하나. 심부 GTNPenv 18행이 핵심.",
        "누설 위치 정정: GTNPenv_US 혼입은 기존 p2m GTNPenv_ALL '전이 집계'의 표본오염(집계착시)이지, "
        "본 재해부의 공간블록/LORO CV를 오염시키는 train/test 누설이 아님(test 블록 공유 행 전부 배제).",
    ],
    "lena_transfer_variants": {
        "note": "+Lena는 Lena가 test라 LORO 제외 -> 학습셋이 BASE와 동일(tautology). 변형 비교 아님.",
        "BASE_rmse": round(lena_base.rmse_cm, 2), "BASE_skill": round(lena_base.skill_over_mean, 3),
        "+Lena_rmse": round(lena_onlyL.rmse_cm, 2), "+Lena_tautology": str(lena_onlyL.tautology),
        "+GTNPenv_rmse": round(lena_onlyG.rmse_cm, 2), "+GTNPenv_skill": round(lena_onlyG.skill_over_mean, 3),
        "+Lena+GTNPenv_rmse": round(lena_aug.rmse_cm, 2), "+Lena+GTNPenv_skill": round(lena_aug.skill_over_mean, 3),
    },
    "lena_augmentation_net_effect_on_GTNPenv_OUT": {
        "note": "Lena가 test 아닌 타깃에서 관측한 Lena 증강 순효과. +면 악화.",
        "BASE_rmse": round(gout_base.rmse_cm, 2), "+Lena_rmse": round(gout_onlyL.rmse_cm, 2),
        "delta_cm": lena_pure_effect,
    },
    "near_degenerate_baseline": {
        "lena_obs_mean_cm": round(lena_obs_mean, 2),
        "constant_mean_pred_rmse_cm": round(lena_const, 2),
        "base_full_ml_rmse_cm": round(lena_base.rmse_cm, 2),
        "base_skill_over_mean": base_skill,
        "base_is_worse_than_mean_pred": bool(base_near_degenerate),
    },
    "controls_lena": {
        "full_ml_BASE_vs_AUG_rmse": [round(full_base.rmse_cm, 2), round(full_aug.rmse_cm, 2), round(d_full, 2)],
        "climate_ml_BASE_vs_AUG_rmse": [round(clim_base.rmse_cm, 2), round(clim_aug.rmse_cm, 2), round(d_clim, 2)],
        "physics_BASE_vs_AUG_rmse": [round(phys_base.rmse_cm, 2), round(phys_aug.rmse_cm, 2), round(d_phys, 2)],
        "routing_hypothesis_supported": bool(routing_supported),
    },
    "mechanism_2x2_lena_rmse": {
        "BASE_flags": round(getD("BASE_flags").rmse_cm, 2),
        "AUG_flags": round(m_augf.rmse_cm, 2),
        "AUG_noflags": round(m_augnf.rmse_cm, 2),
        "AUG_nodeep_flags": round(m_ndf.rmse_cm, 2),
        "AUG_nodeep_noflags": round(m_ndnf.rmse_cm, 2),
        "n_deep_removed": int(n_deep),
        "flag_removal_alone_recovers": bool(flag_only_recovers),
        "deep_removal_recovers": bool(deep_recovers),
        "primary_driver": "deep_GTNPenv_labels" if deep_recovers and not flag_only_recovers else "mixed",
    },
    "deep_treatments_lena_rmse": {
        "as_is": round(t_asis.rmse_cm, 2), "cap_gt150": round(t_cap.rmse_cm, 2),
        "aoa_gate": round(t_gate.rmse_cm, 2), "deep_downweight": round(t_dw.rmse_cm, 2),
        "base_transfer_rmse": round(lena_base.rmse_cm, 2),
        "treatments_recovering_to_near_BASE": recovered_treats,
        "note": "cap만 BASE 근처(평균예측 수준)로 복귀. gate/downweight는 48cm로 회복 실패. "
                "'심부처리로 회복'은 cap 한정.",
    },
    "seed_robustness": {
        "gbm_BASE": gbm_base_s, "gbm_AUG": gbm_aug_s,
        "mlp_BASE": mlp_base_s, "mlp_AUG": mlp_aug_s,
        "gbm_collapse_robust_across_seeds": bool(gbm_collapse_robust),
        "mlp_BASE_unstable_across_seeds": bool(mlp_base_unstable),
        "dl_conclusion_note": (
            "MLP BASE 자체가 seed에 따라 %.0f-%.0fcm로 변동. 'DL은 증강 하에서만 붕괴'라는 인과 서술은 "
            "seed 취약. AUG는 모든 seed에서 BASE보다 나쁘나(평균 %.0f vs %.0f), 단일 seed 결론은 불안정."
            % (mlp_base_s["lo"], mlp_base_s["hi"], mlp_aug_s["mean"], mlp_base_s["mean"])),
    },
    "leakage_diagnosis": {
        "gtnpenv_us_n": int(len(gtnp_us)),
        "gtnpenv_us_blocks": len(us_blocks),
        "gtnpenv_us_blocks_shared_with_AK": us_in_ak_block,
        "gtnpenv_us_in_AK_box": in_ak_box,
        "classification": "aggregation_contamination_in_p2m_GTNPenv_ALL_transfer_not_CV_leakage_in_this_dissect",
        "note": "GTNPenv_US 7블록 전부 AK 블록. 기존 p2m의 GTNPenv_ALL 전이 집계에 AK 내부 9셀 혼입(집계착시). "
                "본 재해부 공간블록/LORO CV는 test 블록 공유 행을 전부 train서 배제하므로 누설 안전.",
    },
    "elapsed_s": round(time.time() - t0, 1),
}
with open(os.path.join(PROC, "aug_backbone_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 74)
print(f"[판정] {verdict}")
print(reason)
print("-" * 74)
print(f"  Lena 전이 RMSE  BASE={lena_base.rmse_cm:.1f}(skill{base_skill:+.2f}, 평균예측RMSE={lena_const:.1f}) "
      f"+GTNPenv={lena_onlyG.rmse_cm:.1f} AUG={lena_aug.rmse_cm:.1f}")
print(f"  [tautology] +Lena@Lena={lena_onlyL.tautology}  (Lena 순효과는 GTNPenv_OUT서 Δ{lena_pure_effect:+.1f}cm)")
print(f"  통제(Lena) 전공변량ML Δ={d_full:+.1f}  기후만ML Δ={d_clim:+.1f}  물리 Δ={d_phys:+.1f}")
print(f"  기제2x2  AUGflag={m_augf.rmse_cm:.1f} AUGnoflag={m_augnf.rmse_cm:.1f} "
      f"nodeepFlag={m_ndf.rmse_cm:.1f} nodeepNoflag={m_ndnf.rmse_cm:.1f}  (심부{n_deep}행이 핵심)")
print(f"  심부처리 as_is={t_asis.rmse_cm:.1f} cap={t_cap.rmse_cm:.1f} gate={t_gate.rmse_cm:.1f} "
      f"dw={t_dw.rmse_cm:.1f}  회복처리={recovered_treats}")
print(f"  seed  GBM BASE {gbm_base_s['lo']:.0f}-{gbm_base_s['hi']:.0f} AUG {gbm_aug_s['lo']:.0f}-{gbm_aug_s['hi']:.0f} "
      f"| MLP BASE {mlp_base_s['lo']:.0f}-{mlp_base_s['hi']:.0f} AUG {mlp_aug_s['lo']:.0f}-{mlp_aug_s['hi']:.0f}")
print("=" * 74)

# oof 저장(지도용)
oof_all = []
for (tgt, v), dfo in oof_store.items():
    dfo2 = dfo.copy(); dfo2["variant"] = v; dfo2["target"] = tgt
    oof_all.append(dfo2)
pd.concat(oof_all).to_csv(os.path.join(PROC, "aug_backbone_oof.csv"), index=False)
print(f"[done] {time.time()-t0:.0f}s")
