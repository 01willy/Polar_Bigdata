"""트랙 3 — 얕은 3D 지중온도장(알래스카 0-3m)·0°C→ALT 정합 검증.

목적
- Phase 1 심부판(스위스 편중 외삽, 0°C→ALT r=0.16·RMSE 172cm 귀무)의 한계를 피해,
  검증 가능한 얕은 0-3m 대역으로 한정한 알래스카 지중 연최대온도 t_max(x, depth)를
  GBM으로 산출하고, 실측 시추공 t_max로 held-out 검증한다. 산출물은 완전한 공변량을 가진
  이산 알래스카 셀의 4개 깊이 산점 예측이며 연속 볼륨/조밀 '온도장'이 아니다.
- 0°C→ALT 정합: 예측 t_max 프로파일의 0°C 교차로 ALT를 유도해, 알래스카 실측
  ALT(시추공 envelope ALT)와 비교한다. 심부판과의 RMSE 절대비교는 대상범위가 달라 성립하지
  않으므로(사과-오렌지), R²·skill로 정직 판정한다.

주의(정직성)
- 온도 예측 자체는 held-out 검증 성립(t_max site-block R²≈0.47). 그러나 유도 ALT 정합은
  자기평균 대비 skill 음수(R²<0)로 성립하지 않는다.
- 유도 ALT는 obs·pred 모두 최심 관측깊이(<=3m)로 상한 절단되어 평가가 얕은 대역에 편향된다.
- 단조성 위반율 mono=0.000은 제약의 자명한 귀결(구조적 항등)이며 물리 사실성의 독립 증거가 아니다.

회의적 원칙(최상위)
- 라벨=시추공 깊이별 연최대온도 t_max. 훈련·평가 모두 실측만 사용(유사라벨 없음).
- 같은 시추공/사이트의 다른 깊이 행이 train·test에 동시에 들지 않도록 사이트 단위로 블록.
  · 공간블록 6-fold: 사이트(site) 단위 GroupKFold(같은 사이트=같은 fold).
  · LORO(leave-one-site-out): 각 사이트를 완전히 제외하고 재적합한 뒤 그 사이트만 예측.
    6-fold OOF의 사이트별 집계와는 다르다(6-fold는 한 fold에 여러 사이트가 묶여
    co-fold 사이트도 함께 훈련에서 빠지므로 진짜 LORO가 아니다). 이 스크립트는
    진짜 LORO를 별도 재적합으로 계산한다.
- RMSE 옆 R²·skill·bias 병기. 깊이밴드별 별도 집계.
- 0°C→ALT 검증은 held-out(공간블록 OOF) 예측 포락선으로만 유도. 훈련 라벨 재사용 금지.
  단, obs·pred ALT 모두 동일한 이산 관측깊이(<=3m) 격자에서 선형보간되므로 유도 ALT의
  상한이 시추공 최심 관측깊이로 사실상 절단된다. 이는 누설은 아니나 평가를 얕은 대역에
  유리하게 편향시키는 구조적 한계다(심부 ALT를 가진 시추공은 과소평가된다).

데이터
- 라벨: ground_temp_gtnp_global.csv, 알래스카 bbox(lat 54-72, lon -170~-129), depth 0-3m,
  t_max(깊이별 연최대 월기후 온도). depth 0은 지표(공기·복사 영향)라 0.02m 이상만 사용
  (지표 극단 진폭 제거). 사이트당 깊이 >=3 인 시추공만.
- 공변량: 기후8(e5_*)+지형6(dem_*)+토양9(sg_*) — 각 시추공을 최근접 알래스카 ALT 셀
  (dl_dataset_cell_v3_soil.csv, region ABoVE_AK·United States (Alaska))에서 취득
  (최근접 거리 중앙값 1km, 최대 35km). depth_m + 푸리에(연 진폭 감쇠 형태 표현):
  depth_m, exp(-depth/d0) 유형이 아닌 순수 데이터구동 GBM이므로 depth·sqrt(depth)·
  1/(1+depth) 및 √TDD·depth 상호작용을 추가해 감쇠 곡률을 학습 가능케 한다.

모델
- HistGradientBoostingRegressor(단조 제약: depth 증가 시 t_max 비증가 monotonic_cst=-1).
  물리(깊을수록 연최대온도 감소)를 약하게 강제. 대조로 무제약 GBM도 학습해 단조성 위반율 비교.

산출
- data/processed/shallow3d_alaska_results.csv  (CV×깊이밴드 지표 + ALT정합 요약행)
- data/processed/shallow3d_alaska_altmatch.csv (시추공별 유도 ALT vs 실측 ALT)
- data/processed/shallow3d_alaska_grid.csv     (깊이 0.5·1·2·3m 알래스카 격자 예측)
- data/processed/shallow3d_alaska_meta.json
- 그림: outputs/figures/14_shallow3d/ + outputs/maps/

실행: /home/anaconda3/bin/python scripts/3_deep_learning/shallow3d_alaska.py  (ROOT에서, CPU)
"""
import sys, os, json, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from polar.eval_metrics import all_metrics

PROC = "data/processed"
GT = os.path.join(PROC, "ground_temp_gtnp_global.csv")
CELL = os.path.join(PROC, "dl_dataset_cell_v3_soil.csv")
AK_BBOX = dict(lat_lo=54.0, lat_hi=72.0, lon_lo=-170.0, lon_hi=-129.0)
TOP_M = 3.0
MIN_DEPTH = 0.02   # 지표 극단(0m) 제외
N_FOLD = 6
RNG = np.random.default_rng(0)
t0 = time.time()

CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
SOIL = ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15", "sg_cfvo_5_15",
        "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15", "sg_soc_15_30"]
COV = CLIMATE + TERRAIN + SOIL

# ---------- 1) 알래스카 얕은 시추공 라벨 ----------
gt = pd.read_csv(GT)
ak = gt[(gt.lat >= AK_BBOX["lat_lo"]) & (gt.lat <= AK_BBOX["lat_hi"]) &
        (gt.lon >= AK_BBOX["lon_lo"]) & (gt.lon <= AK_BBOX["lon_hi"])].copy()
ak = ak.dropna(subset=["depth", "t_max"])
ak = ak[(ak.depth >= MIN_DEPTH) & (ak.depth <= TOP_M)].copy()
# 사이트당 깊이 >=3
bd = ak.groupby("borehole_id").depth.nunique()
keep_bh = bd[bd >= 3].index
ak = ak[ak.borehole_id.isin(keep_bh)].copy()
ak["site_key"] = ak["site"].astype(str).str.strip()
print(f"[label] 알래스카 얕은 라벨 행={len(ak)}  시추공={ak.borehole_id.nunique()}  사이트={ak.site_key.nunique()}")

# ---------- 2) 공변량: 최근접 ALT 셀에서 취득 ----------
cell = pd.read_csv(CELL, low_memory=False)
akc = cell[cell.region.isin(["ABoVE_AK", "United States (Alaska)"])].copy()
for c in COV + ["lat", "lon"]:
    akc[c] = pd.to_numeric(akc[c], errors="coerce")
akc = akc.dropna(subset=COV).reset_index(drop=True)
clat, clon = akc.lat.values, akc.lon.values

def nearest_cov(la, lo):
    dlat = (clat - la) * 111.0
    dlon = (clon - lo) * 111.0 * np.cos(np.radians(la))
    d = dlat ** 2 + dlon ** 2
    i = int(np.argmin(d))
    return i, float(np.sqrt(d[i]))

bh = ak.groupby("borehole_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                   site_key=("site_key", "first")).reset_index()
covrows = {}
maxd = 0.0
for r in bh.itertuples():
    i, dkm = nearest_cov(r.lat, r.lon)
    covrows[r.borehole_id] = akc.loc[i, COV].to_dict()
    maxd = max(maxd, dkm)
covdf = pd.DataFrame(covrows).T
covdf.index.name = "borehole_id"
covdf = covdf.reset_index()
print(f"[cov] 시추공별 최근접 셀 공변량 취득  최대거리={maxd:.1f}km")

ak = ak.merge(covdf, on="borehole_id", how="left")

# ---------- 3) 깊이 피처(푸리에·감쇠 형태) ----------
def depth_feats(d):
    d = np.asarray(d, float)
    return pd.DataFrame({
        "depth_m": d,
        "sqrt_depth": np.sqrt(d),
        "inv_depth": 1.0 / (1.0 + d),
        "exp_depth": np.exp(-d),               # 진폭 감쇠 형태
    })

dfeat = depth_feats(ak.depth.values)
for c in dfeat.columns:
    ak[c] = dfeat[c].values
# √TDD·depth 상호작용(연 진폭이 √TDD 스케일·깊이 감쇠)
ak["sqrttdd_x_expdepth"] = np.sqrt(np.clip(ak["e5_tdd"].values, 0, None)) * ak["exp_depth"].values

DEPTH_FEATS = ["depth_m", "sqrt_depth", "inv_depth", "exp_depth", "sqrttdd_x_expdepth"]
FEATS = COV + DEPTH_FEATS
y = ak["t_max"].values.astype(float)
X = ak[FEATS].values.astype(float)
sites = ak["site_key"].values
print(f"[feats] n_feat={len(FEATS)}  라벨 t_max 범위=[{y.min():.1f}, {y.max():.1f}]°C")

# 단조 제약: depth_m·sqrt_depth 증가 시 t_max 비증가(-1), exp/inv_depth는 depth와 역방향이라 +1,
# 나머지 공변량 무제약(0). GBM monotonic_cst.
def mono_cst(feat_names):
    # 모든 깊이 파생 피처는 시추공 내 depth 오름차순에서 t_max 감소(연 진폭 감쇠)를 표현해야 한다.
    #  depth_m·sqrt_depth : depth와 증가 → t_max와 비증가(-1)
    #  inv_depth·exp_depth·sqrttdd_x_expdepth : depth와 감소 → t_max와 비감소(+1)로 부호 반전
    # 상호작용까지 제약해야 시추공 내 프로파일 감소 단조가 보장된다.
    dec = {"depth_m", "sqrt_depth"}
    inc = {"inv_depth", "exp_depth", "sqrttdd_x_expdepth"}
    cst = []
    for f in feat_names:
        if f in dec:
            cst.append(-1)
        elif f in inc:
            cst.append(1)
        else:
            cst.append(0)
    return cst

def make_gbm(monotone=True):
    kw = dict(max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
              min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    if monotone:
        kw["monotonic_cst"] = mono_cst(FEATS)
    return HistGradientBoostingRegressor(**kw)

# ---------- 4) 공간블록 6-fold(사이트 단위) OOF ----------
# 사이트 수가 fold보다 많아야 함. GroupKFold(사이트)로 같은 사이트=같은 fold.
uniq_sites = np.array(sorted(np.unique(sites)))
n_sites = len(uniq_sites)
nfold = min(N_FOLD, n_sites)
gkf = GroupKFold(n_splits=nfold)

def run_oof(monotone):
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups=sites):
        m = make_gbm(monotone)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof

oof_mono = run_oof(True)
oof_free = run_oof(False)
print(f"[cv] 사이트블록 {nfold}-fold OOF 완료 (사이트 {n_sites}개)")

# ---------- 5) 깊이밴드별 지표 ----------
# 주의: all_metrics는 컬럼을 rmse_cm 등으로 고정 명명하나, 여기 t_max 밴드행의 대상은
# 온도(°C)다. 따라서 밴드행에는 unit="C"를, ALT정합 행에는 unit="cm"를 명시해 단위
# 오독을 막는다(컬럼값 rmse_cm=2.66은 실제 2.66°C).
BANDS = [(0.02, 0.5, "0.02-0.5m"), (0.5, 1.0, "0.5-1m"),
         (1.0, 2.0, "1-2m"), (2.0, 3.0, "2-3m")]
depth = ak.depth.values
res_rows = []
for pred, tag in [(oof_mono, "gbm_mono"), (oof_free, "gbm_free")]:
    m_all = all_metrics(y, pred)
    res_rows.append(dict(model=tag, cv="site_block6", band="all", unit="C",
                         depth_lo=MIN_DEPTH, depth_hi=TOP_M, **m_all))
    for lo, hi, name in BANDS:
        mask = (depth >= lo) & (depth < hi if hi < TOP_M else depth <= hi)
        if mask.sum() < 5:
            continue
        m = all_metrics(y[mask], pred[mask])
        res_rows.append(dict(model=tag, cv="site_block6", band=name, unit="C",
                             depth_lo=lo, depth_hi=hi, **m))

# ---------- 6) LORO(사이트별) — 진짜 leave-one-site-out 재적합 ----------
# 각 사이트를 완전히 제외하고 재적합한 뒤 그 사이트만 예측한다. 6-fold OOF의 사이트별
# 집계와는 다르다(6-fold는 한 fold에 여러 사이트가 묶여 co-fold 사이트도 훈련에서 함께
# 빠지므로 진짜 LORO가 아니다). 두 방식을 모두 저장해 차이를 투명하게 남긴다.
loro_pred = np.full(len(y), np.nan)
for s in uniq_sites:
    te = sites == s
    m = make_gbm(True)
    m.fit(X[~te], y[~te])
    loro_pred[te] = m.predict(X[te])

loro_rows = []
for s in uniq_sites:
    mask = sites == s
    if mask.sum() < 5:
        continue
    m_loro = all_metrics(y[mask], loro_pred[mask])   # 진짜 LORO
    m_oof = all_metrics(y[mask], oof_mono[mask])      # 참고: 6-fold OOF 사이트별 집계
    loro_rows.append(dict(model="gbm_mono", cv="LORO_site", site=s, **m_loro,
                          r2_oof6=m_oof["r2"], rmse_oof6_cm=m_oof["rmse_cm"]))
loro_df = pd.DataFrame(loro_rows)
n_pos_loro = int((loro_df["r2"] > 0).sum())
n_pos_oof6 = int((loro_df["r2_oof6"] > 0).sum())
print(f"[loro] 진짜 LORO 재적합 완료. 양의 skill 사이트 LORO={n_pos_loro}/{len(loro_df)}  "
      f"(참고 6-fold OOF집계={n_pos_oof6}/{len(loro_df)})")

# ---------- 7) 물리 일관성: 단조성 위반율 ----------
def viol_rate(pred_fn, monotone_tag):
    """각 시추공에서 depth 오름차순 예측 프로파일이 감소 단조를 어기는 인접쌍 비율."""
    tot, bad = 0, 0
    m = make_gbm(monotone_tag)
    m.fit(X, y)  # 전 데이터 적합(위반율은 형태 진단, 검증 아님)
    for bid, g in ak.groupby("borehole_id"):
        gg = g.sort_values("depth")
        Xg = gg[FEATS].values.astype(float)
        p = m.predict(Xg)
        if len(p) < 2:
            continue
        d = np.diff(p)  # depth 증가 시 예측 변화
        tot += len(d)
        bad += int(np.sum(d > 1e-6))  # 증가(위반)
    return bad / tot if tot else np.nan

viol_mono = viol_rate(pred_fn=None, monotone_tag=True)
viol_free = viol_rate(pred_fn=None, monotone_tag=False)
print(f"[phys] 단조성 위반율  단조제약={viol_mono:.3f}  무제약={viol_free:.3f}")

# ---------- 8) 0°C→ALT 정합(held-out OOF 포락선 교차) ----------
def crossing_alt(dvals, tvals):
    """depth 오름차순 (d,t)에서 가장 깊은 +→- 교차를 선형보간(m). 없으면 NaN."""
    order = np.argsort(dvals)
    d, t = dvals[order], tvals[order]
    alt = np.nan
    for i in range(len(d) - 1):
        if t[i] > 0 and t[i + 1] <= 0:
            alt = d[i] + (d[i + 1] - d[i]) * t[i] / (t[i] - t[i + 1])
    return alt

alt_rows = []
for bid, g in ak.groupby("borehole_id"):
    gg = g.sort_values("depth")
    idx = gg.index.values
    d = gg.depth.values
    t_obs = gg.t_max.values
    t_pred = pd.Series(oof_mono, index=ak.index).loc[idx].values
    alt_obs = crossing_alt(d, t_obs)
    alt_pred = crossing_alt(d, t_pred)
    if np.isnan(alt_obs):
        continue
    # 유도 ALT는 최심 관측깊이(<=3m)로 상한이 절단된다. obs ALT가 최심 관측깊이의 90%를
    # 넘으면 실제 ALT가 관측격자 밖일 개연성이 높아 절단 편향 후보로 표시한다.
    dmax = float(d.max())
    truncated = int(alt_obs >= 0.9 * dmax)
    alt_rows.append(dict(borehole_id=int(bid), site=str(gg.site_key.iloc[0]),
                         lat=float(gg.lat.iloc[0]), lon=float(gg.lon.iloc[0]),
                         alt_obs_cm=round(alt_obs * 100, 1),
                         alt_pred_cm=(round(alt_pred * 100, 1) if not np.isnan(alt_pred) else np.nan),
                         n_depth=int(len(d)), max_obs_depth_cm=round(dmax * 100, 1),
                         alt_obs_near_maxdepth=truncated))
altm = pd.DataFrame(alt_rows)
both = altm.dropna(subset=["alt_pred_cm"])


def _altmatch_metrics(sub):
    if len(sub) < 3:
        return dict(n=len(sub), r=np.nan, rmse_cm=np.nan, r2=np.nan, bias_cm=np.nan,
                    target_sd_cm=np.nan)
    m = all_metrics(sub.alt_obs_cm.values, sub.alt_pred_cm.values)
    return dict(n=len(sub),
                r=float(np.corrcoef(sub.alt_obs_cm, sub.alt_pred_cm)[0, 1]),
                rmse_cm=m["rmse_cm"], r2=m["r2"], bias_cm=m["bias_cm"],
                target_sd_cm=round(float(sub.alt_obs_cm.std()), 2))


am_full = _altmatch_metrics(both)
am_noFbk = _altmatch_metrics(both[both.site != "Fairbanks"])
r_alt = am_full["r"]; rmse_alt = am_full["rmse_cm"]
r2_alt = am_full["r2"]; bias_alt = am_full["bias_cm"]
n_pred_cross = int(len(both))
print(f"[alt] 시추공 {len(altm)}개 중 유도교차 {n_pred_cross}개  "
      f"r={r_alt:.3f}  RMSE={rmse_alt:.1f}cm  R2={r2_alt:.3f}")
print(f"[alt] Fairbanks 제외(n={am_noFbk['n']})  r={am_noFbk['r']:.3f}  "
      f"RMSE={am_noFbk['rmse_cm']:.1f}cm  (RMSE 급감은 Fairbanks 심부 obs ALT의 3m격자 절단 아티팩트)")

# 참고행: Fairbanks 제외 민감도(41cm를 단독 헤드라인으로 쓰지 않도록 함께 기록)
res_rows.append(dict(model="gbm_mono", cv="alt_match", band="0C_crossing_noFairbanks",
                     unit="cm", depth_lo=MIN_DEPTH, depth_hi=TOP_M, n=am_noFbk["n"],
                     rmse_cm=am_noFbk["rmse_cm"], mae_cm=np.nan, bias_cm=am_noFbk["bias_cm"],
                     r2=am_noFbk["r2"],
                     target_sd_cm=am_noFbk["target_sd_cm"],
                     skill_over_mean=np.nan, coverage_90=np.nan, width_90=np.nan))
res_rows.append(dict(model="gbm_mono", cv="alt_match", band="0C_crossing",
                     unit="cm", depth_lo=MIN_DEPTH, depth_hi=TOP_M, n=n_pred_cross,
                     rmse_cm=rmse_alt, mae_cm=np.nan, bias_cm=bias_alt, r2=r2_alt,
                     target_sd_cm=round(float(both.alt_obs_cm.std()), 2) if n_pred_cross > 1 else np.nan,
                     skill_over_mean=np.nan, coverage_90=np.nan, width_90=np.nan))
res = pd.DataFrame(res_rows)
res.to_csv(os.path.join(PROC, "shallow3d_alaska_results.csv"), index=False)
altm.to_csv(os.path.join(PROC, "shallow3d_alaska_altmatch.csv"), index=False)
loro_df.to_csv(os.path.join(PROC, "shallow3d_alaska_loro.csv"), index=False)

# ---------- 9) 알래스카 격자 예측(깊이 0.5·1·2·3m) ----------
final = make_gbm(True)
final.fit(X, y)
grid_depths = [0.5, 1.0, 2.0, 3.0]
grid_parts = []
gcov = akc[COV + ["lat", "lon"]].copy()
for dm in grid_depths:
    gd = gcov.copy()
    dfeat = depth_feats(np.full(len(gd), dm))
    for c in dfeat.columns:
        gd[c] = dfeat[c].values
    gd["sqrttdd_x_expdepth"] = np.sqrt(np.clip(gd["e5_tdd"].values, 0, None)) * gd["exp_depth"].values
    gd["tmax_pred"] = final.predict(gd[FEATS].values.astype(float))
    gd["depth_m"] = dm
    grid_parts.append(gd[["lat", "lon", "depth_m", "tmax_pred"]])
grid = pd.concat(grid_parts, ignore_index=True)
grid.to_csv(os.path.join(PROC, "shallow3d_alaska_grid.csv"), index=False)
print(f"[grid] 알래스카 격자 {len(gcov)}셀 × {len(grid_depths)}깊이 = {len(grid)}행 저장")

# ---------- meta ----------
meta = dict(
    generated=time.strftime("%Y-%m-%d %H:%M"),
    runtime_sec=round(time.time() - t0, 1),
    n_label_rows=int(len(ak)), n_boreholes=int(ak.borehole_id.nunique()),
    n_sites=int(n_sites), n_fold=int(nfold),
    depth_range_m=[MIN_DEPTH, TOP_M],
    max_cov_join_km=round(maxd, 1),
    feats=FEATS,
    metric_unit={"bands": "C (t_max, 온도)", "alt_match": "cm (0C 교차 유도 ALT)",
                 "note": "all_metrics 컬럼명은 _cm 고정이나 밴드행 값은 °C다. results CSV의 unit 컬럼 참조."},
    metrics_all_mono={k: v for k, v in all_metrics(y, oof_mono).items()},
    metrics_all_free={k: v for k, v in all_metrics(y, oof_free).items()},
    monotonicity_violation={
        "mono": round(viol_mono, 4), "free": round(viol_free, 4),
        "note": ("mono=0.000은 물리 정합의 독립 검증이 아니라 제약의 자명한 귀결이다. "
                 "시추공 내부에서 공변량은 상수이고 depth 파생 5피처가 모두 depth에 단조라, "
                 "5피처 전부에 monotonic_cst를 걸면 프로파일 감소 단조가 수학적으로 보장된다. "
                 "무제약(free) 위반율 0.324와의 대비가 정보이며, 0.000 자체를 '장의 물리적 사실성' "
                 "증거로 인용하면 과대해석이다."),
    },
    loro_true={
        "method": "각 사이트 완전제외 재적합 후 그 사이트만 예측(진짜 leave-one-site-out)",
        "n_sites_reported": int(len(loro_df)),
        "n_pos_skill_loro": n_pos_loro,
        "n_pos_skill_oof6_aggregation": n_pos_oof6,
        "note": ("이전 결과의 'LORO'는 실제로 6-fold OOF의 사이트별 집계였다(co-fold 사이트가 "
                 "함께 훈련에서 빠져 진짜 LORO가 아님). 진짜 LORO 재적합 시 사이트별 R²가 이동한다 "
                 "(예 Barrow 0.15->0.39, Ivotuk 0.54->0.21, Galbraith -0.06->+0.12, Dead Horse 0.72->0.63). "
                 "양전이 다수 방향은 유지되나 표현을 'LORO'로 정정했다. 상세는 loro CSV의 "
                 "r2(=LORO)·r2_oof6(=6-fold 집계) 컬럼 비교."),
    },
    alt_match={"n_boreholes_with_obs": int(len(altm)),
               "n_with_pred_crossing": n_pred_cross,
               "r": round(r_alt, 3) if not np.isnan(r_alt) else None,
               "rmse_cm": rmse_alt if not np.isnan(rmse_alt) else None,
               "r2": r2_alt if not np.isnan(r2_alt) else None,
               "bias_cm": bias_alt if not np.isnan(bias_alt) else None,
               "no_fairbanks": {"n": am_noFbk["n"],
                                "r": round(am_noFbk["r"], 3) if not np.isnan(am_noFbk["r"]) else None,
                                "rmse_cm": am_noFbk["rmse_cm"], "r2": am_noFbk["r2"]},
               "note": ("RMSE 41cm는 헤드라인 단독 사용 부적절. Fairbanks(18개 중 6개) 제외 시 "
                        "RMSE 41->24cm로 급감하는데 이는 모델 개선이 아니라 Fairbanks 심부 obs ALT를 "
                        "3m 관측격자가 못 잡아 생기는 절단 아티팩트다. r(0.28->0.25)은 큰 변화 없다. "
                        "obs·pred ALT 모두 최심 관측깊이(<=3m)로 상한 절단되므로 평가가 얕은 대역에 "
                        "유리하게 편향된다(altmatch CSV의 alt_obs_near_maxdepth 플래그 참조). "
                        "R²<0(=자기평균보다 나쁨)이 정직한 요약이다."),
               },
    phase1_deep_reference={
        "r": 0.162, "rmse_cm": 172.0, "n": 89, "obs_alt_sd_cm": 150, "obs_alt_max_cm": 495,
        "frac_pred_clip_400cm": 0.10,
        "note": ("field3d_reeval 심부판(스위스+알래스카 혼합) 0C->ALT 참조. RMSE 절대비교는 사과-오렌지 "
                 "비교다. 심부판 n=89·obs ALT SD≈150cm·최대 495cm·10%가 400cm 클립 포화, 얕은판 n=18·"
                 "obs ALT SD≈37cm(알래스카 전용). 대상 분산이 약 4배 작아 RMSE 절대차(41 vs 172cm)의 "
                 "상당부는 모델 우수성이 아니라 대상범위 축소를 반영한다. 두 판 모두 자기평균 대비 "
                 "skill이 음수(심부≈-0.21, 얕은 R²≈-0.29)라는 점이 정직한 그림이다. '약 4배 개선'· "
                 "'얕은 장이 심부 장을 명백히 이긴다'는 표현은 부적절."),
    },
    caveats=[
        "라벨=시추공 t_max(연최대 월기후 온도) 실측만. 유사라벨 미사용.",
        "공변량은 최근접 ALT 셀에서 취득(최대 %.1fkm) — 원지점 재추출 아님." % maxd,
        "사이트 %d개로 공간블록·LORO 표본 제한. 툰드라 사면 편중 가능(외삽·표본편향)." % n_sites,
        "격자 예측은 3m 이하 얕은 대역 한정(심부 외삽 아님).",
        "격자 예측은 완전한 공변량을 가진 이산 알래스카 셀을 4개 깊이에서 산점 예측한 것이다. "
        "연속 볼륨/조밀 '온도장'이 아니며 '3D 온도장'이라는 표현은 산출물보다 조밀·연속성을 과장한다.",
        "온도 예측 자체는 held-out 검증 성립(t_max site-block R²≈0.47). ALT 유도 정합은 성립하지 않음(R²<0).",
    ],
)
with open(os.path.join(PROC, "shallow3d_alaska_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[meta] 저장 완료. 총 {meta['runtime_sec']}s")
print("[done] results/altmatch/loro/grid/meta 저장")
