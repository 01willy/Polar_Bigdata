"""S10 — 얕은 3D 온도 볼륨 생성(재현 검증 연동).

목적
- shallow3d_alaska.py(2026-07-20 검증, site-block R2 0.47)와 동일한 데이터·모델 구성으로
  최종 GBM(단조 제약, random_state=0)을 재적합하고, 규칙 격자 x 조밀 깊이(0.05-3.0 m)의
  t_max 볼륨을 산출한다. 단면·3D 렌더(0°C 등온면)의 입력이 된다.
- 재현 앵커: 원 스크립트 산출 shallow3d_alaska_grid.csv(13,606셀 x 4깊이)와 동일 셀·동일
  깊이에서 예측이 일치하는지 assert로 검증한다(모델 동일성 보증).

정직성(원 meta caveat 승계)
- 격자는 관측 공변량 셀의 지원반경(15 km) 안에서만 채운다. 반경 밖은 NaN으로 남겨
  "연속 볼륨"이라는 과장을 피한다(support mask).
- 0°C 교차 깊이(thaw_cm)는 전량학습 모델의 산출이므로 검증 지표가 아니다. 검증용
  상관(r=0.28)은 OOF 포락선 기반 shallow3d_alaska_altmatch.csv를 사용한다.

산출
- data/processed/s10_shallow3d_volume.npz  (lats, lons, depths_m, tmax[°C], thaw_cm, 지원마스크)
  npz는 gitignore 대상(재생성 가능).

실행: SMOKE=1 /home/anaconda3/bin/python scripts/3_deep_learning/s10_shallow3d_volume.py  (ROOT, CPU)
"""
import sys, os, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.spatial import cKDTree

SMOKE = os.environ.get("SMOKE", "0") == "1"
PROC = "data/processed"
GT = os.path.join(PROC, "ground_temp_gtnp_global.csv")
CELL = os.path.join(PROC, "dl_dataset_cell_v3_soil.csv")
AK_BBOX = dict(lat_lo=54.0, lat_hi=72.0, lon_lo=-170.0, lon_hi=-129.0)
TOP_M, MIN_DEPTH = 3.0, 0.02
SUPPORT_KM = 15.0
t0 = time.time()

CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
SOIL = ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15", "sg_cfvo_5_15",
        "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15", "sg_soc_15_30"]
COV = CLIMATE + TERRAIN + SOIL
DEPTH_FEATS = ["depth_m", "sqrt_depth", "inv_depth", "exp_depth", "sqrttdd_x_expdepth"]
FEATS = COV + DEPTH_FEATS

# ---------- 1) 라벨·공변량: shallow3d_alaska.py와 동일 구성 ----------
gt = pd.read_csv(GT)
ak = gt[(gt.lat >= AK_BBOX["lat_lo"]) & (gt.lat <= AK_BBOX["lat_hi"]) &
        (gt.lon >= AK_BBOX["lon_lo"]) & (gt.lon <= AK_BBOX["lon_hi"])].copy()
ak = ak.dropna(subset=["depth", "t_max"])
ak = ak[(ak.depth >= MIN_DEPTH) & (ak.depth <= TOP_M)].copy()
bd = ak.groupby("borehole_id").depth.nunique()
ak = ak[ak.borehole_id.isin(bd[bd >= 3].index)].copy()

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
    return i

bh = ak.groupby("borehole_id").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
covrows = {r.borehole_id: akc.loc[nearest_cov(r.lat, r.lon), COV].to_dict() for r in bh.itertuples()}
covdf = pd.DataFrame(covrows).T
covdf.index.name = "borehole_id"
ak = ak.merge(covdf.reset_index(), on="borehole_id", how="left")

def depth_feats(d):
    d = np.asarray(d, float)
    return pd.DataFrame({"depth_m": d, "sqrt_depth": np.sqrt(d),
                         "inv_depth": 1.0 / (1.0 + d), "exp_depth": np.exp(-d)})

dfeat = depth_feats(ak.depth.values)
for c in dfeat.columns:
    ak[c] = dfeat[c].values
ak["sqrttdd_x_expdepth"] = np.sqrt(np.clip(ak["e5_tdd"].values, 0, None)) * ak["exp_depth"].values
y = ak["t_max"].values.astype(float)
X = ak[FEATS].values.astype(float)
print(f"[label] 라벨 행={len(ak)} (원 764와 일치해야 함)")
assert len(ak) == 764, f"라벨 행 불일치: {len(ak)} != 764"

# ---------- 2) 최종 GBM(원 스크립트 9절과 동일 구성·seed) ----------
def mono_cst(feat_names):
    dec = {"depth_m", "sqrt_depth"}
    inc = {"inv_depth", "exp_depth", "sqrttdd_x_expdepth"}
    return [-1 if f in dec else (1 if f in inc else 0) for f in feat_names]

final = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=None,
                                      max_leaf_nodes=31, min_samples_leaf=20,
                                      l2_regularization=1.0, random_state=0,
                                      monotonic_cst=mono_cst(FEATS))
final.fit(X, y)

# ---------- 3) 재현 앵커: 원 grid CSV와 동일 셀·깊이 예측 일치 assert ----------
ref = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_grid.csv"))
chk = ref if not SMOKE else ref.iloc[::37].copy()
gcov = akc.set_index(akc.index)
# 원 grid CSV는 akc 순서 그대로 4깊이 반복 저장이므로 셀 인덱스 = 행번호 % len(akc)
cell_idx = np.arange(len(ref)) % len(akc)
Xg = akc.loc[cell_idx if not SMOKE else cell_idx[::37], COV].reset_index(drop=True)
dfe = depth_feats(chk.depth_m.values)
for c in dfe.columns:
    Xg[c] = dfe[c].values
Xg["sqrttdd_x_expdepth"] = np.sqrt(np.clip(Xg["e5_tdd"].values, 0, None)) * Xg["exp_depth"].values
pred_chk = final.predict(Xg[FEATS].values.astype(float))
mad = float(np.max(np.abs(pred_chk - chk.tmax_pred.values)))
print(f"[anchor] 원 grid CSV 대비 최대 절대차 = {mad:.2e} °C")
assert mad < 1e-8, "재적합 모델이 원 산출과 불일치(구성·seed 확인 필요)"

# ---------- 4) 규칙 격자 x 조밀 깊이 볼륨 ----------
if SMOKE:
    dlon, dlat, ddep = 0.6, 0.3, 0.5
else:
    dlon, dlat, ddep = 0.12, 0.06, 0.05
lons = np.arange(-166.2, -140.9, dlon)
lats = np.arange(61.0, 71.5, dlat)
depths = np.round(np.arange(ddep, TOP_M + 1e-9, ddep), 3)
lon2d, lat2d = np.meshgrid(lons, lats)

# 지원마스크: 최근접 공변량 셀 <= SUPPORT_KM (근사 평면거리, 위도보정)
latm = np.deg2rad(np.nanmean(clat))
tree = cKDTree(np.c_[clon * np.cos(latm) * 111.0, clat * 111.0])
d_km, near_i = tree.query(np.c_[lon2d.ravel() * np.cos(latm) * 111.0, lat2d.ravel() * 111.0])
support = (d_km <= SUPPORT_KM).reshape(lat2d.shape)
sup_flat = support.ravel()
print(f"[grid] {len(lats)}x{len(lons)} 노드 중 지원(<= {SUPPORT_KM:g} km) {sup_flat.sum()}개 "
      f"({100*sup_flat.mean():.1f}%) x 깊이 {len(depths)}층")

covg = akc.loc[near_i[sup_flat], COV].reset_index(drop=True)
nsup = len(covg)
tmax = np.full((len(depths), len(lats), len(lons)), np.nan, dtype=np.float32)
for k, dm in enumerate(depths):
    gd = covg.copy()
    dfe = depth_feats(np.full(nsup, dm))
    for c in dfe.columns:
        gd[c] = dfe[c].values
    gd["sqrttdd_x_expdepth"] = np.sqrt(np.clip(gd["e5_tdd"].values, 0, None)) * gd["exp_depth"].values
    plane = np.full(lat2d.size, np.nan, dtype=np.float32)
    plane[sup_flat] = final.predict(gd[FEATS].values.astype(float)).astype(np.float32)
    tmax[k] = plane.reshape(lat2d.shape)

# ---------- 5) 0°C 교차 깊이(가장 깊은 +->- 교차, 원 crossing_alt와 동일 규칙) ----------
thaw_cm = np.full(lat2d.shape, np.nan, dtype=np.float32)
prof = tmax.reshape(len(depths), -1)
for j in np.where(sup_flat)[0]:
    t = prof[:, j]
    alt = np.nan
    for i in range(len(depths) - 1):
        if t[i] > 0 and t[i + 1] <= 0:
            alt = depths[i] + (depths[i + 1] - depths[i]) * t[i] / (t[i] - t[i + 1])
    thaw_cm.ravel()[j] = alt * 100.0 if np.isfinite(alt) else np.nan

n_cross = int(np.isfinite(thaw_cm).sum())
print(f"[thaw] 0°C 교차 노드 {n_cross}/{sup_flat.sum()} ({100*n_cross/max(sup_flat.sum(),1):.1f}%)")

out = os.path.join(PROC, "s10_shallow3d_volume.npz")
np.savez_compressed(out, lats=lats, lons=lons, depths_m=depths, tmax_c=tmax,
                    thaw_cm=thaw_cm, support=support, support_km=SUPPORT_KM,
                    anchor_max_abs_diff=mad, smoke=int(SMOKE))
print(f"[save] {out}  tmax {tmax.shape}  {time.time()-t0:.1f}s")
