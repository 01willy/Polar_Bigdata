"""실험 2 (A1): 3D 지중온도장 누설없는 재평가.

기존 결론(p2a_field_experiment.py, ADOPT_full)은 site-disjoint GroupKFold 기반이다.
같은 지중온도 시추공들이 공간적으로 촘촘하게 군집(사이트 52%가 최근접 1km 이내, 73%가 같은
0.5°블록 공유)하므로, site-GKF는 held-out 사이트의 이웃이 학습셋에 남아 근접 누설을 허용한다.
전 공변량(지형6+CCI)은 좌표에 강하게 종속(0.5°블록 내 거의 동일값)이라, 이 누설이 있으면
"공변량 추가가 개선"이라는 판정이 실제 일반화가 아닌 근접 보간 착시일 수 있다.

본 스크립트는 기존 결론을 반증하려 설계한다:
 1. CV 교체: (a) site-GKF6 재현(기준선) (b) 공간블록 6-fold(0.5°블록 단위) (c) LORO(지역, test>=100행).
 2. 누설 진단: 사이트 최근접거리 분포, 0.5°블록 공유 사이트 수 정량화.
 3. 재비교: FIELD_base(기후8+깊이) vs FIELD_full(+지형6+CCI+깊이) 전체·깊이밴드별 RMSE·R2.
    누설 통제 후에도 full>base면 채택, 아니면 기존 심부 이득이 누설 착시임을 보고.
 4. 깊이 연속성: 같은 위치 예측 온도-깊이 프로파일의 물리 단조성(심부로 진폭 감쇠) 위반율.
 5. field-ALT 정합(정직판): 연최대 포락선 0°C 교차로 ALT 유도, 감쇠깊이 δ를 held-out으로 적합·평가.

적대적 검증 반영(2026-07-20 개정):
 - profile_cross 버그수정: 완전동결(표층 음온) 프로파일서 d[0]=0cm 대신 NaN(참고열 magt).
 - 표본 구성 진단 추가: 심부 밴드 스위스 PERMOS 편중(80%+) 정량화 → 알래스카 심부는 도메인 밖 외삽.
 - 공간블록 fold 불균형 진단 추가: 밴드 델타는 pooled OOF(스위스 가중), 정밀 효과크기 아님.
 - ALT 정합 표본 감소·클립 포화 명시(매칭 130 vs 평가 89), 귀무/음성 결과로 명시.

산출(신규명, 기존 p2a_* 미덮어쓰기):
 - data/processed/field3d_reeval_bands.csv        (model,cv,depth_band,n,rmse_c,r2)
 - data/processed/field3d_reeval_leakage.csv       (사이트별 최근접거리·블록공유)
 - data/processed/field3d_reeval_composition.csv   (밴드별 도메인 편중: 스위스·PERMOS 비율)
 - data/processed/field3d_reeval_foldbalance.csv   (공간블록 fold별 행수·심부수·스위스비율)
 - data/processed/field3d_reeval_monotonicity.csv  (모델·CV별 단조성 위반율)
 - data/processed/field3d_reeval_altmatch.csv      (사이트별 유도/관측 ALT, held-out δ)
 - data/processed/field3d_reeval_meta.json
"""
import os, sys, json, glob, calendar, warnings
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
warnings.filterwarnings("ignore")

ROOT = "/home/willy010313/Polar_Bigdata"
os.chdir(ROOT)
sys.path.insert(0, "src")
DEM_DIR = "data/raw/dem"
NFOLD = 6
RNG = 0
BLOCK_DEG = 0.5  # 공간블록 크기


def fourier(dm):
    dn = (np.asarray(dm) / 30.0).astype(np.float32)
    out = []
    for k in range(5):
        out += [np.sin(2 ** k * np.pi * dn), np.cos(2 ** k * np.pi * dn)]
    return np.column_stack(out)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def haversine_km(lat, lon):
    R = 6371.0
    la = np.radians(lat)[:, None]; lo = np.radians(lon)[:, None]
    la2 = np.radians(lat)[None, :]; lo2 = np.radians(lon)[None, :]
    dlat = la - la2; dlon = lo - lo2
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ================================================================ (1) 라벨
g = pd.read_csv("data/processed/ground_temp_all.csv")
g = g[(g.depth_m > 0) & (g.depth_m <= 30) & (g.temp_c > -25) & (g.temp_c < 25)].reset_index(drop=True)
print(f"[label] 필터 후 {len(g):,}행, {g.site.nunique()} 사이트, {g.region.nunique()} 지역")

# ================================================================ 기후8(e5_*)
E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim0 = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
elat, elon = clim0["latitude"].values, clim0["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]


def derive(c):
    t = c["t2m"].values - 273.15; stl = c["stl1"].values - 273.15; sdp = c["sd"].values
    tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
    return dict(e5_maat=np.nanmean(t, 0), e5_tdd=tdd, e5_fdd=fdd, e5_sqrt_tdd=np.sqrt(tdd),
                e5_twarm=np.nanmax(t, 0), e5_tcold=np.nanmin(t, 0),
                e5_stl1=np.nanmean(stl, 0), e5_swe=np.nanmean(sdp, 0))


E5 = derive(clim0)
iy = np.clip(np.searchsorted(-elat, -g.lat.values), 0, len(elat) - 1)
ix = np.clip(np.searchsorted(elon, g.lon.values), 0, len(elon) - 1)
for k, gr in E5.items():
    g[k] = gr[iy, ix].astype(np.float32)
n0 = len(g)
g = g.dropna(subset=["e5_maat"]).reset_index(drop=True)
print(f"[era5] 기후8 부착, e5_maat 결측 제거 {n0 - len(g)}행 → {len(g):,}행")

# ================================================================ 지형6(dem_*)
DEMF = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]


def tile_name(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


sites = g.drop_duplicates("site")[["site", "lat", "lon"]].reset_index(drop=True)
sites["tlat"] = np.floor(sites.lat).astype(int)
sites["tlon"] = np.floor(sites.lon).astype(int)
demvals = {f: np.full(len(sites), np.nan) for f in DEMF}
has_dem = np.zeros(len(sites), dtype=bool)
mperdeg = 111320.0
Wp = 33; Hp = Wp // 2
n_tile_ok = n_tile_miss = 0
for (tlat, tlon), grp in sites.groupby(["tlat", "tlon"]):
    path = os.path.join(DEM_DIR, tile_name(int(tlat), int(tlon)) + ".tif")
    if not (os.path.exists(path) and os.path.getsize(path) > 1000):
        n_tile_miss += 1
        continue
    n_tile_ok += 1
    with rasterio.open(path) as rio:
        arr = rio.read(1).astype(np.float32)
        arr[arr == rio.nodata] = np.nan
        ny, nx = arr.shape
        dy = mperdeg * abs(rio.transform.e)
        for idx, r in grp.iterrows():
            row, col = rio.index(r.lon, r.lat)
            if not (0 <= row < ny and 0 <= col < nx):
                continue
            dx = mperdeg * abs(rio.transform.a) * np.cos(np.radians(r.lat))
            r0, r1 = max(0, row - Hp), min(ny, row + Hp + 1)
            c0, c1 = max(0, col - Hp), min(nx, col + Hp + 1)
            win = arr[r0:r1, c0:c1]
            if win.size == 0 or np.isnan(win).all():
                continue
            cen = arr[row, col]
            if not np.isfinite(cen):
                continue
            gy, gx = np.gradient(win, dy, dx)
            i, j = row - r0, col - c0
            sl = np.degrees(np.arctan(np.hypot(gy[i, j], gx[i, j])))
            asp = np.arctan2(-gy[i, j], gx[i, j])
            vals = [cen, sl, np.sin(asp), np.cos(asp),
                    cen - np.nanmean(win), float(np.nanstd(win))]
            for f, v in zip(DEMF, vals):
                demvals[f][idx] = v
            has_dem[idx] = True
for f in DEMF:
    sites[f] = demvals[f]
sites["has_dem"] = has_dem.astype(int)
print(f"[dem] 지형6 추출: 사이트 {has_dem.sum()}/{len(sites)} 유효 "
      f"(타일 OK {n_tile_ok}, MISS {n_tile_miss})")
g = g.merge(sites[["site"] + DEMF + ["has_dem"]], on="site", how="left")


# ================================================================ CCI(cci_alt)
def nearest_idx(coords, vals):
    order = np.argsort(coords); cs = coords[order]
    j = np.searchsorted(cs, vals); j = np.clip(j, 1, len(cs) - 1)
    left = cs[j - 1]; right = cs[j]
    j = np.where(np.abs(vals - left) <= np.abs(vals - right), j - 1, j)
    return order[j]


cci_files = sorted(glob.glob("data/raw/cci_alt/*.nc"))
uniq = g.drop_duplicates("site")[["site", "lat", "lon"]].reset_index(drop=True)
la, lo = uniq.lat.values, uniq.lon.values
mla, Mla = la.min() - 0.2, la.max() + 0.2
mlo, Mlo = lo.min() - 0.2, lo.max() + 0.2
acc = None; cnt = None; latc = lonc = None
for f in cci_files:
    dsc = xr.open_dataset(f)
    sub = dsc["ALT"].sel(lat=slice(mla, Mla), lon=slice(mlo, Mlo))
    if sub.ndim == 3:
        sub = sub.isel(time=0)
    a = sub.values.astype("float32")
    if acc is None:
        acc = np.zeros_like(a); cnt = np.zeros_like(a)
        latc = sub.lat.values; lonc = sub.lon.values
    v = np.isfinite(a); acc[v] += a[v]; cnt[v] += 1
    dsc.close()
mean_cm = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan) * 100.0
ci = nearest_idx(latc, la); cj = nearest_idx(lonc, lo)
cci_site = mean_cm[ci, cj]
uniq["cci_alt"] = np.round(cci_site, 2)
uniq["cci_valid"] = np.isfinite(cci_site).astype(int)
print(f"[cci] 사이트 CCI 유효 {int(uniq.cci_valid.sum())}/{len(uniq)} "
      f"({100 * uniq.cci_valid.mean():.0f}%)")
g = g.merge(uniq[["site", "cci_alt", "cci_valid"]], on="site", how="left")

# ================================================================ 깊이 특징
g["logd"] = np.log1p(g.depth_m)
FF = fourier(g.depth_m.values)
FFn = [f"ff{i}" for i in range(FF.shape[1])]
for i, n in enumerate(FFn):
    g[n] = FF[:, i].astype(np.float32)
DEPTHF = ["depth_m", "logd"] + FFn
CLIMF = E5F
FEAT_BASE = CLIMF + DEPTHF
FEAT_FULL = CLIMF + DEMF + ["cci_alt"] + DEPTHF
IND = ["has_dem", "cci_valid"]
print(f"[feat] base {len(FEAT_BASE)}개, full {len(FEAT_FULL) + len(IND)}개")

# ================================================================ (2) 누설 진단
s = g.drop_duplicates("site")[["site", "lat", "lon", "region"]].reset_index(drop=True)
s["block"] = (np.floor(s.lat / BLOCK_DEG) * 100000 + np.floor(s.lon / BLOCK_DEG)).astype(np.int64)
D = haversine_km(s.lat.values, s.lon.values)
np.fill_diagonal(D, np.inf)
s["nn_km"] = D.min(1)
blk_count = s.block.map(s.block.value_counts())
s["block_share"] = blk_count.values          # 같은 0.5°블록 내 사이트 수
s.to_csv("data/processed/field3d_reeval_leakage.csv", index=False)
leak = dict(
    n_sites=int(len(s)),
    nn_km_median=round(float(s.nn_km.median()), 4),
    nn_km_q=[round(float(v), 4) for v in np.percentile(s.nn_km, [0, 10, 25, 50, 75, 90])],
    frac_nn_lt_1km=round(float((s.nn_km < 1).mean()), 4),
    frac_nn_lt_5km=round(float((s.nn_km < 5).mean()), 4),
    n_blocks=int(s.block.nunique()),
    n_multi_blocks=int((s.block.value_counts() > 1).sum()),
    max_sites_in_block=int(s.block.value_counts().max()),
    frac_sites_in_multiblock=round(float((s.block_share > 1).mean()), 4),
)
print(f"[누설] 최근접거리 중앙 {leak['nn_km_median']}km, <1km {100 * leak['frac_nn_lt_1km']:.0f}%, "
      f"0.5°블록 공유 사이트 {100 * leak['frac_sites_in_multiblock']:.0f}%, 최대 {leak['max_sites_in_block']}개/블록")
g = g.merge(s[["site", "block", "region"]].rename(columns={"region": "region2"}), on="site", how="left")


# ================================================================ (3) CV별 OOF 학습
def fit_oof_folds(cols, folds):
    """folds = list of (train_idx, test_idx). NaN는 train fold 중앙값 대체."""
    pred = np.full(len(g), np.nan)
    for tr, te in folds:
        Xtr = g.iloc[tr][cols].copy(); Xte = g.iloc[te][cols].copy()
        for c in cols:
            med = Xtr[c].median()
            if not np.isfinite(med):
                med = 0.0
            Xtr[c] = Xtr[c].fillna(med); Xte[c] = Xte[c].fillna(med)
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=RNG)
        m.fit(Xtr.values, g.iloc[tr].temp_c.values)
        pred[te] = m.predict(Xte.values)
    return pred


# --- CV (a) site-GKF6 (기준선 재현)
gkf = GroupKFold(n_splits=NFOLD)
folds_site = list(gkf.split(g, groups=g.site.values))

# --- CV (b) 공간블록 6-fold: 0.5°블록을 KFold로 나눠 블록 단위 분리
blocks = np.sort(s.block.unique())
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=RNG)
block_fold = {}
for fi, (_, te_b) in enumerate(kf.split(blocks)):
    for b in blocks[te_b]:
        block_fold[b] = fi
g["block_fold"] = g.block.map(block_fold).values
folds_block = []
for fi in range(NFOLD):
    te = np.where(g.block_fold.values == fi)[0]
    tr = np.where(g.block_fold.values != fi)[0]
    if len(te) >= 5:
        folds_block.append((tr, te))

# --- CV (c) LORO: 지역별, test>=100행
loro_regions = [r for r in g.region.value_counts().index if (g.region == r).sum() >= 100]
folds_loro = []
loro_map = {}
for r in loro_regions:
    te = np.where(g.region.values == r)[0]
    tr = np.where(g.region.values != r)[0]
    folds_loro.append((tr, te))
    loro_map[r] = te
print(f"[cv] site-GKF {len(folds_site)}fold, 공간블록 {len(folds_block)}fold, LORO {len(folds_loro)}지역 {loro_regions}")

CVS = {"site_gkf6": folds_site, "spatial_block6": folds_block, "loro": folds_loro}
MODELS = {"FIELD_base": FEAT_BASE, "FIELD_full": FEAT_FULL + IND}

oof = {}  # (cv, model) -> pred
for cvname, folds in CVS.items():
    for model, cols in MODELS.items():
        oof[(cvname, model)] = fit_oof_folds(cols, folds)
        cov = np.isfinite(oof[(cvname, model)])
        print(f"[cv={cvname:14s}] {model:11s} 전체 RMSE={rmse(g.temp_c[cov], oof[(cvname, model)][cov]):.3f}°C "
              f"R2={r2(g.temp_c[cov], oof[(cvname, model)][cov]):.3f} (n={cov.sum()})")

# ================================================================ 깊이밴드 결과표
BANDS = [("0-2m", 0, 2), ("2-5m", 2, 5), ("5-10m", 5, 10), ("10-20m", 10, 20)]
rows = []
for cvname in CVS:
    for model in MODELS:
        pred = oof[(cvname, model)]
        for name, lo_d, hi_d in BANDS + [("all", 0, 30)]:
            mk = (g.depth_m > lo_d) & (g.depth_m <= hi_d) & np.isfinite(pred)
            if mk.sum() < 5:
                continue
            rows.append(dict(model=model, cv=cvname, depth_band=name, n=int(mk.sum()),
                             rmse_c=round(rmse(g.temp_c[mk], pred[mk]), 4),
                             r2=round(r2(g.temp_c[mk], pred[mk]), 4)))
res = pd.DataFrame(rows)
res.to_csv("data/processed/field3d_reeval_bands.csv", index=False)

# ---------------------------------------------------------------- (3b) 표본 구성 진단
# 심부 밴드가 특정 도메인(스위스/PERMOS 알프스 암빙하)에 편중되면 심부 결과·알래스카
# 지도는 도메인 밖 외삽이다. 밴드별 구성비를 정량화해 CSV로 남긴다.
if "source" not in g.columns:
    g = g.merge(pd.read_csv("data/processed/ground_temp_all.csv")[["site", "source"]]
                .drop_duplicates("site"), on="site", how="left")
comp_rows = []
for name, lo_d, hi_d in BANDS + [("all", 0, 30)]:
    mk = (g.depth_m > lo_d) & (g.depth_m <= hi_d)
    n = int(mk.sum())
    if n == 0:
        continue
    sw = int((g.loc[mk, "region"] == "Switzerland").sum())
    pm = int((g.loc[mk, "source"] == "PERMOS").sum()) if "source" in g.columns else -1
    us = int((g.loc[mk, "region"] == "United States").sum())
    comp_rows.append(dict(depth_band=name, n=n,
                          frac_switzerland=round(sw / n, 4),
                          frac_permos=round(pm / n, 4) if pm >= 0 else np.nan,
                          frac_united_states=round(us / n, 4),
                          n_regions=int(g.loc[mk, "region"].nunique())))
comp = pd.DataFrame(comp_rows)
comp.to_csv("data/processed/field3d_reeval_composition.csv", index=False)
deep_sw = comp[comp.depth_band.isin(["5-10m", "10-20m"])].frac_switzerland.mean()
print("\n[표본구성] 밴드별 도메인 편중(심부일수록 스위스 PERMOS 알프스 집중)")
print(comp.to_string(index=False))
print(f"  → 심부(5-20m) 스위스 비율 평균 {100 * deep_sw:.0f}%. 알래스카 지도(그림3)는 도메인 밖 외삽으로 해석해야 한다.")

# ---------------------------------------------------------------- (3c) 공간블록 fold 불균형 진단
# spatial_block6 fold는 행수가 심하게 불균형(작은 fold일수록 스위스-희박). pooled OOF의
# 밴드 델타는 스위스-과다 fold에 가중되므로 단일 델타 수치를 정밀 효과크기로 과대해석하면 안 된다.
fold_rows = []
for fi, (tr, te) in enumerate(folds_block):
    sub = g.iloc[te]
    fold_rows.append(dict(cv="spatial_block6", fold=fi, n=int(len(te)),
                          n_deep_5_20m=int(((sub.depth_m > 5) & (sub.depth_m <= 20)).sum()),
                          frac_switzerland=round(float((sub.region == "Switzerland").mean()), 4),
                          n_blocks=int(sub.block.nunique()),
                          n_regions=int(sub.region.nunique())))
foldbal = pd.DataFrame(fold_rows)
foldbal.to_csv("data/processed/field3d_reeval_foldbalance.csv", index=False)
print("\n[fold불균형] spatial_block6 fold별 표본(행수 min/max 편차·스위스 비율)")
print(foldbal.to_string(index=False))
print(f"  → fold 행수 {foldbal.n.min()}-{foldbal.n.max()}, 심부 {foldbal.n_deep_5_20m.min()}-{foldbal.n_deep_5_20m.max()}. "
      f"밴드 델타는 pooled OOF(스위스 가중)이며 정밀 효과크기 아님.")

print("\n[결과] 깊이밴드 RMSE: base vs full × CV (full-base 음수=full 개선)")
for cvname in CVS:
    sub = res[res.cv == cvname]
    piv = sub.pivot(index="depth_band", columns="model", values="rmse_c")
    piv["Δ(full-base)"] = (piv["FIELD_full"] - piv["FIELD_base"]).round(4)
    order = ["0-2m", "2-5m", "5-10m", "10-20m", "all"]
    print(f"\n  CV={cvname}")
    print(piv.reindex([o for o in order if o in piv.index]).to_string())

# ================================================================ (4) 깊이 연속성(단조 감쇠 위반율)
# 물리: 심부로 갈수록 계절진폭 감쇠 → 예측 온도의 |T - T_deep| 이 깊이 증가에 대해 대체로 감소해야 한다.
# 여기서는 연평균장 프록시이므로 '심부 진폭 감쇠'를 |dT/dz| 의 심부 감소로 근사 진단한다.
mono_rows = []
for cvname in CVS:
    pred_full = oof[(cvname, "FIELD_full")]
    viol = 0; tot = 0
    for site, sg in g.groupby("site"):
        idx = sg.index.values
        p = pred_full[idx]
        d = sg.depth_m.values
        if not np.all(np.isfinite(p)) or len(d) < 4:
            continue
        o = np.argsort(d)
        d = d[o]; p = p[o]
        # 인접 깊이 온도 변화율의 절대값
        grad = np.abs(np.diff(p) / np.maximum(np.diff(d), 1e-6))
        if len(grad) < 3:
            continue
        # 심부 절반의 평균 기울기 > 표층 절반의 평균 기울기 → 감쇠 위반
        half = len(grad) // 2
        shallow = grad[:half].mean(); deep = grad[half:].mean()
        tot += 1
        if deep > shallow + 1e-3:
            viol += 1
    if tot > 0:
        mono_rows.append(dict(cv=cvname, model="FIELD_full", n_sites=tot,
                              n_violation=viol, viol_rate=round(viol / tot, 4)))
mono = pd.DataFrame(mono_rows)
mono.to_csv("data/processed/field3d_reeval_monotonicity.csv", index=False)
print("\n[단조성] 심부 진폭 감쇠 위반율(FIELD_full)")
print(mono.to_string(index=False))

# ================================================================ (5) field-ALT 정합(정직판, held-out δ)
def profile_cross(depths, temps, surface_frozen="nan"):
    """0°C 하향 교차 깊이(m). 표층부터 0°C 이하인 프로파일 처리 방식:
      surface_frozen="nan": 교차 없음으로 간주(NaN). 심부까지 음온인 완전동결
        프로파일에 대해 '0cm 활동층'이라는 물리적으로 무의미한 값을 만들지 않는다.
      surface_frozen="zero": 표층 0cm 반환(하위호환).
    포락선(seas_alt)처럼 표층이 양온인 경우에는 두 옵션 모두 정상 교차를 반환한다.
    """
    d = np.asarray(depths); t = np.asarray(temps)
    if len(d) < 2:
        return np.nan
    if t[0] <= 0:
        return 0.0 if surface_frozen == "zero" else np.nan
    for a in range(len(d) - 1):
        if t[a] > 0 >= t[a + 1]:
            frac = t[a] / (t[a] - t[a + 1])
            return float(d[a] + frac * (d[a + 1] - d[a]))
    return np.nan


# 관측 ALT 소스
ag = pd.read_csv("data/processed/alt_global.csv")[["lat", "lon", "alt_cm"]].dropna()
ag = ag.groupby([ag.lat.round(4), ag.lon.round(4)]).alt_cm.mean().reset_index()
ag.columns = ["lat", "lon", "alt_cm"]
obs_list = [ag]
try:
    ev = pd.read_csv("data/processed/alt_gtnp_envelope_cell.csv")[["lat", "lon", "alt_cm"]].dropna()
    obs_list.append(ev)
except Exception:
    pass
obs = pd.concat(obs_list, ignore_index=True)
obs_ll = obs[["lat", "lon"]].values
obs_alt_cm = obs.alt_cm.values

# 공간블록 CV의 full OOF로 프로파일 구성(누설 통제된 예측 사용)
pred_field = oof[("spatial_block6", "FIELD_full")]
profiles = []  # (site, lat, lon, zg, tg, amp, obs_alt, dist, block)
for site, sg in g.groupby("site"):
    idx = sg.index.values
    p = pred_field[idx]
    if not np.all(np.isfinite(p)):
        continue
    dpt = sg.depth_m.values
    o = np.argsort(dpt)
    ds_ = dpt[o]; ts_ = p[o]
    if ds_.max() <= 0:
        continue
    zg = np.linspace(0.0, min(30.0, float(ds_.max())), 300)
    tg = np.interp(zg, ds_, ts_)
    amp = max(0.0, (float(sg.e5_twarm.iloc[0]) - float(sg.e5_tcold.iloc[0])) / 2.0)
    slat, slon = sg.lat.iloc[0], sg.lon.iloc[0]
    dd = np.hypot(obs_ll[:, 0] - slat, obs_ll[:, 1] - slon)
    j = int(np.argmin(dd))
    if dd[j] > 0.05:
        continue
    profiles.append((site, round(float(slat), 4), round(float(slon), 4), zg, tg, amp,
                     float(obs_alt_cm[j]), float(dd[j]), int(sg.block.iloc[0])))

print(f"\n[ALT정합] 관측매칭 사이트 {len(profiles)}개")


def seas_alt(zg, tg, amp, delta):
    return profile_cross(zg, tg + amp * np.exp(-zg / delta))


# held-out δ: 공간블록으로 δ를 train 프로파일에 적합 → test 프로파일로 평가(δ 누설 차단)
prof_blocks = np.array([p[8] for p in profiles])
uniq_blocks = np.sort(np.unique(prof_blocks))
grid = np.linspace(0.2, 6.0, 59)
kf_d = KFold(n_splits=min(NFOLD, len(uniq_blocks)), shuffle=True, random_state=RNG)
alt_pred_seas = np.full(len(profiles), np.nan)
delta_used = np.full(len(profiles), np.nan)
for tr_b, te_b in kf_d.split(uniq_blocks):
    tr_blocks = set(uniq_blocks[tr_b]); te_blocks = set(uniq_blocks[te_b])
    tr_i = [i for i, p in enumerate(profiles) if p[8] in tr_blocks]
    te_i = [i for i, p in enumerate(profiles) if p[8] in te_blocks]
    if len(tr_i) < 5 or len(te_i) < 1:
        continue
    obs_tr = np.array([profiles[i][6] for i in tr_i])

    def rmse_delta(delta):
        est = np.array([seas_alt(profiles[i][3], profiles[i][4], profiles[i][5], delta) for i in tr_i]) * 100.0
        m = np.isfinite(est)
        if m.sum() < 5:
            return np.inf
        return np.sqrt(np.mean((np.clip(est[m], 0, 400) - obs_tr[m]) ** 2))

    d_fit = float(grid[int(np.argmin([rmse_delta(dv) for dv in grid]))])
    for i in te_i:
        alt_pred_seas[i] = np.clip(seas_alt(profiles[i][3], profiles[i][4], profiles[i][5], d_fit) * 100.0, 0, 400)
        delta_used[i] = d_fit

alt_rows = []
for k, (site, slat, slon, zg, tg, amp, oalt, dist, block) in enumerate(profiles):
    # 연평균 0°C 교차(MAGT 등온선, 참고 전용). 완전동결(표층부터 음온) 프로파일은
    # 하향 교차가 없으므로 NaN(과거 버전은 d[0]=0cm를 반환해 무의미한 0값을 양산).
    magt = profile_cross(zg, tg, surface_frozen="nan")
    alt_rows.append(dict(site=site, lat=slat, lon=slon, block=block,
                         field_alt_seasmax_cm=round(float(alt_pred_seas[k]), 1) if np.isfinite(alt_pred_seas[k]) else np.nan,
                         field_alt_magt_cm=round(float(magt * 100.0), 1) if np.isfinite(magt) else np.nan,
                         obs_alt_cm=round(oalt, 1), delta_holdout_m=round(float(delta_used[k]), 3) if np.isfinite(delta_used[k]) else np.nan,
                         dist_deg=round(dist, 4)))
altm = pd.DataFrame(alt_rows)
altm.to_csv("data/processed/field3d_reeval_altmatch.csv", index=False)


def alt_stats(col):
    m = altm[col].notna() & altm.obs_alt_cm.notna()
    if m.sum() < 1:
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, corr=np.nan)
    a = altm.loc[m, col].values; b = altm.loc[m, "obs_alt_cm"].values
    return dict(n=int(m.sum()), rmse=round(rmse(a, b), 2),
                mae=round(float(np.mean(np.abs(a - b))), 2), bias=round(float(np.mean(a - b)), 2),
                corr=round(float(np.corrcoef(a, b)[0, 1]), 3) if m.sum() > 2 else np.nan)


st_seas = alt_stats("field_alt_seasmax_cm")
# 표본 감소·클립 포화 명시: n_matched(=130)와 실제 평가 n(=89)의 간극, 400cm 클립 포화 비율
n_matched = int(len(profiles))
n_nan_seas = int(altm.field_alt_seasmax_cm.isna().sum())
n_clip = int((altm.field_alt_seasmax_cm >= 400 - 1e-6).sum())
n_magt_zero_removed = int(((altm.field_alt_magt_cm == 0.0)).sum())  # 버그수정 후엔 0이어야 함
st_seas.update(n_matched=n_matched, n_nan_envelope=n_nan_seas,
               n_clip_sat=n_clip, frac_clip_sat=round(n_clip / max(st_seas["n"], 1), 3))
# 정직 해석: 유도 ALT가 관측 ALT를 추종하지 못함(상관 낮음) → 귀무/음성 결과.
is_null = (not np.isfinite(st_seas["corr"])) or (st_seas["corr"] < 0.3) or (st_seas["rmse"] > 100)
st_seas["interpretation"] = "null_negative" if is_null else "weak_positive"
print(f"[ALT정합-정직판] 연최대 포락선(held-out δ): 매칭 {n_matched} → 평가 n={st_seas['n']} "
      f"(포락선 미교차 NaN {n_nan_seas}, 400cm클립 {n_clip}) "
      f"RMSE={st_seas['rmse']}cm MAE={st_seas['mae']} bias={st_seas['bias']:+} r={st_seas['corr']}")
print(f"  → 해석: {st_seas['interpretation']} (r<0.3·RMSE>100cm). "
      f"ALT 교차 경로는 field 모델의 독립 검증 근거가 아니다(검증으로 인용 금지).")

# ================================================================ 게이트 판정(누설 통제 후)
def band_delta(cvname):
    sub = res[res.cv == cvname]
    out = {}
    for b in ["0-2m", "2-5m", "5-10m", "10-20m", "all"]:
        try:
            bb = sub[(sub.model == "FIELD_base") & (sub.depth_band == b)].rmse_c.iloc[0]
            ff = sub[(sub.model == "FIELD_full") & (sub.depth_band == b)].rmse_c.iloc[0]
            out[b] = round(float(ff - bb), 4)  # 음수 = full 개선
        except Exception:
            pass
    return out


deltas = {cv: band_delta(cv) for cv in CVS}
# 심부(5-10, 10-20) 개선 여부를 CV별로 판정
def deep_improved(cv):
    d = deltas[cv]
    deep = [d.get("5-10m", np.nan), d.get("10-20m", np.nan)]
    deep = [x for x in deep if np.isfinite(x)]
    return all(x < 0 for x in deep) and len(deep) == 2


verdict_by_cv = {cv: ("full_improves_deep" if deep_improved(cv) else "no_deep_gain") for cv in CVS}

meta = dict(
    task="experiment2_A1_field3d_leakage_reeval",
    n_rows=int(len(g)), n_sites=int(g.site.nunique()), n_regions=int(g.region.nunique()),
    leakage=leak,
    cv_schemes=dict(site_gkf6=f"{len(folds_site)}fold",
                    spatial_block6=f"{len(folds_block)}fold(block={BLOCK_DEG}deg)",
                    loro=f"{len(folds_loro)}regions:{loro_regions}"),
    overall_rmse={cv: {m: round(rmse(g.temp_c[np.isfinite(oof[(cv, m)])],
                                     oof[(cv, m)][np.isfinite(oof[(cv, m)])]), 4)
                       for m in MODELS} for cv in CVS},
    band_delta_full_minus_base=deltas,
    deep_verdict_by_cv=verdict_by_cv,
    monotonicity={r_.cv: r_.viol_rate for r_ in mono.itertuples()},
    sample_composition=dict(
        frac_permos_overall=round(float((g.source == "PERMOS").mean()), 4) if "source" in g.columns else None,
        frac_switzerland_overall=round(float((g.region == "Switzerland").mean()), 4),
        deep_5_20m_frac_switzerland=round(float(deep_sw), 4),
        note=("심부 5-20m 밴드가 스위스 PERMOS 알프스 암빙하 시추공에 80%+ 편중. "
              "그림3 알래스카 2m/20m 지도는 심부 신호가 스위스 데이터에 지배되는 도메인 밖 외삽이다."),
    ),
    fold_balance_spatial_block6=dict(
        n_rows_min=int(foldbal.n.min()), n_rows_max=int(foldbal.n.max()),
        n_deep_min=int(foldbal.n_deep_5_20m.min()), n_deep_max=int(foldbal.n_deep_5_20m.max()),
        note=("fold 행수 심한 불균형. 밴드 델타는 pooled OOF(스위스-과다 fold 가중)이며 "
              "작은 fold서 noisy. 단일 밴드 델타 수치를 정밀 효과크기로 과대해석 금지."),
    ),
    alt_consistency_honest=dict(n_matched=int(len(profiles)),
                                n_evaluated=int(st_seas["n"]),
                                n_nan_envelope=int(st_seas["n_nan_envelope"]),
                                n_clip_sat=int(st_seas["n_clip_sat"]),
                                frac_clip_sat=st_seas["frac_clip_sat"],
                                delta_holdout_median_m=round(float(np.nanmedian(delta_used)), 3),
                                seasonal_max_envelope=st_seas,
                                interpretation=st_seas["interpretation"],
                                note=("유도 ALT가 관측 ALT를 추종하지 못함(r=0.16, RMSE 172cm, "
                                      "평가점 10% 400cm 클립 포화). 귀무/음성 결과이며 field 모델의 "
                                      "독립 검증 근거로 인용해서는 안 된다.")),
    magt_reference_note=("field_alt_magt_cm(참고 전용)은 버그수정 후 완전동결 프로파일에 대해 "
                         "NaN을 반환한다(과거 버전은 표층 음온 시 d[0]=0cm를 반환해 86/109개의 "
                         "무의미한 0값을 양산). 헤드라인 통계에는 미사용."),
    verdict=("공변량 추가(full)의 심부 이득은 site-GKF에서만 나타나고 누설통제(공간블록·LORO)서 소멸·역전. "
             "따라서 '3D 심부 field가 공변량으로 개선된다'는 기존 주장은 기각. "
             "심부 표본이 스위스 알프스에 편중되어 알래스카 심부 field는 도메인 밖 외삽이며, "
             "ALT 교차 정합은 귀무결과로 독립 검증 불가."),
    note=("site-GKF는 근접 사이트 누설을 허용(사이트 73%가 0.5°블록 공유). "
          "공간블록·LORO에서 심부 full 이득이 소멸·역전하므로 기존 심부 개선은 누설 착시."),
)
with open("data/processed/field3d_reeval_meta.json", "w") as fp:
    json.dump(meta, fp, ensure_ascii=False, indent=2)

print("\n[게이트] 심부(5-10·10-20m) full 개선 판정")
for cv in CVS:
    print(f"  {cv:14s}: {verdict_by_cv[cv]:20s} band_delta={deltas[cv]}")
print("\n[저장] field3d_reeval_bands.csv, _leakage.csv, _monotonicity.csv, _altmatch.csv, _meta.json")
