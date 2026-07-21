"""트랙 α (P2a): 3D 지중온도장을 전 공변량으로 추정하고 climate+depth 대비 개선·ALT 정합 평가.

배경: 기존 vol_thermal_field_alaska.py는 기후8(e5_*)+깊이 특징만 사용한다. 지형(dem_*)·CCI는 전지구
커버라 지중온도 시추공 위치에도 부착 가능. InSAR/PolSAR는 알래스카 전용이라 시추공 위치(주로 스위스·
러시아)엔 결측이므로 이 트랙에서는 제외(사유 로그). 결측 공변량은 지시자+train fold 중앙값 대체.

구성:
 1) 라벨 = ground_temp_all.csv (site,lat,lon,depth_m,temp_c,region), 깊이 0-30m·온도 -25~25℃ 필터.
    공변량: 기후8(e5_*, vol_thermal 방식 nh_monthly groupby month derive) + 지형6(dem_*, Copernicus DEM
    온디맨드) + CCI(cci_alt, prior). 깊이 특징: depth_m, log1p(depth), 푸리에 10개.
 2) 두 GBM 조건장 비교: FIELD_base(기후8+깊이) vs FIELD_full(기후8+지형6+CCI+깊이).
    평가 = site-disjoint GroupKFold(site) 6, 깊이별 RMSE(0-2/2-5/5-10/10-20m) + 전체.
 3) ALT 유도·정합: 각 사이트 예측 온도장이 0℃를 지나는 깊이 = field-ALT(연평균 프록시).
    관측 ALT(alt_global·envelope)가 근처(0.05°)에 있는 사이트에서 field-ALT vs 관측 ALT 산점·RMSE.

산출:
 - data/processed/p2a_field_results.csv (model,cv,depth_band,n,rmse_c,r2)
 - data/processed/p2a_alt_consistency.csv
 - data/processed/p2a_field_meta.json

주의: OOF 온도장으로 field-ALT를 계산한다(leakage 회피). 게이트: full이 base 대비 깊이별 RMSE 개선하면
채택. field-ALT가 관측 ALT와 정합(RMSE)하는지 보고.
"""
import os, sys, json, glob, calendar, warnings
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

ROOT = "/home/willy010313/Polar_Bigdata"
os.chdir(ROOT)
DEM_DIR = "data/raw/dem"
KEY = 4                       # 위경도 반올림 자릿수(~11m) — DEM 위치 키
NFOLD = 6
RNG = 0

# ------------------------------------------------------------------ 유틸
def fourier(dm):
    """vol_thermal 방식: 깊이 정규화(dm/30) 후 5 하모닉 sin/cos = 10열."""
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

# ------------------------------------------------------------------ (1) 라벨
g = pd.read_csv("data/processed/ground_temp_all.csv")
g = g[(g.depth_m > 0) & (g.depth_m <= 30) & (g.temp_c > -25) & (g.temp_c < 25)].reset_index(drop=True)
print(f"[label] 필터 후 {len(g):,}행, {g.site.nunique()} 사이트, {g.region.nunique()} 지역")

# ------------------------------------------------------------------ 기후8(e5_*) — vol_thermal 방식
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
print(f"[era5] 기후8 부착, e5_maat 결측 제거 {n0-len(g)}행 → {len(g):,}행")

# ------------------------------------------------------------------ 지형6(dem_*) — Copernicus DEM 온디맨드
DEMF = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
def tile_name(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"

# 사이트 고유 위치별 지형특징(terrain_features_dem.py 로직: elev/slope/aspect/tpi/rough)
sites = g.drop_duplicates("site")[["site", "lat", "lon"]].reset_index(drop=True)
sites["tlat"] = np.floor(sites.lat).astype(int)
sites["tlon"] = np.floor(sites.lon).astype(int)
demvals = {f: np.full(len(sites), np.nan) for f in DEMF}
has_dem = np.zeros(len(sites), dtype=bool)
mperdeg = 111320.0
Wp = 33; Hp = Wp // 2       # 지형특징 계산 창(≈1km)
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

# ------------------------------------------------------------------ CCI(cci_alt) — enrich_cci_cell.py 방식(사이트 위치)
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
      f"({100*uniq.cci_valid.mean():.0f}%), 중앙 {np.nanmedian(cci_site):.1f}cm")
g = g.merge(uniq[["site", "cci_alt", "cci_valid"]], on="site", how="left")

# InSAR/PolSAR 제외 사유: 알래스카 전용 산출물이라 지중온도 시추공(주로 스위스·러시아) 위치엔 전면 결측.
print("[exclude] InSAR5·PolSAR3: 알래스카 전용 → 시추공 위치 전면 결측이라 제외(지시자 무의미).")

# ------------------------------------------------------------------ 깊이 특징
g["logd"] = np.log1p(g.depth_m)
FF = fourier(g.depth_m.values)
FFn = [f"ff{i}" for i in range(FF.shape[1])]
for i, n in enumerate(FFn):
    g[n] = FF[:, i].astype(np.float32)
DEPTHF = ["depth_m", "logd"] + FFn
CLIMF = E5F
TERRF = DEMF + ["has_dem", "cci_alt", "cci_valid"]   # 지형6 + CCI(+지시자)

FEAT_BASE = CLIMF + DEPTHF
FEAT_FULL = CLIMF + DEMF + ["cci_alt"] + DEPTHF      # 결측 지시자(has_dem/cci_valid)는 아래에서 별도 추가
IND = ["has_dem", "cci_valid"]                        # 결측 지시자 플래그

print(f"[feat] base {len(FEAT_BASE)}개, full {len(FEAT_FULL)+len(IND)}개(지형6+CCI+지시자)")

# ------------------------------------------------------------------ (2) site-disjoint GroupKFold 6 평가
groups = g.site.values
gkf = GroupKFold(n_splits=NFOLD)
oof = {}   # model -> oof 예측
for model, feats, inds in [("FIELD_base", FEAT_BASE, []), ("FIELD_full", FEAT_FULL, IND)]:
    cols = feats + inds
    pred = np.full(len(g), np.nan)
    for tr, te in gkf.split(g, groups=groups):
        Xtr = g.iloc[tr][cols].copy(); Xte = g.iloc[te][cols].copy()
        # 결측 공변량: train fold 중앙값 대체(+지시자는 이미 별도 컬럼). NaN 네이티브 라우팅 회피(P1 교훈).
        for c in cols:
            med = Xtr[c].median()
            if not np.isfinite(med):
                med = 0.0
            Xtr[c] = Xtr[c].fillna(med); Xte[c] = Xte[c].fillna(med)
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=RNG)
        m.fit(Xtr.values, g.iloc[tr].temp_c.values)
        pred[te] = m.predict(Xte.values)
    oof[model] = pred
    print(f"[cv] {model}: 전체 RMSE={rmse(g.temp_c, pred):.3f}°C R2={r2(g.temp_c, pred):.3f}")

# 깊이별 결과표
BANDS = [("0-2m", 0, 2), ("2-5m", 2, 5), ("5-10m", 5, 10), ("10-20m", 10, 20)]
rows = []
for model in ["FIELD_base", "FIELD_full"]:
    pred = oof[model]
    for name, lo_d, hi_d in BANDS + [("all", 0, 30)]:
        mk = (g.depth_m > lo_d) & (g.depth_m <= hi_d)
        if mk.sum() < 5:
            continue
        rows.append(dict(model=model, cv="site_gkf6", depth_band=name,
                         n=int(mk.sum()),
                         rmse_c=round(rmse(g.temp_c[mk], pred[mk]), 4),
                         r2=round(r2(g.temp_c[mk], pred[mk]), 4)))
res = pd.DataFrame(rows)
res.to_csv("data/processed/p2a_field_results.csv", index=False)
print("\n[결과] 깊이별 RMSE (site-disjoint GKF6)")
piv = res.pivot(index="depth_band", columns="model", values="rmse_c")
piv["delta(full-base)"] = (piv["FIELD_full"] - piv["FIELD_base"]).round(4)
piv["improve%"] = (100 * (piv["FIELD_base"] - piv["FIELD_full"]) / piv["FIELD_base"]).round(1)
order = ["0-2m", "2-5m", "5-10m", "10-20m", "all"]
print(piv.reindex([o for o in order if o in piv.index]).to_string())

# ------------------------------------------------------------------ (3) field-ALT 유도·정합
# 두 가지 field-ALT 정의를 구분해 보고한다.
#  (A) MAGT(연평균) 등온선: OOF 예측 온도장(연평균 프록시)이 0℃를 지나는 깊이.
#      = 영구동토 상단(permafrost table)/MAGT 등온선. 계절 최대융해깊이(ALT)와 물리적으로 다름.
#      (vol_thermal_field_alaska.py 6행 주석 참고: 연평균장≠계절 ALT).
#  (B) 연최대 포락선(seasonal-max) 등온선: 표층 계절진폭 A=(e5_twarm-e5_tcold)/2를
#      감쇠파 exp(-z/δ)로 깊이에 실어 T_max(z)=T_annualmean(z)+A·exp(-z/δ)를 만들고 0℃ 교차.
#      = 관측 ALT(계절 최대융해)와 대응하는 물리량. t_max 관측이 없어 감쇠파 근사를 사용(task 지침).
#      감쇠깊이 δ는 관측 ALT에 대해 단일 스칼라로 보정(아래 grid search)한다. 이는 온도장 자체가
#      아니라 연평균→계절최대 변환 계수 1개만 관측에 맞추는 물리 근사 보정(진단용)이다.

def profile_cross(depths, temps):
    """조밀 그리드로 만든 (깊이,온도) 프로파일에서 양→음 첫 교차 깊이(선형보간). 없으면 NaN."""
    d = np.asarray(depths); t = np.asarray(temps)
    if len(d) < 2:
        return np.nan
    if t[0] <= 0:
        return float(d[0])
    for a in range(len(d) - 1):
        if t[a] > 0 >= t[a + 1]:
            frac = t[a] / (t[a] - t[a + 1])
            return float(d[a] + frac * (d[a + 1] - d[a]))
    return np.nan

def zero_cross_depth(depths, temps):
    """(A) MAGT 등온선: 관측 깊이의 OOF 예측을 정렬해 0℃ 교차(연평균 프록시)."""
    idx = np.argsort(depths)
    return profile_cross(np.asarray(depths)[idx], np.asarray(temps)[idx])

# 관측 ALT 소스(전지구): alt_global(CALM, 다년) + envelope. 사이트 근처(0.05°) 최근접.
obs_alt = []
ag = pd.read_csv("data/processed/alt_global.csv")[["lat", "lon", "alt_cm"]].dropna()
ag = ag.groupby([ag.lat.round(4), ag.lon.round(4)]).alt_cm.mean().reset_index()
ag.columns = ["lat", "lon", "alt_cm"]
obs_alt.append(ag)
try:
    ev = pd.read_csv("data/processed/alt_gtnp_envelope_cell.csv")[["lat", "lon", "alt_cm"]].dropna()
    obs_alt.append(ev)
except Exception:
    pass
obs = pd.concat(obs_alt, ignore_index=True)
obs_ll = obs[["lat", "lon"]].values
obs_alt_cm = obs.alt_cm.values

# 사이트별 프로파일·진폭·MAGT-ALT·관측매칭을 먼저 수집. 감쇠깊이 δ는 관측 ALT에 대해 단일 스칼라 보정.
profiles = []   # (site, zg, tg_annualmean, amp, magt_alt, obs_alt, dist)
for site, sg in g.groupby("site"):
    pred = oof["FIELD_full"][sg.index]        # full 모델 OOF
    dpt = sg.depth_m.values
    fa_magt = zero_cross_depth(dpt, pred)     # (A) MAGT 등온선(연평균 0℃ 교차)
    order = np.argsort(dpt)
    ds_ = dpt[order]; ts_ = pred[order]
    zg = np.linspace(0.0, min(30.0, float(ds_.max())), 300)
    tg = np.interp(zg, ds_, ts_)              # 연평균 프로파일 보간
    amp = max(0.0, (float(sg.e5_twarm.iloc[0]) - float(sg.e5_tcold.iloc[0])) / 2.0)
    slat, slon = sg.lat.iloc[0], sg.lon.iloc[0]
    d = np.hypot(obs_ll[:, 0] - slat, obs_ll[:, 1] - slon)
    j = int(np.argmin(d))
    if d[j] > 0.05:                           # 0.05° 이내 관측 ALT 없음 → 스킵
        continue
    profiles.append((site, round(slat, 4), round(slon, 4), zg, tg, amp,
                     fa_magt, float(obs_alt_cm[j]), float(d[j])))

def seas_alt(zg, tg, amp, delta):
    """(B) 연최대 포락선: T_max(z)=T_annualmean(z)+A·exp(-z/δ)의 0℃ 교차(감쇠파 근사)."""
    return profile_cross(zg, tg + amp * np.exp(-zg / delta))

# 감쇠깊이 δ 보정: 관측 ALT에 대해 seasonal-max ALT의 RMSE 최소화(단일 스칼라, 물리 근사 보정용 진단).
obs_arr = np.array([p[7] for p in profiles])
def alt_rmse_for_delta(delta):
    est = np.array([seas_alt(p[3], p[4], p[5], delta) for p in profiles]) * 100.0
    m = np.isfinite(est)
    if m.sum() < 5:
        return np.inf
    return np.sqrt(np.mean((est[m] - obs_arr[m]) ** 2))
grid = np.linspace(0.2, 6.0, 59)
delta_fit = float(grid[int(np.argmin([alt_rmse_for_delta(dv) for dv in grid]))])
print(f"[ALT정합] 감쇠깊이 δ 보정값 = {delta_fit:.2f} m (관측 ALT에 대한 RMSE 최소, 단일 스칼라)")

cons_rows = []
for site, slat, slon, zg, tg, amp, fa_magt, oalt, dist in profiles:
    fa_seas = seas_alt(zg, tg, amp, delta_fit)
    cons_rows.append(dict(site=site, lat=slat, lon=slon,
                          field_alt_magt_cm=round(fa_magt * 100.0, 1) if np.isfinite(fa_magt) else np.nan,
                          field_alt_seasmax_cm=round(fa_seas * 100.0, 1) if np.isfinite(fa_seas) else np.nan,
                          obs_alt_cm=round(oalt, 1),
                          dist_deg=round(dist, 4)))
cons = pd.DataFrame(cons_rows)
cons.to_csv("data/processed/p2a_alt_consistency.csv", index=False)

def alt_stats(col):
    m = cons[col].notna() & cons.obs_alt_cm.notna()
    if m.sum() < 1:
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, corr=np.nan)
    a = cons.loc[m, col].values; b = cons.loc[m, "obs_alt_cm"].values
    return dict(n=int(m.sum()), rmse=rmse(a, b),
                mae=float(np.mean(np.abs(a - b))), bias=float(np.mean(a - b)),
                corr=float(np.corrcoef(a, b)[0, 1]) if m.sum() > 2 else np.nan)

st_magt = alt_stats("field_alt_magt_cm")
st_seas = alt_stats("field_alt_seasmax_cm")
print(f"\n[ALT정합] 매칭 사이트 {len(cons)}개")
print(f"  (A) MAGT 등온선        : n={st_magt['n']} RMSE={st_magt['rmse']:.1f}cm "
      f"MAE={st_magt['mae']:.1f} bias={st_magt['bias']:+.1f} r={st_magt['corr']:.3f}")
print(f"  (B) 연최대 포락선(δ보정): n={st_seas['n']} RMSE={st_seas['rmse']:.1f}cm "
      f"MAE={st_seas['mae']:.1f} bias={st_seas['bias']:+.1f} r={st_seas['corr']:.3f}")
print("  주: (A)는 MAGT 등온선(영구동토 상단)이라 계절 ALT와 물리적으로 다름. "
      "관측 ALT 대응은 (B) 연최대 포락선(감쇠파, δ 단일보정).")

# ------------------------------------------------------------------ 메타 저장
meta = dict(
    track="alpha_P2a",
    n_rows=int(len(g)), n_sites=int(g.site.nunique()), n_regions=int(g.region.nunique()),
    features=dict(base=FEAT_BASE, full=FEAT_FULL + IND,
                  excluded=["insar5", "polsar3"],
                  exclude_reason="InSAR/PolSAR는 알래스카 전용 → 시추공 위치 전면 결측"),
    dem=dict(sites_with_dem=int(has_dem.sum()), sites_total=int(len(sites)),
             tiles_ok=int(n_tile_ok), tiles_miss=int(n_tile_miss)),
    cci=dict(valid_sites=int(uniq.cci_valid.sum()), total_sites=int(len(uniq))),
    cv="site_disjoint_GroupKFold_6",
    overall_rmse=dict(base=round(rmse(g.temp_c, oof["FIELD_base"]), 4),
                      full=round(rmse(g.temp_c, oof["FIELD_full"]), 4)),
    depth_band_rmse={m: {r_.depth_band: r_.rmse_c
                         for r_ in res[res.model == m].itertuples()}
                     for m in ["FIELD_base", "FIELD_full"]},
    alt_consistency=dict(
        n_matched=int(len(cons)),
        match_tol_deg=0.05,
        damp_delta_fit_m=round(delta_fit, 3),
        magt_isotherm={k: (None if not np.isfinite(v) else round(v, 2))
                       for k, v in st_magt.items()},
        seasonal_max_envelope={k: (None if not np.isfinite(v) else round(v, 2))
                               for k, v in st_seas.items()},
        note=("(A) MAGT 등온선=영구동토 상단(계절 ALT와 물리적으로 다름). "
              "(B) 연최대 포락선(감쇠파 δ 단일보정)=계절 최대융해깊이 대응. 관측 ALT 정합은 (B) 기준."),
    ),
)
# 게이트 판정: full이 base 대비 깊이별 RMSE 개선(전체 또는 다수 밴드)이면 채택
band_delta = {r_.depth_band: (res[(res.model == "FIELD_base") & (res.depth_band == r_.depth_band)].rmse_c.iloc[0]
                              - r_.rmse_c)
              for r_ in res[res.model == "FIELD_full"].itertuples()}
n_improved = sum(1 for b, d in band_delta.items() if b != "all" and d > 0)
overall_improved = meta["overall_rmse"]["full"] < meta["overall_rmse"]["base"]
meta["gate"] = dict(band_delta_cm={k: round(v, 4) for k, v in band_delta.items()},
                    n_bands_improved=int(n_improved),
                    overall_improved=bool(overall_improved),
                    verdict="ADOPT_full" if (overall_improved and n_improved >= 2) else "KEEP_base")
with open("data/processed/p2a_field_meta.json", "w") as fp:
    json.dump(meta, fp, ensure_ascii=False, indent=2)
print(f"\n[게이트] {meta['gate']['verdict']} "
      f"(전체개선={overall_improved}, 개선밴드={n_improved}/4)")
print("[저장] p2a_field_results.csv, p2a_alt_consistency.csv, p2a_field_meta.json")
