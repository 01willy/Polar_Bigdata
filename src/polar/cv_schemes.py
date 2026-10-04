"""XH 검증 사다리 전용 분할 함수(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.8, 하네스 보고서 2.1).

`scripts/3_deep_learning/m1_cv_scheme_comparison.py` 는 모듈 최상위에서 인자를 읽어 불러 쓸 수 없다. 그 파일의 kNNDM·예측 영역·묶음
함수를 이 모듈로 옮겼다. 아래 표시한 함수는 m1 의 본문을 그대로 옮긴 사본이고(인자 이름과 기본값을 바꾸지 않았다), 시험
`tests/test_x_xg_xh.py` 가 같은 입력에서 m1 함수와 같은 묶음을 내는지 대조한다.

  m1 사본: to_rad, nnd_km, cv_nnd, grid_key, group_folds_random, kmeans_labels, merge_clusters, knndm, prediction_domain(상자·파일 인자만 더함)
  추가    : prediction_domain_box(캐나다: 라벨 경계 상자 ± pad 의 ERA5-Land 육지 ∩ MAAT < 0 격자), prediction_domain_grid(레나 격자 파일),
            region_crs·project_km(k-평균·주성분용 투영 좌표), m1_folds(m1 의 무작위 셀·지점·블록 6겹 재현용 묶음, 난수 소비 순서 그대로)

라벨 값을 쓰지 않는다. 입력은 좌표와 기후 격자뿐이다. k-평균은 scikit-learn 판에 따라 결과가 달라지므로(계획 1절 재현 관문 행) kNNDM 묶음은
로컬에서 한 번 만들어 색인 파일(sha256)로 보낸다.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

EARTH_KM = 6371.0088
ROOT = Path(__file__).resolve().parents[2]
NC_DEFAULT = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2015-2020.nc"
ALASKA_BOX = (-170.0, -138.0, 59.0, 72.0)                  # geomap.ALASKA 의 (lon0, lon1, lat0, lat1)


# ---------------------------------------------------------------- 거리·군집 도우미(m1 사본)
def to_rad(lon, lat):
    return np.deg2rad(np.c_[lat, lon])


def nnd_km(query_rad, ref_rad, exclude_self=False):
    """query 각 점에서 ref 최근접 점까지 대원거리(km). exclude_self=True 면 query==ref(LOO)."""
    from sklearn.neighbors import BallTree
    tree = BallTree(ref_rad, metric="haversine")
    kq = 2 if exclude_self else 1
    d, _ = tree.query(query_rad, k=kq)
    return d[:, -1] * EARTH_KM


def cv_nnd(folds, rad):
    """fold 배정 벡터에 대해 검증 셀→다른 fold 학습 셀 최근접 거리(km), 셀 순서대로."""
    out = np.empty(len(rad))
    for f in np.unique(folds):
        te = folds == f
        out[te] = nnd_km(rad[te], rad[~te])
    return out


def grid_key(lat, lon, deg):
    return np.floor(lat / deg).astype(np.int64) * 100000 + np.floor(lon / deg).astype(np.int64)


def group_folds_random(groups, k, rng, n):
    """그룹 단위 K-fold. 그룹이 N/k 보다 크면 크기 내림차순으로 먼저, 나머지는 무작위 순열로 '가장 작은 fold' 에 탐욕 배정."""
    ug, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    big = np.where(cnt > n / k)[0]
    big = big[np.argsort(-cnt[big])]
    rest = np.setdiff1d(np.arange(len(ug)), big)
    order = np.concatenate([big, rng.permutation(rest)])
    fold_of = np.empty(len(ug), int); load = np.zeros(k, int)
    for g in order:
        f = int(np.argmin(load)); fold_of[g] = f; load[f] += cnt[g]
    return fold_of[inv]


def kmeans_labels(XY, q, seed):
    from sklearn.cluster import KMeans
    if q >= len(XY):
        return np.arange(len(XY))
    km = KMeans(n_clusters=q, n_init=1, max_iter=50, random_state=seed).fit(XY)
    return km.labels_


def merge_clusters(labels, XY, pc1, k, maxp):
    """q 군집 → k fold. 제1주성분 상 중심 순서로 인접 군집에 서로 다른 fold 를 순환 배정. N/k 초과 군집은 병합하지 않고
    단독 fold. 한 fold 가 maxp·N 을 넘거나 fold 수가 k 에 못 미치면 None(후보 제외)."""
    n = len(labels)
    q = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=q)
    cent = np.zeros((q, 2))
    np.add.at(cent, labels, XY)
    cent /= np.maximum(sizes, 1)[:, None]
    order = np.argsort(cent @ pc1)
    fold_of = np.full(q, -1)
    big = [c for c in order if sizes[c] > n / k]
    if len(big) >= k:
        return None
    for i, c in enumerate(big):
        fold_of[c] = i
    rest_folds = list(range(len(big), k))
    j = 0
    for c in order:
        if fold_of[c] < 0:
            fold_of[c] = rest_folds[j % len(rest_folds)]; j += 1
    folds = fold_of[labels]
    fs = np.bincount(folds, minlength=k)
    if fs.max() > maxp * n or (fs == 0).any():
        return None
    return folds


def knndm(XY, rad, gij, k, seed, n_q, maxp, log=None):
    """Linnenbrink 2024 kNNDM (k-means). 반환 (folds, info, gj). m1 본문과 같다(log 가 None 이면 출력하지 않는다)."""
    from scipy.stats import ks_2samp, wasserstein_distance
    n = len(XY)
    gj = nnd_km(rad, rad, exclude_self=True)
    ks = ks_2samp(gj, gij, alternative="greater")          # H0: G_j ≤ G_ij (군집 없음) vs H1: G_j > G_ij
    rng = np.random.RandomState(seed)
    rand_folds = rng.permutation(np.arange(n) % k)
    w_rand = float(wasserstein_distance(cv_nnd(rand_folds, rad), gij))
    Q = np.unique(np.round(np.logspace(np.log10(k), np.log10(n), n_q)).astype(int))
    Xc = XY - XY.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = vt[0]
    curve, best = [], None
    t0 = time.time()
    for q in Q:
        lab = kmeans_labels(XY, int(q), seed)
        folds = lab if q == k else merge_clusters(lab, XY, pc1, k, maxp)
        if folds is None or len(np.unique(folds)) < k:
            curve.append(dict(q=int(q), W_km=None, feasible=False)); continue
        w = float(wasserstein_distance(cv_nnd(folds, rad), gij))
        fs = np.bincount(folds, minlength=k)
        curve.append(dict(q=int(q), W_km=round(w, 4), feasible=True, max_fold_share=round(float(fs.max() / n), 4)))
        if best is None or w < best[0]:
            best = (w, int(q), folds.copy())
    if log is not None:
        log(f"  [knndm seed {seed}] 후보 {len(Q)} (가용 {sum(c['feasible'] for c in curve)}) · 선택 q={best[1]} · {time.time()-t0:.0f}s")
    info = dict(seed=seed, ks_D=float(ks.statistic), ks_p=float(ks.pvalue), clustered=bool(ks.pvalue < 0.05),
                W_random_km=w_rand, q_selected=best[1], W_selected_km=float(best[0]),
                random_would_match_better=bool(w_rand < best[0]), Q=[int(q) for q in Q], curve=curve,
                fold_sizes=[int(x) for x in np.bincount(best[2], minlength=k)],
                gj_median_km=float(np.median(gj)), gj_iqr_km=[float(np.percentile(gj, 25)), float(np.percentile(gj, 75))])
    return best[2], info, gj


# ---------------------------------------------------------------- 예측 영역
def _clim_window(nc, box, pad):
    """ERA5-Land 월 기후값(t2m, K)의 창. 반환 (clim DataArray, t (12, LAT, LON), land mask)."""
    import xarray as xr
    lo0, lo1, la0, la1 = box
    ds = xr.open_dataset(nc)
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    win = ds["t2m"].sel(latitude=slice(la1 + pad, la0 - pad), longitude=slice(lo0 - pad, lo1 + pad)).load()
    ds.close()
    clim = win.assign_coords(month=win[tname].dt.month).groupby("month").mean(tname)
    t = clim.values.astype("float32")                        # (12, LAT, LON) K
    land = np.isfinite(t).all(axis=0)                       # ERA5-Land: 해양 = NaN
    return clim, t, land


def prediction_domain(n_sample, res, seed, log=None, nc=NC_DEFAULT, box=ALASKA_BOX):
    """알래스카 ALT 지도 격자(m1 정의)에서 예측 영역 표본. 반환 (dict(permafrost, land), meta). m1 본문과 같다(nc·box 인자만 더함)."""
    import xarray as xr
    from scipy.ndimage import distance_transform_edt
    lo0, lo1, la0, la1 = box
    pad = 0.6
    clim, t, land = _clim_window(nc, box, pad)
    idx = distance_transform_edt(~land, return_distances=False, return_indices=True)
    filled = xr.DataArray(np.stack([t[m][tuple(idx)] for m in range(t.shape[0])]),
                          coords={"month": clim["month"].values, "latitude": clim["latitude"].values,
                                  "longitude": clim["longitude"].values}, dims=("month", "latitude", "longitude"))
    glon = np.arange(lo0, lo1 + res / 2, res)
    glat = np.arange(la0, la1 + res / 2, res)
    maat = np.nanmean(filled.interp(latitude=glat, longitude=glon, method="linear").values - 273.15, axis=0)
    lmask = xr.DataArray(land.astype("float32"), coords={"latitude": clim["latitude"].values,
                                                         "longitude": clim["longitude"].values},
                         dims=("latitude", "longitude"))
    land_f = lmask.interp(latitude=glat, longitude=glon, method="nearest").values > 0.5
    pf = land_f & (maat < 0.0)
    lon2d, lat2d = np.meshgrid(glon, glat)
    rng = np.random.RandomState(seed)

    def sample(mask):
        pts = np.c_[lon2d[mask], lat2d[mask]]
        sel = rng.choice(len(pts), min(n_sample, len(pts)), replace=False)
        return pts[sel]
    out = dict(permafrost=sample(pf), land=sample(land_f))
    meta = dict(source=str(Path(nc).name), definition=f"m1 정의: ERA5-Land 월 기후값 → {res}° 이중선형 세분, 육지=ERA5-Land 유효, "
                "영구동토=다년평균 MAAT<0 °C", extent_lon=[lo0, lo1], extent_lat=[la0, la1], res_deg=res, n_grid_land=int(land_f.sum()),
                n_grid_permafrost=int(pf.sum()), n_sample=int(len(out["permafrost"])), sample_seed=seed, main_domain="permafrost")
    if log is not None:
        log(f"[예측 영역] 격자 {len(glon)}x{len(glat)} · 육지 {int(land_f.sum()):,} · 영구동토 {int(pf.sum()):,} → 표본 {len(out['permafrost']):,}")
    return out, meta


def label_box(lat, lon, pad):
    """라벨 경계 상자 ± pad(°). 반환 (lon0, lon1, lat0, lat1)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (float(np.nanmin(lon) - pad), float(np.nanmax(lon) + pad), float(np.nanmin(lat) - pad), float(np.nanmax(lat) + pad))


def prediction_domain_box(box, nc=NC_DEFAULT, n_sample=None, seed=0):
    """캐나다 예측 영역(계획 2.8): 상자 안 ERA5-Land 원 격자(세분 없음)에서 육지 ∩ 다년평균 MAAT < 0 °C 인 격자점. n_sample 이 있고 격자점이
    그보다 많으면 RandomState(seed) 로 비복원 추출한다. 반환 (pts (N, 2) = [lon, lat], meta)."""
    lo0, lo1, la0, la1 = box
    clim, t, land = _clim_window(nc, box, 0.0)
    maat = np.nanmean(t - 273.15, axis=0)
    pf = land & (maat < 0.0)
    lat = clim["latitude"].values.astype(float); lon = clim["longitude"].values.astype(float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    pts = np.c_[lon2d[pf], lat2d[pf]]
    n_all = int(len(pts))
    if n_sample is not None and n_all > int(n_sample):
        pts = pts[np.sort(np.random.RandomState(seed).choice(n_all, int(n_sample), replace=False))]
    meta = dict(source=str(Path(nc).name), definition="ERA5-Land 원 격자, 육지 = 12개월 t2m 유효, 영구동토 = 다년평균 MAAT < 0 °C, 상자 = 라벨 경계 ± pad",
                extent_lon=[lo0, lo1], extent_lat=[la0, la1], n_grid_land=int(land.sum()), n_grid_permafrost=n_all, n_sample=int(len(pts)),
                sample_seed=int(seed) if (n_sample is not None and n_all > int(n_sample)) else None)
    return pts, meta


def prediction_domain_grid(path, n_sample=None, seed=0):
    """레나 예측 영역(계획 2.8): 지도 격자 파일의 (lon, lat). 반환 (pts, meta)."""
    import pandas as pd
    g = pd.read_csv(path, usecols=["lat", "lon"])
    pts = np.c_[g.lon.values.astype(float), g.lat.values.astype(float)]
    pts = pts[np.isfinite(pts).all(1)]
    n_all = int(len(pts))
    if n_sample is not None and n_all > int(n_sample):
        pts = pts[np.sort(np.random.RandomState(seed).choice(n_all, int(n_sample), replace=False))]
    meta = dict(source=str(Path(path).name), definition="WF5 지도 격자(map_lena/lena_grid_x25_v1)의 격자점", n_grid=n_all, n_sample=int(len(pts)))
    return pts, meta


# ---------------------------------------------------------------- 투영
def region_crs(region, lat=None, lon=None):
    """k-평균·주성분에 쓰는 투영. 알래스카는 m1 과 같은 EPSG:3338. 그 밖은 라벨 중심의 람베르트 등적 방위 투영(계획이 정하지 않아 구현에서 정함)."""
    if str(region) == "Alaska":
        return "EPSG:3338"
    la0 = float(np.round(np.nanmean(np.asarray(lat, float)), 2)); lo0 = float(np.round(np.nanmean(np.asarray(lon, float)), 2))
    return f"+proj=laea +lat_0={la0} +lon_0={lo0} +datum=WGS84 +units=m +no_defs"


def project_km(lon, lat, crs):
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    px, py = tf.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.c_[px, py] / 1000.0


# ---------------------------------------------------------------- m1 6겹 묶음(관문 (2) 재현용)
def m1_folds(lat, lon, block, seed, k=6):
    """m1 의 random_cell, site_0.05, block_0.5 묶음. m1 의 난수 소비 순서(RandomState(seed) 를 site → block 순서로 쓴다)를 그대로 따른다."""
    from sklearn.model_selection import KFold
    lat, lon, block = np.asarray(lat, float), np.asarray(lon, float), np.asarray(block)
    n = len(lat)
    out = {}
    rng = np.random.RandomState(seed)
    rc = np.empty(n, int)
    for fi, (_, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(np.arange(n))):
        rc[te] = fi
    out["random_cell"] = rc
    out["site_0.05"] = group_folds_random(grid_key(lat, lon, 0.05), k, rng, n)
    out["block_0.5"] = group_folds_random(block, k, rng, n)
    return out
