"""M1 부록: 검증 방식 비교 — 같은 자료(알래스카 13,606셀)·같은 모델로 CV 분할 방식만 바꿔 오차를 채점한다.

목적
    문헌의 무작위 분할(Gautam 2025 Sci Rep: 무작위 70/30, RF 검정 R² 0.24; Liu 2024 ERL: 무작위 10-fold R² 0.97)과
    본 연구의 0.5° 공간블록 6-fold가 같은 자료에서 얼마나 다른 수치를 내는지 정량화한다. Linnenbrink, Milà, Meyer (2024,
    GMD 17:5897) 의 kNNDM(k-fold nearest neighbour distance matching)을 절충 기준으로 포함한다: 예측 영역→학습 셀
    최근접 거리 분포(G_ij)와 CV의 검증→학습 최근접 거리 분포(G*_j)가 Wasserstein 거리 W로 가장 가까운 fold 구성을 고른다.

CV 방식 (모두 6-fold)
    random_cell            무작위 셀 K-fold (seed별 셔플)
    site_0.05              0.05° 격자 '사이트' 단위 GroupKFold (무작위 사이트 분할 모사; seed별 사이트 순열)
    block_0.5              0.5° 공간블록 GroupKFold (현행 규약과 같은 블록, seed별 블록 순열·셀 수 균형 배정)
    block_0.5_canonical    현행 규약 그대로 (polar.fidelity.spatial_block_splits, sklearn GroupKFold, 분할 결정적)
    block_1.0, block_2.0   1.0°·2.0° 공간블록
    knndm                  kNNDM (k-means 군집 q ∈ Q(로그 간격 100개, k..N) → 제1주성분 순서로 k fold 병합 → W 최소 q 선택)
    seed는 분할의 무작위 요소(셔플·그룹 순열·k-means 초기화)와 CatBoost 난수를 함께 바꾼다.

모델 (m1_core 재사용; 계수·표준화 통계는 fold 학습 셀에서만)
    stefan                 ALT = E·√TDD, E는 fold 학습 최소제곱 (λ=0)
    ridge                  x25 직접 능형 회귀 (α=10, fold 학습 중앙값 대체·표준화)
    catboost_lo            x25 직접 저용량 CatBoost (m1_core.fit_model 과 동일 초모수, thread_count 만 4)
    stefan_ridge_l075      Stefan 앵커 + 0.75·능형 잔차

예측 영역 (kNNDM prediction domain)
    report_alt_map_hires.py 의 지도 격자와 동일 정의: ERA5-Land nh_monthly_2015-2020.nc 월 기후값을 알래스카 창
    (geomap.ALASKA: 170–138°W, 59–72°N)에서 0.02° 로 이중선형 세분, 육지(ERA5-Land 유효) ∩ 영구동토(다년평균 MAAT < 0 °C)
    격자점에서 무작위 20,000점. 육지 전체 표본은 민감도용으로 함께 저장한다.

산출
    data/processed/m1/cv_scheme_comparison.csv          scheme × model × seed (풀링 RMSE·MAE·bias·R²·skill, fold 평균·SD)
    data/processed/m1/cv_scheme_comparison_folds.csv    fold 단위 지표
    data/processed/m1/cv_scheme_distances.csv           방식별 검증→학습 최근접 거리(km) 표본 + 예측 영역·LOO 분포
    data/processed/m1/cv_scheme_comparison_meta.json    정의·kNNDM 선택 결과·거리 요약·환경
    outputs/figures/m1/cv_scheme_comparison.{png,pdf}   (a) 방식별 RMSE, (b) 최근접 거리 ECDF

실행 (CPU 전용)
    python3 scripts/3_deep_learning/m1_cv_scheme_comparison.py            # 전체 (약 15–20분)
    python3 scripts/3_deep_learning/m1_cv_scheme_comparison.py --smoke    # 축소 실행(스크래치 폴더)
    python3 scripts/3_deep_learning/m1_cv_scheme_comparison.py --figure-only
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # CPU 전용. GPU 5–9는 다른 실험이 점유 중이므로 사용 금지.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, spatial_block_splits                         # noqa: E402
from polar.preprocessing import fold_prep                                        # noqa: E402
from polar.eval_metrics import all_metrics                                       # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, fit_coefs, anchor_pred, fit_model   # noqa: E402
from polar.geomap import ALASKA                                                  # noqa: E402

EARTH_KM = 6371.0088
NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2015-2020.nc"
K_FOLDS = 6
MODELS = ["stefan", "ridge", "catboost_lo", "stefan_ridge_l075"]
LAM_RESID = 0.75
SCHEMES = ["random_cell", "site_0.05", "block_0.5", "block_0.5_canonical", "block_1.0", "block_2.0", "knndm"]
FEATS = INPUT_SETS["x25"]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n-pred", type=int, default=20000, help="예측 영역 표본 수")
ap.add_argument("--pred-res", type=float, default=0.02, help="예측 영역 격자 해상도(°), 지도 스크립트 기본과 동일")
ap.add_argument("--n-q", type=int, default=100, help="kNNDM 후보 군집 수 Q 길이(로그 간격)")
ap.add_argument("--maxp", type=float, default=0.5, help="kNNDM: 한 fold 최대 비율(CAST 기본 0.5)")
ap.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "m1"))
ap.add_argument("--fig-dir", default=str(ROOT / "outputs" / "figures" / "m1"))
ap.add_argument("--figure-only", action="store_true")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

if args.smoke:
    sp = Path(os.environ.get("SCRATCH", "/tmp")) / "cv_scheme_smoke"
    args.out_dir = str(sp / "data"); args.fig_dir = str(sp / "fig")
    args.seeds, args.n_pred, args.n_q, args.pred_res = 1, 3000, 12, 0.1
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
FIG = Path(args.fig_dir); FIG.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(args.seeds))
TAG = "cv_scheme"


# ---------------------------------------------------------------- 거리·군집 도우미
def to_rad(lon, lat):
    return np.deg2rad(np.c_[lat, lon])


def nnd_km(query_rad, ref_rad, exclude_self=False):
    """query 각 점에서 ref 최근접 점까지 대원거리(km). exclude_self=True 면 query==ref(LOO)."""
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
    """그룹 단위 K-fold. 그룹이 N/k 보다 크면 크기 내림차순으로 먼저, 나머지는 무작위 순열로 '가장 작은 fold' 에 탐욕 배정
    (sklearn GroupKFold 의 크기 내림차순 탐욕 배정을 무작위화한 것; 큰 블록 처리 규칙은 같아 fold 균형이 유지된다)."""
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


def knndm(XY, rad, gij, k, seed, n_q, maxp, log):
    """Linnenbrink 2024 kNNDM (k-means). 반환 (folds, info)."""
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
    log(f"  [knndm seed {seed}] KS D={ks.statistic:.3f} p={ks.pvalue:.2e} · W_random={w_rand:.2f} km · "
        f"후보 {len(Q)} (가용 {sum(c['feasible'] for c in curve)}) · 최적 q={best[1]} W={best[0]:.2f} km · {time.time()-t0:.0f}s")
    info = dict(seed=seed, ks_D=float(ks.statistic), ks_p=float(ks.pvalue), clustered=bool(ks.pvalue < 0.05),
                W_random_km=w_rand, q_selected=best[1], W_selected_km=float(best[0]),
                random_would_match_better=bool(w_rand < best[0]), Q=[int(q) for q in Q], curve=curve,
                fold_sizes=[int(x) for x in np.bincount(best[2], minlength=k)],
                gj_median_km=float(np.median(gj)), gj_iqr_km=[float(np.percentile(gj, 25)), float(np.percentile(gj, 75))])
    return best[2], info, gj


# ---------------------------------------------------------------- 예측 영역
def prediction_domain(n_sample, res, seed, log):
    """알래스카 ALT 지도 격자(report_alt_map_hires.py 정의)에서 예측 영역 표본. 반환 dict(permafrost, land, meta)."""
    import xarray as xr
    from scipy.ndimage import distance_transform_edt
    _, lo0, lo1, la0, la1 = ALASKA
    pad = 0.6
    ds = xr.open_dataset(NC)
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    win = ds["t2m"].sel(latitude=slice(la1 + pad, la0 - pad), longitude=slice(lo0 - pad, lo1 + pad)).load()
    clim = win.assign_coords(month=win[tname].dt.month).groupby("month").mean(tname)
    t = clim.values.astype("float32")                        # (12, LAT, LON) K
    land = np.isfinite(t).all(axis=0)                       # ERA5-Land: 해양 = NaN
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
    ds.close()
    lon2d, lat2d = np.meshgrid(glon, glat)
    rng = np.random.RandomState(seed)

    def sample(mask):
        pts = np.c_[lon2d[mask], lat2d[mask]]
        sel = rng.choice(len(pts), min(n_sample, len(pts)), replace=False)
        return pts[sel]
    out = dict(permafrost=sample(pf), land=sample(land_f))
    meta = dict(source=str(NC.relative_to(ROOT)), definition="report_alt_map_hires.py 지도 격자와 동일: ERA5-Land 월 기후값 → "
                f"{res}° 이중선형 세분, 육지=ERA5-Land 유효, 영구동토=다년평균 MAAT<0 °C",
                extent_lon=[lo0, lo1], extent_lat=[la0, la1], res_deg=res, n_grid_land=int(land_f.sum()),
                n_grid_permafrost=int(pf.sum()), n_sample=int(len(out["permafrost"])), sample_seed=seed,
                main_domain="permafrost", land_domain_role="민감도(참고)")
    log(f"[예측 영역] 격자 {len(glon)}x{len(glat)} · 육지 {land_f.sum():,} · 영구동토 {pf.sum():,} → 표본 {len(out['permafrost']):,}")
    return out, meta


# ---------------------------------------------------------------- 모델
def catboost_lo(Xtr, ytr, Xte, seed):
    """m1_core.fit_model('catboost_lo') 와 같은 초모수, thread_count 만 4 (공유 서버 CPU 규약)."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=4)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte))


def predict_fold(tr, te, seed):
    k = fit_coefs(tr)
    Xtr = tr[FEATS].values.astype(np.float32); Xte = te[FEATS].values.astype(np.float32)
    ytr = tr[TARGET].values.astype(float)
    st_tr, st_te = anchor_pred("stefan", tr, k), anchor_pred("stefan", te, k)
    Xa, Xb = fold_prep(Xtr, Xte, nan_native=False)
    out = {"stefan": st_te,
           "ridge": fit_model("ridge", Xa, ytr, Xb, seed)["pred"],
           "stefan_ridge_l075": st_te + LAM_RESID * fit_model("ridge", Xa, ytr - st_tr, Xb, seed)["pred"],
           "catboost_lo": catboost_lo(Xtr, ytr, Xte, seed)}
    return out, float(k["E"])


# ---------------------------------------------------------------- 실행
def run():
    t_start = time.time()
    logs = []

    def log(s):
        print(s, flush=True); logs.append(s)

    df = load_base(ROOT / "data" / "processed")
    ak = df[df.macro.values == "Alaska"].reset_index(drop=True)
    ak = ak[eval_mask(ak)].reset_index(drop=True)
    n = len(ak)
    y = ak[TARGET].values.astype(float)
    rad = to_rad(ak.lon.values, ak.lat.values)
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    px, py = tf.transform(ak.lon.values, ak.lat.values)
    XY = np.c_[px, py] / 1000.0                              # 알래스카 알버스 등적 투영, km (k-means·PCA 용)
    log(f"[data] 알래스카 {n:,}셀 · 0.5° 블록 {ak.block.nunique()} · 0.05° 사이트 {len(np.unique(grid_key(ak.lat.values, ak.lon.values, 0.05)))}")

    pred, pred_meta = prediction_domain(args.n_pred, args.pred_res, 0, log)
    gij = nnd_km(to_rad(pred["permafrost"][:, 0], pred["permafrost"][:, 1]), rad)
    gij_land = nnd_km(to_rad(pred["land"][:, 0], pred["land"][:, 1]), rad)
    log(f"[G_ij] 영구동토 표본 중앙 {np.median(gij):.2f} km (IQR {np.percentile(gij, 25):.2f}–{np.percentile(gij, 75):.2f}) · "
        f"육지 전체 중앙 {np.median(gij_land):.2f} km")

    # fold 배정 (scheme, seed) → 벡터
    folds_all, knndm_info, gj = {}, [], None
    canon = np.empty(n, int)
    for fi, (_, te) in enumerate(spatial_block_splits(ak, n_splits=K_FOLDS)):
        canon[te] = fi
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        folds_all[("random_cell", seed)] = np.empty(n, int)
        for fi, (_, te) in enumerate(KFold(K_FOLDS, shuffle=True, random_state=seed).split(np.arange(n))):
            folds_all[("random_cell", seed)][te] = fi
        folds_all[("site_0.05", seed)] = group_folds_random(grid_key(ak.lat.values, ak.lon.values, 0.05), K_FOLDS, rng, n)
        folds_all[("block_0.5", seed)] = group_folds_random(ak.block.values, K_FOLDS, rng, n)
        folds_all[("block_0.5_canonical", seed)] = canon
        folds_all[("block_1.0", seed)] = group_folds_random(grid_key(ak.lat.values, ak.lon.values, 1.0), K_FOLDS, rng, n)
        folds_all[("block_2.0", seed)] = group_folds_random(grid_key(ak.lat.values, ak.lon.values, 2.0), K_FOLDS, rng, n)
        kf, info, gj = knndm(XY, rad, gij, K_FOLDS, seed, args.n_q, args.maxp, log)
        folds_all[("knndm", seed)] = kf
        knndm_info.append(info)

    # 거리 분포·모델 채점
    dist_rows = [pd.DataFrame(dict(scheme="prediction_permafrost", seed=0, dist_km=np.round(gij, 4))),
                 pd.DataFrame(dict(scheme="prediction_land", seed=0, dist_km=np.round(gij_land, 4))),
                 pd.DataFrame(dict(scheme="loo_train", seed=0, dist_km=np.round(gj, 4)))]
    dist_summary, rows, fold_rows = [], [], []
    for scheme in SCHEMES:
        for seed in SEEDS:
            folds = folds_all[(scheme, seed)]
            d = cv_nnd(folds, rad)
            w = float(wasserstein_distance(d, gij))
            fs = np.bincount(folds, minlength=K_FOLDS)
            dist_rows.append(pd.DataFrame(dict(scheme=scheme, seed=seed, dist_km=np.round(d, 4))))
            dist_summary.append(dict(scheme=scheme, seed=seed, W_km=round(w, 4), nnd_median_km=round(float(np.median(d)), 4),
                                     nnd_q25_km=round(float(np.percentile(d, 25)), 4), nnd_q75_km=round(float(np.percentile(d, 75)), 4),
                                     nnd_mean_km=round(float(d.mean()), 4), fold_sizes=[int(x) for x in fs]))
            preds = {m: np.full(n, np.nan) for m in MODELS}
            t0 = time.time()
            for f in range(K_FOLDS):
                te = folds == f
                out, E = predict_fold(ak[~te], ak[te], seed)
                for m in MODELS:
                    preds[m][te] = out[m]
                    mm = all_metrics(y[te], out[m])
                    fold_rows.append(dict(scheme=scheme, seed=seed, fold=f, model=m, n_test=int(te.sum()), E_train=round(E, 4),
                                          nnd_median_km=round(float(np.median(d[te])), 3), rmse_cm=mm["rmse_cm"], mae_cm=mm["mae_cm"],
                                          bias_cm=mm["bias_cm"], r2=mm["r2"]))
            for m in MODELS:
                mm = all_metrics(y, preds[m])
                fr = [r["rmse_cm"] for r in fold_rows if r["scheme"] == scheme and r["seed"] == seed and r["model"] == m]
                rows.append(dict(scheme=scheme, model=m, seed=seed, n_folds=K_FOLDS, n=mm["n"], rmse_cm=mm["rmse_cm"], mae_cm=mm["mae_cm"],
                                 bias_cm=mm["bias_cm"], r2=mm["r2"], target_sd_cm=mm["target_sd_cm"], skill_over_mean=mm["skill_over_mean"],
                                 rmse_fold_mean_cm=round(float(np.mean(fr)), 3), rmse_fold_sd_cm=round(float(np.std(fr, ddof=1)), 3),
                                 nnd_median_km=round(float(np.median(d)), 3), W_km=round(w, 3), max_fold_share=round(float(fs.max() / n), 4)))
            log(f"  [{scheme} seed {seed}] W={w:.2f} km · NND 중앙 {np.median(d):.2f} km · fold {fs.tolist()} · "
                + " · ".join(f"{m} {[r for r in rows if r['scheme']==scheme and r['seed']==seed and r['model']==m][0]['rmse_cm']:.2f}" for m in MODELS)
                + f" · {time.time()-t0:.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{TAG}_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / f"{TAG}_comparison_folds.csv", index=False)
    pd.concat(dist_rows, ignore_index=True).to_csv(OUT / f"{TAG}_distances.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                        # noqa: BLE001
        commit = "NA"
    meta = dict(
        stage="M1-appendix", tag=TAG, purpose="검증 방식(무작위·사이트·공간블록·kNNDM)이 같은 자료·모델에서 내는 오차 차이 정량화",
        data=dict(base="fidelity_base_v3.csv", soil="e5_soil_tdd_v3.csv", region="Alaska (macro)", n_cells=int(n),
                  n_blocks_0p5=int(ak.block.nunique()), n_sites_0p05=int(len(np.unique(grid_key(ak.lat.values, ak.lon.values, 0.05)))),
                  n_blocks_1p0=int(len(np.unique(grid_key(ak.lat.values, ak.lon.values, 1.0)))),
                  n_blocks_2p0=int(len(np.unique(grid_key(ak.lat.values, ak.lon.values, 2.0)))), target=TARGET,
                  target_mean_cm=round(float(y.mean()), 3), target_sd_cm=round(float(y.std()), 3)),
        k_folds=K_FOLDS, seeds=SEEDS, schemes=SCHEMES, models=MODELS, lam_resid=LAM_RESID, features="x25 (SHARED_CORE)",
        model_notes=dict(stefan="E 최소제곱, fold 학습 셀만", ridge="Ridge(alpha=10), fold_prep(중앙값 대체·표준화, 학습 통계)",
                         catboost_lo="iterations=200, lr=0.05, depth=3, l2=3, thread_count=4 (m1_core 는 8)",
                         stefan_ridge_l075="Stefan 앵커 + 0.75 × ridge 잔차"),
        scheme_notes=dict(random_cell="KFold(shuffle, random_state=seed)",
                          site_0p05="0.05° 격자 사이트 GroupKFold, 큰 그룹(>N/k) 크기순 선배정 후 무작위 순열 탐욕 균형 배정",
                          block_0p5="0.5° 블록(add_group_keys block), 위와 같은 배정", block_0p5_canonical="spatial_block_splits(GroupKFold) 결정적, seed 는 CatBoost 만",
                          block_1p0="1.0° 블록", block_2p0="2.0° 블록",
                          knndm="Linnenbrink 2024: KS(G_j>G_ij) → k-means q∈Q(log 100, k..N) → PC1 순 병합 → W 최소. 거리 haversine km, 군집은 EPSG:3338 km"),
        prediction_domain=pred_meta,
        gij_summary=dict(permafrost_median_km=float(np.median(gij)), permafrost_iqr_km=[float(np.percentile(gij, 25)), float(np.percentile(gij, 75))],
                         land_median_km=float(np.median(gij_land)), land_iqr_km=[float(np.percentile(gij_land, 25)), float(np.percentile(gij_land, 75))]),
        knndm=knndm_info, distance_summary=dist_summary,
        references=dict(knndm="Linnenbrink, Milà, Meyer (2024) GMD 17:5897-5912, doi:10.5194/gmd-17-5897-2024",
                        caveat="Wadoux et al. (2021) Ecol. Model. 457:109692: 공간 CV 도 설계 기반 확률표본 없이는 편향 없는 지도 정확도 추정이 아님",
                        literature=dict(gautam2025="Sci Rep, 무작위 70/30 사이트 분할, RF 검정 R² 0.24 (학습 0.84)",
                                        liu2024="ERL, 무작위 10-fold, RF R² 0.97")),
        env=dict(device="cpu", cuda_visible_devices="", threads=4, git_commit=commit, elapsed_s=round(time.time() - t_start, 1)),
        log=logs)
    (OUT / f"{TAG}_comparison_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"saved: {OUT / (TAG + '_comparison.csv')} ({len(res)}행) · {time.time()-t_start:.0f}s")


# ---------------------------------------------------------------- 그림
SCHEME_FIG = ["random_cell", "site_0.05", "block_0.5", "block_1.0", "block_2.0", "knndm"]
SCHEME_NAME = {"random_cell": "무작위 셀", "site_0.05": "무작위 사이트(0.05°)", "block_0.5": "공간블록 0.5°(현행)",
               "block_1.0": "공간블록 1.0°", "block_2.0": "공간블록 2.0°", "knndm": "kNNDM"}
# 냉색 6색(붉은 계열 없음). 인접쌍 CVD ΔE≥8·정상시 ΔE≥15 검증 통과(dataviz validate_palette, light).
SCHEME_COLOR = {"random_cell": "#a0a7b0", "site_0.05": "#0b7285", "block_0.5": "#8fb3dd",
                "block_1.0": "#3f6fa8", "block_2.0": "#17365d", "knndm": "#8468b5"}
SCHEME_LS = {"random_cell": (0, (1, 1.2)), "site_0.05": (0, (4, 1.5)), "block_0.5": "-", "block_1.0": "-", "block_2.0": "-", "knndm": (0, (5, 1.5, 1, 1.5))}
MODEL_NAME = {"stefan": "Stefan\n(λ=0)", "ridge": "능형 회귀\n(직접)", "catboost_lo": "저용량 CatBoost\n(직접)", "stefan_ridge_l075": "Stefan+능형 잔차\n(λ=0.75)"}
INK, MUTED = "#333333", "#6b6b6b"


def make_figure():
    from polar.plotstyle import use_polar
    plt = use_polar()
    res = pd.read_csv(OUT / f"{TAG}_comparison.csv")
    folds = pd.read_csv(OUT / f"{TAG}_comparison_folds.csv")
    dist = pd.read_csv(OUT / f"{TAG}_distances.csv")
    meta = json.loads((OUT / f"{TAG}_comparison_meta.json").read_text())
    W = {d["scheme"]: d["W_km"] for d in meta["distance_summary"] if d["seed"] == 0}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.22))
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.7); ax.spines[s].set_color("#7a7a7a")
        ax.grid(False)
        ax.tick_params(labelsize=9, length=2.5, width=0.7, color="#8a8a8a", labelcolor=INK)

    # (a) 방식별 풀링 RMSE (seed 평균), 오차 막대 = fold RMSE 표준편차
    ax1.grid(True, axis="y", color="#aab3bd", lw=0.5, alpha=0.35); ax1.set_axisbelow(True)
    ns, bw = len(SCHEME_FIG), 0.13
    top = 0.0
    for j, sc in enumerate(SCHEME_FIG):
        for i, m in enumerate(MODELS):
            v = res[(res.scheme == sc) & (res.model == m)].rmse_cm.mean()
            sd = folds[(folds.scheme == sc) & (folds.model == m)].rmse_cm.std(ddof=1)
            x = i + (j - (ns - 1) / 2) * bw
            ax1.bar(x, v, width=bw * 0.9, color=SCHEME_COLOR[sc], edgecolor="none", label=SCHEME_NAME[sc] if i == 0 else None, zorder=2)
            ax1.errorbar(x, v, yerr=sd, fmt="none", ecolor="#8a8a8a", elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
            ax1.text(x, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7.0, color=INK, rotation=90, zorder=4,
                     bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9))
            top = max(top, v + sd)
    ax1.set_xticks(range(len(MODELS))); ax1.set_xticklabels([MODEL_NAME[m] for m in MODELS], fontsize=9)
    ax1.set_ylabel("RMSE (cm)", fontsize=10.5, color=INK)
    ax1.set_ylim(0, top * 1.10)
    ax1.set_xlim(-0.5, len(MODELS) - 0.5)
    ax1.legend(loc="upper center", ncol=3, fontsize=8.3, frameon=False, columnspacing=1.1, handlelength=1.2, handletextpad=0.5,
               bbox_to_anchor=(0.5, -0.16))
    ax1.text(0.0, -0.50, "막대: 6-fold 풀링 RMSE의 seed 평균(n=3) · 오차 막대: fold RMSE 표준편차 · 알래스카 13,606셀",
             transform=ax1.transAxes, fontsize=7.8, color=MUTED, va="top")

    # (b) 최근접 학습 셀 거리 ECDF: 예측 영역 vs 방식별 검증→학습
    def ecdf(v):
        v = np.sort(np.asarray(v, float)); return v, np.arange(1, len(v) + 1) / len(v)
    x, yy = ecdf(dist[dist.scheme == "prediction_permafrost"].dist_km)
    ax2.plot(x, yy, color="#111111", lw=2.2, label="예측 영역(영구동토 격자)", zorder=5)
    x, yy = ecdf(dist[dist.scheme == "loo_train"].dist_km)
    ax2.plot(x, yy, color="#9a9a9a", lw=1.2, ls=(0, (2, 1.5)), label="학습 셀 간 LOO", zorder=4)
    for sc in SCHEME_FIG:
        x, yy = ecdf(dist[(dist.scheme == sc) & (dist.seed == 0)].dist_km)
        ax2.plot(x, yy, color=SCHEME_COLOR[sc], lw=1.8, ls=SCHEME_LS[sc], label=f"{SCHEME_NAME[sc]}  W={W[sc]:.1f} km", zorder=3)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 5e2); ax2.set_ylim(0, 1.02)
    ax2.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100]))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))      # mathtext 지수 표기 회피(글꼴 글리프)
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xlabel("최근접 학습 셀까지의 거리 (km)", fontsize=10.5, color=INK)
    ax2.set_ylabel("누적 비율", fontsize=10.5, color=INK)
    ax2.grid(True, color="#aab3bd", lw=0.5, alpha=0.3); ax2.set_axisbelow(True)
    ax2.legend(loc="upper center", ncol=2, fontsize=8.0, frameon=False, handlelength=2.2, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.20))
    ax2.text(0.0, -0.50, "W: 예측 영역 분포와의 Wasserstein 거리(작을수록 예측 상황과 유사) · seed 0 분할",
             transform=ax2.transAxes, fontsize=7.8, color=MUTED, va="top")
    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(-0.08, 1.04, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{TAG}_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{TAG}_comparison.png")


if __name__ == "__main__":
    if not args.figure_only:
        run()
    make_figure()
