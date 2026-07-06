"""
공변량 기반 회귀 모델 (regression) + 검증.

- lobo_cv(): leave-one-borehole-out 교차검증으로 'IDW(거리)' vs 'RF(공변량)' vs 'RF(전체)' 비교.
  공변량이 단순 거리보간보다 나은지(또는 보완하는지) 정량 평가.
- build_rf_volume(): WorldClim 공변량으로 학습한 RandomForest로 3D MAGT 체적 예측.

공변량: WorldClim 연평균기온(bio1)·고도(elev)·기온계절성(bio4)·연강수(bio12).
"""
import numpy as np
import pandas as pd

from . import config as C
from . import geo
from . import covariates as cov
from . import interpolate as itp

FEATS_COV = ["depth", "wc_bio1", "wc_elev", "wc_bio4", "wc_bio12"]            # 위치 제외(공변량만)
FEATS_FULL = ["depth", "lat", "lon", "wc_bio1", "wc_elev", "wc_bio4", "wc_bio12"]


def load_training():
    prof = pd.read_csv(C.PROFILES_CSV).dropna(subset=["magt", "lat", "lon", "depth"])
    bcov = pd.read_csv(C.PROCESSED / "borehole_covariates.csv")
    wc = [c for c in bcov.columns if c.startswith("wc_")]
    prof = prof.merge(bcov[["borehole_id"] + wc], on="borehole_id", how="left").dropna(subset=wc)
    return prof


def _idw(train, test, cols=("lon", "lat", "depth"), k=8, power=2):
    from scipy.spatial import cKDTree
    cols = list(cols)
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    Xtr = ((train[cols] - mu) / sd).to_numpy()
    Xte = ((test[cols] - mu) / sd).to_numpy()
    tree = cKDTree(Xtr)
    d, idx = tree.query(Xte, k=min(k, len(Xtr)))
    d = np.maximum(d, 1e-6)
    w = 1 / d ** power
    return (w * train["magt"].to_numpy()[idx]).sum(1) / w.sum(1)


def lobo_cv():
    """leave-one-borehole-out CV. 메소드별 RMSE/MAE 출력 + 반환."""
    from sklearn.ensemble import RandomForestRegressor
    df = load_training()
    methods = {"IDW(거리)": [], "RF(공변량만)": [], "RF(전체)": []}
    for bid in df["borehole_id"].unique():
        tr, te = df[df.borehole_id != bid], df[df.borehole_id == bid]
        if len(te) < 1 or len(tr) < 20:
            continue
        methods["IDW(거리)"] += list(te["magt"] - _idw(tr, te))
        for key, feats in [("RF(공변량만)", FEATS_COV), ("RF(전체)", FEATS_FULL)]:
            rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                       n_jobs=-1, random_state=0).fit(tr[feats], tr["magt"])
            methods[key] += list(te["magt"] - rf.predict(te[feats]))
    print("  leave-one-borehole-out CV (낮을수록 좋음):")
    out = {}
    for k, e in methods.items():
        e = np.array(e)
        rmse, mae = np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))
        out[k] = dict(rmse=float(rmse), mae=float(mae), n=len(e))
        print(f"    {k:12s}  RMSE={rmse:5.2f} °C   MAE={mae:5.2f} °C   (n={len(e)})")
    pd.DataFrame(out).T.to_csv(C.PROCESSED / "cv_results.csv")
    return out


def build_rf_volume():
    """WorldClim 공변량 RandomForest로 3D MAGT 체적 예측 -> volume_magt_rf.vti/.npz."""
    import pyvista as pv
    from sklearn.ensemble import RandomForestRegressor
    df = load_training()
    x, y = geo.to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    df = df.assign(x=x, y=y)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                               n_jobs=-1, random_state=0).fit(df[FEATS_FULL], df["magt"])

    gx, gy, levels, bx, by, grid_xy, far = itp.make_grid(df)
    lon, lat = geo.to_lonlat(grid_xy[:, 0], grid_xy[:, 1])
    gwc = cov.sample_worldclim(lon, lat)
    NY, NX = itp.NY, itp.NX
    vol = np.full((len(levels), NY, NX), np.nan)
    base = pd.DataFrame({"lat": lat, "lon": lon, "wc_bio1": gwc["bio1"],
                         "wc_elev": gwc["elev"], "wc_bio4": gwc["bio4"], "wc_bio12": gwc["bio12"]})
    ok = base.notna().all(axis=1).to_numpy()           # 공변량 결측(해양 등) 제외
    for li, lev in enumerate(levels):
        feat = base.assign(depth=lev)[FEATS_FULL]
        pred = np.full(len(feat), np.nan)
        pred[ok] = rf.predict(feat[ok])
        pred = pred.reshape(NY, NX)
        pred[far] = np.nan
        vol[li] = pred

    img = itp.to_imagedata(gx, gy, levels, vol)
    img.save(C.PROCESSED / "volume_magt_rf.vti")
    np.savez_compressed(C.PROCESSED / "volume_grid_rf.npz",
                        gx=gx, gy=gy, levels=levels, vol=vol)
    iso = img.contour([0.0], scalars="MAGT", method="flying_edges")
    if iso.n_points:
        iso.save(C.MESHES / "permafrost_zero_isotherm_rf.vtp")
    print(f"  saved volume_magt_rf.vti (+npz), filled {np.isfinite(vol).sum():,} cells, "
          f"iso {iso.n_points} pts")
    # 변수 중요도
    imp = pd.Series(rf.feature_importances_, index=FEATS_FULL).sort_values(ascending=False)
    print("  변수 중요도:", ", ".join(f"{k}={v:.2f}" for k, v in imp.items()))
    return vol


def run():
    C.ensure_dirs()
    print("Leave-one-borehole-out CV (IDW vs RF) ...")
    lobo_cv()
    print("Building covariate RandomForest 3D volume ...")
    build_rf_volume()
    print("Done.")
