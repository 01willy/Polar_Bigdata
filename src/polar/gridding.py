"""점 관측 → 격자 필드 변환 (논문형 지도용). 솔버/렌더 분리: 격자·보간·마스크만.

viz 워크플로 스펙: 예측 지도는 모델 격자추론(실험 스크립트 책임), 이 모듈은 격자 생성·
관측 보간(참고용)·외삽 마스크. 보간면을 feature로 재사용 금지(누설).
"""
from __future__ import annotations
import numpy as np


def make_grid(extent, spacing_km=5.0):
    """extent(name, lo0, lo1, la0, la1)에서 등간격 경위도 격자. 반환 (lon2d, lat2d)."""
    _, lo0, lo1, la0, la1 = extent
    latm = np.deg2rad(0.5 * (la0 + la1))
    dlat = spacing_km / 111.0
    dlon = spacing_km / (111.0 * max(np.cos(latm), 0.2))
    lons = np.arange(lo0, lo1 + dlon, dlon)
    lats = np.arange(la0, la1 + dlat, dlat)
    return np.meshgrid(lons, lats)


def interp_obs(lon_obs, lat_obs, vals, lon2d, lat2d, method="linear", mask_km=25.0):
    """관측을 격자에 보간(참고용, '예측' 아님). convex hull 밖·관측 원거리는 NaN."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    pts = np.c_[np.asarray(lon_obs), np.asarray(lat_obs)]
    v = np.asarray(vals, float)
    m = np.isfinite(v)
    grid = griddata(pts[m], v[m], (lon2d, lat2d), method=method)
    if mask_km is not None:                     # 관측 원거리 격자 절단
        latm = np.deg2rad(np.nanmean(lat_obs))
        tree = cKDTree(np.c_[pts[m, 0] * np.cos(latm) * 111.0, pts[m, 1] * 111.0])
        d, _ = tree.query(np.c_[(lon2d * np.cos(latm) * 111.0).ravel(), (lat2d * 111.0).ravel()])
        grid = np.where(d.reshape(lon2d.shape) > mask_km, np.nan, grid)
    return grid


def grid_predict(model_predict, lon2d, lat2d, covariate_fn):
    """격자 각 셀에 공변량을 만들어 모델 추론(논문형 '예측 지도'). covariate_fn(lon,lat)→X."""
    shape = lon2d.shape
    X = covariate_fn(lon2d.ravel(), lat2d.ravel())
    pred = model_predict(X)
    return np.asarray(pred).reshape(shape)
