"""좌표 투영 유틸 — 경위도(EPSG:4326) <-> NSIDC 북극 polar stereographic(EPSG:3413, m)."""
import numpy as np
from pyproj import Transformer

_FWD = Transformer.from_crs(4326, 3413, always_xy=True)
_INV = Transformer.from_crs(3413, 4326, always_xy=True)


def to_xy(lon, lat):
    """경위도 -> (x, y) meters (EPSG:3413)."""
    x, y = _FWD.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def to_lonlat(x, y):
    """(x, y) meters (EPSG:3413) -> 경위도."""
    lon, lat = _INV.transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)
