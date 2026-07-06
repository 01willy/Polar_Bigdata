"""
⑤ Baseline 3D 보간 — borehole MAGT 점들로 3D 온도 체적(volume) 생성.

방법(층별/2.5D, 희소 borehole에 안정적):
  1) 각 borehole의 수직 MAGT 프로파일을 공통 깊이격자(0..max, DZ 간격)로 1D 보간
  2) 각 깊이층에서 수평(x,y) RBF 보간 → 수평 슬라이스
  3) 슬라이스 적층 → 3D 체적 T(x,y,z).  borehole 근방(거리<MAX_DIST)만 유지(외삽 억제)
  4) 0 °C 등온면(contour) + frozen body(T<0 threshold) 추출 → tri-mesh 저장

  * 희소(35 boreholes)라 정식 kriging variogram은 불안정(Webster-Oliver ~100 필요).
    안정적 RBF를 baseline으로 사용. 격자 공변량 확보 후 regression-kriging/PINN으로 격상.

산출:
  data/processed/volume_magt.vti          3D MAGT 체적(EPSG:3413 m, z=깊이 m)
  outputs/meshes/permafrost_zero_isotherm.vtp   0 °C 등온면
  outputs/meshes/frozen_body.vtp                동결부(MAGT<0) 표면
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from . import config as C
from . import geo

DZ = 2.5           # 깊이 격자 간격 (m)
NX = NY = 80       # 수평 격자 해상도
PAD = 40_000       # 격자 여백 (m)
MAX_DIST = 220_000  # borehole로부터 이 거리(m) 밖은 마스킹(외삽 억제)
RBF_SMOOTH = 1.0
VARIOGRAM = "linear"   # PyKrige 변동도(희소 데이터에 안정적)


def _profiles_xy():
    prof = pd.read_csv(C.PROFILES_CSV).dropna(subset=["magt", "lat", "lon", "depth"])
    x, y = geo.to_xy(prof["lon"].to_numpy(), prof["lat"].to_numpy())
    prof = prof.assign(x=x, y=y)
    return prof


def make_grid(prof):
    """공통 격자 구성. -> (gx, gy, levels, bx, by, grid_xy, far_mask)."""
    max_depth = float(prof["depth"].max())
    levels = np.arange(0.0, max_depth + DZ, DZ)
    bx, by = [], []
    for _bid, g in prof.groupby("borehole_id"):
        if len(g) >= 2:
            bx.append(g["x"].iloc[0]); by.append(g["y"].iloc[0])
    bx, by = np.array(bx), np.array(by)
    gx = np.linspace(bx.min() - PAD, bx.max() + PAD, NX)
    gy = np.linspace(by.min() - PAD, by.max() + PAD, NY)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")        # (NY,NX)
    grid_xy = np.column_stack([GX.ravel(), GY.ravel()])
    dist, _ = cKDTree(np.column_stack([bx, by])).query(grid_xy)
    far = dist.reshape(NY, NX) > MAX_DIST
    return gx, gy, levels, bx, by, grid_xy, far


def build_volume():
    prof = _profiles_xy()
    gx, gy, levels, bx, by, grid_xy, far = make_grid(prof)
    nz = len(levels)

    # borehole별 프로파일 → 깊이격자 보간
    bh = []
    for _bid, g in prof.groupby("borehole_id"):
        g = g.sort_values("depth")
        if len(g) < 2:
            continue
        vals = np.interp(levels, g["depth"].to_numpy(), g["magt"].to_numpy(),
                         left=np.nan, right=np.nan)
        bh.append((g["x"].iloc[0], g["y"].iloc[0], vals))

    vol = np.full((nz, NY, NX), np.nan)
    var = np.full((nz, NY, NX), np.nan)         # 크리깅 분산(불확실성)
    for li in range(nz):
        pts = np.array([(b[0], b[1]) for b in bh if not np.isnan(b[2][li])])
        vals = np.array([b[2][li] for b in bh if not np.isnan(b[2][li])])
        if len(pts) < 4:
            continue
        pred, ss = _krige_layer(pts, vals, gx, gy)
        if pred is None:                         # 크리깅 실패 시 RBF 폴백
            rbf = RBFInterpolator(pts, vals, kernel="linear", smoothing=RBF_SMOOTH)
            pred = rbf(grid_xy).reshape(NY, NX)
        pred[far] = np.nan
        vol[li] = pred
        if ss is not None:
            ss[far] = np.nan
            var[li] = ss

    print(f"  volume {vol.shape}  filled cells: {np.isfinite(vol).sum():,}/{vol.size:,}")
    return gx, gy, levels, vol, var


def _krige_layer(pts, vals, gx, gy):
    """한 깊이층 2D Ordinary Kriging -> (추정값(NY,NX), 분산(NY,NX)). 실패 시 (None,None)."""
    try:
        from pykrige.ok import OrdinaryKriging
        ok = OrdinaryKriging(pts[:, 0], pts[:, 1], vals, variogram_model=VARIOGRAM,
                             verbose=False, enable_plotting=False)
        z, ss = ok.execute("grid", gx, gy)
        return np.asarray(z), np.asarray(ss)
    except Exception:
        return None, None


def to_imagedata(gx, gy, levels, vol, var=None):
    """numpy 체적 -> pyvista ImageData (z=깊이 m, 아래로 +)."""
    import pyvista as pv
    nz, ny, nx = vol.shape
    img = pv.ImageData(dimensions=(nx, ny, nz),
                       spacing=(gx[1] - gx[0], gy[1] - gy[0], levels[1] - levels[0]),
                       origin=(gx[0], gy[0], 0.0))
    # ImageData 포인트 순서: x 최속 -> (nx,ny,nz) F-order. vol=[iz,iy,ix] -> transpose
    img.point_data["MAGT"] = vol.transpose(2, 1, 0).ravel(order="F")
    if var is not None:
        img.point_data["STDDEV"] = np.sqrt(np.clip(var, 0, None)).transpose(2, 1, 0).ravel(order="F")
    return img


def run():
    C.ensure_dirs()
    import pyvista as pv
    print("Building 3D MAGT volume (layer-wise Ordinary Kriging) ...")
    gx, gy, levels, vol, var = build_volume()
    img = to_imagedata(gx, gy, levels, vol, var)
    vti = C.PROCESSED / "volume_magt.vti"
    img.save(vti)
    np.savez_compressed(C.PROCESSED / "volume_grid.npz",
                        gx=gx, gy=gy, levels=levels, vol=vol, var=var)
    print(f"  saved {vti} + volume_grid.npz  (scalars: MAGT, STDDEV)")

    # 0 °C 등온면
    iso = img.contour([0.0], scalars="MAGT", method="flying_edges")
    if iso.n_points:
        iso.save(C.MESHES / "permafrost_zero_isotherm.vtp")
        print(f"  saved permafrost_zero_isotherm.vtp ({iso.n_points} pts)")
    else:
        print("  (0 °C 등온면 없음 — 관측 깊이대 전부 동결/비동결)")

    # frozen body (MAGT < 0)
    frozen = img.threshold(0.0, scalars="MAGT", invert=True)
    if frozen.n_points:
        frozen.extract_surface().save(C.MESHES / "frozen_body.vtp")
        finite = np.isfinite(img.point_data["MAGT"])
        frac = float((img.point_data["MAGT"][finite] < 0).mean())
        print(f"  saved frozen_body.vtp  (동결 비율 {frac:.0%} of filled cells)")
    return vti
