"""보고서 대표 ALT 예측장: 지역 내 최고 구성으로 산출.

기존 report_alt_map_hires.py 는 물리식(Stefan) 단독이었다. 보고서가 지역 내 최고로
제시하는 구성은 "Stefan 앵커 + 능형회귀 잔차(입력 25종)"이므로 그것으로 다시 만든다.

  ŷ(x) = E·√TDD(x) + λ·g(c(x))

E와 잔차 g는 모두 알래스카 실측으로 적합하고, 격자에는 지형·기후·토양·위성 공변량을
깔아 적용한다. 격자 공변량 구축은 report_alt_downscale_demo.py 의 함수를 재사용한다.

실행: python3 report_alt_map_best.py [--res 0.02] [--lam 0.75]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))

from polar.plotstyle import use_polar                                    # noqa: E402
from polar.fidelity import macro_region, fit_stefan_E, CLIMATE, SOIL, TARGET  # noqa: E402
from polar.geomap import make_ax, mask_ocean, add_inset_locator          # noqa: E402
from polar.preprocessing import fold_prep                                # noqa: E402
from cmcrameri import cm as cmc                                          # noqa: E402
from sklearn.linear_model import Ridge                                   # noqa: E402
import report_alt_downscale_demo as dd                                   # 격자 구축 재사용

use_polar()
ap = argparse.ArgumentParser()
ap.add_argument("--res", type=float, default=0.02)
ap.add_argument("--lam", type=float, default=0.5)
args = ap.parse_args()

OUT = ROOT / "outputs" / "maps"
EXT = ("alaska", -168.5, -138.0, 58.5, 71.5)
GREY = "#5b6670"

# ---------------------------------------------------------------- 학습
# 격자 전역에 실제로 깔 수 있는 공변량(기후 8 + 토양 9 = 17종)만 사용한다.
# 공간블록 6-fold 검증에서 이 구성은 13.41 cm로, 25종 전체(13.33)와 사실상 같다.
FEATS = list(CLIMATE) + list(SOIL)
df = pd.read_csv(ROOT / "data" / "processed" / "fidelity_base.csv")
df["macro"] = macro_region(df)
ak = df[df.macro == "Alaska"].reset_index(drop=True)
y = ak[TARGET].values.astype(float)
s = ak["e5_sqrt_tdd"].values.astype(float)
E = fit_stefan_E(y, s)
resid = y - E * s
Xak = ak[FEATS].values.astype(np.float32)
print(f"[학습] Stefan 앵커 E={E:.4f} + 능형회귀 잔차(입력 {len(FEATS)}종, λ={args.lam}) · {len(ak)}셀")
print(f"       검증(공간블록 6-fold): 이 구성 13.41 cm · 물리식 단독 14.24 cm")

# ---------------------------------------------------------------- 격자 공변량
win = (EXT[1], EXT[2], EXT[3], EXT[4])
lat_t = np.arange(EXT[3], EXT[4] + args.res, args.res)
lon_t = np.arange(EXT[1], EXT[2] + args.res, args.res)
print(f"[격자] {len(lat_t)} x {len(lon_t)} = {len(lat_t)*len(lon_t):,} 셀 ({args.res}°)")

C = dd.climate_grid(lat_t, lon_t)
S = dd.soil_grid(lat_t, lon_t)
grid = {}
for c in FEATS:
    if c in C:   grid[c] = C[c]
    elif c in S: grid[c] = S[c]
    else: raise KeyError(c)

Xg = np.stack([grid[c].ravel() for c in FEATS], axis=1).astype(np.float32)
Xtr2, Xg2 = fold_prep(Xak, Xg, nan_native=False)
mdl = Ridge(alpha=10.0).fit(np.nan_to_num(Xtr2), resid)
g_grid = mdl.predict(np.nan_to_num(Xg2)).reshape(len(lat_t), len(lon_t))

sqrt_tdd = grid["e5_sqrt_tdd"]
phys = E * sqrt_tdd

# --- 적용가능 영역(AOA): 잔차 모델은 학습 공변량 공간 안에서만 유효하다.
# 표준화 공간에서 격자점과 가장 가까운 학습점까지의 거리를, 학습 표본 내부의
# 평균 최근접 거리로 정규화한 비유사도지수 DI를 쓰고, 임계는 학습 DI의 Q75+1.5IQR로 둔다.
from sklearn.neighbors import NearestNeighbors
mu, sd = np.nanmean(Xtr2, axis=0), np.nanstd(Xtr2, axis=0) + 1e-9
Ztr = np.nan_to_num((Xtr2 - mu) / sd)
Zg  = np.nan_to_num((Xg2  - mu) / sd)
# 같은 기후 화소를 공유하는 셀이 많아 중복 행이 생긴다. 중복을 제거해야
# 최근접 거리가 0으로 붕괴하지 않는다(보고서 AOA 정의와 동일한 처리).
Zu = np.unique(np.round(Ztr, 6), axis=0)
nn = NearestNeighbors(n_neighbors=2).fit(Zu)
d_tr, _ = nn.kneighbors(Zu); dbar = float(np.mean(d_tr[:, 1]))
di_tr = d_tr[:, 1] / dbar
print(f"[AOA] 학습 고유점 {len(Zu):,} · 평균 최근접거리 {dbar:.4f}")
q1, q3 = np.percentile(di_tr, [25, 75]); thr = q3 + 1.5 * (q3 - q1)
d_g, _ = nn.kneighbors(Zg, n_neighbors=1)
di_g = (d_g[:, 0] / dbar).reshape(len(lat_t), len(lon_t))

pf = grid["e5_maat"] < 0.0
inside = pf & (di_g <= thr)
alt  = np.where(inside, phys + args.lam * g_grid, np.nan)
phys = np.where(pf, phys, np.nan)
print(f"[적용가능] DI 임계 {thr:.2f} · 영구동토 격자 중 적용가능 "
      f"{inside.sum()/max(pf.sum(),1)*100:.1f}%")
print(f"[예측] 유효 {np.isfinite(alt).sum():,}셀 · 결합 평균 {np.nanmean(alt):.1f} cm "
      f"· 물리항만 {np.nanmean(phys):.1f} cm")

# ---------------------------------------------------------------- 렌더
obs = dd.load_obs(win) if hasattr(dd, "load_obs") else None
vmin, vmax = np.nanpercentile(alt, [2, 98])
if not np.isfinite(vmin) or not np.isfinite(vmax):
    raise SystemExit("[중단] 적용가능 격자가 없다. 임계 또는 공변량 정합을 확인하라.")
step = max(5, int(np.ceil((vmax - vmin) / 24 / 5) * 5))
lv = np.arange(np.floor(vmin / step) * step, np.ceil(vmax / step) * step + step, step)
cmap = cmc.oslo_r
norm = BoundaryNorm(lv, ncolors=cmap.N, extend="both")

fig = plt.figure(figsize=(9.4, 5.6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
make_ax(EXT, ax=ax, fig=fig, grid=True)
im = ax.pcolormesh(lon_t, lat_t, np.ma.masked_invalid(alt), cmap=cmap, norm=norm,
                   shading="auto", transform=ccrs.PlateCarree(), zorder=1.5,
                   rasterized=True)
# 적용가능 영역 밖은 단색으로 덮어 예측을 제시하지 않음을 분명히 한다.
outside = np.isfinite(phys) & ~np.isfinite(alt)
if outside.any():
    from matplotlib.colors import ListedColormap
    ax.pcolormesh(lon_t, lat_t, np.ma.masked_where(~outside, outside.astype(float)),
                  cmap=ListedColormap(["#c9d1d8"]), shading="auto",
                  transform=ccrs.PlateCarree(), zorder=2.2, rasterized=True)
mask_ocean(ax, zorder=3)
if obs is not None and len(obs):
    ax.scatter(obs.lon.values, obs.lat.values, s=2.2, c="#1f1f1f", linewidths=0,
               alpha=0.75, transform=ccrs.PlateCarree(), zorder=4,
               label=f"관측 지점 (n={len(obs):,})")
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    if outside.any():
        h.append(Patch(fc="#c9d1d8", ec="none")); l.append("적용가능 영역 밖(예측 제외)")
    ax.legend(h, l, loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)
add_inset_locator(fig, ax, EXT, size=0.20)
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, shrink=0.85, extend="both")
cb.set_label("ALT (cm)", fontsize=10, color=GREY)
cb.ax.tick_params(labelsize=9, labelcolor=GREY, length=2.5)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"alt_prediction_best.{ext}", dpi=300, bbox_inches="tight")
print(f"saved: {OUT}/alt_prediction_best.png")
