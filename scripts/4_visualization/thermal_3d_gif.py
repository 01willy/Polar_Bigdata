"""직관적 3D + GIF: 알래스카 얕은(0~20m) 지중 열구조.
GTN-P 35개 시추공의 깊이별 연평균지중온도(MAGT)를 깊이별 공간보간 → 3D 볼륨.
산출(3종):
  animations/alaska_thermal_depthscan.gif  깊이 0→20m 슬라이스 스캔(활성층→영구동토)
  volumes_3d/alaska_thermal_3d.png         깊이 스택 3D 열구조
  maps/alaska_permafrost_table.png         영구동토 상단면(0°C 도달 깊이) 2D 지도
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, tnorm, FROZEN, add_cbar, style_geo
from polar.outputs import animpath, volpath, mappath
plt = use_polar()

# 이 matplotlib 버전은 cmcrameri 컬러맵을 'cmc.batlow'로만 등록하므로,
# plotstyle의 기본 image.cmap='batlow'(scatter 등 기본 조회)가 실패한다.
# 공용 모듈 수정 없이 스크립트 국소로 'batlow' 별칭만 등록(시각화 전용, 로직 무관).
try:
    import matplotlib
    from cmcrameri import cm as _cmc
    if "batlow" not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmc.batlow, name="batlow")
except Exception:
    pass

# ---------- 1) 시추공 MAGT → 표준 깊이 ----------
prof = pd.read_csv("data/processed/borehole_profiles.csv")
prof = prof.dropna(subset=["magt", "lat", "lon", "depth"])
TARGETS = [0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20]
rows = []
for bid, g in prof.groupby("borehole_id"):
    g = g.sort_values("depth")
    if g.depth.max() < 1:
        continue
    for d in TARGETS:
        if g.depth.min() <= d <= g.depth.max():
            rows.append(dict(bid=bid, lat=g.lat.iloc[0], lon=g.lon.iloc[0],
                             depth=d, magt=float(np.interp(d, g.depth, g.magt))))
bh = pd.DataFrame(rows)
print(f"시추공 {bh.bid.nunique()}개 → 표준깊이 MAGT {len(bh)}점")

# ---------- 2) 깊이별 공간보간 → 3D 볼륨 ----------
glat = np.arange(60, 72.01, 0.2); glon = np.arange(-168, -139.99, 0.2)
GLon, GLat = np.meshgrid(glon, glat)
vol = np.full((len(TARGETS), len(glat), len(glon)), np.nan)
for i, d in enumerate(TARGETS):
    s = bh[bh.depth == d]
    if len(s) < 4:
        continue
    pts = s[["lon", "lat"]].to_numpy()
    lin = griddata(pts, s.magt, (GLon, GLat), method="linear")
    nn = griddata(pts, s.magt, (GLon, GLat), method="nearest")
    vol[i] = np.where(np.isnan(lin), nn, lin)
print(f"3D 볼륨: {vol.shape} (깊이×위도×경도), MAGT {np.nanmin(vol):.1f}~{np.nanmax(vol):.1f}°C")

# 지중온도(발산): 0°C=중립색(동결·융해 경계). 데이터 실제 범위로 노름 설정.
_vmin, _vmax = float(np.nanmin(vol)), float(np.nanmax(vol))
norm = tnorm(_vmin, _vmax)                 # 0°C가 흰색 중심
cmap = CMAP.temp
extent = [glon.min(), glon.max(), glat.min(), glat.max()]

# ---------- 3) GIF: 깊이 스캔 ----------
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(vol[0], extent=extent, origin="lower", cmap=cmap, norm=norm, aspect="auto")
sc = ax.scatter(bh.drop_duplicates("bid").lon, bh.drop_duplicates("bid").lat,
                s=28, color="k", marker="v", edgecolors="w", linewidths=0.6, zorder=5)
add_cbar(fig, im, ax, "연평균 지중온도 MAGT (°C)")
style_geo(ax)
ttl = ax.set_title("")

def frame(i):
    im.set_data(vol[i])
    ttl.set_text(f"알래스카 지중 열구조 — 깊이 {TARGETS[i]:.1f} m\n"
                 f"(청색=동결 영구동토, 적색=해빙 · 검정▽=시추공)")
    return im, ttl

anim = FuncAnimation(fig, frame, frames=len(TARGETS), interval=700, blit=False)
anim.save(animpath("alaska_thermal_depthscan"), writer=PillowWriter(fps=1.6), dpi=110)
plt.close(fig)
print("saved", animpath("alaska_thermal_depthscan"))

# ---------- 4) 3D 스택 렌더 ----------
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
sel = [0, 2, 4, 6, 8, 9]   # 0.5,2,4,7,15,20 m
for i in sel:
    Z = np.full_like(GLon, -TARGETS[i])
    fc = cmap(norm(vol[i]))
    ax.plot_surface(GLon, GLat, Z, facecolors=fc, rstride=2, cstride=2,
                    shade=False, alpha=0.72, linewidth=0)
ax.set_xlabel("경도 (°E)"); ax.set_ylabel("위도 (°N)"); ax.set_zlabel("깊이 (m)")
ax.set_zticks([-t for t in [0.5, 5, 10, 15, 20]])
ax.set_zticklabels([f"{t:.0f}" for t in [0.5, 5, 10, 15, 20]])
ax.view_init(elev=18, azim=-60)
m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
m.set_array([])
cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1)
cb.set_label("연평균 지중온도 MAGT (°C)", fontsize=10)
cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#444444")
cb.ax.tick_params(labelsize=9, length=2.5)
ax.set_title("알래스카 얕은 지중 열구조 (0~20m) — 깊이 스택\n청색 영구동토가 깊이 따라 어떻게 분포하나")
fig.tight_layout(); fig.savefig(volpath("alaska_thermal_3d"))
plt.close(fig)
print("saved", volpath("alaska_thermal_3d"))

# ---------- 5) 영구동토 상단면(0°C 도달 깊이) 2D 지도 ----------
table = np.full((len(glat), len(glon)), np.nan)
for y in range(len(glat)):
    for x in range(len(glon)):
        col = vol[:, y, x]
        below = np.where(col < 0)[0]
        if len(below):
            table[y, x] = TARGETS[below[0]]        # 0°C 아래 도달 최초 깊이
fig, ax = plt.subplots(figsize=(9, 6))
# 상단면 깊이 = 순차형(두께/깊이) → CMAP.alt (옅음→짙은 청, 깊을수록 짙음)
im = ax.imshow(table, extent=extent, origin="lower", cmap=CMAP.alt, aspect="auto")
add_cbar(fig, im, ax, "영구동토 상단 깊이 (m) — 얕을수록 지표까지 동결")
ax.scatter(bh.drop_duplicates("bid").lon, bh.drop_duplicates("bid").lat,
           s=28, color="k", marker="v", edgecolors="w", linewidths=0.6, zorder=5)
style_geo(ax)
ax.set_title("알래스카 영구동토 상단면 깊이 (MAGT 0°C 도달)\n35개 시추공 보간")
fig.tight_layout(); fig.savefig(mappath("alaska_permafrost_table"))
plt.close(fig)
print("saved", mappath("alaska_permafrost_table"))
