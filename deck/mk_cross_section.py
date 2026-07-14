# -*- coding: utf-8 -*-
"""2.5D 지중 열구조 도시 — 위도-깊이 수직 단면 + 깊이 슬라이스 소형지도.
3D 렌더의 한계 대신 논문 표준의 명료한 단면으로 지중 열구조를 보인다.
출력: deck/assets/mid/cross_section.png
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
for _f in ["SemiBold", "ExtraBold", "Medium"]:
    try: fm.fontManager.addfont(f"/home/willy010313/.fonts/Pretendard-{_f}.otf")
    except Exception: pass
FP  = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FPM = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-Medium.otf")
plt.rcParams["axes.unicode_minus"] = False
try:
    from cmcrameri import cm as cmc
    CMAP = cmc.vik
except Exception:
    CMAP = "coolwarm"

OUT = "deck/assets/mid"; os.makedirs(OUT, exist_ok=True)
INK = "#18181B"; SLATE = "#4A5158"; TEAL = "#0E5A61"

vol = pv.read("data/processed/volume_magt_rf.vti")
M = vol["MAGT"].reshape(vol.dimensions, order="F")  # (nx,ny,nz)
nx, ny, nz = M.shape
b = vol.bounds
# 물리좌표축
lat = np.linspace(b[2], b[3], ny) / 1000.0        # km(남북) — 대략 위도 대리
depth = np.linspace(b[4], b[5], nz)               # 0~90 m
# 위도-깊이 단면: 경도(x) index 16(유효 최대)에서 0~20 m만
xi = 16
cross = M[xi, :, :].T                              # (nz, ny)
zmask = depth <= 20.5
cross = cross[zmask, :]
depth20 = depth[zmask]

fig = plt.figure(figsize=(12.4, 5.4), dpi=200)
fig.patch.set_facecolor("white")
gs = GridSpec(5, 2, figure=fig, width_ratios=[3.0, 1.0],
              hspace=0.35, wspace=0.16, left=0.06, right=0.9, top=0.84, bottom=0.12)

# ---- (좌) 위도-깊이 단면 ----
axc = fig.add_subplot(gs[:, 0])
LATg, DEPg = np.meshgrid(lat, depth20)
pcm = axc.pcolormesh(LATg, DEPg, cross, cmap=CMAP, vmin=-9, vmax=9, shading="auto")
# 0°C 등온선(영구동토 상단)
cs = axc.contour(LATg, DEPg, cross, levels=[0], colors="#0E5A61", linewidths=2.2, linestyles="--")
axc.clabel(cs, fmt="0°C", fontsize=10)
axc.invert_yaxis()
axc.set_xlabel("남북 거리 (km)", fontsize=12.5, fontproperties=FP)
axc.set_ylabel("깊이 (m)", fontsize=12.5, fontproperties=FP)
axc.set_title("위도-깊이 수직 단면 (0~20 m)", fontsize=14, fontproperties=FPB, color=INK, loc="left", pad=8)
for t in axc.get_xticklabels()+axc.get_yticklabels(): t.set_fontproperties(FPM); t.set_fontsize(10.5)
axc.text(0.02, 0.10, "청록 파선 = 0°C 등온면\n(영구동토 상단 경계)", transform=axc.transAxes,
         fontsize=10.5, color=TEAL, fontproperties=FP, va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D9D6CC", alpha=0.85))

# ---- (우) 깊이 슬라이스 소형지도 5장 (세로) ----
slice_depths = [0.5, 2, 5, 10, 20]
for k, dd in enumerate(slice_depths):
    ax = fig.add_subplot(gs[k, 1])
    di = int(np.argmin(np.abs(depth - dd)))
    sl = M[:, :, di].T           # (ny, nx)
    ax.imshow(sl, cmap=CMAP, vmin=-9, vmax=9, origin="lower", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(f"{dd:g} m", fontsize=10.5, fontproperties=FPB, color=INK, rotation=0,
                  ha="right", va="center", labelpad=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#D9D6CC")
    if k == 0:
        ax.set_title("깊이 슬라이스", fontsize=11.5, fontproperties=FPB, color=INK, pad=4)

# 컬러바
cax = fig.add_axes([0.915, 0.14, 0.014, 0.62])
cb = fig.colorbar(pcm, cax=cax)
cb.set_label("연평균 지중온도 MAGT (°C)", fontsize=12, fontproperties=FP)
for t in cax.get_yticklabels(): t.set_fontproperties(FPM); t.set_fontsize(10)

fig.suptitle("얕은 지중 열구조: 위도-깊이 단면과 깊이 슬라이스 (2.5D)",
             fontsize=16, fontproperties=FPB, color=INK, x=0.06, ha="left", y=0.965)
fig.savefig(f"{OUT}/cross_section.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig); print("saved cross_section.png")
