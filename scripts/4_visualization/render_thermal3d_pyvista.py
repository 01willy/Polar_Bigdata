# -*- coding: utf-8 -*-
"""논문급 고해상 3D 지중 열구조 컷어웨이 (PyVista off-screen, SSAA).
MAGT 볼륨 전체(온난 표층~냉각 심부)의 한 사분면을 잘라 내부 온도 구조를 노출.
모든 면을 MAGT로 착색(회색 backface 제거), 고해상 supersample.
출력: outputs/volumes_3d/thermal3d_exploded.png
"""
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

vol = pv.read("data/processed/volume_magt_rf.vti")
OUT = "outputs/volumes_3d"
x0, y0 = vol.bounds[0], vol.bounds[2]
xc, yc = vol.center[0], vol.center[1]
EXAG = 2200.0

def tf(mesh):
    p = mesh.points.copy()
    p[:, 0] = (p[:, 0] - x0) / 1000.0
    p[:, 1] = (p[:, 1] - y0) / 1000.0
    p[:, 2] = -(p[:, 2] / 1000.0) * EXAG
    mesh.points = p
    return mesh

try:
    from cmcrameri import cm as cmc
    cmap = cmc.vik
except Exception:
    cmap = "coolwarm"
clim = [-9, 9]

valid = vol.threshold([-20, 20], scalars="MAGT")
cut = valid.clip_box(bounds=[xc, vol.bounds[1], yc, vol.bounds[3], vol.bounds[4], vol.bounds[5]],
                     invert=True)
cut = cut.threshold([-20, 20], scalars="MAGT")  # 클립 후 잔여 무효셀 제거
surf = tf(cut.extract_surface())
iso = valid.contour([0.0], scalars="MAGT")
isoT = tf(iso) if iso.n_points > 0 else None

p = pv.Plotter(off_screen=True, window_size=[3000, 2200], lighting="three lights")
p.set_background("white")
p.enable_anti_aliasing("ssaa")

actor = p.add_mesh(surf, scalars="MAGT", cmap=cmap, clim=clim, opacity=1.0,
                   smooth_shading=False, lighting=False, nan_color="white",
                   show_edges=False,
                   scalar_bar_args=dict(title="MAGT (deg C)", n_labels=5, fmt="%.0f",
                                        title_font_size=42, label_font_size=36, color="black",
                                        position_x=0.89, position_y=0.30, width=0.03, height=0.44,
                                        vertical=True))
if isoT is not None:
    p.add_mesh(isoT, color="#0E5A61", opacity=0.22, smooth_shading=True, show_scalar_bar=False)
p.add_mesh(surf.outline(), color="#9AA0A6", line_width=1.6)

p.camera_position = "iso"
p.camera.azimuth = -48
p.camera.elevation = 16
p.camera.zoom(1.4)
p.screenshot(f"{OUT}/thermal3d_body_raw.png", scale=2)  # supersample x2
p.close()
print("saved thermal3d_body_raw.png")

# ---------------- matplotlib 합성: Pretendard 제목·설명 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FP  = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
plt.rcParams["axes.unicode_minus"] = False

img = plt.imread(f"{OUT}/thermal3d_body_raw.png")
h, w = img.shape[:2]
fig = plt.figure(figsize=(w / 300, h / 300 + 0.85), dpi=300)
ax = fig.add_axes([0.02, 0.0, 0.96, h / (h + 250)])
ax.imshow(img); ax.axis("off")
tax = fig.add_axes([0.02, h / (h + 250), 0.96, 250 / (h + 250)]); tax.axis("off")
tax.text(0.01, 0.66, "알래스카 3D 지중 열구조: 영구동토 블록 컷어웨이",
         fontsize=19, color="#18181B", va="center", fontproperties=FPB)
tax.text(0.01, 0.22,
         "북동 사분면을 잘라 내부 연평균 지중온도(MAGT)를 노출했다. 상단(지표)은 온난, 깊어질수록 냉각(청색)되며 "
         "청록 반투명 면이 0°C 등온면(영구동토 하부경계)이다. 깊이 0~90 m, 수평 대비 과장 표시.",
         fontsize=11.5, color="#555555", va="center", fontproperties=FP)
fig.savefig(f"{OUT}/thermal3d_exploded.png", dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved thermal3d_exploded.png (composited)")
