# -*- coding: utf-8 -*-
"""S10 — 얕은 0-3 m t_max 볼륨 3D 렌더 + 0°C 등온면 (PyVista off-screen, SSAA).

render_thermal3d_pyvista.py 패턴 재활용(기존 파일 수정 없음). 입력은
s10_shallow3d_volume.npz(지원반경 15 km 격자, s10_shallow3d_volume.py 산출).

정직성
- 수평면(x, y)은 km 물리 종횡비를 유지한다. 깊이(0-3 m)는 수평(약 1,100 km) 대비
  보이지 않으므로 시각 과장을 적용하고 배율을 캡션에 명기한다.
- 볼륨은 관측셀 지원반경 안에서만 채워진 회랑 구조다. 연속 온도장이 아니다.

출력: outputs/volumes_3d/s10_shallow3d_hero.png (컷어웨이 아님, 전체 회랑 볼륨+0°C 등온면)
      outputs/volumes_3d/s10_shallow3d_isotherm.png (0°C 등온면 단독, 깊이 착색)

실행: /home/anaconda3/bin/python scripts/4_visualization/s10_render_shallow3d_pyvista.py (ROOT, CPU)
"""
import os
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

OUT = "outputs/volumes_3d"
os.makedirs(OUT, exist_ok=True)
z = np.load("data/processed/s10_shallow3d_volume.npz")
lats, lons, depths = z["lats"], z["lons"], z["depths_m"]
tmax = z["tmax_c"]          # (ndepth, nlat, nlon), 지원 밖 NaN
thaw = z["thaw_cm"]

# 수평 km 좌표(물리 종횡비), 깊이는 시각 과장
LATM = float(np.mean(lats))
KX = 111.0 * np.cos(np.radians(LATM))   # km/°lon
KY = 111.0                              # km/°lat
Z_EXAG = 40_000.0                       # 1 m 깊이 -> 표시 40 km (배율 캡션 명기)
x_km = (lons - lons.min()) * KX
y_km = (lats - lats.min()) * KY
z_km = -depths * Z_EXAG / 1000.0        # 아래 방향 음수, 표시 km

# RectilinearGrid: z 오름차순 정렬(가장 깊은 층이 첫 인덱스)
order = np.argsort(z_km)
grid = pv.RectilinearGrid(x_km, y_km, z_km[order])
vol_t = tmax[order]                                  # (nz, ny, nx)
dep_cm = np.broadcast_to((depths[order] * 100.0)[:, None, None], vol_t.shape)
NOD = -999.0
grid.point_data["tmax"] = np.where(np.isfinite(vol_t), vol_t, NOD).ravel(order="C")
grid.point_data["depth_cm"] = dep_cm.ravel(order="C").astype(np.float32)

valid = grid.threshold([-20.0, 20.0], scalars="tmax")
iso = valid.contour([0.0], scalars="tmax")
print(f"[mesh] valid cells={valid.n_cells}  iso points={iso.n_points}")

try:
    import sys
    sys.path.insert(0, "src")
    from cmcrameri import cm as cmc
    from polar.plotstyle import CMAP as _PC
    CM_T, CM_D = cmc.vik, _PC.alt      # alt = oslo_r 양끝 절단(백색·흑색 포화 방지)
except Exception:
    CM_T, CM_D = "coolwarm", "Blues"
CLIM_T = [-4.0, 4.0]        # 0°C 중심 대칭

# 발표용: 스칼라바를 크게·왼쪽에 판독성 확보(이전 position_x=0.88은 오른쪽 여백에 몰려
# 라벨이 잘리고 작게 보였다). position_x는 제목이 좌측 밖으로 잘리지 않게 0.09로 둔다.
SBAR = dict(n_labels=6, fmt="%.0f", title_font_size=50, label_font_size=42,
            color="black", position_x=0.09, position_y=0.14, width=0.045,
            height=0.58, vertical=True)


def new_plotter():
    p = pv.Plotter(off_screen=True, window_size=[3000, 2100], lighting="three lights")
    p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    return p


def set_cam(p, elev=7, azim=-55, zoom=1.3):
    p.camera_position = "iso"
    p.camera.azimuth = azim
    p.camera.elevation = elev
    p.camera.zoom(zoom)


# ---------- 렌더 1: 회랑 볼륨(반투명) + 0°C 등온면(불투명) ----------
p = new_plotter()
# VTK 내장 폰트(arial/courier/times)는 한글 글리프가 없어 스칼라바 제목에 한글을 넣으면
# 글자가 누락된다. 스칼라바에는 물리량·단위만 ASCII로 표기하고, 한글 라벨
# '연최대 지중온도 (°C)'는 아래 matplotlib 합성 캡션에서 렌더한다(코드 토큰 t_max 미노출).
p.add_mesh(valid.extract_surface(), scalars="tmax", cmap=CM_T, clim=CLIM_T,
           opacity=0.28, lighting=False, nan_color="white",
           scalar_bar_args=dict(title="max annual ground T (°C)", **SBAR))
if iso.n_points > 0:
    p.add_mesh(iso, color="#00838F", opacity=1.0, smooth_shading=True,
               show_scalar_bar=False)
p.add_mesh(grid.outline(), color="#9AA0A6", line_width=1.6)
set_cam(p, elev=22, azim=-40, zoom=1.35)
p.screenshot(f"{OUT}/s10_shallow3d_hero_raw.png", scale=2)
p.close()
print("[render] s10_shallow3d_hero_raw.png")

# ---------- 렌더 2: 0°C 등온선 깊이면(warp 표면, 깊이 착색) ----------
# 등온면 깊이(thaw_cm)를 z로 내린 표면. 관 형태 isosurface보다 판독성이 높다.
xx, yy = np.meshgrid(x_km, y_km)
zz = -(thaw / 100.0) * Z_EXAG / 1000.0            # cm -> m -> 표시 km
sg = pv.StructuredGrid(xx, yy, np.where(np.isfinite(zz), zz, 0.0))
# 스칼라는 표면 z좌표에서 역산(포인트 순서 불일치 원천 차단). 결측(z=0)은 threshold로 제거.
sg.point_data["thaw_cm"] = (-sg.points[:, 2] * 1000.0 / Z_EXAG * 100.0).astype(np.float32)
surf_thaw = sg.threshold([5.0, 1000.0], scalars="thaw_cm")
p = new_plotter()
# clim은 실제 분포(5/95 백분위 약 45/145 cm)에 맞춰 [45,140]으로 좁혀 얕은 회랑이
# near-white로 사라지지 않게 한다(이전 [30,160]은 하한이 흰 배경과 섞여 좌·상단 회랑이
# 소실돼 데이터가 우하단에 편중된 것처럼 보였다). 스칼라바 제목은 캡션 용어와 통일한다.
# VTK 내장 폰트는 한글 글리프가 없어 '깊이'는 표기 못 하므로 ASCII로 통일 용어를 쓴다.
p.add_mesh(surf_thaw, scalars="thaw_cm", cmap=CM_D, clim=[45, 140],
           lighting=False,
           scalar_bar_args=dict(title="0C isotherm depth (cm)", **SBAR))
p.add_mesh(grid.outline(), color="#9AA0A6", line_width=1.6)
# 바닥 윤곽(수평 격자면)을 옅은 청회색으로 추가해 흩어진 회랑이 하나의 조사영역(알래스카)
# 안에 놓임을 보인다. 얕은(near-white) 회랑이 순백 배경에 소실되지 않도록 면색을 살짝 진하게.
foot = pv.Rectangle([[x_km.min(), y_km.min(), 0.0],
                     [x_km.max(), y_km.min(), 0.0],
                     [x_km.max(), y_km.max(), 0.0]])
p.add_mesh(foot, color="#DCE3EA", opacity=0.85, lighting=False, show_scalar_bar=False)
p.add_mesh(foot.extract_feature_edges(), color="#8A929B", line_width=1.6)
# 카메라를 데이터 중심으로. 이전 zoom=1.45는 과확대로 클러스터가 우하단에 몰려 보였다.
set_cam(p, elev=26, azim=-40, zoom=1.15)
p.reset_camera()
p.camera.zoom(1.28)
p.screenshot(f"{OUT}/s10_shallow3d_isotherm_raw.png", scale=2)
p.close()
print("[render] s10_shallow3d_isotherm_raw.png")

# ---------- matplotlib 합성: 제목·캡션(보고서 톤) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FP = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
plt.rcParams["axes.unicode_minus"] = False


def composite(raw, out, title, caption):
    img = plt.imread(raw)
    h, w = img.shape[:2]
    # 발표(10장)·보고서용: 넓은 3D 렌더 위 제목·캡션을 판독 가능한 크기로.
    # 이미지 폭(약 19 in @300dpi)에 비례해 이전보다 제목·캡션 폰트를 키운다.
    band = 470
    fig = plt.figure(figsize=(w / 300, (h + band) / 300), dpi=300)
    ax = fig.add_axes([0.02, 0.0, 0.96, h / (h + band)])
    ax.imshow(img)
    ax.axis("off")
    tax = fig.add_axes([0.02, h / (h + band), 0.96, band / (h + band)])
    tax.axis("off")
    tax.text(0.01, 0.70, title, fontsize=31, color="#18181B", va="center", fontproperties=FPB)
    tax.text(0.01, 0.24, caption, fontsize=17.0, color="#555555", va="center", fontproperties=FP)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[composite] {out}")


composite(f"{OUT}/s10_shallow3d_hero_raw.png", f"{OUT}/s10_shallow3d_hero.png",
          "알래스카 얕은 지중 연최대온도 볼륨(0-3 m)과 0°C 등온면",
          "지역: 알래스카(수평 약 1,100 km, km 물리 종횡비). 깊이 0-3 m는 표시 배율 x40,000 과장. "
          "컬러바=연최대 지중온도 (°C), GBM 조건장(사이트블록 R2 0.47). "
          "볼륨은 관측셀 지원반경 15 km 회랑에 한정(연속 온도장 아님, 반투명 표시). 청록 불투명 면=0°C 등온면.")
composite(f"{OUT}/s10_shallow3d_isotherm_raw.png", f"{OUT}/s10_shallow3d_isotherm.png",
          "0°C 등온면 깊이",
          "지역: 알래스카(수평 약 1,100 km, km 물리 종횡비). 깊이 표시 배율 x40,000 과장, 지원반경 15 km 한정. "
          "등온면을 깊이(cm)로 착색(연최대온도 기준 융해 최대 깊이). "
          "실측 ALT와의 상관 r=0.28(OOF 유도), 절대 정합은 미성립(R2<0).")
print("[done] 3D 렌더 2종 저장")
