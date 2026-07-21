# -*- coding: utf-8 -*-
"""연구 파이프라인 아키텍처 그림 (Digital Rock 논문 톤).
실제 데이터 썸네일(입력)→모델 블록→산출물 썸네일(출력)을 화살표로 연결.
출력: deck/assets/mid/architecture.png
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

for fp in ["/home/willy010313/.fonts/Pretendard-SemiBold.otf",
           "/home/willy010313/.fonts/Pretendard-ExtraBold.otf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
FP = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
plt.rcParams["axes.unicode_minus"] = False

OUT = "deck/assets/mid"; os.makedirs(OUT, exist_ok=True)
TEAL="#0E5A61"; TEAL2="#2C7D83"; NAVY="#1F3A52"; SLATE="#5C666E"; MUTE="#8A9096"
INK="#18181B"; PAPER="#FBFAF6"; CARD="#EEF2F1"; RULE="#D9D6CC"

def crop(path, box=None):
    im = Image.open(path).convert("RGB")
    if box:
        w, h = im.size
        im = im.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))
    return np.asarray(im)

fig = plt.figure(figsize=(12.5, 6.0), dpi=200)
fig.patch.set_facecolor(PAPER)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 100); bg.set_ylim(0, 100); bg.axis("off")

# ---- 입력 썸네일(좌) ----
inputs = [
    ("outputs/figures/01_data/covariates_overview.png", (0.0, 0.0, 0.5, 0.5), "기후 (ERA5-Land)"),
    ("outputs/figures/01_data/covariates_overview.png", (0.5, 0.0, 1.0, 0.5), "지형 (ArcticDEM)"),
    ("outputs/maps/weaklabels_overview.png", (0.0, 0.0, 0.33, 1.0), "SAR · InSAR"),
    ("outputs/maps/ground_temp_global.png", (0.0, 0.0, 1.0, 1.0), "시추공 지중온도"),
]
iy = [77, 56, 35, 14]
for (path, box, lab), yy in zip(inputs, iy):
    ax = fig.add_axes([0.045, yy/100.0, 0.13, 0.155])
    try:
        ax.imshow(crop(path, box)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(RULE); sp.set_linewidth(1.0)
    except Exception:
        ax.axis("off")
    bg.text(11, yy-1.6, lab, fontsize=12, color=INK, ha="center", va="top", fontproperties=FP)
bg.text(11, 94, "입력: 다중모달 관측", fontsize=14.5, color=NAVY, ha="center", fontproperties=FPB)

# ---- 모델 블록(중) ----
mx, mw = 36, 24
bg.add_patch(FancyBboxPatch((mx, 30), mw, 40, boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc=CARD, ec=TEAL, lw=1.6, zorder=2))
bg.text(mx+mw/2, 64.5, "모델", fontsize=16, color=TEAL, ha="center", fontproperties=FPB, zorder=3)
for i, t in enumerate(["GBM 조건장 (주력)", "6모델 토너먼트 비교", "Stefan 물리 결합 (잔차)", "셀 집계 · 1/n 가중"]):
    bg.text(mx+1.8, 57.5-i*7.0, "· "+t, fontsize=13.5, color=INK, ha="left", va="center",
            fontproperties=FP, zorder=3)
bg.text(mx+mw/2, 25.5, "누설 통제 평가", fontsize=12.5, color=SLATE, ha="center", fontproperties=FP)
bg.text(mx+mw/2, 21, "공간블록 · LORO · AOA · conformal", fontsize=11.5, color=MUTE, ha="center", fontproperties=FP)

# ---- 출력 썸네일(우) ----
outputs = [
    ("outputs/maps/alt_alaska_pred.png", None, "활동층 두께 2D 지도"),
    ("outputs/volumes_3d/thermal3d_exploded.png", None, "얕은 3D 열구조"),
    ("outputs/maps/alt_aoa_mask.png", None, "불확실성 · 적용범위(AOA)"),
]
oy = [66, 40, 15]
for (path, box, lab), yy in zip(outputs, oy):
    ax = fig.add_axes([0.70, yy/100.0, 0.19, 0.20])
    try:
        ax.imshow(crop(path, box)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(RULE); sp.set_linewidth(1.0)
    except Exception:
        ax.axis("off")
    bg.text(79.5, yy-2.0, lab, fontsize=12, color=INK, ha="center", va="top", fontproperties=FP)
bg.text(79.5, 94, "출력: 지도 · 3D · 불확실성", fontsize=14.5, color=TEAL2, ha="center", fontproperties=FPB)

# ---- 화살표 ----
bg.add_patch(FancyArrowPatch((30, 50), (35.5, 50), arrowstyle="-|>", mutation_scale=26, lw=2.4, color=SLATE, zorder=1))
bg.add_patch(FancyArrowPatch((60.5, 50), (68, 50), arrowstyle="-|>", mutation_scale=26, lw=2.4, color=SLATE, zorder=1))

fig.savefig(f"{OUT}/architecture.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved architecture.png")
