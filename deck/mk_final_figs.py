# -*- coding: utf-8 -*-
"""mk_final_figs.py — 본선 발표덱 신규 다이어그램 (v2).

시맨틱 구조: figures/architecture_spec.json · 토큰: deck/deck_spec_final.json
출력: deck/assets/final/*.{png,pdf}
실행: cd /home/willy010313/Polar_Bigdata && python3 deck/mk_final_figs.py
v2: 개념도 질감(유기층·얼음렌즈·얼음쐐기)·축겹침 수정, 워크플로 썸네일·미니아이콘,
    딥러닝 아키텍처 그림 신설, 증강설계 실지도 썸네일.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Polygon
from PIL import Image

# ---------- 폰트 ----------
FDIR = os.path.expanduser("~/.fonts")
WEIGHTS = {"r": "Pretendard-Regular.otf", "m": "Pretendard-Medium.otf",
           "s": "Pretendard-SemiBold.otf", "b": "Pretendard-Bold.otf",
           "x": "Pretendard-ExtraBold.otf"}
FP = {}
for k, f in WEIGHTS.items():
    p = os.path.join(FDIR, f)
    fm.fontManager.addfont(p)
    FP[k] = fm.FontProperties(fname=p)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---------- 토큰 ----------
ACC   = "#EA851B"
INK   = "#111111"
GRAY  = "#6B6B6B"
GRAY2 = "#4A4A4A"
HAIR  = "#DDDDDD"
WHITE = "#FFFFFF"
Z = {
    "input":  {"tint": "#EEF2F6", "acc": "#3D6B8E"},
    "gate":   {"tint": "#FBF1E4", "acc": "#C06A14"},
    "aug":    {"tint": "#EDF4EE", "acc": "#4E7D57"},
    "models": {"tint": "#F1EFF7", "acc": "#6A51A3"},
    "output": {"tint": "#F4F4F2", "acc": "#4A4A4A"},
}
OUT = "deck/assets/final"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=220, facecolor=WHITE,
                    bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("saved", name)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=220)
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def zone(ax, x, y, w, h, key, label, label_fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
                                fc=Z[key]["tint"], ec="none", zorder=1))
    ax.text(x + w / 2, y + h - 0.17, label, ha="center", va="center",
            fontsize=label_fs, color=Z[key]["acc"], fontproperties=FP["x"], zorder=3)


def node(ax, x, y, w, h, title, detail=None, acc=None, dashed=False, title_fs=12,
         detail_fs=10.2, detail_color=GRAY, tc=INK, tx=None):
    ec = acc if acc else HAIR
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=WHITE, ec=ec, lw=1.3 if acc else 1.0,
                                ls=(0, (4, 2)) if dashed else "-", zorder=2))
    cx = tx if tx is not None else x + w / 2
    cy = y + h / 2
    if detail:
        ax.text(cx, cy + h * 0.18, title, ha="center", va="center", fontsize=title_fs,
                color=tc, fontproperties=FP["s"], zorder=3)
        ax.text(cx, cy - h * 0.22, detail, ha="center", va="center", fontsize=detail_fs,
                color=detail_color, fontproperties=FP["m"], zorder=3, linespacing=1.25)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_fs,
                color=tc, fontproperties=FP["s"], zorder=3)


def arrow(ax, p0, p1, dashed=False, color=GRAY2, lw=1.6, rad=0.0, ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, ls=(0, (4, 2.5)) if dashed else "-",
                                 shrinkA=2, shrinkB=2, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def thumb_axes(fig, ax, x, y, w, h, img_path=None, crop_frac=None):
    """inch 좌표에 소형 이미지 축 추가. crop_frac=(l,t,r,b) 0-1 비율."""
    W = ax.get_xlim()[1]; H = ax.get_ylim()[1]
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    if img_path:
        im = Image.open(img_path).convert("RGB")
        if crop_frac:
            l, t, r, b = crop_frac
            iw, ih = im.size
            im = im.crop((int(l * iw), int(t * ih), int(r * iw), int(b * ih)))
        a.imshow(np.asarray(im))
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color("#C9C9C4"); s.set_linewidth(0.8)
    return a


# ---------- v3 스타일 (참조덱 도식 문법) ----------
PANEL_BG = "#F8F8F6"
BORDER   = "#C9C9C4"
STEEL    = "#3D6B8E"


def sq_node(ax, x, y, w, h, fc=WHITE, ec=BORDER, lw=1.0, dashed=False, z=2):
    """직각에 가까운 노드 (백색 채움 + 가는 테두리)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=fc, ec=ec, lw=lw,
                                ls=(0, (4, 2)) if dashed else "-", zorder=z))


def sec_head(ax, x, y, num, label, w=None, fs=11.5, sub=None):
    """번호 매긴 단계 소제목: 오렌지 틱바 + 번호 + 잉크 볼드 라벨(+회색 부제), 하단 헤어라인."""
    ax.add_patch(Rectangle((x, y - 0.085), 0.048, 0.24, fc=ACC, ec="none", zorder=3))
    ax.text(x + 0.14, y + 0.03, num, fontsize=fs + 0.5, color=ACC,
            fontproperties=FP["x"], va="center", zorder=3)
    tx = x + 0.14 + 0.19
    ax.text(tx, y + 0.03, label, fontsize=fs, color=INK, fontproperties=FP["x"],
            va="center", zorder=3)
    if sub:
        ax.text(tx, y - 0.22, sub, fontsize=8.4, color=GRAY, fontproperties=FP["m"],
                va="center", zorder=3)
    if w:
        ax.plot([x, x + w], [y - 0.165, y - 0.165], color=HAIR, lw=0.9, zorder=2)


def line_legend(ax, x, y, items, fs=8.6):
    """우상단 선종류 범례. items=[(label, dashed, color), ...]"""
    for lab, dashed, col in items:
        ax.plot([x, x + 0.30], [y, y], color=col, lw=1.5,
                ls=(0, (4, 2.5)) if dashed else "-", zorder=3)
        ax.text(x + 0.37, y, lab, fontsize=fs, color=GRAY2, fontproperties=FP["m"],
                va="center", zorder=3)
        x += 0.37 + 0.115 * len(lab) + 0.28


def notation(ax, x, y, text, fs=8.0):
    """기호 정의 각주."""
    ax.text(x, y, text, fontsize=fs, color=GRAY, fontproperties=FP["m"], va="center", zorder=3)


def real_thumb(fig, ax, x, y, w, h, img_path, crop_frac=None, ec=STEEL, lw=1.1):
    """실데이터 썸네일: 목표 박스 종횡비에 맞춰 중앙 크롭, 역할색 테두리."""
    im = Image.open(img_path).convert("RGB")
    if crop_frac:
        l, t, r, b = crop_frac
        iw, ih = im.size
        im = im.crop((int(l * iw), int(t * ih), int(r * iw), int(b * ih)))
    iw, ih = im.size
    tgt = w / h
    if iw / ih > tgt:   # 너무 넓음 → 좌우 중앙 크롭
        nw = int(ih * tgt)
        x0 = (iw - nw) // 2
        im = im.crop((x0, 0, x0 + nw, ih))
    else:               # 너무 높음 → 상하 중앙 크롭
        nh = int(iw / tgt)
        y0 = (ih - nh) // 2
        im = im.crop((0, y0, iw, y0 + nh))
    W = ax.get_xlim()[1]; H = ax.get_ylim()[1]
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.imshow(np.asarray(im), aspect="auto")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(ec); s.set_linewidth(lw)
    return a


def verdict_chip(ax, x, y, ok=True):
    """게이트 판정 칩: 채택(오렌지 체크) / 제외(회색 엑스). 글리프 대신 선으로 그림."""
    col = ACC if ok else "#9A9A96"
    ax.add_patch(Ellipse((x, y), 0.26, 0.26, fc=col, ec="none", zorder=4))
    if ok:
        ax.plot([x - 0.060, x - 0.014, x + 0.066], [y - 0.004, y - 0.052, y + 0.046],
                color=WHITE, lw=1.9, zorder=5, solid_capstyle="round")
    else:
        d = 0.046
        ax.plot([x - d, x + d], [y - d, y + d], color=WHITE, lw=1.9, zorder=5,
                solid_capstyle="round")
        ax.plot([x - d, x + d], [y + d, y - d], color=WHITE, lw=1.9, zorder=5,
                solid_capstyle="round")


# ============================================================
# 1) fig_workflow — 전체 파이프라인 (v2: 미니아이콘·산출물 썸네일)
# ============================================================
# ---------- v5 스타일: 참조덱 p3·5·6·8 평면 문법 ----------
BACK   = "#F8F8F6"
SEP    = "#C6C6C1"
RULE_D = "#4A4A4A"
RULE_L = "#D8D8D4"
OKC    = "#2F6B33"
NOC    = "#96382C"
ACC_D  = "#B25E0F"
NET_FC, NET_EC, NET_TC = "#F7E9AE", "#C7A85C", "#6B5518"
GRN_FC, GRN_EC = "#DDEBD8", "#7FA579"


def backdrop(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc=BACK, ec="none", zorder=0.5))


def vsep(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color=SEP, lw=0.8, ls=(0, (3, 3)), zorder=1)


def ohead(ax, x, y, label, fs=11.5, color=ACC):
    ax.text(x, y, label, fontsize=fs, color=color, fontproperties=FP["x"], zorder=3)


def rule(ax, x0, x1, y, heavy=False):
    ax.plot([x0, x1], [y, y], color=RULE_D if heavy else RULE_L,
            lw=1.2 if heavy else 0.8, zorder=2)


def sarrow(ax, p0, p1, lw=1.3, color="#3A3A3A", ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=2, shrinkB=2, zorder=4))


def chevron5(ax, x, y, w, h, title, sub=None, fc=NET_FC, ec=NET_EC, tfs=10.2, sfs=7.9):
    tip = 0.22
    pts = [(x, y), (x + w - tip, y), (x + w, y + h / 2), (x + w - tip, y + h), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.2, zorder=3))
    cx = x + (w - tip) / 2 + 0.04
    if sub:
        ax.text(cx, y + h * 0.64, title, ha="center", va="center", fontsize=tfs,
                color=INK, fontproperties=FP["s"], zorder=4)
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center", fontsize=sfs,
                color=GRAY2, fontproperties=FP["m"], zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=tfs,
                color=INK, fontproperties=FP["s"], zorder=4)


def fig_workflow():
    """v6: 백색 배경 · 열 균등 배분 · 행 높이 정렬로 화살표 전부 수평."""
    W, H = 12.8, 5.95
    fig, ax = canvas(W, H)
    for x in (2.62, 5.62, 8.62, 11.24):
        vsep(ax, x, 0.95, 5.42)

    ohead(ax, 0.30, 5.58, "입력 자료")
    ohead(ax, 2.80, 5.58, "라벨 검증 게이트")
    ohead(ax, 5.80, 5.58, "유사라벨 증강")
    ohead(ax, 8.80, 5.58, "예측 모델군")
    ohead(ax, 11.42, 5.58, "산출물")

    # ---- 행 기준 높이 (열 간 정렬) ----
    R1, R2, R3, R4 = 4.91, 3.73, 2.55, 1.37

    # ---- 입력: 이미지 + 하단 캡션 ----
    in_items = [
        (R1, "실측 ALT 13,606셀", "CALM · ABoVE (알래스카)",
         "outputs/maps/data_inventory_world.png", (0.115, 0.132, 0.278, 0.242)),
        (R2, "물리경험식 5종", "Stefan · Kudryavtsev 등",
         "outputs/figures/s2_physics/physics_members_maps_alaska.png", (0.055, 0.095, 0.305, 0.44)),
        (R3, "지중온도 프로파일", "GTN-P 37 · KPDC 콘슬 19",
         "outputs/figures/s10_shallow3d/s10_profiles_trumpet.png", (0.545, 0.04, 0.99, 0.40)),
        (R4, "공변량 34종", "지형·기후·토양·SAR·CCI",
         "outputs/figures/01_data/covariates_overview.png", (0.04, 0.055, 0.355, 0.475)),
    ]
    for cy, t1, t2, img, cf in in_items:
        y = cy - 0.36
        real_thumb(fig, ax, 0.38, y, 1.30, 0.72, img, cf, ec="#8A8A86", lw=0.9)
        ax.text(0.38, y - 0.155, t1, fontsize=9.2, color=INK, fontproperties=FP["s"], zorder=3)
        ax.text(0.38, y - 0.335, t2, fontsize=7.8, color=GRAY, fontproperties=FP["m"], zorder=3)

    # ---- 게이트: 북탭스 표 (행 높이 = 입력과 정렬) ----
    gx0, gx1 = 2.80, 5.44
    rule(ax, gx0, gx1, R1 + 0.44, heavy=True)
    rows = [
        (R1, "실측", True, ["학습 기준 · 세 지역 셀 단위 정렬"]),
        (R2, "물리식 5종 → Stefan", True, ["지역 내 RMSE 14.56 vs 25–46 cm"]),
        (R3, "지중온도 유도", False,
         ["전이 악화 21.8→88.1 cm", "3차원 학습 라벨·현장 검증 전용"]),
    ]
    for cy, title, ok, subs in rows:
        ax.text(gx0, cy + 0.10, title, fontsize=10.3, color=INK,
                fontproperties=FP["s"], zorder=3)
        ax.text(gx1, cy + 0.10, "채택" if ok else "제외", fontsize=9.3,
                color=OKC if ok else NOC, ha="right", fontproperties=FP["x"], zorder=3)
        for k, sb in enumerate(subs):
            ax.text(gx0, cy - 0.17 - 0.20 * k, sb, fontsize=8.2, color=GRAY,
                    fontproperties=FP["m"], zorder=3)
    rule(ax, gx0, gx1, (R1 + R2) / 2 - 0.04)
    rule(ax, gx0, gx1, (R2 + R3) / 2 - 0.04)
    rule(ax, gx0, gx1, R3 - 0.62, heavy=True)

    # ---- 증강: 수식 (행 높이 정렬) ----
    ucx = 7.12
    ax.text(ucx, R1, r"$D_r = D_{\mathrm{obs}} \cup \{(c_k,\ \tilde{y}_k)\}$",
            fontsize=11, color=INK, ha="center", va="center", zorder=3)
    ax.text(ucx, R2, r"$\tilde{y} = \hat{E}\sqrt{\mathrm{TDD}},\ \ r \leq 10$",
            fontsize=11, color=INK, ha="center", va="center", zorder=3)
    ax.text(ucx, R2 - 0.55, "유사라벨 ỹ: 물리식 산출 ALT를\n실측처럼 학습집합에 추가",
            fontsize=8.3, color=GRAY2, ha="center", fontproperties=FP["m"],
            linespacing=1.4, zorder=3, va="center")
    rule(ax, 6.05, 8.20, 2.72)
    ax.text(ucx, 2.44, "상수 라벨 대조군", fontsize=9.8, color=INK, ha="center",
            fontproperties=FP["s"], zorder=3)
    ax.text(ucx, 2.08, "동일 셀·개수·난수, 평균값 라벨\n물리 정보의 순가치 분리",
            fontsize=8.0, color=GRAY, ha="center", fontproperties=FP["m"],
            linespacing=1.35, zorder=3, va="center")

    # ---- 모델군: 북탭스 표 ----
    mx0, mx1 = 8.80, 11.06
    rule(ax, mx0, mx1, 5.34, heavy=True)
    ax.text(mx0, 5.10, "부스팅 4종", fontsize=10, color=INK, fontproperties=FP["s"], zorder=3)
    ax.text(mx0, 4.85, "CatBoost · XGBoost · LightGBM", fontsize=7.9, color=GRAY,
            fontproperties=FP["m"], zorder=3)
    ax.text(mx0, 4.67, "HistGBM", fontsize=7.9, color=GRAY, fontproperties=FP["m"], zorder=3)
    rule(ax, mx0, mx1, 4.47)
    ax.text(mx0, 4.23, "신경망 3종", fontsize=10, color=INK, fontproperties=FP["s"], zorder=3)
    ax.text(mx0, 3.98, "MLP · TabM · FT-Transformer", fontsize=7.9, color=GRAY,
            fontproperties=FP["m"], zorder=3)
    rule(ax, mx0, mx1, 3.78)
    ax.text(mx0, 3.54, "물리 잔차 결합", fontsize=10, color=ACC_D,
            fontproperties=FP["s"], zorder=3)
    ax.text(mx0, 3.24, r"$\hat{y} = \hat{E}\sqrt{\mathrm{TDD}} + \lambda\, g(c)$",
            fontsize=9.2, color=INK, zorder=3)
    rule(ax, mx0, mx1, 3.02)
    ax.text(mx0, 2.78, "독립 관측 결합", fontsize=10, color=ACC_D,
            fontproperties=FP["s"], zorder=3)
    ax.text(mx0, 2.53, "물리식·위성 CCI 앵커 평균", fontsize=7.9, color=GRAY,
            fontproperties=FP["m"], zorder=3)
    ax.text(mx0, 2.35, "편향 상쇄", fontsize=7.9, color=GRAY, fontproperties=FP["m"], zorder=3)
    rule(ax, mx0, mx1, 2.15, heavy=True)

    # ---- 산출물: 이미지 + 하단 캡션 (입력과 같은 행 높이) ----
    outs = [
        (R1, "고해상 ALT 지도", "0.02° · 243 m",
         "outputs/maps/alt_prediction_hires.png", None),
        (R2, "연별 ALT 지도", "2010–2024",
         "outputs/figures/s9_timelapse/alt_annual_fields.png", (0.02, 0.05, 0.30, 0.60)),
        (R3, "표층 3차원 온도장", "0–3 m · 지중온도 라벨",
         "outputs/figures/s10_shallow3d/s10_depth_slices.png", (0.02, 0.03, 0.46, 0.52)),
        (R4, "예측구간 + AOA", "90% · 등순응 보정",
         "outputs/maps/alt_uncertainty_aoa_map.png", (0.0, 0.05, 0.42, 0.66)),
    ]
    for cy, t1, t2, img, cf in outs:
        y = cy - 0.33
        real_thumb(fig, ax, 11.42, y, 1.28, 0.66, img, cf, ec="#8A8A86", lw=0.9)
        ax.text(11.42, y - 0.15, t1, fontsize=8.2, color=INK, fontproperties=FP["s"], zorder=3)
        ax.text(11.42, y - 0.32, t2, fontsize=7.3, color=GRAY, fontproperties=FP["m"], zorder=3)

    # ---- 화살표: 전부 수평 ----
    for cy in (R1, R2, R3):
        sarrow(ax, (1.74, cy), (2.76, cy))
    for cy in (R1, R2):
        sarrow(ax, (5.48, cy), (5.76, cy))
    sarrow(ax, (8.24, R2), (8.76, R2), lw=1.8, ms=13)
    sarrow(ax, (11.10, R2), (11.38, R2), lw=1.8, ms=13)
    # 공변량 → 모델군 (수평·수직 직각 경로)
    ax.plot([1.74, 9.93], [R4, R4], color="#3A3A3A", lw=1.3, zorder=4)
    sarrow(ax, (9.93, R4), (9.93, 2.11))
    ax.text(5.40, R4 + 0.13, "공변량 c(x) 직접 입력", ha="center", fontsize=8.8,
            color=GRAY2, fontproperties=FP["m"], zorder=4)

    # ---- 하단 ----
    rule(ax, 0.30, 12.70, 0.56)
    ax.text(0.30, 0.35, "검증 축   ① 알래스카 공간블록 교차검증(74블록)    ② 지역 간 전이 "
            "LORO(한 지역 전체 제외 후 평가, 세 지역 비가중평균)    ③ KPDC 현장 검증(수어드반도 콘슬)",
            fontsize=9.6, color=INK, fontproperties=FP["s"], zorder=3, va="center")
    notation(ax, 0.30, 0.13,
             "표기   D_obs 실측 학습집합 · c(x) 공변량 벡터 · ỹ 유사라벨 · r 증강 비율(유사라벨/실측) · "
             "Ê 학습 폴드 적합 계수 · TDD 융해기 적산온도(°C·일) · λ 잔차 가중 · ŷ 예측 ALT")
    save(fig, "fig_workflow")


# ============================================================
# 2) fig_dl_arch — 예측 모델군 아키텍처 (신규)
# ============================================================
def fig_dl_arch():
    """v6: 백색 배경 · 레인 출력선 정렬 · 물리 열 수직 사슬 배선 · 열 간 관계는 텍스트로."""
    W, H = 12.8, 5.95
    fig, ax = canvas(W, H)
    for x in (2.55, 7.95, 11.10):
        vsep(ax, x, 0.95, 5.42)

    ohead(ax, 0.30, 5.58, "입력 표현")
    ohead(ax, 2.75, 5.58, "표 형식 딥러닝 3종")
    ax.text(5.05, 5.58, "직접 회귀", fontsize=8.2, color=GRAY,
            fontproperties=FP["m"], zorder=3)
    ohead(ax, 8.15, 5.58, "물리 잔차 결합")
    ax.text(10.05, 5.58, "입력: 동일 공변량", fontsize=8.2, color=GRAY,
            fontproperties=FP["m"], zorder=3)
    ohead(ax, 11.30, 5.58, "출력")

    # ---- 입력 ----
    ax.text(1.30, 5.06, r"$c(x) \in \mathbb{R}^{34}$", fontsize=12, color=INK,
            ha="center", fontproperties=FP["s"], zorder=3)
    groups = [("지형 6", 6, "#2E4E6C"), ("기후 8", 8, "#3D6B8E"), ("토양 9", 9, "#5580A2"),
              ("SAR 8", 8, "#7396B4"), ("CCI 2", 2, "#9BB2C9"), ("표시 1", 1, "#C2CFDC")]
    gx_, gw_, bar_h = 0.55, 0.44, 2.90
    yy = 1.78
    for label, n, c in groups:
        hseg = bar_h * n / 34
        ax.add_patch(Rectangle((gx_, yy), gw_, hseg, fc=c, ec=WHITE, lw=1.0, zorder=3))
        ax.text(gx_ + gw_ + 0.11, yy + hseg / 2, label, fontsize=9.0, color=GRAY2,
                va="center", fontproperties=FP["m"], zorder=3)
        yy += hseg
    ax.text(1.30, 1.48, "표준화: 학습 폴드 내부 통계", fontsize=8.2, color=GRAY,
            ha="center", fontproperties=FP["m"], zorder=3)

    # ---- 딥러닝 3종 ----
    lx = 2.75
    xjn = 7.68
    y1 = 4.58
    ax.text(lx, 5.06, "MLP · 다층 퍼셉트론", fontsize=10.0, color=INK,
            fontproperties=FP["s"], zorder=3)
    xx = lx + 0.06
    for i, (units, bh) in enumerate([("256", 0.56), ("128", 0.40), ("64", 0.28)]):
        ax.add_patch(FancyBboxPatch((xx, y1 - bh / 2), 0.50, bh,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=NET_FC, ec=NET_EC, lw=1.1, zorder=3))
        ax.text(xx + 0.25, y1, units, fontsize=8.7, color=NET_TC, ha="center",
                va="center", fontproperties=FP["s"], zorder=4)
        if i < 2:
            sarrow(ax, (xx + 0.52, y1), (xx + 0.71, y1), lw=1.0, ms=7)
        xx += 0.73
    ax.text(xx + 0.10, y1 + 0.17, "ReLU · BatchNorm · 드롭아웃 0.1", fontsize=7.6,
            color=GRAY, fontproperties=FP["m"], zorder=3)
    ax.plot([xx + 0.04, xjn], [y1, y1], color="#3A3A3A", lw=1.1, zorder=3)
    y2 = 3.40
    ax.text(lx, 3.88, "TabM · 파라미터 효율 앙상블", fontsize=10.0, color=INK,
            fontproperties=FP["s"], zorder=3)
    ax.add_patch(FancyBboxPatch((lx + 0.06, y2 - 0.24), 1.14, 0.48,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc=NET_FC, ec=NET_EC, lw=1.1, zorder=3))
    ax.text(lx + 0.63, y2, "공유 몸체\n256→128", fontsize=7.8, color=NET_TC, ha="center",
            va="center", fontproperties=FP["s"], linespacing=1.2, zorder=4)
    sx = lx + 1.54
    for i in range(3, -1, -1):
        off = i * 0.055
        ax.add_patch(FancyBboxPatch((sx + off, y2 - 0.21 - off * 0.5), 1.06, 0.42,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=NET_FC if i == 0 else WHITE, ec=NET_EC,
                                    lw=1.0, zorder=3.3 - 0.05 * i))
    ax.text(sx + 0.53, y2 - 0.005, "헤드 128→64→1", fontsize=7.7, color=NET_TC,
            ha="center", va="center", fontproperties=FP["s"], zorder=4)
    ax.text(sx + 1.23, y2 + 0.27, "×8", fontsize=9.3, color="#8A6D1F", ha="center",
            fontproperties=FP["x"], zorder=4)
    sarrow(ax, (lx + 1.22, y2), (sx - 0.02, y2), lw=1.0, ms=7)
    sarrow(ax, (sx + 1.12, y2), (sx + 1.34, y2), lw=1.0, ms=7)
    ax.text(sx + 1.40, y2, "예측 평균", fontsize=8.2, color=GRAY, va="center",
            fontproperties=FP["m"], zorder=3)
    ax.plot([6.98, xjn], [y2, y2], color="#3A3A3A", lw=1.1, zorder=3)
    y3 = 2.02
    ax.text(lx, 2.52, "FT-Transformer · 특징 토큰 + 어텐션", fontsize=10.0, color=INK,
            fontproperties=FP["s"], zorder=3)
    for i in range(8):
        fc = "#8A6D1F" if i == 0 else "#DCE4EC"
        ax.add_patch(Rectangle((lx + 0.06 + i * 0.195, y3 + 0.03), 0.155, 0.155,
                               fc=fc, ec=STEEL, lw=0.7, zorder=3))
    ax.text(lx + 0.06, y3 - 0.19, "[CLS] + 특징 토큰 34 · d = 64", fontsize=7.7,
            color=GRAY2, fontproperties=FP["m"], zorder=3)
    bx = lx + 1.96
    for i in range(2, -1, -1):
        off = i * 0.055
        ax.add_patch(FancyBboxPatch((bx + off, y3 - 0.12 - off * 0.5), 1.58, 0.5,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=NET_FC if i == 0 else WHITE, ec=NET_EC,
                                    lw=1.0, zorder=3.3 - 0.05 * i))
    ax.text(bx + 0.79, y3 + 0.225, "Transformer 블록", fontsize=8.1, color=NET_TC,
            ha="center", va="center", fontproperties=FP["s"], zorder=4)
    ax.text(bx + 0.79, y3 + 0.035, "8-헤드 MHSA · FFN 128 · pre-LN", fontsize=7.0,
            color=GRAY2, ha="center", va="center", fontproperties=FP["m"], zorder=4)
    ax.text(bx + 1.73, y3 + 0.40, "×3", fontsize=9.3, color="#8A6D1F", ha="center",
            fontproperties=FP["x"], zorder=4)
    sarrow(ax, (lx + 1.68, y3 + 0.10), (bx - 0.02, y3 + 0.10), lw=1.0, ms=7)
    sarrow(ax, (bx + 1.62, y3 + 0.10), (bx + 1.84, y3 + 0.10), lw=1.0, ms=7)
    ax.text(bx + 1.90, y3 + 0.10, "[CLS] 헤드", fontsize=7.7, color=GRAY, va="center",
            fontproperties=FP["m"], zorder=3)
    ax.plot([7.24, xjn], [y3 + 0.10, y3 + 0.10], color="#3A3A3A", lw=1.1, zorder=3)
    # 레인 출력 라벨(선 위, 동일 오프셋) + 합류 세로선
    for yy_, lb in [(y1, r"$\hat{y}_{\mathrm{MLP}}$"), (y2, r"$\hat{y}_{\mathrm{TabM}}$"),
                    (y3 + 0.10, r"$\hat{y}_{\mathrm{FT}}$")]:
        ax.text(xjn - 0.10, yy_ + 0.12, lb, fontsize=8.6, color="#5F4B9E",
                ha="right", zorder=4)
    ax.plot([xjn, xjn], [y3 + 0.10, y1], color="#3A3A3A", lw=1.1, zorder=3)
    sarrow(ax, (xjn, 3.40), (8.32, 3.40), lw=1.7, ms=12)
    ax.text(8.00, 3.56, "비교", fontsize=7.8, color=GRAY2, ha="center",
            fontproperties=FP["m"], zorder=4)
    ax.text(2.75, 1.12, "부스팅 4종과 동일 조건 비교 · 3-seed 앙상블 · 직접 회귀 ŷ와 물리 잔차 결합 "
            "ŷ는 동일 검증 규약에서 비교해 구성 선택", fontsize=8.3, color=GRAY2,
            fontproperties=FP["m"], zorder=3)

    # ---- 물리 잔차 결합: 수직 사슬 ----
    acx = 9.90
    ax.text(acx, 5.00, "물리식 앵커 (Stefan)", fontsize=10.0, color=INK, ha="center",
            fontproperties=FP["s"], zorder=3)
    ax.text(acx, 4.66, r"$\hat{E}\sqrt{\mathrm{TDD}(x)}$", fontsize=10.6, color=INK,
            ha="center", zorder=3)
    plus_y = 2.42
    sarrow(ax, (acx, 4.48), (acx, plus_y + 0.17), lw=1.2)
    ax.add_patch(FancyBboxPatch((8.18, 2.98), 1.50, 0.72,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc=NET_FC, ec=NET_EC, lw=1.1, zorder=3))
    ax.text(8.93, 3.50, "저용량 잔차 모델 g(c)", fontsize=8.7, color=NET_TC, ha="center",
            va="center", fontproperties=FP["s"], zorder=4)
    ax.text(8.93, 3.22, "능형 회귀 · 얕은 부스팅", fontsize=7.3, color="#7A6A34",
            ha="center", va="center", fontproperties=FP["m"], zorder=4)
    ax.plot([8.93, 8.93], [2.98, plus_y], color="#3A3A3A", lw=1.2, zorder=3)
    sarrow(ax, (8.93, plus_y), (acx - 0.17, plus_y), lw=1.2)
    ax.text(9.30, plus_y + 0.14, "× λ", fontsize=10, color=INK,
            fontproperties=FP["s"], ha="center", zorder=4)
    ax.add_patch(Ellipse((acx, plus_y), 0.30, 0.30, fc=WHITE, ec=NET_EC, lw=1.4, zorder=4))
    ax.text(acx, plus_y, "+", fontsize=14, color=ACC_D, ha="center", va="center",
            fontproperties=FP["x"], zorder=5)
    ax.plot([8.45, 8.45], [2.52, 2.98], color="#B5493A", lw=1.2,
            ls=(0, (4, 2.2)), zorder=4)
    sarrow(ax, (8.45, 2.86), (8.45, 2.97), lw=1.2, color="#B5493A", ms=8)
    ax.text(8.45, 2.38, "실측 y (학습 시)", fontsize=7.3, color="#B5493A",
            ha="center", fontproperties=FP["m"], zorder=4)
    sarrow(ax, (acx, plus_y - 0.17), (acx, 2.06), lw=1.2)
    ax.text(acx, 1.84, r"$\hat{y} = \hat{E}\sqrt{\mathrm{TDD}} + \lambda\, g(c)$",
            fontsize=11, color=INK, ha="center", fontproperties=FP["s"], zorder=3)
    ax.text(acx, 1.48, "λ = 0 물리식 단독 · λ = 1 잔차 전체 반영", fontsize=8.0,
            color=GRAY, ha="center", fontproperties=FP["m"], zorder=3)
    ax.text(acx, 1.27, "g 학습 타깃 " + r"$y_i - \hat{E}\sqrt{\mathrm{TDD}_i}$" +
            " · λ는 검증에서 선택", fontsize=8.0, color=GRAY, ha="center",
            fontproperties=FP["m"], zorder=3)

    # ---- 출력 ----
    real_thumb(fig, ax, 11.30, 4.30, 1.30, 0.70, "outputs/maps/alt_prediction_hires.png",
               None, ec="#8A8A86", lw=0.9)
    ax.text(11.30, 4.14, "ALT 예측 ŷ (cm)", fontsize=8.5, color=INK,
            fontproperties=FP["s"], zorder=3)
    ax.text(11.30, 3.97, "0.02° 광역 지도", fontsize=7.4, color=GRAY,
            fontproperties=FP["m"], zorder=3)
    real_thumb(fig, ax, 11.30, 2.52, 1.30, 0.70, "outputs/maps/alt_uncertainty_aoa_map.png",
               (0.0, 0.05, 0.42, 0.66), ec="#8A8A86", lw=0.9)
    ax.text(11.30, 2.36, "90% 예측구간", fontsize=8.5, color=INK,
            fontproperties=FP["s"], zorder=3)
    ax.text(11.30, 2.19, "등순응 보정 · AOA 동봉", fontsize=7.4, color=GRAY,
            fontproperties=FP["m"], zorder=3)

    sarrow(ax, (2.10, 3.40), (2.68, 3.40), lw=1.7, ms=12)
    ax.plot([10.88, 11.04], [1.84, 1.84], color="#3A3A3A", lw=1.4, zorder=3)
    ax.plot([11.04, 11.04], [1.84, 3.40], color="#3A3A3A", lw=1.4, zorder=3)
    sarrow(ax, (11.04, 3.40), (11.26, 3.40), lw=1.7, ms=12)

    rule(ax, 0.30, 12.70, 0.66)
    ax.text(0.30, 0.44, "학습 설정   SmoothL1 손실 · Adam (lr 10⁻³, FT-T 5×10⁻⁴) · "
            "검증 10% 조기 종료 · 표적 표준화 · 구성 선택 = 공간블록 교차검증",
            fontsize=9.6, color=INK, fontproperties=FP["s"], zorder=3, va="center")
    notation(ax, 0.30, 0.18,
             "표기   c(x) 공변량 벡터 · d 토큰 차원 · MHSA 멀티헤드 셀프어텐션 · FFN 피드포워드 층 · "
             "pre-LN 사전 층정규화 · λ 잔차 가중 · g(c) 잔차 모델 · ŷ 예측 ALT · 적색 파선 = 학습 시에만")
    save(fig, "fig_dl_arch")


# ============================================================
# 3) fig_aug_design — 증강·대조실험 설계 (v2: 실지도 썸네일)
# ============================================================
def fig_aug_design():
    """v5: 평면 2레인 — 파선 레인 구분 + 자유 배치 조건 열 + 셰브런 2개 + 실데이터."""
    W, H = 12.6, 5.70
    fig, ax = canvas(W, H)
    ax.plot([0.50, 12.45], [2.78, 2.78], color=SEP, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for yy_, lab in [(4.15, "설  계"), (1.62, "채점 · 순가치")]:
        ax.text(0.24, yy_, lab, fontsize=10, color=ACC, fontproperties=FP["x"],
                rotation=90, ha="center", va="center", zorder=3)

    # ---- 레인 1 ----
    ohead(ax, 0.55, 5.32, "라벨 조건 설계", fs=11)
    ax.text(2.42, 5.32, "셀 · 개수 · 난수 동일, 라벨 내용만 상이한 세 조건",
            fontsize=8.4, color=GRAY, fontproperties=FP["m"], zorder=3)
    real_thumb(fig, ax, 0.55, 4.28, 1.15, 0.64, "outputs/maps/data_inventory_world.png",
               (0.115, 0.132, 0.278, 0.242), ec="#8A8A86", lw=0.9)
    ax.text(0.55, 4.13, "알래스카 실측", fontsize=8.9, color=INK, fontproperties=FP["s"], zorder=3)
    ax.text(0.55, 3.96, r"$D_{\mathrm{obs}}$ · n = 13,606 · CALM·ABoVE", fontsize=7.6,
            color=GRAY, fontproperties=FP["m"], zorder=3)
    real_thumb(fig, ax, 0.55, 3.20, 1.15, 0.64, "deck/assets/final/crops/lena_maps.png",
               (0.035, 0.12, 0.225, 0.82), ec="#8A8A86", lw=0.9)
    ax.text(0.55, 3.05, "대상 지역 (라벨 없음 가정)", fontsize=8.6, color=INK,
            fontproperties=FP["s"], zorder=3)
    ax.text(0.55, 2.88, "레나델타 예시 · 블록 이분(증강 ½·평가 ½)", fontsize=7.4,
            color=GRAY, fontproperties=FP["m"], zorder=3)

    # 조건 3열 (자유 배치 + 세로 헤어라인)
    conds = [
        (3.78, ACC_D, "정확한 물리 · Stefan",
         r"$\tilde{y} = \hat{E}\sqrt{\mathrm{TDD}}$", "검증 게이트 채택식"),
        (5.58, INK, "부정확한 물리",
         r"$\tilde{y} = \mathrm{ALT}_{\mathrm{Ku}}$", "Kudryavtsev"),
        (7.38, INK, "상수 대조",
         r"$\tilde{y} = \bar{y}_{\mathrm{obs}} = 50.5\ \mathrm{cm}$", "라벨 정보 부재 대조군"),
    ]
    for cx_, tc, t, eq, note in conds:
        ax.text(cx_, 4.58, t, fontsize=9.4, color=tc, ha="center",
                fontproperties=FP["s"], zorder=3)
        ax.text(cx_, 4.14, eq, fontsize=9.6, color=INK, ha="center", zorder=3)
        ax.text(cx_, 3.72, note, fontsize=7.5, color=GRAY, ha="center",
                fontproperties=FP["m"], zorder=3)
    for xh in (4.68, 6.48):
        ax.plot([xh, xh], [3.58, 4.74], color=RULE_L, lw=0.8, zorder=2)

    # 합류 버스(직선) → 학습 셰브런
    bus_y = 5.02
    ax.plot([2.20, 10.90], [bus_y, bus_y], color="#3A3A3A", lw=1.2, zorder=3.5)
    ax.plot([1.74, 2.20], [4.62, 4.62], color="#3A3A3A", lw=1.0, zorder=3.5)
    ax.plot([2.20, 2.20], [4.62, bus_y], color="#3A3A3A", lw=1.0, zorder=3.5)
    for cx_, _, _, _, _ in conds:
        ax.plot([cx_, cx_], [4.74, bus_y], color="#3A3A3A", lw=1.0, zorder=3.5)
    sarrow(ax, (10.90, bus_y), (10.90, 4.74), lw=1.2)
    ax.text(9.10, 5.15, r"조건별 증강 학습집합 $D_r = D_{\mathrm{obs}} \cup \{(c_k,\ \tilde{y}_k)\}$",
            fontsize=8.5, color=GRAY2, ha="center", fontproperties=FP["m"], zorder=3)
    chevron5(ax, 9.55, 3.88, 2.72, 0.84, "동일 모델 학습", "CatBoost · 전이 공통 입력 25종")
    ax.text(10.85, 3.64, r"각 조건의 $D_r$로 적합 · $r \in \{0,\ 0.25,\ \dots,\ 10\}$",
            fontsize=7.9, color=GRAY2, ha="center", fontproperties=FP["m"], zorder=3)
    ax.plot([1.74, 3.78], [3.42, 3.42], color="#3A3A3A", lw=1.2, zorder=3.5)
    sarrow(ax, (3.78, 3.42), (3.78, 3.58), lw=1.2)
    ax.text(2.55, 3.53, "라벨 부여", fontsize=7.3, color=GRAY, ha="center",
            fontproperties=FP["m"], zorder=3)

    # ---- 레인 2 ----
    ohead(ax, 0.55, 2.40, "채점과 순가치 분리", fs=11)
    ax.text(2.62, 2.40, "평가 셀은 세 조건 공통 · 실측으로만 채점 (평가 조건 = 공변량만)",
            fontsize=8.4, color=GRAY, fontproperties=FP["m"], zorder=3)
    chevron5(ax, 0.55, 1.15, 2.32, 0.80, "평가 셀 실측 채점", "조건별 RMSE(r) 곡선",
             fc=GRN_FC, ec=GRN_EC)
    real_thumb(fig, ax, 3.32, 0.90, 2.05, 1.30, "outputs/figures/s3_aug/aug_response_curves.png",
               (0.115, 0.0, 1.00, 0.60), ec="#8A8A86", lw=0.9)
    ax.text(4.34, 0.72, "실데이터: 조건별 ΔRMSE(r) 반응곡선", fontsize=7.3, color=GRAY,
            ha="center", fontproperties=FP["m"], zorder=3)
    ax.text(7.90, 2.02, "물리 정보의 순가치", fontsize=9.8, color=ACC_D, ha="center",
            fontproperties=FP["x"], zorder=3)
    ax.text(7.90, 1.60,
            r"$\Delta_{\mathrm{phys}}(r) = [\mathrm{RMSE}_0 - \mathrm{RMSE}_r^{\mathrm{phys}}]"
            r" - [\mathrm{RMSE}_0 - \mathrm{RMSE}_r^{\mathrm{ctrl}}]$",
            fontsize=9.0, color=INK, ha="center", zorder=3)
    ax.text(7.90, 1.22, "Δ > 0 이면 개선의 원천은 라벨 개수가 아니라 물리 정보",
            fontsize=7.8, color=GRAY2, ha="center", fontproperties=FP["s"], zorder=3)
    real_thumb(fig, ax, 10.45, 0.90, 1.40, 1.30, "outputs/figures/s3_aug/physics_net_value.png",
               (0.16, 0.0, 1.0, 0.86), ec="#8A8A86", lw=0.9)
    ax.text(11.15, 0.72, "실데이터: 순가치 곡선", fontsize=7.3, color=GRAY,
            ha="center", fontproperties=FP["m"], zorder=3)

    # 레인 연결(직선 꺾임) 및 레인 2 흐름
    ax.plot([12.27, 12.42], [4.30, 4.30], color="#3A3A3A", lw=1.2, zorder=3.5)
    ax.plot([12.42, 12.42], [4.30, 2.62], color="#3A3A3A", lw=1.2, zorder=3.5)
    ax.plot([12.42, 2.50], [2.62, 2.62], color="#3A3A3A", lw=1.2, zorder=3.5)
    sarrow(ax, (2.50, 2.62), (2.50, 1.99), lw=1.2)
    ax.text(7.0, 2.49, "조건별 예측", fontsize=7.6, color=GRAY2, ha="center",
            fontproperties=FP["m"], zorder=4)
    sarrow(ax, (2.89, 1.55), (3.29, 1.55), lw=1.2)
    sarrow(ax, (5.40, 1.55), (5.86, 1.55), lw=1.2)
    sarrow(ax, (9.97, 1.55), (10.42, 1.55), lw=1.2)

    notation(ax, 0.55, 0.26,
             "표기   ỹ 유사라벨 · r 증강 비율(유사라벨/실측) · 첨자 0 = 증강 없음(r = 0) · "
             "phys 물리 라벨 조건 · ctrl 상수 대조 조건 · 상수 50.5 cm = 실측 평균 · 평가 셀은 세 조건에서 동일")
    save(fig, "fig_aug_design")


# ============================================================
# 4) fig_concept_alt — 활동층 개념 (v2: 질감 단면 + 축 겹침 수정)
# ============================================================
def fig_concept_alt():
    W, H = 6.5, 4.6
    fig = plt.figure(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(WHITE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], left=0.015, right=0.985,
                          top=0.90, bottom=0.115, wspace=0.30)
    ALT = 0.95
    ZMAX = 3.0
    rng = np.random.default_rng(5)

    # ---- (a) 지반 단면 (질감) ----
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 1); a.set_ylim(ZMAX, -0.42); a.axis("off")
    a.add_patch(Rectangle((0, -0.42), 1, 0.42, fc="#EAF3FA", ec="none"))
    # 유기물층 (지표 아래 얇은 암갈색 띠)
    a.add_patch(Rectangle((0, 0), 1, 0.10, fc="#6B5233", ec="none"))
    # 활동층: 갈색 토양 + 입자 질감
    a.add_patch(Rectangle((0, 0.10), 1, ALT - 0.10, fc="#D9BE92", ec="none"))
    spx = rng.uniform(0.02, 0.98, 380)
    spz = rng.uniform(0.12, ALT - 0.03, 380)
    sps = rng.uniform(2, 9, 380)
    a.scatter(spx, spz, s=sps, color="#B08D5A", alpha=0.55, lw=0, zorder=2)
    for _ in range(9):   # 자갈
        exx, ezz = rng.uniform(0.06, 0.94), rng.uniform(0.2, ALT - 0.1)
        a.add_patch(Ellipse((exx, ezz), rng.uniform(0.03, 0.06), rng.uniform(0.035, 0.06),
                            angle=rng.uniform(0, 180), fc="#9A7B4F", ec="none", alpha=0.75, zorder=2))
    # 영구동토: 한색 + 얼음 렌즈
    a.add_patch(Rectangle((0, ALT), 1, ZMAX - ALT, fc="#BFD3E4", ec="none"))
    for _ in range(26):
        lx0 = rng.uniform(0.02, 0.78)
        lz = rng.uniform(ALT + 0.10, ZMAX - 0.10)
        lw_ = rng.uniform(0.06, 0.24)
        a.add_patch(Ellipse((lx0 + lw_ / 2, lz), lw_, 0.035, fc="#EAF3FA",
                            ec="#9FBAD3", lw=0.4, alpha=0.9, zorder=2))
    # 지표 식생 (툰드라 풀 다발)
    for gx_ in np.linspace(0.04, 0.96, 22):
        gh = rng.uniform(0.05, 0.10)
        a.plot([gx_, gx_ - 0.008], [0, -gh], color="#5E7A3A", lw=1.0, zorder=4)
        a.plot([gx_, gx_ + 0.008], [0, -gh * 0.8], color="#6E8A46", lw=1.0, zorder=4)
    a.plot([0, 1], [0, 0], color="#4A3B25", lw=1.8, zorder=4)
    a.plot([0, 1], [ALT, ALT], color="#3D6B8E", lw=2.0, zorder=4)
    # ALT 화살표: 양 끝 캡(지표·영구동토 상단면)으로 구간을 명시
    ax_ = 0.115
    a.annotate("", xy=(ax_, ALT - 0.015), xytext=(ax_, 0.015),
               arrowprops=dict(arrowstyle="<|-|>", color=ACC, lw=2.6,
                               mutation_scale=16), zorder=5)
    for ye in (0.0, ALT):
        a.plot([ax_ - 0.045, ax_ + 0.045], [ye, ye], color=ACC, lw=3.0, zorder=6,
               solid_capstyle="butt")
    a.text(ax_ + 0.05, 0.40, "ALT", fontsize=15, color=ACC, fontproperties=FP["x"],
           va="center", zorder=5,
           bbox=dict(fc=WHITE, ec="none", alpha=0.65, pad=1.0))
    a.text(0.68, 0.40, "활동층 · 여름 융해·겨울 재동결", fontsize=11, color="#4A3418",
           fontproperties=FP["s"], ha="center", va="center", zorder=5,
           bbox=dict(fc="#F3E7D2", ec="none", alpha=0.8, pad=2.4))
    a.text(0.42, (ALT + ZMAX) / 2 + 0.12, "영구동토\n2년 이상 0°C 이하", fontsize=11.5,
           color="#2F5578", fontproperties=FP["s"], ha="center", va="center",
           linespacing=1.5, zorder=5,
           bbox=dict(fc="#E4EFF8", ec="none", alpha=0.8, pad=2.4))
    a.text(0.975, ALT - 0.07, "영구동토 상단면", fontsize=9.5, color="#3D6B8E",
           fontproperties=FP["m"], ha="right", zorder=5)
    a.text(0.02, -0.13, "지표 · 툰드라 식생", fontsize=9.5, color=GRAY2, fontproperties=FP["m"])
    a.set_title("(a) 지반 단면과 ALT", fontsize=12.5, color=INK, fontproperties=FP["s"], pad=8)

    # ---- (b) 트럼펫 곡선 ----
    b = fig.add_subplot(gs[1])
    z = np.linspace(0, ZMAX, 300)
    tmax = 9.0 * np.exp(-z / (ALT / np.log(3.0))) - 3.0
    tmin = -12.0 * np.exp(-z / 1.1) - 3.0
    b.fill_betweenx(z, tmin, tmax, color="#EDF1F5", zorder=1)
    b.plot(tmax, z, color=ACC, lw=2.0, zorder=3, label="연최대 지온")
    b.plot(tmin, z, color="#3D6B8E", lw=2.0, zorder=3, label="연최소 지온")
    b.axvline(0, color=GRAY2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    b.axhline(ALT, color=ACC, lw=1.2, ls=(0, (4, 3)), zorder=2)
    b.scatter([0], [ALT], s=42, color=ACC, zorder=4)
    b.annotate("0°C 교차 깊이 = ALT", xy=(0, ALT), xytext=(2.2, 1.75),
               fontsize=10.5, color=INK, fontproperties=FP["s"],
               arrowprops=dict(arrowstyle="->", color=GRAY2, lw=1.2))
    b.set_ylim(ZMAX, 0); b.set_xlim(-16, 12)
    b.set_xlabel("지온 (°C)", fontsize=10.5, fontproperties=FP["m"], color=INK)
    b.set_ylabel("깊이 (m)", fontsize=10.5, fontproperties=FP["m"], color=INK)
    b.tick_params(labelsize=9)
    for t in b.get_xticklabels() + b.get_yticklabels():
        t.set_fontproperties(FP["m"]); t.set_color(GRAY2)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        b.spines[s].set_color(GRAY)
    b.grid(alpha=0.22, lw=0.6)
    b.legend(loc="lower left", fontsize=9.5, frameon=False, prop=FP["m"])
    b.set_title("(b) 연최대·연최소 지온 프로파일", fontsize=12.5, color=INK,
                fontproperties=FP["s"], pad=8)
    save(fig, "fig_concept_alt")


# ============================================================
# 5) fig_label_gap — 공변량 조밀 vs 라벨 희소 (유지)
# ============================================================
def fig_label_gap():
    W, H = 6.6, 4.0
    fig = plt.figure(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(WHITE)
    gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.86, bottom=0.20, wspace=0.14)
    rng = np.random.default_rng(3)
    a = fig.add_subplot(gs[0])
    nx, ny = 46, 30
    xg, yg = np.meshgrid(np.linspace(0, 4, nx), np.linspace(0, 3, ny))
    field = (np.sin(1.3 * xg + 0.6) + 0.8 * np.cos(1.7 * yg - 0.4)
             + 0.55 * np.sin(2.3 * xg + 1.9 * yg) + 0.25 * rng.normal(size=xg.shape))
    a.imshow(field, cmap=matplotlib.colormaps["Blues"], alpha=0.9, aspect="auto",
             extent=(0, 1, 0, 1), vmin=field.min() - 0.6)
    a.set_xticks(np.linspace(0, 1, 13)); a.set_yticks(np.linspace(0, 1, 9))
    a.grid(color=WHITE, lw=0.5, alpha=0.8)
    a.set_xticklabels([]); a.set_yticklabels([])
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_color(GRAY); s.set_linewidth(1.2)
    a.set_title("공변량: 광역에서 조밀", fontsize=13, color=INK, fontproperties=FP["x"], pad=7)
    a.text(0.5, -0.115, "ERA5-Land 기후 · 토양 · 위성 CCI\n예측 대상 892,865 격자", fontsize=10.5,
           color=GRAY2, fontproperties=FP["m"], ha="center", va="top",
           transform=a.transAxes, linespacing=1.4)
    b = fig.add_subplot(gs[1])
    b.set_xlim(0, 1); b.set_ylim(0, 1)
    b.set_facecolor("#F4F5F3")
    centers = [(0.22, 0.74), (0.55, 0.62), (0.38, 0.30), (0.76, 0.42), (0.68, 0.82)]
    pts = []
    for k, (cxx, cyy) in enumerate(centers):
        m = [26, 14, 9, 7, 5][k]
        pts.append(np.column_stack([np.clip(rng.normal(cxx, 0.045, m), 0.04, 0.96),
                                    np.clip(rng.normal(cyy, 0.045, m), 0.05, 0.95)]))
    pts = np.vstack(pts)
    b.scatter(pts[:, 0], pts[:, 1], s=13, color="#1F3A52", alpha=0.9, lw=0)
    b.set_xticks([]); b.set_yticks([])
    for s in b.spines.values():
        s.set_color(GRAY); s.set_linewidth(1.2)
    b.set_title("실측 ALT 라벨: 지점에 희소", fontsize=13, color=INK, fontproperties=FP["x"], pad=7)
    b.text(0.5, -0.115, "13,606셀 = 대상 격자의 1.5%\n탐침·시추 관측, 소수 블록에 군집", fontsize=10.5,
           color=GRAY2, fontproperties=FP["m"], ha="center", va="top",
           transform=b.transAxes, linespacing=1.4)
    b.text(0.965, 0.055, "1.5%", fontsize=17, color=ACC, fontproperties=FP["x"],
           ha="right", transform=b.transAxes)
    save(fig, "fig_label_gap")


# ============================================================
# 6) fig_model_bars — 모델 비교 (덱 전용, 표 3 수치)
# ============================================================
def fig_model_bars():
    W, H = 6.4, 3.9
    data = [("MLP", 14.37), ("TabM", 14.40), ("CatBoost", 15.61), ("XGBoost", 16.16),
            ("LightGBM", 16.48), ("HistGBM", 17.21), ("FT-Transformer", 18.56)]
    fig = plt.figure(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0.205, 0.13, 0.775, 0.72])
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]
    cols = ["#8FA3B5"] * 5 + ["#1F3A52", "#1F3A52"]     # 상위 2개 진청
    yy = np.arange(len(vals))
    ax.axvspan(10, 12, color="#EDE7DA", zorder=1)
    ax.barh(yy, vals, height=0.58, color=cols, zorder=3)
    for y_, v in zip(yy, vals):
        ax.text(v + 0.3, y_, f"{v:.2f}", va="center", fontsize=10, color=GRAY2,
                fontproperties=FP["s"], zorder=5,
                bbox=dict(fc=WHITE, ec="none", pad=0.6))
    ax.axvline(14.56, color=ACC, lw=1.3, ls=(0, (4, 2.5)), zorder=2)
    ax.text(14.3, len(vals) - 0.02, "Stefan 물리식 14.56", fontsize=9.5, color=ACC,
            ha="right", fontproperties=FP["s"], zorder=4)
    ax.axvline(17.38, color=GRAY, lw=1.3, ls=(0, (4, 2.5)), zorder=2)
    ax.text(17.62, len(vals) - 0.02, "평균 예측 17.38", fontsize=9.5, color=GRAY,
            ha="left", fontproperties=FP["s"], zorder=4)
    ax.add_patch(Rectangle((0.35, len(vals) - 0.18), 0.55, 0.32, fc="#EDE7DA", ec="none", zorder=4))
    ax.text(1.05, len(vals) - 0.02, "지점 관측 대표성 하한 10–12 cm", fontsize=8.5,
            color="#7A6A4C", ha="left", fontproperties=FP["m"], zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=10.5, color=INK, fontproperties=FP["s"])
    ax.set_xlim(0, 21); ax.set_ylim(-0.6, len(vals) + 0.35)
    ax.set_xlabel("RMSE (cm)", fontsize=10.5, color=INK, fontproperties=FP["m"])
    for t in ax.get_xticklabels():
        t.set_fontproperties(FP["m"]); t.set_color(GRAY2); t.set_fontsize(9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.grid(axis="x", alpha=0.22, lw=0.6, zorder=0)
    ax.set_title("기계학습·딥러닝 7종의 지역 내 예측 오차 (입력 34종, 3-seed 앙상블)",
                 fontsize=11.5, color=INK, fontproperties=FP["s"], pad=10)
    save(fig, "fig_model_bars")


# ============================================================
# 7) fig_aug_spatial — 증강 효과: 확대 지도 2 + 오차·편향 감소 곡선
# ============================================================
def fig_aug_spatial():
    """v3: 관측 밀집 구역 확대 + 공유 컬러바 명시 + 곡선. 세 지도 동일 색 스케일."""
    import pandas as pd
    from PIL import ImageEnhance
    W, H = 12.4, 5.0
    fig = plt.figure(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(WHITE)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.15, 1.0, 1.30],
                          left=0.012, right=0.985, top=0.875, bottom=0.125, wspace=0.14)
    src_im = Image.open("deck/assets/final/crops/lena_maps.png")
    panels = [
        (0, (141, 186, 641, 758), "(a) 관측 ALT", "실측 융해 깊이 (탐침)", GRAY, 1.45),
        (1, (1076, 186, 1576, 758), "(b) 증강 전 예측 · r = 0", "RMSE 21.9 · 과대예측 +16.2 cm", GRAY, 1.45),
        (3, (2867, 230, 3335, 720), "(c) 증강 전후 변화 (r=1 − r=0)",
         "파란색 = 하향 조정(±30 cm 스케일) · RMSE 17.0", "#B25E0F", 2.0),
    ]
    for gi, box, title, cap, cc, sat in panels:
        a = fig.add_subplot(gs[gi])
        im = ImageEnhance.Color(src_im.crop(box).convert("RGB")).enhance(sat)
        if gi == 3:
            im = ImageEnhance.Contrast(im).enhance(1.15)
        a.imshow(np.asarray(im))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(GRAY); sp.set_linewidth(1.0)
        a.set_title(title, fontsize=11, color=INK, fontproperties=FP["s"], pad=6)
        a.text(0.5, -0.065, cap, fontsize=9.0, color=cc,
               fontproperties=FP["s" if cc != GRAY else "m"], ha="center", transform=a.transAxes)
    # 컬러바 2종: (a)(b) ALT 스케일 · (c) 변화량 스케일
    cb = fig.add_subplot(gs[2])
    cb.imshow(np.asarray(src_im.crop((856, 195, 1005, 995)).convert("RGB")))
    cb.set_xticks([]); cb.set_yticks([])
    for sp in cb.spines.values():
        sp.set_visible(False)
    cb.set_title("ALT (cm)", fontsize=8.2, color=GRAY2, fontproperties=FP["m"], pad=5)
    fig.text(0.35, 0.028, "(a)(b) 관측 밀집 구역 확대·동일 색 스케일 · (c) 변화량(±30 cm) · 채도 보정 표시본 · r = 1 예시",
             fontsize=8.8, color="#8E8E8E", ha="center", fontproperties=FP["m"])

    # (d) 증강 비율에 따른 RMSE·편향 감소
    c = fig.add_subplot(gs[4])
    df = pd.read_csv("data/processed/s3_aug_curve_results.csv")
    d_ = df[(df.target == "Lena") & (df.model == "catboost") & (df.phys == "stefan")]
    g = d_.groupby("r")[["rmse_cm", "bias_cm"]].mean().reset_index()
    xs = np.arange(len(g))
    c.plot(xs, g.rmse_cm, color=ACC, lw=2.2, marker="o", ms=5.5, label="RMSE", zorder=3)
    c.plot(xs, g.bias_cm, color="#3D6B8E", lw=2.0, marker="s", ms=5, ls=(0, (4, 2)),
           label="평균 편향 (과대예측)", zorder=3)
    for i, (rv, bv) in [(0, (g.rmse_cm.iloc[0], g.bias_cm.iloc[0])),
                        (len(g) - 1, (g.rmse_cm.iloc[-1], g.bias_cm.iloc[-1]))]:
        c.annotate(f"{rv:.1f}", (xs[i], rv), xytext=(2 if i == 0 else 0, 9),
                   textcoords="offset points", fontsize=10, color=ACC,
                   ha="left" if i == 0 else "center", fontproperties=FP["x"])
        c.annotate(f"+{bv:.1f}", (xs[i], bv), xytext=(10, -6 if i == 0 else -14),
                   textcoords="offset points", fontsize=10, color="#3D6B8E",
                   ha="left", fontproperties=FP["x"])
    c.set_xticks(xs)
    c.set_xticklabels([f"{v:g}" for v in g.r], fontsize=9.5)
    c.set_xlabel("증강 비율 r (유사라벨/실측)", fontsize=10.5, color=INK, fontproperties=FP["m"])
    c.set_ylabel("cm", fontsize=10.5, color=INK, fontproperties=FP["m"])
    c.set_ylim(0, 26)
    for t in c.get_xticklabels() + c.get_yticklabels():
        t.set_fontproperties(FP["m"]); t.set_color(GRAY2)
    for sp in ("top", "right"):
        c.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        c.spines[sp].set_color(GRAY)
    c.grid(alpha=0.22, lw=0.6)
    c.legend(loc="center right", fontsize=9.5, frameon=False, prop=FP["m"])
    c.set_title("(d) 증강 비율에 따른 오차·편향 감소", fontsize=11.5, color=INK,
                fontproperties=FP["s"], pad=6)
    c.text(0.98, 0.03, "CatBoost · Stefan 유사라벨 · 3-seed 평균", fontsize=8.5, color="#8E8E8E",
           ha="right", transform=c.transAxes, fontproperties=FP["m"])
    save(fig, "fig_aug_spatial")


# ============================================================
# 8) 색 보정 사본 — 지도 표시본 채도·대비·밝기 조정 (원본 불변)
# ============================================================
def enhance_copies():
    from PIL import ImageEnhance
    jobs = [
        ("outputs/maps/data_inventory_world.png", f"{OUT}/inv_world_boost.png", 1.22, 1.06, 1.0),
        ("outputs/figures/s0_schema/spatial_block_folds_map.png", f"{OUT}/folds_boost.png", 1.22, 1.06, 1.0),
    ]
    for src, dst, color, contrast, bright in jobs:
        im = Image.open(src).convert("RGB")
        im = ImageEnhance.Color(im).enhance(color)
        im = ImageEnhance.Contrast(im).enhance(contrast)
        im = ImageEnhance.Brightness(im).enhance(bright)
        im.save(dst)
        print("boosted", dst)
    # 연별 지도: 감마 보정으로 어두운 구간만 들어올림 (2016·2019 심색부 완화)
    im = Image.open("outputs/figures/s9_timelapse/alt_annual_fields.png").convert("RGB")
    im = im.point(lambda v: int(255 * ((v / 255) ** (1 / 1.28))))
    im = ImageEnhance.Color(im).enhance(0.86)
    im.save(f"{OUT}/annual_soft.png")
    print("boosted", f"{OUT}/annual_soft.png")


if __name__ == "__main__":
    fig_workflow()
    fig_dl_arch()
    fig_aug_design()
    fig_concept_alt()
    fig_label_gap()
    fig_model_bars()
    fig_aug_spatial()
    enhance_copies()
