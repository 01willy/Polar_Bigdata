# -*- coding: utf-8 -*-
"""
build_deck.py — Polar_Bigdata 진행상황 보고 덱 빌드.
스펙: deck/deck_spec.json · 라이브러리: deck/polar_slides.py
수치 근거: data/processed/*_results.csv (조사 워크플로 검증) · 색: design/brand_tokens.json
실행: cd deck && python3 build_deck.py → render/permafrost_progress.pptx
"""
import os
import polar_slides as S
from polar_slides import (Inches, Pt, PP_ALIGN, MSO_ANCHOR, RGBColor,
                          INK, NAVY, TEAL, TEAL_LT, PURPLE, WARN, SLATE, GRAY,
                          HAIR, HAIR_LT, PANEL, PANEL2, WHITE, NAVY_DK, NAVY_DK2,
                          ML, MR, MT, CONTENT_W, CONTENT_L, CONTENT_R, SLIDE_W_IN, SLIDE_H_IN,
                          F_SEMI, F_MED, F_REG, F_LIGHT)

def R(p):  # 그림 경로(덱 기준 상대)
    return os.path.join("..", p)

TOTAL = 16
prs = S.new_deck()

# ============================================================ 01 표지
S.title_slide(
    prs,
    "영구동토 활성층 두께·얕은 3D 열구조 예측",
    "다중모달 빅데이터 융합 · 정직한 평가 · 알래스카 학습 → 타 영구동토 지대 전이 검증",
    ["2026-07-09  ·  극지 빅데이터 연구",
     "근거: data/processed/*_results.csv  ·  색규약: design/brand_tokens.json"],
    hero_path=R("outputs/volumes_3d/hero_3d_permafrost.png"),
    phase_tag="진행상황 보고")

# ============================================================ 02 개요
sl = S.blank(prs)
y = S.header(sl, "개요", "연구 목적과 발표 순서")
# 좌: 한 줄 목적
S.panel(sl, ML, y + 0.15, 5.55, 3.9, fill=PANEL2, line=HAIR)
S.rect(sl, ML, y + 0.15, 0.08, 3.9, color=TEAL)
S.text(sl, ML + 0.3, y + 0.42, 4.95, 0.4,
       [{'t': "연구 목적", 'size': 13, 'color': NAVY, 'font': F_SEMI, 'spacing': 0.4}])
S.text(sl, ML + 0.3, y + 0.9, 4.95, 3.0,
       [[{'t': "전지구 시추공 지중온도와 CALM·ABoVE 활성층 관측을 위성·재분석·지형 공변량과 융합해 "
               "활성층 두께(ALT) 2D 지도와 얕은 3D 지중 열구조·셀별 불확실성을 산출한다.",
          'size': 14.5, 'color': INK, 'font': F_REG}],
        [{'t': "알래스카에서 학습한 모델이 타 영구동토 지대로 전이되는지 누설을 통제한 평가로 검증한다.",
          'size': 14.5, 'color': INK, 'font': F_REG}]],
       line_spacing=1.42, space_after=10)
# 우: 목차
agenda = [("01", "연구 목적", "무엇을 입력받아 무엇을 산출하는가"),
          ("02", "입력 데이터", "22GB+ 다중모달 · 공변량 구성"),
          ("03", "모델 구조", "2D·3D 파이프라인과 토너먼트"),
          ("04", "정직한 평가", "누설 통제 · 정보병목 진단"),
          ("05", "핵심 결과", "지도 · 지배 정보원 · UQ · 3D"),
          ("06", "기존 연구 대비", "선행연구 포지셔닝"),
          ("07", "다음 단계", "데이터 확장 · 3D · 시각화")]
ax, ay = 6.55, y + 0.2
for i, (no, t, d) in enumerate(agenda):
    yy = ay + i * 0.545
    S.text(sl, ax, yy, 0.55, 0.4, [{'t': no, 'size': 15, 'color': TEAL, 'font': F_SEMI}],
           anchor=MSO_ANCHOR.MIDDLE)
    S.text(sl, ax + 0.6, yy, 2.1, 0.4, [{'t': t, 'size': 13.5, 'color': INK, 'font': F_SEMI}],
           anchor=MSO_ANCHOR.MIDDLE)
    S.text(sl, ax + 2.75, yy, 3.3, 0.4, [{'t': d, 'size': 11, 'color': SLATE, 'font': F_REG}],
           anchor=MSO_ANCHOR.MIDDLE)
    if i < len(agenda) - 1:
        S.hline(sl, ax, yy + 0.5, 5.75, color=HAIR_LT, weight=0.75)
S.footer(sl, 2, TOTAL)

# ============================================================ 03 연구 목적 (개념도)
sl = S.blank(prs)
y = S.header(sl, "01 · 연구 목적", "관측·공변량 → ALT 2D 지도 · 얕은 3D 열구조 · 셀별 불확실성")
S.bullets(sl, ML, y + 0.2, 5.75, 4.2,
          [("목표", "ALT 2D 지도 · 얕은 3D 지중 열구조(0–20m) · 셀별 불확실성을 동시에 산출한다."),
           ("입력", "전지구 시추공 지중온도 + CALM·ABoVE ALT 관측 + 위성·재분석·지형 공변량."),
           ("산출", "ALT 예측면(cm) + 0°C 등온면(영구동토 경계) + 90% 예측구간."),
           ("전이", "알래스카에서 학습해 타 영구동토 지대로 일반화되는지 검증한다.")],
          size=14.5, gap=13)
fb = S.image_fit(sl, R("outputs/figures/00_concept/research_concept.png"),
                 6.7, y + 0.15, 5.95, 3.75, valign='top')
S.caption(sl, 6.7, fb[1] + fb[3] + 0.08, 5.95,
          "연구 개념 단면: 시추공 온도·활성층·공변량을 입력해 3D 영구동토 구조를 복원한다.",
          source="outputs/figures/00_concept/research_concept.png")
S.footer(sl, 3, TOTAL)

# ============================================================ 04 입력 데이터
sl = S.blank(prs)
y = S.header(sl, "02 · 입력 데이터", "22GB+ 다중모달 빅데이터 — 알래스카 학습 · 전지구 전이 구조")
rows = [["데이터", "역할", "규모"],
        ["CALM ALT", "전이 검증 라벨", "3,604 site-yr · 259 사이트"],
        ["ABoVE ALT", "딥러닝 학습 라벨", "223,937 점"],
        ["InSAR ReSALT", "약지도 사전학습", "403.6만 점"],
        ["GTN-P 지중온도", "3D 열구조 라벨", "10,747 점 · 260 사이트"],
        ["ERA5-Land", "기후 공변량", "9km · 월별"],
        ["Copernicus DEM", "지형 공변량", "30m · 234 타일"],
        ["PolSAR / ESA CCI", "물리관측 · prior", "P-band 30m / 1km"]]
S.styled_table(sl, ML, y + 0.18, 6.5, rows, col_w=[1.95, 2.0, 2.55],
               size=11, row_h=0.44, head_h=0.42)
# 우: 커버리지 지도 + KPI
fb = S.image_fit(sl, R("outputs/figures/01_data/global_alt_coverage.png"),
                 7.5, y + 0.15, 5.15, 2.15, valign='top')
S.caption(sl, 7.5, fb[1] + fb[3] + 0.06, 5.15,
          "CALM 전지구 관측망(259 사이트·12개국) — 전이 검증 범위.",
          source="outputs/figures/01_data/global_alt_coverage.png")
ky = fb[1] + fb[3] + 0.6
S.kpi(sl, 7.5, ky, 2.5, 1.5, "22", "GB+", "다중모달 원자료", accent=NAVY,
      sub="위성SAR·재분석·지형·시추공")
S.kpi(sl, 10.15, ky, 2.5, 1.5, "225,421", "점", "ALT 학습셋", accent=TEAL,
      sub="고유위치 14,348")
S.footer(sl, 4, TOTAL)

# ============================================================ 05 공변량 + 약지도
sl = S.blank(prs)
y = S.header(sl, "02 · 입력 데이터", "공변량 14종 + 물리관측 4종 · InSAR 약지도로 라벨 확장")
groups = [
    ("지형 · DEM (6)", NAVY, ["고도", "경사", "사면향 sin·cos", "지형위치지수 TPI", "지표 거칠기"]),
    ("기후 · ERA5-Land (8)", TEAL, ["연평균기온 MAAT", "해동도일 TDD", "동결도일 FDD", "√TDD (Stefan)",
                                     "최난·최한월 기온", "표층 토양온도", "적설수당량 SWE"]),
    ("물리관측 · PHYS (4)", PURPLE, ["PolSAR ALT (P-band)", "PolSAR 셀내 SD",
                                      "InSAR ALT (ReSALT)", "InSAR 계절 침하"]),
]
gw = (CONTENT_W - 2 * 0.35) / 3
for i, (title, acc, items) in enumerate(groups):
    gx = ML + i * (gw + 0.35)
    S.panel(sl, gx, y + 0.15, gw, 1.95, fill=WHITE, line=HAIR)
    S.rect(sl, gx, y + 0.15, gw, 0.09, color=acc)
    S.text(sl, gx + 0.22, y + 0.3, gw - 0.4, 0.36,
           [{'t': title, 'size': 12.5, 'color': acc, 'font': F_SEMI}], anchor=MSO_ANCHOR.MIDDLE)
    S.text(sl, gx + 0.22, y + 0.72, gw - 0.44, 1.3,
           [[{'t': "· ", 'size': 11.5, 'color': acc, 'font': F_SEMI},
             {'t': it, 'size': 11.5, 'color': INK, 'font': F_REG}] for it in items],
           line_spacing=1.2, space_after=2.2)
fb = S.image_fit(sl, R("outputs/maps/weaklabels_overview.png"),
                 ML, y + 2.35, CONTENT_W, 2.15, valign='top')
S.caption(sl, ML, fb[1] + fb[3] + 0.06, CONTENT_W,
          "InSAR 약지도 403.6만 점으로 실측 대비 라벨 밀도를 확대하고 30m 세밀도를 확보(사전학습 신호).",
          source="outputs/maps/weaklabels_overview.png", align=PP_ALIGN.CENTER)
S.footer(sl, 5, TOTAL)

# ============================================================ 06 모델 구조 (네이티브 다이어그램)
sl = S.blank(prs)
y = S.header(sl, "03 · 모델 구조", "2D ALT 경로와 3D 조건장 경로 — 평가·보정 UQ·적용범위가 횡단한다")

def agroup(x, w, top, h, title, accent, chips, chip_h=0.52, note=None):
    S.panel(sl, x, top, w, h, fill=WHITE, line=HAIR)
    S.rect(sl, x, top, w, 0.075, color=accent)
    S.text(sl, x + 0.16, top + 0.14, w - 0.3, 0.3,
           [{'t': title, 'size': 11.5, 'color': accent, 'font': F_SEMI}], anchor=MSO_ANCHOR.MIDDLE)
    cy = top + 0.52
    for c in chips:
        S.chip(sl, x + 0.16, cy, w - 0.32, chip_h, c, fill=PANEL, txt=INK, size=10.5, font=F_MED)
        cy += chip_h + 0.12
    return cy

top = y + 0.12
gh = 3.35
x1, w1 = ML, 2.55
x2, w2 = 3.55, 2.55
x3, w3 = 6.55, 3.35
x4, w4 = 10.25, 2.4
agroup(x1, w1, top, gh, "관측 · 라벨", SLATE,
       ["CALM · ABoVE ALT 관측", "GTN-P 지중온도 0–20m", "InSAR 약지도 403.6만"])
agroup(x2, w2, top, gh, "공변량 (14 + 4)", NAVY,
       ["지형 6 · DEM 30m", "기후 8 · ERA5-Land", "물리관측 4 · PolSAR/InSAR"])
# 모델 그룹(2D/3D 구분)
S.panel(sl, x3, top, w3, gh, fill=WHITE, line=HAIR)
S.rect(sl, x3, top, w3, 0.075, color=TEAL)
S.text(sl, x3 + 0.16, top + 0.14, w3 - 0.3, 0.3,
       [{'t': "모델", 'size': 11.5, 'color': TEAL, 'font': F_SEMI}], anchor=MSO_ANCHOR.MIDDLE)
S.text(sl, x3 + 0.16, top + 0.52, w3 - 0.32, 0.24,
       [{'t': "2D ALT", 'size': 9.5, 'color': TEAL_LT, 'font': F_SEMI, 'spacing': 0.5}])
cy = top + 0.78
for c in ["B0 사전학습 MLP + 미세조정", "B0b +CCI · (DL+GBM) 앙상블", "토너먼트 6모델 (GBM≈DL)"]:
    S.chip(sl, x3 + 0.16, cy, w3 - 0.32, 0.42, c, fill=PANEL, txt=INK, size=10, font=F_MED)
    cy += 0.5
S.text(sl, x3 + 0.16, cy + 0.02, w3 - 0.32, 0.24,
       [{'t': "얕은 3D", 'size': 9.5, 'color': TEAL_LT, 'font': F_SEMI, 'spacing': 0.5}])
S.chip(sl, x3 + 0.16, cy + 0.28, w3 - 0.32, 0.42, "GBM 조건장 + 깊이인코딩 (NF 탈락)",
       fill=PANEL, txt=INK, size=10, font=F_MED)
agroup(x4, w4, top, gh, "산출", PURPLE,
       ["ALT 2D 지도 (cm)", "얕은 3D 열큐브 · 0°C 등온면", "셀별 90% 예측구간"])
# 화살표
ay = top + gh / 2 - 0.15
for (ax0, aw0, ax1) in [(x1 + w1, x2 - (x1 + w1), x2),
                        (x2 + w2, x3 - (x2 + w2), x3),
                        (x3 + w3, x4 - (x3 + w3), x4)]:
    S.arrow(sl, ax0 + 0.06, ay, (ax1 - ax0) - 0.12, 0.3, color=GRAY)
# 하단 횡단 밴드
by = top + gh + 0.22
S.panel(sl, ML, by, CONTENT_W, 0.72, fill=PANEL2, line=HAIR)
S.text(sl, ML + 0.2, by, 3.0, 0.72,
       [{'t': "횡단 평가·UQ", 'size': 11, 'color': NAVY, 'font': F_SEMI}], anchor=MSO_ANCHOR.MIDDLE)
bx = ML + 2.7
for lab in ["공간블록 · LORO CV (누설 통제)", "Conformal CQR (coverage 89%)", "AOA 적용범위 마스크"]:
    S.chip(sl, bx, by + 0.13, 3.0, 0.46, lab, fill=WHITE, txt=INK, line=HAIR, size=10.5, font=F_MED)
    bx += 3.12
S.footer(sl, 6, TOTAL)

# ============================================================ 07 토너먼트 / 정보병목
sl = S.blank(prs)
y = S.header(sl, "03 · 모델 구조", "정적 ALT에서는 GBM ≈ 딥러닝 — 병목은 공변량 정보이다")
kx, kw = ML, 2.02
for i, (v, lab, acc) in enumerate([("16.95", "앙상블 (GBM+FT-T)", PURPLE),
                                    ("17.09", "Diffusion", TEAL),
                                    ("17.24", "GBM", NAVY)]):
    S.kpi(sl, kx + i * (kw + 0.18), y + 0.15, kw, 1.35, v, "cm", lab, accent=acc,
          sub=None)
S.bullets(sl, ML, y + 1.75, 6.55, 3.0,
          [("부트스트랩 동률", "6모델 전부 GBM과 신뢰구간이 겹친다(유의차 없음)."),
           ("정보병목", "예측이 평균으로 수축한다 — 한계는 모델 용량이 아니라 공변량이다."),
           ("Diffusion 채택", "정확도가 아니라 네이티브 불확실성(UQ) 때문이다.")],
          size=13.5, gap=11)
fb = S.image_fit(sl, R("outputs/figures/06_deep_learning/pred_vs_obs.png"),
                 7.55, y + 0.15, 5.1, 3.85, valign='top')
S.caption(sl, 7.55, fb[1] + fb[3] + 0.06, 5.1,
          "예측 vs 관측(hexbin) · pooled OOF 17.8cm(토너먼트 6-fold 16.95–17.24cm와 정합) · 50–75cm로 수축 = 정보병목",
          source="model_tournament_results.csv · 공간블록 CV")
S.footer(sl, 7, TOTAL)

# ============================================================ 08 정직한 평가 ① CV 누설
sl = S.blank(prs)
y = S.header(sl, "04 · 정직한 평가", "무작위 CV는 전이를 과대평가한다 — 공간블록·LORO만 신뢰한다")
fb = S.image_fit(sl, "assets/cv_leakage_cool.png",
                 ML, y + 0.2, 6.9, 4.0, valign='top')
S.caption(sl, ML, fb[1] + fb[3] + 0.06, 6.9,
          "CV 설계별 baseline ALT RMSE — 무작위 K-fold는 낙관, 지역분리 LORO에서 붕괴",
          source="data/processed/cv_leakage_table.csv")
S.bullets(sl, 7.7, y + 0.35, 5.0, 3.6,
          [("문제", "무작위 K-fold는 인접점 누설로 오차를 낙관한다(IDW 28cm)."),
           ("정직한 평가", "지역분리 LORO에서 같은 방법이 111cm로 붕괴한다."),
           ("설계", "헤드라인은 공간블록·LORO CV만 쓰고, RMSE에 R²·skill을 병기한다.")],
          size=13.5, gap=13)
S.footer(sl, 8, TOTAL)

# ============================================================ 09 정직한 평가 ② apparent floor
sl = S.blank(prs)
y = S.header(sl, "04 · 정직한 평가", "17cm은 물리벽이 아니라 정보병목이다 — 비가역하한 약 7cm")
for i, (v, u, lab, acc, sub) in enumerate([
        ("86 / 14", "%", "분산분해", NAVY, "위치간 / 위치내"),
        ("7.16", "cm", "비가역하한", TEAL, "현재 잡음 하한"),
        ("16.95", "cm", "현재 최고 RMSE", WARN, "하한보다 훨씬 큼")]):
    S.kpi(sl, ML + i * 2.55, y + 0.15, 2.35, 1.5, v, u, lab, accent=acc, sub=sub)
S.bullets(sl, ML, y + 1.95, CONTENT_W, 2.6,
          [("폐기된 표현", "‘17cm = 물리하한’은 폐기한다."),
           ("진단", "같은 공변량 셀에 서로 다른 ALT가 공존(pseudo-replication)하고 라벨 30m와 기후 9km의 척도가 불일치한다."),
           ("함의", "비가역하한(≈7cm)이 현재 최고(16.95cm)보다 훨씬 낮다 — 정보병목이며 헤드룸이 남아 있다.")],
          size=14, gap=12)
S.caption(sl, ML, y + 4.55, CONTENT_W,
          "분산분해로 위치간 변동이 86%를 차지 — 남은 오차는 새 모달리티·척도정합으로 줄일 여지가 있다.",
          source="data/processed/apparent_floor_diagnosis.csv")
S.footer(sl, 9, TOTAL)

# ============================================================ 10 결과 ① ALT 지도
sl = S.blank(prs)
y = S.header(sl, "05 · 핵심 결과", "알래스카 전역 ALT 예측면 — 관측 오버레이와 함께 복원")
fb = S.image_fit(sl, R("outputs/maps/alt_alaska_pred.png"),
                 ML, y + 0.15, 7.55, 4.75, valign='top', align='left')
S.caption(sl, ML, fb[1] + fb[3] + 0.06, 7.55,
          "ERA5-Land 공변량 + GBM 조건 · 순차형(옅음=얕음, 짙음=깊음), 20–110cm",
          source="outputs/maps/alt_alaska_pred.png · 공간블록 CV")
sx = fb[0] + fb[2] + 0.45
S.bullets(sl, sx, y + 1.35, 12.6 - sx, 3.2,
          [("예측면", "알래스카 전역의 활성층 두께를 연속면으로 복원한다."),
           ("관측 오버레이", "CALM 1,174 · ABoVE 221 지점을 함께 표시한다."),
           ("색규약", "붉은 계열·rainbow를 쓰지 않는 순차형(냉색)이다.")],
          size=13.5, gap=16)
S.footer(sl, 10, TOTAL)

# ============================================================ 11 결과 ② 지배 정보원
sl = S.blank(prs)
y = S.header(sl, "05 · 핵심 결과", "지배 정보원은 과제마다 다르다 — 보간=기후, 전이=InSAR")
fb = S.image_fit(sl, R("outputs/figures/06_deep_learning/alt_feature_ablation.png"),
                 ML, y + 0.2, 7.0, 3.05, valign='top', align='left')
S.caption(sl, ML, fb[1] + fb[3] + 0.05, 7.0,
          "다중모달 feature ablation(M0–M6) — 공간블록(보간)과 LORO(전이)에서 지배 모달리티가 다름.",
          source="data/processed/alt_feature_ablation_results.csv")
kx = 8.05
for i, (v, u, lab, acc, sub) in enumerate([("16.4", "cm", "+InSAR · 전이 최고", TEAL, "LORO ablation M4"),
                                            ("20", "%", "ERA5 전이 개선", NAVY, "LORO 108.5→87.3cm")]):
    S.kpi(sl, kx + i * 2.4, y + 0.15, 2.25, 1.5, v, u, lab, accent=acc, sub=sub)
S.bullets(sl, kx, y + 1.85, 4.6, 2.8,
          [("보간(공간블록)", "기후(ERA5)가 지배 · 지형 추가는 과적합으로 악화."),
           ("전이(LORO)", "InSAR가 필수 · 지형 단독은 파탄(skill −65%)."),
           ("공간 DL", "패치CNN 17.2 ≈ GBM 17.7cm — 병목은 맥락이 아닌 정보.")],
          size=12.5, gap=10)
S.footer(sl, 11, TOTAL)

# ============================================================ 12 결과 ③ 보정 UQ + AOA
sl = S.blank(prs)
y = S.header(sl, "05 · 핵심 결과", "예측구간을 conformal로 보정(71→89%)하고 AOA로 신뢰 밖을 표기한다")
fbl = S.image_fit(sl, R("outputs/figures/02_evaluation/coverage_calibration.png"),
                  ML, y + 0.2, 5.5, 3.15, valign='top')
S.caption(sl, ML, fbl[1] + fbl[3] + 0.06, 5.5,
          "raw 분위 회귀 71.2% → CQR 보정 89.2%(목표 90%).",
          source="data/processed/alt_conformal_aoa_results.csv", align=PP_ALIGN.CENTER)
fbr = S.image_fit(sl, R("outputs/maps/alt_aoa_mask.png"),
                  6.75, y + 0.5, 5.9, 2.55, valign='top')
S.caption(sl, 6.75, fbr[1] + fbr[3] + 0.06, 5.9,
          "적용범위(AOA): 환경 비유사도↑ → 오차↑·커버리지↓, 밖 영역은 경고 표기.",
          source="data/processed/alt_aoa_transfer_results.csv", align=PP_ALIGN.CENTER)
S.chip(sl, 6.75, y + 3.55, 2.85, 0.5, "보정: 71.2% → 89.2%", fill=PANEL2, txt=NAVY, size=11)
S.chip(sl, 9.78, y + 3.55, 2.87, 0.5, "밖: RMSE 15.5→27.1cm", fill=PANEL2, txt=WARN, size=11)
S.footer(sl, 12, TOTAL)

# ============================================================ 13 결과 ④ 얕은 3D
sl = S.blank(prs)
y = S.header(sl, "05 · 핵심 결과", "위도를 따라 영구동토 경계 심도가 변하는 얕은 3D 열구조")
fbl = S.image_fit(sl, R("outputs/volumes_3d/cross_section_lat_depth.png"),
                  ML, y + 0.2, 6.2, 3.2, valign='top')
S.caption(sl, ML, fbl[1] + fbl[3] + 0.06, 6.2,
          "위도–깊이 온도 단면 · 0°C 등온선 = 영구동토 경계(남부 온난 → 북부 심부 한랭).",
          source="outputs/volumes_3d/cross_section_lat_depth.png", align=PP_ALIGN.CENTER)
fbr = S.image_fit(sl, R("outputs/figures/05_alaska_pilot/depth_slices.png"),
                  7.15, y + 0.2, 5.5, 2.55, valign='top')
S.caption(sl, 7.15, fbr[1] + fbr[3] + 0.05, 5.5,
          "깊이별(0–70m) 평균 지중온도(MAGT) — 깊이에 따른 도달 시추공 감소.",
          source="outputs/figures/05_alaska_pilot/depth_slices.png", align=PP_ALIGN.CENTER)
S.text(sl, 7.15, y + 3.35, 5.5, 1.0,
       [[{'t': "엔진 ", 'size': 11.5, 'color': TEAL, 'font': F_SEMI},
         {'t': "= GBM 조건장 + 깊이인코딩(신경장은 게이트 탈락). ", 'size': 11.5, 'color': INK, 'font': F_REG}],
        [{'t': "구분 ", 'size': 11.5, 'color': TEAL, 'font': F_SEMI},
         {'t': "ALT(계절 최대융해) ≠ MAGT(연평균장).", 'size': 11.5, 'color': INK, 'font': F_REG}]],
       line_spacing=1.3, space_after=3)
S.footer(sl, 13, TOTAL)

# ============================================================ 14 기존 연구 대비
sl = S.blank(prs)
y = S.header(sl, "06 · 기존 연구 대비", "규모 × 전이 × 보정 UQ × 얕은 3D의 결합이 차별점이다")
prows = [["선행연구", "접근", "우리 차별점"],
         ["Ran 2022\n(pan-Arctic ML)", "1km MAGT/ALT 앙상블 지도\n(RMSE 86.9cm / 1.32°C)", "명시적 지역 전이 · 셀별 보정 UQ · 얕은 3D 추가"],
         ["ESA CCI v4\n(Permafrost)", "CryoGrid forward 1km 제품\n(관측기반 아님)", "관측기반 보간·UQ·전이 · 우리 모델이 CCI 20.8cm 격파(공간블록 CV)"],
         ["Gautam 2025\n(Alaska ALT)", "RF vs Stefan · 알래스카 내부\n(RF test R² 0.24)", "지역 전이 벤치마크 · 셀별 UQ · 4D 구조"],
         ["Obu 2019\n(TTOP NH)", "평형 물리모델 MAGT·extent\n(RMSE ~1.48°C)", "ALT 대상 · ML 셀별 UQ · 전이 평가"],
         ["Meyer 2021 · Romano 2019", "AOA 적용범위 · 분위 conformal", "영구동토 ALT에 처음 결합 · 이진→연속 예측구간"]]
S.styled_table(sl, ML, y + 0.15, CONTENT_W, prows, col_w=[2.6, 4.3, 5.0],
               size=10.5, row_h=0.66, head_h=0.4)
S.caption(sl, ML, y + 4.35, CONTENT_W,
          "전이·셀별 UQ는 리뷰(Koven 2025)가 명명한 open gap — 본 연구가 LORO 벤치마크와 보정 UQ로 실제 제공한다.",
          source="references/INDEX.md · docs/PLAN_FORWARD.md")
S.footer(sl, 14, TOTAL)

# ============================================================ 15 다음 단계
sl = S.blank(prs)
y = S.header(sl, "07 · 다음 단계", "데이터 확장 · 3D 조건장 · 전이 심화 · 시각화 통합")
steps = [("01", "데이터 확장", "SoilGrids·Sentinel-1/2·TPDC 시추공 → ablation M7~M9. 활용률↑ (심사 1순위 지렛대).", NAVY),
         ("02", "얕은 3D 조건장", "GBM 조건장 + CCI prior + 물리 단조성 → PyVista 열큐브 · 0°C 등온면.", TEAL),
         ("03", "전이 심화", "kNNDM·LORO 확장 · AOA-조건 conformal로 전이 신뢰를 정량화.", PURPLE),
         ("04", "시각화 통합", "hero figure · 인터랙티브 HTML · 애니메이션 조립.", SLATE)]
cw = (CONTENT_W - 0.4) / 2
ch = 1.85
for i, (no, t, d, acc) in enumerate(steps):
    cx = ML + (i % 2) * (cw + 0.4)
    cyy = y + 0.2 + (i // 2) * (ch + 0.35)
    S.panel(sl, cx, cyy, cw, ch, fill=WHITE, line=HAIR)
    S.rect(sl, cx, cyy, 0.09, ch, color=acc)
    S.text(sl, cx + 0.28, cyy + 0.2, 1.0, 0.7,
           [{'t': no, 'size': 30, 'color': acc, 'font': F_SEMI}])
    S.text(sl, cx + 1.35, cyy + 0.24, cw - 1.6, 0.4,
           [{'t': t, 'size': 15, 'color': INK, 'font': F_SEMI}])
    S.text(sl, cx + 1.35, cyy + 0.72, cw - 1.6, 1.0,
           [{'t': d, 'size': 11.5, 'color': SLATE, 'font': F_REG}], line_spacing=1.3)
S.footer(sl, 15, TOTAL)

# ============================================================ 16 요약(마감)
sl = S.blank(prs)
S.rect(sl, 0, 0, SLIDE_W_IN, SLIDE_H_IN, color=NAVY_DK)
S.rect(sl, 0, 0, 0.28, SLIDE_H_IN, color=TEAL)
S.text(sl, ML + 0.1, 0.75, 6.0, 0.3,
       [{'t': "요약", 'size': 12, 'color': TEAL_LT, 'font': F_SEMI, 'spacing': 1.2}])
S.text(sl, ML + 0.1, 1.12, 11.0, 0.8,
       [{'t': "정직하게 진단한 정보병목을 데이터·전이·불확실성·얕은 3D로 확장한다",
         'size': 25, 'color': WHITE, 'font': F_SEMI}], line_spacing=1.05)
S.hline(sl, ML + 0.1, 2.15, 11.5, color=RGBColor(0x2A, 0x4B, 0x69), weight=1.0)
S.bullets(sl, ML + 0.1, 2.55, 11.6, 3.6,
          [("정보병목 진단", "정적 ALT는 GBM ≈ 딥러닝 — 한계는 모델이 아니라 공변량 정보이다."),
           ("정직한 평가", "무작위 CV의 과대평가를 통제하고 셀별 예측구간을 coverage로 보정한다(71→89%)."),
           ("지배 정보원 분해", "보간=기후·전이=InSAR로 나누고 ERA5로 전이를 20% 개선한다."),
           ("확장 방향", "데이터 확장 · 관측기반 얕은 3D · 전이 심화가 다음 지렛대이다.")],
          size=14.5, gap=13, lead_color=TEAL_LT,
          title_color=WHITE, body_color=RGBColor(0xB9, 0xCA, 0xD8))
S.text(sl, ML + 0.1, 6.5, 11.5, 0.4,
       [{'t': "근거: data/processed/*_results.csv  ·  색규약: design/brand_tokens.json  ·  2026-07-09",
         'size': 9.5, 'color': RGBColor(0x86, 0x9C, 0xAF), 'font': F_MED}])

# ============================================================ 저장
out = "render/permafrost_progress.pptx"
prs.save(out)
print("saved", out, "· slides:", len(prs.slides._sldIdLst))
