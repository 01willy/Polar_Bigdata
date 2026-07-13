# -*- coding: utf-8 -*-
"""build_report.py — Polar_Bigdata 진행 보고 덱 v2 (에디토리얼/학술 보고서).
문체 규율: em-dash(—) 사용 금지, 보고서/논문 문어체, 과장·수사 배제(전역 규칙 writing-tone.md).
실행: cd deck && python3 build_report.py → render/permafrost_report.pptx
"""
import os
import report_lib as R
from report_lib import (ML, MR, MT, CW, CR, SW, SH, PAPER, INK, INK2, SLATE, MUTE, TEAL, TEAL2,
                        NAVY, GOLD, RULE, RULE2, FILL, INK_DK, WHITE, SERIF, SANS, SANS_M, SANS_S, SANS_L)
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
LIGHT = RGBColor(0xB6, 0xC6, 0xCE)
FOOT = RGBColor(0x7D, 0x92, 0x9C)
DKRULE = RGBColor(0x2A, 0x4B, 0x54)

def F(p): return os.path.join("..", p)
TOTAL = 18
prs = R.new_deck()

# 수식 사전 생성
EQ = {}
EQ['skill'] = R.equation(r"\mathrm{skill} = 1 - \mathrm{RMSE}/\sigma_y", "skill", 22, "#0E5A61")
EQ['stefan'] = R.equation(r"\mathrm{ALT} \;\approx\; \sqrt{\dfrac{2\,k_t}{L\,\rho\,\theta}\;\mathrm{TDD}}", "stefan", 22, "#1B1E23")
EQ['altmagt'] = R.equation(r"\mathrm{ALT}=\max_t\{z:T(z,t)=0\},\quad \overline{T}(z)=\frac{1}{\tau}\int T(z,t)\,dt", "altmagt", 20, "#1B1E23")
EQ['var'] = R.equation(r"\sigma^2_{\mathrm{total}} = \sigma^2_{\mathrm{between}} + \sigma^2_{\mathrm{within}}", "var", 22, "#1B1E23")
EQ['cqr'] = R.equation(r"E_i=\max(\hat q_{lo}-y_i,\;y_i-\hat q_{hi});\quad C(x)=[\hat q_{lo}-Q,\;\hat q_{hi}+Q]", "cqr", 19, "#1B1E23")
EQ['di'] = R.equation(r"\mathrm{DI}(x)=\min_j \| x-x_j \|_W\,/\,\bar d", "di", 21, "#1B1E23")
EQ['skillex'] = R.equation(r"1 - 16.1/18.4 = 0.13", "skillex", 20, "#0E5A61")

def eq(s, key, x, y, w, h=0.55, align='left'):
    return R.image(s, EQ[key], x, y, w, h, align=align, valign='top')

# =================================================================== 01 표지
s = R.slide(prs, bg=PAPER)
R.text(s, ML, 1.02, CW, 0.3, [[{'t': "연구 진행 보고", 'size': 12, 'color': TEAL, 'font': SANS_S, 'spc': 1.0},
                               {'t': "   2026년 7월", 'size': 12, 'color': MUTE, 'font': SANS_M}]])
R.rule(s, ML, 1.36, 2.3, color=TEAL, wt=1.6)
R.text(s, ML, 1.62, CW*0.94, 1.7,
       [[{'t': "영구동토 활성층 두께와 얕은 3D 지중 열구조의", 'size': 32, 'color': INK, 'font': SERIF, 'bold': True}],
        [{'t': "관측기반 GeoAI 예측", 'size': 32, 'color': INK, 'font': SERIF, 'bold': True}]], ls=1.14)
R.text(s, ML, 3.7, CW*0.55, 1.2,
       [[{'t': "다중모달 빅데이터 융합, 누설을 통제한 평가, 적용범위(AOA)와 보정된 불확실성. 알래스카에서 학습한 모델을 타 영구동토 지대로 전이해 검증한다.",
          'size': 13.5, 'color': INK2, 'font': SANS}]], ls=1.45)
R.image(s, F("outputs/volumes_3d/cross_section_lat_depth.png"), 7.35, 3.5, 5.35, 2.6, align='right', valign='top')
R.text(s, 7.35, 6.18, 5.35, 0.25, [[{'t': "그림. 위도·깊이 평균 지중온도 단면. 0°C 등온선이 영구동토 경계이다.", 'size': 8, 'color': MUTE, 'font': SANS}]], align=PP_ALIGN.RIGHT)
R.rule(s, ML, 6.5, 4.8, color=RULE, wt=0.8)
R.text(s, ML, 6.64, 6.5, 0.7,
       [[{'t': "Polar_Bigdata 연구팀", 'size': 11, 'color': INK, 'font': SANS_S}],
        [{'t': "데이터: CALM · ABoVE · GTN-P · ERA5-Land · Copernicus DEM · PolSAR/InSAR · ESA CCI", 'size': 9, 'color': MUTE, 'font': SANS}]], ls=1.35, sa=2)

# =================================================================== 02 요약
s = R.slide(prs)
y = R.title_block(s, "요약", "핵심 결과", "정보 병목의 진단과 정직한 평가 체계")
R.text(s, ML, y+0.12, CW, 0.9,
       [[{'t': "정적 ALT 매핑에서 성능의 한계는 모델 용량이 아니라 공변량 정보량에 있다. ", 'size': 13.5, 'color': INK, 'font': SANS},
         {'t': "셀 단위(location-equal) 재평가로 pseudo-replication 착시를 제거하고, 적용범위(AOA)와 보정된 불확실성, 얕은 3D 열구조, 시계열 딥러닝의 사전등록 게이트를 함께 제시한다.",
          'size': 13.5, 'color': INK2, 'font': SANS}]], ls=1.5)
yy = y+1.35
R.fom(s, ML, yy, 2.75, "16.1", "cm", "LORO RMSE, +InSAR", sub="skill 12.7%, 셀 단위 평가")
R.fom(s, ML+2.95, yy, 2.75, "56 → 86", "%", "raw → CQR 커버리지", sub="목표 90%")
R.fom(s, ML+5.9, yy, 2.75, "14.7", "%", "위치 대조군 skill", sub="위경도 2피처")
R.fom(s, ML+8.85, yy, 2.6, "15.9", "cm", "정적 GBM(temporal)", sub="시계열 DL은 미달")
R.rule(s, ML, yy+1.35, CW, color=RULE)
R.text(s, ML, yy+1.52, CW, 1.2,
       [[{'t': "구성   ", 'size': 11, 'color': TEAL, 'font': SANS_S},
         {'t': "배경·동기(3,4), 선행연구·공백(5), 연구질문·기여(6), 데이터·방법(7–9), 정직한 평가(10,11), 결과: 재분석·정확도·불확실성·3D·시계열 게이트(12–16), 확충계획(17), 요약(18).",
          'size': 11, 'color': INK2, 'font': SANS}]], ls=1.4)
R.chrome(s, 2, TOTAL, "요약")

# =================================================================== 03 배경: 영구동토·ALT
s = R.slide(prs)
y = R.title_block(s, "1", "연구 배경", "활성층 두께(ALT)는 영구동토 시스템의 핵심 상태변수",
                  sub="여름철 융해가 도달하는 최대 깊이. 탄소 방출과 지반 안정, 수문에 직접 관여한다.")
R.bullets(s, ML, y+0.15, 6.1,
          [("정의", "활성층은 여름에 0°C 이상으로 융해되는 표층이다. ALT는 그 최대 융해 깊이(cm)이다."),
           ("중요성", "ALT 변화는 영구동토 탄소 노출, 인프라 침하, 수문 변화를 좌우한다."),
           ("물리", "융해는 지표 열에너지의 지중 전달로 결정된다. 해동도일(TDD)의 제곱근에 근사한다(Stefan).")],
          size=12.5, gap=11)
eq(s, 'stefan', ML, y+2.45, 5.2, 0.62)
R.text(s, ML, y+3.35, 6.1, 0.7,
       [[{'t': "여기서 k_t 열전도도, L 잠열, ρ 밀도, θ 함수량, TDD 해동도일이다. √TDD를 공변량으로 쓰는 근거가 된다. ", 'size': 9.5, 'color': MUTE, 'font': SANS},
         {'t': "예: TDD가 크면 융해심 규모도 커진다.", 'size': 9.5, 'color': SLATE, 'font': SANS}]], ls=1.3)
R.image(s, F("outputs/figures/00_concept/research_concept.png"), 6.9, y+0.1, 5.7, 3.3, valign='top')
R.caption(s, 6.9, y+3.5, 5.7, 1, "시추공 온도, 활성층, 공변량을 입력해 ALT 2D장과 얕은 3D 열구조를 복원하는 문제 설정",
          src="outputs/figures/00_concept/research_concept.png")
R.chrome(s, 3, TOTAL, "1 · 연구 배경")

# =================================================================== 04 배경: 관측 한계 + ALT vs MAGT
s = R.slide(prs)
y = R.title_block(s, "1", "연구 배경", "희소한 관측과 두 상태변수의 구분",
                  sub="지점 관측만으로는 공간장을 얻지 못한다. ALT와 연평균 지중온도(MAGT)는 서로 다른 양이다.")
R.finding(s, ML, y+0.25, 6.0, "관측 한계", "CALM 259 사이트와 GTN-P 260 사이트는 광역 영구동토를 성기게만 덮는다. 공간 예측이 필요하다.")
R.finding(s, ML, y+1.15, 6.0, "개념 구분", "ALT는 계절 최대 융해 깊이, MAGT는 연평균 지중온도장이다. 0°C 등온면과 ALT는 서로 다르다.")
eq(s, 'altmagt', ML, y+2.15, 6.1, 0.55)
R.text(s, ML, y+2.85, 6.0, 0.7,
       [[{'t': "왼쪽은 계절 최대 융해 깊이, 오른쪽은 시간평균 지중온도이다. 두 양을 혼동하면 얕은 3D 열구조 해석이 왜곡되므로 명시적으로 분리한다.", 'size': 10.5, 'color': INK2, 'font': SANS}]], ls=1.3)
R.image(s, F("outputs/figures/01_data/global_alt_coverage.png"), 6.85, y+0.15, 5.75, 3.0, valign='top')
R.caption(s, 6.85, y+3.05, 5.75, 2, "전지구 CALM 관측망(259 사이트, 12개국). 전이 검증 범위이자 관측 희소성의 근거이다.",
          src="outputs/figures/01_data/global_alt_coverage.png")
R.chrome(s, 4, TOTAL, "1 · 연구 배경")

# =================================================================== 05 선행연구·공백
s = R.slide(prs)
y = R.title_block(s, "2", "선행연구", "선행연구가 남긴 공백: 전이, 보정 불확실성, 얕은 3D",
                  sub="규모, 전이, 보정 UQ, 얕은 3D의 결합이 본 연구의 위치이다.")
rows = [["선행연구","접근","남은 공백 / 본 연구 기여"],
        ["Ran 2022 (pan-Arctic ML)","1km MAGT/ALT 앙상블 (86.9cm/1.32°C)","지역 전이·셀별 UQ·얕은 3D 부재"],
        ["ESA CCI Permafrost","CryoGrid forward 1km 제품","관측기반 보간·전이·UQ 차별, prior 한정"],
        ["Gautam 2025 (Alaska ALT)","RF vs Stefan, 내부(test R² 0.24)","전이·셀별 UQ 부재(붕괴는 open problem)"],
        ["Rahaman 2025 (Alaska DL)","토양온도 sequential DL(GRU)","ALT 아님, 공간지도·전이·UQ 부재"],
        ["Meyer 2021 · Romano 2019","AOA 적용범위, 분위 conformal","영구동토 ALT에 처음 결합, 이진을 연속 구간으로"]]
R.booktable(s, ML, y+0.15, CW, rows, col_w=[3.0, 4.0, 5.2], size=10.5, row_h=0.52, head_h=0.38)
R.text(s, ML, y+3.55, CW, 0.5,
       [[{'t': "리뷰(Koven 2025)가 명명한 open gap은 전이 일반화와 셀별 UQ이다. ", 'size': 11, 'color': INK2, 'font': SANS},
         {'t': "본 연구는 이를 LORO/kNNDM 벤치마크와 보정 UQ로 실제 제공한다.", 'size': 11, 'color': TEAL, 'font': SANS_S}]], ls=1.3)
R.tcaption(s, ML, y+4.15, CW, 1, "선행연구 대비 포지셔닝", src="references/INDEX.md · docs/PLAN_FORWARD.md")
R.chrome(s, 5, TOTAL, "2 · 선행연구")

# =================================================================== 06 연구질문·기여
s = R.slide(prs)
y = R.title_block(s, "3", "연구 설계", "연구 질문과 기여",
                  sub="데이터 구조, 공변량, 불확실성, 전이 평가를 모델 고도화보다 앞세운다.")
R.text(s, ML, y+0.1, 5.9, 0.28, [[{'t': "연구 질문", 'size': 12, 'color': TEAL, 'font': SANS_S}]])
R.bullets(s, ML, y+0.5, 5.9,
          [("Q1", "무작위 CV와 점 단위 pseudo-replication은 성능을 얼마나 왜곡하는가."),
           ("Q2", "기후·지형·SAR·CCI prior 중 실제 ALT를 설명하는 모달리티는 무엇인가."),
           ("Q3", "예측은 어느 환경에서 신뢰 가능하며(AOA), 구간은 실제 커버리지를 만족하는가."),
           ("Q4", "얕은 3D 열구조를 ALT와 분리하되 하나의 틀에서 연결할 수 있는가."),
           ("Q5", "시계열 DL은 정적 GBM과 persistence를 이기는가.")],
          size=11.5, gap=8)
R.vrule(s, 6.55, y+0.15, 4.4, color=RULE)
R.text(s, 6.8, y+0.1, 5.7, 0.28, [[{'t': "기여", 'size': 12, 'color': TEAL, 'font': SANS_S}]])
R.bullets(s, 6.8, y+0.5, 5.7,
          [("C1", "셀·위치 동등 평가 프로토콜로 매핑 skill 과대평가를 교정한다."),
           ("C2", "동일 공간검증 하에서 모달리티 기여를 정직하게 분해한다."),
           ("C3", "결정론적 지도가 아니라 셀별 보정 예측구간과 AOA 마스크를 제공한다."),
           ("C4", "borehole·CCI 기반 얕은 0–20m 열구조를 ALT와 연결한다."),
           ("C5", "DL 고도화에 명시적 게이트를 두어 근거 없는 아키텍처 과신을 방지한다.")],
          size=11.5, gap=8)
R.rule(s, ML, y+3.7, CW, color=RULE)
R.finding(s, ML, y+3.9, CW, "설계 원칙",
          "데이터 구조와 지표를 먼저 고정한다. 모델 고도화는 정적 tabular가 아니라 고차원 EO/SAR와 시간축에서 게이트를 통과할 때만 확장한다.")
R.chrome(s, 6, TOTAL, "3 · 연구 설계")

# =================================================================== 07 데이터(입력)
s = R.slide(prs)
y = R.title_block(s, "4", "입력 데이터", "22GB 규모의 다중모달 빅데이터와 확충 계획",
                  sub="알래스카에서 학습하고 전지구로 전이하는 구조이다. 데이터 확충이 진행 중이다.")
rows = [["데이터","역할","규모 / 상태"],
        ["CALM ALT","전이 검증 라벨","3,604 site-yr, 259 사이트"],
        ["ABoVE ALT","딥러닝 학습 라벨","223,937 점 → 14,348 셀"],
        ["InSAR ReSALT","약지도 사전학습","403.6만 점"],
        ["GTN-P 지중온도","3D 열구조 라벨","10,747 점, 260 사이트"],
        ["ERA5-Land","기후 공변량(월/연)","9km, 1950–현재"],
        ["Copernicus DEM","지형 공변량","30m, 234 타일"],
        ["ESA CCI ALT","prior / benchmark","1km, 1997–2021 (확보)"],
        ["SoilGrids · Sentinel","토양·식생 확충","추출 진행 중"]]
R.booktable(s, ML, y+0.15, 6.4, rows, col_w=[1.95, 2.1, 2.35], size=10, row_h=0.365, head_h=0.36,
            hi_rows={8: True})
R.image(s, F("outputs/figures/01_data/covariates_overview.png"), 7.55, y+0.15, 5.1, 2.55, valign='top')
R.caption(s, 7.55, y+2.75, 5.1, 3, "격자 공변량(기후, 고도, 계절성, 강수)이 관측 사이의 빈 공간을 채운다.",
          src="outputs/figures/01_data/covariates_overview.png")
R.finding(s, 7.55, y+3.55, 5.0, "확충 계획", "SoilGrids(토양 단열)와 CCI prior를 편입해 ablation M6·M8을 재실행하고 남은 헤드룸을 검증한다.")
R.chrome(s, 7, TOTAL, "4 · 입력 데이터")

# =================================================================== 08 공변량+약지도
s = R.slide(prs)
y = R.title_block(s, "4", "입력 데이터", "공변량 14종과 물리관측 4종, InSAR 약지도로 라벨 확장",
                  sub="지형 6종, 기후 8종, PolSAR/InSAR 4종에 더해 403.6만 점 약지도로 사전학습한다.")
groups = [("지형 · DEM (6)", ["고도","경사","사면향 sin·cos","지형위치지수 TPI","지표 거칠기"]),
          ("기후 · ERA5-Land (8)", ["연평균기온 MAAT","해동도일 TDD","동결도일 FDD","√TDD (Stefan)","최난·최한월","표층 토양온도","적설수당량 SWE"]),
          ("물리관측 · PHYS (4)", ["PolSAR ALT (P-band)","PolSAR 셀내 SD","InSAR ALT (ReSALT)","InSAR 계절 침하"])]
gw = (CW-1.0)/3
for i,(t,items) in enumerate(groups):
    gx = ML + i*(gw+0.5)
    R.text(s, gx, y+0.15, gw, 0.3, [[{'t': t, 'size': 11.5, 'color': TEAL, 'font': SANS_S}]])
    R.rule(s, gx, y+0.5, gw, color=RULE2, wt=1.0)
    R.text(s, gx, y+0.62, gw, 1.6,
           [[{'t': "· ", 'size': 11, 'color': TEAL, 'font': SANS_S}, {'t': it, 'size': 11, 'color': INK2, 'font': SANS}] for it in items],
           ls=1.35, sa=2.5)
R.text(s, ML, y+2.2, CW, 0.35,
       [[{'t': "약지도(weak label): ", 'size': 10, 'color': TEAL, 'font': SANS_S},
         {'t': "InSAR로 추정한 대규모 근사 ALT를 사전학습에 사용한다. 실측이 성긴 영역의 공간 패턴을 미리 학습시키는 용도이며, 미세조정에서 실측으로 보정한다.",
          'size': 10, 'color': INK2, 'font': SANS}]], ls=1.25)
R.image(s, F("outputs/maps/weaklabels_overview.png"), ML, y+2.65, CW, 1.85, valign='top')
R.caption(s, ML, y+2.65+1.9, CW, 1, "InSAR 약지도 403.6만 점으로 실측 대비 라벨 밀도를 크게 확대하고 30m 세밀도를 확보한다.",
          src="outputs/maps/weaklabels_overview.png", align=PP_ALIGN.CENTER)
R.chrome(s, 8, TOTAL, "4 · 입력 데이터")

# =================================================================== 09 방법: 파이프라인 + 평가원칙
s = R.slide(prs)
y = R.title_block(s, "5", "방법", "파이프라인과 누설을 통제한 평가 원칙",
                  sub="2D ALT, 얕은 3D, 횡단 UQ/AOA로 구성한다. 헤드라인 지표는 공간블록과 LORO만 사용한다.")
cols = [("관측·라벨", ["CALM·ABoVE ALT","GTN-P 지중온도","InSAR 약지도 403.6만"]),
        ("공변량 14+4", ["지형 6 · DEM","기후 8 · ERA5-Land","물리관측 4 · SAR"]),
        ("모델", ["2D: GBM 조건장 (GBM≈DL)","사전학습 후 미세조정","3D: GBM 조건장 + 깊이"]),
        ("산출", ["ALT 2D 지도(cm)","0°C 등온면, 열큐브","셀별 90% 예측구간"])]
cw2 = (CW-1.5)/4
for i,(t,items) in enumerate(cols):
    cx = ML + i*(cw2+0.5)
    R.text(s, cx, y+0.2, cw2, 0.3,
           [[{'t': f"{i+1}. ", 'size': 11.5, 'color': TEAL, 'font': SANS_S},
             {'t': t, 'size': 11.5, 'color': INK, 'font': SANS_S}]])
    R.rule(s, cx, y+0.54, cw2, color=RULE2, wt=0.9)
    R.text(s, cx, y+0.66, cw2, 1.5,
           [[{'t': "· "+it, 'size': 10, 'color': INK2, 'font': SANS}] for it in items], ls=1.3, sa=3)
    if i < 3:
        R.text(s, cx+cw2+0.08, y+0.6, 0.4, 0.4, [[{'t': "→", 'size': 15, 'color': MUTE, 'font': SANS}]])
R.rule(s, ML, y+2.35, CW, color=RULE)
R.text(s, ML, y+2.5, 6.2, 0.3, [[{'t': "평가 원칙: 누설 통제와 표준 지표", 'size': 11.5, 'color': TEAL, 'font': SANS_S}]])
R.bullets(s, ML, y+2.88, 6.3,
          [("공간블록 · LORO · kNNDM", "무작위 CV의 공간 누설을 배제한 헤드라인 검증"),
           ("표준 지표 병기", "RMSE 단독 보고 금지, R²·target_SD·skill 병기"),
           ("횡단 UQ/AOA", "Conformal(CQR) 보정과 Meyer DI 적용범위 마스크")],
          size=11, gap=7)
eq(s, 'skill', 7.25, y+2.9, 4.0, 0.5)
R.text(s, 7.25, y+3.42, 5.2, 0.4, [[{'t': "평균예측기 대비 RMSE 개선율. σ_y는 test fold 라벨 SD이다.", 'size': 9.5, 'color': MUTE, 'font': SANS}]], ls=1.2)
R.text(s, 7.25, y+3.78, 5.2, 0.3, [[{'t': "계산 예: ", 'size': 9.5, 'color': TEAL, 'font': SANS_S}]])
eq(s, 'skillex', 8.05, y+3.74, 3.2, 0.42)
R.text(s, 7.25, y+4.28, 5.2, 0.4, [[{'t': "RMSE 16.1cm, σ_y 18.4cm일 때 skill은 0.13이다.", 'size': 9.5, 'color': SLATE, 'font': SANS}]])
R.chrome(s, 9, TOTAL, "5 · 방법")

# =================================================================== 10 정직한 평가 ① 누설
s = R.slide(prs)
y = R.title_block(s, "6", "정직한 평가", "무작위 CV는 전이 성능을 과대평가한다",
                  sub="공간 자기상관에 의한 누설이 원인이다. 헤드라인은 공간블록과 LORO만 신뢰한다.")
R.image(s, "assets/cv_leakage_cool.png", ML, y+0.2, 6.9, 3.9, valign='top')
R.caption(s, ML, y+4.15, 6.9, 1, "CV 설계별 baseline ALT RMSE. 무작위 K-fold의 낙관과 LORO에서의 붕괴를 대비한다.",
          src="data/processed/cv_leakage_table.csv")
R.finding(s, 7.7, y+0.35, 5.0, "누설", "무작위 K-fold에서 IDW의 RMSE는 28cm이나, 지역분리 LORO에서 111cm로 붕괴한다.")
R.finding(s, 7.7, y+1.45, 5.0, "설계", "헤드라인은 공간블록과 LORO만 사용하고, 셀 단위 집계로 pseudo-replication을 통제한다.")
R.finding(s, 7.7, y+2.55, 5.0, "예시", "이웃한 점을 학습과 시험에 나눠 담으면 사실상 같은 위치를 외운다. 이것이 과대평가의 원인이다.")
R.fom(s, 7.7, y+3.5, 2.4, "약 4배", "", "무작위 CV 과대평가", sub="IDW 기준, 28 대 111cm")
R.fom(s, 10.3, y+3.5, 2.4, "14,348", "셀", "위치-동등 단위", sub="225k 점을 집계")
R.chrome(s, 10, TOTAL, "6 · 정직한 평가")

# =================================================================== 11 정직한 평가 ② apparent floor
s = R.slide(prs)
y = R.title_block(s, "6", "정직한 평가", "17cm은 물리 한계가 아니라 정보 병목이다",
                  sub="분산분해에서 위치간 성분이 지배하며, 비가역 하한은 훨씬 낮다.")
R.fom(s, ML, y+0.2, 2.7, "86 / 14", "%", "분산: 위치간 / 위치내", sub="pseudo-replication 진단")
R.fom(s, ML+2.9, y+0.2, 2.7, "7.2", "cm", "비가역 하한", sub="현재 최고 17.0cm보다 낮음")
R.fom(s, ML+5.8, y+0.2, 2.7, "0.13", "", "셀 단위 skill", sub="점 단위 평가보다 정직")
eq(s, 'var', 8.9, y+0.35, 3.6, 0.5)
R.bullets(s, ML, y+1.55, 11.8,
          [("폐기", "‘17cm은 물리 하한’이라는 표현은 폐기한다. 이는 현재 공변량에서 나타나는 정보 병목이다."),
           ("진단", "같은 공변량 셀에 서로 다른 ALT가 공존한다(pseudo-replication). 라벨 30m와 기후 9km의 척도도 불일치한다."),
           ("예시", "동일 공변량 셀 안에서 관측 ALT가 34cm와 96cm로 공존하기도 한다. 같은 입력에 다른 정답이 대응한다."),
           ("함의", "비가역 하한(약 7cm)이 현재 최고(17.0cm)보다 훨씬 낮다. 새 모달리티와 척도 정합에 헤드룸이 남아 있다.")],
          size=12.5, gap=13)
R.tcaption(s, ML, y+4.35, CW, 1, "apparent-floor 진단(분산분해)", src="data/processed/apparent_floor_diagnosis.csv")
R.chrome(s, 11, TOTAL, "6 · 정직한 평가")

# =================================================================== 12 결과 ① 셀 재분석 ablation
s = R.slide(prs)
y = R.title_block(s, "7", "결과", "셀 단위 재분석: 지배 정보원과 위치 대조군",
                  sub="정직한 셀 평가에서 skill이 낮아지고, 위경도만으로도 물리 공변량 이상을 설명한다.")
R.image(s, "assets/figs/cell_ablation.png", ML, y+0.35, 7.35, 3.5, valign='top', align='left')
R.caption(s, ML, y+3.95, 7.35, 1, "셀 단위 다중모달 ablation(GBM 고정). 공간블록과 LORO, 막대 위는 RMSE, 막대 안은 skill.",
          src="data/processed/alt_ablation_cell_results.csv")
R.finding(s, 8.45, y+0.5, 4.15, "보간(공간블록)", "기후(ERA5)가 지배한다. 지형을 더하면 공간 과적합으로 악화된다.")
R.finding(s, 8.45, y+1.55, 4.15, "전이(LORO)", "+InSAR가 물리 조합 중 최고(16.1cm)이다. 지형 단독은 파탄한다.")
R.finding(s, 8.45, y+2.6, 4.15, "위치 대조군", "위경도만으로 skill 14.7%에 이른다. 정보 병목의 직접 증거이다.", accent=GOLD)
R.finding(s, 8.45, y+3.65, 4.15, "CCI prior", "M8은 개선이 없다. 기후와 정보가 중복되기 때문이다.")
R.chrome(s, 12, TOTAL, "7 · 결과")

# =================================================================== 13 결과 ② 정확도 + 지도
s = R.slide(prs)
y = R.title_block(s, "7", "결과", "예측 정확도와 정보 병목의 흔적",
                  sub="예측이 중앙으로 수축하는 평균회귀가 나타난다. 모델 용량이 아니라 정보의 한계이다.")
R.image(s, "assets/figs/pred_obs_cell.png", ML, y+0.15, 4.6, 4.05, valign='top')
R.caption(s, ML, y+4.25, 4.6, 2, "예측 대 관측(셀 OOF). 중앙으로의 수축이 정보 병목을 나타낸다.",
          src="data/processed/alt_cell_best_oof.csv")
R.image(s, F("outputs/maps/alt_alaska_pred.png"), 5.5, y+0.15, 4.4, 4.05, valign='top')
R.caption(s, 5.5, y+4.25, 4.4, 3, "알래스카 ALT 예측면(관측 오버레이). 순차형 냉색.",
          src="outputs/maps/alt_alaska_pred.png")
R.finding(s, 10.15, y+0.4, 2.5, "수축", "높은 ALT는 과소, 낮은 ALT는 과대 예측된다.")
R.finding(s, 10.15, y+1.75, 2.5, "R² 0.11", "셀 평가에서의 정직한 설명력이다.")
R.finding(s, 10.15, y+3.1, 2.5, "색 규약", "붉은 계열과 rainbow를 배제한다.")
R.chrome(s, 13, TOTAL, "7 · 결과")

# =================================================================== 14 결과 ③ 보정 UQ + AOA
s = R.slide(prs)
y = R.title_block(s, "7", "결과", "보정된 불확실성과 적용범위(AOA)",
                  sub="raw 90% 구간은 실제로 56%만 포착한다. CQR로 86%까지 보정하고, 외삽 영역은 경고한다.")
R.image(s, "assets/figs/conformal_aoa.png", ML, y+0.2, 7.25, 3.4, valign='top', align='left')
R.caption(s, ML, y+3.7, 7.25, 1, "(a) conformal 커버리지 보정. (b) DI 구간별 RMSE와 커버리지.",
          src="data/processed/alt_conformal_cell_results.csv · alt_aoa_cell_transfer.csv")
eq(s, 'cqr', 8.3, y+0.35, 4.4, 0.68)
R.text(s, 8.3, y+1.05, 4.4, 0.5, [[{'t': "분위 잔차로 구간을 보정한다. 분포 가정 없이 유효 커버리지를 확보한다.", 'size': 9.5, 'color': MUTE, 'font': SANS}]], ls=1.25)
eq(s, 'di', 8.3, y+1.8, 4.0, 0.55)
R.text(s, 8.3, y+2.35, 4.4, 0.5, [[{'t': "학습 환경공간과의 가중 거리이다. 임계 초과 영역은 외삽으로 판정한다.", 'size': 9.5, 'color': MUTE, 'font': SANS}]], ls=1.25)
R.finding(s, 8.3, y+3.05, 4.3, "정직한 표기", "최고 DI 구간의 RMSE는 30cm, 커버리지는 50%이다(D3에서 88%로 비단조).")
R.chrome(s, 14, TOTAL, "7 · 결과")

# =================================================================== 15 결과 ④ 얕은 3D
s = R.slide(prs)
y = R.title_block(s, "7", "결과", "얕은 3D 지중 열구조",
                  sub="위도를 따라 영구동토 경계 심도가 변한다. 예측 엔진은 GBM 조건장이다(신경장은 게이트 탈락).")
R.image(s, F("outputs/volumes_3d/cross_section_lat_depth.png"), ML, y+0.2, 5.9, 3.5, valign='top')
R.caption(s, ML, y+3.78, 5.9, 1, "위도·깊이 MAGT 단면. 0°C 등온선이 영구동토 경계이다.",
          src="outputs/volumes_3d/cross_section_lat_depth.png", align=PP_ALIGN.CENTER)
R.image(s, F("outputs/maps/magt_alaska_2m_20m.png"), 6.95, y+0.2, 2.95, 3.5, valign='top', align='left')
R.caption(s, 6.95, y+3.78, 2.95, 2, "MAGT 2m·20m, 0°C 등고선",
          src="magt_alaska_2m_20m.png", align=PP_ALIGN.CENTER)
R.finding(s, 10.2, y+0.4, 2.5, "경계 심도", "0°C 등온면이 남부에서 북부로 갈수록 깊어진다.")
R.finding(s, 10.2, y+1.75, 2.5, "구분", "ALT(계절 최대융해)와 MAGT(연평균장)는 다르다.")
R.finding(s, 10.2, y+3.1, 2.5, "엔진", "GBM 조건장에 깊이 인코딩을 결합한다.")
R.chrome(s, 15, TOTAL, "7 · 결과")

# =================================================================== 16 결과 ⑤ T-lite 게이트
s = R.slide(prs)
y = R.title_block(s, "7", "결과", "시계열 딥러닝 게이트: temporal holdout에서 미통과",
                  sub="사전등록 게이트이다. GRU는 site-disjoint에서 baseline을 소폭 앞서나 temporal holdout에서 GBM을 넘지 못한다.")
R.image(s, "assets/figs/tlite_gate.png", ML, y+0.2, 7.0, 3.8, valign='top', align='left')
R.caption(s, ML, y+4.1, 7.0, 1, "CALM site-year(251 사이트) ALT 예측. site-disjoint와 temporal holdout 검증.",
          src="data/processed/tlite_sequence_gate_results.csv")
R.finding(s, 8.1, y+0.35, 4.5, "게이트 규칙", "두 검증을 모두 통과해야 채택한다. 아키텍처 과신을 방지한다.")
R.booktable(s, 8.1, y+1.35, 4.5,
            [["모델","site-disj.","temporal"],["GBM-annual","17.3","15.9"],["persistence","17.0","17.0"],
             ["GRU","16.8","19.2"],["TCN","18.1","23.8"]],
            col_w=[1.9, 1.3, 1.3], size=10, row_h=0.32, head_h=0.32, hi_rows={1: True})
R.text(s, 8.1, y+3.55, 4.5, 0.9, [[{'t': "GRU는 site-disjoint에서 최우수이나 temporal에서 붕괴하여 게이트를 통과하지 못한다. 정적 tabular ALT는 GBM으로 충분하며, DL은 고차원 EO/SAR 입력에만 제한한다.",
                                   'size': 9.5, 'color': INK2, 'font': SANS}]], ls=1.3)
R.chrome(s, 16, TOTAL, "7 · 결과")

# =================================================================== 17 확충계획 + 로드맵
s = R.slide(prs)
y = R.title_block(s, "8", "다음 단계", "데이터 확충과 딥러닝 고도화 게이트",
                  sub="데이터, 평가, 불확실성이 지렛대이다. DL은 고차원 입력과 시간축에서 게이트를 통과할 때만 확장한다.")
R.text(s, ML, y+0.1, 5.9, 0.3, [[{'t': "데이터 확충", 'size': 11.5, 'color': TEAL, 'font': SANS_S}]])
R.bullets(s, ML, y+0.5, 5.9,
          [("토양", "SoilGrids(SOC·용적밀도·점토/모래)로 단열·잠열 결측을 보완하고 ablation M6에 반영한다."),
           ("위성", "Sentinel-1/2 시계열과 PolSAR를 확대한다. 고차원 encoder는 게이트 통과 시에만 채택한다."),
           ("prior", "ESA CCI ALT/GT 0–10m를 benchmark·prior·의사라벨 사전학습에 활용한다."),
           ("시계열", "PANGAEA CALM 1990–2024로 T-lite를 재검증한다(월별 forcing).")],
          size=11, gap=8)
R.vrule(s, 6.55, y+0.15, 4.0, color=RULE)
R.text(s, 6.8, y+0.1, 5.7, 0.3, [[{'t': "DL 고도화 게이트", 'size': 11.5, 'color': TEAL, 'font': SANS_S}]])
R.bullets(s, 6.8, y+0.5, 5.7,
          [("정적 tabular", "GBM으로 충분하므로 새 Transformer/MLP 추가는 중단한다."),
           ("고차원 EO/SAR", "CNN/ViT encoder는 handcrafted feature를 이길 때만 채택한다."),
           ("시계열", "GRU/TCN은 persistence·GBM을 초과할 때만 채택한다(현재 미통과)."),
           ("full 4D", "DeepONet/FNO는 후속 논문 후보로, 본 보고 범위 밖이다.")],
          size=11, gap=8)
R.finding(s, 6.8, y+3.55, 5.6, "원칙", "모델을 복잡하게 만들기 전에 데이터 구조와 지표를 먼저 고정한다.")
R.chrome(s, 17, TOTAL, "8 · 다음 단계")

# =================================================================== 18 요약(마감)
s = R.slide(prs, bg=INK_DK)
R.rect(s, 0, 0, 0.14, SH, fill=DKRULE)
R.text(s, ML, 0.92, CW, 0.3, [[{'t': "요약", 'size': 12, 'color': TEAL2, 'font': SANS_S, 'spc': 1.2}]])
R.text(s, ML, 1.32, CW, 0.9, [[{'t': "정직하게 진단한 정보 병목을 데이터, 전이, 불확실성, 얕은 3D로 확장한다",
                               'size': 23, 'color': WHITE, 'font': SERIF, 'bold': True}]], ls=1.12)
R.rule(s, ML, 2.35, CW, color=DKRULE, wt=0.9)
items = [("정보 병목 진단", "정적 ALT는 GBM과 DL이 대등하다. 한계는 모델이 아니라 공변량 정보이며, 셀 재평가가 이를 확증한다."),
         ("정직한 평가", "무작위 CV의 과대평가를 통제하고, 셀별 예측구간을 conformal로 보정한다(56%에서 86%로)."),
         ("지배 정보원", "보간은 기후, 전이는 InSAR가 지배한다. 위치 대조군(위경도 skill 14.7%)이 정보 병목을 드러낸다."),
         ("게이트 규율", "시계열 DL은 정적 GBM을 이기지 못해 부록으로 내린다. 근거 없는 확장을 배제한다."),
         ("다음 단계", "SoilGrids와 CCI 확충으로 헤드룸을 검증하고, 관측기반 얕은 3D와 전이를 심화한다.")]
yb = 2.62
for i,(h,b) in enumerate(items):
    R.text(s, ML, yb, CW, 0.55,
           [[{'t': h, 'size': 12.5, 'color': TEAL2, 'font': SANS_S}],
            [{'t': b, 'size': 11.5, 'color': LIGHT, 'font': SANS}]], ls=1.2, sa=2)
    yb += 0.82
R.text(s, ML, SH-0.55, CW, 0.3,
       [[{'t': "근거: data/processed/*_results.csv · 그림: deck/assets · 2026-07-10", 'size': 9, 'color': FOOT, 'font': SANS_M}]])

out = "render/permafrost_report.pptx"
prs.save(out)
print("saved", out, "· slides:", len(prs.slides._sldIdLst))
