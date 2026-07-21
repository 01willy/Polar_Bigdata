# -*- coding: utf-8 -*-
"""중간보고 PPT (v3) · 보고서톤·그림중심·논리흐름.
설계 원칙(사용자 피드백 반영):
  - 명사형 제목(kicker=명사구, 부제=서술구). 문장형 제목 금지.
  - 배경·목적부터 시작(요약 선두 금지). 처음 보는 사람이 목적→진행을 이해.
  - 그림 중심(지도·3D 우선), 그래프 최소화, 상세 캡션. 균일 박스(AI틱) 지양.
report_lib.py 재사용. 출력 deck/render/permafrost_midreport.pptx
빌드: (deck/) python3 build_midreport.py · 렌더: soffice --headless --convert-to pdf
"""
import os
import math
import report_lib as R
from report_lib import (
    ML, MR, MT, MB, CW, CR, SW, SH,
    PAPER, INK, INK2, SLATE, MUTE, TEAL, TEAL2, NAVY, GOLD, RULE, RULE2, FILL,
    INK_DK, WHITE, SERIF, SANS, SANS_M, SANS_S, SANS_L,
)
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

LIGHT = RGBColor(0xD5, 0xD9, 0xDD)
TEAL_L = RGBColor(0x6F, 0xA9, 0xAD)
DKRULE = RGBColor(0x2A, 0x44, 0x4E)

def F(p):
    return os.path.join("..", p)

prs = R.new_deck()
TOTAL = 21

# 전역 슬라이드 카운터(표지=1). 순서대로 생성되므로 페이지 번호가 정확해진다.
_SN = [1]
def CH(s, section):
    _SN[0] += 1
    R.chrome(s, _SN[0], TOTAL, section)

# 전역 그림 번호. 생성 순서 = 지면 순서이므로 덱 전체에서 유일·연속.
_FIG = [0]
def FIG():
    _FIG[0] += 1
    return _FIG[0]

# 콘텐츠 하한(푸터 규칙선 SH-0.44 위로 0.12 여유)
BLIM = SH - 0.56

def est_h(txt, w, size, ls=1.3):
    """한글 위주 문단의 렌더 높이(inch) 추정. 전각폭 = size(pt)의 0.95배 가정."""
    cpl = max(int(w * 72.0 / (size * 0.95)), 8)
    lines = max(1, math.ceil(len(txt) / cpl))
    return lines * (size / 72.0) * ls + 0.06

# 지표 스트립: (값, 라벨) 카드 여러 개를 가로로
def metric_strip(s, x, y, w, metrics, vsize=17):
    n = len(metrics)
    cw = w / n
    for i, (val, lab) in enumerate(metrics):
        xx = x + i*cw
        if i > 0:
            R.vrule(s, xx, y+0.05, 0.60, color=RULE, wt=0.9)
        R.text(s, xx+0.1, y, cw-0.15, 0.40,
               [[{'t': val, 'size': vsize, 'color': TEAL, 'font': SERIF, 'bold': True}]], anchor=MSO_ANCHOR.MIDDLE)
        R.text(s, xx+0.1, y+0.42, cw-0.15, 0.28,
               [[{'t': lab, 'size': 9, 'color': SLATE, 'font': SANS_M}]], ls=1.05)

# figure-hero: 그림 + 캡션 + 지표 스트립 + 설명 + 해석. 푸터 침범 없이 배치.
def figure_hero(kicker, no, title, sub, img, cap, explain, metrics, notes, cap_src=None):
    s = R.slide(prs); CH(s, kicker)
    y = R.title_block(s, no, kicker, title, sub)
    path = img if os.path.exists(img) else F(img)
    ar = R._aspect(path)
    fig_no = FIG()
    if ar > 2.1:
        # 넓은 그림: 상단 그림 + 캡션 + 한 행(좌 지표·설명 / 우 해석)
        ih = min(SH - y - 2.62, CW / ar)
        dx, dy, dw, dh = R.image(s, img, ML, y+0.08, CW, ih, valign='top', align='left')
        cy = dy + dh + 0.08
        R.caption(s, ML, cy, CW, fig_no, cap, src=cap_src)
        yb = cy + (0.52 if cap_src else 0.40)
        ew = CW*0.60
        if metrics:
            metric_strip(s, ML, yb, ew, metrics, vsize=13.5)
            ey = yb + 0.80
        else:
            ey = yb
        if explain:
            eh = est_h(explain, ew, 10.5, ls=1.28)
            R.text(s, ML, ey, ew, eh, [[{'t': explain, 'size': 10.5, 'color': INK2, 'font': SANS}]], ls=1.28)
        if notes:
            nx2 = ML + ew + 0.45
            R.vrule(s, nx2-0.22, yb+0.02, min(BLIM-yb-0.06, 1.3), color=RULE, wt=0.9)
            R.text(s, nx2, yb, CR-nx2, 0.3, [[{'t': "해석", 'size': 10.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.5}]])
            R.bullets(s, nx2, yb+0.32, CR-nx2, notes, size=10.5, gap=6)
    else:
        # 좁은 그림: 좌 그림 + 캡션, 우 지표·설명(위) + 해석(하단 고정)
        ah = SH - y - 1.2
        fw = min(ah * ar, 7.05)
        dx, dy, dw, dh = R.image(s, img, ML, y+0.1, fw, ah, valign='top', align='left')
        R.caption(s, ML, dy+dh+0.08, fw, fig_no, cap, src=cap_src)
        nx = ML + fw + 0.5
        nw = CR - nx
        yy = y+0.18
        if metrics:
            metric_strip(s, nx, yy, nw, metrics)
            yy += 0.95
        if explain:
            eh = est_h(explain, nw, 11, ls=1.32)
            R.text(s, nx, yy, nw, eh, [[{'t': explain, 'size': 11, 'color': INK2, 'font': SANS}]], ls=1.32)
            yy += eh + 0.1
        if notes:
            nh = 0.46 + len(notes)*0.30
            ny = max(BLIM - nh, yy + 0.05)
            R.rule(s, nx, ny, nw, color=RULE, wt=0.8)
            R.text(s, nx, ny+0.12, nw, 0.3, [[{'t': "해석", 'size': 10.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.5}]])
            R.bullets(s, nx, ny+0.44, nw, notes, size=10.5, gap=6)
    return s, y

# ================================================================= 1 표지 (흰 배경, EMP 톤)
s = R.slide(prs, bg=WHITE)
R.text(s, ML, 1.55, CW, 0.3,
       [[{'t': "연구 중간보고 · 2026. 07", 'size': 12.5, 'color': SLATE, 'font': SANS_S, 'spc': 1.0}]])
R.text(s, ML, 2.05, CW*0.98, 1.5,
       [[{'t': "북극 영구동토 ", 'size': 33, 'color': INK, 'font': SANS_L},
         {'t': "활동층 두께", 'size': 33, 'color': INK, 'font': SERIF, 'bold': True},
         {'t': "와 얕은 ", 'size': 33, 'color': INK, 'font': SANS_L},
         {'t': "3D 열구조", 'size': 33, 'color': INK, 'font': SERIF, 'bold': True},
         {'t': "의", 'size': 33, 'color': INK, 'font': SANS_L}],
        [{'t': "관측기반 GeoAI 예측", 'size': 33, 'color': INK, 'font': SANS_L}]], ls=1.28)
R.text(s, ML, 3.75, CW*0.92, 0.4,
       [[{'t': "다중모달 융합 · 누설 통제 평가 · 보정 불확실성 · 전이 검증 · 얕은 3D 열구조",
          'size': 13.5, 'color': TEAL, 'font': SANS_S}]])
R.rule(s, ML, 5.1, 2.2, color=TEAL, wt=2.4)
R.text(s, ML, 5.35, CW, 0.6,
       [[{'t': "Polar_Bigdata 연구팀", 'size': 13, 'color': INK, 'font': SANS_S}],
        [{'t': "2026 극지 빅데이터 · 인공지능 활용 경진대회 준비", 'size': 11, 'color': SLATE, 'font': SANS}]], ls=1.3, sa=2)
R.text(s, CR-4.6, SH-0.6, 4.6, 0.3,
       [[{'t': "ALT 2D · MAGT 3D · AOA · conformal UQ · LORO transfer", 'size': 9.5, 'color': TEAL, 'font': SANS_M}]],
       align=PP_ALIGN.RIGHT)

# ================================================================= 2 연구 배경
s = R.slide(prs); CH(s, "연구 배경")
y = R.title_block(s, "01", "연구 배경", "활동층 두께의 역할과 관측의 한계")
R.bullets(s, ML, y+0.25, 6.0, [
    ("현안.", "영구동토 융해는 기후 되먹임(탄소 방출), 지반 침하로 인한 인프라 위험과 직결된다"),
    ("활동층.", "여름에 녹는 표층(활동층)의 두께(ALT)가 융해 정도를 대표하는 지표다"),
    ("한계.", "현장 관측은 희소하고 위성 관측은 간접적이라, 넓은 지역을 신뢰 있게 지도화하기 어렵다"),
], size=12.5, gap=11)
# 배경 서술(불릿)과 연구 질문의 위계 분리: 구분선 + 라벨 블록
R.rule(s, ML, y+2.18, 6.0, color=RULE, wt=0.9)
R.finding(s, ML, y+2.36, 6.0, "핵심 질문",
          "관측이 희소한 극지에서 활동층 두께를 어디까지, 얼마나 정직하게 지도화할 수 있는가")
_d = R.image(s, "assets/mid/concept_alt.png", 6.85, y+0.2, 5.75, 3.9, valign='top')
R.caption(s, 6.85, _d[1]+_d[3]+0.08, 5.75, FIG(),
          "여름 융해로 형성되는 활동층(ALT)과 그 아래 영구동토의 개념. 지중온도 포락선(연중 최고·최저)과 0°C 경계선을 함께 표시")

# ================================================================= 3 연구 목적
s = R.slide(prs); CH(s, "연구 목적")
y = R.title_block(s, "02", "연구 목적", "다중모달 입력에서 산출물까지의 구조")
_d = R.image(s, "assets/mid/architecture.png", ML, y+0.2, CW, SH-y-0.95, valign='top')
R.caption(s, ML, SH-0.72, CW, FIG(),
          "연구 구조. 기후(ERA5-Land)·지형(ArcticDEM)·SAR·시추공 관측을 융합해 활동층 두께 2D 지도, "
          "얕은 3D 열구조, 보정 불확실성, 전이 검증의 4종 산출물을 만든다")

# ================================================================= 4 선행연구
s = R.slide(prs); CH(s, "선행연구")
y = R.title_block(s, "03", "선행연구와 공백", "전이, 보정 불확실성, 얕은 3D가 남은 과제")
R.tcaption(s, ML, y+0.18, CW, 1, "선행연구 계열별 접근과 남은 공백. 각 공백이 본 연구의 설계 요소로 이어진다")
rows = [
    ["선행연구", "접근", "남은 공백 · 본 연구 기여"],
    ["GIPL2 · CCI", "물리 순방향 모델로 지중온도 산출", "관측기반 보간·전이·불확실성 미흡 → 관측기반 GeoAI"],
    ["ALT ML 매핑", "무작위 CV로 높은 정확도 보고", "공간 누설로 과대평가 → 누설 통제 평가(공간블록·LORO)"],
    ["InSAR 침하", "침하로 ALT 간접 역산", "지역 의존·부호 반전 → 다중모달 융합·조건부 처리"],
    ["대부분 2D", "활동층 두께 지도에 한정", "지중 열구조 결합 부재 → 얕은 3D + 0°C 등온면"],
]
_ty = y+0.55
R.booktable(s, ML, _ty, CW, rows, [2.4, 3.6, CW-6.0], size=11, row_h=0.70, head_h=0.44,
            align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
R.finding(s, ML, _ty+0.44+4*0.70+0.32, CW, "요지",
          "정확도 경쟁이 아니라 정직한 평가, 보정된 불확실성, 2D에서 3D로의 확장이 차별점이다. "
          "네 공백은 이후 결과 슬라이드(06·07)에서 각각 검증한다.")

# ================================================================= 5 데이터
s = R.slide(prs); CH(s, "입력 데이터")
y = R.title_block(s, "04", "입력 데이터", "다중모달 빅데이터와 KPDC 현장관측의 결합")
_d = R.image(s, "assets/mid/alt_coverage.png", ML, y+0.2, 7.0, 3.6, valign='top', align='left')
R.caption(s, ML, _d[1]+_d[3]+0.08, 7.0, FIG(),
          "전지구 활동층 관측 분포(좌: 위치, 우: 지역별 표본 수). 알래스카에서 학습하고 타 지대로의 전이를 검증한다",
          src="자료: CALM·GTN-P·ABoVE 통합 전처리 산출물")
_my = _d[1]+_d[3]+0.72
R.rule(s, ML, _my, 7.0, color=RULE, wt=0.8)
metric_strip(s, ML, _my+0.14, 7.0, [
    ("약 22만", "활동층 관측(점)"),
    ("9 km", "기후 공변량 해상도"),
    ("30 m", "InSAR 국소 해상도"),
], vsize=14)
nx = ML + 7.0 + 0.45
R.text(s, nx, y+0.2, CR-nx, 0.3, [[{'t': "데이터 구성", 'size': 11, 'color': TEAL, 'font': SANS_S, 'spc': 0.5}]])
R.bullets(s, nx, y+0.6, CR-nx, [
    ("학습.", "CALM·GTN-P·ABoVE 활동층 관측 약 22만, ERA5-Land 기후, ArcticDEM 지형, InSAR/PolSAR"),
    ("검증 축.", "KPDC Council 지중온도·AWS로 예측과 공변량을 현장 실측으로 독립 검증"),
    ("전이 축.", "KPDC Cambridge Bay, 시베리아(레나델타), 티베트로 지역 다양성 확대"),
    ("확충.", "화재 교란·다층 토양수분·적설 등 미사용 오픈데이터를 게이트 검증 후 편입"),
], size=11, gap=9)

# ================================================================= 5b 데이터 인벤토리·지역 확장 (P0)
s = R.slide(prs); CH(s, "입력 데이터")
y = R.title_block(s, "04", "데이터 인벤토리와 지역 확장",
                  "보유 자산의 공간 분포와 지역별 가용성, 시베리아·티베트 편입")
# 상단 좌: 지도 패널(크롭)  ·  상단 우: 확충 불릿  ·  하단: 가용성 매트릭스(전폭 네이티브 표)
mw = 6.7
_d = R.image(s, "assets/mid/inventory_map.png", ML, y+0.1, mw, 2.24, valign='top', align='left')
R.caption(s, ML, _d[1]+_d[3]+0.04, mw, FIG(),
          "관측·공변량 자산의 공간 분포. 삼각형=지중온도 시추공, 점=활동층 셀, 점선상자=SAR 커버리지")
nx = ML + mw + 0.42
R.text(s, nx, y+0.12, CR-nx, 0.3, [[{'t': "이번 확충", 'size': 11, 'color': TEAL, 'font': SANS_S, 'spc': 0.5}]])
R.bullets(s, nx, y+0.48, CR-nx, [
    ("레나델타.", "ALLena 융해깊이에서 시베리아 ALT 3,037셀 신규 유도(러시아 첫 대규모 전이 시험지)"),
    ("티베트·시추공.", "TPDC QTEC과 GTN-P 지중온도의 0°C 통과 깊이로 ALT 라벨을 다지역 확장"),
    ("결측 축.", "InSAR/PolSAR는 알래스카만 존재, 신규 지역은 결측으로 두고 지시자로 표기"),
    ("확대.", "학습 라벨 14,348→17,423셀. 알래스카 편중을 넓혀 다지역 전이(E1) 실측 기반 확보"),
], size=11, gap=7)
# 하단: 지역×데이터 가용성 매트릭스(네이티브 표 → 임의 축소에도 판독 보장)
_ty = _d[1]+_d[3]+0.42
R.tcaption(s, ML, _ty, CW, 2,
           "지역별 데이터 가용성. O=보유, △=부분, ○=전지구 취득 가능, X=없음 "
           "(자료: dl_dataset_cell_v2 · ground_temp_all · ALLena · TPDC QTEC)")
mrows = [
    ["지역", "ALT 라벨", "지온 3D", "기후·CCI", "지형(DEM)", "InSAR·PolSAR"],
    ["알래스카", "O 13,606", "O 40", "O", "O", "O"],
    ["서캐나다", "O 742", "△", "O", "O", "△ InSAR만"],
    ["레나델타(신규)", "O 3,037", "△", "O", "O", "X"],
    ["티베트 QTP(신규)", "△ 유도", "O QTEC", "O", "O", "X"],
    ["러시아·스발바르", "△ 유도", "O 다수", "O", "○", "X"],
]
R.booktable(s, ML, _ty+0.30, CW, mrows, [2.5, 1.75, 1.4, 1.35, 1.6, CW-8.6],
            size=10.5, row_h=0.285, head_h=0.32,
            align=[PP_ALIGN.LEFT]+[PP_ALIGN.CENTER]*5)

# ================================================================= 6 방법
s = R.slide(prs); CH(s, "방법")
y = R.title_block(s, "05", "방법", "파이프라인과 누설을 통제한 평가 원칙")
R.bullets(s, ML, y+0.16, CW, [
    ("데이터 재구조화.", "같은 위치의 반복 관측을 셀 단위로 집계하고 1/n 가중으로 밀집 편향을 제거한다"),
    ("모델.", "GBM 조건장을 주력으로 6개 모델(GBM·MLP·Transformer·TabM·Flow·Diffusion)을 동일 조건에서 비교한다"),
    ("평가.", "무작위 CV 대신 공간블록·LORO로 미관측 지역 성능을 재고, conformal 예측구간과 적용범위(AOA)를 함께 낸다"),
    ("정직성.", "성공 조건을 사전 등록하고, 통과하지 못한 실험(음성 결과)도 그대로 보고한다"),
], size=11.5, gap=8)
_uy = y+1.52
R.rule(s, ML, _uy, CW, color=RULE, wt=0.8)
# 그림 → 캡션 → 용어 순으로 붙여 배치(캡션과 용어 블록의 공간 분리 해소)
_d = R.image(s, "assets/mid/cv_concept.png", ML, _uy+0.12, CW, BLIM-_uy-1.10, valign='top', align='center')
_cy = _d[1]+_d[3]+0.07
R.caption(s, _d[0], _cy, CR-_d[0], FIG(),
          "누설 통제 평가의 개념. 무작위 CV는 이웃 관측이 학습·시험에 섞여 낙관하고, 공간블록·LORO는 지역째 분리해 실제 전이 성능을 잰다")
R.text(s, ML, _cy+0.42, CW, 0.4,
       [[{'t': "용어  ", 'size': 10, 'color': TEAL, 'font': SANS_S},
         {'t': "공간블록 CV = 500km 블록 단위로 학습·시험을 분리  ·  LORO(leave-one-region-out, 지역 제외 평가) = "
               "한 지역을 통째로 제외하고 학습해 그 지역에서 평가  ·  skill = 평균 예측 대비 RMSE 개선율",
          'size': 10, 'color': SLATE, 'font': SANS_M}]], ls=1.25)

# ================================================================= 7 결과: ALT 지도 (hero)
figure_hero("결과", "06", "활동층 두께 예측 지도",
            "알래스카 전역 활동층 두께와 관측 오버레이",
            "assets/mid/alt_map.png",
            "ERA5-Land 공변량 기반 GBM으로 복원한 알래스카 활동층 두께(cm). 흰 점은 CALM, 삼각형은 ABoVE 현장 관측.",
            "활동층 두께(ALT)는 여름에 녹는 표층의 깊이다. 짙은 청색일수록 두껍게(많이 녹는) 예측된 것으로, "
            "북부 연속동토대는 얇고(20~40cm) 남부로 갈수록 두꺼워진다. 관측이 없는 지역도 공변량으로 연속 예측면을 만든다. "
            "정확도는 처음 보는 지역 기준(공간블록·LORO) RMSE 16.95cm로, 관측 표준편차 18.5cm와 견줄 대표성 있는 수준이다.",
            [("16.95 cm", "전이 RMSE"), ("1,395", "관측점"), ("51 cm", "평균 ALT")],
            [("관측 정합.", "예측면이 관측 분포와 공간적으로 일치"),
             ("냉색 규약.", "붉은 계열·rainbow 배제, 깊이 순차형(oslo_r)"),
             ("한계.", "9km 기후 해상도가 미세지형을 못 담아 국지 편차")])

# ================================================================= 8 결과: 국소 고해상 데모 (hero)
figure_hero("결과", "06", "고해상 국소 데모",
            "북사면 평탄 툰드라의 활동층 두께와 불확실성",
            "assets/mid/local_demo.png",
            "(좌) PolSAR P밴드 원자료 30m · (중) 본 연구 예측(PolSAR+공변량 앙상블) · (우) Diffusion 90% 예측구간 폭(불확실성).",
            "전 지구 규모 지도가 놓치는 30m 미세 융해 패턴을 국소 영역에서 재현했다. 예측값 하나가 아니라 "
            "어디를 얼마나 믿을 수 있는지(90% 예측구간 폭)를 함께 지도로 제시하며, 불확실성이 큰 곳이 관측 추가·위험 점검의 우선 후보가 된다.",
            [("30 m", "공간 해상도"), ("3패널", "원자료·예측·불확실성"), ("90%", "예측구간")],
            [("직관.", "예측값 + 어디가 불확실한지를 함께"),
             ("고해상.", "30m 미세 융해 패턴 재현"),
             ("응용.", "불확실성 상위 = 위험 스크리닝 후보")])

# ================================================================= 9 결과: 불확실성·AOA
s = R.slide(prs); CH(s, "결과")
y = R.title_block(s, "06", "보정된 불확실성과 적용범위", "신뢰구간 보정과 외삽 경고의 결합")
_d1 = R.image(s, "assets/mid/conformal.png", ML, y+0.2, 4.7, 2.95, valign='top', align='left')
_fig7 = FIG()
R.caption(s, ML, _d1[1]+_d1[3]+0.08, 4.7, _fig7,
          "90% 예측구간의 실제 커버리지. raw 56.1%의 과신을 conformal(CQR)이 85.9%로 교정한다(구간 폭 29.6→50.6cm)")
_x2 = ML + 4.9 + 0.5
_d2 = R.image(s, "assets/mid/aoa_di.png", _x2, y+0.2, CR-_x2, 2.95, valign='top', align='left')
_fig8 = FIG()
R.caption(s, _x2, _d2[1]+_d2[3]+0.08, CR-_x2, _fig8,
          "적용범위(AOA) 진단. 학습환경 비유사도(DI) 분위수가 높을수록 전이 RMSE는 15.5에서 27.1cm로 늘고 "
          "90% 커버리지는 69에서 51%로 떨어진다")
_by = y + 3.74
R.rule(s, ML, _by, CW, color=RULE, wt=0.8)
R.text(s, ML, _by+0.1, CW, 0.35,
       [[{'t': "용어  ", 'size': 10, 'color': TEAL, 'font': SANS_S},
         {'t': "conformal = 검증 오차 분포로 예측구간을 재보정하는 분포무가정 기법  ·  AOA(적용범위) = 학습 공변량과의 "
               "비유사도(DI)가 임계 이하인 영역", 'size': 10, 'color': SLATE, 'font': SANS_M}]], ls=1.25)
R.finding(s, ML, _by+0.48, CW, "의미",
          "예측지도는 예측값·신뢰구간·적용범위의 3종 세트로 제공한다. "
          f"커버리지 보정 56→86%는 그림 {_fig7}, DI 상위 구간의 악화(RMSE 15.5→27.1cm, 커버리지 69→51%)는 "
          f"그림 {_fig8}에 근거하며, 외삽 경고가 실제로 필요함을 보인다.")

# ================================================================= 10 결과: 얕은 지중 열구조 (2.5D)
figure_hero("결과", "06", "얕은 지중 열구조 (2.5D)",
            "위도-깊이 수직 단면과 깊이 슬라이스",
            "assets/mid/cross_section.png",
            "(좌) 남북 방향 위도-깊이 단면(°C), 청록 파선이 0°C 등온면(영구동토 상단). (우) 깊이 0.5~20m 슬라이스 지도.",
            "활동층 두께(2D)를 넘어 깊이 0~20m 연평균 지중온도(MAGT)를 시추공 10,748점으로 학습한 GBM 조건장으로 복원했다. "
            "0°C 등온면(청록 파선)이 영구동토 상단을 규정하고, 심부로 갈수록 균질한 냉기로 수렴한다.",
            [("0~20 m", "복원 깊이"), ("0°C", "동토 상단 경계"), ("10,748", "시추공 라벨")],
            [("2D→2.5D.", "활동층 두께에서 지중 열구조로 확장"),
             ("물리 경계.", "0°C 등온면 = 영구동토 상단"),
             ("엔진.", "복원 엔진은 GBM 조건장. 후보였던 신경장은 정확도 게이트 미통과로 제외")])

# ================================================================= 11 결과: MAGT 깊이별 지도
figure_hero("결과", "06", "깊이별 지중온도 지도",
            "2m와 20m 연평균 지중온도의 공간 분포",
            "assets/mid/magt_2m_20m.png",
            "동일 색규약(vik, 0°C 중심 발산)으로 얕은 깊이(2m)와 깊은 깊이(20m)의 연평균 지중온도를 나란히 비교.",
            "얕은 2m는 지표 기후를 따라 변동이 크고 국지 온난역이 나타나는 반면, 20m는 장기 누적된 안정된 냉기가 넓게 분포한다. "
            "두 깊이의 대비가 지중 열구조의 수직 변화를 보여주며, 두 층 사이에서 0°C를 지나는 깊이가 영구동토 상단 경계를 정의한다. "
            "이 깊이별 온도장이 앞의 3D 열큐브를 이루는 재료다.",
            [("2 m / 20 m", "비교 깊이"), ("vik", "0°C 중심 발산"), ("−9~+3°C", "MAGT 범위")],
            [("표층.", "2m는 기후 구배에 민감"),
             ("심부.", "20m는 장기 열상태·동토 분포 대표"),
             ("연결.", "0°C 경계 심도 → 3D 열큐브")])

# ================================================================= 12 정직한 평가 (정량: CV 방식별 누설)
figure_hero("정직한 평가", "07", "누설을 통제한 평가",
            "무작위 CV는 전이 성능을 약 4배 과대평가한다",
            "assets/mid/cv_leakage.png",
            "CV 분할 방식별 RMSE. 같은 데이터·같은 모델에서 분할만 바꾼 결과로, 무작위 분할이 성능을 크게 부풀린다.",
            "공간적으로 가까운 관측은 서로 닮아, 무작위 분할은 학습에서 본 것과 거의 같은 점으로 시험을 본다. "
            "같은 데이터·같은 모델에서 분할만 바꾸면 IDW 오차가 28.0에서 111.4cm로 약 4배 커진다. 헤드라인 지표는 공간블록·LORO만 쓴다.",
            [("약 4배", "무작위→LORO 오차 증가"), ("28.0→111.4", "IDW RMSE(cm)"), ("LORO", "지역 통째 제외 평가")],
            [("원칙.", "헤드라인은 공간블록·LORO만 보고"),
             ("의미.", "낮아 보이는 수치가 실제 전이 성능"),
             ("외부.", "타 논문 높은 R²도 상당수 누설 낙관")])

# ================================================================= 13 정보 병목
figure_hero("핵심 발견", "07", "정보 병목 진단",
            "한계는 모델 용량이 아니라 공변량 정보다",
            "assets/mid/bottleneck.png",
            "셀 단위 전이(LORO) 평가의 feature-group ablation. 막대는 skill(평균 예측 대비 RMSE 개선율), 청록이 위경도 대조군.",
            "여러 모델을 바꿔도 정확도가 거의 같다면 병목은 모델이 아니라 입력 정보다. 결정적 증거가 위경도 대조군이다. "
            "위도·경도 2개만 쓴 모델이 InSAR를 포함한 물리 공변량 전체 조합을 능가한다(skill 14.7% > 12.7%). "
            "위도가 기후를 대리할 뿐인데도 물리 정보가 이를 못 넘는다는 것은 현재 공변량의 정보량 자체가 부족함을 뜻한다. "
            "따라서 지렛대는 모델 교체가 아니라 정보 확충이다.",
            [("14.7%", "위경도 skill"), ("12.7%", "물리 최고(+InSAR)"), ("6모델", "전부 동률")],
            [("동률.", "6모델 전부 앙상블 16.95cm 수렴"),
             ("증거.", "위경도 대조군이 물리 조합 이상"),
             ("지렛대.", "모델이 아니라 정보(물리·새 데이터)")])

# ================================================================= 13b 6모델 공간 오차 지도 (P0, 전폭)
s = R.slide(prs); CH(s, "핵심 발견")
y = R.title_block(s, "07", "6모델 공간 오차 지도", "여섯 모델의 오차 분포가 공간적으로 거의 같다")
# 오차 지도를 전폭으로 크게 배치(판독성 우선). 하단에 캡션 + 지표·해석 한 행.
_d = R.image(s, "assets/mid/tournament_errors.png", ML, y+0.06, CW, SH-y-1.78,
             valign='top', align='center')
_cy = _d[1]+_d[3]+0.05
R.caption(s, ML, _cy, CW, FIG(),
          "위치 동등가중(N=14,348) 여섯 모델·앙상블의 예측 오차(예측−관측, cm)와 RMSE·skill 막대. "
          "색은 0 중심 발산형(broc). 오차가 커지는 위치·크기가 모델과 무관하게 겹친다",
          src="자료: model_tournament_predictions.csv · 위치 groupby 재채점")
# 지표 스트립(좌) + 해석(우) — 푸터 위로 고정
_by = min(_cy + 0.48, BLIM - 0.70)
metric_strip(s, ML, _by, CW*0.52, [("16.1 cm", "최저(GBM)"), ("20.5 cm", "최고(TabM)"),
                                    ("16~18", "6모델 밀집대(cm)")], vsize=14)
nx2 = ML + CW*0.52 + 0.4
R.vrule(s, nx2-0.22, _by+0.02, 0.62, color=RULE, wt=0.9)
R.text(s, nx2, _by, CR-nx2, 0.6,
       [[{'t': "함의  ", 'size': 11, 'color': TEAL, 'font': SANS_S},
         {'t': "오차의 공간 패턴이 6모델에서 사실상 동일하다. 구조적으로 어려운 지역이 모델과 무관하므로, "
               "한계는 모델이 아니라 입력 정보에 있다.", 'size': 11, 'color': INK2, 'font': SANS}]], ls=1.28)

# ================================================================= 14 선행연구 대비
figure_hero("선행연구 대비", "07", "선행연구 대비 성능",
            "절대 오차가 아니라 평가 프로토콜을 통제해 비교한다",
            "assets/mid/sota.png",
            "보고 RMSE 비교. 회색은 무작위 CV·제품, 초록은 물리기반 사이트검증, 남색이 본 연구(공간블록·LORO).",
            "논문마다 평가 방식이 달라 절대 오차만 나란히 두면 오해한다. 통제해 보면, 같은 알래스카 데이터로 학습한 논문이 관대한 무작위 CV로도 "
            "검증 22cm인데 우리는 엄격한 LORO로 16.95cm다. 공개 ESA CCI 제품을 우리 셀에 직접 채점하면 20.6cm로 우리보다 나쁘다. "
            "우리 성능은 물리기반 정직 검증군(14~18cm)과 같은 대역에 있다. 최소 오차가 아니라 정직하게 얻은 대표 성능이 우리 위치다.",
            [("16.95 cm", "본 연구(LORO)"), ("20.6 cm", "공개 CCI 제품"), ("22 cm", "동급 논문(무작위)")],
            [("동급.", "같은 권역·데이터에서 우리가 앞섬"),
             ("제품.", "공개 CCI 제품보다 정확"),
             ("주장.", "최소 오차 아닌 정직한 대표 성능")])

# ================================================================= 15 전이 검증
figure_hero("결과", "07", "전이 검증",
            "실측 공변량이 타 지역 전이를 개선한다",
            "assets/mid/era5_transfer.png",
            "지역간 전이(LORO: 한 지역을 통째로 제외하고 학습해 그 지역에서 평가)의 공변량 종류별 RMSE.",
            "알래스카에서 학습한 모델을 다른 지역에 적용하는 전이가 실제 목표다. 거친 WorldClim 기후 대신 실측 기반 ERA5-Land 공변량으로 바꾸자 "
            "전이 오차가 108.5에서 87.3cm로 20% 낮아졌다. 정보를 늘리는 경로가 실제로 통함을 보인다. 다만 현재 전이 검증의 '지역'이 "
            "사실상 알래스카 내부라, 티베트·시베리아를 편입한 진짜 다지역 전이(E1)로 재검증이 필요하다.",
            [("−20%", "전이 개선"), ("108.5→87.3", "LORO RMSE(cm)"), ("E1", "다지역 재검증")],
            [("개선.", "실측 공변량이 전이 20% 향상"),
             ("발견.", "현재 전이 평가가 사실상 알래스카 내부"),
             ("다음.", "티베트·시베리아로 진짜 다지역 전이(E1)")])

# ================================================================= 15b 다지역 전이 실측 (P1)
s = R.slide(prs); CH(s, "결과")
y = R.title_block(s, "07", "다지역 전이 실측 (E1 착수)",
                  "시베리아·티베트를 편입한 전 공변량 통합 재학습")
_d1 = R.image(s, "assets/mid/unified_transfer.png", ML, y+0.16, 7.15, 3.65, valign='top', align='left')
_figt = FIG()
R.caption(s, ML, _d1[1]+_d1[3]+0.07, 7.15, _figt,
          "지역 제외(LORO) 전이의 지역·모델별 RMSE. 알래스카 전이에서 딥러닝(FT-T·앙상블)이 GBM을 앞서고, "
          "신규 레나델타 전이는 전 모델이 미달한다",
          src="자료: unified_tournament_perregion.csv · 전 공변량(25) 통합 학습")
nx = ML + 7.15 + 0.45
R.text(s, nx, y+0.16, CR-nx, 0.3, [[{'t': "실측 결과", 'size': 11, 'color': TEAL, 'font': SANS_S, 'spc': 0.5}]])
R.bullets(s, nx, y+0.56, CR-nx, [
    ("전이 우세.", "알래스카 전이서 FT-T·앙상블 15.0cm로 GBM 17.6cm 상회. 이질 소표본에서 DL 이점 첫 확인"),
    ("병목 지속.", "레나델타 전이는 25~30cm로 전 모델 미달. 알래스카 학습이 깊은 ALT로 과대예측"),
    ("결측 라우팅.", "신규 지역 InSAR 전면 결측 처리에 따라 전이 성능이 뒤집힌다(중앙값 대체 필수)"),
    ("판정.", "통합 학습은 현 결측 처리에서 미채택. 전이 병목 정량화는 채택"),
], size=11, gap=8)
R.rule(s, nx, y+3.02, CR-nx, color=RULE, wt=0.8)
R.finding(s, nx, y+3.18, CR-nx, "다음",
          "모달리티 드롭아웃으로 결측을 학습 신호화하고, Stefan 물리로 전이 편향을 교정한다.")

# ================================================================= 16 음성 결과
s = R.slide(prs); CH(s, "정직성")
y = R.title_block(s, "07", "음성 결과", "시계열 DL과 CCI prior의 게이트 판정")
tl = "assets/figs/tlite_gate.png"
if os.path.exists(tl):
    _d = R.image(s, tl, ML, y+0.2, 6.6, 3.6, valign='top', align='left')
    R.caption(s, ML, _d[1]+_d[3]+0.08, 6.6, FIG(),
              "시계열 딥러닝(GRU·TCN)은 시간 홀드아웃에서 GBM에 미달해 게이트를 통과하지 못했다",
              src="자료: tlite_sequence_gate_results.csv · ALT RMSE(cm), 낮을수록 정확")
R.text(s, 7.3, y+0.25, CR-7.3, 0.3, [[{'t': "판정", 'size': 11, 'color': TEAL, 'font': SANS_S, 'spc': 0.5}]])
R.bullets(s, 7.3, y+0.6, CR-7.3, [
    ("게이트.", "사전 등록 기준으로, 신규 모듈이 GBM 대비 개선할 때만 채택한다"),
    ("시계열 DL.", "temporal holdout에서 GBM 15.9 < GRU 19.2 < TCN 23.8, 게이트 미통과"),
    ("CCI prior.", "기존 제품 사전정보는 기후 공변량과 중복되어 개선 없음(v4 기준)"),
    ("가치.", "음성 결과를 숨기지 않는 것이 정직 평가 체계의 일부다"),
], size=11.5, gap=10)
R.rule(s, 7.3, y+3.15, CR-7.3, color=RULE, wt=0.8)
R.finding(s, 7.3, y+3.3, CR-7.3, "판정 결과",
          "정확도가 동률이므로 주력은 GBM 조건장으로, 불확실성·생성 경로는 Diffusion으로 확정한다.")

# ================================================================= 17 대회 일정·계획
s = R.slide(prs); CH(s, "일정·계획")
y = R.title_block(s, "08", "연구 현황과 향후 계획", "완료·진행·예정과 대회 일정")
R.image(s, "assets/mid/timeline.png", ML, y+0.35, CW, SH-y-1.6, valign='top')
R.text(s, ML, SH-1.05, CW, 0.6,
       [[{'t': "산출물 목표", 'size': 11, 'color': TEAL, 'font': SANS_S},
         {'t': "  고해상 활동층 두께·불확실성 지도 세트 · 얕은 3D 열큐브 · 전이 검증표 · KPDC 현장 검증 · 재현 코드",
          'size': 11, 'color': INK2, 'font': SANS}]], ls=1.2)

# ================================================================= 18 요약 (흰 배경 통일)
s = R.slide(prs); CH(s, "요약")
y = R.title_block(s, "09", "요약", "정직한 진단 위에 데이터와 물리 결합으로 확장한다")
items = [
    ("연구 목적", "관측이 희소한 극지에서 활동층 두께와 얕은 지중 열구조를 신뢰 있게 지도화한다"),
    ("핵심 발견", "정확도의 한계는 모델이 아니라 공변량 정보다. 위경도 대조군(skill 14.7%)이 물리 공변량 조합을 능가한다"),
    ("차별성", "누설 통제 평가(무작위 CV 약 4배 낙관 규명), 보정 불확실성(56→86%), 적용범위, 관측기반 얕은 3D 열구조"),
    ("다음 지렛대", "정보를 늘리는 경로(Stefan 물리 결합·진짜 다지역 전이)가 모델 교체보다 유망하다"),
]
yb = y+0.32
for h, b in items:
    R.text(s, ML, yb, CW, 0.6,
           [[{'t': h+"    ", 'size': 13, 'color': TEAL, 'font': SANS_S},
             {'t': b, 'size': 12.5, 'color': INK2, 'font': SANS}]], ls=1.3, sa=2)
    yb += 0.80
R.rule(s, ML, yb+0.12, CW, color=RULE, wt=0.8)
metric_strip(s, ML, yb+0.32, CW, [
    ("16.95 cm", "전이 RMSE(공간블록·LORO)"),
    ("14.7%", "위경도 대조군 skill"),
    ("56→86%", "conformal UQ 90% 커버리지 보정"),
    ("−20%", "실측 공변량 전이 개선"),
], vsize=16)

out = "render/permafrost_midreport.pptx"
prs.save(out)
print("saved", out, "· slides:", len(prs.slides._sldIdLst))
