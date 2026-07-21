# -*- coding: utf-8 -*-
"""중간보고 슬림 덱 (11p) · 목적·배경 → 파이프라인 → 본론(main 지도, 상세 설명) → 결론·일정.
처음 보는 사람도 이해하도록 용어를 정의하며 각 슬라이드에 입력·방법·출력을 명시.
방법론 상세(누설통제 원리·conformal 유도·정보병목 진단)는 제외. main 산출=ALT 예측.
report_lib.py 재사용. 출력 deck/render/permafrost_summary.pptx
빌드: (deck/) python3 build_summary.py · 렌더 soffice --headless --convert-to pdf
"""
import os
import report_lib as R
from report_lib import (ML, MR, MT, MB, CW, CR, SW, SH,
                        PAPER, INK, INK2, SLATE, MUTE, TEAL, TEAL2, NAVY, GOLD, RULE, RULE2, FILL,
                        INK_DK, WHITE, SERIF, SANS, SANS_M, SANS_S, SANS_L)
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

def F(p): return os.path.join("..", p)

prs = R.new_deck()
TOTAL = 11
_SN = [1]
def CH(s, section):
    _SN[0] += 1
    R.chrome(s, _SN[0], TOTAL, section)
_FIG = [0]
def FIG():
    _FIG[0] += 1
    return _FIG[0]
BLIM = SH - 0.56

def metric_strip(s, x, y, w, metrics, vsize=15):
    n = len(metrics); cw = w / n
    for i, (val, lab) in enumerate(metrics):
        xx = x + i*cw
        if i > 0:
            R.vrule(s, xx, y+0.05, 0.58, color=RULE, wt=0.9)
        R.text(s, xx+0.1, y, cw-0.15, 0.38,
               [[{'t': val, 'size': vsize, 'color': TEAL, 'font': SERIF, 'bold': True}]], anchor=MSO_ANCHOR.MIDDLE)
        R.text(s, xx+0.1, y+0.40, cw-0.15, 0.26,
               [[{'t': lab, 'size': 8.5, 'color': SLATE, 'font': SANS_M}]], ls=1.02)

def io_block(s, x, y, w, io, size=10):
    """입력·방법·출력·용어 설명 블록(run-in 라벨). io = [(라벨, 본문), ...]."""
    R.bullets(s, x, y, w, io, size=size, gap=5)

def body(section, no, title, sub, img, cap, io, metrics, notes, cap_src=None):
    """본론 슬라이드: 그림 + 입력/방법/출력 설명 + 지표 + 해석."""
    s = R.slide(prs); CH(s, section)
    y = R.title_block(s, no, title, sub) if False else R.title_block(s, no, title, sub)
    path = img if os.path.exists(img) else F(img)
    ar = R._aspect(path)
    fig_no = FIG()
    if ar > 2.0:
        # 넓은 그림: 상단 그림 + 캡션, 하단 좌(설명)·우(지표+해석)
        ih = min(CW / ar, SH - y - 2.62)
        dx, dy, dw, dh = R.image(s, img, ML, y+0.06, CW, ih, valign='top', align='center')
        cy = dy + dh + 0.05
        R.caption(s, ML, cy, CW, fig_no, cap, src=cap_src)
        by = cy + (0.46 if cap_src else 0.34)
        ew = CW*0.585
        R.text(s, ML, by, ew, 0.26, [[{'t': "입력·방법·출력", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
        io_block(s, ML, by+0.26, ew, io, size=9.7)
        nx = ML + ew + 0.4
        R.vrule(s, nx-0.22, by+0.02, min(BLIM-by-0.04, 1.35), color=RULE, wt=0.9)
        if metrics:
            metric_strip(s, nx, by, CR-nx, metrics, vsize=12.5)
            ny = by + 0.78
        else:
            ny = by
        if notes:
            R.text(s, nx, ny, CR-nx, 0.24, [[{'t': "해석", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
            R.bullets(s, nx, ny+0.24, CR-nx, notes, size=9.7, gap=4)
    else:
        # 좁은 그림: 좌 그림, 우 설명(입력/방법/출력) + 지표 + 해석
        ah = SH - y - 1.02
        fw = min(ah * ar, 6.45)
        dx, dy, dw, dh = R.image(s, img, ML, y+0.1, fw, ah, valign='top', align='left')
        R.caption(s, ML, dy+dh+0.06, fw, fig_no, cap, src=cap_src)
        nx = ML + fw + 0.45; nw = CR - nx; yy = y+0.14
        R.text(s, nx, yy, nw, 0.24, [[{'t': "입력·방법·출력", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
        io_block(s, nx, yy+0.26, nw, io, size=10)
        yy += 0.26 + len(io)*0.475 + 0.12
        if metrics:
            R.rule(s, nx, yy, nw, color=RULE, wt=0.7)
            metric_strip(s, nx, yy+0.12, nw, metrics, vsize=13.5)
            yy += 0.95
        if notes:
            R.text(s, nx, yy, nw, 0.24, [[{'t': "해석", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
            R.bullets(s, nx, yy+0.26, nw, notes, size=10, gap=4)
    return s

# ================================================================= 1 표지
s = R.slide(prs, bg=WHITE)
R.text(s, ML, 1.55, CW, 0.3,
       [[{'t': "연구 중간보고 · 2026. 07", 'size': 12.5, 'color': SLATE, 'font': SANS_S, 'spc': 1.0}]])
R.text(s, ML, 2.1, CW*0.98, 1.5,
       [[{'t': "북극 영구동토 ", 'size': 33, 'color': INK, 'font': SANS_L},
         {'t': "활동층 두께(ALT)", 'size': 33, 'color': INK, 'font': SERIF, 'bold': True},
         {'t': " 예측", 'size': 33, 'color': INK, 'font': SANS_L}],
        [{'t': "관측 기반 GeoAI · ", 'size': 33, 'color': INK, 'font': SANS_L},
         {'t': "3D 지중온도장", 'size': 33, 'color': INK, 'font': SERIF, 'bold': True},
         {'t': "으로 증강", 'size': 33, 'color': INK, 'font': SANS_L}]], ls=1.28)
R.text(s, ML, 3.9, CW*0.92, 0.4,
       [[{'t': "다지역 활동층 두께 지도 · 불확실성 · 3D 지중온도장 · 물리 결합 전이",
          'size': 13.5, 'color': TEAL, 'font': SANS_S}]])
R.rule(s, ML, 5.1, 2.2, color=TEAL, wt=2.4)
R.text(s, ML, 5.35, CW, 0.6,
       [[{'t': "Polar_Bigdata 연구팀", 'size': 13, 'color': INK, 'font': SANS_S}],
        [{'t': "2026 극지 빅데이터 · 인공지능 활용 경진대회 준비", 'size': 11, 'color': SLATE, 'font': SANS}]], ls=1.3, sa=2)

# ================================================================= 2 연구 배경·목적
s = R.slide(prs); CH(s, "배경·목적")
y = R.title_block(s, "01", "연구 배경과 목적", "활동층 두께를 왜, 무엇으로 만드는가")
R.bullets(s, ML, y+0.2, 6.05, [
    ("현안.", "영구동토(연중 0°C 이하로 언 땅) 융해는 탄소 방출과 지반 침하로 인한 인프라 위험과 직결된다"),
    ("지표.", "여름에 녹는 표층의 두께(활동층 두께, ALT)가 융해 정도를 대표하는 핵심 지표다"),
    ("한계.", "현장 관측은 희소하고 위성은 간접적이라 넓은 지역의 신뢰 있는 지도가 없다"),
    ("목적.", "관측 기반 GeoAI로 ALT 2D 지도와 불확실성을 만들고, 3D 지중온도장으로 라벨을 증강한다"),
], size=12.5, gap=13)
_d = R.image(s, "assets/mid/concept_alt.png", 6.9, y+0.2, 5.7, 3.7, valign='top')
R.caption(s, 6.9, _d[1]+_d[3]+0.08, 5.7, FIG(),
          "활동층과 영구동토. 연중 최고 지중온도가 0°C를 지나는 깊이가 활동층 두께(ALT)다")
R.rule(s, ML, BLIM-0.62, CW, color=RULE, wt=0.8)
R.text(s, ML, BLIM-0.5, CW, 0.4,
       [[{'t': "최종 산출물   ", 'size': 11.5, 'color': TEAL, 'font': SANS_S},
         {'t': "① ALT 2D 예측 지도   ② 예측 불확실성 지도   ③ 3D 지중온도장(0°C 등온면)   "
               "④ 타 지대 전이 검증", 'size': 11.5, 'color': INK2, 'font': SANS_M}]], ls=1.2)

# ================================================================= 3 접근 개요(연결)
s = R.slide(prs); CH(s, "접근 개요")
y = R.title_block(s, "02", "접근 개요", "ALT 예측을 중심으로 3D 지중온도장과 물리를 결합")
_d = R.image(s, "assets/mid/connection.png", ML, y+0.16, CW, SH-y-1.35, valign='top', align='center')
R.caption(s, ML, _d[1]+_d[3]+0.06, CW, FIG(),
          "데이터가 있는 정도에 따라 ALT를 얻는 세 경로. 셋 다 0°C 등온면을 공유 경계로 하며 ALT 예측으로 수렴한다")
R.text(s, ML, BLIM-0.34, CW, 0.32,
       [[{'t': "main 트랙  ", 'size': 10.5, 'color': TEAL, 'font': SANS_S},
         {'t': "3D 지중온도장 라벨 증강(가장 실효적)  ·  ", 'size': 10.5, 'color': INK2, 'font': SANS_M},
         {'t': "병렬 트랙  ", 'size': 10.5, 'color': GOLD, 'font': SANS_S},
         {'t': "3D 지중온도장 기질 추정, Stefan·개선 물리 잔차학습", 'size': 10.5, 'color': INK2, 'font': SANS_M}]], ls=1.2)

# ================================================================= 4 데이터·모델 파이프라인 (아키텍처)
s = R.slide(prs); CH(s, "파이프라인")
y = R.title_block(s, "03", "데이터와 모델 파이프라인", "입력 공변량에서 산출물까지의 처리 흐름")
_d = R.image(s, "assets/mid/architecture.png", ML, y+0.1, CW, SH-y-2.15, valign='top', align='center')
R.caption(s, ML, _d[1]+_d[3]+0.05, CW, FIG(),
          "다중모달 입력을 위치별 표(tabular)로 정렬해 회귀모델에 넣고, 4종 산출물을 만든다")
_by = _d[1]+_d[3]+0.52
ew = CW*0.60
R.text(s, ML, _by, ew, 0.24, [[{'t': "구성", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, ML, _by+0.26, ew, [
    ("입력(공변량).", "기후(ERA5-Land 재분석 8종)·지형(ArcticDEM 6종)·위성 SAR(InSAR·PolSAR)·시추공 지중온도"),
    ("정렬.", "각 관측 위치에 공변량 값을 부착해 위치×변수 표를 만든다. 공변량=예측에 쓰는 입력 변수"),
    ("모델.", "표 데이터에 강한 GBM(그래디언트 부스팅) 회귀. 공간블록 교차검증으로 낙관적 평가를 통제"),
], size=9.9, gap=5)
nx = ML + ew + 0.4
R.vrule(s, nx-0.22, _by+0.02, 1.3, color=RULE, wt=0.9)
R.text(s, nx, _by, CR-nx, 0.24, [[{'t': "산출물", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, nx, _by+0.26, CR-nx, [
    ("2D.", "ALT 예측 지도 + 불확실성"),
    ("3D.", "깊이별 지중온도장(0°C 등온면)"),
    ("검증.", "타 지대 전이 성능"),
], size=9.9, gap=5)

# ================================================================= 5 본론: ALT 예측 지도
body("본론", "04", "활동층 두께 예측 지도",
     "알래스카 전역 활동층 두께와 현장 관측",
     "assets/mid/alt_map.png",
     "ERA5-Land 기후 공변량으로 복원한 알래스카 활동층 두께(cm). 흰 점=CALM, 삼각형=ABoVE 현장 관측.",
     [("입력.", "ERA5-Land(유럽중기예보센터 지표 기후 재분석, 9km 월별)에서 뽑은 8종. 연평균기온, "
               "융해도일 TDD(여름철 0°C 넘는 기온의 누적), 동결도일 FDD, 최난·최한월 기온, 지표토양온도, 적설량"),
      ("방법.", "각 위치의 8개 기후값을 입력, 관측 ALT를 정답으로 GBM 회귀 학습(log 변환). 알래스카 전 격자에 적용"),
      ("출력.", "격자별 ALT(cm) 연속면. 관측 없는 곳도 기후로 채워 지도화")],
     [("16.95 cm", "전이 RMSE"), ("1,395", "관측점"), ("51 cm", "평균 ALT")],
     [("연속 예측.", "관측 공백을 공변량으로 채움"),
      ("한계.", "9km 기후 해상도의 국지 편차")])

# ================================================================= 6 본론: 불확실성 지도
body("본론", "04", "예측 불확실성 지도",
     "예측값과 함께 어디를 얼마나 믿을지 제시",
     "assets/mid/uncertainty_map.png",
     "(좌) 90% 예측구간 폭 지도(좁을수록 신뢰). (우) 실측 대비 예측 오차 지도. 두 지도가 정합하면 잘 보정된 것.",
     [("방법.", "같은 GBM을 분위수 회귀로 학습해 5%·95% 지점을 예측하면 90% 예측구간이 나온다. "
               "예측구간=참값이 그 안에 들어올 확률이 90%가 되도록 만든 범위"),
      ("보정.", "검증자료의 실제 오차로 구간 폭을 재조정해 명목 90%가 실제 90%에 맞도록 교정"),
      ("출력.", "위치별 구간 폭(신뢰도)과 실측 대비 오차를 지도로 제시")],
     [("50.6 cm", "평균 구간 폭"), ("90%", "구간 신뢰수준")],
     [("직관.", "예측값 + 신뢰 폭을 함께"),
      ("응용.", "불확실 상위 = 관측 추가 후보")])

# ================================================================= 7 본론: 고해상 국소 데모
body("본론", "04", "고해상 국소 데모",
     "북사면 평탄 툰드라의 30m 활동층 두께",
     "assets/mid/local_demo.png",
     "(좌) PolSAR 유도 ALT 30m · (중) 본 연구 예측 · (우) 90% 예측구간 폭.",
     [("PolSAR 원자료.", "편파 합성개구레이더. P밴드(장파장) 레이더는 지표 아래까지 투과해 30m 해상도로 ALT를 유도한다"),
      ("방법.", "PolSAR와 기후·지형 공변량을 함께 넣은 앙상블 모델로 예측(중). Diffusion(확률 생성모델)으로 "
               "예측의 90% 구간 폭을 산출(우)"),
      ("출력.", "전 지구 9km 지도가 놓치는 30m 미세 융해 패턴 + 불확실성")],
     [("30 m", "공간 해상도"), ("90%", "예측구간")],
     [("고해상.", "30m 미세 패턴 재현"),
      ("응용.", "위험 스크리닝 후보 식별")])

# ================================================================= 8 본론: 3D 지중온도장
body("본론", "04", "3D 지중온도장 깊이별 지도",
     "2m와 20m 연평균 지중온도와 0°C 경계",
     "assets/mid/magt_clean.png",
     "연평균 지중온도(MAGT) 2m와 20m. 남색 실선=0°C 등온선(영구동토 상단). 테두리점=시추공 실측.",
     [("입력.", "GTN-P 시추공의 깊이별 지중온도 10,748점 + 기후 공변량 + 깊이(m). MAGT=연평균 지중온도"),
      ("방법.", "깊이를 입력 변수에 포함해 (위치, 깊이)→온도를 GBM으로 회귀. 알래스카 격자·깊이 0~20m에 적용"),
      ("출력·연결.", "깊이별 온도장. 이 장이 0°C를 지나는 깊이가 곧 ALT이므로, 시추공은 있고 ALT 관측이 없는 "
                    "지역의 라벨 증강에 쓴다")],
     [("0~20 m", "복원 깊이"), ("10,748", "시추공 라벨")],
     [("표층.", "2m는 기후 따라 변동 큼"),
      ("심부.", "20m는 안정된 냉기")])

# ================================================================= 9 본론: 모델별 오차 분포 지도
body("본론", "04", "모델별 오차 분포 지도",
     "여섯 모델의 오차 분포가 공간적으로 거의 같다",
     "assets/mid/tournament_errors.png",
     "위치 동등가중(같은 위치의 반복 관측을 1회로 집계, 14,348 위치) 여섯 모델·앙상블의 예측 오차(예측−관측). "
     "우하단은 모델별 RMSE·skill 막대.",
     [("무엇.", "GBM·MLP·Transformer·TabM·Flow·Diffusion 6종과 앙상블을 같은 조건에서 비교"),
      ("수치 차이.", "이 지도는 위치 동등가중(14,348 위치) 기준 16~20cm다. 다른 슬라이드의 헤드라인 16.95cm는 "
                   "점 단위(22만 관측) 앙상블 값으로, 집계 방식만 다를 뿐 같은 평가다"),
      ("함의.", "모델을 바꿔도 오차가 커지는 위치와 크기가 거의 겹친다. 한계는 모델이 아니라 입력 정보다")],
     [("16.1 cm", "최저(GBM)"), ("20.5 cm", "최고(TabM)")],
     [("동률.", "6모델 16~20cm 밀집"),
      ("지렛대.", "모델 아닌 정보 확충")])

# ================================================================= 10 본론: 선행연구 대비 성능
body("본론", "04", "선행연구 대비 성능",
     "평가 프로토콜을 통제해 비교한다",
     "assets/mid/sota.png",
     "보고 RMSE 비교. 회색=무작위 CV·제품, 초록=물리기반 사이트검증, 남색=본 연구(공간블록·전이).",
     [("선정 기준.", "같은 대상(영구동토 활동층)·비슷한 지역의 대표 연구를 골랐다. 핵심은 절대 오차가 아니라 "
                   "평가 프로토콜 통제다. 무작위 CV는 이웃 관측이 학습·시험에 섞여 오차를 낙관한다"),
      ("기법.", "Gautam 2025=랜덤포레스트(68점), Liu 2024=격자 통계·ML, QTP 2025=청장고원 ML, "
               "ESA CCI 제품=CryoGrid 물리모델+위성 강제자료로 만든 공개 영구동토 제품(우리 셀에 직접 채점)"),
      ("13cm 관련.", "무작위 CV나 좁은 범위로 좁히면 13cm급도 보고된다. 우리 평탄툰드라 12.97cm도 같은 "
                    "범위축소 효과이지 진짜 돌파가 아니다. 엄격한 전이로는 16.95cm가 정직한 대표값이다")],
     [("16.95 cm", "본 연구(전이)"), ("20.6 cm", "공개 CCI 제품")],
     [("동급.", "정직 검증군(14~18cm) 대역"),
      ("제품.", "공개 CCI보다 정확")])

# ================================================================= 11 결론 + 완료·진행·예정·일정
s = R.slide(prs); CH(s, "결론·계획")
y = R.title_block(s, "05", "현재까지 결론과 향후 계획", "완료·진행·예정과 대회 일정")
R.bullets(s, ML, y+0.14, CW, [
    ("결론.", "알래스카 전역 ALT 지도·불확실성·3D 지중온도장을 산출. 전이 16.95cm로 정직 검증 대역, 공개 제품(20.6cm)보다 정확"),
    ("병목.", "여섯 모델이 동률이다. 한계는 모델이 아니라 공변량 정보와 지역 다양성이다. 라벨을 14,348에서 17,423셀로 확대"),
], size=11.5, gap=8)
_ty = y+1.28
R.image(s, "assets/mid/timeline.png", ML, _ty, CW, SH-_ty-1.05, valign='top', align='center')
R.rule(s, ML, BLIM-0.62, CW, color=RULE, wt=0.8)
R.text(s, ML, BLIM-0.5, CW, 0.4,
       [[{'t': "다음 지렛대   ", 'size': 11, 'color': TEAL, 'font': SANS_S},
         {'t': "3D 지중온도장 0°C 라벨 증강(main) · 3D 기질추정 · Stefan/개선 물리 잔차학습(병렬). "
               "각 실험은 전 입력 활용·데이터 관리·평가를 기록", 'size': 11, 'color': INK2, 'font': SANS_M}]], ls=1.2)

out = "render/permafrost_summary.pptx"
prs.save(out)
print("saved", out, "· slides:", len(prs.slides._sldIdLst))
