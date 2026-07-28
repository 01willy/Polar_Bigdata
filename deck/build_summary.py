# -*- coding: utf-8 -*-
"""발표 PPT 10장 (대회 예선용 · 블라인드). 최신 결과(S6~E2) 반영 재빌드.
스토리보드: 문제·목표 → 데이터·입력 → 방법 개요 → ALT 지도 → 정보병목 →
전이(물리앵커 vs ML vs co-kriging) → 증강 분석 → UQ 보정 → 시계열+얕은3D → KPDC 검증·결론.
처음 보는 심사자가 각 장에서 (주장형 제목 · 핵심수치 · 데이터출처/기법 1줄)을 읽고 이해하도록 구성.
report_lib.py 재사용. 출력 deck/render/permafrost_summary.pptx
빌드: (deck/) python3 build_summary.py · 렌더 soffice --headless --convert-to pdf
문체: 렌더 텍스트 em-dash/en-dash 연결자 금지(콜론·가운뎃점·괄호·문장분리), 보고서톤(~이다/~한다).
블라인드: 학교·연구실·기관·개인명·이메일 노출 금지.
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
TOTAL = 10
_SN = [0]
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

def _resolve(img):
    return img if os.path.exists(img) else F(img)

def body(section, no, kicker, claim, img, cap, io, metrics, notes, cap_src=None, fw_cap=6.45):
    """본론 슬라이드: 그림 + 입력/방법/출력 설명 + 지표 + 해석.
    kicker=짧은 섹션 라벨(작게), claim=주장형 제목(크게)."""
    s = R.slide(prs); CH(s, section)
    y = R.title_block(s, no, kicker, claim)
    path = _resolve(img)
    ar = R._aspect(path)
    fig_no = FIG()
    if ar > 1.8:
        # 넓은 그림: 상단 그림 + 캡션, 하단 좌(설명)·우(지표+해석)
        # 높이구속 완화(reserve 2.62→2.40)로 그림이 슬라이드 폭·높이를 더 채운다.
        ih = min(CW / ar, SH - y - 2.40)
        dx, dy, dw, dh = R.image(s, path, ML, y+0.06, CW, ih, valign='top', align='center')
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
        fw = min(ah * ar, fw_cap)
        dx, dy, dw, dh = R.image(s, path, ML, y+0.1, fw, ah, valign='top', align='left')
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

# ============================================================ 1 문제·목표
s = R.slide(prs); CH(s, "문제·목표")
y = R.title_block(s, "01", "문제와 목표",
                  "활동층 두께가 영구동토 융해를 대표하는 지표이며, 넓은 지역의 신뢰 지도가 없다")
_d = R.image(s, "assets/mid/concept.png", ML, y+0.14, CW, SH-y-2.10, valign='top', align='center')
R.caption(s, ML, _d[1]+_d[3]+0.06, CW, FIG(),
          "관측 기반 GeoAI 개요. 다중모달 관측을 입력으로 활동층 두께(ALT) 지도·불확실성·3D 지중온도장을 만든다.")
_by = _d[1]+_d[3]+0.5
ew = CW*0.585
R.text(s, ML, _by, ew, 0.24, [[{'t': "배경", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, ML, _by+0.26, ew, [
    ("현안.", "영구동토(연중 0°C 이하로 언 땅) 융해는 탄소 방출·지반 침하 위험과 직결된다"),
    ("지표.", "여름에 녹는 표층의 두께(활동층 두께, ALT)가 융해 정도를 대표한다"),
    ("한계.", "현장 관측은 희소하고 위성은 간접적이라 넓은 지역의 신뢰 지도가 없다"),
], size=9.9, gap=5)
nx = ML + ew + 0.4
R.vrule(s, nx-0.22, _by+0.02, 1.3, color=RULE, wt=0.9)
R.text(s, nx, _by, CR-nx, 0.24, [[{'t': "목표(최종 산출물)", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, nx, _by+0.26, CR-nx, [
    ("2D.", "알래스카 ALT 예측 지도"),
    ("얕은 3D.", "깊이별 지중온도장(0°C 등온면)"),
    ("UQ.", "보정된 예측 불확실성"),
    ("전이.", "타 지대 일반화 검증"),
], size=9.9, gap=4)

# ============================================================ 2 데이터·입력
body("데이터·입력", "02", "데이터와 입력",
     "관측·재분석·위성을 위치별로 정렬해 예측 입력으로 쓴다",
     "outputs/maps/data_inventory_world.png",
     "관측 데이터 인벤토리와 지역별 가용성. KPDC Council(수어드반도)이 검증 자료로 편입된다.",
     [("관측(라벨).", "CALM·ABoVE 활동층 두께, GTN-P 시추공 깊이별 지중온도. KPDC Council 일별 프로파일은 0°C 등온선 검증에 사용"),
      ("공변량 14종.", "지형 6종(ArcticDEM 유래 표고·경사 등)·기후 8종(ERA5-Land 재분석: 연평균기온·융해도일 TDD·동결도일 FDD 등)"),
      ("보조.", "위성 SAR(InSAR·PolSAR)는 스칼라 특징으로 편입. 각 관측 위치에 공변량을 부착해 위치×변수 표를 구성")],
     [("14종", "핵심 공변량"), ("KPDC", "Council 검증"), ("다지역", "AK·Lena·Canada")],
     [("출처.", "재분석·공개 관측망·KPDC"),
      ("정렬.", "위치별 tabular 표로 통합")],
     cap_src="자료: ERA5-Land, ArcticDEM, CALM/GTN-P/ABoVE, KPDC Council. 지도는 육지 마스크·해양 결측 처리.")

# ============================================================ 3 방법 개요
body("방법 개요", "03", "방법 개요",
     "물리 앵커·ML·증강·누설통제·UQ를 한 파이프라인으로 결합한다",
     "assets/mid/architecture.png",
     "입력 공변량에서 산출물까지의 처리 흐름. 물리(Stefan)·ML·소스인지 융합을 비교하고 4종 산출물을 만든다.",
     [("모델·비교.", "Stefan 물리 앵커(전이 최선)·GBM 공변량 회귀·6모델 토너먼트·소스인지 융합·co-kriging/IDW 공간보간 기준선"),
      ("누설통제.", "0.5° 공간블록 교차검증·지역 leave-out(LORO)·대표성 분해·적용범위(AOA)로 낙관적 평가를 배제"),
      ("UQ·시계열.", "분위수 회귀를 conformal로 보정해 90% 구간 제공. 계절내 융해진행 D(t)는 KPDC로 별도 검증")],
     [("Stefan", "물리 앵커"), ("6모델", "토너먼트"), ("conformal", "UQ 보정")],
     [("원칙.", "솔버·렌더·평가 모듈 분리"),
      ("공유 경계.", "0°C 등온면으로 2D·3D 연결")])

# ============================================================ 4 ALT 예측 지도(main)
body("본론", "04", "ALT 예측 지도",
     "알래스카 전역 활동층 두께를 관측과 정합하게 복원한다",
     "outputs/figures/s1_baseline/alaska_obs_vs_pred_maps.png",
     "(a) 현장 관측 ALT와 (b)-(d) 상위 모델의 예측. 공간블록 out-of-fold(OOF) 평가로 낙관적 편향을 통제.",
     [("입력.", "각 격자에 ERA5-Land 기후·지형 공변량을 부착. 관측 ALT를 정답으로 회귀 학습(log 변환)"),
      ("방법.", "0.5° 공간블록 6-fold 교차검증. 이웃 관측이 학습·시험에 섞이지 않도록 셀 단위로 분리"),
      ("출력.", "격자별 ALT(cm) 연속면. 관측 공백을 공변량으로 채워 지도화")],
     [("14.7 cm", "MLP OOF RMSE"), ("15.7 cm", "CatBoost"), ("~50 cm", "평균 ALT")],
     [("in-domain.", "상위 모델 14-16cm 대역"),
      ("한계.", "9km 기후 해상도의 국지 편차")])

# ============================================================ 5 정보 병목 진단
body("본론", "05", "정보 병목 진단",
     "위경도 2개가 물리 공변량 24개와 동률이며 병목은 입력 정보량이다",
     "outputs/figures/06_deep_learning/alt_info_bottleneck.png",
     "공변량 조합별 셀 단위 RMSE. 위경도 2개만 쓴 대조군(점선)이 전체 24개와 동률 이상이다.",
     [("무엇.", "지형·기후·InSAR를 순차 추가한 조합과, 위경도 2개만 쓴 대조군을 같은 프로토콜로 비교"),
      ("결과.", "공간블록·전이(LORO) 두 축 모두 위경도 대조군이 전체 공변량과 동률. 위도가 기후 이상을 대리한다"),
      ("함의.", "예측 한계는 모델이나 물리 특징이 아니라 격자 공변량에 담긴 국소 정보량이다")],
     [("17.2 cm", "위경도(공간블록)"), ("15.7 cm", "위경도(전이)")],
     [("병목.", "입력 정보량이 상한을 정함"),
      ("지렛대.", "모델 교체 아닌 정보 확충")])

# ============================================================ 6 전이: 물리앵커 vs ML vs co-kriging
body("본론", "06", "전이 성능 비교",
     "전이에서 공간보간은 붕괴하고, Stefan 물리 앵커만 양축 최선이다",
     "outputs/figures/e1_kriging/e1_rmse_bars.png",
     "(좌) 알래스카 in-domain 공간블록, (우) 지역 leave-out 전이. co-kriging·IDW·RK·GBM·Stefan 앵커 비교.",
     [("기준선.", "정통 공간보간(OK/IDW/RK)과 공변량 GBM을 물리 Stefan 앵커와 같은 프로토콜로 비교"),
      ("in-domain.", "OK 15.8·IDW 15.8이 GBM 17.7과 경쟁. 밀집 관측 안에서는 보간이 유효"),
      ("전이.", "지역이 바뀌면 보간이 붕괴(OK 29.4·IDW 50.9). variogram이 지역마다 달라 covariate shift에 취약")],
     [("14.5 cm", "Stefan in-domain"), ("21.3 cm", "Stefan 전이(LORO)")],
     [("Stefan.", "양축 모두 최선"),
      ("보간.", "전이에서 신뢰 불가")])

# ============================================================ 7 증강 비교분석
body("본론", "07", "증강 비교분석",
     "어떤 증강도 전이에서 Stefan 앵커를 넘지 못한다",
     "outputs/figures/s3_aug/aug_response_curves.png",
     "증강비율 r에 따른 전이 RMSE 개선량. 물리(Stefan/Ku) pseudo-label과 placebo(상수) 대조. 위가 개선.",
     [("설계.", "물리(Stefan)·부정확 물리(Ku)·placebo(상수) pseudo-label을 증강비율 r을 키우며 비교. 거리버퍼·블록 부트스트랩으로 누설 차단"),
      ("결과.", "물리선이 placebo 위에 있어 순가치가 존재하고, r=10까지 포화 없이 단조 개선한다. 부정확 물리(Ku)는 r이 클수록 악화"),
      ("경계.", "Canada는 물리가 전이를 견인, Lena는 base 모델 품질에 의존. 증강 이득의 크기·부호가 지역·물리 정확도에 좌우된다")],
     [("+10.4 cm", "Canada 물리 순가치"), ("r=10", "포화 없음")],
     [("순가치.", "정확 물리>placebo"),
      ("한계.", "앵커 대체는 못함")],
     fw_cap=5.55)

# ============================================================ 8 불확실성·보정(UQ)
body("본론", "08", "불확실성 보정",
     "원 분위수는 과신하며 conformal 보정으로 명목 90%를 실측 90%에 맞춘다",
     "outputs/figures/s11_uq/s11_calibration_curve.png",
     "(좌) calibration 곡선(점선=이상적). (우) 90% 구간의 실제 커버리지 cov90, 보정 전후.",
     [("문제.", "분위수 CatBoost의 원 90% 구간은 실제로 44.6%만 덮는 과신 상태. 구간 폭이 좁아 신뢰할 수 없다"),
      ("방법.", "학습 블록 내부에서 CQR(conformalized quantile regression)로 구간 폭을 재조정. 명목 90%가 실측 90%에 맞도록 보정"),
      ("결과.", "cov90 44.6%→93.4%[88.9,96.6], 구간 폭 14.7→53.6cm. 정직한 폭을 확보")],
     [("44.6% → 93.4%", "cov90 보정"), ("90%", "명목 신뢰수준")],
     [("과신 교정.", "폭을 정직하게 넓힘"),
      ("응용.", "불확실 상위=관측 후보")],
     fw_cap=6.05)

# ============================================================ 9 시계열 + 얕은 3D
s = R.slide(prs); CH(s, "본론")
y = R.title_block(s, "09", "시계열과 얕은 3D",
                  "연별 ALT를 재현하고 0°C 등온면으로 3D 지중온도장을 연결한다")
# 좌: S9 연별 패널, 우: S10 3D 등온면
gap = 0.4
lw = (CW - gap) * 0.52
rw = CW - gap - lw
# 왼쪽 그림
p9 = _resolve("outputs/figures/s9_timelapse/alt_year_panels_v2.png")
ar9 = R._aspect(p9)
lh = min(lw/ar9, SH - y - 2.55)
d9 = R.image(s, p9, ML, y+0.12, lw, lh, valign='top', align='center')
R.caption(s, ML, d9[1]+d9[3]+0.05, lw, FIG(),
          "연별 대표 ALT 4개 연도(공통 색축 40-75cm). 연도별 ERA5 강제와 연도별 ALT를 정합.")
# 오른쪽 그림
rx = ML + lw + gap
p10 = _resolve("outputs/volumes_3d/s10_shallow3d_isotherm.png")
ar10 = R._aspect(p10)
rh = min(rw/ar10, SH - y - 2.55)
d10 = R.image(s, p10, rx, y+0.12, rw, rh, valign='top', align='center')
R.caption(s, rx, d10[1]+d10[3]+0.05, rw, FIG(),
          "0°C 등온면 깊이(cm). 온도장은 사이트블록 R² 0.47로 성립.")
# 하단 설명 + 지표
_by = max(d9[1]+d9[3], d10[1]+d10[3]) + 0.5
ew = CW*0.60
R.text(s, ML, _by, ew, 0.24, [[{'t': "입력·방법·출력", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, ML, _by+0.26, ew, [
    ("시계열(S9).", "재학습 없이 2010-2024 연별 ALT 프레임을 렌더. 홀드아웃 RMSE 14.97cm·R² 0.34. 연 anomaly 상관 +0.06으로 연도간 변화는 예측 불가(각주)"),
    ("얕은 3D(S10).", "GTN-P 깊이별 지중온도를 (위치, 깊이)→온도로 회귀. 0°C를 지나는 깊이를 등온면으로 시각화"),
], size=9.9, gap=5)
nx = ML + ew + 0.4
R.vrule(s, nx-0.22, _by+0.02, 1.3, color=RULE, wt=0.9)
metric_strip(s, nx, _by, CR-nx, [("R² 0.47", "온도장(사이트블록)"), ("r 0.28", "0°C→ALT 상관")], vsize=12.5)
R.text(s, nx, _by+0.82, CR-nx, 0.24, [[{'t': "해석", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, nx, _by+1.06, CR-nx, [
    ("온도장.", "사이트블록 R² 0.47 성립"),
    ("절대 유도.", "0°C→ALT는 r 0.28(R²<0)"),
    ("계절 D(t).", "E2에서 계절내로 재정의"),
], size=9.5, gap=3)

# ============================================================ 10 KPDC 검증 + 결론
s = R.slide(prs); CH(s, "검증·결론")
y = R.title_block(s, "10", "KPDC 검증과 결론",
                  "KPDC Council 0°C 등온선으로 검증하고 구간검열을 정량화한다")
p7 = _resolve("outputs/figures/s7_kpdc/s7_alt_comparison.png")
ar7 = R._aspect(p7)
ah = SH - y - 1.02
fw = min(ah*ar7, 6.55)
d7 = R.image(s, p7, ML, y+0.1, fw, ah, valign='top', align='left')
R.caption(s, ML, d7[1]+d7[3]+0.06, fw, FIG(),
          "KPDC Council 유도 ALT vs 코어·실측·물리·학습 모델. 검열 시즌은 하한 화살표로 표기.")
nx = ML + fw + 0.45; nw = CR - nx; yy = y+0.14
R.text(s, nx, yy, nw, 0.24, [[{'t': "검증(대회 KPDC 주활용)", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, nx, yy+0.26, nw, [
    ("자료.", "KPDC Council 일별 지중온도 프로파일 19종에서 0°C 등온선 ALT 유도"),
    ("검열.", "2025 비검열 부분집합 조건부 중앙값 136cm(n=5). 38시즌 중 23건이 우측검열로 구간검열 정량화"),
], size=9.7, gap=5)
yy += 0.26 + 2*0.62 + 0.14
R.rule(s, nx, yy, nw, color=RULE, wt=0.7)
metric_strip(s, nx, yy+0.12, nw, [("136 cm", "유도 중앙값(n=5)"), ("23 / 38", "우측검열 시즌")], vsize=13.0)
yy += 0.95
R.text(s, nx, yy, nw, 0.24, [[{'t': "결론", 'size': 9.5, 'color': TEAL, 'font': SANS_S, 'spc': 0.4}]])
R.bullets(s, nx, yy+0.26, nw, [
    ("정보 병목.", "한계는 모델 아닌 입력 정보량"),
    ("물리 앵커.", "전이는 Stefan이 양축 최선"),
    ("정직한 UQ.", "conformal 보정 90% 구간"),
    ("로드맵.", "InSAR 정보 확충·계절 D(t) 심화"),
], size=9.7, gap=3)

out = "render/permafrost_summary.pptx"
prs.save(out)
print("saved", out, "· slides:", len(prs.slides._sldIdLst))
