# -*- coding: utf-8 -*-
"""build_final.py — 2026 극지 빅데이터·인공지능 활용 경진대회 본선 발표덱 빌드 (v2).

스펙: deck/deck_spec_final.json · 라이브러리: deck/final_lib.py
수치 근거: outputs/report/main.tex (정합 감사 완료본)만 인용.
실행: cd deck && python3 build_final.py → render/permafrost_final.pptx
v2: 표지(팀·로고·부제), 어투 전면 명사형 정리, 방법부 그림 보강(딥러닝 아키텍처·
    증강 공간분포 신설), 결론 재구성(무박스 지표열+핵심 그림), 보충자료 정리.
"""
import os
import final_lib as L
from final_lib import (Inches, Pt, PP_ALIGN, MSO_ANCHOR,
                       ACC, ACC_DK, INK, GRAY, GRAY2, MUTE, HAIR, PANEL, PANEL2,
                       WHITE, NAVY, STEEL, F_X, F_B, F_S, F_M, F_R,
                       ML, MR, CW, CR, SW, SH)

R = lambda p: os.path.join("..", p)


def kv_table(sl, x, y, w, rows, row_h=0.34, size=11.5):
    """key-value 표: 좌 라벨(회색)·우 값(잉크 세미볼드), 헤어라인 행 구분."""
    L.hline(sl, x, y, w, color=GRAY2, lw=1.1)
    yy = y + 0.06
    for lab, val in rows:
        L.text(sl, x, yy, w * 0.62, 0.3, [{"t": lab, "size": size, "color": GRAY2, "font": F_M}])
        L.text(sl, x + w * 0.38, yy, w * 0.62, 0.3,
               [{"t": val, "size": size, "color": INK, "font": F_S}], align=PP_ALIGN.RIGHT)
        yy += row_h
        L.hline(sl, x, yy - 0.045, w, color=HAIR, lw=0.8)
    L.hline(sl, x, yy - 0.045, w, color=GRAY2, lw=1.1)
    return yy
A = lambda p: os.path.join("assets/final", p)

prs = L.new_deck()
PN = [0]


def new_slide(title=None, subtitle=None):
    sl = L.blank(prs)
    PN[0] += 1
    if title:
        L.header(sl, title, subtitle, n=PN[0])
    elif PN[0] > 1:
        L.pagenum(sl, PN[0])
    return sl


# ---------- 크롭 자산 ----------
inv_map = L.crop_asset(A("inv_world_boost.png"), (0, 0, 2078, 750), "inv_map_b.png")
inv_tab = L.crop_asset(A("inv_world_boost.png"), (0, 800, 2078, 1370), "inv_tab_b.png")
s12_top = L.crop_asset(R("outputs/figures/s12_hybrid/s12_hybrid_summary.png"), (0, 0, 2905, 1050), "s12_top.png")
s12_bot = L.crop_asset(R("outputs/figures/s12_hybrid/s12_hybrid_summary.png"), (0, 1050, 2905, 1959), "s12_bot.png")
def _s11_panel_b():
    """(b) 패널 크롭. 원본에서 세로축 라벨과 겹쳐 그려진 '목표 90%' 잔여 조각을 흰색으로 정리."""
    from PIL import Image, ImageDraw
    out = os.path.join(L.CROP_DIR, "s11_b7.png")
    if not os.path.exists(out):
        os.makedirs(L.CROP_DIR, exist_ok=True)
        im = Image.open(R("outputs/figures/s11_uq/s11_calibration_curve.png")).crop((1235, 90, 2170, 1085))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 16, 995), fill=(255, 255, 255))      # 세로축 라벨 절단 조각
        d.rectangle((14, 213, 228, 263), fill=(255, 255, 255))  # '목표 90%' 잔여 라벨
        # 패치로 끊긴 세로축 스파인 복원 (주변 픽셀에서 스파인 x 탐색)
        import numpy as np
        arr = np.asarray(im.convert("L"))
        band = arr[400:600, 90:170].mean(axis=0)
        xs = int(band.argmin()) + 90
        shade = int(band.min())
        d.rectangle((xs - 1, 210, xs + 1, 266), fill=(shade, shade, shade))
        im.save(out)
    return out

s11_b = _s11_panel_b()
lena_maps = L.crop_asset(R("outputs/figures/s3_aug/lena_aug_mapping.png"), (0, 150, 3740, 1309), "lena_maps.png")
uq_ab = L.crop_asset(R("outputs/maps/alt_uncertainty_aoa_map.png"), (0, 0, 1676, 668), "uq_ab.png")

# ============================================================ 1 표지
sl = new_slide()
L.text(sl, 1.0, 1.72, SW - 2.0, 1.3,
       [[{"t": "물리경험식 유사라벨 증강과 기계학습을 이용한", "size": 29, "color": INK, "font": F_X}],
        [{"t": "영구동토 활동층 두께 예측과 지역 간 전이", "size": 29, "color": INK, "font": F_X}]],
       align=PP_ALIGN.CENTER, line_spacing=1.22)
L.text(sl, 1.0, 3.10, SW - 2.0, 0.45,
       [{"t": "희소 관측 조건의 광역 예측을 위한 물리 기반 라벨 증강과 독립 관측 결합",
         "size": 15, "color": ACC, "font": F_S}],
       align=PP_ALIGN.CENTER)
L.text(sl, 1.0, 4.02, SW - 2.0, 0.45,
       [[{"t": "백승원", "size": 15, "color": INK, "font": F_S, "underline": True},
         {"t": "1", "size": 15, "color": INK, "font": F_S, "sup": True},
         {"t": "   ·   정현수", "size": 15, "color": INK, "font": F_S},
         {"t": "2", "size": 15, "color": INK, "font": F_S, "sup": True},
         {"t": "   ·   엄태건", "size": 15, "color": INK, "font": F_S},
         {"t": "3", "size": 15, "color": INK, "font": F_S, "sup": True}]],
       align=PP_ALIGN.CENTER)
L.text(sl, 1.0, 4.52, SW - 2.0, 0.6,
       [[{"t": "1) 서울대학교 에너지시스템공학부 석박사통합과정    2) 서울대학교 에너지시스템공학부 석사과정    "
              "3) 서울대학교 에너지자원공학과 학부과정", "size": 10.5, "color": GRAY, "font": F_M}],
        [{"t": "팀 ALT_ctrl", "size": 10.5, "color": GRAY, "font": F_M}]],
       align=PP_ALIGN.CENTER, line_spacing=1.35, space_after=3)
# 하단: 로고(좌) + 대회 정보(우)
logo_y = 6.42
L.image(sl, A("logos/sail.png"), 0.52, logo_y + 0.03, h=0.52)
L.image(sl, A("logos/snu.png"), 2.05, logo_y - 0.04, h=0.66)
L.image(sl, A("logos/kopri.png"), 2.90, logo_y + 0.09, h=0.42)
L.text(sl, 5.0, logo_y + 0.02, SW - 5.0 - MR, 0.6,
       [[{"t": "2026 극지 빅데이터·인공지능 활용 경진대회 · 본선 발표", "size": 12, "color": GRAY2, "font": F_S}],
        [{"t": "2026. 08. 28", "size": 11, "color": GRAY, "font": F_M}]],
       align=PP_ALIGN.RIGHT, line_spacing=1.3, space_after=2)

# ============================================================ 2 배경 및 필요성
sl = new_slide("배경 및 필요성", "영구동토 탄소 되먹임과 극지 인프라 위험의 핵심 지표, 활동층 두께")
w, h = L.image(sl, A("fig_concept_alt.png"), ML, 1.42, w=6.35)
L.text(sl, ML + 0.05, 1.42 + h + 0.12, w, 0.7,
       [[{"t": "활동층 두께(ALT)  ", "size": 12, "color": INK, "font": F_S},
         {"t": "여름 최대 융해 깊이. 지온 프로파일이 0°C를 지나는 깊이로도 유도되며, "
              "이 관계가 본 연구의 지중온도 유도 라벨과 3차원 온도장의 융해면 산출에 쓰이는 기반",
          "size": 11.3, "color": GRAY2, "font": F_M}]],
       line_spacing=1.35)
x = 7.25; wcol = CR - x
L.bullets(sl, x, 1.48, wcol,
          [("탄소 되먹임: 북반구 영구동토의 유기탄소 약 1,300 Pg 격리",
            "활동층이 깊어질수록 유기탄소가 미생물 분해에 노출되어 온실기체로 방출, 온난화를 재촉진 (Hugelius et al. 2014; Schuur et al. 2015)"),
           ("온난화 증폭: 1979년 이래 북극 온난화 속도는 지구 평균의 약 4배",
            "전 지구 관측망에서 영구동토 지중온도의 지속 상승 확인 (Rantanen et al. 2022; Biskaborn et al. 2019)"),
           ("인프라 위험: 융해·재동결에 따른 지반 침하와 열카르스트",
            "금세기 중반까지 극지 기반시설 상당수가 고위험 구역에 놓일 전망 (Hjort et al. 2018)"),
           ("감시 격차: 실측 ALT는 탐침·시추에 의존하는 지점 관측",
            "광역·연속 감시를 위해서는 공변량 기반 예측 모델이 필수 (Brown et al. 2000)")],
          size=13.5, gap=17, sub_size=11.3)
L.takeaway(sl, "ALT의 공간 분포와 시간 변화 정량화는 극지 환경 감시의 핵심 과제")

# ============================================================ 3 문제 정의
sl = new_slide("문제 정의: 라벨 희소", "광역 조밀 공변량 대 지점 희소 라벨의 구조적 비대칭")
w, h = L.image(sl, A("fig_label_gap.png"), ML, 1.42, w=6.35)
L.text(sl, ML + 0.05, 1.42 + h + 0.18, 6.3, 1.0,
       [[{"t": "탐침·시추 실측의 한계  ", "size": 12.5, "color": INK, "font": F_S},
         {"t": "라벨은 소수 장기 감시 지점에 군집해 광역 학습에 필요한 양과 공간 분포를 모두 충족하지 못하며 (Brown et al. 2000), "
              "라벨 희소를 얼마나 완화하는가가 광역 ALT 예측 성능을 좌우",
          "size": 11.5, "color": GRAY2, "font": F_M}]],
       line_spacing=1.4)
x = 7.25; wcol = CR - x
L.section_label(sl, x, 1.42, "기존 연구의 세 한계")
items = [("1", "평가되지 않는 시뮬레이션", "Jafarov et al. (2012)",
          "순방향 물리 시뮬레이션은 광역 구동이 가능하나 산출이 관측 라벨로 직접 채점되지 않아 실측 대비 정확도 확인 불가"),
         ("2", "신뢰 범위 없는 지도", "Obu et al. (2019) · Ran et al. (2022)",
          "광역 지도 제품은 예측구간 없는 단일 값만 제공, 지도 사용자가 신뢰 범위를 판단할 근거 부재"),
         ("3", "물리 주입의 미확장", "Aljubran & Horne (2024) · Liu et al. (2023)",
          "물리 제약을 신경망 구조·손실에 주입하는 연구는 그 물리식을 라벨 없는 지역으로 학습 신호를 넓히는 데 활용하지 않음")]
yy = 1.92
for no, t, ref, d in items:
    L.text(sl, x, yy, 0.5, 0.5, [{"t": no, "size": 23, "color": ACC, "font": F_X}])
    L.text(sl, x + 0.55, yy + 0.02, wcol - 0.55, 0.35,
           [{"t": t + "   ", "size": 14, "color": INK, "font": F_S},
            {"t": ref, "size": 10.5, "color": MUTE, "font": F_M}])
    L.text(sl, x + 0.55, yy + 0.42, wcol - 0.55, 0.8, [{"t": d, "size": 12, "color": GRAY2, "font": F_M}],
           line_spacing=1.35)
    yy += 1.62
L.takeaway(sl, "연구 목적: 물리경험식으로 라벨을 생성·증강해 관측 없는 지역까지 신뢰 범위를 갖춘 ALT 예측을 확장", accent=True)

# ============================================================ 4 연구 개요 (전면 그림)
sl = new_slide()
L.image(sl, R("outputs/figures/00_overview/report_overview.png"), (SW - 12.0) / 2, 0.33, w=12.0)

# ============================================================ 5 사용 데이터
sl = new_slide("사용 데이터", "범북극 3출처 ALT 라벨 · 공변량 34종 · KPDC 현장 자료")
mw = 10.9
w, h = L.image(sl, inv_map, (SW - mw) / 2, 1.18, w=mw)
y0 = 1.18 + h + 0.16
x0 = (SW - mw) / 2
colw = (mw - 0.8) / 3
x = x0
L.section_label(sl, x, y0, "ALT 라벨 3출처")
L.text(sl, x, y0 + 0.34, colw, 1.2,
       [[{"t": "실측  ", "size": 11, "color": INK, "font": F_S},
         {"t": "알래스카 13,606 · 레나델타 3,037 · 캐나다 742", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "지중온도 유도  ", "size": 11, "color": INK, "font": F_S},
         {"t": "GTN-P 37 · KPDC 콘슬 19", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "물리식 유도  ", "size": 11, "color": INK, "font": F_S},
         {"t": "Stefan 산출, 증강 전용", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "지도 범례 = 집계 전 원자료 분포 (본문 = 셀 집계 후)",
          "size": 9.5, "color": MUTE, "font": F_M}]],
       line_spacing=1.24, space_after=2)
x += colw + 0.4
L.vline(sl, x - 0.2, y0 + 0.05, 1.30)
L.section_label(sl, x, y0, "공변량 34종")
L.text(sl, x, y0 + 0.34, colw, 1.2,
       [[{"t": "지형 6 · 기후 8 · 토양 9 · SAR 8 · CCI 2 · 표시 1", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "ERA5-Land 기후 (Muñoz-Sabater et al. 2021)", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "집합  ", "size": 11, "color": INK, "font": F_S},
         {"t": "전체 34 · 전이 공통 25 · 지형기후 14", "size": 11, "color": GRAY2, "font": F_M}]],
       line_spacing=1.24, space_after=2)
x += colw + 0.4
L.vline(sl, x - 0.2, y0 + 0.05, 1.30)
L.section_label(sl, x, y0, "KPDC 현장 자료")
L.text(sl, x, y0 + 0.34, colw, 1.2,
       [[{"t": "콘슬 지온 19지점(연도별 38) · 코어 18 · AWS", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "공변량 대조·현장 검증에 활용", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "KOPRI-KPDC-00002125", "size": 11, "color": ACC_DK, "font": F_S}]],
       line_spacing=1.24, space_after=2)
L.takeaway(sl, "통합 스키마 59,184행을 고유 위치 17,423셀로 정렬해 하나의 표본 공간을 구성")

# ============================================================ 6 전체 워크플로
sl = new_slide("전체 워크플로", "3출처 라벨의 검증 게이트 · 유사라벨 증강 · 물리 결합 · 4종 산출물")
L.image(sl, A("fig_workflow.png"), (SW - 11.7) / 2, 1.22, w=11.7)
L.takeaway(sl, "물리식을 신경망의 제약 조건이 아니라 라벨 생성기·예측 앵커로 활용", accent=True)

# ============================================================ 7 물리경험식의 선택
sl = new_slide("물리경험식의 선정", "산출식 5종 비교, 계수 하나만 학습 폴드에서 적합")
x = ML; wcol = 5.85
L.text(sl, x + 0.25, 1.48, wcol - 0.5, 0.5,
       [{"t": "ALT(x) = E · √TDD(x)", "size": 19, "color": ACC, "font": F_X},
        {"t": "    (Stefan 1891)", "size": 10.5, "color": MUTE, "font": F_M}])
L.text(sl, x + 0.25, 2.00, wcol - 0.5, 0.65,
       [[{"t": "TDD = Σ max(T̄ₘ, 0)·dₘ,   T̄ₘ = 월평균 기온 · dₘ = 월 일수", "size": 10.3, "color": GRAY, "font": F_M}],
        [{"t": "E = 열·수분 물성 집약 계수, 학습 폴드에서만 적합", "size": 10.3, "color": GRAY, "font": F_M}]],
       line_spacing=1.35, space_after=2)
rows = [["물리식 (그림과 같은 순서)", "지역 내", "전이"],
        [{"t": "Stefan (√TDD)", "font": F_S}, {"t": "14.56", "font": F_S}, {"t": "22.24", "font": F_S}],
        ["edaphic (토양물성)", "40.96", "62.68"],
        ["edaphic + TTOP 판정", "31.10", "59.39"],
        ["Kudryavtsev", "25.29", "39.63"],
        ["열전도도 보정", "46.30", "69.43"]]
L.mini_table(sl, x, 3.00, wcol, rows, col_w=[3.0, 1.4, 1.45], size=12, row_h=0.33)
L.text(sl, x, 5.36, wcol, 0.6,
       [[{"t": "RMSE(cm) · 지역 내 = 알래스카 공간블록 CV · 전이 = 세 지역 홀드아웃 비가중평균",
          "size": 10.2, "color": MUTE, "font": F_M}],
        [{"t": "TTOP = 영구동토 상부면 온도 판정 (Riseborough et al. 2008) · 평균 예측 기준선 17.4 cm",
          "size": 10.2, "color": MUTE, "font": F_M}]],
       line_spacing=1.28, space_after=2)
L.bullets(sl, x, 5.86, wcol,
          [("정교한 식일수록 미측정 물성 가정이 늘어 광역에서 불리", "희소 관측 조건의 단순식 우위는 선행 연구와 일치 (Gautam et al. 2025)"),
           ("전 지역에서 평균 예측보다 정확한 식은 Stefan이 유일", None)], size=12.5, gap=5, sub_size=10.5)
L.image(sl, R("outputs/figures/s2_physics/physics_loro_bars.png"), 6.85, 1.25, h=5.30)
L.takeaway(sl, "증강 공급원으로 Stefan 선정, 정확한 물리식이 증강 효과의 전제")

# ============================================================ 8 증강·대조 실험 설계
sl = new_slide("증강·대조 실험 설계", "동일 셀·동일 개수·동일 난수, 라벨 내용만 다른 세 조건")
L.image(sl, A("fig_aug_design.png"), (SW - 11.9) / 2, 1.32, w=11.9)
L.takeaway(sl, "상수 대조 대비 개선분만을 물리 정보의 순가치로 정의")

# ============================================================ 9 예측 모델군 아키텍처 (신설)
sl = new_slide("예측 모델군 아키텍처", "표 형식 딥러닝 3종과 물리 잔차 결합의 구조")
L.image(sl, A("fig_dl_arch.png"), (SW - 11.6) / 2, 1.26, w=11.6)
L.takeaway(sl, "순수 학습 모델과 물리 결합 모델을 동일 검증 규약에서 비교해 구성을 선택")

# ============================================================ 10 검증 설계
sl = new_slide("검증 설계", "공간 자기상관에 의한 낙관 편향을 막는 평가 규약")
w, h = L.image(sl, A("folds_boost.png"), (SW - 11.6) / 2, 1.22, w=11.6)
rows = [["조건", "대상 지역 정보", "평가", "대응 결과"],
        [{"t": "지역 내", "font": F_S}, "실측 라벨까지 사용", "알래스카 0.5° 공간블록 교차검증 (74블록)", "결과 2"],
        [{"t": "공변량만", "font": F_S}, "공변량만 사용, 실측 없음", "대상 지역 블록 이분: 절반 학습, 절반 실측 평가", "결과 1"],
        [{"t": "정보 없음", "font": F_S}, "공변량·라벨 모두 미사용", "지역 단위 홀드아웃 (LORO)", "결과 3"]]
L.mini_table(sl, ML + 0.3, 1.22 + h + 0.18, CW - 0.6, rows,
             col_w=[1.5, 3.1, 5.3, 1.8], size=11.5, row_h=0.30,
             align_cols=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
L.takeaway(sl, "같은 블록은 같은 폴드에 배정, 조건이 다른 수치는 같은 축에서 비교하지 않음")

# ============================================================ 11 결과 1 · 증강 효과
sl = new_slide("결과 1 · 물리 유사라벨 증강", "공변량만 조건 · 전이 공통 25종 입력 · CatBoost · r = 10")
L.image(sl, R("outputs/figures/s3_aug/aug_response_curves.png"), ML, 1.24, h=5.48)
x = 7.0
w, h = L.image(sl, R("outputs/figures/s3_aug/physics_net_value.png"), x, 1.24, h=4.42)
rows = [["대상", "증강 없음", "Stefan", "대조", "부정확"],
        [{"t": "레나델타", "font": F_S}, "21.9", {"t": "16.5", "color": ACC_DK, "font": F_X}, "18.2", "31.7"],
        [{"t": "캐나다", "font": F_S}, "34.0", {"t": "31.8", "color": ACC_DK, "font": F_X}, "42.0", "58.7"]]
L.mini_table(sl, x, 5.86, w, rows, col_w=[w * 0.24, w * 0.19, w * 0.19, w * 0.19, w * 0.19],
             size=11.5, row_h=0.28)
L.takeaway(sl, "정확한 물리 유사라벨은 개선, 부정확한 식은 악화. 캐나다에서 상수 대조 대비 물리 라벨의 개선 폭(순가치)은 약 10 cm")

# ============================================================ 12 결과 1 · 증강 효과의 공간 분포 (신설)
sl = new_slide("결과 1 · 증강이 만든 변화", "레나델타 전이 예측: 공간 분포와 오차·편향의 정량 감소 (CatBoost)")
w, h = L.image(sl, A("fig_aug_spatial.png"), (SW - 11.6) / 2, 1.28, w=11.6)
y0 = 1.32 + h + 0.20
half = (CW - 0.5) / 2
L.bullets(sl, ML, y0, half,
          [("물리식이 대상 지역 정보를 공급해 과대예측 편향이 +16.2에서 +8.3 cm로 절반",
            None)], size=12.5, gap=5)
L.bullets(sl, ML + half + 0.5, y0, half,
          [("관측이 없는 델타 내부까지 일관된 방향의 하향 조정, 지역 전반의 수준 보정",
            None)], size=12.5, gap=5)
L.takeaway(sl, "증강의 이득은 특정 지점이 아니라 대상 지역 전반의 편향 보정에서 발생")

# ============================================================ 13 결과 2 · 지역 내 정확도
sl = new_slide("결과 2 · 지역 내 정확도", "알래스카 공간블록 교차검증, 실측 라벨 학습")
w, h = L.image(sl, A("fig_model_bars.png"), ML, 1.32, w=6.1)
y0 = 1.32 + h + 0.28
L.hline(sl, ML, y0, 6.3, color=GRAY2, lw=1.1)
L.text(sl, ML + 0.02, y0 + 0.12, 6.3, 0.35,
       [{"t": "물리 잔차 결합   ", "size": 13, "color": ACC, "font": F_X},
        {"t": "ŷ = Ê·√TDD + λ·g(c)", "size": 13, "color": INK, "font": F_S}])
L.text(sl, ML + 0.02, y0 + 0.52, 6.3, 0.5,
       [{"t": "13.33 cm", "size": 22, "color": INK, "font": F_X},
        {"t": "   (λ = 0.75, 입력 25종)", "size": 12, "color": GRAY, "font": F_M}])
L.text(sl, ML + 0.02, y0 + 1.02, 6.35, 0.6,
       [[{"t": "물리식 단독 14.46(최소제곱 계수) · 그림 기준선 14.56(중앙값비 계수) · 신경망 최고 14.37 대비 최저",
          "size": 10.4, "color": GRAY2, "font": F_M}],
        [{"t": "λ는 조건별 검증에서 선택: 지역 내 0.75 · 전이 0.25",
          "size": 10.4, "color": GRAY2, "font": F_M}]],
       line_spacing=1.35, space_after=2)
L.image(sl, R("outputs/figures/s4_residual/s4_indomain_vs_transfer.png"), 7.3, 1.30, h=5.10)
L.takeaway(sl, "물리가 설명하는 부분과 데이터가 설명하는 부분의 상호 보완으로 지역 내 최저 오차 달성")

# ============================================================ 14 결과 3 · 전이의 병목
sl = new_slide("결과 3 · 지역 간 전이의 병목", "정보 없음 조건, 세 지역 홀드아웃 비가중평균")
x = ML; wcol = 3.55
L.bullets(sl, x, 1.42, wcol,
          [("순수 기계학습은 34–38 cm로 오차 급증", "학습 지역의 공변량-ALT 관계가 대상 지역에서 미유지"),
           ("공간보간도 29–51 cm로 부진", "좌표만 쓰는 크리깅·역거리가중은 자기상관 구조가 다른 지역으로 이전 불가"),
           ("Stefan 물리식은 21.26 cm 유지", "기후에서 융해 깊이를 직접 산출, 관계가 지역 불변 (식 선별 단계의 22.24와는 평가 셀 기준이 다름)"),
           ("구조 정교화 세 갈래 모두 물리식 미돌파", "물리 잔차 결합 21.83 · 다중충실도(자료원별 신뢰도 반영) 21.84 · 혼합(물리식들을 공변량으로 가중 결합) 29.22"),
           ("유사라벨 증강도 21.28 cm로 동률", "물리식이 이미 가진 정보를 옮겨 심는 경로")],
          size=12.5, gap=10, sub_size=10.5)
L.image(sl, R("outputs/figures/02_evaluation/transfer_loro_summary.png"), 4.85, 1.20, h=5.58)
L.takeaway(sl, "구조 정교화만으로는 물리식 단독을 넘지 못함, 물리식이 갖지 못한 정보가 필요")

# ============================================================ 15 결과 3 · 독립 관측 결합
sl = new_slide("결과 3 · 독립 관측의 결합", "산출 원리가 다른 위성 CCI ALT 제품(Westermann et al. 2024)과 물리식 앵커의 결합")
w, h = L.image(sl, s12_top, (SW - 10.2) / 2, 1.20, w=10.2)
rows = [["방법 (레나델타·캐나다 평균 RMSE, cm)", "대상 공변량 사용", "대상 정보 없음"],
        ["물리식 단독", "22.91", "24.11"],
        ["물리식 + 위성 제품 (앵커 평균)", "21.68", "23.27"],
        [{"t": "+ 잔차 학습 (λ = 0.25)", "font": F_S},
         {"t": "21.32", "color": ACC_DK, "font": F_X}, {"t": "22.92", "color": ACC_DK, "font": F_X}]]
L.mini_table(sl, ML + 1.2, 1.20 + h + 0.14, CW - 2.4, rows, col_w=[5.0, 2.4, 2.5],
             size=11.5, row_h=0.29)
yft = 1.20 + h + 0.14 + 4 * 0.29 + 0.24
L.text(sl, ML + 1.2, yft, CW - 2.4, 0.5,
       [[{"t": "그림·표 모두 위성 제품이 유효한 같은 셀 집합에서 평가한 값",
          "size": 9.5, "color": MUTE, "font": F_M}],
        [{"t": "앵커 = 대상 라벨 없이 계산 가능한 기준 예측 · 앵커 평균 = 물리식·위성 예측의 단순 평균 · "
              "+ 잔차 학습 = 앵커 + λ·g(c), λ = 0.25 · 위성 제품 = ESA CCI ALT",
          "size": 9.5, "color": MUTE, "font": F_M}]],
       line_spacing=1.3, space_after=2)
L.takeaway(sl, "두 지역·두 조건 네 조합 모두에서 오차 감소 (24.11 → 22.92 cm)")

# ============================================================ 16 결과 3 · 편향 상쇄 메커니즘
sl = new_slide("결과 3 · 편향 상쇄 메커니즘", "전이 오차의 편향·산포 분해")
w, h = L.image(sl, s12_bot, (SW - 11.9) / 2, 1.25, w=11.9)
y0 = 1.25 + h + 0.20
half = (CW - 0.5) / 2
L.bullets(sl, ML, y0, half,
          [("레나델타: 물리식 +5.8 cm 과대, 위성 제품 -8.8 cm 과소로 부호 반대, 결합 시 편향 직접 상쇄", None),
           ("캐나다: 부호가 같아도 오차 독립으로 산포(편향을 뺀 오차 성분) 26.4 → 24.9 cm 감소", None),
           ("편향의 부호·크기는 지역 물성에 의존, 보편 규칙으로 일반화하지 않음", None)],
          size=12.5, gap=8)
L.bullets(sl, ML + half + 0.5, y0, half,
          [("잔차 가중 λ = 0.25에서 최소, 과대 반영 시 학습 지역 관계의 과전이", None),
           ("개선 0.85 cm은 95% CI가 0을 포함, 확정 효과가 아닌 일관된 경향으로 보고", None),
           ("타 지역 적용 시 소수 현장 관측이나 지역 특성으로 결합 가중을 보정하는 절차 제안", None)],
          size=12.5, gap=8)
L.takeaway(sl, "전이 개선은 모델 구조가 아니라 정보원의 다양성에서 발생", accent=True)

# ============================================================ 17 결과 4 · 광역 고해상 지도
sl = new_slide("결과 4 · 광역 고해상 ALT 지도", "관측이 없는 영역까지 연속 ALT 장 산출")
w, h = L.image(sl, R("outputs/maps/alt_prediction_hires.png"), ML, 1.35, w=6.45)
L.text(sl, ML + 0.05, 1.35 + h + 0.14, w, 0.7,
       [[{"t": "0.02° 격자 · CatBoost(기후 8·토양 9) · 검은 점 = 실측 203개 위치",
          "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "영구동토 범위(다년평균 연평균기온 0 °C 미만, 육지의 84.8%)만 산출",
          "size": 11, "color": GRAY2, "font": F_M}]],
       line_spacing=1.3, space_after=2)
x = 7.2
w2, h2 = L.image(sl, R("outputs/maps/alt_downscale_demo.png"), x, 1.35, w=5.6)
L.text(sl, x, 1.35 + h2 + 0.26, 5.75, 2.2,
       [[{"t": "243 m 다운스케일  ", "size": 13.5, "color": ACC_DK, "font": F_X},
         {"t": "30 m 수치표고모형 지형 공변량 추가", "size": 12.5, "color": INK, "font": F_S}],
        [{"t": "국소 고주파 성분(표준편차) 0.49 cm에서 3.01 cm로 6배 확대", "size": 12, "color": GRAY2, "font": F_M}],
        [{"t": "하천망과 사면을 따르는 융해 구조가 드러남", "size": 12, "color": GRAY2, "font": F_M}],
        [{"t": "셀 평균 검증 체계상 이 세부 구조는 검증된 향상이 아니라", "size": 12, "color": GRAY2, "font": F_M}],
        [{"t": "지형이 시사하는 공간 변동으로 해석", "size": 12, "color": GRAY2, "font": F_M}]],
       line_spacing=1.32, space_after=5)
L.takeaway(sl, "지도의 실효 해상도는 격자 간격이 아니라 입력 자료가 담은 정보의 규모가 결정")

# ============================================================ 18 결과 5 · 연별 ALT 지도
sl = new_slide("결과 5 · 연별 ALT 지도 (2010–2024)", "그해 ERA5-Land 기후 입력, (위치, 연도) 패널 99,931행 학습 · 지도 생성 모델 = CatBoost")
L.image(sl, A("annual_soft.png"), (SW - 11.2) / 2, 1.30, w=11.2)
L.takeaway(sl, "연도 홀드아웃: 물리 잔차 결합 14.06 · CatBoost 14.24 · 물리식 14.97 cm, 2019년 전역 깊고 2013년 얕음")

# ============================================================ 19 결과 6 · 표층 3D 지중온도장
sl = new_slide("결과 6 · 표층 3차원 지중온도장", "깊이 단조 제약 부스팅, 시추공 깊이별 연최대 지온 764행 학습")
L.image(sl, R("outputs/figures/s10_shallow3d/s10_depth_slices.png"), ML, 1.28, h=5.22)
x = 7.45; wcol = CR - x
L.text(sl, x, 1.50, wcol, 0.3,
       [{"t": "검증 요약  ", "size": 12.5, "color": ACC, "font": F_X},
        {"t": "사이트 블록 홀드아웃", "size": 10.5, "color": GRAY, "font": F_M}])
kv_table(sl, x, 1.86, wcol,
         [("온도장 재현 RMSE", "2.66 °C"),
          ("온도장 재현 R²", "0.47"),
          ("융해면 유도 오차", "40.7 cm")], row_h=0.33, size=11.5)
L.bullets(sl, x, 3.20, wcol,
          [("깊이별 연최대 지온을 회귀해 융해 최성기의 열 상태를 표현", None),
           ("남북 열구배와 0 °C 등온면의 공간 구조를 깊이별로 기술", "색 전환 경계 = 동결·융해 경계"),
           ("깊이 단조 제약으로 물리적으로 타당한 프로파일 확보", None),
           ("융해면은 온도 프로파일의 0 °C 교차 깊이를 선형보간해 산출", None),
           ("온도장 유도 융해면은 오차 40.7 cm로 직접 예측 대비 부정확", "열구조의 공간 형태 제시 용도로 한정")],
          size=13, gap=13, sub_size=11)
L.takeaway(sl, "활동층을 단일 시점 2차원 지도가 아니라 시간·깊이를 포함한 상태로 기술")

# ============================================================ 20 결과 7 · 보정 불확실성
sl = new_slide("결과 7 · 보정 예측구간과 적용가능 영역", "지도 신뢰 범위의 정량 표기")
w, h = L.image(sl, uq_ab, ML, 1.30, w=8.9)
L.text(sl, ML + 0.05, 1.30 + h + 0.14, w, 0.5,
       [{"t": "(a) 보정된 90% 예측구간 폭 · 어두울수록 불확실  ·  (b) 적용가능 영역(AOA) 판정 · 점 = 실측 위치",
         "size": 10.5, "color": MUTE, "font": F_M}])
L.hline(sl, ML + 0.05, 1.30 + h + 0.62, w - 0.4, color=HAIR, lw=0.9)
L.text(sl, ML + 0.05, 1.30 + h + 0.78, w - 0.2, 1.2,
       [[{"t": "지도 읽기  ", "size": 12, "color": ACC, "font": F_X},
         {"t": "(a)에서 어두운 셀일수록 예측 불확실성이 크고, (b)에서 짙은 셀은 학습 환경과 멀어",
          "size": 11.5, "color": GRAY2, "font": F_M}],
        [{"t": "지역 간 전이 시 적용가능 영역 밖으로 판정된다. 두 지도를 함께 보면 어느 지역의 값을",
          "size": 11.5, "color": GRAY2, "font": F_M}],
        [{"t": "신뢰할 수 있는지 지도만으로 판단할 수 있다.", "size": 11.5, "color": GRAY2, "font": F_M}]],
       line_spacing=1.45, space_after=3)
x = 9.6; wcol = CR - x
L.text(sl, x, 1.50, wcol, 0.3,
       [{"t": "보정·판정 요약  ", "size": 12.5, "color": ACC, "font": F_X},
        {"t": "알래스카 공간블록 CV", "size": 10.5, "color": GRAY, "font": F_M}])
kv_table(sl, x, 1.86, wcol,
         [("커버리지 (목표 90%)", "93.4%"),
          ("구간 폭", "14.7 → 53.6 cm"),
          ("구간 보정", "등순응 CQR"),
          ("외삽 판정", "AOA 동봉")], row_h=0.33, size=11.5)
L.bullets(sl, x, 3.62, wcol,
          [("분위 회귀 구간을 검증 잔차로 재보정해 목표 커버리지 확보", None),
           ("구간 폭 보정 전 14.7 → 후 53.6 cm(평균)", "공변량 정보량 반영 · 새 정보원 추가 시 축소"),
           ("AOA = 학습 분포와의 환경 거리 판정", "외삽 영역을 지도에 명시"),
           ("CQR: Romano et al. (2019)", "AOA: Meyer & Pebesma (2021)")],
          size=12, gap=9, sub_size=10.5)
L.takeaway(sl, "보정된 90% 예측구간과 적용가능 영역을 지도에 동봉, 값을 신뢰할 수 있는 범위를 명시")

# ============================================================ 21 KPDC 현장 검증
sl = new_slide("KPDC 현장 검증", "수어드반도 콘슬 사이트 (KOPRI-KPDC-00002125)")
w, h = L.image(sl, R("outputs/figures/s7_kpdc/s7_alt_comparison.png"), ML, 1.35, w=8.55)
L.caption(sl, ML + 0.05, 1.35 + h + 0.1, w,
          "같은 사이트에서도 관측 방식에 따라 35–136 cm로 갈리는 ALT")
x = 9.35; wcol = CR - x
L.text(sl, x, 1.42, wcol, 0.3,
       [{"t": "ERA5-Land 공변량 정합", "size": 12.5, "color": ACC, "font": F_X}])
kv_table(sl, x, 1.76, wcol,
         [("√TDD 상대 편향", "약 1.5%"),
          ("평균 대비 RMSE", "약 2.8%"),
          ("콘슬 2019 제외 시", "약 0.1%")], row_h=0.32, size=11.3)
L.text(sl, x, 2.88, wcol, 0.5,
       [{"t": "관측월을 정합시킨 4개 관측연 기준", "size": 10, "color": MUTE, "font": F_M}])
L.text(sl, x, 3.42, wcol, 0.3,
       [{"t": "관측 방식별 ALT", "size": 12.5, "color": ACC, "font": F_X}])
kv_table(sl, x, 3.76, wcol,
         [("탐침", "35.0 cm"),
          ("코어", "81.3 cm"),
          ("심부 프로파일 유도", "136.4 cm"),
          ("Stefan 다년평균", "59.6 cm")], row_h=0.32, size=11.3)
L.text(sl, x, 5.24, wcol, 0.8,
       [[{"t": "평가는 기준 관측의 선택에 따라 달라짐", "size": 11, "color": GRAY2, "font": F_M}],
        [{"t": "지점 관측 자체의 대표성 한계를 시사 (Nelson et al. 1998)",
          "size": 11, "color": GRAY2, "font": F_M}]],
       line_spacing=1.4, space_after=3)
L.takeaway(sl, "공변량 기반에 큰 계통 편차 없음, 지점 관측의 대표성 한계가 셀 단위(면적 평균) 검증의 타당성을 뒷받침")

# ============================================================ 22 결론 및 시사점
sl = new_slide("결론 및 시사점", "라벨 희소 완화: 물리 유사라벨 증강과 정보원 다양화")
# 상단: 무박스 지표 열 (얇은 세로 구분선)
stats = [("라벨 증강", "오차 -10 cm", "물리 유사라벨의 순가치 (캐나다 전이)"),
         ("예측 정확도", "13.33 cm", "지역 내 ALT 오차 최저 (물리 잔차 결합)"),
         ("전이 오차", "24.11 → 22.92 cm", "위성 관측 결합, 네 조합 모두 개선"),
         ("신뢰 범위", "93.4%", "90% 예측구간 커버리지 (등순응 보정 + AOA)")]
sw4 = CW / 4
L.hline(sl, ML, 1.30, CW, color=GRAY2, lw=1.1)
for i, (tag, v, nte) in enumerate(stats):
    x = ML + i * sw4
    L.text(sl, x + 0.05, 1.40, sw4 - 0.3, 0.3, [{"t": tag, "size": 11.5, "color": ACC, "font": F_X}])
    L.text(sl, x + 0.05, 1.70, sw4 - 0.3, 0.42, [{"t": v, "size": 18.5, "color": INK, "font": F_X}])
    L.text(sl, x + 0.05, 2.16, sw4 - 0.3, 0.32, [{"t": nte, "size": 9.8, "color": GRAY, "font": F_M}])
L.hline(sl, ML, 2.52, CW, color=GRAY2, lw=1.1)
# 좌: 핵심 성과 · 우: 한계 + 핵심 그림
xl = ML; wl = 6.5
L.section_label(sl, xl, 2.80, "핵심 성과")
L.bullets(sl, xl, 3.12, wl,
          [("물리식의 역할 재정의", "제약 조건이 아니라 라벨 생성기·예측 앵커로 활용, 이득을 상수 대조로 분리 입증"),
           ("전이 개선의 원천은 정보원의 다양성", "구조 탐색 대신 산출 원리가 다른 위성 관측을 결합, 후속 연구의 자원 배분 근거"),
           ("시간·깊이로 확장된 산출물", "연별 지도(2010–2024)와 표층 3차원 온도장으로 활동층을 시간·깊이를 포함한 상태로 기술"),
           ("신뢰 범위의 동봉", "보정 예측구간과 AOA 표기로 지도 사용 가능 범위 명시"),
           ("남은 과제", "지점 대표성 하한과 결합 유의성은 면적 평균 검증·대상 지역 확장으로 보완 예정")],
          size=13, gap=10, sub_size=11.2)
xr = xl + wl + 0.5; wr = CR - xr
L.section_label(sl, xr, 2.80, "대표 산출물")
w3, h3 = L.image(sl, R("outputs/maps/alt_prediction_hires.png"), xr + (wr - 4.75) / 2, 3.20, w=4.75)
L.caption(sl, xr, 3.20 + h3 + 0.10, wr, "알래스카 고해상 ALT 예측장 (0.02° · CatBoost)",
          align=PP_ALIGN.CENTER)
L.takeaway(sl, "물리경험식으로 라벨을 만들고 독립 관측으로 편향을 상쇄해, 관측 없는 지역까지 신뢰 범위를 갖춘 ALT 지도를 제시", accent=True)

# ============================================================ 23 마무리 · Q&A
sl = new_slide()
L.text(sl, 1.0, 2.55, SW - 2.0, 0.7, [{"t": "감사합니다", "size": 34, "color": INK, "font": F_X}],
       align=PP_ALIGN.CENTER)
L.text(sl, 1.0, 3.50, SW - 2.0, 0.45, [{"t": "Q&A", "size": 17, "color": ACC, "font": F_S}],
       align=PP_ALIGN.CENTER)
L.hline(sl, ML, 5.38, CW)
L.text(sl, ML, 5.49, CW, 0.3,
       [{"t": "참고문헌 (저자 알파벳순)", "size": 10, "color": GRAY2, "font": F_S}])
refs_l = [
    "Aljubran & Horne (2024) Geotherm. Energy 12  ·  Biskaborn et al. (2019) Nat. Commun. 10",
    "Brown et al. (2000) Polar Geogr. 24  ·  Gautam et al. (2025) Sci. Rep. 15",
    "Gorishniy et al. (2021) NeurIPS 34  ·  Gorishniy et al. (2025) ICLR",
    "Hjort et al. (2018) Nat. Commun. 9  ·  Hugelius et al. (2014) Biogeosciences 11",
    "Jafarov et al. (2012) The Cryosphere 6  ·  Kudryavtsev et al. (1974) Moscow Univ. Press",
    "Liu et al. (2023) Cold Reg. Sci. Technol. 216  ·  Meyer & Pebesma (2021) Methods Ecol. Evol. 12",
]
refs_r = [
    "Muñoz-Sabater et al. (2021) ESSD 13  ·  Nelson et al. (1998) JGR 103",
    "Obu et al. (2019) Earth-Sci. Rev. 193  ·  Prokhorenkova et al. (2018) NeurIPS 31",
    "Ran et al. (2022) ESSD 14  ·  Rantanen et al. (2022) Commun. Earth Environ. 3",
    "Riseborough et al. (2008) Permafr. Periglac. Process. 19  ·  Romano et al. (2019) NeurIPS 32",
    "Schuur et al. (2015) Nature 520  ·  Stefan (1891) Ann. Phys. 278",
    "Westermann et al. (2024) ESA Permafrost CCI v4.0",
]
half = (CW - 0.6) / 2
L.text(sl, ML, 5.78, half, 1.3, [[{"t": t, "size": 8.3, "color": MUTE, "font": F_M}] for t in refs_l],
       line_spacing=1.2, space_after=2)
L.text(sl, ML + half + 0.6, 5.78, half, 1.3, [[{"t": t, "size": 8.3, "color": MUTE, "font": F_M}] for t in refs_r],
       line_spacing=1.2, space_after=2)
L.text(sl, ML, 7.12, CW, 0.3,
       [{"t": "데이터  KPDC(KOPRI-KPDC-00002125) · CALM · ABoVE · GTN-P · ERA5-Land · ESA CCI ALT",
         "size": 9, "color": MUTE, "font": F_M}], align=PP_ALIGN.CENTER)

# ============================================================ 저장
os.makedirs("render", exist_ok=True)
out = "render/permafrost_final.pptx"
prs.save(out)
print("saved", out, "slides:", PN[0])
