"""G1 기존 한계·개선·기대 효과 (덱 전용 쪽, 12 × 4.90 in = 3600 × 1470 px, 300 dpi, 영문 FreeSans). 높이는 G1_H 로 바꿀 수 있다(기본 4.90).

v10 개정(2026-10-16, 사용자 지시: 그림 안 글자는 모두 영어 · 그림 공간 확대): 높이 4.20 → 4.90 in(목표 문장은 덱이 그림 아래에 둔다), 늘어난 높이는
두 행 차트로 간다(layout 이 글자 높이에서 계산). 열 머리 Limits of existing work · Improvement · Expected outcome. 모든 글자 영어(논문 어휘),
TDD 축 달 이름 Jan·Apr·Jul·Oct. 관측일 표식 라벨은 두 줄 'Year-specific' / 'climate input', 파선은 'Mean climate input (2011–2024)'.
저장 전 모든 글자 객체에서 한글을 찾으면 멈춘다(no_hangul).

v9 개정(2026-10-15, 설계 지침 v9 항 5·7·8; 아래 v8 명세보다 앞선다)
- 세로 여유: 해·월 눈금이 스스로 읽히는 x축 제목('연도', '연중 일수 (d)')을 빼고(TDD 축 눈금은 1·4·7·10월), 두 행 차트를 1.1·1.2 in 로 키웠다.
- 주석 축소: 레나 델타 블록 5겹 지도와 겹 열쇠, 2011–2024 범위 띠와 그 글줄, 관측 범위 띠, 선 끝 표식, 지도 경위도 눈금 글자·관측 셀 점, Δ 글줄과 괄호선을 뺐다.
- 열 2 위 라벨: 관측일 표식 = '관측 연도 기후 입력'(관측일까지 누적 TDD), 파선 = '평년 기후 입력'(2011–2024 평균 연 TDD). '강제력' 표현 없음.
- 열 2 아래: 결합 도식만(Stefan 블록 E√TDD ⊕ 잔차 ML 블록 — 산출 지도 썸네일), 연결선은 화살촉 없는 선. 제곱근은 mathtext.
- 열 1 아래 범례: y축에서 0.04 축 폭 띄우고, 범례 아래 끝이 자료보다 위에 오도록 y 상한을 자동으로 올린다.
- 열 3 아래: 제품 이름은 제품 막대 위(제품 색), 주황 막대 열쇠는 축 안 왼쪽 위. 축척 막대(500 km)는 첫 지도 틀 안.
- 덱 설명 줄 '기존 연구 한계 · 개선 · 기대 효과(실자료 요약)' 은 덱 명세(deck_spec) 몫이며 그림에는 없다.


명세
- 메시지: 해마다 달라지는 ALT 의 1 km 예측과 기존 제품과의 같은 셀·같은 해 비교. 목표 글줄은 슬라이드의 주황 상자가 맡으므로
  그림은 세 열 머리(기존 연구 한계 | 개선 | 기대 효과)로 시작한다. 열마다 실자료 소형 차트 둘, 각 차트 위 한 줄 설명.
  채움 머리띠·회색 카드 없음, 헤어라인과 굵은 열 머리만.
- 세로 배치는 글자 크기(pt)에서 계산한다(layout()): 열 머리 → 밑줄 → 설명 1 → 차트 1(위 눈금 글자 반 높이 여유) → x축 제목 →
  설명 2 → 차트 2 → x축 제목 → 바닥. 두 차트 높이는 남는 높이를 44 : 56 으로 나눈다. 캔버스 높이를 바꿔도 겹치지 않는다.
- 자료: labels_cell_year_v5.csv(관측 연도 분포, 장기 관측 셀, 블록), products_cell_year_v5.csv(제품 연도 값), forcing_year.csv(tdd_cum_1..12),
  forcing_clim.csv(clim1124_tdd), maps/Alaska/pred_<year>.csv.gz(연도별 지도), runs/core_v6/common_cells_eval.csv(공통 셀 25 km 마스크 RMSE).
  수치는 모두 파일에서 읽는다.
- 글자(토큰 slide 항목): 축 제목 14 pt(axis_label_pt), 설명·범례·직접 라벨 13 pt(direct_label_pt), 차트 눈금·경위도 눈금 12 pt(tick_pt; 한글 눈금은 13 pt),
  열 머리 15 pt 굵게.
- 색: 제품(Wei·Liu·CCI v5)은 design/style_tokens_v4.json products 항목(style_v5.PRODUCT_HEX), 제안 방법 주홍(R1), 블록 5겹은 Okabe–Ito,
  물리 블록 초록(#DCEBDD/#86AE8A), 학습 블록 노랑(#F7E9AE/#C9AE5B).
- 산출: deck/assets/v7/G1_goal_slide.png(RGB, 흰 바탕; +pdf), G1_goal_spec.json(배치·선택 셀·수치 출처·QA 기록).
- 실행(CPU 만): OMP_NUM_THREADS=2 nice -n 10 [G1_H=4.2] python3 scripts/4_visualization/v7_g1_goal_slide.py [--slide]
  (--slide 는 호환용 인자. 이 그림은 슬라이드판 하나만 만든다.)

검토 반영(10-15, 설계 지침 v8 항 7·8)
- 겹침: 1행 x축 제목과 2행 설명 글줄 사이를 글자 높이에서 계산(G1_H 4.2 에서 '연중 일수 (d)' 와 '물리 + 잔차 ML · 블록 5겹 검증' 충돌 제거).
- 열 2 위: '기후 입력의 시간 정합'. 2014 누적 TDD 곡선(ERA5-Land), 관측일 표식과 '관측일까지 누적 TDD' 안내선, 기후 평균(2011–2024 연 TDD) 파선.
- 열 1 아래: 레나 최장 기록 셀의 관측 연도별 ALT 대 제품 3종(Wei 2026·Liu 2024·ESA CCI v5, 제품 토큰 색, Liu 파선) + 관측 범위 회색 띠.
- 열 2 아래: 결합 도식 = 물리 블록(E·√TDD, Stefan 물리) ⊕ 학습 블록(이진 트리 3개, 잔차 학습) → 산출 지도 썸네일(알래스카 2014, R1).
  아래에 레나 블록 5겹 지도와 겹 색 열쇠.
- 열 3 아래: 공통 셀 RMSE 막대. x 눈금은 지역 이름만, 제품 이름은 제품 막대 위에 제품 색으로, 범례(물리 + 잔차 ML | 최선 기존 제품)는 눈금 아래.
- 이전 반영(10-14): y축 라벨 회전 점검(spec.json qa.ylabel_rotation), 블록 5겹 Okabe–Ito 색각 점검(choices.fold_palette), ⊕ 는 제안 방법 색.

2차 검토 반영(10-15 오후)
- 지역 이름 통일: 레나 델타는 그림 안에서 '레나델타' 하나로(열 1 설명·열 2 설명·열 3 눈금). 산출 썸네일 글줄은 '알래스카 / 2014' 두 줄.
- 여백: 위 0.11 in, 아래 0.08 in(투영·PPTX 삽입 때 잘림 대비; 옆 여백 0.10 in 과 같은 자릿수).
- 열 2 위: 같은 셀의 2011–2024 일 누적 TDD 범위를 회색 띠(equiv_band)로 깔고 2014 곡선을 그 위에 둔다(연도 막대 띠 대신; 기후 평균 파선과 같은 기간).
  '기후 평균' 글줄은 띠 위쪽 가장자리가 파선을 넘기 전 자리(자료에서 계산)에, '2011–2024 범위' 글줄은 띠 꼭대기 위에, 곡선 연도 '2014' 는 고원 오른쪽 끝 아래에.
  자료 출처는 설명 글줄 '기후 입력의 시간 정합 (ERA5-Land)'. 관측일 표식은 세로·가로 안내선과 발치의 '관측일' 글줄로 읽는다(왼쪽 가로 안내선 위의
  '관측일까지 누적 TDD' 글줄은 다른 해의 누적 곡선(띠 위 가장자리)이 두 높이 모두에서 글줄을 지나 뺐다; 값은 y축 '누적 TDD' 로 읽힌다).
- 축척 막대(레나델타 블록 지도): 후보 자리 가운데 막대·글줄이 지도 틀 안(여백 0.04 in)에 들고 블록 채움과 겹치지 않는 첫 자리(place_scale_bar).
- 글자 크기: 축 제목 14 pt(토큰 axis_label_pt), 눈금 12 pt, 한글 눈금(지역 이름) 13 pt, 그 밖 13 pt. y축 여백 X_OFF 는 가장 넓은 y축 제목·눈금에서
  계산해 세 열의 차트 왼쪽 변을 같은 자리에 둔다.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU 사용 금지(CPU 전용)

import argparse  # noqa: E402
import calendar  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))
sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402  (M.canvas, M.T, M.RT, M.hline, M.vline, M.arrow, M.sub_axes, M.block, M.qa_and_save, M.setup_fonts)
import polar.style_v5 as S  # noqa: E402  (제품 색: 토큰 products 항목)

M.OUT = ROOT / "deck" / "assets" / "v7"
PROC = ROOT / "data" / "processed" / "v5"
RUN = PROC / "runs" / "core_v6"
STEM = "G1_goal"

# ================================================================ 배치(인치)
W, H = 12.0, float(os.environ.get("G1_H", "4.90"))
COLS = [(0.10, 3.80), (4.10, 7.80), (8.10, 11.90)]      # 열 안쪽 x 범위
SEPX = [3.95, 7.95]
X_OFF = 0.62                                            # 축 왼쪽 여백(y축 제목·눈금 글자 폭). g1() 에서 글자 치수로 다시 계산한다
FS_T = 13.0                                             # 설명·범례·직접 라벨·한글 눈금 13 pt(토큰 slide.direct_label_pt)
FS_AX = 14.0                                            # 축 제목 14 pt(토큰 slide.axis_label_pt)
FS_K = 12.0                                             # 숫자 눈금 12 pt(토큰 slide.tick_pt)
FS_HEAD = M.FS["head"]                                  # 열 머리 15 pt
TICK_LEN, TICK_PAD, LABEL_PAD = 3.0, 2.0, 2.0           # pt(chart_axes 와 같은 값)
M_TOP, M_BOT = 0.11, 0.08                               # 위·아래 여백(인치)
CAP = {"a1": "Observation years (Alaska)", "a2": "Product disagreement (one Lena Delta cell)", "a3": "Timing of the climate input (ERA5-Land)",
       "a4": "Physics + residual ML", "a5": "Yearly 1 km maps (Alaska)", "a6": "Common-cell RMSE vs best product (cm)"}
YLAB = {"a1": "Cell-years", "a2": "ALT (cm)", "a3": "Cumulative TDD\n(°C·d)"}     # y축 제목(a3 는 두 줄: 한 줄이면 축보다 길다). 열 3 아래는 가로 막대(단위는 설명 글줄)
REG_KO = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}              # 그림 지역 이름(영문, S.REGION_EN 과 같다)
HEADS = ("Limits of existing work", "Improvement", "Expected outcome")
R1 = M.COL["R1"]                                        # 제안 방법 색 #D55E00
ALASKA = M.REG["Alaska"]                                # 지역 색 #332288
# 제품 이름(common_cells_eval.csv series) → 토큰 products 키. 색은 S.PRODUCT_HEX(= design/style_tokens_v4.json color.products[*].hex).
PRODUCT_KEY = {"Liu 2024": "Liu", "ESA CCI v5": "CCI5", "ESA CCI v4": "CCI", "Wei 2026": "Wei", "Aalto 2018": "Aalto", "Yi–Kimball 2019": "YK"}
PROD_COL = {"wei": S.PRODUCT_HEX["Wei"], "liu": S.PRODUCT_HEX["Liu"], "cci5y": S.PRODUCT_HEX["CCI5"]}
PROD_NAME = {"wei": "Wei 2026", "liu": "Liu 2024", "cci5y": "ESA CCI v5"}
DASH = (0, (3.5, 1.8))                                  # style_v5 METHOD_LS 'dashed'
PROD_LS = {"wei": "-", "liu": DASH, "cci5y": "-"}       # Liu 는 토큰대로 파선(CCI 계열과 색상이 가깝다)
MONTHS = [(1, "Jan"), (4, "Apr"), (7, "Jul"), (10, "Oct")]   # 누적 TDD 축 눈금(그 달 1일, 연중 일수 0 기준)
SPEC: dict = {"stem": STEM, "size_in": [W, H], "sources": {}, "choices": {}}
LAY: dict = {}


def lut(cm, lo, hi, n=250):
    """끝을 자른 색표(paper_v3/maps_alt_v3.lut 복사)."""
    c = ListedColormap(cm(np.linspace(lo, hi, n)))
    c.set_bad((0.0, 0.0, 0.0, 0.0))
    return c



def pct_range(v, lo=2.0, hi=98.0, step=5.0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(step * np.floor(np.percentile(v, lo) / step)), float(step * np.ceil(np.percentile(v, hi) / step))]



def minus(s: str) -> str:
    return s.replace("-", "−")



# ================================================================ 글자 치수와 세로 배치
_MEAS: dict = {}


def measure(s, size, weight="normal"):
    """글줄의 (폭, 높이) 인치. 300 dpi 임시 캔버스에서 잰다(글꼴 등록 뒤에 부른다)."""
    key = (s, size, weight)
    if key not in _MEAS:
        f = plt.figure(figsize=(3, 1), dpi=M.DPI)
        t = f.text(0.1, 0.5, s, fontsize=size, fontweight=weight)
        f.canvas.draw(); bb = t.get_window_extent(f.canvas.get_renderer()); plt.close(f)
        _MEAS[key] = (bb.width / M.DPI, bb.height / M.DPI)
    return _MEAS[key]



def yaxis_offset(labels, ticks, margin=0.02):
    """축 왼쪽 여백(인치) = 가장 넓은 눈금 글자 + 눈금·간격 + 제목 간격 + 회전한 제목 폭(여러 줄이면 줄 수 × 줄 높이 × 줄 간격) + 열 안쪽 여백."""
    pt = 1 / 72.0
    tick_w = max(measure(t, FS_K)[0] for t in ticks)
    lab_w = max(len(s.split("\n")) * measure(s.split("\n")[0], FS_AX)[1] * (1.1 if "\n" in s else 1.0) for s in labels)
    return round(tick_w + (TICK_LEN + TICK_PAD) * pt + LABEL_PAD * pt + lab_w + margin, 3)



def layout(H):
    """세로 좌표(인치)를 글자 높이에서 계산한다. 반환 dict 는 LAY 에 들어간다.
    v9: 행마다 x축 제목 없이 눈금 글자 한 줄(해·월·지역 이름)만 둔다 → 열 머리 → 밑줄 → 설명 1 → 차트 1 → 눈금 글자 → 설명 2 → 차트 2 → 눈금 글자 → 바닥."""
    h_head = measure(HEADS[0], FS_HEAD, "bold")[1]
    h_cap = max(measure(s, FS_T)[1] for s in CAP.values())
    h_tick = measure("2010", FS_K)[1]
    h_tick_ko = measure("Alaska", FS_T)[1]
    pt = 1 / 72.0
    xt_zone = (TICK_LEN + TICK_PAD) * pt + max(h_tick, h_tick_ko)               # 축 아래 ~ 눈금 글자 바닥
    m_top, m_bot = M_TOP, M_BOT
    g_head_rule, g_rule_cap, g_cap_ax, g_xt_cap = 0.05, 0.07, 0.07, 0.17
    y = H - m_top
    Y_HEAD = y - h_head / 2; y -= h_head
    Y_HEADRULE = y - g_head_rule; y = Y_HEADRULE
    Y_CAP1 = y - g_rule_cap - h_cap / 2; y -= g_rule_cap + h_cap
    AX1_T = y - g_cap_ax - h_tick / 2
    fixed = (H - AX1_T) + xt_zone + g_xt_cap + h_cap + g_cap_ax + h_tick / 2 + xt_zone + m_bot
    total = H - fixed
    ax1_h, ax2_h = 0.47 * total, 0.53 * total
    AX1_B = AX1_T - ax1_h
    XT1_B = AX1_B - xt_zone
    Y_CAP2 = XT1_B - g_xt_cap - h_cap / 2
    AX2_T = Y_CAP2 - h_cap / 2 - g_cap_ax - h_tick / 2
    AX2_B = AX2_T - ax2_h
    d = dict(Y_HEAD=Y_HEAD, Y_HEADRULE=Y_HEADRULE, Y_CAP1=Y_CAP1, AX1_T=AX1_T, AX1_B=AX1_B, XT1_B=XT1_B, Y_CAP2=Y_CAP2,
             AX2_T=AX2_T, AX2_B=AX2_B, XT2_B=AX2_B - xt_zone, xt_zone=xt_zone, h_cap=h_cap, h_tick=h_tick, h_tick_ko=h_tick_ko,
             CAP2_B=Y_CAP2 - h_cap / 2, CAP1_B=Y_CAP1 - h_cap / 2, m_bot=m_bot, m_top=m_top, ax1_h=ax1_h, ax2_h=ax2_h)
    assert d["XT2_B"] >= m_bot - 1e-6 and ax1_h > 0.8, d
    return {k: round(float(v), 4) for k, v in d.items()}


# ================================================================ 점검(회전·지도 비율)
def _glyph_bitmap(s, rot):
    fig = plt.figure(figsize=(2.5, 2.5), dpi=M.DPI); ax_ = fig.add_axes([0, 0, 1, 1]); ax_.axis("off")
    ax_.text(0.5, 0.5, s, rotation=rot, ha="center", va="center", fontsize=FS_T)
    fig.canvas.draw(); a = np.asarray(fig.canvas.buffer_rgba())[:, :, 0].copy(); plt.close(fig)
    ys, xs = np.where(a < 128)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]



def ylabel_rotation_check(labels):
    """90° 회전 글줄을 되돌린 비트맵과 정립 비트맵을 비교한다(한글·라틴 혼합 글줄이 한 덩어리로 회전되는지). 여러 줄 제목은 줄마다 본다."""
    out = {}
    for s in [ln for lab in labels for ln in lab.split("\n")]:
        a0 = _glyph_bitmap(s, 0); a1 = np.rot90(_glyph_bitmap(s, 90), -1)
        h = min(a0.shape[0], a1.shape[0]); w = min(a0.shape[1], a1.shape[1])
        d = np.abs(a0[:h, :w].astype(int) - a1[:h, :w].astype(int))
        out[s] = dict(shape_upright=list(a0.shape), shape_rotated_back=list(a1.shape), mean_abs_diff=round(float(d.mean()), 2),
                      frac_pixels_differ=round(float((d > 128).mean()), 4), ok=bool(a0.shape == a1.shape and (d > 128).mean() < 0.01))
    return out



def map_aspect(proj, extent):
    """투영 범위의 가로/세로 비(지도 축 상자를 자료 비율에 꼭 맞추기 위해)."""
    import cartopy.crs as ccrs
    f = plt.figure(figsize=(2, 2)); a = f.add_axes([0, 0, 1, 1], projection=proj); a.set_extent(extent, crs=ccrs.PlateCarree())
    xl, yl = a.get_xlim(), a.get_ylim(); plt.close(f)
    return (xl[1] - xl[0]) / (yl[1] - yl[0])



def tick_label_artists(fig):
    """눈금 글자(축 눈금·색 막대 눈금·경위도 눈금) 객체 집합: 12 pt 허용, 그 밖의 글자는 13 pt 이상이어야 한다."""
    ids = set()
    for a in fig.axes:
        if hasattr(a, "projection") or not a.axison:  # GeoAxes·캔버스 축: get_xticklabels() 가 숨은 눈금 글자를 만들어 내므로 건너뛴다
            pass
        else:
            for t in a.get_xticklabels(minor=False) + a.get_yticklabels(minor=False) + a.get_xticklabels(minor=True) + a.get_yticklabels(minor=True):
                ids.add(id(t))
        for gl in getattr(a, "_gridliners", []):
            for t in list(getattr(gl, "xlabel_artists", [])) + list(getattr(gl, "ylabel_artists", [])) + list(getattr(gl, "label_artists", [])):
                ids.add(id(t))
    return ids



def bbox_in(fig, art):
    """아티스트 상자를 인치로."""
    bb = art.get_window_extent(fig.canvas.get_renderer())
    return (bb.x0 / fig.dpi, bb.y0 / fig.dpi, bb.x1 / fig.dpi, bb.y1 / fig.dpi)



# ================================================================ 자료
def labels():
    L = pd.read_csv(PROC / "labels_cell_year_v5.csv")
    SPEC["sources"]["labels"] = str((PROC / "labels_cell_year_v5.csv").relative_to(ROOT))
    return L[L.in_forcing == 1].copy()



def alaska_counts(L):
    s = L[L.region == "Alaska"].groupby("year").size().reindex(range(2010, 2025), fill_value=0)
    SPEC["choices"]["alaska_cell_years_by_year"] = {int(k): int(v) for k, v in s.items()}
    return s



def lena_long_cell(L):
    """레나 최장 기록 셀: 제품 3종(wei, liu, cci5y)이 2010–2024 에 값을 가진 셀 중 관측 연도 수 최다(동률은 loc_id 오름차순; v7_f2 pick_lena_cell 과 같다)."""
    P = pd.read_csv(PROC / "products_cell_year_v5.csv")
    SPEC["sources"]["products"] = str((PROC / "products_cell_year_v5.csv").relative_to(ROOT))
    ln = L[L.region == "Lena"]
    n = ln.groupby("loc_id").year.nunique()
    p = P[P.year.between(2010, 2024) & P["product"].isin(list(PROD_COL))]
    ok = p.groupby(["loc_id", "product"]).value_cm.apply(lambda s: s.notna().any()).unstack().fillna(False)
    ok = ok.index[ok.all(axis=1)]
    n = n[n.index.isin(ok)].sort_index()
    lid = int(n.sort_values(ascending=False, kind="stable").index[0])
    obs = ln[ln.loc_id == lid].groupby("year").alt_cm.mean().reset_index().sort_values("year")
    prods = {}
    for k in PROD_COL:
        s = P[(P.loc_id == lid) & (P["product"] == k) & P.year.between(2010, 2024)].dropna(subset=["value_cm"]).sort_values("year")
        prods[k] = s[["year", "value_cm"]].reset_index(drop=True)
    SPEC["choices"]["lena_long_cell"] = dict(loc_id=lid, n_years=int(n[lid]), lat=round(float(ln[ln.loc_id == lid].lat.iloc[0]), 3),
                                            lon=round(float(ln[ln.loc_id == lid].lon.iloc[0]), 3),
                                            years=[int(y) for y in obs.year], alt_cm=[round(float(v), 1) for v in obs.alt_cm],
                                            obs_range_cm=[round(float(obs.alt_cm.min()), 1), round(float(obs.alt_cm.max()), 1)],
                                            products={k: {int(y): round(float(v), 1) for y, v in zip(s.year, s.value_cm)} for k, s in prods.items()})
    return lid, obs, prods



def tdd_curve(L):
    """알래스카 2014 라벨 중 최빈 관측일 셀, 그 가운데 연 TDD 가 중앙값에 가장 가까운 셀: 2014 일 누적 TDD 곡선(월 누적값 선형 보간),
    관측일까지 값(testbed_v5.with_forcing('year_obs') 와 같은 보간), 기후 평균 연 TDD(forcing_clim clim1124_tdd)."""
    a = L[(L.region == "Alaska") & (L.year == 2014) & L.obs_doy.notna()]
    doy = float(a.obs_doy.mode().iloc[0])
    ids = a[a.obs_doy == doy].loc_id.unique()
    cols = ["loc_id", "year"] + [f"tdd_cum_{k}" for k in range(1, 13)]
    F = pd.read_csv(PROC / "forcing_year.csv", usecols=cols)
    SPEC["sources"]["forcing"] = str((PROC / "forcing_year.csv").relative_to(ROOT))
    F14 = F[(F.year == 2014) & F.loc_id.isin(ids)].set_index("loc_id")
    ann = F14.tdd_cum_12
    lid = int((ann - ann.median()).abs().idxmin())
    cum = F14.loc[lid, [f"tdd_cum_{k}" for k in range(1, 13)]].values.astype(float)
    ends = np.cumsum([calendar.monthrange(2014, m)[1] for m in range(1, 13)])
    x = np.r_[0, ends]; y = np.r_[0.0, cum]
    v_obs = float(np.interp(doy, x, y))
    C = pd.read_csv(PROC / "forcing_clim.csv", usecols=["loc_id", "clim1124_tdd"])
    SPEC["sources"]["forcing_clim"] = str((PROC / "forcing_clim.csv").relative_to(ROOT))
    clim = float(C.loc[C.loc_id == lid, "clim1124_tdd"].iloc[0])
    # 같은 셀의 2011–2024(기후 평균과 같은 기간) 일 누적 곡선 범위: 연도마다 월 누적값을 그 해 월말 일수로 보간해 하루 격자(0–366)에 올린다
    Fc = F[(F.loc_id == lid) & F.year.between(2011, 2024)].sort_values("year")
    xg = np.arange(0, 367, dtype=float)
    curves = {}
    for _, rw in Fc.iterrows():
        yr = int(rw.year)
        e = np.cumsum([calendar.monthrange(yr, m)[1] for m in range(1, 13)])
        curves[yr] = np.interp(xg, np.r_[0, e], np.r_[0.0, rw[[f"tdd_cum_{k}" for k in range(1, 13)]].values.astype(float)])
    stack = np.vstack([curves[k] for k in sorted(curves)])
    band = dict(x=xg, lo=stack.min(axis=0), hi=stack.max(axis=0))
    yrs = sorted(curves)
    assert abs(float(np.mean([curves[k][-1] for k in yrs])) - clim) < 0.5, "기후 평균(clim1124_tdd)은 2011–2024 연 TDD 평균과 같아야 한다"
    SPEC["choices"]["tdd_curve"] = dict(loc_id=lid, year=2014, obs_doy=doy, tdd_obs=round(v_obs, 1), tdd_year=round(float(cum[-1]), 1),
                                       clim_tdd_2011_2024=round(clim, 1), n_cells_same_doy=int(len(ids)),
                                       lat=round(float(a[a.loc_id == lid].lat.iloc[0]), 3), lon=round(float(a[a.loc_id == lid].lon.iloc[0]), 3),
                                       band_years=[yrs[0], yrs[-1]], band_n_years=len(yrs),
                                       band_year_end=dict(min=[int(yrs[int(np.argmin([curves[k][-1] for k in yrs]))]), round(float(band["lo"][-1]), 1)],
                                                          max=[int(yrs[int(np.argmax([curves[k][-1] for k in yrs]))]), round(float(band["hi"][-1]), 1)]))
    return x, y, doy, v_obs, float(cum[-1]), clim, band



def alaska_grid(year, lon0=-168.0, lon1=-141.0, lat0=59.8, lat1=71.4, dx=0.08, dy=0.04):
    """연도 지도 셀(≈1 km)을 0.08° × 0.04° 상자 평균으로 묶는다(소형 지도용 부표본)."""
    f = PROC / "maps" / "Alaska" / f"pred_{year}.csv.gz"
    m = pd.read_csv(f, usecols=["lat", "lon", "r1"])
    nx = int(np.ceil((lon1 - lon0) / dx)); ny = int(np.ceil((lat1 - lat0) / dy))
    ix = np.floor((m.lon.values - lon0) / dx).astype(int); iy = np.floor((m.lat.values - lat0) / dy).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(m.r1.values)
    k = iy[ok] * nx + ix[ok]
    s = np.bincount(k, weights=m.r1.values[ok], minlength=nx * ny); n = np.bincount(k, minlength=nx * ny)
    g = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(ny, nx)
    SPEC["sources"][f"map_{year}"] = str(f.relative_to(ROOT))
    return g, np.linspace(lon0, lon0 + nx * dx, nx + 1), np.linspace(lat0, lat0 + ny * dy, ny + 1), m.r1.values[ok]



def rmse_table():
    e = pd.read_csv(RUN / "common_cells_eval.csv")
    SPEC["sources"]["common_cells"] = str((RUN / "common_cells_eval.csv").relative_to(ROOT))
    e = e[e.mask_km == 25]
    rows = []
    for r in ("Alaska", "Lena", "Canada"):
        ours = e[(e.region == r) & (e.series == "R1 (CatBoost)")].iloc[0]
        prod = e[(e.region == r) & (e.kind == "product")].sort_values("rmse").iloc[0]
        key = PRODUCT_KEY[str(prod.series)]
        rows.append(dict(region=r, ours=float(ours.rmse), n=int(ours.n), product=str(prod.series), product_key=key,
                         product_hex=S.PRODUCT_HEX[key], prod_rmse=float(prod.rmse),
                         delta_ours_minus_product=round(float(ours.rmse - prod.rmse), 2)))
    SPEC["choices"]["rmse_mask25"] = rows
    return rows



# ================================================================ 축 양식
def chart_axes(fig, col, row, xoff=None):
    """xoff = 축 왼쪽 여백(y축 제목·눈금 글자 폭). 기본값은 g1() 이 글자 치수로 계산한 X_OFF(세 열 공통: 차트 왼쪽 변이 같은 자리)."""
    xoff = X_OFF if xoff is None else xoff
    x0, x1 = COLS[col]
    b, t = (LAY["AX1_B"], LAY["AX1_T"]) if row == 0 else (LAY["AX2_B"], LAY["AX2_T"])
    ax = M.sub_axes(fig, x0 + xoff, b, x1 - 0.05 - (x0 + xoff), t - b)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(0.8); ax.spines[k].set_edgecolor(M.ZERO)
    ax.tick_params(labelsize=FS_K, length=TICK_LEN, width=0.8, pad=TICK_PAD, colors=M.INK)
    return ax


def flow_chevron(ax, x, yc, w, h, color="#9A9A9A"):
    """작은 흐름 셰브런(채움, 왼쪽 홈·오른쪽 뾰족). 화살 선·화살촉이 아닌 공정 셰브런 모양(지침 v9 1.3). (x, yc) = 왼쪽 끝·세로 가운데."""
    from matplotlib.patches import Polygon
    tip = 0.55 * w
    pts = [(x, yc + h / 2), (x + w - tip, yc + h / 2), (x + w, yc), (x + w - tip, yc - h / 2), (x, yc - h / 2), (x + tip * 0.75, yc)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=4))


def scale_bar_in(axm, km, label, fx=0.07, fy=0.10):
    """지도 틀 안 왼쪽 아래 축척 막대(투영 좌표 m). 글줄은 막대 위, 막대 시작에 왼맞춤, 흰 테. 반환 (선, 글자)."""
    xl, yl = axm.get_xlim(), axm.get_ylim()
    x0 = xl[0] + fx * (xl[1] - xl[0]); y0 = yl[0] + fy * (yl[1] - yl[0])
    ln, = axm.plot([x0, x0 + km * 1e3], [y0, y0], color=M.INK, lw=1.6, solid_capstyle="butt", zorder=8,
                   path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    t = axm.text(x0, y0 + 0.05 * (yl[1] - yl[0]), label, ha="left", va="bottom", fontsize=FS_K, color=M.INK, zorder=8,
                 path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    return ln, t


def caption(ax, col, row, s):
    return M.T(ax, COLS[col][0], LAY["Y_CAP1"] if row == 0 else LAY["Y_CAP2"], s, size=FS_T, color=M.AUX)



def plus_node(ax, x, y, r=0.09, color=M.INK):
    """⊕ 결합 마디. 제안 방법(물리 + 잔차 ML)을 뜻하므로 제안 방법 색으로 그려 막대 색과 잇는다."""
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=1.2, zorder=6))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=color, lw=1.2, zorder=7, solid_capstyle="butt")
    ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color=color, lw=1.2, zorder=7, solid_capstyle="butt")



def tree_glyph(ax, x, y, w, h, color=M.INK, lw=0.9, r=0.013, z=5):
    """깊이 2 이진 트리(선과 작은 마디). (x, y) = 왼쪽 아래, 뿌리는 위 가운데."""
    root = (x + w / 2, y + h)
    kids = [(x + w / 4, y + h / 2), (x + 3 * w / 4, y + h / 2)]
    leaves = [(x + w * (2 * i + 1) / 8, y) for i in range(4)]
    for k in kids:
        ax.plot([root[0], k[0]], [root[1], k[1]], color=color, lw=lw, solid_capstyle="round", zorder=z)
    for i, lf in enumerate(leaves):
        k = kids[i // 2]
        ax.plot([k[0], lf[0]], [k[1], lf[1]], color=color, lw=lw, solid_capstyle="round", zorder=z)
    for p in [root] + kids + leaves:
        ax.add_patch(Circle(p, r, facecolor="white", edgecolor=color, lw=lw, zorder=z + 1))



def geo_axes(fig, x, y, w, h, proj, extent, scale):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    axm = M.sub_axes(fig, x, y, w, h, projection=proj)
    axm.set_extent(extent, crs=ccrs.PlateCarree())
    axm.add_feature(cfeature.LAND.with_scale(scale), facecolor=M.LAND, edgecolor="none", zorder=0)
    axm.add_feature(cfeature.COASTLINE.with_scale(scale), edgecolor=M.COAST, linewidth=0.4, zorder=3)
    axm.spines["geo"].set_edgecolor(M.HAIR); axm.spines["geo"].set_linewidth(0.7)
    return axm



def data_y_from_in(ax, y_in):
    """캔버스 세로 인치 → 축 자료 좌표."""
    return ax.transData.inverted().transform((0, y_in * ax.figure.dpi))[1]



def in_from_data_y(ax, y):
    return ax.transData.transform((0, y))[1] / ax.figure.dpi



# ================================================================ 그림
def g1():
    global X_OFF
    import re
    import cartopy.crs as ccrs
    import cmcrameri.cm as cmc
    M.setup_fonts()
    LAY.update(layout(H))
    X_OFF = yaxis_offset(list(YLAB.values()), ["8000", "150", "600", "40"])      # 세 열 공통 축 왼쪽 여백(가장 넓은 제목·눈금 기준)
    SPEC["layout_in"] = dict(LAY, x_off=X_OFF)
    SPEC["choices"]["product_hex"] = {k: S.PRODUCT_HEX[k] for k in ("Wei", "Liu", "CCI5")}
    SPEC["choices"]["design_notes"] = dict(
        x_labels="해·월·지역 눈금은 그 자체로 읽히므로 x축 제목을 두지 않는다(세로 여유). TDD 축 눈금 = 1·4·7·10월 1일",
        tdd_panel="같은 셀 2014 일 누적 TDD 곡선 + 관측일 표식(제안 방법 색) · 안내선 + 평년 연 TDD(2011–2024 평균, forcing_clim clim1124_tdd) 파선. "
                  "라벨: 표식 '관측 연도 기후 입력', 파선 '평년 기후 입력', 곡선 '2014'",
        lena_panel="최장 기록 레나 델타 셀: 관측 ALT(점) + 제품 3종(Wei 2026·Liu 2024·ESA CCI v5). 범례 아래 끝이 자료보다 위(자동 y 상한)",
        architecture="Stefan 블록(E√TDD, 초록) ⊕(제안 방법 색) 잔차 ML 블록(트리 3개, 노랑) — 산출 지도 썸네일(알래스카 2014 R1). 연결선은 화살촉 없음",
        maps="알래스카 2014·2018·2022 R1(같은 색표·범위), 경위도 눈금 글자·관측 점 없음, 축척 막대는 첫 지도 틀 안",
        bars="x 눈금 = 지역 이름, 제품 이름은 제품 막대 위(제품 색), 주황 막대 열쇠는 축 안 왼쪽 위. Δ 글줄 없음",
        margins=f"위 {M_TOP} in · 아래 {M_BOT} in · 옆 0.10 in",
        fonts="축 제목 14 pt, 설명·직접 라벨·범례·한글 눈금 13 pt, 숫자 눈금 12 pt, 열 머리 15 pt 굵게")
    fig, ax = M.canvas(W, H)
    L = labels()
    qa_extra: dict = {}

    # ---- 열 머리 · 헤어라인(목표 글줄은 슬라이드의 상자가 맡으므로 그림에는 없다)
    for (x0, x1), s in zip(COLS, HEADS):
        M.T(ax, x0, LAY["Y_HEAD"], s, size=FS_HEAD, weight="bold", color=M.HEAD)     # 열 머리 잉크색(method_figs_v4.HEAD, G0·G2·G5 와 같다)
        M.hline(ax, x0, x1, LAY["Y_HEADRULE"], color=M.HAIR, lw=0.8)
    for x in SEPX:
        M.vline(ax, x, 0.12, LAY["Y_HEADRULE"] - 0.06, color=M.SEP, lw=0.9, ls=(0, (3, 3)))

    # ---- 지도 색 범위(열 3 위 지도 3장과 열 2 도식의 산출 썸네일이 같은 색표·범위를 쓴다)
    years = [2014, 2018, 2022]
    grids = {yr: alaska_grid(yr) for yr in years}
    lo, hi = pct_range(np.concatenate([grids[yr][3] for yr in years]))
    SPEC["choices"]["alaska_map_range_cm"] = [lo, hi]
    cm_alt = lut(cmc.oslo_r, 0.12, 0.90); norm = Normalize(lo, hi)
    projA = ccrs.AlbersEqualArea(central_longitude=-154.5, central_latitude=65.0, standard_parallels=(60.0, 70.0))
    extA = [-168.0, -141.0, 59.8, 71.4]
    aspA = map_aspect(projA, extA)

    # ---- 열 1 · 위: 알래스카 셀-연도 수(연도별)
    cnt = alaska_counts(L)
    a1 = chart_axes(fig, 0, 0)
    a1.bar(cnt.index, cnt.values, width=0.75, color=ALASKA, lw=0)
    top1 = float(cnt.max()) * 1.08
    a1.set_xlim(2009.3, 2024.7); a1.set_xticks([2010, 2015, 2020]); a1.set_ylim(0, top1); a1.set_yticks(np.arange(0, top1, 4000))
    a1.set_ylabel(YLAB["a1"], fontsize=FS_AX, labelpad=3)
    caps = [caption(ax, 0, 0, CAP["a1"])]

    # ---- 열 1 · 아래: 레나 최장 기록 셀의 관측 연도별 ALT 대 제품 3종(제품 토큰 색, Liu 파선)
    lid, obs, prods = lena_long_cell(L)
    a2 = chart_axes(fig, 0, 1)
    for k, s in prods.items():
        a2.plot(s.year, s.value_cm, color=PROD_COL[k], lw=1.8, ls=PROD_LS[k], dash_capstyle="butt", zorder=2)
    a2.plot(obs.year, obs.alt_cm, "o", ms=6.5, mfc=M.INK, mec="white", mew=0.8, zorder=4)
    a2.set_xticks([2010, 2015, 2020]); a2.set_xlim(2009.3, 2024.7)
    a2.set_ylabel(YLAB["a2"], fontsize=FS_AX, labelpad=3)
    hs = [Line2D([], [], color=PROD_COL[k], lw=1.8, ls=PROD_LS[k]) for k in prods] + [Line2D([], [], marker="o", ms=6.5, mfc=M.INK, mec="white", mew=0.8, ls="none")]
    lg2 = a2.legend(hs, [PROD_NAME[k] for k in prods] + ["Observed"], loc="upper left", bbox_to_anchor=(0.04, 1.03), fontsize=FS_T, frameon=False, ncol=2,
                    handlelength=1.2, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0, borderpad=0.0, labelspacing=0.25)
    pts2 = [(float(yy), float(vv)) for s in prods.values() for yy, vv in zip(s.year, s.value_cm)] + [(float(yy), float(vv)) for yy, vv in zip(obs.year, obs.alt_cm)]
    ymax2 = max(v for _, v in pts2); ytop2 = float(np.ceil(ymax2 * 1.1 / 25) * 25)
    for _ in range(16):                                    # 범례 아래 끝(− 0.04 in)이 범례 x 범위의 자료보다 위가 될 때까지 y 상한을 25 cm 씩 올린다
        a2.set_ylim(0, ytop2); fig.canvas.draw()
        lb_ = lg2.get_window_extent(fig.canvas.get_renderer()); inv_ = a2.transData.inverted()
        (lx0_, ly0_), (lx1_, _) = inv_.transform([(lb_.x0, lb_.y0 - 0.04 * fig.dpi), (lb_.x1, lb_.y1)])
        if all(v < ly0_ for x_, v in pts2 if lx0_ - 0.4 <= x_ <= lx1_ + 0.4):
            break
        ytop2 += 25.0
    a2.set_yticks([v for v in (0, 50, 100, 150, 200, 250) if v <= ytop2])
    SPEC["choices"]["lena_panel_ytop_cm"] = ytop2
    caps.append(caption(ax, 0, 1, CAP["a2"]))

    # ---- 열 2 · 위: 2014 일 누적 TDD 곡선, 관측일 표식(= 관측 연도 기후 입력)·안내선, 평년 연 TDD(= 평년 기후 입력) 파선. 눈금 = 1·4·7·10월
    x, y, doy, v_obs, v_year, clim, band = tdd_curve(L)
    a3 = chart_axes(fig, 1, 0)
    ax1_h = LAY["AX1_T"] - LAY["AX1_B"]
    ax3_w = a3.get_position().width * W
    pad_in = 0.03
    h_lab = measure("Mean climate input (2011–2024)", FS_T)[1]
    ytop3 = max(clim, v_year) / (1.0 - (pad_in + h_lab + 0.05) / ax1_h)         # 파선 위에 글줄 한 줄과 여유
    u3 = ytop3 / ax1_h                                                             # 자료 단위 / 인치
    XMIN, XMAX = -24.0, 380.0
    dpi3 = (XMAX - XMIN) / ax3_w                                                   # 일 / 인치
    a3.plot(x, y, color=M.INK, lw=1.8, zorder=3)
    a3.plot([XMIN, XMAX], [clim, clim], color=M.ZERO, lw=1.0, ls=DASH, dash_capstyle="butt", zorder=2)
    a3.plot([doy, doy], [0, v_obs], color=M.AUX, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
    a3.plot([XMIN, doy], [v_obs, v_obs], color=M.AUX, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
    a3.plot([doy], [v_obs], "o", ms=7.5, mfc=R1, mec="white", mew=0.8, zorder=5)
    step = 100.0 if ytop3 < 450 else (200.0 if ytop3 < 700 else 300.0)
    a3.set_yticks(np.arange(0.0, ytop3 + 1e-9, step))
    m_doy = [calendar_doy(2014, m) for m, _ in MONTHS]
    a3.set_xticks(m_doy); a3.set_xticklabels([s_ for _, s_ in MONTHS], fontsize=FS_T)
    a3.set_xlim(XMIN, XMAX); a3.set_ylim(0, ytop3)      # set_yticks 가 범위를 넓히므로 범위는 눈금 뒤에 둔다. XMIN < 0: '1월' 과 y 눈금 '0' 이 모서리에서 떨어진다
    a3.set_ylabel(YLAB["a3"], fontsize=FS_AX, labelpad=3, linespacing=1.1)
    x_lab = XMIN + 0.06 * dpi3                                                     # 글줄 왼쪽: y축에서 0.06 in
    t_cl = a3.text(x_lab, clim + pad_in * u3, "Mean climate input (2011–2024)", ha="left", va="bottom", fontsize=FS_T, color=M.AUX, zorder=6)
    t_ob = a3.text(x_lab, v_obs + pad_in * u3, "Year-specific\nclimate input", ha="left", va="bottom", fontsize=FS_T, color=M.INK, zorder=6,
                   linespacing=1.1)
    t_14 = a3.text(XMAX - 2, v_year - pad_in * u3, "2014", ha="right", va="top", fontsize=FS_T, color=M.INK, zorder=6)
    SPEC["choices"]["tdd_panel_in"] = dict(ytop=round(ytop3, 1), labels=["Year-specific climate input (marker)", "Mean climate input (2011–2024) (dashed)", "2014 (curve)"],
                                           month_ticks_doy=m_doy)
    caps.append(caption(ax, 1, 0, CAP["a3"]))

    # ---- 열 2 · 아래: 결합 도식(Stefan 블록 ⊕ 잔차 ML 블록 — 산출 지도 썸네일), 아래 행 띠 가운데
    cx = (COLS[1][0] + COLS[1][1]) / 2
    bh = 0.72
    gw, yw, thw = 0.98, 1.08, 0.80
    g_plus, g_line = 0.12, 0.09
    pr = 0.11
    line_len = 0.26
    row_w = gw + g_plus + 2 * pr + g_plus + yw + g_line + line_len + g_line + thw
    rx = cx - row_w / 2
    thh = thw / aspA
    lab_h = measure("ALT 2014", FS_T)[1]
    band_top, band_bot = LAY["CAP2_B"] - 0.06, LAY["m_bot"]
    stack_h = max(bh, thh) + 0.06 + lab_h                                        # 블록·썸네일 + 썸네일 아래 글줄
    y_mid = band_bot + (band_top - band_bot - stack_h) / 2 + lab_h + 0.06 + max(bh, thh) / 2
    gy = y_mid - bh / 2; yc = y_mid
    M.block(ax, rx, gy, gw, bh, None, fill=M.GRN)
    M.T(ax, rx + gw / 2, gy + bh - 0.20, r"$E\sqrt{\mathrm{TDD}}$", size=14.0, ha="center")
    M.T(ax, rx + gw / 2, gy + 0.16, "Stefan", size=FS_T, ha="center")
    px = rx + gw + g_plus + pr
    plus_node(ax, px, yc, r=pr, color=R1)
    yx = px + pr + g_plus
    M.block(ax, yx, gy, yw, bh, None, fill=M.YEL)
    tw, th, tg = 0.26, 0.20, 0.07
    tx0 = yx + (yw - (3 * tw + 2 * tg)) / 2
    for i in range(3):
        tree_glyph(ax, tx0 + i * (tw + tg), gy + bh - 0.08 - th, tw, th)
    M.T(ax, yx + yw / 2, gy + 0.16, "Residual ML", size=FS_T, ha="center")
    lx0 = yx + yw + g_line
    flow_chevron(ax, lx0 + (line_len - 0.17) / 2, yc, 0.17, 0.26)            # 흐름 셰브런(화살 선·화살촉 없음; 지침 v9 1.3 의 셰브런 허용)
    thx = lx0 + line_len + g_line
    thy = yc - thh / 2
    g14, lon_e, lat_e, _ = grids[2014]
    axt = geo_axes(fig, thx, thy, thw, thh, projA, extA, "50m")
    axt.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(g14), transform=ccrs.PlateCarree(), cmap=cm_alt, norm=norm, shading="flat",
                   rasterized=True, zorder=1)
    t_th = M.T(ax, thx + thw / 2, thy - 0.06, "ALT 2014", size=FS_T, ha="center", va="top")
    SPEC["choices"]["architecture_glyph_in"] = dict(row_x0=round(rx, 3), row_w=round(row_w, 3), block_h=bh, block_y=round(gy, 3),
                                                     thumb_xywh=[round(v, 3) for v in (thx, thy, thw, thh)], trees=3, connector="small flow chevron (no arrow line, no arrowhead)")
    caps.append(caption(ax, 1, 1, CAP["a4"]))

    # ---- 열 3 · 위: 알래스카 연도별 지도 3장(같은 색 범위). 지도 상자를 자료 비율에 맞추고 행 띠 안에서 가운데 둔다
    cb_w, cb_gap, cb_lab = 0.08, 0.07, 0.58       # 색 막대 폭 · 간격 · 눈금 글자 폭 + 세운 단위 글줄('ALT (cm)')
    gap = 0.07
    x_m = COLS[2][0] + 0.02
    mw3 = (COLS[2][1] - cb_lab - cb_w - cb_gap - x_m - 2 * gap) / 3
    hm = mw3 / aspA
    yr_lab_h = measure("2014", FS_T)[1]
    band_lo, band_hi = LAY["XT1_B"], LAY["CAP1_B"] - 0.05                       # 눈금 글자 바닥 ~ 설명 글줄 아래
    if hm + 0.06 + yr_lab_h > band_hi - band_lo:                               # 띠 높이에 맞춘다
        hm = band_hi - band_lo - 0.06 - yr_lab_h; mw3 = hm * aspA
    stack = hm + 0.06 + yr_lab_h
    map_b = band_lo + (band_hi - band_lo - stack) / 2
    SPEC["choices"]["alaska_maps_in"] = dict(w=round(float(mw3), 3), h=round(float(hm), 3), bottom=round(float(map_b), 3))
    map_axes = []
    for i, yr in enumerate(years):
        g, lon_e, lat_e, _ = grids[yr]
        xm = x_m + i * (mw3 + gap)
        axm = geo_axes(fig, xm, map_b, mw3, hm, projA, extA, "50m")
        axm.pcolormesh(lon_e, lat_e, np.ma.masked_invalid(g), transform=ccrs.PlateCarree(), cmap=cm_alt, norm=norm, shading="flat",
                       rasterized=True, zorder=1)
        if i == 0:
            sb_ln, sb_t = scale_bar_in(axm, 500, "500 km")
        M.T(ax, xm + mw3 / 2, map_b + hm + 0.06 + yr_lab_h / 2, str(yr), size=FS_T, ha="center")
        map_axes.append(axm)
    x_cb = x_m + 3 * mw3 + 2 * gap + cb_gap
    cax = M.sub_axes(fig, x_cb, map_b, cb_w, hm)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm_alt), cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(M.HAIR)
    cb.set_ticks([v for v in (40, 60, 80, 100) if lo <= v <= hi]); cax.tick_params(labelsize=FS_K, length=3, width=0.8, pad=2)
    t_cbl = M.T(ax, COLS[2][1] - 0.02, map_b + hm / 2, "ALT (cm)", size=FS_T, ha="right", va="center", rotation=90)   # 색 막대 눈금 오른쪽에 세워서
    caps.append(caption(ax, 2, 0, CAP["a5"]))

    # ---- 열 3 · 아래: 공통 셀(25 km 마스크) RMSE 가로 막대, 지역마다 물리 + 잔차 ML(위, 주황) 대 최선 제품(아래, 제품 토큰 색).
    #      막대 끝에 이름(첫 지역의 주황 막대 = '물리 + 잔차 ML', 제품 막대 = 제품 이름). 변수·단위는 설명 글줄 'RMSE (cm)'
    rows = rmse_table()
    x_off6 = max(measure(REG_KO[r_], FS_T)[0] for r_ in ("Alaska", "Lena", "Canada")) + (TICK_LEN + TICK_PAD + 4) / 72.0 + 0.04
    a6 = chart_axes(fig, 2, 1, xoff=x_off6)
    yy6 = {r_["region"]: float(2 - i) for i, r_ in enumerate(rows)}           # 알래스카 위
    bh6 = 0.34; off6 = bh6 / 2 + 0.03
    ax6_w = a6.get_position().width * W
    w_lab = max(measure(r_["product"], FS_T)[0] for r_ in rows)
    w_ours = measure("Physics + residual ML", FS_T)[0]
    vmax6 = max(max(r_["ours"] + 0 * w_ours, r_["prod_rmse"]) for r_ in rows)
    # x 상한: 가장 긴 (막대 + 0.05 in + 이름)이 축 안에 들어가게, 10 cm 단위
    need = [(r_["prod_rmse"], w_lab) for r_ in rows]
    X6 = 10.0
    while any(v + (0.05 + w) * X6 / ax6_w > X6 for v, w in need):
        X6 += 5.0
    a6.set_xlim(0, X6); a6.set_ylim(-0.62, 2.90)                                # 위 0.5 단위: 축 안 오른쪽 위 열쇠 자리
    t_prod, t_ours = [], None
    for i, r_ in enumerate(rows):
        y_ = yy6[r_["region"]]
        a6.barh(y_ + off6, r_["ours"], height=bh6, color=R1, lw=0, zorder=2)
        a6.barh(y_ - off6, r_["prod_rmse"], height=bh6, color=r_["product_hex"], lw=0, zorder=2)
        pad6 = 0.05 * X6 / ax6_w
        t_prod.append(a6.text(r_["prod_rmse"] + pad6, y_ - off6, r_["product"], ha="left", va="center", fontsize=FS_T, color=r_["product_hex"], zorder=5))

    # 주황 막대 열쇠: 축 안 오른쪽 위(알래스카 막대·이름보다 오른쪽의 빈 자리)
    lg6 = a6.legend([Patch(facecolor=R1, edgecolor="none")], ["Physics + residual ML"], loc="upper right", bbox_to_anchor=(1.0, 1.04), fontsize=FS_T,
                    frameon=False, handlelength=1.0, handleheight=0.8, handletextpad=0.4, borderaxespad=0.0, borderpad=0.0)
    t_ours = lg6.get_texts()[0]
    a6.set_yticks([yy6[r_["region"]] for r_ in rows]); a6.set_yticklabels([REG_KO[r_["region"]] for r_ in rows], fontsize=FS_T)
    for t_, r_ in zip(a6.get_yticklabels(), rows):
        t_.set_color(S.REGION_HEX[r_["region"]])
    a6.tick_params(axis="y", length=0, pad=4)
    a6.spines["left"].set_visible(False)
    a6.set_xticks(np.arange(0, X6 + 1e-9, 10.0 if X6 <= 50 else 20.0))
    SPEC["choices"]["rmse_bars"] = dict(xlim=[0, X6], orientation="horizontal", labels="bar-end names; first orange bar labelled '물리 + 잔차 ML'",
                                        unit="caption 'RMSE (cm)'")
    caps.append(caption(ax, 2, 1, CAP["a6"]))

    # ---- 점검: 세로 간격, 라벨–선·자료, 범례–자료, 제품 이름 사이, 여백, y축 라벨 위치·회전, 글자 크기(숫자 눈금 12 pt · 그 밖 13 pt)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    pos = {}; ylabels = []
    for a_, (x0, _) in ((a1, COLS[0]), (a2, COLS[0]), (a3, COLS[1])):
        t = a_.yaxis.get_label(); bb = t.get_window_extent(r)
        pos[t.get_text()] = dict(x0_in=round(bb.x0 / fig.dpi, 3), column_left_in=x0, inside_column=bool(bb.x0 / fig.dpi >= x0 - 0.02),
                                 top_in=round(bb.y1 / fig.dpi, 3), bottom_in=round(bb.y0 / fig.dpi, 3))
        ylabels.append(t.get_text())
    cap_bb = {t.get_text(): bbox_in(fig, t) for t in caps}
    # 1행 눈금 글자 바닥 ↔ 2행 설명 글줄 위
    xt_bot = min(min(bbox_in(fig, t)[1] for t in a_.get_xticklabels() if t.get_text()) for a_ in (a1, a3))
    cap2_top = max(cap_bb[CAP[k]][3] for k in ("a2", "a4", "a6"))
    qa_extra["xticks1_to_caption2_gap_in"] = round(xt_bot - cap2_top, 3)
    # y축 제목(회전) 위 끝 ↔ 설명 글줄 바닥
    qa_extra["ylabel_top_to_caption_gap_in"] = dict(row1=round(min(cap_bb[CAP[k]][1] for k in ("a1", "a3")) - max(pos[a_.yaxis.get_label().get_text()]["top_in"] for a_ in (a1, a3)), 3),
                                                    row2=round(cap_bb[CAP["a2"]][1] - pos[a2.yaxis.get_label().get_text()]["top_in"], 3))
    # 결합 도식: 설명 글줄 바닥 ↔ 블록 위, 썸네일 글줄 바닥 ↔ 캔버스 바닥
    qa_extra["glyph_gaps_in"] = dict(caption_to_block=round(cap_bb[CAP["a4"]][1] - (gy + max(bh, thh) / 2 + max(bh, thh) / 2 - (max(bh, thh) - bh) / 2), 3),
                                     thumb_label_bottom=round(bbox_in(fig, t_th)[1], 3))
    # 열 2 위: 라벨 상자와 곡선·파선·안내선
    inv = a3.transData.inverted()

    def box_data(t):
        b = bbox_in(fig, t)
        (x0d, y0d), (x1d, y1d) = inv.transform([(b[0] * fig.dpi, b[1] * fig.dpi), (b[2] * fig.dpi, b[3] * fig.dpi)])
        return x0d, y0d, x1d, y1d

    def curve_hits(t):
        x0d, y0d, x1d, y1d = box_data(t)
        xs_ = np.linspace(max(x0d, 0), min(x1d, 366), 300); ys_ = np.interp(xs_, x, y)
        return bool(((ys_ > y0d) & (ys_ < y1d)).any())

    def hline_hits(t, yv, x_from=0.0, x_to=XMAX):
        x0d, y0d, x1d, y1d = box_data(t)
        return bool(y0d < yv < y1d and x1d > x_from and x0d < x_to)
    qa_extra["tdd_labels"] = {lab: dict(curve=curve_hits(t), clim_line=hline_hits(t, clim), guide=hline_hits(t, v_obs, 0, doy),
                                        obs_guide_vertical=bool(box_data(t)[0] < doy < box_data(t)[2] and box_data(t)[1] < v_obs),
                                        top_gap_in=round(LAY["AX1_T"] - bbox_in(fig, t)[3], 3))
                              for lab, t in (("평년 기후 입력", t_cl), ("관측 연도 기후 입력", t_ob), ("2014", t_14))}
    # 열 1 아래 범례 상자 안에 자료점이 있는지 · 범례 왼쪽과 y축 간격
    lb = bbox_in(fig, lg2); inv2 = a2.transData.inverted()
    (lx0, ly0), (lx1, ly1) = inv2.transform([(lb[0] * fig.dpi, lb[1] * fig.dpi), (lb[2] * fig.dpi, lb[3] * fig.dpi)])
    qa_extra["lena_legend_data_inside"] = int(sum(1 for xx_, yy_ in pts2 if lx0 <= xx_ <= lx1 and ly0 <= yy_ <= ly1))
    qa_extra["lena_legend_left_gap_in"] = round(lb[0] - bbox_in(fig, a2)[0], 3)
    # 열 3 아래: 막대 끝 이름이 축 오른쪽 끝 안인지
    a6b = bbox_in(fig, a6)
    qa_extra["bar_labels_right_margin_in"] = round(a6b[2] - max(bbox_in(fig, t)[2] for t in t_prod), 3)
    lb6 = bbox_in(fig, lg6); al_lab = bbox_in(fig, t_prod[0])
    al_bar_top = a6.transData.transform((0, yy6["Alaska"] + off6 + bh6 / 2))[1] / fig.dpi
    qa_extra["legend6_bottom_to_alaska_items_in"] = round(lb6[1] - max(al_bar_top, al_lab[3]), 3)   # 열쇠는 알래스카 막대·이름 위(세로 간격)
    qa_extra["legend6_bottom_to_lena_label_in"] = round(lb6[1] - bbox_in(fig, t_prod[1])[3], 3)
    # 지도: 축척 막대 글자·선이 지도 틀 안인지
    mb = bbox_in(fig, map_axes[0]); sbt = bbox_in(fig, sb_t)
    qa_extra["scalebar_inside_map"] = bool(sbt[0] >= mb[0] + 0.01 and sbt[2] <= mb[2] - 0.01 and sbt[1] >= mb[1] + 0.01 and sbt[3] <= mb[3] - 0.01)
    cbl = bbox_in(fig, t_cbl)
    cb_ticks_right = max(bbox_in(fig, t)[2] for t in cax.get_yticklabels() if t.get_text())
    qa_extra["cbar_label_to_ticks_gap_in"] = round(cbl[0] - cb_ticks_right, 3)
    # 여백
    heads = [t for t in ax.texts if t.get_text() in HEADS]
    qa_extra["top_margin_in"] = round(H - max(bbox_in(fig, t)[3] for t in heads), 3)
    ticks = tick_label_artists(fig)
    below13 = []; kor_below13 = []
    kor = re.compile("[가-힣]")
    for t in fig.findobj(Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        if kor.search(t.get_text()) and t.get_fontsize() < FS_T - 1e-6:
            kor_below13.append([t.get_text()[:30], float(t.get_fontsize())])
        if t.get_fontsize() < FS_T - 1e-6:
            if (id(t) in ticks or t is sb_t) and t.get_fontsize() >= FS_K - 1e-6:
                continue
            below13.append([t.get_text()[:30], float(t.get_fontsize())])
    no_hangul(fig)
    rec = M.qa_and_save(fig, STEM, allow_overlap=())
    png = M.OUT / f"{STEM}_slide.png"
    from PIL import Image
    with Image.open(png) as im:                      # RGBA → RGB(흰 바탕) 로 평탄화, 화소 크기 확인
        if im.mode != "RGB":
            bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
            bg.save(png, dpi=(M.DPI, M.DPI))
        px = list(im.size)
    with Image.open(png) as im:                      # 잉크 여백(화소): 흰색이 아닌 첫 행·열
        a = np.asarray(im.convert("L")); ys, xs = np.where(a < 250)
        ink_px = dict(top=int(ys.min()), bottom=int(a.shape[0] - 1 - ys.max()), left=int(xs.min()), right=int(a.shape[1] - 1 - xs.max()))
    SPEC["qa"] = {k: rec[k] for k in ("min_font_pt", "texts", "outside", "overlaps", "small", "text_line", "text_axes", "text_patch")}
    SPEC["qa"]["png_px"] = px; SPEC["qa"]["png_px_ok"] = bool(px == [int(W * M.DPI), int(H * M.DPI)])
    SPEC["qa"]["ink_margin_px"] = ink_px; SPEC["qa"]["ink_margin_in"] = {k: round(v / M.DPI, 3) for k, v in ink_px.items()}
    SPEC["qa"]["non_tick_text_below_13pt"] = below13
    SPEC["qa"]["korean_text_below_13pt"] = kor_below13
    SPEC["qa"]["ylabel_position"] = pos
    SPEC["qa"]["ylabel_rotation"] = ylabel_rotation_check(ylabels)
    SPEC["qa"]["bare_sqrt_glyph_texts"] = [t.get_text()[:40] for t in fig.findobj(Text) if "√" in t.get_text()]
    SPEC["qa"].update(qa_extra)
    bad = [k for k, v in SPEC["qa"]["ylabel_rotation"].items() if not v["ok"]] + [k for k, v in pos.items() if not v["inside_column"]]
    tdd_ok = all(not any(v[k] for k in ("curve", "clim_line", "guide", "obs_guide_vertical")) and v["top_gap_in"] >= 0.0 for v in qa_extra["tdd_labels"].values())
    gaps_ok = qa_extra["xticks1_to_caption2_gap_in"] >= 0.08 and min(qa_extra["ylabel_top_to_caption_gap_in"].values()) >= 0.02 and \
        qa_extra["top_margin_in"] >= 0.08 and tdd_ok and qa_extra["lena_legend_data_inside"] == 0 and qa_extra["lena_legend_left_gap_in"] >= 0.08 and \
        qa_extra["bar_labels_right_margin_in"] >= 0.0 and qa_extra["scalebar_inside_map"] and qa_extra["legend6_bottom_to_alaska_items_in"] >= 0.06 and \
        qa_extra["legend6_bottom_to_lena_label_in"] >= 0.03 and \
        qa_extra["cbar_label_to_ticks_gap_in"] >= 0.04
    margins_ok = all(v >= 0.06 for v in SPEC["qa"]["ink_margin_in"].values())
    print("y축 라벨 점검:", "OK" if not bad else f"CHECK {bad}", flush=True)
    print("간격 점검:", "OK" if gaps_ok else "CHECK", json.dumps(qa_extra, ensure_ascii=False), flush=True)
    print("잉크 여백(in):", "OK" if margins_ok else "CHECK", SPEC["qa"]["ink_margin_in"], flush=True)
    print("글자 크기 점검(숫자 눈금 외 13 pt 미만):", "OK" if not below13 else f"CHECK {below13}",
          "· 한글 13 pt 미만:", "OK" if not kor_below13 else f"CHECK {kor_below13}", "· 글리프 √:", SPEC["qa"]["bare_sqrt_glyph_texts"] or "없음", "· 화소", px, flush=True)
    (M.OUT / f"{STEM}_spec.json").write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
    return rec


def no_hangul(fig):
    """그림 안 모든 글자 객체(눈금 글자 포함)에 한글이 있으면 멈춘다(v10: 그림 글자는 영어만)."""
    bad = [t.get_text() for t in fig.findobj(Text) if re.search("[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", t.get_text())]
    if bad:
        raise RuntimeError(f"한글 글자 남음: {bad[:8]}")


def calendar_doy(year, month):
    """그 달 1일의 연중 일수(1월 1일 = 0; tdd_curve 의 일 격자와 같은 원점)."""
    return float(sum(calendar.monthrange(year, m)[1] for m in range(1, month)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", action="store_true", help="호환용. 이 그림은 슬라이드판 하나만 만든다")
    ap.add_argument("--fig", default="G1", help="호환용")
    ap.parse_args()
    M.OUT.mkdir(parents=True, exist_ok=True)
    g1()
    print("출력:", M.OUT / f"{STEM}_slide.png", M.OUT / f"{STEM}_slide.pdf", M.OUT / f"{STEM}_spec.json")
