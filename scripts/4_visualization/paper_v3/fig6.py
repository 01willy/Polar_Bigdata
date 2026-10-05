"""Fig 6 v3: 라벨 0 전이와 예측 구간(재구성 비교판).

근거
  outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 8절(역할, 배치 170 × 160 mm, 패널 명세, 설명문 초안), 1절(공통 규격),
  12절 XG 자리표시(c 의 묶음 'Existing ALT maps'; 높이는 그 묶음 3줄만큼 늘려 170 × 169.1 mm),
  11절(판정 기호 대체), 14절(D-9, D-10, D-16, D-17)
  design/journal_grade_style_guide.md 2절(그림), 6.6절(Fig 6 계획), 7.1절(점검), 부록 A.1 audit_v3, A.2 pdf_audit
  공용 양식: scripts/4_visualization/paper_v3/style.py(다른 그림 모듈과 공유, 이 파일은 고치지 않는다)

패널
  a, b  라벨 0 대상별 오차 변화 지도(직접 ML, 물리 유사라벨 증강), 원천 계수 Stefan 대비, 사후 서술. 범북극 평사 투영,
        Fig 1a 와 같은 범위(50° N 원, 중심 경도 126.698° E), 대상 원 지름 3 mm 고정, 겹치는 원은 지시선 부채꼴
  c     라벨 0 직접 ML 학습기와 물리 기준선의 원천 계수 Stefan 대비, 주 4지역 층화 평균, 두 가중 95 % CI 와 ±0.5 cm 띠.
        묶음 'Existing ALT maps'(XG-1c CCI v5 주 4지역, XG-1w Wei v2 레나·캐나다, 5 km 블록 마스크). 축 밖 값은 가장자리 화살표와 값
  d     라벨 0 90 % 예측 구간의 포함률 대 평균 폭, 구간 척도화 5종(주 4지역 평균과 지역 값)
  e     라벨 40개와 160개 앵커 + 잔차 구간의 포함률 대 라벨 0 대비 폭 비

자료(읽기 전용, 그림 명세 8.3 의 등록 원천)
  a, b  scripts/2_evaluation/wf0_reanalysis.py 의 load(), deltas() 를 불러 n == 0 의 D0_1.0, D1_1.0
        (입력 results/rescale_lg/data/processed/lg/lg_curve.csv, data/processed/lgd/lgd_curve_lic.csv)
        대상 중심 = 대상 라벨 셀의 위·경도 평균. 셀 정의는 LG 하네스(scripts/3_deep_learning/h40_label_grid.py)의
        Data.target_idx 와 같다: 지역은 polar.m1_core.load_base(fidelity_base_v3.csv) 의 macro, 하위 지역은
        h40.apply_subregion_map(lg_subregion_map_v1.csv). LGD 대상(Russia_C, Tibet_LGD)은 LGD 가 실행한 대상 표
        data/processed/lgd/run_tables/<spec>/fidelity_base_v3.csv 의 macro.
        바탕: 영구동토 구역 data/processed/cci_pfr_mean_1997_2021.nc(ESA CCI Permafrost PFR v4.0, 1997–2021 평균),
        육지 Natural Earth 50 m(cartopy 내장 자료).
  c     data/processed/paper_figs/fig6_a.csv(본문 14행) + data/processed/xbatch/XG_product_comparison/sealed/xg_tests.csv(2행, 계획 8.7)
  d     data/processed/paper_figs/fig6_b.csv(method ∈ const, phys, nflow, nflow#placebo, cbq; kind ∈ region, mean4; 1단 CI 열 *_lo1, *_hi1)
  e     data/processed/paper_figs/fig6_d.csv

산출(outputs/figures/paper/v3_restructure/): Fig6.pdf(벡터, 글꼴 내장), Fig6.png(600 dpi), Fig6_source_data.csv, Fig6_values.txt
설명문 Fig6_legend.md 는 손으로 쓴 문서이고, 이 스크립트는 그 단어 수, 금지 표현, 수치 문자열만 점검한다.

실행: OMP_NUM_THREADS=2 nice -n 10 python scripts/4_visualization/paper_v3/fig6.py            (정본, v3_restructure 에 저장)
      OMP_NUM_THREADS=2 nice -n 10 python scripts/4_visualization/paper_v3/fig6.py --draft --out <폴더>   (PNG 만, 점검 요약 출력)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402

ROOT = S.ROOT
for _p in (ROOT / "src", ROOT / "scripts" / "2_evaluation", ROOT / "scripts" / "3_deep_learning"):
    sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
import matplotlib.path as mpath  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

PROC = ROOT / "data" / "processed"
OUT = S.OUT
STEM = "Fig6"
XG_EXTRA_MM = 9.1          # c 에 묶음 'Existing ALT maps'(머리 1줄 + 행 2개 + 묶음 간격 0.35)를 기존 행 간격(2.715 mm)으로 더한 높이
W_MM, H_MM = S.W2_MM, 160.0 + XG_EXTRA_MM

# ---------------------------------------------------------------- 배치(그림 명세 8.2, mm, 그림 왼쪽 위 원점: x, 위끝 y, 폭, 높이)
SLOT = dict(a=(0.0, 0.0, 82.0, 76.0), b=(88.0, 0.0, 82.0, 76.0), cbar=(30.0, 80.0, 110.0, 2.5),
            c=(0.0, 92.0, 96.0, 64.0 + XG_EXTRA_MM), d=(104.0, 92.0, 66.0, 33.5 + XG_EXTRA_MM / 2),
            e=(104.0, 128.5 + XG_EXTRA_MM / 2, 66.0, 27.5 + XG_EXTRA_MM / 2))
# c 의 행 간격을 그대로 두려고 c 슬롯을 XG_EXTRA_MM 만큼 늘렸고, 오른쪽 d·e 는 아래 빈 띠가 생기지 않게 그 높이를 반씩 나눠 가진다[판단]
# d, e 높이는 명세(각 30 mm)에서 d 33.5, e 27.5 mm 로 바꿨다[판단]: d 의 두 줄 직접 라벨(상수·물리 사전)이 CI 막대 위에 들어가려면 축 높이 22 mm 가 필요하다
EDGE_MM = 0.3               # 캔버스 가장자리에서 글자를 띄우는 거리(패널 문자, 묶음 머리, 열 머리)
HEAD_MM = 3.6               # 지도 열 머리(7 pt 한 줄) 높이
MAP_D_MM = 70.0             # 지도 원(50° N) 지름 [판단]: 슬롯 82 × 76 에서 열 머리를 빼고 왼쪽 아래 구석(티베트 삽도)과 왼쪽 위 구석(영구동토 견본)을 비운다
INSET_W_MM, INSET_H_MM = 17.0, 13.0       # 티베트 삽도(슬롯 왼쪽 아래 구석)
AK_INSET_W_MM, AK_INSET_H_MM = 20.0, 25.0  # 알래스카 확대도(슬롯 오른쪽 아래 구석) [판단: 지침 2.11 '밀집 구역은 확대도', Hjort 2018 Fig 1]
# 폭 20 mm: 23 mm 에서는 확대도 틀이 동시베리아 원에 0.07 mm, 확대 사각형이 캐나다 하위 지역 원에 0.03 mm 까지 붙었다(1차 렌더).
# 축척은 남북 방향(대상 범위 20.6 mm)이 정하므로 폭을 줄여도 확대 배율은 같다
AK_PAD_MM = 2.2             # 확대도 안 가장 바깥 대상 중심과 틀 사이 여백
FOREST_NAME_MM = 40.0       # c 행 이름 열 폭(그림 명세 8.2)

# ---------------------------------------------------------------- 지도(지침 2.11, Fig 1a 와 같은 범위)
TRUE_LAT, LAT_MIN = 70.0, 50.0
LON0 = 126.698              # Fig 1a 의 중심 경도(fig1.py zoom_projection: 레나델타 라벨 셀 경도 범위의 중앙값. Fig1_values.txt 에 기록된 값)
LAT_LABEL_LON = -22.0       # 위도 라벨을 다는 경선 [판단: v4 map_base 로 육지가 진해져 −10°(그린란드 동해안)에서 그린란드해(−22°)로 옮겼다]
SCALE = dict(lon=-2.0, lat=66.0, km=1000)      # 축척 막대 중심(노르웨이해 남부, 66° N) [판단: 70° N·12° E 는 v4 에서 아이슬란드·노르웨이 해안과 겹쳐 보여 옮겼다]
PFR_MIN_PPI = 450
PC = ccrs.PlateCarree()
PROJ = ccrs.NorthPolarStereo(central_longitude=LON0, true_scale_latitude=TRUE_LAT)

# ---------------------------------------------------------------- 대상(그림 명세 8.3, 결정 D-9: 모드 x 17대상)
MACRO7 = ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Tibet_LGD"]
SUB10 = ["AL-1", "AL-2", "AL-3", "AL-4", "AL-5", "AL-6", "CA-2", "CA-3", "LE-1", "LE-2"]
LGD_RUN_TABLE = {"Russia_C": "Russia_C", "Tibet_LGD": "Tibet"}          # LGD 실행 명세 이름(data/processed/lgd/run_tables/)
INSET_TARGETS = ["Tibet_LGD"]                                           # 범북극 투영 밖(지침 2.11)
AK_TARGETS = ["Alaska", "AL-1", "AL-2", "AL-3", "AL-4", "AL-5", "AL-6"]  # 확대도에 그리는 알래스카 대상 7개(본 지도에는 확대 범위 사각형만)
SUB_PARENT = {"AL": "Alaska", "CA": "Canada", "LE": "Lena"}
CIRCLE_D_MM = 3.0           # 대상 원 지름 고정(그림 명세 8.3)
CIRCLE_GAP_MM = 3.5         # 원 중심 사이 최소 거리(겹침 해소 기준)
LEADER_MIN_MM = 0.5         # 이보다 많이 옮긴 원은 지시선과 실제 위치 점을 그린다
LEADER = "#737373"          # 지시선 회색(1.0 pt)
MAP_CLEAR_MM = 0.5          # 대상 원 테두리와 다른 선(확대 사각형, 틀, 연결선, 다른 원의 지시선) 사이 최소 간격
LABEL_PAD_MM = 0.5          # 직접 라벨 글상자와 자료 표지 사이 최소 간격
MAP_COLS = {"a": ("D0_1.0", "Direct ML"), "b": ("D1_1.0", "Physics pseudo-labels")}

# ---------------------------------------------------------------- c 포레스트 행(그림 명세 8.3; 행 이름은 4 단어 이하)
# (block, key, variant) 는 fig6_a.csv 의 키. 'CatBoost, larger' 는 명세의 'CatBoost, default' 를 바꾼 이름이다[판단]:
# catboost 는 기본값이 아니라 600회·깊이 6 설정으로 주 학습기(200회·깊이 3)보다 용량이 크다(polar.m1_core 287–295행, C1 README 2.1 '용량 확대').
FOREST = [
    ("Physics baselines", [("baselines", "P*", "solid", "Year-matched Stefan"),
                           ("baselines", "B:ens", "solid", "Anchor ensemble")]),
    ("Boosting and random forest", [("direct_rescale", "catboost_lo", "solid", "CatBoost, main"),
                                    ("direct_rescale", "catboost", "solid", "CatBoost, larger"),
                                    ("direct_rescale", "catboost_tuned", "solid", "CatBoost, tuned"),
                                    ("direct_rescale", "rf", "solid", "Random forest"),
                                    ("direct_rescale", "F1k", "solid", "CatBoost, physics inputs")]),
    ("Tabular foundation models", [("lgt", "tabpfn", "solid", "TabPFN v2"),
                                   ("lgf_f", "tabicl", "solid", "TabICL v2"),
                                   ("lgf_f", "tabicl", "dashed", "TabICL v2, full context")]),
    ("Neural networks", [("lgf_n", "mlp", "solid", "MLP"),
                         ("lgf_n", "tabm", "solid", "Multi-head MLP"),
                         ("lgf_n", "ftt", "solid", "FT-Transformer, reduced"),
                         ("lgf_n", "realmlp", "solid", "RealMLP")]),
]
FOREST_XLIM, FOREST_XTICKS = (-3.6, 7.6), (-2, 0, 2, 4, 6)
# 기존 ALT 지도 묶음(그림 명세 8.3·12절 'Existing ALT maps', 계획 2.7 XG-1 행, 결과 8.7). 판정 표 xg_tests.csv 의 주 열(5 km 블록 마스크).
# (block, key, variant, 행 이름): key = 가설 ID. Wei v2 는 레나델타·캐나다 2지역 풀이라 행 이름에 적는다
FOREST_XG = ("Existing ALT maps", [("xg", "XG-1c", "solid", "ESA CCI v5"), ("xg", "XG-1w", "solid", "Wei v2, two regions")])
# v4 토큰: 포레스트 행 색(행 이름 → 색). 직접 ML 학습기는 직접 ML 색, 물리 입력 CatBoost 는 물리 입력 ML 색, 기준선과 제품은 각자의 색
FOREST_COLOR = {"Year-matched Stefan": S.METHOD["year_matched_stefan"]["color"], "Anchor ensemble": S.METHOD["stefan_cci_anchor"]["color"],
                "CatBoost, physics inputs": S.METHOD["physics_input"]["color"],
                "ESA CCI v5": S.PRODUCT["CCI"]["color"], "Wei v2, two regions": S.PRODUCT["Wei"]["color"]}
for _g, _items in FOREST:
    for *_x, _lab in _items:
        FOREST_COLOR.setdefault(_lab, S.METHOD["direct_ml"]["color"])
XG_POOL_NAME = {"Lena|x,Canada|x,Russia_W|x,Russia_E|x": "four-region stratified mean", "Lena|x,Canada|x": "two-region mean (Lena Delta, Canada)"}

# ---------------------------------------------------------------- d, e(그림 명세 8.3)
D_METHODS = [("const", "Constant"), ("phys", "Physics prior"), ("nflow", "Normalizing flow"),
             ("nflow#placebo", "Permuted flow"), ("cbq", "CatBoost quantile")]
D_COLOR = {m: S.INTERVALS[name] for m, name in D_METHODS}                  # v4 토큰 color.intervals(구간 방법마다 색)
E_COLOR = S.INTERVALS["Conformal"]
D_REGIONS = ["Lena", "Canada", "Russia_W", "Russia_E"]
E_REGIONS = ["Alaska", "Canada", "Lena"]
COV_YLIM, COV_YTICKS = (0.70, 1.00), (0.7, 0.8, 0.9, 1.0)
D_XLIM, D_XTICKS = (64.0, 160.0), (80, 100, 120, 140)
E_XLIM, E_XTICKS = (0.52, 1.02), (0.6, 0.7, 0.8, 0.9, 1.0)
NOMINAL = 0.90
D_NBOOT, E_NBOOT = 1000, 10000   # d 1단 CI(lgu_b_meta.json nboot), e 포함률 CI(lgx_meta.json nboot). source_checks 9 에서 대조
E_BAND = (0.85, 0.95)
KEY_REGIONS = ["Alaska", "Canada", "Lena", "Russia_W", "Russia_E"]
# d 직접 라벨 위치(자료 좌표: 폭 cm, 포함률, 정렬). 거의 겹치는 두 평균(상수·물리 사전, 정규화 흐름·순열 흐름)은
# 두 줄로 쌓은 라벨 한 묶음과 지시선 하나로 가리킨다(위 줄 = 포함률이 높은 쪽)
D_LABEL_POS = {"phys": (67.0, 0.993, "left"), "const": (67.0, 0.958, "left"),
               "nflow": (101.0, 0.800, "left"), "nflow#placebo": (101.0, 0.760, "left"),
               "cbq": (143.0, 0.845, "center")}
# 지시선 시작점: 라벨 글상자의 모서리·변 위 위치(가로 비율, 세로 비율; 0 = 왼쪽/아래, 1 = 오른쪽/위)와 가리키는 평균.
# CI 막대 십자와 겹치지 않게 대각선으로 들어간다. 상수·물리 사전 묶음은 x 67 cm 에서 시작한다[판단]: 65 cm 이면 물리 사전 지시선이
# 동시베리아 지역 점(84.3 cm, 0.926)을 0.24 mm 로 스치고, 묶음 위 줄이 축 위끝을 0.5 mm 남짓 넘는 것은 위 테두리가 없어 보이지 않는다
D_LEADERS = [("phys", (1.0, 0.0), "phys", 1.0), ("nflow", (0.0, 1.0), "nflow", 2.0), ("cbq", (0.55, 1.0), "cbq", 1.0)]
# 지시선 끝과 평균 중심 사이(mm). 정규화 흐름은 2.0 mm[판단]: 평균 0.3–0.5 mm 옆에 캐나다 지역 점 두 개(상수, 물리 사전)가 있어
# 1.0 mm 에서 멈추면 지시선 끝이 그 점에 0.1 mm 까지 붙는다. 2.0 mm 는 CI 십자의 바깥 끝 근처다
D_PAIR = {"phys": "const", "nflow": "nflow#placebo"}     # 두 줄 라벨 묶음(지시선 하나가 두 평균을 함께 가리킨다)
COV_LABEL_X_MM = 100.2      # 공유 y 축 이름("Coverage")의 x(c 의 축 오른쪽 끝 94 mm 와 d 패널 문자 104 mm 사이)

# ---------------------------------------------------------------- 원천 파일(값 대조용)
LG_CURVE = ROOT / "results" / "rescale_lg" / "data" / "processed" / "lg" / "lg_curve.csv"
LGD_CURVE = PROC / "lgd" / "lgd_curve_lic.csv"
WF0_RISK = PROC / "wf" / "wf0_risk.csv"
LGW_BUNDLE = PROC / "lgw" / "lgw_bundle.csv"
LGX_TESTS = PROC / "lgx" / "lgx_tests.csv"
LGT_TESTS = PROC / "lgt" / "lgt_tests.csv"
LGF_TESTS = PROC / "lgf" / "lgf_tests.csv"
LGFN_TESTS = PROC / "lgf" / "lgfn_tests.csv"
LGU_B = PROC / "lgu" / "lgu_b_intervals.csv"
LGX_CONFORMAL = PROC / "lgx" / "lgx_conformal.csv"
TABLE1_ROWS = S.PAPER_FIGS / "table1_rows.csv"
PFR_NC = PROC / "cci_pfr_mean_1997_2021.nc"
C1_README = ROOT / "paper" / "claims" / "C1_label0_safety" / "README.md"
CAPTIONS_V2 = ROOT / "outputs" / "figures" / "paper" / "CAPTIONS.md"
LGU_B_META = PROC / "lgu" / "lgu_b_meta.json"
LGX_META = PROC / "lgx" / "lgx_meta.json"
FIG1_VALUES = OUT / "Fig1_values.txt"
LEGEND_MD = OUT / f"{STEM}_legend.md"
MEAN4 = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"
XG_TESTS = PROC / "xbatch" / "XG_product_comparison" / "sealed" / "xg_tests.csv"
XG_META = PROC / "xbatch" / "XG_product_comparison" / "sealed" / "xg_summary_meta.json"
FOREST_SOURCE = {"lgw_bundle.csv": LGW_BUNDLE, "lgx_tests.csv": LGX_TESTS, "lgt_tests.csv": LGT_TESTS,
                 "lgf_tests.csv": LGF_TESTS, "lgfn_tests.csv": LGFN_TESTS, "xg_tests.csv": XG_TESTS}


# ================================================================ 공용
def check_resources(min_gb=None, max_load=40.0, wait_s=60, max_wait_s=3600):
    min_gb = float(os.environ.get("PAPER_MIN_AVAIL_GB", "30")) if min_gb is None else min_gb   # 기본 30 GB(작업 지시), 조정 담당이 정한 값은 환경 변수로
    """공유 서버 자원 확인(작업 지시). 가용 메모리 < 30 GB 또는 1분 부하 > 40 이면 기다린다."""
    t0 = time.time()
    while True:
        avail = [ln for ln in Path("/proc/meminfo").read_text().splitlines() if ln.startswith("MemAvailable")][0]
        gb = int(avail.split()[1]) / 1024 ** 2
        load = os.getloadavg()[0]
        if gb >= min_gb and load <= max_load:
            return dict(available_gb=round(gb, 1), load1=round(load, 1))
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"자원 부족이 {max_wait_s} s 넘게 이어짐(가용 {gb:.0f} GB, 부하 {load:.0f})")
        print(f"대기: 가용 {gb:.0f} GB, 부하 {load:.0f}", flush=True)
        time.sleep(wait_s)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def _minus(s: str) -> str:
    return s.replace("-", S.MINUS)


def _fmt_tick(nd: int):
    return FuncFormatter(lambda v, _p: _minus(f"{v:.{nd}f}"))


def haversine(la1, lo1, la2, lo2, R=6371.0):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ================================================================ 자료
def load_target_deltas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """wf0_reanalysis.load(), deltas() 를 읽기 전용으로 불러 n == 0 의 대상별 Δ(방법 − 원천 계수 Stefan, cm).
    반환 (30대상 전체, 지도 17대상)."""
    import wf0_reanalysis as W
    R = W.deltas(W.load())
    r0 = R[R.n == 0].copy()
    r0["name"] = r0.target.str.split("|").str[0]
    r0["mode"] = r0.target.str.split("|").str[1]
    if len(r0) != 30:
        raise SystemExit(f"WF0 n == 0 대상 {len(r0)}/30")
    keep = [f"{t}|x" for t in MACRO7 + SUB10]
    m = r0[r0.target.isin(keep)].copy()
    if len(m) != 17:
        raise SystemExit(f"지도 대상 {len(m)}/17")
    m["kind"] = np.where(m.name.isin(MACRO7), "region", "subregion")
    m["region_name"] = [S.REGION_NAME[n] if n in MACRO7 else S.REGION_NAME[SUB_PARENT[n[:2]]] for n in m.name]
    m["display_name"] = [S.REGION_NAME[n] if n in MACRO7 else f"{S.REGION_NAME[SUB_PARENT[n[:2]]]} sub-region {n[3:]}" for n in m.name]
    return r0.reset_index(drop=True), m.reset_index(drop=True)


def target_centroids() -> pd.DataFrame:
    """대상 라벨 셀의 위·경도 평균과 셀 수(LG·LGD 하네스의 대상 정의와 같은 셀 집합)."""
    from polar.m1_core import load_base
    import h40_label_grid as H40
    df = load_base(PROC)
    df["sub"], _subs = H40.apply_subregion_map(df, PROC / "lg_subregion_map_v1.csv")
    rows = []
    for t in MACRO7 + SUB10:
        if t in LGD_RUN_TABLE:
            continue
        q = df[df.macro == t] if t in MACRO7 else df[df["sub"] == t]
        rows.append(dict(name=t, lat=float(q.lat.mean()), lon=float(q.lon.mean()), n_cells=int(len(q)),
                         cell_source="data/processed/fidelity_base_v3.csv (polar.m1_core.load_base)"
                         + ("" if t in MACRO7 else " + data/processed/lg_subregion_map_v1.csv (h40.apply_subregion_map)")))
    for t, spec in LGD_RUN_TABLE.items():
        d = load_base(PROC / "lgd" / "run_tables" / spec)
        q = d[d.macro == t]
        rows.append(dict(name=t, lat=float(q.lat.mean()), lon=float(q.lon.mean()), n_cells=int(len(q)),
                         cell_source=f"data/processed/lgd/run_tables/{spec}/fidelity_base_v3.csv (macro == {t})"))
    return pd.DataFrame(rows)


def v4_crosscheck() -> dict:
    """LGD 대상 셀을 fidelity_base_v4.csv 에서 직접 세어 run table 과 대조(기록용)."""
    from polar.fidelity import macro_region
    from polar.m1_core import add_group_keys
    v4 = add_group_keys(pd.read_csv(PROC / "fidelity_base_v4.csv", low_memory=False))
    v4["macro"] = macro_region(v4)
    out = {}
    for t in LGD_RUN_TABLE:
        q = v4[(v4.macro == t) & v4.alt_cm.notna() & v4.source_id.isin(["F4_direct", "F4_ext_direct"])]
        out[t] = dict(n=int(len(q)), lat=round(float(q.lat.mean()), 3), lon=round(float(q.lon.mean()), 3))
    return out


def load_forest() -> pd.DataFrame:
    a = pd.read_csv(S.PAPER_FIGS / "fig6_a.csv")
    rows = []
    for head, items in FOREST:
        for block, key, variant, label in items:
            q = a[(a.block == block) & (a.key == key) & (a.variant == variant)]
            if len(q) != 1:
                raise SystemExit(f"fig6_a: {block}/{key}/{variant} 행 {len(q)}")
            r = q.iloc[0]
            rows.append(dict(group=head, row_label=label, block=block, key=key, variant=variant, source=r.source,
                             hypothesis=r.hypothesis, contrast=r.contrast, ab=r.ab if isinstance(r.ab, str) else "",
                             delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             delta_beq=r.delta_beq, ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq, verdict4=r.verdict4,
                             ci_dependence=r.ci_dependence if isinstance(r.ci_dependence, str) else "", nboot=int(r.nboot)))
    return pd.DataFrame(rows)


def load_xg() -> pd.DataFrame:
    """기존 ALT 지도 행(FOREST_XG): 봉인 판정 표 xg_tests.csv(계획 8.7, 첫 열람 2026-10-05 03:45:02)의 주 열(5 km 블록 마스크).
    delta·ci_lo·ci_hi = 셀 가중, delta_blockeq·ci_lo_beq·ci_hi_beq = 블록 등가중. 25 km 마스크 값과 판정, 제품 결측 대체 비율은 기록용."""
    x = pd.read_csv(XG_TESTS)
    nboot = int(json.loads(XG_META.read_text(encoding="utf-8"))["nboot"])
    head, items = FOREST_XG
    rows = []
    for block, key, variant, label in items:
        q = x[x.hypothesis == key]
        if len(q) != 1:
            raise SystemExit(f"xg_tests: {key} 행 {len(q)}")
        r = q.iloc[0]
        if abs(float(r.delta) - float(r.delta_blk5)) > 1e-9:
            raise SystemExit(f"xg_tests: {key} 주 열 delta 가 5 km 마스크 값과 다르다")
        rows.append(dict(group=head, row_label=label, block=block, key=key, variant=variant, source="xg_tests.csv",
                         hypothesis=key, contrast=f"B:{r['product']}_raw − P0, n 0", ab="",
                         delta=float(r.delta), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi),
                         delta_beq=float(r.delta_blockeq), ci_lo_beq=float(r.ci_lo_beq), ci_hi_beq=float(r.ci_hi_beq),
                         verdict4=str(r.verdict4_blk5), ci_dependence="", nboot=nboot,
                         product=str(r["product"]), pool=str(r.pool), pool_name=XG_POOL_NAME[str(r.pool)],
                         verdict4_blk25=str(r.verdict4_blk25), p_holm_blk5=float(r.p_holm_blk5), p_holm_blk25=float(r.p_holm_blk25),
                         delta_blk25=float(r.delta_blk25), fill_frac_blk5=float(r.fill_frac_B_blk5), fill_frac_blk25=float(r.fill_frac_B_blk25),
                         rmse_p0_postmask_blk5=float(r.rmse_p0_postmask_blk5), blind=str(r.blind), deviation=str(r.deviation),
                         design=str(r.design)))
    return pd.DataFrame(rows)


def load_intervals() -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(S.PAPER_FIGS / "fig6_b.csv")
    b = b[b.method.isin([m for m, _ in D_METHODS]) & b.kind.isin(["region", "mean4"])].copy()
    if len(b) != 25 or set(b[b.kind == "region"].target) != set(D_REGIONS):
        raise SystemExit(f"fig6_b 행 {len(b)}/25, 지역 {sorted(b[b.kind == 'region'].target.unique())}")
    d = pd.read_csv(S.PAPER_FIGS / "fig6_d.csv")
    if len(d) != 6 or set(d.target) != set(E_REGIONS):
        raise SystemExit(f"fig6_d 행 {len(d)}/6")
    return b, d


def pfr_classes():
    import netCDF4 as nc
    with nc.Dataset(PFR_NC) as f:
        lat, lon, M = f["lat"][:].data.astype(float), f["lon"][:].data.astype(float), f["pfr_mean"][:].filled(np.nan)
        title = f.getncattr("title")
    return lat, lon, M, title


# ================================================================ 공용 그리기
def zero_line(ax, axis="x"):
    f = ax.axvline if axis == "x" else ax.axhline
    ln = f(0.0, color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], ls="-", zorder=1.5)
    ln.set_gid("ref|zero")


def equiv_band(ax, axis="x"):
    h = S.EQUIV_HALF_WIDTH_CM
    f = ax.axvspan if axis == "x" else ax.axhspan
    f(-h, h, color=S.EQUIV_BAND, lw=0, zorder=0.5).set_gid("band|equiv")


def mm_to_data_y(ax, dy_mm: float) -> float:
    h_mm = ax.get_position().height * ax.figure.get_size_inches()[1] * S.MM_PER_IN
    y0, y1 = ax.get_ylim()
    return dy_mm * abs(y1 - y0) / h_mm


def mm_to_data_x(ax, dx_mm: float) -> float:
    w_mm = ax.get_position().width * ax.figure.get_size_inches()[0] * S.MM_PER_IN
    x0, x1 = ax.get_xlim()
    return dx_mm * abs(x1 - x0) / w_mm


def leader_to(ax, start, target, stop_mm=1.0, gid="leader"):
    """라벨에서 자료 표지로 가는 지시선(1.0 pt 회색 직선). 표지 중심 stop_mm 앞에서 멈춘다."""
    to_mm = lambda p: ax.transData.transform(p) / ax.figure.dpi * S.MM_PER_IN  # noqa: E731
    a, b = np.asarray(to_mm(start)), np.asarray(to_mm(target))
    d = a - b
    b2 = b + d / np.hypot(*d) * stop_mm
    inv = ax.transData.inverted()
    pa, pb = inv.transform(a / S.MM_PER_IN * ax.figure.dpi), inv.transform(b2 / S.MM_PER_IN * ax.figure.dpi)
    ln = ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=LEADER, lw=1.0, zorder=5, solid_capstyle="butt")[0]
    ln.set_gid(gid)


def clean_axes(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for sp in ax.spines.values():
        sp.set_linewidth(S.LW["axis"])
    ax.tick_params(which="both", width=S.LW["tick"], length=S.TICK_LEN_PT, pad=1.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def overlay_axes(fig):
    """그림 전체를 mm 좌표(왼쪽 위 원점)로 덮는 투명 축. 열쇠, 지시 화살표, 패널 사이 글자에 쓴다."""
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.set_axis_off()
    ov.set_xlim(0, W_MM)
    ov.set_ylim(H_MM, 0)
    ov.set_zorder(20)
    return ov


# ================================================================ 지도 a, b
def map_rect(slot):
    """슬롯 안 지도 원의 정사각형(mm): 열 머리 아래, 오른쪽 맞춤(왼쪽 구석을 삽도·견본에 쓴다)."""
    x, y, w, h = slot
    return (x + w - MAP_D_MM, y + HEAD_MM + (h - HEAD_MM - MAP_D_MM) / 2, MAP_D_MM, MAP_D_MM)


def polar_axes(fig, rect_mm):
    ax = S.axes_mm(fig, *rect_mm, projection=PROJ)
    rho = abs(PROJ.transform_point(LON0, LAT_MIN, PC)[1])
    ax.set_xlim(-rho, rho)
    ax.set_ylim(-rho, rho)
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.set_boundary(mpath.Path(np.c_[np.sin(theta), np.cos(theta)] * 0.5 + 0.5), transform=ax.transAxes)
    ax.spines["geo"].set_edgecolor(S.BASEMAP["coast"])
    ax.spines["geo"].set_linewidth(1.0)
    ax.patch.set_facecolor(S.BASEMAP["sea"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=S.BASEMAP["land"], edgecolor="none", linewidth=0, zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=S.BASEMAP["coast"], facecolor="none", linewidth=S.LW["coast"], zorder=0.4)
    return ax, rho


def draw_pfr(ax, proj, w_mm, h_mm, P, zorder=0.3):
    """영구동토 2단계(연속 ≥ 90 %, 불연속 50–90 %)를 투영 격자에 최근접 표본화한 래스터. 내장 해상도 ≥ 450 ppi."""
    lat, lon, M, _ = P
    nx = int(np.ceil(w_mm / 25.4 * PFR_MIN_PPI * 1.05))
    ny = int(np.ceil(h_mm / 25.4 * PFR_MIN_PPI * 1.05))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y1, y0, ny))
    ll = PC.transform_points(proj, X, Y)
    LON, LAT = ll[..., 0], ll[..., 1]
    i = np.clip(np.round((LAT - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    j = np.clip(np.round((LON - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    v = M[i, j]
    v[(LAT < lat.min() - 0.05) | (LAT > lat.max() + 0.05) | ~np.isfinite(LAT)] = np.nan
    cls = np.full(v.shape, np.nan)
    cls[(v >= 50) & (v < 90)] = 0
    cls[v >= 90] = 1
    cm = ListedColormap([S.BASEMAP["discontinuous"], S.BASEMAP["continuous"]])
    ax.imshow(np.ma.masked_invalid(cls), extent=(x0, x1, y0, y1), origin="upper", cmap=cm, vmin=0, vmax=1,
              interpolation="nearest", transform=proj, zorder=zorder, rasterized=True)
    return dict(nx=nx, ny=ny, ppi=round(nx / (w_mm / 25.4), 0))


def graticule(ax):
    """경위선 1.0 pt #e0e0e0: 위도선 3개(60°, 70°, 80° N), 경도선 12개(30° 간격)(지침 2.3, 2.11)."""
    ax.gridlines(crs=PC, draw_labels=False, linewidth=S.LW["grid_map"], color=S.BASEMAP["graticule"],
                 xlocs=np.arange(-180, 180, 30), ylocs=[60, 70, 80], zorder=0.5)


def lat_labels(ax):
    out = []
    for la in (60, 70, 80):
        t = ax.annotate(f"{la}°N", xy=PROJ.transform_point(LAT_LABEL_LON, la, PC), xycoords="data",
                        xytext=(2.0, 1.0), textcoords="offset points", ha="left", va="bottom", fontsize=S.FONT_PT, zorder=6)
        t.set_gid("graticule")
        out.append(t)
    return out


def scale_bar(ax, proj, lon, lat, km, polar=True):
    """축척 막대(1.5 pt 검정)와 길이 글자(막대 위, 지침 2.11). 투영 길이 = km × 그 위도의 평사 투영 축척 계수."""
    k = (1 + np.sin(np.radians(TRUE_LAT))) / (1 + np.sin(np.radians(lat))) if polar else 1.0
    cx, cy = proj.transform_point(lon, lat, PC)
    L = km * 1000.0 * k
    x0 = cx - L / 2
    ln = ax.plot([x0, x0 + L], [cy, cy], color=S.INK, lw=S.LW["scale_bar"], solid_capstyle="butt", transform=proj, zorder=8)[0]
    ln.set_gid("scale_bar")
    t = ax.text(x0 + L / 2, cy, f"{S.fmt_int(km)} km", transform=proj, ha="center", va="bottom", zorder=8, fontsize=S.FONT_PT)
    t.set_gid("scale")
    ll = PC.transform_points(proj, np.array([x0, x0 + L]), np.array([cy, cy]))
    return dict(km=km, k=float(k), geodesic_km=float(haversine(ll[0, 1], ll[0, 0], ll[1, 1], ll[1, 0])), text=t)


def resolve_overlaps(P_mm: np.ndarray, center_mm: np.ndarray, r_map_mm: float, gap=CIRCLE_GAP_MM):
    """원(지름 3 mm) 중심 사이 거리가 gap 미만인 묶음을 푼다(그림 명세 8.3 '지시선 부채꼴').
    두 원 묶음은 잇는 선 위에서 맞선 방향으로 최소 이동. 세 원 이상 묶음은 묶음 중심에서 극 반대쪽(바깥쪽)으로 연 부채꼴에
    등각 배치하고, 부채꼴 위 순서는 부채꼴 방향에 수직인 축 위의 실제 위치 순으로 두어 지시선이 엇갈리지 않게 한다.
    끝으로 모든 쌍에 대해 간격을 다시 확인하고(완화 반복) 지도 원 안에 머물게 한다. 반환 (표시 위치, 이동 거리, 묶음 번호)."""
    n = len(P_mm)
    Q = P_mm.astype(float).copy()
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(P_mm[i] - P_mm[j])) < gap:
                parent[find(i)] = find(j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    cluster_id = np.zeros(n, dtype=int)
    for ci, members in enumerate(sorted(comps.values(), key=lambda m: min(m))):
        for k in members:
            cluster_id[k] = ci
        if len(members) == 1:
            continue
        c = P_mm[members].mean(0)
        if len(members) == 2:
            i, j = members
            d = P_mm[j] - P_mm[i]
            dist = float(np.hypot(*d))
            u = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
            Q[i] = c - u * gap / 2
            Q[j] = c + u * gap / 2
            continue
        u = c - center_mm
        u = u / max(float(np.hypot(*u)), 1e-9)
        ang0 = math.atan2(u[1], u[0])
        k_n = len(members)
        span = math.radians(min(160.0, 40.0 * (k_n - 1)))
        step = span / (k_n - 1)
        R = (gap / 2 + 0.05) / math.sin(step / 2)
        perp = np.array([math.cos(ang0 - math.pi / 2), math.sin(ang0 - math.pi / 2)])
        order = sorted(members, key=lambda k: float(np.dot(P_mm[k] - c, perp)))
        for r, k in enumerate(order):
            a = ang0 - (r - (k_n - 1) / 2) * step
            Q[k] = c + R * np.array([math.cos(a), math.sin(a)])
    r_max = r_map_mm - CIRCLE_D_MM / 2 - 0.6
    for _ in range(500):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = Q[j] - Q[i]
                dist = float(np.hypot(*d))
                if dist < gap - 1e-6:
                    u = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    s = (gap - dist) / 2 + 1e-3
                    Q[i] -= u * s
                    Q[j] += u * s
                    moved = True
        for i in range(n):
            v = Q[i] - center_mm
            r = float(np.hypot(*v))
            if r > r_max:
                Q[i] = center_mm + v * r_max / r
                moved = True
        if not moved:
            break
    disp = np.hypot(*(Q - P_mm).T)
    return Q, disp, cluster_id


def draw_targets(ax, rho, M, col, P_mm, Q_mm, disp, norm, cmap, panel):
    """대상 원(지름 3 mm, 검정 테두리 1.0 pt, broc 채움). 옮긴 원은 실제 위치의 작은 점과 지시선(1.0 pt 회색)으로 잇는다."""
    m_per_mm = 2 * rho / MAP_D_MM
    X, Y = Q_mm[:, 0] * m_per_mm - rho, Q_mm[:, 1] * m_per_mm - rho
    Xp, Yp = P_mm[:, 0] * m_per_mm - rho, P_mm[:, 1] * m_per_mm - rho
    names = list(M.name)
    for k in np.where(disp > LEADER_MIN_MM)[0]:
        d = Q_mm[k] - P_mm[k]
        L = float(np.hypot(*d))
        end = P_mm[k] + d / L * max(L - CIRCLE_D_MM / 2 - 0.25, 0.0)
        ln = ax.plot([Xp[k], end[0] * m_per_mm - rho], [Yp[k], end[1] * m_per_mm - rho], color=LEADER, lw=1.0,
                     transform=PROJ, zorder=4, solid_capstyle="butt")[0]
        ln.set_gid(f"leader|{panel}|{names[k]}")
        dot = ax.plot([Xp[k]], [Yp[k]], "o", ms=1.2 / S.MM_PER_IN * 72.0, color=LEADER, mew=0, transform=PROJ, zorder=4.1)[0]
        dot.set_gid(f"truepos|{panel}|{names[k]}")
    s_pt2 = (CIRCLE_D_MM / S.MM_PER_IN * 72.0) ** 2
    sc = ax.scatter(X, Y, s=s_pt2, c=M[col].values, cmap=cmap, norm=norm, edgecolors=S.INK, linewidths=1.0,
                    transform=PROJ, zorder=5)
    sc.set_gid(f"data|{panel}|targets")
    return sc


def inset_axes(fig, rect_mm, proj, ext, P):
    """삽도·확대도 공용: 육지, 영구동토 바탕, 1.0 pt #bdbdbd 틀(지도 삽도 틀, 허용 상자)."""
    ax = S.axes_mm(fig, *rect_mm, projection=proj)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.patch.set_facecolor(S.BASEMAP["sea"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=S.BASEMAP["land"], edgecolor="none", linewidth=0, zorder=0)
    ras = draw_pfr(ax, proj, rect_mm[2], rect_mm[3], P)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=S.BASEMAP["coast"], facecolor="none", linewidth=S.LW["coast"], zorder=0.4)
    ax.spines["geo"].set_visible(True)
    ax.spines["geo"].set_linewidth(1.0)
    ax.spines["geo"].set_edgecolor(S.BASEMAP["coast"])
    ax.spines["geo"].set_gid("allowed_frame")
    return ax, ras


def inset_scale_bar(ax, proj, ext, w_mm, km, gid, corner="left"):
    """삽도 아래 구석의 축척 막대. 글자는 막대의 바깥쪽 끝에 맞춘다(캔버스 가장자리 삽도에서 글자가 밖으로 나가지 않게).
    corner='right' 는 오른쪽 아래 구석(알래스카 확대도: 왼쪽 아래 구석에 하위 지역 원이 있다)."""
    m_per_mm = (ext[1] - ext[0]) / w_mm
    Y0 = ext[2] + 1.6 * m_per_mm
    X0 = ext[0] + 1.6 * m_per_mm if corner == "left" else ext[1] - 1.6 * m_per_mm - km * 1000.0
    ln = ax.plot([X0, X0 + km * 1000.0], [Y0, Y0], color=S.INK, lw=S.LW["scale_bar"], transform=proj, zorder=8,
                 solid_capstyle="butt")[0]
    ln.set_gid(gid)
    tx, ha = (X0, "left") if corner == "left" else (X0 + km * 1000.0, "right")
    t = ax.text(tx, Y0 + 0.5 * m_per_mm, f"{km} km", transform=proj, ha=ha, va="bottom", fontsize=S.FONT_PT, zorder=8)
    t.set_gid("scale")
    ll = PC.transform_points(proj, np.array([X0, X0 + km * 1e3]), np.array([Y0, Y0]))
    return dict(km=km, geodesic_km=float(haversine(ll[0, 1], ll[0, 0], ll[1, 1], ll[1, 0])), km_per_mm=m_per_mm / 1e3, text=t)


def tibet_inset(fig, rect_mm, row, col, norm, cmap, P, panel):
    """티베트 고원 삽도(Lambert 방위 등적, 대상 중심), 자체 축척(지침 2.11)."""
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=row.lon, central_latitude=row.lat)
    half_w = 520e3
    half_h = half_w * rect_mm[3] / rect_mm[2]
    ext = (-half_w, half_w, -half_h, half_h)
    ax, ras = inset_axes(fig, rect_mm, proj, ext, P)
    s_pt2 = (CIRCLE_D_MM / S.MM_PER_IN * 72.0) ** 2
    sc = ax.scatter([0.0], [0.0], s=s_pt2, c=[row[col]], cmap=cmap, norm=norm, edgecolors=S.INK, linewidths=1.0,
                    transform=proj, zorder=5)
    sc.set_gid(f"data|{panel}|inset_tibet")
    info = dict(raster=ras, half_w_km=half_w / 1e3, km_per_mm=2 * half_w / 1e3 / rect_mm[2])
    if panel == "a":
        info["scale"] = inset_scale_bar(ax, proj, ext, rect_mm[2], 200, "scale_bar_inset_tibet")
    return ax, info


def alaska_geometry(M_ak, w_mm, h_mm):
    """알래스카 확대도의 투영(중심 경도 = 대상 7개 경도 평균의 북극 평사)과 범위. 범위는 대상 7개 중심의 투영 상자에
    AK_PAD_MM 여백을 더해 w × h mm 에 맞춘 값이다(축척 = 두 방향 가운데 큰 쪽)."""
    lon_c = float(M_ak.lon.mean())
    proj = ccrs.NorthPolarStereo(central_longitude=lon_c, true_scale_latitude=TRUE_LAT)
    Pq = proj.transform_points(PC, M_ak.lon.values, M_ak.lat.values)[:, :2]
    bw, bh = float(np.ptp(Pq[:, 0])), float(np.ptp(Pq[:, 1]))
    s = max(bw / (w_mm - 2 * AK_PAD_MM), bh / (h_mm - 2 * AK_PAD_MM))        # m / mm
    cx, cy = (Pq[:, 0].min() + Pq[:, 0].max()) / 2, (Pq[:, 1].min() + Pq[:, 1].max()) / 2
    ext = (cx - w_mm / 2 * s, cx + w_mm / 2 * s, cy - h_mm / 2 * s, cy + h_mm / 2 * s)
    P_mm = np.c_[(Pq[:, 0] - ext[0]) / s, (Pq[:, 1] - ext[2]) / s]
    return dict(proj=proj, ext=ext, m_per_mm=s, lon_c=lon_c, P_mm=P_mm, lat_c=float(M_ak.lat.mean()))


def alaska_inset(fig, rect_mm, M_ak, col, norm, cmap, P, G, panel):
    """알래스카 확대도: 대상 7개(지역 + 하위 지역 6)를 지름 3 mm 원으로. 겹치면 resolve_overlaps 로 조금 벌린다."""
    ax, ras = inset_axes(fig, rect_mm, G["proj"], G["ext"], P)
    s = G["m_per_mm"]
    Q_mm, disp, _ = resolve_overlaps(G["P_mm"], np.array([rect_mm[2] / 2, rect_mm[3] / 2]), 1e9)
    X, Y = G["ext"][0] + Q_mm[:, 0] * s, G["ext"][2] + Q_mm[:, 1] * s
    Xp, Yp = G["ext"][0] + G["P_mm"][:, 0] * s, G["ext"][2] + G["P_mm"][:, 1] * s
    names = list(M_ak.name)
    for k in np.where(disp > LEADER_MIN_MM)[0]:
        d = Q_mm[k] - G["P_mm"][k]
        L = float(np.hypot(*d))
        end = G["P_mm"][k] + d / L * max(L - CIRCLE_D_MM / 2 - 0.25, 0.0)
        ax.plot([Xp[k], G["ext"][0] + end[0] * s], [Yp[k], G["ext"][2] + end[1] * s], color=LEADER, lw=1.0, transform=G["proj"],
                zorder=4, solid_capstyle="butt", gid=f"leader|{panel}|{names[k]}")
        ax.plot([Xp[k]], [Yp[k]], "o", ms=1.2 / S.MM_PER_IN * 72.0, color=LEADER, mew=0, transform=G["proj"], zorder=4.1,
                gid=f"truepos|{panel}|{names[k]}")
    s_pt2 = (CIRCLE_D_MM / S.MM_PER_IN * 72.0) ** 2
    sc = ax.scatter(X, Y, s=s_pt2, c=M_ak[col].values, cmap=cmap, norm=norm, edgecolors=S.INK, linewidths=1.0, transform=G["proj"], zorder=5)
    sc.set_gid(f"data|{panel}|inset_alaska")
    info = dict(raster=ras, km_per_mm=s / 1e3, disp_mm=dict(zip(names, np.round(disp, 2))),
                min_pair_gap_mm=float(min(np.hypot(*(Q_mm[i] - Q_mm[j])) for i, j in itertools.combinations(range(len(Q_mm)), 2))))
    if panel == "a":
        info["scale"] = inset_scale_bar(ax, G["proj"], G["ext"], rect_mm[2], 500, "scale_bar_inset_alaska", corner="right")
    return ax, info


def zoom_frame(ax_main, G, ov, ax_inset, panel):
    """본 지도 위 확대 범위 사각형(1.0 pt 검정)과 확대도 틀 모서리로 가는 연결선 2개(1.0 pt #bdbdbd, 지침 2.11).
    연결선은 사각형의 화면상 아래쪽 두 꼭짓점과 확대도의 위쪽 두 모서리를 잇는다."""
    x0, x1, y0, y1 = G["ext"]
    n = 25
    xs = np.r_[np.linspace(x0, x1, n), np.full(n, x1), np.linspace(x1, x0, n), np.full(n, x0)]
    ys = np.r_[np.full(n, y0), np.linspace(y0, y1, n), np.full(n, y1), np.linspace(y1, y0, n)]
    Q = PROJ.transform_points(G["proj"], xs, ys)
    ln = ax_main.plot(Q[:, 0], Q[:, 1], color=S.INK, lw=1.0, zorder=6, transform=PROJ, solid_joinstyle="miter")[0]
    ln.set_gid(f"zoom_rectangle|{panel}")
    Cm = PROJ.transform_points(G["proj"], np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]))[:, :2]
    fig = ax_main.figure
    fig.canvas.draw()
    to_mm = lambda xy: xy / fig.dpi * S.MM_PER_IN  # noqa: E731
    disp = to_mm(ax_main.transData.transform(Cm))                     # 표시 좌표(mm, 왼쪽 아래 원점)
    order = np.argsort(disp[:, 1])[:2]                                # 화면 아래쪽 두 꼭짓점
    order = order[np.argsort(disp[order, 0])]                         # 왼쪽, 오른쪽
    ins = to_mm(ax_inset.transAxes.transform([(0.0, 1.0), (1.0, 1.0)]))
    for k, e in zip(order, ins):
        ov.plot([disp[k, 0], e[0]], [H_MM - disp[k, 1], H_MM - e[1]], color=S.BASEMAP["coast"], lw=1.0, zorder=2, solid_capstyle="butt",
                gid=f"zoom_connector|{panel}")
    return disp


# ================================================================ 그림
def build(medium: str = "paper"):
    """Fig 6 를 그린다. medium='slide' 는 이 비교판에서 만들지 않는다(그림 명세 1.7, 미구현을 Fig6_values.txt 에 기록)."""
    if medium != "paper":
        raise NotImplementedError("slide medium is not built for Fig 6 in the v3 comparison set")
    S.use_v3("paper")
    matplotlib.rcParams.update({"axes.unicode_minus": True, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
                                "axes.labelpad": 1.5})
    from cmcrameri import cm as cmc

    R30, M = load_target_deltas()
    C = target_centroids()
    M = M.merge(C, on="name", how="left", validate="one_to_one")
    F = pd.concat([load_forest(), load_xg()], ignore_index=True)
    B, E = load_intervals()
    P = pfr_classes()

    # 공유 정규화(그림 명세 1.4, D-10): 티베트(삽도)를 뺀 지도 값 |Δ| 의 99 백분위를 5 cm 단위로 올림
    arc_mask = ~M.name.isin(INSET_TARGETS).values
    arc = M[arc_mask]
    vals = np.abs(np.r_[arc["D0_1.0"].values, arc["D1_1.0"].values])
    p99 = float(np.percentile(vals, 99))
    vmax = 5.0 * math.ceil(p99 / 5.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cmc.broc
    lo_out = bool((M["D0_1.0"].min() < -vmax) or (M["D1_1.0"].min() < -vmax))
    hi_out = bool((M["D0_1.0"].max() > vmax) or (M["D1_1.0"].max() > vmax))
    ext_kind = {(False, False): "neither", (True, False): "min", (False, True): "max", (True, True): "both"}[(lo_out, hi_out)]

    fig = S.fig_mm(W_MM, H_MM)
    ov = overlay_axes(fig)
    info = dict(vmax=vmax, p99=p99, n_vmax_values=int(len(vals)), ext_kind=ext_kind,
                over={k: int(((M[k] > vmax) | (M[k] < -vmax)).sum()) for k in ("D0_1.0", "D1_1.0")}, pfr_title=P[3])

    # ------------------------------------------------ a, b 지도(원 배치는 두 지도 공통)
    # 본 지도: 독립 지역 5(Lena, Canada, Russia_W, Russia_E, Russia_C)와 하위 지역 4(CA-2, CA-3, LE-1, LE-2).
    # 알래스카 7대상은 확대도(슬롯 오른쪽 아래), 티베트는 삽도(슬롯 왼쪽 아래).
    main = arc[~arc.name.isin(AK_TARGETS)].reset_index(drop=True)
    ak = M[M.name.isin(AK_TARGETS)].set_index("name").loc[AK_TARGETS].reset_index()
    rho = abs(PROJ.transform_point(LON0, LAT_MIN, PC)[1])
    m_per_mm = 2 * rho / MAP_D_MM
    Pm = PROJ.transform_points(PC, main.lon.values, main.lat.values)[:, :2]
    P_mm = (Pm + rho) / m_per_mm                                   # 지도 축 왼쪽 아래 원점(mm)
    center = np.array([MAP_D_MM / 2, MAP_D_MM / 2])
    Q_mm, disp, cluster = resolve_overlaps(P_mm, center, MAP_D_MM / 2)
    GAK = alaska_geometry(ak, AK_INSET_W_MM, AK_INSET_H_MM)
    info.update(rho_m=float(rho), km_per_mm=rho / 1000.0 / (MAP_D_MM / 2), disp_mm=dict(zip(main.name, np.round(disp, 2))),
                clusters={int(c): [main.name.iloc[k] for k in np.where(cluster == c)[0]] for c in np.unique(cluster)
                          if (cluster == c).sum() > 1},
                min_pair_gap_mm=float(min(np.hypot(*(Q_mm[i] - Q_mm[j])) for i, j in itertools.combinations(range(len(Q_mm)), 2))),
                max_r_mm=float(np.max(np.hypot(*(Q_mm - center).T))) + CIRCLE_D_MM / 2,
                alaska_inset=dict(lon_c=GAK["lon_c"], lat_c=GAK["lat_c"], km_per_mm=GAK["m_per_mm"] / 1e3,
                                  extent_km=((GAK["ext"][1] - GAK["ext"][0]) / 1e3, (GAK["ext"][3] - GAK["ext"][2]) / 1e3)))

    axes_map, texts_a = {}, {}
    for letter, (col, head) in MAP_COLS.items():
        x, y, w, h = SLOT[letter]
        rect = map_rect(SLOT[letter])
        S.panel_letter(fig, x + EDGE_MM, y + EDGE_MM, letter)
        t = ov.text(rect[0] + rect[2] / 2, y + EDGE_MM, head, ha="center", va="top", fontsize=S.FONT_PT)
        t.set_gid("category")
        ax, _ = polar_axes(fig, rect)
        info[f"pfr_raster_{letter}"] = draw_pfr(ax, PROJ, rect[2], rect[3], P)
        graticule(ax)
        draw_targets(ax, rho, main, col, P_mm, Q_mm, disp, norm, cmap, letter)
        axes_map[letter] = ax
        # 티베트 삽도: 슬롯 왼쪽 아래 구석(지도 원 밖). 축척·이름은 a 에만(같은 문구 반복 금지, H13)
        trow = M[M.name == "Tibet_LGD"].iloc[0]
        irect = (x, y + h - INSET_H_MM, INSET_W_MM, INSET_H_MM)
        ax_in, iinfo = tibet_inset(fig, irect, trow, col, norm, cmap, P, letter)
        axes_map[f"inset_tibet_{letter}"] = ax_in
        info[f"inset_tibet_{letter}"] = iinfo
        # 알래스카 확대도: 슬롯 오른쪽 아래 구석. 본 지도에 확대 범위 사각형과 연결선
        arect = (x + w - AK_INSET_W_MM, y + h - AK_INSET_H_MM, AK_INSET_W_MM, AK_INSET_H_MM)
        ax_ak, ainfo = alaska_inset(fig, arect, ak, col, norm, cmap, P, GAK, letter)
        axes_map[f"inset_alaska_{letter}"] = ax_ak
        info[f"inset_alaska_{letter}"] = ainfo
        info[f"zoom_rect_corners_mm_{letter}"] = zoom_frame(ax, GAK, ov, ax_ak, letter).round(1).tolist()
        if letter == "a":
            # 삽도·확대도 이름: 각 틀 아래, 왼쪽 맞춤(지도 원 밖 빈 곳, 컬러바 위) [판단]
            texts_a["inset_label"] = ov.text(irect[0] + EDGE_MM, irect[1] + irect[3] + 0.4, "Tibetan Plateau", ha="left", va="top",
                                             fontsize=S.FONT_PT)
            texts_a["alaska_label"] = ov.text(arect[0], arect[1] + arect[3] + 0.4, "Alaska", ha="left", va="top", fontsize=S.FONT_PT)
            info["inset_rect_mm"], info["alaska_rect_mm"] = irect, arect
            texts_a["scale_tibet"] = info["inset_tibet_a"]["scale"].pop("text")
            texts_a["scale_alaska"] = info["inset_alaska_a"]["scale"].pop("text")

    # a 에만: 위도 라벨 3개, 축척 막대, 영구동토 견본 2줄(지침 2.8, 2.11; 같은 문구 반복 금지로 b 에는 두지 않는다)
    ax = axes_map["a"]
    texts_a["lat"] = lat_labels(ax)
    info["scale_a"] = scale_bar(ax, PROJ, SCALE["lon"], SCALE["lat"], SCALE["km"])
    texts_a["scale"] = info["scale_a"].pop("text")
    xa, ya, wa, ha = SLOT["a"]
    key_rows = (("Continuous", S.BASEMAP["continuous"]), ("Discontinuous", S.BASEMAP["discontinuous"]))
    texts_a["pf_key"] = [ov.text(xa + EDGE_MM, ya + HEAD_MM + 1.8, "Permafrost zone", ha="left", va="center", fontsize=S.FONT_PT, zorder=6)]
    for k, (lab, colr) in enumerate(key_rows):                      # 줄 1 = 구역 이름(머리), 줄 2·3 = 견본(v4 map_base.rule)
        yy = ya + HEAD_MM + 1.8 + (k + 1) * 3.2
        ov.add_patch(Rectangle((xa + EDGE_MM, yy - 0.9), 3.0, 1.8, facecolor=colr, edgecolor="none", zorder=6, gid="key_swatch"))
        texts_a["pf_key"].append(ov.text(xa + 4.1, yy, lab, ha="left", va="center", fontsize=S.FONT_PT, zorder=6))

    # 공유 컬러바(가로, 지도 아래)
    cx, cy, cw, ch = SLOT["cbar"]
    cax = S.axes_mm(fig, cx, cy, cw, ch)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=ext_kind, extendfrac=0.03)
    cax.set_label("<colorbar>")
    ticks = np.linspace(-vmax, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([S.fmt_int(v) for v in ticks])
    cb.outline.set_linewidth(S.LW["axis"])
    cb.outline.set_edgecolor(S.INK)
    cb.dividers.set_linewidth(S.LW["axis"])
    cax.tick_params(width=S.LW["tick"], length=S.TICK_LEN_PT, pad=1.5)
    cax.xaxis.set_minor_locator(NullLocator())
    # v4 토큰: 방향 표지(화살표)를 쓰지 않고 방향은 라벨 괄호 안에 적는다
    cb.set_label("Error change vs source Stefan (cm; negative = lower error)", labelpad=1.5)

    # ------------------------------------------------ c 포레스트
    xc, yc, wc, hc = SLOT["c"]
    S.panel_letter(fig, xc + EDGE_MM, yc + EDGE_MM, "c")
    axc = S.axes_mm(fig, xc + FOREST_NAME_MM, yc + 3.4, wc - FOREST_NAME_MM - 2.0, hc - 3.4 - 8.6)
    clean_axes(axc, left=False)
    ypos, ylab, heads = [], [], []
    yv = 0.0
    groups = FOREST + [FOREST_XG]
    assert [lab for _, items in groups for *_, lab in items] == F.row_label.tolist()
    for gi, (head, items) in enumerate(groups):
        if gi:
            yv += 0.35
        heads.append((yv, head))
        yv += 1.0
        for _b, _k, _v, label in items:
            ypos.append(yv)
            ylab.append(label)
            yv += 1.0
    axc.set_ylim(yv - 0.5, -0.6)
    axc.set_xlim(*FOREST_XLIM)
    equiv_band(axc, "x")
    zero_line(axc, "x")
    dy = mm_to_data_y(axc, S.FOREST_BLOCK_OFFSET_MM)
    offaxis = []
    x_hi = FOREST_XLIM[1]
    for yy, (_, r) in zip(ypos, F.iterrows()):
        if min(r.ci_lo, r.ci_lo_beq) > x_hi:
            # 두 가중 CI 가 모두 축 오른쪽 밖(ESA CCI v5): 다른 행을 줄이지 않도록 축을 넓히지 않고, 축 끝 화살표와 점 추정값을 적는다
            # (덱 부록 3 은 축을 끊었다. 이 패널은 c 슬롯 폭 안에 둘째 축을 둘 자리가 없어 명세 요청의 다른 갈래인 가장자리 화살표를 썼다)
            # v4 토큰: 축 밖 값은 화살표 없이 축 끝에 값을 숫자로 적는다(행 색)
            t = axc.text(x_hi, yy, f"+{r.delta:.1f}", ha="right", va="center", fontsize=S.FONT_PT, zorder=6,
                         color=FOREST_COLOR.get(r.row_label, S.INK))
            t.set_gid("offaxis")
            offaxis.append((r.row_label, t))
            continue
        col = FOREST_COLOR.get(r.row_label, S.INK)                         # v4: 행마다 방법·제품 색
        axc.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=S.LW["ci_forest_cell"], solid_capstyle="round", zorder=3,
                 gid=f"data|c|cell|{r.row_label}")
        axc.plot([r.ci_lo_beq, r.ci_hi_beq], [yy + dy, yy + dy], color=col, lw=S.LW["ci_forest_block"], solid_capstyle="butt",
                 zorder=3, gid=f"data|c|block|{r.row_label}")
        axc.plot([r.delta], [yy], "o", ms=S.MS["main"], color=col, mew=0, zorder=4, gid=f"data|c|point|{r.row_label}")
    axc.yaxis.set_major_locator(FixedLocator(ypos))
    axc.set_yticklabels(ylab)
    axc.tick_params(axis="y", length=0, pad=2.0)
    axc.xaxis.set_major_locator(FixedLocator(FOREST_XTICKS))
    axc.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _minus(f"{v:.0f}")))
    axc.set_xlabel("Error change vs source Stefan (cm)")
    for yy, head in heads:
        t = axc.text(0.0, yy, head, transform=axc.get_yaxis_transform(), ha="left", va="center", fontsize=S.FONT_PT)
        t.set_x(-(FOREST_NAME_MM - EDGE_MM) / (axc.get_position().width * W_MM))
        t.set_gid("category")
    # 열쇠(그림 명세 1.2): 첫 묶음 오른쪽 빈 곳, 견본 3개
    kx0 = 2.2
    seg = mm_to_data_x(axc, 3.2)
    gap = mm_to_data_x(axc, 1.0)
    for k, (lab, kind) in enumerate((("Cell-weighted", "cell"), ("Block-equal", "block"), ("±0.5 cm", "band"))):
        yy = heads[0][0] + k * 1.0
        if kind == "cell":
            axc.plot([kx0, kx0 + seg], [yy, yy], color=S.INK, lw=S.LW["ci_forest_cell"], solid_capstyle="round", zorder=3, gid="key|cell")
        elif kind == "block":
            axc.plot([kx0, kx0 + seg], [yy, yy], color=S.INK, lw=S.LW["ci_forest_block"], solid_capstyle="butt", zorder=3, gid="key|block")
        else:
            hh = mm_to_data_y(axc, 1.8)
            axc.add_patch(Rectangle((kx0, yy - hh / 2), seg, hh, facecolor=S.EQUIV_BAND, edgecolor="none", zorder=2, gid="key_swatch"))
        t = axc.text(kx0 + seg + gap, yy, lab, ha="left", va="center", fontsize=S.FONT_PT, zorder=3)
        t.set_gid("sizekey" if kind == "band" else "key_label")

    # ------------------------------------------------ d 포함률 대 구간 폭
    xd, yd, wd, hd = SLOT["d"]
    S.panel_letter(fig, xd, yd + EDGE_MM, "d")
    yl_mm = 7.0                           # y 눈금 라벨 폭(공유 y 축 이름은 c 와 d 사이 빈 칸)
    d_top, d_bot = yd + 4.3, yd + hd - 7.4
    axd = S.axes_mm(fig, xd + yl_mm, d_top, wd - yl_mm - 1.0, d_bot - d_top)
    clean_axes(axd)
    axd.set_xlim(*D_XLIM)
    axd.set_ylim(*COV_YLIM)
    axd.axhline(NOMINAL, color=S.INK_AUX, lw=1.0, zorder=1, gid="ref|d|nominal")
    pool = B[B.kind == "mean4"].set_index("method")
    reg = B[B.kind == "region"]
    for _, r in reg.iterrows():
        axd.plot([r.wid10], [r.cov10], S.REGION_MARKER[r.target], ms=S.MS["region_point"], color=D_COLOR[r.method], alpha=0.5, mew=0, zorder=2,
                 gid=f"data|d|region|{r.method}|{r.target}")
    for m, _lab in D_METHODS:
        r = pool.loc[m]
        axd.plot([r.wid10_lo1, r.wid10_hi1], [r.cov10, r.cov10], color=D_COLOR[m], lw=S.LW["main"], zorder=3, solid_capstyle="butt", gid=f"data|d|cix|{m}")
        axd.plot([r.wid10, r.wid10], [r.cov10_lo1, r.cov10_hi1], color=D_COLOR[m], lw=S.LW["main"], zorder=3, solid_capstyle="butt", gid=f"data|d|ciy|{m}")
        axd.plot([r.wid10], [r.cov10], "o", ms=S.MS["main"], color=D_COLOR[m], mew=0, zorder=4, gid=f"data|d|mean|{m}")
    lab = dict(D_METHODS)
    d_texts = {}
    for m, (tx, ty, ha) in D_LABEL_POS.items():
        t = axd.text(tx, ty, lab[m], ha=ha, va="center", fontsize=S.FONT_PT, zorder=6, color=D_COLOR[m])   # v4: 직접 라벨은 방법 색
        t.set_gid("direct_label")
        d_texts[m] = t
    fig.canvas.draw()
    for m_from, (fx, fy), m_to, stop in D_LEADERS:
        t = d_texts[m_from]
        bb = t.get_window_extent().transformed(axd.transData.inverted())
        r = pool.loc[m_to]
        # 시작점: 글상자 변 위 (fx, fy) 위치에서 표지 쪽으로 0.4 mm 띄운 점
        sx, sy = bb.x0 + fx * (bb.x1 - bb.x0), bb.y0 + fy * (bb.y1 - bb.y0)
        dx, dy_ = r.wid10 - sx, r.cov10 - sy
        ux, uy = dx / mm_to_data_x(axd, 1.0), dy_ / mm_to_data_y(axd, 1.0)          # mm 단위 방향
        L = math.hypot(ux, uy)
        sx += 0.4 * ux / L * mm_to_data_x(axd, 1.0)
        sy += 0.4 * uy / L * mm_to_data_y(axd, 1.0)
        leader_to(axd, (sx, sy), (r.wid10, r.cov10), stop_mm=stop, gid=f"leader|d|{m_to}")
    axd.xaxis.set_major_locator(FixedLocator(D_XTICKS))
    axd.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
    axd.yaxis.set_major_locator(FixedLocator(COV_YTICKS))
    axd.yaxis.set_major_formatter(_fmt_tick(1))
    axd.set_xlabel("Interval width (cm)")

    # 지역 모양 열쇠 한 줄(d 슬롯 위, 패널 문자 오른쪽). d 의 지역 점과 e 의 점에 함께 쓴다(지침 2.5)
    kx = xd + 3.0
    ky = yd + 1.3
    key_texts = []
    for rname in KEY_REGIONS:
        ov.plot([kx + 0.62], [ky], S.REGION_MARKER[rname], ms=S.MS["main"], color=S.INK_AUX, mew=0, gid="key|region")
        t = ov.text(kx + 1.7, ky, S.REGION_NAME[rname], ha="left", va="center", fontsize=S.FONT_PT)
        t.set_gid("key_label")
        key_texts.append(t)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ov.transData.inverted())
        kx = max(bb.x0, bb.x1) + 1.4
    info["region_key_right_mm"] = float(max(bb.x0, bb.x1))

    # ------------------------------------------------ e 라벨 있는 구간
    xe, ye, we, he = SLOT["e"]
    S.panel_letter(fig, xe, ye + EDGE_MM, "e")
    e_top, e_bot = ye + 2.8, ye + he - 7.4
    axe = S.axes_mm(fig, xe + yl_mm, e_top, we - yl_mm - 1.0, e_bot - e_top)
    clean_axes(axe)
    axe.set_xlim(*E_XLIM)
    axe.set_ylim(*COV_YLIM)
    axe.axhspan(*E_BAND, color=S.EQUIV_BAND, lw=0, zorder=0.5).set_gid("band|e|nominal")
    purple = E_COLOR                                    # v4 토큰 color.intervals["Conformal"](잔차 모형의 conformal 구간)
    for _, r in E.iterrows():
        mk = S.REGION_MARKER[r.target]
        filled = int(r.n) == 40
        axe.plot([r.width_ratio, r.width_ratio], [r.cov_lo, r.cov_hi], color=purple, lw=1.0, zorder=3, solid_capstyle="butt",
                 gid=f"data|e|ci|{r.target}|{int(r.n)}")
        axe.plot([r.width_ratio], [r.coverage], mk, ms=S.MS["main"], zorder=4, color=purple, mfc=purple if filled else "white",
                 mec=purple, mew=0 if filled else S.LW["marker_edge_open"], gid=f"data|e|point|{r.target}|{int(r.n)}")
    axe.xaxis.set_major_locator(FixedLocator(E_XTICKS))
    axe.xaxis.set_major_formatter(_fmt_tick(1))
    axe.yaxis.set_major_locator(FixedLocator(COV_YTICKS))
    axe.yaxis.set_major_formatter(_fmt_tick(1))
    axe.set_xlabel("Width relative to no labels")
    # 채움 열쇠(40 labels 채움, 160 labels 빈 원), e 윗부분 빈 곳(명목 띠 위). 빈 마커는 그림 안에서 이 뜻 하나다
    ex = E_XLIM[0] + mm_to_data_x(axe, 1.0)
    for lab_n, filled in (("40 labels", True), ("160 labels", False)):
        yy = 0.975
        axe.plot([ex + mm_to_data_x(axe, 0.62)], [yy], "o", ms=S.MS["main"], color=purple, mfc=purple if filled else "white",
                 mec=purple, mew=0 if filled else 1.0, zorder=5, gid="key|fill")
        t = axe.text(ex + mm_to_data_x(axe, 1.7), yy, lab_n, ha="left", va="center", fontsize=S.FONT_PT, zorder=5)
        t.set_gid("sizekey")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(axe.transData.inverted())
        ex = bb.x1 + mm_to_data_x(axe, 2.5)
    # 공유 y 축 이름(d, e): c 와 d 사이 빈 칸, 두 그림 영역의 가운데 높이
    yl_y = ((d_top + d_bot) / 2 + (e_top + e_bot) / 2) / 2
    ov.text(COV_LABEL_X_MM, yl_y, "Coverage", rotation=90, ha="left", va="center", fontsize=S.FONT_PT)

    # 축을 끈 덮개 축과 지도(GeoAxes)의 눈금 라벨은 그려지지 않지만 Text 객체로 남아 style.text_overlaps·audit_v3 의 글자 수에 잡힌다.
    # 라벨 자체를 끈다(그림에는 변화 없음)
    for a_ in fig.axes:
        if (not a_.axison) or hasattr(a_, "projection"):
            a_.tick_params(axis="both", which="both", labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    data = dict(R30=R30, M=M, arc=arc, main=main, ak=ak, F=F, B=B, E=E, C=C, offaxis=offaxis)
    geom = dict(P_mm=P_mm, Q_mm=Q_mm, disp=disp, cluster=cluster, rho=rho, m_per_mm=m_per_mm,
                axes=dict(c=axc, d=axd, e=axe, maps=axes_map, cbar=cax, ov=ov), texts_a=texts_a, d_texts=d_texts, key_texts=key_texts)
    return fig, data, info, geom


# ================================================================ 점검
def plotted_numbers(fig, data) -> tuple[list[dict], int, int]:
    """그림 객체에서 자료 수치를 읽어 자료 표와 대조한다(gid 'data|…'). 반환 (대조 행 목록, 대조한 수치 수, 불일치 수)."""
    M, arc, F, B, E = data["M"], data["arc"], data["F"], data["B"], data["E"]
    pool = B[B.kind == "mean4"].set_index("method")
    reg = B[B.kind == "region"].set_index(["method", "target"])
    rows, n_num, n_bad = [], 0, 0

    def add(panel, element, label, plotted, expected, source, filt):
        nonlocal n_num, n_bad
        plotted, expected = np.atleast_1d(np.asarray(plotted, float)), np.atleast_1d(np.asarray(expected, float))
        ok = plotted.shape == expected.shape and np.allclose(plotted, expected, rtol=0, atol=1e-9)
        n_num += len(expected)
        n_bad += 0 if ok else 1
        rows.append(dict(panel=panel, element=element, label=label, plotted=", ".join(f"{v:.4f}" for v in plotted),
                         source=source, filter=filt, ok=ok))

    Mi = M.set_index("name")
    for sc in fig.findobj(PathCollection):
        gid = sc.get_gid() or ""
        if not gid.startswith("data|"):
            continue
        _, panel, what = gid.split("|")
        col = MAP_COLS[panel][0]
        src = "wf0_reanalysis.load()+deltas() ← results/rescale_lg/data/processed/lg/lg_curve.csv, data/processed/lgd/lgd_curve_lic.csv"
        names = {"targets": list(data["main"].name), "inset_alaska": list(data["ak"].name), "inset_tibet": ["Tibet_LGD"]}[what]
        where = {"targets": "main map circle colour", "inset_alaska": "Alaska enlargement circle colour", "inset_tibet": "Tibet inset circle colour"}[what]
        arr = np.asarray(sc.get_array())
        assert len(arr) == len(names), (gid, len(arr), len(names))
        for name, v in zip(names, arr):
            add(panel, where, name, v, Mi.loc[name, col], src,
                f"n == 0, target == '{name}|x', column {col} (= rmse − rmse_p0; placement cell, alpha 1, learner catboost_lo, lam 1.0)")
    Fi = F.set_index("row_label")
    for ln in fig.findobj(Line2D):
        gid = ln.get_gid() or ""
        if not gid.startswith("data|"):
            continue
        parts = gid.split("|")
        panel, kind = parts[1], parts[2]
        xs, ys = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if panel == "c":
            label = parts[3]
            r = Fi.loc[label]
            if r.source == "xg_tests.csv":
                filt = f"hypothesis == '{r.hypothesis}' (main columns = 5 km block mask)"
                src = _rel(XG_TESTS)
            else:
                filt = f"block == '{r.block}', key == '{r.key}', variant == '{r.variant}'"
                src = "data/processed/paper_figs/fig6_a.csv"
            if kind == "cell":
                add("c", "cell-weighted 95% CI", label, xs, [r.ci_lo, r.ci_hi], src, filt + " → ci_lo, ci_hi")
            elif kind == "block":
                add("c", "block-equal 95% CI", label, xs, [r.ci_lo_beq, r.ci_hi_beq], src, filt + " → ci_lo_beq, ci_hi_beq")
            else:
                add("c", "point estimate", label, xs, [r.delta], src, filt + " → delta")
        elif panel == "d":
            src = "data/processed/paper_figs/fig6_b.csv"
            if kind == "region":
                m, t = parts[3], parts[4]
                r = reg.loc[(m, t)]
                add("d", "regional value (x, y)", f"{m} / {t}", [xs[0], ys[0]], [r.wid10, r.cov10], src,
                    f"kind == 'region', method == '{m}', target == '{t}' → wid10, cov10")
            else:
                m = parts[3]
                r = pool.loc[m]
                if kind == "mean":
                    add("d", "four-region mean (x, y)", m, [xs[0], ys[0]], [r.wid10, r.cov10], src,
                        f"kind == 'mean4', method == '{m}' → wid10, cov10")
                elif kind == "cix":
                    add("d", "width 95% interval", m, xs, [r.wid10_lo1, r.wid10_hi1], src, f"kind == 'mean4', method == '{m}' → wid10_lo1, wid10_hi1")
                else:
                    add("d", "coverage 95% interval", m, ys, [r.cov10_lo1, r.cov10_hi1], src, f"kind == 'mean4', method == '{m}' → cov10_lo1, cov10_hi1")
        elif panel == "e":
            t, n = parts[3], int(parts[4])
            r = E[(E.target == t) & (E.n == n)].iloc[0]
            src = "data/processed/paper_figs/fig6_d.csv"
            if kind == "point":
                add("e", "point (x, y)", f"{t} n={n}", [xs[0], ys[0]], [r.width_ratio, r.coverage], src,
                    f"target == '{t}', n == {n} → width_ratio, coverage")
            else:
                add("e", "coverage 95% interval", f"{t} n={n}", ys, [r.cov_lo, r.cov_hi], src, f"target == '{t}', n == {n} → cov_lo, cov_hi")
    for label, t in data.get("offaxis", []):                 # 축 밖 행: 적은 점 추정값(0.1 cm)과 두 CI 하한이 모두 축 밖인지
        r = Fi.loc[label]
        src = _rel(XG_TESTS) if r.source == "xg_tests.csv" else "data/processed/paper_figs/fig6_a.csv"
        v = float(t.get_text().replace("+", "").replace("\u2212", "-"))
        add("c", "off-axis point estimate (printed, 0.1 cm)", label, v, round(float(r.delta), 1), src, f"hypothesis == '{r.hypothesis}' → delta")
        add("c", "off-axis: both 95% CI lower bounds > axis end", label, [r.ci_lo, r.ci_lo_beq],
            [r.ci_lo, r.ci_lo_beq] if min(r.ci_lo, r.ci_lo_beq) > FOREST_XLIM[1] else [np.nan, np.nan], src,
            f"ci_lo, ci_lo_beq > {FOREST_XLIM[1]} cm")
    return rows, n_num, n_bad


def _chk(name, ok, detail):
    return dict(name=name, ok=bool(ok), detail=detail)


def source_checks(data, info, geom) -> list[dict]:
    """자료 표를 원천 표에서 독립 조회로 다시 대조하고, 설명문 수치를 원천에서 확인한다."""
    out = []
    M, arc, F, B, E, R30, C = data["M"], data["arc"], data["F"], data["B"], data["E"], data["R30"], data["C"]
    # 1. 지도 Δ: lg_curve / lgd_curve_lic 에서 직접 필터
    lg = pd.read_csv(LG_CURVE, low_memory=False)
    ld = pd.read_csv(LGD_CURVE, low_memory=False)
    diffs, n = [], 0
    for r in M.itertuples():
        src = ld if r.name in LGD_RUN_TABLE else lg
        for meth, col in (("D0", "D0_1.0"), ("D1", "D1_1.0")):
            q = src[(src.target == r.name) & (src["mode"] == "x") & (src.n == 0) & (src.placement == "cell") & (src.alpha.astype(str) == "1")
                    & (src.learner == "catboost_lo") & (src.method == meth) & np.isclose(src.lam.astype(float), 1.0) & (src.point_only != True)]  # noqa: E712
            assert len(q) == 1, (r.name, meth, len(q))
            diffs.append(abs(float(q.rmse.iloc[0] - q.rmse_p0.iloc[0]) - float(M.loc[r.Index, col])))
            n += 1
    out.append(_chk("지도 Δ = 원천 곡선 직접 필터(rmse − rmse_p0)", max(diffs) < 1e-9, f"17대상 × 2방법 = {n}개, 최대 차 {max(diffs):.2g} cm"))
    # 2. 위험표(설명문 18/30, 2/30)와 30대상 재집계
    risk = pd.read_csv(WF0_RISK)
    r0 = risk[risk.n == 0].set_index("method")
    cnt = {k: int((R30[k] > 2.0).sum()) for k in ("D0_1.0", "D1_1.0")}
    out.append(_chk("설명문 '18 with direct ML', '2 with physics pseudo-labels' = wf0_risk.csv n_worse_2cm = 30대상 재집계",
                    int(r0.loc["D0_1.0", "n_worse_2cm"]) == 18 == cnt["D0_1.0"] and int(r0.loc["D1_1.0", "n_worse_2cm"]) == 2 == cnt["D1_1.0"]
                    and int(r0.loc["D0_1.0", "n_targets"]) == 30 == len(R30),
                    f"wf0_risk n==0: D0_1.0 {int(r0.loc['D0_1.0', 'n_worse_2cm'])}/{int(r0.loc['D0_1.0', 'n_targets'])}, "
                    f"D1_1.0 {int(r0.loc['D1_1.0', 'n_worse_2cm'])}/{int(r0.loc['D1_1.0', 'n_targets'])}; 재집계 {cnt}, 30대상 {len(R30)}"))
    out.append(_chk("지도 17대상 = 모드 x 독립 7 + 하위 지역 10(D-9), 나머지 13 은 모드 i 10 + 확충판 3",
                    sorted(set(R30.target) - set(arc.target) - {"Tibet_LGD|x"}) == sorted([f"{s}|i" for s in SUB10] + ["Canada~exp~lic|x", "Russia_E~exp|x", "Russia_W~exp~lic|x"]),
                    f"지도 밖 13: {sorted(set(R30.target) - set(M.target))}"))
    # 3. 대상 중심: 셀 수를 Table 1 행과 대조, LGD 대상은 v4 직접 집계와 대조
    t1 = pd.read_csv(TABLE1_ROWS)
    t1m = t1.set_index("target")
    cmp_rows = []
    for name in ["Lena", "Canada", "Russia_W", "Russia_E", "Alaska"]:
        cmp_rows.append((name, int(C.set_index("name").loc[name, "n_cells"]), int(t1m.loc[name, "label_rows"])))
    ok = all(a == b for _, a, b in cmp_rows)
    out.append(_chk("대상 중심 셀 수 = Table 1 label_rows(주 지역 5)", ok, "; ".join(f"{nm} {a}/{b}" for nm, a, b in cmp_rows)))
    v4 = v4_crosscheck()
    cr = {t: (int(C.set_index("name").loc[t, "n_cells"]), v4[t]) for t in LGD_RUN_TABLE}
    out.append(_chk("LGD 대상 중심(실행 표) 대 fidelity_base_v4 직접 집계(기록용)",
                    all(abs(cr[t][0] - cr[t][1]["n"]) <= 1 for t in cr),
                    "; ".join(f"{t}: 실행 표 {a}셀 / v4 {b['n']}셀 ({b['lat']}, {b['lon']})" for t, (a, b) in cr.items())
                    + ". 러시아 중부는 1셀 차(실행 표가 LGD 입력이므로 실행 표 값을 쓴다)"))
    sub_n = {s: int(C.set_index("name").loc[s, "n_cells"]) for s in SUB10}
    ak = sum(v for s, v in sub_n.items() if s.startswith("AL"))
    out.append(_chk("알래스카 하위 지역 6개 셀 수 합 = 알래스카 셀 수", ak == int(C.set_index("name").loc["Alaska", "n_cells"]),
                    f"AL-1–AL-6 합 {ak}, Alaska {int(C.set_index('name').loc['Alaska', 'n_cells'])}; 하위 지역 셀 수 {sub_n}"))
    # 4. 포레스트: fig6_a.csv 의 14행을 원천 표(lgw_bundle, lgx_tests, lgt_tests, lgf_tests, lgfn_tests)에서 다시 조회
    diffs, details = [], []
    for r in F.itertuples():
        src = pd.read_csv(FOREST_SOURCE[r.source], low_memory=False)
        if r.source == "lgw_bundle.csv":
            q = src[(src.ab == r.ab) & (src.target == MEAN4)]
            filt = f"ab == '{r.ab}', target == MEAN4"
        elif r.source == "xg_tests.csv":
            q = src[src.hypothesis == r.hypothesis]
            filt = f"hypothesis == '{r.hypothesis}'"
        else:
            tid = r.hypothesis.split("(")[0]
            q = src[(src.test_id == tid) & (src.contrast == r.contrast) & (src.target == MEAN4)]
            if "item" in q.columns and len(q) > 1:
                q = q[q.item.isna()]
            filt = f"test_id == '{tid}', contrast == '{r.contrast}', target == MEAN4"
        assert len(q) == 1, (r.row_label, len(q))
        q = q.iloc[0]
        beq_col = "delta_beq" if "delta_beq" in q.index else "delta_blockeq"
        d = max(abs(q.delta - r.delta), abs(q.ci_lo - r.ci_lo), abs(q.ci_hi - r.ci_hi), abs(q[beq_col] - r.delta_beq),
                abs(q.ci_lo_beq - r.ci_lo_beq), abs(q.ci_hi_beq - r.ci_hi_beq))
        diffs.append(d)
        details.append(f"{r.row_label}: {r.source} [{filt}] 차 {d:.1e}")
    out.append(_chk(f"포레스트 {len(F)}행 = 원천 판정 표(6값씩 {6 * len(F)}개)", max(diffs) < 1e-9 and len(F) == 16,
                    f"최대 차 {max(diffs):.2g} cm; " + "; ".join(details)))
    out.append(_chk("포레스트 재표집 10000회, 4지역 층화 평균", set(F.nboot) == {10000}, f"nboot {sorted(set(F.nboot))}"))
    # 설명문 수치(c)
    cb = F.set_index("row_label").loc["CatBoost, main"]
    pst = F.set_index("row_label").loc["Year-matched Stefan"]
    lgw = pd.read_csv(LGW_BUNDLE)
    ab1 = lgw[(lgw.ab == "AB1") & (lgw.target == MEAN4)].iloc[0]
    out.append(_chk("설명문 CatBoost 2.25 [1.10, 3.42], Holm P 0.023", (f"{cb.delta:.2f}", f"{cb.ci_lo:.2f}", f"{cb.ci_hi:.2f}", f"{ab1.holm_p:.3f}")
                    == ("2.25", "1.10", "3.42", "0.023"), f"fig6_a catboost_lo {cb.delta:.4f} [{cb.ci_lo:.4f}, {cb.ci_hi:.4f}]; lgw_bundle AB1 holm_p {ab1.holm_p}"))
    Fl = F.set_index("row_label")
    six = ["Random forest", "CatBoost, physics inputs", "MLP", "Multi-head MLP", "FT-Transformer, reduced", "RealMLP"]
    five = [Fl.loc[k, "delta"] for k in six]
    out.append(_chk("설명문 '2.44–5.21 cm' = 두 CI 판정이 같은 6행(rf, 물리 출력 입력 CatBoost(F1k, C1 README 2.1 보조), mlp, tabm, ftt, realmlp)의 Δ 범위,"
                    " 6행 모두 두 가중 CI 하한 > 0 이고 분할 독립 가정 의존 표지 없음",
                    (f"{min(five):.2f}", f"{max(five):.2f}") == ("2.44", "5.21") and all(Fl.loc[k, "ci_lo"] > 0 and Fl.loc[k, "ci_lo_beq"] > 0
                                                                                      and Fl.loc[k, "ci_dependence"] == "" for k in six),
                    "; ".join(f"{k} {Fl.loc[k, 'delta']:.4f} [{Fl.loc[k, 'ci_lo']:.2f}, {Fl.loc[k, 'ci_hi']:.2f}] / [{Fl.loc[k, 'ci_lo_beq']:.2f}, {Fl.loc[k, 'ci_hi_beq']:.2f}]"
                              for k in six)))
    other5 = ["CatBoost, main", "CatBoost, larger", "CatBoost, tuned", "TabPFN v2", "TabICL v2", "TabICL v2, full context"]
    out.append(_chk("설명문 'The other five learners ... gave the same verdict under the primary intervals' = 6행(TabICL 두 조건) 모두 주 CI 의 두 가중 하한 > 0",
                    all(Fl.loc[k, "ci_lo"] > 0 and Fl.loc[k, "ci_lo_beq"] > 0 for k in other5),
                    "; ".join(f"{k} [{Fl.loc[k, 'ci_lo']:.2f}, {Fl.loc[k, 'ci_hi']:.2f}] / [{Fl.loc[k, 'ci_lo_beq']:.2f}, {Fl.loc[k, 'ci_hi_beq']:.2f}]" for k in other5)))
    out.append(_chk("설명문 연도 정합 Stefan 'lower error' = 두 가중 CI 상한 < 0", Fl.loc["Year-matched Stefan", "ci_hi"] < 0 and Fl.loc["Year-matched Stefan", "ci_hi_beq"] < 0,
                    f"[{Fl.loc['Year-matched Stefan', 'ci_lo']:.2f}, {Fl.loc['Year-matched Stefan', 'ci_hi']:.2f}] / "
                    f"[{Fl.loc['Year-matched Stefan', 'ci_lo_beq']:.2f}, {Fl.loc['Year-matched Stefan', 'ci_hi_beq']:.2f}]"))
    cap = CAPTIONS_V2.read_text(encoding="utf-8") if CAPTIONS_V2.exists() else ""
    out.append(_chk("설명문 앙상블 정의 'mean of scale-calibrated Stefan, Kudryavtsev and ESA CCI anchors' = CAPTIONS.md v2/Fig6 의 B:ens 정의",
                    "B:ens averages the scale-calibrated Stefan, Kudryavtsev and CCI anchors" in cap, f"{_rel(CAPTIONS_V2)} 문자열 포함 여부"))
    dep = F[F.ci_dependence != ""].row_label.tolist()
    out.append(_chk("분할 독립 가정 의존 행 = CatBoost 3설정, TabPFN v2, TabICL v2 두 조건, 연도 정합 Stefan(7행)",
                    set(dep) == {"CatBoost, main", "CatBoost, larger", "CatBoost, tuned", "TabPFN v2", "TabICL v2", "TabICL v2, full context", "Year-matched Stefan"},
                    f"ci_dependence 행 {dep}"))
    out.append(_chk("설명문 연도 정합 Stefan −1.00 cm", f"{pst.delta:.2f}" == "-1.00", f"{pst.delta:.4f} [{pst.ci_lo:.4f}, {pst.ci_hi:.4f}], {pst.ci_dependence}"))
    readme = C1_README.read_text(encoding="utf-8")
    out.append(_chk("C1 README 1.3 문자열(+2.44 – +5.21, +2.25 cm [1.10, 3.42], 18개, 2개)",
                    all(s in readme for s in ("+2.44 – +5.21", "+2.25 cm [1.10, 3.42]", "직접 ML 18개, 물리 유사라벨 증강 2개")), "문자열 포함 여부"))
    # 5. d: fig6_b.csv 대 lgu_b_intervals.csv(scope all, n 0)
    u = pd.read_csv(LGU_B, low_memory=False)
    u = u[(u.scope == "all") & (u.n == 0)]
    diffs = []
    for r in B.itertuples():
        q = u[(u.target == r.target) & (u.method == r.method)]
        assert len(q) == 1, (r.target, r.method, len(q))
        q = q.iloc[0]
        diffs.append(max(abs(q.cov10 - r.cov10), abs(q.wid10 - r.wid10), abs(q.cov10_lo1 - r.cov10_lo1), abs(q.cov10_hi1 - r.cov10_hi1),
                         abs(q.wid10_lo1 - r.wid10_lo1), abs(q.wid10_hi1 - r.wid10_hi1)))
    out.append(_chk("d 25행 = lgu_b_intervals.csv(scope == 'all', n == 0, target, method; 6값씩 150개)", max(diffs) < 1e-9, f"최대 차 {max(diffs):.2g}"))
    out.append(_chk("d 1단 CI 종류 '2단+1단' 표의 1단 열, 지역 재표집 단위 블록 수 18–39", set(B.ci_kind) == {"2단+1단"},
                    f"ci_kind {set(B.ci_kind)}; n_blocks {sorted(B[B.kind == 'region'].n_blocks.dropna().astype(int).unique().tolist())}"))
    # 6. e: fig6_d.csv 대 lgx_conformal.csv(R1 λ 0.25 level 90, 폭 비의 분모는 R0 n 0)
    cf = pd.read_csv(LGX_CONFORMAL)
    diffs = []
    for r in E.itertuples():
        q = cf[(cf.target == r.target) & (cf["mode"] == "x") & (cf.method == "R1") & (cf.learner == "catboost_lo") & (cf.n == r.n)
               & np.isclose(cf.lam, 0.25) & (cf.level == 90)]
        q0 = cf[(cf.target == r.target) & (cf["mode"] == "x") & (cf.method == "R0") & (cf.learner == "catboost_lo") & (cf.n == 0)
                & np.isclose(cf.lam, 0.25) & (cf.level == 90)]
        assert len(q) == 1 and len(q0) == 1, (r.target, r.n, len(q), len(q0))
        q, q0 = q.iloc[0], q0.iloc[0]
        diffs.append(max(abs(q.coverage - r.coverage), abs(q.cov_lo - r.cov_lo), abs(q.cov_hi - r.cov_hi), abs(q.width_cm - r.width_cm),
                         abs(q0.width_cm - r.width_r0_n0), abs(q.width_cm / q0.width_cm - r.width_ratio)))
    out.append(_chk("e 6행 = lgx_conformal.csv(mode x, R1 λ 0.25 level 90; 폭 비 = width_cm / R0 n 0 width_cm)", max(diffs) < 1e-9, f"최대 차 {max(diffs):.2g}"))
    lx = pd.read_csv(LGX_TESTS, low_memory=False)
    l25 = lx[(lx.test_id == "L25") & (lx.scope == "region")]
    d25 = []
    for r in E.itertuples():
        q = l25[(l25.target == f"{r.target}|x") & (l25.n == r.n)].iloc[0]
        d25.append(max(abs(q.coverage - r.coverage), abs(q.width_cm - r.width_cm), abs(q.width_n0_r0 - r.width_r0_n0)))
    out.append(_chk("e 값 = lgx_tests.csv L25 지역 행(coverage, width_cm, width_n0_r0)", max(d25) < 1e-9, f"최대 차 {max(d25):.2g}"))
    # 7. 지도 투영·범위가 Fig 1a 와 같은지(기록된 중심 경도)
    if FIG1_VALUES.exists():
        m = re.search(r"중심 경도 ([0-9.]+)°", FIG1_VALUES.read_text(encoding="utf-8"))
        got = float(m.group(1)) if m else float("nan")
        out.append(_chk("지도 중심 경도 = Fig 1a 기록값(Fig1_values.txt)", abs(got - LON0) < 1e-3, f"Fig 1 {got}°, Fig 6 {LON0}°; 50° N 원, 진척 위도 70° N"))
    else:
        out.append(_chk("지도 중심 경도 = Fig 1a 기록값", False, "Fig1_values.txt 없음"))
    # 8. vmax 규칙(D-10)
    # 9. 설명문의 지도·구간 수치(티베트 끝 색 값, 알래스카 확대도 중심 경도, 재표집 횟수, 영구동토 자료 판)
    leg = LEGEND_MD.read_text(encoding="utf-8") if LEGEND_MD.exists() else ""
    # 10. 기존 ALT 지도 행(XG, 계획 8.7): 판정과 설명문 문장
    X = F[F.source == "xg_tests.csv"].set_index("key")
    c5, w5 = X.loc["XG-1c"], X.loc["XG-1w"]
    out.append(_chk("XG-1c(CCI v5, 주 4지역) 두 마스크 판정 '열세', Holm p ≤ 0.05, 5 km·25 km 값 같음(학습 지점 없음) → 설명문 'higher error (+23.79 cm; 95% CI 15.34 to 30.16'",
                    c5.verdict4 == "열세" and c5.verdict4_blk25 == "열세" and max(c5.p_holm_blk5, c5.p_holm_blk25) <= 0.05
                    and abs(c5.delta - c5.delta_blk25) < 1e-9 and c5.pool_name == "four-region stratified mean"
                    and f"higher error (+{c5.delta:.2f} cm at axis end; 95% CI {c5.ci_lo:.2f} to {c5.ci_hi:.2f}" in leg,
                    f"Δ {c5.delta:.4f} [{c5.ci_lo:.4f}, {c5.ci_hi:.4f}] / 블록 {c5.delta_beq:.4f} [{c5.ci_lo_beq:.4f}, {c5.ci_hi_beq:.4f}]; Holm p {c5.p_holm_blk5}/{c5.p_holm_blk25};"
                    f" {c5.blind}; {c5.deviation}; {c5.design}"))
    out.append(_chk("XG-1w(Wei v2, 레나·캐나다) 두 마스크 판정 '미결정' → 설명문 'difference (Lena Delta and Canada; 47% of cells gap-filled) was not resolved'",
                    w5.verdict4 == "미결정" and w5.verdict4_blk25 == "미결정" and w5.pool_name.startswith("two-region")
                    and f"(Lena Delta and Canada; {100 * w5.fill_frac_blk5:.0f}% of cells gap-filled) was not resolved" in leg,
                    f"Δ {w5.delta:.4f} [{w5.ci_lo:.4f}, {w5.ci_hi:.4f}] / 블록 {w5.delta_beq:.4f} [{w5.ci_lo_beq:.4f}, {w5.ci_hi_beq:.4f}]; 25 km Δ {w5.delta_blk25:.4f};"
                    f" Holm p {w5.p_holm_blk5}/{w5.p_holm_blk25}; 제품 결측 대체(ρ·s) 5 km {w5.fill_frac_blk5:.4f}, 25 km {w5.fill_frac_blk25:.4f};"
                    f" 마스크 뒤 P0 RMSE {w5.rmse_p0_postmask_blk5:.2f} cm; {w5.blind}; {w5.deviation}"))
    out.append(_chk("XG 행 재표집 = xg_summary_meta.json nboot 10000", set(X.nboot) == {10000}, f"nboot {sorted(set(X.nboot))}"))
    tib = float(M.set_index("name").loc["Tibet_LGD", "D0_1.0"])
    out.append(_chk("설명문 'Tibetan Plateau in a, −59.98 cm' = 티베트 직접 ML Δ 이고 범위 밖 값은 이것 하나",
                    _minus(f"{tib:.2f}") + " cm" in leg and info["over"] == {"D0_1.0": 1, "D1_1.0": 0} and tib < -info["vmax"],
                    f"Tibet_LGD D0_1.0 {tib:.4f} cm, vmax {info['vmax']:.0f}, 범위 밖 {info['over']}"))
    lon_c = info["alaska_inset"]["lon_c"]
    ak_txt = f"{abs(lon_c):.0f}° {'W' if lon_c < 0 else 'E'}"
    out.append(_chk(f"설명문 투영 중심 경도 '127° E'(본 지도 {LON0}°), 알래스카 확대도 '{ak_txt}'(대상 7개 경도 평균 {lon_c:.3f}°)",
                    "central meridian 127° E" in leg and f"Alaska enlargement, {ak_txt}" in leg and f"{LON0:.0f}" == "127", f"확대도 중심 위도 {info['alaska_inset']['lat_c']:.2f}° N"))
    mb = json.loads(LGU_B_META.read_text(encoding="utf-8"))
    mx = json.loads(LGX_META.read_text(encoding="utf-8"))
    out.append(_chk("설명문 재표집 'd 1000 resamples'(lgu_b_meta.json nboot), 'c and e 10,000 resamples'(fig6_a nboot, lgx_meta.json nboot)",
                    int(mb["nboot"]) == D_NBOOT == 1000 and int(mx["nboot"]) == E_NBOOT == 10000 and set(F.nboot) == {10000}
                    and "1000 resamples" in leg and "10,000 resamples" in leg,
                    f"lgu_b nboot {mb['nboot']}, lgx nboot {mx['nboot']}; e 의 포함률 CI 는 h42_label_grid_ext.conformal_table 의 boot_mean_blocks(채점 블록 재표집)"))
    out.append(_chk("설명문 영구동토 자료 'ESA CCI Permafrost fraction v4.0 (1997–2021 mean)' = netCDF title", "v4.0" in info["pfr_title"] and "1997" in info["pfr_title"]
                    and "2021" in info["pfr_title"] and "ESA CCI Permafrost fraction v4.0 (1997–2021 mean)" in leg, f"title '{info['pfr_title']}'"))
    import cartopy
    out.append(_chk("설명문 'Cartopy 판' = 실행 판", f"Cartopy {cartopy.__version__}" in leg, f"cartopy {cartopy.__version__}"))
    out.append(_chk("컬러바 ±vmax = 티베트 뺀 |Δ| 99 백분위를 5 cm 단위로 올림", info["vmax"] == 5.0 * math.ceil(info["p99"] / 5.0),
                    f"값 {info['n_vmax_values']}개(16대상 × 2; 명세 8.3 의 '34값' 은 티베트를 빼면 32값), p99 {info['p99']:.2f}, vmax {info['vmax']:.0f}; "
                    f"범위 밖 {info['over']}(티베트 D0 {M.set_index('name').loc['Tibet_LGD', 'D0_1.0']:.2f}), 컬러바 연장 '{info['ext_kind']}'"))
    return out


def legend_checks() -> tuple[list[dict], dict]:
    out, meta = [], {}
    if not LEGEND_MD.exists():
        return [_chk("설명문 파일", False, f"{LEGEND_MD} 없음")], meta
    txt = LEGEND_MD.read_text(encoding="utf-8")
    body = [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#") and not ln.startswith("<!--")]
    legend = " ".join(body).replace("**", "")
    placeholders = re.findall(r"\[X[A-J]: [^\]]*\]", legend)
    legend_wo = re.sub(r"\[X[A-J]: [^\]]*\]", "", legend)
    nw, nw_wo = len(legend.split()), len(legend_wo.split())
    meta.update(words=nw, words_wo_placeholder=nw_wo, placeholders=placeholders, first=legend[:120])
    out.append(_chk("설명문 350단어 이하(자리표시 포함)", nw <= 350, f"{nw}단어(자리표시 제외 {nw_wo}), 파일 전체 {len(txt.split())}단어"))
    ta = S.text_audit(legend_wo)
    bad = {k: v for k, v in ta.items() if v}
    out.append(_chk("설명문 문장 점검(대시·금지어·가운뎃점 숫자·네 자리 쉼표·내부 약호; 자리표시 제외)", not bad, f"{bad or '0건'}"))
    out.append(_chk("설명문 첫 문장이 'This figure' 로 시작하지 않고 그림 명세 8.6·원고 명세 7절의 제목 문장과 같음",
                    legend.startswith("Without target labels, 2 of 30 targets lost over 2 cm with physics pseudo-labels and 18 with direct ML."),
                    legend[:110]))
    out.append(_chk("설명문 자리표시 0개(XG 자리표시를 수치 문장으로 바꿈, 그림 명세 12절)", placeholders == [], f"{placeholders or '0개'}"))
    m_xg = re.search(r"Existing ALT maps.*?resolved\.", legend)
    n_xg = len(m_xg.group(0).split()) if m_xg else -1
    meta["xg_sentence_words"] = n_xg
    out.append(_chk("XG 문장(Existing ALT maps … resolved.) 50단어 이하(그림 명세 13절)", 0 < n_xg <= 50, f"{n_xg}단어"))
    need = ["10,000 resamples", "±0.5 cm", "70° N", "Natural Earth", "ESA CCI", "Cartopy", "post hoc", "split-independence", "17 targets", "30 targets",
            "Existing ALT maps", "5 km of product training sites", "at axis end"]
    miss = [s for s in need if s not in legend]
    out.append(_chk("설명문 필수 요소(재표집·띠·축척 위도·자료 출처·소프트웨어·사후 서술·의존 표지·대상 수)", not miss, f"빠짐 {miss or '없음'}"))
    color_words = [w for w in ("blue", "brown", "purple", "grey ", "gray ", "black", "triangle", "square", "diamond", "circle") if w in legend.lower()]
    out.append(_chk("설명문에 색 이름·모양 서술 없음(F-15)", not color_words, f"{color_words or '0건'}"))
    return out, meta


def drawn_texts(fig) -> list[Text]:
    """실제로 그려지는 글자만 고른다. 축을 끈 덮개 축과 지도(GeoAxes)의 눈금 라벨 객체는 보이지 않지만 get_visible() 이 참이라 뺀다."""
    hidden = set()
    for a in fig.axes:
        if (not a.axison) or hasattr(a, "projection"):
            for axis in (a.xaxis, a.yaxis):
                hidden.update(axis.get_ticklabels(which="both"))
                hidden.update(axis.get_majorticklabels())
                hidden.add(axis.label)
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in hidden]


def overlap_check(fig, geom) -> list[str]:
    """글자끼리, 글자와 자료(원·점·선)가 겹치는 곳을 찾는다(지침 2.2 '글자 겹침 0')."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = drawn_texts(fig)
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    issues = []
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if b1.width > 0 and b2.width > 0 and b1.overlaps(b2):
            issues.append(f"글자 겹침: '{t1.get_text()}' / '{t2.get_text()}'")
    # 지도 글자 대 대상 원
    circles = []
    for sc in fig.findobj(PathCollection):
        if (sc.get_gid() or "").startswith("data|"):
            xy = sc.get_offset_transform().transform(sc.get_offsets())
            r_px = (CIRCLE_D_MM / 2 + 0.3) / 25.4 * fig.dpi
            circles += [(x, y, r_px, sc.get_gid()) for x, y in xy]
    data_lines = [ln for ln in fig.findobj(Line2D) if (ln.get_gid() or "").split("|")[0] in ("data", "leader", "scale_bar", "truepos")]
    for t, b in boxes:
        gid = t.get_gid() or ""
        for x, y, r, g in circles:
            if b.x0 - r <= x <= b.x1 + r and b.y0 - r <= y <= b.y1 + r:
                issues.append(f"글자-원 겹침: '{t.get_text()}' / {g}")
        if gid in ("direct_label", "direction_label", "scale", "graticule", "key_label", "sizekey", "offaxis"):
            for ln in data_lines:
                xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
                seg = xy if len(xy) == 1 else np.vstack([np.linspace(xy[i], xy[i + 1], 60) for i in range(len(xy) - 1)])
                has_marker = ln.get_marker() not in (None, "None", "none", "", " ")
                # 점 하나짜리 표지는 선이 그려지지 않으므로 표지 반지름만, 선은 굵기의 반을 더한다
                r_pt = ln.get_markersize() / 2 if (has_marker and len(xy) == 1) else (ln.get_markersize() / 2 if has_marker else 0.0) + ln.get_linewidth() / 2
                r_px = r_pt * fig.dpi / 72.0
                if gid == "direct_label" and (ln.get_gid() or "").startswith("data|"):
                    r_px += LABEL_PAD_MM / S.MM_PER_IN * fig.dpi        # 직접 라벨은 자료 표지에서 0.5 mm 이상 띄운다
                inside = ((seg[:, 0] >= b.x0 - r_px) & (seg[:, 0] <= b.x1 + r_px) & (seg[:, 1] >= b.y0 - r_px) & (seg[:, 1] <= b.y1 + r_px))
                if inside.any() and not (gid == "scale" and (ln.get_gid() or "").startswith("scale_bar")):
                    issues.append(f"글자-자료 겹침: '{t.get_text()}' / {ln.get_gid()}")
    return issues


def _seg_dist(p, A) -> float:
    """점 p 와 꺾은선 A(n × 2) 사이 최소 거리."""
    A = np.atleast_2d(A)
    if len(A) == 1:
        return float(np.hypot(*(p - A[0])))
    d = []
    for a, b in zip(A[:-1], A[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0)
        d.append(float(np.hypot(*(p - a - t * ab))))
    return min(d)


def map_clearance_check(fig, geom, data, min_mm=MAP_CLEAR_MM) -> tuple[list[str], dict]:
    """본 지도 대상 원(지름 3 mm)의 테두리와 확대 사각형, 확대도·삽도 틀, 연결선, 다른 원의 지시선 사이 간격(mm).
    min_mm 미만이거나 원 중심이 확대도·삽도 틀 안이면 보고한다. 반환 (문제 목록, 요소 종류별 최소 간격)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px2mm = S.MM_PER_IN / fig.dpi
    ov = geom["axes"]["ov"]
    names = list(data["main"].name)
    out, gmin = [], {}
    for panel in MAP_COLS:
        ax = geom["axes"]["maps"][panel]
        sc = [c for c in ax.collections if (c.get_gid() or "") == f"data|{panel}|targets"][0]
        xy = sc.get_offset_transform().transform(sc.get_offsets()) * px2mm
        polys = []
        for ln in list(ax.lines) + list(ov.lines):
            g = ln.get_gid() or ""
            if g.startswith(f"zoom_rectangle|{panel}") or g.startswith(f"leader|{panel}|") or g == f"zoom_connector|{panel}":
                polys.append((g, ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()])) * px2mm))
        frames = {}
        for key in (f"inset_alaska_{panel}", f"inset_tibet_{panel}"):
            bb = geom["axes"]["maps"][key].get_window_extent(rend)
            x0, y0, x1, y1 = np.array([bb.x0, bb.y0, bb.x1, bb.y1]) * px2mm
            frames[key] = (x0, y0, x1, y1)
            polys.append((f"frame|{key}", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])))
        for n, p in zip(names, xy):
            for g, A in polys:
                if g.startswith("leader|") and g.split("|")[2] == n:
                    continue
                d = _seg_dist(p, A) - CIRCLE_D_MM / 2
                kind = g.split("|")[0]
                gmin[kind] = min(gmin.get(kind, np.inf), round(d, 2))
                if d < min_mm:
                    out.append(f"{panel}: '{n}' 원 테두리와 {g} 간격 {d:.2f} mm")
            for key, (x0, y0, x1, y1) in frames.items():
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                    out.append(f"{panel}: '{n}' 원 중심이 {key} 틀 안")
    return out, gmin


def leader_clearance_check(fig, min_mm=LABEL_PAD_MM) -> tuple[list[str], dict]:
    """d 의 지시선(gid 'leader|d|<방법>')과 다른 방법의 자료 표지(지역 점, 평균, CI 막대) 사이 간격(mm). 자기 방법과 두 줄 묶음 짝
    (D_PAIR)의 평균·CI·지역 점은 뺀다(지시선이 그 평균을 가리킨다). 반환 (문제 목록, 지시선별 최소 간격)."""
    fig.canvas.draw()
    px2mm = S.MM_PER_IN / fig.dpi
    lines = [ln for ln in fig.findobj(Line2D) if (ln.get_gid() or "").startswith(("data|d|", "leader|d|"))]
    leaders = [ln for ln in lines if ln.get_gid().startswith("leader|")]
    out, gmin = [], {}
    for ld in leaders:
        own = ld.get_gid().split("|")[2]
        L = ld.get_transform().transform(np.column_stack([ld.get_xdata(), ld.get_ydata()])) * px2mm
        for ln in lines:
            parts = ln.get_gid().split("|")
            if parts[0] == "leader" or parts[3] in (own, D_PAIR.get(own)):
                continue
            xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()])) * px2mm
            pts = xy if len(xy) == 1 else np.vstack([np.linspace(xy[i], xy[i + 1], 40) for i in range(len(xy) - 1)])
            has_marker = ln.get_marker() not in (None, "None", "none", "", " ")
            r = ((ln.get_markersize() / 2 if has_marker else ln.get_linewidth() / 2) + ld.get_linewidth() / 2) / 72.0 * S.MM_PER_IN
            d = min(_seg_dist(p, L) for p in pts) - r
            gmin[own] = min(gmin.get(own, np.inf), round(d, 2))
            if d < min_mm:
                out.append(f"d 지시선 {own} 과 {ln.get_gid()} 간격 {d:.2f} mm")
    return out, gmin


def tick_spacing_check(fig, axes: dict, category_axes: dict) -> tuple[list[str], dict]:
    """숫자 눈금 라벨 사이 1.5 mm 이상(지침 2.12). 범주 행 이름(c 의 y)은 표 행 간격으로 보고 최소 간격만 잰다."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out, cat = [], {}

    def gaps(axis, attr):
        labs = [t for t in axis.get_ticklabels() if t.get_text() and t.get_visible()]
        labs = sorted(labs, key=lambda t: t.get_window_extent(rend).x0 if attr == "x" else t.get_window_extent(rend).y0)
        res = []
        for t1, t2 in zip(labs, labs[1:]):
            b1, b2 = t1.get_window_extent(rend), t2.get_window_extent(rend)
            res.append((t1.get_text(), t2.get_text(), ((b2.x0 - b1.x1) if attr == "x" else (b2.y0 - b1.y1)) / fig.dpi * 25.4))
        return res
    for k, a in axes.items():
        for axis, attr in ((a.xaxis, "x"), (a.yaxis, "y")):
            if (k, attr) in category_axes:
                g = gaps(axis, attr)
                cat[f"{k} {attr}"] = round(min(v for _, _, v in g), 2) if g else None
                continue
            for t1, t2, gap in gaps(axis, attr):
                if gap < 1.5:
                    out.append(f"{k} {attr}: '{t1}'–'{t2}' 간격 {gap:.2f} mm")
    return out, cat


def extent_check(fig) -> list[str]:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    m = 0.3 / 25.4 * fig.dpi - 0.01         # 0.01 px: 축 좌표 변환의 부동소수 오차(가장자리 0.3 mm 에 맞춘 글자)
    out = []
    for t in drawn_texts(fig):
        b = t.get_window_extent(rend)
        if b.x0 < m or b.y0 < m or b.x1 > W - m or b.y1 > H - m:
            out.append(f"캔버스 밖 또는 여백 부족: '{t.get_text()}' ({b.x0 / fig.dpi * 25.4:.1f}–{b.x1 / fig.dpi * 25.4:.1f} mm)")
    return out


def map_text_inside_circle(fig, geom) -> list[str]:
    """지도 원 밖에 두려 한 글자(삽도 이름, 영구동토 견본)가 원 안으로 들어오는지(바탕 위 글자 금지, 지침 2.2)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out = []
    ax = geom["axes"]["maps"]["a"]
    bb_ax = ax.get_window_extent(rend)
    cxp, cyp, rp = (bb_ax.x0 + bb_ax.x1) / 2, (bb_ax.y0 + bb_ax.y1) / 2, bb_ax.width / 2
    for t in [geom["texts_a"]["inset_label"], geom["texts_a"]["alaska_label"]] + geom["texts_a"]["pf_key"]:
        b = t.get_window_extent(rend)
        corners = [(b.x0, b.y0), (b.x1, b.y0), (b.x0, b.y1), (b.x1, b.y1)]
        dmin = min(np.hypot(x - cxp, y - cyp) for x, y in corners)
        if dmin < rp + 0.3 / 25.4 * fig.dpi:
            out.append(f"'{t.get_text()}' 가 지도 원 안으로 {(rp - dmin) / fig.dpi * 25.4:.1f} mm 들어옴")
    return out


def main_panel_area(info) -> dict:
    a = SLOT["a"][2] * SLOT["a"][3]
    b = SLOT["b"][2] * SLOT["b"][3]
    return dict(a=a, b=b, share=100 * (a + b) / (W_MM * H_MM), largest="a, b (같음)")


# 대상별 Δ 의 두 가중 95 % CI 요약(LG 하네스 h40_label_grid._sig: 셀 가중·블록 등가중 CI 가 모두 0 의 같은 쪽이면
# improve/worse, 그 밖은 ns). 지도에는 그리지 않고 Source Data 에만 둔다(지도는 사후 서술)
SIG_NOTE = {"worse": "95% CIs of both weightings above zero", "improve": "95% CIs of both weightings below zero",
            "ns": "95% CI of at least one weighting includes zero"}


def source_data_table(data, info, geom) -> pd.DataFrame:
    """제출용 Source Data(내부 약호·경로 없음, 그림에 그린 값). drawn_in 은 a·b 대상 원이 그려진 곳
    (본 지도, 알래스카 확대도, 티베트 고원 삽도)이고 c·d·e 행은 비운다."""
    M, arc, F, B, E = data["M"], data["arc"], data["F"], data["B"], data["E"]
    rows = []
    disp = dict(info["disp_mm"])
    disp.update(info["inset_alaska_a"]["disp_mm"])
    for panel, (col, head) in MAP_COLS.items():
        for r in M.itertuples():
            v = M.loc[r.Index, col]
            where = ("Tibetan Plateau inset" if r.name in INSET_TARGETS else "Alaska enlargement" if r.name in AK_TARGETS else "main map")
            rows.append(dict(panel=panel, element="target circle", drawn_in=where, method=head,
                             label=r.display_name, region=r.region_name, target_kind=r.kind, latitude=round(r.lat, 4), longitude=round(r.lon, 4),
                             label_cells=int(r.n_cells), n_labels=0, x_name="", x="", y_name="error change vs source Stefan (cm)",
                             y=round(float(v), 4), lo="", hi="", lo_block_equal="", hi_block_equal="",
                             note=SIG_NOTE.get(str(M.loc[r.Index, f"sig_{col}"]), "no interval (point estimate only)")
                             + (f"; drawn {disp[r.name]:.1f} mm from its centre with a leader line" if r.name in disp and disp[r.name] > LEADER_MIN_MM else "")))
    off = {lab for lab, _ in data.get("offaxis", [])}
    for r in F.itertuples():
        xg = r.source == "xg_tests.csv"
        note = (f"lo, hi: cell-weighted 95% CI; lo_block_equal, hi_block_equal: block-equal 95% CI; block bootstrap, {int(r.nboot):,} resamples"
                + ("; verdict depends on the split-independence assumption" if r.ci_dependence else ""))
        if xg:
            note += ("; existing map without target labels, scored after removing 0.5° scoring blocks within 5 km of product training sites"
                     " (post hoc comparison)")
            if r.fill_frac_blk5 >= 0.0005:
                note += f"; product value missing for {100 * r.fill_frac_blk5:.1f}% of scored cells, gap-filled"
        if r.row_label in off:
            note += "; beyond the x axis in the figure (point estimate printed at the axis end)"
        rows.append(dict(panel="c", element="forest row", drawn_in="", method=r.row_label, label=r.group,
                         region=r.pool_name if xg else "four-region stratified mean",
                         target_kind="", latitude="", longitude="", label_cells="", n_labels=0, x_name="error change vs source Stefan (cm)",
                         x=round(r.delta, 4), y_name="", y="", lo=round(r.ci_lo, 4), hi=round(r.ci_hi, 4),
                         lo_block_equal=round(r.ci_lo_beq, 4), hi_block_equal=round(r.ci_hi_beq, 4), note=note))
    lab = dict(D_METHODS)
    for r in B.sort_values(["kind", "method", "target"]).itertuples():
        rows.append(dict(panel="d", element="four-region mean" if r.kind == "mean4" else "regional value", drawn_in="", method=lab[r.method],
                         label="", region="four-region mean" if r.kind == "mean4" else S.REGION_NAME[r.target], target_kind="", latitude="", longitude="",
                         label_cells="", n_labels=0, x_name="interval width (cm)", x=round(r.wid10, 4), y_name="coverage of 90% interval",
                         y=round(r.cov10, 4), lo=round(r.wid10_lo1, 4) if r.kind == "mean4" else "", hi=round(r.wid10_hi1, 4) if r.kind == "mean4" else "",
                         lo_block_equal="", hi_block_equal="",
                         note=(f"lo, hi: width 95% interval; coverage 95% interval {r.cov10_lo1:.4f} to {r.cov10_hi1:.4f}; bootstrap, {D_NBOOT:,} resamples"
                               if r.kind == "mean4" else f"{int(r.n_blocks)} scoring blocks")))
    for r in E.sort_values(["target", "n"]).itertuples():
        rows.append(dict(panel="e", element="labelled interval", drawn_in="", method="Anchor + residual ML", label="", region=S.REGION_NAME[r.target], target_kind="",
                         latitude="", longitude="", label_cells="", n_labels=int(r.n), x_name="width relative to no labels", x=round(r.width_ratio, 4),
                         y_name="coverage of 90% interval", y=round(r.coverage, 4), lo=round(r.cov_lo, 4), hi=round(r.cov_hi, 4),
                         lo_block_equal="", hi_block_equal="",
                         note=f"lo, hi: coverage 95% interval (block bootstrap, {E_NBOOT:,} resamples); width {r.width_cm:.2f} cm; zero-label width {r.width_r0_n0:.2f} cm;"
                              f" {int(r.n_splits)} splits"))
    return pd.DataFrame(rows)


def values_report(summary: dict, checks: list[dict], plotted: list[dict], data, info, geom, legend_meta: dict) -> list[str]:
    aud = summary["audit"]
    M = data["M"]
    _c5 = data["F"].set_index("key").loc["XG-1c"]
    F_CI_TXT = (f"{_c5.ci_lo:.2f}–{_c5.ci_hi:.2f} cm", f"{_c5.ci_lo_beq:.2f}–{_c5.ci_hi_beq:.2f} cm")
    L = ["Fig 6 v3 수치·점검 기록", "",
         "작성: scripts/4_visualization/paper_v3/fig6.py 실행 때 자동 생성. 그림 명세 FIGURE_SPEC_v3.md 8절, 지침 2절·6.6절·7.1절.", ""]
    L += ["[1] 원천 파일(등록)과 sha256"]
    for p in (LG_CURVE, LGD_CURVE, WF0_RISK, S.PAPER_FIGS / "fig6_a.csv", S.PAPER_FIGS / "fig6_b.csv", S.PAPER_FIGS / "fig6_d.csv",
              LGW_BUNDLE, LGX_TESTS, LGT_TESTS, LGF_TESTS, LGFN_TESTS, LGU_B, LGX_CONFORMAL, TABLE1_ROWS, PFR_NC,
              PROC / "fidelity_base_v3.csv", PROC / "lg_subregion_map_v1.csv", LGU_B_META, LGX_META, CAPTIONS_V2, XG_TESTS, XG_META):
        L.append(f"- {_rel(p)}  {sha256(p)[:16]}…")
    for spec in LGD_RUN_TABLE.values():
        p = PROC / "lgd" / "run_tables" / spec / "fidelity_base_v3.csv"
        L.append(f"- {_rel(p)}  {sha256(p)[:16]}…")
    L += ["- a·b 행 필터: wf0_reanalysis.load()(placement == cell, alpha == 1, learner ∈ {catboost_lo, none}, point_only != True, LGD 는 KEEP_LGD),",
          "  deltas() 의 n == 0 행에서 D0_1.0, D1_1.0 (= 방법 rmse − rmse_p0, cm). 지도는 target ∈ {독립 7, 하위 지역 10} × 모드 x(D-9).",
          "- 대상 중심: 지역은 fidelity_base_v3.csv 의 macro == 지역, 하위 지역은 lg_subregion_map_v1.csv 의 sub == ID, LGD 대상은 실행 표의 macro.",
          "- c 행 필터: fig6_a.csv (block, key, variant) 14쌍(모듈 FOREST) + 봉인 xg_tests.csv 의 hypothesis XG-1c, XG-1w 2행(모듈 FOREST_XG; 주 열 delta·ci_lo·ci_hi·"
          "delta_blockeq·ci_lo_beq·ci_hi_beq = 5 km 블록 마스크). d: fig6_b.csv method ∈ 5종, kind ∈ {region, mean4}. e: fig6_d.csv 6행.",
          "- 계열 대응(코드 안에서만): D0 = Direct ML, D1 = Physics pseudo-labels, P* = Year-matched Stefan, B:ens = Anchor ensemble, F1k = CatBoost, physics inputs,",
          "  R1 λ 0.25 = Anchor + residual ML(e), 구간 척도화 const/phys/nflow/nflow#placebo/cbq = Constant/Physics prior/Normalizing flow/Permuted flow/CatBoost quantile.", ""]
    L += ["[2] 그린 수치 전수 대조(그림 객체 → 자료 표; 자료 표 → 원천 표는 [3])"]
    L.append(f"- 그림 객체에서 읽은 수치 {summary['n_numbers']}개, 요소 {len(plotted)}개, 불일치 {summary['n_bad']}개 "
             f"({'통과' if summary['n_bad'] == 0 else '실패'})")
    for r in plotted:
        L.append(f"- {'통과' if r['ok'] else '실패'} | {r['panel']} | {r['element']} | {r['label']} | 값 {r['plotted']} | {r['source']} | {r['filter']}")
    L.append("- 기준선·띠(자료 아님): c 0선 0 cm 와 ±0.5 cm 띠(지침 2.3), d 명목 포함률 0.90 선, e 명목 띠 0.85–0.95(그림 명세 8.3), 컬러바 눈금 "
             f"{[int(v) for v in np.linspace(-info['vmax'], info['vmax'], 5)]} cm, 축척 1000 km(본 지도)·500 km(알래스카 확대도)·200 km(티베트 삽도),"
             " 위도 라벨 60·70·80° N.")
    L.append("")
    L += ["[3] 원천 대조와 설명문 수치"]
    for c in checks:
        L.append(f"- {'통과' if c['ok'] else '실패'}: {c['name']}. {c['detail']}")
    L.append("")
    L += ["[4] 그림 점검(지침 부록 A.1 audit_v3, A.2 pdf_audit, 7.1)"]
    L.append(f"- audit_v3: 글자 크기 {sorted(aud['sizes'])} pt, 문자 수 {aud['chars']}(한도 800, 그림 명세 8.5 사유: 포레스트 행 이름), 얇은 선 {aud['thin_lines'] or '0'},"
             f" 제목 {aud['titles'] or '0'}, 글씨 상자 {aud['boxed_text']}, 내부 약호 {aud['codes'] or '0'}, 네 자리 쉼표 {aud['comma4'] or '0'},"
             f" 긴 라벨 {aud['long_labels'] or '0'}, 그림 안 수치 {aud['loose_numbers'] or '0'}; 실패 항목 {aud['fails'] or '없음'}")
    L.append("  (그림 안 수치 검사 제외 gid: scale(축척 길이), sizekey(열쇠 값 '±0.5 cm', '40 labels', '160 labels'), graticule(위도 라벨 3개), offaxis(축 밖 행의"
             " 점 추정값 '+23.8', 조정 담당 요청 '가장자리 화살표와 값'). 앞의 셋은 명세 8.3 이 허용한 수치다)")
    pa = summary["pdf"]
    L.append(f"- pdf_audit: {pa['width_mm']} × {pa['height_mm']} mm, 문자 {pa['chars']}, 크기 분포 {pa['sizes']}, 5 pt 미만 {pa['lt5']},"
             f" 글꼴 계열 {pa['families']}, Type 3 {'있음' if pa['type3'] else '없음'}")
    for ln in summary["fonts"].strip().splitlines()[2:]:
        L.append(f"- pdffonts: {' '.join(ln.split()[:2])} 내장 {ln.split()[-5]}")
    L.append(f"- pdfimages(내장 래스터 ppi): {summary['pdfimages'] or '없음'}")
    L.append(f"- 글자 겹침·글자와 자료 겹침(직접 라벨은 자료 표지에서 {LABEL_PAD_MM} mm 이상): {summary['overlaps'] or '0건'}")
    to = summary["text_overlaps"]
    L.append(f"- style.text_overlaps: 글자 {to['n_texts']}개, 겹침 쌍 {to['overlap_pairs'] or '0'}, 캔버스 밖 {to['outside'] or '0'}")
    L.append(f"- 지도 원 테두리와 다른 선·틀 사이 {MAP_CLEAR_MM} mm 미만, d 지시선과 다른 방법의 자료 표지 사이 {LABEL_PAD_MM} mm 미만: {summary['clearance'] or '0건'};"
             f" 지도 종류별 최소 간격(mm) {info['map_clearance_min_mm']}, d 지시선별 최소 간격(mm) {info['leader_clearance_min_mm']}")
    L.append(f"- 지도 원 밖 글자가 원 안으로 들어옴: {summary['circle_text'] or '0건'}")
    L.append(f"- 눈금 라벨 간격 1.5 mm 미만: {summary['tick_spacing'] or '0건'}")
    L.append(f"- 캔버스 밖 글자: {summary['extent'] or '0건'}")
    L.append(f"- 지역 모양 열쇠 오른쪽 끝 {info['region_key_right_mm']:.1f} mm(캔버스 170 mm)")
    area = main_panel_area(info)
    L.append(f"- 주 패널 면적(F-17, 슬롯 기준): a {area['a']:.0f}, b {area['b']:.0f} mm², 합 {area['share']:.0f}%(기준 35% 이상), 가장 큰 패널 {area['largest']}")
    ak = info["alaska_inset"]
    L.append(f"- 지도: 북극 평사 투영(중심 경도 {LON0}°, 진척 위도 {TRUE_LAT}° N), 50° N 원 지름 {MAP_D_MM} mm, 1 mm = {info['km_per_mm']:.1f} km(70° N 기준),"
             f" 축척 막대 1000 km 의 측지 길이 {info['scale_a']['geodesic_km']:.0f} km(축척 계수 {info['scale_a']['k']:.3f});"
             f" 내장 래스터 지도 {info['pfr_raster_a']}; 영구동토 바탕 '{info['pfr_title']}'; 해안선 Natural Earth 50 m; Cartopy {summary['cartopy']}")
    L.append(f"- 티베트 삽도: Lambert 방위 등적(중심 = 대상 중심), {INSET_W_MM} × {INSET_H_MM} mm, 1 mm = {info['inset_tibet_a']['km_per_mm']:.0f} km,"
             f" 축척 200 km 의 측지 길이 {info['inset_tibet_a']['scale']['geodesic_km']:.0f} km, 래스터 {info['inset_tibet_a']['raster']}")
    L.append(f"- 알래스카 확대도: 북극 평사(중심 경도 {ak['lon_c']:.2f}°, 진척 위도 70° N), {AK_INSET_W_MM} × {AK_INSET_H_MM} mm, 범위 {ak['extent_km'][0]:.0f} × {ak['extent_km'][1]:.0f} km,"
             f" 1 mm = {ak['km_per_mm']:.1f} km(본 지도의 {info['km_per_mm'] / ak['km_per_mm']:.1f}배), 축척 500 km 의 측지 길이 {info['inset_alaska_a']['scale']['geodesic_km']:.0f} km,"
             f" 원 최소 중심 간격 {info['inset_alaska_a']['min_pair_gap_mm']:.2f} mm, 옮긴 거리(mm) {dict((k, float(v)) for k, v in info['inset_alaska_a']['disp_mm'].items() if v > 0)},"
             f" 래스터 {info['inset_alaska_a']['raster']}; 본 지도 확대 사각형 꼭짓점(표시 mm) {info['zoom_rect_corners_mm_a']}")
    L.append(f"- 본 지도 대상 원: 지름 {CIRCLE_D_MM} mm 고정, 최소 중심 간격 {info['min_pair_gap_mm']:.2f} mm(기준 {CIRCLE_GAP_MM}), 가장 바깥 원 반지름 {info['max_r_mm']:.1f} mm(원 반지름 {MAP_D_MM / 2});"
             f" 겹침 묶음 {info['clusters']}; 옮긴 거리(mm) {dict((k, float(v)) for k, v in info['disp_mm'].items() if v > LEADER_MIN_MM)}"
             f"(이 원들은 실제 위치 점과 지시선으로 이었다. {LEADER_MIN_MM} mm 이하로 옮긴 원 {sum(1 for v in info['disp_mm'].values() if 0 < v <= LEADER_MIN_MM)}개)")
    L.append(f"- c 행 이름 사이 최소 간격 {info.get('category_row_gap_mm')} mm(행 간격 약 2.7 mm = 7 pt 글자 높이 2.5 mm + 0.4 mm). 지침 2.12 의 1.5 mm 는 숫자 눈금 라벨 규칙이고"
             " 범주 행 이름은 표 행 간격(약 1.1배)으로 두었다 [판단]. 14행 + 묶음 머리 4줄을 명세의 64 mm 슬롯에 넣으려면 이 간격이 필요하다."
          f" XG 묶음(머리 1줄 + 2행)을 더하면서 같은 간격을 지키려고 c 슬롯을 {SLOT['c'][3]:.1f} mm 로 늘렸다(행 간격 2.715 mm 유지).")
    L.append(f"- 컬러바: ±{info['vmax']:.0f} cm(TwoSlopeNorm vcenter 0, cmc.broc), 연장 표지 '{info['ext_kind']}', 범위 밖 값 {info['over']}")
    L.append("")
    L += ["[5] 판단과 명세에서 바꾼 점([판단])",
          f"- 지도 범위를 Fig 1a 와 같게(50° N 원, 중심 경도 {LON0}° E) 두었다(그림 명세 8.3). 원 지름은 {MAP_D_MM} mm 로 슬롯(82 × 76 mm)에서 열 머리를 뺀 72.4 mm 보다"
          f" 조금 작게 하고 오른쪽에 맞췄다. 왼쪽 아래 구석에 티베트 삽도({INSET_W_MM:.0f} × {INSET_H_MM:.0f} mm), 오른쪽 아래 구석에 알래스카 확대도"
          f"({AK_INSET_W_MM:.0f} × {AK_INSET_H_MM:.0f} mm), 왼쪽 위 구석에 영구동토 견본 2줄을 두기 위해서다.",
          "- 알래스카 확대도 폭은 23 mm(1차 렌더)에서 20 mm 로 줄였다. 23 mm 에서는 확대도 틀이 동시베리아 원 테두리에 0.07 mm, 본 지도의 확대 사각형이"
          " 캐나다 하위 지역 원 테두리에 0.03 mm 까지 붙었다. 확대 배율은 남북 방향이 정하므로 바뀌지 않는다. 확대도 축척 막대는 오른쪽 아래 구석에 두었다"
          "(왼쪽 아래 구석에 하위 지역 원이 있다). 축척 막대 길이는 진척 위도 70° N 기준 투영 길이이고 측지 길이는 위 [4] 에 적었다.",
          "- 알래스카 7대상(지역 + 하위 지역 6)은 본 지도에서 9 × 7 mm 안에 모여 명세 8.3 의 지시선 부채꼴로는 지시선이 엇갈리고 원이 최대 8 mm(약 1000 km) 옮겨져 읽을 수 없었다"
          "(1차 렌더). 지침 2.11 '밀집 구역은 확대도'(Hjort 2018 Fig 1)와 Fig 1a 의 확대 문법을 따라 확대도로 바꾸고, 본 지도에는 확대 범위 사각형(1.0 pt 검정)과"
          " 확대도 틀로 가는 연결선 2개(1.0 pt #bdbdbd)만 두었다. 레나델타 3대상은 명세대로 지시선 부채꼴이다.",
          "- 삽도·확대도 이름('Tibetan Plateau', 'Alaska')은 a 의 각 틀 아래에 한 번씩 두었다(틀 위쪽은 지도 원과 겹친다). b 의 삽도·확대도는 같은 범위이고 축척·이름을 반복하지 않는다(H13).",
          f"- 위도 라벨 경선은 {LAT_LABEL_LON}°(Fig 1a 는 −20°), 축척 막대 중심은 ({SCALE['lon']}° E, {SCALE['lat']}° N)(Fig 1a 는 5° E)이다."
          " 72 mm 보다 작은 지도에서 80° N 표지와 축척 글자를 바다 위 빈 곳에 두기 위해서다.",
          "- 대상 원이 겹치는 묶음(세 원 이상)은 묶음 중심에서 극 반대쪽으로 연 부채꼴(묶음 크기에 따라 반지름 5.3–7.8 mm)에 두고 실제 위치의 작은 점과 1.0 pt 회색 지시선으로"
          " 이었다(그림 명세 8.3 '지시선 부채꼴'). 두 원 묶음은 잇는 선 위에서 맞선 방향으로 최소 이동했다. 0.5 mm 넘게 옮긴 원은 모두 지시선을 그렸다.",
          f"- 컬러바 연장 표지는 '{info['ext_kind']}' 이다. 명세 8.3 의 '양끝 연장 표지 [판단]' 과 달리, 범위를 넘는 값은 음의 끝(티베트 직접 ML)뿐이라 양의 끝에는 표지를 두지 않았다.",
          "- vmax 계산의 값 수는 32(티베트를 뺀 16대상 × 2)이다. 명세 8.3 의 '34값(17대상 × 2)' 은 티베트를 빼면 32값이 된다.",
          "- 포레스트 행 이름 'CatBoost, larger' 는 명세의 'CatBoost, default' 를 바꾼 것이다. catboost 는 기본 설정이 아니라 600회·깊이 6 으로 주 학습기(200회·깊이 3)보다 용량이 크다"
          "(polar.m1_core 287–295행, C1 README 2.1 '용량 확대').",
          "- 컬러바 라벨과 c 의 x 축 이름은 같은 문구 'Error change vs source Stefan (cm)' 이다. 그림 명세 8.3 이 두 축에 같은 이름을 지정했고(지침 2.12 의 Δ 축 기준 표기)"
          " 두 축은 같은 양이다. H13 의 '같은 문구 반복' 은 요소 라벨에 대한 규칙으로 보고 축 이름은 명세대로 두었다.",
          "- d 의 거의 겹치는 두 평균(상수·물리 사전, 정규화 흐름·순열 흐름)은 두 줄로 쌓은 라벨 한 묶음과 지시선 하나로 가리킨다(위 줄 = 포함률이 높은 쪽). 읽는 법은 설명문에 적었다.",
          f"- d 라벨 위치(1차 렌더에서 고침): 상수·물리 사전 묶음은 x {D_LABEL_POS['phys'][0]:.0f} cm, y {D_LABEL_POS['phys'][1]}·{D_LABEL_POS['const'][1]}."
          " x 65 cm 에서는 물리 사전 지시선이 동시베리아 지역 점을 0.24 mm 로 스쳤고 'Constant' 가 상수 CI 위끝에 0.44 mm 로 붙었다."
          " 위 줄 글상자가 축 위끝을 약 0.6 mm 넘지만 위 테두리가 없고 지역 모양 열쇠와 1 mm 이상 떨어진다. 정규화 흐름 지시선은 평균 2.0 mm 앞에서 멈춘다"
          "(평균 옆 캐나다 지역 점 두 개와 0.5 mm 이상 띄우기 위해).",
          "- d 의 검정은 '구간 방법 추정' 한 역할이다(그림 명세 D-17, 지침 2.4 검정 역할 목록 밖이라 사용자 확인 필요).",
          "- e 는 관측 대 예측(지침 6.6)이 아니라 라벨 있는 잔차 구간(그림 명세 D-16)이다. 셀 단위 구간 파일이 없다.",
          "- 예측 지도 행(ALT 예측, 구간 폭, 외삽 영역)은 본문에 두지 않았다(LGU-B2 미결정, 그림 명세 8.3). a·b 가 본문 결과 지도이다.",
          "- 지도의 관측 위치(F-12)는 대상 중심 원으로 보인다. 개별 라벨 셀은 Fig 1a 에 있고 이 지도에는 그리지 않았다(결과 지도의 주제가 대상별 Δ 이기 때문).",
          "- 슬라이드판(medium='slide')은 이 비교판에서 만들지 않았다(그림 명세 1.7 의 두 매체 규칙은 채택 뒤 구현).",
          "- SVG 는 내지 않았다(이 폴더의 다른 v3 그림과 같이 PDF 정본 + PNG 600 dpi 검토용).",
          f"- 기존 ALT 지도 묶음(2026-10-05, 조정 담당 요청): c 마지막에 'Existing ALT maps' 두 행(ESA CCI v5 = XG-1c 주 4지역, Wei v2 = XG-1w 레나델타·캐나다 2지역)을"
          " 더했다. 값은 봉인 xg_tests.csv 의 주 열(5 km 블록 마스크)이고 25 km 마스크 판정도 같다([3] 10). 결과 열람 뒤 설계·등록 이탈(WRAPUP 10)이라 설명문에 'added post hoc' 로 적었다."
          " Wei 풀이 2지역이라 행 이름에 'two regions' 를 넣었다(덱 부록 3 은 4지역 평균 그림이라 Wei 를 뺐다).",
          f"- ESA CCI v5 의 두 가중 CI(셀 {F_CI_TXT[0]}, 블록 {F_CI_TXT[1]})는 축 끝 {FOREST_XLIM[1]} cm 보다 오른쪽이다. 다른 행을 줄이지 않으려고 축을 넓히지 않고"
          " 축 끝에 1.0 pt 화살표(3.2 mm)와 점 추정값 '+23.8'(7 pt)을 두었다. 덱 부록 3 은 가로축을 끊어 둘째 축에 그렸으나 c 슬롯 폭(96 mm, 이름 열 40 mm) 안에는"
          " 둘째 축을 같은 축척으로 둘 자리가 없다. 정확한 값과 CI 는 설명문과 Source Data 에 있다.",
          f"- 그림 높이 {H_MM:.1f} mm(명세 160 mm, 지침 상한 {S.HMAX_MM:.0f} mm): c 에 3줄을 더한 높이 {XG_EXTRA_MM} mm. 오른쪽 d·e 슬롯은 아래 빈 띠가 생기지 않게"
          f" 그 높이를 반씩 나눠 가졌다(d 높이 {SLOT['d'][3]:.2f} mm, e 위끝 {SLOT['e'][1]:.2f} mm·높이 {SLOT['e'][3]:.2f} mm). a·b·컬러바 위치는 그대로다(덱 자르기 f6a, f6b,"
          " f6_cbar 상자 유효). 덱 자르기 f6c 는 아래쪽이 늘어난 만큼 상자를 늘려야 한다.", ""]
    L += ["[6] 색각·흑백(F-09)",
          "- 연속 색표 cmc.broc(Crameri, 색각 이상에서도 단조로운 명도) ±vmax 공유, 0 중심. 대상 원 테두리 검정, 지시선 #737373, 영구동토 바탕 #d0d0d0·#e6e6e6, 육지 #f4f4f4.",
          "- v4 토큰 색: c 는 행마다 방법·제품 색, d 는 구간 방법마다 color.intervals, e 는 conformal 색 " + E_COLOR + "(흰 바탕 대비 "
          + f"{summary['purple_contrast']:.2f}" + ":1). 지도 색표는 그대로다.",
          "- 흑백 렌더(사람 판정): a·b 는 broc 의 명도가 0 에서 가장 밝고 양끝에서 어두워 부호를 명도만으로 구분할 수 없다. 부호는 컬러바 위치와 설명문의 대상 수로 읽는다."
          " e 의 40·160 라벨은 채움·빈 마커로 구분된다.", ""]
    L += ["[7] 설명문(Fig6_legend.md, 손으로 작성)",
          f"- 단어 수 {legend_meta.get('words', '없음')}(자리표시 제외 {legend_meta.get('words_wo_placeholder', '없음')}), 자리표시 {legend_meta.get('placeholders', [])}", ""]
    L += ["[8] 산출 파일"] + [f"- {p}" for p in summary["paths"]] + [f"- {LEGEND_MD}(손으로 작성)", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium", default="paper", choices=["paper", "slide"])
    ap.add_argument("--draft", action="store_true", help="PNG 만 저장하고 점검 요약만 출력(--out 필수)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-wait", action="store_true", help="자원 확인을 건너뛴다")
    args = ap.parse_args()
    if args.medium == "slide":
        raise SystemExit("슬라이드판은 이 비교판에서 만들지 않는다(Fig6_values.txt [5])")
    if not args.no_wait:
        print("자원:", check_resources(), flush=True)
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, data, info, geom = build("paper")

    aud = S.audit_v3(fig, max_chars=800, allowed_num_gids=("scale", "sizekey", "graticule", "offaxis"))
    plotted, n_num, n_bad = plotted_numbers(fig, data)
    ovl = overlap_check(fig, geom)
    tov = S.text_overlaps(fig)
    clr, clr_min = map_clearance_check(fig, geom, data)
    info["map_clearance_min_mm"] = clr_min
    lcl, lcl_min = leader_clearance_check(fig)
    info["leader_clearance_min_mm"] = lcl_min
    clr += lcl
    circ = map_text_inside_circle(fig, geom)
    tsp, cat_gap = tick_spacing_check(fig, dict(c=geom["axes"]["c"], d=geom["axes"]["d"], e=geom["axes"]["e"], cbar=geom["axes"]["cbar"]),
                                      category_axes={("c", "y")})
    info["category_row_gap_mm"] = cat_gap
    ext = extent_check(fig)
    if args.draft:
        p = out_dir / f"{STEM}.png"
        fig.savefig(p, dpi=600)
        print("saved", p)
        print("audit_v3:", {k: (sorted(v) if isinstance(v, set) else v) for k, v in aud.items()})
        print("plotted numbers:", n_num, "bad:", n_bad)
        print("overlaps:", ovl)
        print("text_overlaps:", tov)
        print("drawn texts:", len(drawn_texts(fig)))
        print("map clearance:", clr, clr_min, lcl_min)
        print("circle_text:", circ)
        print("tick spacing:", tsp)
        print("extent:", ext)
        print("info:", {k: v for k, v in info.items() if k in ("vmax", "p99", "ext_kind", "over", "min_pair_gap_mm", "max_r_mm", "clusters", "disp_mm", "region_key_right_mm", "km_per_mm")})
        return

    paths = S.save_fig(fig, STEM, out_dir, formats=("pdf", "png"))
    sd = source_data_table(data, info, geom)
    sd_path = out_dir / f"{STEM}_source_data.csv"
    sd.to_csv(sd_path, index=False, encoding="utf-8")
    pdf = paths[0]
    pa = S.pdf_audit(pdf)
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    imgs = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True).stdout
    img_lines = [" ".join(ln.split()[:1] + ln.split()[3:6] + ln.split()[12:14]) for ln in imgs.strip().splitlines()[2:]]
    import cartopy
    sys.path.insert(0, str(ROOT / "src"))
    from polar import cvd
    summary = dict(audit=aud, n_numbers=n_num, n_bad=n_bad, overlaps=ovl, text_overlaps=tov, clearance=clr, circle_text=circ, tick_spacing=tsp,
                   extent=ext, pdf=pa, fonts=fonts,
                   pdfimages=img_lines, cartopy=cartopy.__version__, purple_contrast=float(cvd.contrast(E_COLOR)),
                   paths=[str(p) for p in paths] + [str(sd_path)])
    checks = source_checks(data, info, geom)
    lchk, lmeta = legend_checks()
    checks += lchk
    (out_dir / f"{STEM}_values.txt").write_text("\n".join(values_report(summary, checks, plotted, data, info, geom, lmeta)) + "\n", encoding="utf-8")
    print(json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in summary.items() if k not in ("audit", "fonts")}, default=str,
                     ensure_ascii=False, indent=1))
    print("audit_v3:", {k: (sorted(v) if isinstance(v, set) else v) for k, v in aud.items()})
    print("checks failed:", [c["name"] for c in checks if not c["ok"]])
    print("plotted numbers:", n_num, "bad:", n_bad)


if __name__ == "__main__":
    main()
