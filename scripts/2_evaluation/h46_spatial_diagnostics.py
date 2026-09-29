"""H46 · 공간 정보량 진단(LGU 실험 C). 계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 5절(LGU-C1–C4)의 구현.

목적
  검토 문서(docs/GENERATIVE_MODEL_REVIEW_2026-09-29.md)의 SP0 가운데 D1(변동도, 블록 내 상관, 근접 라벨 비율)과 D2(SAR 제품과 탐침의
  정렬)를 산출 표로 만든다. 가설과 판정은 없다. 학습도 없다. 네 표 모두 검토 단계에서 탐색값을 이미 보았으므로 '재현(비맹검)'이다(계획서 6절).
  결과의 쓰임은 한계 절의 제품 정렬 문장 1개(LGU-C4), SI 표, 후속 안(계획서 9절)의 재심 조건 확인이다.

동결 규칙과 누설 규약
  h40(scripts/3_deep_learning/h40_label_grid.py)과 src/polar 의 기존 모듈, src/polar/lgu_common.py 를 고치지 않는다. h40 은 importlib 로 파일
  경로에서 읽어 sys.modules['h40_label_grid'] 에 등록하고(모듈 최상위, h42 와 같은 방식) 자료 적재(Data), 원천 색인(source_idx, 100 km 버퍼),
  분할 구조(split_structure), 라벨 추출(draw_cells, cells_of)을 그대로 쓴다. 조각 도우미, 셀 표지, 최근접 거리는 lgu_common 을 쓴다.
  LGX 의 산출(data/processed/lgx)과 LG 의 산출은 읽지 않는다. 입력은 load_base 의 표(fidelity_base_v3.csv 와 토양 도일 표), 하위 지역 대응표,
  셀 표지 표뿐이다. 격자 원자료(SAR 영상, CCI 격자)는 읽지 않고 표에 있는 insar_alt, polsar_alt, polsar_valid, polsar_std 열만 쓴다.
  근접 표(LGU-C3)는 좌표와 추출 색인만 쓰고 라벨 값과 예측값을 쓰지 않는다. 변동도, 블록 내 상관, SAR 정렬은 라벨 값을 쓰는 진단이다(비맹검).

방법 정의(계획서 5.1–5.2절)
  공통: z = log y − log s(s = e5_sqrt_tdd), w = z − (macro 지역의 z 평균). 지역 기본값은 Alaska, Lena, Canada. 층은 all, gpr, probe,
    noearly(early 제외)이고 셀 표지 표(lgx_label_flags_v1.csv, 값 0.5 이상이 참)를 loc_id 로 결합한다. 층의 셀이 30개 미만이면 행을 만들지
    않고 meta 에 적는다. 거리는 대권 거리(km, 반지름 6,371 km)다.
  LGU-C1 변동도(lgu_c_variogram.csv): 셀을 --chunk 개씩 나눠 나머지 셀과의 거리 행렬(float32)을 만들고 j > i 인 쌍만 센다. 구간 경계는
    --bins(0, 0.1, …, 200 km, 구간은 [lo, hi))이고 마지막 경계 밖의 쌍은 세지 않는다. gamma = Σ 0.5(w_i − w_j)² / n_pairs,
    gamma_robust = (평균 |Δ|^0.5)^4 / (2·(0.457 + 0.494/n_pairs))(Cressie–Hawkins), rho = 1 − gamma/var_w(var_w 는 층 셀의 표본 분산, ddof 1).
    jk_se 는 블록 하나 제외 잭나이프 표준오차(서술용)다. gamma_(−b) = (S − S_b)/(N − N_b), S_b = (블록 b 셀의 셀별 누적 합) − (블록 b 안 쌍의 합).
    지수 모형 γ(h) = c0 + c1(1 − exp(−h/a))을 scipy.optimize.least_squares 로 적합한다(가중 sqrt(n_pairs), n_pairs ≥ 30 인 구간, 거리 = mean_h_km,
    경계 c0 ≥ 0, c1 ≥ 0, a ∈ [0.05, 500] km, 초기값 (첫 사용 구간의 gamma, var_w − c0, 2)). 적합 행은 row = 'fit' 이다.
  LGU-C2 블록 내 상관(lgu_c_icc.csv): 블록별 셀 수 n_j, 평균, 블록 내 제곱합으로 sigma2 = Σ SSW / Σ(n − 1),
    tau_b2 = max(Var_j(블록 평균) − sigma2·mean_j(1/n_j), 1e-4)(Var 는 ddof 1), icc = tau_b2/(tau_b2 + sigma2), sd_within = sqrt(sigma2).
    CI 는 지역 안 블록 재표집 --nboot 회(RandomState(seed_of('lgu-c2', 지역, 층)))의 백분위 95 % 다.
  LGU-C3 근접 라벨 비율(lgu_c_proximity.csv): 대상·모드·분할마다 A_idx, B_idx = half_split_blocks, 채점 셀 = B 의 eval_mask 셀.
    분할의 유효성은 h40.enumerate_units 와 같은 규칙이다(중복 분할 제외, 유효 분할이 있는 대상의 무효 분할 제외). (n, 추출)은 h40.cells_of,
    선택 셀은 h40.draw_cells(대상, 모드, 분할, n, 추출, |A|)의 A_idx 안 색인이다(LG 와 같은 셀). all 은 n = −1 이다. 채점 셀에서 선택 라벨까지의
    최근접 거리(lgu_common.nearest_km)로 frac_within(d) = 거리 ≤ d 인 채점 셀 비율을 구하고 추출 평균, 최소, 최대를 낸다. med_nn_km 는 최근접
    거리 중앙값의 추출 평균, frac_within_allA 는 A 셀 전체 기준 비율, med_src_km 는 최근접 원천 셀(h40.source_idx, 100 km 버퍼 뒤) 거리의 중앙값이다.
    split = −1 은 분할 평균 행이다.
  LGU-C4 SAR 정렬(lgu_c_sar.csv): 제품 insar_alt(유한한 셀), polsar_alt(polsar_valid > 0 이고 유한한 셀). 단위는 지역 Alaska, Canada 와
    하위 지역 AL-1–AL-6, CA-1–CA-3(Data.df 의 sub 열). 층은 all, gpr, probe, lt99_5(polsar_alt < 99.5, polsar 에만). 척도는 cm 와 log.
    r_cell(Pearson), rho_cell(Spearman, 평균 순위), r_between(셀 3개 이상 블록의 블록 평균 사이 Pearson), r_within(셀 5개 이상 블록에서 블록 평균을
    뺀 값을 합동한 Pearson), bias_cm = 평균(제품 − y), rmse_cm. r_within 의 CI 는 블록 재표집 --nboot 회. 셀 30개 미만 또는 블록 3개 미만인 단위는
    값을 NaN 으로 두고 flag 를 적는다. 요약 행(unit = 'SUMMARY9')은 하위 지역 9개 가운데 r_within_lo > 0.2 인 수다(SP2 재심 조건의 앞부분).
  polsar 분류 부호 점검(lgu_c_sar_codes.csv, 명세 밖의 추가 표): 단위·제품마다 99.5 이상 값의 수(polsar_valid 참·거짓별), 95–105 창과 정확히
    100 인 값의 수, 99.5 이상 값의 고유값 수와 정수 비율, 그 셀과 나머지 셀의 y 중앙값, 가장 많은 고유값 10개와 그 수. 판정 문장은 두지 않는다.

작업 단위와 산출(<out-dir>, 기본 data/processed/lgu)
  하위 단위: variogram·icc 는 지역, proximity 는 (대상, 모드), sar 는 단위다. 하위 단위마다
    shards/<tag>__<항목>__<하위>_{part.csv, unit.json}(sar 는 _codes.csv 도)을 원자적으로 쓰고 unit.json 을 마지막에 쓴다(완료 표지).
    하위 단위는 --workers 개의 spawn 프로세스에서 돈다(0 또는 1 이면 주 프로세스에서 차례로).
  항목 단위: 집계 단계가 하위 단위 조각을 모아 표 lgu_c_<항목>.csv 를 원자적으로 쓰고 shards/<tag>__<항목>_unit.json 을 남긴다.
    하위 단위 가운데 하나라도 없거나 실패이거나 설정 해시가 다르면 그 항목의 표를 쓰지 않고 lgu_c_failed.csv 에 남기며 종료 코드는 1 이다.
  그 밖: lgu_c_meta.json(인자, 코드 해시, 입력 표 해시, 지역·층별 셀 수, 만들지 않은 행의 사유, 시간, blind = '재현(비맹검)'),
    lgu_c_timing.csv, lgu_c_failed.csv, lgu_c_count.csv(--count-only). 스모크와 사전 점검은 이름에 _smoke, _precheck 가 붙는다.
  --resume 은 LC.unit_state(설정 해시가 같고 status 가 failed 가 아닌 조각)로 하위 단위를 건너뛴다. 설정에는 항목별 설계값과 입력 표 해시가
  들어간다(코드 해시는 unit.json 에 따로 적고 해시에는 넣지 않는다. h42 와 같다). --chunk 는 결과를 바꾸지 않으므로 설정 해시에서 뺀다.

실행 환경(사용자 지시 2026-09-30)
  LGU 는 Rescale 이 아니라 로컬 GPU 서버에서 돈다. 이 하네스는 GPU 를 쓰지 않는다(CUDA_VISIBLE_DEVICES = '', --gpus 는 받기만 한다).
  실행 보호: LG_RESCALE=1 또는 --allow-local 이 없으면 --count-only 만 실행하고 나머지(스모크, 사전 점검, 본 실행, 집계)는 자료를 읽기 전에
  거부한다(LC.require_permission). 허용 표지가 없으면 스레드는 1 이다. 허용된 실행에서(LG_RESCALE=1 이어도 같다, 개정 1) --workers ×
  --threads 가 8 을 넘으면 거부한다(이 실험의 CPU 스레드 합계 8 이하 규칙). h44·h45 와 겹쳐 돌 때의 합계는 실행 중 등록부
  (data/processed/lgu/.running, LC.claim_threads)로 확인해 8 을 넘으면 거부한다. 실행 중 프로세스의 nice 값이 10 미만이면 10 으로 올린다.

실행 예(ROOT)
  적합 수:   python3 scripts/2_evaluation/h46_spatial_diagnostics.py --count-only --threads 1
  스모크:    nice -n 10 python3 scripts/2_evaluation/h46_spatial_diagnostics.py --smoke --allow-local --workers 1 --threads 2
  사전 점검: nice -n 10 python3 scripts/2_evaluation/h46_spatial_diagnostics.py --precheck --allow-local --workers 1 --threads 2
  본 실행:   nice -n 10 python3 scripts/2_evaluation/h46_spatial_diagnostics.py --allow-local --workers 2 --threads 4 --resume
  집계만:    nice -n 10 python3 scripts/2_evaluation/h46_spatial_diagnostics.py --allow-local --summarize-only --workers 1 --threads 1

명세(spec_C)와 다르게 구현한 점
  1. 시험 파일 이름은 tests/test_h46_spatial.py 다(과제 지시). spec_C 는 tests/test_h46_diag.py 였다.
  2. 작업 단위를 항목 × 하위 단위(지역, 대상, SAR 단위)로 나눴다. 하위 단위마다 part.csv 와 unit.json 을 두고, 항목 단위의 unit.json 과 표는
     집계 단계에서 쓴다. --resume 의 단위가 작아지고 워커 분배가 쉬워진다. 표의 값은 나누지 않은 경우와 같다.
  3. 인자를 더했다: --sar-units(SAR 단위 목록), --seeds(학습이 없어 쓰지 않는다. 기록만 한다), --precheck(지역 Alaska, 근접 대상 Alaska x,
     분할 1, SAR 단위 Alaska, nboot 2,000. 시간 측정용), --summarize-only, --no-summarize, --buffer-km(고정값 100, h40 의 원천 버퍼).
  4. 표에 열을 더했다. icc: flag(자유도 10 미만, 블록 5개 미만 등), nboot. proximity: n_draws, n_splits. sar: bias_scale, rmse_scale(행의 척도,
     log 행이면 로그 단위), n_blocks_between, n_blocks_within, n_units_eval, n_units_pass(요약 행), nboot. sar 의 bias_cm, rmse_cm 은 척도와 무관하게
     그 행의 셀에서 cm 로 계산한다(log 행은 양수 셀만 들어가므로 cm 행과 셀 집합이 다를 수 있다). 명세의 열 이름 _cm 을 지키기 위해서다.
  5. polsar 분류 부호 점검 표 lgu_c_sar_codes.csv 를 더했다(과제 지시). insar_alt 에도 같은 요약을 낸다.
  6. var_w 와 Var_j(블록 평균)는 ddof 1 이다. var_w 를 ddof 1 로 두면 모든 쌍이 구간 안에 있을 때 쌍 가중 gamma 평균이 var_w 와 정확히 같다.
  7. r_between 과 r_within 은 조건을 만족하는 블록(셀 3개 이상, 셀 5개 이상)이 3개 미만이면 NaN 이고 flag 에 적는다. 명세는 이 경우를 정하지 않았다.
  8. 변동도, 블록 내 상관, SAR 정렬의 셀은 load_base 의 셀 가운데 z(또는 제품 값과 y)가 유한한 셀 전부다(eval_mask 를 적용하지 않는다).
     근접 표는 명세대로 채점 셀에 eval_mask 를 쓴다.
  9. 근접 표의 분할 평균 행(split = −1)에서 frac_within_min, frac_within_max 는 모든 (분할, 추출) 가운데 최솟값과 최댓값이다. 나머지 열은 분할 평균이고
     n_draws 는 분할에 걸친 추출 수의 합이다. --n-grid 에 0 이 있으면 라벨이 없으므로 뺀다.
  10. 잭나이프: 블록이 2개 미만이거나 어느 블록을 빼면 그 구간의 쌍이 0 이 되는 경우 jk_se 는 NaN 이다.
  11. 지수 모형 적합의 fit_flag 는 'ok', 'few_bins(k)', 'no_converge(status)', 'a_at_lower', 'a_at_upper', 'error:…' 를 ';' 로 잇는다.
  12. 대상 지정에서 모드를 생략하면 macro 지역은 x, 하위 지역은 i 다(근접 표의 기본 대상과 같은 모드).
  13. 로컬 실행 규칙에 맞춰 --workers × --threads 상한(8)과 nice 값 10 을 강제한다. 명세에 없는 보호다.
  14. (개정 1, 2026-09-30) LG_RESCALE=1 로 허용된 실행에도 스레드 상한을 적용한다. h44·h45 와 겹쳐 도는 경우의 합계는 실행 중 등록부
      (LC.claim_threads)로 확인한다.

확인하지 못한 것
  개정 1 단계에서 실행한 확인은 py_compile, pyflakes, 스레드 1개의 --count-only, 가벼운 단위 시험(CPU 스레드 2)이다. 스모크는 실행하지 않았다. 알래스카(13,606셀, 층 all 의 쌍 92,554,815개, 네 층 합계 약 2.4억 개. --count-only 값)의 변동도 계산 시간과 최대
  메모리는 재지 않았다. 최대 메모리는 청크 × 셀 수에 비례하며 청크 512 에서 프로세스당 약 0.5–1 GB 로 추정한다(GPR 셀이 몰린 청크는 거의 모든
  쌍이 200 km 안이다). 메모리가 부족하면 --chunk 256 으로 줄인다(결과는 같다).
  지수 모형 적합의 수렴 여부, 표의 값, insar_alt 열의 원자료 판(검토 문서는 ReSALT 로 부른다)은 확인하지 못했다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _permitted(argv=None, environ=None) -> bool:
    """허용 표지(환경 변수 LG_RESCALE=1 또는 --allow-local) 여부. numpy 를 부르기 전에 쓴다."""
    av = list(sys.argv if argv is None else argv)
    env = os.environ if environ is None else environ
    return env.get("LG_RESCALE", "") == "1" or "--allow-local" in av


def _peek_threads(default="4", argv=None, environ=None) -> str:
    """numpy 를 부르기 전에 스레드 수를 정한다(h42 와 같은 규칙). 허용 표지가 없으면 --threads 와 무관하게 1 이다."""
    av = list(sys.argv if argv is None else argv)
    if not _permitted(av, environ):
        return "1"
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return str(av[i + 1])
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return str(default)


_THREADS0 = _peek_threads()
for _v in THREAD_VARS:
    os.environ[_v] = _THREADS0
os.environ["CUDA_VISIBLE_DEVICES"] = ""                       # 이 하네스는 GPU 를 쓰지 않는다
warnings.filterwarnings("ignore", module="threadpoolctl")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
H40_PATH = ROOT / "scripts" / "3_deep_learning" / "h40_label_grid.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_h40():
    """h40 을 파일 경로에서 읽어 sys.modules 에 등록한다(모듈 최상위에서 부른다. 이미 등록되어 있으면 그 모듈을 쓴다)."""
    name = "h40_label_grid"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, H40_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_h40()

import polar.lgu_common as LC                                                                        # noqa: E402
from polar.lgu_common import seed_of, half_split_blocks, eval_mask                                   # noqa: E402

# ================================================================ 고정 설계값(계획서 5절. 바꾸면 사전 등록에서 벗어난다)
ITEMS = ("variogram", "icc", "proximity", "sar")
STRATA = ("all", "gpr", "probe", "noearly")
MIN_STRATUM_CELLS = 30                        # 층의 최소 셀 수(미만이면 행을 만들지 않는다)
FIT_MIN_PAIRS = 30                            # 지수 모형 적합에 넣는 구간의 최소 쌍 수
A_BOUNDS = (0.05, 500.0)                      # 지수 모형 범위 a 의 경계(km)
CH_A, CH_B = 0.457, 0.494                     # Cressie–Hawkins 강건 추정량의 상수
TAU_FLOOR = 1e-4                              # tau_b2 하한(계획서 3.2절)
ICC_MIN_DF = 10                               # 계획서 3.2절의 σ² 자유도 기준(여기서는 flag 로만 적는다)
ICC_MIN_BLOCKS = 5                            # 계획서 3.2절의 τ_b² 블록 수 기준(flag 로만 적는다)
SAR_PRODUCTS = ("insar_alt", "polsar_alt")
SAR_STRATA = ("all", "gpr", "probe", "lt99_5")
SAR_SCALES = ("cm", "log")
SAR_SUB9 = tuple([f"AL-{i}" for i in range(1, 7)] + ["CA-1", "CA-2", "CA-3"])
SAR_UNITS_DEFAULT = ("Alaska", "Canada") + SAR_SUB9
SAR_MIN_CELLS = 30
SAR_MIN_BLOCKS = 3
BETWEEN_MIN_CELLS = 3                         # r_between 에 넣는 블록의 최소 셀 수
WITHIN_MIN_CELLS = 5                          # r_within 에 넣는 블록의 최소 셀 수
CORR_MIN_BLOCKS = 3                           # r_between, r_within 을 계산하는 최소 블록 수(명세 밖 규약, docstring 7)
POLSAR_CODE_MIN = 99.5                        # 분류 부호 의심 하한(cm)
CODE_WINDOW = (95.0, 105.0)                   # 검토 문서 부록 B 의 창
CODE_TOP = 10                                 # 부호 점검 표의 고유값 행 수
SP2_R_LO = 0.2                                # SP2 재심 조건: 블록 내 상관 CI 하한 기준
LOCAL_THREAD_CAP = 8                          # 로컬 실행 규칙(2026-09-30): 이 실험의 CPU 스레드 합계 상한
NICE_TARGET = 10
DEFAULT_BINS = "0,0.1,0.25,0.5,1,2,5,10,25,50,100,200"
DEFAULT_DISTS = "1,2,5,10,25,50,100"
DEFAULT_REGIONS = "Alaska,Lena,Canada"
DEFAULT_TARGETS = [(t, "x") for t in list(H.MAIN4) + [H.ALASKA]] + [(t, "i") for t in list(H.SUB_AL) + list(H.SUB_OTHER)]
FORMAT = "lguc_v1"
BLIND = "재현(비맹검)"

VARIO_COLS = ["region", "stratum", "row", "bin_lo_km", "bin_hi_km", "mean_h_km", "n_pairs", "n_blocks", "frac_same_block", "gamma",
              "gamma_robust", "rho", "jk_se", "var_w", "n_cells", "nugget", "psill", "range_a_km", "practical_range_km", "fit_flag"]
ICC_COLS = ["region", "stratum", "n_cells", "n_blocks", "n_blocks_ge2", "sigma2", "tau_b2", "icc", "icc_lo", "icc_hi", "sd_within",
            "sd_within_lo", "sd_within_hi", "flag", "nboot"]
PROX_COLS = ["target", "mode", "split", "n", "n_lab", "d_km", "frac_within", "frac_within_min", "frac_within_max", "med_nn_km",
             "frac_within_allA", "med_src_km", "n_eval", "n_blocks", "n_draws", "n_splits"]
SAR_COLS = ["product", "unit", "stratum", "scale", "n_cells", "n_blocks", "r_cell", "rho_cell", "r_between", "r_within", "r_within_lo",
            "r_within_hi", "bias_cm", "rmse_cm", "n_ge99_5", "flag", "bias_scale", "rmse_scale", "n_blocks_between", "n_blocks_within",
            "n_units_eval", "n_units_pass", "nboot"]
CODE_COLS = ["product", "unit", "kind", "value", "count", "n_finite", "n_valid", "n_ge99_5_valid", "n_ge99_5_invalid", "frac_ge99_5_valid",
             "n_window_95_105", "n_eq_100", "n_distinct_ge99_5", "frac_integer_ge99_5", "max_value", "y_median_ge99_5", "y_median_lt99_5",
             "prod_median_lt99_5", "std_median_ge99_5", "std_median_lt99_5"]
ITEM_COLS = dict(variogram=VARIO_COLS, icc=ICC_COLS, proximity=PROX_COLS, sar=SAR_COLS)


# ================================================================ 인자
def _floats(txt):
    return [float(v) for v in str(txt).split(",") if v.strip()]


def _n_list(txt):
    return [-1 if v.strip() == "all" else int(v) for v in str(txt).split(",") if v.strip()]


def _names(txt):
    return [v.strip() for v in str(txt).split(",") if v.strip()]


def _target_pairs(txt):
    """'이름' 또는 '이름:모드'. 모드를 생략하면 macro 지역은 x, 하위 지역은 i 다(docstring 12)."""
    out = []
    for tok in _names(txt):
        if ":" in tok:
            t, m = tok.split(":", 1)
            out.append((t.strip(), m.strip()))
        else:
            out.append((tok, "i" if "i" in H.valid_modes(tok) else "x"))
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H46 공간 정보량 진단(LGU 실험 C)")
    ap.add_argument("--items", default=",".join(ITEMS), help="variogram,icc,proximity,sar 가운데 쉼표 목록")
    ap.add_argument("--regions", default=DEFAULT_REGIONS, help="변동도·블록 내 상관의 macro 지역")
    ap.add_argument("--targets", default="", help="근접 표의 대상('이름' 또는 '이름:모드'). 기본: 주 5개 x 와 하위 지역 10개 i")
    ap.add_argument("--sar-units", default=",".join(SAR_UNITS_DEFAULT), help="SAR 정렬 표의 단위(macro 지역 또는 sub 열의 하위 지역)")
    ap.add_argument("--splits", type=int, default=5, help="half_split_blocks split_seed 1..K")
    ap.add_argument("--n-grid", default="10,40,160,all")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1, help="학습이 없으므로 쓰지 않는다(기록만 한다)")
    ap.add_argument("--dists", default=DEFAULT_DISTS, help="근접 표의 거리 기준(km)")
    ap.add_argument("--bins", default=DEFAULT_BINS, help="변동도 거리 구간 경계(km)")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=512, help="변동도 거리 행렬의 행 묶음 크기(결과는 바뀌지 않는다)")
    ap.add_argument("--workers", type=int, default=2, help="하위 단위를 나눠 도는 프로세스 수. 0 또는 1 이면 주 프로세스에서 차례로")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpus", default="", help="받기만 하고 쓰지 않는다(이 하네스는 GPU 를 쓰지 않는다)")
    ap.add_argument("--out-dir", default="data/processed/lgu")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default="lguc")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--label-flags", default="lgx_label_flags_v1.csv")
    ap.add_argument("--allow-local", action="store_true",
                    help="로컬 실행을 허용한다(LG_RESCALE=1 도 허용 표지로 인정하지만 로컬 자원 규칙은 그대로 적용된다)")
    ap.add_argument("--smoke", action="store_true", help="지역 Canada, 대상 Canada x, 분할 1, 추출 2, SAR 단위 Canada·CA-1, nboot 200")
    ap.add_argument("--precheck", action="store_true", help="시간 측정: 지역 Alaska, 대상 Alaska x, 분할 1, SAR 단위 Alaska, nboot 2000")
    ap.add_argument("--count-only", action="store_true", help="단위 수, 쌍 수의 상한, 근접 표의 행 수만 출력한다")
    ap.add_argument("--summarize-only", action="store_true", help="계산하지 않고 조각에서 표만 만든다")
    ap.add_argument("--no-summarize", action="store_true", help="조각만 쓰고 표를 만들지 않는다")
    ap.add_argument("--resume", action="store_true", help="설정 해시가 같고 실패가 아닌 하위 단위 조각은 건너뛴다")
    ap.add_argument("--buffer-km", type=float, default=100.0, help="고정 설계값(h40 의 원천 버퍼). 바꾸면 사전 등록에서 벗어난다")
    return finalize(ap.parse_args(argv))


def finalize(a):
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    if a.summarize_only and a.no_summarize:
        raise SystemExit("--summarize-only 와 --no-summarize 는 함께 쓰지 않는다")
    a.ITEMS = _names(a.items)
    bad = [v for v in a.ITEMS if v not in ITEMS]
    if bad:
        raise SystemExit(f"알 수 없는 항목: {bad}")
    a.REGIONS = _names(a.regions)
    a.TARGETS = _target_pairs(a.targets) if a.targets else list(DEFAULT_TARGETS)
    a.SAR_UNITS = _names(a.sar_units)
    a.SPLITS = list(range(1, int(a.splits) + 1))
    a.N_GRID = [n for n in _n_list(a.n_grid) if n != 0]
    a.DRAWS = int(a.draws)
    a.DISTS = sorted(_floats(a.dists))
    a.EDGES = np.asarray(_floats(a.bins), float)
    if len(a.EDGES) < 2 or np.any(np.diff(a.EDGES) <= 0) or a.EDGES[0] < 0:
        raise SystemExit(f"--bins 는 0 이상에서 시작하는 증가 수열이어야 한다: {a.bins}")
    a.NBOOT = int(a.nboot)
    if a.smoke:
        a.SPLITS = [1]; a.DRAWS = min(a.DRAWS, 2); a.NBOOT = min(a.NBOOT, 200)
        if a.regions == DEFAULT_REGIONS:
            a.REGIONS = ["Canada"]
        if not a.targets:
            a.TARGETS = [("Canada", "x")]
        if a.sar_units == ",".join(SAR_UNITS_DEFAULT):
            a.SAR_UNITS = ["Canada", "CA-1"]
    if a.precheck:
        a.SPLITS = [1]
        if a.regions == DEFAULT_REGIONS:
            a.REGIONS = ["Alaska"]
        if not a.targets:
            a.TARGETS = [("Alaska", "x")]
        if a.sar_units == ",".join(SAR_UNITS_DEFAULT):
            a.SAR_UNITS = ["Alaska"]
    for t, m in a.TARGETS:
        if m not in ("i", "x"):
            raise SystemExit(f"대상 {t} 의 모드 {m} 는 i 또는 x 여야 한다")
    a.SUFFIX = ("_smoke" if a.smoke else "") + ("_precheck" if a.precheck else "")
    a.TAG = a.tag + a.SUFFIX
    a.PREFIX = ("lgu_c" if a.tag == "lguc" else a.tag) + a.SUFFIX
    a.OUT = (ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir)
    a.PROC = (ROOT / a.data_dir) if not os.path.isabs(a.data_dir) else Path(a.data_dir)
    a.SHARDS = a.OUT / "shards"
    a.SUBMAP = Path(a.subregion_map) if os.path.isabs(a.subregion_map) else a.PROC / a.subregion_map
    a.FLAGS = Path(a.label_flags) if os.path.isabs(a.label_flags) else a.PROC / a.label_flags
    a.GPUS = [g.strip() for g in str(a.gpus).split(",") if g.strip()]
    return a


def h40_argv(a):
    """h40.Data 를 만들 인자(자료, 하위 지역 대응표, 원천 버퍼, 분할 범위만 쓴다)."""
    return ["--part", "cpu", "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", str(a.SUBMAP), "--modes", "i,x",
            "--splits", str(max(a.SPLITS)), "--n-grid", str(a.n_grid), "--draws", str(a.DRAWS), "--seeds", "1",
            "--threads", str(a.threads), "--kappa", "10", "--buffer-km", str(a.buffer_km)]


# ================================================================ 자료(프로세스마다 한 번)
_DATA: dict = {}
_FLAGS: dict = {}
_HASHES: dict = {}


def get_data(a):
    sig = (str(a.PROC), str(a.SUBMAP), float(a.buffer_km), tuple(a.SPLITS))
    if sig not in _DATA:
        D = H.Data(H.parse_args(h40_argv(a)))
        _DATA.clear(); _DATA[sig] = D
    return _DATA[sig]


def get_flags(a, D, missing_ok=False):
    k = (id(D.df), str(a.FLAGS), bool(missing_ok))
    if k not in _FLAGS:
        _FLAGS[k] = LC.load_flags(a.FLAGS, D.df.loc_id.values, missing_ok=missing_ok)
    return _FLAGS[k]


def input_hashes(a) -> dict:
    k = (str(a.PROC), str(a.SUBMAP), str(a.FLAGS))
    if k not in _HASHES:
        _HASHES[k] = {"fidelity_base_v3.csv": LC.file_sha(a.PROC / "fidelity_base_v3.csv"),
                      "e5_soil_tdd_v3.csv": LC.file_sha(a.PROC / "e5_soil_tdd_v3.csv"),
                      Path(a.SUBMAP).name: LC.file_sha(a.SUBMAP), Path(a.FLAGS).name: LC.file_sha(a.FLAGS)}
    return dict(_HASHES[k])


def code_shas() -> dict:
    return dict(code_sha=LC.file_sha(Path(__file__)), code_sha_common=LC.file_sha(Path(LC.__file__)), code_sha_h40=LC.file_sha(H40_PATH))


def stratum_mask(FL, idx, stratum):
    """층 표지(셀 표지 표 기준). all 은 전부, noearly 는 early 가 아닌 셀, lt99_5 는 호출자가 따로 처리한다."""
    idx = np.asarray(idx, int)
    if stratum == "all":
        return np.ones(len(idx), bool)
    if stratum in ("gpr", "probe"):
        return np.asarray(FL[stratum], bool)[idx]
    if stratum == "noearly":
        return ~np.asarray(FL["early"], bool)[idx]
    raise ValueError(f"알 수 없는 층 {stratum}")


def region_cells(D, region):
    """macro 지역의 셀 가운데 z 가 유한한 셀의 색인과 w = z − 지역 평균 z. 반환 (idx, w, n_nonfinite)."""
    df = D.df
    idx = np.where(df.macro.values == region)[0]
    if len(idx) == 0:
        raise ValueError(f"지역 {region} 의 셀이 없다")
    z = df.z.values.astype(float)[idx]
    ok = np.isfinite(z)
    idx, z = idx[ok], z[ok]
    return idx, z - z.mean(), int((~ok).sum())


# ================================================================ 거리
def _hav_km(la1, lo1, la2, lo2):
    """대권 거리(km). 입력은 라디안이고 브로드캐스트한다(m1_ext.haversine_km 과 같은 식, 반지름 LC.EARTH_R_KM)."""
    s1 = np.sin((la2 - la1) * 0.5)
    s2 = np.sin((lo2 - lo1) * 0.5)
    h = s1 * s1 + np.cos(la1) * np.cos(la2) * s2 * s2
    return 2.0 * LC.EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def pair_dist_f32(lat, lon, i, j):
    """쌍 (i, j)의 거리(float32 로 바꾼 뒤 float64). pair_accumulate 와 같은 반올림 규칙이다(시험용)."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return _hav_km(la[i], lo[i], la[j], lo[j]).astype(np.float32).astype(float)


# ================================================================ LGU-C1 변동도
def pair_accumulate(lat, lon, w, blk, edges, chunk=512) -> dict:
    """j > i 인 모든 쌍의 거리 구간별 누적. 거리 행렬은 행 묶음(chunk × 나머지 셀, float32)으로 만든다.

    반환 dict: npair(nb,) int64, s2(nb,) = Σ 0.5Δ², sr(nb,) = Σ |Δ|^0.5, sh(nb,) = Σ 거리, nsame(nb,) = 같은 블록 쌍 수,
    cell_s2·cell_n (N, nb) = 셀이 끼인 쌍의 합과 수, blk_s2·blk_n (B, nb) = 블록 안 쌍의 합과 수, tot_s2·tot_n (B, nb) = 블록 셀의 셀별 누적 합,
    codes(N,) 블록 번호, blocks(B,) 블록 식별자, n_total = N(N − 1)/2, n_within = 구간 안 쌍 수.
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float); w = np.asarray(w, float)
    edges = np.asarray(edges, float); nb = len(edges) - 1
    blocks, codes = np.unique(np.asarray(blk), return_inverse=True)
    B = len(blocks); N = len(w)
    la, lo = np.radians(lat), np.radians(lon)
    npair = np.zeros(nb, np.int64); nsame = np.zeros(nb, np.int64)
    s2 = np.zeros(nb); sr = np.zeros(nb); sh = np.zeros(nb)
    cs2 = np.zeros(N * nb); cn = np.zeros(N * nb, np.int64)
    bs2 = np.zeros(B * nb); bn = np.zeros(B * nb, np.int64)
    dmin, dmax = float(edges[0]), float(edges[-1])
    chunk = max(1, int(chunk))
    for i0 in range(0, max(N - 1, 0), chunk):
        i1 = min(N - 1, i0 + chunk)
        j0 = i0 + 1
        d = _hav_km(la[i0:i1, None], lo[i0:i1, None], la[None, j0:], lo[None, j0:]).astype(np.float32)
        # 행 r(= i − i0), 열 c(= j − j0). j > i ⇔ c ≥ r
        keep = np.arange(d.shape[1])[None, :] >= np.arange(d.shape[0])[:, None]
        keep &= (d >= np.float32(dmin)) & (d < np.float32(dmax))
        r, c = np.nonzero(keep)
        del keep
        if len(r) == 0:
            continue
        gi = r + i0; gj = c + j0
        dd = d[r, c].astype(float)
        del d, r, c
        b = np.clip(np.searchsorted(edges, dd, side="right") - 1, 0, nb - 1)
        npair += np.bincount(b, minlength=nb)
        sh += np.bincount(b, weights=dd, minlength=nb)
        del dd                                                              # 중간 배열을 바로 지워 최대 메모리를 줄인다
        dw = w[gi] - w[gj]
        sr += np.bincount(b, weights=np.sqrt(np.abs(dw)), minlength=nb)
        h2 = 0.5 * dw * dw
        del dw
        s2 += np.bincount(b, weights=h2, minlength=nb)
        same = codes[gi] == codes[gj]
        ki = gi * nb + b
        cs2 += np.bincount(ki, weights=h2, minlength=N * nb)
        cn += np.bincount(ki, minlength=N * nb)
        del ki
        kj = gj * nb + b
        cs2 += np.bincount(kj, weights=h2, minlength=N * nb)
        cn += np.bincount(kj, minlength=N * nb)
        del kj
        if same.any():
            kb = codes[gi[same]] * nb + b[same]
            nsame += np.bincount(b[same], minlength=nb)
            bs2 += np.bincount(kb, weights=h2[same], minlength=B * nb)
            bn += np.bincount(kb, minlength=B * nb)
    cs2 = cs2.reshape(N, nb); cn = cn.reshape(N, nb)
    tot_s2 = np.zeros((B, nb)); tot_n = np.zeros((B, nb), np.int64)
    np.add.at(tot_s2, codes, cs2); np.add.at(tot_n, codes, cn)
    return dict(npair=npair, s2=s2, sr=sr, sh=sh, nsame=nsame, cell_s2=cs2, cell_n=cn, blk_s2=bs2.reshape(B, nb), blk_n=bn.reshape(B, nb),
                tot_s2=tot_s2, tot_n=tot_n, codes=codes, blocks=blocks, n_total=int(N * (N - 1) // 2), n_within=int(npair.sum()), edges=edges)


def jackknife_gammas(acc) -> np.ndarray:
    """블록 b 를 뺀 gamma_(−b) (B, nb). S_b = (블록 b 셀의 셀별 누적 합) − (블록 b 안 쌍의 합). 쌍이 남지 않으면 NaN."""
    Sb = acc["tot_s2"] - acc["blk_s2"]
    Nb = acc["tot_n"] - acc["blk_n"]
    num = acc["s2"][None, :] - Sb
    den = (acc["npair"][None, :] - Nb).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def jackknife_se(G) -> np.ndarray:
    """잭나이프 표준오차 sqrt((B − 1)/B · Σ_b (g_b − 평균)²). 블록이 2개 미만이거나 비유한 g_b 가 있는 구간은 NaN."""
    G = np.asarray(G, float)
    B = G.shape[0]
    out = np.full(G.shape[1], np.nan)
    if B < 2:
        return out
    ok = np.all(np.isfinite(G), axis=0)
    if ok.any():
        Gm = G[:, ok]
        out[ok] = np.sqrt((B - 1) / B * ((Gm - Gm.mean(0)) ** 2).sum(0))
    return out


def fit_exponential(h, g, npair, var_w, min_pairs=FIT_MIN_PAIRS) -> dict:
    """지수 모형 γ(h) = c0 + c1(1 − exp(−h/a))의 쌍 수 가중 최소제곱. 반환 dict(nugget, psill, range_a_km, practical_range_km, fit_flag, n_bins, n_pairs)."""
    from scipy.optimize import least_squares
    h = np.asarray(h, float); g = np.asarray(g, float); npair = np.asarray(npair, float)
    m = (npair >= min_pairs) & np.isfinite(g) & np.isfinite(h) & (h > 0)
    out = dict(nugget=np.nan, psill=np.nan, range_a_km=np.nan, practical_range_km=np.nan, n_bins=int(m.sum()), n_pairs=int(npair[m].sum()))
    if m.sum() < 3:
        out["fit_flag"] = f"few_bins({int(m.sum())})"
        return out
    hh, gg, ww = h[m], g[m], np.sqrt(npair[m])
    c0 = max(float(gg[0]), 0.0)
    c1 = float(var_w) - c0 if np.isfinite(var_w) else float(gg.max()) - c0
    c1 = max(c1, 1e-6)
    lb = np.array([0.0, 0.0, A_BOUNDS[0]]); ub = np.array([np.inf, np.inf, A_BOUNDS[1]])
    x0 = np.array([c0 + 1e-9, c1, 2.0])

    def resid(p):
        return ww * (p[0] + p[1] * (1.0 - np.exp(-hh / p[2])) - gg)

    try:
        res = least_squares(resid, x0, bounds=(lb, ub), method="trf")
    except Exception as e:                                                   # noqa: BLE001
        out["fit_flag"] = f"error:{type(e).__name__}"
        return out
    flags = []
    if not res.success:
        flags.append(f"no_converge({res.status})")
    a_hat = float(res.x[2])
    if a_hat <= A_BOUNDS[0] * (1 + 1e-6):
        flags.append("a_at_lower")
    if a_hat >= A_BOUNDS[1] * (1 - 1e-6):
        flags.append("a_at_upper")
    out.update(nugget=float(res.x[0]), psill=float(res.x[1]), range_a_km=a_hat, practical_range_km=3.0 * a_hat,
               fit_flag=";".join(flags) if flags else "ok")
    return out


def variogram_rows(region, stratum, lat, lon, w, blk, edges, chunk=512):
    """한 지역·층의 변동도 행(구간 행과 적합 행). 반환 (rows, info)."""
    t0 = time.time()
    w = np.asarray(w, float)
    edges = np.asarray(edges, float)
    acc = pair_accumulate(lat, lon, w, blk, edges, chunk)
    n = len(w)
    var_w = float(np.var(w, ddof=1)) if n >= 2 else np.nan
    npair = acc["npair"]
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = acc["s2"] / npair
        mean_r = acc["sr"] / npair
        gamma_rb = mean_r ** 4 / (2.0 * (CH_A + CH_B / npair))
        mean_h = acc["sh"] / npair
        frac_same = acc["nsame"] / npair
        rho = 1.0 - gamma / var_w
    nblk_bin = (acc["tot_n"] > 0).sum(0)
    jk = jackknife_se(jackknife_gammas(acc))
    rows = []
    for k in range(len(edges) - 1):
        rows.append(dict(region=region, stratum=stratum, row="bin", bin_lo_km=float(edges[k]), bin_hi_km=float(edges[k + 1]),
                         mean_h_km=float(mean_h[k]), n_pairs=int(npair[k]), n_blocks=int(nblk_bin[k]), frac_same_block=float(frac_same[k]),
                         gamma=float(gamma[k]), gamma_robust=float(gamma_rb[k]), rho=float(rho[k]), jk_se=float(jk[k]), var_w=var_w,
                         n_cells=int(n), nugget=np.nan, psill=np.nan, range_a_km=np.nan, practical_range_km=np.nan, fit_flag=""))
    fit = fit_exponential(mean_h, gamma, npair, var_w)
    rows.append(dict(region=region, stratum=stratum, row="fit", bin_lo_km=np.nan, bin_hi_km=np.nan, mean_h_km=np.nan, n_pairs=fit["n_pairs"],
                     n_blocks=int(len(acc["blocks"])), frac_same_block=np.nan, gamma=np.nan, gamma_robust=np.nan, rho=np.nan, jk_se=np.nan,
                     var_w=var_w, n_cells=int(n), nugget=fit["nugget"], psill=fit["psill"], range_a_km=fit["range_a_km"],
                     practical_range_km=fit["practical_range_km"], fit_flag=fit["fit_flag"]))
    info = dict(n_cells=int(n), n_blocks=int(len(acc["blocks"])), var_w=var_w, n_pairs_total=acc["n_total"], n_pairs_within=acc["n_within"],
                n_pairs_beyond=int(acc["n_total"] - acc["n_within"]), fit_bins=fit["n_bins"], fit_flag=fit["fit_flag"],
                sec=round(time.time() - t0, 2))
    return rows, info


# ================================================================ LGU-C2 블록 내 상관
def block_moments(w, blk):
    """블록별 셀 수, 평균, 블록 내 제곱합. 반환 (blocks, n(B,), mean(B,), ssw(B,))."""
    w = np.asarray(w, float)
    blocks, codes = np.unique(np.asarray(blk), return_inverse=True)
    n = np.bincount(codes, minlength=len(blocks)).astype(float)
    mean = np.bincount(codes, weights=w, minlength=len(blocks)) / n
    dev = w - mean[codes]
    ssw = np.bincount(codes, weights=dev * dev, minlength=len(blocks))
    return blocks, n, mean, ssw


def icc_stats(n, mean, ssw, M=None) -> dict:
    """적률 추정(계획서 3.2절과 같은 식). M(R, B)은 블록 다중도(None 이면 1 행의 1). 반환 dict(sigma2, tau_b2, icc, sd_within), 값은 (R,)."""
    n = np.asarray(n, float); mean = np.asarray(mean, float); ssw = np.asarray(ssw, float)
    M = np.ones((1, len(n))) if M is None else np.asarray(M, float)
    mc = mean - mean.mean() if len(mean) else mean                       # 상쇄 오차를 줄이기 위한 중심화(분산은 바뀌지 않는다)
    dfw = M @ (n - 1.0)
    SS = M @ ssw
    mt = M.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma2 = np.where(dfw > 0, SS / np.where(dfw > 0, dfw, 1.0), np.nan)
        mu = (M @ mc) / mt
        var_b = np.where(mt > 1, (M @ (mc * mc) - mt * mu * mu) / np.where(mt > 1, mt - 1.0, 1.0), np.nan)
        inv_n = (M @ (1.0 / n)) / mt
        tau = np.maximum(var_b - sigma2 * inv_n, TAU_FLOOR)
        tau = np.where(np.isfinite(var_b) & np.isfinite(sigma2), tau, np.nan)
        icc = tau / (tau + sigma2)
    return dict(sigma2=sigma2, tau_b2=tau, icc=icc, sd_within=np.sqrt(sigma2))


def block_boot_mult(nb, nboot, seed) -> np.ndarray:
    """블록 복원 추출의 다중도 (nboot, nb) float64. RandomState(seed).randint(0, nb, (nboot, nb))."""
    if nb == 0 or nboot <= 0:
        return np.zeros((max(nboot, 0), nb))
    pick = np.random.RandomState(int(seed)).randint(0, nb, size=(nboot, nb))
    rows = np.repeat(np.arange(nboot), nb)
    return np.bincount(rows * nb + pick.ravel(), minlength=nboot * nb).reshape(nboot, nb).astype(float)


def _pct(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).any():
        return np.nan, np.nan
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def icc_row(region, stratum, w, blk, nboot, seed) -> dict:
    """한 지역·층의 블록 내 상관 행. CI 는 블록 재표집(다중도 행렬, seed 고정)의 백분위 95 %."""
    blocks, n, mean, ssw = block_moments(w, blk)
    pt = icc_stats(n, mean, ssw)
    flags = []
    if float((n - 1).sum()) < ICC_MIN_DF:
        flags.append(f"df<{ICC_MIN_DF}")
    if len(blocks) < ICC_MIN_BLOCKS:
        flags.append(f"blocks<{ICC_MIN_BLOCKS}")
    if np.isfinite(pt["tau_b2"][0]) and pt["tau_b2"][0] <= TAU_FLOOR:
        flags.append("tau_floor")
    lo = dict(icc=(np.nan, np.nan), sd_within=(np.nan, np.nan))
    if nboot > 0 and len(blocks) >= 2:
        bs = icc_stats(n, mean, ssw, block_boot_mult(len(blocks), nboot, seed))
        lo = dict(icc=_pct(bs["icc"]), sd_within=_pct(bs["sd_within"]))
        nn = int((~np.isfinite(bs["icc"])).sum())
        if nn:
            flags.append(f"boot_nan={nn}")
    return dict(region=region, stratum=stratum, n_cells=int(n.sum()), n_blocks=int(len(blocks)), n_blocks_ge2=int((n >= 2).sum()),
                sigma2=float(pt["sigma2"][0]), tau_b2=float(pt["tau_b2"][0]), icc=float(pt["icc"][0]), icc_lo=lo["icc"][0], icc_hi=lo["icc"][1],
                sd_within=float(pt["sd_within"][0]), sd_within_lo=lo["sd_within"][0], sd_within_hi=lo["sd_within"][1],
                flag=";".join(flags), nboot=int(nboot))


# ================================================================ LGU-C3 근접 라벨 비율
def proximity_split(target, mode, split, lat, lon, A_idx, evB, src_idx, blk, n_grid, n_draws, dists, return_sel=False):
    """한 (대상, 모드, 분할)의 근접 표 행. 선택 라벨은 h40.draw_cells(대상, 모드, 분할, n, 추출, |A|)의 A_idx 안 색인이다.

    좌표와 추출 색인만 쓴다(라벨 값과 예측값을 쓰지 않는다). 반환 rows(return_sel 이면 (rows, {(n, 추출): sel})).
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    A_idx = np.asarray(A_idx, int); evB = np.asarray(evB, int); src_idx = np.asarray(src_idx, int)
    nA = len(A_idx)
    la_e, lo_e = lat[evB], lon[evB]
    dA = LC.nearest_km(la_e, lo_e, lat[A_idx], lon[A_idx])
    dS = LC.nearest_km(la_e, lo_e, lat[src_idx], lon[src_idx]) if len(src_idx) else np.full(len(evB), np.nan)
    med_src = float(np.nanmedian(dS)) if np.isfinite(dS).any() else np.nan
    nbe = int(len(np.unique(np.asarray(blk)[evB]))) if len(evB) else 0
    rows, sels = [], {}
    for n in [v for v in n_grid if v != 0]:
        cells = H.cells_of([n], n_draws, nA)
        if not cells:
            continue
        fr, med, nlab = [], [], []
        for nn, d in cells:
            sel = H.draw_cells(target, mode, split, nn, d, nA)
            lab = A_idx[sel]
            dist = LC.nearest_km(la_e, lo_e, lat[lab], lon[lab])
            fr.append([float(np.mean(dist <= dk)) for dk in dists])
            med.append(float(np.nanmedian(dist)) if np.isfinite(dist).any() else np.nan)
            nlab.append(len(sel))
            if return_sel:
                sels[(int(nn), int(d))] = sel
        fr = np.asarray(fr, float)
        for k, dk in enumerate(dists):
            rows.append(dict(target=target, mode=mode, split=int(split), n=int(n), n_lab=float(np.mean(nlab)), d_km=float(dk),
                             frac_within=float(fr[:, k].mean()), frac_within_min=float(fr[:, k].min()), frac_within_max=float(fr[:, k].max()),
                             med_nn_km=float(np.nanmean(med)) if np.isfinite(med).any() else np.nan,
                             frac_within_allA=float(np.mean(dA <= dk)), med_src_km=med_src, n_eval=int(len(evB)), n_blocks=nbe,
                             n_draws=int(len(cells)), n_splits=1))
    return (rows, sels) if return_sel else rows


def proximity_summary(rows) -> list:
    """분할 평균 행(split = −1). frac_within_min·max 는 모든 (분할, 추출)의 최솟값·최댓값, n_draws 는 합, 나머지는 분할 평균이다."""
    if not rows:
        return []
    d = pd.DataFrame(rows)
    d = d[d.split >= 0]
    out = []
    for (t, m, n, dk), g in d.groupby(["target", "mode", "n", "d_km"], sort=True):
        out.append(dict(target=t, mode=m, split=-1, n=int(n), n_lab=float(g.n_lab.mean()), d_km=float(dk), frac_within=float(g.frac_within.mean()),
                        frac_within_min=float(g.frac_within_min.min()), frac_within_max=float(g.frac_within_max.max()),
                        med_nn_km=float(g.med_nn_km.mean()), frac_within_allA=float(g.frac_within_allA.mean()), med_src_km=float(g.med_src_km.mean()),
                        n_eval=float(g.n_eval.mean()), n_blocks=float(g.n_blocks.mean()), n_draws=int(g.n_draws.sum()), n_splits=int(len(g))))
    return out


def usable_splits(D, target, splits):
    """h40.enumerate_units 와 같은 분할 규칙. 반환 (쓸 분할 목록, 건너뛴 분할 기록)."""
    info = D.split_structure(target)
    any_valid = any(v["valid"] for v in info.values())
    use, skipped = [], []
    for sp in splits:
        v = info[sp]
        if v["dup_of"] >= 0:
            skipped.append(dict(split=sp, status=f"dup_of_{v['dup_of']}"))
        elif v["n_eval"] == 0 or v["n_A"] == 0:
            skipped.append(dict(split=sp, status="no_eval"))
        elif not v["valid"] and any_valid:
            skipped.append(dict(split=sp, status="invalid_nb_eval<2"))
        else:
            use.append(sp)
    return use, skipped


# ================================================================ LGU-C4 SAR 정렬
def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    xc, yc = x - x.mean(), y - y.mean()
    den = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    return float((xc * yc).sum() / den) if den > 0 else np.nan


def _spearman(x, y):
    from scipy.stats import rankdata
    if len(x) < 2:
        return np.nan
    return _pearson(rankdata(x), rankdata(y))


def within_sums(x, y, blk, min_cells=WITHIN_MIN_CELLS):
    """셀 min_cells 개 이상 블록에서 블록 평균을 뺀 값의 블록별 곱합. 반환 (Sxy, Sxx, Syy) 각각 (B',)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    blocks, codes = np.unique(np.asarray(blk), return_inverse=True)
    cnt = np.bincount(codes, minlength=len(blocks))
    keep = cnt >= min_cells
    if not keep.any():
        return np.zeros(0), np.zeros(0), np.zeros(0)
    mx = np.bincount(codes, weights=x, minlength=len(blocks)) / np.maximum(cnt, 1)
    my = np.bincount(codes, weights=y, minlength=len(blocks)) / np.maximum(cnt, 1)
    dx, dy = x - mx[codes], y - my[codes]
    Sxy = np.bincount(codes, weights=dx * dy, minlength=len(blocks))[keep]
    Sxx = np.bincount(codes, weights=dx * dx, minlength=len(blocks))[keep]
    Syy = np.bincount(codes, weights=dy * dy, minlength=len(blocks))[keep]
    return Sxy, Sxx, Syy


def r_within_from_sums(Sxy, Sxx, Syy, M=None):
    """블록 평균을 뺀 값의 합동 Pearson = Σ Sxy / sqrt(Σ Sxx · Σ Syy)(블록마다 평균이 0 이므로 정확하다). M(R, B')은 다중도."""
    M = np.ones((1, len(Sxy))) if M is None else np.asarray(M, float)
    num = M @ Sxy; den = np.sqrt((M @ Sxx) * (M @ Syy))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def sar_row_stats(prod_cm, y_cm, blk, scale, nboot, seed) -> dict:
    """한 제품·단위·층·척도의 정렬 통계. prod_cm, y_cm 은 cm 값(그 단위·층의 제품 유효 셀), blk 는 블록 식별자."""
    prod_cm = np.asarray(prod_cm, float); y_cm = np.asarray(y_cm, float); blk = np.asarray(blk)
    if scale == "log":
        ok = np.isfinite(prod_cm) & np.isfinite(y_cm) & (prod_cm > 0) & (y_cm > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            x, yy = np.log(np.where(ok, prod_cm, 1.0)), np.log(np.where(ok, y_cm, 1.0))
    elif scale == "cm":
        ok = np.isfinite(prod_cm) & np.isfinite(y_cm)
        x, yy = prod_cm, y_cm
    else:
        raise ValueError(f"알 수 없는 척도 {scale}")
    x, yy, b, pc, yc = x[ok], yy[ok], blk[ok], prod_cm[ok], y_cm[ok]
    n = int(len(x)); nb = int(len(np.unique(b)))
    out = dict(n_cells=n, n_blocks=nb, r_cell=np.nan, rho_cell=np.nan, r_between=np.nan, r_within=np.nan, r_within_lo=np.nan,
               r_within_hi=np.nan, bias_cm=np.nan, rmse_cm=np.nan, bias_scale=np.nan, rmse_scale=np.nan, n_blocks_between=0,
               n_blocks_within=0, flag="", nboot=int(nboot))
    flags = []
    if n < SAR_MIN_CELLS:
        flags.append(f"n<{SAR_MIN_CELLS}")
    if nb < SAR_MIN_BLOCKS:
        flags.append(f"blocks<{SAR_MIN_BLOCKS}")
    if flags:
        out["flag"] = ";".join(flags)
        return out
    out.update(r_cell=_pearson(x, yy), rho_cell=_spearman(x, yy), bias_cm=float(np.mean(pc - yc)), rmse_cm=float(np.sqrt(np.mean((pc - yc) ** 2))),
               bias_scale=float(np.mean(x - yy)), rmse_scale=float(np.sqrt(np.mean((x - yy) ** 2))))
    blocks, codes = np.unique(b, return_inverse=True)
    cnt = np.bincount(codes, minlength=len(blocks))
    kb = cnt >= BETWEEN_MIN_CELLS
    out["n_blocks_between"] = int(kb.sum())
    if kb.sum() >= CORR_MIN_BLOCKS:
        mx = np.bincount(codes, weights=x, minlength=len(blocks)) / cnt
        my = np.bincount(codes, weights=yy, minlength=len(blocks)) / cnt
        out["r_between"] = _pearson(mx[kb], my[kb])
    else:
        flags.append(f"between_blocks<{CORR_MIN_BLOCKS}")
    Sxy, Sxx, Syy = within_sums(x, yy, b)
    out["n_blocks_within"] = int(len(Sxy))
    if len(Sxy) >= CORR_MIN_BLOCKS:
        out["r_within"] = float(r_within_from_sums(Sxy, Sxx, Syy)[0])
        if nboot > 0:
            rb = r_within_from_sums(Sxy, Sxx, Syy, block_boot_mult(len(Sxy), nboot, seed))
            out["r_within_lo"], out["r_within_hi"] = _pct(rb)
            nn = int((~np.isfinite(rb)).sum())
            if nn:
                flags.append(f"boot_nan={nn}")
    else:
        flags.append(f"within_blocks<{CORR_MIN_BLOCKS}")
    out["flag"] = ";".join(flags)
    return out


def sar_valid(df, product) -> np.ndarray:
    """제품 유효 셀: insar_alt 는 유한한 셀, polsar_alt 는 polsar_valid > 0 이고 유한한 셀."""
    v = df[product].values.astype(float)
    ok = np.isfinite(v)
    if product == "polsar_alt":
        pv = df["polsar_valid"].values.astype(float) if "polsar_valid" in df else np.zeros(len(df))
        ok &= np.nan_to_num(pv, nan=0.0) > 0
    return ok


def sar_code_rows(product, unit, v, valid, y, std=None, top=CODE_TOP) -> list:
    """분류 부호 점검 행(요약 1행과 99.5 이상 고유값 상위 top 행). v 는 단위 셀의 제품 값, valid 는 제품 유효 표지."""
    v = np.asarray(v, float); valid = np.asarray(valid, bool); y = np.asarray(y, float)
    std = np.full(len(v), np.nan) if std is None else np.asarray(std, float)
    fin = np.isfinite(v)
    ge = fin & (v >= POLSAR_CODE_MIN)
    gev, lt = ge & valid, fin & valid & (v < POLSAR_CODE_MIN)
    vals = v[gev]
    uq, cnt = (np.unique(vals, return_counts=True) if len(vals) else (np.zeros(0), np.zeros(0, int)))

    def med(x):
        x = np.asarray(x, float)
        return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan

    base = dict(product=product, unit=unit, n_finite=int(fin.sum()), n_valid=int((fin & valid).sum()), n_ge99_5_valid=int(gev.sum()),
                n_ge99_5_invalid=int((ge & ~valid).sum()),
                frac_ge99_5_valid=float(gev.sum() / max((fin & valid).sum(), 1)) if (fin & valid).any() else np.nan,
                n_window_95_105=int((fin & valid & (v >= CODE_WINDOW[0]) & (v <= CODE_WINDOW[1])).sum()),
                n_eq_100=int((fin & valid & (np.abs(v - 100.0) < 1e-6)).sum()), n_distinct_ge99_5=int(len(uq)),
                frac_integer_ge99_5=float(np.mean(np.abs(vals - np.round(vals)) < 1e-6)) if len(vals) else np.nan,
                max_value=float(np.nanmax(v[fin & valid])) if (fin & valid).any() else np.nan,
                y_median_ge99_5=med(y[gev]), y_median_lt99_5=med(y[lt]), prod_median_lt99_5=med(v[lt]),
                std_median_ge99_5=med(std[gev]), std_median_lt99_5=med(std[lt]))
    rows = [dict(base, kind="summary", value=np.nan, count=int(gev.sum()))]
    order = np.lexsort((uq, -cnt))[:top] if len(uq) else []
    for j in order:
        rows.append(dict(base, kind="value", value=float(uq[j]), count=int(cnt[j])))
    return rows


def sar_summary_rows(tab, units=SAR_SUB9) -> list:
    """요약 행(unit = 'SUMMARY9'): 하위 지역 가운데 r_within_lo > 0.2 인 수. 판정 문장은 두지 않는다."""
    out = []
    if tab is None or len(tab) == 0:
        return out
    sub = tab[tab.unit.isin(list(units))]
    for (p, st, sc), g in sub.groupby(["product", "stratum", "scale"], sort=False):
        lo = g.r_within_lo.values.astype(float)
        ok = np.isfinite(lo)
        passed = [u for u, v in zip(g.unit.values, lo) if np.isfinite(v) and v > SP2_R_LO]
        present = sorted(set(g.unit.values))
        flag = f"pass={','.join(passed) if passed else '-'};units_present={len(present)}/{len(units)}"
        out.append(dict(product=p, unit="SUMMARY9", stratum=st, scale=sc, n_cells=int(g.n_cells.sum()), n_blocks=int(g.n_blocks.sum()),
                        r_cell=np.nan, rho_cell=np.nan, r_between=np.nan, r_within=np.nan, r_within_lo=np.nan, r_within_hi=np.nan,
                        bias_cm=np.nan, rmse_cm=np.nan, n_ge99_5=np.nan, flag=flag, bias_scale=np.nan, rmse_scale=np.nan,
                        n_blocks_between=np.nan, n_blocks_within=np.nan, n_units_eval=int(ok.sum()), n_units_pass=int(len(passed)),
                        nboot=int(g.nboot.max()) if len(g) else 0))
    return out


# ================================================================ 하위 단위 실행
def run_variogram(a, D, FL, region):
    df = D.df
    idx, w, n_bad = region_cells(D, region)
    lat, lon, blk = df.lat.values[idx], df.lon.values[idx], df.block.values[idx]
    rows, meta = [], dict(region=region, n_cells=int(len(idx)), n_nonfinite_z=n_bad, strata={}, skipped=[])
    for st in STRATA:
        m = stratum_mask(FL, idx, st)
        if m.sum() < MIN_STRATUM_CELLS:
            meta["skipped"].append(dict(item="variogram", region=region, stratum=st, n_cells=int(m.sum()), reason=f"cells<{MIN_STRATUM_CELLS}"))
            continue
        r, info = variogram_rows(region, st, lat[m], lon[m], w[m], blk[m], a.EDGES, a.chunk)
        rows += r
        meta["strata"][st] = info
    return rows, meta, None


def run_icc(a, D, FL, region):
    df = D.df
    idx, w, n_bad = region_cells(D, region)
    blk = df.block.values[idx]
    rows, meta = [], dict(region=region, n_cells=int(len(idx)), n_nonfinite_z=n_bad, strata={}, skipped=[])
    for st in STRATA:
        m = stratum_mask(FL, idx, st)
        if m.sum() < MIN_STRATUM_CELLS:
            meta["skipped"].append(dict(item="icc", region=region, stratum=st, n_cells=int(m.sum()), reason=f"cells<{MIN_STRATUM_CELLS}"))
            continue
        r = icc_row(region, st, w[m], blk[m], a.NBOOT, seed_of("lgu-c2", region, st))
        rows.append(r)
        meta["strata"][st] = dict(n_cells=r["n_cells"], n_blocks=r["n_blocks"], flag=r["flag"])
    return rows, meta, None


def run_proximity(a, D, FL, sub):
    df = D.df
    t, m = sub.split(":", 1)
    if m not in H.valid_modes(t):
        raise ValueError(f"대상 {t} 에 모드 {m} 는 없다")
    t_idx, parent, src_idx, comp = D.source_idx(t, m)
    use, skipped = usable_splits(D, t, a.SPLITS)
    lat, lon, blk = df.lat.values.astype(float), df.lon.values.astype(float), df.block.values
    rows = []
    for sp in use:
        A_idx, B_idx = half_split_blocks(df, t_idx, sp)
        evB = B_idx[eval_mask(df.iloc[B_idx])]
        rows += proximity_split(t, m, sp, lat, lon, A_idx, evB, src_idx, blk, a.N_GRID, a.DRAWS, a.DISTS)
    rows += proximity_summary(rows)
    meta = dict(target=t, mode=m, parent=parent, n_target=int(len(t_idx)), splits_used=use, splits_skipped=skipped,
                **{k: v for k, v in comp.items()})
    return rows, meta, None


def unit_idx(D, unit):
    df = D.df
    if unit in D.macros:
        return np.where(df.macro.values == unit)[0]
    subs = np.asarray(df["sub"].values, object)
    idx = np.where(subs == unit)[0]
    if len(idx) == 0:
        raise ValueError(f"SAR 단위 {unit} 의 셀이 없다(macro 지역도 sub 열의 하위 지역도 아니다)")
    return idx


def run_sar(a, D, FL, unit):
    df = D.df
    idx = unit_idx(D, unit)
    y = df.y.values.astype(float)
    blk = df.block.values
    rows, codes = [], []
    meta = dict(unit=unit, n_cells=int(len(idx)), products={})
    std = df["polsar_std"].values.astype(float) if "polsar_std" in df else None
    for prod in SAR_PRODUCTS:
        if prod not in df:
            meta["products"][prod] = "열 없음"
            continue
        v = df[prod].values.astype(float)
        valid = sar_valid(df, prod)
        codes += sar_code_rows(prod, unit, v[idx], valid[idx], y[idx], std[idx] if (std is not None and prod == "polsar_alt") else None)
        meta["products"][prod] = dict(n_valid=int(valid[idx].sum()))
        base = valid[idx] & np.isfinite(y[idx])
        with np.errstate(invalid="ignore"):
            hi = np.nan_to_num(v[idx], nan=-np.inf) >= POLSAR_CODE_MIN
        for st in SAR_STRATA:
            if st == "lt99_5" and prod != "polsar_alt":
                continue
            if st == "lt99_5":                                               # 분류 부호 의심 셀(99.5 이상)을 뺀 층. 기기 층과 겹치지 않는다
                m = base & ~hi
            else:
                m = base & stratum_mask(FL, idx, st)
            ii = idx[m]
            n_ge = int((m & hi).sum()) if prod == "polsar_alt" else np.nan  # 그 단위·층의 제품 유효 셀 가운데 99.5 이상인 셀 수
            for sc in SAR_SCALES:
                s = sar_row_stats(v[ii], y[ii], blk[ii], sc, a.NBOOT, seed_of("lgu-c4", prod, unit, st, sc))
                rows.append(dict(product=prod, unit=unit, stratum=st, scale=sc, n_ge99_5=n_ge, n_units_eval=np.nan, n_units_pass=np.nan, **s))
    return rows, meta, codes


RUNNERS = dict(variogram=run_variogram, icc=run_icc, proximity=run_proximity, sar=run_sar)


# ================================================================ 조각
def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def shard_paths(a, item, sub) -> dict:
    base = a.SHARDS / f"{a.TAG}__{item}__{_safe(sub)}"
    return dict(part=Path(f"{base}_part.csv"), codes=Path(f"{base}_codes.csv"), unit=Path(f"{base}_unit.json"))


def item_unit_path(a, item) -> Path:
    return a.SHARDS / f"{a.TAG}__{item}_unit.json"


def out_path(a, name) -> Path:
    return a.OUT / f"{a.PREFIX}_{name}"


def task_cfg(a, item, sub) -> dict:
    """하위 단위의 설정(해시 대상). 결과를 바꾸는 설계값과 입력 표 해시만 넣는다(--chunk, --workers, --threads 는 넣지 않는다)."""
    cfg = dict(format=FORMAT, item=item, sub=str(sub), inputs=input_hashes(a), min_stratum_cells=MIN_STRATUM_CELLS)
    if item == "variogram":
        cfg.update(edges=[float(v) for v in a.EDGES], strata=list(STRATA), fit_min_pairs=FIT_MIN_PAIRS, a_bounds=list(A_BOUNDS))
    elif item == "icc":
        cfg.update(nboot=int(a.NBOOT), strata=list(STRATA), tau_floor=TAU_FLOOR)
    elif item == "proximity":
        cfg.update(splits=list(a.SPLITS), n_grid=list(a.N_GRID), draws=int(a.DRAWS), dists=list(a.DISTS), buffer_km=float(a.buffer_km))
    elif item == "sar":
        cfg.update(nboot=int(a.NBOOT), strata=list(SAR_STRATA), scales=list(SAR_SCALES), products=list(SAR_PRODUCTS),
                   min_cells=SAR_MIN_CELLS, min_blocks=SAR_MIN_BLOCKS, corr_min_blocks=CORR_MIN_BLOCKS, code_min=POLSAR_CODE_MIN)
    return cfg


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _frame(rows, cols) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    for c in cols:
        if c not in d:
            d[c] = np.nan
    return d[cols]


def build_tasks(a) -> list:
    """(항목, 하위 단위) 목록. 변동도의 큰 지역이 먼저 오도록 항목 순서는 variogram, icc, proximity, sar 로 고정한다."""
    tasks = []
    for item in ITEMS:
        if item not in a.ITEMS:
            continue
        if item in ("variogram", "icc"):
            tasks += [(item, r) for r in a.REGIONS]
        elif item == "proximity":
            tasks += [(item, f"{t}:{m}") for t, m in a.TARGETS]
        else:
            tasks += [(item, u) for u in a.SAR_UNITS]
    return tasks


def run_task(a, item, sub) -> dict:
    """하위 단위 하나를 계산하고 조각을 쓴다. 예외는 그 하위 단위의 실패로 기록한다(unit.json status failed)."""
    t0 = time.time()
    p = shard_paths(a, item, sub)
    cfg = task_cfg(a, item, sub)
    p["unit"].unlink(missing_ok=True)                                         # 이전 세대의 완료 표지를 먼저 지운다
    rec = dict(item=item, sub=str(sub), status="ok", error="", n_rows=0)
    meta, codes = {}, None
    try:
        D = get_data(a)
        FL = get_flags(a, D)
        rows, meta, codes = RUNNERS[item](a, D, FL, sub)
        LC.atomic_text(p["part"], _frame(rows, ITEM_COLS[item]).to_csv(index=False))
        if codes is not None:
            LC.atomic_text(p["codes"], _frame(codes, CODE_COLS).to_csv(index=False))
        rec["n_rows"] = int(len(rows))
    except (Exception, SystemExit) as e:                                     # noqa: BLE001  (표지 표 오류 등 SystemExit 도 그 단위의 실패로 남긴다)
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=8))
    rec["elapsed_s"] = round(time.time() - t0, 2)
    unit = dict(status=rec["status"], item=item, sub=str(sub), cfg=cfg, cfg_hash=LC.cfg_hash(cfg), **code_shas(), n_rows=rec["n_rows"],
                elapsed_s=rec["elapsed_s"], error=rec["error"], traceback=rec.get("traceback", ""), meta=meta, pid=os.getpid(),
                threads=os.environ.get("OMP_NUM_THREADS", ""), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    LC.atomic_text(p["unit"], json.dumps(_jsonable(unit), ensure_ascii=False, indent=1))
    return rec


_WA = None


def _worker_init(argv, threads):
    global _WA
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    LC.set_thread_env(threads)
    _WA = parse_args(argv)


def _worker_task(item, sub):
    return run_task(_WA, item, sub)


def run_tasks(a, tasks, argv, threads) -> list:
    """하위 단위를 실행한다. --resume 이면 완료 조각을 건너뛴다. 반환: 실행 기록 목록."""
    todo, recs = [], []
    for item, sub in tasks:
        p = shard_paths(a, item, sub)
        need = [p["part"]] + ([p["codes"]] if item == "sar" else [])
        if a.resume:
            ok, why = LC.unit_state(p["unit"], need, task_cfg(a, item, sub))
            if ok:
                recs.append(dict(item=item, sub=sub, status="resumed", error="", n_rows=np.nan, elapsed_s=0.0))
                continue
        todo.append((item, sub))
    print(f"[h46] 하위 단위 {len(tasks)}개 가운데 실행 {len(todo)}개(건너뜀 {len(tasks) - len(todo)}개), 워커 {a.workers}, 스레드 {threads}",
          flush=True)
    if a.workers <= 1 or len(todo) <= 1:
        for item, sub in todo:
            r = run_task(a, item, sub)
            print(f"  [{r['status']}] {item}|{sub} {r['elapsed_s']} s" + (f" {r['error']}" if r["error"] else ""), flush=True)
            recs.append(r)
        return recs
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(a.workers, len(todo)), mp_context=ctx, initializer=_worker_init,
                             initargs=(list(argv), int(threads))) as ex:
        futs = {ex.submit(_worker_task, item, sub): (item, sub) for item, sub in todo}
        for f in as_completed(futs):
            item, sub = futs[f]
            try:
                r = f.result()
            except Exception as e:                                           # noqa: BLE001  (워커 비정상 종료)
                r = dict(item=item, sub=sub, status="failed", error=f"{type(e).__name__}: {e}", n_rows=0, elapsed_s=np.nan)
            print(f"  [{r['status']}] {item}|{sub} {r.get('elapsed_s')} s" + (f" {r['error']}" if r.get("error") else ""), flush=True)
            recs.append(r)
    return recs


# ================================================================ 집계
def summarize(a, tasks, run_recs=None) -> int:
    """하위 단위 조각을 모아 표를 쓴다. 조각이 없거나 실패이거나 설정 해시가 다르면 그 항목의 표를 쓰지 않는다. 반환: 종료 코드."""
    t0 = time.time()
    failed, timing, units_meta, skipped_rows, cells = [], [], [], [], []
    written = {}
    for item in [v for v in ITEMS if v in a.ITEMS]:
        subs = [s for it, s in tasks if it == item]
        frames, code_frames, bad, hashes = [], [], [], {}
        for s in subs:
            p = shard_paths(a, item, s)
            need = [p["part"]] + ([p["codes"]] if item == "sar" else [])
            cfg = task_cfg(a, item, s)
            ok, why = LC.unit_state(p["unit"], need, cfg)
            u = {}
            if p["unit"].exists():
                try:
                    u = json.loads(p["unit"].read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    u = {}
            timing.append(dict(item=item, sub=s, status=u.get("status", "missing") if ok else f"not_ok:{why}", elapsed_s=u.get("elapsed_s"),
                               n_rows=u.get("n_rows"), pid=u.get("pid"), threads=u.get("threads"), finished=u.get("finished")))
            if not ok:
                bad.append(dict(item=item, sub=s, reason=why, error=u.get("error", "")))
                continue
            hashes[s] = u.get("cfg_hash")
            m = u.get("meta", {}) or {}
            units_meta.append(dict(item=item, sub=s, meta=m))
            skipped_rows += list(m.get("skipped", []) or [])
            if item in ("variogram", "icc"):
                for st, inf in (m.get("strata", {}) or {}).items():
                    cells.append(dict(item=item, region=s, stratum=st, n_cells=inf.get("n_cells"), n_blocks=inf.get("n_blocks")))
            frames.append(pd.read_csv(p["part"], keep_default_na=True))
            if item == "sar":
                code_frames.append(pd.read_csv(p["codes"]))
        if bad:
            failed += bad
            print(f"[h46] 항목 {item}: 하위 단위 {len(bad)}/{len(subs)}개가 완료되지 않아 표를 쓰지 않는다", flush=True)
            continue
        cols = ITEM_COLS[item]
        tab = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame(columns=cols)
        tab = _frame(tab.to_dict("records"), cols) if len(tab) else pd.DataFrame(columns=cols)
        if item == "sar":
            summ = sar_summary_rows(tab)
            if summ:
                tab = pd.concat([tab, _frame(summ, cols)], ignore_index=True)
            codes = pd.concat([f for f in code_frames if len(f)], ignore_index=True) if any(len(f) for f in code_frames) \
                else pd.DataFrame(columns=CODE_COLS)
            LC.atomic_text(out_path(a, "sar_codes.csv"), _frame(codes.to_dict("records"), CODE_COLS).to_csv(index=False))
        path = out_path(a, f"{item}.csv")
        LC.atomic_text(path, tab.to_csv(index=False))
        written[item] = str(path)
        iu = dict(status="ok", item=item, subs=list(subs), sub_cfg_hash=hashes, cfg_hash=LC.cfg_hash(hashes), **code_shas(), n_rows=int(len(tab)),
                  table=str(path), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        LC.atomic_text(item_unit_path(a, item), json.dumps(_jsonable(iu), ensure_ascii=False, indent=1))
        print(f"[h46] 항목 {item}: {len(tab)}행 → {path}", flush=True)
    LC.atomic_text(out_path(a, "timing.csv"), pd.DataFrame(timing, columns=["item", "sub", "status", "elapsed_s", "n_rows", "pid", "threads",
                                                                             "finished"]).to_csv(index=False))
    LC.atomic_text(out_path(a, "failed.csv"), pd.DataFrame(failed, columns=["item", "sub", "reason", "error"]).to_csv(index=False))
    try:
        nice_now = os.nice(0)
    except OSError:
        nice_now = None
    meta = dict(harness="h46_spatial_diagnostics", plan="docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 5절", blind=BLIND,
                note="가설과 판정이 없는 진단이다. LGX·LG 의 산출을 읽지 않는다. 격자 원자료를 읽지 않는다.",
                args={k: v for k, v in vars(a).items() if not k.isupper() or k in ("ITEMS", "REGIONS", "TARGETS", "SAR_UNITS", "SPLITS", "N_GRID",
                                                                                     "DRAWS", "DISTS", "EDGES", "NBOOT", "TAG", "PREFIX")},
                **code_shas(), inputs=input_hashes(a), tables=written, cell_counts=cells, rows_not_made=skipped_rows, units=units_meta,
                n_failed=len(failed), run_records=run_recs or [], summarize_sec=round(time.time() - t0, 2), niceness=nice_now,
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    LC.atomic_text(out_path(a, "meta.json"), json.dumps(_jsonable(meta), ensure_ascii=False, indent=1))
    return 1 if failed else 0


# ================================================================ 적합 수(학습이 없으므로 단위 수와 쌍 수)
def count_only(a) -> int:
    """자료 적재 외의 계산을 하지 않는다(거리, 재표집, 원천 버퍼 계산 없음). 단위 수, 쌍 수의 상한, 근접 표의 행 수를 출력한다."""
    t0 = time.time()
    D = get_data(a)
    FL = get_flags(a, D, missing_ok=True)
    df = D.df
    rows = []
    for item in ("variogram", "icc"):
        if item not in a.ITEMS:
            continue
        for r in a.REGIONS:
            try:
                idx, _, _ = region_cells(D, r)
            except ValueError:
                rows.append(dict(item=item, sub=r, detail="셀 없음", n_cells=0, n_blocks=0, pairs_upper=0, rows=0)); continue
            for st in STRATA:
                m = stratum_mask(FL, idx, st); n = int(m.sum())
                made = n >= MIN_STRATUM_CELLS
                rows.append(dict(item=item, sub=r, detail=st, n_cells=n, n_blocks=int(len(np.unique(df.block.values[idx][m]))),
                                 pairs_upper=int(n * (n - 1) // 2) if item == "variogram" else 0,
                                 rows=(len(a.EDGES) if item == "variogram" else 1) if made else 0))
    if "proximity" in a.ITEMS:
        for t, m in a.TARGETS:
            use, skipped = usable_splits(D, t, a.SPLITS)
            info = D.split_structure(t)
            nr, ns = 0, set()
            for sp in use:
                ns_sp = sorted({n for n, _ in H.cells_of(a.N_GRID, a.DRAWS, info[sp]["n_A"])})
                nr += len(ns_sp) * len(a.DISTS); ns |= set(ns_sp)
            nr += len(ns) * len(a.DISTS)
            t_idx = D.target_idx(t)
            rows.append(dict(item="proximity", sub=f"{t}:{m}", detail=f"splits={use};skipped={len(skipped)}", n_cells=int(len(t_idx)),
                             n_blocks=int(len(np.unique(df.block.values[t_idx]))), pairs_upper=0, rows=int(nr)))
    if "sar" in a.ITEMS:
        for u in a.SAR_UNITS:
            try:
                idx = unit_idx(D, u)
            except ValueError:
                rows.append(dict(item="sar", sub=u, detail="셀 없음", n_cells=0, n_blocks=0, pairs_upper=0, rows=0)); continue
            for prod in SAR_PRODUCTS:
                nv = int(sar_valid(df, prod)[idx].sum()) if prod in df else 0
                nst = len(SAR_STRATA) if prod == "polsar_alt" else len(SAR_STRATA) - 1
                rows.append(dict(item="sar", sub=u, detail=prod, n_cells=nv, n_blocks=int(len(np.unique(df.block.values[idx]))), pairs_upper=0,
                                 rows=nst * len(SAR_SCALES)))
    tab = pd.DataFrame(rows, columns=["item", "sub", "detail", "n_cells", "n_blocks", "pairs_upper", "rows"])
    n_units = {it: len([1 for i, _ in build_tasks(a) if i == it]) for it in a.ITEMS}
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 60):
        print(tab.to_string(index=False))
    print(f"[h46 count] 하위 단위 수 {n_units}, 변동도 쌍 수 상한 합계 {int(tab.pairs_upper.sum()):,}, 근접 표 행 수 "
          f"{int(tab[tab.item == 'proximity'].rows.sum()) if len(tab) else 0}, 적합 0건(학습 없음), 경과 {time.time() - t0:.1f} s", flush=True)
    LC.atomic_text(out_path(a, "count.csv"), tab.to_csv(index=False))
    return 0


# ================================================================ 주 함수
def _ensure_nice(target=NICE_TARGET):
    """nice 값이 target 미만이면 target 으로 올린다(공유 서버 규칙). 실패하면 그대로 둔다."""
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
        return os.nice(0)
    except OSError:
        return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse_args(argv)
    permitted = LC.require_permission(a)                                     # 허용 표지 없으면 --count-only 만(자료를 읽기 전에 거부)
    if permitted and max(1, int(a.workers)) * int(a.threads) > LOCAL_THREAD_CAP:   # LG_RESCALE=1 에도 적용한다(개정 1)
        raise SystemExit(f"[거부] 로컬 실행의 CPU 스레드 합계(--workers × --threads = {max(1, a.workers)} × {a.threads})가 "
                         f"{LOCAL_THREAD_CAP} 을 넘는다(사용자 지시 2026-09-30). 값을 줄인다")
    threads = int(a.threads) if permitted else 1
    LC.set_thread_env(threads)
    if a.GPUS:
        print(f"[h46] --gpus {a.gpus} 는 쓰지 않는다(GPU 를 쓰지 않는 진단이다)", flush=True)
    nice_now = _ensure_nice()
    print(f"[h46] tag {a.TAG}, 항목 {a.ITEMS}, nice {nice_now}, 스레드 {threads}", flush=True)
    if a.count_only:
        return count_only(a)
    if not a.summarize_only and not a.FLAGS.exists():                        # 층 계산에 필요한 표. 비용이 들기 전에 중단한다
        raise SystemExit(f"[h46] 셀 표지 표 {a.FLAGS} 가 없다(h42 --write-label-flags 로 만든 표를 쓴다)")
    reg = LC.claim_threads(a.TAG, max(1, int(a.workers)) * int(threads), LOCAL_THREAD_CAP, script=Path(__file__).name)  # 개정 1
    try:
        return _run(a, argv, threads)
    finally:
        LC.release_threads(reg)


def _run(a, argv, threads) -> int:
    """허용된 실행의 본체(하위 단위 계산과 집계). main 이 스레드 등록을 잡고 끝에 푼다."""
    tasks = build_tasks(a)
    recs = None
    code = 0
    if not a.summarize_only:
        recs = run_tasks(a, tasks, argv, threads)
        if any(r.get("status") == "failed" for r in recs):
            code = 1
    if a.no_summarize:
        return code
    return max(code, summarize(a, tasks, recs))


if __name__ == "__main__":
    sys.exit(main())
