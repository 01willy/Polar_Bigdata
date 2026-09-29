"""H40 · 라벨 격자 통합 재실행(LG). 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §1–§5(사전 등록)의 구현.

목적
  기법 × 라벨 수 × 지역 × 학습기를 같은 분할·같은 라벨 추출·같은 채점으로 처음부터 다시 돌린다.

대상(27, 지역 × 원천 모드)
  주 지역 x      Lena·Canada·Russia_W·Russia_E(확인적 평균 AB4), Russia_C·Greenland(점 추정만)
  알래스카 x     Alaska(지역 전체를 새 지역으로, 원천 = 알래스카 밖 전체)
  하위 지역 i·x  AL-1–AL-6, CA-2, CA-3, LE-1, LE-2 (h25 와 같은 블록 중심 k-means, 라벨 미사용, CA-1 제외)
  모드 i = 상위 지역 포함(대상 셀만 제외), x = 상위 macro 지역 전부 제외. 두 모드 모두 대상 경계 100 km 버퍼 제외.
  대상 지정: --targets "Russia_W,AL-3:i,Canada" (모드를 생략하면 그 대상의 유효 모드 ∩ --modes).

분할·라벨 수·추출
  분할 = m1_core.half_split_blocks(split_seed = 1..K) 만 쓴다. 중복 분할(A 블록 집합 동일)은 첫 분할만 실행·집계하고,
  채점 블록 2개 미만 분할은 무효로 기록한다. 유효 분할이 하나라도 있는 대상의 무효 분할은 실행하지 않는다(집계에 쓰이지 않는다).
  유효 분할이 없는 대상(Greenland)만 무효 분할을 실행해 점 추정에 쓴다. 채점 = B 블록의 eval_mask 셀.
  하위 지역 정의는 --subregion-map(블록 → 하위 지역 대응표 CSV)이 있으면 그 표를 쓰고, 없으면 k-means 로 계산한다.
  n 격자 {0, 3, 10, 40, 160, 320, 1000, all} 중 |A| 미만과 all(키 n = -1, 실제 라벨 수는 n_lab).
  추출 = 셀 무작위, seed = h4_common.seed_of(대상, 모드, 분할, n, 추출 번호). 모든 방법·학습기·α 가 같은 sel 을 쓴다.
  블록 분산 추출(placement = block: A 의 모든 블록을 무작위 순서로 순환 배정)은 n ∈ {10, 40, 160} 의 P1·R1(CatBoost)에만 추가한다.

방법(기준 구성: catboost_lo, α = 1, 셀 무작위, λ = 0.25; s = √TDD, E0 = 원천 최소제곱, E_n = κ = 10 수축)
  P0 E0·s | P1 E_n·s | P2 대상 n개 최소제곱 E·s | P3 오프셋 MLE(h4_common.offset_mle_*) | n = 0 에서 P1–P3 은 P0 과 같다(규약).
  D0 직접 ML(원천 실측 ∪ 대상 n 실측) | D1 = D0 + 대상 A 풀 유사라벨 E_n·s(r = 10, 라벨 셀은 실측)
  R0 E0·s + λ·g(원천 잔차 ∪ 대상 n 잔차, 앵커 E0) | R1 E_n·s + λ·g(원천 잔차는 E0 앵커, 대상 잔차는 E_n 앵커)
  R2 = R1 + 대상 A 풀 유사 잔차 행(잔차 0, r = 10, 라벨 셀은 실측 잔차) | R3 = D1 예측을 앵커로 둔 잔차 학습(적층)
  V1 log(y/s) 를 표준화 공변량 ridge 로 적합(원천 ∪ 대상 n), 앵커 = exp(ẑ)·s | V1r = V1 + λ·g
  λ ∈ {0.25, 0.5, 1.0} 은 같은 g 에서 사후 적용한다. BlockStore 키 = (method, learner, alpha, placement, n, draw, seed, lam).
  판정에 쓰는 기준 λ 는 방법별로 하나다: P·V1 = 0.0, D0·D1 = 1.0, 잔차 방법(R0–R3·V1r) = 0.25.
  구현 규약(계획서에 없는 세부): R3·V1r 의 원천 잔차는 앵커 모델의 표본 내 예측으로 계산한다(교차 적합 없음).
    V1 의 ẑ 는 학습 z 의 범위로 자른다(외삽 폭주 방지). 유사라벨 행은 (대상, 모드, 분할, seed) 로 고정한 복원 추출이다.

요인 축과 부분(--part)
  cpu  방법 축(12방법, CatBoost) · α 축(D0·R0·R1 × α {10, 100, cont}, n ≥ 3) · 배치 축(P1·R1 블록 분산)
       · 학습기 축 가운데 ridge · L3 용 α 중첩 선택(R0·R1, 구조 대상만)
  gpu  학습기 축(D0·R1·R2 × mlp·realmlp·ftt·tabm·cfm·ddpm·nflow, n ∈ {0, 10, 40, 160, all}, 대상 12, 분할 3, 추출 2, seed 2)
       · MLP α 축(D0·R0·R1 × α {1, 10, 100, cont}, 학습기 축과 같은 범위의 n ≥ 3)
  α: CatBoost 는 sample_weight(재표집 아님). 'cont' 는 원천만으로 학습한 모델을 init_model 로 두고 대상 행만으로 트리 100개 추가.
     MLP 는 검증 10 % 를 고유 행에서 먼저 떼고 학습 쪽 대상 행만 α 배 반복한다(검증 행은 반복하지 않는다. 표준화 통계와
     y 척도도 고유 행에서 구한다). 'cont' 는 원천 학습 후 대상 행 + 같은 수의 원천 재생 표본으로 학습률 1e-4, 20 epoch 미세조정
     (BatchNorm·Dropout 은 평가 모드로 고정). MLP 학습은 tab_models._torch_mods·_epochs_fit 을 그대로 호출한다.
     학습기 축의 R2 는 유사 잔차 행(복원 추출 r·n_src 행)을 그대로 학습 행렬에 넣는다. 신경망 학습기는 그 뒤에 검증 10 % 를
     떼므로 검증 행에 학습 행의 복제본이 들어간다(채점 셀 누설은 아니다. tab_models 의 절차를 바꾸지 않기 위한 규약).
  α 중첩 선택(alpha = 'nested'): 후보는 --nested-alphas(기본 10, 100, cont. 계획서 L3 의 후보 집합)다. 선택된 n개 라벨 안에서
     A 블록 단위 교차검증(블록 하나 제외; --nested-max-folds > 0 이고 블록이 그보다 많으면 블록을 그 수의 묶음으로 나눈다).
     교차검증 안에서는 E_n 도 학습 쪽 라벨로 다시 구한다. n < 10 이거나 라벨이 한 블록에만 있으면 α = 1(선택 불가 표시).
     B 블록은 쓰지 않는다. 선택은 seed 0 적합으로 하고 같은 α 를 모든 seed 에 적용한다.
  RealMLP 는 --epochs 가 아니라 --realmlp-epochs(기본 256 = pytabkit 기본값)를 쓴다.

누설 규약
  대상 라벨은 선택된 n개만 학습·계수·α 선택에 쓴다. 표준화·중앙값 대체 통계는 학습 행렬에서만 구한다. 채점 셀 라벨은 채점 전용.

채점(h4_common)
  실행 단위마다 채점 블록별 SSE·셀 수를 BlockStore 로 저장하고, 집계에서 분할 안 채점 블록 부트스트랩(1,000회, 방법·추출 공통 인덱스)을 쓴다.
  셀 가중 RMSE(주)와 블록 등가중 RMSE(보조)의 CI 가 모두 0 을 제외할 때만 '유의'(sig_* 열)로 쓴다. L1–L8 판정은 모두 이 기준이다.
  예측에 비유한 값이 하나라도 있는 키는 BlockStore 에 넣지 않고 runs 에 n_nonfinite·fit_flag 로 기록한다(채점 셀 집합 통일).
  적합 단위 예외는 그 키만 실패로 기록하고 나머지를 진행한다. ImportError·MemoryError 는 조각 전체를 실패로 둔다.

산출(<out-dir>, 기본 data/processed/lg)
  shards/<tag>__cpu__<대상>__<모드>__s<분할>_{runs.csv, blocksse.npz, unit.json}            cpu 부분 조각
  shards/<tag>__gpu__<대상>__<모드>__s<분할>__<학습기>_{runs.csv, blocksse.npz, unit.json}   gpu 부분 조각(학습기 단위)
    원자적 기록, unit.json 이 완료 표지다. unit.json 에 설정 요약(cfg)·해시(cfg_hash)·코드 해시·축|학습기|방법별 적합 수와 시간을 남긴다.
    --resume 은 cfg_hash 가 같고 status 가 failed 가 아닌 조각만 건너뛴다.
  <tag>_curve.csv · <tag>_minn.csv · <tag>_tests.csv · <tag>_targets.csv · <tag>_timing.csv · <tag>_failed.csv · <tag>_meta.json
  <tag>_count_<part>.csv

실행 환경(사용자 지시 2026-09-29)
  이 서버는 공유 서버다. 로컬에서는 --count-only, --write-subregion-map, --summarize-only 만 실행한다(스레드 1–2).
  스모크·사전 점검·본 실행은 Rescale 작업에서 한다. 스모크가 아닌 실행은 --allow-local 또는 환경 변수 LG_RESCALE=1 이 없으면 거부한다.
  --gpus 의 기본값은 빈 문자열(CPU)이다. 부모 환경의 CUDA_VISIBLE_DEVICES 가 빈 문자열이면 --gpus 를 무시한다.

실행(ROOT)
  적합 수:     python3 scripts/3_deep_learning/h40_label_grid.py --part cpu --count-only --threads 1
  하위 지역:   python3 scripts/3_deep_learning/h40_label_grid.py --write-subregion-map --threads 1
  사전 점검:   LG_RESCALE=1 python3 scripts/3_deep_learning/h40_label_grid.py --part gpu --precheck --gpus 0 --threads 4
               (Rescale. 학습기 7종, Canada x, 분할 1, n {0, 40}, 본 실행과 같은 epochs. <tag>_precheck_timing.csv 에 학습기별 시간)
  본 실행:     LG_RESCALE=1 python3 scripts/3_deep_learning/h40_label_grid.py --part cpu --workers 10 --threads 4 --resume --no-summarize
               LG_RESCALE=1 python3 scripts/3_deep_learning/h40_label_grid.py --part gpu --gpus 0,1,2,3 --procs-per-gpu 3 --threads 4 --resume --no-summarize
               (Rescale iolite-4 한 노드에서 두 명령을 함께 실행할 수 있다. 조각 이름에 부분이 들어 있어 충돌하지 않는다)
  집계:        python3 scripts/3_deep_learning/h40_label_grid.py --summarize-only --threads 1
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path



def _peek_threads(default="4"):
    """numpy 를 부르기 전에 스레드 수를 정한다(--threads 를 미리 읽는다)."""
    av = sys.argv
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return av[i + 1]
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return default


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _peek_threads())

_UNRAISABLE = sys.unraisablehook


def _quiet_unraisable(u):
    """threadpoolctl 의 ctypes 콜백 예외 출력(동작에는 영향 없음)을 억제한다. 다른 예외는 그대로 보인다."""
    tb = u.exc_traceback
    while tb is not None:
        if "threadpoolctl" in tb.tb_frame.f_code.co_filename:
            return
        tb = tb.tb_next
    _UNRAISABLE(u)


sys.unraisablehook = _quiet_unraisable
warnings.filterwarnings("ignore", module="threadpoolctl")
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*glibc.*")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET                                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402
import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import (BlockStore, save_stores, load_stores, boot_delta_blocks, min_n,         # noqa: E402
                             offset_mle_prior, offset_mle_estimate, seed_of)
from polar.m1_stats import summarize_delta, MIN_BLOCKS_CI                                             # noqa: E402

# 같은 (nb, nboot, seed) 의 재표집 행렬을 재사용한다(값은 동일, 계산만 줄인다. h31 과 같은 처리).
if not hasattr(H4.boot_weights, "cache_info"):
    H4.boot_weights = functools.lru_cache(maxsize=4096)(H4.boot_weights)

FEATS = INPUT_SETS["x25"]
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
MAIN_POINT = ["Russia_C", "Greenland"]
ALASKA = "Alaska"
SUB_AL = [f"AL-{i}" for i in range(1, 7)]
SUB_OTHER = ["CA-2", "CA-3", "LE-1", "LE-2"]
LEARNER_TARGETS = [(t, "x") for t in MAIN4 + MAIN_POINT + [ALASKA] + ["AL-1", "AL-2", "AL-3", "CA-2", "CA-3"]]
NESTED_TARGETS = [("Lena", "x"), ("Canada", "x"), ("AL-2", "i"), ("AL-5", "i"), ("AL-2", "x"), ("AL-5", "x")]
METHODS_ALL = ["P0", "P1", "P2", "P3", "D0", "D1", "R0", "R1", "R2", "R3", "V1", "V1r"]
CPU_LEARNERS = ["catboost_lo", "ridge"]
GPU_LEARNERS = ["mlp", "realmlp", "ftt", "tabm", "cfm", "ddpm", "nflow"]
BASE_LEARNER = "catboost_lo"
ALPHA_METHODS = ["D0", "R0", "R1"]
LEARNER_METHODS = ["D0", "R1", "R2"]
PLACE_METHODS = ["P1", "R1"]
RESID_METHODS = ["R0", "R1", "R2", "R3", "V1r"]
ALPHA_ORDER = ["1", "10", "100", "cont"]
LAM_BASE = 0.25
L3_TARGETS = ["Lena", "Canada", "AL-2", "AL-5"]                   # 계획서 §4 L3 가 열거한 구조 대상
L3_PRIMARY = ("R0", "i")                                          # L3 주 판정: E0 고정 앵커(R0), 알래스카 하위는 모드 i
MIN_POOL_REGIONS = 2                                              # 층화 평균 판정에 필요한 CI 풀 지역 수의 하한
LEARNER_TIER = {"mlp": 0, "cfm": 0, "ddpm": 0, "tabm": 0, "nflow": 1, "ftt": 1, "realmlp": 2}   # 실행 순서(빠른 학습기 먼저)
# 적합 시간 추정용 기준값(--count-only 전용, 결과에는 쓰지 않는다): 학습기 → (기준 행 수에서 1건의 시간 s, 기준 행 수, 행 수 지수).
# 신경망·생성 모델 = M1 조각 fit_s 중앙값(로컬 GPU, 학습 행 8,890–15,125, 검증 보고 2026-09-29). CatBoost = LG 스모크 unit.json
# (Canada x, 4스레드: 1배 행 약 0.5 s, 11배 행 약 2.0 s). 생성 모델의 표본 추출 시간(채점 셀 수에 비례)은 넣지 않았다.
BASE_FIT = {"catboost_lo": (0.5, 16700, 0.6), "ridge": (0.05, 16700, 1.0), "mlp": (1.0, 12000, 1.0), "cfm": (1.2, 12000, 1.0),
            "ddpm": (1.5, 12000, 1.0), "tabm": (2.6, 12000, 1.0), "nflow": (8.6, 12000, 1.0), "ftt": (11.2, 12000, 1.0),
            "realmlp": (161.5, 12000, 1.0)}
KEY_COLS = ["method", "learner", "alpha", "placement", "n", "draw", "seed", "lam"]                  # BlockStore 키 순서
GRP_COLS = ["method", "learner", "alpha", "placement", "n", "lam"]                                   # 곡선 한 행 = 키에서 draw·seed 를 뺀 것
KI = {c: i for i, c in enumerate(KEY_COLS)}
P0_KEY = ("P0", "none", "1", "cell", 0, 0, -1, 0.0)
P0_GRP = ("P0", "none", "1", "cell", 0, 0.0)
TRACE = None            # 시험용: 리스트를 넣으면 모든 적합의 학습 행렬과 정보를 기록한다


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H40 라벨 격자 통합 재실행(LG)")
    ap.add_argument("--part", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--targets", default="", help="쉼표 목록. '이름' 또는 '이름:모드'. 기본 = 계획서 §1 의 27개")
    ap.add_argument("--modes", default="i,x")
    ap.add_argument("--splits", type=int, default=5, help="half_split_blocks split_seed 1..K")
    ap.add_argument("--n-grid", default="0,3,10,40,160,320,1000,all")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--learners", default="", help="기본: cpu = catboost_lo,ridge / gpu = mlp,realmlp,ftt,tabm,cfm,ddpm,nflow")
    ap.add_argument("--methods", default="", help="기본: P0–P3, D0, D1, R0–R3, V1, V1r 전부")
    ap.add_argument("--alphas", default="1,10,100,cont")
    ap.add_argument("--lams", default="0.25,0.5,1.0")
    ap.add_argument("--workers", type=int, default=2, help="CPU 프로세스 수(gpu 부분은 --gpus 가 빈 문자열일 때만 사용). 0 = 풀 없이 차례로. "
                                                           "본 실행 값은 Rescale 작업 명령에서만 준다")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpus", default="", help="GPU 부분: 쉼표 목록, GPU 마다 워커 --procs-per-gpu 개. 기본은 빈 문자열(CPU). "
                                               "로컬 GPU 는 쓰지 않는다. Rescale 작업 명령에서만 명시한다")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--realmlp-epochs", type=int, default=256, help="RealMLP 의 n_epochs(기본 256 = pytabkit 기본값). --epochs 와 별개")
    ap.add_argument("--allow-local", action="store_true", help="스모크가 아닌 실행을 허용한다(환경 변수 LG_RESCALE=1 과 같은 효과)")
    ap.add_argument("--precheck", action="store_true", help="Rescale 사전 점검: 학습기 전부, 대상 1개(기본 Canada x), 분할 1, n {0, 40}, "
                                                           "추출 1, seed 1, 본 실행과 같은 epochs. tag 에 _precheck 를 붙인다")
    ap.add_argument("--pool-retries", type=int, default=2, help="워커 비정상 종료로 풀이 깨졌을 때 남은 단위로 풀을 다시 만드는 횟수")
    ap.add_argument("--allow-mixed-cfg", action="store_true", help="집계: 조각 사이 설정 해시가 달라도 진행한다(기본은 오류)")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv", help="블록 → 하위 지역 대응표(상대 경로는 --data-dir 기준). "
                                                                            "있으면 읽어 쓰고 없으면 k-means 로 계산한다")
    ap.add_argument("--write-subregion-map", action="store_true", help="k-means 하위 지역 정의를 대응표로 저장하고 끝낸다")
    ap.add_argument("--out-dir", default="data/processed/lg")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default="lg")
    ap.add_argument("--resume", action="store_true", help="조각(unit.json)이 있는 작업 단위는 건너뛴다")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--count-only", action="store_true", help="학습 없이 적합 수와 작업 목록만 출력")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true", help="실행 뒤 집계를 생략(조각만 남긴다)")
    # 아래는 계획서 §2–§3 의 고정 설계값이다. 바꾸면 사전 등록에서 벗어난다.
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=10.0)
    ap.add_argument("--buffer-km", type=float, default=100.0)
    ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
    ap.add_argument("--min-cells-prior", type=int, default=3)
    ap.add_argument("--learner-targets", default="", help="학습기 축 대상(기본: 주 7 x + AL-1–3 x + CA-2·CA-3 x)")
    ap.add_argument("--learner-splits", type=int, default=3)
    ap.add_argument("--learner-draws", type=int, default=2)
    ap.add_argument("--learner-n-grid", default="0,10,40,160,all")
    ap.add_argument("--place-n-grid", default="10,40,160")
    ap.add_argument("--nested-targets", default="", help="α 중첩 선택 대상(기본: Lena, Canada, AL-2, AL-5)")
    ap.add_argument("--nested-methods", default="R0,R1")
    ap.add_argument("--nested-alphas", default="10,100,cont", help="α 중첩 선택 후보(계획서 L3: 10, 100, 이어 학습). --alphas 에 있는 값만 쓴다")
    ap.add_argument("--learner-r2-exclude", default="", help="학습기 축에서 R2 를 빼는 학습기 목록. 기본은 없음(사전 등록 범위). "
                                                             "쓰면 계획서 개정 이력에 적는다")
    ap.add_argument("--nested-min-n", type=int, default=10)
    ap.add_argument("--nested-max-folds", type=int, default=0, help="0 = 블록 하나 제외(엄격). 양수면 블록을 그 수의 묶음으로 나눈다")
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--cont-trees", type=int, default=100)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--nboot", type=int, default=1000)
    return finalize(ap.parse_args(argv))


def _n_list(txt):
    return [-1 if v.strip() == "all" else int(v) for v in str(txt).split(",") if v.strip()]


def _pairs(txt, modes):
    out = []
    for tok in [v.strip() for v in str(txt).split(",") if v.strip()]:
        if ":" in tok:
            t, m = tok.split(":"); out.append((t, m))
        else:
            out += [(tok, m) for m in valid_modes(tok) if m in modes]
    return out


def valid_modes(t):
    return ["i", "x"] if t in SUB_AL + SUB_OTHER or (len(t) == 4 and t[2] == "-") else ["x"]


def default_targets(modes):
    out = [(t, "x") for t in MAIN4 + MAIN_POINT + [ALASKA]]
    out += [(t, "i") for t in SUB_AL] + [(t, "x") for t in SUB_AL]
    out += [(t, m) for t in SUB_OTHER for m in ("i", "x")]
    return [(t, m) for t, m in out if m in modes]


def finalize(a):
    a.MODES = [m for m in a.modes.split(",") if m]
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    a.TAG = a.tag + ("_smoke" if a.smoke else "") + ("_precheck" if a.precheck else "")
    a.SPLITS = list(range(1, a.splits + 1)); a.N_GRID = _n_list(a.n_grid); a.DRAWS = a.draws; a.SEEDS = list(range(a.seeds))
    a.TARGETS = _pairs(a.targets, a.MODES) if a.targets else default_targets(a.MODES)
    a.LEARNER_TARGETS = _pairs(a.learner_targets, ["i", "x"]) if a.learner_targets else list(LEARNER_TARGETS)
    a.NESTED_TARGETS = _pairs(a.nested_targets, ["i", "x"]) if a.nested_targets else list(NESTED_TARGETS)
    if a.smoke:                                                    # 스모크: 대상 3, 분할 1, n {0, 3, 40, all}, 추출 1, seed 1
        a.SPLITS = [1]; a.N_GRID = [0, 3, 40, -1]; a.DRAWS = 1; a.SEEDS = [0]
        if not a.targets:
            a.TARGETS = [("Russia_W", "x"), ("AL-3", "i"), ("Canada", "x")]
        if not a.learner_targets:                                   # 스모크에서는 하위 지역 경로도 거치도록 대상 전부를 학습기 축에 넣는다
            a.LEARNER_TARGETS = list(a.TARGETS)
        if a.part == "gpu":
            a.epochs = min(a.epochs, 3)
            if not a.learners:
                a.learners = "mlp,cfm"
            a.realmlp_epochs = min(a.realmlp_epochs, 3)
    if a.precheck:                                                 # 사전 점검: 범위는 작게, epochs 는 본 실행과 같게(학습기별 시간 측정)
        a.SPLITS = [1]; a.N_GRID = [0, 40]; a.DRAWS = 1; a.SEEDS = [0]
        a.learner_n_grid = "0,40"; a.place_n_grid = "40"
        if not a.targets:
            a.TARGETS = [("Canada", "x")]
        if not a.learner_targets:
            a.LEARNER_TARGETS = list(a.TARGETS)
    a.LEARNERS = [v for v in a.learners.split(",") if v] if a.learners else list(CPU_LEARNERS if a.part == "cpu" else GPU_LEARNERS)
    bad = [v for v in a.LEARNERS if v not in (CPU_LEARNERS if a.part == "cpu" else GPU_LEARNERS)]
    if bad:
        raise SystemExit(f"--part {a.part} 에서 쓸 수 없는 학습기: {bad}")
    a.METHODS = [v for v in a.methods.split(",") if v] if a.methods else list(METHODS_ALL)
    a.LAMS = [float(v) for v in a.lams.split(",")]
    a.ALPHAS = [v.strip() for v in a.alphas.split(",") if v.strip()]
    a.LEARNER_N = _n_list(a.learner_n_grid); a.PLACE_N = _n_list(a.place_n_grid)
    a.LEARNER_SPLITS = [s for s in a.SPLITS if s <= a.learner_splits]
    a.NESTED_METHODS = [v for v in a.nested_methods.split(",") if v]
    a.NESTED_ALPHAS = [v.strip() for v in a.nested_alphas.split(",") if v.strip() and v.strip() in a.ALPHAS]
    a.R2_EXCLUDE = [v.strip() for v in a.learner_r2_exclude.split(",") if v.strip()]
    a.GPUS = [g.strip() for g in str(a.gpus).split(",") if g.strip() != ""]
    a.OUT = (ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir)
    a.PROC = (ROOT / a.data_dir) if not os.path.isabs(a.data_dir) else Path(a.data_dir)
    a.SHARDS = a.OUT / "shards"
    a.SUBMAP = Path(a.subregion_map) if os.path.isabs(a.subregion_map) else a.PROC / a.subregion_map
    return a


# ================================================================ 자료·대상·분할
class Data:
    """자료와 하위 지역 정의(프로세스마다 한 번 적재)."""

    def __init__(self, args):
        df = load_base(args.PROC)
        df["s"] = df.e5_sqrt_tdd.values.astype(float); df["y"] = df[TARGET].values.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["z"] = np.log(df.y.values) - np.log(df.s.values)
        sub_km, subs_km = make_subregions(df, args.k_sub)             # 항상 계산한다(가볍다). 대응표가 있으면 대조에만 쓴다
        path = getattr(args, "SUBMAP", None)
        if path is not None and Path(path).exists():
            df["sub"], self.subs = apply_subregion_map(df, path)
            self.sub_src = f"map:{Path(path).name}"
            self.sub_kmeans_diff = int((np.asarray(df["sub"].values, object) != sub_km).sum())
            if self.sub_kmeans_diff:
                print(f"  [warn] 이 환경의 k-means 하위 지역이 대응표와 {self.sub_kmeans_diff}셀 다르다. 대응표를 쓴다", flush=True)
        else:
            df["sub"], self.subs = sub_km, subs_km
            self.sub_src = "kmeans"; self.sub_kmeans_diff = 0
        self.df = df; self.args = args
        self.macros = set(df.macro.unique()); self.sub_parent = dict(zip(self.subs.subregion, self.subs.parent))
        self._src: dict = {}; self._split: dict = {}

    def target_idx(self, t):
        if t in self.macros:
            return np.where(self.df.macro.values == t)[0]
        if t in self.sub_parent:
            return np.where(self.df["sub"].values == t)[0]
        raise ValueError(f"알 수 없는 대상 {t}")

    def parent_of(self, t):
        return t if t in self.macros else self.sub_parent[t]

    def source_idx(self, t, mode):
        """원천 셀 색인과 구성. 대상 셀 제외 → (모드 x, 하위 지역) 상위 macro 지역 제외 → 대상 경계 버퍼 제외(좌표만)."""
        if (t, mode) in self._src:
            return self._src[(t, mode)]
        df = self.df; t_idx = self.target_idx(t); parent = self.parent_of(t)
        if mode not in valid_modes(t) and t in self.macros:
            raise ValueError(f"{t} 는 모드 x 만 쓴다")
        src = np.where(np.isfinite(df.y.values))[0]
        src = src[~np.isin(src, t_idx)]
        n_par_ex = 0
        if mode == "x" and t not in self.macros:
            m = df.macro.values[src] == parent
            n_par_ex = int(m.sum()); src = src[~m]
        n0 = len(src)
        if self.args.buffer_km > 0 and len(src):
            la, lo = df.lat.values, df.lon.values
            tl, tn = la[t_idx], lo[t_idx]; keep = np.ones(len(src), bool)
            for j, i in enumerate(src):
                if abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < self.args.buffer_km:
                    keep[j] = False
            src = src[keep]
        n_par = int((df.macro.values[src] == parent).sum())
        comp = dict(n_src=int(len(src)), n_src_parent=n_par, frac_src_parent=float(n_par / max(len(src), 1)),
                    n_parent_excluded=n_par_ex, n_buffer_excluded=int(n0 - len(src)))
        self._src[(t, mode)] = (t_idx, parent, src, comp)
        return self._src[(t, mode)]

    def split_structure(self, t):
        """분할별 구조(h25 split_structure 와 같은 규칙). dup_of = A 블록 집합이 같은 앞선 분할, mirror_of = A·B 가 바뀐 앞선 분할,
        nb_eval = 채점 블록 수, valid = 중복 아님 · 채점 블록 ≥ 2."""
        if t in self._split:
            return self._split[t]
        df = self.df; t_idx = self.target_idx(t); allb = frozenset(df.block.values[t_idx]); seen, info = [], {}
        for sp in self.args.SPLITS:
            A_idx, B_idx = half_split_blocks(df, t_idx, sp)
            a = frozenset(df.block.values[A_idx]); evB = B_idx[eval_mask(df.iloc[B_idx])]
            dup = next((q for q, aq in seen if aq == a), -1)
            mir = next((q for q, aq in seen if aq == allb - a), -1)
            nb = int(df.iloc[evB].block.nunique())
            info[sp] = dict(dup_of=int(dup), mirror_of=int(mir), nb_eval=nb, n_eval=int(len(evB)), n_A=int(len(A_idx)), nb_A=len(a),
                            valid=bool(dup < 0 and nb >= 2))
            seen.append((sp, a))
        nu = sum(v["dup_of"] < 0 for v in info.values()); nv = sum(v["valid"] for v in info.values())
        for v in info.values():
            v.update(n_unique_splits=int(nu), n_valid_splits=int(nv))
        self._split[t] = info
        return info


def make_subregions(df, k_sub):
    """h25 와 같은 정의: 블록 중심 좌표 k-means(라벨 미사용), 북→남 순으로 이름을 붙인다."""
    from sklearn.cluster import KMeans
    sub = np.array(["" for _ in range(len(df))], dtype=object)
    rows = []
    for spec in k_sub.split(","):
        reg, k = spec.split(":"); k = int(k)
        idx = np.where(df.macro.values == reg)[0]
        if len(idx) == 0:
            continue
        bt = df.iloc[idx].groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size")).reset_index()
        X = np.c_[bt.lat.values, bt.lon.values * np.cos(np.radians(bt.lat.values))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=np.sqrt(bt.n.values))
        lab = km.labels_
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])
        name_of = {c: f"{reg[:2].upper()}-{r + 1}" for r, c in enumerate(order)}
        b2s = dict(zip(bt.block.values, [name_of[c] for c in lab]))
        bl = df.block.values
        for i in idx:
            sub[i] = b2s[bl[i]]
        for c in range(k):
            m = lab == c
            rows.append(dict(subregion=name_of[c], parent=reg, n_blocks=int(m.sum()), n_cells=int(bt.n.values[m].sum()),
                             lat=float(bt.lat.values[m].mean()), lon=float(bt.lon.values[m].mean())))
    return sub, pd.DataFrame(rows)


def subregion_table(df, sub):
    """블록 → 하위 지역 대응표. 키는 (parent, block)이다(지역 경계의 블록은 두 지역에 걸칠 수 있다)."""
    m = np.asarray(sub, object) != ""
    d = pd.DataFrame(dict(parent=df.macro.values[m], block=df.block.values[m], subregion=np.asarray(sub, object)[m],
                          lat=df.lat.values[m], lon=df.lon.values[m]))
    g = d.groupby(["parent", "block"], as_index=False).agg(subregion=("subregion", "first"), n_sub=("subregion", "nunique"),
                                                          n_cells=("lat", "size"), lat=("lat", "mean"), lon=("lon", "mean"))
    assert (g.n_sub == 1).all(), "한 블록이 두 하위 지역에 걸쳐 있다"
    return g.drop(columns="n_sub").sort_values(["parent", "subregion", "block"]).reset_index(drop=True)


def apply_subregion_map(df, path):
    """대응표에서 하위 지역을 읽는다. 표에 없는 블록이 있으면 중단한다(자료와 표의 판이 다르다)."""
    mp = pd.read_csv(path, dtype=dict(parent=str, subregion=str))
    key = {(p, int(b)): s for p, b, s in zip(mp.parent, mp.block, mp.subregion)}
    sub = np.array(["" for _ in range(len(df))], dtype=object)
    mac, bl = df.macro.values, df.block.values
    miss = 0
    for i in np.where(np.isin(mac, sorted(set(mp.parent))))[0]:
        s = key.get((mac[i], int(bl[i])))
        if s is None:
            miss += 1
        else:
            sub[i] = s
    if miss:
        raise SystemExit(f"하위 지역 대응표 {path} 에 없는 블록의 셀이 {miss}개다. 자료 판과 대응표를 확인한다")
    t = subregion_table(df, sub)
    rows = [dict(subregion=s, parent=g.parent.iloc[0], n_blocks=int(len(g)), n_cells=int(g.n_cells.sum()), lat=float(g.lat.mean()),
                 lon=float(g.lon.mean())) for s, g in t.groupby("subregion", sort=False)]
    return sub, pd.DataFrame(rows)


_DATA = None


def get_data(args):
    global _DATA
    sig = (str(args.PROC), args.k_sub, args.buffer_km, tuple(args.SPLITS), str(getattr(args, "SUBMAP", "")))
    if _DATA is None or getattr(_DATA, "sig", None) != sig:
        _DATA = Data(args); _DATA.sig = sig
    return _DATA


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


class Ctx:
    """작업 단위(대상, 모드, 분할)의 자료. 시험에서는 합성 자료로 직접 만든다."""

    def __init__(self, target, mode, split, parent, X_src, y_src, s_src, macro_src, XA, yA, sA, blkA, XB, yB, sB, blkB, min_cells_prior=3,
                 meta=None):
        self.target, self.mode, self.split, self.parent = str(target), str(mode), int(split), str(parent)
        ok = np.isfinite(y_src) & np.isfinite(s_src)
        self.X_src = np.asarray(X_src, np.float32)[ok]; self.y_src = np.asarray(y_src, float)[ok]; self.s_src = np.asarray(s_src, float)[ok]
        self.macro_src = np.asarray(macro_src)[ok]
        self.XA = np.asarray(XA, np.float32); self.yA = np.asarray(yA, float); self.sA = np.asarray(sA, float); self.blkA = np.asarray(blkA)
        self.XB = np.asarray(XB, np.float32); self.yB = np.asarray(yB, float); self.sB = np.asarray(sB, float); self.blkB = np.asarray(blkB)
        self.E0 = ls_E(self.y_src, self.s_src)
        self.r0_src = self.y_src - self.E0 * self.s_src
        with np.errstate(invalid="ignore", divide="ignore"):
            self.zA = np.log(self.yA) - np.log(self.sA)
        try:
            self.prior = offset_mle_prior(pd.DataFrame(dict(macro=self.macro_src, y=self.y_src, s=self.s_src)), region_col="macro",
                                          min_cells=min_cells_prior, y_col="y", s_col="s", logE0=float(np.log(self.E0)))
        except ValueError:
            self.prior = None
        self.meta = dict(meta or {})


def build_ctx(D: Data, args, target, mode, split) -> Ctx:
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx(target, mode)
    src = df.iloc[src_idx]
    A_idx, B_idx = half_split_blocks(df, t_idx, split)
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    A, B = df.iloc[A_idx], df.iloc[evB]
    info = D.split_structure(target)[split]
    d = df.iloc[t_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = d.y.values / np.where(d.s.values > 0, d.s.values, np.nan)
    g = pd.DataFrame(dict(b=d.block.values, r=ratio)).groupby("b").r
    Eb, nb = g.mean(), g.size()
    Xt = d[FEATS].values.astype(float); Xs = src[FEATS].values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smd = float(np.nanmean(np.abs(np.nanmean(Xt, 0) - np.nanmean(Xs, 0)) / (np.nanstd(Xs, 0) + 1e-9))) if len(Xs) else np.nan
    E_A, E_B = ls_E(A.y.values, A.s.values), ls_E(B.y.values, B.s.values)
    meta = dict(n_A=int(len(A_idx)), nb_A=int(A.block.nunique()), n_eval=int(len(evB)), nb_eval=int(B.block.nunique()), **comp,
                n_cells=int(len(t_idx)), n_blocks=int(d.block.nunique()), smd_x25=smd, E_own=ls_E(d.y.values, d.s.values), E_A=E_A, E_B=E_B,
                E_block_cv=float(np.nanstd(Eb[nb >= 3]) / np.nanmean(Eb[nb >= 3])) if (nb >= 3).sum() >= 2 else np.nan,
                y_mean=float(np.nanmean(d.y.values)), y_sd=float(np.nanstd(d.y.values)),
                dup_of=info["dup_of"], mirror_of=info["mirror_of"], valid=info["valid"],
                n_unique_splits=info["n_unique_splits"], n_valid_splits=info["n_valid_splits"],
                subregion_src=getattr(D, "sub_src", ""), subregion_kmeans_diff=int(getattr(D, "sub_kmeans_diff", 0)))
    c = Ctx(target, mode, split, parent, src[FEATS].values, src.y.values, src.s.values, src.macro.values,
            A[FEATS].values, A.y.values, A.s.values, A.block.values, B[FEATS].values, B.y.values, B.s.values, B.block.values,
            min_cells_prior=args.min_cells_prior, meta=meta)
    c.meta.update(E0=c.E0, logE_ratio_own=float(np.log(meta["E_own"] / c.E0)) if meta["E_own"] > 0 and c.E0 > 0 else np.nan,
                  logE_ratio_AB=float(np.log(E_A / E_B)) if np.isfinite(E_A) and np.isfinite(E_B) and E_A > 0 and E_B > 0 else np.nan,
                  tau2=c.prior["tau2"] if c.prior else np.nan, sigma2=c.prior["sigma2"] if c.prior else np.nan)
    return c


# ================================================================ 추출
def draw_cells(target, mode, split, n, draw, nA):
    """셀 무작위 추출. seed = seed_of(대상, 모드, 분할, n, 추출 번호). 모든 방법·학습기·α·부분이 이 함수만 쓴다."""
    if n == 0:
        return np.zeros(0, int)
    if n < 0:
        return np.arange(nA)
    return np.sort(np.random.RandomState(seed_of(target, mode, split, n, draw)).choice(nA, n, replace=False))


def draw_blocks(target, mode, split, n, draw, blkA):
    """블록 분산 추출: A 의 모든 블록을 무작위 순서로 순환 배정(h31 all_blocks 와 같은 규칙)."""
    rng = np.random.RandomState(seed_of(target, mode, split, n, draw, "block"))
    ub = np.unique(blkA)
    order = [int(j) for j in rng.permutation(len(ub))]
    pools = {j: list(rng.permutation(np.where(blkA == ub[j])[0])) for j in range(len(ub))}
    sel, active = [], order
    while len(sel) < n and active:
        nxt = []
        for j in active:
            if len(sel) >= n:
                break
            if pools[j]:
                sel.append(int(pools[j].pop())); nxt.append(j)
        active = nxt
    return np.sort(np.array(sel[:n], int))


def cells_of(n_list, n_draws, nA):
    """(n, draw) 목록. n = 0 과 all(-1)은 추출이 하나뿐이다. 양수 n 은 |A| 미만만."""
    out = []
    for n in n_list:
        if n in (0, -1):
            out.append((n, 0))
        elif 0 < n < nA:
            out += [(n, d) for d in range(n_draws)]
    return out


# ================================================================ 학습기
def _prep_stats(Xtr):
    """표준화·중앙값 대체 통계(학습 행렬에서만)."""
    Xtr = np.asarray(Xtr, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Xtr, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isnan(Xtr), med, Xtr)
    return med, Xf.mean(0), Xf.std(0) + 1e-6


def _prep_apply(X, stats):
    med, mu, sd = stats
    X = np.asarray(X, float)
    return ((np.where(np.isnan(X), med, X) - mu) / sd).astype(np.float32)


def cb_fit(args, Xtr, ytr, seed, w=None, init=None, iters=None):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=int(iters or args.cb_iters), learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=int(seed),
                          verbose=0, allow_writing_files=False, thread_count=int(args.threads))
    m.fit(Xtr, ytr, sample_weight=w, init_model=init)
    return m


def mlp_fit(Xtr, ytr, seed, epochs, rep=None, bs=8192):
    """tab_models._fit_torch('mlp') 와 같은 절차(검증 10 %, 조기 종료)로 학습하되 망과 척도를 돌려준다. Xtr 은 표준화된 float32.
    rep(행별 정수 배수)이 있으면 검증 10 % 를 고유 행에서 먼저 떼고 학습 쪽 행만 rep 배 반복한다. 검증 행은 반복하지 않으므로
    같은 대상 행이 학습과 검증에 함께 들어가지 않는다. y 척도도 고유 행에서 구한다. rep = None 이면 tab_models 와 같은 결과다."""
    import torch
    from polar import tab_models as TM
    MLP, _, _ = TM._torch_mods()
    ytr = np.asarray(ytr, float)
    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    Xtr = np.asarray(Xtr, np.float32)
    torch.manual_seed(int(seed))
    rng = np.random.RandomState(int(seed)); va = rng.rand(len(Xtr)) < 0.1; tr = ~va
    ix = np.where(tr)[0]
    if rep is not None:
        ix = np.repeat(ix, np.maximum(np.asarray(rep, int)[ix], 1))
    if len(ix) % int(bs) == 1 and len(ix) > 1:                      # 마지막 배치가 1행이면 학습 모드 BatchNorm 이 예외를 낸다: 1행을 뺀다
        ix = ix[:-1]
    net = MLP(Xtr.shape[1]).to(TM._dev())
    net = TM._epochs_fit(net, Xtr[ix], yz[ix], Xtr[va], yz[va], int(epochs), bs=int(bs), lr=1e-3)
    net.eval()
    return dict(net=net, ymu=ymu, ysd=ysd)


def realmlp_fit_predict(args, Xtr, ytr, Xte, seed):
    """pytabkit RealMLP-TD. tab_models._fit_realmlp 와 같은 설정이고 n_epochs 만 --realmlp-epochs 로 명시한다(기본 256 = pytabkit 기본값)."""
    from pytabkit import RealMLP_TD_Regressor
    from polar import tab_models as TM
    m = RealMLP_TD_Regressor(random_state=int(seed), device=TM._dev(), n_threads=int(args.threads), n_epochs=int(args.realmlp_epochs))
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float)


def mlp_predict(h, X):
    import torch
    from polar import tab_models as TM
    X = np.asarray(X, np.float32)
    if len(X) == 0:
        return np.zeros(0)
    h["net"].eval()
    with torch.no_grad():
        p = np.concatenate([h["net"](torch.tensor(X[k:k + 65536]).to(TM._dev())).cpu().numpy() for k in range(0, len(X), 65536)])
    return p.astype(float) * h["ysd"] + h["ymu"]


def mlp_finetune(h, Xft, yft, seed, epochs, lr):
    """원천 학습 망을 복제해 낮은 학습률로 미세조정한다. BatchNorm·Dropout 은 평가 모드로 고정(소수 행에서 통계가 무너지지 않게)."""
    import copy
    import torch
    import torch.nn as nn
    from polar import tab_models as TM
    net = copy.deepcopy(h["net"]); net.eval()
    torch.manual_seed(int(seed) + 7)
    Xt = torch.tensor(np.asarray(Xft, np.float32)); yt = torch.tensor(((np.asarray(yft, float) - h["ymu"]) / h["ysd"]).astype(np.float32))
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=1e-5)
    lossf = nn.SmoothL1Loss(); bs = 8192
    for _ in range(int(epochs)):
        idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]; xb, yb = Xt[b].to(TM._dev()), yt[b].to(TM._dev())
            opt.zero_grad(); lossf(net(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    return dict(net=net, ymu=h["ymu"], ysd=h["ysd"])


class Fitter:
    """적합 실행·계수·기록. dry = True 이면 학습 없이 수만 센다(예측은 0)."""

    def __init__(self, args, dry=False):
        self.args, self.dry = args, dry
        self.n = Counter(); self.sec = Counter(); self.fail = Counter()
        self.nd = Counter(); self.secd = Counter(); self.rowsd = Counter(); self.estd = Counter()     # 키 = "축|학습기|방법"
        self.errors: list = []
        self.last_flag = ""                                            # 직전 적합의 표지: "" 정상, "fallback" 원천 모델 대체, "fail" 실패
        self.nrow_fn = None                                            # (축, 학습기, info) → 학습 행 수(시간 추정·기록용)

    def _trace(self, info, Xtr, ytr, w):
        if TRACE is not None:
            TRACE.append(dict(info, Xtr=np.array(Xtr, copy=True), ytr=np.array(ytr, copy=True), w=None if w is None else np.array(w, copy=True)))

    def _count(self, axis, learner, info):
        k = f"{axis}|{learner}|{(info or {}).get('method', '')}"
        self.n[axis] += 1; self.nd[k] += 1; self.last_flag = ""
        if self.nrow_fn is not None:
            nr = max(int(self.nrow_fn(axis, learner, info or {})), 1)
            self.rowsd[k] += nr
            b, ref, ex = BASE_FIT.get(learner, (1.0, 12000, 1.0))
            self.estd[k] += float(b * (nr / ref) ** ex)
        return k

    def _failed(self, k, axis, e, preds):
        """적합 1건의 실패를 기록하고 NaN 예측을 돌려준다(저장 단계에서 그 키만 제외된다)."""
        self.fail[axis] += 1; self.last_flag = "fail"
        if len(self.errors) < 20:
            self.errors.append(f"{k}: {repr(e)[:200]}")
        print(f"    [warn] 적합 실패({k}): {repr(e)[:160]}", flush=True)
        return [np.full(len(p), np.nan) for p in preds]

    def fit(self, learner, axis, build, seed, preds, info=None, init=None, iters=None, cont=False):
        """build() → (Xtr, ytr, w). preds = 예측할 행렬 목록. 반환 (모델 핸들, 예측 목록).
        CatBoost 는 w 를 sample_weight 로 준다. MLP 는 검증 분리 뒤 학습 쪽 행만 반복한다. 그 외 학습기는 w(정수 배수)를 행 반복으로 바꾼다.
        cont = True 는 이어 학습이다(init 이 없으면 실패로 기록). 적합 단위 예외는 잡아서 그 키만 실패로 남긴다."""
        k = self._count(axis, learner, info)
        if self.dry:
            return None, [np.zeros(len(p)) for p in preds]
        t0 = time.time()
        try:
            m, out = self._fit(learner, axis, build, seed, preds, info, init, iters, cont)
        except (ImportError, MemoryError):                               # 환경 문제: 조각 전체를 실패로 둔다(재개 때 다시 실행)
            raise
        except Exception as e:                                           # noqa: BLE001
            m, out = None, self._failed(k, axis, e, preds)
        dt = time.time() - t0
        self.sec[axis] += dt; self.secd[k] += dt
        return m, out

    def _fit(self, learner, axis, build, seed, preds, info, init, iters, cont):
        Xtr, ytr, w = build()
        self._trace(dict(info or {}, learner=learner, axis=axis, seed=seed), Xtr, ytr, w)
        if cont and init is None:
            raise RuntimeError("이어 학습의 출발점(원천 모델)이 없다")
        if not np.all(np.isfinite(np.asarray(ytr, float))):
            raise ValueError("학습 목표에 비유한 값이 있다(앞 단계 적합 실패)")
        if learner == "catboost_lo":
            try:
                m = cb_fit(self.args, Xtr, ytr, seed, w=w, init=init, iters=iters)
            except Exception as e:                                                                     # noqa: BLE001
                if init is None:
                    raise
                self.fail[axis] += 1; self.last_flag = "fallback"      # 이어 학습 실패(예: 목표가 모두 같음) → 원천 모델 그대로
                m = init
                print(f"    [warn] 이어 학습 실패, 원천 모델 사용: {repr(e)[:120]}", flush=True)
            th = int(self.args.threads)                                  # predict 의 기본 thread_count = -1 은 노드 전체 코어다
            return m, [np.asarray(m.predict(p, thread_count=th), float) if len(p) else np.zeros(0) for p in preds]
        rep = None if w is None else np.maximum(np.round(np.asarray(w)).astype(int), 1)
        if learner == "mlp":                                             # 표준화 통계는 고유 행에서, 반복은 검증 분리 뒤 학습 쪽에만
            stats = _prep_stats(Xtr)
            h = mlp_fit(_prep_apply(Xtr, stats), ytr, seed, self.args.epochs, rep=rep); h["stats"] = stats
            return h, [mlp_predict(h, _prep_apply(p, stats)) for p in preds]
        if rep is not None:                                              # 그 외 학습기: 가중 = 행 반복
            ix = np.repeat(np.arange(len(ytr)), rep); Xtr, ytr = Xtr[ix], np.asarray(ytr)[ix]
        stats = _prep_stats(Xtr)
        Xz = _prep_apply(Xtr, stats); Pz = [_prep_apply(p, stats) for p in preds]
        sizes = [len(p) for p in Pz]
        if sum(sizes) == 0:
            return None, [np.zeros(0) for _ in Pz]
        if learner == "realmlp":
            allp = realmlp_fit_predict(self.args, Xz, np.asarray(ytr, float), np.vstack(Pz), seed)
        else:
            from polar.m1_core import fit_model
            allp = np.asarray(fit_model(learner, Xz, np.asarray(ytr, float), np.vstack(Pz), seed=int(seed), epochs=int(self.args.epochs))["pred"],
                              float)
        out, j = [], 0
        for s_ in sizes:
            out.append(allp[j:j + s_]); j += s_
        return None, out

    def mlp_cont(self, axis, base, build, seed, preds, info=None):
        """MLP 이어 학습: build() → (Xft, yft). base = 원천 학습 핸들(stats 포함)."""
        k = self._count(axis, "mlp", info)
        if self.dry:
            return [np.zeros(len(p)) for p in preds]
        t0 = time.time()
        try:
            Xft, yft, _ = build()
            self._trace(dict(info or {}, learner="mlp", axis=axis, seed=seed), Xft, yft, None)
            if base is None:
                raise RuntimeError("이어 학습의 출발점(원천 모델)이 없다")
            h = mlp_finetune(base, _prep_apply(Xft, base["stats"]), yft, seed, self.args.ft_epochs, self.args.ft_lr)
            out = [mlp_predict(h, _prep_apply(p, base["stats"])) for p in preds]
        except (ImportError, MemoryError):
            raise
        except Exception as e:                                           # noqa: BLE001
            out = self._failed(k, axis, e, preds)
        dt = time.time() - t0
        self.sec[axis] += dt; self.secd[k] += dt
        return out


def v1_fit(X, y, s):
    """공변량 의존 계수: z = log(y/s) 를 표준화 공변량 ridge(α = 10)로 적합. 반환 = 앵커 예측 함수 f(X, s) = exp(ẑ)·s."""
    from sklearn.linear_model import Ridge
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.log(y) - np.log(s)
    ok = np.isfinite(z)
    stats = _prep_stats(X[ok])
    rd = Ridge(alpha=10.0).fit(_prep_apply(X[ok], stats), z[ok])
    lo, hi = float(z[ok].min()), float(z[ok].max())

    def f(Xn, sn):
        if len(Xn) == 0:
            return np.zeros(0)
        return np.exp(np.clip(rd.predict(_prep_apply(Xn, stats)), lo, hi)) * np.asarray(sn, float)
    return f


# ================================================================ 작업 단위 실행
def run_ctx(c: Ctx, part, args, learner_axis=True, nested=False, dry=False, learners=None):
    """작업 단위 하나를 실행한다. 반환 (rows, BlockStore, 통계 dict). learners 를 주면 그 학습기만 돈다(gpu 부분의 학습기 단위 조각)."""
    F = Fitter(args, dry)
    if len(c.yB) == 0 or len(c.yA) == 0:
        return [], None, dict(n_fit={}, sec={}, fail={}, n_rows=0, status="no_eval", nested={}, n_fit_detail={}, sec_detail={}, rows_detail={},
                              est_detail={}, errors=[], n_stored=0, n_stored_ml=0, n_nonfinite_keys=0)
    st = BlockStore(f"{c.target}|{c.mode}", c.split, c.blkB, meta=dict(target=c.target, mode=c.mode, part=part))
    rows, n_rows, n_bad = [], [0], [0]
    nA, nsrc = len(c.yA), len(c.y_src)
    M, L = set(args.METHODS), list(learners if learners is not None else args.LEARNERS)
    R2_EX = set(getattr(args, "R2_EXCLUDE", []))
    E0 = c.E0
    base = dict(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part=part)

    def add(axis, method, learner, alpha, placement, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, alpha_sel="", nb_lab=0, flag=""):
        n_rows[0] += 1
        if dry:
            return
        pred = np.asarray(pred, float)
        nf = int((~np.isfinite(pred)).sum())
        row = dict(**base, axis=axis, method=method, learner=learner, alpha=str(alpha), placement=placement, n=int(n), n_lab=int(n_lab),
                   draw=int(d), seed=int(seed), lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan, E_used=float(E_used),
                   alpha_sel=str(alpha_sel), n_blocks_lab=int(nb_lab), n_nonfinite=nf, fit_flag=str(flag) if nf == 0 else (str(flag) or "nonfinite"))
        if nf > 0:                                                       # 비유한 예측이 있는 키는 저장하지 않는다(방법 간 채점 셀 집합 통일)
            n_bad[0] += 1
            rows.append(row)
            return
        key = (method, learner, str(alpha), placement, int(n), int(d), int(seed), float(lam))
        st.add(key, c.yB, pred)
        sse, cnt = st.get(key)
        with np.errstate(invalid="ignore", divide="ignore"):
            beq = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan))) if cnt.sum() else np.nan
            rm = float(np.sqrt(sse.sum() / cnt.sum())) if cnt.sum() else np.nan
        row.update(rmse_cm=rm, rmse_beq_cm=beq, bias_cm=float(np.nanmean(pred - c.yB)) if len(c.yB) else np.nan)
        rows.append(row)

    def coefs(sel):
        m_ = len(sel)
        if m_ == 0:
            return dict(E1=E0, E2=E0, E3=E0)
        E_ls = ls_E(c.yA[sel], c.sA[sel])
        if not np.isfinite(E_ls):
            return dict(E1=E0, E2=E0, E3=E0)
        E3 = offset_mle_estimate(c.zA[sel], c.prior)["E"] if c.prior is not None else E0
        return dict(E1=(m_ * E_ls + args.kappa * E0) / (m_ + args.kappa), E2=E_ls, E3=float(E3))

    seen_analytic = set()

    def analytic(axis, placement, n, d, sel, co):
        if (placement, n, d) in seen_analytic:
            return
        seen_analytic.add((placement, n, d))
        nb = len(np.unique(c.blkA[sel])) if len(sel) else 0
        for m, E in (("P1", co["E1"]), ("P2", co["E2"]), ("P3", co["E3"])):
            if m in M:
                add(axis, m, "none", "1", placement, n, d, -1, 0.0, E * c.sB, E, len(sel), nb_lab=nb)

    add("method", "P0", "none", "1", "cell", 0, 0, -1, 0.0, E0 * c.sB, E0, 0)      # P0 은 모든 비교의 기준이므로 항상 저장한다

    n_ps = int(round(args.r * nsrc))
    _ps: dict = {}

    def nrow(axis, learner, info):
        """적합 1건의 학습 행 수(시간 추정·기록용). 비-CatBoost 의 α 가중은 행 반복이므로 그만큼 더한다."""
        nl_ = len(info.get("sel", ())); a_ = str(info.get("alpha", "1"))
        if axis in ("alpha_cont", "nested_cont"):
            return nl_
        if axis == "mlp_cont":
            return nl_ + min(nl_, nsrc)
        nr = nsrc + nl_ + (n_ps if info.get("method") in ("D1", "R2") else 0)
        if learner != BASE_LEARNER and a_ not in ("1", "cont", "nested"):
            nr += int((float(a_) - 1.0) * nl_)
        return nr

    F.nrow_fn = nrow

    def ps_of(seed):
        if seed not in _ps:
            _ps[seed] = np.random.RandomState(seed_of("lg-ps", c.target, c.mode, c.split, seed)).choice(nA, n_ps, replace=True)
        return _ps[seed]

    def pseudo_rows(sel, E_n, seed, resid):
        """대상 A 풀 유사라벨 행. resid = False: y = E_n·s, True: 잔차 0. 라벨 셀은 실측(또는 실측 잔차)으로 바꾼다."""
        ps = ps_of(seed)
        yps = np.zeros(len(ps)) if resid else E_n * c.sA[ps]
        if len(sel):
            m = np.isin(ps, sel)
            yps[m] = (c.yA[ps[m]] - E_n * c.sA[ps[m]]) if resid else c.yA[ps[m]]
        return c.XA[ps], yps

    def train_rows(kind, sel, anchor_sel, seed=None, alpha=None, pseudo=False, E_n=None, src_target=None):
        """학습 행렬. kind = 'D'(y 직접) 또는 'R'(잔차). anchor_sel = 대상 선택 셀의 앵커 값(kind R). src_target = 원천 목표(기본 y 또는 E0 잔차)."""
        ys = src_target if src_target is not None else (c.y_src if kind == "D" else c.r0_src)
        Xs = [c.X_src]; Ys = [ys]; W = [np.ones(len(ys))]
        if len(sel):
            yt = c.yA[sel] if kind == "D" else c.yA[sel] - anchor_sel
            Xs.append(c.XA[sel]); Ys.append(yt); W.append(np.full(len(sel), float(alpha) if alpha not in (None, "1", "cont") else 1.0))
        if pseudo:
            Xp, yp = pseudo_rows(sel, E_n, seed, resid=(kind == "R"))
            Xs.append(Xp); Ys.append(yp); W.append(np.ones(len(yp)))
        w = np.concatenate(W)
        return np.vstack(Xs), np.concatenate(Ys), (w if alpha not in (None, "1", "cont") else None)

    def emit_resid(axis, method, learner, alpha, placement, n, d, seed, anchorB, g, E_used, n_lab, alpha_sel="", nb_lab=0, flag=""):
        for lam in args.LAMS:
            add(axis, method, learner, alpha, placement, n, d, seed, lam, anchorB + lam * g, E_used, n_lab, alpha_sel, nb_lab, flag)

    # ---------------------------------------------------------------- 원천만으로 학습한 모델(n = 0, 이어 학습의 출발점)
    _base: dict = {}

    def base_model(learner, kind, seed, axis):
        k = (learner, kind, seed)
        if k not in _base:
            _base[k] = F.fit(learner, axis, lambda: train_rows(kind, np.zeros(0, int), None), seed, [c.XB],
                             info=dict(method="D0" if kind == "D" else "R0", n=0, draw=0, alpha="1", placement="cell", sel=np.zeros(0, int)))
        return _base[k]

    # ---------------------------------------------------------------- α 중첩 선택(A 블록 단위 교차검증, B 미사용)
    def nested_select(method, n, d, sel):
        blk = c.blkA[sel]; ub = np.unique(blk)
        cand = list(args.NESTED_ALPHAS)
        if not cand:
            return {lam: "1" for lam in args.LAMS}, "no candidates"
        if (0 < n < args.nested_min_n) or len(ub) < 2 or len(sel) < args.nested_min_n:
            return {lam: "1" for lam in args.LAMS}, "n<min or 1 block"
        if args.nested_max_folds > 0 and len(ub) > args.nested_max_folds:
            rng = np.random.RandomState(seed_of("lg-cv", c.target, c.mode, c.split, n, d))
            fold_of = dict(zip(ub[rng.permutation(len(ub))], np.arange(len(ub)) % args.nested_max_folds))
        else:
            fold_of = {b: j for j, b in enumerate(ub)}
        fid = np.array([fold_of[b] for b in blk]); seed0 = args.SEEDS[0]
        err = {(a, lam): 0.0 for a in cand for lam in args.LAMS}
        for j in np.unique(fid):
            te, tr = sel[fid == j], sel[fid != j]
            E_tr = E0 if method == "R0" else coefs(tr)["E1"]
            a_tr, a_te = E_tr * c.sA[tr], E_tr * c.sA[te]
            for a in cand:
                info = dict(method=method, n=n, draw=d, alpha=a, placement="cell", sel=tr, cv_fold=int(j))
                if a == "cont":
                    b0, _ = base_model(BASE_LEARNER, "R", seed0, "alpha_base")
                    _, (g,) = F.fit(BASE_LEARNER, "nested_cont", lambda: (c.XA[tr], c.yA[tr] - a_tr, None), seed0, [c.XA[te]], info=info,
                                    init=b0, iters=args.cont_trees, cont=True)
                else:
                    _, (g,) = F.fit(BASE_LEARNER, "nested_cv", lambda: train_rows("R", tr, a_tr, alpha=a), seed0, [c.XA[te]], info=info)
                for lam in args.LAMS:
                    err[(a, lam)] += float(np.sum((a_te + lam * g - c.yA[te]) ** 2))
        order = [a for a in ALPHA_ORDER if a in cand] + [a for a in cand if a not in ALPHA_ORDER]
        bad = [a for a in cand if not all(np.isfinite(err[(a, lam)]) for lam in args.LAMS)]      # 교차검증 적합이 실패한 후보는 고르지 않는다

        def score(a, lam):
            return (err[(a, lam)] if np.isfinite(err[(a, lam)]) else np.inf, order.index(a))
        if len(bad) == len(cand):
            return {lam: "1" for lam in args.LAMS}, "cv failed"
        return {lam: min(order, key=lambda a: score(a, lam)) for lam in args.LAMS}, ("cv partly failed" if bad else "")

    # ================================================================ cpu 부분: CatBoost 방법 축·α 축·배치 축
    def cb_cell(n, d, sel, co, placement, methods, axis, do_alpha, do_nested):
        nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
        E1 = co["E1"]; a0_sel, a1_sel = E0 * c.sA[sel], E1 * c.sA[sel]
        lr = BASE_LEARNER
        info = dict(n=n, draw=d, alpha="1", placement=placement, sel=sel)
        v1 = None
        if ("V1" in methods or "V1r" in methods):
            if not dry:
                v1 = v1_fit(np.vstack([c.X_src, c.XA[sel]]), np.concatenate([c.y_src, c.yA[sel]]), np.concatenate([c.s_src, c.sA[sel]]))
            F.n["v1_ridge"] += 1
            vB = v1(c.XB, c.sB) if v1 else np.zeros(len(c.yB))
            if "V1" in methods:
                add(axis, "V1", "none", "1", placement, n, d, -1, 0.0, vB, np.nan, nl, nb_lab=nb)
        for seed in args.SEEDS:
            gB = {}                                                       # (방법, α) → 채점 셀의 g 또는 직접 예측
            fl = {}                                                       # (방법, α) → 적합 표지(이어 학습의 원천 모델 대체 등)
            if "D0" in methods:
                if nl == 0:
                    _, (p,) = base_model(lr, "D", seed, axis)
                else:
                    _, (p,) = F.fit(lr, axis, lambda: train_rows("D", sel, None), seed, [c.XB], info=dict(info, method="D0"))
                gB[("D0", "1")] = p
                add(axis, "D0", lr, "1", placement, n, d, seed, 1.0, p, np.nan, nl, nb_lab=nb)
            pD1 = None
            if "D1" in methods or "R3" in methods:
                _, pD1 = F.fit(lr, axis, lambda: train_rows("D", sel, None, seed=seed, pseudo=True, E_n=E1), seed,
                               [c.XB, c.X_src, c.XA[sel]], info=dict(info, method="D1"))
                if "D1" in methods:
                    add(axis, "D1", lr, "1", placement, n, d, seed, 1.0, pD1[0], E1, nl, nb_lab=nb)
            if "R0" in methods or ("R1" in methods and nl == 0):
                if nl == 0:
                    _, (g,) = base_model(lr, "R", seed, axis)
                else:
                    _, (g,) = F.fit(lr, axis, lambda: train_rows("R", sel, a0_sel), seed, [c.XB], info=dict(info, method="R0"))
                gB[("R0", "1")] = g
                if "R0" in methods:
                    emit_resid(axis, "R0", lr, "1", placement, n, d, seed, E0 * c.sB, g, E0, nl, nb_lab=nb)
            if "R1" in methods:
                if nl == 0:
                    g = gB[("R0", "1")]                                   # n = 0 에서 R1 은 R0 과 같다(E_n = E0, 원천 잔차만)
                else:
                    _, (g,) = F.fit(lr, axis, lambda: train_rows("R", sel, a1_sel), seed, [c.XB], info=dict(info, method="R1"))
                gB[("R1", "1")] = g
                emit_resid(axis, "R1", lr, "1", placement, n, d, seed, E1 * c.sB, g, E1, nl, nb_lab=nb)
            if "R2" in methods:
                _, (g,) = F.fit(lr, axis, lambda: train_rows("R", sel, a1_sel, seed=seed, pseudo=True, E_n=E1), seed, [c.XB],
                                info=dict(info, method="R2"))
                emit_resid(axis, "R2", lr, "1", placement, n, d, seed, E1 * c.sB, g, E1, nl, nb_lab=nb)
            if "R3" in methods:
                _, (g,) = F.fit(lr, axis, lambda: train_rows("R", sel, pD1[2], src_target=c.y_src - pD1[1]), seed, [c.XB],
                                info=dict(info, method="R3"))
                emit_resid(axis, "R3", lr, "1", placement, n, d, seed, pD1[0], g, E1, nl, nb_lab=nb)
            if "V1r" in methods:
                vs, va = (v1(c.X_src, c.s_src), v1(c.XA[sel], c.sA[sel])) if v1 else (np.zeros(nsrc), np.zeros(nl))
                _, (g,) = F.fit(lr, axis, lambda: train_rows("R", sel, va, src_target=c.y_src - vs), seed, [c.XB],
                                info=dict(info, method="V1r"))
                emit_resid(axis, "V1r", lr, "1", placement, n, d, seed, vB, g, np.nan, nl, nb_lab=nb)
            if do_alpha and nl > 0:
                for m in [m for m in ALPHA_METHODS if m in methods]:
                    anc = None if m == "D0" else (a0_sel if m == "R0" else a1_sel)
                    kind = "D" if m == "D0" else "R"
                    for a in [a for a in args.ALPHAS if a != "1"]:
                        inf = dict(info, method=m, alpha=a)
                        if a == "cont":
                            b0, _ = base_model(lr, kind, seed, "alpha_base")
                            yt = c.yA[sel] if kind == "D" else c.yA[sel] - anc
                            _, (g,) = F.fit(lr, "alpha_cont", lambda: (c.XA[sel], yt, None), seed, [c.XB], info=inf, init=b0, iters=args.cont_trees,
                                            cont=True)
                        else:
                            _, (g,) = F.fit(lr, "alpha", lambda: train_rows(kind, sel, anc, alpha=a), seed, [c.XB], info=inf)
                        gB[(m, a)] = g; fl[(m, a)] = F.last_flag
                        if m == "D0":
                            add("alpha", m, lr, a, placement, n, d, seed, 1.0, g, np.nan, nl, nb_lab=nb, flag=fl[(m, a)])
                        else:
                            emit_resid("alpha", m, lr, a, placement, n, d, seed, (E0 if m == "R0" else E1) * c.sB, g, E0 if m == "R0" else E1, nl,
                                       nb_lab=nb, flag=fl[(m, a)])
                if do_nested:
                    for m in [m for m in args.NESTED_METHODS if m in methods and m in ("R0", "R1")]:
                        if seed == args.SEEDS[0]:
                            _nest[(m, n, d)] = nested_select(m, n, d, sel)
                        choice, _flag = _nest[(m, n, d)]
                        for lam in args.LAMS:
                            a = choice[lam] if (m, choice[lam]) in gB else "1"
                            Eu = E0 if m == "R0" else E1
                            add("nested", m, lr, "nested", placement, n, d, seed, lam, Eu * c.sB + lam * gB[(m, a)], Eu, nl, alpha_sel=a, nb_lab=nb,
                                flag=fl.get((m, a), ""))

    _nest: dict = {}

    def learner_cell(learner, n, d, sel, co, axis="learner"):
        """학습기 축: D0·R1·R2 를 학습기 하나로."""
        nl = len(sel); nb = len(np.unique(c.blkA[sel])) if nl else 0
        E1 = co["E1"]; a1_sel = E1 * c.sA[sel]
        info = dict(n=n, draw=d, alpha="1", placement="cell", sel=sel)
        out = {}
        for seed in args.SEEDS:
            if "D0" in M:
                if nl == 0 and learner == "mlp":
                    _, (p,) = base_model(learner, "D", seed, axis)
                else:
                    _, (p,) = F.fit(learner, axis, lambda: train_rows("D", sel, None), seed, [c.XB], info=dict(info, method="D0"))
                add(axis, "D0", learner, "1", "cell", n, d, seed, 1.0, p, np.nan, nl, nb_lab=nb)
            if "R1" in M:
                if nl == 0 and learner == "mlp":
                    _, (g,) = base_model(learner, "R", seed, axis)
                else:
                    _, (g,) = F.fit(learner, axis, lambda: train_rows("R", sel, a1_sel), seed, [c.XB], info=dict(info, method="R1"))
                out[seed] = g
                emit_resid(axis, "R1", learner, "1", "cell", n, d, seed, E1 * c.sB, g, E1, nl, nb_lab=nb)
            if nl == 0 and learner == "mlp" and part == "gpu" and "R0" in M and len(args.ALPHAS) > 0:
                # MLP α 축 R0 의 n = 0 기준 행(라벨 순가치의 기준). R1 n = 0 과 같은 적합을 쓰므로 추가 학습은 없다
                _, (g0,) = base_model("mlp", "R", seed, axis if "R1" in M else "mlp_base")
                emit_resid("mlp_alpha", "R0", "mlp", "1", "cell", n, d, seed, E0 * c.sB, g0, E0, 0, nb_lab=0)
            if "R2" in M and learner not in R2_EX:
                _, (g,) = F.fit(learner, axis, lambda: train_rows("R", sel, a1_sel, seed=seed, pseudo=True, E_n=E1), seed, [c.XB],
                                info=dict(info, method="R2"))
                emit_resid(axis, "R2", learner, "1", "cell", n, d, seed, E1 * c.sB, g, E1, nl, nb_lab=nb)
        return out

    def mlp_alpha_cell(n, d, sel, co, r1_alpha1):
        """MLP α 축: D0·R0·R1 × α. D0·R1 의 α = 1 은 학습기 축 값과 같으므로 다시 적합하지 않는다."""
        nl = len(sel); nb = len(np.unique(c.blkA[sel]))
        E1 = co["E1"]; a0_sel, a1_sel = E0 * c.sA[sel], E1 * c.sA[sel]
        info = dict(n=n, draw=d, placement="cell", sel=sel)
        for seed in args.SEEDS:
            for m in [m for m in ALPHA_METHODS if m in M]:
                kind = "D" if m == "D0" else "R"
                anc = None if m == "D0" else (a0_sel if m == "R0" else a1_sel)
                for a in args.ALPHAS:
                    if a == "1" and m != "R0":
                        continue
                    inf = dict(info, method=m, alpha=a)
                    if a == "cont":
                        b0, _ = base_model("mlp", kind, seed, "mlp_base")
                        rp = np.random.RandomState(seed_of("lg-replay", c.target, c.mode, c.split, n, d, seed)).choice(nsrc, min(nl, nsrc), replace=False)
                        ys = c.y_src if kind == "D" else c.r0_src
                        yt = c.yA[sel] if kind == "D" else c.yA[sel] - anc
                        (g,) = F.mlp_cont("mlp_cont", b0, lambda: (np.vstack([c.XA[sel], c.X_src[rp]]), np.concatenate([yt, ys[rp]]), None),
                                          seed, [c.XB], info=inf)
                    else:
                        _, (g,) = F.fit("mlp", "mlp_alpha", lambda: train_rows(kind, sel, anc, alpha=a), seed, [c.XB], info=inf)
                    fg = F.last_flag
                    if m == "D0":
                        add("mlp_alpha", m, "mlp", a, "cell", n, d, seed, 1.0, g, np.nan, nl, nb_lab=nb, flag=fg)
                    else:
                        Eu = E0 if m == "R0" else E1
                        emit_resid("mlp_alpha", m, "mlp", a, "cell", n, d, seed, Eu * c.sB, g, Eu, nl, nb_lab=nb, flag=fg)

    if part == "cpu":
        if BASE_LEARNER in L:
            for n, d in cells_of(args.N_GRID, args.DRAWS, nA):
                sel = draw_cells(c.target, c.mode, c.split, n, d, nA); co = coefs(sel)
                analytic("method", "cell", n, d, sel, co)
                cb_cell(n, d, sel, co, "cell", M, "method", do_alpha=True, do_nested=nested and (n == -1 or n >= args.nested_min_n))
            for n, d in cells_of([n for n in args.PLACE_N if n > 0 and n in args.N_GRID], args.DRAWS, nA):
                sel = draw_blocks(c.target, c.mode, c.split, n, d, c.blkA); co = coefs(sel)
                if "P1" in M:
                    add("place", "P1", "none", "1", "block", n, d, -1, 0.0, co["E1"] * c.sB, co["E1"], len(sel), nb_lab=len(np.unique(c.blkA[sel])))
                cb_cell(n, d, sel, co, "block", {"R1"} & M, "place", do_alpha=False, do_nested=False)
        if "ridge" in L and learner_axis:
            for n, d in cells_of([n for n in args.LEARNER_N if n in args.N_GRID], min(args.learner_draws, args.DRAWS), nA):
                sel = draw_cells(c.target, c.mode, c.split, n, d, nA); co = coefs(sel)
                analytic("learner", "cell", n, d, sel, co)
                learner_cell("ridge", n, d, sel, co)
    else:
        for learner in L:
            for n, d in cells_of([n for n in args.LEARNER_N if n in args.N_GRID], min(args.learner_draws, args.DRAWS), nA):
                sel = draw_cells(c.target, c.mode, c.split, n, d, nA); co = coefs(sel)
                analytic("learner", "cell", n, d, sel, co)
                g1 = learner_cell(learner, n, d, sel, co)
                if learner == "mlp" and len(sel) >= 3 and len([a for a in args.ALPHAS]) > 0:
                    mlp_alpha_cell(n, d, sel, co, g1)
    n_ml = sum(1 for k in st.keys if k[1] != "none")
    n_fail = int(sum(F.fail.values())) + int(n_bad[0])
    status = "ok" if n_fail == 0 else ("failed" if n_ml == 0 else "partial")      # failed = 학습 결과가 하나도 저장되지 않음
    stats = dict(n_fit=dict(F.n), sec={k: round(v, 1) for k, v in F.sec.items()}, fail=dict(F.fail), n_rows=int(n_rows[0]), status=status,
                 nested={f"{m}|{n}|{d}": dict(choice={str(k): v for k, v in ch.items()}, flag=fl) for (m, n, d), (ch, fl) in _nest.items()},
                 n_fit_detail=dict(F.nd), sec_detail={k: round(v, 2) for k, v in F.secd.items()}, rows_detail=dict(F.rowsd),
                 est_detail={k: round(v, 2) for k, v in F.estd.items()}, errors=list(F.errors), n_stored=int(len(st)), n_stored_ml=int(n_ml),
                 n_nonfinite_keys=int(n_bad[0]))
    return rows, st, stats


# ================================================================ 조각 입출력
def shard_base(args, part, target, mode, split, learner=None):
    """조각 이름. gpu 부분은 학습기 단위 조각이라 이름 끝에 학습기가 붙는다."""
    return args.SHARDS / (f"{args.TAG}__{part}__{target}__{mode}__s{split}" + (f"__{learner}" if learner else ""))


def shard_paths(args, part, target, mode, split, learner=None):
    b = shard_base(args, part, target, mode, split, learner)
    return dict(runs=Path(str(b) + "_runs.csv"), npz=Path(str(b) + "_blocksse.npz"), unit=Path(str(b) + "_unit.json"))


CFG_UNIT_KEYS = ("learners", "learner_axis", "nested")               # 조각마다 달라도 되는 항목(공통 해시에서 뺀다)


def unit_cfg(args, part, target, mode, split, learner=None):
    """결과에 영향을 주는 설정 요약. 대상·분할·워커·GPU 목록은 넣지 않는다(조각의 정체 또는 실행 자원)."""
    d = dict(part=part, learners=[learner] if learner else list(args.LEARNERS), learner_axis=bool(is_learner_unit(args, target, mode, split)),
             n_grid=list(args.N_GRID), draws=int(args.DRAWS), seeds=list(args.SEEDS), methods=list(args.METHODS), alphas=list(args.ALPHAS),
             lams=list(args.LAMS), kappa=float(args.kappa), r=float(args.r), buffer_km=float(args.buffer_km), k_sub=str(args.k_sub),
             min_cells_prior=int(args.min_cells_prior), learner_n=list(args.LEARNER_N), learner_draws=int(min(args.learner_draws, args.DRAWS)),
             subregion_map=Path(args.SUBMAP).name if Path(args.SUBMAP).exists() else "kmeans")
    if part == "cpu":
        d.update(nested=bool((target, mode) in args.NESTED_TARGETS), cb_iters=int(args.cb_iters), cont_trees=int(args.cont_trees),
                 place_n=list(args.PLACE_N), nested_alphas=list(args.NESTED_ALPHAS), nested_methods=list(args.NESTED_METHODS),
                 nested_min_n=int(args.nested_min_n), nested_max_folds=int(args.nested_max_folds))
    else:
        d.update(nested=False, epochs=int(args.epochs), realmlp_epochs=int(args.realmlp_epochs), ft_epochs=int(args.ft_epochs),
                 ft_lr=float(args.ft_lr), r2_exclude=sorted(args.R2_EXCLUDE))
    return d


def cfg_hash(cfg, common=False):
    d = {k: v for k, v in cfg.items() if not (common and k in CFG_UNIT_KEYS)}
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def code_sha():
    try:
        return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    except OSError:
        return "NA"


def unit_state(args, part, target, mode, split, learner=None):
    """(완료 여부, 사유). 완료 = 조각 세 파일이 있고, 설정 해시가 같고, status 가 failed 가 아니다."""
    p = shard_paths(args, part, target, mode, split, learner)
    if not (p["unit"].exists() and p["runs"].exists() and p["npz"].exists()):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != cfg_hash(unit_cfg(args, part, target, mode, split, learner)):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    return True, str(u.get("status", "ok"))


def unit_done(args, part, target, mode, split, learner=None):
    return unit_state(args, part, target, mode, split, learner)[0]


def _atomic_text(path: Path, text: str):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text); os.replace(tmp, path)


def write_shard(args, part, c: Ctx, rows, st, stats, elapsed, learner=None):
    p = shard_paths(args, part, c.target, c.mode, c.split, learner)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    tmp = p["runs"].with_name(p["runs"].name + f".tmp{os.getpid()}")
    pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, p["runs"])
    save_stores([st], p["npz"])
    cfg = unit_cfg(args, part, c.target, c.mode, c.split, learner)
    unit = dict(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part=part, learner=learner or "", tag=args.TAG,
                elapsed_s=round(elapsed, 1), n_fit_total=int(sum(stats["n_fit"].values())), **stats, **c.meta,
                cfg=cfg, cfg_hash=cfg_hash(cfg), cfg_common=cfg_hash(cfg, common=True), code_sha=code_sha(), threads=int(args.threads),
                device=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    _atomic_text(p["unit"], json.dumps(unit, ensure_ascii=False, indent=1, default=float))      # 완료 표지는 마지막에 쓴다
    return unit


def is_learner_unit(args, target, mode, split):
    return (target, mode) in args.LEARNER_TARGETS and split in args.LEARNER_SPLITS


def enumerate_units(args, D: Data, part):
    """작업 단위 목록 (대상, 모드, 분할, 학습기)과 건너뛴 분할의 기록. cpu 부분의 학습기는 None(조각 하나에 CatBoost·ridge).
    중복 분할은 실행하지 않는다. 유효 분할이 있는 대상의 무효 분할(채점 블록 < 2)도 실행하지 않는다(집계에 쓰이지 않는다).
    gpu 부분은 학습기 축 범위만, 학습기 단위로 나눈다."""
    units, skipped = [], []
    for t, m in args.TARGETS:
        if m not in valid_modes(t):
            raise SystemExit(f"대상 {t} 에 모드 {m} 는 없다")
        info = D.split_structure(t)
        any_valid = any(v["valid"] for v in info.values())
        for sp in args.SPLITS:
            v = info[sp]
            if part == "gpu" and not is_learner_unit(args, t, m, sp):
                continue
            if v["dup_of"] >= 0:
                skipped.append(dict(target=t, mode=m, split=sp, part=part, status=f"dup_of_{v['dup_of']}", **v)); continue
            if v["n_eval"] == 0 or v["n_A"] == 0:
                skipped.append(dict(target=t, mode=m, split=sp, part=part, status="no_eval", **v)); continue
            if not v["valid"] and any_valid:
                skipped.append(dict(target=t, mode=m, split=sp, part=part, status="invalid_nb_eval<2", **v)); continue
            units += [(t, m, sp, lr) for lr in (args.LEARNERS if part == "gpu" else [None])]
    return units, skipped


def unit_priority(args, D: Data, u):
    """실행 순서 키. 무효 분할은 맨 뒤, gpu 부분은 빠른 학습기 먼저, 그 안에서 우선순위 묶음 → 분할 번호 → 큰 대상 순이다.
    묶음: 주 4지역 → Alaska x → 알래스카 하위 i → 알래스카 하위 x → 나머지(Russia_C·Greenland·기타 하위)."""
    t, m, sp, lr = u
    v = D.split_structure(t)[sp]
    g = 0 if t in MAIN4 else 1 if t == ALASKA else (2 if m == "i" else 3) if t in SUB_AL else 4
    return (0 if v["valid"] else 1, LEARNER_TIER.get(lr, 0) if lr else 0, g, int(sp), -int(v["n_A"]),
            GPU_LEARNERS.index(lr) if lr in GPU_LEARNERS else 0)


def unit_name(u):
    return f"{u[0]}|{u[1]}|s{u[2]}" + (f"|{u[3]}" if len(u) > 3 and u[3] else "")


def run_unit(args, part, target, mode, split, learner=None, dry=False):
    t0 = time.time()
    D = get_data(args)
    c = build_ctx(D, args, target, mode, split)
    if not dry:                                                          # 이전 세대의 완료 표지를 먼저 지운다(중단되면 미완료로 남는다)
        shard_paths(args, part, target, mode, split, learner)["unit"].unlink(missing_ok=True)
    rows, st, stats = run_ctx(c, part, args, learner_axis=is_learner_unit(args, target, mode, split),
                              nested=(target, mode) in args.NESTED_TARGETS, dry=dry, learners=[learner] if learner else None)
    if dry:
        return dict(target=target, mode=mode, split=split, part=part, learner=learner or "", n_A=c.meta["n_A"], n_eval=c.meta["n_eval"],
                    nb_eval=c.meta["nb_eval"], n_src=c.meta["n_src"], valid=c.meta["valid"], n_rows=stats["n_rows"],
                    est_s=round(float(sum(stats["est_detail"].values())), 1), **{f"fit_{k}": v for k, v in stats["n_fit"].items()},
                    _detail=stats["n_fit_detail"], _rows=stats["rows_detail"], _est=stats["est_detail"])
    if st is None:                                                       # 채점 셀 또는 A 셀이 없다(enumerate_units 에서 걸러지는 경우)
        raise RuntimeError(f"{target}|{mode}|s{split}: 채점 셀 또는 A 셀이 없다")
    return write_shard(args, part, c, rows, st, stats, time.time() - t0, learner)


# ---------------------------------------------------------------- 워커
_WARGS = None


def _worker_init(argv, gpu_queue, threads):
    global _WARGS
    warnings.filterwarnings("ignore")
    if gpu_queue is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get())          # 워커마다 GPU 하나(torch 를 부르기 전에 설정)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _WARGS = parse_args(argv)
    try:
        import torch
        torch.set_num_threads(int(threads))
    except Exception:                                                        # noqa: BLE001
        pass


def _worker_run(part, target, mode, split, learner=None):
    t0 = time.time()
    u = run_unit(_WARGS, part, target, mode, split, learner)
    u["wall_s"] = round(time.time() - t0, 1); u["device"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    u["n_fail"] = int(sum(u.get("fail", {}).values())) + int(u.get("n_nonfinite_keys", 0))
    return {k: u[k] for k in ("target", "mode", "split", "part", "learner", "n_A", "n_eval", "nb_eval", "n_src", "E0", "n_fit_total", "n_rows",
                              "elapsed_s", "wall_s", "device", "status", "valid", "n_fail")}


# ================================================================ 집계
class TMx:
    """대상·모드 하나의 저장소와 키 색인."""

    def __init__(self, name, by_split, info, nboot):
        self.name = name; self.target, self.mode = name.split("|")
        self.by_all = {sp: st for sp, st in by_split.items() if info.get(sp, {}).get("dup_of", -1) < 0}
        self.by_valid = {sp: st for sp, st in self.by_all.items() if info.get(sp, {}).get("valid", True)}
        self.point_only = self.target in MAIN_POINT                     # 계획서 §1: 러시아 C·그린란드는 점 추정만
        self.used = self.by_valid if len(self.by_valid) > 0 else self.by_all
        self.has_ci = len(self.by_valid) > 0 and not self.point_only
        self.nboot = nboot if self.has_ci else 0
        self.seed = seed_of("lgboot", name)
        self.idx = {}
        for sp, st in self.used.items():
            g = {}
            for k in st.keys:
                g.setdefault((k[0], k[1], str(k[2]), k[3], int(k[4]), float(k[7])), []).append(k)
            self.idx[sp] = g
        self.nb_union = len(set().union(*[set(st.blocks.tolist()) for st in self.used.values()])) if self.used else 0
        self.n_eval_mean = float(np.mean([st.ncell.sum() for st in self.used.values()])) if self.used else 0.0
        self.n_valid = len(self.by_valid); self.n_unique = len(self.by_all)

    def groups(self):
        out = {}
        for sp, g in self.idx.items():
            for k in g:
                out.setdefault(k, []).append(sp)
        return out


def _rep_fn(k):
    return k[KI["draw"]]


def contrast(tm: TMx, gA, gB, return_dist=False):
    """Δ = RMSE(A) − RMSE(B). gA·gB = (method, learner, alpha, placement, n, lam). 추출이 여럿인 두 쪽은 공통 추출 번호로 제한한다."""
    mini, kA_all, kB_all = {}, [], []
    for sp, st in tm.used.items():
        a, b = tm.idx[sp].get(gA), tm.idx[sp].get(gB)
        if not a or not b:
            continue
        da, db = {k[5] for k in a}, {k[5] for k in b}
        if len(da) > 1 and len(db) > 1:
            cm = da & db
            a = [k for k in a if k[5] in cm]; b = [k for k in b if k[5] in cm]
            if not a or not b:
                continue
        keys = list(dict.fromkeys(a + b))
        S, C = st.matrices(keys)
        mini[sp] = BlockStore._from_arrays(st.target, st.split, st.blocks, st.ncell, keys, S, C, {})
        kA_all += a; kB_all += b
    if not mini:
        return None
    kA_all, kB_all = list(dict.fromkeys(kA_all)), list(dict.fromkeys(kB_all))
    r = boot_delta_blocks(mini, kA_all, kB_all, nboot=tm.nboot, seed=tm.seed, rep_fn=_rep_fn, return_dist=return_dist)
    r["n_draws"] = len({k[5] for k in kA_all})
    return r


def _sig(lo, hi, lob, hib):
    if not all(np.isfinite([lo, hi, lob, hib])):
        return ""
    if hi < 0 and hib < 0:
        return "improve"
    if lo > 0 and lob > 0:
        return "worse"
    return "ns"


def _ref_grp(g, kind):
    m, lr, a, pl, n, lam = g
    if kind == "p0":
        return P0_GRP
    if kind == "p1":
        return ("P1", "none", "1", pl, n, 0.0) if n != 0 else P0_GRP
    if m in ("P1", "P2", "P3"):
        return P0_GRP                                                   # 물리식의 n = 0 은 P0(규약)
    return (m, lr, "1", "cell", 0, lam)                                   # 같은 방법·학습기·λ 의 n = 0(α·배치는 n = 0 에서 의미가 없다)


def build_curve(tms, runs):
    rows = []
    for tm in tms.values():
        flag0 = "" if tm.has_ci else ("point only(계획서 §1)" if tm.point_only else "no valid split; point only")
        for g, sps in tm.groups().items():
            d = dict(target=tm.target, mode=tm.mode, **dict(zip(GRP_COLS, g)), n_splits_valid=tm.n_valid, n_splits_unique=tm.n_unique,
                     point_only=bool(not tm.has_ci), n_blocks_union=int(tm.nb_union))
            for kind in ("p0", "p1", "n0"):
                r = contrast(tm, g, _ref_grp(g, kind))
                if r is None:
                    d.update({f"d_{kind}": np.nan, f"d_{kind}_lo": np.nan, f"d_{kind}_hi": np.nan, f"d_{kind}_beq": np.nan, f"d_{kind}_beq_lo": np.nan,
                              f"d_{kind}_beq_hi": np.nan, f"split_win_{kind}": np.nan, f"rep_win_{kind}": np.nan, f"sig_{kind}": ""})
                    continue
                d.update({f"d_{kind}": r["delta"], f"d_{kind}_lo": r["ci_lo"], f"d_{kind}_hi": r["ci_hi"], f"d_{kind}_beq": r["delta_beq"],
                          f"d_{kind}_beq_lo": r["ci_lo_beq"], f"d_{kind}_beq_hi": r["ci_hi_beq"], f"split_win_{kind}": r["split_win"],
                          f"rep_win_{kind}": r["rep_win"], f"sig_{kind}": _sig(r["ci_lo"], r["ci_hi"], r["ci_lo_beq"], r["ci_hi_beq"])})
                if kind == "p0":
                    d.update(rmse=r["rmse_A"], rmse_p0=r["rmse_B"], p_p0=r["p_boot"], n_splits_used=r["n_splits"],
                             n_blocks_eval_min=int(min(r["n_blocks"])) if r["n_blocks"] else 0, n_draws=r["n_draws"], n_keys=r["n_keys_A"],
                             ci_flag=";".join(v for v in (flag0, r["ci_flag"]) if v),
                             d_p0_splits=json.dumps({str(k): round(v, 4) for k, v in r["delta_split"].items()}))
                if kind == "p1":
                    d.update(rmse_p1=r["rmse_B"])
            rows.append(d)
    cur = pd.DataFrame(rows)
    if not len(cur):
        return cur
    used = {(tm.target, tm.mode, int(sp)) for tm in tms.values() for sp in tm.used}
    rr = runs[[(t, m, int(s)) in used for t, m, s in zip(runs.target, runs["mode"], runs.split)]]
    K = ["target", "mode"] + GRP_COLS
    g1 = rr.groupby(K + ["split"], as_index=False).agg(rmse_beq=("rmse_beq_cm", "mean"), bias=("bias_cm", "mean"), n_lab=("n_lab", "mean"),
                                                       E_used=("E_used", "mean"), n_bad_keys=("n_nonfinite", lambda v: int((v > 0).sum())),
                                                       n_fallback=("fit_flag", lambda v: int((v.astype(str) == "fallback").sum())))
    g2 = g1.groupby(K, as_index=False).agg(rmse_beq=("rmse_beq", "mean"), bias=("bias", "mean"), n_lab=("n_lab", "mean"), E_used=("E_used", "mean"),
                                           n_bad_keys=("n_bad_keys", "sum"), n_fallback=("n_fallback", "sum"))
    lab = rr.drop_duplicates(subset=K)[K + ["axis", "parent"]]                        # 'axis' 는 agg 의 예약 인자와 겹치므로 따로 붙인다
    g2 = g2.merge(lab, on=K, how="left")
    sel = rr[rr.alpha == "nested"]
    if len(sel):
        a = sel.groupby(K).alpha_sel.agg(lambda v: json.dumps(dict(Counter(v.astype(str))))).rename("alpha_sel_counts").reset_index()
        g2 = g2.merge(a, on=K, how="left")
    cur = cur.merge(g2, on=K, how="left")
    for col, tag in (("n_bad_keys", "nonfinite keys"), ("n_fallback", "fallback keys")):         # 제외·대체된 키가 있는 곡선 행을 표시한다
        cur[col] = cur[col].fillna(0).astype(int)
        m = cur[col] > 0
        if m.any():
            cur.loc[m, "ci_flag"] = [";".join(v for v in (str(f) if isinstance(f, str) else "", f"{tag} {k}") if v)
                                     for f, k in zip(cur.loc[m, "ci_flag"], cur.loc[m, col])]
    cur = cur.rename(columns={"d_n0": "net_value", "d_n0_lo": "net_value_lo", "d_n0_hi": "net_value_hi", "d_n0_beq": "net_value_beq",
                              "d_n0_beq_lo": "net_value_beq_lo", "d_n0_beq_hi": "net_value_beq_hi", "sig_n0": "sig_net"})
    cur = cur.drop(columns=["split_win_n0", "rep_win_n0"])
    first = ["target", "mode", "parent", "axis"] + GRP_COLS + ["n_lab", "rmse", "rmse_beq", "rmse_p0", "rmse_p1"]
    return cur[first + [c_ for c_ in cur.columns if c_ not in first]].sort_values(["target", "mode", "method", "learner", "alpha", "placement",
                                                                                     "lam", "n"]).reset_index(drop=True)


def build_minn(cur):
    G = ["target", "mode", "parent", "method", "learner", "alpha", "placement", "lam"]
    cn = cur[(cur.n > 0) & (cur.method != "P0")].copy()
    if not len(cn):
        return pd.DataFrame()
    cn = cn[cn.n_splits_used == cn.groupby(G).n_splits_used.transform("max")]            # 분할 일부에만 있는 n 은 제외
    mn = min_n(cn, group_cols=G, n_col="n", ci_hi_col="d_p0_hi", split_win_col="split_win_p0", rep_win_col="rep_win_p0")
    m1 = min_n(cn, group_cols=G, n_col="n", ci_hi_col="d_p1_hi", split_win_col="split_win_p1", rep_win_col="rep_win_p1")
    mn = mn.merge(m1.rename(columns=dict(n_star="n_star_p1", censored="censored_p1"))[G + ["n_star_p1", "censored_p1"]], on=G, how="left")
    aux = []
    full = {tuple(v[:-2]): (v[-2], v[-1]) for v in cur[cur.n == -1][G + ["d_p0", "d_p0_hi"]].itertuples(index=False, name=None)}
    for key, sub in cn.groupby(G, dropna=False):
        sub = sub.sort_values("n")
        both = sub[(sub.sig_p0 == "improve") & (sub.split_win_p0 >= H4.SPLIT_WIN_MIN - 1e-12) & (sub.rep_win_p0 >= H4.REP_WIN_MIN - 1e-12)]
        fa = full.get(tuple(key), (np.nan, np.nan))
        nv = int(sub.n_splits_valid.iloc[0]); flags = []
        if nv < 2:
            flags.append("valid_splits<2")
        if (sub.n_blocks_eval_min < H4.MIN_BLOCKS_FLAG).any():
            flags.append(f"blocks<{H4.MIN_BLOCKS_FLAG}")
        aux.append(dict(zip(G, key), n_star_both=float(both.n.iloc[0]) if len(both) else np.nan, n_splits_valid=nv, flag=";".join(flags),
                        n_blocks_eval_min=int(np.nan_to_num(sub.n_blocks_eval_min.min())), n_blocks_union=int(sub.n_blocks_union.iloc[0]),
                        best_n=int(sub.loc[sub.rmse.idxmin(), "n"]) if sub.rmse.notna().any() else -1,
                        best_d_p0=float(sub.d_p0.min()), d_p0_at_nmax=float(sub.d_p0.iloc[-1]), d_p0_hi_at_nmax=float(sub.d_p0_hi.iloc[-1]),
                        d_p0_all=float(fa[0]), d_p0_hi_all=float(fa[1])))
    mn = mn.merge(pd.DataFrame(aux), on=G, how="left")
    mn.loc[mn.n_splits_valid < 2, ["censored", "censored_p1"]] = True                     # 유효 분할 2개 미만은 주 정의 절단
    mn.loc[mn.n_splits_valid < 2, ["n_star", "n_star_p1", "n_star_both"]] = np.nan
    po = mn.target.isin(MAIN_POINT)                                                       # 계획서 §1: 점 추정만(블록 CI·n* 없음)
    mn["point_only"] = po.values
    if po.any():
        mn.loc[po, ["censored", "censored_p1"]] = True
        mn.loc[po, ["n_star", "n_star_p1", "n_star_both"]] = np.nan
        mn.loc[po, "flag"] = [";".join(v for v in (str(f) if isinstance(f, str) else "", "point_only") if v) for f in mn.loc[po, "flag"]]
    return mn


def _strat_rows(name, per, regions):
    """summarize_delta 와 같은 행을 단언 없이 만든다(점 추정치가 백분위 CI 밖일 때의 대체 경로)."""
    from polar.m1_stats import ci as _ci, strat as _strat, boot_p as _bp
    have = [r for r in regions if r in per]
    rows = []
    for r in have:
        d = per[r]; lo, hi = _ci(d["dist_cell"]); lob, hib = _ci(d["dist_blockeq"])
        rows.append(dict(test=name, target=r, n_cells=d["n_cells"], n_blocks=d["n_blocks"], rmse_A=d["rmse_A"], rmse_B=d["rmse_B"],
                         delta=d["delta_per_seed"], ci_lo=lo, ci_hi=hi, p_boot=_bp(d["dist_cell"]), delta_blockeq=d["delta_blockeq"],
                         ci_lo_beq=lob, ci_hi_beq=hib, ci_flag="" if d["dist_cell"] is not None else "blocks<8"))
    have_ci = [r for r in have if per[r]["dist_cell"] is not None]
    pool = have_ci if have_ci else have
    sd, sdb = _strat([per[r]["dist_cell"] for r in have]), _strat([per[r]["dist_blockeq"] for r in have])
    lo, hi = _ci(sd); lob, hib = _ci(sdb)
    neg = sum(per[r]["delta_per_seed"] < 0 for r in have)
    rows.append(dict(test=name, target=f"MEAN[{','.join(pool)}]", n_cells=sum(per[r]["n_cells"] for r in pool),
                     n_blocks=sum(per[r]["n_blocks"] for r in pool), rmse_A=float(np.mean([per[r]["rmse_A"] for r in pool])),
                     rmse_B=float(np.mean([per[r]["rmse_B"] for r in pool])), delta=float(np.mean([per[r]["delta_per_seed"] for r in pool])),
                     ci_lo=lo, ci_hi=hi, p_boot=_bp(sd), delta_blockeq=float(np.mean([per[r]["delta_blockeq"] for r in pool])),
                     ci_lo_beq=lob, ci_hi_beq=hib, delta_allregions=float(np.mean([per[r]["delta_per_seed"] for r in have])),
                     n_regions_all=len(have), regions_all=",".join(have), ci_flag=f"neg {neg}/{len(have)}; ci_regions {len(have_ci)}"))
    return rows


def strat_mean(name, tms, names, gA_fn, gB_fn):
    """주 지역 층화 평균(m1_stats.summarize_delta). 지역별 분포 = 분할 안 블록 부트스트랩의 분할 평균 분포.
    지역이 CI 풀에 들어가는 조건: 유효 분할 ≥ 1 이고 사용 분할의 채점 블록 합집합 ≥ MIN_BLOCKS_CI(8). 분할 안 채점 블록의 최솟값은
    n_blocks_split_min 열에 적고 8 미만이면 ci_flag 에 표시한다(풀에서 빼지는 않는다. 계획서 §4 개정 1).
    MEAN 행에는 CI 풀 지역 수(n_ci_regions)와 판정 가능 여부(undetermined)를 붙인다. 판정 불가 = 풀 지역 수 < MIN_POOL_REGIONS,
    CI 가 비유한, 또는 점 추정치가 백분위 CI 밖(ci_flag 가 'assert' 로 시작)."""
    per, nbmin = {}, {}
    for nm in names:
        tm = tms.get(nm)
        if tm is None:
            continue
        r = contrast(tm, gA_fn(tm), gB_fn(tm), return_dist=True)
        if r is None or not np.isfinite(r["delta"]):
            continue
        ok = tm.has_ci and tm.nb_union >= MIN_BLOCKS_CI and "dist" in r
        per[nm] = dict(n_cells=int(tm.n_eval_mean), n_blocks=int(tm.nb_union), n_seed=1, rmse_A=r["rmse_A"], rmse_B=r["rmse_B"],
                       delta_per_seed=r["delta"], delta_blockeq=r["delta_beq"], delta_cap100=np.nan, block_majority=np.nan,
                       dist_cell=r["dist"] if ok else None, dist_blockeq=r["dist_beq"] if ok else None)
        nbmin[nm] = int(min(r["n_blocks"])) if r["n_blocks"] else 0
    if not per:
        return []
    n_ci = sum(v["dist_cell"] is not None for v in per.values())
    bad = ""
    try:
        rows = summarize_delta(name, per, list(names), seed=0)
    except AssertionError as e:                                          # 점 추정치가 백분위 CI 밖(치우친 분포): 지역 행은 보존, 평균은 판정 불가
        rows = _strat_rows(name, per, list(names)); bad = f"assert: {e}"
    for r in rows:
        r.pop("delta_cap100", None); r.pop("block_majority", None)
        r["sig"] = _sig(r.get("ci_lo", np.nan), r.get("ci_hi", np.nan), r.get("ci_lo_beq", np.nan), r.get("ci_hi_beq", np.nan))
        if str(r.get("target", "")).startswith("MEAN["):
            r["n_ci_regions"] = int(n_ci); r["n_regions_target"] = int(len(names))
            if bad:
                r["ci_flag"] = f"{bad}; {r.get('ci_flag', '')}"; r["sig"] = ""
            r["undetermined"] = bool(bad) or n_ci < MIN_POOL_REGIONS or r["sig"] == ""
        else:
            r["n_blocks_split_min"] = nbmin.get(r["target"], 0)
            if nbmin.get(r["target"], 0) < MIN_BLOCKS_CI:
                r["ci_flag"] = ";".join(v for v in (str(r.get("ci_flag", "")), f"split blocks<{MIN_BLOCKS_CI}") if v)
    return rows


def base_lam(method):
    """판정에 쓰는 방법별 기준 λ(조합마다 한 행). P·V1 = 0.0, D0·D1 = 1.0, 잔차 방법 = LAM_BASE."""
    if method in ("P0", "P1", "P2", "P3", "V1"):
        return 0.0
    if method in ("D0", "D1"):
        return 1.0
    return LAM_BASE


def mean_state(r):
    """MEAN 행의 판정 상태: improve · worse · ns · undetermined."""
    if r.get("undetermined", False) or str(r.get("ci_flag", "")).startswith("assert") or r.get("sig", "") == "":
        return "undetermined"
    return r["sig"]


def pool_text(k, total=4):
    return f"지역 {int(k)}/{int(total)}"


def _g(method, n, lam=None, learner=BASE_LEARNER, alpha="1", placement="cell"):
    if method in ("P0",):
        return P0_GRP
    if method in ("P1", "P2", "P3"):
        return (method, "none", "1", placement, n, 0.0) if not (method == "P1" and n == 0) else ("P1", "none", "1", placement, 0, 0.0)
    if method == "V1":
        return ("V1", "none", "1", placement, n, 0.0)
    if method in ("D0", "D1"):
        return (method, learner, alpha, placement, n, 1.0)
    return (method, learner, alpha, placement, n, LAM_BASE if lam is None else lam)


def build_tests(tms, cur, mn, args):
    """L1–L8 판정량과 판정(계획서 §4, 개정 1). 방법별 기준 λ 는 base_lam(잔차 방법 0.25).
    '유의' = 셀 가중·블록 등가중 CI 가 모두 0 을 제외(sig 열). 모든 가설이 이 기준을 쓰고, 셀 가중 단독 결과는 보조 열로만 남긴다.
    층화 평균 판정은 CI 풀 지역 수를 함께 적는다: 4지역이면 확정, 2–3지역이면 '부분', 2지역 미만·계산 실패는 '판정 불가'."""
    out = []
    main4 = [f"{t}|x" for t in MAIN4]
    ns_all = sorted({int(v) for v in cur.n.unique()}) if len(cur) else []

    def rows_of(test, item, rows, **kw):
        for r in rows:
            is_mean = str(r.get("target", "")).startswith("MEAN[")
            out.append(dict(test_id=test, item=item, scope="MEAN4" if is_mean else "region", **kw, **r))
        return [r for r in rows if str(r.get("target", "")).startswith("MEAN[")]

    def verdict(test, text, stat="", role="주", **kw):
        out.append(dict(test_id=test, item="verdict", scope="verdict" if role == "주" else "verdict_aux", verdict=text, stat=str(stat), role=role,
                        **kw))

    def cv(target, mode, g):
        m = cur[(cur.target == target) & (cur["mode"] == mode)]
        for col, v in zip(GRP_COLS, g):
            m = m[m[col] == v]
        return m.iloc[0] if len(m) else None

    def mean_entries(test, item, name_fn, gA_fn, gB_fn, n_list, **kw):
        """n 마다 층화 평균을 구해 (n, 상태, CI 풀 지역 수, 셀 가중 CI 상한) 목록을 돌려준다. 행은 out 에 쌓는다."""
        res = []
        for n in n_list:
            mr = rows_of(test, item, strat_mean(name_fn(n), tms, main4, lambda tm, n=n: gA_fn(n), lambda tm, n=n: gB_fn(n)), n=n, **kw)
            res += [(n, mean_state(r), int(r.get("n_ci_regions", 0)), r.get("ci_hi", np.nan), r.get("ci_lo", np.nan)) for r in mr]
        return res

    def fmt(res):
        return "; ".join(f"n={n}: {s}({pool_text(k)})" for n, s, k, _, _ in res) if res else "행 없음"

    # L1 직접 ML: D0 는 n ≤ 40 에서 P0 을 넘지 못한다. 전량 행은 실제 라벨 수가 40 이하인 대상(러시아 W·E)에서만 포함한다
    n_beat, n_tested, detail = 0, 0, []
    for nm in main4:
        t = nm.split("|")[0]; beat, cell_only, tested = [], [], []
        for n in [n for n in ns_all if 0 <= n <= 40] + ([-1] if -1 in ns_all else []):
            r = cv(t, "x", _g("D0", n))
            if r is None:
                continue
            if n == -1 and not (np.isfinite(r.n_lab) and r.n_lab <= 40):
                continue
            tested.append(n)
            out.append(dict(test_id="L1", item="D0-P0", scope="region", target=nm, n=n, n_lab=r.n_lab, delta=r.d_p0, ci_lo=r.d_p0_lo,
                            ci_hi=r.d_p0_hi, delta_blockeq=r.d_p0_beq, ci_lo_beq=r.d_p0_beq_lo, ci_hi_beq=r.d_p0_beq_hi, sig=r.sig_p0,
                            ci_flag=r.ci_flag, note="전량(라벨 수 ≤ 40)" if n == -1 else ""))
            if r.sig_p0 == "improve":
                beat.append(n)
            elif np.isfinite(r.d_p0_hi) and r.d_p0_hi < 0:
                cell_only.append(n)
        n_tested += bool(tested); n_beat += bool(beat)
        detail.append(f"{t}: 유의 {beat}, 셀 가중만 {cell_only}, 검사 {tested}")
    if n_beat >= 2:
        txt = "기각"
    elif n_tested == len(main4):
        txt = "지지"
    elif n_tested == 0:
        txt = "판정 불가(행 없음)"
    else:
        txt = f"부분({pool_text(n_tested)}): 검사한 지역에서는 지지 조건 충족"
    verdict("L1", txt, f"두 가중 CI 상한 < 0 인 지역 {n_beat}/{n_tested} ({'; '.join(detail)})")

    # L2 결합: R2·R3 대 R1
    for m in ("R2", "R3"):
        res = mean_entries("L2", f"{m}-R1", lambda n: f"L2|{m}-R1|n{n}", lambda n: _g(m, n), lambda n: _g("R1", n), (10, 40, 160), lam=LAM_BASE)
        ok = [e for e in res if e[1] != "undetermined"]
        hit = [e[0] for e in ok if e[1] == "improve"]
        full = [e[0] for e in ok if e[2] == len(main4)]
        if not ok:
            txt = "판정 불가"
        elif len(hit) == len(ok) == 3 and len(full) == 3:
            txt = "지지"
        elif len(hit) == len(ok):
            txt = "부분 지지(판정 가능한 n 에서는 모두 유의, 지역 수 미달 또는 판정 불가 n 있음)"
        elif hit:
            txt = "부분"
        else:
            txt = "기각(증강은 라벨 0 전용)"
        verdict("L2", txt, f"{m}-R1 유의 개선 n = {hit}, 4지역 평균인 n = {full} | {fmt(res)}", contrast=f"{m}-R1")

    # L3 가중: 중첩 선택 α 의 n* ≤ 40. 적격 = 계획서가 열거한 네 대상 가운데 채점 블록 합집합 ≥ 8(strat_mean 과 같은 기준)
    for m in args.NESTED_METHODS:
        for al_mode in ("i", "x"):
            role = "주" if (m, al_mode) == L3_PRIMARY else "보조"
            label = f"{m}, AL 모드 {al_mode}"
            hits, n_elig, n_red, det = 0, 0, 0, []
            for t in L3_TARGETS:
                md = al_mode if t in SUB_AL else "x"
                if not len(mn):
                    continue
                q = mn[(mn.target == t) & (mn["mode"] == md) & (mn.method == m) & (mn.alpha == "nested") & (mn.lam == LAM_BASE) & (mn.placement == "cell")]
                q1 = mn[(mn.target == t) & (mn["mode"] == md) & (mn.method == m) & (mn.alpha == "1") & (mn.lam == LAM_BASE) & (mn.placement == "cell")
                        & (mn.learner == BASE_LEARNER)]
                if not len(q):
                    continue
                tm = tms.get(f"{t}|{md}")
                nbmin = min([st.nb for st in tm.used.values()]) if tm and tm.used else 0
                nbu = int(tm.nb_union) if tm else 0
                ns = float(q.n_star.iloc[0]); n1 = float(q1.n_star.iloc[0]) if len(q1) else np.nan
                elig = bool(nbu >= MIN_BLOCKS_CI and tm is not None and tm.has_ci)
                hit = bool(elig and np.isfinite(ns) and ns <= 40)
                red = bool(hit and (not np.isfinite(n1) or ns < n1))
                sel = cur[(cur.target == t) & (cur["mode"] == md) & (cur.method == m) & (cur.alpha == "nested") & (cur.lam == LAM_BASE)
                          & (cur.n.isin([10, 40]))] if "alpha_sel_counts" in cur else cur.iloc[:0]
                out.append(dict(test_id="L3", item=f"{m} nested n*", scope="region", target=f"{t}|{md}", n_star=ns, n_star_alpha1=n1,
                                censored=bool(q.censored.iloc[0]), n_blocks_eval_min=int(nbmin), n_blocks_union=nbu, eligible=elig, hit=hit,
                                reduced_vs_alpha1=red, role=role, note=label,
                                alpha_sel_counts=json.dumps({str(int(a)): str(b) for a, b in zip(sel.n, sel.alpha_sel_counts)}) if len(sel) else ""))
                n_elig += elig; hits += hit; n_red += red
                det.append(f"{t}|{md}: n*={ns if np.isfinite(ns) else 'NA'}(α=1 {n1 if np.isfinite(n1) else 'NA'}, 블록 합집합 {nbu}, 분할 최소 {nbmin})")
            if det:
                txt = f"판정 불가(적격 대상 {n_elig}개 < 2)" if n_elig < 2 else ("지지" if hits >= 2 else "기각")
                verdict("L3", txt, f"{label}: 적격 {n_elig}, n* ≤ 40 대상 {hits}, 그중 α = 1 의 n* 보다 작은 대상 {n_red} ({'; '.join(det)})",
                        role=role, contrast=f"{m} nested")

    # L4 순가치: R1 − P1. 최소 n 은 4지역 평균인 n 과 2–3지역 평균인 n 을 나눠 보고한다
    res = mean_entries("L4", "R1-P1", lambda n: f"L4|R1-P1|n{n}", lambda n: _g("R1", n), lambda n: _g("P1", n),
                       [n for n in ns_all if n > 0] + ([-1] if -1 in ns_all else []), lam=LAM_BASE)
    fin = [e for e in res if e[0] > 0 and e[1] != "undetermined"]
    f4 = [e[0] for e in fin if e[1] == "improve" and e[2] == len(main4)]
    fp = [e[0] for e in fin if e[1] == "improve" and e[2] < len(main4)]
    n4 = min(f4) if f4 else None; np_ = min(fp) if fp else None
    kp = {e[0]: e[2] for e in fin}
    if not fin:
        txt = "판정 불가"
    elif n4 is not None and n4 <= 40:
        txt = "희소 라벨에서도 ML 순가치 있음"
    elif np_ is not None and np_ <= 40:
        txt = f"부분({pool_text(kp[np_])}): 희소 라벨에서 순가치 있음"
    elif n4 is not None or np_ is not None:
        nn = min(v for v in (n4, np_) if v is not None)
        txt = "순가치는 n > 40 에서만" + ("" if kp[nn] == len(main4) else f"(부분, {pool_text(kp[nn])})")
    else:
        txt = "유한 n 격자에서 순가치 미확인"
    verdict("L4", txt, f"두 가중 CI 상한 < 0 인 최소 n: 4지역 평균 {n4}, 2–3지역 평균 {np_} | {fmt(res)}")

    # L5 학습기: R1 학습기 − R1 CatBoost
    learners = sorted({v for v in cur.learner.unique() if v not in ("none", BASE_LEARNER)}) if len(cur) else []
    for lr in learners:
        res = mean_entries("L5", f"R1[{lr}]-R1[{BASE_LEARNER}]", lambda n: f"L5|{lr}|n{n}", lambda n: _g("R1", n, learner=lr), lambda n: _g("R1", n),
                           [n for n in args.LEARNER_N if n in ns_all], lam=LAM_BASE, learner=lr)
        ok = [e for e in res if e[1] != "undetermined"]
        better = [e[0] for e in ok if e[1] == "improve"]; worse = [e[0] for e in ok if e[1] == "worse"]
        cell_b = [e[0] for e in ok if e[1] == "ns" and np.isfinite(e[3]) and e[3] < 0]
        cell_w = [e[0] for e in ok if e[1] == "ns" and np.isfinite(e[4]) and e[4] > 0]
        part = [e[0] for e in ok if e[2] < len(main4)]
        if not ok:
            txt = "판정 불가"
        else:
            txt = ("동급" if not better and not worse else "차이 있음") + (f"(부분: 지역 수 미달 n = {part})" if part else "")
        verdict("L5", txt, f"{lr}: 유의 개선 n = {better}, 유의 악화 n = {worse}, 셀 가중만 개선 {cell_b}·악화 {cell_w} | {fmt(res)}", learner=lr)

    # L6 알래스카 이전. 조합 = (방법, α ∈ {1, 10, 100, cont}) × n, 방법별 기준 λ 한 행. 통과 = 두 가중 CI 상한 < 0
    if len(cur):
        lamb = cur.method.map(base_lam)
        keep = (cur.placement == "cell") & (cur.method != "P0") & cur.learner.isin(["none", BASE_LEARNER]) & (cur.alpha != "nested") & (cur.lam == lamb)
        ak = cur[keep & cur.target.isin(SUB_AL) & (cur["mode"] == "i") & cur.n.isin([10, 40, 160])]
        mq = cur[keep & cur.target.isin(MAIN4) & (cur["mode"] == "x")]
        tot = hit = tot5 = hit5 = totc = hitc = 0
        for (m, a, n), sub in ak.groupby(["method", "alpha", "n"]):
            sub = sub[np.isfinite(sub.d_p0_hi) & np.isfinite(sub.d_p0_beq_hi)]
            if not len(sub):
                continue
            frac = float((sub.sig_p0 == "improve").mean()); chosen = frac > 0.5
            frac_c = float((sub.d_p0_hi < 0).mean())
            s5 = sub[sub.n_blocks_eval_min >= H4.MIN_BLOCKS_FLAG]
            frac5 = float((s5.sig_p0 == "improve").mean()) if len(s5) else np.nan
            q = mq[(mq.method == m) & (mq.alpha == a) & (mq.n == n) & np.isfinite(mq.d_p0_hi) & np.isfinite(mq.d_p0_beq_hi)]
            k = int((q.sig_p0 == "improve").sum()); kc = int((q.d_p0_hi < 0).sum())
            out.append(dict(test_id="L6", item="alaska_transfer", scope="combo", method=m, alpha=a, lam=base_lam(m), n=int(n), alaska_pass_frac=frac,
                            alaska_targets=len(sub), selected=bool(chosen), main4_pass=k, main4_tested=len(q),
                            main4_targets=",".join(q.target[q.sig_p0 == "improve"]), alaska_pass_frac_cell=frac_c, main4_pass_cell=kc,
                            alaska_targets_blk5=len(s5), alaska_pass_frac_blk5=frac5,
                            alaska_targets_lowblk=",".join(sorted(set(sub.target[sub.n_blocks_eval_min < H4.MIN_BLOCKS_FLAG])))))
            if chosen:
                tot += len(q); hit += k
            if np.isfinite(frac5) and frac5 > 0.5:
                tot5 += len(q); hit5 += k
            if frac_c > 0.5:
                totc += len(q); hitc += kc
        verdict("L6", "판정 불가(통과 조합 없음)" if tot == 0 else ("지지" if hit / tot >= 0.5 else "기각"),
                f"알래스카 통과 조합의 주 4지역 통과 {hit}/{tot}(두 가중, 방법별 기준 λ)")
        verdict("L6", "판정 불가(통과 조합 없음)" if tot5 == 0 else ("지지" if hit5 / tot5 >= 0.5 else "기각"),
                f"민감도: 분할 안 채점 블록 ≥ {H4.MIN_BLOCKS_FLAG} 인 알래스카 대상만으로 조합을 고른 경우 {hit5}/{tot5}", role="보조")
        verdict("L6", "판정 불가(통과 조합 없음)" if totc == 0 else ("지지" if hitc / totc >= 0.5 else "기각"),
                f"민감도: 셀 가중 CI 만 쓴 경우 {hitc}/{totc}", role="보조")

    # L7 공변량 의존 계수(점 추정 기준, 계획서 §4)
    allok, det, any_row = True, [], False
    for t, md in (("Canada", "x"), ("CA-3", "i"), ("CA-3", "x")):
        for n in [n for n in ns_all if n >= 40] + ([-1] if -1 in ns_all else []):
            ok_n = False
            for m in ("V1", "V1r"):
                r = cv(t, md, _g(m, n))
                if r is None:
                    continue
                any_row = True
                ok = bool(r.d_p0 <= 0 and r.d_p1 < 0)
                ok_n |= ok
                out.append(dict(test_id="L7", item=f"{m}", scope="region", target=f"{t}|{md}", n=int(n), method=m, delta=r.d_p0, ci_lo=r.d_p0_lo,
                                ci_hi=r.d_p0_hi, delta_vs_p1=r.d_p1, rmse_A=r.rmse, rmse_B=r.rmse_p0, passed=ok, note="주" if md != "x" or t == "Canada" else "보조"))
            if cv(t, md, _g("V1", n)) is not None and not (t == "CA-3" and md == "x"):
                allok &= ok_n; det.append(f"{t}|{md} n={n}:{'O' if ok_n else 'X'}")
    verdict("L7", "판정 불가" if not any_row else ("지지" if allok else "기각"), "; ".join(det))

    # L8 전량
    res = mean_entries("L8", "R1-P0", lambda n: "L8|R1-P0|all", lambda n: _g("R1", n), lambda n: P0_GRP, [-1], lam=LAM_BASE)
    mr8 = [r for r in out if r.get("test_id") == "L8" and r.get("scope") == "MEAN4"]
    if not res or all(e[1] == "undetermined" for e in res):
        txt = "판정 불가" + (f"({pool_text(res[0][2])})" if res else "")
    else:
        e = res[0]
        txt = ("지지" if e[1] == "improve" else "기각") if e[2] == len(main4) else f"부분({pool_text(e[2])}): " + ("지지 조건 충족" if e[1] == "improve" else "기각")
    verdict("L8", txt, "; ".join(f"Δ {r.get('delta', np.nan):.2f} [{r.get('ci_lo', np.nan):.2f}, {r.get('ci_hi', np.nan):.2f}], 블록 등가중 "
                                  f"[{r.get('ci_lo_beq', np.nan):.2f}, {r.get('ci_hi_beq', np.nan):.2f}], {pool_text(r.get('n_ci_regions', 0))}"
                                  for r in mr8))
    df = pd.DataFrame(out)
    front = ["test_id", "item", "scope", "role", "target", "n", "lam", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "sig",
             "verdict", "stat"]
    return df[[c_ for c_ in front if c_ in df] + [c_ for c_ in df.columns if c_ not in front]] if len(df) else df


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                                                                 # noqa: BLE001
        return "NA"


def find_shards(args):
    out = []
    if not args.SHARDS.exists():
        return out
    for p in sorted(args.SHARDS.glob("*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")                  # cpu 5마디, gpu 6마디(끝에 학습기)
        if len(parts) not in (5, 6) or parts[0] != args.TAG:
            continue
        b = str(p)[:-len("_unit.json")]
        if Path(b + "_runs.csv").exists() and Path(b + "_blocksse.npz").exists():
            out.append(dict(unit=p, runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), part=parts[1],
                            learner=parts[5] if len(parts) == 6 else ""))
    return out


def check_shard_cfg(args, units):
    """조각 사이 설정 일치 확인. 부분마다 공통 설정 해시(cfg_common)가 하나여야 한다. 다르면 중단한다(--allow-mixed-cfg 로 진행).
    해시가 없는 조각(개정 전 형식)은 'legacy' 로 묶는다. 코드 해시가 여럿이면 경고만 한다."""
    info = {}
    for part in sorted({u.get("part", "") for u in units}):
        us = [u for u in units if u.get("part", "") == part]
        hs = Counter(str(u.get("cfg_common", "legacy")) for u in us)
        cs = Counter(str(u.get("code_sha", "legacy")) for u in us)
        info[part] = dict(cfg_common=dict(hs), code_sha=dict(cs))
        if len(cs) > 1:
            print(f"  [warn] 부분 {part}: 조각의 코드 해시가 {len(cs)}종이다 {dict(cs)}", flush=True)
        if len(hs) > 1:
            msg = f"부분 {part}: 조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}. 설정이 다른 실행이 섞였다"
            if not args.allow_mixed_cfg:
                raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
            print("  [warn] " + msg, flush=True)
    return info


def timing_table(units):
    """조각의 축|학습기|방법별 적합 수·시간·학습 행 수. 사전 점검에서 학습기별 적합 시간을 읽는 표다."""
    rows = []
    for u in units:
        for k, nf in (u.get("n_fit_detail") or {}).items():
            ax, lr, m = (k.split("|") + ["", ""])[:3]
            sec = float((u.get("sec_detail") or {}).get(k, 0.0)); nr = float((u.get("rows_detail") or {}).get(k, 0.0))
            rows.append(dict(part=u.get("part", ""), target=u.get("target", ""), mode=u.get("mode", ""), split=u.get("split", -1), axis=ax,
                             learner=lr, method=m, n_fit=int(nf), sec=sec, sec_per_fit=sec / max(int(nf), 1), rows_per_fit=nr / max(int(nf), 1),
                             est_s=float((u.get("est_detail") or {}).get(k, np.nan)), device=u.get("device", ""), threads=u.get("threads", ""),
                             status=u.get("status", "")))
    return pd.DataFrame(rows)


def summarize(args, elapsed=0.0, skipped=None):
    t0 = time.time()
    sh = find_shards(args)
    if not sh:
        print(f"[summarize] 조각 없음: {args.SHARDS}/{args.TAG}__*", flush=True)
        return None
    D = get_data(args)
    units = [json.loads(s["unit"].read_text()) for s in sh]
    cfg_info = check_shard_cfg(args, units)
    frames = [pd.read_csv(s["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str), keep_default_na=False, na_values=["", "nan", "NaN"])
              for s in sh if s["runs"].stat().st_size > 1]
    runs = pd.concat([f for f in frames if len(f)], ignore_index=True)
    runs["alpha_sel"] = runs.alpha_sel.fillna("")
    if "n_nonfinite" not in runs:                                                                        # 개정 전 형식의 조각
        runs["n_nonfinite"] = 0
    if "fit_flag" not in runs:
        runs["fit_flag"] = ""
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int); runs["fit_flag"] = runs.fit_flag.fillna("").astype(str)
    runs = runs.drop_duplicates(subset=["target", "mode", "split"] + KEY_COLS, keep="first")             # P 행은 cpu·gpu 조각에 모두 있다
    failed = runs[(runs.n_nonfinite > 0) | (runs.fit_flag != "")]                                        # 저장에서 빠졌거나 대체된 키
    stores = load_stores([s["npz"] for s in sh])
    tms = {}
    for nm in sorted({k[0] for k in stores}):
        t = nm.split("|")[0]
        info = D.split_structure(t) if set(D.split_structure(t)) >= {sp for (n_, sp) in stores if n_ == nm} else {}
        tms[nm] = TMx(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, info, args.nboot)
    cur = build_curve(tms, runs)
    mn = build_minn(cur)
    tests = build_tests(tms, cur, mn, args)
    tg = pd.DataFrame([{k: v for k, v in u.items() if not isinstance(v, (dict, list))} | dict(
        n_fit=json.dumps(u.get("n_fit", {})), sec=json.dumps(u.get("sec", {})), fail=json.dumps(u.get("fail", {})),
        errors=json.dumps(u.get("errors", []), ensure_ascii=False)) for u in units])
    if skipped:
        tg = pd.concat([tg, pd.DataFrame(skipped)], ignore_index=True)
    O = args.OUT; O.mkdir(parents=True, exist_ok=True)
    cur.to_csv(O / f"{args.TAG}_curve.csv", index=False); mn.to_csv(O / f"{args.TAG}_minn.csv", index=False)
    tests.to_csv(O / f"{args.TAG}_tests.csv", index=False)
    tg.sort_values(["target", "mode", "split", "part"]).to_csv(O / f"{args.TAG}_targets.csv", index=False)
    tt = timing_table(units)
    tt.to_csv(O / f"{args.TAG}_timing.csv", index=False)
    failed.to_csv(O / f"{args.TAG}_failed.csv", index=False)
    n_unit_bad = Counter(str(u.get("status", "ok")) for u in units)
    if len(failed) or set(n_unit_bad) - {"ok"}:
        print(f"[summarize] 실패 기록: 조각 상태 {dict(n_unit_bad)} · 저장에서 빠졌거나 대체된 키 {len(failed):,} → {args.TAG}_failed.csv", flush=True)
    if len(tt):
        lt = tt.groupby(["part", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), sec=("sec", "sum"))
        lt["sec_per_fit"] = lt.sec / lt.n_fit.clip(lower=1)
        print("[summarize] 학습기별 적합 시간(전 조각 합계)\n" + lt.to_string(index=False), flush=True)
    fit_tot = Counter()
    for u in units:
        for k, v in u.get("n_fit", {}).items():
            fit_tot[f"{u['part']}:{k}"] += int(v)
    meta = dict(stage="H40/LG", plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md", git_commit=git_commit(), tag=args.TAG,
                args={k: v for k, v in vars(args).items() if k.islower()}, targets=[f"{t}|{m}" for t, m in args.TARGETS], splits=args.SPLITS,
                n_grid=args.N_GRID, draws=args.DRAWS, seeds=args.SEEDS, lams=args.LAMS, alphas=args.ALPHAS, methods=args.METHODS,
                learner_targets=[f"{t}|{m}" for t, m in args.LEARNER_TARGETS], nested_targets=[f"{t}|{m}" for t, m in args.NESTED_TARGETS],
                key_fields=KEY_COLS, p0_key=list(P0_KEY), all_n=-1, n_units=len(units), parts=sorted({u["part"] for u in units}),
                n_fit=dict(fit_tot), n_fit_total=int(sum(fit_tot.values())), unit_elapsed_s_sum=float(sum(u.get("elapsed_s", 0) for u in units)),
                elapsed_s=round(elapsed, 1), summarize_s=round(time.time() - t0, 1), n_rows=int(len(runs)), n_curve=int(len(cur)),
                n_minn=int(len(mn)), n_tests=int(len(tests)),
                split_info={t: {str(sp): v for sp, v in D.split_structure(t).items()} for t in sorted({t for t, _ in args.TARGETS})},
                skipped=skipped or [], shard_cfg=cfg_info, unit_status=dict(n_unit_bad), n_failed_keys=int(len(failed)),
                nested_alphas=args.NESTED_ALPHAS, r2_exclude=args.R2_EXCLUDE, subregion_src=getattr(D, "sub_src", ""),
                subregion_kmeans_diff=int(getattr(D, "sub_kmeans_diff", 0)),
                ci_rule="h4_common.boot_delta_blocks: 분할 안 채점 블록 재표집, 방법·추출·seed 공통 인덱스, 분할 분포 평균. 유효 분할만 결합",
                sig_rule="셀 가중·블록 등가중 CI 가 모두 0 을 제외할 때만 improve/worse",
                minn_rule="h4_common.min_n: d_p0_hi < 0 & split_win ≥ 2/3 & rep_win ≥ 0.75. 유효 분할 < 2 는 절단",
                strat_rule="m1_stats.summarize_delta. 지역 분포는 블록 합집합 ≥ 8 이고 유효 분할 ≥ 1 일 때만 CI 풀에 넣는다. 분할 안 채점 블록 "
                           "최솟값이 8 미만이면 표시만 한다. CI 풀 지역 수 4 = 확정, 2–3 = 부분, 2 미만 또는 계산 실패 = 판정 불가",
                test_rule=dict(sig="L1–L8 모두 두 가중 CI 기준(sig). 셀 가중 단독은 보조 열", base_lam="P·V1 0.0, D0·D1 1.0, 잔차 방법 0.25",
                               L1="전량 행은 실제 라벨 수 ≤ 40 인 대상만 포함", L3=f"적격 = {L3_TARGETS} 가운데 채점 블록 합집합 ≥ 8. "
                               f"주 판정 = {L3_PRIMARY[0]}·AL 모드 {L3_PRIMARY[1]}. 후보 α = nested_alphas",
                               L6="조합 = (방법, α ∈ {1, 10, 100, cont}) × n. nested 는 조합에서 뺀다. 채점 블록 < 5 대상은 보조 판정에서 제외",
                               point_only=f"{MAIN_POINT} 는 블록 CI·n* 를 계산하지 않는다"),
                conventions=dict(P_n0="n = 0 에서 P1–P3 = P0", R3="원천 잔차는 D1 표본 내 예측 기준", V1="ẑ 를 학습 z 범위로 자름",
                                 nested="선택은 seed 0 적합, 같은 α 를 모든 seed 에 적용. n < 10 또는 라벨이 한 블록뿐이면 α = 1",
                                 mlp_cont="BatchNorm·Dropout 평가 모드 고정",
                                 mlp_alpha="검증 10 % 를 고유 행에서 먼저 떼고 학습 쪽 대상 행만 α 배 반복. 표준화·y 척도는 고유 행 기준",
                                 learner_r2="신경망 학습기의 R2 는 유사 잔차 행(복원 추출)을 넣은 뒤 검증 10 % 를 뗀다(검증에 복제 행 포함)",
                                 realmlp="n_epochs = realmlp_epochs(기본 256). --epochs 를 쓰지 않는다",
                                 nonfinite="비유한 예측이 있는 키는 저장하지 않고 failed.csv 에 기록",
                                 invalid_split="유효 분할이 있는 대상의 무효 분할(채점 블록 < 2)은 실행하지 않는다"))
    (O / f"{args.TAG}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(f"[summarize] 조각 {len(sh)} · runs {len(runs):,} · curve {len(cur):,} · minn {len(mn):,} · tests {len(tests):,} · "
          f"{time.time() - t0:.0f}s → {O}/{args.TAG}_*", flush=True)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])] if len(tests) else tests
    if len(v):
        print(v[["test_id", "role", "verdict", "stat"]].to_string(index=False), flush=True)
    return dict(curve=cur, minn=mn, tests=tests, targets=tg, meta=meta, timing=tt, failed=failed)


# ================================================================ 실행
def count_only(args, D, units, skipped):
    """학습 없이 적합 수와 추정 시간을 출력한다. 추정 = 학습기별 기준 시간(BASE_FIT) × (학습 행 수 / 기준 행 수)^지수 의 합.
    기준 시간은 로컬 자료에서 온 값이므로 Rescale 사전 점검(--precheck)의 실측으로 바꿔 읽는다."""
    rows = [run_unit(args, args.part, t, m, sp, lr, dry=True) for t, m, sp, lr in units]
    det, nrw, est = Counter(), Counter(), Counter()
    for r in rows:
        for k, v in r.pop("_detail").items():
            det[k] += v
        for k, v in r.pop("_rows").items():
            nrw[k] += v
        for k, v in r.pop("_est").items():
            est[k] += v
    df = pd.DataFrame(rows).fillna(0)
    fc = [c_ for c_ in df.columns if c_.startswith("fit_")]
    df["fit_total"] = df[fc].sum(1).astype(int)
    df["order"] = list(range(len(df)))                                   # 실행 순서(unit_priority 정렬 뒤의 순번)
    args.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.OUT / f"{args.TAG}_count_{args.part}.csv", index=False)
    tot = df[fc].sum().astype(int)
    print(f"[count-only] 부분 {args.part} · 작업 단위 {len(units)} · 건너뜀 {len(skipped)}"
          f"({', '.join(sorted({s['status'] for s in skipped})) or '없음'})", flush=True)
    by_t = df.groupby(["target", "mode"], sort=False).agg(units=("split", "size"), n_A_min=("n_A", "min"), n_A_max=("n_A", "max"),
                                                          fit_total=("fit_total", "sum"), rows=("n_rows", "sum"), est_h=("est_s", "sum"))
    by_t["est_h"] = (by_t.est_h / 3600).round(2)
    print(by_t.to_string(), flush=True)
    print("[count-only] 축별 적합 수: " + " · ".join(f"{k[4:]} {int(v):,}" for k, v in tot.items()), flush=True)
    heavy = int(tot.sum() - sum(int(tot.get(k, 0)) for k in ("fit_v1_ridge", "fit_alpha_cont", "fit_nested_cont", "fit_mlp_cont")))
    print(f"[count-only] 총 적합 {int(tot.sum()):,} (전체 학습 {heavy:,} + 경량[ridge V1·이어 학습] {int(tot.sum()) - heavy:,}) · 저장 행 {int(df.n_rows.sum()):,}",
          flush=True)
    lt = {}
    for k in det:
        ax, lr, m = (k.split("|") + ["", ""])[:3]
        a = lt.setdefault(lr, dict(n_fit=0, rows=0.0, est_s=0.0, n_R2=0, est_R2=0.0))
        a["n_fit"] += det[k]; a["rows"] += nrw[k]; a["est_s"] += est[k]
        if m in ("R2", "D1"):
            a["n_R2"] += det[k]; a["est_R2"] += est[k]
    tab = pd.DataFrame([dict(learner=lr, n_fit=v["n_fit"], rows_per_fit=round(v["rows"] / max(v["n_fit"], 1)), base_s=BASE_FIT.get(lr, (np.nan,))[0],
                             equiv_fits=round(v["est_s"] / BASE_FIT[lr][0]) if lr in BASE_FIT else np.nan, est_h=round(v["est_s"] / 3600, 2),
                             n_fit_pseudo=v["n_R2"], est_h_pseudo=round(v["est_R2"] / 3600, 2)) for lr, v in lt.items()])
    if len(tab):
        tab = tab.sort_values("est_h", ascending=False)
        tab.to_csv(args.OUT / f"{args.TAG}_count_{args.part}_learners.csv", index=False)
        print("[count-only] 학습기별 추정(프로세스 1개 기준 누적 시간. n_fit_pseudo·est_h_pseudo = 유사라벨 행을 넣는 D1·R2 적합)", flush=True)
        print(tab.to_string(index=False), flush=True)
    tot_h = float(sum(v["est_s"] for v in lt.values())) / 3600
    if args.part == "cpu":
        for w in (2, 10, 15):
            print(f"  추정 누적 {tot_h:.1f} h(4스레드 기준) → 워커 {w}개 {tot_h / w:.1f} h", flush=True)
    else:
        for w in (4, 8, 12):
            print(f"  추정 누적 {tot_h:.1f} GPU-h(로컬 GPU 기준) → 워커 {w}개 {tot_h / w:.1f} h · T4 2–3배 가정 {2 * tot_h / w:.1f}–{3 * tot_h / w:.1f} h",
                  flush=True)
        print("  생성 모델(cfm·ddpm·nflow)의 표본 추출 시간은 채점 셀 수에 비례하며 위 추정에 들어 있지 않다", flush=True)
    return df


def run_permitted(args):
    """스모크가 아닌 실행의 허용 여부. Rescale 작업 명령에서 LG_RESCALE=1 또는 --allow-local 을 준다."""
    return bool(args.allow_local) or os.environ.get("LG_RESCALE", "") == "1"


def main(argv=None):
    args = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(args.threads)                                  # 워커(spawn)가 물려받는다
    if args.GPUS and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        print("[warn] 부모 환경의 CUDA_VISIBLE_DEVICES 가 빈 문자열이다. --gpus 를 무시하고 CPU 로 돈다", flush=True)
        args.GPUS = []
    if args.part == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    will_run = not (args.count_only or args.summarize_only or args.write_subregion_map)
    if will_run and not args.smoke and not run_permitted(args):             # 자료를 읽기 전에 거부한다
        raise SystemExit("[거부] 스모크가 아닌 실행(사전 점검·본 실행)은 Rescale 작업에서 한다. Rescale 작업 명령에 LG_RESCALE=1 또는 "
                         "--allow-local 을 준다(사용자 지시 2026-09-29: 공유 서버의 CPU·GPU 를 쓰지 않는다)")
    if args.write_subregion_map:
        df = load_base(args.PROC)
        sub, subs = make_subregions(df, args.k_sub)
        tab = subregion_table(df, sub)
        if args.SUBMAP.exists():
            old = pd.read_csv(args.SUBMAP, dtype=dict(parent=str, subregion=str))
            mg = old[["parent", "block", "subregion"]].merge(tab[["parent", "block", "subregion"]], on=["parent", "block"], how="outer",
                                                             suffixes=("_old", "_new"))
            nd = int((mg.subregion_old != mg.subregion_new).sum())
            print(f"[subregion-map] 기존 대응표와 다른 블록 {nd}개(전체 {len(mg)})", flush=True)
            if nd:                                                           # 대응표가 정의다. 다른 환경의 k-means 로 덮어쓰지 않는다
                raise SystemExit(f"[subregion-map] 기존 대응표 {args.SUBMAP} 와 다르다. 덮어쓰지 않는다(정의를 바꾸려면 파일을 지우고 다시 실행)")
        args.SUBMAP.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(args.SUBMAP, index=False)
        print(f"[subregion-map] 블록 {len(tab)}개 · 하위 지역 {tab.subregion.nunique()}개 → {args.SUBMAP}", flush=True)
        print(subs.to_string(index=False), flush=True)
        return dict(executed=[], skipped=[], resumed=[])
    D = get_data(args)
    units, skipped = enumerate_units(args, D, args.part)
    units.sort(key=lambda u: unit_priority(args, D, u))
    print(f"[data] {len(D.df):,}셀 · 부분 {args.part} · 대상 {len(args.TARGETS)} · 분할 {args.SPLITS} · n {args.N_GRID} · 추출 {args.DRAWS} · "
          f"seed {args.SEEDS} · 학습기 {args.LEARNERS} · α {args.ALPHAS} · 중첩 후보 {args.NESTED_ALPHAS} · λ {args.LAMS} · 하위 지역 {D.sub_src} · "
          f"작업 단위 {len(units)}(건너뜀 {len(skipped)})", flush=True)
    if args.count_only:
        count_only(args, D, units, skipped)
        return dict(executed=[], skipped=skipped, resumed=[])
    if args.summarize_only:
        summarize(args, 0.0, skipped)
        return dict(executed=[], skipped=skipped, resumed=[])
    if not run_permitted(args):                                              # 여기까지 온 것은 스모크뿐이다: 자원을 제한한다
        if args.GPUS:
            print("[warn] 허용 표지 없는 스모크: --gpus 를 무시하고 CPU 로 돈다", flush=True)
            args.GPUS = []
        if args.workers > 2:
            print(f"[warn] 허용 표지 없는 스모크: --workers {args.workers} → 2", flush=True)
            args.workers = 2
    args.SHARDS.mkdir(parents=True, exist_ok=True)
    resumed, todo = [], []
    for u in units:
        ok, why = unit_state(args, args.part, *u) if args.resume else (False, "")
        if ok:
            resumed.append(u)
            print(f"  [resume] 건너뜀 {unit_name(u)} (조각 있음, 상태 {why})", flush=True)
        else:
            todo.append(u)
            if args.resume and why != "조각 없음":
                print(f"  [resume] 다시 실행 {unit_name(u)} ({why})", flush=True)
    use_gpu = args.part == "gpu" and len(args.GPUS) > 0
    nproc = len(args.GPUS) * args.procs_per_gpu if use_gpu else max(args.workers, 1)
    nproc_inline = (not use_gpu) and args.workers <= 0                       # --workers 0: 프로세스 풀 없이 이 프로세스에서 차례로 실행
    print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · 프로세스 {nproc} · 스레드 {args.threads} · "
          f"장치 {'GPU ' + ','.join(args.GPUS) if use_gpu else 'CPU'} · epochs {args.epochs} · RealMLP epochs {args.realmlp_epochs}", flush=True)
    done, failed = [], []

    def log(u):
        done.append(u)
        print(f"  [{u['target']}|{u['mode']}|s{u['split']}|{u['part']}{'|' + u['learner'] if u['learner'] else ''}] 적합 {u['n_fit_total']} · "
              f"행 {u['n_rows']} · A {u['n_A']} · 채점 {u['n_eval']}/{u['nb_eval']}블록{'' if u['valid'] else '(무효 분할)'} · 원천 {u['n_src']} · "
              f"E0 {u['E0']:.3f} · {u['elapsed_s']}s · 장치 '{u['device']}' · 상태 {u['status']}(실패 {u['n_fail']}) · "
              f"완료 {len(done)}/{len(todo)} · 누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append(u)
        print(f"  [FAIL] {unit_name(u)}: {repr(e)[:300]}", flush=True)

    if todo and nproc_inline:
        _worker_init(argv, None, args.threads)
        for u in todo:
            try:
                log(_worker_run(args.part, *u))
            except Exception as e:                                           # noqa: BLE001
                fail(u, e)
    elif todo:
        ctx = multiprocessing.get_context("spawn")                           # fork 후 OpenMP 충돌 회피
        remaining, attempt = list(todo), 0
        while remaining:
            q = None
            if use_gpu:
                q = ctx.Queue()
                for g in args.GPUS:
                    for _ in range(args.procs_per_gpu):
                        q.put(g)
            broken = []
            with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx, initializer=_worker_init, initargs=(argv, q, args.threads)) as ex:
                futs = {ex.submit(_worker_run, args.part, *u): u for u in remaining}
                for f in as_completed(futs):
                    try:
                        log(f.result())
                    except BrokenProcessPool:                                # 워커 비정상 종료(메모리 부족 등): 남은 단위는 새 풀에서 다시 돈다
                        broken.append(futs[f])
                    except Exception as e:                                   # noqa: BLE001  한 단위의 실패가 나머지를 막지 않게 한다
                        fail(futs[f], e)
            if not broken:
                break
            attempt += 1
            if attempt > args.pool_retries:
                for u in broken:
                    fail(u, RuntimeError(f"프로세스 풀이 {attempt}회 깨졌다. 재시도 상한 {args.pool_retries}"))
                break
            remaining = sorted(broken, key=lambda u: unit_priority(args, D, u))
            print(f"[pool] 워커 비정상 종료. 남은 {len(remaining)} 단위로 풀을 다시 만든다(재시도 {attempt}/{args.pool_retries})", flush=True)
    n_fail = len(failed) + sum(1 for u in done if u["status"] == "failed")
    n_part = sum(1 for u in done if u["status"] == "partial")
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 적합 실패 {n_part} · {time.time() - t0:.0f}s", flush=True)
    if not args.no_summarize:
        summarize(args, time.time() - t0, skipped)

    def ident(t, m, sp, lr):
        return (t, m, sp) if not lr else (t, m, sp, lr)
    return dict(executed=[ident(u["target"], u["mode"], u["split"], u["learner"]) for u in done], skipped=skipped,
                resumed=[ident(*u) for u in resumed], n_fail=n_fail, n_partial=n_part)


if __name__ == "__main__":
    res = main()
    sys.exit(1 if res.get("n_fail") else 0)
