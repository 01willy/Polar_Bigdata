"""H41 · 검증 사다리(X3a)와 지역 내 블록 교차검증(X3b). 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6A(개정 5 사전 등록, 개정 7 조작적 정의)의 구현.

목적
  같은 자료·같은 모델 집합·같은 seed 로 검증 설계만 바꿔 가며 오차를 잰다(X3a). 무작위 분할의 결론이 지역 홀드아웃에서 유지되는지를
  본다(L19–L21). 지역 내 조건(라벨이 지역 블록의 4/5)에서 잔차 ML 이 재보정 물리식을 넘는지를 본다(X3b, L22). CPU 전용이다.

자료와 채점 집합
  자료 = h40.get_data(load_base, F4_direct 17,467셀). s = e5_sqrt_tdd, y = alt_cm.
  채점 집합 = eval_mask 셀 가운데 macro ∈ {Alaska, Lena, Canada, Russia_W, Russia_E, Russia_C, Greenland}(17,374셀, 블록 174개).
  모든 단이 같은 채점 집합을 쓰고 각 채점 셀을 반복마다 한 번 예측한다. 학습 행 = y 와 s 가 유한한 전 셀(h40.Ctx 와 같은 규칙).

단(--schemes)과 분할 규칙
  V-R          셀 무작위 5-fold. RandomState(seed_of('lgv', 'V-R', 반복)).permutation(N) 의 위치 mod 5. 반복 3
  V-P          거친 공변량 묶음(기후 8·토양 9·CCI 2열을 소수 6자리로 반올림한 값이 같은 셀, 결측은 표지값) 단위 5-fold. 반복 3
  V-S          0.05° 격자 묶음 (floor(lat/0.05), floor(lon/0.05)) 단위 5-fold. 반복 3
  V-B          0.5° 블록(df.block) 단위 5-fold. 반복 3
               묶음 단은 묶음을 RandomState(seed_of('lgv', 단, 반복))로 순열한 뒤 셀 수가 가장 적은 fold 에 차례로 넣는다(동률은 작은 번호)
  V-C0·100·500 공간 군집(대응표 --cluster-map, 30개) 하나 제외. 제외 군집의 어느 셀과도 d km 안인 학습 셀을 뺀다. 반복 1
  V-G          지역 홀드아웃(macro 7지역). 학습 = h40.Data.source_idx(지역, 'x')(대상 제외, 100 km 버퍼). 반복 1
  W2-<지역>    지역 내 블록 5-fold(seed_of('lgv', 'W', 지역, 반복)), 반복 5. 학습 = 지역 내 학습 fold ∪ 타 지역 원천(source_idx)
  W1-<지역>    같은 fold. 학습 = 지역 내 학습 fold 만
  모든 단에서 학습 셀과 채점 셀의 교집합이 없음을 단언한다. 버퍼 단은 버퍼 안 학습 셀이 0 개임을 haversine 전수 계산으로 단언한다.

모델(키 = (method, learner, seed, lam). 물리식은 learner 'none', seed −1, lam 0.0)
  통합 자료 단(V-*)   PS = E_TR·s(E_TR = 학습 집합 최소제곱) | D0[catboost_lo, catboost, rf](lam 1.0) | D0m = D0 에서 e5_tdd, e5_sqrt_tdd,
                      cci_alt, cci_valid 를 뺀 입력 | F1k = 입력 [x25, p4_ku, p2_edaphic] | V2 = exp(ẑ)·s, z = log(y/s), ẑ 는 학습 z 의
                      범위로 자른다(lam 0.0) | RM0 = E_TR·s·exp(λ·(ẑ − log E_TR)), V2 의 적합에서 계산(λ 1.0 은 V2 와 같아 저장하지 않는다)
                      | RS[catboost_lo, catboost, rf] = E_TR·s + λ·g, g 는 (x_TR, y − E_TR·s)에 적합
  V-G 추가(N1)        B:ed_raw, B:ku_raw, B:cci_raw(무보정 값), P0@soil, P0@ku, P0@ed, P0@cci, P0@tddm(척도 보정 c0·b), B:s_aff, B:cci_aff, B:ens
                      앵커 5종은 h42 의 X9 와 같다. tddm = 연도 정합 √TDD(--tdd-matched 의 표를 loc_id 로 결합한다. 표가 없으면 그 앵커를 건너뛴다).
                      앵커 b 의 비유한 또는 0 이하 셀은 ρ·s 로 바꾼다(ρ = 학습 셀의 b/s 중앙값, 라벨 미사용). 계수는 학습 집합에서만 구한다.
                      p4_ku 결측 셀을 채점에서 빼는 방식은 키 B:ku_raw#ex, P0@ku#ex(결측 셀 예측 NaN)로 함께 저장한다
  W2                  P0 = E0·s(E0 = 원천 최소제곱) | P1w = E_w·s, E_w = (m·E_ls(TRw) + κ·E0)/(m + κ) | P2w = E_ls·s | D0[3종] | F1k | V2
                      | R0 = E0·s + λ·g | R1w[3종] = E_w·s + λ·g(원천 행 E0 앵커, 지역 내 행 E_w 앵커) | RMw = E_w·s·exp(λ·ĥ)(같은 앵커 규칙)
                      | R1w@nest = R1w[catboost_lo] 에 지역 내 학습 fold 안 블록 단위 4-fold 로 고른 λ 를 쓴다(lam 키 −1.0)
  W1                  PSw, D0w[3종], F1kw, V2w, RSw[3종](통합 자료 단과 같은 식을 지역 내 학습 fold 에 적용)
  학습기              catboost_lo = h40.cb_fit(200회, 학습률 0.05, 깊이 3) | catboost = 600회, 학습률 0.03, 깊이 6, l2 3
                      | rf = RandomForestRegressor(500그루, max_features 1/3, min_samples_leaf 5), 결측은 학습 행렬 중앙값
  학습 행렬의 행 순서는 [원천; 지역 내 라벨]이다(h40 의 train_rows 와 같다).

누설 규약
  물리 계수 E, 앵커 척도 계수, 결측 대체 통계(중앙값, ρ)는 각 단위의 학습 셀에서만 구한다. 채점 셀의 y 는 채점에만 쓴다.
  R1w@nest 의 λ 선택은 지역 내 학습 fold 의 라벨만 쓴다(채점 fold 의 라벨을 쓰지 않는다).
  예외(라벨은 쓰지 않는다): p4_ku, p2_edaphic 는 load_base 가 라벨 없이 공변량으로 계산한 열이다. physics._fallback 은 공변량 결측을 전체 셀
  중앙값으로 채운다. 이 통계는 채점 셀의 공변량을 포함한다(h40, h42 와 같은 자료 적재를 쓰기 위해 그대로 둔다. 동결 모듈이다).
  자료 확인 결과 토양 열 결측은 399셀이고 그 가운데 396셀이 레나에 있다(레나 3,037셀의 13 %). 영향을 받는 키는 F1k, F1kw 와 V-G 의 N1 기준선
  B:ku_raw, B:ed_raw, P0@ku, P0@ed, B:ens 다. 확인적 가설 L19(D0 − PS)는 이 열을 쓰지 않는다. 영향의 크기는 h42 의 결측 대체 민감도 행
  (x9 축, 학습 쪽 중앙값으로 채운 열)으로 본다. 이 파일은 민감도 행을 내지 않는다.

채점(집계)
  블록 식별자 = (macro, block). 지역 r 의 재표집 다중도 W_r = h4_common.boot_weights(블록 수, nboot, seed_of('lgvboot', r))를 모든 단·모델·
  반복이 공유한다. 키 묶음(반복·seed)의 RMSE 를 평균한 뒤 차를 구한다(h4_common.boot_delta_blocks 의 분할 하나와 같은 식).
  층화 평균 = m1_stats.strat(채점 블록 8개 이상 지역만 CI 풀). 주 요약 = 5지역(Alaska, Lena, Canada, Russia_W, Russia_E).
  전체 셀 풀의 CI 는 지역 안 블록 재표집 다중도를 이어 붙여 만든다(블록 8개 미만 지역도 지역 안에서 재표집한다).
  4분 판정 verdict4: 우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 네 끝값의 절댓값 ≤ δ(주 0.5, 보조 1.0 cm), 그 밖 미결정.
  풀 지역 수 2 미만, CI 없음, 점 추정치가 백분위 CI 밖이면 판정 불가.
  반복이 같은 블록 집합을 공유하므로 h42 의 보조 CI(지역 블록 공통 재표집)와 주 CI 의 구분이 없다.
  동등성 p(p_equiv)는 셀 가중과 블록 등가중 분포에서 구한 값 가운데 큰 값이다(h42 와 같다).
  판정 문구의 공통 규칙(계획서 §6A.2 의 개정 7, mark_verdict): 풀 지역 수가 5 보다 적으면 '부분(지역 k/5): ', 대비의 행(반복·seed)이 기대보다
  적으면 '부분(반복·seed 행 k/K): '을 붙인다. 확인적 가설 L19 는 행이 기대보다 적으면 판정 불가다. 우세 또는 열세이면서 |Δ| < 0.5 cm 이면
  '통계적으로 구별되나 크기는 0.5 cm 미만'을 붙인다.
  채점 범위가 1 미만인 (단, 반복)은 집계에서 빼고 나머지 단을 계속 집계한다(실패 표와 meta 의 excluded 에 남기고 종료 코드를 1 로 둔다).
  --allow-partial 은 부분 범위의 (단, 반복)을 통계에 넣는 명시적 선택이다. 비유한 예측은 지역 단위로 뺀다(V-G 는 fold 가 지역이므로 한 지역의
  적합 실패가 다른 지역의 같은 키 행을 빼지 않는다). 전체 셀 풀은 모든 지역에서 유한한 행만 쓴다.

산출(<out-dir>, 기본 data/processed/lgx/ladder)
  shards/<tag>__<단>__r<반복>__f<fold>_{pred.npz, unit.json}   pred.npz = idx(자료 행 위치), loc_id, y, keys(JSON), P(float32, 키 × 셀)
  <tag>_metrics.csv, <tag>_contrasts.csv, <tag>_tests.csv(L19–L22), <tag>_timing.csv, <tag>_failed.csv, <tag>_meta.json, <tag>_count.csv

실행 환경(사용자 지시 2026-09-29)
  이 서버는 공유 서버다. 로컬에서는 --count-only, --write-cluster-map, --summarize-only 만 실행한다(스레드 1).
  학습을 하는 실행(스모크 포함)은 환경 변수 LG_RESCALE=1 또는 --allow-local 이 없으면 자료를 읽기 전에 거부한다.
  허용 표지가 없으면 스레드는 --threads 와 무관하게 1 이고 --summarize-only 의 재표집은 1,000회 이하로 낮춘다(표의 nboot 열에 적는다).
  V-G 단을 실행하려면 연도 정합 도일 표(--tdd-matched)가 있어야 한다. 없으면 실행 전에 중단한다.

실행(ROOT)
  적합 수:   python3 scripts/2_evaluation/h41_validation_ladder.py --count-only --threads 1
  군집 표:   python3 scripts/2_evaluation/h41_validation_ladder.py --write-cluster-map --threads 1
  스모크:    LG_RESCALE=1 python3 scripts/2_evaluation/h41_validation_ladder.py --smoke --workers 1 --threads 4
  본 실행:   LG_RESCALE=1 python3 scripts/2_evaluation/h41_validation_ladder.py --part all --workers 14 --threads 4 --resume --no-summarize
  집계:      python3 scripts/2_evaluation/h41_validation_ladder.py --summarize-only --threads 1

명세(implementation_spec_h41)와 다른 점
  1. 실행 거부가 더 엄격하다. 명세는 스모크를 거부 대상에서 뺐으나 이 파일은 스모크도 허용 표지를 요구한다(공유 서버 보호).
  2. 기본값은 계획서 §6A.8 과 명세를 따른다(--tag lgv, --out-dir data/processed/lgx/ladder). 구현 규칙의 'tag vl, out-dir data/processed/lgx'
     는 사전 등록 문서와 달라 쓰지 않았다.
  3. 버퍼 거리는 단위 구 좌표의 KD 트리(현 길이)로 구하고 경계 근처(상대 1e-6)만 haversine 으로 다시 계산한다. 명세의 '위도 차로 먼저
     거르는 haversine'은 단언(버퍼 안 학습 셀 0 개)에 쓴다. 두 계산은 수식이 같고 경로가 다르다.
  4. pred.npz 에 idx(자료 행 위치)와 y 를 더했다. 집계는 idx 로 모으고 loc_id 가 현재 자료와 같은지 확인한다.
  5. N1 의 채점 제외 방식 키(B:ku_raw#ex, P0@ku#ex)를 더했다(계획서 N1 의 '대체 방식과 채점 제외 방식을 모두 계산').
     '#ex' 키가 든 대비는 두 쪽 모두 유한한 셀로 제한한다.
  6. 군집 번호는 군집 평균 위도의 내림차순(같으면 경도 오름차순)으로 다시 매긴다. 명세는 번호 규칙을 정하지 않았다.
  7. 스모크 범위는 명세에 없다. 이 파일의 정의: 단 V-R, V-B, V-C100(채점 셀이 있는 앞 군집 3개), V-G, 지역 내 Lena, 반복 1, seed 1,
     학습기 설정은 본 실행과 같다. 스모크 집계는 채점 범위가 부분이어도 진행한다.
  8. --count-only 는 군집 대응표가 없으면 k-means 를 임시로 계산해 단위 수만 센다(표를 쓰지 않는다). 실행은 표만 읽는다.
  9. 시험 (e)는 h42 실행기가 아니라 h40.run_ctx 와 대조한다(작성 시점에 h42 가 없다. h42 의 시험 (a)가 h42 = h40 을 확인한다).
 10. L20 에서 지지·대체로 지지 조건을 만족하지 못한 경우와 L19, L21 의 그 밖 경우는 '지지하지 않음'으로 적는다. L20 의 '대체로 지지'에도
     V-C500 − V-R 의 CI 하한 > 0(두 가중)을 요구한다. L22 에서 판정 가능한 지역이 3개 미만이면 '부분(지역 k/3)'으로 적는다.
     이 문구와 조건은 계획서 §6A.5 의 개정 7 에 등록했다.
 11. 추가 인자: --within-regions, --nest-folds, --pool-retries, --allow-mixed-cfg, --allow-partial, --subregion-map, --tdd-matched.
 12. 지표에 R² 의 부트스트랩 CI 를 더했다. 환원 가능 오차 대비 비율은 하한(N2, h42 집계)이 필요해 이 파일에서 내지 않는다.
 13. 프로세스 풀은 한 번에 프로세스 수의 2배까지만 제출한다. 풀이 깨질 때 두 번 제출되어 있던 단위는 원인 후보로 빼서 마지막에 단위마다
     새 풀(프로세스 1개)에서 돌린다. --pool-retries 는 원인 후보를 가려내지 못한 풀 재생성의 상한이다.
 14. Holm 묶음은 둘이다(X3a: L19–L21 의 5지역 평균 주 대비, X3b: L22 의 기준 λ 지역 대비 3개). 판정 불가 행은 묶음에 넣지 않는다.

확인하지 못한 것
  로컬에서 학습, 스모크, 시험, 집계를 실행하지 않았다(py_compile, pyflakes, --count-only 만 실행). catboost 와 rf 의 적합 시간은 가정값이다.
  개정 7 의 집계 규칙(부분 범위 단의 제외, 지역 단위 비유한 처리, 부분 표기)과 풀 복구 경로는 시험(tests/test_h41_ladder.py 의 d, d2, g2, g4)으로만
  확인한다. 시험은 Rescale 사전 점검에서 실행한다. 풀 경로(--workers 2)는 HEAVY 시험 (p)가 확인한다.
  군집 30개의 셀 수 분포와 V-C500 에서 남는 학습 셀 수는 군집 대응표를 만든 뒤 --count-only 로 확인한다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time
import warnings
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path


def _peek_threads(default="4"):
    """numpy 를 부르기 전에 스레드 수를 정한다(--threads 를 미리 읽는다).
    허용 표지(환경 변수 LG_RESCALE=1 또는 --allow-local)가 없으면 --threads 와 무관하게 1 이다(공유 서버 보호)."""
    av = sys.argv
    if not (os.environ.get("LG_RESCALE", "") == "1" or "--allow-local" in av):
        return "1"
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return av[i + 1]
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return default


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _peek_threads())

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR, ROOT / "scripts" / "3_deep_learning", ROOT / "src"):   # 자기 디렉터리: 이 모듈을 import 해서 쓸 때 spawn 워커가 모듈을 찾는다
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_h40():
    """h40 을 파일 경로에서 읽어 sys.modules 에 등록한다. 이미 읽혀 있으면 그 모듈을 쓴다(h42, 시험과 같은 모듈을 공유한다)."""
    if "h40_label_grid" in sys.modules:
        return sys.modules["h40_label_grid"]
    spec = importlib.util.spec_from_file_location("h40_label_grid", ROOT / "scripts" / "3_deep_learning" / "h40_label_grid.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h40_label_grid"] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_h40()                                                     # 모듈 최상위에서 읽는다(spawn 워커가 주 모듈을 다시 읽을 때도 등록된다)

from polar.fidelity import CLIMATE, SOIL, CCI                                                         # noqa: E402
from polar.m1_core import eval_mask                                                                   # noqa: E402
from polar.m1_ext import haversine_km                                                                 # noqa: E402
from polar.m1_stats import strat, holm, MIN_BLOCKS_CI                                                 # noqa: E402
import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import seed_of                                                                   # noqa: E402

REG7 = ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]                # 채점 지역
MAIN5 = ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E"]                                          # 주 요약(5지역 층화 평균)
MAIN4 = list(H.MAIN4)                                                                                 # 보조 요약(주 4지역)
WITHIN_DEFAULT = "Alaska,Lena,Canada"
SCHEMES_DEFAULT = "V-R,V-P,V-S,V-B,V-C0,V-C100,V-C500,V-G"
KFOLD = ["V-R", "V-P", "V-S", "V-B"]
GROUPED = ["V-P", "V-S", "V-B"]
BASE = H.BASE_LEARNER                                                                                 # catboost_lo
LEARNERS_OK = ["catboost_lo", "catboost", "rf"]
LAM_BASE = float(H.LAM_BASE)                                                                          # 0.25
DROP_M = ["e5_tdd", "e5_sqrt_tdd", "cci_alt", "cci_valid"]                                            # D0m 이 빼는 물리 관련 입력
KEEP_M = [i for i, c in enumerate(H.FEATS) if c not in DROP_M]
PV_COLS = list(CLIMATE + SOIL + CCI)                                                                  # V-P 묶음 열(기후 8, 토양 9, CCI 2)
PV_SENTINEL = -9999.0
EARTH_KM = 6371.0
KM_PER_DEG = 2.0 * np.pi * EARTH_KM / 360.0
EX_ANCHORS = ["ku"]                                                                                   # 채점 제외 방식 키를 함께 내는 앵커
ANCHOR_KINDS = ["soil", "ku", "ed", "cci", "tddm"]                                                    # h42 의 ANCHORS 와 같다(tddm = 연도 정합 √TDD)
CONFIRMATORY = ("L19",)                                                                               # 이 파일이 판정하는 확인적 가설
NA_VERDICTS = ("행 없음", "판정 불가")
SMALL_EFFECT_CM = 0.5
SMALL_EFFECT_TXT = "통계적으로 구별되나 크기는 0.5 cm 미만"
LOCAL_NBOOT_MAX = 1000                                                                                # 허용 표지 없는 집계의 재표집 횟수 상한
PRIORITY = ["V-G", "V-R", "W2", "V-B", "V-C100", "V-C500", "V-C0", "V-P", "V-S", "W1"]               # 실행 순서(확인적 대비에 쓰는 단 먼저)
# 적합 시간 추정용 기준값(--count-only 전용, 결과에는 쓰지 않는다): 학습기 → (기준 행 수에서 1건의 시간 s, 기준 행 수, 행 수 지수).
# catboost_lo 는 LG 사전 점검(qOkSo) 실측이다. catboost 와 rf 는 재지 않은 가정값이다.
BASE_FIT_V = {"catboost_lo": (0.3, 16700, 0.6), "catboost": (4.0, 16700, 0.8), "rf": (6.0, 16700, 1.0)}
KEY_COLS = ["method", "learner", "seed", "lam"]
DIST_CACHE_MAX = 512    # 집계: 항의 부트스트랩 분포를 최근 이 수만큼만 기억한다(nboot 10,000 에서 약 80 MB)
FITTER = None           # 시험용: FitterV 를 대신할 클래스(학습 없는 대체 적합기)를 넣는다


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H41 검증 사다리(X3a)와 지역 내 블록 교차검증(X3b)")
    ap.add_argument("--part", choices=["pooled", "within", "all"], default="all")
    ap.add_argument("--schemes", default=SCHEMES_DEFAULT, help="통합 자료 단의 쉼표 목록. V-C<d> 의 d 는 버퍼 거리(km)")
    ap.add_argument("--within-regions", default=WITHIN_DEFAULT, help="지역 내 블록 교차검증 지역")
    ap.add_argument("--reps", type=int, default=3, help="K-fold 단(V-R, V-P, V-S, V-B)의 반복 수")
    ap.add_argument("--within-reps", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--learners", default="catboost_lo,catboost,rf", help="D0 와 RS(R1w)에 쓰는 학습기. 나머지 모델은 catboost_lo 만 쓴다")
    ap.add_argument("--lams", default="0.25,0.5,1.0")
    ap.add_argument("--workers", type=int, default=2, help="프로세스 수. 0 = 풀 없이 차례로. 본 실행 값은 Rescale 작업 명령에서만 준다")
    ap.add_argument("--threads", type=int, default=0, help="0 = 허용 표지가 있으면 4, 없으면 1. 허용 표지가 없으면 1 로 낮춘다")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--delta-eq", type=float, default=0.5)
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--cluster-map", default="lgx_cluster_map_v1.csv", help="블록 → 군집 대응표(상대 경로는 --data-dir 기준)")
    ap.add_argument("--write-cluster-map", action="store_true", help="군집 대응표를 만들고 끝낸다. 기존 표와 다르면 덮어쓰지 않는다")
    ap.add_argument("--tdd-matched", default="lgx_tdd_matched_v1.csv", help="연도 정합 도일 표(h42 --write-tdd-matched 가 만든다. 상대 경로는 "
                                                                            "--data-dir 기준). V-G 단의 N1 기준선 P0@tddm 에 쓴다")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv", help="h40 자료 적재에 넘기는 하위 지역 대응표(이 파일은 쓰지 않는다)")
    ap.add_argument("--out-dir", default="data/processed/lgx/ladder")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default="lgv")
    ap.add_argument("--resume", action="store_true", help="설정 해시가 같고 실패하지 않은 조각은 건너뛴다")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--count-only", action="store_true", help="학습 없이 단위 수와 적합 수만 출력")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--allow-local", action="store_true", help="학습을 하는 실행을 허용한다(환경 변수 LG_RESCALE=1 과 같은 효과)")
    ap.add_argument("--allow-partial", action="store_true", help="집계: 채점 셀이 모두 예측되지 않은 (단, 반복)도 통계에 넣는다"
                                                                 "(기본은 그 (단, 반복)만 집계에서 빼고 나머지를 집계한다)")
    ap.add_argument("--allow-mixed-cfg", action="store_true", help="집계: 조각 사이 설정 해시가 달라도 진행한다(기본은 오류)")
    ap.add_argument("--pool-retries", type=int, default=2)
    # 아래는 계획서 §6A 의 고정 설계값이다. 바꾸면 사전 등록에서 벗어난다.
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--buffer-km", type=float, default=100.0, help="V-G 와 지역 내 원천의 대상 경계 버퍼(h40 과 같다)")
    ap.add_argument("--n-clusters", type=int, default=30)
    ap.add_argument("--nest-folds", type=int, default=4)
    return finalize(ap.parse_args(argv))


def scheme_kind(s):
    if s in KFOLD:
        return "kfold"
    if s == "V-G":
        return "region"
    if s.startswith("V-C"):
        try:
            float(s[3:])
        except ValueError:
            raise SystemExit(f"알 수 없는 단 {s}(V-C<거리 km> 형식이어야 한다)")
        return "cluster"
    if s.startswith("W2-") or s.startswith("W1-"):
        return "within"
    raise SystemExit(f"알 수 없는 단 {s}")


def cluster_buffer(s):
    return float(s[3:])


def h40_argv(a):
    """h40 형식의 인자. 자료 적재(get_data), CatBoost 설정(cb_fit), Fitter 에 넘긴다."""
    return ["--part", "cpu", "--threads", str(int(a.threads)), "--data-dir", str(a.data_dir), "--out-dir", str(a.out_dir),
            "--buffer-km", str(float(a.buffer_km)), "--kappa", str(float(a.kappa)), "--subregion-map", str(a.subregion_map)]


def finalize(a):
    a.TAG = a.tag + ("_smoke" if a.smoke else "")
    a.PERMIT = bool(a.allow_local) or os.environ.get("LG_RESCALE", "") == "1"
    a.threads_asked = int(a.threads)
    a.threads = (int(a.threads) if int(a.threads) > 0 else 4) if a.PERMIT else 1          # 허용 표지가 없으면 1(학습 없는 로컬 확인 전용)
    a.SCHEMES = [v.strip() for v in a.schemes.split(",") if v.strip()]
    for s in a.SCHEMES:
        if scheme_kind(s) == "within":
            raise SystemExit("지역 내 단은 --part within 과 --within-regions 로 지정한다")
    a.WREGIONS = [v.strip() for v in a.within_regions.split(",") if v.strip()]
    a.REPS = list(range(1, a.reps + 1)); a.WREPS = list(range(1, a.within_reps + 1))
    a.SEEDS = list(range(a.seeds)); a.LAMS = [float(v) for v in a.lams.split(",") if v.strip()]
    a.LEARNERS = [v.strip() for v in a.learners.split(",") if v.strip()]
    bad = [v for v in a.LEARNERS if v not in LEARNERS_OK]
    if bad:
        raise SystemExit(f"쓸 수 없는 학습기: {bad}(가능: {LEARNERS_OK})")
    a.SMOKE_CLUSTERS = 0
    if a.smoke:                                                      # 스모크: 경로 확인용. 학습기 설정은 본 실행과 같다(시간 측정 겸용)
        a.REPS = [1]; a.WREPS = [1]; a.SEEDS = [0]; a.SMOKE_CLUSTERS = 3
        if a.schemes == SCHEMES_DEFAULT:
            a.SCHEMES = ["V-R", "V-B", "V-C100", "V-G"]
        if a.within_regions == WITHIN_DEFAULT:
            a.WREGIONS = ["Lena"]
        a.allow_partial = True
    a.OUT = (ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir)
    a.PROC = (ROOT / a.data_dir) if not os.path.isabs(a.data_dir) else Path(a.data_dir)
    a.SHARDS = a.OUT / "shards"
    a.CMAP = Path(a.cluster_map) if os.path.isabs(a.cluster_map) else a.PROC / a.cluster_map
    a.TDDM = Path(a.tdd_matched) if os.path.isabs(a.tdd_matched) else a.PROC / a.tdd_matched
    a.HA = H.parse_args(h40_argv(a))
    return a


def schemes_to_run(a):
    out = []
    if a.part in ("pooled", "all"):
        out += list(a.SCHEMES)
    if a.part in ("within", "all"):
        out += [f"W2-{r}" for r in a.WREGIONS] + [f"W1-{r}" for r in a.WREGIONS]
    return out


def run_permitted(a):
    """학습을 하는 실행의 허용 여부. Rescale 작업 명령에서 LG_RESCALE=1 또는 --allow-local 을 준다."""
    return bool(a.allow_local) or os.environ.get("LG_RESCALE", "") == "1"


# ================================================================ 분할 규칙(순수 함수)
def folds_random(n, k, seed):
    """셀 무작위 K-fold. 순열에서의 위치 mod k 가 fold 번호다."""
    perm = np.random.RandomState(int(seed)).permutation(int(n))
    f = np.empty(int(n), np.int64)
    f[perm] = np.arange(int(n)) % int(k)
    return f


def folds_grouped(groups, k, seed):
    """묶음 단위 K-fold. 묶음(정렬된 고유값)을 순열한 뒤 셀 수가 가장 적은 fold 에 차례로 넣는다(동률이면 번호가 작은 fold)."""
    ug, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    order = np.random.RandomState(int(seed)).permutation(len(ug))
    load = np.zeros(int(k), np.int64); gf = np.empty(len(ug), np.int64)
    for j in order:
        f = int(np.argmin(load))                                     # argmin 은 최솟값이 여럿이면 가장 앞(작은 번호)을 준다
        gf[j] = f; load[f] += cnt[j]
    return gf[inv.ravel()]


def groups_pv(df):
    """거친 공변량 묶음: 기후 8·토양 9·CCI 2열을 소수 6자리로 반올림한 값이 같은 셀. 결측은 표지값으로 둔다."""
    K = df[PV_COLS].astype(float).round(6).fillna(PV_SENTINEL)
    return K.groupby(PV_COLS, sort=True).ngroup().values.astype(np.int64)


def groups_s(lat, lon, deg=0.05):
    """0.05° 격자 묶음 (floor(lat/deg), floor(lon/deg))."""
    a = np.floor(np.asarray(lat, float) / deg).astype(np.int64); b = np.floor(np.asarray(lon, float) / deg).astype(np.int64)
    return a * 100000 + b


def unit_xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def min_dist_km(lat_q, lon_q, lat_r, lon_r, d_ref=None):
    """각 질의 셀에서 기준 셀 집합까지의 최소 대권 거리(km). 단위 구 좌표의 최근접 현 길이를 호 길이로 바꾼다(haversine 과 같은 값).
    d_ref 를 주면 그 거리의 경계 근처(상대 1e-6) 셀은 최근접 기준 셀까지의 haversine 으로 다시 계산한다."""
    from scipy.spatial import cKDTree
    lat_q, lon_q = np.asarray(lat_q, float), np.asarray(lon_q, float)
    lat_r, lon_r = np.asarray(lat_r, float), np.asarray(lon_r, float)
    if len(lat_q) == 0:
        return np.zeros(0)
    if len(lat_r) == 0:
        return np.full(len(lat_q), np.inf)
    ch, j = cKDTree(unit_xyz(lat_r, lon_r)).query(unit_xyz(lat_q, lon_q), k=1)
    d = 2.0 * EARTH_KM * np.arcsin(np.clip(ch / 2.0, 0.0, 1.0))
    if d_ref is not None and d_ref > 0:
        near = np.abs(d - d_ref) <= 1e-6 * max(float(d_ref), 1.0)
        if near.any():
            d[near] = haversine_km(lat_q[near], lon_q[near], lat_r[j[near]], lon_r[j[near]])
    return d


def count_within_km(lat_q, lon_q, lat_r, lon_r, d_km, chunk=1_000_000):
    """기준 셀 d_km 안(거리 < d_km)에 있는 질의 셀의 수. 위도 차로 먼저 거르고 haversine 을 전수 계산한다(단언용, 좌표만 쓴다)."""
    lat_q, lon_q = np.asarray(lat_q, float), np.asarray(lon_q, float)
    lat_r, lon_r = np.asarray(lat_r, float), np.asarray(lon_r, float)
    if d_km <= 0 or len(lat_q) == 0 or len(lat_r) == 0:
        return 0
    band = d_km / KM_PER_DEG * (1.0 + 1e-6) + 1e-9                     # 위도 차가 이보다 크면 거리가 d_km 이상이다
    cand = np.where((lat_q >= lat_r.min() - band) & (lat_q <= lat_r.max() + band))[0]
    step = max(1, int(chunk // max(len(lat_r), 1)))
    n = 0
    for k in range(0, len(cand), step):
        c = cand[k:k + step]
        d = haversine_km(lat_q[c][:, None], lon_q[c][:, None], lat_r[None, :], lon_r[None, :])
        n += int((d.min(1) < d_km).sum())
    return n


def make_cluster_table(df, n_clusters):
    """블록 → 군집 대응표. 블록 중심 X = [lat, lon·cos(lat)], KMeans(n_init 10, random_state 0), sample_weight = sqrt(셀 수). 라벨 미사용.
    군집 번호는 군집 평균 위도의 내림차순(같으면 경도 오름차순)이다."""
    from sklearn.cluster import KMeans
    bt = df.groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_cells=("lat", "size")).reset_index()
    k = int(min(n_clusters, len(bt)))
    X = np.c_[bt.lat.values, bt.lon.values * np.cos(np.radians(bt.lat.values))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=np.sqrt(bt.n_cells.values))
    lab = km.labels_
    cen = [(-float(bt.lat.values[lab == c].mean()), float(bt.lon.values[lab == c].mean()), c) for c in range(k)]
    rank = {c: r for r, (_, _, c) in enumerate(sorted(cen))}
    bt["cluster"] = [rank[c] for c in lab]
    return bt[["block", "cluster", "n_cells", "lat", "lon"]].sort_values(["cluster", "block"]).reset_index(drop=True)


# ================================================================ 자료
class Cells:
    """셀 집합의 배열 묶음. X = 입력 25열(float32), Xk = [X, p4_ku, p2_edaphic]. tddm = 연도 정합 √TDD(없으면 NaN)."""

    def __init__(self, X, y, s, ku, ed, cci, soil, tddm=None):
        self.X = np.asarray(X, np.float32); self.y = np.asarray(y, float); self.s = np.asarray(s, float)
        self.ku = np.asarray(ku, float); self.ed = np.asarray(ed, float); self.cci = np.asarray(cci, float); self.soil = np.asarray(soil, float)
        self.tddm = np.full(len(self.y), np.nan) if tddm is None else np.asarray(tddm, float)

    def __len__(self):
        return len(self.y)

    @property
    def Xk(self):
        return np.c_[self.X, self.ku.astype(np.float32), self.ed.astype(np.float32)].astype(np.float32)

    def anchor(self, kind):
        return dict(soil=self.soil, ku=self.ku, ed=self.ed, cci=self.cci, tddm=self.tddm)[kind]


class VData:
    """자료, 채점 집합, 분할 번호(프로세스마다 한 번 만든다). D 는 h40.Data(원천 정의 source_idx 를 그대로 쓴다)."""

    def __init__(self, a, D=None):
        self.D = H.get_data(a.HA) if D is None else D
        df = self.D.df
        self.df = df; self.N = int(len(df))
        self.y = df.y.values.astype(float); self.s = df.s.values.astype(float)
        self.X = np.asarray(df[H.FEATS].values, np.float32)
        self.ku = df.p4_ku.values.astype(float); self.ed = df.p2_edaphic.values.astype(float)
        self.cci = df.cci_alt.values.astype(float); self.soil = df.e5_sqrt_tdd_soil.values.astype(float)
        self.lat = df.lat.values.astype(float); self.lon = df.lon.values.astype(float)
        self.block = df.block.values.astype(np.int64); self.macro = np.asarray(df.macro.values, object)
        self.loc_id = df.loc_id.values.astype(np.int64)
        self.tddm = np.full(self.N, np.nan); self.tdd_src = "none"                                    # 연도 정합 √TDD(표가 없으면 NaN, 앵커 tddm 을 건너뛴다)
        tp = getattr(a, "TDDM", None)
        if tp is not None and Path(tp).exists():
            t = pd.read_csv(tp, usecols=["loc_id", "tdd_matched"]).drop_duplicates("loc_id").set_index("loc_id")
            miss = int((~np.isin(self.loc_id, t.index.values)).sum())
            if miss:
                raise SystemExit(f"연도 정합 도일 표 {tp} 에 없는 loc_id 가 {miss}개다. 자료 판과 표를 확인한다")
            v = t.tdd_matched.reindex(self.loc_id).values.astype(float)
            with np.errstate(invalid="ignore"):
                self.tddm = np.sqrt(np.where(v > 0, v, np.nan))
            self.tdd_src = f"{Path(tp).name}:{_file_sha(tp)}"
        self.gid = np.array([f"{m}|{b}" for m, b in zip(self.macro, self.block)], dtype=object)       # 블록 식별자 (macro, block)
        self.trainable = np.isfinite(self.y) & np.isfinite(self.s)                                    # h40.Ctx 와 같은 규칙
        self.score = np.asarray(eval_mask(df), bool) & np.isin(self.macro, REG7)
        self._fold = {}; self._grp = {}; self._cluster = None; self.cluster_src = ""; self._buf_ok = set()

    def take(self, idx):
        idx = np.asarray(idx, np.int64)
        return Cells(self.X[idx], self.y[idx], self.s[idx], self.ku[idx], self.ed[idx], self.cci[idx], self.soil[idx], self.tddm[idx])

    def groups(self, scheme):
        if scheme not in self._grp:
            if scheme == "V-P":
                self._grp[scheme] = groups_pv(self.df)
            elif scheme == "V-S":
                self._grp[scheme] = groups_s(self.lat, self.lon)
            elif scheme == "V-B":
                self._grp[scheme] = self.block.copy()
            else:
                raise ValueError(scheme)
        return self._grp[scheme]

    def cluster_of(self, a):
        """셀 → 군집 번호. 실행은 대응표만 읽는다. --count-only 는 표가 없으면 임시로 계산한다(표를 쓰지 않는다)."""
        if self._cluster is None:
            if a.CMAP.exists():
                tab = pd.read_csv(a.CMAP); self.cluster_src = f"map:{a.CMAP.name}"
            elif a.count_only:
                tab = make_cluster_table(self.df, a.n_clusters); self.cluster_src = "kmeans(임시, 대응표 없음)"
                print(f"  [warn] 군집 대응표 {a.CMAP} 가 없다. 단위 수를 세기 위해 k-means 를 임시로 계산한다(표를 쓰지 않는다)", flush=True)
            else:
                raise SystemExit(f"군집 대응표 {a.CMAP} 가 없다. --write-cluster-map 으로 먼저 만든다(실행은 표만 읽는다)")
            nk = int(tab.cluster.nunique())
            if nk != int(min(a.n_clusters, tab.block.nunique())):
                raise SystemExit(f"군집 대응표의 군집 수 {nk} 가 --n-clusters {a.n_clusters} 와 다르다")
            b2c = {int(b): int(c) for b, c in zip(tab.block, tab.cluster)}
            miss = [int(b) for b in np.unique(self.block) if int(b) not in b2c]
            if miss:
                raise SystemExit(f"군집 대응표 {a.CMAP} 에 없는 블록이 {len(miss)}개다. 자료 판과 대응표를 확인한다")
            self._cluster = np.array([b2c[int(b)] for b in self.block], np.int64)
        return self._cluster

    def fold_of(self, a, scheme, rep):
        """셀의 fold 번호(길이 N). 단에 속하지 않는 셀은 −1 이다."""
        key = (scheme, int(rep))
        if key in self._fold:
            return self._fold[key]
        kind = scheme_kind(scheme)
        if scheme == "V-R":
            f = folds_random(self.N, a.folds, seed_of("lgv", "V-R", rep))
        elif scheme in GROUPED:
            f = folds_grouped(self.groups(scheme), a.folds, seed_of("lgv", scheme, rep))
        elif kind == "cluster":
            f = self.cluster_of(a)
        elif kind == "region":
            pos = {r: i for i, r in enumerate(REG7)}
            f = np.array([pos.get(m, -1) for m in self.macro], np.int64)
        else:                                                         # 지역 내: W2 와 W1 은 같은 fold 를 쓴다
            R = scheme[3:]
            t_idx = self.D.target_idx(R)
            f = np.full(self.N, -1, np.int64)
            f[t_idx] = folds_grouped(self.block[t_idx], a.folds, seed_of("lgv", "W", R, rep))
            self._fold[(("W1-" if scheme.startswith("W2-") else "W2-") + R, int(rep))] = f
        self._fold[key] = f
        return f

    def score_of(self, scheme):
        if scheme_kind(scheme) == "within":
            return self.score & (self.macro == scheme[3:])
        return self.score


_VD = None


def get_vdata(a):
    global _VD
    sig = (str(a.PROC), float(a.buffer_km), str(a.CMAP), int(a.folds), int(a.n_clusters), str(a.TDDM))
    if _VD is None or getattr(_VD, "sig", None) != sig:
        _VD = VData(a); _VD.sig = sig
    return _VD


# ================================================================ 작업 단위의 셀
def reps_of(a, scheme):
    k = scheme_kind(scheme)
    return list(a.REPS) if k == "kfold" else list(a.WREPS) if k == "within" else [1]


def enumerate_units(a, V):
    """작업 단위 (단, 반복, fold) 목록과 건너뛴 단위. 채점 셀이 없는 fold 는 실행하지 않는다."""
    units, skipped = [], []
    for s in schemes_to_run(a):
        kind = scheme_kind(s)
        if kind == "within" and s[3:] not in set(V.macro):
            raise SystemExit(f"지역 내 단의 지역 {s[3:]} 이 자료에 없다")
        sc = V.score_of(s)
        for rep in reps_of(a, s):
            f = V.fold_of(a, s, rep)
            n_c = 0
            for k in sorted(int(v) for v in np.unique(f[f >= 0])):
                n_te = int(((f == k) & sc).sum())
                if n_te == 0:
                    skipped.append(dict(scheme=s, rep=int(rep), fold=k, status="no_eval", n_held=int((f == k).sum()))); continue
                if kind == "cluster" and a.SMOKE_CLUSTERS and n_c >= a.SMOKE_CLUSTERS:
                    skipped.append(dict(scheme=s, rep=int(rep), fold=k, status="smoke_limit", n_held=int((f == k).sum()))); continue
                n_c += 1
                units.append((s, int(rep), k))
    return units, skipped


def unit_priority(u):
    s, rep, k = u
    head = s[:2] if scheme_kind(s) == "within" else s
    return (PRIORITY.index(head) if head in PRIORITY else len(PRIORITY), s, int(rep), int(k))


def unit_name(u):
    return f"{u[0]}|r{u[1]}|f{u[2]}"


def unit_cells(a, V, scheme, rep, fold, check=True):
    """작업 단위의 학습 셀과 채점 셀(자료 행 위치). check = True 이면 버퍼 단언을 haversine 전수 계산으로 한다.
    반환 dict(tr, te, held, src, trw, info). 지역 내 단이 아니면 src 와 trw 는 None 이다."""
    kind = scheme_kind(scheme); f = V.fold_of(a, scheme, rep)
    info = dict(buffer_km=0.0, n_buffer_excluded=0, region="")
    src = trw = None
    if kind == "within":
        R = scheme[3:]
        t_idx, _, s_idx, comp = V.D.source_idx(R, "x")
        src = np.asarray(s_idx, np.int64); src = src[V.trainable[src]]
        held = np.where(f == fold)[0]
        te = held[V.score[held] & (V.macro[held] == R)]
        trw = np.where((f >= 0) & (f != fold) & V.trainable)[0]
        assert len(np.intersect1d(src, t_idx)) == 0, f"{scheme}: 원천에 대상 지역 셀이 있다"
        assert len(np.intersect1d(V.block[trw], V.block[held])) == 0, f"{scheme}: 같은 블록이 학습 fold 와 채점 fold 에 걸쳐 있다"
        tr = np.concatenate([src, trw]) if scheme.startswith("W2-") else trw
        info.update(buffer_km=float(a.buffer_km) if scheme.startswith("W2-") else 0.0, region=R,
                    n_buffer_excluded=int(comp["n_buffer_excluded"]) if scheme.startswith("W2-") else 0)
        if check and scheme.startswith("W2-") and a.buffer_km > 0 and ("src", R) not in V._buf_ok:
            nin = count_within_km(V.lat[src], V.lon[src], V.lat[t_idx], V.lon[t_idx], a.buffer_km)
            assert nin == 0, f"{scheme}: 버퍼 {a.buffer_km} km 안의 원천 셀이 {nin}개다"
            V._buf_ok.add(("src", R))
    elif kind == "region":
        R = REG7[fold]
        t_idx, _, s_idx, comp = V.D.source_idx(R, "x")
        tr = np.asarray(s_idx, np.int64); tr = tr[V.trainable[tr]]
        held = np.asarray(t_idx, np.int64)
        te = held[V.score[held]]
        info.update(buffer_km=float(a.buffer_km), n_buffer_excluded=int(comp["n_buffer_excluded"]), region=R)
        if check and a.buffer_km > 0 and ("src", R) not in V._buf_ok:
            nin = count_within_km(V.lat[tr], V.lon[tr], V.lat[held], V.lon[held], a.buffer_km)
            assert nin == 0, f"{scheme} {R}: 버퍼 {a.buffer_km} km 안의 학습 셀이 {nin}개다"
            V._buf_ok.add(("src", R))
    else:
        held = np.where(f == fold)[0]
        cand = np.where((f != fold) & V.trainable)[0]
        d = cluster_buffer(scheme) if kind == "cluster" else 0.0
        if d > 0:
            dist = min_dist_km(V.lat[cand], V.lon[cand], V.lat[held], V.lon[held], d_ref=d)
            tr = cand[dist >= d]
            info.update(buffer_km=float(d), n_buffer_excluded=int(len(cand) - len(tr)))
            if check:
                nin = count_within_km(V.lat[tr], V.lon[tr], V.lat[held], V.lon[held], d)
                assert nin == 0, f"{scheme} f{fold}: 버퍼 {d} km 안의 학습 셀이 {nin}개다"
        else:
            tr = cand
        te = held[V.score[held]]
        if scheme in GROUPED:
            g = V.groups(scheme)
            assert len(np.intersect1d(g[tr], g[held])) == 0, f"{scheme}: 같은 묶음이 학습과 채점에 걸쳐 있다"
        if kind == "cluster":
            c = V.cluster_of(a)
            assert len(np.intersect1d(c[tr], c[held])) == 0, f"{scheme}: 제외 군집의 셀이 학습에 있다"
    assert len(np.intersect1d(tr, te)) == 0 and len(np.intersect1d(tr, held)) == 0, f"{scheme} r{rep} f{fold}: 학습 셀과 채점 셀이 겹친다"
    assert np.all(V.trainable[tr]) and np.all(V.score[te]), f"{scheme}: 학습 셀 또는 채점 셀의 조건이 맞지 않는다"
    mc = Counter(str(m) for m in V.macro[tr])
    info.update(n_train=int(len(tr)), n_test=int(len(te)), n_held=int(len(held)), train_macro=dict(mc),
                n_src=int(len(src)) if src is not None else 0, n_trw=int(len(trw)) if trw is not None else 0,
                train_lt_1000=bool(len(tr) < 1000), nb_test=int(len(np.unique(V.gid[te]))))
    return dict(tr=tr, te=te, held=held, src=src, trw=trw, info=info)


# ================================================================ 학습기
class FitterV(H.Fitter):
    """h40.Fitter 에 catboost(고용량)와 rf 를 더한다. catboost_lo 는 h40 의 경로(cb_fit)를 그대로 쓴다.
    적합 1건의 학습 행 수는 info['n_train'] 으로 받는다."""

    def _count(self, axis, learner, info):
        k = f"{axis}|{learner}|{(info or {}).get('method', '')}"
        self.n[axis] += 1; self.nd[k] += 1; self.last_flag = ""
        nr = max(int((info or {}).get("n_train", 0)), 1)
        self.rowsd[k] += nr
        b, ref, ex = BASE_FIT_V.get(learner, (1.0, 16700, 1.0))
        self.estd[k] += float(b * (nr / ref) ** ex)
        return k

    def _fit(self, learner, axis, build, seed, preds, info, init, iters, cont):
        if learner not in ("catboost", "rf"):
            return super()._fit(learner, axis, build, seed, preds, info, init, iters, cont)
        Xtr, ytr, w = build()
        self._trace(dict(info or {}, learner=learner, axis=axis, seed=seed), Xtr, ytr, w)
        if not np.all(np.isfinite(np.asarray(ytr, float))):
            raise ValueError("학습 목표에 비유한 값이 있다")
        th = int(self.args.threads)
        if learner == "catboost":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=int(seed), verbose=0,
                                  allow_writing_files=False, thread_count=th)
            m.fit(Xtr, ytr, sample_weight=w)
            return m, [np.asarray(m.predict(p, thread_count=th), float) if len(p) else np.zeros(0) for p in preds]
        from sklearn.ensemble import RandomForestRegressor
        med = H._prep_stats(Xtr)[0]                                   # 결측 대체 중앙값은 학습 행렬에서만 구한다(표준화 없음)

        def fill(X):
            X = np.asarray(X, float)
            return np.where(np.isnan(X), med, X)
        m = RandomForestRegressor(n_estimators=500, max_features=1.0 / 3.0, min_samples_leaf=5, random_state=int(seed), n_jobs=th)
        m.fit(fill(Xtr), np.asarray(ytr, float), sample_weight=w)
        return m, [np.asarray(m.predict(fill(p)), float) if len(p) else np.zeros(0) for p in preds]


def make_fitter(a, dry=False):
    return (FITTER or FitterV)(a.HA, dry)


# ================================================================ 모델
def shrink(E_ls, m, E0, kappa):
    """κ 수축 계수(h40 의 E1 과 같은 식). E_ls 가 비유한이거나 m = 0 이면 E0."""
    if m <= 0 or not np.isfinite(E_ls):
        return float(E0)
    return float((m * E_ls + kappa * E0) / (m + kappa))


def _ls_affine(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    b, a0 = np.polyfit(p[m], y[m], 1)
    return float(a0), float(b)


def _ls_scale(p, y):
    m = np.isfinite(p) & np.isfinite(y) & (p > 0)
    return float((p[m] @ y[m]) / (p[m] @ p[m])) if m.sum() >= 1 else np.nan


def anchor_fill(b_tr, s_tr, b_te, s_te):
    """앵커의 비유한 또는 0 이하 셀을 ρ·s 로 바꾼다. ρ = 학습 셀의 b/s 중앙값(라벨 미사용)."""
    b_tr, b_te = np.asarray(b_tr, float), np.asarray(b_te, float)
    ok = np.isfinite(b_tr) & (b_tr > 0) & np.isfinite(s_tr) & (s_tr > 0)
    rho = float(np.median(b_tr[ok] / s_tr[ok])) if ok.any() else np.nan
    ftr = np.where(np.isfinite(b_tr) & (b_tr > 0), b_tr, rho * s_tr)
    fte = np.where(np.isfinite(b_te) & (b_te > 0), b_te, rho * s_te)
    return ftr, fte, rho


def baselines_n1(TR, TE, E):
    """N1 물리 기준선(해석 계산). 계수는 학습 집합에서만 구한다."""
    out, co = {}, {}
    p0 = {}
    for kind in ANCHOR_KINDS:
        raw = TE.anchor(kind)
        ftr, fte, rho = anchor_fill(TR.anchor(kind), TR.s, raw, TE.s)
        if not np.isfinite(rho):                                       # 학습 셀에 쓸 수 있는 값이 없다(예: 연도 정합 도일 표 없음). 그 앵커의 키를 내지 않는다
            co[kind] = dict(skipped="학습 셀에 쓸 수 있는 값이 없다")
            continue
        c0 = _ls_scale(ftr, TR.y)
        p0[kind] = c0 * fte
        co[kind] = dict(c0=c0, rho=rho, n_fill_train=int((~(np.isfinite(TR.anchor(kind)) & (TR.anchor(kind) > 0))).sum()),
                        n_fill_test=int((~(np.isfinite(raw) & (raw > 0))).sum()))
        out[(f"P0@{kind}", "none", -1, 0.0)] = p0[kind]
        if kind not in ("soil", "tddm"):                               # 무보정 값은 ALT 단위의 앵커(ed, ku, cci)에만 둔다(계획서 N1)
            out[(f"B:{kind}_raw", "none", -1, 0.0)] = fte
        if kind in EX_ANCHORS:
            ex = np.isfinite(raw) & (raw > 0)
            out[(f"B:{kind}_raw#ex", "none", -1, 0.0)] = np.where(ex, raw, np.nan)
            out[(f"P0@{kind}#ex", "none", -1, 0.0)] = np.where(ex, c0 * raw, np.nan)
    a0, b0 = _ls_affine(TR.s, TR.y)
    out[("B:s_aff", "none", -1, 0.0)] = a0 + b0 * TE.s
    a1, b1 = _ls_affine(TR.cci, TR.y)
    out[("B:cci_aff", "none", -1, 0.0)] = a1 + b1 * TE.cci
    if "ku" in p0 and "cci" in p0:
        out[("B:ens", "none", -1, 0.0)] = (E * TE.s + p0["ku"] + p0["cci"]) / 3.0
    co.update(s_aff=dict(a=a0, b=b0), cci_aff=dict(a=a1, b=b1))
    return out, co


def _log_ratio(y, d):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(np.asarray(y, float)) - np.log(np.asarray(d, float))


def models_pooled(F, a, TR, TE, axis, sfx="", extended=True, n1=False):
    """학습 집합 하나(TR)에 적용하는 모델 집합. extended = False 이면 D0m 과 RM0 을 뺀다(W1). 반환 (예측 dict, 기록 dict)."""
    out = {}
    nt = len(TR)
    E = H.ls_E(TR.y, TR.s)
    aB = E * TE.s
    out[("PS" + sfx, "none", -1, 0.0)] = aB
    rec = dict(E=float(E), n_train=int(nt))
    r_tr = TR.y - E * TR.s
    for lr in a.LEARNERS:
        for seed in a.SEEDS:
            _, (p,) = F.fit(lr, axis, lambda: (TR.X, TR.y, None), seed, [TE.X], info=dict(method="D0" + sfx, n_train=nt))
            out[("D0" + sfx, lr, seed, 1.0)] = p
            _, (g,) = F.fit(lr, axis, lambda: (TR.X, r_tr, None), seed, [TE.X], info=dict(method="RS" + sfx, n_train=nt))
            for lam in a.LAMS:
                out[("RS" + sfx, lr, seed, lam)] = aB + lam * g
    if BASE in a.LEARNERS:
        z = _log_ratio(TR.y, TR.s)
        okz = np.isfinite(z)
        rec["n_z_dropped"] = int((~okz).sum())
        Xk_tr, Xk_te = TR.Xk, TE.Xk
        for seed in a.SEEDS:
            if extended:
                _, (p,) = F.fit(BASE, axis, lambda: (TR.X[:, KEEP_M], TR.y, None), seed, [TE.X[:, KEEP_M]],
                                info=dict(method="D0m" + sfx, n_train=nt))
                out[("D0m" + sfx, BASE, seed, 1.0)] = p
            _, (p,) = F.fit(BASE, axis, lambda: (Xk_tr, TR.y, None), seed, [Xk_te], info=dict(method="F1k" + sfx, n_train=nt))
            out[("F1k" + sfx, BASE, seed, 1.0)] = p
            _, (zh,) = F.fit(BASE, axis, lambda: (TR.X[okz], z[okz], None), seed, [TE.X], info=dict(method="V2" + sfx, n_train=int(okz.sum())))
            zh = np.clip(zh, float(z[okz].min()), float(z[okz].max())) if okz.any() else np.full(len(TE), np.nan)
            out[("V2" + sfx, BASE, seed, 0.0)] = np.exp(zh) * TE.s
            if extended:
                for lam in a.LAMS:
                    if lam != 1.0:                                     # λ = 1.0 은 V2 와 같다
                        out[("RM0" + sfx, BASE, seed, lam)] = aB * np.exp(lam * (zh - np.log(E)))
    if n1:
        b, co = baselines_n1(TR, TE, E)
        out.update(b); rec["n1"] = co
    return out, rec


def nest_select(F, a, S, T, blkT, seed_parts, E0, r0_S):
    """R1w@nest 의 λ 선택. 지역 내 학습 fold(T)의 블록을 묶음으로 나눠 묶음마다 E_w 와 g 를 학습 쪽으로 다시 구하고,
    제외 묶음의 SSE 합이 가장 작은 λ 를 고른다(seed 0 적합). 채점 fold 의 셀은 인자로 받지 않는다."""
    k = int(min(a.nest_folds, len(np.unique(blkT))))
    if k < 2:
        return LAM_BASE, "blocks<2", {}
    gf = folds_grouped(blkT, k, seed_of(*seed_parts))
    err = {float(lam): 0.0 for lam in a.LAMS}
    for j in range(k):
        te = gf == j; tr = ~te
        if not te.any() or not tr.any():
            continue
        Ew = shrink(H.ls_E(T.y[tr], T.s[tr]), int(tr.sum()), E0, a.kappa)
        Xtr = np.vstack([S.X, T.X[tr]]); ytr = np.concatenate([r0_S, T.y[tr] - Ew * T.s[tr]])
        _, (g,) = F.fit(BASE, "w2_nest", lambda: (Xtr, ytr, None), a.SEEDS[0], [T.X[te]],
                        info=dict(method="R1w@nest", n_train=len(ytr), cv_fold=int(j)))
        for lam in a.LAMS:
            err[float(lam)] += float(np.sum((Ew * T.s[te] + lam * g - T.y[te]) ** 2))
    fin = [float(lam) for lam in a.LAMS if np.isfinite(err[float(lam)])]
    if not fin:
        return LAM_BASE, "cv failed", err
    return min(fin, key=lambda lam: (err[lam], lam)), "", err


def models_w2(F, a, S, T, TE, blkT, seed_parts, axis="w2"):
    """지역 내 w2(학습 = 타 지역 원천 S ∪ 지역 내 학습 fold T). 학습 행렬의 행 순서는 [S; T]다. 반환 (예측 dict, 기록 dict)."""
    out = {}
    E0 = H.ls_E(S.y, S.s)
    m = len(T)
    E_ls = H.ls_E(T.y, T.s) if m else np.nan
    E_w = shrink(E_ls, m, E0, a.kappa)
    E_2 = float(E_ls) if np.isfinite(E_ls) else float(E0)
    out[("P0", "none", -1, 0.0)] = E0 * TE.s
    out[("P1w", "none", -1, 0.0)] = E_w * TE.s
    out[("P2w", "none", -1, 0.0)] = E_2 * TE.s
    X = np.vstack([S.X, T.X]); y = np.concatenate([S.y, T.y]); nt = len(y)
    r0_S = S.y - E0 * S.s
    r0 = np.concatenate([r0_S, T.y - E0 * T.s])
    r1 = np.concatenate([r0_S, T.y - E_w * T.s])
    g1 = {}
    for lr in a.LEARNERS:
        for seed in a.SEEDS:
            _, (p,) = F.fit(lr, axis, lambda: (X, y, None), seed, [TE.X], info=dict(method="D0", n_train=nt))
            out[("D0", lr, seed, 1.0)] = p
            _, (g,) = F.fit(lr, axis, lambda: (X, r1, None), seed, [TE.X], info=dict(method="R1w", n_train=nt))
            for lam in a.LAMS:
                out[("R1w", lr, seed, lam)] = E_w * TE.s + lam * g
            if lr == BASE:
                g1[seed] = g
    rec = dict(E0=float(E0), E_w=float(E_w), E_ls=float(E_ls) if np.isfinite(E_ls) else None, n_train=int(nt), n_src=int(len(S)), n_trw=int(m))
    if BASE in a.LEARNERS:
        Xk = np.vstack([S.Xk, T.Xk]); Xk_te = TE.Xk
        z = _log_ratio(y, np.concatenate([S.s, T.s])); okz = np.isfinite(z)
        h = np.concatenate([_log_ratio(S.y, E0 * S.s), _log_ratio(T.y, E_w * T.s)]); okh = np.isfinite(h)
        rec.update(n_z_dropped=int((~okz).sum()), n_h_dropped=int((~okh).sum()))
        for seed in a.SEEDS:
            _, (p,) = F.fit(BASE, axis, lambda: (Xk, y, None), seed, [Xk_te], info=dict(method="F1k", n_train=nt))
            out[("F1k", BASE, seed, 1.0)] = p
            _, (zh,) = F.fit(BASE, axis, lambda: (X[okz], z[okz], None), seed, [TE.X], info=dict(method="V2", n_train=int(okz.sum())))
            zh = np.clip(zh, float(z[okz].min()), float(z[okz].max())) if okz.any() else np.full(len(TE), np.nan)
            out[("V2", BASE, seed, 0.0)] = np.exp(zh) * TE.s
            _, (g,) = F.fit(BASE, axis, lambda: (X, r0, None), seed, [TE.X], info=dict(method="R0", n_train=nt))
            for lam in a.LAMS:
                out[("R0", BASE, seed, lam)] = E0 * TE.s + lam * g
            _, (hh,) = F.fit(BASE, axis, lambda: (X[okh], h[okh], None), seed, [TE.X], info=dict(method="RMw", n_train=int(okh.sum())))
            hh = np.clip(hh, float(h[okh].min()), float(h[okh].max())) if okh.any() else np.full(len(TE), np.nan)
            for lam in a.LAMS:
                out[("RMw", BASE, seed, lam)] = E_w * TE.s * np.exp(lam * hh)
        lam_sel, flag, err = nest_select(F, a, S, T, blkT, seed_parts, E0, r0_S)
        rec["nest"] = dict(lam=float(lam_sel), flag=flag, sse={str(k): float(v) for k, v in err.items()})
        for seed in a.SEEDS:
            out[("R1w@nest", BASE, seed, -1.0)] = E_w * TE.s + lam_sel * g1[seed]
    return out, rec


# ================================================================ 조각
def shard_paths(a, scheme, rep, fold):
    b = a.SHARDS / f"{a.TAG}__{scheme}__r{int(rep)}__f{int(fold)}"
    return dict(pred=Path(str(b) + "_pred.npz"), unit=Path(str(b) + "_unit.json"))


def _file_sha(path):
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "NA"


def code_sha():
    return _file_sha(__file__)


CFG_UNIT_KEYS = ("kind", "cluster_map", "tdd_matched")                # 조각마다 달라도 되는 항목(공통 해시에서 뺀다)


def unit_cfg(a, scheme):
    """결과에 영향을 주는 설정 요약. 반복·fold·워커 수는 넣지 않는다(조각의 정체 또는 실행 자원)."""
    kind = scheme_kind(scheme)
    cm = f"{a.CMAP.name}:{_file_sha(a.CMAP)}" if kind == "cluster" else ""
    tm = f"{a.TDDM.name}:{_file_sha(a.TDDM)}" if kind == "region" else ""       # V-G 의 N1 기준선 P0@tddm
    return dict(kind=kind, folds=int(a.folds), seeds=list(a.SEEDS), learners=list(a.LEARNERS), lams=list(a.LAMS), kappa=float(a.kappa),
                buffer_km=float(a.buffer_km), n_clusters=int(a.n_clusters), nest_folds=int(a.nest_folds), cb_iters=int(a.HA.cb_iters),
                base=BASE, feats=len(H.FEATS), cluster_map=cm, tdd_matched=tm)


def cfg_hash(cfg, common=False):
    d = {k: v for k, v in cfg.items() if not (common and k in CFG_UNIT_KEYS)}
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def unit_state(a, scheme, rep, fold):
    """(완료 여부, 사유). 완료 = 조각 두 파일이 있고, 설정 해시가 같고, status 가 failed 가 아니다."""
    p = shard_paths(a, scheme, rep, fold)
    if not (p["unit"].exists() and p["pred"].exists()):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != cfg_hash(unit_cfg(a, scheme)):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    return True, str(u.get("status", "ok"))


def _atomic_text(path, text):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text); os.replace(tmp, path)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def predict_unit(a, V, scheme, rep, fold, F, check=True):
    """작업 단위 하나의 셀과 예측. 반환 (cells, 예측 dict, 기록 dict). 채점 셀의 y 는 모델 함수에 넘기지만 모델은 쓰지 않는다(시험 f)."""
    cells = unit_cells(a, V, scheme, rep, fold, check=check)
    TE = V.take(cells["te"])
    kind = scheme_kind(scheme)
    if kind == "within" and scheme.startswith("W2-"):
        preds, rec = models_w2(F, a, V.take(cells["src"]), V.take(cells["trw"]), TE, V.block[cells["trw"]],
                               ("lgv", "nest", scheme[3:], int(rep), int(fold)))
    elif kind == "within":
        preds, rec = models_pooled(F, a, V.take(cells["tr"]), TE, "w1", sfx="w", extended=False)
    else:
        preds, rec = models_pooled(F, a, V.take(cells["tr"]), TE, "pooled", n1=(kind == "region"))
    return cells, preds, rec


def run_unit(a, V, scheme, rep, fold, dry=False):
    """작업 단위 하나를 실행하고 조각을 쓴다. dry = True 이면 학습 없이 수만 센다."""
    t0 = time.time()
    p = shard_paths(a, scheme, rep, fold)
    if not dry:                                                        # 이전 세대의 완료 표지를 먼저 지운다(중단되면 미완료로 남는다)
        p["unit"].parent.mkdir(parents=True, exist_ok=True)
        p["unit"].unlink(missing_ok=True)
    F = make_fitter(a, dry)
    cells, preds, rec = predict_unit(a, V, scheme, rep, fold, F, check=not dry)
    info = cells["info"]
    kind = scheme_kind(scheme)
    n_te = int(len(cells["te"]))
    keys = list(preds)
    P = np.vstack([np.asarray(preds[k], float) for k in keys]) if keys else np.zeros((0, n_te))
    bad = [k for k, row in zip(keys, P) if not k[0].endswith("#ex") and not np.all(np.isfinite(row))]
    n_ml = sum(1 for k in keys if k[1] != "none")
    n_ml_ok = sum(1 for k in keys if k[1] != "none" and k not in bad)
    n_fail = int(sum(F.fail.values())) + len(bad)
    status = "ok" if n_fail == 0 else ("failed" if (n_ml > 0 and n_ml_ok == 0) else "partial")
    stats = dict(n_fit=dict(F.n), n_fit_total=int(sum(F.n.values())), sec={k: round(v, 1) for k, v in F.sec.items()}, fail=dict(F.fail),
                 n_fit_detail=dict(F.nd), sec_detail={k: round(v, 2) for k, v in F.secd.items()}, rows_detail=dict(F.rowsd),
                 est_detail={k: round(v, 2) for k, v in F.estd.items()}, errors=list(F.errors), n_keys=int(len(keys)),
                 n_nonfinite_keys=int(len(bad)), nonfinite_keys=[list(k) for k in bad[:40]], status=status)
    if dry:
        return dict(scheme=scheme, rep=int(rep), fold=int(fold), **{k: v for k, v in info.items() if k != "train_macro"},
                    n_keys=len(keys), est_s=round(float(sum(F.estd.values())), 1), _detail=dict(F.nd), _rows=dict(F.rowsd), _est=dict(F.estd),
                    **{f"fit_{k}": v for k, v in F.n.items()})
    tmp = p["pred"].with_name(p["pred"].name + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, idx=cells["te"].astype(np.int64), loc_id=V.loc_id[cells["te"]], y=V.y[cells["te"]].astype(np.float64),
                        keys=np.array([json.dumps([k[0], k[1], int(k[2]), float(k[3])]) for k in keys], dtype=str), P=P.astype(np.float32))
    os.replace(tmp, p["pred"])
    cfg = unit_cfg(a, scheme)
    unit = dict(scheme=scheme, rep=int(rep), fold=int(fold), kind=kind, tag=a.TAG, elapsed_s=round(time.time() - t0, 1), **info, **stats,
                model=rec, cfg=cfg, cfg_hash=cfg_hash(cfg), cfg_common=cfg_hash(cfg, common=True), code_sha=code_sha(),
                code_sha_h40=H.code_sha(), threads=int(a.threads), cluster_src=V.cluster_src if kind == "cluster" else "",
                tdd_src=getattr(V, "tdd_src", "") if kind == "region" else "")
    _atomic_text(p["unit"], json.dumps(_jsonable(unit), ensure_ascii=False, indent=1, default=float))   # 완료 표지는 마지막에 쓴다
    return unit


# ---------------------------------------------------------------- 워커
_WARGS = None


def _worker_init(argv, threads):
    global _WARGS
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _WARGS = parse_args(argv)


def _worker_run(scheme, rep, fold):
    t0 = time.time()
    u = run_unit(_WARGS, get_vdata(_WARGS), scheme, rep, fold)
    u["wall_s"] = round(time.time() - t0, 1)
    u["n_fail"] = int(sum(u.get("fail", {}).values())) + int(u.get("n_nonfinite_keys", 0))
    return {k: u[k] for k in ("scheme", "rep", "fold", "n_train", "n_test", "n_buffer_excluded", "n_fit_total", "n_keys", "elapsed_s", "wall_s",
                              "status", "n_fail")}


# ================================================================ 집계: 통계 함수
def verdict4(lo, hi, lob, hib, delta):
    """4분 판정. 우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 네 끝값의 절댓값 ≤ δ, 그 밖 미결정. 비유한 값은 판정 불가."""
    v = [lo, hi, lob, hib]
    if not all(x is not None and np.isfinite(x) for x in v):
        return "판정 불가"
    if hi < 0 and hib < 0:
        return "우세"
    if lo > 0 and lob > 0:
        return "열세"
    if max(abs(float(x)) for x in v) <= float(delta):
        return "동등"
    return "미결정"


def p_equiv(dist, delta):
    """동등성의 부트스트랩 p = max(P(Δ* ≤ −δ), P(Δ* ≥ +δ))."""
    if dist is None:
        return np.nan
    d = np.asarray(dist, float); d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    return float(max(np.mean(d <= -delta), np.mean(d >= delta)))


def p_equiv2(dist, dist_beq, delta):
    """동등성 p 의 두 가중(셀 가중, 블록 등가중) 가운데 큰 값(h42 의 p_eq 와 같은 정의). 4분 판정의 '동등'은 이 값 ≤ 0.025 에 대응한다."""
    v = [p_equiv(d, delta) for d in (dist, dist_beq) if d is not None]
    v = [x for x in v if np.isfinite(x)]
    return float(max(v)) if v else np.nan


def pct_ci(dist):
    if dist is None:
        return np.nan, np.nan
    d = np.asarray(dist, float)
    if not np.isfinite(d).any():
        return np.nan, np.nan
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def boot_group(S, C, W):
    """키 묶음(행 = 반복·seed)의 부트스트랩 RMSE 평균 분포(셀 가중, 블록 등가중). h4_common.boot_delta_blocks 의 한쪽과 같은 식이다."""
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.sqrt((S @ W.T) / (C @ W.T))
        v = np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), 0.0); m = (C > 0).astype(float)
        e = (v @ W.T) / (m @ W.T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(b, 0), np.nanmean(e, 0)


def boot_r2(S, C, SY, SYY, W):
    with np.errstate(invalid="ignore", divide="ignore"):
        n = C @ W.T
        sst = (SYY @ W.T) - (SY @ W.T) ** 2 / n
        r2 = 1.0 - (S @ W.T) / sst
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(np.where(np.isfinite(r2), r2, np.nan), 0)


def strat_combine(per, regions):
    """지역별 결과 dict(delta, delta_beq, dist, dist_beq)를 층화 평균으로 합친다. CI 풀 = 분포가 있는 지역.
    점 추정은 CI 풀 지역의 평균이다(풀이 비면 값이 있는 지역 전부의 평균)."""
    have = [r for r in regions if per.get(r) is not None and np.isfinite(per[r]["delta"])]
    pool = [r for r in have if per[r].get("dist") is not None]
    use = pool if pool else have
    if not use:
        return None
    out = dict(delta=float(np.mean([per[r]["delta"] for r in use])), delta_beq=float(np.mean([per[r]["delta_beq"] for r in use])),
               dist=strat([per[r]["dist"] for r in pool]) if pool else None,
               dist_beq=strat([per[r]["dist_beq"] for r in pool]) if pool else None,
               rmse_A=float(np.mean([per[r]["rmse_A"] for r in use])), rmse_B=float(np.mean([per[r]["rmse_B"] for r in use])),
               n_cells=int(sum(per[r]["n_cells"] for r in use)), n_blocks=int(sum(per[r]["n_blocks"] for r in use)),
               n_rows=int(min(per[r]["n_rows"] for r in use)),
               n_rows_expected=int(max(per[r].get("n_rows_expected", per[r]["n_rows"]) for r in use)),
               n_ci_regions=int(len(pool)), n_regions_target=int(len(regions)),
               regions=",".join(use), regions_all=",".join(have), neg=int(sum(per[r]["delta"] < 0 for r in have)))
    short = sorted([(int(per[r]["n_rows"]), int(per[r].get("n_rows_expected", per[r]["n_rows"])), r) for r in use
                    if per[r]["n_rows"] < per[r].get("n_rows_expected", per[r]["n_rows"])], key=lambda q: (q[0] / max(q[1], 1), q[2]))
    if short:                                                          # 행(반복·seed)이 기대보다 적은 지역이 있으면 가장 부족한 지역의 값을 적는다
        out.update(n_rows=short[0][0], n_rows_expected=short[0][1], rows_short="; ".join(f"{r} {k}/{K}" for k, K, r in short))
    return out


def pool_text(k, total):
    return f"지역 {int(k)}/{int(total)}"


def finish_row(res, a, need_regions=0):
    """대비 결과에 CI, p, 4분 판정, 효과 크기 표기를 붙인다. need_regions > 0 이면 층화 평균이다(CI 풀 지역 수 2 미만은 판정 불가)."""
    lo, hi = pct_ci(res.get("dist")); lob, hib = pct_ci(res.get("dist_beq"))
    flag = []
    v = verdict4(lo, hi, lob, hib, a.delta_eq); v10 = verdict4(lo, hi, lob, hib, a.delta_eq_aux)
    if res.get("dist") is None:
        flag.append("no CI")
    elif np.isfinite(lo) and (res["delta"] < lo - 1e-9 or res["delta"] > hi + 1e-9):
        flag.append("assert: 점 추정치가 백분위 CI 밖"); v = v10 = "판정 불가"
    if need_regions:
        k = int(res.get("n_ci_regions", 0))
        flag.append(f"neg {res.get('neg', 0)}/{len([r for r in str(res.get('regions_all', '')).split(',') if r])}; ci_regions {k}")
        if k < H.MIN_POOL_REGIONS:
            v = v10 = "판정 불가"
    rB = res.get("rmse_B", np.nan)
    small = bool(abs(res["delta"]) < SMALL_EFFECT_CM)
    row = dict(delta=res["delta"], ci_lo=lo, ci_hi=hi, p_boot=H4.boot_p(res["dist"]) if res.get("dist") is not None else np.nan,
               delta_blockeq=res["delta_beq"], ci_lo_beq=lob, ci_hi_beq=hib, verdict4=v, verdict4_d10=v10,
               p_equiv=p_equiv2(res.get("dist"), res.get("dist_beq"), a.delta_eq), rmse_A=res.get("rmse_A", np.nan), rmse_B=rB,
               delta_pct_ref=float(100.0 * res["delta"] / rB) if np.isfinite(rB) and rB > 0 else np.nan,
               small_effect=small, small_note=SMALL_EFFECT_TXT if (small and v in ("우세", "열세")) else "",
               n_cells=res.get("n_cells", 0), n_blocks=res.get("n_blocks", 0),
               n_rows=res.get("n_rows", 0), n_rows_expected=res.get("n_rows_expected", res.get("n_rows", 0)),
               rows_short=res.get("rows_short", ""), ci_flag="; ".join(flag))
    if need_regions:
        k = int(res.get("n_ci_regions", 0))
        row.update(n_ci_regions=k, n_regions_target=int(need_regions), regions=res.get("regions", ""),
                   pool=pool_text(k, need_regions), pool_partial=bool(H.MIN_POOL_REGIONS <= k < int(need_regions)))
    return row


def valid_row(r):
    """4분 판정(우세, 열세, 동등, 미결정)이 나온 대비 행인지."""
    return r is not None and str(r.get("verdict4", "")) not in NA_VERDICTS and str(r.get("verdict4", "")) != ""


def mark_verdict(test, text, used):
    """판정 문구에 부분 표기와 효과 크기 표기를 붙인다(계획서 §6A.2, 개정 7. h42 의 TestBook.verdict 와 같은 규칙).
    used = 판정에 쓴 대비 행(finish_row 의 dict) 목록. 판정 불가 문구에는 붙이지 않는다.
    - 풀 지역 수가 대상 지역 수보다 적은 층화 평균: '부분(지역 k/N): '
    - 행(반복·seed)이 기대보다 적은 대비: '부분(반복·seed 행 k/K): '. 확인적 가설(L19)은 판정 불가다
    - 우세 또는 열세이면서 |Δ| < 0.5 cm: 문구 끝에 표기"""
    text = str(text)
    if text.startswith("판정 불가"):
        return text
    rows = [r for r in (used or []) if valid_row(r)]
    pools = [(int(r["n_ci_regions"]), int(r["n_regions_target"])) for r in rows
             if r.get("n_regions_target") and int(r.get("n_ci_regions", 0)) < int(r["n_regions_target"])]
    short = [(int(r["n_rows"]), int(r["n_rows_expected"])) for r in rows
             if r.get("n_rows_expected") is not None and int(r.get("n_rows", 0)) < int(r["n_rows_expected"])]
    if short and test in CONFIRMATORY:
        k, K = min(short, key=lambda q: q[0] / max(q[1], 1))
        return f"판정 불가(반복·seed 행 {k}/{K}. 등록한 반복이 모두 채워진 뒤 판정한다)"
    pre = []
    if pools:
        k, N = min(pools, key=lambda q: q[0] / q[1])
        pre.append(pool_text(k, N))
    if short:
        k, K = min(short, key=lambda q: q[0] / max(q[1], 1))
        pre.append(f"반복·seed 행 {k}/{K}")
    if pre:
        text = f"부분({', '.join(pre)}): {text}"
    small = [str(r.get("contrast", "")) for r in rows if r["verdict4"] in ("우세", "열세") and abs(float(r["delta"])) < SMALL_EFFECT_CM]
    if small:
        text = f"{text} [{SMALL_EFFECT_TXT}: {', '.join(small)}]"
    return text


# ================================================================ 집계: 조각 읽기
def find_shards(a):
    out = []
    if not a.SHARDS.exists():
        return out
    for p in sorted(a.SHARDS.glob(f"{a.TAG}__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")
        if len(parts) != 4 or parts[0] != a.TAG or not parts[2].startswith("r") or not parts[3].startswith("f"):
            continue
        pred = Path(str(p)[:-len("_unit.json")] + "_pred.npz")
        if pred.exists():
            out.append(dict(scheme=parts[1], rep=int(parts[2][1:]), fold=int(parts[3][1:]), unit=p, pred=pred))
    return out


def check_shard_cfg(a, units):
    hs = Counter(str(u.get("cfg_common", "legacy")) for u in units)
    cs = Counter(str(u.get("code_sha", "legacy")) for u in units)
    c40 = Counter(str(u.get("code_sha_h40", "legacy")) for u in units)
    if len(cs) > 1 or len(c40) > 1:
        print(f"  [warn] 조각의 코드 해시가 여럿이다: h41 {dict(cs)}, h40 {dict(c40)}", flush=True)
    if len(hs) > 1:
        msg = f"조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}. 설정이 다른 실행이 섞였다"
        if not a.allow_mixed_cfg:
            raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
        print("  [warn] " + msg, flush=True)
    return dict(cfg_common=dict(hs), code_sha=dict(cs), code_sha_h40=dict(c40))


def load_preds(a, V, shards):
    """(단, 반복)마다 fold 조각을 자료 행 위치로 모은다. 채점 집합의 각 셀이 정확히 한 번 있는지 확인한다.
    채점 범위가 1 미만인 (단, 반복)은 집계에서 빼고 나머지를 계속 집계한다(--allow-partial 이면 통계에 넣는다).
    비유한 예측은 지역 단위로 판단한다(키의 행은 비유한 셀이 있는 지역의 통계에서만 빠진다. 전체 셀 풀은 모든 지역에서 유한한 행만 쓴다).
    반환 (PR, 실패 기록, 제외한 (단, 반복) 목록). PR[(단, 반복)] = dict(keys, P(키 × 전체 채점 셀, 단 밖 셀은 NaN), bad_reg(행 → 지역 집합),
    bad_rows, coverage)."""
    sidx = np.where(V.score)[0]
    pos = np.full(V.N, -1, np.int64); pos[sidx] = np.arange(len(sidx))
    by = {}
    for sh in shards:
        by.setdefault((sh["scheme"], sh["rep"]), []).append(sh)
    PR, failed, excluded = {}, [], []
    for (scheme, rep), lst in sorted(by.items()):
        want = np.where(V.score_of(scheme))[0]
        keys, P = None, None
        cnt = np.zeros(len(sidx), np.int64)
        for sh in sorted(lst, key=lambda d: d["fold"]):
            with np.load(sh["pred"], allow_pickle=False) as z:
                idx = z["idx"].astype(np.int64); lid = z["loc_id"].astype(np.int64)
                ks = [tuple(json.loads(str(s))) for s in z["keys"]]
                Pz = np.asarray(z["P"], np.float32)
            ks = [(str(k[0]), str(k[1]), int(k[2]), float(k[3])) for k in ks]
            if not np.array_equal(lid, V.loc_id[idx]):
                raise SystemExit(f"[summarize] {sh['pred'].name}: loc_id 가 현재 자료와 다르다(자료 판이 다르다)")
            if np.any(pos[idx] < 0):
                raise SystemExit(f"[summarize] {sh['pred'].name}: 채점 집합 밖의 셀이 있다")
            if keys is None:
                keys = ks; P = np.full((len(keys), len(sidx)), np.nan, np.float32)
            if ks != keys:
                common = [k for k in keys if k in set(ks)]
                failed.append(dict(scheme=scheme, rep=rep, fold=sh["fold"], reason="fold 사이 키 목록이 다르다",
                                   detail=f"{len(keys)} → {len(common)}"))
                P = P[[keys.index(k) for k in common]]; keys = common
            row_of = {k: i for i, k in enumerate(ks)}
            P[:, pos[idx]] = Pz[[row_of[k] for k in keys]]
            np.add.at(cnt, pos[idx], 1)
        cw = cnt[pos[want]]
        if np.any(cw > 1):
            raise SystemExit(f"[summarize] {scheme} r{rep}: 두 번 이상 예측된 채점 셀이 {int((cw > 1).sum())}개다")
        cov = float((cw == 1).mean()) if len(cw) else 0.0
        if cov < 1.0:
            msg = f"{scheme} r{rep}: 예측되지 않은 채점 셀이 {int((cw == 0).sum())}개다(범위 {cov:.4f})"
            if not a.allow_partial:                                   # 이 (단, 반복)만 집계에서 뺀다. 완결된 단의 집계는 계속한다
                print("  [warn] " + msg + ". 이 (단, 반복)은 집계에서 뺀다(--allow-partial 로 통계에 넣을 수 있다)", flush=True)
                failed.append(dict(scheme=scheme, rep=rep, fold=-1, reason="채점 범위 부분, 집계 제외", detail=f"{cov:.4f}"))
                excluded.append((scheme, int(rep)))
                continue
            print("  [warn] " + msg + ". --allow-partial: 통계에 넣는다", flush=True)
            failed.append(dict(scheme=scheme, rep=rep, fold=-1, reason="채점 범위 부분(--allow-partial 로 통계에 포함)", detail=f"{cov:.4f}"))
        covered = cnt[pos[want]] == 1
        mac_w = np.asarray(V.macro[want]).astype(str)
        bad_reg = {}
        for i, k in enumerate(keys):
            if k[0].endswith("#ex"):
                continue
            nfm = (~np.isfinite(P[i, pos[want]])) & covered
            if nfm.any():
                regs = sorted(set(mac_w[nfm].tolist()))
                bad_reg[i] = set(regs)
                failed.append(dict(scheme=scheme, rep=rep, fold=-1, reason="비유한 예측(해당 지역의 통계에서 뺀다)",
                                   detail=f"{list(k)}: {int(nfm.sum())}셀, 지역 {regs}"))
        PR[(scheme, int(rep))] = dict(keys=keys, P=P, bad_reg=bad_reg, bad_rows=set(bad_reg), coverage=cov, n_folds=len(lst))
    return PR, failed, excluded


class Agg:
    """집계기. 지역별 블록 부호, 재표집 다중도, 키 묶음 색인을 가진다."""

    def __init__(self, a, V, PR, excluded=None):
        self.a = a; self.PR = PR
        self.excluded = [tuple(v) for v in (excluded or [])]           # 채점 범위가 부분이라 집계에서 뺀 (단, 반복)
        self.sidx = np.where(V.score)[0]
        self.y = V.y[self.sidx].astype(float)
        mac = V.macro[self.sidx]; gid = V.gid[self.sidx]
        self.reg = {}
        for r in REG7:
            m = np.where(mac == r)[0]
            if not len(m):
                continue
            ub, codes = np.unique(gid[m].astype(str), return_inverse=True)
            nb = int(len(ub))
            W = H4.boot_weights(nb, int(a.nboot), seed_of("lgvboot", r)) if a.nboot > 0 else None
            self.reg[r] = dict(cells=m, codes=codes.ravel().astype(np.int64), nb=nb, W=W, has_ci=bool(nb >= MIN_BLOCKS_CI and W is not None),
                               blocks=ub, n_cells=int(len(m)))
        self.groups = {}
        for (s, rep), d in sorted(PR.items()):
            for i, k in enumerate(d["keys"]):
                self.groups.setdefault(s, {}).setdefault((k[0], k[1], float(k[3])), []).append((rep, i))
        self._st = {}
        self._dc = OrderedDict()

    def has(self, scheme, g):
        return g in self.groups.get(scheme, {})

    def rows_of(self, scheme, g, r=None):
        """키 묶음의 행(반복, 행 번호) 가운데 지역 r 에서 예측이 모두 유한한 행. r = None 이면 모든 지역에서 유한한 행이다(전체 셀 풀)."""
        out = []
        for rep, i in self.groups.get(scheme, {}).get(g, []):
            bad = self.PR[(scheme, rep)].get("bad_reg", {}).get(i, set())
            if (r is None and bad) or (r is not None and r in bad):
                continue
            out.append((rep, i))
        return out

    def expected_rows(self, scheme, g):
        """키 묶음의 기대 행 수 = 등록한 반복 수 × seed 수(물리식과 기준선은 seed 가 없다)."""
        return int(len(reps_of(self.a, scheme)) * (1 if g[1] == "none" else len(self.a.SEEDS)))

    def finite_mask(self, scheme, g, r):
        """묶음의 첫 행에서 유한한 셀(지역 셀 순서). '#ex' 키의 채점 셀 집합이다."""
        rows = self.groups.get(scheme, {}).get(g, [])
        if not rows or r not in self.reg:
            return None
        rep, i = rows[0]
        return np.isfinite(self.PR[(scheme, rep)]["P"][i, self.reg[r]["cells"]])

    def stats(self, scheme, g, r, mask=None, pool=False):
        """지역 r 의 블록 통계(행 = 반복·seed). S = SSE, C = 셀 수, AE = 절대 오차 합, SE = 오차 합, SY·SYY = 관측값의 합과 제곱합.
        pool = True 는 전체 셀 풀에 이어 붙일 통계다(모든 지역에서 유한한 행만 쓴다. 지역마다 행 수가 같아야 한다)."""
        ck = (scheme, g, r, bool(pool))
        if mask is None and ck in self._st:
            return self._st[ck]
        rows = self.rows_of(scheme, g, None if pool else r)
        if not rows or r not in self.reg:
            return None
        R = self.reg[r]; y = self.y[R["cells"]]; nb = R["nb"]; codes = R["codes"]
        K = len(rows)
        S = np.zeros((K, nb)); C = np.zeros((K, nb)); AE = np.zeros((K, nb)); SE = np.zeros((K, nb)); SY = np.zeros((K, nb)); SYY = np.zeros((K, nb))
        for j, (rep, i) in enumerate(rows):
            p = self.PR[(scheme, rep)]["P"][i, R["cells"]].astype(float)
            if mask is not None:
                p = np.where(mask, p, np.nan)
            s_, c_ = H4.block_sse(y, p, codes, nb)                       # 비유한 셀은 빠진다
            ok = np.isfinite(p) & np.isfinite(y)
            e = p[ok] - y[ok]; cd = codes[ok]
            S[j] = s_; C[j] = c_
            AE[j] = np.bincount(cd, weights=np.abs(e), minlength=nb); SE[j] = np.bincount(cd, weights=e, minlength=nb)
            SY[j] = np.bincount(cd, weights=y[ok], minlength=nb); SYY[j] = np.bincount(cd, weights=y[ok] ** 2, minlength=nb)
        out = dict(S=S, C=C, AE=AE, SE=SE, SY=SY, SYY=SYY, n_rows=K) if C.sum() > 0 else None
        if mask is None:
            self._st[ck] = out
        return out

    def pooled_stats(self, scheme, g, mask_fn=None):
        """전체 셀 풀: 지역의 블록 통계와 재표집 다중도를 이어 붙인다."""
        parts, Ws = [], []
        K = None
        for r, R in self.reg.items():
            st = self.stats(scheme, g, r, None if mask_fn is None else mask_fn(r), pool=True)
            if st is None:
                continue
            K = st["n_rows"] if K is None else K
            if st["n_rows"] != K:
                return None
            parts.append(st); Ws.append(R["W"])
        if not parts:
            return None
        out = {k: np.hstack([p[k] for p in parts]) for k in ("S", "C", "AE", "SE", "SY", "SYY")}
        out["n_rows"] = K
        out["W"] = np.hstack(Ws) if all(w is not None for w in Ws) else None
        return out

    @staticmethod
    def point(st):
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S, C = st["S"], st["C"]
            n = C.sum(1)
            sst = st["SYY"].sum(1) - st["SY"].sum(1) ** 2 / n
            r2 = np.where(sst > 1e-12, 1.0 - S.sum(1) / np.where(sst > 1e-12, sst, 1.0), np.nan)      # 채점 셀이 1개면 R² 를 내지 않는다
            return dict(rmse=float(np.nanmean(H4._rmse_rows(S, C))), beq=float(np.nanmean(H4._beq_rows(S, C))),
                        mae=float(np.nanmean(st["AE"].sum(1) / n)), bias=float(np.nanmean(st["SE"].sum(1) / n)),
                        r2=float(np.nanmean(r2)) if np.isfinite(r2).any() else np.nan, sse=float(np.nanmean(S.sum(1))),
                        sst=float(np.nanmean(sst)),
                        n_cells=int(round(float(np.nanmean(n)))), n_blocks=int((C.max(0) > 0).sum()), n_rows=int(st["n_rows"]))

    def term(self, scheme, g, scope, mask=None, mask_fn=None):
        """항 하나의 점 추정과 부트스트랩 분포. scope = 지역 이름 또는 'POOL'."""
        if scope == "POOL":
            st = self.pooled_stats(scheme, g, mask_fn); W = st["W"] if st is not None else None; ci = W is not None
        else:
            st = self.stats(scheme, g, scope, mask)
            W = self.reg[scope]["W"] if scope in self.reg else None
            ci = bool(scope in self.reg and self.reg[scope]["has_ci"])
        if st is None:
            return None
        out = self.point(st)
        out["st"] = st
        out["n_rows_expected"] = self.expected_rows(scheme, g)
        if ci and W is not None:
            ck = (scheme, g, scope) if (mask is None and mask_fn is None) else None
            if ck is not None and ck in self._dc:
                self._dc.move_to_end(ck)
                out["dist"], out["dist_beq"] = self._dc[ck]
            else:
                out["dist"], out["dist_beq"] = boot_group(st["S"], st["C"], W)
                if ck is not None:
                    self._dc[ck] = (out["dist"], out["dist_beq"])
                    while len(self._dc) > DIST_CACHE_MAX:
                        self._dc.popitem(last=False)
            out["W"] = W
        else:
            out["dist"] = out["dist_beq"] = None
        return out

    def contrast(self, terms, scope):
        """terms = [(부호, 단, 묶음)]. 점 추정과 분포는 부호를 곱해 더한다(같은 재표집 번호끼리). scope = 지역 이름 또는 'POOL'.
        '#ex' 키가 있으면 모든 항을 그 키의 유한 셀로 제한한다."""
        ex = [(s, g) for _, s, g in terms if g[0].endswith("#ex")]
        mask, mask_fn = None, None
        if ex:
            if scope == "POOL":
                def mask_fn(r, ex=ex):
                    ms = [self.finite_mask(s, g, r) for s, g in ex]
                    ms = [m for m in ms if m is not None]
                    return np.logical_and.reduce(ms) if ms else None
            else:
                ms = [self.finite_mask(s, g, scope) for s, g in ex]
                if any(m is None for m in ms):
                    return None
                mask = np.logical_and.reduce(ms)
        vals = [self.term(s, g, scope, mask, mask_fn) for _, s, g in terms]
        if any(v is None for v in vals):
            return None
        sg = [float(t[0]) for t in terms]
        has = all(v["dist"] is not None for v in vals)
        return dict(delta=float(sum(c * v["rmse"] for c, v in zip(sg, vals))), delta_beq=float(sum(c * v["beq"] for c, v in zip(sg, vals))),
                    dist=sum(c * v["dist"] for c, v in zip(sg, vals)) if has else None,
                    dist_beq=sum(c * v["dist_beq"] for c, v in zip(sg, vals)) if has else None,
                    rmse_A=vals[0]["rmse"], rmse_B=vals[1]["rmse"] if len(vals) > 1 else np.nan, n_cells=vals[0]["n_cells"],
                    n_blocks=vals[0]["n_blocks"], **self._row_counts(vals))

    @staticmethod
    def _row_counts(vals):
        """대비의 행 수(반복·seed). 항마다 기대 행 수가 다르다(물리식은 seed 가 없다). 기대보다 적은 항이 있으면 가장 부족한 항의
        (행 수, 기대 행 수)를 적고, 없으면 첫 항(방법 A)의 값을 적는다."""
        short = [(int(v["n_rows"]), int(v["n_rows_expected"])) for v in vals if v["n_rows"] < v["n_rows_expected"]]
        if short:
            k, K = min(short, key=lambda q: q[0] / max(q[1], 1))
            return dict(n_rows=k, n_rows_expected=K)
        return dict(n_rows=int(vals[0]["n_rows"]), n_rows_expected=int(vals[0]["n_rows_expected"]))

    def contrast_mean(self, terms, regions):
        per = {r: self.contrast(terms, r) for r in regions if r in self.reg}
        return strat_combine(per, list(regions)), per


def g_(method, learner=BASE, lam=None):
    """키 묶음 (method, learner, lam). 방법별 기준 λ: 물리식·V2 = 0.0, 직접 계열 = 1.0, 잔차·곱셈 계열 = 0.25, R1w@nest = −1.0."""
    m = method
    if m.startswith(("PS", "P0", "P1w", "P2w", "B:")):
        return (m, "none", 0.0)
    if m.startswith("R1w@nest"):
        return (m, learner, -1.0)
    if m.startswith(("D0", "F1k")):
        return (m, learner, 1.0)
    if m.startswith("V2"):
        return (m, learner, 0.0)
    return (m, learner, LAM_BASE if lam is None else float(lam))


# ================================================================ 집계: 표
def build_metrics(a, G):
    rows = []
    for scheme in sorted(G.groups):
        kind = scheme_kind(scheme)
        regs = [scheme[3:]] if kind == "within" else list(G.reg)
        for g in sorted(G.groups[scheme]):
            per = {}
            for r in regs:
                t = G.term(scheme, g, r)
                if t is None:
                    continue
                lo, hi = pct_ci(t["dist"]); lob, hib = pct_ci(t["dist_beq"])
                r2lo, r2hi = (np.nan, np.nan)
                if t["dist"] is not None:
                    st = t["st"]
                    r2lo, r2hi = pct_ci(boot_r2(st["S"], st["C"], st["SY"], st["SYY"], t["W"]))
                per[r] = t
                rows.append(dict(scheme=scheme, method=g[0], learner=g[1], lam=g[2], scope="region", target=r, n_cells=t["n_cells"],
                                 n_blocks=t["n_blocks"], n_rows=t["n_rows"], rmse=t["rmse"], rmse_lo=lo, rmse_hi=hi, rmse_beq=t["beq"],
                                 rmse_beq_lo=lob, rmse_beq_hi=hib, mae=t["mae"], bias=t["bias"], r2=t["r2"], r2_lo=r2lo, r2_hi=r2hi,
                                 r2_within=t["r2"], n_ci_regions=int(t["dist"] is not None), ci_flag="" if t["dist"] is not None else "blocks<8"))
            if kind == "within":
                continue
            for name, regions in (("MEAN5", MAIN5), ("MEAN4", MAIN4)):
                have = [r for r in regions if r in per]
                pool = [r for r in have if per[r]["dist"] is not None]
                use = pool if pool else have
                if not use:
                    continue
                lo, hi = pct_ci(strat([per[r]["dist"] for r in pool]) if pool else None)
                lob, hib = pct_ci(strat([per[r]["dist_beq"] for r in pool]) if pool else None)
                rows.append(dict(scheme=scheme, method=g[0], learner=g[1], lam=g[2], scope=name, target=f"MEAN[{','.join(use)}]",
                                 n_cells=int(sum(per[r]["n_cells"] for r in use)), n_blocks=int(sum(per[r]["n_blocks"] for r in use)),
                                 n_rows=int(min(per[r]["n_rows"] for r in use)), rmse=float(np.mean([per[r]["rmse"] for r in use])),
                                 rmse_lo=lo, rmse_hi=hi, rmse_beq=float(np.mean([per[r]["beq"] for r in use])), rmse_beq_lo=lob, rmse_beq_hi=hib,
                                 mae=float(np.mean([per[r]["mae"] for r in use])), bias=float(np.mean([per[r]["bias"] for r in use])),
                                 r2=float(np.mean([per[r]["r2"] for r in use])), r2_lo=np.nan, r2_hi=np.nan,
                                 r2_within=float(np.mean([per[r]["r2"] for r in use])), n_ci_regions=len(pool),
                                 ci_flag=f"ci_regions {len(pool)}/{len(regions)}"))
            t = G.term(scheme, g, "POOL")
            if t is not None:
                lo, hi = pct_ci(t["dist"]); lob, hib = pct_ci(t["dist_beq"])
                st = t["st"]
                r2lo, r2hi = pct_ci(boot_r2(st["S"], st["C"], st["SY"], st["SYY"], t["W"])) if t["dist"] is not None else (np.nan, np.nan)
                sse_w = float(sum(per[r]["sse"] for r in per)); sst_w = float(sum(per[r]["sst"] for r in per))
                rows.append(dict(scheme=scheme, method=g[0], learner=g[1], lam=g[2], scope="POOL", target="POOL", n_cells=t["n_cells"],
                                 n_blocks=t["n_blocks"], n_rows=t["n_rows"], rmse=t["rmse"], rmse_lo=lo, rmse_hi=hi, rmse_beq=t["beq"],
                                 rmse_beq_lo=lob, rmse_beq_hi=hib, mae=t["mae"], bias=t["bias"], r2=t["r2"], r2_lo=r2lo, r2_hi=r2hi,
                                 r2_within=float(1.0 - sse_w / sst_w) if sst_w > 0 else np.nan, n_ci_regions=len(G.reg),
                                 ci_flag="지역 안 블록 재표집을 이어 붙인 CI"))
    df = pd.DataFrame(rows)
    if len(df):
        cov = {s: float(np.mean([d["coverage"] for (s2, _), d in G.PR.items() if s2 == s])) for s in df.scheme.unique()}
        df["coverage"] = df.scheme.map(cov)
    return df


def contrast_rows(a, G, name, family, terms, kind, scopes):
    """대비 하나를 범위별 행으로 만든다. scopes 의 원소는 지역 이름, 'POOL', ('MEAN5', 지역 목록) 가운데 하나다."""
    rows = []
    sA, gA = terms[0][1], terms[0][2]
    sB, gB = (terms[1][1], terms[1][2]) if len(terms) > 1 else ("", ("", "", np.nan))
    head = dict(contrast=name, family=family, kind=kind, scheme_A=sA, method_A=gA[0], learner_A=gA[1], lam_A=gA[2],
                scheme_B=sB, method_B=gB[0], learner_B=gB[1], lam_B=gB[2], n_terms=len(terms))
    for sc in scopes:
        if isinstance(sc, tuple):
            res, _ = G.contrast_mean(terms, sc[1])
            if res is None:
                continue
            rows.append(dict(**head, scope=sc[0], target=f"MEAN[{res['regions']}]", **finish_row(res, a, need_regions=len(sc[1]))))
        else:
            res = G.contrast(terms, sc)
            if res is None:
                continue
            rows.append(dict(**head, scope="POOL" if sc == "POOL" else "region", target=sc, **finish_row(res, a)))
    return rows


def build_contrasts(a, G):
    rows = []
    pooled_scopes = list(G.reg) + [("MEAN5", MAIN5), ("MEAN4", MAIN4), "POOL"]
    for scheme in sorted(G.groups):
        kind = scheme_kind(scheme)
        gs = sorted(G.groups[scheme])
        if kind == "within":
            R = scheme[3:]
            if scheme.startswith("W2-"):
                refs = [("P1w", g_("P1w")), ("P0", g_("P0"))]
            else:
                refs = [("PSw", g_("PSw"))]
            for g in gs:
                for nm, ref in refs:
                    if g == ref or not G.has(scheme, ref):
                        continue
                    rows += contrast_rows(a, G, f"{g[0]}[{g[1]},{g[2]}]-{nm}", "vs_physics", [(1, scheme, g), (-1, scheme, ref)], "within", [R])
            if scheme.startswith("W2-") and f"W1-{R}" in G.groups:                 # 타 지역 원천을 더한 효과
                for mA, mB in (("D0", "D0w"), ("R1w", "RSw"), ("F1k", "F1kw"), ("V2", "V2w"), ("P1w", "PSw")):
                    for lr in a.LEARNERS:
                        gA, gB = g_(mA, lr), g_(mB, lr)
                        if G.has(scheme, gA) and G.has(f"W1-{R}", gB):
                            rows += contrast_rows(a, G, f"W2.{mA}-W1.{mB}[{gA[1]}]", "source_added", [(1, scheme, gA), (-1, f"W1-{R}", gB)],
                                                  "within", [R])
                        if gA[1] == "none":
                            break
            continue
        ps = g_("PS")
        for g in gs:
            if g != ps and G.has(scheme, ps):
                rows += contrast_rows(a, G, f"{g[0]}[{g[1]},{g[2]}]-PS", "vs_physics", [(1, scheme, g), (-1, scheme, ps)], "pooled", pooled_scopes)
            if scheme != "V-R" and G.has("V-R", g):
                rows += contrast_rows(a, G, f"{g[0]}[{g[1]},{g[2]}]:{scheme}-V-R", "degradation", [(1, scheme, g), (-1, "V-R", g)], "pooled",
                                      pooled_scopes)
        if scheme != "V-R":
            for lr in a.LEARNERS:
                for lam in a.LAMS:
                    rs, d0 = g_("RS", lr, lam), g_("D0", lr)
                    if all(G.has(s, g) for s in (scheme, "V-R") for g in (rs, d0)):
                        rows += contrast_rows(a, G, f"[RS({lam})-D0][{lr}]:{scheme}-V-R", "double_diff",
                                              [(1, scheme, rs), (-1, "V-R", rs), (-1, scheme, d0), (1, "V-R", d0)], "pooled", pooled_scopes)
    return pd.DataFrame(rows)


def build_tests(a, G):
    """L19–L22 의 대비 행과 판정 행(계획서 §6A.5 와 §6A.2 의 개정 7 규칙).
    Holm 보정 p 는 두 묶음에 따로 붙인다: X3a = L19(V-R, V-G), L20, L21(λ 0.25)의 5지역 평균 주 대비, X3b = L22 의 기준 λ 지역 대비 3개.
    4분 판정이 나오지 않은 행(판정 불가)은 묶음에 넣지 않는다. 보조 열이며 판정에 쓰지 않는다.
    채점 범위가 부분이라 집계에서 뺀 (단, 반복)이 있으면 그 단을 쓰는 가설의 stat 에 적는다. 그 단의 조각이 하나도 남지 않으면 그 가설은
    판정 불가다. 행(반복·seed)이 기대보다 적은 대비를 쓴 판정은 '부분(반복·seed 행 k/K)'이고 확인적 가설 L19 는 판정 불가다."""
    out, bundles = [], {"X3a": [], "X3b": []}
    excl = {}
    for s, rep in getattr(G, "excluded", []):
        excl.setdefault(s, []).append(int(rep))

    def put(test, item, role, blind, rows, bundle=None, scope="MEAN5"):
        for r in rows:
            d = dict(test_id=test, item=item, role=role, blind=bool(blind), confirmatory=bool(test in CONFIRMATORY), **r)
            out.append(d)
            if bundle is not None and r.get("scope") == scope:
                d["holm_bundle"] = bundle
                bundles[bundle].append(d)
        return rows

    def note_excl(schemes):
        """집계에서 뺀 (단, 반복)의 설명. 없으면 빈 문자열."""
        q = [f"{s} 반복 {sorted(excl[s])}" for s in schemes if s in excl]
        return (" | 채점 범위가 부분이라 집계에서 뺀 (단, 반복): " + ", ".join(q)) if q else ""

    def verdict(test, text, stat, role="주", blind=True, used=None, schemes=()):
        miss = [s for s in schemes if s not in G.groups]
        if miss and not str(text).startswith("판정 불가("):
            text = f"판정 불가(집계에 쓸 수 있는 단이 없다: {', '.join(miss)})" if str(text).startswith("판정 불가") else text
        out.append(dict(test_id=test, item="verdict", scope="verdict" if role == "주" else "verdict_aux", role=role, blind=bool(blind),
                        confirmatory=bool(test in CONFIRMATORY), verdict=mark_verdict(test, text, used), stat=str(stat) + note_excl(schemes)))

    def mean_state(rows, scope="MEAN5"):
        m = [r for r in rows if r.get("scope") == scope]
        return (m[0]["verdict4"], m[0]) if m else ("판정 불가", None)

    def fmt(r):
        if r is None:
            return "행 없음"
        return (f"Δ {r['delta']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], 블록 등가중 [{r['ci_lo_beq']:.2f}, {r['ci_hi_beq']:.2f}], "
                f"{r.get('pool', '')}, {r['verdict4']}" + (f", {r['small_note']}" if r.get("small_note") else "")
                + (f", 행 {r['n_rows']}/{r['n_rows_expected']}" if r.get("n_rows_expected") and r["n_rows"] < r["n_rows_expected"] else ""))

    scopes5 = [r for r in MAIN5 if r in G.reg] + [("MEAN5", MAIN5), ("MEAN4", MAIN4), "POOL"]
    ps = g_("PS")

    # L19 검증 설계(확인적): D0 − PS 가 V-R 에서 우세이고 V-G 에서 우세가 아니다. 재현(비맹검)
    for lr in a.LEARNERS:
        role = "주" if lr == BASE else "보조"
        st = {}
        for s in ("V-R", "V-G"):
            rows = contrast_rows(a, G, f"D0[{lr}]-PS@{s}", "L19", [(1, s, g_("D0", lr)), (-1, s, ps)], "pooled", scopes5) \
                if G.has(s, g_("D0", lr)) and G.has(s, ps) else []
            put("L19", f"D0[{lr}]-PS@{s}", role, False, rows, bundle="X3a" if lr == BASE else None)
            st[s] = mean_state(rows)
        deg = contrast_rows(a, G, f"D0[{lr}]:V-G-V-R", "L19", [(1, "V-G", g_("D0", lr)), (-1, "V-R", g_("D0", lr))], "pooled", scopes5) \
            if G.has("V-G", g_("D0", lr)) and G.has("V-R", g_("D0", lr)) else []
        put("L19", f"D0[{lr}] 열화 V-G − V-R", "보조", False, deg)
        vr, vg = st["V-R"][0], st["V-G"][0]
        if "판정 불가" in (vr, vg):
            txt = "판정 불가"
        elif vr == "우세" and vg != "우세":
            txt = "지지"
        elif vr != "우세":
            txt = "지지하지 않음(V-R 에서 우세가 아니다. '역전' 표현을 쓰지 않고 열화 크기만 보고한다)"
        else:
            txt = "지지하지 않음(V-G 에서도 우세다. L1 과 함께 재검토한다)"
        verdict("L19", txt, f"{lr} | V-R: {fmt(st['V-R'][1])} | V-G: {fmt(st['V-G'][1])} | 열화: {fmt(mean_state(deg)[1])}", role=role, blind=False,
                used=[st["V-R"][1], st["V-G"][1]], schemes=("V-R", "V-G"))

    # L20 단조성: D0 의 RMSE 가 V-R < V-B < V-C100 < V-C500. 재현(비맹검)
    ladder = ["V-R", "V-B", "V-C100", "V-C500"]
    for lr in a.LEARNERS:
        role = "주" if lr == BASE else "보조"
        g = g_("D0", lr)
        val, short = {}, []
        for s in ladder + ["V-P", "V-S", "V-C0", "V-G"]:
            if not G.has(s, g):
                continue
            per = {r: G.term(s, g, r) for r in MAIN5 if r in G.reg}
            use = [r for r in MAIN5 if per.get(r) is not None]
            if use:
                val[s] = (float(np.mean([per[r]["rmse"] for r in use])), len(use))
                nr = int(min(per[r]["n_rows"] for r in use)); ne = int(max(per[r]["n_rows_expected"] for r in use))
                if s in ladder and (nr < ne or len(use) < len(MAIN5)):
                    short.append(f"{s}: 지역 {len(use)}/{len(MAIN5)}, 행 {nr}/{ne}")
                out.append(dict(test_id="L20", item=f"D0[{lr}] RMSE@{s}", role=role if s in ladder else "서술", blind=False, scope="MEAN5",
                                confirmatory=False, target=f"MEAN[{','.join(use)}]", rmse_A=val[s][0], n_ci_regions=len(use), n_rows=nr,
                                n_rows_expected=ne))
        rows = contrast_rows(a, G, f"D0[{lr}]:V-C500-V-R", "L20", [(1, "V-C500", g), (-1, "V-R", g)], "pooled", scopes5) \
            if G.has("V-C500", g) and G.has("V-R", g) else []
        put("L20", f"D0[{lr}]:V-C500 − V-R", role, False, rows, bundle="X3a" if lr == BASE else None)
        v, r = mean_state(rows)
        if v == "판정 불가" or any(s not in val for s in ladder):
            txt = "판정 불가" + (f"(점 추정이 없는 단: {', '.join(s for s in ladder if s not in val)})" if any(s not in val for s in ladder) else "")
        else:
            brk = [f"{x} → {y}" for x, y in zip(ladder[:-1], ladder[1:]) if not val[x][0] < val[y][0]]
            if v == "열세" and not brk:
                txt = "지지"
            elif v == "열세" and len(brk) == 1:
                txt = f"대체로 지지(어긋난 단: {brk[0]})"
            else:
                txt = "지지하지 않음(" + ("V-C500 − V-R 의 CI 하한이 0 이하" if v != "열세" else f"순서가 {len(brk)}곳 어긋남: {', '.join(brk)}") + ")"
            if short:                                                  # 순서 비교에 쓴 점 추정의 지역 수 또는 행 수가 기대보다 적다
                txt = f"부분(점 추정 {'; '.join(short)}): {txt}"
        verdict("L20", txt, f"{lr} | " + ", ".join(f"{s} {val[s][0]:.2f}" for s in ladder + ["V-P", "V-S", "V-C0", "V-G"] if s in val)
                + f" | V-C500 − V-R: {fmt(r)}", role=role, blind=False, used=[r], schemes=tuple(ladder))

    # L21 구조별 열화: [RS(V-G) − RS(V-R)] − [D0(V-G) − D0(V-R)]
    for lr in a.LEARNERS:
        for lam in a.LAMS:
            role = "주" if (lr == BASE and lam == LAM_BASE) else "보조"
            rs, d0 = g_("RS", lr, lam), g_("D0", lr)
            ok = all(G.has(s, g) for s in ("V-G", "V-R") for g in (rs, d0))
            rows = contrast_rows(a, G, f"[RS({lam})-D0][{lr}]:V-G-V-R", "L21",
                                 [(1, "V-G", rs), (-1, "V-R", rs), (-1, "V-G", d0), (1, "V-R", d0)], "pooled", scopes5) if ok else []
            put("L21", f"[RS(λ {lam}) − D0][{lr}] 의 V-G − V-R", role, True, rows, bundle="X3a" if role == "주" else None)
            v, r = mean_state(rows)
            txt = "판정 불가" if v == "판정 불가" else ("지지" if v == "우세" else f"지지하지 않음({v})")
            verdict("L21", txt, f"{lr}, λ {lam} | {fmt(r)}", role=role, blind=True, used=[r], schemes=("V-R", "V-G"))

    # L22 지역 내 충분 라벨: W2 의 R1w − P1w 가 우세인 지역 수 k/3
    cases = [("주", "R1w", BASE, LAM_BASE)] + [("보조", "R1w", BASE, lam) for lam in a.LAMS if lam != LAM_BASE] + [("보조", "R1w@nest", BASE, None)] \
        + [("보조", "R1w", lr, LAM_BASE) for lr in a.LEARNERS if lr != BASE]
    for role, m, lr, lam in cases:
        g = g_(m, lr, lam)
        win, det, n_ok, used = [], [], 0, []
        for R in a.WREGIONS:
            s = f"W2-{R}"
            rows = contrast_rows(a, G, f"{m}[{lr},{g[2]}]-P1w@{s}", "L22", [(1, s, g), (-1, s, g_("P1w"))], "within", [R]) \
                if G.has(s, g) and G.has(s, g_("P1w")) else []
            put("L22", f"{m}[{lr}, λ {g[2]}] − P1w", role, R != "Alaska", rows, bundle="X3b" if role == "주" else None, scope="region")
            if rows and rows[0]["verdict4"] != "판정 불가":
                n_ok += 1; used.append(rows[0])
                if rows[0]["verdict4"] == "우세":
                    win.append(R)
                det.append(f"{R}: Δ {rows[0]['delta']:.2f} [{rows[0]['ci_lo']:.2f}, {rows[0]['ci_hi']:.2f}], {rows[0]['verdict4']}")
            else:
                det.append(f"{R}: 판정 불가")
        nR = len(a.WREGIONS)
        if n_ok == 0:
            txt = "판정 불가"
        elif n_ok < nR:
            txt = f"부분({pool_text(n_ok, nR)}): 우세 지역 {win if win else '없음'}"
        elif len(win) == nR == 3:
            txt = "지역 내 조건에서 잔차 ML 은 재보정 물리식을 넘는다"
        elif win:
            txt = f"우세 지역 {len(win)}/{nR}: {', '.join(win)}"
        else:
            txt = "지역 내 조건에서도 재보정 물리식 대비 순가치는 확인되지 않았다"
        verdict("L22", txt, f"{m}[{lr}, λ {g[2]}] | k = {len(win)}/{nR} | " + "; ".join(det) + " | Alaska 는 재현(비맹검)", role=role,
                blind=False, used=used, schemes=tuple(f"W2-{R}" for R in a.WREGIONS))

    for name, bundle in bundles.items():                                   # Holm: 묶음마다 따로. 판정 불가 행은 묶음에 넣지 않는다
        bd = [d for d in bundle if valid_row(d) and np.isfinite(float(d.get("p_boot", np.nan)))]
        if not bd:
            continue
        hp = holm([d.get("p_boot", np.nan) for d in bd])
        for d, p in zip(bd, hp):
            d["holm_p"] = float(p) if np.isfinite(p) else np.nan
            d["holm_flag"] = "보정 전 유의" if (d.get("verdict4") in ("우세", "열세") and np.isfinite(p) and p >= 0.05) else ""
    df = pd.DataFrame(out)
    front = ["test_id", "item", "scope", "role", "confirmatory", "blind", "target", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq",
             "ci_hi_beq", "verdict4", "verdict4_d10", "small_note", "p_boot", "holm_bundle", "holm_p", "holm_flag", "p_equiv", "n_ci_regions", "pool",
             "n_rows", "n_rows_expected", "verdict", "stat"]
    return df[[c for c in front if c in df] + [c for c in df.columns if c not in front]] if len(df) else df


def timing_table(units):
    rows = []
    for u in units:
        for k, nf in (u.get("n_fit_detail") or {}).items():
            ax, lr, m = (k.split("|") + ["", ""])[:3]
            sec = float((u.get("sec_detail") or {}).get(k, 0.0)); nr = float((u.get("rows_detail") or {}).get(k, 0.0))
            rows.append(dict(scheme=u.get("scheme", ""), rep=u.get("rep", -1), fold=u.get("fold", -1), axis=ax, learner=lr, method=m, n_fit=int(nf),
                             sec=sec, sec_per_fit=sec / max(int(nf), 1), rows_per_fit=nr / max(int(nf), 1),
                             est_s=float((u.get("est_detail") or {}).get(k, np.nan)), threads=u.get("threads", ""), status=u.get("status", "")))
    return pd.DataFrame(rows)


def summarize(a, V, elapsed=0.0, skipped=None):
    t0 = time.time()
    sh = find_shards(a)
    if not sh:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{a.TAG}__*", flush=True)
        return None
    units = [json.loads(s["unit"].read_text()) for s in sh]
    cfg_info = check_shard_cfg(a, units)
    PR, failed, excluded = load_preds(a, V, sh)
    G = Agg(a, V, PR, excluded)
    met = build_metrics(a, G); con = build_contrasts(a, G); tests = build_tests(a, G)
    for df_ in (met, con, tests):                                          # 재표집 횟수를 표에 적는다(허용 표지 없는 집계는 1,000 으로 낮춘 값이다)
        if len(df_):
            df_["nboot"] = int(a.nboot)
    have = {}
    for s_, rep_ in PR:
        have.setdefault(s_, []).append(int(rep_))
    reps_short = {s_: dict(have=sorted(v), expected=list(reps_of(a, s_))) for s_, v in have.items() if len(v) < len(reps_of(a, s_))}
    if excluded:
        print(f"[summarize] 채점 범위가 부분이라 집계에서 뺀 (단, 반복) {len(excluded)}건: {excluded}", flush=True)
    for u in units:
        if u.get("status") != "ok" or u.get("errors"):
            failed.append(dict(scheme=u.get("scheme"), rep=u.get("rep"), fold=u.get("fold"), reason=f"조각 상태 {u.get('status')}",
                               detail=json.dumps(u.get("errors", []), ensure_ascii=False)[:500]))
    O = a.OUT; O.mkdir(parents=True, exist_ok=True)
    met.to_csv(O / f"{a.TAG}_metrics.csv", index=False); con.to_csv(O / f"{a.TAG}_contrasts.csv", index=False)
    tests.to_csv(O / f"{a.TAG}_tests.csv", index=False)
    tt = timing_table(units); tt.to_csv(O / f"{a.TAG}_timing.csv", index=False)
    pd.DataFrame(failed, columns=["scheme", "rep", "fold", "reason", "detail"]).to_csv(O / f"{a.TAG}_failed.csv", index=False)
    if len(tt):
        lt = tt.groupby(["axis", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), sec=("sec", "sum"))
        lt["sec_per_fit"] = lt.sec / lt.n_fit.clip(lower=1)
        print("[summarize] 학습기별 적합 시간(전 조각 합계)\n" + lt.to_string(index=False), flush=True)
    tr = pd.DataFrame([dict(scheme=u["scheme"], n_train=u.get("n_train", 0), n_test=u.get("n_test", 0), n_buf=u.get("n_buffer_excluded", 0),
                            lt1000=bool(u.get("train_lt_1000", False))) for u in units])
    tsum = {s: dict(units=int(len(g)), n_train_min=int(g.n_train.min()), n_train_median=float(g.n_train.median()), n_train_max=int(g.n_train.max()),
                    n_buffer_excluded_max=int(g.n_buf.max()), n_units_train_lt_1000=int(g.lt1000.sum())) for s, g in tr.groupby("scheme")}
    meta = dict(stage="H41/LGX ladder", plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6A", git_commit=H.git_commit(), tag=a.TAG,
                args={k: v for k, v in vars(a).items() if k.islower() and not isinstance(v, argparse.Namespace)},
                schemes=sorted({u["scheme"] for u in units}), n_units=len(units), n_cells=int(V.N), n_score=int(V.score.sum()),
                n_blocks_score=int(len(np.unique(V.gid[V.score]))), regions={r: dict(n_cells=R["n_cells"], n_blocks=R["nb"], has_ci=R["has_ci"])
                                                                              for r, R in G.reg.items()},
                n_groups=dict(V_P=int(len(np.unique(V.groups("V-P")))), V_S=int(len(np.unique(V.groups("V-S")))),
                              V_B=int(len(np.unique(V.groups("V-B"))))),
                cluster_map=dict(path=str(a.CMAP), exists=bool(a.CMAP.exists()), sha=_file_sha(a.CMAP), n_clusters=int(a.n_clusters)),
                train_cells=tsum, shard_cfg=cfg_info, unit_status=dict(Counter(str(u.get("status", "ok")) for u in units)),
                n_failed=int(len(failed)), excluded=[list(v) for v in excluded], reps_short=reps_short,
                nboot_capped=getattr(a, "nboot_asked", None) is not None, nboot_asked=int(getattr(a, "nboot_asked", None) or a.nboot),
                tdd_matched=dict(path=str(a.TDDM), exists=bool(a.TDDM.exists()), src=getattr(V, "tdd_src", "none")),
                skipped=skipped or [], key_fields=KEY_COLS, n_metrics=int(len(met)), n_contrasts=int(len(con)),
                n_tests=int(len(tests)), elapsed_s=round(elapsed, 1), summarize_s=round(time.time() - t0, 1),
                unit_elapsed_s_sum=float(sum(u.get("elapsed_s", 0) for u in units)), nboot=int(a.nboot),
                rules=dict(score="채점 집합 = eval_mask ∩ macro 7지역. 모든 단이 같은 집합을 쓴다",
                           ci="지역 r 의 W_r = boot_weights(블록 수, nboot, seed_of('lgvboot', r))를 모든 단·모델·반복이 공유한다. "
                              "키 묶음(반복·seed) 평균 RMSE 의 차. 층화 평균은 m1_stats.strat, 블록 8개 미만 지역은 점 추정만",
                           pool="전체 셀 풀의 CI 는 지역 안 블록 재표집 다중도를 이어 붙인다(블록 8개 미만 지역도 지역 안에서 재표집)",
                           verdict4=f"우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 네 끝값 절댓값 ≤ {a.delta_eq}(보조 "
                                    f"{a.delta_eq_aux}), 그 밖 미결정. CI 풀 지역 2개 미만·CI 없음·점 추정치가 CI 밖이면 판정 불가",
                           holm="두 묶음에 따로 m1_stats.holm 을 적용한다. X3a = L19(V-R, V-G), L20(V-C500 − V-R), L21(λ 0.25)의 5지역 평균 대비, "
                                "X3b = L22 의 기준 λ 지역 대비 3개. 판정 불가 행은 묶음에 넣지 않는다. 보조 열이며 판정에 쓰지 않는다",
                           p_equiv="동등성 p = max(P(Δ* ≤ −δ), P(Δ* ≥ +δ)), 두 가중 가운데 큰 값(h42 와 같다). '동등'은 p_equiv ≤ 0.025 에 대응한다",
                           partial="판정 문구의 부분 표기: 풀 지역 수 < 대상 지역 수이면 '부분(지역 k/N)', 대비의 행(반복·seed)이 기대보다 적으면 "
                                   "'부분(반복·seed 행 k/K)'(확인적 가설 L19 는 판정 불가). 채점 범위가 부분인 (단, 반복)은 집계에서 뺀다"
                                   "(--allow-partial 이면 넣는다). 비유한 예측은 지역 단위로 뺀다(전체 셀 풀은 모든 지역에서 유한한 행만 쓴다)",
                           base_lam="물리식·V2 0.0, 직접 계열 1.0, 잔차·곱셈 계열 0.25, R1w@nest −1.0(고른 값은 unit.json 의 model.nest)",
                           ex="'#ex' 키는 p4_ku 결측 셀을 채점에서 뺀다. 이 키가 든 대비는 모든 항을 같은 셀로 제한한다",
                           leakage="계수·대체 통계는 단위의 학습 셀에서만 구한다. 버퍼 단은 버퍼 안 학습 셀 0 개를 haversine 으로 단언한다. "
                                   "예외: p4_ku, p2_edaphic 은 load_base 가 공변량 결측을 전체 셀 중앙값으로 채워 계산한 열이다(라벨 미사용. "
                                   "토양 열 결측 399셀 가운데 396셀이 레나). 영향을 받는 키는 F1k 계열과 N1 의 ku·ed 기준선, B:ens 다"))
    (O / f"{a.TAG}_meta.json").write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=1, default=str))
    print(f"[summarize] 조각 {len(sh)} · 지표 {len(met):,} · 대비 {len(con):,} · 판정 표 {len(tests):,} · 실패 기록 {len(failed)} · "
          f"{time.time() - t0:.0f}s → {O}/{a.TAG}_*", flush=True)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])] if len(tests) else tests
    if len(v):
        print(v[["test_id", "role", "verdict", "stat"]].to_string(index=False), flush=True)
    return dict(metrics=met, contrasts=con, tests=tests, timing=tt, failed=failed, meta=meta, excluded=excluded)


# ================================================================ 실행
def write_cluster_map(a, V):
    tab = make_cluster_table(V.df, a.n_clusters)
    if a.CMAP.exists():
        old = pd.read_csv(a.CMAP)
        mg = old[["block", "cluster"]].merge(tab[["block", "cluster"]], on="block", how="outer", suffixes=("_old", "_new"))
        nd = int((mg.cluster_old != mg.cluster_new).sum())
        print(f"[cluster-map] 기존 대응표와 다른 블록 {nd}개(전체 {len(mg)})", flush=True)
        if nd:                                                             # 대응표가 정의다. 다른 환경의 k-means 로 덮어쓰지 않는다
            raise SystemExit(f"[cluster-map] 기존 대응표 {a.CMAP} 와 다르다. 덮어쓰지 않는다(정의를 바꾸려면 파일을 지우고 다시 실행)")
        print(f"[cluster-map] 기존 대응표와 같다. 다시 쓰지 않는다: {a.CMAP}", flush=True)
        return tab
    a.CMAP.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.CMAP.with_name(a.CMAP.name + f".tmp{os.getpid()}")
    tab.to_csv(tmp, index=False); os.replace(tmp, a.CMAP)
    g = tab.groupby("cluster").agg(n_blocks=("block", "size"), n_cells=("n_cells", "sum"))
    print(f"[cluster-map] 블록 {len(tab)}개 · 군집 {tab.cluster.nunique()}개 → {a.CMAP}", flush=True)
    print(g.to_string(), flush=True)
    return tab


def count_only(a, V, units, skipped):
    """학습 없이 단위 수, 적합 수, 학습 셀 수, 추정 시간을 출력한다. 추정 시간의 기준값(BASE_FIT_V)은 catboost_lo 만 실측이다."""
    rows = [run_unit(a, V, *u, dry=True) for u in units]
    det, nrw, est = Counter(), Counter(), Counter()
    for r in rows:
        for k, v in r.pop("_detail").items():
            det[k] += v
        for k, v in r.pop("_rows").items():
            nrw[k] += v
        for k, v in r.pop("_est").items():
            est[k] += v
    df = pd.DataFrame(rows).fillna(0)
    fc = [c for c in df.columns if c.startswith("fit_")]
    df["fit_total"] = df[fc].sum(1).astype(int) if fc else 0
    df["order"] = list(range(len(df)))
    a.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.OUT / f"{a.TAG}_count.csv", index=False)
    print(f"[count-only] 작업 단위 {len(units)} · 건너뜀 {len(skipped)}({', '.join(sorted({s['status'] for s in skipped})) or '없음'}) · "
          f"군집 정의 {V.cluster_src or '해당 없음'}", flush=True)
    by = df.groupby("scheme", sort=False).agg(units=("fold", "size"), n_train_min=("n_train", "min"), n_train_med=("n_train", "median"),
                                              n_train_max=("n_train", "max"), n_test_sum=("n_test", "sum"), buf_excl_max=("n_buffer_excluded", "max"),
                                              lt1000=("train_lt_1000", "sum"), fit_total=("fit_total", "sum"), keys=("n_keys", "sum"),
                                              est_h=("est_s", "sum"))
    by["est_h"] = (by.est_h / 3600).round(2)
    print(by.to_string(), flush=True)
    lt = {}
    for k in det:
        ax, lr, m = (k.split("|") + ["", ""])[:3]
        q = lt.setdefault(lr, dict(n_fit=0, rows=0.0, est_s=0.0))
        q["n_fit"] += det[k]; q["rows"] += nrw[k]; q["est_s"] += est[k]
    tab = pd.DataFrame([dict(learner=lr, n_fit=v["n_fit"], rows_per_fit=round(v["rows"] / max(v["n_fit"], 1)), base_s=BASE_FIT_V.get(lr, (np.nan,))[0],
                             est_h=round(v["est_s"] / 3600, 2), measured=(lr == BASE)) for lr, v in lt.items()])
    if len(tab):
        print("[count-only] 학습기별 추정(프로세스 1개, 4스레드 기준 누적 시간. measured = False 는 가정값)", flush=True)
        print(tab.sort_values("est_h", ascending=False).to_string(index=False), flush=True)
    tot_h = float(sum(v["est_s"] for v in lt.values())) / 3600
    print(f"[count-only] 총 적합 {int(df.fit_total.sum()):,} · 저장 키 {int(df.n_keys.sum()):,} · 추정 누적 {tot_h:.1f} h → "
          + " · ".join(f"워커 {w}개 {tot_h / w:.1f} h" for w in (2, 14)), flush=True)
    return df


def limit_local(a):
    """허용 표지가 없는 실행(--count-only, --summarize-only, --write-cluster-map)의 자원을 제한한다. 스레드는 finalize 가 1 로 낮춘다."""
    if run_permitted(a):
        return a
    if int(a.threads_asked) > 1:
        print(f"[local] 허용 표지가 없다: --threads {a.threads_asked} → 1", flush=True)
    if a.summarize_only and int(a.nboot) > LOCAL_NBOOT_MAX:
        print(f"[local] 허용 표지가 없다: --nboot {a.nboot} → {LOCAL_NBOOT_MAX}. 이 집계의 CI 와 판정은 등록한 재표집 횟수(10,000)의 값이 아니다"
              f"(표의 nboot 열과 meta 의 nboot_capped 에 적는다)", flush=True)
        a.nboot_asked = int(a.nboot); a.nboot = LOCAL_NBOOT_MAX
    return a


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(a.threads)                                       # 워커(spawn)가 물려받는다
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    will_run = not (a.count_only or a.summarize_only or a.write_cluster_map)
    if will_run and not run_permitted(a):                                     # 자료를 읽기 전에 거부한다
        raise SystemExit("[거부] 학습을 하는 실행(스모크 포함)은 Rescale 작업에서 한다. Rescale 작업 명령에 LG_RESCALE=1 또는 --allow-local 을 "
                         "준다(사용자 지시 2026-09-29: 공유 서버의 CPU 를 쓰지 않는다)")
    limit_local(a)
    if will_run and "V-G" in schemes_to_run(a) and not a.TDDM.exists():      # N1 기준선 P0@tddm 의 입력 표(비용이 들기 전에 중단한다)
        raise SystemExit(f"연도 정합 도일 표 {a.TDDM} 가 없다. h42 의 --write-tdd-matched 로 먼저 만든다(V-G 단의 N1 기준선 P0@tddm 에 쓴다)")
    V = get_vdata(a)
    if a.write_cluster_map:
        write_cluster_map(a, V)
        return dict(executed=[], skipped=[], resumed=[])
    units, skipped = enumerate_units(a, V)
    units.sort(key=unit_priority)
    print(f"[data] {V.N:,}셀 · 채점 {int(V.score.sum()):,}셀/{len(np.unique(V.gid[V.score]))}블록 · 부분 {a.part} · 단 {schemes_to_run(a)} · "
          f"반복 {len(a.REPS)}(지역 내 {len(a.WREPS)}) · fold {a.folds} · seed {a.SEEDS} · 학습기 {a.LEARNERS} · λ {a.LAMS} · "
          f"작업 단위 {len(units)}(건너뜀 {len(skipped)}) · 연도 정합 도일 표 {V.tdd_src}", flush=True)
    if a.count_only:
        if V.tdd_src == "none":
            print(f"  [warn] 연도 정합 도일 표 {a.TDDM} 가 없다. 저장 키 수는 P0@tddm 을 뺀 값이다(V-G 단위마다 1개)", flush=True)
        count_only(a, V, units, skipped)
        return dict(executed=[], skipped=skipped, resumed=[])
    if a.summarize_only:
        out = summarize(a, V, 0.0, skipped)
        n_ex = len((out or {}).get("excluded", []))
        return dict(executed=[], skipped=skipped, resumed=[], n_fail=n_ex, n_excluded=n_ex)
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    resumed, todo = [], []
    for u in units:
        ok, why = unit_state(a, *u) if a.resume else (False, "")
        if ok:
            resumed.append(u)
            print(f"  [resume] 건너뜀 {unit_name(u)} (조각 있음, 상태 {why})", flush=True)
        else:
            todo.append(u)
            if a.resume and why != "조각 없음":
                print(f"  [resume] 다시 실행 {unit_name(u)} ({why})", flush=True)
    nproc = max(int(a.workers), 1)
    print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · 프로세스 {nproc if a.workers > 0 else '없음(차례로)'} · 스레드 {a.threads}",
          flush=True)
    done, failed = [], []

    def log(u):
        done.append(u)
        print(f"  [{u['scheme']}|r{u['rep']}|f{u['fold']}] 적합 {u['n_fit_total']} · 키 {u['n_keys']} · 학습 {u['n_train']} · 채점 {u['n_test']} · "
              f"버퍼 제외 {u['n_buffer_excluded']} · {u['elapsed_s']}s · 상태 {u['status']}(실패 {u['n_fail']}) · 완료 {len(done)}/{len(todo)} · "
              f"누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append(u)
        print(f"  [FAIL] {unit_name(u)}: {repr(e)[:300]}", flush=True)

    def handle(u, fn):
        try:
            log(fn())
        except BrokenProcessPool:
            raise
        except (Exception, SystemExit) as e:                                 # noqa: BLE001  한 단위의 실패(단언 실패, 워커 안의 SystemExit)가 나머지를 막지 않게 한다
            fail(u, e)

    if todo and a.workers <= 0:                                               # 프로세스 풀 없이 이 프로세스에서 차례로 실행
        _worker_init(argv, a.threads)
        for u in todo:
            handle(u, lambda u=u: _worker_run(*u))
    elif todo:
        ctx = multiprocessing.get_context("spawn")                           # fork 후 OpenMP 충돌 회피
        pending, attempt = list(todo), 0
        hits, suspects = Counter(), []                                       # 단위별 '풀이 깨질 때 제출되어 있던 횟수'와 따로 돌릴 단위
        while pending:
            broken, running, pool_ok = [], {}, True
            with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx, initializer=_worker_init, initargs=(argv, a.threads)) as ex:
                while True:
                    while pool_ok and pending and len(running) < 2 * nproc:  # 한 번에 프로세스 수의 2배까지만 제출한다(깨질 때 제출되어 있던 단위를 좁힌다)
                        u = pending.pop(0)
                        try:
                            running[ex.submit(_worker_run, *u)] = u
                        except BrokenProcessPool:
                            broken.append(u); pool_ok = False
                    if not running:
                        break
                    fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                    for f in fin:
                        u = running.pop(f)
                        try:
                            handle(u, f.result)
                        except BrokenProcessPool:                            # 워커 비정상 종료: 남은 단위는 새 풀에서 다시 돈다
                            broken.append(u); pool_ok = False
            if not broken:
                break
            for u in broken:
                hits[u] += 1
            sus = [u for u in broken if hits[u] >= 2]                         # 두 번 제출되어 있던 단위는 원인 후보다. 마지막에 따로 돌린다
            suspects += sus
            broken = [u for u in broken if hits[u] < 2]
            if not sus:
                attempt += 1                                                 # 원인 후보를 가려내지 못한 풀 재생성만 센다
            if attempt > a.pool_retries:
                for u in broken + pending:
                    fail(u, RuntimeError(f"프로세스 풀이 원인 단위를 가려내지 못한 채 {attempt}회 깨졌다. 재시도 상한 {a.pool_retries}"))
                pending = []
                break
            pending = sorted(broken + pending, key=unit_priority)
            print(f"[pool] 워커 비정상 종료. 남은 {len(pending)} 단위로 풀을 다시 만든다(원인 후보 {len(suspects)} 단위는 마지막에 따로 돌린다. "
                  f"재시도 {attempt}/{a.pool_retries})", flush=True)
        for u in suspects:                                                   # 원인 후보: 단위마다 새 풀(프로세스 1개)에서 돌린다. 여기서도 깨지면 그 단위만 실패다
            try:
                with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_worker_init, initargs=(argv, a.threads)) as ex:
                    handle(u, ex.submit(_worker_run, *u).result)
            except BrokenProcessPool as e:
                fail(u, RuntimeError(f"단위를 따로 돌린 풀에서도 워커가 비정상 종료했다: {repr(e)[:120]}"))
    n_fail = len(failed) + sum(1 for u in done if u["status"] == "failed")
    n_part = sum(1 for u in done if u["status"] == "partial")
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 적합 실패 {n_part} · {time.time() - t0:.0f}s", flush=True)
    n_ex = 0
    if not a.no_summarize:
        out = summarize(a, V, time.time() - t0, skipped)
        n_ex = len((out or {}).get("excluded", []))
    return dict(executed=[(u["scheme"], u["rep"], u["fold"]) for u in done], skipped=skipped, resumed=list(resumed), n_fail=n_fail + n_ex,
                n_partial=n_part, n_excluded=n_ex)


if __name__ == "__main__":
    res = main()
    sys.exit(1 if res.get("n_fail") else 0)
