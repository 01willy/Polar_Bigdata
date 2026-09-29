"""H43 · TabPFN 라벨 격자(LGT). 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6C(개정 9 사전 등록)의 구현.

목적
  로컬 GPU 에서만 돌릴 수 있는 TabPFN 을 본 실행(LG, h40)과 같은 대상, 분할, 라벨 추출 seed, 채점으로 돌려 학습기 축의 빈칸을 메운다.
  짝 비교 대조군은 같은 컨텍스트 행으로 학습한 CatBoost(catboost_ctx)이고 두 학습기를 같은 프로세스에서 짝지어 적합한다.
  가설 L32–L37(모두 보조)의 대비는 LGT 조각 안에서 닫힌다. 본 실행 조각은 집계의 교차 비교(--cross-lg, 보조)에서만 읽는다.

동결 규칙
  h40, h42, h35, h41 과 src/polar 의 기존 모듈을 고치지 않는다. h40 은 importlib 로 파일 경로에서 읽어 sys.modules['h40_label_grid'] 에
  등록하고(모듈 최상위) 자료 적재(Data, get_data), 대상과 분할(build_ctx, enumerate_units), 라벨 추출(draw_cells, cells_of), 계수(ls_E),
  CatBoost 적합(cb_fit), 조각 이름과 설정 해시(shard_paths, cfg_hash, _atomic_text), 집계(TMx, contrast, build_curve, build_minn,
  timing_table)를 그대로 쓴다. h42 는 집계 함수 안에서만 읽는다(find_shards_x, make_tm, verdict4, G, TestBook, floor_table, region_stats).
  src/polar 에서는 h4_common(BlockStore, save_stores, load_stores, seed_of)만 직접 부른다.

기호
  s = e5_sqrt_tdd. 원천 행 S, 대상 A 풀, 선택 라벨 L(h40.draw_cells 의 sel), 채점 셀 B(B 블록의 eval_mask 셀). x = x25.
  E0 = 원천 전체의 최소제곱(h40.Ctx.E0). E_ls = 선택 라벨의 최소제곱. E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10, n = 0 또는 E_ls 비유한이면 E0.
  E0 와 원천 잔차 y − E0·s 는 부분 추출한 행이 아니라 원천 전체로 구한 값이다(h40.Ctx 의 E0, r0_src).

축(--axes)과 조각 tag
  main  lgt   대상 27, 분할 1–5, n {0, 3, 10, 40, 160, 320, 1000, 전량}, 추출 5, seed 2, 방법 D0·R0·R1 × 학습기 2종.
  sens  lgts  대상 12(h40.LEARNER_TARGETS), 분할 1–5, 추출 2, seed 2, 방법 D0·R1 × 학습기 2종. 변형(--sens)마다 한 요인만 바꾼다.
              d3   대상 라벨 행 3배 중복, 키의 alpha '3', n {3, 10, 40, 160}
              d10  대상 라벨 행 10배 중복, 키의 alpha '10', n {10, 40, 160}
              rid  지역 id 범주형 입력(26번째 열), 방법 이름 D0@rid·R1@rid, n {0, 10, 40, 160, 전량}
              tgt  원천 행 없이 대상 라벨 행만, 방법 이름 D0@tgt·R1@tgt, n {40, 160, 320, 1000, 전량} 가운데 실제 라벨 수 40 이상
  --smoke: GPU 1장, 대상 Russia_W x 와 Canada x, 분할 1, n {0, 10, 40, 전량}, 추출 1, seed 1, 축 main·sens, 집계 재표집 1,000회. tag 뒤에 _smoke.

학습기
  tabpfn        tabpfn 8.0.7 의 TabPFNRegressor. 공개 v2 회귀 가중치(--model-path, 기본 tabpfn-v2-regressor.ckpt), n_estimators 8,
                random_state = 학습기 seed, device cuda, ignore_pretraining_limits True, memory_saving_mode False, 출력은 평균.
                입력은 x25 원값(float32, 결측은 NaN 그대로, 표준화 없음). 범주형 지정은 없다(rid 변형만 26번째 열을 지정한다).
  catboost_ctx  catboost_lo 와 같은 초모수(반복 200, 학습률 0.05, 깊이 3, l2 3. h40.cb_fit), random_seed = 학습기 seed, 입력 x25 원값.
                학습 행은 같은 (n, 추출, seed)의 TabPFN 컨텍스트 행과 같다. 중복 행은 행 반복으로 준다(표본 가중이 아니다).

방법과 키
  P0 = E0·s, P1 = E_n·s (해석식. 모든 조각, 모든 (n, 추출) 칸에 저장한다. n = 0 의 P1 은 E0 다)
  D0  목표 = 원천 y ∪ 대상 라벨 y. 예측 f(x). λ 1.0 한 키.
  R0  목표 = 원천 y − E0·s ∪ 대상 라벨 y − E0·s. 예측 E0·s + λ·g(x).
  R1  목표 = 원천 y − E0·s ∪ 대상 라벨 y − E_n·s. 예측 E_n·s + λ·g(x). n = 0 에서는 R0 의 g 를 그대로 쓴다(추가 적합 없음).
  λ ∈ {0.25, 0.5, 1.0} 은 같은 적합에서 사후 적용한다. 판정의 기준 λ 는 D0 1.0, R0·R1 0.25 다.
  키 = (method, learner, alpha, placement, n, draw, seed, lam). 전량은 n = −1. P0 = h40.P0_KEY, P1 = ('P1', 'none', '1', 'cell', n, d, −1, 0.0).
  세 방법은 같은 (n, 추출, seed)에서 같은 컨텍스트 행렬을 쓰고 목표만 다르다. 두 학습기는 같은 행렬과 같은 목표를 받는다.
  R2, D1, R3(유사라벨을 넣는 방법)은 넣지 않는다(계획서 §6C.3: 컨텍스트 상한 안에서 LG 와 같은 정의를 쓸 수 없다).

컨텍스트 규칙(계획서 §6C.4)
  상한 C = --ctx-max(10,000). 대상 행 묶음 T = 중복 배수 × 라벨 수. 원천 행 예산 m = min(원천 행 수, C − max(T, --ctx-reserve)).
  원천 부분 추출(ctx_order): 지역 이름 오름차순으로 지역마다 RandomState(seed_of('lgt-ctx', 대상, 모드, 분할, 학습기 seed)).permutation 을
  뽑고, 순위 p 의 행에 키 u = (p + 0.5)/c_j 를 준 뒤 (u, 지역 이름, p) 순으로 정렬해 앞에서 m행을 쓴다. 작은 예산의 집합은 큰 예산의
  집합에 포함된다. seed 는 n, 추출 번호, 방법, 학습기와 무관하다.
  T > C − --ctx-src-min 이면 라벨 셀을 (C − ctx_src_min) // 중복 배수 개로 무작위 부분 추출하고(seed_of('lgt-tsub', 대상, 모드, 분할, n, 추출))
  fit_flag 에 tsub 를 남긴다. 계수 E_n 은 이 경우에도 선택 라벨 전체로 구한다. 등록 범위에서는 일어나지 않는다.
  행 순서는 원천 행(정렬 순) 다음에 대상 행이다. 채점 셀은 한 번에 예측하고 CUDA 메모리 부족 예외가 난 적합만 --pred-chunk 행 묶음으로
  다시 예측한다(fit_flag 에 chunk).

누설 규약
  대상 라벨은 선택된 n개만 컨텍스트와 계수에 쓴다. 채점 셀 라벨은 채점에만 쓴다. 표준화와 결측 대체는 하지 않는다.
  rid 변형의 지역 부호는 전체 자료의 macro 이름을 정렬한 정수이고 대상 행과 채점 셀은 새 부호(부호 수)를 쓴다(라벨을 쓰지 않는다).

조각(<out-dir>/shards, 기본 data/processed/lgt/shards)
  <tag>__gpu__<대상>__<모드>__s<분할>_{runs.csv, blocksse.npz, cells.npz, unit.json}. 조각 하나에 두 학습기의 결과와 P0, P1 이 함께 들어간다.
  기록 순서는 runs.csv → blocksse.npz → cells.npz → unit.json 이고 모두 임시 파일 뒤 os.replace 다. unit.json 이 완료 표지다.
  실행 시작 때 이전 unit.json 을 지운다. --resume 은 runs, blocksse, unit 이 있고 cfg_hash 가 같고 status 가 failed 가 아닌 조각만 건너뛴다.
  runs.csv 의 열 = h40 의 열 + ctx_set(main, d3, d10, rid, tgt), n_ctx, n_ctx_src, n_ctx_tgt, fit_s, ctx_sha(컨텍스트 행렬과 목표의 SHA-1 앞 12자).
  cells.npz: y, s, block, loc_id, lat, lon, keys(JSON [method, learner, alpha, placement, n, draw, seed, comp]), P(float32, 키 × 채점 셀),
    E(키별 앵커 계수), E0, coef_keys(JSON [n, draw])와 coef_E(E_n). comp = pred(D0 의 예측) 또는 g(잔차 성분). 예측은 E·s + λ·g 로 복원한다.
  비유한 예측이 있는 키는 BlockStore 에 넣지 않고 runs 에 n_nonfinite, fit_flag 로 남긴다. 적합 한 건의 예외는 그 키만 실패로 기록한다.
  ImportError 와 MemoryError 는 조각 전체를 실패로 둔다(unit.json 을 쓰지 않는다).

실행 제어(공유 서버 규칙, 사용자 지시 2026-09-29 밤)
  --gpus 는 필수다(빈 문자열이면 거부. CPU 로 TabPFN 을 돌리지 않는다). GPU 0–4 가 목록에 있으면 거부한다. 실행 직전에 nvidia-smi 로
  메모리 사용을 확인해 --gpu-mem-max-mib(기본 0 MiB)를 넘는 GPU 는 빼고 경고한다. GPU 하나에 워커 프로세스 하나(spawn)다.
  워커는 CUDA_VISIBLE_DEVICES 를 자기 GPU 하나로 두고 OMP, MKL, OpenBLAS, NUMEXPR, torch, CatBoost 의 스레드를 --threads 로 제한한다.
  실행은 --threads 2–4 만 허용한다. 가중치 파일이 없으면 실행하지 않는다(내려받기 시도 방지). --count-only 와 --summarize-only 는
  GPU 없이 스레드 1–2 로 돈다. 실행 순서는 (1) 주 설정·주 4지역 (2) 민감도·주 4지역 (3) 주 설정·Alaska x (4) 주 설정·나머지 학습기 축 대상
  (5) 주 설정·나머지 (6) 민감도·나머지이고 묶음 안에서는 분할 번호, 큰 대상 순이다.

집계(--summarize-only)
  lgt, lgts 조각을 읽어 곡선(h40.build_curve + 4분 판정 열), 최소 n(h40.build_minn), 가설 표(L32–L37, h42.TestBook)를 만든다.
  산출: <tag>_curve.csv, _minn.csv, _tests.csv, _cross.csv, _gate.csv, _timing.csv, _failed.csv, _targets.csv, _meta.json, _count.csv.
  실패(저장하지 못한 키, 상태가 ok 가 아닌 조각, 곡선 계산 실패)가 남으면 종료 코드 1 이다.
  교차 비교(--cross-lg)는 본 실행 cpu 조각과 P0·P1 을 대조한 뒤(허용 차 1e-9 cm, <tag>_gate.csv) 통과한 단위에만 catboost_lo 키를 병합한다.
  <tag>_cross.csv 의 모든 행은 '교차 환경(보조)'이고 <tag>_tests.csv 에 넣지 않는다.

명세(implementation_spec)와 다르게 구현한 점
  1. 시험 파일 이름은 tests/test_h43_tabpfn.py 다(과제 지시). 명세는 tests/test_h43_lgt.py 였다.
  2. --threads 의 기본값은 2 다(구현 규칙). 명세는 4 였다. 시간 추정(계획서 §6C.10)은 4스레드 기록에서 나온 값이다.
  3. GPU 를 빼는 기준은 --gpu-mem-max-mib 인자로 두고 기본값을 0 MiB 로 했다(계획서 §6C.8, 공유 서버 규칙). 구현 규칙의 500 MiB 보다 엄격하다.
  4. 셀 단위 저장(cells.npz, float32)을 조각에 더했다(구현 규칙). 명세의 조각은 세 파일이었다. --no-cells 로 끌 수 있고 --resume 의
     완료 조건에는 넣지 않는다.
  5. 가중치 경로는 h43 이 절대 경로로 바꿔 TabPFN 에 넘긴다. 파일 이름만 주면 TabPFN 캐시 디렉터리(환경 변수 TABPFN_MODEL_CACHE_DIR,
     XDG_CACHE_HOME, 없으면 ~/.cache/tabpfn)에서만 찾는다. tabpfn 8.0.7 은 파일 이름만 받으면 현재 작업 디렉터리를 먼저 본다.
  6. Holm 묶음의 주 대비(primary)는 계획서 §6C.6 대로 L32 의 기준 λ 대비 5개, L33 의 6개, L34 의 2개, L35 의 1개다. L32 의 λ 1.0 대비,
     L36, L37, 병기 대비는 primary 가 아니다. 명세는 등록 대비 전부를 primary 로 적었다.
  7. --gpus 목록은 번호가 큰 순으로 정렬해 쓴다(스모크는 가장 큰 번호 하나).
  8. --targets 를 주면 민감도 축은 그 대상과 학습기 축 12대상의 교집합만 돈다(등록 범위 밖의 민감도 조각을 만들지 않는다).
  9. 워커는 TABPFN_DISABLE_TELEMETRY=1 을 둔다(적합마다 나가는 외부 통신을 막는다. 예측값과 무관하다).
  10. 인자 --rerun-partial(--resume 에서 일부 적합이 실패한 조각도 다시 실행), --dry-build(--count-only 에서 컨텍스트 행렬을 실제로 만들어
      크기와 목표를 확인), --no-cells 를 더했다.
  11. L33 의 '우세인 최소 n' 에서 전량은 격자의 마지막 순서로 두고 'n ≤ 40' 조건에는 넣지 않는다(전량의 실제 라벨 수가 지역마다 다르다).
  12. L37 의 기준 행은 주 설정의 추출 5개를 쓴 대비이고 변형 행은 민감도의 추출 2개를 쓴 대비다. 변형 − 주 설정 대비는 공통 추출 번호로
      짝짓는다(h40.contrast 의 규칙).

확인한 것(코드 읽기, tabpfn 8.0.7)
  - 가중치 경로: model_loading.resolve_model_path 는 파일 이름만 받으면 현재 작업 디렉터리, 그다음 캐시 디렉터리를 본다(위 5번).
  - CUDA 메모리 부족 예외: predict 는 torch.OutOfMemoryError 를 tabpfn.errors.TabPFNCUDAOutOfMemoryError 로 바꿔 올린다.
    is_cuda_oom 은 예외 클래스 이름(OutOfMemoryError 계열)과 RuntimeError 의 'out of memory' 문구를 함께 본다.
  - 범주형 추론: categorical_features_indices 를 빈 목록으로 줘도 컨텍스트가 100행을 넘으면 고유값이 4개 미만인 수치 열을 범주형으로
    추론한다(preprocessing.modality_detection). x25 에 그런 열이 있으면 TabPFN 안에서 범주형으로 처리된다. --count-only 가 해당 열을 출력한다.
  - h42.TestBook 이 읽는 인자 속성은 delta_eq, delta_eq_aux 다. 재표집 횟수는 TMx 의 nboot 에서 읽는다. h42 는 import 때 파일을 쓰지 않는다.

구현 뒤 --count-only 로 확인한 값(2026-09-29, 스레드 1개, 약 22초. --dry-build 를 더하면 약 25초)
  - 작업 단위 172(main 117, sens 55), 건너뛴 분할 23. TabPFN 적합 수는 main 16,800, d3 1,424, d10 1,000, rid 1,440, tgt 1,108 로 설계 단계
    값(계획서 §6C.5)과 같다. catboost_ctx 도 같은 수다. 추정 누적 시간은 TabPFN 35.0 GPU-h, CatBoost 1.9 h 다.
  - 컨텍스트 행 수의 최댓값은 10,000, 대상 행 부분 추출(tsub)은 0건이다. --dry-build 의 행렬 점검 실패는 0건이다.
  - 지역별 행 수와 비례 배분 m·c_j/N 의 차이는 최댓값 2.78행이다. 계획서 §6C.4 는 '1행 이내'로 적었으나 정렬 키 규칙에서 나오는 상한은
    0.5·(1 − w) + w·(J − 1)/2 다(w = 그 지역의 비중, J = 원천 지역 수). 규칙은 계획서대로 구현했고 문구만 실제와 다르다.
  - 고유값이 4개 미만인 x25 열은 cci_valid 하나다(TabPFN 이 범주형으로 추론한다. CatBoost 는 수치로 쓴다).

확인하지 못한 것
  학습, 스모크, pytest 를 이 단계에서 실행하지 않았다(워크플로 규칙). 확인은 py_compile, pyflakes, --count-only 뿐이다.
  시험 파일(tests/test_h43_tabpfn.py)은 작성만 했고 실행하지 않았다. TabPFN 과 CatBoost 의 실제 적합, 조각 기록, 집계 경로(summarize,
  build_tests_t, cross_tables)는 실행해 보지 않았다. 적합 1건의 시간, GPU 메모리 최댓값, 묶음 예측과 한 번 예측의 차이는 스모크에서 잰다.
  CatBoost 의 스레드 수에 따른 재현성(2스레드와 4스레드)과 재표집 10,000회의 집계 시간은 확인하지 못했다.

실행(ROOT)
  적합 수:   python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --count-only --threads 1
  스모크:    python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --smoke --gpus 9 --threads 4
  본 실행:   python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --gpus 9,7,6,5 --threads 4 --resume --no-summarize
  집계:      python3 scripts/3_deep_learning/h43_tabpfn_label_grid.py --summarize-only --threads 2
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

THREADS_DEFAULT = 2                                 # --threads 기본값(구현 규칙)
THREADS_MAX = 4
THREADS_RUN_MIN = 2                                 # 실행은 2–4 만 허용한다
THREADS_LIGHT_MAX = 2                               # --count-only, --summarize-only 의 상한
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_threads(default=THREADS_DEFAULT):
    """numpy 를 부르기 전에 스레드 수를 정한다(--threads 를 미리 읽어 1–4 로 자른다). 학습 없는 실행은 2 이하로 둔다."""
    av = sys.argv
    val = default
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            val = av[i + 1]
        elif v.startswith("--threads="):
            val = v.split("=", 1)[1]
    try:
        n = int(val)
    except (TypeError, ValueError):
        n = int(default)
    n = max(1, min(n, THREADS_MAX))
    if "--count-only" in av or "--summarize-only" in av:
        n = min(n, THREADS_LIGHT_MAX)
    return str(n)


for _v in THREAD_VARS:
    os.environ[_v] = _peek_threads()                # setdefault 가 아니라 대입이다(부모 환경의 큰 값을 물려받지 않는다)

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(name, filename):
    """파일 경로에서 모듈을 읽어 sys.modules 에 등록한다. 이미 등록되어 있으면 그 모듈을 쓴다."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_h40():
    """h40 을 읽는다. 모듈 최상위에서 부른다(spawn 워커가 주 모듈을 다시 읽을 때 등록되어야 한다)."""
    return _load_module("h40_label_grid", "h40_label_grid.py")


def _load_h42():
    """h42 를 읽는다. 집계 함수 안에서만 부른다. h42 는 스레드 환경 변수를 setdefault 로만 건드리므로 h43 의 값이 유지된다."""
    return _load_module("h42_label_grid_ext", "h42_label_grid_ext.py")


H = _load_h40()

import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, seed_of                             # noqa: E402

# ================================================================ 고정 설계값(계획서 §6C. 바꾸면 사전 등록에서 벗어난다)
AXES = ("main", "sens")
AX_T = dict(main=dict(sfx="", draws=5, methods=("D0", "R0", "R1")),
            sens=dict(sfx="s", draws=2, methods=("D0", "R1")))
LEARNERS_T = ("tabpfn", "catboost_ctx")
TP, CBX = LEARNERS_T
FULL_GRID = "0,3,10,40,160,320,1000,all"
MAIN_SET = dict(dup=1, alpha="1", sfx="", rid=False, tgt=False, min_lab=0)
SENS_SETS = dict(
    d3=dict(dup=3, alpha="3", sfx="", rid=False, tgt=False, min_lab=0, grid=(3, 10, 40, 160)),
    d10=dict(dup=10, alpha="10", sfx="", rid=False, tgt=False, min_lab=0, grid=(10, 40, 160)),
    rid=dict(dup=1, alpha="1", sfx="@rid", rid=True, tgt=False, min_lab=0, grid=(0, 10, 40, 160, -1)),
    tgt=dict(dup=1, alpha="1", sfx="@tgt", rid=False, tgt=True, min_lab=40, grid=(40, 160, 320, 1000, -1)))
RESERVED_GPUS = (0, 1, 2, 3, 4)                     # 남겨 두는 GPU(공유 서버 규칙)
DESIGN_FITS = dict(main=16800, d3=1424, d10=1000, rid=1440, tgt=1108)      # 설계 단계의 TabPFN 적합 수(계획서 §6C.5)
NB_SEEN = (3, 10, 40)                               # b4 에서 방향을 본 n(비맹검)
GATE_TOL = 1e-9                                     # 재현 점검의 허용 차(cm)
SMOKE_CHUNK_TOL = 1e-3                              # 스모크: 묶음 예측과 한 번 예측의 허용 차(cm)
TRACE = None                                        # 시험용: 리스트를 넣으면 모든 적합의 컨텍스트 행렬과 정보를 기록한다
RUN_COLS = ["target", "mode", "parent", "split", "part", "axis", "method", "learner", "alpha", "placement", "n", "n_lab", "draw", "seed", "lam",
            "rmse_cm", "rmse_beq_cm", "bias_cm", "E_used", "alpha_sel", "n_blocks_lab", "n_nonfinite", "fit_flag",
            "ctx_set", "n_ctx", "n_ctx_src", "n_ctx_tgt", "fit_s", "ctx_sha"]


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H43 TabPFN 라벨 격자(LGT)")
    ap.add_argument("--axes", default="main,sens", help="쉼표 목록: main(주 설정), sens(민감도)")
    ap.add_argument("--sens", default="d3,d10,rid,tgt", help="민감도 변형의 쉼표 목록")
    ap.add_argument("--targets", default="", help="쉼표 목록. '이름' 또는 '이름:모드'. 기본: main 은 계획서 §1 의 27대상, sens 는 학습기 축 12대상")
    ap.add_argument("--splits", type=int, default=5, help="half_split_blocks split_seed 1..K")
    ap.add_argument("--n-grid", default=FULL_GRID, help="sens 는 변형별 고정 격자와의 교집합을 쓴다")
    ap.add_argument("--draws", type=int, default=0, help="0 = 축 기본값(main 5, sens 2). 주면 축 기본값과 비교해 작은 쪽")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lams", default="0.25,0.5,1.0")
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--ctx-max", type=int, default=10000, help="컨텍스트 총 행 수의 상한 C")
    ap.add_argument("--ctx-reserve", type=int, default=1000, help="대상 행 몫으로 비워 두는 행 수 R(원천 예산 = C − max(T, R))")
    ap.add_argument("--ctx-src-min", type=int, default=1000, help="원천 행의 최소 몫(T 가 C − 이 값을 넘으면 대상 행을 부분 추출)")
    ap.add_argument("--n-est", type=int, default=8, help="TabPFN n_estimators")
    ap.add_argument("--model-path", default="tabpfn-v2-regressor.ckpt", help="TabPFN 가중치. 파일 이름만 주면 TabPFN 캐시 디렉터리에서 찾는다")
    ap.add_argument("--pred-chunk", type=int, default=4096, help="CUDA 메모리 부족 때 다시 예측하는 묶음의 행 수")
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--gpus", default="", help="쉼표 목록(예: 9,7,6,5). 실행에 필수다. GPU 하나에 워커 하나")
    ap.add_argument("--gpu-mem-max-mib", type=int, default=0, help="실행 직전 메모리 사용이 이 값을 넘는 GPU 는 뺀다")
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT, help="프로세스당 스레드 수. 실행은 2–4 만 허용한다")
    ap.add_argument("--out-dir", default="data/processed/lgt")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--tag", default="lgt")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--delta-eq", type=float, default=0.5)
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--cross-lg", action="store_true", help="집계: 본 실행 조각과의 교차 비교(보조)와 재현 점검을 켠다")
    ap.add_argument("--lg-dir", default="data/processed/lg", help="본 실행(LG) 산출 디렉터리. 읽기 전용")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--resume", action="store_true", help="설정 해시가 같고 실패가 아닌 조각은 건너뛴다")
    ap.add_argument("--rerun-partial", action="store_true", help="--resume 에서 일부 적합이 실패한 조각(status partial)도 다시 실행한다")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--count-only", action="store_true", help="학습 없이 작업 단위, 칸, 적합 수와 추정 시간만 낸다")
    ap.add_argument("--dry-build", action="store_true", help="--count-only 에서 컨텍스트 행렬과 목표를 실제로 만들어 크기를 확인한다(학습 없음)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true", help="실행 뒤 집계를 생략한다(조각만 남긴다)")
    ap.add_argument("--no-cells", action="store_true", help="셀 단위 저장(cells.npz)을 생략한다")
    ap.add_argument("--allow-mixed-cfg", action="store_true", help="집계: 조각 사이 설정 해시가 달라도 진행한다(기본은 중단)")
    ap.add_argument("--pool-retries", type=int, default=2, help="워커 비정상 종료로 풀이 깨졌을 때 남은 단위로 풀을 다시 만드는 횟수")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(path, base=ROOT):
    return Path(path) if os.path.isabs(str(path)) else Path(base) / str(path)


def tabpfn_cache_dir():
    """TabPFN 의 캐시 디렉터리(tabpfn.model_loading.get_cache_dir 의 리눅스 규칙과 같다)."""
    v = os.environ.get("TABPFN_MODEL_CACHE_DIR", "").strip()
    if v:
        return Path(v)
    x = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(x) if x else Path.home() / ".cache") / "tabpfn"


def resolve_model(path_txt):
    """가중치 파일의 절대 경로. 절대 경로는 그대로, 디렉터리가 있는 상대 경로는 ROOT 기준, 파일 이름만 있으면 TabPFN 캐시 디렉터리."""
    p = Path(str(path_txt)).expanduser()
    if p.is_absolute():
        return p
    if p.parent != Path("."):
        return ROOT / p
    return tabpfn_cache_dir() / p.name


def finalize(a):
    a.SUFFIX = "_smoke" if a.smoke else ""
    a.TAG = a.tag + a.SUFFIX
    a.AXES = [v.strip() for v in str(a.axes).split(",") if v.strip()]
    bad = [v for v in a.AXES if v not in AXES]
    if bad:
        raise SystemExit(f"알 수 없는 축: {bad} (가능: {list(AXES)})")
    a.SENS = [v.strip() for v in str(a.sens).split(",") if v.strip()]
    bad = [v for v in a.SENS if v not in SENS_SETS]
    if bad:
        raise SystemExit(f"알 수 없는 민감도 변형: {bad} (가능: {list(SENS_SETS)})")
    if a.smoke:                                                     # 스모크: 대상 2, 분할 1, n {0, 10, 40, 전량}, 추출 1, seed 1
        a.splits = 1; a.n_grid = "0,10,40,all"; a.draws = 1; a.seeds = 1
        a.targets = a.targets or "Russia_W:x,Canada:x"
        a.nboot = min(int(a.nboot), 1000)
    a.threads_asked = int(a.threads)
    light = bool(a.count_only or a.summarize_only)
    a.threads = max(1, min(int(a.threads), THREADS_LIGHT_MAX if light else THREADS_MAX))
    try:
        a.GPUS = sorted({int(g) for g in str(a.gpus).split(",") if g.strip() != ""}, reverse=True)      # 번호가 큰 것부터 쓴다
    except ValueError:
        raise SystemExit(f"--gpus 는 정수의 쉼표 목록이다: '{a.gpus}'")
    a.N_USER = H._n_list(a.n_grid)
    a.LAMS = [float(v) for v in str(a.lams).split(",") if v.strip()]
    a.OUT = _abs(a.out_dir); a.PROC = _abs(a.data_dir); a.LGDIR = _abs(a.lg_dir)
    a.SHARDS = a.OUT / "shards"
    a.MODEL = resolve_model(a.model_path)
    a.save_cells = not a.no_cells
    a._ha = {}
    return a


def axis_tag(a, axis):
    return f"{a.tag}{AX_T[axis]['sfx']}{a.SUFFIX}"


def sens_grid(a, v):
    """변형 v 의 n 격자 = 변형의 고정 격자 ∩ --n-grid."""
    return [n for n in SENS_SETS[v]["grid"] if n in a.N_USER]


def axis_grid(a, axis):
    if axis == "main":
        return list(a.N_USER)
    have = {n for v in a.SENS for n in sens_grid(a, v)}
    return [n for n in H._n_list(FULL_GRID) if n in have]


def axis_targets(a, axis):
    """축의 대상 목록. None 은 h40 의 기본값(계획서 §1 의 27대상)이다."""
    want = H._pairs(a.targets, ["i", "x"]) if a.targets else None
    if axis == "main":
        return want
    base = list(H.LEARNER_TARGETS)
    return base if want is None else [p for p in base if p in want]


def _grid_txt(grid):
    return ",".join("all" if n == -1 else str(int(n)) for n in grid)


def h40_args_t(a, axis):
    """축 하나의 h40 형식 인자(HA). part 를 cpu 로 주는 이유는 h40.enumerate_units 가 cpu 부분에서 학습기 축 제한 없이
    (대상, 모드, 분할)을 열거하기 때문이다. 조각 이름의 부분은 gpu 다(shard_paths 에 따로 준다)."""
    if axis in a._ha:
        return a._ha[axis]
    sp = AX_T[axis]
    draws = min(int(a.draws), sp["draws"]) if a.draws else sp["draws"]
    grid = axis_grid(a, axis)
    pairs = axis_targets(a, axis)
    argv = ["--part", "cpu", "--tag", axis_tag(a, axis), "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", a.subregion_map,
            "--splits", str(int(a.splits)), "--n-grid", _grid_txt(grid) or "0", "--draws", str(draws), "--seeds", str(int(a.seeds)),
            "--threads", str(int(a.threads)), "--kappa", str(float(a.kappa)), "--lams", ",".join(str(v) for v in a.LAMS),
            "--cb-iters", str(int(a.cb_iters))]
    if pairs:
        argv += ["--targets", ",".join(f"{t}:{m}" for t, m in pairs)]
    HA = H.parse_args(argv)
    if not grid:
        HA.N_GRID = []
    if pairs is not None and not pairs:
        HA.TARGETS = []
    HA.AXIS = axis
    a._ha[axis] = HA
    return HA


# ================================================================ 컨텍스트 구성
def ctx_order(macro_src, target, mode, split, seed):
    """원천 행 색인의 전체 순서(지역 비례 층화, 중첩 구조). 앞에서 m행을 쓰면 지역별 행 수가 비례 배분에 가깝고, 작은 m 의 집합은
    큰 m 의 집합에 포함된다. 난수는 지역 이름 오름차순으로 지역마다 permutation 하나를 뽑는다(소비 순서 고정)."""
    mac = np.asarray(macro_src).astype(str)
    n = len(mac)
    rng = np.random.RandomState(seed_of("lgt-ctx", target, mode, split, seed))
    u = np.zeros(n); rank = np.zeros(n, np.int64); code = np.zeros(n, np.int64)
    for j, nm in enumerate(sorted(set(mac.tolist()))):
        idx = np.where(mac == nm)[0]
        perm = rng.permutation(len(idx))
        rank[idx[perm]] = np.arange(len(idx))                       # 순열의 p번째 행이 순위 p 를 받는다
        u[idx] = (rank[idx] + 0.5) / len(idx)
        code[idx] = j
    return np.lexsort((rank, code, u)).astype(np.int64)             # 정렬 키 (u, 지역 이름, p)


def src_budget(n_src, T, C, R):
    """원천 행 예산 m = min(원천 행 수, C − max(T, R))."""
    return int(max(min(int(n_src), int(C) - max(int(T), int(R))), 0))


def region_counts(macro_src, rows):
    mac = np.asarray(macro_src).astype(str)[np.asarray(rows, int)]
    return {str(k): int(v) for k, v in sorted(Counter(mac.tolist()).items())}


def region_dev(macro_src, rows):
    """선택한 원천 행의 지역별 행 수와 비례 배분 m·c_j/N 의 차이(절댓값)의 최댓값."""
    mac = np.asarray(macro_src).astype(str)
    rows = np.asarray(rows, int)
    if len(rows) == 0 or len(mac) == 0:
        return 0.0
    tot = Counter(mac.tolist()); got = Counter(mac[rows].tolist())
    return float(max(abs(got.get(k, 0) - len(rows) * c_ / len(mac)) for k, c_ in tot.items()))


def region_dev_bound(macro_src):
    """region_dev 의 이론 상한. 정렬 키 (u, 지역 이름, p)의 앞 m행에서 지역 j 의 행 수는 u*·c_j 와 0.5 이내로 같으므로
    비례 배분과의 차이는 0.5·(1 − w_j) + w_j·(J − 1)/2 이하다(w_j = 지역의 비중, J = 지역 수)."""
    mac = np.asarray(macro_src).astype(str)
    if len(mac) == 0:
        return 0.0
    cnt = Counter(mac.tolist())
    J = len(cnt)
    return float(max(0.5 * (1.0 - c_ / len(mac)) + (c_ / len(mac)) * (J - 1) / 2.0 for c_ in cnt.values()))


def coef_n(c, sel, kappa):
    """E_n = (n·E_ls + κ·E0)/(n + κ). n = 0 이거나 E_ls 가 비유한이면 E0 (h40.run_ctx 의 coefs 와 같은 식)."""
    m_ = len(sel)
    if m_ == 0:
        return float(c.E0)
    E_ls = H.ls_E(c.yA[sel], c.sA[sel])
    if not np.isfinite(E_ls):
        return float(c.E0)
    return float((m_ * E_ls + float(kappa) * c.E0) / (m_ + float(kappa)))


def context_rows(c, order, sel, vs, a, n, draw):
    """(원천 행 색인, 컨텍스트에 넣는 라벨 셀 색인, 표지). 대상 행이 상한에 가까우면 라벨 셀을 부분 추출한다(표지 tsub)."""
    dup = int(vs["dup"])
    tsel = np.asarray(sel, int)
    cap = int(a.ctx_max) if vs["tgt"] else int(a.ctx_max) - int(a.ctx_src_min)
    flag = ""
    if dup * len(tsel) > cap:
        k = max(cap // dup, 0)
        rng = np.random.RandomState(seed_of("lgt-tsub", c.target, c.mode, c.split, n, draw))
        tsel = np.sort(rng.choice(tsel, k, replace=False))
        flag = "tsub"
    if vs["tgt"]:
        src = np.zeros(0, np.int64)
    else:
        src = np.asarray(order[:src_budget(len(order), dup * len(tsel), a.ctx_max, a.ctx_reserve)], np.int64)
    return src, tsel, flag


def macro_codes(names):
    """지역 부호(rid 변형): 이름을 정렬한 정수. 새 부호 = 부호 수(h35 와 같다)."""
    code = {str(m): i for i, m in enumerate(sorted({str(v) for v in names}))}
    return code, len(code)


def build_X(c, src, tsel, dup, code=None, new_code=0):
    """컨텍스트 행렬(원천 행 다음에 대상 행)과 채점 행렬. code 를 주면 26번째 열로 지역 부호를 붙인다."""
    X = np.vstack([c.X_src[src], np.repeat(c.XA[tsel], int(dup), axis=0)]).astype(np.float32)
    XB = np.asarray(c.XB, np.float32)
    if code is not None:
        cs = np.array([code.get(str(m), new_code) for m in np.asarray(c.macro_src)[src]], np.float32)
        col = np.concatenate([cs, np.full(int(dup) * len(tsel), float(new_code), np.float32)])
        X = np.c_[X, col].astype(np.float32)
        XB = np.c_[XB, np.full(len(XB), float(new_code), np.float32)].astype(np.float32)
    return X, XB


def build_y(c, src, tsel, dup, kind, E_anchor=None):
    """컨텍스트 목표. kind = 'D'(y) 또는 'R'(잔차: 원천은 E0 앵커, 대상은 E_anchor 앵커)."""
    if kind == "D":
        ys, yt = c.y_src[src], c.yA[tsel]
    else:
        ys, yt = c.r0_src[src], c.yA[tsel] - float(E_anchor) * c.sA[tsel]
    return np.concatenate([np.asarray(ys, float), np.repeat(np.asarray(yt, float), int(dup))])


def ctx_sha(X, y):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X, np.float32).tobytes()); h.update(np.ascontiguousarray(y, np.float64).tobytes())
    return h.hexdigest()[:12]


# ================================================================ 학습기
def is_cuda_oom(e):
    """CUDA 메모리 부족 예외인지. tabpfn 8.0.7 의 predict 는 torch.OutOfMemoryError 를 TabPFNCUDAOutOfMemoryError 로 바꿔 올린다.
    파이썬의 MemoryError(호스트 메모리)는 여기에 들지 않는다."""
    names = {k.__name__ for k in type(e).__mro__}
    if names & {"OutOfMemoryError", "TabPFNOutOfMemoryError", "TabPFNCUDAOutOfMemoryError"}:
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _free_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                                     # noqa: BLE001
        pass


def _predict_chunks(m, XB, chunk):
    chunk = max(int(chunk), 1)
    return np.concatenate([np.asarray(m.predict(XB[k:k + chunk]), float) for k in range(0, len(XB), chunk)])


def tabpfn_fit_predict(a, X, y, XB, seed, cat_idx=None, check_chunk=0):
    """TabPFN 적합과 예측. 생성자 인자는 h35.tfm_fit_predict 와 같고 범주형 지정만 다르다(기본은 지정 없음, rid 변형은 [25]).
    반환 (예측, 표지, 정보). 예측은 한 번에 하고 CUDA 메모리 부족 예외에서만 --pred-chunk 행 묶음으로 다시 예측한다(표지 chunk).
    check_chunk > 0 이면 같은 모델로 묶음 예측을 한 번 더 해 한 번 예측과의 차이(절댓값의 최댓값)를 정보에 남긴다(스모크 전용)."""
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(device="cuda", random_state=int(seed), n_estimators=int(a.n_est), model_path=str(a.MODEL),
                        ignore_pretraining_limits=True, categorical_features_indices=[] if cat_idx is None else [int(cat_idx)],
                        memory_saving_mode=False)
    m.fit(X, y)
    flag, info = "", {}
    try:
        p = np.asarray(m.predict(XB), float)
    except Exception as e:                                                # noqa: BLE001
        if not is_cuda_oom(e):
            raise
        _free_cuda()
        p = _predict_chunks(m, XB, a.pred_chunk)
        flag = "chunk"
    if check_chunk and flag == "" and len(XB) > int(check_chunk):
        q = _predict_chunks(m, XB, check_chunk)
        info.update(chunk_rows=int(check_chunk), chunk_maxdiff=float(np.max(np.abs(q - p))))
    del m
    return p, flag, info


def cb_fit_predict(HA, X, y, XB, seed, cat_idx=None):
    """같은 컨텍스트 행으로 학습하는 CatBoost. 초모수는 catboost_lo(h40.cb_fit)다. rid 변형은 h35.cb_fit_predict 의 cat_idx 경로와
    같은 방식으로 Pool 에 범주형 열을 지정한다. 반환 (예측, 표지, 정보)."""
    th = int(HA.threads)
    if cat_idx is None:
        m = H.cb_fit(HA, X, y, seed)
        return np.asarray(m.predict(XB, thread_count=th), float), "", {}
    from catboost import CatBoostRegressor, Pool

    def pool(Z, t=None):
        d = pd.DataFrame(np.asarray(Z))
        d[int(cat_idx)] = d[int(cat_idx)].astype(int)
        return Pool(d, t, cat_features=[int(cat_idx)])
    m = CatBoostRegressor(iterations=int(HA.cb_iters), learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=int(seed), verbose=0,
                          allow_writing_files=False, thread_count=th)
    m.fit(pool(X, y))
    return np.asarray(m.predict(pool(XB), thread_count=th), float), "", {}


def est_fit_s(learner, n_ctx, n_eval):
    """적합 1건의 추정 시간(s). 계획서 §6C.10 의 식이다(--count-only 전용. 결과에는 쓰지 않는다)."""
    if learner == TP:
        return 0.5 + (6.7 + 0.6 * float(n_eval) / 1000.0) * (float(n_ctx) / 9900.0)
    return 0.4 * (max(float(n_ctx), 1.0) / 10000.0) ** 0.6


class FitterT:
    """적합 실행, 계수, 기록. dry = True 이면 학습 없이 수와 추정 시간만 센다(예측은 0)."""

    def __init__(self, a, HA, n_eval, dry=False):
        self.a, self.HA, self.n_eval, self.dry = a, HA, int(n_eval), bool(dry)
        self.n = Counter(); self.sec = Counter(); self.fail = Counter()
        self.nd = Counter(); self.secd = Counter(); self.rowsd = Counter(); self.estd = Counter()      # 키 = "축|학습기|방법"(h40.timing_table 의 형식)
        self.errors: list = []
        self.smoke: list = []                                             # 스모크의 묶음 예측 점검 기록
        self._checked: set = set()

    def check(self, X, y, XB, n_ctx, cat_idx):
        """컨텍스트 행렬과 목표의 점검. 어긋나면 ValueError 다(그 적합만 실패로 남는다)."""
        if len(y) != len(X) or len(X) != int(n_ctx):
            raise ValueError(f"컨텍스트 행 수 불일치: X {len(X)}, y {len(y)}, 기대 {int(n_ctx)}")
        if len(X) > int(self.a.ctx_max):
            raise ValueError(f"컨텍스트 행 수 {len(X)} 가 상한 {int(self.a.ctx_max)} 을 넘는다")
        if X.shape[1] != XB.shape[1] or (cat_idx is not None and int(cat_idx) != X.shape[1] - 1):
            raise ValueError(f"입력 열 수 불일치: 컨텍스트 {X.shape[1]}, 채점 {XB.shape[1]}, 범주형 열 {cat_idx}")
        if not np.all(np.isfinite(np.asarray(y, float))):
            raise ValueError("학습 목표에 비유한 값이 있다")

    def fit(self, ctx_set, learner, method, X, y, XB, seed, cat_idx, n_ctx, info=None):
        """반환 (예측 또는 잔차 성분, 표지, 시간 s, 컨텍스트 해시). 적합 단위 예외는 그 적합만 실패로 남긴다(예측 NaN, 표지 fail)."""
        k = f"{ctx_set}|{learner}|{method}"
        self.n[f"{ctx_set}:{learner}"] += 1; self.nd[k] += 1; self.rowsd[k] += int(n_ctx)
        self.estd[k] += est_fit_s(learner, n_ctx, self.n_eval)
        if self.dry:
            if X is not None:                                             # --dry-build: 행렬의 크기와 목표만 확인한다(학습 없음)
                try:
                    self.check(X, y, XB, n_ctx, cat_idx)
                except ValueError as e:
                    self.fail[f"{ctx_set}:{learner}"] += 1
                    if len(self.errors) < 20:
                        self.errors.append(f"{k}: {repr(e)[:200]}")
            return np.zeros(self.n_eval), "", 0.0, ""
        t0 = time.time(); flag, sha = "", ""
        try:
            if TRACE is not None:
                TRACE.append(dict(info or {}, ctx_set=ctx_set, learner=learner, method=method, seed=int(seed), cat_idx=cat_idx,
                                  X=np.array(X, copy=True), y=np.array(y, copy=True), XB=np.array(XB, copy=True)))
            self.check(X, y, XB, n_ctx, cat_idx)
            sha = ctx_sha(X, y)
            if learner == TP:
                chk = 0
                if self.a.smoke and ctx_set not in self._checked and self.n_eval >= 2:
                    chk = int(math.ceil(self.n_eval / 2.0))               # 스모크: 변형마다 첫 TabPFN 적합에서 채점 셀을 두 묶음으로 나눠 본다
                p, flag, ex = tabpfn_fit_predict(self.a, X, y, XB, seed, cat_idx, check_chunk=chk)
                if chk:
                    self._checked.add(ctx_set)
                    self.smoke.append(dict(ex, ctx_set=ctx_set, method=method, n_eval=self.n_eval, n_ctx=int(n_ctx)))
            else:
                p, flag, ex = cb_fit_predict(self.HA, X, y, XB, seed, cat_idx)
            p = np.asarray(p, float)
            if p.shape != (len(XB),):
                raise ValueError(f"예측의 모양 {p.shape} 이 채점 셀 수 {len(XB)} 와 다르다")
        except (ImportError, MemoryError):                                # 환경 문제: 조각 전체를 실패로 둔다(재개 때 다시 실행)
            raise
        except Exception as e:                                            # noqa: BLE001
            self.fail[f"{ctx_set}:{learner}"] += 1
            if len(self.errors) < 20:
                self.errors.append(f"{k}: {repr(e)[:200]}")
            print(f"    [warn] 적합 실패({k}): {repr(e)[:160]}", flush=True)
            p, flag = np.full(len(XB), np.nan), "fail"
            if is_cuda_oom(e):
                _free_cuda()
        dt = time.time() - t0
        self.sec[f"{ctx_set}:{learner}"] += dt; self.secd[k] += dt
        return p, flag, dt, sha


class CellBookT:
    """셀 단위 저장(float32). 성분(comp)은 D0 의 예측(pred) 또는 잔차 성분(g)이다. 예측은 E·s + λ·g 로 복원한다."""

    def __init__(self, c, loc=None, E0=np.nan):
        self.y = np.asarray(c.yB, np.float32); self.s = np.asarray(c.sB, np.float32); self.block = np.asarray(c.blkB).astype(str)
        self.loc = dict(loc or {}); self.E0 = float(E0)
        self.keys: list = []; self.P: list = []; self.E: list = []
        self.coef: dict = {}

    def add(self, method, learner, alpha, n, d, seed, comp, vec, E=np.nan):
        self.keys.append(json.dumps([str(method), str(learner), str(alpha), "cell", int(n), int(d), int(seed), str(comp)]))
        self.P.append(np.asarray(vec, np.float32)); self.E.append(float(E))

    def add_coef(self, n, d, E):
        self.coef[(int(n), int(d))] = float(E)

    def save(self, path):
        path = Path(path)
        P = np.stack(self.P).astype(np.float32) if self.P else np.zeros((0, len(self.y)), np.float32)
        ck = sorted(self.coef)
        arrs = dict(y=self.y, s=self.s, block=self.block, keys=np.array(self.keys, dtype=str), P=P, E=np.asarray(self.E, float),
                    E0=np.array(self.E0), coef_keys=np.array([json.dumps(list(k)) for k in ck], dtype=str),
                    coef_E=np.array([self.coef[k] for k in ck], float))
        for k in ("loc_id", "lat", "lon"):
            if k in self.loc and len(self.loc[k]) == len(self.y):
                arrs[k] = np.asarray(self.loc[k])
        tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, path)
        return path


# ================================================================ 작업 단위 실행
def run_ctx_t(c, axis, a, HA, dry=False, code=None, new_code=0, loc=None):
    """작업 단위 하나(축, 대상, 모드, 분할)를 실행한다. 반환 (rows, BlockStore, 통계 dict, CellBookT 또는 None).
    code = 지역 부호(rid 변형). 주지 않으면 원천 지역 이름으로 만든다(시험용 합성 자료)."""
    F = FitterT(a, HA, len(c.yB), dry)
    empty = dict(n_fit={}, sec={}, fail={}, n_rows=0, status="no_eval", n_fit_detail={}, sec_detail={}, rows_detail={}, est_detail={}, errors=[],
                 n_stored=0, n_stored_ml=0, n_nonfinite_keys=0, ctx={}, ctx_dev_max=0.0, ctx_dev_bound=0.0, n_src_regions=0, n_ctx_max=0, n_ctx_min=0,
                 flags={}, smoke_check=[])
    if len(c.yB) == 0 or len(c.yA) == 0:
        return [], None, empty, None
    st = BlockStore(f"{c.target}|{c.mode}", c.split, c.blkB, meta=dict(target=c.target, mode=c.mode, part="gpu", axis=axis))
    rows, n_rows, n_bad = [], [0], [0]
    nA = len(c.yA)
    E0 = float(c.E0)
    build = (not dry) or bool(getattr(a, "dry_build", False))
    cells = CellBookT(c, loc, E0) if (a.save_cells and not dry) else None
    if code is None:
        code, new_code = macro_codes(np.asarray(c.macro_src).tolist() + [c.parent])
    base = dict(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part="gpu", axis=axis)
    orders = {int(s_): ctx_order(c.macro_src, c.target, c.mode, c.split, int(s_)) for s_ in HA.SEEDS}
    ctx_info: dict = {}
    dev_max, nctx_all, flags = [0.0], [], Counter()

    def add(ctx_set, method, learner, alpha, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, nb_lab=0, flag="", nc=(0, 0, 0), fit_s=0.0, sha=""):
        n_rows[0] += 1
        if dry:
            return
        pred = np.asarray(pred, float)
        nf = int((~np.isfinite(pred)).sum())
        row = dict(**base, method=method, learner=learner, alpha=str(alpha), placement="cell", n=int(n), n_lab=int(n_lab), draw=int(d),
                   seed=int(seed), lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan, E_used=float(E_used), alpha_sel="",
                   n_blocks_lab=int(nb_lab), n_nonfinite=nf, fit_flag=str(flag) if nf == 0 else (str(flag) or "nonfinite"), ctx_set=ctx_set,
                   n_ctx=int(nc[0]), n_ctx_src=int(nc[1]), n_ctx_tgt=int(nc[2]), fit_s=round(float(fit_s), 3), ctx_sha=str(sha))
        if nf > 0:                                                       # 비유한 예측이 있는 키는 저장하지 않는다(방법 간 채점 셀 집합 통일)
            n_bad[0] += 1
            rows.append(row)
            return
        key = (method, learner, str(alpha), "cell", int(n), int(d), int(seed), float(lam))
        st.add(key, c.yB, pred)
        sse, cnt = st.get(key)
        with np.errstate(invalid="ignore", divide="ignore"):
            beq = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan))) if cnt.sum() else np.nan
            rm = float(np.sqrt(sse.sum() / cnt.sum())) if cnt.sum() else np.nan
        row.update(rmse_cm=rm, rmse_beq_cm=beq, bias_cm=float(np.nanmean(pred - c.yB)) if len(c.yB) else np.nan)
        rows.append(row)

    seen_p1: set = set()

    def analytic(ctx_set, n, d, E_n, nl, nb):
        """P1 = E_n·s. (n, 추출)마다 한 번 저장한다(변형과 무관한 값이다)."""
        if (n, d) in seen_p1:
            return
        seen_p1.add((n, d))
        add(ctx_set, "P1", "none", "1", n, d, -1, 0.0, E_n * c.sB, E_n, nl, nb)
        if cells is not None:
            cells.add_coef(n, d, E_n)

    add("main" if axis == "main" else "sens", "P0", "none", "1", 0, 0, -1, 0.0, E0 * c.sB, E0, 0)      # P0 은 모든 비교의 기준이므로 항상 저장한다

    def note_ctx(seed, src):
        m_ = int(len(src))
        slot = ctx_info.setdefault(str(int(seed)), {})
        if str(m_) not in slot:
            slot[str(m_)] = dict(n_src=m_, regions=region_counts(c.macro_src, src))
            dev_max[0] = max(dev_max[0], region_dev(c.macro_src, src))

    def run_set(name, vs, grid, methods):
        dup, alpha, sfx = int(vs["dup"]), str(vs["alpha"]), str(vs["sfx"])
        cat_idx = c.XB.shape[1] if vs["rid"] else None                     # rid: 26번째 열(색인 25)이 지역 부호다
        for n, d in H.cells_of(grid, HA.DRAWS, nA):
            sel = H.draw_cells(c.target, c.mode, c.split, n, d, nA)
            nl = len(sel)
            if nl < int(vs["min_lab"]):                                   # tgt: 실제 라벨 수가 40 미만인 칸은 돌리지 않는다
                continue
            nb = len(np.unique(c.blkA[sel])) if nl else 0
            E_n = coef_n(c, sel, HA.kappa)
            analytic(name, n, d, E_n, nl, nb)
            for seed in HA.SEEDS:
                src, tsel, tflag = context_rows(c, orders[int(seed)], sel, vs, a, n, d)
                nc = (len(src) + dup * len(tsel), len(src), dup * len(tsel))
                if nc[0] == 0:                                            # 컨텍스트가 비면 적합하지 않는다(원천이 없고 n = 0)
                    continue
                note_ctx(seed, src)
                nctx_all.append(nc[0])
                if tflag:
                    flags[tflag] += 1
                X = XB = None
                if build:
                    X, XB = build_X(c, src, tsel, dup, code if vs["rid"] else None, new_code)
                g0: dict = {}
                for method in methods:
                    mname = method + sfx
                    if method == "R1" and nl == 0 and g0:                 # n = 0 에서 R1 은 R0 과 같다(E_n = E0, 같은 적합)
                        for lr in LEARNERS_T:
                            g, fl, sha = g0[lr]
                            for lam in HA.LAMS:
                                add(name, mname, lr, alpha, n, d, seed, lam, E0 * c.sB + lam * g, E0, nl, nb, fl, nc, 0.0, sha)
                            if cells is not None:
                                cells.add(mname, lr, alpha, n, d, seed, "g", g, E0)
                        continue
                    kind = "D" if method == "D0" else "R"
                    E_a = None if kind == "D" else (E0 if method == "R0" else E_n)
                    y = build_y(c, src, tsel, dup, kind, E_a) if build else None
                    for lr in LEARNERS_T:
                        info = dict(n=int(n), draw=int(d), alpha=alpha, sel=np.array(sel, copy=True), tsel=np.array(tsel, copy=True),
                                    src=np.array(src, copy=True), dup=dup, E_anchor=E_a)
                        g, fl, dt, sha = F.fit(name, lr, mname, X, y, XB, seed, cat_idx, nc[0], info)
                        fl = ";".join(v for v in (tflag, fl) if v)
                        if kind == "D":
                            add(name, mname, lr, alpha, n, d, seed, 1.0, g, np.nan, nl, nb, fl, nc, dt, sha)
                        else:
                            for lam in HA.LAMS:
                                add(name, mname, lr, alpha, n, d, seed, lam, float(E_a) * c.sB + lam * g, E_a, nl, nb, fl, nc, dt, sha)
                        if method == "R0" and nl == 0:
                            g0[lr] = (g, fl, sha)
                        if cells is not None:
                            cells.add(mname, lr, alpha, n, d, seed, "pred" if kind == "D" else "g", g, np.nan if kind == "D" else E_a)

    if axis == "main":
        run_set("main", MAIN_SET, list(HA.N_GRID), AX_T["main"]["methods"])
    else:
        for v in a.SENS:
            run_set(v, SENS_SETS[v], [n for n in sens_grid(a, v) if n in HA.N_GRID], AX_T["sens"]["methods"])
    n_ml = sum(1 for k in st.keys if k[1] != "none")
    n_fail = int(sum(F.fail.values())) + int(n_bad[0])
    n_fit = int(sum(F.n.values()))
    status = "ok" if n_fail == 0 else ("failed" if (n_ml == 0 and n_fit > 0) else "partial")      # failed = 학습 결과가 하나도 저장되지 않음
    stats = dict(n_fit=dict(F.n), sec={k: round(v, 1) for k, v in F.sec.items()}, fail=dict(F.fail), n_rows=int(n_rows[0]), status=status,
                 n_fit_detail=dict(F.nd), sec_detail={k: round(v, 2) for k, v in F.secd.items()}, rows_detail=dict(F.rowsd),
                 est_detail={k: round(v, 2) for k, v in F.estd.items()}, errors=list(F.errors), n_stored=int(len(st)), n_stored_ml=int(n_ml),
                 n_nonfinite_keys=int(n_bad[0]), ctx=ctx_info, ctx_dev_max=round(float(dev_max[0]), 3),
                 ctx_dev_bound=round(region_dev_bound(c.macro_src), 3), n_src_regions=int(len(set(np.asarray(c.macro_src).astype(str).tolist()))),
                 n_ctx_max=int(max(nctx_all)) if nctx_all else 0, n_ctx_min=int(min(nctx_all)) if nctx_all else 0, flags=dict(flags),
                 smoke_check=list(F.smoke))
    return rows, st, stats, cells


# ================================================================ 조각 입출력
_SHA: dict = {}


def file_sha(path, short=0):
    """파일의 SHA-1. 읽을 수 없으면 'none'."""
    k = str(path)
    if k not in _SHA:
        try:
            h = hashlib.sha1()
            with open(path, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            _SHA[k] = h.hexdigest()
        except OSError:
            _SHA[k] = "none"
    v = _SHA[k]
    return v[:short] if (short and v != "none") else v


def code_sha_t():
    return file_sha(__file__, 12)


def pkg_version(name):
    try:
        from importlib import metadata
        return str(metadata.version(name))
    except Exception:                                                     # noqa: BLE001
        return "NA"


def env_info(gpu=False):
    """실행 환경 기록. gpu = True 이면 torch 를 불러 GPU 이름을 적는다(워커 전용)."""
    d = dict(python=platform.python_version(), numpy=str(np.__version__), pandas=str(pd.__version__), torch=pkg_version("torch"),
             tabpfn=pkg_version("tabpfn"), catboost=pkg_version("catboost"), sklearn=pkg_version("scikit-learn"), gpu_name="")
    if gpu:
        try:
            import torch
            d["torch"] = str(torch.__version__)
            if torch.cuda.is_available():
                d["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:                                                 # noqa: BLE001
            pass
    return d


def shard_paths_t(a, axis, target, mode, split):
    """조각 경로. 이름은 h40.shard_paths 가 만든다(<tag>__gpu__<대상>__<모드>__s<분할>). cells 는 h43 이 더한 파일이다."""
    p = dict(H.shard_paths(h40_args_t(a, axis), "gpu", target, mode, split))
    p["cells"] = Path(str(p["unit"])[:-len("_unit.json")] + "_cells.npz")
    return p


def unit_cfg_t(a, axis):
    """결과에 영향을 주는 설정 요약. 대상, 분할, tag, 스레드 수, GPU 번호는 넣지 않는다(조각의 정체 또는 실행 자원)."""
    HA = h40_args_t(a, axis)
    return dict(axis=axis, sens=list(a.SENS) if axis == "sens" else [], methods=list(AX_T[axis]["methods"]), learners=list(LEARNERS_T),
                n_grid=list(HA.N_GRID), sens_grid={v: sens_grid(a, v) for v in a.SENS} if axis == "sens" else {},
                draws=int(HA.DRAWS), seeds=list(HA.SEEDS), lams=list(HA.LAMS), kappa=float(HA.kappa), buffer_km=float(HA.buffer_km),
                k_sub=str(HA.k_sub), min_cells_prior=int(HA.min_cells_prior),
                subregion_map=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans",
                ctx_max=int(a.ctx_max), ctx_reserve=int(a.ctx_reserve), ctx_src_min=int(a.ctx_src_min), n_est=int(a.n_est),
                model=Path(a.MODEL).name, model_sha1=file_sha(a.MODEL), tabpfn=pkg_version("tabpfn"), cb_iters=int(HA.cb_iters), feats="x25",
                pred_chunk=int(a.pred_chunk))


def unit_state_t(a, axis, target, mode, split):
    """(완료 여부, 사유). 완료 = runs, blocksse, unit 이 있고, 설정 해시가 같고, status 가 failed 가 아니다."""
    p = shard_paths_t(a, axis, target, mode, split)
    if not (p["unit"].exists() and p["runs"].exists() and p["npz"].exists()):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != H.cfg_hash(unit_cfg_t(a, axis)):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    if getattr(a, "rerun_partial", False) and u.get("status") == "partial":
        return False, "일부 적합 실패(--rerun-partial)"
    return True, str(u.get("status", "ok"))


def _atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False); os.replace(tmp, path)


def write_shard_t(a, axis, c, rows, st, stats, cells, elapsed, extra=None):
    p = shard_paths_t(a, axis, c.target, c.mode, c.split)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    _atomic_csv(df[[k for k in RUN_COLS if k in df] + [k for k in df.columns if k not in RUN_COLS]] if len(df) else df, p["runs"])
    save_stores([st], p["npz"])
    if cells is not None:
        cells.save(p["cells"])
    cfg = unit_cfg_t(a, axis)
    unit = {**stats, **c.meta}
    unit.update(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part="gpu", axis=axis, learner="", tag=axis_tag(a, axis),
                elapsed_s=round(float(elapsed), 1), n_fit_total=int(sum(stats["n_fit"].values())), cfg=cfg, cfg_hash=H.cfg_hash(cfg),
                cfg_common=H.cfg_hash(cfg), code_sha=code_sha_t(), code_sha_h40=H.code_sha(), model_sha1=cfg["model_sha1"],
                threads=int(a.threads), device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), has_cells=bool(cells is not None))
    unit.update(extra or {})
    H._atomic_text(p["unit"], json.dumps(unit, ensure_ascii=False, indent=1, default=float))      # 완료 표지는 마지막에 쓴다
    return unit


# ================================================================ 작업 단위
def eval_cells(D, c, target, split):
    """채점 셀의 식별자와 좌표(셀 단위 저장용). h40.build_ctx 와 같은 순서로 다시 구하고 y 가 같은지 확인한다. 어긋나면 빈 dict."""
    try:
        df = D.df
        t_idx = D.target_idx(target)
        _, B_idx = H.half_split_blocks(df, t_idx, split)
        ev = B_idx[H.eval_mask(df.iloc[B_idx])]
        if len(ev) != len(c.yB) or not np.allclose(df.y.values[ev], c.yB, equal_nan=True):
            return {}
        return {k: df[k].values[ev] for k in ("loc_id", "lat", "lon") if k in df}
    except Exception:                                                     # noqa: BLE001
        return {}


def require_cuda():
    """GPU 를 쓸 수 없으면 실행하지 않는다(CPU 대체 없음)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch.cuda 를 쓸 수 없다(CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'). "
                           "TabPFN 을 CPU 로 돌리지 않는다")
    return torch


def run_unit_t(a, axis, target, mode, split, dry=False):
    t0 = time.time()
    HA = h40_args_t(a, axis)
    D = H.get_data(HA)
    c = H.build_ctx(D, HA, target, mode, split)
    code, new_code = macro_codes(D.df.macro.unique())
    torch = None
    if not dry:
        torch = require_cuda()
        torch.cuda.reset_peak_memory_stats()
        shard_paths_t(a, axis, target, mode, split)["unit"].unlink(missing_ok=True)      # 이전 세대의 완료 표지를 먼저 지운다
    loc = eval_cells(D, c, target, split) if (a.save_cells and not dry) else None
    rows, st, stats, cells = run_ctx_t(c, axis, a, HA, dry=dry, code=code, new_code=new_code, loc=loc)
    if dry:
        return dict(axis=axis, target=target, mode=mode, split=split, n_A=c.meta["n_A"], n_eval=c.meta["n_eval"], nb_eval=c.meta["nb_eval"],
                    n_src=c.meta["n_src"], valid=c.meta["valid"], n_rows=stats["n_rows"], n_ctx_max=stats["n_ctx_max"],
                    n_ctx_min=stats["n_ctx_min"], ctx_dev_max=stats["ctx_dev_max"], ctx_dev_bound=stats["ctx_dev_bound"],
                    n_src_regions=stats["n_src_regions"], n_tsub=int(stats["flags"].get("tsub", 0)),
                    n_check_fail=int(sum(stats["fail"].values())), _detail=stats["n_fit_detail"], _rows=stats["rows_detail"],
                    _est=stats["est_detail"], _errors=stats["errors"])
    if st is None:                                                        # 채점 셀 또는 A 셀이 없다(열거 단계에서 걸러지는 경우)
        raise RuntimeError(f"{target}|{mode}|s{split}: 채점 셀 또는 A 셀이 없다")
    peak = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)          # torch 가 할당한 텐서의 최댓값
    resv = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 2)           # torch 가 잡아 둔 메모리의 최댓값(nvidia-smi 값에 가깝다)
    unit = write_shard_t(a, axis, c, rows, st, stats, cells, time.time() - t0,
                         extra=dict(env=env_info(gpu=True), gpu_mem_peak_mib=round(peak, 1), gpu_mem_reserved_peak_mib=round(resv, 1)))
    del rows, st, cells
    _free_cuda()                                                          # 단위가 끝나면 GPU 캐시를 비운다(모델은 적합마다 지운다)
    return unit


def enumerate_t(a):
    """작업 단위 (축, 대상, 모드, 분할)의 목록과 건너뛴 분할의 기록. 분할 규칙은 h40.enumerate_units 와 같다."""
    units, skipped = [], []
    for axis in a.AXES:
        HA = h40_args_t(a, axis)
        if not HA.TARGETS or not HA.N_GRID or (axis == "sens" and not a.SENS):
            continue
        D = H.get_data(HA)
        us, sk = H.enumerate_units(HA, D, "cpu")
        units += [(axis, t, m, int(sp)) for t, m, sp, _ in us]
        skipped += [dict(s_, axis=axis, part="gpu") for s_ in sk]
    return units, skipped


def unit_group(axis, target, mode):
    """실행 순서의 묶음. 0 주 설정·주 4지역, 1 민감도·주 4지역, 2 주 설정·Alaska x, 3 주 설정·나머지 학습기 축 대상, 4 주 설정·나머지,
    5 민감도·나머지."""
    main4 = target in H.MAIN4 and mode == "x"
    if axis == "sens":
        return 1 if main4 else 5
    if main4:
        return 0
    if target == H.ALASKA:
        return 2
    return 3 if (target, mode) in H.LEARNER_TARGETS else 4


def priority_t(a, D, u):
    axis, t, m, sp = u
    v = D.split_structure(t)[sp]
    return (unit_group(axis, t, m), int(sp), -int(v["n_A"]), str(t), str(m))


def unit_name_t(u):
    return f"{u[0]}|{u[1]}|{u[2]}|s{u[3]}"


# ---------------------------------------------------------------- 워커
_WT = None


def _worker_init_t(argv, gpu_queue, threads):
    """워커 초기화. 큐에서 GPU 번호 하나를 받아 CUDA_VISIBLE_DEVICES 에 넣는다(torch 를 부르기 전). 스레드 수를 제한한다."""
    global _WT
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get()) if gpu_queue is not None else ""
    for v in THREAD_VARS:
        os.environ[v] = str(int(threads))
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    _WT = parse_args(argv)
    try:
        import torch
        torch.set_num_threads(int(threads))
    except Exception:                                                     # noqa: BLE001
        pass


def _worker_run_t(axis, target, mode, split):
    t0 = time.time()
    u = run_unit_t(_WT, axis, target, mode, split)
    u["wall_s"] = round(time.time() - t0, 1)
    u["n_fail"] = int(sum(u.get("fail", {}).values())) + int(u.get("n_nonfinite_keys", 0))
    keep = ("axis", "target", "mode", "split", "n_A", "n_eval", "nb_eval", "n_src", "E0", "n_fit_total", "n_rows", "n_ctx_max", "elapsed_s", "wall_s",
            "device", "status", "valid", "n_fail", "gpu_mem_peak_mib", "gpu_mem_reserved_peak_mib")
    return {k: u.get(k) for k in keep}


# ---------------------------------------------------------------- GPU 확인
def query_gpu_memory():
    """nvidia-smi 의 GPU 별 메모리 사용(MiB). 부를 수 없으면 예외를 올린다."""
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=60)
    used = {}
    for line in out.strip().splitlines():
        p = [v.strip() for v in line.split(",")]
        if len(p) >= 2:
            try:
                used[int(p[0])] = int(float(p[1]))
            except ValueError:
                continue
    return used


def screen_gpus(gpus, used, mem_max=0, reserved=RESERVED_GPUS):
    """쓸 수 있는 GPU 목록과 뺀 GPU 의 (번호, 사유) 목록. 남겨 두는 GPU 가 목록에 있으면 거부한다(SystemExit)."""
    res = [int(g) for g in gpus if int(g) in set(reserved)]
    if res:
        raise SystemExit(f"[거부] 남겨 두는 GPU {sorted(set(reserved))} 가 --gpus 에 있다: {res}")
    ok, dropped = [], []
    for g in gpus:
        g = int(g)
        if g not in used:
            dropped.append((g, "nvidia-smi 목록에 없음"))
        elif int(used[g]) > int(mem_max):
            dropped.append((g, f"메모리 사용 {int(used[g])} MiB > {int(mem_max)} MiB"))
        else:
            ok.append(g)
    return ok, dropped


def check_run_args(a):
    """실행의 거부 조건. 자료를 읽기 전에 확인한다."""
    if not a.GPUS:
        raise SystemExit("[거부] --gpus 가 비어 있다. TabPFN 은 GPU 에서만 돌린다(CPU 대체 없음). 예: --gpus 9,7,6,5")
    screen_gpus(a.GPUS, {g: 0 for g in a.GPUS})                            # 남겨 두는 GPU 확인
    if not (THREADS_RUN_MIN <= int(a.threads_asked) <= THREADS_MAX):
        raise SystemExit(f"[거부] --threads {a.threads_asked}: 실행은 프로세스당 스레드 {THREADS_RUN_MIN}–{THREADS_MAX}개만 허용한다")
    if not Path(a.MODEL).exists():
        raise SystemExit(f"[거부] TabPFN 가중치 파일이 없다: {a.MODEL}. 내려받기를 시도하지 않는다")


# ================================================================ 집계: 조각 읽기와 곡선
def find_shards_t(a, X):
    """주 설정과 민감도 조각(--axes 와 무관하게 있는 것을 모두 읽는다)."""
    out = []
    for axis in AXES:
        out += [dict(s_, axis=axis) for s_ in X.find_shards_x(a.SHARDS, axis_tag(a, axis)) if s_["part"] == "gpu"]
    return out


def read_runs_t(shards):
    frames = []
    for s_ in shards:
        if s_["runs"].stat().st_size > 1:
            f = pd.read_csv(s_["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str, ctx_set=str, ctx_sha=str), keep_default_na=False,
                            na_values=["", "nan", "NaN"])
            if len(f):
                f["tag"] = s_["tag"]
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    for c_ in ("alpha_sel", "fit_flag", "ctx_set", "ctx_sha"):
        runs[c_] = runs[c_].fillna("").astype(str) if c_ in runs else ""
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int) if "n_nonfinite" in runs else 0
    return runs.drop_duplicates(subset=["target", "mode", "split"] + list(H.KEY_COLS), keep="first")      # P0·P1 은 두 축에 모두 있다


def check_cfg_t(a, shards, units):
    """조각 사이 설정 일치 확인. (tag, 축)마다 설정 해시가 하나여야 한다. 다르면 중단한다(--allow-mixed-cfg 로 진행)."""
    info = {}
    for key in sorted({(s_["tag"], s_["axis"]) for s_ in shards}):
        us = [u for s_, u in zip(shards, units) if (s_["tag"], s_["axis"]) == key]
        hs = Counter(str(u.get("cfg_hash", "legacy")) for u in us)
        cs = Counter(str(u.get("code_sha", "legacy")) for u in us)
        ch = Counter(str(u.get("code_sha_h40", "legacy")) for u in us)
        ms = Counter(str(u.get("model_sha1", "legacy")) for u in us)
        info["|".join(key)] = dict(cfg_hash=dict(hs), code_sha=dict(cs), code_sha_h40=dict(ch), model_sha1=dict(ms), n=len(us))
        if len(cs) > 1:
            print(f"  [warn] {key}: 조각의 코드 해시가 {len(cs)}종이다 {dict(cs)}", flush=True)
        if len(hs) > 1:
            msg = f"{key}: 조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}. 설정이 다른 실행이 섞였다"
            if not a.allow_mixed_cfg:
                raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
            print("  [warn] " + msg, flush=True)
    return info


def curve_t(a, X, D, tms, runs, floor, failed):
    """대상·모드의 곡선(h40.build_curve)에 4분 판정 열(기준 P0, P1, 같은 방법의 n = 0)과 효과 크기 열을 붙인다."""
    out = []
    for nm, tm in tms.items():
        rr = runs[(runs.target == tm.target) & (runs["mode"] == tm.mode)] if len(runs) else runs
        try:
            cur = H.build_curve({nm: tm}, rr) if len(rr) else pd.DataFrame()
        except Exception as e:                                            # noqa: BLE001
            print(f"  [warn] 곡선 계산 실패 {nm}: {repr(e)[:200]}", flush=True)
            failed.append(dict(store=nm, target=tm.target, mode=tm.mode, reason=repr(e)[:200]))
            cur = pd.DataFrame()
        if len(cur):
            rec = cur.to_dict("records")
            for kind, pre in (("p0", "d_p0"), ("p1", "d_p1"), ("n0", "net_value")):
                cur[f"verdict4_{kind}"] = [X.verdict4(r_.get(f"{pre}_lo"), r_.get(f"{pre}_hi"), r_.get(f"{pre}_beq_lo"), r_.get(f"{pre}_beq_hi"),
                                                      a.delta_eq) for r_ in rec]
            fl = X.floor_of(floor, D, tm)
            eff = [X.effect_cols(d_, p_, fl) for d_, p_ in zip(cur.d_p0.values, cur.rmse_p0.values)]
            cur["delta_cm"] = [e_["delta_cm"] for e_ in eff]; cur["delta_pct_p0"] = [e_["delta_pct_p0"] for e_ in eff]
            cur["frac_reducible"] = [e_["frac_reducible"] for e_ in eff]; cur["small_effect"] = [e_["small_effect"] for e_ in eff]
            cur["floor_rmse_cm"] = fl
            out.append(cur)
        if hasattr(H4.boot_weights, "cache_clear"):
            H4.boot_weights.cache_clear()                                 # 재표집 행렬은 대상마다 비운다
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ================================================================ 집계: 가설 판정(L32–L37)
def _nl(n):
    return "all" if int(n) == -1 else str(int(n))


def _nt(n):
    return "전량" if int(n) == -1 else f"n={int(n)}"


def _pool_set(r_):
    return frozenset(v for v in str((r_ or {}).get("pool_regions", "")).split(",") if v)


def stability(X, r0, r1):
    """판정 안정성(계획서 §6A 의 L28 과 같은 규칙). 같거나 동등과 미결정 사이의 변화 = 강건, 우세·열세와 동등·미결정 사이의 변화 = 약화,
    우세와 열세가 바뀜 = 의존. 4분 판정이 없는 쪽이 있으면 판정 불가, 풀 지역 집합이 다르면 판정 불가(풀 지역 불일치)."""
    v0, v1 = X._v(r0), X._v(r1)
    if v0 in X.NA_VERDICTS or v1 in X.NA_VERDICTS or not v0 or not v1:
        return "판정 불가"
    if _pool_set(r0) != _pool_set(r1):
        return "판정 불가(풀 지역 불일치)"
    if v0 == v1:
        return "강건"
    if {v0, v1} == {"우세", "열세"}:
        return "의존"
    if (v0 in ("우세", "열세")) != (v1 in ("우세", "열세")):
        return "약화"
    return "강건"


def build_tests_t(a, tms, D, floor, X=None):
    """L32–L37 의 대비 행과 판정 행(계획서 §6C.6). LGT 의 가설은 모두 보조다(h42.CONFIRMATORY 에 없다). 분할이 기대보다 적은 대비를 쓴
    판정에는 h42.TestBook.verdict 가 '부분(분할 k/K): '을 붙인다. Holm 묶음(item 'LGT' 의 primary 행)은 L32 의 기준 λ 대비 5개,
    L33 의 6개, L34 의 2개, L35 의 1개다. L32 의 동등성 p 는 같은 묶음 안에서 따로 보정한다(eq_test)."""
    X = X or _load_h42()
    ns = SimpleNamespace(nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux))
    T = X.TestBook(ns, tms, D, floor)
    V = T.verdict
    ITEM, ALL = "LGT", -1
    LB = float(H.LAM_BASE)
    memo: dict = {}

    def C(test, label, gA, gB, **kw):
        k = (test, label, tuple(kw.get("names") or ()))
        if k not in memo:
            kw.setdefault("role", "보조")
            memo[k] = T.contrast(test, ITEM, label, gA, gB, **kw)
        return memo[k]

    def g(method, n, lr=None, lam=None, alpha="1"):
        return X.G(method, n, lam, lr, alpha)

    def single(test, label, nm, gA, gB, **kw):
        tm = tms.get(nm)
        return T.single(test, ITEM, label, nm, X.region_stats(tm, gA, gB) if tm is not None else None, **kw)

    rows_1000 = ["Lena|x", f"{H.ALASKA}|x"]                                 # n = 1,000 은 층화 평균을 내지 않고 지역 행으로 보고한다

    # ---------------- L32 학습기 동등성: R1[T] − R1[C]
    l32_n = (0, 10, 40, 160, ALL)
    base32 = [(_nt(n), C("L32", f"R1[T]-R1[C]|n{_nl(n)}|lam{LB}", g("R1", n, TP, LB), g("R1", n, CBX, LB), n=n, lam=LB, eq_test=True,
                         blind=n not in NB_SEEN)) for n in l32_n]
    lam32 = [(f"{_nt(n)} λ=1.0", C("L32", f"R1[T]-R1[C]|n{_nl(n)}|lam1.0", g("R1", n, TP, 1.0), g("R1", n, CBX, 1.0), n=n, lam=1.0, eq_test=True,
                                   primary=False, blind=n not in NB_SEEN, role="보조(λ 1.0)", aux3=False)) for n in l32_n]
    for n in (3, 320):
        C("L32", f"R1[T]-R1[C]|n{_nl(n)}|lam{LB}", g("R1", n, TP, LB), g("R1", n, CBX, LB), n=n, lam=LB, primary=False, blind=n not in NB_SEEN,
          role="보조(추가 n)")
    for nm in rows_1000:
        single("L32", f"R1[T]-R1[C]|n1000|lam{LB}", nm, g("R1", 1000, TP, LB), g("R1", 1000, CBX, LB), n=1000, lam=LB, role="보조(지역 행)")
    for m_, lam in (("D0", 1.0), ("R0", LB)):
        for n in l32_n:
            C("L32", f"{m_}[T]-{m_}[C]|n{_nl(n)}", g(m_, n, TP, lam), g(m_, n, CBX, lam), n=n, lam=lam, primary=False, blind=n not in NB_SEEN,
              role=f"보조({m_})", aux3=False)
    vb, v1 = [X._v(r_) for _, r_ in base32], [X._v(r_) for _, r_ in lam32]
    k, m, _ = X.count_valid(base32)
    k1, m1, _ = X.count_valid(lam32)
    dom = [(lab, r_) for (lab, r_), v in zip(base32, vb) if v in ("우세", "열세")]
    dom1 = [f"{lab} TabPFN {v}" for (lab, _), v in zip(lam32, v1) if v in ("우세", "열세")]
    used32, part32 = base32, None
    if dom:
        txt = "TabPFN 과 같은 컨텍스트 CatBoost 의 차이가 있는 라벨 수: " + ", ".join(f"{lab} TabPFN {r_['verdict4']}(Δ {r_['delta']:+.2f} cm)"
                                                                                  for lab, r_ in dom)
        used32, part32 = dom, X.part_mark(base32)
    elif k < m:
        txt = X.na_text(base32)
    elif all(v == "동등" for v in vb) and k1 < m1:
        txt = X.na_text(base32 + lam32)
    elif all(v == "동등" for v in vb + v1):
        txt = f"동등(한계 {float(a.delta_eq):g} cm, λ {LB:g} 와 1.0)"
        used32 = base32 + lam32
    else:
        txt = "차이를 확인하지 못함"
    if dom1 and not txt.startswith("판정 불가"):
        txt += f" [λ 1.0 의 대비(병기): {', '.join(dom1)}]"
    V("L32", ITEM, txt, X._fmt(base32 + lam32), role="보조", blind=False, used=used32, partial=part32,
      note="재현(비맹검): n ∈ {3, 10, 40} 은 b4 에서 방향을 보았다. n = 0, 160, 전량의 대비 행은 맹검이다")

    # ---------------- L33 순가치: R1[T] − P1
    l33_n = (3, 10, 40, 160, 320, ALL)
    resT = [(_nt(n), n, C("L33", f"R1[T]-P1|n{_nl(n)}", g("R1", n, TP), g("P1", n), n=n, lam=LB, blind=n not in NB_SEEN)) for n in l33_n]
    resC = [(_nt(n), n, C("L33", f"R1[C]-P1|n{_nl(n)}", g("R1", n, CBX), g("P1", n), n=n, lam=LB, primary=False, blind=n not in NB_SEEN,
                          role="보조(병기 CatBoost)")) for n in l33_n]
    for nm in rows_1000:
        single("L33", "R1[T]-P1|n1000", nm, g("R1", 1000, TP), g("P1", 1000), n=1000, lam=LB, role="보조(지역 행)")
        single("L33", "R1[C]-P1|n1000", nm, g("R1", 1000, CBX), g("P1", 1000), n=1000, lam=LB, role="보조(지역 행, 병기 CatBoost)")
    pT, pC = [(lab, r_) for lab, _, r_ in resT], [(lab, r_) for lab, _, r_ in resC]
    order33 = {n: i for i, n in enumerate(l33_n)}
    ok33 = [(lab, n, r_) for lab, n, r_ in resT if X.valid_row(r_)]
    win = sorted([q for q in ok33 if q[2]["verdict4"] == "우세"], key=lambda q: order33[q[1]])
    lose = [q for q in ok33 if q[2]["verdict4"] == "열세"]
    w4 = [q for q in win if int(q[2].get("n_ci_regions", 0)) >= len(T.m4)]
    wp = [q for q in win if int(q[2].get("n_ci_regions", 0)) < len(T.m4)]
    s4 = [q for q in w4 if 0 < q[1] <= 40]
    sp_ = [q for q in wp if 0 < q[1] <= 40]
    if not ok33:
        txt, used33 = X.na_text(pT), pT
    elif s4 or sp_:
        txt, q = "희소 라벨에서도 TabPFN 잔차의 순가치가 있다", (s4 or sp_)[0]
        used33 = [(q[0], q[2])]
    elif win:
        txt, used33 = "TabPFN 잔차의 순가치는 n > 40 에서만 확인되었다", [(win[0][0], win[0][2])]
    else:
        txt, used33 = "TabPFN 잔차의 재보정 물리식 대비 순가치는 확인되지 않았다", [(lab, r_) for lab, _, r_ in ok33]
    if ok33:
        txt += f"(우세인 최소 n: 4지역 평균 {w4[0][0] if w4 else '없음'}, 2–3지역 평균 {wp[0][0] if wp else '없음'})"
        if lose:
            txt += f". 열세인 n: {', '.join(q[0] for q in lose)}"
    diff = [lab for (lab, _, rt), (_, _, rc) in zip(resT, resC) if X.valid_row(rt) and X.valid_row(rc) and rt["verdict4"] != rc["verdict4"]]
    V("L33", ITEM, txt, f"{X._fmt(pT)} | 병기 CatBoost: {X._fmt(pC)} | 두 학습기의 4분 판정이 다른 n: {', '.join(diff) if diff else '없음'}",
      role="보조", blind=False, used=used33, partial=X.part_mark(pT) if ok33 else None,
      note="재현(비맹검): n ∈ {3, 10, 40} 은 b4 에서 방향을 보았다. n ≥ 160 과 전량의 대비 행은 맹검이다")

    # ---------------- L34 라벨 0
    ra = C("L34", "D0[T]-P0|n0", g("D0", 0, TP), g("P0", 0), n=0, lam=1.0)
    rb = C("L34", "R0[T]-P0|n0", g("R0", 0, TP), g("P0", 0), n=0, lam=LB)
    ca = C("L34", "D0[C]-P0|n0", g("D0", 0, CBX), g("P0", 0), n=0, lam=1.0, primary=False, blind=False, role="보조(병기 CatBoost)")
    cb = C("L34", "R0[C]-P0|n0", g("R0", 0, CBX), g("P0", 0), n=0, lam=LB, primary=False, blind=False, role="보조(병기 CatBoost)")
    ua = [("D0[T]-P0", ra)]
    if not X.valid_row(ra):
        txt = X.na_text(ua)
    elif X._v(ra) == "우세":
        txt = ("기각: TabPFN 직접 예측이 라벨 0 에서 P0 보다 우세하다. L1 의 결론을 'TabPFN 을 제외한 학습기 한정'으로 적고 TabPFN 의 곡선을 "
               "마스터 곡선에 더한다")
    else:
        txt = f"지지: 표형 파운데이션 모델을 학습기로 써도 직접 ML 은 라벨 0 전이에서 물리식을 넘지 못한다(4분 판정 {X._v(ra)})"
    V("L34", ITEM, txt, X._fmt(ua + [("병기 D0[C]-P0", ca)]), role="보조", blind=True, used=ua, clause="(a) 직접 예측")
    ub = [("R0[T]-P0", rb)]
    say = dict(우세="라벨 0 에서 TabPFN 잔차는 원천 계수 물리식보다 오차가 작다", 열세="라벨 0 에서 TabPFN 잔차는 원천 계수 물리식보다 오차가 크다",
               동등=f"라벨 0 에서 TabPFN 잔차와 원천 계수 물리식은 구별되지 않는다(한계 {float(a.delta_eq):g} cm)", 미결정="차이를 확인하지 못함")
    V("L34", ITEM, X.na_text(ub) if not X.valid_row(rb) else say.get(X._v(rb), X._v(rb)), X._fmt(ub + [("병기 R0[C]-P0", cb)]), role="보조",
      blind=True, used=ub, clause="(b) 잔차(방향 중립)")

    # ---------------- L35 전량
    r35 = C("L35", "R1[T]-P0|all", g("R1", ALL, TP), g("P0", 0), n=ALL, lam=LB)
    c35 = C("L35", "R1[C]-P0|all", g("R1", ALL, CBX), g("P0", 0), n=ALL, lam=LB, primary=False, blind=False, role="보조(병기 CatBoost)")
    d35 = C("L35", "D0[T]-P0|all", g("D0", ALL, TP), g("P0", 0), n=ALL, lam=1.0, primary=False, role="보조(병기 D0)")
    u35 = [("R1[T]-P0 전량", r35)]
    if not X.valid_row(r35):
        txt = X.na_text(u35)
    else:
        txt = "지지" if X._v(r35) == "우세" else f"기각(4분 판정 {X._v(r35)})"
    V("L35", ITEM, txt, X._fmt(u35 + [("병기 R1[C]-P0", c35), ("병기 D0[T]-P0", d35)]), role="보조", blind=True, used=u35,
      note="러시아 W·E 의 전량은 라벨 14–17개다")

    # ---------------- L36 직접 대 잔차(방향 중립): D0[T](λ 1.0) − R1[T](λ 0.25)
    r36 = [(_nt(n), C("L36", f"D0[T]-R1[T]|n{_nl(n)}", g("D0", n, TP), g("R1", n, TP), n=n, primary=False)) for n in l32_n]
    v36 = [X._v(r_) for _, r_ in r36]
    k, m, _ = X.count_valid(r36)
    if k < m:
        txt = X.na_text(r36)
    elif all(v == "열세" for v in v36):
        txt = "TabPFN 에서도 앵커 + 잔차 구조가 직접 구조보다 오차가 작다"
    elif any(v == "우세" for v in v36):
        up = [lab for (lab, _), v in zip(r36, v36) if v == "우세"]
        txt = (f"{', '.join(up)} 에서는 TabPFN 직접 예측이 앵커 + 잔차보다 오차가 작다. 나머지: "
               + ", ".join(f"{lab} {v}" for (lab, _), v in zip(r36, v36) if v != "우세"))
    else:
        txt = "n 별 서술: " + ", ".join(f"{lab} {v}" for (lab, _), v in zip(r36, v36))
    V("L36", ITEM, txt, X._fmt(r36), role="보조", blind=True, used=r36)

    # ---------------- L37 컨텍스트 구성 민감도
    variants = [("d3", SENS_SETS["d3"], (10, 40), False), ("d10", SENS_SETS["d10"], (10, 40), True),
                ("rid", SENS_SETS["rid"], (10, 40), False), ("tgt", SENS_SETS["tgt"], (40,), True)]
    for v, vs, ns_, blind in variants:
        al, mv = str(vs["alpha"]), "R1" + str(vs["sfx"])
        res, stab, dv = [], [], []
        for n in ns_:
            pairs = [("R1[T]-R1[C]", g("R1", n, TP), g("R1", n, CBX), g(mv, n, TP, alpha=al), g(mv, n, CBX, alpha=al)),
                     ("R1[T]-P1", g("R1", n, TP), g("P1", n), g(mv, n, TP, alpha=al), g("P1", n))]
            for lab, a0, b0, a1, b1 in pairs:
                r0 = C("L37", f"기준|{lab}|n{_nl(n)}", a0, b0, n=n, lam=LB, primary=False, blind=False, role="기준", aux3=False)
                r1 = C("L37", f"{v}|{lab}|n{_nl(n)}", a1, b1, n=n, lam=LB, primary=False, blind=blind, role="보조(변형)", aux3=False, variant=v)
                s_ = stability(X, r0, r1)
                res.append((f"{lab} {_nt(n)}", r1)); stab.append((f"{lab} {_nt(n)}", s_, X._v(r0), X._v(r1)))
                T.rows.append(dict(test_id="L37", item=ITEM, contrast=f"{v}|{lab}|n{_nl(n)}", scope="stability", variant=v, n=n, lam=LB, role="보조",
                                   primary=False, blind=bool(blind), verdict4_base=X._v(r0), verdict4_variant=X._v(r1), stability=s_))
            dv.append((_nt(n), C("L37", f"{v}-main|R1[T]|n{_nl(n)}", g(mv, n, TP, alpha=al), g("R1", n, TP), n=n, lam=LB, primary=False,
                                 blind=blind, role="보조(변형 − 주 설정)", aux3=False, variant=v)))
        states = [s_ for _, s_, _, _ in stab]
        det = "; ".join(f"{lab}: {s_}(주 설정 {b0}, 변형 {b1})" for lab, s_, b0, b1 in stab)
        if any(s_.startswith("판정 불가") for s_ in states):
            txt = "판정 불가(" + "; ".join(f"{lab} {s_}" for lab, s_, _, _ in stab if s_.startswith("판정 불가")) + ")"
        elif "의존" in states:
            txt = "의존: " + ", ".join(lab for lab, s_, _, _ in stab if s_ == "의존") + ". 한계 절에 조건을 적는다"
        elif "약화" in states:
            txt = "약화: " + ", ".join(lab for lab, s_, _, _ in stab if s_ == "약화")
        else:
            txt = "강건"
        if v == "tgt" and not txt.startswith("판정 불가"):
            for lab, r_ in dv:
                if X._v(r_) == "우세":
                    txt += f". 원천 컨텍스트는 TabPFN 잔차 예측에 기여하지 않는다({lab})"
                elif X._v(r_) == "열세":
                    txt += f". 원천 컨텍스트가 기여한다({lab})"
        V("L37", ITEM, txt, f"{det} | 변형 − 주 설정(R1[T]): {X._fmt(dv)}", role="보조", blind=bool(blind), used=res, variant=v)
    return T.frame()


# ================================================================ 집계: 교차 비교(보조)와 재현 점검
def cross_tables(a, X, D, floor, stores):
    """본 실행 cpu 조각(읽기 전용)과의 재현 점검과 교차 비교. 반환 (gate 표, cross 표, 사유).
    재현 점검은 P0, P1 의 키별 셀 가중 RMSE 를 대조한다(허용 차 1e-9 cm). 통과한 단위만 catboost_lo(alpha '1', cell)의 D0·R0·R1 키를
    저장소 사본에 병합해 R1[T] − R1[catboost_lo], R1[C] − R1[catboost_lo] 를 계산한다. 가설 판정에 쓰지 않는다."""
    ref = {(s_["target"], s_["mode"], int(s_["split"])): s_ for s_ in X.find_shards_x(a.LGDIR / "shards", a.lg_tag) if s_["part"] == "cpu"}
    if not ref:
        return pd.DataFrame(), pd.DataFrame(), f"본 실행 조각 없음({a.LGDIR / 'shards'}, tag {a.lg_tag}, 부분 cpu)"
    h40_now = H.code_sha()
    gate, merged = [], {}
    for (nm, sp), st in sorted(stores.items()):
        t, m = nm.split("|")
        row = dict(target=t, mode=m, split=int(sp), status="", n_keys=0, max_diff=np.nan, worst_key="", blocks_equal=None, passed=False,
                   code_sha_h40_ref="", code_sha_h40_now=h40_now, tol=GATE_TOL)
        s_ = ref.get((t, m, int(sp)))
        if s_ is None:
            row["status"] = "기준 조각 없음"; gate.append(row); continue
        try:
            sb = load_stores(s_["npz"]).get((nm, int(sp)))
            try:
                row["code_sha_h40_ref"] = str(json.loads(s_["unit"].read_text()).get("code_sha", ""))
            except (OSError, ValueError):
                pass
            if sb is None:
                row["status"] = "저장소 없음"; gate.append(row); continue
            row["blocks_equal"] = bool(np.array_equal(st.blocks, sb.blocks) and np.array_equal(st.ncell, sb.ncell))
            keys = [k for k in st.keys if k in sb and k[0] in ("P0", "P1") and k[1] == "none" and k[3] == "cell"]
            worst = (0.0, "")
            for k in keys:
                d_ = abs(st.rmse(k) - sb.rmse(k))
                if d_ >= worst[0]:
                    worst = (float(d_), json.dumps(list(k)))
            row.update(n_keys=len(keys), max_diff=worst[0] if keys else np.nan, worst_key=worst[1])
            row["passed"] = bool(keys and worst[0] <= GATE_TOL and row["blocks_equal"])
            row["status"] = "ok" if row["passed"] else "불일치"
            if row["passed"]:
                S, C_ = st.matrices(st.keys)
                cp = BlockStore._from_arrays(st.target, st.split, st.blocks, st.ncell, list(st.keys), S, C_, dict(st.meta))
                for k in sb.keys:
                    if k[0] in ("D0", "R0", "R1") and k[1] == H.BASE_LEARNER and str(k[2]) == "1" and k[3] == "cell":
                        cp.add_sse(k, *sb.get(k))
                merged[(nm, int(sp))] = cp
        except Exception as e:                                            # noqa: BLE001
            row["status"] = f"오류: {repr(e)[:120]}"
        gate.append(row)
    gate = pd.DataFrame(gate)
    if not merged:
        return gate, pd.DataFrame(), "재현 점검을 통과한 단위가 없다"
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in merged.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in merged})}
    ns = SimpleNamespace(nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux))
    T = X.TestBook(ns, tms, D, floor)
    LB = float(H.LAM_BASE)
    note = "교차 환경(보조): 실행 환경(하드웨어, python 과 라이브러리 판)과 교락된다. 가설 판정에 쓰지 않는다"
    for n in (0, 10, 40, 160, -1):
        for lr, lab in ((TP, "T"), (CBX, "C")):
            T.contrast("LGT-cross", "LGT", f"R1[{lab}]-R1[catboost_lo]|n{_nl(n)}", X.G("R1", n, LB, lr), X.G("R1", n, LB, H.BASE_LEARNER), n=n, lam=LB,
                       primary=False, blind=True, role="교차 환경(보조)", note=note)
        T.contrast("LGT-cross", "LGT", f"D0[C]-D0[catboost_lo]|n{_nl(n)}", X.G("D0", n, 1.0, CBX), X.G("D0", n, 1.0, H.BASE_LEARNER), n=n, lam=1.0,
                   primary=False, blind=True, role="교차 환경(보조)", note=note, aux3=False)
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()
    return gate, T.frame(), ""


# ================================================================ 집계: 스모크 점검
def smoke_report(units, runs, stores):
    """스모크에서 확인할 것: 두 학습기가 같은 행렬을 받았는지, n = 0 의 R1 = R0, 묶음 예측과 한 번 예측의 차이, 적합 1건 시간, GPU 메모리."""
    out = dict(same_matrix=None, n_pairs=0, r1_eq_r0_n0=None, n_r1_r0=0, chunk_maxdiff=None, chunk_ok=None, gpu_mem_peak_mib=None, sec_per_fit={})
    ml = runs[(runs.learner != "none") & (runs.ctx_sha != "")] if len(runs) else runs
    if len(ml):
        K = ["target", "mode", "split", "ctx_set", "method", "alpha", "n", "draw", "seed"]
        gsha = ml.drop_duplicates(subset=K + ["learner"]).groupby(K).ctx_sha.agg(["nunique", "size"])
        gsha = gsha[gsha["size"] >= 2]
        out.update(same_matrix=bool((gsha["nunique"] == 1).all()) if len(gsha) else None, n_pairs=int(len(gsha)))
    diffs = []
    for st in stores.values():
        for k in st.keys:
            if k[0] == "R1" and int(k[4]) == 0:
                k0 = ("R0",) + tuple(k[1:])
                if k0 in st:
                    diffs.append(float(np.max(np.abs(st.get(k)[0] - st.get(k0)[0]))))
    if diffs:
        out.update(r1_eq_r0_n0=bool(max(diffs) == 0.0), n_r1_r0=len(diffs))
    ch = [float(q["chunk_maxdiff"]) for u in units for q in (u.get("smoke_check") or []) if "chunk_maxdiff" in q]
    if ch:
        out.update(chunk_maxdiff=float(max(ch)), chunk_ok=bool(max(ch) <= SMOKE_CHUNK_TOL))
    pk = [float(u["gpu_mem_peak_mib"]) for u in units if u.get("gpu_mem_peak_mib") is not None]
    if pk:
        out["gpu_mem_peak_mib"] = float(max(pk))
    rv = [float(u["gpu_mem_reserved_peak_mib"]) for u in units if u.get("gpu_mem_reserved_peak_mib") is not None]
    out["gpu_mem_reserved_peak_mib"] = float(max(rv)) if rv else None
    return out


# ================================================================ 집계
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                                     # noqa: BLE001
        return "NA"


def failed_rows(runs):
    """저장하지 못한 키(비유한 예측 또는 적합 실패). chunk, tsub 표지는 실패가 아니다(저장된 키다)."""
    if not len(runs):
        return runs
    bad = runs.fit_flag.astype(str).map(lambda v: any(q in ("fail", "nonfinite") for q in v.split(";")))
    return runs[(runs.n_nonfinite > 0) | bad]


def summarize(a, elapsed=0.0, skipped=None):
    """집계(로컬, 프로세스 1개). 반환 dict 의 n_fail 은 저장하지 못한 키, 상태가 ok 가 아닌 조각, 곡선 계산 실패의 합이다."""
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    X = _load_h42()
    sh = find_shards_t(a, X)
    if not sh:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{axis_tag(a, 'main')}__gpu__*, {axis_tag(a, 'sens')}__gpu__*", flush=True)
        return None
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    cfg_info = check_cfg_t(a, sh, units)
    runs = read_runs_t(sh)
    stores = load_stores([s_["npz"] for s_ in sh])
    D = H.get_data(h40_args_t(a, "main"))
    floor, floor_meta = X.floor_table(D.df)
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in stores})}
    curve_failed: list = []
    cur = curve_t(a, X, D, tms, runs, floor, curve_failed)
    mn = H.build_minn(cur) if len(cur) else pd.DataFrame()
    tests = build_tests_t(a, tms, D, floor, X)
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()
    gate, cross, cross_note = pd.DataFrame(), pd.DataFrame(), "교차 비교를 켜지 않았다(--cross-lg)"
    if a.cross_lg:
        if a.SUFFIX:
            cross_note = "스모크 조각은 본 실행 조각과 교차 비교하지 않는다"
        else:
            gate, cross, cross_note = cross_tables(a, X, D, floor, stores)
        if cross_note:
            print(f"[summarize] 교차 비교: {cross_note}", flush=True)
    failed = failed_rows(runs)
    flagged = runs[runs.fit_flag.astype(str) != ""] if len(runs) else runs
    tg = pd.DataFrame([dict({k: v for k, v in u.items() if not isinstance(v, (dict, list))}, n_fit=json.dumps(u.get("n_fit", {})),
                            sec=json.dumps(u.get("sec", {})), fail=json.dumps(u.get("fail", {})),
                            errors=json.dumps(u.get("errors", []), ensure_ascii=False), env=json.dumps(u.get("env", {}), ensure_ascii=False),
                            ctx=json.dumps(u.get("ctx", {}), ensure_ascii=False)) for u in units])
    if skipped:
        tg = pd.concat([tg, pd.DataFrame(skipped)], ignore_index=True)
    tt = H.timing_table(units)
    O = a.OUT; O.mkdir(parents=True, exist_ok=True)
    _atomic_csv(cur, O / f"{a.TAG}_curve.csv"); _atomic_csv(mn, O / f"{a.TAG}_minn.csv"); _atomic_csv(tests, O / f"{a.TAG}_tests.csv")
    _atomic_csv(cross, O / f"{a.TAG}_cross.csv"); _atomic_csv(gate, O / f"{a.TAG}_gate.csv"); _atomic_csv(tt, O / f"{a.TAG}_timing.csv")
    fa = failed.copy() if len(failed) else pd.DataFrame(columns=list(runs.columns) if len(runs) else RUN_COLS)
    if curve_failed:
        fa = pd.concat([fa, pd.DataFrame([dict(target=q["target"], mode=q["mode"], fit_flag="curve_failed", method=q["reason"]) for q in curve_failed])],
                       ignore_index=True)
    _atomic_csv(fa, O / f"{a.TAG}_failed.csv")
    _atomic_csv(tg.sort_values(["target", "mode", "split", "axis"]) if len(tg) else tg, O / f"{a.TAG}_targets.csv")
    status = Counter(str(u.get("status", "ok")) for u in units)
    n_fail = int(len(failed)) + int(sum(v for k, v in status.items() if k != "ok")) + int(len(curve_failed))
    lt = pd.DataFrame()
    if len(tt):
        lt = tt.groupby(["axis", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), sec=("sec", "sum"), rows=("rows_per_fit", "mean"))
        lt["sec_per_fit"] = lt.sec / lt.n_fit.clip(lower=1)
        print("[summarize] 변형·학습기별 적합 시간(전 조각 합계)\n" + lt.to_string(index=False), flush=True)
    smoke = smoke_report(units, runs, stores) if a.smoke else {}
    if smoke:
        smoke["sec_per_fit"] = {f"{r_.axis}|{r_.learner}": round(float(r_.sec_per_fit), 3) for r_ in lt.itertuples()} if len(lt) else {}
        print(f"[smoke] 같은 행렬 {smoke['same_matrix']}(짝 {smoke['n_pairs']}) · n = 0 의 R1 = R0 {smoke['r1_eq_r0_n0']}(키 {smoke['n_r1_r0']}) · "
              f"묶음 예측 차이 {smoke['chunk_maxdiff']} cm(허용 {SMOKE_CHUNK_TOL}) · GPU 메모리 최댓값 할당 {smoke['gpu_mem_peak_mib']} MiB, "
              f"예약 {smoke['gpu_mem_reserved_peak_mib']} MiB", flush=True)
    fit_tot = Counter()
    for u in units:
        for k, v in (u.get("n_fit") or {}).items():
            fit_tot[k] += int(v)
    meta = dict(stage="H43/LGT", plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6C", git_commit=git_commit(), tag=a.TAG,
                tags={ax: axis_tag(a, ax) for ax in AXES}, args={k: v for k, v in vars(a).items() if k.islower() and not k.startswith("_")},
                key_fields=list(H.KEY_COLS), p0_key=list(H.P0_KEY), all_n=-1, learners=list(LEARNERS_T), n_units=len(units),
                n_units_axis=dict(Counter(s_["axis"] for s_ in sh)), n_fit=dict(fit_tot), n_fit_total=int(sum(fit_tot.values())),
                unit_elapsed_s_sum=float(sum(float(u.get("elapsed_s", 0) or 0) for u in units)), elapsed_s=round(float(elapsed), 1),
                n_rows=int(len(runs)), n_curve=int(len(cur)), n_minn=int(len(mn)), n_tests=int(len(tests)), n_cross=int(len(cross)),
                n_gate=int(len(gate)), cross_note=cross_note, shard_cfg=cfg_info, unit_status=dict(status), n_failed_keys=int(len(failed)),
                n_flagged_rows=int(len(flagged)), flags=dict(Counter(flagged.fit_flag.astype(str))) if len(flagged) else {},
                curve_failed=curve_failed, n_fail=n_fail, skipped=skipped or [], floor=floor_meta, smoke=smoke,
                gpu_mem_peak_mib=max([float(u.get("gpu_mem_peak_mib") or 0.0) for u in units] or [0.0]),
                env=sorted({json.dumps(u.get("env", {}), ensure_ascii=False, sort_keys=True) for u in units}),
                code_sha=code_sha_t(), code_sha_h40=H.code_sha(), code_sha_h42=file_sha(SCRIPT_DIR / "h42_label_grid_ext.py", 12),
                delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux), nboot=int(a.nboot),
                rules=dict(ci="h4_common.boot_delta_blocks: 분할 안 채점 블록 재표집, 방법·추출·seed 공통 인덱스, 분할 분포 평균",
                           verdict4="우세 = 두 가중의 CI 상한 < 0, 열세 = 두 가중의 CI 하한 > 0, 동등 = 네 끝값의 절댓값이 한계 이하, 그 밖은 미결정",
                           hypotheses="L32–L37 은 모두 보조다. 확인적 가설 집합(계획서 §6A.2)을 바꾸지 않는다",
                           holm="item LGT 의 primary 행: L32 기준 λ 5개, L33 6개, L34 2개, L35 1개. 보조 열이며 판정에 쓰지 않는다",
                           base_lam="D0 1.0, R0·R1 0.25", cross="교차 환경(보조). 가설 판정에 쓰지 않는다",
                           failed="비유한 예측 또는 적합 실패 키. chunk 와 tsub 표지는 저장된 키다"),
                summarize_s=round(time.time() - t0, 1))
    H._atomic_text(O / f"{a.TAG}_meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    if n_fail:
        print(f"[summarize] 실패 기록: 조각 상태 {dict(status)} · 저장하지 못한 키 {len(failed):,} · 곡선 실패 {len(curve_failed)} → {a.TAG}_failed.csv",
              flush=True)
    print(f"[summarize] 조각 {len(sh)} · runs {len(runs):,} · curve {len(cur):,} · minn {len(mn):,} · tests {len(tests):,} · cross {len(cross):,} · "
          f"{time.time() - t0:.0f}s → {O}/{a.TAG}_*", flush=True)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])] if len(tests) else tests
    if len(v):
        print(v[[c_ for c_ in ("test_id", "role", "blind", "verdict", "stat") if c_ in v]].to_string(index=False), flush=True)
    return dict(curve=cur, minn=mn, tests=tests, cross=cross, gate=gate, targets=tg, timing=tt, failed=fa, meta=meta, n_fail=n_fail)


# ================================================================ 적합 수(--count-only)
def low_card_columns(D, thr=4):
    """고유값이 thr 개 미만인 x25 열. tabpfn 8.0.7 은 컨텍스트가 100행을 넘으면 이런 수치 열을 범주형으로 추론한다."""
    out = {}
    for c_ in H.FEATS:
        v = pd.Series(D.df[c_].values).dropna()
        k = int(v.nunique())
        if k < int(thr):
            out[str(c_)] = k
    return out


def count_only(a, D, units, skipped):
    """학습 없이 작업 단위, 적합 수, 추정 시간을 낸다. 추정식은 계획서 §6C.10 이다. <tag>_count.csv 에 쓴다."""
    t0 = time.time()
    rows, det, nrw, est, errs = [], Counter(), Counter(), Counter(), []
    for i, u in enumerate(units):
        r_ = run_unit_t(a, *u, dry=True)
        d_, w_, e_ = r_.pop("_detail"), r_.pop("_rows"), r_.pop("_est")
        errs += [f"{unit_name_t(u)}: {v}" for v in r_.pop("_errors")]
        for k, v in d_.items():
            det[k] += v; nrw[k] += w_.get(k, 0); est[k] += e_.get(k, 0.0)
        per = Counter()
        for k, v in d_.items():
            cs, lr, _ = (k.split("|") + ["", ""])[:3]
            per[f"fit_{cs}_{lr}"] += v; per[f"fit_{lr}"] += v
            per[f"est_{lr}_s"] += e_.get(k, 0.0)
        r_.update({k: (round(v, 1) if k.startswith("est_") else int(v)) for k, v in per.items()})
        r_.update(group=unit_group(u[0], u[1], u[2]), order=i)
        rows.append(r_)
    df = pd.DataFrame(rows).fillna(0)
    a.OUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(df, a.OUT / f"{a.TAG}_count.csv")
    tab = []
    for k in sorted(det):
        cs, lr, m_ = (k.split("|") + ["", ""])[:3]
        tab.append(dict(ctx_set=cs, learner=lr, method=m_, n_fit=int(det[k]), rows_per_fit=round(nrw[k] / max(det[k], 1)), est_h=est[k] / 3600.0))
    tab = pd.DataFrame(tab)
    print(f"[count-only] 작업 단위 {len(units)}(축별 {dict(Counter(u[0] for u in units))}) · 건너뜀 {len(skipped)}"
          f"({', '.join(sorted({str(s_['status']) for s_ in skipped})) or '없음'}) · {time.time() - t0:.0f}s", flush=True)
    summary = dict(units=len(units), units_axis=dict(Counter(u[0] for u in units)), fits={}, est_h={}, design={}, design_match=None)
    if len(tab):
        by = tab.groupby(["ctx_set", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), est_h=("est_h", "sum"))
        by["est_h"] = by.est_h.round(2)
        print("[count-only] 변형·학습기별 적합 수와 추정 누적 시간(h)\n" + by.to_string(index=False), flush=True)
        bm = tab.groupby(["ctx_set", "method", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), rows_per_fit=("rows_per_fit", "mean"))
        print("[count-only] 변형·방법별 적합 수\n" + bm.to_string(index=False), flush=True)
        tp = by[by.learner == TP].set_index("ctx_set").n_fit.to_dict()
        full = (not a.smoke) and (not a.targets) and a.N_USER == H._n_list(FULL_GRID) and int(a.splits) == 5 and not a.draws and int(a.seeds) == 2 \
            and set(a.SENS) == set(SENS_SETS) and set(a.AXES) == set(AXES)
        cmp_ = {k: dict(design=int(v), count=int(tp.get(k, 0)), same=bool(int(tp.get(k, 0)) == int(v))) for k, v in DESIGN_FITS.items()}
        summary.update(fits={f"{r_.ctx_set}|{r_.learner}": int(r_.n_fit) for r_ in by.itertuples()},
                       est_h={f"{r_.ctx_set}|{r_.learner}": float(r_.est_h) for r_ in by.itertuples()}, design=cmp_,
                       design_match=bool(all(v["same"] for v in cmp_.values())) if full else None)
        if full:
            print("[count-only] 설계 단계 값(계획서 §6C.5)과의 대조(TabPFN 적합 수): "
                  + " · ".join(f"{k} 설계 {v['design']:,} / 계산 {v['count']:,}{'' if v['same'] else ' (다르다)'}" for k, v in cmp_.items()), flush=True)
            if not summary["design_match"]:
                print("[count-only] 설계 단계 값과 다르다. 계획서 §6C.5 의 표를 이 값으로 고치고 개정 이력에 적는다", flush=True)
        else:
            print("[count-only] 등록 범위 전체가 아니므로 설계 단계 값과 대조하지 않는다", flush=True)
        h_tp = float(by[by.learner == TP].est_h.sum()); h_cb = float(by[by.learner == CBX].est_h.sum())
        tot = int(by[by.learner == TP].n_fit.sum())
        print(f"[count-only] TabPFN 적합 {tot:,}건 · 추정 누적 {h_tp:.1f} GPU-h · catboost_ctx {int(by[by.learner == CBX].n_fit.sum()):,}건 "
              f"추정 누적 {h_cb:.1f} h(4스레드 기준) · 합계 {h_tp + h_cb:.1f} h", flush=True)
        for w in (1, 2, 4):
            print(f"  GPU {w}장(프로세스 {w}개): 벽시계 추정 {(h_tp + h_cb) / w:.1f} h · 추정의 2배 {2 * (h_tp + h_cb) / w:.1f} h", flush=True)
        summary.update(n_fit_tabpfn=tot, est_h_tabpfn=round(h_tp, 2), est_h_catboost=round(h_cb, 2))
    if len(df):
        g = df.groupby(["axis", "group"], as_index=False).agg(units=("split", "size"), fit_tabpfn=(f"fit_{TP}", "sum"), est_tabpfn_s=(f"est_{TP}_s", "sum"),
                                                               n_ctx_max=("n_ctx_max", "max"), ctx_dev_max=("ctx_dev_max", "max"))
        g["est_tabpfn_h"] = (g.est_tabpfn_s / 3600.0).round(2)
        print("[count-only] 실행 순서 묶음별(0 주 설정·주 4지역, 1 민감도·주 4지역, 2 주 설정·Alaska x, 3 주 설정·나머지 학습기 축, 4 주 설정·나머지, "
              "5 민감도·나머지)\n" + g.drop(columns="est_tabpfn_s").to_string(index=False), flush=True)
        print(f"[count-only] 컨텍스트 행 수: 최댓값 {int(df.n_ctx_max.max()):,}(상한 {int(a.ctx_max):,}) · 최솟값 {int(df[df.n_ctx_min > 0].n_ctx_min.min()) if (df.n_ctx_min > 0).any() else 0:,} · "
              f"대상 행 부분 추출(tsub) {int(df.n_tsub.sum())}건 · 지역별 행 수와 비례 배분의 차이 최댓값 {float(df.ctx_dev_max.max()):.2f}행"
              f"(원천 지역 수 {int(df.n_src_regions.min())}–{int(df.n_src_regions.max())}, 이론 상한을 넘는 단위 "
              f"{int((df.ctx_dev_max > df.ctx_dev_bound + 1e-6).sum())}건)", flush=True)
        summary.update(n_ctx_max=int(df.n_ctx_max.max()), n_tsub=int(df.n_tsub.sum()), ctx_dev_max=float(df.ctx_dev_max.max()),
                       ctx_dev_bound_max=float(df.ctx_dev_bound.max()), n_over_bound=int((df.ctx_dev_max > df.ctx_dev_bound + 1e-6).sum()),
                       n_src_regions=[int(df.n_src_regions.min()), int(df.n_src_regions.max())], n_check_fail=int(df.n_check_fail.sum()))
        if a.dry_build:
            print(f"[count-only] --dry-build: 컨텍스트 행렬 점검 실패 {int(df.n_check_fail.sum())}건" + (f" · {errs[:5]}" if errs else ""), flush=True)
    lc = low_card_columns(D)
    summary["low_card_x25"] = lc
    print(f"[count-only] 고유값 4개 미만인 x25 열(TabPFN 이 범주형으로 추론하는 열): {lc if lc else '없음'}", flush=True)
    print(f"[count-only] 가중치 파일 {a.MODEL}: {'있음' if Path(a.MODEL).exists() else '없음'}"
          + (f" · {Path(a.MODEL).stat().st_size:,}바이트 · SHA-1 {file_sha(a.MODEL)}" if Path(a.MODEL).exists() else ""), flush=True)
    H._atomic_text(a.OUT / f"{a.TAG}_count_meta.json", json.dumps(dict(summary, skipped=skipped, errors=errs[:50], tag=a.TAG, code_sha=code_sha_t(),
                                                                      args={k: v for k, v in vars(a).items() if k.islower() and not k.startswith("_")}),
                                                                 ensure_ascii=False, indent=1, default=str))
    return df, summary


# ================================================================ 실행
def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                                     # noqa: BLE001
        return False


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in THREAD_VARS:
        os.environ[v] = str(int(a.threads))                               # 워커(spawn)가 물려받는다
    will_run = not (a.count_only or a.summarize_only)
    if will_run:
        check_run_args(a)                                                 # 자료를 읽기 전에 거부한다
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                           # 학습 없는 실행은 GPU 를 쓰지 않는다
    if a.summarize_only:
        res = summarize(a, 0.0, None)
        return dict(executed=[], skipped=[], resumed=[], n_fail=int(res["n_fail"]) if res else 0)
    units, skipped = enumerate_t(a)
    D = H.get_data(h40_args_t(a, a.AXES[0] if a.AXES else "main"))
    units.sort(key=lambda u: priority_t(a, D, u))
    print(f"[data] {len(D.df):,}셀 · 축 {a.AXES} · 민감도 {a.SENS} · 분할 1–{int(a.splits)} · n {a.N_USER} · seed {int(a.seeds)} · λ {a.LAMS} · "
          f"하위 지역 {D.sub_src} · 작업 단위 {len(units)}(건너뜀 {len(skipped)}) · 스레드 {a.threads}", flush=True)
    if a.count_only:
        count_only(a, D, units, skipped)
        return dict(executed=[], skipped=skipped, resumed=[], n_fail=0)
    try:
        used = query_gpu_memory()
    except Exception as e:                                                # noqa: BLE001
        raise SystemExit(f"[거부] nvidia-smi 로 GPU 상태를 확인할 수 없다: {repr(e)[:200]}")
    gpus, dropped = screen_gpus(a.GPUS, used, a.gpu_mem_max_mib)
    for g_, why in dropped:
        print(f"[warn] GPU {g_} 를 뺀다: {why}", flush=True)
    if not gpus:
        raise SystemExit("[거부] 쓸 수 있는 GPU 가 없다(지정한 GPU 가 모두 사용 중이다)")
    if a.smoke:
        gpus = gpus[:1]                                                   # 스모크는 GPU 1장
    if not cuda_available():
        raise SystemExit("[거부] torch.cuda 를 쓸 수 없다. TabPFN 을 CPU 로 돌리지 않는다")
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    resumed, todo = [], []
    for u in units:
        ok, why = unit_state_t(a, *u) if a.resume else (False, "")
        if ok:
            resumed.append(u)
            print(f"  [resume] 건너뜀 {unit_name_t(u)} (조각 있음, 상태 {why})", flush=True)
        else:
            todo.append(u)
            if a.resume and why != "조각 없음":
                print(f"  [resume] 다시 실행 {unit_name_t(u)} ({why})", flush=True)
    print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · GPU {gpus}(프로세스 {len(gpus)}개) · 스레드 {a.threads} · 가중치 {a.MODEL.name} "
          f"SHA-1 {file_sha(a.MODEL, 12)} · tabpfn {pkg_version('tabpfn')} · n_estimators {a.n_est} · 컨텍스트 상한 {a.ctx_max:,}", flush=True)
    done, failed = [], []

    def log(u):
        done.append(u)
        print(f"  [{u['axis']}|{u['target']}|{u['mode']}|s{u['split']}] 적합 {u['n_fit_total']} · 행 {u['n_rows']} · 컨텍스트 최대 {u['n_ctx_max']} · "
              f"A {u['n_A']} · 채점 {u['n_eval']}/{u['nb_eval']}블록{'' if u['valid'] else '(무효 분할)'} · 원천 {u['n_src']} · {u['elapsed_s']}s · "
              f"GPU '{u['device']}' 메모리 최댓값 {u['gpu_mem_peak_mib']} MiB(예약 {u['gpu_mem_reserved_peak_mib']} MiB) · 상태 {u['status']}(실패 {u['n_fail']}) · 완료 {len(done)}/{len(todo)} · "
              f"누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append(u)
        print(f"  [FAIL] {unit_name_t(u)}: {repr(e)[:300]}", flush=True)

    if todo:
        ctx = multiprocessing.get_context("spawn")                        # fork 후 OpenMP·CUDA 충돌 회피
        remaining, attempt = list(todo), 0
        while remaining:
            q = ctx.Queue()
            for g_ in gpus:
                q.put(g_)
            broken = []
            with ProcessPoolExecutor(max_workers=len(gpus), mp_context=ctx, initializer=_worker_init_t, initargs=(argv, q, a.threads)) as ex:
                futs = {ex.submit(_worker_run_t, *u): u for u in remaining}
                for f in as_completed(futs):
                    try:
                        log(f.result())
                    except BrokenProcessPool:                             # 워커 비정상 종료: 남은 단위는 새 풀에서 다시 돈다
                        broken.append(futs[f])
                    except Exception as e:                                # noqa: BLE001  한 단위의 실패가 나머지를 막지 않게 한다
                        fail(futs[f], e)
            if not broken:
                break
            attempt += 1
            if attempt > int(a.pool_retries):
                for u in broken:
                    fail(u, RuntimeError(f"프로세스 풀이 {attempt}회 깨졌다. 재시도 상한 {a.pool_retries}"))
                break
            remaining = sorted(broken, key=lambda u: priority_t(a, D, u))
            print(f"[pool] 워커 비정상 종료. 남은 {len(remaining)} 단위로 풀을 다시 만든다(재시도 {attempt}/{a.pool_retries})", flush=True)
    n_fail = len(failed) + sum(1 for u in done if u["status"] == "failed")
    n_part = sum(1 for u in done if u["status"] == "partial")
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 적합 실패 {n_part} · {time.time() - t0:.0f}s", flush=True)
    if not a.no_summarize:
        res = summarize(a, time.time() - t0, skipped)
        if res:
            n_fail += int(res["n_fail"])
    return dict(executed=[(u["axis"], u["target"], u["mode"], u["split"]) for u in done], skipped=skipped, resumed=list(resumed), n_fail=n_fail,
                n_partial=n_part)


if __name__ == "__main__":
    _res = main()
    sys.exit(1 if _res.get("n_fail") else 0)
