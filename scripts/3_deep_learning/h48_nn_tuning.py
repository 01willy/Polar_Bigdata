"""H48 · 신경망 초모수 강건성(LGF 의 N 부분, tag lgfn). 계획 docs/EXPERIMENT_PLAN_LGF_2026-09-29.md 4절(개정 1, 커밋 f208516)의 구현.

목적
  LG 학습기 축의 판별 신경망 4종(mlp, tabm, ftt, realmlp)을 두 판으로 적합해 D0·R1 에서 짝짓는다. 기본판은 h40 의 코드 경로를 이 서버에서
  다시 적합한 것이고, 조정판은 원천 지역 하나 제외 교차검증(30셀 이상 macro 지역, 대상 라벨 미사용)으로 고른 초모수다. 두 판과 동반 적합
  (P0, P1, catboost_lo, catboost_tuned_loc)은 같은 조각 묶음 안에서 대비를 닫는다(가설 LGF-N1–N4, N2s).

동결 규칙
  h40, h42, h43 과 src/polar 의 기존 모듈을 고치지 않는다. h40('h40_label_grid')과 h43('h43_tabpfn_label_grid')은 모듈 최상위에서 importlib 로
  읽고(GPU 확인, 잠금, pdeathsig, 풀 정리, 계수 E_n 함수를 다시 쓴다), h42('h42_label_grid_ext')는 CPU 작업과 집계 함수 안에서 한 번만 읽는다.
  h43 의 SHA-1 은 13148bb8…, h40 의 SHA-1 은 ffd0b3a7… 이어야 한다. --h43-sha, --h40-sha 로 새 값을 주면 받아들이되 조각 설정에 '개정 필요'
  표지를 남긴다. h42 의 SHA-1 은 기록만 한다.

작업(--jobs)과 작업 단위
  n_sel_fast  mlp, tabm 의 선택. 단위 = (학습기, 종류, 대상, 모드, 단계, 시도, seed). 1단계는 모든 시도를 seed 0 으로, 2단계는 1단계 주 점수 상위
              k2 개(시도 0 제외)와 시도 0 을 seed 1·2 로 적합한다. 단위 하나가 fold 전부를 적합하고 nsel/<tag>__sel__…json 하나를 쓴다.
  n_p0_fast   mlp, tabm 의 통과 0(n = 0) 공유 적합. 단위 = (학습기, 대상, 모드). 판(기본, 조정, 민감도 cw·l25) × 종류 × seed 마다 한 번 적합해
              분할 1–5 의 채점 셀을 한꺼번에 예측하고 분할별 p0 조각 다섯 개를 쓴다(unit.json 의 shared_n0 = True).
  n_sel_slow  ftt, realmlp 의 선택.     n_p0_slow  ftt, realmlp 의 통과 0.
  n_p1        통과 1(n > 0). 단위 = (학습기, 대상, 모드, 분할). 기본판은 h40.run_ctx, 조정판은 같은 (n, 추출, seed) 칸의 같은 학습 행으로 적합한다.
              순서는 학습기 mlp, tabm, ftt, realmlp, 그 안에서 분할 번호 순이다.
  n_t2        Alaska x 의 선택, 통과 0, 통과 1(축소 S1 이면 건너뛴다).
  n_cb        CPU 작업(--cpu-only, CUDA_VISIBLE_DEVICES='', 스레드 2): catboost_lo(모든 n 칸의 D0·R1)와 catboost_tuned_loc(창 파일의
              cbt_enabled 가 참일 때만. h42.TUNE_GRID 와 같은 원천 fold 의 선택, h42.pick_tuned, 분할별 적합).
  2단계 단위는 해당 1단계가 끝난 뒤, 통과 0·1 단위는 그 학습기·대상의 D·R 선택이 끝난 뒤에 배정한다. 의존성이 충족된 단위만 우선순위
  (작업 순서 → 학습기 → 대상 → 종류 → 단계 → 시도 → seed 또는 분할) 순으로 GPU 에 배정한다.

선택 규칙(계획서 4.3)
  fold 학습 쪽에서 E_tr = h40.ls_E(y, s)(D 이면 0)를 다시 구하고 목표는 D 가 y, R 이 y − E_tr·s 다. 시도 0 은 h40.Fitter.fit(4.1 의 코드 경로),
  그 밖의 시도는 fit_tuned 다. 주 점수는 λ = 1.0 예측의 fold RMSE 의 지역 등가중 평균이다. 최종 비교는 seed 1·2 평균만 쓰고(S_f), c* ≠ 0 이면
  d_f = S_f(c*) − S_f(0), d̄ = 평균, se = sd(ddof 1)/√F 로 d̄ < 0 이고 (−d̄ > se 이거나 d_f < 0 인 fold 수 ≥ ⌈0.75F⌉)일 때만 c* 를 고른다.
  아니면 시도 0(sel_default_pref). F < 2 이면 시도 0(no_cv), 후보가 모두 실패하면 시도 0(all_failed)이다.

조작적 정의(명세와 계획서에 수치가 없는 부분을 이 파일에서 정한 것. 계획서 개정 이력에 옮겨 적을 항목이다)
  1. 무작위 설정의 '{0, LogU[a, b]}' 형식 가중 감쇠와 TabM 의 '{0, U[0, 0.5]}' dropout 은 두 갈래를 각각 확률 0.5 로 뽑는다. MLP dropout 은
     명세대로 0 의 확률 0.3 이다. '16 단위 로그 균등' 폭은 LogU[64, 1024] 값을 16 의 배수로 반올림하고 [64, 1024] 로 자른다. 모양 'half'
     의 i번째 층 폭은 max(16, 폭 >> i) 다. TabM 폭 UniformInt[64, 1024](16 단위)는 64 + 16·UniformInt[0, 60] 이다. FT-T 의 FFN 폭은
     round(비 × d_token) 이다. 뽑는 순서는 입력 변환, 손실, 그다음 계획서 4.2 표의 열 순서다.
  2. RealMLP 의 무작위 설정은 RealMLPParamSampler('default').sample_params(seed) 가 돌려준 dict 가운데 'default' 공간이 뽑는 8개 키
     (num_emb_type, add_front_scale, lr, p_drop, wd, plr_sigma, act, hidden_sizes)만 생성자 인자로 넘긴다. 나머지 키는 TD 기본값과 같다.
  3. 민감도 선택(셀 가중)의 기본값 우선 규칙: 차는 cw(c*) − cw(0) 이고, se 와 fold 승수는 주 점수의 fold 값 S_f 로 구한다. cw(c) 는 seed 1·2
     각각의 √(ΣSSE/Σn) 의 평균이다. λ 0.25 선택은 fold 값을 λ 0.25 의 fold RMSE 로 바꿔 주 규칙과 같게 고른다.
  4. c* 가 유한하고 시도 0 의 점수가 무한대(적합 실패)이면 c* 를 고르고 default_failed 표지를 남긴다.
  5. 선택 JSON 은 fold 적합이 실패해도(점수 무한대) 결과로 본다. --resume 은 unit_sig 가 같은 JSON 을 건너뛰되, --rerun-partial 이면 status 가
     ok 가 아닌 JSON(fold 실패 포함)을 다시 돈다. 실패 fold 의 오류 문자열에는 CUDA 메모리 부족이면 '[OOM]' 표지를 붙인다.
  6. fold 학습 행 가운데 제외 지역 100 km 안의 행 수는 sklearn BallTree(haversine, 지구 반지름 6,371 km)로 센다(h40.haversine_km 과 같은 식).
  7. RealMLP 의 매개변수 수는 적합한 모델 안의 torch 모듈에서 세고, 찾지 못하면 은닉층 크기에서 근사식으로 구한다(동률 판정에만 쓴다).
  8. 통과 0 과 통과 1 의 조정판에서 선택 결과가 시도 0 이면 기본판의 같은 적합 예측을 그대로 쓴다(다시 적합하지 않음, sel_default = True).
     민감도 판의 선택이 주 선택과 같으면 조정판의 예측을, 시도 0 이면 기본판의 예측을 쓴다. 같은 시도는 한 번만 적합한다.
  9. p0 조각에는 P0 와 n = 0 의 P1(= E0·s)을 함께 저장한다(h40.run_ctx 의 n = 0 칸과 같은 키).
  10. 종료 코드의 우선순위는 중단 130 > 실패 1 > 드레인 3 > 창 마감 4 > partial 2 > 0 이다. 멈춤(드레인, 창 마감, 중단, 장치 불일치) 없이 끝났는데
      남은 대기 단위(GPU 를 모두 잃어 배정하지 못한 단위, 선택 실패나 계획 오류로 의존성이 막힌 단위)는 실패로 센다. 집계 거부도 종료 코드 1 이다.
  11. 조각 설정(cfg)에 선택 결과를 넣는다(계획서 4.3 '선택 표의 해시를 조각 설정에 넣는다'): 통과 0 은 종류별 main·cw·l25 의 [시도, 표지],
      통과 1 은 main, catboost_tuned_loc 는 CatBoost 선택 설정(sel_dec, cb_cfg). 선택이 바뀌면 설정 해시가 달라져 --resume 이 그 조각을 다시 돈다.
      조각 사이 설정 대조(cfg_common)에서는 이 대상별 항목을 뺀다. 선택 묶음이 결정되면 이미 완료로 본 통과 0·1 단위를 다시 확인한다.
  12. 집계는 조각(unit.json 의 decisions)에 기록된 선택으로 k_default, '시도 0 이 아닌 지역' 보조 층화 평균, N4 를 계산한다. 조각 사이에 선택이
      다르거나, 조각에 기록이 없거나, 선택 표(현재 sel_sig)와 다르면 멈춘다(--allow-mixed-cfg 이면 경고하고 조각의 기록을 쓴다).
  13. 집계(--summarize-only)와 CPU 작업(--cpu-only)은 창 파일의 E1 과 유효 축소(reduce − restored 의 N 단계)로 k, 통과 1 의 추출·분할, 2순위
      대상을 다시 정한다. 명령행에 --e1 이나 --reduce 를 직접 주었는데 창 파일과 다르면 거부한다. 창 파일에 결정이 없으면 명령행 값을 쓴다.
      GPU 본 실행은 --e1, --reduce 가 창 파일과 같아야 한다(명세 6).
  14. N3·N2s 의 주 비교 대상은 창 파일의 cbt_enabled 로 정한다(참이면 catboost_tuned_loc, 거짓이거나 결정이 없으면 catboost_lo). 조각이 모자라면
      분할 완결성·판정 불가 규칙으로 처리한다(자료 완결 상태로 비교 대상을 바꾸지 않는다).
  15. 정밀도: 반폭 = max(셀 가중 CI 폭, 블록 등가중 CI 폭)/2 는 네 끝값이 모두 유한할 때만 계산한다. 판정 문구의 '(정밀도 미달 대비 k/m)'
      표기는 k ≥ 1 일 때만 붙인다(h47 과 같은 정의). 확인적 표지(confirmatory)는 LGF-N1 의 conf 층화 평균 행과 주 판정 행에만 둔다.
  16. N1 판정 문구: 강한 지지 학습기만으로 등록 문장을 쓰고 약한 지지 학습기는 '조정한 <학습기> 에서 우세가 관찰되지 않았다(CI [a, b] cm)'로
      따로 적는다. 판정 행과 N2s 판정 행(학습기별, 전체)에 종류별 'Δ = 0(선택 결과) 지역 k/4' 와 '조정판 = 기본판(선택 결과)'를 적는다.
      조정판이 들어간 모든 층화 평균 행에 k_default 열을 둔다. 조정판 대비 행은 맹검이고 비맹검 표시는 판정 행과 prior_info 에 둔다.

명세와 다르게 구현한 점
  1. 시험 파일 이름은 tests/test_h48_nn.py 다(과제 지시). 명세는 tests/test_h48_nn_tuning.py 였다.
  2. GPU 워커는 GPU 하나마다 ProcessPoolExecutor(max_workers = 1, spawn)를 따로 둔다. 한 GPU 의 워커가 점유를 확인하거나 비정상 종료해도
     나머지 GPU 의 풀은 계속 돈다. 워커가 하나이므로 원인 단위는 그 풀의 실행 중 단위이고(h43.culprit_pids 와 같은 결론),
     h43.POOL_CRASH_MAX(2)회 깬 단위를 격리하는 규칙은 h43.triage_broken 과 같다. 풀 재생성 전에는 rescreen_n(예약 = GPU 8 과 허용 표지 없는
     GPU 0·1, LGT 잠금 확인. h43.rescreen_gpus 의 기본 예약 0–4 를 쓰지 않는다)으로 그 GPU 를 다시 본다. 제출 시점의 풀 파손도 같은 경로로 다룬다.
  3. 점유 확인에 걸린 워커는 점유 표지(run_<tag>/gpu_busy__<번호>.json)를 남기고 예외를 올린다. 주 프로세스는 그 단위를 되돌리고 그 GPU 의 풀을
     닫는다(h43.gpu_guard_t 는 워커를 os._exit 로 끝낸다).
  4. 스모크의 선택은 점수를 계산하지 않는다(산출 제한). 2단계 후보는 시도 번호 순 상위 1개이고 조정판은 시도 1 로 고정한다(표지 smoke_fixed).
  5. 사전 점검 (b)의 시도 0 epoch 수는 tab_models._epochs_fit 과 같은 코드에 epoch 계수만 더한 대체 함수를 그 워커 안에서만 잠시 써서 잰다.
     본 실행의 시도 0 적합은 원래 함수를 쓰고 epoch 수를 −1 로 적는다.

실행 제어
  학습 실행(스모크, 사전 점검, 본 실행)은 --allow-local 이 필수다. --gpus 는 ALLOWED_GPUS = (2, 3, 4, 5, 6, 7, 9) 안의 번호만 받고, 0·1 은
  --allow-gpus-01 이 있을 때만, 8 은 항상 거부한다. 실행 직전에 nvidia-smi 로 메모리 ≤ --gpu-mem-max-mib(50)이고 계산 프로세스가 없는 GPU 만
  남기고, 6·7·9 는 LGT 잠금(data/processed/lgt/run_lgt/lock.json)의 PID 가 살아 있으면 뺀다. 워커는 단위마다 다시 확인한다. 시작 전 1분
  load average 가 --load-max-start(64)를 넘으면 거부하고, 실행 중 300 s 마다 보아 --load-max-run(96)을 넘으면 새 단위 배정을 보류한다.
  드레인: GPU 작업은 <out>/run_lgfn/drain, CPU 작업은 <out>/run_lgfn_cpu/drain 이 생기면 배정을 멈추고 진행 중 단위를 끝낸 뒤 drain 을 지우고
  종료 코드 3. 창: lgf_window.json 의 deadline 이 지나면 배정을 멈추고 종료 코드 4. 본 실행은 첫 단위를 배정하기 직전에 창 파일에 t0 가 없으면
  t0 와 deadline(t0 + 48 h)을 쓴다. GPU 본 실행은 창 파일의 E1, reduce(restored 를 뺀 N 단계 S1, S5, S6, S7)가 --e1, --reduce 와 같을 때만 돈다.
  CPU 작업과 집계는 창 파일의 값을 받아 쓴다(조작적 정의 13). 실행 상태는 GPU 작업이 lgfn_run_status.json, CPU 작업이 lgfn_cpu_run_status.json 이다.
  실행 잠금(run_<tag>/lock.json)은 계획 단계(선택 표 갱신)보다 먼저 잡는다.
  종료 코드: 중단 130, 실패 1, 드레인 3, 창 마감 4, partial 만 있으면 2, 그 밖 0.

산출 제한(스모크 tag lgfn_smoke, 사전 점검 tag lgfn_pre)
  BlockStore 를 저장하지 않고 runs 의 rmse_cm, rmse_beq_cm, bias_cm 과 선택 JSON 의 RMSE·점수를 NaN 으로 쓴다. 곡선, 판정 표를 쓰지 않는다.
  화면에는 경로 점검 값, 시간, epoch 수, GPU 메모리, 최대 RSS, 적합 수만 낸다. 집계는 이 tag 를 읽지 않는다.

명령(ROOT, PY=.venv_lgf/bin/python, G = 실행 직전 nvidia-smi 로 고른 빈 후보)
  적합 수:   $PY scripts/3_deep_learning/h48_nn_tuning.py --count-only --threads 1
  스모크:    nice -n 10 $PY scripts/3_deep_learning/h48_nn_tuning.py --smoke --allow-local --gpus G --threads 2
  사전 점검: nice -n 10 $PY scripts/3_deep_learning/h48_nn_tuning.py --precheck --allow-local --gpus G --threads 2
  본 실행:   nice -n 10 $PY scripts/3_deep_learning/h48_nn_tuning.py --jobs n_sel_fast,n_p0_fast,n_sel_slow,n_p0_slow --allow-local --gpus G
             --threads 2 --e1 <창 파일 E1> --reduce <창 파일의 유효 N 단계> --resume --no-summarize   (이어서 --jobs n_p1, --jobs n_t2)
             nice -n 10 $PY scripts/3_deep_learning/h48_nn_tuning.py --jobs n_cb --cpu-only --allow-local --threads 2 --resume --no-summarize
             (CPU 작업은 E1·축소를 창 파일에서 읽는다. 드레인은 touch data/processed/lgf/run_lgfn_cpu/drain)
  집계:      $PY scripts/3_deep_learning/h48_nn_tuning.py --summarize-only --allow-local --threads 2 [--cross-lg]
             (E1·축소는 창 파일에서 읽는다. --e1·--reduce 를 주면 창 파일과 같아야 한다. --allow-local 이 없으면 스레드 1, 재표집 1,000 이하)

확인 범위
  구현 단계: py_compile, pyflakes, 스레드 1개의 --count-only. 수정 단계(검증 지적 반영): 단위 시험(tests/test_h48_nn.py, 스레드 2, GPU 없음)과
  GPU 스모크 1회(빈 후보 GPU 한 장). 사전 점검과 본 실행은 하지 않았다.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import multiprocessing
import os
import platform
import resource
import signal
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

THREADS_DEFAULT = 2
THREADS_MAX = 2                                     # 학습 실행, --count-only, --summarize-only 모두 1–2
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_threads(default=THREADS_DEFAULT, argv=None):
    """numpy 를 부르기 전에 --threads 를 읽어 1–2 로 자른다(h43._peek_threads 와 같은 방식). --allow-local 없는 집계는 1 이다(명세 10)."""
    av = sys.argv if argv is None else list(argv)
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
    if "--summarize-only" in av and "--allow-local" not in av:
        n = 1
    return str(n)


_THREADS = _peek_threads()
for _v in THREAD_VARS:
    os.environ[_v] = _THREADS                       # setdefault 가 아니라 대입이다
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def _raise_nice(target=10):
    """niceness 를 target 이상으로 올린다. 반환 현재 niceness(실패하면 −1)."""
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
        return int(os.nice(0))
    except OSError:
        return -1


NICE = _raise_nice()

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(name, filename):
    """파일 경로에서 모듈을 읽어 sys.modules 에 등록한다(h43._load_module 과 같은 방식). 이미 있으면 그 모듈을 쓴다."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_module("h40_label_grid", "h40_label_grid.py")
T43 = _load_module("h43_tabpfn_label_grid", "h43_tabpfn_label_grid.py")
for _v in THREAD_VARS:
    os.environ[_v] = _THREADS                       # h43 은 적재 때 자기 규칙(1–4)으로 대입한다. 이 스크립트의 값으로 되돌린다
_X = None


def _load_h42():
    """h42 를 한 번만 읽는다(CPU 작업과 집계). h42 는 스레드 환경 변수를 setdefault 로만 건드린다."""
    global _X
    if _X is None:
        _X = _load_module("h42_label_grid_ext", "h42_label_grid_ext.py")
    return _X


import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, seed_of                             # noqa: E402

# ================================================================ 고정 설계값(계획서 4절, 8.4. 바꾸면 사전 등록에서 벗어난다)
LEARNERS_ALL = ("mlp", "tabm", "ftt", "realmlp")
FAST, SLOW = ("mlp", "tabm"), ("ftt", "realmlp")
KINDS = ("D", "R")
ARMS = ("default", "tuned", "tuned_cw", "tuned_l25")
JOBS_GPU = ("n_sel_fast", "n_p0_fast", "n_sel_slow", "n_p0_slow", "n_p1", "n_t2")
JOBS_ALL = JOBS_GPU + ("n_cb",)
JOB_RANK = {j: i for i, j in enumerate(JOBS_ALL)}
ALL_STEPS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
N_STEPS = ("S1", "S5", "S6", "S7")                  # N 에 해당하는 축소 단계(S2–S4 는 F 의 단계라 h48 은 무시한다)
K_LIST_MAX = 32                                     # lgfn_configs.json 에 쓰는 무작위 시도 수(E1 목록 포함)
K_MIN = 16                                          # 무작위 시도 수의 하한(계획서 8.4)
N_RANDOM_DEFAULT = "mlp=32,tabm=32,ftt=16,realmlp=16"
STAGE2_TOP_DEFAULT = "mlp=3,tabm=3,ftt=2,realmlp=2"
FTT_PAPER = dict(transform="qnorm", loss="mse", blocks=3, d_token=192, heads=8, ffn=256, dropout=0.1, lr=1e-4, wd=1e-5, batch=512)
RMLP_KEYS = ("num_emb_type", "add_front_scale", "lr", "p_drop", "wd", "plr_sigma", "act", "hidden_sizes")
RMLP_DEFAULT = dict(num_emb_type="pbld", add_front_scale=True, hidden_sizes=[256, 256, 256])
ALLOWED_GPUS = (2, 3, 4, 5, 6, 7, 8, 9)   # LGF 개정 7(2026-10-01): GPU 8 추가
GPU01 = (0, 1)
NEVER = ()   # LGF 개정 7(2026-10-01): GPU 8 을 거부 목록에서 뺐다(허용 목록과 함께)
LGT_GPUS = (6, 7, 9)
H43_SHA = "13148bb899e92c49ba24f2f9cb6463e56da26a2a"
H40_SHA = "ffd0b3a76d36472cef15ac625eeeee344f948a8f"
H42_SHA = "70ec084a1919d35a1526728957e3d20e206bfac6"
PKG_REQ = (("torch", "2.6.0", True), ("numpy", "1.26.4", False), ("pandas", "2.1.4", False), ("scikit-learn", "1.3.0", False),
           ("catboost", "1.2.10", False), ("pytabkit", "1.7.3", False))
PLATFORM = "local-3090-torch2.6.0-numpy1.26.4"
FAIL_STREAK_MAX = int(T43.FAIL_STREAK_MAX)          # 같은 학습기의 연속 실패 상한(5)
FAIL_RATIO_MAX = float(T43.FAIL_RATIO_MAX)          # 학습기별 실패 비율 상한(0.2)
POOL_CRASH_MAX = int(T43.POOL_CRASH_MAX)            # 풀을 이 횟수만큼 깬 단위는 격리한다(2)
EXIT_FAIL, EXIT_PARTIAL, EXIT_DRAIN, EXIT_WINDOW, EXIT_INTERRUPT = 1, 2, 3, 4, 130
TAG_BUSY, TAG_ABORT = "[GPU 점유]", "[적합 중단]"
PRED_CHUNK = 8192                                   # 조정판 예측 묶음의 행 수
LOAD_CHECK_S = 300.0
WINDOW_H = 48.0
KM_BUFFER = 100.0
EARTH_R_KM = 6371.0
CB_ITERS = 200                                      # catboost_lo 와 catboost_ctx 의 반복 수(h40.cb_fit 과 같다)
TIME_DEF = dict(mlp=0.9, tabm=1.0, ftt=28.0, realmlp=38.0)             # 기본판 적합 1건(s, 계획서 8.3. --count-only 추정 전용)
TIME_OPT = dict(mlp=5.0, tabm=8.0, ftt=50.0, realmlp=45.0)             # 조정판 적합 1건 낙관(s)
TIME_PES = dict(mlp=10.0, tabm=20.0, ftt=150.0, realmlp=90.0)          # 조정판 적합 1건 비관(s)
TIME_CB = 0.5                                                           # catboost_lo 1건(s, 2스레드 추정)
ROWS_REF = 15000.0                                                      # 위 1건 시간의 기준 학습 행 수(계획서 8.3)
CONFIRMATORY_N = ("LGF-N1",)
RUN_COLS = ["target", "mode", "parent", "split", "part", "axis", "method", "learner", "alpha", "placement", "n", "n_lab", "draw", "seed", "lam",
            "rmse_cm", "rmse_beq_cm", "bias_cm", "E_used", "alpha_sel", "n_blocks_lab", "n_nonfinite", "fit_flag",
            "arm", "trial", "sel_default", "sel_flag", "n_params", "epochs_run", "pass", "fit_s", "shared_n0"]
EXTRA_DEFAULTS = {"arm": "", "trial": -1, "sel_default": False, "sel_flag": "", "n_params": -1, "epochs_run": -1, "pass": "", "fit_s": np.nan,
                  "shared_n0": False}
TRACE_N = None                                      # 시험용: 리스트를 넣으면 조정판 적합의 학습 행렬과 정보를 기록한다


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H48 신경망 초모수 강건성(LGF-N)")
    ap.add_argument("--jobs", default=None, help=f"쉼표 목록. GPU: {','.join(JOBS_GPU)}(기본 전부). CPU: n_cb(--cpu-only)")
    ap.add_argument("--learners", default=",".join(LEARNERS_ALL))
    ap.add_argument("--reduce", default="", help="적용한 축소 단계(S1, S5, S6, S7 이 N 에 해당). 본 실행은 창 파일과 같아야 한다")
    ap.add_argument("--e1", type=int, choices=(0, 1), default=0, help="E1(FT-T·RealMLP 무작위 32). 본 실행은 창 파일과 같아야 한다")
    ap.add_argument("--targets", default="", help="시험용. 쉼표 목록 '이름:모드'. 기본은 주 4지역 x 와 Alaska x")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--n-grid", default="0,10,40,160,all")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lams", default="0.25,0.5,1.0")
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=100, help="기본판(h40 경로)의 epochs")
    ap.add_argument("--realmlp-epochs", type=int, default=256)
    ap.add_argument("--n-random", default=N_RANDOM_DEFAULT, help="학습기별 무작위 시도 수(E1 이면 ftt·realmlp 32, S7 이면 mlp·tabm 16, 하한 16)")
    ap.add_argument("--stage2-top", default=STAGE2_TOP_DEFAULT)
    ap.add_argument("--stage2-seeds", default="1,2")
    ap.add_argument("--tuned-max-epochs", type=int, default=300)
    ap.add_argument("--ftt-max-epochs", type=int, default=200)
    ap.add_argument("--tuned-patience", type=int, default=16)
    ap.add_argument("--min-region-cells", type=int, default=30)
    ap.add_argument("--cbt-cpu-max-h", type=float, default=8.0)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--allow-gpus-01", action="store_true")
    ap.add_argument("--gpu-mem-max-mib", type=int, default=50)
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--load-max-start", type=float, default=64.0)
    ap.add_argument("--load-max-run", type=float, default=96.0)
    ap.add_argument("--out-dir", default="data/processed/lgf")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--tag", default="lgfn")
    ap.add_argument("--lgt-dir", default="data/processed/lgt", help="LGT 잠금 파일(run_lgt/lock.json)을 읽는다. 읽기 전용")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--delta-eq", type=float, default=0.5)
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--cross-lg", action="store_true", help="집계: LG gpu 조각과 같은 키의 교차 환경 표(보조)")
    ap.add_argument("--lg-dir", default="data/processed/lg")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--h40-sha", default="")
    ap.add_argument("--h43-sha", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rerun-partial", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--pool-retries", type=int, default=2)
    ap.add_argument("--force-interim", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(path, base=ROOT):
    return Path(path) if os.path.isabs(str(path)) else Path(base) / str(path)


def _kv(txt, what):
    out = {}
    for tok in [v.strip() for v in str(txt).split(",") if v.strip()]:
        if "=" not in tok:
            raise SystemExit(f"--{what} 는 '학습기=정수' 의 쉼표 목록이다: '{txt}'")
        k, v = [q.strip() for q in tok.split("=", 1)]
        if k not in LEARNERS_ALL:
            raise SystemExit(f"--{what}: 알 수 없는 학습기 {k}")
        out[k] = int(v)
    return out


def _grid_txt(grid):
    return ",".join("all" if int(n) == -1 else str(int(n)) for n in grid)


def finalize(a):
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    if (a.count_only or a.summarize_only) and (a.smoke or a.precheck):
        raise SystemExit("--count-only, --summarize-only 는 --smoke, --precheck 와 함께 쓰지 않는다")
    a.SUFFIX = "_smoke" if a.smoke else ("_pre" if a.precheck else "")
    a.TAG = a.tag + a.SUFFIX
    a.NO_STORE = bool(a.smoke or a.precheck)        # 산출 제한(BlockStore 미저장, RMSE·점수 NaN)
    a.MAIN = not (a.smoke or a.precheck or a.count_only or a.summarize_only)
    a.threads_asked = int(getattr(a, "threads_asked", a.threads))
    a.threads = max(1, min(int(a.threads), THREADS_MAX))
    a.LOCAL_SUMMARY = bool(a.summarize_only and not a.allow_local)
    if a.LOCAL_SUMMARY:                             # 허용 표지 없는 집계: 스레드 1, 재표집 1,000 이하, 판정에 쓰지 않음(명세 10)
        a.threads = 1
        a.nboot = min(int(a.nboot), 1000)
    if a.cpu_only:
        jobs = [v.strip() for v in (a.jobs or "n_cb").split(",") if v.strip()]
        if set(jobs) != {"n_cb"}:
            raise SystemExit("[거부] --cpu-only 는 --jobs n_cb 만 돈다")
    else:
        jobs = [v.strip() for v in (a.jobs or ",".join(JOBS_GPU)).split(",") if v.strip()]
        if "n_cb" in jobs:
            raise SystemExit("[거부] n_cb 는 --cpu-only 로 돈다")
        bad = [j for j in jobs if j not in JOBS_GPU]
        if bad:
            raise SystemExit(f"알 수 없는 작업: {bad} (가능: {list(JOBS_ALL)})")
    a.JOBS = [j for j in JOBS_ALL if j in jobs]
    lr_ = [v.strip() for v in str(a.learners).split(",") if v.strip()]
    bad = [v for v in lr_ if v not in LEARNERS_ALL]
    if bad:
        raise SystemExit(f"알 수 없는 학습기: {bad}")
    a.LEARNERS = [v for v in LEARNERS_ALL if v in lr_]
    red = sorted({v.strip().upper() for v in str(a.reduce).split(",") if v.strip()})
    bad = [v for v in red if v not in ALL_STEPS]
    if bad:
        raise SystemExit(f"알 수 없는 축소 단계: {bad} (가능: {list(ALL_STEPS)})")
    a.REDUCE = red
    a.REDUCE_N = [v for v in red if v in N_STEPS]
    a.E1 = bool(int(a.e1))
    nr = _kv(N_RANDOM_DEFAULT, "n-random")
    nr.update(_kv(a.n_random, "n-random"))
    if a.E1:
        for lr in SLOW:
            nr[lr] = max(nr[lr], 32)
    if "S7" in a.REDUCE:
        for lr in FAST:
            nr[lr] = 16
    top = _kv(STAGE2_TOP_DEFAULT, "stage2-top")
    top.update(_kv(a.stage2_top, "stage2-top"))
    a.N_RANDOM, a.STAGE2_TOP = nr, top
    a.STAGE2_SEEDS = [int(v) for v in str(a.stage2_seeds).split(",") if v.strip()]
    a.SPLITS = list(range(1, int(a.splits) + 1))
    a.N_GRID = H._n_list(a.n_grid)
    a.DRAWS = int(a.draws)
    a.SEEDS = list(range(int(a.seeds)))
    a.LAMS = [float(v) for v in str(a.lams).split(",") if v.strip()]
    if a.smoke:                                     # 스모크: Canada x, 분할 1, n {0, 10}, 추출 1, seed 1, epoch 3, 시도 0·1, 2단계 상위 1개 seed 1
        a.SPLITS = [1]; a.N_GRID = [0, 10]; a.DRAWS = 1; a.SEEDS = [0]
        a.epochs = min(int(a.epochs), 3); a.realmlp_epochs = min(int(a.realmlp_epochs), 3)
        a.tuned_max_epochs = min(int(a.tuned_max_epochs), 3); a.ftt_max_epochs = min(int(a.ftt_max_epochs), 3)
        a.N_RANDOM = dict(mlp=1, tabm=1, ftt=0, realmlp=1)
        a.STAGE2_TOP = {lr: 1 for lr in LEARNERS_ALL}
        a.STAGE2_SEEDS = [1]
        a.targets = a.targets or "Canada:x"
        a.nboot = min(int(a.nboot), 1000)
    elif not a.precheck:
        bad = {lr: v for lr, v in a.N_RANDOM.items() if not (K_MIN <= int(v) <= K_LIST_MAX)}
        if bad:
            raise SystemExit(f"[거부] 무작위 시도 수는 {K_MIN}–{K_LIST_MAX} 이다(계획서 8.4 의 하한): {bad}")
    a.MAX_EPOCHS = dict(mlp=int(a.tuned_max_epochs), tabm=int(a.tuned_max_epochs), ftt=int(a.ftt_max_epochs), realmlp=int(a.realmlp_epochs))
    a.DRAWS_P1 = 1 if "S5" in a.REDUCE else a.DRAWS
    a.P1_SPLITS = [s for s in a.SPLITS if s <= 3] if "S6" in a.REDUCE else list(a.SPLITS)
    given = H._pairs(a.targets, ["i", "x"]) if a.targets else None
    if given is None:
        a.MAIN_T = [(t, "x") for t in H.MAIN4]
        a.T2 = [(H.ALASKA, "x")]
    else:
        a.MAIN_T = [p for p in given if p[0] != H.ALASKA]
        a.T2 = [p for p in given if p[0] == H.ALASKA]
    if "S1" in a.REDUCE:
        a.T2 = []
    try:
        a.GPUS = list(dict.fromkeys(int(g) for g in str(a.gpus).split(",") if g.strip() != ""))
    except ValueError:
        raise SystemExit(f"--gpus 는 정수의 쉼표 목록이다: '{a.gpus}'")
    a.OUT = _abs(a.out_dir); a.PROC = _abs(a.data_dir); a.LGDIR = _abs(a.lg_dir); a.LGTDIR = _abs(a.lgt_dir)
    a.SHARDS = a.OUT / "shards"
    a.NSEL = a.OUT / "nsel"
    a.RUN_DIR = a.OUT / f"run_{a.TAG}{'_cpu' if a.cpu_only else ''}"
    a.WINDOW = a.OUT / "lgf_window.json"
    a.CONFIGS = a.OUT / f"{a.tag}_configs.json"
    a.SELECT = a.OUT / f"{a.TAG}_nn_select.csv"
    a.SENS_T = a.OUT / f"{a.TAG}_select_sens.csv"
    a.STAB_T = a.OUT / f"{a.TAG}_select_stability.csv"
    a.CBSEL = a.OUT / f"{a.TAG}_cb_select.csv"
    a.LGT_LOCK = a.LGTDIR / "run_lgt" / "lock.json"
    a.n_grid_txt = _grid_txt(a.N_GRID)
    a.RUN_ID = ""
    a.REVISION_NEEDED = []
    a.CODE_SHA = {}
    a.CFG_SIG = ""
    a._ha = {}
    return a


# ================================================================ 환경 확인
def pkg_version(name):
    return T43.pkg_version(name)


def check_env():
    """패키지 판 확인(계획서 6.2). 하나라도 다르면 거부한다."""
    bad = []
    for name, want, prefix in PKG_REQ:
        v = pkg_version(name)
        ok = v.startswith(want) if prefix else v == want
        if not ok:
            bad.append(f"{name} {v}(필요 {want})")
    if bad:
        raise SystemExit("[거부] 패키지 판이 다르다: " + "; ".join(bad) + ". 실행 파이썬은 .venv_lgf/bin/python 이다")


def check_sha(a):
    """h43, h40 판 고정. --h43-sha, --h40-sha 로 준 값은 받아들이되 '개정 필요' 표지를 남긴다. h42 는 기록만 한다."""
    got43 = T43.file_sha(SCRIPT_DIR / "h43_tabpfn_label_grid.py")
    got40 = T43.file_sha(SCRIPT_DIR / "h40_label_grid.py")
    got42 = T43.file_sha(SCRIPT_DIR / "h42_label_grid_ext.py")
    need43, need40 = (a.h43_sha or H43_SHA), (a.h40_sha or H40_SHA)
    if got43 != need43:
        raise SystemExit(f"[거부] h43 의 SHA-1 {got43} 이 고정값 {need43} 과 다르다. LGF 개정과 --h43-sha 가 필요하다")
    if got40 != need40:
        raise SystemExit(f"[거부] h40 의 SHA-1 {got40} 이 고정값 {need40} 과 다르다. LGF 개정과 --h40-sha 가 필요하다")
    rev = []
    if a.h43_sha and a.h43_sha != H43_SHA:
        rev.append(f"개정 필요: h43 SHA-1 {a.h43_sha}")
    if a.h40_sha and a.h40_sha != H40_SHA:
        rev.append(f"개정 필요: h40 SHA-1 {a.h40_sha}")
    if got42 != H42_SHA:
        print(f"  [warn] h42 의 SHA-1 {got42} 이 기록값 {H42_SHA} 과 다르다(기록만 한다)", flush=True)
    a.REVISION_NEEDED = rev
    a.CODE_SHA = dict(h48=T43.file_sha(Path(__file__)), h43=got43, h40=got40, h42=got42)
    return a.CODE_SHA


# ================================================================ 설정 목록(계획서 4.2)
def _pyval(v):
    """numpy 자료형을 파이썬 자료형으로 바꾼다(JSON 기록용)."""
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, np.ndarray):
        return [_pyval(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_pyval(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _pyval(x) for k, x in v.items()}
    return v


def _sha(obj, n=12):
    h = hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_pyval).encode()).hexdigest()
    return h[:n] if n else h


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def cfg_seed(learner, kind, trial):
    return int(seed_of("lgfn", learner, kind, int(trial)) % (2 ** 32))


def sample_config(learner, kind, trial):
    """시도 하나의 설정. 시도 0 = 기본판(원래 코드 경로), FT-T 시도 1 = 원 논문 기본값의 근사, 그 밖은 무작위(시도마다 따로인 seed).
    뽑는 순서: 입력 변환 {std, qnorm}, 손실 {smoothl1, mse}(각 p 0.5), 그다음 계획서 4.2 표의 열 순서. RealMLP 는 pytabkit 표본기."""
    t = int(trial)
    if t == 0:
        return dict(default=True)
    if learner == "ftt" and t == 1:
        return dict(FTT_PAPER)
    s = cfg_seed(learner, kind, t)
    if learner == "realmlp":
        from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
        full = RealMLPParamSampler(is_classification=False, hpo_space_name="default").sample_params(s)
        return {k: _pyval(full[k]) for k in RMLP_KEYS}
    rng = np.random.RandomState(s)
    p = dict(transform="std" if rng.rand() < 0.5 else "qnorm", loss="smoothl1" if rng.rand() < 0.5 else "mse")
    if learner == "mlp":
        p["n_layers"] = int(rng.randint(1, 7))
        p["width"] = int(min(1024, max(64, 16 * int(round(_logu(rng, 64.0, 1024.0) / 16.0)))))
        p["shape"] = "const" if rng.rand() < 0.5 else "half"
        p["norm"] = "bn" if rng.rand() < 0.5 else "none"
        p["dropout"] = 0.0 if rng.rand() < 0.3 else float(rng.uniform(0.0, 0.5))
        p["lr"] = _logu(rng, 1e-4, 1e-2)
        p["wd"] = 0.0 if rng.rand() < 0.5 else _logu(rng, 1e-6, 1e-3)
        p["batch"] = int((256, 512, 1024, 2048)[rng.randint(4)])
    elif learner == "tabm":
        p["n_blocks"] = int(rng.randint(1, 5))
        p["width"] = int(64 + 16 * rng.randint(0, 61))
        p["k"] = int((8, 16, 32)[rng.randint(3)])
        p["dropout"] = 0.0 if rng.rand() < 0.5 else float(rng.uniform(0.0, 0.5))
        p["lr"] = _logu(rng, 1e-4, 5e-3)
        p["wd"] = 0.0 if rng.rand() < 0.5 else _logu(rng, 1e-4, 1e-1)
        p["batch"] = int((256, 512, 1024, 2048)[rng.randint(4)])
    elif learner == "ftt":
        p["blocks"] = int(rng.randint(1, 5))
        p["d_token"] = int((64, 96, 128, 160, 192, 256)[rng.randint(6)])
        p["heads"] = 8
        p["ffn"] = int(round(float(rng.uniform(2.0 / 3.0, 8.0 / 3.0)) * p["d_token"]))
        p["dropout"] = float(rng.uniform(0.0, 0.3))
        p["lr"] = _logu(rng, 3e-5, 1e-3)
        p["wd"] = _logu(rng, 1e-6, 1e-3)
        p["batch"] = int((256, 512, 1024)[rng.randint(3)])
    else:
        raise ValueError(f"알 수 없는 학습기 {learner}")
    return p


def n_trials_list(learner, k):
    """무작위 k 개일 때 쓰는 시도 번호: MLP·TabM·RealMLP 0..k, FT-T 0..k+1(시도 1 은 고정 설정)."""
    return list(range(int(k) + (2 if learner == "ftt" else 1)))


_CFG_CACHE: dict = {}


def build_configs(k_max=K_LIST_MAX):
    """네 학습기 × 종류(D, R) × 시도(k_max 까지, E1 목록 포함)의 설정 목록. 결정적이다(시도마다 seed_of('lgfn', 학습기, 종류, 시도))."""
    if k_max in _CFG_CACHE:
        return _CFG_CACHE[k_max]
    out = {}
    for lr in LEARNERS_ALL:
        out[lr] = {}
        for kind in KINDS:
            lst = []
            for t in n_trials_list(lr, k_max):
                p = sample_config(lr, kind, t)
                lst.append(dict(learner=lr, kind=kind, trial=int(t), params=p, sig=_sha(dict(learner=lr, kind=kind, trial=int(t), params=p))))
            out[lr][kind] = lst
    _CFG_CACHE[k_max] = out
    return out


def configs_sig(cfgs):
    return _sha(cfgs, n=0)


def ensure_configs(a, write=False, require=False):
    """설정 목록과 해시(cfg_sig). 파일이 있으면 해시가 같아야 한다. require = True(본 실행)이면 파일이 있어야 한다."""
    cf = build_configs(K_LIST_MAX)
    sig = configs_sig(cf)
    p = Path(a.CONFIGS)
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except (OSError, ValueError):
            old = {}
        if old.get("cfg_sig") != sig:
            raise SystemExit(f"[거부] 설정 목록 {p} 의 해시 {old.get('cfg_sig')} 가 재계산 값 {sig} 과 다르다(코드 또는 표본기 판이 바뀌었다)")
    elif require:
        raise SystemExit(f"[거부] 설정 목록 {p} 이 없다. 먼저 --count-only 를 실행해 목록과 해시를 남긴다")
    elif write:
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_json(p, dict(cfg_sig=sig, k_max=K_LIST_MAX, created=_now(), n_trials={lr: len(cf[lr]["D"]) for lr in LEARNERS_ALL},
                            seed_rule="seed_of('lgfn', learner, kind, trial) % 2**32", configs=cf), strict=True)
    a.CFG_SIG = sig
    return cf, sig


def cfg_entry(learner, kind, trial):
    lst = build_configs(K_LIST_MAX)[learner][kind]
    return lst[int(trial)]


def trials_of(a, learner):
    return n_trials_list(learner, a.N_RANDOM[learner])


def sel_sig(a, learner):
    """학습기 하나의 선택 설정 해시(선택 표의 sel_sig). 설정 목록, 실제 k, 2단계 규칙, fold 규칙, epoch 규칙이 들어간다."""
    HA = ha_n(a, learner, "0")
    d = dict(cfg_sig=a.CFG_SIG, learner=learner, k=int(a.N_RANDOM[learner]), top=int(a.STAGE2_TOP[learner]), seeds2=list(a.STAGE2_SEEDS),
             min_cells=int(a.min_region_cells), epochs=int(a.epochs), realmlp_epochs=int(a.realmlp_epochs), max_epochs=int(a.MAX_EPOCHS[learner]),
             patience=int(a.tuned_patience), buffer_km=float(HA.buffer_km), k_sub=str(HA.k_sub),
             submap=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans", rule="seed12_default_pref_v1", smoke=bool(a.smoke))
    return _sha(d)


# ================================================================ h40 인자와 자료
def ha_n(a, learner="mlp", lgrid="0", epochs=None, realmlp_epochs=None):
    """기본판 인자(HA_p). 기본판 적합은 모두 h40.Fitter(HA) 로 한다. lgrid = '0'(통과 0, 선택) 또는 양수 격자(통과 1)."""
    key = (learner, str(lgrid), epochs, realmlp_epochs)
    if key in a._ha:
        return a._ha[key]
    argv = ["--part", "gpu", "--learners", learner, "--methods", "P0,P1,D0,R1", "--alphas", "", "--n-grid", a.n_grid_txt,
            "--learner-n-grid", str(lgrid), "--learner-draws", str(int(a.DRAWS_P1)), "--draws", str(int(a.DRAWS)), "--seeds", str(len(a.SEEDS)),
            "--splits", "5", "--learner-splits", "5", "--epochs", str(int(epochs if epochs is not None else a.epochs)),
            "--realmlp-epochs", str(int(realmlp_epochs if realmlp_epochs is not None else a.realmlp_epochs)), "--threads", str(int(a.threads)),
            "--allow-local", "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", str(a.subregion_map),
            "--kappa", str(float(a.kappa)), "--lams", ",".join(str(v) for v in a.LAMS), "--cb-iters", str(CB_ITERS)]
    HA = H.parse_args(argv)
    a._ha[key] = HA
    return HA


def get_data(a):
    return H.get_data(ha_n(a, "mlp", "0"))


def lgrid_pos(a):
    pos = [n for n in a.N_GRID if n != 0]
    return _grid_txt(pos) if pos else ""


_SRC: dict = {}


def src_info(a, D, target, mode):
    """원천 행(h40.Data.source_idx 의 행 가운데 y 와 s 가 유한한 행. h40.Ctx 의 X_src 와 같은 순서와 float32)과 fold 지역."""
    key = (id(D), target, mode, int(a.min_region_cells))
    if key in _SRC:
        return _SRC[key]
    df = D.df
    _, _, src_idx, _ = D.source_idx(target, mode)
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = np.asarray(src_idx)[ok]
    X = np.asarray(df[H.FEATS].values[src], np.float32)
    mac = df.macro.values[src].astype(str)
    regs = sorted(str(r) for r, k in Counter(mac.tolist()).items() if k >= int(a.min_region_cells))
    si = SimpleNamespace(target=target, mode=mode, src=src, X=X, y=df.y.values[src].astype(float), s=df.s.values[src].astype(float), mac=mac,
                         lat=df.lat.values[src].astype(float), lon=df.lon.values[src].astype(float), regions=regs,
                         src_hash=hashlib.sha1(np.ascontiguousarray(src, np.int64).tobytes()).hexdigest()[:12], folds=None)
    _SRC[key] = si
    return si


def near_mask(lat_a, lon_a, lat_b, lon_b, km=KM_BUFFER):
    """a 의 각 행이 b 의 어느 행과 km 안에 있는지(haversine, 반지름 6,371 km)."""
    if len(lat_a) == 0 or len(lat_b) == 0:
        return np.zeros(len(lat_a), bool)
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(np.c_[lat_b, lon_b]), metric="haversine")
    cnt = tree.query_radius(np.radians(np.c_[lat_a, lon_a]), r=float(km) / EARTH_R_KM, count_only=True)
    return np.asarray(cnt) > 0


def fold_rows(si):
    """fold 별 (제외 지역, 채점 셀 수, 학습 행 수, 제외 지역 100 km 안의 학습 행 수)."""
    if si.folds is None:
        out = []
        for r in si.regions:
            te = si.mac == r
            tr = ~te
            nm = near_mask(si.lat[tr], si.lon[tr], si.lat[te], si.lon[te])
            out.append(dict(region=r, n_te=int(te.sum()), n_tr=int(tr.sum()), n_tr_within100km=int(nm.sum())))
        si.folds = out
    return si.folds


def valid_splits(a, D, target, mode, splits=None):
    """h40.enumerate_units 와 같은 규칙으로 실행하는 분할(중복·무효 분할 제외)."""
    HA = copy.copy(ha_n(a, "mlp", "0"))
    HA.TARGETS = [(target, mode)]
    us, _ = H.enumerate_units(HA, D, "cpu")
    keep = set(int(s_) for s_ in (splits if splits is not None else a.SPLITS))
    return [int(sp) for t, m, sp, _ in us if int(sp) in keep]


def src_arrays_hash(c):
    h = hashlib.sha1()
    for v in (np.ascontiguousarray(c.X_src, np.float32), np.ascontiguousarray(c.y_src, np.float64), np.ascontiguousarray(c.r0_src, np.float64)):
        h.update(v.tobytes())
    return h.hexdigest()[:12]


# ================================================================ 매개변수화 신경망과 조정판 적합(계획서 4.2)
_MODS = None


def _mods():
    """조정판 모듈 클래스(torch 를 부를 때만 만든다)."""
    global _MODS
    if _MODS is not None:
        return _MODS
    import torch
    import torch.nn as nn

    class MLPp(nn.Module):
        """층마다 Linear → ReLU → [BatchNorm] → Dropout, 마지막 Linear(·, 1). shape 'half' 는 층마다 폭을 절반(최소 16)."""

        def __init__(self, d_in, n_layers, width, shape="const", norm="bn", dropout=0.0):
            super().__init__()
            widths = [int(width)] * int(n_layers) if shape == "const" else [max(16, int(width) >> i) for i in range(int(n_layers))]
            layers, prev = [], int(d_in)
            for w in widths:
                layers += [nn.Linear(prev, w), nn.ReLU()]
                if norm == "bn":
                    layers.append(nn.BatchNorm1d(w))
                layers.append(nn.Dropout(float(dropout)))
                prev = w
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class TabMp(nn.Module):
        """다중 헤드 MLP. 몸통 Linear → ReLU → BatchNorm → Dropout × n_blocks, 헤드 k 개(width → 64 → 1)는 묶음 매개변수와 einsum 으로 한 번에
        계산하고 k 평균을 낸다. 헤드 초기화는 nn.Linear 기본값과 같은 분포 U(−1/√fan_in, 1/√fan_in)."""

        def __init__(self, d_in, n_blocks, width, k, dropout=0.0, hidden=64):
            super().__init__()
            layers, prev = [], int(d_in)
            for _ in range(int(n_blocks)):
                layers += [nn.Linear(prev, int(width)), nn.ReLU(), nn.BatchNorm1d(int(width)), nn.Dropout(float(dropout))]
                prev = int(width)
            self.trunk = nn.Sequential(*layers)
            k, hd = int(k), int(hidden)
            b1, b2 = 1.0 / math.sqrt(prev), 1.0 / math.sqrt(hd)
            self.W1 = nn.Parameter(torch.empty(k, prev, hd).uniform_(-b1, b1))
            self.b1 = nn.Parameter(torch.empty(k, hd).uniform_(-b1, b1))
            self.W2 = nn.Parameter(torch.empty(k, hd, 1).uniform_(-b2, b2))
            self.b2 = nn.Parameter(torch.empty(k, 1).uniform_(-b2, b2))

        def heads(self, h):
            z = torch.relu(torch.einsum("bd,kdh->bkh", h, self.W1) + self.b1.unsqueeze(0))
            return torch.einsum("bkh,kho->bko", z, self.W2) + self.b2.unsqueeze(0)

        def forward(self, x):
            return self.heads(self.trunk(x)).mean(1).squeeze(-1)

    class FTTp(nn.Module):
        """선형 토큰화(W, b)와 CLS, nn.TransformerEncoderLayer(norm_first, gelu, batch_first), LayerNorm + Linear 헤드."""

        def __init__(self, n_feat, d_token, blocks, ffn, dropout=0.1, heads=8):
            super().__init__()
            d = int(d_token)
            self.W = nn.Parameter(torch.randn(int(n_feat), d) * 0.02)
            self.b = nn.Parameter(torch.zeros(int(n_feat), d))
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, int(heads), int(ffn), float(dropout), activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, int(blocks), enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.W + self.b
            z = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
            return self.head(self.enc(z)[:, 0]).squeeze(-1)

    _MODS = SimpleNamespace(MLPp=MLPp, TabMp=TabMp, FTTp=FTTp)
    return _MODS


def build_net(learner, p, d_in):
    M = _mods()
    if learner == "mlp":
        return M.MLPp(d_in, p["n_layers"], p["width"], p["shape"], p["norm"], p["dropout"])
    if learner == "tabm":
        return M.TabMp(d_in, p["n_blocks"], p["width"], p["k"], p["dropout"])
    if learner == "ftt":
        return M.FTTp(d_in, p["d_token"], p["blocks"], p["ffn"], p["dropout"], p.get("heads", 8))
    raise ValueError(f"매개변수화 클래스가 없는 학습기 {learner}")


def qt_seed(learner, kind, trial, fit_id, seed):
    return int(seed_of("lgfn-qt", learner, kind, int(trial), str(fit_id), int(seed)) % (2 ** 32))


def make_transform(p, Xtr, learner, kind, trial, fit_id, seed):
    """입력 변환(학습 행 전체로 적합). std = h40._prep_stats/_prep_apply. qnorm = 학습 행 중앙값 대체 뒤 QuantileTransformer
    (output_distribution 'normal', n_quantiles = min(1000, n), subsample = 10**9, random_state 고정). 반환 (변환 함수, 기록)."""
    Xtr = np.asarray(Xtr, float)
    if p.get("transform", "std") == "std":
        stats = H._prep_stats(Xtr)
        return (lambda Z: H._prep_apply(Z, stats)), dict(transform="std")
    from sklearn.preprocessing import QuantileTransformer
    med = H._prep_stats(Xtr)[0]
    rs = qt_seed(learner, kind, trial, fit_id, seed)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=int(min(1000, len(Xtr))), subsample=10 ** 9, random_state=rs)
    qt.fit(np.where(np.isnan(Xtr), med, Xtr))

    def f(Z):
        Z = np.asarray(Z, float)
        return np.asarray(qt.transform(np.where(np.isnan(Z), med, Z)), np.float32)
    return f, dict(transform="qnorm", qt_random_state=rs, n_quantiles=int(min(1000, len(Xtr))))


def _pred_chunks(net, X):
    import torch
    if len(X) == 0:
        return torch.zeros(0, device=X.device)
    return torch.cat([net(X[k:k + PRED_CHUNK]) for k in range(0, len(X), PRED_CHUNK)])


def _split(allp, sizes):
    out, j = [], 0
    for s_ in sizes:
        out.append(np.asarray(allp[j:j + s_], float)); j += s_
    return out


def rmlp_kwargs(p):
    return {k: (list(p[k]) if k == "hidden_sizes" else p[k]) for k in RMLP_KEYS if k in p}


def rmlp_param_formula(p, d_in):
    """RealMLP 매개변수 수의 근사(동률 판정 전용). 수치 임베딩이 있으면 특성마다 16 + 16·4 + 4 개, 입력 폭은 특성 × 4 로 둔다."""
    q = dict(RMLP_DEFAULT); q.update({k: v for k, v in p.items() if k in RMLP_KEYS})
    emb = str(q.get("num_emb_type", "pbld"))
    d0 = int(d_in) * (1 if emb == "none" else 4)
    emb_par = 0 if emb == "none" else int(d_in) * (16 + 16 * 4 + 4)
    sizes = [d0] + [int(v) for v in q.get("hidden_sizes", [256, 256, 256])] + [1]
    return int(emb_par + sum(i_ * o_ + o_ for i_, o_ in zip(sizes[:-1], sizes[1:])) + (int(d_in) if q.get("add_front_scale", True) else 0))


def count_params_any(obj, depth=5):
    """적합한 객체 안에서 매개변수가 있는 torch 모듈을 찾아 매개변수 수를 센다. 못 찾으면 −1."""
    try:
        import torch.nn as nn
    except Exception:                                                     # noqa: BLE001
        return -1
    seen = set()

    def walk(o, d):
        if d < 0 or id(o) in seen or isinstance(o, (str, bytes, int, float, bool, np.ndarray)) or o is None:
            return None
        seen.add(id(o))
        if isinstance(o, nn.Module):
            n = int(sum(q.numel() for q in o.parameters()))
            if n > 0:
                return n
        if isinstance(o, dict):
            items = list(o.values())
        elif isinstance(o, (list, tuple)):
            items = list(o)
        elif hasattr(o, "__dict__"):
            items = list(vars(o).values())
        else:
            items = []
        for v in items:
            r = walk(v, d - 1)
            if r:
                return r
        return None
    try:
        r = walk(obj, depth)
    except Exception:                                                     # noqa: BLE001
        r = None
    return int(r) if r else -1


_NP0: dict = {}


def default_n_params(learner, d_in):
    """시도 0(원래 클래스)의 매개변수 수. 전역 난수 상태를 바꾸지 않는다."""
    if learner == "realmlp":
        return rmlp_param_formula({}, d_in)
    key = (learner, int(d_in))
    if key not in _NP0:
        import torch
        from polar import tab_models as TM
        with torch.random.fork_rng(devices=[]):
            MLP, FTT, TabM = TM._torch_mods()
            net = dict(mlp=MLP, ftt=FTT, tabm=TabM)[learner](int(d_in))
            _NP0[key] = int(sum(q.numel() for q in net.parameters()))
    return _NP0[key]


def fit_tuned(learner, entry, Xtr, ytr, preds, seed, fit_id, threads=2, max_epochs=300, patience=16, realmlp_epochs=256, dev=None):
    """조정판 적합 한 건. 반환 (예측 목록, 실제 epoch 수, 매개변수 수).
    MLP·TabM·FT-T: torch.manual_seed(seed), 입력 변환(학습 행 전체), y 는 학습 행 평균·SD 로 z 변환, 검증 = RandomState(seed).rand(n) < 0.1,
    학습·검증 텐서를 장치에 올리고 배치 순서는 장치 위 torch.randperm(generator = 학습기 seed). AdamW, 손실 SmoothL1 또는 MSE, BatchNorm 이
    있으면 1행 배치를 뺀다, 기울기 절단 5, epoch 마다 검증 MSE(최소 개선 1e-4, patience)와 최적 상태 복원, 예측은 8,192행 묶음.
    RealMLP: RealMLP_TD_Regressor(random_state = seed, n_threads, n_epochs, 표본 설정)에 h40 으로 표준화한 x25."""
    import torch
    import torch.nn as nn
    from polar import tab_models as TM
    dev = dev or TM._dev()
    p = dict(entry["params"])
    kind, trial = str(entry.get("kind", "D")), int(entry.get("trial", -1))
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float)
    if not np.all(np.isfinite(ytr)):
        raise ValueError("학습 목표에 비유한 값이 있다")
    sizes = [len(q) for q in preds]
    if TRACE_N is not None:
        TRACE_N.append(dict(learner=learner, trial=trial, kind=kind, fit_id=str(fit_id), seed=int(seed), Xtr=np.array(Xtr, copy=True),
                            ytr=np.array(ytr, copy=True)))
    if learner == "realmlp":
        from pytabkit import RealMLP_TD_Regressor
        stats = H._prep_stats(Xtr)
        m = RealMLP_TD_Regressor(random_state=int(seed), device=dev, n_threads=int(threads), n_epochs=int(realmlp_epochs), **rmlp_kwargs(p))
        m.fit(H._prep_apply(Xtr, stats), ytr)
        allp = np.asarray(m.predict(np.vstack([H._prep_apply(q, stats) for q in preds])), float).reshape(-1) if sum(sizes) else np.zeros(0)
        n_par = count_params_any(m)
        if n_par < 0:
            n_par = rmlp_param_formula(p, Xtr.shape[1])
        del m
        return _split(allp, sizes), int(realmlp_epochs), int(n_par)
    tf, _ = make_transform(p, Xtr, learner, kind, trial, fit_id, seed)
    Xz = np.asarray(tf(Xtr), np.float32)
    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    torch.manual_seed(int(seed))
    va = np.random.RandomState(int(seed)).rand(len(Xz)) < 0.1
    tr = ~va
    net = build_net(learner, p, Xz.shape[1]).to(dev)
    n_par = int(sum(q.numel() for q in net.parameters()))
    Xt, yt = torch.tensor(Xz[tr], device=dev), torch.tensor(yz[tr], device=dev)
    Xv, yv = torch.tensor(Xz[va], device=dev), torch.tensor(yz[va], device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(p["lr"]), weight_decay=float(p["wd"]))
    lossf = nn.MSELoss() if p.get("loss") == "mse" else nn.SmoothL1Loss()
    has_bn = any(isinstance(q, nn.BatchNorm1d) for q in net.modules())
    gen = torch.Generator(device=torch.device(dev)).manual_seed(int(seed))
    bs, n_tr = int(p["batch"]), int(tr.sum())
    best, state, wait_, ep_run = np.inf, None, 0, 0
    for _ in range(int(max_epochs)):
        _orphan_check()
        net.train()
        idx = torch.randperm(n_tr, generator=gen, device=dev)
        for k in range(0, n_tr, bs):
            b = idx[k:k + bs]
            if has_bn and len(b) < 2:                                     # 학습 모드 BatchNorm 은 1행 배치를 받지 않는다
                continue
            opt.zero_grad(set_to_none=True)
            lossf(net(Xt[b]), yt[b]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        ep_run += 1
        if len(Xv) == 0:                                                  # 검증 행이 없으면 최대 epoch 까지 돈다
            continue
        net.eval()
        with torch.no_grad():
            v = float(torch.mean((_pred_chunks(net, Xv) - yv) ** 2).item())
        if v < best - 1e-4:
            best, wait_ = v, 0
            state = {q: t_.detach().clone() for q, t_ in net.state_dict().items()}
        else:
            wait_ += 1
            if wait_ >= int(patience):
                break
    if state is not None:
        net.load_state_dict(state)
    net.eval()
    out = []
    with torch.no_grad():
        for q, s_ in zip(preds, sizes):
            if s_ == 0:
                out.append(np.zeros(0)); continue
            z = _pred_chunks(net, torch.tensor(np.asarray(tf(q), np.float32), device=dev)).cpu().numpy().astype(float)
            out.append(z * ysd + ymu)
    del net, opt, Xt, yt, Xv, yv, state
    return out, int(ep_run), int(n_par)


def tuned_kw(a, learner):
    return dict(threads=int(a.threads), max_epochs=int(a.MAX_EPOCHS[learner]), patience=int(a.tuned_patience), realmlp_epochs=int(a.realmlp_epochs))


def fit_default(F, learner, X, y, preds, seed, info, axis="learner"):
    """기본판 적합(h40.Fitter.fit, 4.1 의 코드 경로). 반환 (예측 목록, 표지)."""
    _, out = F.fit(learner, axis, lambda: (X, y, None), int(seed), preds, info=info)
    return [np.asarray(q, float) for q in out], str(F.last_flag)


def fail_text(e):
    """적합 예외의 기록 문자열. CUDA 메모리 부족이면 앞에 '[OOM]' 표지를 붙인다."""
    return ("[OOM] " if T43.is_cuda_oom(e) else "") + repr(e)[:200]


class FitAbort(RuntimeError):
    """같은 학습기의 연속 실패가 FAIL_STREAK_MAX 에 이르렀다. 단위를 중단한다(조각과 선택 JSON 을 쓰지 않는다)."""


class Streak:
    def __init__(self):
        self.c = Counter()

    def update(self, learner, failed, where=""):
        if failed:
            self.c[learner] += 1
            if self.c[learner] >= FAIL_STREAK_MAX:
                raise FitAbort(f"{TAG_ABORT} {learner} 적합이 {self.c[learner]}회 연속 실패했다({where})")
        else:
            self.c[learner] = 0


# ================================================================ 조각 기록
def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_json(path, obj, strict=False):
    """작은 기록 파일(O_EXCL 임시 파일 뒤 os.replace). strict = False 이면 쓰지 못해도 멈추지 않는다."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.{int(time.time() * 1e6) % 10 ** 9}")
        fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=1, default=_pyval))
        os.replace(tmp, path)
    except OSError:
        if strict:
            raise


def _atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def arm_name(learner, arm):
    return learner if arm == "default" else f"{learner}_{arm}"


class ShardBook:
    """조각 하나(대상, 모드, 분할)의 저장소와 runs 행. 행 형식은 h40.run_ctx 의 add 와 같고 LGF-N 열(arm, trial, sel_default, sel_flag,
    n_params, epochs_run, pass, fit_s, shared_n0)을 더한다. 비유한 예측이 있는 키는 저장하지 않는다. hide = True 이면 RMSE 열을 NaN 으로 둔다."""

    def __init__(self, c, part, axis, hide=False, st=None, rows=None):
        self.c, self.hide = c, bool(hide)
        self.st = st if st is not None else BlockStore(f"{c.target}|{c.mode}", c.split, c.blkB, meta=dict(target=c.target, mode=c.mode, part=part,
                                                                                                          axis=axis))
        self.base = dict(target=c.target, mode=c.mode, parent=c.parent, split=int(c.split), part=part, axis=axis)
        self.rows = list(rows or [])
        self.n_bad = 0
        self.bias: dict = {}
        self.fits, self.fails = Counter(), Counter()                     # 학습기 이름별 적합 수와 실패 수(예외, 비유한 예측, 복사 원본 없음)

    def _row(self, method, learner, n, d, seed, lam, E_used, n_lab, nb_lab, flag, nf, ex):
        row = dict(**self.base, method=method, learner=learner, alpha="1", placement="cell", n=int(n), n_lab=int(n_lab), draw=int(d), seed=int(seed),
                   lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan, E_used=float(E_used), alpha_sel="", n_blocks_lab=int(nb_lab),
                   n_nonfinite=int(nf), fit_flag=str(flag) if nf == 0 else (str(flag) or "nonfinite"))
        row.update(EXTRA_DEFAULTS)
        row.update(ex or {})
        return row

    def _fill(self, row, key):
        if self.hide:
            return
        sse, cnt = self.st.get(key)
        with np.errstate(invalid="ignore", divide="ignore"):
            row["rmse_cm"] = float(np.sqrt(sse.sum() / cnt.sum())) if cnt.sum() else np.nan
            row["rmse_beq_cm"] = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan))) if cnt.sum() else np.nan
        row["bias_cm"] = float(self.bias.get(tuple(key), np.nan))

    def add(self, method, learner, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, nb_lab=0, flag="", ex=None):
        pred = np.asarray(pred, float)
        nf = int((~np.isfinite(pred)).sum())
        row = self._row(method, learner, n, d, seed, lam, E_used, n_lab, nb_lab, flag, nf, ex)
        if nf > 0:
            self.n_bad += 1
            self.rows.append(row)
            return None
        key = (method, learner, "1", "cell", int(n), int(d), int(seed), float(lam))
        self.st.add(key, self.c.yB, pred)
        self.bias[key] = float(np.nanmean(pred - self.c.yB)) if len(self.c.yB) else np.nan
        self._fill(row, key)
        self.rows.append(row)
        return key

    def copy(self, src_key, method, learner, n, d, seed, lam, E_used=np.nan, n_lab=0, nb_lab=0, ex=None):
        """저장된 키의 블록 SSE 를 다른 키로 복사한다(선택 결과가 시도 0 인 조정판). 원본이 없으면 실패 행으로 남긴다."""
        src_key = tuple(src_key)
        if src_key not in self.st:
            row = self._row(method, learner, n, d, seed, lam, E_used, n_lab, nb_lab, "copy_missing", 0, ex)
            row["n_nonfinite"] = len(self.c.yB)
            self.n_bad += 1
            self.fails[learner] += 1
            self.rows.append(row)
            return None
        key = (method, learner, "1", "cell", int(n), int(d), int(seed), float(lam))
        sse, cnt = self.st.get(src_key)
        self.st.add_sse(key, np.array(sse, copy=True), np.array(cnt, copy=True))
        self.bias[key] = self.bias.get(src_key, np.nan)
        row = self._row(method, learner, n, d, seed, lam, E_used, n_lab, nb_lab, "copy", 0, ex)
        self._fill(row, key)
        self.rows.append(row)
        return key

    def status(self):
        """ok, partial, failed(계획서 명세 7: 학습기 저장 키가 0 이거나 실패 비율이 0.2 를 넘으면 failed)."""
        stored = Counter(k[1] for k in self.st.keys if k[1] != "none")
        bad = [lr for lr, nf in self.fits.items() if nf > 0 and (stored[lr] == 0 or self.fails[lr] / nf > FAIL_RATIO_MAX)]
        n_fail = int(sum(self.fails.values()))
        return ("ok" if n_fail == 0 and self.n_bad == 0 else ("failed" if bad else "partial")), bad


def shard_base(a, pass_, learner, target, mode, split):
    if pass_ in ("p0", "p1"):
        return a.SHARDS / f"{a.TAG}__gpu__{learner}__{target}__{mode}__s{int(split)}__{pass_}"
    return a.SHARDS / f"{a.TAG}__cpu__{learner}__{target}__{mode}__s{int(split)}"


def shard_paths(a, pass_, learner, target, mode, split):
    b = str(shard_base(a, pass_, learner, target, mode, split))
    return dict(runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), unit=Path(b + "_unit.json"))


CFG_LOCAL_KEYS = ("sel_dec", "cb_cfg")               # 대상마다 다른 설정 항목(선택 결과). 조각 사이 설정 대조(cfg_common)에서는 뺀다


def dec_brief(dec, crits):
    """선택 기록 {종류: {기준: dict(trial, flag, …)}}(또는 {종류: dict(trial, flag)}) → 조각 설정의 요약 {종류: {기준: [시도, 표지]}}."""
    out = {}
    for kind in KINDS:
        dk = dict((dec or {}).get(kind) or {})
        if "trial" in dk:
            dk = {"main": dk}
        out[kind] = {c_: [int(dk[c_]["trial"]), str(dk[c_].get("flag", "") or "")] for c_ in crits if dk.get(c_) is not None}
    return out


def shard_cfg(a, pass_, learner, dec=None, cb_cfg=None):
    """결과에 영향을 주는 설정 요약(대상, 분할, tag, 스레드, GPU 번호는 넣지 않는다. CPU 조각은 CatBoost 의 스레드 수를 넣는다).
    통과 0·1 은 그 (학습기, 대상, 모드)의 선택 결과(sel_dec: 종류별 시도와 표지. 통과 0 은 main·cw·l25, 통과 1 은 main)를, catboost_tuned_loc 는
    CatBoost 선택 설정(cb_cfg)을 넣는다(계획서 4.3 '선택 표의 해시를 조각 설정에 넣는다'). 선택이 바뀌면 설정 해시가 달라져 조각이 미완료가 된다."""
    HA = ha_n(a, "mlp", "0")
    d = dict(pass_=pass_, learner=learner, n_grid=list(a.N_GRID), seeds=list(a.SEEDS), lams=list(a.LAMS), kappa=float(a.kappa),
             buffer_km=float(HA.buffer_km), k_sub=str(HA.k_sub), min_cells_prior=int(HA.min_cells_prior), feats="x25",
             subregion_map=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans", no_store=bool(a.NO_STORE),
             revision_needed=list(a.REVISION_NEEDED))
    if pass_ in ("p0", "p1"):
        d.update(epochs=int(a.epochs), realmlp_epochs=int(a.realmlp_epochs), max_epochs=int(a.MAX_EPOCHS[learner]), patience=int(a.tuned_patience),
                 cfg_sig=str(a.CFG_SIG), sel_sig=sel_sig(a, learner), k=int(a.N_RANDOM[learner]))
        if pass_ == "p0":
            d.update(splits=list(a.SPLITS), arms=list(ARMS), sel_dec=dec_brief(dec, ("main", "cw", "l25")))
        else:
            d.update(draws=int(a.DRAWS_P1), arms=["default", "tuned"], sel_dec=dec_brief(dec, ("main",)))
    else:
        d.update(draws=int(a.DRAWS_P1), cb_iters=CB_ITERS, cb_threads=int(a.threads), cb_sel_sig=cb_sel_sig(a) if learner == "catboost_tuned_loc" else "")
        if learner == "catboost_tuned_loc":
            d["cb_cfg"] = {str(k): {str(q): v for q, v in sorted(dict(cv).items())} for k, cv in sorted((cb_cfg or {}).items())}
    return d


def current_dec(a, learner, target, mode):
    """선택 표(현재 sel_sig)의 종류별 선택 {종류: {기준: dict}}. D·R 가운데 하나라도 주 선택이 없으면 None."""
    tab = decisions_table(a)
    out = {}
    for kind in KINDS:
        d = tab.get((learner, kind, target, mode))
        if not d or "main" not in d:
            return None
        out[kind] = d
    return out


def shard_state(a, pass_, learner, target, mode, split, run_id=None, dec=None):
    """(완료 여부, 사유). 완료 = runs, blocksse(산출 제한이 아니면), unit 이 있고, 설정 해시가 같고, status 가 failed 가 아니다.
    통과 0·1 의 설정 해시에는 현재 선택 표의 선택 결과가 들어간다(dec 를 주지 않으면 선택 표에서 읽는다. 선택이 없으면 미완료).
    run_id 를 주면 그 실행에서 쓴 조각만 완료로 본다(풀이 깨진 뒤 확인용)."""
    p = shard_paths(a, pass_, learner, target, mode, split)
    need = [p["unit"], p["runs"]] + ([] if a.NO_STORE else [p["npz"]])
    if not all(q.exists() for q in need):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if pass_ in ("p0", "p1"):
        dec = current_dec(a, learner, target, mode) if dec is None else dec
        if dec is None:
            return False, "선택 표에 이 학습기·대상의 선택이 없다"
        cfg = shard_cfg(a, pass_, learner, dec=dec)
    elif learner == "catboost_tuned_loc":
        cb = load_cb_tuned(a, target, mode)
        if not all(k in cb for k in KINDS):
            return False, "CatBoost 선택 표에 이 대상의 선택이 없다"
        cfg = shard_cfg(a, pass_, learner, cb_cfg=cb)
    else:
        cfg = shard_cfg(a, pass_, learner)
    if u.get("cfg_hash") != H.cfg_hash(cfg):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    if getattr(a, "rerun_partial", False) and u.get("status") == "partial":
        return False, "일부 적합 실패(--rerun-partial)"
    if run_id is not None and u.get("run_id") != run_id:
        return False, "다른 실행의 조각"
    return True, str(u.get("status", "ok"))


def res_block(t0=None):
    """자원 기록: 최대 RSS(MiB), torch 의 GPU 메모리 최댓값(할당, 예약 MiB)."""
    d = dict(rss_max_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1), gpu_mem_peak_mib=None,
             gpu_mem_reserved_peak_mib=None, nice=int(os.nice(0)))
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            d.update(gpu_mem_peak_mib=round(torch.cuda.max_memory_allocated() / 2.0 ** 20, 1),
                     gpu_mem_reserved_peak_mib=round(torch.cuda.max_memory_reserved() / 2.0 ** 20, 1))
    except Exception:                                                     # noqa: BLE001
        pass
    if t0 is not None:
        d["elapsed_s"] = round(time.time() - t0, 1)
    return d


def reset_peak():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                                     # noqa: BLE001
        pass


def env_block():
    return dict(python=platform.python_version(), numpy=str(np.__version__), pandas=str(pd.__version__), torch=pkg_version("torch"),
                sklearn=pkg_version("scikit-learn"), catboost=pkg_version("catboost"), pytabkit=pkg_version("pytabkit"))


def unit_meta(a, extra=None):
    d = dict(tag=a.TAG, run_id=str(a.RUN_ID or ""), code_sha=dict(a.CODE_SHA), code_sha_h40=H.code_sha(), env=env_block(), threads=int(a.threads),
             device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), gpu_uuid=str(_WINFO.get("gpu_uuid", "")), device_check=str(_WINFO.get("device_check", "")),
             platform=PLATFORM, revision_needed=list(a.REVISION_NEEDED), cfg_sig=str(a.CFG_SIG))
    d.update(extra or {})
    return d


def write_shard(a, pass_, learner, book, cfg, extra):
    """runs.csv → blocksse.npz(산출 제한이 아니면) → unit.json 순서로 쓴다. 모두 임시 파일 뒤 교체다. unit.json 이 완료 표지다."""
    c = book.c
    p = shard_paths(a, pass_, learner, c.target, c.mode, c.split)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(book.rows)
    if len(df):
        if a.NO_STORE:
            for col in ("rmse_cm", "rmse_beq_cm", "bias_cm"):
                df[col] = np.nan
        df = df[[k for k in RUN_COLS if k in df] + [k for k in df.columns if k not in RUN_COLS]]
    _atomic_csv(df, p["runs"])
    if not a.NO_STORE:
        save_stores([book.st], p["npz"])
    status, bad = book.status()
    unit = dict(target=c.target, mode=c.mode, parent=c.parent, split=int(c.split), part="cpu" if pass_ == "cpu" else "gpu", learner=learner,
                pass_=pass_, cfg=cfg, cfg_hash=H.cfg_hash(cfg), cfg_common=H.cfg_hash({k: v for k, v in cfg.items() if k not in CFG_LOCAL_KEYS}),
                status=status, failed_learners=bad,
                n_stored=int(len(book.st)), n_nonfinite_keys=int(book.n_bad), fits=dict(book.fits), fails=dict(book.fails), has_store=not a.NO_STORE,
                n_A=int(c.meta.get("n_A", len(c.yA))), n_eval=int(c.meta.get("n_eval", len(c.yB))), nb_eval=int(c.meta.get("nb_eval", 0)),
                n_src=int(c.meta.get("n_src", len(c.y_src))), E0=float(c.E0), valid=bool(c.meta.get("valid", True)))
    unit.update(unit_meta(a, extra))
    H._atomic_text(p["unit"], json.dumps(unit, ensure_ascii=False, indent=1, default=_pyval))
    return unit


# ================================================================ 워커(GPU 하나에 워커 하나)
_W = None
_WINFO: dict = {}


def _orphan_check():
    """워커 전용: 부모가 바뀌었으면(부모 종료) 바로 끝낸다(h43._orphan_check 와 같다)."""
    pp = _WINFO.get("ppid")
    if pp and os.getppid() != int(pp):
        os._exit(0)


def _worker_init(argv, gpu, threads, run_dir="", ppid=0, run_id=""):
    """워커 초기화: SIGINT 무시, 부모 종료 때 SIGTERM(h43._set_pdeathsig), CUDA_VISIBLE_DEVICES = 배정 GPU(torch 를 부르기 전),
    스레드 제한(torch intra-op = threads, inter-op = 1), 워커 기록(run_dir/worker__<PID>.json)."""
    global _W, _WINFO
    warnings.filterwarnings("ignore")
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    pds = T43._set_pdeathsig()
    if int(ppid) and os.getppid() != int(ppid):
        os._exit(0)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    for v in THREAD_VARS:
        os.environ[v] = str(int(threads))
    _W = parse_args(argv)
    _W.RUN_ID = str(run_id or "")
    check_sha(_W)
    ensure_configs(_W)
    _WINFO = dict(gpu="" if gpu is None else str(gpu), ppid=int(ppid) or os.getppid(), run_dir=str(run_dir), cuda_checked=False, pdeathsig=bool(pds))
    try:
        import torch
        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:                                                     # noqa: BLE001
        pass
    if run_dir:
        _write_json(Path(run_dir) / f"worker__{os.getpid()}.json", dict(pid=os.getpid(), ppid=_WINFO["ppid"], gpu=_WINFO["gpu"], start=_now(),
                                                                        pdeathsig=bool(pds), threads=int(threads), nice=int(os.nice(0))))


def _gpu_busy(g, why):
    rd = _WINFO.get("run_dir")
    if rd:
        _write_json(Path(rd) / f"gpu_busy__{int(g)}.json", dict(gpu=int(g), pid=os.getpid(), why=str(why), time=_now()))
    raise RuntimeError(f"{TAG_BUSY} GPU {g}: {why}")


def gpu_guard_n():
    """워커 전용 GPU 확인(h43.gpu_guard_t 와 같은 규칙). 첫 CUDA 초기화 전: 메모리 ≤ --gpu-mem-max-mib 이고 계산 프로세스 없음. 그 뒤: 자기 PID
    외의 계산 프로세스 없음. 걸리면 점유 표지를 남기고 TAG_BUSY 예외를 올린다. 첫 초기화 직후 자기 PID 의 GPU(없으면 torch UUID)가 배정과
    다르면 h43.TAG_MISMATCH 예외(주 프로세스가 실행 전체를 멈춘다). nvidia-smi 를 부를 수 없으면 경고하고 진행한다."""
    gpu = _WINFO.get("gpu")
    if gpu in (None, ""):
        return
    g = int(gpu)
    try:
        info, apps = T43.query_gpu_info(), T43.query_gpu_apps()
    except Exception as e:                                                # noqa: BLE001
        print(f"[worker] GPU {g}: nvidia-smi 확인 실패({repr(e)[:120]}). 확인 없이 진행한다", flush=True)
        return
    me = info.get(g)
    if me is None:
        print(f"[worker] GPU {g}: nvidia-smi 목록에 없다. 확인 없이 진행한다", flush=True)
        return
    foreign = sorted({int(p) for p, u in apps if u == me["uuid"] and int(p) != os.getpid()})
    if _WINFO.get("cuda_checked"):
        if foreign:
            _gpu_busy(g, f"다른 계산 프로세스 {foreign}")
        return
    if int(me["used"]) > int(_W.gpu_mem_max_mib) or foreign:
        _gpu_busy(g, f"첫 CUDA 초기화 전 메모리 사용 {int(me['used'])} MiB, 다른 계산 프로세스 {foreign}")
    import re
    import torch
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    tu = T43._norm_uuid(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    try:
        mine = sorted({u for p, u in T43.query_gpu_apps() if int(p) == os.getpid()})
    except Exception:                                                     # noqa: BLE001
        mine = []
    if mine:
        ok = mine == [me["uuid"]]
    elif re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", tu):
        ok = tu == T43._norm_uuid(me["uuid"])
    else:
        ok = None
    if ok is False:
        raise RuntimeError(f"{T43.TAG_MISMATCH} 배정 GPU {g}(UUID {me['uuid']})와 CUDA 장치가 다르다: 자기 PID 의 GPU {mine or '없음'}, torch UUID {tu or '없음'}")
    if ok is None:
        print(f"[worker] GPU {g}: 장치 대응을 확인하지 못했다", flush=True)
    _WINFO.update(cuda_checked=True, gpu_uuid=me["uuid"], device_check=("PID" if mine else ("UUID" if ok else "없음")))


def unit_name(u):
    return "|".join(str(v) for v in u)


def _worker_run(u):
    """작업 단위 하나(워커). 시작 때 GPU 를 확인한다. 주 모듈에서 정의한 예외는 내장 RuntimeError 로 바꿔 넘긴다."""
    _orphan_check()
    gpu_guard_n()
    t0 = time.time()
    try:
        res = run_unit(_W, u)
    except BaseException as e:                                            # noqa: BLE001
        if _WINFO.get("gpu") not in (None, "") and _WINFO.get("cuda_checked") and not T43.cuda_healthy():
            print(f"[worker] {unit_name(u)}: CUDA 문맥을 쓸 수 없다({repr(e)[:160]}). 워커를 끝낸다", flush=True)
            os._exit(76)
        if type(e).__module__ == "builtins":
            raise
        raise RuntimeError(f"{type(e).__name__}: {str(e)[:500]}") from None
    res = dict(res or {})
    res.update(unit=unit_name(u), wall_s=round(time.time() - t0, 1), gpu=_WINFO.get("gpu", ""))
    return res


def run_unit(a, u):
    """단위 종류별 실행."""
    fn = dict(sel=run_sel_unit, p0=run_p0_unit, p1=run_p1_unit, cbl=run_cbl_unit, cbsel=run_cbsel_unit, cbt=run_cbt_unit, chk=run_chk_unit,
              pre_a=run_pre_a, pre_b=run_pre_b, pre_c=run_pre_c, pre_d=run_pre_d)[u[0]]
    reset_peak()
    try:
        return fn(a, u)
    finally:
        gc.collect()
        T43._free_cuda()


# ================================================================ 선택(원천 지역 하나 제외 교차검증, 계획서 4.3)
def sel_path(a, u):
    _, lr, kind, t, m, stage, trial, seed = u
    return a.NSEL / f"{a.TAG}__sel__{lr}__{kind}__{t}__{m}__st{int(stage)}__t{int(trial)}__sd{int(seed)}.json"


def sel_unit_sig(a, u, si):
    _, lr, kind, t, m, stage, trial, seed = u
    ent = cfg_entry(lr, kind, trial)
    d = dict(learner=lr, kind=kind, target=t, mode=m, stage=int(stage), trial=int(trial), seed=int(seed), entry_sig=ent["sig"], src_hash=si.src_hash,
             regions=list(si.regions), epochs=int(a.epochs) if int(trial) == 0 else None, realmlp_epochs=int(a.realmlp_epochs),
             max_epochs=int(a.MAX_EPOCHS[lr]), patience=int(a.tuned_patience), no_store=bool(a.NO_STORE), tag=a.TAG)
    return _sha(d)


def load_sel(a, u):
    try:
        return json.loads(sel_path(a, u).read_text())
    except (OSError, ValueError):
        return None


def sel_done(a, D, u, run_id=None):
    r = load_sel(a, u)
    if r is None:
        return False
    si = src_info(a, D, u[3], u[4])
    if r.get("unit_sig") != sel_unit_sig(a, u, si):
        return False
    if getattr(a, "rerun_partial", False) and str(r.get("status", "ok")) != "ok":
        return False                                                      # --rerun-partial: fold 실패가 있는 선택 JSON 은 다시 돈다
    return run_id is None or r.get("run_id") == run_id


def sel_scores(folds, kind):
    """주 점수(fold RMSE 의 지역 등가중 평균), 셀 가중(√(ΣSSE/Σn)), 500셀 이상 fold 만의 점수, λ 0.25 점수(종류 R). 실패 fold 는 무한대."""
    if not folds:
        return dict(score_main=np.nan, score_cw=np.nan, score_ge500=np.nan, score_l25=np.nan)
    r = np.array([f["rmse"] for f in folds], float)
    sse = np.array([f["sse"] for f in folds], float)
    n = np.array([f["n_te"] for f in folds], float)
    r25 = np.array([f.get("rmse_l25", np.nan) for f in folds], float)
    ge = r[n >= 500]
    with np.errstate(invalid="ignore"):
        return dict(score_main=float(np.mean(r)), score_cw=float(np.sqrt(np.sum(sse) / np.sum(n))) if np.sum(n) > 0 else np.nan,
                    score_ge500=float(np.mean(ge)) if len(ge) else np.nan, score_l25=float(np.mean(r25)) if kind == "R" else np.nan)


def run_sel_unit(a, u):
    """선택 단위 하나: (학습기, 종류, 대상, 모드, 단계, 시도, seed) 의 fold 전부를 적합하고 선택 JSON 하나를 쓴다."""
    t0 = time.time()
    _, lr, kind, t, m, stage, trial, seed = u
    D = get_data(a)
    si = src_info(a, D, t, m)
    folds = fold_rows(si)
    ent = cfg_entry(lr, kind, trial)
    F = H.Fitter(ha_n(a, lr, "0"))
    streak, out = Streak(), []
    for fr in folds:
        _orphan_check()
        te = si.mac == fr["region"]
        tr = ~te
        E = H.ls_E(si.y[tr], si.s[tr]) if kind == "R" else 0.0
        yt = si.y[tr] - E * si.s[tr]
        t1, err, ep, npar, free = time.time(), "", -1, -1, False
        g = np.full(int(te.sum()), np.nan)
        try:
            if int(trial) == 0:
                (g,), flag = fit_default(F, lr, si.X[tr], yt, [si.X[te]], seed, dict(method="D0" if kind == "D" else "R0", n=0, draw=0, alpha="1",
                                                                                      placement="cell", sel=np.zeros(0, int)), axis="sel")
                if flag == "fail":
                    err = F.errors[-1] if F.errors else "fail"
                npar = default_n_params(lr, si.X.shape[1])
            else:
                (g,), ep, npar = fit_tuned(lr, ent, si.X[tr], yt, [si.X[te]], seed, f"sel|{t}|{m}|{fr['region']}", **tuned_kw(a, lr))
        except (ImportError, MemoryError):
            raise
        except Exception as e:                                            # noqa: BLE001
            err, free = fail_text(e), True
        if free:                                                          # 예외의 traceback 이 풀린 뒤(except 밖)에 메모리를 돌려준다
            gc.collect(); T43._free_cuda()
        g = np.asarray(g, float)
        fail = bool(err) or g.shape != (int(te.sum()),) or not np.all(np.isfinite(g))
        streak.update(lr, fail, f"{unit_name(u)}|{fr['region']}")
        rec = dict(fr, E_tr=float(E), fail=bool(fail), err=str(err), epochs_run=int(ep), n_params=int(npar), fit_s=round(time.time() - t1, 3))
        if a.NO_STORE:
            rec.update(rmse=np.nan, rmse_l25=np.nan, sse=np.nan, sse_l25=np.nan)
        elif fail:
            rec.update(rmse=np.inf, rmse_l25=np.inf, sse=np.inf, sse_l25=np.inf)
        else:
            yte, ste = si.y[te], si.s[te]
            s1 = float(np.sum((E * ste + g - yte) ** 2))
            s25 = float(np.sum((E * ste + 0.25 * g - yte) ** 2)) if kind == "R" else np.nan
            rec.update(rmse=float(np.sqrt(s1 / len(yte))), sse=s1, rmse_l25=float(np.sqrt(s25 / len(yte))) if kind == "R" else np.nan, sse_l25=s25)
        out.append(rec)
    nf = sum(f["fail"] for f in out)
    status = "ok" if nf == 0 else ("failed" if nf == len(out) else "partial")
    sc = sel_scores(out, kind) if not a.NO_STORE else sel_scores([], kind)
    npar = max([f["n_params"] for f in out] + [-1])
    rec = dict(learner=lr, kind=kind, target=t, mode=m, stage=int(stage), trial=int(trial), seed=int(seed), cfg_entry=ent, cfg_sig=str(a.CFG_SIG),
               sel_sig=sel_sig(a, lr), src_hash=si.src_hash, n_src=int(len(si.y)), regions=list(si.regions), n_folds=len(out), folds=out,
               n_params=int(npar), status=status, n_fail=int(nf), unit_sig=sel_unit_sig(a, u, si), fit_s_sum=round(sum(f["fit_s"] for f in out), 2),
               epochs_run=[f["epochs_run"] for f in out], **sc)
    rec.update(unit_meta(a, res_block(t0)))
    _write_json(sel_path(a, u), rec, strict=True)
    return dict(kind_="sel", status=status, n_fit=len(out), n_fail=int(nf), fit_s=rec["fit_s_sum"], epochs=rec["epochs_run"],
                gpu_mem_peak_mib=rec.get("gpu_mem_peak_mib"), rss_max_mib=rec.get("rss_max_mib"))


def stage1_units(a, lr, kind, t, m):
    return [("sel", lr, kind, t, m, 1, int(tr), 0) for tr in trials_of(a, lr)]


def stage2_units(a, lr, kind, t, m, trials):
    return [("sel", lr, kind, t, m, 2, int(tr), int(sd)) for tr in trials for sd in a.STAGE2_SEEDS]


def stage2_trials(a, lr, recs1):
    """1단계 기록 {시도: JSON} → 2단계 후보(시도 0 포함). 시도 1 이상에서 주 점수 상위 k2 개(동률이면 매개변수 수가 작은 쪽, 그다음 시도 번호).
    점수가 무한대(적합 실패)인 시도는 후보가 아니다. 스모크는 점수를 계산하지 않으므로 시도 번호 순 상위 k2 개다."""
    k2 = int(a.STAGE2_TOP[lr])
    rest = sorted(int(tr) for tr in recs1 if int(tr) != 0)
    if a.NO_STORE:
        return [0] + rest[:k2]
    cand = []
    for tr in rest:
        sc = float(recs1[tr].get("score_main", np.inf))
        if np.isfinite(sc):
            cand.append((sc, int(recs1[tr].get("n_params", -1)), tr))
    return [0] + sorted(q[2] for q in sorted(cand)[:k2])


def choose(S, npar, score=None, need_frac=0.75):
    """기본값 우선 규칙(계획서 4.3). S = {시도: fold 값 배열(seed 1·2 평균, 실패 inf)}, npar = {시도: 매개변수 수}, score = {시도: 점수}
    (기본 = S 의 fold 평균). c* = argmin 점수(동률이면 매개변수 수, 그다음 시도 번호). c* ≠ 0 이면 d_f = S_f(c*) − S_f(0), se = sd(ddof 1)/√F,
    차 = 점수(c*) − 점수(0) 가 음수이고 (−차 > se 이거나 d_f < 0 인 fold 수 ≥ ⌈need_frac·F⌉)일 때만 c* 를 고른다."""
    trials = sorted(int(c) for c in S)
    F = int(len(next(iter(S.values())))) if S else 0
    out = dict(trial=0, flag="", cstar=0, diff=np.nan, d_bar=np.nan, se=np.nan, wins=0, F=F, need=int(math.ceil(need_frac * F)) if F else 0,
               score_cstar=np.nan, score_0=np.nan)
    if F < 2:
        out["flag"] = "no_cv"
        return out
    sc = {c: (float(score[c]) if score is not None and c in score else float(np.mean(np.asarray(S[c], float)))) for c in trials}
    fin = [c for c in trials if np.isfinite(sc[c])]
    if not fin:
        out["flag"] = "all_failed"
        return out
    cstar = min(fin, key=lambda c: (sc[c], int(npar.get(c, 10 ** 12)) if int(npar.get(c, -1)) >= 0 else 10 ** 12, c))
    out.update(cstar=int(cstar), score_cstar=sc[cstar], score_0=sc.get(0, np.nan))
    if cstar == 0:
        return out
    if 0 not in sc or not np.isfinite(sc[0]):
        out.update(trial=int(cstar), flag="default_failed")
        return out
    d = np.asarray(S[cstar], float) - np.asarray(S[0], float)
    diff = float(sc[cstar] - sc[0])
    se = float(np.std(d, ddof=1) / math.sqrt(F)) if F >= 2 else np.nan
    wins = int(np.sum(d < 0))
    ok = diff < 0 and ((-diff > se) or wins >= out["need"])
    out.update(diff=diff, d_bar=float(np.mean(d)), se=se, wins=wins, trial=int(cstar) if ok else 0, flag="" if ok else "sel_default_pref")
    return out


def _fold_mat(recs2, trials, seeds, field):
    """{시도: fold 값의 seed 평균}. 한 seed 라도 없거나 무한대면 그 fold 는 무한대."""
    out = {}
    for tr in trials:
        vals = []
        for sd in seeds:
            r = recs2.get((int(tr), int(sd)))
            if r is None:
                vals.append(None); continue
            vals.append(np.array([f.get(field, np.nan) for f in r["folds"]], float))
        if any(v is None for v in vals) or not vals:
            F = max((len(v) for v in vals if v is not None), default=0)
            out[int(tr)] = np.full(F, np.inf)
            continue
        V = np.vstack(vals)
        V = np.where(np.isfinite(V), V, np.inf)
        out[int(tr)] = np.mean(V, 0)
    return out


def _cw_score(recs2, trials, seeds):
    out = {}
    for tr in trials:
        v = []
        for sd in seeds:
            r = recs2.get((int(tr), int(sd)))
            if r is None:
                v.append(np.inf); continue
            sse = np.array([f.get("sse", np.nan) for f in r["folds"]], float)
            n = np.array([f.get("n_te", 0) for f in r["folds"]], float)
            v.append(float(np.sqrt(np.sum(sse) / np.sum(n))) if np.all(np.isfinite(sse)) and np.sum(n) > 0 else np.inf)
        out[int(tr)] = float(np.mean(v)) if v else np.inf
    return out


def decide_group(a, lr, kind, recs2, trials):
    """2단계 기록 {(시도, seed): JSON} → 주 선택, 민감도 선택(cw, l25). 스모크는 시도 1 고정(smoke_fixed)."""
    seeds = list(a.STAGE2_SEEDS)
    if a.NO_STORE:
        tr = max(int(v) for v in trials) if trials else 0
        d = dict(trial=tr, flag="smoke_fixed", cstar=tr, F=0)
        return dict(main=dict(d), cw=dict(d), l25=dict(d) if kind == "R" else None)
    npar = {}
    for (tr, sd), r in recs2.items():
        npar[int(tr)] = int(r.get("n_params", -1))
    S = _fold_mat(recs2, trials, seeds, "rmse")
    main = choose(S, npar)
    cw = choose(S, npar, score=_cw_score(recs2, trials, seeds))
    l25 = choose(_fold_mat(recs2, trials, seeds, "rmse_l25"), npar) if kind == "R" else None
    return dict(main=main, cw=cw, l25=l25)


def fit_rows_of(a, lr, kind, t, m, recs):
    rows = []
    for r in recs:
        if r is None:
            continue
        fo = r.get("folds", [])
        rows.append(dict(row_type="fit", criterion="", target=t, mode=m, src_hash=r.get("src_hash", ""), learner=lr, kind=kind, trial=int(r["trial"]),
                         stage=int(r["stage"]), seed=int(r["seed"]), cfg_json=json.dumps(r.get("cfg_entry", {}).get("params", {}), sort_keys=True),
                         fold_regions=json.dumps([f["region"] for f in fo]), fold_rmse=json.dumps([f.get("rmse") for f in fo], default=_pyval),
                         fold_rmse_l25=json.dumps([f.get("rmse_l25") for f in fo], default=_pyval),
                         fold_sse=json.dumps([f.get("sse") for f in fo], default=_pyval), fold_n_te=json.dumps([f["n_te"] for f in fo]),
                         fold_n_tr=json.dumps([f["n_tr"] for f in fo]), fold_n_tr_within100km=json.dumps([f["n_tr_within100km"] for f in fo]),
                         score_main=r.get("score_main"), score_cw=r.get("score_cw"), score_ge500=r.get("score_ge500"), score_l25=r.get("score_l25"),
                         n_params=r.get("n_params"), epochs_run=json.dumps(r.get("epochs_run", [])), fit_s=r.get("fit_s_sum"), n_fail=r.get("n_fail"),
                         status=r.get("status"), selected=False, sel_flag="", sel_sig=sel_sig(a, lr)))
    return rows


def dec_rows_of(a, lr, kind, t, m, dec, trials2):
    rows = []
    for crit in ("main", "cw", "l25"):
        d = dec.get(crit)
        if d is None:
            continue
        rows.append(dict(row_type="decision", criterion=crit, target=t, mode=m, learner=lr, kind=kind, trial=int(d["trial"]), stage=2, seed=-1,
                         cfg_json=json.dumps(cfg_entry(lr, kind, d["trial"])["params"], sort_keys=True), selected=True, sel_flag=str(d.get("flag", "")),
                         cstar=int(d.get("cstar", 0)), diff=d.get("diff", np.nan), d_bar=d.get("d_bar", np.nan), se=d.get("se", np.nan),
                         wins=int(d.get("wins", 0)), F=int(d.get("F", 0)), need=int(d.get("need", 0)), score_cstar=d.get("score_cstar", np.nan),
                         score_0=d.get("score_0", np.nan), stage2_trials=json.dumps([int(v) for v in trials2]), sel_sig=sel_sig(a, lr)))
    return rows


def write_select(a, lr, kind, t, m, rows):
    """선택 표를 원자적으로 갱신한다. 이 (학습기, 종류, 대상, 모드)의 행을 새 행으로 바꾸고, 다른 행은 현재 sel_sig 와 같은 것만 보존한다.
    선택된 시도의 fit 행에 selected, sel_flag 를 적는다. 민감도 선택 표와 안정성 표도 다시 쓴다."""
    new = pd.DataFrame(rows)
    old = read_select(a)
    if len(old):
        cur = {v: sel_sig(a, v) for v in LEARNERS_ALL}
        keep = old.apply(lambda r_: str(r_.get("sel_sig", "")) == cur.get(str(r_.get("learner", "")), "~")
                         and not (r_["learner"] == lr and r_["kind"] == kind and r_["target"] == t and r_["mode"] == m), axis=1)
        old = old[keep]
        df = pd.concat([old, new], ignore_index=True) if len(old) else new
    else:
        df = new
    dm = df[(df.row_type == "decision") & (df.criterion == "main")]
    sel_set = {(str(r_.learner), str(r_.kind), str(r_.target), str(r_["mode"]), int(r_.trial)): str(r_.sel_flag) for _, r_ in dm.iterrows()}
    fr = df.row_type == "fit"
    df.loc[fr, "selected"] = [(str(r_.learner), str(r_.kind), str(r_.target), str(r_["mode"]), int(r_.trial)) in sel_set and int(r_.stage) == 2
                              for _, r_ in df[fr].iterrows()]
    df.loc[fr, "sel_flag"] = [sel_set.get((str(r_.learner), str(r_.kind), str(r_.target), str(r_["mode"]), int(r_.trial)), "")
                              if bool(r_.selected) else "" for _, r_ in df[fr].iterrows()]
    _atomic_csv(df, a.SELECT)
    write_sens_tables(a, df)
    return df


def read_select(a):
    if not Path(a.SELECT).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(a.SELECT, dtype=dict(target=str, mode=str, learner=str, kind=str, sel_flag=str, criterion=str, row_type=str, sel_sig=str),
                         keep_default_na=False, na_values=["", "nan", "NaN"])
    except Exception:                                                     # noqa: BLE001
        return pd.DataFrame()
    for c_ in ("sel_flag", "criterion", "sel_sig"):
        if c_ in df:
            df[c_] = df[c_].fillna("").astype(str)
    return df


_DEC_CACHE: dict = {}


def decisions_table(a, df=None):
    """{(학습기, 종류, 대상, 모드): {기준: dict(trial, flag, …)}} (현재 sel_sig 의 행만). df 를 주지 않으면 선택 표 파일을 읽는다
    (파일의 수정 시각·크기와 sel_sig 가 같으면 캐시를 쓴다. 조각 재개 판정이 단위마다 부르기 때문이다)."""
    if df is None:
        p = Path(a.SELECT)
        try:
            stt = p.stat()
            fkey = (str(p), int(stt.st_mtime_ns), int(stt.st_size))
        except OSError:
            fkey = (str(p), 0, 0)
        key = fkey + tuple(sel_sig(a, v) for v in LEARNERS_ALL)
        if key not in _DEC_CACHE:
            _DEC_CACHE.clear()
            _DEC_CACHE[key] = _decisions_from(a, read_select(a))
        return copy.deepcopy(_DEC_CACHE[key])
    return _decisions_from(a, df)


def _decisions_from(a, df):
    out: dict = {}
    if not len(df) or "row_type" not in df:
        return out
    cur = {v: sel_sig(a, v) for v in LEARNERS_ALL}
    for _, r_ in df[df.row_type == "decision"].iterrows():
        if str(r_.sel_sig) != cur.get(str(r_.learner), "~"):
            continue
        out.setdefault((str(r_.learner), str(r_.kind), str(r_.target), str(r_["mode"])), {})[str(r_.criterion)] = dict(
            trial=int(r_.trial), flag=str(r_.sel_flag), cstar=int(r_.get("cstar", 0) if pd.notna(r_.get("cstar", 0)) else 0))
    return out


def load_decision(a, lr, t, m):
    """통과 0·1 이 쓰는 선택(종류 D·R). 없으면 RuntimeError."""
    tab = decisions_table(a)
    out = {}
    for kind in KINDS:
        d = tab.get((lr, kind, t, m))
        if not d or "main" not in d:
            raise RuntimeError(f"선택 표에 {lr}|{kind}|{t}|{m} 의 선택이 없다(sel_sig {sel_sig(a, lr)})")
        out[kind] = d
    return out


def write_sens_tables(a, df):
    """민감도 선택 표(기준 cw, l25)와 안정성 표(주 선택과 민감도 선택이 다른 (학습기, 종류, 대상))."""
    dec = df[df.row_type == "decision"] if len(df) and "row_type" in df else pd.DataFrame()
    if not len(dec):
        return
    cols = ["learner", "kind", "target", "mode", "criterion", "trial", "cstar", "sel_flag", "diff", "d_bar", "se", "wins", "F", "need", "sel_sig"]
    sens = dec[dec.criterion.isin(["cw", "l25"])][[c_ for c_ in cols if c_ in dec]].copy()
    main = dec[dec.criterion == "main"].set_index(["learner", "kind", "target", "mode"]).trial.to_dict()
    if len(sens):
        sens["main_trial"] = [main.get((r_.learner, r_.kind, r_.target, r_["mode"]), np.nan) for _, r_ in sens.iterrows()]
        sens["same_as_main"] = sens.trial == sens.main_trial
    _atomic_csv(sens, a.SENS_T)
    rows = []
    for (lr, kind, t, m), g in dec.groupby(["learner", "kind", "target", "mode"]):
        d = dict(zip(g.criterion, g.trial))
        rows.append(dict(learner=lr, kind=kind, target=t, mode=m, main_trial=d.get("main"), cw_trial=d.get("cw"), l25_trial=d.get("l25"),
                         changed_cw=bool(d.get("cw") is not None and d.get("cw") != d.get("main")),
                         changed_l25=bool(d.get("l25") is not None and d.get("l25") != d.get("main")),
                         sel_default=bool(d.get("main") == 0), main_flag=str(g[g.criterion == "main"].sel_flag.iloc[0]) if (g.criterion == "main").any() else ""))
    _atomic_csv(pd.DataFrame(rows), a.STAB_T)


# ================================================================ 통과 0(n = 0 공유 적합, 계획서 4.4)
def p0_plan(dec, kind):
    """종류 하나의 판 → 시도. 기본판은 시도 0, 조정판은 주 선택, cw 는 셀 가중 선택, l25(종류 R)는 λ 0.25 선택."""
    d = dec[kind]
    plan = [("default", 0), ("tuned", int(d["main"]["trial"]))]
    if "cw" in d:
        plan.append(("tuned_cw", int(d["cw"]["trial"])))
    if kind == "R" and "l25" in d:
        plan.append(("tuned_l25", int(d["l25"]["trial"])))
    return plan


def p0_books(a, lr, t, m, cs, dec):
    """통과 0 의 핵심(합성 자료 시험에서도 부른다). cs = 분할별 h40.Ctx(원천 배열이 같아야 한다), dec = 종류별 선택.
    판 × 종류 × seed 마다 한 번 적합해 모든 분할의 채점 셀을 예측하고 분할별 ShardBook 을 만든다. 같은 시도는 한 번만 적합한다.
    반환 (books, fits, 원천 해시)."""
    hs = sorted({src_arrays_hash(c) for c in cs})
    if len(hs) != 1:
        raise RuntimeError(f"{t}|{m}: 분할 사이 원천 배열(X_src, y_src, r0_src)이 다르다 {hs}")
    c0 = cs[0]
    preds_B = [c.XB for c in cs]
    F = H.Fitter(ha_n(a, lr, "0"))
    streak = Streak()
    fits: dict = {}                                                       # (종류, seed, 시도) → dict(p = 분할별 예측, flag, fit_s, epochs, n_params)
    books = [ShardBook(c, "gpu", "lgfn_p0", hide=a.NO_STORE) for c in cs]
    E0 = float(c0.E0)
    for b_ in books:
        b_.add("P0", "none", 0, 0, -1, 0.0, E0 * b_.c.sB, E0, 0, ex={"pass": "p0", "shared_n0": True})
        b_.add("P1", "none", 0, 0, -1, 0.0, E0 * b_.c.sB, E0, 0, ex={"pass": "p0", "shared_n0": True})
    for kind in KINDS:
        ytr = c0.y_src if kind == "D" else c0.r0_src
        plan = p0_plan(dec, kind)
        for seed in a.SEEDS:
            for arm, trial in plan:
                key = (kind, int(seed), int(trial))
                name = arm_name(lr, arm)
                if key not in fits:
                    t1, err, ep, npar, free = time.time(), "", -1, -1, False
                    try:
                        if trial == 0:
                            out, flag = fit_default(F, lr, c0.X_src, ytr, preds_B, seed, dict(method="D0" if kind == "D" else "R0", n=0, draw=0,
                                                                                            alpha="1", placement="cell", sel=np.zeros(0, int)))
                            if flag == "fail":
                                err = F.errors[-1] if F.errors else "fail"
                            npar = default_n_params(lr, c0.X_src.shape[1])
                        else:
                            out, ep, npar = fit_tuned(lr, cfg_entry(lr, kind, trial), c0.X_src, ytr, preds_B, seed, f"p0|{t}|{m}", **tuned_kw(a, lr))
                    except (ImportError, MemoryError):
                        raise
                    except Exception as e:                                # noqa: BLE001
                        err, out, free = fail_text(e), [np.full(len(q), np.nan) for q in preds_B], True
                    if free:
                        gc.collect(); T43._free_cuda()
                    fail = bool(err) or not all(np.all(np.isfinite(np.asarray(q, float))) for q in out)
                    streak.update(lr, fail, f"p0|{t}|{m}|{kind}|s{seed}|t{trial}")
                    fits[key] = dict(p=out, flag="fail" if err else "", fit_s=round(time.time() - t1, 3), epochs=int(ep), n_params=int(npar),
                                     err=str(err), owner=name)
                    for b_ in books:
                        b_.fits[name] += 1
                        b_.fails[name] += int(fail)
                fr = fits[key]
                d = dec[kind].get({"tuned": "main", "tuned_cw": "cw", "tuned_l25": "l25"}.get(arm, "main"), {})
                ex = {"arm": arm, "trial": int(trial), "sel_default": bool(arm != "default" and trial == 0),
                      "sel_flag": str(d.get("flag", "")) if arm != "default" else "", "n_params": fr["n_params"], "epochs_run": fr["epochs"], "pass": "p0",
                      "fit_s": fr["fit_s"] if fr["owner"] == name else 0.0, "shared_n0": True}
                for i, b_ in enumerate(books):
                    q = np.asarray(fr["p"][i], float)
                    if kind == "D":
                        b_.add("D0", name, 0, 0, seed, 1.0, q, np.nan, 0, 0, fr["flag"], ex)
                    else:
                        for lam in a.LAMS:
                            b_.add("R1", name, 0, 0, seed, lam, E0 * b_.c.sB + lam * q, E0, 0, 0, fr["flag"], ex)
    return books, fits, hs[0]


def run_p0_unit(a, u):
    """(학습기, 대상, 모드)의 통과 0. 분할 1–5 의 조각을 한꺼번에 쓴다(unit.json 의 shared_n0 = True)."""
    t0 = time.time()
    _, lr, t, m = u
    D = get_data(a)
    HA = ha_n(a, lr, "0")
    splits = valid_splits(a, D, t, m, a.SPLITS)
    if not splits:
        raise RuntimeError(f"{t}|{m}: 실행할 분할이 없다")
    for sp in splits:                                                     # 이전 세대의 완료 표지를 먼저 지운다(분할 1–5 전부, 명세 7)
        shard_paths(a, "p0", lr, t, m, sp)["unit"].unlink(missing_ok=True)
    cs = [H.build_ctx(D, HA, t, m, sp) for sp in splits]
    dec = load_decision(a, lr, t, m)
    books, fits, hs = p0_books(a, lr, t, m, cs, dec)
    smoke = {}
    if a.smoke:                                                           # 경로 점검: 공유 적합의 분할별 자르기와 분할별 단독 적합(h40.run_ctx n = 0)
        smoke = p0_slice_check(a, lr, cs[0], books[0])
    cfg = shard_cfg(a, "p0", lr, dec=dec)
    units = []
    fdet = {f"{k_[0]}|s{k_[1]}|t{k_[2]}": dict(fit_s=v["fit_s"], epochs=v["epochs"], n_params=v["n_params"], err=v["err"]) for k_, v in fits.items()}
    for b_ in books:
        units.append(write_shard(a, "p0", lr, b_, cfg, dict(res_block(t0), shared_n0=True, splits_shared=list(splits), src_hash=hs,
                                                           sel_sig=sel_sig(a, lr), k=int(a.N_RANDOM[lr]), cfg_sig=str(a.CFG_SIG), decisions=dec,
                                                           n_fit=len(fits), fit_detail=fdet, smoke_check=smoke)))
    st = [u_["status"] for u_ in units]
    status = "failed" if "failed" in st else ("partial" if "partial" in st else "ok")
    return dict(kind_="p0", status=status, n_fit=len(fits), n_fail=int(sum(bool(v["err"]) for v in fits.values())),
                fit_s=round(sum(v["fit_s"] for v in fits.values()), 2), splits=list(splits), smoke_check=smoke, **res_block(t0))


def p0_slice_check(a, lr, c, book):
    """스모크: 분할 1 에서 h40.run_ctx(n = 0, 학습기 하나)로 다시 적합한 기본판 키와 공유 적합 키의 블록 RMSE 최대 차(cm)."""
    HA = ha_n(a, lr, "0")
    _, st_ref, _ = H.run_ctx(c, "gpu", HA, learner_axis=True, learners=[lr])
    diffs = []
    for k in st_ref.keys:
        if k[1] == lr and k in book.st:
            s1, c1 = st_ref.get(k)
            s2, c2 = book.st.get(k)
            with np.errstate(invalid="ignore", divide="ignore"):
                r1, r2 = np.sqrt(s1 / np.maximum(c1, 1)), np.sqrt(s2 / np.maximum(c2, 1))
            diffs.append(float(np.nanmax(np.abs(r1 - r2))) if len(r1) else 0.0)
    return dict(p0_slice_keys=len(diffs), p0_slice_maxdiff_cm=float(max(diffs)) if diffs else None,
                p0_slice_ok=bool(diffs and max(diffs) <= 1e-6))


# ================================================================ 통과 1(n > 0)
def p1_book(a, lr, t, m, sp, c, dec, trace_on=False):
    """통과 1 의 핵심(합성 자료 시험에서도 부른다). 기본판은 h40.run_ctx(learner-n-grid = 양수 격자), 조정판은 같은 (n, 추출, seed) 칸의
    같은 학습 행(원천 ∪ 라벨 n개, D 는 y, R 은 원천 E0 잔차 ∪ 라벨 E_n 잔차)으로 fit_tuned 한다. 선택이 시도 0 이면 기본판 키의 SSE 를 복사한다.
    반환 (book, 기본판 통계, 조정판 적합 기록, TRACE 기록)."""
    lg = lgrid_pos(a)
    if not lg:
        raise RuntimeError("n 격자에 양수 n 이 없다")
    HA1 = ha_n(a, lr, lg)
    if trace_on:
        H.TRACE = []
    try:
        rows, st, stats = H.run_ctx(c, "gpu", HA1, learner_axis=True, learners=[lr])
        trace = list(H.TRACE or [])
    finally:
        H.TRACE = None
    if st is None:
        raise RuntimeError(f"{t}|{m}|s{sp}: 채점 셀 또는 A 셀이 없다")
    npar0 = default_n_params(lr, c.X_src.shape[1])
    for r_ in rows:
        r_.update(EXTRA_DEFAULTS)
        r_.update({"pass": "p1"})
        if r_["learner"] == lr:
            r_.update(arm="default", trial=0, n_params=npar0)
    book = ShardBook(c, "gpu", "lgfn_p1", hide=a.NO_STORE, st=st, rows=rows)
    nd = dict(stats.get("n_fit_detail", {}))
    book.fits[lr] = int(sum(v for k, v in nd.items() if k.split("|")[1:2] == [lr]))
    book.fails[lr] = len({(r_["method"], int(r_["n"]), int(r_["draw"]), int(r_["seed"])) for r_ in rows if r_["learner"] == lr
                          and (int(r_["n_nonfinite"]) > 0 or str(r_["fit_flag"]) == "fail")})
    name = arm_name(lr, "tuned")
    nA = len(c.yA)
    streak = Streak()
    fit_log = []
    for n, d in H.cells_of([n for n in a.N_GRID if n != 0], int(a.DRAWS_P1), nA):
        sel = H.draw_cells(t, m, int(sp), n, d, nA)
        nl = len(sel)
        nb = len(np.unique(c.blkA[sel])) if nl else 0
        E_n = T43.coef_n(c, sel, a.kappa)
        Xtr = np.vstack([c.X_src, c.XA[sel]])
        for kind in KINDS:
            trial = int(dec[kind]["main"]["trial"])
            flag_sel = str(dec[kind]["main"].get("flag", ""))
            for seed in a.SEEDS:
                ex = {"arm": "tuned", "trial": trial, "sel_default": trial == 0, "sel_flag": flag_sel, "pass": "p1"}
                if trial == 0:                                            # 선택이 시도 0: 기본판 키의 SSE 를 복사한다(다시 적합하지 않음)
                    if kind == "D":
                        book.copy(("D0", lr, "1", "cell", n, d, seed, 1.0), "D0", name, n, d, seed, 1.0, np.nan, nl, nb, ex)
                    else:
                        for lam in a.LAMS:
                            book.copy(("R1", lr, "1", "cell", n, d, seed, float(lam)), "R1", name, n, d, seed, lam, E_n, nl, nb, ex)
                    continue
                ytr = np.concatenate([c.y_src, c.yA[sel]]) if kind == "D" else np.concatenate([c.r0_src, c.yA[sel] - E_n * c.sA[sel]])
                t1, err, ep, npar, free = time.time(), "", -1, -1, False
                try:
                    (g,), ep, npar = fit_tuned(lr, cfg_entry(lr, kind, trial), Xtr, ytr, [c.XB], seed, f"p1|{t}|{m}|{sp}|{n}|{d}", **tuned_kw(a, lr))
                except (ImportError, MemoryError):
                    raise
                except Exception as e:                                    # noqa: BLE001
                    err, g, free = fail_text(e), np.full(len(c.yB), np.nan), True
                if free:
                    gc.collect(); T43._free_cuda()
                g = np.asarray(g, float)
                fail = bool(err) or g.shape != (len(c.yB),) or not np.all(np.isfinite(g))
                streak.update(lr, fail, f"p1|{t}|{m}|s{sp}|n{n}|d{d}|{kind}|s{seed}")
                book.fits[name] += 1
                book.fails[name] += int(fail)
                ex.update(n_params=int(npar), epochs_run=int(ep), fit_s=round(time.time() - t1, 3))
                fit_log.append(dict(kind=kind, n=int(n), draw=int(d), seed=int(seed), trial=trial, fit_s=ex["fit_s"], epochs=int(ep), err=err))
                if kind == "D":
                    book.add("D0", name, n, d, seed, 1.0, g, np.nan, nl, nb, "fail" if err else "", ex)
                else:
                    for lam in a.LAMS:
                        book.add("R1", name, n, d, seed, lam, E_n * c.sB + lam * g, E_n, nl, nb, "fail" if err else "", ex)
    return book, stats, fit_log, trace


def run_p1_unit(a, u):
    """(학습기, 대상, 모드, 분할)의 통과 1 조각."""
    t0 = time.time()
    _, lr, t, m, sp = u
    shard_paths(a, "p1", lr, t, m, sp)["unit"].unlink(missing_ok=True)   # 이전 세대의 완료 표지를 먼저 지운다(명세 7)
    D = get_data(a)
    c = H.build_ctx(D, ha_n(a, lr, lgrid_pos(a) or "0"), t, m, int(sp))
    dec = load_decision(a, lr, t, m)
    book, stats, fit_log, trace = p1_book(a, lr, t, m, sp, c, dec, trace_on=bool(a.smoke))
    smoke = trace_check(c, trace, lr, a.kappa) if a.smoke else {}
    name = arm_name(lr, "tuned")
    unit = write_shard(a, "p1", lr, book, shard_cfg(a, "p1", lr, dec=dec), dict(res_block(t0), sel_sig=sel_sig(a, lr), k=int(a.N_RANDOM[lr]),
                                                                      decisions={k_: dv["main"] for k_, dv in dec.items()}, default_stats=dict(
                                                                          n_fit=stats.get("n_fit"), sec=stats.get("sec"), fail=stats.get("fail")),
                                                                      tuned_fits=fit_log, smoke_check=smoke, shared_n0=False,
                                                                      n_fit_detail=stats.get("n_fit_detail", {}), sec_detail=stats.get("sec_detail", {})))
    return dict(kind_="p1", status=unit["status"], n_fit=int(book.fits[lr] + book.fits[name]), n_fail=int(sum(book.fails.values())),
                fit_s=round(sum(q["fit_s"] for q in fit_log), 2), smoke_check=smoke, **res_block(t0))


def trace_check(c, trace, lr, kappa):
    """스모크: h40.run_ctx 의 기본판 학습 행렬(H.TRACE)이 조정판이 쓰는 학습 행렬(원천 ∪ 라벨 n개, D 는 y, R 은 E_n 잔차)과 같은지."""
    n_ok, n_all = 0, 0
    for e in trace:
        if e.get("learner") != lr or e.get("method") not in ("D0", "R1"):
            continue
        sel = np.asarray(e.get("sel", np.zeros(0, int)), int)
        if len(sel) == 0:
            continue
        n_all += 1
        X = np.vstack([c.X_src, c.XA[sel]])
        if e["method"] == "D0":
            y = np.concatenate([c.y_src, c.yA[sel]])
        else:
            y = np.concatenate([c.r0_src, c.yA[sel] - T43.coef_n(c, sel, kappa) * c.sA[sel]])
        ok = np.array_equal(np.asarray(e["Xtr"], np.float32), X.astype(np.float32), equal_nan=True) and np.array_equal(np.asarray(e["ytr"], float), y)
        n_ok += int(ok)
    return dict(trace_fits=n_all, trace_match=n_ok, trace_ok=bool(n_all > 0 and n_ok == n_all))


# ================================================================ CPU 작업(catboost_lo, catboost_tuned_loc)
def cb_sel_sig(a):
    X = _load_h42()
    HA = ha_n(a, "mlp", "0")
    d = dict(grid=[list(v) for v in X.TUNE_GRID], lr=float(X.TUNE_LR), min_cells=int(a.min_region_cells), buffer_km=float(HA.buffer_km),
             k_sub=str(HA.k_sub), subregion_map=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans", threads=int(a.threads), no_store=bool(a.NO_STORE))
    return _sha(d)


def cb_select_rows(a, D, target, mode, hide=False):
    """catboost_tuned_loc 선택 표(h42.select_rows 와 같은 행 형식). 원천 30셀 이상 macro 지역 하나 제외, kind × h42.TUNE_GRID, seed 0,
    thread_count = --threads. hide = True 이면 점수 열을 NaN 으로 둔다(사전 점검)."""
    X = _load_h42()
    si = src_info(a, D, target, mode)
    regs = list(si.regions)
    base = dict(target=target, mode=mode, n_src=int(len(si.y)), n_regions=len(regs), regions=",".join(regs), sel_sig=cb_sel_sig(a))
    th = int(a.threads)
    rows = []
    for kind in KINDS:
        for dp, it in X.TUNE_GRID:
            t0 = time.time(); r1, r25 = [], []
            for r in regs:
                te = si.mac == r; tr = ~te
                E = H.ls_E(si.y[tr], si.s[tr]) if kind == "R" else 0.0
                g = _cb_tuned_fit(dict(iterations=int(it), learning_rate=float(X.TUNE_LR), depth=int(dp), l2_leaf_reg=3.0), si.X[tr],
                                  si.y[tr] - E * si.s[tr], si.X[te], 0, th)
                r1.append(float(np.sqrt(np.mean((E * si.s[te] + g - si.y[te]) ** 2))))
                r25.append(float(np.sqrt(np.mean((E * si.s[te] + H.LAM_BASE * g - si.y[te]) ** 2))) if kind == "R" else np.nan)
            row = dict(base, kind=kind, depth=int(dp), iterations=int(it), learning_rate=float(X.TUNE_LR),
                       score_rmse=float(np.mean(r1)) if r1 else np.nan, score_rmse_lam025=float(np.mean(r25)) if r25 else np.nan,
                       rmse_by_region=json.dumps(dict(zip(regs, [round(v, 4) for v in r1]))), n_fit=len(regs), sec=round(time.time() - t0, 1))
            if hide:
                row.update(score_rmse=np.nan, score_rmse_lam025=np.nan, rmse_by_region="")
            rows.append(row)
    return rows


def write_cb_select(a, rows):
    """h42.pick_tuned 로 selected 를 붙여 CatBoost 선택 표를 갱신한다(같은 sel_sig 의 다른 대상 행은 보존)."""
    X = _load_h42()
    df = X.pick_tuned(rows)
    if not len(df):
        return df
    if Path(a.CBSEL).exists():
        try:
            old = pd.read_csv(a.CBSEL, dtype=dict(target=str, mode=str, kind=str))
            if len(old) and "sel_sig" in old:
                keys = set(zip(df.target, df["mode"]))
                old = old[(old.sel_sig.astype(str) == cb_sel_sig(a)) & ~pd.Series(list(zip(old.target, old["mode"])), index=old.index).isin(keys)]
                df = pd.concat([old, df], ignore_index=True)
        except Exception:                                                 # noqa: BLE001
            pass
    _atomic_csv(df, a.CBSEL)
    return df


def load_cb_tuned(a, target, mode):
    """CatBoost 선택 표에서 대상·모드의 설정 {'D': {...}, 'R': {...}}. 없으면 빈 dict."""
    if not Path(a.CBSEL).exists():
        return {}
    try:
        df = pd.read_csv(a.CBSEL, dtype=dict(target=str, mode=str, kind=str))
    except Exception:                                                     # noqa: BLE001
        return {}
    if not len(df) or "sel_sig" not in df:
        return {}
    df = df[(df.sel_sig.astype(str) == cb_sel_sig(a)) & (df.target == target) & (df["mode"] == mode) & (df.selected.astype(str).str.lower() == "true")]
    return {str(r_.kind): dict(iterations=int(r_.iterations), depth=int(r_.depth), learning_rate=float(r_.learning_rate), l2_leaf_reg=3.0)
            for r_ in df.itertuples()}


def run_cbsel_unit(a, u):
    t0 = time.time()
    _, t, m = u
    rows = cb_select_rows(a, get_data(a), t, m, hide=a.NO_STORE)
    write_cb_select(a, rows)
    return dict(kind_="cbsel", status="ok", n_fit=int(sum(r_["n_fit"] for r_ in rows)), fit_s=round(sum(r_["sec"] for r_ in rows), 1), **res_block(t0))


def _cb_tuned_fit(cfg, X, y, XB, seed, threads):
    """CatBoost 적합과 예측(catboost_tuned_loc 의 선택과 적합). catboost 1.2.10 은 fit 에 numpy 배열을 주면 학습 Pool 을 thread_count 기본값
    −1(모든 논리 코어)로 만든다. 그래서 학습과 채점의 Pool 을 thread_count = threads 로 직접 만든다(h43.cb_fit_predict 와 같은 방식)."""
    from catboost import CatBoostRegressor, Pool
    th = int(threads)
    mdl = CatBoostRegressor(iterations=int(cfg["iterations"]), learning_rate=float(cfg["learning_rate"]), depth=int(cfg["depth"]),
                            l2_leaf_reg=float(cfg.get("l2_leaf_reg", 3.0)), random_seed=int(seed), verbose=0, allow_writing_files=False,
                            thread_count=th)
    mdl.fit(Pool(np.asarray(X, float), np.asarray(y, float), thread_count=th))
    return np.asarray(mdl.predict(Pool(np.asarray(XB, float), thread_count=th), thread_count=th), float)


def run_cpu_fits(a, u, learner):
    """분할 하나의 CatBoost 적합(모든 n 칸의 D0·R1, h40 train_rows 와 같은 학습 행). catboost_lo 는 h43.cb_fit_predict(반복 200, 깊이 3,
    학습률 0.05, 학습과 채점 Pool 의 thread_count), catboost_tuned_loc 는 선택 표의 설정(h42.FitterX 의 catboost_tuned 와 같은 인자)."""
    t0 = time.time()
    _, t, m, sp = u
    shard_paths(a, "cpu", learner, t, m, sp)["unit"].unlink(missing_ok=True)   # 이전 세대의 완료 표지를 먼저 지운다(명세 7)
    D = get_data(a)
    c = H.build_ctx(D, ha_n(a, "mlp", "0"), t, m, int(sp))
    cfg_t = load_cb_tuned(a, t, m) if learner == "catboost_tuned_loc" else {}
    if learner == "catboost_tuned_loc" and not all(k in cfg_t for k in KINDS):
        raise RuntimeError(f"{t}|{m}: catboost_tuned_loc 의 선택 표 행이 없다({a.CBSEL})")
    book = ShardBook(c, "cpu", "lgfn_cpu", hide=a.NO_STORE)
    E0 = float(c.E0)
    book.add("P0", "none", 0, 0, -1, 0.0, E0 * c.sB, E0, 0, ex={"pass": "cpu"})
    nA = len(c.yA)
    ns = SimpleNamespace(threads=int(a.threads), cb_iters=CB_ITERS)
    streak = Streak()
    for n, d in H.cells_of(list(a.N_GRID), int(a.DRAWS_P1), nA):
        sel = H.draw_cells(t, m, int(sp), n, d, nA)
        nl = len(sel)
        nb = len(np.unique(c.blkA[sel])) if nl else 0
        E_n = T43.coef_n(c, sel, a.kappa)
        book.add("P1", "none", n, d, -1, 0.0, E_n * c.sB, E_n, nl, nb, ex={"pass": "cpu"})
        Xtr = np.vstack([c.X_src, c.XA[sel]]) if nl else c.X_src
        for kind in KINDS:
            ytr = (np.concatenate([c.y_src, c.yA[sel]]) if kind == "D" else np.concatenate([c.r0_src, c.yA[sel] - E_n * c.sA[sel]]))
            for seed in a.SEEDS:
                t1, err = time.time(), ""
                try:
                    if learner == "catboost_lo":
                        g = np.asarray(T43.cb_fit_predict(ns, Xtr, ytr, c.XB, seed)[0], float)
                    else:
                        g = _cb_tuned_fit(cfg_t[kind], Xtr, ytr, c.XB, seed, a.threads)
                except (ImportError, MemoryError):
                    raise
                except Exception as e:                                    # noqa: BLE001
                    err, g = repr(e)[:200], np.full(len(c.yB), np.nan)
                fail = bool(err) or not np.all(np.isfinite(g))
                streak.update(learner, fail, f"{learner}|{t}|{m}|s{sp}|n{n}|d{d}|{kind}|s{seed}")
                book.fits[learner] += 1
                book.fails[learner] += int(fail)
                ex = {"arm": "default" if learner == "catboost_lo" else "tuned", "pass": "cpu", "fit_s": round(time.time() - t1, 3),
                      "trial": -1, "sel_flag": json.dumps(cfg_t.get(kind, {}), sort_keys=True) if cfg_t else ""}
                if kind == "D":
                    book.add("D0", learner, n, d, seed, 1.0, g, np.nan, nl, nb, "fail" if err else "", ex)
                else:
                    for lam in a.LAMS:
                        book.add("R1", learner, n, d, seed, lam, E_n * c.sB + lam * g, E_n, nl, nb, "fail" if err else "", ex)
    unit = write_shard(a, "cpu", learner, book, shard_cfg(a, "cpu", learner, cb_cfg=cfg_t), dict(res_block(t0), cb_cfg=cfg_t))
    return dict(kind_=learner, status=unit["status"], n_fit=int(book.fits[learner]), n_fail=int(book.fails[learner]), **res_block(t0))


def run_cbl_unit(a, u):
    return run_cpu_fits(a, u, "catboost_lo")


def run_cbt_unit(a, u):
    return run_cpu_fits(a, u, "catboost_tuned_loc")


# ================================================================ 스모크·사전 점검의 정적 점검과 시간 측정 단위
def tabm_loop_diff(dev="cpu", d_in=25, width=48, k=8, n_blocks=2, n=33, seed=0):
    """TabMp 의 묶음 헤드와, 같은 가중치를 복사한 nn.Linear 헤드 k 개를 차례로 계산한 결과의 최대 절대 차(평가 모드)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(int(seed))
    net = _mods().TabMp(d_in, n_blocks, width, k, dropout=0.0).to(dev).eval()
    heads = []
    for j in range(int(k)):
        h1, h2 = nn.Linear(width, 64), nn.Linear(64, 1)
        with torch.no_grad():
            h1.weight.copy_(net.W1[j].T); h1.bias.copy_(net.b1[j]); h2.weight.copy_(net.W2[j].T); h2.bias.copy_(net.b2[j])
        heads.append(nn.Sequential(h1, nn.ReLU(), h2).to(dev).eval())
    x = torch.randn(int(n), int(d_in), device=dev)
    with torch.no_grad():
        h = net.trunk(x)
        loop = torch.stack([hd(h).squeeze(-1) for hd in heads], 0).mean(0)
        return float(torch.max(torch.abs(net(x) - loop)).item())


def qt_repro(n=2500, d=25, seed=0):
    """같은 입력과 같은 적합 식별자이면 분위 정규 변환이 같은지(최대 절대 차)."""
    rng = np.random.RandomState(int(seed))
    X = rng.randn(n, d)
    X[rng.rand(n, d) < 0.05] = np.nan
    p = dict(transform="qnorm")
    f1, _ = make_transform(p, X, "mlp", "D", 3, "chk|a", 0)
    f2, _ = make_transform(p, X.copy(), "mlp", "D", 3, "chk|a", 0)
    return float(np.max(np.abs(f1(X) - f2(X))))


def rmlp_key_check():
    """RealMLP 표본 설정의 키가 모두 RealMLP_TD_Regressor 생성자 인자인지(inspect.signature)."""
    from pytabkit import RealMLP_TD_Regressor
    sig = set(inspect.signature(RealMLP_TD_Regressor.__init__).parameters)
    keys = set()
    for kind in KINDS:
        for e in build_configs(K_LIST_MAX)["realmlp"][kind]:
            keys |= set(e["params"]) - {"default"}
    miss = sorted(keys - sig)
    return dict(rmlp_keys=sorted(keys), rmlp_missing=miss, rmlp_keys_ok=not miss)


def run_chk_unit(a, u):
    """정적 점검(학습 없음): 클래스 출력 모양, TabM 묶음 헤드와 루프 헤드, 분위 변환 재현, RealMLP 인자 키."""
    import torch
    from polar import tab_models as TM
    t0 = time.time()
    dev = TM._dev()
    rows = []
    for lr in ("mlp", "tabm", "ftt"):
        for kind in KINDS:
            for e in build_configs(K_LIST_MAX)[lr][kind][1:4]:
                net = build_net(lr, e["params"], 25).to(dev).eval()
                with torch.no_grad():
                    out = net(torch.randn(7, 25, device=dev))
                rows.append(dict(check="output_shape", learner=lr, kind=kind, trial=int(e["trial"]), value=str(tuple(out.shape)),
                                 ok=bool(tuple(out.shape) == (7,)), n_params=int(sum(q.numel() for q in net.parameters()))))
                del net
    v = tabm_loop_diff(dev)
    rows.append(dict(check="tabm_heads_maxdiff", learner="tabm", value=v, ok=bool(v <= 1e-6)))
    v = qt_repro()
    rows.append(dict(check="qt_repro_maxdiff", learner="", value=v, ok=bool(v == 0.0)))
    rk = rmlp_key_check()
    rows.append(dict(check="realmlp_keys", learner="realmlp", value=json.dumps(rk["rmlp_missing"]), ok=bool(rk["rmlp_keys_ok"])))
    return dict(kind_="chk", status="ok" if all(r_["ok"] for r_ in rows) else "partial", rows=rows, **res_block(t0))


@contextlib.contextmanager
def count_epochs():
    """사전 점검 (b) 전용: tab_models._epochs_fit 과 같은 코드에 epoch 계수만 더한 대체 함수를 이 프로세스 안에서만 잠시 쓴다.
    반환 목록에 적합마다 실제 epoch 수가 쌓인다. 파일은 고치지 않는다."""
    from polar import tab_models as TM
    orig = TM._epochs_fit
    box = []

    def counting(net, Xtr, ytr, Xva, yva, epochs, bs=8192, lr=1e-3, wd=1e-5, pat=6):
        import torch
        import torch.nn as nn
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); Xv = torch.tensor(Xva).to(TM._dev())
        best, state, p, ep_run = 1e9, None, 0, 0
        for _ in range(epochs):
            net.train(); idx = torch.randperm(len(Xt))
            for k in range(0, len(Xt), bs):
                b = idx[k:k + bs]; xb, yb = Xt[b].to(TM._dev()), yt[b].to(TM._dev())
                opt.zero_grad(); nn.SmoothL1Loss()(net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            ep_run += 1
            net.eval()
            with torch.no_grad():
                v = float(np.mean((net(Xv).cpu().numpy() - yva) ** 2))
            if v < best - 1e-4:
                best, state, p = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
            else:
                p += 1
                if p >= pat:
                    break
        if state:
            net.load_state_dict(state)
        box.append(ep_run)
        return net
    TM._epochs_fit = counting
    try:
        yield box
    finally:
        TM._epochs_fit = orig


PRE_TARGET = ("Lena", "x")
PRE_FOLD_OUT = "Canada"


def run_pre_a(a, u):
    """(a) 설정 하나(종류 D)를 레나 x 원천 전체에서 2 epoch 적합해 epoch 당 시간을 잰다(RealMLP 는 n_epochs 2)."""
    t0 = time.time()
    _, lr, trial = u
    D = get_data(a)
    si = src_info(a, D, *PRE_TARGET)
    e = cfg_entry(lr, "D", trial)
    t1 = time.time()
    if int(trial) == 0:
        F = H.Fitter(ha_n(a, lr, "0", epochs=2, realmlp_epochs=2))
        fit_default(F, lr, si.X, si.y, [si.X[:64]], 0, dict(method="D0", n=0, draw=0, alpha="1", placement="cell", sel=np.zeros(0, int)), axis="pre")
        npar = default_n_params(lr, si.X.shape[1])
        err = F.errors[-1] if F.errors else ""
    else:
        try:
            _, _, npar = fit_tuned(lr, e, si.X, si.y, [si.X[:64]], 0, "pre|a", threads=a.threads, max_epochs=2, patience=10 ** 9, realmlp_epochs=2)
            err = ""
        except Exception as ex_:                                          # noqa: BLE001
            npar, err = -1, repr(ex_)[:200]
    fs = time.time() - t1
    return dict(kind_="pre_a", status="ok" if not err else "partial", rows=[dict(part="a", learner=lr, trial=int(trial), kind="D", n_rows=int(len(si.y)),
                                                                                 epochs=2, fit_s=round(fs, 3), sec_per_epoch=round(fs / 2.0, 4),
                                                                                 n_params=int(npar), err=err, **res_block(t0))])


def run_pre_b(a, u):
    """(b) 시도 하나를 레나 x 의 캐나다 제외 fold 에서 전체 적합(seed 0, 종류 D)해 실제 epoch 수와 시간을 잰다."""
    t0 = time.time()
    _, lr, trial = u
    D = get_data(a)
    si = src_info(a, D, *PRE_TARGET)
    te = si.mac == PRE_FOLD_OUT
    tr = ~te
    t1, err, ep = time.time(), "", -1
    try:
        if int(trial) == 0:
            F = H.Fitter(ha_n(a, lr, "0"))
            if lr == "realmlp":
                fit_default(F, lr, si.X[tr], si.y[tr], [si.X[te]], 0, dict(method="D0", n=0, draw=0, alpha="1", placement="cell", sel=np.zeros(0, int)),
                            axis="pre")
                ep = int(a.realmlp_epochs)
            else:
                with count_epochs() as box:
                    fit_default(F, lr, si.X[tr], si.y[tr], [si.X[te]], 0, dict(method="D0", n=0, draw=0, alpha="1", placement="cell",
                                                                             sel=np.zeros(0, int)), axis="pre")
                ep = int(box[-1]) if box else -1
            err = F.errors[-1] if F.errors else ""
        else:
            _, ep, _ = fit_tuned(lr, cfg_entry(lr, "D", trial), si.X[tr], si.y[tr], [si.X[te]], 0, f"pre|b|{PRE_FOLD_OUT}", **tuned_kw(a, lr))
    except Exception as ex_:                                              # noqa: BLE001
        err = repr(ex_)[:200]
    fs = time.time() - t1
    mx = int(a.epochs) if int(trial) == 0 and lr != "realmlp" else int(a.MAX_EPOCHS[lr])
    return dict(kind_="pre_b", status="ok" if not err else "partial", rows=[dict(part="b", learner=lr, trial=int(trial), kind="D", n_rows=int(tr.sum()),
                                                                                 epochs_run=int(ep), max_epochs=mx, fit_s=round(fs, 3), err=err,
                                                                                 **res_block(t0))])


def run_pre_c(a, u):
    """(c) 기본판의 통과 0 공유 적합 1건(레나 x 분할 1–5, 종류 D, seed 0)과 통과 1 적합 2건(분할 1, n = 10 추출 0, 전량)의 시간."""
    t0 = time.time()
    _, lr = u
    D = get_data(a)
    HA = ha_n(a, lr, "0")
    t, m = PRE_TARGET
    splits = valid_splits(a, D, t, m, [1, 2, 3, 4, 5])
    cs = [H.build_ctx(D, HA, t, m, sp) for sp in splits]
    F = H.Fitter(HA)
    rows = []
    info = dict(method="D0", n=0, draw=0, alpha="1", placement="cell", sel=np.zeros(0, int))
    t1 = time.time()
    fit_default(F, lr, cs[0].X_src, cs[0].y_src, [c.XB for c in cs], 0, info, axis="pre")
    rows.append(dict(part="c", learner=lr, trial=0, kind="D", what="p0_shared", n_rows=int(len(cs[0].y_src)), n_eval=int(sum(len(c.yB) for c in cs)),
                     fit_s=round(time.time() - t1, 3), err=F.errors[-1] if F.errors else ""))
    c = cs[0]
    nA = len(c.yA)
    for n in (10, -1):
        sel = H.draw_cells(t, m, c.split, n, 0, nA)
        t1 = time.time()
        fit_default(F, lr, np.vstack([c.X_src, c.XA[sel]]), np.concatenate([c.y_src, c.yA[sel]]), [c.XB], 0, dict(info, n=n, sel=sel), axis="pre")
        rows.append(dict(part="c", learner=lr, trial=0, kind="D", what=f"p1_n{'all' if n == -1 else n}", n_rows=int(len(c.y_src) + len(sel)),
                         n_eval=int(len(c.yB)), fit_s=round(time.time() - t1, 3), err=F.errors[-1] if F.errors else ""))
    rb = res_block(t0)
    for r_ in rows:
        r_.update(rb)
    return dict(kind_="pre_c", status="ok", rows=rows)


def run_pre_d(a, u):
    """(d) catboost_tuned_loc 의 레나 x 선택 전체(9 설정 × 종류 2 × fold, 스레드 --threads)와 9 설정의 원천 전체 적합 1건씩의 CPU 시간.
    선택 점수는 쓰지 않는다(산출 제한)."""
    t0 = time.time()
    D = get_data(a)
    X = _load_h42()
    t, m = PRE_TARGET
    rows = []
    t1 = time.time()
    sel = cb_select_rows(a, D, t, m, hide=True)
    for r_ in sel:
        rows.append(dict(part="d", learner="catboost_tuned_loc", what="select", kind=r_["kind"], depth=r_["depth"], iterations=r_["iterations"],
                         n_fit=r_["n_fit"], fit_s=r_["sec"]))
    sel_s = time.time() - t1
    si = src_info(a, D, t, m)
    for dp, it in X.TUNE_GRID:
        t1 = time.time()
        _cb_tuned_fit(dict(iterations=it, depth=dp, learning_rate=X.TUNE_LR), si.X, si.y, si.X[:64], 0, a.threads)
        rows.append(dict(part="d", learner="catboost_tuned_loc", what="full_fit", kind="D", depth=int(dp), iterations=int(it), n_fit=1,
                         fit_s=round(time.time() - t1, 3)))
    rb = res_block(t0)
    for r_ in rows:
        r_.update(rb)
    return dict(kind_="pre_d", status="ok", rows=rows, select_s=round(sel_s, 1))


# ================================================================ 창 파일(lgf_window.json, 명세 6)
def _parse_time(v):
    """창 파일의 시각(ISO 문자열 또는 epoch 초) → epoch 초. 읽을 수 없으면 None."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        from datetime import datetime
        return float(datetime.fromisoformat(str(v)).timestamp())
    except (TypeError, ValueError):
        return None


def _iso(ts):
    from datetime import datetime
    return datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="seconds")


def read_window(a):
    try:
        return json.loads(Path(a.WINDOW).read_text())
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def window_lock(a, timeout=30.0):
    """창 파일 갱신용 잠금(O_EXCL). 죽은 PID 의 잠금은 지운다."""
    p = Path(str(a.WINDOW) + ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + float(timeout)
    while True:
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid = int(p.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and not T43.pid_alive(pid):
                p.unlink(missing_ok=True)
                continue
            if time.time() > t_end:
                raise SystemExit(f"[거부] 창 파일 잠금 {p} 을 얻지 못했다")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            if p.exists() and p.read_text().strip() == str(os.getpid()):
                p.unlink()
        except OSError:
            pass


def window_init_t0(a):
    """본 실행의 첫 단위를 배정하기 직전: 창 파일에 t0 가 없으면 t0 = 지금, deadline = t0 + 48 h 를 쓴다(있으면 읽기만 한다)."""
    with window_lock(a):
        w = read_window(a)
        if _parse_time(w.get("t0")) is None:
            now = time.time()
            w.update(t0=_iso(now), deadline=_iso(now + WINDOW_H * 3600.0), window_h=WINDOW_H)
            for k, v in (("reduce", []), ("restored", []), ("gpu_events", []), ("decisions", [])):
                w.setdefault(k, v)
            w["decisions"] = list(w.get("decisions") or []) + [dict(time=_iso(now), what=f"t0 설정(h48 tag {a.TAG} 의 첫 본 단위)", viewing_state="")]
            _write_json(a.WINDOW, w, strict=True)
            print(f"[window] t0 = {w['t0']} · deadline = {w['deadline']}", flush=True)
    return read_window(a)


def window_closed(a):
    dl = _parse_time(read_window(a).get("deadline"))
    return dl is not None and time.time() > dl


def window_settings(w):
    """창 파일의 사전 점검 결정(E1, 유효 축소 = reduce − restored 가운데 N 단계 S1, S5, S6, S7). 결정(B, E1, reduce)이 없으면 None."""
    if not w or "E1" not in w or "reduce" not in w or w.get("B") is None:
        return None
    res = {str(s_) for s_ in (w.get("restored") or [])}
    eff = [str(s_) for s_ in (w.get("reduce") or []) if str(s_) not in res]
    return dict(E1=bool(w.get("E1")), reduce_n=sorted(s_ for s_ in eff if s_ in N_STEPS))


def _given(argv, flag):
    """명령행에 flag 를 직접 주었는지(기본값과 구분한다)."""
    return any(str(v) == flag or str(v).startswith(flag + "=") for v in (argv or []))


def adopt_window(a, w=None):
    """집계(--summarize-only)와 CPU 작업(--cpu-only)의 설정: 창 파일의 E1 과 유효 축소(N 단계)로 k(E1, S7), 통과 1 의 추출(S5)·분할(S6),
    2순위 대상(S1)을 다시 정한다(finalize). 명령행에 --e1 이나 --reduce 를 직접 주었는데 창 파일과 다르면 거부한다. 창 파일에 결정이 없으면
    명령행 값을 그대로 쓴다. sel_sig(k 포함)와 조각 설정 해시가 본 실행과 같아지도록 하기 위해서다. 반환 설정 출처 문자열."""
    w = read_window(a) if w is None else w
    ws = window_settings(w)
    if ws is None:
        a.SETTINGS_SRC = "명령행(창 파일에 사전 점검 결정 없음)"
        return a.SETTINGS_SRC
    if _given(a.ARGV, "--e1") and bool(a.E1) != ws["E1"]:
        raise SystemExit(f"[거부] --e1 {int(a.E1)} 이 창 파일의 E1 {int(ws['E1'])} 과 다르다. 집계와 CPU 작업은 창 파일의 값을 쓴다(--e1 을 빼거나 같게 준다)")
    if _given(a.ARGV, "--reduce") and sorted(a.REDUCE_N) != ws["reduce_n"]:
        raise SystemExit(f"[거부] --reduce 의 N 단계 {sorted(a.REDUCE_N)} 가 창 파일의 유효 축소 {ws['reduce_n']} 와 다르다. 집계와 CPU 작업은 "
                         "창 파일의 값을 쓴다(--reduce 를 빼거나 같게 준다)")
    if bool(a.E1) != ws["E1"] or sorted(a.REDUCE_N) != ws["reduce_n"]:
        keep = dict(CODE_SHA=a.CODE_SHA, REVISION_NEEDED=a.REVISION_NEEDED, RUN_ID=a.RUN_ID)
        a.e1 = int(ws["E1"])
        a.reduce = ",".join(ws["reduce_n"])
        finalize(a)
        for k_, v_ in keep.items():
            setattr(a, k_, v_)
    a.SETTINGS_SRC = "창 파일"
    return a.SETTINGS_SRC


def check_window_args(a):
    """본 실행: 창 파일에 사전 점검 개정의 결정(B, E1, reduce)이 있어야 한다(h47 과 같은 규칙). GPU 작업은 --reduce 의 N 단계와 --e1 이
    창 파일(restored 를 뺀 N 단계, E1)과 같아야 한다. CPU 작업은 E1 과 무관하므로 창 파일의 값을 받아 쓴다(adopt_window)."""
    w = read_window(a)
    ws = window_settings(w)
    if ws is None:
        raise SystemExit(f"[거부] 창 파일 {a.WINDOW} 에 사전 점검 개정의 결정(B, E1, reduce)이 없다. h47 --window-set 을 먼저 실행한다")
    if a.cpu_only:
        adopt_window(a, w)
        return w
    if ws["reduce_n"] != sorted(a.REDUCE_N):
        raise SystemExit(f"[거부] --reduce 의 N 단계 {sorted(a.REDUCE_N)} 가 창 파일 {ws['reduce_n']} 과 다르다")
    if ws["E1"] != bool(a.E1):
        raise SystemExit(f"[거부] --e1 {int(a.E1)} 이 창 파일의 E1 {w.get('E1')} 과 다르다")
    return w


def drain_path(a):
    return Path(a.RUN_DIR) / "drain"


# ================================================================ GPU 거부·선별
def check_run_args(a):
    """학습 실행의 거부 조건(자료를 읽기 전). --allow-local 필수, 스레드 1–2, GPU 번호 규칙."""
    if not a.allow_local:
        raise SystemExit("[거부] 학습 실행(스모크, 사전 점검, 본 실행)은 --allow-local 이 필요하다")
    if not (1 <= int(a.threads_asked) <= THREADS_MAX):
        raise SystemExit(f"[거부] --threads {a.threads_asked}: 학습 실행은 프로세스당 스레드 1–{THREADS_MAX}개만 허용한다")
    if a.cpu_only:
        if a.GPUS:
            raise SystemExit("[거부] --cpu-only 에서는 --gpus 를 주지 않는다")
        return
    if not a.GPUS:
        raise SystemExit("[거부] --gpus 가 비어 있다(CPU 작업은 --cpu-only)")
    bad = [g for g in a.GPUS if g in NEVER]
    if bad:
        raise SystemExit(f"[거부] GPU {bad} 는 쓰지 않는다(다른 사용자)")
    b01 = [g for g in a.GPUS if g in GPU01]
    if b01 and not a.allow_gpus_01:
        raise SystemExit(f"[거부] GPU {b01} 는 --allow-gpus-01 과 개정 기록이 있을 때만 쓴다")
    out = [g for g in a.GPUS if g not in set(ALLOWED_GPUS) | set(GPU01)]
    if out:
        raise SystemExit(f"[거부] 허용 후보 {list(ALLOWED_GPUS)} 밖의 GPU {out}")


def lgt_alive(a):
    try:
        pid = int(json.loads(Path(a.LGT_LOCK).read_text()).get("pid", 0) or 0)
    except (OSError, ValueError):
        return 0
    return pid if pid and T43.pid_alive(pid) else 0


def screen_gpus_n(a, info, apps, cand=None):
    """nvidia-smi 선별(h43.screen_gpus: 메모리 ≤ --gpu-mem-max-mib, 계산 프로세스 없음)과 LGT 잠금 이중 확인(6·7·9). 예약은 GPU 8 과
    (허용 표지가 없으면) GPU 0·1 이다(h43 의 기본 예약 0–4 를 쓰지 않는다)."""
    reserved = tuple(NEVER) + (() if a.allow_gpus_01 else tuple(GPU01))
    ok, dropped = T43.screen_gpus(list(a.GPUS if cand is None else cand), {g: v["used"] for g, v in info.items()}, a.gpu_mem_max_mib,
                                  reserved=reserved, apps=T43.gpu_apps_by_index(info, apps))
    pid = lgt_alive(a)
    if pid:
        for g in [g for g in ok if g in LGT_GPUS]:
            ok.remove(g)
            dropped.append((g, f"LGT 잠금의 PID {pid} 가 살아 있다"))
    return ok, dropped


def rescreen_n(a, cand, excluded=()):
    """풀을 다시 만들 때의 GPU 재확인(h47.rescreen 과 같은 방식). screen_gpus_n(예약 = GPU 8 과 허용 표지 없는 GPU 0·1, LGT 잠금 확인)을
    h43.GPU_SETTLE_S 동안 되풀이해 종료한 워커의 메모리가 풀리기를 기다린다. nvidia-smi 오류와 거부(SystemExit)는 잡아 빈 목록을 돌려준다
    (그 GPU 만 빠지고 실행 전체는 멈추지 않는다)."""
    cand = [int(g) for g in cand if int(g) not in {int(x) for x in excluded}]
    if not cand:
        return []
    t_end = time.time() + float(T43.GPU_SETTLE_S)
    while True:
        try:
            info, apps = T43.query_gpu_info(), T43.query_gpu_apps()
            ok, dropped = screen_gpus_n(a, info, apps, cand)
        except SystemExit as e:
            print(f"[pool] GPU {cand} 재확인 거부({str(e)[:160]}). 풀을 다시 만들지 않는다", flush=True)
            return []
        except Exception as e:                                            # noqa: BLE001
            print(f"[pool] nvidia-smi 확인 실패({repr(e)[:160]}). 풀을 다시 만들지 않는다", flush=True)
            return []
        if not dropped or time.time() >= t_end:
            break
        time.sleep(3.0)
    for g_, why in dropped:
        print(f"[pool] GPU {g_} 를 뺀다: {why}", flush=True)
    return ok


# ================================================================ 작업 계획(의존성과 우선순위)
def t_rank(a, t, m):
    order = list(a.MAIN_T) + [p for p in a.T2 if p not in a.MAIN_T]
    return order.index((t, m)) if (t, m) in order else len(order)


class Planner:
    """작업 단위의 상태(pending, running, done, failed, quar, missing)와 의존성. 선택 묶음(학습기, 종류, 대상, 모드)은 1단계가 끝나면
    2단계 단위를 만들고, 2단계가 끝나면 선택을 계산해 선택 표에 쓴다. 통과 0·1 단위는 그 학습기·대상의 D·R 선택이 있을 때 배정할 수 있다."""

    def __init__(self, a, D, fresh=False):
        self.a, self.D, self.fresh = a, D, bool(fresh)
        self.state: dict = {}
        self.units: dict = {}
        self.prio: dict = {}
        self.groups: dict = {}
        self.dec: dict = {}
        self.cb_ready: set = set()
        self.prev_done: list = []
        self.errors: dict = {}
        self.active: dict = {}

    # ---------------------------------------------------------------- 등록
    def _done_on_disk(self, u):
        a, D = self.a, self.D
        typ = u[0]
        if typ == "sel":
            return sel_done(a, D, u)
        if typ == "p0":
            _, lr, t, m = u
            sps = valid_splits(a, D, t, m, a.SPLITS)
            return bool(sps) and all(shard_state(a, "p0", lr, t, m, sp)[0] for sp in sps)
        if typ == "p1":
            _, lr, t, m, sp = u
            return shard_state(a, "p1", lr, t, m, sp)[0]
        if typ in ("cbl", "cbt"):
            _, t, m, sp = u
            return shard_state(a, "cpu", "catboost_lo" if typ == "cbl" else "catboost_tuned_loc", t, m, sp)[0]
        if typ == "cbsel":
            return bool(load_cb_tuned(a, u[1], u[2])) and all(k in load_cb_tuned(a, u[1], u[2]) for k in KINDS)
        return False

    def add(self, u, prio, active=True):
        nm = unit_name(u)
        if nm in self.state:
            return
        self.units[nm], self.prio[nm] = u, tuple(int(v) for v in prio)
        self.active[nm] = bool(active)
        on_disk = self._done_on_disk(u) if (not self.fresh or not active) else False
        if not active:
            self.state[nm] = "done" if on_disk else "missing"
            return
        if on_disk and self.a.resume:
            self.state[nm] = "done"
        else:
            if self.fresh and self._done_on_disk(u):
                self.prev_done.append(nm)
            elif not self.fresh and on_disk:
                self.prev_done.append(nm)
            self.state[nm] = "pending"

    def add_group(self, lr, kind, t, m, jr, sub=0, active=True):
        key = (lr, kind, t, m)
        if key in self.groups:
            return
        base = (jr, sub, LEARNERS_ALL.index(lr), t_rank(self.a, t, m), KINDS.index(kind))
        self.groups[key] = dict(base=base, active=active, stage2=None, decided=False)
        for u in stage1_units(self.a, lr, kind, t, m):
            self.add(u, base + (1, u[6], u[7]), active=active)

    # ---------------------------------------------------------------- 진행
    def advance_all(self):
        for key in list(self.groups):
            self.advance(key)

    def advance(self, key):
        a = self.a
        g = self.groups[key]
        lr, kind, t, m = key
        if g["decided"]:
            return
        s1 = stage1_units(a, lr, kind, t, m)
        if not all(self.state.get(unit_name(u)) == "done" for u in s1):
            if not g["active"]:
                self._from_table(key)
            return
        if g["stage2"] is None:
            recs1 = {int(u[6]): load_sel(a, u) for u in s1}
            if any(v is None for v in recs1.values()):
                self.errors[str(key)] = "1단계 JSON 을 읽을 수 없다"
                return
            trials2 = stage2_trials(a, lr, recs1)
            g["stage2"] = stage2_units(a, lr, kind, t, m, trials2)
            g["trials2"] = trials2
            for u in g["stage2"]:
                self.add(u, g["base"] + (2, u[6], u[7]), active=g["active"])
        if not all(self.state.get(unit_name(u)) == "done" for u in g["stage2"]):
            if not g["active"]:
                self._from_table(key)
            return
        recs1 = [load_sel(a, u) for u in s1]
        recs2 = {(int(u[6]), int(u[7])): load_sel(a, u) for u in g["stage2"]}
        if any(v is None for v in recs2.values()):
            self.errors[str(key)] = "2단계 JSON 을 읽을 수 없다"
            return
        dec = decide_group(a, lr, kind, recs2, g["trials2"])
        rows = fit_rows_of(a, lr, kind, t, m, recs1 + list(recs2.values())) + dec_rows_of(a, lr, kind, t, m, dec, g["trials2"])
        write_select(a, lr, kind, t, m, rows)
        g["decided"] = True
        self.dec[key] = dec
        self._recheck_passes(lr, t, m)
        print(f"  [select] {lr}|{kind}|{t}|{m}: 2단계 후보 {g['trials2']} → 선택 시도 {dec['main']['trial']}"
              f"{'(' + dec['main']['flag'] + ')' if dec['main'].get('flag') else ''}", flush=True)

    def _recheck_passes(self, lr, t, m):
        """선택 표를 다시 쓴 뒤: 이 (학습기, 대상, 모드)의 통과 0·1 단위 가운데 완료로 본 것을 다시 확인한다. 조각 설정에 선택 결과가 들어가므로
        선택이 바뀌었으면 미완료가 되어 다시 돈다."""
        for nm, u in self.units.items():
            if u[0] in ("p0", "p1") and (u[1], u[2], u[3]) == (lr, t, m) and self.state.get(nm) == "done" and self.active.get(nm, True):
                if not self._done_on_disk(u):
                    self.state[nm] = "pending"
                    print(f"  [resume] 선택이 바뀌어 다시 실행한다: {nm}", flush=True)

    def _from_table(self, key):
        """요청하지 않은 선택 묶음: JSON 이 모자라면 선택 표의 행(같은 sel_sig)을 쓴다."""
        d = decisions_table(self.a).get(key)
        if d and "main" in d:
            self.groups[key]["decided"] = True
            self.dec[key] = d

    def decided(self, lr, t, m):
        return all(self.groups.get((lr, k, t, m), {}).get("decided") for k in KINDS)

    def ready(self, u):
        typ = u[0]
        if typ in ("p0", "p1"):
            lr, t, m = u[1], u[2], u[3]
            return self.decided(lr, t, m)
        if typ == "cbt":
            return self.state.get(unit_name(("cbsel", u[1], u[2]))) == "done"
        return True

    def next_ready(self):
        cand = sorted((self.prio[nm], nm) for nm, s_ in self.state.items() if s_ == "pending")
        for _, nm in cand:
            if self.ready(self.units[nm]):
                return self.units[nm]
        return None

    def mark(self, u, s_):
        self.state[unit_name(u)] = s_

    def on_done(self, u):
        self.state[unit_name(u)] = "done"
        if u[0] == "sel":
            self.advance((u[1], u[2], u[3], u[4]))

    def counts(self):
        return dict(Counter(self.state.values()))

    def blocked(self):
        return [nm for nm, s_ in self.state.items() if s_ == "pending"]


def build_planner(a, D, fresh):
    """--jobs 와 축소 단계에서 작업 단위를 만든다. 요청하지 않은 선택 묶음은 통과 0·1 의 의존성으로만 등록한다(active = False)."""
    P = Planner(a, D, fresh=fresh)
    if a.cpu_only:
        cbt_on = bool(read_window(a).get("cbt_enabled")) if a.MAIN else False
        for t, m in list(a.MAIN_T) + list(a.T2):
            tr = t_rank(a, t, m)
            for sp in valid_splits(a, D, t, m, a.SPLITS):
                P.add(("cbl", t, m, int(sp)), (0, tr, int(sp), 0, 0, 0, 0, 0))
            if cbt_on:
                P.add(("cbsel", t, m), (1, tr, 0, 0, 0, 0, 0, 0))
                for sp in valid_splits(a, D, t, m, a.SPLITS):
                    P.add(("cbt", t, m, int(sp)), (2, tr, int(sp), 0, 0, 0, 0, 0))
        if not cbt_on:
            print("[plan] 창 파일의 cbt_enabled 가 참이 아니다: catboost_tuned_loc(선택과 적합)을 돌리지 않는다", flush=True)
        return P
    jobs = set(a.JOBS)
    main_t = list(a.MAIN_T)
    for lr in a.LEARNERS:
        fast = lr in FAST
        js, jp = ("n_sel_fast", "n_p0_fast") if fast else ("n_sel_slow", "n_p0_slow")
        for t, m in main_t:
            for kind in KINDS:
                P.add_group(lr, kind, t, m, JOB_RANK[js], 0, active=js in jobs)
            if jp in jobs:
                P.add(("p0", lr, t, m), (JOB_RANK[jp], 0, LEARNERS_ALL.index(lr), t_rank(a, t, m), 0, 0, 0, 0))
            if "n_p1" in jobs:
                for sp in valid_splits(a, D, t, m, a.P1_SPLITS):
                    P.add(("p1", lr, t, m, int(sp)), (JOB_RANK["n_p1"], 0, LEARNERS_ALL.index(lr), int(sp), t_rank(a, t, m), 0, 0, 0))
        for t, m in a.T2:
            for kind in KINDS:
                P.add_group(lr, kind, t, m, JOB_RANK["n_t2"], 0, active="n_t2" in jobs)
            if "n_t2" in jobs:
                P.add(("p0", lr, t, m), (JOB_RANK["n_t2"], 1, LEARNERS_ALL.index(lr), t_rank(a, t, m), 0, 0, 0, 0))
                for sp in valid_splits(a, D, t, m, a.P1_SPLITS):
                    P.add(("p1", lr, t, m, int(sp)), (JOB_RANK["n_t2"], 2, LEARNERS_ALL.index(lr), int(sp), t_rank(a, t, m), 0, 0, 0))
    if a.smoke:                                                           # 스모크: 정적 점검 단위를 맨 앞에 둔다
        P.add(("chk",), (-1, 0, 0, 0, 0, 0, 0, 0))
    P.advance_all()
    return P


class StaticPlanner(Planner):
    """사전 점검: 의존성 없는 단위 목록."""

    def __init__(self, a, D, units):
        super().__init__(a, D, fresh=True)
        for i, u in enumerate(units):
            nm = unit_name(u)
            self.units[nm], self.prio[nm], self.state[nm] = u, (i,), "pending"


def precheck_units(a):
    """(e) 정적 점검, (a) 모든 설정(E1 목록 포함)의 2 epoch 시간, (b) 학습기마다 시도 0 과 무작위 2개(FT-T 는 2, 3)의 실제 epoch 수,
    (c) 기본판 통과 0 공유 1건과 통과 1 적합 2건, (d) catboost_tuned_loc 의 선택과 원천 적합."""
    units = [("chk",)]
    for lr in a.LEARNERS:
        units += [("pre_b", lr, int(tr)) for tr in ((0, 2, 3) if lr == "ftt" else (0, 1, 2))]
    for lr in a.LEARNERS:
        units.append(("pre_c", lr))
    for lr in a.LEARNERS:
        units += [("pre_a", lr, int(tr)) for tr in n_trials_list(lr, K_LIST_MAX)]
    units.append(("pre_d",))
    return units


# ================================================================ 실행기
def _log_unit(res, n_done, n_all, t0):
    """단위 완료 기록(결과 값은 내지 않는다: 시간, 적합 수, 상태, 메모리)."""
    print(f"  [{res.get('unit', '')}] 상태 {res.get('status', '')} · 적합 {res.get('n_fit', '')} · 실패 {res.get('n_fail', 0)} · 적합 시간 "
          f"{res.get('fit_s', '')}s · 벽시계 {res.get('wall_s', '')}s · GPU '{res.get('gpu', '')}' 메모리 최댓값 {res.get('gpu_mem_peak_mib')} MiB · "
          f"RSS {res.get('rss_max_mib')} MiB · 완료 {n_done}/{n_all} · 누적 {time.time() - t0:.0f}s", flush=True)


def run_gpu(a, P, gpus, argv, t0):
    """GPU 하나에 풀 하나(워커 하나). 배정 전마다 드레인·창·load 를 확인한다. 반환 결과 dict."""
    ctx = multiprocessing.get_context("spawn")
    exs, running = {}, {}
    crashes, stall = Counter(), Counter()
    busy, active = set(), list(gpus)
    results, failed, quar = [], [], []
    drained = closed = aborted = interrupted = ""
    last_load, hold, first = 0.0, False, True
    n_all = sum(1 for s_ in P.state.values() if s_ == "pending")

    def make(g):
        return ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_worker_init,
                                   initargs=(argv, int(g), int(a.threads), str(a.RUN_DIR), os.getpid(), a.RUN_ID))

    def drop(g, why):
        if g in active:
            active.remove(g)
        ex = exs.pop(g, None)
        if ex is not None:
            T43.kill_pool(ex)
        print(f"[pool] GPU {g} 를 뺀다: {why}", flush=True)

    def crash(g, why):
        """GPU g 의 풀이 깨졌다(워커 비정상 종료). 풀을 끄고, 재확인(rescreen_n)을 통과하면 다시 만들고 아니면 그 GPU 를 뺀다.
        진전 없이 --pool-retries 를 넘게 깨지면 뺀다. 다른 GPU 의 풀은 계속 돈다."""
        stall[g] += 1
        ex = exs.pop(g, None)
        if ex is not None:
            T43.kill_pool(ex)
        ok = rescreen_n(a, [g], busy) if (g in active and stall[g] <= int(a.pool_retries)) else []
        if ok:
            exs[g] = make(g)
            print(f"[pool] GPU {g}: {why}. 풀을 다시 만든다(진전 없는 연속 {stall[g]}회)", flush=True)
        else:
            drop(g, f"{why}. 재확인 실패 또는 진전 없는 재생성 {stall[g]}회")

    for g in active:
        exs[g] = make(g)
    handlers = T43.install_stop_handlers()
    try:
        while True:
            now = time.time()
            if not drained and drain_path(a).exists():
                drained = "drain"
                print(f"[drain] {drain_path(a)} 가 있다. 새 단위를 배정하지 않고 진행 중 단위가 끝나기를 기다린다", flush=True)
            if a.MAIN and not closed and window_closed(a):
                closed = "window"
                print("[window] 창 마감이 지났다. 새 단위를 배정하지 않는다", flush=True)
            if now - last_load >= LOAD_CHECK_S:
                last_load = now
                la = os.getloadavg()[0]
                h_ = la > float(a.load_max_run)
                if h_ != hold:
                    print(f"[load] 1분 load average {la:.1f} {'>' if h_ else '≤'} {a.load_max_run}: 새 단위 배정 {'보류' if h_ else '재개'}", flush=True)
                hold = h_
            if not (drained or closed or hold or aborted):
                for g in list(active):
                    if any(gg == g for gg, _ in running.values()):
                        continue
                    u = P.next_ready()
                    if u is None:
                        break
                    if a.MAIN and first:
                        window_init_t0(a)
                        first = False
                        if window_closed(a):
                            closed = "window"
                            break
                    P.mark(u, "running")
                    try:
                        fut = exs[g].submit(_worker_run, u)
                    except BrokenProcessPool:                             # 대기 중인 워커가 죽었다(예: 호스트 메모리 부족 종료): 단위를 되돌린다
                        P.mark(u, "pending")
                        crash(g, "대기 중 워커가 비정상 종료했다(제출 실패)")
                        continue
                    running[fut] = (g, u)
            if not running:
                if drained or closed or aborted:
                    break
                if hold:
                    time.sleep(30.0)
                    continue
                if not active:
                    print("[pool] 쓸 수 있는 GPU 가 없다. 남은 단위를 배정하지 않는다", flush=True)
                    break
                if P.next_ready() is None:
                    break
                continue
            done_set, _ = wait(list(running), timeout=30.0, return_when=FIRST_COMPLETED)
            for fut in done_set:
                g, u = running.pop(fut)
                nm = unit_name(u)
                try:
                    res = fut.result()
                    P.on_done(u)
                    stall[g] = 0
                    results.append(res)
                    _log_unit(res, sum(1 for s_ in P.state.values() if s_ == "done"), n_all, t0)
                except BrokenProcessPool:
                    if P._done_on_disk(u) and _done_this_run(a, u):
                        P.on_done(u)
                    else:
                        crashes[nm] += 1
                        if crashes[nm] >= POOL_CRASH_MAX:
                            P.mark(u, "quar"); quar.append(nm)
                            failed.append((nm, f"실행 중에 풀이 {crashes[nm]}회 깨졌다(격리)"))
                        else:
                            P.mark(u, "pending")
                    crash(g, f"워커가 비정상 종료했다({nm})")
                except Exception as e:                                    # noqa: BLE001
                    msg = str(e)
                    if TAG_BUSY in msg:
                        P.mark(u, "pending")
                        busy.add(g)
                        drop(g, msg[:200])
                    elif T43.TAG_MISMATCH in msg:
                        aborted = msg
                        P.mark(u, "failed"); failed.append((nm, msg[:300]))
                    else:
                        P.mark(u, "failed"); failed.append((nm, msg[:300]))
                        print(f"  [FAIL] {nm}: {msg[:300]}", flush=True)
            if aborted:
                print(f"[중단] GPU 장치 대응 불일치: {aborted[:200]}. 실행 전체를 멈춘다", flush=True)
                for g in list(exs):
                    T43.kill_pool(exs.pop(g))
                break
    except KeyboardInterrupt as e:
        interrupted = str(e) or "KeyboardInterrupt"
        for g in list(exs):
            T43.kill_pool(exs.pop(g))
        print(f"[중단] {interrupted}. 워커를 종료했다. 이어 돌리려면 --resume 을 준다", flush=True)
    finally:
        T43.restore_handlers(handlers)
        for ex in exs.values():
            try:
                ex.shutdown(wait=True)
            except Exception:                                             # noqa: BLE001
                pass
    if drained and drain_path(a).exists():
        drain_path(a).unlink(missing_ok=True)
    return dict(results=results, failed=failed, quarantined=quar, drained=drained, closed=closed, aborted=aborted, interrupted=interrupted,
                busy_gpus=sorted(busy), gpus_last=list(active), gpu_lost=bool(not active))


def _done_this_run(a, u):
    typ = u[0]
    if typ == "sel":
        return sel_done(a, get_data(a), u, run_id=a.RUN_ID)
    if typ == "p1":
        return shard_state(a, "p1", u[1], u[2], u[3], u[4], run_id=a.RUN_ID)[0]
    if typ == "p0":
        sps = valid_splits(a, get_data(a), u[2], u[3], a.SPLITS)
        return all(shard_state(a, "p0", u[1], u[2], u[3], sp, run_id=a.RUN_ID)[0] for sp in sps)
    return False


def run_cpu(a, P, t0):
    """CPU 작업(이 프로세스에서 차례로, 스레드 --threads, GPU 없음). 단위 사이에 드레인·창·load 를 확인한다."""
    results, failed = [], []
    drained = closed = interrupted = ""
    last_load, first = 0.0, True
    n_all = sum(1 for s_ in P.state.values() if s_ == "pending")
    handlers = T43.install_stop_handlers()
    try:
        while True:
            if drain_path(a).exists():
                drained = "drain"; break
            if a.MAIN and window_closed(a):
                closed = "window"; break
            if time.time() - last_load >= LOAD_CHECK_S:
                last_load = time.time()
                la = os.getloadavg()[0]
                if la > float(a.load_max_run):
                    print(f"[load] 1분 load average {la:.1f} > {a.load_max_run}: 새 단위 배정 보류(300 s)", flush=True)
                    time.sleep(LOAD_CHECK_S)
                    continue
            u = P.next_ready()
            if u is None:
                break
            if a.MAIN and first:
                window_init_t0(a)
                first = False
                if window_closed(a):
                    closed = "window"; break
            P.mark(u, "running")
            t1 = time.time()
            try:
                res = dict(run_unit(a, u))
                res.update(unit=unit_name(u), wall_s=round(time.time() - t1, 1), gpu="")
                P.on_done(u)
                results.append(res)
                _log_unit(res, sum(1 for s_ in P.state.values() if s_ == "done"), n_all, t0)
            except (ImportError, MemoryError) as e:
                P.mark(u, "failed"); failed.append((unit_name(u), repr(e)[:300]))
                print(f"  [FAIL] {unit_name(u)}: {repr(e)[:300]}", flush=True)
            except Exception as e:                                        # noqa: BLE001
                P.mark(u, "failed"); failed.append((unit_name(u), repr(e)[:300]))
                print(f"  [FAIL] {unit_name(u)}: {repr(e)[:300]}", flush=True)
    except KeyboardInterrupt as e:
        interrupted = str(e) or "KeyboardInterrupt"
        print(f"[중단] {interrupted}", flush=True)
    finally:
        T43.restore_handlers(handlers)
    if drained and drain_path(a).exists():
        drain_path(a).unlink(missing_ok=True)
    return dict(results=results, failed=failed, quarantined=[], drained=drained, closed=closed, aborted="", interrupted=interrupted,
                busy_gpus=[], gpus_last=[])


def exit_code_n(res):
    """종료 코드: 중단 130 > 실패 1 > 드레인 3 > 창 마감 4 > partial 2 > 0."""
    if res.get("interrupted"):
        return EXIT_INTERRUPT
    if res.get("n_fail") or res.get("refused"):
        return EXIT_FAIL
    if res.get("drained"):
        return EXIT_DRAIN
    if res.get("closed"):
        return EXIT_WINDOW
    if res.get("n_partial"):
        return EXIT_PARTIAL
    return 0


# ================================================================ 적합 수(--count-only)
def count_only(a):
    """학습 없이 fold 구성, 설정 목록(lgfn_configs.json, cfg_sig), 작업별 적합 수, 낙관·비관 GPU-h 를 낸다(lgfn_count.csv, _meta.json)."""
    t0 = time.time()
    cf, sig = ensure_configs(a, write=True)
    D = get_data(a)
    folds, rows = [], []
    fold_of = {}
    for t, m in list(a.MAIN_T) + list(a.T2):
        si = src_info(a, D, t, m)
        fr = fold_rows(si)
        fold_of[(t, m)] = (si, fr)
        for f in fr:
            folds.append(dict(target=t, mode=m, n_src=int(len(si.y)), src_hash=si.src_hash, n_regions_src=int(len(set(si.mac.tolist()))), **f))
    small = {}
    for t, m in list(a.MAIN_T) + list(a.T2):
        si = fold_of[(t, m)][0]
        small[f"{t}|{m}"] = {str(k): int(v) for k, v in sorted(Counter(si.mac.tolist()).items()) if v < int(a.min_region_cells)}
    for lr in a.LEARNERS:
        k = int(a.N_RANDOM[lr]); tr1 = len(trials_of(a, lr)); k2 = int(a.STAGE2_TOP[lr]) + 1
        for job, pairs in (("main", list(a.MAIN_T)), ("n_t2", list(a.T2))):
            for t, m in pairs:
                si, fr = fold_of[(t, m)]
                F = len(fr)
                ratio = float(np.mean([f["n_tr"] for f in fr]) / ROWS_REF) if fr else 0.0      # fold 학습 행 / 기준 행 수
                fscale = float(len(si.y)) / ROWS_REF                                               # 본 적합 학습 행 / 기준 행 수
                s1 = tr1 * 2 * F
                s2 = k2 * len(a.STAGE2_SEEDS) * 2 * F
                p0d, p0t, p0s = 2 * len(a.SEEDS), 2 * len(a.SEEDS), 3 * len(a.SEEDS)
                p1 = 0
                for sp in valid_splits(a, D, t, m, a.P1_SPLITS):
                    nA = int(D.split_structure(t)[sp]["n_A"])
                    p1 += len(H.cells_of([n for n in a.N_GRID if n != 0], int(a.DRAWS_P1), nA)) * len(a.SEEDS) * 2
                sel_h = [(s1 - 2 * F) * ratio * TIME_OPT[lr] + 2 * F * ratio * TIME_DEF[lr] + s2 * ratio * TIME_OPT[lr],
                         (s1 - 2 * F) * ratio * TIME_PES[lr] + 2 * F * ratio * TIME_DEF[lr] + s2 * ratio * TIME_PES[lr]]
                fit_h = [fscale * (p0d * TIME_DEF[lr] + (p0t + p0s) * TIME_OPT[lr] + p1 * (TIME_DEF[lr] + TIME_OPT[lr])),
                         fscale * (p0d * TIME_DEF[lr] + (p0t + p0s) * TIME_PES[lr] + p1 * (TIME_DEF[lr] + TIME_PES[lr]))]
                rows.append(dict(learner=lr, scope=job, target=t, mode=m, k=k, n_trials_stage1=tr1, n_folds=F, fold_row_ratio=round(ratio, 3),
                                 fit_stage1=s1, fit_stage2_max=s2, fit_p0_default=p0d, fit_p0_tuned_max=p0t, fit_p0_sens_max=p0s,
                                 fit_p1_default=p1, fit_p1_tuned_max=p1, est_gpu_h_opt=round((sel_h[0] + fit_h[0]) / 3600.0, 2),
                                 est_gpu_h_pes=round((sel_h[1] + fit_h[1]) / 3600.0, 2), est_sel_gpu_h_opt=round(sel_h[0] / 3600.0, 2),
                                 est_sel_gpu_h_pes=round(sel_h[1] / 3600.0, 2)))
    cb = []
    X = _load_h42()
    for t, m in list(a.MAIN_T) + list(a.T2):
        si, fr = fold_of[(t, m)]
        n_lo = 0
        for sp in valid_splits(a, D, t, m, a.SPLITS):
            nA = int(D.split_structure(t)[sp]["n_A"])
            n_lo += len(H.cells_of(list(a.N_GRID), int(a.DRAWS_P1), nA)) * len(a.SEEDS) * 2
        cb.append(dict(target=t, mode=m, fit_catboost_lo=n_lo, fit_cbt_select=len(X.TUNE_GRID) * 2 * len(fr), fit_catboost_tuned_loc=n_lo,
                       est_cpu_h_lo=round(n_lo * TIME_CB / 3600.0, 3)))
    df = pd.DataFrame(rows)
    a.OUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(df, a.OUT / f"{a.TAG}_count.csv")
    fd = pd.DataFrame(folds)
    print(f"[count-only] 대상 {len(a.MAIN_T) + len(a.T2)} · 학습기 {a.LEARNERS} · k {a.N_RANDOM} · 2단계 상위 {a.STAGE2_TOP} · seed(2단계) {a.STAGE2_SEEDS} · "
          f"축소 {a.REDUCE or '없음'} · E1 {int(a.E1)} · 설정 목록 해시 {sig[:12]}", flush=True)
    print("[count-only] fold 구성(제외 지역, 채점 셀 수, fold 학습 행 수, 제외 지역 100 km 안의 학습 행 수)\n"
          + fd[["target", "mode", "region", "n_te", "n_tr", "n_tr_within100km", "n_src"]].to_string(index=False), flush=True)
    print(f"[count-only] 30셀 미만 원천 지역(모든 fold 에서 학습 쪽): {json.dumps(small, ensure_ascii=False)}", flush=True)
    if len(df):
        agg = df.groupby(["learner", "scope"], as_index=False)[["fit_stage1", "fit_stage2_max", "fit_p0_default", "fit_p0_tuned_max", "fit_p0_sens_max",
                                                                 "fit_p1_default", "fit_p1_tuned_max", "est_gpu_h_opt", "est_gpu_h_pes"]].sum()
        print("[count-only] 학습기·범위별 적합 수와 추정 GPU-h(계획서 8.3 의 1건 시간 가정. 사전 점검 실측으로 바꾼다)\n" + agg.to_string(index=False), flush=True)
        tot = df.groupby("scope")[["est_gpu_h_opt", "est_gpu_h_pes"]].sum()
        print("[count-only] 범위별 합계 GPU-h(낙관·비관): " + " · ".join(f"{k} {v.est_gpu_h_opt:.1f}·{v.est_gpu_h_pes:.1f}" for k, v in tot.iterrows()), flush=True)
    cbd = pd.DataFrame(cb)
    print("[count-only] CPU 작업(catboost_lo, catboost_tuned_loc 의 선택과 적합) 적합 수\n" + cbd.to_string(index=False), flush=True)
    n_cfg = {lr: {kind: len(cf[lr][kind]) for kind in KINDS} for lr in LEARNERS_ALL}
    meta = dict(tag=a.TAG, cfg_sig=sig, configs=str(a.CONFIGS), n_configs=n_cfg, k=a.N_RANDOM, stage2_top=a.STAGE2_TOP, stage2_seeds=a.STAGE2_SEEDS,
                reduce=a.REDUCE, e1=bool(a.E1), jobs=a.JOBS, learners=a.LEARNERS, targets=[f"{t}|{m}" for t, m in a.MAIN_T],
                targets_t2=[f"{t}|{m}" for t, m in a.T2], folds=folds, small_regions=small, cpu=cb, sel_sig={lr: sel_sig(a, lr) for lr in LEARNERS_ALL},
                code_sha=a.CODE_SHA, revision_needed=a.REVISION_NEEDED, elapsed_s=round(time.time() - t0, 1),
                time_basis=dict(default=TIME_DEF, tuned_opt=TIME_OPT, tuned_pes=TIME_PES, catboost=TIME_CB,
                                note="계획서 8.3 의 가정값. 사전 점검(6.3 단계 4)의 실측으로 바꾼다"))
    _write_json(a.OUT / f"{a.TAG}_count_meta.json", meta, strict=True)
    print(f"[count-only] → {a.OUT / (a.TAG + '_count.csv')}, {a.OUT / (a.TAG + '_count_meta.json')}, {a.CONFIGS} · {time.time() - t0:.0f}s", flush=True)
    return df, meta


# ================================================================ 스모크와 사전 점검의 산출
def smoke_report(a, res):
    """스모크 점검 표(경로 점검 값과 일치 여부)와 시간 표. RMSE, Δ, 판정, 선택 점수는 내지 않는다."""
    chk = []
    for r_ in res.get("results", []):
        for q in r_.get("rows", []) if r_.get("kind_") == "chk" else []:
            chk.append(dict(q, source="chk"))
        sc = r_.get("smoke_check") or {}
        if sc:
            for k, v in sc.items():
                chk.append(dict(check=k, learner=str(r_.get("unit", "")).split("|")[1] if "|" in str(r_.get("unit", "")) else "", value=v,
                                ok=v if isinstance(v, bool) else None, source=str(r_.get("unit", ""))))
    tim = []
    for r_ in res.get("results", []):
        tim.append(dict(unit=r_.get("unit"), kind=r_.get("kind_"), status=r_.get("status"), n_fit=r_.get("n_fit"), n_fail=r_.get("n_fail"),
                        fit_s=r_.get("fit_s"), wall_s=r_.get("wall_s"), epochs=json.dumps(r_.get("epochs", [])),
                        gpu_mem_peak_mib=r_.get("gpu_mem_peak_mib"), gpu_mem_reserved_peak_mib=r_.get("gpu_mem_reserved_peak_mib"),
                        rss_max_mib=r_.get("rss_max_mib")))
    ck, tm = pd.DataFrame(chk), pd.DataFrame(tim)
    _atomic_csv(ck, a.OUT / f"{a.TAG}_check.csv")
    _atomic_csv(tm, a.OUT / f"{a.TAG}_timing.csv")
    bad = [q for q in chk if q.get("ok") is False]
    print(f"[smoke] 점검 {len(chk)}건 · 불일치 {len(bad)}건 → {a.OUT / (a.TAG + '_check.csv')}", flush=True)
    for q in chk:
        print(f"  [smoke] {q.get('check')} {q.get('learner', '')}: {q.get('value')} ({'일치' if q.get('ok') else ('불일치' if q.get('ok') is False else '기록')})",
              flush=True)
    if len(tm):
        print("[smoke] 단위별 시간(s), 적합 수, 메모리\n" + tm.drop(columns=["epochs"]).to_string(index=False), flush=True)
    return ck, tm


def precheck_outputs(a, res):
    """사전 점검 표(lgfn_precheck_timing.csv, lgfn_precheck_epochs.csv)와 lgf_projection.csv 의 N 행(계획서 8.3 의 식, 8.4 의 축소량).
    lgf_projection.csv 는 h47 과 같은 열 형식(script, row, est_gpu_h)을 쓰고, 창 파일과 같은 잠금 아래에서 N 행만 바꾼다."""
    rows = [q for r_ in res.get("results", []) if r_.get("kind_") in ("pre_a", "pre_b", "pre_c", "pre_d", "chk") for q in r_.get("rows", [])]
    tm = pd.DataFrame([q for q in rows if q.get("part") in ("a", "c", "d")])
    ep = pd.DataFrame([q for q in rows if q.get("part") == "b"])
    ck = pd.DataFrame([q for q in rows if "check" in q])
    _atomic_csv(tm, a.OUT / "lgfn_precheck_timing.csv")
    _atomic_csv(ep, a.OUT / "lgfn_precheck_epochs.csv")
    _atomic_csv(ck, a.OUT / f"{a.TAG}_check.csv")
    proj = projection_n(a, tm, ep)
    write_projection(a, proj, "N")
    rss = [float(q.get("rss_max_mib") or 0) for q in rows]
    print(f"[precheck] 시간 {len(tm)}행 · epoch {len(ep)}행 · 점검 {len(ck)}행 · 최대 RSS {max(rss) if rss else 0:.0f} MiB → "
          f"lgfn_precheck_timing.csv, lgfn_precheck_epochs.csv, lgf_projection.csv(N 행 {len(proj)})", flush=True)
    if len(proj):
        agg = proj[~proj.row.astype(str).str.contains(r"\|")]
        print(agg[["row", "est_gpu_h", "est_cpu_h", "note"]].to_string(index=False), flush=True)
    return tm, ep, proj


def write_projection(a, rows, script):
    """lgf_projection.csv 의 script 행(F 또는 N)을 바꾼다. h47 과 같은 창 파일 잠금(lgf_window.json.lock) 아래에서 읽고 쓴다."""
    path = a.OUT / "lgf_projection.csv"
    new = pd.DataFrame(rows)
    with window_lock(a):
        old = pd.DataFrame()
        if path.exists():
            try:
                old = pd.read_csv(path)
            except Exception:                                             # noqa: BLE001
                old = pd.DataFrame()
        if len(old) and "script" in old:
            old = old[old.script.astype(str) != str(script)]
        elif len(old):
            old = pd.DataFrame()                                          # 형식이 다른 옛 파일(script 열 없음)은 버린다
        out = pd.concat([old, new], ignore_index=True) if len(old) else new
        _atomic_csv(out, path)
    return out


def projection_n(a, tm, ep):
    """계획서 8.3 의 P 계산(N)과 8.4 의 축소·확장량. 조정판 fold 적합 = (설정별 epoch 당 시간) × (fold 학습 행 비) × E_c,
    E_c = min(최대 epoch, 1.25 × (b)의 학습기별 평균 epoch 수)(RealMLP 256). 본 적합은 fold 비 대신 (학습 행 / 레나 원천 행) 비, 기본판은
    (c)의 실측. 기준은 사전 점검 개정 전의 설정(k = mlp·tabm 32, ftt·realmlp 16, 통과 1 추출 2, 분할 1–5, 2순위 Alaska x)이다.
    행(script N): J2, J3, J4, J5, J7, J10, C1(CPU-h), P_N(J2–J5, J7, J10 의 합), E1(ΔE1, 더해지는 양), S1, S5, S6, S7(줄어드는 양),
    그리고 학습기별 세부 행(row '<J>|<학습기>|<항목>')."""
    if not len(tm):
        return pd.DataFrame()
    D = get_data(a)
    lena = src_info(a, D, *PRE_TARGET)
    n_lena = float(len(lena.y))
    a_ = tm[tm.part == "a"] if "part" in tm else pd.DataFrame()
    c_ = tm[tm.part == "c"] if "part" in tm else pd.DataFrame()
    d_ = tm[tm.part == "d"] if "part" in tm else pd.DataFrame()
    base_k = _kv(N_RANDOM_DEFAULT, "n-random")
    main_t = [(t, "x") for t in H.MAIN4]
    t2 = [(H.ALASKA, "x")]
    sp_all = [1, 2, 3, 4, 5]
    now = _now()

    def tpe(lr, trials):
        q = a_[(a_.learner == lr) & (a_.trial.isin(list(trials)))] if len(a_) else a_
        return float(q.sec_per_epoch.mean()) if len(q) else np.nan

    def e_c(lr, tuned=True):
        if lr == "realmlp":
            return float(a.realmlp_epochs)
        q = ep[(ep.learner == lr) & ((ep.trial > 0) if tuned else (ep.trial == 0))] if len(ep) else ep
        mx = float(a.MAX_EPOCHS[lr]) if tuned else float(a.epochs)
        return float(min(mx, 1.25 * float(q.epochs_run.mean()))) if len(q) else mx

    def cdef(lr, what):
        q = c_[(c_.learner == lr) & (c_.what.astype(str).str.startswith(what))] if len(c_) else c_
        return float(q.fit_s.mean()) if len(q) else np.nan

    def parts(lr, k, draws, splits, pairs):
        """(선택 s, 통과 0 s, 통과 1 s). 선택: 1단계(무작위 시도 수 = 시도 목록 − 1, 시도 0) + 2단계(상위 k2 와 시도 0, seed 수)."""
        n_rand = len(n_trials_list(lr, k)) - 1
        t_rand = tpe(lr, n_trials_list(lr, K_LIST_MAX)[1:])
        t0_ = tpe(lr, [0])
        E_t, E_0 = e_c(lr, True), e_c(lr, False)
        sel_s = p0_s = p1_s = 0.0
        for t, m in pairs:
            si = src_info(a, D, t, m)
            scale = len(si.y) / n_lena
            for f in fold_rows(si):
                rr = f["n_tr"] / n_lena
                sel_s += 2 * rr * (n_rand * t_rand * E_t + t0_ * E_0)
                sel_s += 2 * len(a.STAGE2_SEEDS) * rr * (int(a.STAGE2_TOP[lr]) * t_rand * E_t + t0_ * E_0)
            p0_s += 2 * len(a.SEEDS) * (cdef(lr, "p0_shared") + scale * t_rand * E_t * 2.5)
            for sp in valid_splits(a, D, t, m, splits):
                nA = int(D.split_structure(t)[sp]["n_A"])
                ncell = len(H.cells_of([n for n in a.N_GRID if n != 0], int(draws), nA))
                p1_s += ncell * len(a.SEEDS) * 2 * (np.nanmean([cdef(lr, "p1_n10"), cdef(lr, "p1_nall")]) + scale * t_rand * E_t)
        return sel_s / 3600.0, p0_s / 3600.0, p1_s / 3600.0, dict(n_rand=n_rand, t_rand=t_rand, t0=t0_, E_t=E_t, E_0=E_0)

    rows, agg = [], Counter()
    delta = Counter()
    for lr in a.LEARNERS:
        k = int(base_k[lr])
        js, jp = ("J2", "J3") if lr in FAST else ("J4", "J5")
        sel, p0, p1, info = parts(lr, k, a.DRAWS, sp_all, main_t)
        s2, q2, r2, _ = parts(lr, k, a.DRAWS, sp_all, t2)
        note = (f"k={k}, epoch 당 {info['t_rand']:.3f}s(무작위 평균)·{info['t0']:.3f}s(시도 0), E_c {info['E_t']:.0f}·{info['E_0']:.0f}")
        for row, v in ((f"{js}|{lr}|select", sel), (f"{jp}|{lr}|p0", p0), (f"J7|{lr}|p1", p1), (f"J10|{lr}|select", s2), (f"J10|{lr}|p0", q2),
                       (f"J10|{lr}|p1", r2)):
            rows.append(dict(script="N", row=row, job=row.split("|")[0], learner=lr, est_gpu_h=round(v, 3), est_cpu_h=np.nan, time=now, note=note))
        agg[js] += sel; agg[jp] += p0; agg["J7"] += p1; agg["J10"] += s2 + q2 + r2
        p1_d1 = parts(lr, k, 1, sp_all, main_t)[2]
        p1_d1_s3 = parts(lr, k, 1, [1, 2, 3], main_t)[2]
        delta["S5"] += p1 - p1_d1
        delta["S6"] += p1_d1 - p1_d1_s3
        if lr in SLOW:
            delta["E1"] += parts(lr, 32, a.DRAWS, sp_all, main_t)[0] - sel
        if lr in FAST:
            delta["S7"] += sel - parts(lr, 16, a.DRAWS, sp_all, main_t)[0]
    for j in ("J2", "J3", "J4", "J5", "J7", "J10"):
        rows.append(dict(script="N", row=j, job=j, learner="", est_gpu_h=round(agg[j], 3), est_cpu_h=np.nan, time=now,
                         note="사전 점검 실측과 계획서 8.3 의 식(기준 설정)"))
    rows.append(dict(script="N", row="P_N", job="J2–J5,J7,J10", learner="", est_gpu_h=round(sum(agg.values()), 3), est_cpu_h=np.nan, time=now,
                     note="N 기본 범위의 측정 기반 추정(E1·축소 없음)"))
    rows.append(dict(script="N", row="E1", job="J4", learner="ftt,realmlp", est_gpu_h=round(delta["E1"], 3), est_cpu_h=np.nan, time=now,
                     note="더해지는 양: FT-T·RealMLP 무작위 16 → 32(1단계)"))
    rows.append(dict(script="N", row="S1", job="J10", learner="", est_gpu_h=round(agg["J10"], 3), est_cpu_h=np.nan, time=now,
                     note="줄어드는 양: N 2순위(Alaska x) 제외"))
    rows.append(dict(script="N", row="S5", job="J7", learner="", est_gpu_h=round(delta["S5"], 3), est_cpu_h=np.nan, time=now,
                     note="줄어드는 양: 통과 1 추출 2 → 1"))
    rows.append(dict(script="N", row="S6", job="J7", learner="", est_gpu_h=round(delta["S6"], 3), est_cpu_h=np.nan, time=now,
                     note="줄어드는 양: S5 뒤 통과 1 분할 1–5 → 1–3"))
    rows.append(dict(script="N", row="S7", job="J2", learner="mlp,tabm", est_gpu_h=round(delta["S7"], 3), est_cpu_h=np.nan, time=now,
                     note="줄어드는 양: MLP·TabM 무작위 32 → 16(1단계)"))
    if len(d_):
        sel_t = float(d_[d_.what == "select"].fit_s.sum())
        full = d_[d_.what == "full_fit"]
        mean_full = float(full.fit_s.mean()) if len(full) else np.nan
        pairs = main_t + t2
        n_fit = 0
        for t, m in pairs:
            for sp in valid_splits(a, D, t, m, sp_all):
                nA = int(D.split_structure(t)[sp]["n_A"])
                n_fit += len(H.cells_of(list(a.N_GRID), int(a.DRAWS), nA)) * len(a.SEEDS) * 2
        cpu_h = (sel_t * len(pairs) + mean_full * n_fit) / 3600.0
        rows.append(dict(script="N", row="C1", job="C1", learner="catboost_tuned_loc", est_gpu_h=np.nan, est_cpu_h=round(cpu_h, 3), time=now,
                         note=f"선택 {sel_t:.0f}s × 대상 {len(pairs)} + 적합 {n_fit}건 × 평균 {mean_full:.2f}s(9 설정 평균, 2스레드). "
                              f"조건 {a.cbt_cpu_max_h} h 이하이면 cbt 1"))
    return pd.DataFrame(rows)


# ================================================================ 실행
def status_path(a):
    """실행 상태 기록 파일. GPU 작업과 CPU 작업은 tag 가 같으므로 CPU 작업은 '_cpu' 를 붙인다(드레인 경로도 run_<tag>_cpu/drain 으로 다르다)."""
    return a.OUT / f"{a.TAG}{'_cpu' if a.cpu_only else ''}_run_status.json"


def unrun_units(P, res):
    """실행이 멈춤(드레인, 창 마감, 중단, 장치 불일치) 없이 끝났는데 남은 대기 단위. GPU 를 모두 잃어 배정하지 못한 단위와 의존성(선택 실패,
    계획 오류)이 막힌 단위다. 실패로 센다(종료 코드 1). 반환 (단위 이름 목록, 사유)."""
    if res.get("drained") or res.get("closed") or res.get("interrupted") or res.get("aborted"):
        return [], ""
    pend = sorted(nm for nm, s_ in P.state.items() if s_ == "pending")
    if not pend:
        return [], ""
    why = "GPU 를 모두 잃어 배정하지 못함" if res.get("gpu_lost") else "의존성 미충족(선택 실패 또는 계획 오류)"
    return pend, why


def check_overwrite(a, P):
    if P.prev_done and not (a.resume or a.overwrite or a.smoke or a.precheck):
        raise SystemExit(f"[거부] 완료된 단위가 {len(P.prev_done)}개 있다(예: {P.prev_done[0]}). 이어 돌리려면 --resume, 다시 돌려 덮어쓰려면 --overwrite")


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in THREAD_VARS:
        os.environ[v] = str(int(a.threads))
    check_env()
    check_sha(a)
    if a.count_only or a.summarize_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.count_only:
        count_only(a)
        return dict(n_fail=0)
    if a.summarize_only:
        r_ = summarize(a)
        return dict(n_fail=int(r_.get("n_fail", 0)) if r_ else 0, refused=bool(r_ and r_.get("refused")))
    check_run_args(a)
    la = os.getloadavg()[0]
    if la > float(a.load_max_start):
        raise SystemExit(f"[거부] 1분 load average {la:.1f} 가 --load-max-start {a.load_max_start} 를 넘는다")
    if a.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.MAIN:
        check_window_args(a)
        if window_closed(a):
            print("[window] 창 마감이 지났다. 새 단위를 시작하지 않는다", flush=True)
            return dict(n_fail=0, closed="window")
    ensure_configs(a, write=not a.MAIN, require=a.MAIN)
    D = get_data(a)
    gpus = []
    if not a.cpu_only:
        try:
            info, apps = T43.query_gpu_info(), T43.query_gpu_apps()
        except Exception as e:                                            # noqa: BLE001
            raise SystemExit(f"[거부] nvidia-smi 로 GPU 상태를 확인할 수 없다: {repr(e)[:200]}")
        gpus, dropped = screen_gpus_n(a, info, apps)
        for g_, why in dropped:
            print(f"[warn] GPU {g_} 를 뺀다: {why}", flush=True)
        if not gpus:
            print("[거부] 쓸 수 있는 GPU 가 없다(지정한 GPU 가 모두 사용 중이다)", flush=True)
            return dict(n_fail=1)
        if a.smoke or a.precheck:
            gpus = gpus[:1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        if not T43.cuda_available(gpus[0]):
            raise SystemExit("[거부] torch.cuda 를 쓸 수 없다")
    a.RUN_ID = f"{os.getpid()}-{time.strftime('%Y%m%dT%H%M%S')}"
    a.NSEL.mkdir(parents=True, exist_ok=True)
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    lock = T43.acquire_lock(a)                                            # 계획 단계(선택 표 갱신)보다 먼저 잠근다
    try:
        P = StaticPlanner(a, D, precheck_units(a)) if a.precheck else build_planner(a, D, fresh=bool(a.overwrite))
        check_overwrite(a, P)
    except BaseException:
        T43.release_lock(lock)
        raise
    T43.clear_run_notes(a.RUN_DIR, ("running", "gpu_busy", "worker"))
    res = {}
    try:
        print(f"[plan] tag {a.TAG} · 작업 {a.JOBS} · 학습기 {a.LEARNERS} · 단위 {P.counts()} · GPU {gpus or 'CPU'} · 스레드 {a.threads} · nice {os.nice(0)} · "
              f"k {a.N_RANDOM} · 축소 {a.REDUCE or '없음'} · E1 {int(a.E1)} · 설정 출처 {getattr(a, 'SETTINGS_SRC', '명령행')} · run_id {a.RUN_ID}",
              flush=True)
        res = run_cpu(a, P, t0) if a.cpu_only else run_gpu(a, P, gpus, argv, t0)
    finally:
        blocked = P.blocked()
        unrun, unrun_why = unrun_units(P, res)
        n_part = sum(1 for r_ in res.get("results", []) if r_.get("status") == "partial")
        n_fail = (len(res.get("failed", [])) + sum(1 for r_ in res.get("results", []) if r_.get("status") == "failed") + len(unrun)
                  + (len(P.errors) if not unrun else 0))
        stat = dict(run_id=a.RUN_ID, tag=a.TAG, cpu_only=bool(a.cpu_only), argv=argv, jobs=a.JOBS, elapsed_s=round(time.time() - t0, 1), gpus_first=gpus,
                    gpus_last=res.get("gpus_last", []), busy_gpus=res.get("busy_gpus", []), unit_states=P.counts(), n_done=len(res.get("results", [])),
                    n_fail=n_fail, n_partial=n_part, failed_units=[dict(unit=u_, reason=r_) for u_, r_ in res.get("failed", [])],
                    quarantined=res.get("quarantined", []), blocked=blocked[:200], n_blocked=len(blocked), planner_errors=P.errors,
                    unrun_units=[dict(unit=u_, reason=unrun_why) for u_ in unrun[:200]], n_unrun=len(unrun), gpu_lost=bool(res.get("gpu_lost")),
                    drained=res.get("drained", ""), closed=res.get("closed", ""), aborted=res.get("aborted", ""), interrupted=res.get("interrupted", ""),
                    settings=dict(E1=bool(a.E1), reduce_n=list(a.REDUCE_N), k=dict(a.N_RANDOM), source=getattr(a, "SETTINGS_SRC", "명령행")))
        _write_json(status_path(a), stat)
        T43.release_lock(lock)
    print(f"[done] 단위 상태 {P.counts()} · 실패 {n_fail}(실행하지 못한 단위 {len(unrun)}{': ' + unrun_why if unrun else ''}) · partial {n_part} · "
          f"대기(의존성 미충족) {len(blocked)} · {time.time() - t0:.0f}s", flush=True)
    out = dict(n_fail=n_fail, n_partial=n_part, drained=res.get("drained"), closed=res.get("closed"), interrupted=res.get("interrupted"))
    if res.get("interrupted"):
        return out
    if a.smoke:
        smoke_report(a, res)
    elif a.precheck:
        precheck_outputs(a, res)
    elif not a.no_summarize and not res.get("drained"):
        r_ = summarize(a)
        if r_:
            out["n_fail"] += int(r_.get("n_fail", 0))
    return out


# ================================================================ 집계(--summarize-only)
LEARNER_TXT = dict(mlp="MLP", tabm="다중 헤드 MLP", ftt="축소 FT-Transformer", realmlp="RealMLP")
SUP_STRONG = "지지(물리식보다 오차가 크거나 구별되지 않음)"
SUP_WEAK = "지지(우세 근거 없음)"


def comparator_n3(cbt_enabled):
    """N3·N2s 의 주 비교 대상(계획서 4.4·4.5). 창 파일의 cbt_enabled 로 정한다(자료 완결 상태로 정하지 않는다). 참이면 catboost_tuned_loc 이고
    조각이 모자라면 분할 완결성·판정 불가 규칙으로 처리한다. 거짓이거나 결정이 없으면 catboost_lo 이고 '양쪽 조정' 문장을 쓰지 않는다.
    반환 (학습기 이름, 약칭, 사유)."""
    if cbt_enabled is True:
        return "catboost_tuned_loc", "CBT", ""
    why = "창 파일의 cbt_enabled 가 거짓: catboost_tuned_loc 미실행" if cbt_enabled is False else "창 파일에 cbt 결정 없음"
    return "catboost_lo", "CB", why


def shard_decisions(sh, units, dec_tab):
    """조각(unit.json 의 decisions)에 기록된 선택 {(학습기, 종류, 대상, 모드): {기준: dict(trial, flag)}}과 문제 목록.
    통과 0 조각은 main·cw·l25, 통과 1 조각은 main 을 기록한다. 조각 사이에 선택이 다르거나, 조각에 기록이 없거나, 선택 표(dec_tab, 현재
    sel_sig)와 다르면 문제로 적는다. 집계의 k_default, '시도 0 이 아닌 지역' 보조 층화 평균, N4 는 이 조각 기록으로 계산한다."""
    out: dict = {}
    probs: list = []
    for s_, u in zip(sh, units):
        if s_["pass_"] not in ("p0", "p1"):
            continue
        lr, t, m = s_["learner"], s_["target"], s_["mode"]
        where = f"{s_['pass_']}|{lr}|{t}|{m}|s{s_['split']}"
        d = u.get("decisions")
        if not d:
            probs.append(f"{where}: 조각에 선택 기록이 없다(선택 표 없음)")
            continue
        for kind in KINDS:
            dk = d.get(kind)
            if not dk:
                probs.append(f"{where}: 종류 {kind} 의 선택 기록이 없다")
                continue
            crits = {"main": dk} if "trial" in dk else dk
            for crit, v in crits.items():
                if not v:
                    continue
                rec = dict(trial=int(v["trial"]), flag=str(v.get("flag", "") or ""))
                slot = out.setdefault((lr, kind, t, m), {})
                if crit in slot and slot[crit]["trial"] != rec["trial"]:
                    probs.append(f"{where}: 종류 {kind} 기준 {crit} 의 시도 {rec['trial']} 가 다른 조각의 {slot[crit]['trial']} 와 다르다")
                    continue
                slot.setdefault(crit, rec)
    for key, slot in sorted(out.items()):
        tab = dec_tab.get(key) or {}
        for crit, rec in slot.items():
            tv = tab.get(crit)
            if tv is None:
                probs.append(f"{'|'.join(key)}: 선택 표에 기준 {crit} 의 선택이 없다(sel_sig 불일치 가능)")
            elif int(tv.get("trial", -1)) != rec["trial"]:
                probs.append(f"{'|'.join(key)}: 기준 {crit} 의 조각 선택 {rec['trial']} 가 선택 표 {tv.get('trial')} 와 다르다")
    return out, probs


def find_shards_n(a, tag=None):
    """집계용 조각(gpu 7마디: tag, gpu, 학습기, 대상, 모드, s분할, 통과 / cpu 6마디: tag, cpu, 학습기, 대상, 모드, s분할)."""
    tag = tag or a.TAG
    out = []
    if not Path(a.SHARDS).exists():
        return out
    for p in sorted(Path(a.SHARDS).glob(f"{tag}__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")
        if not parts or parts[0] != tag:
            continue
        if len(parts) == 7 and parts[1] == "gpu":
            lr, t, m, sp, ps = parts[2], parts[3], parts[4], parts[5], parts[6]
        elif len(parts) == 6 and parts[1] == "cpu":
            lr, t, m, sp, ps = parts[2], parts[3], parts[4], parts[5], "cpu"
        else:
            continue
        b = str(p)[:-len("_unit.json")]
        if Path(b + "_runs.csv").exists() and Path(b + "_blocksse.npz").exists():
            out.append(dict(unit=p, runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), learner=lr, target=t, mode=m, split=int(sp[1:]), pass_=ps))
    return out


def read_runs_n(shards):
    frames = []
    for s_ in shards:
        if s_["runs"].stat().st_size > 1:
            f = pd.read_csv(s_["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str, arm=str, sel_flag=str, learner=str, method=str),
                            keep_default_na=False, na_values=["", "nan", "NaN"])
            if len(f):
                frames.append(f)
    if not frames:
        return pd.DataFrame(columns=RUN_COLS)
    runs = pd.concat(frames, ignore_index=True)
    for c_ in ("alpha_sel", "fit_flag", "arm", "sel_flag"):
        runs[c_] = runs[c_].fillna("").astype(str) if c_ in runs else ""
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int) if "n_nonfinite" in runs else 0
    return runs.drop_duplicates(subset=["target", "mode", "split"] + list(H.KEY_COLS), keep="first")


def check_cfg_n(a, shards, units):
    """(통과, 학습기)마다 설정 해시가 하나여야 한다(--allow-mixed-cfg 로 진행)."""
    info = {}
    for key in sorted({(s_["pass_"], s_["learner"]) for s_ in shards}):
        us = [u for s_, u in zip(shards, units) if (s_["pass_"], s_["learner"]) == key]
        hs = Counter(str(u.get("cfg_hash", "legacy")) for u in us)
        cs = Counter(str((u.get("code_sha") or {}).get("h48", "legacy")) for u in us)
        info["|".join(key)] = dict(cfg_hash=dict(hs), code_sha=dict(cs), n=len(us))
        if len(cs) > 1:
            print(f"  [warn] {key}: 조각의 코드 해시가 {len(cs)}종이다 {dict(cs)}", flush=True)
        if len(hs) > 1:
            msg = f"{key}: 조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}"
            if not a.allow_mixed_cfg:
                raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
            print("  [warn] " + msg, flush=True)
    return info


def scope_missing(a, D):
    """기본 범위(주 4지역과, S1 이 아니면 Alaska x)의 p0, p1, catboost_lo 조각 가운데 완료되지 않은 것."""
    miss = []
    for t, m in list(a.MAIN_T) + list(a.T2):
        for lr in a.LEARNERS:
            miss += [f"p0|{lr}|{t}|{m}|s{sp}" for sp in valid_splits(a, D, t, m, a.SPLITS) if not shard_state(a, "p0", lr, t, m, sp)[0]]
            miss += [f"p1|{lr}|{t}|{m}|s{sp}" for sp in valid_splits(a, D, t, m, a.P1_SPLITS) if not shard_state(a, "p1", lr, t, m, sp)[0]]
        miss += [f"cpu|catboost_lo|{t}|{m}|s{sp}" for sp in valid_splits(a, D, t, m, a.SPLITS) if not shard_state(a, "cpu", "catboost_lo", t, m, sp)[0]]
    return miss


def _pair_fn(la, lb):
    """학습기 짝 대비용 저장소 변환: 학습기 la·lb 의 키는 상대의 같은 (method, alpha, placement, n, draw, seed, lam) 키가 있을 때만 남긴다."""
    def fn(st):
        keep, drop = [], []
        for k in st.keys:
            if k[1] in (la, lb):
                other = lb if k[1] == la else la
                if (k[0], other) + tuple(k[2:]) not in st:
                    drop.append(k); continue
            keep.append(k)
        return (T43._sub_store(st, keep), drop) if drop else (st, [])
    return fn


def make_testbook(X, ns, tms, D, floor):
    """h42.TestBook 에 LGF-N 의 확인적 가설(LGF-N1) 규칙을 더한다: 분할이 기대보다 적으면 판정 불가(분할 k/K)."""
    class TestBookN(X.TestBook):
        def verdict(self, test, item, text, stat="", role="주", blind=True, used=None, partial=None, point=False, **kw):
            text = str(text)
            conf = test in CONFIRMATORY_N
            mk = self.marks(used, point)
            pre = [str(v) for v in (partial or []) if v]
            if not text.startswith("판정 불가"):
                if mk["splits"] is not None and conf:
                    text = f"판정 불가(분할 {mk['splits'][0]}/{mk['splits'][1]}. 등록한 분할이 모두 채워진 뒤 판정한다)"
                else:
                    if mk["pool"] is not None:
                        pre.append(f"지역 {mk['pool'][0]}/{mk['pool'][1]}")
                    if mk["splits"] is not None:
                        pre.append(f"분할 {mk['splits'][0]}/{mk['splits'][1]}")
                    if pre:
                        text = f"부분({', '.join(pre)}): {text}"
                    if mk["small"]:
                        text = f"{text} [{X.SMALL_EFFECT_TXT}: {', '.join(mk['small'])}]"
            self.rows.append(dict(test_id=test, item=item, scope="verdict" if role == "주" else "verdict_aux", verdict=text, stat=str(stat), role=role,
                                  primary=False, blind=bool(blind), n_used=len(used or []),
                                  n_used_valid=int(sum((r is not None) if point else X.valid_row(r) for _, r in (used or []))), **kw))
    return TestBookN(ns, tms, D, floor)


def _nl(n):
    return "all" if int(n) == -1 else str(int(n))


def _nt(n):
    return "전량" if int(n) == -1 else f"n={int(n)}"


def halfwidth(r):
    """95 % CI 반폭 = max(셀 가중 폭, 블록 등가중 폭)/2(계획서 2.2). 네 끝값이 모두 유한할 때만 계산하고 아니면 NaN(h47.halfwidth 와 같은 정의)."""
    if r is None:
        return np.nan
    v = [r.get("ci_lo"), r.get("ci_hi"), r.get("ci_lo_beq", r.get("ci_lo_b")), r.get("ci_hi_beq", r.get("ci_hi_b"))]
    try:
        v = [float(x) for x in v]
    except (TypeError, ValueError):
        return np.nan
    if not all(np.isfinite(v)):
        return np.nan
    return max(v[1] - v[0], v[3] - v[2]) / 2.0


def precision_ok(r, delta):
    hw = halfwidth(r)
    return bool(np.isfinite(hw) and hw <= float(delta))


def support_class(X, r):
    """'우세가 아니면 지지' 형식의 두 갈래(계획서 2.2)."""
    if not X.valid_row(r):
        return "판정 불가"
    v = X._v(r)
    if v == "우세":
        return "기각"
    return SUP_STRONG if v in ("열세", "동등") else SUP_WEAK


def eq_verdict(X, a, base, lam1, up, down, eq_txt):
    """동등성을 묻는 가설(N2, N3)의 판정 문구와 갈래. base = 기준 λ 대비(와 D0), lam1 = R1 의 λ 1.0 대비. 정밀도 표기와 λ 조합 규칙을 쓴다."""
    allr = list(base) + list(lam1)
    k, m, _ = X.count_valid(allr)
    vs = [(lab, X._v(r), r) for lab, r in allr]
    dom = [(lab, v, r) for lab, v, r in vs if v in ("우세", "열세")]
    dtxt = ", ".join(f"{lab} {v}(Δ {r['delta']:+.2f} cm)" for lab, v, r in dom)
    kp = sum(1 for _, r in allr if X.valid_row(r) and not precision_ok(r, a.delta_eq))
    mark = f"(정밀도 미달 대비 {kp}/{m})" if kp else ""                  # 표기는 정밀도 미달 대비가 있을 때만(h47.prec_suffix 와 같은 규칙)
    if k < m:
        return X.na_text(allr) + (f". 판정이 나온 대비의 우세·열세: {dtxt}" if dom else "") + f" [부분(대비 {k}/{m})]", "na"
    vb, v1 = [X._v(r) for _, r in base], [X._v(r) for _, r in lam1]
    if all(v == "동등" for v in vb + v1):
        return eq_txt + mark, "eq"
    if all(v == "동등" for v in vb) and any(v in ("우세", "열세") for v in v1):
        d1 = ", ".join(f"{lab} {v}" for (lab, _), v in zip(lam1, v1) if v in ("우세", "열세"))
        return f"기준 λ {H.LAM_BASE:g} 에서는 동등, λ 1.0 에서는 차이가 있다({d1}). 동등으로 쓰지 않는다" + mark, "lam"
    ups, downs = [q for q in dom if q[1] == "우세"], [q for q in dom if q[1] == "열세"]
    if ups and not downs:
        return f"{up}({dtxt})" + mark, "up"
    if downs and not ups:
        return f"{down}({dtxt})" + mark, "down"
    if ups and downs:
        return f"혼재({dtxt})" + mark, "mixed"
    und = [r for _, r in allr if X._v(r) == "미결정"]
    if und and all(not precision_ok(r, a.delta_eq) for r in und):
        return f"정밀도 미달(동등성 판정 불가, 대비 {kp}/{m})", "prec"
    return "차이를 확인하지 못함" + mark, "ns"


def n1_text(X, res1, cls, learners):
    """LGF-N1 판정 문구(계획서 4.5). res1 = [(학습기, D0[l*] − P0 대비 행)], cls = 학습기별 지지 갈래(support_class).
    판정이 나오지 않은 대비가 있으면 판정 불가(나온 대비에 우세가 있으면 '기각(일부 대비 판정 불가)'). 우세가 있으면 기각(한정어와 곡선 규칙).
    그 밖은 지지: 강한 지지 학습기만으로 등록 문장을 쓰고, 약한 지지 학습기는 '조정한 <학습기> 에서 우세가 관찰되지 않았다(CI [a, b] cm)'로 따로 적는다."""
    labs = [(f"조정한 {lr}", r) for lr, r in res1]
    kv, mv, _ = X.count_valid(labs)
    wins = [lr for lr, r in res1 if X.valid_row(r) and X._v(r) == "우세"]
    if kv < mv:
        return f"기각(일부 대비 판정 불가, 대비 {kv}/{mv})" if wins else X.na_text(labs)
    if wins:
        wl = ", ".join(LEARNER_TXT.get(lr, lr) for lr in wins)
        return (f"기각: 조정한 {wl} 의 직접 ML 이 라벨 0 에서 원천 계수 물리식(P0)보다 우세하다. L1 의 라벨 0 부분 결론을 '기본 초모수 한정'으로 적고, "
                f"조정한 {wl} 의 D0 곡선을 플랫폼 표지와 로컬 기본판·catboost_lo 곡선과 함께 마스터 곡선에 더한다")
    strong = [lr for lr in learners if cls.get(lr) == SUP_STRONG]
    weak = [(lr, r) for lr, r in res1 if cls.get(lr) == SUP_WEAK]
    parts = []
    if strong:
        parts.append(f"{SUP_STRONG}({', '.join(strong)}): 원천 지역 하나 제외 교차검증으로 초모수를 골라도 시험한 판별 신경망 {len(strong)}종"
                     f"({', '.join(LEARNER_TXT.get(lr, lr) for lr in strong)})의 직접 ML 은 라벨 0 전이에서 원천 계수 물리식을 넘지 못한다")
    if weak:
        parts.append(f"{SUP_WEAK}({', '.join(lr for lr, _ in weak)}): " + "; ".join(
            f"조정한 {LEARNER_TXT.get(lr, lr)} 에서 우세가 관찰되지 않았다(CI [{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}] cm)" for lr, r in weak))
    return ("지지. " if strong and weak else "") + ". ".join(parts)


def region_hits(X, tms, names, gA_fn, gB_fn, ns_):
    """L1 형식: 지역 블록 CI(두 가중) 상한 < 0 인 (지역, n). h42.region_stats 의 분포를 쓴다."""
    hits = []
    for nm in names:
        tm = tms.get(nm)
        if tm is None:
            continue
        for n in ns_:
            s_ = X.region_stats(tm, gA_fn(n), gB_fn(n))
            if s_ is None or s_.get("dist") is None:
                continue
            _, hi = X._ci(s_["dist"])
            _, hib = X._ci(s_["dist_beq"])
            if np.isfinite(hi) and np.isfinite(hib) and hi < 0 and hib < 0:
                hits.append((nm, int(n)))
    return hits


def build_tests_n(a, tms, D, floor, X, dec, SEL, cbt_enabled=None):
    """LGF-N1, N2, N2s, N3, N4 의 대비 행과 판정 행(계획서 4.5). Holm 묶음: conf = N1 의 4개, aux = N2 의 기준 λ 대비 24 + N3 12
    (item 'aux', primary), eq = 같은 대비의 동등성 p(eq_test). dec = 조각에 기록된 선택(shard_decisions). N3·N2s 의 주 비교 대상은
    창 파일의 cbt_enabled 로 정한다(comparator_n3)."""
    ns = SimpleNamespace(nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux))
    T = make_testbook(X, ns, tms, D, floor)
    V = T.verdict
    LB, ALL = float(H.LAM_BASE), -1
    main_t = list(a.MAIN_T)
    m4 = [f"{t}|{m}" for t, m in main_t]
    K = len(main_t)
    views: dict = {None: tms}
    unp: dict = {}
    memo: dict = {}

    def view(pair):
        if pair not in views:
            fn = _pair_fn(*pair)
            out, cnt_by = {}, {}
            for nm, tm in tms.items():
                t2, dr = T43.tm_view(tm, fn)
                cnt = Counter()
                for sp, ks in dr.items():
                    if sp in tm.used:
                        for k in ks:
                            cnt[(k[0], str(k[2]), k[3], int(k[4]), float(k[7]))] += 1
                out[nm], cnt_by[nm] = t2, cnt
            views[pair], unp[pair] = out, cnt_by
        return views[pair]

    def n_unpaired(pair, gA, gB, names):
        tab = unp.get(pair, {})
        grp = {(q[0], str(q[2]), q[3], int(q[4]), float(q[5])) for q in (gA, gB)}
        return int(sum(tab.get(nm, Counter())[q] for nm in names for q in grp))

    def C(test, item, label, gA, gB, pair=None, names=None, **kw):
        key = (test, label, tuple(names or ()), pair)
        if key in memo:
            return memo[key]
        kw.setdefault("role", "보조"); kw.setdefault("primary", False); kw.setdefault("blind", True)
        vw = view(pair) if pair else tms
        extra = dict(key_set="pair" if pair else "all", holm_family=item if kw.get("primary") else "", platform=PLATFORM)
        if pair:
            extra["n_unpaired"] = n_unpaired(pair, gA, gB, list(names or m4) + ([T.m3[2]] if not names else []))
        T.tms = vw
        try:
            r = T.contrast(test, item, label, gA, gB, names=list(names) if names else None, **kw, **extra)
        finally:
            T.tms = tms
        memo[key] = r
        return r

    def G(method, n, lr=None, lam=None):
        return X.G(method, n, lam, lr)

    def dtrial(lr, kind, t, m, crit="main"):
        return (dec.get((lr, kind, t, m)) or {}).get(crit, {}).get("trial")

    def kd(lr, kind, crit="main"):
        return sum(1 for t, m in main_t if dtrial(lr, kind, t, m, crit) == 0), K

    def kdt(lr, kind, crit="main"):
        """층화 평균 행의 k_default 열(계획서 4.3: 조정판이 들어간 대비의 'Δ = 0(선택 결과) 지역 k/4')."""
        return f"Δ = 0(선택 결과) 지역 {kd(lr, kind, crit)[0]}/{K}" + ("" if crit == "main" else f"(기준 {crit})")

    def nondef(lr, kind):
        return [f"{t}|{m}" for t, m in main_t if dtrial(lr, kind, t, m) not in (None, 0)]

    def tu(lr, arm="tuned"):
        return arm_name(lr, arm)

    cmp_, cmp_txt, cmp_why = comparator_n3(cbt_enabled)
    has_cbt = cmp_ == "catboost_tuned_loc"

    # ---------------- LGF-N1(확인적): D0[l*] − P0, n = 0, λ 1.0
    res1, cls = [], {}
    for lr in a.LEARNERS:
        k, _ = kd(lr, "D")
        r = C("LGF-N1", "conf", f"D0[{lr}*]-P0|n0", G("D0", 0, tu(lr), 1.0), G("P0", 0), n=0, lam=1.0, primary=True, blind=True, role="주",
              learner=lr, prior_info="M1(사전 정보: 기본 초모수 신경망의 방향)", k_default=f"Δ = 0(선택 결과) 지역 {k}/{K}")
        res1.append((lr, r))
        cls[lr] = support_class(X, r)
    txt = n1_text(X, res1, cls, a.LEARNERS)
    alldef = [lr for lr in a.LEARNERS if kd(lr, "D")[0] == K]
    ktxt = "Δ = 0(선택 결과) 지역(종류 D): " + ", ".join(f"{lr} {kd(lr, 'D')[0]}/{K}" for lr in a.LEARNERS)
    if alldef:
        ktxt += f". 조정판 = 기본판(선택 결과): {', '.join(alldef)}(이 학습기의 N1 은 로컬 기본판의 대비와 같다)"
    labs = [(f"조정한 {lr}", r) for lr, r in res1]
    txt = f"{txt} [{ktxt}]"
    stat = X._fmt(labs) + " | " + ktxt
    V("LGF-N1", "conf", txt, stat, role="주", blind=False, used=labs, support_class=json.dumps(cls, ensure_ascii=False),
      prior_info="M1. 비맹검 부분 포함(기본판의 방향)", k_default=json.dumps({lr: f"{kd(lr, 'D')[0]}/{K}" for lr in a.LEARNERS}))
    for lr in a.LEARNERS:
        C("LGF-N1", "N1aux", f"D0[{lr}]-P0|n0", G("D0", 0, lr, 1.0), G("P0", 0), n=0, lam=1.0, role="보조(a) 로컬 기본판", blind=False, learner=lr,
          prior_info="재현(비맹검, M1)")
        bl = [(f"{_nt(n)}", C("LGF-N1", "N1aux", f"D0[{lr}*]-P0|n{n}", G("D0", n, tu(lr), 1.0), G("P0", 0), n=n, lam=1.0, role="보조(b)",
                             learner=lr, k_default=kdt(lr, "D"))) for n in (10, 40)]
        C("LGF-N1", "N1aux", f"D0[{lr}_tuned_cw]-P0|n0", G("D0", 0, tu(lr, "tuned_cw"), 1.0), G("P0", 0), n=0, lam=1.0, role="보조(e) 민감도 판(셀 가중)",
          learner=lr, k_default=kdt(lr, "D", "cw"))
        C("LGF-N1", "N1aux", "D0[catboost_lo]-P0|n0", G("D0", 0, "catboost_lo", 1.0), G("P0", 0), n=0, lam=1.0, role="병기(catboost_lo)",
          blind=False, prior_info="재현(비맹검, M1·a2 의 CatBoost 방향)")
        hits = region_hits(X, tms, m4, lambda n, lr=lr: G("D0", n, tu(lr), 1.0), lambda n: G("P0", 0), (0, 10, 40))
        regs = sorted({h[0] for h in hits})
        T.rows.append(dict(test_id="LGF-N1", item="N1aux", contrast=f"L1형식|D0[{lr}*]-P0|n0,10,40", scope="l1_count", role="보조(c)", learner=lr,
                           n_regions_beat=len(regs), regions_beat=json.dumps(hits), blind=True, primary=False, platform=PLATFORM))
        bwin = [lab for lab, r in bl if X._v(r) == "우세"]
        parts = []
        if len(regs) >= 2:
            parts.append(f"L1 결론에 '조정한 {lr} 는 n ≤ 40 에서 예외({', '.join(f'{nm} n={n}' for nm, n in hits)})'를 적는다")
        if bwin:
            parts.append(f"보조 행 (b)에서 우세인 n({', '.join(bwin)})을 L1 결론의 한계로 적는다")
        V("LGF-N1", "N1aux", "; ".join(parts) if parts else "L1 형식 우세 지역 2개 미만, n ∈ {10, 40} 우세 없음",
          f"L1 형식 지역 수 {len(regs)}/{K} {hits} | (b) {X._fmt(bl)}", role="보조", blind=True, used=bl, learner=lr, clause="L1 과의 관계")

    # ---------------- LGF-N2: 조정판 − 기본판(짝 키)
    robust_all = []
    for lr in a.LEARNERS:
        pair = (tu(lr), lr)
        kR, _ = kd(lr, "R"); kD, _ = kd(lr, "D")
        base, lam1, nd_rows = [], [], []
        spec = []
        for n in (0, 10, ALL):
            spec.append(("R1", n, LB, "R"))
        for n in (0, 10, ALL):
            spec.append(("D0", n, 1.0, "D"))
        for mth, n, lam, kind in spec:
            lab = f"{mth} {_nt(n)}" + (f" λ{lam:g}" if mth == "R1" else "")
            r = C("LGF-N2", "aux", f"{mth}[{lr}*]-{mth}[{lr}]|n{_nl(n)}|lam{lam:g}", G(mth, n, tu(lr), lam), G(mth, n, lr, lam), pair=pair, n=n, lam=lam,
                  primary=True, eq_test=True, role="주", learner=lr, k_default=f"Δ = 0(선택 결과) 지역 {kd(lr, kind)[0]}/{K}")
            base.append((lab, r))
            nm_ = nondef(lr, kind)
            if nm_:
                nd_rows.append((lab, C("LGF-N2", "N2aux", f"{mth}[{lr}*]-{mth}[{lr}]|n{_nl(n)}|lam{lam:g}|nondefault", G(mth, n, tu(lr), lam),
                                       G(mth, n, lr, lam), pair=pair, names=nm_, n=n, lam=lam, role="보조(시도 0 이 아닌 지역)", learner=lr,
                                       k_default=f"Δ = 0(선택 결과) 지역 0/{len(nm_)}(시도 0 이 아닌 지역만)")))
            else:
                nd_rows.append((lab, None))
        for n in (0, 10, ALL):
            lab = f"R1 {_nt(n)} λ1"
            r = C("LGF-N2", "aux", f"R1[{lr}*]-R1[{lr}]|n{_nl(n)}|lam1", G("R1", n, tu(lr), 1.0), G("R1", n, lr, 1.0), pair=pair, n=n, lam=1.0,
                  primary=False, eq_test=True, role="보조(λ 1.0)", learner=lr, k_default=f"Δ = 0(선택 결과) 지역 {kR}/{K}")
            lam1.append((lab, r))
            nm_ = nondef(lr, "R")
            nd_rows.append((lab, C("LGF-N2", "N2aux", f"R1[{lr}*]-R1[{lr}]|n{_nl(n)}|lam1|nondefault", G("R1", n, tu(lr), 1.0), G("R1", n, lr, 1.0),
                                   pair=pair, names=nm_, n=n, lam=1.0, role="보조(시도 0 이 아닌 지역)", learner=lr,
                                   k_default=f"Δ = 0(선택 결과) 지역 0/{len(nm_)}(시도 0 이 아닌 지역만)") if nm_ else None))
        for n in (40, 160):
            C("LGF-N2", "N2aux", f"R1[{lr}*]-R1[{lr}]|n{n}|lam{LB:g}", G("R1", n, tu(lr), LB), G("R1", n, lr, LB), pair=pair, n=n, lam=LB,
              role="보조(n 40·160)", learner=lr, k_default=kdt(lr, "R"))
            C("LGF-N2", "N2aux", f"D0[{lr}*]-D0[{lr}]|n{n}", G("D0", n, tu(lr), 1.0), G("D0", n, lr, 1.0), pair=pair, n=n, lam=1.0,
              role="보조(n 40·160)", learner=lr, k_default=kdt(lr, "D"))
        for arm, mth, lam, kind, crit in (("tuned_cw", "R1", LB, "R", "cw"), ("tuned_l25", "R1", LB, "R", "l25"), ("tuned_cw", "D0", 1.0, "D", "cw")):
            C("LGF-N2", "N2aux", f"{mth}[{tu(lr, arm)}]-{mth}[{lr}]|n0", G(mth, 0, tu(lr, arm), lam), G(mth, 0, lr, lam), pair=(tu(lr, arm), lr), n=0,
              lam=lam, role="보조(민감도 판)", learner=lr, k_default=kdt(lr, kind, crit))
        txt, br = eq_verdict(X, a, base, lam1, "조정판의 오차가 작다(대비, n, 크기. LGF N3 표로 보고)", "원천 교차검증 조정이 전이 오차를 키웠다", "강건 후보(한계 0.5 cm)")
        robust = False
        if br == "eq":
            if kR == K and kD == K:
                sent = "원천 교차검증 조정 절차는 모든 대상에서 기본 설정을 골랐다(조정판 = 기본판)"
            elif all(r is not None and X._v(r) == "동등" for _, r in nd_rows):
                sent, robust = f"조정한 {lr} 의 결론은 초모수 선택에 강건하다", True
            else:
                sent = f"원천 교차검증 조정 절차를 적용해도 차이는 한계 안이다(선택 결과 Δ = 0 지역 R {kR}/{K}, D {kD}/{K} 포함)"
            txt = f"{txt}. {sent}"
        if kR >= 1 or kD >= 1:
            txt = f"Δ = 0(선택 결과) 지역 {kR}/{K}(종류 R), {kD}/{K}(종류 D): {txt}"
        robust_all.append((lr, robust))
        V("LGF-N2", "aux", txt, f"{X._fmt(base + lam1)} | 시도 0 이 아닌 지역만: {X._fmt([(lab, r) for lab, r in nd_rows if r is not None])}",
          role="보조", blind=True, used=base + lam1, learner=lr, branch=br, k_default=f"R {kR}/{K}, D {kD}/{K}")
    if robust_all:
        ok = all(v for _, v in robust_all)
        V("LGF-N2", "aux", "신경망의 결론은 초모수 선택에 강건하다" if ok else
          f"전체 문장 조건 미충족(강건 문장 규칙을 채운 학습기: {', '.join(lr for lr, v in robust_all if v) or '없음'})",
          json.dumps(dict(robust_all), ensure_ascii=False), role="보조", blind=True, used=[], clause="전체")

    # ---------------- LGF-N2s: 핵심 대비의 판정 안정성(기본판 대 조정판)
    stab_all = []
    for lr in a.LEARNERS:
        items = []

        def kk(arm, kind, lr=lr):
            return kdt(lr, kind) if arm != lr else ""
        spec = [("D0−P0 n=0", lambda arm: C("LGF-N2s", "N2s", f"D0[{arm}]-P0|n0", G("D0", 0, arm, 1.0), G("P0", 0), n=0, lam=1.0, role="보조(N2s)",
                                            k_default=kk(arm, "D"))),
                ("R1−P1 n=10", lambda arm: C("LGF-N2s", "N2s", f"R1[{arm}]-P1|n10", G("R1", 10, arm, LB), G("P1", 10), n=10, lam=LB, role="보조(N2s)",
                                             k_default=kk(arm, "R"))),
                ("R1−P1 전량", lambda arm: C("LGF-N2s", "N2s", f"R1[{arm}]-P1|nall", G("R1", ALL, arm, LB), G("P1", ALL), n=ALL, lam=LB, role="보조(N2s)",
                                            k_default=kk(arm, "R"))),
                ("R1−P0 전량", lambda arm: C("LGF-N2s", "N2s", f"R1[{arm}]-P0|nall", G("R1", ALL, arm, LB), G("P0", 0), n=ALL, lam=LB, role="보조(N2s)",
                                            k_default=kk(arm, "R")))]
        for n in (0, 10, ALL):
            spec.append((f"R1−{cmp_txt} {_nt(n)}", lambda arm, n=n: C("LGF-N2s", "N2s", f"R1[{arm}]-R1[{cmp_}]|n{_nl(n)}", G("R1", n, arm, LB),
                                                                      G("R1", n, cmp_, LB), pair=(arm, cmp_), n=n, lam=LB, role="보조(N2s)",
                                                                      k_default=kk(arm, "R"))))
        for lab, fn in spec:
            r0, r1 = fn(lr), fn(tu(lr))
            s_ = T43.stability(X, r0, r1)
            items.append((lab, s_))
            T.rows.append(dict(test_id="LGF-N2s", item="N2s", contrast=f"{lr}|{lab}", scope="stability", learner=lr, role="보조", primary=False,
                               blind=True, verdict4_base=X._v(r0), verdict4_tuned=X._v(r1), stability=s_, platform=PLATFORM))
        txt = T43.stab_text(items)
        kR, _ = kd(lr, "R"); kD, _ = kd(lr, "D")
        same = [f"종류 {kind}" for kind, k_ in (("D", kD), ("R", kR)) if k_ == K]
        ktxt = f"Δ = 0(선택 결과) 지역 R {kR}/{K}, D {kD}/{K}" + (f". 조정판 = 기본판(선택 결과, {', '.join(same)})" if same else "")
        stab_all.append((lr, txt, ktxt, same))
        V("LGF-N2s", "N2s", f"{txt} [{ktxt}]", "; ".join(f"{lab}: {s_}" for lab, s_ in items) + f" | {ktxt}", role="보조",
          blind=True, used=[], learner=lr, k_default=f"R {kR}/{K}, D {kD}/{K}")
    if stab_all:
        ok = all(t_ == "강건" for _, t_, _, _ in stab_all)
        head = ("초모수를 원천 교차검증으로 골라도 핵심 판정은 바뀌지 않았다" if ok else
                "핵심 판정이 바뀐 학습기가 있다: " + "; ".join(f"{lr} {t_}" for lr, t_, _, _ in stab_all if t_ != "강건"))
        ks = "; ".join(f"{lr} R {kd(lr, 'R')[0]}/{K}, D {kd(lr, 'D')[0]}/{K}" for lr, _, _, _ in stab_all)
        eqs = [f"{lr}({', '.join(sm)})" for lr, _, _, sm in stab_all if sm]
        tail = f"Δ = 0(선택 결과) 지역: {ks}" + (f". 조정판 = 기본판(선택 결과): {'; '.join(eqs)}" if eqs else "")
        V("LGF-N2s", "N2s", f"{head} [{tail}]", json.dumps({lr: t_ for lr, t_, _, _ in stab_all}, ensure_ascii=False), role="보조", blind=True,
          used=[], clause="전체")

    # ---------------- LGF-N3: 조정 신경망 − 조정 CatBoost(없으면 catboost_lo)
    for lr in a.LEARNERS:
        pair = (tu(lr), cmp_)
        base = [(f"R1 {_nt(n)} λ{LB:g}", C("LGF-N3", "aux", f"R1[{lr}*]-R1[{cmp_txt}]|n{_nl(n)}|lam{LB:g}", G("R1", n, tu(lr), LB), G("R1", n, cmp_, LB),
                                        pair=pair, n=n, lam=LB, primary=True, eq_test=True, role="주", learner=lr, k_default=kdt(lr, "R")))
                for n in (0, 10, ALL)]
        lam1 = [(f"R1 {_nt(n)} λ1", C("LGF-N3", "aux", f"R1[{lr}*]-R1[{cmp_txt}]|n{_nl(n)}|lam1", G("R1", n, tu(lr), 1.0), G("R1", n, cmp_, 1.0),
                                    pair=pair, n=n, lam=1.0, primary=False, eq_test=True, role="보조(λ 1.0)", learner=lr, k_default=kdt(lr, "R")))
                for n in (0, 10, ALL)]
        for n in (0, 10, ALL):
            if has_cbt:
                C("LGF-N3", "N3aux", f"R1[{lr}*]-R1[CB]|n{_nl(n)}", G("R1", n, tu(lr), LB), G("R1", n, "catboost_lo", LB), pair=(tu(lr), "catboost_lo"),
                  n=n, lam=LB, role="보조(CB)", learner=lr, k_default=kdt(lr, "R"))
            C("LGF-N3", "N3aux", f"R1[{lr}]-R1[CB]|n{_nl(n)}", G("R1", n, lr, LB), G("R1", n, "catboost_lo", LB), pair=(lr, "catboost_lo"), n=n, lam=LB,
              role="보조(로컬 기본판)", blind=False, learner=lr, prior_info="재현(비맹검, M1)")
            C("LGF-N3", "N3aux", f"D0[{lr}*]-D0[{cmp_txt}]|n{_nl(n)}", G("D0", n, tu(lr), 1.0), G("D0", n, cmp_, 1.0), pair=(tu(lr), cmp_), n=n, lam=1.0,
              role="보조(D0)", learner=lr, k_default=kdt(lr, "D"))
        eqt = "동등(한계 0.5 cm, 양쪽 조정)" if has_cbt else "동등(한계 0.5 cm, 조정 신경망 대 고정 초모수 CatBoost(반복 200, 깊이 3))"
        txt, br = eq_verdict(X, a, base, lam1, "조정 신경망의 오차가 작다(표로 보고)", "조정 신경망의 오차가 크다(표로 보고)", eqt)
        if not has_cbt:
            txt = f"주 비교 대상 catboost_lo({cmp_why}). 판정 문구는 조정 신경망 대 고정 초모수 CatBoost(반복 200, 깊이 3)로 한정한다. {txt}"
        V("LGF-N3", "aux", txt, X._fmt(base + lam1), role="보조", blind=True, used=base + lam1, learner=lr, branch=br, comparator=cmp_)

    # ---------------- LGF-N4(서술)
    desc = n4_rows(a, X, tms, dec, SEL, main_t, view, G, tu)
    T.rows += desc
    return T.frame()


def n4_rows(a, X, tms, dec, SEL, main_t, view, G, tu):
    """N4 서술 행: 선택 표 요약(기본값 우선 표지, sel_default, 민감도 선택의 변화), fold 학습 행 수, 원천 CV 이득과 대상 Δ(l* − l, n = 0)의
    순위 상관(검정 없음)."""
    rows = []
    LB = float(H.LAM_BASE)
    pts = []
    for lr in a.LEARNERS:
        for kind in KINDS:
            fl = Counter(); nsel0 = 0; chg = Counter()
            for t, m in main_t:
                d = dec.get((lr, kind, t, m)) or {}
                mn = d.get("main")
                if mn is None:
                    continue
                fl[mn.get("flag", "") or "선택"] += 1
                nsel0 += int(mn.get("trial") == 0)
                for crit in ("cw", "l25"):
                    if crit in d and d[crit].get("trial") != mn.get("trial"):
                        chg[crit] += 1
                q = SEL[(SEL.row_type == "decision") & (SEL.criterion == "main") & (SEL.learner == lr) & (SEL.kind == kind) & (SEL.target == t)
                        & (SEL["mode"] == m)] if len(SEL) else SEL
                gain = 0.0 if mn.get("trial") == 0 else (float(q.score_cstar.iloc[0]) - float(q.score_0.iloc[0]) if len(q) else np.nan)
                mth, lam = ("D0", 1.0) if kind == "D" else ("R1", LB)
                tm = view((tu(lr), lr)).get(f"{t}|{m}")
                s_ = X.region_stats(tm, G(mth, 0, tu(lr), lam), G(mth, 0, lr, lam)) if tm is not None else None
                pts.append(dict(learner=lr, kind=kind, target=t, gain=gain, delta=float(s_["delta"]) if s_ else np.nan))
            rows.append(dict(test_id="LGF-N4", item="N4", scope="descriptive", contrast=f"선택 표|{lr}|{kind}", learner=lr, kind=kind,
                             flags=json.dumps(dict(fl), ensure_ascii=False), n_sel_default=int(nsel0), n_changed_cw=int(chg["cw"]),
                             n_changed_l25=int(chg["l25"]), role="서술", primary=False, blind=True, platform=PLATFORM))
    fr = SEL[SEL.row_type == "fit"] if len(SEL) and "row_type" in SEL else pd.DataFrame()
    if len(fr):
        for (t, m), g in fr.groupby(["target", "mode"]):
            r0 = g.iloc[0]
            rows.append(dict(test_id="LGF-N4", item="N4", scope="descriptive", contrast=f"fold 구성|{t}|{m}", target=f"{t}|{m}",
                             fold_regions=r0.get("fold_regions", ""), fold_n_tr=r0.get("fold_n_tr", ""), fold_n_te=r0.get("fold_n_te", ""),
                             fold_n_tr_within100km=r0.get("fold_n_tr_within100km", ""), role="서술", primary=False, blind=True, platform=PLATFORM))
        g2 = fr.groupby(["learner", "kind", "trial"], as_index=False).agg(fit_s=("fit_s", "mean"), n_fit_rows=("fit_s", "size"))
        rows.append(dict(test_id="LGF-N4", item="N4", scope="descriptive", contrast="설정별 적합 시간(선택 단위 합의 평균, s)", role="서술",
                         table=g2.to_json(orient="records", force_ascii=False), primary=False, blind=True, platform=PLATFORM))
    pdf = pd.DataFrame(pts)
    if len(pdf):
        ok = pdf[np.isfinite(pdf.gain) & np.isfinite(pdf.delta)]
        rho = float(ok.gain.rank().corr(ok.delta.rank())) if len(ok) >= 3 else np.nan
        rows.append(dict(test_id="LGF-N4", item="N4", scope="descriptive", contrast="원천 CV 이득(seed 1·2) 대 대상 Δ(l* − l, n = 0)의 순위 상관",
                         spearman=rho, n_points=int(len(ok)), points=pdf.to_json(orient="records", force_ascii=False), role="서술(검정 없음)",
                         primary=False, blind=True, platform=PLATFORM))
    return rows


def finish_tests(df, a, local_note="", interim=""):
    """판정 표 열 정리(명세 10): halfwidth, precision_ok, verdict4_aux, precision_note, eq_p, holm_family, flag_uncorr, confirmatory, platform."""
    if not len(df):
        return df
    for c_ in ("ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq", "p_eq", "holm_p", "verdict4", "verdict4_d10", "primary", "eq_test", "item", "verdict"):
        if c_ not in df:
            df[c_] = np.nan
    df["ci_lo_b"], df["ci_hi_b"] = pd.to_numeric(df.ci_lo_beq, errors="coerce"), pd.to_numeric(df.ci_hi_beq, errors="coerce")
    df["halfwidth"] = [halfwidth(r_) for r_ in df.to_dict("records")]     # 네 끝값이 모두 유한할 때만(h47 과 같은 정의)
    df["precision_ok"] = [bool(h <= float(a.delta_eq)) if np.isfinite(h) else np.nan for h in df.halfwidth.astype(float)]
    df["verdict4_aux"] = df.verdict4_d10
    df["precision_note"] = np.where((df.verdict4.astype(str) == "미결정") & (df.precision_ok == False), "정밀도 미달(동등성 판정 불가)", "")  # noqa: E712
    df["eq_p"] = df.p_eq
    prim = df.primary.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
    eqt = df.eq_test.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
    df["holm_family"] = np.where(prim, df.item.astype(str), "")
    df["holm_family_eq"] = np.where(prim & eqt, "eq", "")
    hp = pd.to_numeric(df.holm_p, errors="coerce")
    df["flag_uncorr"] = np.where(prim & (df.item.astype(str) == "aux") & df.verdict4.isin(["우세", "열세"]) & (hp >= 0.05), "보정 전 유의", "")
    sc = df.scope.astype(str) if "scope" in df else pd.Series([""] * len(df), index=df.index)
    df["confirmatory"] = df.test_id.isin(list(CONFIRMATORY_N)) & (((sc == "MEAN") & (df.holm_family == "conf")) | (sc == "verdict"))
    df["platform"] = PLATFORM
    df["verdict_text"] = df.verdict.where(df.verdict.notna(), "")
    df["local_note"] = local_note
    df["interim"] = interim
    front = ["test_id", "item", "contrast", "scope", "n", "lam", "pool_regions", "splits_min", "splits_expected", "delta", "ci_lo", "ci_hi", "ci_lo_b",
             "ci_hi_b", "halfwidth", "precision_ok", "verdict4", "verdict4_aux", "eq_p", "holm_family", "holm_p", "holm_p_eq", "flag_uncorr", "small_note",
             "confirmatory", "blind", "prior_info", "n_unpaired", "key_set", "k_default", "support_class", "platform", "verdict_text"]
    for c_ in front:
        if c_ not in df:
            df[c_] = np.nan
    return df[front + [c_ for c_ in df.columns if c_ not in front]]


def timing_n(runs):
    """적합 1건 시간(통과, 학습기, 판, 방법별). λ 행의 중복을 빼고 적합 시간이 있는 행만 센다."""
    if not len(runs) or "fit_s" not in runs:
        return pd.DataFrame()
    ml = runs[(runs.learner != "none") & (pd.to_numeric(runs.fit_s, errors="coerce") > 0)]
    ml = ml.drop_duplicates(subset=["target", "mode", "split", "pass", "learner", "method", "n", "draw", "seed"])
    if not len(ml):
        return pd.DataFrame()
    return ml.groupby(["pass", "learner", "arm", "method"], as_index=False).agg(n_fit=("fit_s", "size"), sec_mean=("fit_s", "mean"), sec_sum=("fit_s", "sum"))


def cross_tables_n(a, X, D, floor, stores):
    """교차 환경(보조): LG gpu 조각(읽기 전용)과 P0·P1 을 대조(허용 차 1e-9 cm, h43.gate_compare)한 뒤 통과한 단위에서 로컬 기본판과
    LG 기본판(학습기 이름 뒤에 '@lg')의 같은 키 대비를 낸다. 판정에 쓰지 않는다."""
    ref = {(s_["target"], s_["mode"], int(s_["split"]), s_["learner"]): s_ for s_ in X.find_shards_x(a.LGDIR / "shards", a.lg_tag) if s_["part"] == "gpu"}
    if not ref:
        return pd.DataFrame(), pd.DataFrame(), f"LG gpu 조각 없음({a.LGDIR / 'shards'}, tag {a.lg_tag})"
    gate, merged = [], {}
    for (nm, sp), st in sorted(stores.items()):
        t, m = nm.split("|")
        cp = None
        for lr in a.LEARNERS:
            s_ = ref.get((t, m, int(sp), lr))
            if s_ is None:
                continue
            row = dict(target=t, mode=m, split=int(sp), learner=lr)
            try:
                sb = load_stores(s_["npz"]).get((nm, int(sp)))
                if sb is None:
                    row.update(status="저장소 없음", passed=False); gate.append(row); continue
                row.update(T43.gate_compare(st, sb))
                row["status"] = "ok" if row["passed"] else "불일치"
                if row["passed"]:
                    if cp is None:
                        S, C_ = st.matrices(st.keys)
                        cp = BlockStore._from_arrays(st.target, st.split, st.blocks, st.ncell, list(st.keys), S, C_, dict(st.meta))
                    for k in sb.keys:
                        if k[1] == lr and k[0] in ("D0", "R1") and str(k[2]) == "1" and k[3] == "cell":
                            cp.add_sse((k[0], f"{lr}@lg") + tuple(k[2:]), *sb.get(k))
            except Exception as e:                                        # noqa: BLE001
                row.update(status=f"오류: {repr(e)[:120]}", passed=False)
            gate.append(row)
        if cp is not None:
            merged[(nm, int(sp))] = cp
    gate = pd.DataFrame(gate)
    if not merged:
        return gate, pd.DataFrame(), "P0·P1 대조를 통과한 단위가 없다"
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in merged.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in merged})}
    ns = SimpleNamespace(nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux))
    T = X.TestBook(ns, tms, D, floor)
    LB = float(H.LAM_BASE)
    note = "교차 환경(보조): 하드웨어, torch·numpy 판과 교락된다. 두 플랫폼 값의 차를 학습기 차로 쓰지 않는다. 판정에 쓰지 않는다"
    for lr in a.LEARNERS:
        for n in (0, 10, 40, 160, -1):
            T.contrast("LGF-cross", "cross", f"R1[{lr}]-R1[{lr}@lg]|n{_nl(n)}", X.G("R1", n, LB, lr), X.G("R1", n, LB, f"{lr}@lg"), n=n, lam=LB,
                       primary=False, blind=False, role="교차 환경(보조)", note=note, aux3=False)
            T.contrast("LGF-cross", "cross", f"D0[{lr}]-D0[{lr}@lg]|n{_nl(n)}", X.G("D0", n, 1.0, lr), X.G("D0", n, 1.0, f"{lr}@lg"), n=n, lam=1.0,
                       primary=False, blind=False, role="교차 환경(보조)", note=note, aux3=False)
    return gate, T.frame(), ""


def summarize(a):
    """집계(로컬, 프로세스 1개). 허용 표지(--allow-local)가 없으면 재표집 1,000 이하와 '판정에 쓰지 않음' 표지. 본 실행 중(창 마감 전이고 기본
    범위 미완료)에는 거부하고, --force-interim 이면 '중간 집계' 표지를 붙인다. 반환 dict(n_fail)."""
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    w = read_window(a)
    src = adopt_window(a, w)                                              # 창 파일의 E1·유효 축소로 k, 추출, 분할, 2순위 대상을 맞춘다
    print(f"[summarize] 설정 출처 {src}: E1 {int(a.E1)} · 축소(N) {a.REDUCE_N or '없음'} · k {a.N_RANDOM} · 통과 1 추출 {a.DRAWS_P1} · "
          f"통과 1 분할 {a.P1_SPLITS} · 2순위 {a.T2 or '없음'} · 스레드 {a.threads}", flush=True)
    ensure_configs(a)
    local_note = ""
    if not a.allow_local:
        a.nboot = min(int(a.nboot), 1000)
        local_note = "판정에 쓰지 않음(허용 표지 없는 로컬 집계: 스레드 1, 재표집 1,000 이하)"
    D = get_data(a)
    dl = _parse_time(w.get("deadline"))
    miss = scope_missing(a, D)
    interim = ""
    if dl is not None and time.time() < dl and miss:
        if not a.force_interim:
            print(f"[summarize] 거부: 본 실행 중이다(창 마감 전, 기본 범위 미완료 {len(miss)}개. 예: {miss[:3]}). "
                  "--force-interim 을 주면 '중간 집계' 표지를 붙여 돌리고 개정 이력에 기록한다", flush=True)
            return dict(n_fail=0, refused=True)
        interim = "중간 집계(개정 이력에 기록한다)"
    sh = find_shards_n(a)
    if not sh:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{a.TAG}__*", flush=True)
        return None
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    cfg_info = check_cfg_n(a, sh, units)
    runs = read_runs_n(sh)
    stores = load_stores([s_["npz"] for s_ in sh])
    X = _load_h42()
    floor, floor_meta = X.floor_table(D.df)
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in stores})}
    ns = SimpleNamespace(delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux), nboot=int(a.nboot))
    curve_failed: list = []
    cur = T43.curve_t(ns, X, D, tms, runs, floor, curve_failed)
    SEL = read_select(a)
    dec_tab = decisions_table(a, SEL)
    dec, dec_problems = shard_decisions(sh, units, dec_tab)
    if dec_problems:
        msg = f"조각에 기록된 선택과 선택 표가 맞지 않는다({len(dec_problems)}건, 예: {dec_problems[:3]})"
        if not a.allow_mixed_cfg:
            raise SystemExit("[summarize] " + msg + ". 창 파일의 E1·축소와 선택 표(sel_sig)를 확인한다(--allow-mixed-cfg 로 진행하면 조각의 선택을 쓴다)")
        print("  [warn] " + msg + ". 조각에 기록된 선택으로 k_default 와 보조 층화 평균을 계산한다", flush=True)
    cbt_on = w.get("cbt_enabled") if "cbt_enabled" in w else None
    tests = finish_tests(build_tests_n(a, tms, D, floor, X, dec, SEL, cbt_enabled=cbt_on), a, local_note, interim)
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()
    gate, cross, cross_note = pd.DataFrame(), pd.DataFrame(), "교차 비교를 켜지 않았다(--cross-lg)"
    if a.cross_lg:
        gate, cross, cross_note = cross_tables_n(a, X, D, floor, stores)
        if cross_note:
            print(f"[summarize] 교차 비교: {cross_note}", flush=True)
    bad = runs.fit_flag.astype(str).map(lambda v: any(q in ("fail", "nonfinite", "copy_missing") for q in v.split(";"))) if len(runs) else []
    failed = runs[(runs.n_nonfinite > 0) | bad] if len(runs) else runs
    tt = timing_n(runs)
    O = a.OUT
    _atomic_csv(cur, O / f"{a.TAG}_curve.csv")
    _atomic_csv(tests, O / f"{a.TAG}_tests.csv")
    _atomic_csv(tt, O / f"{a.TAG}_timing.csv")
    _atomic_csv(failed, O / f"{a.TAG}_failed.csv")
    if a.cross_lg:
        _atomic_csv(cross, O / f"{a.TAG}_cross.csv")
        _atomic_csv(gate, O / f"{a.TAG}_cross_gate.csv")
    status = Counter(str(u.get("status", "ok")) for u in units)
    n_fail = int(len(failed)) + int(sum(v for k, v in status.items() if k != "ok")) + int(len(curve_failed))
    meta = dict(stage="H48/LGF-N", plan="docs/EXPERIMENT_PLAN_LGF_2026-09-29.md 4절(개정 1)", tag=a.TAG, n_units=len(units), unit_status=dict(status),
                shard_cfg=cfg_info, n_rows=int(len(runs)), n_curve=int(len(cur)), n_tests=int(len(tests)), n_failed_keys=int(len(failed)),
                curve_failed=curve_failed, n_fail=n_fail, floor=floor_meta, nboot=int(a.nboot), delta_eq=float(a.delta_eq),
                delta_eq_aux=float(a.delta_eq_aux), local_note=local_note, interim=interim, scope_missing=miss[:200], n_scope_missing=len(miss),
                window=w, cross_note=cross_note, code_sha=a.CODE_SHA, revision_needed=a.REVISION_NEEDED, platform=PLATFORM,
                sel_sig={lr: sel_sig(a, lr) for lr in LEARNERS_ALL}, cfg_sig=a.CFG_SIG, comparator_n3=comparator_n3(cbt_on)[0],
                cbt_enabled=cbt_on, settings=dict(E1=bool(a.E1), reduce_n=list(a.REDUCE_N), k=dict(a.N_RANDOM), draws_p1=int(a.DRAWS_P1),
                                                  p1_splits=list(a.P1_SPLITS), t2=[f"{t}|{m}" for t, m in a.T2], source=src),
                decision_problems=dec_problems[:50], n_decision_problems=len(dec_problems),
                rules=dict(verdict4="우세 = 두 가중의 CI 상한 < 0, 열세 = 두 가중의 CI 하한 > 0, 동등 = 네 끝값의 절댓값 ≤ 0.5 cm, 그 밖은 미결정",
                           precision="반폭 = max(셀 가중, 블록 등가중 CI 폭)/2 > 0.5 cm 이고 미결정이면 '정밀도 미달(동등성 판정 불가)'",
                           holm="conf = N1 4개, aux = N2 기준 λ 24 + N3 12(p_boot), eq = 같은 대비의 동등성 p. 판정에 쓰지 않는다",
                           pair="학습기 짝 대비는 두 쪽에 모두 있는 키만(n_unpaired)", confirmatory="LGF-N1(분할 미완결이면 판정 불가)"),
                summarize_s=round(time.time() - t0, 1))
    H._atomic_text(O / f"{a.TAG}_meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=_pyval))
    print(f"[summarize] 조각 {len(sh)} · runs {len(runs):,} · curve {len(cur):,} · tests {len(tests):,} · 실패 {n_fail} · {time.time() - t0:.0f}s → {O}/{a.TAG}_*"
          + (f" · {local_note}" if local_note else "") + (f" · {interim}" if interim else ""), flush=True)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])] if len(tests) and "scope" in tests else tests
    if len(v):
        print(v[[c_ for c_ in ("test_id", "role", "learner", "verdict") if c_ in v]].to_string(index=False), flush=True)
    return dict(n_fail=n_fail, tests=tests, curve=cur, meta=meta)


if __name__ == "__main__":
    _res = main()
    sys.exit(exit_code_n(_res))
