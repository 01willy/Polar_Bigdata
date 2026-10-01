"""H47 · 표형 파운데이션 모델 확장(LGF 의 F 부분). 계획 docs/EXPERIMENT_PLAN_LGF_2026-09-29.md §2–§3, §6–§8(개정 1, 사전 등록)의 구현.

목적
  TabICL v2(tabicl 2.0.2, 회귀 가중치 tabicl-regressor-v2-20260212.ckpt)를 LGT(h43)와 같은 컨텍스트 규약으로 D0·R0·R1 에 적용하고,
  같은 컨텍스트 행으로 적합한 CatBoost(catboost_ctx)와 조각 안에서 짝짓는다. 원천 전체를 컨텍스트로 쓰는 full 변형(D0@full, R1@full)을 둔다.
  TabPFN v2 와의 대비(F3)는 LGT 조각을 읽기 전용으로 짝짓는다(단위마다 ctx_sha 전부 일치 등 다섯 조건, lgf_gate.csv).

동결 규칙
  h40, h42, h43 과 src/polar 의 기존 모듈을 고치지 않는다. h40 은 'h40_label_grid', h43 은 'h43_tabpfn_label_grid' 로 모듈 최상위에서
  파일 경로로 읽는다(spawn 워커가 주 모듈을 다시 읽을 때도 등록된다). h42 는 집계 함수 안에서만 읽는다.
  컨텍스트 구성은 h43 의 함수(ctx_order, src_budget, context_rows, build_X, build_y, coef_n, ctx_sha)를 그대로 부른다(새로 구현하지 않는다).
  GPU 확인(query_gpu_info, gpu_apps_by_index, screen_gpus, gpu_guard_t), 잠금(acquire_lock, release_lock), pdeathsig, 풀 재생성의
  원인 판별(culprit_pids, triage_broken), CUDA 메모리 부족 판별(is_cuda_oom, _free_cuda)도 h43 의 것을 쓴다.
  판 고정: h43 SHA-1 13148bb899e92c49ba24f2f9cb6463e56da26a2a(커밋 45ef3b6, LGT 본 실행 판), h40 SHA-1 ffd0b3a76d36472cef15ac625eeeee344f948a8f.
  다르면 실행하지 않는다(SystemExit). --h43-sha, --h40-sha 로 새 값을 주면 받아들이고 조각 설정에 '개정 필요' 표지를 남긴다. h42 는 기록만 한다.

학습기
  tabicl        TabICLRegressor(n_estimators 8, norm_methods None, feat_shuffle_method 'latin', outlier_threshold 4.0, batch_size 8,
                model_path = 가중치 절대 경로, allow_auto_download False, checkpoint_version 'tabicl-regressor-v2-20260212.ckpt', device 'cuda',
                use_amp·use_fa3·offload_mode 'auto', random_state = 학습기 seed, n_jobs None, verbose False). 입력은 열 이름이 h40.FEATS 인
                DataFrame(float32, 결측 NaN). tabicl 2.0.2 는 numpy 입력의 NaN 을 받지 않는다. DataFrame 경로는 수치 열의 결측을 컨텍스트 평균으로
                채운다(TransformToNumerical 의 SimpleImputer). 예측은 predict(output_type='mean'). 가중치는 TabICLCached(_load_model 만 바꾼 하위
                클래스)가 워커마다 한 번 읽은 체크포인트 사전으로 모델을 만든다(순서는 2.0.2 의 _load_model 과 같다). --tabicl-stock 이면 원래 클래스.
  catboost_ctx  h43.cb_fit_predict(반복 200, 학습률 0.05, 깊이 3, l2 3, random_seed = 학습기 seed, Pool thread_count = --threads).

축, 방법, 키
  main(tag lgf)  컨텍스트 상한 10,000행(h43 의 6C.4 규칙 그대로), 방법 D0·R0·R1. n = 0 의 R1 은 R0 의 적합을 다시 쓴다.
  full(tag lgfs) 상한 없음(원천 전부를 ctx_order 순서로, 대상 라벨 행 1배), 방법 D0@full·R1@full. n = 0 의 R1@full 은 E0 앵커로 적합한다.
  키 = (method, learner, '1', 'cell', n, draw, seed, lam). P0 = h40.P0_KEY, P1 = ('P1', 'none', '1', 'cell', n, d, −1, 0.0). P0 은 모든 조각에 둔다.
  λ ∈ {0.25, 0.5, 1.0} 은 같은 적합에서 사후 적용한다(기준 λ: D0 1.0, R0·R1 0.25).

작업(--jobs)과 단위
  단위 = (작업, 축, 대상, 모드, 분할, 부분). 부분 n0 = n 0, pos = 격자의 n > 0, all = 격자 전체.
  f_n0   main·full, 주 4지역 x, n0                                    (J1)
  f_t1   main, 주 4지역 x, pos                                         (J6)
  f_full full, 주 4지역 x, pos(n ∈ {10, 40, 160, 전량}, 추출 2)          (J8, S2 이면 건너뛴다)
  f_t2ak main, Alaska x, all                                           (J9)
  f_t2   main, Russia_C·Greenland·AL-1–AL-3·CA-2·CA-3 x, all            (J11, S3 이면 추출 2, S4 이면 건너뛴다)
  f_t3   main, LGT 주 설정 27대상에서 학습기 축 12대상을 뺀 15대상, all   (J12, 명시할 때만. 창 조건을 채울 때만 돈다)
  유효 분할은 h43.enumerate_t 와 같은 규칙이다. 우선순위는 작업 순서, 대상 순서, 분할, 축 순이다.

조각(<out-dir>/shards)
  <tag>__gpu__<대상>__<모드>__s<분할>__<부분>_{runs.csv, blocksse.npz, cells.npz, unit.json}. 쓰는 순서는 runs → blocksse → cells → unit 이고
  모두 임시 파일 뒤 os.replace 다. 단위를 시작할 때 이전 unit.json 과 cells.npz 를 지운다. status: ok, partial, failed(학습기 저장 키 0 또는
  실패 비율 > 0.2). 같은 학습기의 연속 실패 5회는 단위 중단(unit.json 없음). runs.csv 의 열 = h43.RUN_COLS + amp_used, fa3_flag, has_fa3,
  offload_mode, n_allnan_cols, gpu_mem_fit_mib, rss_mib.
  스모크(--smoke, tag lgf_smoke·lgfs_smoke)와 사전 점검(--precheck, tag lgf_pre·lgfs_pre)은 BlockStore 와 cells 를 저장하지 않고 runs 의
  rmse_cm, rmse_beq_cm, bias_cm 을 NaN 으로 쓴다. 곡선, 최소 n, 판정 표를 쓰지 않고 RMSE, Δ, 판정을 화면에 내지 않는다.

실행 제어(공유 서버 규칙, 계획서 §6)
  학습을 하는 실행은 --allow-local 과 --gpus 가 모두 필요하다. GPU 8 은 항상 거부, GPU 0·1 은 --allow-gpus-01 이 있어야 하고, 후보
  (2, 3, 4, 5, 6, 7, 9) 밖의 번호는 거부한다. 실행 직전 nvidia-smi 로 메모리 ≤ --gpu-mem-max-mib(50)이고 계산 프로세스가 없는 GPU 만 쓴다.
  GPU 6·7·9 는 LGT 잠금(data/processed/lgt/run_lgt/lock.json)의 PID 가 살아 있으면 뺀다. 워커는 단위마다 h43.gpu_guard_t 로 다시 확인한다.
  스레드는 프로세스당 1–2(OMP·MKL·OpenBLAS·NUMEXPR, torch intra, CatBoost), torch inter-op 1, nice 10 이상.
  시작 전 1분 load average 가 --load-max-start(64)를 넘으면 거부하고, 실행 중 300 s 마다 보아 --load-max-run(96)을 넘으면 새 단위 배정을 보류한다.
  드레인: 배정 전마다 <out>/run_lgf/drain(또는 run_lgfs/drain)을 본다. 있으면 배정을 멈추고 진행 중 단위를 끝낸 뒤 drain 을 지우고 종료 코드 3.
  창: 본 실행에서 드레인·창 마감·load 판정을 모두 통과한 첫 배정 직전에 lgf_window.json 에 t0 가 없으면 t0 와 마감(t0 + 48 h)을 쓴다(load 보류
  중에는 창을 시작하지 않는다). 마감이 지나면 배정을 멈추고 종료 코드 4(마감 뒤에 본 실행을 시작해도 4). 창 파일과 lgf_projection.csv 는
  잠금(lgf_window.json.lock, 내용 = PID, 죽은 PID 의 잠금은 바로 지운다) 아래에서 고친다.
  풀: GPU 하나에 ProcessPoolExecutor(워커 하나)를 따로 둔다. 한 GPU 의 워커가 점유를 확인해(h43.gpu_guard_t 의 os._exit) 끝나거나 비정상 종료해도
  그 GPU 의 단위만 되돌리거나 격리하고 나머지 GPU 는 계속 돈다. 그 GPU 는 rescreen 을 통과하면 풀을 다시 만들고 아니면 뺀다.
  종료 코드: 중단 130, 실패 1, 드레인 3, 창 마감 4, partial 만 있으면 2, 그 밖 0.

집계(--summarize-only)
  lgf, lgfs 조각을 (대상, 모드, 분할)별 BlockStore 로 합친다. LGT 짝 게이트(조각 상태, 설정, P0·P1 1e-9 cm, 키별 ctx_sha 전부 일치, 블록 일치)를
  통과한 단위의 tabpfn 키(D0, R0, R1)만 병합한다. catboost_ctx 는 LGT 와 키별로 대조한다(0.02 cm, 표시만).
  산출: lgf_curve.csv, lgf_minn.csv, lgf_tests.csv(LGF-F1–F6), lgf_gate.csv, lgf_missing_desc.csv, lgf_timing.csv, lgf_failed.csv,
  lgf_targets.csv, lgf_meta.json. 본 실행 중(창 마감 전이고 기본 범위 미완료)에는 --force-interim 없이 거부한다('중간 집계' 표지).
  --allow-local 없이 집계하면 스레드 1, 재표집 1,000 이하로 돌리고 '판정에 쓰지 않음' 표지를 붙인다.

명세(impl_spec)와 다르게 구현한 점, 조작적 정의
  1. 시험 파일 이름은 tests/test_h47_fm.py 다(과제 지시). 명세는 tests/test_h47_foundation.py 였다.
  2. Holm 묶음의 열: holm_family 는 부트스트랩 p 의 묶음(conf 또는 aux)이고, 동등성 p 의 묶음(eq)은 holm_family_eq 열에 'eq' 로 적는다.
     한 대비가 aux 와 eq 두 묶음에 모두 들기 때문이다. 묶음 구성은 층화 평균(scope MEAN) 행만이다. 지역 행과 3지역 행은 묶음에 넣지 않는다.
  3. L1 형식 지역 수: 지역 행의 4분 판정이 '우세'(셀 가중과 블록 등가중의 CI 상한이 모두 0 미만)인 지역을 n ∈ {0, 3, 10, 40} 에 걸쳐 센다.
  4. 정밀도 표기 'k/m' 의 m 은 그 판정에 쓴 대비의 수다('차이를 확인하지 못함' 갈래는 기준 λ 와 λ 1.0 대비를 모두 센다).
  5. lgf_missing_desc.csv: 셀 수는 사용 분할의 부분집합 셀 수 합이다. 합이 30 미만이면 점 추정을 비운다. 층화 평균 행은 비지 않은 지역만의 평균이다.
  6. 게이트 조건 2 의 중복 배수(dup): LGT 조각의 설정에는 dup 이 없다. LGT 주 설정(axis main)은 h43.MAIN_SET 의 dup 1 이므로 그 값을 쓴다.
     n 격자는 LGT 의 n_grid 와 LGF 주 설정 축 전체 격자(n_grid_axis)를 비교한다(LGF 조각은 부분 격자를 가진다).
  7. f_t3 의 조건 'J1–J11 이 모두 끝났다' 가운데 N 작업(h48, J2–J5, J7, J10)의 완료는 h47 이 확인할 수 없으므로 --n-jobs-done 으로 호출자가
     확인해 준다. F 작업(J1, J6, J8, J9, J11)의 완료는 조각으로 확인한다.
  8. 본 실행은 창 파일에 B 결정(--window-set)이 없으면 거부한다(사전 점검 개정의 결정이 본 실행보다 앞서야 한다). --reduce 는 창 파일의
     유효 축소(reduce − restored) 가운데 F 에 관계된 것(S2, S3, S4)과 같아야 한다.
  9. --window-set 과 --gpu-event 는 --viewing-state(열람 상태 문자열)가 없으면 거부한다(계획서 §5: 결정은 열람 상태와 함께 기록한다).
  10. 종료 코드의 우선순위: 중단 130 > 실패 1 > 드레인 3 > 창 마감 4 > partial 2 > 0.
  11. offload 방식의 실제 적용 값은 tabicl 의 InferenceManager._resolve_offload_mode 의 반환값을 기록하는 감싸기 함수로 읽는다(반환값은
      그대로 돌려준다. 동작은 바뀌지 않는다). 적합 한 건에서 결정된 방식의 집합을 ';' 로 적는다.
  12. 스모크의 ctx_sha 독립 재계산: main 은 h43.run_ctx_t 를 대체 학습기로 돌려 h43 경로의 ctx_sha 를 얻고, full 은 h43 의 컨텍스트 함수로
      다시 만든다.
  13. 스모크 점검(4)(CatBoost Pool 경로와 h40.cb_fit)은 h40.cb_fit 에 thread_count = --threads 인 학습 Pool 을 넘겨 모형 정의를 대조한다
      (numpy 배열을 넘기면 catboost 가 학습 Pool 을 모든 코어로 만든다). 점검의 오류는 적합 실패로 세지 않고 점검 행에 남긴다.
  14. --window-set: 창이 시작된 뒤(t0 있음)에는 B, E1, cbt, S7 을 바꾸지 않고(거부), 복원 기록(restored)은 적용 중인 축소에 대해서 남긴다.
  15. lgf_projection.csv 의 열 형식은 h48 과 같다(script, row, est_gpu_h). F 행: J1, J6, J8, J9, J11, J12, P_F(기본 범위 합), S2(= J8),
      S3(J11 의 추출 5 → 2 로 줄어드는 양), S4(S3 뒤 남은 J11). N 행(h48): J2–J5, J7, J10, C1, P_N, E1, S1, S5, S6, S7.
      --window-status 의 복원 권고는 이 S 행을 쓴다. f_t3 와 복원 권고의 GPU 수에는 h48 워커 기록(run_lgfn/worker__*.json)의 살아 있는 GPU 를 더한다.
  16. 정밀도 표기와 반폭의 정의는 h48 과 같다: 반폭은 네 끝값이 모두 유한할 때만, '(정밀도 미달 대비 k/m)' 는 k ≥ 1 일 때만.

확인 범위
  구현 단계: py_compile, pyflakes, 스레드 1개의 --count-only. 수정 단계(검증 지적 반영): 단위 시험(tests/test_h47_fm.py, 스레드 2, GPU 없음)과
  GPU 스모크 1회(빈 후보 GPU 한 장). 사전 점검, 본 실행, 여러 GPU 의 풀 재생성과 드레인의 실제 동작은 확인하지 못했다.

실행(ROOT, PY=.venv_lgf/bin/python)
  적합 수:   $PY scripts/3_deep_learning/h47_foundation_models.py --count-only --dry-build --threads 1
  스모크:    nice -n 10 $PY scripts/3_deep_learning/h47_foundation_models.py --smoke --allow-local --gpus <G> --threads 2
  사전 점검: nice -n 10 $PY scripts/3_deep_learning/h47_foundation_models.py --precheck --allow-local --gpus <G> --threads 2
  결정 기록: $PY scripts/3_deep_learning/h47_foundation_models.py --window-set --B <B> --e1 <0|1> --cbt <0|1> --reduce <목록> --viewing-state '<상태>'
  본 실행:   nice -n 10 $PY scripts/3_deep_learning/h47_foundation_models.py --jobs f_n0 --allow-local --gpus <G> --threads 2 --resume --no-summarize
  집계:      $PY scripts/3_deep_learning/h47_foundation_models.py --summarize-only --allow-local --threads 2
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import gc
import json
import math
import multiprocessing
import os
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
NICE_MIN = 10


def _peek_threads(default=THREADS_DEFAULT):
    """numpy 를 부르기 전에 스레드 수를 정한다(h43._peek_threads 와 같은 방식). 1–2 로 자르고, --allow-local 없는 집계는 1 이다."""
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
    if "--summarize-only" in av and "--allow-local" not in av:
        n = 1
    return str(n)


_THREADS = _peek_threads()
for _v in THREAD_VARS:
    os.environ[_v] = _THREADS                       # setdefault 가 아니라 대입이다
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HUB_OFFLINE"] = "1"                  # 가중치 내려받기 시도를 막는다(캐시와 명시 경로만 쓴다)


def ensure_nice(target=NICE_MIN):
    """프로세스의 niceness 를 target 이상으로 올린다. 반환 현재 값(바꿀 수 없으면 None)."""
    try:
        cur = os.nice(0)
        if cur < int(target):
            os.nice(int(target) - cur)
        return os.nice(0)
    except OSError:
        return None


ensure_nice()

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(name, filename):
    """h43._load_module 과 같은 방식: 파일 경로에서 모듈을 읽어 sys.modules 에 등록한다. 이미 있으면 그 모듈을 쓴다."""
    import importlib.util
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

from polar.h4_common import BlockStore, save_stores, load_stores                                     # noqa: E402
import polar.h4_common as H4                                                                         # noqa: E402

# ================================================================ 고정 설계값(계획서 §3, §6, §8. 바꾸면 사전 등록에서 벗어난다)
LEARNERS_F = ("tabicl", "catboost_ctx")
TI, CBX = LEARNERS_F
TP = T43.TP                                          # 'tabpfn'(LGT 조각에서 읽기만 한다)
AXES_F = ("main", "full")
AX_F = dict(main=dict(sfx="", draws=5, methods=("D0", "R0", "R1"), tag_sfx=""),
            full=dict(sfx="@full", draws=2, methods=("D0", "R1"), tag_sfx="s"))
FULL_SET = dict(T43.MAIN_SET, sfx="@full")
FULL_CTX = SimpleNamespace(ctx_max=10 ** 7, ctx_reserve=0, ctx_src_min=0)
FULL_POS_GRID = (10, 40, 160, -1)
PARTS = ("n0", "pos", "all")
JOBS = ("f_n0", "f_t1", "f_full", "f_t2ak", "f_t2", "f_t3")
JOB_J = dict(f_n0="J1", f_t1="J6", f_full="J8", f_t2ak="J9", f_t2="J11", f_t3="J12")
DEFAULT_JOBS = "f_n0,f_t1,f_full,f_t2ak,f_t2"
MAIN4_X = [(t, "x") for t in H.MAIN4]
T2_TARGETS = [("Russia_C", "x"), ("Greenland", "x"), ("AL-1", "x"), ("AL-2", "x"), ("AL-3", "x"), ("CA-2", "x"), ("CA-3", "x")]
T3_TARGETS = [p for p in H.default_targets(["i", "x"]) if p not in set(H.LEARNER_TARGETS)]          # 15대상
REDUCE_ALL = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
REDUCE_F = ("S2", "S3", "S4")
ALLOWED_GPUS = (2, 3, 4, 5, 6, 7, 8, 9)   # LGF 개정 7(2026-10-01): GPU 8 추가
GPU01 = (0, 1)
NEVER = ()   # LGF 개정 7(2026-10-01): GPU 8 을 거부 목록에서 뺐다(허용 목록과 함께)
LGT_GPUS = (6, 7, 9)
PIN_H43 = "13148bb899e92c49ba24f2f9cb6463e56da26a2a"
PIN_H40 = "ffd0b3a76d36472cef15ac625eeeee344f948a8f"
REF_H42 = "70ec084a1919d35a1526728957e3d20e206bfac6"
MODEL_PATH_DEFAULT = ("/home/willy010313/.cache/huggingface/hub/models--jingang--TabICL/snapshots/"
                      "4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt")
MODEL_SHA1 = "95c81f65469aae055d8b98993959c42f5110c58c"
CKPT_VERSION = "tabicl-regressor-v2-20260212.ckpt"
PKG_PINS = {"torch": "2.6.0", "numpy": "1.26.4", "pandas": "2.1.4", "scikit-learn": "1.3.0", "catboost": "1.2.10", "tabicl": "2.0.2"}
WINDOW_H = 48.0
LOAD_CHECK_S = 300.0
CACHED_TOL = 1e-6                                   # 스모크: 캐시 하위 클래스와 원래 클래스의 예측 차 허용(cm)
CHUNK_TOL = 1e-3                                    # 스모크: 두 묶음 예측과 한 번 예측의 차 허용(cm)
GATE_TOL = 1e-9                                     # LGT 짝: P0·P1 의 셀 가중 RMSE 차 허용(cm)
CB_TOL = 0.02                                       # LGT 짝(보조): catboost_ctx 의 키별 RMSE 차 허용(cm)
MISS_MIN_CELLS = 30
PLATFORM = "local-3090-torch2.6.0-numpy1.26.4"
FAIL_STREAK_MAX = T43.FAIL_STREAK_MAX
FAIL_RATIO_MAX = T43.FAIL_RATIO_MAX
EXIT_FAIL, EXIT_PARTIAL, EXIT_DRAIN, EXIT_WINDOW, EXIT_INTERRUPT = 1, 2, 3, 4, 130
EXTRA_COLS = ["amp_used", "fa3_flag", "has_fa3", "offload_mode", "n_allnan_cols", "gpu_mem_fit_mib", "rss_mib"]
RUN_COLS_F = list(T43.RUN_COLS) + EXTRA_COLS
TRACE = None                                        # 시험용: 리스트를 넣으면 모든 적합의 컨텍스트 행렬과 정보를 기록한다
_FORCE_STOCK = [False]                              # 스모크 점검에서 캐시 하위 클래스의 차가 허용을 넘으면 워커 안에서 원래 클래스로 바꾼다
_OFFLOAD_SEEN: set = set()
_CKPT: dict = {}
_CACHED_CLS = None
MIB = 1024.0 ** 2
ITEM = "LGF"
TEST_COLS = ["test_id", "item", "contrast", "scope", "n", "lam", "pool_regions", "splits_min", "splits_expected", "delta", "ci_lo", "ci_hi",
             "ci_lo_b", "ci_hi_b", "halfwidth", "precision_ok", "verdict4", "verdict4_aux", "eq_p", "holm_family", "holm_p", "holm_p_eq",
             "flag_uncorr", "small_note", "confirmatory", "blind", "prior_info", "n_unpaired", "key_set", "k_default", "support_class", "platform",
             "verdict_text"]


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H47 표형 파운데이션 모델 확장(LGF-F)")
    ap.add_argument("--jobs", default=DEFAULT_JOBS, help="쉼표 목록: f_n0, f_t1, f_full, f_t2ak, f_t2, f_t3(f_t3 은 명시할 때만)")
    ap.add_argument("--reduce", default="", help="적용한 축소 단계(S2, S3, S4). 본 실행은 창 파일과 같아야 한다. --window-set 에서는 S1–S7")
    ap.add_argument("--targets", default="", help="시험용: '이름:모드' 쉼표 목록으로 대상을 제한한다")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--n-grid", default=T43.FULL_GRID)
    ap.add_argument("--draws", type=int, default=0, help="0 = 축 기본값(main 5, full 2). 주면 기본값과 비교해 작은 쪽")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lams", default="0.25,0.5,1.0")
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--ctx-max", type=int, default=10000)
    ap.add_argument("--ctx-reserve", type=int, default=1000)
    ap.add_argument("--ctx-src-min", type=int, default=1000)
    ap.add_argument("--n-est", type=int, default=8, help="TabICL n_estimators")
    ap.add_argument("--tabicl-batch-size", type=int, default=8)
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--model-sha1", default=MODEL_SHA1, help="가중치 파일의 SHA-1. 다르면 실행하지 않는다")
    ap.add_argument("--tabicl-stock", action="store_true", help="가중치 캐시 하위 클래스 대신 원래 TabICLRegressor 를 쓴다")
    ap.add_argument("--pred-chunk", type=int, default=4096)
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--gpus", default="", help="쉼표 목록. 학습 실행에 필수. 후보 2, 3, 4, 5, 6, 7, 9(0·1 은 --allow-gpus-01)")
    ap.add_argument("--allow-gpus-01", action="store_true")
    ap.add_argument("--gpu-mem-max-mib", type=int, default=50)
    ap.add_argument("--threads", type=int, default=THREADS_DEFAULT)
    ap.add_argument("--allow-local", action="store_true", help="학습을 하는 실행(스모크, 사전 점검, 본 실행)의 허용 표지")
    ap.add_argument("--load-max-start", type=float, default=64.0)
    ap.add_argument("--load-max-run", type=float, default=96.0)
    ap.add_argument("--out-dir", default="data/processed/lgf")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--tag", default="lgf", help="주 설정 tag. full 은 tag 뒤에 s 를 붙인다(lgfs)")
    ap.add_argument("--lgt-dir", default="data/processed/lgt")
    ap.add_argument("--lgt-tag", default="lgt")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--delta-eq", type=float, default=0.5)
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--h43-sha", default="", help="고정 판과 다른 h43 을 쓸 때의 SHA-1(개정 필요 표지)")
    ap.add_argument("--h40-sha", default="", help="고정 판과 다른 h40 을 쓸 때의 SHA-1(개정 필요 표지)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rerun-partial", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--dry-build", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--no-cells", action="store_true")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--pool-retries", type=int, default=2)
    ap.add_argument("--force-interim", action="store_true", help="본 실행 중의 집계를 '중간 집계' 표지로 허용한다(개정 이력에 기록)")
    ap.add_argument("--n-jobs-done", action="store_true", help="f_t3 조건: 호출자가 N 작업(h48)의 완료를 확인했다")
    ap.add_argument("--window-set", action="store_true", help="창 파일에 사전 점검 개정의 결정(B, E1, cbt, 축소) 또는 복원(--restore)을 쓴다")
    ap.add_argument("--window-status", action="store_true")
    ap.add_argument("--gpu-event", default="", help="'<GPU>:<add|remove>' 를 창 파일에 기록한다")
    ap.add_argument("--B", type=float, default=None, help="--window-set: 예산 B(GPU-h)")
    ap.add_argument("--e1", choices=["0", "1"], default=None)
    ap.add_argument("--cbt", choices=["0", "1"], default=None)
    ap.add_argument("--restore", default="", help="--window-set: 되돌린 축소 단계(쉼표 목록)")
    ap.add_argument("--viewing-state", default="", help="--window-set, --gpu-event 의 열람 상태(LG, LGX, LGT, LGF 각각)")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(path, base=ROOT):
    return Path(path) if os.path.isabs(str(path)) else Path(base) / str(path)


def _tokens(txt):
    return [v.strip() for v in str(txt or "").split(",") if v.strip()]


def finalize(a):
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    a.SUFFIX = "_smoke" if a.smoke else ("_pre" if a.precheck else "")
    a.TAG = a.tag + a.SUFFIX
    a.MAIN_MODE = not (a.smoke or a.precheck or a.count_only or a.summarize_only)
    a.JOBS = _tokens(a.jobs)
    bad = [j for j in a.JOBS if j not in JOBS]
    if bad:
        raise SystemExit(f"알 수 없는 작업: {bad} (가능: {list(JOBS)})")
    a.REDUCE = _tokens(a.reduce)
    bad = [s_ for s_ in a.REDUCE if s_ not in REDUCE_ALL]
    if bad:
        raise SystemExit(f"알 수 없는 축소 단계: {bad} (가능: {list(REDUCE_ALL)})")
    a.RESTORE = _tokens(a.restore)
    if a.smoke:                                                     # 스모크: 대상 2, 분할 1, n {0, 10, 40, 전량}, 추출 1, seed 1, main·full
        a.splits = 1; a.n_grid = "0,10,40,all"; a.draws = 1; a.seeds = 1
        a.targets = a.targets or "Russia_W:x,Canada:x"
        a.nboot = min(int(a.nboot), 1000)
    if a.precheck:                                                  # 사전 점검: 추출 0, seed 0(단위는 precheck_units 가 정한다)
        a.splits = 5; a.draws = 1; a.seeds = 1
    a.threads_asked = int(a.threads)
    a.threads = max(1, min(int(a.threads), THREADS_MAX))
    a.LOCAL_SUMMARY = bool(a.summarize_only and not a.allow_local)
    if a.LOCAL_SUMMARY:                                             # 허용 표지 없는 집계: 스레드 1, 재표집 1,000 이하, 판정에 쓰지 않음
        a.threads = 1
        a.nboot = min(int(a.nboot), 1000)
    try:
        a.GPUS = sorted({int(g) for g in str(a.gpus).split(",") if g.strip() != ""}, reverse=True)
    except ValueError:
        raise SystemExit(f"--gpus 는 정수의 쉼표 목록이다: '{a.gpus}'")
    a.N_USER = H._n_list(a.n_grid)
    a.LAMS = [float(v) for v in _tokens(a.lams)]
    a.OUT = _abs(a.out_dir); a.PROC = _abs(a.data_dir); a.LGTDIR = _abs(a.lgt_dir)
    a.SHARDS = a.OUT / "shards"
    a.RUN_DIR = a.OUT / f"run_{a.TAG}"
    a.RUN_ID = ""
    a.MODEL = Path(str(a.model_path)).expanduser()
    if not a.MODEL.is_absolute():
        a.MODEL = ROOT / a.MODEL
    a.STORE = not (a.smoke or a.precheck)                            # 스모크·사전 점검은 BlockStore 와 cells 를 저장하지 않는다
    a.save_cells = bool(a.STORE and not a.no_cells)
    a.TARGET_FILTER = H._pairs(a.targets, ["i", "x"]) if a.targets else None
    a.INTERIM = False
    a.PINS = pin_info(a)
    a._a43 = {}
    return a


def axis_tag(a, axis):
    return f"{a.tag}{AX_F[axis]['tag_sfx']}{a.SUFFIX}"


def axis_draws(a, axis):
    d0 = int(AX_F[axis]["draws"])
    return min(int(a.draws), d0) if int(a.draws) > 0 else d0


# ================================================================ 판 고정과 환경
def pin_info(a, sha_fn=None):
    """h43, h40, h42 의 SHA-1 과 고정 판 대조 결과(거부는 check_pins 가 한다)."""
    sha_fn = sha_fn or T43.file_sha
    got43 = sha_fn(SCRIPT_DIR / "h43_tabpfn_label_grid.py")
    got40 = sha_fn(SCRIPT_DIR / "h40_label_grid.py")
    got42 = sha_fn(SCRIPT_DIR / "h42_label_grid_ext.py")
    want43 = str(getattr(a, "h43_sha", "") or PIN_H43)
    want40 = str(getattr(a, "h40_sha", "") or PIN_H40)
    rev = bool(want43 != PIN_H43 or want40 != PIN_H40)
    return dict(h43=got43, h40=got40, h42=got42, want_h43=want43, want_h40=want40, h42_ref=REF_H42, h42_same=bool(got42 == REF_H42),
                revision_needed=rev, note="개정 필요" if rev else "")


def check_pins(a, sha_fn=None):
    """h43·h40 판 고정. 다르면 SystemExit. 반환 판 정보(a.PINS 를 갱신한다)."""
    p = pin_info(a, sha_fn)
    if p["h43"] != p["want_h43"]:
        raise SystemExit(f"[거부] h43 의 SHA-1 이 고정 판과 다르다: {p['h43']} (기대 {p['want_h43']}). 바뀐 h43 을 쓰려면 LGF 개정으로 등록하고 "
                         "--h43-sha 로 새 값을 준다")
    if p["h40"] != p["want_h40"]:
        raise SystemExit(f"[거부] h40 의 SHA-1 이 고정 판과 다르다: {p['h40']} (기대 {p['want_h40']}). LGF 개정과 --h40-sha 가 필요하다")
    if not p["h42_same"]:
        print(f"[warn] h42 의 SHA-1 이 계획서 작성 때와 다르다: {p['h42']} (기록 {REF_H42}). 기록만 한다", flush=True)
    if p["revision_needed"]:
        print("[warn] 고정 판과 다른 h43·h40 을 쓴다. 조각 설정에 '개정 필요' 표지를 남긴다", flush=True)
    a.PINS = p
    return p


def check_env(ver_fn=None, need=None):
    """패키지 판 확인(importlib.metadata). torch 는 앞부분 일치, 나머지는 같은 판이어야 한다. 다르면 SystemExit."""
    ver_fn = ver_fn or T43.pkg_version
    bad = []
    for k in (need or list(PKG_PINS)):
        want, got = PKG_PINS[k], str(ver_fn(k))
        ok = got.startswith(want) if k == "torch" else got == want
        if not ok:
            bad.append(f"{k} {got} (기대 {want})")
    if bad:
        raise SystemExit("[거부] 실행 환경의 패키지 판이 다르다: " + "; ".join(bad) + ". PY=.venv_lgf/bin/python 으로 실행한다")
    return True


def pkg_versions():
    return {k: T43.pkg_version(k) for k in PKG_PINS}


def rss_mib():
    try:
        return round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 1)
    except Exception:                                                     # noqa: BLE001
        return float("nan")


def _torch():
    return sys.modules.get("torch")


def _cuda_reset():
    tc = _torch()
    try:
        if tc is not None and tc.cuda.is_available():
            tc.cuda.reset_peak_memory_stats()
    except Exception:                                                     # noqa: BLE001
        pass


def _cuda_peak():
    tc = _torch()
    try:
        if tc is not None and tc.cuda.is_available():
            return float(tc.cuda.max_memory_allocated()) / MIB, float(tc.cuda.max_memory_reserved()) / MIB
    except Exception:                                                     # noqa: BLE001
        pass
    return float("nan"), float("nan")


# ================================================================ 창 파일(lgf_window.json)
def _now_iso():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _to_ts(v):
    """창 파일의 시각(ISO 문자열 또는 epoch 초) → epoch 초. 읽을 수 없으면 None."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        d = _dt.datetime.fromisoformat(str(v))
        if d.tzinfo is None:
            d = d.astimezone()
        return float(d.timestamp())
    except ValueError:
        return None


def window_path(a):
    return Path(a.OUT) / "lgf_window.json"


def read_window(a):
    try:
        return json.loads(window_path(a).read_text())
    except (OSError, ValueError):
        return {}


def _write_json_excl(path, obj):
    """O_EXCL 로 만든 임시 파일에 쓴 뒤 os.replace 로 바꾼다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.{time.time_ns()}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def window_lock(a, timeout=60.0):
    """창 파일과 lgf_projection.csv 의 읽기-수정-쓰기 잠금(lgf_window.json.lock, O_EXCL, 내용 = PID). h48.window_lock 과 같은 파일과 규칙이다.
    잠금의 PID 가 살아 있지 않으면 바로 지운다. PID 를 읽을 수 없는 잠금(쓰는 도중)은 10 s 가 지나면 지운다."""
    p = window_path(a)
    p.parent.mkdir(parents=True, exist_ok=True)
    lk = p.with_name(p.name + ".lock")
    t_end = time.time() + float(timeout)
    while True:
        try:
            fd = os.open(str(lk), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            break
        except FileExistsError:
            try:
                txt = lk.read_text().strip()
                pid = int(txt) if txt else 0
                if (pid and not T43.pid_alive(pid)) or (not pid and time.time() - lk.stat().st_mtime > 10.0):
                    lk.unlink(missing_ok=True)                            # 죽은 프로세스의 잠금
                    continue
            except (OSError, ValueError):
                continue
            if time.time() > t_end:
                raise SystemExit(f"[거부] 창 파일 잠금을 얻지 못했다: {lk}")
            time.sleep(0.2)
    try:
        yield lk
    finally:
        try:
            if lk.exists() and lk.read_text().strip() == str(os.getpid()):
                lk.unlink()
        except OSError:
            pass


def update_window(a, fn, timeout=60.0):
    """창 파일의 읽기-수정-쓰기(window_lock 아래). h48 과 같은 파일을 쓰므로 잠금을 둔다. 반환 새 내용."""
    p = window_path(a)
    with window_lock(a, timeout):
        w = read_window(a)
        base = dict(window_h=WINDOW_H, reduce=[], restored=[], gpu_events=[], decisions=[])
        new = fn(copy.deepcopy({**base, **w}))                            # 깊은 복사: 목록에 덧붙이기만 하는 변경(--gpu-event)도 새 내용으로 비교된다
        if new is not None and new != w:
            _write_json_excl(p, new)
        return new if new is not None else w


def window_start(a, now=None):
    """본 실행의 첫 단위 배정 직전: t0 가 없으면 t0 = now, deadline = t0 + 48 h 를 쓴다. 있으면 읽기만 한다."""
    now = time.time() if now is None else float(now)

    def fn(w):
        if w.get("t0"):
            return None
        t0 = _dt.datetime.fromtimestamp(now).astimezone()
        w.update(t0=t0.isoformat(timespec="seconds"), deadline=(t0 + _dt.timedelta(hours=WINDOW_H)).isoformat(timespec="seconds"),
                 window_h=WINDOW_H)
        return w
    return update_window(a, fn)


def window_deadline_passed(w, now=None):
    dl = _to_ts((w or {}).get("deadline"))
    return bool(dl is not None and (time.time() if now is None else float(now)) >= dl)


def effective_reduce(w):
    red = [str(s_) for s_ in (w or {}).get("reduce", []) or []]
    res = {str(s_) for s_ in (w or {}).get("restored", []) or []}
    return [s_ for s_ in red if s_ not in res]


def check_window_for_run(a, w=None):
    """본 실행의 창 조건: B 결정이 있고, --reduce 가 창 파일의 유효 축소 가운데 F 관련 단계와 같다. 어긋나면 SystemExit."""
    bad = [s_ for s_ in a.REDUCE if s_ not in REDUCE_F]
    if bad:
        raise SystemExit(f"[거부] 실행의 --reduce 는 S2, S3, S4 만 받는다: {bad}")
    w = read_window(a) if w is None else w
    if w.get("B") is None:
        raise SystemExit("[거부] 창 파일에 B 결정이 없다. 사전 점검 개정 뒤 --window-set --B --e1 --cbt --reduce --viewing-state 로 기록한다")
    want = sorted(s_ for s_ in effective_reduce(w) if s_ in REDUCE_F)
    if sorted(a.REDUCE) != want:
        raise SystemExit(f"[거부] --reduce {sorted(a.REDUCE)} 가 창 파일의 유효 축소(F 관련) {want} 와 다르다")
    return w                                                              # 창 마감 경과는 main 이 stop = 'window'(종료 코드 4)로 처리한다


def window_tool(a):
    """--window-set, --window-status, --gpu-event."""
    if a.window_status:
        return window_status(a)
    vs = str(a.viewing_state or "").strip()
    if (a.window_set or a.gpu_event) and not vs:
        raise SystemExit("[거부] 결정 기록에는 --viewing-state(LG, LGX, LGT, LGF 각각의 열람 상태)가 필요하다")
    out = {}
    if a.window_set:
        if a.RESTORE:
            bad = [s_ for s_ in a.RESTORE if s_ not in REDUCE_ALL or s_ in ("S7",)]
            if bad:
                raise SystemExit(f"[거부] 되돌릴 수 없는 단계: {bad}(S7 과 E1 은 선택이 시작된 뒤 바꾸지 않는다)")

            def fn(w):
                have = set(w.get("reduce", []) or [])
                miss = [s_ for s_ in a.RESTORE if s_ not in have]
                if miss:
                    raise SystemExit(f"[거부] 적용하지 않은 단계는 되돌릴 수 없다: {miss}")
                w["restored"] = sorted(set(w.get("restored", []) or []) | set(a.RESTORE))
                w.setdefault("decisions", []).append(dict(time=_now_iso(), what=f"복원 {','.join(a.RESTORE)}", viewing_state=vs))
                return w
        else:
            if a.B is None or a.e1 is None or a.cbt is None:
                raise SystemExit("[거부] --window-set 은 --B, --e1, --cbt 가 필요하다(복원은 --restore)")

            def fn(w):
                newB, newE1, newcbt = float(a.B), bool(a.e1 == "1"), bool(a.cbt == "1")
                if w.get("t0"):                                           # 창이 시작된 뒤: B, E1, cbt, S7 은 바꾸지 않는다(계획서 8.1, 8.4, 4.4)
                    bad = []
                    if w.get("B") is not None and float(w["B"]) != newB:
                        bad.append("B")
                    if "E1" in w and bool(w["E1"]) != newE1:
                        bad.append("E1")
                    if "cbt_enabled" in w and bool(w["cbt_enabled"]) != newcbt:
                        bad.append("cbt")
                    if ("S7" in (w.get("reduce") or [])) != ("S7" in a.REDUCE):
                        bad.append("S7")
                    if bad:
                        raise SystemExit(f"[거부] 창이 시작된 뒤(t0 {w['t0']})에는 {bad} 를 바꾸지 않는다(B 는 한 번 고정, E1·S7 은 1단계 선택이 시작된 뒤 "
                                         "고정, cbt 는 사전 점검 결정). 줄인 단계의 복원은 --restore 로 한다")
                restored = [s_ for s_ in (w.get("restored") or []) if s_ in a.REDUCE]     # 복원 기록은 지우지 않는다(적용 중인 축소에 대해서만 남긴다)
                w.update(B=newB, E1=newE1, cbt_enabled=newcbt, reduce=list(a.REDUCE), restored=restored)
                w.setdefault("decisions", []).append(dict(time=_now_iso(), what=f"B {newB:g}, E1 {a.e1}, cbt {a.cbt}, 축소 {a.REDUCE or '없음'}"
                                                          + (f", 복원 유지 {restored}" if restored else ""), viewing_state=vs))
                return w
        out = update_window(a, fn)
        print(f"[window] 기록했다: {window_path(a)}", flush=True)
    if a.gpu_event:
        try:
            g, ev = str(a.gpu_event).split(":")
            g = int(g)
        except ValueError:
            raise SystemExit("--gpu-event 는 '<GPU>:<add|remove>' 형식이다")
        if ev not in ("add", "remove"):
            raise SystemExit("--gpu-event 의 사건은 add 또는 remove 다")

        def fe(w):
            w.setdefault("gpu_events", []).append(dict(time=_now_iso(), gpu=g, event=ev, viewing_state=vs))
            return w
        out = update_window(a, fe)
        print(f"[window] GPU {g} {ev} 를 기록했다", flush=True)
    return out


def _projection_rows(a):
    p = Path(a.OUT) / "lgf_projection.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except (OSError, ValueError):
        return pd.DataFrame()


def projection_value(a, row, script="F", col="est_gpu_h"):
    pj = _projection_rows(a)
    if not len(pj) or "row" not in pj or col not in pj:
        return None
    q = pj[(pj["row"].astype(str) == str(row)) & (pj.get("script", pd.Series(["F"] * len(pj))).astype(str) == script)]
    if not len(q):
        return None
    v = pd.to_numeric(q[col], errors="coerce").dropna()
    return float(v.iloc[-1]) if len(v) else None


N_TAG = "lgfn"                                       # h48(N)의 tag. 워커 기록 run_lgfn/worker__*.json 으로 LGF 가 쓰는 GPU 를 센다


def live_gpus(a, include_n=True):
    """LGF 의 살아 있는 실행이 쓰는 GPU(워커 기록 run_<tag>/worker__*.json 의 살아 있는 PID). include_n 이면 h48 의 run_lgfn 도 센다
    (계획서 8.4 의 '그 시점에 LGF 가 쓰는 GPU 수')."""
    out = set()
    for tag in (a.tag, a.tag + "s") + ((N_TAG,) if include_n else ()):
        d = Path(a.OUT) / f"run_{tag}"
        for m_ in T43.read_run_notes(d, "worker") if d.exists() else []:
            try:
                if T43.pid_alive(int(m_.get("pid", 0))) and str(m_.get("gpu", "")) != "":
                    out.add(int(m_["gpu"]))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def window_status(a):
    """남은 창, 쓰는 GPU, 작업 완료 상태, 복원 규칙(계획서 §8.4)의 권고를 출력한다."""
    w = read_window(a)
    now = time.time()
    dl = _to_ts(w.get("deadline"))
    left_h = (dl - now) / 3600.0 if dl else None
    g = live_gpus(a)
    print(f"[window] t0 {w.get('t0', '없음')} · 마감 {w.get('deadline', '없음')} · 남은 시간 "
          f"{'창 시작 전' if left_h is None else f'{left_h:.1f} h'} · B {w.get('B')} · E1 {w.get('E1')} · cbt {w.get('cbt_enabled')} · "
          f"축소 {w.get('reduce', [])} · 복원 {w.get('restored', [])} · 쓰는 GPU(h47·h48 워커) {g or '없음'}", flush=True)
    try:
        a2 = parse_args(["--jobs", DEFAULT_JOBS, "--reduce", ",".join(s_ for s_ in effective_reduce(w) if s_ in REDUCE_F), "--threads", "1",
                         "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--count-only"])
        units, _ = enumerate_f(a2)
        by = Counter(); done = Counter()
        for u in units:
            by[u[0]] += 1
            done[u[0]] += int(unit_state_f(a2, u)[0])
        for j in [j for j in JOBS if by[j]]:
            print(f"  {JOB_J[j]} {j}: 완료 {done[j]}/{by[j]}", flush=True)
    except SystemExit as e:
        print(f"  작업 완료 상태를 셀 수 없다: {e}", flush=True)
    red = effective_reduce(w)
    if red and left_h is not None:
        cap = max(left_h, 0.0) * max(len(g), 1)
        for s_ in [s_ for s_ in ("S6", "S5", "S4", "S3", "S2", "S1") if s_ in red]:
            need = projection_value(a, s_, script="F")
            need = projection_value(a, s_, script="N") if need is None else need
            if need is None:
                print(f"  복원 권고 {s_}: lgf_projection.csv 에 추정량이 없다", flush=True)
                continue
            print(f"  복원 권고 {s_}: 여유 {cap:.1f} GPU-h 대 추정 {need:.1f} GPU-h → {'되돌린다(큐 끝)' if cap > need else '되돌리지 않는다'}", flush=True)
            if cap > need:
                cap -= need
    return w


def drain_requested(a):
    return any((Path(a.OUT) / f"run_{t}" / "drain").exists() for t in (a.TAG, f"{a.tag}s{a.SUFFIX}"))


def clear_drain(a):
    for t in (a.TAG, f"{a.tag}s{a.SUFFIX}"):
        (Path(a.OUT) / f"run_{t}" / "drain").unlink(missing_ok=True)


def assign_gate(a, state, now=None, load_fn=None):
    """단위를 배정해도 되는지. 반환 'ok', 'drain', 'window', 'load'. state = dict(last_load, high, started)를 갱신한다.
    본 실행에서는 드레인·창 마감·load 판정을 모두 통과한 첫 배정 직전에만 창의 t0 를 쓴다(load 보류 중에는 창을 시작하지 않는다)."""
    now = time.time() if now is None else float(now)
    if drain_requested(a):
        return "drain"
    if a.MAIN_MODE and window_deadline_passed(read_window(a), now):     # 창은 h48 이 먼저 시작했을 수 있다
        return "window"
    if now - float(state.get("last_load", -1e18)) >= LOAD_CHECK_S:
        state["last_load"] = now
        try:
            la = float((load_fn or os.getloadavg)()[0])
        except OSError:
            la = 0.0
        state["high"] = bool(la > float(a.load_max_run))
        state["load"] = la
    if state.get("high"):
        return "load"
    if a.MAIN_MODE and not state.get("started"):                          # t0 는 실제 배정(제출) 직전에 쓴다(계획서 8.1)
        window_start(a, now)
        state["started"] = True
        if window_deadline_passed(read_window(a), now):
            return "window"
    return "ok"


# ================================================================ GPU 확인
def reserved_gpus(a):
    return tuple(sorted(set(NEVER) | (set() if a.allow_gpus_01 else set(GPU01))))


def check_gpu_list(a):
    """--gpus 의 거부 규칙: 빈 목록, GPU 8, 허용 표지 없는 GPU 0·1, 후보 밖 번호."""
    if not a.GPUS:
        raise SystemExit("[거부] --gpus 가 비어 있다. 학습을 하는 실행은 GPU 를 지정한다(예: --gpus 5)")
    nv = [g for g in a.GPUS if g in NEVER]
    if nv:
        raise SystemExit(f"[거부] GPU {nv} 는 쓰지 않는다(다른 사용자)")
    g01 = [g for g in a.GPUS if g in GPU01]
    if g01 and not a.allow_gpus_01:
        raise SystemExit(f"[거부] GPU {g01} 는 --allow-gpus-01 과 개정 기록이 있을 때만 쓴다")
    out = [g for g in a.GPUS if g not in set(ALLOWED_GPUS) | set(GPU01)]
    if out:
        raise SystemExit(f"[거부] 후보 {list(ALLOWED_GPUS)} 밖의 GPU: {out}")
    return list(a.GPUS)


def lgt_lock_alive(a):
    """LGT 실행 잠금(run_<lgt-tag>/lock.json)의 PID 가 살아 있으면 그 PID, 아니면 0."""
    p = Path(a.LGTDIR) / f"run_{a.lgt_tag}" / "lock.json"
    try:
        pid = int(json.loads(p.read_text()).get("pid", 0) or 0)
    except (OSError, ValueError):
        return 0
    return pid if (pid and T43.pid_alive(pid)) else 0


def screen_now(a, cand, info=None, apps=None):
    """실행 직전의 점유 판정. 반환 (쓸 GPU, [(번호, 사유)]). nvidia-smi 를 부를 수 없으면 SystemExit."""
    try:
        info = T43.query_gpu_info() if info is None else info
        apps = T43.query_gpu_apps() if apps is None else apps
    except Exception as e:                                                # noqa: BLE001
        raise SystemExit(f"[거부] nvidia-smi 로 GPU 상태를 확인할 수 없다: {repr(e)[:200]}")
    ok, dropped = T43.screen_gpus(list(cand), {g: v["used"] for g, v in info.items()}, a.gpu_mem_max_mib, reserved=reserved_gpus(a),
                                  apps=T43.gpu_apps_by_index(info, apps))
    pid = lgt_lock_alive(a) if any(g in LGT_GPUS for g in ok) else 0
    if pid:
        dropped += [(g, f"LGT 실행 잠금의 PID {pid} 가 살아 있다") for g in ok if g in LGT_GPUS]
        ok = [g for g in ok if g not in LGT_GPUS]
    return ok, dropped


def rescreen(a, cand, excluded=()):
    """풀을 다시 만들 때의 재확인(h43.rescreen_gpus 와 같은 방식이고 후보·예약 규칙만 이 스크립트의 것이다)."""
    cand = [int(g) for g in cand if int(g) not in {int(x) for x in excluded}]
    t_end = time.time() + float(T43.GPU_SETTLE_S)
    while True:
        try:
            ok, dropped = screen_now(a, cand)
        except SystemExit as e:
            print(f"[pool] {e}. 풀을 다시 만들지 않는다", flush=True)
            return []
        if not dropped or time.time() >= t_end:
            break
        time.sleep(3.0)
    for g_, why in dropped:
        print(f"[pool] GPU {g_} 를 뺀다: {why}", flush=True)
    return ok


def check_run_args(a, load_fn=None):
    """학습을 하는 실행(스모크, 사전 점검, 본 실행)의 거부 조건. 자료를 읽기 전에 확인한다."""
    if not a.allow_local:
        raise SystemExit("[거부] 학습을 하는 실행(스모크, 사전 점검, 본 실행)은 --allow-local 이 필요하다")
    check_gpu_list(a)
    if not (1 <= int(a.threads_asked) <= THREADS_MAX):
        raise SystemExit(f"[거부] --threads {a.threads_asked}: 학습 실행은 프로세스당 스레드 1–{THREADS_MAX} 만 허용한다")
    if not Path(a.MODEL).exists():
        raise SystemExit(f"[거부] TabICL 가중치 파일이 없다: {a.MODEL}. 내려받기를 시도하지 않는다")
    got = T43.file_sha(a.MODEL)
    if got != str(a.model_sha1):
        raise SystemExit(f"[거부] 가중치 SHA-1 {got} 이 --model-sha1 {a.model_sha1} 과 다르다")
    try:
        la = float((load_fn or os.getloadavg)()[0])
    except OSError:
        la = 0.0
    if la > float(a.load_max_start):
        raise SystemExit(f"[거부] 1분 load average {la:.1f} 가 --load-max-start {a.load_max_start:g} 를 넘는다")
    return True


# ================================================================ 학습기
def frame(X):
    """TabICL 입력: 열 이름이 h40.FEATS 인 float32 DataFrame(결측은 NaN). 열 수가 다르면(시험) f0.. 이름을 쓴다."""
    X = np.asarray(X, np.float32)
    cols = list(H.FEATS) if X.shape[1] == len(H.FEATS) else [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols).astype("float32")


def tabicl_kwargs(a, seed):
    return dict(n_estimators=int(a.n_est), norm_methods=None, feat_shuffle_method="latin", outlier_threshold=4.0, batch_size=int(a.tabicl_batch_size),
                model_path=str(a.MODEL), allow_auto_download=False, checkpoint_version=CKPT_VERSION, device="cuda", use_amp="auto", use_fa3="auto",
                offload_mode="auto", random_state=int(seed), n_jobs=None, verbose=False)


def cached_class():
    """TabICLCached: _load_model 만 바꾼 하위 클래스. tabicl 2.0.2 의 regressor._load_model(명시 경로가 있는 경우)과 같은 순서로
    'config'·'state_dict' 확인, model_path_, TabICL(**config), model_config_, load_state_dict, eval 을 한다. 다른 점은 torch.load 를 워커 전역
    캐시 {경로: 체크포인트} 에서 한 번만 한다는 것뿐이다."""
    global _CACHED_CLS
    if _CACHED_CLS is None:
        import torch
        from tabicl import TabICL, TabICLRegressor

        class TabICLCached(TabICLRegressor):
            def _load_model(self):
                if self.model_path is None:
                    raise ValueError("TabICLCached 는 model_path 가 필요하다")
                path = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
                ck = _CKPT.get(str(path))
                if ck is None:
                    ck = torch.load(path, map_location="cpu", weights_only=True)
                    _CKPT[str(path)] = ck
                assert "config" in ck, "The checkpoint doesn't contain the model configuration."
                assert "state_dict" in ck, "The checkpoint doesn't contain the model state."
                self.model_path_ = path
                config = ck["config"]
                self.model_ = TabICL(**config)
                self.model_config_ = config
                self.model_.load_state_dict(ck["state_dict"])
                self.model_.eval()
        _CACHED_CLS = TabICLCached
    return _CACHED_CLS


def install_offload_probe():
    """offload 방식의 실제 적용 값을 기록한다: InferenceManager._resolve_offload_mode 의 반환값을 그대로 돌려주면서 방식 이름을 모은다."""
    try:
        from tabicl.model import inference as _inf
    except Exception:                                                     # noqa: BLE001
        return False
    cls = getattr(_inf, "InferenceManager", None)
    if cls is None or not hasattr(cls, "_resolve_offload_mode"):
        return False
    if getattr(cls, "_lgf_probe", False):
        return True
    orig = cls._resolve_offload_mode

    def _probe(self, *args, **kw):
        res = orig(self, *args, **kw)
        try:
            _OFFLOAD_SEEN.add(str(getattr(res[0], "name", res[0])))
        except Exception:                                                 # noqa: BLE001
            pass
        return res
    cls._resolve_offload_mode = _probe
    cls._lgf_probe = True
    return True


def _predict_frame_chunks(m, XBdf, chunk):
    chunk = max(int(chunk), 1)
    return np.concatenate([np.asarray(m.predict(XBdf.iloc[k:k + chunk], output_type="mean"), float) for k in range(0, len(XBdf), chunk)])


def tabicl_fit_predict(a, X, y, XB, seed, check_chunk=0, check_stock=False):
    """TabICL 적합과 예측. 반환 (예측, 표지, 정보). 정보: amp_used, fa3_flag(_resolve_amp_fa3), has_fa3(HAS_FLASH_ATTN3), offload_mode.
    예측 중 CUDA 메모리 부족이면 except 블록 밖에서 gc.collect() 와 캐시 비우기 뒤 --pred-chunk 행 묶음으로 다시 예측한다(표지 chunk).
    적합(fit) 중의 메모리 부족은 예외로 올린다(그 적합의 실패). check_chunk > 0 이면 두 묶음 예측과의 차, check_stock 이면 원래 클래스와의
    예측 차를 정보에 남긴다(스모크 전용)."""
    from tabicl import TabICLRegressor
    import tabicl.model.attention as _att
    install_offload_probe()
    stock = bool(a.tabicl_stock or _FORCE_STOCK[0])
    cls = TabICLRegressor if stock else cached_class()
    Xdf, XBdf = frame(X), frame(XB)
    yv = np.asarray(y, float)
    m = cls(**tabicl_kwargs(a, seed))
    _OFFLOAD_SEEN.clear()
    m.fit(Xdf, yv)
    amp_used, fa3_flag = m._resolve_amp_fa3()
    flag = ""
    info = dict(amp_used=bool(amp_used), fa3_flag=bool(fa3_flag), has_fa3=bool(getattr(_att, "HAS_FLASH_ATTN3", False)),
                impl="stock" if stock else "cached")
    oom, p = False, None
    try:
        p = np.asarray(m.predict(XBdf, output_type="mean"), float)
    except Exception as e:                                                # noqa: BLE001
        if not T43.is_cuda_oom(e):
            raise
        oom = True
    if oom:
        gc.collect()
        T43._free_cuda()
        p = _predict_frame_chunks(m, XBdf, a.pred_chunk)
        flag = "chunk"
    info["offload_mode"] = ";".join(sorted(_OFFLOAD_SEEN)) or "none"
    t1 = time.time()
    if check_chunk and flag == "" and len(XB) > int(check_chunk):
        q = _predict_frame_chunks(m, XBdf, check_chunk)
        info.update(chunk_rows=int(check_chunk), chunk_maxdiff=float(np.max(np.abs(q - p))))
    if check_stock and not stock:
        ms = TabICLRegressor(**tabicl_kwargs(a, seed))
        ms.fit(Xdf, yv)
        q = np.asarray(ms.predict(XBdf, output_type="mean"), float)
        info.update(stock_maxdiff=float(np.max(np.abs(q - p))))
        del ms
    info["check_s"] = round(time.time() - t1, 3) if (check_chunk or check_stock) else 0.0
    del m
    return p, flag, info


def cb_fit_predict(ns, X, y, XB, seed):
    """catboost_ctx(h43.cb_fit_predict). ns = SimpleNamespace(threads, cb_iters)."""
    return T43.cb_fit_predict(ns, X, y, XB, seed)


def cb_h40_pred(ns, X, y, XB, seed):
    """스모크 점검(4)과 시험: h40.cb_fit 의 모형 정의(초모수, random_seed)로 적합한 예측. h40.cb_fit 은 numpy 배열을 fit 에 넘겨 학습 Pool 을
    thread_count −1(모든 코어)로 만들므로, 여기서는 thread_count = ns.threads 인 Pool 을 X 로 넘긴다(y 는 Pool 안에 둔다)."""
    from catboost import Pool
    th = int(ns.threads)
    m = H.cb_fit(ns, Pool(np.asarray(X, float), np.asarray(y, float), thread_count=th), None, seed)
    return np.asarray(m.predict(Pool(np.asarray(XB, float), thread_count=th), thread_count=th), float)


class FitterF:
    """적합 실행과 기록. dry = True 이면 학습 없이 수와 추정 시간만 센다(예측 0). 실패 규칙은 h43.FitterT 와 같다: 점검 실패는 그 적합만,
    학습기 실패(예외, 비유한 예측)는 학습기별 연속 횟수를 세어 FAIL_STREAK_MAX 에 이르면 h43.FitAbort 로 단위를 중단한다."""

    def __init__(self, a, axis, n_eval, dry=False, cap=None):
        self.a, self.axis, self.n_eval, self.dry, self.cap = a, axis, int(n_eval), bool(dry), cap
        self.cbns = SimpleNamespace(threads=int(a.threads), cb_iters=int(a.cb_iters))
        self.n = Counter(); self.sec = Counter(); self.fail = Counter(); self.nonfin = Counter()        # 키 = "축:학습기"
        self.nd = Counter(); self.secd = Counter(); self.rowsd = Counter(); self.estd = Counter()       # 키 = "축|학습기|방법"
        self.errors: list = []
        self.smoke: list = []
        self.streak = Counter()
        self._checked: set = set()
        self._cb_checked = False
        self.allnan_max = 0
        self.n_check_fail = 0
        self.gpu_peak = 0.0
        self.gpu_resv = 0.0
        self.amp = Counter(); self.offload = Counter(); self.impl = Counter()

    def check(self, X, y, XB, n_ctx):
        if len(y) != len(X) or len(X) != int(n_ctx):
            raise ValueError(f"컨텍스트 행 수 불일치: X {len(X)}, y {len(y)}, 기대 {int(n_ctx)}")
        if self.cap is not None and len(X) > int(self.cap):
            raise ValueError(f"컨텍스트 행 수 {len(X)} 가 상한 {int(self.cap)} 을 넘는다")
        if X.shape[1] != XB.shape[1]:
            raise ValueError(f"입력 열 수 불일치: 컨텍스트 {X.shape[1]}, 채점 {XB.shape[1]}")
        if not np.all(np.isfinite(np.asarray(y, float))):
            raise ValueError("학습 목표에 비유한 값이 있다")

    def _fail(self, k, learner, err):
        self.fail[f"{self.axis}:{learner}"] += 1
        if len(self.errors) < 20:
            self.errors.append(f"{k}: {err}")
        print(f"    [warn] 적합 실패({k}): {err[:160]}", flush=True)

    def fit(self, learner, method, X, y, XB, seed, n_ctx, info=None):
        """반환 (예측 또는 잔차 성분, 표지, 시간 s, 컨텍스트 해시, 추가 열 dict)."""
        k = f"{self.axis}|{learner}|{method}"
        self.n[f"{self.axis}:{learner}"] += 1; self.nd[k] += 1; self.rowsd[k] += int(n_ctx)
        self.estd[k] += T43.est_fit_s(TP if learner == TI else learner, n_ctx, self.n_eval)
        ex = dict(amp_used="", fa3_flag="", has_fa3="", offload_mode="", n_allnan_cols=-1, gpu_mem_fit_mib=np.nan, rss_mib=np.nan)
        if X is not None and len(X):
            ex["n_allnan_cols"] = int(np.isnan(np.asarray(X, float)).all(0).sum())
            self.allnan_max = max(self.allnan_max, ex["n_allnan_cols"])
        if self.dry:
            if X is not None:                                             # --dry-build: 행렬의 크기와 목표만 확인한다(학습 없음)
                try:
                    self.check(X, y, XB, n_ctx)
                except ValueError as e:
                    self.fail[f"{self.axis}:{learner}"] += 1; self.n_check_fail += 1
                    if len(self.errors) < 20:
                        self.errors.append(f"{k}: {repr(e)[:200]}")
            return np.zeros(self.n_eval), "", 0.0, "", ex
        T43._orphan_check()
        t0 = time.time(); flag, sha, err, oom, lfail, extra_s, p = "", "", "", False, False, 0.0, None
        try:
            if TRACE is not None:
                TRACE.append(dict(info or {}, axis=self.axis, learner=learner, method=method, seed=int(seed), X=np.array(X, copy=True),
                                  y=np.array(y, copy=True), XB=np.array(XB, copy=True)))
            self.check(X, y, XB, n_ctx)
        except ValueError as e:
            err = "점검: " + repr(e)[:200]
            self.n_check_fail += 1
        if not err:
            try:
                sha = T43.ctx_sha(X, y)
                if learner == TI:
                    first = bool(self.a.smoke and self.axis not in self._checked)
                    chk = int(math.ceil(self.n_eval / 2.0)) if (first and self.n_eval >= 2) else 0
                    _cuda_reset()
                    p, flag, inf = tabicl_fit_predict(self.a, X, y, XB, seed, check_chunk=chk, check_stock=first)
                    al, rs = _cuda_peak()
                    extra_s += float(inf.get("check_s", 0.0) or 0.0)
                    ex.update(amp_used=bool(inf.get("amp_used")), fa3_flag=bool(inf.get("fa3_flag")), has_fa3=bool(inf.get("has_fa3")),
                              offload_mode=str(inf.get("offload_mode", "")), gpu_mem_fit_mib=round(al, 1) if np.isfinite(al) else np.nan)
                    if np.isfinite(al):
                        self.gpu_peak = max(self.gpu_peak, al); self.gpu_resv = max(self.gpu_resv, rs)
                    self.amp[str(ex["amp_used"])] += 1; self.offload[ex["offload_mode"]] += 1; self.impl[str(inf.get("impl", ""))] += 1
                    if first:
                        self._checked.add(self.axis)
                        if "chunk_maxdiff" in inf:
                            self.smoke.append(dict(check="chunk2", ctx_set=self.axis, method=method, value=float(inf["chunk_maxdiff"]), tol=CHUNK_TOL,
                                                   ok=bool(inf["chunk_maxdiff"] <= CHUNK_TOL), n_eval=self.n_eval, n_ctx=int(n_ctx)))
                        if "stock_maxdiff" in inf:
                            okc = bool(inf["stock_maxdiff"] <= CACHED_TOL)
                            if not okc:
                                _FORCE_STOCK[0] = True                    # 이후 적합은 원래 클래스를 쓴다(기록)
                            self.smoke.append(dict(check="cached_vs_stock", ctx_set=self.axis, method=method, value=float(inf["stock_maxdiff"]),
                                                   tol=CACHED_TOL, ok=okc, forced_stock=bool(not okc), n_eval=self.n_eval, n_ctx=int(n_ctx)))
                else:
                    p, flag, _ = cb_fit_predict(self.cbns, X, y, XB, seed)
                    if self.a.smoke and not self._cb_checked:             # 스모크: h43 Pool 경로와 h40.cb_fit(모형 정의)의 예측 차
                        t1 = time.time()
                        self._cb_checked = True
                        try:
                            q = cb_h40_pred(self.cbns, X, y, XB, seed)
                            self.smoke.append(dict(check="cb_pool", ctx_set=self.axis, method=method, tol=np.nan, ok=True, n_ctx=int(n_ctx),
                                                   value=float(np.max(np.abs(q - np.asarray(p, float))))))
                        except Exception as e_:                           # noqa: BLE001  점검의 오류는 적합 실패로 세지 않는다
                            self.smoke.append(dict(check="cb_pool", ctx_set=self.axis, method=method, value=np.nan, tol=np.nan, ok=False,
                                                   err=repr(e_)[:200], n_ctx=int(n_ctx)))
                        extra_s += time.time() - t1
                p = np.asarray(p, float)
                if p.shape != (len(XB),):
                    raise ValueError(f"예측의 모양 {p.shape} 이 채점 셀 수 {len(XB)} 와 다르다")
            except (ImportError, MemoryError):
                raise
            except Exception as e:                                        # noqa: BLE001
                err, oom, lfail = repr(e)[:200], T43.is_cuda_oom(e), True
        ex["rss_mib"] = rss_mib()
        if err:
            self._fail(k, learner, err)
            p, flag = np.full(len(XB), np.nan), "fail"
            if oom:
                gc.collect()
                T43._free_cuda()
        elif not np.all(np.isfinite(p)):
            self.nonfin[f"{self.axis}:{learner}"] += 1
            lfail = True
        if lfail:
            self.streak[learner] += 1
            if self.streak[learner] >= FAIL_STREAK_MAX:
                raise T43.FitAbort(f"{T43.TAG_ABORT} {learner} 적합이 {self.streak[learner]}회 연속 실패했다({k}). 마지막: {err or '비유한 예측'}")
        elif not err:
            self.streak[learner] = 0
        dt = max(time.time() - t0 - extra_s, 0.0)
        self.sec[f"{self.axis}:{learner}"] += dt; self.secd[k] += dt
        return p, flag, dt, sha, ex


def learner_counts_f(F, st):
    """학습기별 적합 수, 실패 수, 저장 키 수와 조각을 failed 로 두게 하는 학습기(h43.learner_counts_t 와 같은 규칙)."""
    fit_by, fail_by, stored_by = Counter(), Counter(), Counter()
    for kk, v in F.n.items():
        fit_by[kk.split(":", 1)[-1]] += int(v)
    for src in (F.fail, F.nonfin):
        for kk, v in src.items():
            fail_by[kk.split(":", 1)[-1]] += int(v)
    for k in (st.keys if st is not None else []):
        if k[1] != "none":
            stored_by[k[1]] += 1
    bad = [lr for lr in LEARNERS_F if fit_by[lr] > 0 and (stored_by[lr] == 0 or fail_by[lr] / fit_by[lr] > FAIL_RATIO_MAX)]
    return dict(n_fit_learner=dict(fit_by), n_fail_learner=dict(fail_by), n_stored_learner=dict(stored_by), failed_learners=bad)


def ctx_setting(a, axis):
    """(컨텍스트 설정 dict, 컨텍스트 인자, 행 수 상한). main 은 h43.MAIN_SET 과 --ctx-max 규칙, full 은 상한 없음."""
    if axis == "main":
        return T43.MAIN_SET, a, int(a.ctx_max)
    return FULL_SET, FULL_CTX, None


# ================================================================ 작업 단위 실행
def run_ctx_f(c, axis, part, a, HA, dry=False, loc=None):
    """작업 단위 하나(축, 대상, 모드, 분할, 부분). 반환 (rows, BlockStore, 통계 dict, CellBookT 또는 None).
    순서는 h43.run_ctx_t 의 run_set 과 같다: seed 별 ctx_order → P0 → (n, 추출) 칸마다 E_n 과 P1 → seed 마다 컨텍스트 행과 행렬 →
    방법(D0, R0, R1 또는 D0@full, R1@full) → 학습기(tabicl, catboost_ctx). main 의 n = 0 R1 은 R0 적합을 다시 쓴다."""
    vs, a_ctx, cap = ctx_setting(a, axis)
    sfx = str(vs["sfx"])
    methods = AX_F[axis]["methods"]
    F = FitterF(a, axis, len(c.yB), dry, cap)
    empty = dict(n_fit={}, sec={}, fail={}, n_rows=0, status="no_eval", n_fit_detail={}, sec_detail={}, rows_detail={}, est_detail={}, errors=[],
                 n_stored=0, n_stored_ml=0, n_nonfinite_keys=0, ctx={}, ctx_dev_max=0.0, ctx_dev_bound=0.0, n_src_regions=0, n_ctx_max=0, n_ctx_min=0,
                 flags={}, smoke_check=[], nonfinite_fits={}, n_allnan_cols_max=0, n_check_fail=0, n_fit_learner={}, n_fail_learner={},
                 n_stored_learner={}, failed_learners=[], amp_counts={}, offload_counts={}, impl_counts={}, gpu_fit_peak_mib=0.0,
                 gpu_fit_reserved_peak_mib=0.0)
    if len(c.yB) == 0 or len(c.yA) == 0:
        return [], None, empty, None
    st = BlockStore(f"{c.target}|{c.mode}", c.split, c.blkB, meta=dict(target=c.target, mode=c.mode, part=part, axis=axis))
    rows, n_rows, n_bad = [], [0], [0]
    nA = len(c.yA)
    E0 = float(c.E0)
    build = (not dry) or bool(getattr(a, "dry_build", False))
    cells = T43.CellBookT(c, loc, E0) if (a.save_cells and not dry) else None
    base = dict(target=c.target, mode=c.mode, parent=c.parent, split=c.split, part=part, axis=axis)
    orders = {int(s_): T43.ctx_order(c.macro_src, c.target, c.mode, c.split, int(s_)) for s_ in HA.SEEDS}
    ctx_info: dict = {}
    dev_max, nctx_all, flags = [0.0], [], Counter()
    metrics = bool(a.STORE)

    def add(method, learner, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, nb_lab=0, flag="", nc=(0, 0, 0), fit_s=0.0, sha="", ex=None):
        n_rows[0] += 1
        if dry:
            return
        pred = np.asarray(pred, float)
        nf = int((~np.isfinite(pred)).sum())
        row = dict(**base, method=method, learner=learner, alpha="1", placement="cell", n=int(n), n_lab=int(n_lab), draw=int(d), seed=int(seed),
                   lam=float(lam), rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan, E_used=float(E_used), alpha_sel="", n_blocks_lab=int(nb_lab),
                   n_nonfinite=nf, fit_flag=str(flag) if nf == 0 else (str(flag) or "nonfinite"), ctx_set=axis, n_ctx=int(nc[0]), n_ctx_src=int(nc[1]),
                   n_ctx_tgt=int(nc[2]), fit_s=round(float(fit_s), 3), ctx_sha=str(sha))
        row.update({k_: (ex or {}).get(k_, "" if k_ in ("amp_used", "fa3_flag", "has_fa3", "offload_mode") else np.nan) for k_ in EXTRA_COLS})
        if nf > 0:                                                       # 비유한 예측이 있는 키는 저장하지 않는다
            n_bad[0] += 1
            rows.append(row)
            return
        key = (method, learner, "1", "cell", int(n), int(d), int(seed), float(lam))
        st.add(key, c.yB, pred)
        if metrics:                                                      # 스모크·사전 점검은 RMSE 를 쓰지 않는다(NaN)
            sse, cnt = st.get(key)
            with np.errstate(invalid="ignore", divide="ignore"):
                beq = float(np.nanmean(np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan))) if cnt.sum() else np.nan
                rm = float(np.sqrt(sse.sum() / cnt.sum())) if cnt.sum() else np.nan
            row.update(rmse_cm=rm, rmse_beq_cm=beq, bias_cm=float(np.nanmean(pred - c.yB)) if len(c.yB) else np.nan)
        rows.append(row)

    seen_p1: set = set()

    def analytic(n, d, E_n, nl, nb):
        if (n, d) in seen_p1:
            return
        seen_p1.add((n, d))
        add("P1", "none", n, d, -1, 0.0, E_n * c.sB, E_n, nl, nb)
        if cells is not None:
            cells.add_coef(n, d, E_n)

    add("P0", "none", 0, 0, -1, 0.0, E0 * c.sB, E0, 0)                 # P0 은 모든 조각에 저장한다

    def note_ctx(seed, src):
        m_ = int(len(src))
        slot = ctx_info.setdefault(str(int(seed)), {})
        if str(m_) not in slot:
            slot[str(m_)] = dict(n_src=m_, regions=T43.region_counts(c.macro_src, src))
            dev_max[0] = max(dev_max[0], T43.region_dev(c.macro_src, src))

    r1r0 = []
    for n, d in H.cells_of(list(HA.N_GRID), HA.DRAWS, nA):
        sel = H.draw_cells(c.target, c.mode, c.split, n, d, nA)
        nl = len(sel)
        nb = len(np.unique(c.blkA[sel])) if nl else 0
        E_n = T43.coef_n(c, sel, HA.kappa)
        analytic(n, d, E_n, nl, nb)
        for seed in HA.SEEDS:
            src, tsel, tflag = T43.context_rows(c, orders[int(seed)], sel, vs, a_ctx, n, d)
            nc = (len(src) + len(tsel), len(src), len(tsel))
            if nc[0] == 0:
                continue
            note_ctx(seed, src)
            nctx_all.append(nc[0])
            if tflag:
                flags[tflag] += 1
            X = XB = None
            if build:
                X, XB = T43.build_X(c, src, tsel, 1)
            g0: dict = {}
            for method in methods:
                mname = method + sfx
                if axis == "main" and method == "R1" and nl == 0 and g0:   # n = 0 에서 R1 은 R0 과 같다(E_n = E0, 같은 적합)
                    for lr in LEARNERS_F:
                        g, fl, sha, ex = g0[lr]
                        for lam in HA.LAMS:
                            add(mname, lr, n, d, seed, lam, E0 * c.sB + lam * g, E0, nl, nb, fl, nc, 0.0, sha, ex)
                            if not dry and a.smoke:
                                k1, k0 = (mname, lr, "1", "cell", int(n), int(d), int(seed), float(lam)), ("R0", lr, "1", "cell", int(n), int(d),
                                                                                                          int(seed), float(lam))
                                if k1 in st and k0 in st:
                                    r1r0.append(float(np.max(np.abs(st.get(k1)[0] - st.get(k0)[0]))))
                        if cells is not None:
                            cells.add(mname, lr, "1", n, d, seed, "g", g, E0)
                    continue
                kind = "D" if method == "D0" else "R"
                E_a = None if kind == "D" else (E0 if method == "R0" else E_n)
                y = T43.build_y(c, src, tsel, 1, kind, E_a) if build else None
                for lr in LEARNERS_F:
                    info = dict(n=int(n), draw=int(d), sel=np.array(sel, copy=True), tsel=np.array(tsel, copy=True), src=np.array(src, copy=True),
                                E_anchor=E_a)
                    g, fl, dt, sha, ex = F.fit(lr, mname, X, y, XB, seed, nc[0], info)
                    fl = ";".join(v for v in (tflag, fl) if v)
                    if kind == "D":
                        add(mname, lr, n, d, seed, 1.0, g, np.nan, nl, nb, fl, nc, dt, sha, ex)
                    else:
                        for lam in HA.LAMS:
                            add(mname, lr, n, d, seed, lam, float(E_a) * c.sB + lam * g, E_a, nl, nb, fl, nc, dt, sha, ex)
                    if axis == "main" and method == "R0" and nl == 0:
                        g0[lr] = (g, fl, sha, ex)
                    if cells is not None:
                        cells.add(mname, lr, "1", n, d, seed, "pred" if kind == "D" else "g", g, np.nan if kind == "D" else E_a)
    if r1r0:
        F.smoke.append(dict(check="r1_eq_r0_n0", ctx_set=axis, value=float(max(r1r0)), tol=0.0, ok=bool(max(r1r0) == 0.0), n_keys=len(r1r0)))
    n_ml = sum(1 for k in st.keys if k[1] != "none")
    n_fail = int(sum(F.fail.values())) + int(n_bad[0])
    lc = learner_counts_f(F, st)
    status = "ok" if n_fail == 0 else ("failed" if lc["failed_learners"] else "partial")
    stats = dict(n_fit=dict(F.n), sec={k: round(v, 1) for k, v in F.sec.items()}, fail=dict(F.fail), n_rows=int(n_rows[0]), status=status,
                 n_fit_detail=dict(F.nd), sec_detail={k: round(v, 2) for k, v in F.secd.items()}, rows_detail=dict(F.rowsd),
                 est_detail={k: round(v, 2) for k, v in F.estd.items()}, errors=list(F.errors), n_stored=int(len(st)), n_stored_ml=int(n_ml),
                 n_nonfinite_keys=int(n_bad[0]), ctx=ctx_info, ctx_dev_max=round(float(dev_max[0]), 3),
                 ctx_dev_bound=round(T43.region_dev_bound(c.macro_src), 3), n_src_regions=int(len(set(np.asarray(c.macro_src).astype(str).tolist()))),
                 n_ctx_max=int(max(nctx_all)) if nctx_all else 0, n_ctx_min=int(min(nctx_all)) if nctx_all else 0, flags=dict(flags),
                 smoke_check=list(F.smoke), nonfinite_fits=dict(F.nonfin), n_allnan_cols_max=int(F.allnan_max), n_check_fail=int(F.n_check_fail),
                 amp_counts=dict(F.amp), offload_counts=dict(F.offload), impl_counts=dict(F.impl), gpu_fit_peak_mib=round(F.gpu_peak, 1),
                 gpu_fit_reserved_peak_mib=round(F.gpu_resv, 1), **lc)
    return rows, st, stats, cells


# ================================================================ 조각 입출력과 설정
def a43_args(a, grid, draws, seeds, targets=None):
    """h43 형식의 인자(A43). 컨텍스트 매개변수와 격자·추출·seed 를 이 스크립트의 값으로 준다. (격자, 추출, seed, 대상)마다 캐시한다."""
    key = (tuple(int(n) for n in grid), int(draws), int(seeds), tuple(targets or ()))
    cache = a.__dict__.setdefault("_a43", {})
    if key in cache:
        return cache[key]
    argv = ["--axes", "main", "--splits", str(int(a.splits)), "--n-grid", T43._grid_txt(grid) or "0", "--draws", str(int(draws)),
            "--seeds", str(int(seeds)), "--lams", ",".join(str(v) for v in a.LAMS), "--kappa", str(float(a.kappa)), "--ctx-max", str(int(a.ctx_max)),
            "--ctx-reserve", str(int(a.ctx_reserve)), "--ctx-src-min", str(int(a.ctx_src_min)), "--threads", str(int(a.threads)),
            "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", a.subregion_map, "--tag", a.tag, "--no-cells",
            "--cb-iters", str(int(a.cb_iters))]
    if targets:
        argv += ["--targets", ",".join(f"{t}:{m}" for t, m in targets)]
    A = T43.parse_args(argv)
    if not grid:
        T43.h40_args_t(A, "main").N_GRID = []
    cache[key] = A
    return A


def ha_of(a, u):
    return T43.h40_args_t(a43_args(a, u[6], u[7], u[8]), "main")


def unit_name_f(u):
    return f"{u[1]}|{u[2]}|{u[3]}|s{u[4]}|{u[5]}"


def shard_paths_f(a, u):
    b = Path(a.SHARDS) / f"{axis_tag(a, u[1])}__gpu__{u[2]}__{u[3]}__s{int(u[4])}__{u[5]}"
    return dict(runs=Path(str(b) + "_runs.csv"), npz=Path(str(b) + "_blocksse.npz"), cells=Path(str(b) + "_cells.npz"), unit=Path(str(b) + "_unit.json"))


def axis_grid_all(a, axis):
    """축 전체의 n 격자(게이트 조건 2 에서 LGT 의 n_grid 와 비교한다)."""
    if axis == "main":
        return list(a.N_USER)
    return [n for n in a.N_USER if n == 0] + [n for n in FULL_POS_GRID if n in a.N_USER]


def unit_cfg_f(a, u):
    """결과에 영향을 주는 설정 요약. 대상, 분할, tag, 스레드, GPU 번호는 넣지 않는다."""
    job, axis, t, m, sp, part, grid, draws, seeds = u
    HA = ha_of(a, u)
    vs, a_ctx, cap = ctx_setting(a, axis)
    d = dict(axis=axis, part=part, methods=[mm + str(vs["sfx"]) for mm in AX_F[axis]["methods"]], learners=list(LEARNERS_F), n_grid=list(grid),
             n_grid_axis=axis_grid_all(a, axis), draws=int(draws), seeds=list(range(int(seeds))), lams=list(a.LAMS), kappa=float(a.kappa),
             buffer_km=float(HA.buffer_km), k_sub=str(HA.k_sub), min_cells_prior=int(HA.min_cells_prior),
             subregion_map=Path(HA.SUBMAP).name if Path(HA.SUBMAP).exists() else "kmeans",
             ctx_max=int(a_ctx.ctx_max), ctx_reserve=int(a_ctx.ctx_reserve), ctx_src_min=int(a_ctx.ctx_src_min), dup=int(vs["dup"]), ctx_seed="lgt-ctx",
             n_est=int(a.n_est), tabicl_batch_size=int(a.tabicl_batch_size), model=Path(a.MODEL).name, model_sha1=str(a.model_sha1),
             tabicl=T43.pkg_version("tabicl"), tabicl_params=dict(norm_methods=None, feat_shuffle_method="latin", outlier_threshold=4.0, use_amp="auto",
                                                                  use_fa3="auto", offload_mode="auto", n_jobs=None, output_type="mean",
                                                                  input="DataFrame float32 NaN"),
             cb_iters=int(a.cb_iters), feats="x25", pred_chunk=int(a.pred_chunk), h43_sha1=str(a.PINS.get("h43", "")),
             h40_sha1=str(a.PINS.get("h40", "")))
    if a.PINS.get("revision_needed"):
        d["pin_note"] = "개정 필요"
    return d


def unit_state_f(a, u):
    """(완료 여부, 사유). 완료 = 조각 파일(저장 모드는 runs, blocksse, unit. 스모크·사전 점검은 runs, unit)이 있고, 설정 해시가 같고, status 가
    failed 가 아니다. --rerun-partial 이면 partial 도 미완료로 본다."""
    p = shard_paths_f(a, u)
    need = ("unit", "runs", "npz") if a.STORE else ("unit", "runs")
    if not all(p[k].exists() for k in need):
        return False, "조각 없음"
    try:
        uj = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if uj.get("cfg_hash") != H.cfg_hash(unit_cfg_f(a, u)):
        return False, "설정 불일치(cfg_hash)"
    if uj.get("status") == "failed":
        return False, "이전 실행 실패"
    if getattr(a, "rerun_partial", False) and uj.get("status") == "partial":
        return False, "일부 적합 실패(--rerun-partial)"
    return True, str(uj.get("status", "ok"))


def write_shard_f(a, u, c, rows, st, stats, cells, elapsed, extra=None):
    p = shard_paths_f(a, u)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    T43._atomic_csv(df[[k for k in RUN_COLS_F if k in df] + [k for k in df.columns if k not in RUN_COLS_F]] if len(df) else df, p["runs"])
    if a.STORE:
        save_stores([st], p["npz"])
        if cells is not None:
            cells.save(p["cells"])
    cfg = unit_cfg_f(a, u)
    unit = {**stats, **c.meta}
    unit.update(job=u[0], target=c.target, mode=c.mode, parent=c.parent, split=c.split, part=u[5], axis=u[1], learner="", tag=axis_tag(a, u[1]),
                elapsed_s=round(float(elapsed), 1), n_fit_total=int(sum(stats["n_fit"].values())), cfg=cfg, cfg_hash=H.cfg_hash(cfg),
                code_sha=dict(h47=T43.file_sha(__file__), h43=a.PINS.get("h43"), h40=a.PINS.get("h40"), h42=a.PINS.get("h42")),
                code_sha_h43=str(a.PINS.get("h43", ""))[:12], pins=dict(a.PINS), threads=int(a.threads), device=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                has_cells=bool(a.STORE and cells is not None), stored=bool(a.STORE), packages=pkg_versions(), nice=_nice_now())
    unit.update(extra or {})
    H._atomic_text(p["unit"], json.dumps(unit, ensure_ascii=False, indent=1, default=float))        # 완료 표지는 마지막에 쓴다
    return unit


def _nice_now():
    try:
        return int(os.nice(0))
    except OSError:
        return None


def cuda_backend():
    """GPU 백엔드(torch). CUDA 를 쓸 수 없으면 예외다(시험에서 대체한다)."""
    return T43.require_cuda()


def h43_ctx_shas(c, a, grid, draws, seeds):
    """스모크 점검(1)·시험: h43.run_ctx_t(주 설정)를 대체 학습기로 돌려 h43 경로의 ctx_sha 를 얻는다. {(방법, n, 추출, seed): sha}.
    h43 의 학습기 함수는 이 함수 안에서만 바꾸고 끝나면 되돌린다."""
    A = a43_args(a, grid, draws, seeds)
    HA = T43.h40_args_t(A, "main")
    old = (T43.tabpfn_fit_predict, T43.cb_fit_predict)
    T43.tabpfn_fit_predict = lambda a_, X, y, XB, seed, cat_idx=None, check_chunk=0: (np.zeros(len(XB)), "", {})
    T43.cb_fit_predict = lambda HA_, X, y, XB, seed, cat_idx=None: (np.zeros(len(XB)), "", {})
    try:
        rows, _, _, _ = T43.run_ctx_t(c, "main", A, HA, dry=False)
    finally:
        T43.tabpfn_fit_predict, T43.cb_fit_predict = old
    return {(str(r["method"]), int(r["n"]), int(r["draw"]), int(r["seed"])): str(r["ctx_sha"]) for r in rows if r["learner"] == TP and r["ctx_sha"]}


def full_ctx_shas(c, a, grid, draws, seeds, kappa):
    """스모크 점검(1)·시험(full): h43 의 컨텍스트 함수로 full 컨텍스트를 다시 만들어 ctx_sha 를 구한다."""
    out, nA = {}, len(c.yA)
    for n, d in H.cells_of(list(grid), int(draws), nA):
        sel = H.draw_cells(c.target, c.mode, c.split, n, d, nA)
        E_n = T43.coef_n(c, sel, kappa)
        for seed in range(int(seeds)):
            order = T43.ctx_order(c.macro_src, c.target, c.mode, c.split, seed)
            src, tsel, _ = T43.context_rows(c, order, sel, FULL_SET, FULL_CTX, n, d)
            if len(src) + len(tsel) == 0:
                continue
            X, _ = T43.build_X(c, src, tsel, 1)
            for mth in ("D0", "R1"):
                y = T43.build_y(c, src, tsel, 1, "D" if mth == "D0" else "R", None if mth == "D0" else E_n)
                out[(mth + "@full", int(n), int(d), int(seed))] = T43.ctx_sha(X, y)
    return out


def sha_check(rows, ref):
    """runs 행의 tabicl ctx_sha 와 기준 {(방법, n, 추출, seed): sha} 의 대조. 반환 (비교 수, 불일치 수)."""
    mine = {}
    for r in rows:
        if r.get("learner") == TI and r.get("ctx_sha"):
            mine[(str(r["method"]), int(r["n"]), int(r["draw"]), int(r["seed"]))] = str(r["ctx_sha"])
    common = [k for k in mine if k in ref]
    return len(common), int(sum(mine[k] != ref[k] for k in common))


def run_unit_f(a, u, dry=False):
    """작업 단위 u = (작업, 축, 대상, 모드, 분할, 부분, 격자, 추출, seed 수)를 실행한다. dry 이면 수만 센다."""
    job, axis, t, m, sp, part, grid, draws, seeds = u
    t0 = time.time()
    HA = ha_of(a, u)
    D = H.get_data(HA)
    c = H.build_ctx(D, HA, t, m, sp)
    torch = None
    if not dry:
        torch = cuda_backend()
        torch.cuda.reset_peak_memory_stats()
        p_ = shard_paths_f(a, u)
        for k_ in ("unit", "cells"):                                      # 이전 세대의 완료 표지와 셀 파일을 먼저 지운다
            p_[k_].unlink(missing_ok=True)
    loc = T43.eval_cells(D, c, t, sp) if (a.save_cells and not dry) else None
    rows, st, stats, cells = run_ctx_f(c, axis, part, a, HA, dry=dry, loc=loc)
    if dry:
        return dict(job=job, J=JOB_J.get(job, job), axis=axis, target=t, mode=m, split=int(sp), part=part, n_A=c.meta["n_A"], n_eval=c.meta["n_eval"],
                    nb_eval=c.meta["nb_eval"], n_src=c.meta["n_src"], valid=c.meta["valid"], n_rows=stats["n_rows"], n_ctx_max=stats["n_ctx_max"],
                    n_ctx_min=stats["n_ctx_min"], ctx_dev_max=stats["ctx_dev_max"], ctx_dev_bound=stats["ctx_dev_bound"],
                    n_src_regions=stats["n_src_regions"], n_tsub=int(stats["flags"].get("tsub", 0)), n_check_fail=int(stats["n_check_fail"]),
                    n_allnan_cols_max=int(stats["n_allnan_cols_max"]), _detail=stats["n_fit_detail"], _rows=stats["rows_detail"],
                    _est=stats["est_detail"], _errors=stats["errors"])
    if st is None:
        raise RuntimeError(f"{unit_name_f(u)}: 채점 셀 또는 A 셀이 없다")
    if a.smoke:                                                           # 스모크 점검(1): h43 함수로 독립 재계산한 ctx_sha
        ref = h43_ctx_shas(c, a, grid, draws, seeds) if axis == "main" else full_ctx_shas(c, a, grid, draws, seeds, HA.kappa)
        n_cmp, n_bad = sha_check(rows, ref)
        stats["smoke_check"].append(dict(check="ctx_sha_h43", ctx_set=axis, value=int(n_bad), tol=0, ok=bool(n_cmp > 0 and n_bad == 0), n_keys=int(n_cmp)))
    peak = float(torch.cuda.max_memory_allocated()) / MIB
    resv = float(torch.cuda.max_memory_reserved()) / MIB
    unit = write_shard_f(a, u, c, rows, st, stats, cells, time.time() - t0,
                         extra=dict(env=T43.env_info(gpu=True), gpu_mem_peak_mib=round(max(peak, stats["gpu_fit_peak_mib"]), 1),
                                    gpu_mem_reserved_peak_mib=round(max(resv, stats["gpu_fit_reserved_peak_mib"]), 1), rss_max_mib=rss_mib(),
                                    run_id=str(getattr(a, "RUN_ID", "") or ""), gpu_uuid=str(T43._WINFO.get("gpu_uuid", "")),
                                    device_check=str(T43._WINFO.get("device_check", "")), model_sha1_file=T43.file_sha(a.MODEL),
                                    has_flash_attn3=_has_fa3(), tabicl_impl="stock" if (a.tabicl_stock or _FORCE_STOCK[0]) else "cached"))
    del rows, st, cells
    T43._free_cuda()
    return unit


def _has_fa3():
    mod = sys.modules.get("tabicl.model.attention")
    return bool(getattr(mod, "HAS_FLASH_ATTN3", False)) if mod is not None else None


# ================================================================ 작업과 단위 열거
def job_specs(a):
    """작업별 (작업, 축, 부분, 대상, 격자, 추출)의 목록과 건너뛴 작업의 사유. 축소(S2, S3, S4)는 a.REDUCE 로 적용한다."""
    grid = list(a.N_USER)
    pos = [n for n in grid if n != 0]
    n0 = [0] if 0 in grid else []
    dm, df_ = axis_draws(a, "main"), axis_draws(a, "full")
    fpos = [n for n in FULL_POS_GRID if n in grid]
    specs, notes = [], []
    if a.smoke:                                                           # 스모크: 두 축 모두 부분 all
        tg = a.TARGET_FILTER or [("Russia_W", "x"), ("Canada", "x")]
        return [dict(job="smoke", axis="main", part="all", targets=tg, grid=grid, draws=dm),
                dict(job="smoke", axis="full", part="all", targets=tg, grid=n0 + fpos, draws=min(df_, dm))], notes
    for job in a.JOBS:
        if job == "f_n0":
            specs += [dict(job=job, axis="main", part="n0", targets=MAIN4_X, grid=n0, draws=dm),
                      dict(job=job, axis="full", part="n0", targets=MAIN4_X, grid=n0, draws=df_)]
        elif job == "f_t1":
            specs.append(dict(job=job, axis="main", part="pos", targets=MAIN4_X, grid=pos, draws=dm))
        elif job == "f_full":
            if "S2" in a.REDUCE:
                notes.append("f_full(J8): 축소 S2 로 건너뛴다")
                continue
            specs.append(dict(job=job, axis="full", part="pos", targets=MAIN4_X, grid=fpos, draws=df_))
        elif job == "f_t2ak":
            specs.append(dict(job=job, axis="main", part="all", targets=[(H.ALASKA, "x")], grid=grid, draws=dm))
        elif job == "f_t2":
            if "S4" in a.REDUCE:
                notes.append("f_t2(J11): 축소 S4 로 건너뛴다")
                continue
            specs.append(dict(job=job, axis="main", part="all", targets=T2_TARGETS, grid=grid, draws=min(dm, 2) if "S3" in a.REDUCE else dm))
        elif job == "f_t3":
            specs.append(dict(job=job, axis="main", part="all", targets=T3_TARGETS, grid=grid, draws=dm))
    if a.TARGET_FILTER:
        keep = set(a.TARGET_FILTER)
        for sp_ in specs:
            sp_["targets"] = [p for p in sp_["targets"] if p in keep]
    return specs, notes


def precheck_units(a):
    """사전 점검 단위(계획서 §6.3 단계 4): 추출 0, seed 0."""
    g3 = (0, 40, -1)
    return [("pre", "main", "Lena", "x", 1, "all", g3, 1, 1), ("pre", "full", "Lena", "x", 1, "all", g3, 1, 1),
            ("pre", "main", H.ALASKA, "x", 5, "all", g3, 1, 1), ("pre", "full", "Russia_W", "x", 1, "all", (0,), 1, 1)]


def enumerate_f(a):
    """작업 단위 목록과 건너뛴 분할의 기록. 유효 분할은 h43.enumerate_t 와 같은 규칙이다. 순서: 작업, 대상, 분할, 축."""
    if a.precheck:
        return list(precheck_units(a)), []
    specs, notes = job_specs(a)
    for nt in notes:
        print(f"[plan] {nt}", flush=True)
    units, skipped, order = [], [], {}
    for sp_ in specs:
        if not sp_["grid"] or not sp_["targets"]:
            continue
        A = a43_args(a, sp_["grid"], sp_["draws"], a.seeds, targets=sp_["targets"])
        us, sk = T43.enumerate_t(A)
        tord = {tuple(p): i for i, p in enumerate(sp_["targets"])}
        jidx = JOBS.index(sp_["job"]) if sp_["job"] in JOBS else 0
        for _, t, m, s in us:
            u = (sp_["job"], sp_["axis"], t, m, int(s), sp_["part"], tuple(int(n) for n in sp_["grid"]), int(sp_["draws"]), int(a.seeds))
            units.append(u)
            order[u] = (jidx, tord.get((t, m), 99), int(s), AXES_F.index(sp_["axis"]))
        skipped += [dict(s_, job=sp_["job"], axis=sp_["axis"], part=sp_["part"]) for s_ in sk]
    units.sort(key=lambda u: order[u])
    return units, skipped


# ================================================================ 워커
_WF = None


def _worker_init_f(argv, gpu_queue, threads, run_dir="", ppid=0, run_id="", overrides=None):
    """워커 초기화(h43._worker_init_t 와 같은 방식). h43 의 GPU 확인 함수(gpu_guard_t)가 읽는 전역(_WINFO, _WT)을 이 워커의 값으로 둔다.
    gpu_queue 는 GPU 번호(정수, GPU 하나에 풀 하나) 또는 번호를 담은 큐다."""
    global _WF
    warnings.filterwarnings("ignore")
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    pds = T43._set_pdeathsig()
    if int(ppid) and os.getppid() != int(ppid):
        os._exit(0)
    gpu = "" if gpu_queue is None else (str(gpu_queue.get()) if hasattr(gpu_queue, "get") else str(int(gpu_queue)))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    for v in THREAD_VARS:
        os.environ[v] = str(int(threads))
    ensure_nice()
    _WF = parse_args(argv)
    _WF.RUN_ID = str(run_id or "")
    for k, v in (overrides or {}).items():
        setattr(_WF, k, v)
    T43._WINFO = dict(gpu=gpu, ppid=int(ppid) or os.getppid(), run_dir=str(run_dir), cuda_checked=False, pdeathsig=bool(pds))
    T43._WT = SimpleNamespace(gpu_mem_max_mib=int(_WF.gpu_mem_max_mib))
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
        T43._write_json(Path(run_dir) / f"worker__{os.getpid()}.json", dict(pid=os.getpid(), ppid=T43._WINFO["ppid"], gpu=gpu, start=T43._now(),
                                                                            pdeathsig=bool(pds), threads=int(threads), nice=_nice_now()))


UNIT_KEEP = ("job", "axis", "target", "mode", "split", "part", "n_A", "n_eval", "nb_eval", "n_src", "E0", "n_fit_total", "n_rows", "n_ctx_max",
             "elapsed_s", "wall_s", "device", "status", "valid", "gpu_mem_peak_mib", "gpu_mem_reserved_peak_mib", "rss_max_mib", "failed_learners",
             "run_id")


def unit_summary_f(uj):
    d = {k: uj.get(k) for k in UNIT_KEEP}
    d["n_fail"] = int(sum((uj.get("fail") or {}).values())) + int(uj.get("n_nonfinite_keys", 0) or 0)
    return d


def _worker_run_f(u):
    """작업 단위 하나(워커). h43._worker_run_t 와 같은 처리(GPU 재확인, 실행 중 표지, CUDA 문맥 이상 때 워커 종료, 예외의 내장 예외 변환)."""
    T43._orphan_check()
    name = unit_name_f(u)
    T43.gpu_guard_t()
    mk = T43._marker_path(name)
    if mk is not None:
        T43._write_json(mk, dict(unit=name, pid=os.getpid(), gpu=T43._WINFO.get("gpu"), start=T43._now()))
    t0 = time.time()
    try:
        uj = run_unit_f(_WF, u)
    except BaseException as e:                                            # noqa: BLE001
        if T43._WINFO.get("gpu") not in (None, "") and not T43.cuda_healthy():
            print(f"[worker] {name}: CUDA 문맥을 쓸 수 없다({repr(e)[:160]}). 워커를 끝낸다", flush=True)
            os._exit(76)
        if mk is not None:
            mk.unlink(missing_ok=True)
        if type(e).__module__ == "builtins":
            raise
        raise RuntimeError(f"{type(e).__name__}: {str(e)[:500]}") from None
    if mk is not None:
        mk.unlink(missing_ok=True)
    uj["wall_s"] = round(time.time() - t0, 1)
    return unit_summary_f(uj)


def unit_done_run_f(a, u):
    """이번 실행(run_id)에서 쓴 완료 조각의 unit dict. 없으면 None."""
    p = shard_paths_f(a, u)
    need = ("unit", "runs", "npz") if a.STORE else ("unit", "runs")
    if not (a.RUN_ID and all(p[k].exists() for k in need)):
        return None
    try:
        uj = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return None
    return uj if (uj.get("run_id") == a.RUN_ID and uj.get("cfg_hash") == H.cfg_hash(unit_cfg_f(a, u))) else None


def check_overwrite_f(a, units):
    prev = [u for u in units if shard_paths_f(a, u)["unit"].exists()]
    if prev and not (a.resume or a.overwrite or a.smoke or a.precheck):
        raise SystemExit(f"[거부] 완료 표지가 있는 조각 {len(prev)}개가 있다(예: {unit_name_f(prev[0])}). 이어 돌리려면 --resume, 덮어쓰려면 --overwrite")
    return prev


def exit_code_f(res):
    """종료 코드: 중단 130 > 실패 1 > 드레인 3 > 창 마감 4 > partial 2 > 0."""
    if res.get("interrupted"):
        return EXIT_INTERRUPT
    if res.get("n_fail"):
        return EXIT_FAIL
    if res.get("stop") == "drain":
        return EXIT_DRAIN
    if res.get("stop") == "window":
        return EXIT_WINDOW
    if res.get("n_partial"):
        return EXIT_PARTIAL
    return 0


# ================================================================ 적합 수(--count-only)
def count_only(a, units, skipped):
    """학습 없이 단위별 적합 수, 추정 시간(h43.est_fit_s('tabpfn') × r ∈ {0.5, 1, 1.5}, full 의 상한은 ×2), 행렬 점검을 낸다.
    주 설정의 적합 수를 lgt_count.csv 의 fit_main_tabpfn 과 (대상, 모드, 분할)별로 대조한다."""
    t0 = time.time()
    rows, errs = [], []
    for i, u in enumerate(units):
        r_ = run_unit_f(a, u, dry=True)
        d_, e_ = r_.pop("_detail"), r_.pop("_est")
        r_.pop("_rows")
        errs += [f"{unit_name_f(u)}: {v}" for v in r_.pop("_errors")]
        per = Counter()
        for k, v in d_.items():
            ax, lr, mth = (k.split("|") + ["", ""])[:3]
            per[f"fit_{lr}"] += v; per[f"fit_{lr}_{mth}"] += v
            per[f"est_{lr}_s"] += e_.get(k, 0.0)
        r_.update({k: (round(v, 1) if k.startswith("est_") else int(v)) for k, v in per.items()})
        base = float(per.get(f"est_{TI}_s", 0.0))
        hi = 1.5 * base * (2.0 if u[1] == "full" else 1.0)
        r_.update(est_tabicl_s_r05=round(0.5 * base, 1), est_tabicl_s_r10=round(base, 1), est_tabicl_s_hi=round(hi, 1), order=i)
        rows.append(r_)
    df = pd.DataFrame(rows).fillna(0) if rows else pd.DataFrame()
    a.OUT.mkdir(parents=True, exist_ok=True)
    tag = a.TAG
    T43._atomic_csv(df, a.OUT / f"{tag}_count.csv")
    summary = dict(units=len(units), skipped=len(skipped), jobs={}, lgt_compare={}, dry_build=bool(a.dry_build))
    print(f"[count-only] 작업 단위 {len(units)} · 건너뛴 분할 {len(skipped)} · {time.time() - t0:.0f}s", flush=True)
    if len(df):
        for c_ in (f"fit_{TI}", f"fit_{CBX}", f"est_{TI}_s", f"est_{CBX}_s"):
            if c_ not in df:
                df[c_] = 0
        g = df.groupby(["job", "J", "axis", "part"], as_index=False, sort=False).agg(
            units=("split", "size"), fit_tabicl=(f"fit_{TI}", "sum"), fit_cb=(f"fit_{CBX}", "sum"), est_s=("est_tabicl_s_r10", "sum"),
            est_lo_s=("est_tabicl_s_r05", "sum"), est_hi_s=("est_tabicl_s_hi", "sum"), est_cb_s=(f"est_{CBX}_s", "sum"), n_ctx_max=("n_ctx_max", "max"),
            n_allnan_max=("n_allnan_cols_max", "max"), n_check_fail=("n_check_fail", "sum"))
        for c_ in ("est_s", "est_lo_s", "est_hi_s", "est_cb_s"):
            g[c_.replace("_s", "_h")] = (g[c_] / 3600.0).round(2)
        print("[count-only] 작업·축·부분별 적합 수와 추정 시간(h, TabICL 은 TabPFN 식 × r, r = 0.5·1.0·1.5, full 상한 ×2)\n"
              + g.drop(columns=["est_s", "est_lo_s", "est_hi_s", "est_cb_s"]).to_string(index=False), flush=True)
        for j, gj in g.groupby("J", sort=False):
            summary["jobs"][str(j)] = dict(units=int(gj.units.sum()), fit_tabicl=int(gj.fit_tabicl.sum()), fit_cb=int(gj.fit_cb.sum()),
                                           est_gpu_h=round(float(gj.est_s.sum()) / 3600.0, 2), est_gpu_h_lo=round(float(gj.est_lo_s.sum()) / 3600.0, 2),
                                           est_gpu_h_hi=round(float(gj.est_hi_s.sum()) / 3600.0, 2), est_cb_h=round(float(gj.est_cb_s.sum()) / 3600.0, 2))
        tot = df[f"fit_{TI}"].sum()
        print(f"[count-only] TabICL 적합 {int(tot):,}건 · 추정 {df.est_tabicl_s_r10.sum() / 3600:.1f} GPU-h(범위 {df.est_tabicl_s_r05.sum() / 3600:.1f}–"
              f"{df.est_tabicl_s_hi.sum() / 3600:.1f}) · catboost_ctx {int(df[f'fit_{CBX}'].sum()):,}건 추정 {df[f'est_{CBX}_s'].sum() / 3600:.1f} h · "
              f"컨텍스트 행 최댓값 {int(df.n_ctx_max.max()):,} · 대상 행 부분 추출(tsub) {int(df.n_tsub.sum())}건", flush=True)
        if a.dry_build:
            print(f"[count-only] --dry-build: 행렬 점검 실패 {int(df.n_check_fail.sum())}건 · 모두 결측인 열 수 최댓값 {int(df.n_allnan_cols_max.max())}"
                  + (f" · {errs[:3]}" if errs else ""), flush=True)
        summary.update(n_fit_tabicl=int(tot), n_ctx_max=int(df.n_ctx_max.max()), n_tsub=int(df.n_tsub.sum()), n_check_fail=int(df.n_check_fail.sum()),
                       n_allnan_cols_max=int(df.n_allnan_cols_max.max()))
        summary["lgt_compare"] = compare_lgt_count(a, df)
        lc = summary["lgt_compare"]
        if lc.get("n_compared"):
            print(f"[count-only] lgt_count.csv 대조(주 설정 TabICL 적합 수 = TabPFN 적합 수): 비교 {lc['n_compared']} 단위, 불일치 {lc['n_mismatch']}"
                  + (f" · 예: {lc['mismatch'][:3]}" if lc["n_mismatch"] else ""), flush=True)
            if lc["n_mismatch"]:
                print("[count-only][warn] 적합 수가 LGT 와 다르다. 컨텍스트 규약 또는 범위가 다르다", flush=True)
        else:
            print(f"[count-only] lgt_count.csv 대조: {lc.get('note', '비교할 단위 없음')}", flush=True)
    H._atomic_text(a.OUT / f"{tag}_count_meta.json", json.dumps(dict(summary, errors=errs[:50], skipped=skipped, code_sha=T43.file_sha(__file__, 12),
                                                                      pins=a.PINS, jobs_arg=a.JOBS, reduce=a.REDUCE, model=str(a.MODEL),
                                                                      model_exists=bool(Path(a.MODEL).exists())),
                                                                 ensure_ascii=False, indent=1, default=str))
    return df, summary


def compare_lgt_count(a, df):
    """주 설정 단위의 TabICL 적합 수(부분 합)를 lgt_count.csv 의 fit_main_tabpfn 과 대조한다. 부분이 격자 전체를 덮고 추출 5 인 단위만."""
    p = Path(a.LGTDIR) / f"{a.lgt_tag}_count.csv"
    if not p.exists():
        return dict(note=f"{p} 없음", n_compared=0, n_mismatch=0, mismatch=[])
    try:
        ref = pd.read_csv(p, usecols=["axis", "target", "mode", "split", "fit_main_tabpfn"])
    except (OSError, ValueError) as e:
        return dict(note=f"읽기 실패 {repr(e)[:100]}", n_compared=0, n_mismatch=0, mismatch=[])
    ref = ref[ref["axis"] == "main"]
    rmap = {(str(t), str(m), int(s)): float(v) for t, m, s, v in zip(ref.target, ref["mode"], ref.split, ref.fit_main_tabpfn)}
    mine = df[df["axis"] == "main"]
    out, bad = 0, []
    full_grid = set(H._n_list(T43.FULL_GRID))
    for (t, m, s), g in mine.groupby(["target", "mode", "split"]):
        parts = set(g.part)
        if not (parts >= {"n0", "pos"} or "all" in parts) or set(a.N_USER) != full_grid or axis_draws(a, "main") != 5 or int(a.seeds) != 2:
            continue
        key = (str(t), str(m), int(s))
        if key not in rmap:
            continue
        out += 1
        got = float(g[f"fit_{TI}"].sum())
        if abs(got - rmap[key]) > 0.5:
            bad.append(dict(unit=f"{t}|{m}|s{s}", lgf=got, lgt=rmap[key]))
    return dict(n_compared=out, n_mismatch=len(bad), mismatch=bad, flag="mismatch" if bad else "")


# ================================================================ 스모크·사전 점검 산출(판정 없음)
def find_runs_only(a, axis):
    """스모크·사전 점검 조각(runs, unit 만)."""
    tag = axis_tag(a, axis)
    out = []
    for p in sorted(Path(a.SHARDS).glob(f"{tag}__gpu__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")
        if len(parts) != 6 or parts[0] != tag:
            continue
        b = str(p)[:-len("_unit.json")]
        if Path(b + "_runs.csv").exists():
            out.append(dict(unit=p, runs=Path(b + "_runs.csv"), tag=tag, axis=axis, target=parts[2], mode=parts[3], split=int(parts[4][1:]), part=parts[5]))
    return out


def _read_runs_meta(sh):
    """runs.csv 를 읽되 RMSE 열(rmse_cm, rmse_beq_cm, bias_cm)은 읽지 않는다(스모크·사전 점검은 NaN 이다)."""
    frames = []
    for s_ in sh:
        f = pd.read_csv(s_["runs"], dtype=dict(alpha=str, fit_flag=str, ctx_set=str, ctx_sha=str, offload_mode=str), keep_default_na=False,
                        na_values=["", "nan", "NaN"])
        f = f[[c_ for c_ in f.columns if c_ not in ("rmse_cm", "rmse_beq_cm", "bias_cm")]]
        if len(f):
            f["tag"] = s_["tag"]
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_timing(runs, units):
    """적합 1건 시간(축, 학습기, 방법, n 별)과 TabPFN 추정식 대비 비. 키마다 한 번 센다(λ 행 중복 제거). 재사용 행(fit_s 0)은 뺀다."""
    if not len(runs):
        return pd.DataFrame(), {}
    ml = runs[(runs.learner != "none") & (pd.to_numeric(runs.fit_s, errors="coerce") > 0)].copy()
    ml = ml.drop_duplicates(subset=["target", "mode", "split", "ctx_set", "method", "learner", "n", "draw", "seed"])
    if not len(ml):
        return pd.DataFrame(), {}
    ne = {(str(u["target"]), str(u["mode"]), int(u["split"])): int(u.get("n_eval", 0) or 0) for u in units}
    ml["n_eval"] = [ne.get((str(t), str(m), int(s)), 0) for t, m, s in zip(ml.target, ml["mode"], ml.split)]
    ml["est_s"] = [T43.est_fit_s(TP if lr == TI else lr, nc, nv) for lr, nc, nv in zip(ml.learner, ml.n_ctx, ml.n_eval)]
    ml["ratio"] = ml.fit_s.astype(float) / ml.est_s.clip(lower=1e-9)
    tab = ml.groupby(["ctx_set", "learner", "method", "n"], as_index=False).agg(
        n_fit=("fit_s", "size"), sec_mean=("fit_s", "mean"), sec_max=("fit_s", "max"), n_ctx_mean=("n_ctx", "mean"), n_eval_mean=("n_eval", "mean"),
        est_mean=("est_s", "mean"), gpu_mem_fit_mib_max=("gpu_mem_fit_mib", "max"), rss_mib_max=("rss_mib", "max"))
    for c_ in ("sec_mean", "sec_max", "n_ctx_mean", "n_eval_mean", "est_mean"):
        tab[c_] = tab[c_].astype(float).round(3)
    ratio = {f"{cs}|{lr}": round(float(g_.ratio.median()), 3) for (cs, lr), g_ in ml.groupby(["ctx_set", "learner"])}
    ratio.update({lr: round(float(g_.ratio.median()), 3) for lr, g_ in ml.groupby("learner")})
    return tab, ratio


def smoke_summary(a):
    """스모크 산출: lgf_smoke_fit_timing.csv, lgf_smoke_check.csv. 경로 점검 값, 시간, GPU 메모리, 최대 RSS, 적합 수만 낸다."""
    sh = find_runs_only(a, "main") + find_runs_only(a, "full")
    if not sh:
        print("[smoke] 조각 없음", flush=True)
        return None
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    runs = _read_runs_meta(sh)
    tab, ratio = fit_timing(runs, units)
    O = a.OUT
    T43._atomic_csv(tab, O / f"{a.tag}_smoke_fit_timing.csv")
    chk = []
    for s_, uj in zip(sh, units):
        for q in uj.get("smoke_check") or []:
            chk.append(dict(unit=f"{s_['axis']}|{s_['target']}|{s_['mode']}|s{s_['split']}|{s_['part']}", **q))
        chk.append(dict(unit=f"{s_['axis']}|{s_['target']}|{s_['mode']}|s{s_['split']}|{s_['part']}", check="unit", ctx_set=s_["axis"],
                        value=np.nan, tol=np.nan, ok=bool(uj.get("status") == "ok"), status=uj.get("status"), n_fit=int(uj.get("n_fit_total", 0)),
                        gpu_mem_peak_mib=uj.get("gpu_mem_peak_mib"), gpu_mem_reserved_peak_mib=uj.get("gpu_mem_reserved_peak_mib"),
                        rss_max_mib=uj.get("rss_max_mib"), elapsed_s=uj.get("elapsed_s"), device_check=uj.get("device_check"),
                        tabicl_impl=uj.get("tabicl_impl"), has_flash_attn3=uj.get("has_flash_attn3"), amp_counts=json.dumps(uj.get("amp_counts", {})),
                        offload_counts=json.dumps(uj.get("offload_counts", {}))))
    ck = pd.DataFrame(chk)
    T43._atomic_csv(ck, O / f"{a.tag}_smoke_check.csv")
    for c_ in ("ctx_sha_h43", "cached_vs_stock", "chunk2", "cb_pool", "r1_eq_r0_n0"):
        q = ck[ck.check == c_] if len(ck) else ck
        if len(q):
            print(f"[smoke] {c_}: 최대 {pd.to_numeric(q.value, errors='coerce').max()} · 모두 통과 {bool(q.ok.astype(bool).all())} · 항목 {len(q)}", flush=True)
        else:
            print(f"[smoke] {c_}: 기록 없음", flush=True)
    uq = ck[ck.check == "unit"] if len(ck) else ck
    if len(uq):
        print("[smoke] 단위: " + "; ".join(f"{r_.unit} 상태 {r_.status} 적합 {r_.n_fit} GPU {r_.gpu_mem_peak_mib}/{r_.gpu_mem_reserved_peak_mib} MiB "
                                          f"RSS {r_.rss_max_mib} MiB {r_.elapsed_s}s" for r_ in uq.itertuples()), flush=True)
    if len(tab):
        print("[smoke] 적합 1건 시간(s, 스모크 점검 시간 제외)\n" + tab.to_string(index=False), flush=True)
    print(f"[smoke] 실측/추정식 비(중앙값): {ratio}", flush=True)
    return dict(check=ck, timing=tab, ratio=ratio, n_fail=int(sum(1 for u in units if u.get("status") != "ok")))


def smoke_forced_stock(a):
    """스모크 점검에서 캐시 하위 클래스의 차가 허용을 넘었는지(lgf_smoke_check.csv). 넘었으면 본 실행·사전 점검에 원래 클래스를 강제한다."""
    p = Path(a.OUT) / f"{a.tag}_smoke_check.csv"
    if not p.exists():
        return False
    try:
        ck = pd.read_csv(p)
    except (OSError, ValueError):
        return False
    q = ck[ck["check"] == "cached_vs_stock"] if "check" in ck else ck.iloc[0:0]
    return bool(len(q) and (~q["ok"].astype(bool)).any())


def projection_acc(a, jobs, reduce, r_main, r_full, r_cb):
    """작업(J)별 적합 수와 추정 GPU-h(TabPFN 추정식 × 실측 r, catboost_ctx 포함). 학습 없이 단위 수만 센다(dry). reduce = 적용할 축소(S2–S4)."""
    a2 = parse_args(["--jobs", ",".join(jobs), "--reduce", str(reduce), "--threads", str(int(a.threads)), "--out-dir", str(a.OUT),
                     "--data-dir", str(a.PROC), "--lgt-dir", str(a.LGTDIR), "--count-only"])
    units2, _ = enumerate_f(a2)
    acc = {}
    for u in units2:
        r_ = run_unit_f(a2, u, dry=True)
        e_ = r_["_est"]
        rI = r_full if u[1] == "full" else r_main
        s_I = sum(v for k, v in e_.items() if k.split("|")[1] == TI)
        s_C = sum(v for k, v in e_.items() if k.split("|")[1] == CBX)
        nI = sum(v for k, v in r_["_detail"].items() if k.split("|")[1] == TI)
        q = acc.setdefault(JOB_J[u[0]], dict(n_units=0, n_fit_tabicl=0, est_formula_h=0.0, est_gpu_h=0.0, est_cb_h=0.0))
        q["n_units"] += 1; q["n_fit_tabicl"] += int(nI); q["est_formula_h"] += s_I / 3600.0
        q["est_gpu_h"] += (s_I * rI + s_C * r_cb) / 3600.0; q["est_cb_h"] += s_C * r_cb / 3600.0
    return acc


def write_projection_f(a, rows):
    """lgf_projection.csv 의 F 행을 바꾼다(열 형식 script, row, est_gpu_h 는 h48 의 N 행과 같다). 창 파일 잠금 아래에서 읽고 쓴다."""
    p = a.OUT / "lgf_projection.csv"
    with window_lock(a):
        old = pd.DataFrame()
        if p.exists():
            try:
                old = pd.read_csv(p)
            except (OSError, ValueError):
                old = pd.DataFrame()
        if len(old) and "script" in old:
            old = old[old.script.astype(str) != "F"]
        elif len(old):
            old = pd.DataFrame()                                          # 형식이 다른 옛 파일(script 열 없음)은 버린다
        new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
        T43._atomic_csv(new, p)
    return new


def precheck_summary(a):
    """사전 점검 산출: lgf_precheck_timing.csv, lgf_projection.csv 의 F 행(r = 실측/식 비의 중앙값, 작업별 추정 GPU-h)."""
    sh = find_runs_only(a, "main") + find_runs_only(a, "full")
    if not sh:
        print("[precheck] 조각 없음", flush=True)
        return None
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    runs = _read_runs_meta(sh)
    ml = runs[(runs.learner != "none") & (pd.to_numeric(runs.fit_s, errors="coerce") > 0)].drop_duplicates(
        subset=["target", "mode", "split", "ctx_set", "method", "learner", "n", "draw", "seed"]).copy()
    ne = {(str(u["target"]), str(u["mode"]), int(u["split"])): int(u.get("n_eval", 0) or 0) for u in units}
    ml["n_eval"] = [ne.get((str(t), str(m), int(s)), 0) for t, m, s in zip(ml.target, ml["mode"], ml.split)]
    ml["unit"] = [f"{ax}|{t}|{m}|s{s}" for ax, t, m, s in zip(ml.ctx_set, ml.target, ml["mode"], ml.split)]
    pt = ml.rename(columns=dict(gpu_mem_fit_mib="gpu_mem_max_mib", rss_mib="rss_max_mib"))[
        ["unit", "learner", "method", "n", "n_ctx", "n_eval", "fit_s", "gpu_mem_max_mib", "rss_max_mib"]]
    T43._atomic_csv(pt, a.OUT / f"{a.tag}_precheck_timing.csv")
    _, ratio = fit_timing(runs, units)
    r_main = ratio.get(f"main|{TI}", ratio.get(TI, 1.0)); r_full = ratio.get(f"full|{TI}", r_main); r_cb = ratio.get(CBX, 1.0)
    acc = projection_acc(a, JOBS, "", r_main, r_full, r_cb)
    acc_s3 = projection_acc(a, ("f_t2",), "S3", r_main, r_full, r_cb)     # S3(J11 추출 5 → 2) 적용 뒤의 J11
    now = _now_iso()
    rows = [dict(script="F", row=j, job=[k for k, v in JOB_J.items() if v == j][0], r_tabicl_main=r_main, r_tabicl_full=r_full, r_cb=r_cb,
                 **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()}, time=now, note="사전 점검 실측 r 과 TabPFN 추정식")
            for j, q in acc.items()]
    gh = {j: float(q["est_gpu_h"]) for j, q in acc.items()}
    j11_s3 = float(acc_s3.get("J11", {}).get("est_gpu_h", 0.0))
    for row, job, v, note in (("P_F", "J1,J6,J8,J9,J11", sum(gh.get(j, 0.0) for j in ("J1", "J6", "J8", "J9", "J11")), "F 기본 범위의 측정 기반 추정"),
                              ("S2", "J8", gh.get("J8", 0.0), "줄어드는 양: full 변형의 n > 0(J8) 제외"),
                              ("S3", "J11", gh.get("J11", 0.0) - j11_s3, "줄어드는 양: J11 의 추출 5 → 2"),
                              ("S4", "J11", j11_s3, "줄어드는 양: S3 뒤 남은 J11 제외")):
        rows.append(dict(script="F", row=row, job=job, r_tabicl_main=r_main, r_tabicl_full=r_full, r_cb=r_cb, est_gpu_h=round(v, 3), time=now, note=note))
    write_projection_f(a, rows)
    print(f"[precheck] 적합 {len(ml)}건 · r(TabICL main) {r_main} · r(full) {r_full} · r(CatBoost) {r_cb} · GPU 메모리 최댓값 "
          f"{pd.to_numeric(ml.gpu_mem_fit_mib, errors='coerce').max()} MiB · 최대 RSS {pd.to_numeric(ml.rss_mib, errors='coerce').max()} MiB", flush=True)
    print("[precheck] 작업별 추정(GPU-h, 적합 + catboost_ctx)과 축소량: " + ", ".join(f"{r_['row']} {r_['est_gpu_h']}" for r_ in rows), flush=True)
    return dict(timing=pt, projection=pd.DataFrame(rows), n_fail=int(sum(1 for u in units if u.get("status") != "ok")))


# ================================================================ 집계: LGT 짝 게이트(계획서 §3.4)
GATE_CFG_KEYS = ("ctx_max", "ctx_reserve", "ctx_src_min", "dup", "kappa", "seeds", "draws", "n_grid", "subregion_map")


def gate_cfg_diff(cfg_lgf, cfg_lgt):
    """게이트 조건 2: 컨텍스트 매개변수와 하위 지역 대응표 이름의 대조. LGT 설정에 dup 이 없으면 주 설정(axis main)의 h43.MAIN_SET dup 을 쓴다.
    n 격자는 LGF 의 n_grid_axis 와 비교한다. 반환 다른 항목 목록."""
    diff = []
    for k in GATE_CFG_KEYS:
        mine = cfg_lgf.get("n_grid_axis", cfg_lgf.get("n_grid")) if k == "n_grid" else cfg_lgf.get(k)
        other = cfg_lgt.get(k)
        if k == "dup" and other is None and str(cfg_lgt.get("axis", "main")) == "main":
            other = int(T43.MAIN_SET["dup"])
        if json.dumps(mine, sort_keys=True, default=str) != json.dumps(other, sort_keys=True, default=str):
            diff.append(f"{k}: LGF {mine} / LGT {other}")
    return diff


def _lgt_sha_rows(runs_path):
    """LGT runs.csv 에서 짝 확인에 필요한 열만 읽는다(RMSE 열은 읽지 않는다)."""
    cols = ["method", "learner", "alpha", "n", "draw", "seed", "ctx_set", "ctx_sha"]
    try:
        return pd.read_csv(runs_path, usecols=cols, dtype=dict(alpha=str, ctx_set=str, ctx_sha=str), keep_default_na=False, na_values=[""])
    except (OSError, ValueError):
        return pd.DataFrame(columns=cols)


def _sha_map(df, learner):
    q = df[(df.learner == learner) & (df.ctx_set.astype(str) == "main") & (df.ctx_sha.astype(str) != "")] if len(df) else df
    return {(str(m), int(n), int(d), int(s)): str(h) for m, n, d, s, h in zip(q.method, q.n, q.draw, q.seed, q.ctx_sha)}


def gate_lgt(a, X, stores, runs, lgf_units, lgt_shards=None):
    """(대상, 모드, 분할)마다 LGT 짝 게이트의 다섯 조건을 확인한다. 통과한 단위의 tabpfn 키(D0, R0, R1)만 저장소 사본에 병합한다.
    보조로 catboost_ctx 를 키별로 대조한다(0.02 cm, 표시만). 반환 (gate 표, 병합한 저장소 dict)."""
    ref = {(s_["target"], s_["mode"], int(s_["split"])): s_ for s_ in (lgt_shards if lgt_shards is not None else
                                                                   X.find_shards_x(Path(a.LGTDIR) / "shards", a.lgt_tag)) if s_["part"] == "gpu"}
    cfg_by, meta_by = {}, {}
    for uj in lgf_units:
        key = (str(uj.get("target")), str(uj.get("mode")), int(uj.get("split", -1)))
        if uj.get("axis") == "main" and key not in cfg_by:
            cfg_by[key] = uj.get("cfg", {})
            meta_by[key] = dict(threads=uj.get("threads"), code_sha_h43=str(uj.get("code_sha_h43", "")))
    lgf_sha = {}
    if len(runs):
        q = runs[(runs.learner == TI) & (runs.ctx_set.astype(str) == "main") & (runs.ctx_sha.astype(str) != "")]
        for t, m, sp, mth, n, d, s, h in zip(q.target, q["mode"], q.split, q.method, q.n, q.draw, q.seed, q.ctx_sha):
            lgf_sha.setdefault((str(t), str(m), int(sp)), {})[(str(mth), int(n), int(d), int(s))] = str(h)
    rows, out = [], {}
    for (nm, sp), st in sorted(stores.items()):
        t, m = nm.split("|")
        key = (t, m, int(sp))
        row = dict(target=t, mode=m, split=int(sp), status="", lgt_status="", cfg_ok=False, cfg_diff="", p01_max_diff=np.nan, n_missing_ref=np.nan,
                   blocks_equal=None, n_sha_keys=0, n_sha_mismatch=0, passed=False, n_tabpfn_merged=0, cb_n_keys=0, cb_max_diff=np.nan, cb_flag="",
                   threads_lgf=meta_by.get(key, {}).get("threads"), threads_lgt=None, code_sha_h43_lgf=meta_by.get(key, {}).get("code_sha_h43", ""),
                   code_sha_h43_lgt="", code_sha_warn="", tol=GATE_TOL, cb_tol=CB_TOL)
        out[(nm, int(sp))] = st
        s_ = ref.get(key)
        if s_ is None:
            row["status"] = "LGT 조각 없음"; rows.append(row); continue
        try:
            uj = json.loads(Path(s_["unit"]).read_text())
        except (OSError, ValueError):
            row["status"] = "LGT unit.json 을 읽을 수 없음"; rows.append(row); continue
        row.update(lgt_status=str(uj.get("status", "")), threads_lgt=uj.get("threads"), code_sha_h43_lgt=str(uj.get("code_sha", "")))
        if row["code_sha_h43_lgf"] and row["code_sha_h43_lgt"] and row["code_sha_h43_lgf"][:12] != row["code_sha_h43_lgt"][:12]:
            row["code_sha_warn"] = "h43 코드 해시가 다르다(짝 판단은 조건 4)"
            print(f"  [warn] {t}|{m}|s{sp}: LGF 의 h43 {row['code_sha_h43_lgf'][:12]} 과 LGT 의 code_sha {row['code_sha_h43_lgt'][:12]} 가 다르다", flush=True)
        reasons = []
        if row["lgt_status"] not in ("ok", "partial"):                    # 조건 1
            reasons.append(f"LGT 상태 {row['lgt_status'] or '없음'}")
        cfg_l = cfg_by.get(key)
        if cfg_l is None:
            reasons.append("LGF 주 설정 조각 없음")
        else:
            diff = gate_cfg_diff(cfg_l, uj.get("cfg", {}))                # 조건 2
            row.update(cfg_ok=not diff, cfg_diff="; ".join(diff))
            if diff:
                reasons.append("설정 불일치")
        try:
            sb = load_stores(s_["npz"]).get((nm, int(sp)))
        except Exception as e:                                            # noqa: BLE001
            sb = None
            reasons.append(f"LGT 저장소 오류 {repr(e)[:80]}")
        if sb is None:
            reasons.append("LGT 저장소 없음")
        else:
            gc_ = T43.gate_compare(st, sb)                                # 조건 3(P0·P1)과 5(채점 블록)
            row.update(p01_max_diff=gc_["max_diff"], n_missing_ref=gc_["n_missing_ref"], blocks_equal=gc_["blocks_equal"])
            if not (gc_["n_keys"] and gc_["n_missing_ref"] == 0 and gc_["max_diff"] <= GATE_TOL):
                reasons.append("P0·P1 불일치")
            if not gc_["blocks_equal"]:
                reasons.append("채점 블록 불일치")
            cb = [k for k in st.keys if k[1] == CBX and k[0] in ("D0", "R0", "R1") and k in sb]     # 보조: catboost_ctx 대조(표시만)
            if cb:
                dd = [abs(st.rmse(k) - sb.rmse(k)) for k in cb]
                row.update(cb_n_keys=len(cb), cb_max_diff=float(np.nanmax(dd)), cb_flag="허용 차 초과(표시만)" if np.nanmax(dd) > CB_TOL else "")
        lm = _sha_map(_lgt_sha_rows(s_["runs"]), TP)                       # 조건 4: 키별 ctx_sha
        mine = lgf_sha.get(key, {})
        common = [k for k in mine if k in lm]
        nbad = int(sum(mine[k] != lm[k] for k in common))
        row.update(n_sha_keys=len(common), n_sha_mismatch=nbad)
        if not common:
            reasons.append("ctx_sha 를 비교할 키가 없다")
        elif nbad:
            reasons.append(f"ctx_sha 불일치 {nbad}/{len(common)}")
        row["passed"] = not reasons
        row["status"] = "ok" if not reasons else "; ".join(reasons)
        if row["passed"]:
            S, C_ = st.matrices(st.keys)
            cp = BlockStore._from_arrays(st.target, st.split, st.blocks, st.ncell, list(st.keys), S, C_, dict(st.meta))
            nmg = 0
            for k in sb.keys:
                if k[1] == TP and k[0] in ("D0", "R0", "R1") and str(k[2]) == "1" and k[3] == "cell":
                    cp.add_sse(k, *sb.get(k)); nmg += 1
            row["n_tabpfn_merged"] = nmg
            out[(nm, int(sp))] = cp
        rows.append(row)
    return pd.DataFrame(rows), out


# ================================================================ 집계: 판정 도우미
def halfwidth(r):
    """95 % CI 반폭 = max(셀 가중 폭, 블록 등가중 폭)/2. 끝값이 비유한이면 NaN."""
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


def prec_count(X, res, delta):
    """판정이 나온 대비 가운데 정밀도 미달(반폭 > 한계)의 수와 판정이 나온 대비의 수."""
    ok = [r for _, r in res if X.valid_row(r)]
    return int(sum(not precision_ok(r, delta) for r in ok)), len(ok)


def prec_suffix(X, res, delta):
    k, m = prec_count(X, res, delta)
    return f"(정밀도 미달 대비 {k}/{m})" if k else ""


def undetermined_all_imprecise(X, res, delta):
    und = [r for _, r in res if X.valid_row(r) and r["verdict4"] == "미결정"]
    return bool(und) and all(not precision_ok(r, delta) for r in und)


def f1_verdict(X, used):
    """LGF-F1 판정(분할 완결성 전). used = [(이름, 대비 행)] 두 개. 반환 (문구, support_class). 둘 다 우세가 아니면 지지, 하나라도 우세이면 기각.
    지지 A: 두 대비가 모두 열세 또는 동등. 지지 B: 우세가 없고 미결정이 하나라도 있다(CI 를 적는다). 판정이 나오지 않은 대비가 있으면 판정 불가이되
    판정이 나온 대비가 이미 우세이면 '기각(일부 대비 판정 불가, 대비 k/2)'."""
    k, m, _ = X.count_valid(used)
    vs = [X._v(r) for _, r in used]
    if any(v == "우세" for v in vs):
        if k < m:
            return f"기각(일부 대비 판정 불가, 대비 {k}/{m})", ""
        return ("기각: TabICL v2 직접 예측이 라벨 0 에서 원천 계수 물리식보다 우세하다. L1 의 라벨 0 부분 결론을 'TabICL 을 제외한 학습기 한정'으로 "
                "적고 TabICL 의 D0 곡선을 플랫폼 표지와 로컬 D0[C] 곡선과 함께 마스터 곡선에 더한다"), ""
    if k < m:
        return X.na_text(used), ""
    if all(v in ("열세", "동등") for v in vs):
        return ("지지(물리식보다 오차가 크거나 구별되지 않음): 표형 파운데이션 모델 TabICL v2 를 학습기로 써도(컨텍스트 상한 10,000행과 원천 전체의 "
                "두 조건) 직접 ML 은 라벨 0 전이에서 원천 계수 물리식을 넘지 못한다"), "A"
    ci = "; ".join(f"{lab} CI [{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}] cm" for lab, r in used)
    return (f"지지(우세 근거 없음, {ci}): 라벨 0 전이에서 TabICL v2 직접 ML 이 원천 계수 물리식보다 낫다는 근거는 관찰되지 않았다(두 조건)"), "B"


def l1_region_hits(X, ns, tms, names, gA_fn, gB_fn, n_list):
    """L1 형식 지역 수: 지역 행의 4분 판정이 우세인 (지역, n). 반환 {지역: [n]}."""
    hits = {}
    for nm in names:
        tm = tms.get(nm)
        if tm is None:
            continue
        for n in n_list:
            s_ = X.region_stats(tm, gA_fn(n), gB_fn(n))
            if s_ is None:
                continue
            if X.stats_row(s_, ns, nm)["verdict4"] == "우세":
                hits.setdefault(nm, []).append(int(n))
    return hits


def pair_fn(lrA, lrB):
    """학습기 짝 대비용 저장소 변환: lrA·lrB 키는 상대 학습기의 같은 (method, alpha, placement, n, draw, seed, lam) 키가 있을 때만 남긴다."""
    other = {lrA: lrB, lrB: lrA}

    def fn(st):
        keep, drop = [], []
        for k in st.keys:
            if k[1] in other and (tuple(k[:1]) + (other[k[1]],) + tuple(k[2:])) not in st:
                drop.append(k)
            else:
                keep.append(k)
        return (T43._sub_store(st, keep), drop) if drop else (st, [])
    return fn


def _nl(n):
    return "all" if int(n) == -1 else str(int(n))


def _nt(n):
    return "전량" if int(n) == -1 else f"n={int(n)}"


def equiv_text(X, base, lam1, delta, who, dom_word="TabICL"):
    """F2 형식(L32 규칙)의 동등성 판정 문구. base = 기준 λ 대비, lam1 = λ 1.0 대비. 반환 (문구, 판정에 쓴 대비, 부분 표기)."""
    vb, v1 = [X._v(r) for _, r in base], [X._v(r) for _, r in lam1]
    k, m, _ = X.count_valid(base)
    k1, m1, _ = X.count_valid(lam1)
    dom = [(lab, r) for (lab, r), v in zip(base, vb) if v in ("우세", "열세")]
    dom1 = [f"{lab} {dom_word} {v}" for (lab, _), v in zip(lam1, v1) if v in ("우세", "열세")]
    used, part = base, None
    if dom:
        txt = f"{who} 의 차이가 있는 대비: " + ", ".join(f"{lab} {dom_word} {r['verdict4']}(Δ {float(r['delta']):+.2f} cm)" for lab, r in dom)
        used, part = dom, X.part_mark(base)
    elif k < m:
        return X.na_text(base), base, None
    elif all(v == "동등" for v in vb) and k1 < m1:
        return X.na_text(base + lam1), base + lam1, None
    elif all(v == "동등" for v in vb + v1):
        txt = f"동등(한계 {float(delta):g} cm, λ {float(H.LAM_BASE):g} 와 1.0)"
        used = base + lam1
    elif all(v == "동등" for v in vb) and dom1:
        txt = f"기준 λ {float(H.LAM_BASE):g} 에서는 동등, λ 1.0 에서는 차이가 있다({', '.join(dom1)}). 동등으로 쓰지 않는다"
        used = base + [(lab, r) for (lab, r), v in zip(lam1, v1) if v in ("우세", "열세")]
        dom1 = []
    else:
        used = base + lam1
        k_, m_ = prec_count(X, used, delta)
        txt = f"정밀도 미달(동등성 판정 불가, 대비 {k_}/{m_})" if undetermined_all_imprecise(X, used, delta) else "차이를 확인하지 못함"
    if not txt.startswith("판정 불가") and not txt.startswith("정밀도 미달"):
        txt += prec_suffix(X, used, delta)
    if dom1 and not txt.startswith("판정 불가"):
        txt += f" [λ 1.0 의 대비(병기): {', '.join(dom1)}]"
    return txt, used, part


# ================================================================ 집계: 가설 판정(LGF-F1–F6, 계획서 §3.6)
def build_tests_f(a, tms, D, floor, X=None, lgt_tests=None):
    """LGF-F1–F6 의 대비 행과 판정 행. 반환 표(열은 TEST_COLS 가 앞이다). Holm 묶음: conf(F1 의 2개), aux(27개), eq(F2·F3·F6 의 동등성 p, 17개)."""
    X = X or T43._load_h42()
    ns = SimpleNamespace(nboot=int(a.nboot), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux))
    T = X.TestBook(ns, tms, D, floor)
    V = T.verdict
    ALL, LB, DE = -1, float(H.LAM_BASE), float(a.delta_eq)
    memo: dict = {}
    views: dict = {"all": tms}
    unp: dict = {}
    fns = dict(pair_IC=pair_fn(TI, CBX), pair_IT=pair_fn(TI, TP), d01=T43.draw_filter(T43.AUX_DRAWS))

    def get_view(name):
        if name not in views:
            out, cnt_by = {}, {}
            for nm, tm in tms.items():
                t2, dr = T43.tm_view(tm, fns[name])
                cnt = Counter()
                if name.startswith("pair"):
                    for sp, ks in dr.items():
                        if sp in tm.used:
                            for k in ks:
                                cnt[(k[0], str(k[2]), k[3], int(k[4]), float(k[7]))] += 1
                out[nm], cnt_by[nm] = t2, cnt
            views[name], unp[name] = out, cnt_by
        return views[name]

    def n_unpaired(name, gA, gB, names):
        tab = unp.get(name, {})
        grp = {(q[0], str(q[2]), q[3], int(q[4]), float(q[5])) for q in (gA, gB)}
        return int(sum(tab.get(nm, Counter())[q] for nm in names for q in grp))

    def C(test, label, gA, gB, view="all", fam="", eq=False, prior="", blind=True, role="보조", names=None, aux3=True, **kw):
        k = (test, label, tuple(names or ()), view)
        if k not in memo:
            vw = get_view(view)
            if view.startswith("pair"):
                nms = list(names or T.m4) + ([T.m3[2]] if aux3 and not names else [])
                kw["n_unpaired"] = n_unpaired(view, gA, gB, nms)
            T.tms = vw
            try:
                memo[k] = T.contrast(test, ITEM, label, gA, gB, names=names, primary=False, blind=blind, role=role, aux3=aux3, key_set=view,
                                     holm_family=fam, eq_test=bool(eq), prior_info=prior, **kw)
            finally:
                T.tms = tms
        return memo[k]

    def single(test, label, nm, gA, gB, view="all", prior="", blind=True, role="보조(지역 행)", **kw):
        tm = get_view(view).get(nm)
        if view.startswith("pair"):
            kw["n_unpaired"] = n_unpaired(view, gA, gB, [nm])
        return T.single(test, ITEM, label, nm, X.region_stats(tm, gA, gB) if tm is not None else None, key_set=view, blind=blind, role=role,
                        prior_info=prior, **kw)

    def g(method, n, lr=None, lam=None):
        return X.G(method, n, lam, lr)

    P0 = g("P0", 0)
    rows_1000 = ["Lena|x", f"{H.ALASKA}|x"]
    CPRIOR = "재현(비맹검, M1·a2)"
    TMARK = "T 열: 등록 시점(개정 1)에 결과 존재(미열람)"
    res: dict = {}

    # ---------------- LGF-F1(확인적): D0[I] − P0, D0@full[I] − P0(n = 0, λ 1.0)
    f1a = C("LGF-F1", "D0[I]-P0|n0", g("D0", 0, TI), P0, n=0, lam=1.0, fam="conf", prior="M1", role="판정(확인적)")
    f1b = C("LGF-F1", "D0@full[I]-P0|n0", g("D0@full", 0, TI), P0, n=0, lam=1.0, fam="conf", prior="M1", role="판정(확인적)")
    c1a = C("LGF-F1", "D0[C]-P0|n0", g("D0", 0, CBX), P0, n=0, lam=1.0, blind=False, prior=CPRIOR, role="병기(C)")
    c1b = C("LGF-F1", "D0@full[C]-P0|n0", g("D0@full", 0, CBX), P0, n=0, lam=1.0, blind=False, prior=CPRIOR, role="병기(C)")
    used1 = [("D0[I]-P0", f1a), ("D0@full[I]-P0", f1b)]
    txt, sup = f1_verdict(X, used1)
    mk = T.marks(used1)
    if mk["splits"] is not None and not txt.startswith("판정 불가"):
        txt, sup = f"판정 불가(분할 {mk['splits'][0]}/{mk['splits'][1]}. 등록한 분할이 모두 채워진 뒤 판정한다)", ""
    res["F1"] = (txt, sup)
    V("LGF-F1", ITEM, txt, X._fmt(used1 + [("병기 D0[C]-P0", c1a), ("병기 D0@full[C]-P0", c1b)]), role="주", blind=True, used=used1,
      support_class=sup, clause="판정", prior_info="M1", confirmatory_verdict=True)
    aux_a = [(_nt(n), C("LGF-F1", f"D0[I]-P0|n{n}", g("D0", n, TI), P0, n=n, lam=1.0, prior="M1", role="보조(a)")) for n in (3, 10, 40)]
    dom_a = [lab for lab, r in aux_a if X._v(r) == "우세"]
    V("LGF-F1", ITEM, (f"보조 행 (a): 우세인 n {', '.join(dom_a)}. 그 n 을 L1 결론의 한계로 적는다" if dom_a else "보조 행 (a): 우세인 n 없음"),
      X._fmt(aux_a), role="보조", blind=True, used=aux_a, clause="(a) n ∈ {3, 10, 40}")
    hits = l1_region_hits(X, ns, tms, T.m4, lambda n: g("D0", n, TI), lambda n: P0, (0, 3, 10, 40))
    det = ", ".join(f"{nm.split('|')[0]}({','.join(str(v) for v in ns_)})" for nm, ns_ in hits.items())
    txt_b = f"L1 형식 지역 수 {len(hits)}/{len(T.m4)}" + (f". TabICL v2 는 n ≤ 40 에서 예외({det})" if len(hits) >= 2 else "")
    V("LGF-F1", ITEM, txt_b, det or "우세 지역 없음", role="보조", blind=True, clause="(b) L1 형식 지역 수", l1_regions=len(hits))

    # ---------------- LGF-F2: R1[I] − R1[C]
    l2n = (0, 10, 40, 160, ALL)
    base2 = [(_nt(n), C("LGF-F2", f"R1[I]-R1[C]|n{_nl(n)}|lam{LB}", g("R1", n, TI, LB), g("R1", n, CBX, LB), view="pair_IC", n=n, lam=LB, fam="aux",
                        eq=True)) for n in l2n]
    lam2 = [(f"{_nt(n)} λ=1.0", C("LGF-F2", f"R1[I]-R1[C]|n{_nl(n)}|lam1.0", g("R1", n, TI, 1.0), g("R1", n, CBX, 1.0), view="pair_IC", n=n, lam=1.0,
                                  role="보조(λ 1.0)", aux3=False)) for n in l2n]
    for n in (3, 320):
        C("LGF-F2", f"R1[I]-R1[C]|n{_nl(n)}|lam{LB}", g("R1", n, TI, LB), g("R1", n, CBX, LB), view="pair_IC", n=n, lam=LB, role="보조(추가 n)")
    for nm in rows_1000:
        single("LGF-F2", f"R1[I]-R1[C]|n1000|lam{LB}", nm, g("R1", 1000, TI, LB), g("R1", 1000, CBX, LB), view="pair_IC", n=1000, lam=LB)
    for m_, lam in (("D0", 1.0), ("R0", LB)):
        for n in l2n:
            C("LGF-F2", f"{m_}[I]-{m_}[C]|n{_nl(n)}", g(m_, n, TI, lam), g(m_, n, CBX, lam), view="pair_IC", n=n, lam=lam, role=f"보조({m_})", aux3=False)
    txt2, used2, part2 = equiv_text(X, base2, lam2, DE, "TabICL 과 같은 컨텍스트 CatBoost")
    res["F2"] = txt2
    V("LGF-F2", ITEM, txt2, X._fmt(base2 + lam2), role="보조", blind=True, used=used2, partial=part2,
      note="결측 서술 표(lgf_missing_desc.csv)를 함께 본다. TabICL 은 결측을 컨텍스트 평균으로 채운다")

    # ---------------- LGF-F3: TabICL v2 − TabPFN v2(게이트 통과 단위, 방향 중립)
    l3n = (0, 10, 40, 160, ALL)
    base3 = [(f"R1 {_nt(n)}", C("LGF-F3", f"R1[I]-R1[T]|n{_nl(n)}|lam{LB}", g("R1", n, TI, LB), g("R1", n, TP, LB), view="pair_IT", n=n, lam=LB,
                                fam="aux", eq=True, prior=f"b4; {TMARK}")) for n in l3n]
    base3.append(("D0 n=0", C("LGF-F3", "D0[I]-D0[T]|n0", g("D0", 0, TI), g("D0", 0, TP), view="pair_IT", n=0, lam=1.0, fam="aux", eq=True,
                              prior=f"b4; {TMARK}")))
    lam3 = [(f"R1 {_nt(n)} λ=1.0", C("LGF-F3", f"R1[I]-R1[T]|n{_nl(n)}|lam1.0", g("R1", n, TI, 1.0), g("R1", n, TP, 1.0), view="pair_IT", n=n,
                                     lam=1.0, role="보조(λ 1.0)", aux3=False, prior=TMARK)) for n in l3n]
    k3, m3, _ = X.count_valid(base3)
    dom3 = [(lab, r) for lab, r in base3 if X._v(r) in ("우세", "열세")]
    used3, part3 = [(lab, r) for lab, r in base3 if X.valid_row(r)], ([f"대비 {k3}/{m3}"] if 0 < k3 < m3 else None)
    if k3 == 0:
        txt3, used3 = X.na_text(base3), base3
    elif dom3:
        txt3 = "TabICL v2 와 TabPFN v2 의 차이가 있는 대비: " + ", ".join(f"{lab} TabICL {r['verdict4']}(Δ {float(r['delta']):+.2f} cm)" for lab, r in dom3)
        used3 = dom3
    else:
        v_ok = [X._v(r) for _, r in used3]
        v1 = [X._v(r) for _, r in lam3 if X.valid_row(r)]
        d1 = [f"{lab} TabICL {X._v(r)}" for lab, r in lam3 if X._v(r) in ("우세", "열세")]
        if all(v == "동등" for v in v_ok) and v1 and all(v == "동등" for v in v1) and len(v1) == sum(1 for lab, r in used3 if lab.startswith("R1")):
            txt3 = f"TabICL v2 와 TabPFN v2 의 차이는 한계 {DE:g} cm 안이다"
        elif all(v == "동등" for v in v_ok) and d1:
            txt3 = f"기준 λ {LB:g} 에서는 동등, λ 1.0 에서는 차이가 있다({', '.join(d1)}). 동등으로 쓰지 않는다"
        else:
            full = used3 + [(lab, r) for lab, r in lam3 if X.valid_row(r)]
            k_, m_ = prec_count(X, full, DE)
            txt3 = f"정밀도 미달(동등성 판정 불가, 대비 {k_}/{m_})" if undetermined_all_imprecise(X, full, DE) else "두 모델의 차이를 확인하지 못함"
            used3 = full
        if not txt3.startswith("정밀도 미달"):
            txt3 += prec_suffix(X, used3, DE)
    V("LGF-F3", ITEM, txt3, X._fmt(base3 + lam3), role="보조", blind=True, used=used3, partial=part3, prior_info=f"b4; {TMARK}",
      note="두 모델의 차이에는 입력 처리의 차이(TabICL 의 결측 평균 대치, TabPFN 의 cci_valid 범주형 추론)가 포함된다. 결측 서술 표를 함께 본다. "
           "LGT 를 열람한 뒤 판정하면 'TabPFN 쪽 비맹검'을 붙인다")

    # ---------------- LGF-F4: R1[I] − P1(순가치), 전량 R1[I] − P0
    l4n = (3, 10, 40, 160, 320, ALL)
    resI = [(_nt(n), n, C("LGF-F4", f"R1[I]-P1|n{_nl(n)}", g("R1", n, TI), g("P1", n), n=n, lam=LB, fam="aux")) for n in l4n]
    resC = [(_nt(n), n, C("LGF-F4", f"R1[C]-P1|n{_nl(n)}", g("R1", n, CBX), g("P1", n), n=n, lam=LB, blind=False, prior=CPRIOR, role="병기(C)"))
            for n in l4n]
    resT = [(_nt(n), n, C("LGF-F4", f"R1[T]-P1|n{_nl(n)}", g("R1", n, TP), g("P1", n), n=n, lam=LB, blind=not (0 < n <= 40), role="병기(T)",
                          prior="재현(비맹검, b4)" if 0 < n <= 40 else TMARK)) for n in l4n]
    order4 = {n: i for i, n in enumerate(l4n)}
    ok4 = [(lab, n, r) for lab, n, r in resI if X.valid_row(r)]
    win = sorted([q for q in ok4 if q[2]["verdict4"] == "우세"], key=lambda q: order4[q[1]])
    lose = [q for q in ok4 if q[2]["verdict4"] == "열세"]
    w4 = [q for q in win if int(q[2].get("n_ci_regions", 0)) >= len(T.m4)]
    wp = [q for q in win if int(q[2].get("n_ci_regions", 0)) < len(T.m4)]
    s4 = [q for q in w4 if 0 < q[1] <= 40]
    sp_ = [q for q in wp if 0 < q[1] <= 40]
    pI = [(lab, r) for lab, _, r in resI]
    if not ok4:
        txt4, used4 = X.na_text(pI), pI
    elif s4 or sp_:
        q = (s4 or sp_)[0]
        txt4, used4 = "희소 라벨에서도 TabICL 잔차의 순가치가 있다", [(q[0], q[2])]
    elif win:
        txt4, used4 = "TabICL 잔차의 순가치는 n > 40 에서만 확인되었다", [(win[0][0], win[0][2])]
    else:
        txt4, used4 = "TabICL 잔차의 재보정 물리식 대비 순가치는 확인되지 않았다", [(lab, r) for lab, _, r in ok4]
    if ok4:
        txt4 += f"(우세인 최소 n: 4지역 평균 {w4[0][0] if w4 else '없음'}, 2–3지역 평균 {wp[0][0] if wp else '없음'})"
        if lose:
            txt4 += f". 열세인 n: {', '.join(q[0] for q in lose)}"
    V("LGF-F4", ITEM, txt4, f"{X._fmt(pI)} | 병기 C: {X._fmt([(lab, r) for lab, _, r in resC])} | 병기 T: {X._fmt([(lab, r) for lab, _, r in resT])}",
      role="보조", blind=True, used=used4, partial=X.part_mark(pI) if ok4 else None, clause="순가치")
    r35 = C("LGF-F4", "R1[I]-P0|all", g("R1", ALL, TI), P0, n=ALL, lam=LB, fam="aux")
    f35 = C("LGF-F4", "R1@full[I]-P0|all", g("R1@full", ALL, TI), P0, n=ALL, lam=LB, fam="aux", role="보조(@full, 판정에 쓰지 않음)")
    u35 = [("R1[I]-P0 전량", r35)]
    txt35 = X.na_text(u35) if not X.valid_row(r35) else ("라벨 전량에서 TabICL 잔차는 원천 계수 물리식을 넘는다" if X._v(r35) == "우세"
                                                          else f"라벨 전량에서 TabICL 잔차가 원천 계수 물리식을 넘는다는 판정은 나오지 않았다(4분 판정 {X._v(r35)})")
    V("LGF-F4", ITEM, txt35, X._fmt(u35 + [("보조 R1@full[I]-P0 전량", f35)]), role="보조", blind=True, used=u35, clause="전량 R1[I] − P0(L35 형식)",
      note="러시아 W·E 의 전량은 라벨 14–17개다")

    # ---------------- LGF-F5: R0[I] − P0(n = 0, 방향 중립)
    r5 = C("LGF-F5", "R0[I]-P0|n0", g("R0", 0, TI), P0, n=0, lam=LB, fam="aux")
    c5 = C("LGF-F5", "R0[C]-P0|n0", g("R0", 0, CBX), P0, n=0, lam=LB, blind=False, prior=CPRIOR, role="병기(C)")
    f5 = C("LGF-F5", "R1@full[I]-P0|n0", g("R1@full", 0, TI), P0, n=0, lam=LB, fam="aux", role="보조(@full, 판정에 쓰지 않음)")
    say5 = dict(우세="라벨 0 에서 TabICL 잔차는 원천 계수 물리식보다 오차가 작다", 열세="라벨 0 에서 TabICL 잔차는 원천 계수 물리식보다 오차가 크다",
                동등=f"라벨 0 에서 TabICL 잔차와 원천 계수 물리식은 구별되지 않는다(한계 {DE:g} cm)", 미결정="차이를 확인하지 못함")
    u5 = [("R0[I]-P0", r5)]
    V("LGF-F5", ITEM, X.na_text(u5) if not X.valid_row(r5) else say5.get(X._v(r5), X._v(r5)),
      X._fmt(u5 + [("병기 R0[C]-P0", c5), ("보조 R1@full[I]-P0", f5)]), role="보조", blind=True, used=u5)

    # ---------------- LGF-F6: R1@full[I] − R1[I], D0@full[I] − D0[I](n ∈ {0, 10, 전량}, 공통 추출 0·1)
    l6n = (0, 10, ALL)
    r6 = [(f"R1 {_nt(n)}", n, C("LGF-F6", f"R1@full[I]-R1[I]|n{_nl(n)}|lam{LB}", g("R1@full", n, TI, LB), g("R1", n, TI, LB), view="d01", n=n, lam=LB,
                                fam="aux", eq=True)) for n in l6n]
    d6 = [(f"D0 {_nt(n)}", n, C("LGF-F6", f"D0@full[I]-D0[I]|n{_nl(n)}", g("D0@full", n, TI), g("D0", n, TI), view="d01", n=n, lam=1.0, fam="aux",
                                eq=True)) for n in l6n]
    r6l = [(f"R1 {_nt(n)} λ=1.0", n, C("LGF-F6", f"R1@full[I]-R1[I]|n{_nl(n)}|lam1.0", g("R1@full", n, TI, 1.0), g("R1", n, TI, 1.0), view="d01", n=n,
                                       lam=1.0, role="보조(λ 1.0)", aux3=False)) for n in l6n]
    x6 = [(f"R1@full {_nt(n)}", C("LGF-F6", f"R1@full[I]-R1@full[C]|n{_nl(n)}|lam{LB}", g("R1@full", n, TI, LB), g("R1@full", n, CBX, LB),
                                  view="pair_IC", n=n, lam=LB, role="병기(원천 전체 조건의 학습기 대비)")) for n in l6n]
    all6 = [(lab, r) for lab, _, r in r6 + d6]
    k6, m6, _ = X.count_valid(all6)
    v6 = [X._v(r) for _, r in all6]
    vl = [X._v(r) for _, _, r in r6l]
    if k6 == 0:
        txt6, used6 = X.na_text(all6), all6
    elif all(v == "동등" for _, r in all6 if X.valid_row(r) for v in [X._v(r)]) and all(v == "동등" for v in vl):
        txt6, used6 = f"컨텍스트 상한 10,000행의 영향은 한계 {DE:g} cm 안이다(TabICL)", all6 + [(lab, r) for lab, _, r in r6l]
    else:
        parts6 = []
        up = [_nt(n) for _, n, r in r6 if X._v(r) == "우세"]
        dn = [_nt(n) for _, n, r in r6 if X._v(r) == "열세"]
        if up:
            parts6.append(f"원천 전체를 컨텍스트로 주면 TabICL 잔차의 오차가 줄어든다({', '.join(up)})")
        if dn:
            parts6.append(f"원천 전체를 컨텍스트로 주면 TabICL 잔차의 오차가 늘어난다({', '.join(dn)})")
        if not up and not dn:
            parts6.append("n 별 서술: " + ", ".join(f"{lab} {v}" for (lab, _), v in zip(all6, v6)))
        else:
            parts6.append("D0: " + ", ".join(f"{lab} {X._v(r)}" for lab, _, r in d6))
        if all(v == "동등" for v in v6) and any(v in ("우세", "열세") for v in vl):
            parts6.append(f"기준 λ {LB:g} 에서는 동등, λ 1.0 에서는 차이가 있다. 동등으로 쓰지 않는다")
        txt6, used6 = ". ".join(parts6), all6
        txt6 += prec_suffix(X, used6, DE)
    V("LGF-F6", ITEM, txt6, f"{X._fmt(all6)} | λ 1.0: {X._fmt([(lab, r) for lab, _, r in r6l])} | 병기 R1@full[I]-R1@full[C]: {X._fmt(x6)}",
      role="보조", blind=True, used=used6, partial=[f"대비 {k6}/{m6}"] if 0 < k6 < m6 else None)

    # ---------------- 계열 문장 표지(lgt_tests.csv 가 있을 때만)
    if lgt_tests is not None and len(lgt_tests):
        lt = lgt_tests
        q34 = lt[(lt.test_id.astype(str) == "L34") & (lt.contrast.astype(str) == "D0[T]-P0|n0") & (lt.scope.astype(str) == "MEAN")]
        v34 = str(q34.verdict4.iloc[0]) if len(q34) else "행 없음"
        q32 = lt[(lt.test_id.astype(str) == "L32") & lt.scope.astype(str).isin(["verdict", "verdict_aux"])]
        t32 = str(q32.verdict.iloc[0]) if len(q32) else ""
        lab0 = bool(res["F1"][1] == "A" and v34 in ("열세", "동등"))
        eqv = bool(str(res["F2"]).startswith("동등") and re_eq(t32))
        V("LGF-family", ITEM, ("표형 파운데이션 모델 2종(TabPFN v2, TabICL v2)을 학습기로 써도 직접 ML 은 라벨 0 전이에서 원천 계수 물리식을 넘지 못한다"
                               if lab0 else "라벨 0 직접 예측은 모델별로 쓴다(두 모델의 같은 형식 판정이 같지 않다)"),
          f"F1 {res['F1'][1] or '지지 아님'}; LGT L34(a) D0[T]-P0 {v34}", role="보조", blind=False, clause="라벨 0 직접 예측", family_ok=lab0)
        V("LGF-family", ITEM, ("표형 파운데이션 모델 2종과 같은 컨텍스트 CatBoost 의 R1 차이는 동등(한계 0.5 cm)" if eqv
                               else "학습기 동등성은 모델별로 쓴다(F2 와 L32 가 모두 동등이 아니다)"),
          f"F2 {res['F2'][:40]}; LGT L32 {t32[:40]}", role="보조", blind=False, clause="학습기 동등성", family_ok=eqv)
    return finish_tests(a, T.frame())


def re_eq(t):
    t = str(t)
    return t.startswith("동등") or ("): 동등" in t and t.startswith("부분("))


def finish_tests(a, df):
    """TestBook 표 → lgf_tests.csv 형식: 열 이름(ci_lo_b 등), 정밀도 열, Holm 묶음(conf, aux, eq), 확인적 표지, 플랫폼."""
    X = T43._load_h42()
    from polar import m1_stats as MS
    if not len(df):
        return pd.DataFrame(columns=TEST_COLS)
    for c_ in ("holm_family", "eq_test", "prior_info", "support_class", "key_set", "n_unpaired", "pool_regions", "splits_min", "splits_expected",
               "p_boot", "p_eq", "verdict4", "verdict4_d10", "ci_lo_beq", "ci_hi_beq", "small_note", "verdict", "n", "lam"):
        if c_ not in df:
            df[c_] = np.nan
    df["holm_family"] = df.holm_family.fillna("").astype(str)
    is_mean = df.scope.astype(str) == "MEAN"
    df.loc[~is_mean, "holm_family"] = ""
    eqt = df.eq_test.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
    df["holm_family_eq"] = np.where(is_mean & eqt & (df.holm_family == "aux"), "eq", "")
    judged = df.verdict4.notna() & ~df.verdict4.astype(str).isin(list(X.NA_VERDICTS))
    df["holm_p"] = np.nan; df["holm_p_eq"] = np.nan; df["flag_uncorr"] = ""
    for fam in ("conf", "aux"):
        m = is_mean & (df.holm_family == fam) & judged & pd.to_numeric(df.p_boot, errors="coerce").notna()
        if m.any():
            df.loc[m, "holm_p"] = MS.holm(pd.to_numeric(df.loc[m, "p_boot"]).values.astype(float))
    m = (df.holm_family_eq == "eq") & judged & pd.to_numeric(df.p_eq, errors="coerce").notna()
    if m.any():
        df.loc[m, "holm_p_eq"] = MS.holm(pd.to_numeric(df.loc[m, "p_eq"]).values.astype(float))
    fl = (df.holm_family == "aux") & df.verdict4.isin(["우세", "열세"]) & (pd.to_numeric(df.holm_p, errors="coerce") >= 0.05)
    df.loc[fl, "flag_uncorr"] = "보정 전 유의"
    df["ci_lo_b"] = df.ci_lo_beq; df["ci_hi_b"] = df.ci_hi_beq
    df["verdict4_aux"] = df.verdict4_d10
    df["eq_p"] = df.p_eq
    df["halfwidth"] = [halfwidth(r) for r in df.to_dict("records")]
    df["precision_ok"] = [bool(np.isfinite(h) and h <= float(a.delta_eq)) if np.isfinite(h) else np.nan for h in df.halfwidth.astype(float)]
    df["precision_note"] = np.where((df.verdict4.astype(str) == "미결정") & (df.precision_ok == False), "정밀도 미달(동등성 판정 불가)", "")  # noqa: E712
    df["confirmatory"] = (df.test_id.astype(str) == "LGF-F1") & df.scope.astype(str).isin(["MEAN", "verdict"]) & \
        ((df.holm_family == "conf") | (df.scope.astype(str) == "verdict"))
    df["k_default"] = ""
    df["platform"] = PLATFORM
    df["verdict_text"] = df.verdict.fillna("").astype(str)
    if getattr(a, "LOCAL_SUMMARY", False) or getattr(a, "INTERIM", False):
        mark = "[판정에 쓰지 않음] " if getattr(a, "LOCAL_SUMMARY", False) else "[중간 집계] "
        m = df.verdict_text != ""
        df.loc[m, "verdict_text"] = mark + df.loc[m, "verdict_text"]
        df["not_for_judgement"] = bool(getattr(a, "LOCAL_SUMMARY", False))
        df["interim"] = bool(getattr(a, "INTERIM", False))
    df["support_class"] = df.support_class.fillna("").astype(str)
    return df[[c_ for c_ in TEST_COLS if c_ in df] + [c_ for c_ in df.columns if c_ not in TEST_COLS]]


# ================================================================ 집계: 결측 서술 표(계획서 §2.4, 판정 없음)
def _cells_preds(path, lam_by_comp=True):
    """cells.npz → {키(method, learner, n, draw, seed): (성분 배열, 성분 종류, 앵커 계수)}, y, s. 읽을 수 없으면 ({}, None, None)."""
    try:
        z = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return {}, None, None
    with z:
        keys = [json.loads(str(k)) for k in z["keys"]]
        P, E = z["P"].astype(float), z["E"].astype(float)
        y, s = z["y"].astype(float), z["s"].astype(float)
    out = {}
    for i, k in enumerate(keys):
        out[(str(k[0]), str(k[1]), int(k[4]), int(k[5]), int(k[6]))] = (P[i], str(k[7]), float(E[i]))
    return out, y, s


def _pred_of(v, s, lam):
    comp, kind, E = v
    return comp if kind == "pred" else E * s + float(lam) * comp


def missing_desc(a, D, cells_lgf, cells_lgt, names=None):
    """F2(R1[I] − R1[C])와 F3(R1[I] − R1[T], D0[I] − D0[T] n = 0)의 Δ 를 채점 셀의 결측 유무(x25 가운데 하나라도 NaN)로 나눈 점 추정.
    cells_lgf, cells_lgt = {(대상, 모드, 분할): [cells.npz 경로]}. 결측 표지는 h40.build_ctx 로 c.XB 를 다시 만들어 정한다(cells 순서와 같다).
    지역 행의 셀 수는 사용 분할의 부분집합 셀 수 합이고 30 미만이면 점 추정을 비운다. 층화 평균 행은 비지 않은 지역만의 평균이다."""
    names = names or [f"{t}|x" for t in H.MAIN4]
    HA = T43.h40_args_t(a43_args(a, list(a.N_USER), axis_draws(a, "main"), a.seeds), "main")
    specs = [("F2", "R1[I]-R1[C]", "R1", CBX, n, float(H.LAM_BASE)) for n in (0, 10, 40, 160, -1)]
    specs += [("F3", "R1[I]-R1[T]", "R1", TP, n, float(H.LAM_BASE)) for n in (0, 10, 40, 160, -1)] + [("F3", "D0[I]-D0[T]", "D0", TP, 0, 1.0)]
    acc: dict = {}
    for (t, m, sp), paths in sorted(cells_lgf.items()):
        nm = f"{t}|{m}"
        if nm not in names:
            continue
        c = H.build_ctx(D, HA, t, m, sp)
        miss = np.isnan(np.asarray(c.XB, float)).any(1)
        own = {}
        y = s = None
        for pth in paths:
            d_, y_, s_ = _cells_preds(pth)
            if y_ is not None and len(y_) == len(miss):
                own.update(d_); y, s = y_, s_
        oth = {}
        for pth in cells_lgt.get((t, m, sp), []):
            d_, y_, _ = _cells_preds(pth)
            if y_ is not None and len(y_) == len(miss):
                oth.update(d_)
        if y is None:
            continue
        for hyp, lab, mth, lrB, n, lam in specs:
            src = own if lrB == CBX else oth
            pairs = [(k, (mth, lrB) + k[2:]) for k in own if k[0] == mth and k[1] == TI and k[2] == n and (mth, lrB) + k[2:] in src]
            if not pairs:
                continue
            for sub, mask in (("결측 있음", miss), ("결측 없음", ~miss), ("전체", np.ones(len(miss), bool))):
                if not mask.any():
                    continue
                dA = [np.sqrt(np.mean((_pred_of(own[kA], s, lam)[mask] - y[mask]) ** 2)) for kA, _ in pairs]
                dB = [np.sqrt(np.mean((_pred_of(src[kB], s, lam)[mask] - y[mask]) ** 2)) for _, kB in pairs]
                q = acc.setdefault((hyp, lab, n, lam, nm, sub), dict(d=[], cells=0))
                q["d"].append(float(np.mean(dA) - np.mean(dB))); q["cells"] += int(mask.sum())
    rows = []
    for (hyp, lab, n, lam, nm, sub), q in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][4], kv[0][5])):
        ok = q["cells"] >= MISS_MIN_CELLS
        rows.append(dict(hypothesis=hyp, contrast=lab, n=int(n), lam=lam, target=nm, subset=sub, n_cells=int(q["cells"]), n_splits=len(q["d"]),
                         delta=float(np.mean(q["d"])) if ok else np.nan, note="" if ok else f"셀 수 {MISS_MIN_CELLS} 미만(비움)"))
    df = pd.DataFrame(rows)
    if len(df):
        mean = []
        for (hyp, lab, n, lam, sub), g_ in df.groupby(["hypothesis", "contrast", "n", "lam", "subset"]):
            ok = g_[g_.delta.notna()]
            mean.append(dict(hypothesis=hyp, contrast=lab, n=int(n), lam=lam, target=f"MEAN[{','.join(ok.target)}]", subset=sub,
                             n_cells=int(ok.n_cells.sum()), n_splits=int(ok.n_splits.sum()), delta=float(ok.delta.mean()) if len(ok) else np.nan,
                             note=f"비지 않은 지역 {len(ok)}/{len(names)}. 판정과 CI 없음"))
        df = pd.concat([df, pd.DataFrame(mean)], ignore_index=True)
    return df


# ================================================================ 집계
def find_shards_f(a, X):
    """lgf, lgfs 조각(부분 n0, pos, all). h42.find_shards_x 의 learner 칸이 부분이다."""
    out = []
    for axis in AXES_F:
        for s_ in X.find_shards_x(a.SHARDS, axis_tag(a, axis)):
            if s_["part"] == "gpu" and s_["learner"] in PARTS:
                out.append(dict(s_, axis=axis, part_name=s_["learner"]))
    return out


def read_runs_f(shards):
    frames = []
    for s_ in shards:
        if Path(s_["runs"]).stat().st_size > 1:
            f = pd.read_csv(s_["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str, ctx_set=str, ctx_sha=str, offload_mode=str),
                            keep_default_na=False, na_values=["", "nan", "NaN"])
            if len(f):
                f["tag"] = s_["tag"]
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    for c_ in ("alpha_sel", "fit_flag", "ctx_set", "ctx_sha"):
        runs[c_] = runs[c_].fillna("").astype(str) if c_ in runs else ""
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int) if "n_nonfinite" in runs else 0
    return runs.drop_duplicates(subset=["target", "mode", "split"] + list(H.KEY_COLS), keep="first")


def check_cfg_f(a, shards, units):
    """(tag, 부분, 작업)마다 설정 해시가 하나여야 한다. 다르면 중단한다(--allow-mixed-cfg 로 진행)."""
    info = {}
    for key in sorted({(s_["tag"], s_["part_name"], str(u.get("job", ""))) for s_, u in zip(shards, units)}):
        us = [u for s_, u in zip(shards, units) if (s_["tag"], s_["part_name"], str(u.get("job", ""))) == key]
        hs = Counter(str(u.get("cfg_hash", "legacy")) for u in us)
        info["|".join(key)] = dict(cfg_hash=dict(hs), n=len(us), code_sha_h47=dict(Counter(str((u.get("code_sha") or {}).get("h47", "")) for u in us)))
        if len(hs) > 1:
            msg = f"{key}: 조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}"
            if not a.allow_mixed_cfg:
                raise SystemExit("[summarize] " + msg + " (--allow-mixed-cfg 로 진행할 수 있다)")
            print("  [warn] " + msg, flush=True)
    return info


def default_scope_status(a, w=None):
    """기본 범위(J1, J6, J8, J9, J11. 창 파일의 유효 축소 적용)의 완료 상태. 반환 (완료 단위 수, 전체 단위 수)."""
    w = read_window(a) if w is None else w
    red = [s_ for s_ in effective_reduce(w) if s_ in REDUCE_F]
    a2 = parse_args(["--jobs", DEFAULT_JOBS, "--reduce", ",".join(red), "--threads", str(int(a.threads)), "--out-dir", str(a.OUT),
                     "--data-dir", str(a.PROC), "--count-only"])
    units, _ = enumerate_f(a2)
    return int(sum(unit_state_f(a2, u)[0] for u in units)), len(units)


def summarize_guard(a):
    """본 실행 중(창 마감 전이고 기본 범위 미완료)에는 집계를 거부한다. --force-interim 이면 '중간 집계' 표지를 붙여 돈다."""
    w = read_window(a)
    if not w.get("t0") or window_deadline_passed(w):
        return
    k, n = default_scope_status(a, w)
    if k >= n:
        return
    if not a.force_interim:
        raise SystemExit(f"[거부] 본 실행 중이다(창 마감 {w.get('deadline')} 전, 기본 범위 완료 {k}/{n}). 중간 집계는 --force-interim 으로 하고 "
                         "개정 이력에 기록한다")
    a.INTERIM = True
    print(f"[summarize] 중간 집계(기본 범위 완료 {k}/{n}). 판정 행에 '중간 집계' 표지를 붙인다. 개정 이력에 기록한다", flush=True)


def summarize(a, elapsed=0.0, skipped=None):
    """집계. 스모크·사전 점검은 각자의 산출만 쓴다. 반환 dict 의 n_fail 은 저장하지 못한 키, 상태가 ok 가 아닌 조각, 곡선 실패의 합이다."""
    if a.smoke:
        return smoke_summary(a)
    if a.precheck:
        return precheck_summary(a)
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    X = T43._load_h42()
    sh = find_shards_f(a, X)
    if not sh:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{axis_tag(a, 'main')}__gpu__*, {axis_tag(a, 'full')}__gpu__*", flush=True)
        return None
    units = [json.loads(Path(s_["unit"]).read_text()) for s_ in sh]
    cfg_info = check_cfg_f(a, sh, units)
    runs = read_runs_f(sh)
    stores = load_stores([s_["npz"] for s_ in sh])
    gate, stores_g = gate_lgt(a, X, stores, runs, units)
    n_pass = int(gate.passed.sum()) if len(gate) else 0
    print(f"[summarize] LGT 짝 게이트: 통과 {n_pass}/{len(gate)} 단위", flush=True)
    HA = T43.h40_args_t(a43_args(a, list(a.N_USER), axis_draws(a, "main"), a.seeds), "main")
    D = H.get_data(HA)
    floor, floor_meta = X.floor_table(D.df)
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in stores_g.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in stores_g})}
    curve_failed: list = []
    cur = T43.curve_t(a, X, D, tms, runs, floor, curve_failed)
    mn = H.build_minn(cur) if len(cur) else pd.DataFrame()
    lt_path = Path(a.LGTDIR) / f"{a.lgt_tag}_tests.csv"
    lgt_tests = pd.read_csv(lt_path) if lt_path.exists() else None
    tests = build_tests_f(a, tms, D, floor, X, lgt_tests)
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()
    cl_own, cl_oth = {}, {}
    for s_ in sh:
        if s_["axis"] == "main" and Path(s_["cells"]).exists():
            cl_own.setdefault((s_["target"], s_["mode"], int(s_["split"])), []).append(s_["cells"])
    passed = {(r_.target, r_.mode, int(r_.split)) for r_ in gate.itertuples() if r_.passed} if len(gate) else set()
    for s_ in X.find_shards_x(Path(a.LGTDIR) / "shards", a.lgt_tag):
        key = (s_["target"], s_["mode"], int(s_["split"]))
        if s_["part"] == "gpu" and key in passed and Path(s_["cells"]).exists():
            cl_oth.setdefault(key, []).append(s_["cells"])
    miss = missing_desc(a, D, cl_own, cl_oth)
    failed = T43.failed_rows(runs)
    tg = pd.DataFrame([dict({k: v for k, v in u.items() if not isinstance(v, (dict, list))}, n_fit=json.dumps(u.get("n_fit", {})),
                            sec=json.dumps(u.get("sec", {})), fail=json.dumps(u.get("fail", {})), errors=json.dumps(u.get("errors", []), ensure_ascii=False),
                            ctx=json.dumps(u.get("ctx", {}), ensure_ascii=False), failed_learners=",".join(u.get("failed_learners", []) or []))
                       for u in units])
    if skipped:
        tg = pd.concat([tg, pd.DataFrame(skipped)], ignore_index=True)
    tt = H.timing_table(units)
    O = a.OUT; O.mkdir(parents=True, exist_ok=True)
    for nm_, df_ in (("curve", cur), ("minn", mn), ("tests", tests), ("gate", gate), ("missing_desc", miss), ("timing", tt)):
        T43._atomic_csv(df_, O / f"{a.TAG}_{nm_}.csv")
    fa = failed.copy() if len(failed) else pd.DataFrame(columns=list(runs.columns) if len(runs) else RUN_COLS_F)
    if curve_failed:
        fa = pd.concat([fa, pd.DataFrame([dict(target=q["target"], mode=q["mode"], fit_flag="curve_failed", method=q["reason"]) for q in curve_failed])],
                       ignore_index=True)
    T43._atomic_csv(fa, O / f"{a.TAG}_failed.csv")
    T43._atomic_csv(tg.sort_values(["target", "mode", "split"]) if len(tg) else tg, O / f"{a.TAG}_targets.csv")
    status = Counter(str(u.get("status", "ok")) for u in units)
    n_fail = int(len(failed)) + int(sum(v for k, v in status.items() if k != "ok")) + int(len(curve_failed))
    fit_tot = Counter()
    for u in units:
        for k, v in (u.get("n_fit") or {}).items():
            fit_tot[k] += int(v)
    meta = dict(stage="H47/LGF-F", plan="docs/EXPERIMENT_PLAN_LGF_2026-09-29.md §3(개정 1)", git_commit=T43.git_commit(), tag=a.TAG,
                tags={ax: axis_tag(a, ax) for ax in AXES_F}, args={k: v for k, v in vars(a).items() if k.islower() and not k.startswith("_")},
                key_fields=list(H.KEY_COLS), p0_key=list(H.P0_KEY), all_n=-1, learners=list(LEARNERS_F) + [TP + "(LGT 조각, 읽기 전용)"],
                n_units=len(units), n_fit=dict(fit_tot), n_rows=int(len(runs)), n_curve=int(len(cur)), n_minn=int(len(mn)), n_tests=int(len(tests)),
                n_gate=int(len(gate)), n_gate_passed=n_pass, n_missing_desc=int(len(miss)), shard_cfg=cfg_info, unit_status=dict(status),
                n_failed_keys=int(len(failed)), curve_failed=curve_failed, n_fail=n_fail, skipped=skipped or [], floor=floor_meta,
                pins=a.PINS, code_sha=T43.file_sha(__file__, 12), delta_eq=float(a.delta_eq), delta_eq_aux=float(a.delta_eq_aux), nboot=int(a.nboot),
                platform=PLATFORM, interim=bool(a.INTERIM), not_for_judgement=bool(a.LOCAL_SUMMARY), lgt_tests_used=bool(lgt_tests is not None),
                window=read_window(a), elapsed_s=round(float(elapsed), 1), summarize_s=round(time.time() - t0, 1),
                rules=dict(holm="conf = F1 의 2개, aux = F2 기준 λ 5 + F3 6 + F4 6 + 전량 R1[I]−P0 1 + F5 1 + F6 6 + @full 보조 2(27개). "
                                "eq = F2·F3·F6 의 동등성 p(17개). 층화 평균 행만 묶음에 넣는다",
                           precision="halfwidth = max(셀 가중 CI 폭, 블록 등가중 CI 폭)/2. 반폭 > 0.5 cm 이고 미결정이면 '정밀도 미달(동등성 판정 불가)'",
                           l1_form="지역 행의 4분 판정이 우세인 지역을 n ∈ {0, 3, 10, 40} 에 걸쳐 센다",
                           gate="LGT 조각 상태, 설정(ctx·dup·κ·seed·추출·n 격자·대응표), P0·P1 1e-9 cm, 키별 ctx_sha 전부 일치, 채점 블록 일치",
                           missing_desc="셀 수 = 사용 분할의 부분집합 셀 수 합, 30 미만이면 비움"))
    H._atomic_text(O / f"{a.TAG}_meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[summarize] 조각 {len(sh)} · runs {len(runs):,} · curve {len(cur):,} · minn {len(mn):,} · tests {len(tests):,} · gate {len(gate)} · "
          f"missing_desc {len(miss)} · {time.time() - t0:.0f}s → {O}/{a.TAG}_*", flush=True)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])] if len(tests) else tests
    if len(v):
        print(v[[c_ for c_ in ("test_id", "clause", "support_class", "verdict_text") if c_ in v]].to_string(index=False), flush=True)
    return dict(curve=cur, minn=mn, tests=tests, gate=gate, missing=miss, meta=meta, n_fail=n_fail)


# ================================================================ 실행
def f_t3_allowed(a, gpus, w=None, now=None):
    """f_t3(J12) 조건: F 기본 범위가 모두 끝났고, N 작업의 완료를 호출자가 확인했고(--n-jobs-done), (창 마감 − 지금) × 쓰는 GPU 수가
    lgf_projection.csv 의 J12 추정보다 크다. 반환 (허용 여부, 사유)."""
    w = read_window(a) if w is None else w
    dl = _to_ts(w.get("deadline"))
    if dl is None:
        return False, "창이 시작되지 않았다(t0 없음)"
    k, n = default_scope_status(a, w)
    if k < n:
        return False, f"F 기본 범위 미완료({k}/{n})"
    if not a.n_jobs_done:
        return False, "N 작업(h48)의 완료 확인이 없다(--n-jobs-done)"
    est = projection_value(a, "J12")
    if est is None:
        return False, "lgf_projection.csv 에 J12 추정이 없다"
    use = sorted(set(int(g) for g in gpus) | set(live_gpus(a)))             # 이 실행의 GPU 와 h48 등 살아 있는 LGF 워커의 GPU
    cap = max(dl - (time.time() if now is None else float(now)), 0.0) / 3600.0 * max(len(use), 1)
    if cap <= est:
        return False, f"여유 {cap:.1f} GPU-h(GPU {use}) ≤ J12 추정 {est:.1f} GPU-h"
    return True, f"여유 {cap:.1f} GPU-h(GPU {use}) > J12 추정 {est:.1f} GPU-h"


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in THREAD_VARS:
        os.environ[v] = str(int(a.threads))
    ensure_nice()
    if a.window_set or a.window_status or a.gpu_event:
        window_tool(a)
        return dict(n_fail=0)
    check_pins(a)
    check_env()
    will_run = not (a.count_only or a.summarize_only)
    if will_run:
        check_run_args(a)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.summarize_only:
        summarize_guard(a)
        res = summarize(a, 0.0, None)
        return dict(n_fail=int(res["n_fail"]) if res else 0)
    units, skipped = enumerate_f(a)
    print(f"[data] 작업 {a.JOBS if not (a.smoke or a.precheck) else ('smoke' if a.smoke else 'precheck')} · 축소 {a.REDUCE or '없음'} · 분할 1–{int(a.splits)} · "
          f"n {a.N_USER} · seed {int(a.seeds)} · 작업 단위 {len(units)}(건너뛴 분할 {len(skipped)}) · 스레드 {a.threads} · nice {_nice_now()}", flush=True)
    if a.count_only:
        count_only(a, units, skipped)
        return dict(n_fail=0)
    w = check_window_for_run(a) if a.MAIN_MODE else {}
    if a.MAIN_MODE and window_deadline_passed(w):
        print(f"[window] 창 마감({w.get('deadline')})이 지났다. 새 단위를 시작하지 않는다", flush=True)
        return dict(n_fail=0, stop="window")
    gpus, dropped = screen_now(a, a.GPUS)
    for g_, why in dropped:
        print(f"[warn] GPU {g_} 를 뺀다: {why}", flush=True)
    if not gpus:
        raise SystemExit("[거부] 쓸 수 있는 GPU 가 없다(점유 판정)")
    if a.smoke or a.precheck:
        gpus = gpus[:1]
    if any(u[0] == "f_t3" for u in units):
        ok3, why3 = f_t3_allowed(a, gpus, w)
        if not ok3:
            units = [u for u in units if u[0] != "f_t3"]
            print(f"[plan] f_t3(J12)를 돌리지 않는다: {why3}", flush=True)
            update_window(a, lambda w_: dict(w_, decisions=list(w_.get("decisions", [])) + [dict(time=_now_iso(), what=f"J12 미실행: {why3}",
                                                                                                    viewing_state=str(a.viewing_state or ""))]))
            if not units:
                return dict(n_fail=0)
        else:
            print(f"[plan] f_t3(J12)를 돈다: {why3}", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    if not T43.cuda_available(gpus[0]):
        raise SystemExit("[거부] torch.cuda 를 쓸 수 없다")
    overrides = {}
    if not a.smoke and (a.tabicl_stock or smoke_forced_stock(a)):
        overrides["tabicl_stock"] = True
        if not a.tabicl_stock:
            print("[plan] 스모크 점검에서 캐시 하위 클래스의 차가 허용을 넘었다. 원래 TabICLRegressor 를 쓴다", flush=True)
    check_overwrite_f(a, units)
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    a.RUN_ID = f"{os.getpid()}-{time.strftime('%Y%m%dT%H%M%S')}"
    lock = T43.acquire_lock(a)
    T43.clear_run_notes(a.RUN_DIR, ("running", "gpu_busy", "worker"))
    done, failed, quarantined, resumed, todo = [], [], [], [], []
    done_names, failed_names = set(), set()
    interrupted, abort, handlers, busy_gpus, gpu_all, stop = "", "", {}, set(), list(gpus), ""

    def log(u):
        done.append(u); done_names.add(f"{u['axis']}|{u['target']}|{u['mode']}|s{u['split']}|{u['part']}")
        print(f"  [{u['job']}|{u['axis']}|{u['target']}|{u['mode']}|s{u['split']}|{u['part']}] 적합 {u['n_fit_total']} · 행 {u['n_rows']} · 컨텍스트 최대 "
              f"{u['n_ctx_max']} · A {u['n_A']} · 채점 {u['n_eval']} · 원천 {u['n_src']} · {u['elapsed_s']}s · GPU '{u['device']}' {u['gpu_mem_peak_mib']}"
              f"/{u['gpu_mem_reserved_peak_mib']} MiB · RSS {u['rss_max_mib']} MiB · 상태 {u['status']}(실패 {u['n_fail']}) · 완료 {len(done)}/{len(todo)} · "
              f"누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append((u, repr(e)[:300])); failed_names.add(unit_name_f(u))
        print(f"  [FAIL] {unit_name_f(u)}: {repr(e)[:300]}", flush=True)

    try:
        for u in units:
            ok, why = unit_state_f(a, u) if a.resume else (False, "")
            if ok:
                resumed.append(u)
            else:
                todo.append(u)
                if a.resume and why != "조각 없음":
                    print(f"  [resume] 다시 실행 {unit_name_f(u)} ({why})", flush=True)
        print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · GPU {gpus} · 스레드 {a.threads} · 가중치 SHA-1 {T43.file_sha(a.MODEL, 12)} · "
              f"tabicl {T43.pkg_version('tabicl')} · run_id {a.RUN_ID} · 기록 {a.RUN_DIR}", flush=True)
        if todo:
            handlers = T43.install_stop_handlers()
            ctx = multiprocessing.get_context("spawn")
            order = {u: i for i, u in enumerate(todo)}
            pending = list(todo)
            exs, running = {}, {}                                         # GPU 하나에 풀 하나(워커 하나). 한 GPU 의 워커가 끝나도 나머지 GPU 는 계속 돈다
            crashes, stall = Counter(), Counter()
            active = list(gpus)
            gate_state: dict = {}

            def make(g):
                return ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_worker_init_f,
                                           initargs=(argv, int(g), a.threads, str(a.RUN_DIR), os.getpid(), a.RUN_ID, overrides))

            def drop(g, why):
                if g in active:
                    active.remove(g)
                ex_ = exs.pop(g, None)
                if ex_ is not None:
                    T43.kill_pool(ex_)
                print(f"[pool] GPU {g} 를 뺀다: {why}", flush=True)

            def requeue(u):
                pending.append(u)
                pending.sort(key=lambda q_: order.get(q_, 1 << 30))

            def broken(g, u, why):
                """GPU g 의 풀(워커 하나)이 깨졌다. 그 GPU 의 단위만 분류한다: 이번 실행에서 조각을 이미 썼으면 완료, 워커가 점유 표지를 남겼으면
                (GPU 점유) 단위를 되돌리고 그 GPU 를 뺀다, 그 밖은 단위의 풀 파손 횟수를 세어 h43.POOL_CRASH_MAX 에 이르면 격리한다.
                그 GPU 는 재확인(rescreen)을 통과하면 풀을 다시 만들고, 진전 없이 --pool-retries 를 넘게 깨지면 뺀다."""
                ex_ = exs.pop(g, None)
                if ex_ is not None:
                    T43.kill_pool(ex_)
                bnotes = [m_ for m_ in T43.read_run_notes(a.RUN_DIR, "gpu_busy") if str(m_.get("gpu", "")) == str(g)]
                (Path(a.RUN_DIR) / f"gpu_busy__{int(g)}.json").unlink(missing_ok=True)
                progressed = False
                if u is not None:
                    uj = unit_done_run_f(a, u)
                    if uj is not None:
                        log(unit_summary_f(uj))
                        progressed = True
                    elif bnotes:
                        requeue(u)
                    else:
                        nm = unit_name_f(u)
                        crashes[nm] += 1
                        if crashes[nm] >= T43.POOL_CRASH_MAX:
                            quarantined.append(nm)
                            fail(u, RuntimeError(f"실행 중에 풀이 {crashes[nm]}회 깨졌다(격리, 상한 {T43.POOL_CRASH_MAX})"))
                            progressed = True
                        else:
                            requeue(u)
                if bnotes:
                    busy_gpus.add(int(g))
                    drop(g, f"워커가 점유 표지를 남겼다({str(bnotes[-1].get('why', ''))[:120]})")
                    return
                stall[g] = 0 if progressed else stall[g] + 1
                ok = rescreen(a, [g], busy_gpus) if (g in active and stall[g] <= int(a.pool_retries)) else []
                if ok:
                    exs[g] = make(g)
                    print(f"[pool] GPU {g}: {why}. 풀을 다시 만든다(진전 없는 연속 {stall[g]}회)", flush=True)
                else:
                    drop(g, f"{why}. 재확인 실패 또는 진전 없는 재생성 {stall[g]}회(상한 {a.pool_retries})")

            clean = False
            try:
                for g in active:
                    exs[g] = make(g)
                while True:
                    for g in list(active):
                        if stop or abort or not pending:
                            break
                        if g not in exs or any(gg == g for gg, _ in running.values()):
                            continue
                        gs = assign_gate(a, gate_state)
                        if gs in ("drain", "window"):
                            stop = gs
                            print(f"[plan] {'드레인 요청' if gs == 'drain' else '창 마감'}: 새 단위를 배정하지 않는다(진행 중 {len(running)})", flush=True)
                            break
                        if gs == "load":
                            break
                        u = pending.pop(0)
                        try:
                            fut = exs[g].submit(_worker_run_f, u)
                        except BrokenProcessPool:                         # 대기 중인 워커가 죽었다: 단위를 되돌리고 그 GPU 의 풀만 다룬다
                            requeue(u)
                            broken(g, None, "대기 중 워커가 비정상 종료했다(제출 실패)")
                            continue
                        running[fut] = (g, u)
                    if not running:
                        if stop or abort or not pending:
                            break
                        if not active:
                            for u in pending:
                                fail(u, RuntimeError("쓸 수 있는 GPU 가 없다(점유 또는 풀 재생성 때 재확인 실패)"))
                            pending.clear()
                            break
                        if gate_state.get("high"):
                            print(f"[plan] load average {gate_state.get('load', 0):.1f} > {a.load_max_run:g}: 배정을 보류한다", flush=True)
                            time.sleep(30.0)
                        continue
                    fin, _ = wait(list(running), timeout=30.0, return_when=FIRST_COMPLETED)
                    for f in fin:
                        g, u = running.pop(f)
                        try:
                            log(f.result())
                            stall[g] = 0
                        except BrokenProcessPool:
                            broken(g, u, f"워커가 비정상 종료했다({unit_name_f(u)})")
                        except Exception as e:                            # noqa: BLE001
                            fail(u, e)
                            if T43.TAG_MISMATCH in str(e):
                                abort = str(e)
                    if abort:
                        for _, u in list(running.values()):
                            fail(u, RuntimeError(f"GPU 장치 대응 불일치로 실행을 멈췄다: {abort[:160]}"))
                        for u in pending:
                            fail(u, RuntimeError(f"GPU 장치 대응 불일치로 실행을 멈췄다: {abort[:160]}"))
                        running.clear(); pending.clear()
                        break
                clean = not abort
            finally:
                for ex_ in list(exs.values()):
                    if clean:
                        ex_.shutdown(wait=True)
                    else:
                        T43.kill_pool(ex_)
                exs.clear()
            gpus = list(active)
            if stop == "drain":
                clear_drain(a)
                print("[plan] 드레인 완료: drain 표지를 지웠다. --resume 으로 다시 시작한다", flush=True)
    except KeyboardInterrupt as e:
        interrupted = str(e) or "KeyboardInterrupt"
        print(f"[중단] {interrupted}. 워커를 종료했다. 이어 돌리려면 같은 명령에 --resume 을 준다", flush=True)
    finally:
        T43.restore_handlers(handlers)
        n_fail = len(failed) + sum(1 for u in done if u["status"] == "failed")
        part = [f"{u['axis']}|{u['target']}|{u['mode']}|s{u['split']}|{u['part']}" for u in done if u["status"] == "partial"]
        stat = dict(run_id=a.RUN_ID, tag=a.TAG, argv=argv, start_s=round(t0, 1), elapsed_s=round(time.time() - t0, 1), gpus_first=gpu_all, gpus_last=gpus,
                    busy_gpus=sorted(busy_gpus), n_todo=len(todo), n_done=len(done), n_resumed=len(resumed), n_fail=n_fail, n_partial=len(part),
                    partial_units=part, failed_units=[dict(unit=unit_name_f(u), reason=r_) for u, r_ in failed], quarantined=quarantined,
                    interrupted=interrupted, abort=abort, stop=stop, jobs=a.JOBS, reduce=a.REDUCE, overrides=overrides, window=read_window(a),
                    next_step="같은 명령에 --resume(필요하면 --rerun-partial)")
        T43._write_json(a.OUT / f"{a.TAG}_run_status.json", stat)
        T43.release_lock(lock)
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 적합 실패 {len(part)} · 격리 {len(quarantined)} · 멈춤 {stop or '없음'} · {time.time() - t0:.0f}s",
          flush=True)
    res = dict(n_fail=n_fail, n_partial=len(part), interrupted=interrupted, stop=stop)
    if interrupted:
        return res
    if not a.no_summarize and (a.smoke or a.precheck or not stop):
        if a.MAIN_MODE:                                                   # 본 실행 중(창 마감 전, 기본 범위 미완료)에는 판정용 집계를 하지 않는다
            try:
                summarize_guard(a)
            except SystemExit as e:
                print(f"[summarize] 집계를 건너뛴다: {e}", flush=True)
                return res
        sres = summarize(a, time.time() - t0, skipped)
        if sres:
            res["n_fail"] += int(sres.get("n_fail", 0) or 0)
    return res


if __name__ == "__main__":
    _res = main()
    sys.exit(exit_code_f(_res))
