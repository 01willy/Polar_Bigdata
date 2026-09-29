"""LGU(예측 분포 실험) 공용 모듈. 계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 2절(공통 규약)의 구현이다.

목적
  실험 A(h44, 계층 예측 분포 사다리), 실험 B(h45, 정규화기 비교형 계층 conformal), 실험 C(h46, 공간 정보량 진단)가 함께 쓰는
  상수, 실행 보호, 분위 격자, 생성 학습기 래퍼, 분포 점수, 채점 블록 합 저장소(ScoreStore), 계층 보정, 의사 대상 모사,
  전역 블록 재표집, 4분 판정, 물리 계수 표본을 한곳에 둔다.

실행 환경(사용자 지시 2026-09-30)
  LGU 는 Rescale 이 아니라 로컬 GPU 서버에서 실행한다. 실행 보호(LG_RESCALE=1 또는 --allow-local 이 없으면 거부)는 유지하고
  로컬 실행은 --allow-local 로 한다. 공유 서버 규칙: 이 실험의 CPU 스레드 합계는 8 이하(OMP·MKL·OPENBLAS 환경 변수,
  torch.set_num_threads, CatBoost thread_count 모두), nice -n 10, GPU 는 배정된 번호(2번, 필요하면 1번)만 쓰고 쓰기 직전에
  메모리 사용이 0 MiB 인지 확인한다. 이 모듈은 도우미(set_thread_env, set_torch_threads, gpu_memory_used, free_gpus, claim_threads)를
  제공하고 규칙의 적용은 h44–h46 이 맡는다. 환경 변수 LG_RESCALE=1 은 허용 표지로만 인정하며 로컬 자원 규칙을 끄지 않는다(개정 1).
  하네스 사이의 스레드 합계는 실행 중 등록부(claim_threads, data/processed/lgu/.running)로 확인한다.

동결 규칙
  기존 모듈(tab_models, h4_common, m1_core, m1_stats, physics)은 고치지 않고 import 한다. scripts/ 의 파일은 import 하지 않는다.
  h37 의 wquantile 과 h40 의 _prep_stats, _prep_apply, ls_E 는 같은 식을 이 모듈에 다시 둔다(h37 은 모듈 최상위에서 인자를 읽으므로
  import 할 수 없다). 생성 학습기는 tab_models 를 고치지 않고 내부 구성 요소(_cond_net, _nflow_mods, _rq_spline, _dev)를 import 해
  학습 반복문을 다시 쓴다. torch, polar.tab_models, catboost, scipy 는 쓰는 함수 안에서 import 한다(CPU 경로에서 torch 를 읽지 않는다).
  모듈 최상위에서 자료 적재, 난수 생성, 스레드 설정을 하지 않는다.

공개 함수(인자와 반환은 각 docstring 에 있다)
  실행 보호: peek_threads, set_thread_env, set_torch_threads, gpu_memory_used, free_gpus, run_permitted, require_permission,
            atomic_text, atomic_npz, file_sha, cfg_hash, unit_state, load_flags, nearest_km, is_env_error, cal_groups,
            check_blockindex, claim_threads, release_threads, running_threads
  전처리: prep_stats, prep_apply, row_weights, block_val_mask
  분위 격자: q_from_samples, crossing_stats, rearrange, grid_index, interval_of, median_of, grid_from_center, loc_only, to_cm
  생성 학습기: GenHandle, make_flow_class, fit_gen, gen_samples, nflow_quantiles, nflow_logpdf, nflow_cdf_table, gen_quantiles,
              cfm_quantiles_ode, fit_failed, cb_multiquantile
  점수: pinball, crps_grid, interval_parts, pit_grid, pit_hist, score_cells, score_interval_only, alpha_suffix
  저장소: ScoreStore, save_score_stores, load_score_stores, stores_for_target
  보정: wquantile, hier_cell_weights, hier_quantiles, pooled_quantile, hier_quantiles_boot, ls_E, center_coef, emulation_plan
  재표집과 판정: BlockIndex, global_mult, agg_cell, agg_beq, agg_rmse_cell, agg_rmse_beq, ci95, combine_same_index, strat_mean,
                one_stage_delta, verdict4, eq_state, rel_margin, half_width
  물리 표본: phys_log_samples

명세(spec_common)와 다르게 구현한 점
  1. 저장소의 블록 합은 y 가 유한한 셀 전부의 합이다. 명세는 '지표마다 유한 셀의 블록 합'이었다. 무한 구간(inf)이나 계산 실패(NaN)를
     빼고 합하면 셀 수(cnt, y 유한 셀 수)와 분자가 어긋나 평균이 낮게 나온다. 합에는 inf 와 NaN 이 그대로 전파되고, 집계 함수(agg_*)는
     재표집에서 그 블록이 뽑히면 inf 또는 NaN 을, 뽑히지 않으면 나머지 블록의 값을 낸다(0·inf 를 NaN 으로 만들지 않는다).
     정의되지 않는 지표(구간만 있는 방법의 CRPS 등)는 NaN 이 된다.
  2. score_cells 와 score_interval_only 는 지표 외에 'pit' 와 'valid'(y 유한 표지)를 돌려준다. ScoreStore.add 는 'valid' 로 셀 수를 센다.
  3. peek_threads 는 이 모듈이 numpy 를 import 하므로 스크립트의 numpy import 전에 부를 수 없다. 스크립트는 같은 규칙을 머리에 다시 두고
     (h42 와 같은 방식) 이 함수는 규칙의 기준과 시험, 워커 초기화에 쓴다.
  4. loc_only 와 fit_failed 는 중앙값 위치 mid 를 인자로 받는다. 기본은 행 수 99 이면 49, 행 수 5(Q5)이면 2 다(실험 B 의 보정 분위가 Q5 형식).
  5. cb_multiquantile 의 반환은 (예측 목록, info)다. info 는 crossing_stats 의 키(frac_cells, frac_pairs, max_drop, 개수)에 flag(손실 방식)와
     per_set(입력 행렬별 교차 통계)을 더한 dict 다.
  6. fit_gen 의 val_loss 는 검증 손실이 한 번도 개선되지 않았으면(최선 상태가 없으면) NaN 이다. tab_models 는 1e9 를 돌려준다.
     가중 학습의 검증 손실은 검증 행 전체의 가중 평균이다. 가중이 없으면 tab_models 와 같이 배치 평균의 평균이다.
     입력에 비유한 값이 있으면 학습 전에 ValueError 를 낸다(tab_models 는 확인하지 않는다. 유한 입력에서는 결과가 같다).
  7. emulation_plan 은 지역 k 의 원천 셀을 y, s 유한이고 s > 0 인 셀로 먼저 제한한 뒤 분할한다. 기록에 E_center, n_A, nb_cal 을 더했다.
  8. combine_same_index 는 명세대로 nanmean 이다. strat_mean 은 한 지역이라도 NaN 인 재표집 번호를 NaN 으로 둔다(풀 구성이 번호마다 바뀌지
     않게 한다). one_stage_delta 는 (개정 1) 분할마다 A·B 키를 (n, 추출) 단위로 묶어 두 쪽에 모두 있는 단위만 쓰고, 단위 안의 키 평균
     (nanmean)으로 단위별 차를 구한 뒤 단위 축 nanmean 한다. 모든 단위의 키 수가 같고 NaN 이 없으면 LG 규칙(키 평균의 차)과 같은 값이다.
     한쪽에만 있는 단위 수와 NaN 을 가진 키 수를 반환하고(n_unpaired_*, n_nan_keys_*, pair_ok), 확인적 대비는 pair_ok 가 거짓이면
     '판정 불가(계산 실패)'로 둔다(호출자 규칙). 분할 결합은 np.mean 이다. 반환에 n_keys_A, n_keys_B, delta_split_beq 를 더했다.
  9. eq_state 와 verdict4 는 가중별 한계(rel_margin 의 2원소)를 받을 수 있다. 정밀도 규칙은 가중마다 자기 한계와 반폭을 비교한다.
     스칼라 한계이면 명세의 max(반폭) > 한계 와 같다.
  10. hier_quantiles, hier_quantiles_boot, pooled_quantile 은 NaN 점수를 뺀다(+inf 점수는 남긴다).
  11. block_val_mask 는 groups 의 원래 자료형으로 np.unique 를 쓴다(블록 번호가 정수이면 정수 순서).
  12. hier_cell_weights 는 가중만 돌려준다(h37.group_weights 는 (가중, 그룹 수)).
  13. load_flags 는 missing_ok 인자를 더했다(참이면 표가 없을 때 표지 0). 기본은 명세대로 중단이다.
  14. ScoreStore 의 block_cols 는 셀 단위 배열(block_ids 와 같은 길이)이다. 저장소 안에서 블록 단위(u{i}_cols)로 바꾼다.
  15. METRICS 의 is10_w 와 wid10 은 같은 값(U − L)이다(명세의 지표 목록을 그대로 둔다).
  16. pit_grid 는 np.interp 의 규칙(마지막 Q_j ≤ y 의 구간, 동률 처리 포함)을 벡터화해 셀마다 np.interp 를 부른 값과 같게 했다.
  17. 로컬 실행 규칙(2026-09-30)에 맞춰 set_torch_threads, gpu_memory_used, free_gpus 를 더했다. 명세에 없는 도우미다.
  18. (개정 1) is_env_error: 환경·자원 오류(ImportError, MemoryError, torch.cuda.OutOfMemoryError, CUDA 런타임 오류)를 가려 하네스가
      키 하나의 실패가 아니라 조각 실패로 두게 한다. cal_groups: 보정 집단 G 를 로그 비가 정의되는 원천 셀 수로 정한다(h44·h45 공통).
      check_blockindex: lgu_blockindex.csv 의 쓰기와 대조(h44·h45 공통). claim_threads·release_threads·running_threads: 하네스 사이의
      스레드 합계 8 규칙을 실행 중 등록부로 강제한다.
  19. (개정 1) fit_gen 의 device 기본값: CUDA_VISIBLE_DEVICES 가 없는 프로세스는 'cpu' 를 쓴다(명세는 tab_models._dev()).
  20. (개정 1) atomic_text·atomic_npz 는 실패하면 임시 파일(<이름>.tmp<pid>)을 지운다.

확인하지 못한 것
  개정 1 단계에서 py_compile, pyflakes, 가벼운 단위 시험(tests/test_lgu_common.py 의 학습 없는 시험, CPU 스레드 2)을 실행했다. 학습을 하는
  시험(LG_RUN_HEAVY=1)과 GPU 스모크는 실행하지 않았다(GPU 2·1번이 사용 중이었다). 생성 학습기 래퍼가 legacy 설정에서 tab_models.fit_predict 와 같은 표본을 내는지(시험 k), nflow 분위와 밀도(시험 l),
  CatBoost 1.2.10 의 MultiQuantile 손실 동작, GPU 에서의 수치 재현성은 아직 확인하지 않았다. 로컬 환경은 python 3.9, torch 2.6.0,
  numpy 1.26.4, catboost 1.2.10 이다(Rescale 이미지의 python 3.11, torch 2.4.1 과 다르다).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import polar.h4_common as H4
from polar.h4_common import seed_of, boot_weights, boot_p, norm_key, select_keys
from polar.m1_stats import holm
from polar.m1_core import half_split_blocks, eval_mask
from polar.physics import load_physics_inputs, johansen_k, _stefan_full

__all__ = [
    # 재수출(h44–h46 이 LC.<이름> 으로 쓴다)
    "seed_of", "boot_weights", "boot_p", "norm_key", "select_keys", "holm", "half_split_blocks", "eval_mask",
    # 상수
    "TAUS", "LEVELS", "ALPHAS", "Q5", "S_GEN", "NFLOW_CLAMP", "LEGACY_CLAMP", "CAP_CELLS", "SIGMA_CLIP", "NBOOT", "MIN_CAL_CELLS",
    "MIN_BLOCKS_CI", "KAPPA", "EMU_SPLITS", "EMU_DRAWS", "EQ_CM", "EQ_REL", "EQ_REL_AUX", "COV_MIN", "FAIL_LOGABS", "FAIL_FRAC",
    "H_GRID", "PHYS_SR", "PHYS_NT", "PHYS_INPUT_COLS", "METRICS", "METRICS_2STAGE", "FORMAT", "GEN_T", "PIT_BINS", "EARTH_R_KM",
    "THREAD_VARS",
    # 함수와 클래스
    "peek_threads", "set_thread_env", "set_torch_threads", "gpu_memory_used", "free_gpus", "run_permitted", "require_permission",
    "atomic_text", "atomic_npz", "file_sha", "cfg_hash", "unit_state", "load_flags", "nearest_km", "prep_stats", "prep_apply", "row_weights", "block_val_mask", "q_from_samples",
    "crossing_stats", "rearrange", "grid_index", "interval_of", "median_of", "grid_from_center", "loc_only", "to_cm", "GenHandle",
    "make_flow_class", "fit_gen", "gen_samples", "nflow_quantiles", "nflow_logpdf", "nflow_cdf_table", "gen_quantiles",
    "cfm_quantiles_ode", "fit_failed", "cb_multiquantile", "pinball", "crps_grid", "interval_parts", "pit_grid", "pit_hist",
    "score_cells", "score_interval_only", "alpha_suffix", "ScoreStore", "save_score_stores", "load_score_stores",
    "stores_for_target", "wquantile", "hier_cell_weights", "hier_quantiles", "pooled_quantile", "hier_quantiles_boot", "ls_E",
    "center_coef", "emulation_plan", "BlockIndex", "global_mult", "agg_cell", "agg_beq", "agg_rmse_cell", "agg_rmse_beq", "ci95",
    "combine_same_index", "strat_mean", "one_stage_delta", "verdict4", "eq_state", "rel_margin", "half_width",
    "phys_log_samples", "is_env_error", "cal_groups", "check_blockindex", "claim_threads", "release_threads", "running_threads",
]

# ================================================================ 1. 상수(바꾸면 사전 등록에서 벗어난다)
TAUS = np.round(np.arange(1, 100) / 100, 2)                 # 분위 격자 99점 0.01–0.99
LEVELS = np.round(np.arange(1, 50) * 0.02, 2)               # 대칭 구간의 포함 수준 49점 0.02–0.98
ALPHAS = (0.10, 0.20)                                        # 구간의 명목 미포함률(주 0.1, 보조 0.2)
Q5 = (0.05, 0.10, 0.50, 0.90, 0.95)                          # 5점 분위(CatBoost 다분위, 보정 셀의 생성 분위)
S_GEN = 512                                                  # cfm 셀당 표본 수
NFLOW_CLAMP = 2.0                                            # nflow 아핀 로그 척도 절단(LGU 규약)
LEGACY_CLAMP = 5.0                                           # tab_models 의 절단(기존 방식)
CAP_CELLS = 100                                              # 행 가중 min(1, 100/블록 셀 수)
SIGMA_CLIP = (0.02, 2.0)                                     # 정규화기 σ 의 절단 범위(로그 척도)
NBOOT = 2000
MIN_CAL_CELLS = 20                                           # 보정 집단 지역의 최소 셀 수
MIN_BLOCKS_CI = 8                                            # CI 풀에 들어가는 지역의 최소 채점 블록 수
KAPPA = 10.0                                                 # E_n 수축 강도
EMU_SPLITS = (1, 2, 3)                                       # 의사 대상 모사의 분할
EMU_DRAWS = 3                                                # 의사 대상 모사의 추출 수
EQ_CM = 0.5                                                  # 점 예측 동등성 한계(cm)
EQ_REL = 0.05                                                # 구간 점수 동등성 주 한계(기준 점수 대비)
EQ_REL_AUX = 0.10                                            # 구간 점수 동등성 보조 한계
COV_MIN = 0.80                                               # 구간 점수 우세를 문장으로 쓰는 커버리지 하한
FAIL_LOGABS = 3.0                                            # 적합 실패: 표본 밖 중앙값의 로그 척도 절댓값 기준
FAIL_FRAC = 0.01                                             # 적합 실패: 위 기준을 넘는 셀 비율
H_GRID = 0.005                                               # 로그 척도 구적 격자 간격
PHYS_SR = (0.4, 1.0)                                         # 포화도 Sr 표집 범위(가정값)
PHYS_NT = (0.6, 1.0)                                         # n_t 표집 범위(가정값)
PHYS_INPUT_COLS = ("e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_maat", "e5_twarm", "e5_tcold", "e5_swe",
                   "sg_bdod_5_15", "sg_sand_5_15", "sg_cfvo_5_15", "sg_soc_5_15")
METRICS = ("is10", "is10_w", "is10_u", "is10_o", "cov10", "wid10", "lhw10", "inf10",
           "is20", "is20_w", "is20_u", "is20_o", "cov20", "wid20", "lhw20", "inf20",
           "crps", "crps_log", "pb05", "pb50", "pb95", "se", "ae", "bias")
METRICS_2STAGE = ("is10", "cov10", "wid10", "lhw10", "is20", "cov20", "wid20", "lhw20", "se")
FORMAT = "lgu_blockscore_v1"
GEN_T = 200                                                  # ddpm 단계 수(tab_models 의 코드 값. 머리말의 T=100 은 코드와 다르다)
PIT_BINS = 10
EARTH_R_KM = 6371.0                                          # h42, m1_ext.haversine_km 과 같은 반지름
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

PERMIT_MSG = ("[거부] 학습과 채점을 하는 실행(스모크, 사전 점검, 본 실행, 집계)은 허용 표지가 있어야 한다. 로컬 GPU 서버에서는 "
              "--allow-local 을 주고 nice -n 10 으로 실행한다(사용자 지시 2026-09-30: LGU 는 로컬 GPU 서버에서 실행한다. "
              "CPU 스레드 합계 8 이하, GPU 는 배정된 번호만 쓰고 쓰기 직전에 메모리 사용 0 MiB 를 확인한다). "
              "환경 변수 LG_RESCALE=1 은 허용 표지로만 인정하며 로컬 자원 규칙(nice, 스레드 합계, GPU 허용 목록과 0 MiB 확인)은 "
              "그대로 적용된다. 허용 표지가 없으면 --count-only 만 실행한다")
RUN_REGISTRY_ENV = "LGU_RUN_REGISTRY"                        # 실행 중 스레드 등록부 위치를 바꾸는 환경 변수(시험용)
RUN_REGISTRY_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "processed" / "lgu" / ".running"
ENV_ERROR_MARKERS = ("CUDA error", "CUDA out of memory", "out of memory", "CUBLAS_STATUS", "CUDNN_STATUS", "cuDNN error",
                     "device-side assert", "NCCL error", "CUDA driver")


# ================================================================ 2. 실행 보호와 조각 도우미
def peek_threads(default="4", argv=None, environ=None) -> str:
    """스레드 수 결정 규칙(h42._peek_threads 와 같다). --threads 값을 읽고, 허용 표지(LG_RESCALE=1 또는 --allow-local)가 없으면 '1'.

    이 모듈은 numpy 를 import 하므로 스크립트는 같은 규칙을 numpy import 전에 머리에 다시 둔다. 이 함수는 기준과 시험용이다.
    """
    av = list(sys.argv if argv is None else argv)
    env = os.environ if environ is None else environ
    if not (env.get("LG_RESCALE", "") == "1" or "--allow-local" in av):
        return "1"
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return str(av[i + 1])
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return str(default)


def set_thread_env(n) -> None:
    """OMP·OPENBLAS·MKL·NUMEXPR 스레드 환경 변수를 n 으로 둔다(워커 초기화용. 이미 적재된 BLAS 에는 영향이 없을 수 있다)."""
    for v in THREAD_VARS:
        os.environ[v] = str(int(n))


def set_torch_threads(n) -> int:
    """torch 의 연산 스레드 수를 n 으로 둔다(torch 를 이 함수 안에서 import 한다). 반환은 설정 뒤의 torch.get_num_threads().

    연산자 간 병렬 스레드 수(set_num_interop_threads)는 병렬 작업이 시작된 뒤에는 바꿀 수 없으므로 실패하면 그대로 둔다.
    로컬 규칙(스레드 합계 8 이하)은 호출자가 워커 수 × n 으로 맞춘다.
    """
    import torch
    n = max(1, int(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass
    return int(torch.get_num_threads())


def gpu_memory_used(ids=None, timeout=30) -> dict:
    """nvidia-smi 로 GPU 메모리 사용량(MiB)을 읽는다. 반환 {물리 번호: MiB}. ids 가 있으면 그 번호만 돌려주고 목록에 없는 번호는 None.

    nvidia-smi 가 없거나 실패하면 빈 dict(ids 가 있으면 모두 None). 번호는 CUDA_VISIBLE_DEVICES 재배치 전의 물리 번호다.
    """
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout, check=False)
        out = {}
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                a = [x.strip() for x in line.split(",")]
                if len(a) >= 2 and a[0].isdigit():
                    try:
                        out[int(a[0])] = float(a[1])
                    except ValueError:
                        out[int(a[0])] = None
    except (OSError, subprocess.SubprocessError):
        out = {}
    if ids is None:
        return out
    return {int(i): out.get(int(i)) for i in ids}


def free_gpus(ids, max_mib=0.0) -> list:
    """ids 가운데 메모리 사용이 max_mib 이하인 GPU 물리 번호 목록(순서 유지). 읽지 못한 번호는 사용 중으로 본다.

    로컬 규칙(2026-09-30): 쓰기 직전에 메모리 사용이 0 MiB 인지 확인하고 사용 중이면 쓰지 않는다(기본 max_mib = 0).
    """
    used = gpu_memory_used(ids)
    return [int(i) for i in ids if used.get(int(i)) is not None and float(used[int(i)]) <= float(max_mib)]


def run_permitted(allow_local=False, environ=None) -> bool:
    """학습·채점 실행의 허용 여부: allow_local 또는 환경 변수 LG_RESCALE == '1'."""
    env = os.environ if environ is None else environ
    return bool(allow_local) or env.get("LG_RESCALE", "") == "1"


def require_permission(a, environ=None) -> bool:
    """허용 표지가 없고 --count-only 가 아니면 자료를 읽기 전에 SystemExit. --summarize-only, --smoke 도 거부한다.

    a: argparse 결과(allow_local, count_only 속성). 반환: 허용이면 True, 허용 표지 없는 --count-only 이면 False.
    """
    if run_permitted(getattr(a, "allow_local", False), environ):
        return True
    if bool(getattr(a, "count_only", False)):
        return False
    raise SystemExit(PERMIT_MSG)


def atomic_text(path, text: str) -> Path:
    """임시 파일에 쓴 뒤 os.replace 로 교체한다(UTF-8). 쓰기나 교체가 실패하면 임시 파일을 지우고 예외를 다시 던진다."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def atomic_npz(path, **arrays) -> Path:
    """임시 파일에 np.savez_compressed 로 쓴 뒤 os.replace 로 교체한다(pickle 을 쓰지 않는 배열만 넣는다). 실패하면 임시 파일을 지운다."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def is_env_error(ex) -> bool:
    """환경·자원 오류 여부. ImportError, MemoryError, torch.cuda.OutOfMemoryError(형 이름 OutOfMemoryError), CUDA·cuBLAS·cuDNN 런타임
    오류(메시지 표지 ENV_ERROR_MARKERS 를 가진 RuntimeError)는 참이다. 이런 오류는 키 하나의 실패로 두지 않고 조각 전체를 실패로 두어
    --resume 이 다시 실행하게 한다(h40 의 'ImportError·MemoryError 는 조각 전체를 실패로 둔다' 규약의 확장). torch 를 import 하지 않는다."""
    if isinstance(ex, (ImportError, MemoryError)):
        return True
    if type(ex).__name__ == "OutOfMemoryError":
        return True
    if isinstance(ex, RuntimeError):
        msg = str(ex)
        return any(m in msg for m in ENV_ERROR_MARKERS)
    return False


def cal_groups(macro, src_idx, ok, min_cells=MIN_CAL_CELLS) -> list:
    """보정 집단 G(계획서 2.1절): 원천 셀 가운데 로그 비가 정의되는 셀(ok: y, s 유한이고 양수)이 min_cells 개 이상인 macro 지역(사전순).

    h44(실험 A)와 h45(실험 B)가 같은 정의를 쓰도록 이 함수를 공유한다(보정 셀은 ok 셀만 쓰므로 ok 셀 수를 기준으로 한다).
    """
    mac = np.asarray(macro).astype(str)
    src = np.asarray(src_idx, np.int64)
    okm = np.asarray(ok, bool)
    s_ok = src[okm[src]]
    reg, cnt = np.unique(mac[s_ok], return_counts=True)
    return sorted(str(r) for r, c in zip(reg, cnt) if int(c) >= int(min_cells))


def check_blockindex(path, bi) -> Path:
    """전역 블록 색인 대조 파일(lgu_blockindex.csv)을 쓰거나(없을 때) 대조한다(있을 때). h44 와 h45 가 같은 파일을 쓴다.
    내용(macro, block, n_cells)이 다르면 자료 판이 다르므로 SystemExit(덮어쓰지 않는다)."""
    p = Path(path)
    fr = bi.to_frame()
    if p.exists():
        old = pd.read_csv(p, dtype=dict(macro=str, block=str))
        same = (list(old.columns) == list(fr.columns) and len(old) == len(fr)
                and np.array_equal(old.macro.astype(str).values, fr.macro.astype(str).values)
                and np.array_equal(old.block.astype(str).values, fr.block.astype(str).values)
                and np.array_equal(old.n_cells.values.astype(np.int64), fr.n_cells.values.astype(np.int64)))
        if not same:
            raise SystemExit(f"[blockindex] {p} 가 현재 자료의 전역 블록 색인과 다르다. 자료 판을 확인한다(덮어쓰지 않는다)")
        return p
    return atomic_text(p, fr.to_csv(index=False))


def _registry_dir(registry=None) -> Path:
    if registry is not None:
        return Path(registry)
    env = os.environ.get(RUN_REGISTRY_ENV, "")
    return Path(env) if env else RUN_REGISTRY_DEFAULT


def _start_ticks(pid) -> str:
    """/proc/<pid>/stat 의 시작 시각(22번째 필드, 부팅 뒤 틱). 읽을 수 없으면 빈 문자열."""
    try:
        txt = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
        return txt.rpartition(")")[2].split()[19]
    except (OSError, ValueError, IndexError):
        return ""


def _pid_alive(pid, start="") -> bool:
    """pid 가 살아 있는지. 등록 때의 시작 시각(start)이 있으면 현재 프로세스의 시작 시각과 같아야 한다(pid 재사용 대비)."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except (OSError, ValueError):
        return False
    if start:
        cur = _start_ticks(pid)
        return (cur == str(start)) if cur else True
    return True


def running_threads(registry=None, exclude_pid=None) -> list:
    """실행 중 등록부의 살아 있는 항목 목록 [dict(pid, tag, script, threads, started)]. 죽은 항목의 파일은 지운다."""
    d = _registry_dir(registry)
    out = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = int(r.get("pid", -1))
        if exclude_pid is not None and pid == int(exclude_pid):
            continue
        if _pid_alive(pid, str(r.get("start", ""))):
            out.append(r)
        else:
            try:
                p.unlink()
            except OSError:
                pass
    return out


def claim_threads(tag, threads, cap=8, registry=None, script=None) -> Path:
    """LGU 하네스 사이의 CPU 스레드 합계 규칙(사용자 지시 2026-09-30: 이 실험의 스레드 합계 8 이하)을 등록부로 강제한다.

    등록부(기본 data/processed/lgu/.running, 환경 변수 LGU_RUN_REGISTRY 로 바꿀 수 있다)의 살아 있는 항목의 스레드 합과 이번 실행의
    threads 를 더해 cap 을 넘으면 SystemExit. 넘지 않으면 <pid>.json 을 쓰고 경로를 돌려준다. 파일 잠금(fcntl)으로 동시 등록을 막는다.
    실행이 끝나면 release_threads 로 지운다(비정상 종료로 남은 파일은 다음 확인에서 pid 로 걸러 지운다).
    """
    import fcntl
    d = _registry_dir(registry)
    d.mkdir(parents=True, exist_ok=True)
    script = Path(script or sys.argv[0] or "python").name
    lock = d / ".lock"
    with open(lock, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            others = running_threads(d, exclude_pid=os.getpid())
            used = int(sum(int(r.get("threads", 0)) for r in others))
            if used + int(threads) > int(cap):
                desc = ", ".join(f"{r.get('tag')}(pid {r.get('pid')}, 스레드 {r.get('threads')})" for r in others)
                raise SystemExit(f"[거부] 실행 중인 LGU 하네스의 스레드 합 {used} + 이번 실행 {int(threads)} 이 상한 {int(cap)} 을 넘는다"
                                 f"(실행 중: {desc}). 앞 실행이 끝난 뒤 다시 실행하거나 --workers·--threads 를 줄인다")
            p = d / f"{os.getpid()}.json"
            p.write_text(json.dumps(dict(pid=os.getpid(), start=_start_ticks(os.getpid()), tag=str(tag), script=script, threads=int(threads),
                                         started=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False), encoding="utf-8")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return p


def release_threads(path) -> None:
    """claim_threads 가 쓴 등록 파일을 지운다(없으면 무시)."""
    if path is None:
        return
    try:
        Path(path).unlink()
    except OSError:
        pass


def file_sha(path) -> str:
    """파일 내용의 sha1 앞 12자. 읽을 수 없으면 'none'."""
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "none"


def cfg_hash(cfg: dict) -> str:
    """설정 dict 의 json(sort_keys) sha1 앞 12자."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def unit_state(unit_json, need_files, cfg: dict):
    """조각의 완료 여부 (bool, 사유). 완료 = unit.json 과 need_files 가 모두 있고, cfg_hash 가 같고, status 가 failed 가 아니다."""
    uj = Path(unit_json)
    miss = [Path(p).name for p in [uj, *list(need_files or [])] if not Path(p).exists()]
    if miss:
        return False, "조각 없음(" + ", ".join(miss) + ")"
    try:
        u = json.loads(uj.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != cfg_hash(cfg):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    return True, str(u.get("status", "ok"))


def load_flags(path, loc_id, missing_ok=False) -> dict:
    """셀 표지 표(lgx_label_flags_v1.csv)를 loc_id 순서로 읽는다. 반환 dict(early, gpr, probe: bool 배열).

    값 0.5 이상이 참이다(h42.ExtTables 와 같은 기준). 표에 없는 loc_id 가 있으면 SystemExit.
    표 파일이 없으면 missing_ok 가 참일 때 모두 거짓, 아니면 SystemExit.
    """
    loc = np.asarray(loc_id)
    p = Path(path)
    if not p.exists():
        if missing_ok:
            z = np.zeros(len(loc), bool)
            return dict(early=z.copy(), gpr=z.copy(), probe=z.copy())
        raise SystemExit(f"[flags] 셀 표지 표 {p} 가 없다")
    f = pd.read_csv(p, usecols=["loc_id", "early", "gpr", "probe"]).drop_duplicates("loc_id").set_index("loc_id")
    miss = int((~np.isin(loc, f.index.values)).sum())
    if miss:
        raise SystemExit(f"[flags] 셀 표지 표 {p} 에 없는 loc_id 가 {miss}개다. 자료 판과 표를 확인한다")
    return {c: f[c].reindex(loc).values.astype(float) >= 0.5 for c in ("early", "gpr", "probe")}


def _xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def nearest_km(lat, lon, lat_ref, lon_ref) -> np.ndarray:
    """각 점에서 기준 점 집합까지의 최근접 대권 거리(km). 단위 구의 현 길이 cKDTree 최근접 후 2R·arcsin(현/2).

    기준 점이 없으면 NaN. 좌표가 비유한인 점은 NaN.
    """
    from scipy.spatial import cKDTree
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    lr = np.asarray(lat_ref, float); lnr = np.asarray(lon_ref, float)
    out = np.full(len(lat), np.nan)
    okr = np.isfinite(lr) & np.isfinite(lnr)
    okq = np.isfinite(lat) & np.isfinite(lon)
    if okr.sum() == 0 or okq.sum() == 0:
        return out
    ch, _ = cKDTree(_xyz(lr[okr], lnr[okr])).query(_xyz(lat[okq], lon[okq]), k=1)
    out[okq] = 2.0 * EARTH_R_KM * np.arcsin(np.clip(ch / 2.0, 0.0, 1.0))
    return out


# ================================================================ 3. 전처리
def prep_stats(Xtr):
    """표준화·중앙값 대체 통계 (med, mu, sd). h40._prep_stats 와 같은 식(학습 행렬에서만 구한다, sd + 1e-6)."""
    Xtr = np.asarray(Xtr, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Xtr, 0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isnan(Xtr), med, Xtr)
    return med, Xf.mean(0), Xf.std(0) + 1e-6


def prep_apply(X, stats) -> np.ndarray:
    """prep_stats 의 통계로 결측 대체와 표준화. float32. h40._prep_apply 와 같은 식."""
    med, mu, sd = stats
    X = np.asarray(X, float)
    return ((np.where(np.isnan(X), med, X) - mu) / sd).astype(np.float32)


def row_weights(blocks) -> np.ndarray:
    """행 가중 min(1, CAP_CELLS / 그 행이 속한 블록의 셀 수)."""
    b = np.asarray(blocks)
    if b.size == 0:
        return np.zeros(0)
    _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    return np.minimum(1.0, CAP_CELLS / cnt[inv].astype(float))


def block_val_mask(groups, seed, frac=0.1):
    """블록 단위 검증 행 표지 (mask, flag).

    ub = np.unique(groups), order = RandomState(seed).permutation(len(ub)). 순서대로 블록을 검증에 넣고 누적 셀 수가 frac·n 이상이면 멈춘다.
    블록이 1개이거나, 검증 몫이 0.5 를 넘거나, 학습 쪽 또는 검증 쪽이 비면 RandomState(seed).rand(n) < 0.1 로 되돌리고 flag = 'val_random'.
    정상이면 flag = 'val_block'.
    """
    g = np.asarray(groups)
    n = len(g)
    ub, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    order = np.random.RandomState(int(seed)).permutation(len(ub))
    pick, acc = [], 0
    for j in order:
        if acc >= frac * n:
            break
        pick.append(int(j)); acc += int(cnt[j])
    mask = np.isin(inv, pick) if n else np.zeros(0, bool)
    share = float(mask.mean()) if n else 0.0
    if len(ub) < 2 or share > 0.5 or mask.sum() == 0 or (~mask).sum() == 0:
        return np.random.RandomState(int(seed)).rand(n) < 0.1, "val_random"
    return mask, "val_block"


# ================================================================ 4. 분위 격자
def q_from_samples(samples, taus=TAUS) -> np.ndarray:
    """표본 (S, n) → 분위 (T, n). np.quantile(axis=0, method='linear'). 비유한 표본이 있는 셀은 NaN 열."""
    S = np.asarray(samples, float)
    tau = np.asarray(taus, float)
    ok = np.all(np.isfinite(S), axis=0)
    out = np.full((len(tau), S.shape[1]), np.nan)
    if ok.any():
        out[:, ok] = np.quantile(S[:, ok], tau, axis=0, method="linear")
    return out


def crossing_stats(Q, tol=1e-9) -> dict:
    """재배열 전 분위 격자 Q (T, n)의 인접 분위 역전 통계. 모든 값이 유한한 열만 센다.

    반환 dict(frac_cells(역전이 있는 셀 비율), frac_pairs(역전 쌍 비율), max_drop(최대 역전 폭), n_cells, n_cross_cells, n_pairs, n_cross_pairs).
    """
    Q = np.asarray(Q, float)
    if Q.ndim == 1:
        Q = Q[:, None]
    fin = np.all(np.isfinite(Q), axis=0)
    D = np.diff(Q[:, fin], axis=0)
    viol = D < -float(tol)
    nc, npair = int(fin.sum()), int(D.size)
    cc, cp = int(viol.any(axis=0).sum()) if nc else 0, int(viol.sum())
    md = float((-D[viol]).max()) if cp else 0.0
    return dict(frac_cells=cc / nc if nc else np.nan, frac_pairs=cp / npair if npair else np.nan, max_drop=md,
                n_cells=nc, n_cross_cells=cc, n_pairs=npair, n_cross_pairs=cp)


def rearrange(Q) -> np.ndarray:
    """분위 방향 정렬(재배열). np.sort(Q, axis=0)."""
    return np.sort(np.asarray(Q, float), axis=0)


def grid_index(tau) -> int:
    """TAUS 의 행 번호 int(round(tau·100)) − 1. TAUS 에 없는 값이면 ValueError."""
    t = float(tau)
    i = int(round(t * 100)) - 1
    if not (0 <= i < len(TAUS)) or abs(TAUS[i] - t) > 1e-9:
        raise ValueError(f"분위 {tau} 는 격자 TAUS 에 없다")
    return i


def interval_of(Q, alpha):
    """격자 Q (99, …)의 (1 − α) 구간 (Q[α/2], Q[1 − α/2])."""
    a = float(alpha)
    return Q[grid_index(a / 2)], Q[grid_index(1 - a / 2)]


def median_of(Q):
    """격자 Q (99, …)의 중앙값 행 Q[49]."""
    return Q[49]


def grid_from_center(center, half) -> np.ndarray:
    """대칭 구간 방법의 격자. center (n,), half (49,) 또는 (49, n)의 로그 반폭(수준 LEVELS 순서) → Q (99, n).

    half 는 수준 방향 누적 최댓값으로 단조화한다. j = 1..49 에 대해 Q[49 − j] = center·exp(−half[j − 1]),
    Q[49 + j] = center·exp(half[j − 1]), Q[49] = center. half 가 inf 이면 하한 0, 상한 inf 다.
    """
    c = np.asarray(center, float).ravel()
    h = np.asarray(half, float)
    if h.ndim == 1:
        h = h[:, None]
    if h.shape[0] != len(LEVELS):
        raise ValueError(f"half 의 행 수 {h.shape[0]} 가 수준 수 {len(LEVELS)} 와 다르다")
    h = np.maximum.accumulate(h, axis=0)
    Q = np.empty((len(TAUS), len(c)))
    j = np.arange(1, len(LEVELS) + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        Q[49 - j] = c[None, :] * np.exp(-h[j - 1])
        Q[49 + j] = c[None, :] * np.exp(h[j - 1])
    Q[49] = c
    return Q


def _mid_index(T, mid=None) -> int:
    if mid is not None:
        return int(mid)
    if T == len(TAUS):
        return 49
    if T == len(Q5):
        return 2
    raise ValueError(f"분위 행 수 {T} 의 중앙값 위치를 정할 수 없다(mid 를 준다)")


def loc_only(Qz, lam, mid=None) -> np.ndarray:
    """위치 전용 λ 규약: lam·Qz[mid] + (Qz − Qz[mid]). λ 는 중심에만 곱한다. mid 기본은 99행 49, 5행(Q5) 2."""
    Q = np.asarray(Qz, float)
    m = Q[_mid_index(Q.shape[0], mid)]
    return float(lam) * m[None] + (Q - m[None])


def to_cm(anchor, Qz) -> np.ndarray:
    """로그 비 분위 → cm: anchor[None, :]·exp(Qz)."""
    return np.asarray(anchor, float)[None, :] * np.exp(np.asarray(Qz, float))


# ================================================================ 5. 생성 학습기 래퍼(tab_models 를 고치지 않는다)
class GenHandle:
    """생성 학습기 적합 결과. net 은 eval 상태의 torch 모듈이다. 목표 표준화는 (y − ymu)/ysd 다."""

    def __init__(self, name, net, ymu, ysd, d, clamp, median_zero, val_loss, epochs_run, n_train, n_val, flag, legacy,
                 device="cpu", n_nonfinite=0):
        self.name, self.net, self.ymu, self.ysd, self.d = str(name), net, float(ymu), float(ysd), int(d)
        self.clamp, self.median_zero = float(clamp), bool(median_zero)
        self.val_loss, self.epochs_run = float(val_loss), int(epochs_run)
        self.n_train, self.n_val, self.flag, self.legacy = int(n_train), int(n_val), str(flag), bool(legacy)
        self.device, self.n_nonfinite = str(device), int(n_nonfinite)

    def info(self) -> dict:
        """JSON 직렬화 가능한 요약(runs.csv, unit.json 용)."""
        return dict(name=self.name, ymu=self.ymu, ysd=self.ysd, d=self.d, clamp=self.clamp, median_zero=self.median_zero,
                    val_loss=self.val_loss, epochs_run=self.epochs_run, n_train=self.n_train, n_val=self.n_val, flag=self.flag,
                    legacy=self.legacy, device=self.device, n_nonfinite=self.n_nonfinite)


def make_flow_class(clamp=NFLOW_CLAMP, median_zero=False):
    """tab_models._nflow_mods() 의 조건부 spline 흐름 하위 클래스.

    _affine(x) -> (ps, mu, ls): ls 를 ±clamp 로 자른다. median_zero 이면 z = 0 을 ps 순서의 순변환에 통과시킨 g0 로 mu = −exp(ls)·g0 를
    쓴다(출력 = exp(ls)·(g(z) − g(0)), 모든 x 에서 중앙값 0). log_prob, sample 은 Base 와 같은 연산 순서로 _affine 을 쓴다
    (clamp 5.0, median_zero 거짓이면 Base 와 같은 값). quantile(x, z0(T,)) -> (T, n), log_prob_params(y, ps, mu, ls) 를 더했다.
    """
    import torch
    from polar import tab_models as TM
    Base = TM._nflow_mods()
    rq = TM._rq_spline
    c_, mz_ = float(clamp), bool(median_zero)
    log2pi = float(np.log(2 * np.pi))

    class LGUSplineFlow(Base):
        CLAMP = c_
        MEDIAN_ZERO = mz_

        def _affine(self, x):
            ps, mu, ls = self._params(x)
            ls = ls.clamp(-self.CLAMP, self.CLAMP)
            if self.MEDIAN_ZERO:
                g0 = torch.zeros(x.shape[0], device=x.device, dtype=ls.dtype)
                for (uw, uh, ud) in ps:
                    g0, _ = rq(g0, uw, uh, ud, inverse=False)
                mu = -torch.exp(ls) * g0
            return ps, mu, ls

        def log_prob_params(self, y, ps, mu, ls):
            z = (y - mu) * torch.exp(-ls); lad = -ls
            for (uw, uh, ud) in reversed(ps):
                z, l_ = rq(z, uw, uh, ud, inverse=True); lad = lad + l_
            return -0.5 * z ** 2 - 0.5 * log2pi + lad

        def log_prob(self, y, x):
            ps, mu, ls = self._affine(x)
            return self.log_prob_params(y, ps, mu, ls)

        def sample(self, x, S):
            ps, mu, ls = self._affine(x)
            n = len(x)
            z = torch.randn(S, n, device=x.device)
            outs = []
            for s in range(S):
                zz = z[s]
                for (uw, uh, ud) in ps:
                    zz, _ = rq(zz, uw, uh, ud, inverse=False)
                outs.append(mu + torch.exp(ls) * zz)
            return torch.stack(outs, 0)

        def quantile(self, x, z0):
            ps, mu, ls = self._affine(x)
            sc = torch.exp(ls)
            outs = []
            for v in np.asarray(z0, float).ravel():
                zz = torch.full((x.shape[0],), float(v), device=x.device, dtype=ls.dtype)
                for (uw, uh, ud) in ps:
                    zz, _ = rq(zz, uw, uh, ud, inverse=False)
                outs.append(mu + sc * zz)
            return torch.stack(outs, 0)

    LGUSplineFlow.__name__ = f"LGUSplineFlow_c{c_:g}_{'mz' if mz_ else 'free'}"
    return LGUSplineFlow


def fit_gen(name, Xtr, ytr, groups=None, weights=None, seed=0, epochs=100, clamp=NFLOW_CLAMP, median_zero=False,
            val="block", device=None) -> GenHandle:
    """조건부 생성 학습기(cfm, ddpm, nflow)를 학습해 GenHandle 을 돌려준다(표본을 만들지 않는다).

    절차는 tab_models._fit_generative 의 학습부와 같은 순서다: torch.manual_seed(seed) → ymu, ysd → 검증 표지 → 망 생성 →
    Adam(1e-3, weight_decay 1e-5), 배치 8,192, 기울기 절단 5.0, ddpm T = 200·betas linspace(1e-4, 0.05) → epoch 마다 torch.randperm,
    손실이 비유한이면 그 epoch 의 배치 반복을 멈춘다 → 검증 손실은 torch.manual_seed(seed + 1000 + rep), rep 0–2 의 평균 →
    best − 1e-4 기준 조기 종료(인내 12, 최소 15 epoch) → 최선 상태 적재.
    - median_zero(nflow 만): ymu = 0, ysd = sqrt(가중 평균 y²) + 1e-6. 아니면 ymu = mean(y), ysd = std(y) + 1e-6(가중 없음).
    - val: 'random' 이면 RandomState(seed).rand(n) < 0.1, 'block' 이면 block_val_mask(groups, seed). groups 가 없으면 'random'.
    - weights: 행 가중(보통 row_weights(블록)). 손실은 셀별 값의 가중 평균이다. None 이면 tab_models 와 같은 단순 평균이다.
    legacy 설정(val='random', weights=None, clamp=LEGACY_CLAMP, median_zero=False)은 tab_models.fit_predict 와 같은 학습을 한다.
    Xtr 는 결측이 없는 표준화 행렬(prep_apply 결과)이어야 한다.
    - device: None 이면 환경 변수 CUDA_VISIBLE_DEVICES 가 있을 때만 tab_models._dev()(보이는 첫 GPU 또는 CPU)를 쓰고, 없으면 'cpu' 다.
      공유 서버 규칙(배정 GPU 만 사용)에 따라 GPU 를 쓰려면 device 또는 CUDA_VISIBLE_DEVICES 를 명시한다.
    """
    if name not in ("cfm", "ddpm", "nflow"):
        raise ValueError(f"알 수 없는 생성 학습기 {name}")
    if median_zero and name != "nflow":
        raise ValueError("위치 고정 변형(median_zero)은 nflow 에만 있다")
    if val not in ("random", "block"):
        raise ValueError(f"val 은 'random' 또는 'block' 이다: {val}")
    seed = int(seed)
    Xtr = np.asarray(Xtr).astype(np.float32)
    ytr = np.asarray(ytr)
    if not np.issubdtype(ytr.dtype, np.floating):
        ytr = ytr.astype(float)
    n = len(ytr)
    if Xtr.ndim != 2 or Xtr.shape[0] != n:
        raise ValueError(f"Xtr 의 모양 {Xtr.shape} 가 목표 수 {n} 와 맞지 않는다")
    if not (np.all(np.isfinite(Xtr)) and np.all(np.isfinite(ytr))):
        raise ValueError("fit_gen 의 입력에 비유한 값이 있다(prep_apply 로 결측 대체와 표준화를 한 행렬과 유한한 목표를 준다)")
    w_all = None if weights is None else np.asarray(weights, float)
    if w_all is not None and (w_all.shape != (n,) or not np.all(np.isfinite(w_all)) or (w_all < 0).any() or w_all.sum() <= 0):
        raise ValueError("weights 는 길이가 목표 수와 같은 유한한 비음수 배열이고 합이 양수여야 한다")
    if median_zero:
        ymu = 0.0
        y2 = ytr.astype(float) ** 2
        msq = float(y2.mean()) if w_all is None else float((w_all * y2).sum() / w_all.sum())
        ysd = float(np.sqrt(msq) + 1e-6)
    else:
        ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    if val == "random" or groups is None:
        va = np.random.RandomState(seed).rand(n) < 0.1
        vflag = "val_random" if val == "random" else "val_random(no_groups)"
    else:
        if len(np.asarray(groups)) != n:
            raise ValueError("groups 의 길이가 목표 수와 다르다")
        va, vflag = block_val_mask(groups, seed)
    tr = ~va
    # 입력 확인과 numpy 전처리를 마친 뒤 torch 를 적재한다. torch 난수는 아래 manual_seed 뒤에만 쓰이므로
    # tab_models._fit_generative(manual_seed → 전처리 → 망 생성)와 난수 순서가 같다.
    import torch
    from polar import tab_models as TM
    if device is not None:
        dev = device
    elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        dev = "cpu"                                                      # 배정 GPU 를 명시하지 않은 프로세스는 GPU 를 쓰지 않는다
    else:
        dev = TM._dev()
    torch.manual_seed(seed)
    Xt, yt = torch.tensor(Xtr[tr]), torch.tensor(yz[tr])
    Xv, yv = torch.tensor(Xtr[va]).to(dev), torch.tensor(yz[va]).to(dev)
    wt = wv = None
    if w_all is not None:
        wt = torch.tensor(w_all[tr].astype(np.float32)); wv = torch.tensor(w_all[va].astype(np.float32)).to(dev)
    bs = 8192
    if name == "nflow":
        net = make_flow_class(clamp, median_zero)(Xtr.shape[1]).to(dev)
    else:
        net = TM._cond_net()(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    T = GEN_T
    betas = acp = None
    if name == "ddpm":
        betas = torch.linspace(1e-4, 0.05, T, device=dev); acp = torch.cumprod(1 - betas, 0)

    def cell_loss(xb, yb):
        if name == "cfm":
            y0 = torch.randn_like(yb); t = torch.rand(len(yb), device=dev)
            ytt = (1 - t) * y0 + t * yb
            return (net(ytt, t, xb) - (yb - y0)) ** 2
        if name == "ddpm":
            ti = torch.randint(0, T, (len(yb),), device=dev); a = acp[ti]
            eps = torch.randn_like(yb)
            ytt = torch.sqrt(a) * yb + torch.sqrt(1 - a) * eps
            return (net(ytt, ti.float() / T, xb) - eps) ** 2
        return -net.log_prob(yb, xb)

    def loss_fn(xb, yb, wb=None):
        if wb is None:
            if name == "nflow":
                return -net.log_prob(yb, xb).mean()                   # tab_models 와 같은 식
            return cell_loss(xb, yb).mean()
        return (cell_loss(xb, yb) * wb).sum() / wb.sum()

    best, state, p = 1e9, None, 0
    pat, min_ep = 12, 15
    ep_run, n_nonfinite = 0, 0
    for ep in range(int(epochs)):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]; xb, yb = Xt[b].to(dev), yt[b].to(dev)
            wb = wt[b].to(dev) if wt is not None else None
            opt.zero_grad(); loss = loss_fn(xb, yb, wb)
            if not torch.isfinite(loss):
                n_nonfinite += 1
                break
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        ep_run = ep + 1
        net.eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vs = []
            for rep in range(3):
                torch.manual_seed(seed + 1000 + rep)
                if wv is None:
                    vs.append(np.mean([float(loss_fn(Xv[k:k + bs], yv[k:k + bs])) for k in range(0, len(Xv), bs)]))
                else:
                    num = den = 0.0
                    for k in range(0, len(Xv), bs):
                        wb_ = wv[k:k + bs]
                        num += float((cell_loss(Xv[k:k + bs], yv[k:k + bs]) * wb_).sum()); den += float(wb_.sum())
                    vs.append(num / den if den > 0 else float("nan"))
            v = float(np.mean(vs))
        if np.isfinite(v) and v < best - 1e-4:
            best, state, p = v, {k2: t_.cpu().clone() for k2, t_ in net.state_dict().items()}, 0
        else:
            p += 1
            if p >= pat and ep >= min_ep:
                break
    if state:
        net.load_state_dict(state)
    net.eval()
    legacy = (val == "random" and weights is None and float(clamp) == LEGACY_CLAMP and not median_zero)
    flag = vflag + (f";nonfinite_loss{n_nonfinite}" if n_nonfinite else "") + ("" if state else ";no_improvement")
    return GenHandle(name, net, ymu, ysd, Xtr.shape[1], clamp, median_zero, best if state else np.nan, ep_run,
                     int(tr.sum()), int(va.sum()), flag, legacy, dev, n_nonfinite)


def gen_samples(h: GenHandle, X, S=S_GEN, seed=0, chunk=None) -> np.ndarray:
    """표본 (S, n), 원 단위. torch.manual_seed(seed + 2000) 한 번 뒤 셀 묶음(chunk)마다 추출한다.

    chunk 기본값은 legacy 이면 4,096(tab_models 와 같다), 아니면 max(64, 2^20 // S). cfm 은 Euler 20단계, ddpm 은 역과정 200단계,
    nflow 는 net.sample. legacy 핸들과 같은 seed 이면 tab_models.fit_predict 의 samples 와 같은 값이다.
    """
    import torch
    X = np.asarray(X).astype(np.float32)
    S = int(S)
    if chunk is None:
        chunk = 4096 if h.legacy else max(64, (2 ** 20) // max(S, 1))
    dev, net, T = h.device, h.net, GEN_T
    net.eval()
    betas = acp = None
    if h.name == "ddpm":
        betas = torch.linspace(1e-4, 0.05, T, device=dev); acp = torch.cumprod(1 - betas, 0)
    outs = []
    with torch.no_grad():
        torch.manual_seed(int(seed) + 2000)
        for k in range(0, len(X), int(chunk)):
            xb = torch.tensor(X[k:k + int(chunk)]).to(dev); n = len(xb)
            if h.name == "nflow":
                y = net.sample(xb, S)
            else:
                xb_e = xb.unsqueeze(0).expand(S, -1, -1).reshape(S * n, -1)
                y = torch.randn(S * n, device=dev)
                if h.name == "cfm":
                    steps = 20
                    for i in range(steps):
                        t = torch.full((S * n,), i / steps, device=dev)
                        y = y + net(y, t, xb_e) / steps
                else:
                    for ti in reversed(range(T)):
                        a = acp[ti]; b_ = betas[ti]
                        tt = torch.full((S * n,), ti / T, device=dev)
                        eps = net(y, tt, xb_e)
                        y = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
                        if ti > 0:
                            y = y + torch.sqrt(b_) * torch.randn_like(y)
                y = y.reshape(S, n)
            outs.append(y.cpu().numpy())
    if not outs:
        return np.zeros((S, 0))
    return np.concatenate(outs, 1) * h.ysd + h.ymu


def _chunks(n, size):
    size = max(1, int(size))
    for k in range(0, n, size):
        yield k, min(n, k + size)


def nflow_quantiles(h: GenHandle, X, taus=TAUS) -> np.ndarray:
    """nflow 의 표본 없는 분위 (T, n), 원 단위. z0 = Φ⁻¹(taus) 를 spline 순변환에 넣고 ·ysd + ymu."""
    import torch
    from scipy.special import ndtri
    if h.name != "nflow":
        raise ValueError("nflow_quantiles 는 nflow 핸들에만 쓴다")
    X = np.asarray(X).astype(np.float32)
    z0 = ndtri(np.asarray(taus, float).ravel())
    out = np.empty((len(z0), len(X)))
    with torch.no_grad():
        for a, b in _chunks(len(X), 65536):
            xb = torch.tensor(X[a:b]).to(h.device)
            out[:, a:b] = h.net.quantile(xb, z0).cpu().numpy().astype(float)
    return out * h.ysd + h.ymu


def nflow_logpdf(h: GenHandle, X, E) -> np.ndarray:
    """nflow 조건부 로그 밀도 (n, G), 원 단위. E (n, G)는 셀마다의 평가 점이다. 표준화 뒤 log_prob − log(ysd).

    셀별 매개변수는 한 번만 계산해 G 번 반복한다. 한 번에 계산하는 행 수는 2^20 이하다.
    """
    import torch
    if h.name != "nflow":
        raise ValueError("nflow_logpdf 는 nflow 핸들에만 쓴다")
    X = np.asarray(X).astype(np.float32)
    E = np.asarray(E, float)
    if E.ndim != 2 or E.shape[0] != len(X):
        raise ValueError("E 는 (셀 수, 격자 수) 이어야 한다")
    n, G = E.shape
    out = np.empty((n, G))
    per = max(1, (2 ** 20) // max(G, 1))
    with torch.no_grad():
        for a, b in _chunks(n, per):
            xb = torch.tensor(X[a:b]).to(h.device)
            ps, mu, ls = h.net._affine(xb)

            def rep(t):
                return t.repeat_interleave(G, dim=0)
            ez = torch.tensor(((E[a:b] - h.ymu) / h.ysd).astype(np.float32).reshape(-1)).to(h.device)
            lp = h.net.log_prob_params(ez, [(rep(u), rep(v), rep(w)) for (u, v, w) in ps], rep(mu), rep(ls))
            out[a:b] = lp.reshape(b - a, G).cpu().numpy().astype(float)
    return out - np.log(h.ysd)


def nflow_cdf_table(h: GenHandle, X, eps_grid, n_tau=200) -> np.ndarray:
    """nflow 조건부 CDF 표 (n, G) float32. τ' = (m − 0.5)/n_tau(m = 1..n_tau)의 분위를 구해(정렬) 셀마다
    np.interp(eps_grid, q, τ', left=0, right=1). 격자 밖의 꼬리 질량(τ' 범위 밖)은 0 과 1 로 자른다."""
    tp = (np.arange(1, int(n_tau) + 1) - 0.5) / int(n_tau)
    q = np.sort(nflow_quantiles(h, X, tp), axis=0)
    eg = np.asarray(eps_grid, float)
    out = np.empty((q.shape[1], len(eg)), np.float32)
    for i in range(q.shape[1]):
        out[i] = np.interp(eg, q[:, i], tp, left=0.0, right=1.0)
    return out


def gen_quantiles(h: GenHandle, X, taus=TAUS, S=S_GEN, seed=0) -> np.ndarray:
    """분위 (T, n), 원 단위. nflow 는 nflow_quantiles, 그 외는 q_from_samples(gen_samples(h, X, S, seed), taus)."""
    if h.name == "nflow":
        return nflow_quantiles(h, X, taus)
    return q_from_samples(gen_samples(h, X, S=S, seed=seed), taus)


def cfm_quantiles_ode(h: GenHandle, X, taus=TAUS, steps=20) -> np.ndarray:
    """cfm 의 결정적 분위 사상 (T, n), 원 단위. y0 = Φ⁻¹(τ) 에서 출발한 Euler 적분(steps 단계). 교차 점검 진단용이며 채점에 쓰지 않는다."""
    import torch
    from scipy.special import ndtri
    if h.name != "cfm":
        raise ValueError("cfm_quantiles_ode 는 cfm 핸들에만 쓴다")
    X = np.asarray(X).astype(np.float32)
    z0 = ndtri(np.asarray(taus, float).ravel()).astype(np.float32)
    Tq = len(z0)
    out = np.empty((Tq, len(X)))
    with torch.no_grad():
        for a, b in _chunks(len(X), max(1, (2 ** 20) // max(Tq, 1))):
            xb = torch.tensor(X[a:b]).to(h.device); nc = b - a
            xb_e = xb.unsqueeze(0).expand(Tq, -1, -1).reshape(Tq * nc, -1)
            y = torch.tensor(z0).to(h.device)[:, None].expand(Tq, nc).reshape(Tq * nc).clone()
            for i in range(int(steps)):
                t = torch.full((Tq * nc,), i / int(steps), device=h.device)
                y = y + h.net(y, t, xb_e) / int(steps)
            out[:, a:b] = y.reshape(Tq, nc).cpu().numpy().astype(float)
    return out * h.ysd + h.ymu


def fit_failed(h, Qz_oos, mid=None):
    """적합 실패 판정 (bool, 사유). 실패 = val_loss 비유한, 분위에 비유한 값, 또는 |중앙값| > FAIL_LOGABS 인 셀 비율 ≥ FAIL_FRAC.

    h 는 GenHandle 또는 None(CatBoost 다분위처럼 검증 손실이 없는 경우). Qz_oos 는 로그 비 척도의 표본 밖 분위 (T, n).
    """
    if h is not None and not np.isfinite(float(getattr(h, "val_loss", np.nan))):
        return True, "val_loss 비유한"
    Q = np.asarray(Qz_oos, float)
    if Q.size == 0:
        return False, ""
    if not np.all(np.isfinite(Q)):
        return True, "분위 비유한"
    if Q.ndim == 1:
        Q = Q[:, None]
    frac = float(np.mean(np.abs(Q[_mid_index(Q.shape[0], mid)]) > FAIL_LOGABS))
    if frac >= FAIL_FRAC:
        return True, f"중앙값 |로그| > {FAIL_LOGABS:g} 셀 비율 {frac:.4f}"
    return False, ""


def cb_multiquantile(Xtr, ytr, X_list, seed, weights=None, threads=4, alphas=Q5, iterations=200, depth=3, lr=0.05, l2=3.0):
    """CatBoost 다분위 적합과 예측. 반환 (예측 목록[(n_j, 5)], info).

    손실 'MultiQuantile:alpha=0.05,0.1,0.5,0.9,0.95'(allow_writing_files=False, verbose=0, thread_count=threads, predict 에도
    thread_count 명시). 이 손실이 동작하지 않으면 분위마다 'Quantile:alpha=..' 로 따로 적합하고 info['flag'] 에 적는다.
    예측은 열 방향 정렬로 재배열한다. info = 재배열 전 교차 통계(전체 합산의 crossing_stats 키)와 flag, per_set(입력별 통계).
    """
    from catboost import CatBoostRegressor
    al = [float(a) for a in alphas]
    astr = ",".join(f"{a:g}" for a in al)
    common = dict(iterations=int(iterations), depth=int(depth), learning_rate=float(lr), l2_leaf_reg=float(l2), random_seed=int(seed),
                  verbose=0, allow_writing_files=False, thread_count=int(threads))
    Xtr = np.asarray(Xtr, float); ytr = np.asarray(ytr, float)
    w = None if weights is None else np.asarray(weights, float)
    Xs = [np.asarray(X, float) for X in X_list]
    flag = "multiquantile"
    try:
        m = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={astr}", **common)
        m.fit(Xtr, ytr, sample_weight=w)
        raw = [np.asarray(m.predict(X, thread_count=int(threads)), float).reshape(len(X), -1) for X in Xs]
        if any(r.shape[1] != len(al) for r in raw):
            raise ValueError("MultiQuantile 예측의 열 수가 분위 수와 다르다")
    except Exception as e:                                               # noqa: BLE001
        if is_env_error(e):                                              # 환경·자원 오류는 손실 방식 대체로 숨기지 않는다(개정 1)
            raise
        flag = f"per_quantile({type(e).__name__})"
        cols = [[] for _ in Xs]
        for a in al:
            m = CatBoostRegressor(loss_function=f"Quantile:alpha={a:g}", **common)
            m.fit(Xtr, ytr, sample_weight=w)
            for j, X in enumerate(Xs):
                cols[j].append(np.asarray(m.predict(X, thread_count=int(threads)), float))
        raw = [np.stack(c, 1) if c else np.zeros((len(X), 0)) for c, X in zip(cols, Xs)]
    per = [crossing_stats(r.T) for r in raw]
    nc = sum(p["n_cells"] for p in per); cc = sum(p["n_cross_cells"] for p in per)
    npair = sum(p["n_pairs"] for p in per); cp = sum(p["n_cross_pairs"] for p in per)
    info = dict(frac_cells=cc / nc if nc else np.nan, frac_pairs=cp / npair if npair else np.nan,
                max_drop=max([p["max_drop"] for p in per] or [0.0]), n_cells=nc, n_cross_cells=cc, n_pairs=npair, n_cross_pairs=cp,
                flag=flag, per_set=per)
    return [np.sort(r, axis=1) for r in raw], info


# ================================================================ 6. 점수(numpy 만 쓴다)
def alpha_suffix(alpha) -> str:
    """α → 지표 접미사(0.1 → '10', 0.2 → '20')."""
    return f"{int(round(float(alpha) * 100)):02d}"


def pinball(y, q, tau):
    """ρ_τ(y − q) = (y − q)·(τ − 1{y < q}). 배열은 방송 규칙을 따른다."""
    y = np.asarray(y, float); q = np.asarray(q, float); tau = np.asarray(tau, float)
    with np.errstate(invalid="ignore"):
        return (y - q) * (tau - (y < q))


def crps_grid(y, Q) -> np.ndarray:
    """CRPS 격자 근사 (2/99)·Σ_j ρ_τj(y − Q_j). y (n,), Q (99, n). 1 % 밖과 99 % 밖의 꼬리는 들어가지 않는다."""
    y = np.asarray(y, float); Q = np.asarray(Q, float)
    return (2.0 / len(TAUS)) * pinball(y[None, :], Q, TAUS[:, None]).sum(axis=0)


def interval_parts(y, lo, hi, alpha) -> dict:
    """구간 점수 IS_α = (U − L) + (2/α)(L − y)₊ + (2/α)(y − U)₊ 와 분해. 반환 dict(total, width, under, over, inf).

    끝점이 비유한이면 total = width = inf, inf 표지 1 이다.
    """
    y = np.asarray(y, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float); a = float(alpha)
    with np.errstate(invalid="ignore", over="ignore"):
        fin = np.isfinite(lo) & np.isfinite(hi)
        width = hi - lo
        under = (2.0 / a) * np.maximum(lo - y, 0.0)
        over = (2.0 / a) * np.maximum(y - hi, 0.0)
        total = width + under + over
    total = np.where(fin, total, np.inf); width = np.where(fin, width, np.inf)
    return dict(total=total, width=width, under=under, over=over, inf=(~fin).astype(float))


def pit_grid(y, Q) -> np.ndarray:
    """PIT: 셀마다 np.interp(y, Q[:, i], TAUS, left=0.005, right=0.995) 와 같은 값(벡터화). Q (99, n)는 열마다 정렬된 격자다.

    np.interp 의 규칙을 그대로 옮겼다. j = Q_j ≤ y 인 마지막 번호. y < Q_0 이면 0.005, y > Q_98 이면 0.995, j = 98(y = Q_98)이면 0.99,
    그 밖은 τ_j + (τ_(j+1) − τ_j)/(Q_(j+1) − Q_j)·(y − Q_j). 동률이 있는 격자에서도 np.interp 와 같은 구간을 고른다.
    y 또는 Q 열이 비유한이면 NaN.
    """
    y = np.asarray(y, float); Q = np.asarray(Q, float)
    if Q.ndim != 2 or Q.shape[0] != len(TAUS) or Q.shape[1] != len(y):
        raise ValueError(f"Q 의 모양 {Q.shape} 가 (99, {len(y)}) 가 아니다")
    n = len(y)
    out = np.full(n, np.nan)
    ok = np.isfinite(y) & np.all(np.isfinite(Q), axis=0)
    if not ok.any():
        return out
    Qo, yo = Q[:, ok], y[ok]
    T = Qo.shape[0]
    j = (Qo <= yo[None, :]).sum(axis=0) - 1                         # Q_j ≤ y 인 마지막 번호(정렬된 열). −1 이면 y < Q_0
    r = np.empty(len(yo))
    left = j < 0
    right = yo > Qo[-1]
    end = ~right & (j == T - 1)
    r[left] = 0.005
    r[right] = 0.995
    r[end] = TAUS[-1]
    mid = ~(left | right | end)
    if mid.any():
        cols = np.where(mid)[0]; jj = j[mid]
        q0, q1 = Qo[jj, cols], Qo[jj + 1, cols]                     # q0 ≤ y < q1 이므로 q1 − q0 > 0
        slope = (TAUS[jj + 1] - TAUS[jj]) / (q1 - q0)
        r[mid] = slope * (yo[mid] - q0) + TAUS[jj]
    out[ok] = r
    return out


def pit_hist(pit, bins=PIT_BINS) -> np.ndarray:
    """PIT 히스토그램 개수 (bins,). 구간 [0, 1] 등분. 비유한 값은 뺀다."""
    p = np.asarray(pit, float); p = p[np.isfinite(p)]
    return np.histogram(p, bins=np.linspace(0.0, 1.0, int(bins) + 1))[0].astype(np.int64)


def _interval_metrics(y, lo, hi, alpha) -> dict:
    sfx = alpha_suffix(alpha)
    p = interval_parts(y, lo, hi, alpha)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = ((lo <= y) & (y <= hi)).astype(float)
        lhw = (np.log(hi) - np.log(lo)) / 2.0
    return {f"is{sfx}": p["total"], f"is{sfx}_w": p["width"], f"is{sfx}_u": p["under"], f"is{sfx}_o": p["over"],
            f"cov{sfx}": cov, f"wid{sfx}": p["width"], f"lhw{sfx}": lhw, f"inf{sfx}": p["inf"]}


def score_cells(y, Q, alphas=ALPHAS) -> dict:
    """셀별 점수. y (n,), Q (99, n) cm 격자(채점 전에 호출자가 rearrange 한다). 반환 dict(METRICS 이름 → (n,), 'pit', 'valid').

    se = (Q[49] − y)², ae = |Q[49] − y|, bias = Q[49] − y, lhw = (log hi − log lo)/2. y 가 비유한이거나 Q 열에 NaN 이 있는 셀의
    지표는 NaN 이다. 'valid' 는 y 유한 표지(ScoreStore.add 가 셀 수를 센다).
    """
    y = np.asarray(y, float); Q = np.asarray(Q, float)
    if Q.ndim != 2 or Q.shape[0] != len(TAUS) or Q.shape[1] != len(y):
        raise ValueError(f"Q 의 모양 {Q.shape} 가 (99, {len(y)}) 가 아니다")
    ok = np.isfinite(y) & ~np.isnan(Q).any(axis=0)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for a in alphas:
            lo, hi = interval_of(Q, a)
            out.update(_interval_metrics(y, lo, hi, a))
        out["crps"] = crps_grid(y, Q)
        out["crps_log"] = crps_grid(np.log(y), np.log(Q))
        for t, nm in ((0.05, "pb05"), (0.50, "pb50"), (0.95, "pb95")):
            out[nm] = pinball(y, Q[grid_index(t)], t)
        med = Q[49]
        out["se"] = (med - y) ** 2; out["ae"] = np.abs(med - y); out["bias"] = med - y
        out["pit"] = pit_grid(y, Q)
    for k in list(out):
        out[k] = np.where(ok, out[k], np.nan)
    out["valid"] = np.isfinite(y)
    return out


def score_interval_only(y, lo10, hi10, lo20, hi20, med) -> dict:
    """구간만 있는 방법(iii+c 등)의 점수. 구간 지표와 se, ae, bias. crps, crps_log, pb*, pit 는 NaN. 'valid' 포함."""
    y = np.asarray(y, float); med = np.asarray(med, float)
    arrs = [np.asarray(v, float) for v in (lo10, hi10, lo20, hi20)]
    ok = np.isfinite(y) & np.isfinite(med) & ~np.any([np.isnan(v) for v in arrs], axis=0)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out.update(_interval_metrics(y, arrs[0], arrs[1], 0.10))
        out.update(_interval_metrics(y, arrs[2], arrs[3], 0.20))
        out["se"] = (med - y) ** 2; out["ae"] = np.abs(med - y); out["bias"] = med - y
    for k in ("crps", "crps_log", "pb05", "pb50", "pb95", "pit"):
        out[k] = np.full(len(y), np.nan)
    for k in list(out):
        out[k] = np.where(ok, out[k], np.nan)
    out["valid"] = np.isfinite(y)
    return out


# ================================================================ 7. ScoreStore(채점 블록별 점수 합)
def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class ScoreStore:
    """작업 단위(대상, 분할) 하나의 채점 블록별 점수 합 저장소(형식 lgu_blockscore_v1).

    block_ids: 채점 셀의 블록 식별자(셀 순서). np.unique(문자열)로 codes 를 만든다. block_cols: 셀별 전역 블록 열 번호(BlockIndex.cols).
    key = (method, variant, n, draw, seed). add(key, scores) 는 지표마다 y 유한 셀의 블록 합(inf, NaN 전파)과 y 유한 셀 수를 저장한다.
    같은 key 를 다시 넣으면 덮어쓴다.
    """

    def __init__(self, target, split, block_ids, block_cols=None, meta=None, metrics=METRICS):
        self.target = str(target); self.split = int(split)
        self.metrics = list(metrics); self._mi = {m: i for i, m in enumerate(self.metrics)}
        ub, codes = np.unique(np.asarray(block_ids).astype(str), return_inverse=True)
        self.blocks = ub; self.codes = codes.astype(np.int64).ravel(); self.nb = int(len(ub))
        self.ncell = np.bincount(self.codes, minlength=self.nb).astype(np.int64)
        if block_cols is None:
            self.cols = np.full(self.nb, -1, np.int32)
        else:
            bc = np.asarray(block_cols, np.int64).ravel()
            if bc.shape != self.codes.shape:
                raise ValueError("block_cols 는 block_ids 와 같은 길이의 셀 단위 배열이어야 한다")
            cols = np.full(self.nb, -1, np.int64); cols[self.codes] = bc
            if not np.array_equal(cols[self.codes], bc):
                raise ValueError("같은 블록의 셀에 다른 전역 열 번호가 있다")
            self.cols = cols.astype(np.int32)
        self.meta = dict(meta or {})
        self.keys: list = []; self._idx: dict = {}; self._sum: list = []; self._cnt: list = []

    # -- 입력
    def add(self, key, scores: dict, valid=None):
        if self.codes is None:
            raise RuntimeError("파일에서 적재한 ScoreStore 는 셀 순서가 없다. add_sums() 를 쓴다")
        n = len(self.codes)
        v = scores.get("valid") if valid is None else valid
        v = np.ones(n, bool) if v is None else np.asarray(v, bool)
        if v.shape != (n,):
            raise ValueError("valid 의 길이가 저장소의 셀 수와 다르다")
        c = self.codes[v]
        sums = np.full((len(self.metrics), self.nb), np.nan)
        for j, m in enumerate(self.metrics):
            val = scores.get(m)
            if val is None:
                continue
            val = np.asarray(val, float)
            if val.shape != (n,):
                raise ValueError(f"지표 {m} 의 길이가 저장소의 셀 수와 다르다")
            with np.errstate(invalid="ignore"):
                sums[j] = np.bincount(c, weights=val[v], minlength=self.nb)
        cnt = np.bincount(c, minlength=self.nb).astype(np.int64)
        self.add_sums(key, sums, cnt)

    def add_sums(self, key, sums, cnt):
        key = norm_key(key)
        sums = np.asarray(sums, float); cnt = np.asarray(cnt, np.int64)
        if sums.shape != (len(self.metrics), self.nb) or cnt.shape != (self.nb,):
            raise ValueError("합 또는 셀 수의 모양이 저장소와 다르다")
        if key in self._idx:
            i = self._idx[key]; self._sum[i] = sums; self._cnt[i] = cnt
        else:
            self._idx[key] = len(self.keys); self.keys.append(key); self._sum.append(sums); self._cnt.append(cnt)

    # -- 조회
    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return norm_key(key) in self._idx

    def get(self, key):
        """(합 (M, nb), 셀 수 (nb,))."""
        i = self._idx[norm_key(key)]
        return self._sum[i], self._cnt[i]

    def select(self, key_fn) -> list:
        """술어 함수, 키 목록, 또는 단일 키로 키를 고른다(h4_common.select_keys 와 같은 규칙)."""
        return select_keys(self.keys, key_fn)

    def matrices(self, keys, metric):
        """(S (K, nb), C (K, nb)). 지표가 저장소에 없으면 S 는 NaN."""
        idx = [self._idx[norm_key(k)] for k in keys]
        if not idx:
            return np.zeros((0, self.nb)), np.zeros((0, self.nb), np.int64)
        C = np.stack([self._cnt[i] for i in idx])
        if metric not in self._mi:
            return np.full((len(idx), self.nb), np.nan), C
        j = self._mi[metric]
        return np.stack([self._sum[i][j] for i in idx]), C

    def mean(self, key, metric, weight="cell") -> float:
        """한 키의 지표 평균. weight 'cell'(셀 가중) 또는 'beq'(블록 등가중). se 는 평균 제곱 오차다(RMSE 는 rmse())."""
        S, C = self.matrices([key], metric)
        one = np.ones((1, self.nb))
        f = agg_cell if weight == "cell" else agg_beq
        return float(f(S, C, one)[0, 0])

    def rmse(self, key, weight="cell") -> float:
        S, C = self.matrices([key], "se")
        one = np.ones((1, self.nb))
        f = agg_rmse_cell if weight == "cell" else agg_rmse_beq
        return float(f(S, C, one)[0, 0])

    def merge(self, other: "ScoreStore"):
        if (self.target, self.split) != (other.target, other.split):
            raise ValueError("다른 작업 단위는 병합할 수 없다")
        if not (np.array_equal(self.blocks, other.blocks) and np.array_equal(self.ncell, other.ncell)):
            raise ValueError("채점 블록 집합이 다르다")
        if self.metrics != other.metrics:
            raise ValueError("지표 목록이 다르다")
        if np.all(self.cols < 0) and np.any(other.cols >= 0):
            self.cols = other.cols.copy()
        elif np.any(other.cols >= 0) and not np.array_equal(self.cols, other.cols):
            raise ValueError("전역 블록 열 번호가 다르다")
        for k in other.keys:
            s, c = other.get(k); self.add_sums(k, s, c)
        self.meta.update(other.meta)
        return self

    @classmethod
    def _from_arrays(cls, target, split, blocks, ncell, cols, keys, sums, cnt, meta, metrics):
        st = cls.__new__(cls)
        st.target = str(target); st.split = int(split)
        st.metrics = list(metrics); st._mi = {m: i for i, m in enumerate(st.metrics)}
        st.blocks = np.asarray(blocks).astype(str); st.nb = int(len(st.blocks))
        st.ncell = np.asarray(ncell, np.int64); st.cols = np.asarray(cols, np.int32); st.codes = None
        st.meta = dict(meta or {}); st.keys = []; st._idx = {}; st._sum = []; st._cnt = []
        for k, s, c in zip(keys, sums, cnt):
            st.add_sums(k, s, c)
        return st


def save_score_stores(stores, path) -> Path:
    """여러 저장소를 한 npz 로 저장한다(pickle 없음, 원자적 교체).

    meta = JSON {format, metrics, units:[{target, split, nb, n_keys, meta}]}, u{i}_keys = 키 JSON 문자열 (K,),
    u{i}_sum float64 (K, M, nb), u{i}_cnt int32 (K, nb), u{i}_blocks 문자열 (nb,), u{i}_cols int32 (nb,), u{i}_ncell int64 (nb,).
    """
    stores = list(stores); arrs = {}; units = []
    metrics = stores[0].metrics if stores else list(METRICS)
    for i, st in enumerate(stores):
        if st.metrics != metrics:
            raise ValueError("한 파일의 저장소는 지표 목록이 같아야 한다")
        K = len(st.keys)
        arrs[f"u{i}_keys"] = np.array([json.dumps(list(k), ensure_ascii=False, default=_py) for k in st.keys], dtype=str)
        arrs[f"u{i}_sum"] = (np.stack(st._sum) if K else np.zeros((0, len(metrics), st.nb))).astype(np.float64)
        arrs[f"u{i}_cnt"] = (np.stack(st._cnt) if K else np.zeros((0, st.nb))).astype(np.int32)
        arrs[f"u{i}_blocks"] = np.asarray(st.blocks).astype(str)
        arrs[f"u{i}_cols"] = np.asarray(st.cols, np.int32)
        arrs[f"u{i}_ncell"] = np.asarray(st.ncell, np.int64)
        units.append(dict(target=st.target, split=st.split, nb=st.nb, n_keys=K, meta=st.meta))
    arrs["meta"] = np.array(json.dumps(dict(format=FORMAT, metrics=list(metrics), units=units), ensure_ascii=False, default=_py))
    return atomic_npz(path, **arrs)


def load_score_stores(paths) -> dict:
    """npz 하나 또는 여러 개를 읽어 {(target, split): ScoreStore} 로 병합한다(같은 단위는 키 합집합)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: dict = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError(f"{p}: 형식 {meta.get('format')} 은 {FORMAT} 이 아니다")
            for i, u in enumerate(meta["units"]):
                keys = [tuple(json.loads(s)) for s in z[f"u{i}_keys"]]
                st = ScoreStore._from_arrays(u["target"], u["split"], z[f"u{i}_blocks"], z[f"u{i}_ncell"], z[f"u{i}_cols"], keys,
                                             z[f"u{i}_sum"], z[f"u{i}_cnt"], u.get("meta"), meta["metrics"])
                k = (st.target, st.split)
                out[k] = out[k].merge(st) if k in out else st
    return out


def stores_for_target(stores: dict, target) -> dict:
    """{(target, split): store} → 한 대상의 {split: store}(분할 순서)."""
    return {sp: st for (t, sp), st in sorted(stores.items(), key=lambda kv: kv[0][1]) if t == str(target)}


# ================================================================ 8. 계층 보정과 의사 대상 모사
def wquantile(scores, w, level, w_test=0.0) -> float:
    """가중 경험분포(+∞ 질량 w_test 포함 정규화)의 level 분위. h37.wquantile 과 같은 식(안정 정렬, searchsorted(level − 1e-12)).
    누적 가중이 level 에 못 미치면 inf."""
    o = np.argsort(scores, kind="stable"); s, ww = np.asarray(scores, float)[o], np.asarray(w, float)[o]
    cum = np.cumsum(ww) / (ww.sum() + w_test)
    k = int(np.searchsorted(cum, level - 1e-12))
    return float(s[k]) if k < len(s) else np.inf


def _hier_w(inv, m, nreg):
    """지역 등가중 셀 가중(원소별 계산이므로 셀 순서와 무관하게 같은 값). 반환 (w, K)."""
    m = np.asarray(m, float)
    tot = np.bincount(inv, weights=m, minlength=nreg)
    K = int((tot > 0).sum())
    w = np.zeros(len(m))
    if K == 0:
        return w, 0
    t = tot[inv]
    pos = t > 0
    w[pos] = m[pos] / (K * t[pos])
    return w, K


def hier_cell_weights(groups, mult=None) -> np.ndarray:
    """w_i = m_i / (K·Σ_(i 의 지역) m). mult 가 None 이면 m = 1. 총 다중도가 0 인 지역은 빼고 K 를 줄인다."""
    g = np.asarray(groups)
    ug, inv = np.unique(g, return_inverse=True)
    m = np.ones(len(g)) if mult is None else np.asarray(mult, float)
    return _hier_w(inv.ravel(), m, len(ug))[0]


def _wq_sorted(s_o, w_o, levels):
    cum = np.cumsum(w_o) / (w_o.sum() + 0.0)
    k = np.searchsorted(cum, np.asarray(levels, float) - 1e-12)
    out = np.full(len(k), np.inf)
    fin = k < len(s_o)
    out[fin] = s_o[k[fin]]
    return out


def hier_quantiles(scores, groups, levels, mult=None) -> np.ndarray:
    """지역 등가중 합동 CDF 의 수준별 분위 (L,). wquantile(scores, hier_cell_weights(groups, mult), level) 과 같다.
    NaN 점수는 뺀다. 다중도가 모두 0 이면 NaN."""
    s = np.asarray(scores, float); g = np.asarray(groups)
    lv = np.atleast_1d(np.asarray(levels, float))
    keep = ~np.isnan(s)
    s, g = s[keep], g[keep]
    m = np.ones(len(s)) if mult is None else np.asarray(mult, float)[keep]
    if len(s) == 0:
        return np.full(len(lv), np.nan)
    ug, inv = np.unique(g, return_inverse=True)
    w, K = _hier_w(inv.ravel(), m, len(ug))
    if K == 0:
        return np.full(len(lv), np.nan)
    o = np.argsort(s, kind="stable")
    return _wq_sorted(s[o], w[o], lv)


def pooled_quantile(scores, alpha) -> float:
    """셀 교환성 유한표본 분위: ceil((N + 1)(1 − α))번째 순서통계. N 을 넘으면 inf. NaN 점수는 뺀다. 점수가 없으면 NaN."""
    s = np.asarray(scores, float); s = np.sort(s[~np.isnan(s)])
    N = len(s)
    if N == 0:
        return np.nan
    k = int(math.ceil((N + 1) * (1.0 - float(alpha)) - 1e-9))
    return float(s[max(k, 1) - 1]) if k <= N else np.inf


def hier_quantiles_boot(scores, groups, cols, W, levels, chunk=200) -> np.ndarray:
    """전역 블록 재표집의 보정 분위 (nboot, L). 재표집 r 의 셀 다중도는 W[r, cols[i]] 다.

    점수를 한 번 정렬하고 재표집마다 지역 등가중 누적 가중으로 분위를 구한다. 값은 hier_quantiles(scores, groups, levels,
    mult=W[r, cols]) 를 재표집마다 부른 것과 같다. 다중도가 모두 0 인 재표집은 NaN.
    """
    s = np.asarray(scores, float); g = np.asarray(groups); c = np.asarray(cols, np.int64)
    lv = np.atleast_1d(np.asarray(levels, float))
    W = np.asarray(W)
    R = W.shape[0]
    out = np.full((R, len(lv)), np.nan)
    keep = ~np.isnan(s)
    s, g, c = s[keep], g[keep], c[keep]
    if len(s) == 0:
        return out
    ug, inv = np.unique(g, return_inverse=True)
    o = np.argsort(s, kind="stable")
    s_o, c_o, inv_o = s[o], c[o], inv.ravel()[o]
    for r0 in range(0, R, max(1, int(chunk))):
        M = W[r0:r0 + int(chunk)][:, c_o].astype(float)
        for i in range(M.shape[0]):
            w_o, K = _hier_w(inv_o, M[i], len(ug))
            if K == 0:
                continue
            out[r0 + i] = _wq_sorted(s_o, w_o, lv)
    return out


def ls_E(y, s) -> float:
    """최소제곱 계수 Σ s·y / Σ s²(y, s 유한, s > 0). h40.ls_E 와 같은 식. 셀이 없으면 NaN."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    with np.errstate(invalid="ignore"):
        m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def center_coef(y_lab, s_lab, E0, kappa=KAPPA) -> float:
    """E_n = (n·E_ls + κ·E0)/(n + κ), n = 라벨 수(len). n = 0 또는 E_ls 비유한이면 E0(h40 의 coefs 와 같은 규칙)."""
    n = len(np.asarray(y_lab))
    if n == 0:
        return float(E0)
    E_ls = ls_E(y_lab, s_lab)
    if not np.isfinite(E_ls):
        return float(E0)
    return float((n * E_ls + float(kappa) * float(E0)) / (n + float(kappa)))


def emulation_plan(df, src_idx, groups, n_grid, target, mode, emu_splits=EMU_SPLITS, emu_draws=EMU_DRAWS) -> list:
    """의사 대상 모사 계획(계획서 2.3절). 기록 목록.

    기록 = dict(region, n, e, d, cal_idx, lab_idx, E0_mk, E_center, n_A, nb_cal). df 는 y, s, macro, block 열이 있는 전체 표,
    색인은 df 의 행 위치다. E0_mk = src_idx 에서 지역 k 를 뺀 셀의 ls_E. 지역 k 의 셀은 src_idx 가운데 y, s 유한이고 s > 0 인 셀이다.
    n = 0: cal_idx = 지역 k 의 셀 전체, e = d = 0, E_center = E0_mk.
    n > 0: e 마다 half_split_blocks(df, 지역 k 의 셀, e)로 A_k, B_k. B_k 의 블록이 2개 미만이거나 len(A_k) ≤ n 이면 건너뛴다.
    d 마다 sel = sort(RandomState(seed_of('lgu-emu', target, mode, k, e, n, d)).choice(len(A_k), n, replace=False)),
    lab_idx = A_k[sel], cal_idx = B_k, E_center = center_coef(y[lab], s[lab], E0_mk). 음수 n(전량)은 모사하지 않는다.
    """
    y = df["y"].values.astype(float); s = df["s"].values.astype(float)
    mac = np.asarray(df["macro"].values).astype(str); blk = df["block"].values
    src = np.asarray(src_idx, np.int64)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(y[src]) & np.isfinite(s[src]) & (s[src] > 0)
    recs = []
    for k in groups:
        in_k = mac[src] == str(k)
        rest = src[~in_k]
        E0_mk = ls_E(y[rest], s[rest])
        idx_k = src[in_k & ok]
        for n in n_grid:
            n = int(n)
            if n < 0:
                continue
            if n == 0:
                recs.append(dict(region=str(k), n=0, e=0, d=0, cal_idx=idx_k.copy(), lab_idx=np.zeros(0, np.int64), E0_mk=E0_mk,
                                 E_center=E0_mk, n_A=int(len(idx_k)), nb_cal=int(len(np.unique(blk[idx_k])))))
                continue
            for e in emu_splits:
                if len(idx_k) == 0:
                    continue
                A_k, B_k = half_split_blocks(df, idx_k, int(e))
                nb_cal = int(len(np.unique(blk[B_k])))
                if nb_cal < 2 or len(A_k) <= n:
                    continue
                for d in range(int(emu_draws)):
                    rs = np.random.RandomState(seed_of("lgu-emu", target, mode, k, e, n, d))
                    sel = np.sort(rs.choice(len(A_k), n, replace=False))
                    lab = A_k[sel]
                    recs.append(dict(region=str(k), n=n, e=int(e), d=int(d), cal_idx=np.asarray(B_k, np.int64),
                                     lab_idx=np.asarray(lab, np.int64), E0_mk=E0_mk, E_center=center_coef(y[lab], s[lab], E0_mk),
                                     n_A=int(len(A_k)), nb_cal=nb_cal))
    return recs


# ================================================================ 9. 부트스트랩과 판정
class BlockIndex:
    """전역 블록 색인. 키 'macro|block' 을 사전순으로 정렬해 열 번호를 준다.

    속성: keys, region_of_col, block_of_col, n_cells(열별 셀 수), n_cols, regions(사전순). cols(idx) → 셀의 열 번호.
    """

    def __init__(self, df):
        mac = np.asarray(df["macro"].values).astype(str)
        blk = np.asarray(df["block"].values).astype(str)
        key = np.char.add(np.char.add(mac, "|"), blk)
        self.keys, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        self._cell_col = inv.astype(np.int64).ravel()
        parts = [str(k).split("|", 1) for k in self.keys]
        self.region_of_col = np.array([p[0] for p in parts]).astype(str)
        self.block_of_col = np.array([p[1] for p in parts]).astype(str)
        self.n_cells = cnt.astype(np.int64)
        self.n_cols = int(len(self.keys))
        self.regions = sorted(set(self.region_of_col.tolist()))

    def cols(self, idx) -> np.ndarray:
        return self._cell_col[np.asarray(idx, np.int64)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(dict(col=np.arange(self.n_cols), macro=self.region_of_col, block=self.block_of_col, n_cells=self.n_cells))


def global_mult(bi: BlockIndex, nboot=NBOOT, seed=0) -> np.ndarray:
    """전역 블록 재표집의 다중도 (nboot, NC) uint8(최대값이 255 를 넘으면 uint16).

    지역 r(사전순)마다 pick = RandomState(seed_of('lgu-boot', seed, r)).randint(0, nb_r, size=(nboot, nb_r)) 의 횟수를
    그 지역의 열(열 번호 순서)에 둔다. 지역마다 행 합은 그 지역의 블록 수다.
    """
    nboot = int(nboot)
    W = np.zeros((nboot, bi.n_cols), np.int32)
    for r in bi.regions:
        cr = np.where(bi.region_of_col == r)[0]
        nb_r = len(cr)
        if nb_r == 0:
            continue
        pick = np.random.RandomState(seed_of("lgu-boot", seed, r)).randint(0, nb_r, size=(nboot, nb_r))
        flat = (pick + (np.arange(nboot)[:, None] * nb_r)).ravel()
        W[:, cr] = np.bincount(flat, minlength=nboot * nb_r).reshape(nboot, nb_r)
    return W.astype(np.uint8 if W.max(initial=0) <= 255 else np.uint16)


def _as_w2(Wt):
    W = np.asarray(Wt, float)
    return (W[None, :], True) if W.ndim == 1 else (W, False)


def _wsum(S, W):
    """(K, nb) 합과 (R, nb) 다중도의 곱 (K, R). 뽑힌 블록에 inf 가 있으면 inf, NaN 이 있으면 NaN(0·inf 를 NaN 으로 만들지 않는다)."""
    S = np.asarray(S, float)
    fin = np.isfinite(S)
    out = np.where(fin, S, 0.0) @ W.T
    if not fin.all():
        pos = (W > 0).astype(float).T
        pinf = ((S == np.inf).astype(float) @ pos) > 0
        ninf = ((S == -np.inf).astype(float) @ pos) > 0
        nan = (np.isnan(S).astype(float) @ pos) > 0
        out = np.where(pinf, np.inf, out)
        out = np.where(ninf, -np.inf, out)
        out = np.where(nan | (pinf & ninf), np.nan, out)
    return out


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def agg_cell(S, C, Wt) -> np.ndarray:
    """셀 가중 평균 (K, R) = (S@Wt.T)/(C@Wt.T). Wt 가 1차원(nb,)이면 (K,). 분모 0 은 NaN."""
    W, one = _as_w2(Wt)
    out = _ratio(_wsum(S, W), np.asarray(C, float) @ W.T)
    return out[:, 0] if one else out


def agg_beq(S, C, Wt) -> np.ndarray:
    """블록 등가중 평균 (K, R) = ((S/C, C > 0 만)@Wt.T)/((C > 0)@Wt.T)."""
    W, one = _as_w2(Wt)
    C = np.asarray(C, float); S = np.asarray(S, float)
    has = C > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(has, S / np.where(has, C, 1.0), 0.0)
    out = _ratio(_wsum(M, W), has.astype(float) @ W.T)
    return out[:, 0] if one else out


def agg_rmse_cell(S, C, Wt) -> np.ndarray:
    """셀 가중 RMSE: sqrt(agg_cell(se 합))."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(agg_cell(S, C, Wt))


def agg_rmse_beq(S, C, Wt) -> np.ndarray:
    """블록 등가중 RMSE: 블록별 RMSE sqrt(S/C) 의 평균."""
    W, one = _as_w2(Wt)
    C = np.asarray(C, float); S = np.asarray(S, float)
    has = C > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(has, np.sqrt(S / np.where(has, C, 1.0)), 0.0)
    out = _ratio(_wsum(M, W), has.astype(float) @ W.T)
    return out[:, 0] if one else out


def ci95(dist):
    """(lo, hi, n_nan): nanpercentile 2.5, 97.5 와 NaN 개수. 유효 값이 없으면 (NaN, NaN, n)."""
    d = np.asarray(dist, float)
    nn = int(np.isnan(d).sum())
    if d.size == 0 or nn == d.size:
        return np.nan, np.nan, nn
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)), nn


def _stack_min(dists):
    ds = [np.asarray(d, float) for d in dists if d is not None]
    if not ds:
        return None
    L = min(len(d) for d in ds)
    return np.stack([d[:L] for d in ds])


def combine_same_index(dists) -> np.ndarray:
    """같은 재표집 번호끼리 nanmean(길이가 다르면 짧은 쪽에 맞춘다). 입력이 없으면 빈 배열."""
    M = _stack_min(dists)
    if M is None:
        return np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(M, axis=0)


def strat_mean(dists_by_region) -> np.ndarray:
    """같은 재표집 번호끼리 지역 등가중 평균. dict 또는 목록. 한 지역이라도 NaN 인 번호는 NaN 이다."""
    vals = list(dists_by_region.values()) if isinstance(dists_by_region, dict) else list(dists_by_region)
    M = _stack_min(vals)
    if M is None:
        return np.zeros(0)
    return M.mean(axis=0)


def _key_mean(M):
    """키 축(0) nanmean. 모든 키가 NaN 인 열은 경고 없이 NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(M, axis=0)


def _pair_groups(keys):
    """키를 짝지음 단위 (n, 추출)로 묶는다. 반환 {(n, draw): [키 위치]}. 5원소 키가 아니면 모든 키를 한 묶음으로 둔다."""
    g: dict = {}
    for i, k in enumerate(keys):
        gk = (k[2], k[3]) if isinstance(k, tuple) and len(k) >= 4 else ("all",)
        g.setdefault(gk, []).append(i)
    return g


def _paired_diff(VA, VB, gA, gB, common):
    """(K, R) 키별 값에서 짝지음 단위마다 A 와 B 의 seed·방법 평균(nanmean)을 구해 차를 내고, 단위 축 nanmean 한다.
    한 단위의 A 나 B 가 NaN 이면 그 재표집에서 그 단위의 차 전체가 빠진다(양쪽에서 같은 단위가 빠진다)."""
    D = np.stack([_key_mean(VA[gA[c]]) - _key_mean(VB[gB[c]]) for c in common])
    return _key_mean(D)


def one_stage_delta(stores_by_split: dict, keysA, keysB, metric, nboot=NBOOT, seed=0) -> dict:
    """1단 CI(LG 규칙): 분할마다 W = h4_common.boot_weights(nb, nboot, seed_of(seed, split)) 를 A·B 모든 키에 공유한다.

    짝지음(계획서 2.7절 '같은 분할, 추출, seed', 개정 1): 분할마다 A·B 의 키를 (n, 추출) 단위로 묶고 두 쪽에 모두 있는 단위만 쓴다.
    단위 안에서는 키(seed, 위약 순열 등)의 지표 평균을 nanmean 하고, 단위마다 A − B 를 구한 뒤 단위 축 nanmean 한다. 모든 단위가
    같은 수의 키를 가지면 키 평균의 차(LG 규칙, h4_common.boot_delta_blocks)와 같다. metric 이 'se' 이면 RMSE 로 계산한다.
    분할 분포는 같은 번호끼리 평균(np.mean)한다. keysA, keysB 는 술어, 키 목록 또는 단일 키다(ScoreStore.select).
    A 나 B 의 키가 없는 분할, 공통 단위가 없는 분할은 건너뛴다.
    반환 dict(delta, ci_lo, ci_hi, p_boot, delta_beq, ci_lo_beq, ci_hi_beq, dist, dist_beq, n_splits, n_blocks(분할별 목록),
    split_win, delta_split, delta_split_beq, n_keys_A, n_keys_B, n_nan, n_unpaired_A, n_unpaired_B, n_nan_keys_A, n_nan_keys_B,
    pair_ok). n_unpaired_* 는 한쪽에만 있어 뺀 (분할, n, 추출) 단위 수, n_nan_keys_* 는 한 블록이라도 지표 합이 NaN 인 키 수(계산 실패),
    pair_ok 는 둘 다 0 인지다. 쓸 수 있는 분할이 없으면 수치는 NaN, dist 는 None, pair_ok 는 False.
    """
    rmse = metric == "se"
    fc, fb = (agg_rmse_cell, agg_rmse_beq) if rmse else (agg_cell, agg_beq)
    d_split, db_split, dists, dists_b, nbs = {}, {}, [], [], []
    nkA = nkB = unA = unB = nnA = nnB = 0
    for sp, st in sorted(stores_by_split.items()):
        kA, kB = st.select(keysA), st.select(keysB)
        if not kA or not kB:
            continue
        gA, gB = _pair_groups(kA), _pair_groups(kB)
        common = sorted(set(gA) & set(gB), key=lambda c: json.dumps(c, default=str))
        unA += len(set(gA) - set(gB)); unB += len(set(gB) - set(gA))
        if not common:
            continue
        SA, CA = st.matrices(kA, metric); SB, CB = st.matrices(kB, metric)
        nnA += int(np.isnan(SA).any(axis=1).sum()); nnB += int(np.isnan(SB).any(axis=1).sum())
        one = np.ones((1, st.nb))
        d_split[sp] = float(_paired_diff(fc(SA, CA, one), fc(SB, CB, one), gA, gB, common)[0])
        db_split[sp] = float(_paired_diff(fb(SA, CA, one), fb(SB, CB, one), gA, gB, common)[0])
        nbs.append(int(st.nb)); nkA += len(kA); nkB += len(kB)
        if int(nboot) > 0:
            W = H4.boot_weights(st.nb, int(nboot), seed_of(seed, sp))
            dists.append(_paired_diff(fc(SA, CA, W), fc(SB, CB, W), gA, gB, common))
            dists_b.append(_paired_diff(fb(SA, CA, W), fb(SB, CB, W), gA, gB, common))
    nanres = dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_boot=np.nan, delta_beq=np.nan, ci_lo_beq=np.nan, ci_hi_beq=np.nan,
                  dist=None, dist_beq=None, n_splits=0, n_blocks=[], split_win=np.nan, delta_split={}, delta_split_beq={},
                  n_keys_A=0, n_keys_B=0, n_nan=0, n_unpaired_A=int(unA), n_unpaired_B=int(unB), n_nan_keys_A=int(nnA),
                  n_nan_keys_B=int(nnB), pair_ok=False)
    if not d_split:
        return nanres
    dv = np.array(list(d_split.values()))
    with np.errstate(invalid="ignore"):
        win = float(np.mean(dv < 0))
    out = dict(nanres, delta=float(dv.mean()), delta_beq=float(np.mean(list(db_split.values()))), n_splits=len(d_split), n_blocks=nbs,
               split_win=win, delta_split=d_split, delta_split_beq=db_split, n_keys_A=nkA, n_keys_B=nkB,
               pair_ok=bool(unA == 0 and unB == 0 and nnA == 0 and nnB == 0))
    if dists:
        dist = np.mean(np.stack(dists), 0); dist_b = np.mean(np.stack(dists_b), 0)
        lo, hi, nn = ci95(dist); lob, hib, nnb = ci95(dist_b)
        out.update(ci_lo=lo, ci_hi=hi, ci_lo_beq=lob, ci_hi_beq=hib, p_boot=boot_p(dist), dist=dist, dist_beq=dist_b, n_nan=max(nn, nnb))
    return out


def _margins(margin):
    if isinstance(margin, (tuple, list, np.ndarray)):
        m = list(margin)
        if len(m) != 2:
            raise ValueError("margin 은 스칼라 또는 (셀 가중, 블록 등가중) 2원소다")
        return float(m[0]), float(m[1])
    return float(margin), float(margin)


def verdict4(lo, hi, lob, hib, margin) -> str:
    """4분 판정. 우세 = 두 가중의 CI 상한 < 0. 열세 = 두 가중의 CI 하한 > 0. 동등 = 우세도 열세도 아니고 두 가중의 CI 가 모두
    [−margin, +margin] 안. 미결정 = 그 외. CI 끝점에 NaN 이 있으면 '판정 불가'. margin 은 스칼라 또는 (셀 가중, 블록 등가중)."""
    v = [float(x) for x in (lo, hi, lob, hib)]
    if any(np.isnan(x) for x in v):
        return "판정 불가"
    lo, hi, lob, hib = v
    if hi < 0 and hib < 0:
        return "우세"
    if lo > 0 and lob > 0:
        return "열세"
    mc, mb = _margins(margin)
    if np.isfinite(mc) and np.isfinite(mb) and -mc <= lo and hi <= mc and -mb <= lob and hib <= mb:
        return "동등"
    return "미결정"


def half_width(lo, hi, lob, hib) -> float:
    """두 가중의 95 % CI 반폭 가운데 큰 값."""
    return float(max((float(hi) - float(lo)) / 2.0, (float(hib) - float(lob)) / 2.0))


def eq_state(lo, hi, lob, hib, margin) -> str:
    """동등성 상태. 가중마다 CI 반폭이 그 가중의 한계를 넘으면 '검정 불가(정밀도 미달)'(스칼라 한계이면 max(반폭) > 한계와 같다).
    아니면 verdict4 가 '동등' 이면 '동등', 그 밖은 '동등 아님'. CI 에 NaN 이 있으면 '판정 불가'. 부호를 쓰지 않는다."""
    v = [float(x) for x in (lo, hi, lob, hib)]
    if any(np.isnan(x) for x in v):
        return "판정 불가"
    mc, mb = _margins(margin)
    if (v[1] - v[0]) / 2.0 > mc or (v[3] - v[2]) / 2.0 > mb or not (np.isfinite(mc) and np.isfinite(mb)):
        return "검정 불가(정밀도 미달)"
    return "동등" if verdict4(*v, (mc, mb)) == "동등" else "동등 아님"


def rel_margin(ref_cell, ref_beq, rel=EQ_REL):
    """가중별 동등성 한계 (rel·|기준 셀 가중 점수|, rel·|기준 블록 등가중 점수|)."""
    return float(rel) * abs(float(ref_cell)), float(rel) * abs(float(ref_beq))


# ================================================================ 10. 물리 표본
def phys_log_samples(df_cells, df_fill, n_draws, seed, unit_tdd=False) -> np.ndarray:
    """물리식 로그 표본 (n_draws, n_cells).

    PHYS_INPUT_COLS 의 결측을 df_fill(원천 셀)의 열 중앙값으로 채운 복사본에 load_physics_inputs 를 부른다.
    rng = RandomState(seed_of('lgu-phys', seed)), Sr = rng.uniform(*PHYS_SR, n_draws), n_t = rng.uniform(*PHYS_NT, n_draws)
    (셀 공통 난수, Sr 을 먼저 뽑는다). 추출 m 마다 k_t = johansen_k(ρ_d, sand, φ, f_om, Sr=Sr[m]), θ = θ_sat·Sr[m],
    out[m] = log(_stefan_full(TDD 또는 1, k_t, θ, n_t[m])). unit_tdd 이면 TDD = 1 로 두어 물리 계수 E_phys(cm/√(°C·일))의 로그를 준다.
    TDD 는 1e-6 이상으로 자른다.
    """
    cols = list(PHYS_INPUT_COLS)
    d = df_cells[cols].copy()
    for c in cols:
        ref = np.asarray(df_fill[c].values, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = float(np.nanmedian(ref)) if np.isfinite(ref).any() else np.nan
        v = np.asarray(d[c].values, float).copy()
        v[~np.isfinite(v)] = med
        d[c] = v
    x = load_physics_inputs(d)
    rng = np.random.RandomState(seed_of("lgu-phys", seed))
    nd = int(n_draws)
    Sr = rng.uniform(PHYS_SR[0], PHYS_SR[1], size=nd)
    nt = rng.uniform(PHYS_NT[0], PHYS_NT[1], size=nd)
    n = len(d)
    tdd = np.ones(n) if unit_tdd else np.clip(np.asarray(x["tdd"], float), 1e-6, None)
    out = np.empty((nd, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        for m in range(nd):
            k_t = johansen_k(x["rho_d"], x["sand_f"], x["phi"], x["f_om"], Sr=Sr[m])
            out[m] = np.log(_stefan_full(tdd, k_t, x["theta_sat"] * Sr[m], nt[m]))
    return out
