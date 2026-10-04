"""XD_placement_policy(계획 2.4) 구현: 규칙형 배치 S8 계열(XD-alg), Algorithm P 의 기계적 선택(알래스카 계열만), 학습 정책 S9 의
학습 자료와 S9-GBM, 동결 기록, 시험 과제 평가 팔(XD-4·5).

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100) 2.4절, 0.3절(커밋 시점과 열람 순서), 1절(공통 규약).
설계 근거: docs/research/2026-10-04/placement_and_workflow_algorithm.md 4·6절, harness_implementation_plan.md 4.4절.
계획 해석과 구현 결정: docs/research/2026-10-04/impl_notes/x_placement_policy.md. 가설과 판정 규칙은 계획 그대로 쓴다.
S9-DS(DeepSets, 로컬 GPU)는 scripts/3_deep_learning/x_placement_deepsets.py 다. 공용 골격은 xbatch_core.py(동결 모듈 h40·h42·h54 를
sha256 대조 뒤 적재)다. 이 모듈은 동결 모듈을 고치지 않고 하위 클래스와 함수 호출로만 쓴다.

실험(--exp)
  xd_r      지역 내(모드 r, WF2 형식). 대상 Alaska, Lena, Canada, AL-1, AL-2. 분할 1–5(h40 규칙의 유효 분할), n {20, 50, 100, 200, 500}
            (|A| 미만), 추출 5, seed 0·1. 전략 S1, S2, S4(h54 의 WF2 구현과 같은 함수·seed), S8a, S8b, S8c, S8p. 방법 P1, R1(λ 0.25·0.5·1.0
            저장, 판정 λ 0.25), D1. S8a 조각은 같은 라벨로 계수만 바꾼 S8w(설계 가중)·S8r(블록 무작위 효과)의 P1·R1 을 함께 낸다.
  xd_t      전이(모드 x, WF8 형식). 대상 Lena, Canada, Russia_W, Russia_E, Alaska, AL-1–AL-6, Tibet_LGD, Russia_C~lgd. 분할 1–5(유효),
            n {10, 40}(|A| 미만), 추출 5, seed 0·1. 전략은 xd_r 과 같다(S1·S2·S4 = WF8 구현). 방법 P1, R1(λ 0.25).
  xd_learn  XD-learn 학습 자료(작업 R1a). 개발 과제 Alaska, AL-1–AL-6(모드 x), 분할 1–5, n {10, 20, 40}. 과제·분할·n 마다 라벨 집합 200개
            (S1 40 %, S2·S4·S8 계열 다른 seed 30 %, 섭동 30 %)와 기준 S1 추출 5개의 R1(λ 0.25, seed 0). 집합 특징(라벨 미사용)을 함께 쓴다.
  xd_test   XD-4 시험 팔(작업 R2b). 시험 과제 LE-1, LE-2, CA-2, CA-3, Tibet_LGD(모드 x), 분할 1–5, n {20, 40}, 정책 seed 0–4, 학습 seed 0·1.
            동결된 S9* 의 색인 파일(sha256 대조)과, 같은 단위에서 다시 적합한 S1·고정된 Algorithm P 의 P1·R1(λ 0.25). Algorithm P 가 S8
            후보이면 누적 예산 일정은 XC 의 Algorithm P 팔과 같은 AP_SCHEDULE(10, 40, 160)이다(시험 격자 20·40 이 아니다).
            시작 전에 색인 파일이 모든 시험 과제·유효 분할·정책 seed 를 덮는지 확인한다(빠지면 거부, 서술 과제 Russia_C~lgd 는 건너뜀).
            본 실행은 동결 기록이 git 에 커밋되어 있고 수정되지 않았을 때만 한다(git 이 없는 Rescale 묶음은 --attest-freeze 기록으로 대조).

배치 S8(γ)(계획 2.4 의사코드, 함수 algorithm_p_order·power_alloc)
  Z = A 풀 x25 의 중앙값 대체 뒤 A 풀 평균·표준편차 표준화(S4 와 같은 h40._prep_stats). 블록 순서 β = 블록 중심의 farthest-first(첫 블록은
  seed). 단계 n_k 마다 q = power_alloc(N_b, n_k, γ)(최대 나머지, 상한 N_b, 소수부 동률은 β 앞), deficit 가 큰 블록부터 한 셀씩: 라벨이 없는
  블록은 중심 근접 셀(S8b 는 farthest-first), 라벨이 있는 블록은 전역 최소 거리 최대 셀. 누적 예산 일정은 XD-alg 에서 그 팔의 n 격자
  (xd_t 10·40 은 (10, 40, 160) 의 앞부분과 같은 순서), XD-4 의 고정된 Algorithm P 와 XC 에서 (10, 40, 160)이다.
  S8a = γ 0, S8b = S8a 에서 블록 안 첫 셀도 farthest-first, S8c = γ 0.5, S8p = γ 1. seed = seed_of('xd-S8', 대상, 모드, 분할, 추출).
  S8w·S8r 은 S8a 의 라벨에 수축하지 않은 E_w·E_RE 를 쓴다(계획 2.4 문구 그대로, 비유한이면 E0).

Algorithm P 선택(계획 2.3 고정 규칙, 함수 select_algorithm_p): 알래스카 계열 조각만 읽는다(파일 이름으로 거른 뒤 연다). 후보 S2, S4, S8a, S8b,
  S8c, S8p. 제외 = 알래스카 계열 행(전이 Alaska·AL-1–AL-6 의 n 10·40, 지역 내 Alaska·AL-1·AL-2 의 n 20–500) 가운데 하나라도 S_k − S1
  (R1@0.25)의 2단 CI 하한이 어느 한 가중에서 0 초과. 점수 = 전이 알래스카 계열 대상마다 상대 변화를 n 10·40 에서 평균하고 두 가중 가운데 큰
  값, 대상 단순 평균. 최소 점수 후보. 1e-9 미만 동률은 S8a, S8c, S8p, S8b, S2, S4 순. 모두 빠지면 S1. 선택 파일과 sha256 을 쓴다.
  PE1·티베트 행은 화면과 로그에 쓰지 않는다(읽지도 않는다).

산출(<out-dir>, 기본 data/processed/xbatch/XD_placement_policy)
  shards/<tag>__cpu__<대상>__<모드>__s<분할>__<변형>_{runs.csv, blocksse.npz, unit.json}(xd_learn 은 _sets.npz 를 더한다).
    tag = xdr, xdt, xdl, xdx(스모크는 '_smoke' 를 붙인다). 변형 = 전략(xd_r, xd_t), n<값>(xd_learn), 없음(xd_test).
    shards/ 에는 알래스카 계열 xd_t·xd_r 조각(열람 순서 1), xd_learn, xd_test 조각만 둔다.
  sealed/shards/: 알래스카 계열이 아닌 xd_t·xd_r 조각(PE1, Tibet_LGD, 레나·캐나다 지역 내). 계획 0.3 열람 순서 3·6 과 출력 제한의
    '봉인 폴더' 규칙을 원자료 조각에도 적용한다. 집계(summarize_alg)와 재현 관문만 읽고, 선택(select_algorithm_p)과 학습 자료
    (load_learning, S9-DS)는 읽지 않는다(읽으려 하면 거부).
  sealed/shards_smoke/: 스모크 조각 전부(스모크 대상은 알래스카 계열만).
  selection/algorithm_p_selection.json(+ .sha256), selection/algorithm_p_alaska_table.csv(+ .sha256): 알래스카 계열 행만(열람 순서 1).
  policy/: S9-GBM 모형·교차검증 기록, 정책 색인 파일(+ .sha256). freeze/s9_freeze_manifest.json(+ .sha256).
  sealed/: XD-alg 대비·판정 표(XD-1·2·3, Holm m 20, XD-6), XD-4·5 표. 화면에는 행 수와 sha256 만 쓴다(계획 0.3).
  gate/xd_gate.csv: 재현 관문(S1·S2·S4 의 블록 SSE 를 WF2·WF8 조각과 비교, 같은 노드 종류 차 0). count/xd_count*.csv: --count-only.

명령(로컬은 세기·스모크·시험만. 본 실행과 재표집 10,000회 집계는 Rescale, WF_RESCALE=1. 로컬 실행(WF_RESCALE·LG_RESCALE 없음)은
main 시작 때 가용 메모리 30 GB 를 확인하고(모자라면 --mem-wait 초 동안 기다린 뒤 거부) 자료 영역 상한 10 GB(RLIMIT_DATA)를 건다.
prlimit --as(RLIMIT_AS) 아래에서는 CatBoost 1.2.10 이 가끔 특징 중요도 계산(_calc_fstr)에서 멈춘다(2026-10-04 로컬 시험에서 재현))
  세기:   CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/x_placement_policy.py --count-only --exp xd_r,xd_t,xd_learn
  스모크: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 scripts/3_deep_learning/x_placement_policy.py --smoke
          --threads 2 --workers 0 2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE'
  R1a:    WF_RESCALE=1 python3 .../x_placement_policy.py --exp xd_r,xd_t,xd_learn --workers 22 --threads 4 --resume --no-summarize
          WF_RESCALE=1 python3 .../x_placement_policy.py --summarize-only            # XD-alg 봉인 표 + 알래스카 계열 선택(Algorithm P)
          WF_RESCALE=1 python3 .../x_placement_policy.py --train-gbm --threads 4    # S9-GBM(R1a 마지막 단계)
          WF_RESCALE=1 python3 .../x_placement_policy.py --export-selection gbm    # S9-GBM 의 시험 과제 선택 색인(라벨 미사용)
  로컬:   python3 .../x_placement_policy.py --gate                                 # 재현 관문(WF2·WF8 조각과 비교, 수만 출력)
          XBATCH_GPU=1 python3 .../x_placement_deepsets.py --train --export        # S9-DS(로컬 GPU 0–4)
          python3 .../x_placement_policy.py --freeze --gbm-meta ... --ds-meta ...   # S9* 동결 기록(커밋은 사용자 확인 뒤)
          python3 .../x_placement_policy.py --attest-freeze --freeze-manifest <동결 기록>   # 커밋 뒤: 커밋 확인 기록(R2b 묶음에 넣는다)
  R2b:    WF_RESCALE=1 python3 .../x_placement_policy.py --exp xd_test --freeze-manifest <동결 기록> --selection-file <선택 파일> --workers 22
          WF_RESCALE=1 python3 .../x_placement_policy.py --summarize-test --freeze-manifest ... --selection-file ...   # XD-4·5 봉인 표
  XD-5:   (XD-4 표 열람 뒤. --xd5 는 sealed/xd4_summary.json 과 그 봉인 기록 항목이 있어야 한다)
          WF_RESCALE=1 python3 .../x_placement_policy.py --exp xd_learn --dev-tasks LE-1,LE-2,CA-2,CA-3,Tibet_LGD --xd5 ...
          WF_RESCALE=1 python3 .../x_placement_policy.py --summarize-xd5 --xd5 --gbm-meta <s9_gbm_meta.json>  # 후회, 계열·과제 하나 제외
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import xbatch_core as XB                                                    # noqa: E402  numpy 보다 먼저(스레드·장치 환경 변수)

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402

H, X, W = XB.H, XB.X, XB.W
seed_of = XB.seed_of
LO = XB.LO
LAM = XB.LAM_BASE                                                           # λ 0.25(판정 λ)
ROOT = XB.ROOT

# ================================================================ 1. 고정값(계획 2.4. 바꾸면 사전 등록에서 벗어난다)
EXP_NAME = XB.EXP_NAMES["XD"]                                               # XD_placement_policy
DEFAULT_OUT = XB.out_dir("XD")
TAGS = dict(xd_r="xdr", xd_t="xdt", xd_learn="xdl", xd_test="xdx")
EXPS = tuple(TAGS)
MODE_OF = dict(xd_r="r", xd_t="x", xd_learn="x", xd_test="x")
AL_SUBS = tuple(f"AL-{i}" for i in range(1, 7))
XDR_TARGETS = ("Alaska", "Lena", "Canada", "AL-1", "AL-2")
XDR_GRID = (20, 50, 100, 200, 500)
XDT_TARGETS = ("Lena", "Canada", "Russia_W", "Russia_E", "Alaska") + AL_SUBS + ("Tibet_LGD", "Russia_C~lgd")
XDT_GRID = (10, 40)
DRAWS = 5
BASE_STRATS = ("S1", "S2", "S4")                                            # WF2·WF8 구현 그대로(재현 관문)
S8_SPECS = {"S8a": (0.0, "center"), "S8b": (0.0, "farthest"), "S8c": (0.5, "center"), "S8p": (1.0, "center")}
STRATS = BASE_STRATS + tuple(S8_SPECS)
COEF_VARIANTS = ("S8w", "S8r")                                              # S8a 의 라벨 + 계수만 다르다(탐색 팔, Algorithm P 후보 아님)
COEF_BASE = "S8a"
S8_SEED_TAG = "xd-S8"
# Algorithm P 고정 규칙(계획 2.3)
AP_CANDIDATES = ("S2", "S4", "S8a", "S8b", "S8c", "S8p")
AP_TIE_ORDER = ("S8a", "S8c", "S8p", "S8b", "S2", "S4")
AP_DEFAULT = "S8a"                                                          # P_default(마감 초과 시)
AP_TIE_TOL = 1e-9
AP_NONE = "S1"                                                              # 모든 후보가 빠지면 Algorithm P = S1
ALASKA_X = ("Alaska",) + AL_SUBS                                            # 전이 알래스카 계열(n 10·40)
ALASKA_R = ("Alaska", "AL-1", "AL-2")                                       # 지역 내 알래스카 계열(n 20–500)
ALASKA_X_N = (10, 40)
ALASKA_R_N = (20, 50, 100, 200, 500)
AP_SCHEDULE = (10, 40, 160)                                                 # 고정된 Algorithm P 의 누적 예산 일정(2.4 의사코드, XC 의 Algorithm P 팔)
# 스모크(계획 0.3 출력 제한): 대상은 알래스카 계열만(PE1·티베트·시험 과제의 행을 만들지 않는다), 조각은 봉인 폴더(sealed/shards_smoke)
SMOKE_TARGETS = dict(xd_r=("AL-2",), xd_t=("AL-3", "AL-5"), xd_learn=("AL-3", "AL-5"), xd_test=("AL-5",))
SMOKE_GRID = dict(xd_r=(20,), xd_t=(10, 40), xd_learn=(10,), xd_test=(20,))
SEALED_SHARDS = "shards"                                                    # sealed/shards: 알래스카 계열이 아닌 xd_t·xd_r 조각
SMOKE_SHARDS = "shards_smoke"                                               # sealed/shards_smoke: 스모크 조각
MEM_MIN_GB = 30.0                                                           # 1절 로컬 자원: 시작 전 가용 메모리 하한
MEM_LIMIT_GB = 10.0                                                         # 1절 로컬 자원: 작업 하나의 메모리 상한
XD4_SEALED = "xd4_summary.json"                                             # XD-5 의 전제(XD-4 시험 표, 접미사 없음)
# XD-learn
DEV_TASKS = ALASKA_X                                                        # 개발 과제(모드 x). BNZ 등 새 하위 과제는 넣지 않는다
DEV_N = (10, 20, 40)
N_SETS = 200
SET_MIX = (0.4, 0.3, 0.3)                                                   # S1, 구조 집합(S2·S4·S8 계열), 섭동 집합
PERT_FRAC = (0.2, 0.5)
STRUCT_GENS = ("S2", "S4", "S8a", "S8b", "S8c", "S8p")
TEST_TASKS = ("LE-1", "LE-2", "CA-2", "CA-3", "Tibet_LGD")                  # 시험 과제(모드 x), 한 번만 평가
TEST_DESC = ("Russia_C~lgd",)                                               # |A| 26–35 라 n 40 이 없어 서술로만(k/5 에 넣지 않는다)
TEST_N = (20, 40)
POLICY_SEEDS = (0, 1, 2, 3, 4)
FAMILY_OF = {"LE-1": "Lena", "LE-2": "Lena", "CA-2": "Canada", "CA-3": "Canada", "Tibet_LGD": "Tibet", "Russia_C~lgd": "Russia_C"}
FAMILY_TASKS = {"Alaska": DEV_TASKS, "Lena": ("LE-1", "LE-2"), "Canada": ("CA-2", "CA-3"), "Tibet": ("Tibet_LGD",)}   # XD-5 계열 하나 제외
POLICY_BATCH = 5
POLICY_FF_TOP = 300
POLICY_RAND = 200
POLICY_NMAX = max(TEST_N)
COV_SUB = 1000                                                              # 덮임 특징의 후보 부분표본 수
CTX_FEATS = ["ctx_ncand", "ctx_nblocks", "ctx_effblocks", "ctx_cov_between", "ctx_src_diff"]
SET_FEATS = ["n_set", "cov_mean", "cov_p90", "cov_max", "energy", "blk_cov_frac", "blk_entropy", "blk_l1", "s_mean", "s_sd", "s_ratio",
             "s_sum2", "di_mean"]
FEATS = SET_FEATS + CTX_FEATS                                               # S9-GBM 입력(라벨 미사용)
EL_FEATS = [f"z{j}" for j in range(len(XB.FEATS))] + ["s_std", "blk_frac", "di", "nn_dist"]   # S9-DS 원소 특징
GBM_ITERS = (100, 200, 400, 800)                                            # 조기 종료 후보(교차검증으로 고른다)
GBM_CFG = dict(learning_rate=0.05, depth=4, l2_leaf_reg=3.0)
GBM_SEED = 0
HOLM_M = 20                                                                 # XD Holm 가족: XD-1 2 + XD-2 2 + XD-3 16
VARIANTS_S9 = ("S9-GBM", "S9-DS")
INDEX_FORMAT = "xd_index_v1"


def n_draws(a) -> int:
    """추출 수(5, --draws-cap 이 있으면 그 이하)."""
    cap = int(getattr(a, "draws_cap", 0) or 0)
    return max(1, min(DRAWS, cap)) if cap > 0 else DRAWS


def assert_smoke_alaska(targets=None) -> bool:
    """스모크 대상이 알래스카 계열(ALASKA_X ∪ ALASKA_R) 안에 있다(계획 0.3 열람 순서 3·6: PE1·티베트·시험 과제의 행을 스모크에서 만들지 않는다)."""
    targets = SMOKE_TARGETS if targets is None else targets
    fam = set(ALASKA_X) | set(ALASKA_R)
    bad = sorted({str(t) for v in targets.values() for t in v} - fam)
    if bad:
        raise AssertionError(f"스모크 대상 {bad} 가 알래스카 계열 밖이다")
    return True


assert_smoke_alaska()


def unit_sealed(exp, alias, mode) -> bool:
    """조각을 봉인 폴더(sealed/shards)에 써야 하는 단위인가: 알래스카 계열(xd_t 의 ALASKA_X 모드 x, xd_r 의 ALASKA_R 모드 r)이 아닌 xd_t·xd_r.
    xd_learn(개발 과제, 또는 XD-4 표 열람 뒤의 --xd5)과 xd_test(동결 커밋 뒤)는 봉인하지 않는다."""
    if exp == "xd_t":
        return not (str(mode) == "x" and str(alias) in ALASKA_X)
    if exp == "xd_r":
        return not (str(mode) == "r" and str(alias) in ALASKA_R)
    return False


def out_shard_dirs(out_dir, smoke=False) -> tuple:
    """(봉인하지 않는 조각 폴더, 봉인 조각 폴더). 본 실행 = (<out>/shards, sealed/shards), 스모크 = 둘 다 sealed/shards_smoke."""
    out = Path(out_dir) if os.path.isabs(str(out_dir)) else ROOT / str(out_dir)
    sd = XB.sealed_dir(EXP_NAME, out.parent)
    if smoke:
        return sd / SMOKE_SHARDS, sd / SMOKE_SHARDS
    return out / "shards", sd / SEALED_SHARDS


def shard_dir_of(a, exp, alias, mode) -> Path:
    """단위의 조각 폴더(unit_sealed 이면 봉인 폴더)."""
    return a.XD_SEALED_SHARDS if unit_sealed(exp, alias, mode) else a.XD_SHARDS


def alg_dirs(a) -> list:
    """XD-alg 집계·재현 관문이 읽는 조각 폴더(봉인하지 않는 폴더와 봉인 폴더, 같으면 하나)."""
    return list(dict.fromkeys([Path(a.XD_SHARDS), Path(a.XD_SEALED_SHARDS)]))


def refuse_sealed_rows(shards_dir, who) -> Path:
    """봉인 조각 폴더(sealed/shards)를 읽으려 하면 거부한다(선택과 학습 자료는 봉인하지 않는 폴더만 읽는다, 계획 0.3 열람 순서 1·3)."""
    p = Path(os.path.realpath(str(shards_dir)))
    if p.name == SEALED_SHARDS and p.parent.name == "sealed":
        raise SystemExit(f"[거부] {who} 는 봉인 조각 폴더({p})를 읽지 않는다(계획 0.3 열람 순서 3)")
    return Path(shards_dir)


def on_rescale() -> bool:
    """Rescale 작업 안인가(WF_RESCALE=1 또는 LG_RESCALE=1). --allow-local 은 로컬 실행이므로 로컬 자원 규약을 따른다."""
    return os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1"


def local_resources(wait_s=0.0, log=None) -> dict:
    """1절 로컬 자원: 가용 메모리가 MEM_MIN_GB 아래면 wait_s 동안 기다리고(그래도 모자라면 거부), 자료 영역 상한 MEM_LIMIT_GB 를 건다."""
    g = XB.require_memory(MEM_MIN_GB, wait_s=float(wait_s))
    ok = limit_data(MEM_LIMIT_GB)
    if log is not None:
        log(f"[자원] 가용 메모리 {g:.1f} GB(하한 {MEM_MIN_GB:.0f}) · 자료 영역 상한 {MEM_LIMIT_GB:.0f} GB {'설정' if ok else '실패'}")
    return dict(mem_available_gb=float(g), limit_data=bool(ok))


def limit_data(gb=10.0) -> bool:
    """프로세스 자료 영역 상한(RLIMIT_DATA, prlimit --data 와 같다). CatBoost 1.2.10 은 주소 공간 상한(RLIMIT_AS, prlimit --as)
    아래에서 가끔 특징 중요도 계산(_calc_fstr)에 멈춘다(로컬 시험에서 확인). 그래서 이 모듈의 로컬 실행은 RLIMIT_DATA 를 쓴다."""
    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        lim = int(float(gb) * 2 ** 30)
        if soft != resource.RLIM_INFINITY and soft <= lim:
            return True
        resource.setrlimit(resource.RLIMIT_DATA, (lim, hard))
        return True
    except (ValueError, OSError):
        return False


def _sha256(path) -> str:
    return XB.sha256_file(path)


def _write_json(path, obj, allowed=None, sidecar=True) -> str:
    """JSON 을 원자적으로 쓰고 sha256 사이드카(<파일>.sha256)를 둔다. 반환 sha256."""
    path = Path(path)
    XB.check_out_dir(path.parent, allowed)
    path.parent.mkdir(parents=True, exist_ok=True)
    XB._atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default))
    sha = _sha256(path)
    if sidecar:
        XB._atomic_write(Path(str(path) + ".sha256"), f"{sha}  {path.name}\n")
    return sha


def _write_csv(path, df, allowed=None, sidecar=True) -> str:
    path = Path(path)
    XB.check_out_dir(path.parent, allowed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    sha = _sha256(path)
    if sidecar:
        XB._atomic_write(Path(str(path) + ".sha256"), f"{sha}  {path.name}\n")
    return sha


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    return str(v)


def verify_sidecar(path) -> str:
    """<파일>.sha256 사이드카와 파일의 sha256 을 대조한다. 다르거나 없으면 SystemExit. 반환 sha256."""
    path = Path(path)
    side = Path(str(path) + ".sha256")
    if not path.exists() or not side.exists():
        raise SystemExit(f"[거부] {path} 또는 sha256 사이드카가 없다")
    want = side.read_text().split()[0]
    got = _sha256(path)
    if got != want:
        raise SystemExit(f"[거부] {path.name} 의 sha256 이 사이드카와 다르다(파일이 바뀌었다)")
    return got


# ================================================================ 2. Algorithm P(S8 계열)
def power_alloc(Nb, n, gamma=0.0, rank=None) -> np.ndarray:
    """블록 배분(계획 2.4): w_b = N_b^γ, q_b = n·w_b/Σw 의 최대 나머지 배분, 상한 N_b, 소수부 동률은 β 순위(rank)가 앞인 블록 우선.
    상한에 걸린 블록은 N_b 로 고정하고 남은 수를 나머지 블록에 같은 비례로 다시 나눈다(반복). 합 = min(n, ΣN_b)."""
    Nb = np.asarray(Nb, np.int64)
    B = len(Nb)
    if B == 0:
        return np.zeros(0, np.int64)
    rank = np.arange(B) if rank is None else np.asarray(rank, np.int64)
    n = int(min(max(int(n), 0), int(Nb.sum())))
    w = np.power(Nb.astype(float), float(gamma))
    capped = np.zeros(B, bool)
    q = np.zeros(B)
    for _ in range(B + 1):
        free = ~capped
        rest = n - int(Nb[capped].sum())
        q = np.where(capped, Nb.astype(float), 0.0)
        if free.any() and w[free].sum() > 0:
            q[free] = rest * w[free] / w[free].sum()
        over = free & (q > Nb + 1e-12)
        if not over.any():
            break
        capped |= over
    alloc = np.minimum(np.floor(q + 1e-9).astype(np.int64), Nb)
    frac = np.round(q - alloc, 12)
    left = n - int(alloc.sum())
    for j in np.lexsort((rank, -frac)):                                     # 소수부 내림차순, 동률은 β 순위
        if left <= 0:
            break
        if alloc[j] < Nb[j]:
            alloc[j] += 1
            left -= 1
    for j in np.argsort(rank, kind="stable"):                               # 안전장치(상한 때문에 남는 몫)
        while left > 0 and alloc[j] < Nb[j]:
            alloc[j] += 1
            left -= 1
    return alloc


def _dist_to(Z, z) -> np.ndarray:
    return np.sqrt(((Z - z) ** 2).sum(1))


def block_order(mu, seed, init_pos=None) -> np.ndarray:
    """블록 순서 β: 블록 중심 μ_b 의 farthest-first(동률은 작은 위치). 첫 블록 = seed 로 고른 블록(RandomState(seed % 2^32).randint(B)),
    이미 있는 라벨의 블록(init_pos)이 있으면 그 블록들에서 시작한다."""
    mu = np.asarray(mu, float)
    B = len(mu)
    if B == 0:
        return np.zeros(0, int)
    if init_pos is not None and len(init_pos):
        chosen = sorted({int(v) for v in init_pos})
    else:
        chosen = [int(np.random.RandomState(int(seed) % (2 ** 32)).randint(B))]
    mind = np.full(B, np.inf)
    for j in chosen:
        mind = np.minimum(mind, _dist_to(mu, mu[j]))
    mind[chosen] = -1.0
    order = list(chosen)
    while len(order) < B:
        j = int(np.argmax(mind))
        order.append(j)
        mind = np.minimum(mind, _dist_to(mu, mu[j]))
        mind[j] = -1.0
    return np.asarray(order, int)


def block_centroids(Z, blk):
    """(고유 블록(문자열 정렬), 셀 → 블록 위치, 블록 셀 수, 블록 중심 μ_b). 블록 번호는 문자열로 정렬한다(BlockStore 와 XC 참조 구현의
    규약. 이 자료의 블록 번호는 음수와 자릿수가 섞여 숫자 정렬과 문자열 정렬이 다르다)."""
    Z = np.asarray(Z, float)
    ub, inv = np.unique(np.asarray(blk).astype(str), return_inverse=True)
    B = len(ub)
    Nb = np.bincount(inv, minlength=B).astype(np.int64)
    mu = np.vstack([Z[inv == j].mean(0) for j in range(B)]) if B else np.zeros((0, Z.shape[1]))
    return ub, inv, Nb, mu


def algorithm_p_order(Z, blk, schedule, gamma=0.0, seed=0, init=None, first="center", return_info=False):
    """S8(γ) 배치 순서 π(계획 2.4 의사코드). Z = 표준화 공변량(후보 셀 행), blk = 후보 셀의 0.5° 블록, schedule = 누적 예산 일정,
    init = 이미 있는 라벨 L0(후보 색인, 없으면 None), first = 라벨이 없는 블록의 첫 셀 규칙('center' 블록 중심 근접, 'farthest'
    전역 최소 거리 최대). 단계 n_k 의 배치는 π[:n_k − |L0|](중첩). 라벨 값을 쓰지 않는다.
    - deficit 이 가장 큰 블록(동률은 β 순서)을 고르되 남은 셀이 있는 블록만 본다. deficit 이 모두 0 이하가 되어도(상한·배분 비단조)
      같은 규칙으로 계속 고른다.
    - π ∪ L0 가 비어 있으면(첫 셀) 'farthest' 도 블록 중심 근접 셀을 쓴다(farthest-first 는 기준 집합이 있어야 정의된다).
    - 동률은 작은 색인."""
    if first not in ("center", "farthest"):
        raise ValueError(f"first 는 center 또는 farthest: {first}")
    Z = np.asarray(Z, float)
    N = len(Z)
    ub, inv, Nb, mu = block_centroids(Z, blk)
    B = len(ub)
    L0 = np.unique(np.asarray(init, int)) if init is not None and len(init) else np.zeros(0, int)
    beta = block_order(mu, seed, np.unique(inv[L0]) if len(L0) else None)
    rank = np.empty(B, np.int64)
    rank[beta] = np.arange(B)
    taken = np.zeros(N, bool)
    taken[L0] = True
    cnt = np.bincount(inv[L0], minlength=B).astype(np.int64)
    mind = np.full(N, np.inf)
    for l_ in L0:
        mind = np.minimum(mind, _dist_to(Z, Z[l_]))
    pi, stages = [], []
    for nk in sorted({int(v) for v in schedule if int(v) > 0}):
        nk = min(nk, N)
        if nk - len(L0) <= len(pi):
            continue
        q = power_alloc(Nb, nk, gamma, rank)
        deficit = np.maximum(0, q - cnt).astype(np.int64)
        stages.append(dict(n=int(nk), alloc=q.tolist()))
        while len(pi) < nk - len(L0):
            cand = np.where(Nb - cnt > 0)[0]
            if not len(cand):
                break
            bstar = int(cand[np.lexsort((rank[cand], -deficit[cand]))[0]])
            cells = np.where((inv == bstar) & ~taken)[0]
            if cnt[bstar] == 0 and (first == "center" or not np.isfinite(mind).any()):
                c_ = int(cells[np.argmin(_dist_to(Z[cells], mu[bstar]))])
            else:
                c_ = int(cells[np.argmax(mind[cells])])
            pi.append(c_)
            taken[c_] = True
            cnt[bstar] += 1
            deficit[bstar] -= 1
            mind = np.minimum(mind, _dist_to(Z, Z[c_]))
    out = np.asarray(pi, int)
    if return_info:
        return out, dict(beta=[str(v) for v in ub[beta].tolist()], stages=stages)
    return out


def round_robin_order(blk, rng_seed) -> np.ndarray:
    """블록 순환 배정(h40.draw_blocks 와 같은 RNG 소비 순서)의 전체 순서. draw_blocks(…, n, …) = sort(이 순서의 앞 n 개)
    (seed = seed_of(대상, 모드, 분할, n, 추출, 'block') 일 때). 고정 seed 이면 중첩 순서가 된다(S2 의 중첩판)."""
    blk = np.asarray(blk)
    rng = np.random.RandomState(rng_seed)
    ub = np.unique(blk)
    order = [int(j) for j in rng.permutation(len(ub))]
    pools = {j: list(rng.permutation(np.where(blk == ub[j])[0])) for j in range(len(ub))}
    sel, active = [], order
    while active:
        nxt = []
        for j in active:
            if pools[j]:
                sel.append(int(pools[j].pop()))
                nxt.append(j)
        active = nxt
    return np.asarray(sel, int)


def placement_order(name, Z, blk, schedule, seed, init=None) -> np.ndarray:
    """배치 후보 이름(S2, S4, S8a, S8b, S8c, S8p) → 중첩 순서. XC 의 Algorithm P 팔(seed_of('xc-P', …))이 이 함수를 쓴다.
    S2 = 고정 seed 블록 순환(round_robin_order), S4 = h54.kcenter_order(첫 점 seed, init 이 있으면 그 라벨에서 시작)."""
    if name in S8_SPECS:
        g, f = S8_SPECS[name]
        return algorithm_p_order(Z, blk, schedule, g, seed, init=init, first=f)
    if name == "S4":
        Z = np.asarray(Z, float)
        nmax = min(max(int(v) for v in schedule), len(Z))
        if init is not None and len(init):
            init = np.asarray(init, int)
            order = W.kcenter_order(Z, nmax + len(init), seed, init=Z[init])
            return np.asarray([v for v in order if v not in set(init.tolist())], int)[:nmax]
        return W.kcenter_order(Z, nmax, seed)
    if name == "S2":
        return round_robin_order(blk, seed)
    raise ValueError(f"알 수 없는 배치 {name}")


def ap_spec(name) -> dict:
    """Algorithm P 후보의 설정(선택 파일과 XC 에 쓴다)."""
    if name in S8_SPECS:
        g, f = S8_SPECS[name]
        return dict(name=name, kind="S8", gamma=g, first=f, schedule="XC·XD-4 = (10, 40, 160), XD-alg = 그 팔의 n 격자",
                    schedule_fixed=list(AP_SCHEDULE))
    if name in ("S2", "S4", "S1"):
        return dict(name=name, kind=name, gamma=None, first=None, schedule="해당 없음")
    raise ValueError(name)


# ================================================================ 3. 계수 변형(S8w, S8r)
def coef_w(y, s, blk, Nb_of) -> float:
    """설계 가중 계수 E_w = Σ w s y / Σ w s², w_i = N_b(i)/n_b(i)(후보 블록 셀 수 / 그 블록 라벨 수). 비유한이면 NaN."""
    y = np.asarray(y, float); s = np.asarray(s, float); blk = np.asarray(blk)
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    if not m.any():
        return float("nan")
    ub, inv = np.unique(blk[m], return_inverse=True)
    nb = np.bincount(inv).astype(float)
    Nb = np.array([float(Nb_of[b]) for b in ub.tolist()])
    w = (Nb / nb)[inv]
    den = float((w * s[m] ** 2).sum())
    return float((w * s[m] * y[m]).sum() / den) if den > 0 else float("nan")


def coef_re(y, s, blk) -> tuple:
    """블록 무작위 효과 계수(DerSimonian–Laird). 블록 계수 E_b = Σ_b s y / Σ_b s², 표집 분산 v_b = σ²/Σ_b s².
    σ² = 블록 안 잔차 분산(자유도 Σ(n_b − 1) ≥ 1), 블록마다 라벨이 하나뿐이면 합동 최소제곱 잔차 분산(자유도 n − 1).
    τ² = max(0, (Q − (k − 1))/(Σw − Σw²/Σw)), Q 는 고정 효과 평균(= 합동 E_ls) 둘레. E_RE = Σ E_b/(v_b + τ²) / Σ 1/(v_b + τ²).
    τ² = 0 이면 셀 가중(E_ls), τ² 가 크면 블록 균등 평균에 가깝다. 반환 (E_RE, 정보)."""
    y = np.asarray(y, float); s = np.asarray(s, float); blk = np.asarray(blk)
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    y, s, blk = y[m], s[m], blk[m]
    n = len(y)
    if n == 0:
        return float("nan"), dict(k=0, df_within=0, tau2=np.nan, sigma2=np.nan, sigma2_src="none")
    E_ls = float((s @ y) / (s @ s))
    ub, inv = np.unique(blk, return_inverse=True)
    k = len(ub)
    if k < 2:
        return E_ls, dict(k=int(k), df_within=int(n - k), tau2=0.0, sigma2=np.nan, sigma2_src="one_block")
    Sb = np.bincount(inv, weights=s * s, minlength=k)
    Eb = np.bincount(inv, weights=s * y, minlength=k) / Sb
    df = int(n - k)
    if df >= 1:
        r = y - Eb[inv] * s
        sig2, src = float((r @ r) / df), "within"
    else:
        r = y - E_ls * s
        sig2, src = float((r @ r) / max(n - 1, 1)), "pooled"
    sig2 = max(sig2, 1e-12)
    v = sig2 / Sb
    w = 1.0 / v
    Q = float((w * (Eb - E_ls) ** 2).sum())
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1.0 / (v + tau2)
    return float((wr * Eb).sum() / wr.sum()), dict(k=int(k), df_within=df, tau2=float(tau2), sigma2=sig2, sigma2_src=src)


# ================================================================ 4. 라벨 없는 과제 특징(S9 의 입력)
class TaskFeat:
    """과제 하나(대상, 모드, 분할)의 라벨 없는 자료. 공변량 XA, √TDD sA, 블록 blkA, 원천 공변량만 읽는다(라벨 y 는 읽지 않는다).
    Z = A 풀 표준화(S4·S8 과 같다), DI = 비가중 AOA 비유사도(h54.aoa_di, 원천 통계 표준화, seed_of('wf-aoa', 대상, 분할) = WF2 S5 와 같다)."""

    def __init__(self, target, mode, split, XA, sA, blkA, Xsrc=None):
        self.target, self.mode, self.split = str(target), str(mode), int(split)
        XA = np.asarray(XA, np.float32)
        self.Z = W.standardize(XA, W.standardize_fit(XA))
        self.N = len(self.Z)
        s = np.asarray(sA, float)
        self.s_mean_A = float(np.nanmean(s)) if np.isfinite(s).any() else 1.0
        self.s = np.where(np.isfinite(s), s, self.s_mean_A)
        sd = float(np.std(self.s)) + 1e-6
        self.s_std = (self.s - self.s.mean()) / sd
        ub, inv = np.unique(np.asarray(blkA), return_inverse=True)
        self.inv, self.B = inv, len(ub)
        self.Nb = np.bincount(inv, minlength=self.B).astype(float)
        self.p_b = self.Nb / max(self.N, 1)
        self.blkfrac = self.p_b[inv]
        if Xsrc is not None and len(Xsrc):
            Xsrc = np.asarray(Xsrc, np.float32)
            ss = W.standardize_fit(Xsrc)
            self.DI = np.asarray(W.aoa_di(W.standardize(XA, ss), W.standardize(Xsrc, ss), seed_of("wf-aoa", self.target, self.split)), float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mA, mS, sS = np.nanmean(XA.astype(float), 0), np.nanmean(Xsrc.astype(float), 0), np.nanstd(Xsrc.astype(float), 0)
                src_diff = float(np.nanmean(np.abs(mA - mS) / (sS + 1e-9)))
        else:
            self.DI = np.zeros(self.N)
            src_diff = 0.0
        rs = np.random.RandomState(seed_of("xd-cov", self.target, self.mode, self.split))
        self.sub = np.sort(rs.choice(self.N, min(COV_SUB, self.N), replace=False))
        self.Zs = self.Z[self.sub]
        self.XX = float(_cdist(self.Zs, self.Zs).mean())                      # 후보 분포 안 평균 거리(V 통계)
        mu_all = self.Z.mean(0)
        mu_b = np.vstack([self.Z[inv == j].mean(0) for j in range(self.B)])
        tot = float(((self.Z - mu_all) ** 2).sum())
        betw = float((self.Nb[:, None] * (mu_b - mu_all) ** 2).sum())
        self.ctx = np.array([float(self.N), float(self.B), float(1.0 / max((self.p_b ** 2).sum(), 1e-12)),
                             betw / tot if tot > 0 else 0.0, src_diff], float)
        self.a_fp = a_fingerprint(XA, sA, blkA)

    @classmethod
    def from_ctx(cls, c):
        """문맥(h40.Ctx, h54.RCtx)에서 공변량 속성만 읽는다(XA, sA, blkA, X_src 또는 src_X, target, mode, split)."""
        Xs = getattr(c, "X_src", None)
        if Xs is None:
            Xs = getattr(c, "src_X", None)
        return cls(c.target, c.mode, c.split, c.XA, c.sA, c.blkA, Xs)

    # ------------------------------------------------------------ 집합 특징(라벨 미사용)
    def features(self, L) -> np.ndarray:
        """집합 L(A 색인)의 특징(FEATS 순서). 처음부터 계산한다."""
        L = np.asarray(L, int)
        n = len(L)
        ZL = self.Z[L]
        D = _cdist(self.Zs, ZL)
        dmin = D.min(1)
        LL = float(_cdist(ZL, ZL).mean())
        energy = 2.0 * float(D.mean()) - self.XX - LL
        cnt = np.bincount(self.inv[L], minlength=self.B).astype(float)
        q = cnt / n
        ent = float(-(q[q > 0] * np.log(q[q > 0])).sum())
        sL = self.s[L]
        return np.r_[float(n), dmin.mean(), np.percentile(dmin, 90), dmin.max(), energy, float((cnt > 0).mean()), ent,
                     float(np.abs(q - self.p_b).sum()), sL.mean(), sL.std(), sL.mean() / self.s_mean_A, float((sL ** 2).sum()),
                     float(self.DI[L].mean()), self.ctx]

    def features_batch(self, L, C) -> np.ndarray:
        """집합 L ∪ {c}(c ∈ C)의 특징 행렬(|C| × 특징 수). 탐욕 정책용 증분 계산이며 features 와 수치 오차 안에서 같다."""
        L = np.asarray(L, int); C = np.asarray(C, int)
        n1 = len(L) + 1
        nS = len(self.sub)
        if len(L):
            DL = _cdist(self.Zs, self.Z[L])
            dminL, crossL, LLs = DL.min(1), float(DL.sum()), float(_cdist(self.Z[L], self.Z[L]).sum())
            DLC = _cdist(self.Z[L], self.Z[C])
        else:
            dminL, crossL, LLs = np.full(nS, np.inf), 0.0, 0.0
            DLC = np.zeros((0, len(C)))
        DC = _cdist(self.Zs, self.Z[C])
        dmin = np.minimum(dminL[:, None], DC)
        cross = (crossL + DC.sum(0)) / (nS * n1)
        LL = (LLs + 2.0 * DLC.sum(0)) / (n1 * n1)
        energy = 2.0 * cross - self.XX - LL
        cntL = np.bincount(self.inv[L], minlength=self.B).astype(float)
        Q = np.tile(cntL, (len(C), 1))
        Q[np.arange(len(C)), self.inv[C]] += 1.0
        Q /= n1
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.where(Q > 0, Q * np.log(np.where(Q > 0, Q, 1.0)), 0.0).sum(1)
        l1 = np.abs(Q - self.p_b[None, :]).sum(1)
        cov = (Q > 0).mean(1)
        S1 = float(self.s[L].sum()); S2 = float((self.s[L] ** 2).sum())
        sc = self.s[C]
        smean = (S1 + sc) / n1
        svar = np.maximum((S2 + sc ** 2) / n1 - smean ** 2, 0.0)
        di = (float(self.DI[L].sum()) + self.DI[C]) / n1
        ctx = np.tile(self.ctx, (len(C), 1))
        return np.column_stack([np.full(len(C), float(n1)), dmin.mean(0), np.percentile(dmin, 90, axis=0), dmin.max(0), energy, cov, ent, l1,
                                smean, np.sqrt(svar), smean / self.s_mean_A, S2 + sc ** 2, di, ctx])

    # ------------------------------------------------------------ DeepSets 원소 특징
    def element_static(self) -> np.ndarray:
        """셀별 원소 특징의 고정 부분 [Z(25), s 표준화, 블록 셀 비율, DI](N × 28, float32)."""
        return np.column_stack([self.Z, self.s_std, self.blkfrac, self.DI]).astype(np.float32)


def _cdist(A, B) -> np.ndarray:
    """유클리드 거리 행렬(scipy cdist: 쌍마다 직접 계산해 BLAS 스레드 수와 무관하게 같은 값을 낸다)."""
    from scipy.spatial.distance import cdist
    A = np.asarray(A, float); B = np.asarray(B, float)
    if not len(A) or not len(B):
        return np.zeros((len(A), len(B)))
    return cdist(A, B, "euclidean")


def elements_of(static, L) -> np.ndarray:
    """집합 L 의 원소 특징(|L| × 29): 고정 부분 + 같은 집합 안 최근접 라벨 거리(표준화 공변량, 원소 1개면 0)."""
    L = np.asarray(L, int)
    E = np.asarray(static, np.float32)[L]
    Zl = E[:, :len(XB.FEATS)].astype(float)
    D = _cdist(Zl, Zl)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1) if len(L) > 1 else np.zeros(len(L))
    return np.column_stack([E, nn]).astype(np.float32)


def a_fingerprint(XA, sA, blkA) -> str:
    """A 풀 지문(공변량·√TDD·블록, 라벨 미사용). 색인 파일과 실행 문맥의 A 가 같은지 대조한다."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(XA, np.float32)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(sA, np.float64)).tobytes())
    h.update("|".join(str(v) for v in np.asarray(blkA).tolist()).encode())
    return h.hexdigest()[:16]


# ================================================================ 5. 학습 자료의 라벨 집합(XD-learn)
def learning_sets(Z, blk, target, mode, split, n, n_sets=N_SETS, dry=False) -> list:
    """과제·분할·n 의 라벨 집합 n_sets 개 [(종류, 생성 seed, 정렬 색인)]. S1 무작위 40 %, 구조 집합 30 %(S2·S4·S8a·S8b·S8c·S8p 를 돌아가며
    다른 seed 로), 섭동 집합 30 %(구조 집합의 20–50 % 를 그 집합 밖 무작위 셀로 바꾼다). S8 계열의 일정은 DEV_N 가운데 n 이하와 n.
    dry 이면 구조 집합을 무작위 순열로 대신한다(세기 전용)."""
    blk = np.asarray(blk)
    N = len(blk)
    n = int(n)
    k1 = int(round(SET_MIX[0] * n_sets)); k2 = int(round(SET_MIX[1] * n_sets))
    if n_sets - k1 - k2 > 0 and k2 == 0:
        k2 = 1
    k3 = int(n_sets) - k1 - k2
    sched = tuple(sorted({v for v in DEV_N if v < n} | {n}))
    out = []
    for j in range(k1):
        rs = seed_of("xdl", target, mode, int(split), n, j, "S1")
        out.append(("S1", int(rs), np.sort(np.random.RandomState(rs).choice(N, n, replace=False))))
    struct = []
    for j in range(k2):
        g = STRUCT_GENS[j % len(STRUCT_GENS)]
        rs = seed_of("xdl", target, mode, int(split), n, j // len(STRUCT_GENS), g)
        if dry:
            idx = np.random.RandomState(rs).permutation(N)[:n]
        elif g == "S2":
            idx = round_robin_order(blk, rs)[:n]
        elif g == "S4":
            idx = W.kcenter_order(Z, n, rs)
        else:
            gm, fr = S8_SPECS[g]
            idx = algorithm_p_order(Z, blk, sched, gm, rs, first=fr)[:n]
        struct.append((g, int(rs), np.sort(np.asarray(idx, int))))
    out += struct
    for j in range(k3):
        bk, _, base = struct[j % k2]
        rs = seed_of("xdl", target, mode, int(split), n, j, "pert")
        rng = np.random.RandomState(rs)
        f = rng.uniform(*PERT_FRAC)
        m = int(min(max(1, int(round(f * n))), n, N - n))
        if m <= 0:
            out.append((f"pert:{bk}", int(rs), base.copy()))
            continue
        drop = rng.choice(n, m, replace=False)
        keep = np.delete(base, drop)
        add = rng.choice(np.setdiff1d(np.arange(N), base), m, replace=False)
        out.append((f"pert:{bk}", int(rs), np.sort(np.r_[keep, add]).astype(int)))
    return out


# ================================================================ 6. 작업 단위
class _XDMixin:
    """XD 단위의 공용 부분: S1·S2·S4 는 h54 의 WF2 구현(RUnit.strategy_sets·s4_order·std_stats)을 그대로 빌리고 S8 계열을 더한다."""

    std_stats = W.RUnit.std_stats
    s4_order = W.RUnit.s4_order
    strategy_sets = W.RUnit.strategy_sets

    def _xd_init(self):
        self._std = None
        self._Z = None
        c = self.c
        ub, cnt = np.unique(np.asarray(c.blkA), return_counts=True)
        self.Nb_of = {b: int(k) for b, k in zip(ub.tolist(), cnt.tolist())}
        if not self.dry:
            self.notes["traits"] = region_traits(c)

    def zA(self):
        if self._Z is None:
            self._Z = W.standardize(self.c.XA, self.std_stats())
        return self._Z

    def xd_sets(self, strat, budgets, d):
        """전략 → {n: 정렬 색인}. S1·S2·S4 = h54(WF2·WF8)와 같은 함수·seed, S8 계열 = algorithm_p_order(일정 = budgets)."""
        if strat in BASE_STRATS:
            return self.strategy_sets(strat, budgets, d)
        if strat in S8_SPECS:
            if self.dry:
                order = np.random.RandomState(int(d) + 23).permutation(self.nA)[:max(budgets)]
            else:
                g, f = S8_SPECS[strat]
                c = self.c
                order = algorithm_p_order(self.zA(), c.blkA, budgets, g, seed_of(S8_SEED_TAG, c.target, c.mode, int(c.split), int(d)), first=f)
            return {n: np.sort(np.asarray(order[:n], int)) for n in budgets}
        raise ValueError(f"알 수 없는 전략 {strat}")

    def coef_variant(self, kind, sel):
        """S8w·S8r 계수(같은 라벨, 계획 2.4 'S8a 의 라벨 + 설계 가중 계수 E_w = Σ w s y / Σ w s²', '계수만 바꿔 계산'): 수축하지 않은
        E_w·E_RE 를 P1·R1 의 앵커로 쓴다(κ 수축을 하지 않는다, impl_notes 3절). 비유한이면 E0 를 쓰고 표지를 단다.
        반환 (앵커 계수, 원시 계수(비유한이면 NaN), 정보)."""
        c = self.c
        if kind == "S8w":
            E_raw, info = coef_w(c.yA[sel], c.sA[sel], c.blkA[sel], self.Nb_of), {}
        elif kind == "S8r":
            E_raw, info = coef_re(c.yA[sel], c.sA[sel], c.blkA[sel])
        else:
            raise ValueError(kind)
        if np.isfinite(E_raw):
            return float(E_raw), float(E_raw), dict(info, flag="")
        return float(c.E0), float("nan"), dict(info, flag="coef_nonfinite_E0")


def region_traits(c) -> dict:
    """XD-6 서술용 지역 특성(A 풀): 블록 수, 유효 블록 수, 공변량 블록 사이 비율(라벨 미사용)과 셀 가중 E·블록 평균 E(셀 3개 이상 블록,
    A 라벨 전체). 판정·예측에는 쓰지 않는다."""
    XA = np.asarray(c.XA, np.float32)
    Z = W.standardize(XA, W.standardize_fit(XA))
    ub, inv = np.unique(np.asarray(c.blkA), return_inverse=True)
    Nb = np.bincount(inv, minlength=len(ub)).astype(float)
    p = Nb / max(Nb.sum(), 1.0)
    mu = Z.mean(0)
    tot = float(((Z - mu) ** 2).sum())
    betw = float(sum(Nb[j] * ((Z[inv == j].mean(0) - mu) ** 2).sum() for j in range(len(ub))))
    y, s, b = np.asarray(c.yA, float), np.asarray(c.sA, float), np.asarray(c.blkA)
    Eb = []
    for v in np.unique(b):
        m = (b == v)
        if m.sum() >= 3:
            Eb.append(H.ls_E(y[m], s[m]))
    Ec = H.ls_E(y, s)
    Ebm = float(np.nanmean(Eb)) if Eb else float("nan")
    return dict(n_blocks=int(len(ub)), eff_blocks=float(1.0 / max((p ** 2).sum(), 1e-12)), cov_between=betw / tot if tot > 0 else 0.0,
                E_cell=float(Ec), E_blockmean=Ebm,
                ln_cell_over_block=float(np.log(Ec / Ebm)) if (np.isfinite(Ec) and np.isfinite(Ebm) and Ec > 0 and Ebm > 0) else float("nan"))


class XDTUnit(_XDMixin, W.TUnit):
    """xd_t: 전이(모드 x) 배치 단위. 변형 = 전략. S1·S2·S4 는 h54.T8Unit 과 같은 계산(같은 함수·seed·행렬), 같은 라벨 집합이 다른 추출에서
    다시 나오면 적합을 재사용하고 그 추출 번호로 저장한다(T8Unit 규칙). S8a 조각은 S8w·S8r 의 P1·R1 을 더한다."""

    def __init__(self, a, c, alias, variant, dry=False, exp="xd_t"):
        if variant not in STRATS:
            raise ValueError(f"xd_t 전략은 {STRATS} 가운데 하나다: {variant}")
        W.TUnit.__init__(self, a, c, alias, dry, exp=exp, variant=variant)
        self._xd_init()

    def _r1(self, placement, sel, E, n, d, nl, nb, cache):
        c = self.c
        a1A, a1B = E * c.sA[sel], E * c.sB
        for seed in self.seeds:
            if seed in cache:
                g, fl = cache[seed]
            else:
                (g,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel, placement=placement)
                g, fl = np.asarray(g, float), self.F.last_flag
                cache[seed] = (g, fl)
            self.add("R1", LO, placement, n, d, seed, LAM, a1B + LAM * g, E, nl, nb, flag=fl, nrow=self.nsrc + nl)

    def run(self):
        c = self.c
        strat = self.variant
        budgets = [int(b) for b in self.a.G[self.exp] if 0 < int(b) < self.nA]
        if not budgets:
            return self
        seen: dict = {}
        for d in range(n_draws(self.a)):
            sets = self.xd_sets(strat, budgets, d)
            for n in budgets:
                sel = np.sort(np.asarray(sets[n], int))
                nl, nb = len(sel), self.nb(sel)
                key = (n, tuple(sel.tolist()))
                self.trace("select", sel, n=n, draw=str(d), strategy=strat)
                E1, _ = self.coefs(sel)
                self.trace("coef", sel, n=n, draw=str(d))
                self.add("P1", "none", strat, n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
                if key in seen:
                    self.notes.setdefault("dup_draws", []).append([int(n), int(d), int(seen[key][0])])
                else:
                    seen[key] = (d, {})
                cache = seen[key][1]
                self._r1(strat, sel, E1, n, d, nl, nb, cache.setdefault(strat, {}))
                if strat == COEF_BASE:
                    for cv in COEF_VARIANTS:
                        Ec, Eraw, info = self.coef_variant(cv, sel)
                        self.trace("coef", sel, n=n, draw=str(d), coef=cv)
                        self.add("P1", "none", cv, n, d, -1, 0.0, Ec * c.sB, Ec, nl, nb, flag=info.get("flag", ""))
                        self._r1(cv, sel, Ec, n, d, nl, nb, cache.setdefault(cv, {}))
                        if cv == "S8r" and not self.dry:
                            self.notes.setdefault("s8r", []).append([int(n), int(d), info.get("k"), info.get("df_within"), info.get("sigma2_src")])
        return self


class XDRUnit(_XDMixin, W.RUnit):
    """xd_r: 지역 내(모드 r) 배치 단위. 변형 = 전략. S1·S2·S4 는 h54.RUnit.run_wf2 와 같은 계산(P1, R1 λ 0.25·0.5·1.0, D1)이고
    S8 계열을 더한다. 같은 라벨 집합의 추출은 적합을 재사용해 그 추출 번호로 저장한다(WF8 규칙, 2단 CI 의 추출 재표집을 위해). S8a 조각은
    S8w·S8r 의 P1·R1 을 더한다(계획 2.4: 계수만 바꾸어 P1 닫힌 형식, R1 다시 적합)."""

    def __init__(self, a, c, variant, dry=False, exp="xd_r"):
        if variant not in STRATS:
            raise ValueError(f"xd_r 전략은 {STRATS} 가운데 하나다: {variant}")
        W.RUnit.__init__(self, a, c, exp, variant, dry)
        self._xd_init()

    def _r1d1(self, placement, sel, E, n, d, nl, nb, cache, with_d1=True):
        c = self.c
        a1A, a1B = E * c.sA[sel], E * c.sB
        for seed in self.seeds:
            if ("R1", seed) in cache:
                g, fl = cache[("R1", seed)]
            else:
                (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d, placement=placement)
                g, fl = np.asarray(g, float), self.F.last_flag
                cache[("R1", seed)] = (g, fl)
            self.emit("R1", LO, n, d, seed, a1B, g, E, nl, nb, flag=fl, placement=placement, nrow=nl)
            if with_d1:
                if ("D1", seed) in cache:
                    p, fl2 = cache[("D1", seed)]
                else:
                    (p,) = self.fit_direct(LO, "D1", sel, seed, n, d, pseudo_E=E, placement=placement)
                    p, fl2 = np.asarray(p, float), self.F.last_flag
                    cache[("D1", seed)] = (p, fl2)
                self.add("D1", LO, placement, n, d, seed, 1.0, p, E, nl, nb, flag=fl2, nrow=nl + int(round(XB.R_PS * nl)))

    def run(self):
        c = self.c
        strat = self.variant
        budgets = [int(b) for b in self.a.G[self.exp] if 0 < int(b) < self.nA]
        if not budgets:
            return self
        seen: dict = {}
        for d in range(n_draws(self.a)):
            sets = self.xd_sets(strat, budgets, d)
            for n in budgets:
                sel = np.sort(np.asarray(sets[n], int))
                nl, nb = len(sel), self.nb(sel)
                key = (n, tuple(sel.tolist()))
                self.trace("select", sel, n=n, draw=str(d), strategy=strat)
                E1, _ = self.coefs(sel)
                self.trace("coef", sel, n=n, draw=str(d))
                self.add("P1", "none", strat, n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
                if key in seen:
                    self.notes.setdefault("dup_draws", []).append([int(n), int(d), int(seen[key][0])])
                else:
                    seen[key] = (d, {})
                cache = seen[key][1]
                self._r1d1(strat, sel, E1, n, d, nl, nb, cache.setdefault(strat, {}))
                if strat == COEF_BASE:
                    for cv in COEF_VARIANTS:
                        Ec, Eraw, info = self.coef_variant(cv, sel)
                        self.trace("coef", sel, n=n, draw=str(d), coef=cv)
                        self.add("P1", "none", cv, n, d, -1, 0.0, Ec * c.sB, Ec, nl, nb, flag=info.get("flag", ""))
                        self._r1d1(cv, sel, Ec, n, d, nl, nb, cache.setdefault(cv, {}), with_d1=False)
        return self


class XDLUnit(_XDMixin, W.TUnit):
    """xd_learn: 학습 자료 단위(과제, 분할, n). 기준 S1 추출 d = 0–4(h40.draw_cells, WF8·XD-alg 의 S1 과 같다)와 라벨 집합 200개의
    R1(λ 0.25, seed 0)·P1 을 저장한다. 키의 placement = 'S1'(기준) 또는 'L<번호>'(집합). 집합 특징·색인은 sets 에 둔다(<조각>_sets.npz)."""

    def __init__(self, a, c, alias, n, dry=False, n_sets=N_SETS, exp="xd_learn"):
        W.TUnit.__init__(self, a, c, alias, dry, exp=exp, variant=f"n{int(n)}")
        self._xd_init()
        self.n = int(n)
        self.n_sets = int(n_sets)
        self.sets_out = None

    def _fit_one(self, placement, sel, n, d, cache):
        c = self.c
        sel = np.sort(np.asarray(sel, int))
        nl, nb = len(sel), self.nb(sel)
        self.trace("select", sel, n=n, draw=str(d), strategy=placement)
        E1, _ = self.coefs(sel)
        self.trace("coef", sel, n=n, draw=str(d))
        self.add("P1", "none", placement, n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
        seed = self.seeds[0]
        key = tuple(sel.tolist())
        if key in cache:
            g, fl = cache[key]
        else:
            a1A = E1 * c.sA[sel]
            (g,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel, placement=placement)
            g, fl = np.asarray(g, float), self.F.last_flag
            cache[key] = (g, fl)
        self.add("R1", LO, placement, n, d, seed, LAM, E1 * c.sB + LAM * g, E1, nl, nb, flag=fl, nrow=self.nsrc + nl)

    def run(self):
        c = self.c
        n = self.n
        if not (0 < n < self.nA):
            return self
        cache: dict = {}
        for d in range(n_draws(self.a)):
            self._fit_one("S1", self.draw(n, d), n, d, cache)
        Z = None if self.dry else self.zA()
        sets = learning_sets(Z, c.blkA, c.target, c.mode, c.split, n, self.n_sets, dry=self.dry)
        for j, (_, _, sel) in enumerate(sets):
            self._fit_one(f"L{j:03d}", sel, n, 0, cache)
        if self.dry:
            return self
        tf_ = TaskFeat.from_ctx(c)
        self.sets_out = dict(kind=np.array([k for k, _, _ in sets]), gen_seed=np.array([s for _, s, _ in sets], np.int64),
                             idx=np.vstack([s for _, _, s in sets]).astype(np.int32), feat=np.vstack([tf_.features(s) for _, _, s in sets]),
                             feat_names=np.array(FEATS), static=tf_.element_static(), ctx=tf_.ctx, a_fp=np.array(tf_.a_fp),
                             n=np.array(n), nA=np.array(self.nA))
        self.notes["n_unique_sets"] = int(len({tuple(s.tolist()) for _, _, s in sets}))
        return self


class XDXUnit(_XDMixin, W.TUnit):
    """xd_test(작업 R2b): 시험 과제(대상, 분할). S9* 의 선택(색인 파일, 정책 seed = 추출 번호, placement 'S9'), 같은 단위에서 다시 적합한
    S1(h40.draw_cells, 'S1')과 고정된 Algorithm P('AP', ap_sets: S8 후보는 일정 AP_SCHEDULE(10, 40, 160) = XC 와 같은 순서, seed 는 XD-alg
    규칙). P1·R1(λ 0.25), 학습 seed 0·1. 색인 파일의 |A| 와 A 지문(공변량)이 문맥과 다르면 중단한다."""

    def __init__(self, a, c, alias, entries, ap_name, dry=False, exp="xd_test"):
        W.TUnit.__init__(self, a, c, alias, dry, exp=exp, variant="")
        self._xd_init()
        self.entries = dict(entries)
        self.ap = str(ap_name)
        fp = a_fingerprint(c.XA, c.sA, c.blkA)
        for p, e in self.entries.items():
            if int(e["nA"]) != int(self.nA) or str(e["a_fp"]) != fp:
                raise SystemExit(f"[거부] 색인 파일의 A({alias} 분할 {c.split} 정책 seed {p})가 문맥과 다르다(|A| 또는 지문)")

    def _fit_set(self, placement, sel, n, d, cache):
        c = self.c
        sel = np.sort(np.asarray(sel, int))
        nl, nb = len(sel), self.nb(sel)
        self.trace("select", sel, n=n, draw=str(d), strategy=placement)
        E1, _ = self.coefs(sel)
        self.trace("coef", sel, n=n, draw=str(d))
        self.add("P1", "none", placement, n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
        key = (n, tuple(sel.tolist()))
        a1A, a1B = E1 * c.sA[sel], E1 * c.sB
        for seed in self.seeds:
            if (key, seed) in cache:
                g, fl = cache[(key, seed)]
            else:
                (g,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel, placement=placement)
                g, fl = np.asarray(g, float), self.F.last_flag
                cache[(key, seed)] = (g, fl)
            self.add("R1", LO, placement, n, d, seed, LAM, a1B + LAM * g, E1, nl, nb, flag=fl, nrow=self.nsrc + nl)

    def ap_sets(self, budgets, d):
        """고정된 Algorithm P 의 라벨 집합 {n: 정렬 색인}(XD-4 의 비교 상대). S8 후보 = placement_order(후보, Z, 블록, AP_SCHEDULE,
        seed_of('xd-S8', 대상, 모드, 분할, 추출))의 앞 n 개. 일정은 XC 의 Algorithm P 팔·계획 2.4 의사코드의 (10, 40, 160)이고 시험 격자
        (20, 40)가 아니다(S8 계열은 일정에 따라 순서가 달라진다). S2·S4 는 일정이 없으므로 XD-alg 과 같은 WF8 구현(xd_sets). S1 이면 None."""
        if self.ap == AP_NONE:
            return None
        if self.ap in S8_SPECS:
            c = self.c
            if self.dry:
                order = np.random.RandomState(int(d) + 23).permutation(self.nA)[:max(budgets)]
            else:
                order = placement_order(self.ap, self.zA(), c.blkA, AP_SCHEDULE, seed_of(S8_SEED_TAG, c.target, c.mode, int(c.split), int(d)))
            return {n: np.sort(np.asarray(order[:n], int)) for n in budgets}
        return self.xd_sets(self.ap, budgets, d)

    def run(self):
        budgets = [int(b) for b in self.a.G[self.exp] if 0 < int(b) < self.nA]
        if not budgets:
            return self
        cache: dict = {}
        ps = [p for p in POLICY_SEEDS[:n_draws(self.a)] if p in self.entries]
        for n in budgets:
            for p in ps:
                self._fit_set("S9", np.asarray(self.entries[p]["order"], int)[:n], n, p, cache)
        for d in range(n_draws(self.a)):
            ap_sets = self.ap_sets(budgets, d)
            for n in budgets:
                self._fit_set("S1", self.draw(n, d), n, d, cache)
                self._fit_set("AP", self.draw(n, d) if ap_sets is None else ap_sets[n], n, d, cache)
        return self


# ================================================================ 7. 탐욕 정책(S9)
def greedy_policy(tf_, score_fn, n_max, policy_seed, batch=POLICY_BATCH, ff_top=POLICY_FF_TOP, n_rand=POLICY_RAND) -> np.ndarray:
    """학습 정책의 탐욕 선택(계획 2.4). 첫 셀 = RandomState(seed_of('xd-pol0', 대상, 모드, 분할, 정책 seed)) 무작위(정책 seed 는 첫 셀만
    바꾼다). 그 뒤 5개 단위가 되도록 더한다(1 → 5 → 10 → …). 후보 = 현재 집합 기준 farthest-first 상위 300개(최소 거리 큰 순, 동률은 작은 색인)
    ∪ 고정 seed 무작위 200개(seed_of('xd-polr', 대상, 모드, 분할, 현재 크기), 정책 seed 와 무관). score_fn(tf, L, C) 는 L ∪ {c} 의 예측 효용
    (작을수록 좋다)을 내고, 작은 순(동률은 작은 색인)으로 그 단계의 수만큼 더한다. 반환 = 더한 순서(앞 n 개 = 크기 n 의 선택)."""
    N = tf_.N
    n_max = int(min(n_max, N))
    t, m, sp = tf_.target, tf_.mode, tf_.split
    c0 = int(np.random.RandomState(seed_of("xd-pol0", t, m, sp, int(policy_seed))).randint(N))
    L = [c0]
    taken = np.zeros(N, bool)
    taken[c0] = True
    mind = _dist_to(tf_.Z, tf_.Z[c0])
    mind[c0] = -1.0
    while len(L) < n_max:
        k = int(min(batch - (len(L) % batch), n_max - len(L)))
        free = np.where(~taken)[0]
        ff = free[np.lexsort((free, -mind[free]))][:int(ff_top)]
        rr = np.random.RandomState(seed_of("xd-polr", t, m, sp, len(L))).choice(N, min(int(n_rand), N), replace=False)
        rr = rr[~taken[rr]]
        cand = np.union1d(ff, rr).astype(int)
        sc = np.asarray(score_fn(tf_, np.asarray(L, int), cand), float)
        sc = np.where(np.isfinite(sc), sc, np.inf)
        pick = cand[np.lexsort((cand, sc))[:k]]
        for c_ in pick.tolist():
            L.append(int(c_))
            taken[c_] = True
            mind = np.minimum(mind, _dist_to(tf_.Z, tf_.Z[c_]))
            mind[c_] = -1.0
    return np.asarray(L, int)


def gbm_score_fn(model, threads=1):
    """S9-GBM 의 점수 함수: 집합 특징(features_batch) → CatBoost 예측 효용."""
    def f(tf_, L, C):
        return np.asarray(model.predict(tf_.features_batch(L, C), thread_count=int(threads)), float)
    return f


# ================================================================ 8. 학습 자료 적재·교차검증·S9-GBM
def cv_folds(tasks=DEV_TASKS) -> list:
    """조기 종료와 변형 선택의 교차검증 접기(계획 2.4): 'AL-k 와 알래스카 x 를 함께 제외'. 검증 = AL-k 과제, 학습 = 나머지 AL 과제
    (알래스카 x 는 어느 접기의 학습·검증에도 들어가지 않는다). 최종 적합은 개발 과제 전부."""
    tasks = [str(t) for t in tasks]
    out = []
    for k in [t for t in AL_SUBS if t in tasks]:
        out.append(dict(val=k, train=[t for t in tasks if t not in (k, "Alaska")], excluded=[k, "Alaska"]))
    return out


def assert_dev_test_disjoint(dev=DEV_TASKS, test=TEST_TASKS):
    """개발 과제와 시험 과제가 겹치지 않고 시험 과제의 계열이 알래스카가 아니다(과제 누설 방지)."""
    if set(dev) & set(test):
        raise AssertionError(f"개발·시험 과제가 겹친다: {sorted(set(dev) & set(test))}")
    if any(FAMILY_OF.get(t, "") == "Alaska" or str(t).startswith("AL-") for t in test):
        raise AssertionError("시험 과제에 알래스카 계열이 있다")
    return True


def load_learning(shards_dir, tag, tasks=DEV_TASKS, with_elements=False):
    """학습 자료 조각(<tag>__cpu__<과제>__x__s<분할>__n<값>) → 집합 표(과제, 분할, n, 번호, 종류, U_cell, U_beq, U, P1 효용, 특징)와
    (선택) 원소 자료 dict(static = {(과제, 분할): 셀별 고정 원소 특징}, ctx = {(과제, 분할): 문맥}, idx = {(과제, 분할, n): 집합 색인 행렬}). 효용 U = [RMSE(R1@0.25|L) − mean_d RMSE(R1@0.25|S1_d)] / mean_d RMSE(R1@0.25|S1_d)
    (셀 가중·블록 등가중 따로, 학습 목표 = 두 가중 평균, seed 0). 지정 과제의 조각만 연다. 봉인 조각 폴더(sealed/shards)는 읽지 않는다."""
    refuse_sealed_rows(shards_dir, "load_learning")
    rows, el = [], dict(static={}, ctx={}, idx={})
    sh = [s for s in XB.find_shards(shards_dir, tag) if s["target"] in set(tasks)]
    for s in sh:
        base = str(s["unit"])[:-len("_unit.json")]
        sp_path = Path(base + "_sets.npz")
        if not sp_path.exists():
            continue
        runs = XB.read_runs([s])
        if not len(runs):
            continue
        z = np.load(sp_path, allow_pickle=False)
        n = int(z["n"])
        r1 = runs[(runs.method == "R1") & (runs.n.astype(int) == n)]
        p1 = runs[(runs.method == "P1") & (runs.n.astype(int) == n)]
        ref1 = r1[r1.placement == "S1"]
        refp = p1[p1.placement == "S1"]
        if not len(ref1):
            continue
        rc, rb = float(ref1.rmse_cm.mean()), float(ref1.rmse_beq_cm.mean())
        pc, pb = float(refp.rmse_cm.mean()), float(refp.rmse_beq_cm.mean())
        by1 = {str(p): (float(a), float(b)) for p, a, b in zip(r1.placement, r1.rmse_cm, r1.rmse_beq_cm)}
        byp = {str(p): (float(a), float(b)) for p, a, b in zip(p1.placement, p1.rmse_cm, p1.rmse_beq_cm)}
        F = np.asarray(z["feat"], float)
        for j in range(len(z["kind"])):
            pl = f"L{j:03d}"
            if pl not in by1:
                continue
            uc, ub = (by1[pl][0] - rc) / rc, (by1[pl][1] - rb) / rb
            if not (np.isfinite(uc) and np.isfinite(ub)):
                continue
            qc = (byp[pl][0] - pc) / pc if pl in byp else np.nan
            qb = (byp[pl][1] - pb) / pb if pl in byp else np.nan
            rows.append(dict(task=s["target"], mode=s["mode"], split=int(s["split"]), n=n, j=j, kind=str(z["kind"][j]), U_cell=uc, U_beq=ub,
                             U=0.5 * (uc + ub), U_p1_cell=qc, U_p1_beq=qb, **{f: float(v) for f, v in zip(FEATS, F[j])}))
        if with_elements:
            el["static"][(s["target"], int(s["split"]))] = np.asarray(z["static"], np.float32)
            el["ctx"][(s["target"], int(s["split"]))] = np.asarray(z["ctx"], float)
            el["idx"][(s["target"], int(s["split"]), n)] = np.asarray(z["idx"], int)
    df = pd.DataFrame(rows)
    return (df, el) if with_elements else df


def cv_utility(df_val, pred) -> float:
    """개발 교차검증 효용: 검증 묶음(과제, 분할, n)마다 예측 효용이 가장 작은 집합(동률은 작은 번호)의 실제 효용 U(두 가중 평균)의 묶음 평균.
    탐욕 정책의 새 선택은 적합 없이 채점할 수 없으므로 후보 200개 안의 선택으로 잰다."""
    d = df_val.assign(_p=np.asarray(pred, float))
    vals = []
    for _, g in d.groupby(["task", "split", "n"], sort=True):
        i = np.lexsort((g["j"].values, g["_p"].values))[0]
        vals.append(float(g.U.values[i]))
    return float(np.mean(vals)) if vals else float("nan")


def _cb_fit(Xtr, ytr, iters, seed, threads):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=int(iters), random_seed=int(seed), verbose=0, allow_writing_files=False, thread_count=int(threads), **GBM_CFG)
    m.fit(np.asarray(Xtr, float), np.asarray(ytr, float))
    return m


def train_gbm(df, threads=4, iters_grid=GBM_ITERS, seed=GBM_SEED):
    """S9-GBM(계획 2.4): CatBoost 회귀(CPU), 목표 U(두 가중 평균). 반복 수(조기 종료)는 알래스카 하위 지역 하나 제외 교차검증의 효용으로
    고르고(동률은 작은 반복 수), 최종 적합은 개발 과제 전부. 반환 (최종 모형, 교차검증 기록 목록, 고른 반복 수, 접기 기록)."""
    folds = cv_folds(sorted(df.task.unique()))
    if not folds:
        raise SystemExit("[S9-GBM] 교차검증 접기가 없다(알래스카 하위 과제가 없다)")
    rep, fold_log = [], []
    for it in iters_grid:
        us = []
        for f in folds:
            tr, va = df[df.task.isin(f["train"])], df[df.task == f["val"]]
            if not len(tr) or not len(va):
                continue
            m = _cb_fit(tr[FEATS].values, tr.U.values, it, seed, threads)
            us.append(cv_utility(va, m.predict(va[FEATS].values, thread_count=int(threads))))
            fold_log.append(dict(iterations=int(it), val=f["val"], train=list(f["train"]), n_train=int(len(tr)), n_val=int(len(va))))
        rep.append(dict(iterations=int(it), cv_utility=float(np.mean(us)) if us else float("nan"), per_fold=[float(v) for v in us]))
    ok = [r for r in rep if np.isfinite(r["cv_utility"])]
    best = min(ok, key=lambda r: (r["cv_utility"], r["iterations"]))
    final = _cb_fit(df[FEATS].values, df.U.values, best["iterations"], seed, threads)
    return final, rep, int(best["iterations"]), fold_log


# ================================================================ 9. 정책 색인 파일과 동결
def task_feats_for(a, tasks, splits=None):
    """과제 목록의 (별칭, 분할) → TaskFeat(라벨 미사용, 공변량만). 분할은 h40 규칙의 유효 분할(1–5)."""
    out = {}
    for alias in tasks:
        _, D = XB.get_data(a, alias)
        tgt = XB.resolve(alias)[1]
        keep, _, _ = W.split_plan(D, tgt, a.SPLITS)
        for sp in keep:
            if splits is not None and int(sp) not in set(int(v) for v in splits):
                continue
            c = XB.build_tctx(a, alias, "x", sp)
            out[(alias, int(sp))] = TaskFeat.from_ctx(c)
    return out


def export_index(tfs, score_fn, policy, model_sha, path, seeds=POLICY_SEEDS, n_max=POLICY_NMAX, allowed=None, extra=None) -> str:
    """정책 색인 파일(JSON + sha256 사이드카). 항목 = (별칭, 모드, 분할, 정책 seed, |A|, A 지문, 더한 순서). 라벨을 쓰지 않는다."""
    entries = []
    for (alias, sp), tf_ in sorted(tfs.items()):
        for p in seeds:
            order = greedy_policy(tf_, score_fn, n_max, p)
            entries.append(dict(alias=alias, target=tf_.target, mode=tf_.mode, split=int(sp), policy_seed=int(p), nA=int(tf_.N), a_fp=tf_.a_fp,
                                order=[int(v) for v in order]))
    obj = dict(format=INDEX_FORMAT, policy=str(policy), model_sha256=str(model_sha), n_max=int(n_max), batch=POLICY_BATCH, ff_top=POLICY_FF_TOP,
               n_rand=POLICY_RAND, policy_seeds=[int(v) for v in seeds], created=time.strftime("%Y-%m-%d %H:%M:%S"), code_sha256=_sha256(__file__),
               plan_commit=XB.PLAN_COMMIT, entries=entries, **(extra or {}))
    return _write_json(path, obj, allowed)


class IndexFile:
    """정책 색인 파일 적재(사이드카 sha256 대조, 동결 기록이 있으면 그 sha256 과도 대조)."""

    def __init__(self, path, freeze_manifest=None):
        self.path = Path(path)
        self.sha256 = verify_sidecar(self.path)
        if freeze_manifest is not None:
            fz = load_freeze(freeze_manifest)
            if fz.get("index_sha256") != self.sha256:
                raise SystemExit("[거부] 색인 파일의 sha256 이 동결 기록의 값과 다르다(XD-4 는 동결된 선택만 평가한다)")
            self.freeze = fz
        else:
            self.freeze = None
        obj = json.loads(self.path.read_text())
        if obj.get("format") != INDEX_FORMAT:
            raise SystemExit(f"[거부] 색인 파일 형식이 {INDEX_FORMAT} 가 아니다")
        self.policy = obj.get("policy", "")
        self.n_max = int(obj.get("n_max", 0) or 0)
        self.by = {}
        for e in obj["entries"]:
            self.by.setdefault((str(e["alias"]), str(e["mode"]), int(e["split"])), {})[int(e["policy_seed"])] = e

    def entries_for(self, alias, mode, split) -> dict:
        return dict(self.by.get((str(alias), str(mode), int(split)), {}))


def index_coverage(index, expected, seeds, n_need) -> dict:
    """색인 파일이 xd_test 의 모든 (별칭, 유효 분할)과 정책 seed 를 덮는가(시작 전 확인). expected = enumerate_units 의 기대 분할
    {(실험, 별칭, 모드): 분할 목록}. 반환 dict(missing = [(별칭, 분할, 사유)], counted = 시험 과제(TEST_TASKS)의 빠진 항목,
    desc = 그 밖(서술 과제 Russia_C~lgd 등)의 빠진 항목, short = 색인 n_max 가 시험 격자 최댓값보다 작음)."""
    missing = []
    for (exp, alias, mode), keep in sorted(expected.items()):
        if exp != "xd_test":
            continue
        for sp in keep:
            ent = index.entries_for(alias, "x", sp) if index is not None else {}
            lack = [int(p) for p in seeds if int(p) not in ent]
            if not ent:
                missing.append((alias, int(sp), "항목 없음"))
            elif lack:
                missing.append((alias, int(sp), f"정책 seed {lack} 없음"))
    counted = [m for m in missing if m[0] in TEST_TASKS]
    desc = [m for m in missing if m[0] not in TEST_TASKS]
    short = bool(index is not None and index.n_max and int(index.n_max) < int(n_need))
    return dict(missing=missing, counted=counted, desc=desc, short=short)


def rebase_path(p) -> Path:
    """기록에 적힌 경로를 이 저장소 뿌리에 맞춘다(다른 기계의 절대 경로이면 'data/processed/' 뒤를 ROOT 에 붙인다. Rescale 묶음용)."""
    p = Path(str(p))
    if p.exists() or not p.is_absolute():
        return p if p.is_absolute() else ROOT / p
    s = str(p).replace(os.sep, "/")
    k = s.find("data/processed/")
    return ROOT / s[k:] if k >= 0 else p


def load_freeze(path) -> dict:
    """S9* 동결 기록 적재(사이드카 sha256 대조)."""
    verify_sidecar(path)
    return json.loads(Path(path).read_text())


def git_available() -> bool:
    """이 저장소가 git 작업 트리 안인가(Rescale 묶음에는 .git 이 없다)."""
    return XB._git("rev-parse", "--is-inside-work-tree") == "true"


def freeze_commit_state(path) -> dict:
    """동결 기록이 git 에 커밋되어 있고(ls-files) HEAD 대비 수정되지 않았는가(diff --quiet HEAD, 색인·작업 트리 모두). 계획 0.3 'XD-4 는
    정책 동결을 개정 이력에 적고 커밋한 뒤에만'. 반환 dict(ok, tracked, clean, commit, rel)."""
    import subprocess
    real, root = os.path.realpath(str(path)), os.path.realpath(str(ROOT))
    rel = os.path.relpath(real, root)
    if rel.startswith(".."):
        return dict(ok=False, tracked=False, clean=False, commit="", rel=rel, why="저장소 밖 경로")

    def rc(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60).returncode
        except Exception:                                                 # noqa: BLE001
            return 99
    tracked = rc("ls-files", "--error-unmatch", "--", rel) == 0
    clean = tracked and rc("diff", "--quiet", "HEAD", "--", rel) == 0
    commit = XB._git("log", "-1", "--format=%H", "--", rel) if tracked else ""
    ok = bool(tracked and clean and commit)
    why = "" if ok else ("git 에 없다(커밋 전)" if not tracked else ("HEAD 대비 수정되었다" if not clean else "커밋 기록 없음"))
    return dict(ok=ok, tracked=tracked, clean=clean, commit=commit, rel=rel, why=why)


def attestation_path(manifest) -> Path:
    """동결 기록의 커밋 확인 기록 경로(<동결 기록 이름>_commit.json, 같은 폴더)."""
    p = Path(manifest)
    return p.with_name(p.stem + "_commit.json")


def write_freeze_attestation(manifest, allowed=None, state=None) -> dict:
    """로컬(git 작업 트리)에서 동결 기록의 커밋을 확인하고 확인 기록(+ sha256 사이드카)을 쓴다. R2b 묶음(git 없음)이 이 기록으로 대조한다.
    커밋되지 않았거나 수정되었으면 거부한다."""
    msha = verify_sidecar(manifest)
    st = freeze_commit_state(manifest) if state is None else dict(state)
    if not st.get("ok"):
        raise SystemExit(f"[거부] 동결 기록 {Path(manifest).name}: {st.get('why', '커밋 확인 실패')}(커밋 뒤에 다시 한다)")
    obj = dict(manifest=str(st.get("rel", "")), manifest_sha256=msha, commit=st.get("commit", ""), checked=time.strftime("%Y-%m-%d %H:%M:%S"),
               code_sha256=_sha256(__file__), plan_commit=XB.PLAN_COMMIT)
    sha = _write_json(attestation_path(manifest), obj, allowed)
    print(f"[동결 확인] 커밋 {str(obj['commit'])[:12]} · 기록 sha256 {msha[:16]} · 확인 기록 sha256 {sha[:16]}", flush=True)
    return obj


def require_freeze_committed(manifest) -> dict:
    """xd_test 본 실행의 전제(계획 0.3): 동결 기록이 커밋되어 있고 수정되지 않았다. git 작업 트리이면 직접 확인하고, git 이 없으면(Rescale 묶음)
    확인 기록(attestation_path)의 사이드카와 동결 기록 sha256 을 대조한다. 실패하면 SystemExit."""
    if git_available():
        st = freeze_commit_state(manifest)
        if not st["ok"]:
            raise SystemExit(f"[거부] xd_test: 동결 기록 {Path(manifest).name} 이 {st['why']}. 계획 0.3: 동결을 개정 이력에 적고 커밋한 뒤에만 "
                             f"시험 과제를 평가한다")
        return dict(st, source="git")
    att = attestation_path(manifest)
    if not att.exists():
        raise SystemExit(f"[거부] xd_test: git 이 없고 커밋 확인 기록 {att.name} 도 없다(로컬에서 --attest-freeze 뒤 묶음에 넣는다)")
    verify_sidecar(att)
    obj = json.loads(att.read_text())
    if obj.get("manifest_sha256") != verify_sidecar(manifest):
        raise SystemExit("[거부] xd_test: 커밋 확인 기록의 동결 기록 sha256 이 지금 동결 기록과 다르다")
    return dict(ok=True, commit=obj.get("commit", ""), source="attestation")


def require_xd4_sealed(out_dir) -> Path:
    """XD-5 의 전제(계획 0.3 열람 순서 4): XD-4 시험 표(sealed/xd4_summary.json, 접미사 없음)가 있고 봉인 기록(sealed_manifest.json)에
    항목이 있다. 표 자체는 열지 않는다(존재와 봉인 기록 항목만 본다). 없으면 SystemExit."""
    out = Path(out_dir) if os.path.isabs(str(out_dir)) else ROOT / str(out_dir)
    sd = XB.sealed_dir(EXP_NAME, out.parent)
    p, man = sd / XD4_SEALED, sd / "sealed_manifest.json"
    entries = []
    if man.exists():
        try:
            entries = json.loads(man.read_text()).get("entries", [])
        except (OSError, ValueError):
            entries = []
    if not p.exists() or not any(str(e.get("file")) == XD4_SEALED for e in entries):
        raise SystemExit(f"[거부] --xd5: XD-4 시험 표({XD4_SEALED})와 봉인 기록 항목이 없다. XD-5 자료는 XD-4 시험 표를 연 뒤에만 만든다(계획 0.3 열람 순서 4)")
    return p


def load_selection(path) -> dict:
    """Algorithm P 선택 파일 적재(사이드카 sha256 대조). 반환 dict(chosen, spec, sha256, …)."""
    sha = verify_sidecar(path)
    obj = json.loads(Path(path).read_text())
    obj["sha256"] = sha
    if obj.get("chosen") not in AP_CANDIDATES + (AP_NONE,):
        raise SystemExit(f"[거부] 선택 파일의 후보 {obj.get('chosen')} 가 등록 후보가 아니다")
    return obj


def gpu_driver_info() -> dict:
    """nvidia-smi 의 드라이버 판과 GPU 목록(동결 기록용). 없으면 빈 값."""
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"], text=True, timeout=30)
        rows = [r.strip() for r in out.strip().splitlines() if r.strip()]
        return dict(driver=rows[0].split(",")[-1].strip() if rows else "", gpus=rows)
    except Exception:                                                     # noqa: BLE001
        return dict(driver="", gpus=[])


def freeze(gbm_meta, ds_meta, out_path, allowed=None) -> dict:
    """S9* 동결 기록(계획 2.4 '동결'): 두 변형 가운데 개발 교차검증 효용이 작은 하나(동률은 S9-GBM). 한 변형의 기록이 없으면 있는 변형으로
    정하고 그 사실을 적는다. 기록 = 모형 해시, torch·CUDA·드라이버 판, GPU 번호, seed, 특징 목록, 색인 파일 해시. 화면에는 고른 변형 이름과
    파일 해시만 쓴다(효용 값은 쓰지 않는다)."""
    cand = []
    for nm, p in (("S9-GBM", gbm_meta), ("S9-DS", ds_meta)):
        if p is None or not Path(p).exists():
            continue
        verify_sidecar(p)
        m = json.loads(Path(p).read_text())
        if not np.isfinite(float(m.get("cv_utility", np.nan))):
            continue
        cand.append((nm, m, Path(p)))
    if not cand:
        raise SystemExit("[동결] 교차검증 효용이 있는 변형 기록이 없다")
    order = {v: i for i, v in enumerate(VARIANTS_S9)}
    nm, m, p = min(cand, key=lambda t: (float(t[1]["cv_utility"]), order[t[0]]))
    idx = rebase_path(m["index_path"])
    man = dict(s9_star=nm, chosen_by="개발 교차검증 효용(AL-k 와 알래스카 x 함께 제외)이 작은 변형", variants_available=[c[0] for c in cand],
               cv_utility={c[0]: float(c[1]["cv_utility"]) for c in cand}, meta_path=str(p), meta_sha256=_sha256(p),
               model_path=m.get("model_path"), model_sha256=m.get("model_sha256"), index_path=str(idx), index_sha256=verify_sidecar(idx),
               features=m.get("features"), seeds=m.get("seeds"), torch=m.get("torch"), cuda=m.get("cuda"), cudnn=m.get("cudnn"),
               cublas=m.get("cublas"), deterministic=m.get("deterministic"), device_order=m.get("device_order"),
               gpu=m.get("gpu"), gpu_name=m.get("gpu_name"), driver=m.get("driver") or gpu_driver_info().get("driver"), catboost=m.get("catboost"),
               created=time.strftime("%Y-%m-%d %H:%M:%S"), plan_commit=XB.PLAN_COMMIT, code_sha256=_sha256(__file__),
               index_relpath=os.path.relpath(os.path.realpath(str(idx)), os.path.realpath(str(ROOT))),
               note="이 기록을 계획 개정 이력에 적고 커밋한 뒤에만 작업 R2b(XD-4 시험)를 제출한다(커밋 뒤 --attest-freeze 로 확인 기록을 만들어 "
                    "R2b 묶음에 넣는다). 비알래스카 XD-alg 행(sealed/xd_alg_*, sealed/shards)은 그 뒤에 연다")
    sha = _write_json(out_path, man, allowed)
    print(f"[동결] S9* = {nm} · 기록 {out_path} · sha256 {sha[:16]}", flush=True)
    return man


# ================================================================ 10. Algorithm P 선택(알래스카 계열만)
def load_tms_filtered(shards_dir, tag, keep, nboot, allow_mixed=False):
    """조각 가운데 (대상, 모드) ∈ keep 인 것만 열어 {저장소 이름: TMx} 를 만든다. 다른 조각은 파일 이름만 보고 열지 않는다.
    봉인 조각 폴더(sealed/shards)는 읽지 않는다."""
    refuse_sealed_rows(shards_dir, "Algorithm P 선택")
    keep = {(str(t), str(m)) for t, m in keep}
    sh = [s_ for s_ in XB.find_shards(shards_dir, tag) if (s_["target"], s_["mode"]) in keep]
    if not sh:
        return {}, []
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    XB.check_cfg(units, allow_mixed)
    stores = XB.load_stores([s_["npz"] for s_ in sh])
    by = {}
    for (nm, sp), st in stores.items():
        by.setdefault(nm, {})[int(sp)] = st
    tms = {nm: W.make_tm(nm, bs, W.units_for(units, nm), int(nboot), XB.is_point_only_name(nm)) for nm, bs in sorted(by.items())}
    return tms, [str(s_["npz"].name) for s_ in sh]


def _ci2(s):
    lo, hi = X._ci(s.get("dist")) if s else (np.nan, np.nan)
    lob, hib = X._ci(s.get("dist_beq")) if s else (np.nan, np.nan)
    return lo, hi, lob, hib


def alaska_rows(tms_x, tms_r, nboot=None) -> pd.DataFrame:
    """알래스카 계열 행: 후보 S_k − S1(R1@0.25)의 2단 CI 와 상대 변화(두 가중). 전이 n 10·40, 지역 내 n 20–500(|A| 미만, 행이 있는 n)."""
    rows = []
    for arm, tms, fam, ns in (("x", tms_x, ALASKA_X, ALASKA_X_N), ("r", tms_r, ALASKA_R, ALASKA_R_N)):
        for t in fam:
            nm = f"{t}|{arm}"
            tm = tms.get(nm)
            if tm is None:
                continue
            for n in ns:
                gB = W.gk("R1", n, LO, LAM, "S1")
                rbc, rbb = XB.grp_rmse2(tm, gB)
                if not np.isfinite(rbc):
                    continue
                for cand in AP_CANDIDATES:
                    gA = W.gk("R1", n, LO, LAM, cand)
                    rac, rab = XB.grp_rmse2(tm, gA)
                    if not np.isfinite(rac):
                        continue
                    s = XB.region_stat_two_stage(tm, gA, gB, nboot=nboot)
                    lo, hi, lob, hib = _ci2(s if (s and s.get("ok")) else None)
                    excl = bool((np.isfinite(lo) and lo > 0) or (np.isfinite(lob) and lob > 0))
                    rows.append(dict(arm=arm, target=t, store=nm, n=int(n), cand=cand, rmse_cand_cell=rac, rmse_s1_cell=rbc, rmse_cand_beq=rab,
                                     rmse_s1_beq=rbb, rel_cell=(rac - rbc) / rbc, rel_beq=(rab - rbb) / rbb,
                                     delta_cell=s["delta"] if s else np.nan, delta_beq=s["delta_beq"] if s else np.nan,
                                     ci_lo=lo, ci_hi=hi, ci_lo_beq=lob, ci_hi_beq=hib, has_ci=bool(s and s.get("ok")),
                                     n_splits=int(s["n_splits"]) if s else 0, excl_row=excl))
            XB.boot_cache_clear()
    return pd.DataFrame(rows)


def choose_algorithm_p(tab) -> dict:
    """계획 2.3 의 고정 규칙(기계적): 제외 → 점수 → 최소(동률 순서) → 모두 빠지면 S1."""
    cands = {}
    for cand in AP_CANDIDATES:
        q = tab[tab.cand == cand] if len(tab) else tab
        excluded = bool(len(q) and q.excl_row.any())
        qx = q[q.arm == "x"] if len(q) else q
        per = []
        for t, g in (qx.groupby("target", sort=True) if len(qx) else []):
            vc, vb = float(g.rel_cell.mean()), float(g.rel_beq.mean())
            per.append(dict(target=t, rel_cell_mean=vc, rel_beq_mean=vb, worse=max(vc, vb), n_list=[int(v) for v in sorted(g.n.unique())]))
        score = float(np.mean([p["worse"] for p in per])) if per else float("nan")
        cands[cand] = dict(excluded=excluded, n_rows=int(len(q)), n_rows_excl=int(q.excl_row.sum()) if len(q) else 0,
                           excl_rows=[f"{r.store} n{r.n}" for r in q[q.excl_row].itertuples()] if len(q) else [], score=score, per_target=per)
    ok = [c for c in AP_CANDIDATES if not cands[c]["excluded"] and np.isfinite(cands[c]["score"])]
    if not ok:
        return dict(chosen=AP_NONE, reason="모든 후보가 제외되었다(또는 점수 없음): Algorithm P = S1. XC-5b·5c 와 S-XC3 은 시험하지 않음",
                    candidates=cands)
    best = min(cands[c]["score"] for c in ok)
    tied = [c for c in ok if cands[c]["score"] - best < AP_TIE_TOL]
    chosen = min(tied, key=lambda c: AP_TIE_ORDER.index(c))
    return dict(chosen=chosen, reason="제외되지 않은 후보 가운데 점수 최소" + (f"(동률 {len(tied)}개, 순서 규칙)" if len(tied) > 1 else ""),
                candidates=cands)


def selection_fn_sha256() -> str:
    """선택 규칙 함수(select_algorithm_p, choose_algorithm_p, alaska_rows)의 원문 sha256(부록 XC-0 에 적는 '선택 스크립트 해시'의 보조값)."""
    src = inspect.getsource(select_algorithm_p) + inspect.getsource(choose_algorithm_p) + inspect.getsource(alaska_rows)
    return hashlib.sha256(src.encode()).hexdigest()


def select_algorithm_p(shards_dir, tag_t, tag_r, out_dir, nboot=XB.NBOOT, allowed=None, sealed=False, suffix="", sealed_root=None) -> dict:
    """Algorithm P 의 기계적 선택(계획 2.3, 0.3 열람 순서 1). 알래스카 계열 조각만 연다(봉인하지 않는 폴더 shards/ 에서 파일 이름으로 거른 뒤
    연다. 비알래스카 xd_t·xd_r 조각은 sealed/shards 에 있고 이 함수는 그 폴더를 거부한다). 선택 파일(JSON)과
    알래스카 계열 표(CSV), 각 sha256 사이드카를 out_dir 에 쓴다. 화면에는 저장소 수, 행 수, sha256 앞 16자, 고른 후보 이름만 쓴다.
    sealed 이면(스모크) 두 파일을 봉인 폴더에 쓰고(write_sealed, root = sealed_root) 고른 후보도 화면에 쓰지 않는다."""
    keep_x = [(t, "x") for t in ALASKA_X]
    keep_r = [(t, "r") for t in ALASKA_R]
    tms_x, files_x = load_tms_filtered(shards_dir, tag_t, keep_x, nboot)
    tms_r, files_r = load_tms_filtered(shards_dir, tag_r, keep_r, nboot)
    tab = alaska_rows(tms_x, tms_r, nboot)
    res = choose_algorithm_p(tab)
    tname, jname = f"algorithm_p_alaska_table{suffix}.csv", f"algorithm_p_selection{suffix}.json"
    obj = dict(chosen=res["chosen"], spec=ap_spec(res["chosen"]), reason=res["reason"], candidates=res["candidates"], rule="계획 2.3 Algorithm P 의 고정 규칙",
               alaska_family=dict(transfer=[f"{t}|x" for t in ALASKA_X], inregion=[f"{t}|r" for t in ALASKA_R], n_x=list(ALASKA_X_N), n_r=list(ALASKA_R_N)),
               stores_read=sorted(tms_x) + sorted(tms_r), shard_files=files_x + files_r, nboot=int(nboot), table_file=tname,
               script_sha256=_sha256(__file__), selection_fn_sha256=selection_fn_sha256(), core_sha256=_sha256(XB.__file__),
               frozen=dict(XB.FROZEN_SHA16), plan_commit=XB.PLAN_COMMIT, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if sealed:
        recs = XB.write_sealed(EXP_NAME, {tname: tab, jname: obj}, root=sealed_root, allowed=allowed)
        return dict(obj, path=str(XB.sealed_dir(EXP_NAME, sealed_root) / jname), sha256=recs[-1]["sha256"], sealed=True)
    tpath, jpath = Path(out_dir) / tname, Path(out_dir) / jname
    tsha = _write_csv(tpath, tab, allowed)
    obj.update(table_path=str(tpath), table_sha256=tsha)
    jsha = _write_json(jpath, obj, allowed)
    print(f"[선택] 알래스카 계열 저장소 {len(tms_x) + len(tms_r)}개 · 표 행 {len(tab)} · 표 sha256 {tsha[:16]} · 선택 파일 {jpath.name} sha256 {jsha[:16]}",
          flush=True)
    print(f"[선택] Algorithm P = {res['chosen']}", flush=True)
    return dict(obj, path=str(jpath), sha256=jsha, sealed=False)


def select_default(out_dir, allowed=None, suffix="") -> dict:
    """마감(R1a 제출 + 48 h) 초과: Algorithm P = P_default(S8a, γ 0)(계획 2.3 규칙 6)."""
    obj = dict(chosen=AP_DEFAULT, spec=ap_spec(AP_DEFAULT), reason="XD-alg 결과가 R1a 제출 뒤 48 시간 안에 나오지 않았다: P_default", candidates={},
               script_sha256=_sha256(__file__), plan_commit=XB.PLAN_COMMIT, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = Path(out_dir) / f"algorithm_p_selection{suffix}.json"
    sha = _write_json(p, obj, allowed)
    print(f"[선택] P_default 기록 {p.name} sha256 {sha[:16]}", flush=True)
    return dict(obj, path=str(p), sha256=sha)


# ================================================================ 11. 집계(봉인)
TEMPLATES = {
    "XD-1": {"우세": "블록 배분과 공변량 덮임을 합친 배치는 전이 조건에서 무작위보다 라벨 {n}개에서 오차가 {a} cm 작았다({m}).",   # m = 풀 표지(지역 5/5, 부분(지역 2/5))
             "열세": "블록 배분과 공변량 덮임을 합친 배치는 전이 조건에서 무작위보다 라벨 {n}개에서 오차가 {a} cm 컸다.",
             "동등": "블록 배분과 공변량 덮임을 합친 배치는 전이 조건에서 무작위와 라벨 {n}개에서 0.5 cm 안에서 같았다.",
             "미결정": "블록 배분과 공변량 덮임을 합친 배치와 무작위의 차이를 라벨 {n}개에서 확인하지 못했다.",
             "판정 불가": "블록 배분과 공변량 덮임을 합친 배치와 무작위의 차이를 라벨 {n}개에서 판정할 수 없었다."},
    "XD-2": {"지지": "설계 가중 계수는 레나에서 블록 균등 배치의 손해를 줄였다(사후 설계).",
             "열세": "설계 가중 계수는 지역 {x} 에서 오차를 키웠다.",
             "동등": "설계 가중 계수와 기본 계수는 0.5 cm 안에서 같았다.",
             "기각": "계수 추정 방식으로 레나형 손해를 줄이는 것을 확인하지 못했다.",
             "판정 불가": "판정할 수 없었다."},
    "XD-3": {"비열등": "S8 변형 {v} 의 오차 증가는 {ref} 대비 0.5 cm 안이었다(라벨 {n}개).",
             "열세": "S8 변형 {v} 의 오차는 {ref} 대비 라벨 {n}개에서 컸다.",
             "동등": "S8 변형 {v} 와 {ref} 는 라벨 {n}개에서 0.5 cm 안에서 같았다.",
             "미결정": "S8 변형 {v} 의 {ref} 대비 비열등을 확인하지 못했다(라벨 {n}개).",
             "판정 불가": "S8 변형 {v} 의 {ref} 대비 비열등을 라벨 {n}개에서 판정할 수 없었다."},   # 1절 다섯 갈래의 판정 불가 문장
    "XD-4": "알래스카 하위 지역에서 학습해 동결한 배치 정책은 알래스카 밖 시험 과제 5개 가운데 {k}개(계열 3개 가운데 {j}개)에서 Algorithm P 보다 "
            "오차가 작았다(두 가중 모두, 검정 없음).",
    "XD-4-worse": "알래스카 하위 지역에서 학습해 동결한 배치 정책은 알래스카 밖 시험 과제 5개 가운데 {k}개(계열 3개 가운데 {j}개)에서 Algorithm P 보다 "
                  "오차가 컸다(두 가중 모두, 검정 없음).",
    "XD-4-none": "학습 정책은 시험하지 않았다.",
    "공통": "후보 풀은 이미 측정한 셀이므로 현장 후보(육지 격자 전체)로 이득이 옮겨진다는 보장이 없다.",
}


def _stat(kind, tm, gA, gB, nboot=None):
    return XB.region_stat_same(tm, gA, gB) if kind == "same" else XB.region_stat_two_stage(tm, gA, gB, nboot=nboot)


def _methods_of(arm):
    return (("R1", LO, LAM), ("P1", "none", None)) + ((("D1", LO, 1.0),) if arm == "r" else ())


def _g(method, n, lr, lam, pl):
    return W.gk(method, n, lr, lam, pl) if lr != "none" else W.gk(method, n, "none", placement=pl)


def contrast_specs(arm):
    """(이름, 종류, A 배치, B 배치) 목록: S_k − S1(2단), S8 변형 − S2·S4(2단), S8w·S8r − S8a(같은 라벨), S8w·S8r − S1(2단, 서술)."""
    out = [(f"{s}-S1", "two_stage", s, "S1") for s in STRATS[1:] + COEF_VARIANTS]
    out += [(f"{v}-{ref}", "two_stage", v, ref) for v in S8_SPECS for ref in ("S2", "S4")]
    out += [(f"{cv}-{COEF_BASE}", "same", cv, COEF_BASE) for cv in COEF_VARIANTS]
    return out


def _row(base, r, scope) -> dict:
    """표 행: 풀 함수의 행(분포 키 제외) + 대비 표지(base) + 범위(scope: region, PE1, PE2, AUX3, INREGION3)."""
    d = {k: v for k, v in r.items() if not str(k).startswith("_")}
    d.update(base)
    d["scope"] = scope
    return d


def alg_contrast_rows(tms, arm, nboot=None) -> tuple:
    """한 팔의 모든 대비 행(지역 행 + PE1·PE2·3지역 풀 행). 반환 (행 목록, {(대비, 방법, n, 풀): 풀 행})."""
    rows, pools = [], {}
    names = sorted(tms)
    for method, lr, lam in _methods_of(arm):
        ns = sorted({n for nm in names for n in W.ns_of(tms[nm], method, None if lr == "none" else lr, "S1")}, key=lambda v: (v == -1, v))
        for cname, kind, pa, pb in contrast_specs(arm):
            if method == "D1" and (pa in COEF_VARIANTS or pb in COEF_VARIANTS):
                continue
            for n in ns:
                gA, gB = _g(method, n, lr, lam, pa), _g(method, n, lr, lam, pb)
                per = {}
                for nm in names:
                    s = _stat(kind, tms[nm], gA, gB, nboot)
                    if s is not None:
                        per[nm] = s
                XB.boot_cache_clear()
                if not per:
                    continue
                base = dict(arm=arm, contrast=cname, kind=kind, method=method, lam=lam if lam is not None else 0.0, n=int(n))
                for r in XB.pool(per, sorted(per), "all")[:-1]:
                    rows.append(_row(base, r, "region"))
                if arm == "x":
                    for pname, pnames in (("PE1", XB.PE1), ("PE2", XB.PE2), ("AUX3", XB.AUX3)):
                        sub = {k: per[k] for k in pnames if k in per}
                        if not sub:
                            continue
                        pr = XB.pool(sub, pnames, pname, registered=len(pnames))
                        mr = pr[-1]
                        rows.append(_row(base, mr, pname))
                        pools[(cname, method, int(n), pname)] = mr
                else:
                    sub = {k: per[k] for k in XB.INREGION3 if k in per}
                    if sub:
                        mr = XB.pool(sub, XB.INREGION3, "INREGION3", registered=3)[-1]
                        rows.append(_row(base, mr, "INREGION3"))
                        pools[(cname, method, int(n), "INREGION3")] = mr
    return rows, pools


def _na(v) -> bool:
    """판정할 수 없는 판정(행 없음, 판정 불가, 빈 값)."""
    return v is None or str(v) in XB.NA_VERDICTS or str(v) in ("", "None", "nan")


def xd2_verdict(lena, worse_rows) -> tuple:
    """XD-2 종합(계획 2.4 '레나 전이 n 10 과 n 40 이 모두 우세이고 알래스카·레나·캐나다의 어느 행에도 열세가 없다: 충족이면 지지, 그 밖은
    기각', 1절 '행 없음·실패는 지지로 세지 않고 판정 불가 갈래'). 반환 (판정, 문장 갈래, 사유). 순서:
    (1) 알래스카·레나·캐나다 행 가운데 열세가 있으면 기각(열세 갈래. 레나 행이 없어도 지지가 될 수 없으므로 판정할 수 있다),
    (2) 레나 n 10·40 행이 없거나 판정할 수 없으면(CI 없음) 판정 불가, (3) 둘 다 우세이면 지지, (4) 둘 다 동등이면 기각(동등 갈래),
    (5) 그 밖은 기각(미결정·기각 갈래)."""
    vs = {n: (None if lena.get(n) is None else lena[n].get("verdict4")) for n in ALASKA_X_N}
    if worse_rows:
        return "기각", "열세", ""
    na = [f"레나 n{n} {'행 없음' if lena.get(n) is None else str(v)}" for n, v in vs.items() if _na(v)]
    if na:
        return "판정 불가", "판정 불가", ", ".join(na)
    if all(str(v) == "우세" for v in vs.values()):
        return "지지", "우세", ""
    if all(str(v) == "동등" for v in vs.values()):
        return "기각", "동등", ""
    return "기각", "미결정", ""


def xd3_parts(r) -> list:
    """XD-3 한 행의 문장 갈래. 행 없음·판정할 수 없음 → ['판정 불가']. 비열등이면 '비열등', 4분 판정이 열세이면 '열세'를 함께 둔다(CI 가
    (0, 0.5) 안이면 비열등과 열세가 함께 성립한다). 둘 다 아니면 동등 또는 미결정."""
    if r is None or _na(r.get("verdict4")):
        return ["판정 불가"]
    v4 = str(r.get("verdict4"))
    parts = (["비열등"] if bool(r.get("noninf")) else []) + (["열세"] if v4 == "열세" else [])
    if parts:
        return parts
    return ["동등"] if v4 == "동등" else ["미결정"]


def _fin(v) -> bool:
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def xd_tests(rows_x, pools_x, rows_r) -> tuple:
    """XD-1·2·3 판정과 Holm(가족 m 20, 보조 열). XD-1 = PE1 풀 S8a − S1(R1@0.25, 2단, n 10·40, _rule3 우세).
    XD-2 = 레나 전이 S8w − S8a(R1@0.25, 같은 라벨) n 10·40 모두 우세이고 알래스카·레나·캐나다의 S8w − S8a(R1) 행(전이·지역 내)에 열세가 없으면
    지지, 그 밖은 기각(xd2_verdict: 열세 행이 없고 레나 행이 없거나 CI 가 없으면 판정 불가). XD-3 = 각 S8 변형 − S2·S4 의 PE1 풀 비열등
    (두 가중 2단 CI 상한 < 0.5)과 4분 판정(n 10·40). 문장은 Holm 보정 뒤 XB.compose_sentence 로 만든다(1절 '보정 전 유의', 0.5 cm 미만 표기,
    판정 불가 사유). XD-3 의 Holm 가족 p 는 비열등 p 이므로 '보정 전 유의'는 비열등 문장에만 붙고, 4분 판정 열세 문장에는 Holm p 가 없어
    0.5 cm 미만 표기만 붙는다."""
    out = []
    holm_lab, holm_p = [], []
    # XD-1
    items = []
    for n in ALASKA_X_N:
        r = pools_x.get(("S8a-S1", "R1", n, "PE1"))
        v = None if r is None else str(r.get("verdict4"))
        items.append((f"n{n}", v))
        p = float(r.get("p_two", np.nan)) if r is not None else np.nan
        holm_lab.append(f"XD-1 n{n}"); holm_p.append(p)
        out.append(dict(test_id="XD-1", item=f"S8a-S1|R1(0.25)|n{n}|PE1", n=n, verdict4=v or "행 없음", p_two=p,
                        delta=r.get("delta") if r is not None else np.nan, pool=r.get("pool", "") if r is not None else "",
                        sentence_branch=XB.branch_of(v), _row=r))
    out.append(dict(test_id="XD-1", item="종합(_rule3, 우세 기준)", verdict=XB.rule3(items, "우세")))
    # XD-2
    rx = [r for r in rows_x if r["contrast"] == "S8w-S8a" and r["method"] == "R1" and r["scope"] == "region"]
    rr = [r for r in rows_r if r["contrast"] == "S8w-S8a" and r["method"] == "R1" and r["scope"] == "region"]
    lena = {int(r["n"]): r for r in rx if r["target"] == "Lena|x"}
    fam = [r for r in rx + rr if str(r["target"]).split("|")[0] in ("Alaska", "Lena", "Canada")]
    worse_rows = [r for r in fam if str(r.get("verdict4")) == "열세"]
    for n in ALASKA_X_N:
        r = lena.get(n)
        p = float(r.get("p_two", np.nan)) if r is not None else np.nan
        holm_lab.append(f"XD-2 n{n}"); holm_p.append(p)
        out.append(dict(test_id="XD-2", item=f"S8w-S8a|R1(0.25)|n{n}|Lena|x", n=n, verdict4=str(r.get("verdict4")) if r is not None else "행 없음",
                        p_two=p, delta=r.get("delta") if r is not None else np.nan))
    v2, br2, why2 = xd2_verdict(lena, worse_rows)
    worse_lab = [f"{r['target']} n{r['n']}" for r in worse_rows]
    out.append(dict(test_id="XD-2", item="종합", verdict=v2 + (f"(열세 행: {', '.join(worse_lab)})" if worse_lab else "") + (f"({why2})" if why2 else ""),
                    sentence_branch=br2, design_label="사후 설계(레나 표적)", _xd2=(br2, why2, worse_rows, lena)))
    # XD-3
    for v in S8_SPECS:
        for ref in ("S2", "S4"):
            for n in ALASKA_X_N:
                r = pools_x.get((f"{v}-{ref}", "R1", n, "PE1"))
                ni = bool(r is not None and r.get("noninf"))
                p = float(r.get("p_ni_x2", np.nan)) if r is not None else np.nan
                holm_lab.append(f"XD-3 {v}-{ref} n{n}"); holm_p.append(p)
                v4 = str(r.get("verdict4")) if r is not None else "행 없음"
                parts = xd3_parts(r)
                out.append(dict(test_id="XD-3", item=f"{v}-{ref}|R1(0.25)|n{n}|PE1", n=n, verdict4=v4, noninf=ni, p_ni=r.get("p_ni") if r is not None else np.nan,
                                p_ni_x2=p, delta=r.get("delta") if r is not None else np.nan, pool=r.get("pool", "") if r is not None else "",
                                sentence_branch="+".join(parts), _row=r, _xd3=(v, ref, parts)))
    ht = XB.holm_table(holm_lab, holm_p, HOLM_M)
    hp = dict(zip(ht.label, ht.p_holm))
    for o in out:
        key = None
        if o["test_id"] == "XD-1" and "n" in o:
            key = f"XD-1 n{o['n']}"
        elif o["test_id"] == "XD-2" and "n" in o:
            key = f"XD-2 n{o['n']}"
        elif o["test_id"] == "XD-3":
            key = f"XD-3 {o['item'].split('|')[0]} n{o['n']}"
        if key is not None:
            o["holm_p"] = float(hp.get(key, np.nan))
            o["holm_m"] = HOLM_M
    # 문장(Holm 보정 뒤)
    T1, T2, T3 = TEMPLATES["XD-1"], TEMPLATES["XD-2"], TEMPLATES["XD-3"]
    for o in out:
        if o["test_id"] == "XD-1" and "n" in o:
            r, n = o.get("_row"), int(o["n"])
            d = r.get("delta") if r is not None else None
            tmpl = {k: s.format(n=n, a=f"{abs(float(d)):.2f}" if _fin(d) else "a", m=(r.get("pool", "") if r is not None else "")) for k, s in T1.items()}
            wr = XB.worse_regions([q for q in rows_x if q["contrast"] == "S8a-S1" and q["method"] == "R1" and q["scope"] == "region"
                                   and int(q["n"]) == n and q["target"] in XB.PE1])
            o["sentence"] = XB.compose_sentence(tmpl, o.get("verdict4"), o.get("holm_p"), d, wr, reason="행 없음" if r is None else "CI 없음")
        elif o.get("_xd2") is not None:
            br, why, wrows, lena_ = o["_xd2"]
            if br == "우세":
                hs = [hp.get(f"XD-2 n{n}", np.nan) for n in ALASKA_X_N]
                ds = [float(lena_[n]["delta"]) for n in ALASKA_X_N if _fin(lena_[n].get("delta"))]
                o["sentence"] = XB.compose_sentence({"우세": T2["지지"]}, "우세", float(np.nanmax(hs)) if any(_fin(h) for h in hs) else None,
                                                    min(ds, key=abs) if ds else None)
            elif br == "열세":
                labs = []
                for r in wrows:
                    notes = []
                    hk = f"XD-2 n{int(r['n'])}" if (r.get("arm") == "x" and r["target"] == "Lena|x") else None
                    if hk is not None and _fin(hp.get(hk)) and float(hp[hk]) >= XB.HOLM_ALPHA:
                        notes.append(XB.UNCORRECTED_TXT)
                    if _fin(r.get("delta")) and abs(float(r["delta"])) < XB.SMALL_EFFECT_CM:
                        notes.append(XB.SMALL_EFFECT_TXT)
                    labs.append(f"{r['target']} n{r['n']}" + (f"({', '.join(notes)})" if notes else ""))
                o["sentence"] = XB.compose_sentence({"열세": T2["열세"].format(x=", ".join(labs))}, "열세")
            elif br == "판정 불가":
                o["sentence"] = XB.compose_sentence({"판정 불가": T2["판정 불가"]}, "판정 불가", reason=why)
            elif br == "동등":
                o["sentence"] = XB.compose_sentence({"동등": T2["동등"]}, "동등")
            else:
                o["sentence"] = XB.compose_sentence({"미결정": T2["기각"]}, "미결정")
        elif o.get("_xd3") is not None:
            v, ref, parts = o["_xd3"]
            r, n = o.get("_row"), int(o["n"])
            ss = []
            for b in parts:
                txt = T3[b].format(v=v, ref=ref, n=n)
                if b == "비열등":
                    ss.append(XB.compose_sentence({"우세": txt}, "우세", o.get("holm_p")))
                elif b == "열세":
                    ss.append(XB.compose_sentence({"열세": txt}, "열세", None, r.get("delta") if r is not None else None))
                elif b == "판정 불가":
                    ss.append(XB.compose_sentence({"판정 불가": txt}, "판정 불가", reason="행 없음" if r is None else "CI 없음"))
                else:
                    ss.append(XB.compose_sentence({b: txt}, b))
            o["sentence"] = " ".join(ss)
    df = pd.DataFrame([{k: v for k, v in o.items() if not str(k).startswith("_")} for o in out])
    df["blind"] = "비맹검 부분 포함"
    df["common_note"] = TEMPLATES["공통"]
    return df, ht


def xd6_table(units_x, units_r, rows_x, rows_r) -> pd.DataFrame:
    """XD-6 서술: 지역 특성(조각 unit.json 의 traits, 분할 평균)과 배치 효과(S2·S4·S8a − S1, R1@0.25, 지역 행)의 표."""
    tr = {}
    for u in list(units_x) + list(units_r):
        t = (u.get("notes") or {}).get("traits")
        if not t:
            continue
        tr.setdefault(f"{u['target']}|{u['mode']}", []).append(t)
    out = []
    for r in list(rows_x) + list(rows_r):
        if r["scope"] != "region" or r["method"] != "R1" or r["contrast"] not in ("S2-S1", "S4-S1", "S8a-S1"):
            continue
        ts = tr.get(r["target"], [])
        base = {k: float(np.nanmean([t.get(k, np.nan) for t in ts])) if ts else np.nan
                for k in ("n_blocks", "eff_blocks", "cov_between", "E_cell", "E_blockmean", "ln_cell_over_block")}
        out.append(dict(target=r["target"], arm=r["arm"], contrast=r["contrast"], n=r["n"], delta=r.get("delta"), delta_beq=r.get("delta_blockeq"),
                        verdict4=r.get("verdict4"), **base))
    return pd.DataFrame(out)


def _as_dirs(dirs) -> list:
    if isinstance(dirs, (str, Path)):
        return [Path(dirs)]
    return list(dict.fromkeys(Path(d) for d in dirs))


def find_shards_dirs(dirs, tag) -> list:
    """여러 조각 폴더(봉인하지 않는 폴더와 봉인 폴더)의 <tag> 조각. 같은 단위(대상, 모드, 분할, 변형)가 두 폴더에 있으면 중단한다."""
    out, seen = [], {}
    for d in _as_dirs(dirs):
        for s_ in XB.find_shards(d, tag):
            k = (s_["target"], s_["mode"], int(s_["split"]), s_["variant"])
            if k in seen:
                raise SystemExit(f"[집계] 같은 단위 {k} 의 조각이 두 폴더에 있다({seen[k]}, {d})")
            seen[k] = str(d)
            out.append(s_)
    return out


def load_tms_dirs(dirs, tag, nboot=XB.NBOOT):
    """XB.load_tms 와 같은 경로(load_stores, h54.make_tm, h54.units_for)를 여러 조각 폴더에 적용한다. 반환 (tms, units)."""
    sh = find_shards_dirs(dirs, tag)
    if not sh:
        return {}, []
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    XB.check_cfg(units)
    stores = XB.load_stores([s_["npz"] for s_ in sh])
    by = {}
    for (nm, sp), st in stores.items():
        by.setdefault(nm, {})[int(sp)] = st
    tms = {nm: W.make_tm(nm, bs, W.units_for(units, nm), int(nboot), XB.is_point_only_name(nm)) for nm, bs in sorted(by.items())}
    return tms, units


def summarize_alg(shards_dirs, tag_t, tag_r, nboot=XB.NBOOT, root=None, allowed=None, suffix="") -> list:
    """XD-alg 집계(작업 R1a 집계, 계획 0.3 열람 순서 3·6): 대비 행, XD-1·2·3 판정과 Holm, XD-6 표를 봉인 폴더에 쓴다. 화면에는 행 수와
    sha256 만 쓴다(write_sealed). shards_dirs = 봉인하지 않는 조각 폴더와 봉인 조각 폴더(alg_dirs) 둘 다."""
    tms_x, units_x = load_tms_dirs(shards_dirs, tag_t, nboot)
    tms_r, units_r = load_tms_dirs(shards_dirs, tag_r, nboot)
    rows_x, pools_x = alg_contrast_rows(tms_x, "x", nboot) if tms_x else ([], {})
    rows_r, _ = alg_contrast_rows(tms_r, "r", nboot) if tms_r else ([], {})
    tests, ht = xd_tests(rows_x, pools_x, rows_r)
    meta = dict(plan=XB.PLAN_DOC, plan_commit=XB.PLAN_COMMIT, nboot=int(nboot), holm_m=HOLM_M, stores_x=sorted(tms_x), stores_r=sorted(tms_r),
                code_sha256=_sha256(__file__), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                opening_order="계획 0.3: 비알래스카 행은 S9* 동결 커밋 뒤, 나머지 R1a 판정 표는 R2a 제출 뒤에 연다")
    return XB.write_sealed(EXP_NAME, {f"xd_alg_contrasts{suffix}.csv": pd.DataFrame(rows_x + rows_r), f"xd_alg_tests{suffix}.csv": tests,
                                      f"xd_alg_holm{suffix}.csv": ht, f"xd6_traits{suffix}.csv": xd6_table(units_x, units_r, rows_x, rows_r),
                                      f"xd_alg_meta{suffix}.json": meta}, root=root, allowed=allowed)


def _key_rmse(st, keys):
    """키 목록 → (추출 번호 → (셀 가중, 블록 등가중) RMSE 의 seed 평균)."""
    by = {}
    for k in keys:
        by.setdefault(int(k[5]), []).append(k)
    out = {}
    for d, ks in sorted(by.items()):
        S, C = st.matrices(ks)
        rc, rb = XB.rmse_rows(S, C)
        out[d] = (float(np.nanmean(rc)), float(np.nanmean(rb)))
    return out


def rel_pairs(tm, gA, gB) -> tuple:
    """서술 지표 v_w = 평균_{분할, (A 추출, B 추출) 쌍} (RMSE_A − RMSE_B)/RMSE_B(가중별). 반환 (v_cell, v_beq, 쌍 수)."""
    vc, vb, k = [], [], 0
    for sp, st in sorted(tm.used.items()):
        ka, kb = tm.idx[sp].get(gA), tm.idx[sp].get(gB)
        if not ka or not kb:
            continue
        ra, rb_ = _key_rmse(st, ka), _key_rmse(st, kb)
        for _, (ac, ab) in ra.items():
            for _, (bc, bb) in rb_.items():
                vc.append((ac - bc) / bc); vb.append((ab - bb) / bb); k += 1
    return (float(np.mean(vc)) if vc else np.nan, float(np.mean(vb)) if vb else np.nan, k)


def xd4_rows(tms, ref="AP", nboot=None) -> tuple:
    """XD-4(ref = AP) 또는 XD-5(ref = S1)의 과제 행과 계열 행. 과제마다 v_w 를 n {20, 40} 을 합쳐 평균하고(분할·추출 쌍 평균의 n 평균),
    두 가중 모두 음수면 '우세 방향'. 계열 값 = 계열 안 과제 평균, 두 가중 모두 음수면 '방향 일치'. 과제·n 마다 2단 CI(서술)를 붙인다.
    러시아 C(TEST_DESC)는 서술 행(counted False)이고 k/5 와 계열 j/3 에 넣지 않는다."""
    trow, ci_rows = [], []
    for t in TEST_TASKS + TEST_DESC:
        nm = f"{t}|x"
        tm = tms.get(nm)
        counted = t in TEST_TASKS
        if tm is None:
            trow.append(dict(task=t, family=FAMILY_OF[t], counted=counted, v_cell=np.nan, v_beq=np.nan, direction="행 없음", n_pairs=0))
            continue
        vcs, vbs, kk = [], [], 0
        for n in TEST_N:
            gA, gB = W.gk("R1", n, LO, LAM, "S9"), W.gk("R1", n, LO, LAM, ref)
            vc, vb, k = rel_pairs(tm, gA, gB)
            if k:
                vcs.append(vc); vbs.append(vb); kk += k
            s = XB.region_stat_two_stage(tm, gA, gB, nboot=nboot)
            lo, hi, lob, hib = _ci2(s if (s and s.get("ok")) else None)
            ci_rows.append(dict(task=t, n=n, ref=ref, delta=s["delta"] if s else np.nan, delta_beq=s["delta_beq"] if s else np.nan, ci_lo=lo, ci_hi=hi,
                                ci_lo_beq=lob, ci_hi_beq=hib, v_cell_n=vc, v_beq_n=vb))
        XB.boot_cache_clear()
        vc, vb = (float(np.mean(vcs)), float(np.mean(vbs))) if vcs else (np.nan, np.nan)
        d = "우세 방향" if (vc < 0 and vb < 0) else ("열세 방향" if (vc > 0 and vb > 0) else "엇갈림")
        trow.append(dict(task=t, family=FAMILY_OF[t], counted=counted, v_cell=vc, v_beq=vb, direction=d if np.isfinite(vc) else "행 없음",
                         n_pairs=kk, n_list=",".join(str(v) for v in TEST_N if any(r_["n"] == v and np.isfinite(r_["v_cell_n"]) for r_ in ci_rows
                                                                                   if r_["task"] == t))))
    tdf = pd.DataFrame(trow)
    frows = []
    for fam_ in ("Lena", "Canada", "Tibet"):
        q = tdf[(tdf.family == fam_) & tdf.counted & tdf.v_cell.notna()]
        vc, vb = (float(q.v_cell.mean()), float(q.v_beq.mean())) if len(q) else (np.nan, np.nan)
        frows.append(dict(family=fam_, v_cell=vc, v_beq=vb, agree=bool(vc < 0 and vb < 0), worse=bool(vc > 0 and vb > 0), n_tasks=int(len(q))))
    return tdf, pd.DataFrame(frows), pd.DataFrame(ci_rows)


def summarize_test(shards_dir, tag_x, nboot=XB.NBOOT, root=None, allowed=None, suffix="", freeze_info=None) -> list:
    """XD-4·XD-5(S9* − S1) 서술 표(판정어·p 없음)를 봉인 폴더에 쓴다. 문장 = '시험 과제 5개 가운데 k개, 계열 3개 가운데 j개'."""
    tms, units, _ = XB.load_tms(shards_dir, tag_x, nboot)
    out = {}
    for ref, lab in (("AP", "xd4"), ("S1", "xd5_s1")):
        tdf, fdf, cdf = xd4_rows(tms, ref, nboot)
        k = int(((tdf.direction == "우세 방향") & tdf.counted).sum()); j = int(fdf.agree.sum())
        kw = int(((tdf.direction == "열세 방향") & tdf.counted).sum()); jw = int(fdf.worse.sum())
        summ = dict(ref=ref, k_better=k, j_families_better=j, k_worse=kw, j_families_worse=jw, n_tasks=len(TEST_TASKS), n_families=3,
                    sentence_better=TEMPLATES["XD-4"].format(k=k, j=j) if ref == "AP" else "",
                    sentence_worse=TEMPLATES["XD-4-worse"].format(k=kw, j=jw) if ref == "AP" else "", no_test_p="계열 3개라 최소 p 0.125, p 를 내지 않는다")
        out[f"{lab}_tasks{suffix}.csv"] = tdf
        out[f"{lab}_families{suffix}.csv"] = fdf
        out[f"{lab}_ci{suffix}.csv"] = cdf
        out[f"{lab}_summary{suffix}.json"] = summ
    out[f"xd4_meta{suffix}.json"] = dict(freeze=freeze_info or {}, stores=sorted(tms), code_sha256=_sha256(__file__), nboot=int(nboot),
                                          created=time.strftime("%Y-%m-%d %H:%M:%S"))
    return XB.write_sealed(EXP_NAME, out, root=root, allowed=allowed)


# ---------------------------------------------------------------- XD-5(XD-4 표 열람 뒤에만)
def family_of_task(t) -> str:
    """과제의 계열(XD-5): 개발 과제는 알래스카, 시험 과제는 FAMILY_OF."""
    return "Alaska" if t in DEV_TASKS else FAMILY_OF.get(str(t), str(t))


def lofo_folds(tasks) -> list:
    """XD-5 계열 하나 제외: 검증 = 한 계열(알래스카, 레나, 캐나다, 티베트)의 과제 전부, 학습 = 나머지 계열의 과제."""
    tasks = [str(t) for t in tasks]
    out = []
    for f in [f_ for f_ in FAMILY_TASKS if any(family_of_task(t) == f_ for t in tasks)]:
        out.append(dict(scheme="family_out", fold=f, val=[t for t in tasks if family_of_task(t) == f],
                        train=[t for t in tasks if family_of_task(t) != f]))
    return out


def loto_folds(tasks, families=("Lena", "Canada", "Tibet")) -> list:
    """XD-5 계열 안 과제 하나 제외(참고값): 검증 = 시험 과제 하나, 학습 = 나머지 과제 전부(같은 계열의 다른 과제 포함).
    계열 하나 제외와의 차이가 과제 의존 때문에 생기는 낙관의 크기다."""
    tasks = [str(t) for t in tasks]
    return [dict(scheme="task_out", fold=t, val=[t], train=[u for u in tasks if u != t]) for t in tasks if family_of_task(t) in families]


def cv_by_folds(df, folds, iters, seed=GBM_SEED, threads=1) -> pd.DataFrame:
    """정해진 반복 수의 S9-GBM 을 접기마다 다시 학습해 검증 과제별 개발 교차검증 효용을 낸다(XD-5)."""
    rows = []
    for f in folds:
        tr = df[df.task.isin(f["train"])]
        if not len(tr):
            continue
        m = _cb_fit(tr[FEATS].values, tr.U.values, iters, seed, threads)
        for t in f["val"]:
            va = df[df.task == t]
            if len(va):
                rows.append(dict(scheme=f["scheme"], fold=f["fold"], task=t, family=family_of_task(t), n_train=int(len(tr)), n_val=int(len(va)),
                                 cv_utility=cv_utility(va, m.predict(va[FEATS].values, thread_count=int(threads)))))
    return pd.DataFrame(rows)


def _u_of(st, placement, n, seed=0):
    """저장소 한 분할에서 placement 의 R1@0.25(seed) 키별 (셀 가중, 블록 등가중) RMSE 목록(추출 순)."""
    ks = sorted([k for k in st.keys if k[0] == "R1" and k[3] == placement and int(k[4]) == int(n) and int(k[6]) == int(seed)], key=lambda k: k[5])
    if not ks:
        return []
    S, C = st.matrices(ks)
    rc, rb = XB.rmse_rows(S, C)
    return list(zip(rc.tolist(), rb.tolist()))


def xd5_regret(tms_x, df_learn) -> pd.DataFrame:
    """XD-5 오라클 대비 후회: (과제, 분할, n)마다 U(S9*)(정책 seed 평균), U(Algorithm P)(추출 평균)와 후보 200개 가운데 최선의 U(오라클).
    U 는 학습 자료와 같은 정의(R1@0.25, seed 0, 기준 = 같은 분할의 S1 추출 d 0–4 평균, 두 가중 평균)다."""
    rows = []
    for t in TEST_TASKS + TEST_DESC:
        tm = tms_x.get(f"{t}|x")
        if tm is None:
            continue
        for sp, st in sorted(tm.used.items()):
            for n in TEST_N:
                ref = _u_of(st, "S1", n)
                if not ref:
                    continue
                rc, rb = float(np.mean([v[0] for v in ref])), float(np.mean([v[1] for v in ref]))

                def u(pl):
                    v = _u_of(st, pl, n)
                    return float(np.mean([0.5 * ((a_ - rc) / rc + (b_ - rb) / rb) for a_, b_ in v])) if v else np.nan
                q = df_learn[(df_learn.task == t) & (df_learn.split == int(sp)) & (df_learn.n == int(n))] if len(df_learn) else df_learn
                orc = float(q.U.min()) if len(q) else np.nan
                us9, uap = u("S9"), u("AP")
                rows.append(dict(task=t, family=FAMILY_OF.get(t, t), counted=t in TEST_TASKS, split=int(sp), n=int(n), U_S9=us9, U_AP=uap,
                                 U_oracle=orc, U_median=float(q.U.median()) if len(q) else np.nan, n_sets=int(len(q)),
                                 regret_S9=us9 - orc if np.isfinite(orc) else np.nan, regret_AP=uap - orc if np.isfinite(orc) else np.nan))
    return pd.DataFrame(rows)


def summarize_xd5(shards_dir, tag_x, tag_l, gbm_meta=None, threads=1, root=None, allowed=None, suffix="") -> list:
    """XD-5 서술(계획 2.4, 0.3 열람 순서 4: XD-4 시험 표를 연 뒤): 오라클 대비 후회, 계열 하나 제외와 계열 안 과제 하나 제외의 S9-GBM
    개발 교차검증 효용(반복 수 = 동결 S9-GBM 의 값). 시험 과제의 학습 자료(xd_learn --xd5)가 있어야 한다. 봉인 폴더에 쓴다."""
    tms, _, _ = XB.load_tms(shards_dir, tag_x, 0)
    df = load_learning(shards_dir, tag_l, DEV_TASKS + TEST_TASKS + TEST_DESC)
    out = {f"xd5_regret{suffix}.csv": xd5_regret(tms, df)}
    meta = dict(n_learning_rows=int(len(df)), tasks=sorted(df.task.unique()) if len(df) else [], created=time.strftime("%Y-%m-%d %H:%M:%S"),
                code_sha256=_sha256(__file__))
    if gbm_meta is not None and len(df):
        verify_sidecar(gbm_meta)
        it = int(json.loads(Path(gbm_meta).read_text())["iterations"])
        tasks = sorted(df.task.unique())
        out[f"xd5_cv_gbm{suffix}.csv"] = cv_by_folds(df, lofo_folds(tasks) + loto_folds(tasks), it, threads=threads)
        meta.update(gbm_iterations=it, gbm_meta=str(gbm_meta))
    out[f"xd5_meta{suffix}.json"] = meta
    return XB.write_sealed(EXP_NAME, out, root=root, allowed=allowed)


def gate_alg(shards_dirs, tag_t, tag_r, ref8_dir, ref2_dir, out_path, allowed=None) -> dict:
    """재현 관문(계획 2.4): S1·S2·S4 의 블록 SSE 가 WF8(전이, results/rescale_wf2)·WF2(지역 내, results/rescale_wf) 조각과 같다(같은 노드 종류,
    차 0). shards_dirs = 봉인하지 않는 폴더와 봉인 조각 폴더(alg_dirs). 표에는 키와 SSE 차만 있다. 화면에는 키 수·실패 수·통과 여부만 쓴다."""
    out = []
    for tag, ref_dir, ref_tag in ((tag_t, ref8_dir, "wf8"), (tag_r, ref2_dir, "wf2")):
        sh = [s_ for s_ in find_shards_dirs(shards_dirs, tag) if s_["variant"] in BASE_STRATS]
        rs = [s_ for s_ in XB.find_shards(ref_dir, ref_tag) if s_["variant"] in BASE_STRATS] if ref_dir and Path(ref_dir).exists() else []
        if not sh or not rs:
            continue
        new = XB.load_stores([s_["npz"] for s_ in sh])
        ref = XB.load_stores([s_["npz"] for s_ in rs])
        df = XB.gate_compare(new, ref, "same_node", key_fn=lambda k: k[3] in BASE_STRATS and k[0] in ("P1", "R1", "D1"))
        df.insert(0, "ref", ref_tag)
        out.append(df)
    tab = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["ref", "store", "split", "key", "max_abs_dsse", "max_rel_dsse",
                                                                               "cnt_equal", "ok", "note"])
    sha = _write_csv(out_path, tab, allowed)
    s = XB.gate_summary(tab)
    print(f"[관문] 공통 키 {s['n_keys']} · 실패 {s['n_fail']} · 통과 {s['passed']} · 표 sha256 {sha[:16]}", flush=True)
    return s


# ================================================================ 12. 실행(단위 열거, 실행, 세기)
def _n_list(txt):
    return [-1 if v.strip() == "all" else int(v) for v in str(txt).split(",") if v.strip()]


def _list(txt):
    return [v.strip() for v in str(txt).split(",") if v.strip()]


def enumerate_units(a):
    """작업 단위 (실험, 별칭, 모드, 분할, 변형)과 기대 분할. 분할 = h40 규칙(중복·채점 블록 2 미만 제외)의 분할 1–5."""
    units, expected, skipped = [], {}, []
    for exp in a.XD_EXPS:
        mode = MODE_OF[exp]
        for alias in a.XD_T[exp]:
            if exp == "xd_r":
                _, D = XB.get_data(a)
                tgt = alias
            else:
                spec = XB.resolve(alias)[0]
                if spec is not None and not (a.LGD / spec / "fidelity_base_v3.csv").exists():
                    skipped.append(dict(exp=exp, target=alias, status="lgd_table_missing"))
                    continue
                _, D = XB.get_data(a, alias)
                tgt = XB.resolve(alias)[1]
            keep, skip, info = W.split_plan(D, tgt, a.SPLITS)
            expected[(exp, alias, mode)] = keep
            skipped += [dict(exp=exp, target=alias, split=sp, status=st_) for sp, st_, _ in skip]
            if exp in ("xd_r", "xd_t"):
                units += [(exp, alias, mode, sp, s) for sp in keep for s in a.XD_STRATS]
            elif exp == "xd_learn":
                units += [(exp, alias, mode, sp, f"n{n}") for sp in keep for n in a.G[exp] if 0 < int(n) < int(info[sp]["n_A"])]
            elif exp == "xd_test":
                units += [(exp, alias, mode, sp, "") for sp in keep]
    if a.XD_SHARD is not None:
        i, k = a.XD_SHARD
        units = [u for j, u in enumerate(units) if j % k == i]
    return units, expected, skipped


def unit_cfg(a, exp, variant="", data_sha=""):
    """결과에 영향을 주는 설정(대상·분할·tag·워커는 넣지 않는다)."""
    kw = dict(grid=[int(v) for v in a.G[exp]], draws=n_draws(a), seeds=list(a.SEEDS), cb_iters=int(a.cb_iters), mode=MODE_OF[exp])
    if exp in ("xd_r", "xd_t"):
        kw.update(strategies=list(STRATS), s8=S8_SPECS, s8_seed=S8_SEED_TAG, coef=list(COEF_VARIANTS), coef_base=COEF_BASE, schedule="grid",
                  dup="reuse_fit_store_all", methods=["P1", "R1", "D1"] if exp == "xd_r" else ["P1", "R1"])
    if exp == "xd_learn":
        kw.update(n_sets=int(a.XD_NSETS), mix=list(SET_MIX), pert=list(PERT_FRAC), gens=list(STRUCT_GENS), feats=list(FEATS), el_feats=list(EL_FEATS),
                  cov_sub=COV_SUB, fit_seed=int(a.SEEDS[0]), method="R1@0.25")
    if exp == "xd_test":
        kw.update(index_sha256=a.XD_INDEX.sha256 if a.XD_INDEX else "", ap=a.XD_AP, policy_seeds=list(POLICY_SEEDS[:n_draws(a)]), s8_seed=S8_SEED_TAG,
                  ap_schedule=list(AP_SCHEDULE))
    if exp in ("xd_r", "xd_t"):
        kw.update(coef_shrink="none")                                       # S8w·S8r 앵커 = 수축하지 않은 E_w·E_RE(2026-10-04 검토 반영)
    return XB.make_unit_cfg(exp, variant, data_sha, **kw)


def run_unit(a, exp, alias, mode, split, variant="", dry=False, expected=None):
    """작업 단위 하나를 돌리고 조각을 쓴다(dry 이면 적합 수만 센다)."""
    t0 = time.time()
    if exp == "xd_r":
        c = XB.build_rctx(a, alias, split)
        U = XDRUnit(a, c, variant, dry)
    else:
        c = XB.build_tctx(a, alias, "x", split)
        if exp == "xd_t":
            U = XDTUnit(a, c, alias, variant, dry)
        elif exp == "xd_learn":
            U = XDLUnit(a, c, alias, int(str(variant)[1:]), dry, n_sets=a.XD_NSETS)
        elif exp == "xd_test":
            ent = a.XD_INDEX.entries_for(alias, "x", split) if a.XD_INDEX is not None else {}
            if not ent and not dry:
                if alias not in TEST_TASKS:                                 # 서술 과제(Russia_C~lgd): 건너뛴다(k/5 에 넣지 않는 행)
                    return dict(exp=exp, target=alias, mode=mode, split=int(split), variant=variant, n_fit_total=0, elapsed_s=0.0,
                                status="not_indexed")
                raise RuntimeError(f"색인 파일에 {alias} 분할 {split} 항목이 없다")   # 시작 전 확인(run_units)을 지난 뒤에는 실패로만 남긴다
            if not ent:                                                     # 세기: 색인 파일 없이 같은 수의 적합을 센다(가상 순서)
                fp = a_fingerprint(c.XA, c.sA, c.blkA)
                ent = {p: dict(nA=len(c.yA), a_fp=fp, order=np.random.RandomState(p).permutation(len(c.yA))[:POLICY_NMAX].tolist())
                       for p in POLICY_SEEDS}
            U = XDXUnit(a, c, alias, ent, a.XD_AP, dry)
        else:
            raise ValueError(exp)
    U.run()
    if dry:
        rows, st, stats = U.finish()
        return dict(exp=exp, target=alias, mode=mode, split=int(split), variant=variant, n_A=len(c.yA), n_eval=len(c.yB), n_rows=stats["n_rows"],
                    fit_total=int(sum(stats["n_fit"].values())), est_s=round(float(sum(stats["est_detail"].values())), 2),
                    _detail=dict(stats["n_fit_detail"]), _est=dict(stats["est_detail"]))
    return write_unit_shard(a, exp, alias, mode, split, variant, U, c, t0, expected,
                            XB.data_sha(a, None if exp == "xd_r" else alias))


def write_unit_shard(a, exp, alias, mode, split, variant, U, c, t0, expected=None, data_sha=""):
    """단위의 조각 쓰기(h54.write_shard 와 같은 내용: 단위 통계 + 문맥 메타). xd_learn 은 <조각>_sets.npz 를 unit.json 보다 먼저 쓴다.
    폴더 = shard_dir_of(알래스카 계열이 아닌 xd_t·xd_r 은 sealed/shards, 스모크는 sealed/shards_smoke)."""
    rows, st, stats = U.finish()
    tag = TAGS[exp] + a.XD_SUFFIX
    cfg = unit_cfg(a, exp, variant, data_sha)
    unit = {**stats, **{k: v for k, v in c.meta.items() if not isinstance(v, (dict, list))}}
    unit.update(exp=exp, target=alias, mode=mode, parent=c.parent, split=int(split), variant=variant, elapsed_s=round(time.time() - t0, 1),
                n_fit_total=int(sum(stats["n_fit"].values())), n_A=int(len(c.yA)), n_eval=int(len(c.yB)), E0=float(c.E0))
    sdir = shard_dir_of(a, exp, alias, mode)
    unit.update(sealed_rows=bool(unit_sealed(exp, alias, mode) or a.XD_SUFFIX))
    paths = XB.shard_paths(sdir, tag, alias, mode, split, variant)
    if exp == "xd_learn":
        XB.check_out_dir(sdir, a.XD_ALLOWED)
        paths["unit"].unlink(missing_ok=True)
        sp_path = Path(str(XB.shard_base(sdir, tag, alias, mode, split, variant)) + "_sets.npz")
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        if U.sets_out is not None:
            tmp = sp_path.with_name(sp_path.name + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp, **U.sets_out)
            os.replace(tmp, sp_path)
            unit["sets_sha256"] = _sha256(sp_path)
    return XB.write_shard(sdir, tag, alias, mode, split, rows, st, cfg, unit=unit, variant=variant,
                          expected=(expected or {}).get((exp, alias, mode), [split]), code_file=__file__, allowed=a.XD_ALLOWED)


_WA = None
_WEXP = None


def _worker_init(argv, expected_items):
    global _WA, _WEXP
    _WA = parse(argv)
    _WEXP = {tuple(k): v for k, v in expected_items}


def _worker_run(exp, alias, mode, split, variant):
    u = run_unit(_WA, exp, alias, mode, split, variant, expected=_WEXP)
    return {k: u.get(k) for k in ("exp", "target", "mode", "split", "variant", "n_fit_total", "elapsed_s", "status")}


def resume_state(a, u) -> tuple:
    exp, alias, mode, split, variant = u
    cfg = unit_cfg(a, exp, variant, XB.data_sha(a, None if exp == "xd_r" else alias))
    sdir = shard_dir_of(a, exp, alias, mode)
    ok, why = XB.unit_state(sdir, TAGS[exp] + a.XD_SUFFIX, alias, mode, split, cfg, variant)
    if ok and exp == "xd_learn":
        sp_path = Path(str(XB.shard_base(sdir, TAGS[exp] + a.XD_SUFFIX, alias, mode, split, variant)) + "_sets.npz")
        if not sp_path.exists():
            return False, "sets.npz 없음"
    return ok, why


def count_only(a, units, skipped) -> pd.DataFrame:
    """학습 없이 적합 수와 추정 시간을 센다(라벨 값을 쓰지 않는 세기 범주. 화면에는 단위·적합 수와 추정 시간만)."""
    t0 = time.time()
    rows, det, est = [], Counter(), Counter()
    for u in units:
        r = run_unit(a, *u, dry=True)
        for k, v in r.pop("_detail").items():
            det[f"{u[0]}|{k}"] += v
        for k, v in r.pop("_est").items():
            est[f"{u[0]}|{k}"] += v
        rows.append(r)
    df = pd.DataFrame(rows)
    out = a.XD_OUT / "count"
    XB.check_out_dir(out, a.XD_ALLOWED)
    out.mkdir(parents=True, exist_ok=True)
    if not len(df):
        print("[세기] 작업 단위 없음", flush=True)
        return df
    df.to_csv(out / f"xd_count{a.XD_SUFFIX}.csv", index=False)
    tab = pd.DataFrame([dict(exp=k.split("|")[0], key=k, fits=int(v), est_h=round(est[k] / 3600.0, 4)) for k, v in sorted(det.items())])
    tab.to_csv(out / f"xd_count_detail{a.XD_SUFFIX}.csv", index=False)
    by = df.groupby("exp").agg(units=("split", "size"), targets=("target", "nunique"), fits=("fit_total", "sum"), est_s=("est_s", "sum"))
    for exp, r in by.iterrows():
        h = float(r.est_s) / 3600.0
        print(f"[세기] {exp}: 단위 {int(r.units)} · 대상 {int(r.targets)} · 적합 {int(r.fits):,} · 추정 {h:.2f} 워커·시간(4스레드, h54 추정식) · "
              f"실측 비 {W.RESCALE_RATIO} 적용 {h * W.RESCALE_RATIO:.2f} · 워커 22개 약 {h * W.RESCALE_RATIO / 22:.2f} h", flush=True)
    tot = float(df.est_s.sum()) / 3600.0
    print(f"[세기] 합계: 단위 {len(df)} · 적합 {int(df.fit_total.sum()):,} · 추정 {tot:.2f} 워커·시간(비 적용 {tot * W.RESCALE_RATIO:.2f}) · "
          f"건너뜀 {len(skipped)} · 세기 {time.time() - t0:.0f}s · 표 {out.relative_to(ROOT) if str(out).startswith(str(ROOT)) else out}", flush=True)
    return df


# ================================================================ 13. 인자와 명령행
def build_parser():
    ap = argparse.ArgumentParser(description="XD_placement_policy(계획 2.4): XD-alg, Algorithm P 선택, XD-learn, S9-GBM, 동결, XD-4 시험 팔")
    ap.add_argument("--exp", default="xd_r,xd_t,xd_learn", help="xd_r, xd_t, xd_learn, xd_test(쉼표 목록)")
    ap.add_argument("--xdr-targets", default=",".join(XDR_TARGETS))
    ap.add_argument("--xdt-targets", default=",".join(XDT_TARGETS))
    ap.add_argument("--dev-tasks", default=",".join(DEV_TASKS))
    ap.add_argument("--test-tasks", default=",".join(TEST_TASKS + TEST_DESC))
    ap.add_argument("--xdr-grid", default=",".join(str(v) for v in XDR_GRID))
    ap.add_argument("--xdt-grid", default=",".join(str(v) for v in XDT_GRID))
    ap.add_argument("--learn-grid", default=",".join(str(v) for v in DEV_N))
    ap.add_argument("--test-grid", default=",".join(str(v) for v in TEST_N))
    ap.add_argument("--strategies", default=",".join(STRATS))
    ap.add_argument("--n-sets", type=int, default=N_SETS)
    ap.add_argument("--splits", type=int, default=5, help="분할 1..K(전이·지역 내 모두 1–5)")
    ap.add_argument("--seeds", type=int, default=len(XB.SEEDS))
    ap.add_argument("--draws-cap", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lgd-dir", default="data/processed/lgd/run_tables")
    ap.add_argument("--shard", default="", help="I/K: 단위 목록의 I 번째 몫만(0 ≤ I < K)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="작은 설정(분할 1, 추출 1, seed 0, 대상은 알래스카 계열만). 조각과 산출 표는 봉인 폴더")
    ap.add_argument("--mem-wait", type=float, default=1800.0, help="로컬 실행: 가용 메모리가 30 GB 아래면 이 초만큼 기다린 뒤 거부(계획 1절)")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--summarize-only", action="store_true", help="XD-alg 봉인 표 + 알래스카 계열 선택")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--select-p", action="store_true", help="Algorithm P 선택만(알래스카 계열 조각만 연다)")
    ap.add_argument("--select-default", action="store_true", help="마감 초과: P_default(S8a) 기록")
    ap.add_argument("--gate", action="store_true", help="재현 관문(WF2·WF8 조각과 비교)")
    ap.add_argument("--wf8-ref", default="results/rescale_wf2/data/processed/wf/shards")
    ap.add_argument("--wf2-ref", default="results/rescale_wf/data/processed/wf/shards")
    ap.add_argument("--train-gbm", action="store_true", help="S9-GBM 학습(작업 R1a 마지막 단계)")
    ap.add_argument("--export-selection", default="", choices=["", "gbm"], help="S9-GBM 정책으로 시험 과제 선택 색인을 쓴다")
    ap.add_argument("--model", default="", help="S9-GBM 모형 경로(기본 <out>/policy/s9_gbm.cbm)")
    ap.add_argument("--index-out", default="")
    ap.add_argument("--freeze", action="store_true", help="S9* 동결 기록")
    ap.add_argument("--attest-freeze", action="store_true", help="커밋 뒤: 동결 기록의 커밋 확인 기록(R2b 묶음용, --freeze-manifest 와 함께)")
    ap.add_argument("--gbm-meta", default="")
    ap.add_argument("--ds-meta", default="")
    ap.add_argument("--freeze-manifest", default="", help="xd_test·--summarize-test 에 필요")
    ap.add_argument("--selection-file", default="", help="Algorithm P 선택 파일(xd_test 의 AP)")
    ap.add_argument("--summarize-test", action="store_true", help="XD-4·5 봉인 표")
    ap.add_argument("--xd5", action="store_true", help="XD-5: 개발 과제 밖 학습 자료 생성 허용(XD-4 표 열람 뒤에만)")
    ap.add_argument("--summarize-xd5", action="store_true", help="XD-5 봉인 표(후회, 계열 하나 제외·과제 하나 제외 S9-GBM). --xd5 와 함께")
    return ap


def parse(argv=None):
    """명령행 → h54 인자 계약을 만족하는 인자 객체(XB.h54_args)에 XD 항목을 더한다."""
    av = list(sys.argv[1:] if argv is None else argv)
    o = build_parser().parse_args(av)
    exps = [e for e in _list(o.exp) if e in EXPS]
    splits = list(range(1, int(o.splits) + 1))
    seeds, draws_cap, nboot = int(o.seeds), int(o.draws_cap), int(o.nboot)
    T = dict(xd_r=_list(o.xdr_targets), xd_t=_list(o.xdt_targets), xd_learn=_list(o.dev_tasks), xd_test=_list(o.test_tasks))
    G = dict(xd_r=_n_list(o.xdr_grid), xd_t=_n_list(o.xdt_grid), xd_learn=_n_list(o.learn_grid), xd_test=_n_list(o.test_grid))
    n_sets = int(o.n_sets)
    suffix = ""
    if o.smoke:                                                               # 스모크: 알래스카 계열만(PE1·티베트·시험 과제의 행을 만들지 않는다)
        splits, seeds, draws_cap, nboot, n_sets, suffix = [1], 1, (draws_cap or 1), min(nboot, 500), min(n_sets, 12), "_smoke"
        T = {k: list(v) for k, v in SMOKE_TARGETS.items()}
        G = {k: list(v) for k, v in SMOKE_GRID.items()}
        assert_smoke_alaska(T)
    if not o.smoke and "xd_learn" in exps and not o.xd5:
        bad = [t for t in T["xd_learn"] if t not in DEV_TASKS]
        if bad:
            raise SystemExit(f"[거부] 개발 과제 밖 학습 자료({bad})는 XD-5 단계(--xd5, XD-4 표 열람 뒤)에서만 만든다")
    if not o.smoke and "xd_test" in exps and not o.freeze_manifest and not o.count_only:
        raise SystemExit("[거부] xd_test 는 S9* 동결 기록(--freeze-manifest)이 있어야 한다(계획 0.3: 동결 커밋 뒤에만 시험 과제 평가)")
    a = XB.h54_args(exp="xd_t", splits=splits, grid=G["xd_t"], threads=o.threads, cb_iters=o.cb_iters, seeds=seeds, nboot=nboot,
                    data_dir=o.data_dir, lgd_dir=o.lgd_dir, tag="xd", draws_cap=draws_cap, allow_local=o.allow_local)
    a.G.update(G)
    a.O = o
    a.XD_EXPS = exps
    a.XD_T = T
    a.XD_STRATS = [s for s in _list(o.strategies) if s in STRATS]
    a.XD_NSETS = n_sets
    a.XD_SUFFIX = suffix
    a.XD_OUT = Path(o.out_dir) if os.path.isabs(o.out_dir) else ROOT / o.out_dir
    a.XD_SHARDS, a.XD_SEALED_SHARDS = out_shard_dirs(a.XD_OUT, o.smoke)       # 스모크는 둘 다 sealed/shards_smoke
    if o.xd5 and not o.smoke and not o.count_only:                            # XD-5 는 XD-4 시험 표가 봉인 폴더에 있을 때만(열람 순서 4)
        require_xd4_sealed(a.XD_OUT)
    a.XD_ALLOWED = None
    a.XD_SHARD = None
    if o.shard:
        i, k = (int(v) for v in o.shard.split("/"))
        if not (0 <= i < k):
            raise SystemExit("--shard 는 I/K(0 ≤ I < K)")
        a.XD_SHARD = (i, k)
    a.XD_INDEX = None
    a.XD_FREEZE_COMMIT = None
    a.XD_AP = AP_DEFAULT
    if "xd_test" in exps and not o.count_only:
        if o.smoke:
            idx = a.XD_OUT / "policy" / f"s9_gbm_index{suffix}.json"
            a.XD_INDEX = IndexFile(idx) if Path(str(idx) + ".sha256").exists() else None
        else:
            fz = load_freeze(o.freeze_manifest)
            a.XD_FREEZE_COMMIT = require_freeze_committed(o.freeze_manifest)  # 동결 기록 커밋 확인(계획 0.3)
            a.XD_INDEX = IndexFile(rebase_path(fz.get("index_relpath") or fz["index_path"]), o.freeze_manifest)
        if o.selection_file:
            a.XD_AP = load_selection(o.selection_file)["chosen"]
        elif not o.smoke:
            raise SystemExit("[거부] xd_test 는 Algorithm P 선택 파일(--selection-file)이 있어야 한다")
    return a


def _log(s):
    print(s, flush=True)


def _drop_opt(argv, name):
    """명령행에서 옵션 name(값 포함, '--name=값' 형식 포함)을 뺀다."""
    out, skip = [], False
    for v in argv:
        if skip:
            skip = False
            continue
        if v == name:
            skip = True
            continue
        if v.startswith(name + "="):
            continue
        out.append(v)
    return out


def run_units(a, argv):
    global _WA, _WEXP
    units, expected, skipped = enumerate_units(a)
    _log(f"[자료] 실험 {a.XD_EXPS} · 작업 단위 {len(units)}(건너뜀 {len(skipped)}) · 실험별 {dict(Counter(u[0] for u in units))} · 분할 {a.SPLITS} · "
         f"seed {a.SEEDS} · 추출 {n_draws(a)} · 스레드 {a.threads} · 워커 {a.O.workers}")
    if a.O.count_only:
        count_only(a, units, skipped)
        return dict(done=[], failed=[])
    if "xd_test" in a.XD_EXPS:                                                # 시작 전: 색인이 시험 과제·유효 분할·정책 seed 를 덮는가
        cov = index_coverage(a.XD_INDEX, expected, POLICY_SEEDS[:n_draws(a)], max([int(v) for v in a.G["xd_test"]] or [0]))
        if cov["counted"] or cov["short"]:
            raise SystemExit(f"[거부] 색인 파일이 시험 과제를 덮지 않는다(빠진 항목 {[m[:2] for m in cov['counted']]}, n_max 부족 {cov['short']}). "
                             f"S9 색인 내보내기의 과제 목록을 xd_test 와 맞춘다")
        if cov["desc"]:
            drop = {(m[0], m[1]) for m in cov["desc"]}
            units = [u for u in units if not (u[0] == "xd_test" and (u[1], int(u[3])) in drop)]
            skipped += [dict(exp="xd_test", target=m[0], split=m[1], status="not_indexed") for m in cov["desc"]]
            _log(f"[건너뜀] 서술 과제의 색인 항목 없음(not_indexed): {sorted(drop)}")
    for d_ in alg_dirs(a):
        XB.check_out_dir(d_, a.XD_ALLOWED)
    a.XD_SHARDS.mkdir(parents=True, exist_ok=True)
    todo = []
    for u in units:
        ok, why = resume_state(a, u) if a.O.resume else (False, "")
        if not ok:
            todo.append(u)
    _log(f"[계획] 실행 {len(todo)} · 재개로 건너뜀 {len(units) - len(todo)}")
    exp_items = [(list(k), v) for k, v in expected.items()]
    if int(a.O.workers) <= 0:                                                 # 이 프로세스에서 차례로(같은 인자 객체를 쓴다)
        _WA, _WEXP = a, dict(expected)
        done, failed = XB.execute(todo, _worker_run, workers=0, log=_log)
    else:
        done, failed = XB.execute(todo, _worker_run, workers=int(a.O.workers), threads=int(a.threads), init=_worker_init,
                                  init_args=(argv, exp_items), log=_log)
    _log(f"[완료] 완료 {len(done)} · 실패 {len(failed)}")
    return dict(done=done, failed=failed)


def train_gbm_cli(a):
    """S9-GBM 학습(R1a 마지막 단계). 모형·기록은 <out>/policy/ 에 쓴다. 화면에는 행 수, 반복 수 후보, 파일 해시만."""
    df = load_learning(a.XD_SHARDS, TAGS["xd_learn"] + a.XD_SUFFIX, a.XD_T["xd_learn"])
    if not len(df):
        raise SystemExit("[S9-GBM] 학습 자료가 없다")
    model, rep, iters, folds = train_gbm(df, threads=a.threads)
    pol = a.XD_OUT / "policy"
    XB.check_out_dir(pol, a.XD_ALLOWED)
    pol.mkdir(parents=True, exist_ok=True)
    mpath = pol / f"s9_gbm{a.XD_SUFFIX}.cbm"
    model.save_model(str(mpath))
    msha = _sha256(mpath)
    import catboost
    meta = dict(variant="S9-GBM", model_path=str(mpath), model_sha256=msha, iterations=iters, cv=rep, cv_utility=float(min(r["cv_utility"] for r in rep
                                                                                                                          if np.isfinite(r["cv_utility"]))),
                folds=folds, features=FEATS, gbm_cfg=GBM_CFG, seeds=[GBM_SEED], catboost=catboost.__version__, threads=int(a.threads),
                n_rows=int(len(df)), tasks=sorted(df.task.unique()), created=time.strftime("%Y-%m-%d %H:%M:%S"), code_sha256=_sha256(__file__),
                index_path=str(pol / f"s9_gbm_index{a.XD_SUFFIX}.json"))
    if a.XD_SUFFIX:
        sealed = XB.write_sealed(EXP_NAME, {f"s9_gbm_meta{a.XD_SUFFIX}.json": meta}, root=a.XD_OUT.parent, allowed=a.XD_ALLOWED)
        _log(f"[S9-GBM] 스모크 기록은 봉인 폴더에 썼다({len(sealed)}개)")
        meta_path = XB.sealed_dir(EXP_NAME, a.XD_OUT.parent) / f"s9_gbm_meta{a.XD_SUFFIX}.json"
    else:
        meta_path = pol / "s9_gbm_meta.json"
        sha = _write_json(meta_path, meta, a.XD_ALLOWED)
        _log(f"[S9-GBM] 학습 행 {len(df)} · 반복 수 후보 {list(GBM_ITERS)} · 모형 sha256 {msha[:16]} · 기록 sha256 {sha[:16]}")
    return meta_path


def export_gbm_cli(a):
    """S9-GBM 정책의 시험 과제 선택 색인(라벨 미사용)."""
    from catboost import CatBoostRegressor
    pol = a.XD_OUT / "policy"
    mpath = Path(a.O.model) if a.O.model else pol / f"s9_gbm{a.XD_SUFFIX}.cbm"
    m = CatBoostRegressor()
    m.load_model(str(mpath))
    tasks = a.XD_T["xd_test"]
    if not a.XD_SUFFIX:
        assert_dev_test_disjoint(a.XD_T["xd_learn"], tasks)
    tfs = task_feats_for(a, tasks)
    seeds = POLICY_SEEDS[:n_draws(a)]
    path = Path(a.O.index_out) if a.O.index_out else pol / f"s9_gbm_index{a.XD_SUFFIX}.json"
    sha = export_index(tfs, gbm_score_fn(m, a.threads), "S9-GBM", _sha256(mpath), path, seeds=seeds, n_max=max(a.G["xd_test"]), allowed=a.XD_ALLOWED)
    _log(f"[색인] S9-GBM · 과제·분할 {len(tfs)} · 정책 seed {len(seeds)} · {path.name} sha256 {sha[:16]}")
    return path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse(argv)
    o = a.O
    if o.count_only:
        XB.guard("count", argv)
    elif o.smoke:
        XB.guard("smoke", argv)
    elif o.summarize_only or o.select_p or o.select_default or o.gate or o.freeze or o.summarize_test or o.attest_freeze:
        XB.guard("summarize", argv)
    else:
        XB.guard("run", argv)                                                # 본 실행·S9-GBM 학습·색인은 허용 표지가 있어야 한다
    if not XB.run_permitted(argv) and int(o.workers) > 2:                       # 허용 표지 없는 로컬 실행: 워커 2 × 스레드 2 = 코어 4개 이하
        o.workers = 2
    for v in XB.THREAD_VARS:
        os.environ[v] = str(a.threads)
    rc = 0
    with XB.restricted_output():
        if not on_rescale():                                                  # 1절 로컬 자원: 가용 메모리 30 GB 확인(대기) + 자료 영역 상한 10 GB
            local_resources(o.mem_wait, _log)
        if a.threads < int(o.threads):
            _log(f"[local] 허용 표지가 없다: --threads {o.threads} → {a.threads}")
        sel_dir = a.XD_OUT / "selection"
        tag_t, tag_r = TAGS["xd_t"] + a.XD_SUFFIX, TAGS["xd_r"] + a.XD_SUFFIX
        if o.attest_freeze:
            if not o.freeze_manifest:
                raise SystemExit("[거부] --attest-freeze 는 --freeze-manifest 가 있어야 한다")
            write_freeze_attestation(o.freeze_manifest, a.XD_ALLOWED)
            return 0
        if o.select_default:
            select_default(sel_dir, a.XD_ALLOWED, a.XD_SUFFIX)
            return 0
        if o.freeze:
            freeze(o.gbm_meta or None, o.ds_meta or None, a.XD_OUT / "freeze" / "s9_freeze_manifest.json", a.XD_ALLOWED)
            return 0
        if o.gate:
            gate_alg(alg_dirs(a), tag_t, tag_r, ROOT / o.wf8_ref, ROOT / o.wf2_ref, a.XD_OUT / "gate" / "xd_gate.csv", a.XD_ALLOWED)
            return 0
        if o.select_p or o.summarize_only or o.summarize_test:
            nb = int(a.nboot)
            if not o.smoke and (nb != XB.NBOOT or not XB.run_permitted(argv)):
                raise SystemExit(f"[거부] 선택·집계는 재표집 {XB.NBOOT}회로 하고 허용 표지(WF_RESCALE=1 또는 --allow-local)가 있어야 한다"
                                 f"(허용 표지 없는 집계 상한 {XB.LOCAL_NBOOT_MAX}회, 계획 1절)")
            if o.summarize_test:
                fz = load_freeze(o.freeze_manifest) if o.freeze_manifest else {}
                summarize_test(a.XD_SHARDS, TAGS["xd_test"] + a.XD_SUFFIX, nb, root=a.XD_OUT.parent, allowed=a.XD_ALLOWED, suffix=a.XD_SUFFIX,
                               freeze_info=fz)
                return 0
            if o.summarize_only:
                summarize_alg(alg_dirs(a), tag_t, tag_r, nb, root=a.XD_OUT.parent, allowed=a.XD_ALLOWED, suffix=a.XD_SUFFIX)
            select_algorithm_p(a.XD_SHARDS, tag_t, tag_r, sel_dir, nb, a.XD_ALLOWED, sealed=bool(a.XD_SUFFIX), suffix=a.XD_SUFFIX,
                               sealed_root=a.XD_OUT.parent)
            return 0
        if o.train_gbm:
            train_gbm_cli(a)
            return 0
        if o.summarize_xd5:
            if not o.xd5:
                raise SystemExit("[거부] XD-5 집계는 XD-4 시험 표를 연 뒤에만 한다(--xd5 를 함께 준다, 계획 0.3 열람 순서 4)")
            summarize_xd5(a.XD_SHARDS, TAGS["xd_test"] + a.XD_SUFFIX, TAGS["xd_learn"] + a.XD_SUFFIX, o.gbm_meta or None, a.threads,
                          root=a.XD_OUT.parent, allowed=a.XD_ALLOWED, suffix=a.XD_SUFFIX)
            return 0
        if o.export_selection:
            export_gbm_cli(a)
            return 0
        res = run_units(a, argv)
        rc = 1 if res["failed"] else 0
        if o.count_only or o.no_summarize:
            return rc
        if o.smoke:                                                           # 스모크: 모든 경로를 한 번씩 지난다(산출 표는 봉인 폴더)
            nb = int(a.nboot)
            summarize_alg(alg_dirs(a), tag_t, tag_r, nb, root=a.XD_OUT.parent, allowed=a.XD_ALLOWED, suffix=a.XD_SUFFIX)
            select_algorithm_p(a.XD_SHARDS, tag_t, tag_r, sel_dir, nb, a.XD_ALLOWED, sealed=True, suffix=a.XD_SUFFIX, sealed_root=a.XD_OUT.parent)
            if "xd_learn" in a.XD_EXPS:
                train_gbm_cli(a)
                export_gbm_cli(a)
                argv2 = _drop_opt(argv, "--exp") + ["--exp", "xd_test"]
                a2 = parse(argv2)
                r2 = run_units(a2, argv2)
                rc = max(rc, 1 if r2["failed"] else 0)
                summarize_test(a2.XD_SHARDS, TAGS["xd_test"] + a2.XD_SUFFIX, nb, root=a2.XD_OUT.parent, allowed=a2.XD_ALLOWED, suffix=a2.XD_SUFFIX)
            _log(f"[스모크] 최대 RSS {XB.max_rss_mb()} MB")
    return rc


if __name__ == "__main__":
    sys.exit(main())
