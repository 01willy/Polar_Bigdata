"""XA–XJ 최종 추가 실험 묶음(xbatch) 공용 골격.

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100, 2026-10-04 14:22:39 +0900). 이 모듈은 계획 1절(공통 규약)과
4절(묶음 목록)을 구현한다. 설계 보고서는 docs/research/2026-10-04/harness_implementation_plan.md 2절이다. 계획의 해석과 구현 결정은
docs/research/2026-10-04/impl_notes/xbatch_core.md 에 적었다. 가설·판정 규칙은 이 모듈에서 정하지 않는다(실험 모듈이 계획 그대로 쓴다).

구획(이 파일 안의 번호)
  1. 동결 모듈 로더. h40, h42, h54, h41 을 파일 경로에서 읽고 sha256 앞 16자를 계획 머리말 값과 대조한다. 다르면 중단한다.
  2. 고정값. δ 0.5·1.0·δ_rel(0.02 × P0 RMSE), 재표집 10,000회, 풀 PE1·PE2·주 4지역·3지역, 분할 seed(전이 1–5, XC 201–210, XC-r 211–220),
     CatBoost seed 0·1, 맹검 표지, 실험 이름.
  3. 실행 보호와 로컬 자원. WF_RESCALE=1(h54 와 같이 LG_RESCALE=1 도) 또는 --allow-local 이 없으면 스모크·세기·집계·시험 외 실행을 거부한다.
     허용 표지가 없으면 스레드 상한 4, 집계 재표집 상한 1,000 이다. 가용 메모리 확인과 최대 RSS 기록을 둔다.
  4. 출력 제한(계획 0.3). RMSE·Δ·판정 문자열이 든 줄을 표준 출력과 표준 오류에서 거른다.
  5. 인자. h54 의 인자 계약(a.SEEDS, a.threads, a.cb_iters, a._ha, a.PROC, a.LGD, a.subregion_map, a.SPLITS, a.G)을 만족하는 인자 객체.
  6. 자료. v3(fidelity_base_v3 + 토양 도일 v3)와 LGD 약관 확인분 실행 표(Tibet, Russia_C, NAtlantic_lic, Canada_expanded_lic). 약관 점검.
  7. 분할. 새 seed 의 제외 규칙(1절 '분할 seed' 행)과 부록 A 대조.
  8. 추출. h40.draw_cells(seed_of(대상, 모드, 분할, n, 추출)), 블록 분산 추출, 중첩 순열 추출(XC 의 무작위 팔).
  9. 채점. 셀 가중·블록 등가중 RMSE, 곡선 키, 재현 관문 비교(1절 허용 오차: 같은 노드·elm·hematite 0, 로컬·Rescale 1e-4 cm² 또는 상대 1e-9).
  10. 대비와 CI. 같은 라벨 집합 대비는 h40.contrast(= h4_common.boot_delta_blocks)와 보조 h42.boot_delta_common. 두 팔의 라벨 집합이 다른
      대비는 2단 재표집(h39 l43_region 의 일반화)과 보조 '2단 공통', 추출 조건부 보조 열.
  11. 판정. 4분 판정(δ 0.5, 1.0, δ_rel)과 '한계 의존', 비열등, p 값, Holm, 다수 n 종합(_rule3), 해석 문장의 다섯 갈래.
  12. 풀. h42.pool_rows 와 h42.region_inference 의 감싸기(층화 평균, 부분 풀, 소수 블록, HK 지역 수준 구간).
  13. 조각. <tag>__cpu__<대상>__<모드>__s<분할>[__<변형>]_{runs.csv, blocksse.npz, unit.json}, --resume 상태, 조각 → TMx.
  14. 봉인. data/processed/xbatch/<새 이름>/sealed/ 에만 쓰고 화면에는 행 수와 sha256 앞 16자만 남긴다.
  15. 산출 경로 보호. data/processed/xbatch/ 밖과 기존 결과 폴더에는 쓰지 않는다.
  16. 누설 시험 도구. 선택 밖 A 라벨과 B 라벨을 바꿔 예측이 같은지 본다(h54 시험 b, d, m 형식).
  17. XI warm_trim(계획 2.9, R10Unit 하위 클래스).
  18. 묶음. 제약 파일, 설치 판 확인, 입력 표 sha256, git 미커밋 경로.
  19. 프로세스 풀, GPU 결정성 설정(XD-learn 전용).
  20. 명령행. --self-check, --split-plan, --lic-check, --deps-check, --manifest. 모두 라벨 값을 쓰지 않는다(세기 범주).

모듈 작성자 규칙
  - x_*.py 는 numpy 보다 먼저 이 모듈을 불러 스레드 환경 변수를 정한다. GPU 를 쓰는 XD-learn 만 XBATCH_GPU=1 을 준 뒤 부른다.
  - 동결 모듈은 이 모듈의 H(h40), X(h42), W(h54)와 frozen('h41')로만 쓴다. 고치지 않고 다른 경로에서 다시 읽지 않는다.
  - 조각은 write_shard, 열람 순서 전 집계 표는 write_sealed 로만 쓴다. 스모크·세기·로컬 시험·집계의 화면 출력은 restricted_output() 안에서 한다.
  - 라벨 집합이 다른 두 팔의 대비는 region_stat_two_stage, 같은 라벨 집합 대비는 region_stat_same 을 쓰고 pool 로 풀을 만든다.

명령행(세기 범주, 라벨 값 미사용)
  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/xbatch_core.py --self-check
  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/xbatch_core.py --split-plan          # 부록 A 를 다시 내고 대조한다
  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/xbatch_core.py --lic-check
  python3 scripts/3_deep_learning/xbatch_core.py --deps-check                                          # Rescale 설치 뒤 판 확인 로그
  python3 scripts/3_deep_learning/xbatch_core.py --manifest work/xbatch_manifest.json                   # 묶음 정보(sha256)
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
import resource
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

# ================================================================ 0. 스레드와 장치(numpy 를 부르기 전)
LOCAL_MAX_THREADS = 4                                                       # 허용 표지 없는 실행의 스레드 상한(공유 서버, 1절 로컬 자원)


def run_permitted(argv=None) -> bool:
    """본 실행 허용 표지. 환경 변수 WF_RESCALE=1(h54 와 같이 LG_RESCALE=1 도 받는다) 또는 명령행 --allow-local."""
    av = sys.argv if argv is None else list(argv)
    return os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1" or "--allow-local" in av


def _peek_threads(argv=None) -> str:
    """--threads 를 미리 읽는다. 허용 표지가 없으면 LOCAL_MAX_THREADS 로 자른다(h54._peek_threads 와 같은 규칙)."""
    av = sys.argv if argv is None else list(argv)
    val = None
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            val = av[i + 1]
        elif v.startswith("--threads="):
            val = v.split("=", 1)[1]
    try:
        t = int(val) if val is not None else 1
    except ValueError:
        t = 1
    if not run_permitted(av):
        t = min(t, LOCAL_MAX_THREADS)
    return str(max(t, 1))


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _v in THREAD_VARS:
    os.environ.setdefault(_v, _peek_threads())
if os.environ.get("XBATCH_GPU", "") != "1":                                 # GPU 는 XD 의 DeepSets 정책에만 쓴다(1절 플랫폼)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore", module="threadpoolctl")
warnings.filterwarnings("ignore", message=".*glibc.*")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ================================================================ 1. 동결 모듈 로더
class FrozenModuleError(AssertionError):
    """동결 모듈의 sha256 이 계획 머리말 값과 다르거나 다른 파일에서 적재되었다."""


# 키 → (sys.modules 이름, 저장소 상대 경로, sha256 앞 16자). 계획 머리말 '이 문서를 쓰면서 계산한 것' 초판 항목의 값이다.
FROZEN = {
    "h40": ("h40_label_grid", "scripts/3_deep_learning/h40_label_grid.py", "7098f59dabe2b73f"),
    "h42": ("h42_label_grid_ext", "scripts/3_deep_learning/h42_label_grid_ext.py", "22e215e435e157be"),
    "h54": ("h54_workflow", "scripts/3_deep_learning/h54_workflow.py", "cf9a1f6d0929c892"),
    "h41": ("h41_validation_ladder", "scripts/2_evaluation/h41_validation_ladder.py", "492373e4d37b5ea4"),
}
FROZEN_SHA16 = {k: v[2] for k, v in FROZEN.items()}


def sha256_file(path, n=None) -> str:
    """파일의 sha256(16진). n 이 있으면 앞 n 자."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    s = h.hexdigest()
    return s[:int(n)] if n else s


def assert_frozen(key, path=None, expected=None) -> str:
    """동결 파일의 sha256 앞 16자를 대조한다. 다르면 FrozenModuleError. 반환은 실제 앞 16자."""
    name, rel, sha = FROZEN[key]
    p = Path(path) if path is not None else ROOT / rel
    want = str(expected or sha)
    got = sha256_file(p, 16)
    if got != want:
        raise FrozenModuleError(f"동결 모듈 {key}({p}) 의 sha256 앞 16자 {got} 가 계획 값 {want} 와 다르다. 동결 파일을 고치지 않는다")
    return got


def load_frozen(key):
    """동결 모듈을 파일 경로에서 읽어 sys.modules 에 등록한다(h42·h54 가 h40 을 읽는 방식과 같다). 이미 등록되어 있으면 그 모듈의 파일이
    동결 경로인지 확인하고 그대로 쓴다. 읽기 전에 sha256 을 대조한다."""
    name, rel, _ = FROZEN[key]
    path = (ROOT / rel).resolve()
    assert_frozen(key)
    mod = sys.modules.get(name)
    if mod is not None:
        mf = Path(str(getattr(mod, "__file__", "") or "")).resolve()
        if mf != path:
            raise FrozenModuleError(f"sys.modules['{name}'] 가 동결 경로가 아닌 {mf} 에서 적재되었다")
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


for _k in FROZEN:                                                           # 네 파일 모두 불러오기 전에 대조한다(h41 은 쓸 때 읽는다)
    assert_frozen(_k)
H = load_frozen("h40")
X = load_frozen("h42")
W = load_frozen("h54")


def frozen(key):
    """동결 모듈 하나(h40, h42, h54, h41). h41 은 처음 부를 때 읽는다(XH 전용)."""
    return dict(h40=H, h42=X, h54=W).get(key) or load_frozen(key)


import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, seed_of, boot_delta_blocks, block_sse  # noqa: E402,F401
from polar import m1_stats as MS                                                                     # noqa: E402
from polar.m1_core import eval_mask, half_split_blocks                                               # noqa: E402

# ================================================================ 2. 고정값(계획 1절. 바꾸면 사전 등록에서 벗어난다)
PLAN_DOC = "docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md"
PLAN_REVISION = "개정 1"
PLAN_COMMIT = "2678100"
PLAN_T0 = "2026-10-04 14:22:39 +0900"
FEATS = list(H.FEATS)                                                       # x25
KAPPA = float(W.KAPPA)                                                      # κ 10 수축
R_PS = float(W.R_PS)                                                        # 유사라벨 배수 10
LAMS = tuple(W.LAMS)                                                        # λ 0.25, 0.5, 1.0
LAM_BASE = float(W.LAM_BASE)                                                # λ 0.25
LAM_CV = float(W.LAM_CV)                                                    # 교차검증 λ 의 키 −1
CV_K = int(W.CV_K)
SEEDS = (0, 1)                                                              # CatBoost seed 0·1
LO, HI = W.LO, W.HI                                                         # catboost_lo, catboost
P0_GRP = H.P0_GRP
KEY_COLS = list(H.KEY_COLS)                                                 # (method, learner, alpha, placement, n, draw, seed, lam)
GRP_COLS = list(H.GRP_COLS)
NBOOT = 10000
LOCAL_NBOOT_MAX = 1000                                                      # 허용 표지 없는 집계의 재표집 상한(h54 와 같다)
DELTA = 0.5                                                                 # 주 δ(cm)
DELTA_AUX = 1.0                                                             # 보조 δ(cm)
REL_COEF = 0.02                                                             # δ_rel = 0.02 × P0 RMSE(WRAPUP 9.4)
NI_MARGIN = 0.5                                                             # 비열등 한계(cm)
ALPHA_ONE_SIDED = 0.025
HOLM_ALPHA = 0.05                                                           # 가족 오류율(양측 p 에 적용, 단측 0.025 와 같다)
MIN_BLOCKS_CI = int(MS.MIN_BLOCKS_CI)                                       # CI 를 내는 지역의 채점 블록 합집합 하한 8
MIN_POOL_REGIONS = int(H.MIN_POOL_REGIONS)                                  # 층화 평균 CI 의 지역 수 하한 2
FEW_BLOCKS = 5                                                              # 사용 분할의 최소 채점 블록 < 5 → '소수 블록'(WRAPUP 7.2 (a)4)
SMALL_EFFECT_CM = 0.5
SMALL_EFFECT_TXT = X.SMALL_EFFECT_TXT                                       # '통계적으로 구별되나 크기는 0.5 cm 미만'
FEW_BLOCKS_TXT = "소수 블록"
LIMIT_DEP_TXT = "한계 의존"
CI_DEP_TXT = "분할 독립 가정 의존"
DRAW_DEP_TXT = "추출 변동 의존"
UNCORRECTED_TXT = "보정 전 유의"
NA_VERDICTS = tuple(X.NA_VERDICTS)                                          # ('행 없음', '판정 불가')
VERDICTS4 = ("우세", "열세", "동등", "미결정")
BRANCHES = VERDICTS4 + ("판정 불가",)                                        # 해석 문장의 다섯 갈래
BLIND_LABELS = ("맹검", "비맹검 부분 포함", "재현(비맹검)", "등록 시점에 결과 존재(미열람)")
DESIGN_LABELS = ("결과 열람 뒤 설계", "사후 분석", "사후 설계(지역 표적)")

# 풀(1절 '풀 이름'). 저장소 이름 = '<대상 별칭>|<모드>'(h54 의 TUnit 이름 규칙)
MAIN4 = ["Lena|x", "Canada|x", "Russia_W|x", "Russia_E|x"]
PE1 = MAIN4 + ["Russia_C~lgd|x"]
PE2 = PE1 + ["Tibet_LGD|x"]
AUX3 = ["Lena|x", "Canada|x", "Alaska|x"]
INREGION3 = ["Alaska|r", "Lena|r", "Canada|r"]
POOLS = dict(MAIN4=MAIN4, PE1=PE1, PE2=PE2, AUX3=AUX3, INREGION3=INREGION3)

# 분할 seed(1절 '분할 seed' 행)
TRANSFER_SPLITS = tuple(range(1, 6))                                        # XD-alg, XB, XG, XJ, XD-learn
XC_SPLITS = tuple(range(201, 211))
XCR_SPLITS = tuple(range(211, 221))
INREGION_SPLITS = tuple(range(1, 26))                                       # WF6·WF9·XB·XE
WF10_SPLITS = tuple(range(1, 11))                                           # WF10·XI
PRIOR_LGXN4 = tuple(range(1, 201))                                          # LGX-N4 가 모드 x·i 에서 쓴 분할
PRIOR_LG = tuple(range(1, 6))                                               # LG·LGD·WF4 가 쓴 분할
LGXN4_TARGETS = ("Lena", "Canada", "Russia_W", "Russia_E", "Alaska") + tuple(H.SUB_AL) + tuple(H.SUB_OTHER)
STRUCT_MIN5_ALIASES = ("Lena", "Canada", "Russia_W", "Russia_E", "Russia_C~lgd", "Tibet_LGD", "Alaska")   # PE1·PE2·알래스카
MIN_EVAL_BLOCKS = 2
MIN_EVAL_BLOCKS_STRUCT = 5
SPLIT_FAMILIES = dict(xc=XC_SPLITS, xcr=XCR_SPLITS)

# 실험 이름(paper/registry/experiments.csv, 계획 0.1)과 산출 경로
EXP_NAMES = dict(XA="XA_c2_gain_decomposition", XB="XB_multisource_stacking", XC="XC_workflow_end_to_end", XD="XD_placement_policy",
                 XE="XE_hires_covariates", XF="XF_new_regions", XG="XG_product_comparison", XH="XH_validation_ladder",
                 XI="XI_climate_extrapolation_retest", XJ="XJ_tempderived_aux_labels")
XBATCH_ROOT = ROOT / "data" / "processed" / "xbatch"
CONSTRAINTS = ROOT / "scripts" / "rescale" / "xbatch_constraints.txt"
PINNED = {"catboost": "1.2.10", "scikit-learn": "1.9.1", "pandas": "2.3.3", "scipy": "1.17.1", "numpy": "2.1.1"}   # 1절 '패키지 판'


# ================================================================ 3. 실행 보호와 로컬 자원
FREE_MODES = ("smoke", "count", "summarize", "test")


def guard(mode, argv=None) -> bool:
    """실행 보호(1절 '실행 보호'). mode = smoke, count, summarize, test 는 허용한다. run(사전 점검·본 실행)은 허용 표지가 있을 때만 허용한다.
    자료를 읽기 전에 부른다. 거부하면 SystemExit."""
    if mode in FREE_MODES:
        return True
    if mode == "run" and run_permitted(argv):
        return True
    raise SystemExit(f"[거부] 모드 {mode} 는 Rescale 작업에서 한다. 작업 명령에 WF_RESCALE=1 또는 --allow-local 을 준다(계획 1절 '실행 보호')")


def local_limits(threads, nboot=None, argv=None):
    """허용 표지가 없으면 스레드 ≤ 4, 집계 재표집 ≤ 1,000. 반환 (threads, nboot)."""
    t = max(int(threads), 1)
    b = None if nboot is None else int(nboot)
    if not run_permitted(argv):
        t = min(t, LOCAL_MAX_THREADS)
        if b is not None:
            b = min(b, LOCAL_NBOOT_MAX)
    return t, b


def mem_available_gb() -> float:
    """/proc/meminfo 의 MemAvailable(GB). 읽을 수 없으면 NaN."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return float("nan")


def require_memory(min_gb=30.0, wait_s=0.0, poll_s=60.0) -> float:
    """시작 전 가용 메모리 확인(1절 로컬 자원: 30 GB 아래면 기다린다). wait_s 동안 poll_s 간격으로 다시 보고, 그래도 모자라면 SystemExit."""
    t0 = time.time()
    while True:
        g = mem_available_gb()
        if not np.isfinite(g) or g >= float(min_gb):
            return g
        if time.time() - t0 >= float(wait_s):
            raise SystemExit(f"[대기 초과] 가용 메모리 {g:.1f} GB 가 하한 {min_gb} GB 아래다(계획 1절 로컬 자원)")
        time.sleep(float(poll_s))


def limit_memory(gb=10.0) -> bool:
    """프로세스 주소 공간 상한(RLIMIT_AS, 1절의 prlimit --as 와 같은 효과). 설정하면 True. 이미 더 낮은 상한이 있으면 그대로 둔다."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        lim = int(float(gb) * 2 ** 30)
        if soft != resource.RLIM_INFINITY and soft <= lim:
            return True
        resource.setrlimit(resource.RLIMIT_AS, (lim, hard))
        return True
    except (ValueError, OSError):
        return False


def max_rss_mb() -> float:
    """이 프로세스와 끝난 자식 프로세스의 최대 RSS(MB, 메타 기록용)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(r, c) / 1024.0, 1)


def smoke_env_report() -> dict:
    """로컬 스모크 규약(코어 4개, 스레드 2) 점검. 어긋나면 warn 목록에 적는다(중단하지 않는다)."""
    warn = []
    th = os.environ.get("OMP_NUM_THREADS", "")
    if not run_permitted():
        if th and th.isdigit() and int(th) > 2:
            warn.append(f"OMP_NUM_THREADS={th} > 2")
        if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) > 4:
            warn.append(f"코어 묶음 {len(os.sched_getaffinity(0))}개 > 4(taskset -c 로 4개에 묶는다)")
    return dict(threads=th, cores=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else -1, warn=warn,
                mem_available_gb=round(mem_available_gb(), 1), permitted=run_permitted())


# ================================================================ 4. 출력 제한(계획 0.3)
FORBIDDEN_OUT = re.compile(r"판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각|ci_lo|ci_hi|p_two|p_boot|p_ni")
SHELL_FILTER = "2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'"   # WRAPUP 783행 규칙에 RMSE·판정어를 더했다


def allowed_line(line) -> bool:
    """출력 제한에 걸리지 않는 줄인가."""
    return FORBIDDEN_OUT.search(str(line)) is None


class FilteredStream:
    """줄 단위로 출력 제한 패턴에 걸리는 줄을 버리는 스트림. 버린 줄 수만 센다."""

    def __init__(self, target, pattern=FORBIDDEN_OUT):
        self.target, self.pattern, self.buf, self.dropped = target, pattern, "", 0

    def write(self, s):
        s = str(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self._emit(line + "\n")
        return len(s)

    def _emit(self, line):
        if self.pattern.search(line):
            self.dropped += 1
        else:
            self.target.write(line)

    def flush(self):
        if self.buf:
            self._emit(self.buf)
            self.buf = ""
        try:
            self.target.flush()
        except (AttributeError, ValueError):
            pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.target, "encoding", "utf-8")


@contextlib.contextmanager
def restricted_output(report=True):
    """스모크·세기·로컬 시험·집계의 화면 출력에서 RMSE·Δ·판정 줄을 거른다(계획 0.3). C 확장이 파일 기술자에 직접 쓰는 출력은 거르지 못하므로
    셸에서도 SHELL_FILTER 를 붙인다."""
    o, e = sys.stdout, sys.stderr
    fo, fe = FilteredStream(o), FilteredStream(e)
    sys.stdout, sys.stderr = fo, fe
    try:
        yield fo
    finally:
        fo.flush(); fe.flush()
        sys.stdout, sys.stderr = o, e
        if report and (fo.dropped or fe.dropped):
            print(f"[출력 제한] 걸러낸 줄 {fo.dropped + fe.dropped}개", flush=True)


def safe_print(*args, sep=" ", **kw):
    """출력 제한 패턴에 걸리면 그 줄을 쓰지 않고 표지만 남긴다."""
    txt = sep.join(str(a) for a in args)
    if allowed_line(txt):
        print(txt, **kw)
    else:
        print("[출력 제한] 한 줄을 걸렀다", **kw)


# ================================================================ 5. 인자(h54 인자 계약)
def h54_args(exp="wf4", splits=TRANSFER_SPLITS, grid=None, threads=1, cb_iters=200, seeds=len(SEEDS), nboot=NBOOT, data_dir="data/processed",
             lgd_dir="data/processed/lgd/run_tables", out_dir=None, tag="x", draws_cap=0, allow_local=False, extra=()):
    """h54.parse_args 로 만든 인자 객체에 분할 목록(a.SPLITS, 새 seed 포함)과 격자(a.G[exp])를 넣는다. h54 의 단위(RUnit, TUnit, R10Unit)와
    자료 함수(get_data, build_rctx, build_rctx10)가 그대로 받는다. 허용 표지가 없으면 스레드는 h54 규칙대로 4 로 잘린다."""
    argv = ["--threads", str(int(threads)), "--cb-iters", str(int(cb_iters)), "--seeds", str(int(seeds)), "--nboot", str(int(nboot)),
            "--data-dir", str(data_dir), "--lgd-dir", str(lgd_dir), "--out-dir", str(out_dir or (XBATCH_ROOT / "_h54_unused")),
            "--tag", str(tag), "--draws-cap", str(int(draws_cap))] + (["--allow-local"] if allow_local else []) + [str(v) for v in extra]
    a = W.parse_args(argv)
    a.SPLITS = [int(v) for v in splits]
    a.G = dict(a.G)
    if grid is not None:
        a.G[exp] = [int(v) for v in grid]
    a.XEXP = str(exp)
    return a


# ================================================================ 6. 자료
# 저장소 별칭 → (실행 표 디렉터리, 표 안의 대상 이름, 점 추정만). 앞의 셋은 h54.LGD_SPECS 와 같다. Canada~exp~lic 은 xb_t 의 민감도 행과
# XI 의 자료(약관 확인분 캐나다 확충판, 787셀)다.
RUN_TABLES = {"Tibet_LGD": ("Tibet", "Tibet_LGD", False), "NAtlantic~lic": ("NAtlantic_lic", "NAtlantic", True),
              "Russia_C~lgd": ("Russia_C", "Russia_C", False), "Canada~exp~lic": ("Canada_expanded_lic", "Canada", False)}
RUN_TABLE_DIRS = tuple(v[0] for v in RUN_TABLES.values())                   # 4절 약관 점검 대상(네 실행 표 모두)
LGD_DIR = ROOT / "data" / "processed" / "lgd" / "run_tables"
LABELS_V4 = ROOT / "data" / "processed" / "fidelity_base_v4_labels.csv"
assert all(W.LGD_SPECS[k] == RUN_TABLES[k] for k in W.LGD_SPECS), "h54.LGD_SPECS 와 RUN_TABLES 의 앞 세 항목이 달라졌다"


def resolve(alias):
    """저장소 별칭 → (실행 표 디렉터리 또는 None(v3), 표 안의 대상 이름, 점 추정만)."""
    alias = str(alias).split("|")[0]
    if alias in RUN_TABLES:
        return RUN_TABLES[alias]
    return None, alias, alias in H.MAIN_POINT


def is_point_only(alias) -> bool:
    """점 추정만 내는 대상(LGD 부적격 NAtlantic~lic, v3 의 Russia_C·Greenland). 별칭 표를 먼저 본다(Russia_C~lgd 는 CI 대상이다)."""
    return bool(resolve(alias)[2])


def is_point_only_name(name) -> bool:
    """저장소 이름('<별칭>[~변형]|<모드>')의 점 추정 여부. 별칭이 실행 표에 있으면 그 값, 아니면 '~' 앞 이름으로 본다."""
    al = str(name).split("|")[0]
    if al in RUN_TABLES:
        return RUN_TABLES[al][2]
    return al.split("~")[0] in H.MAIN_POINT


def store_name(alias, mode) -> str:
    return f"{alias}|{mode}"


def get_data(a, alias=None):
    """(h40 인자, h40.Data). alias 가 없거나 v3 대상이면 a.PROC(v3), 실행 표 별칭이면 a.LGD/<디렉터리>. h54.get_data 의 프로세스 캐시를 쓴다."""
    spec = resolve(alias)[0] if alias is not None else None
    return W.get_data(a, spec)


def build_tctx(a, alias, mode, split, plan_row=None):
    """전이 문맥(h40.build_ctx 그대로). 실행 표 별칭(Canada~exp~lic 포함)을 받는다. plan_row(split_plan 의 행)를 주면 새 seed 규칙의 분할
    구조를 메타에 덮어쓴다(h54.make_tm 이 unit.json 의 dup_of·valid 를 읽는다). 분할은 a.SPLITS 안에 있어야 한다."""
    spec, tgt, _ = resolve(alias)
    if int(split) not in [int(v) for v in a.SPLITS]:
        raise ValueError(f"분할 {split} 이 a.SPLITS 에 없다(h40.Data.split_structure 는 a.SPLITS 만 계산한다)")
    HA, D = W.get_data(a, spec)
    c = H.build_ctx(D, HA, tgt, mode, int(split))
    c.meta.update(alias=str(alias), data_spec=spec or "v3")
    if plan_row is not None:
        _apply_plan_meta(c.meta, plan_row)
    return c


def build_rctx(a, target, split, plan_row=None):
    """지역 내 문맥(h54.build_rctx 그대로, v3). plan_row 를 주면 그 분할 구조(dup_of, valid, n_valid_splits, n_unique_splits)를 쓴다."""
    c = W.build_rctx(a, target, int(split), info=plan_row)
    if plan_row is not None:
        _apply_plan_meta(c.meta, plan_row)
    return c


def _apply_plan_meta(meta, row):
    meta.update(dup_of=int(row.get("dup_of", -1)), valid=bool(row.get("valid", True)), n_valid_splits=int(row.get("n_valid_splits", 0)),
                n_unique_splits=int(row.get("n_unique_splits", 0)), split_status=str(row.get("status", "ok")),
                split_rule="xbatch 1절 새 seed 제외 규칙")


def data_sha(a, alias=None) -> str:
    """자료 판 식별(h54.data_sha 와 같은 형식: sha1 앞 12자). 실행 표는 기본 표와 토양 표, v3 는 하위 지역 대응표도 넣는다."""
    spec = resolve(alias)[0] if alias is not None else None
    if spec is not None:
        d = a.LGD / spec
        return f"{W.file_sha(d / 'fidelity_base_v3.csv')}:{W.file_sha(d / 'e5_soil_tdd_v3.csv')}"
    return f"{W.file_sha(a.PROC / 'fidelity_base_v3.csv')}:{W.file_sha(a.PROC / 'e5_soil_tdd_v3.csv')}:{W.file_sha(a.PROC / a.subregion_map)}"


def lic_check(lgd_dir=LGD_DIR, specs=RUN_TABLE_DIRS, labels=LABELS_V4, strict=True) -> list:
    """4절 약관 점검: 실행 표의 loc_id 가운데 fidelity_base_v4_labels.csv 의 lic_unverified = 1 인 셀이 없어야 한다(라벨 값 미사용).
    반환 [dict(spec, n_rows, n_unverified, sha256_16)]. strict 이면 하나라도 있을 때 SystemExit."""
    lab = pd.read_csv(labels, usecols=["loc_id", "lic_unverified"], low_memory=False)
    bad = set(lab.loc[lab.lic_unverified == 1, "loc_id"])
    out = []
    for s in specs:
        p = Path(lgd_dir) / s / "fidelity_base_v3.csv"
        t = pd.read_csv(p, usecols=["loc_id"], low_memory=False)
        out.append(dict(spec=s, n_rows=int(len(t)), n_unverified=int(t.loc_id.isin(bad).sum()), sha256_16=sha256_file(p, 16),
                        soil_link=os.readlink(Path(lgd_dir) / s / "e5_soil_tdd_v3.csv") if (Path(lgd_dir) / s / "e5_soil_tdd_v3.csv").is_symlink() else ""))
    if strict and any(r["n_unverified"] for r in out):
        raise SystemExit("[약관] 실패: " + ", ".join(f"{r['spec']} {r['n_unverified']}셀" for r in out if r["n_unverified"]))
    return out


# ================================================================ 7. 분할(1절 '분할 seed' 행, 부록 A)
def prior_seeds_of(alias):
    """그 대상에서 이미 평가에 쓴 분할. LGX-N4 대상(주 4지역, 알래스카, 하위 지역 10)은 1–200, LGD 실행 표 대상과 그 밖은 1–5."""
    al = str(alias).split("|")[0]
    if al in RUN_TABLES:
        return PRIOR_LG
    return PRIOR_LGXN4 if al in LGXN4_TARGETS else PRIOR_LG


def min_eval_blocks_of(alias, family) -> int:
    """채점 블록 하한. XC·XC-r 의 PE1·PE2·알래스카 대상은 5(구조 규칙, 개정 1), 그 밖은 2."""
    al = str(alias).split("|")[0]
    return MIN_EVAL_BLOCKS_STRUCT if (family in SPLIT_FAMILIES and al in STRUCT_MIN5_ALIASES) else MIN_EVAL_BLOCKS


def split_plan_new(D, target, new_seeds, prior_seeds=(), min_eval_blocks=MIN_EVAL_BLOCKS):
    """새 seed 분할의 구조와 제외(라벨 값 미사용). 규칙(1절, 적용 순서):
      (1) 앞서 쓴 분할(prior_seeds)과 A 블록 집합이 같거나 여집합 → prior_dup / prior_mirror
      (2) 새 범위의 앞 seed 와 A 블록 집합이 같거나 여집합 → dup_new / mirror_new(빠진 seed 와의 비교도 한다)
      (3) 채점 블록 < 2 → nb_eval<2
      (4) 채점 블록 < min_eval_blocks(5) → nb_eval<5
    빈 자리를 다른 seed 로 채우지 않는다. 반환 (실행 분할 목록, 행 목록). 행은 h40.Data.split_structure 의 키(dup_of, valid, n_A, nb_A,
    n_eval, nb_eval, n_valid_splits, n_unique_splits)와 status, ref_split, eval_blocks 를 담는다."""
    df = D.df
    t_idx = np.asarray(D.target_idx(target))
    blk = df.block.values
    allb = frozenset(blk[t_idx].tolist())
    prior = {}
    for q in prior_seeds:
        A_q, _ = half_split_blocks(df, t_idx, int(q))
        aq = frozenset(blk[A_q].tolist())
        prior.setdefault(aq, (int(q), "prior_dup"))
        prior.setdefault(allb - aq, (int(q), "prior_mirror"))
    rows, seen = [], []
    for sp in new_seeds:
        sp = int(sp)
        A_idx, B_idx = half_split_blocks(df, t_idx, sp)
        a = frozenset(blk[A_idx].tolist())
        evB = B_idx[eval_mask(df.iloc[B_idx])]
        eb = np.unique(blk[evB])
        status, ref = "ok", -1
        if a in prior:
            ref, status = prior[a]
        else:
            for q, aq in seen:
                if a == aq:
                    status, ref = "dup_new", q
                    break
                if a == allb - aq:
                    status, ref = "mirror_new", q
                    break
        if status == "ok" and len(eb) < MIN_EVAL_BLOCKS:
            status = f"nb_eval<{MIN_EVAL_BLOCKS}"
        elif status == "ok" and len(eb) < int(min_eval_blocks):
            status = f"nb_eval<{int(min_eval_blocks)}"
        seen.append((sp, a))
        rows.append(dict(split=sp, status=status, ref_split=int(ref), n_A=int(len(A_idx)), nb_A=int(len(a)), n_eval=int(len(evB)),
                         nb_eval=int(len(eb)), eval_blocks=[str(v) for v in eb.tolist()], min_eval_blocks=int(min_eval_blocks),
                         dup_of=int(ref) if status in ("prior_dup", "dup_new") else -1,
                         mirror_of=int(ref) if status in ("prior_mirror", "mirror_new") else -1, valid=status == "ok"))
    keep = [r["split"] for r in rows if r["status"] == "ok"]
    n_unique = sum(1 for r in rows if not r["status"].startswith(("prior", "dup_new", "mirror_new")))
    for r in rows:
        r.update(n_valid_splits=len(keep), n_unique_splits=int(n_unique))
    return keep, rows


def split_plan(a, alias, family="xc", seeds=None, check=True):
    """XC(201–210)·XC-r(211–220) 분할 계획. 별칭으로 자료를 고르고 앞서 쓴 분할과 채점 블록 하한을 정한다. check 이면 부록 A 와 대조한다
    (부록 A 에 있는 행만). 반환 (실행 분할, 행 목록)."""
    spec, tgt, _ = resolve(alias)
    _, D = W.get_data(a, spec)
    seeds = SPLIT_FAMILIES[family] if seeds is None else tuple(int(v) for v in seeds)
    keep, rows = split_plan_new(D, tgt, seeds, prior_seeds_of(alias), min_eval_blocks_of(alias, family))
    if check and (family, str(alias).split("|")[0]) in APPENDIX_A and tuple(seeds) == SPLIT_FAMILIES.get(family):
        check_appendix_a(family, str(alias).split("|")[0], rows)
    return keep, rows


def _status_class(st) -> str:
    st = str(st)
    if st.startswith("prior"):
        return "prior"
    if st in ("dup_new", "mirror_new"):
        return "dupnew"
    if st.startswith("nb_eval<"):
        return "nb<" + st.split("<", 1)[1]
    return st


def plan_summary(rows) -> dict:
    """분할 계획 행의 요약(셀 수와 블록 수만): 유효 분할 수, 뺀 seed 와 사유, |A| 범위, 분할별 채점 블록 범위, 채점 블록 합집합."""
    ok = [r for r in rows if r["status"] == "ok"]
    excl = {int(r["split"]): r["status"] for r in rows if r["status"] != "ok"}
    rng = (lambda k: (int(min(r[k] for r in ok)), int(max(r[k] for r in ok))) if ok else None)  # noqa: E731
    U = set().union(*[set(r["eval_blocks"]) for r in ok]) if ok else set()
    return dict(n_valid=len(ok), n_seeds=len(rows), excluded=excl, excluded_class=dict(Counter(_status_class(s) for s in excl.values())),
                n_A=rng("n_A"), nb_eval=rng("nb_eval"), nb_union=len(U), nb_eval_min=int(min(r["nb_eval"] for r in ok)) if ok else 0)


# 부록 A(개정 1 계산, 라벨 값 미사용). 키 (계열, 별칭) → 유효 분할 수, 사유별 제외 수, 사유를 적은 seed, |A| 범위, 채점 블록 범위, 합집합.
APPENDIX_A = {
    ("xc", "Lena"): dict(n=9, excl={"nb<5": 1}, seeds={207: "nb<5"}, nA=(946, 1975), nb=(5, 13), union=20),
    ("xc", "Canada"): dict(n=10, excl={}, nA=(314, 410), nb=(9, 27), union=36),
    ("xc", "Russia_W"): dict(n=10, excl={}, nA=(14, 16), nb=(7, 11), union=18),
    ("xc", "Russia_E"): dict(n=10, excl={}, nA=(14, 16), nb=(7, 12), union=19),
    ("xc", "Russia_C~lgd"): dict(n=9, excl={"nb<5": 1}, seeds={208: "nb<5"}, nA=(28, 34), nb=(5, 10), union=13),
    ("xc", "Tibet_LGD"): dict(n=10, excl={}, nA=(66, 70), nb=(13, 19), union=34),
    ("xc", "Alaska"): dict(n=10, excl={}, nA=(6064, 8501), nb=(10, 56), union=74),
    ("xc", "AL-1"): dict(n=3, excl={"prior": 7}, nA=(436, 4720), nb=(3, 5), union=9),
    ("xc", "AL-2"): dict(n=7, excl={"prior": 3}, nA=(1488, 1749), nb=(6, 18), union=20),
    ("xc", "AL-3"): dict(n=10, excl={}, nA=(264, 409), nb=(4, 11), union=16),
    ("xc", "AL-4"): dict(n=0, excl={"prior": 10}),
    ("xc", "AL-5"): dict(n=8, excl={"prior": 2}, nA=(198, 410), nb=(5, 13), union=17),
    ("xc", "AL-6"): dict(n=0, excl={"prior": 10}),
    ("xc", "CA-2"): dict(n=10, excl={}, nA=(123, 237), nb=(3, 10), union=13),
    ("xc", "CA-3"): dict(n=10, excl={}, nA=(110, 245), nb=(2, 14), union=18),
    ("xc", "LE-1"): dict(n=8, excl={"prior": 2}, nA=(443, 1438), nb=(2, 7), union=14),
    ("xc", "LE-2"): dict(n=0, excl={"prior": 10}),
    ("xc", "NAtlantic~lic"): dict(n=4, excl={"prior": 1, "dupnew": 5}, nA=(5, 15), nb=(4, 5), union=6),
    ("xcr", "Alaska"): dict(n=10, excl={}, nA=(5566, 8794), nb=(24, 56), union=74),
    ("xcr", "Lena"): dict(n=8, excl={"nb<5": 2}, seeds={215: "nb<5", 217: "nb<5"}, nA=(931, 1948), nb=(6, 17), union=20),
    ("xcr", "Canada"): dict(n=10, excl={}, nA=(308, 447), nb=(15, 24), union=36),
}
# 부록 A 머리 문장의 분할 1–5 재계산 기록(h40 규칙: 중복·채점 블록 < 2 제외)의 최소 채점 블록
SPLITS_1_5_MIN_NB = {"Lena": 5, "Canada": 12, "Russia_W": 7, "Russia_E": 8, "Alaska": 21, "Russia_C~lgd": 6, "Tibet_LGD": 14}
# XC 의 모드(부록 A 의 대상 열): 거시 지역과 실행 표 대상은 x, 하위 지역은 i·x(구조는 모드와 무관하다)
XC_TARGETS = [k[1] for k in APPENDIX_A if k[0] == "xc"]
XCR_TARGETS = [k[1] for k in APPENDIX_A if k[0] == "xcr"]


def check_appendix_a(family, alias, rows) -> dict:
    """분할 계획이 부록 A 의 기록(유효 분할 수, 사유별 제외 수, 사유를 적은 seed, |A| 범위, 채점 블록 범위, 합집합)과 같은지 단언한다."""
    exp = APPENDIX_A[(family, alias)]
    s = plan_summary(rows)
    err = []
    if s["n_valid"] != exp["n"]:
        err.append(f"유효 분할 {s['n_valid']} ≠ {exp['n']}")
    if s["excluded_class"] != exp["excl"]:
        err.append(f"제외 사유 {s['excluded_class']} ≠ {exp['excl']}")
    for sp, cls in exp.get("seeds", {}).items():
        if _status_class(s["excluded"].get(int(sp), "ok")) != cls:
            err.append(f"seed {sp} 의 사유 {s['excluded'].get(int(sp), 'ok')} ≠ {cls}")
    if exp["n"] > 0:
        for k, kk in (("nA", "n_A"), ("nb", "nb_eval")):
            if tuple(s[kk]) != tuple(exp[k]):
                err.append(f"{kk} 범위 {s[kk]} ≠ {exp[k]}")
        if s["nb_union"] != exp["union"]:
            err.append(f"채점 블록 합집합 {s['nb_union']} ≠ {exp['union']}")
    if err:
        raise AssertionError(f"부록 A 불일치({family}|{alias}): " + "; ".join(err))
    return s


# ================================================================ 8. 추출
def draw_cells(target, mode, split, n, draw, nA):
    """셀 무작위 추출(h40.draw_cells, seed = seed_of(대상, 모드, 분할, n, 추출)). n = 0 은 빈 배열, n = −1 은 A 전체."""
    return H.draw_cells(target, mode, int(split), int(n), int(draw), int(nA))


def draw_blocks(target, mode, split, n, draw, blkA):
    """블록 분산 추출(h40.draw_blocks, S2)."""
    return H.draw_blocks(target, mode, int(split), int(n), int(draw), blkA)


def nested_order(tag, target, mode, split, draw, nA):
    """중첩 순열 RNG(seed_of(tag, 대상, 모드, 분할, 추출)).permutation(|A|). XC 의 무작위 팔은 tag = 'xc-rand'(계획 2.3)."""
    return np.random.RandomState(seed_of(tag, target, mode, int(split), int(draw))).permutation(int(nA))


def draw_nested(tag, target, mode, split, n, draw, nA):
    """중첩 추출: nested_order 의 앞 n 개(순서 보존). n = 0 은 빈 배열, n = −1 은 A 전체(0..|A|−1)."""
    if int(n) == 0:
        return np.zeros(0, int)
    if int(n) < 0:
        return np.arange(int(nA))
    return nested_order(tag, target, mode, split, draw, nA)[:int(n)].astype(int)


def cells_grid(grid, draws, nA, zero_n=False):
    """(n, 추출) 목록. 전량(−1)은 추출 하나, n = 0 은 zero_n 일 때만 하나, 양수 n 은 |A| 미만만(h40.cells_of·h54.cells_for 와 같은 규칙).
    draws 는 정수 또는 n → 추출 수 함수."""
    out = []
    for n in grid:
        n = int(n)
        if n == -1:
            out.append((-1, 0))
        elif n == 0:
            if zero_n:
                out.append((0, 0))
        elif 0 < n < int(nA):
            k = int(draws(n) if callable(draws) else draws)
            out += [(n, d) for d in range(k)]
    return out


# ================================================================ 9. 채점
def rmse(y, pred) -> float:
    """유한 셀의 RMSE(cm)."""
    y = np.asarray(y, float); p = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.any() else float("nan")


def rmse_rows(S, C):
    """블록 SSE·셀 수 행렬 → 키별 (셀 가중 RMSE, 블록 등가중 RMSE)."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        return H4._rmse_rows(np.asarray(S, float), np.asarray(C, float)), H4._beq_rows(np.asarray(S, float), np.asarray(C, float))


def grp_rmse2(tm, g):
    """곡선 키의 (셀 가중, 블록 등가중) RMSE. 추출·seed 평균 뒤 분할 평균(h39.grp_rmse2 와 같은 정의)."""
    vc, vb = [], []
    for sp, st in tm.used.items():
        ks = tm.idx[sp].get(g)
        if not ks:
            continue
        S, C = st.matrices(ks)
        rc, rb = rmse_rows(S, C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vc.append(float(np.nanmean(rc))); vb.append(float(np.nanmean(rb)))
    return (float(np.mean(vc)) if vc else float("nan"), float(np.mean(vb)) if vb else float("nan"))


gk = W.gk                                                                  # 곡선 키 (method, learner, alpha, placement, n, lam)
nlab = W.nlab

# 재현 관문 허용 오차(1절 '재현 관문 허용 오차'): 수준 → (절대 |블록 SSE 차| cm², 상대). 같은 노드 종류와 elm·hematite 사이는 0,
# 로컬과 Rescale 사이는 1e-4 cm² 또는 상대 1e-9. 로컬 기준 관문은 CatBoost·물리식·ridge 키로 한정한다(key_fn 으로 고른다).
GATE_TOL = {"same_node": (0.0, 0.0), "elm_hematite": (0.0, 0.0), "local_rescale": (1e-4, 1e-9)}


def gate_compare(new, ref, level="same_node", key_fn=None, key_map=None) -> pd.DataFrame:
    """재현 관문: 두 저장소 묶음({(저장소 이름, 분할): BlockStore}, load_stores 결과)의 공통 키마다 블록 SSE 의 최대 절대 차와 상대 차.
    key_fn = 비교할 새 키의 술어, key_map = 새 키 → 기준 키(예: XC 의 placement 'rand' → WF4 의 'cell'). 같은 노드·elm·hematite 는 차 0 이어야
    통과이고, local_rescale 은 절대 1e-4 cm² 또는 상대 1e-9 안이면 통과다. 셀 수 벡터가 다르면 실패다. 표의 열에는 RMSE·Δ 가 없다."""
    ta, tr = GATE_TOL[level]
    rows = []
    for k in sorted(set(new) & set(ref)):
        a, b = new[k], ref[k]
        if not np.array_equal(np.asarray(a.blocks).astype(str), np.asarray(b.blocks).astype(str)):
            rows.append(dict(store=k[0], split=int(k[1]), key="", max_abs_dsse=np.nan, max_rel_dsse=np.nan, cnt_equal=False, ok=False,
                             note="채점 블록 집합 불일치"))
            continue
        for key in a.keys:
            if key_fn is not None and not key_fn(key):
                continue
            rk = tuple(key_map(key)) if key_map is not None else key
            if rk not in b:
                continue
            sa, ca = a.get(key); sb, cb = b.get(rk)
            dd = float(np.max(np.abs(sa - sb))) if len(sa) else 0.0
            rel = dd / max(float(np.max(np.abs(sb))) if len(sb) else 0.0, 1e-300)
            ceq = bool(np.array_equal(ca, cb))
            ok = ceq and (dd <= ta or (tr > 0 and rel <= tr))
            rows.append(dict(store=k[0], split=int(k[1]), key=json.dumps(list(key), ensure_ascii=False), max_abs_dsse=dd, max_rel_dsse=rel,
                             cnt_equal=ceq, ok=bool(ok), note=""))
    return pd.DataFrame(rows, columns=["store", "split", "key", "max_abs_dsse", "max_rel_dsse", "cnt_equal", "ok", "note"])


def gate_summary(df) -> dict:
    """관문 표의 요약(화면 출력용, 수만): 공통 키 수, 실패 키 수, 통과 여부."""
    n = int(len(df)); nf = int((~df.ok.astype(bool)).sum()) if n else 0
    return dict(n_keys=n, n_fail=nf, passed=bool(n > 0 and nf == 0))


def group_tag(g) -> str:
    """곡선 키의 문자열 표지(2단 재표집의 팔 seed 에 쓴다)."""
    return "|".join(str(v) for v in g)


# ================================================================ 10. 대비와 CI
def boot_cache_clear():
    """h40 이 감싼 boot_weights 캐시를 비운다(메모리 상한. 지역 묶음 계산 뒤 부른다)."""
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()


def region_stat_same(tm, gA, gB):
    """같은 라벨 집합 대비(두 방법이 같은 분할·추출·seed 의 라벨을 쓰거나 한쪽이 라벨을 쓰지 않는다). 주 분포 = h40.contrast
    (h4_common.boot_delta_blocks, seed = tm.seed 라 같은 대상의 모든 대비가 재표집 번호를 공유한다), 보조 = h42.boot_delta_common.
    CI 는 채점 블록 합집합 8 이상일 때만 낸다(h42.region_stats). P0 RMSE(δ_rel)를 붙인다."""
    s = X.region_stats(tm, gA, gB)
    if s is None:
        return None
    p0c, p0b = grp_rmse2(tm, P0_GRP)
    s.update(p0_cell=p0c, p0_beq=p0b, ci_kind="same", xdist=None, xdist_beq=None)
    return s


def _draw_groups(keys):
    """키(추출 × seed) → 추출별 행 색인 묶음(추출 번호 순, h39._draw_R 와 같다)."""
    by = {}
    for j, k in enumerate(keys):
        by.setdefault(int(k[5]), []).append(j)
    return [by[d] for d in sorted(by)]


def _arm_two_stage(S, C, Wm, grp, nboot, rs_seed):
    """팔 하나의 점 추정(추출별 seed 평균 RMSE)과 2단 분포. 블록 가중 Wm 은 두 팔이 공유하고 추출 번호는 팔마다 따로 복원 추출한다."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        r_pt = np.array([np.nanmean(np.sqrt(S[g].sum(1) / C[g].sum(1))) for g in grp])
        v = np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), np.nan)
        r_ptb = np.array([np.nanmean(np.nanmean(v[g], 1)) for g in grp])
        if Wm is None:
            return r_pt, r_ptb, None, None
        Rk = np.sqrt((S @ Wm.T) / (C @ Wm.T))
        vv = np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), 0.0)
        mm = (C > 0).astype(float)
        Ek = (vv @ Wm.T) / (mm @ Wm.T)
        Rd = np.vstack([np.nanmean(Rk[g], 0) for g in grp])
        Ed = np.vstack([np.nanmean(Ek[g], 0) for g in grp])
        I = np.random.RandomState(rs_seed).randint(0, len(grp), size=(int(nboot), len(grp)))
        cols = np.arange(int(nboot))[:, None]
        return r_pt, r_ptb, np.nanmean(Rd.T[cols, I], 1), np.nanmean(Ed.T[cols, I], 1)


def two_stage(tm, gA, gB, nboot, seed, tags=None, common=False, w_seed=None):
    """두 팔의 라벨 집합이 다른 대비의 2단 재표집(1절, h39 l43_region 의 일반화). 재표집 b 마다
      (1) 분할마다 채점 블록 가중을 한 번 뽑아 두 팔이 공유한다(h4_common.boot_weights, seed_of(seed, 분할)).
          common = True 이면 사용 분할 채점 블록의 합집합에서 한 번 뽑는다('2단 공통', seed_of(seed, 'U') 또는 w_seed).
      (2) 팔마다 따로 그 분할의 추출 번호를 복원 추출하고(seed_of(seed, 분할, 팔 표지)) 뽑힌 추출의 RMSE(seed 평균 뒤)를 평균한다.
    같은 번호끼리 분할 평균(nanmean)을 낸다. 팔 표지 tags 의 기본은 곡선 키 문자열이라 같은 대상의 다른 대비에서 같은 팔은 같은 추출
    재표집을 쓴다(대비 사이 가법성). h39 l43_region 은 tags = ('block', 'cell'), common = False 와 같다.
    반환 dict(delta, delta_beq, rmse_A, rmse_B, n_splits, n_blocks, n_draws_A, n_draws_B, dist, dist_beq) 또는 None."""
    tags = tuple(tags) if tags is not None else (group_tag(gA), group_tag(gB))
    if tags[0] == tags[1]:
        raise ValueError("두 팔의 표지가 같다(추출 재표집이 팔마다 따로여야 한다)")
    nboot = int(nboot)
    Wu, pos = None, None
    if common and nboot > 0 and tm.used:
        U = sorted(set().union(*[set(st.blocks.tolist()) for st in tm.used.values()]))
        pos = {b: i for i, b in enumerate(U)}
        Wu = H4.boot_weights(len(U), nboot, seed_of(seed, "U") if w_seed is None else int(w_seed))
    pts, ptb, rA, rB, dc, db, nbs, ndA, ndB = [], [], [], [], [], [], [], [], []
    for sp in sorted(tm.used):
        st = tm.used[sp]
        kA, kB = tm.idx[sp].get(gA), tm.idx[sp].get(gB)
        if not kA or not kB:
            continue
        if nboot > 0:
            Wm = Wu[:, [pos[b] for b in st.blocks.tolist()]] if common else H4.boot_weights(st.nb, nboot, seed_of(seed, sp))
        else:
            Wm = None
        res = []
        for ks, tag in ((kA, tags[0]), (kB, tags[1])):
            S, C = st.matrices(ks)
            res.append(_arm_two_stage(S, C, Wm, _draw_groups(ks), nboot, seed_of(seed, sp, tag)))
        pts.append(float(res[0][0].mean() - res[1][0].mean())); ptb.append(float(res[0][1].mean() - res[1][1].mean()))
        rA.append(float(res[0][0].mean())); rB.append(float(res[1][0].mean()))
        nbs.append(int(st.nb)); ndA.append(len(res[0][0])); ndB.append(len(res[1][0]))
        if Wm is not None:
            dc.append(res[0][2] - res[1][2]); db.append(res[0][3] - res[1][3])
    if not pts:
        return None
    out = dict(delta=float(np.mean(pts)), delta_beq=float(np.mean(ptb)), rmse_A=float(np.mean(rA)), rmse_B=float(np.mean(rB)),
               n_splits=len(pts), n_blocks=nbs, n_draws_A=int(min(ndA)), n_draws_B=int(min(ndB)), dist=None, dist_beq=None)
    if dc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.update(dist=np.nanmean(np.vstack(dc), 0), dist_beq=np.nanmean(np.vstack(db), 0))
    return out


def region_stat_two_stage(tm, gA, gB, nboot=None, seed=None, tags=None):
    """두 팔의 라벨 집합이 다른 대비의 지역 통계(h42.region_stats 와 같은 키). 주 분포 = 2단(dist), 보조 = 2단 공통(cdist),
    추출 조건부 보조 = h40.contrast(boot_delta_blocks, xdist). CI 는 has_ci 이고 채점 블록 합집합 8 이상일 때만 낸다.
    seed 기본 = seed_of('xbatch-2s', 저장소 이름)(같은 대상의 모든 2단 대비가 블록 가중을 공유한다)."""
    nb = int(tm.nboot if nboot is None else nboot) if tm.has_ci else 0
    seed = seed_of("xbatch-2s", tm.name) if seed is None else seed
    m = two_stage(tm, gA, gB, nb, seed, tags)
    if m is None or not np.isfinite(m["delta"]):
        return None
    ok = bool(tm.has_ci and tm.nb_union >= MIN_BLOCKS_CI and m.get("dist") is not None)
    c = two_stage(tm, gA, gB, nb, seed, tags, common=True) if ok else None
    x = H.contrast(tm, gA, gB, return_dist=True) if ok else None
    p0c, p0b = grp_rmse2(tm, P0_GRP)
    return dict(delta=m["delta"], delta_beq=m["delta_beq"], rmse_A=m["rmse_A"], rmse_B=m["rmse_B"], n_splits=int(m["n_splits"]),
                n_splits_expected=X.n_expected(tm), n_blocks_min=int(min(m["n_blocks"])) if m["n_blocks"] else 0, ok=ok,
                dist=m["dist"] if ok else None, dist_beq=m["dist_beq"] if ok else None,
                cdist=c.get("dist") if c else None, cdist_beq=c.get("dist_beq") if c else None,
                xdist=x.get("dist") if x else None, xdist_beq=x.get("dist_beq") if x else None,
                p0_cell=p0c, p0_beq=p0b, ci_kind="two_stage", n_draws_A=m["n_draws_A"], n_draws_B=m["n_draws_B"])


def p_two(dist, dist_beq=None) -> float:
    """양측 p = 두 가중 가운데 큰 값(h4_common.boot_p, 1절 'p 값')."""
    ps = [H4.boot_p(d) for d in (dist, dist_beq) if d is not None]
    ps = [p for p in ps if np.isfinite(p)]
    return float(max(ps)) if ps else float("nan")


def p_noninf(dist, dist_beq=None, margin=NI_MARGIN) -> float:
    """비열등 단측 p = 두 가중 가운데 큰 P(Δ^b ≥ 한계)(h54.ni_p 와 같은 정의)."""
    ps = []
    for d in (dist, dist_beq):
        if d is None:
            continue
        d = np.asarray(d, float); d = d[np.isfinite(d)]
        if len(d):
            ps.append(float(np.mean(d >= float(margin))))
    return float(max(ps)) if ps else float("nan")


def p_noninf_for_holm(p) -> float:
    """Holm 에 넣는 비열등 p = 단측 p × 2(1 에서 자른다, 1절 '유의수준')."""
    return float(min(1.0, 2.0 * float(p))) if p is not None and np.isfinite(p) else float("nan")


def noninf(hi, hib, margin=NI_MARGIN) -> bool:
    """비열등 = 두 가중 CI 상한 < 한계."""
    return bool(hi is not None and hib is not None and np.isfinite(hi) and np.isfinite(hib) and hi < margin and hib < margin)


# ================================================================ 11. 판정
def verdict4(lo, hi, lob, hib, delta=DELTA) -> str:
    """4분 판정(h42.verdict4): 우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 네 끝값 절댓값 ≤ δ, 미결정 = 그 밖."""
    return X.verdict4(lo, hi, lob, hib, delta)


def verdict4_2(lo, hi, lob, hib, dc, db) -> str:
    """가중마다 한계가 다른 4분 판정(δ_rel, h39.verdict4_2 와 같다)."""
    v = [lo, hi, lob, hib, dc, db]
    if not all(x is not None and np.isfinite(x) for x in v):
        return "판정 불가"
    if hi < 0 and hib < 0:
        return "우세"
    if lo > 0 and lob > 0:
        return "열세"
    if max(abs(lo), abs(hi)) <= dc and max(abs(lob), abs(hib)) <= db:
        return "동등"
    return "미결정"


def verdict_cols(lo, hi, lob, hib, p0_cell=float("nan"), p0_beq=float("nan")) -> dict:
    """주 δ 0.5, 보조 δ 1.0, δ_rel(0.02 × P0 RMSE, 가중별) 세 판정과 '한계 의존'(세 판정이 다르면, 1절 '4분 판정')."""
    v, v10 = verdict4(lo, hi, lob, hib, DELTA), verdict4(lo, hi, lob, hib, DELTA_AUX)
    dc = REL_COEF * p0_cell if p0_cell is not None and np.isfinite(p0_cell) else float("nan")
    db = REL_COEF * p0_beq if p0_beq is not None and np.isfinite(p0_beq) else float("nan")
    vr = verdict4_2(lo, hi, lob, hib, dc, db)
    vs = [v, v10, vr]
    dep = LIMIT_DEP_TXT if (all(x not in NA_VERDICTS for x in vs) and len(set(vs)) > 1) else ""
    return dict(verdict4=v, verdict4_d10=v10, verdict4_rel=vr, delta_rel=dc, delta_rel_beq=db, rmse_p0=p0_cell, rmse_p0_beq=p0_beq,
                limit_dependence=dep)


def holm(pvals, m=None) -> np.ndarray:
    """Holm 보정 p(가족 크기 m). 행 없음·판정 불가(None, NaN)는 p = 1 로 넣어 m 을 유지한다(1절 XC 규칙). m 이 입력 수보다 크면 모자란
    가설을 p = 1 로 채워 계산하고 입력 순서의 보정 p 만 돌려준다."""
    p = np.array([1.0 if (v is None or not np.isfinite(float(v))) else float(v) for v in pvals], float)
    k = len(p)
    m = k if m is None else int(m)
    if m < k:
        raise ValueError(f"가족 크기 m = {m} 이 가설 수 {k} 보다 작다")
    full = np.r_[p, np.ones(m - k)]
    return MS.holm(full)[:k]


def holm_table(labels, pvals, m=None) -> pd.DataFrame:
    """Holm 표: 가설, 입력 p(결측 1), 순위(1 = 가장 작은 p, 동률은 입력 순서), 보정 배수 m − 순위 + 1, Holm 보정 p.
    보정 배수는 XC-2s 의 고정 순서 우월 시험(우월 p × 그 n 의 배수)에 쓴다."""
    p = np.array([1.0 if (v is None or not np.isfinite(float(v))) else float(v) for v in pvals], float)
    k = len(p)
    m = k if m is None else int(m)
    order = np.argsort(p, kind="stable")
    rank = np.empty(k, int); rank[order] = np.arange(1, k + 1)
    adj = holm(p, m)
    return pd.DataFrame(dict(label=list(labels), p=p, rank=rank, multiplier=m - rank + 1, p_holm=adj, m=m))


def rule3(items, kind="우세") -> str:
    """다수 n 가설의 종합(1절 _rule3, h54._rule3 와 같은 순서). items = [(이름, 판정 문자열 또는 None)]. kind 는 기준 충족을 뜻하는 판정
    (우세 또는 열세, 비열등 가설은 모듈이 '우세' 로 바꿔 넣는다). 반대 판정이 있으면 기각, 충족이 없으면 판정 불가(판정할 수 없는 대비가 있을 때)
    또는 기각, 모두 충족이면 지지, 그 밖은 부분 지지."""
    items = list(items)
    other = "열세" if kind == "우세" else "우세"
    if not items or all(v is None for _, v in items):
        return "판정 불가(행 없음)"
    vs = [("행 없음" if v is None else str(v)) for _, v in items]
    hit = [str(k) for (k, _), v in zip(items, vs) if v == kind]
    bad = [str(k) for (k, _), v in zip(items, vs) if v == other]
    miss = [str(k) for (k, _), v in zip(items, vs) if v in NA_VERDICTS]
    if bad:
        return f"기각: {other}인 대비가 있다({', '.join(bad)})"
    if not hit:
        return f"판정 불가(대비 {len(items) - len(miss)}/{len(items)}; 없는 대비: {', '.join(miss)})" if miss else f"기각: {kind}인 대비가 없다"
    if len(hit) == len(items):
        return f"지지: 모든 대비가 {kind}({', '.join(hit)})"
    return f"부분 지지: {kind}인 대비 = {', '.join(hit)}" + (f"; 판정 불가 {', '.join(miss)}" if miss else "")


def branch_of(verdict) -> str:
    """판정 → 해석 문장의 갈래(우세, 열세, 동등, 미결정, 판정 불가)."""
    v = str(verdict)
    return v if v in VERDICTS4 else "판정 불가"


def compose_sentence(templates, verdict, holm_p=None, delta=None, worse_regions=(), mde=None, reason="") -> str:
    """해석 문장(1절 '해석 문장의 다섯 갈래'). templates = 갈래 → 사전 고정 문장. 우세·열세이면 Holm 보정 p ≥ 0.05 일 때 '보정 전 유의',
    |Δ| < 0.5 cm 이면 '통계적으로 구별되나 크기는 0.5 cm 미만', 미결정이면 최소 검출 효과, 판정 불가이면 사유를 괄호로 붙인다.
    worse_regions = [(지역, Δ, CI 하한, CI 상한)] 이면 풀 문장 뒤에 '지역 X 에서는 오차가 컸다'를 덧붙인다."""
    b = branch_of(verdict)
    base = str(templates.get(b, "")).rstrip().rstrip(".")
    notes = []
    if b in ("우세", "열세"):
        if holm_p is not None and np.isfinite(holm_p) and float(holm_p) >= HOLM_ALPHA:
            notes.append(UNCORRECTED_TXT)
        if delta is not None and np.isfinite(delta) and abs(float(delta)) < SMALL_EFFECT_CM:
            notes.append(SMALL_EFFECT_TXT)
    if b == "미결정" and mde is not None and np.isfinite(mde):
        notes.append(f"최소 검출 효과 약 {float(mde):.1f} cm")
    if b == "판정 불가" and reason:
        notes.append(str(reason))
    out = base + (f"({', '.join(notes)})" if notes else "") + "."
    for nm, d, lo, hi in worse_regions:
        out += f" 지역 {nm} 에서는 오차가 컸다(Δ {float(d):+.2f} cm, CI [{float(lo):.2f}, {float(hi):.2f}])."
    return out


# ================================================================ 12. 풀
_POOL_A = SimpleNamespace(delta_eq=DELTA, delta_eq_aux=DELTA_AUX)


def pool_label(k, m) -> str:
    """풀 표지: 지역 k/m, 부분(지역 k/m)(풀 지역이 등록 수보다 적다), 판정 불가(2 미만)."""
    if int(k) < MIN_POOL_REGIONS:
        return "판정 불가"
    return f"지역 {k}/{m}" if int(k) >= int(m) else f"부분(지역 {k}/{m})"


def _augment(row, s):
    """h42.stats_row 의 행에 1절의 보조 열을 더한다: 양측 p, 비열등 p, 비열등(δ 0.5·1.0, 공통 CI), δ_rel 판정과 '한계 의존',
    '분할 독립 가정 의존', 추출 조건부 판정과 '추출 변동 의존', 열세 여부, 효과 크기 표기."""
    lo, hi, lob, hib = (row.get(k) for k in ("ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq"))
    row["p_two"] = p_two(s.get("dist"), s.get("dist_beq"))
    row["p_ni"] = p_noninf(s.get("dist"), s.get("dist_beq"), NI_MARGIN)
    row["p_ni_x2"] = p_noninf_for_holm(row["p_ni"])
    row["noninf"] = noninf(hi, hib, NI_MARGIN)
    row["noninf_d10"] = noninf(hi, hib, DELTA_AUX)
    row["noninf_common"] = noninf(row.get("ci_hi_c"), row.get("ci_hi_beq_c"), NI_MARGIN)
    row["superior_common"] = noninf(row.get("ci_hi_c"), row.get("ci_hi_beq_c"), 0.0)
    row["worse"] = bool(lo is not None and lob is not None and np.isfinite(lo) and np.isfinite(lob) and lo > 0 and lob > 0)
    vc = verdict_cols(lo, hi, lob, hib, s.get("p0_cell", float("nan")), s.get("p0_beq", float("nan")))
    row.update({k: v for k, v in vc.items() if k not in ("verdict4", "verdict4_d10")})
    v, vcm = row.get("verdict4"), row.get("verdict4_common")
    row["ci_dependence"] = CI_DEP_TXT if (v not in NA_VERDICTS and vcm not in NA_VERDICTS and v is not None and vcm is not None and v != vcm) else ""
    row["ci_kind"] = s.get("ci_kind", "")
    if s.get("xdist") is not None:
        xlo, xhi = X._ci(s.get("xdist")); xlob, xhib = X._ci(s.get("xdist_beq"))
        vx = verdict4(xlo, xhi, xlob, xhib, DELTA)
        row.update(ci_lo_x=xlo, ci_hi_x=xhi, ci_lo_beq_x=xlob, ci_hi_beq_x=xhib, verdict4_draw_cond=vx,
                   draw_dependence=DRAW_DEP_TXT if (v not in NA_VERDICTS and vx not in NA_VERDICTS and v != vx) else "")
    else:
        row.update(verdict4_draw_cond="", draw_dependence="")
    d = row.get("delta")
    row["small_note"] = SMALL_EFFECT_TXT if (v in ("우세", "열세") and d is not None and np.isfinite(d) and abs(d) < SMALL_EFFECT_CM) else ""
    if row.get("undetermined"):
        row.update(verdict4_rel="판정 불가", limit_dependence="", noninf=False, noninf_d10=False, noninf_common=False, superior_common=False,
                   worse=False, ci_dependence="", draw_dependence="", small_note="")
    return row


def region_inference_rows(per, names) -> dict:
    """지역 수준 추론(h42.region_inference: 부호 검정, 부호 뒤집기, Hartung–Knapp). 지역 Δ = 점 추정, 분산 = 주 분포(셀 가중)의 분산.
    region_general = HK CI 가 0 을 제외한다('지역 일반' 문장의 조건, WRAPUP 7.2 (a)2)."""
    used = [nm for nm in names if nm in per and per[nm].get("dist") is not None]
    if len(used) < 2:
        return dict(ri_k=len(used), region_general=False)
    th = np.array([per[nm]["delta"] for nm in used], float)
    vv = np.array([float(np.nanvar(per[nm]["dist"], ddof=1)) if np.isfinite(per[nm]["dist"]).sum() > 2 else np.nan for nm in used], float)
    r = X.region_inference(th, vv)
    out = {f"ri_{k}": v for k, v in r.items()}
    out["ri_regions"] = ",".join(used)
    lo, hi = r.get("hk_lo", np.nan), r.get("hk_hi", np.nan)
    out["region_general"] = bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0))
    return out


def pool(per, names, label="", registered=None) -> list:
    """지역별 통계 dict(region_stat_same 또는 region_stat_two_stage)의 지역 행과 층화 평균 행(h42.pool_rows: 풀 지역 2개 이상, 각 지역 채점
    블록 합집합 8 이상만 CI). 각 행에 _augment 의 보조 열을, 평균 행에 풀 표지(registered = 등록 지역 수), 소수 블록 지역, 지역 수준
    추론(HK)을 더한다. 평균 행의 분포는 '_dist', '_dist_beq', '_cdist', '_cdist_beq' 키에 둔다(표에 쓰지 않는다)."""
    rows = X.pool_rows(per, names, _POOL_A, label)
    if not rows:
        return []
    have = [nm for nm in names if nm in per]
    for r in rows[:-1]:
        s = per[r["target"]]
        _augment(r, s)
        r["few_blocks"] = FEW_BLOCKS_TXT if int(r.get("n_blocks_split_min", 0)) < FEW_BLOCKS else ""
        r["label"] = label
    mr = rows[-1]
    pl = [nm for nm in have if per[nm].get("dist") is not None]
    use = pl if pl else have

    def st(key):
        ds = [per[nm].get(key) for nm in pl if per[nm].get(key) is not None]
        return MS.strat(ds) if ds else None
    sm = dict(dist=st("dist"), dist_beq=st("dist_beq"), cdist=st("cdist"), cdist_beq=st("cdist_beq"),
              xdist=st("xdist") if all(per[nm].get("xdist") is not None for nm in pl) and pl else None,
              xdist_beq=st("xdist_beq") if all(per[nm].get("xdist_beq") is not None for nm in pl) and pl else None,
              p0_cell=float(np.nanmean([per[nm].get("p0_cell", np.nan) for nm in use])) if use else np.nan,
              p0_beq=float(np.nanmean([per[nm].get("p0_beq", np.nan) for nm in use])) if use else np.nan,
              ci_kind=per[use[0]].get("ci_kind", "") if use else "")
    _augment(mr, sm)
    m = int(registered) if registered is not None else len(names)
    mr.update(label=label, pool=pool_label(len(pl), m), n_regions_registered=m,
              few_block_regions=",".join(nm for nm in have if int(per[nm].get("n_blocks_min", 0)) < FEW_BLOCKS),
              worse_regions=";".join(f"{r['target']}" for r in rows[:-1] if r.get("worse")))
    mr.update(region_inference_rows(per, pl))
    mr["_dist"], mr["_dist_beq"], mr["_cdist"], mr["_cdist_beq"] = sm["dist"], sm["dist_beq"], sm["cdist"], sm["cdist_beq"]
    return rows


def contrast_pool(tms, names, gA, gB, label="", kind="same", registered=None, nboot=None):
    """풀 하나의 대비. kind = same(같은 라벨 집합) 또는 two_stage(라벨 집합이 다른 두 팔). 반환 (행 목록, 지역 통계 dict)."""
    per = {}
    for nm in names:
        tm = tms.get(nm)
        if tm is None:
            continue
        s = region_stat_same(tm, gA, gB) if kind == "same" else region_stat_two_stage(tm, gA, gB, nboot=nboot)
        if s is not None:
            per[nm] = s
    boot_cache_clear()
    return pool(per, names, label, registered), per


def worse_regions(rows) -> list:
    """지역 행 가운데 열세(두 가중 CI 하한 > 0)인 지역 [(지역, Δ, CI 하한, CI 상한)](풀 문장 뒤 덧붙임 규칙)."""
    return [(r["target"], r["delta"], r["ci_lo"], r["ci_hi"]) for r in rows if r.get("scope") == "region" and r.get("worse")]


def criterion_met(row, kind="superior", margin=NI_MARGIN) -> bool:
    """XC 주 가설의 CI 기준(1절 보조 CI 행, 2.3절): 2단 CI 와 2단 공통 CI 의 상한이 두 가중 모두 0(우월) 또는 한계(비열등) 미만.
    Holm 조건은 모듈이 따로 본다."""
    if row is None or row.get("undetermined") or row.get("verdict4") in NA_VERDICTS:
        return False
    lim = 0.0 if kind == "superior" else float(margin)
    return noninf(row.get("ci_hi"), row.get("ci_hi_beq"), lim) and noninf(row.get("ci_hi_c"), row.get("ci_hi_beq_c"), lim)


def clean_rows(rows) -> pd.DataFrame:
    """'_' 로 시작하는 분포 키를 뺀 표."""
    return pd.DataFrame([{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows])


# ================================================================ 13. 조각
SHARD_PART = "cpu"
CFG_UNIT_KEYS = ("variant", "data_sha")                                     # 조각마다 달라도 되는 항목(h54 와 같다)


def _name_part(v, what):
    v = str(v)
    if v == "" or "__" in v or "/" in v or os.sep in v or v.strip() != v:
        raise ValueError(f"조각 이름의 {what} '{v}' 에 '__', '/', 공백을 쓸 수 없다")
    return v


def shard_base(shards_dir, tag, target, mode, split, variant="") -> Path:
    """조각 이름 <tag>__cpu__<대상>__<모드>__s<분할>[__<변형>](1절 '산출 경로')."""
    parts = [_name_part(tag, "tag"), SHARD_PART, _name_part(target, "대상"), _name_part(mode, "모드"), f"s{int(split)}"]
    if variant:
        parts.append(_name_part(variant, "변형"))
    return Path(shards_dir) / "__".join(parts)


def shard_paths(shards_dir, tag, target, mode, split, variant="") -> dict:
    b = str(shard_base(shards_dir, tag, target, mode, split, variant))
    return dict(runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), unit=Path(b + "_unit.json"))


def cfg_hash(cfg, common=False) -> str:
    """설정 해시(sha1 앞 12자, h54.wf_cfg_hash 와 같은 규칙). common 이면 variant·data_sha 를 뺀다."""
    d = {k: v for k, v in cfg.items() if not (common and k in CFG_UNIT_KEYS)}
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def make_unit_cfg(exp, variant="", data_sha="", **kw) -> dict:
    """결과에 영향을 주는 설정 요약(대상·분할·tag·워커는 넣지 않는다). 공통 고정값과 계획·동결 모듈 표지를 넣고 실험 고유 항목은 kw 로 받는다."""
    d = dict(exp=str(exp), variant=str(variant), data_sha=str(data_sha), kappa=KAPPA, r=R_PS, lams=list(LAMS), lam_base=LAM_BASE, cv_k=CV_K,
             cb_hi=dict(X.CB_HI), plan=f"{PLAN_DOC}@{PLAN_COMMIT}", frozen=dict(FROZEN_SHA16))
    d.update(kw)
    return d


def code_sha(path) -> str:
    """코드 파일의 sha1 앞 12자(h54.file_sha 와 같은 형식)."""
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return "none"


def unit_state(shards_dir, tag, target, mode, split, cfg, variant="") -> tuple:
    """(완료 여부, 사유). 완료 = 세 파일이 있고, 설정 해시가 같고, status 가 failed 가 아니다(--resume 규칙)."""
    p = shard_paths(shards_dir, tag, target, mode, split, variant)
    if not (p["unit"].exists() and p["runs"].exists() and p["npz"].exists()):
        return False, "조각 없음"
    try:
        u = json.loads(p["unit"].read_text())
    except (OSError, ValueError):
        return False, "unit.json 을 읽을 수 없음"
    if u.get("cfg_hash") != cfg_hash(cfg):
        return False, "설정 불일치(cfg_hash)"
    if u.get("status") == "failed":
        return False, "이전 실행 실패"
    return True, str(u.get("status", "ok"))


def _atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_shard(shards_dir, tag, target, mode, split, rows, stores, cfg, unit=None, variant="", expected=None, code_file=None,
                allowed=None) -> dict:
    """조각 쓰기(h54.write_shard 와 같은 형식). runs.csv(원자적) → blocksse.npz(h4_common.save_stores) → unit.json(마지막, 원자적, 완료 표지).
    unit 에는 split, dup_of, valid, expected_splits, store_names 를 넣어 h54.make_tm·units_for 가 읽게 한다. 산출 경로는 check_out_dir 로 확인한다."""
    check_out_dir(shards_dir, allowed)
    p = shard_paths(shards_dir, tag, target, mode, split, variant)
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    p["unit"].unlink(missing_ok=True)                                     # 이전 세대의 완료 표지를 먼저 지운다
    tmp = p["runs"].with_name(p["runs"].name + f".tmp{os.getpid()}")
    pd.DataFrame(list(rows)).to_csv(tmp, index=False)
    os.replace(tmp, p["runs"])
    stores = list(stores) if isinstance(stores, (list, tuple)) else [stores]
    save_stores(stores, p["npz"])
    u = dict(unit or {})
    u.setdefault("status", "ok")
    u.setdefault("dup_of", -1)
    u.setdefault("valid", True)
    u.update(tag=str(tag), part=SHARD_PART, target=str(target), mode=str(mode), split=int(split), variant=str(variant),
             expected_splits=[int(v) for v in (expected if expected is not None else [split])], store_names=[st.target for st in stores],
             n_keys=int(sum(len(st) for st in stores)), cfg=cfg, cfg_hash=cfg_hash(cfg), cfg_common=cfg_hash(cfg, common=True),
             code_sha=code_sha(code_file) if code_file else "none", code_sha_core=code_sha(__file__), frozen=dict(FROZEN_SHA16),
             plan_commit=PLAN_COMMIT, threads=os.environ.get("OMP_NUM_THREADS", ""), device=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
             max_rss_mb=max_rss_mb(), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    _atomic_write(p["unit"], json.dumps(u, ensure_ascii=False, indent=1, default=float))
    return u


def find_shards(shards_dir, tag) -> list:
    """<tag>__* 조각 목록(세 파일이 모두 있는 것). 이름 마디 5(변형 없음) 또는 6."""
    d = Path(shards_dir)
    out = []
    if not d.exists():
        return out
    for p in sorted(d.glob(f"{tag}__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")
        if len(parts) not in (5, 6) or parts[0] != tag or parts[1] != SHARD_PART:
            continue
        b = str(p)[:-len("_unit.json")]
        if Path(b + "_runs.csv").exists() and Path(b + "_blocksse.npz").exists():
            out.append(dict(unit=p, runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), tag=tag, target=parts[2], mode=parts[3],
                            split=int(parts[4][1:]), variant=parts[5] if len(parts) == 6 else ""))
    return out


def read_runs(shards) -> pd.DataFrame:
    """조각 runs.csv 결합(문자열 열은 문자열로, 빈 칸은 빈 문자열). h54.read_runs 와 같은 형식에 없는 열을 허용한다."""
    frames = []
    str_cols = dict(alpha=str, alpha_sel=str, fit_flag=str, variant=str, cv_flag=str, placement=str, target=str, mode=str, method=str, learner=str)
    for s_ in shards:
        if Path(s_["runs"]).stat().st_size > 1:
            f = pd.read_csv(s_["runs"], dtype=str_cols, keep_default_na=False, na_values=["", "nan", "NaN"])
            if len(f):
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    for c_ in ("alpha_sel", "fit_flag", "variant", "cv_flag"):
        if c_ in runs:
            runs[c_] = runs[c_].fillna("").astype(str)
    return runs


def check_cfg(units, allow_mixed=False) -> dict:
    """조각 사이 공통 설정 해시가 하나인지 확인한다(다르면 중단, allow_mixed 이면 경고만)."""
    hs = Counter(str(u.get("cfg_common", "legacy")) for u in units)
    cs = Counter(str(u.get("code_sha", "legacy")) for u in units)
    if len(hs) > 1:
        msg = f"조각의 공통 설정 해시가 {len(hs)}종이다 {dict(hs)}"
        if not allow_mixed:
            raise SystemExit("[집계] " + msg)
        print("  [warn] " + msg, flush=True)
    return dict(cfg_common=dict(hs), code_sha=dict(cs), n=len(units))


def load_tms(shards_dir, tag, nboot=NBOOT, point_only=None, allow_mixed=False):
    """조각 → {저장소 이름: h40.TMx}(h54.summarize_round 와 같은 경로: load_stores, h54.make_tm, h54.units_for). 반환 (tms, units, runs).
    point_only = 저장소 이름 → bool(기본 is_point_only_name)."""
    sh = find_shards(shards_dir, tag)
    if not sh:
        return {}, [], pd.DataFrame()
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    check_cfg(units, allow_mixed)
    runs = read_runs(sh)
    stores = load_stores([s_["npz"] for s_ in sh])
    by = {}
    for (nm, sp), st in stores.items():
        by.setdefault(nm, {})[int(sp)] = st
    pof = point_only or is_point_only_name
    tms = {nm: W.make_tm(nm, bs, W.units_for(units, nm), int(nboot), pof(nm)) for nm, bs in sorted(by.items())}
    return tms, units, runs


# ================================================================ 14. 봉인(계획 0.3 열람 순서 6)
SEALED_README = """봉인 폴더(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 0.3 열람 순서)

이 폴더의 표는 열람 순서보다 먼저 만들어진 집계 표다(R1a·R1b 집계, XD-alg 의 비알래스카 행 등). 0.3 의 열람 순서에 정한 시점 전에는
열지 않는다. 봉인을 푼 시각은 계획 8절에 적는다. 쓰기는 xbatch_core.write_sealed 만 하며 화면에는 파일 이름, 행 수, sha256 앞 16자만 남긴다.
sealed_manifest.json 에는 파일별 행 수, sha256, 기록 시각만 있다(표의 내용은 없다).
"""


def sealed_dir(exp_name, root=None) -> Path:
    """data/processed/xbatch/<새 이름>/sealed."""
    return Path(root or XBATCH_ROOT) / str(exp_name) / "sealed"


def write_sealed(exp_name, tables, root=None, allowed=None, quiet=False) -> list:
    """봉인 폴더에 표를 쓴다. tables = 파일 이름 → DataFrame(CSV) 또는 dict·list(JSON). 화면에는 행 수와 sha256 앞 16자만 쓴다.
    반환 [dict(file, rows, sha256)]. 봉인 폴더 밖이나 허용 경로 밖이면 SystemExit."""
    d = sealed_dir(exp_name, root)
    check_out_dir(d, allowed)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "README.txt").exists():
        _atomic_write(d / "README.txt", SEALED_README)
    man_p = d / "sealed_manifest.json"
    try:
        man = json.loads(man_p.read_text()) if man_p.exists() else dict(entries=[])
    except (OSError, ValueError):
        man = dict(entries=[])
    out = []
    for fname, obj in tables.items():
        fname = str(fname)
        if "/" in fname or os.sep in fname or fname.startswith(".") or fname in ("README.txt", "sealed_manifest.json"):
            raise ValueError(f"봉인 파일 이름 {fname} 을 쓸 수 없다")
        p = d / fname
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(tmp, index=False)
            nrow = int(len(obj))
        else:
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))
            nrow = int(len(obj)) if hasattr(obj, "__len__") else 1
        os.replace(tmp, p)
        sha = sha256_file(p)
        rec = dict(file=fname, rows=nrow, sha256=sha, written=time.strftime("%Y-%m-%d %H:%M:%S"))
        man["entries"] = [e for e in man.get("entries", []) if e.get("file") != fname] + [rec]
        out.append(rec)
        if not quiet:
            print(f"[봉인] {exp_name}/sealed/{fname}: 행 {nrow}, sha256 {sha[:16]}", flush=True)
    _atomic_write(man_p, json.dumps(man, ensure_ascii=False, indent=1))
    return out


def assert_not_sealed(path):
    """읽기 보호: 봉인 폴더 안의 파일을 읽으려 하면 PermissionError(열람 순서 전 열람 방지)."""
    if "sealed" in Path(path).resolve().parts:
        raise PermissionError(f"{path} 는 봉인 폴더 안이다. 계획 0.3 의 열람 순서에 따라 연다")
    return Path(path)


# ================================================================ 15. 산출 경로 보호
def out_dir(exp_id) -> Path:
    """data/processed/xbatch/<새 이름>/(1절 '산출 경로'). exp_id = XA–XJ."""
    return XBATCH_ROOT / EXP_NAMES[str(exp_id).upper()[:2]]


def _real(p) -> Path:
    p = Path(p)
    return Path(os.path.realpath(str(p if p.is_absolute() else ROOT / p)))


def check_out_dir(path, allowed=None) -> Path:
    """쓰기 보호: 경로가 허용 뿌리(기본 data/processed/xbatch, 환경 변수 XBATCH_OUT_ROOTS 의 추가 뿌리) 안이어야 한다. 기호 연결을 푼 실제 경로로
    비교한다. 기존 표 폴더(data/processed 의 다른 폴더, results/)에는 쓰지 않는다."""
    p = _real(path)
    roots = list(allowed) if allowed is not None else [XBATCH_ROOT] + [Path(v) for v in os.environ.get("XBATCH_OUT_ROOTS", "").split(os.pathsep) if v]
    rr = [_real(r) for r in roots]
    if not any(p == r or r in p.parents for r in rr):
        raise SystemExit(f"[거부] 출력 경로 {p} 는 허용 뿌리 {', '.join(str(r) for r in rr)} 밖이다(계획 1절 '산출 경로')")
    return p


# ================================================================ 16. 누설 시험 도구(1절 '누설 시험')
def perturb_labels(c, keep_A, seed=0, lo=1.0, hi=500.0):
    """문맥 사본에서 선택 밖 A 라벨(keep_A 에 없는 색인)과 모든 B 라벨을 무작위 값으로 바꾼다. 원천 행(y_src)은 그대로다.
    h40.Ctx·h54.RCtx 모두 받는다(zA 가 있으면 다시 계산한다)."""
    c2 = copy.copy(c)
    rng = np.random.RandomState(int(seed))
    yA = np.array(c.yA, float, copy=True)
    m = ~np.isin(np.arange(len(yA)), np.asarray(keep_A, int))
    yA[m] = rng.uniform(lo, hi, int(m.sum()))
    c2.yA = yA
    c2.yB = rng.uniform(lo, hi, len(c.yB))
    if hasattr(c, "zA"):
        with np.errstate(invalid="ignore", divide="ignore"):
            c2.zA = np.log(yA) - np.log(np.asarray(c.sA, float))
    return c2


@contextlib.contextmanager
def h54_trace():
    """h54 단위의 예측(PRED_TRACE)과 라벨 사용 기록(WF_TRACE)을 잡는다. 반환 (예측 dict, 기록 list)."""
    old = (W.PRED_TRACE, W.WF_TRACE)
    preds, trace = {}, []
    W.PRED_TRACE, W.WF_TRACE = preds, trace
    try:
        yield preds, trace
    finally:
        W.PRED_TRACE, W.WF_TRACE = old


def selected_union(trace, kinds=("coef", "select", "fit", "cv")) -> np.ndarray:
    """기록에서 선택 라벨(A 색인)의 합집합."""
    s = set()
    for e in trace:
        if e.get("kind") in kinds:
            s.update(int(v) for v in np.asarray(e["idx"]).ravel())
    return np.array(sorted(s), int)


def compare_predictions(p1, p2, atol=0.0) -> dict:
    """두 예측 기록의 비교(키 집합, 최대 절대 차)."""
    same = set(p1) == set(p2)
    md = 0.0
    for k in set(p1) & set(p2):
        a, b = np.asarray(p1[k], float), np.asarray(p2[k], float)
        if a.shape != b.shape:
            md = float("inf"); break
        if len(a):
            md = max(md, float(np.nanmax(np.abs(a - b))))
    return dict(n_keys=len(p1), same_keys=bool(same), max_abs_diff=md, ok=bool(same and md <= atol and len(p1) > 0))


def leakage_invariance(run_fn, c, keep_A=None, seed=0, atol=0.0) -> dict:
    """누설 시험 틀: run_fn(c) 를 h54_trace 안에서 돌려 예측을 모으고, 선택 라벨 합집합(keep_A 가 없으면 기록에서)을 남긴 채 선택 밖 A 라벨과
    B 라벨을 바꾼 문맥에서 다시 돌려 모든 예측이 같은지 본다. run_fn 은 h54 의 UnitBase.add 경로로 예측을 저장해야 한다(PRED_TRACE)."""
    with h54_trace() as (p1, tr):
        run_fn(c)
    keep = selected_union(tr) if keep_A is None else np.asarray(keep_A, int)
    c2 = perturb_labels(c, keep, seed)
    with h54_trace() as (p2, _):
        run_fn(c2)
    out = compare_predictions(p1, p2, atol)
    out.update(n_keep=int(len(keep)), n_A=int(len(c.yA)))
    return out


# ================================================================ 17. XI warm_trim(계획 2.9)
def trim_ctx_warm(c, s_W_min):
    """warm_trim: A 에서 s ≥ min(s_W) 인 셀을 뺀 RCtx 사본(NaN s 는 남긴다). 채점 셀(W·I, c.yB 와 c.maskW)은 그대로다.
    반환 (사본, 남긴 A 마스크). meta 의 n_A, nb_A, s_A_*, frac_W_outside_A 를 다시 계산하고 n_trim 을 적는다."""
    sA = np.asarray(c.sA, float)
    keep = ~(sA >= float(s_W_min))
    c2 = copy.copy(c)
    for attr in ("XA", "yA", "sA", "blkA", "latA", "lonA", "PA"):
        if getattr(c, attr, None) is not None:
            setattr(c2, attr, np.asarray(getattr(c, attr))[keep])
    if getattr(c, "XA34", None) is not None:
        c2.XA34 = np.asarray(c.XA34)[keep]
    sA2 = sA[keep]
    sW = np.asarray(c.sB, float)[np.asarray(c.maskW, bool)] if getattr(c, "maskW", None) is not None else np.zeros(0)
    c2.meta = dict(c.meta)
    c2.meta.update(n_A=int(keep.sum()), nb_A=int(len(np.unique(np.asarray(c.blkA)[keep]))), n_trim=int((~keep).sum()), s_W_min_all=float(s_W_min),
                   s_A_min=float(np.nanmin(sA2)) if len(sA2) else np.nan, s_A_max=float(np.nanmax(sA2)) if len(sA2) else np.nan,
                   s_A_mean=float(np.nanmean(sA2)) if len(sA2) else np.nan,
                   frac_W_outside_A=float(np.mean(sW > np.nanmax(sA2))) if (len(sW) and len(sA2)) else np.nan,
                   trim_rule="warm_trim: A 에서 s ≥ min(s_W) 셀 제외(s_W = W 블록의 모든 셀)")
    return c2, keep


def build_rctx10_trim(a, target, split, variant="warm"):
    """XI 의 warm_trim 문맥: h54.build_rctx10(같은 W·I·A 분할)에서 A 만 줄인다. s_W = W 블록의 모든 셀의 √TDD(h54.wf10_split)."""
    if variant != "warm":
        raise ValueError("warm_trim 은 warm 변형에만 등록되었다(계획 2.9)")
    c = W.build_rctx10(a, target, int(split), variant)
    _, D = W.get_data(a)
    q = W.wf10_split(D, target, variant, int(split))
    sW_all = D.df.s.values[q["W"]].astype(float)
    c2, _ = trim_ctx_warm(c, float(np.nanmin(sW_all)))
    c2.variant = "warm_trim"
    return c2


class R10TrimUnit(W.R10Unit):
    """XI warm_trim 단위(h54.R10Unit 하위 클래스, 방법·격자·추출 규칙은 R10Unit 과 같다). 저장소 이름은 <대상>~warm_trim|r 와
    <대상>~warm_trimW|r, <대상>~warm_trimI|r 다. 추출 seed 의 모드 표지는 R10Unit 과 같은 'r10w' 이고 |A| 가 줄어 추출 자체는 다르다."""

    VARIANT = "warm_trim"

    def __init__(self, a, c, exp="wf10", variant="warm_trim", dry=False):
        if variant != self.VARIANT:
            raise ValueError(f"R10TrimUnit 의 변형은 {self.VARIANT} 하나다: {variant}")
        self._init_base(a, c, exp, f"{c.target}~{variant}|{c.mode}", variant, dry)
        self._km: dict = {}
        self._std = None
        self._di = None
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)


# ================================================================ 18. 묶음(4절)
PAYLOAD_INPUTS = ("data/processed/fidelity_base_v3.csv", "data/processed/e5_soil_tdd_v3.csv", "data/processed/e5_soil_tdd_v4.csv",
                  "data/processed/lgx_tdd_matched_v1.csv", "data/processed/covariates_ext_v1.csv", "data/processed/covariates_ext_v1_meta.json",
                  "data/processed/map_lena/lena_grid_x25_v1.csv.gz", "data/processed/lg_subregion_map_v1.csv",
                  # 4절 'v4 새 셀 tdd_matched 표'(1c 산출, XB 의 P* 와 XC 의 W+ 가 읽는다. 없으면 XB 가 중단한다)
                  "data/processed/xbatch/XB_multisource_stacking/inputs/xb_tdd_matched_v4.csv",
                  "data/processed/xbatch/XB_multisource_stacking/inputs/xb_tdd_matched_v4_meta.json",
                  # XJ 대상 쪽(2.10): 지온 유도 보조 행(F3_ext_temp)은 v4 표에만 있고 약관 열은 labels 표에 있다
                  "data/processed/fidelity_base_v4.csv", "data/processed/fidelity_base_v4_labels.csv") + tuple(
    f"data/processed/lgd/run_tables/{s}/fidelity_base_v3.csv" for s in RUN_TABLE_DIRS) + tuple(
    # 실행 표의 토양 표(e5_soil_tdd_v4.csv 로 가는 기호 연결). XI 의 주 판(Canada_expanded_lic)과 XJ 대상 쪽(Tibet)이 읽는다. 묶음에서 연결을 보존하거나 파일로 푼다
    f"data/processed/lgd/run_tables/{s}/e5_soil_tdd_v3.csv" for s in RUN_TABLE_DIRS)
PAYLOAD_OPTIONAL = ("data/processed/xe/xe_feat_v1.csv",                     # 1c·2단계 산출 뒤 생긴다(R1b·R3)
                    "data/processed/xe/xe_feat_v1_meta.json",               # XE: 90 % 규칙 결과와 군 열 해시(FeatTable 이 요구한다)
                    "data/processed/xe/xe_feat_v1_final.json",              # XE: 최종 해시(R3 의 xh·SI 변형)
                    "data/processed/xbatch/XE_hires_covariates/inputs/xe_vwc_v1.csv",        # XE-e 셀별 VWC 표(--xe-e-prep)
                    "data/processed/xbatch/XE_hires_covariates/inputs/xe_vwc_v1_meta.json",
                    PLAN_DOC,                                               # XE 최종 해시의 개정 이력 대조(git 이 없는 노드)
                    "data/processed/xbatch/XJ_tempderived_aux_labels/inputs/xj_aux_lic_token.json",   # XJ 원천 쪽 약관 표지(근거를 기록한 경우)
                    # XG 작업 R3 의 입력(로컬 1c·2단계 산출: 제품 값 표, 채점 셀 거리 표, 추출·마스크 기록). 모듈의 PAYLOAD_EXTRA 와 같다
                    "data/processed/xbatch/XG_product_comparison/xg_product_values_v1.csv",
                    "data/processed/xbatch/XG_product_comparison/xg_leak_cells_v1.csv",
                    "data/processed/xbatch/XG_product_comparison/xg_meta.json",
                    # XH 작업 R1b 의 입력(로컬 1c 산출: kNNDM 색인, 색인 메타(sha256), 예측 영역). 모듈의 PAYLOAD_EXTRA 와 같다
                    "data/processed/xbatch/XH_validation_ladder/xh_knndm_index_v1.csv",
                    "data/processed/xbatch/XH_validation_ladder/xh_knndm_meta_v1.json",
                    "data/processed/xbatch/XH_validation_ladder/xh_pred_domain_v1.csv.gz")
# 선택 파일 가운데 앞 파일이 있으면 함께 있어야 하는 파일(없으면 묶음 정보의 missing 에 적는다)
PAYLOAD_REQUIRES = {"data/processed/xe/xe_feat_v1.csv": ("data/processed/xe/xe_feat_v1_meta.json",),
                    "data/processed/xbatch/XE_hires_covariates/inputs/xe_vwc_v1.csv": (
                        "data/processed/xbatch/XE_hires_covariates/inputs/xe_vwc_v1_meta.json",),
                    "data/processed/xbatch/XG_product_comparison/xg_product_values_v1.csv": (
                        "data/processed/xbatch/XG_product_comparison/xg_leak_cells_v1.csv",
                        "data/processed/xbatch/XG_product_comparison/xg_meta.json"),
                    "data/processed/xbatch/XH_validation_ladder/xh_knndm_index_v1.csv": (
                        "data/processed/xbatch/XH_validation_ladder/xh_knndm_meta_v1.json",
                        "data/processed/xbatch/XH_validation_ladder/xh_pred_domain_v1.csv.gz")}
PAYLOAD_CODE_GLOBS = ("scripts/3_deep_learning/xbatch_core.py", "scripts/3_deep_learning/x_*.py", "scripts/2_evaluation/xa_*.py",
                      "scripts/2_evaluation/x_*.py", "src/polar/*.py", "tests/test_xbatch_core.py", "tests/test_x_*.py", "tests/test_xa_*.py",
                      "scripts/rescale/xbatch_constraints.txt") + tuple(v[1] for v in FROZEN.values())


def read_constraints(path=CONSTRAINTS) -> dict:
    """제약 파일 → {패키지: 판}('이름==판' 줄만, '#' 주석 무시)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if "==" in s:
            k, v = s.split("==", 1)
            out[k.strip().lower()] = v.strip()
    return out


def deps_versions(names=("catboost", "scikit-learn", "pandas", "scipy", "numpy", "torch")) -> dict:
    """설치 판(importlib.metadata). 없으면 '없음'."""
    import importlib.metadata as md
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "없음"
    return out


def check_constraints(path=CONSTRAINTS) -> list:
    """제약 파일과 설치 판의 차이 [(패키지, 고정 판, 설치 판)]. 로컬(scikit-learn 1.3.0 등)에서는 차이가 정상이고 Rescale 설치 뒤에는 비어야 한다."""
    want = read_constraints(path)
    have = deps_versions(tuple(want))
    return [(k, v, have.get(k, "없음")) for k, v in want.items() if have.get(k) != v]


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                                     # noqa: BLE001
        return ""


def payload_manifest(extra=(), out=None, allowed=None) -> dict:
    """묶음 정보(4절): 코드·시험·제약 파일·입력 표·실행 표의 sha256, 없는 파일, git 커밋과 미커밋 경로, 동결 표지, 약관 점검 결과.
    out 이 있으면 JSON 으로 쓴다(허용 경로: data/processed/xbatch, work)."""
    files = []
    for g in PAYLOAD_CODE_GLOBS:
        files += sorted(str(p.relative_to(ROOT)) for p in ROOT.glob(g) if p.is_file())
    files += list(PAYLOAD_INPUTS) + [p for p in PAYLOAD_OPTIONAL if (ROOT / p).exists()] + [str(v) for v in extra]
    files += [r for k, rs in PAYLOAD_REQUIRES.items() if (ROOT / k).exists() for r in rs]
    files = list(dict.fromkeys(files))
    sha, missing = {}, []
    for f in files:
        p = ROOT / f
        if p.exists():
            sha[f] = sha256_file(p)
        else:
            missing.append(f)
    unc = [ln[3:] for ln in _git("status", "--porcelain", "--", *files).splitlines() if ln.strip()] if files else []
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), plan=PLAN_DOC, plan_commit=PLAN_COMMIT, git_commit=_git("rev-parse", "--short", "HEAD"),
               git_uncommitted_paths=unc, n_files=len(sha), sha256=sha, missing=missing, frozen=dict(FROZEN_SHA16),
               constraints=read_constraints() if CONSTRAINTS.exists() else {}, lic=lic_check(strict=False))
    if out is not None:
        o = Path(out)
        check_out_dir(o.parent, allowed if allowed is not None else [XBATCH_ROOT, ROOT / "work"])
        o.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(o, json.dumps(man, ensure_ascii=False, indent=1))
    return man


# ================================================================ 19. 프로세스 풀과 GPU 결정성
def core_blocks(workers, threads, permit=None):
    """워커별 코어 묶음(스레드 수만큼, h54.core_blocks 와 같은 규칙). 허용 표지가 없거나 코어가 모자라면 None."""
    permit = run_permitted() if permit is None else permit
    if not permit or not hasattr(os, "sched_getaffinity") or int(workers) <= 0:
        return None
    avail = sorted(os.sched_getaffinity(0))
    T, Wn = int(threads), int(workers)
    if Wn * T > len(avail):
        return None
    return [avail[i * T:(i + 1) * T] for i in range(Wn)]


def _pool_init(init, init_args, core_queue, threads):
    warnings.filterwarnings("ignore")
    if os.environ.get("XBATCH_GPU", "") != "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for v in THREAD_VARS:
        os.environ[v] = str(int(threads))
    if core_queue is not None:
        try:
            os.sched_setaffinity(0, set(core_queue.get(timeout=30)))
        except Exception:                                                 # noqa: BLE001  고정에 실패하면 고정 없이 돈다
            pass
    if init is not None:
        init(*init_args)


def execute(units, fn, workers=0, threads=1, init=None, init_args=(), retries=2, permit=None, log=None):
    """작업 단위 실행(h54.execute 와 같은 구조: workers ≤ 0 이면 이 프로세스에서 차례로, 그 밖은 spawn 풀, 코어 고정, 풀 깨짐 재시도).
    fn(*unit) 은 모듈 최상위 함수이고 요약 dict 를 돌려준다. 반환 (완료 목록, 실패 목록[(단위, 오류 문자열)]). 로그에는 단위 이름과 수만 쓴다."""
    log = log or (lambda s: print(s, flush=True))
    done, failed = [], []
    units = list(units)
    if not units:
        return done, failed
    if int(workers) <= 0:
        if init is not None:
            init(*init_args)
        for u in units:
            try:
                done.append(fn(*u)); log(f"  [완료] {'|'.join(str(v) for v in u)} · {len(done)}/{len(units)}")
            except Exception as e:                                       # noqa: BLE001
                failed.append((u, repr(e)[:300])); log(f"  [실패] {'|'.join(str(v) for v in u)}: {repr(e)[:200]}")
        return done, failed
    ctx = multiprocessing.get_context("spawn")
    blocks = core_blocks(workers, threads, permit)
    remaining, attempt = list(units), 0
    while remaining:
        broken = []
        q = None
        if blocks:
            q = ctx.Queue()
            for b_ in blocks:
                q.put(b_)
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx, initializer=_pool_init, initargs=(init, tuple(init_args), q, threads)) as ex:
            futs = {ex.submit(fn, *u): u for u in remaining}
            for f in as_completed(futs):
                u = futs[f]
                try:
                    done.append(f.result()); log(f"  [완료] {'|'.join(str(v) for v in u)} · {len(done)}/{len(units)}")
                except BrokenProcessPool:
                    broken.append(u)
                except Exception as e:                                   # noqa: BLE001
                    failed.append((u, repr(e)[:300])); log(f"  [실패] {'|'.join(str(v) for v in u)}: {repr(e)[:200]}")
        if not broken:
            break
        attempt += 1
        if attempt > int(retries):
            failed += [(u, f"프로세스 풀이 {attempt}회 깨졌다") for u in broken]
            break
        remaining = broken
        log(f"[pool] 워커 비정상 종료. 남은 {len(remaining)} 단위로 풀을 다시 만든다({attempt}/{retries})")
    return done, failed


def torch_deterministic(seed=0) -> dict:
    """XD-learn(DeepSets)의 결정적 설정(1절 플랫폼): CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms(True), cudnn 결정성,
    seed 고정. 동결 기록에 쓸 판(torch, CUDA, cuDNN, 장치)을 돌려준다. CUDA 문맥을 만들기 전에 부른다."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return dict(torch=torch.__version__, cuda=str(torch.version.cuda), cudnn=str(torch.backends.cudnn.version()),
                device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""))


# ================================================================ 20. 명령행(세기 범주, 라벨 값 미사용)
def _cli_split_plan(a):
    """부록 A 를 다시 내고 대조한다. 출력은 분할 수, seed, 셀 수, 블록 수뿐이다."""
    rows_out = []
    for fam, al in APPENDIX_A:
        keep, rows = split_plan(a, al, fam, check=False)
        s = plan_summary(rows)
        try:
            check_appendix_a(fam, al, rows); okv = "같음"
        except AssertionError as e:
            okv = f"다름: {e}"
        rows_out.append((fam, al, s, okv))
        ex = ", ".join(f"{sp}({st})" for sp, st in sorted(s["excluded"].items()))
        print(f"  [{fam}] {al}: 유효 분할 {s['n_valid']}/{s['n_seeds']} · 뺀 seed {ex or '없음'} · |A| {s['n_A']} · 채점 블록 {s['nb_eval']} · "
              f"합집합 {s['nb_union']} · 부록 A 와 {okv}", flush=True)
    bad = [r for r in rows_out if r[3] != "같음"]
    print(f"[split-plan] 부록 A 행 {len(rows_out)}개 가운데 다른 행 {len(bad)}개", flush=True)
    return 1 if bad else 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="xbatch 공용 골격 점검(세기 범주, 라벨 값 미사용)")
    ap.add_argument("--self-check", action="store_true", help="동결 해시, 제약 파일, 실행 표 존재, 로컬 자원")
    ap.add_argument("--split-plan", action="store_true", help="새 seed 분할 계획을 다시 내고 부록 A 와 대조한다")
    ap.add_argument("--lic-check", action="store_true", help="네 실행 표의 약관 미확인 셀 수")
    ap.add_argument("--deps-check", action="store_true", help="설치 판과 제약 파일 대조(Rescale 설치 뒤 로그)")
    ap.add_argument("--manifest", default="", help="묶음 정보 JSON 경로(data/processed/xbatch 또는 work 아래)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--allow-local", action="store_true")
    a = ap.parse_args(argv)
    guard("count")
    rc = 0
    with restricted_output():
        if a.self_check or not (a.split_plan or a.lic_check or a.deps_check or a.manifest):
            print(f"[self-check] 계획 {PLAN_DOC} {PLAN_REVISION}, T0 {PLAN_COMMIT}", flush=True)
            for k in FROZEN:
                print(f"  동결 {k}: sha256 앞 16자 {assert_frozen(k)} (계획 값과 같음)", flush=True)
            pins = read_constraints()
            print(f"  제약 파일 {CONSTRAINTS.relative_to(ROOT)}: {', '.join(f'{k}=={v}' for k, v in pins.items())}", flush=True)
            if pins != PINNED:
                print("  [warn] 제약 파일이 1절의 고정 판과 다르다", flush=True); rc = 1
            print(f"  설치 판과 다른 고정 판 {len(check_constraints())}개(로컬에서는 정상, Rescale 설치 뒤에는 0 이어야 한다)", flush=True)
            for s in RUN_TABLE_DIRS:
                print(f"  실행 표 {s}: {'있음' if (LGD_DIR / s / 'fidelity_base_v3.csv').exists() else '없음'}", flush=True)
            r = smoke_env_report()
            print(f"  로컬 자원: 가용 메모리 {r['mem_available_gb']} GB, 코어 묶음 {r['cores']}, 허용 표지 {r['permitted']}", flush=True)
        if a.deps_check:
            v = deps_versions()
            print("[deps] 확인: " + " · ".join(f"{k} {x}" for k, x in v.items()), flush=True)
            mm = check_constraints()
            print(f"[deps] 제약 파일과 다른 판 {len(mm)}개" + (": " + ", ".join(f"{k} {w}≠{h}" for k, w, h in mm) if mm else ""), flush=True)
            rc = max(rc, 1 if mm else 0)
        if a.lic_check:
            for r in lic_check(strict=False):
                print(f"[약관] {r['spec']}: 행 {r['n_rows']}, 약관 미확인 셀 {r['n_unverified']}, sha256 {r['sha256_16']}, 토양 표 연결 {r['soil_link']}",
                      flush=True)
                rc = max(rc, 1 if r["n_unverified"] else 0)
        if a.split_plan:
            ha = h54_args(splits=TRANSFER_SPLITS, threads=a.threads)
            rc = max(rc, _cli_split_plan(ha))
        if a.manifest:
            man = payload_manifest(out=a.manifest)
            print(f"[manifest] 파일 {man['n_files']}개, 없는 파일 {len(man['missing'])}개, 미커밋 경로 {len(man['git_uncommitted_paths'])}개 → {a.manifest}",
                  flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
