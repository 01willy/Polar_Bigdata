"""H51 · LGD(새 지역) 실행기. 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.6(개정 10, 13, 14)과
docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 7절(커밋 316714c, 개정 1)의 실행 표 규칙을 구현한다.

호출 방식(개정 13, WRAPUP 7.4 (b)2 의 (가))
  대상별 자료 디렉터리(data/processed/lgd/run_tables/<spec>/)에 실행 표를 h40 의 기본 파일 이름(fidelity_base_v3.csv)으로 두고,
  토양 도일은 e5_soil_tdd_v4.csv 를 기본 파일 이름(e5_soil_tdd_v3.csv)의 기호 연결로 둔 뒤 h40 을 --data-dir 로 부른다.
  하위 지역 대응표는 두지 않는다(LGD 대상은 모두 macro 수준이다). h40 은 고치지 않고 importlib 로 불러 h40.main 을 호출한다.
  실행 표는 lgd_eligibility_v1.write_run_table_text(v4 원문 줄 선택, 새 행의 source_id 만 F4_direct)로 만든다.

실행 표(spec)
  주 설정     Tibet(Tibet_LGD), NAtlantic, Russia_C(v3 loc 17557 제외, WRAPUP 7.3 (a)8)
  L40         Russia_W·Russia_E·Canada 확충판, Russia_E 확충판의 Kytalyk 제외 변형
  L39         Tibet 지온 유도 라벨(aux_temp_L39 행을 이 표에서만 F4_direct 로 둔다)
  L41         변형 (a)–(h)(개정 13 의 정의). 적격 표에서 부적격인 변형은 실행하지 않는다('변형 불가'). (h)는 점 추정만
  L42         주 설정 + 다른 새 지역(Tibet, NAtlantic, Russia_C)의 대상 직접 라벨 셀을 원천에 더한 표. NAtlantic 셀을 원천에 더하면
              v3 하위 지점 평균 행(17520, 17569)을 뺀다(개정 13). v3 QTP_CN(17389)은 원천에 남긴다
  repro       v4 에서 v3 행만 고른 표(러시아 W, 분할 1). WRAPUP 7.5 (c)2 의 재현 점검용(LG 본 실행 조각 회수 뒤)
  v3local     v4 에서 v3 행만 고른 표(러시아 W·E·캐나다, 분할 5). (c)2 에서 CatBoost 만 허용 차를 넘을 때 L40 의 로컬 v3 기준값
  야말 GGD402 보조 148셀은 어느 표에도 넣지 않는다. 주 설정의 원천은 v3 F4_direct 셀이다(다른 새 지역 셀은 L42 에서만).

범위(6B.6): CPU 방법 축(P0–P3, D0, D1, R0–R3, V1, V1r, CatBoost), 분할 5, n 격자 {0, 3, 10, 40, 160, 320, 1000, 전량}(|A| 미만과 전량),
  추출 5, seed 2, λ {0.25, 0.5, 1.0}. α 축(--alphas 1), 배치 축(--place-n-grid 빈 값), 학습기 축(--learners catboost_lo,
  --learner-targets NONE:x), α 중첩 선택(--nested-targets NONE:x)은 등록 범위가 아니므로 끈다. 조각 tag 는 lgd, 산출은
  data/processed/lgd/<spec>/shards/.

실행 표 점검(WRAPUP 7.4 (b)4, --count-only)
  대상마다 h40 Data(실행 표)의 대상 셀 수를 ext_cells_v1 의 대상 셀 수 + v3 셀 수와 대조하고, 적격 표(lgd_eligibility_v1.csv,
  _splits.csv)의 셀 수, 분할 구조(n_A, n_eval, nb_eval, n_src, valid, dup_of)와 대조한다. 불일치가 있는 표는 실행하지 않는다.
  적합 수는 h40.run_ctx 의 dry 실행으로 센다. 이때 작업 단위의 라벨 자리에 √TDD 를 넣은 가짜 라벨을 쓴다(적합 수는 라벨 값과 무관하다).
  WRAPUP 7.2 (a)7(새 지역의 계수 비와 P0 오차는 LGD 실행 전에 계산하지 않는다) 때문에 h40 --count-only(build_ctx 가 대상 라벨의
  계수 비 E_own, E_A, E_B 를 계산해 적합 수 표에 쓴다)를 쓰지 않는다. h40 Data 를 적재할 때 모든 셀의 z = log(ALT/√TDD) 가 메모리에서
  계산되지만(h40 전처리) 이 값은 집계하거나 출력하지 않는다. 같은 단계에서 WRAPUP 7.2 (a)10 의 티베트 공변량 지지 밖 비율(고도, √TDD 가
  원천 셀의 0.5–99.5 백분위 밖)을 계산한다(공변량만 쓴다). 추정 시간은 본 묶음(main, l40, l39, l41, l42 에서 실제로 적합할 표)과
  선택 묶음(repro, v3local. 필요할 때만 돌린다)을 나눠 적는다(manifest 의 count.bundles).

같은 실행 표(same_as, 개정 14): L41 변형의 실행 표가 같은 대상의 주 설정 표와 같으면 다시 적합하지 않고 주 설정 조각을 쓴다. '같다'는
  바이트가 같거나, 열 이름·행 수·문자열 열이 같고 수치 열의 최대 절대 차가 1e-9 이하인 경우다(판정 근거는 manifest 의 same_as_basis).

재현 점검(--repro-check, WRAPUP 7.5 (c)2, 개정 14 의 키 범위): repro 표의 조각과 LG 본 실행 조각(--lg-shards, tag lg)을 공통 키(방법 축,
  α 1, 셀 무작위)의 키별 셀 가중 RMSE 로 대조한다. 키 분류(key_class): 물리식 = P0–P3(허용 차 1e-9 cm), ridge = V1(sklearn Ridge 앵커)과
  ridge 학습기(0.02 cm), CatBoost = catboost_lo 키(0.02 cm). 범위는 둘이다. 주 판정 = 6A.6 의 기준 방법(P0, P1, P2, D0, D1, R0, R1, R2).
  병기 = 방법 축 전체(P3, R3, V1, V1r 을 더한다). (c)2 의 상태(물리식 불통과, CatBoost 불통과, 통과)는 주 판정 범위로 정한다.
  L40 의 대비(D0 − P0, R1 − P0, P1 − P0)가 기준 방법만 쓰기 때문이다. Δ 와 판정은 출력하지 않는다.
  산출 data/processed/lgd/lgd_repro_gate.csv(범위마다 한 행, scope 열).

스모크(--smoke, 개정 14): 요청한 표의 실행 표를 그대로 쓰되 대상 셀(새 행과 v3 구성 행)의 alt_cm 을 합성 라벨(1.4·√TDD·exp(0.2ε),
  seed 고정, 5–590 cm)로 바꾼 표를 data/processed/lgd/smoke/run_tables/<표>/ 에 두고 h40 을 부른다. 분할 1, n {0, 3, 10, 전량},
  추출 1, seed 1, tag lgd_smoke, 조각 data/processed/lgd/smoke/<표>/shards/. WRAPUP 7.2 (a)7(새 지역의 계수 비와 P0 오차를 LGD 본 실행
  전에 계산하지 않는다) 때문에 실제 대상 라벨을 쓰지 않는다. 경로(실행 표 → h40 → 조각)와 로컬 적합 시간만 확인한다. 합성 라벨 조각의
  RMSE 는 어디에도 쓰지 않는다. 산출 data/processed/lgd/smoke/lgd_smoke_meta.json(단위별 상태, 적합 수, 벽시계, h40 추정 대비 비).

실행 보호와 자원(개정 13, 14, WRAPUP 8.5, 워크플로 지시)
  학습을 하는 실행(본 실행, 재현 점검, 스모크)은 --allow-local 이 없으면 거부한다. 워커 기본 2, 스레드 4(워커당. CatBoost thread_count 와
  BLAS 스레드). nice 10 이상으로 낮춘다. 시작 전 1분 load average 가 64 를 넘으면 시작하지 않는다. WRAPUP 8.5 의 '실행 중 96 을 넘으면
  새 작업 단위를 받지 않는다'는 표 단위로 적용한다(개정 14): 표(h40 호출 한 번, 작업 단위 1–5개)를 시작하기 전에 확인하고 96 을 넘으면
  남은 표를 시작하지 않는다. h40 을 고치지 않으므로 h40 안의 작업 단위 사이에서는 확인하지 않는다.
  본 실행(PE·L38–L42 의 표)은 사용자 확인 뒤 시작한다(6B.6 5단계). --count-only 는 학습을 하지 않는다(스레드 1–2 권장).
  시험 전용 인자(--skip-hash-check, --h40-extra)는 환경 변수 LGD_TEST=1 일 때만 받는다. 실행 인자 전체를 manifest 의 runs 에 적는다.

산출(data/processed/lgd/)
  run_tables/<spec>/                 실행 표와 토양 표 연결(약관 미확인 행 포함, .gitignore 로 커밋 제외)
  <spec>/shards/lgd__cpu__<대상>__x__s<분할>_{runs.csv, blocksse.npz, unit.json}
  lgd_count.csv, lgd_count_summary.csv  표·분할별 구조와 적합 수, 점검 결과
  lgd_run_manifest.json              표 정의, 해시, same_as 와 근거, 점검 결과, 묶음별 추정 시간, 티베트 지지 밖 비율, 판 목록, 실행 인자,
                                     스모크 요약(smoke)
  lgd_repro_gate.csv                 재현 점검(scope = base 주 판정, all 병기)
  smoke/                             스모크(합성 대상 라벨). lgd_smoke_meta.json 만 커밋 대상이다(smoke/.gitignore)
  logs/                              실행 기록

실행(ROOT)
  점검:   OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/h51_lgd_run.py --count-only --threads 2
  본 실행: (사용자 확인 뒤) setsid nohup nice -n 10 python3 scripts/3_deep_learning/h51_lgd_run.py --specs main,l40,l39,l41,l42 \
            --allow-local --workers 2 --threads 4 --resume > data/processed/lgd/logs/run.log 2>&1 &
  재현 점검(LG 본 실행 회수 뒤): nice -n 10 python3 scripts/3_deep_learning/h51_lgd_run.py --specs repro --allow-local --workers 0 \
            --threads 4 --repro-check --lg-shards results/rescale_lg/data/processed/lg/shards
  스모크:  OMP_NUM_THREADS=4 nice -n 10 python3 scripts/3_deep_learning/h51_lgd_run.py --smoke --specs NAtlantic,Russia_C --allow-local \
            --workers 1 --threads 4
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path


def _peek(flag, default=None):
    av = sys.argv
    for i, v in enumerate(av):
        if v == flag and i + 1 < len(av):
            return av[i + 1]
        if v.startswith(flag + "="):
            return v.split("=", 1)[1]
    return default


def _threads_env():
    """numpy 를 부르기 전에 스레드 수를 정한다. 학습을 하는 실행(--allow-local)만 --threads(기본 4)를 쓰고, 그 밖은 1 이다."""
    if "--allow-local" in sys.argv and "--count-only" not in sys.argv:
        return str(_peek("--threads", "4"))
    return str(min(int(_peek("--threads", "1")), 2))


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _threads_env())

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, path):
    """파일 경로에서 모듈을 읽어 sys.modules 에 등록한다. 모듈 최상위에서 부른다(spawn 워커가 주 모듈을 다시 읽을 때 등록되어야
    h40._worker_init 의 역직렬화가 된다). 이미 등록되어 있으면 그 모듈을 쓴다."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("h40_label_grid", SCRIPT_DIR / "h40_label_grid.py")

from polar.m1_core import eval_mask, half_split_blocks                                               # noqa: E402
from polar.h4_common import load_stores, seed_of                                                     # noqa: E402

_ELIG = None


def elig_mod():
    """lgd_eligibility_v1(실행 표 작성 함수와 상수). 부모 프로세스에서만 필요하므로 늦게 읽는다."""
    global _ELIG
    if _ELIG is None:
        p = ROOT / "scripts" / "1_data_prep"
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        _ELIG = _load("lgd_eligibility_v1", p / "lgd_eligibility_v1.py")
    return _ELIG


# ================================================================ 고정값(계획서 6B.6, 개정 13. 바꾸면 사전 등록에서 벗어난다)
PROC = ROOT / "data" / "processed"
EXT = PROC / "ext_labels"
LOC0 = 20000
TAG = "lgd"
N_GRID = "0,3,10,40,160,320,1000,all"
SPLITS = 5
DRAWS = 5
SEEDS = 2
THREADS = 4
WORKERS_MAX = 2
LOAD_START_MAX = 64.0
LOAD_RUN_MAX = 96.0
NICE_MIN = 10
GATE_TOL_PHYS = 1e-9
GATE_TOL_ML = 0.02
PHYS_METHODS = ("P0", "P1", "P2", "P3")
BASE_METHODS = ("P0", "P1", "P2", "D0", "D1", "R0", "R1", "R2")                               # 6A.6 의 기준 방법(h42.BASE_METHODS)
SAME_TOL = 1e-9
SMOKE_TAG = "lgd_smoke"
SMOKE_N_GRID = "0,3,10,all"
NEW_MAIN = {"Tibet": "Tibet_LGD", "NAtlantic": "NAtlantic", "Russia_C": "Russia_C"}          # spec → h40 대상 이름(v4 macro)
EXT_MACRO = {"Tibet_LGD": "Tibet"}                                                            # v4 macro → ext_cells_v1 의 macro
EXPANDED = {"Russia_W_expanded": "Russia_W", "Russia_E_expanded": "Russia_E", "Canada_expanded": "Canada"}
V3LOCAL = ("Russia_W", "Russia_E", "Canada")
L41_VARIANTS = ("a", "b", "c", "d", "e", "f", "g", "h")
REGISTERED = {                                                                                 # 개정 13 에 등록된 sha256
    "fidelity_base_v4.csv": "4c387b2b4fb43ef602d9e0b8fe50054a6699428d5a593425b3d309e798fce52a",
    "e5_soil_tdd_v4.csv": "2ef333b24d60332af1a432ceff5ee7bb73a2154d0fc23659e8e694d6674029f6",
    "fidelity_base_v4_labels.csv": "9f5af274f4b835045d90e202e35ebdc872e54e0affba218835735a43a87b2247",
    "lgd_eligibility_v1.csv": "5b23a9a8d7bc79eb663605a142a58e7d1ed6a5fa1555f436e007611c155f364b",
    "lgd_eligibility_v1_splits.csv": "765b3f25997ae779270da7f726288e9b174f6391be438f23a55253c6b3ba8cfa",
}
GROUPS = dict(main=["Tibet", "NAtlantic", "Russia_C"],
              l40=["Russia_W_expanded", "Russia_E_expanded", "Canada_expanded", "Russia_E_expanded_noKytalyk"],
              l39=["Tibet_L39_temp"],
              l41=[f"{t}_L41{v}" for t in NEW_MAIN for v in L41_VARIANTS],
              l42=[f"{t}_L42" for t in NEW_MAIN],
              repro=["repro_Russia_W"],
              v3local=[f"v3local_{t}" for t in V3LOCAL])
DEFAULT_GROUPS = ("main", "l40", "l39", "l41", "l42")


def tm_name(spec):
    """집계에서 쓰는 저장소 이름. 주 설정은 v3 와 같은 대상 이름(대상|x), 그 밖은 h42 의 변형 저장소 표기(대상~표지|x)다."""
    t = spec["target"]
    suf = spec.get("suffix", "")
    return f"{t}~{suf}|x" if suf else f"{t}|x"


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H51 LGD 실행기(계획서 6B.6, WRAPUP 7절)")
    ap.add_argument("--specs", default=",".join(DEFAULT_GROUPS),
                    help="쉼표 목록. 묶음 이름(main, l40, l39, l41, l42, repro, v3local, all) 또는 표 이름. 기본 = main,l40,l39,l41,l42")
    ap.add_argument("--count-only", action="store_true", help="학습 없이 실행 표를 만들고 셀 수·분할 구조·적합 수를 점검한다(WRAPUP 7.4 (b)4)")
    ap.add_argument("--allow-local", action="store_true", help="학습을 하는 실행을 허용한다(개정 13: LGD 는 로컬 서버에서 한다)")
    ap.add_argument("--workers", type=int, default=WORKERS_MAX, help=f"CPU 프로세스 수(최대 {WORKERS_MAX}, WRAPUP 8.5)")
    ap.add_argument("--threads", type=int, default=THREADS, help="프로세스당 스레드(BLAS, CatBoost). 본 실행은 4(6B.6)")
    ap.add_argument("--resume", action="store_true", help="h40 --resume(설정 해시가 같은 완료 조각은 건너뛴다)")
    ap.add_argument("--rerun-identical", action="store_true", help="주 설정과 같은 실행 표의 변형도 다시 적합한다(기본은 주 설정 조각을 쓴다)")
    ap.add_argument("--out-dir", default="data/processed/lgd")
    ap.add_argument("--repro-check", action="store_true", help="repro 조각을 LG 본 실행 조각과 대조한다(WRAPUP 7.5 (c)2)")
    ap.add_argument("--lg-shards", default="results/rescale_lg/data/processed/lg/shards", help="LG 본 실행 조각 디렉터리(읽기 전용)")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--load-start-max", type=float, default=LOAD_START_MAX)
    ap.add_argument("--load-run-max", type=float, default=LOAD_RUN_MAX)
    ap.add_argument("--smoke", action="store_true", help="합성 대상 라벨로 경로와 로컬 적합 시간을 확인한다(분할 1, n 소수, 추출 1, seed 1). "
                                                        "산출 <out-dir>/smoke/")
    ap.add_argument("--smoke-n-grid", default=SMOKE_N_GRID)
    ap.add_argument("--skip-hash-check", action="store_true", help="시험 전용(LGD_TEST=1): 등록 해시 점검을 건너뛴다")
    ap.add_argument("--h40-extra", default="", help="시험 전용(LGD_TEST=1): h40 인자를 덧붙인다(공백 구분). 본 실행에서는 쓰지 않는다")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    if (a.skip_hash_check or a.h40_extra) and os.environ.get("LGD_TEST", "") != "1":
        raise SystemExit("[거부] --skip-hash-check, --h40-extra 는 시험 전용이다(환경 변수 LGD_TEST=1 일 때만 받는다. 개정 14)")
    if a.smoke and (a.count_only or a.repro_check):
        raise SystemExit("--smoke 는 --count-only, --repro-check 와 함께 쓰지 않는다")
    a.OUT = Path(a.out_dir) if os.path.isabs(a.out_dir) else ROOT / a.out_dir
    a.LG_SHARDS = Path(a.lg_shards) if os.path.isabs(a.lg_shards) else ROOT / a.lg_shards
    if a.workers > WORKERS_MAX:
        raise SystemExit(f"--workers 는 {WORKERS_MAX} 이하다(WRAPUP 8.5, 개정 13)")
    if a.workers < 0:
        raise SystemExit("--workers 는 0 이상이다")
    names = []
    for tok in [v.strip() for v in a.specs.split(",") if v.strip()]:
        if tok == "all":
            names += [s for g in DEFAULT_GROUPS for s in GROUPS[g]]
        elif tok in GROUPS:
            names += GROUPS[tok]
        else:
            names.append(tok)
    a.SPEC_NAMES = list(dict.fromkeys(names))
    return a


# ================================================================ 해시·자원
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_registered(proc=PROC):
    """개정 13 에 등록된 입력 표 해시와 대조한다. 다르면 중단한다."""
    got = {k: sha256(proc / k) for k in REGISTERED}
    bad = {k: v for k, v in got.items() if v != REGISTERED[k]}
    if bad:
        raise SystemExit(f"[거부] 등록 해시와 다른 입력 표: {sorted(bad)}. 개정 13 의 판(v4 sha256 4c387b2b…)이 아니다")
    return got


def load_avg():
    try:
        return float(os.getloadavg()[0])
    except OSError:
        return float("nan")


def lower_priority(min_nice=NICE_MIN):
    """프로세스 우선순위를 nice min_nice 이상으로 낮춘다(워커는 물려받는다)."""
    try:
        cur = os.nice(0)
        if cur < min_nice:
            os.nice(min_nice - cur)
        return os.nice(0)
    except OSError:
        return None


def versions():
    out = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, host=platform.node(),
               machine=platform.machine())
    for m in ("sklearn", "catboost", "scipy"):
        try:
            out[m] = __import__(m).__version__
        except Exception:                                                                            # noqa: BLE001
            out[m] = "NA"
    return out


# ================================================================ 실행 표 정의
class Tables:
    """v4 표, 라벨 부가 표, ext_cells_v1, 적격 표. 표 정의(spec)를 만든다."""

    def __init__(self, proc=PROC):
        self.proc = Path(proc)
        self.v4_path = self.proc / "fidelity_base_v4.csv"
        self.soil_path = self.proc / "e5_soil_tdd_v4.csv"
        self.lab = pd.read_csv(self.proc / "fidelity_base_v4_labels.csv", low_memory=False)
        self.el = pd.read_csv(self.proc / "lgd_eligibility_v1.csv")
        self.el_sp = pd.read_csv(self.proc / "lgd_eligibility_v1_splits.csv")
        self.ext = pd.read_csv(EXT / "ext_cells_v1.csv", low_memory=False) if (EXT / "ext_cells_v1.csv").exists() else None
        E = elig_mod()
        self.DROP_V3 = {k: list(v) for k, v in E.DROP_V3.items()}
        self.LARGEST = dict(E.LARGEST_SURVEY)
        new = self.lab[self.lab.part == "new"].copy()
        self.new = new
        self.tgt = new[new.lgd_role == "target"].copy()
        self.v3lab = self.lab[self.lab.part == "v3"].set_index("loc_id")
        parent_avg = self.v3lab.index[self.v3lab.lgd_role.astype(str).str.startswith("v3_parent_average_of_subsites")]
        self.parent_avg = sorted(int(x) for x in parent_avg)
        assert self.parent_avg == [17520, 17569], f"v3 하위 지점 평균 행이 개정 13 기록과 다르다: {self.parent_avg}"
        self.qtp_v3 = sorted(int(x) for x in self.v3lab.index[self.v3lab.lgd_role.astype(str).str.startswith("v3_f4_temp_derived")])

    # -- 새 행 선택(lgd_eligibility_v1.main 의 선택과 같은 규칙)
    def ids(self, macro, ok=None):
        m = (self.tgt.macro_v4 == macro).values
        if ok is not None:
            m &= np.asarray(ok, bool)
        return self.tgt.loc_id.values[m].astype(int)

    def masks(self, t_macro):
        g = self.tgt
        largest = self.LARGEST.get(t_macro)
        return dict(
            a=(g.l41a_all_direct3 == 1).values, b=(g.l41b_early_single == 0).values, c=(g.l41c_dataset_statement == 0).values,
            d=(g.year_max >= 2010).values, e=(g.cens_affected == 0).values, g=(g.e5_glacier_grid == 0).values,
            h=(~g.sources.fillna("").str.split(";").map(lambda L: largest in L)).values if largest else np.zeros(len(g), bool))

    def v3_members(self, macro):
        return sorted(int(x) for x in self.v3lab.index[(self.v3lab.lgd_role == f"v3_member_of_{macro}")])

    def elig_row(self, spec_name):
        r = self.el[self.el.spec == spec_name]
        return r.iloc[0].to_dict() if len(r) else None

    def specs(self):
        """모든 등록 표 정의. 반환 {이름: dict}."""
        out = {}

        def add(name, target, kind, keep, suffix="", drop=(), alt=None, roles=("target",), splits=SPLITS, base=None, point_only=False,
                el_name=None, note="", group=""):
            elr = self.elig_row(el_name or name)
            out[name] = dict(name=name, target=target, kind=kind, keep=np.asarray(sorted(set(int(x) for x in keep)), int), suffix=suffix,
                             drop_v3=sorted(set(int(x) for x in drop)), alt=alt, roles=list(roles), splits=int(splits), base=base,
                             point_only=bool(point_only), elig=elr, el_name=el_name or (name if elr is not None else None), note=note,
                             group=group)
        for sp, t in NEW_MAIN.items():
            add(sp, t, "new_region", self.ids(t), drop=self.DROP_V3.get(t, []), group="main")
        for sp, t in EXPANDED.items():
            add(sp, t, "expanded_L40", self.ids(t), suffix="exp", group="l40")
        kyt = self.tgt.label_subtypes.fillna("").str.contains("pf_top_below_frozen_al").values
        add("Russia_E_expanded_noKytalyk", "Russia_E", "expanded_L40_variant", self.ids("Russia_E", ~kyt), suffix="expnokyt", group="l40")
        aux = self.new[self.new.lgd_role == "aux_temp_L39"].loc_id.values
        add("Tibet_L39_temp", "Tibet_LGD", "aux_L39", aux, suffix="L39", roles=("aux_temp_L39",), base="Tibet", group="l39",
            note="지온 유도 라벨. 직접 라벨과 위치가 달라 위치 효과가 섞인다(6B.5 L39)")
        for sp, t in NEW_MAIN.items():
            mk = self.masks(t)
            gl_v3 = [int(i) for i in self.v3lab.index[(self.v3lab.macro_v4 == t) & (self.v3lab.e5_glacier_grid == 1)
                                                      & (self.v3lab.source_id == "F4_direct")]]
            for v in L41_VARIANTS:
                name = f"{sp}_L41{v}"
                elr = self.elig_row(name)
                drop = list(self.DROP_V3.get(t, []))
                alt = None
                if v == "f":
                    keep = self.ids(t)
                    alt = self.tgt.set_index("loc_id").loc[keep, "alt_cm_cens_lb"].to_dict()
                elif v == "g":
                    keep = self.ids(t, mk["g"]); drop += gl_v3
                elif v == "h" and t not in self.LARGEST:
                    keep = np.zeros(0, int)
                else:
                    keep = self.ids(t, mk[v])
                elig = bool(elr is not None and bool(elr.get("eligible", False)))
                notes = (["점 추정만(WRAPUP 7.3 (a)9)"] if v == "h" else []) + ([] if elig else ["변형 불가(부적격)"])
                add(name, t, f"variant_L41{v}", keep, suffix=f"L41{v}", drop=drop, alt=alt, base=sp, group="l41",
                    point_only=(v == "h") or not elig, note="; ".join(notes))
        for sp, t in NEW_MAIN.items():
            others = [o for o in NEW_MAIN if o != sp]
            keep = np.concatenate([self.ids(t)] + [self.ids(NEW_MAIN[o]) for o in others])
            drop = list(self.DROP_V3.get(t, [])) + (self.parent_avg if "NAtlantic" in others else [])
            add(f"{sp}_L42", t, "source_L42", keep, suffix="L42", drop=drop, base=sp, group="l42", el_name=sp,
                note=f"원천에 {', '.join(others)} 의 대상 직접 라벨 셀을 더한다" + ("; v3 하위 지점 평균 행 17520·17569 제외" if "NAtlantic" in others
                                                                          else "") + ("; v3 QTP_CN 17389 는 원천에 남긴다" if self.qtp_v3 else ""))
        add("repro_Russia_W", "Russia_W", "repro", [], suffix="repro", splits=1, group="repro", el_name="v3ref_Russia_W",
            note="v4 에서 v3 행만 고른 표(WRAPUP 7.5 (c)2). 분할 1")
        for t in V3LOCAL:
            add(f"v3local_{t}", t, "v3local", [], suffix="v3local", group="v3local", el_name=f"v3ref_{t}",
                note="v4 에서 v3 행만 고른 표. (c)2 의 CatBoost 불통과 때 L40 의 로컬 v3 기준값")
        return out

    # -- (b)4 기대 셀 수: ext_cells_v1 의 대상 셀 수 + v3 셀 수
    def expected_counts(self, s):
        t = s["target"]
        kind = s["kind"]
        drop = set(s["drop_v3"])
        members = [i for i in self.v3_members(t) if i not in drop]
        if kind == "aux_L39":
            members = []                                                      # L39 표의 대상 셀은 지온 유도 새 행뿐이다(v3 QTP_CN 은 macro Tibet)
        v3n = len(members)
        if kind in ("repro", "v3local"):
            return dict(expected_new=0, expected_v3=v3n, ext_source="없음(v3 행만)")
        if self.ext is None:
            return dict(expected_new=np.nan, expected_v3=v3n, ext_source="ext_cells_v1.csv 없음")
        ex = self.ext.copy()
        ex["cell_uid"] = ex.label_set.astype(str) + "|" + ex.cell_key.astype(str)
        nl = self.new.copy()
        nl["cell_uid"] = nl.label_set.astype(str) + "|" + nl.cell_key.astype(str)
        ex = ex.merge(nl[["cell_uid", "loc_id", "e5_glacier_grid"]], on="cell_uid", how="left")
        mac = EXT_MACRO.get(t, t)
        if kind == "aux_L39":
            sel = (ex.label_set == "temp") & (ex.in_v4 == 1) & (ex.macro == mac)
        else:
            sel = (ex.target == 1) & (ex.in_v4 == 1) & (ex.macro == mac)
        v = kind.replace("variant_L41", "") if kind.startswith("variant_L41") else ""
        if v == "a":
            sel &= ex.l41a_all_direct3 == 1
        elif v == "b":
            sel &= ex.l41b_early_single == 0
        elif v == "c":
            sel &= ex.l41c_dataset_statement == 0
        elif v == "d":
            sel &= ex.year_max >= 2010
        elif v == "e":
            sel &= ex.cens_affected == 0
        elif v == "g":
            sel &= ex.e5_glacier_grid == 0
        elif v == "h":
            largest = self.LARGEST.get(t)
            sel &= ~ex.sources.fillna("").str.split(";").map(lambda L: largest in L) if largest else False
        elif kind == "expanded_L40_variant":
            sel &= ~ex.label_subtypes.fillna("").str.contains("pf_top_below_frozen_al")
        return dict(expected_new=int(sel.sum()), expected_v3=v3n, ext_source="ext_cells_v1.csv(대상 셀, in_v4 = 1)")


def write_table(T: Tables, s, root: Path):
    """실행 표와 토양 표 연결을 대상별 자료 디렉터리에 쓴다(호출 방식 (가)). 반환 (디렉터리, 기록)."""
    E = elig_mod()
    d = root / "run_tables" / s["name"]
    d.mkdir(parents=True, exist_ok=True)
    gi = root / "run_tables" / ".gitignore"
    if not gi.exists():
        gi.write_text("# 실행 표에는 약관 미확인 자료원의 행(lic_unverified)이 들어 있다. 커밋하지 않는다(WRAPUP 7.5 (c)3).\n*\n!.gitignore\n")
    tab = d / "fidelity_base_v3.csv"
    tmp = d / f".tmp{os.getpid()}.csv"
    wr = E.write_run_table_text(T.v4_path, tmp, s["keep"], drop_v3=s["drop_v3"], alt_override=s["alt"],
                                expect_roles=s["roles"] if len(s["keep"]) else None, lab=T.lab if len(s["keep"]) else None)
    os.replace(tmp, tab)
    soil = d / "e5_soil_tdd_v3.csv"
    if soil.is_symlink() or soil.exists():
        soil.unlink()
    os.symlink(os.path.relpath(T.soil_path, d), soil)
    rec = dict(dir=str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d), table_sha256=sha256(tab), n_lines=sum(1 for _ in open(tab)) - 1,
               soil_target=str(T.soil_path.name), **wr)
    return d, rec


def v3_only_text_check(T: Tables, tab: Path):
    """v3 행만 고른 표가 v3 원문과 region 을 바꾼 3줄만 다른지(6B.6 재현 점검의 해시 대조)."""
    l_run = tab.read_text(encoding="utf-8").splitlines()
    l_v3 = (T.proc / "fidelity_base_v3.csv").read_text(encoding="utf-8").splitlines()
    diff = [(a, b) for a, b in zip(l_run, l_v3) if a != b]
    only_region = True
    if l_run:
        hdr = l_run[0].split(",")
        ir = hdr.index("region")
        for a, b in diff:
            fa, fb = a.split(","), b.split(",")
            if [x for i, x in enumerate(fa) if i != ir] != [x for i, x in enumerate(fb) if i != ir]:
                only_region = False
    return dict(n_lines_run=len(l_run), n_lines_v3=len(l_v3), n_lines_differ=len(diff), differ_only_in_region=bool(only_region),
                ok=bool(len(l_run) == len(l_v3) and len(diff) == 3 and only_region))


def _as_float(vals):
    out = np.full(len(vals), np.nan)
    for i, v in enumerate(vals):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def same_table(p_a: Path, p_b: Path, tol=SAME_TOL):
    """두 실행 표가 같은가(개정 14 의 same_as 규칙). 바이트가 같으면 '바이트 동일'. 아니면 열 이름·행 수·문자열 열이 같고 수치 열의
    최대 절대 차가 tol 이하일 때 '수치 동일'. 반환 (같음, 기록 dict). 기록에는 다른 열과 최대 차만 적는다(값은 적지 않는다)."""
    if sha256(p_a) == sha256(p_b):
        return True, dict(basis="바이트 동일")
    A = pd.read_csv(p_a, dtype=str, keep_default_na=False)
    B = pd.read_csv(p_b, dtype=str, keep_default_na=False)
    if list(A.columns) != list(B.columns) or len(A) != len(B):
        return False, dict(basis="열 또는 행 수가 다르다")
    worst, cols = 0.0, []
    for c in A.columns:
        va, vb = A[c].values, B[c].values
        neq = va != vb
        if not neq.any():
            continue
        fa, fb = _as_float(va[neq]), _as_float(vb[neq])                   # 파이썬 float(정확한 반올림) 로 읽는다
        if not (np.isfinite(fa).all() and np.isfinite(fb).all()):
            return False, dict(basis=f"문자열 또는 결측이 다른 열 {c}")
        d_ = float(np.max(np.abs(fa - fb)))
        cols.append(dict(col=c, n_rows=int(neq.sum()), max_abs_diff=d_))
        worst = max(worst, d_)
    if worst <= tol:
        return True, dict(basis=f"수치 동일(최대 절대 차 {worst:.3g}, 허용 차 {tol:g})", diff_cols=cols)
    return False, dict(basis=f"수치 차 {worst:.3g} > {tol:g}", diff_cols=cols)


def smoke_table(T: Tables, s, d_src: Path, d_out: Path):
    """스모크 실행 표(개정 14): 실행 표를 복사하되 대상 셀(새 행 keep 과 v3 구성 행에서 뺀 행을 제외한 것)의 alt_cm 을 합성 라벨로 바꾼다.
    합성 라벨 = clip(1.4·√TDD·exp(0.2ε), 5, 590) cm, ε 는 seed_of("lgd-smoke", 표) 로 고정. 실제 대상 라벨은 읽기만 하고 쓰지 않는다
    (WRAPUP 7.2 (a)7). 토양 표는 e5_soil_tdd_v4.csv 연결이다. 반환 기록 dict(대상 loc_id 목록 포함)."""
    d_out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(d_src / "fidelity_base_v3.csv", low_memory=False)
    drop = set(int(x) for x in s["drop_v3"])
    tgt = sorted(set(int(x) for x in s["keep"]) | {i for i in T.v3_members(s["target"]) if i not in drop})
    m = df.loc_id.astype(int).isin(tgt).values
    rng = np.random.RandomState(seed_of("lgd-smoke", s["name"]))
    sq = df.e5_sqrt_tdd.values.astype(float)
    sq = np.where(np.isfinite(sq), sq, float(np.nanmedian(sq)))
    syn = np.clip(1.4 * sq * np.exp(0.2 * rng.randn(len(df))), 5.0, 590.0)
    df.loc[m, "alt_cm"] = syn[m]
    df.to_csv(d_out / "fidelity_base_v3.csv", index=False)
    soil = d_out / "e5_soil_tdd_v3.csv"
    if soil.is_symlink() or soil.exists():
        soil.unlink()
    os.symlink(os.path.relpath(T.soil_path, d_out), soil)
    return dict(n_rows=int(len(df)), n_target_rows_synthetic=int(m.sum()), target_loc_ids=tgt, label_rule="clip(1.4·√TDD·exp(0.2ε), 5, 590)")


# ================================================================ h40 인자
def h40_argv(data_dir: Path, out_dir: Path, target: str, splits: int, threads: int, workers: int, resume=True, count=False, extra=(),
             n_grid=N_GRID, draws=DRAWS, seeds=SEEDS, tag=TAG):
    """등록 범위(6B.6)의 h40 인자. α 축·배치 축·학습기 축·α 중첩 선택은 끈다. n_grid·draws·seeds·tag 는 스모크에서만 바꾼다."""
    av = ["--part", "cpu", "--modes", "x", "--targets", f"{target}:x", "--splits", str(int(splits)), "--n-grid", str(n_grid),
          "--draws", str(int(draws)), "--seeds", str(int(seeds)), "--learners", "catboost_lo", "--alphas", "1", "--place-n-grid", "",
          "--nested-targets", "NONE:x", "--learner-targets", "NONE:x", "--data-dir", str(data_dir), "--out-dir", str(out_dir),
          "--tag", str(tag), "--threads", str(int(threads)), "--workers", str(int(workers)), "--no-summarize"]
    if resume:
        av.append("--resume")
    if count:
        av.append("--count-only")
    else:
        av.append("--allow-local")
    return av + list(extra)


# ================================================================ 점검(--count-only): 라벨 값을 쓰지 않는 구조·적합 수
def dummy_ctx(D, args, target, mode, split):
    """h40.build_ctx 와 같은 색인으로 작업 단위를 만들되 라벨 자리에 √TDD 를 넣는다(적합 수 세기 전용).
    대상 라벨의 계수 비와 오차를 계산하지 않는다(WRAPUP 7.2 (a)7). 원천·A·B 의 행 구성은 build_ctx 와 같다."""
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx(target, mode)
    src = df.iloc[src_idx]
    A_idx, B_idx = half_split_blocks(df, t_idx, split)
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    A, B = df.iloc[A_idx], df.iloc[evB]
    F = H.FEATS
    c = H.Ctx(target, mode, split, parent, src[F].values, src.s.values.astype(float), src.s.values, src.macro.values,
              A[F].values, A.s.values.astype(float), A.s.values, A.block.values, B[F].values, B.s.values.astype(float), B.s.values,
              B.block.values, min_cells_prior=args.min_cells_prior)
    c.prior = None                                  # 가짜 라벨의 사전분포는 분산이 0 이다. P3 는 E0 로 두고 세기만 한다(적합 수 불변)
    info = D.split_structure(target)[split]
    meta = dict(n_A=int(len(A_idx)), nb_A=int(A.block.nunique()), n_eval=int(len(evB)), nb_eval=int(B.block.nunique()), **comp,
                n_cells=int(len(t_idx)), n_blocks=int(df.iloc[t_idx].block.nunique()), dup_of=info["dup_of"], valid=info["valid"])
    return c, meta


def support_fraction(D, target, lo=0.5, hi=99.5):
    """WRAPUP 7.2 (a)10: 대상 채점 셀(eval_mask) 가운데 고도(dem_elev)와 √TDD(e5_sqrt_tdd)가 원천 셀의 lo–hi 백분위 밖인 비율(공변량만)."""
    df = D.df
    t_idx, _, src_idx, _ = D.source_idx(target, "x")
    tv = df.iloc[t_idx]
    tv = tv[eval_mask(tv)]
    out = dict(n_eval_cells=int(len(tv)), n_src=int(len(src_idx)), pct=[lo, hi])
    outside_any = np.zeros(len(tv), bool)
    for col in ("dem_elev", "e5_sqrt_tdd"):
        s = df.iloc[src_idx][col].values.astype(float)
        s = s[np.isfinite(s)]
        a, b = np.percentile(s, lo), np.percentile(s, hi)
        v = tv[col].values.astype(float)
        o = np.isfinite(v) & ((v < a) | (v > b))
        outside_any |= o
        out[col] = dict(src_lo=float(a), src_hi=float(b), frac_outside=float(o.mean()) if len(v) else np.nan, n_outside=int(o.sum()))
    out["frac_outside_either"] = float(outside_any.mean()) if len(tv) else np.nan
    return out


def eval_block_union(D, t_idx, info, splits):
    """사용 분할(유효 분할, 없으면 중복이 아닌 분할)의 B 채점 블록 합집합 크기(h40 TMx.nb_union 과 같은 규칙)."""
    uniq = [sp for sp in splits if info.get(sp, {}).get("dup_of", -1) < 0]
    used = [sp for sp in uniq if info.get(sp, {}).get("valid", False)] or uniq
    out = set()
    for sp in used:
        B_idx = half_split_blocks(D.df, t_idx, sp)[1]
        evB = B_idx[eval_mask(D.df.iloc[B_idx])]
        out |= set(D.df.block.values[evB].tolist())
    return len(out)


def count_spec(T: Tables, s, d: Path, out: Path, threads: int):
    """표 하나의 점검(WRAPUP 7.4 (b)4). 반환 (분할별 행 목록, 요약 dict).
    대조: 대상 셀 수 = ext_cells_v1 대상 셀 수 + v3 셀 수, 적격 표의 셀 수와 분할 구조(n_A, n_eval, nb_eval, valid, dup_of, n_src).
    L42 표는 원천이 다르므로 n_src 를 대조하지 않는다(분할 구조는 주 설정과 같아야 한다)."""
    argv = h40_argv(d, out / s["name"], s["target"], s["splits"], threads, 0, resume=False, count=True)
    args = H.parse_args(argv)
    D = H.get_data(args)
    t_idx = D.target_idx(s["target"]) if s["target"] in D.macros else np.zeros(0, int)
    exp = T.expected_counts(s)
    n_cells = int(len(t_idx))
    ok_cells = bool(np.isfinite(exp["expected_new"]) and n_cells == int(exp["expected_new"]) + int(exp["expected_v3"]))
    elr = s["elig"]
    ok_elig = bool(elr is None or int(elr["n_cells"]) == n_cells)
    units, skipped = H.enumerate_units(args, D, "cpu") if n_cells else ([], [])
    info = D.split_structure(s["target"]) if n_cells else {}
    rows = []
    ref = T.el_sp[T.el_sp.spec == s["el_name"]].set_index("split") if s["el_name"] else pd.DataFrame()
    n_mis = 0
    for sp in args.SPLITS:
        v = info.get(sp, {})
        u = [x for x in units if x[2] == sp]
        r = dict(spec=s["name"], target=s["target"], split=sp, status="run" if u else next((k["status"] for k in skipped if k["split"] == sp), "none"),
                 dup_of=v.get("dup_of", np.nan), valid=v.get("valid", np.nan), n_A=v.get("n_A", np.nan), nb_A=v.get("nb_A", np.nan),
                 n_eval=v.get("n_eval", np.nan), nb_eval=v.get("nb_eval", np.nan))
        if u:
            c, meta = dummy_ctx(D, args, s["target"], "x", sp)
            _, _, stats = H.run_ctx(c, "cpu", args, learner_axis=H.is_learner_unit(args, s["target"], "x", sp),
                                    nested=(s["target"], "x") in args.NESTED_TARGETS, dry=True)
            r.update(n_src=meta["n_src"], n_buffer_excluded=meta["n_buffer_excluded"], n_fit=int(sum(stats["n_fit"].values())),
                     n_rows=int(stats["n_rows"]), est_s=round(float(sum(stats["est_detail"].values())), 1),
                     n_fit_detail=json.dumps(stats["n_fit"], ensure_ascii=False))
        if len(ref) and sp in ref.index:
            q = ref.loc[sp]
            mis = [k for k in ("n_A", "n_eval", "nb_eval", "valid", "dup_of") if k in q and str(q[k]) != str(r.get(k))
                   and not (pd.isna(q[k]) and pd.isna(r.get(k)))]
            if "n_src" in r and s["kind"] != "source_L42" and elr is not None and not pd.isna(elr.get("n_src", np.nan)):
                if int(elr["n_src"]) != int(r["n_src"]):
                    mis.append("n_src")
            r["mismatch_vs_eligibility"] = ";".join(mis)
            n_mis += len(mis)
        rows.append(r)
    first = next((r for r in rows if "n_src" in r), {})
    summ = dict(spec=s["name"], target=s["target"], kind=s["kind"], group=s["group"], n_cells=n_cells, **exp,
                n_cells_eligibility=(int(elr["n_cells"]) if elr is not None else None), ok_cells=ok_cells, ok_eligibility=ok_elig,
                n_split_mismatch=int(n_mis), nb_union=int(eval_block_union(D, t_idx, info, args.SPLITS)) if n_cells else 0,
                nb_union_eligibility=(int(elr["nb_union"]) if elr is not None and not pd.isna(elr.get("nb_union", np.nan)) else None),
                n_units=int(len(units)), n_fit=int(sum(r.get("n_fit", 0) for r in rows)),
                est_h_1proc=round(sum(r.get("est_s", 0.0) for r in rows) / 3600, 3),
                n_src=first.get("n_src"), n_buffer_excluded=first.get("n_buffer_excluded"))
    ok_union = bool(summ["nb_union_eligibility"] is None or s["splits"] != SPLITS or summ["nb_union"] == summ["nb_union_eligibility"])
    summ["ok_union"] = ok_union
    summ["check_ok"] = bool(ok_cells and ok_elig and ok_union and n_mis == 0 and len(units) > 0)
    if s["target"] == "Tibet_LGD" and s["kind"] == "new_region" and n_cells:
        summ["tibet_support"] = support_fraction(D, s["target"])
    return rows, summ


# ================================================================ 재현 점검(WRAPUP 7.5 (c)2, 개정 14 의 키 범위)
def key_class(k):
    """키 분류(개정 14). 물리식 = P0–P3(학습기 none), ridge = V1(학습기 none, sklearn Ridge 앵커)과 ridge 학습기, CatBoost = 그 밖의 학습기.
    허용 차는 물리식 1e-9 cm, ridge·CatBoost 0.02 cm(6A.6, WRAPUP 8.4). scripts/2_evaluation/lgw_xenv_gate_i.py 와 같은 규칙이다."""
    if k[1] == "none":
        return "phys" if k[0] in PHYS_METHODS else "ridge"
    return "ridge" if k[1] == "ridge" else "catboost"


def key_tol(cls, tol_phys=GATE_TOL_PHYS, tol_ml=GATE_TOL_ML):
    return tol_phys if cls == "phys" else tol_ml


def compare_stores(sa, sb, scope="base", tol_phys=GATE_TOL_PHYS, tol_ml=GATE_TOL_ML):
    """두 BlockStore 의 공통 키(방법 축: 학습기 none·catboost_lo, α 1, 셀 무작위)를 키별 셀 가중 RMSE 로 대조한다. Δ 는 계산하지 않는다.
    scope = 'base'(6A.6 의 기준 방법 P0, P1, P2, D0, D1, R0, R1, R2. 주 판정) 또는 'all'(방법 축 전체. 병기).
    pass_phys = 물리식 키가 있고 모두 1e-9 cm 이하. pass_ml = ridge·CatBoost 키가 모두 0.02 cm 이하."""
    keys = [k for k in sa.keys if k in sb and k[1] in ("none", H.BASE_LEARNER) and str(k[2]) == "1" and k[3] == "cell"]
    if scope == "base":
        keys = [k for k in keys if k[0] in BASE_METHODS]
    by_c = {"phys": [], "ridge": [], "catboost": []}
    by_m, worst, over = {}, [], []
    for k in keys:
        c = key_class(k)
        d_ = abs(sa.rmse(k) - sb.rmse(k))
        by_c[c].append(d_)
        by_m.setdefault(k[0], []).append(d_)
        worst.append((d_, list(k)))
        if d_ > key_tol(c, tol_phys, tol_ml):
            over.append((c, k[0]))
    worst = sorted(worst, key=lambda x: -x[0])[:5]
    dm = by_c["ridge"] + by_c["catboost"]

    def mx(v):
        return float(max(v)) if v else np.nan
    return dict(scope=scope, n_keys=len(keys), n_keys_phys=len(by_c["phys"]), n_keys_ridge=len(by_c["ridge"]),
                n_keys_catboost=len(by_c["catboost"]), n_keys_ml=len(dm), max_diff_phys=mx(by_c["phys"]), max_diff_ridge=mx(by_c["ridge"]),
                max_diff_catboost=mx(by_c["catboost"]), max_diff_ml=mx(dm), median_diff_ml=float(np.median(dm)) if dm else np.nan,
                n_ml_over_tol=int(sum(1 for c, _ in over if c != "phys")), methods_ml_over_tol=";".join(sorted({m for c, m in over if c != "phys"})),
                methods_phys_over_tol=";".join(sorted({m for c, m in over if c == "phys"})),
                max_diff_by_method=json.dumps({m: float(max(v)) for m, v in sorted(by_m.items())}),
                pass_phys=bool(by_c["phys"] and max(by_c["phys"]) <= tol_phys), pass_ml=bool((not dm) or max(dm) <= tol_ml),
                blocks_equal=bool(np.array_equal(sa.blocks, sb.blocks) and np.array_equal(sa.ncell, sb.ncell)),
                worst_keys=json.dumps(worst))


def gate_status(row):
    """(c)2 의 상태 문자열. 주 판정 범위(base)의 행만 상태를 정한다. 병기 범위(all)의 행에는 '병기:' 를 붙인다."""
    st = ("ok" if (row["pass_phys"] and row["pass_ml"] and row["blocks_equal"]) else (
        "물리식 불통과(LGD 집계 중단, WRAPUP 7.5 (c)2 (1))" if not row["pass_phys"] or not row["blocks_equal"]
        else "CatBoost 불통과(교차 환경, L40 로컬 v3 기준값 필요, (c)2 (2))"))
    return st if row.get("scope") == "base" else f"병기: {st}(상태를 정하지 않는다)"


def repro_gate(a, T=None):
    """repro 표의 분할 1 조각을 LG 본 실행 조각과 대조한다. 산출 lgd_repro_gate.csv(범위 base·all 의 두 행)."""
    s_dir = a.OUT / "repro_Russia_W" / "shards"
    rows = []
    for sp in (1,):
        mine = s_dir / f"{TAG}__cpu__Russia_W__x__s{sp}_blocksse.npz"
        ref = a.LG_SHARDS / f"{a.lg_tag}__cpu__Russia_W__x__s{sp}_blocksse.npz"
        base = dict(target="Russia_W", mode="x", split=sp, mine=str(mine), ref=str(ref), status="")
        if not mine.exists() or not ref.exists():
            for scope in ("base", "all"):
                rows.append(dict(base, scope=scope, status="LGD repro 조각 없음" if not mine.exists() else "LG 본 실행 조각 없음(ZovWo 회수 전)"))
            continue
        A = load_stores(mine)[("Russia_W|x", sp)]
        B = load_stores(ref)[("Russia_W|x", sp)]
        ua = Path(str(mine)[:-len("_blocksse.npz")] + "_unit.json")
        ub = Path(str(ref)[:-len("_blocksse.npz")] + "_unit.json")
        for tag_, p in (("mine", ua), ("ref", ub)):
            if p.exists():
                u = json.loads(p.read_text())
                base[f"code_sha_{tag_}"] = u.get("code_sha", ""); base[f"cfg_common_{tag_}"] = u.get("cfg_common", "")
        for scope in ("base", "all"):
            row = dict(base)
            row.update(compare_stores(A, B, scope=scope))
            row["role"] = "주 판정(6A.6 기준 방법)" if scope == "base" else "병기(방법 축 전체, P3·R3·V1·V1r 포함)"
            row["status"] = gate_status(row)
            rows.append(row)
    df = pd.DataFrame(rows)
    a.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.OUT / "lgd_repro_gate.csv", index=False)
    cols = [c for c in ("scope", "status", "n_keys", "max_diff_phys", "max_diff_ml", "n_ml_over_tol", "methods_ml_over_tol") if c in df]
    print("[repro] " + df[cols].to_string(index=False), flush=True)
    return df


# ================================================================ 실행
def run_spec(argv):
    """h40.main 을 부른다(학습). 반환 h40.main 의 결과 dict."""
    return H.main(argv)


def plan_specs(a, T: Tables):
    """요청한 표 정의 목록과 실행하지 않는 표(부적격 변형)."""
    allspec = T.specs()
    bad = [n for n in a.SPEC_NAMES if n not in allspec]
    if bad:
        raise SystemExit(f"알 수 없는 표 이름: {bad}. 가능: {sorted(allspec)}")
    run, skip = [], []
    for n in a.SPEC_NAMES:
        s = allspec[n]
        if s["kind"].startswith("variant_L41") and (s["elig"] is None or not bool(s["elig"].get("eligible", False))):
            skip.append(dict(spec=n, reason="변형 불가(적격 표에서 부적격)" + ("; 점 추정 전용 변형" if s["kind"] == "variant_L41h" else "")))
            continue
        run.append(s)
    return allspec, run, skip


def ineligible_variants(allspec):
    """적격 표에서 부적격인 L41 변형(실행하지 않는 표). 요청한 표와 무관하게 전체 등록 표에서 정한다(manifest 의 count.skipped)."""
    return [dict(spec=n, reason="변형 불가(적격 표에서 부적격)" + ("; 점 추정 전용 변형" if s["kind"] == "variant_L41h" else ""))
            for n, s in allspec.items() if s["kind"].startswith("variant_L41") and (s["elig"] is None or not bool(s["elig"].get("eligible", False)))]


def bundle_estimates(man):
    """manifest 의 표별 점검 기록에서 묶음별 적합 수와 추정 시간을 낸다. 본 묶음 = main, l40, l39, l41, l42 가운데 same_as 가 없고 점검을
    통과한 표. 선택 묶음 = repro, v3local(필요할 때만 돌린다). 추정은 h40.BASE_FIT 환산이다(스모크 실측 비는 smoke 항목에 따로 적는다)."""
    out = {}
    specs = man.get("specs", {})
    for b, groups in (("main", DEFAULT_GROUPS), ("optional", ("repro", "v3local"))):
        sel = {k: v for k, v in specs.items() if v.get("group") in groups and "est_h_1proc" in v and not v.get("same_as") and v.get("check_ok")}
        h = float(sum(float(v.get("est_h_1proc", 0.0)) for v in sel.values()))
        out[b] = dict(groups=list(groups), n_tables=len(sel), n_fit=int(sum(int(v.get("n_fit", 0)) for v in sel.values())),
                      est_h_1proc=round(h, 2), est_h_workers2=round(h / 2, 2), tables=sorted(sel))
    out["same_as"] = sorted(k for k, v in specs.items() if v.get("same_as"))
    return out


def smoke_run(a, T: Tables, specs, tables, man, man_path, t0):
    """스모크(개정 14): 합성 대상 라벨 실행 표로 h40 을 부른다. 분할 1, n --smoke-n-grid, 추출 1, seed 1, tag lgd_smoke.
    출력은 단위별 상태, 적합 수, 벽시계, h40 추정 대비 비뿐이다(RMSE, Δ, 판정은 출력하지 않는다)."""
    SM = a.OUT / "smoke"
    SM.mkdir(parents=True, exist_ok=True)
    gi = SM / ".gitignore"
    if not gi.exists():
        gi.write_text("# 스모크 산출(합성 대상 라벨, 약관 미확인 행 포함 실행 표). 커밋하지 않는다(메타 lgd_smoke_meta.json 만 남긴다).\n*\n"
                      "!.gitignore\n!lgd_smoke_meta.json\n")
    out = dict(created=time.strftime("%Y-%m-%d %H:%M"), argv=a.ARGV, threads=a.threads, workers=a.workers, n_grid=a.smoke_n_grid, splits=1,
               draws=1, seeds=1, tag=SMOKE_TAG, label="합성 대상 라벨(WRAPUP 7.2 (a)7). RMSE 는 어디에도 쓰지 않는다", specs={})
    for s in specs:
        ms = man["specs"].get(s["name"], {})
        if not ms.get("check_ok", False) or ms.get("table_sha256") != tables[s["name"]][1]["table_sha256"]:
            out["specs"][s["name"]] = dict(status="점검 통과 기록 없음(--count-only 를 먼저 돌린다)")
            print(f"  [skip] {s['name']}: 점검 통과 기록 없음", flush=True)
            continue
        la_now = load_avg()
        if np.isfinite(la_now) and la_now > a.load_run_max:
            out["specs"][s["name"]] = dict(status=f"load average {la_now:.1f} > {a.load_run_max}")
            break
        d_src = tables[s["name"]][0]
        d_sm = SM / "run_tables" / s["name"]
        rec = smoke_table(T, s, d_src, d_sm)
        av = h40_argv(d_sm, SM / s["name"], s["target"], 1, a.threads, a.workers, resume=False, n_grid=a.smoke_n_grid, draws=1, seeds=1,
                      tag=SMOKE_TAG)
        args = H.parse_args(av)
        D = H.get_data(args)
        t_idx = D.target_idx(s["target"]) if s["target"] in D.macros else np.zeros(0, int)
        ids = sorted(int(x) for x in D.df.loc_id.values[t_idx])
        if ids != rec["target_loc_ids"]:
            out["specs"][s["name"]] = dict(status="중단: h40 대상 셀과 합성 라벨 셀이 다르다", n_h40=len(ids), n_syn=len(rec["target_loc_ids"]))
            print(f"  [stop] {s['name']}: h40 대상 셀 {len(ids)} 과 합성 라벨 셀 {len(rec['target_loc_ids'])} 이 다르다. 실행하지 않는다", flush=True)
            continue
        del D
        t1 = time.time()
        print(f"[smoke] {s['name']} · 대상 {s['target']} · 합성 라벨 셀 {rec['n_target_rows_synthetic']} · h40 {' '.join(av)}", flush=True)
        try:
            res = run_spec(av)
            err = ""
        except Exception as e:                                                                        # noqa: BLE001
            res, err = {}, repr(e)[:300]
        units = []
        for p in sorted((SM / s["name"] / "shards").glob(f"{SMOKE_TAG}__cpu__{s['target']}__x__s*_unit.json")):
            u = json.loads(p.read_text())
            est = float(sum((u.get("est_detail") or {}).values()))
            units.append(dict(unit=p.name[:-len("_unit.json")], status=u.get("status"), n_fit_total=u.get("n_fit_total"), n_fail=len(u.get("fail") or {}),
                              elapsed_s=u.get("elapsed_s"), est_s=round(est, 2), ratio_actual_to_est=(round(float(u["elapsed_s"]) / est, 3)
                                                                                                     if est > 0 else None),
                              n_A=u.get("n_A"), n_eval=u.get("n_eval"), nb_eval=u.get("nb_eval"), n_src=u.get("n_src"),
                              n_stored=u.get("n_stored"), threads=u.get("threads"), code_sha=u.get("code_sha")))
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        out["specs"][s["name"]] = dict(status=("ok" if units and all(x["status"] == "ok" for x in units) and not err else "실패"), error=err,
                                        n_fail=int(res.get("n_fail", 0) or 0) if res else None, wall_s=round(time.time() - t1, 1), units=units,
                                        synthetic=dict(n_rows=rec["n_rows"], n_target_rows=rec["n_target_rows_synthetic"], rule=rec["label_rule"]),
                                        h40_argv=av, max_rss_children_kb=int(ru.ru_maxrss), load_start=la_now)
        for x in units:
            print(f"  [smoke] {x['unit']} · 상태 {x['status']} · 적합 {x['n_fit_total']} · {x['elapsed_s']} s · h40 추정 {x['est_s']} s · "
                  f"비 {x['ratio_actual_to_est']}", flush=True)
    rat = [x["ratio_actual_to_est"] for v in out["specs"].values() for x in v.get("units", []) if x.get("ratio_actual_to_est")]
    out["ratio_median"] = float(np.median(rat)) if rat else None
    bund = (man.get("count") or {}).get("bundles") or bundle_estimates(man)
    if rat:
        out["main_bundle_local_est"] = dict(n_fit=bund["main"]["n_fit"], est_h_1proc_basefit=bund["main"]["est_h_1proc"],
                                            est_h_1proc_local=round(bund["main"]["est_h_1proc"] * out["ratio_median"], 2),
                                            est_h_workers2_local=round(bund["main"]["est_h_1proc"] * out["ratio_median"] / 2, 2),
                                            basis="h40.BASE_FIT 환산 × 스모크 단위의 실측/추정 비 중앙값(워커 1, 스레드 --threads)")
    out["elapsed_s"] = round(time.time() - t0, 1)
    (SM / "lgd_smoke_meta.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    man["smoke"] = dict(meta=str((SM / "lgd_smoke_meta.json").relative_to(ROOT)) if (SM / "lgd_smoke_meta.json").is_relative_to(ROOT) else
                        str(SM / "lgd_smoke_meta.json"), created=out["created"], ratio_median=out["ratio_median"],
                        main_bundle_local_est=out.get("main_bundle_local_est"),
                        status={k: v.get("status") for k, v in out["specs"].items()})
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str))
    n_bad = sum(1 for v in out["specs"].values() if v.get("status") != "ok")
    print(f"[smoke] 표 {len(out['specs'])} · 실패·중단 {n_bad} · 실측/추정 비 중앙값 {out['ratio_median']} · {out['elapsed_s']:.0f}s → "
          f"{SM / 'lgd_smoke_meta.json'}", flush=True)
    return dict(smoke=out, n_fail=n_bad)


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    nice = lower_priority()
    la = load_avg()
    will_train = not a.count_only
    if will_train and not a.allow_local:
        raise SystemExit("[거부] LGD 의 학습 실행(본 실행, 재현 점검, 스모크)은 --allow-local 이 있어야 한다(개정 13: 로컬 서버, 본 실행은 "
                         "사용자 확인 뒤). 구조 점검은 --count-only 로 한다")
    if will_train and np.isfinite(la) and la > a.load_start_max:
        raise SystemExit(f"[거부] 1분 load average {la:.1f} > {a.load_start_max}. 시작하지 않는다(WRAPUP 8.5)")
    reg = check_registered() if not a.skip_hash_check else {}
    T = Tables()
    allspec, specs, skipped = plan_specs(a, T)
    a.OUT.mkdir(parents=True, exist_ok=True)
    (a.OUT / "logs").mkdir(exist_ok=True)
    print(f"[plan] 표 {len(specs)} · 실행하지 않는 표 {len(skipped)} · load {la:.1f} · nice {nice} · 스레드 {a.threads} · 워커 {a.workers} · "
          f"{'점검' if a.count_only else ('스모크(합성 대상 라벨)' if a.smoke else '학습')}", flush=True)
    man_path = a.OUT / "lgd_run_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man.setdefault("specs", {})
    man.update(stage="LGD 6B.6(h51)", plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B(개정 10, 13, 14), WRAPUP 7절(커밋 316714c)",
               script="scripts/3_deep_learning/h51_lgd_run.py", script_sha256=sha256(Path(__file__)), h40_code_sha=H.code_sha(),
               registered_inputs=reg or man.get("registered_inputs", {}), versions=versions(), platform="local",
               h40_fixed_args=dict(n_grid=N_GRID, splits=SPLITS, draws=DRAWS, seeds=SEEDS, learners="catboost_lo", alphas="1",
                                   place_n_grid="", nested_targets="NONE:x", learner_targets="NONE:x", tag=TAG, threads=THREADS,
                                   workers_max=WORKERS_MAX),
               rules=dict(call="(가) 대상별 자료 디렉터리 + --data-dir. 실행 표 = fidelity_base_v3.csv, 토양 = e5_soil_tdd_v4.csv 연결",
                          subregion_map="자료 디렉터리에 두지 않는다(k-means 계산값은 macro 대상에 쓰이지 않는다). 설정 해시가 LG 와 다르다: "
                                        "LGD 조각끼리만 대조한다(WRAPUP 7.4 (b)2)",
                          source="주 설정: v3 F4_direct − 대상 − 100 km 버퍼. L42: + 다른 새 지역(Tibet, NAtlantic, Russia_C) 대상 직접 라벨 셀",
                          same_as="실행 표가 같은 대상의 주 설정 표와 같으면(바이트 동일, 또는 문자열 열이 같고 수치 열의 최대 절대 차 ≤ 1e-9) "
                                  "다시 적합하지 않고 주 설정 조각을 쓴다(개정 14)",
                          count_only="작업 단위의 라벨 자리에 √TDD 를 넣은 dry 실행으로 적합 수를 센다(WRAPUP 7.2 (a)7). h40 Data 적재 때 셀별 z 가 "
                                     "메모리에서 계산되지만 집계하거나 출력하지 않는다",
                          skipped="적격 표에서 부적격인 L41 변형은 실행하지 않는다('변형 불가')",
                          repro_gate="키 분류 key_class(물리식 P0–P3 1e-9 cm, ridge = V1·ridge 학습기 0.02 cm, CatBoost 0.02 cm). 주 판정 = 6A.6 기준 "
                                     "방법, 병기 = 방법 축 전체(개정 14)",
                          load_check="시작 전 64, 표 사이 96(h40 안의 작업 단위 사이에서는 확인하지 않는다. 개정 14)",
                          smoke="대상 셀의 alt_cm 을 합성 라벨로 바꾼 실행 표로 경로와 시간만 확인한다(WRAPUP 7.2 (a)7)",
                          test_flags="--skip-hash-check, --h40-extra 는 LGD_TEST=1 일 때만 받는다(개정 14)"))
    tables = {}
    for s in specs:
        d, rec = write_table(T, s, a.OUT)
        tables[s["name"]] = (d, rec)
    # 같은 대상의 주 설정과 같은 실행 표
    for s in specs:
        base = s.get("base")
        rec = tables[s["name"]][1]
        same = None
        if base and base in allspec and s["kind"].startswith("variant_L41") and not a.rerun_identical:
            bd = a.OUT / "run_tables" / base / "fidelity_base_v3.csv"
            if base not in tables:
                write_table(T, allspec[base], a.OUT)
            eq, info = same_table(bd, tables[s["name"]][0] / "fidelity_base_v3.csv")
            rec["same_as_basis"] = info
            if eq:
                same = base
        rec["same_as"] = same
    rows, summs = [], []
    if a.count_only:
        for s in specs:
            d, rec = tables[s["name"]]
            r_, sm = count_spec(T, s, d, a.OUT, a.threads)
            if s["kind"] in ("repro", "v3local"):
                sm["v3_text_check"] = v3_only_text_check(T, d / "fidelity_base_v3.csv")
                sm["check_ok"] = bool(sm["check_ok"] and sm["v3_text_check"]["ok"])
            sm.update(table_sha256=rec["table_sha256"], same_as=rec.get("same_as"), same_as_basis=(rec.get("same_as_basis") or {}).get("basis", ""),
                      n_new_written=rec["n_new"], n_drop_v3=rec["n_drop_v3"], drop_v3=s["drop_v3"], point_only=s["point_only"], note=s["note"])
            rows += r_; summs.append(sm)
            print(f"  [count] {s['name']:30s} 셀 {sm['n_cells']:4d}(기대 {sm['expected_new']}+{sm['expected_v3']}, 적격 표 "
                  f"{sm['n_cells_eligibility']}) 분할 불일치 {sm['n_split_mismatch']} 단위 {sm['n_units']} 적합 {sm['n_fit']:,} "
                  f"추정 {sm['est_h_1proc']:.2f} h(1프로세스) same_as {sm['same_as']} → {'통과' if sm['check_ok'] else '불일치'}", flush=True)
        cdf = pd.DataFrame(rows)
        sdf = pd.DataFrame(summs)
        cdf.to_csv(a.OUT / "lgd_count.csv", index=False)
        sdf.drop(columns=[c for c in ("tibet_support", "v3_text_check") if c in sdf], errors="ignore").to_csv(a.OUT / "lgd_count_summary.csv", index=False)
        for sm in summs:
            man["specs"][sm["spec"]] = dict({k: v for k, v in sm.items() if k not in ("spec",)}, table=tables[sm["spec"]][1],
                                             tm_name=tm_name(allspec[sm["spec"]]), base=allspec[sm["spec"]].get("base"))
        bund = bundle_estimates(man)
        man["count"] = dict(created=time.strftime("%Y-%m-%d %H:%M"), argv=a.ARGV, n_specs_this_run=int(len(sdf)),
                            n_check_fail_this_run=int((~sdf.check_ok).sum()) if len(sdf) else 0,
                            n_check_fail_all=int(sum(1 for v in man["specs"].values() if "check_ok" in v and not v["check_ok"])),
                            bundles=bund, est_basis="h40.BASE_FIT(CatBoost 4스레드 1배 행 약 0.5 s, LG 스모크 기준)의 행 수 환산. 로컬 실측이 아니다"
                                                    "(스모크의 실측 비는 manifest 의 smoke 항목)",
                            skipped=ineligible_variants(allspec), load_avg=la, threads=a.threads)
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str))
        bm, bo = bund["main"], bund["optional"]
        print(f"[count] 이번 점검 표 {len(sdf)} · 불일치 {man['count']['n_check_fail_this_run']} · {time.time() - t0:.0f}s → {a.OUT}/lgd_count*.csv", flush=True)
        print(f"[count] 본 묶음(main·l40·l39·l41·l42, same_as 제외): 표 {bm['n_tables']} · 적합 {bm['n_fit']:,} · 추정 {bm['est_h_1proc']:.2f} h(1프로세스)"
              f" → 워커 2개 {bm['est_h_workers2']:.2f} h. 같은 표 재사용 {len(bund['same_as'])}", flush=True)
        print(f"[count] 선택 묶음(repro·v3local, 필요할 때만): 표 {bo['n_tables']} · 적합 {bo['n_fit']:,} · 추정 {bo['est_h_1proc']:.2f} h(1프로세스)",
              flush=True)
        return dict(count=cdf, summary=sdf, manifest=man)

    if a.smoke:
        return smoke_run(a, T, specs, tables, man, man_path, t0)

    # 학습: 점검 통과 기록이 있는 표만 실행한다(WRAPUP 7.4 (b)4)
    done, failed, blocked = [], [], []
    extra = a.h40_extra.split() if a.h40_extra else []
    for s in specs:
        rec = tables[s["name"]][1]
        ms = man["specs"].get(s["name"], {})
        if not ms.get("check_ok", False) or ms.get("table_sha256") != rec["table_sha256"]:
            blocked.append(dict(spec=s["name"], reason="--count-only 점검 기록이 없거나 불일치(또는 실행 표 해시가 점검 때와 다르다)"))
            print(f"  [skip] {s['name']}: 점검 통과 기록 없음. --count-only 를 먼저 돌린다", flush=True)
            continue
        if rec.get("same_as"):
            print(f"  [same] {s['name']}: 주 설정 {rec['same_as']} 와 같은 실행 표. 주 설정 조각을 쓴다", flush=True)
            man["specs"][s["name"]]["same_as"] = rec["same_as"]
            continue
        la_now = load_avg()
        if np.isfinite(la_now) and la_now > a.load_run_max:
            blocked.append(dict(spec=s["name"], reason=f"load average {la_now:.1f} > {a.load_run_max}"))
            print(f"  [stop] load average {la_now:.1f} > {a.load_run_max}. 남은 표를 시작하지 않는다", flush=True)
            break
        d = tables[s["name"]][0]
        av = h40_argv(d, a.OUT / s["name"], s["target"], s["splits"], a.threads, a.workers, resume=a.resume, extra=extra)
        t1 = time.time()
        print(f"[run] {s['name']} · 대상 {s['target']} · 분할 {s['splits']} · h40 {' '.join(av)}", flush=True)
        try:
            res = run_spec(av)
            nf = int(res.get("n_fail", 0) or 0)
            (failed if nf else done).append(dict(spec=s["name"], n_fail=nf, n_executed=len(res.get("executed", [])),
                                                 n_resumed=len(res.get("resumed", [])), wall_s=round(time.time() - t1, 1)))
        except SystemExit as e:
            failed.append(dict(spec=s["name"], error=str(e)[:300]))
        except Exception as e:                                                                        # noqa: BLE001
            failed.append(dict(spec=s["name"], error=repr(e)[:300]))
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        man["specs"][s["name"]].setdefault("runs", []).append(dict(time=time.strftime("%Y-%m-%d %H:%M"), wall_s=round(time.time() - t1, 1),
                                                                   threads=a.threads, workers=a.workers, platform="local",
                                                                   max_rss_children_kb=int(ru.ru_maxrss), load_start=la_now, argv=a.ARGV,
                                                                   h40_argv=av, test_flags=dict(skip_hash_check=bool(a.skip_hash_check),
                                                                                                 h40_extra=a.h40_extra)))
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str))
    man["last_run"] = dict(time=time.strftime("%Y-%m-%d %H:%M"), argv=a.ARGV, done=done, failed=failed, blocked=blocked, skipped=skipped,
                           elapsed_s=round(time.time() - t0, 1))
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str))
    if a.repro_check:
        repro_gate(a, T)
    print(f"[done] 완료 {len(done)} · 실패 {len(failed)} · 막힘 {len(blocked)} · {time.time() - t0:.0f}s", flush=True)
    return dict(done=done, failed=failed, blocked=blocked, n_fail=len(failed))


if __name__ == "__main__":
    res = main()
    sys.exit(1 if res.get("n_fail") else 0)
