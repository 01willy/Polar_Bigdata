"""H53 · 조건부 격자 σ(nflow 정규화기)의 재적합. 계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 개정 4 '조건부 격자 σ 규약',
docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 3.1 A7 와 6.3, docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.6 (d), 6.9, 6.10.

지위와 시점
  LGU-B2 판정이 '전이'(네 조건 모두 만족)일 때만 필요한 산출이다. 그때 지도 패널 (d)의 정규화기가 nflow σ 가 되고, h49_transfer_map 은
  --sigma-file 이 없으면 중단한다(h49 990–993행). h44–h46 은 적합 모델을 저장하지 않으므로 LGU 실험 B 와 같은 설정으로 다시 적합한다.
  이 스크립트는 LGU 판정 표(lgu_b_tests.csv 등)를 읽지 않는다. 실행 여부는 J3 에서 B2 를 기록한 운영자가 정하고, 본 실행은 --allow-run 이
  있을 때만 한다. B2 가 '전이'가 아니면 실행하지 않는다(지도 패널 (d)는 const).

규약(결과 열람 전 고정)
  원천·설정: LGU 실험 B(h45)의 nflow 적합과 같다. 대상 Lena, 모드 x, 원천 = h40.Data.source_idx("Lena", "x")(레나 셀과 경계 100 km 버퍼
    제외), 입력 x25, 목표 z = log y − log(E0^학습·s), 표준화·중앙값 대체는 학습 행렬에서만, 행 가중 min(1, 100/블록 셀 수), 블록 단위 검증,
    epochs 100(조기 종료 인내 12, 최소 15), 아핀 로그 척도 절단 ±2, 위치 고정 없음. σ = clip((q_0.95 − q_0.05)/2, 0.02, 2.0).
    설정 객체는 h45.parse_args 의 기본값으로 만들고(seed 만 바꾼다) 적합은 h45._fit_one(lgu_common.fit_gen, nflow_quantiles)을 그대로 부른다.
  seed: 실험 B 의 seed 0, 1, 2 가운데 두 개(기본 0, 1). 두 seed 의 σ 를 셀마다 평균한다. 두 seed 가 모두 유효해야 파일을 쓴다.
  보정 셀: 지도 패널 (d)의 보정 집단(WRAPUP 6.6 (d), 원천에서 로그 비가 정의되는 셀 30개 이상인 macro 지역. h49.cal_groups 로 정한다).
    지역 k 의 σ 는 원천에서 k 를 뺀 학습 집합의 적합(표본 밖)이다. 실험 B 의 보정 집단(20개 이상)이 이 집단을 포함하므로 학습 집합은 실험 B 의
    지역 하나 제외 학습 집합과 같다. 셀 순서는 h49 real_ctx 의 원천 순서(h49 calibrate 의 te 순서)와 같고, 실행마다 y·s 대조로 확인한다.
  격자 셀: 원천 전체(full) 학습 집합의 적합으로 레나 격자(lena_grid_x25_v1.csv.gz)의 모든 셀에 σ 를 준다. 결측 공변량은 full 학습 행렬의
    중앙값으로 채운다(실험 B 의 대상 셀과 같은 처리). 회색 셀도 값을 두며 h49 가 그리지 않는다.
  적합 실패: 실험 B 와 같은 규칙(lgu_common.fit_failed: 검증 손실 비유한, 분위 비유한, 표본 밖 중앙값 |로그| > 3 셀 1 % 이상)을 실험 B 와 같은
    예측 셀(지역 k 의 보정 셀, full 은 레나 라벨 셀 3,037개의 99점 분위)에 적용한다. 격자 σ 가 비유한이면 실패다. 한 학습 집합이라도 실패한
    seed 는 모든 학습 집합에서 뺀다(h45 541–543행과 같다). 유효 seed 가 2개 미만이면 npz 를 쓰지 않고 종료 코드 5 로 끝난다.

산출(h49 --sigma-file 형식. h49 994–996행이 읽는 키는 cal, grid_cell_id, grid_sigma 셋이다)
  <out>.npz  cal = {지역: σ 배열(보정 셀, h49 원천 순서)}(object, allow_pickle 로 읽는다), grid_cell_id(str), grid_sigma(float64).
             추가 키(pickle 없음): format, seeds, groups, groups_lgu, sigma_clip, cal_region, cal_loc, cal_sigma, cal_sigma_seeds,
             grid_sigma_seeds, lena_loc, lena_sigma, lena_sigma_seeds.
  <out>_meta.json  코드 해시, 설정(h45 fit_cfg)과 실험 B 조각 설정의 대조, seed, 입력 해시, GPU, 적합 기록, 벽시계, h49 형식 점검, 재현 대조.
  기본 <out> = data/processed/map_lena/lena_sigma_nflow_v1. 실패하면 <out>_failed_meta.json 만 쓴다.

실행 보호와 자원(docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 4.2, 4.3)
  본 실행은 --allow-run 이 있어야 한다. LGU 계획서에 개정 4 가 있고 커밋되어 있어야 한다. 1분 load average 64 이하에서만 시작한다.
  GPU 후보는 8, 9, 7, 6, 5 순서이고 0–4 번은 받지 않는다. 메모리 사용 50 MiB 이하이고 계산 프로세스가 없는 첫 번호 하나를 쓴다.
  CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<번호> 를 torch 적재 전에 둔다. 적합마다 그 GPU 에 다른 계산 프로세스가 생겼는지
  확인하고, 생겼으면 진행 중인 적합만 끝내고 GPU 를 놓는다(종료 코드 4, 파일 없음). CPU 스레드는 --threads(4 이하), nice 10 이다.
  --dry-run 과 --synthetic 은 GPU 를 쓰지 않는다(CUDA_VISIBLE_DEVICES='').
  실험 B 조각(lgub__fit__Lena__x__nflow_unit.json)에서는 설정·시간·코드·입력 해시 키만 읽는다(status, valid_seeds, fails, E0 는 읽지 않는다).
  본 실행은 설정(seed 제외)이나 입력 해시가 실험 B 조각과 다르면 거부한다(--allow-mismatch 로 진행하면 메타에 적는다).
  본 실행 뒤 재현 대조: 실험 B 의 σ 조각(lgub__fit__Lena__x__nflow_sigma.npz)과 같은 seed 의 σ 차이(최대 절대 차)를 메타에 적는다.
  B2 판정 뒤에만 도는 경로라 이 대조는 열람 규칙에 걸리지 않는다. --no-lgu-compare 로 끈다.

종료 코드: 0 완료, 2 인자·등록·설정 거부, 3 빈 GPU 없음 또는 load 초과, 4 실행 중 GPU 상실, 5 유효 seed 부족.

실행(저장소 루트)
  계획만(적합 없음, CPU): nice -n 10 python3 scripts/2_evaluation/h53_grid_sigma_nflow.py --dry-run --threads 2
  합성 시험(CPU):         CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/2_evaluation/h53_grid_sigma_nflow.py --synthetic --out <폴더>/s --threads 2
  본 실행(B2 '전이' 뒤): nice -n 10 python3 scripts/2_evaluation/h53_grid_sigma_nflow.py --allow-run --gpus 8,9,7,6,5 --threads 4
  지도:                   POST_SIGMA_FILE=data/processed/map_lena/lena_sigma_nflow_v1.npz scripts/local/run_post_results.sh map

확인하지 못한 것(작성 시점)
  실제 자료의 GPU 적합은 하지 않았다(결과 전 작성 규칙). 실행한 것은 py_compile, pyflakes, --dry-run(실제 자료, 적합 없음), --synthetic(CPU,
  epochs 3)과 단위 시험이다. GPU 재적합이 실험 B 의 σ 와 같은 값을 내는지(GPU 비결정성)는 본 실행의 재현 대조로만 확인된다.
"""
from __future__ import annotations

import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_threads(argv, default):
    for i, v in enumerate(argv):
        if v == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return default


_REAL_RUN = "--allow-run" in sys.argv and "--dry-run" not in sys.argv and "--synthetic" not in sys.argv
if __name__ == "__main__":                                              # numpy 적재 전에 스레드 수를 정한다(셸 값보다 --threads 가 우선)
    for _v in THREAD_VARS:
        os.environ[_v] = str(_peek_threads(sys.argv, "4" if _REAL_RUN else "2"))
    if not _REAL_RUN:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                          # 계획·합성 시험은 GPU 를 쓰지 않는다
else:                                                                    # 시험의 import: 부른 쪽의 설정을 따르고 GPU 는 기본으로 숨긴다
    for _v in THREAD_VARS:
        os.environ.setdefault(_v, "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                           # nvidia-smi 번호와 CUDA 번호를 맞춘다(h45 와 같다)

import argparse                                                                                         # noqa: E402
import importlib.util                                                                                   # noqa: E402
import json                                                                                             # noqa: E402
import resource                                                                                         # noqa: E402
import subprocess                                                                                       # noqa: E402
import time                                                                                             # noqa: E402
import warnings                                                                                         # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import polar.lgu_common as LC                                                                          # noqa: E402

# ---------------------------------------------------------------- 고정값(LGU 개정 4, WRAPUP 6.6 (d)·6.9·6.10, 실행 계획 4.2·4.3)
TARGET, MODE, NORM = "Lena", "x", "nflow"
SEEDS_EXP_B = (0, 1, 2)                         # 실험 B 의 seed(h45 --seeds 3 기본값)
SEEDS_DEFAULT = (0, 1)
N_SEEDS = 2
GPU_ORDER = (8, 9, 7, 6, 5)                     # 실행 계획 4.2: GPU 8 먼저, 그다음 9, 7, 6, 5
GPU_FORBIDDEN = (0, 1, 2, 3, 4)                 # 다른 사용자 GPU(쓰지 않는다)
GPU_MEM_MAX_MIB = 50                            # 점유 판정: 메모리 50 MiB 초과 또는 계산 프로세스 있음(4.3 공통 규칙)
LOAD_MAX = 64.0                                 # 새 작업 시작 전 1분 load average 상한(4.3)
MAX_THREADS = 4
NICE = 10
SYN_EPOCHS = 3                                  # 합성 시험 전용(본 실행은 실험 B 의 epochs 100)
FORMAT = "h53_grid_sigma_v1"
OUT_DEFAULT = "data/processed/map_lena/lena_sigma_nflow_v1"
GRID_DEFAULT = "data/processed/map_lena/lena_grid_x25_v1.csv.gz"
GRID_META_DEFAULT = "data/processed/map_lena/lena_grid_x25_v1_meta.json"
PLAN_LGU = ROOT / "docs" / "EXPERIMENT_PLAN_LGU_2026-09-29.md"
PREREG_MARKS = ("개정 4(", "조건부 격자 σ 규약")
REF_KEYS = ("cfg", "cfg_hash", "elapsed_s", "device", "threads", "n_fit", "K_groups", "groups", "N_cal", "N_target", "n_src", "code_sha",
            "code_sha_common", "code_sha_h40", "tag", "inputs")      # 실험 B 조각에서 읽는 키(결과가 아닌 설정·시간·해시)
PER_FIT_FALLBACK_S = 30.0                       # 실험 B 조각이 없을 때의 적합당 시간(LGU 8절, T4 추정)
EXIT_REFUSE, EXIT_NO_GPU, EXIT_GPU_LOST, EXIT_FIT_FAIL = 2, 3, 4, 5


def log(*a):
    print(*a, flush=True)


def refuse(msg, code=EXIT_REFUSE):
    log(msg)
    raise SystemExit(code)


# ---------------------------------------------------------------- 모듈 적재(h45, h49 는 고치지 않고 읽는다)
def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def h45():
    return _load("h45_normalized_conformal", "scripts/2_evaluation/h45_normalized_conformal.py")


def h49():
    """h49 는 import 할 때 CUDA_VISIBLE_DEVICES 를 '' 로 둔다(h49 모듈 머리). 본 실행은 GPU 를 고정하기 전에 이 함수를 먼저 부른다."""
    return _load("h49_transfer_map", "scripts/2_evaluation/h49_transfer_map.py")


# ---------------------------------------------------------------- GPU 점유 판정(4.3 공통 규칙, run_lgx_gpu_local.sh 의 gpu_free 와 같은 기준)
def _smi(args) -> str:
    return subprocess.check_output(["nvidia-smi"] + list(args), text=True, timeout=60)


def parse_gpu_info(text) -> dict:
    """nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 의 해석. {번호: dict(uuid, used)}."""
    out = {}
    for line in str(text).strip().splitlines():
        p = [v.strip() for v in line.split(",")]
        if len(p) >= 3:
            try:
                out[int(p[0])] = dict(uuid=p[1], used=float(p[2]))
            except ValueError:
                continue
    return out


def parse_gpu_apps(text) -> list:
    """nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 의 해석. [(PID, UUID)]."""
    out = []
    for line in str(text).strip().splitlines():
        p = [v.strip() for v in line.split(",")]
        if len(p) >= 2:
            try:
                out.append((int(p[0]), p[1]))
            except ValueError:
                continue
    return out


def query_gpus() -> tuple[dict, dict]:
    """(정보 {번호: dict(uuid, used)}, 계산 프로세스 {번호: [PID]})."""
    info = parse_gpu_info(_smi(["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]))
    apps = parse_gpu_apps(_smi(["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]))
    by = {v["uuid"]: g for g, v in info.items()}
    procs: dict = {}
    for pid, u in apps:
        if u in by:
            procs.setdefault(by[u], []).append(int(pid))
    return info, procs


def gpu_candidates(txt) -> list:
    """--gpus 의 후보 목록(순서 유지). 0–4 번과 허용 목록(8, 9, 7, 6, 5) 밖의 번호는 거부한다."""
    try:
        c = [int(v) for v in str(txt).split(",") if v.strip() != ""]
    except ValueError:
        refuse(f"[거부] --gpus 는 쉼표로 나눈 정수 목록이다: {txt!r}")
    bad = [g for g in c if g in GPU_FORBIDDEN]
    if bad:
        refuse(f"[거부] GPU {bad} 는 다른 사용자 GPU(0–4)라 쓰지 않는다")
    far = [g for g in c if g not in GPU_ORDER]
    if far:
        refuse(f"[거부] GPU {far} 는 허용 목록 {list(GPU_ORDER)} 밖이다")
    if not c:
        refuse("[거부] GPU 후보가 없다")
    return c


def screen_gpus(cands, info, procs, mem_max=GPU_MEM_MAX_MIB) -> tuple[list, list]:
    """빈 GPU(메모리 mem_max MiB 이하, 계산 프로세스 없음) 목록(후보 순서)과 판정표 [dict(gpu, used, procs, free, why)]."""
    free, table = [], []
    for g in cands:
        v = info.get(int(g))
        pr = sorted(procs.get(int(g), []))
        if v is None:
            why = "nvidia-smi 목록에 없음"
        elif float(v["used"]) > float(mem_max):
            why = f"메모리 {v['used']:.0f} MiB > {mem_max} MiB"
        elif pr:
            why = f"계산 프로세스 {pr}"
        else:
            why = ""
        table.append(dict(gpu=int(g), used=None if v is None else float(v["used"]), procs=pr, free=not why, why=why))
        if not why:
            free.append(int(g))
    return free, table


def pick_gpu(cands, wait_min=0.0, poll_s=60.0) -> tuple[int, dict, list]:
    """빈 GPU 가운데 후보 순서의 첫 번호. 없으면 wait_min 분까지 poll_s 간격으로 다시 본다. 반환 (번호, 정보, 판정표 기록)."""
    t_end = time.time() + 60.0 * float(wait_min)
    history = []
    while True:
        try:
            info, procs = query_gpus()
        except Exception as e:                                          # noqa: BLE001
            refuse(f"[중단] nvidia-smi 를 부를 수 없다({type(e).__name__}: {str(e)[:120]})", EXIT_NO_GPU)
        free, table = screen_gpus(cands, info, procs)
        history.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), table=table))
        for r in table:
            log(f"  [GPU] {r['gpu']}: {'비어 있음' if r['free'] else r['why']}")
        if free:
            g = free[0]
            return g, info[g], history[-5:]
        if time.time() >= t_end:
            refuse(f"[중단] 후보 GPU {cands} 가 모두 사용 중이다(50 MiB·계산 프로세스 0 규칙). 나중에 다시 실행한다", EXIT_NO_GPU)
        time.sleep(float(poll_s))


def bind_gpu(g: int, info: dict, threads: int) -> dict:
    """torch 적재 전에 CUDA_VISIBLE_DEVICES 를 한 번호로 고정하고 장치 수 1 을 확인한다."""
    if "torch" in sys.modules:
        refuse("[중단] torch 가 이미 적재되어 GPU 번호를 고정할 수 없다")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(g))
    LC.set_thread_env(threads)
    nth = LC.set_torch_threads(threads)                                   # 여기서 torch 를 적재한다(h45 _worker_init 과 같은 순서)
    import torch
    if torch.cuda.device_count() != 1:
        refuse(f"[중단] CUDA_VISIBLE_DEVICES={g} 에서 보이는 장치 수가 {torch.cuda.device_count()} 다")
    return dict(index=int(g), uuid=info.get("uuid"), mem_before_mib=info.get("used"), name=torch.cuda.get_device_name(0),
                torch=torch.__version__, torch_threads=int(nth))


class GpuLost(RuntimeError):
    pass


def gpu_guard(gpu: dict) -> dict:
    """적합 전 확인: 우리 GPU 에 다른 계산 프로세스가 있으면 GpuLost(진행 중인 적합은 이미 끝났다). 반환은 계산 프로세스 표."""
    try:
        _, procs = query_gpus()
    except Exception as e:                                              # noqa: BLE001
        raise GpuLost(f"nvidia-smi 확인 실패({type(e).__name__})") from None
    me = os.getpid()
    foreign = [p for p in procs.get(int(gpu["index"]), []) if p != me]
    if foreign:
        raise GpuLost(f"GPU {gpu['index']} 에 다른 계산 프로세스 {sorted(foreign)}")
    return procs


def pid_check(gpu: dict) -> dict:
    """첫 적합 뒤(CUDA 문맥이 생긴 뒤) 우리 PID 가 고정한 GPU 에만 있는지 확인한다. 다른 GPU 에 있으면 GpuLost."""
    procs = gpu_guard(gpu)
    me = os.getpid()
    where = sorted(g for g, ps in procs.items() if me in ps)
    if where and where != [int(gpu["index"])]:
        raise GpuLost(f"우리 프로세스가 GPU {where} 에 있다(고정한 번호 {gpu['index']} 와 다르다)")
    return dict(our_pid=me, our_gpus=where, ok=(where == [int(gpu["index"])]), note="" if where else "nvidia-smi 에 우리 PID 가 보이지 않는다")


# ---------------------------------------------------------------- 등록·자원·보호
def prereg_state() -> dict:
    """LGU 계획서에 개정 4(조건부 격자 σ 규약)가 있고 작업 트리 변경 없이 커밋되어 있는지."""
    try:
        txt = PLAN_LGU.read_text(encoding="utf-8")
    except OSError:
        return dict(present=False, committed=False)
    present = all(m in txt for m in PREREG_MARKS)
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PLAN_LGU)], cwd=ROOT, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(present=bool(present), committed=bool(present and r.returncode == 0), head=head)


def read_ref_unit(path: Path) -> dict | None:
    """실험 B 레나 nflow fit 조각의 unit.json 에서 REF_KEYS 만 돌려준다(다른 키는 읽은 뒤 버리고 출력하지 않는다)."""
    p = Path(path)
    if not p.exists():
        return None
    u = json.loads(p.read_text(encoding="utf-8"))
    return {k: u.get(k) for k in REF_KEYS}


def compare_ref(M45, a45, ref: dict | None) -> dict:
    """h45.fit_cfg(설정)·code_shas·input_shas 를 실험 B 조각과 대조한다(seed 목록은 뺀다)."""
    mine = M45.fit_cfg(a45, NORM)
    out = dict(cfg_mine=mine, cfg_hash_mine=LC.cfg_hash(mine), code_mine=M45.code_shas(), inputs_mine=M45.input_shas(a45))
    if ref is None:
        out.update(ref_found=False, cfg_equal_except_seeds=False, seeds_subset=False, inputs_equal=False, code_equal=False)
        return out
    rc = dict(ref.get("cfg") or {})
    A = {k: v for k, v in json.loads(json.dumps(mine, default=str)).items() if k != "seeds"}
    B = {k: v for k, v in rc.items() if k != "seeds"}
    diff = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
    code_diff = sorted(k for k in ("code_sha", "code_sha_common", "code_sha_h40") if out["code_mine"].get(k) != ref.get(k))
    inp_ref = ref.get("inputs") or {}
    inp_diff = sorted(k for k in set(out["inputs_mine"]) | set(inp_ref) if out["inputs_mine"].get(k) != inp_ref.get(k))
    out.update(ref_found=True, cfg_ref=rc, cfg_hash_ref=ref.get("cfg_hash"), cfg_hash_ref_ok=(LC.cfg_hash(rc) == ref.get("cfg_hash")),
               cfg_diff_keys=diff, cfg_equal_except_seeds=not diff, seeds_ref=rc.get("seeds"),
               seeds_subset=set(mine["seeds"]) <= set(rc.get("seeds") or []), code_diff=code_diff, code_equal=not code_diff,
               inputs_diff=inp_diff, inputs_equal=not inp_diff, ref_elapsed_s=ref.get("elapsed_s"), ref_n_fit=ref.get("n_fit"),
               ref_device=ref.get("device"), ref_threads=ref.get("threads"), ref_groups=ref.get("groups"))
    return out


def out_paths(stem) -> dict:
    s = str(stem)
    if s.endswith(".npz"):
        s = s[:-4]
    p = Path(s) if os.path.isabs(s) else ROOT / s
    return dict(npz=Path(f"{p}.npz"), meta=Path(f"{p}_meta.json"), failed=Path(f"{p}_failed_meta.json"), stem=p)


def check_out_dir(M49, path: Path, synthetic: bool) -> Path:
    """출력 보호: h49.check_out(실험 산출 폴더와 results/ 거부). 합성 시험은 본 산출 폴더(data/processed/map_lena) 안에 쓰지 않는다."""
    forbid = (PROC / "map_lena",) if synthetic else ()
    return M49.check_out(path.parent, forbid=forbid)


# ---------------------------------------------------------------- 자료
def make_a45(M45, seeds, threads, synthetic: bool):
    """실험 B 의 설정 객체(h45.parse_args 기본값). 바꾸는 것은 대상(레나), 정규화기(nflow), seed 목록, 합성 시험의 epochs 뿐이다."""
    argv = ["--normalizers", NORM, "--targets", TARGET, "--seeds", str(len(SEEDS_EXP_B)), "--threads", str(int(threads))]
    if synthetic:
        argv += ["--epochs", str(SYN_EPOCHS)]
    a = M45.parse_args(argv)
    a.SEEDS = [int(v) for v in seeds]                                   # fit_cfg 의 seeds 가 이 실행의 seed 를 가리키게 한다
    return a


def synthetic_inputs(M45, seed=0) -> tuple:
    """합성 원천·레나 라벨·격자(실제 자료를 읽지 않는다). h40.Data 와 같은 속성을 가진 객체와 격자 표를 돌려준다.
    보정 집단 경계를 시험하도록 Russia_E 는 24셀(20 이상 30 미만), Tibet 는 10셀이다. Canada 한 셀은 y = 0(로그 비 정의 안 됨)이다."""
    H = M45.H
    rng = np.random.RandomState(seed)
    regs = (("Lena", 8, 10, 72.0, 126.0, 3.2), ("Alaska", 8, 12, 66.0, -150.0, 4.8), ("Canada", 6, 10, 62.0, -120.0, 4.2),
            ("Russia_W", 4, 10, 67.0, 70.0, 3.6), ("Russia_E", 3, 8, 68.0, 160.0, 4.0), ("Tibet", 2, 5, 34.0, 93.0, 9.0))
    rows = []
    for mac, nb, nc, lat0, lon0, E in regs:
        for b in range(nb):
            for _ in range(nc):
                rows.append(dict(macro=mac, block=f"{mac[:2]}{b:03d}", lat=lat0 + (b // 4) * 0.5 + rng.rand() * 0.02,
                                 lon=lon0 + (b % 4) * 0.5 + rng.rand() * 0.02, E=E))
    df = pd.DataFrame(rows)
    n = len(df)
    for c in H.FEATS:
        df[c] = rng.randn(n)
    df["cci_valid"] = 1.0
    df["e5_sqrt_tdd"] = rng.uniform(20, 40, n)
    df["cci_alt"] = rng.uniform(40, 90, n)
    df.loc[rng.rand(n) < 0.05, "cci_alt"] = np.nan
    df["e5_sqrt_tdd_soil"] = df.e5_sqrt_tdd + 1.0
    df["p4_ku"] = rng.uniform(40, 90, n)
    df["p2_edaphic"] = rng.uniform(40, 90, n)
    sig = 0.08 + 0.3 / (1.0 + np.exp(-2.0 * df[H.FEATS[0]].values))          # 이분산 로그 비
    df[H.TARGET] = df.E * df.e5_sqrt_tdd * np.exp(sig * rng.randn(n))
    df.loc[df.index[df.macro == "Canada"][0], H.TARGET] = 0.0
    df["s"] = df.e5_sqrt_tdd.values.astype(float)
    df["y"] = df[H.TARGET].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["z"] = np.log(df.y) - np.log(df.s)
    df["loc_id"] = np.arange(n) + 1000
    df["sub"] = ""
    D = H.Data.__new__(H.Data)
    D.df = df
    D.args = H.parse_args(["--splits", "5", "--threads", "1"])
    D.subs = pd.DataFrame(dict(subregion=[], parent=[]))
    D.macros = set(df.macro.unique()); D.sub_parent = {}; D._src = {}; D._split = {}
    D.sub_src = "synthetic"; D.sub_kmeans_diff = 0
    ng = 150
    g = pd.DataFrame({c: rng.randn(ng) * 1.3 for c in H.FEATS})
    g["cci_valid"] = 1.0
    g["e5_sqrt_tdd"] = rng.uniform(18, 42, ng)
    g["cci_alt"] = rng.uniform(40, 90, ng)
    for c in (H.FEATS[1], "sg_clay_5_15"):
        g.loc[rng.rand(ng) < 0.08, c] = np.nan                               # 결측 공변량(학습 행렬 중앙값으로 채운다)
    g.insert(0, "cell_id", [f"{7900 + i // 15}_{4300 + i % 15}" for i in range(ng)])
    g["ky"] = 7900 + np.arange(ng) // 15; g["kx"] = 4300 + np.arange(ng) % 15
    g["lat"] = 71.6 + (np.arange(ng) // 15) * 0.009; g["lon"] = 124.0 + (np.arange(ng) % 15) * 0.03
    g["e5_sqrt_tdd_soil"] = g.e5_sqrt_tdd * 0.97
    g["p4_ku"] = 3.0 * g.e5_sqrt_tdd; g["p2_edaphic"] = 3.5 * g.e5_sqrt_tdd
    gray = rng.rand(ng) < 0.1
    g["gray"] = gray.astype(int); g["land"] = 1; g["gray_reason"] = np.where(gray, "water", "")
    return D, g


def read_grid(path: Path, feats: list) -> pd.DataFrame:
    g = pd.read_csv(path, dtype={"cell_id": str}, low_memory=False)
    miss = [c for c in ["cell_id"] + list(feats) if c not in g.columns]
    if miss:
        refuse(f"[중단] 격자 표에 열이 없다: {miss}")
    if g.cell_id.duplicated().any():
        refuse("[중단] 격자 cell_id 가 중복된다(h49 는 cell_id 로 σ 를 붙인다)")
    return g


def h49_alignment(M49, D, grid_df, info, threads) -> tuple:
    """h49 의 원천 순서·보정 집단(real_ctx, cal_groups)과 실험 B 의 보정 셀(target_info.cal_idx)을 y·s 로 대조한다(적합 없음).
    반환 (대조 기록, h49 문맥). 문맥은 h49 형식 점검에 다시 쓴다."""
    mctx = M49.real_ctx(None, int(threads), D=D, grid_df=grid_df)
    src = mctx.src
    G = M49.cal_groups(src)                                               # 셀 30개 이상(WRAPUP 6.6 (d), h49 MIN_CAL_CELLS)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(src.y) & np.isfinite(src.s) & (src.y > 0) & (src.s > 0)
    y = D.df.y.values.astype(float); s = D.df.s.values.astype(float)
    per, not_lgu = {}, []
    for k in G:
        te = ok & (src.macro.astype(str) == k)
        if k not in info["G"]:
            not_lgu.append(k); per[k] = dict(n=int(te.sum()), aligned=False); continue
        idx = info["cal_idx"][info["cal_gi"] == info["G"].index(k)]
        eq = bool(len(idx) == int(te.sum()) and np.array_equal(src.y[te], y[idx]) and np.array_equal(src.s[te], s[idx]))
        per[k] = dict(n=int(te.sum()), aligned=eq)
    gid_same = bool(np.array_equal(mctx.grid_df.cell_id.astype(str).values, grid_df.cell_id.astype(str).values))
    return dict(groups_map=list(G), groups_lgu=list(info["G"]), min_cells_map=int(M49.MIN_CAL_CELLS), min_cells_lgu=int(LC.MIN_CAL_CELLS),
                per_group=per, not_in_lgu=not_lgu, grid_ids_same=gid_same, n_src_h49=int(len(src)),
                aligned=bool(G and not not_lgu and all(v["aligned"] for v in per.values()) and gid_same)), mctx


def build_context(a, M45, M49, seeds, synthetic: bool) -> dict:
    t0 = time.time()
    a45 = make_a45(M45, seeds, a.threads, synthetic)
    if synthetic:
        D, grid_df = synthetic_inputs(M45)
        grid_info = dict(path="synthetic", n=int(len(grid_df)))
    else:
        gp = Path(a.grid) if os.path.isabs(a.grid) else ROOT / a.grid
        if not gp.exists():
            refuse(f"[중단] 격자 파일이 없다: {gp}")
        D = M45.get_D(a45)
        grid_df = read_grid(gp, M45.FEATS)
        gm = Path(a.grid_meta) if os.path.isabs(a.grid_meta) else ROOT / a.grid_meta
        grid_info = dict(grid=M49.file_info(gp), grid_meta=M49.file_info(gm), n=int(len(grid_df)),
                         qa_passed=bool(json.loads(gm.read_text()).get("qa", {}).get("passed")) if gm.exists() else None)
    bi = M45.get_bi(D)
    info = M45.target_info(a45, D, TARGET)
    align, mctx = h49_alignment(M49, D, grid_df, info, a.threads)
    keep = set(align["groups_map"]) | {"full"}
    sets = [st for st in M45.train_sets(D, info) if st[0] in keep]
    Xg = grid_df[M45.FEATS].values.astype(float)
    ref_path = M45.fit_paths(a45, TARGET, NORM)["unit"]
    ref = None if synthetic else read_ref_unit(ref_path)
    return dict(a45=a45, D=D, bi=bi, info=info, grid_df=grid_df, Xg=Xg, grid_info=grid_info, align=align, mctx=mctx, sets=sets,
                ref_path=ref_path, ref=ref, cmp=compare_ref(M45, a45, ref) if not synthetic else dict(ref_found=False, synthetic=True),
                load_s=round(time.time() - t0, 1))


def plan_table(C, seeds) -> dict:
    sets = [dict(train_set=n, n_train=int(len(tr)), n_pred=int(len(pr)), pred=("레나 라벨 셀(실패 판정)과 격자" if n == "full" else f"{n} 보정 셀"))
            for n, tr, pr, _ in C["sets"]]
    n_fit = len(C["sets"]) * len(seeds)
    ref = C.get("cmp", {})
    per_fit = (float(ref["ref_elapsed_s"]) / float(ref["ref_n_fit"])) if ref.get("ref_found") and ref.get("ref_n_fit") else PER_FIT_FALLBACK_S
    return dict(sets=sets, seeds=list(seeds), n_fit=int(n_fit), n_grid=int(len(C["grid_df"])), n_lena=int(len(C["info"]["t_idx"])),
                n_src=int(len(C["info"]["src_idx"])), per_fit_s=round(per_fit, 2), est_fit_s=round(per_fit * n_fit, 1),
                per_fit_basis=("실험 B 레나 nflow 조각의 elapsed_s / n_fit" if ref.get("ref_found") else "LGU 8절의 T4 추정 30 s"))


def print_plan(C, P, a, synthetic):
    al = C["align"]
    log(f"[h53] 모드 {'합성' if synthetic else '실제 자료'} · {'계획만(적합 없음)' if a.dry_run else '적합'}")
    qa = C["grid_info"].get("qa_passed")
    log(f"[자료] 원천 {P['n_src']:,}셀(Lena x, 100 km 버퍼) · 레나 라벨 {P['n_lena']:,}셀 · 격자 {P['n_grid']:,}셀"
        + ("" if synthetic else f"(조립 QA {'통과' if qa else '미통과·없음'})") + f" · 적재 {C['load_s']} s")
    cnt = ", ".join(f"{k} {v['n']:,}" for k, v in al["per_group"].items())
    log(f"[보정 집단] 지도(h49, 셀 {al['min_cells_map']}개 이상): {cnt or '없음'} · 실험 B(셀 {al['min_cells_lgu']}개 이상): {al['groups_lgu']}")
    log(f"[h49 정렬] 원천 순서 y·s 대조 {'일치' if al['aligned'] else '불일치'} · 격자 cell_id {'일치' if al['grid_ids_same'] else '불일치'}"
        + (f" · 실험 B 집단 밖 {al['not_in_lgu']}" if al["not_in_lgu"] else ""))
    for r in P["sets"]:
        extra = f", 격자 {P['n_grid']:,}" if r["train_set"] == "full" else ""
        log(f"  [학습 집합] {r['train_set']:>9s}: 학습 {r['n_train']:,}행 · 예측 {r['n_pred']:,}셀({r['pred']}{extra})")
    a45 = C["a45"]
    log(f"[적합 계획] nflow {len(P['sets'])} 학습 집합 × seed {P['seeds']} = {P['n_fit']}건 · epochs {a45.epochs} · 절단 ±{a45.clamp:g} · "
        f"σ 절단 [{a45.sigma_lo:g}, {a45.sigma_hi:g}] · 행 가중 상한 {a45.cap_cells} · 검증 block")
    cm = C["cmp"]
    if cm.get("ref_found"):
        log(f"[설정 대조] 실험 B 조각 {Path(C['ref_path']).name}: cfg(seed 제외) {'일치' if cm['cfg_equal_except_seeds'] else '불일치 ' + str(cm['cfg_diff_keys'])}"
            f" · 원 seed {cm['seeds_ref']} ⊇ {P['seeds']}: {'예' if cm['seeds_subset'] else '아니오'} · 원 cfg_hash {cm['cfg_hash_ref']}"
            f"(재계산 {'일치' if cm['cfg_hash_ref_ok'] else '불일치'})")
        log(f"[코드 대조] h45·lgu_common·h40: {'같음' if cm['code_equal'] else '다름 ' + str(cm['code_diff'])} · "
            f"[입력 대조] {'같음' if cm['inputs_equal'] else '다름 ' + str(cm['inputs_diff'])}")
        log(f"[시간 추정] 실험 B 조각 {cm['ref_elapsed_s']} s / {cm['ref_n_fit']}건(GPU {cm['ref_device']}, 스레드 {cm['ref_threads']}) = "
            f"{P['per_fit_s']} s/건 → {P['n_fit']}건 약 {P['est_fit_s']} s(약 {P['est_fit_s'] / 3600:.3f} GPU-h) + 격자 분위와 자료 적재")
    elif not synthetic:
        log(f"[설정 대조] 실험 B 조각 없음({C['ref_path']}). 시간 추정은 {P['per_fit_s']} s/건 → 약 {P['est_fit_s']} s")


# ---------------------------------------------------------------- 적합(h45 _fit_unit 498–543행과 같은 순서와 규칙)
def fit_all(C, M45, seeds, gpu: dict | None = None) -> dict:
    a45 = C["a45"]; D = C["D"]; df = D.df; bi = C["bi"]; info = C["info"]
    X_all = df[M45.FEATS].values.astype(float)
    y = df.y.values.astype(float); s = df.s.values.astype(float)
    lo, hi = float(a45.sigma_lo), float(a45.sigma_hi)
    S = len(seeds)
    cal = {n: np.full((S, len(pr)), np.nan) for n, _, pr, _ in C["sets"] if n != "full"}
    lena = np.full((S, len(info["t_idx"])), np.nan)
    grid = np.full((S, len(C["Xg"])), np.nan)
    ok_seed = np.ones(S, bool)
    rows = []
    pidc: dict = {}
    t_fit = time.time()
    for name, tr, pred, cpos in C["sets"]:
        full = name == "full"
        E0t = LC.ls_E(y[tr], s[tr])
        z = np.log(y[tr]) - np.log(E0t * s[tr])
        cols = bi.cols(tr)
        w = M45.row_w(cols, a45.cap_cells)
        Xtr, Xp = X_all[tr], X_all[pred]
        st = LC.prep_stats(Xtr)
        Xtr_z = LC.prep_apply(Xtr, st); Xp_z = LC.prep_apply(Xp, st)
        Xg_z = LC.prep_apply(C["Xg"], st) if full else None
        for si, seed in enumerate(seeds):
            if gpu is not None:
                gpu_guard(gpu)
            t1 = time.time()
            rec = dict(train_set=name, seed=int(seed), n_train=int(len(tr)), n_pred=int(len(pred)), E0_train=float(E0t))
            h, Q5, Qg, cross, flag = None, None, None, None, ""
            try:
                Q5, Qg, cross, h, flag = M45._fit_one(a45, NORM, name, full, Xtr, Xtr_z, z, cols, w, Xp, Xp_z, int(seed), df, tr, pred, TARGET)
                failed, reason = LC.fit_failed(h, Qg if full else Q5)
            except Exception as e:                                          # noqa: BLE001  한 건의 예외는 그 적합만 실패로 둔다(h45 와 같다)
                if LC.is_env_error(e):
                    raise
                failed, reason = True, f"{type(e).__name__}: {e}"[:300]
            sig, clip = M45.sigma_from_q5(Q5, lo, hi) if Q5 is not None else (None, np.nan)
            if not failed and sig is not None and not np.all(np.isfinite(sig)):
                failed, reason = True, "σ 비유한"
            if full and not failed:
                Q5g = LC.nflow_quantiles(h, Xg_z, LC.Q5)
                gsig, gclip = M45.sigma_from_q5(Q5g, lo, hi)
                gfail, greason = LC.fit_failed(None, Q5g)                      # 격자의 중앙값 규칙은 진단으로만 적는다(판정은 실험 B 의 예측 셀)
                rec.update(grid_clip_frac=float(gclip), grid_median_rule=(greason if gfail else "통과"))
                if not np.all(np.isfinite(gsig)):
                    failed, reason = True, "격자 σ 비유한"
                else:
                    grid[si] = gsig
            if failed:
                ok_seed[si] = False
            elif full:
                lena[si] = sig
            else:
                cal[name][si] = sig
            rec.update(n_val=int(h.n_val) if h is not None else 0, val_loss=float(h.val_loss) if h is not None else np.nan,
                       epochs_run=int(h.epochs_run) if h is not None else 0, clip_frac=float(clip),
                       cross_frac=float(cross["frac_cells"]) if cross else np.nan, failed=bool(failed), fail_reason=str(reason),
                       flag=(h.flag if h is not None else "") + (";" if h is not None and flag else "") + flag,
                       sec=round(time.time() - t1, 2), device=(h.device if h is not None else ""))
            rows.append(rec)
            log(f"  [fit] {name:>9s} seed {seed}: {rec['sec']} s · epochs {rec['epochs_run']} · 장치 {rec['device']}"
                + (f" · 실패({reason})" if failed else ""))
            if gpu is not None and h is not None and not str(h.device).startswith("cuda"):
                refuse(f"[중단] 본 실행의 적합 장치가 {h.device} 다(GPU 고정 실패)")
            if gpu is not None and h is not None and not pidc:
                pidc = pid_check(gpu)
            del h
    for si in range(S):                                                    # 한 학습 집합이라도 실패한 seed 는 전부 뺀다(h45 541–543행)
        if not ok_seed[si]:
            lena[si] = np.nan; grid[si] = np.nan
            for n in cal:
                cal[n][si] = np.nan
    valid = [int(seeds[si]) for si in range(S) if ok_seed[si]]
    return dict(seeds=[int(v) for v in seeds], valid=valid, cal=cal, lena=lena, grid=grid, rows=rows, fit_s=round(time.time() - t_fit, 1),
                pid_check=pidc)


def write_npz(path: Path, C, res) -> dict:
    """h49 형식 npz. cal 은 {지역: σ 평균}(h49 994–995행이 z['cal'].item() 으로 읽는다)."""
    V = [res["seeds"].index(v) for v in res["valid"]]
    G = list(C["align"]["groups_map"])
    info = C["info"]; df_loc = C["D"].df.loc_id.values
    cal_mean = {k: np.mean(res["cal"][k][V], 0).astype(float) for k in G}
    cal_idx = {k: info["cal_idx"][info["cal_gi"] == info["G"].index(k)] for k in G}
    grid_mean = np.mean(res["grid"][V], 0).astype(float)
    lena_mean = np.mean(res["lena"][V], 0).astype(float)
    a45 = C["a45"]
    LC.atomic_npz(path, cal=cal_mean, grid_cell_id=C["grid_df"].cell_id.astype(str).values.astype(str), grid_sigma=grid_mean,
                  format=np.array(FORMAT), seeds=np.array(res["valid"], np.int64), groups=np.array(G, dtype=str),
                  groups_lgu=np.array(info["G"], dtype=str), sigma_clip=np.array([a45.sigma_lo, a45.sigma_hi], float),
                  cal_region=np.concatenate([np.full(len(cal_mean[k]), k) for k in G]).astype(str) if G else np.zeros(0, str),
                  cal_loc=np.concatenate([df_loc[cal_idx[k]] for k in G]) if G else np.zeros(0, np.int64),
                  cal_sigma=np.concatenate([cal_mean[k] for k in G]) if G else np.zeros(0),
                  cal_sigma_seeds=np.concatenate([res["cal"][k][V] for k in G], 1) if G else np.zeros((len(V), 0)),
                  grid_sigma_seeds=res["grid"][V], lena_loc=df_loc[info["t_idx"]], lena_sigma=lena_mean, lena_sigma_seeds=res["lena"][V])

    def q(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        return [float(x) for x in np.percentile(v, [5, 50, 95])] if len(v) else []
    lo, hi = float(a45.sigma_lo), float(a45.sigma_hi)
    return dict(n_cal={k: int(len(v)) for k, v in cal_mean.items()}, n_grid=int(len(grid_mean)),
                sigma_q05_q50_q95=dict(grid=q(grid_mean), lena=q(lena_mean), **{f"cal_{k}": q(v) for k, v in cal_mean.items()}),
                clip_frac_grid=float(np.mean((grid_mean <= lo + 1e-12) | (grid_mean >= hi - 1e-12))))


def h49_schema_check(M49, mctx, npz_path: Path, groups) -> dict:
    """h49 가 읽는 식(994–996행)으로 npz 를 다시 읽고 h49.calibrate(P0, nflow σ)와 h49.interval 을 부른다(적합 없음).
    h49 에는 σ 파일 적재 함수가 따로 없어 적재 두 줄만 같은 식으로 둔다. q 값은 적지 않는다(h49 본 실행의 산출)."""
    z = np.load(npz_path, allow_pickle=True)
    sigma = dict(cal={str(k): np.asarray(v, float) for k, v in z["cal"].item().items()},
                 grid=pd.Series(np.asarray(z["grid_sigma"], float), index=np.asarray(z["grid_cell_id"]).astype(str)))
    F = M49.Fitter(threads=1)
    cal = M49.calibrate("P0", mctx.src, F, sigma_cal=sigma["cal"])
    sg = sigma["grid"].reindex(mctx.grid_df.cell_id.astype(str)).values
    (p0,), _ = M49.fit_predict("P0", mctx.src, M49.empty_rows(mctx.grid.X.shape[1]), [mctx.grid], F)
    _, _, wid = M49.interval(p0, cal["q"], sg)
    shown = (mctx.grid_df.gray.values == 0) if "gray" in mctx.grid_df else np.ones(len(sg), bool)
    s_ok = np.isfinite(mctx.grid.s)
    res = dict(normalizer=cal["normalizer"], groups=list(cal["groups"]), groups_match=list(cal["groups"]) == list(groups),
               q_finite=bool(np.isfinite(cal["q"])), n_grid=int(len(sg)), n_sigma_missing=int(np.isnan(sg).sum()),
               n_shown=int(shown.sum()), n_shown_s_finite=int((shown & s_ok).sum()),
               n_shown_width_finite=int(np.isfinite(wid[shown & s_ok]).sum()), fits=int(F.n_fit))
    res["passed"] = bool(res["normalizer"] == "nflow" and res["groups_match"] and res["q_finite"] and res["n_sigma_missing"] == 0
                         and res["n_shown_width_finite"] == res["n_shown_s_finite"])
    return res


def h49_consumer_run(M49, D, grid_df, npz_path: Path, out_dir: Path, threads) -> dict:
    """합성 시험 전용: h49.run 을 LGU-B2 '전이' 합성 판정 표로 끝까지 돌려 σ 파일을 소비하게 한다."""
    ctx = M49.real_ctx(None, int(threads), D=D, grid_df=grid_df)
    meta = M49.run(ctx, M49.synthetic_tables("b2"), out_dir, int(threads), dict(synthetic=True, note="h53 합성 소비 점검. 실제 결과가 아니다"),
                   sigma_file=str(npz_path), strict=True)
    pred = pd.read_csv(out_dir / "lena_pred_v1.csv.gz", dtype={"cell_id": str})
    cal = meta["calibration"]
    shown = pred.gray.values == 0 if "gray" in pred else np.ones(len(pred), bool)
    w = pred.width90_a.values[shown]
    return dict(passed=bool(cal["normalizer"] == "nflow" and np.all(np.isfinite(w))), normalizer=cal["normalizer"], groups=list(cal["groups"]),
                n_shown=int(shown.sum()), n_width_finite=int(np.isfinite(w).sum()), out_dir=str(out_dir))


def lgu_repro(M45, C, res) -> dict:
    """본 실행 전용: 실험 B 의 σ 조각과 같은 seed 의 σ 차(레나 라벨 셀, 지도 보정 집단 셀). 값은 메타에만 적는다."""
    p = M45.fit_paths(C["a45"], TARGET, NORM)["sigma"]
    if not p.exists():
        return dict(status="조각 없음", path=str(p))
    with np.load(p, allow_pickle=False) as z:
        seeds_ref = [int(v) for v in z["seeds"]]; valid_ref = [int(v) for v in z["valid_seeds"]]
        tgt_loc, tgt_sig = z["tgt_loc"], z["tgt_sigma"]
        cal_loc, cal_reg, cal_sig = z["cal_loc"], z["cal_region"].astype(str), z["cal_sigma"]
    info = C["info"]; df_loc = C["D"].df.loc_id.values
    out = dict(status="대조", path=str(p), per_seed={})
    for si, sd in enumerate(res["seeds"]):
        if sd not in res["valid"] or sd not in valid_ref:
            out["per_seed"][str(sd)] = "비교 불가(한쪽에서 유효 seed 아님)"
            continue
        r = seeds_ref.index(sd)
        d = dict(lena_loc_same=bool(np.array_equal(tgt_loc, df_loc[info["t_idx"]])),
                 lena_max_abs=float(np.nanmax(np.abs(tgt_sig[r] - res["lena"][si]))))
        for k in C["align"]["groups_map"]:
            m = cal_reg == k
            idx = info["cal_idx"][info["cal_gi"] == info["G"].index(k)]
            d[f"cal_{k}_loc_same"] = bool(np.array_equal(cal_loc[m], df_loc[idx]))
            d[f"cal_{k}_max_abs"] = float(np.nanmax(np.abs(cal_sig[r][m] - res["cal"][k][si])))
        out["per_seed"][str(sd)] = d
    return out


# ---------------------------------------------------------------- 인자와 실행
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H53 조건부 격자 σ(nflow, LGU 실험 B 설정의 재적합)")
    ap.add_argument("--allow-run", action="store_true", help="본 실행(GPU 재적합)을 허용한다. LGU-B2 '전이' 기록 뒤에만 쓴다")
    ap.add_argument("--dry-run", action="store_true", help="계획(학습 집합, 크기, 적합 수, 시간 추정, GPU 상태, 대조)만 출력한다")
    ap.add_argument("--synthetic", action="store_true", help="합성 입력으로 CPU 적합(epochs 3)과 h49 소비 점검을 한다(--out 필요)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS_DEFAULT)), help="실험 B 의 seed 0, 1, 2 가운데 두 개(기본 0,1)")
    ap.add_argument("--gpus", default=",".join(map(str, GPU_ORDER)), help="GPU 후보 순서(8, 9, 7, 6, 5 가운데). 0–4 는 거부한다")
    ap.add_argument("--gpu-wait-min", type=float, default=0.0, help="빈 GPU 가 없을 때 기다리는 분(60 s 간격 확인). 기본 0")
    ap.add_argument("--threads", type=int, default=None, help="CPU 스레드(BLAS, torch). 본 실행 기본 4, 그 밖 2, 상한 4")
    ap.add_argument("--out", default=None, help=f"산출 경로 줄기(.npz 와 _meta.json). 기본 {OUT_DEFAULT}")
    ap.add_argument("--grid", default=GRID_DEFAULT)
    ap.add_argument("--grid-meta", default=GRID_META_DEFAULT)
    ap.add_argument("--overwrite", action="store_true", help="기존 npz 를 바꾼다(메타에 적는다)")
    ap.add_argument("--allow-mismatch", action="store_true", help="실험 B 조각과 설정·입력 해시가 달라도 본 실행을 한다(메타에 적는다)")
    ap.add_argument("--no-lgu-compare", action="store_true", help="본 실행 뒤 실험 B σ 조각과의 재현 대조를 하지 않는다")
    a = ap.parse_args(argv)
    real = a.allow_run and not a.dry_run and not a.synthetic
    if a.threads is None:
        a.threads = 4 if real else 2
    return a


def _seeds(txt) -> list:
    try:
        v = [int(x) for x in str(txt).split(",") if x.strip() != ""]
    except ValueError:
        refuse(f"[거부] --seeds 는 정수 목록이다: {txt!r}")
    if len(v) != N_SEEDS or len(set(v)) != N_SEEDS or not set(v) <= set(SEEDS_EXP_B):
        refuse(f"[거부] seed 는 실험 B 의 seed {list(SEEDS_EXP_B)} 가운데 서로 다른 {N_SEEDS}개다(LGU 개정 4): {v}")
    return v


def _write_json(path: Path, obj, M49):
    LC.atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=M49._jsonable))


def main(argv=None) -> int:
    t_start = time.time()
    a = parse_args(argv)
    synthetic, dry = bool(a.synthetic), bool(a.dry_run)
    real = bool(a.allow_run and not dry and not synthetic)
    if a.threads < 1 or a.threads > MAX_THREADS:
        refuse(f"[거부] 스레드 {a.threads} 는 1–{MAX_THREADS} 밖이다")
    if not (dry or synthetic or a.allow_run):
        refuse("[거부] 본 실행은 --allow-run 으로만 한다(LGU-B2 '전이' 기록 뒤). 지금은 --dry-run 또는 --synthetic 으로 점검한다")
    seeds = _seeds(a.seeds)
    cands = gpu_candidates(a.gpus)
    try:
        if os.nice(0) < NICE:
            os.nice(NICE - os.nice(0))
    except OSError:
        pass
    if not real and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                          # 계획·합성은 GPU 를 쓰지 않는다
    M45, M49 = h45(), h49()                                               # h49 import 가 CUDA_VISIBLE_DEVICES 를 비우므로 GPU 고정보다 먼저 읽는다
    warnings.filterwarnings("ignore")
    if synthetic and not dry and not a.out:
        refuse("[거부] --synthetic 은 --out(본 산출 폴더 밖)을 준다")
    op = out_paths(a.out or OUT_DEFAULT)
    if not dry:
        check_out_dir(M49, op["npz"], synthetic)
    pre = prereg_state()
    if real:
        if not pre["committed"]:
            refuse("[거부] LGU 계획서 개정 4(조건부 격자 σ 규약)가 없거나 커밋되지 않았다(결과 전 등록 확인)")
        if op["npz"].exists() and not a.overwrite:
            refuse(f"[거부] {op['npz']} 가 있다. 바꾸려면 --overwrite 를 준다")
        la = os.getloadavg()[0]
        if la > LOAD_MAX:
            refuse(f"[중단] 1분 load average {la:.1f} > {LOAD_MAX:g}. 뒤로 미룬다", EXIT_NO_GPU)

    C = build_context(a, M45, M49, seeds, synthetic)
    P = plan_table(C, seeds)
    print_plan(C, P, a, synthetic)
    al, cm = C["align"], C["cmp"]
    log(f"[등록] LGU 개정 4 {'있음' if pre['present'] else '없음'} · 커밋 {'예' if pre['committed'] else '아니오'}(HEAD {pre.get('head', '')})")
    out_state = "있음" if op["npz"].exists() else "없음"
    log(f"[출력] {op['npz']}({out_state})")
    if dry:
        if not synthetic:
            try:
                info, procs = query_gpus()
                free, table = screen_gpus(cands, info, procs)
                for r in table:
                    log(f"  [GPU] {r['gpu']}: {'비어 있음' if r['free'] else r['why']}")
                log(f"[GPU] 지금 쓸 수 있는 후보: {free[:1] or '없음'}(본 실행이 시작할 때 다시 확인한다)")
            except Exception as e:                                          # noqa: BLE001
                log(f"[GPU] nvidia-smi 확인 실패({type(e).__name__})")
        problems = []
        if not al["aligned"]:
            problems.append("h49 정렬 불일치")
        if not synthetic and not (cm.get("cfg_equal_except_seeds") and cm.get("seeds_subset")):
            problems.append("실험 B 설정 불일치 또는 조각 없음")
        if not synthetic and not cm.get("inputs_equal"):
            problems.append("입력 해시 불일치")
        if not synthetic and C["grid_info"].get("qa_passed") is not True:
            problems.append("격자 조립 QA 미통과 또는 메타 없음")
        log(f"[점검] {'문제 없음' if not problems else '; '.join(problems)}")
        return 0 if not problems else EXIT_REFUSE

    if not al["aligned"]:
        refuse("[중단] h49 의 보정 집단·원천 순서가 실험 B 의 보정 셀과 맞지 않는다(σ 배열을 h49 에 붙일 수 없다)")
    if real and C["grid_info"].get("qa_passed") is not True:
        refuse("[거부] 격자 조립 QA(WRAPUP 6.4)를 통과한 격자 메타가 없다(h49 와 같은 조건)")
    if real and not (cm.get("cfg_equal_except_seeds") and cm.get("seeds_subset") and cm.get("inputs_equal")):
        if not a.allow_mismatch:
            refuse("[거부] 실험 B 조각과 설정(seed 제외)·seed·입력 해시가 맞지 않거나 조각이 없다. 원인을 확인하거나 --allow-mismatch 로 진행한다")
        log("[경고] --allow-mismatch: 실험 B 조각과 다른 조건으로 진행한다(메타에 적는다)")
    if real and not cm.get("code_equal"):
        log(f"[경고] 코드 해시가 실험 B 실행 때와 다르다: {cm.get('code_diff')}(재현 대조로 확인한다)")

    gpu, gpu_hist = None, []
    if real:
        g, ginfo, gpu_hist = pick_gpu(cands, a.gpu_wait_min)
        gpu = bind_gpu(g, ginfo, a.threads)
        log(f"[GPU] {g} 사용({gpu['name']}, 사용 전 메모리 {gpu['mem_before_mib']:.0f} MiB)")
    else:
        LC.set_thread_env(a.threads)
        LC.set_torch_threads(a.threads)
        import torch
        if torch.cuda.device_count() != 0:
            refuse("[중단] 합성 시험에서 CUDA 장치가 보인다(CUDA_VISIBLE_DEVICES='' 가 필요하다)")

    base_meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), script="scripts/2_evaluation/h53_grid_sigma_nflow.py",
                     script_sha256=M49.sha256_file(Path(__file__)), mode="synthetic" if synthetic else "real",
                     plan=["docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 개정 4 조건부 격자 σ 규약", "docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.6 (d), 6.9, 6.10",
                           "docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 3.1 A7, 6.3"],
                     run_condition="LGU-B2 '전이'(J3 기록) 뒤에만 본 실행을 한다. 이 스크립트는 LGU 판정 표를 읽지 않는다",
                     prereg=pre, argv=list(sys.argv if argv is None else argv), seeds_requested=seeds, threads=int(a.threads), nice=os.nice(0),
                     code=dict(this=M49.file_info(Path(__file__)), h45=M49.file_info(ROOT / "scripts/2_evaluation/h45_normalized_conformal.py"),
                               lgu_common=M49.file_info(Path(LC.__file__)), h40=M49.file_info(M45.H40_PATH),
                               h49=M49.file_info(ROOT / "scripts/2_evaluation/h49_transfer_map.py"),
                               tab_models=M49.file_info(ROOT / "src/polar/tab_models.py"), sha1_12=cm.get("code_mine")),
                     config=dict(fit_cfg=cm.get("cfg_mine", M45.fit_cfg(C["a45"], NORM)), sigma_rule="셀마다 두 seed σ 의 평균, seed 별 σ = clip((q95 − q05)/2)",
                                 cal_population=f"h49.cal_groups(셀 {al['min_cells_map']}개 이상)", cal_order="h49 real_ctx 원천 순서",
                                 grid_fit="full 학습 집합(원천 전체), 결측은 학습 행렬 중앙값", epochs=int(C["a45"].epochs)),
                     reference=dict(unit=str(C["ref_path"]), **{k: v for k, v in cm.items() if k not in ("cfg_mine", "code_mine")}),
                     inputs=dict(sha1_12=cm.get("inputs_mine"), **C["grid_info"]), plan_table=P, h49_alignment=al,
                     allow_mismatch=bool(a.allow_mismatch), overwrite=bool(a.overwrite), gpu=gpu, gpu_screening=gpu_hist)
    try:
        res = fit_all(C, M45, seeds, gpu)
    except GpuLost as e:
        _write_json(op["failed"], dict(base_meta, status="gpu_lost", error=str(e), wall_s=round(time.time() - t_start, 1)), M49)
        refuse(f"[중단] {e}. 진행 중인 적합을 끝내고 GPU 를 놓았다. npz 를 쓰지 않았다", EXIT_GPU_LOST)
    except (Exception, SystemExit) as e:                                    # noqa: BLE001
        _write_json(op["failed"], dict(base_meta, status="error", error=f"{type(e).__name__}: {e}"[:500],
                                       wall_s=round(time.time() - t_start, 1)), M49)
        raise
    fits = dict(rows=res["rows"], fit_s=res["fit_s"], valid_seeds=res["valid"], n_fit=len(res["rows"]), pid_check=res["pid_check"])
    if len(res["valid"]) < N_SEEDS:
        _write_json(op["failed"], dict(base_meta, status="fit_failed", fits=fits, wall_s=round(time.time() - t_start, 1)), M49)
        log(f"[중단] 유효 seed {res['valid']} < {N_SEEDS}. npz 를 쓰지 않았다(메타 {op['failed']})")
        return EXIT_FIT_FAIL
    summary = write_npz(op["npz"], C, res)
    schema = h49_schema_check(M49, C["mctx"], op["npz"], al["groups_map"])
    meta = dict(base_meta, status="ok" if schema["passed"] else "schema_failed", fits=fits, outputs=dict(npz=M49.file_info(op["npz"]), **summary),
                h49_schema_check=schema)
    if synthetic:
        meta["h49_consumer_run"] = h49_consumer_run(M49, C["D"], C["grid_df"], op["npz"], op["stem"].parent / "h49_check", a.threads)
    if real and not a.no_lgu_compare:
        meta["lgu_repro"] = lgu_repro(M45, C, res)
    meta.update(wall_s=round(time.time() - t_start, 1), max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    _write_json(op["meta"], meta, M49)
    log(f"[done] 적합 {len(res['rows'])}건 {res['fit_s']} s · 유효 seed {res['valid']} · h49 형식 점검 {'통과' if schema['passed'] else '실패'}"
        + (f" · h49 소비 점검 {'통과' if meta['h49_consumer_run']['passed'] else '실패'}" if synthetic else "")
        + f" · 산출 {op['npz']}")
    ok = schema["passed"] and (not synthetic or meta["h49_consumer_run"]["passed"])
    return 0 if ok else EXIT_REFUSE


if __name__ == "__main__":
    sys.exit(main())
