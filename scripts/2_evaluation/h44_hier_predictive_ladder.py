"""H44 · 계층 예측 분포 사다리(LGU 실험 A). 계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 3절(사전 등록)의 구현이다.

목적
  지역 오프셋의 사후 분포로 만든 예측 분포(단 i0–vi)가 같은 라벨로 보정한 상수 폭 구간(B4)보다 구간 점수가 낮은지 본다.
  가설 LGU-A1(확인적, 단 iii − B4), A2(구현 검증, n = 3 무한 구간), A3(보조, 중앙값 RMSE 의 iii − P1), A4(흐름 게이트),
  A5(탐색, 초사전분포 민감도), A6(서술, 러시아 W), A7(보조, 사다리 대비)을 판정 또는 보고한다.

동결 규칙
  h40(scripts/3_deep_learning/h40_label_grid.py), h42, h37, src/polar 의 기존 모듈과 src/polar/lgu_common.py 를 고치지 않는다.
  h40 은 importlib 로 파일 경로에서 읽어 sys.modules['h40_label_grid'] 에 등록하고(모듈 최상위) 자료 적재(Data), 대상·원천(100 km 버퍼,
  Data.source_idx), 분할 구조(Data.split_structure), 라벨 추출(draw_cells, cells_of)을 그대로 쓴다. 분포 점수, 분위 격자, 채점 블록 합
  저장소, 계층 보정, 의사 대상 모사, 전역 블록 재표집, 4분 판정은 polar.lgu_common(LC)을 쓴다. 이 파일에 새로 둔 것은 초모수 적률 추정,
  구적 사후와 예측 CDF, 척도 모형, 흐름 게이트, 2단 재표집, 집계다. 출력은 data/processed/lgu/ 에 둔다.

기호(계획서 2.2절)
  s = e5_sqrt_tdd, E0 = 원천 최소제곱 계수, E0^(−k) = 원천에서 집단 지역 k 를 뺀 계수, u = log y − log(E0·s).
  집단 지역 k 의 원천 셀은 u = log y − log(E0^(−k)·s). E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10. 예측 분포는 Q_i(τ) = E0·s_i·exp(q_u,i(τ)) 이다.

방법(계획서 3.3–3.4절)
  P0 E0·s, P1 E_n·s(점 예측 지표만). B4 중심 E_n·s, 상수 로그 반폭 q_n(c)(의사 대상 모사, 지역 등가중 분위).
  hier2@h37 n = 0, 점수 |log y − log(E0·s)| 의 지역 등가중 분위(h37 규약 참조 행). wconf n ∈ {10, 40, 160}, 라벨 점수 가중 1, 원천 지역 총
  가중 1, 검정점 가중 1, 중심 E0·s(α 0.1, 0.2 구간만).
  i0 h4_common.offset_mle_prior 의 사전 그대로. i 정규 사전 N(0, τ_r1²), 잔차 N(0, σ_1²), 블록 효과 없음(offset_mle_estimate).
  ii 정규 사전 N(0, τ_r²) + 블록 효과 N(0, τ_b²) + 잔차 N(0, σ²)(닫힌 형태). iii t_4(0, τ_r) 사전, τ_r 에 half-normal(0.5) 초사전(1차원 격자).
  iii+c 단 (iii) 구간에 의사 대상 conformal 후보정(초모수는 의사 대상을 뺀 집단으로 다시 추정). iv 이분산 Student-t 잔차, 척도는
  CatBoost(목표 log(|ẽ| + 0.01)), 자유도 ν_ε 는 원천 지역 하나 제외 표본 밖 로그 우도로 고른다, 라벨 우도의 블록 적분은 101점 구적.
  v nflow 위치 고정 변형 잔차(게이트 LGU-A4 를 통과한 대상만, --part flow).
  주 점 예측은 예측 분포의 중앙값 Q(0.50)이다.

부분(--part)과 작업 단위(shards/)
  cpu   emu 단위 (대상, 모드): <tag>__emu__<대상>__<모드>_{emu.npz, unit.json}. 원천 블록 표, 초모수(주 분석, 민감도), 사전 격자,
        의사 대상 모사 항목, 단 (iii+c)의 후보정 폭, 단 (iv)의 척도 적합(주 대상), h4 사전.
        cpu 단위 (대상, 모드, 분할): <tag>__cpu__<대상>__<모드>__s<분할>_{runs.csv, scores.npz, cells.npz, unit.json}.
        emu 단위를 먼저 모두 끝낸 뒤 cpu 단위를 돌린다(같은 프로세스 풀).
  gate  게이트 단위 (대상, 모드): <tag>__gate__<대상>__<모드>_{runs.csv, scores.npz, unit.json}. 주 대상 5개(모드 x)만.
  flow  흐름 단위 (대상, 모드): <tag>__flow__<대상>__<모드>_{runs.csv, scores.npz, unit.json}. 게이트 통과 대상만.
        통과하지 못한 대상은 status 'skipped_gate' 의 unit.json 만 쓴다.
  unit.json 은 마지막에 원자적으로 쓰며 완료 표지다. --resume 은 설정 해시가 같고 status 가 failed 가 아닌 조각만 건너뛴다.
  저장 키 = (method, variant, n, draw, seed). 결정적 방법 seed −1, iv 0–1(CatBoost), v 0–2(nflow).
  저장소 이름: 범위 B '<대상>', 범위 all(n = 0 참조 행) '<대상>|all'(split 0), 6–7월 관측 제외 '<대상>~noearly',
  '<대상>|all~noearly', 게이트 '<대상>~gate:<뺀 지역>'(split 0).

집계(--summarize-only, 실행 뒤 자동)
  1단 CI: LC.one_stage_delta(분할 안 채점 블록 재표집, 보정 고정. LG 규칙).
  2단 CI: LC.global_mult 로 만든 전역 블록 다중도 하나를 보정 쪽(초모수, τ_r 사후, B4 의 q_n)과 채점 쪽이 함께 쓴다. 방법 P1, B4, i, ii, iii,
  지표 METRICS_2STAGE. 주 대상 5개(모드 x)만 계산한다. 대상별 분포는 <tag>_boot__<대상>__<모드>.npz 다.
  산출: lgu_a_curve.csv, lgu_a_tests.csv, lgu_a_hyper.csv, lgu_a_gate.csv, lgu_a_sens.csv, lgu_a_pit.csv, lgu_a_strata.csv,
  lgu_a_timing.csv, lgu_a_failed.csv, lgu_a_meta.json, lgu_a_precision.csv(--precision-only), lgu_a_count.csv(--count-only),
  lgu_a_precheck_quadrature.csv(--precheck: 단 ii 닫힌 형태와 격자 해의 분위 차, 합성 자료 SBC. 계획서 2.9절의 구적 격자 확정 근거).
  스모크와 사전 점검의 산출 파일 이름에는 tag 접미사(_smoke, _precheck)를 붙인다(예: lgu_a_smoke_curve.csv).

실행 환경(사용자 지시 2026-09-30)
  LGU 는 Rescale 이 아니라 로컬 GPU 서버에서 실행한다. 학습과 채점을 하는 실행은 --allow-local(또는 LG_RESCALE=1)이 있어야 하며,
  허용 표지가 없으면 --count-only 만 실행한다(자료 적재 전에 거부, 스레드 1). 허용된 실행에서는(LG_RESCALE=1 이어도 같다, 개정 1)
  이 스크립트가 (1) nice 10 을 적용하고, (2) 동시에 도는 프로세스 수 × --threads 가 --max-threads-total(기본 8)을 넘으면 거부하고
  (끝에 자동 집계를 하면 GPU 프로세스 수와 --workers 의 큰 쪽으로 센다), (3) --gpus 가 --gpu-allow(기본 2,1) 밖이면 거부하며,
  (4) 풀을 만들기 직전에 nvidia-smi 로 메모리 사용이 0 MiB 인 GPU 만 쓴다(CUDA_DEVICE_ORDER=PCI_BUS_ID 로 번호 체계를 맞춘다).
  (5) h44·h45·h46 사이의 스레드 합계는 실행 중 등록부(data/processed/lgu/.running, LC.claim_threads)로 확인한다. 앞 실행과 합해 8 을
  넘으면 거부하므로 LGU 부분들은 합계 8 안에서만 겹쳐 돈다(권고: 부분을 차례로 실행한다).

실행 예(ROOT, 로컬 GPU 서버)
  적합 수:  python3 scripts/2_evaluation/h44_hier_predictive_ladder.py --count-only --threads 1
  스모크:   nice -n 10 python3 scripts/2_evaluation/h44_hier_predictive_ladder.py --smoke --allow-local --part cpu --workers 2 --threads 4
            nice -n 10 python3 scripts/2_evaluation/h44_hier_predictive_ladder.py --smoke --allow-local --part gate --gpus 2 --threads 4 --no-summarize
            nice -n 10 python3 scripts/2_evaluation/h44_hier_predictive_ladder.py --smoke --allow-local --part flow --gpus 2 --threads 4
  사전 점검: --precheck 를 붙여 같은 순서로 돌린다(집계는 --precision-only 로만 한다).
  본 실행:  --part cpu --workers 2 --threads 4 --resume --no-summarize → --part gate --gpus 2 --resume --no-summarize
            → --part flow --gpus 2 --resume --no-summarize → --summarize-only --workers 2 --threads 4

명세(spec_A)와 다르게 구현한 점
  1. 예측 CDF 는 사후 격자 질량과 연속 핵 CDF(정규 Φ, Student-t)의 격자 합성곱(scipy.signal.fftconvolve)으로 직접 계산한다. 명세의
     '이산 핵 합성곱 뒤 누적합'과 같은 양을 핵 쪽 이산화 없이 구한 것이다. sd < 2h 이면 flag 만 남긴다(핵을 생략하지 않아도 식이 성립한다).
  2. 단 (ii)–(iii)의 라벨 우도 Π_j N(ū_j; a, τ_b² + σ²/n_j)는 a 에 대한 2차식이므로 (P, M) 두 수로 줄여 계산한다(상수만 다르다).
  3. τ_r 사후의 N(ā_k; a, v_k)는 격자 칸 적분 질량/h 로, t 사전 성분 t_ν(a; 0, τ)는 격자 칸 적분 질량(Student-t CDF 의 차)으로 계산하고
     격자 밖 질량으로 다시 정규화하지 않는다(개정 1. 앞 판은 τ 마다 합 1 로 다시 정규화해 τ_r 의 주변 우도가 지역마다 1/P(|a| ≤ 2 | τ)
     배 커졌다. 곧 절단 t 사전 모형이었다). 칸 적분이므로 작은 τ 의 격자 해상도 문제도 생기지 않는다. a 의 주변 사전
     ∫ t_ν(a; 0, τ) p(τ | 원천) dτ 는 격자 범위 [−2 − h/2, 2 + h/2] 에서 한 번 정규화한다(주변 사전의 격자 절단). a 의 사후는 그 격자 위의 분포다.
  4. 단 (iv)의 척도 적합(주 대상, CatBoost (K + 1) × seed)은 분할과 무관하므로 cpu 단위가 아니라 emu 단위에서 한 번 계산해 emu.npz 에
     저장하고 분할 단위가 읽는다(명세의 '(대상, seed)마다 한 번'과 같은 결과다). emu 단위는 주 프로세스가 아니라 같은 프로세스 풀에서
     cpu 단위보다 먼저 돈다.
  5. 단 (iii+c)의 후보정 폭 c_n(α)는 대상의 분할과 무관하므로 emu 단위에서 계산해 저장한다.
  6. 흐름(단 v)과 게이트의 셀별 CDF 합성곱은 torch 가 아니라 scipy.signal.fftconvolve(CPU, 셀 묶음)로 한다. nflow 적합과 분위·밀도
     계산은 LC 의 래퍼(torch, 장치 tab_models._dev())다. 흐름의 c = a + b 질량은 간격 0.005 격자에서 0.01 격자로 옮긴다(홀수 칸은 반씩 나눈다).
  7. 민감도 변형 t3, phys, stau025, stau100, w_cell, w_beq, sig_pool 은 B4 를 바꾸지 않으므로 B4 에는 probe_sig(탐침 셀 보정 항목만),
     noearly(early 셀을 보정 항목·라벨·채점에서 제외)만 적용한다. noearly 는 E0, E0^(−k)를 바꾸지 않는다(원천 전체 계수 유지).
  8. 하위 지역 대상(보충)은 P0, P1, B4, i0, i, ii, iii, iii+c 만 돈다(계획서 3.5절. hier2@h37, wconf, iv 는 주 대상만).
  9. 채점 셀은 eval_mask 셀 가운데 y, s 가 유한하고 양수인 셀이다(로그 비가 정의되는 셀). 뺀 셀 수는 runs 와 unit.json 에 적는다.
  10. 2단 CI 는 주 대상 5개(모드 x)만 계산한다(확인적·보조 판정이 주 대상만 쓴다). 하위 지역의 곡선 행은 1단 CI 만 있다(ci_kind 열).
  11. 곡선 표의 1단 수준 CI 는 CURVE_CI_METRICS(is10, cov10, wid10, lhw10, is20, cov20, crps, crps_log, se)만 계산한다.
  12. 커버리지 조건(풀 평균 cov10 ≥ 0.80)은 셀 가중 점 추정으로 판단하고 블록 등가중 값은 열로 병기한다.
  13. 게이트의 뺀 지역별 1단 CI 는 지역마다 다른 seed(seed_of('lgu-gate', 대상, 지역))의 재표집을 쓴다.
  14. --seeds 는 명세의 --flow-seeds, --cb-seeds 와 함께 둔 공통 상한이다(0 이면 두 값을 그대로 쓴다).
  15. 로컬 자원 규칙(nice, 스레드 합계, GPU 허용 목록과 0 MiB 확인)을 스크립트가 강제한다(--gpu-allow, --max-threads-total).
  16. CRPS(로그 척도) 대비(LGU-A7)의 동등 한계는 명세에 없어 구간 점수와 같은 상대 5 %(보조 10 %)를 쓴다.
  17. --precision-only 에서도 대비는 내부에서 계산하지만 파일과 화면에는 반폭의 기준 점수 대비 비율(구간 점수 대비)이나 반폭(cm, 점 예측
      대비)과 정밀도 상태만 낸다(precision_frame. 개정 1 에서 기준 점수 열을 뺐다).
  18. emu.npz 의 배열 이름은 명세를 따른다(eg_*, en_*, lab_*, nest_*, bt_*). 탐침 셀과 early 제외 블록 표는 btp_*, btne_* 다.
      단 (iv)의 척도 적합은 iv_h(seed, 대상 셀), iv_c, iv_nu, iv_ll 이다.

개정 1(2026-09-30, 검증 지적 반영. 계획서 개정 이력 참조)
  a. 게이트의 단 (iv) 척도 보정 계수 c 를 표본 안 예측이 아니라 학습 지역 q 마다 {k, q} 를 뺀 적합의 표본 밖 예측으로 구한다(3.3절의 정의,
     emu 의 iv_scale_fit 과 같은 식). 대상당 CatBoost 적합이 K 에서 K² 로 늘어난다. gate runs 에 scale_c_method 를 적는다.
  b. 게이트 상태 gate_state ∈ {통과, 미통과, 판정 불가(적합 실패), 판정 불가(계산 실패), 판정 불가(지역 부족)}. 판정 불가는 미통과로 세지
     않는다. '전 대상 미통과' 문장은 주 대상 5개의 게이트 조각이 모두 있고 모두 미통과일 때만 쓴다. 예외로 생긴 적합 실패는 조각을 failed 로 둔다.
     LGU-A7 의 (v − iv)는 흐름 유효 seed 2개 미만이면 '판정 불가(적합 실패)'다.
  c. LGU-A1(과 2.6절상 2단 CI 가 주인 A3)은 2단 분포가 없으면 '판정 불가(2단 CI 없음)'이고, 1단 짝지음 점검(한쪽에만 있는 (n, 추출) 단위,
     NaN 키, cpu 조각의 계산 실패)이 실패하면 '판정 불가(계산 실패)'다. 1단 판정과 CI 는 verdict_1stage, ci_*_1stage 열에만 둔다.
  d. 분할 완결성은 지역별로 비교한다(splits_complete). a1_decide 는 열세 검사를 판정 불가 검사보다 먼저 한다('기각(다른 n 판정 불가)').
     알래스카 의존 표기는 부분 지지의 n 까지 비교한다. LGU-A3 의 4개 행에 Holm 보조 열을 채운다.
  e. τ_r 사후의 t 성분을 칸 적분 질량(다시 정규화하지 않음)으로 바꿨다(명세와 다른 점 3).
  f. 보조 열(계획서 2.8절, 3.7절): 변형 'aux:tmean'(단 i, ii 의 1–99 % 절단 평균 예측)과 'aux:Emean'(단 i0, i, ii, iii 의 사후 평균 E 예측).
     점 예측 지표만 저장하며 판정에 쓰지 않는다. 곡선 표에 같은 방법의 행으로 나온다.
  g. 2단 분포 파일은 분할 목록, cells·emu 해시, 코드 해시가 같을 때만 다시 쓰고, 다시 계산할 대상의 옛 파일은 계산 전에 지운다.
     흐름 조각은 게이트 unit.json 해시(gate_ref)가 같을 때만 완료로 본다.
  h. 환경·자원 오류(LC.is_env_error)는 키 하나의 실패가 아니라 조각 실패로 둔다. 보정 집단 G 는 LC.cal_groups(h45 와 같은 정의)이다.
     summarize 는 lgu_blockindex.csv 를 h45 와 같이 쓰거나 대조한다(LC.check_blockindex).

확인하지 못한 것
  개정 1 단계에서 실행한 확인은 py_compile, pyflakes, 스레드 1개의 --count-only(대상 15, 약 10 s), 가벼운 단위 시험(합성 자료. emu·cpu
  단위의 계산, 2단 재표집 함수, 판정·정밀도 표, 게이트 판정, 로컬 규칙. CPU 스레드 2)이다. GPU 2·1번이 사용 중이어서 스모크와 사전 점검은
  하지 않았다. 따라서 실제 자료의 emu·cpu·gate·flow 단위, 게이트의 표본 밖 c 계산(합성 자료 시험 o 는 추가만 하고 실행하지 않았다),
  2단 분포 재사용 조건, 실행 중 스레드 등록부의 실제 동작은 확인하지 않았다. 구적 격자 간격 0.005 의 정확도(계획서 2.9절 기준 0.002),
  2단 재표집의 계산 시간, 단 (iv)의 CatBoost 척도 모형과 자유도 선택의 실제 자료 동작, nflow 위치 고정 변형의 학습 안정성, 게이트의 판정
  분포, 흐름 단의 메모리 사용량은 확인하지 않았다. physics.load_physics_inputs 가 실제 원천 셀에서 bdod 범위 확인을 통과하는지(phys 변형)도
  확인하지 않았다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _permitted(argv=None, environ=None) -> bool:
    """허용 표지: 환경 변수 LG_RESCALE=1 또는 --allow-local."""
    av = sys.argv if argv is None else argv
    env = os.environ if environ is None else environ
    return env.get("LG_RESCALE", "") == "1" or "--allow-local" in av


def _peek_threads(default="4") -> str:
    """numpy 를 적재하기 전에 스레드 수를 정한다(h42 와 같은 규칙). 허용 표지가 없으면 --threads 와 무관하게 1 이다."""
    av = sys.argv
    if not _permitted(av):
        return "1"
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return av[i + 1]
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return default


if __name__ == "__main__":                                             # 스크립트: numpy 적재 전에 스레드 수를 강제한다
    for _v in _THREAD_VARS:
        os.environ[_v] = _peek_threads()
else:                                                                  # spawn 워커(__mp_main__)와 시험의 import: 부모가 둔 값을 존중한다
    for _v in _THREAD_VARS:
        os.environ.setdefault(_v, _peek_threads())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                          # CUDA 장치 번호를 nvidia-smi 번호(PCI 버스 순)와 맞춘다(h43 과 같다)

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402
from scipy.signal import fftconvolve                                                                 # noqa: E402
from scipy.special import gammaln, logsumexp, ndtr, ndtri, stdtr, stdtrit                            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
H40_PATH = ROOT / "scripts" / "3_deep_learning" / "h40_label_grid.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_h40():
    """h40 을 파일 경로에서 읽어 sys.modules 에 등록한다(h41·h42 와 같은 방식). 이미 등록되어 있으면 그 모듈을 쓴다."""
    name = "h40_label_grid"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, H40_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_h40()

import polar.h4_common as H4                                                                         # noqa: E402
import polar.lgu_common as LC                                                                        # noqa: E402
from polar.h4_common import offset_mle_estimate, offset_mle_prior, seed_of                           # noqa: E402
from polar.m1_core import eval_mask, half_split_blocks                                               # noqa: E402
from polar.m1_stats import holm                                                                      # noqa: E402

# ================================================================ 1. 고정 설계값(계획서 3절. 바꾸면 사전 등록에서 벗어난다)
FMT = "lgua_v1"
MAIN4 = list(H.MAIN4)                                              # Lena, Canada, Russia_W, Russia_E
MAIN5 = MAIN4 + [H.ALASKA]
MAIN3 = [t for t in MAIN4 if t != "Russia_W"]
SUB10 = list(H.SUB_AL) + list(H.SUB_OTHER)
POOL3 = ["Lena", "Canada", H.ALASKA]                               # 실험 A 의 풀(계획서 2.7절)
POOL2 = ["Lena", "Canada"]
RUSSIA_W = "Russia_W"
RUNGS_ALL = ["P0", "P1", "B4", "hier2@h37", "wconf", "i0", "i", "ii", "iii", "iii+c", "iv"]
SUB_RUNGS = {"P0", "P1", "B4", "i0", "i", "ii", "iii", "iii+c"}      # 하위 지역(보충) 대상의 단(계획서 3.5절)
SENS_ALL = ["t3", "phys", "stau025", "stau100", "w_cell", "w_beq", "sig_pool", "probe_sig", "noearly"]
B4_SENS = ("probe_sig", "noearly")                                  # B4 를 바꾸는 민감도(명세와 다른 점 7)
VARIANT_SPEC = {                                                    # 민감도 → (초모수 이름, ν, half-normal 척도, 물리 절단)
    "main": ("main", None, None, False), "t3": ("main", 3.0, None, False), "phys": ("main", None, None, True),
    "stau025": ("main", None, 0.25, False), "stau100": ("main", None, 1.0, False), "w_cell": ("w_cell", None, None, False),
    "w_beq": ("w_beq", None, None, False), "sig_pool": ("sig_pool", None, None, False), "probe_sig": ("probe_sig", None, None, False),
    "noearly": ("noearly", None, None, False)}
STRATA = ("gpr", "probe", "early")
WCONF_N = (10, 40, 160)
NU_EPS_CAND = (3.0, 5.0, 10.0, float("inf"))                        # 단 (iv) 잔차 자유도 후보
SIG_LEVELS = np.geomspace(LC.SIGMA_CLIP[0], LC.SIGMA_CLIP[1], 32)   # 단 (iv) σ 수준 32개
TAU_R = np.geomspace(0.01, 2.0, 120)                               # τ_r 격자
N_BGRID = 101                                                      # 블록 효과 b 의 구적 점 수
HG = float(LC.H_GRID)                                              # 로그 척도 격자 간격 0.005
A_HALF, U_HALF = 400, 800
A_GRID = np.arange(-A_HALF, A_HALF + 1) * HG                       # a 격자 [−2, 2] 801점
A2 = A_GRID ** 2
U_GRID = np.arange(-U_HALF, U_HALF + 1) * HG                       # u 격자 [−4, 4] 1601점
A_OFF = U_HALF - A_HALF                                            # a 격자의 첫 점 = u 격자의 400번째 점
EPS_H = 0.01
EPS_GRID = np.arange(-400, 401) * EPS_H                            # 흐름의 ε 격자 [−4, 4] 간격 0.01
LOGF_FLOOR = -30.0                                                 # ε 격자 밖의 로그 밀도
PHYS_DRAWS = 64
CB_PARAMS = dict(iterations=200, depth=3, learning_rate=0.05, l2_leaf_reg=3.0)   # catboost_lo 설정
FIXED = dict(kappa=10.0, buffer_km=100.0, min_cal_cells=20, nu=4.0, s_tau=0.5, cap_cells=100, clamp=2.0)
Q5 = tuple(float(t) for t in LC.Q5)
Q5_IDX = [LC.grid_index(t) for t in Q5]
LADDER = [("ii", "i"), ("iii", "ii"), ("iv", "iii"), ("iii+c", "iii"), ("v", "iv")]
A7_N = (0, 10, 40)
TWO_STAGE_METHODS = ("P1", "B4", "i", "ii", "iii")
M2 = list(LC.METRICS_2STAGE)
CURVE_CI_METRICS = ("is10", "cov10", "wid10", "lhw10", "is20", "cov20", "crps", "crps_log", "se")
EG_COLS = ("region", "n", "e", "d", "E0_mk", "E_center", "E_center_ne", "n_lab_ne", "delta", "start", "stop", "lab_start", "lab_stop")
SHARD_FILES = {"emu": ("emu.npz",), "cpu": ("runs.csv", "scores.npz", "cells.npz"), "gate": ("runs.csv", "scores.npz"),
               "flow": ("runs.csv", "scores.npz")}


# ================================================================ 2. 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H44 계층 예측 분포 사다리(LGU 실험 A)")
    ap.add_argument("--part", choices=["cpu", "gate", "flow"], default="cpu")
    ap.add_argument("--targets", default="", help="쉼표 목록 '이름' 또는 '이름:모드'. 기본: cpu 는 주 5개 x + 하위 지역 10개 i, gate·flow 는 주 5개 x")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--n-grid", default="0,3,10,40,160")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=0, help="0 = --flow-seeds, --cb-seeds 를 그대로 쓴다. 양수면 두 값의 공통 상한")
    ap.add_argument("--flow-seeds", type=int, default=3)
    ap.add_argument("--cb-seeds", type=int, default=2)
    ap.add_argument("--rungs", default=",".join(RUNGS_ALL))
    ap.add_argument("--sens", default=",".join(SENS_ALL))
    ap.add_argument("--workers", type=int, default=2, help="CPU 프로세스 수. 0 = 풀 없이 차례로")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpus", default="", help="gate·flow 부분의 GPU 물리 번호 목록. 빈 문자열이면 CPU")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    ap.add_argument("--gpu-allow", default="2,1,5", help="--allow-local 실행에서 쓸 수 있는 GPU 물리 번호(사용자 지시 2026-09-30)")
    ap.add_argument("--max-threads-total", type=int, default=8, help="--allow-local 실행의 CPU 스레드 합계 상한")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--nboot", type=int, default=LC.NBOOT)
    ap.add_argument("--out-dir", default="data/processed/lgu")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default="lgua")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--label-flags", default="lgx_label_flags_v1.csv")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--precision-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-gate", choices=["auto", "pass", "fail"], default="auto", help="스모크 전용")
    ap.add_argument("--pool-retries", type=int, default=2)
    # 아래는 고정 설계값이다. 바꾸면 사전 등록에서 벗어난다(meta 의 fixed_deviation 에 적는다).
    ap.add_argument("--kappa", type=float, default=FIXED["kappa"])
    ap.add_argument("--buffer-km", type=float, default=FIXED["buffer_km"])
    ap.add_argument("--min-cal-cells", type=int, default=FIXED["min_cal_cells"])
    ap.add_argument("--nu", type=float, default=FIXED["nu"])
    ap.add_argument("--s-tau", type=float, default=FIXED["s_tau"])
    ap.add_argument("--cap-cells", type=int, default=FIXED["cap_cells"])
    ap.add_argument("--clamp", type=float, default=FIXED["clamp"])
    return finalize(ap.parse_args(argv))


def _n_list(txt):
    out = []
    for v in str(txt).split(","):
        v = v.strip()
        if not v:
            continue
        if v == "all":
            raise SystemExit("LGU 는 n 격자에 전량(all)을 쓰지 않는다(계획서 2.1절)")
        out.append(int(v))
    return out


def _targets(txt, part):
    if not txt:
        base = [(t, "x") for t in MAIN5]
        return base + ([(t, "i") for t in SUB10] if part == "cpu" else [])
    out = []
    for tok in [v.strip() for v in str(txt).split(",") if v.strip()]:
        if ":" in tok:
            t, m = tok.split(":", 1)
        else:
            t, m = tok, ("i" if tok in SUB10 else "x")
        if m not in H.valid_modes(t):
            raise SystemExit(f"대상 {t} 에 모드 {m} 는 없다")
        out.append((t, m))
    return out


def finalize(a):
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    if a.force_gate != "auto" and not a.smoke:
        raise SystemExit("--force-gate 는 스모크 전용이다")
    if int(a.cap_cells) != LC.CAP_CELLS:
        raise SystemExit(f"--cap-cells 는 고정 설계값 {LC.CAP_CELLS} 이다(LC.row_weights 가 이 값을 쓴다)")
    a.TAG = a.tag + ("_smoke" if a.smoke else "") + ("_precheck" if a.precheck else "")
    a.SFX = ("_smoke" if a.smoke else "") + ("_precheck" if a.precheck else "")
    a.SPLITS = list(range(1, int(a.splits) + 1))
    a.N_GRID = _n_list(a.n_grid)
    a.DRAWS = int(a.draws)
    a.FLOW_SEEDS = list(range(int(a.flow_seeds)))
    a.CB_SEEDS = list(range(int(a.cb_seeds)))
    if int(a.seeds) > 0:
        a.FLOW_SEEDS = a.FLOW_SEEDS[:int(a.seeds)]; a.CB_SEEDS = a.CB_SEEDS[:int(a.seeds)]
    a.RUNGS = [v.strip() for v in a.rungs.split(",") if v.strip()]
    a.SENS = [v.strip() for v in a.sens.split(",") if v.strip()]
    bad = [v for v in a.RUNGS if v not in RUNGS_ALL] + [v for v in a.SENS if v not in SENS_ALL]
    if bad:
        raise SystemExit(f"알 수 없는 단 또는 민감도: {bad}")
    a.TARGETS = _targets(a.targets, a.part)
    if a.smoke:                                                     # 계획서 명세: 대상 3, 분할 1, n {0, 3, 10, 40}, 추출 1, nboot 200
        a.SPLITS = [1]; a.N_GRID = [0, 3, 10, 40]; a.DRAWS = 1; a.nboot = min(int(a.nboot), 200)
        a.epochs = min(int(a.epochs), 3); a.FLOW_SEEDS = [0]; a.CB_SEEDS = [0]
        if a.force_gate == "auto":
            a.force_gate = "pass"
        if not a.targets:
            a.TARGETS = [("Russia_W", "x"), ("Canada", "x"), ("AL-3", "i")] if a.part == "cpu" else [("Canada", "x")]
    if a.precheck:                                                  # 사전 점검: 분할 1, n {0, 10, 40}, 추출 2, 본 실행과 같은 epochs
        a.SPLITS = [1]; a.N_GRID = [0, 10, 40]; a.DRAWS = 2; a.FLOW_SEEDS = [0]; a.CB_SEEDS = [0]
        if not a.targets:
            a.TARGETS = [("Lena", "x"), ("Canada", "x"), (H.ALASKA, "x")] if a.part == "cpu" else [("Canada", "x")]
        if a.summarize_only and not a.precision_only:
            print("[precheck] 사전 점검의 집계는 --precision-only 로만 한다(계획서 6절). --precision-only 를 켠다", flush=True)
            a.precision_only = True
    if a.part in ("gate", "flow"):
        bad = [(t, m) for t, m in a.TARGETS if not (t in MAIN5 and m == "x")]
        if bad:
            raise SystemExit(f"--part {a.part} 는 주 대상 5개(모드 x)만 돈다: {bad}")
    if any(n < 0 for n in a.N_GRID) or 0 not in a.N_GRID:
        raise SystemExit("n 격자는 0 을 포함한 음이 아닌 정수여야 한다")
    a.GPUS = [g.strip() for g in str(a.gpus).split(",") if g.strip() != ""]
    a.OUT = (ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir)
    a.PROC = (ROOT / a.data_dir) if not os.path.isabs(a.data_dir) else Path(a.data_dir)
    a.SHARDS = a.OUT / "shards"
    a.SUBMAP = Path(a.subregion_map) if os.path.isabs(a.subregion_map) else a.PROC / a.subregion_map
    a.FLAGS = Path(a.label_flags) if os.path.isabs(a.label_flags) else a.PROC / a.label_flags
    a.threads_asked = int(a.threads)
    if not LC.run_permitted(a.allow_local):
        a.threads = 1
    a.fixed_deviation = {k: getattr(a, k) for k, v in FIXED.items() if float(getattr(a, k)) != float(v)}
    return a


def out_name(a, stem, ext="csv"):
    """집계 파일 이름: lgu_a_<stem>.csv(스모크·사전 점검은 lgu_a_<접미사>_<stem>)."""
    return a.OUT / (f"lgu_a{a.SFX}_{stem}.{ext}")


def h40_argv(a):
    """h40.parse_args 인자(자료 적재, 원천 버퍼, 분할 구조가 h40 과 같은 규칙으로 돌게 한다)."""
    return ["--part", "cpu", "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", str(a.SUBMAP), "--modes", "i,x",
            "--splits", str(max(a.SPLITS)), "--n-grid", ",".join(str(n) for n in a.N_GRID), "--draws", str(a.DRAWS), "--seeds", "1",
            "--threads", str(a.threads), "--kappa", str(a.kappa), "--buffer-km", str(a.buffer_km)]


def is_main(t, m):
    return t in MAIN5 and m == "x"


def rungs_for(a, t, m):
    r = [v for v in a.RUNGS if (v in SUB_RUNGS or is_main(t, m))]
    return r


# ================================================================ 3. 자료 보기
class Frame:
    """자료 보기. df 는 y, s, macro, block, lat, lon, loc_id 와 입력 x25 열을 가진다(본 실행은 h40.Data 의 df, 시험은 합성 자료).
    속성: y, s, uok(로그 비가 정의되는 셀), mac, bi(LC.BlockIndex), col(셀의 전역 블록 열 번호), flags(early, gpr, probe), feats."""

    def __init__(self, df, flags, feats=None):
        self.df = df
        self.y = df["y"].values.astype(float)
        self.s = df["s"].values.astype(float)
        with np.errstate(invalid="ignore"):
            self.uok = np.isfinite(self.y) & np.isfinite(self.s) & (self.y > 0) & (self.s > 0)
        self.mac = np.asarray(df["macro"].values).astype(str)
        self.bi = LC.BlockIndex(df)
        self.col = self.bi.cols(np.arange(len(df)))
        self.flags = {k: np.asarray(flags[k], bool) for k in STRATA}
        self.feats = list(feats if feats is not None else H.FEATS)
        self._X = None

    @property
    def X(self):
        if self._X is None:
            self._X = self.df[self.feats].values.astype(float)
        return self._X

    def u_of(self, idx, E):
        """로그 비 log y − log(E·s). E 는 스칼라 또는 셀별 배열. 정의되지 않는 셀은 NaN."""
        idx = np.asarray(idx, np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.log(self.y[idx]) - np.log(np.asarray(E, float) * self.s[idx])
        return np.where(self.uok[idx], u, np.nan)


_FRAME: dict = {}


def get_D(a):
    """h40.Data(프로세스마다 한 번 적재, h40.get_data 의 캐시)."""
    if not hasattr(a, "HA"):
        a.HA = H.parse_args(h40_argv(a))
    return H.get_data(a.HA)


def frame_of(a):
    D = get_D(a)
    key = (id(D.df), str(a.FLAGS))
    if key not in _FRAME:
        _FRAME.clear()
        _FRAME[key] = Frame(D.df, LC.load_flags(a.FLAGS, D.df["loc_id"].values))
    return _FRAME[key]


# ================================================================ 4. 분포 계산의 기본 함수
def t_logpdf(x, nu):
    """표준 Student-t(자유도 nu) 로그 밀도. nu = inf 이면 표준 정규."""
    x = np.asarray(x, float)
    if not np.isfinite(nu):
        return -0.5 * x * x - 0.5 * np.log(2.0 * np.pi)
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
    return c - (nu + 1.0) / 2.0 * np.log1p(x * x / nu)


def t_cdf(x, nu):
    return ndtr(x) if not np.isfinite(nu) else stdtr(nu, x)


def t_ppf(p, nu):
    return ndtri(p) if not np.isfinite(nu) else stdtrit(nu, p)


def kappa_nu(nu):
    """κ_ν = sqrt((ν − 2)/ν). σ 가 잔차의 표준편차가 되도록 t 척도를 맞춘다. ν = inf 이면 1."""
    return 1.0 if not np.isfinite(nu) else float(np.sqrt((nu - 2.0) / nu))


def _trap_w(x):
    """비등간격 격자의 사다리꼴 가중."""
    x = np.asarray(x, float)
    w = np.zeros(len(x))
    if len(x) > 1:
        d = np.diff(x)
        w[:-1] += d / 2.0
        w[1:] += d / 2.0
    return w


def _bin_mass(center, mu, var, h=HG):
    """격자 칸 [c − h/2, c + h/2] 에 들어가는 N(mu, var) 질량. center (G,), mu·var (K,) → (G, K). 꼬리의 상쇄 오차를 피한다."""
    c = np.asarray(center, float)[:, None]
    mu = np.atleast_1d(np.asarray(mu, float))[None, :]
    sd = np.sqrt(np.atleast_1d(np.asarray(var, float)))[None, :]
    up = (c + h / 2.0 - mu) / sd
    lo = (c - h / 2.0 - mu) / sd
    return np.where(up + lo > 0, ndtr(-lo) - ndtr(-up), ndtr(up) - ndtr(lo))


_COMP: dict = {}


def _tau_comp(nu):
    """τ_r 격자의 t_ν(0, τ) 성분의 a 격자 칸 적분 질량 (120, 801): F((a + h/2)/τ) − F((a − h/2)/τ), F 는 t_ν(ν = inf 이면 정규) CDF.
    격자 밖 질량으로 다시 정규화하지 않는다(개정 1). τ 마다 행 합은 P(|a| ≤ 2 + h/2 | τ) ≤ 1 이다. 칸 적분이므로 작은 τ 의 해상도
    문제도 생기지 않는다. 오른쪽 꼬리는 생존 함수의 차로 계산해 상쇄 오차를 피한다."""
    key = float(nu)
    if key not in _COMP:
        up = (A_GRID[None, :] + HG / 2.0) / TAU_R[:, None]
        lo = (A_GRID[None, :] - HG / 2.0) / TAU_R[:, None]
        _COMP[key] = np.where(lo > 0, t_cdf(-lo, nu) - t_cdf(-up, nu), t_cdf(up, nu) - t_cdf(lo, nu))
    return _COMP[key]


def tau_r_posterior(a_k, v_k, nu=FIXED["nu"], s_tau=FIXED["s_tau"]):
    """τ_r 의 사후 (tau_grid, p_tau). p(τ) ∝ HN(τ; s_tau)·Π_k ∫ t_ν(a; 0, τ)·N(ā_k; a, v_k) da (계획서 3.3절 단 iii).
    p_tau 는 τ 격자의 사다리꼴 가중으로 정규화한 밀도다. 지역이 없으면 초사전분포 그대로다."""
    a_k = np.atleast_1d(np.asarray(a_k, float)); v_k = np.atleast_1d(np.asarray(v_k, float))
    ok = np.isfinite(a_k) & np.isfinite(v_k) & (v_k > 0)
    a_k, v_k = a_k[ok], v_k[ok]
    s_tau = float(s_tau)
    logp = np.log(2.0) - 0.5 * np.log(2.0 * np.pi) - np.log(s_tau) - TAU_R ** 2 / (2.0 * s_tau ** 2)
    if len(a_k):
        dens = _bin_mass(A_GRID, a_k, v_k) / HG                         # (801, K) 칸 평균 밀도
        I = _tau_comp(nu) @ dens                                        # (120, K)
        logp = logp + np.log(np.maximum(I, 1e-300)).sum(axis=1)
    p = np.exp(logp - logp.max())
    p = p / float((_trap_w(TAU_R) * p).sum())
    return TAU_R.copy(), p


def tau_quantiles(tau_grid, p_tau, qs=(0.05, 0.5, 0.95)):
    """τ_r 사후의 분위(사다리꼴 누적)."""
    w = _trap_w(tau_grid) * np.asarray(p_tau, float)
    cdf = np.cumsum(w) / w.sum()
    return [float(np.interp(q, cdf, tau_grid)) for q in qs]


def prior_a(tau_grid, p_tau, nu=FIXED["nu"], trunc=None):
    """a 의 사전 질량 (801,), 합 1. p(a) = ∫ t_ν(a; 0, τ)·p(τ | 원천) dτ 의 칸 적분 질량을 격자 범위 [−2 − h/2, 2 + h/2] 에서
    한 번 정규화한다(주변 사전의 격자 절단, 개정 1). trunc = (lo, hi)이면 밖을 0 으로 두고 다시 정규화한다
    (절단 범위가 격자와 겹치지 않으면 절단하지 않는다)."""
    tg = np.asarray(tau_grid, float)
    if len(tg) != len(TAU_R) or not np.allclose(tg, TAU_R):
        raise ValueError("tau_grid 는 TAU_R 이어야 한다")
    pa = (_trap_w(tg) * np.asarray(p_tau, float)) @ _tau_comp(nu)
    if trunc is not None and all(np.isfinite(trunc)):
        m = (A_GRID >= float(trunc[0])) & (A_GRID <= float(trunc[1]))
        if m.sum() >= 2 and pa[m].sum() > 0:
            pa = np.where(m, pa, 0.0)
    return pa / pa.sum()


def normal_prior_mass(tau2):
    """정규 사전 N(0, τ²)의 a 격자 질량(시험과 정규 변형용)."""
    m = _bin_mass(A_GRID, [0.0], [float(tau2)])[:, 0]
    return m / m.sum()


def label_blocks(u_L, blk_L):
    """라벨을 블록별로 묶는다 → (n_j, ū_j). 비유한 u 는 뺀다."""
    u = np.asarray(u_L, float); b = np.asarray(blk_L)
    ok = np.isfinite(u)
    if not ok.any():
        return np.zeros(0), np.zeros(0)
    _, inv = np.unique(b[ok], return_inverse=True)
    inv = inv.ravel()
    n_j = np.bincount(inv).astype(float)
    return n_j, np.bincount(inv, weights=u[ok]) / n_j


def lik_PM(n_j, ubar_j, tau_b2, sigma2):
    """라벨 우도 Π_j N(ū_j; a, V_j), V_j = τ_b² + σ²/n_j 의 2차식 계수 (P = Σ 1/V_j, M = Σ ū_j/V_j)."""
    n_j = np.asarray(n_j, float); ubar_j = np.asarray(ubar_j, float)
    if len(n_j) == 0:
        return 0.0, 0.0
    V = float(tau_b2) + float(sigma2) / n_j
    return float((1.0 / V).sum()), float((ubar_j / V).sum())


def posterior_normal(n_j, ubar_j, tau_r2, tau_b2, sigma2):
    """단 (ii)의 닫힌 형태 (m, v): P = 1/τ_r² + Σ_j 1/(τ_b² + σ²/n_j), m = Σ_j ū_j/(τ_b² + σ²/n_j) / P, v = 1/P. 라벨이 없으면 (0, τ_r²)."""
    P, M = lik_PM(n_j, ubar_j, tau_b2, sigma2)
    prec = 1.0 / float(tau_r2) + P
    return float(M / prec), float(1.0 / prec)


def _safe_log(p):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(p, float))


def post_from_PM(log_pa, P, M):
    """log p(a | 라벨) = log p(a) − P·a²/2 + M·a + 상수. P, M 은 스칼라 또는 (B,). 반환 질량 (B, 801)(행마다 합 1)."""
    P = np.atleast_1d(np.asarray(P, float)); M = np.atleast_1d(np.asarray(M, float))
    lp = np.asarray(log_pa, float)[None, :] - 0.5 * P[:, None] * A2[None, :] + M[:, None] * A_GRID[None, :]
    mx = np.max(np.where(np.isfinite(lp), lp, -np.inf), axis=1, keepdims=True)
    p = np.exp(lp - mx)
    p[~np.isfinite(p)] = 0.0
    return p / p.sum(axis=1, keepdims=True)


def posterior_grid(p_a, n_j, ubar_j, tau_b2, sigma2):
    """단 (iii)의 a 사후 질량 (801,): 로그 영역에서 사전과 블록 라벨 우도 N(ū_j; a, τ_b² + σ²/n_j)를 곱하고 정규화한다."""
    P, M = lik_PM(n_j, ubar_j, tau_b2, sigma2)
    return post_from_PM(_safe_log(p_a), P, M)[0]


def post_stats(p, grid=A_GRID):
    """격자 질량의 중앙값과 표준편차."""
    p = np.asarray(p, float)
    c = np.cumsum(p)
    med = float(np.interp(0.5, c, grid))
    mu = float((p * grid).sum())
    return med, float(np.sqrt(max((p * grid ** 2).sum() - mu ** 2, 0.0)))


def lattice_cdf(w, off, G, n_out=len(U_GRID)):
    """격자 합성곱으로 CDF 를 만든다. w (L,) 또는 (B, L): 질량 w_k 가 u 격자 번호 off + k 에 있다. G(x): 차이 x 의 연속 CDF.
    F_j = Σ_k w_k G((j − off − k)·h), j = 0..n_out − 1. 누적 최댓값으로 단조화하고 [0, 1] 로 자른다."""
    w = np.asarray(w, float)
    one = w.ndim == 1
    W2 = w[None, :] if one else w
    L = W2.shape[1]
    dmin, dmax = -off - (L - 1), n_out - 1 - off
    g = np.asarray(G(np.arange(dmin, dmax + 1) * HG), float)
    conv = fftconvolve(W2, g[None, :], axes=1)
    F = np.clip(np.maximum.accumulate(conv[:, L - 1:L - 1 + n_out], axis=1), 0.0, 1.0)
    return F[0] if one else F


def q_from_cdf(F, grid, taus):
    """단조 CDF 격자 F (n,) 또는 (B, n) → 분위 (T,) 또는 (B, T). 첫 F_j ≥ τ 인 j 와 j − 1 사이를 선형 보간한다.
    τ 가 F_0 이하이면 grid[0], F 의 최댓값보다 크면 grid[−1]."""
    F = np.asarray(F, float)
    one = F.ndim == 1
    F2 = F[None, :] if one else F
    B, n = F2.shape
    t = np.asarray(taus, float)
    off = 2.0 * np.arange(B)[:, None]
    flat = (F2 + off).ravel()
    tgt = t[None, :] + off
    pos = np.searchsorted(flat, tgt.ravel(), side="left").reshape(B, len(t)) - (np.arange(B)[:, None] * n)
    pos = np.clip(pos, 0, n)
    grid = np.asarray(grid, float)
    q = np.empty((B, len(t)))
    lo_end, hi_end = pos <= 0, pos >= n
    mid = ~(lo_end | hi_end)
    if mid.any():
        r = np.broadcast_to(np.arange(B)[:, None], pos.shape)[mid]
        j = pos[mid]
        tv = np.broadcast_to(t[None, :], pos.shape)[mid]
        F0, F1 = F2[r, j - 1], F2[r, j]
        q[mid] = grid[j - 1] + (tv - F0) / np.maximum(F1 - F0, 1e-300) * (grid[j] - grid[j - 1])
    q[lo_end] = grid[0]
    q[hi_end] = grid[-1]
    return q[0] if one else q


def predictive_q(p_post, sd, taus=LC.TAUS):
    """단 (iii)의 예측 분위: a 사후 질량과 N(0, sd²)의 합성곱 CDF(u 격자 [−4, 4])에서 분위를 읽는다. p_post (801,) 또는 (B, 801)."""
    sd = float(sd)
    if sd > 0:
        G = lambda x: ndtr(x / sd)                                       # noqa: E731
    else:
        G = lambda x: (x >= 0).astype(float)                             # noqa: E731
    return q_from_cdf(lattice_cdf(p_post, A_OFF, G), U_GRID, taus)


def normal_q(m, v, taus=LC.TAUS):
    """정규 예측 분포의 분위 m + sqrt(v)·Φ⁻¹(τ)."""
    return float(m) + np.sqrt(max(float(v), 0.0)) * ndtri(np.asarray(taus, float))


# ---------------------------------------------------------------- 단 (iv)·(v)·게이트의 도우미
def pc_from_post(p_post, tau_b2):
    """c = a + b 의 질량(u 격자 1601점). b ~ N(0, τ_b²)의 격자 칸 질량(범위 ±6τ_b)과 a 사후 질량의 합성곱.
    격자 밖 질량은 끝점에 둔다. τ_b < 2h 이면 합성곱을 생략하고 flag 'tau_b<2h' 를 돌려준다."""
    p = np.asarray(p_post, float)
    out = np.zeros(len(U_GRID))
    tb = float(np.sqrt(max(float(tau_b2), 0.0)))
    if tb < 2 * HG:
        out[A_OFF:A_OFF + len(p)] = p
        return out / out.sum(), "tau_b<2h"
    K = int(math.ceil(6.0 * tb / HG))
    ker = _bin_mass(np.arange(-K, K + 1) * HG, [0.0], [tb * tb])[:, 0]
    conv = np.convolve(p, ker / ker.sum())
    idx = np.clip(A_OFF - K + np.arange(len(conv)), 0, len(U_GRID) - 1)
    np.add.at(out, idx, conv)
    return out / out.sum(), ""


def point_mass_a0():
    """a = 0 의 점 질량(게이트의 '지역 오프셋을 아는' 조건)."""
    p = np.zeros(len(A_GRID)); p[A_HALF] = 1.0
    return p


_GCACHE: dict = {}


def _g_t(nu, sig):
    """단 (iv) σ 수준 표의 핵 CDF 배열(차이 격자 [−8, 8], 간격 h). (ν, σ) 마다 캐시한다."""
    key = (float(nu), float(sig))
    if key not in _GCACHE:
        if len(_GCACHE) > 4096:
            _GCACHE.clear()
        x = np.arange(-(len(U_GRID) - 1), len(U_GRID)) * HG
        _GCACHE[key] = t_cdf(x / (float(sig) * kappa_nu(nu)), nu)
    return _GCACHE[key]


def iv_tables(p_c, nu, levels=SIG_LEVELS, taus=LC.TAUS):
    """단 (iv)의 분위 표 (32, T): σ 수준 l 마다 u = c + σ_l·κ_ν·T_ν 의 분위. p_c 는 u 격자 질량(1601점)."""
    p_c = np.asarray(p_c, float)
    L = len(p_c)
    out = np.empty((len(levels), len(taus)))
    for l, sg in enumerate(levels):
        g = _g_t(nu, sg)
        conv = fftconvolve(p_c, g)
        F = np.clip(np.maximum.accumulate(conv[L - 1:L - 1 + len(U_GRID)]), 0.0, 1.0)
        out[l] = q_from_cdf(F, U_GRID, taus)
    return out


def iv_cell_quantiles(Qtab, sigma):
    """σ 수준 표 (32, T)를 셀의 log σ_i 로 선형 보간 → (T, n)."""
    ls = np.log(SIG_LEVELS)
    x = np.log(np.clip(np.asarray(sigma, float), LC.SIGMA_CLIP[0], LC.SIGMA_CLIP[1]))
    pos = np.interp(x, ls, np.arange(len(ls), dtype=float))
    l0 = np.clip(np.floor(pos).astype(int), 0, len(ls) - 2)
    f = (pos - l0)[:, None]
    return (Qtab[l0] * (1.0 - f) + Qtab[l0 + 1] * f).T


def _b_nodes(tau_b2, n_b=N_BGRID):
    """블록 효과 구적 점과 로그 가중: b ∈ [−4τ_b, 4τ_b] 101점, 정규 가중(합 1). τ_b 가 매우 작으면 b = 0 한 점."""
    tb = float(np.sqrt(max(float(tau_b2), 0.0)))
    if tb < 1e-8:
        return np.zeros(1), np.zeros(1)
    b = np.linspace(-4.0 * tb, 4.0 * tb, int(n_b))
    lw = -b * b / (2.0 * tb * tb)
    return b, lw - logsumexp(lw)


def iv_label_loglik(u_L, blk_L, sig_L, tau_b2, nu, n_b=N_BGRID):
    """단 (iv)의 라벨 로그 우도 (801,): Σ_j log ∫ N(b; 0, τ_b²)·Π_(i∈j) f_i(u_i − a − b) db. f_i = t_ν(e/(σ_i κ_ν))/(σ_i κ_ν)."""
    u = np.asarray(u_L, float); sg = np.asarray(sig_L, float); blk = np.asarray(blk_L)
    ok = np.isfinite(u) & np.isfinite(sg) & (sg > 0)
    tot = np.zeros(len(A_GRID))
    if not ok.any():
        return tot
    b, lw = _b_nodes(tau_b2, n_b)
    kap = kappa_nu(nu)
    base = A_GRID[:, None] + b[None, :]
    for bj in np.unique(blk[ok]):
        S = np.zeros(base.shape)
        for i in np.where(ok & (blk == bj))[0]:
            sc = sg[i] * kap
            S += t_logpdf((u[i] - base) / sc, nu) - np.log(sc)
        tot += logsumexp(S + lw[None, :], axis=1)
    return tot


def v_label_loglik(u_L, blk_L, logf, tau_b2, n_b=N_BGRID):
    """단 (v)의 라벨 로그 우도 (801,). logf (n_L, 801): ε 격자(간격 0.01)의 셀별 로그 밀도 표. 선형 보간, 격자 밖은 LOGF_FLOOR."""
    u = np.asarray(u_L, float); blk = np.asarray(blk_L); lf = np.asarray(logf, float)
    ok = np.isfinite(u)
    tot = np.zeros(len(A_GRID))
    if not ok.any():
        return tot
    b, lw = _b_nodes(tau_b2, n_b)
    base = A_GRID[:, None] + b[None, :]
    for bj in np.unique(blk[ok]):
        S = np.zeros(base.shape)
        for i in np.where(ok & (blk == bj))[0]:
            row = np.where(np.isfinite(lf[i]), lf[i], LOGF_FLOOR)
            S += np.interp(u[i] - base, EPS_GRID, row, left=LOGF_FLOOR, right=LOGF_FLOOR)
        tot += logsumexp(S + lw[None, :], axis=1)
    return tot


def post_from_loglik(p_a, loglik):
    lp = _safe_log(p_a) + np.asarray(loglik, float)
    mx = np.max(lp[np.isfinite(lp)]) if np.isfinite(lp).any() else 0.0
    p = np.exp(lp - mx)
    p[~np.isfinite(p)] = 0.0
    return p / p.sum()


def coarsen2(p_u):
    """u 격자(간격 0.005, 1601점) 질량 → ε 격자(간격 0.01, 801점) 질량. 홀수 칸은 양옆에 반씩 나눈다."""
    p = np.asarray(p_u, float)
    out = p[0::2].copy()
    odd = p[1::2]
    out[:-1] += odd / 2.0
    out[1:] += odd / 2.0
    return out


def v_quantiles(Fe, p01, taus=LC.TAUS, chunk=2048):
    """셀별 ε CDF 표 Fe (n, 801)(ε 격자)와 c 의 질량 p01 (801,)(같은 격자)로 u = c + ε 의 분위 (T, n).
    F_u(u_k) = Σ_i p_i F_ε(u_k − c_i). 격자 밖의 ε 는 CDF 0(아래)과 1(위)로 둔다. u 범위는 [−4, 4] 다."""
    Fe = np.asarray(Fe, float)
    n = Fe.shape[0]
    G = len(EPS_GRID)
    out = np.empty((len(taus), n))
    p01 = np.asarray(p01, float)[None, :]
    for s0 in range(0, n, chunk):
        blk = Fe[s0:s0 + chunk]
        m = blk.shape[0]
        Fx = np.concatenate([np.zeros((m, 400)), blk, np.ones((m, 400))], axis=1)
        conv = fftconvolve(Fx, p01, axes=1)[:, 800:800 + G]
        F = np.clip(np.maximum.accumulate(conv, axis=1), 0.0, 1.0)
        out[:, s0:s0 + m] = q_from_cdf(F, EPS_GRID, taus).T
    return out


# ================================================================ 5. 초모수(계획서 3.2절)
def block_table(region, col, u, mask=None):
    """셀 단위 (지역, 전역 블록 열, u) → 블록 표 dict(region, col, n, su, su2). u 비유한 셀과 mask 거짓 셀은 뺀다. 지역·열 순서로 정렬."""
    region = np.asarray(region).astype(str); col = np.asarray(col, np.int64); u = np.asarray(u, float)
    ok = np.isfinite(u)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if not ok.any():
        return dict(region=np.zeros(0, dtype=str), col=np.zeros(0, np.int64), n=np.zeros(0), su=np.zeros(0), su2=np.zeros(0))
    d = pd.DataFrame(dict(region=region[ok], col=col[ok], u=u[ok], u2=u[ok] ** 2))
    g = d.groupby(["region", "col"], sort=True).agg(n=("u", "size"), su=("u", "sum"), su2=("u2", "sum")).reset_index()
    return dict(region=g.region.values.astype(str), col=g.col.values.astype(np.int64), n=g.n.values.astype(float),
                su=g.su.values.astype(float), su2=g.su2.values.astype(float))


def bt_subset(bt, keep):
    keep = np.asarray(keep, bool)
    return {k: np.asarray(v)[keep] for k, v in bt.items()}


def hyper_from_blocks(region, n, sum_u, sum_u2, mult=None, weight_mode="gls", sigma_pool="region", fixed=None):
    """원천 블록 표의 적률 추정(계획서 3.2절). mult 는 블록 다중도(2단 재표집. 블록이 m 번 뽑히면 서로 다른 블록 m 개로 본다).

    σ_k² = Σ_j SSW_kj / Σ_j (n_kj − 1). σ² = 자유도 10 이상 지역의 σ_k² 평균(지역 등가중. sigma_pool='cell' 이면 셀 합동).
    τ_(b,k)² = max(Var_j(ū_kj) − σ²·mean_j(1/n_kj), 1e-4)(블록 5개 이상 지역, Var 는 ddof 1). τ_b² = 그 평균.
    지역 오프셋: gls w_kj = 1/(τ_b² + σ²/n_kj), ā_k = Σ w ū / Σ w, v_k = 1/Σ w. cell = 셀 평균(v_k = τ_b²·Σn²/N² + σ²/N).
    beq = 블록 평균의 평균(v_k = Σ(τ_b² + σ²/n)/J²). τ_r² = max(mean_k(ā_k² − v_k), 1e-4).
    단 (i): σ_1² = 지역 셀 평균 기준 분산의 지역 등가중 평균, ā_k^cell = 지역 셀 평균, τ_r1² = max(mean_k((ā_k^cell)² − σ_1²/N_k), 1e-4).
    fixed = (σ², τ_b²)이면 그 값을 쓰고 지역 오프셋만 계산한다(기기별 척도 민감도). 해당 지역이 없으면 셀 합동 값으로 두고 flags 에 적는다.
    """
    if weight_mode not in ("gls", "cell", "beq"):
        raise ValueError(f"weight_mode {weight_mode}")
    if sigma_pool not in ("region", "cell"):
        raise ValueError(f"sigma_pool {sigma_pool}")
    region = np.asarray(region).astype(str); n = np.asarray(n, float)
    su = np.asarray(sum_u, float); su2 = np.asarray(sum_u2, float)
    m = np.ones(len(n)) if mult is None else np.asarray(mult, float)
    flags = []
    with np.errstate(invalid="ignore", divide="ignore"):
        ub = np.where(n > 0, su / np.where(n > 0, n, 1.0), np.nan)
        ssw = np.maximum(su2 - n * ub * ub, 0.0)
    regs = [k for k in sorted(set(region.tolist())) if float(m[region == k].sum()) > 0]
    K = len(regs)
    nanres = dict(sigma2=np.nan, sigma_k2=[], df_k=[], tau_b2=np.nan, tau_bk2=[], a_k=[], v_k=[], tau_r2=np.nan, sigma1_2=np.nan,
                  a_cell=[], tau_r1_2=np.nan, K=0, regions=[], n_blocks=[], n_cells=[], flags=["no_regions"])
    if K == 0:
        return nanres
    sel = {k: (region == k) & (m > 0) for k in regs}
    df_k = np.array([float((m[sel[k]] * (n[sel[k]] - 1)).sum()) for k in regs])
    ss_k = np.array([float((m[sel[k]] * ssw[sel[k]]).sum()) for k in regs])
    with np.errstate(invalid="ignore", divide="ignore"):
        sk2 = np.where(df_k > 0, ss_k / np.where(df_k > 0, df_k, 1.0), np.nan)
    d_all, s_all = float(df_k.sum()), float(ss_k.sum())
    s_cell = s_all / d_all if d_all > 0 else np.nan
    if sigma_pool == "cell":
        sigma2 = s_cell
    else:
        use = df_k >= 10
        if use.any():
            sigma2 = float(np.mean(sk2[use]))
        else:
            sigma2 = s_cell
            flags.append("sigma2_cellpool")
    if not np.isfinite(sigma2):
        sigma2 = 1e-4
        flags.append("sigma2_undef")
    sigma2 = max(float(sigma2), 1e-4)
    J_k = np.array([float(m[sel[k]].sum()) for k in regs])
    mu_k = np.array([float((m[sel[k]] * ub[sel[k]]).sum()) / J_k[i] for i, k in enumerate(regs)])
    tbk = np.full(K, np.nan)
    for i, k in enumerate(regs):
        if J_k[i] >= 5:
            mm, uu, nn = m[sel[k]], ub[sel[k]], n[sel[k]]
            var = float((mm * (uu - mu_k[i]) ** 2).sum()) / (J_k[i] - 1.0)
            tbk[i] = max(var - sigma2 * float((mm / nn).sum()) / J_k[i], 1e-4)
    if np.isfinite(tbk).any():
        tau_b2 = float(np.nanmean(tbk))
    else:
        num = den = inv = cnt = 0.0
        for i, k in enumerate(regs):
            if J_k[i] >= 2:
                mm, uu, nn = m[sel[k]], ub[sel[k]], n[sel[k]]
                num += float((mm * (uu - mu_k[i]) ** 2).sum()); den += J_k[i] - 1.0
                inv += float((mm / nn).sum()); cnt += J_k[i]
        tau_b2 = max(num / den - sigma2 * inv / cnt, 1e-4) if den > 0 else 1e-4
        flags.append("tau_b2_pooled")
    if fixed is not None:
        sigma2, tau_b2 = max(float(fixed[0]), 1e-4), max(float(fixed[1]), 1e-4)
        flags.append("fixed_sigma_tau")
    a_k, v_k = np.empty(K), np.empty(K)
    for i, k in enumerate(regs):
        mm, uu, nn = m[sel[k]], ub[sel[k]], n[sel[k]]
        if weight_mode == "gls":
            w = mm / (tau_b2 + sigma2 / nn)
            a_k[i] = float((w * uu).sum() / w.sum()); v_k[i] = float(1.0 / w.sum())
        elif weight_mode == "cell":
            N = float((mm * nn).sum())
            a_k[i] = float((mm * nn * uu).sum()) / N; v_k[i] = tau_b2 * float((mm * nn ** 2).sum()) / N ** 2 + sigma2 / N
        else:
            J = float(mm.sum())
            a_k[i] = float((mm * uu).sum()) / J; v_k[i] = float((mm * (tau_b2 + sigma2 / nn)).sum()) / J ** 2
    tau_r2 = max(float(np.mean(a_k ** 2 - v_k)), 1e-4)
    N_k = np.array([float((m[sel[k]] * n[sel[k]]).sum()) for k in regs])
    a_cell = np.array([float((m[sel[k]] * su[sel[k]]).sum()) for k in regs]) / N_k
    S2_k = np.array([float((m[sel[k]] * su2[sel[k]]).sum()) for k in regs])
    with np.errstate(invalid="ignore", divide="ignore"):
        var_k = np.where(N_k >= 2, (S2_k - N_k * a_cell ** 2) / np.maximum(N_k - 1.0, 1.0), np.nan)
    sigma1_2 = max(float(np.nanmean(var_k)), 1e-4) if np.isfinite(var_k).any() else 1e-4
    tau_r1_2 = max(float(np.mean(a_cell ** 2 - sigma1_2 / N_k)), 1e-4)
    return dict(sigma2=float(sigma2), sigma_k2=sk2.tolist(), df_k=df_k.tolist(), tau_b2=float(tau_b2), tau_bk2=tbk.tolist(),
                a_k=a_k.tolist(), v_k=v_k.tolist(), tau_r2=float(tau_r2), sigma1_2=float(sigma1_2), a_cell=a_cell.tolist(),
                tau_r1_2=float(tau_r1_2), K=int(K), regions=list(regs), n_blocks=J_k.tolist(), n_cells=N_k.tolist(), flags=flags)


def hyper_bt(bt, **kw):
    return hyper_from_blocks(bt["region"], bt["n"], bt["su"], bt["su2"], **kw)


def cell_offsets(region_c, col_c, bt, hyp):
    """셀별 (ā_k, b̂_kj). b̂_kj = τ_b²/(τ_b² + σ²/n_kj)·(ū_kj − ā_k). 블록 표(bt)와 초모수(hyp)는 같은 u 기준이다. 표에 없는 셀은 NaN."""
    region_c = np.asarray(region_c).astype(str); col_c = np.asarray(col_c, np.int64)
    amap = dict(zip(hyp["regions"], hyp["a_k"]))
    a_c = np.array([amap.get(r, np.nan) for r in region_c], float)
    key_bt = pd.MultiIndex.from_arrays([bt["region"], bt["col"]])
    pos = key_bt.get_indexer(pd.MultiIndex.from_arrays([region_c, col_c]))
    n_c = np.where(pos >= 0, np.asarray(bt["n"], float)[np.maximum(pos, 0)], np.nan)
    ub_c = np.where(pos >= 0, (np.asarray(bt["su"], float) / np.maximum(np.asarray(bt["n"], float), 1.0))[np.maximum(pos, 0)], np.nan)
    tb2, s2 = float(hyp["tau_b2"]), float(hyp["sigma2"])
    return a_c, tb2 / (tb2 + s2 / n_c) * (ub_c - a_c)


# ================================================================ 6. emu 단위(대상, 모드)의 계산
def cb_fit(X, y, w, seed, threads):
    """CatBoost(catboost_lo 설정) 회귀. 스레드는 --threads(로컬 규칙: 합계 8 이하)."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**CB_PARAMS, random_seed=int(seed), verbose=0, allow_writing_files=False, thread_count=int(threads))
    m.fit(np.asarray(X, float), np.asarray(y, float), sample_weight=None if w is None else np.asarray(w, float))
    return m


def cb_pred(m, X, threads):
    return np.asarray(m.predict(np.asarray(X, float), thread_count=int(threads)), float)


def iv_scale_fit(fr, t_idx, gi, g_reg, g_col, g_u, bt, hyp, groups, seeds, threads):
    """단 (iv)의 척도 모형(계획서 3.3절). 목표 log(|ẽ| + 0.01), ẽ = u − ā_k − b̂_kj, 입력 x25 원값, 행 가중 LC.row_weights(블록).
    seed 마다: 집단 지역 하나 제외 적합으로 표본 밖 ĥ, c² = 지역 등가중 평균_k[mean((ẽ/exp(ĥ_oos))²)], ν_ε = 표본 밖 로그 우도(지역 등가중)
    최대(동률이면 큰 ν), 전체 적합으로 대상 셀(t_idx 순서)의 ĥ. 반환 seed 별 dict(seed, h_t, c, nu, ll, n_fit, sec, fail)."""
    a_c, b_c = cell_offsets(g_reg, g_col, bt, hyp)
    e = np.asarray(g_u, float) - a_c - b_c
    ok = np.isfinite(e)
    gi2, reg2, col2, e2 = gi[ok], np.asarray(g_reg)[ok], np.asarray(g_col)[ok], e[ok]
    ytar = np.log(np.abs(e2) + 0.01)
    X, Xt, w = fr.X[gi2], fr.X[np.asarray(t_idx, np.int64)], LC.row_weights(col2)
    out = []
    for seed in seeds:
        t0 = time.time()
        rec = dict(seed=int(seed), h_t=None, c=np.nan, nu=np.nan, ll=[np.nan] * len(NU_EPS_CAND), n_fit=0, sec=0.0, fail="")
        try:
            h_oos = np.full(len(gi2), np.nan)
            for k in groups:
                tr = reg2 != k
                if tr.sum() == 0 or (~tr).sum() == 0:
                    continue
                mdl = cb_fit(X[tr], ytar[tr], w[tr], seed, threads); rec["n_fit"] += 1
                h_oos[~tr] = cb_pred(mdl, X[~tr], threads)
            regs = [k for k in groups if np.isfinite(h_oos[reg2 == k]).any()]
            c2 = float(np.mean([np.nanmean((e2[reg2 == k] / np.exp(h_oos[reg2 == k])) ** 2) for k in regs]))
            c = float(np.sqrt(c2))
            sig = np.clip(c * np.exp(h_oos), LC.SIGMA_CLIP[0], LC.SIGMA_CLIP[1])
            ll = []
            for nu in NU_EPS_CAND:
                kap = kappa_nu(nu)
                per = [np.nanmean(t_logpdf(e2[reg2 == k] / (sig[reg2 == k] * kap), nu) - np.log(sig[reg2 == k] * kap)) for k in regs]
                ll.append(float(np.mean(per)))
            best = None
            for j, nu in enumerate(NU_EPS_CAND):                          # 오름차순: 동률이면 뒤의 큰 ν 가 이긴다
                if best is None or ll[j] >= ll[best] - 1e-12:
                    best = j
            full = cb_fit(X, ytar, w, seed, threads); rec["n_fit"] += 1
            rec.update(h_t=cb_pred(full, Xt, threads), c=c, nu=float(NU_EPS_CAND[best]), ll=ll)
        except Exception as ex:                                            # noqa: BLE001  seed 하나의 실패는 그 키만 실패로 둔다
            if LC.is_env_error(ex):                                        # 환경·자원 오류는 emu 조각 전체를 실패로 둔다
                raise
            rec["fail"] = f"{type(ex).__name__}: {str(ex)[:240]}"
        rec["sec"] = round(time.time() - t0, 2)
        out.append(rec)
    return out


def iiic_calibration(fr, src, groups, gi, g_reg, recs, en_u, lab_u, lab_col, n_grid, cfg):
    """단 (iii+c)의 후보정 폭 c_n(α)(계획서 3.3절 단 vi). 의사 대상 p 마다 G 에서 p 를 뺀 집단(u 는 E0^(−p,−q) 기준)으로 초모수와
    τ_r 사후를 다시 추정하고, 모사 기록의 라벨로 단 (iii) 구간 [q_lo, q_hi]를 만든다. 점수 max(q_lo − u, u − q_hi)의 지역 등가중
    (1 − α) 분위가 c_n(α)다. 반환 (cn (len(n_grid), 2), nest 표 dict, 정보 dict)."""
    y, s, mac = fr.y, fr.s, fr.mac
    nest = dict(p=[], q=[], col=[], n=[], sum=[], sum2=[])
    per_n = {int(n): ([], [], []) for n in n_grid}
    info = {}
    for p in groups:
        others = [q for q in groups if q != p]
        if not others:
            continue
        rl, cl, ul = [], [], []
        for q in others:
            rest = src[(mac[src] != p) & (mac[src] != q)]
            Epq = LC.ls_E(y[rest], s[rest])
            iq = gi[g_reg == q]
            rl.append(np.full(len(iq), q)); cl.append(fr.col[iq]); ul.append(fr.u_of(iq, Epq))
        btp = block_table(np.concatenate(rl), np.concatenate(cl), np.concatenate(ul))
        for k_, v_ in (("q", btp["region"]), ("col", btp["col"]), ("n", btp["n"]), ("sum", btp["su"]), ("sum2", btp["su2"])):
            nest[k_].append(np.asarray(v_))
        nest["p"].append(np.full(len(btp["n"]), p))
        hp = hyper_bt(btp)
        if hp["K"] == 0:
            info[p] = "no_regions"
            continue
        tg, pt = tau_r_posterior(hp["a_k"], hp["v_k"], cfg.nu, cfg.s_tau)
        pa = prior_a(tg, pt, cfg.nu)
        sd = float(np.sqrt(hp["tau_b2"] + hp["sigma2"]))
        info[p] = dict(K=hp["K"], sigma2=hp["sigma2"], tau_b2=hp["tau_b2"], tau_r2=hp["tau_r2"])
        for r in recs:
            if r["region"] != p or int(r["n"]) not in per_n:
                continue
            nJ, ub = label_blocks(lab_u[r["lab_start"]:r["lab_stop"]], lab_col[r["lab_start"]:r["lab_stop"]])
            q = predictive_q(posterior_grid(pa, nJ, ub, hp["tau_b2"], hp["sigma2"]), sd, (0.05, 0.10, 0.90, 0.95))
            eu = en_u[r["start"]:r["stop"]]
            eu = eu[np.isfinite(eu)]
            a10, a20, rg = per_n[int(r["n"])]
            a10.append(np.maximum(q[0] - eu, eu - q[3])); a20.append(np.maximum(q[1] - eu, eu - q[2])); rg.append(np.full(len(eu), p))
    cn = np.full((len(n_grid), 2), np.nan)
    for i, n in enumerate(n_grid):
        a10, a20, rg = per_n[int(n)]
        if rg and sum(len(v) for v in rg):
            reg = np.concatenate(rg)
            cn[i, 0] = LC.hier_quantiles(np.concatenate(a10), reg, [0.90])[0]
            cn[i, 1] = LC.hier_quantiles(np.concatenate(a20), reg, [0.80])[0]
    nest = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in nest.items()}
    return cn, nest, info


def emu_compute(fr, t_idx, src_idx, target, mode, cfg, do_iv=False):
    """emu 단위: 원천 블록 표(주, 탐침, early 제외), 초모수(주 분석과 민감도), a 사전(변형별), h4 사전, h37 점수, 의사 대상 모사 항목,
    단 (iii+c)의 c_n(α), 단 (iv)의 척도 적합(do_iv). 초모수와 보정은 대상과 100 km 버퍼를 뺀 원천(src_idx)에서만 구한다.
    반환 dict(배열..., meta=dict)."""
    t0 = time.time()
    y, s, mac, uok, col = fr.y, fr.s, fr.mac, fr.uok, fr.col
    early, probe = fr.flags["early"], fr.flags["probe"]
    t_idx = np.asarray(t_idx, np.int64); src = np.asarray(src_idx, np.int64)
    E0 = LC.ls_E(y[src], s[src])
    if not (np.isfinite(E0) and E0 > 0):
        raise RuntimeError(f"{target}|{mode}: 원천 계수 E0 가 정의되지 않는다")
    logE0 = float(np.log(E0))
    src_ok = src[uok[src]]
    groups = LC.cal_groups(mac, src, uok, cfg.min_cal_cells)            # 로그 비가 정의되는 원천 셀 20개 이상(h45 와 같은 정의)
    if len(groups) < 2:
        raise RuntimeError(f"{target}|{mode}: 보정 집단 지역이 {len(groups)}개다(2개 미만)")
    E0_mk = {}
    for k in groups:
        rest = src[mac[src] != k]
        E0_mk[k] = LC.ls_E(y[rest], s[rest])
    gi = src_ok[np.isin(mac[src_ok], groups)]
    g_reg, g_col = mac[gi], col[gi]
    g_u = fr.u_of(gi, np.array([E0_mk[k] for k in g_reg], float))
    bt = block_table(g_reg, g_col, g_u)
    bt_p = block_table(g_reg, g_col, g_u, probe[gi])
    bt_ne = block_table(g_reg, g_col, g_u, ~early[gi])
    flags = []
    hyp = {"main": hyper_bt(bt), "w_cell": hyper_bt(bt, weight_mode="cell"), "w_beq": hyper_bt(bt, weight_mode="beq"),
           "sig_pool": hyper_bt(bt, sigma_pool="cell")}
    hp = hyper_bt(bt_p)
    if hp["K"] > 0:
        hyp["probe_sig"] = hyper_bt(bt, fixed=(hp["sigma2"], hp["tau_b2"]))
        hyp["probe_sig"]["probe_only"] = dict(sigma2=hp["sigma2"], tau_b2=hp["tau_b2"], K=hp["K"], flags=hp["flags"])
    else:
        flags.append("probe_sig:no_probe_blocks")
    hn = hyper_bt(bt_ne)
    if hn["K"] > 0:
        hyp["noearly"] = hn
    else:
        flags.append("noearly:no_blocks")
    trunc = None
    if "phys" in cfg.SENS:
        try:
            ls = LC.phys_log_samples(fr.df.iloc[src], fr.df.iloc[src], PHYS_DRAWS, 0, unit_tdd=True)
            v = ls[np.isfinite(ls)]
            lo, hi = np.percentile(v, [1.0, 99.0])
            trunc = (float(lo - logE0), float(hi - logE0))
        except Exception as ex:                                            # noqa: BLE001  물리 입력 오류는 그 변형만 뺀다
            if LC.is_env_error(ex):
                raise
            flags.append(f"phys:{type(ex).__name__}:{str(ex)[:120]}")
    arr = {}
    tq = {}
    for v in ["main"] + [x for x in cfg.SENS]:
        hname, nu_v, st_v, tr = VARIANT_SPEC[v]
        if hname not in hyp or (tr and trunc is None):
            continue
        h = hyp[hname]
        nu = float(nu_v if nu_v is not None else cfg.nu); st = float(st_v if st_v is not None else cfg.s_tau)
        tg, pt = tau_r_posterior(h["a_k"], h["v_k"], nu, st)
        arr[f"prior__{v}"] = prior_a(tg, pt, nu, trunc if tr else None)
        arr[f"ptau__{v}"] = pt
        tq[v] = tau_quantiles(tg, pt)
    prior_h4 = None
    try:
        pr = offset_mle_prior(pd.DataFrame(dict(macro=mac[src], y=y[src], s=s[src])), region_col="macro", min_cells=3, y_col="y",
                              s_col="s", logE0=logE0)
        prior_h4 = dict(logE0=float(pr["logE0"]), tau2=float(pr["tau2"]), sigma2=float(pr["sigma2"]), n_regions=int(pr["n_regions"]))
    except ValueError as ex:
        flags.append(f"i0:{str(ex)[:120]}")
    # 의사 대상 모사(LC.emulation_plan). 채점 항목은 로그 비가 정의되는 셀만 둔다. E_center 는 --kappa 로 다시 계산한다.
    plan = LC.emulation_plan(fr.df, src, groups, [int(n) for n in cfg.N_GRID], target, mode)
    recs, en_idx, en_u, lab_idx, lab_u = [], [], [], [], []
    ne = nl = 0
    for r in plan:
        k, Ek = str(r["region"]), float(r["E0_mk"])
        cal = np.asarray(r["cal_idx"], np.int64)
        cal = cal[uok[cal]]
        lab = np.asarray(r["lab_idx"], np.int64)
        Ec = LC.center_coef(y[lab], s[lab], Ek, cfg.kappa) if int(r["n"]) > 0 else Ek
        lab_ne = lab[~early[lab]]
        Ec_ne = LC.center_coef(y[lab_ne], s[lab_ne], Ek, cfg.kappa) if int(r["n"]) > 0 else Ek
        recs.append(dict(region=k, n=int(r["n"]), e=int(r["e"]), d=int(r["d"]), E0_mk=Ek, E_center=float(Ec), E_center_ne=float(Ec_ne),
                         n_lab_ne=int(len(lab_ne)), delta=float(np.log(Ec / Ek)), start=ne, stop=ne + len(cal), lab_start=nl,
                         lab_stop=nl + len(lab)))
        ne += len(cal); nl += len(lab)
        en_idx.append(cal); en_u.append(fr.u_of(cal, Ek)); lab_idx.append(lab); lab_u.append(fr.u_of(lab, Ek))

    def cat(v, dt):
        return np.concatenate(v).astype(dt) if v else np.zeros(0, dt)
    arr.update(en_idx=cat(en_idx, np.int64), en_u=cat(en_u, float), lab_idx=cat(lab_idx, np.int64), lab_u=cat(lab_u, float))
    arr["lab_blk"] = col[arr["lab_idx"]]
    arr["en_col"] = col[arr["en_idx"]]
    rt = pd.DataFrame(recs, columns=list(EG_COLS))
    for c in rt.columns:                                                  # 빈 표에서도 pickle 없는 자료형이 되게 명시한다
        arr[f"eg_{c}"] = (rt[c].values.astype(str) if c == "region" else
                          rt[c].values.astype(np.int64 if c in ("n", "e", "d", "n_lab_ne", "start", "stop", "lab_start", "lab_stop") else float))
    cn = np.full((len(cfg.N_GRID), 2), np.nan)
    cn_info = {}
    if "iii+c" in cfg.RUNGS:
        cn, nest, cn_info = iiic_calibration(fr, src, groups, gi, g_reg, recs, arr["en_u"], arr["lab_u"], arr["lab_blk"], cfg.N_GRID, cfg)
        for k_, v_ in nest.items():
            arr[f"nest_{k_}"] = np.asarray(v_).astype(str) if k_ in ("p", "q") else np.asarray(v_)
    arr["cn"] = cn
    arr["cn_n"] = np.asarray(cfg.N_GRID, np.int64)
    iv_meta = []
    if do_iv:
        fits = iv_scale_fit(fr, t_idx, gi, g_reg, g_col, g_u, bt, hyp["main"], groups, cfg.CB_SEEDS, cfg.threads)
        S = len(fits)
        arr["iv_h"] = np.full((S, len(t_idx)), np.nan, np.float32)
        arr["iv_c"] = np.full(S, np.nan); arr["iv_nu"] = np.full(S, np.nan); arr["iv_ll"] = np.full((S, len(NU_EPS_CAND)), np.nan)
        arr["iv_seed"] = np.asarray([f["seed"] for f in fits], np.int64)
        for j, f in enumerate(fits):
            if f["h_t"] is not None:
                arr["iv_h"][j] = f["h_t"]; arr["iv_c"][j] = f["c"]; arr["iv_nu"][j] = f["nu"]; arr["iv_ll"][j] = f["ll"]
            iv_meta.append({k: f[k] for k in ("seed", "c", "nu", "ll", "n_fit", "sec", "fail")})
    arr.update(t_idx=t_idx, src_idx=src, gi=gi, g_reg=g_reg.astype(str), g_col=g_col.astype(np.int64), g_u=g_u,
               h37_s=np.abs(fr.u_of(gi, E0)))
    for nm, b in (("bt", bt), ("btp", bt_p), ("btne", bt_ne)):
        for k_, v_ in b.items():
            arr[f"{nm}_{k_}"] = np.asarray(v_).astype(str) if k_ == "region" else np.asarray(v_)
    meta = dict(format=FMT, target=target, mode=mode, E0=float(E0), logE0=logE0, groups=groups, E0_mk={k: float(v) for k, v in E0_mk.items()},
                n_src=int(len(src)), n_src_ok=int(len(src_ok)), n_group_cells=int(len(gi)), n_blocks=int(len(bt["n"])), hyper=hyp,
                tau_q=tq, prior_h4=prior_h4, phys_trunc=trunc, flags=flags, n_records=int(len(recs)), n_entries=int(ne), n_labels=int(nl),
                cn=cn.tolist(), cn_info=cn_info, iv=iv_meta, variants=[v for v in ["main"] + list(cfg.SENS) if f"prior__{v}" in arr],
                sec=round(time.time() - t0, 2))
    arr["meta"] = meta
    return arr


def save_emu(path, emu):
    arrs = {k: v for k, v in emu.items() if k != "meta"}
    arrs["meta"] = np.array(json.dumps(emu["meta"], ensure_ascii=False, default=LC._py))
    return LC.atomic_npz(path, **arrs)


def load_emu(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


class EmuView:
    """emu 단위 결과의 조회 도우미(cpu, gate, flow 단위와 2단 재표집이 쓴다)."""

    def __init__(self, emu, fr):
        self.a, self.meta, self.fr = emu, emu["meta"], fr
        self.E0 = float(self.meta["E0"]); self.logE0 = float(np.log(self.E0))
        self.groups = list(self.meta["groups"])
        self.t_idx = np.asarray(emu["t_idx"], np.int64)
        self.recs = pd.DataFrame({c: emu[f"eg_{c}"] for c in EG_COLS})
        self._b4: dict = {}
        self._h37 = None

    def hyp(self, name="main"):
        return self.meta["hyper"].get(name)

    def prior(self, v="main"):
        return self.a.get(f"prior__{v}")

    def bt(self, name="bt"):
        return {k: self.a[f"{name}_{k}"] for k in ("region", "col", "n", "su", "su2")}

    def tpos(self, idx):
        """df 행 번호 → t_idx 안의 위치(t_idx 는 정렬되어 있다)."""
        idx = np.asarray(idx, np.int64)
        p = np.searchsorted(self.t_idx, idx)
        if len(idx) and (np.any(p >= len(self.t_idx)) or np.any(self.t_idx[np.minimum(p, len(self.t_idx) - 1)] != idx)):
            raise ValueError("대상 셀이 아닌 색인이 있다")
        return p

    def b4_entries(self, n, variant=""):
        """B4 보정 항목 (점수 |u − δ|, 지역, 전역 블록 열). δ = log(E_center/E0^(−k)). variant: '' , 'probe_sig'(탐침 항목만),
        'noearly'(early 항목 제외, 라벨도 early 제외로 다시 구한 중심)."""
        r = self.recs[self.recs.n == int(n)]
        sc, rg, cl = [], [], []
        early, probe = self.fr.flags["early"], self.fr.flags["probe"]
        for rec in r.itertuples(index=False):
            idx = self.a["en_idx"][rec.start:rec.stop]; u = self.a["en_u"][rec.start:rec.stop]; c = self.a["en_col"][rec.start:rec.stop]
            Ec = rec.E_center_ne if variant == "noearly" else rec.E_center
            keep = np.isfinite(u)
            if variant == "noearly":
                keep &= ~early[idx]
            elif variant == "probe_sig":
                keep &= probe[idx]
            sc.append(np.abs(u[keep] - np.log(Ec / rec.E0_mk))); rg.append(np.full(int(keep.sum()), rec.region)); cl.append(c[keep])
        if not sc or sum(len(v) for v in sc) == 0:
            return np.zeros(0), np.zeros(0, dtype=str), np.zeros(0, np.int64)
        return np.concatenate(sc), np.concatenate(rg), np.concatenate(cl)

    def b4_q(self, n, variant=""):
        """B4 의 수준별 로그 반폭 q_n(c) (49,). 항목이 없으면 None."""
        key = (int(n), variant)
        if key not in self._b4:
            sc, rg, _ = self.b4_entries(n, variant)
            self._b4[key] = LC.hier_quantiles(sc, rg, LC.LEVELS) if len(sc) else None
        return self._b4[key]

    def h37(self):
        """hier2@h37 과 wconf 의 원천 점수 (|log y − log(E0·s)|, 지역)."""
        if self._h37 is None:
            s_ = np.asarray(self.a["h37_s"], float); g = np.asarray(self.a["g_reg"]).astype(str)
            ok = np.isfinite(s_)
            self._h37 = (s_[ok], g[ok], LC.hier_quantiles(s_[ok], g[ok], LC.LEVELS))
        return self._h37

    def cn(self, n, alpha):
        ns = list(np.asarray(self.a["cn_n"]).tolist())
        if int(n) not in ns:
            return np.nan
        return float(self.a["cn"][ns.index(int(n)), 0 if abs(float(alpha) - 0.1) < 1e-9 else 1])

    def iv(self, seed):
        """(ĥ(t_idx 순서), c, ν) 또는 None(적합 실패나 미실행)."""
        if "iv_seed" not in self.a:
            return None
        sd = list(np.asarray(self.a["iv_seed"]).tolist())
        if int(seed) not in sd:
            return None
        j = sd.index(int(seed))
        if not np.isfinite(self.a["iv_c"][j]):
            return None
        return np.asarray(self.a["iv_h"][j], float), float(self.a["iv_c"][j]), float(self.a["iv_nu"][j])


# ================================================================ 7. cpu 단위(대상, 모드, 분할)의 계산
class Scope:
    """채점 셀 집합 하나와 그 저장소. A = E0·s(로그 비의 기준), u = log y − log A."""

    def __init__(self, fr, idx, store_name, split, E0, scope):
        self.idx = np.asarray(idx, np.int64)
        self.y, self.s = fr.y[self.idx], fr.s[self.idx]
        self.col = fr.col[self.idx]
        self.A = float(E0) * self.s
        self.u = np.log(self.y) - np.log(self.A)
        self.scope = scope
        self.flags = {k: fr.flags[k][self.idx] for k in STRATA}
        self.store = LC.ScoreStore(store_name, split, self.col.astype(str), self.col, meta=dict(scope=scope, n_cells=int(len(self.idx))))


def _grid_scores(sc, qu):
    """u 척도 분위 qu (99,) 또는 (99, n) → cm 격자 → LC.score_cells. 반환 (점수, 재배열 전 교차 셀 비율)."""
    qu = np.asarray(qu, float)
    Q = sc.A[None, :] * np.exp(qu[:, None] if qu.ndim == 1 else qu)
    cross = LC.crossing_stats(Q)["frac_cells"]
    return LC.score_cells(sc.y, LC.rearrange(Q)), cross


def _point_scores(sc, pred):
    pred = np.asarray(pred, float)
    return dict(se=(pred - sc.y) ** 2, ae=np.abs(pred - sc.y), bias=pred - sc.y, valid=np.isfinite(sc.y))


def _add(sc, key, scores, rows, base, info, valid=None, cross=np.nan, flag="", sec=0.0):
    """저장소에 넣고 runs 행을 만든다(지표는 셀 가중 평균)."""
    key = LC.norm_key(key)
    v = np.ones(len(sc.y), bool) if valid is None else np.asarray(valid, bool)
    if not v.any():
        return
    sc.store.add(key, scores, valid=v)
    _, cnt = sc.store.get(key)
    row = dict(base, scope=sc.scope, store=sc.store.target, method=key[0], variant=key[1], n=key[2], draw=key[3], seed=key[4],
               n_eval=int(v.sum()), n_blocks=int((cnt > 0).sum()))
    row.update(info)
    for m in LC.METRICS:
        row[m] = sc.store.mean(key, m, "cell")
    row["rmse"] = sc.store.rmse(key, "cell"); row["rmse_beq"] = sc.store.rmse(key, "beq")
    row["inf"] = row["inf10"]; row["cross_frac"] = cross; row["flag"] = flag; row["sec"] = round(float(sec), 3)
    pit = scores.get("pit")
    hist = LC.pit_hist(np.asarray(pit)[v]) if pit is not None else np.full(LC.PIT_BINS, -1)
    for j, c in enumerate(hist):
        row[f"pit{j}"] = int(c)
    rows.append(row)


class UnitCtx:
    """단위 계산의 공통 상태(자료 보기, emu, 설정)."""

    def __init__(self, fr, ev, cfg, target, mode):
        self.fr, self.ev, self.cfg, self.target, self.mode = fr, ev, cfg, target, mode
        self.rungs = set(rungs_for(cfg, target, mode))
        self.sens = [v for v in cfg.SENS if ev.prior(v) is not None or v in B4_SENS]
        self.E0, self.logE0 = ev.E0, ev.logE0
        self.prior_h4 = ev.meta.get("prior_h4")


def _labels(U, lab):
    """라벨 셀의 (u, 블록 열, n_j, ū_j, 라벨 수, 블록 수, E_n)."""
    fr = U.fr
    lab = np.asarray(lab, np.int64)
    u_L = fr.u_of(lab, U.E0)
    col_L = fr.col[lab]
    nJ, ub = label_blocks(u_L, col_L)
    E_n = LC.center_coef(fr.y[lab], fr.s[lab], U.E0, U.cfg.kappa)
    return u_L, col_L, nJ, ub, int(np.isfinite(u_L).sum()), int(len(nJ)), float(E_n)


def _info(U, n_lab, nb_lab, E_n, h=None, apost=(np.nan, np.nan), nu=np.nan, c=np.nan):
    h = h or {}
    return dict(n_lab=n_lab, nb_lab=nb_lab, E0=U.E0, E_center=E_n, a_post_med=apost[0], a_post_sd=apost[1],
                sigma2=h.get("sigma2", np.nan), tau_b2=h.get("tau_b2", np.nan), tau_r2=h.get("tau_r2", np.nan), nu_eps=nu, scale_c=c)


def iii_quantiles(prior, nJ, ub, h, return_post=False):
    """단 (iii)(또는 그 변형)의 예측 분위 (99,)와 a 사후 요약. return_post 이면 a 사후 질량 (801,)을 네 번째로 돌려준다."""
    post = posterior_grid(prior, nJ, ub, h["tau_b2"], h["sigma2"])
    sd = float(np.sqrt(h["tau_b2"] + h["sigma2"]))
    out = (predictive_q(post, sd), post_stats(post), ("sd<2h" if sd < 2 * HG else ""))
    return out + (post,) if return_post else out


def _aux_points(sc, method, n, d, rows, base, info, qu=None, E_mean=None):
    """보조 점 예측 열(계획서 2.8절, 3.7절. 판정에는 쓰지 않는다). 변형 'aux:tmean' = 예측 분포의 1–99 % 절단 평균
    (u 척도 분위 격자 τ 0.01–0.99 의 exp 를 사다리꼴로 적분해 0.98 로 나눈 값 × A, cm). 변형 'aux:Emean' = 사후 평균 E
    (E0·E[exp(a) | 라벨])·s. 두 변형 모두 점 예측 지표(se, ae, bias)만 저장한다."""
    if qu is not None:
        eq = np.exp(np.asarray(qu, float))
        tm = (eq[0] / 2.0 + eq[1:-1].sum() + eq[-1] / 2.0) / (len(eq) - 1)
        _add(sc, (method, "aux:tmean", n, d, -1), _point_scores(sc, sc.A * tm), rows, base, info)
    if E_mean is not None and np.isfinite(E_mean):
        _add(sc, (method, "aux:Emean", n, d, -1), _point_scores(sc, float(E_mean) * sc.s), rows, base, info)


def run_ladder(U, sc, n, d, lab, rows, base, kind="main", fail=None):
    """라벨 n개(추출 d)의 사다리 전체를 채점 범위 sc 에 계산한다. kind: 'main'(주 분석과 민감도, 층), 'all'(범위 all 참조 행),
    'noearly'(early 제외 변형: B4·iii 만)."""
    fail = fail if fail is not None else []
    rg = U.rungs
    u_L, col_L, nJ, ub, n_lab, nb_lab, E_n = _labels(U, lab)
    hm = U.ev.hyp("noearly" if kind == "noearly" else "main")
    var = "noearly" if kind == "noearly" else ""

    def strata(key, scores, info, cross=np.nan, flag=""):
        if kind != "main":
            return
        for st in STRATA:
            m = sc.flags[st]
            if m.any():
                _add(sc, (key[0], f"#{st}", key[2], key[3], key[4]), scores, rows, base, info, valid=m, cross=cross, flag=flag)

    def guard(name, fn):
        t0 = time.time()
        try:
            fn(t0)
        except Exception as ex:                                            # noqa: BLE001  한 방법의 실패는 그 키만 실패로 둔다
            if LC.is_env_error(ex):                                        # 환경·자원 오류는 조각 전체를 실패로 둔다(--resume 이 다시 한다)
                raise
            fail.append(dict(method=name, n=int(n), draw=int(d), kind=kind, error=f"{type(ex).__name__}: {str(ex)[:240]}"))

    if kind == "main":
        if "P0" in rg and n == 0:
            guard("P0", lambda t0: _add(sc, ("P0", "", 0, 0, -1), _point_scores(sc, U.E0 * sc.s), rows, base,
                                        _info(U, 0, 0, U.E0), sec=time.time() - t0))
        if "P1" in rg:
            guard("P1", lambda t0: _add(sc, ("P1", "", n, d, -1), _point_scores(sc, E_n * sc.s), rows, base,
                                        _info(U, n_lab, nb_lab, E_n), sec=time.time() - t0))
    if kind == "all":
        if "P0" in rg:
            guard("P0", lambda t0: _add(sc, ("P0", "", 0, 0, -1), _point_scores(sc, U.E0 * sc.s), rows, base,
                                        _info(U, 0, 0, U.E0), sec=time.time() - t0))

    def b4(variant):
        def f(t0):
            q = U.ev.b4_q(n, variant)
            if q is None:
                fail.append(dict(method="B4", variant=variant, n=int(n), draw=int(d), kind=kind, error="보정 항목 없음(K_n = 0)"))
                return
            scores = LC.score_cells(sc.y, LC.grid_from_center(E_n * sc.s, q))
            info = _info(U, n_lab, nb_lab, E_n, apost=(float(np.log(E_n / U.E0)), np.nan))
            _add(sc, ("B4", variant, n, d, -1), scores, rows, base, info, sec=time.time() - t0)
            if variant == "":
                strata(("B4", "", n, d, -1), scores, info)
        return f
    if "B4" in rg:
        guard("B4", b4(var))
        if kind == "main":
            for v in B4_SENS:
                if v in U.sens and v != "noearly":
                    guard(f"B4:{v}", b4(v))
    if kind == "noearly":
        guard("iii:noearly", lambda t0: _iii(U, sc, n, d, nJ, ub, n_lab, nb_lab, E_n, "noearly", "noearly", rows, base, t0, None))
        return
    if "hier2@h37" in rg and n == 0:
        def f_h37(t0):
            q = U.ev.h37()[2]
            _add(sc, ("hier2@h37", "", 0, 0, -1), LC.score_cells(sc.y, LC.grid_from_center(U.E0 * sc.s, q)), rows, base,
                 _info(U, 0, 0, U.E0), sec=time.time() - t0)
        guard("hier2@h37", f_h37)
    if "wconf" in rg and n in WCONF_N and kind == "main":
        def f_wconf(t0):
            s_src, g_src, _ = U.ev.h37()
            _, inv, cnt = np.unique(g_src, return_inverse=True, return_counts=True)
            w_src = 1.0 / cnt[inv.ravel()]                                  # 원천 지역 총 가중 1(h37 규약)
            sL = np.abs(u_L[np.isfinite(u_L)])
            scs = np.concatenate([sL, s_src]); ws = np.concatenate([np.ones(len(sL)), w_src])
            q10 = LC.wquantile(scs, ws, 0.90, w_test=1.0); q20 = LC.wquantile(scs, ws, 0.80, w_test=1.0)
            c_ = U.E0 * sc.s
            with np.errstate(over="ignore", invalid="ignore"):
                scores = LC.score_interval_only(sc.y, c_ * np.exp(-q10), c_ * np.exp(q10), c_ * np.exp(-q20), c_ * np.exp(q20), c_)
            _add(sc, ("wconf", "", n, d, -1), scores, rows, base, _info(U, n_lab, nb_lab, U.E0), sec=time.time() - t0,
                 flag="" if np.isfinite(q10) else "q_inf")
        guard("wconf", f_wconf)
    for name, pr in (("i0", U.prior_h4), ("i", None)):
        if name not in rg or (name == "i0" and pr is None):
            continue

        def f_i(t0, name=name, pr=pr):
            h = hm
            prior = pr if name == "i0" else dict(logE0=U.logE0, tau2=h["tau_r1_2"], sigma2=h["sigma1_2"])
            zL = u_L[np.isfinite(u_L)] + U.logE0
            est = offset_mle_estimate(zL, prior)
            m, v = est["logE_post"] - U.logE0, est["logE_sd"] ** 2
            qu = normal_q(m, v + prior["sigma2"])
            scores, cross = _grid_scores(sc, qu)
            info = _info(U, n_lab, nb_lab, E_n, dict(sigma2=prior["sigma2"], tau_r2=prior["tau2"]), (m, float(np.sqrt(v))))
            _add(sc, (name, "", n, d, -1), scores, rows, base, info, cross=cross, sec=time.time() - t0)
            if kind in ("main", "all"):                                    # 보조 열: 절단 평균(단 i 만), 사후 평균 E
                _aux_points(sc, name, n, d, rows, base, info, qu=qu if name == "i" else None, E_mean=U.E0 * np.exp(m + v / 2.0))
        guard(name, f_i)
    if "ii" in rg:
        def f_ii(t0):
            m, v = posterior_normal(nJ, ub, hm["tau_r2"], hm["tau_b2"], hm["sigma2"])
            qu = normal_q(m, v + hm["tau_b2"] + hm["sigma2"])
            scores, cross = _grid_scores(sc, qu)
            info = _info(U, n_lab, nb_lab, E_n, hm, (m, float(np.sqrt(v))))
            _add(sc, ("ii", "", n, d, -1), scores, rows, base, info, cross=cross, sec=time.time() - t0)
            if kind in ("main", "all"):
                _aux_points(sc, "ii", n, d, rows, base, info, qu=qu, E_mean=U.E0 * np.exp(m + v / 2.0))
        guard("ii", f_ii)
    q_iii = {}
    if "iii" in rg or "iii+c" in rg:
        guard("iii", lambda t0: q_iii.update(main=_iii(U, sc, n, d, nJ, ub, n_lab, nb_lab, E_n, "main", "", rows, base, t0,
                                                         strata if "iii" in rg else None, store="iii" in rg)))
        if kind in ("main", "all"):
            for v in U.sens:
                if v in ("noearly",) or U.ev.prior(v) is None:
                    continue
                guard(f"iii:{v}", lambda t0, v=v: _iii(U, sc, n, d, nJ, ub, n_lab, nb_lab, E_n, v, v, rows, base, t0, None))
    if "iii+c" in rg and q_iii.get("main") is not None:
        def f_c(t0):
            qu = q_iii["main"]
            c10, c20 = U.ev.cn(n, 0.10), U.ev.cn(n, 0.20)
            if not (np.isfinite(c10) and np.isfinite(c20)):
                fail.append(dict(method="iii+c", n=int(n), draw=int(d), kind=kind, error="c_n 없음"))
                return
            A = sc.A
            lims = []
            for alpha, cc in ((0.10, c10), (0.20, c20)):
                lo, hi = qu[LC.grid_index(alpha / 2)] - cc, qu[LC.grid_index(1 - alpha / 2)] + cc
                if lo > hi:
                    lo = hi = qu[49]
                lims += [A * np.exp(lo), A * np.exp(hi)]
            scores = LC.score_interval_only(sc.y, lims[0], lims[1], lims[2], lims[3], A * np.exp(qu[49]))
            _add(sc, ("iii+c", "", n, d, -1), scores, rows, base, _info(U, n_lab, nb_lab, E_n, hm), sec=time.time() - t0,
                 flag=f"c10={c10:.4f};c20={c20:.4f}")
        guard("iii+c", f_c)
    if "iv" in rg:
        for seed in U.cfg.CB_SEEDS:
            guard(f"iv:{seed}", lambda t0, seed=seed: _iv(U, sc, n, d, lab, u_L, col_L, n_lab, nb_lab, E_n, seed, rows, base, t0, fail))


def _iii(U, sc, n, d, nJ, ub, n_lab, nb_lab, E_n, prior_name, variant, rows, base, t0, strata, store=True):
    hname = VARIANT_SPEC[prior_name][0]
    h = U.ev.hyp(hname)
    pr = U.ev.prior(prior_name)
    if h is None or pr is None:
        return None
    qu, apost, fl, post = iii_quantiles(pr, nJ, ub, h, return_post=True)
    if store:
        scores, cross = _grid_scores(sc, qu)
        info = _info(U, n_lab, nb_lab, E_n, h, apost)
        _add(sc, ("iii", variant, n, d, -1), scores, rows, base, info, cross=cross, flag=fl, sec=time.time() - t0)
        if strata is not None:
            strata(("iii", variant, n, d, -1), scores, info, cross, fl)
        if variant == "" and sc.scope in ("B", "all") and not sc.store.target.endswith("~noearly"):
            _aux_points(sc, "iii", n, d, rows, base, info, E_mean=U.E0 * float(np.sum(post * np.exp(A_GRID))))   # 보조 열: 사후 평균 E
    return qu


def _iv(U, sc, n, d, lab, u_L, col_L, n_lab, nb_lab, E_n, seed, rows, base, t0, fail):
    got = U.ev.iv(seed)
    if got is None:
        fail.append(dict(method="iv", seed=int(seed), n=int(n), draw=int(d), error="척도 적합 없음(emu 실패 또는 미실행)"))
        return
    h_t, c, nu = got
    hm = U.ev.hyp("main")
    sig_L = np.clip(c * np.exp(h_t[U.ev.tpos(lab)]), *LC.SIGMA_CLIP) if len(lab) else np.zeros(0)
    sig_e = np.clip(c * np.exp(h_t[U.ev.tpos(sc.idx)]), *LC.SIGMA_CLIP)
    ll = iv_label_loglik(u_L, col_L, sig_L, hm["tau_b2"], nu)
    post = post_from_loglik(U.ev.prior("main"), ll)
    pc, fl = pc_from_post(post, hm["tau_b2"])
    Qu = iv_cell_quantiles(iv_tables(pc, nu), sig_e)
    scores, cross = _grid_scores(sc, Qu)
    _add(sc, ("iv", "", n, d, int(seed)), scores, rows, base, _info(U, n_lab, nb_lab, E_n, hm, post_stats(post), nu, c), cross=cross,
         flag=fl, sec=time.time() - t0)


def cpu_compute(fr, ev, target, mode, split, cfg, all_scope=False):
    """cpu 단위: 분할의 A 블록에서 h40.draw_cells 로 라벨을 뽑고 채점 셀(B 블록의 eval_mask 셀)에 사다리를 계산한다.
    반환 (저장소 목록, runs 행, cells 배열 dict(2단 CI 용), 통계 dict)."""
    t0 = time.time()
    U = UnitCtx(fr, ev, cfg, target, mode)
    uok, early = fr.uok, fr.flags["early"]
    A_idx, B_idx = half_split_blocks(fr.df, ev.t_idx, split)
    evB = B_idx[eval_mask(fr.df.iloc[B_idx])]
    ev_idx = evB[uok[evB]]
    if len(ev_idx) == 0 or len(A_idx) == 0:
        raise RuntimeError(f"{target}|{mode}|s{split}: 채점 셀 또는 A 셀이 없다")
    base = dict(target=target, mode=mode, split=int(split))
    rows, fail = [], []
    main = Scope(fr, ev_idx, target, split, ev.E0, "B")
    ne = None
    if "noearly" in U.sens and ev.hyp("noearly") is not None and (~early[ev_idx]).any():
        ne = Scope(fr, ev_idx[~early[ev_idx]], f"{target}~noearly", split, ev.E0, "B")
    nd = H.cells_of(cfg.N_GRID, cfg.DRAWS, len(A_idx))
    nd_n, nd_d, nd_En, nd_ls, nd_le, lab_pos, lab_all = [], [], [], [], [], [], []
    for n, d in nd:
        sel = H.draw_cells(target, mode, split, n, d, len(A_idx))
        lab = A_idx[np.asarray(sel, np.int64)]
        run_ladder(U, main, n, d, lab, rows, base, "main", fail)
        if ne is not None:
            run_ladder(U, ne, n, d, lab[~early[lab]], rows, base, "noearly", fail)
        nd_n.append(n); nd_d.append(d); nd_En.append(LC.center_coef(fr.y[lab], fr.s[lab], ev.E0, cfg.kappa))
        nd_ls.append(sum(len(v) for v in lab_all)); lab_all.append(lab); nd_le.append(nd_ls[-1] + len(lab)); lab_pos.append(np.asarray(sel))
    stores = [main.store] + ([ne.store] if ne is not None else [])
    n_all = 0
    if all_scope:
        ev_all = ev.t_idx[eval_mask(fr.df.iloc[ev.t_idx])]
        ev_all = ev_all[uok[ev_all]]
        if len(ev_all):
            sa = Scope(fr, ev_all, f"{target}|all", 0, ev.E0, "all")
            run_ladder(U, sa, 0, 0, np.zeros(0, np.int64), rows, dict(base, split=0), "all", fail)
            stores.append(sa.store); n_all = len(ev_all)
            if "noearly" in U.sens and ev.hyp("noearly") is not None and (~early[ev_all]).any():
                sn = Scope(fr, ev_all[~early[ev_all]], f"{target}|all~noearly", 0, ev.E0, "all")
                run_ladder(U, sn, 0, 0, np.zeros(0, np.int64), rows, dict(base, split=0), "noearly", fail)
                stores.append(sn.store)
    labs = np.concatenate(lab_all).astype(np.int64) if lab_all else np.zeros(0, np.int64)
    cells = dict(E0=np.array(ev.E0), ev_idx=main.idx, ev_loc=fr.df["loc_id"].values[main.idx], ev_y=main.y, ev_s=main.s, ev_A=main.A,
                 ev_u=main.u, ev_col=main.col.astype(np.int64), ev_early=main.flags["early"], ev_gpr=main.flags["gpr"],
                 ev_probe=main.flags["probe"], nd_n=np.asarray(nd_n, np.int64), nd_d=np.asarray(nd_d, np.int64),
                 nd_En=np.asarray(nd_En, float), nd_ls=np.asarray(nd_ls, np.int64), nd_le=np.asarray(nd_le, np.int64),
                 lab_pos=np.concatenate(lab_pos).astype(np.int64) if lab_pos else np.zeros(0, np.int64), lab_idx=labs,
                 lab_u=fr.u_of(labs, ev.E0), lab_col=fr.col[labs].astype(np.int64))
    stats = dict(n_A=int(len(A_idx)), n_eval_mask=int(len(evB)), n_eval=int(len(ev_idx)), n_eval_dropped=int(len(evB) - len(ev_idx)),
                 nb_eval=int(main.store.nb), n_all=int(n_all), n_nd=int(len(nd)), n_keys=int(sum(len(s_) for s_ in stores)), fail=fail,
                 sec=round(time.time() - t0, 2))
    return stores, rows, cells, stats


# ================================================================ 8. 게이트 단위(LGU-A4)와 흐름 단위(단 v)
def gate_compute(fr, ev, target, mode, cfg):
    """게이트: 집단 지역 k 를 하나씩 빼고 단 (iv)와 (v)를 학습해 뺀 지역을 채점한다(지역 오프셋을 아는 조건의 예측 분포).
    초모수는 G 에서 k 를 뺀 블록 표로, 뺀 지역의 ā_k·b̂ 는 그 지역 블록 표와 같은 초모수로 구한다. ν_ε 는 대상의 전체 집단으로 고른 값
    (emu 의 CatBoost seed 0). 단 (iv)의 척도 모형은 G − {k} 전체로 적합하고, 척도 보정 계수 c 는 학습 지역 q 마다 {k, q} 를 뺀 적합의
    표본 밖 예측으로 구한다(계획서 3.3절의 정의, 개정 1. 적합 수는 뺀 지역마다 1 + (K − 1)). 채점은 로그 척도 CRPS 만: blk(N(0, τ_b²)와 잔차 분포의 합성, w = u − ā_k), res(잔차 분포, ẽ),
    n0(사전 예측, u). 반환 (저장소 목록, runs 행, 지역 정보, 실패 목록, 적합 수)."""
    got = ev.iv(0)
    if got is None:
        raise RuntimeError("단 (iv)의 ν_ε 선택이 없다(emu 단위의 척도 적합이 없거나 실패했다)")
    nu = got[2]
    kap = kappa_nu(nu)
    bt = ev.bt("bt")
    gi = np.asarray(ev.a["gi"], np.int64); g_reg = np.asarray(ev.a["g_reg"]).astype(str)
    g_col = np.asarray(ev.a["g_col"], np.int64); g_u = np.asarray(ev.a["g_u"], float)
    X = fr.X
    stores, rows, info, fails = [], [], [], []
    n_fit = Counter()
    for k in ev.groups:
        keep, hold = g_reg != k, g_reg == k
        bt_tr, bt_h = bt_subset(bt, bt["region"] != k), bt_subset(bt, bt["region"] == k)
        hk = hyper_bt(bt_tr)
        if hk["K"] == 0 or hold.sum() == 0 or len(bt_h["n"]) == 0:
            info.append(dict(region=k, status="skipped", reason="훈련 집단 또는 뺀 지역 셀 없음"))
            continue
        tb2, s2 = hk["tau_b2"], hk["sigma2"]
        a_tr, b_tr = cell_offsets(g_reg[keep], g_col[keep], bt_tr, hk)
        e_tr = g_u[keep] - a_tr - b_tr
        okt = np.isfinite(e_tr)
        hh = hyper_bt(bt_h, fixed=(s2, tb2))
        a_h, b_h = cell_offsets(g_reg[hold], g_col[hold], bt_h, hh)
        u_h = g_u[hold]; w_h = u_h - a_h; e_h = w_h - b_h
        okh = np.isfinite(u_h) & np.isfinite(w_h) & np.isfinite(e_h)
        idx_tr, et, col_tr, reg_tr = gi[keep][okt], e_tr[okt], g_col[keep][okt], g_reg[keep][okt]
        idx_h, uh, wh, eh, col_h = gi[hold][okh], u_h[okh], w_h[okh], e_h[okh], g_col[hold][okh]
        tg, pt = tau_r_posterior(hk["a_k"], hk["v_k"], cfg.nu, cfg.s_tau)
        pa = prior_a(tg, pt, cfg.nu)
        pb_u, _ = pc_from_post(point_mass_a0(), tb2)
        pc_u, _ = pc_from_post(pa, tb2)
        st = LC.ScoreStore(f"{target}~gate:{k}", 0, col_h.astype(str), col_h, meta=dict(heldout=k, n_cells=int(len(idx_h))))
        base = dict(target=target, mode=mode, split=0, heldout=k, n_train=int(len(idx_tr)), n_eval=int(len(idx_h)), sigma2=s2,
                    tau_b2=tb2, a_k=float(hh["a_k"][0]) if hh["K"] else np.nan, nu_eps=nu)

        def add(key, vals, Qu, extra, st=st, base=base):
            Qu = LC.rearrange(np.asarray(Qu, float))
            sc_ = dict(crps_log=LC.crps_grid(vals, Qu), valid=np.ones(len(vals), bool))
            st.add(key, sc_)
            rows.append(dict(base, method=key[0], variant=key[1], seed=key[4], crps_log=st.mean(key, "crps_log", "cell"),
                             crps_log_beq=st.mean(key, "crps_log", "beq"), **extra))
        t0 = time.time()
        try:
            ytar, w_tr, X_tr = np.log(np.abs(et) + 0.01), LC.row_weights(col_tr), X[idx_tr]
            m_ = cb_fit(X_tr, ytar, w_tr, 0, cfg.threads); n_fit["catboost"] += 1
            h_h = cb_pred(m_, X[idx_h], cfg.threads)
            # 척도 보정 계수 c(계획서 3.3절 단 iv, 개정 1): 학습 지역 q 마다 {k, q} 를 뺀 적합으로 q 의 표본 밖 ĥ^oos 를 만들고
            # c² = 지역 등가중 평균_q[mean((ẽ/exp(ĥ^oos))²)] 로 둔다(iv_scale_fit 과 같은 식. 뺀 지역 k 는 어느 적합에도 들어가지 않는다)
            h_oos = np.full(len(idx_tr), np.nan)
            regs_tr = sorted(set(reg_tr.tolist()))
            for q in regs_tr:
                trq = reg_tr != q
                if trq.sum() == 0:
                    continue
                mq = cb_fit(X_tr[trq], ytar[trq], w_tr[trq], 0, cfg.threads); n_fit["catboost"] += 1
                h_oos[~trq] = cb_pred(mq, X_tr[~trq], cfg.threads)
            regs_c = [q for q in regs_tr if np.isfinite(h_oos[reg_tr == q]).any()]
            if not regs_c:
                raise RuntimeError("척도 보정 계수 c 의 표본 밖 예측을 만들 학습 지역이 없다(학습 집단 1개)")
            c = float(np.sqrt(np.mean([np.nanmean((et[reg_tr == q] / np.exp(h_oos[reg_tr == q])) ** 2) for q in regs_c])))
            sig = np.clip(c * np.exp(h_h), *LC.SIGMA_CLIP)
            ex = dict(scale_c=c, scale_c_method="oos_leave_one_region", scale_c_regions=len(regs_c), sec=round(time.time() - t0, 2))
            add(("iv", "blk", 0, 0, 0), wh, iv_cell_quantiles(iv_tables(pb_u, nu), sig), ex)
            add(("iv", "res", 0, 0, 0), eh, (sig * kap)[None, :] * t_ppf(LC.TAUS, nu)[:, None], ex)
            add(("iv", "n0", 0, 0, 0), uh, iv_cell_quantiles(iv_tables(pc_u, nu), sig), ex)
        except Exception as ex_:                                            # noqa: BLE001
            if LC.is_env_error(ex_):                                       # 환경·자원 오류는 게이트 조각 전체를 실패로 둔다
                raise
            fails.append(dict(region=k, method="iv", seed=0, exc=True, error=f"{type(ex_).__name__}: {str(ex_)[:240]}"))
        stx = LC.prep_stats(X[idx_tr])
        Xz_tr, Xz_h = LC.prep_apply(X[idx_tr], stx), LC.prep_apply(X[idx_h], stx)
        pb01, pc01 = coarsen2(pb_u), coarsen2(pc_u)
        for seed in cfg.FLOW_SEEDS:
            t0 = time.time()
            try:
                h = LC.fit_gen("nflow", Xz_tr, et, groups=col_tr, weights=LC.row_weights(col_tr), seed=int(seed), epochs=int(cfg.epochs),
                               clamp=float(cfg.clamp), median_zero=True, val="block")
                n_fit["nflow"] += 1
                Qres = LC.nflow_quantiles(h, Xz_h, LC.TAUS)
                bad, why = LC.fit_failed(h, Qres)
                if bad:
                    fails.append(dict(region=k, method="v", seed=int(seed), exc=False, error=f"적합 실패: {why}", **h.info()))
                    continue
                Fe = LC.nflow_cdf_table(h, Xz_h, EPS_GRID)
                ex = dict(val_loss=h.val_loss, epochs_run=h.epochs_run, fit_flag=h.flag, sec=round(time.time() - t0, 2))
                add(("v", "blk", 0, 0, int(seed)), wh, v_quantiles(Fe, pb01), ex)
                add(("v", "res", 0, 0, int(seed)), eh, Qres, ex)
                add(("v", "n0", 0, 0, int(seed)), uh, v_quantiles(Fe, pc01), ex)
            except Exception as ex_:                                        # noqa: BLE001  seed 하나의 실패는 그 키만 실패로 둔다
                if LC.is_env_error(ex_):                                   # CUDA 메모리 부족 등은 조각 실패(재개 대상)로 둔다
                    raise
                fails.append(dict(region=k, method="v", seed=int(seed), exc=True, error=f"{type(ex_).__name__}: {str(ex_)[:240]}"))
        stores.append(st)
        info.append(dict(region=k, status="ok", nb=int(st.nb), n_cells=int(len(idx_h)), n_train=int(len(idx_tr)), sigma2=s2, tau_b2=tb2,
                         tau_r2=hk["tau_r2"], K_train=hk["K"]))
    return stores, rows, info, fails, dict(n_fit)


GATE_STATES = ("통과", "미통과", "판정 불가(적합 실패)", "판정 불가(계산 실패)", "판정 불가(지역 부족)")


def gate_decide(stores, nboot, target, mode, seeds, force="auto"):
    """LGU-A4 의 판정. 뺀 지역 가운데 채점 블록 8개 이상인 지역(사용 지역)마다 crps_log 의 (v − iv) 1단 CI(v 는 유효 seed 평균)를 구하고
    같은 번호끼리 지역 등가중 평균한다. 유효 seed = 모든 사용 지역에서 흐름 적합이 성공한 seed. 상태(gate_state, 개정 1):
    '판정 불가(지역 부족)' = 사용 지역이 없다. '판정 불가(적합 실패)' = 사용 지역 가운데 단 (iv) 키가 없는 지역이 있거나(척도 적합 실패)
    유효 seed 가 2개 미만이다(계획서 2.10절). '판정 불가(계산 실패)' = 위가 아닌데 CI 가 비유한이다. '통과' = 두 가중의 CI 상한 < 0.
    '미통과' = 그 밖(CI 유한, 유효 seed 2개 이상). gate_pass 는 상태가 '통과' 일 때만 참이다(판정 불가는 미통과로 세지 않는다).
    보조: res, n0 의 같은 대비. force('pass'·'fail')는 스모크 전용 강제 값이다(gate_pass 만 바꾸고 gate_state 는 계산값을 둔다)."""
    out = dict(gate_pass=False, gate_state="판정 불가(지역 부족)", forced=force if force != "auto" else "", regions=[], n_valid_seeds=0)
    used = [st for st in stores if st.nb >= LC.MIN_BLOCKS_CI]
    valid = [int(s_) for s_ in seeds if used and all(("v", "blk", 0, 0, int(s_)) in st for st in used)]
    iv_ok = bool(used) and all(("iv", "blk", 0, 0, 0) in st for st in used)
    vs = set(valid) if valid else {int(s_) for s_ in seeds}
    for var in ("blk", "res", "n0"):
        per = {}
        for st in used:
            k = st.meta.get("heldout")
            kA = st.select(lambda key, var=var: key[0] == "v" and key[1] == var and int(key[4]) in vs)
            kB = st.select(lambda key, var=var: key[0] == "iv" and key[1] == var)
            if not kA or not kB:
                continue
            per[k] = LC.one_stage_delta({0: st}, kA, kB, "crps_log", nboot, seed_of("lgu-gate", target, mode, k))
        if per:
            dist = LC.strat_mean([r["dist"] for r in per.values()])
            dist_b = LC.strat_mean([r["dist_beq"] for r in per.values()])
            lo, hi, nn = LC.ci95(dist); lob, hib, nnb = LC.ci95(dist_b)
            d_ = float(np.mean([r["delta"] for r in per.values()])); db_ = float(np.mean([r["delta_beq"] for r in per.values()]))
        else:
            lo = hi = lob = hib = d_ = db_ = np.nan; nn = nnb = 0
        out[var] = dict(delta=d_, ci_lo=lo, ci_hi=hi, delta_beq=db_, ci_lo_beq=lob, ci_hi_beq=hib, regions=sorted(per), n_nan=max(nn, nnb),
                        per_region={k: dict(delta=r["delta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], delta_beq=r["delta_beq"],
                                            ci_lo_beq=r["ci_lo_beq"], ci_hi_beq=r["ci_hi_beq"], n_blocks=r["n_blocks"])
                                    for k, r in per.items()})
    out["n_valid_seeds"] = len(valid); out["valid_seeds"] = valid; out["regions"] = out["blk"]["regions"]
    out["regions_used"] = [st.meta.get("heldout") for st in used]; out["iv_ok"] = bool(iv_ok)
    b = out["blk"]
    if not used:
        state = "판정 불가(지역 부족)"
    elif not iv_ok or len(valid) < 2:
        state = "판정 불가(적합 실패)"
    elif not (np.isfinite(b["ci_hi"]) and np.isfinite(b["ci_hi_beq"])):
        state = "판정 불가(계산 실패)"
    elif b["ci_hi"] < 0 and b["ci_hi_beq"] < 0:
        state = "통과"
    else:
        state = "미통과"
    out["gate_state"] = state
    out["gate_pass"] = state == "통과"
    if force == "pass":
        out["gate_pass"] = True
    elif force == "fail":
        out["gate_pass"] = False
    return out


def flow_compute(fr, ev, target, mode, cfg, splits):
    """단 (v): 전체 집단 셀의 ẽ 로 nflow 위치 고정 변형을 학습(seed 마다)하고, 분할마다 라벨 우도(밀도 표의 선형 보간, b 101점 구적)로
    a 사후를 구해 채점 셀의 CDF 표와 합성한다. 반환 (저장소 목록, runs 행, 실패 목록, 적합 수)."""
    hm = ev.hyp("main")
    tb2 = float(hm["tau_b2"])
    bt = ev.bt("bt")
    gi = np.asarray(ev.a["gi"], np.int64); g_reg = np.asarray(ev.a["g_reg"]).astype(str)
    g_col = np.asarray(ev.a["g_col"], np.int64); g_u = np.asarray(ev.a["g_u"], float)
    a_c, b_c = cell_offsets(g_reg, g_col, bt, hm)
    e = g_u - a_c - b_c
    ok = np.isfinite(e)
    idx_tr, et, col_tr = gi[ok], e[ok], g_col[ok]
    stx = LC.prep_stats(fr.X[idx_tr])
    Xz_tr = LC.prep_apply(fr.X[idx_tr], stx)
    pa = ev.prior("main")
    plan = []
    for sp in splits:
        A_idx, B_idx = half_split_blocks(fr.df, ev.t_idx, sp)
        evB = B_idx[eval_mask(fr.df.iloc[B_idx])]
        ev_idx = evB[fr.uok[evB]]
        if len(ev_idx) == 0 or len(A_idx) == 0:
            continue
        labs = [(n, d, A_idx[np.asarray(H.draw_cells(target, mode, sp, n, d, len(A_idx)), np.int64)])
                for n, d in H.cells_of(cfg.N_GRID, cfg.DRAWS, len(A_idx))]
        plan.append((sp, ev_idx, labs))
    E_all = np.unique(np.concatenate([p_[1] for p_ in plan])) if plan else np.zeros(0, np.int64)
    L_all = np.unique(np.concatenate([lab for p_ in plan for _, _, lab in p_[2]] + [np.zeros(0, np.int64)]))
    Xz_E, Xz_L = LC.prep_apply(fr.X[E_all], stx), LC.prep_apply(fr.X[L_all], stx)
    scopes = {sp: Scope(fr, ev_idx, target, sp, ev.E0, "B") for sp, ev_idx, _ in plan}
    U = UnitCtx(fr, ev, cfg, target, mode)
    rows, fails, n_fit = [], [], Counter()
    for seed in cfg.FLOW_SEEDS:
        t0 = time.time()
        try:
            h = LC.fit_gen("nflow", Xz_tr, et, groups=col_tr, weights=LC.row_weights(col_tr), seed=int(seed), epochs=int(cfg.epochs),
                           clamp=float(cfg.clamp), median_zero=True, val="block")
            n_fit["nflow"] += 1
            bad, why = LC.fit_failed(h, LC.nflow_quantiles(h, Xz_E, Q5))
            if bad:
                fails.append(dict(method="v", seed=int(seed), exc=False, error=f"적합 실패: {why}", **h.info()))
                continue
            Fe = LC.nflow_cdf_table(h, Xz_E, EPS_GRID)
            LF = (LC.nflow_logpdf(h, Xz_L, np.broadcast_to(EPS_GRID, (len(L_all), len(EPS_GRID)))).astype(np.float32)
                  if len(L_all) else np.zeros((0, len(EPS_GRID)), np.float32))
            fit_s = round(time.time() - t0, 2)
        except Exception as ex_:                                            # noqa: BLE001
            if LC.is_env_error(ex_):
                raise
            fails.append(dict(method="v", seed=int(seed), exc=True, error=f"{type(ex_).__name__}: {str(ex_)[:240]}"))
            continue
        for sp, ev_idx, labs in plan:
            sc = scopes[sp]
            posE = np.searchsorted(E_all, ev_idx)
            base = dict(target=target, mode=mode, split=int(sp))
            for n, d, lab in labs:
                t1 = time.time()
                try:
                    u_L = fr.u_of(lab, ev.E0); col_L = fr.col[lab]
                    ll = v_label_loglik(u_L, col_L, LF[np.searchsorted(L_all, lab)], tb2) if len(lab) else np.zeros(len(A_GRID))
                    post = post_from_loglik(pa, ll)
                    pc, fl = pc_from_post(post, tb2)
                    scores, cross = _grid_scores(sc, v_quantiles(Fe[posE], coarsen2(pc)))
                    E_n = LC.center_coef(fr.y[lab], fr.s[lab], ev.E0, cfg.kappa)
                    nJ, _ = label_blocks(u_L, col_L)
                    _add(sc, ("v", "", n, d, int(seed)), scores, rows, base,
                         dict(_info(U, int(np.isfinite(u_L).sum()), len(nJ), E_n, hm, post_stats(post)),
                              val_loss=h.val_loss, epochs_run=h.epochs_run, fit_s=fit_s), cross=cross, flag=fl, sec=time.time() - t1)
                except Exception as ex_:                                    # noqa: BLE001
                    if LC.is_env_error(ex_):
                        raise
                    fails.append(dict(method="v", seed=int(seed), split=int(sp), n=int(n), draw=int(d), exc=True,
                                      error=f"{type(ex_).__name__}: {str(ex_)[:240]}"))
    return [scopes[sp].store for sp in sorted(scopes)], rows, fails, dict(n_fit)


# ================================================================ 9. 조각 입출력과 작업 단위
def shard_paths(a, kind, t, m, sp=None):
    b = str(a.SHARDS / (f"{a.TAG}__{kind}__{t}__{m}" + (f"__s{sp}" if sp is not None else "")))
    d = {f.split(".")[0]: Path(f"{b}_{f}") for f in SHARD_FILES[kind]}
    d["unit"] = Path(b + "_unit.json")
    return d


def unit_cfg(a, kind):
    """결과에 영향을 주는 설정. 대상·분할·워커·스레드·GPU 는 넣지 않는다(조각의 정체 또는 실행 자원)."""
    c = dict(format=FMT, kind=kind, kappa=float(a.kappa), buffer_km=float(a.buffer_km), min_cal_cells=int(a.min_cal_cells), nu=float(a.nu),
             s_tau=float(a.s_tau), cap_cells=int(a.cap_cells), n_grid=list(a.N_GRID), splits=list(a.SPLITS), sens=list(a.SENS),
             subregion_map=a.SUBMAP.name if a.SUBMAP.exists() else "kmeans", label_flags=a.FLAGS.name)
    emu = dict(c, kind="emu", rungs=sorted(set(a.RUNGS) & {"iii+c", "iv"}), cb_seeds=list(a.CB_SEEDS))
    if kind == "emu":
        return emu
    c["emu_hash"] = LC.cfg_hash(emu)
    if kind == "cpu":
        c.update(draws=int(a.DRAWS), rungs=list(a.RUNGS), cb_seeds=list(a.CB_SEEDS))
    elif kind == "gate":
        c.update(epochs=int(a.epochs), flow_seeds=list(a.FLOW_SEEDS), clamp=float(a.clamp))
    elif kind == "flow":
        c.update(epochs=int(a.epochs), flow_seeds=list(a.FLOW_SEEDS), clamp=float(a.clamp), draws=int(a.DRAWS), force_gate=a.force_gate)
    return c


_HASH: dict = {}


def input_hashes(a):
    key = (str(a.PROC), str(a.SUBMAP), str(a.FLAGS))
    if key not in _HASH:
        _HASH[key] = {nm: LC.file_sha(p) for nm, p in (("fidelity_base_v3.csv", a.PROC / "fidelity_base_v3.csv"),
                                                       ("e5_soil_tdd_v3.csv", a.PROC / "e5_soil_tdd_v3.csv"),
                                                       (a.SUBMAP.name, a.SUBMAP), (a.FLAGS.name, a.FLAGS))}
    return _HASH[key]


def code_shas():
    return dict(code_sha=LC.file_sha(__file__), code_sha_common=LC.file_sha(LC.__file__), code_sha_h40=LC.file_sha(H40_PATH))


def write_unit(path, a, kind, t, m, sp, status, extra):
    cfg = unit_cfg(a, kind)
    u = dict(kind=kind, target=t, mode=m, split=sp, tag=a.TAG, status=status, cfg=cfg, cfg_hash=LC.cfg_hash(cfg), **code_shas(),
             inputs=input_hashes(a), threads=int(a.threads), device=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
             written=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    return LC.atomic_text(path, json.dumps(u, ensure_ascii=False, indent=1, default=LC._py))


def atomic_csv(path, df):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pd.DataFrame(df).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_unit(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def unit_done(a, kind, t, m, sp=None):
    """(완료 여부, 사유). LC.unit_state 와 같은 규칙. 게이트 미통과로 건너뛴 흐름 단위는 unit.json 만 있으면 완료다.
    흐름 단위는 기록한 게이트 식별자(gate_ref)가 현재 게이트 조각과 같아야 완료다(게이트를 다시 실행하면 흐름도 다시 한다, 개정 1)."""
    p = shard_paths(a, kind, t, m, sp)
    need = [v for k, v in p.items() if k != "unit"]
    u = read_unit(p["unit"])
    if u.get("status") == "skipped_gate":
        need = []
    ok, why = LC.unit_state(p["unit"], need, unit_cfg(a, kind))
    if ok and kind == "flow" and u.get("gate_ref") != gate_ref(a, t, m):
        return False, "게이트 조각이 바뀌었다(gate_ref 불일치)"
    return ok, why


def cpu_splits(a, D, t):
    """(실행 분할 목록, 건너뛴 분할 기록). h40.enumerate_units 와 같은 규칙: 중복 분할과 채점·A 셀이 없는 분할은 빼고, 유효 분할이 있는
    대상의 무효 분할(채점 블록 < 2)도 뺀다."""
    info = D.split_structure(t)
    any_valid = any(v["valid"] for v in info.values())
    run, skipped = [], []
    for sp in a.SPLITS:
        v = info[sp]
        if v["dup_of"] >= 0:
            skipped.append(dict(target=t, split=sp, status=f"dup_of_{v['dup_of']}"))
        elif v["n_eval"] == 0 or v["n_A"] == 0:
            skipped.append(dict(target=t, split=sp, status="no_eval"))
        elif not v["valid"] and any_valid:
            skipped.append(dict(target=t, split=sp, status="invalid_nb_eval<2"))
        else:
            run.append(sp)
    return run, skipped


def _fail_status(fail):
    return "partial" if any(not str(f.get("error", "")).startswith(("보정 항목 없음", "c_n 없음")) for f in fail) else "ok"


def run_emu_unit(a, t, m, sp=None):
    t0 = time.time()
    p = shard_paths(a, "emu", t, m)
    p["unit"].unlink(missing_ok=True)
    D, fr = get_D(a), frame_of(a)
    t_idx, parent, src_idx, comp = D.source_idx(t, m)
    try:
        emu = emu_compute(fr, t_idx, src_idx, t, m, a, do_iv=(is_main(t, m) and "iv" in a.RUNGS))
    except Exception as ex:
        write_unit(p["unit"], a, "emu", t, m, None, "failed", dict(error=f"{type(ex).__name__}: {str(ex)[:400]}"))
        raise
    save_emu(p["emu"], emu)
    mt = emu["meta"]
    iv_fail = [f for f in mt["iv"] if f.get("fail")]
    status = "partial" if iv_fail else "ok"
    write_unit(p["unit"], a, "emu", t, m, None, status,
               dict(parent=parent, comp=comp, E0=mt["E0"], groups=mt["groups"], n_src=mt["n_src"], n_group_cells=mt["n_group_cells"],
                    n_blocks=mt["n_blocks"], n_records=mt["n_records"], n_entries=mt["n_entries"], flags=mt["flags"],
                    n_fit=dict(catboost=int(sum(f["n_fit"] for f in mt["iv"]))), iv=mt["iv"], sec=mt["sec"],
                    elapsed_s=round(time.time() - t0, 1)))
    return dict(kind="emu", target=t, mode=m, split=-1, status=status, sec=round(time.time() - t0, 1),
                note=f"집단 {len(mt['groups'])} · 모사 기록 {mt['n_records']} · CatBoost {sum(f['n_fit'] for f in mt['iv'])}")


def _need_emu(a, t, m, fr):
    pe = shard_paths(a, "emu", t, m)
    ok, why = LC.unit_state(pe["unit"], [pe["emu"]], unit_cfg(a, "emu"))
    if not ok:
        raise RuntimeError(f"{t}|{m}: emu 단위가 없다({why})")
    return EmuView(load_emu(pe["emu"]), fr)


def run_cpu_unit(a, t, m, sp):
    t0 = time.time()
    p = shard_paths(a, "cpu", t, m, sp)
    p["unit"].unlink(missing_ok=True)
    D, fr = get_D(a), frame_of(a)
    ev = _need_emu(a, t, m, fr)
    run, _ = cpu_splits(a, D, t)
    try:
        stores, rows, cells, stats = cpu_compute(fr, ev, t, m, sp, a, all_scope=bool(run) and sp == min(run))
    except Exception as ex:
        write_unit(p["unit"], a, "cpu", t, m, sp, "failed", dict(error=f"{type(ex).__name__}: {str(ex)[:400]}"))
        raise
    LC.save_score_stores(stores, p["scores"])
    LC.atomic_npz(p["cells"], **cells)
    atomic_csv(p["runs"], pd.DataFrame(rows))
    status = _fail_status(stats["fail"])
    write_unit(p["unit"], a, "cpu", t, m, sp, status, dict(stats, split_info=D.split_structure(t)[sp], E0=ev.E0,
                                                           elapsed_s=round(time.time() - t0, 1)))
    return dict(kind="cpu", target=t, mode=m, split=sp, status=status, sec=round(time.time() - t0, 1),
                note=f"채점 {stats['n_eval']}셀/{stats['nb_eval']}블록 · (n, 추출) {stats['n_nd']} · 키 {stats['n_keys']} · 실패 {len(stats['fail'])}")


def run_gate_unit(a, t, m, sp=None):
    t0 = time.time()
    p = shard_paths(a, "gate", t, m)
    p["unit"].unlink(missing_ok=True)
    fr = frame_of(a)
    ev = _need_emu(a, t, m, fr)
    try:
        stores, rows, info, fails, n_fit = gate_compute(fr, ev, t, m, a)
        dec = gate_decide(stores, a.nboot, t, m, a.FLOW_SEEDS, a.force_gate)
    except Exception as ex:
        write_unit(p["unit"], a, "gate", t, m, None, "failed", dict(error=f"{type(ex).__name__}: {str(ex)[:400]}"))
        raise
    LC.save_score_stores(stores, p["scores"])
    atomic_csv(p["runs"], pd.DataFrame(rows))
    status = "partial" if fails else "ok"
    if dec["gate_state"] == "판정 불가(적합 실패)" and any(f.get("exc") for f in fails):
        status = "failed"                                                 # 예외로 생긴 적합 실패는 --resume 이 다시 실행한다
    write_unit(p["unit"], a, "gate", t, m, None, status, dict(gate_pass=dec["gate_pass"], gate_state=dec["gate_state"], gate=dec,
                                                              regions=info, fail=fails, n_fit=n_fit, nboot=int(a.nboot),
                                                              elapsed_s=round(time.time() - t0, 1)))
    b = dec["blk"]
    return dict(kind="gate", target=t, mode=m, split=-1, status=status, sec=round(time.time() - t0, 1),
                note=f"상태 {dec['gate_state']} · 통과 {dec['gate_pass']}{'(강제)' if dec['forced'] else ''} · 유효 seed {dec['n_valid_seeds']} · "
                     f"지역 {len(b['regions'])} · 적합 {n_fit} · 실패 {len(fails)}")


def gate_state(a, t, m):
    """흐름 단위의 실행 여부 (bool, 사유). 게이트 상태가 '통과' 일 때만 참이다(판정 불가는 흐름을 실행하지 않는다).
    --force-gate 는 스모크 전용이다."""
    if a.force_gate == "pass":
        return True, "강제 통과(스모크)"
    if a.force_gate == "fail":
        return False, "강제 미통과(스모크)"
    ok, why = unit_done(a, "gate", t, m)
    if not ok:
        return None, f"게이트 조각 없음({why})"
    u = read_unit(shard_paths(a, "gate", t, m)["unit"])
    st = str(u.get("gate_state") or ("통과" if u.get("gate_pass") else "미통과"))
    return bool(u.get("gate_pass", False)) and st == "통과", f"게이트 판정: {st}"


def gate_ref(a, t, m):
    """흐름 조각이 기대는 게이트 결과의 식별자: 게이트 unit.json 의 파일 해시(강제 값이면 'forced:<값>'). 게이트를 다시 실행하면 바뀐다."""
    if a.force_gate != "auto":
        return f"forced:{a.force_gate}"
    return LC.file_sha(shard_paths(a, "gate", t, m)["unit"])


def run_flow_unit(a, t, m, sp=None):
    t0 = time.time()
    p = shard_paths(a, "flow", t, m)
    p["unit"].unlink(missing_ok=True)
    D, fr = get_D(a), frame_of(a)
    gp, why = gate_state(a, t, m)
    if gp is None:
        raise RuntimeError(f"{t}|{m}: {why}. --part gate 를 먼저 실행한다")
    gref = gate_ref(a, t, m)
    if not gp:
        write_unit(p["unit"], a, "flow", t, m, None, "skipped_gate", dict(reason=why, gate_ref=gref, elapsed_s=round(time.time() - t0, 1)))
        return dict(kind="flow", target=t, mode=m, split=-1, status="skipped_gate", sec=round(time.time() - t0, 1), note=why)
    ev = _need_emu(a, t, m, fr)
    run, _ = cpu_splits(a, D, t)
    try:
        stores, rows, fails, n_fit = flow_compute(fr, ev, t, m, a, run)
    except Exception as ex:
        write_unit(p["unit"], a, "flow", t, m, None, "failed", dict(error=f"{type(ex).__name__}: {str(ex)[:400]}"))
        raise
    LC.save_score_stores(stores, p["scores"])
    atomic_csv(p["runs"], pd.DataFrame(rows))
    ok_seeds = sorted({int(r["seed"]) for r in rows})
    status = "partial" if fails else "ok"
    write_unit(p["unit"], a, "flow", t, m, None, status, dict(gate=why, gate_ref=gref, splits=run, fail=fails, n_fit=n_fit,
                                                              valid_seeds=ok_seeds, elapsed_s=round(time.time() - t0, 1)))
    return dict(kind="flow", target=t, mode=m, split=-1, status=status, sec=round(time.time() - t0, 1),
                note=f"{why} · 분할 {run} · 유효 seed {ok_seeds} · 적합 {n_fit} · 실패 {len(fails)}")


# ---------------------------------------------------------------- 워커
_WA = None


def _worker_init(argv, gpu_queue, threads, torch_on):
    global _WA
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                          # 0 MiB 확인(nvidia-smi 번호)과 같은 번호 체계
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get()) if gpu_queue is not None else ""   # torch 를 부르기 전에 정한다
    LC.set_thread_env(threads)
    _WA = parse_args(argv)
    _WA.threads = int(threads)                                              # 주 프로세스의 값(로컬 규칙 초과는 거부되므로 CLI 값과 같다)
    if torch_on:
        try:
            LC.set_torch_threads(threads)
        except Exception:                                                   # noqa: BLE001
            pass


def _worker_run(kind, t, m, sp):
    t0 = time.time()
    r = RUNNERS[kind](_WA, t, m, sp)
    r["wall_s"] = round(time.time() - t0, 1)
    r["device"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return r


def run_tasks(a, argv, tasks, use_gpu, torch_on, t_start):
    """작업 목록을 프로세스 풀(spawn)에서 돌린다. GPU 풀은 워커마다 GPU 번호 하나(대기열). 반환 (완료 목록, 실패 목록)."""
    done, failed = [], []
    if not tasks:
        return done, failed

    def log(r):
        done.append(r)
        print(f"  [{r['kind']}|{r['target']}|{r['mode']}{'|s' + str(r['split']) if r['split'] not in (-1, None) else ''}] {r['status']} · "
              f"{r.get('note', '')} · {r['sec']}s · 장치 '{r.get('device', '')}' · 완료 {len(done)}/{len(tasks)} · 누적 {time.time() - t_start:.0f}s",
              flush=True)

    def fail(u, e):
        failed.append(dict(kind=u[0], target=u[1], mode=u[2], split=u[3] if u[3] is not None else -1, error=repr(e)[:300]))
        print(f"  [FAIL] {u}: {repr(e)[:300]}", flush=True)

    nproc = len(a.GPUS) * int(a.procs_per_gpu) if use_gpu else max(int(a.workers), 1)
    if not use_gpu and int(a.workers) <= 0:
        _worker_init(list(argv), None, a.threads, torch_on)
        for u in tasks:
            try:
                log(_worker_run(*u))
            except (Exception, SystemExit) as e:                            # noqa: BLE001
                fail(u, e)
        return done, failed
    ctx = multiprocessing.get_context("spawn")                             # fork 뒤 OpenMP 충돌을 피한다
    remaining, attempt = list(tasks), 0
    while remaining:
        q = None
        if use_gpu:
            q = ctx.Queue()
            for g in a.GPUS:
                for _ in range(int(a.procs_per_gpu)):
                    q.put(g)
        broken = []
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx, initializer=_worker_init,
                                 initargs=(list(argv), q, a.threads, torch_on)) as ex:
            futs = {ex.submit(_worker_run, *u): u for u in remaining}
            for f in as_completed(futs):
                try:
                    log(f.result())
                except BrokenProcessPool:
                    broken.append(futs[f])
                except (Exception, SystemExit) as e:                        # noqa: BLE001  한 단위의 실패가 나머지를 막지 않게 한다
                    fail(futs[f], e)
        if not broken:
            break
        attempt += 1
        if attempt > int(a.pool_retries):
            for u in broken:
                fail(u, RuntimeError(f"프로세스 풀이 {attempt}회 깨졌다(재시도 상한 {a.pool_retries})"))
            break
        remaining = broken
        print(f"[pool] 워커 비정상 종료. 남은 {len(remaining)} 단위로 풀을 다시 만든다(재시도 {attempt}/{a.pool_retries})", flush=True)
    return done, failed


# ================================================================ 10. 2단 재표집(계획서 2.6절)
def split_prep(c):
    """2단 채점 준비(분할 하나): 블록마다 u 오름차순 정렬, A(= E0·s)와 y 의 누적합, 블록 합(셀 수, ΣA, Σy, ΣA², ΣAy, Σy²)."""
    u, A, y = np.asarray(c["ev_u"], float), np.asarray(c["ev_A"], float), np.asarray(c["ev_y"], float)
    ub, codes = np.unique(np.asarray(c["ev_col"], np.int64), return_inverse=True)
    codes = codes.ravel()
    nb = len(ub)
    blocks = []
    for b in range(nb):
        ii = np.where(codes == b)[0]
        ii = ii[np.argsort(u[ii], kind="stable")]
        blocks.append((u[ii], np.concatenate([[0.0], np.cumsum(A[ii])]), np.concatenate([[0.0], np.cumsum(y[ii])])))
    return dict(cols=ub.astype(np.int64), blocks=blocks, cnt=np.bincount(codes, minlength=nb).astype(float),
                SA=np.bincount(codes, A, nb), SY=np.bincount(codes, y, nb), SA2=np.bincount(codes, A * A, nb),
                SAY=np.bincount(codes, A * y, nb), SY2=np.bincount(codes, y * y, nb))


def block_sums(prep, Qm):
    """분위 (C, 5)(u 척도, Q5 순서) → 채점 블록 합 (C, 9, nb)(지표 순서 METRICS_2STAGE). 구간 [A·e^(q_lo), A·e^(q_hi)], 중앙값 A·e^(q_50).
    블록마다 u 정렬과 누적합으로 계산하므로 셀별 계산과 같은 값이다(부동소수 오차 범위). 구간 끝점이 NaN 이면 그 지표는 NaN."""
    Qm = np.asarray(Qm, float)
    C, nb = Qm.shape[0], len(prep["cols"])
    S = np.full((C, len(M2), nb), np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        e50 = np.exp(Qm[:, 2])
        for b, (ub_, cA, cY) in enumerate(prep["blocks"]):
            cnt, SA, SY = prep["cnt"][b], prep["SA"][b], prep["SY"][b]
            L = len(ub_)
            for alpha, ilo, ihi, off in ((0.10, 0, 4, 0), (0.20, 1, 3, 4)):
                ql, qh = Qm[:, ilo], Qm[:, ihi]
                okq = ~(np.isnan(ql) | np.isnan(qh))
                kl = np.searchsorted(ub_, np.where(okq, ql, 0.0), side="left")
                kh = np.searchsorted(ub_, np.where(okq, qh, 0.0), side="right")
                el, eh = np.exp(ql), np.exp(qh)
                width = (eh - el) * SA
                under = np.where(kl > 0, (2.0 / alpha) * (el * cA[kl] - cY[kl]), 0.0)
                over = np.where(kh < L, (2.0 / alpha) * ((SY - cY[kh]) - eh * (SA - cA[kh])), 0.0)
                S[:, off, b] = np.where(okq, width + under + over, np.nan)
                S[:, off + 1, b] = np.where(okq, kh - kl, np.nan)
                S[:, off + 2, b] = np.where(okq, width, np.nan)
                S[:, off + 3, b] = np.where(okq, cnt * (qh - ql) / 2.0, np.nan)
            S[:, 8, b] = np.maximum(e50 * e50 * prep["SA2"][b] - 2.0 * e50 * prep["SAY"][b] + prep["SY2"][b], 0.0)
    return S


def agg_r(S, cnt, w):
    """재표집 하나의 다중도 w (nb,)로 셀 가중·블록 등가중 평균 (C, 9, 2). se 는 RMSE(셀 가중: 합의 제곱근, 블록 등가중: 블록 RMSE 의 평균).
    다중도 0 인 블록은 계산에서 뺀다(0·inf 를 만들지 않는다). 뽑힌 블록이 없으면 NaN."""
    sel = np.asarray(w, float) > 0
    out = np.full(S.shape[:2] + (2,), np.nan)
    if not sel.any():
        return out
    Ss, cs, ws = S[..., sel], np.asarray(cnt, float)[sel], np.asarray(w, float)[sel]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        cell = (Ss @ ws) / float(cs @ ws)
        M = Ss / cs
        se_b = np.sqrt(M[:, 8, :])
        beq = (M @ ws) / float(ws.sum())
        beq[:, 8] = (se_b @ ws) / float(ws.sum())
        cell[:, 8] = np.sqrt(cell[:, 8])
    out[..., 0], out[..., 1] = cell, beq
    return out


def two_stage_compute(ev, cells_by_split, W, cfg, methods=TWO_STAGE_METHODS):
    """2단 CI 의 재표집 분포. 재표집 r 마다 다중도 W[r] 로 초모수·τ_r 사후·a 사전(단 iii), 단 (i)(ii)의 값, B4 의 q_n 을 다시 구하고
    (E0, E0^(−k), 라벨 추출, 모사의 분할과 추출은 고정), 채점 블록의 다중도 W[r, 채점 블록 열]로 셀 가중·블록 등가중 통계를 낸다.
    추출 평균(nanmean), 분할의 같은 번호 평균(np.mean) 순으로 결합한다. 반환 dict(dist (n, 방법, R, 9, 2), n_list, methods, metrics,
    n_nan_rep(다중도가 모두 0 이거나 초모수가 정의되지 않은 재표집 수), sec)."""
    t0 = time.time()
    W = np.asarray(W)
    R = W.shape[0]
    methods = list(methods)
    NM = len(methods)
    mi = {m: i for i, m in enumerate(methods)}
    splits = sorted(cells_by_split)
    n_list = sorted({int(n) for sp in splits for n in cells_by_split[sp]["nd_n"]})
    bt = ev.bt("bt")
    Wb = np.asarray(W[:, np.asarray(bt["col"], np.int64)], float)
    qB4 = {}
    if "B4" in mi:
        for n in n_list:
            sc, rg, cl = ev.b4_entries(n)
            qB4[n] = (LC.hier_quantiles_boot(sc, rg, cl, W, (0.80, 0.90), chunk=max(1, int(2e6 // max(len(sc), 1))))   # 묶음 약 16 MB
                      if len(sc) else np.full((R, 2), np.nan))
    preps = {sp: split_prep(cells_by_split[sp]) for sp in splits}
    combos = []
    for sp in splits:
        c = cells_by_split[sp]
        for j in range(len(c["nd_n"])):
            lu = np.asarray(c["lab_u"][c["nd_ls"][j]:c["nd_le"][j]], float)
            lc = np.asarray(c["lab_col"][c["nd_ls"][j]:c["nd_le"][j]])
            nJ, ub = label_blocks(lu, lc)
            uf = lu[np.isfinite(lu)]
            combos.append(dict(sp=sp, n=int(c["nd_n"][j]), delta=float(np.log(float(c["nd_En"][j]) / ev.E0)), nJ=nJ, ub=ub,
                               nL=float(len(uf)), sL=float(uf.sum())))
    C = len(combos)
    sn_keys = sorted({(cb["sp"], cb["n"]) for cb in combos})
    sn_idx = {k: i for i, k in enumerate(sn_keys)}
    grp = np.array([sn_idx[(cb["sp"], cb["n"])] for cb in combos], np.int64)
    by_split = {sp: np.array([i for i, cb in enumerate(combos) if cb["sp"] == sp], np.int64) for sp in splits}
    delta = np.array([cb["delta"] for cb in combos])
    nL = np.array([cb["nL"] for cb in combos]); sL = np.array([cb["sL"] for cb in combos])
    ns = np.array([cb["n"] for cb in combos])
    zq = ndtri(np.asarray(Q5, float))
    acc = np.full((R, len(sn_keys), NM, len(M2), 2), np.nan)
    n_nan = 0
    for r in range(R):
        h = hyper_from_blocks(bt["region"], bt["n"], bt["su"], bt["su2"], mult=Wb[r])
        if h["K"] == 0:
            n_nan += 1
            continue
        tb2, s2 = h["tau_b2"], h["sigma2"]
        PM = np.array([lik_PM(cb["nJ"], cb["ub"], tb2, s2) for cb in combos]).reshape(C, 2)
        Qm = np.full((C, NM, 5), np.nan)
        if "iii" in mi:
            tg, pt = tau_r_posterior(h["a_k"], h["v_k"], cfg.nu, cfg.s_tau)
            post = post_from_PM(_safe_log(prior_a(tg, pt, cfg.nu)), PM[:, 0], PM[:, 1])
            Qm[:, mi["iii"]] = predictive_q(post, float(np.sqrt(tb2 + s2)), Q5)
        if "ii" in mi:
            prec = 1.0 / h["tau_r2"] + PM[:, 0]
            Qm[:, mi["ii"]] = (PM[:, 1] / prec)[:, None] + np.sqrt(1.0 / prec + tb2 + s2)[:, None] * zq[None, :]
        if "i" in mi:
            s1, t1 = h["sigma1_2"], h["tau_r1_2"]
            prec = 1.0 / t1 + nL / s1
            Qm[:, mi["i"]] = ((sL / s1) / prec)[:, None] + np.sqrt(1.0 / prec + s1)[:, None] * zq[None, :]
        if "B4" in mi:
            q80 = np.array([qB4[int(n)][r, 0] for n in ns]); q90 = np.array([qB4[int(n)][r, 1] for n in ns])
            Qm[:, mi["B4"]] = np.stack([delta - q90, delta - q80, delta, delta + q80, delta + q90], 1)
        if "P1" in mi:
            Qm[:, mi["P1"], 2] = delta
        for sp in splits:
            ids = by_split[sp]
            prep = preps[sp]
            S = block_sums(prep, Qm[ids].reshape(-1, 5))
            vals = agg_r(S, prep["cnt"], W[r, prep["cols"]]).reshape(len(ids), NM, len(M2), 2)
            for gidx in np.unique(grp[ids]):
                acc[r, gidx] = _nanmean(vals[grp[ids] == gidx], 0)
    dist = np.full((len(n_list), NM, R, len(M2), 2), np.nan)
    for ni, n in enumerate(n_list):
        gs = [sn_idx[(sp, n)] for sp in splits if (sp, n) in sn_idx]
        if gs:
            dist[ni] = np.mean(acc[:, gs], axis=1).transpose(1, 0, 2, 3)
    return dict(dist=dist, n_list=n_list, methods=methods, metrics=list(M2), splits=splits, n_nan_rep=int(n_nan), R=int(R),
                sec=round(time.time() - t0, 1))


def boot_path(a, t, m):
    return a.OUT / f"{a.TAG}_boot__{t}__{m}.npz"


def boot_inputs(a, t, m):
    """2단 분포가 기대는 입력(개정 1): 완료된 cpu 조각의 분할 목록, 분할별 cells.npz 해시, emu.npz 해시, 코드 해시.
    load_boot 는 저장된 값이 이것과 같을 때만 기존 분포를 다시 쓴다."""
    D = get_D(a)
    run, _ = cpu_splits(a, D, t)
    splits = [int(s_) for s_ in run if unit_done(a, "cpu", t, m, s_)[0]]
    return dict(splits=splits, cells_sha={str(s_): LC.file_sha(shard_paths(a, "cpu", t, m, s_)["cells"]) for s_ in splits},
                emu_sha=LC.file_sha(shard_paths(a, "emu", t, m)["emu"]), **code_shas())


def run_boot_task(a, t, m, sp=None):
    """대상 하나의 2단 재표집(집계 단계의 작업). cells.npz 와 emu.npz 를 읽는다."""
    t0 = time.time()
    fr = frame_of(a)
    ev = _need_emu(a, t, m, fr)
    inp = boot_inputs(a, t, m)
    cells = {}
    for s_ in inp["splits"]:
        with np.load(shard_paths(a, "cpu", t, m, s_)["cells"], allow_pickle=False) as z:
            cells[s_] = {k: z[k] for k in z.files}
    if not cells:
        raise RuntimeError(f"{t}|{m}: 완료된 cpu 조각이 없다")
    W = LC.global_mult(fr.bi, int(a.nboot), 0)
    res = two_stage_compute(ev, cells, W, a)
    meta = dict(target=t, mode=m, nboot=int(a.nboot), splits=res["splits"], n_nan_rep=res["n_nan_rep"], sec=res["sec"],
                cfg_hash=LC.cfg_hash(unit_cfg(a, "cpu")), inputs=inp, **code_shas())
    LC.atomic_npz(boot_path(a, t, m), dist=res["dist"], n_list=np.asarray(res["n_list"], np.int64), methods=np.asarray(res["methods"]),
                  metrics=np.asarray(res["metrics"]), meta=np.array(json.dumps(meta, ensure_ascii=False, default=LC._py)))
    return dict(kind="boot", target=t, mode=m, split=-1, status="ok", sec=round(time.time() - t0, 1),
                note=f"재표집 {a.nboot} · 분할 {res['splits']} · 정의 안 된 재표집 {res['n_nan_rep']}")


RUNNERS = {"emu": run_emu_unit, "cpu": run_cpu_unit, "gate": run_gate_unit, "flow": run_flow_unit, "boot": run_boot_task}


def load_boot(a, t, m):
    """2단 분포 파일을 읽는다. 설정 해시, nboot, 입력(분할 목록, cells·emu 해시, 코드 해시)이 현재와 같을 때만 쓴다(개정 1)."""
    p = boot_path(a, t, m)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("cfg_hash") != LC.cfg_hash(unit_cfg(a, "cpu")) or int(meta.get("nboot", -1)) != int(a.nboot):
            return None
        if meta.get("inputs") != json.loads(json.dumps(boot_inputs(a, t, m))):
            return None
        return dict(dist=z["dist"], n_list=[int(v) for v in z["n_list"]], methods=[str(v) for v in z["methods"]],
                    metrics=[str(v) for v in z["metrics"]], meta=meta)


def boot_level(bt_, method, n, metric):
    """2단 분포에서 (셀 가중, 블록 등가중) 수준 분포 (R,), (R,). 없으면 None."""
    if bt_ is None or method not in bt_["methods"] or int(n) not in bt_["n_list"] or metric not in bt_["metrics"]:
        return None
    d = bt_["dist"][bt_["n_list"].index(int(n)), bt_["methods"].index(method), :, bt_["metrics"].index(metric)]
    return d[:, 0], d[:, 1]


# ================================================================ 11. 집계와 판정(계획서 3.6절)
def _nanmean(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def kfn(method, variant, n):
    return lambda k, m_=method, v_=variant, n_=int(n): k[0] == m_ and k[1] == v_ and int(k[2]) == n_


def level_point(by_split, fn, metric):
    """분할마다 키 평균(nanmean) 후 분할 평균. 반환 dict(cell, beq, n_splits, n_blocks(합집합), n_keys, n_eval(분할 평균))."""
    vc, vb, blocks, nk, ncell = [], [], set(), 0, []
    fc, fb = (LC.agg_rmse_cell, LC.agg_rmse_beq) if metric == "se" else (LC.agg_cell, LC.agg_beq)
    for sp, st in sorted(by_split.items()):
        ks = st.select(fn)
        if not ks:
            continue
        S, C = st.matrices(ks, metric)
        one = np.ones((1, st.nb))
        vc.append(float(_nanmean(fc(S, C, one)[:, 0]))); vb.append(float(_nanmean(fb(S, C, one)[:, 0])))
        blocks |= set(np.asarray(st.blocks)[C.max(axis=0) > 0].tolist()); nk += len(ks); ncell.append(float(C.sum(1).mean()))
    if not vc:
        return dict(cell=np.nan, beq=np.nan, n_splits=0, n_blocks=0, n_keys=0, n_eval=np.nan)
    return dict(cell=float(np.mean(vc)), beq=float(np.mean(vb)), n_splits=len(vc), n_blocks=len(blocks), n_keys=nk, n_eval=float(np.mean(ncell)))


def level_ci1(by_split, fn, metric, nboot, seed):
    """수준의 1단 재표집 분포(셀 가중, 블록 등가중). LC.one_stage_delta 와 같은 재표집 행렬(seed_of(seed, 분할))을 쓴다."""
    fc, fb = (LC.agg_rmse_cell, LC.agg_rmse_beq) if metric == "se" else (LC.agg_cell, LC.agg_beq)
    dc, db = [], []
    for sp, st in sorted(by_split.items()):
        ks = st.select(fn)
        if not ks:
            continue
        S, C = st.matrices(ks, metric)
        Wt = H4.boot_weights(st.nb, int(nboot), seed_of(seed, sp))
        dc.append(_nanmean(fc(S, C, Wt), 0)); db.append(_nanmean(fb(S, C, Wt), 0))
    if not dc:
        return None
    return np.mean(np.stack(dc), 0), np.mean(np.stack(db), 0)


class TS:
    """대상 하나(대상, 모드)의 집계 자료: 저장소, runs, emu 메타, 2단 분포, 게이트, cpu 조각의 방법별 실패 기록."""

    def __init__(self, t, m):
        self.t, self.m = t, m
        self.stores, self.runs, self.emu_meta, self.boot, self.gate, self.flow = {}, [], None, None, None, None
        self.splits_expected = []
        self.fails = []                                                    # cpu 조각 unit.json 의 fail 목록(분할 번호를 붙인다)
        self.gate_status = ""                                              # 게이트 조각의 상태('failed' 이면 판정에 쓰지 않는다)

    def by(self, name):
        return {sp: st for (nm, sp), st in self.stores.items() if nm == name}


def pool_contrast(a, TSs, regions, A, B, n, metric, varA="", varB="", store_sfx=""):
    """풀 대비 A − B. 지역마다 1단(LC.one_stage_delta, seed_of('lgu-a', 대상, 모드))과 2단(가능하면)의 분포를 구하고 같은 번호끼리
    지역 등가중 평균한다. 풀 조건: 사용 분할의 채점 블록 합집합 8개 이상. 기준 점수는 B 의 풀 평균 수준(가중별).
    짝지음 점검(개정 1): 지역마다 1단 계산의 pair_ok(한쪽에만 있는 (n, 추출) 단위와 NaN 키가 없다)와, 변형이 없는 대비에서는 cpu 조각에
    기록된 A·B 방법의 계산 실패(같은 n, 범위 B)가 없는지를 본다. 결과는 pair_ok, pair_issue 로 돌려준다."""
    per1, per2, why, spl, ref_c, ref_b, cov_c, cov_b, issues = {}, {}, {}, {}, [], [], [], [], {}
    for t in regions:
        ts = TSs.get((t, "x"))
        if ts is None:
            why[t] = "행 없음"
            continue
        bs = ts.by(t + store_sfx)
        fA, fB = kfn(A, varA, n), kfn(B, varB, n)
        la, lb = level_point(bs, fA, metric), level_point(bs, fB, metric)
        if la["n_splits"] == 0 or lb["n_splits"] == 0:
            why[t] = "행 없음"
            continue
        if min(la["n_blocks"], lb["n_blocks"]) < LC.MIN_BLOCKS_CI:
            why[t] = f"채점 블록 {min(la['n_blocks'], lb['n_blocks'])} < {LC.MIN_BLOCKS_CI}"
            continue
        r1 = LC.one_stage_delta(bs, fA, fB, metric, int(a.nboot), seed_of("lgu-a", t, "x"))
        if r1["n_splits"] == 0 or r1["dist"] is None:
            why[t] = "1단 CI 없음"
            continue
        per1[t] = r1
        spl[t] = (int(r1["n_splits"]), len(ts.splits_expected))
        ref_c.append(lb["cell"]); ref_b.append(lb["beq"])
        lc = level_point(bs, fA, "cov10") if metric != "se" else dict(cell=np.nan, beq=np.nan)
        cov_c.append(lc["cell"]); cov_b.append(lc["beq"])
        iss = []
        if r1["n_unpaired_A"] or r1["n_unpaired_B"]:
            iss.append(f"짝 없는 (분할, n, 추출) 단위 A {r1['n_unpaired_A']}·B {r1['n_unpaired_B']}")
        if r1["n_nan_keys_A"] or r1["n_nan_keys_B"]:
            iss.append(f"NaN 키 A {r1['n_nan_keys_A']}·B {r1['n_nan_keys_B']}")
        if not store_sfx and varA == "" and varB == "":
            nf = sum(1 for f in ts.fails if str(f.get("method", "")) in (A, B) and int(f.get("n", -1)) == int(n)
                     and str(f.get("kind", "main")) == "main" and not str(f.get("error", "")).startswith(("보정 항목 없음", "c_n 없음")))
            if nf:
                iss.append(f"cpu 조각의 계산 실패 {nf}건")
            dA, dB = boot_level(ts.boot, A, n, metric), boot_level(ts.boot, B, n, metric)
            if dA is not None and dB is not None:
                per2[t] = (dA[0] - dB[0], dA[1] - dB[1])
        if iss:
            issues[t] = "; ".join(iss)
    pool = [t for t in regions if t in per1]
    out = dict(pool=pool, why=why, splits=spl, n_pool=len(pool), n_regions=len(regions), pair_ok=not issues,
               pair_issue=" | ".join(f"{k}: {v}" for k, v in issues.items()))
    if not pool:
        out.update(delta=np.nan, delta_beq=np.nan, lo1=np.nan, hi1=np.nan, lob1=np.nan, hib1=np.nan, lo2=np.nan, hi2=np.nan, lob2=np.nan,
                   hib2=np.nan, dist1=None, dist2=None, ref_cell=np.nan, ref_beq=np.nan, cov_cell=np.nan, cov_beq=np.nan, has2=False)
        return out
    d1 = LC.strat_mean([per1[t]["dist"] for t in pool]); d1b = LC.strat_mean([per1[t]["dist_beq"] for t in pool])
    lo1, hi1, _ = LC.ci95(d1); lob1, hib1, _ = LC.ci95(d1b)
    has2 = all(t in per2 for t in pool)
    if has2:
        d2 = LC.strat_mean([per2[t][0] for t in pool]); d2b = LC.strat_mean([per2[t][1] for t in pool])
        lo2, hi2, n2 = LC.ci95(d2); lob2, hib2, n2b = LC.ci95(d2b)
    else:
        d2 = d2b = None; lo2 = hi2 = lob2 = hib2 = np.nan; n2 = n2b = 0
    out.update(delta=float(np.mean([per1[t]["delta"] for t in pool])), delta_beq=float(np.mean([per1[t]["delta_beq"] for t in pool])),
               lo1=lo1, hi1=hi1, lob1=lob1, hib1=hib1, dist1=d1, lo2=lo2, hi2=hi2, lob2=lob2, hib2=hib2, dist2=d2, has2=has2,
               n_nan2=max(n2, n2b), ref_cell=float(np.mean(ref_c)), ref_beq=float(np.mean(ref_b)), cov_cell=float(np.mean(cov_c)),
               cov_beq=float(np.mean(cov_b)))
    return out


def contrast_verdict(res, metric, rel=LC.EQ_REL, rel_aux=LC.EQ_REL_AUX, require2=False):
    """4분 판정(2단이 있으면 2단이 주, 1단은 보조). 한계: 구간 점수·CRPS 는 기준 점수의 5 %(보조 10 %), se 는 0.5 cm.
    require2(개정 1, 계획서 2.6절에서 2단 CI 가 주인 확인적 대비 LGU-A1 과 A3): 2단 분포가 없으면 판정은 '판정 불가(2단 CI 없음)'이고
    1단 판정은 verdict_1stage 에만 둔다. 짝지음 점검(res['pair_ok'])이 거짓이면 '판정 불가(계산 실패)'(계획서 2.7절: 계산 실패는 지지 쪽으로
    세지 않는다)."""
    if metric == "se":
        mg = mg_aux = (LC.EQ_CM, LC.EQ_CM)
    else:
        mg, mg_aux = LC.rel_margin(res["ref_cell"], res["ref_beq"], rel), LC.rel_margin(res["ref_cell"], res["ref_beq"], rel_aux)
    ci1 = (res["lo1"], res["hi1"], res["lob1"], res["hib1"])
    v1 = LC.verdict4(*ci1, mg)
    if res.get("has2"):
        ci = (res["lo2"], res["hi2"], res["lob2"], res["hib2"]); kind = "2단"
    elif require2:
        ci = (np.nan,) * 4; kind = "1단만(판정에 쓰지 않음)"
    else:
        ci = ci1; kind = "1단"
    v = LC.verdict4(*ci, mg)
    if require2 and not res.get("has2"):
        v = "판정 불가(2단 CI 없음)"
    elif require2 and not res.get("pair_ok", True):
        v = "판정 불가(계산 실패)"
    hw = LC.half_width(*ci) if all(np.isfinite(ci)) else np.nan
    dist = res["dist2"] if res.get("has2") else (None if require2 else res["dist1"])
    decided = not str(v).startswith("판정 불가") and not str(v1).startswith("판정 불가")
    return dict(verdict=v, verdict_1stage=v1, ci_kind=kind, ci=ci, ci1=ci1, margin=mg, margin_aux=mg_aux, half_width=hw,
                eq_state=LC.eq_state(*ci, mg), eq_state_aux=LC.eq_state(*ci, mg_aux),
                note_cal="보정 불확실성 의존" if (kind == "2단" and decided and v != v1) else "",
                p_boot=LC.boot_p(dist) if dist is not None else np.nan)


A1_TEXT = {"지지": "지역 오프셋의 사후 분포를 쓰는 계층 모형 구간은 같은 라벨로 보정한 상수 폭 구간보다 구간 점수가 낮다",
           "동등": "계층 모형 구간과 상수 폭 보정 구간의 구간 점수는 5 % 한계 안에서 같다",
           "기각": "계층 모형 구간은 상수 폭 보정 구간보다 구간 점수가 높다(분산 성분 표 lgu_a_hyper.csv 병기)",
           "미결정": "계층 모형 구간과 상수 폭 보정 구간의 구간 점수 차이는 확인되지 않았다(분산 성분 표 lgu_a_hyper.csv 병기)"}
PRECHECK_REL = 0.10                                                    # 계획서 2.9절: 사전 점검의 반폭 비율 기준(본 실행 전 검정 불가 등록)


def a1_decide(per_n, pool_ok=True, splits_ok=True, ns=(10, 40)):
    """LGU-A1 의 판정(계획서 3.6절). per_n[n] = dict(verdict, cov_ok). 반환 (판정, 문장).
    풀 지역 3개 미만이나 분할 부족이면 판정 불가(계획서 명시 조건이 먼저다). 그다음 기각: 어느 n 이든 열세(다른 n 이 판정 불가이면
    '기각(다른 n 판정 불가)', 개정 1. 앞 판은 판정 불가 검사가 먼저 와 실제 열세를 가렸다). 판정 불가 대비가 있으면 판정 불가(행 없음,
    2단 CI 없음, 계산 실패는 지지 쪽으로 세지 않는다). 지지: 두 n 모두 우세이고 커버리지 조건 충족. 부분 지지: 한 n 만 (커버리지를 충족한)
    우세이고 다른 n 이 열세 아님. 동등: 두 n 모두 동등. 우세이나 커버리지 미달이면 '우세(커버리지 미달)'(지지 근거로 쓰지 않는다).
    그 밖은 미결정."""
    if not pool_ok:
        return "판정 불가(풀 지역 3개 미만)", ""
    if not splits_ok:
        return "판정 불가(분할 부족)", ""
    vs = [str(per_n.get(n, {}).get("verdict", "판정 불가")) for n in ns]
    undec = [v for v in vs if v.startswith("판정 불가")]
    if any(v == "열세" for v in vs):
        return ("기각(다른 n 판정 불가)" if undec else "기각"), A1_TEXT["기각"]
    if undec:
        return undec[0], ""
    cov = [bool(per_n[n].get("cov_ok", False)) for n in ns]
    strong = [n for n, v, c in zip(ns, vs, cov) if v == "우세" and c]
    weak = [n for n, v, c in zip(ns, vs, cov) if v == "우세" and not c]
    if len(strong) == len(ns):
        return "지지", A1_TEXT["지지"]
    if strong:
        return f"부분 지지(n = {strong[0]})", A1_TEXT["지지"] + f"(n = {strong[0]})"
    if all(v == "동등" for v in vs):
        return "동등", A1_TEXT["동등"]
    if weak:
        return "우세(커버리지 미달)", ""
    return "미결정", A1_TEXT["미결정"]


def _verdict_key(v):
    """판정 비교 키: 판정 종류(괄호 앞). '부분 지지' 는 괄호 안의 n 까지 비교한다(개정 1)."""
    s = str(v)
    cls = s.split("(")[0]
    return s if cls == "부분 지지" else cls


def alaska_note(verdict3, verdict2):
    """풀 3지역(레나, 캐나다, 알래스카)과 레나·캐나다 2지역 평균의 판정이 다르면 '알래스카 의존'. 판정 종류를 비교하고, 부분 지지는
    지지하는 n 까지 비교한다(괄호 안의 다른 보충 설명은 비교하지 않는다)."""
    return "알래스카 의존" if _verdict_key(verdict3) != _verdict_key(verdict2) else ""


def pool_text(res):
    return "" if res["n_pool"] >= res["n_regions"] else f"부분(지역 {res['n_pool']}/{res['n_regions']}): "


def test_row(test, role, contrast, pool_name, res, vd, n, metric, statement="", blind="맹검", nboot=0, **kw):
    ci = vd["ci"] if vd else (np.nan,) * 4
    ci1 = vd["ci1"] if vd else (np.nan,) * 4
    sp = res.get("splits", {})
    row = dict(test=test, role=role, contrast=contrast, pool=pool_name, regions=",".join(res.get("pool", [])), n=n, metric=metric,
               delta=res.get("delta", np.nan), ci_lo=ci[0], ci_hi=ci[1], delta_beq=res.get("delta_beq", np.nan), ci_lo_beq=ci[2],
               ci_hi_beq=ci[3], ci_kind=vd["ci_kind"] if vd else "", verdict=vd["verdict"] if vd else "판정 불가",
               verdict_1stage=vd["verdict_1stage"] if vd else "", note_cal=vd["note_cal"] if vd else "",
               ci_lo_1stage=ci1[0], ci_hi_1stage=ci1[1], ci_lo_beq_1stage=ci1[2], ci_hi_beq_1stage=ci1[3],
               margin=json.dumps(vd["margin"]) if vd else "", margin_aux=json.dumps(vd["margin_aux"]) if vd else "",
               half_width=vd["half_width"] if vd else np.nan, ref_score=res.get("ref_cell", np.nan), ref_score_beq=res.get("ref_beq", np.nan),
               eq_state=vd["eq_state"] if vd else "", eq_state_aux=vd["eq_state_aux"] if vd else "", coverage_pool=res.get("cov_cell", np.nan),
               coverage_pool_beq=res.get("cov_beq", np.nan), cov_ok=np.nan, p_boot=vd["p_boot"] if vd else np.nan, holm_p=np.nan,
               pool_regions=res.get("n_pool", 0), splits_min=min([v[0] for v in sp.values()], default=0),
               splits_expected=max([v[1] for v in sp.values()], default=0), splits_complete=splits_complete(res),
               pair_ok=bool(res.get("pair_ok", True)), pair_issue=res.get("pair_issue", ""), verdict_2region="", note_alaska="",
               statement=statement, blind=blind, nboot=int(nboot),
               excluded="; ".join(f"{k}: {v}" for k, v in res.get("why", {}).items()))
    row.update(kw)
    return row


def splits_complete(res):
    """풀 지역마다 사용한 분할 수가 그 지역의 기대 분할 수 이상인지(지역별 비교, LGX 6A.5a 의 분할 완결성 규칙)."""
    return all(v[0] >= v[1] for v in res.get("splits", {}).values()) and bool(res.get("splits"))


def _prec_row(test, contrast, metric, pool, n, weight, kind, hw, ref, nb, n_pool, rel=True):
    """정밀도 표의 한 행(계획서 6절: 반폭만. 개정 1: 기준 점수는 쓰지 않는다). 구간 점수 대비는 반폭의 기준 점수 대비 비율만,
    점 예측(se) 대비는 반폭(cm)만 쓴다(둘을 함께 쓰면 기준 점수가 역산된다). precheck_rule_10pct 는 계획서 2.9절의 사전 점검 확정 규칙
    (LGU-A1, 비율 10 % 초과면 본 실행 전 동등 판정을 검정 불가로 등록), main_rule_5pct 는 2.7절 본 실행 정밀도 규칙(주 한계)이다."""
    if rel:
        r = hw / abs(ref) if (np.isfinite(hw) and np.isfinite(ref) and ref != 0) else np.nan
        main = "판정 불가" if not np.isfinite(r) else ("검정 불가(정밀도 미달)" if r > LC.EQ_REL else "검정 가능")
        pre = ("" if test != "LGU-A1" else "판정 불가" if not np.isfinite(r) else
               "본 실행 전 동등 판정 검정 불가 등록" if r > PRECHECK_REL else "등록 없음")
        return dict(test=test, contrast=contrast, metric=metric, pool=pool, n=n, weight=weight, ci_kind=kind, half_width_rel=r,
                    half_width_abs=np.nan, margin_kind="rel", margin=LC.EQ_REL, precheck_rule_10pct=pre, main_rule_5pct=main, nboot=nb,
                    pool_regions=n_pool)
    main = "판정 불가" if not np.isfinite(hw) else ("검정 불가(정밀도 미달)" if hw > LC.EQ_CM else "검정 가능")
    return dict(test=test, contrast=contrast, metric=metric, pool=pool, n=n, weight=weight, ci_kind=kind, half_width_rel=np.nan,
                half_width_abs=hw, margin_kind="abs_cm", margin=LC.EQ_CM, precheck_rule_10pct="", main_rule_5pct=main, nboot=nb,
                pool_regions=n_pool)


def _flow_seeds_ok(ts):
    """흐름 조각의 유효 seed(행이 있는 seed)가 2개 이상인지(계획서 2.10절)."""
    return ts is not None and ts.flow is not None and len(ts.flow.get("valid_seeds") or []) >= 2


def build_tests(a, TSs):
    """LGU-A1–A7 의 판정 표. 반환 (tests DataFrame, sens 행, 정밀도 행)."""
    rows, prec = [], []
    nb = int(a.nboot)
    # ---- LGU-A1(확인적): 단 (iii) − B4, 구간 점수(α 0.1), 풀 레나·캐나다·알래스카, n 10·40, 2단 CI
    per_n, per_n2, a1rows, r3s, r2s = {}, {}, [], {}, {}
    for n in (10, 40):
        r3 = pool_contrast(a, TSs, POOL3, "iii", "B4", n, "is10")
        r2 = pool_contrast(a, TSs, POOL2, "iii", "B4", n, "is10")
        r3s[n], r2s[n] = r3, r2
        v3, v2 = contrast_verdict(r3, "is10", require2=True), contrast_verdict(r2, "is10", require2=True)
        cov_ok = bool(np.isfinite(r3["cov_cell"]) and r3["cov_cell"] >= LC.COV_MIN)
        per_n[n] = dict(verdict=v3["verdict"], cov_ok=cov_ok)
        per_n2[n] = dict(verdict=v2["verdict"], cov_ok=bool(np.isfinite(r2["cov_cell"]) and r2["cov_cell"] >= LC.COV_MIN))
        a1rows.append(test_row("LGU-A1", "확인적(주 대비)", "iii − B4", "Lena·Canada·Alaska", r3, v3, n, "is10", nboot=nb,
                               cov_ok=cov_ok, verdict_2region=v2["verdict"]))
        for res_, pname in ((r3, "POOL3"), (r2, "POOL2")):
            for kind in ("2단", "1단"):
                ci = (res_["lo2"], res_["hi2"], res_["lob2"], res_["hib2"]) if kind == "2단" else (res_["lo1"], res_["hi1"], res_["lob1"], res_["hib1"])
                if not all(np.isfinite(ci)):
                    continue
                for w, (lo, hi, ref) in (("cell", (ci[0], ci[1], res_["ref_cell"])), ("beq", (ci[2], ci[3], res_["ref_beq"]))):
                    prec.append(_prec_row("LGU-A1", "iii − B4", "is10", pname, n, w, kind, (hi - lo) / 2.0, ref, nb, res_["n_pool"]))
    pool_ok = all(r3s[n]["n_pool"] == len(POOL3) for n in (10, 40))
    splits_ok = all(splits_complete(r3s[n]) for n in (10, 40))
    verdict, text = a1_decide(per_n, pool_ok, splits_ok)
    verdict2, _ = a1_decide(per_n2, all(r2s[n]["n_pool"] == len(POOL2) for n in (10, 40)),
                            all(splits_complete(r2s[n]) for n in (10, 40)))
    note = alaska_note(verdict, verdict2)
    for r in a1rows:
        r["note_alaska"] = note
    rows += a1rows
    rows.append(dict(test="LGU-A1", role="확인적(주 대비)", contrast="iii − B4", pool="Lena·Canada·Alaska", n="10,40", metric="is10",
                     verdict=verdict, verdict_2region=verdict2, note_alaska=note, statement=text, blind="맹검", nboot=nb,
                     ci_kind="2단" if all(r["ci_kind"] == "2단" for r in a1rows) else "2단 없음",
                     splits_complete=bool(splits_ok), pair_ok=all(bool(r["pair_ok"]) for r in a1rows),
                     note_cal="보정 불확실성 의존" if any(r["note_cal"] for r in a1rows) else ""))
    # ---- LGU-A2(구현 검증): n = 3 의 무한 구간 비율(B4, 단 i–iv)
    for mth in ("B4", "i", "ii", "iii", "iv"):
        vals, tg = [], []
        for (t, m), ts in TSs.items():
            lp = level_point(ts.by(t), kfn(mth, "", 3), "inf10")
            if lp["n_splits"]:
                vals.append(lp["cell"]); tg.append(f"{t}:{m}")
        v = ("판정 불가(행 없음)" if not vals else "통과(무한 구간 0)" if all(x == 0 for x in vals) else
             "구현 결함(무한 구간 있음. 고친 뒤 해석한다)")
        rows.append(dict(test="LGU-A2", role="구현 검증", contrast=f"{mth} inf10", pool="전 대상", regions=",".join(tg), n=3, metric="inf10",
                         delta=float(np.mean(vals)) if vals else np.nan, verdict=v, blind="맹검", nboot=0,
                         statement="커버리지는 lgu_a_curve.csv 의 CI 와 함께 서술한다"))
    # ---- LGU-A3(보조, 방향 중립): 중앙값 RMSE 의 iii − P1, 주 4지역과 3지역, n 3·10, δ 0.5 cm, 2단 CI(계획서 2.6절), Holm 보조 열
    a3 = []
    for n in (3, 10):
        for regs, pname in ((MAIN4, "주 4지역"), (MAIN3, "러시아 W 제외 3지역")):
            r = pool_contrast(a, TSs, regs, "iii", "P1", n, "se")
            vd = contrast_verdict(r, "se", require2=True)
            a3.append(test_row("LGU-A3", "보조(방향 중립)", "iii − P1(RMSE)", pname, r, vd, n, "se", nboot=nb,
                               statement=pool_text(r) + "4분 판정을 그대로 보고한다(우세 가설 없음)"))
            for kind, ci in (("2단", (r["lo2"], r["hi2"], r["lob2"], r["hib2"])), ("1단", (r["lo1"], r["hi1"], r["lob1"], r["hib1"]))):
                if all(np.isfinite(ci)):
                    prec.append(_prec_row("LGU-A3", "iii − P1", "se", pname, n, "max", kind, LC.half_width(*ci), np.nan, nb, r["n_pool"],
                                          rel=False))
    hp3 = holm(np.array([r["p_boot"] for r in a3], float)) if a3 else []
    for r, h_ in zip(a3, hp3):
        r["holm_p"] = h_
    rows += a3
    # ---- LGU-A4(게이트): 대상별 상태(통과, 미통과, 판정 불가)와 요약 행
    for (t, m), ts in TSs.items():
        if ts.gate is None:
            if ts.gate_status == "failed":
                rows.append(dict(test="LGU-A4", role="게이트", contrast="v − iv(crps_log, blk)", pool=f"{t}:{m}", n=0, metric="crps_log",
                                 ci_kind="1단", verdict="판정 불가(게이트 조각 실패)", blind="맹검", nboot=nb,
                                 statement="게이트 조각이 status failed 로 끝났다(--resume 으로 다시 실행한다)"))
            continue
        g = ts.gate.get("gate", {})
        b = g.get("blk", {})
        stt = str(ts.gate.get("gate_state") or g.get("gate_state") or ("통과" if ts.gate.get("gate_pass") else "미통과"))
        rows.append(dict(test="LGU-A4", role="게이트", contrast="v − iv(crps_log, blk)", pool=f"{t}:{m}", regions=",".join(b.get("regions", [])),
                         n=0, metric="crps_log", delta=b.get("delta", np.nan), ci_lo=b.get("ci_lo", np.nan), ci_hi=b.get("ci_hi", np.nan),
                         delta_beq=b.get("delta_beq", np.nan), ci_lo_beq=b.get("ci_lo_beq", np.nan), ci_hi_beq=b.get("ci_hi_beq", np.nan),
                         ci_kind="1단", verdict=stt + ("(강제: 통과)" if g.get("forced") == "pass" else "(강제: 미통과)" if g.get("forced") == "fail" else ""),
                         statement=f"유효 seed {g.get('n_valid_seeds', 0)} · 흐름 실행 {'예' if ts.gate.get('gate_pass') else '아니오'}",
                         blind="맹검", nboot=nb))
    g_main = {t: TSs.get((t, "x")) for t in MAIN5}
    has_gate = {t: (ts_ is not None and ts_.gate is not None) for t, ts_ in g_main.items()}
    g_state = {t: str(ts_.gate.get("gate_state") or ("통과" if ts_.gate.get("gate_pass") else "미통과")) if has_gate[t] else ""
               for t, ts_ in g_main.items()}
    if not any(ts_.gate for ts_ in TSs.values()):
        rows.append(dict(test="LGU-A4", role="게이트", verdict="미실행(게이트 조각 없음)", blind="맹검"))
    elif not any(ts_.gate and ts_.gate.get("gate_pass") for ts_ in TSs.values()):
        if all(has_gate.values()) and all(v == "미통과" for v in g_state.values()):
            rows.append(dict(test="LGU-A4", role="게이트", verdict="전 대상 미통과",
                             statement="조건부 흐름을 셀 잔차 분포로 쓴 구성은 이분산 모수 모형 대비 이득이 확인되지 않아 대상 채점을 하지 않았다",
                             blind="맹검"))
        else:
            miss = [t for t, v in has_gate.items() if not v]
            und = [f"{t}({v})" for t, v in g_state.items() if v and v != "미통과"]
            rows.append(dict(test="LGU-A4", role="게이트", verdict="판정 불가(통과한 대상 없음. 게이트가 없거나 판정 불가인 주 대상 있음)",
                             statement=("게이트 조각 없음: " + ",".join(miss) if miss else "") + (" · 판정 불가: " + ",".join(und) if und else "")
                             + ". '대상 채점을 하지 않았다' 문장은 쓰지 않는다(계산 실패는 가설 쪽으로 세지 않는다)",
                             blind="맹검"))
    # ---- LGU-A5(탐색): 민감도 표만
    sens = []
    for (t, m), ts in TSs.items():
        for scope, name in (("B", t), ("all", f"{t}|all"), ("B", f"{t}~noearly"), ("all", f"{t}|all~noearly")):
            bs = ts.by(name)
            if not bs:
                continue
            keys = sorted({(k[0], k[1]) for st in bs.values() for k in st.keys if int(k[2]) == 0 and k[0] in ("iii", "ii", "B4")
                           and not str(k[1]).startswith(("#", "aux:"))})
            for mth, var in keys:
                row = dict(target=t, mode=m, scope=scope, store=name, method=mth, variant=var or "main", n=0)
                for met in ("is10", "cov10", "wid10", "lhw10", "is20", "cov20", "crps_log"):
                    lp = level_point(bs, kfn(mth, var, 0), met)
                    row[met], row[f"{met}_beq"] = lp["cell"], lp["beq"]
                    if met in ("is10", "cov10"):
                        d_ = level_ci1(bs, kfn(mth, var, 0), met, nb, seed_of("lgu-a", t, m))
                        if d_ is not None:
                            row[f"{met}_lo"], row[f"{met}_hi"] = LC.ci95(d_[0])[:2]
                sens.append(row)
    rows.append(dict(test="LGU-A5", role="탐색", verdict="표만 보고(lgu_a_sens.csv)", statement="커버리지 주장을 하지 않는다", blind="맹검"))
    # ---- LGU-A6(서술): 러시아 W 의 n 3·10
    ts = TSs.get((RUSSIA_W, "x"))
    for n in (3, 10):
        for mth in ("iii", "B4"):
            if ts is None:
                rows.append(dict(test="LGU-A6", role="서술", contrast=mth, pool=RUSSIA_W, n=n, verdict="행 없음", blind="맹검"))
                continue
            bs = ts.by(RUSSIA_W)
            r_ = dict(test="LGU-A6", role="서술", contrast=mth, pool=RUSSIA_W, n=n, verdict="서술(범위 판정 없음)", blind="맹검", nboot=nb)
            for met in ("is10", "cov10", "wid10"):
                lp = level_point(bs, kfn(mth, "", n), met)
                d_ = level_ci1(bs, kfn(mth, "", n), met, nb, seed_of("lgu-a", RUSSIA_W, "x"))
                lo, hi = LC.ci95(d_[0])[:2] if d_ is not None else (np.nan, np.nan)
                r_.update({met: lp["cell"], f"{met}_beq": lp["beq"], f"{met}_lo": lo, f"{met}_hi": hi})
            rows.append(r_)
    # ---- LGU-A7(보조): 사다리 대비, 풀 레나·캐나다·알래스카, n 0·10·40, is10 과 crps_log
    a7 = []
    flow_all = all(TSs.get((t, "x")) is not None and TSs[(t, "x")].gate is not None and TSs[(t, "x")].gate.get("gate_pass") for t in POOL3)
    for A, B in LADDER:
        for n in A7_N:
            for met in ("is10", "crps_log"):
                if met == "crps_log" and "iii+c" in (A, B):
                    continue
                if A == "v" and not flow_all:
                    for t in POOL3:
                        tsx = TSs.get((t, "x"))
                        if tsx is None or tsx.gate is None or not tsx.gate.get("gate_pass"):
                            continue
                        r = pool_contrast(a, TSs, [t], A, B, n, met)
                        vd = contrast_verdict(r, met)
                        if not _flow_seeds_ok(tsx):
                            vd = dict(vd, verdict="판정 불가(적합 실패)")
                        a7.append(test_row("LGU-A7", "보조(대상별 서술)", f"{A} − {B}", t, r, vd, n, met, nboot=nb,
                                           statement="풀 지역 3개가 모두 게이트를 통과하지 않아 대상별로 적는다"
                                           + ("" if _flow_seeds_ok(tsx) else ". 흐름 유효 seed 2개 미만(계획서 2.10절)")))
                    continue
                r = pool_contrast(a, TSs, POOL3, A, B, n, met)
                vd = contrast_verdict(r, met)
                st_ = pool_text(r)
                if A == "v" and not all(_flow_seeds_ok(TSs.get((t, "x"))) for t in POOL3):
                    vd = dict(vd, verdict="판정 불가(적합 실패)")
                    st_ += "흐름 유효 seed 2개 미만인 풀 지역이 있다(계획서 2.10절)"
                a7.append(test_row("LGU-A7", "보조", f"{A} − {B}", "Lena·Canada·Alaska", r, vd, n, met, nboot=nb, statement=st_))
    ps = np.array([r["p_boot"] for r in a7], float)
    hp = holm(ps) if len(ps) else ps
    for r, h_ in zip(a7, hp):
        r["holm_p"] = h_
    rows += a7
    return pd.DataFrame(rows), sens, prec


def build_curve(a, TSs):
    """대상·범위·방법·변형·n 별 지표(두 가중), 1단 CI(CURVE_CI_METRICS), 2단 CI(주 대상, 방법 P1·B4·i·ii·iii, 변형 '').
    net_is10 은 같은 저장소·방법·변형의 n = 0 대비 구간 점수 차(셀 가중)다."""
    out = []
    nb = int(a.nboot)
    for (t, m), ts in TSs.items():
        for scope, name in (("B", t), ("all", f"{t}|all"), ("B", f"{t}~noearly"), ("all", f"{t}|all~noearly")):
            bs = ts.by(name)
            if not bs:
                continue
            groups = sorted({(str(k[0]), str(k[1]), int(k[2])) for st in bs.values() for k in st.keys})
            part = []
            for mth, var, n in groups:
                fn = kfn(mth, var, n)
                row = dict(target=t, mode=m, scope=scope, store=name, method=mth, variant=var, n=n)
                for met in LC.METRICS:
                    lp = level_point(bs, fn, met)
                    row[met], row[f"{met}_beq"] = lp["cell"], lp["beq"]
                row["rmse"], row["rmse_beq"] = row["se"], row["se_beq"]
                lp = level_point(bs, fn, "is10")
                row.update(n_runs=lp["n_keys"], n_splits=lp["n_splits"], n_eval=lp["n_eval"], n_blocks=lp["n_blocks"])
                for met in CURVE_CI_METRICS:
                    d_ = level_ci1(bs, fn, met, nb, seed_of("lgu-a", t, m))
                    if d_ is not None:
                        row[f"{met}_lo1"], row[f"{met}_hi1"] = LC.ci95(d_[0])[:2]
                        row[f"{met}_lo1_beq"], row[f"{met}_hi1_beq"] = LC.ci95(d_[1])[:2]
                kind = "1단"
                if scope == "B" and name == t and var == "" and mth in TWO_STAGE_METHODS:
                    for met in M2:
                        d2 = boot_level(ts.boot, mth, n, met)
                        if d2 is not None:
                            row[f"{met}_lo2"], row[f"{met}_hi2"] = LC.ci95(d2[0])[:2]
                            row[f"{met}_lo2_beq"], row[f"{met}_hi2_beq"] = LC.ci95(d2[1])[:2]
                            kind = "2단+1단"
                row["ci_kind"] = kind
                row["in_band"] = bool(np.isfinite(row["cov10"]) and 0.85 <= row["cov10"] <= 0.95)
                row["flag"] = ("하위 지역(보충, 독립 아님)" if t in SUB10 else "") + ("" if "2단" in kind else " 2단 없음")
                part.append(row)
            is0 = {(r["method"], r["variant"]): r["is10"] for r in part if r["n"] == 0}
            for r in part:
                ref = is0.get((r["method"], r["variant"]))
                r["net_is10"] = r["is10"] - ref if ref is not None and np.isfinite(ref) else np.nan
            out += part
    return pd.DataFrame(out)


def build_hyper(TSs):
    rows = []
    for (t, m), ts in TSs.items():
        mt = ts.emu_meta
        if not mt:
            continue
        for v, h in (mt.get("hyper") or {}).items():
            if not h:
                continue
            tq = (mt.get("tau_q") or {}).get(v, [np.nan] * 3)
            base = dict(target=t, mode=m, variant=v, E0=mt.get("E0"), K=h.get("K"), sigma2=h.get("sigma2"), tau_b2=h.get("tau_b2"),
                        tau_r2_plugin=h.get("tau_r2"), sigma1_2=h.get("sigma1_2"), tau_r1_2=h.get("tau_r1_2"), tau_r_q05=tq[0], tau_r_q50=tq[1],
                        tau_r_q95=tq[2], flags=";".join(h.get("flags", [])))
            rows.append(dict(base, level="target"))
            for j, r in enumerate(h.get("regions", [])):
                rows.append(dict(target=t, mode=m, variant=v, level="region", region=r, E0_mk=(mt.get("E0_mk") or {}).get(r),
                                 a_k=h["a_k"][j], v_k=h["v_k"][j], sigma_k2=h["sigma_k2"][j], df_k=h["df_k"][j], tau_bk2=h["tau_bk2"][j],
                                 n_blocks=h["n_blocks"][j], n_cells=h["n_cells"][j], a_cell=h["a_cell"][j]))
        for f in mt.get("iv", []):
            rows.append(dict(target=t, mode=m, variant="iv", level="scale", seed=f.get("seed"), scale_c=f.get("c"), nu_eps=f.get("nu"),
                             ll_nu3=(f.get("ll") or [np.nan] * 4)[0], ll_nu5=(f.get("ll") or [np.nan] * 4)[1],
                             ll_nu10=(f.get("ll") or [np.nan] * 4)[2], ll_nuinf=(f.get("ll") or [np.nan] * 4)[3], fail=f.get("fail", "")))
        ph = mt.get("prior_h4") or {}
        rows.append(dict(target=t, mode=m, variant="i0", level="h4_prior", tau2=ph.get("tau2"), sigma2=ph.get("sigma2"),
                         K=ph.get("n_regions"), phys_trunc=json.dumps(mt.get("phys_trunc")), cn=json.dumps(mt.get("cn")),
                         flags=";".join(mt.get("flags", []))))
    return pd.DataFrame(rows)


def build_gate(TSs):
    rows = []
    for (t, m), ts in TSs.items():
        if ts.gate is None:
            continue
        g = ts.gate.get("gate", {})
        for var in ("blk", "res", "n0"):
            s_ = g.get(var, {})
            rows.append(dict(target=t, mode=m, level="pool", variant=var, delta=s_.get("delta"), ci_lo=s_.get("ci_lo"), ci_hi=s_.get("ci_hi"),
                             delta_beq=s_.get("delta_beq"), ci_lo_beq=s_.get("ci_lo_beq"), ci_hi_beq=s_.get("ci_hi_beq"),
                             regions=",".join(s_.get("regions", [])), gate_pass=ts.gate.get("gate_pass"),
                             gate_state=ts.gate.get("gate_state") or g.get("gate_state", ""), n_valid_seeds=g.get("n_valid_seeds"),
                             iv_ok=g.get("iv_ok"), forced=g.get("forced", "")))
            for k, r in (s_.get("per_region") or {}).items():
                rows.append(dict(target=t, mode=m, level="region", variant=var, region=k, **{kk: (json.dumps(vv) if isinstance(vv, list) else vv)
                                                                                            for kk, vv in r.items()}))
        for r in ts.gate.get("regions", []):
            rows.append(dict(target=t, mode=m, level="heldout_info", **{kk: vv for kk, vv in r.items()}))
    return pd.DataFrame(rows)


def build_pit(runs):
    if runs.empty or "pit0" not in runs:
        return pd.DataFrame()
    pc = [f"pit{j}" for j in range(LC.PIT_BINS)]
    r = runs[runs[pc[0]] >= 0]
    g = r.groupby(["target", "mode", "scope", "store", "method", "variant", "n"], dropna=False)[pc].sum().reset_index()
    tot = g[pc].sum(1).replace(0, np.nan)
    for c in pc:
        g[f"{c}_frac"] = g[c] / tot
    return g


PREC_COLS = ("test", "contrast", "metric", "pool", "n", "weight", "ci_kind", "half_width_rel", "half_width_abs", "margin_kind", "margin",
             "precheck_rule_10pct", "main_rule_5pct", "nboot", "pool_regions")


def precision_frame(rows):
    """정밀도 표(--precision-only): 반폭만 남긴다(계획서 6절). 구간 점수 대비는 기준 점수 대비 비율(half_width_rel)만, 점 예측 대비는 반폭
    (half_width_abs, cm)만 쓴다. 대비의 점 추정(delta), CI 끝점, 판정, 기준 점수(ref_score) 열은 쓰지 않는다(개정 1)."""
    df = pd.DataFrame(list(rows))
    return df[[c for c in PREC_COLS if c in df.columns]] if len(df) else pd.DataFrame(columns=list(PREC_COLS))


def find_shards(a):
    out = {k: [] for k in SHARD_FILES}
    if not a.SHARDS.exists():
        return out
    for p in sorted(a.SHARDS.glob(f"{a.TAG}__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")
        if parts[0] != a.TAG or len(parts) not in (4, 5) or parts[1] not in SHARD_FILES:
            continue
        sp = int(parts[4][1:]) if len(parts) == 5 else None
        out[parts[1]].append(dict(kind=parts[1], target=parts[2], mode=parts[3], split=sp, unit=read_unit(p),
                                  paths=shard_paths(a, parts[1], parts[2], parts[3], sp)))
    return out


def summarize(a, elapsed=0.0):
    """조각을 읽어 1단·2단 CI 와 판정 표를 만든다. 설정 해시가 섞여 있으면 중단한다. 실패 단위가 있으면 n_fail > 0(종료 코드 1)."""
    t0 = time.time()
    sh = find_shards(a)
    if not sh["cpu"] and not sh["emu"]:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{a.TAG}__*", flush=True)
        return dict(n_fail=1)
    for kind, lst in sh.items():
        hs = Counter(str(s_["unit"].get("cfg_hash", "none")) for s_ in lst)
        if len(hs) > 1:
            raise SystemExit(f"[summarize] {kind} 조각의 설정 해시가 {len(hs)}종이다 {dict(hs)}. 설정이 다른 실행이 섞였다")
        cur = LC.cfg_hash(unit_cfg(a, kind))
        if hs and cur not in hs:
            print(f"  [warn] {kind} 조각의 설정 해시 {list(hs)} 가 현재 인자의 해시 {cur} 와 다르다(집계 인자를 확인한다)", flush=True)
    D = get_D(a)
    LC.check_blockindex(a.OUT / "lgu_blockindex.csv", frame_of(a).bi)       # h45 와 같은 대조 파일(자료 판이 다르면 중단)
    failed = []
    done = {k: {(s_["target"], s_["mode"], s_["split"]): s_ for s_ in lst} for k, lst in sh.items()}
    cpu_targets = [(t, m) for t, m in _targets(a.targets, "cpu")] if not (a.smoke or a.precheck) else \
        ([(t, m) for t, m in a.TARGETS] if a.part == "cpu" else [])
    if not cpu_targets:
        cpu_targets = sorted({(s_["target"], s_["mode"]) for s_ in sh["cpu"]})
    TSs = {}
    for t, m in cpu_targets:
        ts = TS(t, m)
        e = done["emu"].get((t, m, None))
        if e is None or e["unit"].get("status") == "failed":
            failed.append(dict(kind="emu", target=t, mode=m, split=-1, status="missing" if e is None else "failed",
                               error=(e or {}).get("unit", {}).get("error", "")))
        else:
            ts.emu_meta = load_emu(e["paths"]["emu"])["meta"]
        run, _ = cpu_splits(a, D, t)
        ts.splits_expected = run
        paths = []
        for sp in run:
            c = done["cpu"].get((t, m, sp))
            if c is None or c["unit"].get("status") == "failed":
                failed.append(dict(kind="cpu", target=t, mode=m, split=sp, status="missing" if c is None else "failed",
                                   error=(c or {}).get("unit", {}).get("error", "")))
                continue
            paths.append(c["paths"]["scores"])
            ts.fails += [dict(f_, split=sp) for f_ in (c["unit"].get("fail") or [])]
            if c["paths"]["runs"].exists():
                ts.runs.append(pd.read_csv(c["paths"]["runs"]))
        g = done["gate"].get((t, m, None))
        if g is not None:
            if g["unit"].get("status") == "failed":
                ts.gate_status = "failed"
                failed.append(dict(kind="gate", target=t, mode=m, split=-1, status="failed",
                                   error=g["unit"].get("error", "") or g["unit"].get("gate_state", "")))
            else:
                ts.gate = g["unit"]
        f = done["flow"].get((t, m, None))
        if f is not None:
            st_ = f["unit"].get("status")
            ok_f, why_f = unit_done(a, "flow", t, m)
            if st_ == "failed":
                failed.append(dict(kind="flow", target=t, mode=m, split=-1, status="failed", error=f["unit"].get("error", "")))
            elif not ok_f and "gate_ref" in why_f:
                failed.append(dict(kind="flow", target=t, mode=m, split=-1, status="stale", error=why_f))
            elif st_ != "skipped_gate":
                paths.append(f["paths"]["scores"]); ts.flow = f["unit"]
                if f["paths"]["runs"].exists():
                    ts.runs.append(pd.read_csv(f["paths"]["runs"]))
        if paths:
            ts.stores = LC.load_score_stores(paths)
        if ts.stores or ts.emu_meta:
            TSs[(t, m)] = ts
    # 2단 재표집(주 대상 5개, 모드 x)
    boot_tasks = [("boot", t, m, None) for (t, m), ts in TSs.items() if is_main(t, m) and ts.stores and ts.emu_meta]
    if a.resume:
        boot_tasks = [u for u in boot_tasks if load_boot(a, u[1], u[2]) is None]
    if a.precision_only:
        boot_tasks = [u for u in boot_tasks if u[1] in POOL3 + MAIN4]
    for u in boot_tasks:                                                        # 다시 계산할 분포의 옛 파일은 지운다(실패하면 옛 분포를 쓰지 않는다)
        boot_path(a, u[1], u[2]).unlink(missing_ok=True)
    print(f"[summarize] 대상 {len(TSs)} · 실패·없는 단위 {len(failed)} · 2단 재표집 작업 {len(boot_tasks)}(재표집 {a.nboot})", flush=True)
    bdone, bfail = run_tasks(a, list(getattr(a, "_argv", [])), boot_tasks, False, False, t0)
    for u in bfail:
        failed.append(dict(u, status="boot_failed"))
    for (t, m), ts in TSs.items():
        ts.boot = load_boot(a, t, m) if is_main(t, m) else None
    tests, sens, prec = build_tests(a, TSs)
    a.OUT.mkdir(parents=True, exist_ok=True)
    if a.precheck:                                                              # 계획서 2.9절: 구적 격자 간격의 확정 근거(합성 자료만 쓴다)
        qc = quadrature_check()
        atomic_csv(out_name(a, "quadrature"), qc)
        print(qc.to_string(index=False), flush=True)
    if a.precision_only:
        pdf = precision_frame(prec)
        atomic_csv(out_name(a, "precision"), pdf)
        atomic_csv(out_name(a, "timing"), pd.DataFrame(bdone))
        print(f"[summarize] 정밀도 표만 썼다(대비의 점 추정, CI 끝점, 판정은 쓰지 않는다): {out_name(a, 'precision')}", flush=True)
        if len(pdf):
            print(pdf.to_string(index=False), flush=True)
        return dict(n_fail=len(failed), precision=pdf)
    curve = build_curve(a, TSs)
    runs = pd.concat([pd.concat(ts.runs, ignore_index=True) for ts in TSs.values() if ts.runs], ignore_index=True) if any(ts.runs for ts in TSs.values()) else pd.DataFrame()
    atomic_csv(out_name(a, "curve"), curve)
    atomic_csv(out_name(a, "tests"), tests)
    atomic_csv(out_name(a, "sens"), pd.DataFrame(sens))
    atomic_csv(out_name(a, "precision"), precision_frame(prec))
    atomic_csv(out_name(a, "hyper"), build_hyper(TSs))
    atomic_csv(out_name(a, "gate"), build_gate(TSs))
    atomic_csv(out_name(a, "pit"), build_pit(runs))
    atomic_csv(out_name(a, "strata"), curve[curve.variant.astype(str).str.startswith("#")] if len(curve) else curve)
    tim = [dict(kind=s_["kind"], target=s_["target"], mode=s_["mode"], split=s_["split"] if s_["split"] is not None else -1,
                status=s_["unit"].get("status"), elapsed_s=s_["unit"].get("elapsed_s"), device=s_["unit"].get("device"),
                threads=s_["unit"].get("threads"), n_fit=json.dumps(s_["unit"].get("n_fit", {})))
           for lst in sh.values() for s_ in lst] + [dict(r, kind="boot") for r in bdone]
    atomic_csv(out_name(a, "timing"), pd.DataFrame(tim))
    atomic_csv(out_name(a, "failed"), pd.DataFrame(failed, columns=["kind", "target", "mode", "split", "status", "error"]))
    meta = dict(tag=a.TAG, nboot=int(a.nboot), args={k: v for k, v in vars(a).items() if (k.isupper() and k != "HA") or k in (
        "part", "kappa", "buffer_km", "min_cal_cells", "nu", "s_tau", "cap_cells", "clamp", "epochs", "threads", "workers", "gpus")},
                fixed_deviation=a.fixed_deviation, **code_shas(), inputs=input_hashes(a), n_targets=len(TSs), n_failed=len(failed),
                gate={f"{t}:{m}": dict(gate_pass=(ts.gate or {}).get("gate_pass"), gate_state=(ts.gate or {}).get("gate_state"))
                      for (t, m), ts in TSs.items() if ts.gate is not None},
                boot={f"{t}:{m}": (ts.boot or {}).get("meta") for (t, m), ts in TSs.items() if ts.boot is not None},
                blind=dict(A="맹검(LG·LGX 결과 열람 전 등록)", B1="비맹검"), elapsed_s=round(elapsed + time.time() - t0, 1))
    LC.atomic_text(out_name(a, "meta", "json"), json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[summarize] 곡선 {len(curve)}행 · 판정 {len(tests)}행 · 실패 {len(failed)} · {time.time() - t0:.0f}s → {a.OUT}", flush=True)
    v = tests[tests.test.isin(["LGU-A1", "LGU-A2", "LGU-A4"])][["test", "contrast", "n", "verdict"]] if len(tests) else tests
    if len(v):
        print(v.to_string(index=False), flush=True)
    return dict(n_fail=len(failed), curve=curve, tests=tests)


def quadrature_check(n_sim=1500, seed=3):
    """구적 격자 점검(계획서 2.9절, 결과 자료를 쓰지 않는다). (1) 단 (ii) 닫힌 형태와 격자 해(정규 사전, posterior_grid + predictive_q)의
    분위 최대 절대 차(기준 0.002 이하), (2) 단 (iii) 모형에서 생성한 합성 자료의 모의 기반 보정 점검(SBC): PIT 평균 0.5 ± 0.03,
    90 % 커버리지 0.90 ± 0.03. 반환 DataFrame(check, value, threshold, ok)."""
    tau_r2, tau_b2, sigma2 = 0.05, 0.03, 0.08
    pa = normal_prior_mass(tau_r2)
    mx = 0.0
    for n_j, ub in ((np.zeros(0), np.zeros(0)), (np.array([3.0]), np.array([0.2])), (np.array([5.0, 2.0, 8.0]), np.array([0.1, -0.3, 0.25])),
                    (np.full(20, 8.0), np.linspace(-0.4, 0.4, 20))):
        m, v = posterior_normal(n_j, ub, tau_r2, tau_b2, sigma2)
        q_gr = predictive_q(posterior_grid(pa, n_j, ub, tau_b2, sigma2), np.sqrt(tau_b2 + sigma2))
        mx = max(mx, float(np.max(np.abs(normal_q(m, v + tau_b2 + sigma2) - q_gr))))
    rng = np.random.RandomState(seed)
    tg, pt = tau_r_posterior([0.2, -0.15, 0.05, 0.3], [0.002] * 4, FIXED["nu"], FIXED["s_tau"])
    pa3 = prior_a(tg, pt, FIXED["nu"])
    cdf_a = np.cumsum(pa3)
    sd = float(np.sqrt(0.03 + 0.06))
    pits, cov = [], []
    for _ in range(int(n_sim)):
        a_ = A_GRID[min(int(np.searchsorted(cdf_a, rng.rand())), len(A_GRID) - 1)]
        blk = np.repeat(np.arange(4), 3)
        u = a_ + rng.normal(0, np.sqrt(0.03), 4)[blk] + rng.normal(0, np.sqrt(0.06), len(blk))
        nJ, ub = label_blocks(u, blk)
        q = predictive_q(posterior_grid(pa3, nJ, ub, 0.03, 0.06), sd)
        us = a_ + rng.normal(0, sd)
        pits.append(float(np.interp(us, q, LC.TAUS, left=0.005, right=0.995))); cov.append(bool(q[4] <= us <= q[94]))
    pm, cv = float(np.mean(pits)), float(np.mean(cov))
    return pd.DataFrame([dict(check="closed_form_max_abs_diff", value=mx, threshold="<= 0.002", ok=mx <= 0.002, h=HG),
                         dict(check="sbc_pit_mean", value=pm, threshold="0.5 ± 0.03", ok=abs(pm - 0.5) <= 0.03, h=HG, n_sim=int(n_sim)),
                         dict(check="sbc_cov90", value=cv, threshold="0.90 ± 0.03", ok=abs(cv - 0.9) <= 0.03, h=HG, n_sim=int(n_sim))])


# ================================================================ 12. 적합 수(--count-only)
EST_S = dict(catboost=0.3, nflow=30.0, grid_post=0.003, iv_post=0.4, score=0.02)   # 추정용 기준 시간(s). 결과에는 쓰지 않는다


def count_only(a, D, fr):
    """학습 없이 작업 단위 수, 방법별 키 수, 적합 수, 추정 시간을 출력한다(자료 적재와 색인 계산만 한다)."""
    t0 = time.time()
    rows = []
    for t, m in a.TARGETS:
        t_idx, parent, src_idx, comp = D.source_idx(t, m)
        groups = LC.cal_groups(fr.mac, src_idx, fr.uok, a.min_cal_cells)
        K = len(groups)
        run, skipped = cpu_splits(a, D, t)
        info = D.split_structure(t)
        rg = rungs_for(a, t, m)
        n_nd = sum(len(H.cells_of(a.N_GRID, a.DRAWS, info[sp]["n_A"])) for sp in run)
        n_ev = n_drop = 0
        for sp in run:
            A_idx, B_idx = half_split_blocks(fr.df, t_idx, sp)
            evB = B_idx[eval_mask(fr.df.iloc[B_idx])]
            n_ev += int(fr.uok[evB].sum()); n_drop += int((~fr.uok[evB]).sum())
        plan = LC.emulation_plan(fr.df, src_idx, groups, a.N_GRID, t, m) if K >= 2 else []
        n_iii = 1 + sum(1 for v in a.SENS if v not in ("noearly",))
        n_aux = 2 * ("i" in rg) + 2 * ("ii" in rg) + ("iii" in rg) + ("i0" in rg)      # 보조 열 aux:tmean·aux:Emean(개정 1)
        per_nd = (len([r for r in rg if r not in ("iv", "iii+c")]) + (n_iii - 1) + (1 if "iii+c" in rg else 0)
                  + (len(a.CB_SEEDS) if "iv" in rg else 0) + 2 + 6 + n_aux)
        cb_iv = (K + 1) * len(a.CB_SEEDS) if (is_main(t, m) and "iv" in rg) else 0
        gate_nf = K * len(a.FLOW_SEEDS) if is_main(t, m) else 0
        gate_cb = K * K if is_main(t, m) else 0                         # 뺀 지역 K 마다 전체 적합 1 + 척도 보정 c 의 표본 밖 적합 K − 1(개정 1)
        flow_nf = len(a.FLOW_SEEDS) if is_main(t, m) else 0
        n_post = n_nd * n_iii + len(plan)
        n_ivpost = n_nd * len(a.CB_SEEDS) if "iv" in rg else 0
        est_cpu = (cb_iv * EST_S["catboost"] + n_post * EST_S["grid_post"] + n_ivpost * EST_S["iv_post"] + n_nd * per_nd * EST_S["score"])
        rows.append(dict(target=t, mode=m, n_src=int(len(src_idx)), K_groups=K, groups=",".join(groups), cpu_units=len(run),
                         splits_run=",".join(map(str, run)), splits_skipped=";".join(f"s{s_['split']}:{s_['status']}" for s_ in skipped),
                         n_nd=n_nd, n_eval=n_ev, n_eval_dropped=n_drop, emu_records=len(plan),
                         emu_entries=int(sum(len(r["cal_idx"]) for r in plan)), keys_est=int(n_nd * per_nd), fit_catboost_iv=cb_iv,
                         fit_nflow_gate=gate_nf, fit_catboost_gate=gate_cb, fit_nflow_flow_max=flow_nf, posteriors=n_post,
                         est_cpu_s=round(est_cpu, 1), est_gpu_s=round((gate_nf + flow_nf) * EST_S["nflow"], 1)))
    df = pd.DataFrame(rows)
    a.OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(out_name(a, "count"), df)
    print(df.drop(columns=["groups", "splits_run"]).to_string(index=False), flush=True)
    tot = df[["cpu_units", "fit_catboost_iv", "fit_nflow_gate", "fit_catboost_gate", "fit_nflow_flow_max", "posteriors", "est_cpu_s",
              "est_gpu_s"]].sum()
    print(f"[count-only] 대상 {len(df)} · emu 단위 {len(df)} · cpu 단위 {int(tot.cpu_units)} · 게이트·흐름 단위 최대 "
          f"{int((df.fit_nflow_gate > 0).sum())} · CatBoost(단 iv) {int(tot.fit_catboost_iv)} · 게이트 nflow {int(tot.fit_nflow_gate)}"
          f" + CatBoost {int(tot.fit_catboost_gate)} · 흐름 nflow 최대 {int(tot.fit_nflow_flow_max)} · 격자 사후 {int(tot.posteriors):,}", flush=True)
    print(f"[count-only] 추정(기준 시간 {EST_S}, 실측 아님): CPU 누적 {tot.est_cpu_s / 3600:.2f} h(2단 재표집 제외) · "
          f"GPU 누적 {tot.est_gpu_s / 3600:.2f} h · {time.time() - t0:.1f}s → {out_name(a, 'count')}", flush=True)
    return df


# ================================================================ 13. 실행
def _local_limits(a):
    """허용 표지가 있는 실행의 공유 서버 규칙(사용자 지시 2026-09-30): nice 10, 스레드 합계 상한(--max-threads-total, 기본 8).
    LG_RESCALE=1 로 허용된 실행에도 같은 규칙을 적용한다(개정 1). 동시에 도는 프로세스 수 × --threads 가 상한을 넘으면 --threads 를
    낮추지 않고 거부한다(spawn 워커는 부모의 argv 로 numpy 적재 전 스레드 수를 정하므로 낮춘 값이 BLAS 에 반영되지 않는다. h45·h46 과 같다).
    프로세스 수는 GPU 부분이면 GPU 프로세스 수, CPU 부분·집계면 --workers 이고, 끝에 자동 집계(2단 재표집 풀)를 하는 실행은 두 값의 큰 쪽이다.
    반환: 이 실행이 동시에 쓰는 스레드 수(등록부 claim_threads 에 적는 값)."""
    try:
        cur = os.nice(0)
        if cur < 10:
            os.nice(10 - cur)
    except OSError:
        pass
    th = max(1, int(a.threads))
    if a.count_only:
        nproc = 1
    else:
        use_gpu = a.part in ("gate", "flow") and bool(a.GPUS) and not a.summarize_only
        will_sum = bool(a.summarize_only or not a.no_summarize)
        n_gpu = len(a.GPUS) * max(1, int(a.procs_per_gpu)) if use_gpu else 0
        n_cpu = max(int(a.workers), 1) if (not use_gpu or will_sum) else 0
        nproc = max(n_gpu, n_cpu, 1)
    cap = int(a.max_threads_total)
    tot = nproc * th
    if tot > cap:
        raise SystemExit(f"[거부] 동시에 도는 프로세스 {nproc}개 × 스레드 {th} = {tot} 가 스레드 합계 상한 {cap} 을 넘는다"
                         f"(사용자 지시 2026-09-30). --workers, --procs-per-gpu, --threads 를 줄인다(예: --workers 2 --threads 4, "
                         f"--gpus 2 --threads 4 --workers 2)")
    return tot


def _check_gpus(a):
    """GPU 풀을 만들기 직전의 확인. --gpu-allow 밖의 번호를 거부하고 메모리 사용이 0 MiB 인 GPU 만 남긴다(LG_RESCALE=1 에도 적용, 개정 1)."""
    if not a.GPUS:
        return a.GPUS
    allow = [int(g) for g in str(a.gpu_allow).split(",") if g.strip()]
    bad = [g for g in a.GPUS if not g.isdigit() or int(g) not in allow]
    if bad:
        raise SystemExit(f"[거부] --gpus {bad} 는 이 실험에 배정된 GPU({allow})가 아니다")
    free = LC.free_gpus([int(g) for g in a.GPUS], max_mib=50.0)   # 빈 GPU 표시 2 MiB 때문에 50 MiB 기준(LGU 개정 3)
    busy = [g for g in a.GPUS if int(g) not in free]
    if busy:
        print(f"[local] 사용 중인 GPU {busy}(메모리 사용 0 MiB 아님)는 쓰지 않는다", flush=True)
    a.GPUS = [g for g in a.GPUS if int(g) in free]
    if not a.GPUS:
        raise SystemExit("[거부] 배정된 GPU 가 모두 사용 중이다. 비어 있을 때 다시 실행한다")
    return a.GPUS


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    a._argv = argv
    t0 = time.time()
    permitted = LC.require_permission(a)                                      # 허용 표지가 없으면 --count-only 만(자료 적재 전 거부)
    if a.GPUS and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        print("[warn] 부모 환경의 CUDA_VISIBLE_DEVICES 가 빈 문자열이다. --gpus 를 무시하고 CPU 로 돈다", flush=True)
        a.GPUS = []
    if a.part == "cpu" or a.summarize_only or a.count_only:
        a.GPUS = []
    tot = _local_limits(a) if permitted else 1                               # LG_RESCALE=1 에도 로컬 규칙을 적용한다(개정 1)
    if a.GPUS and permitted:                                                 # 배정 밖 번호는 자료를 읽기 전에 거부한다
        allow = [int(g) for g in str(a.gpu_allow).split(",") if g.strip()]
        bad = [g for g in a.GPUS if not g.isdigit() or int(g) not in allow]
        if bad:
            raise SystemExit(f"[거부] --gpus {bad} 는 이 실험에 배정된 GPU({allow})가 아니다")
    for v in _THREAD_VARS:
        os.environ[v] = str(a.threads)                                         # 워커(spawn)가 물려받는다
    if not a.GPUS:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.fixed_deviation:
        print(f"[warn] 고정 설계값과 다른 인자: {a.fixed_deviation}(사전 등록에서 벗어난다)", flush=True)
    if a.count_only:
        D, fr = get_D(a), frame_of(a)
        print(f"[data] {len(D.df):,}셀 · 부분 {a.part} · 대상 {len(a.TARGETS)} · 분할 {a.SPLITS} · n {a.N_GRID} · 추출 {a.DRAWS} · "
              f"스레드 {a.threads}", flush=True)
        count_only(a, D, fr)
        return dict(n_fail=0)
    reg = LC.claim_threads(a.TAG, tot, int(a.max_threads_total), script=Path(__file__).name)   # 하네스 사이의 스레드 합계(개정 1)
    try:
        return _run(a, argv, t0)
    finally:
        LC.release_threads(reg)


def _run(a, argv, t0):
    """허용된 실행의 본체(학습, 채점, 집계). main 이 스레드 등록을 잡고 끝에 푼다."""
    if a.summarize_only:
        out = summarize(a, 0.0)
        return dict(n_fail=int((out or {}).get("n_fail", 0)))
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    D = get_D(a)
    print(f"[data] {len(D.df):,}셀 · 부분 {a.part} · 대상 {a.TARGETS} · 분할 {a.SPLITS} · n {a.N_GRID} · 추출 {a.DRAWS} · 단 {a.RUNGS} · "
          f"민감도 {a.SENS} · CatBoost seed {a.CB_SEEDS} · nflow seed {a.FLOW_SEEDS} · epochs {a.epochs} · 워커 {a.workers} · "
          f"스레드 {a.threads} · GPU {a.GPUS or 'CPU'}", flush=True)
    failed, done = [], []

    def todo_of(tasks):
        out = []
        for u in tasks:
            ok, why = unit_done(a, u[0], u[1], u[2], u[3]) if a.resume else (False, "")
            if ok:
                print(f"  [resume] 건너뜀 {u} ({why})", flush=True)
            else:
                out.append(u)
        return out
    if a.part == "cpu":
        emu_tasks = todo_of([("emu", t, m, None) for t, m in a.TARGETS])
        print(f"[plan] emu 단위 {len(emu_tasks)}", flush=True)
        d_, f_ = run_tasks(a, argv, emu_tasks, False, False, t0)
        done += d_; failed += f_
        cpu_units = []
        for t, m in a.TARGETS:
            run, skipped = cpu_splits(a, D, t)
            for s_ in skipped:
                print(f"  [skip] {t}|{m}|s{s_['split']} {s_['status']}", flush=True)
            if not unit_done(a, "emu", t, m)[0]:
                failed += [dict(kind="cpu", target=t, mode=m, split=sp, error="emu 단위 없음") for sp in run]
                continue
            cpu_units += [("cpu", t, m, sp) for sp in run]
        tasks = todo_of(cpu_units)
        print(f"[plan] cpu 단위 {len(tasks)}(재개로 건너뜀 {len(cpu_units) - len(tasks)})", flush=True)
        d_, f_ = run_tasks(a, argv, tasks, False, False, t0)
        done += d_; failed += f_
    else:
        units = []
        for t, m in a.TARGETS:
            if not unit_done(a, "emu", t, m)[0]:
                failed.append(dict(kind=a.part, target=t, mode=m, split=-1, error="emu 단위 없음(--part cpu 를 먼저 실행한다)"))
                continue
            units.append((a.part, t, m, None))
        tasks = todo_of(units)
        if tasks and a.GPUS:
            _check_gpus(a)
        print(f"[plan] {a.part} 단위 {len(tasks)} · 장치 {'GPU ' + ','.join(a.GPUS) if a.GPUS else 'CPU'}", flush=True)
        d_, f_ = run_tasks(a, argv, tasks, bool(a.GPUS), True, t0)
        done += d_; failed += f_
    n_fail = len(failed) + sum(1 for r in done if r.get("status") == "failed")
    n_part = sum(1 for r in done if r.get("status") == "partial")
    print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 일부 실패 {n_part} · {time.time() - t0:.0f}s", flush=True)
    for f in failed:
        print(f"  [failed] {f}", flush=True)
    if not a.no_summarize:
        if a.precheck and not a.precision_only:
            a.precision_only = True
        out = summarize(a, time.time() - t0)
        n_fail += int((out or {}).get("n_fail", 0))
    return dict(n_fail=n_fail, n_partial=n_part, executed=[(r["kind"], r["target"], r["mode"], r["split"]) for r in done])


if __name__ == "__main__":
    res = main()
    sys.exit(1 if res.get("n_fail") else 0)
