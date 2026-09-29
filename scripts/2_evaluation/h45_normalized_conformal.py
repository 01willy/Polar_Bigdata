"""H45 · 정규화기 비교형 계층 conformal(LGU 실험 B). 계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 4절(사전 등록)의 구현.

목적
  라벨이 없는 새 지역에서 물리 앵커 ŷ = E0·√TDD 의 대칭 로그 구간을 계층 conformal 로 보정한다. 점수는 셀별 로그 척도 σ_i 로 나눈
  s_i = |log y_i − log ŷ_i| / σ_i 이다. σ_i 를 주는 정규화기가 폭을 셀 사이에서 섞은 위약 정규화기보다 구간 점수가 낮은지(LGU-B2)를
  판정한다. 정규화기 사이의 짝 비교(LGU-B3), 상수 정규화기 대비(LGU-B4), 셀 교환성 보정의 한계(LGU-B1), 층별 진단(LGU-B5),
  라벨 n개 조건의 보조 대비(LGU-B6)를 함께 낸다. h37(C2)의 hier2_cdf 는 σ_i = 1 이고 앵커가 E0 인 경우다(참조 행 const@h37).

동결 규칙
  h37 은 모듈 최상위에서 인자를 읽고 자료를 적재하므로 import 하지 않는다. 같은 식(지역 등가중 합동 CDF 의 plug-in 분위)을
  polar.lgu_common.hier_quantiles 로 쓴다. h40 은 importlib 로 파일 경로에서 읽어 자료 적재(Data), 원천(source_idx, 100 km 버퍼),
  분할 구조(split_structure), 라벨 추출(draw_cells, cells_of)에 쓴다. h40, h37, polar 의 기존 모듈, lgu_common 은 고치지 않는다.

정규화기와 방법(저장 키 = (method, variant, n, draw, seed). 결정적 방법의 seed 는 −1)
  const        σ_i = 1. 보정 점수의 앵커는 E0^(−k)(집단 지역 k 를 뺀 원천의 최소제곱 계수, 표본 밖).
  const@h37    σ_i = 1, 보정 점수의 앵커를 E0(원천 전체, 표본 안)로 둔 h37 규약. 참조 행이다.
  nflow, cfm   생성 학습기(lgu_common.fit_gen: 블록 단위 검증, 행 가중 min(1, 100/블록 셀 수), nflow 절단 ±2)의 로그 비 분위로 만든
               σ_i = clip((q_0.95 − q_0.05)/2, 0.02, 2.0). nflow 는 표본 없는 분위, cfm 은 셀당 표본 512개의 경험 분위다.
  cbq          CatBoost 다분위(0.05, 0.10, 0.50, 0.90, 0.95. 반복 200, 깊이 3, 학습률 0.05, l2 3)의 같은 폭.
  phys         물리 사전 예측 로그 폭. Sr ~ U(0.4, 1.0), n_t ~ U(0.6, 1.0) 256회(셀 공통 난수)의 log ALT 5 %·95 % 분위 폭의 절반.
  <nm>#p<j>    위약(nflow, cbq). σ 를 보정 셀은 집단 지역 안에서, 대상 셀은 채점 셀 안에서 섞는다. 순열의 seed 는
               seed_of('lgu-placebo', 대상, 정규화기, seed, j, 지역 이름 또는 'target') 이다.
  <nm>@mw      평균 로그 폭 정합. h_i(c) = q_const(c)·σ_i / mean(σ, 채점 셀). 평균 폭을 const 와 같게 두고 셀 사이 배분만 비교한다.
  <m>@noak     알래스카를 보정 집단에서 뺀 분위(const, nflow, cbq. 대상이 알래스카가 아니고 알래스카가 집단일 때).
  ~noearly     저장소 '<대상>~noearly'. early 셀을 보정 셀과 채점 셀에서 뺀다(const, nflow, cbq). 정규화기는 다시 학습하지 않는다.
  cqr_pool:<L>@lam<λ>, cqr_hier:<L>@lam<λ>
               생성 학습기 L(nflow, cfm)의 분위 구간에 CQR 점수 max(q_lo − z, z − q_hi)를 쓴다. q 는 위치 전용 λ 규약(loc_only)을
               적용한 값이다. pool 은 셀 교환성 유한표본 분위, hier 는 지역 등가중 분위다. α 0.1 은 (q_0.05, q_0.95), α 0.2 는
               (q_0.10, q_0.90). 구간은 [ŷ'·exp(q_lo − Q), ŷ'·exp(q_hi + Q)], ŷ' = E0^학습·s 다.
  보정(n = 0): 집단 지역 k(원천에서 셀 20개 이상인 macro 지역)마다 k 를 뺀 원천으로 정규화기를 학습해 k 의 셀에 표본 밖 σ 를 준다.
  점수의 지역 등가중 합동 CDF 의 수준 c 분위 q(c)로 격자 Q_i(0.5 ∓ c/2) = ŷ_i·exp(∓q(c)·σ_i), c ∈ {0.02, …, 0.98} 을 만든다.
  대상 셀의 σ 는 원천 전체로 학습한 정규화기의 예측이다. 채점은 대상의 eval_mask 셀 전체(범위 all)다.
  보조 n > 0(레나, 캐나다, 알래스카 가운데 --targets 에 있는 것. const, nflow, cbq): 정규화기를 다시 학습하지 않는다.
  중심은 E_n·s(κ = 10 수축, 라벨 = h40.draw_cells, A 의 순서는 half_split_blocks 의 A_idx 순서), 보정은 의사 대상 모사
  (lgu_common.emulation_plan, 모사 분할 3 × 추출 3)의 점수 |log y − log(E_center·s)| / σ 의 지역 등가중 분위, 채점은 범위 B 다.

누설 규약
  보정 점수, 정규화기의 학습, E0 는 대상과 100 km 버퍼를 뺀 원천에서만 구한다. 대상 라벨은 보조 분석의 선택된 n개만 E_n 에 쓴다.
  채점 셀의 라벨은 채점에만 쓴다. 표준화와 결측 대체 통계는 학습 행렬에서만 구한다.

작업 단위와 조각(<out-dir>/shards)
  fit   (대상, 정규화기): lgub__fit__<대상>__x__<정규화기>_{sigma.npz, runs.csv, unit.json}. 학습 집합은 집단 수 K + 1 개다.
        const, const@h37 은 적합이 없으므로 단위가 없다. 적합 한 건의 예외는 그 (학습 집합, seed)만 실패로 기록한다.
        (학습 집합 가운데 하나라도 실패한 seed 는 그 정규화기의 유효 seed 에서 뺀다.)
  score (대상): lgub__score__<대상>__x_{runs.csv, scores.npz, cells.npz, unit.json}. 그 대상의 fit 조각을 읽는다.
        없거나 실패한 정규화기는 그 방법의 행을 빼고 unit.json 의 missing 에 적는다.
  unit.json 은 마지막에 원자적으로 쓰는 완료 표지다. --resume 은 설정 해시가 같고 status 가 failed 가 아닌 조각만 건너뛴다.
  score 의 설정에는 읽은 sigma.npz 의 해시가 들어 있어 fit 을 다시 하면 score 도 다시 한다.

신뢰구간과 판정
  2단 CI(주): 전역 블록 재표집(lgu_common.global_mult, 2,000회)의 다중도 하나를 보정 쪽과 채점 쪽이 함께 쓴다. 재표집마다 보정
  분위를 다중도 가중으로 다시 구하고(hier_quantiles_boot. cqr_pool 은 다중도 가중 셀 합동 유한표본 분위) 채점 블록의 다중도로
  셀 가중, 블록 등가중 평균을 다시 낸다. 지표는 METRICS_2STAGE 이고 재표집 200개씩 묶어 벡터화한다. 보조 n 은 추출 평균 뒤 분할의
  같은 번호 평균이다. 정규화기, E0, E0^(−k), 라벨 추출, 모사 분할은 고정한다.
  1단 CI(보조): LG 규칙(lgu_common.one_stage_delta, 분할 안 채점 블록 재표집). 모든 지표에 낸다.
  판정은 계획서 2.7절과 4.4절이다(4분 판정, 구간 점수 동등성 한계 5 %·10 %, 정밀도 규칙, 커버리지 조건 0.80, 풀 조건 채점 블록 8개).

산출(<out-dir>. 접두어는 lgu_b, 스모크 lgu_b_smoke, 사전 점검 lgu_b_precheck)
  <접두어>_intervals.csv  대상·범위·방법·변형·n 별 지표의 두 가중 평균, 2단·1단 CI, MEAN4·MEAN3·MEAN_AUX3 행
  <접두어>_tests.csv      LGU-B1–B6 의 대비와 판정
  <접두어>_cal.csv        집단별 정규화 점수 분포, 분위 초과 비율, σ 분포, 절단 비율
  <접두어>_strata.csv     기기(gpr, probe)·시기(early)·σ 5분위·최근접 원천 셀 거리 3분위 층의 커버리지와 폭
  <접두어>_pit.csv        PIT 10구간 개수(서술), <접두어>_cells.csv 채점 셀의 σ 와 90 % 구간 끝점(폭 지도의 입력)
  <접두어>_precision.csv  --precision-only(사전 점검): 대비별 2단 CI 반폭의 기준 점수 대비 비율과 정밀도 상태만(부호, 점 추정, 기준 점수는
                          쓰지 않는다)
  <접두어>_timing.csv, _failed.csv, _meta.json, _count.csv(--count-only), _boot__<대상>.npz(대상별 2단 분포, float32)
  lgu_blockindex.csv      전역 블록 색인(LC.check_blockindex. h44 의 집계도 같은 파일을 쓰거나 대조한다. 내용이 다르면 중단한다)

실행 환경(사용자 지시 2026-09-30)
  로컬 GPU 서버에서 실행한다(Rescale 을 쓰지 않는다). 허용 표지(--allow-local 또는 LG_RESCALE=1)가 없으면 --count-only 만 실행하고
  스레드는 1 이다. 허용된 실행은(LG_RESCALE=1 이어도 같다, 개정 1) 프로세스 우선순위를 nice 10 으로 낮추고, 동시에 도는 프로세스 수 ×
  --threads 가 8 을 넘으면 거부한다(CPU 스레드 합계 8 이하 규칙. BLAS, torch, CatBoost 가 모두 --threads 를 쓴다). 이 실험에 배정된
  GPU 는 2번(필요하면 1번)이고 --gpu-allow(기본 2,1) 밖의 번호는 거부한다. GPU 풀을 만들기 직전에 nvidia-smi 로 메모리 사용이 0 MiB 인
  번호만 남긴다(남는 번호가 없으면 중단한다. CUDA_DEVICE_ORDER=PCI_BUS_ID 로 번호 체계를 맞춘다). 적합 부분은 CPU 풀(cbq, phys)을 먼저
  돌리고 GPU 풀(nflow, cfm)을 그 뒤에 돌린다(두 풀은 동시에 돌지 않는다). h44·h45·h46 사이의 스레드 합계는 실행 중 등록부
  (data/processed/lgu/.running, LC.claim_threads)로 확인한다. 앞 실행과 합해 8 을 넘으면 거부한다(권고: 부분을 차례로 실행한다).

실행 예(ROOT, 로컬)
  적합 수:   python3 scripts/2_evaluation/h45_normalized_conformal.py --count-only
  스모크:    nice -n 10 python3 scripts/2_evaluation/h45_normalized_conformal.py --smoke --part all --allow-local --gpus 2 --workers 1 --threads 4
  사전 점검: nice -n 10 python3 scripts/2_evaluation/h45_normalized_conformal.py --precheck --part all --allow-local --gpus 2 --workers 1 --threads 4
  본 실행:   nice -n 10 python3 scripts/2_evaluation/h45_normalized_conformal.py --part fit --allow-local --gpus 2 --workers 1 --threads 4 --resume
             nice -n 10 python3 scripts/2_evaluation/h45_normalized_conformal.py --part score --allow-local --workers 2 --threads 4 --resume
             (score 뒤에 집계까지 한다. --no-summarize 로 생략할 수 있다)
  집계만:    nice -n 10 python3 scripts/2_evaluation/h45_normalized_conformal.py --summarize-only --allow-local --workers 2 --threads 4

명세(spec_B)와 다르게 구현한 점
  1. --part 에 'all'(fit 뒤 score)을 더했다. 기본은 명세대로 fit 이다. fit 부분만 실행하면 집계하지 않는다.
  2. 인자 --n-grid 를 --n-grid-aux 의 별칭으로 두었다(구현 규칙의 인자 목록). --seeds 는 seed 개수다(0 부터).
  3. 시험 파일 이름은 과제 지시에 따라 tests/test_h45_normconf.py 다(spec_B 는 tests/test_h45_conformal.py).
  4. 채점 셀과 보정 셀은 y, s 가 유한하고 양수인 셀로 제한한다(log 가 정의되는 셀). 뺀 채점 셀 수를 unit.json 의 n_drop_eval 에 적는다.
     보조 분석의 모사 항목 가운데 보정 셀 집합에 없는 셀(y ≤ 0 등)은 빼고 n_drop_aux 에 적는다.
  5. 정규화기의 학습 목표 z 의 앵커 E0^학습 은 학습 행(z 유한)의 최소제곱 계수다. 보정 점수의 앵커 E0^(−k) 는 lgu_common.emulation_plan 과
     같은 정의(원천에서 k 를 뺀 셀, y·s 유한)다. y > 0 이면 둘은 같다. CQR 점수는 z 를 E0^학습 기준으로 되돌려 계산한다.
  6. ~noearly 는 early 셀을 채점 셀뿐 아니라 보정 셀에서도 뺀다(h44 의 _noearly 블록 표와 같은 방향). 정규화기는 다시 학습하지 않는다.
  7. σ 5분위 층(#w1–#w5)은 const 에서 σ 가 상수이므로 ŷ 의 5분위로 둔다(cm 폭이 ŷ 에 비례한다). 분위는 순위 기준 등분이다.
  8. CQR 구간의 하한이 상한을 넘으면(보정 분위가 음수) 두 끝점을 중심 ŷ'·exp(λ·q_0.5)로 둔다(h44 의 iii+c 와 같은 처리).
  9. phys 의 sigma.npz 의 분위(cal_q, tgt_q5)는 log ALT 표본 분위에서 중앙값을 뺀 값이다(z 척도가 없다). tgt_qgrid 는 행이 0개다.
     tgt_q5(대상 셀의 5점 분위)를 명세의 배열에 더했다(CQR 에 쓴다).
  10. LGU-B1 의 행은 학습기마다 한 행이고 delta 열에 대비가 아니라 수준(주 4지역 평균 cov10)을 적는다. 판정은 두 행에 같은 값이다.
  11. LGU-B2 의 조건 (4)(풀 지역마다 유효 seed 2개 이상)가 어긋나면 계획서 2.10절에 따라 '판정 불가(적합 실패)'로 적는다.
      러시아 W 를 뺀 3지역 평균의 판정에서는 조건 (2)의 기준을 3지역 가운데 2지역 이상으로 둔다(4지역 기준 '하나만 예외'의 대응).
      커버리지 조건 (3)은 셀 가중 평균으로 판정하고 블록 등가중 값을 함께 적는다.
  12. tests.csv 에 margin_beq(블록 등가중 한계) 열을 더했다. 동등성 한계와 정밀도 규칙은 가중별 한계를 쓴다(lgu_common.eq_state).
  13. intervals.csv 의 se 지표는 rmse 로 이름을 바꿔 적는다(값은 RMSE). 위약은 순열·seed 를 합친 한 행('<nm>#placebo')이다.
  14. --allow-local 실행에 nice 10 적용, 스레드 합계 8 초과 거부, GPU 0 MiB 확인을 더했다(로컬 자원 규칙의 강제).
  15. 명세의 phys 학습 행 결측 대체는 lgu_common.phys_log_samples 의 규칙(학습 행 중앙값)이다. seed 는 0 으로 고정했다.
  16. 적합 실패의 처리: 명세는 실패한 (학습 집합, seed)만 NaN 으로 두라고 했다. 한 학습 집합의 σ 가 없으면 그 seed 의 보정이 성립하지 않으므로
      한 학습 집합이라도 실패한 seed 는 그 정규화기의 모든 학습 집합에서 NaN 으로 두고 valid_seeds 에서 뺀다(runs.csv 에는 건별로 적는다).
  17. score 는 그 대상의 fit 단위가 모두 끝나야(unit.json 이 있고 설정이 같아야) 실행한다. status 'failed' 로 끝난 fit 단위는 끝난 것으로
      보고 그 정규화기를 missing 에 적는다. 설정이 다르거나 조각이 없는 fit 단위가 있으면 그 대상의 채점을 하지 않고 실패로 센다.
  18. 명세에 없는 산출을 더했다: <접두어>_boot__<대상>.npz(대상별 2단 분포), intervals.csv 의 MEAN_AUX3 행(보조 n 의 풀),
      tests.csv 의 1단 CI 열(ci_*_1stage)과 cov_beq_pool, runs.csv 의 n_eval, n_blocks, rmse, pit_h0–pit_h9 열.
  19. 집계의 2단 재표집은 --workers > 1 이면 대상 단위 프로세스 풀로 나눈다. 인자 --pool-retries 를 더했다(h40 과 같은 풀 재시도).

개정 1(2026-09-30, 검증 지적 반영. 계획서 개정 이력 참조)
  a. 1단 재표집의 seed 를 대상·범위마다 seed_of('lgu-b', 대상, 범위)로 둔다(앞 판은 모든 대상이 0 이어서 블록 수가 비슷한 지역의 재표집이
     거의 같았다). 같은 대상·범위의 수준과 대비는 같은 seed 를 쓴다.
  b. 스크립트 실행(__main__)에서는 스레드 환경 변수를 --threads 로 강제하고, spawn 워커와 시험의 import 에서만 기존 값을 존중한다.
     CUDA_DEVICE_ORDER=PCI_BUS_ID 를 둔다. --gpu-allow 밖의 번호는 거부한다. LG_RESCALE=1 도 로컬 자원 규칙을 끄지 않는다.
  c. fit 설정 해시에서 --threads 를 뺐다(실행 자원. unit.json 의 threads 에만 적는다).
  d. 보정 집단 G 는 LC.cal_groups(로그 비가 정의되는 원천 셀 20개 이상)이다. h44 와 같은 정의다.
  e. 보조 분석의 의사 대상 중심 E_center 를 모사 기록의 라벨로 --kappa 에 맞춰 다시 계산한다(앞 판은 LC.KAPPA 고정).
  f. 적합의 환경·자원 오류(LC.is_env_error: CUDA 메모리 부족 포함)는 적합 실패가 아니라 조각 실패로 둔다. 워커의 SystemExit 도 단위 실패로
     기록한다. 2단 재표집은 대상별로 예외를 잡아 실패 목록에 적고 나머지 대상으로 표를 만든다(풀이 깨지면 남은 대상을 차례로 계산한다).
  g. 정밀도 표에서 기준 점수와 반폭의 절댓값을 뺐다(반폭 비율과 정밀도 상태만). LGU-B2 는 1단 짝지음 점검(LC.one_stage_delta 의
     pair_ok)이 실패한 풀 지역이 있으면 '판정 불가(계산 실패)'다.

확인하지 못한 것
  개정 1 단계에서 실행한 확인은 py_compile, pyflakes, 스레드 1개의 --count-only, 가벼운 단위 시험(합성 자료, CPU 스레드 2)이다.
  GPU 2·1번이 사용 중이어서 스모크, 사전 점검, GPU 실행은 하지 않았다(실제 자료의 fit·score 단위와 집계 경로는 실행되지 않았다). 2단 CI 의 계산 시간(특히 알래스카 대상과 보조 분석), cfm 표본 512개의 GPU 메모리, CatBoost 1.2.10 의
  MultiQuantile 손실 동작, phys_log_samples 가 실제 셀에서 physics.load_physics_inputs 의 bdod 범위 확인을 통과하는지, 정규화 점수
  분위의 수치 안정성은 확인하지 않았다.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_threads(default="4"):
    """numpy 를 부르기 전에 스레드 수를 정한다(h42 와 같은 규칙). 허용 표지(LG_RESCALE=1 또는 --allow-local)가 없으면 '1'."""
    av = sys.argv
    if not (os.environ.get("LG_RESCALE", "") == "1" or "--allow-local" in av):
        return "1"
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return av[i + 1]
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return default


if __name__ == "__main__":                                  # 스크립트: numpy 적재 전에 스레드 수를 강제한다(셸 값보다 --threads 가 우선)
    for _v in _THREAD_VARS:
        os.environ[_v] = _peek_threads()
else:                                                       # spawn 워커(__mp_main__)와 시험의 import: 부모가 둔 값을 존중한다
    for _v in _THREAD_VARS:
        os.environ.setdefault(_v, _peek_threads())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"              # CUDA 장치 번호를 nvidia-smi 번호(PCI 버스 순)와 맞춘다(h43 과 같다)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
H40_PATH = ROOT / "scripts" / "3_deep_learning" / "h40_label_grid.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_h40():
    """h40 을 파일 경로에서 읽어 sys.modules 에 등록한다(모듈 최상위에서 부른다. spawn 워커가 주 모듈을 다시 읽을 때도 등록된다).
    이미 등록되어 있으면 그 모듈을 쓴다(h42 와 같은 방식)."""
    name = "h40_label_grid"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, H40_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load_h40()

import polar.lgu_common as LC  # noqa: E402
import polar.h4_common as H4  # noqa: E402
from polar.h4_common import seed_of  # noqa: E402

# ================================================================ 고정 설계값(계획서 4절. 바꾸면 사전 등록에서 벗어난다)
MAIN4 = list(H.MAIN4)                                   # Lena, Canada, Russia_W, Russia_E
ALASKA = H.ALASKA
RUSSIA_W = "Russia_W"
DEFAULT_TARGETS = MAIN4 + [ALASKA]
POOL3 = [t for t in MAIN4 if t != RUSSIA_W]
POOL_AUX = ["Lena", "Canada", ALASKA]
POOL_AUX2 = ["Lena", "Canada"]
FIT_NORMS = ("nflow", "cfm", "cbq", "phys")             # 적합 단위가 있는 정규화기
GEN_NORMS = ("nflow", "cfm")                            # GPU 학습기
SEEDED = ("nflow", "cfm", "cbq")                        # seed 축이 있는 정규화기(phys 는 seed 축 길이 1, 키 seed −1)
PLACEBO_NORMS = ("nflow", "cbq")
CQR_LEARNERS = ("nflow", "cfm")
AUX_NORMS = ("const", "nflow", "cbq")
STRATA_NORMS = ("const", "nflow", "cbq")
SENS_NORMS = ("const", "nflow", "cbq")                  # @noak, ~noearly
B4_NORMS = ("nflow", "cbq", "cfm", "phys")
ALPHAS2 = (0.10, 0.20)
Q5_ROWS = [LC.grid_index(t) for t in LC.Q5]            # 99점 격자에서 5점 분위의 행
LV90 = int(np.argmin(np.abs(LC.LEVELS - 0.90)))
LV80 = int(np.argmin(np.abs(LC.LEVELS - 0.80)))
BOOT_LEVELS = (0.90, 0.80)                              # 2단 재표집의 보정 수준(열 0 = α 0.1, 열 1 = α 0.2)
BOOT_CHUNK = 200                                        # 재표집 묶음 크기
ODE_MAX_CELLS = 1000                                    # cfm 결정적 분위 사상의 교차 점검 셀 수 상한
CB_CFG = dict(iterations=200, depth=3, lr=0.05, l2=3.0)
FEATS = list(H.FEATS)                                   # x25
INPUT_TABLES = ("fidelity_base_v3.csv", "e5_soil_tdd_v3.csv")
LOCAL_THREAD_CAP = 8                                    # 로컬 규칙: 이 실험의 CPU 스레드 합계
LOCAL_GPUS = ("2", "1", "5")                                 # 로컬 규칙: 이 실험에 배정된 GPU(--gpu-allow 기본값. 밖의 번호는 거부한다)
NICE_LOCAL = 10
MAX_TXT = 300
STRATA_METRICS = ("cov10", "wid10", "lhw10", "is10", "cov20", "crps_log")
SCORE_UNIT_KEYS = ("aux", "fit_sha")                    # 대상마다 달라도 되는 설정 항목(공통 해시에서 뺀다)


def lam_label(x) -> str:
    """λ 의 키 표기(repr(float)): 0 → '0.0', 0.25 → '0.25', 1 → '1.0'. 명세의 'cqr_pool:nflow@lam1.0' 과 같은 표기다."""
    return repr(float(x))


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H45 정규화기 비교형 계층 conformal(LGU 실험 B)")
    ap.add_argument("--part", choices=["fit", "score", "all"], default="fit",
                    help="fit = 정규화기 적합(σ), score = 채점(fit 조각 필요, 뒤이어 집계), all = fit 뒤 score")
    ap.add_argument("--normalizers", default="nflow,cfm,cbq,phys", help="적합하는 정규화기(const 는 적합이 없어 항상 포함)")
    ap.add_argument("--targets", default="", help="쉼표 목록 '이름' 또는 '이름:x'. 기본 Lena, Canada, Russia_W, Russia_E, Alaska")
    ap.add_argument("--seeds", type=int, default=3, help="seed 개수(0 부터). nflow, cfm, cbq 에 쓴다")
    ap.add_argument("--perms", type=int, default=10, help="위약 순열 수")
    ap.add_argument("--lams", default="0,0.25,1.0", help="CQR 의 위치 전용 λ 목록")
    ap.add_argument("--splits", type=int, default=5, help="보조 분석의 분할 1..K")
    ap.add_argument("--n-grid-aux", "--n-grid", dest="n_grid_aux", default="10,40,160", help="보조 분석의 라벨 수")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--aux-targets", default="Lena,Canada,Alaska")
    ap.add_argument("--s-gen", type=int, default=LC.S_GEN, help="cfm 의 셀당 표본 수")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2, help="CPU 프로세스 수. 0 = 풀 없이 이 프로세스에서 차례로")
    ap.add_argument("--threads", type=int, default=4, help="프로세스당 스레드(BLAS, torch, CatBoost). 허용 표지가 없으면 1")
    ap.add_argument("--gpus", default="", help="GPU 물리 번호 쉼표 목록(nflow, cfm). 기본 빈 문자열(CPU)")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    ap.add_argument("--gpu-allow", default=",".join(LOCAL_GPUS), help="쓸 수 있는 GPU 물리 번호(사용자 지시 2026-09-30). 밖의 번호는 거부한다")
    ap.add_argument("--nboot", type=int, default=LC.NBOOT)
    ap.add_argument("--out-dir", default="data/processed/lgu")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default="lgub")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--label-flags", default="lgx_label_flags_v1.csv")
    ap.add_argument("--allow-local", action="store_true", help="로컬 GPU 서버 실행 허용(nice 10, 스레드 합계 8 이하, GPU 허용 목록과 0 MiB 확인)")
    ap.add_argument("--smoke", action="store_true", help="Canada x, Russia_W x, epochs 3, seed 1, 순열 2, nboot 200, 보조 n = Canada {10}")
    ap.add_argument("--precheck", action="store_true", help="Canada x, epochs 100, seed 1, 순열 10, nboot 2000. 집계는 --precision-only")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--precision-only", action="store_true", help="집계에서 대비의 2단 CI 반폭만 쓴다(부호와 점 추정 없음)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pool-retries", type=int, default=1, help="워커 비정상 종료로 풀이 깨졌을 때 남은 단위로 풀을 다시 만드는 횟수")
    # 아래는 계획서 2절·4절의 고정 설계값이다. 바꾸면 사전 등록에서 벗어난다.
    ap.add_argument("--kappa", type=float, default=LC.KAPPA)
    ap.add_argument("--buffer-km", type=float, default=100.0)
    ap.add_argument("--min-cal-cells", type=int, default=LC.MIN_CAL_CELLS)
    ap.add_argument("--cap-cells", type=int, default=LC.CAP_CELLS)
    ap.add_argument("--clamp", type=float, default=LC.NFLOW_CLAMP)
    ap.add_argument("--sigma-lo", type=float, default=LC.SIGMA_CLIP[0])
    ap.add_argument("--sigma-hi", type=float, default=LC.SIGMA_CLIP[1])
    ap.add_argument("--phys-draws", type=int, default=256)
    return finalize(ap.parse_args(argv))


def _targets(txt):
    out = []
    for tok in [v.strip() for v in str(txt).split(",") if v.strip()]:
        t, _, m = tok.partition(":")
        if m and m != "x":
            raise SystemExit(f"[args] 실험 B 의 대상은 모드 x 만 쓴다: {tok}")
        out.append(t)
    return out


def finalize(a):
    if a.smoke and a.precheck:
        raise SystemExit("--smoke 와 --precheck 는 함께 쓰지 않는다")
    a.NORMS = [v.strip() for v in a.normalizers.split(",") if v.strip()]
    bad = [v for v in a.NORMS if v not in FIT_NORMS]
    if bad:
        raise SystemExit(f"[args] 알 수 없는 정규화기 {bad}(가능: {', '.join(FIT_NORMS)})")
    a.TARGETS = _targets(a.targets) if a.targets else list(DEFAULT_TARGETS)
    a.SEEDS = list(range(max(1, int(a.seeds))))
    a.LAMS = [float(v) for v in a.lams.split(",") if v.strip()]
    a.SPLITS = list(range(1, int(a.splits) + 1))
    a.N_AUX = [int(v) for v in a.n_grid_aux.split(",") if v.strip() and v.strip() != "all" and int(v) > 0]
    a.DRAWS = int(a.draws)
    a.PERMS = int(a.perms)
    a.AUX_TARGETS = _targets(a.aux_targets) if a.aux_targets else []
    if a.smoke:
        if not a.targets:
            a.TARGETS = ["Canada", RUSSIA_W]
        a.epochs = min(int(a.epochs), 3); a.SEEDS = [0]; a.PERMS = min(a.PERMS, 2); a.nboot = min(int(a.nboot), 200)
        a.AUX_TARGETS = ["Canada"]; a.N_AUX = [10]; a.SPLITS = [1]; a.DRAWS = 1
    if a.precheck:
        if not a.targets:
            a.TARGETS = ["Canada"]
        a.SEEDS = [0]; a.PERMS = 10; a.precision_only = True
    a.AUX_EFF = [t for t in a.AUX_TARGETS if t in a.TARGETS]
    a.TAG = a.tag + ("_smoke" if a.smoke else "") + ("_precheck" if a.precheck else "")
    a.PREFIX = ("lgu_b" + a.TAG[len("lgub"):]) if a.TAG.startswith("lgub") else a.TAG
    a.GPUS = [g.strip() for g in str(a.gpus).split(",") if g.strip() != ""]
    a.OUT = (ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir)
    a.PROC = (ROOT / a.data_dir) if not os.path.isabs(a.data_dir) else Path(a.data_dir)
    a.SHARDS = a.OUT / "shards"
    a.FLAGS = Path(a.label_flags) if os.path.isabs(a.label_flags) else a.PROC / a.label_flags
    a.SUBMAP = Path(a.subregion_map) if os.path.isabs(a.subregion_map) else a.PROC / a.subregion_map
    return a


def seeds_of(a, nm):
    """정규화기의 seed 목록. phys 는 [0](키 seed −1)."""
    return list(a.SEEDS) if nm in SEEDED else [0]


def key_seed(nm, seed):
    """저장 키의 seed. seed 축이 없는 방법(const, const@h37, phys)은 −1."""
    return int(seed) if nm in SEEDED else -1


def _js(o):
    """JSON 직렬화 기본값(numpy 스칼라·배열, 경로)."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=_js)


# ================================================================ 자료, 색인, 해시
def h40_args(a):
    """h40 의 인자 객체(자료 적재와 분할 구조에만 쓴다)."""
    grid = ",".join(str(n) for n in a.N_AUX) if a.N_AUX else "0"
    argv = ["--part", "cpu", "--out-dir", str(a.OUT), "--data-dir", str(a.PROC), "--subregion-map", str(a.subregion_map),
            "--modes", "x", "--splits", str(max(a.SPLITS) if a.SPLITS else 1), "--n-grid", grid, "--draws", str(max(1, a.DRAWS)),
            "--seeds", "1", "--threads", str(a.threads), "--kappa", repr(float(a.kappa)), "--buffer-km", repr(float(a.buffer_km))]
    return H.parse_args(argv)


def get_D(a):
    """h40.Data(프로세스마다 한 번 적재. h40.get_data 의 모듈 캐시)."""
    return H.get_data(h40_args(a))


def get_bi(D) -> "LC.BlockIndex":
    bi = getattr(D, "_lgu_bi", None)
    if bi is None:
        bi = LC.BlockIndex(D.df)
        D._lgu_bi = bi
    return bi


def get_flags(a, D) -> dict:
    """셀 표지(early, gpr, probe)를 자료의 행 순서로 읽는다. 표에 없는 loc_id 가 있으면 lgu_common.load_flags 가 중단한다."""
    f = getattr(D, "_lgu_flags", None)
    if f is None:
        f = LC.load_flags(a.FLAGS, D.df.loc_id.values)
        D._lgu_flags = f
    return f


_SHA: dict = {}
_INPUT_SHA: dict = {}


def code_shas() -> dict:
    if not _SHA:
        _SHA.update(code_sha=LC.file_sha(Path(__file__)), code_sha_common=LC.file_sha(Path(LC.__file__)), code_sha_h40=LC.file_sha(H40_PATH))
    return dict(_SHA)


def input_shas(a) -> dict:
    """입력 표의 해시. 캐시는 _INPUT_SHA 에 따로 둔다(개정 2: 앞 판은 _SHA 에 튜플 키로 넣어 code_shas() 의 dict 에 문자열이 아닌 키가
    섞였고, `**code_shas()` 가 'keywords must be strings' 로 실패해 fit 단위와 집계가 모두 실패했다. heavy 시험 m 에서 확인)."""
    k = (str(a.PROC), str(a.FLAGS), str(a.SUBMAP))
    if k not in _INPUT_SHA:
        d = {n: LC.file_sha(a.PROC / n) for n in INPUT_TABLES}
        d[Path(a.SUBMAP).name] = LC.file_sha(a.SUBMAP)
        d[Path(a.FLAGS).name] = LC.file_sha(a.FLAGS)
        _INPUT_SHA[k] = d
    return dict(_INPUT_SHA[k])


def check_blockindex(a, bi):
    """lgu_blockindex.csv 를 쓰거나(없을 때) 대조한다(있을 때). 내용이 다르면 자료 판이 다르므로 중단한다(LC.check_blockindex, h44 도 같은 파일)."""
    return LC.check_blockindex(a.OUT / "lgu_blockindex.csv", bi)


def target_info(a, D, t) -> dict:
    """대상의 원천, 보정 집단, 보정 셀, 채점 셀(프로세스 안 캐시).

    원천 = h40.Data.source_idx(대상, 'x')(대상 셀 제외, 경계 100 km 버퍼 제외). E0 = 원천의 최소제곱 계수.
    집단 G = 원천에서 로그 비가 정의되는 셀(y, s 유한이고 양수)이 20개 이상인 macro 지역(사전순, LC.cal_groups. h44 와 같다).
    E0_mk[k] = 원천에서 k 를 뺀 셀의 최소제곱 계수(lgu_common.emulation_plan 과 같다).
    보정 셀 = G 의 원천 셀 가운데 y, s 가 유한하고 양수인 셀(G 순서로 이어 붙인다). cal_u = log y − log(E0_mk·s).
    채점 셀 = 대상의 eval_mask 셀 가운데 y, s 가 유한하고 양수인 셀.
    """
    cache = D.__dict__.setdefault("_lgu_tinfo", {})
    ck = (t, int(a.min_cal_cells))
    if ck in cache:
        return cache[ck]
    if t not in D.macros:
        raise SystemExit(f"[target] {t} 는 macro 지역이 아니다(실험 B 의 대상은 macro 지역, 모드 x)")
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx(t, "x")
    y = df.y.values.astype(float); s = df.s.values.astype(float)
    mac = np.asarray(df.macro.values).astype(str)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(y) & np.isfinite(s) & (s > 0) & (y > 0)
    E0 = LC.ls_E(y[src_idx], s[src_idx])
    G = LC.cal_groups(mac, src_idx, ok, a.min_cal_cells)                  # 로그 비가 정의되는 원천 셀 20개 이상(h44 와 같은 정의, 개정 1)
    E0_mk = np.array([LC.ls_E(y[src_idx[mac[src_idx] != k]], s[src_idx[mac[src_idx] != k]]) for k in G], float)
    parts = [src_idx[(mac[src_idx] == k) & ok[src_idx]] for k in G]
    cal_idx = np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)
    cal_gi = np.concatenate([np.full(len(p_), i, np.int64) for i, p_ in enumerate(parts)]) if parts else np.zeros(0, np.int64)
    cal_region = np.array(G, dtype=str)[cal_gi] if len(G) else np.zeros(0, dtype=str)
    with np.errstate(invalid="ignore", divide="ignore"):
        cal_u = np.log(y[cal_idx]) - np.log(E0_mk[cal_gi] * s[cal_idx]) if len(cal_idx) else np.zeros(0)
    ev_raw = t_idx[LC.eval_mask(df.iloc[t_idx])]
    ev = ev_raw[ok[ev_raw]]
    info = dict(t=t, t_idx=np.asarray(t_idx, np.int64), src_idx=np.asarray(src_idx, np.int64), comp=comp, E0=float(E0), G=G, E0_mk=E0_mk,
                cal_idx=cal_idx, cal_gi=cal_gi, cal_region=cal_region, cal_u=cal_u, ev=np.asarray(ev, np.int64),
                n_drop_eval=int(len(ev_raw) - len(ev)), ok=ok)
    cache[ck] = info
    return info


def train_sets(D, info) -> list:
    """정규화기의 학습 집합 [(이름, 학습 행, 예측 행, 보정 배열의 위치 또는 None)]. 집단 k 마다 'k 를 뺀 원천'(예측 = k 의 보정 셀)과
    'full'(원천 전체, 예측 = 대상의 모든 셀). 학습 행은 z = log y − log(E0^학습·s) 가 유한한 원천 셀이다."""
    mac = np.asarray(D.df.macro.values).astype(str)
    ok = info["ok"]; src = info["src_idx"]
    out = []
    for gi, k in enumerate(info["G"]):
        tr = src[(mac[src] != k) & ok[src]]
        cpos = np.where(info["cal_gi"] == gi)[0]
        out.append((k, tr, info["cal_idx"][cpos], cpos))
    out.append(("full", src[ok[src]], info["t_idx"], None))
    return out


def row_w(blocks, cap):
    """행 가중 min(1, cap / 블록 셀 수). cap 이 설계값(100)이면 lgu_common.row_weights 와 같다."""
    if int(cap) == LC.CAP_CELLS:
        return LC.row_weights(blocks)
    _, inv, cnt = np.unique(np.asarray(blocks), return_inverse=True, return_counts=True)
    return np.minimum(1.0, float(cap) / cnt[inv.ravel()].astype(float))


def sigma_from_q5(Q5, lo, hi):
    """σ = clip((q_0.95 − q_0.05)/2, lo, hi) 와 절단에 걸린 셀 비율."""
    raw = (np.asarray(Q5[4], float) - np.asarray(Q5[0], float)) / 2.0
    with np.errstate(invalid="ignore"):
        clip = float(np.mean((raw < lo) | (raw > hi))) if raw.size else np.nan
    return np.clip(raw, lo, hi), clip


# ================================================================ 로컬 자원 규칙(사용자 지시 2026-09-30)
def local_guard(a, parts, will_summarize):
    """허용된 실행의 자원 규칙: nice 10, 동시 스레드 합계 8 이하, GPU 허용 목록(--gpu-allow) 밖의 번호 거부. LG_RESCALE=1 로 허용된 실행에도
    적용한다(개정 1). 반환: 이 실행이 동시에 쓰는 스레드 수(실행 중 등록부에 적는 값). 허용 표지가 없으면 0."""
    if not LC.run_permitted(a.allow_local):
        return 0
    try:
        cur = os.nice(0)
        if cur < NICE_LOCAL:
            os.nice(NICE_LOCAL - cur)
    except OSError:
        pass
    th = max(1, int(a.threads))
    nproc = max(int(a.workers), 1)
    need = [th]
    if "fit" in parts:
        need.append(nproc * th)
        if a.GPUS:
            need.append(len(a.GPUS) * max(1, int(a.procs_per_gpu)) * th)
    if "score" in parts or will_summarize:
        need.append(nproc * th)
    tot = max(need)
    if tot > LOCAL_THREAD_CAP:
        raise SystemExit(f"[거부] 동시에 도는 스레드 합계 {tot} 가 로컬 상한 {LOCAL_THREAD_CAP} 를 넘는다(프로세스 수 × --threads). "
                         f"--workers, --procs-per-gpu, --threads 를 줄인다(예: --workers 2 --threads 4, 또는 --gpus 2 --workers 1 --threads 4)")
    allow = [g.strip() for g in str(a.gpu_allow).split(",") if g.strip()]
    far = [g for g in a.GPUS if g not in allow]
    if far:
        raise SystemExit(f"[거부] GPU {far} 는 이 실험에 배정된 번호({', '.join(allow)})가 아니다(사용자 지시 2026-09-30)")
    return int(tot)


def usable_gpus(a) -> list:
    """GPU 풀을 만들기 직전의 확인. 허용 목록 밖의 번호를 거부하고 메모리 사용 0 MiB 인 번호만 남긴다(없으면 중단). LG_RESCALE=1 에도 적용한다."""
    if not a.GPUS:
        return []
    allow = [g.strip() for g in str(a.gpu_allow).split(",") if g.strip()]
    far = [g for g in a.GPUS if g not in allow]
    if far:
        raise SystemExit(f"[거부] GPU {far} 는 이 실험에 배정된 번호({', '.join(allow)})가 아니다(사용자 지시 2026-09-30)")
    ids = [int(g) for g in a.GPUS]
    free = set(LC.free_gpus(ids, max_mib=50.0))   # 빈 GPU 표시 2 MiB 때문에 50 MiB 기준(LGU 개정 3)
    used = LC.gpu_memory_used(ids)
    busy = [g for g in ids if g not in free]
    if busy:
        print(f"[GPU] 사용 중이거나 조회할 수 없는 번호 {busy}(MiB {[used.get(g) for g in busy]})는 쓰지 않는다", flush=True)
    out = [str(g) for g in ids if g in free]
    if not out:
        raise SystemExit("[GPU] 요청한 GPU 가 모두 사용 중이거나 조회할 수 없다(0 MiB 규칙). 실행하지 않는다")
    print(f"[GPU] 사용 {out}(메모리 사용 0 MiB 확인)", flush=True)
    return out


# ================================================================ fit 단위
def fit_paths(a, t, nm) -> dict:
    b = a.SHARDS / f"{a.TAG}__fit__{t}__x__{nm}"
    return dict(sigma=Path(f"{b}_sigma.npz"), runs=Path(f"{b}_runs.csv"), unit=Path(f"{b}_unit.json"))


def fit_cfg(a, nm) -> dict:
    """결과에 영향을 주는 fit 설정(대상은 조각 이름에 있으므로 넣지 않는다)."""
    d = dict(part="fit", normalizer=nm, seeds=seeds_of(a, nm), buffer_km=float(a.buffer_km), min_cal_cells=int(a.min_cal_cells),
             cap_cells=int(a.cap_cells), sigma_clip=[float(a.sigma_lo), float(a.sigma_hi)], feats="x25", q5=list(LC.Q5),
             subregion_map=Path(a.SUBMAP).name if Path(a.SUBMAP).exists() else "kmeans")
    if nm in GEN_NORMS:
        d.update(epochs=int(a.epochs), val="block", median_zero=False)
    if nm == "nflow":
        d["clamp"] = float(a.clamp)
    if nm == "cfm":
        d.update(s_gen=int(a.s_gen), ode_max_cells=ODE_MAX_CELLS)
    if nm == "cbq":
        d.update(cb=dict(CB_CFG))                                          # 스레드는 실행 자원이라 넣지 않는다(unit.json 의 threads 에만 적는다)
    if nm == "phys":
        d.update(phys_draws=int(a.phys_draws), phys_seed=0, phys_sr=list(LC.PHYS_SR), phys_nt=list(LC.PHYS_NT))
    return d


def fit_state(a, t, nm):
    p = fit_paths(a, t, nm)
    return LC.unit_state(p["unit"], [p["sigma"], p["runs"]], fit_cfg(a, nm))


def _fit_one(a, nm, name, full, Xtr, Xtr_z, z, cols, w, Xp, Xp_z, seed, df, tr, pred, t):
    """적합 한 건. 반환 (Q5 (5, n) 로그 비 척도, Qg (99, n) 또는 None, 교차 통계, 핸들 또는 None, flag)."""
    h, Qg, flag = None, None, ""
    if nm == "nflow":
        h = LC.fit_gen("nflow", Xtr_z, z, groups=cols, weights=w, seed=seed, epochs=a.epochs, clamp=a.clamp, median_zero=False, val="block")
        if full:
            Qg = LC.nflow_quantiles(h, Xp_z, LC.TAUS); Q5 = Qg[Q5_ROWS]
        else:
            Q5 = LC.nflow_quantiles(h, Xp_z, LC.Q5)
        cross = LC.crossing_stats(Qg if full else Q5)
    elif nm == "cfm":
        h = LC.fit_gen("cfm", Xtr_z, z, groups=cols, weights=w, seed=seed, epochs=a.epochs, val="block")
        if full:
            Qg = LC.gen_quantiles(h, Xp_z, LC.TAUS, S=a.s_gen, seed=seed); Q5 = Qg[Q5_ROWS]
            m = min(len(pred), ODE_MAX_CELLS)
            sub = np.sort(np.random.RandomState(seed_of("lgu-ode", t, seed)).choice(len(pred), m, replace=False)) if m else np.zeros(0, int)
            cross = LC.crossing_stats(LC.cfm_quantiles_ode(h, Xp_z[sub], LC.TAUS)) if m else LC.crossing_stats(np.zeros((99, 0)))
        else:
            Q5 = LC.gen_quantiles(h, Xp_z, LC.Q5, S=a.s_gen, seed=seed)
            cross = LC.crossing_stats(Q5)
    elif nm == "cbq":
        preds, info = LC.cb_multiquantile(Xtr, z, [Xp], seed, weights=w, threads=a.threads, alphas=LC.Q5,
                                          iterations=CB_CFG["iterations"], depth=CB_CFG["depth"], lr=CB_CFG["lr"], l2=CB_CFG["l2"])
        Q5 = preds[0].T
        if full:
            Qg = np.full((len(LC.TAUS), len(pred)), np.nan); Qg[Q5_ROWS] = Q5
        cross = {k: info[k] for k in ("frac_cells", "frac_pairs", "max_drop", "n_cells", "n_cross_cells", "n_pairs", "n_cross_pairs")}
        flag = str(info.get("flag", ""))
    else:                                                                  # phys: 적합 없음(물리 사전 예측 폭)
        L = LC.phys_log_samples(df.iloc[pred], df.iloc[tr], int(a.phys_draws), seed=0)
        q = np.quantile(L, LC.Q5, axis=0)
        Q5 = q - q[2][None, :]
        cross = LC.crossing_stats(Q5)
    return np.asarray(Q5, float), Qg, cross, h, flag


def run_fit_unit(a, t, nm) -> dict:
    """fit 단위(대상, 정규화기). 예외가 나면 status 'failed' 의 unit.json 을 쓰고 예외를 다시 던진다."""
    p = fit_paths(a, t, nm)
    p["unit"].unlink(missing_ok=True)                                       # 이전 세대의 완료 표지를 먼저 지운다
    t0 = time.time()
    cfg = fit_cfg(a, nm)
    try:
        return _fit_unit(a, t, nm, p, cfg, t0)
    except (Exception, SystemExit) as e:                                    # noqa: BLE001  SystemExit(표지 표 누락 등)도 실패로 기록한다
        LC.atomic_text(p["unit"], _dump(dict(target=t, mode="x", part="fit", normalizer=nm, tag=a.TAG, status="failed",
                                             error=f"{type(e).__name__}: {e}"[:MAX_TXT], elapsed_s=round(time.time() - t0, 1),
                                             cfg=cfg, cfg_hash=LC.cfg_hash(cfg), **code_shas())))
        raise


def _fit_unit(a, t, nm, p, cfg, t0):
    D = get_D(a); df = D.df; bi = get_bi(D); info = target_info(a, D, t)
    X_all = df[FEATS].values.astype(float)
    y = df.y.values.astype(float); s = df.s.values.astype(float)
    seeds = seeds_of(a, nm); S = len(seeds)
    G = info["G"]; N_cal = len(info["cal_idx"]); N_t = len(info["t_idx"])
    lo, hi = float(a.sigma_lo), float(a.sigma_hi)
    cal_q = np.full((S, 5, N_cal), np.nan, np.float32); cal_sigma = np.full((S, N_cal), np.nan)
    tgt_q5 = np.full((S, 5, N_t), np.nan, np.float32); tgt_sigma = np.full((S, N_t), np.nan)
    tgt_qgrid = np.full((S, len(LC.TAUS) if nm != "phys" else 0, N_t), np.nan, np.float32)
    E0_tr, rows, crossing = {}, [], []
    ok_seed = np.ones(S, bool)
    n_fit = n_fail = 0
    for name, tr, pred, cpos in train_sets(D, info):
        full = name == "full"
        E0t = LC.ls_E(y[tr], s[tr]); E0_tr[name] = E0t
        z = np.log(y[tr]) - np.log(E0t * s[tr])
        cols = bi.cols(tr)
        w = row_w(cols, a.cap_cells)
        Xtr, Xp = X_all[tr], X_all[pred]
        Xtr_z = Xp_z = None
        if nm in GEN_NORMS:
            st = LC.prep_stats(Xtr); Xtr_z = LC.prep_apply(Xtr, st); Xp_z = LC.prep_apply(Xp, st)
        for si, seed in enumerate(seeds):
            t1 = time.time()
            rec = dict(target=t, normalizer=nm, train_set=name, seed=int(seed), n_train=int(len(tr)), n_pred=int(len(pred)), E0_train=E0t)
            h, Q5, Qg, cross, flag = None, None, None, None, ""
            try:
                Q5, Qg, cross, h, flag = _fit_one(a, nm, name, full, Xtr, Xtr_z, z, cols, w, Xp, Xp_z, int(seed), df, tr, pred, t)
                failed, reason = LC.fit_failed(h, Qg if (full and nm in GEN_NORMS) else Q5)
            except Exception as e:                                          # noqa: BLE001  한 건의 예외는 그 적합만 실패로 둔다
                if LC.is_env_error(e):                                      # 환경·자원 오류(CUDA 메모리 부족 등)는 조각 실패로 둔다(재개 대상)
                    raise
                failed, reason = True, f"{type(e).__name__}: {e}"[:MAX_TXT]
            n_fit += int(nm != "phys")
            sig, clip = sigma_from_q5(Q5, lo, hi) if Q5 is not None else (None, np.nan)
            if not failed and sig is not None and not np.all(np.isfinite(sig)):
                failed, reason = True, "σ 비유한"
            if failed:
                ok_seed[si] = False; n_fail += 1
            elif full:
                tgt_q5[si] = Q5; tgt_sigma[si] = sig
                if Qg is not None and tgt_qgrid.shape[1]:
                    tgt_qgrid[si] = Qg
            else:
                cal_q[si][:, cpos] = Q5; cal_sigma[si, cpos] = sig
            qs = np.nanpercentile(sig, [5, 50, 95]) if sig is not None and np.isfinite(sig).any() else [np.nan] * 3
            rec.update(n_val=int(h.n_val) if h is not None else 0, val_loss=float(h.val_loss) if h is not None else np.nan,
                       epochs_run=int(h.epochs_run) if h is not None else 0, sigma_q05=float(qs[0]), sigma_q50=float(qs[1]),
                       sigma_q95=float(qs[2]), clip_frac=clip, cross_frac=float(cross["frac_cells"]) if cross else np.nan,
                       cross_max_drop=float(cross["max_drop"]) if cross else np.nan, failed=bool(failed), fail_reason=str(reason),
                       flag=(h.flag if h is not None else "") + (";" if h is not None and flag else "") + flag,
                       sec=round(time.time() - t1, 2), device=(h.device if h is not None else "cpu"))
            rows.append(rec)
            crossing.append(dict(train_set=name, seed=int(seed), **({k: v for k, v in (cross or {}).items()})))
            del h
    for si in range(S):                                                    # 한 학습 집합이라도 실패한 seed 는 전부 뺀다
        if not ok_seed[si]:
            cal_q[si] = np.nan; cal_sigma[si] = np.nan; tgt_q5[si] = np.nan; tgt_sigma[si] = np.nan; tgt_qgrid[si] = np.nan
    valid = [int(seeds[si]) for si in range(S) if ok_seed[si]]
    df_loc = df.loc_id.values
    LC.atomic_npz(p["sigma"], groups=np.array(G, dtype=str), E0=np.array(info["E0"]), E0_mk=np.asarray(info["E0_mk"], float),
                  E0_train_k=np.array([E0_tr[k] for k in G], float), E0_train_full=np.array(E0_tr["full"]),
                  cal_loc=df_loc[info["cal_idx"]], cal_region=np.asarray(info["cal_region"], dtype=str),
                  cal_col=bi.cols(info["cal_idx"]).astype(np.int32), cal_u=np.asarray(info["cal_u"], float),
                  cal_q=cal_q, cal_sigma=cal_sigma, tgt_loc=df_loc[info["t_idx"]], tgt_sigma=tgt_sigma, tgt_q5=tgt_q5, tgt_qgrid=tgt_qgrid,
                  seeds=np.array(seeds, np.int64), valid_seeds=np.array(valid, np.int64),
                  fit_info=np.array(json.dumps(rows, ensure_ascii=False, default=_js)),
                  crossing=np.array(json.dumps(crossing, ensure_ascii=False, default=_js)))
    LC.atomic_text(p["runs"], pd.DataFrame(rows).to_csv(index=False))
    status = "ok" if n_fail == 0 else "partial"
    unit = dict(target=t, mode="x", part="fit", normalizer=nm, tag=a.TAG, status=status, n_fit=int(n_fit), n_fit_failed=int(n_fail),
                seeds=seeds, valid_seeds=valid, K_groups=len(G), groups=G, N_cal=int(N_cal), N_target=int(N_t), n_src=int(len(info["src_idx"])),
                E0=info["E0"], E0_train_full=E0_tr["full"], comp=info["comp"], elapsed_s=round(time.time() - t0, 1),
                device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), threads=int(a.threads),
                fails=[dict(train_set=r["train_set"], seed=r["seed"], reason=r["fail_reason"]) for r in rows if r["failed"]],
                cfg=cfg, cfg_hash=LC.cfg_hash(cfg), inputs=input_shas(a), **code_shas())
    LC.atomic_text(p["unit"], _dump(unit))                                  # 완료 표지는 마지막에 쓴다
    return unit


# ================================================================ score 단위: 입력
def score_paths(a, t) -> dict:
    b = a.SHARDS / f"{a.TAG}__score__{t}__x"
    return dict(runs=Path(f"{b}_runs.csv"), scores=Path(f"{b}_scores.npz"), cells=Path(f"{b}_cells.npz"), unit=Path(f"{b}_unit.json"))


def score_cfg(a, t) -> dict:
    """score 설정. fit_sha(읽을 sigma.npz 의 해시)와 aux(보조 분석 대상 여부)는 대상마다 다르다(공통 해시에서 뺀다)."""
    return dict(part="score", normalizers=list(a.NORMS), seeds=list(a.SEEDS), perms=int(a.PERMS), lams=[lam_label(v) for v in a.LAMS],
                splits=list(a.SPLITS), n_aux=list(a.N_AUX), draws=int(a.DRAWS), emu_splits=list(LC.EMU_SPLITS), emu_draws=int(LC.EMU_DRAWS),
                kappa=float(a.kappa), buffer_km=float(a.buffer_km), min_cal_cells=int(a.min_cal_cells), levels=len(LC.LEVELS),
                label_flags=Path(a.FLAGS).name, aux=bool(t in a.AUX_EFF),
                fit_sha={nm: LC.file_sha(fit_paths(a, t, nm)["sigma"]) for nm in a.NORMS})


def cfg_common_hash(cfg: dict) -> str:
    return LC.cfg_hash({k: v for k, v in cfg.items() if k not in SCORE_UNIT_KEYS})


def score_state(a, t):
    p = score_paths(a, t)
    return LC.unit_state(p["unit"], [p["runs"], p["scores"], p["cells"]], score_cfg(a, t))


def load_fit(a, t, nm, info, D):
    """fit 조각을 읽어 유효 seed 의 σ 와 분위를 돌려준다. 반환 (dict 또는 None, 사유)."""
    ok, why = fit_state(a, t, nm)
    if not ok:
        return None, why
    p = fit_paths(a, t, nm)
    with np.load(p["sigma"], allow_pickle=False) as z:
        Z = {k: z[k] for k in z.files}
    loc = D.df.loc_id.values
    if not (np.array_equal(Z["cal_loc"], loc[info["cal_idx"]]) and np.array_equal(Z["tgt_loc"], loc[info["t_idx"]])):
        return None, "보정 셀 또는 대상 셀이 현재 자료와 다르다"
    seeds = [int(v) for v in Z["seeds"]]; valid = [int(v) for v in Z["valid_seeds"]]
    if not valid:
        return None, "유효 seed 없음(적합 실패)"
    rows = [seeds.index(v) for v in valid]
    pos = np.searchsorted(info["t_idx"], info["ev"])
    out = dict(seeds=valid, sig_c=np.asarray(Z["cal_sigma"][rows], float), sig_t=np.asarray(Z["tgt_sigma"][rows][:, pos], float))
    if not (np.all(np.isfinite(out["sig_c"])) and np.all(np.isfinite(out["sig_t"]))):
        return None, "유효 seed 의 σ 에 비유한 값이 있다"
    if nm in CQR_LEARNERS:
        out.update(q5_c=np.asarray(Z["cal_q"][rows], float), q5_t=np.asarray(Z["tgt_q5"][rows][:, :, pos], float),
                   e0k=np.asarray(Z["E0_train_k"], float), e0f=float(Z["E0_train_full"]))
    return out, ""


def build_cells(a, D, t, info, fits):
    """채점 셀·보정 셀·정규화기 σ·보조 분석 항목을 한 dict(cells.npz 의 내용)로 모은다. 반환 (C, 보조 정보)."""
    df = D.df; bi = get_bi(D); fl = get_flags(a, D)
    y = df.y.values.astype(float); s = df.s.values.astype(float)
    lat = df.lat.values.astype(float); lon = df.lon.values.astype(float)
    ev, cal, src = info["ev"], info["cal_idx"], info["src_idx"]
    E0 = info["E0"]; G = info["G"]
    e_col = bi.cols(ev).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_h37 = np.abs(np.log(y[cal]) - np.log(E0 * s[cal]))
    C = dict(e_loc=df.loc_id.values[ev], e_y=y[ev], e_s=s[ev], e_center=E0 * s[ev], e_col=e_col, e_bkey=np.asarray(bi.keys[e_col]).astype(str),
             e_early=fl["early"][ev], e_gpr=fl["gpr"][ev], e_probe=fl["probe"][ev],
             e_dsrc=LC.nearest_km(lat[ev], lon[ev], lat[src], lon[src]), e_lat=lat[ev], e_lon=lon[ev],
             c_loc=df.loc_id.values[cal], c_region=np.asarray(info["cal_region"], dtype=str), c_col=bi.cols(cal).astype(np.int64),
             c_u=np.asarray(info["cal_u"], float), c_h37=c_h37, c_early=fl["early"][cal], E0_mk=np.asarray(info["E0_mk"], float))
    for nm, f in fits.items():
        C[f"sig_c__{nm}"] = f["sig_c"]; C[f"sig_t__{nm}"] = f["sig_t"]
        if nm in CQR_LEARNERS:
            C[f"q5_c__{nm}"] = f["q5_c"]; C[f"q5_t__{nm}"] = f["q5_t"]; C[f"e0k__{nm}"] = f["e0k"]; C[f"e0f__{nm}"] = np.array(f["e0f"])
    meta = dict(target=t, perms=int(a.PERMS), lams=[float(v) for v in a.LAMS], norms=[nm for nm in FIT_NORMS if nm in fits],
                seeds={nm: list(f["seeds"]) for nm, f in fits.items()}, G=list(G), E0=float(E0), alaska=ALASKA, aux_n=[], aux_splits=[],
                sigma_clip=[float(a.sigma_lo), float(a.sigma_hi)])
    aux = dict(records={}, n_drop_aux=0, splits_used=[], splits_skipped=[])
    if t in a.AUX_EFF and a.N_AUX:
        plan = LC.emulation_plan(df, src, G, a.N_AUX, t, "x")
        cmap = np.full(len(df), -1, np.int64); cmap[cal] = np.arange(len(cal))
        for n in a.N_AUX:
            recs = [r for r in plan if int(r["n"]) == int(n)]
            rr, gg, cc, cp = [], [], [], []
            for r in recs:
                ci = np.asarray(r["cal_idx"], np.int64)
                lab_r = np.asarray(r["lab_idx"], np.int64)
                Ec = LC.center_coef(y[lab_r], s[lab_r], float(r["E0_mk"]), kappa=a.kappa)   # 대상 중심과 같은 κ(--kappa, 개정 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    res = np.log(y[ci]) - np.log(Ec * s[ci])
                pos = cmap[ci]
                keep = (pos >= 0) & np.isfinite(res)
                aux["n_drop_aux"] += int((~keep).sum())
                rr.append(res[keep]); gg.append(np.full(int(keep.sum()), str(r["region"]))); cc.append(bi.cols(ci[keep])); cp.append(pos[keep])
            if not recs or sum(len(v) for v in rr) == 0:
                continue
            C[f"ax_r__{n}"] = np.concatenate(rr); C[f"ax_g__{n}"] = np.concatenate(gg).astype(str)
            C[f"ax_col__{n}"] = np.concatenate(cc).astype(np.int64); C[f"ax_cpos__{n}"] = np.concatenate(cp).astype(np.int64)
            meta["aux_n"].append(int(n))
            aux["records"][int(n)] = dict(n_records=len(recs), K_n=len({r["region"] for r in recs}), n_entries=int(len(C[f"ax_r__{n}"])),
                                          regions=sorted({str(r["region"]) for r in recs}))
        ss = D.split_structure(t)
        any_valid = any(v["valid"] for v in ss.values())
        ok = info["ok"]; t_idx = info["t_idx"]
        for sp in a.SPLITS:
            v = ss.get(sp)
            if v is None or v["dup_of"] >= 0 or v["n_eval"] == 0 or v["n_A"] == 0 or (not v["valid"] and any_valid):
                why = "no_split" if v is None else f"dup_of_{v['dup_of']}" if v["dup_of"] >= 0 else "no_eval" if (v["n_eval"] == 0 or v["n_A"] == 0) \
                    else "invalid_nb_eval<2"
                aux["splits_skipped"].append(dict(split=sp, status=why)); continue
            A_idx, B_idx = LC.half_split_blocks(df, t_idx, sp)
            evB = B_idx[LC.eval_mask(df.iloc[B_idx])]
            evB = evB[ok[evB]]
            pos = np.searchsorted(ev, evB)
            if len(evB) == 0 or np.any(pos >= len(ev)) or not np.array_equal(ev[np.minimum(pos, len(ev) - 1)], evB):
                aux["splits_skipped"].append(dict(split=sp, status="evB_not_subset_or_empty")); continue
            En = []
            for n, d in H.cells_of(a.N_AUX, a.DRAWS, len(A_idx)):
                if n not in meta["aux_n"]:
                    continue
                sel = H.draw_cells(t, "x", sp, n, d, len(A_idx))
                lab = A_idx[sel]
                En.append((n, d, LC.center_coef(y[lab], s[lab], E0, kappa=a.kappa)))
            C[f"ax_b__{sp}"] = pos.astype(np.int64); C[f"ax_En__{sp}"] = np.array(En, float).reshape(-1, 3)
            meta["aux_splits"].append(int(sp)); aux["splits_used"].append(int(sp))
    C["meta"] = meta
    return C, aux


def save_cells(path, C):
    arrs = {k: v for k, v in C.items() if k != "meta"}
    arrs["meta"] = np.array(json.dumps(C["meta"], ensure_ascii=False, default=_js))
    return LC.atomic_npz(path, **arrs)


def load_cells(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        C = {k: z[k] for k in z.files}
    C["meta"] = json.loads(str(C["meta"]))
    return C


# ================================================================ 방법 명세(score 단위와 집계가 같은 함수를 쓴다)
def rank_bins(v, k) -> np.ndarray:
    """순위 기준 k 등분(0..k−1). 비유한 값은 −1."""
    v = np.asarray(v, float); n = len(v)
    out = np.full(n, -1, np.int64)
    ok = np.isfinite(v); m = int(ok.sum())
    if m == 0:
        return out
    o = np.argsort(v[ok], kind="stable")
    b = np.empty(m, np.int64); b[o] = (np.arange(m) * int(k)) // m
    out[ok] = b
    return out


def strata_masks(C, width_var) -> list:
    """층 표지 [(variant, mask)]: 기기(gpr, probe), 시기(early), 폭 5분위(width_var), 최근접 원천 셀 거리 3분위."""
    out = [("#gpr", np.asarray(C["e_gpr"], bool)), ("#probe", np.asarray(C["e_probe"], bool)), ("#early", np.asarray(C["e_early"], bool))]
    wb = rank_bins(width_var, 5)
    out += [(f"#w{i + 1}", wb == i) for i in range(5)]
    db = rank_bins(C["e_dsrc"], 3)
    out += [(f"#d{i + 1}", db == i) for i in range(3)]
    return out


def placebo_perm(target, nm, seed, j, groups_cal, n_eval):
    """위약 순열 (perm_c, perm_t). σ_위약 = σ[perm]. 보정 셀은 지역 안(지역마다 seed_of('lgu-placebo', 대상, 정규화기, seed, j, 지역)),
    대상 셀은 채점 셀 안(seed_of(..., 'target'))에서 섞는다."""
    g = np.asarray(groups_cal).astype(str)
    perm_c = np.arange(len(g))
    for k in np.unique(g):
        pos = np.where(g == k)[0]
        rs = np.random.RandomState(seed_of("lgu-placebo", target, nm, int(seed), int(j), str(k)))
        perm_c[pos] = pos[rs.permutation(len(pos))]
    perm_t = np.random.RandomState(seed_of("lgu-placebo", target, nm, int(seed), int(j), "target")).permutation(int(n_eval))
    return perm_c, perm_t


def region_index(G, regions) -> np.ndarray:
    Gs = np.asarray(G).astype(str)
    o = np.argsort(Gs)
    r = np.asarray(regions).astype(str)
    gi = o[np.searchsorted(Gs[o], r)]
    if len(r) and not np.array_equal(Gs[gi], r):
        raise ValueError("보정 셀의 지역이 집단 목록에 없다")
    return gi


def cqr_arrays(C, nm, j, lam) -> dict:
    """CQR 의 보정 점수와 대상 구간의 로그 끝점(위치 전용 λ 규약). α 0.1 은 (q_0.05, q_0.95), α 0.2 는 (q_0.10, q_0.90).

    z(보정) = log y − log(E0^학습_k·s) = c_u + log E0_mk − log E0^학습_k. 점수 max(q_lo − z, z − q_hi).
    대상: log 끝점 = log(E0^학습_full·s) + q'_lo 또는 q'_hi, 중심 = E0^학습_full·s·exp(λ·q_0.5).
    """
    gi = region_index(C["meta"]["G"], C["c_region"])
    z = np.asarray(C["c_u"], float) + np.log(np.asarray(C["E0_mk"], float)[gi]) - np.log(np.asarray(C[f"e0k__{nm}"], float)[gi])
    qc = LC.loc_only(np.asarray(C[f"q5_c__{nm}"][j], float), lam, mid=2)
    qt = LC.loc_only(np.asarray(C[f"q5_t__{nm}"][j], float), lam, mid=2)
    anc = np.log(float(C[f"e0f__{nm}"])) + np.log(np.asarray(C["e_s"], float))
    return dict(s10=np.maximum(qc[0] - z, z - qc[4]), s20=np.maximum(qc[1] - z, z - qc[3]),
                a_lo=[anc + qt[0], anc + qt[1]], a_hi=[anc + qt[4], anc + qt[3]], med=np.exp(anc + qt[2]))


def method_specs(C, aux=True) -> list:
    """방법 명세 목록. 명세 = dict(store, split, key, kind('grid'|'interval'), sel(채점 셀 번호), qid(보정 분위 공유 번호), cal,
    center·b(격자) 또는 a_lo·a_hi·med(구간), strata, sig). 보정 쪽과 채점 쪽 배열은 C 에서만 만든다(집계가 같은 명세를 다시 만든다)."""
    M = C["meta"]; t = M["target"]; perms = int(M["perms"]); lams = [float(v) for v in M["lams"]]
    ak = M.get("alaska", ALASKA)
    n_eval = len(C["e_y"]); allsel = np.arange(n_eval)
    g = np.asarray(C["c_region"]).astype(str); cc = np.asarray(C["c_col"], np.int64)
    au = np.abs(np.asarray(C["c_u"], float)); ah37 = np.asarray(C["c_h37"], float)
    center = np.asarray(C["e_center"], float)
    ne_sel = np.where(~np.asarray(C["e_early"], bool))[0]; ne_c = ~np.asarray(C["c_early"], bool)
    noak_c = g != ak
    do_noak = (t != ak) and bool((g == ak).any()) and bool(noak_c.any())
    ones = np.ones(n_eval)
    specs = []

    def cal(s_, mask=None):
        if mask is None:
            return dict(s=s_, g=g, c=cc)
        return dict(s=s_[mask], g=g[mask], c=cc[mask])

    def grid(store, key, sel, cal_, b, qid, strata=None, sig=None):
        specs.append(dict(store=store, split=0, key=LC.norm_key(key), kind="grid", sel=sel, qid=qid, cal=cal_, center=center[sel],
                          b=np.asarray(b, float), strata=strata, sig=sig))

    const_cal = cal(au)
    grid("all", ("const", "", 0, 0, -1), allsel, const_cal, ones, ("all", "const"), strata=strata_masks(C, center))
    grid("all", ("const@h37", "", 0, 0, -1), allsel, cal(ah37), ones, ("all", "const@h37"))
    if do_noak:
        grid("all", ("const@noak", "", 0, 0, -1), allsel, cal(au, noak_c), ones, ("all", "const@noak"))
    grid("noearly", ("const", "", 0, 0, -1), ne_sel, cal(au, ne_c), ones[ne_sel], ("noearly", "const"))
    for nm in [v for v in FIT_NORMS if v in M["norms"]]:
        seeds = [int(v) for v in M["seeds"][nm]]
        for j, seed in enumerate(seeds):
            ks = key_seed(nm, seed)
            sc = np.asarray(C[f"sig_c__{nm}"][j], float); stt = np.asarray(C[f"sig_t__{nm}"][j], float)
            grid("all", (nm, "", 0, 0, ks), allsel, cal(au / sc), stt, ("all", nm, ks),
                 strata=strata_masks(C, stt) if nm in STRATA_NORMS else None, sig=stt)
            grid("all", (f"{nm}@mw", "", 0, 0, ks), allsel, const_cal, stt / float(np.mean(stt)), ("all", "const"), sig=stt)
            if nm in PLACEBO_NORMS:
                for p in range(perms):
                    pc, pt = placebo_perm(t, nm, seed, p, g, n_eval)
                    grid("all", (f"{nm}#p{p}", "", 0, 0, ks), allsel, cal(au / sc[pc]), stt[pt], ("all", f"{nm}#p{p}", ks), sig=stt[pt])
            if nm in SENS_NORMS:
                if do_noak:
                    grid("all", (f"{nm}@noak", "", 0, 0, ks), allsel, cal(au / sc, noak_c), stt, ("all", f"{nm}@noak", ks), sig=stt)
                grid("noearly", (nm, "", 0, 0, ks), ne_sel, cal(au / sc, ne_c), stt[ne_sel], ("noearly", nm, ks), sig=stt[ne_sel])
            if nm in CQR_LEARNERS and f"q5_c__{nm}" in C:
                for lam in lams:
                    arr = cqr_arrays(C, nm, j, lam)
                    for kind in ("pool", "hier"):
                        m = f"cqr_{kind}:{nm}@lam{lam_label(lam)}"
                        specs.append(dict(store="all", split=0, key=LC.norm_key((m, "", 0, 0, ks)), kind="interval", sel=allsel,
                                          qid=("all", m, ks), cal=dict(s=[arr["s10"], arr["s20"]], g=g, c=cc, pooled=(kind == "pool")),
                                          a_lo=arr["a_lo"], a_hi=arr["a_hi"], med=arr["med"], strata=None, sig=None))
    if aux:
        specs += aux_specs(C)
    return specs


def aux_specs(C) -> list:
    """보조 n > 0 의 명세(저장소 (대상, 분할), 키 (방법, '', n, 추출, seed)). 보정 분위는 (n, 방법, seed)마다 하나(qid 공유)."""
    M = C["meta"]; es = np.asarray(C["e_s"], float); specs = []
    for n in M.get("aux_n", []):
        r = np.abs(np.asarray(C[f"ax_r__{n}"], float)); ga = np.asarray(C[f"ax_g__{n}"]).astype(str)
        ca = np.asarray(C[f"ax_col__{n}"], np.int64); cp = np.asarray(C[f"ax_cpos__{n}"], np.int64)
        for nm in AUX_NORMS:
            if nm == "const":
                runs = [(-1, None)]
            elif nm in M["norms"]:
                runs = [(key_seed(nm, s), j) for j, s in enumerate(M["seeds"][nm])]
            else:
                continue
            for ks, j in runs:
                if j is None:
                    sc, sig_t = r, None
                else:
                    sc, sig_t = r / np.asarray(C[f"sig_c__{nm}"][j], float)[cp], np.asarray(C[f"sig_t__{nm}"][j], float)
                cal_ = dict(s=sc, g=ga, c=ca)
                qid = ("aux", int(n), nm, int(ks))
                for sp in M.get("aux_splits", []):
                    pos = np.asarray(C[f"ax_b__{sp}"], np.int64)
                    for nn, d, En in np.asarray(C[f"ax_En__{sp}"], float):
                        if int(nn) != int(n):
                            continue
                        b = np.ones(len(pos)) if sig_t is None else sig_t[pos]
                        specs.append(dict(store="aux", split=int(sp), key=LC.norm_key((nm, "", int(n), int(d), ks)), kind="grid", sel=pos,
                                          qid=qid, cal=cal_, center=float(En) * es[pos], b=b, strata=None,
                                          sig=None if sig_t is None else sig_t[pos]))
    return specs


def store_id(spec, t):
    if spec["store"] == "all":
        return (f"{t}|all", 0)
    if spec["store"] == "noearly":
        return (f"{t}~noearly", 0)
    return (str(t), int(spec["split"]))


def interval_q(cal) -> np.ndarray:
    """구간 명세의 보정 분위 (α 0.1, α 0.2). pooled 는 셀 교환성 유한표본 분위, 아니면 지역 등가중 분위."""
    out = []
    for ai, al in enumerate(ALPHAS2):
        s_ = cal["s"][ai]
        out.append(LC.pooled_quantile(s_, al) if cal["pooled"] else float(LC.hier_quantiles(s_, cal["g"], [1.0 - al])[0]))
    return np.array(out, float)


def interval_ends(a_lo, a_hi, med, Q):
    """로그 끝점과 보정 분위로 구간 [exp(a_lo − Q), exp(a_hi + Q)]. 하한이 상한을 넘으면 두 끝점을 med 로 둔다."""
    with np.errstate(over="ignore", invalid="ignore"):
        lo = np.exp(np.asarray(a_lo, float) - Q); hi = np.exp(np.asarray(a_hi, float) + Q)
    inv = lo > hi
    return np.where(inv, med, lo), np.where(inv, med, hi)


def score_from_specs(C, specs):
    """명세를 채점해 ScoreStore 와 runs 행을 만든다. 격자 방법은 LEVELS 49점의 보정 분위로 99점 격자를 만들어 score_cells 로 채점한다."""
    t = C["meta"]["target"]
    stores, rows, qcache = {}, [], {}
    e_y = np.asarray(C["e_y"], float); e_bkey = np.asarray(C["e_bkey"]).astype(str); e_col = np.asarray(C["e_col"], np.int64)
    for sp in specs:
        sid = store_id(sp, t); sel = np.asarray(sp["sel"], np.int64)
        st = stores.get(sid)
        if st is None:
            st = LC.ScoreStore(sid[0], sid[1], e_bkey[sel], e_col[sel], meta=dict(scope=sp["store"], n_cells=int(len(sel))))
            stores[sid] = st
        elif len(st.codes) != len(sel):
            raise RuntimeError(f"저장소 {sid} 의 셀 수가 명세와 다르다")
        y = e_y[sel]; cal_ = sp["cal"]
        if sp["kind"] == "grid":
            q = qcache.get(sp["qid"])
            if q is None:
                q = LC.hier_quantiles(cal_["s"], cal_["g"], LC.LEVELS); qcache[sp["qid"]] = q
            Q = LC.grid_from_center(sp["center"], q[:, None] * np.asarray(sp["b"], float)[None, :])
            sc = LC.score_cells(y, Q)
            q90, q80 = float(q[LV90]), float(q[LV80])
        else:
            Qa = qcache.get(sp["qid"])
            if Qa is None:
                Qa = interval_q(cal_); qcache[sp["qid"]] = Qa
            lo10, hi10 = interval_ends(sp["a_lo"][0], sp["a_hi"][0], sp["med"], Qa[0])
            lo20, hi20 = interval_ends(sp["a_lo"][1], sp["a_hi"][1], sp["med"], Qa[1])
            sc = LC.score_interval_only(y, lo10, hi10, lo20, hi20, sp["med"])
            q90, q80 = float(Qa[0]), float(Qa[1])
        st.add(sp["key"], sc)
        m, v, n, d, seed = sp["key"]
        for var, mask in (sp.get("strata") or []):
            st.add((m, var, n, d, seed), sc, valid=np.asarray(sc["valid"], bool) & np.asarray(mask, bool))
        rows.append(run_row(t, sp, st, sc, q90, q80))
    return stores, rows


def run_row(t, sp, st, sc, q90, q80) -> dict:
    m, v, n, d, seed = sp["key"]
    cal_ = sp["cal"]
    s0 = cal_["s"][0] if sp["kind"] == "interval" else cal_["s"]
    r = dict(target=t, scope={"all": "all", "noearly": "all~noearly", "aux": "B"}[sp["store"]], method=m, variant=v, n=n, draw=d, seed=seed,
             split=int(st.split), K_groups=int(len(np.unique(cal_["g"]))), N_cal=int(len(s0)), q90=q90, q80=q80,
             n_eval=int(len(sp["sel"])), n_blocks=int(st.nb))
    for mt in LC.METRICS:
        r[mt] = st.mean(sp["key"], mt, "cell")
    r["rmse"] = st.rmse(sp["key"], "cell")
    r["coverage_beq"] = st.mean(sp["key"], "cov10", "beq")
    r["inf_frac"] = r["inf10"]
    r["sigma_mean"] = float(np.mean(sp["sig"])) if sp.get("sig") is not None else 1.0
    r["flag"] = "interval_only" if sp["kind"] == "interval" else ""
    ph = LC.pit_hist(sc["pit"]) if sp["kind"] == "grid" else np.zeros(LC.PIT_BINS, np.int64)
    r.update({f"pit_h{i}": int(c) for i, c in enumerate(ph)})
    return r


# ================================================================ 2단 재표집(전역 블록 재표집)
def pooled_quantile_mult(scores, mult, alpha) -> float:
    """다중도 가중 셀 합동 유한표본 분위: 다중도로 늘린 표본의 ceil((M + 1)(1 − α))번째 순서통계. M = Σ 다중도. 넘으면 inf, M = 0 이면 NaN.
    다중도가 모두 1 이면 lgu_common.pooled_quantile 과 같다."""
    s = np.asarray(scores, float); m = np.asarray(mult, float)
    keep = ~np.isnan(s); s, m = s[keep], m[keep]
    o = np.argsort(s, kind="stable"); s, m = s[o], m[o]
    cum = np.cumsum(m)
    M = float(cum[-1]) if len(cum) else 0.0
    if M <= 0:
        return np.nan
    k = math.ceil((M + 1.0) * (1.0 - float(alpha)) - 1e-9)
    if k > M:
        return np.inf
    return float(s[int(np.searchsorted(cum, max(k, 1) - 0.5))])


def pooled_quantile_boot(scores, cols, W, alpha, chunk=BOOT_CHUNK) -> np.ndarray:
    """재표집마다 pooled_quantile_mult(scores, W[r, cols], alpha). (R,)."""
    s = np.asarray(scores, float); c = np.asarray(cols, np.int64)
    keep = ~np.isnan(s); s, c = s[keep], c[keep]
    R = int(np.asarray(W).shape[0]); out = np.full(R, np.nan)
    if len(s) == 0:
        return out
    o = np.argsort(s, kind="stable"); s_o, c_o = s[o], c[o]
    for r0 in range(0, R, int(chunk)):
        Mx = np.asarray(W[r0:r0 + int(chunk)])[:, c_o].astype(float)
        cum = np.cumsum(Mx, axis=1); tot = cum[:, -1]
        k = np.ceil((tot + 1.0) * (1.0 - float(alpha)) - 1e-9)
        idx = (cum < (k[:, None] - 0.5)).sum(axis=1)
        res = np.where(k <= tot, s_o[np.minimum(idx, len(s_o) - 1)], np.inf)
        out[r0:r0 + Mx.shape[0]] = np.where(tot > 0, res, np.nan)
    return out


def qboot_spec(sp, W, chunk=BOOT_CHUNK) -> np.ndarray:
    """명세의 재표집별 보정 분위 (R, 2): 열 0 = α 0.1, 열 1 = α 0.2."""
    cal_ = sp["cal"]
    if sp["kind"] == "grid":
        return LC.hier_quantiles_boot(cal_["s"], cal_["g"], cal_["c"], W, BOOT_LEVELS, chunk=chunk)
    cols = []
    for ai, al in enumerate(ALPHAS2):
        if cal_["pooled"]:
            cols.append(pooled_quantile_boot(cal_["s"][ai], cal_["c"], W, al, chunk))
        else:
            cols.append(LC.hier_quantiles_boot(cal_["s"][ai], cal_["g"], cal_["c"], W, [1.0 - al], chunk=chunk)[:, 0])
    return np.stack(cols, 1)


def _agg_rc(Sb, Wc, cntf, den_c, den_b):
    """재표집별 블록 합 Sb (Rc, nb)와 다중도 Wc (Rc, nb)의 셀 가중·블록 등가중 평균. 뽑힌 블록의 inf·NaN 은 전파한다."""
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        pos = Wc > 0
        cell = np.where(pos, Wc * Sb, 0.0).sum(1)
        beq = np.where(pos, Wc * (Sb / cntf[None, :]), 0.0).sum(1)
        cell = np.where(den_c > 0, cell / np.where(den_c > 0, den_c, 1.0), np.nan)
        beq = np.where(den_b > 0, beq / np.where(den_b > 0, den_b, 1.0), np.nan)
    return cell, beq


def twostage_cells(y, cols, A_lo, A_hi, B, med, Qb, W, chunk=BOOT_CHUNK) -> dict:
    """셀 구간 log L = A_lo − B·Q, log U = A_hi + B·Q(하한 > 상한이면 두 끝점 = med)의 재표집 분포.

    y, cols, B, med: (n,). A_lo, A_hi: α 0.1, 0.2 의 (n,) 두 개. Qb (R, 2). W (R, NC) 전역 다중도.
    반환 {지표: (셀 가중 (R,), 블록 등가중 (R,))}, 지표 = METRICS_2STAGE(se 는 RMSE). 다중도가 모두 1 이면 ScoreStore 의 평균과 같다.
    """
    y = np.asarray(y, float); cols = np.asarray(cols, np.int64)
    order = np.argsort(cols, kind="stable")
    ub, starts, cnt = np.unique(cols[order], return_index=True, return_counts=True)
    ys = y[order]; Bs = np.asarray(B, float)[order]; ms = np.asarray(med, float)[order]
    Alo = [np.asarray(v, float)[order] for v in A_lo]; Ahi = [np.asarray(v, float)[order] for v in A_hi]
    Qb = np.asarray(Qb, float); R = Qb.shape[0]; cntf = cnt.astype(float)
    out = {m: (np.full(R, np.nan), np.full(R, np.nan)) for m in LC.METRICS_2STAGE}
    se_blk = np.add.reduceat((ms - ys) ** 2, starts) if len(ys) else np.zeros(0)
    rse_blk = np.sqrt(se_blk / cntf) if len(ys) else np.zeros(0)
    for r0 in range(0, R, int(chunk)):
        r1 = min(R, r0 + int(chunk))
        Wc = np.asarray(W[r0:r1])[:, ub].astype(float)
        den_c = Wc @ cntf; den_b = Wc.sum(1)
        for ai, al in enumerate(ALPHAS2):
            sfx = LC.alpha_suffix(al)
            Q = Qb[r0:r1, ai][:, None]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                lo = np.exp(Alo[ai][None, :] - Bs[None, :] * Q); hi = np.exp(Ahi[ai][None, :] + Bs[None, :] * Q)
                inv = lo > hi
                lo = np.where(inv, ms[None, :], lo); hi = np.where(inv, ms[None, :], hi)
                fin = np.isfinite(lo) & np.isfinite(hi)
                wid = np.where(fin, hi - lo, np.inf)
                IS = np.where(fin, wid + (2.0 / al) * np.maximum(lo - ys[None, :], 0.0) + (2.0 / al) * np.maximum(ys[None, :] - hi, 0.0), np.inf)
                cov = ((lo <= ys[None, :]) & (ys[None, :] <= hi)).astype(float)
                lhw = (np.log(hi) - np.log(lo)) / 2.0
            for nm_, Mx in ((f"is{sfx}", IS), (f"cov{sfx}", cov), (f"wid{sfx}", wid), (f"lhw{sfx}", lhw)):
                with np.errstate(invalid="ignore", over="ignore"):
                    Sb = np.add.reduceat(Mx, starts, axis=1)
                c_, b_ = _agg_rc(Sb, Wc, cntf, den_c, den_b)
                out[nm_][0][r0:r1] = c_; out[nm_][1][r0:r1] = b_
        with np.errstate(invalid="ignore", divide="ignore"):
            c_ = np.where(den_c > 0, (Wc @ se_blk) / np.where(den_c > 0, den_c, 1.0), np.nan)
            b_ = np.where(den_b > 0, (Wc @ rse_blk) / np.where(den_b > 0, den_b, 1.0), np.nan)
        out["se"][0][r0:r1] = np.sqrt(c_); out["se"][1][r0:r1] = b_
    bad = ~np.all(np.isfinite(Qb) | np.isinf(Qb), axis=1)                  # 보정 분위가 NaN 인 재표집(다중도 0)
    if bad.any():
        for m in out:
            out[m][0][bad] = np.nan; out[m][1][bad] = np.nan
    return out


def _nanmean_stack(arrs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(arrs), axis=0)


def twostage_from_specs(C, specs, W, chunk=BOOT_CHUNK) -> dict:
    """모든 명세의 2단 분포. 반환 {('all'|'noearly', key) 또는 ('aux', (method, n, seed)): {지표: (셀, 블록)}}.
    보조 n 은 같은 분할 안에서 추출 평균(nanmean), 분할의 같은 번호 평균(np.mean) 순으로 결합한다(LG 규칙)."""
    qcache, out, aux = {}, {}, {}
    e_y = np.asarray(C["e_y"], float); e_col = np.asarray(C["e_col"], np.int64)
    for sp in specs:
        Qb = qcache.get(sp["qid"])
        if Qb is None:
            Qb = qboot_spec(sp, W, chunk); qcache[sp["qid"]] = Qb
        sel = np.asarray(sp["sel"], np.int64)
        if sp["kind"] == "grid":
            lc = np.log(np.asarray(sp["center"], float))
            res = twostage_cells(e_y[sel], e_col[sel], [lc, lc], [lc, lc], sp["b"], sp["center"], Qb, W, chunk)
        else:
            res = twostage_cells(e_y[sel], e_col[sel], sp["a_lo"], sp["a_hi"], np.ones(len(sel)), sp["med"], Qb, W, chunk)
        if sp["store"] == "aux":
            m, v, n, d, seed = sp["key"]
            aux.setdefault((m, int(n), int(seed)), {}).setdefault(int(sp["split"]), []).append(res)
        else:
            out[(sp["store"], sp["key"])] = res
    for k, by in aux.items():
        per = []
        for spl in sorted(by):
            per.append({mt: tuple(_nanmean_stack([r[mt][w] for r in by[spl]]) for w in (0, 1)) for mt in LC.METRICS_2STAGE})
        out[("aux", k)] = {mt: tuple(np.mean(np.stack([p_[mt][w] for p_ in per]), axis=0) for w in (0, 1)) for mt in LC.METRICS_2STAGE}
    return out


# ================================================================ score 단위
def run_score_unit(a, t) -> dict:
    """score 단위(대상). 예외가 나면 status 'failed' 의 unit.json 을 쓰고 예외를 다시 던진다."""
    p = score_paths(a, t)
    p["unit"].unlink(missing_ok=True)
    t0 = time.time()
    cfg = score_cfg(a, t)
    try:
        return _score_unit(a, t, p, cfg, t0)
    except (Exception, SystemExit) as e:                                    # noqa: BLE001  SystemExit 도 실패로 기록한다
        LC.atomic_text(p["unit"], _dump(dict(target=t, mode="x", part="score", tag=a.TAG, status="failed",
                                             error=f"{type(e).__name__}: {e}"[:MAX_TXT], elapsed_s=round(time.time() - t0, 1),
                                             cfg=cfg, cfg_hash=LC.cfg_hash(cfg), cfg_common=cfg_common_hash(cfg), **code_shas())))
        raise


def _score_unit(a, t, p, cfg, t0):
    D = get_D(a); info = target_info(a, D, t)
    if len(info["ev"]) == 0:
        raise RuntimeError(f"{t}: 채점 셀이 없다")
    fits, missing = {}, {}
    for nm in a.NORMS:
        f, why = load_fit(a, t, nm, info, D)
        if f is None:
            missing[nm] = why
        else:
            fits[nm] = f
    C, auxinfo = build_cells(a, D, t, info, fits)
    specs = method_specs(C)
    stores, rows = score_from_specs(C, specs)
    order = sorted(stores, key=lambda k: (0 if k[0].endswith("|all") else 1 if k[0].endswith("~noearly") else 2, k[1]))
    LC.save_score_stores([stores[k] for k in order], p["scores"])
    save_cells(p["cells"], C)
    LC.atomic_text(p["runs"], pd.DataFrame(rows).to_csv(index=False))
    unit = dict(target=t, mode="x", part="score", tag=a.TAG, status="ok" if not missing else "partial", missing=missing,
                normalizers=sorted(fits), seeds={nm: f["seeds"] for nm, f in fits.items()}, n_eval=int(len(info["ev"])),
                n_drop_eval=info["n_drop_eval"], N_cal=int(len(info["cal_idx"])), K_groups=len(info["G"]), groups=info["G"],
                E0=info["E0"], E0_mk=list(map(float, info["E0_mk"])), n_src=int(len(info["src_idx"])), comp=info["comp"],
                n_specs=len(specs), n_keys={f"{k[0]}|s{k[1]}": len(stores[k]) for k in order}, aux=auxinfo,
                elapsed_s=round(time.time() - t0, 1), threads=int(a.threads), cfg=cfg, cfg_hash=LC.cfg_hash(cfg),
                cfg_common=cfg_common_hash(cfg), inputs=input_shas(a), **code_shas())
    LC.atomic_text(p["unit"], _dump(unit))
    return unit


# ================================================================ 집계: 수준과 대비
def _kn(M):
    """키 축(0) nanmean. 모든 키가 NaN 인 열은 경고 없이 NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(M, float), axis=0)


def level1(sb, keys, metric, nboot, seed=0):
    """한 방법 묶음(키 목록)의 지표 수준과 1단 분포(LG 규칙: 분할마다 boot_weights, 키 nanmean, 분할의 같은 번호 평균).
    metric 'se' 는 RMSE. 반환 dict(point=(셀, 블록), dist=((R,), (R,)) 또는 None, n_splits, n_keys) 또는 None."""
    rm = metric == "se"
    fc, fb = (LC.agg_rmse_cell, LC.agg_rmse_beq) if rm else (LC.agg_cell, LC.agg_beq)
    pts, ptsb, ds, dsb, nk = [], [], [], [], 0
    for sp, st in sorted(sb.items()):
        ks = st.select(keys)
        if not ks:
            continue
        S_, Cn = st.matrices(ks, metric)
        one = np.ones((1, st.nb))
        pts.append(float(_kn(fc(S_, Cn, one)[:, 0]))); ptsb.append(float(_kn(fb(S_, Cn, one)[:, 0]))); nk += len(ks)
        if int(nboot) > 0:
            Wb = H4.boot_weights(st.nb, int(nboot), seed_of(seed, sp))
            ds.append(_kn(fc(S_, Cn, Wb))); dsb.append(_kn(fb(S_, Cn, Wb)))
    if not pts:
        return None
    return dict(point=(float(np.mean(pts)), float(np.mean(ptsb))),
                dist=(np.mean(np.stack(ds), 0), np.mean(np.stack(dsb), 0)) if ds else None, n_splits=len(pts), n_keys=nk)


def pool_of(per: dict, regions) -> dict:
    """지역 층화 평균. 풀 조건: 결과가 있고 채점 블록(사용 분할의 합집합) 8개 이상이고 점 추정이 유한한 지역.
    분포는 같은 재표집 번호끼리 지역 등가중 평균(lgu_common.strat_mean)."""
    avail = [t for t in regions if per.get(t) is not None and per[t]["nb"] >= LC.MIN_BLOCKS_CI and np.isfinite(per[t]["point"][0])]
    res = dict(avail=avail, per=per, point=(np.nan, np.nan), d1=None, d2=None, n_splits_min=0)
    if not avail:
        return res
    res["point"] = tuple(float(np.mean([per[t]["point"][w] for t in avail])) for w in (0, 1))
    for k in ("d1", "d2"):
        if all(per[t][k] is not None for t in avail):
            res[k] = tuple(LC.strat_mean([per[t][k][w] for t in avail]) for w in (0, 1))
    res["n_splits_min"] = int(min(per[t]["n_splits"] for t in avail))
    return res


def ci_pair(d):
    """(dist_cell, dist_beq) → (lo, hi, lo_beq, hi_beq). 없으면 NaN."""
    if d is None:
        return (np.nan, np.nan, np.nan, np.nan)
    lo, hi, _ = LC.ci95(d[0]); lob, hib, _ = LC.ci95(d[1])
    return lo, hi, lob, hib


def _cls(v) -> str:
    return str(v).split("(")[0]


def b2_verdict(ci, points, cov, seeds_ok, n_pool, need_neg=3, n_min=3) -> str:
    """LGU-B2 의 판정(계획서 4.4절). ci = (lo, hi, lo_beq, hi_beq)(2단), points = 풀 지역의 셀 가중 점 추정, cov = 풀 평균 cov10(셀 가중).
    '전이'(네 조건 모두), '조건부 우세(어긋난 조건: …)', '미결정', '악화', '판정 불가', '판정 불가(적합 실패)'."""
    lo, hi, lob, hib = [float(x) for x in ci]
    if n_pool < n_min or any(np.isnan(v) for v in (lo, hi, lob, hib)):
        return "판정 불가"
    if not seeds_ok:
        return "판정 불가(적합 실패)"
    if hi < 0 and hib < 0:
        bad = []
        nn = int(sum(1 for p_ in points if np.isfinite(p_) and p_ < 0))
        if nn < need_neg:
            bad.append(f"(2) 셀 가중 점 추정이 음수인 지역 {nn}/{len(points)}")
        if not (np.isfinite(cov) and cov >= LC.COV_MIN):
            bad.append(f"(3) 풀 평균 cov10 {cov:.3f} < {LC.COV_MIN:.2f}")
        return "전이" if not bad else "조건부 우세(어긋난 조건: " + "; ".join(bad) + ")"
    if lo > 0 and lob > 0:
        return "악화"
    return "미결정"


class Summ:
    """집계 문맥: 저장소(1단), 대상별 2단 분포, cells 메타."""

    def __init__(self, a, stores, cells, T2):
        self.a, self.stores, self.cells, self.T2 = a, stores, cells, T2
        self.targets = [t for t in a.TARGETS if t in cells]

    def meta(self, t):
        return self.cells[t]["meta"]

    def perms(self, t):
        return int(self.meta(t)["perms"])

    def seeds(self, t, nm) -> list:
        """저장 키의 seed 목록(유효 seed). const 계열은 [−1]. 없는 정규화기는 []."""
        if nm.startswith("const"):
            return [-1]
        base = nm.split("@")[0].split("#")[0]
        M = self.meta(t) if t in self.cells else None
        if M is None or base not in M["norms"]:
            return []
        return [key_seed(base, s) for s in M["seeds"][base]]

    @staticmethod
    def boot_seed(t, scope) -> int:
        """1단 재표집의 seed(개정 1): 대상·범위마다 seed_of('lgu-b', 대상, 범위). 같은 대상·범위의 수준(level)과 대비(contrast)는 같은
        seed 를 써서 짝지음을 유지하고, 대상 사이에서는 재표집 난수열이 달라 층화 평균의 지역별 재표집이 독립이다."""
        return int(seed_of("lgu-b", str(t), str(scope)))

    def sb(self, t, scope) -> dict:
        if scope == "aux":
            return {sp: st for (nm, sp), st in self.stores.items() if nm == str(t)}
        k = (f"{t}|all", 0) if scope == "all" else (f"{t}~noearly", 0)
        return {0: self.stores[k]} if k in self.stores else {}

    def keys(self, scope, pairs, n=0) -> list:
        if scope in ("all", "noearly"):
            return [LC.norm_key((m, "", 0, 0, s)) for m, s in pairs]
        return [LC.norm_key((m, "", int(n), d, s)) for m, s in pairs for d in range(max(1, int(self.a.DRAWS)))]

    def d2(self, t, scope, pairs, metric, n=0):
        T = self.T2.get(t, {})
        if scope in ("all", "noearly"):
            ks = [(scope, LC.norm_key((m, "", 0, 0, s))) for m, s in pairs]
        else:
            ks = [("aux", (m, int(n), int(s))) for m, s in pairs]
        ds = [T[k] for k in ks if k in T]
        if not ds or metric not in ds[0]:
            return None
        return tuple(_kn([d[metric][w] for d in ds]) for w in (0, 1))

    @staticmethod
    def nblocks(sb, keys) -> int:
        bl = set()
        for st in sb.values():
            if st.select(keys):
                bl |= set(np.asarray(st.blocks).astype(str).tolist())
        return len(bl)

    def level(self, t, scope, pairs, metric, n=0, nboot=None):
        sb = self.sb(t, scope)
        if not sb or not pairs:
            return None
        ks = self.keys(scope, pairs, n)
        L = level1(sb, ks, metric, self.a.nboot if nboot is None else nboot, self.boot_seed(t, scope))
        if L is None:
            return None
        return dict(point=L["point"], d1=L["dist"], d2=self.d2(t, scope, pairs, metric, n) if metric in LC.METRICS_2STAGE else None,
                    nb=self.nblocks(sb, ks), n_splits=L["n_splits"], n_keys=L["n_keys"])

    def contrast(self, t, scope, A, B, metric="is10", n=0):
        sb = self.sb(t, scope)
        if not sb or not A or not B:
            return None
        kA, kB = self.keys(scope, A, n), self.keys(scope, B, n)
        r1 = LC.one_stage_delta(sb, kA, kB, metric, self.a.nboot, self.boot_seed(t, scope))
        if not r1["n_splits"]:
            return None
        dA, dB = self.d2(t, scope, A, metric, n), self.d2(t, scope, B, metric, n)
        d2 = (dA[0] - dB[0], dA[1] - dB[1]) if dA is not None and dB is not None else None
        return dict(point=(r1["delta"], r1["delta_beq"]), d1=(r1["dist"], r1["dist_beq"]) if r1["dist"] is not None else None, d2=d2,
                    nb=self.nblocks(sb, kA + kB), n_splits=int(r1["n_splits"]), n_keys=(int(r1["n_keys_A"]), int(r1["n_keys_B"])),
                    pair_ok=bool(r1["pair_ok"]), n_unpaired=(int(r1["n_unpaired_A"]), int(r1["n_unpaired_B"])),
                    n_nan_keys=(int(r1["n_nan_keys_A"]), int(r1["n_nan_keys_B"])))


TEST_COLS = ["test", "role", "contrast", "pool", "regions", "n", "metric", "delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq",
             "ci_kind", "verdict", "verdict_1stage", "note_cal", "margin", "margin_beq", "margin_aux", "half_width", "ref_score", "eq_state",
             "eq_state_aux", "coverage_pool", "cov_ok", "p_boot", "holm_p", "pool_regions", "splits_min", "splits_expected", "verdict_2region",
             "note_alaska", "statement", "blind", "nboot", "n_neg_regions", "seeds_ok", "verdict_3region", "note_rw",
             "ci_lo_1stage", "ci_hi_1stage", "ci_lo_beq_1stage", "ci_hi_beq_1stage"]


def test_row(a, test, role, contrast, pool_name, regions, n, metric, P, ref=None, blind="맹검", statement="", verdict=None,
             verdict_1=None, level=False, splits_expected=1, **extra) -> dict:
    """판정 표의 한 행. P = pool_of 결과. 2단 CI 가 주(없으면 1단 CI 로 대신하고 ci_kind '1단'). 구간 점수의 동등성 한계는 기준(ref, B 쪽
    방법의 풀 점 추정)의 5 %(보조 10 %)이고 가중별로 둔다. level 이면 수준 행(대비가 아님)이라 4분 판정을 쓰지 않는다."""
    lo1, hi1, lob1, hib1 = ci_pair(P["d1"])
    has2 = P["d2"] is not None
    lo, hi, lob, hib = ci_pair(P["d2"]) if has2 else (lo1, hi1, lob1, hib1)
    k, N = len(P["avail"]), len(regions)
    m5 = LC.rel_margin(ref[0], ref[1], LC.EQ_REL) if ref is not None else (np.nan, np.nan)
    m10 = LC.rel_margin(ref[0], ref[1], LC.EQ_REL_AUX) if ref is not None else (np.nan, np.nan)
    if level:
        v2 = verdict or ""; v1 = verdict_1 if verdict_1 is not None else ""
    else:
        v2 = verdict if verdict is not None else (LC.verdict4(lo, hi, lob, hib, m5) if k >= 2 else "판정 불가")
        v1 = verdict_1 if verdict_1 is not None else (LC.verdict4(lo1, hi1, lob1, hib1, m5) if k >= 2 else "판정 불가")
    note_cal = ""
    if has2 and not level and _cls(v1) != _cls(v2) and not str(v1).startswith("판정 불가") and not str(v2).startswith("판정 불가"):
        note_cal = "보정 불확실성 의존"
    row = dict(test=test, role=role, contrast=contrast, pool=pool_name, regions=",".join(P["avail"]), n=int(n), metric=metric,
               delta=P["point"][0], ci_lo=lo, ci_hi=hi, delta_beq=P["point"][1], ci_lo_beq=lob, ci_hi_beq=hib,
               ci_kind="2단" if has2 else ("1단" if P["d1"] is not None else ""), verdict=v2, verdict_1stage=v1, note_cal=note_cal,
               margin=m5[0], margin_beq=m5[1], margin_aux=m10[0], half_width=LC.half_width(lo, hi, lob, hib),
               ref_score=ref[0] if ref is not None else np.nan,
               eq_state=LC.eq_state(lo, hi, lob, hib, m5) if (ref is not None and not level) else "",
               eq_state_aux=LC.eq_state(lo, hi, lob, hib, m10) if (ref is not None and not level) else "",
               coverage_pool=np.nan, cov_ok="", p_boot=LC.boot_p(P["d2"][0]) if has2 else (LC.boot_p(P["d1"][0]) if P["d1"] is not None else np.nan),
               holm_p=np.nan, pool_regions=f"{k}/{N}" if k == N else f"부분(지역 {k}/{N})", splits_min=int(P["n_splits_min"]),
               splits_expected=int(splits_expected), verdict_2region="", note_alaska="", statement=statement, blind=blind, nboot=int(a.nboot),
               n_neg_regions=np.nan, seeds_ok="", verdict_3region="", note_rw="",
               ci_lo_1stage=lo1, ci_hi_1stage=hi1, ci_lo_beq_1stage=lob1, ci_hi_beq_1stage=hib1)
    row.update(extra)
    return row


def expected_splits(a, D, t) -> int:
    """보조 분석의 기대 분할 수(중복이 아니고 유효한 분할. 유효 분할이 없으면 중복이 아닌 분할)."""
    ss = D.split_structure(t)
    v = [sp for sp in a.SPLITS if sp in ss and ss[sp]["dup_of"] < 0 and ss[sp]["valid"]]
    return len(v) if v else len([sp for sp in a.SPLITS if sp in ss and ss[sp]["dup_of"] < 0])


def build_tests(S, D, precision=False):
    """LGU-B1–B6 의 판정 행과 정밀도 항목. precision 이면 판정 행을 만들지 않고 정밀도 항목만 돌려준다."""
    a = S.a; T = S.targets
    rows, prec = [], []
    lam1 = lam_label(1.0)

    def ref_level(regions, method, scope="all", n=0):
        per = {t: S.level(t, scope, [(method, s) for s in S.seeds(t, method)], "is10", n, nboot=0) if t in T else None for t in regions}
        return pool_of(per, regions)

    def add_prec(contrast, pool_name, n, P, ref):
        prec.append(dict(contrast=contrast, pool=pool_name, n=n, d2=P["d2"] if P["d2"] is not None else P["d1"],
                         kind="2단" if P["d2"] is not None else "1단", ref=ref))

    def prec_all(contrast, per, regions, pools, n, refm, scope="all"):
        for t in regions:
            if per.get(t) is not None:
                r = ref_level([t], refm, scope, n)
                add_prec(contrast, t, n, pool_of({t: per[t]}, [t]), r["point"])
        for name, reg in pools:
            P = pool_of({t: per.get(t) for t in reg}, reg)
            if len(P["avail"]) == len(reg):
                add_prec(contrast, name, n, P, ref_level(reg, refm, scope, n)["point"])

    # ---------------------------------------------------------------- LGU-B1(한계 확인, 비맹검)
    if not precision:
        b1 = {}
        for L in CQR_LEARNERS:
            m = f"cqr_pool:{L}@lam{lam1}"
            per = {t: (S.level(t, "all", [(m, s) for s in S.seeds(t, L)], "cov10") if (t in T and S.seeds(t, L)) else None) for t in MAIN4}
            b1[L] = pool_of(per, MAIN4)
        vals = [b1[L]["point"][w] for L in CQR_LEARNERS for w in (0, 1)]
        complete = all(len(b1[L]["avail"]) == len(MAIN4) for L in CQR_LEARNERS)
        if not complete or not all(np.isfinite(vals)):
            vb1 = "판정 불가"
        elif all(v < 0.85 for v in vals):
            vb1 = "지지"
        else:
            vb1 = "값 보고(H15 서술 수정)"
        st1 = ("원천에서 셀 교환성으로만 보정한 생성 분위 구간은 새 지역에서 목표 커버리지에 못 미친다" if vb1 == "지지"
               else "값을 그대로 보고하고 H15 서술을 고친다" if vb1.startswith("값") else "판정할 수 없다(행 없음 또는 지역 부족)")
        for L in CQR_LEARNERS:
            rows.append(test_row(a, "LGU-B1", "한계 확인(비맹검)", f"cov10 수준(대비 아님): cqr_pool:{L}@lam{lam1}", "MEAN4", MAIN4, 0, "cov10",
                                 b1[L], blind="재현(비맹검)", statement=st1, verdict=vb1, verdict_1="", level=True,
                                 coverage_pool=b1[L]["point"][0], cov_ok=str(bool(np.isfinite(b1[L]["point"][0]) and b1[L]["point"][0] < 0.85))))

    # ---------------------------------------------------------------- LGU-B2(확인적, 주 대비)
    per2, cov2, seeds_n = {}, {}, {}
    for t in MAIN4:
        sd = S.seeds(t, "nflow") if t in T else []
        seeds_n[t] = len(sd)
        if not sd:
            per2[t] = cov2[t] = None; continue
        A = [("nflow", s) for s in sd]
        B = [(f"nflow#p{j}", s) for s in sd for j in range(S.perms(t))]
        per2[t] = S.contrast(t, "all", A, B, "is10")
        cov2[t] = S.level(t, "all", A, "cov10", nboot=0)
    P4, P3 = pool_of(per2, MAIN4), pool_of({t: per2.get(t) for t in POOL3}, POOL3)
    if precision:
        for t in MAIN4:
            if per2.get(t) is not None:
                sd = S.seeds(t, "nflow")
                pr = pool_of({t: S.level(t, "all", [(f"nflow#p{j}", s) for s in sd for j in range(S.perms(t))], "is10", nboot=0)}, [t])
                add_prec("LGU-B2 is10: nflow − 위약(nflow)", t, 0, pool_of({t: per2[t]}, [t]), pr["point"])
        for name, reg in (("MEAN4", MAIN4), ("MEAN3", POOL3)):
            P = pool_of({t: per2.get(t) for t in reg}, reg)
            if len(P["avail"]) == len(reg):
                pr = pool_of({t: S.level(t, "all", [(f"nflow#p{j}", s) for s in S.seeds(t, "nflow") for j in range(S.perms(t))], "is10",
                                         nboot=0) for t in reg}, reg)
                add_prec("LGU-B2 is10: nflow − 위약(nflow)", name, 0, P, pr["point"])
    else:
        C4, C3 = pool_of(cov2, MAIN4), pool_of({t: cov2.get(t) for t in POOL3}, POOL3)
        # 조건 (4): 채점된 풀 지역마다 nflow 유효 seed 2개 이상(채점 조각이 없는 지역은 적합 실패가 아니라 풀 부족으로 센다)
        ok4 = all(seeds_n.get(t, 0) >= 2 for t in MAIN4 if t in T); ok3 = all(seeds_n.get(t, 0) >= 2 for t in POOL3 if t in T)
        pts4 = [per2[t]["point"][0] for t in P4["avail"]]; pts3 = [per2[t]["point"][0] for t in P3["avail"]]
        v4 = b2_verdict(ci_pair(P4["d2"]), pts4, C4["point"][0], ok4, len(P4["avail"]), need_neg=3, n_min=3)
        v4_1 = b2_verdict(ci_pair(P4["d1"]), pts4, C4["point"][0], ok4, len(P4["avail"]), need_neg=3, n_min=3)
        v3 = b2_verdict(ci_pair(P3["d2"]), pts3, C3["point"][0], ok3, len(P3["avail"]), need_neg=2, n_min=2)
        pair4 = all(bool(per2[t].get("pair_ok", True)) for t in P4["avail"])   # 1단 짝지음 점검(개정 1): 실패하면 계산 실패로 둔다
        if not pair4:
            v4 = "판정 불가(계산 실패)"
        if not all(bool(per2[t].get("pair_ok", True)) for t in P3["avail"]):
            v3 = "판정 불가(계산 실패)"
        refp = pool_of({t: S.level(t, "all", [(f"nflow#p{j}", s) for s in S.seeds(t, "nflow") for j in range(S.perms(t))], "is10", nboot=0)
                        if t in T and S.seeds(t, "nflow") else None for t in MAIN4}, MAIN4)
        st2 = ("nflow 분위 폭으로 정규화한 계층 conformal 구간은 위약 정규화 구간보다 구간 점수(α 0.1)가 낮다. 조건부 폭이 지역 간에 전이된다"
               "(주 4지역, 라벨 0)" if v4 == "전이" else f"'조건부 폭이 지역 간에 전이된다'는 쓰지 않는다(판정 {v4})")
        rows.append(test_row(a, "LGU-B2", "확인적(주 대비)", "is10: nflow − 위약(nflow, 순열 평균, seed 짝지음)", "MEAN4", MAIN4, 0, "is10", P4,
                             ref=refp["point"], statement=st2, verdict=v4, verdict_1=v4_1, coverage_pool=C4["point"][0],
                             cov_ok=str(bool(np.isfinite(C4["point"][0]) and C4["point"][0] >= LC.COV_MIN)), n_neg_regions=int(sum(p_ < 0 for p_ in pts4)),
                             seeds_ok=str(ok4), verdict_3region=v3, note_rw="러시아 W 의존" if _cls(v3) != _cls(v4) else "",
                             cov_beq_pool=C4["point"][1], pair_ok=str(pair4)))

    # ---------------------------------------------------------------- LGU-B3(짝 비교), LGU-B4(보조)
    def paired(t, mA, mB):
        cs = sorted(set(S.seeds(t, mA)) & set(S.seeds(t, mB))) if t in T else []
        return [(mA, s) for s in cs], [(mB, s) for s in cs]

    per3 = {}
    for t in MAIN4:
        A, B = paired(t, "nflow", "cbq")
        per3[t] = S.contrast(t, "all", A, B, "is10") if A else None
    if precision:
        prec_all("LGU-B3 is10: nflow − cbq", per3, MAIN4, [("MEAN4", MAIN4), ("MEAN3", POOL3)], 0, "cbq")
    else:
        P4b, P3b = pool_of(per3, MAIN4), pool_of({t: per3.get(t) for t in POOL3}, POOL3)
        ref3, ref33 = ref_level(MAIN4, "cbq"), ref_level(POOL3, "cbq")
        r = test_row(a, "LGU-B3", "짝 비교", "is10: nflow − cbq(seed 짝지음)", "MEAN4", MAIN4, 0, "is10", P4b, ref=ref3["point"])
        r3 = test_row(a, "LGU-B3", "짝 비교", "", "MEAN3", POOL3, 0, "is10", P3b, ref=ref33["point"])
        r["verdict_3region"] = r3["verdict"]; r["note_rw"] = "러시아 W 의존" if _cls(r3["verdict"]) != _cls(r["verdict"]) else ""
        covn = pool_of({t: S.level(t, "all", [("nflow", s) for s in S.seeds(t, "nflow")], "cov10", nboot=0) if t in T else None for t in MAIN4}, MAIN4)
        r["coverage_pool"] = covn["point"][0]
        v = r["verdict"]
        r["statement"] = ("nflow 와 CatBoost 다분위 폭 정규화의 구간 점수 차는 기준 점수의 ±5 % 한계 안이다(주 4지역, 라벨 0)" if v == "동등"
                          else "nflow 폭 정규화의 구간 점수가 CatBoost 다분위 폭 정규화보다 낮다" if v == "우세"
                          else "nflow 폭 정규화의 구간 점수가 CatBoost 다분위 폭 정규화보다 높다" if v == "열세"
                          else f"두 정규화기의 구간 점수 차는 결정되지 않았다(동등 상태: {r['eq_state']})")
        rows.append(r)

    b4 = []
    for nm in B4_NORMS:
        for cond in ("", "@mw"):
            meth = nm + cond
            per = {t: (S.contrast(t, "all", [(meth, s) for s in S.seeds(t, nm)], [("const", -1)], "is10") if (t in T and S.seeds(t, nm)) else None)
                   for t in MAIN4}
            lab = f"LGU-B4 is10: {meth} − const"
            if precision:
                prec_all(lab, per, MAIN4, [("MEAN4", MAIN4), ("MEAN3", POOL3)], 0, "const"); continue
            P4c, P3c = pool_of(per, MAIN4), pool_of({t: per.get(t) for t in POOL3}, POOL3)
            r = test_row(a, "LGU-B4", "보조", f"is10: {meth} − const" + ("(평균 로그 폭 정합)" if cond else "(보정 그대로)"), "MEAN4", MAIN4, 0,
                         "is10", P4c, ref=ref_level(MAIN4, "const")["point"])
            r3 = test_row(a, "LGU-B4", "보조", "", "MEAN3", POOL3, 0, "is10", P3c, ref=ref_level(POOL3, "const")["point"])
            r["verdict_3region"] = r3["verdict"]; r["note_rw"] = "러시아 W 의존" if _cls(r3["verdict"]) != _cls(r["verdict"]) else ""
            covm = pool_of({t: S.level(t, "all", [(meth, s) for s in S.seeds(t, nm)], "cov10", nboot=0) if (t in T and S.seeds(t, nm)) else None
                            for t in MAIN4}, MAIN4)
            r["coverage_pool"] = covm["point"][0]
            r["cov_ok"] = str(bool(np.isfinite(covm["point"][0]) and covm["point"][0] >= LC.COV_MIN))
            r["statement"] = f"{meth} − const: {r['verdict']}" + ("(큰 틀 U2 의 원래 판정량)" if meth == "cbq" else "")
            b4.append(r)
    if b4:
        hp = LC.holm([r["p_boot"] for r in b4])
        for r, h in zip(b4, hp):
            r["holm_p"] = h
        rows += b4

    if not precision:
        rows.append(dict(test="LGU-B5", role="진단", contrast="기기·시기 층, σ 5분위, 최근접 원천 셀 거리 3분위의 커버리지와 폭", pool="대상별",
                         statement=f"서술만 한다({a.PREFIX}_strata.csv)", blind="맹검", nboot=int(a.nboot)))

    # ---------------------------------------------------------------- LGU-B6(보조 n > 0)
    b6 = []
    for n in a.N_AUX:
        for nm in ("nflow", "cbq"):
            per = {t: (S.contrast(t, "aux", [(nm, s) for s in S.seeds(t, nm)], [("const", -1)], "is10", n)
                       if (t in T and t in a.AUX_EFF and S.seeds(t, nm)) else None) for t in POOL_AUX}
            lab = f"LGU-B6 is10: {nm} − const"
            if precision:
                prec_all(lab, per, POOL_AUX, [("MEAN_AUX3", POOL_AUX), ("MEAN_AUX2", POOL_AUX2)], n, "const", scope="aux"); continue
            P3a, P2a = pool_of(per, POOL_AUX), pool_of({t: per.get(t) for t in POOL_AUX2}, POOL_AUX2)
            exp_ = min([expected_splits(a, D, t) for t in P3a["avail"]] or [0])
            r = test_row(a, "LGU-B6", "보조", f"is10: {nm} − const(범위 B, 중심 E_n)", "MEAN_AUX3", POOL_AUX, n, "is10", P3a,
                         ref=ref_level(POOL_AUX, "const", "aux", n)["point"], splits_expected=exp_)
            r2 = test_row(a, "LGU-B6", "보조", "", "MEAN_AUX2", POOL_AUX2, n, "is10", P2a, ref=ref_level(POOL_AUX2, "const", "aux", n)["point"])
            r["verdict_2region"] = r2["verdict"]; r["note_alaska"] = "알래스카 의존" if _cls(r2["verdict"]) != _cls(r["verdict"]) else ""
            if r["splits_min"] < exp_:
                r["statement"] = f"분할 {r['splits_min']}/{exp_}(기대보다 적다)"
            b6.append(r)
    if b6:
        hp = LC.holm([r["p_boot"] for r in b6])
        for r, h in zip(b6, hp):
            r["holm_p"] = h
        rows += b6
    return rows, prec


PREC_COLS = ("contrast", "pool", "n", "weight", "half_width_rel", "main_rule_5pct", "nboot", "ci_kind")


def precision_rows(entries, nboot) -> list:
    """정밀도 표(계획서 6절: 반폭만): 대비별 95 % CI 반폭의 기준 점수 대비 비율(half_width_rel)과 본 실행 정밀도 규칙(2.7절, 주 한계 5 %)의
    상태만 쓴다. delta, CI 끝점, 판정, 기준 점수와 반폭의 절댓값은 쓰지 않는다(개정 1: 둘을 함께 쓰면 기준 점수가 역산된다).
    실험 B 의 대비는 모두 구간 점수(상대 한계)다."""
    out = []
    for e in entries:
        for wi, w in enumerate(("cell", "beq")):
            d = e["d2"][wi] if e.get("d2") is not None else None
            if d is not None:
                lo, hi, _ = LC.ci95(d)
                hw = (hi - lo) / 2.0
            else:
                hw = np.nan
            ref = float(e["ref"][wi]) if e.get("ref") is not None else np.nan
            rel = hw / abs(ref) if (np.isfinite(ref) and ref != 0 and np.isfinite(hw)) else np.nan
            out.append(dict(contrast=e["contrast"], pool=e["pool"], n=int(e["n"]), weight=w, half_width_rel=rel,
                            main_rule_5pct=("판정 불가" if not np.isfinite(rel) else "검정 불가(정밀도 미달)" if rel > LC.EQ_REL else "검정 가능"),
                            nboot=int(nboot), ci_kind=e.get("kind", "2단")))
    return out


# ================================================================ 집계: 표
def _mgroup(m) -> str:
    """위약 순열 방법('nflow#p3')을 한 행('nflow#placebo')으로 묶는다."""
    head, sep, tail = str(m).rpartition("#p")
    return f"{head}#placebo" if sep and tail.isdigit() else str(m)


def _fill_metric(rec, mt, L):
    nmv = "rmse" if mt == "se" else mt
    pt = L["point"] if L is not None else (np.nan, np.nan)
    rec[nmv], rec[f"{nmv}_beq"] = pt
    lo1, hi1, lob1, hib1 = ci_pair(L["d1"] if L is not None else None)
    rec[f"{nmv}_lo1"], rec[f"{nmv}_hi1"], rec[f"{nmv}_lo1_beq"], rec[f"{nmv}_hi1_beq"] = lo1, hi1, lob1, hib1
    if mt in LC.METRICS_2STAGE:
        lo2, hi2, lob2, hib2 = ci_pair(L["d2"] if L is not None else None)
        rec[f"{nmv}_lo2"], rec[f"{nmv}_hi2"], rec[f"{nmv}_lo2_beq"], rec[f"{nmv}_hi2_beq"] = lo2, hi2, lob2, hib2


def interval_rows(S) -> list:
    """대상·범위·방법 묶음·변형·n 별 지표(두 가중), 1단·2단 CI. 층(#…) 키는 뺀다(strata 표). MEAN4·MEAN3(범위 all, n = 0)과
    MEAN_AUX3(범위 B) 행을 더한다."""
    rows, keep = [], {}
    scope_name = {"all": "all", "noearly": "all~noearly", "aux": "B"}
    for t in S.targets:
        for scope in ("all", "noearly", "aux"):
            sb = S.sb(t, scope)
            groups = {}
            for st in sb.values():
                for k in st.keys:
                    m, v, n, d, s = k
                    if str(v).startswith("#"):
                        continue
                    groups.setdefault((_mgroup(m), str(v), int(n)), set()).add((str(m), int(s)))
            for (mg, v, n), pairs in sorted(groups.items()):
                pairs = sorted(pairs)
                rec = dict(target=t, scope=scope_name[scope], method=mg, variant=v, n=n)
                vals = {}
                for mt in LC.METRICS:
                    L = S.level(t, scope, pairs, mt, n)
                    vals[mt] = L
                    _fill_metric(rec, mt, L)
                L0 = vals.get("cov10")
                rec.update(in_band=float(0.85 <= rec["cov10"] <= 0.95) if np.isfinite(rec["cov10"]) else np.nan,
                           n_seeds_valid=len({s for _, s in pairs}), n_keys=L0["n_keys"] if L0 else 0, n_splits=L0["n_splits"] if L0 else 0,
                           n_blocks=L0["nb"] if L0 else 0, n_eval=int(np.mean([st.ncell.sum() for st in sb.values()])) if sb else 0,
                           ci_kind="2단+1단" if (L0 and L0["d2"] is not None) else "1단", flag="")
                rows.append(rec)
                keep.setdefault((scope, mg, v, n), {})[t] = vals
    for (scope, mg, v, n), per_t in sorted(keep.items()):
        pools = [("MEAN4", MAIN4), ("MEAN3", POOL3)] if (scope == "all" and n == 0) else [("MEAN_AUX3", POOL_AUX)] if scope == "aux" else []
        for name, reg in pools:
            rec = dict(target=name, scope=scope_name[scope], method=mg, variant=v, n=n)
            P0 = None
            for mt in LC.METRICS:
                P = pool_of({t: per_t.get(t, {}).get(mt) for t in reg}, reg)
                if mt == "cov10":
                    P0 = P
                _fill_metric(rec, mt, dict(point=P["point"], d1=P["d1"], d2=P["d2"]) if P["avail"] else None)
            k = len(P0["avail"]) if P0 else 0
            rec.update(in_band=float(0.85 <= rec["cov10"] <= 0.95) if np.isfinite(rec["cov10"]) else np.nan, n_seeds_valid=np.nan,
                       n_keys=np.nan, n_splits=P0["n_splits_min"] if P0 else 0, n_blocks=np.nan, n_eval=np.nan,
                       ci_kind="2단+1단" if (P0 and P0["d2"] is not None) else "1단", regions=",".join(P0["avail"]) if P0 else "",
                       flag="" if k == len(reg) else f"부분(지역 {k}/{len(reg)})")
            rows.append(rec)
    return rows


def cal_rows(C, lo, hi) -> list:
    """집단별 정규화 점수 분포, q90 초과 비율, σ 분포(계획서 4.5절). 방법 const, const@h37, 정규화기(seed 별)."""
    M = C["meta"]; t = M["target"]; g = np.asarray(C["c_region"]).astype(str)
    au = np.abs(np.asarray(C["c_u"], float)); one = np.ones(len(au))
    items = [("const", -1, au, one), ("const@h37", -1, np.asarray(C["c_h37"], float), one)]
    for nm in [v for v in FIT_NORMS if v in M["norms"]]:
        for j, s in enumerate(M["seeds"][nm]):
            sc = np.asarray(C[f"sig_c__{nm}"][j], float)
            items.append((nm, key_seed(nm, s), au / sc, sc))
    out = []
    for m, ks, s_, sig in items:
        q90 = float(LC.hier_quantiles(s_, g, [0.9])[0])
        for k in M["G"]:
            mk = g == k
            sk, gk = s_[mk], sig[mk]
            if not mk.any():
                continue
            ps = np.nanpercentile(sk, [5, 25, 50, 75, 90, 95]); qs = np.nanpercentile(gk, [5, 50, 95])
            out.append(dict(target=t, method=m, seed=ks, region=k, n_cal=int(mk.sum()), s_p05=ps[0], s_p25=ps[1], s_p50=ps[2], s_p75=ps[3],
                            s_p90=ps[4], s_p95=ps[5], q90=q90, exceed90=float(np.mean(sk > q90)), sigma_p05=qs[0], sigma_p50=qs[1],
                            sigma_p95=qs[2], clip_frac=float(np.mean((gk <= lo + 1e-12) | (gk >= hi - 1e-12))) if m in FIT_NORMS else 0.0))
    return out


def cells_frame(C) -> pd.DataFrame:
    """채점 셀의 σ(seed 평균)와 90 % 구간 끝점(seed 평균). 폭 지도의 입력(계획서 4.5절)."""
    M = C["meta"]; g = np.asarray(C["c_region"]).astype(str); au = np.abs(np.asarray(C["c_u"], float))
    center = np.asarray(C["e_center"], float)
    df = pd.DataFrame(dict(target=M["target"], loc_id=C["e_loc"], lat=C["e_lat"], lon=C["e_lon"], center=center, d_src_km=C["e_dsrc"]))
    q = float(LC.hier_quantiles(au, g, [0.9])[0])
    df["sigma_const"] = 1.0; df["lo90_const"] = center * np.exp(-q); df["hi90_const"] = center * np.exp(q)
    for nm in [v for v in FIT_NORMS if v in M["norms"]]:
        sg, lo_, hi_ = [], [], []
        for j in range(len(M["seeds"][nm])):
            sc = np.asarray(C[f"sig_c__{nm}"][j], float); stt = np.asarray(C[f"sig_t__{nm}"][j], float)
            qj = float(LC.hier_quantiles(au / sc, g, [0.9])[0])
            sg.append(stt); lo_.append(center * np.exp(-qj * stt)); hi_.append(center * np.exp(qj * stt))
        df[f"sigma_{nm}"] = np.mean(sg, 0); df[f"lo90_{nm}"] = np.mean(lo_, 0); df[f"hi90_{nm}"] = np.mean(hi_, 0)
    return df


def strata_rows(S) -> list:
    """층(#…) 키의 커버리지와 폭(seed 평균, 1단 CI). 계획서 4.5절, LGU-B5."""
    out = []
    for t in S.targets:
        sb = S.sb(t, "all")
        if not sb:
            continue
        st = sb[0]
        groups = {}
        for k in st.keys:
            m, v, n, d, s = k
            if str(v).startswith("#"):
                groups.setdefault((str(m), str(v)), set()).add((str(m), int(s)))
        for (m, v), pairs in sorted(groups.items()):
            ks = [LC.norm_key((mm, v, 0, 0, s)) for mm, s in sorted(pairs)]
            _, cnt = st.get(ks[0])
            rec = dict(target=t, method=m, stratum=v, n_seeds=len(pairs), n_cells=int(cnt.sum()), n_blocks=int((cnt > 0).sum()))
            for mt in STRATA_METRICS:
                L = level1(sb, ks, mt, S.a.nboot if mt in ("cov10", "is10") else 0, Summ.boot_seed(t, "all"))
                rec[mt], rec[f"{mt}_beq"] = L["point"] if L else (np.nan, np.nan)
                if mt in ("cov10", "is10"):
                    lo1, hi1, lob1, hib1 = ci_pair(L["dist"] if L else None)
                    rec[f"{mt}_lo1"], rec[f"{mt}_hi1"], rec[f"{mt}_lo1_beq"], rec[f"{mt}_hi1_beq"] = lo1, hi1, lob1, hib1
            out.append(rec)
    return out


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ts_job(path, W, chunk):
    """2단 분포 계산 작업(대상 하나). 반환 (대상, 분포 dict, 초)."""
    t0 = time.time()
    C = load_cells(path)
    out = twostage_from_specs(C, method_specs(C), W, chunk)
    return C["meta"]["target"], out, time.time() - t0


def run_twostage(a, cells_paths, W):
    """대상별 2단 분포. --workers > 1 이면 대상 단위 프로세스 풀(spawn). 반환 (T2, secs, failed).
    대상 하나의 예외는 그 대상만 실패로 기록하고(failed 에 (대상, 사유)) 나머지 대상의 분포를 만든다(개정 1). 풀이 깨지면(BrokenProcessPool)
    남은 대상을 이 프로세스에서 차례로 다시 계산한다. 2단 분포가 없는 대상은 판정 표에서 2단 CI 가 없는 것으로 처리된다."""
    T2, secs, failed = {}, {}, []
    items = list(cells_paths.items())

    def seq(its):
        for t, p_ in its:
            try:
                t_, out, sec = _ts_job(str(p_), W, BOOT_CHUNK)
            except Exception as e:                                           # noqa: BLE001  한 대상의 실패가 나머지를 막지 않는다
                if LC.is_env_error(e):
                    raise
                failed.append((t, f"{type(e).__name__}: {e}"[:MAX_TXT]))
                print(f"  [2단 FAIL] {t}: {repr(e)[:MAX_TXT]}", flush=True)
                continue
            T2[t] = out; secs[t] = sec
            print(f"  [2단] {t}: 분포 {len(out)}개 · {sec:.0f}s", flush=True)

    if int(a.workers) > 1 and len(items) > 1:
        ctx = multiprocessing.get_context("spawn")
        broken = []
        with ProcessPoolExecutor(max_workers=min(int(a.workers), len(items)), mp_context=ctx) as ex:
            futs = {ex.submit(_ts_job, str(p_), W, BOOT_CHUNK): (t, p_) for t, p_ in items}
            for f in as_completed(futs):
                t, p_ = futs[f]
                try:
                    t_, out, sec = f.result()
                except BrokenProcessPool:
                    broken.append((t, p_)); continue
                except (Exception, SystemExit) as e:                        # noqa: BLE001
                    failed.append((t, f"{type(e).__name__}: {e}"[:MAX_TXT]))
                    print(f"  [2단 FAIL] {t}: {repr(e)[:MAX_TXT]}", flush=True)
                    continue
                T2[t] = out; secs[t] = sec
                print(f"  [2단] {t}: 분포 {len(out)}개 · {sec:.0f}s", flush=True)
        if broken:
            print(f"[pool] 2단 재표집 풀이 깨졌다. 남은 대상 {[t for t, _ in broken]} 를 이 프로세스에서 차례로 계산한다", flush=True)
            seq(broken)
    else:
        seq(items)
    return T2, secs, failed


def save_boot(a, t, out):
    keys = sorted(out, key=lambda k: json.dumps(k, default=_js))
    arr = np.full((len(keys), len(LC.METRICS_2STAGE), 2, int(a.nboot)), np.nan, np.float32)
    for i, k in enumerate(keys):
        for j, mt in enumerate(LC.METRICS_2STAGE):
            for w in (0, 1):
                arr[i, j, w] = out[k][mt][w]
    return LC.atomic_npz(a.OUT / f"{a.PREFIX}_boot__{t}.npz", keys=np.array([json.dumps(k, default=_js) for k in keys], dtype=str),
                         metrics=np.array(LC.METRICS_2STAGE, dtype=str), dist=arr)


def summarize(a) -> int:
    """집계(계획서 4.4–4.5절). 반환 종료 코드(단위 실패 또는 score 조각 없음 = 1)."""
    t0 = time.time()
    D = get_D(a); bi = get_bi(D); check_blockindex(a, bi)
    a.OUT.mkdir(parents=True, exist_ok=True)
    failed, timing, ok_targets = [], [], []
    for t in a.TARGETS:
        for nm in a.NORMS:
            p = fit_paths(a, t, nm); u = _read_json(p["unit"])
            ok, why = fit_state(a, t, nm)
            if u:
                timing.append(dict(part="fit", target=t, normalizer=nm, status=u.get("status"), elapsed_s=u.get("elapsed_s"), n_fit=u.get("n_fit"),
                                   n_fit_failed=u.get("n_fit_failed"), device=u.get("device"), threads=u.get("threads")))
            if not ok:
                failed.append(dict(kind="fit_unit", target=t, normalizer=nm, reason=why + (f"; {u.get('error')}" if u and u.get("error") else "")))
            if p["runs"].exists():
                fr = pd.read_csv(p["runs"])
                if "failed" in fr:
                    for _, r in fr[fr.failed.astype(bool)].iterrows():
                        failed.append(dict(kind="fit", target=t, normalizer=nm, train_set=r.train_set, seed=r.seed, reason=r.fail_reason))
        p = score_paths(a, t); u = _read_json(p["unit"]); cfg = score_cfg(a, t)
        if u:
            timing.append(dict(part="score", target=t, normalizer="", status=u.get("status"), elapsed_s=u.get("elapsed_s"), threads=u.get("threads")))
            if u.get("status") != "failed" and u.get("cfg_common") not in (None, cfg_common_hash(cfg)):
                raise SystemExit(f"[summary] {t} 의 score 조각 설정이 현재 설정과 다르다(설정 해시가 섞여 있다). 집계를 중단한다")
            for nm, why in (u.get("missing") or {}).items():
                failed.append(dict(kind="score_missing", target=t, normalizer=nm, reason=why))
        ok, why = LC.unit_state(p["unit"], [p["runs"], p["scores"], p["cells"]], cfg)
        if ok:
            ok_targets.append(t)
        elif why.startswith("설정 불일치"):
            raise SystemExit(f"[summary] {t}: score 조각이 현재 fit 조각·설정과 맞지 않는다(설정 해시가 섞여 있다). --part score --resume 으로 다시 채점한다")
        else:
            failed.append(dict(kind="score_unit", target=t, normalizer="", reason=why + (f"; {u.get('error')}" if u and u.get("error") else "")))
    n_unit_fail = sum(1 for f in failed if f["kind"] in ("fit_unit", "score_unit"))
    pre = a.OUT / a.PREFIX
    LC.atomic_text(Path(f"{pre}_failed.csv"), pd.DataFrame(failed, columns=["kind", "target", "normalizer", "train_set", "seed", "reason"]).to_csv(index=False))
    meta = dict(stage="H45/LGU-B", plan="docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 4절", prefix=a.PREFIX, tag=a.TAG, targets=a.TARGETS,
                ok_targets=ok_targets, normalizers=a.NORMS, seeds=a.SEEDS, perms=a.PERMS, lams=a.LAMS, n_aux=a.N_AUX, aux_targets=a.AUX_EFF,
                splits=a.SPLITS, draws=a.DRAWS, nboot=int(a.nboot), precision_only=bool(a.precision_only), n_unit_fail=n_unit_fail,
                blind={"LGU-B1": "재현(비맹검)", "LGU-B2": "맹검", "LGU-B3": "맹검", "LGU-B4": "맹검(const 는 C2 hier2_cdf 와 규약이 다르다)",
                       "LGU-B5": "진단", "LGU-B6": "맹검"},
                inputs=input_shas(a), **code_shas())
    if not ok_targets:
        LC.atomic_text(Path(f"{pre}_timing.csv"), pd.DataFrame(timing).to_csv(index=False))
        LC.atomic_text(Path(f"{pre}_meta.json"), _dump(dict(meta, elapsed_s=round(time.time() - t0, 1))))
        print(f"[summary] 채점 조각이 없다. 실패 {len(failed)}건 → {pre}_failed.csv", flush=True)
        return 1
    stores = LC.load_score_stores([score_paths(a, t)["scores"] for t in ok_targets])
    cpaths = {t: score_paths(a, t)["cells"] for t in ok_targets}
    W = LC.global_mult(bi, int(a.nboot), 0)
    print(f"[summary] 대상 {ok_targets} · 재표집 {a.nboot} · 전역 블록 {bi.n_cols}", flush=True)
    T2, secs, tfail = run_twostage(a, cpaths, W)
    for t, sec in secs.items():
        timing.append(dict(part="summary_2stage", target=t, normalizer="", status="ok", elapsed_s=round(sec, 1)))
    for t, why in tfail:
        timing.append(dict(part="summary_2stage", target=t, normalizer="", status="failed", elapsed_s=np.nan))
        failed.append(dict(kind="twostage", target=t, normalizer="", reason=why))
    n_unit_fail += len(tfail)
    LC.atomic_text(Path(f"{pre}_failed.csv"), pd.DataFrame(failed, columns=["kind", "target", "normalizer", "train_set", "seed", "reason"]).to_csv(index=False))
    cells = {t: load_cells(p_) for t, p_ in cpaths.items()}
    n_nan = {}
    for t, out in T2.items():
        save_boot(a, t, out)
        k0 = ("all", LC.norm_key(("const", "", 0, 0, -1)))
        n_nan[t] = int(np.isnan(out[k0]["is10"][0]).sum()) if k0 in out else -1
    S = Summ(a, stores, cells, T2)
    if a.precision_only:
        _, prec = build_tests(S, D, precision=True)
        LC.atomic_text(Path(f"{pre}_precision.csv"), pd.DataFrame(precision_rows(prec, a.nboot)).to_csv(index=False))
        LC.atomic_text(Path(f"{pre}_timing.csv"), pd.DataFrame(timing).to_csv(index=False))
        LC.atomic_text(Path(f"{pre}_meta.json"), _dump(dict(meta, n_nan_boot=n_nan, elapsed_s=round(time.time() - t0, 1))))
        print(f"[summary] 정밀도 표만 썼다(대비의 부호와 점 추정은 쓰지 않는다) → {pre}_precision.csv · {time.time() - t0:.0f}s", flush=True)
        return 1 if n_unit_fail else 0
    irows = interval_rows(S)
    LC.atomic_text(Path(f"{pre}_intervals.csv"), pd.DataFrame(irows).to_csv(index=False))
    trows, _ = build_tests(S, D)
    tdf = pd.DataFrame(trows)
    for c in TEST_COLS:
        if c not in tdf:
            tdf[c] = np.nan
    tdf = tdf[TEST_COLS + [c for c in tdf.columns if c not in TEST_COLS]]
    LC.atomic_text(Path(f"{pre}_tests.csv"), tdf.to_csv(index=False))
    LC.atomic_text(Path(f"{pre}_cal.csv"), pd.DataFrame([r for t in cells for r in cal_rows(cells[t], a.sigma_lo, a.sigma_hi)]).to_csv(index=False))
    LC.atomic_text(Path(f"{pre}_strata.csv"), pd.DataFrame(strata_rows(S)).to_csv(index=False))
    pit = []
    for t in ok_targets:
        rr = pd.read_csv(score_paths(a, t)["runs"])
        rr = rr[(rr.scope == "all") & (rr.flag.fillna("") != "interval_only")]
        hc = [f"pit_h{i}" for i in range(LC.PIT_BINS)]
        for _, r in rr.iterrows():
            pit.append(dict(target=t, method=r.method, variant=r.variant if isinstance(r.variant, str) else "", seed=int(r.seed),
                            n=int(sum(int(r[c]) for c in hc)), **{c: int(r[c]) for c in hc}))
    LC.atomic_text(Path(f"{pre}_pit.csv"), pd.DataFrame(pit).to_csv(index=False))
    LC.atomic_text(Path(f"{pre}_cells.csv"), pd.concat([cells_frame(cells[t]) for t in ok_targets], ignore_index=True).to_csv(index=False))
    LC.atomic_text(Path(f"{pre}_timing.csv"), pd.DataFrame(timing).to_csv(index=False))
    LC.atomic_text(Path(f"{pre}_meta.json"), _dump(dict(meta, n_nan_boot=n_nan, n_rows_intervals=len(irows), n_rows_tests=len(trows),
                                                        elapsed_s=round(time.time() - t0, 1))))
    show = tdf[tdf.test.isin(["LGU-B1", "LGU-B2", "LGU-B3"])][["test", "pool", "n", "verdict", "pool_regions"]]
    print(show.to_string(index=False), flush=True)
    print(f"[summary] 표 {pre}_*.csv · 실패 {len(failed)}건(단위 {n_unit_fail}) · {time.time() - t0:.0f}s", flush=True)
    return 1 if n_unit_fail else 0


# ================================================================ 적합 수(--count-only, 학습 없음)
def count_keys(a, t, info) -> dict:
    """score 단위의 저장 키 수(모든 seed 가 유효하다고 가정)."""
    S = {nm: (len(seeds_of(a, nm)) if nm in a.NORMS else 0) for nm in FIT_NORMS}
    do_noak = t != ALASKA and ALASKA in info["G"]
    nall = 2 + int(do_noak) + 2 * sum(S.values()) + sum(S[nm] * a.PERMS for nm in PLACEBO_NORMS)
    nall += sum(S[nm] for nm in SENS_NORMS if nm in S) * int(do_noak) + sum(S[nm] * len(a.LAMS) * 2 for nm in CQR_LEARNERS)
    nne = 1 + sum(S[nm] for nm in SENS_NORMS if nm in S)
    nst = 11 * (1 + sum(S[nm] for nm in STRATA_NORMS if nm in S))
    return dict(all=int(nall), noearly=int(nne), strata=int(nst), per_aux_draw=int(1 + S["nflow"] + S["cbq"]))


def count_only(a, D) -> pd.DataFrame:
    """작업 단위 수, 적합 수, 키 수, 시간 추정(설계 단계 기준값의 환산이며 실측이 아니다)."""
    get_flags(a, D)                                                          # 표지 표의 누락을 여기서 확인한다(없으면 중단)
    per_fit = dict(nflow=30.0, cfm=3.4, cbq=0.3)                             # 학습 행 16,700 기준 1건의 초(LG 사전 점검 qOkSo)
    rows = []
    for t in a.TARGETS:
        info = target_info(a, D, t)
        sets = train_sets(D, info)
        ntr = [len(tr) for _, tr, _, _ in sets]
        K = len(info["G"])
        for nm in a.NORMS:
            S = len(seeds_of(a, nm))
            est = 0.0 if nm == "phys" else S * sum(per_fit[nm] * n / 16700.0 for n in ntr)
            rows.append(dict(part="fit", target=t, normalizer=nm, K_groups=K, groups=",".join(info["G"]), n_train_sets=K + 1, n_seeds=S,
                             n_fit=0 if nm == "phys" else (K + 1) * S, n_train_full=ntr[-1], n_cal=int(len(info["cal_idx"])),
                             n_target=int(len(info["t_idx"])), n_eval=int(len(info["ev"])), n_drop_eval=info["n_drop_eval"],
                             est_fit_s=round(est, 1), device="GPU" if (nm in GEN_NORMS and a.GPUS) else "CPU",
                             note="적합 시간 추정(표본 추출·분위 계산 제외, 실측 아님)"))
        nk = count_keys(a, t, info)
        n_split = n_nd = n_rec = 0
        if t in a.AUX_EFF and a.N_AUX:
            ss = D.split_structure(t)
            any_valid = any(v["valid"] for v in ss.values())
            for sp in a.SPLITS:
                v = ss.get(sp)
                if v is None or v["dup_of"] >= 0 or v["n_eval"] == 0 or v["n_A"] == 0 or (not v["valid"] and any_valid):
                    continue
                n_split += 1
                n_nd += len(H.cells_of(a.N_AUX, a.DRAWS, v["n_A"]))
            n_rec = len(LC.emulation_plan(D.df, info["src_idx"], info["G"], a.N_AUX, t, "x"))
        rows.append(dict(part="score", target=t, normalizer="", K_groups=K, groups=",".join(info["G"]), n_eval=int(len(info["ev"])),
                         n_cal=int(len(info["cal_idx"])), n_keys_all=nk["all"], n_keys_noearly=nk["noearly"], n_keys_strata=nk["strata"],
                         n_aux_splits=n_split, n_aux_nd=n_nd, n_keys_aux=n_nd * nk["per_aux_draw"], n_emu_records=n_rec,
                         aux=bool(t in a.AUX_EFF)))
    df = pd.DataFrame(rows)
    a.OUT.mkdir(parents=True, exist_ok=True)
    LC.atomic_text(a.OUT / f"{a.PREFIX}_count.csv", df.to_csv(index=False))
    f = df[df.part == "fit"]
    tot = {nm: int(f[f.normalizer == nm].n_fit.sum()) for nm in a.NORMS}
    print(f"[count] 대상 {len(a.TARGETS)} · fit 단위 {len(f)} · score 단위 {int((df.part == 'score').sum())} · 적합 {tot} · "
          f"추정 적합 시간 {f.est_fit_s.sum() / 3600:.2f} h(추정, 실측 아님) → {a.OUT / (a.PREFIX + '_count.csv')}", flush=True)
    show = df[["part", "target", "normalizer", "K_groups", "n_fit", "n_eval", "n_cal"]]
    print(show.to_string(index=False), flush=True)
    return df


# ================================================================ 워커와 실행
_WA = None


def _worker_init(argv, gpu_queue, threads, torch_on):
    global _WA
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                          # 0 MiB 확인(nvidia-smi 번호)과 같은 번호 체계
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get()) if gpu_queue is not None else ""   # torch 를 부르기 전에 정한다
    LC.set_thread_env(threads)
    _WA = parse_args(argv)
    if torch_on:
        try:
            LC.set_torch_threads(threads)
        except Exception:                                                   # noqa: BLE001
            pass


def _unit_summary(u, wall) -> dict:
    return dict(part=u.get("part"), target=u.get("target"), normalizer=u.get("normalizer", ""), status=u.get("status"),
                elapsed_s=u.get("elapsed_s"), wall_s=round(wall, 1), n_fit=u.get("n_fit", 0), n_fail=u.get("n_fit_failed", 0),
                device=os.environ.get("CUDA_VISIBLE_DEVICES", ""), missing=u.get("missing", {}))


def _worker_run(part, target, nm=None):
    t0 = time.time()
    u = run_fit_unit(_WA, target, nm) if part == "fit" else run_score_unit(_WA, target)
    return _unit_summary(u, time.time() - t0)


def run_pool(a, argv, units, nproc, gpus, torch_on, log, fail):
    """프로세스 풀(spawn). gpus 가 있으면 워커마다 GPU 하나(대기열). 풀이 깨지면 남은 단위로 --pool-retries 번 다시 만든다."""
    ctx = multiprocessing.get_context("spawn")
    remaining, attempt = list(units), 0
    while remaining:
        q = None
        if gpus:
            q = ctx.Queue()
            for g in gpus:
                for _ in range(max(1, int(a.procs_per_gpu))):
                    q.put(g)
        broken = []
        with ProcessPoolExecutor(max_workers=max(1, int(nproc)), mp_context=ctx, initializer=_worker_init,
                                 initargs=(argv, q, int(a.threads), bool(torch_on))) as ex:
            futs = {ex.submit(_worker_run, *u): u for u in remaining}
            for f in as_completed(futs):
                try:
                    log(f.result())
                except BrokenProcessPool:
                    broken.append(futs[f])
                except (Exception, SystemExit) as e:                        # noqa: BLE001  한 단위의 실패(SystemExit 포함)가 나머지를 막지 않는다
                    fail(futs[f], e)
        if not broken:
            break
        attempt += 1
        if attempt > int(a.pool_retries):
            for u in broken:
                fail(u, RuntimeError(f"프로세스 풀이 {attempt}회 깨졌다(재시도 상한 {a.pool_retries})"))
            break
        remaining = broken
        print(f"[pool] 워커 비정상 종료. 남은 {len(remaining)} 단위로 풀을 다시 만든다({attempt}/{a.pool_retries})", flush=True)


def fit_pending(a, t) -> list:
    """채점 전에 끝나 있어야 하는 fit 단위 가운데 조각이 없거나 설정이 다른 것. status 'failed' 로 끝난 단위는 끝난 것으로 본다
    (채점은 그 정규화기를 빼고 unit.json 의 missing 에 적는다)."""
    out = []
    for nm in a.NORMS:
        ok, why = fit_state(a, t, nm)
        if not ok and not why.startswith("이전 실행 실패"):
            out.append(f"{nm}: {why}")
    return out


def run_part(a, argv, part, t0):
    """한 부분(fit 또는 score)의 작업 단위를 실행한다. 반환 (완료 목록, 실패 목록)."""
    units = [("fit", t, nm) for t in a.TARGETS for nm in a.NORMS] if part == "fit" else [("score", t, None) for t in a.TARGETS]
    todo = []
    for u in units:
        ok, why = ((fit_state(a, u[1], u[2]) if part == "fit" else score_state(a, u[1])) if a.resume else (False, ""))
        if ok:
            print(f"  [resume] 건너뜀 {part}|{u[1]}|{u[2] or ''} (상태 {why})", flush=True)
        else:
            todo.append(u)
            if a.resume and not why.startswith("조각 없음"):
                print(f"  [resume] 다시 실행 {part}|{u[1]}|{u[2] or ''} ({why})", flush=True)
    done, failed = [], []
    if part == "score":                                                     # 그 대상의 fit 단위가 모두 끝나야 채점한다
        ready = []
        for u in todo:
            pend = fit_pending(a, u[1])
            if pend:
                failed.append(u)
                print(f"  [skip] score|{u[1]}: 끝나지 않았거나 설정이 다른 fit 조각 {pend}. 채점하지 않는다", flush=True)
            else:
                ready.append(u)
        todo = ready

    def log(r):
        done.append(r)
        print(f"  [{r['part']}|{r['target']}|{r['normalizer'] or '-'}] 상태 {r['status']} · 적합 {r['n_fit']}(실패 {r['n_fail']}) · "
              f"{r['elapsed_s']}s · 장치 '{r['device']}'" + (f" · 없는 정규화기 {sorted(r['missing'])}" if r.get("missing") else "")
              + f" · 완료 {len(done)}/{len(todo)} · 누적 {time.time() - t0:.0f}s", flush=True)

    def fail(u, e):
        failed.append(u)
        print(f"  [FAIL] {u[0]}|{u[1]}|{u[2] or ''}: {repr(e)[:MAX_TXT]}", flush=True)

    if not todo:
        return done, failed
    gen_units = [u for u in todo if part == "fit" and u[2] in GEN_NORMS]
    if int(a.workers) <= 0:                                                 # 풀 없이 이 프로세스에서 차례로
        global _WA
        gp = usable_gpus(a)[:1] if (gen_units and a.GPUS) else []
        os.environ["CUDA_VISIBLE_DEVICES"] = gp[0] if gp else ""
        _WA = a
        if gen_units:
            LC.set_torch_threads(a.threads)
        for u in todo:
            t1 = time.time()
            try:
                uu = run_fit_unit(a, u[1], u[2]) if part == "fit" else run_score_unit(a, u[1])
                log(_unit_summary(uu, time.time() - t1))
            except (Exception, SystemExit) as e:                            # noqa: BLE001  unit.json(failed)은 run_*_unit 이 이미 썼다
                fail(u, e)
        return done, failed
    cpu_u = [u for u in todo if u not in gen_units] if a.GPUS else list(todo)
    gpu_u = gen_units if a.GPUS else []
    if cpu_u:
        run_pool(a, argv, cpu_u, max(1, int(a.workers)), None, any(u[2] in GEN_NORMS for u in cpu_u), log, fail)
    if gpu_u:
        gp = usable_gpus(a)                                                 # 쓰기 직전의 0 MiB 확인
        run_pool(a, argv, gpu_u, len(gp) * max(1, int(a.procs_per_gpu)), gp, True, log, fail)
    return done, failed


def main(argv=None) -> int:
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    permitted = LC.require_permission(a)                                    # 자료를 읽기 전에 거부한다
    if not permitted:
        a.threads = 1
    for v in _THREAD_VARS:
        os.environ[v] = str(a.threads)                                     # 워커(spawn)가 물려받는다
    if a.GPUS and os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        print("[warn] 부모 환경의 CUDA_VISIBLE_DEVICES 가 빈 문자열이다. --gpus 를 무시하고 CPU 로 돈다", flush=True)
        a.GPUS = []
    if a.count_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        count_only(a, get_D(a))
        return 0
    parts = [] if a.summarize_only else (["fit"] if a.part == "fit" else ["score"] if a.part == "score" else ["fit", "score"])
    will_sum = bool(a.summarize_only or ("score" in parts and not a.no_summarize))
    tot = local_guard(a, parts, will_sum)                                   # LG_RESCALE=1 에도 적용한다(개정 1)
    reg = LC.claim_threads(a.TAG, max(1, tot), LOCAL_THREAD_CAP, script=Path(__file__).name)   # 하네스 사이의 스레드 합계(개정 1)
    try:
        return _run(a, argv, parts, will_sum, t0)
    finally:
        LC.release_threads(reg)


def _run(a, argv, parts, will_sum, t0) -> int:
    """허용된 실행의 본체(적합, 채점, 집계). main 이 스레드 등록을 잡고 끝에 푼다."""
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    D = get_D(a)
    for t in a.TARGETS:
        if t not in D.macros:
            raise SystemExit(f"[target] {t} 는 macro 지역이 아니다")
    check_blockindex(a, get_bi(D))
    print(f"[plan] 부분 {parts or ['summary']} · 대상 {a.TARGETS} · 정규화기 {a.NORMS} · seed {a.SEEDS} · 순열 {a.PERMS} · λ {a.LAMS} · "
          f"보조 {a.AUX_EFF} n {a.N_AUX} 분할 {a.SPLITS} 추출 {a.DRAWS} · epochs {a.epochs} · nboot {a.nboot} · 워커 {a.workers} × 스레드 {a.threads} · "
          f"GPU {a.GPUS or '없음'} · 태그 {a.TAG}", flush=True)
    n_fail = 0
    for part in parts:
        done, failed = run_part(a, argv, part, t0)
        n_fail += len(failed) + sum(1 for d in done if d["status"] == "failed")
        print(f"[done] {part}: 완료 {len(done)} · 실패 {len(failed)} · {time.time() - t0:.0f}s", flush=True)
    rc = 1 if n_fail else 0
    if will_sum:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        rc = max(rc, summarize(a))
    return rc


if __name__ == "__main__":
    sys.exit(main())
