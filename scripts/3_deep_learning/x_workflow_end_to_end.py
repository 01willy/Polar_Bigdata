"""XC_workflow_end_to_end(워크플로 순차 적용) 하네스. 계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100) 2.3절.

범위
  XC     전이(모드 x). 배치(Algorithm P) → 라벨 10개 편향 진단 → 라벨 수별 방법 규칙을 같은 셀의 새 분할 201–210 에 차례로 적용한다.
         대상: PE1(레나, 캐나다, 러시아 W, 러시아 E, Russia_C~lgd), Tibet_LGD, 알래스카 x(F1), 하위 지역 AL-1, AL-2, AL-3, AL-5, CA-2, CA-3, LE-1
         의 모드 i·x(F2). n ∈ {0, 10, 40, 160, 전량}(|A| 미만), 추출 d = 0–4, CatBoost seed 0·1.
  XC-r   지역 내(모드 r) 보조 팔. 알래스카, 레나, 캐나다, 분할 211–220, n {200, 500, 1,000}(|A| 미만), 추출 3, seed 0·1.
  XC-F3  외부 계열(부록 XC-F3 의 새 지역, 작업 R4). XC 와 같은 단위를 쓰고 조각 이름의 tag 만 다르다(<tag>f3).
  관문   재현 관문(2.3절): 추출만 h40.draw_cells 로 바꾼 분할 1 의 레나 x·캐나다 x 단위. 키는 WF4 조각과 같은 이름(placement 'cell')이다.

팔(2.3절 표). 저장 키 = (method, learner, alpha, placement, n, draw, seed, lam). placement 'rand' = 무작위 중첩 라벨, 'algP' = Algorithm P 라벨.
  A1  워크플로      algP. n 10 은 R1(λ 0.25), n 40·160·전량은 규칙 W(키 method 'W').
  A1+ 보조          algP. n 10 은 A1 과 같고, n 40·160·전량은 규칙 W+(키 method 'W+', XB 의 적층 후보를 더한 규칙).
  A2  주 기준       rand. R1(λ 0.25).
  A3  비열등 기준   rand. P1(κ 10 재보정 Stefan).
  A4  배치 효과     algP. R1(λ 0.25).
  A5  현재 권장     rand. 규칙 W(n 10 포함).
  A6  후회값 참고   rand. W 후보 가운데 그 (대상, n)에서 셀 가중 RMSE 평균이 가장 작은 고정 방법(집계 단계에서 고른다).
  단계 0            P0(모든 단위에 저장).
  전량(n = −1)은 두 배치가 같은 라벨(A 전체)이라 placement 'rand' 하나만 저장한다. Algorithm P = S1(부록 XC-0 규칙 5)이면 algP 라벨이
  무작위 라벨과 같으므로 algP 키를 따로 적합하지 않고 A1·A4·A1+ 는 rand 키를 쓴다. 두 팔이 같은 키가 되는 대비는 모두 '시험하지 않음'이다
  (XC-5b·5c·S-XC3 은 규칙 5 의 문구 그대로, XC-F3·S-XC0(n 10 에서 A1 = A2), S-XC8 의 같은 키 행, F2 의 같은 키 대비는 같은 이유. Holm p 1).

라벨 집합
  무작위: XB.draw_nested('xc-rand', 대상, 모드, 분할, n, 추출)(중첩 순열 RNG(seed_of('xc-rand', 대상, 모드, 분할, d)).permutation(|A|)의 앞 n 개).
  Algorithm P: 부록 XC-0 선택 파일이 정한 후보(S2, S4, S8a, S8b, S8c, S8p 또는 규칙 5 의 S1)의 순서 π 를 seed_of('xc-P', 대상, 모드, 분할, d)
  로 만들고 앞 n 개를 쓴다(누적 예산 일정 = 등록 격자 가운데 |A| 미만 n). 대상 이름은 h54 와 같이 표 안의 이름(c.target)이다.
  적합에 넣는 라벨 집합은 색인을 정렬한 것이다(같은 집합이면 같은 적합).

배치 구현: x_placement_policy(XD 모듈)가 xc_placement_order(name, Z, blk, schedule, seed) 를 제공하면 그것을 쓰고, 없으면 이 모듈의 참조 구현
(2.4절 의사코드의 문자 그대로 판)을 쓴다. 어느 쪽을 썼는지 설정 해시와 unit.json 에 적는다(--placement-impl, --placement-check).
W+ 의 적층 후보는 x_multisource_stacking(XB 모듈)의 xc_wplus_cv·xc_wplus_final 이 있을 때만 계산한다(--wplus). 계약은 WPlusProvider 참조.

통계(1절, 2.3절)
  라벨 집합이 다른 대비(A1 − A2, A1 − A3, A1 − A5, A4 − A2, A1 − A6)는 2단 재표집(XB.region_stat_two_stage: 주 2단, 보조 2단 공통, 추출 조건부),
  같은 라벨 집합 대비(A5 − A2, A1+ − A1, A2 − A6, A1 − P0, 전량 행)는 h40.contrast(XB.region_stat_same). 층화 평균은 XB.pool(h42.pool_rows).
  주 가설 Holm 가족 m = 8: XC-1b, XC-1c(A1 − A2 우월, n 40·160), XC-2a·b·c(A1 − A3 비열등 0.5 cm, n 10·40·160), XC-5b, XC-5c(A1 − A5 우월),
  XC-F3(A1 − A2, n 10, F3). 기준 = 2단 CI 와 2단 공통 CI 의 상한이 두 가중 모두 0(우월) 또는 0.5(비열등) 미만이고 Holm 보정 p < 0.05.
  XC-2s: XC-2 가 기준을 만족한 n 에서만 우월을 시험한다(우월 양측 p × XC-2 의 그 n 의 Holm 보정 배수 < 0.05).
  보조 S-XC0–S-XC12 와 XC-r 의 4분 판정은 확인적으로 세지 않는다.

출력 제한(계획 0.3): 스모크·세기·로컬 시험·집계의 화면에는 RMSE·Δ·판정을 쓰지 않는다. 집계 표는 모두 봉인 폴더
data/processed/xbatch/XC_workflow_end_to_end/sealed/ 에 쓰고 화면에는 행 수와 sha256 만 쓴다(재현 관문 통과 전 판정 표 열람 금지, 0.3 의 5).
조각의 runs.csv 에는 RMSE·편향 열을 쓰지 않고(집계는 blocksse.npz 만 쓴다) 규칙 W 의 교차검증 RMSE 는 봉인 폴더 unit_cv/ 에 둔다.
스모크는 평가에 쓰지 않는 분할(XC 1 = 관문 분할, XC-r 1 = 지역 내 1–25 가운데 하나)을 쓰고 조각을 봉인 폴더 shards_smoke/ 에 쓴다.
부록 XC-F3 전(--f3-status pending, xcf3 조각 없음)의 집계는 XC-F3 을 '미정'으로 두고 Holm 의존 표(hyp, holm, sentences)에 _provisional 을 붙인다.
재표집 수가 등록값(10,000)과 다른 스모크 밖 집계는 파일 이름에 _nboot<N> 을 붙이고 meta 에 등록 이탈을 적는다.
해석과 구현 결정은 docs/research/2026-10-04/impl_notes/x_workflow_end_to_end.md 에 적었다.

명령행(로컬은 세기·스모크·시험·배치 점검만. 본 실행은 Rescale, WF_RESCALE=1)
  세기:   CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/x_workflow_end_to_end.py --count-only --threads 1
  스모크: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c <코어 4개> python3 scripts/3_deep_learning/x_workflow_end_to_end.py
          --smoke --threads 2 --workers 0 2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'
          (코어 묶음이 4개를 넘거나 OMP_NUM_THREADS 가 2 를 넘으면 거부한다. 스레드는 2, nice 는 10 으로 맞춘다)
  배치 점검(R2a 제출 전 통과 조건, 실제 A 풀, 라벨 미사용):
          python3 ... --placement-check --xc0 <부록 XC-0 선택 파일> --placement-impl xd
          (XD 함수와 참조 구현의 순서 대조, 분할 201–210 의 대상·추출마다 Algorithm P 순서를 두 번 계산한 결정성)
  관문:   WF_RESCALE=1 python3 ... --gate --workers 22 --threads 4   →   python3 ... --gate-check results/<작업>/.../wf/shards
          --xc0 <같은 파일> [--r2a-shards <R2a 조각 폴더>](WF4 블록 SSE 대조 + Algorithm P 결정성과 R2a 조각 orders_head 대조)
  본 실행: WF_RESCALE=1 python3 ... --xc0 <부록 XC-0 선택 파일> --placement-impl xd --wplus require --workers 22 --threads 4 --resume
          --no-summarize(본 실행·집계는 --placement-impl xd, --wplus require 로 강제한다. ref·off 를 주면 거부한다)
  집계:   WF_RESCALE=1 python3 ... --summarize-only --xc0 <같은 파일> --placement-impl xd --wplus require --f3-status pending --workers 0 --threads 4
          (R2a 시점. 부록 XC-F3 이 0곳으로 확정되면 --f3-status none, R4 의 xcf3 조각이 있으면 auto 또는 final)
  F3:     WF_RESCALE=1 python3 ... --part xcf3 --f3 <별칭>=<실행 표 디렉터리> --xc0 <같은 파일> --placement-impl xd --wplus require ...
  조각 나눔: --shard i/k(작업 단위 목록을 k 묶음으로 나눈 i 번째, 0 부터) 또는 --shard <조각 기본 이름>.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import xbatch_core as XB                                                   # noqa: E402  numpy 보다 먼저(스레드 환경 변수)

import argparse                                                            # noqa: E402
import importlib                                                           # noqa: E402
import importlib.util                                                      # noqa: E402
import json                                                                # noqa: E402
import re                                                                  # noqa: E402
import time                                                                # noqa: E402
from collections import Counter                                            # noqa: E402

import numpy as np                                                         # noqa: E402
import pandas as pd                                                        # noqa: E402

H, X, W, H4, MS = XB.H, XB.X, XB.W, XB.H4, XB.MS
seed_of = XB.seed_of
ROOT = XB.ROOT
LO = XB.LO

# ================================================================ 1. 고정값(계획 2.3절. 바꾸면 사전 등록에서 벗어난다)
EXP_ID = "XC"
EXP_NAME = XB.EXP_NAMES[EXP_ID]                                            # XC_workflow_end_to_end
OUT_DEFAULT = XB.out_dir(EXP_ID)
XC_GRID = (10, 40, 160, -1)                                                # n 0 은 P0(모든 단위에 저장)
XC_DRAWS = 5
XCR_GRID = (200, 500, 1000)
XCR_DRAWS = 3
GATE_GRID = tuple(W.WF4_GRID)                                              # WF4 와 같은 격자(10, 40, 160, 전량)
GATE_DRAWS = int(W.WF4_DRAWS)
GATE_SPLIT = 1
GATE_TARGETS = (("Lena", "x"), ("Canada", "x"))
GATE_REF_DEFAULT = ROOT / "results" / "rescale_wf" / "data" / "processed" / "wf" / "shards"
GATE_REF_TAG = "wf4"
RAND_TAG, P_TAG = "xc-rand", "xc-P"
PL_RAND, PL_P, PL_GATE = "rand", "algP", "cell"
A1_R1_N = (10,)                                                            # A1 의 n 10 은 R1(λ 0.25)(단계 1)
DIAG_N = int(W.DIAG_N)                                                     # 10
W_CANDS = tuple(W.W_CANDS)                                                 # P0, P1, P2, R1@0.25, R1@1.0, R2@0.25, D1(동률 순서)
W_MIN_N = int(W.W_MIN_N)
WPLUS_EXTRA = ("Stack", "StackR@0.25")                                     # W+ = W 후보 + 적층 후보(동률은 W 후보 다음)
R_PS = XB.R_PS
LAM_BASE, LAM_CV = XB.LAM_BASE, XB.LAM_CV

F1 = (("Lena", "x"), ("Canada", "x"), ("Russia_W", "x"), ("Russia_E", "x"), ("Russia_C~lgd", "x"), ("Tibet_LGD", "x"), ("Alaska", "x"))
F2_SUBS = ("AL-1", "AL-2", "AL-3", "AL-5", "CA-2", "CA-3", "LE-1")         # AL-4, AL-6, LE-2 는 새 분할 0개(부록 A)
F2 = tuple((t, m) for t in F2_SUBS for m in ("i", "x"))
XC_TARGETS_DEFAULT = F1 + F2
XCR_TARGETS_DEFAULT = ("Alaska", "Lena", "Canada")
PE1, PE2, AUX3, MAIN4 = list(XB.PE1), list(XB.PE2), list(XB.AUX3), list(XB.MAIN4)
INREGION3 = list(XB.INREGION3)
INDEP7 = PE1 + ["Alaska|x", "Tibet_LGD|x"]                                 # S-XC4 (b), XC-2 덧붙임 문장의 독립 지역 7곳
F2_NAMES = [f"{t}|{m}" for t, m in F2]
SMOKE_TARGETS = (("Russia_W", "x"), ("Canada", "x"))
SMOKE_XCR = ("Canada",)
# 스모크 분할: 평가 분할(XC 201–210, XC-r 211–220)을 쓰지 않는다. XC 는 관문과 같은 분할 1(WF4·LG 가 평가에 쓴 분할), XC-r 은 지역 내
# 분할 1–25(WF6·WF9 가 쓴 분할) 가운데 1. 스모크 조각은 봉인 폴더 아래 shards_smoke/ 에 쓴다(계획 0.3 '걸러지지 않은 출력은 봉인 폴더')
SMOKE_SPLIT = {"xc": 1, "xcr": 1}
SMOKE_SHARDS = "shards_smoke"
LABEL_RESULT_COLS = ("rmse_cm", "rmse_beq_cm", "bias_cm")                 # 조각 runs.csv 에서 뺀다(집계는 blocksse.npz 만 쓴다)
UNIT_CV_DIR = "unit_cv"                                                     # 규칙 W 교차검증 RMSE(notes.w_cv_rmse)의 봉인 기록
SMOKE_MAX_THREADS = 2
LOCAL_NICE = 10
MEM_MIN_GB, MEM_WAIT_S, MEM_POLL_S = 30.0, 1800.0, 60.0                    # 1절 로컬 자원: 가용 30 GB 아래면 기다린다(최대 30분)
SEL_FAMILY_TXT = "선택 계열"                                                # Algorithm P 선택에 쓴 알래스카 계열(알래스카, AL-k)

# 부록 XC-F3 상태(계획 0.3 의 2단계 등록, 2.6절 마감). R2a 집계 시점에는 정해지지 않았다
F3_STATUS = ("auto", "pending", "none", "final")
F3_PENDING_TXT = "미정(부록 XC-F3 전)"
F3_NONE_TXT = "시험하지 못함(F3 적격 0곳)"
PROVISIONAL_TABLES = ("hyp", "holm", "sentences")                          # Holm 순위·보정 p 에 의존하는 표
# 같은 키가 되는 대비(Algorithm P = S1, 부록 XC-0 규칙 5)
RULE5_TXT = "시험하지 않음(Algorithm P = S1, 부록 XC-0 규칙 5)"
AUX2S_ARMS = ("A1", "A1+", "A2", "A3", "A4", "A5", "A6")                   # 같은 라벨 집합 대비에도 2단 CI 를 보조 열로 내는 XC 팔(1절 '적용')

# Algorithm P(2.3절 고정 규칙, 2.4절 정의)
ALGP_CANDIDATES = ("S2", "S4", "S8a", "S8b", "S8c", "S8p")
ALGP_NAMES = ALGP_CANDIDATES + ("S1",)                                     # S1 = 규칙 5(모든 후보 제외)
P_DEFAULT = "S8a"                                                          # 규칙 6(48 시간 안에 XD-alg 결과가 없을 때)
ALGP_TIE_ORDER = ("S8a", "S8c", "S8p", "S8b", "S2", "S4")                 # 규칙 3 의 동률 순서(선택은 XD 의 선택 스크립트가 한다)
S8_PARAMS = {"S8a": dict(gamma=0.0, first="center"), "S8b": dict(gamma=0.0, first="ff"),
             "S8c": dict(gamma=0.5, first="center"), "S8p": dict(gamma=1.0, first="center")}
XC0_DEFAULT = XB.out_dir("XD") / "selection" / "algorithm_p_selection.json"   # XD 선택 스크립트(select_algorithm_p)의 산출
XC0_RULE = "계획 2.3 Algorithm P 의 고정 규칙"                                 # XD select_algorithm_p 가 'rule' 키에 쓰는 값(규칙 1–5)
XC0_DEFAULT_MARK = "P_default"                                              # XD select_default(규칙 6)의 'reason' 에 들어가는 표지
XD_MODULE, XD_FNS = "x_placement_policy", ("xc_placement_order", "placement_order")   # (name, Z, blk, schedule, seed) → 순서
XS_MODULE, XS_CV, XS_FINAL, XS_PREP = "x_multisource_stacking", "xc_wplus_cv", "xc_wplus_final", "attach_cands_t"

# 주 가설(Holm 가족 m = 8, 등록 순서)
HOLM_M = 8
HYPS = (("XC-1b", "A1", "A2", 40, "superior"), ("XC-1c", "A1", "A2", 160, "superior"),
        ("XC-2a", "A1", "A3", 10, "noninf"), ("XC-2b", "A1", "A3", 40, "noninf"), ("XC-2c", "A1", "A3", 160, "noninf"),
        ("XC-5b", "A1", "A5", 40, "superior"), ("XC-5c", "A1", "A5", 160, "superior"),
        ("XC-F3", "A1", "A2", 10, "superior"))
MDE_HYPS = ("XC-1b", "XC-1c", "XC-5b", "XC-5c")                           # 미결정 문장에 최소 검출 효과를 붙이는 가설(2.3절 해석 표·검정력 표)
MDE_FACTOR, Z975 = 2.80, 1.96                                              # MDE ≈ 2.80 × SE, SE ≈ (2단 CI 반폭)/1.96
BLIND_MAIN = "비맹검 부분 포함(WF2·WF4·WF8, LGX-N4)"
BLIND_F3 = "맹검(실행 전 등록 확인 시험)"
DESIGN_MAIN = "결과 열람 뒤 설계, 재사용 지역의 재검정"
RESPLIT = "같은 지역을 다시 무작위로 나눈 분할에서"
COMMON_SENTENCE = ("Algorithm P 는 알래스카 계열에서 골랐고, 평가 지역의 셀은 구성 요소(배치, 규칙 W) 설계에 쓰였다. "
                   "평가 분할은 같은 셀을 새로 나눈 것이다.")
HARM_CM = 2.0                                                              # S-XC4 (a): P0 보다 2 cm 넘게 나빠짐(WF0 위험표와 같은 정의)
WORSE_THAN_A3_CM = 0.5                                                     # S-XC4 (b): A3 보다 0.5 cm 넘게 나쁨(셀 가중 점 추정)
DIAG_THRESHOLD_CM = 8.0                                                    # S-XC6 사후 임계값(WF4-c 관찰)
FLOOR_FILE = ROOT / "data" / "processed" / "lgx" / "lgx_floor.csv"         # S-XC11 (b)
FLOOR_REGIONS = {"Alaska|x": "Alaska", "Lena|x": "Lena", "Canada|x": "Canada"}
LABEQ_TARGETS = ("Lena|x", "Canada|x", "Alaska|x", "Tibet_LGD|x")          # S-XC11 (a): 격자 점 3개 이상인 대상
# 세기 범주의 시간 추정(적합 1건 실측 초, 4스레드 워커): 전이 = WF4(results/rescale_wf/.../wf_timing.csv: R1 0.222, R2 0.99, D1 1.01),
# 지역 내 = WF6(results/rescale_wf2/.../wf2b_timing.csv: R1 0.06, R2 0.14, D1 은 R2 와 같은 행 구성이라 같은 값으로 둔다)
MEAS_SPF = {("x", "R1"): 0.222, ("x", "R2"): 0.99, ("x", "D1"): 1.01, ("r", "R1"): 0.06, ("r", "R2"): 0.14, ("r", "D1"): 0.14}
RESCALE_WORKERS = 22

# 해석 문장(2.3절 표). '...' 는 우세 갈래의 앞부분으로 채웠다. {k} = 라벨 수, {a} = |Δ|(셀 가중, cm), {m} = 지역 수.
_XC1_HEAD = (f"배치 알고리즘, 라벨 10개 편향 진단, 라벨 수별 방법 규칙을 차례로 적용한 워크플로는 {RESPLIT} ")
SENTENCES = {
    "XC-1": {"우세": _XC1_HEAD + "무작위 배치와 고정 잔차 레시피보다 라벨 {k}개에서 오차가 {a} cm(셀 가중) 작았다(레나·캐나다 2지역).",
             "열세": _XC1_HEAD + "무작위 배치와 고정 잔차 레시피보다 라벨 {k}개에서 오차가 {a} cm 컸다.",
             "동등": _XC1_HEAD + "라벨 {k}개에서 무작위 배치와 고정 잔차 레시피와 0.5 cm 안에서 같았다.",
             "미결정": _XC1_HEAD + "라벨 {k}개에서 무작위 배치와 고정 잔차 레시피와의 차이를 확인하지 못했다.",
             "판정 불가": _XC1_HEAD + "라벨 {k}개에서 무작위 배치와 고정 잔차 레시피와의 차이를 판정할 수 없었다."},
    "XC-2": {"우세": f"{RESPLIT} 워크플로의 오차 증가는 재보정 Stefan 대비 0.5 cm 안이었다(비열등, 라벨 {{k}}개).",
             "열세": f"{RESPLIT} 워크플로는 라벨 {{k}}개에서 재보정 Stefan 보다 오차가 {{a}} cm 컸다.",
             "동등": f"{RESPLIT} 워크플로는 라벨 {{k}}개에서 재보정 Stefan 과 0.5 cm 안에서 같았다.",
             "미결정": f"{RESPLIT} 라벨 {{k}}개에서 재보정 Stefan 대비 비열등을 확인하지 못했다.",
             "판정 불가": f"{RESPLIT} 라벨 {{k}}개에서 재보정 Stefan 대비 비열등을 판정할 수 없었다."},
    "XC-2s": "재보정 Stefan 보다 오차가 {a} cm 작았다.",
    "XC-5": {"우세": f"{RESPLIT} 라벨 수별 방법 규칙을 쓰는 같은 조건에서 배치 알고리즘은 무작위 배치보다 라벨 {{k}}개에서 오차가 {{a}} cm 작았다.",
             "열세": f"{RESPLIT} 라벨 수별 방법 규칙을 쓰는 같은 조건에서 배치 알고리즘은 무작위 배치보다 라벨 {{k}}개에서 오차가 {{a}} cm 컸다.",
             "동등": f"{RESPLIT} 라벨 수별 방법 규칙을 쓰는 같은 조건에서 배치 알고리즘과 무작위 배치는 라벨 {{k}}개에서 0.5 cm 안에서 같았다.",
             "미결정": f"{RESPLIT} 라벨 수별 방법 규칙을 쓰는 같은 조건에서 배치 알고리즘과 무작위 배치의 차이를 확인하지 못했다(라벨 {{k}}개).",
             "판정 불가": f"{RESPLIT} 라벨 수별 방법 규칙을 쓰는 같은 조건에서 배치 알고리즘과 무작위 배치의 차이를 판정할 수 없었다(라벨 {{k}}개)."},
    "XC-F3": {"우세": f"{RESPLIT} 이번 조사 뒤 확보한 독립 지역 {{m}}곳에서 워크플로는 무작위 배치와 고정 잔차 레시피보다 라벨 10개에서 오차가 {{a}} cm 작았다(독립 지역 확인).",
              "열세": f"{RESPLIT} 이번 조사 뒤 확보한 독립 지역 {{m}}곳에서 워크플로는 무작위 배치와 고정 잔차 레시피보다 라벨 10개에서 오차가 {{a}} cm 컸다.",
              "동등": f"{RESPLIT} 이번 조사 뒤 확보한 독립 지역 {{m}}곳에서 워크플로는 무작위 배치와 고정 잔차 레시피와 라벨 10개에서 0.5 cm 안에서 같았다.",
              "미결정": f"{RESPLIT} 이번 조사 뒤 확보한 독립 지역 {{m}}곳에서 워크플로와 무작위 배치·고정 잔차 레시피의 차이를 확인하지 못했다.",
              "판정 불가": f"{RESPLIT} 이번 조사 뒤 확보한 독립 지역 {{m}}곳에서 워크플로와 무작위 배치·고정 잔차 레시피의 차이를 판정할 수 없었다."},
    "XC-F3-none": "공개 자료로 확보한 새 독립 지역이 적격 조건을 만족하지 못해 외부 계열 시험은 하지 못했다.",
    "XC-2-indep": "독립 지역 7곳(PE1 5곳, 알래스카 x, Tibet_LGD) 가운데 워크플로가 재보정 Stefan 보다 0.5 cm 넘게 나빴던 지역은 {j}곳({lst})이었다.",
    "XC-2-indep-tail": "비열등은 풀 평균에 대한 것이며 지역 {lst} 에서는 성립하지 않았다.",
    "S-XC6": "사후 임계값(WF4-c 관찰) |b10| ≥ 8 cm",
}


# ================================================================ 2. Algorithm P 선택(부록 XC-0)
def algp_spec(name, source="", **kw) -> dict:
    """Algorithm P 의 정의(이름 → 매개변수). S1 은 규칙 5(무작위)."""
    if name not in ALGP_NAMES:
        raise ValueError(f"Algorithm P 후보는 {ALGP_NAMES} 가운데 하나다: {name}")
    d = dict(name=str(name), source=str(source), is_rand=(name == "S1"))
    d.update(S8_PARAMS.get(name, {}))
    d.update(kw)
    return d


def xc0_origin(raw) -> str:
    """선택 파일이 XD 선택 스크립트의 산출인지 확인한다. 'rule'(규칙 1–5, select_algorithm_p) 또는 'P_default'(규칙 6, select_default).
    알래스카 계열 표지(alaska_family)가 없거나 규칙 문구가 다르면 빈 문자열."""
    if raw.get("rule") == XC0_RULE and isinstance(raw.get("alaska_family"), dict):
        return "rule"
    if "rule" not in raw and raw.get("chosen") == P_DEFAULT and XC0_DEFAULT_MARK in str(raw.get("reason", "")):
        return "P_default"
    return ""


def load_xc0(path, require_sidecar=False, expect_sha256="") -> dict:
    """부록 XC-0 선택 파일(JSON, XD 의 select_algorithm_p 산출). 후보 이름 키는 'chosen'(XD), 'algorithm_p', 'selected' 가운데 있는 것.
    <파일>.sha256 사이드카가 있으면 대조하고 다르면 거부한다. require_sidecar(본 실행·집계)이면 사이드카가 없을 때도 거부하고, 파일이 XD 선택
    스크립트의 산출(xc0_origin)인지 확인한다. expect_sha256(부록 XC-0 개정 이력에 적은 값, 앞부분 16자 이상)이 있으면 파일 sha256 과 대조한다.
    파일 sha256 을 붙이고 그 밖의 스칼라 항목은 기록만 한다(근거 표는 읽지 않는다)."""
    p = Path(path)
    sha = XB.sha256_file(p)
    side = Path(str(p) + ".sha256")
    if side.exists() and side.read_text().split()[:1] != [sha]:
        raise SystemExit(f"[거부] {p.name} 의 sha256 이 사이드카와 다르다(부록 XC-0 선택 파일이 바뀌었다)")
    if require_sidecar and not side.exists():
        raise SystemExit(f"[거부] {p.name} 의 sha256 사이드카({side.name})가 없다. 본 실행·집계는 XD 선택 스크립트가 쓴 파일과 사이드카만 받는다")
    exp = str(expect_sha256 or "").strip().lower()
    if exp:
        if len(exp) < 16 or not sha.startswith(exp):
            raise SystemExit(f"[거부] {p.name} 의 sha256 앞부분이 부록 XC-0 에 적은 값과 다르다(--xc0-sha256)")
    raw = json.loads(p.read_text())
    name = next((raw[k] for k in ("chosen", "algorithm_p", "selected") if k in raw), None)
    if name is None:
        raise ValueError(f"{p}: 'chosen'(또는 'algorithm_p', 'selected') 키가 없다")
    origin = xc0_origin(raw)
    if require_sidecar and not origin:
        raise SystemExit(f"[거부] {p.name} 은 XD 선택 스크립트의 산출이 아니다('rule' = '{XC0_RULE}' 와 alaska_family, 또는 규칙 6 의 P_default 기록)")
    keep = {k: v for k, v in raw.items() if isinstance(v, (str, int, float, bool))}
    return algp_spec(str(name), source=str(raw.get("source", raw.get("reason", "부록 XC-0")))[:200], file=str(p), file_sha256=sha, xc0=keep,
                     xc0_origin=origin or "검사 안 함", xc0_sidecar=bool(side.exists()))


# ================================================================ 3. 배치(2.4절 S8 의사코드의 참조 구현, S2·S4 의 중첩 판)
def zstd(XA):
    """표준화 Z = (중앙값 대체 X − 평균) / (표준편차 + 1e-6), 후보 풀(A) 통계(h40._prep_stats, 라벨 미사용)."""
    return W.standardize(XA, W.standardize_fit(XA))


def power_alloc_ref(Nb, n, gamma, rank=None) -> np.ndarray:
    """q_b: w_b = N_b^γ, q_b = n·w_b/Σw 의 최대 나머지 배분, 상한 N_b, 소수부 동률은 β 순서(rank 작은 블록) 우선.
    상한에 걸린 블록은 N_b 로 고정하고 남은 예산을 나머지 블록에 같은 규칙으로 다시 나눈다(물 채우기). Σq = min(n, ΣN_b)."""
    N = np.asarray(Nb, int)
    K = len(N)
    rank = np.arange(K) if rank is None else np.asarray(rank, int)
    n = int(min(int(n), int(N.sum())))
    w = np.where(N > 0, N.astype(float) ** float(gamma), 0.0)
    q = np.zeros(K, int)
    capped = N <= 0
    while True:
        free = ~capped
        rem = n - int(q.sum())
        if rem <= 0 or not free.any():
            break
        share = rem * w[free] / w[free].sum()
        over = share >= N[free]
        if not over.any():
            break
        idx = np.where(free)[0][over]
        q[idx] = N[idx]
        capped[idx] = True
    free = np.where(~capped)[0]
    rem = n - int(q.sum())
    if rem > 0 and len(free):
        share = rem * w[free] / w[free].sum()
        base = np.minimum(np.floor(share + 1e-9).astype(int), N[free])
        q[free] = base
        frac = np.round(share - base, 12)                                   # 부동소수 잡음(예: 1/3 의 세 표현)을 동률로 본다(XD power_alloc 과 같다)
        order = sorted(range(len(free)), key=lambda j: (-frac[j], rank[free[j]]))
        left = rem - int(base.sum())
        for j in order:
            if left <= 0:
                break
            if q[free[j]] < N[free[j]]:
                q[free[j]] += 1
                left -= 1
    return q


def _block_codes(blk):
    ub, code = np.unique(np.asarray(blk).astype(str), return_inverse=True)
    return ub, code.astype(int)


def block_order_ref(mu, seed, first_block=None) -> np.ndarray:
    """블록 순서 β: 블록 중심 μ_b 에 farthest-first. 첫 블록은 seed 로 고른다(first_block 이 있으면 그 블록). 동률은 작은 블록 번호."""
    K = len(mu)
    j0 = int(first_block) if first_block is not None else int(np.random.RandomState(int(seed) % (2 ** 32)).randint(K))
    order = [j0]
    md = np.sqrt(((mu - mu[j0]) ** 2).sum(1))
    md[j0] = -1.0
    while len(order) < K:
        j = int(np.argmax(md))
        order.append(j)
        md = np.minimum(md, np.sqrt(((mu - mu[j]) ** 2).sum(1)))
        md[j] = -1.0
    return np.array(order, int)


def algorithm_p_order_ref(Z, blk, schedule, gamma=0.0, seed=0, first="center", init=None) -> np.ndarray:
    """2.4절 Algorithm P(S8) 의사코드의 문자 그대로 판. 반환 = 순서 있는 A 색인 π(길이 max(schedule) − |L0|). 단계 k 의 배치는 π[:n_k − |L0|].
    first = 'center'(블록에 라벨이 없을 때 블록 중심에 가장 가까운 셀, S8a·S8c·S8p) 또는 'ff'(블록 안 첫 셀도 farthest-first, S8b).
    S8b 의 맨 첫 셀(라벨이 하나도 없을 때)은 farthest-first 가 정의되지 않아 중심 근접 셀로 둔다. 동률은 작은 A 색인(블록 안), β 순서(블록)."""
    Z = np.asarray(Z, float)
    N = len(Z)
    ub, bc = _block_codes(blk)
    K = len(ub)
    Nb = np.bincount(bc, minlength=K)
    mu = np.vstack([Z[bc == k].mean(0) for k in range(K)])
    L0 = np.asarray(init if init is not None else [], int)
    beta = block_order_ref(mu, seed, first_block=int(bc[L0[0]]) if len(L0) else None)
    rank = np.empty(K, int)
    rank[beta] = np.arange(K)
    taken = np.zeros(N, bool)
    taken[L0] = True
    cnt = np.bincount(bc[L0], minlength=K) if len(L0) else np.zeros(K, int)
    mind = np.full(N, np.inf)
    for l_ in L0:
        mind = np.minimum(mind, np.sqrt(((Z - Z[l_]) ** 2).sum(1)))
    pi = []
    for nk in sorted(int(v) for v in schedule):
        nk = min(nk, N)
        q = power_alloc_ref(Nb, nk, gamma, rank)
        deficit = np.maximum(0, q - cnt)
        while len(pi) < nk - len(L0):
            avail = np.where(Nb - cnt > 0)[0]
            if not len(avail):
                break
            b = int(avail[np.lexsort((rank[avail], -deficit[avail]))[0]])
            cells = np.where((bc == b) & ~taken)[0]
            if cnt[b] == 0 and (first == "center" or (len(pi) + len(L0)) == 0):
                c_ = int(cells[np.argmin(np.sqrt(((Z[cells] - mu[b]) ** 2).sum(1)))])
            else:
                c_ = int(cells[np.argmax(mind[cells])])
            pi.append(c_)
            taken[c_] = True
            cnt[b] += 1
            deficit[b] -= 1
            mind = np.minimum(mind, np.sqrt(((Z - Z[c_]) ** 2).sum(1)))
    return np.array(pi, int)


def s2_order(blk, n_max, seed) -> np.ndarray:
    """S2(블록 순환 배정, h40.draw_blocks 와 같은 절차)의 중첩 판: seed 하나로 블록 순열과 블록 안 순열을 정하고 n_max 까지의 순서를 낸다.
    같은 seed 의 앞 n 개는 n 으로 돌린 결과와 같다(정렬 전)."""
    blk = np.asarray(blk)
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    ub = np.unique(blk)
    order = [int(j) for j in rng.permutation(len(ub))]
    pools = {j: list(rng.permutation(np.where(blk == ub[j])[0])) for j in range(len(ub))}
    sel, active = [], order
    while len(sel) < n_max and active:
        nxt = []
        for j in active:
            if len(sel) >= n_max:
                break
            if pools[j]:
                sel.append(int(pools[j].pop()))
                nxt.append(j)
        active = nxt
    return np.array(sel[:n_max], int)


def xd_placement(strict=False):
    """XD 모듈의 배치 함수(xc_placement_order 또는 placement_order, 인자 (name, Z, blk, schedule, seed)). 없으면 None.
    strict 이면 모듈을 불러오다 난 예외를 삼키지 않고 거부한다(본 실행·집계의 --placement-impl xd)."""
    try:
        if importlib.util.find_spec(XD_MODULE) is None:
            return None
        mod = importlib.import_module(XD_MODULE)
    except Exception as e:                                                # noqa: BLE001
        if strict:
            raise SystemExit(f"[거부] {XD_MODULE} 를 불러올 수 없다: {e!r}"[:300])
        return None
    for nm in XD_FNS:
        fn = getattr(mod, nm, None)
        if callable(fn):
            return mod, fn
    return None


def resolve_placement_impl(impl="auto"):
    """('xd' 또는 'ref', XD 모듈 sha1 앞 12자 또는 'ref'). auto 는 XD 함수가 있으면 xd(스모크·세기·시험 전용 기본값)."""
    xd = xd_placement(strict=(impl == "xd")) if impl in ("auto", "xd") else None
    if impl == "xd" and xd is None:
        raise SystemExit(f"[거부] --placement-impl xd 인데 {XD_MODULE} 의 배치 함수({', '.join(XD_FNS)})가 없다")
    if xd is not None:
        return "xd", XB.code_sha(xd[0].__file__)
    return "ref", "ref"


def placement_order(spec, Z, blk, schedule, seed, impl="ref") -> np.ndarray:
    """Algorithm P 의 순서(A 색인, 길이 ≤ max(schedule)). spec = algp_spec. impl = 'xd' 이면 XD 함수, 'ref' 이면 참조 구현."""
    name = spec["name"]
    sched = sorted(int(v) for v in schedule if int(v) > 0)
    if not sched:
        return np.zeros(0, int)
    n_max = min(max(sched), len(Z))
    if name == "S1":
        raise ValueError("S1 은 무작위 중첩 라벨을 쓴다(placement_order 를 부르지 않는다)")
    if impl == "xd":
        _, fn = xd_placement()
        return np.asarray(fn(name, Z, blk, sched, int(seed)), int)[:n_max]
    if name == "S2":
        return s2_order(blk, n_max, seed)
    if name == "S4":
        return W.kcenter_order(Z, n_max, seed)
    return algorithm_p_order_ref(Z, blk, sched, spec["gamma"], seed, spec["first"])[:n_max]


# ================================================================ 4. W+ 공급자(XB 의 적층 함수)
class WPlusProvider:
    """W+ 의 적층 후보 공급자(계획 2.3절 A1+, 7.2-1). XB 모듈(x_multisource_stacking)이 아래 두 함수를 제공해야 한다.
      xc_wplus_cv(unit, tr, te, n, d, fold, seed) → {'Stack': te 예측, 'StackR@0.25': te 예측}: 교차검증 묶음(tr 로 가중·계수·잔차 학습, te 예측)
      xc_wplus_final(unit, sel, n, d, seed) → {'Stack': B 예측, 'StackR@0.25': B 예측}: 선택 라벨 전체로 학습한 채점 셀 예측
    unit = XCUnit(h54.TUnit). 라벨은 unit.c.yA[tr](또는 [sel])만 쓰고, 학습기 적합은 unit.fit 으로 불러 세기와 추적을 받는다(누설 시험 (b)).
    prepare(a, c, alias, mode, split)(선택, XB 의 attach_cands_t)는 문맥에 후보 표를 붙인다. 세기 범주에서 라벨을 지우기 전에 부른다."""

    def __init__(self, cv, final, sha="none", name=XS_MODULE, prepare=None):
        self.cv, self.final, self.sha, self.name, self.prepare = cv, final, str(sha), str(name), prepare


def load_wplus(mode="auto"):
    """(공급자 또는 None, 상태 문자열). mode = off, auto, require."""
    if mode == "off":
        return None, "off"
    try:
        spec = importlib.util.find_spec(XS_MODULE)
        mod = importlib.import_module(XS_MODULE) if spec is not None else None
    except Exception as e:                                                # noqa: BLE001
        mod = None
        if mode == "require":
            raise SystemExit(f"[거부] {XS_MODULE} 를 불러올 수 없다: {e!r}"[:300])
    cv, fin = (getattr(mod, XS_CV, None), getattr(mod, XS_FINAL, None)) if mod is not None else (None, None)
    if callable(cv) and callable(fin):
        prep = getattr(mod, XS_PREP, None)
        return (WPlusProvider(cv, fin, XB.code_sha(mod.__file__), prepare=prep if callable(prep) else None),
                f"{XS_MODULE}@{XB.code_sha(mod.__file__)}")
    if mode == "require":
        raise SystemExit(f"[거부] --wplus require 인데 {XS_MODULE}.{XS_CV}·{XS_FINAL} 이 없다(W+ 코드는 R1 제출 전에 커밋한다)")
    return None, "absent"


def _pick(sse, cands):
    """교차검증 SSE 가 가장 작은 후보(동률은 후보 순서, h54.TUnit.select_w 와 같다)."""
    return min(cands, key=lambda m: (sse[m] if np.isfinite(sse[m]) else np.inf, cands.index(m)))


# ================================================================ 5. 작업 단위
class XCUnit(W.TUnit):
    """XC 단위(h54.TUnit 하위 클래스, 전이 모드). 문맥·원천 행·유사라벨 색인·교차검증 묶음은 TUnit 과 같다(rows_R, rows_D, ps, folds).
    gate = True 이면 무작위 팔의 추출이 h40.draw_cells 이고 placement 'cell' 이다(재현 관문: WF4 조각과 같은 키)."""

    def __init__(self, a, c, alias, algp, dry=False, gate=False, wplus=None, placement_impl="ref", exp="xc"):
        super().__init__(a, c, alias, dry, exp=exp, variant="")
        self.algp, self.gate, self.wplus = algp, bool(gate), wplus
        self.placement_impl = placement_impl
        self.pl_rand = PL_GATE if self.gate else PL_RAND
        self.grid = [int(v) for v in a.G[exp]]
        self.draws = int(getattr(a, "XC_DRAWS", XC_DRAWS))
        self._orders: dict = {}
        self._z = None
        self.notes.update(algp=algp["name"], algp_source=algp.get("source", ""), placement_impl=placement_impl,
                          wplus="on" if wplus is not None else "off", gate=self.gate)

    # ------------------------------------------------------------ 라벨 집합
    def schedule(self):
        return [n for n in self.grid if 0 < n < self.nA]

    def zstd(self):
        if self._z is None:
            self._z = zstd(self.c.XA)
        return self._z

    def rand_set(self, n, d):
        c = self.c
        if self.gate:
            return H.draw_cells(c.target, c.mode, c.split, n, d, self.nA)
        return np.sort(XB.draw_nested(RAND_TAG, c.target, c.mode, c.split, n, d, self.nA))

    def algp_order(self, d):
        if d not in self._orders:
            c = self.c
            self._orders[d] = placement_order(self.algp, self.zstd(), c.blkA, self.schedule(), seed_of(P_TAG, c.target, c.mode, c.split, d),
                                              self.placement_impl)
        return self._orders[d]

    def algp_set(self, n, d):
        return np.sort(self.algp_order(d)[:int(n)])

    # ------------------------------------------------------------ 규칙 W·W+ 의 교차검증(h54.TUnit.select_w 와 같은 계산 + 적층 후보)
    def _w_cv(self, sel, n, d, wplus=False):
        """선택 라벨 안 5겹 블록 교차검증(seed 0 적합)의 후보별 SSE. 반환 (sse 또는 None, 묶음 수, 표지). 라벨 < 10 이거나 묶음 < 2 이면 None."""
        c = self.c
        if len(sel) < W_MIN_N:
            return None, 0, f"n<{W_MIN_N}"
        fid, K, flag = self.folds(sel, n, d)
        if K < 2:
            return None, int(K), flag or "folds<2"
        seed0 = self.seeds[0]
        cands = W_CANDS + (WPLUS_EXTRA if wplus else ())
        sse = {m: 0.0 for m in cands}
        for j in range(K):
            tr, te = sel[fid != j], sel[fid == j]
            self.trace("cv", tr, method="W", n=n, draw=str(d), fold=j, held=te)
            E1, E2 = self.coefs(tr)
            s, y = c.sA[te], c.yA[te]
            a_tr, a_te = E1 * c.sA[tr], E1 * s
            nl = len(tr)
            (g1,) = self.fit(LO, "R1cv", lambda: self.rows_R(tr, a_tr), self.nsrc + nl, seed0, [c.XA[te]], n, d, tr, fold=j)
            (g2,) = self.fit(LO, "R2cv", lambda: self.rows_R(tr, a_tr, seed0, E1), self.nsrc + nl + self.n_ps, seed0, [c.XA[te]], n, d, tr, fold=j)
            (pd1,) = self.fit(LO, "D1cv", lambda: self.rows_D(tr, seed0, E1), self.nsrc + nl + self.n_ps, seed0, [c.XA[te]], n, d, tr, fold=j)
            g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
            pr = {"P0": c.E0 * s, "P1": E1 * s, "P2": E2 * s, "R1@0.25": a_te + 0.25 * g1, "R1@1.0": a_te + g1, "R2@0.25": a_te + 0.25 * g2,
                  "D1": np.asarray(pd1, float)}
            if wplus:
                pr.update({k: np.asarray(v, float) for k, v in self.wplus.cv(self, tr, te, n, d, j, seed0).items() if k in WPLUS_EXTRA})
            for m in cands:
                sse[m] += float(np.sum((pr[m] - y) ** 2))
        return sse, int(K), flag

    def select_wplus(self, sel, n, d, wplus=False):
        """규칙 W 와 W+ 의 선택(계획 2.3절 select_wplus). W 는 W 후보의 SSE 에서, W+ 는 W 후보 + 적층 후보의 SSE 에서 고른다(추가 적합 없음).
        반환 (W 선택, W+ 선택 또는 None, 묶음 수, 표지, 후보별 교차검증 RMSE)."""
        sse, K, flag = self._w_cv(sel, n, d, wplus)
        if sse is None or self.dry:
            return "P1", ("P1" if wplus else None), K, flag, {}
        cw = _pick(sse, W_CANDS)
        cp = _pick(sse, W_CANDS + WPLUS_EXTRA) if wplus else None
        return cw, cp, K, flag, {m: float(np.sqrt(v / max(len(sel), 1))) for m, v in sse.items()}

    # ------------------------------------------------------------ 한 라벨 집합의 적합
    def _set(self, sel, n, d, pl, full=True, w=True, wplus=False):
        """라벨 집합 하나: P1·P2(닫힌 형식), R1(λ 0.25·0.5·1.0), full 이면 R2·D1, w 이면 규칙 W(와 W+). TUnit.run 의 칸 하나와 같은 적합이다."""
        c = self.c
        sel = np.sort(np.asarray(sel, int))
        nl, nb = len(sel), self.nb(sel)
        self.trace("select", sel, n=n, draw=str(d), placement=pl)
        E1, E2 = self.coefs(sel)
        self.trace("coef", sel, n=n, draw=str(d), placement=pl)
        pB = {"P0": c.E0 * c.sB, "P1": E1 * c.sB, "P2": E2 * c.sB}
        self.add("P1", "none", pl, n, d, -1, 0.0, pB["P1"], E1, nl, nb)
        self.add("P2", "none", pl, n, d, -1, 0.0, pB["P2"], E2, nl, nb)
        a1A, a1B = E1 * c.sA[sel], E1 * c.sB
        preds = {}
        for seed in self.seeds:
            (g1,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel, placement=pl)
            self.emit("R1", LO, n, d, seed, a1B, g1, E1, nl, nb, flag=self.F.last_flag, placement=pl, nrow=self.nsrc + nl)
            g1 = np.asarray(g1, float)
            preds[seed] = dict(pB, **{"R1@0.25": a1B + 0.25 * g1, "R1@1.0": a1B + g1})
            if full:
                (g2,) = self.fit(LO, "R2", lambda: self.rows_R(sel, a1A, seed, E1), self.nsrc + nl + self.n_ps, seed, [c.XB], n, d, sel, placement=pl)
                self.emit("R2", LO, n, d, seed, a1B, g2, E1, nl, nb, flag=self.F.last_flag, placement=pl, nrow=self.nsrc + nl + self.n_ps)
                (p1,) = self.fit(LO, "D1", lambda: self.rows_D(sel, seed, E1), self.nsrc + nl + self.n_ps, seed, [c.XB], n, d, sel, placement=pl)
                self.add("D1", LO, pl, n, d, seed, 1.0, p1, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl + self.n_ps)
                preds[seed].update({"R2@0.25": a1B + 0.25 * np.asarray(g2, float), "D1": np.asarray(p1, float)})
        if not w:
            return preds
        cw, cp, K, flag, cv = self.select_wplus(sel, n, d, wplus)
        for seed in self.seeds:
            self.add("W", LO, pl, n, d, seed, 0.0, preds[seed][cw], np.nan, nl, nb, sel_info=cw, cv_folds=K, cv_flag=flag)
        if wplus:
            for seed in self.seeds:
                pr = preds[seed][cp] if cp in W_CANDS else np.asarray(self.wplus.final(self, sel, n, d, seed)[cp], float)
                self.add("W+", LO, pl, n, d, seed, 0.0, pr, np.nan, nl, nb, sel_info=cp, cv_folds=K, cv_flag=flag)
        if cv:
            self.notes.setdefault("w_cv_rmse", {})[f"{pl}|{n}|{d}"] = {m: round(v, 4) for m, v in cv.items()}
        return preds

    def _diag(self, sel, n, d, pl):
        """편향 진단 b10 = mean(E0·s − y)(n 10 의 라벨, 2.3절). 라벨은 그 집합만 쓴다."""
        if self.dry:
            return
        c = self.c
        sel = np.sort(np.asarray(sel, int))
        self.trace("diag", sel, n=n, draw=str(d), placement=pl)
        b = float(np.mean(c.E0 * c.sA[sel] - c.yA[sel]))
        self.diag.append(dict(n=int(n), draw=int(d), placement=pl, split=int(c.split), n_lab=int(len(sel)), nb_lab=self.nb(sel), bias=b,
                              abs_bias=abs(b)))

    def run(self):
        wp = self.wplus is not None
        rand_alg = bool(self.algp.get("is_rand"))
        for n, d in XB.cells_grid(self.grid, self.draws, self.nA, zero_n=False):
            if n == -1:                                                    # 전량: 두 배치가 같은 라벨(A 전체)
                self._set(np.arange(self.nA), n, d, self.pl_rand, full=True, w=True, wplus=wp and not self.gate)
                continue
            sel_r = self.rand_set(n, d)
            self._set(sel_r, n, d, self.pl_rand, full=True, w=True, wplus=wp and rand_alg and not self.gate and n not in A1_R1_N)
            if n == DIAG_N:
                self._diag(sel_r, n, d, self.pl_rand)
            if self.gate or rand_alg:
                continue
            sel_p = self.algp_set(n, d)
            stage1 = n in A1_R1_N
            self._set(sel_p, n, d, PL_P, full=not stage1, w=not stage1, wplus=wp and not stage1)
            if n == DIAG_N:
                self._diag(sel_p, n, d, PL_P)
        return self

    def finish(self):
        rows, st, stats = super().finish()
        stats["orders_head"] = {str(d): [int(v) for v in o[:10]] for d, o in self._orders.items()}   # 결정성 점검용(색인만)
        return rows, st, stats


class XCRUnit(W.RUnit):
    """XC-r 단위(h54.RUnit 하위 클래스, 지역 내 모드 r). 팔: 무작위 + R1(교차검증 λ, WF6 레시피), 무작위 + P1, Algorithm P + R1(교차검증 λ),
    무작위 + 규칙 W(모드 r 판: 후보와 동률 순서는 전이 규칙 W 와 같고 적합은 선택 라벨과 유사라벨 행만 쓴다)."""

    def __init__(self, a, c, algp, dry=False, placement_impl="ref", exp="xcr"):
        super().__init__(a, c, exp, "", dry)
        self.algp, self.placement_impl = algp, placement_impl
        self.grid = [int(v) for v in a.G[exp]]
        self.draws = int(getattr(a, "XCR_DRAWS", XCR_DRAWS))
        self._orders: dict = {}
        self._z = None
        self.notes.update(algp=algp["name"], placement_impl=placement_impl)

    def schedule(self):
        return [n for n in self.grid if 0 < n < self.nA]

    def rand_set(self, n, d):
        c = self.c
        return np.sort(XB.draw_nested(RAND_TAG, c.target, c.mode, c.split, n, d, self.nA))

    def algp_set(self, n, d):
        if d not in self._orders:
            c = self.c
            if self._z is None:
                self._z = zstd(c.XA)
            self._orders[d] = placement_order(self.algp, self._z, c.blkA, self.schedule(), seed_of(P_TAG, c.target, c.mode, c.split, d),
                                              self.placement_impl)
        return np.sort(self._orders[d][:int(n)])

    def _w_cv_r(self, sel, n, d):
        c = self.c
        if len(sel) < W_MIN_N:
            return None, 0, f"n<{W_MIN_N}"
        fid, K, flag = self.folds(sel, n, d)
        if K < 2:
            return None, int(K), flag or "folds<2"
        seed0 = self.seeds[0]
        sse = {m: 0.0 for m in W_CANDS}
        for j in range(K):
            tr, te = sel[fid != j], sel[fid == j]
            self.trace("cv", tr, method="Wr", n=n, draw=str(d), fold=j, held=te)
            E1, E2 = self.coefs(tr)
            s, y = c.sA[te], c.yA[te]
            a_tr, a_te = E1 * c.sA[tr], E1 * s
            (g1,) = self.fit_resid(LO, "R1wcv", tr, a_tr, seed0, n, d, preds=[c.XA[te]], tag=f"wcv{j}")
            (g2,) = self.fit_resid(LO, "R2wcv", tr, a_tr, seed0, n, d, pseudo_E=E1, preds=[c.XA[te]], tag=f"wcv{j}")
            (pd1,) = self.fit_direct(LO, "D1wcv", tr, seed0, n, d, pseudo_E=E1, preds=[c.XA[te]], tag=f"wcv{j}")
            g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
            pr = {"P0": c.E0 * s, "P1": E1 * s, "P2": E2 * s, "R1@0.25": a_te + 0.25 * g1, "R1@1.0": a_te + g1, "R2@0.25": a_te + 0.25 * g2,
                  "D1": np.asarray(pd1, float)}
            for m in W_CANDS:
                sse[m] += float(np.sum((pr[m] - y) ** 2))
        return sse, int(K), flag

    def _set_r(self, sel, n, d, pl, full=True):
        c = self.c
        sel = np.sort(np.asarray(sel, int))
        nl, nb = len(sel), self.nb(sel)
        self.trace("select", sel, n=n, draw=str(d), placement=pl)
        E1, E2 = self.coefs(sel)
        self.trace("coef", sel, n=n, draw=str(d), placement=pl)
        self.add("P1", "none", pl, n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
        self.add("P2", "none", pl, n, d, -1, 0.0, E2 * c.sB, E2, nl, nb)
        lcv, K, cf = self.lam_cv("R1", LO, "x25", sel, n, d)
        a1A, a1B = E1 * c.sA[sel], E1 * c.sB
        nps = int(round(R_PS * nl))
        preds = {}
        for seed in self.seeds:
            (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d, placement=pl)
            fl = self.F.last_flag
            g = np.asarray(g, float)
            self.emit("R1", LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, fl, placement=pl, nrow=nl)
            preds[seed] = {"P0": c.E0 * c.sB, "P1": E1 * c.sB, "P2": E2 * c.sB, "R1@0.25": a1B + 0.25 * g, "R1@1.0": a1B + g}
            if full:
                (g2,) = self.fit_resid(LO, "R2", sel, a1A, seed, n, d, pseudo_E=E1, placement=pl)
                fl2 = self.F.last_flag
                g2 = np.asarray(g2, float)
                self.emit("R2", LO, n, d, seed, a1B, g2, E1, nl, nb, flag=fl2, placement=pl, nrow=nl + nps)
                (p1,) = self.fit_direct(LO, "D1", sel, seed, n, d, pseudo_E=E1, placement=pl)
                self.add("D1", LO, pl, n, d, seed, 1.0, p1, E1, nl, nb, flag=self.F.last_flag, nrow=nl + nps)
                preds[seed].update({"R2@0.25": a1B + 0.25 * g2, "D1": np.asarray(p1, float)})
        if not full:
            return
        sse, K2, f2 = self._w_cv_r(sel, n, d)
        cw = "P1" if (sse is None or self.dry) else _pick(sse, W_CANDS)
        for seed in self.seeds:
            self.add("W", LO, pl, n, d, seed, 0.0, preds[seed][cw], np.nan, nl, nb, sel_info=cw, cv_folds=K2, cv_flag=f2)

    def run(self):
        rand_alg = bool(self.algp.get("is_rand"))
        for n, d in XB.cells_grid(self.grid, self.draws, self.nA, zero_n=False):
            if n == -1:
                self._set_r(np.arange(self.nA), n, d, PL_RAND, full=True)
                continue
            self._set_r(self.rand_set(n, d), n, d, PL_RAND, full=True)
            if not rand_alg:
                self._set_r(self.algp_set(n, d), n, d, PL_P, full=False)
        return self


# ================================================================ 6. 인자·자료·분할
def _pairs(txt, mode="x"):
    out = []
    for v in str(txt).split(","):
        v = v.strip()
        if v:
            out.append(tuple(v.split(":", 1)) if ":" in v else (v, mode))
    return out


def _grid(txt):
    return [-1 if v.strip() == "all" else int(v) for v in str(txt).split(",") if v.strip()]


def register_f3(specs) -> list:
    """부록 XC-F3 의 새 지역 실행 표를 별칭으로 등록한다('별칭=디렉터리[@표 안 대상 이름]', 대상 이름의 기본값은 별칭의 '~' 앞).
    XB.RUN_TABLES 에 더하고(점 추정 아님, 구조 적격은 집계에서 다시 센다) 별칭 목록을 돌려준다. 디렉터리는 절대 경로 또는 저장소 기준 상대 경로다."""
    out = []
    for s in [v.strip() for v in str(specs or "").split(",") if v.strip()]:
        if "=" not in s:
            raise SystemExit(f"--f3 형식은 '별칭=디렉터리[@대상]' 이다: {s}")
        al, rest = s.split("=", 1)
        d, tgt = rest.split("@", 1) if "@" in rest else (rest, al.split("~")[0])
        if "|" in al or "__" in al or not al:
            raise SystemExit(f"--f3 별칭 '{al}' 에 '|', '__' 를 쓸 수 없다")
        p = Path(d) if os.path.isabs(d) else ROOT / d
        XB.RUN_TABLES[al] = (str(p.resolve()), tgt, False)
        out.append(al)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XC_workflow_end_to_end(계획 2.3): 워크플로 순차 적용, XC-r, XC-F3, 재현 관문")
    ap.add_argument("--part", default="xc,xcr", help="xc, xcr, xcf3(쉼표 목록). --gate 는 관문 단위만 돈다")
    ap.add_argument("--targets", default=",".join(f"{t}:{m}" for t, m in XC_TARGETS_DEFAULT), help="XC 대상 '별칭:모드' 목록")
    ap.add_argument("--xcr-targets", default=",".join(XCR_TARGETS_DEFAULT))
    ap.add_argument("--f3", default="", help="부록 XC-F3 의 새 지역 '별칭=실행 표 디렉터리[:표 안 대상]' 목록(작업 R4)")
    ap.add_argument("--xc0", default="", help=f"부록 XC-0 선택 파일(JSON, 기본 {XC0_DEFAULT.relative_to(ROOT)}). 본 실행·집계에 필요하다")
    ap.add_argument("--xc0-sha256", default="", help="부록 XC-0 개정 이력에 적은 선택 파일 sha256(앞 16자 이상). 주면 파일과 대조한다")
    ap.add_argument("--algorithm-p", default="", help="스모크·세기·시험 전용 Algorithm P 이름(본 실행에서는 거부한다)")
    ap.add_argument("--placement-impl", default="auto", choices=["auto", "ref", "xd"],
                    help="배치 구현. 본 실행·집계는 xd 로 강제한다(ref 는 거부). auto 는 스모크·세기·시험 전용")
    ap.add_argument("--wplus", default="auto", choices=["auto", "off", "require"],
                    help="A1+ 의 W+(XB 적층 함수). 본 실행·집계는 require 로 강제한다(off 는 거부). auto 는 스모크·세기·시험 전용")
    ap.add_argument("--f3-status", default="auto", choices=list(F3_STATUS),
                    help="부록 XC-F3 상태. pending = 부록 XC-F3 전(XC-F3 미정, Holm 표 잠정), none = 0곳 확정, final = R4 의 xcf3 조각으로 확정. "
                         "auto 는 xcf3 조각이 있으면 final, 없으면 pending")
    ap.add_argument("--gate", action="store_true", help="재현 관문 단위(분할 1, h40.draw_cells, 레나 x·캐나다 x, WF4 격자)")
    ap.add_argument("--gate-check", default="", help="관문 조각을 WF4 조각 폴더와 대조한다(기준 폴더 경로, 'default' 는 1차 elm 조각)")
    ap.add_argument("--r2a-shards", default="", help="Algorithm P 결정성 점검에서 orders_head 를 대조할 R2a 조각 폴더(기본: --out-dir 의 shards)")
    ap.add_argument("--gate-level", default="same_node", choices=list(XB.GATE_TOL))
    ap.add_argument("--grid", default=",".join("all" if v == -1 else str(v) for v in XC_GRID))
    ap.add_argument("--xcr-grid", default=",".join(str(v) for v in XCR_GRID))
    ap.add_argument("--draws-cap", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=len(XB.SEEDS))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT.relative_to(ROOT)))
    ap.add_argument("--tag", default="xc")
    ap.add_argument("--shard", default="", help="'i/k'(단위 목록의 i 번째 묶음, 0 부터) 또는 조각 기본 이름")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="로컬 제한 스모크(분할 1, 추출 1, seed 1, 대상 축소, 화면 출력 제한)")
    ap.add_argument("--count-only", action="store_true", help="적합 수와 추정 시간(라벨 값 미사용: 문맥의 라벨을 지운 뒤 dry 실행)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--split-plan", action="store_true", help="XC·XC-r·F3 분할 계획(부록 A 대조, 세기 범주)")
    ap.add_argument("--placement-check", action="store_true", help="XD 배치 함수와 참조 구현의 순서 대조(라벨 미사용)")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--pool-retries", type=int, default=2)
    o = ap.parse_args(argv)
    o.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(o)


def finalize(o):
    o.PERMIT = XB.run_permitted(o.ARGV)
    o.nboot_arg = int(o.nboot)
    o.threads, nb = XB.local_limits(o.threads, o.nboot, o.ARGV)
    o.nboot = int(nb)
    if o.smoke and not o.PERMIT:
        o.threads = min(int(o.threads), SMOKE_MAX_THREADS)                  # 1절·5절 1a: 로컬 스모크는 코어 4개, 2스레드
    o.placement_impl_arg, o.wplus_arg = o.placement_impl, o.wplus
    o.PARTS = [v for v in (p.strip() for p in o.part.split(",")) if v in ("xc", "xcr", "xcf3")]
    o.F3 = register_f3(o.f3)
    if o.F3 and "xcf3" not in o.PARTS and not (o.summarize_only or o.count_only):
        o.PARTS.append("xcf3")
    o.T_XC = _pairs(o.targets)
    o.T_XCR = [t for t, _ in _pairs(o.xcr_targets, "r")]
    o.G_XC, o.G_XCR = _grid(o.grid), _grid(o.xcr_grid)
    o.D_XC = XC_DRAWS if o.draws_cap <= 0 else max(1, min(XC_DRAWS, int(o.draws_cap)))
    o.D_XCR = XCR_DRAWS if o.draws_cap <= 0 else max(1, min(XCR_DRAWS, int(o.draws_cap)))
    o.SUFFIX = "_smoke" if o.smoke else ""
    if o.smoke:
        o.T_XC, o.T_XCR = list(SMOKE_TARGETS), list(SMOKE_XCR)
        o.G_XC, o.G_XCR = [10, 40, -1], [200]
        o.D_XC = o.D_XCR = 1
        o.seeds = 1
        o.nboot = min(int(o.nboot), 500)
        if "xcf3" in o.PARTS:
            o.PARTS.remove("xcf3")
    if o.gate:
        o.PARTS = ["gate"]
    o.OUT = Path(o.out_dir) if os.path.isabs(o.out_dir) else ROOT / o.out_dir
    o.SHARDS = o.OUT / "shards"
    if o.smoke:                                                            # 스모크 조각은 봉인 폴더 안(평가 분할을 쓰지 않아도 결과 파일이다)
        o.SHARDS = XB.sealed_dir(EXP_NAME, sealed_root(o)) / SMOKE_SHARDS
    o.R2A_SHARDS = (Path(o.r2a_shards) if os.path.isabs(o.r2a_shards) else ROOT / o.r2a_shards) if o.r2a_shards else o.OUT / "shards"
    smoke_sp = set(SMOKE_SPLIT.values()) if o.smoke else set()
    splits = sorted(set(XB.XC_SPLITS) | set(XB.XCR_SPLITS) | ({GATE_SPLIT} if o.gate else set()) | smoke_sp)
    grid_main = list(GATE_GRID) if o.gate else list(o.G_XC)
    o.ha = XB.h54_args(exp="xc", splits=splits, grid=grid_main, threads=o.threads, cb_iters=o.cb_iters, seeds=o.seeds, nboot=o.nboot,
                       out_dir=o.OUT, tag=o.tag, draws_cap=o.draws_cap, allow_local=o.allow_local)
    o.ha.SPLITS = splits
    o.ha.G["xcr"] = list(o.G_XCR)
    o.ha.XC_DRAWS = GATE_DRAWS if o.gate else o.D_XC
    if o.gate and o.draws_cap > 0:
        o.ha.XC_DRAWS = max(1, min(GATE_DRAWS, int(o.draws_cap)))
    o.ha.XCR_DRAWS = o.D_XCR
    o.placement_impl, o.placement_sha = resolve_placement_impl(o.placement_impl)
    o.algp = None
    return o


def resolve_algp(o, need_file):
    """Algorithm P 를 정한다. 본 실행·집계(need_file)는 부록 XC-0 선택 파일만 받는다. 스모크·세기·시험은 --algorithm-p 또는 파일, 둘 다 없으면
    P_default(S8a)를 '부록 XC-0 미확정' 표지와 함께 쓴다."""
    path = Path(o.xc0) if o.xc0 else XC0_DEFAULT
    path = path if path.is_absolute() else ROOT / path
    if o.algorithm_p:
        if need_file:
            raise SystemExit("[거부] 본 실행·집계에서는 --algorithm-p 를 쓸 수 없다. 부록 XC-0 선택 파일(--xc0)을 쓴다(계획 0.3 2단계 등록)")
        return algp_spec(o.algorithm_p, source="명령행(스모크·세기 전용)")
    if path.exists():
        return load_xc0(path, require_sidecar=bool(need_file), expect_sha256=getattr(o, "xc0_sha256", ""))
    if need_file:
        raise SystemExit(f"[거부] 부록 XC-0 선택 파일이 없다: {path}. XD-alg 선택 뒤 부록 XC-0 을 커밋하고 파일을 준다")
    return algp_spec(P_DEFAULT, source="부록 XC-0 미확정(스모크·세기 기본값 P_default)")


def enforce_main_impl(o):
    """본 실행·집계(_need_file)의 구현 강제(계획 2.3 '배치 함수는 x_placement_policy 의 algorithm_p_order·power_alloc', A1+ 는 커밋된 XB 적층
    함수). --placement-impl 은 xd(auto 는 xd 로 바꾼다, ref 는 거부), --wplus 는 require(auto 는 require 로 바꾼다, off 는 거부)."""
    if not _need_file(o):
        return o
    if o.placement_impl_arg == "ref":
        raise SystemExit("[거부] 본 실행·집계에서 --placement-impl ref 는 쓸 수 없다(계획 2.3: XD 모듈의 배치 함수를 쓴다)")
    if o.wplus_arg == "off":
        raise SystemExit("[거부] 본 실행·집계에서 --wplus off 는 쓸 수 없다(계획 2.3: A1+ 는 커밋된 XB 적층 함수를 쓴다)")
    o.placement_impl, o.placement_sha = resolve_placement_impl("xd")
    o.wplus = "require"
    return o


def tag_of(o, part):
    return {"xc": f"{o.tag}{o.SUFFIX}", "xcr": f"{o.tag}r{o.SUFFIX}", "xcf3": f"{o.tag}f3{o.SUFFIX}", "gate": f"{o.tag}gate{o.SUFFIX}"}[part]


def split_plan_201(a, alias, family="xc", check=True):
    """새 seed 분할 계획(XC 201–210, XC-r 211–220). 1절 제외 규칙과 부록 A 대조(XB.split_plan). 반환 (실행 분할, 행 목록)."""
    return XB.split_plan(a, alias, family, check=check)


def f3_eligible(rows) -> bool:
    """XC-F3 의 구조 적격(2.6절 적격 규칙 가운데 분할 201–210 에서 다시 세는 부분): 유효 분할 1개 이상, 사용 분할 채점 블록 합집합 8 이상."""
    s = XB.plan_summary(rows)
    return bool(s["n_valid"] >= 1 and s["nb_union"] >= XB.MIN_BLOCKS_CI)


def unit_cfg(o, part, data_sha=""):
    """설정 해시의 원천(대상·분할·tag·워커는 넣지 않는다). 공통 해시는 variant·data_sha 를 뺀다."""
    a = o.ha
    if part == "gate":
        kw = dict(part=part, grid=list(GATE_GRID), draws=int(a.XC_DRAWS), seeds=list(a.SEEDS), cb_iters=int(a.cb_iters), w_cands=list(W_CANDS),
                  draw="h40.draw_cells", placement=PL_GATE)
        return XB.make_unit_cfg(EXP_ID, "", data_sha, **kw)
    al = o.algp or {}
    kw = dict(part=part, grid=list(a.G["xcr"] if part == "xcr" else a.G["xc"]), draws=int(a.XCR_DRAWS if part == "xcr" else a.XC_DRAWS),
              seeds=list(a.SEEDS), cb_iters=int(a.cb_iters), w_cands=list(W_CANDS), w_min_n=W_MIN_N, diag_n=DIAG_N, a1_r1_n=list(A1_R1_N),
              rand_tag=RAND_TAG, p_tag=P_TAG, algp=al.get("name"), algp_gamma=al.get("gamma"), algp_first=al.get("first"),
              algp_source=al.get("source"), algp_file_sha256=al.get("file_sha256", ""), placement_impl=o.placement_impl,
              placement_sha=o.placement_sha, split_family="xcr" if part == "xcr" else "xc")
    if part == "xc":
        kw.update(wplus=o.wplus_status, wplus_extra=list(WPLUS_EXTRA))
    if part == "xcf3":
        kw.update(f3=sorted(o.F3), wplus="미사용(XC-F3 는 A1 − A2, n 10)")
    kw.update(shard_cols_dropped=list(LABEL_RESULT_COLS), w_cv_rmse=f"sealed/{UNIT_CV_DIR}")
    return XB.make_unit_cfg(EXP_ID, "", data_sha, **kw)


def enumerate_units(o):
    """작업 단위 (part, 별칭, 모드, 분할, 변형)과 건너뛴 분할, 기대 분할."""
    a = o.ha
    units, skipped, expected = [], [], {}
    if "gate" in o.PARTS:
        for al, m in GATE_TARGETS:
            units.append(("gate", al, m, GATE_SPLIT, ""))
            expected[("gate", al, m)] = [GATE_SPLIT]
        return units, skipped, expected
    plans = []
    if "xc" in o.PARTS:
        plans += [("xc", al, m, "xc") for al, m in o.T_XC]
    if "xcr" in o.PARTS:
        plans += [("xcr", t, "r", "xcr") for t in o.T_XCR]
    if "xcf3" in o.PARTS:
        plans += [("xcf3", al, "x", "xc") for al in o.F3]
    for part, al, m, fam in plans:
        if o.smoke:                                                        # 평가 분할 밖의 분할 하나(SMOKE_SPLIT), 분할 계획을 쓰지 않는다
            sp = SMOKE_SPLIT[part]
            expected[(part, al, m)] = [sp]
            units.append((part, al, m, sp, ""))
            continue
        keep, rows = split_plan_201(a, al, fam)
        expected[(part, al, m)] = keep
        skipped += [dict(part=part, target=al, mode=m, split=r["split"], status=r["status"]) for r in rows if r["status"] != "ok"]
        units += [(part, al, m, sp, "") for sp in keep]
    return units, skipped, expected


def shard_name(o, u):
    return XB.shard_base(o.SHARDS, tag_of(o, u[0]), u[1], u[2], u[3], u[4]).name


def select_shard(o, units):
    """--shard i/k 또는 조각 기본 이름으로 단위를 고른다."""
    s = str(o.shard or "").strip()
    if not s:
        return units
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    us = sorted(units, key=lambda u: shard_name(o, u))
    if m:
        i, k = int(m.group(1)), int(m.group(2))
        if not 0 <= i < k:
            raise SystemExit(f"--shard {s}: 0 ≤ i < k 여야 한다")
        return us[i::k]
    pick = [u for u in us if shard_name(o, u) == s]
    if not pick:
        raise SystemExit(f"--shard {s} 에 해당하는 단위가 없다")
    return pick


def scrub_labels(c):
    """세기 범주(--count-only): 문맥의 라벨과 라벨에서 나온 값을 지운다(적합 수는 셀 수·블록·배치만으로 정해진다)."""
    c.yA = np.full(len(c.yA), np.nan)
    c.yB = np.full(len(c.yB), np.nan)
    if hasattr(c, "zA"):
        c.zA = np.full(len(c.yA), np.nan)
    if hasattr(c, "y_src"):
        c.y_src = np.zeros(len(c.y_src))
        c.r0_src = np.zeros(len(c.y_src))
    c.E0 = 1.0
    for k in list(c.meta):
        if k.startswith(("E_", "E0", "y_", "logE", "tau2", "sigma2")):
            c.meta[k] = float("nan")
    return c


def _plan_row(a, part, alias, split):
    fam = "xcr" if part == "xcr" else "xc"
    _, rows = split_plan_201(a, alias, fam)
    r = next((r for r in rows if int(r["split"]) == int(split)), None)
    if r is None or r["status"] != "ok":
        raise SystemExit(f"[거부] {part}|{alias}|s{split} 는 분할 계획의 실행 분할이 아니다")
    return r


def build_unit(o, part, alias, mode, split, dry=False, scrub=False):
    """(문맥, 단위, 분할 행). 문맥의 분할 구조 메타는 새 seed 규칙의 값이다."""
    a = o.ha
    algp = o.algp or algp_spec(P_DEFAULT, source="관문")
    smoke = bool(getattr(o, "smoke", False)) and part in SMOKE_SPLIT and int(split) == SMOKE_SPLIT[part]
    if part == "gate" or (smoke and part == "xc"):                         # 분할 계획 밖 분할(관문, 스모크): h40 분할 구조 그대로
        c = XB.build_tctx(a, alias, mode, split)
        prow = None
    elif part == "xcr":
        prow = None if smoke else _plan_row(a, part, alias, split)
        c = XB.build_rctx(a, alias, split, prow)
    else:
        prow = _plan_row(a, part, alias, split)
        c = XB.build_tctx(a, alias, mode, split, prow)
    wp = None if part in ("gate", "xcr", "xcf3") else o.wplus              # XC-F3(A1 − A2, n 10)에는 W+ 가 필요 없다
    if wp is not None and wp.prepare is not None and getattr(c, "xb_cand", None) is None:
        wp.prepare(a, c, alias, mode, split)                              # 후보 표(XB)를 라벨을 지우기 전에 붙인다
    if scrub:
        scrub_labels(c)
    if part == "xcr":
        U = XCRUnit(a, c, algp, dry=dry, placement_impl=o.placement_impl)
    else:
        U = XCUnit(a, c, alias, algp, dry=dry, gate=(part == "gate"), wplus=wp, placement_impl=o.placement_impl)
    return c, U, prow


def run_unit(o, part, alias, mode, split, variant="", dry=False, expected=None):
    t0 = time.time()
    c, U, prow = build_unit(o, part, alias, mode, split, dry=dry, scrub=dry)
    U.run()
    rows, st, stats = U.finish()
    if dry:
        return dict(part=part, target=alias, mode=mode, split=int(split), n_A=len(c.yA), n_eval=len(c.yB), n_rows=stats["n_rows"],
                    fit_total=int(sum(stats["n_fit"].values())), est_s=round(float(sum(stats["est_detail"].values())), 2),
                    _detail=dict(stats["n_fit_detail"]))
    tgt = c.target if part == "xcr" else alias
    dsha = XB.data_sha(a=o.ha, alias=None if part == "xcr" else alias)
    cfg = unit_cfg(o, part, dsha)
    meta = {k: v for k, v in c.meta.items() if not isinstance(v, (dict, list))}
    notes = dict(stats.get("notes") or {})
    wcv = notes.pop("w_cv_rmse", None)                                      # 교차검증 RMSE 는 봉인 기록으로 옮긴다
    unit = {**stats, **meta}
    unit.update(part=part, alias=alias, n_A=int(len(c.yA)), n_eval=int(len(c.yB)), E0=float(c.E0), elapsed_s=round(time.time() - t0, 1),
                n_fit_total=int(sum(stats["n_fit"].values())), dup_of=-1, valid=True, notes=notes,
                plan=({k: v for k, v in prow.items() if k != "eval_blocks"} if prow else {}),
                algp=(o.algp or {}).get("name", ""), placement_impl=o.placement_impl)
    exp_sp = expected if expected is not None else [int(split)]
    rows = strip_result_cols(rows)
    if wcv:
        write_unit_cv(o, part, tgt, mode, split, variant, wcv)
    return XB.write_shard(o.SHARDS, tag_of(o, part), tgt, mode, split, rows, st, cfg, unit=unit, variant=variant, expected=exp_sp,
                          code_file=__file__)


def strip_result_cols(rows) -> list:
    """조각 runs.csv 에서 라벨로 계산한 결과 열(RMSE, 블록 등가 RMSE, 편향)을 뺀다. 집계는 blocksse.npz 만 쓴다(계획 0.3 출력 제한)."""
    return [{k: v for k, v in r.items() if k not in LABEL_RESULT_COLS} for r in rows]


def write_unit_cv(o, part, tgt, mode, split, variant, wcv) -> Path:
    """규칙 W 의 후보별 교차검증 RMSE(notes.w_cv_rmse)를 봉인 폴더 unit_cv/<조각 기본 이름>_wcv.json 에 쓴다(화면 출력 없음, 매니페스트 없음:
    워커가 동시에 쓰므로 파일 하나씩 원자적으로 쓴다)."""
    d = XB.sealed_dir(EXP_NAME, sealed_root(o)) / UNIT_CV_DIR
    XB.check_out_dir(d)
    d.mkdir(parents=True, exist_ok=True)
    base = XB.shard_base(d, tag_of(o, part), tgt, mode, split, variant)
    p = Path(str(base) + "_wcv.json")
    XB._atomic_write(p, json.dumps(dict(w_cv_rmse=wcv), ensure_ascii=False, default=float))
    return p


# ---------------------------------------------------------------- 프로세스 풀(XB.execute)
_O = None
_EXP: dict = {}


def _worker_init(argv, exp_items=()):
    global _O, _EXP
    _O = enforce_main_impl(parse_args(list(argv)))
    _O.algp = resolve_algp(_O, need_file=_need_file(_O))
    _O.wplus, _O.wplus_status = load_wplus("off" if "gate" in _O.PARTS else _O.wplus)
    _EXP = {tuple(k): v for k, v in exp_items}


def _worker_run(part, alias, mode, split, variant):
    t0 = time.time()
    u = run_unit(_O, part, alias, mode, split, variant, expected=_EXP.get((part, alias, mode)))
    return dict(part=part, target=alias, mode=mode, split=int(split), status=u.get("status"), n_fit=u.get("n_fit_total"),
                wall_s=round(time.time() - t0, 1))


def _need_file(o):
    return not (o.smoke or o.count_only or o.split_plan or o.placement_check or "gate" in o.PARTS or o.gate_check)


# ================================================================ 7. 집계(대비·가설·보조)
def cand_key(m, n, pl=PL_RAND):
    """W 후보 이름 → 곡선 키."""
    if m == "P0":
        return H.P0_GRP
    if m in ("P1", "P2"):
        return W.gk(m, n, "none", placement=pl)
    if m == "D1":
        return W.gk("D1", n, LO, 1.0, pl)
    meth, lam = m.split("@")
    return W.gk(meth, n, LO, float(lam), pl)


def has_group(tm, g):
    return tm is not None and any(g in gd for gd in tm.idx.values())


def a6_key(tm, n):
    """A6: 무작위 라벨의 W 후보 가운데 그 (대상, n)에서 셀 가중 RMSE(추출·seed 평균 뒤 분할 평균)가 가장 작은 고정 방법(동률은 후보 순서)."""
    best, bv = None, np.inf
    for m in W_CANDS:
        g = cand_key(m, n)
        if not has_group(tm, g):
            continue
        v = XB.grp_rmse2(tm, g)[0]
        if np.isfinite(v) and v < bv:
            best, bv = g, v
    return best


def arm_key(arm, n, is_rand=False, tm=None):
    """팔 → 곡선 키(모듈 머리말의 팔 표). 전량과 Algorithm P = S1 은 algP 대신 rand 키다."""
    pp = PL_RAND if (is_rand or int(n) == -1) else PL_P
    n = int(n)
    if arm == "P0":
        return H.P0_GRP
    if arm == "A1":
        return W.gk("R1", n, LO, LAM_BASE, pp) if n in A1_R1_N else W.gk("W", n, LO, 0.0, pp)
    if arm == "A1+":
        return W.gk("R1", n, LO, LAM_BASE, pp) if n in A1_R1_N else W.gk("W+", n, LO, 0.0, pp)
    if arm == "A2":
        return W.gk("R1", n, LO, LAM_BASE, PL_RAND)
    if arm == "A3":
        return W.gk("P1", n, "none", placement=PL_RAND)
    if arm == "A4":
        return W.gk("R1", n, LO, LAM_BASE, pp)
    if arm == "A5":
        return W.gk("W", n, LO, 0.0, PL_RAND)
    if arm == "A6":
        return a6_key(tm, n)
    # XC-r 의 팔
    if arm == "rA2":
        return W.gk("R1", n, LO, LAM_CV, PL_RAND)
    if arm == "rA3":
        return W.gk("P1", n, "none", placement=PL_RAND)
    if arm == "rA4":
        return W.gk("R1", n, LO, LAM_CV, PL_RAND if is_rand else PL_P)
    if arm == "rA5":
        return W.gk("W", n, LO, 0.0, PL_RAND)
    raise ValueError(arm)


def arm_set(arm, n, is_rand=False):
    """팔의 라벨 집합 표지(None = 라벨 없음). 두 팔의 표지가 같으면 같은 라벨 집합 대비다."""
    if arm == "P0":
        return None
    if arm in ("A1", "A1+", "A4", "rA4"):
        return PL_RAND if (is_rand or int(n) == -1) else PL_P
    return PL_RAND


def contrast_kind(armA, armB, n, is_rand=False):
    sA, sB = arm_set(armA, n, is_rand), arm_set(armB, n, is_rand)
    return "same" if (sA is None or sB is None or sA == sB) else "two_stage"


def identical_arms(armA, armB, n, is_rand=False) -> bool:
    """두 팔이 구조상 같은 곡선 키인가(Δ ≡ 0 인 대비). A6 은 자료에 따라 정해지므로 넣지 않는다(후회값 0 은 결과다).
    예: Algorithm P = S1 이면 n 10 의 A1 = A2(XC-F3, S-XC0), n 40·160 의 A1 = A5(XC-5), A4 = A2(S-XC3)."""
    if "A6" in (armA, armB):
        return False
    try:
        return arm_key(armA, n, is_rand) == arm_key(armB, n, is_rand)
    except ValueError:
        return False


def untested_reason(armA, armB, n, is_rand=False, hid="") -> str:
    """같은 키 대비의 '시험하지 않음' 사유. 규칙 5 가 이름으로 정한 대비(XC-5b·5c, S-XC3)는 규칙 문구 그대로 쓴다."""
    if not identical_arms(armA, armB, n, is_rand):
        return ""
    if hid.startswith("XC-5") or hid == "S-XC3":
        return RULE5_TXT
    return f"시험하지 않음(Algorithm P = S1, 라벨 {W.nlab(n)} 에서 {armA} = {armB}, 부록 XC-0 규칙 5 와 같은 이유)"


def selection_family(name) -> str:
    """Algorithm P 선택에 쓴 알래스카 계열(알래스카 x·r, AL-k 의 모든 모드)이면 '선택 계열'(2.3절 3지역 보조 열의 표지)."""
    t = str(name).split("|")[0]
    return SEL_FAMILY_TXT if (t.split("~")[0] == "Alaska" or t.startswith("AL-")) else ""


def mark_family(df):
    """target 열이 있는 표에 selection_family 열을 더한다(이미 값이 있는 행, 예: 풀 평균 행의 '선택 계열 포함(...)' 은 그대로 둔다)."""
    if not (isinstance(df, pd.DataFrame) and len(df) and "target" in df.columns):
        return df
    df = df.copy()
    fam = [selection_family(t) if not str(t).startswith("MEAN") else "" for t in df["target"].astype(str)]
    if "selection_family" in df.columns:
        old = df["selection_family"].fillna("").astype(str)
        df["selection_family"] = [v if v else f for v, f in zip(old, fam)]
    else:
        df["selection_family"] = fam
    return df


def aux_two_stage(tms, names, gA, gB, nboot=None) -> dict:
    """같은 라벨 집합 XC 팔 대비의 보조 2단 CI(1절 '적용: XC 의 모든 팔 대비'): 지역마다 XB.two_stage(주 2단과 같은 seed 규약)의 분포."""
    out = {}
    for nm in names:
        tm = tms.get(nm)
        if tm is None or not has_group(tm, gA) or not has_group(tm, gB):
            continue
        nb = int(tm.nboot if nboot is None else nboot) if (tm.has_ci and tm.nb_union >= XB.MIN_BLOCKS_CI) else 0
        try:
            m = XB.two_stage(tm, gA, gB, nb, seed_of("xbatch-2s", tm.name))
        except ValueError:
            m = None
        if m is not None:
            out[nm] = m
    return out


def _attach_aux2s(rows, per2):
    """지역 행과 평균 행에 보조 2단 CI 열(ci_lo_2s 등, 4분 판정, 비열등·우월)을 붙인다. 평균은 분포가 있는 지역 2개 이상일 때 층화 평균."""
    def cols(dc, db):
        lo, hi = X._ci(dc) if dc is not None else (np.nan, np.nan)
        lob, hib = X._ci(db) if db is not None else (np.nan, np.nan)
        ok = dc is not None and db is not None
        return dict(ci_lo_2s=lo, ci_hi_2s=hi, ci_lo_beq_2s=lob, ci_hi_beq_2s=hib,
                    verdict4_2s=XB.verdict4(lo, hi, lob, hib) if ok else "판정 불가",
                    noninf_2s=XB.noninf(hi, hib, XB.NI_MARGIN), superior_2s=XB.noninf(hi, hib, 0.0), aux2s="같은 라벨 집합 대비의 보조 2단 CI")
    have = []
    for r in rows:
        if r.get("scope") == "region":
            m = per2.get(r.get("target"))
            r.update(cols(m.get("dist") if m else None, m.get("dist_beq") if m else None))
            if m is not None and m.get("dist") is not None and m.get("dist_beq") is not None:
                have.append(m)
    mr = mean_row(rows)
    if mr is not None:
        if len(have) >= XB.MIN_POOL_REGIONS:
            mr.update(cols(MS.strat([m["dist"] for m in have]), MS.strat([m["dist_beq"] for m in have])))
        else:
            mr.update(cols(None, None))
    return rows


def two_stage_ci(tm, gA, gB, nboot=None, seed=None, tags=None):
    """2단 CI(1절, h39 l43_region 의 일반화). 반환 dict(delta, delta_beq, dist, dist_beq, ci_lo, ci_hi, ci_lo_beq, ci_hi_beq) 또는 None."""
    nb = int(tm.nboot if nboot is None else nboot)
    s = seed_of("xbatch-2s", tm.name) if seed is None else seed
    r = XB.two_stage(tm, gA, gB, nb, s, tags)
    if r is not None:
        r.update(zip(("ci_lo", "ci_hi"), X._ci(r.get("dist"))))
        r.update(zip(("ci_lo_beq", "ci_hi_beq"), X._ci(r.get("dist_beq"))))
    return r


def two_stage_ci_common(tm, gA, gB, nboot=None, seed=None, tags=None, w_seed=None):
    """2단 공통 CI(1절 보조 CI: 1단의 블록 가중을 사용 분할 채점 블록의 합집합에서 한 번 뽑는다)."""
    nb = int(tm.nboot if nboot is None else nboot)
    s = seed_of("xbatch-2s", tm.name) if seed is None else seed
    r = XB.two_stage(tm, gA, gB, nb, s, tags, common=True, w_seed=w_seed)
    if r is not None:
        r.update(zip(("ci_lo", "ci_hi"), X._ci(r.get("dist"))))
        r.update(zip(("ci_lo_beq", "ci_hi_beq"), X._ci(r.get("dist_beq"))))
    return r


def region_stat(tm, armA, armB, n, is_rand=False, nboot=None):
    """지역 하나의 대비 통계. 라벨 집합이 다르면 2단(주)·2단 공통·추출 조건부, 같으면 h40.contrast·공통 재표집."""
    if tm is None:
        return None
    gA, gB = arm_key(armA, n, is_rand, tm), arm_key(armB, n, is_rand, tm)
    if gA is None or gB is None or not has_group(tm, gA) or not has_group(tm, gB):
        return None
    if contrast_kind(armA, armB, n, is_rand) == "same":
        return XB.region_stat_same(tm, gA, gB)
    return XB.region_stat_two_stage(tm, gA, gB, nboot=nboot)


def cpool(tms, names, armA, armB, n, label="", is_rand=False, nboot=None, registered=None):
    """풀 하나의 대비(지역 행 + 층화 평균 행). 반환 (행 목록, 지역 통계 dict)."""
    per = {}
    for nm in names:
        s = region_stat(tms.get(nm), armA, armB, n, is_rand, nboot)
        if s is not None:
            per[nm] = s
    XB.boot_cache_clear()
    rows = XB.pool(per, list(names), label, registered) if per else []
    kind = contrast_kind(armA, armB, n, is_rand)
    if rows and kind == "same" and armA in AUX2S_ARMS and armB in AUX2S_ARMS:
        per2 = {}
        for nm in names:                                                   # A6 는 대상마다 키가 다르다
            tm = tms.get(nm)
            ga, gb = arm_key(armA, n, is_rand, tm), arm_key(armB, n, is_rand, tm)
            if tm is not None and ga is not None and gb is not None and ga != gb:
                per2.update(aux_two_stage({nm: tm}, [nm], ga, gb, nboot))
        XB.boot_cache_clear()
        _attach_aux2s(rows, per2)
    fam = [nm for nm in names if nm in per and selection_family(nm)]
    for r in rows:
        r.update(contrast=f"{armA}-{armB}", n=int(n), ci_kind_reg=kind, label=label)
        if r.get("scope") == "region":
            r["selection_family"] = selection_family(r.get("target"))
        elif r.get("scope") == "MEAN":
            r["selection_family"] = f"{SEL_FAMILY_TXT} 포함({','.join(fam)})" if fam else ""
    return rows, per


def mean_row(rows):
    return rows[-1] if rows and rows[-1].get("scope") == "MEAN" else None


def loo_rows(per, names, label):
    """지역 하나 제외 평균(2.3절 보조 열): 풀 지역 가운데 하나를 뺀 나머지의 층화 평균과 CI."""
    pl = [nm for nm in names if nm in per and per[nm].get("dist") is not None]
    out = []
    for r in pl:
        rest = [nm for nm in pl if nm != r]
        if not rest:
            continue
        dc = MS.strat([per[nm]["dist"] for nm in rest]); db = MS.strat([per[nm]["dist_beq"] for nm in rest])
        lo, hi = X._ci(dc); lob, hib = X._ci(db)
        out.append(dict(label=label, scope="loo", left_out=r, regions=",".join(rest), delta=float(np.mean([per[nm]["delta"] for nm in rest])),
                        delta_beq=float(np.mean([per[nm]["delta_beq"] for nm in rest])), ci_lo=lo, ci_hi=hi, ci_lo_beq=lob, ci_hi_beq=hib,
                        verdict4=XB.verdict4(lo, hi, lob, hib)))
    return out


def mde_of(row):
    """최소 검출 효과(2.3절 검정력): SE ≈ (2단 CI 반폭)/1.96, MDE ≈ 2.80 × SE(셀 가중)."""
    if row is None:
        return np.nan
    lo, hi = row.get("ci_lo"), row.get("ci_hi")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    return float(MDE_FACTOR * (hi - lo) / 2.0 / Z975)


def _fmt(v):
    return f"{abs(float(v)):.2f}" if v is not None and np.isfinite(v) else "nan"


def hyp_branch(kind, row, untested_reason=""):
    """(갈래, 사유). superior: 주 2단 CI 의 4분 판정. noninf: 비열등(주 2단 CI 상한 < 0.5, 두 가중)이면 '우세' 갈래(비열등 성립 문장),
    아니면 열세·동등·미결정. 행이 없거나 풀이 판정 불가면 '판정 불가'. 부록 XC-F3 전의 XC-F3 은 '미정(부록 XC-F3 전)'."""
    if untested_reason == F3_PENDING_TXT:
        return F3_PENDING_TXT, untested_reason
    if untested_reason:
        return "판정 불가", untested_reason
    if row is None:
        return "판정 불가", "행 없음"
    if row.get("undetermined") or row.get("verdict4") in XB.NA_VERDICTS:
        return "판정 불가", str(row.get("pool") or row.get("ci_flag") or "판정 불가")
    v = str(row.get("verdict4"))
    if kind == "noninf":
        if XB.noninf(row.get("ci_hi"), row.get("ci_hi_beq"), XB.NI_MARGIN):
            return "우세", ""
        return (v if v in ("열세", "동등") else "미결정"), ""
    return XB.branch_of(v), ""


def _finite(*vs) -> bool:
    return all(v is not None and np.isfinite(v) for v in vs)


def dep_flags(kind, row):
    """(분할 독립 가정 의존, 추출 변동 의존)(1절 보조 CI 행). 분할 독립: 주 2단 CI 와 2단 공통 CI 의 기준 판단(우월 0, 비열등 0.5)이 다르거나
    4분 판정이 다르다. 추출 변동: 추출 조건부 CI 의 기준 판단 또는 4분 판정이 주 CI 와 다르다(추출 조건부 CI 가 있을 때)."""
    if row is None:
        return False, False
    lim = 0.0 if kind == "superior" else XB.NI_MARGIN
    main_ok = XB.noninf(row.get("ci_hi"), row.get("ci_hi_beq"), lim)
    comm_ok = XB.noninf(row.get("ci_hi_c"), row.get("ci_hi_beq_c"), lim)
    ci_dep = bool(main_ok != comm_ok or row.get("ci_dependence"))
    xhi, xhib = row.get("ci_hi_x"), row.get("ci_hi_beq_x")
    draw_dep = bool(row.get("draw_dependence") or (_finite(xhi, xhib) and XB.noninf(xhi, xhib, lim) != main_ok))
    return ci_dep, draw_dep


def hyp_tags(kind, row, holm_p):
    """판정 행의 표지: 보정 전 유의, 분할 독립 가정 의존(2단 공통 CI 와 판정이 다름), 추출 변동 의존, 한계 의존, 소수 블록 지역, 지역 일반 허용."""
    if row is None:
        return ""
    t = []
    lim = 0.0 if kind == "superior" else XB.NI_MARGIN
    main_ok = XB.noninf(row.get("ci_hi"), row.get("ci_hi_beq"), lim)
    if (main_ok or row.get("verdict4") in ("열세",)) and holm_p is not None and np.isfinite(holm_p) and holm_p >= XB.HOLM_ALPHA:
        t.append(XB.UNCORRECTED_TXT)
    ci_dep, draw_dep = dep_flags(kind, row)
    if ci_dep:
        t.append(XB.CI_DEP_TXT)
    if draw_dep:
        t.append(XB.DRAW_DEP_TXT)
    if row.get("limit_dependence"):
        t.append(str(row["limit_dependence"]))
    if row.get("few_block_regions"):
        t.append(f"{XB.FEW_BLOCKS_TXT}({row['few_block_regions']})")
    if row.get("region_general"):
        t.append("지역 일반 문장 허용(HK CI 가 0 을 제외)")
    return "; ".join(dict.fromkeys(t))


def sentence_deps(kind, row, branch) -> list:
    """해석 문장의 괄호 표기에 넣는 의존 표지(1절: 주 CI 와 판정이 다르면 '분할 독립 가정 의존', 추출 조건부와 다르면 '추출 변동 의존').
    판정을 말하는 갈래(우세 또는 비열등 성립, 열세, 동등)에만 붙인다. 예: 주 2단 CI 로 우세인데 2단 공통 CI 가 기준을 못 넘으면 기준 미충족이고
    문장은 '...작았다(분할 독립 가정 의존)'이 된다."""
    if row is None or branch not in ("우세", "열세", "동등"):
        return []
    ci_dep, draw_dep = dep_flags(kind, row)
    return [t for t, f in ((XB.CI_DEP_TXT, ci_dep), (XB.DRAW_DEP_TXT, draw_dep)) if f]


def _add_notes(tpl, branch, s, extra) -> str:
    """XB.compose_sentence 가 만든 풀 문장(지역 문장을 붙이기 전)의 괄호 표기에 extra 를 더한다. 사전 고정 문장 base 뒤의 '(표기)' 를 찾아 합친다."""
    if not extra:
        return s
    base = str(tpl.get(XB.branch_of(branch), "")).rstrip().rstrip(".")
    if not (s.startswith(base) and s.endswith(".")):
        raise ValueError("해석 문장의 형식이 XB.compose_sentence 와 다르다")
    rest = s[len(base):-1]
    inner = rest[1:-1] if (rest.startswith("(") and rest.endswith(")")) else ""
    return base + f"({', '.join([v for v in [inner] if v] + list(extra))})."


def _fam_mark(nm) -> str:
    f = selection_family(nm)
    return f"({f})" if f else ""


def compose_hyp_sentence(hid, kind, n, row, branch, holm_p, reason="", m_regions=0, worse=(), lena=None, xc2s=None, indep=None,
                         deps=(), f3_none=False, indep_k=None):
    """사전 고정 해석 문장(2.3절 표)을 채운다. 우세·열세는 Holm 보정 p ≥ 0.05 이면 '보정 전 유의', 미결정은 MDE 가설에서 최소 검출 효과,
    deps(분할 독립 가정 의존, 추출 변동 의존)를 같은 괄호에 넣고, 풀 문장 뒤 지역 열세 문장, XC-2s 우세 문장, XC-2 의 독립 지역 문장(이 라벨 수에서
    평가한 지역 수 indep_k 를 괄호로), 레나 행을 덧붙인다. 'F3 0곳' 문장은 f3_none(부록 XC-F3 0곳 확정 또는 R4 조각에서 적격 0곳)일 때만 쓰고,
    부록 XC-F3 전이면 자리표시를 둔다."""
    fam = hid[:5] if hid.startswith("XC-F3") else hid[:4]
    if branch == F3_PENDING_TXT:
        return f"[{hid}: {F3_PENDING_TXT}. 부록 XC-F3 확정 뒤 집계에서 문장을 정한다]"
    if hid == "XC-F3" and f3_none:
        return SENTENCES["XC-F3-none"]
    tpl = {k: v.format(k=int(n), a=_fmt(row.get("delta") if row else np.nan), m=int(m_regions)) for k, v in SENTENCES[fam].items()}
    delta = None if (kind == "noninf" and branch == "우세") else (row.get("delta") if row else None)
    mde = mde_of(row) if hid in MDE_HYPS else None
    out = _add_notes(tpl, branch, XB.compose_sentence(tpl, branch, holm_p, delta, (), mde, reason), list(deps))
    for nm, d, lo, hi in worse:                                             # XB.compose_sentence 의 지역 열세 문장과 같은 형식
        out += f" 지역 {nm} 에서는 오차가 컸다(Δ {float(d):+.2f} cm, CI [{float(lo):.2f}, {float(hi):.2f}])."
    if xc2s is not None and xc2s.get("result") == "우세":
        out += " " + SENTENCES["XC-2s"].format(a=_fmt(xc2s.get("delta")))
    if indep is not None:
        lst = ", ".join(f"{nm}{_fam_mark(nm)} Δ {d:+.2f} cm" for nm, d in indep) if indep else "없음"
        s_ = SENTENCES["XC-2-indep"].format(j=len(indep), lst=lst)
        if indep_k is not None:
            s_ = s_[:-1] + f"(이 라벨 수에서 평가한 지역 {int(indep_k)}곳)."
        out += " " + s_
        if indep:
            out += " " + SENTENCES["XC-2-indep-tail"].format(lst=", ".join(nm for nm, _ in indep))
    if lena is not None:
        out += f" 레나 행: Δ {lena['delta']:+.2f} cm(셀 가중), CI [{lena.get('ci_lo', np.nan):.2f}, {lena.get('ci_hi', np.nan):.2f}], {lena.get('verdict4', '')}."
    return out


def xc2s_test(row_ni, met_ni, multiplier):
    """XC-2s 고정 순서 우월 시험: XC-2 가 기준을 만족한 n 에서만 우월(2단·2단 공통 CI 상한 < 0, 두 가중)과 우월 양측 p × 보정 배수 < 0.05 를 본다.
    비열등이 성립하지 않은 n 은 '시험하지 않음'. 보정 배수는 holm_with_ties 의 multiplier_tie(동률 묶음의 첫 순위 배수)."""
    if not met_ni or row_ni is None:
        return dict(result="시험하지 않음", reason="비열등 불성립", p_adj=np.nan, delta=np.nan)
    p = row_ni.get("p_two", np.nan)
    p_adj = float(min(1.0, p * float(multiplier))) if np.isfinite(p) else np.nan
    sup = XB.criterion_met(row_ni, "superior")
    ok = bool(sup and np.isfinite(p_adj) and p_adj < XB.HOLM_ALPHA)
    return dict(result="우세" if ok else "우세 아님", reason="" if ok else ("CI 기준 불충족" if not sup else "보정 p ≥ 0.05"), p_adj=p_adj,
                delta=row_ni.get("delta"), criterion_ci=bool(sup))


def holm_with_ties(labels, pvals, m) -> pd.DataFrame:
    """XB.holm_table 에 동률 처리를 더한다. p 가 같은 가설은 Holm 보정 p 가 같고(누적 최댓값) 그 값은 동률 묶음의 첫 순위 배수에서 나온다.
    XC-2s 의 보정 배수는 그 첫 순위의 배수(multiplier_tie = m − 첫 순위 + 1)다. Holm 보정 p ≥ min(1, multiplier_tie × p) 를 단언한다
    (m1_stats.holm 의 정렬 안정성과 무관하게 성립한다)."""
    ht = XB.holm_table(labels, pvals, m)
    p = ht["p"].to_numpy(float)
    rk = ht["rank"].to_numpy(int)
    first = np.array([int(rk[p == v].min()) for v in p], int)
    ht["rank_tie"] = first
    ht["multiplier_tie"] = int(m) - first + 1
    rhs = np.minimum(1.0, ht["multiplier_tie"].to_numpy(float) * p)
    if not np.all(ht["p_holm"].to_numpy(float) >= rhs - 1e-12):
        raise AssertionError("Holm 보정 p 가 동률 묶음의 첫 배수 × p 보다 작다")
    return ht


def indep_worse(tms, n, is_rand, nboot):
    """S-XC4 (b): 독립 지역 7곳 가운데 A1 이 A3 보다 0.5 cm 넘게 나빴던 지역(셀 가중 점 추정 Δ > 0.5) [(지역, Δ)], 지역 행(이 n 에서 평가한 지역)."""
    rows, per = cpool(tms, INDEP7, "A1", "A3", n, f"S-XC4b|n{W.nlab(n)}", is_rand, nboot, len(INDEP7))
    regs = [r for r in rows if r.get("scope") == "region"]
    out = [(r["target"], float(r["delta"])) for r in regs if np.isfinite(r["delta"]) and r["delta"] > WORSE_THAN_A3_CM]
    return out, regs


def tests_xc(tms, units, algp, f3_names=(), nboot=None, floors=None, f3_status="pending"):
    """주 가설 8개(Holm m = 8), XC-2s, 해석 문장, 보조 S-XC0–S-XC12. 반환 {표 이름: DataFrame 또는 dict}(키 '_state' 는 봉인하지 않는 상태 기록).
    f3_status: pending = 부록 XC-F3 전(XC-F3 '미정', p 1 로 넣은 Holm 표는 잠정), none = 부록 XC-F3 0곳 확정('F3 0곳' 문장),
    final = R4 의 xcf3 조각으로 확정(f3_names 가 그 지역). Algorithm P = S1 이면 XC-F3 은 F3 상태와 관계없이 '시험하지 않음'(n 10 에서 A1 = A2)
    이라 Holm 표가 확정된다. 화면에는 쓰지 않는다."""
    if f3_status not in ("pending", "none", "final"):
        raise ValueError(f"f3_status 는 pending, none, final 가운데 하나다: {f3_status}")
    is_rand = bool(algp.get("is_rand"))
    f3 = [nm if "|" in nm else f"{nm}|x" for nm in f3_names]
    f3_ok = [nm for nm in f3 if nm in tms and tms[nm].has_ci and tms[nm].nb_union >= XB.MIN_BLOCKS_CI] if f3_status == "final" else []
    allrows, hyp_rows, loo = [], [], []
    res = {}
    for hid, A, B, n, kind in HYPS:
        same = untested_reason(A, B, n, is_rand, hid)
        untested, f3_none = "", False
        if hid == "XC-F3":
            names = f3_ok
            if f3_status == "pending" and not same:
                untested = F3_PENDING_TXT
            elif f3_status in ("none", "final") and not names:
                untested, f3_none = F3_NONE_TXT, True
            elif same:
                untested = same
            rows, per = cpool(tms, names, A, B, n, hid, is_rand, nboot, len(names) or None) if (names and not untested) else ([], {})
            row = mean_row(rows) if len(names) >= 2 else (rows[0] if rows else None)
        else:
            untested = same
            rows, per = cpool(tms, PE1, A, B, n, hid, is_rand, nboot, len(PE1)) if not untested else ([], {})
            row = mean_row(rows)
            loo += loo_rows(per, PE1, hid)
        allrows += rows
        pkey = "p_ni_x2" if kind == "noninf" else "p_two"
        p = np.nan if (untested or row is None or row.get("undetermined")) else row.get(pkey, np.nan)
        res[hid] = dict(row=row, rows=rows, p=p, untested=untested, kind=kind, n=n, A=A, B=B, f3_none=f3_none,
                        m_regions=len(f3_ok) if hid == "XC-F3" else 0)
    holm_final = bool(res["XC-F3"]["untested"] != F3_PENDING_TXT)
    ht = holm_with_ties([h[0] for h in HYPS], [res[h[0]]["p"] for h in HYPS], HOLM_M)
    ht["holm_final"] = holm_final
    for (hid, A, B, n, kind), (_, hr) in zip(HYPS, ht.iterrows()):
        q = res[hid]
        row = q["row"]
        crit = bool(not q["untested"] and XB.criterion_met(row, kind if kind == "noninf" else "superior"))
        met = bool(crit and np.isfinite(hr.p_holm) and hr.p_holm < XB.HOLM_ALPHA)
        branch, reason = hyp_branch(kind, row, q["untested"])
        q.update(holm_p=float(hr.p_holm), multiplier=int(hr.multiplier_tie), multiplier_stable=int(hr.multiplier), rank=int(hr["rank"]),
                 criterion_ci=crit, met=met, branch=branch, reason=reason, deps=sentence_deps(kind, row, branch))
    # XC-2s(고정 순서 우월, 가족 크기 유지)
    s2 = {}
    for hid in ("XC-2a", "XC-2b", "XC-2c"):
        q = res[hid]
        s2[hid] = xc2s_test(q["row"], q["met"], q["multiplier"])
    # 독립 지역 손해(XC-2 의 덧붙임, S-XC4 (b))
    indep, indep_k = {}, {}
    harm_rows = []
    for n in (10, 40, 160):
        lst, regs = indep_worse(tms, n, is_rand, nboot)
        indep[n], indep_k[n] = lst, len(regs)
        for r in regs:
            harm_rows.append(dict(item="S-XC4b", n=n, target=r["target"], delta=r["delta"], delta_beq=r["delta_blockeq"], ci_lo=r["ci_lo"],
                                  ci_hi=r["ci_hi"], ci_lo_beq=r["ci_lo_beq"], ci_hi_beq=r["ci_hi_beq"], verdict4=r["verdict4"],
                                  worse_than_A3_05=bool(np.isfinite(r["delta"]) and r["delta"] > WORSE_THAN_A3_CM),
                                  n_regions_evaluated=len(regs), n_regions_registered=len(INDEP7)))
    for hid, A, B, n, kind in HYPS:
        q = res[hid]
        row = q["row"]
        lena = next((r for r in q["rows"] if r.get("scope") == "region" and r.get("target") == "Lena|x"), None) if hid != "XC-F3" else None
        worse = XB.worse_regions(q["rows"]) if hid != "XC-F3" or len(f3_ok) >= 2 else []
        is2 = hid.startswith("XC-2")
        sent = compose_hyp_sentence(hid, kind, n, row, q["branch"], q["holm_p"], q["reason"], q["m_regions"], worse, lena,
                                    xc2s=s2.get(hid), indep=indep.get(n) if is2 else None, deps=q["deps"], f3_none=q["f3_none"],
                                    indep_k=indep_k.get(n) if is2 else None)
        g = (row or {}).get
        hyp_rows.append(dict(hypothesis=hid, contrast=f"{A}-{B}", n=n, kind=kind, pool="F3" if hid == "XC-F3" else "PE1",
                             pool_label=g("pool", ""), delta=g("delta", np.nan), delta_beq=g("delta_blockeq", np.nan), ci_lo=g("ci_lo", np.nan),
                             ci_hi=g("ci_hi", np.nan), ci_lo_beq=g("ci_lo_beq", np.nan), ci_hi_beq=g("ci_hi_beq", np.nan), ci_hi_c=g("ci_hi_c", np.nan),
                             ci_hi_beq_c=g("ci_hi_beq_c", np.nan), verdict4=g("verdict4", "행 없음"), verdict4_common=g("verdict4_common", ""),
                             verdict4_draw_cond=g("verdict4_draw_cond", ""), verdict4_d10=g("verdict4_d10", ""), verdict4_rel=g("verdict4_rel", ""),
                             ci_kind=g("ci_kind", ""), ci_hi_2s=g("ci_hi_2s", np.nan), ci_hi_beq_2s=g("ci_hi_beq_2s", np.nan),
                             verdict4_2s=g("verdict4_2s", ""),
                             p_raw=q["p"], p_holm=q["holm_p"], holm_rank=q["rank"], holm_multiplier=q["multiplier"],
                             holm_multiplier_stable=q["multiplier_stable"], m=HOLM_M, holm_final=holm_final, f3_status=f3_status,
                             criterion_ci=q["criterion_ci"], criterion_met=q["met"], branch=q["branch"], reason=q["reason"],
                             tags=hyp_tags(kind, row, q["holm_p"]), sentence_notes="; ".join(q["deps"]),
                             mde_cm=mde_of(row) if hid in MDE_HYPS else np.nan,
                             worse_regions=g("worse_regions", ""), region_general=g("region_general", False), selection_family=g("selection_family", ""),
                             blind=BLIND_F3 if hid == "XC-F3" else BLIND_MAIN, design=DESIGN_MAIN if hid != "XC-F3" else "실행 전 등록 확인 시험",
                             xc2s=s2.get(hid, {}).get("result", ""), xc2s_p_adj=s2.get(hid, {}).get("p_adj", np.nan),
                             algp=algp.get("name"), sentence=sent))
    out = dict(hyp=pd.DataFrame(hyp_rows), holm=ht, contrasts_main=XB.clean_rows(allrows), loo=pd.DataFrame(loo), harm_b=pd.DataFrame(harm_rows))
    out.update(aux_tables(tms, units, algp, nboot, floors))
    out = {k: mark_family(v) if k not in ("hyp", "holm") else v for k, v in out.items()}
    out["sentences"] = dict(common=COMMON_SENTENCE, hypotheses={r["hypothesis"]: r["sentence"] for r in hyp_rows}, holm_final=holm_final,
                            f3_status=f3_status,
                            note="모든 문장은 '같은 지역을 다시 무작위로 나눈 분할에서'를 넣었다(2.3절). 원고 초록 규칙은 7.1(수치만 또는 제외)")
    out["_state"] = dict(holm_final=holm_final, f3_status=f3_status, f3_regions=list(f3_ok))
    return out


def aux_tables(tms, units, algp, nboot=None, floors=None):
    """보조 S-XC0–S-XC12(판정어는 붙이되 확인적으로 세지 않는다)와 F2 대상 수, F3 지역 행."""
    is_rand = bool(algp.get("is_rand"))
    rows = []

    def add(item, names, A, B, n, reg=None):
        untested = untested_reason(A, B, n, is_rand, item)                 # 같은 키 대비(Algorithm P = S1): S-XC0, S-XC3, S-XC8 의 일부
        if untested:
            rows.append(dict(item=item, contrast=f"{A}-{B}", n=n, scope="MEAN", verdict4="판정 불가", note=untested))
            return
        r, _ = cpool(tms, names, A, B, n, item, is_rand, nboot, reg or len(names))
        for q in r:
            q["item"] = item
        rows.extend(r)
    add("S-XC0", PE1, "A1", "A2", 10)
    for n in (10, 40, 160):
        add("S-XC1", PE1, "A5", "A2", n)
    for n in (40, 160):
        add("S-XC2", PE1, "A1+", "A1", n)
        add("S-XC3", PE1, "A4", "A2", n)
    for n in (10, 40, 160):
        add("S-XC5", PE1, "A1", "A6", n)
        add("S-XC5", PE1, "A2", "A6", n)
        add("S-XC10", PE1, "A1", "P0", n)
    for nm_pool, names in (("PE2", PE2), ("AUX3", AUX3)):
        for n in (10, 40, 160, -1):
            for A, B in (("A1", "A2"), ("A1", "A3"), ("A1", "A5")):
                if n == -1 and B == "A5":
                    continue
                add(f"S-XC8|{nm_pool}", names, A, B, n)
    for A, B in (("A1", "A2"), ("A1", "A3")):
        add("S-XC8|PE1", PE1, A, B, -1)
    aux = XB.clean_rows(rows)
    if len(aux) and "noninf" in aux:
        aux["s_xc10_noninf"] = np.where(aux["item"] == "S-XC10", aux["noninf"], False)
    out = dict(contrasts_aux=aux)
    out["f2_counts"] = f2_counts(tms, is_rand, nboot)
    out["harm_a"] = harm_a(tms, is_rand)
    out["diag"] = diag_table(tms, units, is_rand)
    out["labeleq"] = label_equivalence(tms, units, is_rand, floors)
    out["alloc"] = allocation_analysis(tms, units, is_rand)
    out["s12"] = s12_rho(tms, units, is_rand, nboot)
    return out


def f2_counts(tms, is_rand, nboot):
    """F2 하위 지역(대상 수로 서술): 주 대비마다 우세 k, 열세 m, 동등 e, 미결정 j, 판정 불가. detail 의 대상별 기록에 '소수 블록'(사용 분할의
    최소 채점 블록 < 5, 부록 A 의 AL-1·AL-3·CA-2·CA-3·LE-1 행)과 '선택 계열'(AL-k)을 붙인다. 같은 키 대비(Algorithm P = S1)는 세지 않는다."""
    out = []
    for A, B, ns in (("A1", "A2", (10, 40, 160)), ("A1", "A3", (10, 40, 160)), ("A1", "A5", (40, 160))):
        for n in ns:
            same = untested_reason(A, B, n, is_rand)
            if same:
                out.append(dict(contrast=f"{A}-{B}", n=n, n_targets=0, **{k: 0 for k in XB.BRANCHES}, detail="", few_block_targets="", note=same))
                continue
            c_ = Counter()
            names, few = [], []
            for nm in F2_NAMES:
                s = region_stat(tms.get(nm), A, B, n, is_rand, nboot)
                if s is None:
                    continue
                lo, hi = X._ci(s.get("dist")); lob, hib = X._ci(s.get("dist_beq"))
                v = XB.verdict4(lo, hi, lob, hib) if s.get("dist") is not None else "판정 불가"
                c_[v] += 1
                marks = [m_ for m_, f in ((XB.FEW_BLOCKS_TXT, int(s.get("n_blocks_min", 0)) < XB.FEW_BLOCKS), (SEL_FAMILY_TXT, bool(selection_family(nm))))
                         if f]
                if int(s.get("n_blocks_min", 0)) < XB.FEW_BLOCKS:
                    few.append(nm)
                names.append(f"{nm}:{v}" + (f"({', '.join(marks)})" if marks else ""))
            XB.boot_cache_clear()
            out.append(dict(contrast=f"{A}-{B}", n=n, n_targets=len(names), **{k: int(c_.get(k, 0)) for k in XB.BRANCHES}, detail=";".join(names),
                            few_block_targets=",".join(few), note=""))
    return pd.DataFrame(out)


def harm_a(tms, is_rand):
    """S-XC4 (a): P0 보다 2 cm 넘게 나빠진 대상 수(셀 가중 점 추정, WF0 위험표와 같은 정의). A1, A2, A3, n 10·40·160·전량."""
    out = []
    for scope, names in (("F1", [f"{t}|{m}" for t, m in F1]), ("F2", F2_NAMES), ("all", sorted(tms))):
        for n in (10, 40, 160, -1):
            for arm in ("A1", "A2", "A3"):
                hit, k = [], 0
                for nm in names:
                    tm = tms.get(nm)
                    g = arm_key(arm, n, is_rand, tm)
                    if tm is None or not has_group(tm, g):
                        continue
                    k += 1
                    d = XB.grp_rmse2(tm, g)[0] - XB.grp_rmse2(tm, H.P0_GRP)[0]
                    if np.isfinite(d) and d > HARM_CM:
                        hit.append(nm)
                out.append(dict(item="S-XC4a", scope=scope, arm=arm, n=n, n_targets=k, n_worse_2cm=len(hit), targets=",".join(hit)))
    return pd.DataFrame(out)


def _diag_of(units):
    by = {}
    for u in units:
        nm = f"{u.get('target')}|{u.get('mode')}"
        for e in u.get("diag") or []:
            by.setdefault(nm, []).append(dict(e, split=int(u.get("split", e.get("split", -1)))))
    return by


def diag_table(tms, units, is_rand):
    """S-XC6·S-XC7: 대상별 |b10|(A1 라벨, A2 라벨), 추출 사이 표준편차(분할 안 ddof 1 의 분할 평균), P0 − A1@최대 유한 n, P0 − A1@전량,
    |b10| ≥ 8 cm 여부(사후 임계값, WF4-c 관찰)."""
    by = _diag_of(units)
    out = []
    pl_a1 = PL_RAND if is_rand else PL_P
    for nm in sorted(set(by) | set(tms)):
        ds = pd.DataFrame(by.get(nm, []))
        row = dict(target=nm)
        for pl, lab in ((pl_a1, "A1"), (PL_RAND, "A2")):
            q = ds[ds.placement == pl] if len(ds) else ds
            row[f"absb10_{lab}"] = float(q.abs_bias.mean()) if len(q) else np.nan
            row[f"b10_{lab}"] = float(q.bias.mean()) if len(q) else np.nan
            sd = [float(np.std(g.bias.values, ddof=1)) for _, g in q.groupby("split") if len(g) >= 2] if len(q) else []
            row[f"sd_b10_{lab}"] = float(np.mean(sd)) if sd else np.nan
            row[f"n_diag_{lab}"] = int(len(q))
        tm = tms.get(nm)
        if tm is not None:
            p0c, p0b = XB.grp_rmse2(tm, H.P0_GRP)
            ns = [n for n in (160, 40, 10) if has_group(tm, arm_key("A1", n, is_rand, tm))]
            nmax = ns[0] if ns else None
            if nmax is not None:
                c_, b_ = XB.grp_rmse2(tm, arm_key("A1", nmax, is_rand, tm))
                row.update(n_max=nmax, gain_maxn=p0c - c_, gain_maxn_beq=p0b - b_)
            if has_group(tm, arm_key("A1", -1, is_rand, tm)):
                c_, b_ = XB.grp_rmse2(tm, arm_key("A1", -1, is_rand, tm))
                row.update(gain_all=p0c - c_, gain_all_beq=p0b - b_)
        row["absb10_ge8"] = bool(np.isfinite(row.get("absb10_A1", np.nan)) and row["absb10_A1"] >= DIAG_THRESHOLD_CM)
        row["note"] = SENTENCES["S-XC6"]
        out.append(row)
    return pd.DataFrame(out)


def _mean_nA(units, nm):
    v = [float(u.get("n_A", np.nan)) for u in units if f"{u.get('target')}|{u.get('mode')}" == nm]
    return float(np.nanmean(v)) if v else np.nan


def _interp_neq(ns, vals, v):
    """log n 선형 보간으로 곡선 vals(n)이 v 에 닿는 n. 첫 교차점, 범위 밖이면 NaN."""
    x = np.log(np.asarray(ns, float)); y = np.asarray(vals, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or not np.isfinite(v) or v < y.min() or v > y.max():
        return np.nan
    for j in range(len(x) - 1):
        y0, y1 = y[j], y[j + 1]
        if (y0 - v) * (y1 - v) <= 0:
            if y1 == y0:
                return float(np.exp(x[j]))
            t = (v - y0) / (y1 - y0)
            return float(np.exp(x[j] + t * (x[j + 1] - x[j])))
    return np.nan


def label_equivalence(tms, units, is_rand, floors=None):
    """S-XC11: (a) 라벨 등가 n_eq(A2 곡선을 log n 으로 선형 보간해 A1@n 의 RMSE 에 닿는 라벨 수, 범위 밖이면 '범위 밖'). 격자 점 3개 이상인 대상
    (레나, 캐나다, 알래스카 x, 티베트). 전량의 n 은 분할 평균 |A|. (b) 줄일 수 있는 오차의 감소 비율 = (A2@n − A1@n)/(A2@n − floor)(셀 가중,
    floor = lgx_floor.csv 의 eval 행, 알래스카·레나·캐나다). 둘 다 서술이다."""
    fl = floors if floors is not None else load_floors()
    out = []
    for nm in LABEQ_TARGETS:
        tm = tms.get(nm)
        if tm is None:
            continue
        nA = _mean_nA(units, nm)
        pts = [(n, n if n > 0 else nA) for n in (10, 40, 160, -1) if has_group(tm, arm_key("A2", n, is_rand, tm))]
        if len(pts) < 3:
            continue
        for w_i, wname in ((0, "cell"), (1, "beq")):
            curve = [(nn, XB.grp_rmse2(tm, arm_key("A2", n, is_rand, tm))[w_i]) for n, nn in pts]
            xs, ys = [p[0] for p in curve], [p[1] for p in curve]
            for n, _ in pts:
                g1 = arm_key("A1", n, is_rand, tm)
                if not has_group(tm, g1):
                    continue
                v1 = XB.grp_rmse2(tm, g1)[w_i]
                v2 = XB.grp_rmse2(tm, arm_key("A2", n, is_rand, tm))[w_i]
                neq = _interp_neq(xs, ys, v1)
                row = dict(item="S-XC11a", target=nm, weight=wname, n=n, n_eq=neq, out_of_range=bool(not np.isfinite(neq)),
                           note="범위 밖" if not np.isfinite(neq) else "", rmse_A1=v1, rmse_A2=v2)
                fr = fl.get(FLOOR_REGIONS.get(nm, ""), np.nan)
                if wname == "cell" and np.isfinite(fr):
                    den = v2 - fr
                    row.update(floor_cm=fr, reducible_ratio=(v2 - v1) / den if den > 0 else np.nan)
                out.append(row)
    return pd.DataFrame(out)


def load_floors(path=FLOOR_FILE) -> dict:
    try:
        f = pd.read_csv(path)
        f = f[f.scope == "eval"]
        return {str(r.region): float(r.floor_rmse_cm) for r in f.itertuples()}
    except (OSError, ValueError, AttributeError):
        return {}


def allocation_analysis(tms, units, is_rand, pools=(("PE1", PE1), ("AUX3", AUX3)), budgets=(40, 160)):
    """S-XC9(서술): 지역 사이 예산 배분. 지역 수 R, 지역당 평균 예산 B̄ 에서 총 예산 T = R·B̄. 균등 = 지역마다 B̄. |b10| 비례 = 지역마다 n 10 을 쓴 뒤
    남은 T − 10R 을 |b10|(A1 라벨) 비례로 나눈다. 곡선이 격자 n 위에서만 정의되므로 배분값을 그 지역에서 쓸 수 있는 격자 n 가운데 log 거리가 가장
    가까운 값으로 맞추고(상한 = 그 지역의 최대 유한 격자 n) 실제 배정 합을 함께 적는다. 층화 평균 = 지역 A1 RMSE(셀 가중, 블록 등가중)의 비가중 평균."""
    by = _diag_of(units)
    pl_a1 = PL_RAND if is_rand else PL_P
    out = []
    for pname, names in pools:
        have = [nm for nm in names if nm in tms]
        absb = {}
        for nm in have:
            q = [e["abs_bias"] for e in by.get(nm, []) if e.get("placement") == pl_a1]
            absb[nm] = float(np.mean(q)) if q else np.nan
        regs = [nm for nm in have if np.isfinite(absb[nm])]
        if len(regs) < 2:
            continue
        for Bbar in budgets:
            T = Bbar * len(regs)
            tot = sum(absb[nm] for nm in regs)
            for rule in ("equal", "absb10"):
                ns, rc, rb = [], [], []
                for nm in regs:
                    tm = tms[nm]
                    avail = [n for n in (10, 40, 160) if has_group(tm, arm_key("A1", n, is_rand, tm))]
                    if not avail:
                        continue
                    want = Bbar if rule == "equal" else 10 + (T - 10 * len(regs)) * (absb[nm] / tot if tot > 0 else 1.0 / len(regs))
                    n_ = min(avail, key=lambda v: (abs(np.log(v) - np.log(max(want, 1e-9))), v))
                    c_, b_ = XB.grp_rmse2(tm, arm_key("A1", n_, is_rand, tm))
                    ns.append(f"{nm}:{n_}"); rc.append(c_); rb.append(b_)
                out.append(dict(item="S-XC9", pool=pname, budget_per_region=Bbar, total_budget=T, rule=rule, n_regions=len(rc),
                                allocation=";".join(ns), realized_total=int(sum(int(v.split(":")[-1]) for v in ns)),
                                strat_rmse_cell=float(np.mean(rc)) if rc else np.nan, strat_rmse_beq=float(np.mean(rb)) if rb else np.nan))
    return pd.DataFrame(out)


def family_of(name) -> str:
    """S-XC12 의 계열(군집 단위): 알래스카(알래스카 x, AL-k), 캐나다(캐나다 x, CA-k), 레나(레나 x, LE-k), 그 밖은 대상 자신."""
    t = str(name).split("|")[0]
    base = t.split("~")[0]
    if base == "Alaska" or t.startswith("AL-"):
        return "Alaska"
    if base == "Canada" or t.startswith("CA-"):
        return "Canada"
    if base == "Lena" or t.startswith("LE-"):
        return "Lena"
    return t


def _spearman_vec(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def s12_rho(tms, units, is_rand, nboot=None, seed_tag="xc-s12"):
    """S-XC12(수치만): ρ(|b10|_A1, (RMSE(P1@전량) − RMSE(A1@전량)) / RMSE(P0)). CI = 계열 군집 재표집(계열 복원 추출)과 블록 재표집(이득의
    h40.contrast 분포)과 |b10| 의 (분할, 추출) 재표집을 같은 번호 b 로 묶은 결합 재표집. 셀 가중·블록 등가중 따로."""
    by = _diag_of(units)
    pl_a1 = PL_RAND if is_rand else PL_P
    recs = []
    for nm in sorted(tms):
        tm = tms[nm]
        if not tm.has_ci:
            continue
        ab = np.array([e["abs_bias"] for e in by.get(nm, []) if e.get("placement") == pl_a1], float)
        gA, gB = arm_key("A1", -1, is_rand, tm), arm_key("A3", -1, is_rand, tm)
        if not len(ab) or not has_group(tm, gA) or not has_group(tm, gB):
            continue
        r = H.contrast(tm, gA, gB, return_dist=True)
        if r is None or "dist" not in r:
            continue
        p0c, p0b = XB.grp_rmse2(tm, H.P0_GRP)
        recs.append(dict(name=nm, fam=family_of(nm), ab=ab, gc=-float(r["delta"]) / p0c, gb=-float(r["delta_beq"]) / p0b,
                         dc=-np.asarray(r["dist"], float) / p0c, db=-np.asarray(r["dist_beq"], float) / p0b))
    XB.boot_cache_clear()
    if len(recs) < 3:
        return pd.DataFrame([dict(item="S-XC12", n_targets=len(recs), note="대상 3 미만")])
    fams = sorted({q["fam"] for q in recs})
    B = int(min(len(q["dc"]) for q in recs))
    rng = np.random.RandomState(seed_of(seed_tag, len(recs)))
    pick_f = rng.randint(0, len(fams), size=(B, len(fams)))
    rngs = {q["name"]: np.random.RandomState(seed_of(seed_tag, q["name"])) for q in recs}
    abs_b = {q["name"]: q["ab"][rngs[q["name"]].randint(0, len(q["ab"]), size=(B, len(q["ab"])))].mean(1) for q in recs}
    idx_f = {f: [j for j, q in enumerate(recs) if q["fam"] == f] for f in fams}
    out = []
    for wname, gk_, dk in (("cell", "gc", "dc"), ("beq", "gb", "db")):
        x0 = np.array([q["ab"].mean() for q in recs]); y0 = np.array([q[gk_] for q in recs])
        rho = _spearman_vec(x0, y0)
        rb, nf = np.full(B, np.nan), np.zeros(B, int)
        for b in range(B):
            js = [j for f in pick_f[b] for j in idx_f[fams[f]]]
            nf[b] = len(set(pick_f[b].tolist()))
            xb = np.array([abs_b[recs[j]["name"]][b] for j in js]); yb = np.array([recs[j][dk][b] for j in js])
            rb[b] = _spearman_vec(xb, yb) if len(js) >= 3 else np.nan
        lo, hi = (float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))) if np.isfinite(rb).any() else (np.nan, np.nan)
        out.append(dict(item="S-XC12", weight=wname, n_targets=len(recs), n_families=len(fams), rho=rho, ci_lo=lo, ci_hi=hi, nboot=B,
                        frac_lt3_families=float(np.mean(nf < 3)), note="수치만(판정어 없음), 척도 없는 판(XA 와 같은 정의)"))
    return pd.DataFrame(out)


def tests_xcr(tms, nboot=None, is_rand=False):
    """XC-r(서술, 4분 판정, Holm 가족 밖): 알래스카·레나·캐나다(모드 r)와 3대상 층화. 분할 완결성 표지는 h42.pool_rows 의 splits_short."""
    rows = []
    grid = sorted({int(g[4]) for tm in tms.values() for gd in tm.idx.values() for g in gd if g[0] == "W"})
    for n in grid:
        for A, B in (("rA4", "rA2"), ("rA5", "rA2"), ("rA2", "rA3"), ("rA5", "rA3"), ("rA4", "rA3")):
            if is_rand and "rA4" in (A, B):
                rows.append(dict(item="XC-r", contrast=f"{A}-{B}", n=n, scope="MEAN", verdict4="판정 불가", note="시험하지 않음(Algorithm P = S1)"))
                continue
            r, _ = cpool(tms, INREGION3, A, B, n, f"XC-r|{A}-{B}|n{n}", is_rand, nboot, len(INREGION3))
            for q in r:
                q["item"] = "XC-r"
            rows.extend(r)
    return dict(xcr_contrasts=mark_family(XB.clean_rows(rows)))


def resolve_f3_status(o, has_f3_shards) -> str:
    """부록 XC-F3 상태(pending, none, final). auto 는 xcf3 조각이 있으면 final, 없으면 pending. 조각과 맞지 않는 명시값은 거부한다."""
    st = str(getattr(o, "f3_status", "auto") or "auto")
    if st == "auto":
        return "final" if has_f3_shards else "pending"
    if st == "final" and not has_f3_shards:
        raise SystemExit("[거부] --f3-status final 인데 xcf3 조각이 없다(R4 의 XC-F3 조각을 받은 뒤 집계한다)")
    if st in ("none", "pending") and has_f3_shards:
        raise SystemExit(f"[거부] --f3-status {st} 인데 xcf3 조각이 있다(부록 XC-F3 과 조각 폴더를 확인한다)")
    return st


def sealed_names(o, tables, holm_final) -> tuple:
    """봉인 파일 이름. 기본 '<tag>_<표>.csv|json'. Holm 표가 잠정(부록 XC-F3 전)이면 Holm 의존 표(hyp, holm, sentences)에 '_provisional',
    스모크 밖에서 재표집 수가 등록값과 다르면 모든 파일에 '_nboot<N>' 을 붙인다. 반환 (이름 → 표, 이탈 기록 dict)."""
    pre = tag_of(o, "xc")
    nb = int(o.nboot)
    off_reg = (not o.smoke) and nb != int(XB.NBOOT)
    nbs = f"_nboot{nb}" if off_reg else ""
    named = {}
    for k, v in tables.items():
        if str(k).startswith("_"):
            continue
        prov = "_provisional" if (not holm_final and k in PROVISIONAL_TABLES) else ""
        named[f"{pre}_{k}{prov}{nbs}.{'csv' if isinstance(v, pd.DataFrame) else 'json'}"] = v
    dev = dict(nboot=nb, nboot_registered=int(XB.NBOOT), nboot_requested=int(getattr(o, "nboot_arg", nb)),
               registration_deviation=(f"재표집 {nb}회(등록 {XB.NBOOT}회)" if off_reg else ""),
               provisional_files=sorted(n for n in named if "_provisional" in n))
    return named, dev


def summarize(o):
    """조각 → 표(봉인 폴더). 화면에는 대상 수, 조각 수, 표 이름, 행 수, sha256 만 쓴다."""
    nb = int(o.nboot)
    tables = {}
    tms, units, _ = XB.load_tms(o.SHARDS, tag_of(o, "xc"), nb, allow_mixed=o.allow_mixed_cfg)
    tms3, units3, _ = XB.load_tms(o.SHARDS, tag_of(o, "xcf3"), nb, allow_mixed=o.allow_mixed_cfg)
    f3_status = resolve_f3_status(o, bool(tms3))
    print(f"[summarize] XC 저장소 {len(tms)} · 조각 {len(units)} · F3 저장소 {len(tms3)} · 부록 XC-F3 상태 {f3_status} · 재표집 {nb}", flush=True)
    if not o.smoke and nb != int(XB.NBOOT):
        print(f"[summarize] 재표집 {nb}회는 등록값({XB.NBOOT}회)과 다르다. 봉인 파일 이름에 _nboot{nb} 을 붙이고 meta 에 등록 이탈을 적는다"
              f"{'(허용 표지 없는 로컬 집계의 상한)' if not o.PERMIT else ''}", flush=True)
    state = dict(holm_final=None, f3_status=f3_status)
    if tms or tms3:
        f3_names = sorted(tms3)
        allt = dict(tms)
        allt.update(tms3)
        tables.update(tests_xc(allt, units + units3, o.algp, f3_names, nb, f3_status=f3_status))
        state = tables.get("_state", state)
    tmr, unitsr, _ = XB.load_tms(o.SHARDS, tag_of(o, "xcr"), nb, allow_mixed=o.allow_mixed_cfg)
    print(f"[summarize] XC-r 저장소 {len(tmr)} · 조각 {len(unitsr)}", flush=True)
    if tmr:
        tables.update(tests_xcr(tmr, nb, bool(o.algp.get("is_rand"))))
    holm_final = bool(state.get("holm_final"))
    named, dev = sealed_names(o, tables, holm_final)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), plan=f"{XB.PLAN_DOC}@{XB.PLAN_COMMIT}", algp=o.algp, placement_impl=o.placement_impl,
                placement_sha=o.placement_sha, n_units_xc=len(units), n_units_f3=len(units3), n_units_xcr=len(unitsr),
                holm_m=HOLM_M, holm_final=holm_final, f3_status=f3_status, f3_regions=state.get("f3_regions", []),
                code_sha=XB.code_sha(__file__), max_rss_mb=XB.max_rss_mb(), **dev)            # dev 가 nboot(= nb)·등록 이탈 표지를 담는다
    meta_name = next(iter(sealed_names(o, {"meta": meta}, True)[0]))
    named[meta_name] = meta
    XB.write_sealed(EXP_NAME, named, root=sealed_root(o))
    if not holm_final:
        print(f"[summarize] 부록 XC-F3 전: Holm 의존 표 {len(dev['provisional_files'])}개를 _provisional 로 봉인했다(holm_final = false)", flush=True)
    return named


def sealed_root(o) -> Path:
    """봉인 폴더의 뿌리. 기본 산출 경로(data/processed/xbatch/XC_workflow_end_to_end)이면 data/processed/xbatch(→ .../XC_workflow_end_to_end/sealed),
    그 밖의 --out-dir 이면 그 안(→ <out-dir>/XC_workflow_end_to_end/sealed)."""
    return o.OUT.parent if o.OUT.name == EXP_NAME else o.OUT


# ================================================================ 8. 세기·관문·점검
def count_only(o, units, skipped):
    """세기 범주(라벨 값 미사용): 문맥의 라벨을 지운 뒤 dry 단위로 적합 수를 센다. 화면에는 단위·적합 수·추정 시간만 쓴다."""
    t0 = time.time()
    rows, det = [], Counter()
    for u in units:
        r = run_unit(o, *u, dry=True)
        for k, v in r.pop("_detail").items():
            det[(u[0], u[2], k)] += v
        rows.append(r)
    df = pd.DataFrame(rows)
    if not len(df):
        print("[count-only] 작업 단위 없음", flush=True)
        return df
    meas = []
    for (part, mode, k), v in det.items():
        meth = k.split("|")[-1]
        base = ("R1" if meth.startswith(("R1", "StackR")) else ("R2" if meth.startswith("R2") else ("D1" if meth.startswith("D1") else meth)))
        # StackR 의 학습 행 = 원천 행 + 선택 라벨(R1 과 같은 구성)이라 R1 의 실측 초를 쓴다. W+ 최종 StackR 적합(W+ 가 적층을 고른 칸만)은
        # 선택 결과에 따라 정해져 dry 세기에 들어가지 않는다(칸마다 최대 seed 수만큼 더해진다)
        spf = MEAS_SPF.get(("r" if part == "xcr" else "x", base), np.nan)
        meas.append(dict(part=part, mode=mode, method=meth, n_fit=int(v), sec_per_fit=spf, est_h=v * spf / 3600.0))
    meas = pd.DataFrame(meas)
    XB.check_out_dir(o.OUT)
    o.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(o.OUT / f"{tag_of(o, 'xc')}_count.csv", index=False)
    meas.to_csv(o.OUT / f"{tag_of(o, 'xc')}_count_detail.csv", index=False)
    by = df.groupby(["part", "target", "mode"]).agg(units=("split", "size"), n_A_min=("n_A", "min"), n_A_max=("n_A", "max"), fits=("fit_total", "sum"),
                                                    est_lg_h=("est_s", "sum"))
    by["est_lg_h"] = (by.est_lg_h / 3600).round(3)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(f"[count-only] 작업 단위 {len(units)} · 건너뛴 분할 {len(skipped)} · 세기 {time.time() - t0:.0f}s · Algorithm P {o.algp['name']}"
              f"({o.algp.get('source', '')}) · 배치 구현 {o.placement_impl} · W+ {o.wplus_status}", flush=True)
        print(by.to_string(), flush=True)
        mm = meas.groupby(["part", "method"]).agg(n_fit=("n_fit", "sum"), est_h=("est_h", "sum")).round(3)
        print("[count-only] 방법별 적합 수와 실측 기반 추정 시간(4스레드 워커, WF4·WF6 실측 초/적합)", flush=True)
        print(mm.to_string(), flush=True)
    tot = int(df.fit_total.sum())
    for part in sorted(df.part.unique()):
        q, qm = df[df.part == part], meas[meas.part == part]
        h = float(qm.est_h.sum())
        print(f"[count-only] {part}: 단위 {len(q)} · 적합 {int(q.fit_total.sum()):,} · 실측 기반 {h:.2f} 워커·시간 · 워커 {RESCALE_WORKERS}개 약 "
              f"{h / RESCALE_WORKERS:.2f} h · LG 모형 추정 {float(q.est_s.sum()) / 3600:.2f} 워커·시간", flush=True)
    h_all = float(meas.est_h.sum())
    print(f"[count-only] 합계 적합 {tot:,} · 실측 기반 {h_all:.2f} 워커·시간 · 워커 {RESCALE_WORKERS}개 약 {h_all / RESCALE_WORKERS:.2f} h"
          f"(단위 크기 불균형·설치·2단 재표집 집계 제외) → {tag_of(o, 'xc')}_count.csv", flush=True)
    if skipped:
        print(f"[count-only] 건너뛴 분할 사유: {dict(Counter(s['status'] for s in skipped))}", flush=True)
    return df


def gate_check(o, ref_dir, level="same_node"):
    """재현 관문: 관문 조각(<tag>gate)과 WF4 조각(wf4)의 공통 키 가운데 W 와 R1(λ 0.25)의 블록 SSE 대조. 표에는 RMSE·Δ 가 없다."""
    ref = GATE_REF_DEFAULT if ref_dir in ("", "default") else (Path(ref_dir) if os.path.isabs(ref_dir) else ROOT / ref_dir)
    new_sh = XB.find_shards(o.SHARDS, tag_of(o, "gate"))
    if not new_sh:
        raise SystemExit(f"[gate] 관문 조각이 없다: {o.SHARDS}/{tag_of(o, 'gate')}__*")
    ref_npz = [ref / f"{GATE_REF_TAG}__cpu__{s['target']}__{s['mode']}__s{s['split']}_blocksse.npz" for s in new_sh]
    miss = [str(p) for p in ref_npz if not p.exists()]
    if miss:
        raise SystemExit(f"[gate] 기준 조각이 없다: {miss[:3]}")
    new = H4.load_stores([s["npz"] for s in new_sh])
    old = H4.load_stores(ref_npz)

    def key_fn(k):
        return k[0] == "W" or (k[0] == "R1" and abs(float(k[7]) - LAM_BASE) < 1e-12)
    df = XB.gate_compare(new, old, level, key_fn=key_fn)
    s = XB.gate_summary(df)
    XB.check_out_dir(o.OUT)
    o.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(o.OUT / f"{tag_of(o, 'gate')}_gate_{level}.csv", index=False)
    print(f"[gate] 수준 {level} · 공통 키 {s['n_keys']} · 실패 키 {s['n_fail']} · 통과 {s['passed']} → {tag_of(o, 'gate')}_gate_{level}.csv", flush=True)
    o.algp = o.algp or resolve_algp(o, need_file=False)                     # 2.3절 관문의 Algorithm P 결정성 항목(라벨 미사용)
    det = placement_determinism(o, shards_dir=o.R2A_SHARDS, out_name=f"{tag_of(o, 'gate')}_placement_determinism.csv")
    s.update(sse_passed=bool(s["passed"]), placement_determinism=det)
    s["passed"] = bool(s["passed"] and det["passed"])
    print(f"[gate] 관문 통과(블록 SSE 와 Algorithm P 결정성) {s['passed']}", flush=True)
    return s


def _shard_heads(shards_dir, tag) -> dict:
    """R2a 조각 unit.json 의 orders_head(배치 순서의 앞 10개 색인, 라벨 미사용)만 읽는다. 키 (대상, 모드, 분할) → {추출: 색인 목록}."""
    out = {}
    for s_ in XB.find_shards(shards_dir, tag):
        try:
            u = json.loads(Path(s_["unit"]).read_text())
        except (OSError, ValueError):
            continue
        out[(s_["target"], s_["mode"], int(s_["split"]))] = {str(k): [int(v) for v in vs] for k, vs in (u.get("orders_head") or {}).items()}
    return out


def placement_determinism(o, shards_dir=None, targets=None, draws=None, out_name="") -> dict:
    """재현 관문의 'Algorithm P 결정성(같은 seed 두 번 실행 순서 일치)'(2.3절): XC 대상마다 분할 계획(201–210)의 실행 분할과 추출 d 에서
    placement_order 를 두 번(표준화부터 다시) 계산해 같은지 보고, R2a 조각(shards_dir 의 tag xc)이 있으면 그 orders_head 와 앞 10개가 같은지
    본다. 일정과 seed 는 XCUnit 과 같다(등록 격자 가운데 |A| 미만 n, seed_of('xc-P', 대상, 모드, 분할, d)). 라벨을 쓰지 않는다.
    화면에는 건수만 쓴다. 반환 dict(n_cases, n_same, n_head, n_head_same, passed)."""
    algp = o.algp or resolve_algp(o, need_file=False)
    if algp.get("is_rand"):
        print("[placement-check] Algorithm P = S1(무작위 중첩 라벨): 배치 순서가 없어 결정성 항목은 해당 없음", flush=True)
        return dict(n_cases=0, n_same=0, n_head=0, n_head_same=0, passed=True, algp=algp["name"], note="S1")
    a = o.ha
    heads = _shard_heads(shards_dir, tag_of(o, "xc")) if shards_dir is not None else {}
    nd = int(o.D_XC if draws is None else draws)
    rows = []
    for al, m in (targets or o.T_XC):
        keep, _ = split_plan_201(a, al, "xc")
        for sp in keep:
            c = XB.build_tctx(a, al, m, sp, _plan_row(a, "xc", al, sp))
            sched = [n for n in o.G_XC if 0 < n < len(c.XA)]
            hd = heads.get((al, m, int(sp)), {})
            for d in range(nd):
                s_ = seed_of(P_TAG, c.target, c.mode, c.split, d)
                o1 = placement_order(algp, zstd(c.XA), c.blkA, sched, s_, o.placement_impl)
                o2 = placement_order(algp, zstd(c.XA), c.blkA, sched, s_, o.placement_impl)
                h = hd.get(str(d))
                rows.append(dict(target=al, mode=m, split=int(sp), draw=int(d), n_A=int(len(c.XA)), len_order=int(len(o1)),
                                 same_twice=bool(np.array_equal(o1, o2)), head_in_shard=h is not None,
                                 head_same=bool(h is not None and [int(v) for v in o1[:len(h)]] == list(h))))
    df = pd.DataFrame(rows)
    n_head = int(df.head_in_shard.sum()) if len(df) else 0
    res = dict(n_cases=int(len(df)), n_same=int(df.same_twice.sum()) if len(df) else 0, n_head=n_head,
               n_head_same=int(df.head_same.sum()) if len(df) else 0, algp=algp["name"], placement_impl=o.placement_impl)
    res["passed"] = bool(res["n_cases"] > 0 and res["n_same"] == res["n_cases"] and res["n_head_same"] == n_head)
    if out_name:
        XB.check_out_dir(o.OUT)
        o.OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(o.OUT / out_name, index=False)
    print(f"[placement-check] Algorithm P {algp['name']} 결정성: 두 번 계산 일치 {res['n_same']}/{res['n_cases']} · R2a 조각 orders_head 일치 "
          f"{res['n_head_same']}/{n_head} · 통과 {res['passed']}" + (f" → {out_name}" if out_name else ""), flush=True)
    return res


def placement_check(o):
    """R2a 제출 전 배치 점검(라벨 미사용). (1) XD 배치 함수와 참조 구현의 순서 대조: 대상마다 첫 실행 분할, 추출 0–1, 후보 S2·S4·S8 계열.
    (2) 부록 XC-0 의 Algorithm P 결정성: 분할 201–210 의 실행 분할·추출마다 두 번 계산한 순서의 일치(R2a 조각이 있으면 orders_head 대조)."""
    xd = xd_placement(strict=(o.placement_impl_arg == "xd"))
    if xd is None:
        print(f"[placement-check] {XD_MODULE} 의 배치 함수가 없다. 참조 구현만 있다", flush=True)
        return 1
    a = o.ha
    n_eq = n_all = 0
    for al, m in o.T_XC:
        keep, _ = split_plan_201(a, al, "xc")
        if not keep:
            continue
        c = XB.build_tctx(a, al, m, keep[0], _plan_row(a, "xc", al, keep[0]))
        Z = zstd(c.XA)
        sched = [n for n in o.G_XC if 0 < n < len(c.XA)]
        for name in ALGP_CANDIDATES:
            sp = algp_spec(name)
            for d in (0, 1):
                s_ = seed_of(P_TAG, c.target, c.mode, c.split, d)
                eq = np.array_equal(placement_order(sp, Z, c.blkA, sched, s_, "xd"), placement_order(sp, Z, c.blkA, sched, s_, "ref"))
                n_all += 1; n_eq += int(eq)
        print(f"  [placement-check] {al}|{m}|s{keep[0]}: 누적 일치 {n_eq}/{n_all}", flush=True)
    print(f"[placement-check] 순서 일치 {n_eq}/{n_all}", flush=True)
    o.algp = o.algp or resolve_algp(o, need_file=False)
    det = placement_determinism(o, shards_dir=o.R2A_SHARDS, out_name=f"{tag_of(o, 'xc')}_placement_determinism.csv")
    return 0 if (n_eq == n_all and det["passed"]) else 1


def print_split_plan(o):
    a = o.ha
    fams = [("xc", al) for al, _ in o.T_XC] + [("xcr", t) for t in o.T_XCR] + [("xc", al) for al in o.F3]
    seen = set()
    for fam, al in fams:
        if (fam, al) in seen:
            continue
        seen.add((fam, al))
        keep, rows = split_plan_201(a, al, fam)
        s = XB.plan_summary(rows)
        ex = ", ".join(f"{sp}({st})" for sp, st in sorted(s["excluded"].items()))
        f3 = f" · F3 구조 적격 {f3_eligible(rows)}" if al in o.F3 else ""
        print(f"  [{fam}] {al}: 유효 분할 {s['n_valid']}/{s['n_seeds']} · 뺀 seed {ex or '없음'} · |A| {s['n_A']} · 채점 블록 {s['nb_eval']} · "
              f"합집합 {s['nb_union']}{f3}", flush=True)
    return 0


# ================================================================ 9. 실행
def limit_malloc_arenas(n=2) -> bool:
    """glibc malloc 아레나 수 상한(로컬 실행 전용). CatBoost 적합을 되풀이하면 스레드별 아레나 예약으로 가상 주소 공간이 10 GB 를 넘고
    (실측 VmPeak 10.8 GB, RSS 0.24 GB), RLIMIT_AS 10 GB(계획 1절 prlimit --as) 아래에서는 CatBoost 가 멈춘다. 아레나 2개면 VmPeak 2.8 GB 였다.
    자식 프로세스(spawn)를 위해 환경 변수도 정한다."""
    os.environ["MALLOC_ARENA_MAX"] = str(int(n))
    try:
        import ctypes
        return bool(ctypes.CDLL("libc.so.6").mallopt(-8, int(n)))           # M_ARENA_MAX = −8
    except (OSError, AttributeError):
        return False


def set_local_nice(target=LOCAL_NICE) -> int:
    """로컬 실행의 nice 값을 target 이상으로 맞춘다(1절 'nice 10'). os.nice 는 증분이라 현재 값과의 차이만 더한다. 반환 맞춘 뒤의 값."""
    try:
        cur = os.nice(0)
        return os.nice(int(target) - cur) if cur < int(target) else cur
    except OSError:
        return -1


def main(argv=None):
    o = parse_args(argv)
    if o.count_only or o.split_plan or o.placement_check:
        XB.guard("count")
    elif o.summarize_only or o.gate_check:
        XB.guard("summarize")
    elif o.smoke:
        XB.guard("smoke")
    else:
        XB.guard("run", o.ARGV)
    enforce_main_impl(o)                                                   # 본 실행·집계: --placement-impl xd, --wplus require
    if not o.PERMIT:
        set_local_nice(LOCAL_NICE)
        if o.smoke:                                                        # 5절 1a: 코어 4개, 2스레드(시작 환경으로 확인, 어긋나면 중단)
            rep = XB.smoke_env_report()
            if rep["warn"]:
                raise SystemExit(f"[거부] 로컬 스모크 규약(코어 4개, OMP_NUM_THREADS ≤ 2)을 어겼다: {'; '.join(rep['warn'])}")
    for v in XB.THREAD_VARS:
        os.environ[v] = str(o.threads)
    if not o.PERMIT:
        XB.require_memory(MEM_MIN_GB, wait_s=MEM_WAIT_S, poll_s=MEM_POLL_S)  # 1절: 가용 30 GB 아래면 기다린다
        limit_malloc_arenas(2)
        XB.limit_memory(10.0)
        if int(o.workers) > 1:
            o.workers = 1
    t0 = time.time()
    with XB.restricted_output():
        if o.split_plan:
            return print_split_plan(o)
        if o.placement_check:
            return placement_check(o)
        if o.gate_check:
            s = gate_check(o, o.gate_check, o.gate_level)
            return 0 if s["passed"] else 1
        o.algp = resolve_algp(o, need_file=_need_file(o))
        o.wplus, o.wplus_status = load_wplus("off" if "gate" in o.PARTS else o.wplus)
        if o.summarize_only:
            summarize(o)
            return 0
        units, skipped, expected = enumerate_units(o)
        units = select_shard(o, units)
        print(f"[plan] 부분 {o.PARTS} · 단위 {len(units)} · 건너뛴 분할 {len(skipped)} · Algorithm P {o.algp['name']}({o.algp.get('source', '')}) · "
              f"배치 구현 {o.placement_impl} · W+ {o.wplus_status} · 스레드 {o.threads} · 워커 {o.workers}", flush=True)
        if o.count_only:
            count_only(o, units, skipped)
            return 0
        XB.check_out_dir(o.SHARDS)
        o.SHARDS.mkdir(parents=True, exist_ok=True)
        todo = []
        for u in units:
            if o.resume:
                ok, why = XB.unit_state(o.SHARDS, tag_of(o, u[0]), XB.resolve(u[1])[1] if u[0] == "xcr" else u[1], u[2], u[3],
                                        unit_cfg(o, u[0], XB.data_sha(o.ha, None if u[0] == "xcr" else u[1])), u[4])
                if ok:
                    continue
            todo.append(u)
        exp_items = [(list(k), v) for k, v in expected.items()]
        done, failed = XB.execute(todo, _worker_run, workers=o.workers, threads=o.threads, init=_worker_init, init_args=(o.ARGV, exp_items),
                                  retries=o.pool_retries, permit=o.PERMIT)
        print(f"[done] 완료 {len(done)} · 실패 {len(failed)} · 재개로 건너뜀 {len(units) - len(todo)} · {time.time() - t0:.0f}s", flush=True)
        if failed:
            XB.check_out_dir(o.OUT)
            o.OUT.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([dict(unit="|".join(str(v) for v in u), error=e) for u, e in failed]).to_csv(o.OUT / f"{tag_of(o, 'xc')}_failed.csv", index=False)
        if not o.no_summarize and "gate" not in o.PARTS:
            summarize(o)
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
