"""XB_multisource_stacking(계획 2.2): 여러 물리식·제품의 라벨 가중 적층(Stack)과 그 위 잔차(StackR).

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100) 2.2절과 1절(공통 규약). 설계 근거는
docs/research/2026-10-04/harness_implementation_plan.md 3.3·4.2절, alt_products.md 8절이다. 계획 문구의 해석과 구현 결정은
docs/research/2026-10-04/impl_notes/x_multisource_stacking.md 에 적었다. 가설·판정 규칙은 계획 그대로이며 이 파일에서 바꾸지 않는다.

후보(핵심, K ≤ 7, 순서 = 동률 순서)
  P1  재보정 Stefan. b = e5_sqrt_tdd, c0 = E0(원천 최소제곱), c_n = shrink(LS, c0, n, κ 10). h54 의 E_n 과 같은 식이다.
  P*  연도 정합 도일 Stefan. b = √tdd_matched(lgx_tdd_matched_v1 + v4 새 셀 표 xb_tdd_matched_v4, sha256 앞 16자 고정), 척도 보정.
      대상의 A 풀이 될 수 있는 셀(대상 셀 전체) 또는 채점 셀(대상의 eval_mask 셀)의 5 % 넘게 값이 없으면 그 대상의 모든 분할에서 뺀다.
      v4 표가 없거나 해시가 다르면 중단한다(--allow-missing-tdd-v4 일 때만 없는 표를 허용하고 unit.json 에 적는다).
  Ku  Kudryavtsev(p4_ku), 척도 보정 + h42.anchor_fill(비유한·0 이하 셀은 ρ·s, ρ = 원천 b/s 중앙값).
  Ed  토양형 Stefan(p2_edaphic), 척도 보정 + anchor_fill.
  Ss  토양 도일 Stefan(e5_sqrt_tdd_soil), 척도 보정 + anchor_fill.
      (계획은 anchor_fill 을 Ku 에만 적었다. 이 구현은 h42 x9 의 앵커 경로와 같이 척도 앵커 P*·Ku·Ed·Ss 모두에 쓴다. SI 에 적는다: ANCHOR_NOTE)
  Cr  CCI v4 원값(cci_alt). 무효 셀(비유한 또는 cci_valid < 0.5)은 같은 보정의 P1 값.
  Ca  아핀 보정 CCI v4. 원천 (a0, b0)(유효 셀 최소제곱)에서 각 계수를 κ 10 수축. 무효 셀은 P1 값.
  기준선 팔 Pe(Stefan·CCI 평균 = h54.re_anchor) 는 후보가 아니다(Pe = 0.5·P1 + 0.5·Cr 이라 볼록 결합 안에 있다).
  2단계 민감도 cci5y, 3단계 확장 Wei(XG 마스크 통과 대상만, Stack0 제외, StackR 원천 행의 학습 지점 5 km 안 제외), YK(알래스카만)는
  --variant cci5y·ext 와 --product 로 넣는다(제품 표 형식은 구현 기록 4절). Wei 표에는 mask_ok·train_dist_km 열이 반드시 있어야 하고
  모든 원천 행의 train_dist_km 가 유한해야 한다(없으면 중단). 5 km 제외는 Wei 표를 준 확장 판 전체의 StackR 에 적용하고 뺀 행 수를 unit.json 에 적는다.

가중 학습(계획 2.2 '가중 학습' 1–8)
  1. 선택 라벨 L 의 묶음 F = h42.cv_folds_of(L 의 A 블록). n < 10, 또는 cv_folds_of 가 낸 묶음 수 K 가 2 미만이면 적층 = P1 이고
     stack_fallback 에 사유를 적는다(문구 그대로. 라벨이 한 블록에만 있어 cv_folds_of 가 셀 묶음을 낸 경우도 K ≥ 2 이면 대체가 아니다.
     셀 묶음 여부는 적층 기록의 fold_flag 에 남긴다). γ̂ = 1 도 사유로 적는다.
  2. 교차 적합 기저 Z(n × K): 묶음 j 마다 후보 보정 계수를 L \\ L_j 로 구해 L_j 를 예측한다(oof_base).
  3. 단체 제약 최소제곱(simplex_ls): 지지 집합 2^K − 1 개를 모두 닫힌 해로 풀어 음수 없는 해 가운데 SSE 최소. 동률은 P1 포함, 원소 수 적음 순.
  4. γ ∈ {0, 0.25, 0.5, 0.75, 1}: 같은 묶음의 2단 교차검증(묶음 j 를 뺀 Z 행으로 w_j 를 맞추고 (1 − γ)·w_j + γ·e_P1 로 L_j 를 채점).
     SSE 최소, 동률이면 큰 γ(choose_gamma). 구현 보고서 3.3 의 절차이며 중첩 교차검증이 아니다(GAMMA_CV_NOTE, SI 에 적는다).
  5. w* = (1 − γ̂)·ŵ + γ̂·e_P1. 채점 셀의 기저 예측은 L 전체로 보정한 계수로 낸다.
  6. StackR: catboost_lo 잔차 모형. 학습 목표 = 라벨 행 y − 적층(표본 안, 최종 계수와 w*), 전이에서는 원천 행 y − 적층(원천 계수 c0, w*)을 더한다.
     λ 0.25, 지역 내는 교차검증 λ(묶음마다 그 학습 쪽 라벨로 적층을 다시 맞춘다).
  7. Stack0(전이, n 0): 원천 행의 원천 거시 지역 하나 제외 교차 적합 기저에 단체 제약 최소제곱. B:ens 는 h42 x9 의 등가중 기준선이다.
  8. 대상 라벨은 선택된 n 개만 가중·계수·γ·잔차에 쓴다. anchor_fill 의 ρ 와 c0·(a0, b0)는 원천에서만 구한다.

팔(계획 2.2 '팔과 범위')
  xb_r  지역 내(모드 r). Alaska, Lena, Canada. 분할 1–25(h54 wf6 규칙, 레나 분할 24 무효). n {20, 50, 100, 200, 500, 1000, 전량}(|A| 미만),
        추출 3(전량 1), seed 0·1, 추출 = h40.draw_cells(대상, 'r', …)(WF1·WF6 과 같다).
        방법 P0, P1, R1(λ 0.25·0.5·1.0·교차검증), Re(λ 0.25·교차검증), Pe, Stack, StackR(λ 0.25·0.5·1.0·교차검증), 개별 보정 후보 P*·cand_*(서술).
        P1·R1·Re 는 h54.RUnit(wf6)과 같은 함수·seed 로 적합한다(WF6 조각과 키 대조, 재현 관문).
  xb_t  전이(LG 원천, 100 km 버퍼). LG 27 (대상, 모드), Tibet_LGD, Russia_C~lgd, NAtlantic~lic(점 추정), Canada~exp~lic(민감도). 분할 1–5.
        n {0, 3, 10, 40, 160, 320, 1000, 전량}, 추출 5(0·전량 1), seed 0·1, 추출 = h40.draw_cells(LG·WF7 과 같다).
        방법 P0, P1, P*, R1(0.25), Re(0.25), Pe, Stack, StackR(0.25), Stack0(n 0), B:ens(n 0). P0·P1·R1·Re 는 h54.T7Unit 과 같은 행렬·seed.

W+ (XC 의 A1+ 팔, 계획 2.3. 코드 해시를 R1a·R1b 제출 전에 커밋한다)
  select_w_wplus(unit, sel, n, d): h54.TUnit.select_w 와 같은 묶음·seed 0 적합·SSE 누적으로 W(후보 W_CANDS)와 W+(W_CANDS + Stack, StackR@0.25)를
  한 번의 교차검증에서 함께 고른다. W 의 선택은 h54 와 같다. wplus_predictions(unit, sel, n, d, seed, g1) 는 채점 셀의 Stack, StackR@0.25 예측이다.
  XC 공급자 계약(x_workflow_end_to_end.WPlusProvider): xc_wplus_cv(unit, tr, te, n, d, fold, seed), xc_wplus_final(unit, sel, n, d, seed).
  문맥에 후보 표가 없으면 attach_cands_t 로 붙인다.

출력 제한(계획 0.3): 스모크·세기·집계의 화면에는 RMSE·Δ·판정을 쓰지 않는다. 집계 표는 data/processed/xbatch/XB_multisource_stacking/sealed/ 에만 쓴다.
스모크 조각도 봉인 폴더(sealed/smoke_shards)에 둔다. 본 실행 조각의 runs.csv 에는 rmse_cm·rmse_beq_cm·bias_cm 열을 쓰지 않는다(RUNS_SEALED_COLS.
집계와 관문은 blocksse.npz 만 쓴다).

로컬 자원(계획 1절): Rescale(WF_RESCALE=1) 밖에서는 시작 전 가용 메모리 30 GB 를 기다리고(--mem-wait 초), 세기·집계는 주소 공간 10 GB 상한,
적합이 있는 스모크·로컬 실행은 glibc 할당 영역을 2개로 묶은 뒤에만 같은 상한을 건다(묶지 못하면 상한 없이 경고). 스모크 환경 점검
(xbatch_core.smoke_env_report)은 스모크 집계 메타와 조각 unit.json 에 적는다.

명령행(ROOT)
  세기:     CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/x_multisource_stacking.py --count-only --threads 1
  스모크:   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c <코어 4개> python3 scripts/3_deep_learning/x_multisource_stacking.py \\
              --smoke --threads 2 --workers 0 2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'
  본 실행:  WF_RESCALE=1 python3 scripts/3_deep_learning/x_multisource_stacking.py --exp xb_r,xb_t --workers 22 --threads 4 --resume --no-summarize
  조각 나눔: --shard i/N(단위 목록을 이름 순으로 N 등분한 i 번째, 1 부터)
  집계:     WF_RESCALE=1 python3 scripts/3_deep_learning/x_multisource_stacking.py --summarize-only --threads 4       (봉인 폴더에만 쓴다)
  재현 관문: python3 scripts/3_deep_learning/x_multisource_stacking.py --gate --gate-dir results/rescale_wf2/data/processed/wf/shards
  1c 입력:  nice -n 10 python3 scripts/3_deep_learning/x_multisource_stacking.py --write-tdd-v4 --threads 2   (v4 새 셀 tdd_matched, 라벨 값 미사용)
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
import itertools                                                           # noqa: E402
import json                                                                # noqa: E402
import time                                                                # noqa: E402
from collections import Counter                                            # noqa: E402

import numpy as np                                                         # noqa: E402
import pandas as pd                                                        # noqa: E402

H, X, W = XB.H, XB.X, XB.W
from polar.h4_common import BlockStore, seed_of                            # noqa: E402
from polar.m1_core import eval_mask, half_split_blocks                     # noqa: E402

ROOT = XB.ROOT
LO = W.LO
KAPPA = float(W.KAPPA)
LAMS = tuple(W.LAMS)
LAM_BASE = float(W.LAM_BASE)
LAM_CV = float(W.LAM_CV)
CCI_COL, CCIV_COL = W.CCI_COL, W.CCIV_COL

# ================================================================ 1. 고정값(계획 2.2. 바꾸면 사전 등록에서 벗어난다)
EXP_ID = "XB"
EXP_NAME = XB.EXP_NAMES[EXP_ID]                                            # XB_multisource_stacking
CANDS_CORE = ("P1", "P*", "Ku", "Ed", "Ss", "Cr", "Ca")                   # 순서 = 동률 순서(P1 이 첫째)
SCALE_COLS = {"P1": "e5_sqrt_tdd", "P*": "tdd_matched", "Ku": "p4_ku", "Ed": "p2_edaphic", "Ss": "e5_sqrt_tdd_soil"}
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
STACK_MIN_N = 10                                                           # n < 10 이면 적층 = P1
MIN_FOLDS = 2                                                              # 블록 묶음 < 2 이면 적층 = P1
PSTAR_MAX_MISS = 0.05                                                      # A 또는 채점 셀의 5 % 넘게 P* 가 없으면 P* 를 뺀다
FALLBACK_MAX = 0.5                                                         # (대상, n) 의 대체 비율 > 50 % → 판정 불가(적층 미작동)
TIE_REL = 1e-10                                                            # SSE 동률의 상대 허용(지지 집합·γ)
NEG_TOL = 1e-12                                                            # 음수 가중 허용(닫힌 해의 반올림)
WEI_KM = 5.0                                                               # 확장 판 StackR 원천 행의 Wei 학습 지점 제외 반경
FB_N, FB_FOLDS, FB_GAMMA = "n<10", "folds<2", "gamma=1"
XBR_TARGETS = ("Alaska", "Lena", "Canada")
XBR_GRID = (20, 50, 100, 200, 500, 1000, -1)
XBR_DRAWS = 3
XBR_SPLITS = tuple(range(1, 26))                                           # 지역 내 분할 1–25(WF6·WF9·XB·XE)
XBT_GRID = (0, 3, 10, 40, 160, 320, 1000, -1)
XBT_DRAWS = 5
XBT_LGD = ("Tibet_LGD", "Russia_C~lgd", "NAtlantic~lic", "Canada~exp~lic")
ARMS = ("xb_r", "xb_t")
TAGS = {"xb_r": "xbr", "xb_t": "xbt"}
VARIANTS = ("", "cci5y", "ext")
PRODUCTS = {                                                               # 이름 → (변형, 결측 처리, Stack0 제외, 5 km 규칙, 알래스카만)
    "cci5y": ("cci5y", "p1", False, False, False),
    "Wei": ("ext", "drop", True, True, False),
    "YK": ("ext", "drop", False, False, True),
}
# XG 누설 마스크가 걸리는 제품(alt_products.md 2.2·7.7: Wei 는 L1, 학습 지점 좌표 공개). L0 제품(CCI v5, Yi·Kimball)에는 마스크가 없다
# (alt_products.md 7.7 '주 마스크'). 마스크 제품의 표에는 mask_ok 열이, 5 km 규칙 제품의 표에는 train_dist_km 열이 반드시 있어야 한다.
PRODUCT_MASK_REQUIRED = {"cci5y": False, "Wei": True, "YK": False}
WPLUS_EXTRA = ("Stack", "StackR@0.25")
WPLUS_CANDS = tuple(W.W_CANDS) + WPLUS_EXTRA                               # W 후보 다음에 적층 둘(동률 순서)
TDD_V1 = ROOT / "data" / "processed" / "lgx_tdd_matched_v1.csv"
TDD_V4 = XB.out_dir(EXP_ID) / "inputs" / "xb_tdd_matched_v4.csv"
TDD_V4_META = TDD_V4.with_name(TDD_V4.stem + "_meta.json")
TDD_V4_SHA16 = "4b22ca458685950c"                                          # 1c 산출(--write-tdd-v4, 2026-10-04 15:23)의 sha256 앞 16자. 다르면 중단
# 묶음(4절)에 넣을 XB 입력(xbatch_core.PAYLOAD_INPUTS 에도 넣었다). payload_manifest(extra=PAYLOAD_EXTRA) 로도 넣을 수 있다
PAYLOAD_EXTRA = tuple(str(p.relative_to(ROOT)) for p in (TDD_V4, TDD_V4_META))
RUNS_SEALED_COLS = ("rmse_cm", "rmse_beq_cm", "bias_cm")                   # 본 실행 조각 runs.csv 에 쓰지 않는 열(계획 0.3 열람 순서 6)
MEM_MIN_GB = 30.0                                                          # 1절 로컬 자원: 시작 전 가용 메모리 하한
MEM_LIMIT_GB = 10.0                                                        # 1절 로컬 자원: 작업 하나 메모리 상한
MALLOC_ARENAS = 2                                                          # 적합이 있는 로컬 실행의 glibc 할당 영역 수(주소 공간 상한 아래 CatBoost 멈춤 방지)
A2_CELLS = ROOT / "data" / "processed" / "m1" / "a2_year_matched_tdd_cells.csv"
ERA5_NC = ROOT / "data" / "raw" / "era5land" / "nh_monthly_2010-2024.nc"
PLAN_SEC = "2.2"

# 가설(계획 2.2 '가설'). 판정 규칙은 계획 1절·2.2절 그대로다.
HYP_N = {"XB-1": (10, 40, 160, -1), "XB-2": (10, 40, 160, -1), "XB-3": (200, 500, 1000, -1), "XB-4": (200, 500, 1000, -1)}
HOLM_M_MAIN = 12                                                           # XB-1–XB-3 의 n 별 우세 p
HOLM_M_NI = 4                                                              # XB-4 의 비열등 p
BLIND = {"XB-1": "비맹검 부분 포함(LGX L29 의 기준선, WF7 의 Pe)", "XB-2": "비맹검 부분 포함(WF7 의 Re − R1)",
         "XB-3": "비맹검 부분 포함(WF1·WF6 의 지역 내 Re)", "XB-4": "비맹검 부분 포함(WF6 의 알래스카 Re 열세)", "XB-5": "비맹검 부분 포함",
         "XB-6": ""}
DESIGN = "결과 열람 뒤 설계"
TEMPLATES = {
    "XB-1": {"우세": "대상 라벨로 여러 물리식과 위성 제품의 가중을 학습한 적층은 라벨 {k}개에서 재보정 Stefan 보다 오차가 {a} cm 작았다({pool})",
             "열세": "적층은 라벨 {k}개에서 재보정 Stefan 보다 오차가 {a} cm 컸다({pool})",
             "동등": "적층은 라벨 {k}개에서 재보정 Stefan 과 0.5 cm 안에서 같았다({pool})",
             "미결정": "적층과 재보정 Stefan 의 차이를 확인하지 못했다(라벨 {k}개, {pool}). 물리식·제품 결합의 이득은 지역 조건부라는 서술(WF7, L29)을 유지한다",
             "판정 불가": "적층이 작동하지 않아(대체 비율 {r}) 판정할 수 없었다(라벨 {k}개)",
             "판정 불가_일반": "적층과 재보정 Stefan 의 차이를 판정할 수 없었다"},
    "XB-2": {"우세": "대상 라벨로 여러 물리식과 위성 제품의 가중을 학습한 적층 위의 잔차는 라벨 {k}개에서 재보정 앵커 잔차보다 오차가 {a} cm 작았다({pool})",
             "열세": "적층 잔차는 라벨 {k}개에서 재보정 앵커 잔차보다 오차가 {a} cm 컸다({pool})",
             "동등": "적층 잔차는 라벨 {k}개에서 재보정 앵커 잔차와 0.5 cm 안에서 같았다({pool})",
             "미결정": "적층 잔차와 재보정 앵커 잔차의 차이를 확인하지 못했다(라벨 {k}개, {pool}). 물리식·제품 결합의 이득은 지역 조건부라는 서술(WF7, L29)을 유지한다",
             "판정 불가": "적층이 작동하지 않아(대체 비율 {r}) 판정할 수 없었다(라벨 {k}개)",
             "판정 불가_일반": "적층 잔차와 재보정 앵커 잔차의 차이를 판정할 수 없었다"},
    "XB-3": {"우세": "라벨이 많은 지역 안에서 적층 잔차는 재보정 앵커 잔차보다 라벨 {k}개에서 오차가 {a} cm 작았다({pool})",
             "열세": "라벨이 많은 지역 안에서 적층 잔차는 재보정 앵커 잔차보다 라벨 {k}개에서 오차가 {a} cm 컸다({pool})",
             "동등": "라벨이 많은 지역 안에서 적층 잔차는 재보정 앵커 잔차와 라벨 {k}개에서 0.5 cm 안에서 같았다({pool})",
             "미결정": "라벨이 많은 지역 안에서 적층 잔차와 재보정 앵커 잔차의 차이를 확인하지 못했다(라벨 {k}개, {pool})",
             "판정 불가": "라벨이 많은 지역 안에서 적층이 작동하지 않아(대체 비율 {r}) 판정할 수 없었다(라벨 {k}개)",
             "판정 불가_일반": "라벨이 많은 지역 안에서 적층 잔차와 재보정 앵커 잔차의 차이를 판정할 수 없었다"},
}
STACK_NA_NOTE = "적층 미작동"                                              # verdict_note: 퇴화 추출 규칙(대체 비율 > 50 %)으로 판정 불가
XB4_SENT = {"비열등": "물리식이 오차 하한에 가까운 알래스카 지역 안에서 적층의 오차 증가는 재보정 Stefan 대비 0.5 cm 안이었다(비열등, 라벨 {k}개)",
            "아님": "물리식이 오차 하한에 가까운 알래스카 지역 안에서 적층의 재보정 Stefan 대비 비열등을 확인하지 못했다(라벨 {k}개)",
            "판정 불가": "물리식이 오차 하한에 가까운 알래스카 지역 안에서 적층이 작동하지 않아(대체 비율 {r}) 판정할 수 없었다(라벨 {k}개)",
            "판정 불가_일반": "물리식이 오차 하한에 가까운 알래스카 지역 안에서 적층의 재보정 Stefan 대비 비열등을 판정할 수 없었다(라벨 {k}개, {why})"}
WEIGHT_NOTE = "가중은 서술로만 쓰고 원인 문장(예: 'CCI 가 Stefan 을 보완한다')은 쓰지 않는다. 후보 사이 공선성 때문에 가중의 해석이 불안정하다"
N10_NOTE = "n 10 의 XB-1 은 적층 대체(사유 n<10, 묶음<2, γ̂ = 1) 때문에 사실상 P1 대 P1 비교일 수 있다(계획 2.2 퇴화 추출 규칙, 이 행의 대체 비율 {r})"
GAMMA_CV_NOTE = ("SI: γ 의 2단 교차검증은 구현 보고서 3.3 의 절차(묶음 j 를 뺀 Z 행으로 w_j 를 맞추고 묶음 j 의 Z 행을 채점)이며 중첩 교차검증이 아니다. "
                 "Z 의 다른 묶음 행은 묶음 j 의 라벨이 들어간 계수로 만들어졌으므로 γ 선택의 채점은 낙관적일 수 있다(작은 γ 쪽). 대상 밖 라벨은 쓰지 않는다")
ANCHOR_NOTE = ("SI: anchor_fill(비유한·0 이하 셀을 ρ·s 로, ρ = 원천 b/s 중앙값)은 계획 문구의 Ku 뿐 아니라 척도 앵커 후보 P*·Ku·Ed·Ss 모두에 썼다"
               "(h42 x9 의 앵커 경로와 같다)")


# ================================================================ 2. 순수 함수(시험 대상)
def simplex_ls(Z, y, p1=0, trace=None):
    """w = argmin ||y − Z w||², w ≥ 0, Σw = 1. 지지 집합 2^K − 1 개를 모두 닫힌 해(마지막 원소를 기준으로 한 차분 최소제곱)로 풀고,
    음수가 없는 해 가운데 SSE 가 가장 작은 것을 고른다. 차분 행렬의 계수가 모자란 지지 집합(해가 하나가 아님)은 건너뛴다(같은 SSE 를 내는
    더 작은 지지 집합이 있다). 동률(상대 TIE_REL)은 P1 을 포함하고 원소가 적은 지지 집합, 그다음 색인 사전 순이다. 반복 해법이 없다.
    비유한 행은 뺀다. 반환 (w(길이 K), SSE, 지지 집합 튜플)."""
    Z = np.asarray(Z, float); y = np.asarray(y, float)
    K = Z.shape[1]
    ok = np.isfinite(y) & np.all(np.isfinite(Z), 1)
    Z, y = Z[ok], y[ok]
    best = []
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            S = tuple(S)
            if r == 1:
                v = np.zeros(0)
                wS = np.ones(1)
                res = y - Z[:, S[0]]
            else:
                ref = S[-1]
                Dm = Z[:, list(S[:-1])] - Z[:, [ref]]
                t = y - Z[:, ref]
                if len(t) < r - 1:
                    continue
                v, _, rank, _ = np.linalg.lstsq(Dm, t, rcond=None)
                if int(rank) < r - 1:
                    continue
                wS = np.r_[v, 1.0 - v.sum()]
                res = t - Dm @ v
            if np.any(wS < -NEG_TOL) or not np.all(np.isfinite(wS)):
                continue
            sse = float(res @ res)
            best.append((sse, S, np.clip(wS, 0.0, None)))
    if not best:                                                           # 행이 없을 때: P1 단위 벡터
        w = np.zeros(K); w[p1] = 1.0
        return w, float("nan"), (p1,)
    smin = min(b[0] for b in best)
    tol = TIE_REL * max(abs(smin), 1.0)
    cands = [b for b in best if b[0] <= smin + tol]
    sse, S, wS = min(cands, key=lambda b: (p1 not in b[1], len(b[1]), b[1]))
    w = np.zeros(K)
    w[list(S)] = wS / wS.sum()
    if trace is not None:
        trace.append(dict(kind="simplex", n_rows=int(len(y)), support=list(S)))
    return w, float(sse), S


def choose_gamma(Z, y, fid, K, p1=0, gammas=GAMMAS):
    """P1 쪽 수축 γ 의 2단 교차검증(같은 묶음): 묶음 j 를 뺀 Z 행으로 w_j = simplex_ls 를 맞추고 w_j,γ = (1 − γ)·w_j + γ·e_P1 로 묶음 j 를
    채점한다. SSE 최소, 동률(상대 TIE_REL)이면 큰 γ. 반환 (γ̂, {γ: SSE})."""
    Z = np.asarray(Z, float); y = np.asarray(y, float); fid = np.asarray(fid, int)
    e = np.zeros(Z.shape[1]); e[p1] = 1.0
    sse = {g: 0.0 for g in gammas}
    for j in range(int(K)):
        tr, te = fid != j, fid == j
        if not tr.any() or not te.any():
            continue
        wj, _, _ = simplex_ls(Z[tr], y[tr], p1)
        Zt, yt = Z[te], y[te]
        m = np.isfinite(yt) & np.all(np.isfinite(Zt), 1)
        for g in gammas:
            wg = (1.0 - g) * wj + g * e
            pr = _wsum(Zt[m], wg, p1)
            sse[g] += float(np.sum((pr - yt[m]) ** 2))
    smin = min(sse.values())
    tol = TIE_REL * max(abs(smin), 1.0)
    g_hat = max(g for g in gammas if sse[g] <= smin + tol)
    return float(g_hat), sse


def _wsum(M, w, p1=0):
    """Σ_k w_k·M[:, k] 를 0 이 아닌 가중만 더한다(0·NaN 을 피한다). w = e_P1 이면 P1 열 그대로다."""
    M = np.asarray(M, float)
    nz = [k for k in range(len(w)) if w[k] != 0.0]
    if nz == [p1] and w[p1] == 1.0:
        return M[:, p1].copy()
    out = np.zeros(M.shape[0])
    for k in nz:
        out = out + w[k] * M[:, k]
    return out


def _is_p1(w, p1=0):
    return bool(w[p1] == 1.0 and all(w[k] == 0.0 for k in range(len(w)) if k != p1))


def block_folds(blk_sel, target, mode, split, n, draw):
    """적층의 묶음(계획 2.2 '가중 학습' 1, 문구 그대로): F = h42.cv_folds_of(선택 라벨의 A 블록), 묶음 수 K 가 2 미만이면 대체(folds<2).
    라벨이 한 블록에만 있어 cv_folds_of 가 셀 묶음(표지 cell_folds)을 낸 경우도 K ≥ 2 이면 그 묶음으로 적층한다(대체가 아니다).
    반환 (묶음 번호, 묶음 수, 표지, 대체 사유)."""
    fid, K, flag = X.cv_folds_of(np.asarray(blk_sel), target, mode, split, n, draw)
    reason = FB_FOLDS if int(K) < MIN_FOLDS else ""
    return fid, int(K), flag, reason


def fallback_reason(n_lab, blk_sel, target, mode, split, n, draw):
    """라벨 값 없이 정할 수 있는 대체 사유(n < 10, 블록 묶음 < 2). 세기(--count-only)와 본 실행이 같은 규칙을 쓴다."""
    if int(n_lab) < STACK_MIN_N:
        return FB_N
    return block_folds(blk_sel, target, mode, split, n, draw)[3]


# ================================================================ 3. 후보 표
class CandSet:
    """후보 하나마다 원천(src)·A 풀(A)·채점 셀(B)의 기저 값과 원천 계수. 대상 라벨은 담지 않는다(누설 규약: 보정은 호출 때 선택 라벨로만).
    kind: scale(c·b), raw(유효 셀 x, 무효 셀 P1), affine(유효 셀 a + b·x, 무효 셀 P1)."""

    def __init__(self):
        self.names, self.kind, self.b, self.valid, self.c0, self.info, self.dropped = [], {}, {}, {}, {}, {}, {}
        self.stack0_exclude: set = set()
        self.resid_src_keep = None                                         # StackR 원천 행 유지 마스크(확장 판 Wei 5 km 규칙)
        self.ens = None                                                    # B:ens 의 재료(h42 x9 정의)
        self.meta: dict = {}

    @property
    def K(self):
        return len(self.names)

    def describe(self):
        return dict(names=list(self.names), K=self.K, dropped=dict(self.dropped), info=self.info, stack0_exclude=sorted(self.stack0_exclude),
                    resid_src_excluded=int((~self.resid_src_keep).sum()) if self.resid_src_keep is not None else 0, **self.meta)

    @classmethod
    def build(cls, s, y_src, raw, cci, ccv, products=None, parent="", variant="", pstar_rule=True, pstar_target=None):
        """s, cci, ccv = {src, A, B} 배열. raw = {후보 이름: {src, A, B}}(P*, Ku, Ed, Ss 의 원값. P* 는 tdd_matched). products = {이름: dict(src, A,
        B, mask_A, mask_B, dist_src)}. 원천 계수와 ρ 는 원천 값만으로 구한다.
        pstar_target = dict(miss_A, miss_B): 대상 수준 P* 결측 비율(A 풀이 될 수 있는 대상 셀 전체, 대상의 채점 셀 전체). 주면 P* 를 뺄지는 이 값으로
        정하고(대상의 모든 분할에서 같은 결정), 없으면(합성 시험) 이 단위의 A·채점 셀 비율로 정한다."""
        cs = cls()
        y_src = np.asarray(y_src, float)
        s = {k: np.asarray(v, float) for k, v in s.items()}
        cs.names.append("P1"); cs.kind["P1"] = "scale"; cs.b["P1"] = s; cs.c0["P1"] = float(H.ls_E(y_src, s["src"]))
        cs.info["P1"] = dict(c0=cs.c0["P1"])
        for nm in ("P*", "Ku", "Ed", "Ss"):
            if nm not in raw:
                cs.dropped[nm] = "열 없음"
                continue
            r = {k: np.asarray(v, float) for k, v in raw[nm].items()}
            if nm == "P*":
                with np.errstate(invalid="ignore"):
                    r = {k: np.sqrt(np.where(v > 0, v, np.nan)) for k, v in r.items()}
                miss = {k: float(np.mean(~np.isfinite(r[k]))) if len(r[k]) else 0.0 for k in ("A", "B")}
                cs.info["P*"] = dict(miss_A=miss["A"], miss_B=miss["B"], miss_src=float(np.mean(~np.isfinite(r["src"]))) if len(r["src"]) else 0.0)
                if pstar_target is not None:                              # 대상 수준 결정(계획 2.2 '그 대상에서 P* 후보를 빼고')
                    dec = dict(A=float(pstar_target["miss_A"]), B=float(pstar_target["miss_B"]))
                    cs.info["P*"].update(target_miss_A=dec["A"], target_miss_B=dec["B"], decision="target")
                else:
                    dec = miss
                    cs.info["P*"]["decision"] = "unit"
                if pstar_rule and (dec["A"] > PSTAR_MAX_MISS or dec["B"] > PSTAR_MAX_MISS):
                    cs.dropped[nm] = (f"tdd_matched 결측({cs.info['P*']['decision']}) A {dec['A']:.3f}, 채점 {dec['B']:.3f} > {PSTAR_MAX_MISS}")
                    continue
            b, rho, nrep = X.anchor_fill(r["src"], r["A"], r["B"], s["src"], s["A"], s["B"])
            if not np.isfinite(rho) or not np.all(np.isfinite(b["src"])):
                cs.dropped[nm] = "원천 값 없음(ρ 비유한)"
                continue
            cs.names.append(nm); cs.kind[nm] = "scale"; cs.b[nm] = {k: np.asarray(v, float) for k, v in b.items()}
            cs.c0[nm] = float(H.ls_E(y_src, b["src"]))
            cs.info.setdefault(nm, {}).update(rho=float(rho), n_replaced=dict(nrep), c0=cs.c0[nm])
        val = {k: np.isfinite(np.asarray(cci[k], float)) & np.isfinite(np.asarray(ccv[k], float)) & (np.asarray(ccv[k], float) >= 0.5)
               for k in ("src", "A", "B")}
        xc = {k: np.asarray(cci[k], float) for k in ("src", "A", "B")}
        cs.names.append("Cr"); cs.kind["Cr"] = "raw"; cs.b["Cr"] = xc; cs.valid["Cr"] = val
        cs.info["Cr"] = dict(valid_A=float(val["A"].mean()) if len(val["A"]) else np.nan, valid_B=float(val["B"].mean()) if len(val["B"]) else np.nan)
        a0, b0 = X.affine_ls(xc["src"][val["src"]], y_src[val["src"]])
        if np.isfinite(a0) and np.isfinite(b0):
            cs.names.append("Ca"); cs.kind["Ca"] = "affine"; cs.b["Ca"] = xc; cs.valid["Ca"] = val; cs.c0["Ca"] = (float(a0), float(b0))
            cs.info["Ca"] = dict(a0=float(a0), b0=float(b0))
        else:
            cs.dropped["Ca"] = "원천 아핀 계수 비유한"
        # B:ens(h42 x9): (E0·s + c0_ku·ku + c0_cci·cci)/3, ku·cci 는 anchor_fill(cci 는 원값 그대로 채운다), c0 = 원천 최소제곱
        bc, rho_c, _ = X.anchor_fill(xc["src"], xc["A"], xc["B"], s["src"], s["A"], s["B"])
        if "Ku" in cs.names and np.isfinite(rho_c) and np.all(np.isfinite(bc["src"])):
            cs.ens = dict(ku_B=cs.b["Ku"]["B"], c0_ku=cs.c0["Ku"], cci_B=np.asarray(bc["B"], float), c0_cci=float(H.ls_E(y_src, bc["src"])))
        for pn, pv in (products or {}).items():
            cs._add_product(pn, pv, y_src, parent)
        cs.meta.update(variant=str(variant), parent=str(parent))
        return cs

    def _add_product(self, name, pv, y_src, parent):
        """제품 후보(아핀 보정). cci5y 는 결측을 P1 값으로, Wei·YK 는 A 풀이나 채점 셀에 마스크 실패·결측이 하나라도 있으면 그 대상에서 뺀다.
        누설 통제(계획 2.2 '가중 학습' 8, 확장 후보)는 선택 열에 기대지 않는다: 마스크 제품(Wei)은 mask_A·mask_B 가, 5 km 규칙 제품(Wei)은 모든
        원천 행의 유한한 거리(dist_src)가 있어야 하고 없으면 중단한다. 5 km 제외는 이 판(확장 판)의 StackR 원천 행 전체에 적용한다(그 대상의
        후보에 들어가는지와 관계없이, 문구 '그 판의 잔차 학습에서 뺀다'). 뺀 행 수는 meta 와 unit.json 에 적는다."""
        _, missing, ex0, km5, ak_only = PRODUCTS[name]
        x = {k: np.asarray(pv[k], float) for k in ("src", "A", "B")}
        if km5:
            d = pv.get("dist_src")
            if d is None:
                raise SystemExit(f"[제품] {name}: 학습 지점 거리(train_dist_km)가 없다. 5 km 규칙(계획 2.2 '가중 학습' 8)을 적용할 수 없다")
            d = np.asarray(d, float)
            if len(d) != len(x["src"]) or not np.all(np.isfinite(d)):
                nbad = int((~np.isfinite(d)).sum()) if len(d) == len(x["src"]) else -1
                raise SystemExit(f"[제품] {name}: 원천 행 {len(x['src'])}개 가운데 train_dist_km 가 유한하지 않은 행 {nbad}개(모든 원천 행에 거리가 있어야 한다)")
            keep = ~(d < WEI_KM)
            self.resid_src_keep = keep if self.resid_src_keep is None else (self.resid_src_keep & keep)
            self.meta[f"km5_src_excluded_{name}"] = int((~keep).sum())
        if ak_only and parent != "Alaska":
            self.dropped[name] = "알래스카 계열이 아니다"
            return
        val = {k: np.isfinite(x[k]) for k in ("src", "A", "B")}
        if missing == "drop":
            if PRODUCT_MASK_REQUIRED.get(name) and (pv.get("mask_A") is None or pv.get("mask_B") is None):
                raise SystemExit(f"[제품] {name}: XG 마스크(mask_ok)가 없다. 계획 2.2 의 'A 풀과 채점 셀이 모두 XG 마스크를 통과한 대상만'을 적용할 수 없다")
            mA, mB = pv.get("mask_A"), pv.get("mask_B")                    # 마스크 없는 L0 제품(YK)은 결측만 본다
            okA = val["A"] & (np.ones(len(x["A"]), bool) if mA is None else np.asarray(mA, bool))
            okB = val["B"] & (np.ones(len(x["B"]), bool) if mB is None else np.asarray(mB, bool))
            if not (okA.all() and okB.all()):
                self.dropped[name] = f"XG 마스크 또는 결측(A {int((~okA).sum())}셀, B {int((~okB).sum())}셀)"
                return
        a0, b0 = X.affine_ls(x["src"][val["src"]], y_src[val["src"]])
        if not (np.isfinite(a0) and np.isfinite(b0)):
            self.dropped[name] = "원천 아핀 계수 비유한"
            return
        self.names.append(name); self.kind[name] = "affine"; self.b[name] = x; self.valid[name] = val; self.c0[name] = (float(a0), float(b0))
        self.info[name] = dict(a0=float(a0), b0=float(b0), missing=missing, valid_A=float(val["A"].mean()) if len(val["A"]) else np.nan)
        if ex0:
            self.stack0_exclude.add(name)

    # ------------------------------------------------------------ 보정과 예측
    def calibrate(self, idx, yA, trace=None):
        """선택 라벨(A 색인 idx)로 후보 계수를 보정한다. scale: c = shrink(LS(y, b), c0, n, κ)(n = 0 또는 LS 비유한이면 c0, h54 의 E_n 과 같은 식).
        affine: 유효 라벨로 (a, b) 최소제곱 뒤 각 계수를 원천 (a0, b0)로 κ 수축(유효 라벨 수 m). raw: 계수 없음."""
        idx = np.asarray(idx, int)
        yA = np.asarray(yA, float)
        if trace is not None:
            trace.append(dict(kind="coef", idx=idx.copy(), what="stack_calibrate"))
        out = {}
        for nm in self.names:
            k = self.kind[nm]
            if k == "scale":
                c0 = self.c0[nm]
                if len(idx) == 0:
                    out[nm] = float(c0)
                    continue
                E_ls = H.ls_E(yA[idx], self.b[nm]["A"][idx])
                out[nm] = float(c0) if not np.isfinite(E_ls) else X.shrink(E_ls, c0, len(idx), KAPPA)
            elif k == "affine":
                a0, b0 = self.c0[nm]
                v = self.valid[nm]["A"][idx]
                m = int(v.sum())
                if m > 0:
                    al, bl = X.affine_ls(self.b[nm]["A"][idx][v], yA[idx][v])
                else:
                    al, bl = np.nan, np.nan
                out[nm] = (X.shrink(al, a0, m, KAPPA), X.shrink(bl, b0, m, KAPPA))
            else:
                out[nm] = None
        return out

    def source_params(self):
        """원천 계수(c0, (a0, b0)). 전이 StackR 의 원천 행 목표와 Stack0 의 채점 셀 예측에 쓴다."""
        return {nm: (self.c0[nm] if self.kind[nm] != "raw" else None) for nm in self.names}

    def matrix(self, params, part, rows=None):
        """후보 예측 행렬(행 = 셀, 열 = self.names 순서). raw·affine 의 무효 셀은 같은 계수의 P1 값이다."""
        sl = (lambda v: v) if rows is None else (lambda v: np.asarray(v)[np.asarray(rows, int)])  # noqa: E731
        p1 = float(params["P1"]) * sl(self.b["P1"][part])
        cols = []
        with np.errstate(invalid="ignore"):
            for nm in self.names:
                k = self.kind[nm]
                if nm == "P1":
                    cols.append(p1)
                elif k == "scale":
                    cols.append(float(params[nm]) * sl(self.b[nm][part]))
                elif k == "raw":
                    cols.append(np.where(sl(self.valid[nm][part]), sl(self.b[nm][part]), p1))
                else:
                    a_, b_ = params[nm]
                    cols.append(np.where(sl(self.valid[nm][part]), float(a_) + float(b_) * sl(self.b[nm][part]), p1))
        return np.column_stack(cols) if cols else np.zeros((len(p1), 0))

    def subset(self, names):
        """후보 부분집합의 사본(Stack0 의 Wei 제외)."""
        c2 = CandSet()
        c2.names = [nm for nm in self.names if nm in names]
        for at in ("kind", "b", "valid", "c0", "info"):
            setattr(c2, at, {k: v for k, v in getattr(self, at).items() if k in c2.names})
        c2.dropped = dict(self.dropped); c2.meta = dict(self.meta)
        return c2


def oof_base(cs, idx, yA, fid, K, trace=None):
    """교차 적합 기저 Z(len(idx) × K): 묶음 j 마다 L \\ L_j 로 보정한 계수로 L_j 를 예측한다."""
    idx = np.asarray(idx, int); fid = np.asarray(fid, int)
    Z = np.full((len(idx), cs.K), np.nan)
    for j in range(int(K)):
        te = fid == j
        if not te.any():
            continue
        tr_idx, te_idx = idx[~te], idx[te]
        if trace is not None:
            trace.append(dict(kind="zfold", fold=int(j), cal=tr_idx.copy(), pred=te_idx.copy()))
        prm = cs.calibrate(tr_idx, yA, trace)
        Z[te] = cs.matrix(prm, "A", te_idx)
    return Z


class StackFit:
    """적층 한 번의 결과. w_star 는 cs.names 순서의 가중, params 는 L 전체로 보정한 계수."""

    def __init__(self, cs, params, w_hat, w_star, gamma, fallback, n_lab, n_folds, fold_flag, support=(), dry=False):
        self.cs, self.params, self.w_hat, self.w_star = cs, params, np.asarray(w_hat, float), np.asarray(w_star, float)
        self.gamma, self.fallback, self.n_lab, self.n_folds, self.fold_flag = gamma, str(fallback), int(n_lab), int(n_folds), str(fold_flag)
        self.support, self.dry = tuple(support), bool(dry)
        self.is_p1 = _is_p1(self.w_star)

    @property
    def reuse_r1(self):
        """StackR 적합을 R1 의 성분으로 대신할 수 있는가(적층 = P1 이면 학습 행렬이 R1 과 같다). 세기(dry)에서는 라벨 없이 정한 대체 사유가 있을
        때만 다시 쓴다고 센다(γ̂ = 1 은 라벨이 있어야 알 수 있으므로 적합으로 센다)."""
        return bool(self.fallback) if self.dry else self.is_p1

    def pred(self, part, rows=None, src_params=False):
        """적층 예측. part = A(rows 필요), B, src. src 는 원천 계수 c0 로 낸다(전이 StackR 의 원천 행 목표)."""
        prm = self.cs.source_params() if (part == "src" or src_params) else self.params
        M = self.cs.matrix(prm, part, rows)
        return _wsum(M, self.w_star)

    def weights(self):
        return {nm: float(w) for nm, w in zip(self.cs.names, self.w_star) if w != 0.0}

    def log(self, n, d):
        return dict(n=int(n), draw=str(d), n_lab=self.n_lab, fallback=self.fallback, gamma=None if self.gamma is None else float(self.gamma),
                    folds=self.n_folds, fold_flag=self.fold_flag, K=self.cs.K, names=list(self.cs.names),
                    w_hat=[round(float(v), 10) for v in self.w_hat], w_star=[round(float(v), 10) for v in self.w_star], support=list(self.support),
                    is_p1=self.is_p1, dry=self.dry)


def fit_stack(cs, idx, yA, blkA, target, mode, split, n, draw, dry=False, trace=None):
    """선택 라벨 idx 의 적층(계획 2.2 '가중 학습' 1–5). 반환 StackFit. dry 이면 라벨을 쓰지 않고 대체 사유(n, 묶음)만 정한다."""
    idx = np.asarray(idx, int)
    e = np.zeros(cs.K); e[0] = 1.0
    reason = FB_N if len(idx) < STACK_MIN_N else ""
    if not reason:
        fid, Kf, fflag, reason = block_folds(np.asarray(blkA)[idx], target, mode, split, n, draw)
    else:
        fid, Kf, fflag = np.zeros(len(idx), int), 0, ""
    if dry:
        return StackFit(cs, cs.source_params(), e, e, None, reason, len(idx), Kf, fflag, dry=True)
    params = cs.calibrate(idx, yA, trace)
    if reason:
        return StackFit(cs, params, e, e, 1.0, reason, len(idx), Kf, fflag)
    y = np.asarray(yA, float)[idx]
    Z = oof_base(cs, idx, yA, fid, Kf, trace)
    w_hat, _, S = simplex_ls(Z, y, 0, trace)
    g, _ = choose_gamma(Z, y, fid, Kf, 0)
    w_star = (1.0 - g) * w_hat + g * e
    if g == 1.0:
        w_star = e.copy()
        reason = FB_GAMMA
    return StackFit(cs, params, w_hat, w_star, g, reason, len(idx), Kf, fflag, S)


def fit_stack0(cs, y_src, macro_src, trace=None):
    """Stack0(계획 2.2 '가중 학습' 7): 원천 행의 원천 거시 지역 하나 제외 교차 적합 기저에 단체 제약 최소제곱. 각 지역 행은 그 지역을 뺀 원천
    행의 최소제곱 계수(scale: LS, affine: 유효 셀 아핀, raw: 계수 없음)로 예측한다. γ 수축은 없다. 거시 지역이 2개 미만이면 P1."""
    y_src = np.asarray(y_src, float); mac = np.asarray(macro_src).astype(str)
    regs = sorted(set(mac.tolist()))
    e = np.zeros(cs.K); e[0] = 1.0
    if len(regs) < 2:
        return StackFit(cs, cs.source_params(), e, e, None, "regions<2", 0, len(regs), "")
    Z = np.full((len(y_src), cs.K), np.nan)
    for r in regs:
        te = mac == r
        tr = ~te
        prm = {}
        for nm in cs.names:
            k = cs.kind[nm]
            if k == "scale":
                E = H.ls_E(y_src[tr], cs.b[nm]["src"][tr])
                prm[nm] = float(E) if np.isfinite(E) else float(cs.c0[nm])
            elif k == "affine":
                v = cs.valid[nm]["src"] & tr
                al, bl = X.affine_ls(cs.b[nm]["src"][v], y_src[v])
                prm[nm] = (al, bl) if (np.isfinite(al) and np.isfinite(bl)) else cs.c0[nm]
            else:
                prm[nm] = None
        if trace is not None:
            trace.append(dict(kind="stack0_fold", region=r, n_cal=int(tr.sum()), n_pred=int(te.sum())))
        Z[te] = cs.matrix(prm, "src", np.where(te)[0])
    w, _, S = simplex_ls(Z, y_src, 0, trace)
    return StackFit(cs, cs.source_params(), w, w, None, "", 0, len(regs), "macro_loo", S)


# ================================================================ 4. 자료(후보 표를 문맥에 붙인다)
_TDD: dict = {}


def tdd_tables(tdd_v1=TDD_V1, tdd_v4=TDD_V4, allow_missing=False):
    """연도 정합 도일 표. 값 = lgx_tdd_matched_v1(v3 셀) + v4 표(새 셀). 표지(match_flag) = v4 표(v3 셀 포함). 두 표가 겹치는 셀은 값이
    같아야 한다. v4 표(1c 산출, 계획 2.2 '자료')가 없거나 sha256 앞 16자가 TDD_V4_SHA16 과 다르면 중단한다. allow_missing(명령행
    --allow-missing-tdd-v4)일 때만 없는 표를 허용한다(LGD 새 셀의 P* 가 빠져 K 가 줄어든다. 출처 dict 와 unit.json 에 적는다).
    반환 (값 Series, 표지 Series, 출처 dict)."""
    key = (str(tdd_v1), str(tdd_v4), bool(allow_missing), TDD_V4_SHA16)
    if key not in _TDD:
        p4 = Path(tdd_v4)
        if not p4.exists():
            if not allow_missing:
                raise SystemExit(f"[tdd] v4 새 셀 tdd_matched 표 {p4} 가 없다(계획 2.2 '자료', 1c 산출 --write-tdd-v4). 없이 돌리려면 "
                                 "--allow-missing-tdd-v4 를 준다(LGD 대상의 P* 가 빠진다)")
        else:
            sha = XB.sha256_file(p4)
            if sha[:16] != TDD_V4_SHA16:
                raise SystemExit(f"[tdd] {p4} 의 sha256 앞 16자 {sha[:16]} 가 기록값 {TDD_V4_SHA16} 과 다르다(1c 산출이 바뀌었다)")
        v = pd.read_csv(tdd_v1, usecols=["loc_id", "tdd_matched"]).drop_duplicates("loc_id").set_index("loc_id").tdd_matched.astype(float)
        src = dict(v1=f"{Path(tdd_v1).name}:{W.file_sha(tdd_v1)}", v4="없음(--allow-missing-tdd-v4)", v4_missing_allowed=not p4.exists())
        flags = pd.Series(dtype=object)
        if p4.exists():
            t = pd.read_csv(tdd_v4).drop_duplicates("loc_id").set_index("loc_id")
            both = t.index.intersection(v.index)
            if len(both):
                d = np.abs(t.loc[both, "tdd_matched"].values.astype(float) - v.loc[both].values)
                if np.nanmax(d) > 1e-6:
                    raise SystemExit(f"[tdd] {tdd_v4} 와 {tdd_v1} 의 겹치는 셀 값이 다르다(최대 차 {np.nanmax(d):.3g})")
            new = t.loc[t.index.difference(v.index), "tdd_matched"].astype(float)
            v = pd.concat([v, new])
            if "match_flag" in t:
                flags = t.match_flag.astype(str)
            src["v4"] = f"{Path(tdd_v4).name}:{W.file_sha(tdd_v4)}"
            src["v4_sha256_16"] = TDD_V4_SHA16
        _TDD[key] = (v, flags, src)
    return _TDD[key]


def tdd_args(a):
    """인자의 tdd 표 경로와 허용 표지(XC 의 h54 인자처럼 속성이 없으면 기본 경로·허용 없음)."""
    return getattr(a, "TDDM", TDD_V1), getattr(a, "TDDV4", TDD_V4), bool(getattr(a, "TDDV4_ALLOW_MISSING", False))


def _read_product(spec, name=""):
    """제품 표 'path[:열]' → loc_id 색인 DataFrame(value, mask_ok, train_dist_km). 마스크 제품(PRODUCT_MASK_REQUIRED)은 mask_ok 열, 5 km 규칙
    제품은 train_dist_km 열이 없으면 중단한다(누설 통제를 선택 열에 맡기지 않는다). 마스크 없는 L0 제품에 mask_ok 열이 없으면 모든 셀이 통과다."""
    path, col = (spec.split(":", 1) + ["value"])[:2] if ":" in spec else (spec, "value")
    t = pd.read_csv(path)
    need = ["loc_id", col] + (["mask_ok"] if PRODUCT_MASK_REQUIRED.get(name) else []) + (["train_dist_km"] if name in PRODUCTS and PRODUCTS[name][3] else [])
    miss = [c_ for c_ in need if c_ not in t.columns]
    if miss:
        raise SystemExit(f"[제품] {name} 표 {path} 에 열 {miss} 가 없다(계획 2.2 확장 후보의 누설 통제, 구현 기록 4절의 표 형식)")
    t = t.drop_duplicates("loc_id").set_index("loc_id")
    out = pd.DataFrame(index=t.index)
    out["value"] = t[col].astype(float)
    out["mask_ok"] = (t["mask_ok"].astype(float) >= 0.5) if "mask_ok" in t else True
    out["train_dist_km"] = t["train_dist_km"].astype(float) if "train_dist_km" in t else np.nan
    return out


_PROD: dict = {}


def product_tables(a):
    """변형의 제품 표. 인자에 VARIANT·PRODUCTS 가 없으면(XC 의 h54 인자) 핵심 판으로 본다."""
    variant, prods = str(getattr(a, "VARIANT", "") or ""), dict(getattr(a, "PRODUCTS", {}) or {})
    if variant == "":
        return {}
    key = (variant,) + tuple(sorted(prods.items()))
    if key not in _PROD:
        _PROD[key] = {nm: _read_product(sp, nm) for nm, sp in prods.items() if PRODUCTS[nm][0] == variant}
    return _PROD[key]


def pstar_target_miss(tv, df, tgt_idx):
    """대상 수준 P* 결측 비율(계획 2.2 '대상의 A 또는 채점 셀 가운데 5 % 넘게 P* 가 없으면 그 대상에서 P* 후보를 빼고'). A 쪽 = 어느 분할에서든
    A 풀이 될 수 있는 대상 셀 전체, 채점 쪽 = 대상의 eval_mask 셀 전체. 분할 집합에 기대지 않아 대상의 모든 분할(XC 의 새 seed 포함)에서 같은
    결정을 낸다. 라벨 값은 읽지 않는다. 반환 dict(miss_A, miss_B, n_A, n_B)."""
    t_idx = np.asarray(tgt_idx, int)
    ev = t_idx[eval_mask(df.iloc[t_idx])]

    def frac(ix):
        if not len(ix):
            return 0.0
        v = tv.reindex(df.loc_id.values[ix]).values.astype(float)
        return float(np.mean(~(np.isfinite(v) & (v > 0))))
    return dict(miss_A=frac(t_idx), miss_B=frac(ev), n_A=int(len(t_idx)), n_B=int(len(ev)))


def cands_from_df(a, df, src_idx, A_idx, B_idx, y_src, parent, variant="", tgt_idx=None):
    """자료 표의 색인(원천, A, 채점 셀)에서 후보 표를 만든다. 대상 라벨은 읽지 않는다(원천 라벨은 c0 에만). tgt_idx(대상 셀 전체)를 주면 P* 를
    뺄지는 대상 수준으로 정한다(pstar_target_miss)."""
    idx3 = dict(src=np.asarray(src_idx, int), A=np.asarray(A_idx, int), B=np.asarray(B_idx, int))
    col = lambda c_: {k: df[c_].values[ix].astype(float) for k, ix in idx3.items()}   # noqa: E731
    tv, flags, tsrc = tdd_tables(*tdd_args(a))
    loc = {k: df.loc_id.values[ix] for k, ix in idx3.items()}
    raw = {"P*": {k: tv.reindex(loc[k]).values.astype(float) for k in idx3}}
    for nm in ("Ku", "Ed", "Ss"):
        raw[nm] = col(SCALE_COLS[nm])
    prods = {}
    for nm, t in product_tables(a).items():
        r = {k: t.reindex(loc[k]) for k in idx3}
        prods[nm] = dict(src=r["src"].value.values, A=r["A"].value.values, B=r["B"].value.values,
                         mask_A=r["A"].mask_ok.fillna(False).values.astype(bool), mask_B=r["B"].mask_ok.fillna(False).values.astype(bool),
                         dist_src=r["src"].train_dist_km.values)
    pt = pstar_target_miss(tv, df, tgt_idx) if tgt_idx is not None else None
    cs = CandSet.build(col("s"), y_src, raw, col("cci_alt"), col("cci_valid"), prods, parent, variant, pstar_target=pt)
    if pt is not None:
        cs.meta["pstar_target"] = pt
    pre = {}
    if len(flags):
        for k in ("A", "B"):
            fl = flags.reindex(loc[k])
            pre[f"pre2010_{k}"] = float(np.mean(fl.values == "pre2010")) if len(fl) else np.nan
            pre[f"flag_missing_{k}"] = int(fl.isna().sum())
    else:
        pre = dict(pre2010_A=np.nan, pre2010_B=np.nan, flag_missing_A=-1, flag_missing_B=-1)
    cs.meta.update(tdd_src=tsrc, **pre)
    return cs


def attach_cands_t(a, c, alias, mode, split, variant=""):
    """전이 문맥(h40.Ctx, xbatch_core.build_tctx)에 후보 표를 붙인다(c.xb_cand). 색인은 h40.build_ctx 와 같은 규칙으로 다시 구하고 일치를
    단언한다(h42.build_unit 과 같다). XC 의 새 seed(201–220) 분할에도 쓴다."""
    spec, tgt, _ = XB.resolve(alias)
    _, D = W.get_data(a, spec)
    df = D.df
    t_idx, parent, src_idx, _ = D.source_idx(tgt, mode)
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = src_idx[ok]
    A_idx, B_idx = half_split_blocks(df, t_idx, int(split))
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    assert len(src) == len(c.y_src) and np.array_equal(df.y.values[src], c.y_src), "원천 색인이 문맥과 다르다"
    assert len(A_idx) == len(c.yA) and np.array_equal(df.y.values[A_idx], c.yA, equal_nan=True), "A 색인이 문맥과 다르다"
    assert len(evB) == len(c.yB) and np.array_equal(df.y.values[evB], c.yB, equal_nan=True), "채점 색인이 문맥과 다르다"
    assert np.array_equal(df.block.values[A_idx], c.blkA) and np.array_equal(df.block.values[evB], c.blkB), "블록이 문맥과 다르다"
    root = D.parent_of(tgt)
    c.xb_cand = cands_from_df(a, df, src, A_idx, evB, c.y_src, root, variant, tgt_idx=t_idx)
    return c


def build_t(a, alias, mode, split, variant=""):
    """xb_t 문맥: xbatch_core.build_tctx(h40.build_ctx, 실행 표 별칭 포함) + 후보 표."""
    c = XB.build_tctx(a, alias, mode, split)
    return attach_cands_t(a, c, alias, mode, split, variant)


def build_r(a, target, split, info, variant=""):
    """xb_r 문맥: h54.build_rctx(분할 1–25 의 구조 info) + 후보 표. 원천 = LG 모드 x 원천(100 km 버퍼, c0·ρ 전용)."""
    c = W.build_rctx(a, target, int(split), info)
    _, D = W.get_data(a)
    df = D.df
    t_idx = D.target_idx(target)
    A_idx, B_idx = half_split_blocks(df, t_idx, int(split))
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    _, parent, src_idx, _ = D.source_idx(target, "x")
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = src_idx[ok]
    assert np.array_equal(df.y.values[A_idx], c.yA, equal_nan=True) and np.array_equal(df.y.values[evB], c.yB, equal_nan=True), "색인이 문맥과 다르다"
    assert np.array_equal(df.block.values[A_idx], c.blkA) and np.array_equal(df.block.values[evB], c.blkB), "블록이 문맥과 다르다"
    y_src = df.y.values[src].astype(float)
    assert abs(float(H.ls_E(y_src, df.s.values[src].astype(float))) - float(c.E0)) <= 1e-12 * max(1.0, abs(c.E0)), "원천 E0 가 문맥과 다르다"
    c.xb_cand = cands_from_df(a, df, src, A_idx, evB, y_src, D.parent_of(target), variant, tgt_idx=t_idx)
    return c


# ================================================================ 5. 작업 단위
XB_ROW_DEFAULT = dict(stack_fallback="", stack_gamma=np.nan, stack_K=np.nan, stack_w="")


class _XBMixin:
    """적층 공용 부분: runs 의 적층 열(stack_fallback, stack_gamma, stack_K, stack_w), 적층 기록(notes['stack']), 적층 적합."""

    def _xb_setup(self):
        self.cs = self.c.xb_cand
        self._xb_extra = None
        self.stack_log: list = []
        self.notes["xb_cands"] = self.cs.describe()

    def add(self, method, learner, placement, n, d, seed, lam, pred, E_used=np.nan, n_lab=0, nb_lab=0, flag="", sel_info="", xb=None, **kw):
        key = super().add(method, learner, placement, n, d, seed, lam, pred, E_used, n_lab, nb_lab, flag, sel_info, **kw)
        if not self.dry and self.rows:
            ex = xb if xb is not None else self._xb_extra
            self.rows[-1].update(XB_ROW_DEFAULT if ex is None else ex)
        return key

    def stack(self, idx, n, d):
        """선택 라벨 idx 의 적층(계획 2.2 1–5). 라벨 사용을 추적에 남긴다."""
        return unit_stack(self, idx, n, d)

    @staticmethod
    def stack_extra(sf):
        return dict(stack_fallback=sf.fallback, stack_gamma=np.nan if sf.gamma is None else float(sf.gamma), stack_K=int(sf.cs.K),
                    stack_w=json.dumps({k: round(v, 8) for k, v in sf.weights().items()}, ensure_ascii=False))

    def finish(self):
        rows, st, stats = super().finish()
        stats["stack"] = list(self.stack_log)
        return rows, st, stats


class XBRUnit(_XBMixin, W.RUnit):
    """xb_r(지역 내, 모드 r). h54.RUnit 의 추출·계수·교차검증·잔차 적합을 그대로 쓴다(P1·R1·Re 는 WF6 와 같은 함수·seed)."""

    def __init__(self, a, c, exp="xb_r", variant="", dry=False, grid=XBR_GRID, draws=XBR_DRAWS):
        name = f"{c.target}|{c.mode}" if not variant else f"{c.target}~{variant}|{c.mode}"
        self._xb_extra = None
        self._init_base(a, c, exp, name, variant, dry)
        self._km: dict = {}
        self._std = None
        self._di = None
        self.grid, self.ndraws = tuple(int(v) for v in grid), int(draws)
        self._xb_setup()
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)

    def cells(self):
        cap = int(getattr(self.a, "draws_cap", 0) or 0)
        k = min(self.ndraws, cap) if cap > 0 else self.ndraws
        return XB.cells_grid(self.grid, k, self.nA)

    def lam_cv_stack(self, sel, n, d):
        """StackR 의 λ 교차검증(seed 0 적합): h54.RUnit.lam_cv 와 같은 묶음(cv_folds_of, 셀 묶음 포함)과 동률 규칙. 묶음마다 학습 쪽 라벨로
        적층을 다시 맞추고(안쪽 묶음 seed 의 추출 표지 = '추출.묶음'), 잔차 목표 = y − 적층(표본 안)."""
        c = self.c
        fid, K, flag = self.folds(sel, n, d)
        if K < 2:
            return LAM_BASE, int(K), flag or "folds<2"
        sse = {lam: 0.0 for lam in LAMS}
        seed0 = self.seeds[0]
        for j in range(K):
            tr, te = sel[fid != j], sel[fid == j]
            sf = self.stack(tr, n, f"{d}.{j}")
            a_tr, a_te = sf.pred("A", tr), sf.pred("A", te)
            (g,) = self.fit_resid(LO, "StackRcv", tr, a_tr, seed0, n, d, preds=[c.XA[te]], tag=f"cv{j}")
            for lam in LAMS:
                sse[lam] += float(np.sum((a_te + lam * np.asarray(g, float) - c.yA[te]) ** 2))
        if self.dry:
            return LAM_BASE, int(K), flag
        if not any(np.isfinite(sse[lam]) for lam in LAMS):
            return LAM_BASE, int(K), "cv_failed"
        return float(min(LAMS, key=lambda lam: (sse[lam] if np.isfinite(sse[lam]) else np.inf, lam))), int(K), flag

    def run(self):
        c, cs = self.c, self.cs
        for n, d in self.cells():
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            self.trace("select", sel, n=n, draw=str(d))
            E1, E2 = self.coefs(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            self.add("P1", "none", "cell", n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)                      # h54.RUnit.physics 와 같은 키·값
            lam = {m: self.lam_cv(m, LO, "x25", sel, n, d) for m in ("R1", "Re")}
            sf = self.stack(sel, n, d)
            self.stack_log.append(sf.log(n, d))
            xb = self.stack_extra(sf)
            self.add("Stack", "none", "cell", n, d, -1, 0.0, sf.pred("B"), np.nan, nl, nb, xb=xb)
            aeA, aeB = self.anchor_A("Re", E1, sel), self.anchor_B("Re", E1)
            self.add("Pe", "none", "cell", n, d, -1, 0.0, aeB, E1, nl, nb)
            if not self.dry:
                M = cs.matrix(sf.params, "B")
                for k, nm in enumerate(cs.names):
                    if nm == "P1":
                        continue
                    self.add("P*" if nm == "P*" else f"cand_{nm}", "none", "cell", n, d, -1, 0.0, M[:, k],
                             sf.params[nm] if cs.kind[nm] == "scale" else np.nan, nl, nb)
            lcv_s = self.lam_cv_stack(sel, n, d)
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            stA, stB = sf.pred("A", sel), sf.pred("B")
            for seed in self.seeds:
                (g,) = self.fit_resid(LO, "R1", sel, a1A, seed, n, d)
                fl1 = self.F.last_flag
                g1 = np.asarray(g, float)
                lcv, K, cf = lam["R1"]
                self.emit("R1", LO, n, d, seed, a1B, g1, E1, nl, nb, lcv, K, cf, fl1, nrow=nl)
                (g,) = self.fit_resid(LO, "Re", sel, aeA, seed, n, d)
                fl = self.F.last_flag
                g = np.asarray(g, float)
                lcv, K, cf = lam["Re"]
                self.add("Re", LO, "cell", n, d, seed, LAM_BASE, aeB + LAM_BASE * g, E1, nl, nb, flag=fl, nrow=nl)
                self.add("Re", LO, "cell", n, d, seed, LAM_CV, aeB + lcv * g, E1, nl, nb, flag=fl, sel_info=f"lam={lcv}", cv_folds=K, cv_flag=cf,
                         nrow=nl)
                if sf.reuse_r1:                                             # 적층 = P1: 학습 행렬이 R1 과 같아 적합을 다시 하지 않는다
                    gS, flS = g1, (fl1 + ";" if fl1 else "") + "reuse_R1"
                else:
                    (gS,) = self.fit_resid(LO, "StackR", sel, stA, seed, n, d)
                    gS, flS = np.asarray(gS, float), self.F.last_flag
                lcv, K, cf = lcv_s
                self._xb_extra = xb
                self.emit("StackR", LO, n, d, seed, stB, gS, np.nan, nl, nb, lcv, K, cf, flS, nrow=nl)
                self._xb_extra = None
        return self


class XBTUnit(_XBMixin, W.T7Unit):
    """xb_t(전이). 문맥·원천·추출·seed 와 P0·P1·Pe·R1·Re 의 행렬은 h54.T7Unit(wf7)과 같다. 적층·StackR·Stack0·B:ens 를 더한다."""

    def __init__(self, a, c, alias, variant="", dry=False, grid=XBT_GRID, draws=XBT_DRAWS, exp="xb_t"):
        name = f"{alias}|{c.mode}" if not variant else f"{alias}~{variant}|{c.mode}"
        self._xb_extra = None
        self._init_base(a, c, exp, name, variant, dry)                    # h54.TUnit.__init__ 와 같은 내용(저장소 이름만 변형을 넣는다)
        self.alias = alias
        self.nsrc = len(c.y_src)
        self.n_ps = int(round(W.R_PS * self.nsrc))
        self._ps: dict = {}
        self.diag: list = []
        self.grid, self.ndraws = tuple(int(v) for v in grid), int(draws)
        self._xb_setup()
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)
        self.re_src = W.re_src_resid(c)                                    # h54.T7Unit.__init__ 와 같다
        ok = np.isfinite(c.X_src[:, CCI_COL]) & np.isfinite(c.X_src[:, CCIV_COL]) & (c.X_src[:, CCIV_COL] >= 0.5)
        okB = np.isfinite(c.XB[:, CCI_COL]) & np.isfinite(c.XB[:, CCIV_COL]) & (c.XB[:, CCIV_COL] >= 0.5)
        self.notes["cci_valid_frac"] = dict(src=float(ok.mean()) if len(ok) else np.nan, B=float(okB.mean()) if len(okB) else np.nan)

    def cells(self):
        cap = int(getattr(self.a, "draws_cap", 0) or 0)
        k = min(self.ndraws, cap) if cap > 0 else self.ndraws
        return XB.cells_grid(self.grid, k, self.nA, zero_n=True)

    def run(self):
        c, cs = self.c, self.cs
        for n, d in self.cells():
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            E1, _ = self.coefs(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            aeA = W.re_anchor(a1A, c.XA[sel, CCI_COL], c.XA[sel, CCIV_COL])
            aeB = W.re_anchor(a1B, c.XB[:, CCI_COL], c.XB[:, CCIV_COL])
            self.add("P1", "none", "cell", n, d, -1, 0.0, a1B, E1, nl, nb)
            self.add("Pe", "none", "cell", n, d, -1, 0.0, aeB, E1, nl, nb)
            sf = self.stack(sel, n, d)
            self.stack_log.append(sf.log(n, d))
            xb = self.stack_extra(sf)
            self.add("Stack", "none", "cell", n, d, -1, 0.0, sf.pred("B"), np.nan, nl, nb, xb=xb)
            if "P*" in cs.names and not self.dry:
                k = cs.names.index("P*")
                self.add("P*", "none", "cell", n, d, -1, 0.0, cs.matrix(sf.params, "B")[:, k], sf.params["P*"], nl, nb)
            if n == 0 and not self.dry:
                self.zero_n_baselines()
            stA, stB = sf.pred("A", sel), sf.pred("B")
            for seed in self.seeds:
                (g1,) = self.fit(LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel)
                fl1 = self.F.last_flag
                g1 = np.asarray(g1, float)
                self.add("R1", LO, "cell", n, d, seed, LAM_BASE, a1B + LAM_BASE * g1, E1, nl, nb, flag=fl1, nrow=self.nsrc + nl)
                (ge,) = self.fit(LO, "Re", lambda: self.rows_Re(sel, aeA), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.add("Re", LO, "cell", n, d, seed, LAM_BASE, aeB + LAM_BASE * np.asarray(ge, float), E1, nl, nb, flag=self.F.last_flag,
                         nrow=self.nsrc + nl)
                if sf.reuse_r1 and self.cs.resid_src_keep is None:          # 적층 = P1: 학습 행렬이 R1 과 같다(원천 c0 = E0, 라벨 E_n)
                    gS, flS = g1, (fl1 + ";" if fl1 else "") + "reuse_R1"
                else:
                    (gS,) = self.fit(LO, "StackR", lambda: stackr_rows(self, sf, sel, stA), self.nsrc + nl, seed, [c.XB], n, d, sel)
                    gS, flS = np.asarray(gS, float), self.F.last_flag
                self.add("StackR", LO, "cell", n, d, seed, LAM_BASE, stB + LAM_BASE * gS, np.nan, nl, nb, flag=flS, nrow=self.nsrc + nl, xb=xb)
        return self

    def zero_n_baselines(self):
        """n 0 의 Stack0 와 B:ens(키 n 0, 추출 0). Stack0 는 Stack0 제외 후보(Wei)를 뺀 후보로 맞춘다."""
        c, cs = self.c, self.cs
        cs0 = cs.subset([nm for nm in cs.names if nm not in cs.stack0_exclude])
        sf0 = fit_stack0(cs0, c.y_src, c.macro_src)
        self.notes["stack0"] = dict(names=list(cs0.names), w=[round(float(v), 10) for v in sf0.w_star], fallback=sf0.fallback, regions=sf0.n_folds,
                                    excluded=sorted(cs.stack0_exclude))
        self.add("Stack0", "none", "cell", 0, 0, -1, 0.0, sf0.pred("B"), np.nan, 0, 0, xb=dict(XB_ROW_DEFAULT, stack_fallback=sf0.fallback,
                                                                                             stack_K=int(cs0.K), stack_w=json.dumps(sf0.weights())))
        if cs.ens is not None:
            e = cs.ens
            pr = (c.E0 * c.sB + e["c0_ku"] * e["ku_B"] + e["c0_cci"] * e["cci_B"]) / 3.0
            self.add("B:ens", "none", "cell", 0, 0, -1, 0.0, pr, np.nan, 0, 0)


def stackr_rows(unit, sf, idx, st_lab=None):
    """전이 StackR 의 학습 행렬(h40 R1 과 같은 구성): 원천 행 목표 = y − 적층(원천 계수 c0, w*), 선택 라벨 행 목표 = y − 적층(표본 안).
    확장 판(Wei)은 학습 지점 5 km 안의 원천 행을 뺀다. 원천 적층이 비유한인 행도 뺀다(핵심 판에서는 없다)."""
    c = unit.c
    st_src = sf.pred("src")
    keep = np.isfinite(st_src)
    if unit.cs.resid_src_keep is not None:
        keep &= unit.cs.resid_src_keep
    st_lab = sf.pred("A", idx) if st_lab is None else st_lab
    return X.stack_rows(c.X_src[keep], np.asarray(c.y_src, float)[keep] - st_src[keep], c.XA[idx], c.yA[idx] - np.asarray(st_lab, float), None, None)


# ================================================================ 6. W+(XC 의 A1+ 팔, 계획 2.3)
def unit_stack(unit, idx, n, d):
    """작업 단위(XB 단위 또는 XC 단위 = h54 UnitBase 하위 클래스, 문맥에 c.xb_cand)에서 선택 라벨 idx 의 적층. 보정·교차 적합이 쓴 라벨 색인을
    h54 추적(WF_TRACE)에 남긴다(누설 시험의 selected_union 이 읽는다)."""
    c = unit.c
    tr = [] if W.WF_TRACE is not None else None
    sf = fit_stack(c.xb_cand, idx, c.yA, c.blkA, c.target, c.mode, c.split, n, d, dry=unit.dry, trace=tr)
    for e in tr or []:
        if "idx" in e:
            unit.trace("coef", e["idx"], n=n, draw=str(d), what="stack")
        if e.get("kind") == "zfold":
            unit.trace("cv", e["cal"], method="Stack", n=n, draw=str(d), fold=e["fold"], held=e["pred"])
    return sf


def _stackr_g(unit, sf, idx, seed, n, d, preds, g_r1, method, **tr):
    """StackR 잔차 성분. 적층 = P1 이면 R1 의 성분(같은 학습 행렬)을 다시 쓴다."""
    if sf.reuse_r1 and getattr(unit.c.xb_cand, "resid_src_keep", None) is None and g_r1 is not None:
        return np.asarray(g_r1, float), "reuse_R1"
    (g,) = unit.fit(LO, method, lambda: stackr_rows(_U(unit), sf, idx), unit.nsrc + len(idx), seed, preds, n, d, idx, **tr)
    return np.asarray(g, float), unit.F.last_flag


class _U:
    """stackr_rows 가 쓰는 최소 속성(c, cs). XC 단위에 cs 속성이 없어도 되게 한다."""

    def __init__(self, unit):
        self.c = unit.c
        self.cs = unit.c.xb_cand


def select_w_wplus(unit, sel, n, d):
    """규칙 W 와 W+ 를 한 번의 선택 라벨 안 5겹 블록 교차검증(seed 0 적합)으로 함께 고른다. W 부분(후보 h54.W_CANDS)의 묶음·적합·SSE 누적·동률
    순서는 h54.TUnit.select_w 와 같아 같은 선택을 낸다. W+ 는 W_CANDS 다음에 Stack, StackR@0.25 를 둔다(동률 순서). 묶음 j 의 Stack 은
    학습 쪽 라벨로 다시 맞춘 적층(안쪽 묶음 추출 표지 '추출.묶음'), StackR@0.25 는 원천 행 + 학습 쪽 라벨의 잔차 모형이다. 라벨 < 10 이면 둘 다 P1.
    반환 dict(W=(선택, 묶음 수, 표지, 묶음별 RMSE), Wplus=(같은 형식))."""
    c = unit.c
    if len(sel) < W.W_MIN_N:
        r = ("P1", 0, f"n<{W.W_MIN_N}", {})
        return dict(W=r, Wplus=r)
    fid, K, flag = unit.folds(sel, n, d)
    if K < 2:
        r = ("P1", int(K), flag or "folds<2", {})
        return dict(W=r, Wplus=r)
    seed0 = unit.seeds[0]
    sse = {m: 0.0 for m in WPLUS_CANDS}
    for j in range(K):
        tr, te = sel[fid != j], sel[fid == j]
        unit.trace("cv", tr, method="W", n=n, draw=str(d), fold=j, held=te)
        E1, E2 = unit.coefs(tr)
        s, y = c.sA[te], c.yA[te]
        a_tr, a_te = E1 * c.sA[tr], E1 * s
        nl = len(tr)
        (g1,) = unit.fit(LO, "R1cv", lambda: unit.rows_R(tr, a_tr), unit.nsrc + nl, seed0, [c.XA[te]], n, d, tr, fold=j)
        (g2,) = unit.fit(LO, "R2cv", lambda: unit.rows_R(tr, a_tr, seed0, E1), unit.nsrc + nl + unit.n_ps, seed0, [c.XA[te]], n, d, tr, fold=j)
        (pd1,) = unit.fit(LO, "D1cv", lambda: unit.rows_D(tr, seed0, E1), unit.nsrc + nl + unit.n_ps, seed0, [c.XA[te]], n, d, tr, fold=j)
        g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
        sf = unit_stack(unit, tr, n, f"{d}.{j}")
        st_te = sf.pred("A", te)
        gS, _ = _stackr_g(unit, sf, tr, seed0, n, d, [c.XA[te]], g1, "StackRcv", fold=j)
        pr = {"P0": c.E0 * s, "P1": E1 * s, "P2": E2 * s, "R1@0.25": a_te + 0.25 * g1, "R1@1.0": a_te + g1, "R2@0.25": a_te + 0.25 * g2,
              "D1": np.asarray(pd1, float), "Stack": st_te, "StackR@0.25": st_te + 0.25 * gS}
        for m in WPLUS_CANDS:
            sse[m] += float(np.sum((pr[m] - y) ** 2))
    if unit.dry:
        r = ("P1", int(K), flag, {})
        return dict(W=r, Wplus=r)
    best_w = min(W.W_CANDS, key=lambda m: (sse[m] if np.isfinite(sse[m]) else np.inf, W.W_CANDS.index(m)))
    best_p = min(WPLUS_CANDS, key=lambda m: (sse[m] if np.isfinite(sse[m]) else np.inf, WPLUS_CANDS.index(m)))
    rm = {m: float(np.sqrt(v / max(len(sel), 1))) for m, v in sse.items()}
    return dict(W=(best_w, int(K), flag, {m: rm[m] for m in W.W_CANDS}), Wplus=(best_p, int(K), flag, rm))


def wplus_predictions(unit, sel, n, d, seed, g_r1=None):
    """채점 셀의 W+ 추가 후보 예측 dict(Stack, StackR@0.25)와 적층 기록. g_r1 = 같은 seed 의 R1 잔차 성분(적층 = P1 일 때 다시 쓴다)."""
    _ensure_cands(unit)
    c = unit.c
    sf = unit_stack(unit, np.asarray(sel, int), n, d)
    stB = sf.pred("B")
    gS, flag = _stackr_g(unit, sf, np.asarray(sel, int), seed, n, d, [c.XB], g_r1, "StackR")
    return {"Stack": stB, "StackR@0.25": stB + 0.25 * gS}, dict(sf.log(n, d), stackr_flag=flag)


def _ensure_cands(unit):
    """XC 단위의 문맥에 후보 표가 없으면 붙인다(attach_cands_t, 별칭 = unit.alias). 이미 있으면 그대로 쓴다."""
    c = unit.c
    if getattr(c, "xb_cand", None) is None:
        attach_cands_t(unit.a, c, getattr(unit, "alias", None) or c.meta.get("alias", c.target), c.mode, c.split)
    return c.xb_cand


def xc_wplus_cv(unit, tr, te, n, d, fold, seed):
    """XC 공급자 계약(x_workflow_end_to_end.WPlusProvider): 교차검증 묶음 하나의 W+ 추가 후보. 학습 쪽 라벨 tr 로 적층(안쪽 블록 묶음의 추출
    표지 '추출.묶음')과 StackR(원천 행 + tr 행, seed)을 맞추고 te 를 예측한다. select_w_wplus 의 묶음 안 계산과 같다(R1cv 성분 재사용만 없다.
    같은 행렬·seed 의 적합이라 값은 같다). 반환 {'Stack': te 예측, 'StackR@0.25': te 예측}."""
    _ensure_cands(unit)
    c = unit.c
    tr, te = np.asarray(tr, int), np.asarray(te, int)
    sf = unit_stack(unit, tr, n, f"{d}.{fold}")
    st_te = sf.pred("A", te)
    gS, _ = _stackr_g(unit, sf, tr, seed, n, d, [c.XA[te]], None, "StackRcv", fold=fold)
    return {"Stack": st_te, "StackR@0.25": st_te + 0.25 * gS}


def xc_wplus_final(unit, sel, n, d, seed):
    """XC 공급자 계약: 선택 라벨 sel 전체로 맞춘 적층과 StackR(seed)의 채점 셀 예측. 반환 {'Stack': B 예측, 'StackR@0.25': B 예측}."""
    pr, _ = wplus_predictions(unit, sel, n, d, seed, None)
    return pr


# ================================================================ 7. 인자·단위 목록·실행
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XB_multisource_stacking(계획 2.2): 라벨 가중 적층과 적층 잔차")
    ap.add_argument("--exp", "--exps", dest="exps", default="xb_r,xb_t", help="xb_r, xb_t 쉼표 목록")
    ap.add_argument("--variant", default="", choices=list(VARIANTS), help="'' = 핵심 후보 7, cci5y = 2단계 민감도, ext = 3단계 확장(Wei·YK)")
    ap.add_argument("--product", action="append", default=[], help="제품 표 '이름=경로[:열]'(이름 cci5y, Wei, YK. 표 형식은 구현 기록 4절)")
    ap.add_argument("--targets-r", default=",".join(XBR_TARGETS))
    ap.add_argument("--targets-t", default="", help="'이름:모드' 쉼표 목록. 기본 = LG 27 (대상, 모드) + Tibet_LGD, Russia_C~lgd, NAtlantic~lic, Canada~exp~lic")
    ap.add_argument("--grid-r", default="")
    ap.add_argument("--grid-t", default="")
    ap.add_argument("--splits-r", type=int, default=len(XBR_SPLITS), help="지역 내 분할 1..K(기본 25)")
    ap.add_argument("--splits", type=int, default=5, help="전이 분할 1..K(기본 5)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--draws-cap", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="프로세스 수. 0 = 풀 없이 차례로")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--out-dir", default=str(XB.out_dir(EXP_ID).relative_to(ROOT)))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lgd-dir", default="data/processed/lgd/run_tables")
    ap.add_argument("--tdd-matched", default=str(TDD_V1.relative_to(ROOT)))
    ap.add_argument("--tdd-v4", default=str(TDD_V4.relative_to(ROOT)),
                    help="v4 새 셀 tdd_matched 표(--write-tdd-v4 의 산출, sha256 앞 16자 고정). 없거나 해시가 다르면 중단")
    ap.add_argument("--allow-missing-tdd-v4", action="store_true",
                    help="v4 표 없이 돈다(LGD 새 셀의 P* 가 빠져 K 가 준다). unit.json 의 tdd_v4_missing_allowed 에 적는다")
    ap.add_argument("--mem-wait", type=float, default=1800.0, help="로컬 실행 시작 전 가용 메모리 30 GB 를 기다리는 최대 초(계획 1절 로컬 자원)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="", help="i/N: 단위 목록(이름 순)의 N 등분 가운데 i 번째(1 부터)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--pool-retries", type=int, default=2)
    ap.add_argument("--gate", action="store_true", help="재현 관문: xb_r 의 P1·R1 과 WF6, xb_t 의 P0·P1·R1·Re 와 WF7 조각의 블록 SSE 대조")
    ap.add_argument("--gate-dir", default="results/rescale_wf2/data/processed/wf/shards")
    ap.add_argument("--gate-level", default="same_node", choices=list(XB.GATE_TOL))
    ap.add_argument("--write-tdd-v4", action="store_true", help="1c: v4 새 셀 tdd_matched 표(a2 정의, 라벨 값 미사용)")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(p):
    return Path(p) if os.path.isabs(str(p)) else ROOT / str(p)


def _n_list(txt):
    return [-1 if v.strip() == "all" else int(v) for v in str(txt).split(",") if v.strip()]


def finalize(a):
    a.PERMIT = XB.run_permitted(a.ARGV)
    a.threads_asked = int(a.threads)
    a.threads, nb = XB.local_limits(a.threads, a.nboot, a.ARGV)
    a.nboot_capped = nb < int(a.nboot)
    a.nboot = nb
    a.EXPS = [v.strip() for v in a.exps.split(",") if v.strip() in ARMS]
    a.VARIANT = str(a.variant)
    a.PRODUCTS = {}
    for spec in a.product:
        nm, path = spec.split("=", 1)
        if nm not in PRODUCTS:
            raise SystemExit(f"--product 이름은 {sorted(PRODUCTS)} 가운데 하나다: {nm}")
        a.PRODUCTS[nm] = str(_abs(path.split(":")[0])) + (":" + path.split(":", 1)[1] if ":" in path else "")
    if a.VARIANT and not any(PRODUCTS[k][0] == a.VARIANT for k in a.PRODUCTS):
        raise SystemExit(f"--variant {a.VARIANT} 에는 그 변형의 --product 가 필요하다")
    a.T_R = [v.strip() for v in a.targets_r.split(",") if v.strip()]
    a.T_T = ([tuple(v.split(":")) if ":" in v else (v, "x") for v in a.targets_t.split(",") if v.strip()] if a.targets_t
             else list(H.default_targets(["i", "x"])) + [(k, "x") for k in XBT_LGD])
    a.G_R = _n_list(a.grid_r) if a.grid_r else list(XBR_GRID)
    a.G_T = _n_list(a.grid_t) if a.grid_t else list(XBT_GRID)
    a.SPLITS_R = list(range(1, int(a.splits_r) + 1))
    a.SPLITS_T = list(range(1, int(a.splits) + 1))
    a.OUT = _abs(a.out_dir)
    a.SHARDS = a.OUT / "shards"
    a.SUFFIX = ""
    a.SEALED_ROOT, a.SEALED_NAME = a.OUT.parent, a.OUT.name
    if a.smoke:                                                            # 스모크: 모든 경로를 작은 설정으로 한 번씩 지난다(조각은 봉인 폴더)
        a.T_R = ["Canada"]; a.SPLITS_R = [1]; a.G_R = [20, -1]
        a.T_T = [("Russia_W", "x"), ("Tibet_LGD", "x")]; a.SPLITS_T = [1]; a.G_T = [0, 3, 10, -1]
        a.seeds = 1; a.draws_cap = a.draws_cap or 1; a.nboot = min(int(a.nboot), 200)
        a.SHARDS = XB.sealed_dir(a.SEALED_NAME, a.SEALED_ROOT) / "smoke_shards"
        a.SUFFIX = "_smoke"
    a.TAG = {arm: (f"{a.tag}{TAGS[arm][2:]}" if a.tag else TAGS[arm]) + a.SUFFIX for arm in ARMS}
    a.TDDM, a.TDDV4 = _abs(a.tdd_matched), _abs(a.tdd_v4)
    a.TDDV4_ALLOW_MISSING = bool(a.allow_missing_tdd_v4)
    a.ON_RESCALE = os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1"
    a.SMOKE_ENV = XB.smoke_env_report() if a.smoke else None              # 스모크 규약 점검(코어 4, 스레드 2, 가용 메모리). 조각 unit.json·집계 메타
    a.ha = XB.h54_args(exp="wf6", splits=XB.TRANSFER_SPLITS, threads=a.threads, cb_iters=a.cb_iters, seeds=a.seeds, nboot=a.nboot,
                       data_dir=a.data_dir, lgd_dir=a.lgd_dir, draws_cap=a.draws_cap, allow_local=a.PERMIT)
    a.ha.SEEDS = list(range(int(a.seeds)))
    a.ha.TDDM, a.ha.TDDV4, a.ha.VARIANT, a.ha.PRODUCTS = a.TDDM, a.TDDV4, a.VARIANT, a.PRODUCTS
    a.ha.TDDV4_ALLOW_MISSING = a.TDDV4_ALLOW_MISSING
    a.ha.draws_cap = int(a.draws_cap)
    return a


def unit_cfg(a, arm, variant, data_sha):
    """결과에 영향을 주는 설정(대상·분할·tag·워커 제외). xbatch_core.make_unit_cfg 의 공통 항목에 XB 고유 항목을 더한다."""
    _, _, tsrc = tdd_tables(*tdd_args(a))
    prod = {k: W.file_sha(v.split(":")[0]) for k, v in sorted(a.PRODUCTS.items()) if PRODUCTS[k][0] == variant}
    grid, draws = (a.G_R, XBR_DRAWS) if arm == "xb_r" else (a.G_T, XBT_DRAWS)
    methods = (["P0", "P1", "P*", "cand_*", "Pe", "Stack", "R1", "Re", "StackR"] if arm == "xb_r"
               else ["P0", "P1", "P*", "Pe", "Stack", "R1", "Re", "StackR", "Stack0", "B:ens"])
    return XB.make_unit_cfg(arm, variant, data_sha, grid=[int(v) for v in grid], draws=int(draws), draws_cap=int(a.draws_cap),
                            seeds=list(range(int(a.seeds))), cb_iters=int(a.cb_iters), cands=list(CANDS_CORE), gammas=list(GAMMAS),
                            stack_min_n=STACK_MIN_N, min_folds=MIN_FOLDS, fold_rule="cv_folds_of(A 블록) 의 묶음 수 K < 2 = 묶음 부족(셀 묶음도 K ≥ 2 이면 적층)",
                            pstar_max_miss=PSTAR_MAX_MISS, tie_rel=TIE_REL, neg_tol=NEG_TOL, tdd=tsrc, products=prod, methods=methods,
                            lam_cv="h54.lam_cv 묶음, 묶음마다 적층 재적합" if arm == "xb_r" else "", wei_km=WEI_KM,
                            pstar_rule="대상 수준(대상 셀 전체·대상 채점 셀)", wei_km5_scope="확장 판 전체", mask_required=dict(PRODUCT_MASK_REQUIRED),
                            draw="h40.draw_cells(대상, 모드, 분할, n, 추출)", splits_rule="wf6 1–25" if arm == "xb_r" else "h40 1–5")


def _unit_name(u):
    return "|".join(str(v) for v in u if v != "")


def enumerate_units(a):
    """작업 단위 (arm, 대상, 모드, 분할, 변형)과 건너뛴 기록, 기대 분할."""
    units, skipped, expected = [], [], {}
    v = a.VARIANT
    if "xb_r" in a.EXPS:
        _, D = W.get_data(a.ha)
        for t in a.T_R:
            info = W.split_structure_ext(D, t, XBR_SPLITS)
            keep, skip, _ = W.split_plan(D, t, [s for s in a.SPLITS_R if s in info], info)
            expected[("xb_r", t, "r")] = keep
            skipped += [dict(arm="xb_r", target=t, mode="r", split=sp, status=st_) for sp, st_, _ in skip]
            units += [("xb_r", t, "r", sp, v) for sp in keep]
    if "xb_t" in a.EXPS:
        for alias, mode in a.T_T:
            spec, tgt, _ = XB.resolve(alias)
            if spec is not None and not (a.ha.LGD / spec / "fidelity_base_v3.csv").exists():
                skipped.append(dict(arm="xb_t", target=alias, mode=mode, split=-1, status="lgd_table_missing"))
                continue
            _, D = W.get_data(a.ha, spec)
            keep, skip, _ = W.split_plan(D, tgt, a.SPLITS_T)
            expected[("xb_t", alias, mode)] = keep
            skipped += [dict(arm="xb_t", target=alias, mode=mode, split=sp, status=st_) for sp, st_, _ in skip]
            units += [("xb_t", alias, mode, sp, v) for sp in keep]
    return shard_select(units, a.shard), skipped, expected


def shard_select(units, spec=""):
    """--shard i/N: 단위 목록을 이름 순으로 정렬해 k % N == i − 1 인 단위만 고른다(N 개 작업이 겹치지 않고 모두를 덮는다)."""
    units = sorted(units, key=_unit_name)
    if not spec:
        return units
    i, N = (int(x) for x in str(spec).split("/"))
    if not (1 <= i <= N):
        raise SystemExit("--shard 는 i/N(1 ≤ i ≤ N)")
    return [u for k, u in enumerate(units) if k % N == i - 1]


def make_unit(a, arm, target, mode, split, variant="", dry=False):
    """문맥과 작업 단위 객체."""
    if arm == "xb_r":
        _, D = W.get_data(a.ha)
        info = W.split_structure_ext(D, target, XBR_SPLITS)[int(split)]
        c = build_r(a.ha, target, split, info, variant)
        return c, XBRUnit(a.ha, c, "xb_r", variant, dry, grid=a.G_R)
    c = build_t(a.ha, target, mode, split, variant)
    return c, XBTUnit(a.ha, c, target, variant, dry, grid=a.G_T)


def data_sha_of(a, arm, target):
    return XB.data_sha(a.ha, target if arm == "xb_t" else None)


def run_unit(a, arm, target, mode, split, variant="", dry=False, expected=None):
    t0 = time.time()
    c, U = make_unit(a, arm, target, mode, split, variant, dry)
    U.run()
    rows, st, stats = U.finish()
    if dry:
        return dict(arm=arm, target=target, mode=mode, split=int(split), variant=variant, n_A=len(c.yA), n_eval=len(c.yB), n_rows=stats["n_rows"],
                    fit_total=int(sum(stats["n_fit"].values())), est_s=round(float(sum(stats["est_detail"].values())), 2),
                    _detail=stats["n_fit_detail"], stack=stats["stack"], K=c.xb_cand.K, dropped=dict(c.xb_cand.dropped))
    cfg = unit_cfg(a, arm, variant, data_sha_of(a, arm, target))
    meta = {k: v for k, v in c.meta.items() if not isinstance(v, (dict, list, np.ndarray))}
    unit = {**stats, **meta}
    cs = c.xb_cand
    tsrc = tdd_tables(*tdd_args(a))[2]
    unit.update(arm=arm, alias=target, n_A=int(len(c.yA)), n_eval=int(len(c.yB)), E0=float(c.E0), elapsed_s=round(time.time() - t0, 1),
                n_fit_total=int(sum(stats["n_fit"].values())), xb_cands=cs.describe(), plan_section=PLAN_SEC,
                tdd_src=tsrc, tdd_v4_missing_allowed=bool(tsrc.get("v4_missing_allowed", False)), K=int(cs.K),
                pstar_dropped=cs.dropped.get("P*", ""), pstar_target=cs.meta.get("pstar_target"),
                resid_src_excluded_km5=int((~cs.resid_src_keep).sum()) if cs.resid_src_keep is not None else 0,
                runs_sealed_cols=list(RUNS_SEALED_COLS))
    if getattr(a, "SMOKE_ENV", None) is not None:
        unit["smoke_env"] = a.SMOKE_ENV
    return XB.write_shard(a.SHARDS, a.TAG[arm], target, mode, split, public_runs(rows), st, cfg, unit=unit, variant=variant,
                          expected=expected if expected is not None else [int(split)], code_file=__file__)


def public_runs(rows):
    """조각 runs.csv 에 쓰는 행: RMSE·편향 열(RUNS_SEALED_COLS)을 뺀다(계획 0.3: 열람 순서 전의 Stack − P1 방향을 봉인 밖 파일에서 읽을 수
    없게 한다). 집계·관문·대체 비율은 blocksse.npz 와 나머지 열(method, n, split, draw, stack_fallback 등)만 쓴다."""
    return [{k: v for k, v in r.items() if k not in RUNS_SEALED_COLS} for r in rows]


_WA = None
_WEXP = None


def _worker_init(argv, expected_items):
    global _WA, _WEXP
    _WA = parse_args(argv)
    _WEXP = {tuple(k): v for k, v in expected_items}


def _worker_run(arm, target, mode, split, variant):
    t0 = time.time()
    u = run_unit(_WA, arm, target, mode, split, variant, expected=_WEXP.get((arm, target, mode)))
    return dict(arm=arm, target=target, mode=mode, split=int(split), variant=variant, status=u.get("status"), n_fit=u.get("n_fit_total"),
                n_fit_fail=int(sum((u.get("fail") or {}).values())) + int(u.get("n_nonfinite_keys", 0)), wall_s=round(time.time() - t0, 1))


# ================================================================ 8. 세기(라벨 값 미사용)
def count_only(a, units, skipped):
    """학습 없이(dry) 적합 수, 추정 시간, 라벨 없이 셀 수 있는 대체 사유(n < 10, cv_folds_of 묶음 수 K < 2)의 (대상, n) 별 수를 센다.
    라벨이 한 블록에만 있어 셀 묶음으로 적층한 추출 수(n_cell_folds, 대체 아님)도 서술로 센다. 표준 출력에는 셀·적합·추출 수만 쓴다(계획 0.3)."""
    t0 = time.time()
    rows, det, fb = [], Counter(), []
    for u in units:
        r = run_unit(a, *u, dry=True)
        for k, v in r.pop("_detail").items():
            det[f"{u[0]}|{k}"] += v
        for e in r.pop("stack"):
            fb.append(dict(arm=u[0], target=u[1], mode=u[2], split=u[3], variant=u[4], n=e["n"], draw=e["draw"], n_lab=e["n_lab"],
                           fallback_label_free=e["fallback"], fold_flag=e.get("fold_flag", "")))
        r["dropped"] = ";".join(f"{k}:{v}" for k, v in r["dropped"].items())
        rows.append(r)
    df = pd.DataFrame(rows)
    out = a.OUT
    XB.check_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "xb" + a.SUFFIX + (f"_{a.VARIANT}" if a.VARIANT else "")
    if not len(df):
        print("[count-only] 작업 단위 없음", flush=True)
        return df
    df.to_csv(out / f"{tag}_count.csv", index=False)
    fbd = pd.DataFrame(fb)
    if len(fbd):
        g = fbd.groupby(["arm", "target", "mode", "n"])
        fbs = g.agg(n_draws=("draw", "size"), n_fb_n=("fallback_label_free", lambda s: int((s == FB_N).sum())),
                    n_fb_folds=("fallback_label_free", lambda s: int((s == FB_FOLDS).sum())),
                    n_cell_folds=("fold_flag", lambda s: int((s == "cell_folds").sum()))).reset_index()
        fbs["frac_fb_label_free"] = (fbs.n_fb_n + fbs.n_fb_folds) / fbs.n_draws
        fbs.to_csv(out / f"{tag}_count_fallback.csv", index=False)
    tab = pd.DataFrame([dict(arm=k.split("|")[0], learner=k.split("|")[2], method=k.split("|")[3], n_fit=v) for k, v in det.items()])
    if len(tab):
        tab = tab.groupby(["arm", "learner", "method"], as_index=False).n_fit.sum()
        tab.to_csv(out / f"{tag}_count_detail.csv", index=False)
    print(f"[count-only] 작업 단위 {len(units)} · 건너뜀 {len(skipped)} · 세기 {time.time() - t0:.0f}s", flush=True)
    for arm, q in df.groupby("arm"):
        h = float(q.est_s.sum()) / 3600
        print(f"  [{arm}] 단위 {len(q)} · 대상 {q.target.nunique()} · 적합 {int(q.fit_total.sum()):,} · 추정 누적 {h:.2f} CPU-h(4스레드 프로세스 기준, h40 BASE_FIT) · "
              f"실측 비 {W.RESCALE_RATIO} 적용 {h * W.RESCALE_RATIO:.2f} h · 워커 22개 약 {h * W.RESCALE_RATIO / 22:.2f} h", flush=True)
    if len(tab):
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(tab.to_string(index=False), flush=True)
    if len(fbd):
        s = fbs.groupby(["arm", "n"], as_index=False)[["n_draws", "n_fb_n", "n_fb_folds", "n_cell_folds"]].sum()
        print("[count-only] 라벨 없이 센 적층 대체(n < 10, 묶음 수 K < 2)와 셀 묶음 적층(대체 아님)의 (arm, n) 합계(추출 수)", flush=True)
        for r in s.to_dict("records"):
            print(f"  {r['arm']} n {W.nlab(r['n'])}: 추출 {r['n_draws']} · n<10 {r['n_fb_n']} · 묶음<2 {r['n_fb_folds']} · 셀 묶음 {r['n_cell_folds']}",
                  flush=True)
    dk = df[df.dropped != ""]
    if len(dk):
        print(f"[count-only] 후보를 뺀 단위 {len(dk)}개(사유는 {tag}_count.csv 의 dropped 열)", flush=True)
    if skipped:
        print(f"[count-only] 건너뛴 분할·대상: {dict(Counter(s_['status'] for s_ in skipped))}", flush=True)
    return df


# ================================================================ 9. 집계(봉인 폴더에만 쓴다)
def variant_shards(a, arm):
    """이 변형(a.VARIANT)의 조각만(핵심 판과 cci5y·ext 조각이 한 폴더에 있어도 섞지 않는다. 조각 이름의 여섯째 마디가 변형)."""
    return [s_ for s_ in XB.find_shards(a.SHARDS, a.TAG[arm]) if str(s_.get("variant", "")) == str(a.VARIANT)]


def load_tms_variant(a, arm):
    """xbatch_core.load_tms 와 같은 경로(load_stores, h54.make_tm, h54.units_for)를 이 변형의 조각만으로 밟는다. 반환 (tms, units, runs)."""
    from polar.h4_common import load_stores
    sh = variant_shards(a, arm)
    if not sh:
        return {}, [], pd.DataFrame()
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    XB.check_cfg(units, a.allow_mixed_cfg)
    runs = XB.read_runs(sh)
    stores = load_stores([s_["npz"] for s_ in sh])
    by = {}
    for (nm, sp), st in stores.items():
        by.setdefault(nm, {})[int(sp)] = st
    tms = {nm: W.make_tm(nm, bs, W.units_for(units, nm), int(a.nboot), _point_only(nm)) for nm, bs in sorted(by.items())}
    return tms, units, runs


def _point_only(nm):
    """저장소 이름의 점 추정 여부. 변형 꼬리(~cci5y, ~ext)를 떼고 xbatch_core.is_point_only_name 에 넘긴다."""
    al, mode = nm.split("|")
    for v in VARIANTS[1:]:
        if al.endswith("~" + v):
            al = al[: -len(v) - 1]
    return XB.is_point_only_name(f"{al}|{mode}")


def _store_of(row):
    return f"{row['target']}|{row['mode']}"


def fallback_index(runs):
    """runs 의 Stack 행에서 (저장소, n) → 대체 비율과 대체 추출 목록 {(분할, n, 추출)}."""
    ratio, drops = {}, {}
    if runs is None or not len(runs) or "stack_fallback" not in runs:
        return ratio, drops
    q = runs[runs.method == "Stack"].copy()
    q["stack_fallback"] = q.stack_fallback.fillna("").astype(str)
    q["store"] = [_store_of(r) for r in q.to_dict("records")]
    for (nm, n), g in q.groupby(["store", "n"]):
        fbm = g.stack_fallback != ""
        ratio[(nm, int(n))] = dict(ratio=float(fbm.mean()), n_draws=int(len(g)), n_fb=int(fbm.sum()),
                                   reasons=dict(Counter(g.stack_fallback[fbm])))
        drops[(nm, int(n))] = {(int(sp), int(n), int(dr)) for sp, dr in zip(g.split[fbm], g.draw[fbm])}
    return ratio, drops


def tm_without(tm, drop):
    """(분할, n, 추출) 집합의 키를 모두 뺀 TMx(대체 없는 추출만으로 다시 낸 판정용)."""
    by = {}
    for sp, st in tm.by_all.items():
        keys = [k for k in st.keys if (int(sp), int(k[4]), int(k[5])) not in drop]
        S, C = st.matrices(keys)
        by[sp] = BlockStore._from_arrays(st.target, st.split, st.blocks, st.ncell, keys, S, C, dict(st.meta))
    info = {sp: dict(dup_of=-1, valid=sp in tm.by_valid) for sp in by}
    t2 = H.TMx(tm.name, by, info, tm.nboot)
    t2.point_only = tm.point_only
    t2.has_ci = len(t2.by_valid) > 0 and not t2.point_only
    t2.nboot = tm.nboot if t2.has_ci else 0
    t2.expected_splits = getattr(tm, "expected_splits", sorted(t2.used))
    return t2


def contrast_rows(hyp, tms, names, gA, gB, n, registered, fbr, fbd, uses_stack=True, label=""):
    """한 (가설, n) 대비의 지역 행과 층화 평균 행(같은 라벨 집합 대비, xbatch_core.contrast_pool). 퇴화 추출 규칙: (대상, n) 의 대체 비율이 50 % 를
    넘는 지역은 '판정 불가(적층 미작동)'로 두고 풀에서 뺀다. 동등은 대체 없는 추출만으로 다시 낸 판정이 같을 때만 남기고, 다르면 미결정으로 쓴다."""
    flagged = [nm for nm in names if nm in tms and uses_stack and fbr.get((nm, int(n)), {}).get("ratio", 0.0) > FALLBACK_MAX]
    use = {nm: tms[nm] for nm in names if nm in tms and nm not in flagged}
    rows, per = XB.contrast_pool(use, names, gA, gB, label=label or hyp, kind="same", registered=registered)
    for r in rows:
        nm = r["target"] if r.get("scope") == "region" else None
        r.update(hyp=hyp, n=int(n), fallback_ratio=fbr.get((nm, int(n)), {}).get("ratio", np.nan) if nm else np.nan, verdict_final=r.get("verdict4"),
                 verdict_note="")
    if uses_stack:
        for r in rows:
            if r.get("verdict4") != "동등":
                continue
            if r.get("scope") == "region":
                nm = r["target"]
                drop = fbd.get((nm, int(n)), set())
                if not drop:
                    continue
                rr, _ = XB.contrast_pool({nm: tm_without(tms[nm], drop)}, [nm], gA, gB, label="nf", kind="same", registered=1)
                v2 = rr[0].get("verdict4") if rr else "행 없음"
            else:
                tm2 = {nm: tm_without(t, fbd.get((nm, int(n)), set())) for nm, t in use.items()}
                rr, _ = XB.contrast_pool(tm2, names, gA, gB, label="nf", kind="same", registered=registered)
                v2 = rr[-1].get("verdict4") if rr else "행 없음"
            r["verdict4_nofallback"] = v2
            if v2 != "동등":
                r.update(verdict_final="미결정", verdict_note=f"동등은 대체 없는 추출 판정({v2})과 달라 쓰지 않는다")
    allr = [fbr.get((nm, int(n)), {}) for nm in names if nm in tms]
    tot = sum(q.get("n_draws", 0) for q in allr)
    pooled_fb = float(sum(q.get("n_fb", 0) for q in allr) / tot) if (uses_stack and tot) else np.nan
    if uses_stack and rows:
        rows[-1]["fallback_ratio"] = pooled_fb
    for nm in flagged:
        q = fbr[(nm, int(n))]
        rows.insert(max(len(rows) - 1, 0), dict(hyp=hyp, n=int(n), target=nm, scope="region", label=label or hyp, verdict4="판정 불가",
                                                verdict_final="판정 불가", verdict_note=STACK_NA_NOTE, fallback_ratio=q["ratio"],
                                                fallback_reasons=json.dumps(q["reasons"], ensure_ascii=False)))
    if flagged:
        mean = [r for r in rows if r.get("scope") == "MEAN"]
        if not mean:                                                       # 모든 지역이 퇴화: 풀 행이 없으면 판정 불가(적층 미작동) 풀 행을 둔다
            rows.append(dict(hyp=hyp, n=int(n), target="MEAN[]", scope="MEAN", label=label or hyp, verdict4="판정 불가", verdict_final="판정 불가",
                             verdict_note=STACK_NA_NOTE, fallback_ratio=pooled_fb, undetermined=True, n_ci_regions=0,
                             n_regions_target=len(names), n_regions_registered=int(registered), pool=XB.pool_label(0, int(registered)),
                             pool_regions="", ci_flag=f"{STACK_NA_NOTE}: 대체 비율 > 50 % 지역 {len(flagged)}/{len(allr)}",
                             flagged_regions=",".join(flagged), p_two=np.nan))
        else:
            mr = mean[-1]
            mr["flagged_regions"] = ",".join(flagged)
            n_ci = int(mr.get("n_ci_regions", 0) or 0)
            if mr.get("verdict4") in XB.NA_VERDICTS and n_ci < XB.MIN_POOL_REGIONS <= n_ci + len(flagged):
                mr["verdict_note"] = f"{STACK_NA_NOTE}(퇴화 지역을 빼자 풀 지역이 {XB.MIN_POOL_REGIONS}개 미만)"   # 풀 미달의 원인이 적층 미작동
    return rows, per


def _g(method, n, learner="none", lam=None):
    return W.gk(method, n, learner, lam)


def _true(v):
    """표 칸의 참 여부(NaN·None 은 거짓)."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def _pool_text(hyp, n, pool_label):
    """해석 문장의 풀 괄호: XB-1·XB-2 는 '주 4지역, 지역 m/4', n 40·160 은 '레나·캐나다 2지역', XB-3 은 '지역 k/3'."""
    k = str(pool_label or "")
    if hyp in ("XB-1", "XB-2"):
        return "레나·캐나다 2지역" + (f", {k}" if k else "") if int(n) in (40, 160) else "주 4지역" + (f", {k}" if k else "")
    return k


def _fill(tpl, **kw):
    out = str(tpl)
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def summarize(a):
    """집계(계획 2.2 가설 XB-1–XB-6, Holm 가족 m 12·m 4, 퇴화 추출 규칙). 표는 봉인 폴더에만 쓰고 화면에는 행 수와 해시만 남긴다."""
    t0 = time.time()
    tms_t, units_t, runs_t = load_tms_variant(a, "xb_t")
    tms_r, units_r, runs_r = load_tms_variant(a, "xb_r")
    fbr_t, fbd_t = fallback_index(runs_t)
    fbr_r, fbd_r = fallback_index(runs_r)
    sfx = "" if not a.VARIANT else f"~{a.VARIANT}"
    nmx = lambda names: [f"{nm.split('|')[0]}{sfx}|{nm.split('|')[1]}" for nm in names]   # noqa: E731
    rows = []
    main4 = nmx(XB.MAIN4)
    for hyp, (mA, lA, lamA), (mB, lB, lamB) in (("XB-1", ("Stack", "none", None), ("P1", "none", None)),
                                                 ("XB-2", ("StackR", LO, LAM_BASE), ("R1", LO, LAM_BASE))):
        for n in HYP_N[hyp]:
            reg = 2 if n in (40, 160) else 4
            rr, _ = contrast_rows(hyp, tms_t, main4, _g(mA, n, lA, lamA), _g(mB, n, lB, lamB), n, reg, fbr_t, fbd_t, label=f"{hyp}|MAIN4")
            for r in rr:
                r.update(role="주", pool_name="MAIN4", registered=reg, two_region=n in (40, 160))
            rows += rr
    inr = nmx(XB.INREGION3)
    for n in HYP_N["XB-3"]:
        rr, _ = contrast_rows("XB-3", tms_r, inr, _g("StackR", n, LO, LAM_CV), _g("R1", n, LO, LAM_CV), n, 3, fbr_r, fbd_r, label="XB-3|INREGION3")
        for r in rr:
            r.update(role="주", pool_name="INREGION3", registered=3)
        rows += rr
    ak = nmx(["Alaska|r"])
    for n in HYP_N["XB-4"]:
        rr, _ = contrast_rows("XB-4", tms_r, ak, _g("Stack", n), _g("P1", n), n, 1, fbr_r, fbd_r, label="XB-4|Alaska")
        for r in rr:
            r.update(role="주(비열등)", pool_name="Alaska|r", registered=1)
        rows += [r for r in rr if r.get("scope") == "region"]
    # XB-5(보조): 등록 대비 + 주 대비의 다른 풀, 대상별 행
    aux_t = [("Stack0−P0", ("Stack0", "none", None), ("P0", "none", None), (0,), False), ("Stack0−B:ens", ("Stack0", "none", None), ("B:ens", "none", None), (0,), False),
             ("Stack−Pe", ("Stack", "none", None), ("Pe", "none", None), tuple(a.G_T), True), ("StackR−Re", ("StackR", LO, LAM_BASE), ("Re", LO, LAM_BASE), tuple(a.G_T), True),
             ("Stack−P1", ("Stack", "none", None), ("P1", "none", None), tuple(a.G_T), True), ("StackR−R1", ("StackR", LO, LAM_BASE), ("R1", LO, LAM_BASE), tuple(a.G_T), True),
             ("P*−P1", ("P*", "none", None), ("P1", "none", None), tuple(a.G_T), False)]
    pools_t = dict(MAIN4=main4, PE1=nmx(XB.PE1), PE2=nmx(XB.PE2), AUX3=nmx(XB.AUX3))
    for lab, (mA, lA, lamA), (mB, lB, lamB), ns, us in aux_t:
        for n in ns:
            gA = H.P0_GRP if mA == "P0" else _g(mA, n, lA, lamA)
            gB = H.P0_GRP if mB == "P0" else _g(mB, n, lB, lamB)
            for pn, names in pools_t.items():
                rr, _ = contrast_rows("XB-5", tms_t, names, gA, gB, n, len(names), fbr_t, fbd_t, uses_stack=us, label=f"XB-5|{lab}|{pn}")
                for r in rr:
                    r.update(role="보조", pool_name=pn, contrast=lab, registered=len(names))
                rows += [r for r in rr if r.get("scope") != "region" or pn == "PE2"]
            allnm = sorted(tms_t)
            rr, _ = contrast_rows("XB-5", tms_t, allnm, gA, gB, n, len(allnm), fbr_t, fbd_t, uses_stack=us, label=f"XB-5|{lab}|targets")
            for r in rr:
                r.update(role="보조(대상별)", pool_name="targets", contrast=lab, subregion=r.get("target", "").split("|")[0].split("~")[0] in
                         (list(H.SUB_AL) + list(H.SUB_OTHER)))
            rows += [r for r in rr if r.get("scope") == "region"]
    aux_r = [("Stack−Pe", ("Stack", "none", None), ("Pe", "none", None)), ("StackR−Re", ("StackR", LO, LAM_CV), ("Re", LO, LAM_CV)),
             ("Stack−P1", ("Stack", "none", None), ("P1", "none", None)), ("StackR(0.25)−R1(0.25)", ("StackR", LO, LAM_BASE), ("R1", LO, LAM_BASE))]
    for lab, (mA, lA, lamA), (mB, lB, lamB) in aux_r:
        for n in a.G_R:
            rr, _ = contrast_rows("XB-5", tms_r, inr, _g(mA, n, lA, lamA), _g(mB, n, lB, lamB), n, 3, fbr_r, fbd_r, label=f"XB-5|r|{lab}")
            for r in rr:
                r.update(role="보조", pool_name="INREGION3", contrast=lab, registered=3)
            rows += rr
    kmap = {}
    for u in units_t + units_r:
        for nm in (u.get("store_names") or []):
            kmap[nm] = max(int(kmap.get(nm, 0)), int((u.get("xb_cands") or {}).get("K", 0)))
    for r in rows:
        if r.get("scope") != "region":
            r["K_by_target"] = ",".join(f"{nm.split('|')[0]}:{kmap.get(nm, 0)}" for nm in str(r.get("pool_regions", "")).split(",") if nm)
        else:
            r["K"] = kmap.get(r.get("target"), np.nan)
    tests = XB.clean_rows(rows)
    if len(tests):
        tests = _verdict_columns(tests)
    weights, wsum, cci_rel = _weights_tables(units_t, units_r, runs_t, runs_r, tms_t, tms_r)
    fb_tab = pd.DataFrame([dict(arm=arm, store=k[0], n=k[1], **{kk: (json.dumps(vv, ensure_ascii=False) if isinstance(vv, dict) else vv)
                                                                 for kk, vv in v.items()})
                           for arm, fbr in (("xb_t", fbr_t), ("xb_r", fbr_r)) for k, v in fbr.items()])
    rmse_rows = []
    for arm, tms in (("xb_t", tms_t), ("xb_r", tms_r)):
        for nm, tm in tms.items():
            gs = tm.groups()
            for g in sorted(gs, key=str):
                rc, rb = XB.grp_rmse2(tm, g)
                rmse_rows.append(dict(arm=arm, store=nm, method=g[0], learner=g[1], n=g[4], lam=g[5], rmse=rc, rmse_beq=rb, n_splits=len(gs[g])))
    meta = dict(plan=XB.PLAN_DOC, plan_commit=XB.PLAN_COMMIT, section=PLAN_SEC, design=DESIGN, blind=BLIND, variant=a.VARIANT, nboot=int(a.nboot),
                nboot_capped=bool(a.nboot_capped), n_units_t=len(units_t), n_units_r=len(units_r), n_stores_t=len(tms_t), n_stores_r=len(tms_r),
                holm=dict(main=HOLM_M_MAIN, ni=HOLM_M_NI), notes=[WEIGHT_NOTE, _fill(N10_NOTE, r="xb_tests 의 fallback_ratio 열"), GAMMA_CV_NOTE, ANCHOR_NOTE], elapsed_s=round(time.time() - t0, 1),
                smoke_env=getattr(a, "SMOKE_ENV", None), local_resources=getattr(a, "LOCAL_RES", None), shards_variant=a.VARIANT,
                max_rss_mb=XB.max_rss_mb(),
                tdd_v4_missing_allowed=dict(Counter(bool(u.get("tdd_v4_missing_allowed", False)) for u in units_t + units_r)),
                cfg=dict(cfg_common=dict(Counter(str(u.get("cfg_common", "legacy")) for u in units_t + units_r)),
                         code_sha=dict(Counter(str(u.get("code_sha", "legacy")) for u in units_t + units_r))),
                written=time.strftime("%Y-%m-%d %H:%M:%S"))
    pre = "smoke_" if a.smoke else ""
    sv = f"_{a.VARIANT}" if a.VARIANT else ""
    tables = {f"{pre}xb_tests{sv}.csv": tests, f"{pre}xb_weights{sv}.csv": weights, f"{pre}xb_weights_summary{sv}.csv": wsum,
              f"{pre}xb_cci_relation{sv}.csv": cci_rel, f"{pre}xb_fallback{sv}.csv": fb_tab, f"{pre}xb_group_error{sv}.csv": pd.DataFrame(rmse_rows),
              f"{pre}xb_meta{sv}.json": meta}
    XB.write_sealed(a.SEALED_NAME, tables, root=a.SEALED_ROOT)                # <out-dir>/sealed/(기본 data/processed/xbatch/XB_multisource_stacking/sealed)
    return tests


def _verdict_columns(t):
    """Holm(XB-1–XB-3 의 n 별 우세 p, m 12; XB-4 의 비열등 p × 2, m 4. 보조 열), 가설 종합(_rule3, XB-4 규칙), 해석 문장."""
    t = t.copy()
    t["holm_family"] = ""; t["p_holm"] = np.nan; t["holm_multiplier"] = np.nan
    main = []
    for hyp in ("XB-1", "XB-2", "XB-3"):
        for n in HYP_N[hyp]:
            q = t[(t.hyp == hyp) & (t.n == n) & (t.scope == "MEAN")]
            main.append(((hyp, n), float(q.p_two.iloc[0]) if len(q) and q.verdict_final.iloc[0] not in XB.NA_VERDICTS and "p_two" in q else np.nan))
    ht = XB.holm_table([f"{h}|{n}" for (h, n), _ in main], [p for _, p in main], HOLM_M_MAIN)
    for ((hyp, n), _), r in zip(main, ht.to_dict("records")):
        m = (t.hyp == hyp) & (t.n == n) & (t.scope == "MEAN")
        t.loc[m, "p_holm"] = r["p_holm"]; t.loc[m, "holm_multiplier"] = r["multiplier"]; t.loc[m, "holm_family"] = "XB-1–XB-3(m 12)"
    ni = []
    for n in HYP_N["XB-4"]:
        q = t[(t.hyp == "XB-4") & (t.n == n) & (t.scope == "region")]
        ni.append(float(q.p_ni_x2.iloc[0]) if len(q) and "p_ni_x2" in q and q.verdict_final.iloc[0] not in XB.NA_VERDICTS else np.nan)
    ht4 = XB.holm_table([f"XB-4|{n}" for n in HYP_N["XB-4"]], ni, HOLM_M_NI)
    for n, r in zip(HYP_N["XB-4"], ht4.to_dict("records")):
        m = (t.hyp == "XB-4") & (t.n == n) & (t.scope == "region")
        t.loc[m, "p_holm"] = r["p_holm"]; t.loc[m, "holm_multiplier"] = r["multiplier"]; t.loc[m, "holm_family"] = "XB-4 비열등(m 4)"
    t["hypothesis_verdict"] = ""; t["sentence"] = ""; t["blind"] = t.hyp.map(BLIND).fillna(""); t["design"] = DESIGN
    t["region_summary"] = ""
    for hyp in ("XB-1", "XB-2", "XB-3"):
        items = []
        for n in HYP_N[hyp]:
            sub = t[(t.hyp == hyp) & (t.n == n)]
            q = sub[sub.scope == "MEAN"]
            v = q.verdict_final.iloc[0] if len(q) else None
            items.append((W.nlab(n), v))
            if not len(q):
                continue
            i = q.index[0]
            r = t.loc[i]
            regs = sub[sub.scope == "region"].to_dict("records")
            rs = "; ".join(f"{g['target']} {g.get('verdict_final', '')}" + (f" Δ {float(g['delta']):+.2f}" if np.isfinite(g.get("delta", np.nan)) else "")
                           for g in regs)
            t.loc[i, "region_summary"] = rs
            fr = r.get("fallback_ratio", np.nan)
            tpl = {k: _fill(s_, k=W.nlab(n), a=f"{abs(float(r.delta)):.2f}" if np.isfinite(r.delta) else "",
                            r=f"{float(fr):.2f}" if np.isfinite(fr) else "", pool=_pool_text(hyp, n, r.get("pool", "")))
                   for k, s_ in TEMPLATES[hyp].items()}
            wr = [(g["target"], g["delta"], g["ci_lo"], g["ci_hi"]) for g in regs if _true(g.get("worse"))]
            b = XB.branch_of(v)
            if b == "판정 불가" and not _str(r.get("verdict_note")).startswith(STACK_NA_NOTE):
                # 적층 미작동이 아닌 판정 불가(풀 지역 2 미만, CI 비유한 등): '적층이 작동하지 않아' 문장을 쓰지 않고 사유를 붙인다
                sent = XB.compose_sentence(dict(tpl, **{"판정 불가": tpl["판정 불가_일반"]}), b, r.p_holm, r.delta, wr,
                                           reason=f"라벨 {W.nlab(n)}개, 사유 {_na_reason(r)}")
            else:
                sent = XB.compose_sentence(tpl, b, r.p_holm, r.delta, wr, reason="")
            if hyp in ("XB-1", "XB-2") and b == "우세":
                sent += f" 지역 행: {rs}."
            if hyp == "XB-1" and int(n) == 10:
                sent += " " + _fill(N10_NOTE, r=f"{float(fr):.2f}" if np.isfinite(fr) else "미상") + "."
            t.loc[i, "sentence"] = sent
        t.loc[t.hyp == hyp, "hypothesis_verdict"] = XB.rule3(items, "우세")
    # XB-4(비열등): 등록 n 모두 비열등이면 지지, 일부면 부분 지지, 없으면 기각. 1절 '행 없음·실패 … 판정 불가 갈래'와 _rule3 의 '판정할 수 없는
    # 대비가 있고 충족이 없음 = 판정 불가'에 따라, 비열등이 하나도 없고 판정할 수 없는 n(행 없음, 판정 불가)이 있으면 판정 불가로 둔다.
    q4 = t[(t.hyp == "XB-4") & (t.scope == "region")]
    ok4 = {int(i): _true(t.loc[i].get("noninf")) and t.loc[i, "verdict_final"] not in XB.NA_VERDICTS for i in q4.index}
    hit = sorted({int(t.loc[i, "n"]) for i in q4.index if ok4[int(i)]}, key=lambda v: HYP_N["XB-4"].index(v) if v in HYP_N["XB-4"] else 99)
    na, why = [], []
    for n in HYP_N["XB-4"]:
        qn = q4[q4.n == n]
        if not len(qn):
            na.append(n); why.append("행 없음")
        elif all(t.loc[i, "verdict_final"] in XB.NA_VERDICTS for i in qn.index):
            na.append(n)
            why += [STACK_NA_NOTE if _str(t.loc[i].get("verdict_note")).startswith(STACK_NA_NOTE) else _na_reason(t.loc[i]) for i in qn.index]
    reg = len(HYP_N["XB-4"])
    if len(hit) == reg:
        v4 = "지지"
    elif hit:
        v4 = "부분 지지"
    elif na:
        v4 = f"판정 불가({', '.join(dict.fromkeys(why))})"
    else:
        v4 = "기각"
    t.loc[t.hyp == "XB-4", "hypothesis_verdict"] = (f"{v4}: 비열등 n = {', '.join(W.nlab(v) for v in hit) or '없음'}"
                                                    + (f"; 판정 불가 n = {', '.join(W.nlab(v) for v in na)}" if na else ""))
    for i in q4.index:
        k = W.nlab(int(t.loc[i, "n"]))
        if ok4[int(i)]:
            sent = _fill(XB4_SENT["비열등"], k=k)
        elif t.loc[i, "verdict_final"] in XB.NA_VERDICTS:
            if _str(t.loc[i].get("verdict_note")).startswith(STACK_NA_NOTE):
                fr = t.loc[i].get("fallback_ratio", np.nan)
                sent = _fill(XB4_SENT["판정 불가"], k=k, r=f"{float(fr):.2f}" if _finite(fr) else "미상")
            else:
                sent = _fill(XB4_SENT["판정 불가_일반"], k=k, why=_na_reason(t.loc[i]))
        else:
            sent = _fill(XB4_SENT["아님"], k=k)
        t.loc[i, "sentence"] = sent + "."
    return t


def _str(v):
    """표 칸 → 문자열(NaN·None 은 빈 문자열)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return str(v)


def _finite(v):
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


NA_REASON_TXT = {"pool<2": "풀 지역 2 미만", "ci nonfinite": "CI 비유한", "assert: point outside CI": "점 추정이 CI 밖"}


def _na_reason(r):
    """판정 불가 행의 사유(ci_flag 의 첫 마디, 없으면 판정 문자열)."""
    cf = _str(r.get("ci_flag"))
    w = cf.split(";")[0].strip() if cf else ""
    w = NA_REASON_TXT.get(w, w)
    return w or _str(r.get("verdict_final")) or "행 없음"


def _weights_tables(units_t, units_r, runs_t, runs_r, tms_t, tms_r):
    """XB-6(서술): 추출별 가중·γ̂·대체 사유(unit.json 의 stack), 대상·n 별 요약, 빠진 후보, 대상별 K, CCI 가중과 P1 오차의 관계."""
    rows = []
    for arm, units in (("xb_t", units_t), ("xb_r", units_r)):
        for u in units:
            nm = (u.get("store_names") or [f"{u.get('target')}|{u.get('mode')}"])[0]
            for e in u.get("stack", []) or []:
                r = dict(arm=arm, store=nm, split=int(u.get("split", -1)), n=int(e["n"]), draw=str(e["draw"]), n_lab=e["n_lab"], fallback=e["fallback"],
                         gamma=e["gamma"], K=e["K"], folds=e["folds"], dropped=json.dumps((u.get("xb_cands") or {}).get("dropped", {}), ensure_ascii=False))
                for nm_c, w in zip(e["names"], e["w_star"]):
                    r[f"w_{nm_c}"] = w
                for nm_c, w in zip(e["names"], e["w_hat"]):
                    r[f"what_{nm_c}"] = w
                rows.append(r)
    wt = pd.DataFrame(rows)
    if not len(wt):
        return wt, pd.DataFrame(), pd.DataFrame()
    wcols = [c_ for c_ in wt.columns if c_.startswith("w_")]
    wsum = wt.groupby(["arm", "store", "n"]).agg(n_draws=("draw", "size"), frac_fallback=("fallback", lambda s: float((s != "").mean())),
                                                 gamma_mean=("gamma", "mean"), K=("K", "max"), dropped=("dropped", "first"),
                                                 **{c_: (c_, "mean") for c_ in wcols}).reset_index()
    rel = []
    for arm, tms in (("xb_t", tms_t), ("xb_r", tms_r)):
        q = wsum[(wsum.arm == arm) & (wsum.n == -1)]
        for r in q.to_dict("records"):
            tm = tms.get(r["store"])
            if tm is None:
                continue
            p1c, _ = XB.grp_rmse2(tm, _g("P1", -1))
            p0c, _ = XB.grp_rmse2(tm, H.P0_GRP)
            rel.append(dict(arm=arm, store=r["store"], w_cci=float(np.nansum([r.get("w_Cr", 0.0), r.get("w_Ca", 0.0)])), rmse_p1_all=p1c, rmse_p0=p0c))
    rel = pd.DataFrame(rel)
    if len(rel) >= 3:
        from scipy.stats import spearmanr
        for arm, g in rel.groupby("arm"):
            if len(g) >= 3:
                rho, p = spearmanr(g.w_cci, g.rmse_p1_all)
                rel.loc[rel.arm == arm, "spearman_wcci_rmse_p1"] = rho
                rel.loc[rel.arm == arm, "spearman_p"] = p
    return wt, wsum, rel


# ================================================================ 10. 재현 관문(계획 2.2 '재현 관문')
GATE_KEYS = {"xb_r": ("wf6", lambda k: k[0] == "P1" or (k[0] == "R1" and k[1] == LO and float(k[7]) in (LAM_BASE, LAM_CV))),
             "xb_t": ("wf7", lambda k: k[0] in ("P0", "P1") or (k[0] in ("R1", "Re") and k[1] == LO and float(k[7]) == LAM_BASE))}


def gate(a):
    """xb_r 의 P1·R1(교차검증 λ, λ 0.25) 키와 WF6 조각, xb_t 의 P0·P1·R1(0.25)·Re 키와 WF7 조각의 블록 SSE 대조(1절 허용 오차). 표에는 |ΔSSE| 만 있다."""
    from polar.h4_common import load_stores
    out = []
    for arm in a.EXPS:
        ref_tag, kf = GATE_KEYS[arm]
        new = load_stores([s_["npz"] for s_ in variant_shards(a, arm)])
        refp = sorted(Path(_abs(a.gate_dir)).glob(f"{ref_tag}__cpu__*_blocksse.npz"))
        ref = load_stores(refp) if refp else {}
        g = XB.gate_compare(new, ref, a.gate_level, key_fn=kf)
        g.insert(0, "arm", arm)
        out.append(g)
        s = XB.gate_summary(g)
        print(f"[gate] {arm} ↔ {ref_tag}: 공통 키 {s['n_keys']} · 실패 {s['n_fail']} · 통과 {'예' if s['passed'] else '아니오'}", flush=True)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    XB.check_out_dir(a.OUT)
    a.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.OUT / f"xb_gate{a.SUFFIX}.csv", index=False)
    return df


# ================================================================ 11. 1c 입력: v4 새 셀 tdd_matched(a2 정의, 라벨 값 미사용)
def a2_tdd_cells(lat, lon, ymin, ymax, nc=ERA5_NC):
    """scripts/2_evaluation/a2_year_matched_tdd.py 의 2) 단계와 같은 정의: 셀별 최근접 육지 격자(±2 → ±5셀 유클리드 폴백), 연별 TDD
    (월평균 0 °C 초과 달의 (T − 0)·일수 합), matched = [year_min, year_max] ∩ [2010, 2024] 연별 TDD 평균(2010 이전 관측은 2010–2014, 표지
    pre2010, 일부만 2010 이전이면 partial). 반환 DataFrame(tdd_matched, match_flag, n_years_matched, match_lo, match_hi, fallback_deg)."""
    import calendar
    import xarray as xr
    NC_LO, NC_HI, PRE_LO, PRE_HI = 2010, 2024, 2010, 2014
    ds = xr.open_dataset(nc)
    tn = "valid_time" if "valid_time" in ds.coords else "time"
    glat, glon = ds.latitude.values, ds.longitude.values
    years, months = ds[tn].dt.year.values, ds[tn].dt.month.values
    YEARS = np.arange(NC_LO, NC_HI + 1)
    assert set(YEARS) == set(np.unique(years)) and len(years) == 12 * len(YEARS)
    t2m = ds["t2m"].values.astype(np.float32) - 273.15
    land = np.isfinite(t2m).all(axis=0)

    def fallback(iy, ix, la, lo, nb):
        best = None
        for dy in range(-nb, nb + 1):
            for dx in range(-nb, nb + 1):
                y2, x2 = iy + dy, (ix + dx) % len(glon)
                if 0 <= y2 < len(glat) and land[y2, x2]:
                    d2 = (glat[y2] - la) ** 2 + (glon[x2] - lo) ** 2
                    if best is None or d2 < best[0]:
                        best = (d2, y2, x2)
        return best
    n = len(lat)
    IY, IX, FB = np.empty(n, int), np.empty(n, int), np.zeros(n)
    for i, (la, lo) in enumerate(zip(lat, lon)):
        iy, ix = int(np.abs(glat - la).argmin()), int(np.abs(glon - lo).argmin())
        if not land[iy, ix]:
            b = fallback(iy, ix, la, lo, 2)
            if b is not None:
                FB[i] = 0.2
            else:
                b = fallback(iy, ix, la, lo, 5)
                FB[i] = 0.5 if b is not None else np.nan
            if b is not None:
                iy, ix = b[1], b[2]
        IY[i], IX[i] = iy, ix
    TM = t2m[:, IY, IX]
    del t2m
    days_y = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(years, months)])
    TDD_Y = np.stack([(np.clip(TM[years == y], 0, None) * days_y[years == y][:, None]).sum(0) for y in YEARS])
    ymin, ymax = np.asarray(ymin).astype(int), np.asarray(ymax).astype(int)
    pre = ymax < NC_LO
    partial = (ymin < NC_LO) & ~pre
    lo_y = np.where(pre, PRE_LO, np.clip(ymin, NC_LO, NC_HI))
    hi_y = np.where(pre, PRE_HI, np.clip(ymax, NC_LO, NC_HI))
    MY = (YEARS[:, None] >= lo_y[None, :]) & (YEARS[:, None] <= hi_y[None, :])
    n_match = MY.sum(0)
    assert (n_match >= 1).all()
    tdd = (np.where(MY, TDD_Y, 0.0)).sum(0) / n_match
    return pd.DataFrame(dict(tdd_matched=tdd, match_flag=np.where(pre, "pre2010", np.where(partial, "partial", "full")), n_years_matched=n_match,
                             match_lo=lo_y, match_hi=hi_y, fallback_deg=FB))


def write_tdd_v4(a, n_check=400):
    """v4 새 셀(네 실행 표의 F4_direct 셀 가운데 lgx_tdd_matched_v1 에 없는 셀)의 tdd_matched 를 a2 정의로 만들고, v3 셀의 표지(a2 셀 표)와 함께
    쓴다. 재현 점검: 같은 코드로 v3 셀 n_check 개를 다시 계산해 lgx_tdd_matched_v1 과 대조한다(최대 |차| ≤ 1e-6 °C·day). 라벨 값은 읽지 않는다."""
    XB.require_memory(30)
    XB.limit_memory(10)
    v1 = pd.read_csv(a.TDDM, usecols=["loc_id", "tdd_matched"]).drop_duplicates("loc_id")
    parts = []
    for spec in XB.RUN_TABLE_DIRS:
        t = pd.read_csv(a.ha.LGD / spec / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon", "source_id", "region"], low_memory=False)
        t = t[(t.source_id == "F4_direct") & ~t.loc_id.isin(v1.loc_id)]
        parts.append(t.assign(run_table=spec))
    new = pd.concat(parts, ignore_index=True)
    dup = new.groupby("loc_id").agg(nlat=("lat", "nunique"), nlon=("lon", "nunique"))
    assert (dup.nlat == 1).all() and (dup.nlon == 1).all(), "같은 loc_id 의 좌표가 실행 표마다 다르다"
    tabs = new.groupby("loc_id").run_table.apply(lambda s: ";".join(sorted(set(s))))
    new = new.drop_duplicates("loc_id").set_index("loc_id")
    new["run_tables"] = tabs
    yr = pd.read_csv(XB.LABELS_V4, usecols=["loc_id", "year_min", "year_max"], low_memory=False).drop_duplicates("loc_id").set_index("loc_id")
    new = new.join(yr, how="left")
    if new.year_min.isna().any() or new.year_max.isna().any():
        raise SystemExit(f"[tdd-v4] 관측 연도가 없는 새 셀 {int(new.year_min.isna().sum())}개")
    a2c = pd.read_csv(A2_CELLS, usecols=["loc_id", "lat", "lon", "year_min", "year_max", "tdd_matched", "match_flag", "n_years_matched", "match_lo",
                                         "match_hi", "fallback_deg"]).drop_duplicates("loc_id").set_index("loc_id")
    rng = np.random.RandomState(seed_of("xb-tdd-check"))
    chk = a2c.iloc[np.sort(rng.choice(len(a2c), min(int(n_check), len(a2c)), replace=False))]
    lat = np.r_[new.lat.values, chk.lat.values]; lon = np.r_[new.lon.values, chk.lon.values]
    res = a2_tdd_cells(lat, lon, np.r_[new.year_min.values, chk.year_min.values], np.r_[new.year_max.values, chk.year_max.values])
    rn, rc = res.iloc[: len(new)].set_index(new.index), res.iloc[len(new):].set_index(chk.index)
    ref = v1.set_index("loc_id").tdd_matched.reindex(chk.index).values
    dmax = float(np.nanmax(np.abs(rc.tdd_matched.values - ref)))
    flag_ok = bool((rc.match_flag.values == chk.match_flag.values).all())
    if not (dmax <= 1e-6 and flag_ok):
        raise SystemExit(f"[tdd-v4] 재현 점검 실패: v3 셀 {len(chk)}개 최대 차 {dmax:.3g}, 표지 일치 {flag_ok}")
    v3 = a2c[a2c.index.isin(v1.loc_id)][["tdd_matched", "match_flag", "n_years_matched", "match_lo", "match_hi", "fallback_deg"]].assign(part="v3")
    d3 = np.abs(v3.tdd_matched.values - v1.set_index("loc_id").tdd_matched.reindex(v3.index).values)
    assert np.nanmax(d3) <= 1e-9, "a2 셀 표와 lgx_tdd_matched_v1 의 값이 다르다"
    nw = rn.assign(part="v4new", run_tables=new.run_tables.values, region=new.region.values)
    out = pd.concat([v3, nw]).reset_index().rename(columns={"index": "loc_id"}).sort_values("loc_id")
    path = a.TDDV4
    XB.check_out_dir(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), definition="a2_year_matched_tdd.py 2) 단계(matched)", nc=str(ERA5_NC.relative_to(ROOT)),
                nc_sha1=W.file_sha(ERA5_NC), years_src=str(XB.LABELS_V4.relative_to(ROOT)), n_v3=int((out.part == "v3").sum()),
                n_v4new=int((out.part == "v4new").sum()), v4new_by_region={str(k): int(v) for k, v in nw.region.value_counts().items()},
                v4new_pre2010_by_region={str(k): int(v) for k, v in nw[nw.match_flag == "pre2010"].region.value_counts().items()},
                check=dict(n=int(len(chk)), max_abs_diff=dmax, flag_equal=flag_ok), sha256=XB.sha256_file(path), max_rss_mb=XB.max_rss_mb(),
                label_values_used=False)
    (path.with_name(path.stem + "_meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(f"[tdd-v4] v3 셀 {meta['n_v3']} · v4 새 셀 {meta['n_v4new']}(pre2010 {sum(meta['v4new_pre2010_by_region'].values())}) · 재현 점검 v3 {len(chk)}셀 통과 · "
          f"sha256 {meta['sha256'][:16]} → {path.relative_to(ROOT)}", flush=True)
    return out


# ================================================================ 12. 명령행
FIT_MODES = ("smoke", "run")


def local_resource_plan(mode, on_rescale):
    """계획 1절 로컬 자원의 적용 방식(순수 함수, 시험 대상). Rescale(WF_RESCALE=1) 밖이면 모든 모드에서 시작 전 가용 메모리 30 GB 를 기다린다.
    상한: 적합이 없는 세기·집계는 주소 공간 10 GB(RLIMIT_AS, prlimit --as 와 같다). 적합이 있는 스모크·로컬 실행은 glibc 할당 영역을 2개로
    묶은 뒤에만 같은 상한을 건다(묶지 않으면 CatBoost 적합이 수백 건 쌓인 뒤 RLIMIT_AS 아래에서 멈춘다. 구현 기록 9절, x_tempderived_aux_labels·
    x_workflow_end_to_end 와 같은 처리). 반환 dict(require, limit = none·as·arena_as)."""
    if on_rescale:
        return dict(require=False, limit="none")
    return dict(require=True, limit="arena_as" if mode in FIT_MODES else "as")


def limit_malloc_arenas(n=MALLOC_ARENAS) -> bool:
    """glibc malloc 할당 영역 수 상한(로컬 적합 실행 전용). 시작 때 MALLOC_ARENA_MAX 가 있으면 그대로 두고, 없으면 mallopt(M_ARENA_MAX)로
    정하고 자식 프로세스를 위해 환경 변수도 정한다. 성공하면 True."""
    if os.environ.get("MALLOC_ARENA_MAX", "").strip().isdigit():
        return True
    os.environ["MALLOC_ARENA_MAX"] = str(int(n))
    try:
        import ctypes
        return bool(ctypes.CDLL("libc.so.6").mallopt(-8, int(n)))           # M_ARENA_MAX = −8
    except (OSError, AttributeError):
        return False


def apply_local_resources(a, mode):
    """local_resource_plan 을 적용하고 기록을 돌려준다(스모크 집계 메타에 적는다)."""
    plan = local_resource_plan(mode, a.ON_RESCALE)
    rec = dict(plan, mode=mode, mem_available_gb=None, as_limit_gb=None, malloc_arenas=None)
    if plan["require"]:
        rec["mem_available_gb"] = round(float(XB.require_memory(MEM_MIN_GB, wait_s=float(a.mem_wait))), 1)
    if plan["limit"] == "as":
        rec["as_limit_gb"] = MEM_LIMIT_GB if XB.limit_memory(MEM_LIMIT_GB) else None
    elif plan["limit"] == "arena_as":
        if limit_malloc_arenas(MALLOC_ARENAS):
            rec["malloc_arenas"] = os.environ.get("MALLOC_ARENA_MAX")
            rec["as_limit_gb"] = MEM_LIMIT_GB if XB.limit_memory(MEM_LIMIT_GB) else None
        else:
            print("[local] glibc 할당 영역을 묶지 못해 주소 공간 상한을 걸지 않는다(CatBoost 멈춤 방지). 최대 RSS 는 unit.json 에 적는다", flush=True)
    return rec


def check_inputs(a):
    """입력 점검(계획 2.2 '자료', 4절 묶음): v4 새 셀 tdd_matched 표가 있고 sha256 앞 16자가 TDD_V4_SHA16 인지. 없으면 --allow-missing-tdd-v4
    일 때만 진행한다. 화면에는 출처와 해시만 쓴다."""
    _, _, src = tdd_tables(*tdd_args(a))
    print(f"[tdd] v1 {src['v1']} · v4 {src['v4']}" + (" · v4 표 없이 진행(허용 표지)" if src.get("v4_missing_allowed") else f" · sha256 {TDD_V4_SHA16} 확인"),
          flush=True)
    return src


def main(argv=None):
    a = parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    for v in XB.THREAD_VARS:
        os.environ[v] = str(a.threads)
    if a.write_tdd_v4:
        XB.guard("count")
        with XB.restricted_output():
            write_tdd_v4(a)
        return dict(n_fail=0)
    if a.gate:
        XB.guard("summarize")
        a.LOCAL_RES = apply_local_resources(a, "summarize")                # 적합 없음: 30 GB 대기와 주소 공간 상한(Rescale 밖)
        with XB.restricted_output():
            gate(a)
        return dict(n_fail=0)
    mode = "count" if a.count_only else ("summarize" if a.summarize_only else ("smoke" if a.smoke else "run"))
    XB.guard(mode, a.ARGV)                                                 # 허용 표지 없는 본 실행은 자료를 읽기 전에 거부한다
    a.LOCAL_RES = apply_local_resources(a, mode)                           # 1절 로컬 자원(Rescale 밖에서만): 30 GB 대기, 메모리 상한
    if not a.PERMIT:
        if int(a.workers) > 1:
            a.workers = 1
    with XB.restricted_output():
        check_inputs(a)                                                    # v4 tdd 표(1c)의 존재와 sha256(없으면 --allow-missing-tdd-v4 일 때만)
        if a.smoke and a.SMOKE_ENV and a.SMOKE_ENV.get("warn"):
            print(f"[local] 스모크 규약 점검: {'; '.join(a.SMOKE_ENV['warn'])}", flush=True)
    if a.summarize_only:
        with XB.restricted_output():
            summarize(a)
        return dict(n_fail=0)
    with XB.restricted_output():
        units, skipped, expected = enumerate_units(a)
        print(f"[data] arm {a.EXPS} · 변형 '{a.VARIANT}' · 작업 단위 {len(units)}(건너뜀 {len(skipped)}) · arm 별 {dict(Counter(u[0] for u in units))} · "
              f"seed {a.seeds} · 스레드 {a.threads} · 워커 {a.workers}", flush=True)
        if a.count_only:
            count_only(a, units, skipped)
            return dict(n_fail=0)
        XB.check_out_dir(a.SHARDS)
        a.SHARDS.mkdir(parents=True, exist_ok=True)
        todo, resumed = [], []
        for u in units:
            ok, why = (XB.unit_state(a.SHARDS, a.TAG[u[0]], u[1], u[2], u[3], unit_cfg(a, u[0], u[4], data_sha_of(a, u[0], u[1])), u[4])
                       if a.resume else (False, ""))
            (resumed if ok else todo).append(u)
        print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)}", flush=True)
        exp_items = [(list(k), v) for k, v in expected.items()]
        done, failed = XB.execute(todo, _worker_run, workers=a.workers, threads=a.threads, init=_worker_init, init_args=(argv, exp_items),
                                  retries=a.pool_retries, permit=a.PERMIT)
        n_fail = len(failed) + sum(1 for r in done if r.get("status") == "failed")
        print(f"[done] 완료 {len(done)} · 실패 {n_fail} · 단위 상태 {dict(Counter(str(r.get('status')) for r in done))} · "
              f"적합 {sum(int(r.get('n_fit') or 0) for r in done):,}(적합 실패·비유한 키 {sum(int(r.get('n_fit_fail') or 0) for r in done)}) · "
              f"{time.time() - t0:.0f}s · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
        if failed:
            XB.check_out_dir(a.OUT)
            pd.DataFrame([dict(unit=_unit_name(u), error=e) for u, e in failed]).to_csv(a.OUT / f"xb_failed{a.SUFFIX}.csv", index=False)
        if not a.no_summarize:
            summarize(a)
    return dict(n_fail=n_fail)


if __name__ == "__main__":
    res = main()
    sys.exit(1 if res.get("n_fail") else 0)
