"""XI_climate_extrapolation_retest(계획 2.9): WF10 재시험. 주 판정은 캐나다 v4 약관 확인분 확충판의 warm_trim 판 단독이다.

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100) 2.9절, 1절 공통 규약, 0.3 열람 규칙.
공용 골격: scripts/3_deep_learning/xbatch_core.py(동결 모듈 h40·h42·h54 의 sha256 대조, 조각, 봉인, 출력 제한, 판정 도구).
구현 기록(계획 해석과 구현 결정): docs/research/2026-10-04/impl_notes/x_climate_extrapolation_retest.md

판(저장소 별칭)과 역할
  Canada~exp~lic  v4 약관 확인분 캐나다 확충판(data/processed/lgd/run_tables/Canada_expanded_lic, 787셀). warm_trim 이 주 판정(XI-a·b·c),
                  warm(기본 분할 판)은 민감도, cold 는 서술(XI-d).
  Canada~exp      v4 전체판(run_tables/Canada_expanded, 825셀). warm·cold 서술(XI-d). 실행 표가 없으면 건너뛰고 meta 에 적는다.
  Alaska          v3(data/processed). warm·cold 재현(비맹검) 행(XI-d). cold 는 WF10 과 같이 W 1블록이라 유효 분할이 없다.
  Canada          v3. 재현 관문 전용(3차 본 실행 Vppbeb 의 캐나다 키와 대조). 가설·서술 표에는 넣지 않는다.

설계(WF10 과 같다. h54 를 고치지 않고 그 함수와 단위를 쓴다)
  분할 = h54.wf10_split(블록 평균 √TDD 순서로 W 를 대상 셀 25 % 이상, seed_of('wf10-I', 대상, 변형, 분할) 순열로 I 를 25 % 이상, 나머지 A),
  분할 1–10, h54.wf10_plan 의 중복·무효 규칙. n 격자 h54.WF10_GRID = {100, 500, 전량}(|A| 미만만 둔다. 캐나다 v4 는 |A| 237–383 이라
  500 이 없다), 추출 3(전량 1), seed 0·1, catboost_lo 반복 200. 방법 = h54.R10Unit(P1, P2, R1(교차검증 λ, λ 0.25·0.5·1.0),
  R2(교차검증 λ, 고정 λ), D0(catboost, catboost_lo), D1(catboost_lo)). warm_trim = xbatch_core.R10TrimUnit(A 에서 s ≥ min(s_W) 셀을 빼고
  W·I 채점 셀은 그대로). 저장소 이름은 '<별칭>~<변형>|r'(W ∪ I), '<별칭>~<변형>W|r', '<별칭>~<변형>I|r' 이다. v3 판은 별칭이 대상 이름이라
  저장소 이름이 h54(WF10)와 같고, 재현 관문은 이 이름으로 키를 대조한다.

가설(계획 2.9 표, 결과 열람 뒤 설계, XI-a·b·c 맹검)
  XI-a  EP(D0(catboost) − P1) 열세(외삽 손실 양). n {100, 전량}. 지역 행의 4분 판정을 n 마다 내고 WF10-a 규칙(_rule3, 기준 '열세')으로 묶는다.
  XI-b  EP(R1(교차검증 λ) − D0(catboost)) 우세. 같은 형식(WF10-b, 기준 '우세').
  XI-c  W 에서 R1(교차검증 λ) − P1 비열등(두 가중 CI 상한 < 0.5 cm, WF10-c 규칙 h54._ni_text). 보조 δ 1.0 cm 판을 병기한다.
  XI-d  서술: 기본 판, cold, 전체판, 알래스카 재현 행, 다른 레시피의 외삽 손실, 분할별 frac_W_outside_A 와 외삽 폭.
  EP = Δ_W − Δ_I(h54.region_ep: W·I 저장소 각각 h40.contrast 의 분할 안 채점 블록 재표집, 같은 번호끼리 뺀다). 같은 라벨 집합 대비이므로
  주 CI 는 h40.contrast, 보조는 h42.boot_delta_common(1절). δ_rel 의 P0 RMSE 는 EP 행은 총 저장소(W ∪ I), XI-c 행은 W 저장소의 값이다.
  사전 규칙: 판마다 frac_W_outside_A(unit.json, W 채점 셀 가운데 A 의 √TDD 범위 밖 비율)의 분할 평균이 0.5 미만이면 그 판의 판정은
  '외삽 시험 아님'이다(xi_tests.csv 의 그 판 행의 verdict4·verdict4_d10·verdict4_rel·verdict4_common 도 이 문구로 덮고 CI 열은 남긴다).
  warm_trim 과 기본 판의 판정이 다르면 두 판정을 모두 적고 약한 쪽으로 문장을 쓴다(구현 기록 3절의 순서). XI-a·b 는 4분 판정, XI-c 는
  비열등 상태(충족 > 미충족 > 판정 불가·행 없음 > 외삽 시험 아님)로 비교한다. 가설 수준 판정(rule3, _ni_text)도 두 판을 따로 내고 약한 쪽을
  verdict_hypothesis_written 에 쓴다(지지 > 부분 지지 > 기각 > 판정 불가 > 외삽 시험 아님).
  다중성: XI-a·b·c 의 n 별 p 6개(a·b 는 양측 p, c 는 비열등 단측 p × 2)에 Holm, m = 6(보조 열).
  해석 문장: 1절 다섯 갈래. 모든 문장에 '같은 시기의 따뜻한 블록을 쓴 공간 대용'과 외삽 폭(√TDD 차)을 넣는다. 주 행이 모두 판정 불가이면
  '기후 외삽은 판정할 수 없었다'를 쓴다.

재현 관문(--gate, 적합 없음): v3 판(Alaska, Canada)의 키가 3차 본 실행 Vppbeb 조각(results/rescale_wf3/data/processed/wf/shards/wf10__cpu__*)과
  같다(1절 허용 오차, 기본 수준 elm_hematite = 0). 표에는 블록 SSE 차만 있고 RMSE·Δ 는 없다. 범위 점검: 등록 관문 단위(알래스카 warm,
  캐나다 v3 warm·cold)의 기대 분할(양쪽 unit.json 의 expected_splits)마다 두 쪽 조각과 저장소(총, W, I)가 모두 있어야 하고, 이번 실행 키는
  모두 기준에 있어야 하며, 설계 범위(n 격자·추출·seed) 안의 기준 키는 모두 이번 실행에 있어야 한다. 어긋나면 실패다. 두 쪽의 패키지 판
  (기준 deps.log, 이번 unit.json 의 deps)을 표와 <tag>_gate_meta.json 에 적는다(허용 오차는 1절 그대로 0).

산출(<out-dir> = data/processed/xbatch/XI_climate_extrapolation_retest)
  shards/<tag>__cpu__<별칭>__r__s<분할>__<변형>_{runs.csv, blocksse.npz, unit.json}(unit.json 이 완료 표지, --resume 은 설정 해시 대조).
    runs.csv 에는 키별 RMSE·편향 열(rmse_cm, rmse_beq_cm, bias_cm)을 쓰지 않는다(0.3 열람 순서. 집계는 blocksse.npz 의 SSE 만 쓴다).
  봉인(sealed/, 계획 0.3 열람 순서 2: 작업 R2a 제출 뒤 연다): <tag>_tests.csv(모든 대비 행), <tag>_hyp.csv(가설·Holm·해석 문장),
    <tag>_curve.csv(저장소별 곡선 값), <tag>_meta.json. 화면에는 행 수와 sha256 만 쓴다.
  봉인 밖(라벨 값 미사용 또는 차만 담는 표): <tag>_count.csv(--count-only), <tag>_structure.csv(분할별 셀·블록 수, frac_W_outside_A, 외삽 폭),
    <tag>_timing.csv, <tag>_failed.csv(식별 열만), <tag>_gate.csv(--gate).

명령행(실행 보호와 출력 제한은 xbatch_core. 로컬은 세기·스모크·집계·관문만)
  로컬 자원(1절): 로컬 실행은 모드와 관계없이 가용 메모리 30 GB 미만이면 최대 1시간 기다리고, 주소 공간 10 GB 상한을 건다. MALLOC_ARENA_MAX 가
  없으면 명령행 실행은 MALLOC_ARENA_MAX=2 로 다시 시작한다(함수 호출은 거부). 허용 표지 없는 스모크는 스레드 2 로 자른다(5절 1a).
  Rescale 본 실행(허용 표지, 스모크 아님)에서 등록 판(Canada~exp~lic, 관문용 Alaska·Canada v3)의 실행 표가 없으면 중단한다(전체판 825셀만 예외).
  세기:   CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/x_climate_extrapolation_retest.py --count-only --threads 1
  스모크: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c <코어 4개> python3 scripts/3_deep_learning/x_climate_extrapolation_retest.py \
            --smoke --threads 2 --workers 0 2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'
  본 실행: WF_RESCALE=1 python3 scripts/3_deep_learning/x_climate_extrapolation_retest.py --workers 22 --threads 4 --resume --no-summarize
  분할 실행: 같은 명령에 --shard I/K(정렬한 단위 목록의 I 번째 몫, 0 ≤ I < K)
  집계:   WF_RESCALE=1 python3 scripts/3_deep_learning/x_climate_extrapolation_retest.py --summarize-only   (로컬은 재표집 1,000회 상한)
  관문:   python3 scripts/3_deep_learning/x_climate_extrapolation_retest.py --gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import xbatch_core as XB                                                                             # noqa: E402  numpy 보다 먼저(스레드 환경 변수)

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4
for _k in ("h40", "h42", "h54"):                                            # 동결 모듈은 xbatch_core 가 sha256 을 대조해 읽은 것만 쓴다
    XB.assert_frozen(_k)

# ================================================================ 고정값(계획 2.9. 바꾸면 사전 등록에서 벗어난다)
EXP_ID = "XI"
EXP_NAME = XB.EXP_NAMES[EXP_ID]
TAG = "xi"
MODE = W.MODE_R                                                             # 'r'
LO, HI = XB.LO, XB.HI
LAM_CV, LAM_BASE = XB.LAM_CV, XB.LAM_BASE
GRID = tuple(int(v) for v in W.WF10_GRID)                                   # (100, 500, −1)
SPLITS = tuple(XB.WF10_SPLITS)                                              # 1–10
PRIMARY_NS = (100, -1)                                                      # XI-a·b·c 의 n
FRAC_MIN = 0.5                                                              # '외삽 시험 아님' 규칙의 하한(분할 평균)
HOLM_M = 6
TRIM = "warm_trim"
VARIANTS_ALL = ("warm_trim", "warm", "cold")
DESIGN_LABEL = "결과 열람 뒤 설계"
SPACE_PROXY = "같은 시기의 따뜻한 블록을 쓴 공간 대용"
NOT_EXTRAP = "외삽 시험 아님"
NOT_TESTED = "시험하지 않음"

# 판: 별칭 → (실행 표 디렉터리 또는 None(v3), 표 안의 대상 이름, 변형, 자료 설명)
VERSIONS = {
    "Canada~exp~lic": dict(spec="Canada_expanded_lic", target="Canada", variants=("warm_trim", "warm", "cold"), data="v4 약관 확인분(787셀)"),
    "Canada~exp": dict(spec="Canada_expanded", target="Canada", variants=("warm", "cold"), data="v4 전체판(825셀)"),
    "Alaska": dict(spec=None, target="Alaska", variants=("warm", "cold"), data="v3"),
    "Canada": dict(spec=None, target="Canada", variants=("warm", "cold"), data="v3"),
}
PRIMARY = ("Canada~exp~lic", TRIM)
BASIC = ("Canada~exp~lic", "warm")
GATE_ALIASES = ("Alaska", "Canada")                                         # v3 판: 저장소 이름이 Vppbeb 와 같다
GATE_ONLY = (("Canada", "warm"), ("Canada", "cold"))                        # 가설·서술 표에 넣지 않는 판
ROLE = {("Canada~exp~lic", TRIM): "주", ("Canada~exp~lic", "warm"): "민감도(기본 판)", ("Canada~exp~lic", "cold"): "서술(cold)",
        ("Canada~exp", "warm"): "서술(전체판 825셀)", ("Canada~exp", "cold"): "서술(전체판 825셀, cold)",
        ("Alaska", "warm"): "서술(알래스카 재현 행)", ("Alaska", "cold"): "서술(알래스카 재현 행, cold)",
        ("Canada", "warm"): "재현 관문(v3)", ("Canada", "cold"): "재현 관문(v3)"}
BLIND = {"Alaska": "재현(비맹검)", "Canada": "재현(비맹검)"}               # 그 밖의 판은 '맹검'(캐나다 v4 의 모형 예측은 본 적 없다)

# 대비 정의: 항목 → (종류, gA, gB 를 만드는 함수). 종류 ep = 외삽 손실 Δ_W − Δ_I, W = W 저장소의 Δ
ITEMS = {
    "a": ("ep", lambda n: (W.gk("D0", n, HI, 1.0), W.gk("P1", n, "none")), "EP[D0(catboost)-P1]"),
    "b": ("ep", lambda n: (W.gk("R1", n, LO, LAM_CV), W.gk("D0", n, HI, 1.0)), "EP[R1(λ cv)-D0(catboost)]"),
    "c": ("W", lambda n: (W.gk("R1", n, LO, LAM_CV), W.gk("P1", n, "none")), "R1(λ cv)-P1[W]"),
}
HYP_OF_ITEM = {"a": "XI-a", "b": "XI-b", "c": "XI-c"}
KIND_OF_HYP = {"XI-a": "열세", "XI-b": "우세"}
OTHERS = (("D0", LO, 1.0), ("D1", LO, 1.0), ("R2", LO, LAM_CV), ("R1", LO, LAM_BASE), ("R1", LO, 1.0), ("P2", "none", None))

# 해석 문장(1절 다섯 갈래). {a} = |Δ|(cm)
SENT = {
    "a": {"우세": "외삽 블록에서 직접 ML(catboost)의 재보정 Stefan 대비 오차는 보간 블록보다 {a} cm 작았다",
          "열세": "외삽 블록에서 직접 ML(catboost)의 재보정 Stefan 대비 오차는 보간 블록보다 {a} cm 컸다",
          "동등": "외삽 블록과 보간 블록에서 직접 ML(catboost)의 재보정 Stefan 대비 오차는 0.5 cm 안에서 같았다",
          "미결정": "직접 ML(catboost)의 외삽 손실(재보정 Stefan 대비)은 차이를 확인하지 못했다",
          "판정 불가": "직접 ML(catboost)의 외삽 손실(재보정 Stefan 대비)은 판정할 수 없었다"},
    "b": {"우세": "재보정 앵커 잔차(교차검증 λ)의 직접 ML(catboost) 대비 오차는 외삽 블록에서 보간 블록보다 {a} cm 작았다",
          "열세": "재보정 앵커 잔차(교차검증 λ)의 직접 ML(catboost) 대비 오차는 외삽 블록에서 보간 블록보다 {a} cm 컸다",
          "동등": "재보정 앵커 잔차(교차검증 λ)의 직접 ML(catboost) 대비 오차는 외삽 블록과 보간 블록에서 0.5 cm 안에서 같았다",
          "미결정": "재보정 앵커 잔차와 직접 ML 의 외삽 손실 차이는 확인하지 못했다",
          "판정 불가": "재보정 앵커 잔차와 직접 ML 의 외삽 손실 차이는 판정할 수 없었다"},
    "c": {"우세": "외삽 블록에서 재보정 앵커 잔차(교차검증 λ)는 재보정 Stefan 보다 오차가 {a} cm 작았다",
          "열세": "외삽 블록에서 재보정 앵커 잔차(교차검증 λ)는 재보정 Stefan 보다 오차가 {a} cm 컸다",
          "동등": "외삽 블록에서 재보정 앵커 잔차(교차검증 λ)와 재보정 Stefan 의 오차는 0.5 cm 안에서 같았다",
          "미결정": "외삽 블록에서 재보정 앵커 잔차(교차검증 λ)와 재보정 Stefan 의 오차 차이는 확인하지 못했다",
          "판정 불가": "외삽 블록에서 재보정 앵커 잔차(교차검증 λ)와 재보정 Stefan 의 오차 차이는 판정할 수 없었다"},
}
ALL_NA_SENTENCE = "기후 외삽은 판정할 수 없었다."
# 약한 쪽 순서(구현 기록 3절): 방향 판정 2 > 동등 1 > 미결정 0 > 판정 불가·행 없음 −1 > 외삽 시험 아님 −2. 같은 순위의 반대 방향은 미결정.
STRENGTH = {"우세": 2, "열세": 2, "동등": 1, "미결정": 0, "판정 불가": -1, "행 없음": -1, NOT_EXTRAP: -2}
# XI-c(비열등)의 n 별 상태와 약한 쪽 순서: 충족 1 > 미충족 0 > 판정 불가·행 없음 −1 > 외삽 시험 아님 −2(h54._ni_text 의 n 별 규칙과 같다)
NI_OK, NI_NO = "충족", "미충족"
NI_STRENGTH = {NI_OK: 1, NI_NO: 0, "판정 불가": -1, "행 없음": -1, NOT_EXTRAP: -2}
# 가설 수준 판정(rule3, _ni_text 문구의 앞머리)의 약한 쪽 순서: 지지 2 > 부분 지지 1 > 기각 0 > 판정 불가 −1 > 외삽 시험 아님 −2
HYP_PREFIX = (("부분 지지", 1), ("지지", 2), ("기각", 0), ("판정 불가", -1), (NOT_EXTRAP, -2))
# 외삽 시험 아님 규칙이 덮는 판정 열(CI 열은 남긴다)과 비우는 판정 부속 열
NOT_EXTRAP_VERDICT_COLS = ("verdict4", "verdict4_d10", "verdict4_rel", "verdict4_common")
NOT_EXTRAP_CLEAR_COLS = ("limit_dependence", "ci_dependence", "draw_dependence", "small_note")
NOT_EXTRAP_FLAG_COLS = ("noninf", "noninf_d10", "noninf_common", "superior_common", "worse")
# runs.csv 에서 빼는 열(0.3 열람 순서: 키별 RMSE·편향은 봉인 밖에 두지 않는다. 집계는 blocksse.npz 의 SSE 만 쓴다)
RUN_DROP_COLS = ("rmse_cm", "rmse_beq_cm", "bias_cm")
DEPS_NAMES = ("catboost", "scikit-learn", "pandas", "scipy", "numpy")
# 재현 관문의 등록 단위(2.9 재현 관문: 알래스카 wf10 키, 캐나다 v3 판). 알래스카 cold 는 WF10 과 같이 유효 분할이 없다
GATE_REQUIRED = (("Alaska", "warm"), ("Canada", "warm"), ("Canada", "cold"))
# 1절 로컬 자원: 가용 메모리 하한 30 GB(미만이면 기다린다), 대기 상한, 작업 하나 10 GB. 5절 1a 로컬 스모크 2스레드
MEM_MIN_GB, MEM_WAIT_S, MEM_CAP_GB = 30.0, 3600.0, 10.0
SMOKE_THREADS = 2
# 묶음(4절)에 있어야 하는 입력(xbatch_core.PAYLOAD_INPUTS 에도 넣었다). 실행 표의 토양 표는 e5_soil_tdd_v4.csv 로 가는 기호 연결이다
PAYLOAD_EXTRA = ("data/processed/fidelity_base_v3.csv", "data/processed/e5_soil_tdd_v3.csv", "data/processed/lg_subregion_map_v1.csv",
                 "data/processed/e5_soil_tdd_v4.csv", "data/processed/lgd/run_tables/Canada_expanded_lic/fidelity_base_v3.csv",
                 "data/processed/lgd/run_tables/Canada_expanded_lic/e5_soil_tdd_v3.csv")
OPTIONAL_ALIASES = ("Canada~exp",)                                          # 실행 표가 없어도 되는 판(전체판 825셀, 구현 기록 5절 1)
COUNT_COLS = ["alias", "variant", "split", "n_A", "nb_A", "n_trim", "n_eval_W", "n_eval_I", "nb_eval_W", "nb_eval_I", "fits", "est_s",
              "frac_W_outside_A", "extrap_width"]                          # 세기 표·화면의 허용 열(라벨 값에서 나온 통계는 없다)
STRUCT_COLS = ["alias", "variant", "split", "role", "data_version", "n_A", "nb_A", "n_trim", "n_eval", "n_eval_W", "n_eval_I", "nb_eval_W",
               "nb_eval_I", "s_A_min", "s_A_max", "s_W_median", "frac_W_outside_A", "extrap_width", "W_blocks", "I_blocks", "dup_of", "valid",
               "n_valid_splits"]


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XI WF10 재시험(계획 2.9)")
    ap.add_argument("--count-only", action="store_true", help="학습 없이 작업 단위·적합 수·구조(셀·블록 수, frac_W_outside_A, 외삽 폭)만 센다")
    ap.add_argument("--smoke", action="store_true", help="작은 설정(Canada~exp~lic warm_trim·warm, v3 Canada warm, 분할 1, 추출 1, seed 1). 로컬 허용(스레드 2)")
    ap.add_argument("--summarize-only", action="store_true", help="조각을 읽어 봉인 폴더에 판정 표를 쓴다")
    ap.add_argument("--gate", action="store_true", help="재현 관문: v3 판 키를 Vppbeb 조각과 대조한다(적합 없음)")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-local", action="store_true", help="본 실행을 로컬에서 허용한다(WF_RESCALE=1 과 같은 효과)")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--shard", default="", help="I/K: 정렬한 작업 단위 목록에서 색인 mod K = I 인 단위만 실행한다")
    ap.add_argument("--workers", type=int, default=1, help="프로세스 수. 0 = 풀 없이 차례로")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--versions", default=",".join(VERSIONS), help="판 별칭 쉼표 목록")
    ap.add_argument("--variants", default=",".join(VARIANTS_ALL), help="변형 쉼표 목록(판마다 등록된 것만 쓴다)")
    ap.add_argument("--splits", type=int, default=len(SPLITS), help="I 블록 순열 seed 1..K(등록 10)")
    ap.add_argument("--seeds", type=int, default=len(XB.SEEDS), help="CatBoost seed 수(등록 2). 시험용")
    ap.add_argument("--cb-iters", type=int, default=200, help="catboost_lo 반복 수(등록 200). 시험용")
    ap.add_argument("--draws-cap", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--out-dir", default=str(XB.out_dir(EXP_ID).relative_to(XB.ROOT)))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lgd-dir", default="data/processed/lgd/run_tables")
    ap.add_argument("--gate-ref", default="results/rescale_wf3/data/processed/wf/shards", help="3차 본 실행 Vppbeb 의 wf10 조각 폴더")
    ap.add_argument("--gate-level", default="elm_hematite", choices=sorted(XB.GATE_TOL), help="1절 재현 관문 허용 오차 수준")
    ap.add_argument("--gate-ref-deps", default="results/rescale_wf3/logs_wf/deps.log", help="기준(Vppbeb) 작업의 패키지 판 기록")
    o = ap.parse_args(argv)
    o.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(o)


def _abs(p):
    return Path(p) if os.path.isabs(str(p)) else XB.ROOT / str(p)


def finalize(o):
    o.PERMIT = XB.run_permitted(o.ARGV)
    o.threads_asked = int(o.threads)
    o.threads, nb = XB.local_limits(o.threads, o.nboot, o.ARGV)
    if o.smoke and not o.PERMIT:                                   # 5절 1a 로컬 스모크(코어 4개, 2스레드). local_limits 의 상한 4 보다 좁다
        o.threads = min(int(o.threads), SMOKE_THREADS)
    o.nboot_asked, o.nboot = int(o.nboot), int(nb)
    o.VERSIONS = [v for v in (s.strip() for s in o.versions.split(",")) if v]
    bad = [v for v in o.VERSIONS if v not in VERSIONS]
    if bad:
        raise SystemExit(f"알 수 없는 판 {bad}. 가능한 판: {list(VERSIONS)}")
    o.VARIANTS = [v for v in (s.strip() for s in o.variants.split(",")) if v in VARIANTS_ALL]
    o.SPLITS = list(range(1, int(o.splits) + 1))
    o.GRID = list(GRID)
    o.SUFFIX = "_smoke" if o.smoke else ""
    if o.smoke:                                                    # 스모크: 주 판·기본 판과 재현 관문 경로(v3 캐나다 warm)를 작은 설정으로 한 번 지난다
        o.VERSIONS = [PRIMARY[0], "Canada"]; o.VARIANTS = [TRIM, "warm"]; o.SPLITS = [1]; o.seeds = 1; o.draws_cap = o.draws_cap or 1
        o.GRID = [100, -1]
        o.nboot = min(int(o.nboot), 500)
    o.TAG = o.tag + o.SUFFIX
    o.OUT = _abs(o.out_dir); o.PROC = _abs(o.data_dir); o.LGD = _abs(o.lgd_dir)
    o.SHARDS = o.OUT / "shards"
    o.ALLOW_LOCAL = "--allow-local" in o.ARGV
    return o


_ARGS: dict = {}


def version_dir(o, alias):
    spec = VERSIONS[alias]["spec"]
    return o.PROC if spec is None else o.LGD / spec


def version_available(o, alias):
    """판의 실행 표(기본 표, 토양 표. v3 는 하위 지역 대응표도)가 있는지. 실행 표의 토양 표는 기호 연결이라 대상이 있어야 참이다."""
    d = version_dir(o, alias)
    ok = (d / "fidelity_base_v3.csv").exists() and (d / "e5_soil_tdd_v3.csv").exists()
    return ok and ((d / "lg_subregion_map_v1.csv").exists() if VERSIONS[alias]["spec"] is None else True)


def require_versions(o, missing):
    """허용 표지가 있는 본 실행(스모크 아님)에서 등록 판의 실행 표가 없으면 중단한다(전체판 825셀만 예외, 구현 기록 5절 1).
    묶음에서 빠진 입력이 '실행 표 없음'으로 조용히 건너뛰어져 주 판정이 판정 불가가 되는 것을 막는다."""
    hard = [al for al in missing if al not in OPTIONAL_ALIASES]
    if hard and o.PERMIT and not o.smoke:
        raise SystemExit(f"[입력] 등록 판의 실행 표가 없다: {hard}(묶음 목록 xbatch_core.PAYLOAD_INPUTS, 이 모듈의 PAYLOAD_EXTRA 를 확인한다)")
    return hard


def version_args(o, alias):
    """판마다 h54 인자(자료 디렉터리 = 그 판의 표). h54.build_rctx10·wf10_plan 이 a.PROC 의 자료를 읽으므로 판마다 따로 만든다."""
    key = (alias, str(o.PROC), str(o.LGD), int(o.threads), int(o.cb_iters), int(o.seeds), int(o.draws_cap), tuple(o.GRID), tuple(o.SPLITS))
    if key not in _ARGS:
        a = XB.h54_args(exp="wf10", splits=XB.TRANSFER_SPLITS, grid=o.GRID, threads=o.threads, cb_iters=o.cb_iters, seeds=o.seeds, nboot=o.nboot,
                        data_dir=str(version_dir(o, alias)), lgd_dir=str(o.LGD), out_dir=str(o.OUT), tag=o.TAG, draws_cap=o.draws_cap,
                        allow_local=o.ALLOW_LOCAL)
        a.SPLITS10 = list(o.SPLITS)
        a.VAR10 = ["warm", "cold"]
        _ARGS[key] = a
    return _ARGS[key]


def data_sha(o, alias):
    """자료 판 식별(h54.data_sha 와 같은 형식). v3 는 하위 지역 대응표도 넣는다."""
    d = version_dir(o, alias)
    s = f"{W.file_sha(d / 'fidelity_base_v3.csv')}:{W.file_sha(d / 'e5_soil_tdd_v3.csv')}"
    return s + (f":{W.file_sha(d / 'lg_subregion_map_v1.csv')}" if VERSIONS[alias]["spec"] is None else "")


def base_variant(variant):
    return "warm" if variant == TRIM else str(variant)


def unit_cfg(o, variant, dsha):
    """결과에 영향을 주는 설정(대상·분할·tag·워커는 넣지 않는다). 판 차이는 data_sha 에 들어간다(공통 해시에서 빠진다)."""
    return XB.make_unit_cfg(EXP_ID, variant=variant, data_sha=dsha, grid=[int(v) for v in o.GRID], draws=[int(W.WF10_DRAWS)], draws_cap=int(o.draws_cap),
                            seeds=list(range(int(o.seeds))), cb_iters=int(o.cb_iters), frac=float(W.WF10_FRAC),
                            methods=["P1", "P2", "R1", "R2", "D0", "D1"], learners=dict(D0=[HI, LO], R1=LO, R2=LO, D1=LO), mode=MODE,
                            draw_mode="r10+variant[0]", unit="h54.R10Unit", trim="xbatch_core.R10TrimUnit(s ≥ min(s_W), W 블록의 모든 셀)",
                            split_rule="h54.wf10_split, I = seed_of('wf10-I', target, variant, split)")


# ================================================================ 문맥과 단위
def extrap_meta(c, variant):
    """외삽 폭(공변량 √TDD 만): warm = W 채점 셀 √TDD 중앙값 − A 의 √TDD 최댓값, cold = A 의 최솟값 − W 중앙값. frac_W_outside_A 는 h54
    (또는 warm_trim) 의 meta 값을 그대로 둔다."""
    sA = np.asarray(c.sA, float)
    sW = np.asarray(c.sB, float)[np.asarray(c.maskW, bool)]
    if not (np.isfinite(sA).any() and np.isfinite(sW).any()):
        return dict(s_W_median=np.nan, extrap_width=np.nan)
    med = float(np.nanmedian(sW))
    width = med - float(np.nanmax(sA)) if variant == "warm" else float(np.nanmin(sA)) - med
    return dict(s_W_median=med, extrap_width=float(width), s_A_min=float(np.nanmin(sA)), s_A_max=float(np.nanmax(sA)),
                width_rule="warm: median(s_W) − max(s_A), cold: min(s_A) − median(s_W)")


def build_ctx(o, alias, variant, split):
    """(h54 인자, 문맥). warm_trim 은 xbatch_core.build_rctx10_trim, 그 밖은 h54.build_rctx10."""
    a = version_args(o, alias)
    tgt = VERSIONS[alias]["target"]
    c = XB.build_rctx10_trim(a, tgt, int(split)) if variant == TRIM else W.build_rctx10(a, tgt, int(split), variant)
    c.meta.setdefault("n_trim", 0)
    c.meta.update(extrap_meta(c, base_variant(variant)))
    c.meta.update(alias=alias, data_version=VERSIONS[alias]["data"], role=ROLE.get((alias, variant), ""))
    return a, c


class XIUnit(W.R10Unit):
    """WF10 단위(h54.R10Unit 그대로)에서 저장소 이름의 대상 자리만 판 별칭으로 둔다. 별칭이 대상 이름이면 h54 와 같다."""

    def __init__(self, a, c, alias, variant, dry=False):
        if variant not in W.WF10_VARIANTS:
            raise ValueError(f"wf10 변형은 {W.WF10_VARIANTS} 가운데 하나다: {variant}")
        self._init_base(a, c, "wf10", f"{alias}~{variant}|{c.mode}", variant, dry)
        self._km: dict = {}
        self._std = None
        self._di = None
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)


class XITrimUnit(XB.R10TrimUnit):
    """warm_trim 단위(xbatch_core.R10TrimUnit 그대로)에서 저장소 이름의 대상 자리만 판 별칭으로 둔다."""

    def __init__(self, a, c, alias, dry=False):
        self._init_base(a, c, "wf10", f"{alias}~{TRIM}|{c.mode}", TRIM, dry)
        self._km: dict = {}
        self._std = None
        self._di = None
        self.add("P0", "none", "cell", 0, 0, -1, 0.0, c.E0 * c.sB, c.E0, 0)


def make_unit(a, c, alias, variant, dry=False):
    return XITrimUnit(a, c, alias, dry) if variant == TRIM else XIUnit(a, c, alias, variant, dry)


def _plain(meta):
    return {k: v for k, v in meta.items() if not isinstance(v, (dict, list))}


def public_rows(rows):
    """조각 runs.csv 에 쓸 행: 키별 RMSE·편향 열(RUN_DROP_COLS)을 뺀다. 실패 표(n_nonfinite, fit_flag)와 시간 표에 쓰는 열은 남는다."""
    return [{k: v for k, v in r.items() if k not in RUN_DROP_COLS} for r in rows]


def count_row(c, stats, alias, variant, split):
    """세기 표의 한 행(허용 열만). 셀 수·블록 수·적합 수·추정 시간과 공변량 √TDD 만 쓴 값이다."""
    m = c.meta
    r = dict(alias=alias, variant=variant, split=int(split), n_A=int(len(c.yA)), nb_A=int(m.get("nb_A", 0)), n_trim=int(m.get("n_trim", 0)),
             n_eval_W=int(np.sum(c.maskW)), n_eval_I=int(np.sum(~np.asarray(c.maskW, bool))), nb_eval_W=int(m.get("nb_eval_W", 0)),
             nb_eval_I=int(m.get("nb_eval_I", 0)), fits=int(sum(stats["n_fit"].values())), est_s=round(float(sum(stats["est_detail"].values())), 2),
             frac_W_outside_A=float(m.get("frac_W_outside_A", np.nan)), extrap_width=float(m.get("extrap_width", np.nan)))
    return {k: r[k] for k in COUNT_COLS}


def run_unit(o, alias, variant, split, dry=False, expected=None):
    """작업 단위 하나. dry 이면 세기 행(허용 열)을, 아니면 조각을 쓰고 unit dict 를 돌려준다."""
    t0 = time.time()
    a, c = build_ctx(o, alias, variant, split)
    U = make_unit(a, c, alias, variant, dry)
    U.run()
    rows, stores, stats = U.finish()
    if dry:
        return count_row(c, stats, alias, variant, split)
    cfg = unit_cfg(o, variant, data_sha(o, alias))
    unit = {**stats, **_plain(c.meta)}
    unit.update(exp=EXP_ID, alias=alias, elapsed_s=round(time.time() - t0, 1), n_fit_total=int(sum(stats["n_fit"].values())), n_A=int(len(c.yA)),
                n_eval=int(len(c.yB)), E0=float(c.E0), n_eval_W=int(np.sum(c.maskW)), n_eval_I=int(np.sum(~np.asarray(c.maskW, bool))),
                deps=XB.deps_versions(DEPS_NAMES))
    return XB.write_shard(o.SHARDS, o.TAG, alias, MODE, int(split), public_rows(rows), stores, cfg, unit=unit, variant=variant,
                          expected=expected if expected is not None else [int(split)], code_file=__file__)


# ================================================================ 작업 단위 목록
def enumerate_units(o):
    """(별칭, 변형, 분할) 목록, 건너뛴 기록, 기대 분할. 분할 구조는 h54.wf10_plan(warm_trim 은 warm 의 구조).
    허용 표지가 있는 본 실행에서 등록 판의 실행 표가 없으면 require_versions 가 중단한다."""
    units, skipped, expected = [], [], {}
    missing = [al for al in o.VERSIONS if any(v in o.VARIANTS for v in VERSIONS[al]["variants"]) and not version_available(o, al)]
    require_versions(o, missing)
    for alias in o.VERSIONS:
        vars_ = [v for v in VERSIONS[alias]["variants"] if v in o.VARIANTS]
        if not vars_:
            continue
        if alias in missing:
            skipped += [dict(alias=alias, variant=v, split=-1, status="실행 표 없음") for v in vars_]
            continue
        a = version_args(o, alias)
        _, D = W.get_data(a)
        tgt = VERSIONS[alias]["target"]
        for v in vars_:
            keep, skip, _ = W.wf10_plan(a, D, tgt, base_variant(v))
            expected[(alias, v)] = [int(s_) for s_ in keep]
            skipped += [dict(alias=alias, variant=v, split=int(sp), status=st_) for sp, st_, _ in skip]
            units += [(alias, v, int(sp)) for sp in keep]
    return sorted(units, key=lambda u: (u[0], u[1], u[2])), skipped, expected


def shard_select(units, spec):
    """--shard I/K: 정렬한 목록에서 색인 mod K = I 인 단위."""
    if not spec:
        return list(units)
    try:
        i, k = (int(v) for v in str(spec).split("/"))
    except ValueError:
        raise SystemExit(f"--shard 는 I/K 형식이다: {spec}")
    if not (k >= 1 and 0 <= i < k):
        raise SystemExit(f"--shard {spec}: 0 ≤ I < K 이어야 한다")
    return [u for j, u in enumerate(units) if j % k == i]


def unit_name(u):
    return "|".join(str(v) for v in u)


# ================================================================ 실행(워커)
_STATE: dict = {}


def _worker_init(argv, expected_items):
    _STATE["o"] = parse_args(list(argv))
    _STATE["expected"] = {tuple(k): v for k, v in expected_items}


def _worker_run(alias, variant, split):
    o = _STATE["o"]
    u = run_unit(o, alias, variant, int(split), expected=_STATE["expected"].get((alias, variant)))
    return dict(alias=alias, variant=variant, split=int(split), status=str(u.get("status", "")))


def count_only(o, units, skipped, unit_fn=None, write=True):
    """세기(라벨 값 미사용 범주): 단위마다 dry 실행으로 적합 수와 구조를 센다. 화면과 표에는 COUNT_COLS 만 쓴다."""
    fn = unit_fn or (lambda u: run_unit(o, *u, dry=True))
    rows = [{k: r.get(k) for k in COUNT_COLS} for r in (fn(u) for u in units)]
    df = pd.DataFrame(rows, columns=COUNT_COLS)
    if write and len(df):
        XB.check_out_dir(o.OUT)
        o.OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(o.OUT / f"{o.TAG}_count.csv", index=False)
    print(f"[count-only] 작업 단위 {len(units)} · 건너뜀 {len(skipped)} · 분할 {o.SPLITS[0]}–{o.SPLITS[-1]} · seed {o.seeds} · n 격자 {o.GRID}", flush=True)
    if len(df):
        g = df.groupby(["alias", "variant"], sort=False).agg(units=("split", "size"), fits=("fits", "sum"), est_h=("est_s", "sum"),
                                                              nA_min=("n_A", "min"), nA_max=("n_A", "max"), trim_min=("n_trim", "min"),
                                                              trim_max=("n_trim", "max"), nbW=("nb_eval_W", "max"), nbI_min=("nb_eval_I", "min"),
                                                              frac_mean=("frac_W_outside_A", "mean"), width_min=("extrap_width", "min"),
                                                              width_max=("extrap_width", "max")).reset_index()
        g["est_h"] = (g.est_h / 3600).round(3)
        for c_ in ("frac_mean", "width_min", "width_max"):
            g[c_] = g[c_].round(3)
        g["rule_frac≥0.5"] = g.frac_mean >= FRAC_MIN
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(g.to_string(index=False), flush=True)
        tot_h = float(df.est_s.sum()) / 3600
        print(f"[count-only] 총 적합 {int(df.fits.sum()):,} · 추정 누적 {tot_h:.2f} CPU-h(4스레드 프로세스, h54 추정식) · 워커 22개 약 {tot_h / 22:.3f} h",
              flush=True)
    if skipped:
        print(f"[count-only] 건너뜀: {dict(Counter(s_['status'] for s_ in skipped))}", flush=True)
    return df


# ================================================================ 집계(봉인)
def structure_table(units):
    """unit.json 의 구조 열(라벨 값 미사용: 셀·블록 수, 공변량 √TDD, frac_W_outside_A, 외삽 폭)."""
    rows = []
    for u in units:
        r = {k: u.get(k, np.nan) for k in STRUCT_COLS}
        r["alias"], r["variant"] = u.get("target", u.get("alias")), u.get("variant", "")
        rows.append(r)
    df = pd.DataFrame(rows, columns=STRUCT_COLS)
    return df.sort_values(["alias", "variant", "split"]).reset_index(drop=True) if len(df) else df


def frac_summary(struct):
    """판·변형별 frac_W_outside_A 분할 평균, 외삽 폭 평균, 사전 규칙 결과(평균 < 0.5 → 외삽 시험 아님)."""
    out = {}
    if not len(struct):
        return out
    for (al, v), g in struct.groupby(["alias", "variant"]):
        fm = float(np.nanmean(g.frac_W_outside_A.astype(float))) if g.frac_W_outside_A.notna().any() else np.nan
        out[(al, v)] = dict(frac_mean=fm, width_mean=float(np.nanmean(g.extrap_width.astype(float))) if g.extrap_width.notna().any() else np.nan,
                            n_splits=int(len(g)), extrap_test=bool(np.isfinite(fm) and fm >= FRAC_MIN))
    return out


def _with_p0(s, tm_ref):
    if s is None:
        return None
    p0c, p0b = XB.grp_rmse2(tm_ref, XB.P0_GRP) if tm_ref is not None else (np.nan, np.nan)
    s.update(p0_cell=p0c, p0_beq=p0b, ci_kind="same", xdist=None, xdist_beq=None)
    return s


def ep_stat(tms, alias, variant, gA, gB):
    """외삽 손실 Δ_W(A − B) − Δ_I(A − B)(h54.region_ep). δ_rel 의 P0 RMSE 는 총 저장소(W ∪ I)."""
    base = f"{alias}~{variant}"
    s = W.region_ep(tms.get(f"{base}W|{MODE}"), tms.get(f"{base}I|{MODE}"), gA, gB)
    return _with_p0(s, tms.get(f"{base}|{MODE}"))


def part_stat(tms, alias, variant, part, gA, gB):
    """W 또는 I 저장소의 같은 라벨 집합 대비(xbatch_core.region_stat_same: h40.contrast, 보조 h42.boot_delta_common, P0 RMSE)."""
    tm = tms.get(f"{alias}~{variant}{part}|{MODE}")
    return XB.region_stat_same(tm, gA, gB) if tm is not None else None


def _row(s, name, label, **kw):
    """지역 행 하나(xbatch_core.pool 의 보조 열 포함). 통계가 없으면 None."""
    if s is None:
        return None
    rows = XB.pool({name: s}, [name], label, registered=1)
    r = next((q for q in rows if q.get("scope") == "region"), None)
    if r is not None:
        r.update(kw)
    return r


def apply_extrap_rule(r, extrap_ok):
    """2.9 사전 규칙: 판의 frac_W_outside_A 분할 평균이 0.5 미만이면 그 판 행의 판정을 '외삽 시험 아님'으로 적는다. 판정 열(4분 판정 세 한계와
    공통 CI 판정, 추출 조건부 판정)을 덮고 판정에 딸린 표기·판정 표지를 비운다. CI·p 열은 그대로 둔다. extrap_rule 열에 적용 여부를 남긴다."""
    if extrap_ok:
        r["extrap_rule"] = ""
        return r
    for k in NOT_EXTRAP_VERDICT_COLS:
        r[k] = NOT_EXTRAP
    if r.get("verdict4_draw_cond"):
        r["verdict4_draw_cond"] = NOT_EXTRAP
    for k in NOT_EXTRAP_CLEAR_COLS:
        r[k] = ""
    for k in NOT_EXTRAP_FLAG_COLS:
        r[k] = None
    r["extrap_rule"] = NOT_EXTRAP
    return r


def xi_tests(tms, fracs):
    """모든 대비 행(XI-a·b·c 의 주 행, 그 밖은 XI-d 서술). 재현 관문 전용 판(v3 캐나다)은 넣지 않는다. 사전 규칙을 만족하지 않는 판의 행은
    apply_extrap_rule 로 판정을 '외삽 시험 아님'으로 덮는다(XI-d 서술 행 포함)."""
    rows = []
    combos = sorted({tuple(nm.split("|")[0].rsplit("~", 1)) for nm in tms if nm.endswith(f"|{MODE}") and "~" in nm})
    combos = [(al, v) for al, v in combos if v in VARIANTS_ALL and (al, v) not in GATE_ONLY]
    for al, v in combos:
        fr = fracs.get((al, v), {})
        nm_tot = f"{al}~{v}|{MODE}"
        tmT = tms.get(nm_tot)
        if tmT is None:
            continue
        n0 = len(rows)
        ns = sorted({int(g[4]) for gd in tmT.idx.values() for g in gd if g[0] == "P1"}, key=lambda q: (q == -1, q))
        common = dict(alias=al, variant=v, role=ROLE.get((al, v), ""), data_version=VERSIONS.get(al, {}).get("data", ""),
                      blind=BLIND.get(al, "맹검"), design=DESIGN_LABEL, frac_mean=fr.get("frac_mean", np.nan), width_mean=fr.get("width_mean", np.nan),
                      extrap_test=fr.get("extrap_test", False))
        for n in ns:
            prim = (al, v) == PRIMARY and n in PRIMARY_NS
            for item, (kind, gfn, lab) in ITEMS.items():
                gA, gB = gfn(n)
                s = ep_stat(tms, al, v, gA, gB) if kind == "ep" else part_stat(tms, al, v, "W", gA, gB)
                r = _row(s, nm_tot, f"{lab}|{al}~{v}|n{W.nlab(n)}", test_id=HYP_OF_ITEM[item] if prim else "XI-d", item=item, n=int(n),
                         kind=kind, hypothesis=bool(prim), **common)
                if r is not None:
                    rows.append(r)
            for m, lr, lam in OTHERS:                                      # 서술: 다른 레시피의 외삽 손실과 W·I 의 Δ(P1 기준)
                gA, gB = W.gk(m, n, lr, lam), W.gk("P1", n, "none")
                lab = W._plab(m, lr, lam)
                r = _row(ep_stat(tms, al, v, gA, gB), nm_tot, f"EP[{lab}-P1]|{al}~{v}|n{W.nlab(n)}", test_id="XI-d", item=f"EP:{lab}", n=int(n),
                         kind="ep", hypothesis=False, **common)
                if r is not None:
                    rows.append(r)
            for part in ("W", "I"):
                for m, lr, lam in (("R1", LO, LAM_CV), ("D0", HI, 1.0)) + OTHERS:
                    gA, gB = W.gk(m, n, lr, lam), W.gk("P1", n, "none")
                    lab = W._plab(m, lr, lam)
                    r = _row(part_stat(tms, al, v, part, gA, gB), nm_tot, f"{lab}-P1[{part}]|{al}~{v}|n{W.nlab(n)}", test_id="XI-d",
                             item=f"Δ_{part}:{lab}", n=int(n), kind=part, hypothesis=False, **common)
                    if r is not None:
                        rows.append(r)
            XB.boot_cache_clear()
        ok = bool(fr.get("extrap_test", False))
        for r in rows[n0:]:
            apply_extrap_rule(r, ok)
    return rows


def effective_verdict(row, extrap_ok):
    """주 판정에 쓰는 판정: 사전 규칙 미충족이면 '외삽 시험 아님', 행이 없으면 '행 없음', 그 밖은 4분 판정."""
    if not extrap_ok:
        return NOT_EXTRAP
    if row is None:
        return "행 없음"
    return str(row.get("verdict4", "판정 불가"))


def ni_status(row, extrap_ok, margin=XB.NI_MARGIN):
    """XI-c 의 n 별 비열등 상태(h54._ni_text 의 n 별 규칙과 같다). 사전 규칙 미충족이면 '외삽 시험 아님', 행이 없으면 '행 없음', 4분 판정이
    없는 행(h42.valid_row 거짓)은 '판정 불가', 두 가중 CI 상한(nanmax)이 margin 미만이면 '충족', 그 밖은 '미충족'."""
    if not extrap_ok:
        return NOT_EXTRAP
    if row is None:
        return "행 없음"
    if not X.valid_row(row):
        return "판정 불가"
    his = np.array([float(row.get("ci_hi", np.nan)), float(row.get("ci_hi_beq", np.nan))], float)
    hi = float(np.max(his[np.isfinite(his)])) if np.isfinite(his).any() else np.nan
    return NI_OK if np.isfinite(hi) and hi < float(margin) else NI_NO


def weaker(v1, v2):
    """두 4분 판정 가운데 약한 쪽(STRENGTH 순서). 같은 순위의 반대 방향(우세 대 열세)은 미결정."""
    if v1 == v2:
        return v1
    r1, r2 = STRENGTH.get(v1, -1), STRENGTH.get(v2, -1)
    if r1 == r2:
        return "미결정" if {v1, v2} == {"우세", "열세"} else (v1 if r1 < 0 else "미결정")
    return v1 if r1 < r2 else v2


def weaker_ni(v1, v2):
    """두 비열등 상태 가운데 약한 쪽(NI_STRENGTH: 충족 > 미충족 > 판정 불가·행 없음 > 외삽 시험 아님). 같은 순위이면 앞쪽(warm_trim)."""
    if v1 == v2:
        return v1
    return v1 if NI_STRENGTH.get(v1, -1) <= NI_STRENGTH.get(v2, -1) else v2


def hyp_class(txt):
    """가설 수준 판정 문구(rule3, _ni_text, 외삽 시험 아님)의 갈래 이름. HYP_PREFIX 에 없으면 문구 그대로."""
    t = str(txt)
    for p, _ in HYP_PREFIX:
        if t.startswith(p):
            return p
    return t


def hyp_rank(txt):
    return dict(HYP_PREFIX).get(hyp_class(txt), -1)


def weaker_hyp(t1, t2):
    """두 가설 수준 판정 가운데 약한 쪽(지지 > 부분 지지 > 기각 > 판정 불가 > 외삽 시험 아님). 같은 순위이면 앞쪽(warm_trim) 문구."""
    return t1 if hyp_rank(t1) <= hyp_rank(t2) else t2


def _branch(v):
    return v if v in XB.VERDICTS4 else "판정 불가"


def _find(rows, key, item, n):
    for r in rows:
        if (r.get("alias"), r.get("variant")) == key and r.get("item") == item and int(r.get("n", -9)) == int(n):
            return r
    return None


def _get(r, k, dflt=np.nan):
    return r.get(k, dflt) if r is not None else dflt


def xi_hypotheses(rows, fracs):
    """가설 표(XI-a·b·c × n {100, 전량})와 가설 수준 판정, Holm(m = 6), 해석 문장. 반환 (가설 표, Holm 표, 요약 dict).
    n 별 판정: XI-a·b 는 4분 판정, XI-c 는 비열등 상태(ni_status). 두 판(warm_trim, 기본 판)의 판정을 verdict_trim·verdict_basic 에,
    약한 쪽을 verdict_sentence 에 두고 다르면 both_written. 문장 갈래는 4분 판정의 약한 쪽(verdict4_sentence, XI-c 는 W 대비)이다.
    가설 수준 판정도 두 판을 따로 내고(verdict_hypothesis, verdict_hypothesis_basic) 약한 쪽을 verdict_hypothesis_written 에 쓴다."""
    fp, fb = fracs.get(PRIMARY, {}), fracs.get(BASIC, {})
    okp, okb = bool(fp.get("extrap_test", False)), bool(fb.get("extrap_test", False))
    has_basic = BASIC in fracs                                             # 기본 판을 실행하지 않았으면(스모크 등) 주 판의 판정만 쓴다
    out, labels, ps = [], [], []
    for item in ("a", "b", "c"):
        hyp = HYP_OF_ITEM[item]
        for n in PRIMARY_NS:
            rp, rb = _find(rows, PRIMARY, item, n), _find(rows, BASIC, item, n)
            v4p = effective_verdict(rp, okp)
            v4b = effective_verdict(rb, okb) if has_basic else NOT_TESTED
            v4w = weaker(v4p, v4b) if has_basic else v4p
            if item == "c":                                                # 비열등 가설: 판정은 비열등 상태, 문장 갈래는 W 대비의 4분 판정
                vp = ni_status(rp, okp)
                vb = ni_status(rb, okb) if has_basic else NOT_TESTED
                vw = weaker_ni(vp, vb) if has_basic else vp
                d10p = ni_status(rp, okp, XB.DELTA_AUX)
                d10b = ni_status(rb, okb, XB.DELTA_AUX) if has_basic else NOT_TESTED
                p = _get(rp, "p_ni_x2")
            else:
                vp, vb, vw, d10p, d10b = v4p, v4b, v4w, "", ""
                p = _get(rp, "p_two")
            labels.append(f"{hyp}|n{W.nlab(n)}"); ps.append(p if okp else np.nan)
            src = rp if (v4w == v4p or rb is None) else rb
            out.append(dict(test_id=hyp, n=int(n), label=f"{hyp}|n{W.nlab(n)}", verdict_trim=vp, verdict_basic=vb, verdict_sentence=vw,
                            both_written=bool(has_basic and vp != vb), verdict4_trim=v4p, verdict4_basic=v4b, verdict4_sentence=v4w,
                            both_written_v4=bool(has_basic and v4p != v4b), noninf_d10_trim=d10p, noninf_d10_basic=d10b,
                            delta=_get(src, "delta"), ci_lo=_get(src, "ci_lo"), ci_hi=_get(src, "ci_hi"), ci_lo_beq=_get(src, "ci_lo_beq"),
                            ci_hi_beq=_get(src, "ci_hi_beq"), ci_side="warm_trim" if src is rp else "기본 판",
                            noninf=bool(ni_status(rp, okp) == NI_OK), noninf_d10=bool(ni_status(rp, okp, XB.DELTA_AUX) == NI_OK),
                            noninf_basic=bool(has_basic and ni_status(rb, okb) == NI_OK),
                            p_input=p, frac_trim=fp.get("frac_mean", np.nan), frac_basic=fb.get("frac_mean", np.nan),
                            width_trim=fp.get("width_mean", np.nan), width_basic=fb.get("width_mean", np.nan),
                            small_note=_get(src, "small_note", ""), limit_dependence=_get(src, "limit_dependence", ""),
                            ci_dependence=_get(src, "ci_dependence", ""), blind="맹검", design=DESIGN_LABEL))
    holm = XB.holm_table(labels, ps, m=HOLM_M)
    hp = dict(zip(holm.label, holm.p_holm))
    for r in out:
        r["p_holm"] = float(hp.get(r["label"], 1.0))
        r["sentence"] = sentence(r)
    summ = {}
    for item in ("a", "b", "c"):
        hyp = HYP_OF_ITEM[item]
        res_p = [(f"n={W.nlab(n)}", _find(rows, PRIMARY, item, n)) for n in PRIMARY_NS]
        res_b = [(f"n={W.nlab(n)}", _find(rows, BASIC, item, n)) for n in PRIMARY_NS]
        if item == "c":
            vtxt = W._ni_text(res_p, XB.NI_MARGIN) if okp else NOT_EXTRAP
            vtxt10 = W._ni_text(res_p, XB.DELTA_AUX) if okp else NOT_EXTRAP
            btxt = W._ni_text(res_b, XB.NI_MARGIN) if okb else NOT_EXTRAP
            btxt10 = W._ni_text(res_b, XB.DELTA_AUX) if okb else NOT_EXTRAP
        else:
            vtxt = XB.rule3([(k, None if r is None else r.get("verdict4")) for k, r in res_p], KIND_OF_HYP[hyp]) if okp else NOT_EXTRAP
            btxt = XB.rule3([(k, None if r is None else r.get("verdict4")) for k, r in res_b], KIND_OF_HYP[hyp]) if okb else NOT_EXTRAP
            vtxt10 = btxt10 = ""
        if not has_basic:
            btxt = btxt10 = NOT_TESTED
        wtxt = weaker_hyp(vtxt, btxt) if has_basic else vtxt
        summ[hyp] = dict(verdict=vtxt, verdict_noninf_d10=vtxt10, verdict_basic=btxt, verdict_basic_noninf_d10=btxt10, verdict_written=wtxt,
                         both_written=bool(has_basic and hyp_class(vtxt) != hyp_class(btxt)), extrap_test_trim=okp, extrap_test_basic=okb,
                         basic_run=has_basic)
    finals = [r["verdict4_sentence"] for r in out]
    overall = ALL_NA_SENTENCE if all(_branch(v) == "판정 불가" for v in finals) else " ".join(r["sentence"] for r in out)
    for r in out:
        s_ = summ[r["test_id"]]
        r.update(verdict_hypothesis=s_["verdict"], verdict_hypothesis_basic=s_["verdict_basic"], verdict_hypothesis_written=s_["verdict_written"],
                 hyp_both_written=s_["both_written"])
    return pd.DataFrame(out), holm, dict(hypotheses=summ, overall_sentence=overall)


def sentence(r):
    """해석 문장(1절 다섯 갈래, 공간 대용 표지와 외삽 폭, 보정 전 유의, 효과 크기, 두 판의 판정). 갈래는 4분 판정의 약한 쪽이다.
    XI-c 는 비열등 기준의 상태를 약한 쪽(verdict_sentence)으로 덧붙이고, 두 판의 상태가 다르면 둘 다 적는다."""
    item = {"XI-a": "a", "XI-b": "b", "XI-c": "c"}[r["test_id"]]
    v4 = r.get("verdict4_sentence", r["verdict_sentence"])
    v4t, v4b = r.get("verdict4_trim", r["verdict_trim"]), r.get("verdict4_basic", r["verdict_basic"])
    both4 = bool(r.get("both_written_v4", r.get("both_written")))
    b = _branch(v4)
    d = r.get("delta", np.nan)
    a_txt = f"{abs(float(d)):.2f}" if d is not None and np.isfinite(d) else "a"
    w_txt = f"{float(r['width_trim']):.2f}" if np.isfinite(r.get("width_trim", np.nan)) else "미상"
    extra = f"{SPACE_PROXY}, 외삽 폭 √TDD 차 {w_txt}"
    if both4 or r.get("both_written"):
        wb = f"{float(r['width_basic']):.2f}" if np.isfinite(r.get("width_basic", np.nan)) else "미상"
        extra += f"(기본 판 {wb})"
    if both4:
        extra += f", warm_trim 판 {v4t}, 기본 판 {v4b}, 약한 쪽으로 씀"
    extra += f", n {W.nlab(r['n'])}"
    tpl = {k: f"{SENT[item][k].format(a=a_txt)}({extra})" for k in SENT[item]}
    reason = ""
    if b == "판정 불가":
        reason = NOT_EXTRAP if NOT_EXTRAP in (v4t, v4b) else "CI 없음 또는 행 없음"
    txt = XB.compose_sentence(tpl, b, holm_p=r.get("p_holm"), delta=d, reason=reason)
    if item == "c":
        st = r["verdict_sentence"]
        notes = [] if st in (NI_OK, NI_NO) else [str(st)]
        if r.get("both_written"):
            notes.append(f"warm_trim 판 {r['verdict_trim']}, 기본 판 {r['verdict_basic']}, 약한 쪽으로 씀")
        body = f"{st}이다" if st in (NI_OK, NI_NO) else "판정할 수 없었다"
        txt += f" 비열등 기준(두 가중 CI 상한 < 0.5 cm)은 {body}" + (f"({'; '.join(notes)})" if notes else "") + "."
    return txt


def summarize(o, root=None, quiet=False):
    """조각 → 봉인 표. 화면에는 행 수와 sha256 만 쓴다(xbatch_core.write_sealed)."""
    t0 = time.time()
    tms, units, runs = XB.load_tms(o.SHARDS, o.TAG, o.nboot, allow_mixed=o.allow_mixed_cfg)
    if not tms:
        print(f"[summarize] 조각 없음({o.SHARDS}/{o.TAG}__*)", flush=True)
        return None
    struct = structure_table(units)
    fracs = frac_summary(struct)
    rows = xi_tests(tms, fracs)
    tests = XB.clean_rows(rows)
    hyp, holm, summ = xi_hypotheses(rows, fracs)
    rt = W.store_rmse_table({EXP_ID: tms})
    meta = dict(stage=EXP_NAME, plan=XB.PLAN_DOC, plan_commit=XB.PLAN_COMMIT, plan_revision=XB.PLAN_REVISION, tag=o.TAG, nboot=int(o.nboot),
                nboot_asked=int(o.nboot_asked), nboot_capped=bool(o.nboot < o.nboot_asked), git_commit=XB._git("rev-parse", "--short", "HEAD"),
                code_sha=XB.code_sha(__file__), code_sha_core=XB.code_sha(XB.__file__), frozen=dict(XB.FROZEN_SHA16), n_units=len(units),
                n_stores=len(tms), unit_status=dict(Counter(str(u.get("status")) for u in units)), holm_m=HOLM_M, frac_min=FRAC_MIN,
                fracs={f"{k[0]}|{k[1]}": v for k, v in fracs.items()}, primary=list(PRIMARY), basic=list(BASIC),
                hypotheses=summ["hypotheses"], overall_sentence=summ["overall_sentence"], summarize_s=round(time.time() - t0, 1),
                rules=dict(ep="h54.region_ep(W·I 저장소 h40.contrast 분포의 같은 번호 차)", verdict4="xbatch_core.verdict_cols(δ 0.5·1.0·δ_rel)",
                           xi_a_b="xbatch_core.rule3(n {100, 전량})", xi_c="h54._ni_text(두 가중 CI 상한 < 0.5 cm)",
                           frac="판마다 frac_W_outside_A 분할 평균 < 0.5 → 외삽 시험 아님(xi_tests 의 판정 열도 덮는다)",
                           weaker="구현 기록 3절(n 별 4분 판정, XI-c 비열등 상태, 가설 수준 판정)"))
    root_ = Path(o.OUT).parent if root is None else Path(root)
    XB.write_sealed(Path(o.OUT).name, {f"{o.TAG}_tests.csv": tests, f"{o.TAG}_hyp.csv": hyp, f"{o.TAG}_holm.csv": holm,
                                       f"{o.TAG}_curve.csv": rt, f"{o.TAG}_meta.json": meta}, root=root_, quiet=quiet)
    XB.check_out_dir(o.OUT)
    o.OUT.mkdir(parents=True, exist_ok=True)
    struct.to_csv(o.OUT / f"{o.TAG}_structure.csv", index=False)
    W.timing_table(units).to_csv(o.OUT / f"{o.TAG}_timing.csv", index=False)
    fcols = [c_ for c_ in ("exp", "target", "mode", "split", "variant", "method", "learner", "n", "draw", "seed", "lam", "n_nonfinite", "fit_flag")
             if c_ in runs]
    failed = runs[(runs.n_nonfinite.astype(float) > 0) | runs.fit_flag.isin(["fail", "nonfinite", "fallback"])][fcols] if len(runs) else pd.DataFrame()
    failed.to_csv(o.OUT / f"{o.TAG}_failed.csv", index=False)
    print(f"[summarize] 조각 {len(units)} · 저장소 {len(tms)} · 대비 행 {len(tests)} · 가설 행 {len(hyp)} · 실패 키 {len(failed)} · "
          f"{time.time() - t0:.0f}s → 봉인 폴더", flush=True)
    return dict(tests=tests, hyp=hyp, holm=holm, meta=meta, struct=struct)


# ================================================================ 재현 관문
def read_deps_log(path):
    """기준 작업의 패키지 판(deps.log 의 마지막 '[deps] 확인:' 줄, 예 'catboost 1.2.10 · numpy 2.4.6'). 파일이나 줄이 없으면 빈 dict."""
    p = _abs(path) if path else None
    if p is None or not p.exists():
        return {}
    for line in reversed(p.read_text(errors="replace").splitlines()):
        if "[deps] 확인:" in line:
            out = {}
            for part in line.split("확인:", 1)[1].split("·"):
                tok = part.split()
                if len(tok) >= 2:
                    out[tok[0]] = tok[1]
            return out
    return {}


def deps_text(d):
    return " · ".join(f"{k} {v}" for k, v in sorted(d.items())) if d else "기록 없음"


def units_of(shards):
    """조각 목록 → {(별칭, 변형, 분할): unit dict}(구조 열만 쓴다)."""
    out = {}
    for s_ in shards:
        try:
            u = json.loads(Path(s_["unit"]).read_text())
        except (OSError, ValueError):
            u = {}
        out[(s_["target"], s_["variant"], int(s_["split"]))] = u
    return out


def unique_deps(units):
    """unit.json 의 deps(이번 실행 패키지 판) 목록(중복 제거). 기록이 없는 조각은 빈 dict."""
    seen, out = set(), []
    for u in units:
        d = dict(u.get("deps") or {})
        k = json.dumps(d, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def gate_design_key(o):
    """설계 범위 안의 기준 키 술어(이번 실행의 n 격자와 P0 의 n 0, 추출 수 h54 규칙, CatBoost seed 수). 이 술어를 만족하는 기준 키는 이번 실행에
    모두 있어야 한다(스모크처럼 작은 설정에서도 같은 규칙으로 범위를 본다)."""
    grid = {int(v) for v in o.GRID} | {0}
    nd = max(1, min(int(W.WF10_DRAWS), int(o.draws_cap))) if int(o.draws_cap) > 0 else int(W.WF10_DRAWS)
    ns = int(o.seeds)

    def ok(k):
        return int(k[4]) in grid and int(k[5]) < nd and (int(k[6]) == -1 or 0 <= int(k[6]) < ns)
    return ok


def _cov_row(store, split, key, note):
    return dict(store=str(store), split=int(split), key=json.dumps(list(key), ensure_ascii=False) if key is not None else "",
                max_abs_dsse=np.nan, max_rel_dsse=np.nan, cnt_equal=False, ok=False, note=note)


def gate_coverage(new, ref, mine_units, ref_units, expect, required, design_key, key_fn=None):
    """재현 관문의 범위 점검(gate_compare 와 같은 열). expect = {(별칭, 변형): 기대 분할 집합}, required = 등록 관문 단위 목록.
    (1) 등록 관문 단위마다 기대 분할이 하나 이상이다. (2) 기대 분할마다 두 쪽 조각이 있고 두 unit.json 의 store_names 저장소가 두 쪽에 모두 있다.
    (3) 이번 실행의 키(key_fn 이 있으면 그 키만)는 모두 기준에 있다. (4) design_key 를 만족하는 기준 키(key_fn 적용)는 모두 이번 실행에 있다.
    어긋나는 항목마다 ok = False 행을 쓴다. 반환 (행 DataFrame, 요약 dict(단위별 비교한 분할·저장소·키 수))."""
    rows, comp = [], {}
    kf = key_fn or (lambda k: True)
    for al, v in required:
        if not expect.get((al, v)):
            rows.append(_cov_row(f"{al}~{v}", -1, None, "등록 관문 단위 없음(기대 분할 0)"))
    for (al, v), sps in sorted(expect.items()):
        c = comp.setdefault(f"{al}|{v}", dict(splits=[], stores=0, keys=0))
        for sp in sorted(sps):
            um, ur = mine_units.get((al, v, sp)), ref_units.get((al, v, sp))
            if um is None or ur is None:
                rows.append(_cov_row(f"{al}~{v}", sp, None, "이번 실행 조각 없음" if um is None else "기준 조각 없음"))
                continue
            names = list(dict.fromkeys(list(um.get("store_names") or []) + list(ur.get("store_names") or [])))
            if not names:
                rows.append(_cov_row(f"{al}~{v}", sp, None, "store_names 없음"))
                continue
            done = True
            for nm in names:
                a_, b_ = new.get((nm, sp)), ref.get((nm, sp))
                if a_ is None or b_ is None:
                    rows.append(_cov_row(nm, sp, None, "이번 실행 저장소 없음" if a_ is None else "기준 저장소 없음"))
                    done = False
                    continue
                ka = {tuple(k) for k in a_.keys if kf(k)}
                kb = {tuple(k) for k in b_.keys if kf(k)}
                for k in sorted(ka - kb, key=str):
                    rows.append(_cov_row(nm, sp, k, "기준 키 없음"))
                for k in sorted({k for k in kb if design_key(k)} - ka, key=str):
                    rows.append(_cov_row(nm, sp, k, "이번 실행 키 없음(설계 범위 안 기준 키)"))
                c["stores"] += 1
                c["keys"] += len(ka & kb)
            if done:
                c["splits"].append(int(sp))
    cols = ["store", "split", "key", "max_abs_dsse", "max_rel_dsse", "cnt_equal", "ok", "note"]
    return pd.DataFrame(rows, columns=cols), comp


def gate(o):
    """v3 판(Alaska, Canada) 조각의 블록 SSE 를 Vppbeb 조각과 대조한다(1절 허용 오차). 범위 점검(gate_coverage)으로 등록 관문 단위(알래스카 warm,
    캐나다 v3 warm·cold)의 기대 분할·저장소·키를 모두 비교했는지 확인한다. 두 쪽 패키지 판을 표와 <tag>_gate_meta.json 에 적는다.
    표와 화면에는 키 수, SSE 차, 패키지 판만 쓴다."""
    sel = [(al, v) for al in GATE_ALIASES if al in o.VERSIONS for v in VERSIONS[al]["variants"] if v in o.VARIANTS]
    sp_ok = {int(v) for v in o.SPLITS}
    mine = [s_ for s_ in XB.find_shards(o.SHARDS, o.TAG) if (s_["target"], s_["variant"]) in sel and int(s_["split"]) in sp_ok]
    ref_dir = _abs(o.gate_ref)
    refs = [s_ for s_ in XB.find_shards(ref_dir, "wf10") if (s_["target"], s_["variant"]) in sel and int(s_["split"]) in sp_ok]
    if not mine or not refs:
        print(f"[gate] 대조할 조각이 없다(이번 실행 {len(mine)}, 기준 {len(refs)})", flush=True)
        return 1
    um, ur = units_of(mine), units_of(refs)
    expect = {}
    for (al, v, sp), u in list(um.items()) + list(ur.items()):
        e = expect.setdefault((al, v), set())
        e.add(int(sp))
        e |= {int(x) for x in (u.get("expected_splits") or []) if int(x) in sp_ok}
    required = [k for k in GATE_REQUIRED if k in sel]
    new, ref = H4.load_stores([s_["npz"] for s_ in mine]), H4.load_stores([s_["npz"] for s_ in refs])
    df = XB.gate_compare(new, ref, level=o.gate_level)
    cov, comp = gate_coverage(new, ref, um, ur, expect, required, gate_design_key(o))
    s = XB.gate_summary(df)
    n_cov = int(len(cov))
    passed = bool(s["passed"] and n_cov == 0)
    deps_ref, deps_new = read_deps_log(o.gate_ref_deps), unique_deps(um.values())
    np_ref = deps_ref.get("numpy", "")
    np_mismatch = bool(np_ref and any(d.get("numpy", "") != np_ref for d in deps_new))
    parts = [f for f in (df.assign(check="SSE"), cov.assign(check="범위")) if len(f)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(df.columns) + ["check"])
    out["level"] = o.gate_level
    out["deps_ref"] = deps_text(deps_ref)
    deps_new_txt = "; ".join(deps_text(d) for d in deps_new) if deps_new else "기록 없음"
    out["deps_new"] = deps_new_txt
    meta = dict(stage=EXP_NAME, level=o.gate_level, tol=list(XB.GATE_TOL[o.gate_level]), ref_dir=str(o.gate_ref), ref_deps_source=str(o.gate_ref_deps),
                deps_ref=deps_ref, deps_new=deps_new, numpy_mismatch=np_mismatch,
                deps_note="허용 오차는 1절 그대로 둔다. numpy 판 차이는 구현 기록 5절의 미해결 불일치다" if np_mismatch else "",
                required=[f"{al}|{v}" for al, v in required], expected={f"{al}|{v}": sorted(e) for (al, v), e in sorted(expect.items())},
                compared=comp, n_keys=int(s["n_keys"]), n_fail_sse=int(s["n_fail"]), n_fail_coverage=n_cov, passed=passed,
                plan=XB.PLAN_DOC, plan_commit=XB.PLAN_COMMIT, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    XB.check_out_dir(o.OUT)
    o.OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(o.OUT / f"{o.TAG}_gate.csv", index=False)
    (o.OUT / f"{o.TAG}_gate_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    stores = sorted({k[0] for k in new} & {k[0] for k in ref})
    print(f"[gate] 수준 {o.gate_level} · 공통 저장소 {len(stores)} · 공통 키 {s['n_keys']} · 불일치 키 {s['n_fail']} · 범위 결손 {n_cov} · "
          f"통과 {passed} → {o.OUT / (o.TAG + '_gate.csv')}", flush=True)
    print("[gate] 단위별 비교 분할: " + "; ".join(f"{k} {v['splits']}" for k, v in comp.items()), flush=True)
    print(f"[gate] 패키지 판: 기준 {deps_text(deps_ref)} · 이번 {deps_new_txt}"
          + (" · numpy 판이 다르다(허용 오차는 0 그대로, 구현 기록 5절)" if np_mismatch else ""), flush=True)
    return 0 if passed else 1


# ================================================================ 명령행
def is_local():
    """Rescale 작업 표지(WF_RESCALE=1, LG_RESCALE=1)가 없으면 로컬 실행이다(--allow-local 로 연 로컬 본 실행도 로컬이다)."""
    return not (os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1")


def local_resources(cli_argv=None):
    """1절 로컬 자원. (1) MALLOC_ARENA_MAX 가 없으면 명령행 실행(cli_argv 가 목록)은 MALLOC_ARENA_MAX=2 로 같은 명령을 다시 시작하고(os.execve,
    nice·taskset·rlimit 는 이어진다), 함수 호출(cli_argv = None)은 거부한다. glibc 할당 영역 때문에 MALLOC_ARENA_MAX 없이 시작하면 RSS 가
    수백 MB 여도 가상 크기가 약 9 GB 가 되어 10 GB 상한에서 CatBoost 가 멈춘다(2026-10-04 로컬 확인: VmPeak 9.1 GB 대 2.8 GB).
    (2) 가용 메모리가 30 GB 아래면 MEM_WAIT_S(1시간)까지 60초 간격으로 기다리고 그래도 모자라면 중단한다. (3) 주소 공간 10 GB 상한을 건다."""
    if not os.environ.get("MALLOC_ARENA_MAX", ""):
        if cli_argv is None:
            raise SystemExit("[local] MALLOC_ARENA_MAX=2 로 시작한다(1절 로컬 자원: 작업 하나 10 GB 상한)")
        print("[local] MALLOC_ARENA_MAX=2 로 다시 시작한다(1절 로컬 자원)", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        env = dict(os.environ, MALLOC_ARENA_MAX="2")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())] + [str(v) for v in cli_argv], env)
    g = XB.require_memory(MEM_MIN_GB, wait_s=MEM_WAIT_S)
    if not XB.limit_memory(MEM_CAP_GB):
        raise SystemExit(f"[local] 주소 공간 상한 {MEM_CAP_GB:g} GB 를 걸 수 없다(1절 로컬 자원)")
    return g


def main(argv=None):
    cli = argv is None
    argv = list(sys.argv[1:] if argv is None else argv)
    o = parse_args(argv)
    mode = "count" if o.count_only else ("summarize" if (o.summarize_only or o.gate) else ("smoke" if o.smoke else "run"))
    XB.guard(mode, argv)                                                   # 자료를 읽기 전에 거부한다
    if is_local():                                                         # 1절 로컬 자원(세기·스모크·집계·관문·로컬 본 실행 모두)
        local_resources(argv if cli else None)
    t0 = time.time()
    with XB.restricted_output():
        if o.threads_asked > o.threads:
            print(f"[local] 허용 표지가 없다: --threads {o.threads_asked} → {o.threads}", flush=True)
        if o.gate:
            return gate(o)
        if o.summarize_only:
            if o.nboot < o.nboot_asked:
                print(f"[local] 허용 표지가 없다: 재표집 {o.nboot_asked} → {o.nboot}(등록 횟수의 값이 아니다)", flush=True)
            return 0 if summarize(o) is not None else 1
        units, skipped, expected = enumerate_units(o)
        units = shard_select(units, o.shard)
        if o.count_only:
            count_only(o, units, skipped)
            return 0
        if not o.PERMIT and int(o.workers) > 1:
            print(f"[local] 허용 표지 없는 스모크: --workers {o.workers} → 0", flush=True)
            o.workers = 0
        rep = XB.smoke_env_report()
        if rep["warn"]:
            print(f"[local] 스모크 규약 점검: {'; '.join(rep['warn'])}", flush=True)
        todo, resumed = [], []
        for u in units:
            al, v, sp = u
            ok, why = XB.unit_state(o.SHARDS, o.TAG, al, MODE, sp, unit_cfg(o, v, data_sha(o, al)), v) if o.resume else (False, "")
            (resumed if ok else todo).append(u)
            if o.resume and not ok and why != "조각 없음":
                print(f"  [resume] 다시 실행 {unit_name(u)} ({why})", flush=True)
        print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · 건너뜀(구조) {len(skipped)} · 워커 {max(int(o.workers), 0)} · 스레드 {o.threads}",
              flush=True)
        exp_items = [(list(k), v) for k, v in expected.items()]
        done, failed = XB.execute(todo, _worker_run, workers=int(o.workers), threads=int(o.threads), init=_worker_init,
                                  init_args=(o.ARGV, exp_items))
        n_fail = len(failed) + sum(1 for d in done if d.get("status") == "failed")
        print(f"[done] 완료 {len(done)} · 실패 {n_fail} · {time.time() - t0:.0f}s · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
        if not o.no_summarize and not o.shard:
            summarize(o)
        return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
