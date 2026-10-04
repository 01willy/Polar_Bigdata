"""XJ_tempderived_aux_labels(계획 2.10): 지온 유도 라벨을 낮은 가중 w 의 보조 행으로 쓴다(우선순위 최하위, SI 한 줄).

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100) 2.10절, 1절 공통 규약, 0.3 열람 규칙, 5절 마감 표.
공용 골격: scripts/3_deep_learning/xbatch_core.py(동결 모듈 h40·h42·h54 의 sha256 대조, 조각, 봉인, 출력 제한, 판정 도구).
구현 기록(계획 해석과 구현 결정): docs/research/2026-10-04/impl_notes/x_tempderived_aux_labels.md

자료와 약관
  원천 쪽 보조 행: v3(data/processed/fidelity_base_v3.csv)의 F4_calm_temp 68행, F2_gtnp_env 37행. 두 자료의 약관 근거는 [미확인]이다.
    --aux-lic-ref 로 근거를 적은 개정 이력 문구를 주지 않으면 원천 쪽 설계는 '시험하지 않음'이다(5절 마감 표, 7.2-10). 문구는 계획 문서의
    개정 이력 절에 있고, 커밋된 판(HEAD)의 개정 이력에도 있으며, T0 판에는 없어야 한다(로컬에서 git 으로 대조한다). 계획 문서가 없는 환경
    (Rescale 묶음)에서는 로컬에서 대조하고 만든 약관 표지(--aux-lic-token, 문구·sha256·문구를 더한 계획 커밋)가 있어야 하고 문구가 표지와
    같아야 한다(없거나 다르면 중단). 표지는 '--count-only --aux-lic-ref <문구> --write-lic-token' 으로 만들고 묶음에 넣는다.
    자료원별로 --aux-src-sources 에 남긴 것만 쓴다.
  대상 쪽 보조 행: v4(data/processed/fidelity_base_v4.csv)의 F3_ext_temp 39행(Tibet_LGD, fidelity_base_v4_labels.csv 의 license 'CC BY 4.0',
    lic_unverified 0). 약관 열이 비었거나 미확인인 행은 뺀다. F2_ext_unknown(러시아 W 야말)은 쓰지 않는다.

설계(계획 2.10)
  (1) 원천 쪽: 대상 주 4지역(레나, 캐나다, 러시아 W, 러시아 E, 모드 x, 자료 v3), 분할 1–5(h40 구조 규칙), n {0, 10, 40}(|A| 미만), 추출 5
      (h40.draw_cells, n = 0 은 1), seed 0·1, catboost_lo 반복 200. 방법 D0, D1, R1(λ 0.25 판정, 0.5·1.0 도 저장). LG(h40)의 학습 행렬
      (원천 행, 선택 라벨 행, 유사라벨 행, h54.TUnit.rows_D·rows_R = h42.stack_rows)에 보조 행을 끝에 붙이고 CatBoost sample_weight 로 보조 행에
      w ∈ {0.1, 0.3, 1}, 나머지 행에 1 을 준다. w = 0 은 보조 행이 없는 LG 와 같은 키다(재현 관문). 보조 행의 목표는 D0·D1 이 y, R1 이
      y − E0·s(원천 행과 같은 E0 앵커)다. E0(P0, κ 수축의 사전값)는 직접 라벨 원천으로만 구한다(LG 와 같다). 보조 행에도 LG 원천 규칙을 적용한다:
      대상 지역(macro) 안의 보조 행과 대상 셀에서 100 km(h40 buffer_km) 안의 보조 행을 뺀다(h40.Data.source_idx 와 같은 코드 경로).
      지온 유도 표지 열 판: 학습·예측 행렬 끝에 표지 열(보조 행 1, 그 밖 0)을 더한 판을 w ∈ {0.1, 0.3, 1} 에서 함께 적합한다(서술).
  (2) 대상 쪽: Tibet_LGD(LGD 약관 확인분 실행 표 run_tables/Tibet). '39행을 A 풀의 보조 라벨로 둔다'를 문자 그대로 읽어, 분할마다 B 블록
      (채점 블록) 안의 행만 누설 방지로 빼고 A 블록과 직접 라벨이 없는 블록(other)의 F3_ext_temp 행을 A 풀의 보조 라벨로 둔다.
      직접 라벨 n ∈ {0, 3, 10}(추출 5, h40.draw_cells)일 때 P1_aux(w) = 가중 최소제곱 계수(직접 라벨 가중 1, 보조 라벨 가중 w)를 유효 라벨 수
      m = n + w·n_aux(A 풀) 로 κ 10 수축한 E·s. w = 0 이면 LG 의 P1 과 같다. 채점은 B 블록의 직접 라벨 eval_mask 셀만이다(보조 행은 채점에 없다).
  키 이름: 방법 '<방법>@w<w>'(예: R1@w0.1), 표지 열 판 '<방법>@wf<w>', 대상 쪽 'P1@w<w>'. 나머지 키 자리는 h40 과 같다.

가설(계획 2.10 표, 방향 중립 서술, 4분 판정)
  XJ-1  R1(0.25, w) − R1(0.25, w = 0), n 0·10, 주 4지역 층화 평균(xbatch_core.pool, 등록 지역 4), w ∈ {0.1, 0.3, 1}. 맹검.
  XJ-2  D1(w) − D1(w = 0), n 0, 주 4지역 층화 평균, w ∈ {0.1, 0.3, 1}. 맹검.
  XJ-3  티베트 P1_aux(w) − P1, n 3·10, 대상 행, w ∈ {0.1, 0.3, 1}. 비맹검 부분 포함(L39).
  CI 는 같은 라벨 집합 대비(h40.contrast, 보조 h42.boot_delta_common). 다중성: Holm m = 15(XJ-1 6, XJ-2 3, XJ-3 6, 보조 열, 행 없음 p 1).
  해석 문장(사전 고정): '지온 유도 보조 행(가중 w)은 라벨 k개의 전이 오차를 [a cm 줄였다 / 0.5 cm 안에서 같았다 / 차이를 확인하지 못했다 /
  a cm 늘렸다]. 지온 유도 라벨은 직접 측정과 정의가 달라(L39) 이 결과를 라벨 확충의 근거로 일반화하지 않는다.'

재현 관문(--gate, 적합 없음): w = 0 키(P0, P1, D0, D1, R1)가 LG 조각(results/rescale_lg/data/processed/lg/shards/lg__cpu__<대상>__x__s<분할>)과
  LGD 티베트 조각(data/processed/lgd/Tibet/shards/lgd__cpu__Tibet_LGD__x__s<분할>, 로컬 산출이라 물리식 키만 local_rescale 허용 오차)과 같다.
  범위 점검: 원천 쪽(시험할 때)은 주 4지역마다, 대상 쪽은 티베트의 기대 분할(이번 unit.json 의 expected_splits 와 기준 조각)마다 두 쪽 조각이
  있어야 하고, 이번 실행의 w = 0 키는 모두 기준에 있어야 하며, 설계 범위(방법·n 격자·추출·seed) 안의 기준 키는 모두 이번 실행에 있어야 한다.
  두 쪽 패키지 판(LG deps.log, 이번 unit.json 의 deps)을 표와 <tag>_gate_meta.json 에 적는다(허용 오차는 1절 그대로).

산출(<out-dir> = data/processed/xbatch/XJ_tempderived_aux_labels)
  shards/<tag>__cpu__<대상>__x__s<분할>_{runs.csv, blocksse.npz, unit.json}(원천 쪽 대상 = Lena·Canada·Russia_W·Russia_E, 대상 쪽 = Tibet_LGD).
    runs.csv 에는 키별 RMSE·편향 열을 쓰지 않고, unit.json 에는 h40.build_ctx 메타 가운데 구조 열(META_KEEP)만 넣는다(대상 라벨 통계
    E_A, E_B, E_own, y_mean, y_sd, logE 비 등은 넣지 않는다, 0.3 열람 순서).
  봉인(sealed/, 계획 0.3 열람 순서 2): <tag>_tests.csv, <tag>_hyp.csv(가설·Holm·해석 문장), <tag>_holm.csv, <tag>_curve.csv, <tag>_meta.json.
  봉인 밖(라벨 값 미사용): <tag>_count.csv, <tag>_structure.csv(셀·블록·보조 행 수), <tag>_timing.csv, <tag>_failed.csv, <tag>_gate.csv.
  묶음(payload): 4절 목록에 더해 data/processed/fidelity_base_v4.csv 와 fidelity_base_v4_labels.csv 가 필요하다(PAYLOAD_EXTRA,
    xbatch_core.PAYLOAD_INPUTS 에도 넣었다). 약관 표지(있으면)는 xbatch_core.PAYLOAD_OPTIONAL 로 들어간다.

명령행
  로컬 자원(1절): 로컬 실행은 모드와 관계없이 가용 메모리 30 GB 미만이면 최대 1시간 기다리고, 주소 공간 10 GB 상한을 건다. MALLOC_ARENA_MAX 가
  없으면 명령행 실행은 MALLOC_ARENA_MAX=2 로 다시 시작한다(함수 호출은 거부). 허용 표지 없는 스모크는 스레드 2 로 자른다(5절 1a).
  Rescale 본 실행(허용 표지, 스모크 아님)에서 대상 쪽 입력(티베트 실행 표, v4 보조 행·약관 표, v4 토양 표)이 없으면 중단한다.
  세기:   CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/x_tempderived_aux_labels.py --count-only --threads 1 [--aux-lic-ref '<문구>']
  약관 표지: 위 세기 명령에 --aux-lic-ref '<문구>' --write-lic-token 을 더한다(로컬, 계획 문서와 git 필요)
  스모크: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c <코어 4개> python3 scripts/3_deep_learning/x_tempderived_aux_labels.py \
            --smoke --threads 2 --workers 0 2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'
  본 실행: WF_RESCALE=1 python3 scripts/3_deep_learning/x_tempderived_aux_labels.py --workers 22 --threads 4 --resume --no-summarize [--aux-lic-ref '<문구>']
  분할 실행: --shard I/K. 집계: --summarize-only. 관문: --gate
"""
from __future__ import annotations

import argparse
import hashlib
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

from polar.fidelity import TARGET                                                                    # noqa: E402
from polar.m1_core import load_base, half_split_blocks                                               # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402

H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4
for _k in ("h40", "h42", "h54"):                                            # 동결 모듈은 xbatch_core 가 sha256 을 대조해 읽은 것만 쓴다
    XB.assert_frozen(_k)

# ================================================================ 고정값(계획 2.10. 바꾸면 사전 등록에서 벗어난다)
EXP_ID = "XJ"
EXP_NAME = XB.EXP_NAMES[EXP_ID]
TAG = "xj"
MODE = "x"
LO = XB.LO
LAM_BASE = XB.LAM_BASE
KAPPA = XB.KAPPA
W_GRID = (0.1, 0.3, 1.0)
SRC_GRID = (0, 10, 40)
TGT_GRID = (0, 3, 10)
DRAWS = 5
SPLITS = tuple(XB.TRANSFER_SPLITS)                                          # 1–5
SRC_TARGETS = tuple(nm.split("|")[0] for nm in XB.MAIN4)                    # Lena, Canada, Russia_W, Russia_E
TGT_ALIAS = "Tibet_LGD"
SRC_AUX = ("F4_calm_temp", "F2_gtnp_env")
TGT_AUX = ("F3_ext_temp",)
SRC_METHODS = (("D0", 1.0), ("D1", 1.0), ("R1", LAM_BASE))                 # (방법, 판정 λ)
BASE_KEYS = ("P0", "P1", "D0", "D1", "R1")                                  # w = 0 키(재현 관문)
HOLM_M = 15
FLAG_COL = "is_tempderived"
DESIGN_LABEL = "결과 열람 뒤 설계"
NOT_TESTED = "시험하지 않음"
# 묶음(4절)에 있어야 하는 입력(xbatch_core.PAYLOAD_INPUTS 에도 넣었다). 실행 표의 토양 표는 e5_soil_tdd_v4.csv 로 가는 기호 연결이다
PAYLOAD_EXTRA = ("data/processed/fidelity_base_v4.csv", "data/processed/fidelity_base_v4_labels.csv", "data/processed/e5_soil_tdd_v4.csv",
                 "data/processed/lgd/run_tables/Tibet/fidelity_base_v3.csv", "data/processed/lgd/run_tables/Tibet/e5_soil_tdd_v3.csv")
LIC_TOKEN = "data/processed/xbatch/XJ_tempderived_aux_labels/inputs/xj_aux_lic_token.json"   # 약관 표지(xbatch_core.PAYLOAD_OPTIONAL)
COUNT_COLS = ["side", "alias", "split", "n_A", "nb_A", "n_eval", "nb_eval", "fits", "est_s", "n_aux_total", "n_aux_excl_target",
              "n_aux_excl_buffer", "n_aux_used", "n_aux_A", "n_aux_B", "n_aux_other", "n_aux_pool"]
STRUCT_COLS = ["side", "alias", "split", "n_A", "nb_A", "n_eval", "nb_eval", "n_src", "n_aux_total", "n_aux_excl_target", "n_aux_excl_buffer",
               "n_aux_used", "n_aux_A", "n_aux_B", "n_aux_other", "n_aux_pool", "aux_sources", "dup_of", "valid", "n_valid_splits"]
# unit.json 에 넣는 h40.build_ctx 메타(구조 열만). 대상 라벨 통계(E_A, E_B, E_own, E_block_cv, y_mean, y_sd, logE 비)와 원천 사전 분포(tau2,
# sigma2)는 넣지 않는다(0.3 열람 순서: 봉인 밖 조각에 라벨 통계를 두지 않는다). E0 는 원천 계수로 따로 넣는다(LG 조각과 같다)
META_KEEP = ("n_A", "nb_A", "n_eval", "nb_eval", "n_src", "n_src_parent", "frac_src_parent", "n_parent_excluded", "n_buffer_excluded", "n_cells",
             "n_blocks", "smd_x25", "dup_of", "mirror_of", "valid", "n_unique_splits", "n_valid_splits", "subregion_src", "subregion_kmeans_diff",
             "alias", "data_spec", "split_status", "split_rule")
# runs.csv 에서 빼는 열(0.3 열람 순서: 키별 RMSE·편향은 봉인 밖에 두지 않는다. 집계는 blocksse.npz 의 SSE 만 쓴다)
RUN_DROP_COLS = ("rmse_cm", "rmse_beq_cm", "bias_cm")
DEPS_NAMES = ("catboost", "scikit-learn", "pandas", "scipy", "numpy")
# 1절 로컬 자원: 가용 메모리 하한 30 GB(미만이면 기다린다), 대기 상한, 작업 하나 10 GB. 5절 1a 로컬 스모크 2스레드
MEM_MIN_GB, MEM_WAIT_S, MEM_CAP_GB = 30.0, 3600.0, 10.0
SMOKE_THREADS = 2

# 가설 15개(계획 2.10 표의 순서): (가설, 방법, w, n)
HYPS = ([("XJ-1", "R1", w, n) for w in W_GRID for n in (0, 10)] + [("XJ-2", "D1", w, 0) for w in W_GRID]
        + [("XJ-3", "P1", w, n) for w in W_GRID for n in (3, 10)])
BLIND = {"XJ-1": "맹검", "XJ-2": "맹검", "XJ-3": "비맹검 부분 포함(L39)"}
SENT = {"우세": "지온 유도 보조 행(가중 {w})은 라벨 {k}개의 전이 오차를 {a} cm 줄였다",
        "동등": "지온 유도 보조 행(가중 {w})을 더해도 라벨 {k}개의 전이 오차는 0.5 cm 안에서 같았다",
        "미결정": "지온 유도 보조 행(가중 {w})이 라벨 {k}개의 전이 오차에 준 차이는 확인하지 못했다",
        "열세": "지온 유도 보조 행(가중 {w})은 라벨 {k}개의 전이 오차를 {a} cm 늘렸다",
        "판정 불가": "지온 유도 보조 행(가중 {w})이 라벨 {k}개의 전이 오차에 준 효과는 판정할 수 없었다"}
SENT_TAIL = "지온 유도 라벨은 직접 측정과 정의가 달라(L39) 이 결과를 라벨 확충의 근거로 일반화하지 않는다."


def wtxt(w) -> str:
    return f"{float(w):g}"


def mname(m, w, flag=False) -> str:
    """키의 방법 이름. w = 0 은 기준(LG 와 같은 이름), 그 밖은 '<방법>@w<w>', 표지 열 판은 '<방법>@wf<w>'."""
    if float(w) == 0.0:
        return str(m)
    return f"{m}@w{'f' if flag else ''}{wtxt(w)}"


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XJ 지온 유도 보조 라벨(계획 2.10)")
    ap.add_argument("--count-only", action="store_true", help="학습 없이 작업 단위·적합 수·셀·블록·보조 행 수만 센다")
    ap.add_argument("--smoke", action="store_true", help="작은 설정(Russia_W 원천 쪽과 Tibet_LGD 대상 쪽, 분할 1, 추출 1, seed 1)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--gate", action="store_true", help="재현 관문: w = 0 키를 LG·LGD 조각과 대조한다(적합 없음)")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--shard", default="", help="I/K: 정렬한 작업 단위 목록에서 색인 mod K = I 인 단위만 실행한다")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sides", default="src,tgt", help="src(원천 쪽), tgt(대상 쪽)")
    ap.add_argument("--src-targets", default=",".join(SRC_TARGETS))
    ap.add_argument("--aux-lic-ref", default="", help="원천 쪽 보조 행(F4_calm_temp, F2_gtnp_env)의 약관 근거를 적은 계획 개정 이력 문구. "
                                                       "없으면 원천 쪽은 '시험하지 않음'")
    ap.add_argument("--aux-src-sources", default=",".join(SRC_AUX), help="약관 근거가 있는 원천 쪽 보조 자료원(이 목록만 쓴다)")
    ap.add_argument("--aux-v4", default="data/processed/fidelity_base_v4.csv")
    ap.add_argument("--aux-v4-soil", default="e5_soil_tdd_v4.csv", help="--aux-v4 와 같은 폴더의 토양 도일 표 이름")
    ap.add_argument("--labels-v4", default="data/processed/fidelity_base_v4_labels.csv")
    ap.add_argument("--splits", type=int, default=len(SPLITS))
    ap.add_argument("--seeds", type=int, default=len(XB.SEEDS), help="CatBoost seed 수(등록 2). 시험용")
    ap.add_argument("--cb-iters", type=int, default=200, help="catboost_lo 반복 수(등록 200). 시험용")
    ap.add_argument("--draws-cap", type=int, default=0)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--out-dir", default=str(XB.out_dir(EXP_ID).relative_to(XB.ROOT)))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lgd-dir", default="data/processed/lgd/run_tables")
    ap.add_argument("--gate-ref-lg", default="results/rescale_lg/data/processed/lg/shards")
    ap.add_argument("--gate-ref-lgd", default="data/processed/lgd/Tibet/shards")
    ap.add_argument("--gate-level-lg", default="elm_hematite", choices=sorted(XB.GATE_TOL))
    ap.add_argument("--gate-level-lgd", default="local_rescale", choices=sorted(XB.GATE_TOL))
    ap.add_argument("--gate-ref-lg-deps", default="results/rescale_lg/logs_lg/deps.log", help="기준(LG) 작업의 패키지 판 기록")
    ap.add_argument("--gate-ref-lgd-deps", default="", help="기준(LGD 티베트, 로컬 산출) 패키지 판 기록(없으면 '기록 없음')")
    ap.add_argument("--aux-lic-token", default=LIC_TOKEN, help="약관 표지 파일(계획 문서가 없는 환경에서 --aux-lic-ref 를 대조한다)")
    ap.add_argument("--write-lic-token", action="store_true", help="로컬에서 --aux-lic-ref 를 개정 이력·커밋과 대조한 뒤 약관 표지를 쓴다")
    o = ap.parse_args(argv)
    o.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(o)


def _abs(p):
    return Path(p) if os.path.isabs(str(p)) else XB.ROOT / str(p)


def _list(txt):
    return [v.strip() for v in str(txt).split(",") if v.strip()]


def finalize(o):
    o.PERMIT = XB.run_permitted(o.ARGV)
    o.threads_asked = int(o.threads)
    o.threads, nb = XB.local_limits(o.threads, o.nboot, o.ARGV)
    if o.smoke and not o.PERMIT:                                   # 5절 1a 로컬 스모크(코어 4개, 2스레드). local_limits 의 상한 4 보다 좁다
        o.threads = min(int(o.threads), SMOKE_THREADS)
    o.nboot_asked, o.nboot = int(o.nboot), int(nb)
    o.SIDES = [v for v in _list(o.sides) if v in ("src", "tgt")]
    o.SRC_TARGETS = [v for v in _list(o.src_targets) if v in SRC_TARGETS]
    o.SPLITS = list(range(1, int(o.splits) + 1))
    o.SRC_GRID, o.TGT_GRID = list(SRC_GRID), list(TGT_GRID)
    o.SUFFIX = "_smoke" if o.smoke else ""
    o.SMOKE_LIC = ""
    if o.smoke:                                                    # 스모크: 두 쪽의 코드 경로를 작은 설정으로 한 번 지난다(결과는 쓰지 않는다)
        o.SIDES = ["src", "tgt"]; o.SRC_TARGETS = ["Russia_W"]; o.SPLITS = [1]; o.seeds = 1; o.draws_cap = o.draws_cap or 1
        o.SRC_GRID, o.TGT_GRID = [0, 10], [0, 3]
        o.nboot = min(int(o.nboot), 500)
        o.SMOKE_LIC = "스모크(코드 경로 점검, 결과 미사용): 원천 쪽 약관 조건을 적용하지 않는다"
    o.DRAWS_EFF = max(1, min(DRAWS, int(o.draws_cap))) if int(o.draws_cap) > 0 else DRAWS
    o.TAG = o.tag + o.SUFFIX
    o.OUT = _abs(o.out_dir); o.PROC = _abs(o.data_dir); o.LGD = _abs(o.lgd_dir)
    o.SHARDS = o.OUT / "shards"
    o.AUX_V4 = _abs(o.aux_v4); o.LABELS_V4 = _abs(o.labels_v4)
    o.ALLOW_LOCAL = "--allow-local" in o.ARGV
    o.SRC_AUX_OK = [v for v in _list(o.aux_src_sources) if v in SRC_AUX]
    o.LIC_TOKEN = _abs(o.aux_lic_token) if o.aux_lic_token else None
    o.LIC = check_aux_lic_ref(o.aux_lic_ref, token_path=o.LIC_TOKEN)
    if o.write_lic_token and o.LIC.get("method") != "local":
        raise SystemExit("[약관] --write-lic-token 은 --aux-lic-ref 를 주고 계획 문서와 git 이 있는 로컬에서만 쓴다")
    o.SRC_TESTED = bool(o.smoke or (o.LIC["ok"] and o.SRC_AUX_OK))
    if o.smoke:
        o.SRC_AUX_OK = list(SRC_AUX)
    return o


def _rev_section(txt):
    """계획 문서의 '## 개정 이력' 절(없으면 빈 문자열)."""
    txt = str(txt or "")
    i = txt.find("## 개정 이력")
    return txt[i:] if i >= 0 else ""


def lic_local_check(ref, wt_txt, head_txt, t0_txt):
    """약관 근거 문구의 로컬 대조(5절 마감 표 'R1a 제출 전', 7.2-10). (1) 작업 트리의 개정 이력 절에 있다. (2) T0 판(개정 1)에는 없다(새 개정 이력
    항목). (3) 커밋된 판(HEAD)의 개정 이력 절에도 있다(R1a 제출 전에 커밋되어 있어야 한다). 어긋나면 SystemExit."""
    if ref not in _rev_section(wt_txt):
        raise SystemExit(f"[약관] --aux-lic-ref 문구가 계획 문서의 개정 이력 절에 없다: {ref}")
    if not t0_txt:
        raise SystemExit(f"[약관] T0 판({XB.PLAN_COMMIT}:{XB.PLAN_DOC})을 읽을 수 없어 새 개정 이력 항목인지 대조할 수 없다")
    if ref in t0_txt:
        raise SystemExit("[약관] --aux-lic-ref 문구가 T0 판(개정 1)에 이미 있다. 근거를 적은 새 개정 이력 항목의 문구를 준다")
    if ref not in _rev_section(head_txt):
        raise SystemExit("[약관] --aux-lic-ref 문구가 커밋된 판(HEAD)의 개정 이력에 없다. 개정 이력을 커밋한 뒤 R1a 를 제출한다(5절 마감 표)")
    return True


def ref_sha256(ref):
    return hashlib.sha256(str(ref).encode("utf-8")).hexdigest()


def check_aux_lic_ref(ref, token_path=None, plan_path=None, git=None):
    """원천 쪽 보조 행의 약관 근거(5절 마감 표, 7.2-10). ref 가 비면 '근거 미기록'(원천 쪽 시험하지 않음).
    계획 문서와 git 이 있는 로컬: lic_local_check 로 작업 트리·HEAD·T0 판을 대조하고, 문구를 처음 더한 계획 커밋(git log -S)을 찾는다.
    계획 문서가 없는 환경(Rescale 묶음): 로컬에서 만든 약관 표지(token_path)가 있어야 하고 문구·sha256·T0 커밋이 표지와 같아야 한다.
    표지가 없거나 다르면 SystemExit(대조를 건너뛰지 않는다). 반환 dict(ok, ref, method, commit, check, ...)."""
    ref = str(ref or "").strip()
    if not ref:
        return dict(ok=False, ref="", method="", commit="", check="근거 미기록(--aux-lic-ref 없음)")
    git = git or XB._git
    plan = Path(plan_path) if plan_path is not None else XB.ROOT / XB.PLAN_DOC
    if plan.exists():
        if not git("rev-parse", "HEAD"):
            raise SystemExit("[약관] 계획 문서는 있으나 git 이 없어 개정 이력 커밋을 대조할 수 없다")
        lic_local_check(ref, plan.read_text(), git("show", f"HEAD:{XB.PLAN_DOC}"), git("show", f"{XB.PLAN_COMMIT}:{XB.PLAN_DOC}"))
        found = [c for c in git("log", "--reverse", "--format=%H", "-S", ref, "--", XB.PLAN_DOC).splitlines() if c.strip()]
        commit = found[0].strip() if found else git("rev-parse", "HEAD")
        return dict(ok=True, ref=ref, ref_sha256=ref_sha256(ref), method="local", commit=commit, plan_doc=XB.PLAN_DOC,
                    check=f"개정 이력 대조 통과(문구를 더한 계획 커밋 {commit[:12]})")
    tp = Path(token_path) if token_path else None
    if tp is None or not tp.exists():
        raise SystemExit(f"[약관] 계획 문서가 없는 환경에서는 로컬에서 만든 약관 표지가 있어야 한다({tp}). 로컬에서 '--count-only --aux-lic-ref "
                         "<문구> --write-lic-token' 으로 만들고 묶음에 넣는다")
    try:
        t = json.loads(tp.read_text())
    except (OSError, ValueError):
        raise SystemExit(f"[약관] 약관 표지를 읽을 수 없다: {tp}")
    bad = [k for k, ok in (("ref", t.get("ref") == ref), ("ref_sha256", t.get("ref_sha256") == ref_sha256(ref)),
                           ("plan_commit_t0", str(t.get("plan_commit_t0", "")) == XB.PLAN_COMMIT), ("plan_doc", t.get("plan_doc") == XB.PLAN_DOC),
                           ("commit", bool(str(t.get("commit", "")).strip()))) if not ok]
    if bad:
        raise SystemExit(f"[약관] --aux-lic-ref 가 약관 표지와 맞지 않는다(어긋난 항목 {bad}): {tp}")
    commit = str(t["commit"])
    tsha = XB.sha256_file(tp)
    return dict(ok=True, ref=ref, ref_sha256=ref_sha256(ref), method="token", commit=commit, token_sha256=tsha, plan_doc=XB.PLAN_DOC,
                check=f"약관 표지 대조 통과(계획 커밋 {commit[:12]}, 표지 sha256 {tsha[:12]})")


def write_lic_token(lic, path):
    """로컬 대조를 통과한 약관 근거의 표지(문구, sha256, 계획 문서, T0 커밋, 문구를 더한 계획 커밋)를 쓴다. 묶음에 넣어 Rescale 에서 대조한다."""
    if not lic.get("ok") or lic.get("method") != "local":
        raise SystemExit("[약관] 로컬 대조를 통과한 문구만 표지로 쓴다")
    path = Path(path)
    XB.check_out_dir(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = dict(ref=lic["ref"], ref_sha256=lic["ref_sha256"], plan_doc=XB.PLAN_DOC, plan_commit_t0=XB.PLAN_COMMIT, commit=lic["commit"],
               sources=list(SRC_AUX), created=time.strftime("%Y-%m-%d %H:%M:%S"), rule="계획 5절 마감 표 'XJ 원천 보조 행 약관 근거', 7.2-10")
    path.write_text(json.dumps(tok, ensure_ascii=False, indent=1))
    return tok


_ARGS: dict = {}


def xj_args(o):
    """h54 인자(자료 v3 = --data-dir, LGD 실행 표 = --lgd-dir). 분할은 전이 분할 1–5(LG 와 같다)."""
    key = (str(o.PROC), str(o.LGD), int(o.threads), int(o.cb_iters), int(o.seeds), int(o.draws_cap), tuple(o.SPLITS))
    if key not in _ARGS:
        _ARGS[key] = XB.h54_args(exp="wf4", splits=o.SPLITS, threads=o.threads, cb_iters=o.cb_iters, seeds=o.seeds, nboot=o.nboot,
                                 data_dir=str(o.PROC), lgd_dir=str(o.LGD), out_dir=str(o.OUT), tag=o.TAG, draws_cap=o.draws_cap,
                                 allow_local=o.ALLOW_LOCAL)
    return _ARGS[key]


# ================================================================ 보조 행
_AUX: dict = {}


def load_aux(path, soil, sources):
    """보조 행 표(load_base(sources=…): 공변량 x25, s = e5_sqrt_tdd, y = ALT). y·s 가 유한한 행만 남긴다(h40.Ctx 의 원천 행 규칙)."""
    path = Path(path)
    key = (str(path), str(soil), tuple(sources))
    if key not in _AUX:
        df = load_base(path.parent, base=path.name, soil=str(soil), sources=tuple(sources))
        df = df.assign(s=df.e5_sqrt_tdd.values.astype(float), y=df[TARGET].values.astype(float))
        ok = np.isfinite(df.y.values) & np.isfinite(df.s.values)
        _AUX[key] = df[ok].reset_index(drop=True)
    return _AUX[key]


def aux_licence(df, labels_path):
    """대상 쪽 보조 행의 약관: fidelity_base_v4_labels.csv 의 license 가 비어 있지 않고 lic_unverified = 0 인 loc_id 만 남긴다.
    반환 (남긴 표, dict(n_in, n_ok, licenses))."""
    lab = pd.read_csv(labels_path, usecols=["loc_id", "license", "lic_unverified"], low_memory=False)
    m = lab[lab.loc_id.isin(set(df.loc_id))]
    good = m[(m.lic_unverified.fillna(1).astype(float) == 0) & m.license.notna() & (m.license.astype(str).str.strip() != "")]
    out = df[df.loc_id.isin(set(good.loc_id))].reset_index(drop=True)
    return out, dict(n_in=int(len(df)), n_ok=int(len(out)), licenses=sorted(set(good.license.astype(str))))


def aux_source_mask(D, lat, lon, macro, target, mode=MODE):
    """보조 행에 LG 원천 규칙(h40.Data.source_idx 와 같은 순서와 코드)을 적용한다. (1) 대상 지역(macro) 안의 보조 행을 뺀다(대상 셀 제외).
    (2) 대상 셀에서 buffer_km(100 km) 안의 보조 행을 뺀다(위도 1° 사전 걸러냄 뒤 haversine 최솟값). 원천 쪽 대상은 macro 지역만이다.
    반환 (남길 행 마스크, 구성 dict)."""
    if target not in D.macros:
        raise ValueError(f"XJ 원천 쪽 대상은 macro 지역만이다: {target}")
    if mode not in H.valid_modes(target):
        raise ValueError(f"{target} 는 모드 {H.valid_modes(target)} 만 쓴다")
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    t_idx = D.target_idx(target)
    keep = np.asarray(macro).astype(str) != str(target)
    n_t = int((~keep).sum())
    bk = float(D.args.buffer_km)
    la, lo = D.df.lat.values, D.df.lon.values
    tl, tn = la[t_idx], lo[t_idx]
    n_b = 0
    if bk > 0 and len(t_idx):
        for j in np.where(keep)[0]:
            if abs(lat[j] - tl).min() < 1.0 and haversine_km(lat[j], lon[j], tl, tn).min() < bk:
                keep[j] = False
                n_b += 1
    return keep, dict(n_aux_total=int(len(keep)), n_aux_excl_target=n_t, n_aux_excl_buffer=int(n_b), n_aux_used=int(keep.sum()))


def src_aux_for(o, a, alias):
    """원천 쪽 보조 행(대상별 원천 규칙 적용 뒤). 반환 (dict(X, y, s, loc_id), 구성 dict)."""
    srcs = tuple(o.SRC_AUX_OK)
    if not srcs:
        return dict(X=np.zeros((0, len(XB.FEATS)), np.float32), y=np.zeros(0), s=np.zeros(0), loc_id=np.zeros(0, int)), dict(
            n_aux_total=0, n_aux_excl_target=0, n_aux_excl_buffer=0, n_aux_used=0, aux_sources="")
    _, D = W.get_data(a)
    df = load_aux(a.PROC / "fidelity_base_v3.csv", "e5_soil_tdd_v3.csv", srcs)
    keep, comp = aux_source_mask(D, df.lat.values, df.lon.values, df.macro.values, alias)
    q = df[keep]
    comp.update(aux_sources=",".join(srcs), n_aux_by_source=json.dumps(dict(Counter(q.source_id.astype(str))), ensure_ascii=False))
    return dict(X=q[XB.FEATS].values.astype(np.float32), y=q.y.values.astype(float), s=q.s.values.astype(float), loc_id=q.loc_id.values), comp


def tgt_aux_all(o):
    """대상 쪽 보조 행(F3_ext_temp, 약관 확인분). 반환 (표, 약관 dict)."""
    df = load_aux(o.AUX_V4, o.aux_v4_soil, TGT_AUX)
    return aux_licence(df, o.LABELS_V4)


def tgt_aux_split(D, target, split, c, df):
    """분할의 A 풀 보조 행. 계획 2.10 '지온 유도 39행을 A 풀의 보조 라벨로 둔다'를 문자 그대로 읽되, B 블록(h40 half_split_blocks 의 B, 채점 블록)
    안의 보조 행은 누설 방지로 뺀다(1절 분할: B 의 eval_mask 셀은 채점 전용). A 블록 안의 행과 직접 라벨이 없는 블록(other)의 행은 B 정보가
    없으므로 A 풀에 넣는다. 반환 (A 풀 마스크, 구성 dict(n_aux_A, n_aux_B, n_aux_other, n_aux_pool = A + other))."""
    blkA = set(np.asarray(c.blkA).tolist())
    t_idx = D.target_idx(target)
    _, B_idx = half_split_blocks(D.df, t_idx, int(split))
    blkB = set(D.df.block.values[B_idx].tolist())
    ab = df.block.values
    inA = np.array([b in blkA for b in ab], bool)
    inB = np.array([b in blkB for b in ab], bool)
    pool = ~inB
    return pool, dict(n_aux_total=int(len(ab)), n_aux_A=int(inA.sum()), n_aux_B=int(inB.sum()), n_aux_other=int((~inA & ~inB).sum()),
                      n_aux_pool=int(pool.sum()), aux_sources=",".join(TGT_AUX))


def aux_coef(y_sel, s_sel, y_aux, s_aux, w, E0, kappa=KAPPA):
    """대상 쪽 P1_aux 계수. 가중 최소제곱 E = (Σ s·y + w Σ s_a·y_a)/(Σ s² + w Σ s_a²)(유한·s > 0 인 행), 유효 라벨 수 m = n + w·n_aux(A 풀) 로
    κ 수축 (m·E + κ·E0)/(m + κ). w = 0 이면 h40·h54 의 P1(E_n) 과 같은 값이다. 반환 (계수, m)."""
    y_sel, s_sel = np.asarray(y_sel, float), np.asarray(s_sel, float)
    y_aux, s_aux = np.asarray(y_aux, float), np.asarray(s_aux, float)
    d_ok = np.isfinite(y_sel) & np.isfinite(s_sel) & (s_sel > 0)
    a_ok = np.isfinite(y_aux) & np.isfinite(s_aux) & (s_aux > 0)
    w = float(w)
    meff = float(len(y_sel)) + w * float(a_ok.sum())
    if meff <= 0:
        return float(E0), 0.0
    num = float(s_sel[d_ok] @ y_sel[d_ok]) if d_ok.any() else 0.0
    den = float(s_sel[d_ok] @ s_sel[d_ok]) if d_ok.any() else 0.0
    if w > 0 and a_ok.any():
        num += w * float(s_aux[a_ok] @ y_aux[a_ok])
        den += w * float(s_aux[a_ok] @ s_aux[a_ok])
    if not den > 0 or not np.isfinite(num / den):
        return float(E0), meff
    E = num / den
    return float((meff * E + float(kappa) * float(E0)) / (meff + float(kappa))), meff


# ================================================================ 단위
class XJSrcUnit(W.TUnit):
    """원천 쪽 단위(h54.TUnit 의 문맥·추출·행렬·저장). 기준(w = 0) 키는 LG 와 같은 학습 행렬이고, w > 0 은 같은 행렬 끝에 보조 행을 붙여
    sample_weight w 로 적합한다. 표지 열 판은 학습·예측 행렬 끝에 표지 열을 더한다."""

    def __init__(self, a, c, alias, aux, dry=False, ws=W_GRID, grid=SRC_GRID, draws=DRAWS, flag=True):
        super().__init__(a, c, alias, dry, exp="xj", variant="src")
        self.aux = dict(X=np.asarray(aux["X"], np.float32).reshape(-1, c.XA.shape[1]), y=np.asarray(aux["y"], float), s=np.asarray(aux["s"], float))
        self.naux = int(len(self.aux["y"]))
        self.raux = self.aux["y"] - float(c.E0) * self.aux["s"]            # R1 의 보조 행 목표(원천 행과 같은 E0 앵커)
        self.ws, self.grid, self.draws, self.flag = tuple(float(w) for w in ws), [int(v) for v in grid], int(draws), bool(flag)
        self.XB1 = np.hstack([np.asarray(c.XB, np.float32), np.zeros((len(c.XB), 1), np.float32)])

    def variants(self):
        """(w, 표지 열) 목록. 보조 행이 없으면 기준만."""
        if self.naux == 0:
            return [(0.0, False)]
        return [(0.0, False)] + [(w, False) for w in self.ws] + ([(w, True) for w in self.ws] if self.flag else [])

    def with_aux(self, build, kind, w, flag):
        """학습 행렬 함수 build() → (X, y, None) 에 보조 행을 붙인다. w = 0 이면 build 그대로(LG 와 같은 행렬, 가중 없음)."""
        if float(w) == 0.0:
            return build

        def b():
            Xb, yb, wb = build()
            n0 = len(yb)
            Xn = np.vstack([np.asarray(Xb, np.float32), self.aux["X"]])
            if flag:
                Xn = np.hstack([Xn, np.r_[np.zeros(n0), np.ones(self.naux)].astype(np.float32)[:, None]])
            yn = np.r_[np.asarray(yb, float), self.aux["y"] if kind == "D" else self.raux]
            wt = np.r_[np.ones(n0) if wb is None else np.asarray(wb, float), np.full(self.naux, float(w))]
            return Xn, yn, wt
        return b

    def run(self):
        c = self.c
        for n, d in H.cells_of(self.grid, self.draws, self.nA):
            sel = self.draw(n, d)
            nl, nb = len(sel), self.nb(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            E1, _ = self.coefs(sel)
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            for seed in self.seeds:
                for w, fl in self.variants():
                    XBp = self.XB1 if fl else c.XB
                    na = self.naux if w > 0 else 0
                    m0, m1, mr = mname("D0", w, fl), mname("D1", w, fl), mname("R1", w, fl)
                    nr = self.nsrc + nl + na
                    (p,) = self.fit(LO, m0, self.with_aux(lambda: self.rows_D(sel), "D", w, fl), nr, seed, [XBp], n, d, sel)
                    self.add(m0, LO, "cell", n, d, seed, 1.0, p, np.nan, nl, nb, flag=self.F.last_flag, nrow=nr)
                    (p,) = self.fit(LO, m1, self.with_aux(lambda: self.rows_D(sel, seed, E1), "D", w, fl), nr + self.n_ps, seed, [XBp], n, d, sel)
                    self.add(m1, LO, "cell", n, d, seed, 1.0, p, E1, nl, nb, flag=self.F.last_flag, nrow=nr + self.n_ps)
                    (g,) = self.fit(LO, mr, self.with_aux(lambda: self.rows_R(sel, a1A), "R", w, fl), nr, seed, [XBp], n, d, sel)
                    self.emit(mr, LO, n, d, seed, a1B, g, E1, nl, nb, flag=self.F.last_flag, nrow=nr)
        return self


class XJTgtUnit(W.TUnit):
    """대상 쪽 단위(물리식만, 적합 없음). P1(LG 와 같다)과 P1_aux(w)를 B 블록 직접 라벨 채점 셀에서 저장한다."""

    def __init__(self, a, c, alias, auxA, dry=False, ws=W_GRID, grid=TGT_GRID, draws=DRAWS):
        super().__init__(a, c, alias, dry, exp="xj", variant="tgt")
        self.auxA = dict(y=np.asarray(auxA["y"], float), s=np.asarray(auxA["s"], float))
        self.ws, self.grid, self.draws = tuple(float(w) for w in ws), [int(v) for v in grid], int(draws)

    def run(self):
        c = self.c
        for n, d in H.cells_of(self.grid, self.draws, self.nA):
            sel = self.draw(n, d)
            nl, nb = len(sel), self.nb(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            E1, _ = self.coefs(sel)
            self.add("P1", "none", "cell", n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
            for w in self.ws:
                Ew, meff = aux_coef(c.yA[sel], c.sA[sel], self.auxA["y"], self.auxA["s"], w, c.E0, KAPPA)
                self.add(mname("P1", w), "none", "cell", n, d, -1, 0.0, Ew * c.sB, Ew, nl, nb, sel_info=f"m_eff={meff:g};n_aux={len(self.auxA['y'])}")
        return self


# ================================================================ 작업 단위
def _plain(meta):
    return {k: v for k, v in meta.items() if not isinstance(v, (dict, list))}


def public_meta(meta):
    """unit.json 에 넣는 문맥 메타: META_KEEP 의 구조 열만(대상 라벨 통계와 원천 사전 분포는 뺀다)."""
    return {k: v for k, v in _plain(meta).items() if k in META_KEEP}


def public_rows(rows):
    """조각 runs.csv 에 쓸 행: 키별 RMSE·편향 열(RUN_DROP_COLS)을 뺀다. 실패 표(n_nonfinite, fit_flag)와 시간 표에 쓰는 열은 남는다."""
    return [{k: v for k, v in r.items() if k not in RUN_DROP_COLS} for r in rows]


def data_sha(o, a, side, alias):
    """자료 판 식별. 원천 쪽 = v3(기본 표·토양 표·하위 지역 대응표, 보조 행도 같은 표), 대상 쪽 = Tibet 실행 표 + v4 보조 행 표·토양 표·약관 표."""
    if side == "src":
        return XB.data_sha(a, alias)
    return (XB.data_sha(a, alias) + f":{W.file_sha(o.AUX_V4)}:{W.file_sha(o.AUX_V4.parent / o.aux_v4_soil)}:{W.file_sha(o.LABELS_V4)}")


def unit_cfg(o, side, dsha):
    """결과에 영향을 주는 설정. 두 쪽이 같은 설정을 쓰고(공통 해시 하나) 쪽 표지는 variant 자리에 둔다(공통 해시에서 빠진다)."""
    return XB.make_unit_cfg(EXP_ID, variant=side, data_sha=dsha, ws=list(W_GRID), flag_variant=True, flag_col=FLAG_COL, src_grid=list(o.SRC_GRID),
                            tgt_grid=list(o.TGT_GRID), draws=DRAWS, draws_cap=int(o.draws_cap), seeds=list(range(int(o.seeds))), cb_iters=int(o.cb_iters),
                            src_methods=[m for m, _ in SRC_METHODS], src_tested=bool(o.SRC_TESTED), src_aux=sorted(o.SRC_AUX_OK) if o.SRC_TESTED else [],
                            tgt_aux=list(TGT_AUX), aux_lic_ref=o.LIC.get("ref", ""), smoke_lic=o.SMOKE_LIC, buffer_km=100.0, aux_rows="append_last",
                            aux_anchor="E0", aux_E0="direct_only", tgt_coef="wls, m = n + w*n_aux(A pool), kappa 10",
                            tgt_aux_pool="not_B_blocks(A_blocks + other)")


def build_unit(o, side, alias, split, dry=False):
    a = xj_args(o)
    c = XB.build_tctx(a, alias, MODE, int(split))
    if side == "src":
        aux, comp = src_aux_for(o, a, alias)
        U = XJSrcUnit(a, c, alias, aux, dry, grid=o.SRC_GRID, draws=o.DRAWS_EFF)
    else:
        spec = XB.resolve(alias)[0]
        _, D = W.get_data(a, spec)
        df, lic = tgt_aux_all(o)
        pool, comp = tgt_aux_split(D, XB.resolve(alias)[1], split, c, df)
        comp.update(n_aux_licence_ok=lic["n_ok"], n_aux_licence_in=lic["n_in"], aux_licenses=";".join(lic["licenses"]))
        U = XJTgtUnit(a, c, alias, dict(y=df.y.values[pool], s=df.s.values[pool]), dry, grid=o.TGT_GRID, draws=o.DRAWS_EFF)
    return a, c, U, comp


def count_row(side, alias, split, c, stats, comp):
    r = dict(side=side, alias=alias, split=int(split), n_A=int(len(c.yA)), nb_A=int(len(np.unique(c.blkA))), n_eval=int(len(c.yB)),
             nb_eval=int(len(np.unique(c.blkB))), fits=int(sum(stats["n_fit"].values())), est_s=round(float(sum(stats["est_detail"].values())), 2))
    for k in COUNT_COLS:
        if k not in r:
            r[k] = int(comp.get(k, 0)) if comp.get(k) is not None else 0
    return {k: r[k] for k in COUNT_COLS}


def run_unit(o, side, alias, split, dry=False, expected=None):
    t0 = time.time()
    a, c, U, comp = build_unit(o, side, alias, split, dry)
    U.run()
    rows, st, stats = U.finish()
    if dry:
        return count_row(side, alias, split, c, stats, comp)
    cfg = unit_cfg(o, side, data_sha(o, a, side, alias))
    unit = {**stats, **public_meta(c.meta), **comp}
    unit.update(exp=EXP_ID, side=side, alias=alias, elapsed_s=round(time.time() - t0, 1), n_fit_total=int(sum(stats["n_fit"].values())),
                n_A=int(len(c.yA)), n_eval=int(len(c.yB)), E0=float(c.E0), n_src=int(len(c.y_src)), aux_lic=o.LIC.get("check", ""),
                aux_lic_commit=o.LIC.get("commit", ""), deps=XB.deps_versions(DEPS_NAMES))
    return XB.write_shard(o.SHARDS, o.TAG, alias, MODE, int(split), public_rows(rows), [st], cfg, unit=unit, variant="",
                          expected=expected if expected is not None else [int(split)], code_file=__file__)


def tgt_inputs(o, a):
    """대상 쪽 입력 목록(티베트 실행 표, 그 토양 표 기호 연결, v4 보조 행 표·토양 표, v4 약관 표)과 없는 것."""
    spec = XB.resolve(TGT_ALIAS)[0]
    need = [a.LGD / spec / "fidelity_base_v3.csv", a.LGD / spec / "e5_soil_tdd_v3.csv", o.AUX_V4, o.AUX_V4.parent / o.aux_v4_soil, o.LABELS_V4]
    return need, [str(p) for p in need if not Path(p).exists()]


def require_inputs(o, missing, what):
    """허용 표지가 있는 본 실행(스모크 아님)에서 등록 입력이 없으면 중단한다(묶음에서 빠진 입력이 조용히 건너뛰어지는 것을 막는다)."""
    if missing and o.PERMIT and not o.smoke:
        raise SystemExit(f"[입력] {what} 입력이 없다: {missing}(묶음 목록 xbatch_core.PAYLOAD_INPUTS, 이 모듈의 PAYLOAD_EXTRA 를 확인한다)")
    return missing


def enumerate_units(o):
    """(쪽, 대상, 분할) 목록, 건너뛴 기록, 기대 분할. 분할 구조는 h40 규칙(h54.split_plan: 중복·채점 블록 2 미만 제외, LG 와 같다).
    허용 표지가 있는 본 실행에서 대상 쪽 입력이 없으면 require_inputs 가 중단한다."""
    units, skipped, expected = [], [], {}
    a = xj_args(o)
    if "src" in o.SIDES:
        if not o.SRC_TESTED:
            skipped.append(dict(side="src", alias="*", split=-1, status=f"{NOT_TESTED}({o.LIC['check']})"))
        else:
            _, D = W.get_data(a)
            for t in o.SRC_TARGETS:
                keep, skip, _ = W.split_plan(D, t, a.SPLITS)
                expected[("src", t)] = [int(s_) for s_ in keep]
                skipped += [dict(side="src", alias=t, split=int(sp), status=st_) for sp, st_, _ in skip]
                units += [("src", t, int(sp)) for sp in keep]
    if "tgt" in o.SIDES:
        spec, tgt, _ = XB.resolve(TGT_ALIAS)
        _, miss = tgt_inputs(o, a)
        if require_inputs(o, miss, "대상 쪽(XJ-3)"):
            skipped.append(dict(side="tgt", alias=TGT_ALIAS, split=-1, status="실행 표 또는 v4 보조 행·약관 표 없음"))
        else:
            _, D = W.get_data(a, spec)
            keep, skip, _ = W.split_plan(D, tgt, a.SPLITS)
            expected[("tgt", TGT_ALIAS)] = [int(s_) for s_ in keep]
            skipped += [dict(side="tgt", alias=TGT_ALIAS, split=int(sp), status=st_) for sp, st_, _ in skip]
            units += [("tgt", TGT_ALIAS, int(sp)) for sp in keep]
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


_STATE: dict = {}


def _worker_init(argv, expected_items):
    _STATE["o"] = parse_args(list(argv))
    _STATE["expected"] = {tuple(k): v for k, v in expected_items}


def _worker_run(side, alias, split):
    o = _STATE["o"]
    u = run_unit(o, side, alias, int(split), expected=_STATE["expected"].get((side, alias)))
    return dict(side=side, alias=alias, split=int(split), status=str(u.get("status", "")))


def count_only(o, units, skipped, unit_fn=None, write=True):
    """세기(라벨 값 미사용 범주): dry 실행으로 적합 수, 셀·블록 수, 보조 행 수를 센다. 화면과 표에는 COUNT_COLS 만 쓴다."""
    fn = unit_fn or (lambda u: run_unit(o, *u, dry=True))
    rows = [{k: r.get(k) for k in COUNT_COLS} for r in (fn(u) for u in units)]
    df = pd.DataFrame(rows, columns=COUNT_COLS)
    if write and len(df):
        XB.check_out_dir(o.OUT)
        o.OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(o.OUT / f"{o.TAG}_count.csv", index=False)
    print(f"[count-only] 작업 단위 {len(units)} · 건너뜀 {len(skipped)} · 분할 {o.SPLITS} · seed {o.seeds} · 추출 {o.DRAWS_EFF} · "
          f"원천 쪽 {'실행' if o.SRC_TESTED else NOT_TESTED}({o.LIC['check']}) · 원천 보조 자료원 {o.SRC_AUX_OK if o.SRC_TESTED else []}", flush=True)
    if len(df):
        g = df.groupby(["side", "alias"], sort=False).agg(units=("split", "size"), fits=("fits", "sum"), est_h=("est_s", "sum"),
                                                           nA_min=("n_A", "min"), nA_max=("n_A", "max"), nb_eval_min=("nb_eval", "min"),
                                                           aux_used=("n_aux_used", "max"), aux_excl_target=("n_aux_excl_target", "max"),
                                                           aux_excl_buffer=("n_aux_excl_buffer", "max"), aux_A_min=("n_aux_A", "min"),
                                                           aux_A_max=("n_aux_A", "max"), aux_B_max=("n_aux_B", "max"),
                                                           aux_pool_min=("n_aux_pool", "min"), aux_pool_max=("n_aux_pool", "max")).reset_index()
        g["est_h"] = (g.est_h / 3600).round(3)
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
    rows = []
    for u in units:
        r = {k: u.get(k, np.nan) for k in STRUCT_COLS}
        r["alias"] = u.get("target", u.get("alias"))
        rows.append(r)
    df = pd.DataFrame(rows, columns=STRUCT_COLS)
    return df.sort_values(["side", "alias", "split"]).reset_index(drop=True) if len(df) else df


def xj_tests(tms):
    """모든 대비 행. 원천 쪽 = 주 4지역 지역 행과 층화 평균 행(등록 지역 4), 대상 쪽 = 티베트 지역 행. 가설 행은 hypothesis = True."""
    rows = []
    for w in W_GRID:
        for fl in (False, True):
            for m, lam in SRC_METHODS:
                for n in SRC_GRID:
                    gA, gB = W.gk(mname(m, w, fl), n, LO, lam), W.gk(m, n, LO, lam)
                    if not any(nm in tms for nm in XB.MAIN4):
                        continue
                    rr, _ = XB.contrast_pool(tms, XB.MAIN4, gA, gB, label=f"{mname(m, w, fl)}-{m}|n{W.nlab(n)}", kind="same", registered=len(XB.MAIN4))
                    hyp = "XJ-1" if (m == "R1" and not fl and n in (0, 10)) else ("XJ-2" if (m == "D1" and not fl and n == 0) else "XJ-d")
                    for r in rr:
                        r.update(test_id=hyp, side="src", method=m, w=float(w), flag=bool(fl), n=int(n),
                                 hypothesis=bool(hyp != "XJ-d" and r.get("scope") == "MEAN"), blind=BLIND.get(hyp, "맹검"), design=DESIGN_LABEL)
                    rows += rr
    nm = f"{TGT_ALIAS}|{MODE}"
    tm = tms.get(nm)
    if tm is not None:
        for w in W_GRID:
            for n in TGT_GRID:
                s = XB.region_stat_same(tm, W.gk(mname("P1", w), n, "none"), W.gk("P1", n, "none"))
                if s is None:
                    continue
                rr = [r for r in XB.pool({nm: s}, [nm], f"{mname('P1', w)}-P1|n{W.nlab(n)}", registered=1) if r.get("scope") == "region"]
                hyp = "XJ-3" if n in (3, 10) else "XJ-d"
                for r in rr:
                    r.update(test_id=hyp, side="tgt", method="P1", w=float(w), flag=False, n=int(n),
                             hypothesis=bool(hyp == "XJ-3" and r.get("scope") == "region"), blind=BLIND.get(hyp, "비맹검 부분 포함(L39)"),
                             design=DESIGN_LABEL)
                rows += rr
    XB.boot_cache_clear()
    return rows


def _hyp_row(rows, hyp, m, w, n):
    want_scope = "MEAN" if hyp in ("XJ-1", "XJ-2") else "region"
    for r in rows:
        if (r.get("test_id") == hyp and r.get("method") == m and float(r.get("w", -1)) == float(w) and not r.get("flag")
                and int(r.get("n", -9)) == int(n) and r.get("scope") == want_scope):
            return r
    return None


def xj_hypotheses(rows, src_tested=True):
    """가설 표 15행(계획 2.10 표 순서), Holm(m = 15, 행 없음·시험하지 않음 p 1), 해석 문장. 반환 (가설 표, Holm 표)."""
    out, labels, ps = [], [], []
    for hyp, m, w, n in HYPS:
        lab = f"{hyp}|{mname(m, w)}|n{W.nlab(n)}"
        r = _hyp_row(rows, hyp, m, w, n)
        if hyp in ("XJ-1", "XJ-2") and not src_tested:
            v, p = NOT_TESTED, np.nan
        elif r is None:
            v, p = "행 없음", np.nan
        else:
            v, p = str(r.get("verdict4", "판정 불가")), r.get("p_two", np.nan)
        labels.append(lab); ps.append(p)

        def g(k, dflt=np.nan, r=r):
            return r.get(k, dflt) if r is not None else dflt
        worse = []
        if r is not None and hyp in ("XJ-1", "XJ-2"):                    # 풀 문장 뒤 지역 열세 덧붙임(1절): 같은 대비의 지역 행
            worse = XB.worse_regions([q for q in rows if q.get("label") == r.get("label") and q.get("scope") == "region"])
        out.append(dict(test_id=hyp, label=lab, method=m, w=float(w), n=int(n), verdict4=v, delta=g("delta"), ci_lo=g("ci_lo"), ci_hi=g("ci_hi"),
                        ci_lo_beq=g("ci_lo_beq"), ci_hi_beq=g("ci_hi_beq"), verdict4_d10=g("verdict4_d10", ""), verdict4_rel=g("verdict4_rel", ""),
                        limit_dependence=g("limit_dependence", ""), ci_dependence=g("ci_dependence", ""), small_note=g("small_note", ""),
                        pool=g("pool", ""), worse_regions=g("worse_regions", ""), few_blocks=g("few_blocks", ""), p_input=p,
                        blind=BLIND[hyp], design=DESIGN_LABEL, _worse=worse))
    holm = XB.holm_table(labels, ps, m=HOLM_M)
    hp = dict(zip(holm.label, holm.p_holm))
    for r in out:
        r["p_holm"] = float(hp.get(r["label"], 1.0))
        r["sentence"] = sentence(r)
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in out]), holm


def sentence(r):
    """해석 문장(계획 2.10 의 사전 고정 문장, 1절 다섯 갈래와 보정 전 유의·효과 크기·지역 열세 덧붙임)."""
    v = r["verdict4"]
    if v == NOT_TESTED:
        return f"지온 유도 보조 행(가중 {wtxt(r['w'])})의 원천 쪽 설계는 시험하지 않았다(약관 근거 미기록)."
    b = v if v in XB.VERDICTS4 else "판정 불가"
    d = r.get("delta", np.nan)
    a_txt = f"{abs(float(d)):.2f}" if d is not None and np.isfinite(d) else "a"
    tpl = {k: s.format(w=wtxt(r["w"]), k=int(r["n"]), a=a_txt) for k, s in SENT.items()}
    reason = "" if b != "판정 불가" else ("행 없음" if v == "행 없음" else "CI 없음")
    txt = XB.compose_sentence(tpl, b, holm_p=r.get("p_holm"), delta=d, worse_regions=r.get("_worse", []), reason=reason)
    return f"{txt} {SENT_TAIL}"


def summarize(o, root=None, quiet=False):
    t0 = time.time()
    tms, units, runs = XB.load_tms(o.SHARDS, o.TAG, o.nboot, allow_mixed=o.allow_mixed_cfg)
    if not tms:
        print(f"[summarize] 조각 없음({o.SHARDS}/{o.TAG}__*)", flush=True)
        return None
    src_tested = any(u.get("side") == "src" for u in units)
    rows = xj_tests(tms)
    tests = XB.clean_rows(rows)
    hyp, holm = xj_hypotheses(rows, src_tested)
    rt = W.store_rmse_table({EXP_ID: tms})
    struct = structure_table(units)
    meta = dict(stage=EXP_NAME, plan=XB.PLAN_DOC, plan_commit=XB.PLAN_COMMIT, plan_revision=XB.PLAN_REVISION, tag=o.TAG, nboot=int(o.nboot),
                nboot_asked=int(o.nboot_asked), nboot_capped=bool(o.nboot < o.nboot_asked), git_commit=XB._git("rev-parse", "--short", "HEAD"),
                code_sha=XB.code_sha(__file__), code_sha_core=XB.code_sha(XB.__file__), frozen=dict(XB.FROZEN_SHA16), n_units=len(units),
                n_stores=len(tms), unit_status=dict(Counter(str(u.get("status")) for u in units)), holm_m=HOLM_M, src_tested=bool(src_tested),
                aux_lic=sorted({str(u.get("aux_lic", "")) for u in units}), cfg=units[0].get("cfg") if units else {},
                summarize_s=round(time.time() - t0, 1),
                rules=dict(ci="xbatch_core.region_stat_same(h40.contrast, 보조 h42.boot_delta_common)", pool="xbatch_core.pool(주 4지역, 등록 4)",
                           verdict4="xbatch_core.verdict_cols(δ 0.5·1.0·δ_rel)", holm="m = 15, 행 없음·시험하지 않음 p 1"))
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
def base_key(k) -> bool:
    """w = 0 키(LG·LGD 와 같은 이름): P0, P1, D0, D1, R1 의 α '1', 셀 무작위 배치."""
    return str(k[0]) in BASE_KEYS and str(k[2]) == "1" and str(k[3]) == "cell"


def gate_design_key(o, side):
    """설계 범위 안의 기준 키 술어(base_key 를 만족하는 키 가운데). 원천 쪽: P0, catboost_lo 의 D0·D1(λ 1.0)과 R1(λ 0.25·0.5·1.0), n ∈ 원천 격자,
    추출 < 추출 수, seed < seed 수. 대상 쪽: P0, P1(n ∈ 대상 격자, 추출 < 추출 수). 이 술어를 만족하는 기준 키는 이번 실행에 모두 있어야 한다."""
    nd, ns = int(o.DRAWS_EFF), int(o.seeds)
    if side == "src":
        grid = {int(v) for v in o.SRC_GRID}

        def ok(k):
            if not base_key(k):
                return False
            if k[0] == "P0":
                return True
            lam_ok = float(k[7]) in [float(v) for v in XB.LAMS] if k[0] == "R1" else float(k[7]) == 1.0
            return (k[0] in ("D0", "D1", "R1") and str(k[1]) == LO and lam_ok and int(k[4]) in grid and int(k[5]) < nd and 0 <= int(k[6]) < ns)
        return ok
    grid = {int(v) for v in o.TGT_GRID}

    def ok_t(k):
        if not base_key(k):
            return False
        return k[0] == "P0" or (k[0] == "P1" and int(k[4]) in grid and int(k[5]) < nd)
    return ok_t


def read_deps_log(path):
    """기준 작업의 패키지 판(deps.log 의 마지막 '[deps] 확인:' 줄). 파일이나 줄이 없으면 빈 dict."""
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


def unique_deps(units):
    seen, out = set(), []
    for u in units:
        d = dict(u.get("deps") or {})
        k = json.dumps(d, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def _cov_row(side, store, split, key, note):
    return dict(side=side, store=str(store), split=int(split), key=json.dumps(list(key), ensure_ascii=False) if key is not None else "",
                max_abs_dsse=np.nan, max_rel_dsse=np.nan, cnt_equal=False, ok=False, note=note)


def gate_side_coverage(side, new, ref, expect, required, design_key):
    """한쪽(원천 또는 대상)의 범위 점검. expect = {대상: 기대 분할 집합}, required = 기대 분할이 하나 이상 있어야 하는 대상.
    기대 분할마다 두 쪽 저장소가 있고, 이번 실행의 w = 0 키(base_key)는 모두 기준에 있으며, design_key 를 만족하는 기준 키는 모두 이번 실행에
    있어야 한다. 반환 (ok = False 행 목록, 요약 dict(대상별 비교한 분할·키 수))."""
    rows, comp = [], {}
    for al in required:
        if not expect.get(al):
            rows.append(_cov_row(side, f"{al}|{MODE}", -1, None, "등록 관문 단위 없음(기대 분할 0)"))
    for al, sps in sorted(expect.items()):
        nm = f"{al}|{MODE}"
        c = comp.setdefault(al, dict(splits=[], keys=0))
        for sp in sorted(sps):
            a_, b_ = new.get((nm, sp)), ref.get((nm, sp))
            if a_ is None or b_ is None:
                rows.append(_cov_row(side, nm, sp, None, "이번 실행 저장소 없음" if a_ is None else "기준 저장소 없음"))
                continue
            ka = {tuple(k) for k in a_.keys if base_key(k)}
            kb = {tuple(k) for k in b_.keys if base_key(k)}
            for k in sorted(ka - kb, key=str):
                rows.append(_cov_row(side, nm, sp, k, "기준 키 없음"))
            for k in sorted({k for k in kb if design_key(k)} - ka, key=str):
                rows.append(_cov_row(side, nm, sp, k, "이번 실행 키 없음(설계 범위 안 기준 키)"))
            c["splits"].append(int(sp))
            c["keys"] += len(ka & kb)
    return rows, comp


def gate(o):
    """w = 0 키의 블록 SSE 를 LG 조각(원천 쪽 주 4지역)과 LGD 티베트 조각(대상 쪽)에 대조한다(1절 허용 오차). 범위 점검(gate_side_coverage)으로
    원천 쪽(시험할 때) 주 4지역과 대상 쪽 티베트의 기대 분할·키를 모두 비교했는지 확인한다. 두 쪽 패키지 판을 표와 <tag>_gate_meta.json 에 적는다.
    표와 화면에는 키 수, SSE 차, 패키지 판만 쓴다."""
    mine = XB.find_shards(o.SHARDS, o.TAG)
    if not mine:
        print(f"[gate] 이번 실행 조각이 없다({o.SHARDS}/{o.TAG}__*)", flush=True)
        return 1
    sp_ok = {int(v) for v in o.SPLITS}
    units = {}
    for s_ in mine:
        try:
            units[(s_["target"], int(s_["split"]))] = json.loads(Path(s_["unit"]).read_text())
        except (OSError, ValueError):
            units[(s_["target"], int(s_["split"]))] = {}
    new = H4.load_stores([s_["npz"] for s_ in mine])
    lg_dir, lgd_dir = _abs(o.gate_ref_lg), _abs(o.gate_ref_lgd)
    side_of = {k: str(u.get("side") or ("tgt" if k[0] == TGT_ALIAS else "src")) for k, u in units.items()}
    src_run = any(v == "src" for v in side_of.values())
    plan = []
    if "src" in o.SIDES and (o.SRC_TESTED or src_run):
        tg = list(o.SRC_TARGETS)
        refs = {(t, sp): lg_dir / f"lg__cpu__{t}__x__s{sp}_blocksse.npz" for t in tg for sp in sorted(sp_ok)}
        plan.append(("src", tg, {k: p_ for k, p_ in refs.items() if p_.exists()}, o.gate_level_lg, read_deps_log(o.gate_ref_lg_deps), o.gate_ref_lg_deps))
    if "tgt" in o.SIDES:
        refs = {(TGT_ALIAS, sp): lgd_dir / f"lgd__cpu__{TGT_ALIAS}__x__s{sp}_blocksse.npz" for sp in sorted(sp_ok)}
        plan.append(("tgt", [TGT_ALIAS], {k: p_ for k, p_ in refs.items() if p_.exists()}, o.gate_level_lgd, read_deps_log(o.gate_ref_lgd_deps),
                     o.gate_ref_lgd_deps or "기록 없음(로컬 산출)"))
    frames, comps, n_cov, ok, deps_ref = [], {}, 0, bool(plan), {}
    deps_new = unique_deps(units.values())
    deps_new_txt = "; ".join(deps_text(d) for d in deps_new) if deps_new else "기록 없음"
    for side, tg, refs, lvl, dref, dsrc in plan:
        expect = {}
        for (t, sp), u in units.items():
            if side_of[(t, sp)] == side and t in tg:
                e = expect.setdefault(t, set())
                e.add(int(sp))
                e |= {int(x) for x in (u.get("expected_splits") or []) if int(x) in sp_ok}
        for (t, sp) in refs:
            expect.setdefault(t, set()).add(int(sp))
        nsub = {k: v for k, v in new.items() if k[0].split("|")[0] in tg}
        rsub = H4.load_stores(list(refs.values())) if refs else {}
        df = XB.gate_compare(nsub, rsub, level=lvl, key_fn=base_key) if (nsub and rsub) else pd.DataFrame(
            columns=["store", "split", "key", "max_abs_dsse", "max_rel_dsse", "cnt_equal", "ok", "note"])
        cov, comp = gate_side_coverage(side, nsub, rsub, expect, tg, gate_design_key(o, side))
        s = XB.gate_summary(df)
        n_cov += len(cov)
        ok = ok and s["passed"] and not cov
        np_ref = dref.get("numpy", "")
        np_mis = bool(np_ref and any(d.get("numpy", "") != np_ref for d in deps_new))
        deps_ref[side] = dict(deps=dref, source=str(dsrc), numpy_mismatch=np_mis)
        df = df.assign(side=side, check="SSE")
        covdf = pd.DataFrame(cov, columns=["side", "store", "split", "key", "max_abs_dsse", "max_rel_dsse", "cnt_equal", "ok", "note"]).assign(check="범위")
        parts = [x for x in (df, covdf) if len(x)]
        f = pd.concat(parts, ignore_index=True) if parts else covdf
        f["level"], f["deps_ref"], f["deps_new"] = lvl, deps_text(dref), deps_new_txt
        frames.append(f)
        comps[side] = comp
        print(f"[gate] {side}: 수준 {lvl} · 공통 키 {s['n_keys']} · 불일치 키 {s['n_fail']} · 범위 결손 {len(cov)} · 통과 {bool(s['passed'] and not cov)} · "
              f"비교 분할 " + "; ".join(f"{t} {v['splits']}" for t, v in comp.items()), flush=True)
        print(f"[gate] {side}: 패키지 판 기준 {deps_text(dref)} · 이번 {deps_new_txt}"
              + (" · numpy 판이 다르다(허용 오차는 1절 그대로, 구현 기록 6절)" if np_mis else ""), flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta = dict(stage=EXP_NAME, sides=[p_[0] for p_ in plan], levels={p_[0]: p_[3] for p_ in plan},
                tol={p_[0]: list(XB.GATE_TOL[p_[3]]) for p_ in plan}, deps_ref=deps_ref, deps_new=deps_new,
                deps_note="허용 오차는 1절 그대로 둔다. 기준과 이번 실행의 numpy 판 차이는 구현 기록 6절의 미해결 불일치다",
                compared=comps, n_fail_coverage=int(n_cov), passed=bool(ok), src_tested=bool(o.SRC_TESTED), plan=XB.PLAN_DOC,
                plan_commit=XB.PLAN_COMMIT, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    XB.check_out_dir(o.OUT)
    o.OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(o.OUT / f"{o.TAG}_gate.csv", index=False)
    (o.OUT / f"{o.TAG}_gate_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[gate] 전체 통과 {bool(ok)} → {o.OUT / (o.TAG + '_gate.csv')}", flush=True)
    return 0 if ok else 1


# ================================================================ 명령행
def is_local():
    """Rescale 작업 표지(WF_RESCALE=1, LG_RESCALE=1)가 없으면 로컬 실행이다(--allow-local 로 연 로컬 본 실행도 로컬이다)."""
    return not (os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1")


def local_resources(cli_argv=None):
    """1절 로컬 자원(XI 모듈과 같은 규칙). (1) MALLOC_ARENA_MAX 가 없으면 명령행 실행(cli_argv 가 목록)은 MALLOC_ARENA_MAX=2 로 같은 명령을 다시
    시작하고(os.execve, nice·taskset·rlimit 는 이어진다), 함수 호출(cli_argv = None)은 거부한다(glibc 할당 영역 때문에 10 GB 상한에서 CatBoost 가
    멈춘다, 2026-10-04 로컬 확인). (2) 가용 메모리가 30 GB 아래면 1시간까지 60초 간격으로 기다리고 그래도 모자라면 중단한다. (3) 주소 공간 10 GB 상한."""
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
    XB.guard(mode, argv)
    if is_local():                                                         # 1절 로컬 자원(세기·스모크·집계·관문·로컬 본 실행 모두)
        local_resources(argv if cli else None)
    t0 = time.time()
    with XB.restricted_output():
        if o.threads_asked > o.threads:
            print(f"[local] 허용 표지가 없다: --threads {o.threads_asked} → {o.threads}", flush=True)
        print(f"[약관] 원천 쪽 보조 행: {o.LIC['check']}" + (f" · {o.SMOKE_LIC}" if o.SMOKE_LIC else ""), flush=True)
        if o.write_lic_token:
            tok = write_lic_token(o.LIC, o.LIC_TOKEN)
            print(f"[약관] 표지를 썼다: {o.LIC_TOKEN}(문구 sha256 {tok['ref_sha256'][:12]}, 계획 커밋 {tok['commit'][:12]}). 묶음에 넣는다", flush=True)
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
            side, al, sp = u
            a = xj_args(o)
            ok, why = XB.unit_state(o.SHARDS, o.TAG, al, MODE, sp, unit_cfg(o, side, data_sha(o, a, side, al)), "") if o.resume else (False, "")
            (resumed if ok else todo).append(u)
            if o.resume and not ok and why != "조각 없음":
                print(f"  [resume] 다시 실행 {unit_name(u)} ({why})", flush=True)
        todo.sort(key=lambda u: (u[0] != "src", -{"Lena": 4, "Canada": 3}.get(u[1], 1), u[2]))   # 긴 단위(원천 쪽 레나·캐나다) 먼저
        print(f"[plan] 실행 {len(todo)} · 재개로 건너뜀 {len(resumed)} · 건너뜀 {len(skipped)} · 워커 {max(int(o.workers), 0)} · 스레드 {o.threads}",
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
