"""XH_validation_ladder(검증 사다리). 계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.8 의 구현(개정 1, T0 = git 2678100).

목적
  같은 모형(P1 형 PSw, R1 형 RSw, 직접 ML D0w)을 셀 무작위, 0.05° 지점, 0.5° 블록, kNNDM, 지역 홀드아웃으로 채점해 오차 사다리를 만든다
  (그림 Fig 1 패널 e). 판정어 없는 서술이다(계획 2.8 '다중성: 없음').

단(계획 2.8 표)
  W1R  셀 무작위 5겹. 대상 지역 셀(h40 target_idx, 라벨 있는 전 셀)을 h41.folds_random(seed_of('xh','W1R',지역,반복))로 나눈다
  W1S  0.05° 지점 묶음 5겹. h41.groups_s 묶음을 h41.folds_grouped(seed_of('xh','W1S',지역,반복))로 나눈다
  W1B  0.5° 블록 5겹. h41 W1 과 같은 묶음(h41.folds_grouped(블록, 5, seed_of('lgv','W',지역,반복)))이다. h41 W1 결과의 재현용으로 다시 돈다
  W1K  kNNDM 5겹. 묶음은 로컬 1c 에서 --build-knndm 으로 한 번 만들어 색인 파일(sha256)로 보낸다. Rescale 에서 다시 군집하지 않는다
       예측 영역: 알래스카 = m1 정의(ERA5-Land 0.02° 세분, 육지 ∩ MAAT < 0, 20,000점), 레나 = map_lena/lena_grid_x25_v1.csv.gz,
       캐나다 = 라벨 경계 상자 ± 1° 의 ERA5-Land 육지 ∩ MAAT < 0 격자(주, 변형 ''), ± 2°(민감도, 변형 'pm2')
  V-G  지역 홀드아웃. h41 결과(data/processed/lgx/ladder/lgv_metrics.csv)를 다시 쓴다(새 적합 없음)
  M1C  관문 (2) 전용: 알래스카 6겹 random_cell·site_0.05·block_0.5, 모형 stefan·ridge·catboost_lo·stefan_ridge_l075 를 m1 정의로 다시 돈다

  학습 = 대상 지역 안의 학습 묶음(W1 정의, 원천 행 없음), 채점 = 채점 묶음의 eval_mask 셀(h41.VData.score). 반복 3(1–3).
  모든 단에서 학습 셀과 채점 셀의 교집합이 0 임을 단언한다(W1S 는 지점 묶음, W1B 는 블록도 겹치지 않음을 단언한다).

모형(키 = (method, learner, '1', 단, −1, 반복, seed, λ). h41 models_pooled 의 W1 식과 같다)
  PSw = E·s(E = 학습 묶음 최소제곱) | D0w[학습기, seed] | RSw[학습기, seed, λ 0.25·1.0] = E·s + λ·g
  학습기 catboost_lo(주), catboost·rf(보조, h41.FitterV). seed 0·1(계획 1절 '추출·seed'. h41 W1 조각과 같다)
  알래스카의 대회 구성(연속성 확인용): D0w[ridge](직접 능형) 과 RSw[ridge, λ 0.75] = E·s + 0.75·ridge 잔차(m1 stefan_ridge_l075 식)

집계(--summarize-only, 출력은 봉인 폴더)
  셀 단위 제곱 오차의 반복·seed 평균을 블록별로 더한 값(S̄_b)과 셀 수(C_b)로 RMSE(셀 가중 = √(ΣS̄/ΣC), 블록 등가중 = 블록 RMSE 평균)를 낸다.
  CI 는 지역마다 채점 블록 재표집(h4_common.boot_weights, seed_of('xhboot', 지역), 10,000회)을 모든 단·모형이 공유하고, 3지역 층화 평균은
  m1_stats.strat 이다.
  XH-1 ΔΔ = [RMSE_K(D0) − RMSE_R(D0)] − [RMSE_K(P1) − RMSE_R(P1)], XH-2 는 P1 대신 R1(λ 0.25). XH-3 단별·지역별 RMSE, 단별 D0 − P1·R1 − P1,
  학습기 3종, 캐나다 ± 2° 민감도. 가로축 = 채점 셀에서 가장 가까운 학습 셀까지의 거리 중앙값(km).
  판정어를 쓰지 않는다. 표의 열에는 수치와 CI 만 있다.

재현 관문(계획 2.8)
  (1) W1B 를 h41 W1 조각(lgv__W1-<지역>__r<반복>__f<묶음>_pred.npz, 반복 1–3)과 키별 블록 SSE 로 대조한다. h41 이 예측을 float32 로 저장했으므로
      두 쪽 모두 float32 로 바꾼 예측에서 블록 SSE 를 만든다. 통과는 1절 허용 오차(--gate-level, 기본 로컬–Rescale |ΔSSE| ≤ 1e-4 cm² 또는
      상대 1e-9)만으로 정한다. 예측 차의 float32 ulp 는 정보 열(max_ulp, pass_ulp_only = 허용 오차는 넘고 1 ulp 이내)로만 싣는다.
      rf 키는 넣지 않는다(scikit-learn 판 의존, 계획 2.8).
  (2) M1C 를 data/processed/m1/cv_scheme_comparison.csv 와 대조한다. m1 표는 RMSE 를 소수 3자리로 반올림해 저장했으므로 반올림 단위
      (0.001 cm, 경계 뒤집힘 허용 0.0011) 안이면 통과다. 넘으면 XH 를 멈추지 않고 '대회 연속성 재현 실패(환경 미기록)'를 적고 알래스카
      연속성 행(ridge 키)만 뺀다. 집계는 관문 기록(xh_gate_meta.json)을 읽어 이 규칙을 적용하고 관문 (1)·(2) 요약을 봉인 메타
      xh_summary_meta.json 에 쓴다. 관문 기록이 없거나 조각보다 오래되면 능형 행을 남기되 '관문 2 미확인' 표지를 단다.
  kNNDM 단은 관문에 넣지 않는다. 시험이 cv_schemes.knndm 과 m1 함수의 같은 입력 같은 묶음을 확인한다.

조각(1절 '산출 경로')
  data/processed/xbatch/XH_validation_ladder/shards/xh__cpu__<지역>__<단>__s<반복>[__<변형>]_{runs.csv, blocksse.npz, unit.json} + _cells.npz
  (_cells.npz = 채점 셀 색인, loc_id, 키, 예측(float64), 묶음 번호, 최근접 학습 셀 거리). runs.csv 와 unit.json 에는 RMSE 를 쓰지 않는다.

출력 제한(계획 0.3): 스모크·세기·집계의 화면에는 수와 해시만 쓴다. 판정 표 성격의 집계 표는 봉인 폴더
  data/processed/xbatch/XH_validation_ladder/sealed/ 에 쓴다(R1b 산출, 열람 순서 2: R2a 제출 뒤 연다).

명령행
  세기:     CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 10 python3 scripts/3_deep_learning/x_validation_ladder.py --count-only
  색인(1c): nice -n 10 python3 scripts/3_deep_learning/x_validation_ladder.py --build-knndm --threads 4      # 라벨 미사용, 로컬 1회
  스모크:   OMP_NUM_THREADS=2 nice -n 10 taskset -c 100-103 python3 scripts/3_deep_learning/x_validation_ladder.py --smoke --threads 2 --workers 0
  묶음 점검: python3 scripts/3_deep_learning/x_validation_ladder.py --payload-check     # 제출 전. 색인·메타·예측 영역(PAYLOAD_EXTRA)
  본 실행:  WF_RESCALE=1 python3 scripts/3_deep_learning/x_validation_ladder.py --workers 22 --threads 4 --resume --no-summarize
            W1K 단위가 있으면 적합 전에 색인을 읽고 sha256 을 대조한다(없거나 다르면 멈춘다)
  단위 하나: WF_RESCALE=1 python3 scripts/3_deep_learning/x_validation_ladder.py --shard Canada:W1K:1:pm2 --threads 4
  관문:     python3 scripts/3_deep_learning/x_validation_ladder.py --gate-only
  집계:     OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/x_validation_ladder.py --summarize-only --allow-local   # 재표집 10,000회
            허용 표지가 없으면 재표집이 1,000회로 제한되고, 이때는 본 봉인 폴더에 쓰지 않고 멈춘다. 허용 표지가 있어도 로컬에서는 상주 메모리
            감시(10 GB)와 가용 메모리 대기(30 GB 미만이면 최대 1시간)를 건다(계획 1절 로컬 자원, 환경 변수 WF_RESCALE·LG_RESCALE 로 판단)

적합 수(--count-only 로 다시 센다): 지역 3 × 단 4 × 반복 3 × 묶음 5 × (D0w·RSw × 학습기 3 × seed 2) = 2,160 + 캐나다 pm2 180 + 알래스카 ridge 120
  + M1C 54(catboost_lo). 계획 2.8 의 '적합 약 810건'은 seed 1·단 3 기준의 근사다(구현 기록 x_validation_ladder.md).

파일 위치 이탈: 계획 2.8 은 scripts/2_evaluation/x_validation_ladder_regional.py 와 tests/test_x_ladder.py 를 적었다. 이 파일과
  tests/test_x_xg_xh.py 로 두었다(docs/research/2026-10-04/impl_notes/x_validation_ladder.md 1절). 묶음 정보(--payload-check)에도 적는다.
"""
from __future__ import annotations

import xbatch_core as XB                                                   # numpy 보다 먼저(스레드 환경 변수)

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

H = XB.H
H41 = XB.frozen("h41")
import polar.h4_common as H4                                               # noqa: E402
from polar import cv_schemes as CV                                         # noqa: E402
from polar import m1_stats as MS                                           # noqa: E402
from polar.h4_common import BlockStore, seed_of                            # noqa: E402

ROOT = XB.ROOT
EXP_ID = "XH"
NAME = XB.EXP_NAMES[EXP_ID]
TAG = "xh"
OUT = XB.out_dir(EXP_ID)
REGIONS = ("Alaska", "Lena", "Canada")
STAGES = ("W1R", "W1S", "W1B", "W1K")
REPS = (1, 2, 3)
K_FOLDS = 5
SEEDS = tuple(XB.SEEDS)
LEARNERS = ("catboost_lo", "catboost", "rf")
MAIN_LEARNER = "catboost_lo"
LAMS = (0.25, 1.0)
CONT_REGION = "Alaska"
CONT_LAM = 0.75
VARIANTS = {"Canada": ("", "pm2")}                                         # 캐나다 W1K 의 ± 2° 민감도
PAD_OF = {"": 1.0, "pm2": 2.0}
M1C = "M1C"
M1C_SEEDS = (0, 1, 2)
M1C_K = 6
M1C_SCHEMES = ("random_cell", "site_0.05", "block_0.5")
M1C_MODELS = ("stefan", "ridge", "catboost_lo", "stefan_ridge_l075")
M1C_TOL = 0.0011                                                           # m1 표의 반올림(소수 3자리) 단위 + 경계 뒤집힘
KNNDM_N_Q = 100
KNNDM_MAXP = 0.5
ALASKA_NPRED = 20000
ALASKA_RES = 0.02
LENA_GRID = ROOT / "data" / "processed" / "map_lena" / "lena_grid_x25_v1.csv.gz"
M1_TABLE = ROOT / "data" / "processed" / "m1" / "cv_scheme_comparison.csv"
H41_SHARDS = ROOT / "data" / "processed" / "lgx" / "ladder" / "shards"
H41_METRICS = ROOT / "data" / "processed" / "lgx" / "ladder" / "lgv_metrics.csv"
INDEX_NAME = "xh_knndm_index_v1.csv"
INDEX_META = "xh_knndm_meta_v1.json"
DOMAIN_NAME = "xh_pred_domain_v1.csv.gz"
BLIND = {"Alaska": "재현(비맹검)", "Lena": "맹검", "Canada": "맹검"}
DESIGN = "결과 열람 뒤 설계, 서술(판정어 없음)"
EST_RIDGE_S = 0.02                                                         # 능형 1건의 시간 가정값(세기 전용)
GATE_META = "xh_gate_meta.json"
CONT_FAIL_TXT = "대회 연속성 재현 실패(환경 미기록)"
CONT_UNCHECKED_TXT = "관문 2 미확인"
MEM_MIN_GB = 30.0                                                          # 시작 전 가용 메모리 하한(계획 1절 로컬 자원)
MEM_WAIT_S = 3600.0                                                        # 하한 아래면 로컬에서 기다리는 최대 시간(초)
RSS_LIMIT_GB = 10.0                                                        # 로컬 작업 하나의 상주 메모리 상한
_REL_OUT = OUT.relative_to(ROOT)
PAYLOAD_EXTRA = (f"{_REL_OUT}/{INDEX_NAME}", f"{_REL_OUT}/{INDEX_META}", f"{_REL_OUT}/{DOMAIN_NAME}")   # 작업 R1b 묶음의 필수 입력(1c 산출)
LOCATION_NOTE = ("파일 위치 이탈: 계획 2.8 의 scripts/2_evaluation/x_validation_ladder_regional.py, tests/test_x_ladder.py 대신 "
                 "scripts/3_deep_learning/x_validation_ladder.py 와 tests/test_x_xg_xh.py(docs/research/2026-10-04/impl_notes/x_validation_ladder.md 1절)")


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XH 검증 사다리(계획 2.8)")
    ap.add_argument("--regions", default=",".join(REGIONS))
    ap.add_argument("--stages", default=",".join(STAGES) + "," + M1C, help="단 목록. M1C 는 관문 (2) 전용(알래스카)")
    ap.add_argument("--reps", default=",".join(str(r) for r in REPS))
    ap.add_argument("--learners", default=",".join(LEARNERS))
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--workers", type=int, default=1, help="프로세스 수. 0 = 풀 없이 차례로")
    ap.add_argument("--threads", type=int, default=1, help="프로세스당 스레드. 허용 표지가 없으면 상한 4")
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--out-dir", default=str(OUT.relative_to(ROOT)))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--knndm-index", default="", help="kNNDM 색인 파일(기본 <out-dir>/xh_knndm_index_v1.csv)")
    ap.add_argument("--knndm-sha256", default="", help="색인 파일의 sha256(주면 대조한다. 비우면 메타 파일의 값과 대조한다)")
    ap.add_argument("--gate-level", default="local_rescale", choices=list(XB.GATE_TOL), help="관문 (1)의 1절 허용 오차 층")
    ap.add_argument("--shard", default="", help="단위 하나만: <지역>:<단>:<반복>[:<변형>]")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="캐나다, 단 W1R·W1B, 반복 1, seed 1(학습기 설정은 본 실행과 같다). 출력은 smoke/ 아래")
    ap.add_argument("--count-only", action="store_true", help="단위·묶음·적합 수만 센다(라벨 값 미사용)")
    ap.add_argument("--build-knndm", action="store_true", help="예측 영역과 kNNDM 색인을 만든다(로컬 1c, 라벨 미사용)")
    ap.add_argument("--gate-only", action="store_true", help="재현 관문 (1)·(2)만 계산한다")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--payload-check", action="store_true", help="작업 R1b 묶음의 필수 입력(PAYLOAD_EXTRA) 점검과 묶음 정보 기록(제출 전)")
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    a.PERMIT = XB.run_permitted(a.argv)
    a.threads, a.nboot = XB.local_limits(a.threads, a.nboot, a.argv)
    a.REGIONS = [v.strip() for v in a.regions.split(",") if v.strip()]
    a.STAGES = [v.strip() for v in a.stages.split(",") if v.strip()]
    a.REPS = [int(v) for v in a.reps.split(",") if v.strip()]
    a.LEARNERS = [v.strip() for v in a.learners.split(",") if v.strip()]
    a.SEEDS = list(SEEDS[:max(1, int(a.seeds))])
    bad = [s for s in a.STAGES if s not in STAGES + (M1C,)] + [r for r in a.REGIONS if r not in REGIONS] + \
          [lr for lr in a.LEARNERS if lr not in LEARNERS]
    if bad:
        raise SystemExit(f"알 수 없는 값: {bad}")
    a.TAG = a.tag
    if a.smoke:                                                            # 스모크: 가장 작은 지역, 두 단, 반복 1, seed 1
        a.REGIONS = ["Canada"]; a.STAGES = ["W1R", "W1B"]; a.REPS = [1]; a.SEEDS = [SEEDS[0]]
        a.TAG = a.tag + "_smoke"
        a.nboot = min(a.nboot, 200)
    a.OUT = (ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir)
    if a.smoke:
        a.OUT = a.OUT / "smoke"
    a.SHARDS = a.OUT / "shards"
    a.PROC = (ROOT / a.data_dir) if not os.path.isabs(a.data_dir) else Path(a.data_dir)
    a.INDEX = Path(a.knndm_index) if a.knndm_index else (a.OUT if not a.smoke else a.OUT.parent) / INDEX_NAME
    a.INDEX_META = a.INDEX.with_name(INDEX_META)
    a.HA41 = None
    return a


def h41_args(a):
    """h41 인자(VData, FitterV 에 넘긴다). 허용 표지가 있으면 h41 도 스레드 제한을 풀도록 --allow-local 을 넘긴다(h41 은 LG_RESCALE 만 본다)."""
    if a.HA41 is None:
        argv = ["--part", "within", "--within-regions", ",".join(REGIONS), "--threads", str(int(a.threads)), "--seeds", str(len(a.SEEDS)),
                "--learners", ",".join(LEARNERS), "--lams", ",".join(str(v) for v in LAMS), "--within-reps", str(max(REPS)),
                "--out-dir", str(a.OUT / "_h41_unused"), "--data-dir", str(a.PROC)] + (["--allow-local"] if a.PERMIT else [])
        a.HA41 = H41.parse_args(argv)
        a.HA41.threads = int(a.threads)                                    # h41 은 허용 표지가 없으면 1 로 둔다. 스모크(2스레드)에서는 이 값을 쓴다
        a.HA41.HA.threads = int(a.threads)
    return a.HA41


def get_vdata(a):
    return H41.get_vdata(h41_args(a))


def on_rescale() -> bool:
    """Rescale 작업 환경인가(환경 변수 WF_RESCALE=1 또는 LG_RESCALE=1). --allow-local 은 로컬 실행 허용 표지일 뿐이라 여기서 보지 않는다.
    로컬 자원 보호(상주 메모리 감시, 가용 메모리 대기)는 이 값이 거짓이면 언제나 건다(계획 1절 로컬 자원)."""
    return os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1"


def wait_memory(min_gb=MEM_MIN_GB, wait_s=None, poll_s=60.0) -> float:
    """시작 전 가용 메모리 확인(계획 1절: 30 GB 아래면 기다린다). 로컬은 최대 MEM_WAIT_S 초 기다리고 Rescale 은 기다리지 않는다.
    기다리기 시작할 때 한 줄을 쓴다. 그래도 모자라면 xbatch_core.require_memory 가 SystemExit 을 낸다."""
    wait = (0.0 if on_rescale() else MEM_WAIT_S) if wait_s is None else float(wait_s)
    g = XB.mem_available_gb()
    if np.isfinite(g) and g < float(min_gb) and wait > 0:
        print(f"[메모리] 가용 {g:.1f} GB 가 하한 {float(min_gb):g} GB 아래다. 최대 {wait / 60:.0f}분 기다린다(계획 1절 로컬 자원)", flush=True)
    return XB.require_memory(float(min_gb), wait_s=wait, poll_s=float(poll_s))


# ================================================================ 묶음(라벨 미사용)
def region_index(V, region):
    """대상 지역 셀(h40 target_idx, 오름차순)과 그 가운데 채점 셀(eval_mask ∩ 지역)."""
    t_idx = np.asarray(V.D.target_idx(region), np.int64)
    sc = t_idx[V.score[t_idx] & (V.macro[t_idx] == region)]
    return t_idx, sc


def load_index(a, verify=True):
    """kNNDM 색인 표. 파일 sha256 을 --knndm-sha256 또는 메타 파일의 값과 대조한다. 없으면 None."""
    if not a.INDEX.exists():
        return None, ""
    sha = XB.sha256_file(a.INDEX)
    want = a.knndm_sha256
    if not want and a.INDEX_META.exists():
        want = json.loads(a.INDEX_META.read_text()).get("index_sha256", "")
    if verify and want and sha != want:
        raise SystemExit(f"[색인] {a.INDEX.name} 의 sha256 {sha[:16]} 이 기록 {want[:16]} 과 다르다(Rescale 에서 다시 군집하지 않는다)")
    if verify and not want:
        raise SystemExit("[색인] sha256 기록이 없다(--knndm-sha256 또는 메타 파일)")
    return pd.read_csv(a.INDEX, dtype=dict(region=str, variant=str), keep_default_na=False), sha


def stage_folds(a, V, region, stage, rep, variant="", index=None):
    """단의 묶음 번호(대상 지역 셀 순서, 길이 |t_idx|). 라벨 값을 쓰지 않는다."""
    t_idx, _ = region_index(V, region)
    if stage == "W1R":
        return H41.folds_random(len(t_idx), K_FOLDS, seed_of("xh", "W1R", region, rep))
    if stage == "W1S":
        return H41.folds_grouped(H41.groups_s(V.lat[t_idx], V.lon[t_idx]), K_FOLDS, seed_of("xh", "W1S", region, rep))
    if stage == "W1B":                                                     # h41 W1 과 같은 묶음(seed_of('lgv','W',지역,반복))
        return H41.folds_grouped(V.block[t_idx], K_FOLDS, seed_of("lgv", "W", region, rep))
    if stage == "W1K":
        if index is None:
            raise SystemExit("[W1K] kNNDM 색인 파일이 없다(--build-knndm 으로 로컬에서 만든다)")
        q = index[(index.region == region) & (index.variant == str(variant)) & (index.rep.astype(int) == int(rep))]
        mp = dict(zip(q.loc_id.astype(np.int64), q.fold.astype(int)))
        lid = V.loc_id[t_idx]
        miss = int(sum(1 for v in lid if int(v) not in mp))
        if miss:
            raise SystemExit(f"[W1K] 색인에 없는 셀이 {miss}개다({region}, 변형 '{variant}', 반복 {rep})")
        return np.array([mp[int(v)] for v in lid], np.int64)
    raise ValueError(stage)


def fold_cells(V, region, f_t, fold, stage):
    """묶음 하나의 학습 셀과 채점 셀(자료 행 위치). h41.unit_cells 의 W1 규칙(학습 = 지역 내 학습 묶음 ∩ 학습 가능, 채점 = 채점 묶음의
    채점 셀)과 같다. 교집합 0 과 단별 묶음 분리를 단언한다."""
    t_idx, _ = region_index(V, region)
    held = t_idx[f_t == fold]
    te = held[V.score[held] & (V.macro[held] == region)]
    tr = t_idx[(f_t != fold) & V.trainable[t_idx]]
    assert len(np.intersect1d(tr, te)) == 0 and len(np.intersect1d(tr, held)) == 0, f"{stage} f{fold}: 학습 셀과 채점 셀이 겹친다"
    assert np.all(V.trainable[tr]) and np.all(V.score[te]), f"{stage}: 학습 셀 또는 채점 셀의 조건이 맞지 않는다"
    if stage == "W1S":
        g = H41.groups_s(V.lat, V.lon)
        assert len(np.intersect1d(g[tr], g[held])) == 0, "W1S: 같은 지점 묶음이 학습과 채점에 걸쳐 있다"
    if stage == "W1B":
        assert len(np.intersect1d(V.block[tr], V.block[held])) == 0, "W1B: 같은 블록이 학습과 채점에 걸쳐 있다"
    return tr, te


def nnd_to_train(V, te, tr):
    """채점 셀에서 가장 가까운 학습 셀까지의 거리(km, 대원 거리). 가로축(계획 2.8)."""
    if len(te) == 0 or len(tr) == 0:
        return np.full(len(te), np.nan)
    return CV.nnd_km(CV.to_rad(V.lon[te], V.lat[te]), CV.to_rad(V.lon[tr], V.lat[tr]))


# ================================================================ 모형(h41 models_pooled 의 W1 부분집합)
def models_w1x(F, TR, TE, axis, learners, seeds, lams, cont=False):
    """h41.models_pooled(sfx='w', extended=False)의 PSw, D0w, RSw 식을 그대로 옮긴 것(F1kw, V2w 는 계획 2.8 에 없어 적합하지 않는다).
    cont = True 이면 대회 구성(m1 의 ridge 와 stefan_ridge_l075 식)을 더한다. 반환 (키 (method, learner, seed, λ) → 예측, 기록)."""
    out = {}
    nt = len(TR)
    E = H.ls_E(TR.y, TR.s)
    aB = E * TE.s
    out[("PSw", "none", -1, 0.0)] = aB
    rec = dict(E=float(E), n_train=int(nt))
    r_tr = TR.y - E * TR.s
    for lr in learners:
        for seed in seeds:
            _, (p,) = F.fit(lr, axis, lambda: (TR.X, TR.y, None), seed, [TE.X], info=dict(method="D0w", n_train=nt))
            out[("D0w", lr, seed, 1.0)] = p
            _, (g,) = F.fit(lr, axis, lambda: (TR.X, r_tr, None), seed, [TE.X], info=dict(method="RSw", n_train=nt))
            for lam in lams:
                out[("RSw", lr, seed, float(lam))] = aB + float(lam) * g
    if cont:
        from polar.m1_core import fit_model
        from polar.preprocessing import fold_prep
        Xa, Xb = fold_prep(np.asarray(TR.X, np.float32), np.asarray(TE.X, np.float32), nan_native=False)
        out[("D0w", "ridge", -1, 1.0)] = np.asarray(fit_model("ridge", Xa, TR.y, Xb, 0)["pred"], float)
        g = np.asarray(fit_model("ridge", Xa, r_tr, Xb, 0)["pred"], float)
        out[("RSw", "ridge", -1, CONT_LAM)] = aB + CONT_LAM * g
    return out, rec


def full_key(k, stage, rep):
    """(method, learner, seed, λ) → 저장 키 (method, learner, '1', 단, −1, 반복, seed, λ)."""
    return (str(k[0]), str(k[1]), "1", str(stage), -1, int(rep), int(k[2]), float(k[3]))


def store_name(region, stage, variant=""):
    return f"{region}{'~' + variant if variant else ''}|{stage}"


def unit_cfg(a, index_sha):
    """모든 단위가 같은 공통 설정(단·변형·반복은 조각 이름에 있다)."""
    return XB.make_unit_cfg(EXP_ID, "", "", stages=list(STAGES), reps=list(REPS), folds=K_FOLDS, seeds=list(a.SEEDS), learners=list(a.LEARNERS),
                            lams=list(LAMS), cont=dict(region=CONT_REGION, lam=CONT_LAM), cb_iters=int(h41_args(a).HA.cb_iters),
                            knndm_index_sha=index_sha, m1c=dict(seeds=list(M1C_SEEDS), k=M1C_K, schemes=list(M1C_SCHEMES)),
                            fold_seed=dict(W1R="seed_of('xh','W1R',지역,반복)", W1S="seed_of('xh','W1S',지역,반복)", W1B="seed_of('lgv','W',지역,반복)"))


# ================================================================ 작업 단위
def enumerate_units(a):
    """(지역, 단, 반복, 변형) 목록. M1C 는 (알래스카, M1C, seed, '')."""
    units = []
    for R in a.REGIONS:
        for st in a.STAGES:
            if st == M1C:
                if R == CONT_REGION:
                    units += [(R, M1C, int(s), "") for s in M1C_SEEDS]
                continue
            vs = VARIANTS.get(R, ("",)) if st == "W1K" else ("",)
            for rep in a.REPS:
                for v in vs:
                    units.append((R, st, int(rep), v))
    return units


def parse_shard(txt):
    p = txt.split(":")
    if len(p) not in (3, 4):
        raise SystemExit("--shard 는 <지역>:<단>:<반복>[:<변형>] 형식이다")
    return (p[0], p[1], int(p[2]), p[3] if len(p) == 4 else "")


def run_unit(a, region, stage, rep, variant="", dry=False):
    """단위 하나를 실행하고 조각을 쓴다. dry 이면 학습 없이 수만 센다(라벨 값 미사용)."""
    if stage == M1C:
        return run_m1c(a, region, rep, dry)
    t0 = time.time()
    V = get_vdata(a)
    index, isha = (load_index(a) if stage == "W1K" else (None, ""))
    f_t = stage_folds(a, V, region, stage, rep, variant, index)
    t_idx, sc = region_index(V, region)
    pos = np.full(V.N, -1, np.int64); pos[sc] = np.arange(len(sc))
    F = H41.make_fitter(h41_args(a), dry)
    cont = region == CONT_REGION
    preds, folds_info, nnd = {}, [], np.full(len(sc), np.nan)
    fold_of = np.full(len(sc), -1, np.int64)
    for k in range(K_FOLDS):
        tr, te = fold_cells(V, region, f_t, k, stage)
        d = nnd_to_train(V, te, tr)
        nnd[pos[te]] = d; fold_of[pos[te]] = k
        if dry:
            n_fit = 2 * len(a.LEARNERS) * len(a.SEEDS) + (2 if cont else 0)
            folds_info.append(dict(fold=k, n_train=int(len(tr)), n_test=int(len(te)), n_fit=n_fit))
            continue
        TR, TE = V.take(tr), V.take(te)
        out, rec = models_w1x(F, TR, TE, f"xh_{stage}", a.LEARNERS, a.SEEDS, LAMS, cont=cont)
        for kk, p in out.items():
            preds.setdefault(kk, np.full(len(sc), np.nan))[pos[te]] = np.asarray(p, float)
        folds_info.append(dict(fold=k, n_train=int(len(tr)), n_test=int(len(te)), E=rec["E"], nnd_median_km=float(np.nanmedian(d)) if len(d) else np.nan))
    cover = int((fold_of >= 0).sum())
    assert cover == len(sc), f"{region} {stage} r{rep}: 채점 셀 {len(sc) - cover}개가 어느 묶음에도 없다"
    base = dict(region=region, stage=stage, rep=int(rep), variant=variant, n_cells=int(len(t_idx)), n_scored=int(len(sc)),
                nb_scored=int(len(np.unique(V.block[sc]))), nnd_median_km=float(np.nanmedian(nnd)) if len(nnd) else np.nan,
                fold_sizes=[int((f_t == k).sum()) for k in range(K_FOLDS)], knndm_index_sha=isha)
    if dry:
        return dict(base, n_fit=int(sum(r["n_fit"] for r in folds_info)), folds=folds_info)
    name = store_name(region, stage, variant)
    st = BlockStore(name, int(rep), V.block[sc].astype(str), meta=dict(target=name.split("|")[0], mode=stage, exp=EXP_ID, variant=variant))
    keys = sorted(preds, key=lambda k: (k[0], k[1], k[2], k[3]))
    y = V.y[sc].astype(float)
    nonfin = {}
    for k in keys:
        p = preds[k]
        nf = int((~np.isfinite(p)).sum())
        if nf:
            nonfin["|".join(str(v) for v in k)] = nf
        st.add(full_key(k, stage, rep), y, p)
    paths = XB.shard_paths(a.SHARDS, a.TAG, region, stage, rep, variant)
    XB.check_out_dir(a.SHARDS)
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    cells = Path(str(paths["unit"])[:-len("_unit.json")] + "_cells.npz")
    tmp = cells.with_name(cells.name + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, idx=sc.astype(np.int64), loc_id=V.loc_id[sc].astype(np.int64), fold=fold_of, nnd_km=nnd.astype(np.float64),
                        keys=np.array([json.dumps(list(full_key(k, stage, rep))) for k in keys], dtype=str),
                        P=np.vstack([preds[k] for k in keys]).astype(np.float64))
    os.replace(tmp, cells)
    rows = [dict(region=region, stage=stage, rep=int(rep), variant=variant, **r) for r in folds_info]
    n_fit = int(sum(F.n.values())); n_fail = int(sum(F.fail.values()))
    unit = dict(base, status="ok" if (n_fail == 0 and not nonfin) else "partial", n_fit=n_fit, fail=dict(F.fail), n_nonfinite=nonfin,
                folds=folds_info, sec=round(time.time() - t0, 1), fit_sec={k_: round(v, 2) for k_, v in F.secd.items()}, n_keys=len(keys),
                blind=BLIND.get(region, ""), design=DESIGN, cells_file=cells.name, cells_sha256=XB.sha256_file(cells))
    XB.write_shard(a.SHARDS, a.TAG, region, stage, rep, rows, [st], unit_cfg(a, isha_or_index(a)), unit, variant, expected=a.REPS,
                   code_file=__file__)
    return dict(region=region, stage=stage, rep=int(rep), variant=variant, n_fit=n_fit, status=unit["status"])


def isha_or_index(a):
    """공통 설정에 넣는 색인 sha256(색인이 없으면 빈 문자열)."""
    return XB.sha256_file(a.INDEX) if a.INDEX.exists() else ""


def m1_alaska(V):
    """m1 의 알래스카 채점 집합(ak = macro 알래스카 ∩ eval_mask, load_base 행 순서)."""
    from polar.m1_core import eval_mask
    df = V.D.df
    m = (df.macro.values == "Alaska")
    ak = np.where(m)[0]
    return ak[eval_mask(df.iloc[ak])]


def catboost_lo_m1(Xtr, ytr, Xte, seed):
    """m1 의 catboost_lo(m1_core.fit_model 과 같은 초모수, thread_count 4)."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed, verbose=0, allow_writing_files=False,
                          thread_count=4)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte))


def m1_predict_fold(tr, te, seed):
    """m1 predict_fold 와 같은 식(stefan, ridge, stefan_ridge_l075, catboost_lo)."""
    from polar.m1_core import INPUT_SETS, fit_coefs, anchor_pred, fit_model
    from polar.fidelity import TARGET
    from polar.preprocessing import fold_prep
    feats = INPUT_SETS["x25"]
    k = fit_coefs(tr)
    Xtr = tr[feats].values.astype(np.float32); Xte = te[feats].values.astype(np.float32)
    ytr = tr[TARGET].values.astype(float)
    st_tr, st_te = anchor_pred("stefan", tr, k), anchor_pred("stefan", te, k)
    Xa, Xb = fold_prep(Xtr, Xte, nan_native=False)
    return {"stefan": st_te, "ridge": fit_model("ridge", Xa, ytr, Xb, seed)["pred"],
            "stefan_ridge_l075": st_te + 0.75 * fit_model("ridge", Xa, ytr - st_tr, Xb, seed)["pred"],
            "catboost_lo": catboost_lo_m1(Xtr, ytr, Xte, seed)}


def run_m1c(a, region, seed, dry=False):
    """관문 (2): m1 정의의 알래스카 6겹(random_cell, site_0.05, block_0.5) 재실행. 키 = (모형, 'm1', '1', 방식, −1, 0, seed, 0)."""
    t0 = time.time()
    V = get_vdata(a)
    ak = m1_alaska(V)
    df = V.D.df
    folds = CV.m1_folds(V.lat[ak], V.lon[ak], V.block[ak], int(seed), M1C_K)
    if dry:
        return dict(region=region, stage=M1C, rep=int(seed), variant="", n_scored=int(len(ak)), n_fit=len(M1C_SCHEMES) * M1C_K,
                    fold_sizes={s: [int((f == k).sum()) for k in range(M1C_K)] for s, f in folds.items()})
    from polar.fidelity import TARGET
    y = df[TARGET].values[ak].astype(float)
    st = BlockStore(store_name(region, M1C), int(seed), V.block[ak].astype(str), meta=dict(target=region, mode=M1C, exp=EXP_ID))
    rows = []
    for sch in M1C_SCHEMES:
        f = folds[sch]
        P = {m: np.full(len(ak), np.nan) for m in M1C_MODELS}
        for k in range(M1C_K):
            te = f == k
            out = m1_predict_fold(df.iloc[ak[~te]], df.iloc[ak[te]], int(seed))
            for m in M1C_MODELS:
                P[m][te] = np.asarray(out[m], float)
            rows.append(dict(scheme=sch, fold=k, n_test=int(te.sum())))
        for m in M1C_MODELS:
            st.add((m, "m1", "1", sch, -1, 0, int(seed), 0.0), y, P[m])
    unit = dict(region=region, stage=M1C, rep=int(seed), status="ok", sec=round(time.time() - t0, 1), n_scored=int(len(ak)), design="관문 (2) 전용")
    XB.check_out_dir(a.SHARDS)
    XB.write_shard(a.SHARDS, a.TAG, region, M1C, int(seed), rows, [st], unit_cfg(a, isha_or_index(a)), unit, "", expected=list(M1C_SEEDS),
                   code_file=__file__)
    return dict(region=region, stage=M1C, rep=int(seed), variant="", status="ok")


# ---------------------------------------------------------------- 워커
_WA = None


def _worker_init(argv):
    global _WA
    warnings.filterwarnings("ignore")
    _WA = parse_args(argv)
    if not on_rescale():                                                   # spawn 워커마다 로컬 상주 메모리 감시(계획 1절: 작업 하나 10 GB)
        rss_watchdog(RSS_LIMIT_GB)


def _worker_run(region, stage, rep, variant):
    return run_unit(_WA, region, stage, rep, variant)


# ================================================================ kNNDM 색인(로컬 1c, 라벨 미사용)
def build_knndm(a, regions=None, reps=None, n_q=KNNDM_N_Q, log=print, domain_fn=None):
    """예측 영역과 kNNDM 색인을 만든다. 입력은 좌표와 기후 격자뿐이다. 반환 (색인 DataFrame, 영역 DataFrame, 메타 dict).
    domain_fn(region, variant, lat, lon) → (pts, meta) 를 주면 예측 영역을 그 함수로 만든다(시험용)."""
    V = get_vdata(a)
    regions = list(regions or a.REGIONS); reps = list(reps or a.REPS)
    idx_rows, dom_rows, meta = [], [], dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), k=K_FOLDS, n_q=n_q, maxp=KNNDM_MAXP, entries=[],
                                            deps=XB.deps_versions(("scikit-learn", "numpy", "scipy", "pyproj", "xarray")), plan=XB.PLAN_DOC,
                                            plan_commit=XB.PLAN_COMMIT, rule="계획 2.8 W1K, cv_schemes(m1 사본)")
    for R in regions:
        t_idx, _ = region_index(V, R)
        lat, lon = V.lat[t_idx], V.lon[t_idx]
        crs = CV.region_crs(R, lat, lon)
        XY = CV.project_km(lon, lat, crs)
        rad = CV.to_rad(lon, lat)
        for v in VARIANTS.get(R, ("",)):
            if domain_fn is not None:
                pts, dmeta = domain_fn(R, v, lat, lon)
            elif R == "Alaska":
                dom, dmeta = CV.prediction_domain(ALASKA_NPRED, ALASKA_RES, 0, log)
                pts = dom["permafrost"]
            elif R == "Lena":
                pts, dmeta = CV.prediction_domain_grid(LENA_GRID)
            else:
                pts, dmeta = CV.prediction_domain_box(CV.label_box(lat, lon, PAD_OF[v]))
            gij = CV.nnd_km(CV.to_rad(pts[:, 0], pts[:, 1]), rad)
            dom_rows.append(pd.DataFrame(dict(region=R, variant=v, lon=np.round(pts[:, 0], 5), lat=np.round(pts[:, 1], 5))))
            for rep in reps:
                seed = int(seed_of("xh", "W1K", R, v, rep))
                folds, info, _ = CV.knndm(XY, rad, gij, K_FOLDS, seed, n_q, KNNDM_MAXP, log)
                idx_rows.append(pd.DataFrame(dict(region=R, variant=v, rep=int(rep), loc_id=V.loc_id[t_idx].astype(np.int64), fold=folds.astype(int))))
                info = {k_: v_ for k_, v_ in info.items() if k_ != "curve"}
                meta["entries"].append(dict(region=R, variant=v, rep=int(rep), crs=crs, n_cells=int(len(t_idx)), domain=dmeta,
                                            gij_median_km=float(np.median(gij)), **info))
    return pd.concat(idx_rows, ignore_index=True), pd.concat(dom_rows, ignore_index=True), meta


def write_knndm(a, idx, dom, meta):
    XB.check_out_dir(a.INDEX.parent)
    a.INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.INDEX.with_name(a.INDEX.name + f".tmp{os.getpid()}")
    idx.to_csv(tmp, index=False); os.replace(tmp, a.INDEX)
    dp = a.INDEX.with_name(DOMAIN_NAME)
    dom.to_csv(dp, index=False, compression="gzip")
    meta.update(index_file=a.INDEX.name, index_sha256=XB.sha256_file(a.INDEX), domain_file=dp.name, domain_sha256=XB.sha256_file(dp),
                max_rss_mb=XB.max_rss_mb())
    a.INDEX_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    return meta


# ================================================================ 세기(라벨 미사용)
def count_only(a):
    units = enumerate_units(a)
    index, _ = load_index(a, verify=False) if a.INDEX.exists() else (None, "")
    tot_fit, tot_est, rows = 0, 0.0, []
    for u in units:
        if u[1] == "W1K" and index is None:
            rows.append(dict(unit="|".join(str(v) for v in u), status="색인 없음(--build-knndm 필요)"))
            continue
        r = run_unit(a, *u, dry=True)
        est = 0.0
        for fi in r.get("folds", []):
            for lr in a.LEARNERS:
                b, ref, ex = H41.BASE_FIT_V.get(lr, (1.0, 16700, 1.0))
                est += 2 * len(a.SEEDS) * b * (max(fi["n_train"], 1) / ref) ** ex
            est += (2 * EST_RIDGE_S if u[0] == CONT_REGION else 0.0)
        tot_fit += int(r["n_fit"]); tot_est += est
        rows.append(dict(unit="|".join(str(v) for v in u), n_scored=r["n_scored"], n_fit=r["n_fit"], est_s=round(est, 1),
                         n_train=[fi["n_train"] for fi in r.get("folds", [])], n_test=[fi["n_test"] for fi in r.get("folds", [])]))
    for r in rows:
        print(f"  [세기] {r['unit']}: " + " · ".join(f"{k} {v}" for k, v in r.items() if k != "unit"), flush=True)
    print(f"[세기] 단위 {len(units)}개 · 적합 {tot_fit}건 · 추정 {tot_est / 3600:.2f} 워커·시간(catboost·rf 는 h41 가정값) · 색인 "
          f"{'있음' if index is not None else '없음'}", flush=True)
    return dict(n_units=len(units), n_fit=tot_fit, est_h=tot_est / 3600)


# ================================================================ 집계(봉인 폴더)
def _rmse_pair(S, C):
    with np.errstate(invalid="ignore", divide="ignore"):
        cell = float(np.sqrt(S.sum() / C.sum())) if C.sum() > 0 else np.nan
        beq = float(np.nanmean(np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), np.nan))) if (C > 0).any() else np.nan
    return cell, beq


def _boot_pair(S, C, W):
    """재표집 분포(셀 가중, 블록 등가중). S·C 는 블록 벡터, W = (nboot, nb)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cell = np.sqrt((W @ S) / (W @ C))
        v = np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), 0.0); m = (C > 0).astype(float)
        beq = (W @ v) / (W @ m)
    return cell, beq


def avg_terms(stores, name):
    """저장소 이름 하나의 반복·seed 평균 블록 SSE: {(method, learner, λ): (S̄, C̄, 행 수)}. 반복마다 블록 집합이 같아야 한다."""
    per = [st for (nm, _), st in sorted(stores.items()) if nm == name]
    if not per:
        return {}, None
    blocks = per[0].blocks
    acc = {}
    for st in per:
        if not np.array_equal(st.blocks, blocks):
            raise SystemExit(f"[집계] {name}: 반복 사이 채점 블록 집합이 다르다")
        for k in st.keys:
            s, c = st.get(k)
            acc.setdefault((k[0], k[1], float(k[7])), []).append((s, c))
    out = {g: (np.mean([v[0] for v in L], 0), np.mean([v[1] for v in L], 0), len(L)) for g, L in acc.items()}
    return out, blocks


def check_nboot(a):
    """본 봉인 폴더에는 등록 재표집 수(10,000회)로만 쓴다. 허용 표지 없는 로컬 실행(1,000회 상한)이나 --nboot 를 줄인 실행은 멈춘다."""
    if not a.smoke and int(a.nboot) < int(XB.NBOOT):
        raise SystemExit(f"[집계] 재표집 {int(a.nboot)}회는 등록 {int(XB.NBOOT)}회보다 적다. 본 봉인 폴더에 쓰지 않는다"
                         "(로컬은 --allow-local, 계획 2.8 '셀 단위 제곱 오차의 블록 재표집 CI')")


def gate_status(a, shards=None) -> dict:
    """관문 기록(xh_gate_meta.json, run_gates 가 쓴다)에서 관문 (1)·(2)의 상태(통과, 실패, 미실행, 기록 없음)를 읽는다. 기록이 조각(unit.json)보다
    오래되면 stale 이다. 관문 (2)가 '실패'이고 기록이 오래되지 않았을 때만 알래스카 연속성 행(ridge 키)을 뺀다(cont_action = 'drop')."""
    p = a.OUT / GATE_META
    if not p.exists():
        return dict(found=False, stale=False, gate1="기록 없음", gate2="기록 없음", gate2_note="", cont_action="flag", summary={})
    m = json.loads(p.read_text())
    sh = XB.find_shards(a.SHARDS, a.TAG) if shards is None else shards
    mt = [Path(s_["unit"]).stat().st_mtime for s_ in sh if Path(s_["unit"]).exists()]
    stale = bool(mt and max(mt) > p.stat().st_mtime)

    def st(g, note=""):
        if not g or int(g.get("n_keys", 0)) == 0 or "미실행" in str(note):
            return "미실행"
        return "통과" if bool(g.get("passed")) else "실패"
    g1, g2 = st(m.get("gate1")), st(m.get("gate2"), m.get("gate2_note", ""))
    return dict(found=True, stale=stale, created=m.get("created", ""), level=m.get("level", ""), gate1=g1, gate2=g2,
                gate2_note=m.get("gate2_note", ""), cont_action="drop" if (g2 == "실패" and not stale) else "flag",
                summary=dict(gate1=m.get("gate1"), gate2=m.get("gate2")))


def continuity_label(gs) -> str:
    """알래스카 연속성 행(ridge 키)의 표지. 관문 (2) 통과면 '통과', 그 밖은 '관문 2 미확인(사유)'(실패면 행을 빼므로 쓰이지 않는다)."""
    if gs.get("gate2") == "통과" and not gs.get("stale"):
        return "관문 2 통과"
    why = "관문 기록이 조각보다 오래됨" if gs.get("stale") else str(gs.get("gate2", "기록 없음"))
    return f"{CONT_UNCHECKED_TXT}({why})"


def summarize(a, write=True):
    """조각 → 사다리 표, XH-1·XH-2 의 ΔΔ, 단별 대비(XH-3). 표는 봉인 폴더에 쓰고 화면에는 행 수와 해시만 쓴다.
    본 봉인 폴더(스모크 아님)에는 재표집 10,000회로만 쓴다. 관문 (2)가 실패하면 알래스카 연속성 행(ridge 키)을 빼고 그 사유를 메타에 쓴다."""
    if write:
        check_nboot(a)
    tms_units = XB.find_shards(a.SHARDS, a.TAG)
    if not tms_units:
        print("[집계] 조각이 없다", flush=True)
        return {}
    units = [json.loads(s_["unit"].read_text()) for s_ in tms_units]
    XB.check_cfg(units, a.allow_mixed_cfg)
    stores = H4.load_stores([s_["npz"] for s_ in tms_units])
    unit_by = {(u["target"], u["mode"], int(u["split"]), u.get("variant", "")): u for u in units}
    terms, blocks_of = {}, {}
    names = sorted({nm for nm, _ in stores})
    for nm in names:
        if nm.endswith("|" + M1C):
            continue
        t, blk = avg_terms(stores, nm)
        terms[nm] = t
        reg = nm.split("|")[0].split("~")[0]
        if reg in blocks_of and not np.array_equal(blocks_of[reg], blk):
            raise SystemExit(f"[집계] {reg}: 단 사이 채점 블록 집합이 다르다")
        blocks_of[reg] = blk
    W = {r: H4.boot_weights(len(b), int(a.nboot), seed_of("xhboot", r)) if a.nboot > 0 else None for r, b in blocks_of.items()}
    ladder, dd_rows, sc_rows = [], [], []

    def term(nm, g):
        v = terms.get(nm, {}).get(g)
        if v is None:
            return None
        reg = nm.split("|")[0].split("~")[0]
        S, C, nr = v
        pt = _rmse_pair(S, C)
        dist = _boot_pair(S, C, W[reg]) if W.get(reg) is not None else (None, None)
        return dict(pt=pt, dist=dist, n_rows=nr, nb=int(len(S)))

    for nm in sorted(terms):
        reg, stage = nm.split("|")[0], nm.split("|")[1]
        var = reg.split("~")[1] if "~" in reg else ""
        reg0 = reg.split("~")[0]
        nnd = [u.get("nnd_median_km", np.nan) for (t_, m_, r_, v_), u in unit_by.items() if t_ == reg0 and m_ == stage and v_ == var]
        for g in sorted(terms[nm]):
            tm = term(nm, g)
            lo, hi = XB.X._ci(tm["dist"][0]); lob, hib = XB.X._ci(tm["dist"][1])
            ladder.append(dict(region=reg0, variant=var, stage=stage, method=g[0], learner=g[1], lam=g[2], rmse=tm["pt"][0], rmse_lo=lo, rmse_hi=hi,
                               rmse_beq=tm["pt"][1], rmse_beq_lo=lob, rmse_beq_hi=hib, n_rows=tm["n_rows"], n_blocks=tm["nb"],
                               nnd_median_km=float(np.nanmedian(nnd)) if nnd else np.nan, blind=BLIND.get(reg0, ""), source="새 실행",
                               n_reps=len(nnd)))
    # 단별 대비(XH-3): D0 − P1, R1 − P1(학습기 3종, λ 0.25)
    p1 = ("PSw", "none", 0.0)
    for nm in sorted(terms):
        reg, stage = nm.split("|")
        for lr in a.LEARNERS:
            for lab, gA in (("D0−P1", ("D0w", lr, 1.0)), ("R1−P1", ("RSw", lr, 0.25))):
                A, B = term(nm, gA), term(nm, p1)
                if A is None or B is None:
                    continue
                dc = A["dist"][0] - B["dist"][0] if A["dist"][0] is not None else None
                db = A["dist"][1] - B["dist"][1] if A["dist"][1] is not None else None
                lo, hi = XB.X._ci(dc); lob, hib = XB.X._ci(db)
                sc_rows.append(dict(region=reg.split("~")[0], variant=reg.split("~")[1] if "~" in reg else "", stage=stage, contrast=lab, learner=lr,
                                    delta=A["pt"][0] - B["pt"][0], ci_lo=lo, ci_hi=hi, delta_beq=A["pt"][1] - B["pt"][1], ci_lo_beq=lob,
                                    ci_hi_beq=hib, nboot=int(a.nboot)))
    # XH-1·XH-2 ΔΔ(지역 행과 3지역 층화 평균)
    for item, gm in (("XH-1", lambda lr: ("PSw", "none", 0.0)), ("XH-2", lambda lr: ("RSw", lr, 0.25))):
        for lr in a.LEARNERS:
            for var in ("", "pm2"):
                per = {}
                for reg in REGIONS:
                    vk = var if (var and reg in VARIANTS) else ""
                    nmK = store_name(reg, "W1K", vk); nmR = store_name(reg, "W1R")
                    tKD, tRD = term(nmK, ("D0w", lr, 1.0)), term(nmR, ("D0w", lr, 1.0))
                    tKM, tRM = term(nmK, gm(lr)), term(nmR, gm(lr))
                    if None in (tKD, tRD, tKM, tRM):
                        continue
                    pt = [(tKD["pt"][i] - tRD["pt"][i]) - (tKM["pt"][i] - tRM["pt"][i]) for i in (0, 1)]
                    ds = [None if tKD["dist"][i] is None else (tKD["dist"][i] - tRD["dist"][i]) - (tKM["dist"][i] - tRM["dist"][i]) for i in (0, 1)]
                    per[reg] = dict(pt=pt, dist=ds, nb=tKD["nb"], terms=[tKD["pt"][0], tRD["pt"][0], tKM["pt"][0], tRM["pt"][0]])
                if var and "Canada" not in per:
                    continue
                for reg, v in per.items():
                    lo, hi = XB.X._ci(v["dist"][0]); lob, hib = XB.X._ci(v["dist"][1])
                    dd_rows.append(dict(item=item, learner=lr, variant=var, scope="region", target=reg, dd=v["pt"][0], ci_lo=lo, ci_hi=hi,
                                        dd_beq=v["pt"][1], ci_lo_beq=lob, ci_hi_beq=hib, n_blocks=v["nb"], rmse_K_D0=v["terms"][0],
                                        rmse_R_D0=v["terms"][1], rmse_K_M=v["terms"][2], rmse_R_M=v["terms"][3], blind=BLIND.get(reg, ""),
                                        nboot=int(a.nboot)))
                pool = [r for r in REGIONS if r in per and per[r]["dist"][0] is not None and per[r]["nb"] >= XB.MIN_BLOCKS_CI]
                if len(pool) >= XB.MIN_POOL_REGIONS:
                    dc, db = MS.strat([per[r]["dist"][0] for r in pool]), MS.strat([per[r]["dist"][1] for r in pool])
                    lo, hi = XB.X._ci(dc); lob, hib = XB.X._ci(db)
                    dd_rows.append(dict(item=item, learner=lr, variant=var, scope="MEAN", target=f"MEAN[{','.join(pool)}]",
                                        dd=float(np.mean([per[r]["pt"][0] for r in pool])), ci_lo=lo, ci_hi=hi,
                                        dd_beq=float(np.mean([per[r]["pt"][1] for r in pool])), ci_lo_beq=lob, ci_hi_beq=hib,
                                        n_blocks=int(sum(per[r]["nb"] for r in pool)), pool=XB.pool_label(len(pool), len(REGIONS)),
                                        nboot=int(a.nboot)))
    # 관문 (2)(계획 2.8): 실패하면 알래스카 연속성 행(ridge 키)을 빼고, 확인되지 않았으면 표지를 단다
    gs = gate_status(a, tms_units)
    n_ridge = sum(1 for r in ladder if r["learner"] == "ridge")
    if gs["cont_action"] == "drop":
        ladder = [r for r in ladder if r["learner"] != "ridge"]
    lab = continuity_label(gs)
    for r in ladder:
        r["continuity_gate"] = lab if r["learner"] == "ridge" else ""
    gs.update(ridge_rows=int(n_ridge), ridge_rows_dropped=int(n_ridge if gs["cont_action"] == "drop" else 0),
              ridge_note=CONT_FAIL_TXT + ": 알래스카 연속성 행(ridge 키)을 뺐다" if gs["cont_action"] == "drop" else lab)
    reuse = reuse_h41_rows()
    out = dict(xh_ladder=pd.DataFrame(ladder), xh_dd=pd.DataFrame(dd_rows), xh_stage_contrasts=pd.DataFrame(sc_rows), xh_reuse_h41=reuse)
    out_meta = dict(n_shards=len(tms_units), nboot=int(a.nboot), nboot_registered=int(XB.NBOOT), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    cfg=XB.check_cfg(units, True), regions=list(REGIONS), blind=BLIND, design=DESIGN, gates=gs, file_location=LOCATION_NOTE)
    if write:
        sealed_name = NAME if not a.smoke else f"{NAME}/smoke"
        XB.write_sealed(sealed_name, {f"{k}.csv": v for k, v in out.items()}, root=XB.XBATCH_ROOT)
        XB.write_sealed(sealed_name, {"xh_summary_meta.json": out_meta}, root=XB.XBATCH_ROOT)
        print(f"[집계] 관문 1 {gs['gate1']} · 관문 2 {gs['gate2']} · 오래됨 {gs['stale']} · 능형 행 뺌 {gs['ridge_rows_dropped']}", flush=True)
    a.SUMMARY_META = out_meta
    return out


def reuse_h41_rows():
    """V-G(지역 홀드아웃)와 h41 W1(반복 5) 행을 h41 표에서 옮긴다(새 적합 없음). V-G 의 PS·RS·D0 는 각각 P0·R0·D0 에 해당한다."""
    if not H41_METRICS.exists():
        return pd.DataFrame()
    m = pd.read_csv(H41_METRICS, dtype=dict(scheme=str, method=str, learner=str, scope=str, target=str))
    keep = (m.scope == "region") & m.target.isin(REGIONS) & (
        ((m.scheme == "V-G") & m.method.isin(["PS", "RS", "D0"])) | (m.scheme.str.startswith("W1-") & m.method.isin(["PSw", "RSw", "D0w"])))
    out = m[keep].copy()
    out["xh_stage"] = np.where(out.scheme == "V-G", "V-G", "W1B(h41, 반복 5)")
    out["xh_model"] = out.method.map({"PS": "P0", "RS": "R0", "D0": "D0", "PSw": "P1형(PSw)", "RSw": "R1형(RSw)", "D0w": "D0w"})
    out["source"] = "h41 재사용(lgv_metrics.csv)"
    return out


# ================================================================ 재현 관문
def gate1_decision(d, rel, ulp, same_cnt, level="local_rescale"):
    """관문 (1) 키 하나의 통과 규칙(계획 1절 '재현 관문 허용 오차'의 문자 그대로): 유한 셀 집합이 같고 |ΔSSE| ≤ 절대 허용(cm²) 또는 상대 허용
    이내면 통과. 예측 차의 float32 ulp 는 통과 근거가 아니다. 반환 (ok, pass_ulp_only = 허용 오차는 넘었으나 예측 차가 1 ulp 이내, 정보 열)."""
    ta, tr_ = XB.GATE_TOL[level]
    ok = bool(same_cnt and np.isfinite(d) and (d <= ta or (tr_ > 0 and np.isfinite(rel) and rel <= tr_)))
    ulp_only = bool(same_cnt and not ok and np.isfinite(ulp) and ulp <= 1.0)
    return ok, ulp_only


def gate_w1b(a, level=None):
    """관문 (1): 새 W1B 와 h41 W1 조각의 키별 블록 SSE(두 쪽 모두 float32 예측). rf 키는 넣지 않는다. 통과는 1절 허용 오차(level)만으로 정한다.
    max_ulp(예측 차의 float32 ulp)와 pass_ulp_only(허용 오차는 넘고 1 ulp 이내)는 정보 열이다."""
    level = level or a.gate_level
    V = get_vdata(a)
    rows = []
    for R in a.REGIONS:
        _, sc = region_index(V, R)
        pos = np.full(V.N, -1, np.int64); pos[sc] = np.arange(len(sc))
        for rep in a.REPS:
            p = XB.shard_paths(a.SHARDS, a.TAG, R, "W1B", rep)
            cp = Path(str(p["unit"])[:-len("_unit.json")] + "_cells.npz")
            if not cp.exists():
                rows.append(dict(item="W1B_vs_h41", region=R, rep=rep, key="", n_cells=0, max_abs_dsse=np.nan, max_ulp=np.nan, ok=False, note="새 조각 없음"))
                continue
            with np.load(cp, allow_pickle=False) as z:
                idx = z["idx"]; keys = [tuple(json.loads(s)) for s in z["keys"]]; P = z["P"]
            new = {(k[0], k[1], int(k[6]), float(k[7])): P[i] for i, k in enumerate(keys)}
            ref = {}
            cov = np.zeros(len(sc), int)
            for f in range(K_FOLDS):
                hp = H41_SHARDS / f"lgv__W1-{R}__r{rep}__f{f}_pred.npz"
                if not hp.exists():
                    continue
                with np.load(hp, allow_pickle=False) as z:
                    hidx = z["idx"].astype(np.int64); hk = [tuple(json.loads(s)) for s in z["keys"]]; HP = np.asarray(z["P"], np.float32)
                if np.any(pos[hidx] < 0):
                    raise SystemExit(f"[관문 1] h41 조각 {hp.name} 의 채점 셀이 새 채점 집합 밖이다")
                cov[pos[hidx]] += 1
                for i, k in enumerate(hk):
                    kk = (str(k[0]), str(k[1]), int(k[2]), float(k[3]))
                    ref.setdefault(kk, np.full(len(sc), np.nan, np.float32))[pos[hidx]] = HP[i]
            if not ref or not np.array_equal(idx, sc):
                rows.append(dict(item="W1B_vs_h41", region=R, rep=rep, key="", n_cells=int((cov > 0).sum()), max_abs_dsse=np.nan, max_ulp=np.nan,
                                 ok=False, note="h41 조각 없음 또는 채점 집합 불일치"))
                continue
            y = V.y[sc].astype(float); blk = V.block[sc].astype(str)
            for kk in sorted(set(new) & set(ref)):
                if kk[1] == "rf":
                    continue
                pn = np.asarray(new[kk], np.float32); pr = ref[kk]
                sa = BlockStore("n", 0, blk); sb = BlockStore("r", 0, blk)
                sa.add(("k",), y, pn.astype(float)); sb.add(("k",), y, pr.astype(float))
                d = float(np.max(np.abs(sa.get(("k",))[0] - sb.get(("k",))[0])))
                rel = d / max(float(np.max(np.abs(sb.get(("k",))[0]))), 1e-300)
                fin = np.isfinite(pn) & np.isfinite(pr)
                ulp = float(np.max(np.abs(pn[fin] - pr[fin]) / np.spacing(np.maximum(np.abs(pr[fin]), np.float32(1e-30))))) if fin.any() else np.inf
                same_cnt = bool(np.array_equal(np.isfinite(pn), np.isfinite(pr)))
                ok, ulp_only = gate1_decision(d, rel, ulp, same_cnt, level)   # 1절 허용 오차만(ulp 는 정보 열)
                rows.append(dict(item="W1B_vs_h41", region=R, rep=rep, key="|".join(str(v) for v in kk), n_cells=int(fin.sum()), max_abs_dsse=d,
                                 max_rel_dsse=rel, max_ulp=ulp, ok=ok, pass_ulp_only=ulp_only,
                                 note=f"1절 허용 오차 {level}" + (". 예측 차는 1 ulp 이내(정보, 통과 근거 아님)" if ulp_only else "")))
    return pd.DataFrame(rows)


def gate_m1c(a):
    """관문 (2): M1C 의 풀링 RMSE(소수 3자리 반올림)를 m1 표와 대조한다. 표에는 차의 절댓값과 통과 여부만 쓴다."""
    rows = []
    if not M1_TABLE.exists():
        return pd.DataFrame([dict(item="M1C_vs_m1", ok=False, note="m1 표 없음")])
    ref = pd.read_csv(M1_TABLE)
    sh = [s_ for s_ in XB.find_shards(a.SHARDS, a.TAG) if s_["mode"] == M1C]
    if not sh:
        return pd.DataFrame([dict(item="M1C_vs_m1", ok=False, note="M1C 조각 없음")])
    stores = H4.load_stores([s_["npz"] for s_ in sh])
    for (nm, seed), st in sorted(stores.items()):
        for k in st.keys:
            s, c = st.get(k)
            new = round(float(np.sqrt(s.sum() / c.sum())), 3) if c.sum() > 0 else np.nan
            q = ref[(ref.scheme == k[3]) & (ref.model == k[0]) & (ref.seed.astype(int) == int(seed))]
            if not len(q):
                rows.append(dict(item="M1C_vs_m1", region=nm, rep=int(seed), key=f"{k[3]}|{k[0]}", abs_diff=np.nan, ok=False, note="m1 행 없음"))
                continue
            dif = abs(new - float(q.rmse_cm.iloc[0]))
            rows.append(dict(item="M1C_vs_m1", region=nm, rep=int(seed), key=f"{k[3]}|{k[0]}", abs_diff=dif, n_cells=int(c.sum()),
                             ok=bool(np.isfinite(dif) and dif <= M1C_TOL), note="m1 표 반올림 단위 안 일치"))
    return pd.DataFrame(rows)


def run_gates(a):
    g1, g2 = gate_w1b(a), gate_m1c(a)
    XB.check_out_dir(a.OUT)
    a.OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.concat([g1, g2], ignore_index=True)
    tab.to_csv(a.OUT / "xh_gate.csv", index=False)
    s1 = dict(n_keys=int(len(g1)), n_fail=int((~g1.ok.astype(bool)).sum()) if len(g1) else 0)
    s2 = dict(n_keys=int(len(g2)), n_fail=int((~g2.ok.astype(bool)).sum()) if len(g2) else 0)
    s1["passed"] = bool(s1["n_keys"] > 0 and s1["n_fail"] == 0); s2["passed"] = bool(s2["n_keys"] > 0 and s2["n_fail"] == 0)
    no_m1c = bool(len(g2) and "note" in g2 and g2["note"].astype(str).str.contains("조각 없음").all())
    note2 = "" if s2["passed"] else ("M1C 조각 없음(관문 2 미실행)" if no_m1c else "대회 연속성 재현 실패(환경 미기록): 알래스카 연속성 행(ridge 키)을 뺀다")
    meta = dict(gate1=s1, gate2=s2, gate2_note=note2, level=a.gate_level, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (a.OUT / "xh_gate_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    s1["n_ulp_only"] = int(g1.pass_ulp_only.astype(bool).sum()) if (len(g1) and "pass_ulp_only" in g1) else 0
    print(f"[관문 1] W1B 와 h41 W1: 키 {s1['n_keys']} · 실패 {s1['n_fail']} · 통과 {s1['passed']} · 1 ulp 이내 실패 {s1['n_ulp_only']}", flush=True)
    print(f"[관문 2] M1C 와 m1 표: 키 {s2['n_keys']} · 실패 {s2['n_fail']} · 통과 {s2['passed']}" + (f" · {note2}" if note2 else ""), flush=True)
    return meta


def rss_watchdog(limit_gb=10.0, poll_s=2.0):
    """로컬 메모리 상한(계획 1절: 작업 하나 10 GB 이하). xbatch_core.limit_memory 의 RLIMIT_AS(주소 공간 10 GB)는 CatBoost(catboost 600회, 깊이 6)가
    큰 가상 영역을 예약해 적합 안에서 멈추게 했다(로컬 스모크에서 확인, 구현 기록). systemd-run 은 이 서버에서 쓸 수 없다. 그래서 상주 메모리(VmRSS)를
    주기적으로 보고 상한을 넘으면 프로세스를 끝내는 감시 스레드를 쓴다. 반환 스레드(이미 있으면 None)."""
    import threading
    if getattr(rss_watchdog, "_on", False):
        return None

    def rss_gb():
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
        except OSError:
            pass
        return 0.0

    def loop():
        while True:
            g = rss_gb()
            if g > float(limit_gb):
                print(f"[메모리] 상주 메모리 {g:.1f} GB 가 상한 {limit_gb} GB 를 넘어 끝낸다(계획 1절 로컬 자원)", flush=True)
                os._exit(3)
            time.sleep(float(poll_s))
    t = threading.Thread(target=loop, name="rss_watchdog", daemon=True)
    t.start()
    rss_watchdog._on = True
    return t


# ================================================================ 묶음 점검(4절)과 사전 점검
def payload_check(a, write=True) -> list:
    """작업 R1b 묶음의 필수 입력(PAYLOAD_EXTRA: kNNDM 색인, 색인 메타, 예측 영역) 점검. xbatch_core.payload_manifest(extra=PAYLOAD_EXTRA)의
    missing 에 든 필수 입력 목록을 돌려준다(비어 있어야 제출한다). write 이면 묶음 정보(파일 위치 이탈 기록 포함)를 <out>/xh_payload_manifest.json 에 쓴다."""
    man = XB.payload_manifest(extra=PAYLOAD_EXTRA)
    miss = [f for f in PAYLOAD_EXTRA if f in set(man.get("missing", []))]
    man["module_inputs"] = dict(module="x_validation_ladder", job="R1b", required=list(PAYLOAD_EXTRA), missing=miss, file_location=LOCATION_NOTE)
    if write:
        out = a.OUT if not a.smoke else a.OUT.parent
        XB.check_out_dir(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "xh_payload_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str))
    print(f"[묶음] 필수 입력 {len(PAYLOAD_EXTRA)}개 · 없음 {len(miss)}개" + (f": {', '.join(Path(f).name for f in miss)}" if miss else ""), flush=True)
    return miss


def preflight_w1k(a, units) -> str:
    """W1K 단위가 하나라도 있으면 적합 전에 색인 파일을 읽고 sha256 을 대조한다(없거나 다르면 SystemExit, 계획 1절: Rescale 에서 다시 군집하지 않는다).
    단위 안에서 실패하면 '실패 단위'로만 남고 작업이 계속 돌기 때문에 여기서 먼저 멈춘다. 반환 색인 sha256(W1K 단위가 없으면 빈 문자열)."""
    if not any(u[1] == "W1K" for u in units):
        return ""
    index, sha = load_index(a)
    if index is None:
        raise SystemExit(f"[W1K] kNNDM 색인 {a.INDEX} 이 없다(로컬 1c 의 --build-knndm 산출을 묶음에 넣는다. PAYLOAD_EXTRA, --payload-check)")
    need = {(u[0], u[3], int(u[2])) for u in units if u[1] == "W1K"}
    have = {(str(r), str(v), int(p)) for r, v, p in index[["region", "variant", "rep"]].drop_duplicates().values}
    miss = sorted(need - have)
    if miss:
        raise SystemExit(f"[W1K] 색인에 없는 (지역, 변형, 반복) {miss}")
    return sha


# ================================================================ 주 함수
def main(argv=None):
    a = parse_args(argv)
    if a.count_only or a.build_knndm or a.payload_check:
        mode = "count"
    elif a.smoke:
        mode = "smoke"
    elif a.summarize_only or a.gate_only:
        mode = "summarize"
    else:
        mode = "run"
    XB.guard(mode, a.argv)
    if not on_rescale():
        rss_watchdog(RSS_LIMIT_GB)                                         # 로컬이면 허용 표지와 관계없이 상주 메모리 감시(구현 기록 7절)
    with XB.restricted_output():
        print(f"[XH] {NAME} · 계획 {XB.PLAN_DOC} {XB.PLAN_REVISION} · 모드 {mode} · 스레드 {a.threads} · 재표집 {a.nboot}", flush=True)
        if a.payload_check:
            return 1 if payload_check(a) else 0
        if a.count_only:
            count_only(a)
            return 0
        if a.build_knndm:
            wait_memory()
            idx, dom, meta = build_knndm(a)
            meta = write_knndm(a, idx, dom, meta)
            print(f"[색인] 행 {len(idx)} · sha256 {meta['index_sha256'][:16]} · 영역 점 {len(dom)} · 최대 RSS {meta['max_rss_mb']} MB", flush=True)
            return 0
        if a.gate_only:
            wait_memory()
            run_gates(a)
            return 0
        if not a.summarize_only:
            wait_memory()
            units = [parse_shard(a.shard)] if a.shard else enumerate_units(a)
            preflight_w1k(a, units)                                        # W1K 가 있으면 적합 전에 색인·sha256 확인
            if a.resume:
                cfg = unit_cfg(a, isha_or_index(a))
                units = [u for u in units if not XB.unit_state(a.SHARDS, a.TAG, u[0], u[1], u[2], cfg, u[3])[0]]
            print(f"[XH] 단위 {len(units)}개", flush=True)
            done, failed = XB.execute(units, _worker_run, a.workers, a.threads, init=_worker_init, init_args=(a.argv,), permit=a.PERMIT)
            print(f"[XH] 완료 {len(done)} · 실패 {len(failed)} · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
            if failed:
                XB.check_out_dir(a.OUT)
                a.OUT.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([dict(unit="|".join(str(v) for v in u), error=e) for u, e in failed]).to_csv(a.OUT / f"{a.TAG}_failed.csv", index=False)
            if a.no_summarize:
                return 1 if failed else 0
        check_nboot(a)                                                     # 무거운 재표집 전에 확인(본 봉인 폴더는 10,000회만)
        wait_memory()
        out = summarize(a)
        print(f"[집계] 표 {len(out)}개 봉인 · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
