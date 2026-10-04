"""XF_new_regions(계획 2.6): 공개 자료 새 지역의 v5 표 조립, 적격 세기, 실행 표, 적합 단위, 대비·판정, 부록 XC-F3 목록.

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100, 2026-10-04 14:22:39 +0900) 2.6절, 1절 공통 규약,
0.3 열람·출력 규칙, 4절 묶음, 5절 마감. 계획의 해석과 구현 결정은 docs/research/2026-10-04/impl_notes/x_new_regions.md 에 적었다.
가설·판정 규칙은 계획 그대로다. 공용 골격은 xbatch_core(XB)이고 동결 모듈(h40, h42, h54)은 XB 를 거쳐서만 쓴다.

단계(--stage, 쉼표 목록. 라벨 통계를 화면에 쓰지 않는다)
  cells   새 자료원의 점 자료를 1 km 셀로 모은다. 입력은 <산출>/ext_labels/<src_id>_points.csv 뿐이다(새 자료원 파서
          scripts/1_data_prep/parse_ext_<id>.py 는 이 폴더에 쓴다. 기존 data/processed/ext_labels 는 읽지 않는다). 점 자료의 macro 는
          LGD 6B.4 의 지리 정의(geo_macro: 국가, 위도, 경도)와 같아야 하고 다르면 멈춘다. 점 자료 목록(이름, sha256)은
          xf_points_manifest.json 에 적고 XF 자료 마감(T0 + 7일) 뒤 첫 실행에서 동결한다. 마감 뒤에는 동결 목록 밖의 파일을 거부한다.
          build_ext_cells_v1.main 을 고치지 않고 입력·산출 폴더 전역 변수만 바꿔 부른다(6B.3·6B.4: 라벨 집합, 위치·셀 값, v3 중복,
          100 km 독립성). 그 모듈의 화면 출력(라벨 평균 포함)은 버리고, 메타에서 라벨 통계 항목을 지운다(계획 0.3, 2.6 열람 상태).
          v5 표지(라벨 값 미사용): dup_v4(v4 새 직접 라벨 셀과 체비쇼프 0.01° 이내), merge_v4(v4 에 들어간 셀과 같은 라벨 집합·셀 색인),
          v1_cell_not_v4(v4 에 들어가지 않은 v1 셀과 같은 셀. 제외하지 않고 표시만 한다), lic_nc(비상업 약관, 공개 v5 표에서 뺀다),
          역할(new_macro, natl_v5, augment_L40, alaska_subtask). 역할은 지리 macro 로 정한다(NAtlantic = 민감도, Alaska = 하위 과제 AL-7,
          그 밖의 6B.4·v3 macro = 셀 보강, 6B.4·v3·v4 어디에도 없는 macro = 새 macro 후보).
          in_v5 = 직접 라벨 ∩ dup_v3·dup_v4·merge_v4·lic_nc 아님. xf_target = in_v5 ∩ 독립성 통과.
  cov     ext_cells_covariates_v1 의 단계 함수(dem, e5, cci, sg, elev)를 고치지 않고 부르되 산출 폴더와 DEM·SoilGrids 폴더 전역 변수만
          덮개 폴더(data/raw/xf_dem_overlay, data/raw/xf_soilgrids_overlay: 기존 파일의 기호 연결 + 새 파일)로 바꾼다. --fetch 가 없으면
          내려받지 않는다(빠진 타일·창은 결측).
  v5      fidelity_base_v5.csv = v4 원문 줄 + 새 행(loc_id 30000 + 일련번호), e5_soil_tdd_v5.csv = v4 원문 줄 + 새 행, 새 행 부가 표.
          새 행 규약은 build_fidelity_base_v4 와 같다(알래스카 하위 과제 셀의 region 은 AL-7). v4 원문 줄이 바이트 단위로 같음을 단언한다.
  tdd     v5 새 행의 연도 정합 도일(a2_year_matched_tdd.py 정의, x_multisource_stacking.a2_tdd_cells 재사용, 라벨 값 미사용)
          v5/xf_tdd_matched_v5.csv. XB 의 P* 가 XF 표에서 쓴다(--xb 진입점).
  tables  spec 마다 run_tables/<별칭>/fidelity_base_v3.csv(v5 원문 줄 선택, 고른 새 행의 source_id 만 F4_direct,
          lgd_eligibility_v1.write_run_table_text)와 토양 표 기호 연결, 하위 지역 대상이면 블록 대응표(lg_subregion_map_v1 + 새 블록의
          최근접 중심 배정), xf_specs.json(표 경로는 산출 뿌리 기준 상대 경로).
명령
  --count-only   적격 표(분할 1–5 는 h40 규칙, 분할 201–210 은 xbatch 새 seed 규칙으로 따로 센다)와 적합 수(가짜 라벨 dry 실행).
                 라벨은 셀 유무(채점 마스크)에만 쓴다. 평균·분산·계수 비를 계산하지 않는다. 우선 자료원(Stordalen, Adventdalen)의 점 자료를
                 처음 넣은 세기는 NAtlantic v5 첫 세기 기록(xf_natl_v5_first_count.json, 한 번만 쓴다)으로 남긴다. 부록 XC-F3 최종판
                 (적격 동결) 뒤에는 세기를 거부한다(LGD 6B.4 '판정 시점').
  (기본)         적합 실행(WF_RESCALE=1 또는 --allow-local, XF 자료 마감 뒤 최종 부록 XC-F3 가 있어야 한다). (spec, 분할)마다 LG CPU 방법 축
                 (h40.run_ctx, LGD 와 같은 범위)과 WF4 팔(h54.TUnit: 규칙 W, 진단값)을 같은 문맥에서 적합하고 조각 두 개를 쓴다.
  --summarize-only  XF-1(D0 − P0, n 0), XF-2(R1(0.25) − P1, n 10), XF-3(R1(0.25) − P0, 전량)의 지역 행 4분 판정, 적격 2곳 이상이면 층화 평균,
                 Holm(m = 3 × 적격 지역 수, 보조 열), 다섯 갈래 문장, NAtlantic v5 민감도 행(첫 세기가 T0 + 24 h 안이고 적격일 때만 4분 판정),
                 셀 보강 행(L40 형식: L1 지역 조건, 전량 R1 − P0, n 10 P1 − P0 를 같은 작업의 기준 표와 비교해 L28 분류), XF-4(규칙 W − R1(0.25),
                 n 10·40·160, 진단값). 적격 표·spec·v5·조각 표 해시를 동결 기록과 대조한다. 봉인 폴더에만 쓰고 화면에는 행 수와 해시만 쓴다.
  --xc-f3        부록 XC-F3 목록(분할 201–210 에서 다시 센 적격 새 macro 지역. NAtlantic v5 는 넣지 않는다). XF 자료 마감 뒤 처음 만든
                 최종판이 적격 동결 기록이다(적격 표·spec·v5·이 스크립트의 sha256). 최종판은 다시 쓰지 않는다.
  --repro-gate   재현 관문(2.6): v5 의 v4 줄 불변, v4 표가 v4 메타의 해시와 같음, LGD 실행 표 네 개가 lgd_run_manifest.json 의 해시와 같음,
                 Tibet·NAtlantic_lic 실행 표를 새 경로로 다시 만들어 바이트 대조, 적격 세기와 LGD 적격 표 대조, NAtlantic~lic 의 분할
                 201–210 구조와 부록 A 대조, (--gate-b1) v1 점 자료로 다시 만든 셀 표 대조.
  --smoke        합성 대상 라벨 스모크(h51 스모크와 같은 합성 규칙). 분할 하나, n {0, 10, 전량}, 추출 1, seed 1. 코어 4개·스레드 2 규약을
                 점검하고(XB.smoke_env_report) 어긋나면 허용 표지 없이는 거부한다.
  --xb -- <XB 인자>  XB(x_multisource_stacking)를 XF 별칭과 v5 연도 정합 도일을 붙여 이 프로세스에서 부른다(XF-4 의 XB 행). 실제 적합은
                 적격 동결 기록이 있어야 하고 --workers 0 만 받는다(병렬은 --shard).
조각 이름(1절 '산출 경로'): <tag>__cpu__<별칭>__x__s<분할>__{lg, wf4}_{runs.csv, blocksse.npz, unit.json}. tag 기본 xf.
  저장소 이름은 LG 방법 축 '<별칭>|x', WF4 팔 '<별칭>~wf4|x' 다(한 tag 를 함께 읽어도 키가 섞이지 않는다).
로컬 자원(1절): WF_RESCALE=1 이 아니면(--allow-local 포함) 가용 메모리 30 GB 대기, 작업 하나 RSS 10 GB 감시(워커마다), 워커당 스레드 4 이하,
  워커 × 스레드 32 이하를 지킨다. 최대 RSS 는 세기·관문·스모크 메타에 적는다.

명령(ROOT)
  조립:   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/x_new_regions.py --stage cells,cov,v5,tdd,tables
  세기:   CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/x_new_regions.py --count-only --threads 2
  관문:   ... --repro-gate [--gate-b1]
  스모크:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 ... --smoke --threads 2 --workers 0 <XB.SHELL_FILTER>
  본 실행: WF_RESCALE=1 python3 scripts/3_deep_learning/x_new_regions.py --workers 22 --threads 4 --resume [--shard K/N] --no-summarize
  집계:   WF_RESCALE=1 python3 scripts/3_deep_learning/x_new_regions.py --summarize-only --threads 4
  XC-F3:  python3 scripts/3_deep_learning/x_new_regions.py --xc-f3
  XB 행:  WF_RESCALE=1 python3 scripts/3_deep_learning/x_new_regions.py --xb -- --exps xb_t --targets-t '<별칭>:x' --workers 0 --shard i/N ...
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import importlib
import io
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import xbatch_core as XB                                                   # noqa: E402  numpy 보다 먼저(스레드 환경 변수)

import numpy as np                                                         # noqa: E402
import pandas as pd                                                        # noqa: E402
from scipy.spatial import cKDTree                                          # noqa: E402

H, X, W = XB.H, XB.X, XB.W
ROOT = XB.ROOT
PROC = ROOT / "data" / "processed"
EXT = PROC / "ext_labels"
PREP = ROOT / "scripts" / "1_data_prep"
if str(PREP) not in sys.path:
    sys.path.insert(0, str(PREP))
import build_ext_cells_v1 as B1                                            # noqa: E402  셀 규칙(6B.3·6B.4). 고치지 않는다
import build_fidelity_base_v4 as B4                                        # noqa: E402  새 행 규약, 원문 줄 이어 쓰기
import lgd_eligibility_v1 as E1                                            # noqa: E402  실행 표 원문 줄 선택, DROP_V3
from polar.m1_core import load_base, eval_mask, half_split_blocks         # noqa: E402
from polar.m1_ext import haversine_km                                      # noqa: E402
from polar.fidelity import MACRO_REGION, SHARED_CORE                       # noqa: E402

# ================================================================ 고정값(계획 2.6, 1절)
EXP_ID = "XF"
EXP_NAME = XB.EXP_NAMES[EXP_ID]
TAG = "xf"
T0 = dt.datetime(2026, 10, 4, 14, 22, 39, tzinfo=dt.timezone(dt.timedelta(hours=9)))
DEADLINE_COUNT = T0 + dt.timedelta(hours=24)                              # Stordalen·Adventdalen 세기(NAtlantic v5)
DEADLINE_DATA = T0 + dt.timedelta(days=7)                                 # XF 자료 마감(5절)
LOC0_V4 = int(E1.LOC0)                                                     # 20000
LOC0_V5 = 30000                                                            # v5 새 행
V4_TABLE = PROC / "fidelity_base_v4.csv"
V4_SOIL = PROC / "e5_soil_tdd_v4.csv"
V4_LABELS = PROC / "fidelity_base_v4_labels.csv"
V4_META = PROC / "fidelity_base_v4_meta.json"                             # outputs 의 sha256(재현 관문)
LGD_MANIFEST = PROC / "lgd" / "lgd_run_manifest.json"                     # specs.<표>.table_sha256(재현 관문)
SUBMAP_V1 = PROC / "lg_subregion_map_v1.csv"                              # 블록 → 하위 지역(CA-1 등, 라벨 미사용)
SUBMAP_NAME = "lg_subregion_map_v1.csv"                                   # 실행 표 폴더 안의 대응표 이름(h40·h54 의 기본 --subregion-map)
TDD_V1 = PROC / "lgx_tdd_matched_v1.csv"
V1_CELLS = EXT / "ext_cells_v1.csv"
V1_META = EXT / "ext_cells_v1_meta.json"
PRIORITY_SOURCES = ("swe_stordalen_crill", "sjm_adventdalen_wendt2023")    # 2.6 순서 1·2(T0 + 24 h 안의 NAtlantic v5 세기)
LOCAL_TOTAL_THREADS = 32                                                   # 1절 로컬 자원: 우리 작업 합계 32스레드 이하
DEM_OVERLAY = ROOT / "data" / "raw" / "xf_dem_overlay"
SG_OVERLAY = ROOT / "data" / "raw" / "xf_soilgrids_overlay"
SPLITS = tuple(XB.TRANSFER_SPLITS)                                         # XF-1–XF-3: 분할 1–5(LGD 와 같다)
XC_SPLITS = tuple(XB.XC_SPLITS)                                            # XC-F3: 분할 201–210
INDEP_KM = float(B1.INDEP_KM)
MIN_UNION = int(XB.MIN_BLOCKS_CI)                                          # 채점 블록 합집합 8
# LG CPU 방법 축의 범위(h51 의 LGD 범위와 같다: n {0, 3, 10, 40, 160, 320, 1000, 전량}, 추출 5, seed 2, λ 0.25·0.5·1.0, catboost_lo, α 1)
LG_N_GRID = "0,3,10,40,160,320,1000,all"
LG_DRAWS, LG_SEEDS = 5, 2
WF4_GRID = (10, 40, 160, -1)                                               # WF4 팔(규칙 W, 진단값)
SMOKE_LG_N_GRID, SMOKE_WF4_GRID = "0,10,all", (10, -1)
RESCALE_RATIO = float(W.RESCALE_RATIO)                                     # 1차 Rescale 실측 비(h54)
# 역할과 표 종류
ROLE_NEW, ROLE_NATL, ROLE_AUG, ROLE_AK = "new_macro", "natl_v5", "augment_L40", "alaska_subtask"
KIND_NEW, KIND_NATL, KIND_NATL_FULL, KIND_AUG = "new_region", "natl_v5", "natl_v5_full", "augment_L40"
KIND_AUG_REF, KIND_AK = "augment_ref", "alaska_subtask"                   # 셀 보강의 같은 작업 기준 표, 알래스카 하위 과제(XC F2 서술 전용)
NATL_ALIAS, NATL_FULL_ALIAS = "NAtlantic~lic~v5", "NAtlantic~v5"
AL7 = "AL-7"                                                               # 새 알래스카 하위 과제(BNZ, ViPER, Anaktuvuk 등 새 셀의 대상 이름)
AL7_ALIAS = f"{AL7}~xf"
NC_PAT = re.compile(r"(?i)(?:\bNC\b|-NC-|non[- ]?commercial)")
BLIND = {KIND_NEW: "맹검", KIND_NATL: "비맹검 부분 포함", KIND_NATL_FULL: "비맹검 부분 포함", KIND_AUG: "비맹검 부분 포함",
         KIND_AUG_REF: "비맹검 부분 포함", KIND_AK: "맹검"}
# LGD 6B.4 의 지리 정의(국가, 위도, 경도. 라벨 미사용). 점 자료의 macro 문자열을 믿지 않고 이 정의로 다시 계산해 대조한다
GEO_6B4 = ("Tibet", "NAtlantic", "Russia_C", "Russia_W", "Russia_E", "Canada", "Lena", "Alaska")
LENA_BOX = tuple(B1.LENA_BOX)                                              # 71.5–73.6°N, 123.3–130.1°E(6B.4 레나 델타 영역)
TIBET_BOX = (26.0, 40.0, 73.0, 105.0)                                      # 청장고원과 치롄산(6B.4)
COUNTRY_ALIAS = {"usa": "united states", "us": "united states", "united states of america": "united states",
                 "united states (alaska)": "united states", "alaska": "united states", "russian federation": "russia",
                 "kalaallit nunaat": "greenland", "svalbard and jan mayen": "svalbard", "people's republic of china": "china", "prc": "china"}
NATL_COUNTRIES = ("greenland", "svalbard", "norway", "sweden", "finland")  # 그린란드, 스발바르, 노르웨이·스웨덴·핀란드 본토
TIBET_COUNTRIES = ("china", "india", "nepal", "bhutan", "pakistan")
# v3 의 점 추정·서술 macro 가운데 국가로 정해지는 것(새 macro 로 세지 않는다. CALM_Mongolia_CAsia, CALM_Alps, GTNPenv_AQ)
V3_COUNTRY_MACRO = {"mongolia": "Mongolia_CAsia", "kazakhstan": "Mongolia_CAsia", "kyrgyzstan": "Mongolia_CAsia", "tajikistan": "Mongolia_CAsia",
                    "switzerland": "Alps", "austria": "Alps", "italy": "Alps", "france": "Alps", "antarctica": "Antarctica"}
NO_NAME = ("", "other", "nan", "none", "unknown")
# build_ext_cells_v1.main 의 셀 표 열(자료가 없을 때 빈 표의 열로 쓴다)
B1_COLS = ["cell_uid", "label_set", "cell_key", "ky", "kx", "lat", "lon", "block", "macro", "subunit", "country", "alt_cm",
           "alt_sd_loc", "alt_sd_all", "n_loc", "n_values", "n_src", "sources", "loc_by_src", "methods", "eos_bases", "value_kinds",
           "obs_months", "drill_months", "year_min", "year_max", "l41a_all_direct3", "l41b_early_single", "l41b_rd_no_day",
           "l41c_dataset_statement", "n_cens_removed", "n_cens_loc_only", "cens_lb_max", "cens_lb_mean", "cens_lb_gt_cell",
           "alt_cm_cens_lb", "n_cens_suspect", "cens_affected", "n_upper_bound", "n_approx", "alt_cm_no_upper",
           "label_subtypes", "q_flags", "n_obs_total", "date_basis", "aliases",
           "v3_f4_cheb_deg", "v3_f4_near_loc", "dup_v3", "v3_rel", "v3_f4_km", "v3_f4_km_loc", "v3_f4_lt1p1km",
           "dist_other_macro_km", "other_macro_nearest", "indep_ok",
           "same_cell_direct", "in_v4", "target", "excl_reason", "license", "lic_unverified", "sites"]
# 라벨 값이 없는 셀 역할 표의 열(v5/cells_xf_roles.csv)
ROLE_COLS = ["cell_uid", "label_set", "cell_key", "lat", "lon", "block", "macro", "geo_macro", "geo_basis", "country", "subunit", "sources",
             "n_loc", "n_src", "methods", "eos_bases", "obs_months", "year_min", "year_max", "license", "lic_unverified", "lic_nc", "dup_v3",
             "v3_f4_km", "dup_v4", "v4_cheb_deg", "merge_v4", "v1_cell_not_v4", "dist_other_macro_km", "other_macro_nearest", "indep_ok",
             "xf_role", "in_v5", "xf_target", "xf_excl", "xf_note"]
CELL_META_KEEP = ("stage", "created", "git_head", "script", "script_sha256", "plan", "inputs", "rules", "n_points_in", "row_disposition",
                  "n_rows_dup_of", "n_locations", "location_range_excluded", "cens_rows_by_set", "elapsed_s")
COV_STAGES = ("dem", "e5", "cci", "sg", "elev")
COV_TAG = "xf_v5"
# XF-1–XF-3 의 대비(같은 라벨 집합 대비: 분할 안 채점 블록 재표집, 1절). 키는 h40 저장소의 곡선 키(h40._g)
XF_HYPS = (("XF-1", "D0-P0", 0), ("XF-2", "R1(0.25)-P1", 10), ("XF-3", "R1(0.25)-P0", -1))
XF4_NS = (10, 40, 160)


def xf_groups(hyp):
    """가설 → (gA, gB). XF-1 = D0(n 0, catboost_lo) − P0, XF-2 = R1(λ 0.25, n 10) − P1(n 10), XF-3 = R1(λ 0.25, 전량) − P0."""
    if hyp == "XF-1":
        return H._g("D0", 0), H.P0_GRP
    if hyp == "XF-2":
        return H._g("R1", 10, XB.LAM_BASE), H._g("P1", 10)
    if hyp == "XF-3":
        return H._g("R1", -1, XB.LAM_BASE), H.P0_GRP
    raise KeyError(hyp)


def xf4_groups(n):
    """XF-4: 규칙 W − R1(λ 0.25)(h54 TUnit 저장소의 키, h54.tests_wf4 와 같다)."""
    return W.gk("W", n, XB.LO, 0.0), W.gk("R1", n, XB.LO, XB.LAM_BASE)


# ================================================================ 경로
class Paths:
    """산출 경로. 기본 뿌리 data/processed/xbatch 아래 XF_new_regions. --out-root 는 XB.check_out_dir 의 허용 뿌리 안이어야 한다."""

    def __init__(self, base=None):
        self.base = Path(base) if base else XB.XBATCH_ROOT
        r = self.root = self.base / EXP_NAME
        self.points = r / "ext_labels"                                    # 새 자료원 파서의 산출 폴더(parse_ext_<id>.py 가 여기에 쓴다)
        self.points_manifest = r / "xf_points_manifest.json"              # 점 자료 목록(이름, sha256). XF 자료 마감 뒤 동결
        self.natl_first = r / "xf_natl_v5_first_count.json"               # NAtlantic v5 첫 세기 기록(한 번만 쓴다)
        self.v5 = r / "v5"
        self.build = self.v5 / "build"
        self.cov = self.v5 / "cov_parts"
        self.v5_table = self.v5 / "fidelity_base_v5.csv"
        self.v5_soil = self.v5 / "e5_soil_tdd_v5.csv"
        self.v5_labels = self.v5 / "fidelity_base_v5_labels_xf.csv"
        self.v5_meta = self.v5 / "fidelity_base_v5_meta.json"
        self.v5_submap = self.v5 / "xf_subregion_map_v5.csv"             # lg_subregion_map_v1 + v5 의 새 블록(최근접 중심 배정)
        self.tdd_v5 = self.v5 / "xf_tdd_matched_v5.csv"                  # v5 새 행의 연도 정합 도일(a2 정의)
        self.tdd_v5_meta = self.v5 / "xf_tdd_matched_v5_meta.json"
        self.cells_cls = self.build / "cells_xf_class.csv"
        self.cells_roles = self.v5 / "cells_xf_roles.csv"
        self.runs = r / "run_tables"
        self.shards = r / "shards"
        self.specs = r / "xf_specs.json"
        self.elig = r / "xf_eligibility.csv"
        self.elig_splits = r / "xf_eligibility_splits.csv"
        self.count = r / "xf_count.csv"
        self.count_sum = r / "xf_count_summary.csv"
        self.count_meta = r / "xf_count_meta.json"
        self.gate = r / "xf_repro_gate.json"
        self.xcf3_csv = r / "xc_f3_list.csv"
        self.xcf3_json = r / "xc_f3_list.json"
        self.build_meta = r / "xf_build_meta.json"
        self.sealed = XB.sealed_dir(EXP_NAME, self.base)

    def check(self):
        XB.check_out_dir(self.root)
        return self


def _gitignore(d, text="*\n!.gitignore\n"):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("# XF 중간 산출: 라벨 값이나 약관 미확인 행이 든다. 커밋하지 않는다(계획 0.3, 2.6).\n" + text)


def now_kst():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))


def on_rescale() -> bool:
    """Rescale 작업 환경 표지(WF_RESCALE=1, h54 와 같이 LG_RESCALE=1 도 받는다). --allow-local 은 로컬 실행이므로 1절 로컬 자원 규칙을
    풀지 않는다(본 실행 허용 표지 XB.run_permitted 와 다르다)."""
    return os.environ.get("WF_RESCALE", "") == "1" or os.environ.get("LG_RESCALE", "") == "1"


def v5_region(macro, role=None):
    """새 행의 v5 region. 알래스카 하위 과제 셀은 AL-7(그 실행 표의 대상 이름), 그 밖은 build_fidelity_base_v4.REGION_OF, 없으면 macro 이름."""
    if role == ROLE_AK:
        return AL7
    return B4.REGION_OF.get(str(macro), str(macro))


def table_macro(macro, role=None):
    """점 자료 macro → v5 region(v5_region) → 실행 표의 macro(fidelity.MACRO_REGION)."""
    reg = v5_region(macro, role)
    return MACRO_REGION.get(reg, reg)


# ================================================================ 1. 점 자료와 셀(build_ext_cells_v1 재사용)
def _norm_country(c) -> str:
    s = str(c).strip().lower()
    if s in ("", "nan", "none"):
        return ""
    return COUNTRY_ALIAS.get(s, s)


def geo_macro(country, lat, lon) -> tuple:
    """LGD 6B.4 의 지리 정의로 macro 를 정한다(국가·좌표만 쓴다. 라벨 미사용). 반환 (macro 또는 '', 근거).
    NAtlantic = 그린란드, 스발바르, 노르웨이·스웨덴·핀란드 본토(덴마크 국적 표기의 서경 10° 서쪽 좌표 = 그린란드 포함). Canada = 캐나다.
    러시아 = 레나 델타 영역 안 Lena, 90°E 미만 Russia_W, 90°E 이상 140°E 미만 Russia_C, 140°E 이상과 서경 Russia_E.
    미국의 알래스카 좌표(51°N 이상, 129°W 서쪽 또는 170°E 동쪽) = Alaska. 청장고원 상자(26–40°N, 73–105°E) 안의 중국·인도·네팔·부탄·파키스탄
    = Tibet. 6B.4 밖이지만 국가로 정해지는 v3 macro(몽골·중앙아시아, 알프스, 남극)는 그 v3 macro 다. 그 밖은 ''(새 macro 후보)."""
    c = _norm_country(country)
    la, lo = float(lat), float(lon)
    if c in NATL_COUNTRIES or (c == "denmark" and lo < -10.0):
        return "NAtlantic", "6B.4 국가(그린란드·스발바르·노르웨이·스웨덴·핀란드)"
    if c == "canada":
        return "Canada", "6B.4 국가(캐나다)"
    if c == "russia":
        if LENA_BOX[0] <= la <= LENA_BOX[1] and LENA_BOX[2] <= lo <= LENA_BOX[3]:
            return "Lena", "6B.4 레나 델타 영역"
        if 0.0 <= lo < 90.0:
            return "Russia_W", "6B.4 러시아 90°E 미만"
        if 90.0 <= lo < 140.0:
            return "Russia_C", "6B.4 러시아 90°E 이상 140°E 미만"
        return "Russia_E", "6B.4 러시아 140°E 이상·서경"
    if c == "united states" and la >= 51.0 and (lo <= -129.0 or lo >= 170.0):
        return "Alaska", "6B.4 미국 알래스카 좌표"
    if c in TIBET_COUNTRIES and TIBET_BOX[0] <= la <= TIBET_BOX[1] and TIBET_BOX[2] <= lo <= TIBET_BOX[3]:
        return "Tibet", "6B.4 청장고원 상자"
    if c in V3_COUNTRY_MACRO:
        return V3_COUNTRY_MACRO[c], "v3 macro(국가)"
    return "", "6B.4·v3 정의 밖"


def cell_geo(country, lat, lon) -> tuple:
    """셀(또는 점)의 지리 macro. country 는 ';' 로 이어진 목록일 수 있다. 반환 (macro, 근거, 오류). 국가가 없거나 국가마다 macro 가 다르면 오류."""
    cs = [v for v in str(country).split(";") if _norm_country(v)]
    if not cs:
        return "", "", "국가(country)가 비었다(6B.4 지리 정의를 적용할 수 없다)"
    if not (np.isfinite(float(lat)) and np.isfinite(float(lon))):
        return "", "", "좌표가 없다"
    got = sorted({geo_macro(v, lat, lon) for v in cs})
    if len({g[0] for g in got}) > 1:
        return "", "", f"국가 {cs} 의 지리 macro 가 서로 다르다({[g[0] for g in got]})"
    return got[0][0], got[0][1], ""


def role_of(macro, known, geo="") -> tuple:
    """셀 역할(라벨 미사용). geo = 지리 macro(cell_geo). 반환 (역할, 오류 문자열).
    지리 macro 가 있으면 점 자료 macro 가 그와 같아야 한다. NAtlantic = 민감도(natl_v5), Alaska = 하위 과제(AL-7, XC F2 서술),
    그 밖의 6B.4·v3 macro = 셀 보강(L40 형식). 지리 macro 가 없으면(6B.4·v3 밖) 점 자료 macro 가 6B.4·v3·v4 어디에도 없는 이름일 때만
    새 macro 후보(이번 조사 뒤 처음 확보)다."""
    m = str(macro).strip()
    if geo:
        if m != geo:
            return "", f"점 자료 macro '{m}' 가 6B.4 지리 정의 '{geo}' 와 다르다"
        if geo == "NAtlantic":
            return ROLE_NATL, ""
        if geo == "Alaska":
            return ROLE_AK, ""
        return ROLE_AUG, ""
    if m.lower() in NO_NAME:
        return "", "6B.4·v3 정의 밖의 셀에는 새 macro 이름이 필요하다(macro 가 비었거나 other)"
    if m in known or table_macro(m) in known or m in GEO_6B4:
        return "", f"점 자료 macro '{m}' 는 기존 macro 이름인데 좌표·국가가 그 지리 정의 밖이다"
    return ROLE_NEW, ""


def check_point_macros(points, known) -> int:
    """셀 조립 전에 점 자료의 macro 를 6B.4 지리 정의와 대조한다(src_id, site_id, country, macro, lat, lon 열만 읽는다. 라벨 미사용).
    어긋나는 행이 있으면 멈춘다. 반환 점검한 행 수."""
    bad, n = [], 0
    for p in points:
        d = pd.read_csv(p, usecols=["src_id", "site_id", "country", "macro", "lat", "lon"], dtype=str, keep_default_na=False)
        for sid, site, c, m, la, lo in zip(d.src_id, d.site_id, d.country, d.macro, d.lat, d.lon):
            try:
                la_, lo_ = float(la), float(lo)
            except ValueError:
                continue                                                       # 좌표 없는 행은 셀이 되지 않는다(B1 규칙)
            n += 1
            gm, _, err = cell_geo(c, la_, lo_)
            if not err:
                err = role_of(m, known, gm)[1]
            if err:
                bad.append(f"{sid}:{site}({err})")
    if bad:
        raise SystemExit(f"[cells] 점 자료 macro 가 LGD 6B.4 지리 정의와 맞지 않는 행 {len(bad)}개: {sorted(set(bad))[:20]}. "
                         "파서의 macro·country 를 고친다(XF 역할은 지리 정의로 정한다)")
    return n


def v1_points(meta=V1_META) -> dict:
    """v4 조립(ext_cells_v1)에 쓴 점 자료 파일 → sha256."""
    return dict(json.loads(Path(meta).read_text())["inputs"]["points"])


def xf_point_files(P, v1=None) -> list:
    """새 자료원의 점 자료: <산출>/ext_labels/*_points.csv 만 읽는다(새 파서는 이 폴더에 쓴다. 기존 data/processed/ext_labels 에는 쓰지 않는다).
    v4 조립에 쓴 점 자료와 같은 이름이면 멈춘다(새 자료원만 받는다)."""
    v1 = v1_points() if v1 is None else v1
    out = sorted(Path(P.points).glob("*_points.csv")) if Path(P.points).exists() else []
    same = [p.name for p in out if p.name in v1]
    if same:
        raise SystemExit(f"[cells] v4 조립에 쓴 점 자료와 같은 이름이다: {same}. 새 자료원의 src_id 를 쓴다")
    return out


def points_manifest(P, files, now=None) -> dict:
    """점 자료 목록(이름 → sha256)의 기록과 동결(5절 'XF 자료 마감 T0 + 7일'). 마감 전에는 새 파일과 바뀐 파일을 적는다. 마감 뒤 첫 실행에서
    동결한다(마감 전 기록이 없으면 수정 시각이 마감 이전인 파일로 동결). 동결 뒤에는 목록 밖이거나 해시가 바뀐 파일을 거부한다(SystemExit).
    반환 기록 dict."""
    now = now or now_kst()
    ts = now.isoformat(timespec="seconds")
    cur = {Path(p).name: XB.sha256_file(p) for p in files}
    doc = json.loads(P.points_manifest.read_text()) if P.points_manifest.exists() else dict(files={}, frozen=False)
    doc.setdefault("files", {})
    if not doc.get("frozen"):
        if now <= DEADLINE_DATA:
            for k, v in cur.items():
                e = doc["files"].get(k)
                if e is None:
                    doc["files"][k] = dict(sha256=v, first_seen=ts, updated=ts)
                elif e.get("sha256") != v or e.get("removed"):
                    e.update(sha256=v, updated=ts)
                    e.pop("removed", None)
            for k, e in doc["files"].items():
                if k not in cur and not e.get("removed"):
                    e["removed"] = ts
            doc["last_update"] = ts
        else:
            if not doc["files"]:
                mt = {Path(p).name: dt.datetime.fromtimestamp(Path(p).stat().st_mtime, dt.timezone.utc) for p in files}
                doc["files"] = {k: dict(sha256=v, first_seen="", updated="", basis="mtime") for k, v in cur.items() if mt[k] <= DEADLINE_DATA}
                doc["freeze_basis"] = "마감 전 기록 없음: 수정 시각이 마감 이전인 파일로 동결"
            else:
                doc["freeze_basis"] = "마감 전 마지막 기록"
            doc.update(frozen=True, frozen_at=ts)
    doc.update(deadline_data=DEADLINE_DATA.isoformat(), rule="5절 'XF 자료 T0 + 7일'. 동결 뒤 목록 밖·해시가 바뀐 점 자료는 거부한다")
    XB.check_out_dir(P.root)
    P.root.mkdir(parents=True, exist_ok=True)
    XB._atomic_write(P.points_manifest, json.dumps(doc, ensure_ascii=False, indent=1))
    if doc.get("frozen"):
        allowed = {k: e["sha256"] for k, e in doc["files"].items() if not e.get("removed")}
        bad = sorted(k for k, v in cur.items() if allowed.get(k) != v)
        if bad:
            raise SystemExit(f"[cells] XF 자료 마감({DEADLINE_DATA.isoformat()}) 뒤 동결 목록 밖이거나 바뀐 점 자료: {bad}. "
                             "마감 뒤 자료는 v5 에 넣지 않는다(5절). 파일을 ext_labels 밖으로 옮긴다")
    return doc


def v1_points_unchanged(ext_dir=EXT) -> dict:
    """v4 조립의 점 자료가 지금도 같은 파일인가(sha256 대조, 세기 범주)."""
    v1 = v1_points()
    return {k: (Path(ext_dir) / k).exists() and XB.sha256_file(Path(ext_dir) / k) == v for k, v in v1.items()}


LABEL_KEY = re.compile(r"(?:^|_)alt(?:_|$)")                               # alt_mean, alt_a, alt_cens_lb 같은 라벨 통계 키(firealt_ 같은 자료원 이름은 아니다)


def _strip_label_stats(obj):
    """라벨 통계 항목(키가 LABEL_KEY 에 맞는 항목)을 지운 사본."""
    if isinstance(obj, dict):
        return {k: _strip_label_stats(v) for k, v in obj.items() if not LABEL_KEY.search(str(k))}
    if isinstance(obj, list):
        return [_strip_label_stats(v) for v in obj]
    return obj


def empty_cells() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object if c in ("cell_uid", "label_set", "cell_key", "macro", "license", "sources") else float)
                         for c in B1_COLS})


SENTINEL_SRC = "_xf_sentinel"


def _write_sentinel(work):
    """build_ext_cells_v1.main 은 절단 행이 하나도 없으면 절단 고아 위치 목록(395행)에서 KeyError 로 멈춘다(빈 목록 색인이 열 선택이 된다).
    셀을 만들 수 없는 절단 행 하나(적도 태평양, 우측 절단)를 넣어 그 경로를 지나게 한다. 이 행은 위치·셀·라벨 값에 들어가지 않는다
    (절단 행은 셀 값에서 빠지고, 같은 위치의 비절단 행이 없어 고아 목록에만 남는다). 메타에서 지운다."""
    row = dict(src_id=SENTINEL_SRC, site_id="S0", site_name="S0", lat=0.0, lon=-150.0, year=2000, month=9, alt_cm=100.0, method="probe",
               label_def="direct_eos", n_obs=1, country="", macro="other", citation="", license="sentinel", subunit="", date="2000-09-01",
               eos_basis="record_date", value_kind="single_visit", year_min="", year_max="", alt_sd_cm="", coord_prec_deg="", disturbed=0,
               disturb_type="", right_censored=1, orig_source="", dup_of="", qc_flag="ok", notes="x_new_regions sentinel")
    pd.DataFrame([row]).to_csv(Path(work) / f"{SENTINEL_SRC}_points.csv", index=False)


def run_b1(points, work_dir, v3_dir=None, sentinel=True):
    """build_ext_cells_v1.main 을 그대로 부른다. 입력 = work_dir 안의 점 자료 기호 연결, 산출 = work_dir(ext_cells_v1*.csv 이름 그대로).
    v3_dir 를 주면 v3 표 폴더(B1.PROC)도 바꾼다(시험). 화면 출력은 버리고 메타는 라벨 통계 항목을 지운 cells_xf_meta.json 으로 바꾼다.
    sentinel 이면 절단 행 하나를 더한다(_write_sentinel). 반환 (셀 표, 메타)."""
    work = Path(work_dir)
    XB.check_out_dir(work)
    _gitignore(work)
    for q in work.glob("*_points.csv"):
        q.unlink()
    if not points:
        meta = dict(n_points_in=0, note="새 점 자료 없음")
        (work / "cells_xf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        return empty_cells(), meta
    for p in points:
        (work / Path(p).name).symlink_to(os.path.realpath(p))
    if sentinel:
        _write_sentinel(work)
    old = (B1.EXT, B1.PROC)
    B1.EXT = work
    if v3_dir is not None:
        B1.PROC = Path(v3_dir)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            B1.main()
    finally:
        B1.EXT, B1.PROC = old
    meta_p = work / "ext_cells_v1_meta.json"
    raw = json.loads(meta_p.read_text())
    meta = {k: raw[k] for k in CELL_META_KEEP if k in raw}
    meta["cells"] = _strip_label_stats(raw.get("cells", {}))
    if sentinel:
        meta["n_points_in"] = int(meta.get("n_points_in", 1)) - 1
        meta["row_disposition"] = [r for r in meta.get("row_disposition", []) if r.get("src_id") != SENTINEL_SRC]
        meta.get("inputs", {}).get("points", {}).pop(f"{SENTINEL_SRC}_points.csv", None)
        cr = dict(meta.get("cens_rows_by_set", {}))
        if cr.get("direct"):
            cr["direct"] = int(cr["direct"]) - 1
        meta["cens_rows_by_set"] = {k: v for k, v in cr.items() if v}
        meta["sentinel"] = "절단 행 하나(적도 태평양)를 넣어 build_ext_cells_v1 의 빈 절단 목록 경로를 피했다. 위 수에서 뺐다"
        (work / f"{SENTINEL_SRC}_points.csv").unlink(missing_ok=True)
    meta["n_stdout_lines_discarded"] = len(buf.getvalue().splitlines())
    meta["note"] = ("build_ext_cells_v1.main 의 화면 출력과 라벨 통계 항목(셀 평균, 절단 요약, 값 경계 점검, 근접 쌍의 값)은 계획 0.3·2.6 에 따라 "
                    "버렸다. 셀 값은 ext_cells_v1.csv(이 폴더, 커밋 제외)에만 있다")
    meta_p.unlink()
    (work / "cells_xf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    cells = pd.read_csv(work / "ext_cells_v1.csv", low_memory=False)
    return cells, meta


def known_macros(table=V4_TABLE) -> set:
    """기존 표(v4: v3 행과 LGD 새 행)의 macro 목록. 새 macro 판정의 기준이다(라벨 미사용)."""
    t = pd.read_csv(table, usecols=["region"], low_memory=False)
    return set(MACRO_REGION.get(r, r) for r in t.region.astype(str).unique()) | set(B1.TARGET_MACROS) | {"Alaska", "Lena"}


def classify_cells(cells, v4_cells, known) -> pd.DataFrame:
    """B1 셀 표에 v5 표지를 더한다(국가·좌표·셀 색인·약관 문자열만 쓴다. 라벨 값 미사용).
    - 지리 macro(cell_geo, LGD 6B.4)를 다시 계산해 점 자료 macro 와 대조하고, 어긋나면 멈춘다. 역할은 지리 macro 로 정한다(role_of).
    - merge_v4 = v4 에 들어간 셀(in_v4 = 1)과 같은 cell_uid(그 v4 행이 바뀌어야 하므로 뺀다). v4 에 들어가지 않은 v1 셀과 같은 cell_uid 는
      v1_cell_not_v4 로 표시만 한다(v4 행이 바뀌지 않는다. 셀 값은 새 점 자료만으로 정해진다는 사실을 xf_note 에 적는다).
    - xf_target = in_v5 ∩ 독립성 통과(새 macro, NAtlantic, 셀 보강, 알래스카 하위 과제 모두)."""
    c = (cells if len(cells) else empty_cells()).copy().reset_index(drop=True)
    v4_in = v4_cells[pd.to_numeric(v4_cells.in_v4, errors="coerce").fillna(0).astype(int) == 1]
    v4d = v4_in[v4_in.label_set.astype(str) == "direct"]
    if len(c) and len(v4d):
        dd, jj = cKDTree(v4d[["lat", "lon"]].values.astype(float)).query(c[["lat", "lon"]].values.astype(float), p=np.inf)
        c["v4_cheb_deg"] = np.round(dd, 6)
        c["v4_near_cell"] = v4d.cell_uid.astype(str).values[jj]
    else:
        c["v4_cheb_deg"] = np.inf
        c["v4_near_cell"] = ""
    c["dup_v4"] = (c.v4_cheb_deg.astype(float) <= float(B1.DUP_DEG)).astype(int)
    uid = c.cell_uid.astype(str)
    in4 = set(v4_in.cell_uid.astype(str))
    not4 = v4_cells[~v4_cells.cell_uid.astype(str).isin(in4)]
    not4_why = dict(zip(not4.cell_uid.astype(str), not4.excl_reason.fillna("").astype(str) if "excl_reason" in not4 else [""] * len(not4)))
    c["merge_v4"] = uid.isin(in4).astype(int)
    c["v1_cell_not_v4"] = uid.isin(set(not4_why)).astype(int)
    c["lic_nc"] = c.license.fillna("").astype(str).str.contains(NC_PAT).astype(int)
    geo, basis, roles, errs = [], [], [], []
    ctry = c.country if "country" in c.columns else pd.Series([""] * len(c))
    for m, cc, la, lo, u in zip(c.macro, ctry, c.lat, c.lon, uid):
        gm, gb, err = cell_geo(cc, la, lo)
        rl, err2 = role_of(m, known, gm) if not err else ("", err)
        if err or err2:
            errs.append(f"{u}({err or err2})")
        geo.append(gm); basis.append(gb); roles.append(rl)
    if errs:
        raise SystemExit(f"[cells] 셀 macro 가 LGD 6B.4 지리 정의와 맞지 않는다 {len(errs)}개: {errs[:20]}. 파서의 macro·country 를 고친다")
    c["geo_macro"], c["geo_basis"], c["xf_role"] = geo, basis, roles
    c["table_macro"] = [table_macro(m, r) for m, r in zip(c.macro, c.xf_role)]
    direct = c.label_set.astype(str) == "direct"
    dup3 = pd.to_numeric(c.dup_v3, errors="coerce").fillna(0).astype(int)
    indep = pd.to_numeric(c.indep_ok, errors="coerce").fillna(0).astype(int)
    c["in_v5"] = (direct & (dup3 == 0) & (c.dup_v4 == 0) & (c.merge_v4 == 0) & (c.lic_nc == 0)).astype(int)
    c["xf_target"] = ((c.in_v5 == 1) & (indep == 1) & c.xf_role.isin([ROLE_NEW, ROLE_NATL, ROLE_AUG, ROLE_AK])).astype(int)
    why, note = [], []
    for r, d3, ind, u in zip(c.itertuples(index=False), dup3, indep, uid):
        w, nt = [], []
        if r.label_set != "direct":
            w.append(f"label_set={r.label_set}")
        if d3:
            w.append("dup_v3")
        if r.dup_v4:
            w.append(f"dup_v4:{r.v4_near_cell}")
        if r.merge_v4:
            w.append("merge_v4(v4 행 불변 규칙)")
        if r.lic_nc:
            w.append("license_nc(공개 v5 표 제외)")
        if not ind:
            w.append(f"indep<{INDEP_KM:.0f}km")
        if r.v1_cell_not_v4:
            nt.append(f"v1_cell_not_v4({not4_why.get(u, '') or 'v4 밖'}; v4 행 없음, 셀 값은 새 점 자료만)")
        if r.xf_role == ROLE_AK:
            nt.append(f"alaska_subtask({AL7}, XC F2 서술 전용, XD 개발 과제 아님)")
        why.append(";".join(w))
        note.append(";".join(nt))
    c["xf_excl"] = why
    c["xf_note"] = note
    return c


# ================================================================ 2. 공변량(ext_cells_covariates_v1 재사용)
def _mirror(src, dst, copy_names=()):
    """dst 에 src 의 항목을 기호 연결로 비춘다(copy_names 는 복사). 이미 있는 항목은 그대로 둔다."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return 0
    n = 0
    for p in src.iterdir():
        q = dst / p.name
        if q.exists() or q.is_symlink():
            continue
        if p.name in copy_names:
            shutil.copy2(p, q)
        else:
            q.symlink_to(p.resolve())
        n += 1
    return n


def run_cov(P, cells, fetch=False, stages=COV_STAGES, cov_mod=None, overlays=(DEM_OVERLAY, SG_OVERLAY)):
    """in_v5 셀의 공변량. ext_cells_covariates_v1 의 단계 함수를 그대로 부르고 전역 변수(PARTS, DEM, SG_RAW, fetch_tile)만 바꾼다.
    SoilGrids 새 창 이름이 기존 창과 겹치지 않게 macro 앞에 'xf5_' 를 붙인다(창 이름에만 쓰인다). 반환 공변량 표(cell_uid 키)."""
    COV = cov_mod or importlib.import_module("ext_cells_covariates_v1")
    XB.check_out_dir(P.cov)
    _gitignore(P.cov)
    sel = cells[cells.in_v5 == 1].reset_index(drop=True)
    out_p = P.cov / "cells_xf_cov.csv"
    if not len(sel):
        o = pd.DataFrame(columns=["id", "cell_uid", "lat", "lon", "macro"])
        o.to_csv(out_p, index=False)
        return o
    dem_ov, sg_ov = (Path(v) for v in overlays)
    for d in (dem_ov, sg_ov):
        XB.check_out_dir(d, allowed=[d])
    _mirror(COV.DEM, dem_ov)
    _mirror(COV.SG_RAW, sg_ov, copy_names=("windows_wcs_meta_v4.json",))
    old = dict(PARTS=COV.PARTS, DEM=COV.DEM, SG_RAW=COV.SG_RAW, fetch_tile=COV.fetch_tile)
    COV.PARTS, COV.DEM, COV.SG_RAW = P.cov, dem_ov, sg_ov
    if not fetch:
        COV.fetch_tile = lambda tlat, tlon: ("exists" if (dem_ov / (COV.tname(tlat, tlon) + ".tif")).exists() else "not_fetched")
    cell = pd.DataFrame(dict(id=("xf|" + sel.cell_uid.astype(str)).values, lat=sel.lat.values.astype(float), lon=sel.lon.values.astype(float),
                             macro=("xf5_" + sel.macro.astype(str)).values))
    log, timing = io.StringIO(), {}
    try:
        with contextlib.redirect_stdout(log):
            if fetch:
                COV.stage_tiles(cell, COV_TAG)
            for st in stages:
                t1 = time.time()
                if st == "sg":
                    COV.stage_sg(cell, COV_TAG, allow_download=bool(fetch))
                else:
                    getattr(COV, f"stage_{st}")(cell, COV_TAG)
                timing[st] = round(time.time() - t1, 1)
            o = COV.merge(cell, COV_TAG)
    finally:
        COV.PARTS, COV.DEM, COV.SG_RAW, COV.fetch_tile = old["PARTS"], old["DEM"], old["SG_RAW"], old["fetch_tile"]
    if "e5_tdd_soil" in o:
        le0 = (pd.to_numeric(o.e5_tdd_soil, errors="coerce") <= 0).fillna(False)
        o.loc[le0, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan                  # v3 규칙(ext_cells_covariates_v1 merge 와 같다)
    o["macro"] = sel.macro.values
    o["cell_uid"] = sel.cell_uid.astype(str).values
    o.to_csv(out_p, index=False)
    (P.cov / "cov_log.txt").write_text(log.getvalue())
    (P.cov / "cov_meta.json").write_text(json.dumps(dict(created=now_kst().isoformat(timespec="seconds"), n_cells=int(len(o)), stages=list(stages),
                                                         fetch=bool(fetch), timing_s=timing, overlays=[str(dem_ov), str(sg_ov)]),
                                                    ensure_ascii=False, indent=1))
    return o


# ================================================================ 3. v5 표(build_fidelity_base_v4 의 새 행 규약)
def _header(path):
    with open(path, encoding="utf-8") as f:
        return f.readline().rstrip("\n").split(",")


def v5_rows(new, cols):
    """새 셀 → v4 형식 45열 행(build_fidelity_base_v4 152–178행과 같은 규약). region = v5_region(macro, 역할): REGION_OF(macro) 또는 macro,
    알래스카 하위 과제 셀은 AL-7."""
    nr = pd.DataFrame(index=new.index)
    nr["loc_id"] = new.loc_id.astype(int)
    nr["lat"], nr["lon"] = new.lat.astype(float).round(6), new.lon.astype(float).round(6)
    roles = new.xf_role if "xf_role" in new.columns else [None] * len(new)
    nr["region"] = [v5_region(m, r) for m, r in zip(new.macro, roles)]
    nr["block"] = (np.floor(nr.lat / 0.5).astype(int) * 100000 + np.floor(nr.lon / 0.5).astype(int))
    nr["alt_cm"] = new.alt_cm
    nr["source_id"] = "F4_ext_direct"
    nr["fidelity_level"] = 4
    nr["spatial_support_m"] = 1000.0
    nr["sigma_prior_cm"] = pd.to_numeric(new.alt_sd_all, errors="coerce").fillna(12.0).clip(3, 40)
    nr["right_censored"] = 0
    for c in cols:
        if c in nr.columns:
            continue
        if c in new.columns:
            nr[c] = new[c].values
        elif c in ("insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n", "polsar_alt", "polsar_std"):
            nr[c] = np.nan
        elif c == "polsar_valid":
            nr[c] = 0.0
        elif c == "insar_miss":
            nr[c] = 1
        else:
            raise KeyError(f"v5 새 행에 열 {c} 를 만들 수 없다(공변량 단계 산출 확인)")
    return nr[cols]


def build_v5(P, cells, cov, v4_table=V4_TABLE, v4_soil=V4_SOIL):
    """v5 = v4 원문 줄 + 새 행. 토양 도일 v5 = v4 원문 줄 + 새 행. 새 행 부가 표(라벨 값 열 없음). v4 줄 바이트 불변을 단언한다."""
    XB.check_out_dir(P.v5)
    P.v5.mkdir(parents=True, exist_ok=True)
    cols, scols = _header(v4_table), _header(v4_soil)
    new = cells[cells.in_v5 == 1].copy()
    if len(new):
        cv = cov.drop(columns=[c for c in ("id", "lat", "lon", "macro") if c in cov.columns])
        new = new.merge(cv, on="cell_uid", how="left", validate="one_to_one")
        new = new.sort_values(["macro", "ky", "kx"]).reset_index(drop=True)
    new["loc_id"] = LOC0_V5 + np.arange(len(new))
    nr = v5_rows(new, cols) if len(new) else pd.DataFrame(columns=cols)
    sn = pd.DataFrame(dict(loc_id=new.loc_id, lat=nr.lat if len(new) else [], lon=nr.lon if len(new) else [], src="xf_v5"))
    for c in scols:
        if c not in sn.columns:
            sn[c] = new[c].values if (c in new.columns and len(new)) else np.nan
    if len(new):
        zero = (pd.to_numeric(sn.e5_tdd_soil, errors="coerce") <= 0).fillna(False)
        sn.loc[zero, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan              # v3 규칙
    sn = sn[scols]
    B4.append_rows_text(Path(v4_table), P.v5_table, nr)
    B4.append_rows_text(Path(v4_soil), P.v5_soil, sn)
    l4 = Path(v4_table).read_text(encoding="utf-8").splitlines()
    l5 = P.v5_table.read_text(encoding="utf-8").splitlines()
    s4 = Path(v4_soil).read_text(encoding="utf-8").splitlines()
    s5 = P.v5_soil.read_text(encoding="utf-8").splitlines()
    ok_t, ok_s = l5[:len(l4)] == l4, s5[:len(s4)] == s4
    assert ok_t and ok_s, "v5 의 v4 원문 줄이 바뀌었다(2.6 재현 관문)"
    roles = new.xf_role if "xf_role" in new.columns else [None] * len(new)
    lab = pd.DataFrame(dict(loc_id=new.loc_id.astype(int), part="xf", region_v5=nr.region if len(new) else [],
                            macro_v5=[table_macro(m, r) for m, r in zip(new.macro, roles)], source_id="F4_ext_direct",
                            lat=nr.lat if len(new) else [], lon=nr.lon if len(new) else [], block=nr.block if len(new) else []))
    for c in ("xf_role", "xf_target", "label_set", "cell_uid", "cell_key", "macro", "geo_macro", "subunit", "country", "sources", "loc_by_src", "n_src",
              "n_loc", "n_values", "methods", "eos_bases", "value_kinds", "obs_months", "year_min", "year_max", "l41a_all_direct3",
              "l41b_early_single", "l41c_dataset_statement", "cens_affected", "q_flags", "license", "lic_unverified", "lic_nc",
              "dist_other_macro_km", "other_macro_nearest", "indep_ok", "v3_f4_km", "v4_cheb_deg", "v1_cell_not_v4", "xf_note", "e5_fallback_deg",
              "soil_fallback_deg", "cci_n_years", "dem_tile_missing", "e5_grid_sd_min_m", "sg_windows"):
        lab[c] = new[c].values if c in new.columns else np.nan
    lab["e5_glacier_grid"] = (pd.to_numeric(lab.e5_grid_sd_min_m, errors="coerce") >= float(B4.GLACIER_SD_M)).astype(int)
    lab.to_csv(P.v5_labels, index=False)
    feats = [c for c in SHARED_CORE if c in nr.columns]
    comp = {}
    if len(new):
        for (role, mac), g in nr.assign(_r=new.xf_role.values, _m=new.macro.values).groupby(["_r", "_m"]):
            soil_ok = pd.to_numeric(sn.loc[g.index, "e5_sqrt_tdd_soil"], errors="coerce").notna()
            comp[f"{role}|{mac}"] = dict(n=int(len(g)), n_blocks=int(g.block.nunique()), x25_all=round(float(g[feats].notna().all(axis=1).mean()), 4),
                                         cci_valid=round(float(pd.to_numeric(g.cci_valid, errors="coerce").fillna(0).mean()), 4),
                                         soil_tdd=round(float(soil_ok.mean()), 4),
                                         eval_ready=int((g.cci_alt.notna().values & soil_ok.values).sum()))
    meta = dict(stage="XF v5 조립(계획 2.6)", created=now_kst().isoformat(timespec="seconds"), script=str(Path(__file__).relative_to(ROOT)),
                script_sha256=XB.sha256_file(__file__), plan=f"{XB.PLAN_DOC}@{XB.PLAN_COMMIT}",
                inputs=dict(v4=dict(path=str(v4_table), sha256=XB.sha256_file(v4_table)), soil_v4=dict(path=str(v4_soil), sha256=XB.sha256_file(v4_soil))),
                outputs={p.name: XB.sha256_file(p) for p in (P.v5_table, P.v5_soil, P.v5_labels)},
                n_rows=dict(v4=len(l4) - 1, new=int(len(new)), total=len(l5) - 1, soil_v4=len(s4) - 1, soil_new=int(len(sn))),
                v4_rows_bytes_equal=bool(ok_t), soil_v4_rows_bytes_equal=bool(ok_s),
                new_by_role=dict(Counter(new.xf_role)) if len(new) else {}, new_by_macro=dict(Counter(new.macro)) if len(new) else {},
                covariate_completeness=comp,
                conventions=dict(loc_id=f"새 행 loc_id = {LOC0_V5} + 일련번호(macro, ky, kx 순)", source_id="F4_ext_direct(실행 표가 대상 행만 F4_direct 로 바꾼다)",
                                 region=f"build_fidelity_base_v4.REGION_OF, 없으면 macro 이름. 알래스카 하위 과제 셀은 {AL7}", spatial_support_m=1000, sigma_prior_cm="alt_sd_all 을 3–40 으로 자름, 없으면 12",
                                 soil="loc_id 키, e5_tdd_soil <= 0 이면 결측(v3 규칙)", in_v5="직접 라벨 ∩ dup_v3·dup_v4·merge_v4·비상업 약관 아님"))
    P.v5_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    return meta


def extended_submap(v5_table, submap=SUBMAP_V1) -> pd.DataFrame:
    """lg_subregion_map_v1(블록 → 하위 지역, 라벨 미사용)에 v5 의 새 블록을 더한 표. 대응표의 parent(Alaska, Canada, Lena) macro 에 속하는
    v5 행의 블록 가운데 대응표에 없는 블록은 블록 중심(그 블록 v5 행 좌표의 평균)에서 대권 거리가 가장 가까운 같은 parent 의 하위 지역
    중심(대응표 블록 중심의 셀 수 가중 평균)에 배정한다(h25·h40 의 블록 중심 k-means 와 같은 최근접 규칙). 원래 행은 바꾸지 않는다.
    h40.apply_subregion_map 은 표에 없는 블록이 있으면 멈추므로 실행 표마다 이 표를 둔다. 반환 열 parent, block, subregion, n_cells, lat, lon, extra."""
    mp = pd.read_csv(submap, dtype=dict(parent=str, subregion=str))
    mp["block"] = mp.block.astype(int)
    t = pd.read_csv(v5_table, usecols=["loc_id", "lat", "lon", "region"], low_memory=False)
    t["macro"] = [MACRO_REGION.get(r, r) for r in t.region.astype(str)]
    t = t[t.macro.isin(set(mp.parent))].copy()
    t["block"] = np.floor(t.lat / 0.5).astype(int) * 100000 + np.floor(t.lon / 0.5).astype(int)
    have = set(zip(mp.parent, mp.block))
    cen = {}
    for (par, sub), g in mp.groupby(["parent", "subregion"]):
        w = g.n_cells.astype(float).values
        cen[(par, sub)] = (float(np.average(g.lat.astype(float), weights=w)), float(np.average(g.lon.astype(float), weights=w)))
    extra = []
    for (par, blk), g in t.groupby(["macro", "block"]):
        if (par, int(blk)) in have:
            continue
        la, lo = float(g.lat.mean()), float(g.lon.mean())
        cands = sorted((float(haversine_km(la, lo, np.array([c[0]]), np.array([c[1]]))[0]), sub) for (p_, sub), c in cen.items() if p_ == par)
        extra.append(dict(parent=par, block=int(blk), subregion=cands[0][1], n_cells=int(len(g)), lat=la, lon=lo, extra=1))
    out = pd.concat([mp.assign(extra=0), pd.DataFrame(extra, columns=list(mp.columns) + ["extra"])], ignore_index=True)
    return out[["parent", "block", "subregion", "n_cells", "lat", "lon", "extra"]]


def sub_of(submap, parent, block) -> str:
    """(parent, block) → 하위 지역 이름(extended_submap 표). 없으면 ''."""
    q = submap[(submap.parent == str(parent)) & (submap.block.astype(int) == int(block))]
    return str(q.subregion.iloc[0]) if len(q) else ""


def write_tdd_v5(P, n_check=400, xs=None):
    """v5 새 행(loc_id ≥ 30000)의 연도 정합 도일(a2_year_matched_tdd.py 2) 단계 정의, 계획 2.6 '새 셀의 연도 정합 도일'). XB 의 --write-tdd-v4 와
    같은 계산(x_multisource_stacking.a2_tdd_cells)을 고치지 않고 부르고, 같은 코드로 v3 셀 n_check 개를 다시 계산해 lgx_tdd_matched_v1 과 대조한다
    (최대 |차| ≤ 1e-6 °C·day, 표지 일치). 관측 연도는 v5 새 행 부가 표의 year_min, year_max 다. 라벨 값은 읽지 않는다. 새 행이 없으면 빈 표."""
    cols = ["loc_id", "tdd_matched", "match_flag", "n_years_matched", "match_lo", "match_hi", "fallback_deg", "region", "part"]
    t = pd.read_csv(P.v5_table, usecols=["loc_id", "lat", "lon", "region"], low_memory=False)
    new = t[t.loc_id >= LOC0_V5].drop_duplicates("loc_id").set_index("loc_id")
    check = dict(n=0, max_abs_diff=None, flag_equal=None)
    if len(new):
        XS = xs or importlib.import_module("x_multisource_stacking")
        yr = pd.read_csv(P.v5_labels, usecols=["loc_id", "year_min", "year_max"]).drop_duplicates("loc_id").set_index("loc_id")
        new = new.join(yr, how="left")
        if new.year_min.isna().any() or new.year_max.isna().any():
            raise SystemExit(f"[tdd] 관측 연도가 없는 v5 새 행 {int(new.year_min.isna().sum())}개")
        a2c = pd.read_csv(XS.A2_CELLS, usecols=["loc_id", "lat", "lon", "year_min", "year_max", "match_flag"]).drop_duplicates("loc_id").set_index("loc_id")
        v1 = pd.read_csv(TDD_V1, usecols=["loc_id", "tdd_matched"]).drop_duplicates("loc_id").set_index("loc_id").tdd_matched
        rng = np.random.RandomState(XB.seed_of("xf-tdd-check"))
        chk = a2c.iloc[np.sort(rng.choice(len(a2c), min(int(n_check), len(a2c)), replace=False))] if n_check else a2c.iloc[:0]
        res = XS.a2_tdd_cells(np.r_[new.lat.values, chk.lat.values], np.r_[new.lon.values, chk.lon.values],
                              np.r_[new.year_min.values, chk.year_min.values], np.r_[new.year_max.values, chk.year_max.values])
        rn, rc = res.iloc[:len(new)].set_index(new.index), res.iloc[len(new):].set_index(chk.index)
        if len(chk):
            dmax = float(np.nanmax(np.abs(rc.tdd_matched.values - v1.reindex(chk.index).values)))
            fl = bool((rc.match_flag.values == chk.match_flag.values).all())
            check = dict(n=int(len(chk)), max_abs_diff=dmax, flag_equal=fl)
            if not (dmax <= 1e-6 and fl):
                raise SystemExit(f"[tdd] 재현 점검 실패: v3 셀 {len(chk)}개 최대 차 {dmax:.3g}, 표지 일치 {fl}")
        out = rn.assign(region=new.region.values, part="v5new").reset_index().rename(columns={"index": "loc_id"})[cols]
    else:
        out = pd.DataFrame(columns=cols)
    XB.check_out_dir(P.v5)
    P.v5.mkdir(parents=True, exist_ok=True)
    tmp = P.tdd_v5.with_name(P.tdd_v5.name + f".tmp{os.getpid()}")
    out.to_csv(tmp, index=False)
    os.replace(tmp, P.tdd_v5)
    meta = dict(created=now_kst().isoformat(timespec="seconds"), definition="a2_year_matched_tdd.py 2) 단계(matched), x_multisource_stacking.a2_tdd_cells",
                v5_sha256=XB.sha256_file(P.v5_table), years_src=P.v5_labels.name, n_new=int(len(out)), check=check, sha256=XB.sha256_file(P.tdd_v5),
                script_sha256=XB.sha256_file(__file__), max_rss_mb=XB.max_rss_mb(), label_values_used=False)
    XB._atomic_write(P.tdd_v5_meta, json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    return meta


# ================================================================ 4. 실행 표(spec)
def _lic_ok(df):
    return pd.to_numeric(df.lic_unverified, errors="coerce").fillna(0).astype(float) != 1


def make_specs(lab_v4, lab_xf, natl_confirmed=(), submap=None):
    """실행 표 정의. 새 macro 지역마다 '<macro>~xf'(약관 확인분 대상 셀만), NAtlantic 약관 확인분 + 새 셀 'NAtlantic~lic~v5',
    약관 회신으로 확인된 자료원이 있으면 'NAtlantic~v5'(전체 판 + 새 셀), 셀 보강 '<대상>~aug~v5'(그 macro 의 약관 확인분 v4 대상 셀 + 새 셀)와
    같은 작업의 기준 표 '<대상>~aug~ref'(새 셀 없음. L40 비교 기준을 같은 작업·노드에서 다시 적합한다, 1절 플랫폼), 알래스카 하위 과제
    'AL-7~xf'(새 알래스카 셀이 대상, XC F2 서술 전용이라 XF 는 적합하지 않는다).
    셀 보강의 대상: parent macro 에 하위 지역이 있으면(Canada, Lena; submap = extended_submap) 새 셀 블록의 하위 지역(예 CA-1), 없으면 macro
    (예 Tibet_LGD). 반환 {별칭: dict}. keep_v4·keep_v5 는 F4_direct 로 바꿀 새 행 loc_id 다."""
    specs = {}
    tx = lab_xf[pd.to_numeric(lab_xf.xf_target, errors="coerce").fillna(0).astype(int) == 1] if len(lab_xf) else lab_xf
    tv = tx[_lic_ok(tx)] if len(tx) else tx
    v4t = lab_v4[(lab_v4.part == "new") & (lab_v4.lgd_role == "target")]

    def v4_ids(mac, confirmed=()):
        g = v4t[v4t.macro_v4 == mac]
        ok = _lic_ok(g)
        if confirmed:
            okset = set(";".join(g.sources[ok].fillna("")).split(";")) | set(confirmed)
            ok = ok | g.sources.fillna("").str.split(";").map(lambda L: all(s in okset for s in L if s))
        return sorted(int(v) for v in g.loc_id[ok]), int((~ok).sum())

    def add(alias, target, kind, keep_v4, keep_v5, n_drop_v4=0, n_drop_v5=0, note="", **extra):
        if "__" in alias or "/" in alias or " " in alias:
            raise ValueError(f"별칭 {alias} 에 '__', '/', 공백을 쓸 수 없다(조각 이름)")
        par = extra.get("parent") or str(target)
        specs[alias] = dict(alias=alias, target=str(target), kind=kind, blind=BLIND[kind], keep_v4=[int(v) for v in keep_v4],
                            keep_v5=[int(v) for v in keep_v5], drop_v3=[int(v) for v in E1.DROP_V3.get(par, [])],
                            n_lic_dropped_v4=int(n_drop_v4), n_lic_dropped_v5=int(n_drop_v5), note=note,
                            run=True, run_skip="", **extra)
    if len(tx):
        for mac in sorted(set(tx.macro_v5[tx.xf_role == ROLE_NEW])):
            ids = tv.loc_id[(tv.macro_v5 == mac) & (tv.xf_role == ROLE_NEW)]
            nd = int(((tx.macro_v5 == mac) & (tx.xf_role == ROLE_NEW)).sum() - len(ids))
            add(f"{mac}~xf", mac, KIND_NEW, [], ids, n_drop_v5=nd, note="새 macro 후보(이번 조사 뒤 확보). 약관 확인분 대상 셀만")
    x_n = tv.loc_id[tv.xf_role == ROLE_NATL] if len(tv) else []
    k4, nd4 = v4_ids("NAtlantic")
    add(NATL_ALIAS, "NAtlantic", KIND_NATL, k4, x_n, nd4, (int((tx.xf_role == ROLE_NATL).sum()) - len(x_n)) if len(tx) else 0,
        note="NAtlantic 약관 확인분 판(LGD NAtlantic~lic) + 새 공개 자료 셀. 민감도 행만(WRAPUP 7.2 (a)6)")
    if not len(x_n):
        specs[NATL_ALIAS].update(run=False, run_skip="새 셀 0(LGD NAtlantic~lic 와 같은 표, 적합하지 않는다)")
    if natl_confirmed:
        kf, ndf = v4_ids("NAtlantic", tuple(natl_confirmed))
        add(NATL_FULL_ALIAS, "NAtlantic", KIND_NATL_FULL, kf, x_n, ndf, 0,
            note=f"약관 회신 확인 자료원 {sorted(natl_confirmed)} 을 넣은 전체 판 + 새 셀. 민감도 행만")
    if len(tv):
        parents = set(submap.parent) if submap is not None else set()
        groups = {}
        for r in tv[tv.xf_role == ROLE_AUG].itertuples(index=False):
            mac = str(r.macro_v5)
            sub = sub_of(submap, mac, r.block) if mac in parents else ""
            if mac in parents and not sub:
                raise SystemExit(f"[specs] 셀 보강 셀 {r.loc_id} 의 블록이 하위 지역 대응표에 없다(extended_submap 을 v5 로 다시 만든다)")
            groups.setdefault((mac, sub), []).append(int(r.loc_id))
        for (mac, sub), ids in sorted(groups.items()):
            tgt = sub or mac
            k4a, nd4a = v4_ids(mac)
            extra = dict(parent=mac) if sub else {}
            al, ref = f"{tgt}~aug~v5", f"{tgt}~aug~ref"
            add(al, tgt, KIND_AUG, k4a, ids, nd4a, 0, note=f"셀 보강(L40 형식 민감도, 서술). 대상 {tgt}" + (f"(parent {mac}, 하위 지역)" if sub else ""),
                ref=ref, **extra)
            add(ref, tgt, KIND_AUG_REF, k4a, [], nd4a, 0, note=f"{al} 의 L40 기준 표(새 셀 없음, 같은 작업에서 다시 적합. LG 방법 축만)", aug=al,
                parts=["lg"], **extra)
    ak = tv[tv.xf_role == ROLE_AK] if len(tv) else tv
    if len(ak):
        nd = int((tx.xf_role == ROLE_AK).sum() - len(ak))
        add(AL7_ALIAS, AL7, KIND_AK, [], ak.loc_id, 0, nd, note="새 알래스카 하위 과제. 독립 지역 아님. XD 개발 과제에 넣지 않는다. XC F2 서술 전용")
        specs[AL7_ALIAS].update(run=False, run_skip="XC F2 서술 전용(XF 는 적합하지 않는다. XC 가 register_xf_aliases 로 읽는다)", f2_only=True)
    return specs


def spec_dir(P, alias) -> Path:
    """실행 표 폴더. 언제나 지금 산출 뿌리 아래 run_tables/<별칭> 으로 푼다(xf_specs.json 의 table.dir 은 산출 뿌리 기준 상대 경로라
    Rescale 노드처럼 ROOT 가 달라도 같은 곳을 가리킨다)."""
    return Path(P.runs) / str(alias)


def write_tables(P, specs, v5_table=None, v5_soil=None, submap=None):
    """spec 마다 실행 표(v5 원문 줄 선택, lgd_eligibility_v1.write_run_table_text)와 토양 표 기호 연결. 하위 지역 대상 spec(parent 있음)에는
    블록 대응표(submap = extended_submap)를 그 폴더의 lg_subregion_map_v1.csv 로 둔다(h40·h54 의 기본 --subregion-map 이 읽는다).
    specs 에 표 기록(table.dir = 산출 뿌리 기준 상대 경로)을 더해 xf_specs.json 에 쓴다."""
    v5_table, v5_soil = Path(v5_table or P.v5_table), Path(v5_soil or P.v5_soil)
    XB.check_out_dir(P.runs)
    _gitignore(P.runs)
    for al, s in specs.items():
        d = spec_dir(P, al)
        d.mkdir(parents=True, exist_ok=True)
        keep = sorted(set(s["keep_v4"]) | set(s["keep_v5"]))
        tmp = d / f".tmp{os.getpid()}.csv"
        wr = E1.write_run_table_text(v5_table, tmp, keep, drop_v3=s["drop_v3"])
        os.replace(tmp, d / "fidelity_base_v3.csv")
        soil = d / "e5_soil_tdd_v3.csv"
        if soil.is_symlink() or soil.exists():
            soil.unlink()
        os.symlink(os.path.relpath(v5_soil, d), soil)
        mp = d / SUBMAP_NAME
        if s.get("parent"):
            if submap is None:
                raise SystemExit(f"[tables] {al} 는 하위 지역 대상이라 블록 대응표가 필요하다")
            submap.to_csv(mp, index=False)
        elif mp.exists():
            mp.unlink()
        s["table"] = dict(dir=os.path.relpath(d, P.root), sha256=XB.sha256_file(d / "fidelity_base_v3.csv"), soil=os.path.relpath(v5_soil, d),
                          submap_sha256=XB.sha256_file(mp) if mp.exists() else "", **wr)
    save_specs(P, specs, v5_table)
    return specs


def save_specs(P, specs, v5_table=None):
    XB.check_out_dir(P.root)
    P.root.mkdir(parents=True, exist_ok=True)
    doc = dict(created=now_kst().isoformat(timespec="seconds"), plan=f"{XB.PLAN_DOC}@{XB.PLAN_COMMIT}", exp=EXP_NAME,
               v5_sha256=XB.sha256_file(v5_table) if v5_table and Path(v5_table).exists() else "", specs=specs)
    XB._atomic_write(P.specs, json.dumps(doc, ensure_ascii=False, indent=1, default=int))


def load_specs(P) -> dict:
    if not P.specs.exists():
        raise SystemExit(f"[specs] {P.specs} 가 없다. --stage tables 를 먼저 한다")
    return json.loads(P.specs.read_text())["specs"]


def final_xcf3(P):
    """XF 자료 마감 뒤 처음 만든 부록 XC-F3 최종판(적격 동결 기록). 없으면 None."""
    if not P.xcf3_json.exists():
        return None
    doc = json.loads(P.xcf3_json.read_text())
    return doc if doc.get("final") else None


def frozen_hashes(P) -> dict:
    """적격 동결 대조 대상의 현재 sha256(적격 표, spec, v5)."""
    return dict(elig_sha256=XB.sha256_file(P.elig) if P.elig.exists() else "", specs_sha256=XB.sha256_file(P.specs) if P.specs.exists() else "",
                v5_sha256=XB.sha256_file(P.v5_table) if P.v5_table.exists() else "")


def check_frozen(P, require=True):
    """LGD 6B.4 '판정 시점'(학습 전 --count-only 로 정하고 그 뒤 바꾸지 않는다)의 강제. 최종 부록 XC-F3 의 적격 표·spec·v5 sha256 이 지금 파일과
    같은지 확인한다. 다르면 멈춘다. 최종판이 없으면 require 일 때 멈추고 아니면 None. 반환 최종판 dict."""
    doc = final_xcf3(P)
    if doc is None:
        if require:
            raise SystemExit(f"[동결] 최종 부록 XC-F3({P.xcf3_json})가 없다. XF 자료 마감({DEADLINE_DATA.isoformat()}) 뒤 --xc-f3 로 적격을 동결한다")
        return None
    cur = frozen_hashes(P)
    bad = [k for k, v in cur.items() if doc.get(k) != v]
    if bad:
        raise SystemExit(f"[동결] 동결 뒤 바뀐 파일: {bad}(XC-F3 기록과 sha256 이 다르다). 적격은 학습 전 기록에서 바꾸지 않는다(LGD 6B.4)")
    return doc


def natl_status(P, specs, elig) -> dict:
    """NAtlantic v5 민감도 행의 판정 방식(5절 마감 'XF Stordalen·Adventdalen 세기 T0 + 24 h, 넘기면 점 추정', 2.6 '합집합이 8 미만이면 점 추정으로
    확정'). 첫 세기 기록(xf_natl_v5_first_count.json)이 없거나, 마감 뒤이거나, 그 세기에 그 판이 없었거나, 그 세기에서 부적격이었거나, 지금
    부적격이면 점 추정이다. 전체 판(NAtlantic~v5)은 약관 회신 뒤에 생기므로 첫 세기 기록에 없으면 마감 규칙의 대상인 약관 확인분 판
    (NAtlantic~lic~v5: Stordalen·Adventdalen 세기)의 기록으로 '첫 세기 부적격'을 판정하고, 지금 적격 여부는 전체 판 자신의 적격 표로 본다.
    반환 {별칭: (점 추정 여부, 사유)}."""
    rec = json.loads(P.natl_first.read_text()) if P.natl_first.exists() else None
    el = dict(zip(elig.alias, elig.eligible.astype(bool))) if len(elig) else {}
    out = {}
    for al, s in specs.items():
        if s.get("kind") not in (KIND_NATL, KIND_NATL_FULL):
            continue
        r0, basis = (rec or {}).get("specs", {}).get(al), al
        if r0 is None and rec is not None and s.get("kind") == KIND_NATL_FULL:
            r0, basis = rec.get("specs", {}).get(NATL_ALIAS), NATL_ALIAS
        if rec is None:
            out[al] = (True, "T0 + 24 h 안의 첫 세기 기록 없음(5절 마감)")
        elif dt.datetime.fromisoformat(rec["created"]) > DEADLINE_COUNT:
            out[al] = (True, f"첫 세기 {rec['created']} 가 마감 {DEADLINE_COUNT.isoformat()} 뒤")
        elif r0 is None:
            out[al] = (True, "첫 세기에 없던 판")
        elif not bool(r0.get("eligible")):
            out[al] = (True, f"첫 세기({basis})에서 부적격(채점 블록 합집합 {r0.get('nb_union')}): 점 추정으로 확정(2.6)")
        elif not el.get(al, False):
            out[al] = (True, "지금 적격 표에서 부적격")
        else:
            out[al] = (False, "")
    return out


def point_only_map(P, specs, elig) -> dict:
    """별칭 → 점 추정 여부(적격 표의 부적격 + NAtlantic v5 마감 규칙)."""
    el = dict(zip(elig.alias, elig.eligible.astype(bool))) if len(elig) else {}
    po = {al: not bool(el.get(al, False)) for al in specs}
    for al, (p, _) in natl_status(P, specs, elig).items():
        po[al] = po[al] or p
    return po


def register_xf_aliases(P=None, require_frozen=True) -> dict:
    """XF 실행 표 별칭을 xbatch_core.RUN_TABLES 에 더한다(이 프로세스 안에서만). XB.build_tctx, XB.get_data, XB.split_plan 이 XF 표를 읽게 한다
    (XB.get_data 는 a.LGD / 디렉터리 이므로 실행 시점의 산출 뿌리에서 푼 절대 경로를 넣는다). XC(XC-F3, AL-7 F2 서술)·XB(--xb 진입점)가 부른다.
    require_frozen 이면 최종 부록 XC-F3 의 적격 동결 해시를 대조한다(R4). 점 추정 여부는 적격 표와 NAtlantic v5 마감 규칙을 따른다. 반환 더한 항목."""
    P = P or Paths()
    check_frozen(P, require=require_frozen)
    specs = load_specs(P)
    el = pd.read_csv(P.elig) if P.elig.exists() else pd.DataFrame(columns=["alias", "eligible"])
    po = point_only_map(P, specs, el)
    add = {al: (str(spec_dir(P, al).resolve()), s["target"], bool(po.get(al, True))) for al, s in specs.items()}
    XB.RUN_TABLES.update(add)
    return add


# ================================================================ 5. 적격 세기(라벨 값 미사용)
def table_df(d):
    """실행 표 디렉터리 → load_base(F4_direct, 토양 도일 병합, macro). 라벨은 유무만 쓴다."""
    return load_base(Path(d))


def target_index(df, target):
    return np.where(df.macro.values == str(target))[0]


def spec_target_idx(df, spec, d):
    """spec 의 대상 색인과 parent. macro 대상이면 (macro 가 대상인 행, None). 하위 지역 대상(parent 있음)이면 실행 표 폴더의 블록 대응표에서
    (parent, 블록) → 하위 지역이 대상인 행(h40.apply_subregion_map 과 같은 규칙)과 parent."""
    par = spec.get("parent")
    if not par:
        return target_index(df, spec["target"]), None
    mp = pd.read_csv(Path(d) / SUBMAP_NAME, dtype=dict(parent=str, subregion=str))
    blocks = mp.block[(mp.parent == par) & (mp.subregion == str(spec["target"]))].astype(int).values
    return np.where((df.macro.values == par) & np.isin(df.block.values.astype(int), blocks))[0], par


def structure15(df, t_idx, splits=SPLITS, buffer_km=float(E1.BUFFER_KM), parent=None):
    """분할 1–5 의 구조(lgd_eligibility_v1.structure 와 같은 규칙, 라벨 평균은 계산하지 않는다). parent 를 주면(하위 지역 대상, 모드 x)
    h40.source_idx 와 같이 원천에서 parent macro 를 뺀다. 반환 (요약 dict, 분할별 dict)."""
    if len(t_idx) == 0:
        return dict(n_cells=0, n_blocks=0, n_eval_cells=0, n_eval_blocks=0, n_src=np.nan, n_buffer_excluded=np.nan, n_unique_splits=0,
                    n_valid_splits=0, nb_union=0, min_nb_eval_used=np.nan, max_block_share_used=np.nan, few_blocks=False, used_splits=""), {}
    y = df.alt_cm.values.astype(float)
    src = np.where(np.isfinite(y))[0]
    src = src[~np.isin(src, t_idx)]
    if parent:
        src = src[df.macro.values[src] != parent]
    n0 = len(src)
    la, lo = df.lat.values, df.lon.values
    tl, tn = la[t_idx], lo[t_idx]
    keep = np.ones(len(src), bool)
    for j, i in enumerate(src):
        if np.abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < buffer_km:
            keep[j] = False
    src = src[keep]
    allb = frozenset(df.block.values[t_idx])
    seen, info = [], {}
    for sp in splits:
        A_idx, B_idx = half_split_blocks(df, t_idx, int(sp))
        a = frozenset(df.block.values[A_idx])
        evB = B_idx[eval_mask(df.iloc[B_idx])]
        dup = next((q for q, aq in seen if aq == a), -1)
        mir = next((q for q, aq in seen if aq == allb - a), -1)
        nb = int(df.iloc[evB].block.nunique())
        vc = pd.Series(df.block.values[evB]).value_counts()
        info[int(sp)] = dict(split=int(sp), dup_of=int(dup), mirror_of=int(mir), n_A=int(len(A_idx)), nb_A=len(a), n_eval=int(len(evB)),
                             nb_eval=nb, valid=bool(dup < 0 and nb >= 2), max_block_share=round(float(vc.iloc[0] / len(evB)), 4) if len(evB) else np.nan,
                             eval_blocks=sorted(set(df.block.values[evB].tolist())))
        seen.append((int(sp), a))
    uniq = [v for v in info.values() if v["dup_of"] < 0]
    valid = [v for v in uniq if v["valid"]]
    used = valid if valid else uniq
    union = set().union(*[set(v["eval_blocks"]) for v in used]) if used else set()
    em = eval_mask(df.iloc[t_idx])
    mn = min(v["nb_eval"] for v in used) if used else np.nan
    return dict(n_cells=int(len(t_idx)), n_blocks=int(len(allb)), n_eval_cells=int(em.sum()),
                n_eval_blocks=int(len(set(df.block.values[t_idx][em]))), n_src=int(len(src)), n_buffer_excluded=int(n0 - len(src)),
                n_unique_splits=len(uniq), n_valid_splits=len(valid), nb_union=len(union), min_nb_eval_used=mn,
                max_block_share_used=max(v["max_block_share"] for v in used) if used else np.nan,
                few_blocks=bool(used and mn < XB.FEW_BLOCKS), used_splits=",".join(str(v["split"]) for v in used)), info


def run_splits15(info):
    """실행 분할(h40.enumerate_units 규칙): 중복 아님, A·채점 셀 있음, 유효 분할이 있으면 무효 분할 제외."""
    any_valid = any(v["valid"] for v in info.values())
    out = []
    for sp, v in sorted(info.items()):
        if v["dup_of"] >= 0 or v["n_eval"] == 0 or v["n_A"] == 0:
            continue
        if not v["valid"] and any_valid:
            continue
        out.append(int(sp))
    return out


def count_spec(spec, d, lab_xf=None):
    """spec 하나의 적격 세기. 반환 (요약 행, 분할 행 목록). 라벨은 유무(채점 마스크, 원천 유효 행)에만 쓴다."""
    df = table_df(d)
    tgt = spec["target"]
    t_idx, parent = spec_target_idx(df, spec, d)
    ids = df.loc_id.values.astype(int)
    tid = ids[t_idx]
    n_v3, n_v4, n_x = int((tid < LOC0_V4).sum()), int(((tid >= LOC0_V4) & (tid < LOC0_V5)).sum()), int((tid >= LOC0_V5).sum())
    st, info = structure15(df, t_idx, parent=parent)
    D = SimpleNamespace(df=df, target_idx=lambda t, _i=t_idx: _i)
    keep_xc, rows_xc = XB.split_plan_new(D, tgt, XC_SPLITS, XB.PRIOR_LG, XB.MIN_EVAL_BLOCKS)
    xs = XB.plan_summary(rows_xc)
    blk_old = set(df.block.values[t_idx][tid < LOC0_V5].tolist())
    blk_new = set(df.block.values[t_idx][tid >= LOC0_V5].tolist())
    xr = df.iloc[t_idx][tid >= LOC0_V5]
    feats = [c for c in SHARED_CORE if c in df.columns]
    exp_v4 = len(spec["keep_v4"])
    exp_x = len(spec["keep_v5"])
    row = dict(alias=spec["alias"], target=tgt, kind=spec["kind"], blind=spec.get("blind", ""), **{k: v for k, v in st.items()},
               n_v3_target=n_v3, n_v4_target=n_v4, n_xf_target=n_x, n_keep_v4=exp_v4, n_keep_v5=exp_x,
               check_target_rows=bool((n_v4 <= exp_v4 and set(spec["keep_v5"]) <= set(tid.tolist())) if parent else (n_v4 == exp_v4 and n_x == exp_x)),
               parent=parent or "",
               n_xf_blocks=len(blk_new), n_xf_blocks_new=len(blk_new - blk_old),
               xf_x25_all=round(float(xr[feats].notna().all(axis=1).mean()), 4) if len(xr) else np.nan,
               xf_eval_ready=int(eval_mask(xr).sum()) if len(xr) else 0,
               n_lic_dropped_v4=spec.get("n_lic_dropped_v4", 0), n_lic_dropped_v5=spec.get("n_lic_dropped_v5", 0),
               lic_ok=True, run_splits=",".join(str(v) for v in run_splits15(info)),
               xc_n_valid=xs["n_valid"], xc_keep=",".join(str(v) for v in keep_xc), xc_excluded=json.dumps(xs["excluded"]),
               xc_nb_union=xs["nb_union"], xc_nb_min=xs["nb_eval_min"], xc_few_blocks=bool(keep_xc and xs["nb_eval_min"] < XB.FEW_BLOCKS))
    row["eligible"] = bool(row["n_valid_splits"] >= 1 and row["nb_union"] >= MIN_UNION)
    row["eligible_xc"] = bool(len(keep_xc) >= 1 and xs["nb_union"] >= MIN_UNION)
    row["point_only"] = not row["eligible"]
    if lab_xf is not None and len(lab_xf):
        lx = lab_xf[lab_xf.loc_id.isin(spec["keep_v5"])]
        row["xf_min_dist_other_macro_km"] = float(pd.to_numeric(lx.dist_other_macro_km, errors="coerce").min()) if len(lx) else np.nan
        row["xf_sources"] = ";".join(sorted({s for v in lx.sources.fillna("") for s in str(v).split(";") if s}))
    row["table_sha256"] = XB.sha256_file(Path(d) / "fidelity_base_v3.csv")
    if spec["kind"] == KIND_NEW:
        row["role"] = ("XF-1–XF-3 대상(독립 지역 확인)" if row["eligible"] else "점 추정만(부적격)") + ("; XC-F3 적격" if row["eligible_xc"] else "")
    elif spec["kind"] in (KIND_NATL, KIND_NATL_FULL):
        row["role"] = "민감도 행(" + ("4분 판정 후보(첫 세기 마감 규칙 적용)" if row["eligible"] else "점 추정") + "). XF 확인·XC-F3 에 넣지 않는다"
    elif spec["kind"] == KIND_AUG_REF:
        row["role"] = f"셀 보강의 L40 기준 표({spec.get('aug', '')})"
    elif spec["kind"] == KIND_AK:
        row["role"] = "알래스카 하위 과제(XC F2 서술 전용, 독립 지역 아님, XD 개발 과제 아님)"
    else:
        row["role"] = "셀 보강(L40 형식 민감도, 서술)"
    srows = [dict(alias=spec["alias"], family="1-5", **{k: v for k, v in r.items() if k != "eval_blocks"}) for r in info.values()]
    srows += [dict(alias=spec["alias"], family="201-210", **{k: v for k, v in r.items() if k not in ("eval_blocks",)}) for r in rows_xc]
    return row, srows


# ---------------------------------------------------------------- 적합 수(가짜 라벨 dry 실행)
def h40_argv(spec_dir, target, threads=1, cb_iters=200, n_grid=LG_N_GRID, draws=LG_DRAWS, seeds=LG_SEEDS, splits=max(SPLITS),
             out_dir=None, tag=TAG):
    """LG CPU 방법 축의 h40 인자(h51.h40_argv 와 같은 범위: α 1, catboost_lo, 배치·학습기·α 중첩 축 끔)."""
    return ["--part", "cpu", "--modes", "x", "--targets", f"{target}:x", "--splits", str(int(splits)), "--n-grid", str(n_grid),
            "--draws", str(int(draws)), "--seeds", str(int(seeds)), "--learners", "catboost_lo", "--alphas", "1", "--place-n-grid", "",
            "--nested-targets", "NONE:x", "--learner-targets", "NONE:x", "--data-dir", str(spec_dir),
            "--out-dir", str(out_dir or (XB.XBATCH_ROOT / "_h40_unused")), "--tag", str(tag), "--threads", str(int(threads)),
            "--workers", "0", "--no-summarize", "--cb-iters", str(int(cb_iters))]


def h40_args_for(a, spec_dir, target):
    return H.parse_args(h40_argv(spec_dir, target, a.threads, a.cb_iters, a.LG_NGRID, a.LG_DRAWS, a.LG_SEEDS))


def h54_args_for(a):
    return XB.h54_args(exp="wf4", splits=SPLITS, grid=list(a.W4_GRID), threads=a.threads, cb_iters=a.cb_iters, seeds=a.W4_SEEDS,
                       nboot=a.nboot, draws_cap=a.W4_DRAWS_CAP, allow_local=bool(a.allow_local))


def dummy_ctx(D, HA, target, split):
    """h40.build_ctx 와 같은 색인의 작업 단위. 라벨 자리에 √TDD 를 넣는다(h51.dummy_ctx 와 같다. 적합 수 세기 전용, 라벨 값 미사용)."""
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx(target, "x")
    src = df.iloc[src_idx]
    A_idx, B_idx = half_split_blocks(df, t_idx, int(split))
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    A, B = df.iloc[A_idx], df.iloc[evB]
    F = H.FEATS
    c = H.Ctx(target, "x", int(split), parent, src[F].values, src.s.values.astype(float), src.s.values, src.macro.values,
              A[F].values, A.s.values.astype(float), A.s.values, A.block.values, B[F].values, B.s.values.astype(float), B.s.values,
              B.block.values, min_cells_prior=HA.min_cells_prior)
    c.prior = None
    info = D.split_structure(target)[int(split)]
    c.meta = dict(n_A=int(len(A_idx)), nb_A=int(A.block.nunique()), n_eval=int(len(evB)), nb_eval=int(B.block.nunique()), **comp,
                  dup_of=info["dup_of"], valid=info["valid"])
    return c


def compute_unit(c, alias, HA, a54, dry=False, parts=("lg", "wf4")):
    """한 문맥의 LG CPU 방법 축(h40.run_ctx)과 WF4 팔(h54.TUnit). 본 실행(run_unit), 적합 수 세기(count_fits), 누설 시험이 모두 이 함수를 지난다.
    parts 로 고른 부분만 적합한다(--resume 에서 완료 조각을 건너뛴다). 반환 dict(lg=(rows, store, stats), wf4=(rows, store, stats),
    elapsed={부분: 초}). 저장소 이름: LG '<별칭>|x', WF4 '<별칭>~wf4|x'. 추출 seed 는 표 안의 대상 이름(c.target)으로 정한다(LG·LGD·WF4 와 같다)."""
    out = dict(elapsed={})
    if "lg" in parts:
        t0 = time.time()
        rows, st, stats = H.run_ctx(c, "cpu", HA, learner_axis=False, nested=False, dry=dry)
        if st is not None:
            st.target = f"{alias}|x"
            st.meta.update(alias=alias, part="lg")
        for r in rows:
            r["alias"] = alias
        out["lg"] = (rows, st, stats)
        out["elapsed"]["lg"] = round(time.time() - t0, 1)
    if "wf4" in parts:
        t0 = time.time()
        U = W.TUnit(a54, c, f"{alias}~wf4", dry=dry, exp="wf4")
        U.run()
        r2, st2, stats2 = U.finish()
        if not dry:
            st2.meta.update(alias=alias, part="wf4")
        for r in r2:
            r["alias"] = alias
        out["wf4"] = (r2, st2, stats2)
        out["elapsed"]["wf4"] = round(time.time() - t0, 1)
    return out


def count_fits(a, spec, d):
    """spec 의 실행 분할마다 적합 수와 추정 시간(dry, 가짜 라벨). 반환 행 목록."""
    HA = h40_args_for(a, d, spec["target"])
    D = H.get_data(HA)
    info = D.split_structure(spec["target"])
    a54 = h54_args_for(a)
    out = []
    for sp in run_splits15(info):
        c = dummy_ctx(D, HA, spec["target"], sp)
        res = compute_unit(c, spec["alias"], HA, a54, dry=True, parts=tuple(spec.get("parts", ["lg", "wf4"])))
        z = dict(n_fit={}, est_detail={}, n_rows=0)
        s1, s2 = res.get("lg", (None, None, z))[2], res.get("wf4", (None, None, z))[2]
        f1, f2 = int(sum(s1["n_fit"].values())), int(sum(s2["n_fit"].values()))
        e1, e2 = float(sum(s1["est_detail"].values())), float(sum(s2["est_detail"].values()))
        out.append(dict(alias=spec["alias"], split=int(sp), n_A=c.meta["n_A"], nb_A=c.meta["nb_A"], n_eval=c.meta["n_eval"], nb_eval=c.meta["nb_eval"],
                        n_src=c.meta["n_src"], valid=bool(c.meta["valid"]), fits_lg=f1, fits_wf4=f2, fits=f1 + f2, rows_lg=int(s1["n_rows"]),
                        rows_wf4=int(s2["n_rows"]), est_s=round(e1 + e2, 1)))
    return out


def natl_first_record(P, specs, el, now=None) -> dict:
    """NAtlantic v5 첫 세기 기록(5절 마감 'T0 + 24 h', 2.6 '합집합이 8 미만이면 점 추정으로 확정'). 우선 자료원(Stordalen, Adventdalen)의 점
    자료가 셀 조립에 처음 들어간 뒤의 첫 --count-only 만 기록한다. 파일은 배타 생성으로 한 번만 쓴다(덮어쓰지 않는다). 반환 상태 dict."""
    now = now or now_kst()
    if P.natl_first.exists():
        return dict(status="기록 있음(덮어쓰지 않는다)", path=str(P.natl_first.name))
    natl = [al for al, s in specs.items() if s.get("kind") in (KIND_NATL, KIND_NATL_FULL)]
    man = json.loads(P.points_manifest.read_text()) if P.points_manifest.exists() else dict(files={})
    used = {k: e.get("sha256", "") for k, e in man.get("files", {}).items() if not e.get("removed")}   # 셀 조립(cells 단계)이 읽은 점 자료
    pri = sorted(k for k in used if k.replace("_points.csv", "") in PRIORITY_SOURCES)
    if not natl or not pri:
        return dict(status="기록하지 않음(" + ("NAtlantic v5 spec 없음" if not natl else "우선 자료원 점 자료가 셀 조립에 아직 없다") + ")")
    e = el.set_index("alias")
    doc = dict(created=now.isoformat(timespec="seconds"), deadline_count=DEADLINE_COUNT.isoformat(), before_deadline=bool(now <= DEADLINE_COUNT),
               priority_points={k: used[k] for k in pri}, points_manifest_sha256=XB.sha256_file(P.points_manifest) if P.points_manifest.exists() else "",
               n_points_files=len(man.get("files", {})), v5_sha256=XB.sha256_file(P.v5_table) if P.v5_table.exists() else "",
               specs_sha256=XB.sha256_file(P.specs) if P.specs.exists() else "", script_sha256=XB.sha256_file(__file__),
               specs={al: dict(table_sha256=str(e.loc[al, "table_sha256"]), n_cells=int(e.loc[al, "n_cells"]), n_xf_target=int(e.loc[al, "n_xf_target"]),
                               n_blocks=int(e.loc[al, "n_blocks"]), n_valid_splits=int(e.loc[al, "n_valid_splits"]), nb_union=int(e.loc[al, "nb_union"]),
                               eligible=bool(e.loc[al, "eligible"]), xf_sources=str(e.loc[al, "xf_sources"]) if "xf_sources" in e.columns else "",
                               run=bool(specs[al].get("run", True)))
                      for al in natl if al in e.index},
               rule="이 기록이 없거나 마감 뒤이거나 부적격이면 NAtlantic v5 민감도 행은 점 추정이다(집계기 natl_status). 라벨 값 미사용")
    XB.check_out_dir(P.root)
    with open(P.natl_first, "x", encoding="utf-8") as f:                    # 배타 생성(한 번만)
        f.write(json.dumps(doc, ensure_ascii=False, indent=1, default=str))
    return dict(status="기록함", before_deadline=doc["before_deadline"], specs={k: dict(eligible=v["eligible"], nb_union=v["nb_union"])
                                                                                 for k, v in doc["specs"].items()})


def count_only(a, P=None, specs=None, now=None):
    """적격 표와 적합 수. 화면에는 셀 수·블록 수·분할 수·적합 수만 쓴다. 최종 부록 XC-F3(적격 동결) 뒤에는 거부한다(LGD 6B.4 '판정 시점')."""
    P = P or a.P
    if final_xcf3(P) is not None:
        raise SystemExit(f"[count-only] 적격은 최종 부록 XC-F3({P.xcf3_json.name})로 동결되었다. 다시 세지 않는다(LGD 6B.4 '판정 시점')")
    specs = specs if specs is not None else load_specs(P)
    lab_xf = pd.read_csv(P.v5_labels, low_memory=False) if P.v5_labels.exists() else None
    rows, srows, crows = [], [], []
    for al in sorted(specs):
        s = specs[al]
        d = spec_dir(P, al)
        r, sr = count_spec(s, d, lab_xf)
        rows.append(r)
        srows += sr
        if a.no_fit_count:
            continue
        crows += count_fits(a, s, d)                                       # 적합하지 않는 spec 도 센다(비용 참고, 요약의 run 열)
    el = pd.DataFrame(rows)
    sp = pd.DataFrame(srows)
    cnt = pd.DataFrame(crows, columns=["alias", "split", "n_A", "nb_A", "n_eval", "nb_eval", "n_src", "valid", "fits_lg", "fits_wf4", "fits",
                                       "rows_lg", "rows_wf4", "est_s"])
    XB.check_out_dir(P.root)
    P.root.mkdir(parents=True, exist_ok=True)
    el.to_csv(P.elig, index=False)
    sp.to_csv(P.elig_splits, index=False)
    cnt.to_csv(P.count, index=False)
    summ = []
    for al in sorted(specs):
        q = cnt[cnt.alias == al]
        h = float(q.est_s.sum()) / 3600 if len(q) else 0.0
        summ.append(dict(alias=al, kind=specs[al]["kind"], run=bool(specs[al].get("run", True)), units=int(len(q)), fits=int(q.fits.sum()) if len(q) else 0,
                         est_cpu_h=round(h, 3), est_cpu_h_rescale=round(h * RESCALE_RATIO, 3), longest_unit_h=round(float(q.est_s.max()) / 3600, 3) if len(q) else 0.0))
    sm = pd.DataFrame(summ)
    sm.to_csv(P.count_sum, index=False)
    t = now or now_kst()
    first = natl_first_record(P, specs, el, now=t) if len(el) else dict(status="기록하지 않음(spec 없음)")
    meta = dict(created=t.isoformat(timespec="seconds"), deadline_count=DEADLINE_COUNT.isoformat(), deadline_data=DEADLINE_DATA.isoformat(),
                after_deadline_count=bool(t > DEADLINE_COUNT), after_deadline_data=bool(t > DEADLINE_DATA),
                rule=dict(eligible="유효 분할(1–5, h40 규칙) ≥ 1 이고 사용 분할 채점 블록 합집합 ≥ 8", eligible_xc="분할 201–210(xbatch 새 seed 규칙: 앞선 "
                          "분할 1–5 와 같거나 여집합, 새 범위 중복, 채점 블록 2 미만 제외) 유효 ≥ 1 이고 합집합 ≥ 8", point_only="부적격이면 점 추정",
                          natl_v5="NAtlantic v5 는 첫 세기 기록(xf_natl_v5_first_count.json)이 T0 + 24 h 안이고 적격일 때만 4분 판정"),
                natl_first_count=first, v5_sha256=XB.sha256_file(P.v5_table) if P.v5_table.exists() else "",
                specs_sha256=XB.sha256_file(P.specs) if P.specs.exists() else "",
                outputs={p.name: XB.sha256_file(p) for p in (P.elig, P.elig_splits, P.count, P.count_sum)},
                label_use="라벨은 셀 유무(채점 마스크·원천 유효 행)에만 썼다. 적합 수는 √TDD 가짜 라벨 dry 실행이다",
                max_rss_mb=XB.max_rss_mb(), script_sha256=XB.sha256_file(__file__))
    XB._atomic_write(P.count_meta, json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[count-only] spec {len(specs)} · 적격(1–5) {int(el.eligible.sum()) if len(el) else 0} · 적격(201–210) "
          f"{int(el.eligible_xc.sum()) if len(el) else 0} · NAtlantic v5 첫 세기 기록: {first['status']}", flush=True)
    for r in rows:
        print(f"  {r['alias']}: 종류 {r['kind']} · 대상 셀 {r['n_cells']}(v3 {r['n_v3_target']}, v4 {r['n_v4_target']}, 새 {r['n_xf_target']}) · "
              f"블록 {r['n_blocks']}(새 블록 {r['n_xf_blocks_new']}) · 채점 셀 {r['n_eval_cells']} · 원천 {r['n_src']} · 유효 분할 {r['n_valid_splits']} · "
              f"채점 블록 합집합 {r['nb_union']} · 최소 채점 블록 {r['min_nb_eval_used']} · 적격 {r['eligible']} · 201–210 유효 {r['xc_n_valid']} "
              f"합집합 {r['xc_nb_union']} 적격 {r['eligible_xc']} · 대상 행 점검 {r['check_target_rows']}", flush=True)
    for r in summ:
        print(f"  [적합 수] {r['alias']}: 단위 {r['units']} · 적합 {r['fits']:,} · 추정 {r['est_cpu_h']:.2f} CPU-h(4스레드) · "
              f"실측 비 {RESCALE_RATIO} 적용 {r['est_cpu_h_rescale']:.2f} · 가장 긴 단위 {r['longest_unit_h']:.2f} h", flush=True)
    print(f"  [자원] 최대 RSS {meta['max_rss_mb']} MB", flush=True)
    return el, sp, cnt


# ================================================================ 6. 실행
_WA = None


def unit_cfg(a, spec, part):
    """조각 설정(결과에 영향을 주는 것만, XB.make_unit_cfg). part(lg, wf4)는 variant, 실행 표 내용은 data_sha 라 둘 다 공통 해시에서 빠진다.
    나머지 항목은 모든 조각에서 같아야 한다(XB.check_cfg). spec 종류·표 해시는 unit.json 에 따로 적는다."""
    d = spec_dir(a.P, spec["alias"])
    dsha = f"{XB.code_sha(d / 'fidelity_base_v3.csv')}:{XB.code_sha(d / 'e5_soil_tdd_v3.csv')}"
    if (d / SUBMAP_NAME).exists():
        dsha += f":{XB.code_sha(d / SUBMAP_NAME)}"
    kw = dict(h40=dict(n_grid=str(a.LG_NGRID), draws=int(a.LG_DRAWS), seeds=int(a.LG_SEEDS), cb_iters=int(a.cb_iters), learners=["catboost_lo"],
                       alphas=["1"], lams=[0.25, 0.5, 1.0], methods=list(H.METHODS_ALL), buffer_km=float(E1.BUFFER_KM)),
              wf4=dict(grid=[int(v) for v in a.W4_GRID], seeds=int(a.W4_SEEDS), draws_cap=int(a.W4_DRAWS_CAP), cb_iters=int(a.cb_iters),
                       w_cands=list(W.W_CANDS), diag_n=int(W.DIAG_N)))
    return XB.make_unit_cfg("xf", variant=part, data_sha=dsha, **kw)


def _splits_of(v) -> list:
    """'1,2,3' 또는 3, 3.0, NaN → 분할 목록."""
    out = []
    for x in str(v).split(","):
        x = x.strip()
        if x and x.lower() != "nan":
            out.append(int(float(x)))
    return out


def run_unit(a, alias, split):
    """(spec, 분할) 하나: 문맥을 한 번 만들고 LG 방법 축과 WF4 팔을 적합해 조각 두 개(변형 lg, wf4)를 쓴다. --resume 이면 완료 조각을 건너뛴다."""
    t0 = time.time()
    spec = a.SPECS[alias]
    d = spec_dir(a.P, alias)
    tsha = spec.get("table", {}).get("sha256", "")
    if tsha and XB.sha256_file(d / "fidelity_base_v3.csv") != tsha:
        raise SystemExit(f"[run] {alias} 실행 표의 sha256 이 xf_specs.json 기록과 다르다(표를 다시 만들었거나 다른 판이다)")
    cfg_lg, cfg_w4 = unit_cfg(a, spec, "lg"), unit_cfg(a, spec, "wf4")
    sp_parts = [p for p in ("lg", "wf4") if p in spec.get("parts", ["lg", "wf4"])]
    done = {p: (XB.unit_state(a.P.shards, a.tag, alias, "x", split, cfg, p)[0] if a.resume else False) if p in sp_parts else True
            for p, cfg in (("lg", cfg_lg), ("wf4", cfg_w4))}
    if all(done.values()):
        return dict(alias=alias, split=int(split), status="resumed", elapsed_s=0.0)
    HA = h40_args_for(a, d, spec["target"])
    D = H.get_data(HA)
    info = D.split_structure(spec["target"])[int(split)]
    c = H.build_ctx(D, HA, spec["target"], "x", int(split))
    a54 = h54_args_for(a)
    expected = [int(v) for v in spec.get("run_splits", run_splits15(D.split_structure(spec["target"])))]
    meta = {k: v for k, v in c.meta.items() if not isinstance(v, (dict, list))}
    out = dict(alias=alias, split=int(split))
    res = compute_unit(c, alias, HA, a54, dry=False, parts=tuple(p for p in ("lg", "wf4") if not done[p]))   # 누설 시험과 같은 경로
    for part, cfg in (("lg", cfg_lg), ("wf4", cfg_w4)):
        if part not in res:
            continue
        rows, st, stats = res[part]
        u = dict(stats)
        u.update(meta)
        u.update(dup_of=int(info["dup_of"]), valid=bool(info["valid"]), elapsed_s=res["elapsed"][part],
                 n_fit_total=int(sum(stats["n_fit"].values())), spec_kind=spec["kind"], table_sha256=tsha)
        XB.write_shard(a.P.shards, a.tag, alias, "x", split, rows, [st], cfg, unit=u, variant=part, expected=expected, code_file=__file__)
        out.update({f"fits_{part}": u["n_fit_total"], f"status_{part}": u.get("status", "ok")})
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["status"] = "ok"
    return out


def _worker_init(argv):
    global _WA
    _WA = parse_args(argv)
    if not on_rescale() and _WA.max_mem_gb > 0:                          # 1절 로컬 자원: 워커마다 RSS 상한 감시
        rss_watchdog(_WA.max_mem_gb)
    _WA.SPECS = select_specs(_WA)


def _worker_unit(alias, split):
    return run_unit(_WA, alias, int(split))


def select_specs(a):
    """spec 과 실행 분할(적격 표의 run_splits). 적격 동결 기록(최종 부록 XC-F3)이 있으면 그 해시를 대조한다."""
    check_frozen(a.P, require=False)
    specs = load_specs(a.P)
    el = pd.read_csv(a.P.elig) if a.P.elig.exists() else None
    if el is not None and "run_splits" in el.columns:
        rs = dict(zip(el.alias, el.run_splits))
        for al, s in specs.items():
            if al in rs:
                s["run_splits"] = _splits_of(rs[al])
    if a.SPECS_SEL:
        bad = [v for v in a.SPECS_SEL if v not in specs]
        if bad:
            raise SystemExit(f"알 수 없는 spec: {bad}. 가능: {sorted(specs)}")
        specs = {k: v for k, v in specs.items() if k in a.SPECS_SEL}
    return specs


def plan_units(a, specs):
    """작업 단위 (별칭, 분할). run = False 인 spec 은 뺀다. --units 와 --shard K/N 을 적용한다(정렬 뒤 i % N == K)."""
    units = []
    for al in sorted(specs):
        s = specs[al]
        if not s.get("run", True):
            continue
        if "run_splits" not in s:
            raise SystemExit(f"[plan] {al} 의 실행 분할이 없다. --count-only 를 먼저 한다(xf_eligibility.csv 의 run_splits)")
        units += [(al, int(sp)) for sp in s["run_splits"]]
    if a.UNITS:
        units = [u for u in units if u in a.UNITS]
    if a.SHARD:
        k, n = a.SHARD
        units = [u for i, u in enumerate(units) if i % n == k]
    return units


def run_all(a, now=None):
    """R4 본 실행. 5절: XF 자료 마감(T0 + 7일) 뒤 부록 XC-F3 최종판(적격 동결)이 있어야 한다. 마감 전이거나 최종판이 없으면 거부한다."""
    now = now or now_kst()
    if now <= DEADLINE_DATA:
        raise SystemExit(f"[run] XF 자료 마감 {DEADLINE_DATA.isoformat()} 전이다. R4 는 마감 뒤 부록 XC-F3 최종판을 만든 다음에 실행한다(5절 단계 4)")
    check_frozen(a.P, require=True)
    a.SPECS = select_specs(a)
    units = plan_units(a, a.SPECS)
    XB.check_out_dir(a.P.shards)
    a.P.shards.mkdir(parents=True, exist_ok=True)
    _gitignore(a.P.shards)
    print(f"[plan] 작업 단위 {len(units)} · spec {len({u[0] for u in units})} · 워커 {a.workers} · 스레드 {a.threads} · tag {a.tag}", flush=True)
    done, failed = XB.execute(units, _worker_unit, workers=a.workers, threads=a.threads, init=_worker_init, init_args=(a.ARGV,))
    print(f"[done] 완료 {len(done)} · 실패 {len(failed)}", flush=True)
    if failed:
        XB._atomic_write(a.P.root / f"{a.tag}_failed.json", json.dumps([dict(unit=list(u), error=e) for u, e in failed], ensure_ascii=False, indent=1))
    return done, failed


# ================================================================ 7. 집계(봉인)
def _templates(hyp):
    """XF-1–XF-3 의 다섯 갈래 문장(1절 '해석 문장의 다섯 갈래', 2.6 예문). {k} = 지역 수, {a} = |Δ|(셀 가중, cm)."""
    subj = {"XF-1": "라벨 0개 직접 ML 은 원천 계수 Stefan 보다", "XF-2": "라벨 10개의 재보정 앵커 잔차는 같은 라벨로 재보정한 Stefan 보다",
            "XF-3": "전량 라벨의 재보정 앵커 잔차는 원천 계수 Stefan 보다"}[hyp]
    pair = {"XF-1": "라벨 0개 직접 ML 과 원천 계수 Stefan", "XF-2": "라벨 10개의 재보정 앵커 잔차와 재보정 Stefan",
            "XF-3": "전량 라벨의 재보정 앵커 잔차와 원천 계수 Stefan"}[hyp]
    lead = "추가 독립 지역 {k}곳에서 "
    return {"우세": lead + subj + " 오차가 {a} cm 작았다", "열세": lead + subj + " 오차가 {a} cm 컸다",
            "동등": lead + pair + " 의 오차는 0.5 cm 안에서 같았다", "미결정": lead + pair + " 의 차이를 확인하지 못했다",
            "판정 불가": lead + pair + " 의 차이는 판정할 수 없었다"}


ZERO_SENTENCE = ("공개 자료 조사(2026-10-04)에서 직접 측정 계절 말 라벨, 기존 대상과 100 km 이상 거리, 채점 블록 8개 이상의 조건을 만족하는 새 독립 지역을 "
                 "찾지 못했다(Supplementary Table).")


def compose(template, verdict, k, delta=None, holm_p=None, worse=(), reason="", tag="독립 지역 확인"):
    """문장 하나(XB.compose_sentence 와 같은 규칙에 '독립 지역 확인' 표지를 앞에 둔다). 우세·열세이고 Holm 보정 p ≥ 0.05 이면 '보정 전 유의',
    |Δ| < 0.5 cm 이면 효과 크기 표기, 판정 불가이면 사유, 풀 문장 뒤에는 지역 열세 문장."""
    b = XB.branch_of(verdict)
    a = abs(float(delta)) if delta is not None and np.isfinite(delta) else float("nan")
    base = template[b].format(k=int(k), a=f"{a:.2f}" if np.isfinite(a) else "a")
    notes = [tag] if tag else []
    if b in ("우세", "열세"):
        if holm_p is not None and np.isfinite(holm_p) and float(holm_p) >= XB.HOLM_ALPHA:
            notes.append(XB.UNCORRECTED_TXT)
        if np.isfinite(a) and a < XB.SMALL_EFFECT_CM:
            notes.append(XB.SMALL_EFFECT_TXT)
    if b == "판정 불가" and reason:
        notes.append(str(reason))
    out = base + (f"({', '.join(notes)})" if notes else "") + "."
    for nm, d, lo, hi in worse:
        out += f" 지역 {nm} 에서는 오차가 컸다(Δ {float(d):+.2f} cm, CI [{float(lo):.2f}, {float(hi):.2f}])."
    return out


def natl_sentence(hyp, row, n_sources, point_only):
    """NAtlantic v5 민감도 문장(2.6 사전 고정 문장의 틀)."""
    subj = {"XF-1": "라벨 0개 직접 ML 과 원천 계수 Stefan", "XF-2": "라벨 10개의 재보정 앵커 잔차와 재보정 Stefan",
            "XF-3": "전량 라벨의 재보정 앵커 잔차와 원천 계수 Stefan"}[hyp]
    if row is None:
        body = f"{subj} 의 차이는 판정할 수 없었다(행 없음)"
    elif point_only:
        body = f"{subj} 의 오차 차는 {float(row['delta']):+.2f} cm(셀 가중 점 추정)였다"
    else:
        t = {"우세": f"{subj} 가운데 앞쪽의 오차가 {abs(float(row['delta'])):.2f} cm 작았다",
             "열세": f"{subj} 가운데 앞쪽의 오차가 {abs(float(row['delta'])):.2f} cm 컸다",
             "동등": f"{subj} 의 오차는 0.5 cm 안에서 같았다", "미결정": f"{subj} 의 차이를 확인하지 못했다"}
        body = t.get(row.get("verdict4"), f"{subj} 의 차이는 판정할 수 없었다")
    kind = "점 추정" if point_only else "4분 판정"
    return f"북대서양 약관 확인분 판에 공개 자료 {int(n_sources)}건을 더한 민감도에서 {body}({kind}). 이 지역의 라벨 일부는 앞선 분석에서 열람했다."


def _row_meta(rows, **kw):
    for r in rows:
        r.update(kw)
    return rows


NA_V = ("행 없음", "판정 불가", "")
L40_D0_NS = (0, 3, 10, 40)                                                 # L1 지역 조건의 n(h52.L1_NS 와 같다)


def l28_shift(v0, v1):
    """L28 분류(LG 6A.5a, h52_lgd_pool.l28_shift 와 같은 규칙): 같으면 강건, 동등과 미결정 사이도 강건, 우세·열세와 동등·미결정 사이는 약화,
    우세와 열세가 바뀌면 의존. 한쪽이 행 없음·판정 불가이면 판정 불가."""
    if v0 in NA_V or v1 in NA_V:
        return "판정 불가"
    if v0 == v1:
        return "강건"
    if {v0, v1} == {"우세", "열세"}:
        return "의존"
    if (v0 in ("우세", "열세")) != (v1 in ("우세", "열세")):
        return "약화"
    return "강건"


def worst_shift(vals):
    """대비별 L28 분류의 종합(h52_lgd_pool.worst_shift 와 같은 규칙)."""
    vals = list(vals)
    if "의존" in vals:
        return "의존"
    if "약화" in vals:
        return "약화"
    ok = [v for v in vals if not str(v).startswith("판정 불가")]
    if not ok:
        return f"판정 불가(판정한 대비 0/{len(vals)})"
    return "강건" + (f"(판정한 대비 {len(ok)}/{len(vals)})" if len(ok) < len(vals) else "")


def _region_row(tms, al, gA, gB, nboot=None):
    """저장소 '<별칭>|x' 의 같은 라벨 집합 대비 지역 행(없으면 '행 없음' 행)."""
    rr, _ = XB.contrast_pool(tms, [f"{al}|x"], gA, gB, label="", kind="same", registered=1, nboot=nboot)
    rr = [r for r in rr if r.get("scope") == "region"]
    return rr[0] if rr else dict(target=f"{al}|x", scope="region", verdict4="행 없음")


def l40_rows(tms_lg, aug, ref, nlab=None, nboot=None):
    """셀 보강의 L40 형식 민감도(LG 6B.5 L40, 분류 L28). 확충판(aug, 새 셀 포함)과 같은 작업의 기준 표(ref, 새 셀 없음)에서
    (1) L1 지역 조건: n ≤ 40(0, 3, 10, 40 과 실제 라벨 수 ≤ 40 인 전량 행)에서 D0 − P0 가 우세인 n 이 있는가, (2) 전량 R1(0.25) − P0,
    (3) n 10 P1 − P0 를 같은 라벨 집합 대비(4분 판정)로 계산하고 기준 → 확충판의 판정 변화를 L28 로 분류한다. 반환 (지역 행 목록, 분류 행)."""
    rows, sh, l1c, d0 = [], {}, {}, {}
    sides = (("확충판", aug), ("기준", ref))
    for side, al in sides:
        vs = []
        ns = list(L40_D0_NS) + ([-1] if (nlab or {}).get(al) is not None and float(nlab[al]) <= 40 else [])
        for n in ns:
            r = _region_row(tms_lg, al, H._g("D0", n), H.P0_GRP, nboot)
            if r.get("verdict4") == "행 없음" and n != 0:
                continue                                                       # n ≥ |A| 인 격자 점은 없다(h40 n 격자 규칙)
            r.update(item="D0-P0(L1 조건)", n=n, side=side, alias=al)
            d0[(side, n)] = r
            vs.append(r.get("verdict4"))
            rows.append(r)
        l1c[side] = None if (not vs or any(v in NA_V for v in vs)) else any(v == "우세" for v in vs)
    for n in sorted({n for s_, n in d0 if s_ == "확충판"} & {n for s_, n in d0 if s_ == "기준"}):
        lab = f"D0-P0|n{'all' if n == -1 else n}"
        sh[lab] = l28_shift(d0[("기준", n)].get("verdict4"), d0[("확충판", n)].get("verdict4"))
        d0[("확충판", n)]["l28"] = sh[lab]
    for ma, mb, n in (("R1", "P0", -1), ("P1", "P0", 10)):
        gA = H._g(ma, n, XB.LAM_BASE) if ma == "R1" else H._g(ma, n)
        sr = {}
        for side, al in sides:
            r = _region_row(tms_lg, al, gA, H.P0_GRP, nboot)
            r.update(item=f"{ma}-{mb}", n=n, side=side, alias=al)
            sr[side] = r
            rows.append(r)
        lab = f"{ma}-{mb}|n{'all' if n == -1 else n}"
        sh[lab] = l28_shift(sr["기준"].get("verdict4"), sr["확충판"].get("verdict4"))
        sr["확충판"]["l28"] = sh[lab]
    l1s = "판정 불가" if l1c.get("확충판") is None or l1c.get("기준") is None else ("같음" if l1c["확충판"] == l1c["기준"] else "다름")
    vrow = dict(item="L40 분류", scope="verdict", target=f"{aug}|x", reference=f"{ref}|x", l28=worst_shift(sh.values()),
                stat=f"L1 지역 조건 {l1s}(기준 {l1c.get('기준')}, 확충판 {l1c.get('확충판')}); " + "; ".join(f"{k_}: {v}" for k_, v in sh.items()),
                note="기준 표는 같은 작업에서 다시 적합했다(1절 플랫폼: 한 대비는 한 작업·한 노드 종류 안에서 닫는다)")
    return rows, vrow


def region_reason(r, el) -> str:
    """판정 불가·행 없음 지역 행의 사유(1절 '행 없음·실패'): 행 없음, 사용 분할 수, 채점 블록 합집합, CI 유무."""
    if r.get("verdict4") == "행 없음":
        return str(r.get("reason") or "행 없음(Holm 가족에 p 1)")
    out = []
    al = str(r.get("target", "")).split("|")[0]
    ns, ne = r.get("n_splits"), r.get("n_splits_expected")
    try:
        if ns is not None and ne is not None and int(ns) < int(ne):
            out.append(f"분할 {int(ns)}/{int(ne)}")
    except (TypeError, ValueError):
        pass
    if len(el) and al in el.index and "nb_union" in el.columns:
        nbu = el.loc[al, "nb_union"]
        if pd.notna(nbu) and int(nbu) < MIN_UNION:
            out.append(f"채점 블록 합집합 {int(nbu)} < {MIN_UNION}")
    if not all(np.isfinite(float(r.get(c_, np.nan) if r.get(c_) is not None else np.nan)) for c_ in ("ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq")):
        out.append("CI 없음(점 추정)")
    return "; ".join(out)


def summarize_core(tms_lg, tms_w4, specs, elig, units_w4=(), nboot=None, natl_info=None, nlab=None):
    """집계 본체(봉인 전 표). 반환 dict(tests, holm, sentences, meta). 화면에 쓰지 않는다.
    natl_info = {별칭: (점 추정 여부, 사유)}(natl_status: 5절 T0 + 24 h 마감 규칙). 없으면 적격 표만 본다.
    nlab = {별칭: 전량 실제 라벨 수}(L40 의 L1 조건에서 전량 행을 넣을지 정한다)."""
    el = elig.set_index("alias") if len(elig) else pd.DataFrame()

    def is_elig(al):
        return bool(al in el.index and bool(el.loc[al, "eligible"]))

    def natl_po(al):
        if natl_info is not None and al in natl_info:
            return bool(natl_info[al][0]) or not is_elig(al), str(natl_info[al][1])
        return not is_elig(al), ("" if is_elig(al) else "부적격")
    run = {al: s for al, s in specs.items() if s.get("run", True)}
    new_all = [al for al, s in run.items() if s["kind"] == KIND_NEW]
    new_el = [al for al in new_all if is_elig(al)]
    k = len(new_el)
    names = [f"{al}|x" for al in new_el]
    rows, fam_lab, fam_p, fam_row = [], [], [], []
    sentences = dict(XF=[], NAtlantic=[], augment=[], notes=[])
    for hyp, item, n in XF_HYPS:
        gA, gB = xf_groups(hyp)
        if k:
            rr, per = XB.contrast_pool(tms_lg, names, gA, gB, label=hyp, kind="same", registered=k, nboot=nboot)
            have = {r["target"] for r in rr if r.get("scope") == "region"}
            for nm in names:                                                   # 1절 '행 없음·실패': 적격 지역의 빠진 행은 판정 불가로 남기고 Holm 에 p 1
                if nm not in have:
                    rr.append(dict(target=nm, scope="region", verdict4="행 없음", p_two=1.0, reason="행 없음(Holm 가족에 p 1)"))
            _row_meta(rr, test_id=hyp, item=item, n=n, group="new_region", hypothesis=True, blind="맹검", role="주")
            for r in rr:
                if r.get("scope") == "region":
                    r["reason"] = region_reason(r, el)
                    fam_lab.append(f"{hyp}|{r['target']}"); fam_p.append(r.get("p_two", 1.0)); fam_row.append(r)
            rows += rr
        # 점 추정만인 새 지역(부적격): 서술 행
        po = [f"{al}|x" for al in new_all if al not in new_el]
        if po:
            rp, _ = XB.contrast_pool(tms_lg, po, gA, gB, label=hyp, kind="same", registered=len(po), nboot=nboot)
            rows += _row_meta([r for r in rp if r.get("scope") == "region"], test_id=hyp, item=item, n=n, group="new_region_point_only",
                              hypothesis=False, blind="맹검", role="서술(점 추정, 부적격)")
        for al, s in run.items():
            if s["kind"] in (KIND_NATL, KIND_NATL_FULL):
                pnt, why = natl_po(al)
                r1 = _region_row(tms_lg, al, gA, gB, nboot)
                rows += _row_meta([r1], test_id=hyp, item=item, n=n, group="natl_v5", hypothesis=False, blind=s.get("blind", ""), role="민감도",
                                  point_only=pnt, point_only_reason=why)
    # Holm(m = 3 × 적격 지역 수, 보조 열. 1절: 행 없음·판정 불가는 p 1)
    holm = XB.holm_table(fam_lab, fam_p, m=3 * k) if k else pd.DataFrame(columns=["label", "p", "rank", "multiplier", "p_holm", "m"])
    for r, ph in zip(fam_row, holm.p_holm.values if len(holm) else []):
        r["p_holm"] = float(ph)
        r["holm_m"] = 3 * k
    # 문장
    if k == 0:
        sentences["XF"].append(dict(hyp="XF-1–XF-3", status="시험하지 못함(적격 0곳)", sentence=ZERO_SENTENCE))
        if any(s["kind"] in (KIND_AUG, KIND_AK) for s in specs.values()) or new_all:
            sentences["notes"].append("하위 과제·셀 보강만 더해졌으므로 독립 지역 문장을 쓰지 않고 대상 수로만 쓴다(2.6 해석 문장 '하위 과제만 추가')")
    for hyp, item, n in XF_HYPS:
        T = _templates(hyp)
        hr = [r for r in rows if r.get("test_id") == hyp and r.get("group") == "new_region"]
        reg = [r for r in hr if r.get("scope") == "region"]

        def reg_sentence(r):
            v = r.get("verdict4")
            vv = "판정 불가" if v in NA_V else v
            return dict(hyp=hyp, scope="region", target=r["target"], verdict=vv, reason=r.get("reason", ""),
                        sentence=compose(T, vv, 1, r.get("delta"), r.get("p_holm"), reason=r.get("reason", "")))
        if k >= 2:
            mr = [r for r in hr if r.get("scope") == "MEAN"]
            if mr:
                m_ = mr[0]
                sentences["XF"].append(dict(hyp=hyp, scope="MEAN", verdict=m_.get("verdict4"), pool=m_.get("pool", ""),
                                            sentence=compose(T, m_.get("verdict4"), k, m_.get("delta"), None, XB.worse_regions(hr),
                                                             reason=m_.get("ci_flag", "")),
                                            note="층화 평균 행은 Holm 가족 밖(보조). Holm 보정 p 는 지역 행에 있다"))
        if k >= 1:
            sentences["XF"] += [reg_sentence(r) for r in reg]
        for al, s in run.items():
            if s["kind"] in (KIND_NATL, KIND_NATL_FULL):
                rn = [r for r in rows if r.get("test_id") == hyp and r.get("group") == "natl_v5" and r.get("target") == f"{al}|x"]
                rn = [r for r in rn if r.get("verdict4") != "행 없음"]
                nsrc = len({v for v in str(el.loc[al, "xf_sources"]).split(";") if v and v != "nan"}) if (al in el.index and "xf_sources" in el.columns) else 0
                pnt, why = natl_po(al)
                sentences["NAtlantic"].append(dict(hyp=hyp, alias=al, point_only=pnt, reason=why,
                                                   sentence=natl_sentence(hyp, rn[0] if rn else None, nsrc, pnt)))
    for al, s in specs.items():
        if s["kind"] in (KIND_NATL, KIND_NATL_FULL) and not s.get("run", True):
            sentences["NAtlantic"].append(dict(hyp="XF-1–XF-3", alias=al, point_only=True, reason=s.get("run_skip", ""),
                                               sentence=f"북대서양 약관 확인분 판에 더한 공개 자료 셀이 없어 민감도 행을 계산하지 않았다({s.get('run_skip', '')})."))
    # 셀 보강: L40 형식(확충판과 같은 작업의 기준 표)
    for al, s in run.items():
        if s["kind"] != KIND_AUG:
            continue
        ref = s.get("ref", "")
        if not ref or ref not in run:
            rows.append(dict(test_id="L40(XF 셀 보강)", item="L40 분류", scope="verdict", target=f"{al}|x", group=KIND_AUG, hypothesis=False,
                             role="서술(L40 형식)", blind=s.get("blind", ""), l28="판정 불가(기준 표 없음)"))
            continue
        lr, vr = l40_rows(tms_lg, al, ref, nlab=nlab, nboot=nboot)
        rows += _row_meta(lr, test_id="L40(XF 셀 보강)", group=KIND_AUG, hypothesis=False, role="서술(L40 형식)", blind=s.get("blind", ""))
        rows += _row_meta([vr], test_id="L40(XF 셀 보강)", group=KIND_AUG, hypothesis=False, role="서술(L40 형식)", blind=s.get("blind", ""))
        sentences["augment"].append(dict(alias=al, target=s["target"], n_new_cells=len(s.get("keep_v5", [])), l28=vr["l28"], stat=vr["stat"],
                                         sentence=f"셀 보강({s['target']}, 새 셀 {len(s.get('keep_v5', []))}개)의 L40 형식 민감도 분류는 '{vr['l28']}' 였다"
                                                  f"(기준 표와 같은 작업에서 비교, {vr['stat'].split(';')[0]})."))
    # XF-4: 규칙 W − R1(0.25)(n 10·40·160), WF4 팔이 있는 실행 spec(셀 보강 기준 표 제외), 지역 행(서술)
    for al, s in run.items():
        if s["kind"] == KIND_AUG_REF or "wf4" not in s.get("parts", ["lg", "wf4"]):
            continue
        pnt = natl_po(al)[0] if s["kind"] in (KIND_NATL, KIND_NATL_FULL) else not is_elig(al)
        for n in XF4_NS:
            gW, gR = xf4_groups(n)
            r4, _ = XB.contrast_pool(tms_w4, [f"{al}~wf4|x"], gW, gR, label="XF-4", kind="same", registered=1, nboot=nboot)
            rows += _row_meta([r for r in r4 if r.get("scope") == "region"], test_id="XF-4", item="W-R1(0.25)", n=n, group=s["kind"],
                              hypothesis=False, blind=s.get("blind", ""), role="보조(서술)", point_only=pnt)
    diag = {}
    for u in units_w4 or []:
        al = str(u.get("target", ""))
        for v in u.get("diag") or []:
            diag.setdefault(al, []).append(float(v["abs_bias"]))
    for al, vals in sorted(diag.items()):
        rows.append(dict(test_id="XF-4", item="진단값 |b10|", scope="region", target=f"{al}~wf4|x", group=specs.get(al, {}).get("kind", ""),
                         hypothesis=False, role="보조(서술)", diag_abs_mean=float(np.mean(vals)), n_diag=len(vals)))
    tests = XB.clean_rows(rows)
    meta = dict(k_eligible=k, eligible=new_el, new_point_only=[al for al in new_all if al not in new_el],
                natl=[al for al, s in specs.items() if s["kind"] in (KIND_NATL, KIND_NATL_FULL)],
                natl_status={al: dict(point_only=natl_po(al)[0], reason=natl_po(al)[1]) for al, s in specs.items() if s["kind"] in (KIND_NATL, KIND_NATL_FULL)},
                augment=[al for al, s in run.items() if s["kind"] == KIND_AUG], augment_ref=[al for al, s in run.items() if s["kind"] == KIND_AUG_REF],
                f2_only=[al for al, s in specs.items() if s["kind"] == KIND_AK], holm_m=3 * k,
                rule="XF-1–XF-3: 지역 행 4분 판정(δ 0.5, 보조 δ 1.0·δ_rel), 2곳 이상이면 층화 평균(보조), Holm m = 3 × 적격 지역 수(보조 열, 빠진 행 p 1). "
                     "셀 보강: L40 형식(L1 지역 조건, 전량 R1 − P0, n 10 P1 − P0, L28 분류, 같은 작업의 기준 표)",
                xb_xc_rows="XF-4 의 XB 행은 이 스크립트의 --xb 진입점(register_xf_aliases + v5 연도 정합 도일), XC 행은 x_workflow_end_to_end 가 "
                           "register_xf_aliases 로 낸다")
    return dict(tests=tests, holm=holm, sentences=sentences, meta=meta)


def unit_table_check(units, specs):
    """조각 unit.json 의 table_sha256 이 지금 spec 의 표 해시와 같은지(동결 뒤 다른 판의 표로 적합한 조각을 섞지 않는다). 다르면 멈춘다."""
    bad = []
    for u in units:
        al = str(u.get("target", ""))
        s = specs.get(al)
        if s is None:
            bad.append(f"{al}(spec 없음)")
        elif str(u.get("table_sha256", "")) != str(s.get("table", {}).get("sha256", "")):
            bad.append(f"{al} s{u.get('split')} {u.get('variant', '')}")
    if bad:
        raise SystemExit(f"[집계] 조각의 실행 표 해시가 spec 과 다르다 {len(bad)}개: {bad[:10]}")
    return len(units)


def summarize(a, P=None, nboot=None, require_gate=True, require_freeze=True):
    """조각 → 봉인 표. 거부 조건: 재현 관문(xf_repro_gate.json)이 통과하지 않았거나(계획 0.3 열람 순서 5, --allow-no-gate 는 기록과 함께 넘어간다),
    관문 기록의 v5 해시가 지금 v5 와 다르거나, 적격 동결 기록(최종 부록 XC-F3)과 적격 표·spec·v5 해시가 다르거나, 조각의 표 해시가 spec 과 다르다.
    NAtlantic v5 는 첫 세기 마감 규칙(natl_status)으로 점 추정 여부를 정한다."""
    P = P or a.P
    nb = XB.local_limits(a.threads, a.nboot if nboot is None else nboot, a.ARGV)[1]
    gate = json.loads(P.gate.read_text()) if P.gate.exists() else None
    if require_gate and not (gate and gate.get("passed")) and not a.allow_no_gate:
        raise SystemExit(f"[집계] 재현 관문 기록 {P.gate} 이 없거나 통과하지 않았다(계획 0.3 열람 순서 5). --repro-gate 를 먼저 한다")
    if require_gate and gate and gate.get("passed") and gate.get("v5_sha256") != (XB.sha256_file(P.v5_table) if P.v5_table.exists() else ""):
        raise SystemExit("[집계] 재현 관문 기록의 v5 해시가 지금 v5 와 다르다. --repro-gate 를 다시 한다")
    frz = check_frozen(P, require=require_freeze)
    specs = load_specs(P)
    elig = pd.read_csv(P.elig) if P.elig.exists() else pd.DataFrame(columns=["alias", "eligible"])
    nst = natl_status(P, specs, elig)
    po = point_only_map(P, specs, elig)

    def pof(name):
        al = str(name).split("|")[0]
        al = al[:-len("~wf4")] if al.endswith("~wf4") else al
        return bool(po.get(al, True))
    tms, units, _ = XB.load_tms(P.shards, a.tag, nb, point_only=pof)
    unit_table_check(units, specs)
    tms_lg = {k_: v for k_, v in tms.items() if not k_.split("|")[0].endswith("~wf4")}
    tms_w4 = {k_: v for k_, v in tms.items() if k_.split("|")[0].endswith("~wf4")}
    u_w4 = [dict(u, target=str(u.get("target", ""))) for u in units if u.get("variant") == "wf4"]
    nlab = {}
    for u in units:
        if u.get("variant") == "lg" and u.get("n_A") is not None:
            nlab[str(u.get("target"))] = max(nlab.get(str(u.get("target")), 0), int(u["n_A"]))
    res = summarize_core(tms_lg, tms_w4, specs, elig, u_w4, nboot=nb, natl_info=nst, nlab=nlab)
    res["meta"].update(created=now_kst().isoformat(timespec="seconds"), nboot=int(nb), n_shards=len(units), n_stores=len(tms),
                       gate_passed=bool(gate and gate.get("passed")), gate_skipped=bool(a.allow_no_gate and not (gate and gate.get("passed"))),
                       frozen=bool(frz), frozen_record=dict(created=frz.get("created"), elig_sha256=frz.get("elig_sha256")) if frz else None,
                       v5_sha256=XB.sha256_file(P.v5_table) if P.v5_table.exists() else "", script_sha256=XB.sha256_file(__file__))
    out = XB.write_sealed(EXP_NAME, {"xf_tests.csv": res["tests"], "xf_holm.csv": res["holm"], "xf_sentences.json": res["sentences"],
                                     "xf_summary_meta.json": res["meta"]}, root=P.base)
    print(f"[집계] 저장소 {len(tms)} · 조각 {len(units)} · 봉인 파일 {len(out)}", flush=True)
    return res


# ================================================================ 8. 부록 XC-F3 목록
def write_xc_f3(P, elig=None, specs=None, now=None):
    """부록 XC-F3(0.3 커밋 규칙, 2.3 F3 계열): 이번 조사 뒤 처음 확보해 XF 에서 적격 판정을 받은 새 macro 지역 가운데 분할 201–210 에서 다시 센
    적격 지역의 목록. NAtlantic v5 와 셀 보강은 넣지 않는다. 0곳이면 XC-F3 는 p 1 과 '시험하지 못함'. 라벨 값을 쓰지 않는다.
    부록 XC-0 과 같은 방식으로 선택 스크립트(이 파일)·v5·spec·적격 표의 sha256 을 적는다. XF 자료 마감 뒤 처음 만든 최종판이 적격 동결 기록이며
    다시 쓰지 않는다(같은 해시면 기존 최종판을 돌려주고, 다르면 멈춘다). 알래스카 하위 과제(AL-7)는 F2 서술 전용 별칭으로 따로 적는다."""
    now = now or now_kst()
    old = final_xcf3(P)
    if old is not None:
        cur = frozen_hashes(P)
        bad = [k_ for k_, v in cur.items() if old.get(k_) != v]
        if bad:
            raise SystemExit(f"[XC-F3] 최종판({old.get('created')})이 있고 그 뒤 바뀐 파일이 있다: {bad}. 최종판은 다시 쓰지 않는다(적격 동결)")
        for t in old.get("appendix_text", []):
            print(t, flush=True)
        return old
    elig = elig if elig is not None else pd.read_csv(P.elig)
    specs = specs if specs is not None else load_specs(P)
    e = elig.copy()
    f3 = e[(e.kind == KIND_NEW) & e.eligible.astype(bool) & e.eligible_xc.astype(bool) & e.lic_ok.astype(bool)]

    def rel(al):
        return os.path.relpath(spec_dir(P, al), ROOT)                       # ROOT 기준 상대 경로(Rescale 노드에서도 같은 곳)
    rows = []
    for r in f3.itertuples(index=False):
        rows.append(dict(alias=r.alias, store=f"{r.alias}|x", target=r.target, run_table=rel(r.alias),
                         table_sha256=r.table_sha256, xc_keep=r.xc_keep, xc_n_valid=int(r.xc_n_valid), xc_nb_union=int(r.xc_nb_union),
                         xc_nb_min=int(r.xc_nb_min), few_blocks=bool(r.xc_few_blocks), n_cells=int(r.n_cells), n_blocks=int(r.n_blocks),
                         blind="맹검(실행 전 등록 확인 시험)"))
    f2 = []
    for r in e[e.kind == KIND_AK].itertuples(index=False):
        f2.append(dict(alias=r.alias, store=f"{r.alias}|x", target=r.target, run_table=rel(r.alias), table_sha256=r.table_sha256,
                       eligible=bool(r.eligible), eligible_xc=bool(r.eligible_xc), xc_keep=r.xc_keep, n_cells=int(r.n_cells), n_blocks=int(r.n_blocks),
                       role="XC F2 서술 전용(독립 지역 아님, XD 개발 과제 아님). XC 는 register_xf_aliases 로 이 별칭을 연다"))
    reason = {KIND_NATL: "NAtlantic v5 는 민감도 행만(2.3, WRAPUP 7.2 (a)6)", KIND_NATL_FULL: "NAtlantic v5 는 민감도 행만(2.3, WRAPUP 7.2 (a)6)",
              KIND_AUG: "셀 보강(독립 지역 아님)", KIND_AUG_REF: "셀 보강의 기준 표", KIND_AK: "알래스카 하위 과제(F2 서술 전용)"}
    excluded = [dict(alias=r.alias, kind=r.kind, reason=reason.get(r.kind) or ("분할 1–5 부적격" if not bool(r.eligible) else "분할 201–210 부적격"))
                for r in e.itertuples(index=False) if r.alias not in set(f3.alias)]
    k = len(rows)
    final = bool(now > DEADLINE_DATA)
    hs = frozen_hashes(P)
    script_sha = XB.sha256_file(__file__)
    specs_sha = hs["specs_sha256"]
    status = ("시험하지 못함(적격 0곳, Holm 가족에 p 1)" if k == 0 else
              ("그 지역 행으로 시험한다" if k == 1 else f"적격 {k}곳의 층화 평균으로 시험한다"))
    text = ["부록 XC-F3(" + ("XF 마감 뒤 확정" if final else "XF 마감 전 잠정. 마감 T0 + 7일 = " + DEADLINE_DATA.strftime("%Y-%m-%d %H:%M:%S %z")) + ")",
            f"- F3 적격 새 독립 지역: {k}곳" + (": " + ", ".join(f"{r['alias']}(분할 {r['xc_keep']}, 채점 블록 합집합 {r['xc_nb_union']})" for r in rows) if k else ""),
            f"- XC-F3 처리: {status}. 적격은 분할 201–210 에서 xbatch 새 seed 규칙으로 다시 셌다(2.6 적격 규칙).",
            "- F2 서술 전용 별칭: " + (", ".join(f"{x['alias']}(대상 셀 {x['n_cells']}, 블록 {x['n_blocks']})" for x in f2) if f2 else "없음"),
            "- 넣지 않은 항목: " + ("; ".join(f"{x['alias']}({x['reason']})" for x in excluded) if excluded else "없음"),
            f"- 선택 스크립트 {Path(__file__).name} sha256 {script_sha}; v5 sha256 {hs['v5_sha256'] or '없음'}; xf_specs.json sha256 "
            f"{specs_sha or '없음'}; xf_eligibility.csv sha256 {hs['elig_sha256'] or '없음'}"]
    if k == 0:
        text.append("- 해석 문장(2.3 XC-F3 표의 F3 0곳 갈래): '공개 자료로 확보한 새 독립 지역이 적격 조건을 만족하지 못해 외부 계열 시험은 하지 못했다.'")
    doc = dict(created=now.isoformat(timespec="seconds"), final=final, deadline_data=DEADLINE_DATA.isoformat(), k=k, status=status, f3=rows,
               f2_only=f2, excluded=excluded, appendix_text=text, script=str(Path(__file__).name), script_sha256=script_sha,
               elig_sha256=hs["elig_sha256"], specs_sha256=specs_sha, v5_sha256=hs["v5_sha256"],
               points_manifest_sha256=XB.sha256_file(P.points_manifest) if P.points_manifest.exists() else "",
               rule="kind = new_region ∩ 분할 1–5 적격 ∩ 분할 201–210 적격 ∩ 약관 확인분. 최종판(final)은 적격 동결 기록이다(LGD 6B.4 '판정 시점')")
    XB.check_out_dir(P.root)
    P.root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["alias", "store", "target", "run_table", "table_sha256", "xc_keep", "xc_n_valid", "xc_nb_union", "xc_nb_min",
                                "few_blocks", "n_cells", "n_blocks", "blind"]).to_csv(P.xcf3_csv, index=False)
    XB._atomic_write(P.xcf3_json, json.dumps(doc, ensure_ascii=False, indent=1))
    for t in text:
        print(t, flush=True)
    return doc


# ================================================================ 9. 재현 관문
def lgd_target_ids(lab_v4, macro, lic_only=False):
    g = lab_v4[(lab_v4.part == "new") & (lab_v4.lgd_role == "target") & (lab_v4.macro_v4 == macro)]
    if lic_only:
        g = g[_lic_ok(g)]
    return sorted(int(v) for v in g.loc_id)


def repro_gate(a, P=None, gate_b1=False, lgd_dir=None):
    """2.6 재현 관문. 결과는 참·거짓과 셀 수만 쓴다. v5 가 없으면 중단한다."""
    P = P or a.P
    lgd_dir = Path(lgd_dir or XB.LGD_DIR)
    if not P.v5_table.exists():
        raise SystemExit(f"[관문] {P.v5_table} 가 없다. --stage cells,cov,v5,tables 를 먼저 한다")
    checks = []

    def chk(name, ok, **kw):
        checks.append(dict(name=name, ok=bool(ok), **kw))
        print(f"  [관문] {name}: {'통과' if ok else '불통과'} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)
    l4 = V4_TABLE.read_text(encoding="utf-8").splitlines()
    l5 = P.v5_table.read_text(encoding="utf-8").splitlines()
    chk("v5_v4_rows_bytes", l5[:len(l4)] == l4, n_v4=len(l4) - 1, n_v5=len(l5) - 1)
    s4 = V4_SOIL.read_text(encoding="utf-8").splitlines()
    s5 = P.v5_soil.read_text(encoding="utf-8").splitlines()
    chk("soil_v5_v4_rows_bytes", s5[:len(s4)] == s4, n_v4=len(s4) - 1, n_v5=len(s5) - 1)
    m4 = json.loads(V4_META.read_text())
    chk("v4_table_matches_meta", XB.sha256_file(V4_TABLE) == m4["outputs"]["fidelity_base_v4.csv"], basis="fidelity_base_v4_meta.outputs")
    chk("v4_soil_matches_meta", XB.sha256_file(V4_SOIL) == m4["outputs"]["e5_soil_tdd_v4.csv"], basis="fidelity_base_v4_meta.outputs")
    lic = pd.read_csv(PROC / "lgd" / "lgd_eligibility_lic.csv")
    rec = dict(zip(lic.spec, lic.table_sha256))
    man = json.loads(LGD_MANIFEST.read_text()).get("specs", {})
    for spec in XB.RUN_TABLE_DIRS:
        f = lgd_dir / spec / "fidelity_base_v3.csv"
        now = XB.sha256_file(f) if f.exists() else ""
        ref_m = str(man.get(spec, {}).get("table_sha256", ""))
        chk(f"lgd_table_unchanged:{spec}", bool(ref_m) and now == ref_m, basis="lgd_run_manifest.specs.table_sha256", sha256_16=now[:16])
        if spec in rec:
            chk(f"lgd_table_matches_eligibility_lic:{spec}", now == rec[spec], basis="lgd_eligibility_lic.table_sha256")
    lab_v4 = pd.read_csv(V4_LABELS, low_memory=False)
    gdir = P.root / "_gate"
    XB.check_out_dir(gdir)
    _gitignore(gdir)
    v1 = pd.read_csv(PROC / "lgd_eligibility_v1.csv")
    for name, macro, lic_only, ref_dir, ref_row, xc_alias in (("Tibet", "Tibet_LGD", False, "Tibet", ("v1", "Tibet"), None),
                                                              ("NAtlantic_lic", "NAtlantic", True, "NAtlantic_lic", ("lic", "NAtlantic_lic"), "NAtlantic~lic")):
        d = gdir / name
        d.mkdir(parents=True, exist_ok=True)
        keep = lgd_target_ids(lab_v4, macro, lic_only)
        E1.write_run_table_text(P.v5_table, d / "fidelity_base_v3.csv", keep, drop_v3=E1.DROP_V3.get(macro, []))
        soil = d / "e5_soil_tdd_v3.csv"
        if soil.is_symlink() or soil.exists():
            soil.unlink()
        os.symlink(os.path.relpath(P.v5_soil, d), soil)
        ref = lgd_dir / ref_dir / "fidelity_base_v3.csv"
        chk(f"run_table_rebuilt_bytes:{name}", (d / "fidelity_base_v3.csv").read_bytes() == ref.read_bytes(), n_keep=len(keep))
        r, _ = count_spec(dict(alias=name, target=macro, kind="gate", keep_v4=keep, keep_v5=[]), d)
        tab, nm = ref_row
        ref_r = (v1[v1.spec == nm] if tab == "v1" else lic[lic.spec == nm]).iloc[0]
        cols = ["n_cells", "n_valid_splits", "nb_union", "min_nb_eval_used", "n_src", "eligible"]
        same = {c: (float(r[c]) == float(ref_r[c])) if c != "eligible" else (bool(r[c]) == bool(ref_r[c])) for c in cols}
        chk(f"eligibility_count_matches_lgd:{name}", all(same.values()), compared=",".join(cols), mismatch=[c for c, v in same.items() if not v])
        if xc_alias:
            df = table_df(d)
            t_idx = target_index(df, macro)
            _, xr = XB.split_plan_new(SimpleNamespace(df=df, target_idx=lambda t, _i=t_idx: _i), macro, XC_SPLITS, XB.PRIOR_LG, XB.MIN_EVAL_BLOCKS)
            try:
                XB.check_appendix_a("xc", xc_alias, xr)
                chk(f"appendix_a_201_210:{xc_alias}", True)
            except AssertionError as e:
                chk(f"appendix_a_201_210:{xc_alias}", False, note=str(e)[:200])
    if gate_b1:
        v1p = v1_points()
        pts = [EXT / k for k in sorted(v1p)]
        work = gdir / "b1_v1"
        run_b1(pts, work, sentinel=False)                                       # v1 에는 절단 행이 있다(원 경로 그대로)
        for f in ("ext_cells_v1.csv", "ext_cells_v1_locs.csv"):
            chk(f"b1_rebuild_bytes:{f}", (work / f).read_bytes() == (EXT / f).read_bytes())
        un = v1_points_unchanged()
        chk("v1_points_unchanged", all(un.values()), n=len(un))
    passed = all(c["ok"] for c in checks)
    doc = dict(created=now_kst().isoformat(timespec="seconds"), passed=passed, checks=checks, v5_sha256=XB.sha256_file(P.v5_table),
               gate_b1=bool(gate_b1), script_sha256=XB.sha256_file(__file__), max_rss_mb=XB.max_rss_mb())
    XB._atomic_write(P.gate, json.dumps(doc, ensure_ascii=False, indent=1, default=str))
    print(f"[관문] 점검 {len(checks)} · 불통과 {sum(not c['ok'] for c in checks)} · 통과 {passed} · 최대 RSS {doc['max_rss_mb']} MB", flush=True)
    return doc


# ================================================================ 10. 스모크(합성 대상 라벨)
def smoke_table(src_dir, dst_dir, target, seed_name, soil_target, spec=None):
    """실행 표를 복사하되 대상 셀(실행 표 안의 macro = target, 하위 지역 spec 이면 블록 대응표의 대상)의 alt_cm 을 합성 라벨
    clip(1.4·√TDD·exp(0.2ε), 5, 590)으로 바꾼다(h51.smoke_table 과 같은 규칙, ε 는 seed_of('xf-smoke', 별칭)). 실제 대상 라벨은 쓰지 않는다.
    원문 줄의 다른 칸은 그대로 둔다. 블록 대응표가 있으면 함께 복사한다."""
    df = table_df(src_dir)
    t_idx = spec_target_idx(df, spec, src_dir)[0] if spec is not None else target_index(df, target)
    tids = set(int(v) for v in df.loc_id.values[t_idx])
    lines = (Path(src_dir) / "fidelity_base_v3.csv").read_text(encoding="utf-8").splitlines()
    hdr = lines[0].split(",")
    i_id, i_alt, i_s = hdr.index("loc_id"), hdr.index("alt_cm"), hdr.index("e5_sqrt_tdd")
    rng = np.random.RandomState(XB.seed_of("xf-smoke", seed_name))
    eps = rng.randn(len(lines))
    out = [lines[0]]
    sv = []
    for ln in lines[1:]:
        f = ln.split(",")
        try:
            sv.append(float(f[i_s]))
        except ValueError:
            pass
    smed = float(np.nanmedian(sv)) if sv else 30.0
    n_syn = 0
    for j, ln in enumerate(lines[1:], start=1):
        f = ln.split(",")
        if int(f[i_id]) in tids:
            try:
                s = float(f[i_s])
            except ValueError:
                s = smed
            s = s if np.isfinite(s) else smed
            f[i_alt] = repr(float(np.clip(1.4 * s * np.exp(0.2 * eps[j]), 5.0, 590.0)))
            ln = ",".join(f)
            n_syn += 1
        out.append(ln)
    d = Path(dst_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "fidelity_base_v3.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    if (Path(src_dir) / SUBMAP_NAME).exists():
        shutil.copy2(Path(src_dir) / SUBMAP_NAME, d / SUBMAP_NAME)
    soil = d / "e5_soil_tdd_v3.csv"
    if soil.is_symlink() or soil.exists():
        soil.unlink()
    os.symlink(os.path.relpath(Path(soil_target).resolve(), d.resolve()), soil)
    return dict(n_target_rows_synthetic=n_syn, n_rows=len(lines) - 1)


def smoke(a):
    """합성 대상 라벨 스모크. 첫 실행 spec(없으면 NAtlantic~lic~v5)의 첫 실행 분할에서 LG 방법 축(n {0, 10, 전량}, 추출 1, seed 1)과 WF4 팔
    (n {10, 전량}, 추출 1, seed 1)을 적합하고 봉인 집계까지 지난다. 산출 <산출>/smoke/. 화면에는 단위 수·적합 수·시간만 쓴다."""
    t0 = time.time()
    env = XB.smoke_env_report()
    if env["warn"] and not XB.run_permitted(a.ARGV):
        raise SystemExit(f"[smoke] 로컬 스모크 규약(코어 4개, 스레드 2)과 어긋난다: {env['warn']}. taskset -c <코어 4개>, OMP_NUM_THREADS=2 로 다시 한다")
    specs = select_specs(a)
    el = pd.read_csv(a.P.elig) if a.P.elig.exists() else None
    pick = [al for al in sorted(specs) if specs[al].get("run", True)] or ([NATL_ALIAS] if NATL_ALIAS in specs else sorted(specs)[:1])
    pick = pick[:1] if not a.SPECS_SEL else pick
    S = Paths(a.P.root / "smoke")
    S.check()
    _gitignore(S.root)
    sspecs, recs = {}, {}
    for al in pick:
        s = copy.deepcopy(specs[al])
        src = spec_dir(a.P, al)
        dst = spec_dir(S, al)
        soil_target = (src / "e5_soil_tdd_v3.csv").resolve()
        recs[al] = smoke_table(src, dst, s["target"], al, soil_target, spec=s)
        s["table"] = dict(dir=os.path.relpath(dst, S.root), sha256=XB.sha256_file(dst / "fidelity_base_v3.csv"))
        rs = _splits_of(el.set_index("alias").loc[al, "run_splits"]) if (el is not None and al in set(el.alias)) else [1]
        s["run_splits"] = (rs or [1])[:1]
        s["run"] = True
        sspecs[al] = s
    save_specs(S, sspecs)
    b = copy.copy(a)
    b.P, b.SPECS, b.tag, b.resume = S, sspecs, f"{TAG}_smoke", False
    b.LG_NGRID, b.LG_DRAWS, b.LG_SEEDS = SMOKE_LG_N_GRID, 1, 1
    b.W4_GRID, b.W4_SEEDS, b.W4_DRAWS_CAP = SMOKE_WF4_GRID, 1, 1
    units = [(al, s["run_splits"][0]) for al, s in sspecs.items()]
    S.shards.mkdir(parents=True, exist_ok=True)
    res = [run_unit(b, al, sp) for al, sp in units]
    el_s = pd.DataFrame([dict(alias=al, eligible=True, run_splits=str(sspecs[al]["run_splits"][0])) for al in sspecs])
    el_s.to_csv(S.elig, index=False)
    b.allow_no_gate = True
    summarize(b, P=S, nboot=min(int(a.nboot), 200), require_gate=False, require_freeze=False)
    meta = dict(created=now_kst().isoformat(timespec="seconds"), units=[list(u) for u in units], results=res, tables=recs, wall_s=round(time.time() - t0, 1),
                settings=dict(lg_n_grid=b.LG_NGRID, lg_draws=1, lg_seeds=1, wf4_grid=list(b.W4_GRID), wf4_seeds=1, cb_iters=int(a.cb_iters),
                              threads=int(a.threads)), smoke_env=env, max_rss_mb=XB.max_rss_mb(), label_rule="clip(1.4·√TDD·exp(0.2ε), 5, 590)")
    XB._atomic_write(S.root / "xf_smoke_meta.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[smoke] 단위 {len(units)} · 적합 " + ", ".join(f"{r['alias']} s{r['split']} LG {r.get('fits_lg', 0)} WF4 {r.get('fits_wf4', 0)} "
                                                       f"{r['elapsed_s']}s" for r in res) + f" · 벽시계 {time.time() - t0:.0f}s · 최대 RSS {XB.max_rss_mb()} MB",
          flush=True)
    return meta


# ================================================================ 11. 조립(단계)
STAGES = ("cells", "cov", "v5", "tdd", "tables")


def build(a, stages, now=None):
    """조립 단계(cells, cov, v5, tdd, tables). 화면에는 행 수·셀 수·블록 수만 쓴다. 적격 동결(최종 부록 XC-F3) 뒤에는 v5·실행 표를 바꾸는 단계
    (cells, v5, tdd, tables)를 거부한다(LGD 6B.4 '판정 시점'). 점 자료는 셀 조립 전에 6B.4 지리 정의와 대조하고 목록을 기록·동결한다."""
    P = a.P
    P.check()
    t0 = time.time()
    now = now or now_kst()
    if final_xcf3(P) is not None and set(stages) & {"cells", "v5", "tdd", "tables"}:
        raise SystemExit(f"[build] 적격이 최종 부록 XC-F3 로 동결되었다. 단계 {sorted(set(stages) & {'cells', 'v5', 'tdd', 'tables'})} 를 다시 하지 않는다")
    rec = dict(created=now.isoformat(timespec="seconds"), stages=list(stages))
    if P.build_meta.exists():                                              # 앞선 조립의 점 자료 기록을 잇는다(cells 없이 다른 단계만 할 때)
        rec["points"] = json.loads(P.build_meta.read_text()).get("points", {})
    known = known_macros()
    if "cells" in stages:
        pts = xf_point_files(P)
        rec["points_checked_rows"] = check_point_macros(pts, known)
        man = points_manifest(P, pts, now=now)
        rec["points"] = {p.name: XB.sha256_file(p) for p in pts}
        rec["points_manifest"] = dict(frozen=bool(man.get("frozen")), n=len(man.get("files", {})), sha256=XB.sha256_file(P.points_manifest))
        _gitignore(P.points)
        cells, cmeta = run_b1(pts, P.build)
        v4c = pd.read_csv(V1_CELLS, usecols=["cell_uid", "label_set", "lat", "lon", "in_v4", "excl_reason"], low_memory=False)
        cls = classify_cells(cells, v4c, known)
        cls.to_csv(P.cells_cls, index=False)
        P.v5.mkdir(parents=True, exist_ok=True)
        cls[[c for c in ROLE_COLS if c in cls.columns]].to_csv(P.cells_roles, index=False)
        rec["cells"] = dict(n_points_files=len(pts), n_cells=int(len(cls)), by_role=dict(Counter(cls.xf_role)) if len(cls) else {},
                            n_in_v5=int(cls.in_v5.sum()) if len(cls) else 0, n_target=int(cls.xf_target.sum()) if len(cls) else 0,
                            n_v1_cell_not_v4=int(cls.v1_cell_not_v4.sum()) if len(cls) else 0)
        print(f"[cells] 점 자료 {len(pts)}개 · 셀 {len(cls)} · in_v5 {rec['cells']['n_in_v5']} · 대상 {rec['cells']['n_target']} · 역할 "
              f"{rec['cells']['by_role']} · 점 자료 목록 동결 {rec['points_manifest']['frozen']}", flush=True)
    cls = pd.read_csv(P.cells_cls, low_memory=False) if P.cells_cls.exists() else classify_cells(empty_cells(), pd.DataFrame(
        columns=["cell_uid", "label_set", "lat", "lon", "in_v4", "excl_reason"]), known)
    cov_p = P.cov / "cells_xf_cov.csv"
    if "cov" in stages:
        cov = run_cov(P, cls, fetch=a.fetch)
        rec["cov"] = dict(n=int(len(cov)))
        print(f"[cov] 셀 {len(cov)} · 내려받기 {'허용' if a.fetch else '안 함'}", flush=True)
    cov = pd.read_csv(cov_p, low_memory=False) if cov_p.exists() else pd.DataFrame(columns=["id", "cell_uid"])
    if "v5" in stages:
        if int(cls.in_v5.sum()) and not len(cov):
            raise SystemExit("[v5] 공변량 표가 없다. --stage cov 를 먼저 한다")
        meta = build_v5(P, cls, cov)
        rec["v5"] = meta["n_rows"]
        sm = extended_submap(P.v5_table)
        sm.to_csv(P.v5_submap, index=False)
        rec["v5_submap"] = dict(n=int(len(sm)), n_extra=int(sm.extra.sum()), sha256=XB.sha256_file(P.v5_submap))
        print(f"[v5] v4 행 {meta['n_rows']['v4']} · 새 행 {meta['n_rows']['new']} · v4 줄 바이트 같음 {meta['v4_rows_bytes_equal']} · "
              f"하위 지역 대응표 새 블록 {rec['v5_submap']['n_extra']}", flush=True)
    if "tdd" in stages:
        tm = write_tdd_v5(P)
        rec["tdd"] = dict(n_new=tm["n_new"], check=tm["check"], sha256=tm["sha256"])
        print(f"[tdd] v5 새 행 {tm['n_new']} · 재현 점검 v3 {tm['check']['n']}셀 · sha256 {tm['sha256'][:16]}", flush=True)
    if "tables" in stages:
        lab_v4 = pd.read_csv(V4_LABELS, low_memory=False)
        lab_xf = pd.read_csv(P.v5_labels, low_memory=False) if P.v5_labels.exists() else pd.DataFrame(columns=["loc_id", "xf_target"])
        sm = pd.read_csv(P.v5_submap, dtype=dict(parent=str, subregion=str)) if P.v5_submap.exists() else extended_submap(P.v5_table)
        specs = make_specs(lab_v4, lab_xf, a.NATL_CONFIRMED, submap=sm)
        specs = write_tables(P, specs, submap=sm)
        rec["tables"] = {al: dict(kind=s["kind"], run=s["run"], target=s["target"], n_keep_v4=len(s["keep_v4"]), n_keep_v5=len(s["keep_v5"]),
                                  n_lines=s["table"]["n_new"]) for al, s in specs.items()}
        for al, s in specs.items():
            print(f"[tables] {al}: 종류 {s['kind']} · 대상 {s['target']} · v4 새 행 {len(s['keep_v4'])} · v5 새 행 {len(s['keep_v5'])} · 적합 "
                  f"{'함' if s['run'] else '안 함(' + s['run_skip'] + ')'}", flush=True)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["max_rss_mb"] = XB.max_rss_mb()
    XB._atomic_write(P.build_meta, json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    return rec


# ================================================================ 12. 명령행
def _parse_shard(txt):
    if not txt:
        return None
    k, n = (int(v) for v in str(txt).split("/"))
    if not (0 <= k < n):
        raise SystemExit(f"--shard {txt}: 0 ≤ K < N 이어야 한다")
    return (k, n)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XF_new_regions(계획 2.6) 실행기")
    ap.add_argument("--stage", default="", help="조립 단계 쉼표 목록: cells, cov, v5, tdd, tables")
    ap.add_argument("--fetch", action="store_true", help="공변량 단계에서 DEM 타일과 SoilGrids 창을 내려받는다(덮개 폴더)")
    ap.add_argument("--count-only", action="store_true", help="적격 표와 적합 수(라벨 값 미사용)")
    ap.add_argument("--no-fit-count", action="store_true", help="--count-only 에서 적합 수 세기를 건너뛴다(적격 표만)")
    ap.add_argument("--repro-gate", action="store_true")
    ap.add_argument("--gate-b1", action="store_true", help="관문에 v1 점 자료 셀 재조립 바이트 대조를 더한다")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--allow-no-gate", action="store_true", help="집계: 재현 관문 기록 없이 진행한다(기록에 남긴다)")
    ap.add_argument("--xc-f3", action="store_true", help="부록 XC-F3 목록")
    ap.add_argument("--specs", default="", help="spec 별칭 쉼표 목록(기본 전부)")
    ap.add_argument("--units", default="", help="'별칭:분할' 쉼표 목록")
    ap.add_argument("--shard", default="", help="K/N: 정렬한 작업 단위 가운데 i % N == K 만")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--out-root", default="", help="산출 뿌리(그 아래 XF_new_regions). 허용 뿌리: data/processed/xbatch 와 XBATCH_OUT_ROOTS")
    ap.add_argument("--natl-lic-confirmed", default="", help="약관 회신으로 확인된 NAtlantic 자료원(쉼표). 있으면 NAtlantic~v5 전체 판을 만든다")
    ap.add_argument("--min-mem-gb", type=float, default=30.0, help="시작 전 가용 메모리 하한(GB, 0 = 확인 안 함)")
    ap.add_argument("--max-mem-gb", type=float, default=10.0, help="프로세스 RSS 상한(GB, 감시 스레드. 0 = 감시 안 함. WF_RESCALE=1 일 때만 감시하지 않는다)")
    ap.add_argument("--as-cap-gb", type=float, default=0.0, help="주소 공간 상한(GB, prlimit --as 와 같다. 기본 0 = 걸지 않음)")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    a.threads, _ = XB.local_limits(a.threads, None, a.ARGV)
    if not on_rescale():                                                  # 1절 로컬 자원: --allow-local 이어도 워커당 4스레드, 합계 32스레드 이하
        a.threads = min(int(a.threads), XB.LOCAL_MAX_THREADS)
        if max(int(a.workers), 1) * int(a.threads) > LOCAL_TOTAL_THREADS:
            raise SystemExit(f"[자원] 워커 {a.workers} × 스레드 {a.threads} 가 로컬 상한 {LOCAL_TOTAL_THREADS} 스레드를 넘는다(계획 1절 로컬 자원)")
    a.P = Paths(a.out_root or None)
    a.SPECS_SEL = [v.strip() for v in a.specs.split(",") if v.strip()]
    a.UNITS = [(u.rsplit(":", 1)[0], int(u.rsplit(":", 1)[1])) for u in a.units.split(",") if u.strip()]
    a.SHARD = _parse_shard(a.shard)
    a.NATL_CONFIRMED = tuple(v.strip() for v in a.natl_lic_confirmed.split(",") if v.strip())
    a.LG_NGRID, a.LG_DRAWS, a.LG_SEEDS = LG_N_GRID, LG_DRAWS, LG_SEEDS
    a.W4_GRID, a.W4_SEEDS, a.W4_DRAWS_CAP = WF4_GRID, len(XB.SEEDS), 0
    return a


def rss_mb() -> float:
    """이 프로세스의 현재 RSS(MB, /proc/self/status 의 VmRSS)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


_WATCHDOG = {}


def rss_watchdog(gb, poll_s=2.0):
    """1절 로컬 자원 '작업 하나 10 GB 이하'의 강제: RSS 가 상한을 넘으면 기록을 남기고 프로세스를 끝낸다(종료 코드 3).
    systemd-run --user --scope 는 이 서버에서 실패하고(2026-10-04 확인), 주소 공간 상한(prlimit --as, XB.limit_memory)은 CatBoost·스레드의 가상
    예약(RSS 0.25 GB 일 때 VmSize 9.5 GB)에 걸려 커널 안에서 멈춘다. 그래서 상주 메모리로 강제한다. 한 프로세스에 하나만 둔다."""
    import threading
    if gb <= 0 or _WATCHDOG.get("on"):
        return False
    lim = float(gb) * 1024.0

    def run():
        while True:
            r = rss_mb()
            if np.isfinite(r) and r > lim:
                sys.__stderr__.write(f"[자원] RSS {r:.0f} MB 가 상한 {lim:.0f} MB 를 넘어 끝낸다(계획 1절 로컬 자원)\n")
                sys.__stderr__.flush()
                os._exit(3)
            time.sleep(poll_s)
    threading.Thread(target=run, name="xf_rss_watchdog", daemon=True).start()
    _WATCHDOG["on"] = True
    return True


def _cap_threads(a):
    """로컬(WF_RESCALE=1 아님)에서 이 프로세스의 수치 라이브러리 스레드를 a.threads 로 묶는다. xbatch_core 가 가져올 때 --allow-local 이면
    요청 스레드로 환경 변수를 정했을 수 있어서, 자식 프로세스용 환경 변수와 이미 적재된 BLAS(threadpoolctl)를 함께 묶는다."""
    if on_rescale():
        return None
    for v in XB.THREAD_VARS:
        os.environ[v] = str(int(a.threads))
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(int(a.threads))
    except Exception:                                                     # noqa: BLE001  threadpoolctl 이 없으면 환경 변수만
        return None


def _resources(a, heavy=True):
    """1절 로컬 자원: Rescale(WF_RESCALE=1) 밖이면 --allow-local 이어도 가용 메모리가 하한(기본 30 GB) 아래일 때 최대 1시간까지 1분 간격으로
    기다리고, RSS 상한(기본 10 GB)을 감시한다(워커는 _worker_init 에서 따로 감시한다). 스레드 상한은 parse_args 가 정한다.
    --as-cap-gb 를 주면 주소 공간 상한(XB.limit_memory)도 건다(가상 예약이 커서 기본은 끈다). 반환 적용 기록."""
    relax = on_rescale()
    rec = dict(on_rescale=relax, mem_wait=False, watchdog=False, as_cap=False)
    if heavy and a.min_mem_gb > 0 and not relax:
        XB.require_memory(a.min_mem_gb, wait_s=3600.0, poll_s=60.0)
        rec["mem_wait"] = True
    if not relax and a.max_mem_gb > 0:
        rss_watchdog(a.max_mem_gb)
        rec["watchdog"] = True
    if not relax and a.as_cap_gb > 0:
        rec["as_cap"] = bool(XB.limit_memory(a.as_cap_gb))
    return rec


# ================================================================ 13. XB 진입점(XF-4 의 XB 행)
XB_NO_FIT = ("--smoke", "--count-only", "--summarize-only", "--write-tdd-v4", "--gate")


def xb_prepare(P, require_frozen=True, xs=None):
    """XB(x_multisource_stacking)가 XF 실행 표를 읽게 한다(이 프로세스 안에서만, XB 파일은 고치지 않는다).
    (1) register_xf_aliases 로 xbatch_core.RUN_TABLES 에 XF 별칭을 더한다(--targets-t '<별칭>:x' 가 XF 표를 연다).
    (2) XB 의 tdd_tables 를 감싸 v5 새 행의 연도 정합 도일(xf_tdd_matched_v5.csv, 메타의 sha256 대조)을 더한다(P* 후보). 겹치는 loc_id 가 있으면 멈춘다.
    반환 (XB 모듈, 더한 별칭 dict)."""
    XS = xs or importlib.import_module("x_multisource_stacking")
    add = register_xf_aliases(P, require_frozen=require_frozen)
    if not (P.tdd_v5.exists() and P.tdd_v5_meta.exists()):
        raise SystemExit(f"[xb] v5 연도 정합 도일 표 {P.tdd_v5} 가 없다. --stage tdd 를 먼저 한다")
    meta = json.loads(P.tdd_v5_meta.read_text())
    sha = XB.sha256_file(P.tdd_v5)
    if sha != meta.get("sha256") or meta.get("v5_sha256") != (XB.sha256_file(P.v5_table) if P.v5_table.exists() else ""):
        raise SystemExit("[xb] v5 연도 정합 도일 표의 해시가 메타나 지금 v5 와 다르다. --stage tdd 를 다시 한다")
    t5 = pd.read_csv(P.tdd_v5).drop_duplicates("loc_id").set_index("loc_id")
    orig = getattr(XS.tdd_tables, "_xf_orig", XS.tdd_tables)

    def tdd_tables(*args, **kw):
        v, flags, src = orig(*args, **kw)
        both = t5.index.intersection(v.index)
        if len(both):
            raise SystemExit(f"[xb] v5 연도 정합 도일 표와 v1·v4 표의 loc_id 가 겹친다({len(both)}개)")
        v = pd.concat([v, t5.tdd_matched.astype(float)])
        if len(t5):
            flags = pd.concat([flags, t5.match_flag.astype(str)])
        return v, flags, dict(src, v5=f"{P.tdd_v5.name}:{sha[:16]}")
    tdd_tables._xf_orig = orig
    XS.tdd_tables = tdd_tables
    return XS, add


def _xb_workers(xb_argv):
    for i, v in enumerate(xb_argv):
        if v == "--workers" and i + 1 < len(xb_argv):
            return int(xb_argv[i + 1])
        if v.startswith("--workers="):
            return int(v.split("=", 1)[1])
    return None


def xb_main(xf_argv, xb_argv):
    """'x_new_regions.py [XF 인자] --xb -- <XB 인자>': XF 별칭과 v5 연도 정합 도일을 붙여 XB 의 main 을 이 프로세스에서 부른다.
    실제 적합(XB_NO_FIT 이 없는 실행)은 적격 동결 기록이 있어야 한다. 별칭 등록은 프로세스 안에서만 유효하므로 XB 집계만 하는 경우가 아니면
    --workers 0 을 요구한다(병렬은 XB 의 --shard i/N 으로 프로세스를 나눈다)."""
    a = parse_args(xf_argv)
    no_fit = any(f in xb_argv for f in XB_NO_FIT)
    if "--summarize-only" not in xb_argv and _xb_workers(xb_argv) != 0:
        raise SystemExit("[xb] XF 별칭은 이 프로세스에만 등록된다. XB 인자에 --workers 0 을 주고 병렬은 --shard i/N 으로 나눈다")
    XS, add = xb_prepare(a.P, require_frozen=not no_fit)
    print(f"[xb] XF 별칭 {len(add)}개 등록 · v5 연도 정합 도일 연결 · 적격 동결 확인 {not no_fit}", flush=True)
    return XS.main(list(xb_argv))


def main(argv=None):
    av = list(sys.argv[1:] if argv is None else argv)
    if "--xb" in av:
        i = av.index("--xb")
        rest = av[i + 1:]
        return xb_main(av[:i], rest[1:] if rest[:1] == ["--"] else rest)
    a = parse_args(argv)
    stages = [v.strip() for v in a.stage.split(",") if v.strip()]
    bad = [v for v in stages if v not in STAGES]
    if bad:
        raise SystemExit(f"알 수 없는 단계: {bad}")
    if stages or a.count_only or a.repro_gate or a.xc_f3:
        XB.guard("count")
    elif a.smoke:
        XB.guard("smoke")
    elif a.summarize_only:
        XB.guard("summarize")
    else:
        XB.guard("run", a.ARGV)
    a.P.check()
    a.THREAD_CAP = _cap_threads(a)
    rc = 0
    with XB.restricted_output():
        print(f"[xf] 계획 {XB.PLAN_DOC} {XB.PLAN_REVISION}(T0 {XB.PLAN_COMMIT}) · 산출 {a.P.root} · 스레드 {a.threads} · 허용 표지 {XB.run_permitted(a.ARGV)}",
              flush=True)
        if stages:
            _resources(a, heavy=bool({"cov", "tdd"} & set(stages)))
            build(a, stages)
        if a.count_only:
            _resources(a)
            count_only(a)
        if a.repro_gate:
            _resources(a)
            rc = 0 if repro_gate(a, gate_b1=a.gate_b1)["passed"] else 1
        if a.xc_f3:
            write_xc_f3(a.P)
        if a.smoke:
            _resources(a)
            smoke(a)
        if a.summarize_only:
            _resources(a, heavy=False)
            summarize(a)
        if not (stages or a.count_only or a.repro_gate or a.xc_f3 or a.smoke or a.summarize_only):
            _resources(a)
            done, failed = run_all(a)
            rc = 1 if failed else 0
            if not a.no_summarize and not failed:
                summarize(a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
