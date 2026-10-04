"""XG_product_comparison(기존 ALT 지도와 같은 시험지 비교). 계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.7 의 구현(개정 1, T0 = git 2678100).

등록 이탈(WRAPUP 10): WRAPUP 10 은 제품 비교를 판정 없는 서술 SI 표로 등록했다. XG 는 확인 대비 6개를 Holm(m = 6)으로 판정한다(사용자 지시 2,
2026-10-04, LGU-B2 결과를 본 뒤의 결정). 모든 판정 행에 '등록 이탈(WRAPUP 10)'을 붙인다.

단계와 명령(모두 ROOT 에서)
  1. 추출(로컬, T0 커밋 확인 뒤, 라벨 미사용): 제품 래스터를 라벨 셀(v3 직접 라벨 + v4 새 셀) 좌표에서 최근접 화소로 읽는다. 연도 정합 규칙:
     셀 라벨 연도와 제품 기간이 겹치면 겹친 연도 평균, 겹치지 않으면 기간 평균과 match_flag. 3 × 3 평균은 민감도 열로 둔다.
       nice -n 10 python3 scripts/3_deep_learning/x_product_comparison.py --extract --threads 4
     산출 xg_product_values_v1.csv(loc_id, product, value_cm, valid, match_flag, n_years, value_static_cm, value3_cm, pix_dist_km),
     xg_meta.json(T0 확인, 단위·좌표계·결측값 확인 기록, Zenodo md5 대조, 약관 문구, 입력 sha256).
  2. 누설 마스크(로컬, 좌표만): 제품 학습 지점에서 d km 안(≤ d)의 셀이 하나라도 있는 블록(d = 5, 25)과 셀 단위 5 km.
       python3 scripts/3_deep_learning/x_product_comparison.py --mask
     학습 지점: Wei = 보충 표 ALT_variables.xlsx 의 (LAT, LONG) 고유값(L1, 정확). Aalto = CALM ∪ GTN-P 활동층 좌표(L2 대리 마스크).
     CCI v4·v5, Yi·Kimball 은 L0(학습 지점 없음)이라 마스크가 비어 있다.
  3. 재채점(로컬 분석, 적합 없음): LG·LGD·WF6·LGX x9 조각의 블록 SSE 에 제품 키(해석식)를 더하고 블록 마스크로 다시 채점한다.
       OMP_NUM_THREADS=2 nice -n 10 python3 scripts/3_deep_learning/x_product_comparison.py --summarize-only --allow-local   # 재표집 10,000회
     허용 표지가 없으면 재표집이 1,000회로 제한되고, 이때는 본 봉인 폴더에 쓰지 않고 멈춘다(등록 10,000회). 허용 표지가 있어도 로컬에서는
     상주 메모리 감시(10 GB)와 가용 메모리 대기(30 GB 미만이면 최대 1시간)를 건다(계획 1절 로컬 자원, 환경 변수 WF_RESCALE·LG_RESCALE 로 판단).
     산출은 봉인 폴더(data/processed/xbatch/XG_product_comparison/sealed/). 화면에는 행 수와 해시만 쓴다.
  4. 셀 단위 5 km 민감도(Rescale 작업 R3): R1·D0 를 다시 적합하고(LG 와 같은 추출·행렬) 채점 셀을 셀 단위로 가린 저장소를 함께 쓴다.
       WF_RESCALE=1 python3 scripts/3_deep_learning/x_product_comparison.py --part cell --workers 10 --threads 4 --resume --no-summarize
     단위 하나: --part cell --shard Lena:x:1
     입력: 묶음에 xg_product_values_v1.csv, xg_leak_cells_v1.csv, xg_meta.json(PAYLOAD_EXTRA)이 있어야 한다. 없거나 Wei 값·채점 셀 거리가 없으면
     적합 전에 멈춘다. 제출 전 점검: python3 scripts/3_deep_learning/x_product_comparison.py --payload-check
  세기: python3 scripts/3_deep_learning/x_product_comparison.py --count-only     (조각 수, 블록 수, 가린 블록 수, 적합 수. 라벨 값 미사용)
  스모크: OMP_NUM_THREADS=2 nice -n 10 taskset -c 100-103 python3 scripts/3_deep_learning/x_product_comparison.py --smoke --threads 2 --workers 0
  관문: python3 scripts/3_deep_learning/x_product_comparison.py --gate-only

파일 위치 이탈: 계획 2.7 은 scripts/1_data_prep/xg_extract_products_v1.py, scripts/2_evaluation/xg_leak_mask_v1.py,
  scripts/2_evaluation/x_product_comparison.py, tests/test_x_products.py 를 적었다. 이 파일의 단계(--extract, --mask, 재채점, --part cell)와
  tests/test_x_xg_xh.py 로 두었다(docs/research/2026-10-04/impl_notes/x_product_comparison.md 1절). 묶음 정보(--payload-check)에도 적는다.

키(LGX 키 이름 규칙, alt_products.md 7.2)
  B:{p}_raw  제품 값 그대로(cm, n 0). 결측·0 이하 셀은 ρ·s(ρ = 원천 셀의 b/s 중앙값, 라벨 미사용)로 바꾼다(h42 anchor_fill, B:cci_raw 와 같다).
             바꾼 채점 셀의 비율을 모든 대비 행(fill_frac_B)과 판정 표에 싣고, 0 보다 크면 '제품 결측 대체 k%'를 붙인다. 바꾼 셀이 있는 블록을
             두 팔에서 같이 뺀 서술 민감도(xg_fill_sensitivity)를 함께 낸다(블록 SSE 조각에는 셀 단위 예측이 없어 블록 단위로 뺀다)
  B:{p}_aff  원천 라벨 아핀 a + b·제품(h42 affine_ls, B:cci_aff 와 같다)
  P0@{p}     원천 척도 c0·b(c0 = h40.ls_E(y_src, b_src))
  P1@{p}     대상 라벨 n 개의 κ 10 수축 척도 c_n·b(h42.shrink, P1@cci 와 같은 식. 추출은 h40.draw_cells 로 LG 와 같다)
  P0, P1, P*(= P0@tddm, LGX x9), R1(λ 0.25), D0 는 기존 조각 키를 그대로 쓴다(다시 적합하지 않는다)
  {p} = cci4(v4 다년 평균, 기존 cci_alt 열), cci5y(v5 연도 정합), wei(Wei v2 연도 정합), aalto(Aalto 2018 기준기), yk(Yi·Kimball 연도 정합,
  알래스카), cci5s·weis(기간 평균, 민감도)

확인 대비(Holm 가족 m = 6, Holm 보정 p 가 판정 문구를 정한다. WRAPUP 1.1 (a)–(e) 규칙)
  XG-1w B:wei_raw − P0(n 0) · XG-2w R1(0.25) − P1@wei(n 10) · XG-3w 같음(전량), 풀 레나·캐나다
  XG-1c B:cci5y_raw − P0(n 0) · XG-2c R1(0.25) − P1@cci5y(n 10) · XG-3c 같음(전량), 풀 주 4지역
  공동 주 마스크: 블록 단위 5 km 와 25 km. 두 마스크의 판정이 같을 때만 방향 문장을 쓰고, 다르면 '마스크 의존'을 붙여 약한 쪽으로 쓴다.
  CI 는 같은 라벨 집합 대비(h40.contrast = h4_common.boot_delta_blocks, 보조 h42.boot_delta_common)이고 사용 분할 채점 블록 합집합 8 이상에서만
  낸다. 분할별 최소 채점 블록 < 5 → '소수 블록', 마스크 뒤 분할별 최소 채점 블록 < 8 → '마스크 뒤 소수 블록'.
  판정 표(xg_tests)에는 마스크마다 MEAN 행의 보조 판정(δ 1.0, δ_rel), '한계 의존', '분할 독립 가정 의존', 지역 일반 조건(HK), 풀 표지를 옮기고,
  문장의 표지·'부분(지역 k/m)'·열세 지역은 두 공동 주 마스크의 합집합으로 쓴다.
서술 대비: XG-4r R1 − B:{p}_raw(두 마스크 결합 갈래와 같은 문장 규칙, Holm 없음), XG-4 P1@{p} − P1, XG-5 WF6 지역 내 R1·P1·D0 − B:{p}_raw,
  XG-6 알래스카(x)·Russia_C·Tibet 행, XG-7 러시아 W·E 의 CALM 학습 제품 참고값(별표, 판정 없음). CALM 학습 제품(Wei, Aalto)의 XG-4 풀에서는
  러시아 W·E 를 빼고 그 지역 행은 별표 참고값으로만 싣는다. Aalto(L2, 학습 지점 좌표 없음, 대리 마스크) 행에는 '누설 점검 불가(대리 마스크)'와
  SI 전용 표지를 단다.

재현 관문(계획 2.7)
  (g1) LGX cells.npz 에서 다시 만든 블록 SSE 가 lgxb BlockStore 와 같다(cells.npz 의 예측·y 가 float32 라 float32 반올림 상한 안이면 같음으로 본다)
  (g1b) 해석식 제품 키 경로로 계산한 cci4 키(B:cci4_raw, B:cci4_aff, P0@cci4, P1@cci4)가 LGX x9 의 B:cci_raw, B:cci_aff, P0@cci, P1@cci 와 같다
  (g2) 마스크 전 P0·R1(0.25) RMSE 가 LG 곡선 표(results/rescale_lg/.../lg_curve.csv)와 같다(상대 1e-9, 1절 로컬–Rescale 상대 허용 오차)
  (g3) 내려받은 Zenodo 파일의 md5 가 alt_products.md 9절 값과 같다(추출 단계가 대조해 xg_meta.json 에 쓴다)
  관문 표 xg_gate.csv 에는 키 수, 최대 차, 통과 여부만 쓴다. 집계는 관문 요약(xg_gate_meta.json)을 봉인 메타 xg_summary_meta.json 에 옮긴다.

출력 제한(계획 0.3): 스모크·세기·집계의 화면에는 수와 해시만 쓴다. 판정·대비 표는 봉인 폴더에 쓴다.
"""
from __future__ import annotations

import xbatch_core as XB                                                   # numpy 보다 먼저(스레드 환경 변수)

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

H, X, W = XB.H, XB.X, XB.W
import polar.h4_common as H4                                               # noqa: E402
from polar.h4_common import BlockStore                                     # noqa: E402
from polar.m1_core import eval_mask, half_split_blocks                     # noqa: E402

ROOT = XB.ROOT
EXP_ID = "XG"
NAME = XB.EXP_NAMES[EXP_ID]
OUT = XB.out_dir(EXP_ID)
TAG_CELL = "xgc"
DEVIATION = "등록 이탈(WRAPUP 10)"
DESIGN = "결과 열람 뒤 설계"
KAPPA = XB.KAPPA
LG_GRID = (0, 3, 10, 40, 160, 320, 1000, -1)                               # LG 의 n 격자(h40 기본)
LG_DRAWS = 5
MASK_D = (5.0, 25.0)
MASKS = ("none", "blk5", "blk25")
CO_PRIMARY = ("blk5", "blk25")
CELL_D = 5.0
VALUES_FILE = "xg_product_values_v1.csv"
MASK_FILE = "xg_leak_mask_v1.csv"
CELLDIST_FILE = "xg_leak_cells_v1.csv"
META_FILE = "xg_meta.json"
GATE_META = "xg_gate_meta.json"
RAW = ROOT / "data" / "raw"
GDAL_CACHEMAX_MB = 512                                                     # GDAL 블록 캐시 상한(MB). 기본값(RAM 의 5 %)은 이 서버에서 약 12.5 GB 다
GDAL_TMP_NAME = "tmp_gdal"                                                 # CPL_TMPDIR = data/raw/tmp_gdal(/home, 계획 1절 디스크)
MEM_MIN_GB = 30.0                                                          # 시작 전 가용 메모리 하한(계획 1절 로컬 자원)
MEM_WAIT_S = 3600.0                                                        # 하한 아래면 로컬에서 기다리는 최대 시간(초)
RSS_LIMIT_GB = 10.0                                                        # 로컬 작업 하나의 상주 메모리 상한
FILL_TXT = "제품 결측 대체"
FILL_SENS_TXT = "민감도(서술): 제품 결측 대체 셀이 있는 블록을 두 팔에서 같이 뺐다"
LEAK_CHECK = {"L0": "", "L1": "", "L2": "누설 점검 불가(대리 마스크)"}        # 계획 2.7 공통 행: 학습 지점 좌표가 없는 제품
SI_ONLY_TXT = "SI 전용"
REF_STAR = "학습 지점 포함*"
REF_ROLE = "참고값(판정 없음)"
VERDICT_COLS = ("verdict4", "verdict4_d10", "verdict4_rel", "verdict4_common", "verdict4_draw_cond", "limit_dependence", "ci_dependence",
                "draw_dependence", "small_note", "worse", "noninf", "noninf_d10", "noninf_common", "superior_common")
_REL_OUT = OUT.relative_to(ROOT)
PAYLOAD_EXTRA = (f"{_REL_OUT}/{VALUES_FILE}", f"{_REL_OUT}/{CELLDIST_FILE}", f"{_REL_OUT}/{META_FILE}")   # 작업 R3 묶음의 필수 입력
LOCATION_NOTE = ("파일 위치 이탈: 계획 2.7 의 scripts/1_data_prep/xg_extract_products_v1.py, scripts/2_evaluation/xg_leak_mask_v1.py, "
                 "scripts/2_evaluation/x_product_comparison.py, tests/test_x_products.py 대신 scripts/3_deep_learning/x_product_comparison.py 의 "
                 "단계와 tests/test_x_xg_xh.py(docs/research/2026-10-04/impl_notes/x_product_comparison.md 1절)")

# 조각 원천: 저장소 이름 → (원천, 조각 폴더, tag, 파일 안의 대상 이름, 모드)
SH_LG = ROOT / "results" / "rescale_lg" / "data" / "processed" / "lg" / "shards"
SH_LGX = ROOT / "data" / "processed" / "lgx" / "shards"
SH_WF6 = ROOT / "results" / "rescale_wf2" / "data" / "processed" / "wf" / "shards"
LG_CURVE = ROOT / "results" / "rescale_lg" / "data" / "processed" / "lg" / "lg_curve.csv"
SOURCES = {
    "Lena|x": ("lg", SH_LG, "lg", "Lena", "x"), "Canada|x": ("lg", SH_LG, "lg", "Canada", "x"),
    "Russia_W|x": ("lg", SH_LG, "lg", "Russia_W", "x"), "Russia_E|x": ("lg", SH_LG, "lg", "Russia_E", "x"),
    "Alaska|x": ("lg", SH_LG, "lg", "Alaska", "x"),
    "Russia_C~lgd|x": ("lgd", ROOT / "data" / "processed" / "lgd" / "Russia_C" / "shards", "lgd", "Russia_C", "x"),
    "Tibet_LGD|x": ("lgd", ROOT / "data" / "processed" / "lgd" / "Tibet" / "shards", "lgd", "Tibet_LGD", "x"),
    "Alaska|r": ("wf6", SH_WF6, "wf6", "Alaska", "r"), "Lena|r": ("wf6", SH_WF6, "wf6", "Lena", "r"),
    "Canada|r": ("wf6", SH_WF6, "wf6", "Canada", "r"),
}
POOL_W = ["Lena|x", "Canada|x"]
POOL_C = list(XB.MAIN4)
XG6 = ["Alaska|x", "Russia_C~lgd|x", "Tibet_LGD|x"]
XG7 = ["Russia_W|x", "Russia_E|x"]
INREG = list(XB.INREGION3)

# 제품 명세. leak = 누설 등급(alt_products.md 2.2), train = 학습 지점 집합, alt_def = ALT 정의(공통 문장)
PRODUCTS = {
    "cci4": dict(kind="column", column="cci_alt", period=(1997, 2021), rule="static", leak="L0", train=None, label="CCI v4",
                 alt_def="CCI 는 모형 화소의 연 최대 융해 깊이(v4 다년 평균, 기존 cci_alt)"),
    "cci5y": dict(kind="cci_nc", folder="cci_alt_v5", tag="fv05.0", period=(1997, 2023), rule="matched", leak="L0", train=None, label="CCI v5",
                  alt_def="CCI 는 모형 화소의 연 최대 융해 깊이"),
    "cci5s": dict(base="cci5y", rule="static", leak="L0", train=None, label="CCI v5(기간 평균)", alt_def="CCI 는 모형 화소의 연 최대 융해 깊이"),
    "wei": dict(kind="zip_tif", folder="wei2026_alt_v2", period=(2000, 2024), rule="matched", leak="L1", train="wei", label="Wei v2",
                zips={"ALT_2000_2009.zip": "74d778dd80abfaa552013c60416e8358", "ALT_2010_2019.zip": "c7e673a22225c58c693aa49f7a46da39",
                      "ALT_2020_2024.zip": "a2ac7a831765519de819826d50066fd2"},
                member="ALT_{y0}_{y1}/ALT_{year}.tif", alt_def="Wei 는 CALM 라벨로 학습한 기계학습 지도", overlap=True),
    "weis": dict(base="wei", rule="static", leak="L1", train="wei", label="Wei v2(기간 평균)", alt_def="Wei 는 CALM 라벨로 학습한 기계학습 지도",
                 overlap=True),
    "aalto": dict(kind="zip_tif", folder="aalto2018_alt", period=(2000, 2014), rule="static", leak="L2", train="calm_gtnp", label="Aalto 2018",
                  zips={"ALT_Rasters.zip": "47752e4d140dda92d9ddbf806adc4508"}, member="ALT_Baseline.tif",
                  alt_def="Aalto 는 GTN-P·CALM 지점으로 학습한 통계 모형 지도(기준기 2000–2014)", overlap=True),
    "yk": dict(kind="nc_latlon", file="ornl_ds1760/Alaska_active_layer_thickness_1km_2001-2015.nc4", var="ALT", period=(2001, 2015),
               rule="matched", leak="L0", train=None, label="Yi·Kimball", alt_def="Yi·Kimball 은 위성 기반 토양 과정 모형(알래스카)", regions=("Alaska",)),
}
PRODUCT_KEYS = ("cci4", "cci5y", "cci5s", "wei", "weis", "aalto", "yk")
MISSING_CODES = {"ds2332": (1.0, 2.0)}                                     # ds2332 의 1.0·2.0 화소는 결측(계획 2.7, 서술 비교에만 해당)

# 확인 대비 6개(계획 2.7 표 순서). gA − gB.
CONFIRM = [
    dict(id="XG-1w", group="XG-1", product="wei", gA=("B:wei_raw", 0), gB=("P0", 0), pool=POOL_W, blind="맹검"),
    dict(id="XG-2w", group="XG-2", product="wei", gA=("R1", 10), gB=("P1@wei", 10), pool=POOL_W, blind="맹검"),
    dict(id="XG-3w", group="XG-3", product="wei", gA=("R1", -1), gB=("P1@wei", -1), pool=POOL_W, blind="맹검"),
    dict(id="XG-1c", group="XG-1", product="cci5y", gA=("B:cci5y_raw", 0), gB=("P0", 0), pool=POOL_C,
         blind="비맹검 부분 포함(LGX L29 의 CCI v4 원값 − P0, 블록 등가중 +4.99 – +13.93 cm, 러시아 W +0.41 미결정, data/processed/lgx/lgx_curve.csv)"),
    dict(id="XG-2c", group="XG-2", product="cci5y", gA=("R1", 10), gB=("P1@cci5y", 10), pool=POOL_C, blind="비맹검 부분 포함(L29 의 CCI v4 척도 재보정)"),
    dict(id="XG-3c", group="XG-3", product="cci5y", gA=("R1", -1), gB=("P1@cci5y", -1), pool=POOL_C, blind="비맹검 부분 포함(L29 의 CCI v4 척도 재보정)"),
]
HOLM_M = 6

# 사전 고정 해석 문장(계획 2.7). {X} = 제품, {a} = |Δ|(cm, 셀 가중), {k} = 지역 수, {nl} = 라벨 수
TEMPLATES = {
    "XG-1": {"우세": "라벨이 없을 때 기존 지도 {X} 는 원천 계수 Stefan 보다 오차가 {a} cm 작았다(지역 {k}, 학습 지점 5·25 km 블록 마스크 뒤)",
             "열세": "라벨이 없을 때 기존 지도 {X} 는 원천 계수 Stefan 보다 오차가 {a} cm 컸다(지역 {k}, 학습 지점 5·25 km 블록 마스크 뒤)",
             "동등": "라벨이 없을 때 기존 지도 {X} 와 원천 계수 Stefan 의 오차는 0.5 cm 안에서 같았다(지역 {k}, 학습 지점 5·25 km 블록 마스크 뒤)",
             "미결정": "라벨이 없을 때 기존 지도 {X} 와 원천 계수 Stefan 의 오차 차이를 확인하지 못했다(지역 {k}, 학습 지점 5·25 km 블록 마스크 뒤)",
             "판정 불가": "라벨이 없을 때 기존 지도 {X} 와 원천 계수 Stefan 의 비교는 판정할 수 없었다"},
    "XG-2": {"우세": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차는 그 라벨로 재보정한 지도 {X} 보다 오차가 {a} cm 작았다",
             "열세": "같은 라벨로 재보정한 지도 {X} 는 재보정 앵커 잔차보다 오차가 {a} cm 작았다",
             "동등": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차와 그 라벨로 재보정한 지도 {X} 의 오차는 0.5 cm 안에서 같았다",
             "미결정": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차와 그 라벨로 재보정한 지도 {X} 의 오차 차이를 확인하지 못했다",
             "판정 불가": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차와 그 라벨로 재보정한 지도 {X} 의 비교는 판정할 수 없었다"},
}
TEMPLATES["XG-3"] = TEMPLATES["XG-2"]
TEMPLATES["XG-4r"] = {"우세": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차는 기존 지도 {X} 원값보다 오차가 {a} cm 작았다",
                      "열세": "기존 지도 {X} 원값은 같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차보다 오차가 {a} cm 작았다",
                      "동등": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차와 기존 지도 {X} 원값의 오차는 0.5 cm 안에서 같았다",
                      "미결정": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차와 기존 지도 {X} 원값의 오차 차이를 확인하지 못했다",
                      "판정 불가": "같은 대상 라벨 {nl} 쓴 재보정 앵커 잔차와 기존 지도 {X} 원값의 비교는 판정할 수 없었다"}
CALM_CLAUSE = "지도 {X} 는 대상 지역 안 CALM 지점도 학습에 썼고 원천 계수 Stefan 은 대상 지역과 100 km 버퍼 밖 라벨만 썼다"
OVERLAP_TXT = "학습 자료 중복 가능"
STRENGTH = {"우세": 3, "열세": 3, "동등": 2, "미결정": 1, "판정 불가": 0}


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XG 제품 비교(계획 2.7)")
    ap.add_argument("--extract", action="store_true", help="제품 값 추출(로컬, T0 커밋 확인 뒤)")
    ap.add_argument("--mask", action="store_true", help="누설 마스크 표(좌표만)")
    ap.add_argument("--part", default="rescore", choices=["rescore", "cell"], help="rescore = 기존 조각 재채점(적합 없음), cell = 셀 단위 민감도 적합")
    ap.add_argument("--products", default=",".join(PRODUCT_KEYS))
    ap.add_argument("--units", default="", help="단위 지정(예: wei=cm,aalto=cm). 없으면 메타데이터 또는 값 범위로 정한다")
    ap.add_argument("--targets", default="", help="재채점 저장소 이름 목록(기본 전부, 예: Lena|x,Canada|x)")
    ap.add_argument("--cell-targets", default="Lena,Canada", help="셀 단위 민감도의 대상(모드 x)")
    ap.add_argument("--cell-grid", default="0,10,-1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=len(XB.SEEDS))
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--out-dir", default=str(OUT.relative_to(ROOT)))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--lgd-dir", default="data/processed/lgd/run_tables")
    ap.add_argument("--shard", default="", help="셀 단위 단위 하나: <대상>:x:<분할>")
    ap.add_argument("--payload-check", action="store_true", help="작업 R3 묶음의 필수 입력(PAYLOAD_EXTRA) 점검과 묶음 정보 기록(제출 전)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="레나, 분할 1, 재표집 200. 셀 단위는 n 0·10, 추출 1, seed 1. 출력은 smoke/ 아래")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    a.PERMIT = XB.run_permitted(a.argv)
    a.threads, a.nboot = XB.local_limits(a.threads, a.nboot, a.argv)
    a.PRODUCTS = [v.strip() for v in a.products.split(",") if v.strip()]
    bad = [p for p in a.PRODUCTS if p not in PRODUCTS]
    if bad:
        raise SystemExit(f"알 수 없는 제품: {bad}")
    a.UNITS = dict(kv.split("=", 1) for kv in a.units.split(",") if "=" in kv)
    a.CELL_TARGETS = [v.strip() for v in a.cell_targets.split(",") if v.strip()]
    a.CELL_GRID = [int(v) for v in a.cell_grid.split(",") if v.strip()]
    a.SEEDS = list(XB.SEEDS[:max(1, int(a.seeds))])
    a.DRAWS = LG_DRAWS
    a.SPLITS = list(XB.TRANSFER_SPLITS)
    a.TARGETS = [v.strip() for v in a.targets.split(",") if v.strip()] or list(SOURCES)
    if any(t not in SOURCES for t in a.TARGETS):
        raise SystemExit(f"알 수 없는 대상: {[t for t in a.TARGETS if t not in SOURCES]}")
    if a.smoke:
        a.TARGETS = ["Lena|x"]; a.SPLITS = [1]; a.nboot = min(a.nboot, 200)
        a.CELL_TARGETS = ["Lena"]; a.CELL_GRID = [0, 10]; a.DRAWS = 1; a.SEEDS = [XB.SEEDS[0]]
    a.OUT = (ROOT / a.out_dir) if not os.path.isabs(a.out_dir) else Path(a.out_dir)
    a.BASE_OUT = a.OUT
    if a.smoke:
        a.OUT = a.OUT / "smoke"
    a.SHARDS = a.OUT / "shards"
    a.TAG = TAG_CELL + ("_smoke" if a.smoke else "")
    a.A54 = None
    a.INSHA = None
    return a


def a54(a):
    """h54 인자 계약을 만족하는 인자(xbatch_core.h54_args). 분할 1–5."""
    if a.A54 is None:
        a.A54 = XB.h54_args(exp="wf4", splits=XB.TRANSFER_SPLITS, threads=a.threads, cb_iters=a.cb_iters, seeds=len(a.SEEDS), nboot=a.nboot,
                            data_dir=a.data_dir, lgd_dir=a.lgd_dir, out_dir=a.OUT / "_h54_unused", tag=a.TAG, allow_local=a.PERMIT)
    return a.A54


# ================================================================ 공통 도우미
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


def gdal_env(raw=None) -> dict:
    """GDAL 블록 캐시와 임시 폴더(계획 1절 디스크 행: CPL_TMPDIR·GDAL_CACHEMAX 는 /home 의 data/raw 아래). rasterio 를 처음 부르기 전에
    부른다(GDAL 은 블록 캐시 상한을 처음 쓸 때 읽는다). 반환 기록 dict(xg_meta.json)."""
    tmp = Path(raw or RAW) / GDAL_TMP_NAME
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["GDAL_CACHEMAX"] = str(int(GDAL_CACHEMAX_MB))
    os.environ["CPL_TMPDIR"] = str(tmp)
    return dict(GDAL_CACHEMAX=os.environ["GDAL_CACHEMAX"], CPL_TMPDIR=os.environ["CPL_TMPDIR"], unit="MB")


def t0_committed() -> bool:
    """계획 문서 커밋(T0 = 2678100)이 HEAD 의 조상인지(계획 0.3: 제품 값 추출 전에 커밋되어 있어야 한다)."""
    try:
        return subprocess.run(["git", "merge-base", "--is-ancestor", XB.PLAN_COMMIT, "HEAD"], cwd=ROOT, capture_output=True).returncode == 0
    except OSError:
        return False


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_alias(name):
    """저장소 이름 → (자료 별칭, 표 안의 대상 이름, 모드)."""
    al, mode = name.split("|")
    return al, XB.resolve(al)[1], mode


def year_rule(V, years, ymin, ymax, rule):
    """연도 정합. V = (셀, 연도) 값, years = 연도 목록. rule = matched 이면 [ymin, ymax] 와 겹친 연도의 평균(유한 값), 겹침이 없거나 겹친 연도가 모두
    결측이면 기간 평균과 match_flag 'period_mean'. static 이면 기간 평균('static'). 반환 (값, match_flag, 쓴 연도 수)."""
    V = np.asarray(V, float); years = np.asarray(years, int)
    n = V.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        allm = np.nanmean(V, 1) if V.shape[1] else np.full(n, np.nan)
    nall = np.isfinite(V).sum(1)
    if rule == "static":
        return allm, np.where(np.isfinite(allm), "static", "missing").astype(object), nall
    ymin = np.asarray(ymin, float); ymax = np.asarray(ymax, float)
    M = (years[None, :] >= ymin[:, None]) & (years[None, :] <= ymax[:, None]) & np.isfinite(V)
    cnt = M.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mm = np.where(M, V, 0.0).sum(1) / np.where(cnt > 0, cnt, 1)
    val = np.where(cnt > 0, mm, allm)
    flag = np.where(cnt > 0, "overlap", np.where(np.isfinite(allm), "period_mean", "missing")).astype(object)
    return val, flag, np.where(cnt > 0, cnt, nall)


def infer_units(raw, declared=None, override=None):
    """단위 확인 기록: override(명령행) > 파일 메타데이터 > 값 범위(유한 값 중앙값 < 10 이면 m, 아니면 cm). 반환 (cm 배수, 기록 dict)."""
    fin = np.asarray(raw, float); fin = fin[np.isfinite(fin)]
    med = float(np.median(fin)) if len(fin) else np.nan
    if override:
        u, how = override, "명령행"
    elif declared:
        u, how = declared, "메타데이터"
    else:
        u, how = ("m" if (np.isfinite(med) and med < 10.0) else "cm"), "값 범위(중앙값 < 10 이면 m)"
    u = {"metres": "m", "meter": "m", "meters": "m"}.get(str(u).lower(), str(u).lower())
    if u not in ("m", "cm"):
        raise SystemExit(f"[단위] 알 수 없는 단위 {u}")
    return (100.0 if u == "m" else 1.0), dict(units=u, how=how, raw_median=med, n_finite=int(len(fin)))


def nearest_index(axis, q):
    """정렬된 1차원 좌표축(오름·내림)에서 q 의 최근접 색인."""
    axis = np.asarray(axis, float); q = np.asarray(q, float)
    desc = axis[0] > axis[-1]
    ax = axis[::-1] if desc else axis
    j = np.clip(np.searchsorted(ax, q), 1, len(ax) - 1)
    j = np.where(np.abs(ax[j - 1] - q) <= np.abs(ax[j] - q), j - 1, j)
    return (len(ax) - 1 - j) if desc else j


def read_points(read_window, rows, cols, shape, tile=512, nb=1):
    """화소 (rows, cols) 의 값과 (2nb+1)² 창 평균. read_window(r0, r1, c0, c1) → 2차원 float(결측 NaN). 타일 단위로 창을 읽는다(전체를 올리지 않는다)."""
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    v = np.full(len(rows), np.nan); v3 = np.full(len(rows), np.nan)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    key = (rows // tile) * 100000 + (cols // tile)
    for t in np.unique(key[ok]):
        m = ok & (key == t)
        r0, r1 = max(int(rows[m].min()) - nb, 0), min(int(rows[m].max()) + nb + 1, shape[0])
        c0, c1 = max(int(cols[m].min()) - nb, 0), min(int(cols[m].max()) + nb + 1, shape[1])
        A = np.asarray(read_window(r0, r1, c0, c1), float)
        rr, cc = rows[m] - r0, cols[m] - c0
        v[m] = A[rr, cc]
        v3[m] = window_mean(A, nb)[rr, cc]
    return v, v3


def window_mean(A, nb=1):
    """(2nb+1)² 창의 유한 값 평균(결측은 빼고 센다, 가장자리는 창 안의 화소만)."""
    from scipy.ndimage import uniform_filter
    A = np.asarray(A, float)
    fin = np.isfinite(A)
    k = 2 * nb + 1
    S = uniform_filter(np.where(fin, A, 0.0), size=k, mode="constant") * k * k
    N = uniform_filter(fin.astype(float), size=k, mode="constant") * k * k
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(N > 0.5, S / np.where(N > 0.5, N, 1.0), np.nan)


# ================================================================ 1. 추출(로컬, T0 뒤, 라벨 미사용)
def label_cells(a):
    """라벨 셀 좌표와 라벨 연도(라벨 값은 읽지 않는다). v3 직접 라벨(fidelity_base_v3, F4_direct) + 네 실행 표의 새 셀.
    연도: v3 = data/processed/m1/a2_year_matched_tdd_cells.csv, 새 셀 = fidelity_base_v4_labels.csv(year_min, year_max)."""
    proc = ROOT / a.data_dir
    cols = ["loc_id", "lat", "lon", "source_id"]
    parts = [pd.read_csv(proc / "fidelity_base_v3.csv", usecols=cols, low_memory=False)]
    for s in XB.RUN_TABLE_DIRS:
        parts.append(pd.read_csv(ROOT / a.lgd_dir / s / "fidelity_base_v3.csv", usecols=cols, low_memory=False))
    c = pd.concat(parts, ignore_index=True)
    c = c[c.source_id == "F4_direct"].drop_duplicates("loc_id").reset_index(drop=True)
    y3 = pd.read_csv(proc / "m1" / "a2_year_matched_tdd_cells.csv", usecols=["loc_id", "year_min", "year_max"]).drop_duplicates("loc_id")
    y4 = pd.read_csv(proc / "fidelity_base_v4_labels.csv", usecols=["loc_id", "part", "year_min", "year_max"])
    y4 = y4[y4.part == "new"][["loc_id", "year_min", "year_max"]]
    yr = pd.concat([y3, y4], ignore_index=True).drop_duplicates("loc_id")
    c = c.merge(yr, on="loc_id", how="left")
    return c.drop(columns=["source_id"])


def extract_cci(spec, cells, override=None):
    """CCI v5(0.01° 위경도 격자, uint16 × 0.01 m). 연도마다 최근접 화소와 3 × 3 평균."""
    import netCDF4
    folder = RAW / spec["folder"]
    files = {}
    for p in sorted(folder.glob(f"ESACCI-PERMAFROST-L4-ALT-*-{spec['tag']}.nc")):
        y = int(p.name.split("_PP-")[1][:4])
        files[y] = p
    years = sorted(y for y in files if spec["period"][0] <= y <= spec["period"][1])
    if not years:
        raise SystemExit(f"[추출] {folder} 에 CCI 파일이 없다")
    with netCDF4.Dataset(files[years[0]]) as d:
        lat = np.asarray(np.ma.filled(np.ma.asarray(d["lat"][:]).astype(float), np.nan))
        lon = np.asarray(np.ma.filled(np.ma.asarray(d["lon"][:]).astype(float), np.nan))
        units = getattr(d["ALT"], "units", None); fill = getattr(d["ALT"], "_FillValue", None); crs = "위경도(spatial_ref)"
        res = float(abs(lat[1] - lat[0]))
    rows = nearest_index(lat, cells.lat.values); cc = nearest_index(lon, cells.lon.values)
    pix_lat, pix_lon = lat[rows], lon[cc]
    dist = H4_haversine(cells.lat.values, cells.lon.values, pix_lat, pix_lon)
    inside = dist <= 1.5 * res * 111.2
    V = np.full((len(cells), len(years)), np.nan); V3 = V.copy()
    for j, y in enumerate(years):
        with netCDF4.Dataset(files[y]) as d:
            var = d["ALT"]
            var.set_auto_maskandscale(True)

            def rw(r0, r1, c0, c1, var=var):
                x = var[0, r0:r1, c0:c1]
                return np.ma.filled(x.astype(float), np.nan)
            v, v3 = read_points(rw, rows, cc, (len(lat), len(lon)), tile=1000)
        V[:, j], V3[:, j] = v, v3
    V[~inside] = np.nan; V3[~inside] = np.nan
    mult, urec = infer_units(V, declared=units, override=override)
    rec = dict(files=[files[y].name for y in years], crs=crs, res_deg=res, fill_value=None if fill is None else int(fill), units=urec,
               nodata_rule="_FillValue 65535(netCDF 자동 가림)", scale="scale_factor 0.01 자동 적용")
    return V * mult, V3 * mult, years, dist, rec


def H4_haversine(lat1, lon1, lat2, lon2):
    from polar.m1_ext import haversine_km
    return np.asarray(haversine_km(np.asarray(lat1, float), np.asarray(lon1, float), np.asarray(lat2, float), np.asarray(lon2, float)), float)


def _tif_points(path, cells):
    """GeoTIFF(압축 파일 안은 /vsizip/)의 최근접 화소 값과 3 × 3 평균. 좌표계 변환은 pyproj. 반환 (값, 3×3, 거리 km, 기록)."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(path) as r:
        crs = r.crs.to_string() if r.crs else ""
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tf.transform(cells.lon.values, cells.lat.values)
        inv = ~r.transform
        cf, rf = inv * (np.asarray(xs), np.asarray(ys))
        rows, cols = np.floor(rf).astype(int), np.floor(cf).astype(int)
        nod = r.nodata

        def rw(r0, r1, c0, c1):
            from rasterio.windows import Window
            x = r.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
            bad = ~np.isfinite(x)
            if nod is not None and np.isfinite(nod):
                bad |= (x == nod)
            bad |= (x < -1e30)
            return np.where(bad, np.nan, x)
        v, v3 = read_points(rw, rows, cols, (r.height, r.width))
        cx, cy = r.transform * (cols + 0.5, rows + 0.5)
        if r.crs and r.crs.is_geographic:
            dist = H4_haversine(cells.lat.values, cells.lon.values, np.asarray(cy, float), np.asarray(cx, float))
            pix_km = abs(r.res[0]) * 111.2
        else:
            dist = np.hypot(np.asarray(cx) - np.asarray(xs), np.asarray(cy) - np.asarray(ys)) / 1000.0
            pix_km = abs(r.res[0]) / 1000.0
        rec = dict(crs=crs, res=list(r.res), nodata=None if nod is None else float(nod), shape=[r.height, r.width], dtype=r.dtypes[0])
    inside = dist <= 1.5 * pix_km
    v[~inside] = np.nan; v3[~inside] = np.nan
    return v, v3, dist, rec


def extract_tif(spec, cells, override=None, check_md5=True):
    """Wei v2(연별 GeoTIFF, zip 3개)와 Aalto 2018(기준기 GeoTIFF, zip). zip 의 md5 를 alt_products.md 9절 값과 대조한다."""
    folder = RAW / spec["folder"]
    md5 = {}
    for z, want in spec["zips"].items():
        p = folder / z
        if not p.exists():
            raise SystemExit(f"[추출] {p} 가 없다")
        got = md5_file(p) if check_md5 else want
        md5[z] = dict(md5=got, expected=want, ok=bool(got == want))
        if got != want:
            raise SystemExit(f"[관문 g3] {z} 의 md5 {got} 가 alt_products.md 9절 값 {want} 와 다르다")
    if spec["rule"] == "static":
        z = next(iter(spec["zips"]))
        v, v3, dist, rec = _tif_points(f"/vsizip/{folder / z}/{spec['member']}", cells)
        years = [None]
        V, V3 = v[:, None], v3[:, None]
    else:
        years = list(range(spec["period"][0], spec["period"][1] + 1))
        V = np.full((len(cells), len(years)), np.nan); V3 = V.copy(); rec, dist = {}, None
        for j, y in enumerate(years):
            y0 = (y // 10) * 10
            y1 = min(y0 + 9, spec["period"][1])
            z = f"ALT_{y0}_{y1}.zip"
            member = spec["member"].format(y0=y0, y1=y1, year=y)
            v, v3, dist, rec = _tif_points(f"/vsizip/{folder / z}/{member}", cells)
            V[:, j], V3[:, j] = v, v3
    mult, urec = infer_units(V, override=override)
    rec.update(units=urec, md5=md5)
    return V * mult, V3 * mult, years, dist, rec


def extract_nc_latlon(spec, cells, override=None):
    """Yi·Kimball(ds1760, 극 투영 1 km, 위경도 2차원 배열). 화소 중심의 단위 구 좌표 KD 트리로 최근접 화소를 찾는다."""
    import netCDF4
    from scipy.spatial import cKDTree
    p = RAW / spec["file"]
    with netCDF4.Dataset(p) as d:
        lat2 = np.asarray(d["lat"][:], float); lon2 = np.asarray(d["lon"][:], float)
        var = d[spec["var"]]
        units = getattr(var, "units", None); fill = getattr(var, "_FillValue", None)
        A = np.ma.filled(var[:].astype(float), np.nan)
        tv = np.asarray(d["time"][:], float); tu = d["time"].units
    import datetime as _dt
    base = _dt.datetime.strptime(tu.split("since ")[1][:10], "%Y-%m-%d")
    years = [(base + _dt.timedelta(days=float(t))).year for t in tv]

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    ok = np.isfinite(lat2) & np.isfinite(lon2)
    pts = np.c_[np.where(ok)]
    tree = cKDTree(xyz(lat2[ok], lon2[ok]))
    ch, j = tree.query(xyz(cells.lat.values, cells.lon.values), k=1)
    dist = 2.0 * 6371.0 * np.arcsin(np.clip(ch / 2.0, 0, 1))
    rr, cc = pts[j, 0], pts[j, 1]
    V = A[:, rr, cc].T
    V3 = np.full_like(V, np.nan)
    for t in range(A.shape[0]):
        V3[:, t] = window_mean(A[t], 1)[rr, cc]
    inside = dist <= 1.5
    V[~inside] = np.nan; V3[~inside] = np.nan
    mult, urec = infer_units(V, declared=units, override=override)
    sha_ok = None
    shp = p.with_name(p.name + ".sha256")
    if shp.exists():
        sha_ok = bool(XB.sha256_file(p) == shp.read_text().split()[0].strip())
    rec = dict(crs="극 투영(위경도 2차원 배열로 최근접)", fill_value=None if fill is None else float(fill), units=urec, sha256_matches_file=sha_ok,
               shape=list(A.shape))
    return V * mult, V3 * mult, years, dist, rec


def license_texts():
    """약관 문구(xg_meta.json 기록용, 계획 2.7 '약관과 내려받기')."""
    return {"cci5y": "CEDA 공개(등록·비등록 사용자). 약관 https://artefacts.ceda.ac.uk/licences/specific_licences/esacci_permafrost_terms_and_conditions.pdf"
                     ": ESA CCI 와 Permafrost CCI 프로젝트 사의 표시와 자료 DOI 인용(10.5285/a6fbedd8ee5b472c8e84e55f746c1704)",
            "wei": "Zenodo 21667583(v2), cc-by-4.0", "aalto": "Zenodo 5003804, CC0", "yk": "ORNL DAAC ds1760, EOSDIS 자유 이용",
            "cci4": "CEDA 공개(v4, 기존 cci_alt 열)"}


def extract(a):
    """제품 값 추출. T0 커밋을 확인하고, 라벨 셀 좌표에서 제품 값을 읽어 연도 정합 값을 쓴다. 라벨 값은 읽지 않는다."""
    genv = gdal_env()                                                      # rasterio 를 부르기 전(계획 1절 디스크 행)
    if not t0_committed():
        raise SystemExit(f"[추출] 계획 커밋 {XB.PLAN_COMMIT} 이 HEAD 의 조상이 아니다(계획 0.3: 커밋 뒤에만 추출한다)")
    cells = label_cells(a)
    rows, meta = [], dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), t0_commit=XB.PLAN_COMMIT, t0_ancestor=True, n_cells=int(len(cells)),
                          n_cells_no_year=int(cells.year_min.isna().sum()), products={}, licenses=license_texts(), plan=XB.PLAN_DOC,
                          rule="최근접 화소(주), 3 × 3 평균(민감도). 연도 정합: 겹친 연도 평균, 겹침 없으면 기간 평균(match_flag)", gdal=genv)
    cache = {}
    for p in a.PRODUCTS:
        spec = PRODUCTS[p]
        if spec.get("kind") == "column":
            continue                                                       # cci4 는 자료의 cci_alt 열을 쓴다(추출하지 않는다)
        base = spec.get("base", p)
        if base not in cache:
            bs = PRODUCTS[base]
            t0 = time.time()
            if bs["kind"] == "cci_nc":
                cache[base] = extract_cci(bs, cells, a.UNITS.get(base))
            elif bs["kind"] == "zip_tif":
                cache[base] = extract_tif(bs, cells, a.UNITS.get(base))
            elif bs["kind"] == "nc_latlon":
                cache[base] = extract_nc_latlon(bs, cells, a.UNITS.get(base))
            cache[base][4]["sec"] = round(time.time() - t0, 1)
        V, V3, years, dist, rec = cache[base]
        rule = spec["rule"]
        yrs = [y for y in years if y is not None]
        val, flag, ny = (year_rule(V, yrs, cells.year_min.values, cells.year_max.values, rule) if yrs
                         else (V[:, 0], np.where(np.isfinite(V[:, 0]), "static", "missing").astype(object), np.isfinite(V[:, 0]).astype(int)))
        val3, _, _ = year_rule(V3, yrs, cells.year_min.values, cells.year_max.values, rule) if yrs else (V3[:, 0], None, None)
        stat, _, _ = year_rule(V, yrs, cells.year_min.values, cells.year_max.values, "static") if yrs else (V[:, 0], None, None)
        for code in MISSING_CODES.get(p, ()):
            val = np.where(val == code, np.nan, val)
        rows.append(pd.DataFrame(dict(loc_id=cells.loc_id.values, product=p, value_cm=val, valid=np.isfinite(val).astype(int), match_flag=flag,
                                      n_years=ny, value_static_cm=stat, value3_cm=val3, pix_dist_km=np.round(dist, 4))))
        meta["products"][p] = dict(rec, base=base, rule=rule, period=list(PRODUCTS[base]["period"]), leak=spec["leak"], label=spec["label"],
                                   n_valid=int(np.isfinite(val).sum()), match_flags={str(k): int(v) for k, v in pd.Series(flag).value_counts().items()})
    tab = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    XB.check_out_dir(a.BASE_OUT)
    a.BASE_OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.BASE_OUT / VALUES_FILE, index=False)
    meta.update(values_file=VALUES_FILE, values_sha256=XB.sha256_file(a.BASE_OUT / VALUES_FILE), max_rss_mb=XB.max_rss_mb())
    old = json.loads((a.BASE_OUT / META_FILE).read_text()) if (a.BASE_OUT / META_FILE).exists() else {}
    old.update(extract=meta)
    (a.BASE_OUT / META_FILE).write_text(json.dumps(old, ensure_ascii=False, indent=1, default=float))
    print(f"[추출] 셀 {len(cells)} · 제품 {len(rows)} · 행 {len(tab)} · sha256 {meta['values_sha256'][:16]}", flush=True)
    return tab, meta


# ================================================================ 2. 누설 마스크(좌표만)
def train_points(kind):
    """제품 학습 지점 좌표 (lat, lon). wei = 보충 표(정확, L1), calm_gtnp = CALM ∪ GTN-P 활동층(대리, L2)."""
    if kind is None:
        return np.zeros((0, 2))
    if kind == "wei":
        x = pd.read_excel(RAW / "qtp_alt_open_2026-09-29" / "essd-2026-447_supplement" / "ALT_variables.xlsx", usecols=["LAT", "LONG"])
        return x[["LAT", "LONG"]].dropna().drop_duplicates().values.astype(float)
    if kind == "calm_gtnp":
        txt = (RAW / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab").read_text(errors="replace")
        body = txt.split("*/", 1)[1].lstrip("\n")
        from io import StringIO
        cm = pd.read_csv(StringIO(body), sep="\t", usecols=["Latitude", "Longitude"]).dropna().drop_duplicates().values.astype(float)
        js = json.loads((RAW / "gtnp" / "sites.json").read_text())
        gt = np.array([(al["latitude"], al["longitude"]) for s in js for al in (s.get("activelayers") or [])
                       if al.get("latitude") is not None and al.get("longitude") is not None], float)
        return np.unique(np.vstack([cm, gt.reshape(-1, 2)]), axis=0)
    raise ValueError(kind)


def block_mask(lat, lon, block, train, d_km):
    """블록 단위 마스크: 학습 지점에서 d km 안(≤ d)의 셀이 하나라도 있는 블록. 반환 (가린 블록 집합, 블록별 최소 거리 dict, 셀 거리)."""
    block = np.asarray(block).astype(str)
    if len(train) == 0:
        dist = np.full(len(block), np.inf)
    else:
        dist = X.nearest_km(np.asarray(lat, float), np.asarray(lon, float), train[:, 0], train[:, 1])
    mind = pd.Series(dist).groupby(block).min().to_dict()
    return {b for b, v in mind.items() if v <= float(d_km)}, mind, dist


def target_cells(a, name):
    """저장소 대상의 셀(자료 행 위치)과 자료. 모드 x·r 모두 대상 지역의 모든 셀이다(블록 단위 마스크의 '블록 안 셀')."""
    al, tgt, mode = resolve_alias(name)
    HA, D = XB.get_data(a54(a), al)
    return D, np.asarray(D.target_idx(tgt), np.int64)


def build_masks(a, names=None):
    """대상 × 제품 × d 의 블록 마스크 표와 셀 거리 표. 좌표만 쓴다."""
    names = list(names or a.TARGETS)
    trains = {k: train_points(k) for k in {PRODUCTS[p]["train"] for p in a.PRODUCTS}}
    rows, crows = [], []
    for nm in names:
        D, t_idx = target_cells(a, nm)
        df = D.df
        for p in a.PRODUCTS:
            tr = trains[PRODUCTS[p]["train"]]
            for d in MASK_D:
                mk, mind, dist = block_mask(df.lat.values[t_idx], df.lon.values[t_idx], df.block.values[t_idx], tr, d)
                cnt = pd.Series(df.block.values[t_idx].astype(str)).value_counts().to_dict()
                rows += [dict(target=nm, product=p, d_km=d, block=b, n_cells=int(cnt.get(b, 0)), min_dist_km=float(v), masked=bool(b in mk))
                         for b, v in sorted(mind.items())]
            if PRODUCTS[p]["train"] is not None:
                crows.append(pd.DataFrame(dict(target=nm, product=p, loc_id=df.loc_id.values[t_idx].astype(np.int64), dist_km=np.round(dist, 4),
                                               within_cell_d=(dist <= CELL_D).astype(int))))
    cells = pd.concat(crows, ignore_index=True) if crows else pd.DataFrame(columns=["target", "product", "loc_id", "dist_km", "within_cell_d"])
    return pd.DataFrame(rows), cells, {k: int(len(v)) for k, v in trains.items() if k}


def write_masks(a):
    tab, cells, ntr = build_masks(a)
    XB.check_out_dir(a.BASE_OUT)
    a.BASE_OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.BASE_OUT / MASK_FILE, index=False)
    cells.to_csv(a.BASE_OUT / CELLDIST_FILE, index=False)
    old = json.loads((a.BASE_OUT / META_FILE).read_text()) if (a.BASE_OUT / META_FILE).exists() else {}
    old["mask"] = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), d_km=list(MASK_D), cell_d_km=CELL_D, rule="학습 지점에서 d km 안(≤ d)의 셀이 있는 블록을 뺀다",
                       n_train_points=ntr, mask_sha256=XB.sha256_file(a.BASE_OUT / MASK_FILE), cells_sha256=XB.sha256_file(a.BASE_OUT / CELLDIST_FILE))
    (a.BASE_OUT / META_FILE).write_text(json.dumps(old, ensure_ascii=False, indent=1, default=float))
    g = tab.groupby(["target", "product", "d_km"]).masked.agg(["sum", "size"]).reset_index()
    for tg, pr, d, n_m, n_b in g.values:
        print(f"  [마스크] {tg} {pr} {float(d):g} km: 가린 블록 {int(n_m)}/{int(n_b)}", flush=True)
    print(f"[마스크] 행 {len(tab)} · 학습 지점 {ntr}", flush=True)
    return tab


def load_masks(a):
    p = a.BASE_OUT / MASK_FILE
    if not p.exists():
        return None
    t = pd.read_csv(p, dtype=dict(target=str, product=str, block=str))
    out = {}
    for (tg, pr, d), g in t.groupby(["target", "product", "d_km"]):
        out[(str(tg), str(pr), float(d))] = set(g.block[g.masked.astype(bool)].astype(str))
    return out


def masked_blocks(masks, name, product, mask):
    """마스크 이름 → 가린 블록 집합. 'none' 과 L0 제품은 빈 집합이다."""
    if mask == "none" or PRODUCTS[product]["train"] is None:
        return set()
    d = 5.0 if mask == "blk5" else 25.0
    if masks is None or (name, product, d) not in masks:
        raise SystemExit(f"[마스크] {name} {product} {d:g} km 의 마스크 표가 없다(--mask 를 먼저 실행한다)")
    return masks[(name, product, d)]


# ================================================================ 3. 재채점(기존 조각 + 해석식 제품 키)
def load_products(a):
    """제품 값 표 → {제품: Series(loc_id → value_cm)}. cci4 는 자료 열에서 읽으므로 넣지 않는다."""
    p = a.BASE_OUT / VALUES_FILE
    if not p.exists():
        return {}
    t = pd.read_csv(p, usecols=["loc_id", "product", "value_cm"])
    return {k: g.set_index("loc_id").value_cm for k, g in t.groupby("product")}


def shard_files(name, splits):
    """저장소 대상의 조각 파일 [(분할, npz, unit.json)]."""
    src, d, tag, tgt, mode = SOURCES[name]
    out = []
    for sp in splits:
        b = d / f"{tag}__cpu__{tgt}__{mode}__s{int(sp)}"
        npz, uj = Path(str(b) + "_blocksse.npz"), Path(str(b) + "_unit.json")
        if npz.exists() and uj.exists():
            out.append((int(sp), npz, uj))
    return out


def splits_of(name, a):
    return list(range(1, 26)) if SOURCES[name][0] == "wf6" and not a.smoke else list(a.SPLITS)


def load_raw_stores(a, name):
    """조각 → {분할: BlockStore}(저장소 이름을 name 으로 맞춘다)와 unit 목록. LG 대상은 LGX x9 의 P0@tddm·P1@tddm·cci 키를 병합한다."""
    files = shard_files(name, splits_of(name, a))
    by, units = {}, []
    for sp, npz, uj in files:
        st_ = list(H4.load_stores(npz).values())[0]
        st = BlockStore._from_arrays(name, sp, st_.blocks, st_.ncell, [], [], [], st_.meta)
        S, C = st_.matrices(st_.keys)
        for k, s, c in zip(st_.keys, S, C):
            st.add_sse(k, s, c)
        if SOURCES[name][0] == "lg":
            x9 = SH_LGX / f"lgx9__cpu__{SOURCES[name][3]}__x__s{sp}_blocksse.npz"
            if x9.exists():
                sx = list(H4.load_stores(x9).values())[0]
                keep = [k for k in sx.keys if k[0] in ("P0@tddm", "P1@tddm", "P0@cci", "P1@cci", "B:cci_raw", "B:cci_aff")]
                if keep:
                    if not (np.array_equal(sx.blocks, st.blocks) and np.array_equal(sx.ncell, st.ncell)):
                        raise SystemExit(f"[조각] {x9.name} 의 채점 블록이 LG 조각과 다르다")
                    S, C = sx.matrices(keep)
                    for k, s, c in zip(keep, S, C):
                        st.add_sse(tuple(k), s, c)
        by[sp] = st
        u = json.loads(uj.read_text())
        u.setdefault("split", sp)
        units.append(u)
    return by, units


def unit_indices(a, name, split):
    """작업 단위의 원천·A·채점 색인(자료 행 위치). 전이(x)는 h40.build_ctx 와 같은 규칙, 지역 내(r)는 h54.build_rctx 와 같은 규칙이다."""
    al, tgt, mode = resolve_alias(name)
    HA, D = XB.get_data(a54(a), al)
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx(tgt, "x")
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src = np.asarray(src_idx[ok], np.int64)
    A_idx, B_idx = half_split_blocks(df, t_idx, int(split))
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    return D, src, np.asarray(A_idx, np.int64), np.asarray(evB, np.int64), tgt, mode


def product_arrays(D, idx_sets, prod, pv):
    """제품 원값(cm) 배열 dict(src, A, B). cci4 는 자료의 cci_alt 열, 그 밖은 제품 값 표를 loc_id 로 결합한다(없으면 NaN)."""
    df = D.df
    out = {}
    for k, idx in idx_sets.items():
        if PRODUCTS[prod].get("kind") == "column":
            out[k] = df[PRODUCTS[prod]["column"]].values[idx].astype(float)
        else:
            s = pv.get(prod)
            out[k] = np.full(len(idx), np.nan) if s is None else s.reindex(df.loc_id.values[idx]).values.astype(float)
    return out


def product_keys(D, src, A, B, prod, pv, target, mode, split, grid, draws, inregion=False):
    """자료 행 위치로 제품 키를 낸다(product_keys_arrays 의 감싸기)."""
    df = D.df
    raw = product_arrays(D, dict(src=src, A=A, B=B), prod, pv)
    return product_keys_arrays(df.y.values[src].astype(float), df.s.values[src].astype(float), df.y.values[A].astype(float),
                               df.s.values[A].astype(float), df.s.values[B].astype(float), raw["src"], raw["A"], raw["B"], prod, target, mode,
                               split, grid, draws, inregion)


def product_keys_arrays(y_src, s_src, yA, sA, sB, raw_src, raw_A, raw_B, prod, target, mode, split, grid, draws, inregion=False):
    """제품 키의 채점 셀 예측 {키: 예측}과 기록. h42 의 앵커 규칙(anchor_fill, affine_ls, shrink)을 쓴다. B 라벨은 받지 않는다.
    inregion 이면 B:{p}_raw 만 낸다(XG-5). 원천 ρ 가 없으면(알래스카 전용 제품의 전이) A 쪽 ρ 로 B:{p}_raw 만 낸다.
    P1@{p} 의 추출은 h40.draw_cells(대상, 모드, 분할, n, 추출)라 LG 의 R1·P1 과 같은 라벨 집합이다.
    기록에는 채점 셀 가운데 ρ·s 로 바꾼 셀(비유한 또는 0 이하, anchor_fill 과 같은 규칙)의 수·비율과 그 셀 표지('_badB', 표에 쓰지 않는다)를 넣는다."""
    fill, rho, nrep = X.anchor_fill(raw_src, raw_A, raw_B, s_src, sA, sB)
    rb = np.asarray(raw_B, float)
    bad_B = ~(np.isfinite(rb) & (rb > 0))
    rec = dict(product=prod, rho=rho, n_replaced=nrep, rho_from="원천", n_B=int(len(rb)), n_B_filled=int(bad_B.sum()),
               frac_B_filled=float(bad_B.mean()) if len(rb) else np.nan, _badB=bad_B)
    out = {}
    if not np.isfinite(rho):
        fa, rho_a, nrep_a = X.anchor_fill(raw_A, raw_A, raw_B, sA, sA, sB)
        rec.update(rho=rho_a, n_replaced=nrep_a, rho_from="A 쪽(원천에 값 없음, B:{p}_raw 만)")
        if np.isfinite(rho_a):
            out[(f"B:{prod}_raw", "none", "1", "cell", 0, 0, -1, 0.0)] = fa["B"]
        return out, rec
    out[(f"B:{prod}_raw", "none", "1", "cell", 0, 0, -1, 0.0)] = fill["B"]
    if inregion:
        return out, rec
    y_src = np.asarray(y_src, float); yA = np.asarray(yA, float)
    a0, b0 = X.affine_ls(raw_src, y_src)
    out[(f"B:{prod}_aff", "none", "1", "cell", 0, 0, -1, 0.0)] = a0 + b0 * fill["B"]
    c0 = H.ls_E(y_src, fill["src"])
    out[(f"P0@{prod}", "none", "1", "cell", 0, 0, -1, 0.0)] = c0 * fill["B"]
    nA = len(yA)
    for n, d in H.cells_of(list(grid), draws, nA):
        sel = H.draw_cells(target, mode, int(split), n, d, nA)
        if len(sel) == 0:
            c1 = c0
        else:
            c1 = X.shrink(H.ls_E(yA[sel], fill["A"][sel]), c0, len(sel), KAPPA)
        out[(f"P1@{prod}", "none", "1", "cell", int(n), int(d), -1, 0.0)] = c1 * fill["B"]
    rec.update(c0=c0, aff=(a0, b0))
    return out, rec


def add_product_keys(a, name, by, pv, products, rec_out=None, fill_out=None):
    """{분할: BlockStore} 에 제품 키를 더한다. 채점 셀의 블록과 블록별 셀 수가 조각과 같은지 단언한다.
    fill_out 이 있으면 fill_out[(저장소, 제품)][분할] = 블록별 ρ·s 대체 채점 셀 수(조각의 블록 순서)를 쓴다(제품 키를 낸 경우만)."""
    al, tgt, mode = resolve_alias(name)
    for sp, st in by.items():
        D, src, A, B, tgt_, _ = unit_indices(a, name, sp)
        blk = D.df.block.values[B].astype(str)
        ub, codes, cnt = np.unique(blk, return_inverse=True, return_counts=True)
        if not (np.array_equal(ub, st.blocks) and np.array_equal(cnt, st.ncell)):
            raise SystemExit(f"[제품 키] {name} s{sp}: 다시 만든 채점 셀이 조각의 블록·셀 수와 다르다")
        tmp = BlockStore(name, sp, blk)
        y = D.df.y.values[B].astype(float)
        for p in products:
            if PRODUCTS[p].get("regions") and al.split("~")[0] not in PRODUCTS[p]["regions"]:
                continue
            keys, rec = product_keys(D, src, A, B, p, pv, tgt_, "x", sp, LG_GRID, LG_DRAWS, inregion=(mode == "r"))
            bad = np.asarray(rec.get("_badB", np.zeros(len(B), bool)), bool)
            if rec_out is not None:
                rec_out.append(dict(target=name, split=sp, **{k: v for k, v in rec.items() if k != "aff" and not str(k).startswith("_")}))
            if fill_out is not None and keys:
                fill_out.setdefault((name, p), {})[sp] = np.bincount(codes, weights=bad.astype(float), minlength=len(ub)).astype(np.int64)
            for k, pred in keys.items():
                if np.all(np.isfinite(pred)):
                    tmp.add(k, y, pred)
        for k in tmp.keys:
            s, c = tmp.get(k)
            st.add_sse(k, s, c)
    return by


def mask_store(st, drop):
    """채점 블록 집합에서 drop 블록을 뺀 저장소 사본(블록 SSE 열을 고른다). 남은 블록이 없으면 None."""
    keep = np.array([b not in drop for b in st.blocks.astype(str)])
    if not keep.any():
        return None
    S, C = st.matrices(st.keys)
    return BlockStore._from_arrays(st.target, st.split, st.blocks[keep], st.ncell[keep], st.keys, S[:, keep], C[:, keep], st.meta)


def make_tm(a, name, by, units, drop=None):
    """마스크(drop 블록)를 적용한 h40.TMx(h54.make_tm 규칙: dup_of·valid 는 unit.json). 반환 (TMx, 분할별 블록 수 dict)."""
    bs = {}
    for sp, st in by.items():
        st2 = mask_store(st, drop or set())
        if st2 is not None:
            bs[sp] = st2
    if not bs:
        return None, {}
    tm = W.make_tm(name, bs, [u for u in units if int(u["split"]) in bs], int(a.nboot), XB.is_point_only_name(name))
    return tm, {sp: int(st.nb) for sp, st in tm.used.items()}


def grp(method, n, lam=None):
    """곡선 키. R1 은 catboost_lo λ 0.25, D0 는 catboost_lo λ 1.0(전이) 또는 catboost λ 1.0(WF6), 제품 키·P1 은 learner none."""
    if method == "P0":
        return H.P0_GRP
    if method == "R1":
        return ("R1", XB.LO, "1", "cell", int(n), float(XB.LAM_BASE if lam is None else lam))
    if method == "D0":
        return ("D0", XB.LO, "1", "cell", int(n), 1.0)
    if method == "D0hi":
        return ("D0", XB.HI, "1", "cell", int(n), 1.0)
    return (method, "none", "1", "cell", int(n), 0.0)


class Rescore:
    """재채점 상태: 대상별 원저장소, 제품 키, 마스크별 TMx, 제품별 ρ·s 대체 채점 셀 수(fill)."""

    def __init__(self, a, products=None):
        self.a = a
        self.products = list(products or a.PRODUCTS)
        self.pv = load_products(a)
        self.masks = load_masks(a)
        self.raw, self.units, self.recs = {}, {}, []
        self.fill = {}
        self._tm = {}

    def load(self, name):
        if name not in self.raw:
            by, units = load_raw_stores(self.a, name)
            if not by:
                self.raw[name], self.units[name] = {}, []
                return {}
            prods = [p for p in self.products if (PRODUCTS[p].get("kind") == "column" or p in self.pv)]
            add_product_keys(self.a, name, by, self.pv, prods, self.recs, self.fill)
            self.raw[name], self.units[name] = by, units
        return self.raw[name]

    def fill_blocks(self, name, product):
        """ρ·s 로 바꾼 채점 셀이 하나라도 있는 블록(분할 합집합)."""
        F = getattr(self, "fill", {}).get((name, product)) or {}
        by = self.raw.get(name, {})
        out = set()
        for sp, nf in F.items():
            if sp in by:
                out |= set(by[sp].blocks.astype(str)[np.asarray(nf) > 0])
        return out

    def fill_counts(self, name, product, drop=(), splits=None):
        """drop 블록을 뺀 채점 셀의 ρ·s 대체 수. 반환 (대체 셀 수, 채점 셀 수, 분할별 비율 목록). 기록이 없으면 (0, 0, [])."""
        F = getattr(self, "fill", {}).get((name, product)) or {}
        by = self.raw.get(name, {})
        drop = {str(b) for b in drop}
        nf_t, nc_t, fr = 0, 0, []
        for sp, nf in F.items():
            if sp not in by or (splits is not None and sp not in splits):
                continue
            st = by[sp]
            keep = np.array([b not in drop for b in st.blocks.astype(str)], bool)
            c, f = int(np.asarray(st.ncell)[keep].sum()), int(np.asarray(nf)[keep].sum())
            if c > 0:
                nf_t += f; nc_t += c; fr.append(f / c)
        return nf_t, nc_t, fr

    def tm(self, name, product, mask, excl_fill=False):
        """마스크(none, blk5, blk25)를 적용한 TMx. excl_fill 이면 ρ·s 대체 채점 셀이 있는 블록도 함께 뺀다(서술 민감도, 두 팔 같은 블록)."""
        k = (name, product if (mask != "none" or excl_fill) else "", mask, bool(excl_fill))
        if k not in self._tm:
            by = self.load(name)
            if not by:                                                     # 조각이 없는 대상은 마스크 표를 보지 않고 건너뛴다
                self._tm[k] = (None, {})
                return self._tm[k]
            drop = set(masked_blocks(self.masks, name, product, mask)) if mask != "none" else set()
            if excl_fill:
                drop |= self.fill_blocks(name, product)
            self._tm[k] = make_tm(self.a, name, by, self.units.get(name, []), drop)
        return self._tm[k]

    def drop_set(self, name, product, mask, excl_fill=False):
        drop = set(masked_blocks(self.masks, name, product, mask)) if mask != "none" else set()
        return drop | (self.fill_blocks(name, product) if excl_fill else set())


def fill_note(frac) -> str:
    """'제품 결측 대체 k%'(채점 셀 가운데 ρ·s 로 바꾼 셀의 비율). 0 이거나 계산할 수 없으면 빈 문자열."""
    if frac is None or not np.isfinite(float(frac)) or float(frac) <= 0:
        return ""
    pct = 100.0 * float(frac)
    return f"{FILL_TXT} {pct:.1f}%" if pct >= 0.05 else f"{FILL_TXT} 0.1% 미만"


def leak_cols(product) -> dict:
    """누설 점검 열(계획 2.7 공통 행): 학습 지점 좌표가 없는 L2 제품(Aalto)은 '누설 점검 불가(대리 마스크)'와 SI 전용."""
    lk = PRODUCTS[product]["leak"]
    return dict(leak_grade=lk, leak_check=LEAK_CHECK.get(lk, ""), si_only=bool(lk == "L2"))


def contrast_rows(R, names, gA, gB, product, mask, label, registered=None, excl_fill=False):
    """풀 대비(같은 라벨 집합). 지역 행·층화 평균 행(xbatch_core.pool)에 마스크 전후 P0 RMSE, 소수 블록 표지, h42 의 동등성 p 를 더한다.
    모든 행에 채점 셀 가운데 제품 값을 ρ·s 로 바꾼 셀의 비율(fill_frac_B = 대체 셀 수 / 채점 셀 수, 사용 분할 합계)과 분할별 최댓값,
    '제품 결측 대체 k%' 표지를 싣는다. excl_fill 이면 대체 셀이 있는 블록을 두 팔에서 같이 뺀 서술 민감도다."""
    tms, pre_min, post_min, p0pre, p0post, fills = {}, {}, {}, {}, {}, {}
    for nm in names:
        tm, nb = R.tm(nm, product, mask, excl_fill)
        tm0, nb0 = R.tm(nm, product, "none")
        if tm is None:
            continue
        tms[nm] = tm
        pre_min[nm] = int(min(nb0.values())) if nb0 else 0
        post_min[nm] = int(min(nb.values())) if nb else 0
        p0pre[nm] = XB.grp_rmse2(tm0, H.P0_GRP) if tm0 is not None else (np.nan, np.nan)
        p0post[nm] = XB.grp_rmse2(tm, H.P0_GRP)
        fills[nm] = R.fill_counts(nm, product, R.drop_set(nm, product, mask, excl_fill), splits=set(nb))
    rows, per = XB.contrast_pool(tms, names, gA, gB, label, kind="same", registered=registered or len(names))
    for r in rows:
        nm = r.get("target", "")
        use = [nm] if r.get("scope") == "region" else [n for n in names if n in tms]
        r.update(product=product, mask=mask, gA="|".join(str(v) for v in gA), gB="|".join(str(v) for v in gB),
                 nb_split_min_premask=int(min(pre_min[n] for n in use)) if use else 0, nb_split_min_postmask=int(min(post_min[n] for n in use)) if use else 0,
                 rmse_p0_premask=float(np.nanmean([p0pre[n][0] for n in use])) if use else np.nan,
                 rmse_p0_postmask=float(np.nanmean([p0post[n][0] for n in use])) if use else np.nan)
        r["few_blocks"] = XB.FEW_BLOCKS_TXT if r["nb_split_min_premask"] < XB.FEW_BLOCKS else ""
        r["mask_few_blocks"] = "마스크 뒤 소수 블록" if ((mask != "none" or excl_fill) and r["nb_split_min_postmask"] < XB.MIN_BLOCKS_CI) else ""
        nf = sum(fills[n][0] for n in use if n in fills); nc = sum(fills[n][1] for n in use if n in fills)
        fr = [v for n in use if n in fills for v in fills[n][2]]
        frac = (nf / nc) if nc > 0 else np.nan
        r.update(n_B_filled=int(nf), n_B_scored=int(nc), fill_frac_B=frac, fill_frac_B_max_split=float(max(fr)) if fr else np.nan,
                 fill_note=fill_note(frac), fill_excluded=bool(excl_fill),
                 n_blocks_fill_dropped=int(len(set().union(*[R.fill_blocks(n, product) for n in use]))) if (excl_fill and use) else 0)
    return rows, per


def wording(v4, p_holm, p_eq_holm):
    """WRAPUP 1.1 (a)–(d): Holm 보정 p 로 판정 문구의 갈래를 정한다. 반환 (갈래, 표지)."""
    if v4 in ("우세", "열세"):
        return (v4, "") if (np.isfinite(p_holm) and p_holm < XB.HOLM_ALPHA) else ("미결정", XB.UNCORRECTED_TXT)
    if v4 == "동등":
        return ("동등", "") if (np.isfinite(p_eq_holm) and p_eq_holm <= XB.ALPHA_ONE_SIDED) else ("미결정", "보정 전 동등")
    if v4 == "미결정":
        return "미결정", ""
    return "판정 불가", ""


def combine_masks(branches):
    """두 공동 주 마스크의 갈래를 합친다. 같으면 그대로, 다르면 약한 쪽과 '마스크 의존'(우세와 열세가 엇갈리면 미결정)."""
    b = list(branches)
    if len(set(b)) == 1:
        return b[0], ""
    if {"우세", "열세"} <= set(b):
        return "미결정", "마스크 의존"
    return min(b, key=lambda v: STRENGTH.get(v, 0)), "마스크 의존"


def sentence(group, branch, product, delta, k, m, n, notes, worse):
    """사전 고정 해석 문장(계획 2.7)과 공통 표지."""
    spec = PRODUCTS[product]
    nlp = "" if n is None else ("전량을" if int(n) == -1 else f"{int(n)}개를")
    t = TEMPLATES[group][branch].format(X=spec["label"], a=f"{abs(float(delta)):.2f}" if np.isfinite(delta) else "NA", k=f"{k}/{m}", nl=nlp)
    extra = list(notes)
    if group in ("XG-2", "XG-3", "XG-4r") and branch == "열세" and spec.get("overlap"):
        extra.append(OVERLAP_TXT)
    out = t + (f"({', '.join(e for e in extra if e)})" if any(extra) else "") + "."
    if group == "XG-1" and spec["train"] is not None:
        out += " " + CALM_CLAUSE.format(X=spec["label"]) + "."
    for nm, d, lo, hi in worse:
        out += f" 지역 {nm} 에서는 오차가 컸다(Δ {float(d):+.2f} cm, CI [{float(lo):.2f}, {float(hi):.2f}])."
    return out


MASK_KM = {"blk5": "5", "blk25": "25"}
REGION_GENERAL_TXT = "지역 일반 문장 허용(HK CI 가 0 을 제외)"
CALM_TRAIN = ("wei", "calm_gtnp")                                         # CALM 지점으로 학습한 제품(러시아 W·E 는 학습 표본 안, XG-7)
MASK_COPY_TEXT = ("verdict4_d10", "verdict4_rel", "limit_dependence", "ci_dependence", "pool", "few_block_regions", "mask_few_blocks", "fill_note")
MASK_COPY_NUM = ("delta", "fill_frac_B", "fill_frac_B_max_split", "rmse_p0_postmask", "n_ci_regions")


def _mean_row(rows):
    return rows[-1] if rows and rows[-1].get("scope") == "MEAN" else None


def mask_tags(mr) -> list:
    """MEAN 행의 1절 의존 표지: '한계 의존'(주 δ·보조 δ·δ_rel 판정이 다름), '분할 독립 가정 의존'(h42.boot_delta_common 판정이 다름)."""
    mr = mr or {}
    return [str(mr.get(k)) for k in ("limit_dependence", "ci_dependence") if mr.get(k)]


def worse_union(info) -> list:
    """두 공동 주 마스크의 지역 행 가운데 열세(두 가중 CI 하한 > 0) 지역의 합집합 [(지역(마스크), Δ, CI 하한, CI 상한)].
    두 마스크 모두 열세면 앞 마스크(5 km)의 값을 쓰고 이름 뒤에 '5·25 km 마스크'를 적는다."""
    seen = {}
    for m in CO_PRIMARY:
        for nm, d, lo, hi in (XB.worse_regions(info[m]["rows"]) if info[m].get("rows") else []):
            if nm in seen:
                seen[nm][1].append(m)
            else:
                seen[nm] = ((d, lo, hi), [m])
    return [(f"{nm}({'·'.join(MASK_KM[m] for m in ms)} km 마스크)", *v) for nm, (v, ms) in seen.items()]


def combined_notes(info, final, mdep, delta=np.nan) -> list:
    """두 공동 주 마스크의 문장 표지 합집합(순서: '부분(지역 k/m)', Holm 문구 표지, '마스크 의존', '한계 의존'·'분할 독립 가정 의존',
    효과 크기, '제품 결측 대체 k%', 판정 불가 사유). info[마스크] = dict(mr, rows, note, branch)."""
    notes = []
    for m in CO_PRIMARY:
        pl = str((info[m]["mr"] or {}).get("pool", ""))
        if pl.startswith("부분"):
            notes.append(pl)
    notes += [info[m].get("note", "") for m in CO_PRIMARY]
    notes.append(mdep)
    for m in CO_PRIMARY:
        notes += mask_tags(info[m]["mr"])
    if final in ("우세", "열세") and delta is not None and np.isfinite(float(delta)) and abs(float(delta)) < XB.SMALL_EFFECT_CM:
        notes.append(XB.SMALL_EFFECT_TXT)
    fr = [float((info[m]["mr"] or {}).get("fill_frac_B", np.nan)) for m in CO_PRIMARY]
    fr = [v for v in fr if np.isfinite(v)]
    if fr:
        notes.append(fill_note(max(fr)))
    if final == "판정 불가":
        why = [str((info[m]["mr"] or {}).get("ci_flag", "") or "") for m in CO_PRIMARY if info[m].get("branch") == "판정 불가"]
        notes += [w for w in why if w] or ["행 없음"]
    return list(dict.fromkeys(n for n in notes if n))


def common_suffix(product, info) -> str:
    """공통 문장(계획 2.7 공통 행): 제품 ALT 정의, 마스크 전후 P0 RMSE(두 마스크), 등록 이탈 표지."""
    ref = info[CO_PRIMARY[0]]["mr"] or {}
    post = "·".join(f"{float((info[m]['mr'] or {}).get('rmse_p0_postmask', np.nan)):.2f}({MASK_KM[m]} km)" for m in CO_PRIMARY)
    return (f" 공통: {PRODUCTS[product]['alt_def']}, 우리 라벨은 GPR·탐침 혼합. 마스크 전후 P0 RMSE "
            f"{float(ref.get('rmse_p0_premask', np.nan)):.2f} → {post} cm. {DEVIATION}.")


def mask_cols(info) -> dict:
    """마스크마다 MEAN 행에서 판정 표로 옮기는 열(1절 보조 판정 δ 1.0·δ_rel, '한계 의존', '분할 독립 가정 의존', 지역 일반 조건(HK), 풀 표지,
    소수 블록, 제품 결측 대체 비율). 열 이름 = <열>_<마스크>(풀 표지는 pool_label_<마스크>). region_general 은 두 마스크 모두 참일 때 참."""
    out = {}
    for m in CO_PRIMARY:
        mr = info[m]["mr"] or {}
        for c in MASK_COPY_TEXT:
            out[f"{'pool_label' if c == 'pool' else c}_{m}"] = str(mr.get(c, "") or "")
        for c in MASK_COPY_NUM:
            v = mr.get(c, np.nan)
            out[f"{c}_{m}"] = float(v) if v is not None and not isinstance(v, str) else np.nan
        out[f"region_general_{m}"] = bool(mr.get("region_general", False))
    out["region_general"] = all(out[f"region_general_{m}"] for m in CO_PRIMARY)
    return out


def confirm_tests(R):
    """확인 대비 6개. 마스크마다 Holm(m = 6, 두 가중 가운데 큰 양측 p, 행 없음 p 1)과 동등성 Holm, 두 마스크 결합 갈래와 해석 문장.
    문장의 표지(부분, 의존, 결측 대체)와 열세 지역은 두 공동 주 마스크의 합집합이다. 마스크별 MEAN 행의 보조 열은 mask_cols 로 옮긴다."""
    per_mask, all_rows = {}, []
    for mask in CO_PRIMARY:
        res = []
        for h in CONFIRM:
            gA = grp(h["gA"][0], h["gA"][1]); gB = grp(h["gB"][0], h["gB"][1])
            rows, per = contrast_rows(R, h["pool"], gA, gB, h["product"], mask, h["id"], registered=len(h["pool"]))
            mr = _mean_row(rows)
            res.append((h, mr, rows))
            for r in rows:
                all_rows.append(dict(r, hypothesis=h["id"], blind=h["blind"], deviation=DEVIATION, design=DESIGN))
        ph = XB.holm([None if (mr is None or mr.get("undetermined")) else mr.get("p_two") for _, mr, _ in res], HOLM_M)
        pe = XB.holm([None if (mr is None or mr.get("undetermined")) else mr.get("p_eq") for _, mr, _ in res], HOLM_M)
        per_mask[mask] = [(h, mr, rows, float(ph[i]), float(pe[i])) for i, (h, mr, rows) in enumerate(res)]
    out = []
    for i, h in enumerate(CONFIRM):
        br, info = [], {}
        for mask in CO_PRIMARY:
            _, mr, rows, ph, pe = per_mask[mask][i]
            v4 = mr.get("verdict4") if mr is not None else "판정 불가"
            b, note = wording(v4, ph, pe)
            br.append(b)
            info[mask] = dict(mr=mr, rows=rows, ph=ph, pe=pe, v4=v4, branch=b, note=note)
        final, mdep = combine_masks(br)
        mr = info[CO_PRIMARY[0]]["mr"] or {}
        notes = combined_notes(info, final, mdep, mr.get("delta", np.nan))
        worse = worse_union(info)
        k = int(mr.get("n_ci_regions", 0) or 0)
        txt = sentence(h["group"], final, h["product"], mr.get("delta", np.nan), k, len(h["pool"]), h["gA"][1] if h["gA"][0] == "R1" else None, notes,
                       worse) + common_suffix(h["product"], info)
        mc = mask_cols(info)
        tags = notes + ([REGION_GENERAL_TXT] if mc["region_general"] else [])
        fr = [mc[f"fill_frac_B_{m}"] for m in CO_PRIMARY if np.isfinite(mc[f"fill_frac_B_{m}"])]
        out.append(dict(hypothesis=h["id"], product=h["product"], pool=",".join(h["pool"]), blind=h["blind"], deviation=DEVIATION, design=DESIGN,
                        **{f"verdict4_{m}": info[m]["v4"] for m in CO_PRIMARY}, **{f"p_holm_{m}": info[m]["ph"] for m in CO_PRIMARY},
                        **{f"p_eq_holm_{m}": info[m]["pe"] for m in CO_PRIMARY}, **{f"branch_{m}": info[m]["branch"] for m in CO_PRIMARY},
                        branch=final, mask_dependence=mdep, holm_m=HOLM_M, sentence=txt, tags="; ".join(tags),
                        worse_regions_union=";".join(w[0] for w in worse), fill_frac_B=max(fr) if fr else np.nan,
                        delta=mr.get("delta", np.nan), ci_lo=mr.get("ci_lo", np.nan), ci_hi=mr.get("ci_hi", np.nan),
                        delta_blockeq=mr.get("delta_blockeq", np.nan), ci_lo_beq=mr.get("ci_lo_beq", np.nan), ci_hi_beq=mr.get("ci_hi_beq", np.nan),
                        **mc))
    return pd.DataFrame(out), XB.clean_rows(all_rows)


def fill_sensitivity(R):
    """서술 민감도(Holm 없음, 판정 표가 아니다): 확인 대비 6개를 제품 결측 대체(ρ·s) 채점 셀이 있는 블록을 두 팔에서 같이 뺀 뒤 다시 계산한다.
    블록 SSE 조각(LG)에는 셀 단위 예측이 없으므로 셀 대신 그 셀이 든 블록을 뺀다. 두 마스크 결합 갈래는 보정 전 4분 판정으로 정한다."""
    rows = []
    for h in CONFIRM:
        gA = grp(h["gA"][0], h["gA"][1]); gB = grp(h["gB"][0], h["gB"][1])
        info = {}
        for mask in CO_PRIMARY:
            rr, _ = contrast_rows(R, h["pool"], gA, gB, h["product"], mask, h["id"], registered=len(h["pool"]), excl_fill=True)
            info[mask] = _mean_row(rr), rr
        final, mdep = combine_masks([XB.branch_of(info[m][0].get("verdict4")) if info[m][0] else "판정 불가" for m in CO_PRIMARY])
        for mask in CO_PRIMARY:
            for r in info[mask][1]:
                d = dict(r, hypothesis=h["id"], role=FILL_SENS_TXT, deviation=DEVIATION, design=DESIGN, holm="없음(서술)")
                if r.get("scope") == "MEAN":
                    d.update(branch_descriptive=final, mask_dependence=mdep)
                rows.append(d)
    return XB.clean_rows(rows)


def _masks_of(product):
    return CO_PRIMARY if PRODUCTS[product]["train"] else ("none",)


def descriptive_tests(R):
    """서술 대비: XG-4r, XG-4, XG-5, XG-6. 판정 열은 보조로만 싣는다(Holm 없음).
    XG-4r 은 확인 대비와 같은 문장 규칙(두 마스크 결합 갈래, 사전 고정 문장, 공통 표지)을 따른다(계획 2.7 공통 행).
    XG-4 는 CALM 학습 제품(Wei, Aalto)의 풀에서 러시아 W·E 를 빼고, 그 지역 행은 별표 참고값(판정 열 비움)으로만 싣는다(계획 2.7 XG-7).
    모든 행에 누설 점검 열(L2 = '누설 점검 불가(대리 마스크)', SI 전용)을 단다."""
    rows = []

    def role_of(product):
        return "서술" + (f"({SI_ONLY_TXT})" if PRODUCTS[product]["leak"] == "L2" else "")

    def add(hid, names, gA, gB, product, mask, n=None):
        rr, _ = contrast_rows(R, names, gA, gB, product, mask, hid)
        rows.extend(dict(r, hypothesis=hid, n_lab=n, deviation=DEVIATION, role=role_of(product), ref_star="", **leak_cols(product)) for r in rr)
        return rr

    for p, pool in (("wei", POOL_W), ("cci5y", POOL_C)):                   # XG-4r(같은 문장 규칙, Holm 없음)
        if p not in R.products:
            continue
        for n in (10, -1):
            info = {}
            for mask in CO_PRIMARY:
                rr, _ = contrast_rows(R, pool, grp("R1", n), grp(f"B:{p}_raw", 0), p, mask, "XG-4r")
                mr = _mean_row(rr)
                info[mask] = dict(mr=mr, rows=rr, note="", branch=XB.branch_of(mr.get("verdict4")) if mr else "판정 불가")
            final, mdep = combine_masks([info[m]["branch"] for m in CO_PRIMARY])
            mr0 = info[CO_PRIMARY[0]]["mr"] or {}
            notes = combined_notes(info, final, mdep, mr0.get("delta", np.nan))
            txt = sentence("XG-4r", final, p, mr0.get("delta", np.nan), int(mr0.get("n_ci_regions", 0) or 0), len(pool), n, notes,
                           worse_union(info)) + common_suffix(p, info)
            for mask in CO_PRIMARY:
                for r in info[mask]["rows"]:
                    d = dict(r, hypothesis="XG-4r", n_lab=n, deviation=DEVIATION, role="보조, 서술", ref_star="", **leak_cols(p))
                    if r.get("scope") == "MEAN":
                        d.update(branch=final, mask_dependence=mdep, sentence=txt, tags="; ".join(notes), holm="없음(서술)")
                    rows.append(d)
    for p in R.products:                                                   # XG-4
        calm = PRODUCTS[p]["train"] in CALM_TRAIN
        pool = [nm for nm in POOL_C if not (calm and nm in XG7)]
        for n in (10, -1):
            for mask in _masks_of(p):
                add("XG-4", pool, grp(f"P1@{p}", n), grp("P1", n), p, mask, n)
                if not calm:
                    continue
                for nm in XG7:                                             # XG-7: 학습 표본 안 참고값(별표, 판정 없음, 풀에 넣지 않는다)
                    rr, _ = contrast_rows(R, [nm], grp(f"P1@{p}", n), grp("P1", n), p, mask, "XG-4")
                    for r in rr:
                        if r.get("scope") != "region":
                            continue
                        rows.append(dict(r, hypothesis="XG-4", n_lab=n, deviation=DEVIATION, role=REF_ROLE, ref_star=REF_STAR, **leak_cols(p),
                                         **{c: "" for c in VERDICT_COLS if c in r}))
    for p in ("cci5y", "wei", "yk"):                                       # XG-5(지역 내 WF6)
        if p not in R.products:
            continue
        names = ["Alaska|r"] if p == "yk" else INREG
        for n in (200, 500, 1000, -1):
            for mth in ("R1", "P1", "D0hi"):
                for mask in _masks_of(p):
                    add("XG-5", names, grp(mth, n), grp(f"B:{p}_raw", 0), p, mask, n)
    for p in R.products:                                                   # XG-6(대상 행)
        for nm in XG6:
            for mask in _masks_of(p):
                add("XG-6", [nm], grp(f"B:{p}_raw", 0), grp("P0", 0), p, mask, 0)
                for n in (10, -1):
                    add("XG-6", [nm], grp("R1", n), grp(f"P1@{p}", n), p, mask, n)
    return XB.clean_rows(rows)


def rmse_table(R, names, products):
    """키별 RMSE(셀 가중·블록 등가중) 표. XG-7(러시아 W·E 의 CALM 학습 제품 참고값)은 별표 열을 달고, L2 제품(Aalto)은 누설 점검 불가·SI 전용이다."""
    rows = []
    for nm in names:
        for p in products:
            for mask in ("none",) + (CO_PRIMARY if PRODUCTS[p]["train"] else ()):
                tm, _ = R.tm(nm, p, mask)
                if tm is None:
                    continue
                gs = [H.P0_GRP, grp(f"B:{p}_raw", 0), grp(f"B:{p}_aff", 0), grp(f"P0@{p}", 0), grp("P0@tddm", 0)] + \
                     [g for n in (10, -1) for g in (grp("P1", n), grp(f"P1@{p}", n), grp("R1", n))]
                star = bool(nm in XG7 and PRODUCTS[p]["train"] in CALM_TRAIN)
                role = REF_ROLE if nm in XG7 else "서술"
                if PRODUCTS[p]["leak"] == "L2":
                    role += f"({SI_ONLY_TXT})"
                for g in gs:
                    c, b = XB.grp_rmse2(tm, g)
                    if np.isfinite(c):
                        rows.append(dict(target=nm, product=p, mask=mask, key="|".join(str(v) for v in g), rmse=c, rmse_beq=b,
                                         ref_star=REF_STAR if star else "", role=role, **leak_cols(p)))
    return pd.DataFrame(rows)


def check_nboot(a):
    """본 봉인 폴더에는 등록 재표집 수(10,000회)로만 쓴다. 허용 표지 없는 로컬 실행(1,000회 상한)이나 --nboot 를 줄인 실행은 멈춘다."""
    if not a.smoke and int(a.nboot) < int(XB.NBOOT):
        raise SystemExit(f"[집계] 재표집 {int(a.nboot)}회는 등록 {int(XB.NBOOT)}회보다 적다. 본 봉인 폴더에 쓰지 않는다"
                         "(로컬은 --allow-local, 계획 1절 '대비와 CI')")


def gate_status(a) -> dict:
    """관문 요약(xg_gate_meta.json)을 봉인 메타에 옮길 형태로 읽는다. 관문 기록보다 새 입력(제품 값·마스크·추출 기록·셀 단위 조각)이 있으면 stale."""
    p = a.OUT / GATE_META
    if not p.exists():
        return dict(found=False, all_passed=False, stale=False, note="관문 기록 없음(--gate-only 를 먼저 실행한다)")
    m = json.loads(p.read_text())
    summ = m.get("summary", {}) or {}
    newer = [q for q in (a.BASE_OUT / VALUES_FILE, a.BASE_OUT / MASK_FILE, a.BASE_OUT / META_FILE) if q.exists()]
    if a.SHARDS.exists():
        newer += list(a.SHARDS.glob(f"{a.TAG}__*_unit.json"))
    stale = bool(newer and max(q.stat().st_mtime for q in newer) > p.stat().st_mtime)
    return dict(found=True, created=m.get("created", ""), summary=summ, all_passed=bool(summ) and all(bool(v.get("passed")) for v in summ.values()),
                stale=stale, note="관문 기록이 입력보다 오래됐다(관문을 다시 실행한다)" if stale else "")


def summarize(a):
    """재채점 전체. 표는 봉인 폴더에 쓴다. 본 봉인 폴더에는 재표집 10,000회로만 쓰고, 관문 요약을 메타에 옮긴다."""
    check_nboot(a)
    R = Rescore(a)
    if not R.pv:
        print("[집계] 제품 값 표가 없다. cci4(기존 열)만 계산한다", flush=True)
        R.products = [p for p in R.products if PRODUCTS[p].get("kind") == "column"]
    names = [n for n in a.TARGETS if shard_files(n, splits_of(n, a))]
    for nm in names:
        R.load(nm)
    out = {}
    if all(p in R.products for p in ("wei", "cci5y")):
        tests, rows = confirm_tests(R)
        out.update(xg_tests=tests, xg_confirm_rows=rows, xg_fill_sensitivity=fill_sensitivity(R))
    out["xg_descriptive"] = descriptive_tests(R)
    out["xg_key_scores"] = rmse_table(R, [n for n in names if n.endswith("|x")], R.products)
    out["xg_product_key_notes"] = pd.DataFrame(R.recs)
    cp = cell_summary(a)
    if cp is not None:
        out["xg_tests_cell"] = cp
    sealed = NAME if not a.smoke else f"{NAME}/smoke"
    gs = gate_status(a)
    XB.write_sealed(sealed, {f"{k}.csv": v for k, v in out.items()}, root=XB.XBATCH_ROOT)
    XB.write_sealed(sealed, {"xg_summary_meta.json": dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), nboot=int(a.nboot), nboot_registered=int(XB.NBOOT),
                                                          targets=names, products=R.products, holm_m=HOLM_M, deviation=DEVIATION,
                                                          masks=list(CO_PRIMARY), n_rows={k: int(len(v)) for k, v in out.items()}, gates=gs,
                                                          leak_check={p: LEAK_CHECK.get(PRODUCTS[p]["leak"], "") for p in R.products},
                                                          file_location=LOCATION_NOTE)},
                    root=XB.XBATCH_ROOT)
    print(f"[집계] 관문 기록 {'있음' if gs['found'] else '없음'} · 모두 통과 {gs['all_passed']} · 오래됨 {gs['stale']}", flush=True)
    return out


# ================================================================ 4. 셀 단위 민감도(R3, 적합)
def cell_dist(a, name, loc_id, lat, lon, product="wei", strict=False):
    """채점 셀에서 제품 학습 지점까지의 거리(km). 마스크 단계의 셀 거리 표(xg_leak_cells_v1.csv)가 있으면 그 값을 쓰고(Rescale 묶음에는 이 표만
    보낸다), 없으면 학습 지점 표에서 다시 계산한다(좌표만, 세기·스모크 전용). strict 이면(작업 R3 본 실행) 표가 없거나 대상·제품 행이 없거나
    채점 셀 가운데 거리가 비유한인 셀이 있으면 멈춘다(학습 지점 표는 묶음에 없다)."""
    p = a.BASE_OUT / CELLDIST_FILE
    if p.exists():
        t = pd.read_csv(p, usecols=["target", "product", "loc_id", "dist_km"], dtype=dict(target=str, product=str))
        t = t[(t.target == name) & (t["product"] == product)].drop_duplicates("loc_id").set_index("loc_id").dist_km
        if len(t):
            d = t.reindex(np.asarray(loc_id, np.int64)).values.astype(float)
            if np.isfinite(d).all():
                return d
            if strict:
                raise SystemExit(f"[셀 단위] {CELLDIST_FILE} 에서 {name} {product} 채점 셀 {int((~np.isfinite(d)).sum())}개의 거리가 없다(--mask 를 다시 만든다)")
        elif strict:
            raise SystemExit(f"[셀 단위] {CELLDIST_FILE} 에 {name} {product} 행이 없다(--mask 를 다시 만든다)")
    elif strict:
        raise SystemExit(f"[셀 단위] {p} 가 없다. 작업 R3 묶음에 넣는다(PAYLOAD_EXTRA, --payload-check)")
    tr = train_points(PRODUCTS[product]["train"])
    return X.nearest_km(np.asarray(lat, float), np.asarray(lon, float), tr[:, 0], tr[:, 1]) if len(tr) else np.full(len(loc_id), np.inf)


def require_cell_inputs(a, pv, product="wei"):
    """작업 R3 본 실행의 입력 확인: 제품 값 표에 product 값이 있어야 한다(없으면 B:wei_raw·P1@wei 키 없이 'ok' 조각이 생긴다). 없으면 SystemExit."""
    if product not in pv:
        raise SystemExit(f"[셀 단위] {a.BASE_OUT / VALUES_FILE} 에 {product} 값이 없다. 작업 R3 묶음에 넣는다(PAYLOAD_EXTRA, --payload-check)")
    return True


def input_shas(a) -> dict:
    """셀 단위 조각 설정에 넣는 입력 표 sha256(제품 값 표, 셀 거리 표). 다른 표로 만든 조각을 --resume 이 다시 쓰지 않게 한다."""
    if getattr(a, "INSHA", None) is None:
        a.INSHA = {k: (XB.sha256_file(a.BASE_OUT / f) if (a.BASE_OUT / f).exists() else "")
                   for k, f in (("values_sha256", VALUES_FILE), ("celldist_sha256", CELLDIST_FILE))}
    return dict(a.INSHA)


class XGCellUnit(W.TUnit):
    """셀 단위 5 km 민감도 단위(h54.TUnit 하위 클래스). 추출·행렬은 LG 와 같고(h40.draw_cells, 원천 + 선택 라벨) R1(catboost_lo)과 D0(catboost_lo)를
    다시 적합한다. 같은 예측을 채점 셀 전체 저장소(<대상>|x)와 Wei 학습 지점 5 km 안 셀을 뺀 저장소(<대상>~c5|x)에 함께 쓴다."""

    def __init__(self, a, c, alias, keep, prod_fn, grid, draws, dry=False):
        self._keep = np.asarray(keep, bool)
        self._alias = alias
        self._prod_fn, self._grid, self._draws = prod_fn, list(grid), int(draws)
        super().__init__(a, c, alias, dry=dry, exp=TAG_CELL, variant="")

    def _make_xstores(self):
        keep = self._keep
        nm = f"{self._alias}~c5|{self.c.mode}"
        st = BlockStore(nm, self.c.split, self.c.blkB[keep], meta=dict(target=nm.split("|")[0], mode=self.c.mode, exp=TAG_CELL, cell_mask_km=CELL_D))
        return [(st, lambda y, p, keep=keep: (np.asarray(y)[keep], np.asarray(p)[keep]))]

    def run(self):
        c = self.c
        prod = self._prod_fn() if not self.dry else {}
        for k, pred in prod.items():
            if int(k[4]) == 0 and not str(k[0]).startswith("P1@"):
                self.add(k[0], "none", "cell", 0, 0, -1, 0.0, pred, np.nan, 0)
        for n, d in XB.cells_grid(self._grid, self._draws, self.nA, zero_n=True):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            E1, E2 = self.coefs(sel)
            self.trace("coef", sel, n=n, draw=str(d))
            self.add("P1", "none", "cell", n, d, -1, 0.0, E1 * c.sB, E1, nl, nb)
            for k, pred in prod.items():
                if k[0].startswith("P1@") and int(k[4]) == int(n) and int(k[5]) == int(d):
                    self.add(k[0], "none", "cell", n, d, -1, 0.0, pred, np.nan, nl, nb)
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            for seed in self.seeds:
                (g1,) = self.fit(XB.LO, "R1", lambda: self.rows_R(sel, a1A), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.emit("R1", XB.LO, n, d, seed, a1B, g1, E1, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl)
                (p0,) = self.fit(XB.LO, "D0", lambda: self.rows_D(sel), self.nsrc + nl, seed, [c.XB], n, d, sel)
                self.add("D0", XB.LO, "cell", n, d, seed, 1.0, p0, np.nan, nl, nb, flag=self.F.last_flag, nrow=self.nsrc + nl)
        return self


def cell_units(a):
    out = []
    for t in a.CELL_TARGETS:
        al = t
        spec, tgt, _ = XB.resolve(al)
        HA, D = XB.get_data(a54(a), al)
        keep, skip, info = W.split_plan(D, tgt, a.SPLITS)
        out += [(al, "x", int(sp)) for sp in keep]
    return out


def cell_cfg(a):
    """셀 단위 조각의 공통 설정. 제품 값 표와 셀 거리 표의 sha256 을 넣어 다른 표로 만든 조각을 --resume 이 다시 쓰지 않게 한다."""
    return XB.make_unit_cfg(EXP_ID, "", "", part="cell", grid=list(a.CELL_GRID), draws=int(a.DRAWS), seeds=list(a.SEEDS), cb_iters=int(a.cb_iters),
                            cell_mask_km=CELL_D, mask_product="wei", products=["wei"], lg_grid=list(LG_GRID), **input_shas(a))


def run_cell_unit(a, alias, mode, split, dry=False):
    """셀 단위 민감도 단위 하나. 조각 xgc__cpu__<대상>__x__s<분할>.
    본 실행(dry 아님, 스모크 아님)은 Wei 값과 셀 거리 표가 있어야 하며 없으면 적합 전에 멈춘다. 세기(dry)와 스모크는 없는 대로 진행한다."""
    t0 = time.time()
    strict = (not dry) and (not a.smoke)
    pv = load_products(a)
    if strict:
        require_cell_inputs(a, pv, "wei")
    c = XB.build_tctx(a54(a), alias, mode, split)
    D, src, A, B, tgt, _ = unit_indices(a, f"{alias}|{mode}", split)
    if not (len(B) == len(c.yB) and np.array_equal(D.df.y.values[B].astype(float), c.yB, equal_nan=True)):
        raise SystemExit(f"[셀 단위] {alias} s{split}: 채점 색인이 문맥과 다르다")
    dist = cell_dist(a, f"{alias}|{mode}", D.df.loc_id.values[B], D.df.lat.values[B], D.df.lon.values[B], strict=strict)
    keep = ~(dist <= CELL_D)
    prec = {}

    def prod_fn():
        if "wei" not in pv:
            return {}
        keys, rec = product_keys(D, src, A, B, "wei", pv, tgt, mode, split, a.CELL_GRID, a.DRAWS)
        prec.update(n_B_filled=int(rec.get("n_B_filled", 0)), rho_from=str(rec.get("rho_from", "")))
        if strict and not any(str(k[0]) == "B:wei_raw" for k in keys):
            raise SystemExit(f"[셀 단위] {alias} s{split}: B:wei_raw 키를 만들지 못했다(원천·A 쪽 Wei 값 확인)")
        return keys
    u = XGCellUnit(a54(a), c, alias, keep, prod_fn, a.CELL_GRID, a.DRAWS, dry=dry)
    if dry:
        u.run()
        return dict(alias=alias, split=int(split), n_fit=int(sum(u.F.n.values())), n_B=int(len(B)), n_B_masked=int((~keep).sum()))
    u.run()
    rows, stores, stats = u.finish()
    for r in rows:                                                         # 조각 runs 에는 RMSE 를 남기지 않는다(출력 제한)
        r.update(rmse_cm=np.nan, rmse_beq_cm=np.nan, bias_cm=np.nan)
    unit = dict(status=stats["status"], n_fit=stats["n_fit"], fail=stats["fail"], n_B=int(len(B)), n_B_masked=int((~keep).sum()),
                nb_B=int(len(np.unique(c.blkB))), nb_B_kept=int(len(np.unique(c.blkB[keep]))), dup_of=int(c.meta.get("dup_of", -1)),
                valid=bool(c.meta.get("valid", True)), sec=round(time.time() - t0, 1), product_available=bool("wei" in pv),
                n_B_wei_filled=int(prec.get("n_B_filled", -1)), deviation=DEVIATION, store_names=[st.target for st in stores], **input_shas(a))
    XB.write_shard(a.SHARDS, a.TAG, alias, mode, split, rows, stores, cell_cfg(a), unit, "", expected=a.SPLITS, code_file=__file__)
    return dict(alias=alias, split=int(split), status=unit["status"])


_WA = None


def _worker_init(argv):
    global _WA
    warnings.filterwarnings("ignore")
    _WA = parse_args(argv)
    if not on_rescale():                                                   # spawn 워커마다 로컬 상주 메모리 감시(계획 1절: 작업 하나 10 GB)
        rss_watchdog(RSS_LIMIT_GB)


def _worker_run(alias, mode, split):
    return run_cell_unit(_WA, alias, mode, split)


def cell_summary(a):
    """셀 단위 민감도(봉인 표용): XG-1w·2w·3w 를 셀 단위 5 km 저장소(<대상>~c5|x)와 전체 저장소에서 계산한다."""
    tms, _, _ = XB.load_tms(a.SHARDS, a.TAG, a.nboot, allow_mixed=a.allow_mixed_cfg)
    if not tms:
        return None
    rows = []
    for h in CONFIRM[:3]:
        gA = grp(h["gA"][0], h["gA"][1]); gB = grp(h["gB"][0], h["gB"][1])
        for sfx, lab in (("", "전체"), ("~c5", "셀 단위 5 km")):
            names = [f"{n.split('|')[0]}{sfx}|x" for n in h["pool"]]
            rr, _ = XB.contrast_pool(tms, names, gA, gB, h["id"], kind="same", registered=len(names))
            rows += [dict(r, hypothesis=h["id"], cell_mask=lab, role="민감도", deviation=DEVIATION) for r in rr]
    return XB.clean_rows(rows)


# ================================================================ 재현 관문
def gate_lgx_cells(a, names=None, max_files=None):
    """(g1) LGX lgxb cells.npz 에서 다시 만든 블록 SSE 와 같은 조각의 BlockStore. 예측·y 가 float32 로 저장되어 있으므로 셀마다 float32 반올림 상한
    (|e|·2δ + δ², δ = (|p| + |y|)·2⁻²³)의 합을 허용 폭으로 쓴다. 'pred' 성분 키만 대조한다(잔차 성분 g 는 λ 조합이라 따로 대조하지 않는다)."""
    rows = []
    files = sorted(SH_LGX.glob("lgxb__cpu__*__x__s*_cells.npz"))
    if names is not None:
        files = [f for f in files if f.name.split("__")[2] in {n.split("|")[0] for n in names}]
    for f in files[:max_files] if max_files else files:
        npz = Path(str(f).replace("_cells.npz", "_blocksse.npz"))
        if not npz.exists():
            continue
        st = list(H4.load_stores(npz).values())[0]
        with np.load(f, allow_pickle=False) as z:
            y = z["y"].astype(np.float64); blk = z["block"].astype(str); keys = [json.loads(s) for s in z["keys"]]; P = z["P"].astype(np.float64)
        ub, codes = np.unique(blk, return_inverse=True)
        if not np.array_equal(ub, st.blocks):
            rows.append(dict(item="g1", file=f.name, key="", ok=False, note="블록 집합 불일치")); continue
        for i, k in enumerate(keys):
            if k[-1] != "pred":
                continue
            m = [kk for kk in st.keys if list(kk[:7]) == list(k[:7])]
            if not m:
                continue
            s_new, c_new = H4.block_sse(y, P[i], codes, len(ub))
            s_ref, c_ref = st.get(m[0])
            e = np.abs(P[i] - y); dlt = (np.abs(P[i]) + np.abs(y)) * 2.0 ** -23
            tol = np.bincount(codes, weights=np.where(np.isfinite(e), 2 * e * dlt + dlt ** 2, 0), minlength=len(ub))
            diff = np.abs(s_new - s_ref)
            ok = bool(np.array_equal(c_new, c_ref) and np.all(diff <= tol * (1 + 1e-6) + 1e-12))
            rows.append(dict(item="g1", file=f.name, key=json.dumps(k), max_abs_dsse=float(diff.max()), max_tol=float(tol.max()), ok=ok, note=""))
    return pd.DataFrame(rows)


def gate_cci4(a, names=None):
    """(g1b) 해석식 제품 키 경로의 cci4 키가 LGX x9 의 cci 키(B:cci_raw, B:cci_aff, P0@cci, P1@cci)와 같은가(상대 1e-9)."""
    R = Rescore(a, products=["cci4"])
    rows = []
    for nm in names or [n for n in a.TARGETS if SOURCES[n][0] == "lg"]:
        by = R.load(nm)
        for sp, st in by.items():
            for k in st.keys:
                if not str(k[0]).endswith("@cci4") and k[0] not in ("B:cci4_raw", "B:cci4_aff"):
                    continue
                rk = (k[0].replace("cci4", "cci"),) + tuple(k[1:])
                if rk not in st:
                    continue
                s1, c1 = st.get(k); s2, c2 = st.get(rk)
                d = float(np.max(np.abs(s1 - s2))); rel = d / max(float(np.max(np.abs(s2))), 1e-300)
                rows.append(dict(item="g1b", target=nm, split=int(sp), key="|".join(str(v) for v in k), max_abs_dsse=d, max_rel=rel,
                                 ok=bool(np.array_equal(c1, c2) and rel <= 1e-9), note="해석식 cci4 = LGX x9 cci"))
    return pd.DataFrame(rows)


def gate_lg_curve(a, names=None):
    """(g2) 마스크 전 P0·R1(0.25) RMSE 가 LG 곡선 표와 같은가. 허용 오차는 1절 로컬–Rescale 행의 상대 1e-9 다(LG 곡선 표는 Rescale 조각에서
    만든 값이고 여기서는 같은 조각을 로컬에서 다시 집계한다. --gate-level 인자는 쓰지 않아 뺐다). 표에는 차의 절댓값과 통과 여부만 쓴다."""
    if not LG_CURVE.exists():
        return pd.DataFrame([dict(item="g2", ok=False, note="LG 곡선 표 없음")])
    if list(a.SPLITS) != list(XB.TRANSFER_SPLITS):
        return pd.DataFrame([dict(item="g2", ok=False, note="분할 1–5 전부가 아닐 때(스모크)는 대조하지 않는다")])
    cur = pd.read_csv(LG_CURVE, dtype=dict(target=str, mode=str, method=str, learner=str, alpha=str, placement=str))
    R = Rescore(a, products=[])
    rows = []
    for nm in names or [n for n in a.TARGETS if SOURCES[n][0] == "lg"]:
        tm, _ = R.tm(nm, "", "none")
        if tm is None:
            continue
        al, tgt, mode = resolve_alias(nm)
        for g in [H.P0_GRP] + [grp("R1", n) for n in (10, 40, 160, -1)]:
            q = cur[(cur.target == tgt) & (cur["mode"] == mode) & (cur.method == g[0]) & (cur.learner == g[1]) & (cur.alpha == g[2])
                    & (cur.placement == g[3]) & (cur.n.astype(int) == int(g[4])) & (np.isclose(cur.lam.astype(float), g[5]))]
            if not len(q):
                continue
            c, _ = XB.grp_rmse2(tm, g)
            col = "rmse_cm" if "rmse_cm" in q else ("rmse" if "rmse" in q else None)
            if col is None:
                continue
            ref = float(q[col].iloc[0])
            d = abs(c - ref)
            rows.append(dict(item="g2", target=nm, key="|".join(str(v) for v in g), abs_diff=d, ok=bool(d <= 1e-9 * max(abs(ref), 1.0)), note=""))
    return pd.DataFrame(rows)


def gate_md5(a):
    """(g3) 추출 단계가 기록한 Zenodo md5 대조 결과."""
    p = a.BASE_OUT / META_FILE
    if not p.exists():
        return pd.DataFrame([dict(item="g3", ok=False, note="xg_meta.json 없음(추출 전)")])
    m = json.loads(p.read_text()).get("extract", {}).get("products", {})
    rows = []
    for prod, rec in m.items():
        for z, v in (rec.get("md5") or {}).items():
            rows.append(dict(item="g3", product=prod, file=z, ok=bool(v.get("ok")), note="alt_products.md 9절 md5"))
    return pd.DataFrame(rows) if rows else pd.DataFrame([dict(item="g3", ok=False, note="md5 기록 없음")])


def run_gates(a, write=True, names=None):
    names = names or [n for n in a.TARGETS if SOURCES[n][0] == "lg"]
    parts = [gate_lgx_cells(a, names), gate_cci4(a, names), gate_lg_curve(a, names), gate_md5(a)]
    tab = pd.concat([p for p in parts if len(p)], ignore_index=True)
    summ = {}
    for it, g in tab.groupby("item"):
        summ[it] = dict(n=int(len(g)), n_fail=int((~g.ok.astype(bool)).sum()), passed=bool(len(g) > 0 and g.ok.astype(bool).all()))
        print(f"[관문 {it}] 키 {summ[it]['n']} · 실패 {summ[it]['n_fail']} · 통과 {summ[it]['passed']}", flush=True)
    if write:
        XB.check_out_dir(a.OUT)
        a.OUT.mkdir(parents=True, exist_ok=True)
        tab.to_csv(a.OUT / "xg_gate.csv", index=False)
        (a.OUT / "xg_gate_meta.json").write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), summary=summ), ensure_ascii=False, indent=1))
    return summ


# ================================================================ 세기(라벨 미사용)
def count_only(a):
    tot = 0
    for nm in a.TARGETS:
        fs = shard_files(nm, splits_of(nm, a))
        nb = []
        for sp, npz, uj in fs:
            with np.load(npz, allow_pickle=False) as z:
                nb.append(int(len(z["u0_blocks"])))
        print(f"  [세기] {nm}: 조각 {len(fs)} · 분할별 채점 블록 {nb}", flush=True)
        tot += len(fs)
    mp = a.BASE_OUT / MASK_FILE
    if mp.exists():
        t = pd.read_csv(mp, dtype=dict(target=str, product=str, block=str))
        g = t.groupby(["target", "product", "d_km"]).masked.agg(["sum", "size"]).reset_index()
        for tg, pr, d, n_m, n_b in g.values:
            print(f"  [세기] 마스크 {tg} {pr} {float(d):g} km: 가린 블록 {int(n_m)}/{int(n_b)}", flush=True)
    else:
        print("  [세기] 마스크 표 없음(--mask 로 만든다)", flush=True)
    pv = a.BASE_OUT / VALUES_FILE
    print(f"  [세기] 제품 값 표 {'있음' if pv.exists() else '없음(--extract 로 만든다)'}", flush=True)
    nfit = 0
    for u in cell_units(a):
        r = run_cell_unit(a, *u, dry=True)
        nfit += r["n_fit"]
        print(f"  [세기] 셀 단위 {u[0]} s{u[2]}: 적합 {r['n_fit']} · 채점 셀 {r['n_B']} · 5 km 안 {r['n_B_masked']}", flush=True)
    print(f"[세기] 재채점 조각 {tot} · 셀 단위 단위 {len(cell_units(a))} · 적합 {nfit}건(재채점은 적합 0)", flush=True)
    return dict(n_shards=tot, n_fit=nfit)


def rss_watchdog(limit_gb=10.0, poll_s=2.0):
    """로컬 메모리 상한(계획 1절: 작업 하나 10 GB 이하). xbatch_core.limit_memory 의 RLIMIT_AS(주소 공간 10 GB)는 CatBoost(catboost 600회, 깊이 6)가
    큰 가상 영역을 예약해 적합 안에서 멈추게 했다(로컬 스모크에서 확인, 구현 기록). systemd-run 은 이 서버에서 쓸 수 없다. 그래서 상주 메모리(VmRSS)를
    주기적으로 보고 상한을 넘으면 프로세스를 끝내는 감시 스레드를 쓴다. 반환 스레드(이 프로세스에 이미 있으면 None). 프로세스 번호로 기록하므로
    워커 프로세스는 자기 감시 스레드를 따로 띄운다."""
    import threading
    if getattr(rss_watchdog, "_pid", None) == os.getpid():
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
    rss_watchdog._pid = os.getpid()
    return t


# ================================================================ 묶음 점검(4절)
def payload_check(a, write=True) -> list:
    """작업 R3 묶음의 필수 입력(PAYLOAD_EXTRA: 제품 값 표, 셀 거리 표, xg_meta.json) 점검. xbatch_core.payload_manifest(extra=PAYLOAD_EXTRA)의
    missing 에 든 필수 입력 목록을 돌려준다. write 이면 묶음 정보(파일 위치 이탈 기록 포함)를 <out>/xg_payload_manifest.json 에 쓴다."""
    man = XB.payload_manifest(extra=PAYLOAD_EXTRA)
    miss = [f for f in PAYLOAD_EXTRA if f in set(man.get("missing", []))]
    man["module_inputs"] = dict(module="x_product_comparison", job="R3", required=list(PAYLOAD_EXTRA), missing=miss, file_location=LOCATION_NOTE)
    if write:
        XB.check_out_dir(a.BASE_OUT)
        a.BASE_OUT.mkdir(parents=True, exist_ok=True)
        (a.BASE_OUT / "xg_payload_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str))
    print(f"[묶음] 필수 입력 {len(PAYLOAD_EXTRA)}개 · 없음 {len(miss)}개" + (f": {', '.join(Path(f).name for f in miss)}" if miss else ""), flush=True)
    return miss


def preflight_cell(a):
    """작업 R3 본 실행 전 확인(적합 전에 멈춘다): 제품 값 표의 Wei 값과 셀 거리 표."""
    require_cell_inputs(a, load_products(a), "wei")
    if not (a.BASE_OUT / CELLDIST_FILE).exists():
        raise SystemExit(f"[셀 단위] {a.BASE_OUT / CELLDIST_FILE} 가 없다. 작업 R3 묶음에 넣는다(PAYLOAD_EXTRA, --payload-check)")
    return True


# ================================================================ 주 함수
def main(argv=None):
    a = parse_args(argv)
    if a.count_only or a.mask or a.extract or a.payload_check:
        mode = "count"
    elif a.smoke:
        mode = "smoke"
    elif a.summarize_only or a.gate_only or a.part == "rescore":
        mode = "summarize"
    else:
        mode = "run"
    XB.guard(mode, a.argv)
    if not on_rescale():
        rss_watchdog(RSS_LIMIT_GB)                                         # 로컬이면 허용 표지와 관계없이 상주 메모리 감시(구현 기록 7절)
    with XB.restricted_output():
        print(f"[XG] {NAME} · {DEVIATION} · 모드 {mode} · 스레드 {a.threads} · 재표집 {a.nboot}", flush=True)
        if a.payload_check:
            return 1 if payload_check(a) else 0
        if a.count_only:
            count_only(a)
            return 0
        if a.extract:
            wait_memory()
            extract(a)
            return 0
        if a.mask:
            write_masks(a)
            return 0
        if a.gate_only:
            wait_memory()
            run_gates(a)
            return 0
        if a.part == "cell" or a.smoke:
            if not a.summarize_only:
                wait_memory()
                if not a.smoke:
                    preflight_cell(a)
                units = [tuple([*a.shard.split(":")[:2], int(a.shard.split(":")[2])])] if a.shard else cell_units(a)
                if a.smoke:
                    units = units[:1]
                if a.resume:
                    units = [u for u in units if not XB.unit_state(a.SHARDS, a.TAG, u[0], u[1], u[2], cell_cfg(a))[0]]
                print(f"[XG 셀 단위] 단위 {len(units)}개", flush=True)
                done, failed = XB.execute(units, _worker_run, a.workers, a.threads, init=_worker_init, init_args=(a.argv,), permit=a.PERMIT)
                print(f"[XG 셀 단위] 완료 {len(done)} · 실패 {len(failed)} · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
                if a.no_summarize:
                    return 1 if failed else 0
        if a.smoke:
            run_gates(a, names=list(a.TARGETS))
        check_nboot(a)                                                     # 무거운 재채점 전에 확인
        wait_memory()
        out = summarize(a)
        print(f"[집계] 표 {len(out)}개 봉인 · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
