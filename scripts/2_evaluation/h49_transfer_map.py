"""H49 · 지도 과제(MAP) 예측기. 레나 x 1 km 격자의 ALT 예측, 90 % 구간, 외삽 표시. 계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.6–6.10.

지위와 시점(6.11)
  가설이 없는 서술 산출이다. 패널 방법은 6.7 의 규칙이 LG 집계, LGX L29, h39 집계, LGU-B2 판정으로 정한다. 그래서 실제 예측은 그 판정 표가
  모두 나온 뒤에만 실행한다. --allow-run 이 없거나 판정 표가 하나라도 없으면 거부한다. 판정 표를 읽기 전에는 합성 입력(--synthetic-test)으로만
  시험한다. --check-inputs 는 판정 표의 존재만 확인하고 내용을 열지 않는다. 본 실행(--allow-run)은 판정 표를 읽은 뒤 규칙에 필요한 대비 행이
  모두 있는지 먼저 확인하고(check_rule_inputs), 하나라도 없으면 선택을 기록하지 않고 중단한다(행 없음을 기본 방법으로 넘기지 않는다).

입력
  격자      data/processed/map_lena/lena_grid_x25_v1.csv.gz(scripts/1_data_prep/build_map_grid_lena_v1.py 산출, 6.4 QA 통과 필요)
  원천·라벨 h40 의 Data(load_base, v3 F4_direct), source_idx("Lena", "x")(레나 셀과 100 km 버퍼 제외), split_structure, half_split_blocks,
            draw_cells. h40 과 h42 는 importlib 로 읽고 고치지 않는다.
  판정 표(읽기 전용. 규칙 입력은 모두 있어야 본 실행을 한다)
    --lg-dir  lg_tests.csv                  L8 판정 행(test_id L8, item verdict)
    --lgx-dir lgx_lg_aux.csv, lgx_tests.csv R1 − P1(n 10, 전량), L29 의 기준선 − P0(n 0), 기준선 − P1(n 10, 전량), R1 − P1*, 학습기 축 L5
    --lgw-dir lgw_bundle.csv                h39 summarize 의 10,000회 재표집(AB5 'R1-P1|n10', AB9 'R1-P1|all')
              lgw_map_aux.csv               h49_map_contrasts.py 의 10,000회 재표집(h39 의 대비 엔진). R0 − P0(n 0)과 목록 안 기준선
                                            b ∈ {P1@soil, P1@ku, P1@ed, P1@cci, B:ens} 에 대한 R1 − b(n 10, 전량). 부록 6.7 은 R0 − P0 를
                                            h39 가 lgw_aux4.csv 에 계산한다고 정했으나 현재 h39 aux4_table 에 그 행이 없어 별도 파일로 둔다
              lgw_aux4.csv(선택)            있으면 함께 읽는다(h39 가 같은 대비를 더하면 두 표의 판정을 모두 본다)
              기본 폴더는 data/processed/wrapup(h39 의 실제 산출 폴더, WRAPUP 결과 절). 부록 3.6 의 data/processed/lgw 와 다르다
    --lgu-dir lgu_b_tests.csv               LGU-B2 행(test LGU-B2, verdict '전이' = 네 조건 모두 만족)
  학습기 사실(캡션 전용. 지도에는 쓰지 않는다)
    --lgt-dir lgt_tests.csv                 L32 'R1[T]-R1[C]|n…|lam0.25'(TabPFN 대 같은 컨텍스트 CatBoost). '등록 시점에 결과 존재(미열람)' 표지
    --lgf-dir lgf_tests.csv, lgfn_tests.csv LGF-F2 'R1[I]-R1[C]|n…|lam0.25'(TabICL), LGF-N3 'R1[<신경망>*]-R1[CB 또는 CBT]|n…'
    lgx_lg_aux.csv 의 L5 'R1[<학습기>]-R1[catboost_lo]|n…'. 세 묶음 가운데 없는 표가 있으면 본 실행을 중단한다. --allow-missing-gpu-tables 를
    주면 진행하고 메타와 캡션에 'LGT·LGF 학습기 사실 미포함'을 적는다.

6.7 의 규칙(결과 열람 전 고정, 코드는 rule_select)
  공통 판정 읽기: 대비 (A, B, n) 의 주 4지역 층화 평균 행(scope MEAN/MEAN4)을 모든 입력 표에서 찾는다. MEAN 행의 target(MEAN[…])에 주 4지역
    (Lena|x, Canada|x, Russia_W|x, Russia_E|x) 밖의 이름이 있으면 주 4지역 평균이 아니므로 쓰지 않는다. 풀 지역 수(pool 또는 n_ci_regions)는
    선택 기록에 옮긴다. '우세'는 찾은 행이 하나 이상이고 모두 '우세'일 때만이다(표마다 판정이 다르면 '재표집 의존'을 적고 우세로 보지 않는다.
    '행 없음' 자리 표시 행은 행이 없는 것으로 보고 '판정 불가' 행은 다른 판정으로 센다).
    본 실행의 입력 확인(check_rule_inputs): R1 − P1(n 10, 전량)의 층화 평균 행이 lgw_bundle 과 lgx_lg_aux 각각에, R0 − P0(n 0)의 층화 평균
    행과 레나 행이, P1* 가 있으면 R1 − P1* 의 층화 평균 행과 레나 행이, P* 가 있으면 P* − P0 의 레나 행이 있어야 한다. L29 의 세 묶음
    (n 0 − P0, n 10 − P1, 전량 − P1)마다 lgx_tests 에 기준선 행이 하나 이상 있어야 하고, 표마다 해석한 대비 행이 0 이면 안 된다.
    lgw 표에 재표집 제한(nboot_capped 또는 nboot < 10,000) 행이 있어도 중단한다. 표마다 해석한 행 수는 선택 기록(input_check)에 적는다.
    '레나 행이 열세가 아니다'는 레나 지역 행(target Lena|x)이 하나 이상 있고 모두 4분 판정(우세·동등·미결정·열세)이 나왔고 열세가 없을 때만이다.
  격자에서 계산할 수 있는 기준선(목록): n 0 은 B:s_aff, B:cci_aff, B:cci_raw, P0@cci, P0@soil, B:ku_raw, B:ed_raw, P0@ku, P0@ed, B:ens,
    라벨 n개는 P1@cci, P1@soil, P1@ku, P1@ed, B:ens. 목록 밖은 P0@tddm, P1@tddm(연도 정합 √TDD, 정적 격자에서 정의되지 않는다).
  P*, P1*: h42 L29 와 같은 절차(기준선 − P0(또는 P1)이 우세인 것 가운데 층화 평균 점 추정이 가장 낮은 것)를 lgx_tests.csv 의 행으로 재현한다.
    목록 밖이면 우세이고 레나 행이 열세가 아닌 목록 안 기준선 가운데 점 추정이 가장 낮은 것으로 대체하고, 없으면 P0(또는 P1)이다.
  (a) R0 − P0(n 0) 우세 · 레나 행 열세 아님 → R0 후보. P* 가 있고 레나 행의 P* − P0 가 열세 아님 → P* 후보. 둘 다면 층화 평균 점 추정이
      낮은 쪽, 하나면 그것, 없으면 P0.
  (b) Pr = P1*(n 10) 또는 P1. R1 − P1(n 10) 우세, P1* 가 있으면 R1 − P1* 도 우세, 두 대비의 레나 행 열세 아님 → R1. 그 밖은 Pr.
  (c) 같은 규칙을 전량(n = −1)으로. L8 이 기각이면 캡션 표지를 단다.
  (d) (a) 에서 고른 방법의 구간. 정규화기는 LGU-B2 가 '전이'이면 nflow σ(--sigma-file 필요, 없으면 중단), 아니면 const.
  공통: 지도는 catboost_lo 와 해석식만 쓴다. GPU 학습기(LG L5, LGT L32, LGF-F2, LGF-N3)가 CatBoost 대비 층화 평균 우세여도 지도에 쓰지 않고
    학습기, n, 대비 값을 캡션 사실로 적는다(기준 λ 0.25 또는 λ 표기 없는 행. LGT 행에는 '등록 시점에 결과 존재(미열람)' 표지).
  예측 지도를 본 뒤 방법을 바꾸지 않는다(선택 기록은 예측 전에 메타에 쓴다).

방법(h40·h42 와 같은 식. s = e5_sqrt_tdd, κ = 10, λ = 0.25, CatBoost = h40.cb_fit(200회, 깊이 3, 학습률 0.05, l2 3), seed 0·1 평균)
  P0 E0·s | P1 E_n·s, E_n = (n·E_ls + κ·E0)/(n + κ) | R0 E0·s + λ·g(원천 잔차) | R1 E_n·s + λ·g(원천 E0 잔차 ∪ 레나 라벨 E_n 잔차)
  P0@b c0·b, P1@b 수축 c1·b(b 의 비유한·0 이하 값은 ρ·s, ρ = 원천 b/s 중앙값: h42.anchor_fill) | B:<b>_raw b | B:s_aff a + b·s(원천 최소제곱)
  B:cci_aff a + b·cci(원천 최소제곱, 채운 cci) | B:ens (E·s + c_ku·ku + c_cci·cci)/3 (n > 0 은 수축 계수)
  라벨: (b) h40.draw_cells("Lena", "x", s*, 10, 0, |A|) 의 A 행(s* = 유효 분할 가운데 가장 작은 split_seed), (c) 레나 F4_direct 전체(A ∪ B).

구간(6.6 (d)): 보정 집단 = 원천에서 로그 비가 정의되는 셀(y, s 유한·양수)이 30개 이상인 macro 지역(부록 6.6 의 30. LGU h45 는 20 이며
  두 집합을 메타에 함께 적는다). 지역 k 마다 k 를 뺀 원천으로 방법을 적합해 k 의 셀을 예측하고 점수 |log(y / max(ŷ, 1 cm))| 의 지역 등가중
  합동 CDF 0.9 분위 q 를 구한다(lgu_common.hier_quantiles). 격자 구간 = [ŷ·e^(−q·σ), ŷ·e^(q·σ)], const 는 σ = 1.
외삽 표시(6.6 (e)): x25 의 연속 24열(cci_valid 제외) 각각이 원천 셀의 0.5–99.5 백분위 밖인 열 수 + 결측 열 수(6.5), 범주 0, 1, 2, 3 이상.
회색 육지(6.5): 격자의 land·gray·gray_reason 열로 육지 셀 수, 회색 육지 셀 수와 비율, 첫 사유별 수를 계산해 캡션 사실(caption.gray_land)에 둔다.

산출(--out-dir, 기본 data/processed/map_lena. 부록 6.10 의 data/processed/map/ 대신 워크플로 지시의 폴더)
  lena_pred_v1.csv.gz      셀별 패널 예측, 기본 방법 예측, 90 % 구간, 차이 (c) − (a), 외삽 열 수, 마스크
  lena_pred_v1_meta.json   규칙 적용 기록(입력 판정 행과 선택), 계수, q, 보정 집단, 외삽 기준, 캡션 사실, 입력·코드 해시, 적합 수, 최대 RSS
  (합성 시험은 --synthetic-test <폴더> 에 같은 이름으로 쓴다. 그 폴더는 본 산출 폴더(--out-dir, data/processed/map_lena) 안일 수 없다)
  출력 보호: data/processed/ 의 lg, lgx, lgt, lgd, lgu, lgf, ext_labels 와 results/ 안에는 쓰지 않는다.

실행(ROOT, 공유 서버 규칙: CPU 스레드 4 이하, nice 10, GPU 없음). 스크립트로 실행하면 OMP·OpenBLAS·MKL·NUMEXPR 스레드를 --threads 로 둔다
  판정 표 존재 확인: python3 scripts/2_evaluation/h49_transfer_map.py --check-inputs --lgw-dir data/processed/wrapup
  합성 시험:        nice -n 10 python3 scripts/2_evaluation/h49_transfer_map.py --synthetic-test <폴더> --threads 2
  누락 대비 집계:    nice -n 10 python3 scripts/2_evaluation/h49_map_contrasts.py --allow-run --allow-local --threads 2 --out-dir data/processed/wrapup
  본 실행(판정 뒤):  nice -n 10 python3 scripts/2_evaluation/h49_transfer_map.py --allow-run --threads 4 --lgw-dir data/processed/wrapup
"""
from __future__ import annotations

import os
import sys


def _peek_threads(default="2"):
    av = sys.argv
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return av[i + 1]
        if v.startswith("--threads="):
            return v.split("=", 1)[1]
    return default


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _v in THREAD_VARS:
    if __name__ == "__main__":                                           # 스크립트 실행: --threads 로 명시한다(셸의 더 큰 값을 따르지 않는다)
        os.environ[_v] = str(_peek_threads())
    else:                                                                # 다른 모듈이 읽을 때: 부른 쪽의 설정을 따른다
        os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""                                  # GPU 를 쓰지 않는다(6.9)

import argparse                                                                                         # noqa: E402
import hashlib                                                                                          # noqa: E402
import importlib.util                                                                                   # noqa: E402
import json                                                                                             # noqa: E402
import re                                                                                               # noqa: E402
import resource                                                                                         # noqa: E402
import time                                                                                             # noqa: E402
from dataclasses import dataclass, field                                                                # noqa: E402
from pathlib import Path                                                                                # noqa: E402
from types import SimpleNamespace                                                                       # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
for _p in (str(ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------- 고정 설계값(부록 6.6–6.7, LG §3)
KAPPA = 10.0
LAM = 0.25
SEEDS = (0, 1)
CB_ITERS = 200
MAX_THREADS = 4                                                   # 6.9
Q_LEVEL = 0.9
MIN_CAL_CELLS = 30                                                # 6.6 (d)
MIN_CAL_CELLS_LGU = 20                                            # LGU h45 의 값(메타에 함께 적는다)
Y_FLOOR_CM = 1.0                                                  # 점수·구간의 ŷ 하한(h42.log_score 와 같다)
EXTRAP_PCT = (0.5, 99.5)
N_PANEL_B = 10
N_ALL = -1
TARGET, MODE = "Lena", "x"
ANCHOR_COL = dict(soil="e5_sqrt_tdd_soil", ku="p4_ku", ed="p2_edaphic", cci="cci_alt")
L29_N0_H42 = ("B:s_aff", "B:ed_raw", "B:ku_raw", "B:cci_raw", "B:cci_aff", "B:ens", "P0@soil", "P0@ku", "P0@ed", "P0@cci", "P0@tddm")
L29_LN_H42 = ("B:ens", "P1@soil", "P1@ku", "P1@ed", "P1@cci", "P1@tddm")
GRID_LIST_N0 = ("B:s_aff", "B:cci_aff", "B:cci_raw", "P0@cci", "P0@soil", "B:ku_raw", "B:ed_raw", "P0@ku", "P0@ed", "B:ens")
GRID_LIST_LN = ("P1@cci", "P1@soil", "P1@ku", "P1@ed", "B:ens")
VALID4 = ("우세", "동등", "미결정", "열세")
MAIN4_T = ("Lena|x", "Canada|x", "Russia_W|x", "Russia_E|x")          # 주 4지역 층화 평균의 지역
NBOOT_MIN = 10000                                                 # lgw 표의 재표집 횟수(부록 0.5)
# 규칙 입력(모두 있어야 본 실행), 선택 입력(있으면 읽는다), 학습기 사실(캡션 전용)
RULE_FILES = dict(lg=("lg_tests.csv",), lgx=("lgx_lg_aux.csv", "lgx_tests.csv"), lgw=("lgw_bundle.csv", "lgw_map_aux.csv"), lgu=("lgu_b_tests.csv",))
OPTIONAL_FILES = dict(lgw=("lgw_aux4.csv",))
GPU_FILES = dict(lgt=("lgt_tests.csv",), lgf=("lgf_tests.csv", "lgfn_tests.csv"))
BOOK_SOURCES = ("lgx_lg_aux", "lgx_tests", "lgw_bundle", "lgw_map_aux", "lgw_aux4")
# 학습기 대비의 CatBoost 쪽 이름(h42 L5 catboost_lo, h43·h47 C = catboost_ctx, h48 CB = catboost_lo·CBT = catboost_tuned_loc)
CB_NAMES = ("C", "CB", "CBT", "catboost", "catboost_lo", "catboost_ctx", "catboost_tuned", "catboost_tuned_loc")
LEARNER_EN = dict(T="TabPFN", I="TabICL", mlp="MLP", tabm="TabM", ftt="FT-Transformer", realmlp="RealMLP", cfm="CFM", ddpm="DDPM",
                  nflow="normalising flow", ridge="ridge")
GPU_SOURCES = (("lgx_lg_aux", "LG L5", ""), ("lgt_tests", "LGT L32", "등록 시점에 결과 존재(미열람)"), ("lgf_tests", "LGF-F2", ""),
               ("lgfn_tests", "LGF-N3", ""))
PROTECT = ("lg", "lgx", "lgt", "lgd", "lgu", "lgf", "ext_labels")    # data/processed/ 아래의 실험 산출 폴더(쓰지 않는다)
MAP_B = ("P1@soil", "P1@ku", "P1@ed", "P1@cci", "B:ens")               # h49_map_contrasts 가 R1 − b 를 계산하는 목록 안 재보정 기준선
SCENARIOS = ("baseline", "all_ml", "pstar", "replace1", "b2")


def log(*a):
    print(*a, flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_info(path) -> dict:
    p = Path(path)
    rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    return dict(path=rel, bytes=int(p.stat().st_size), sha256=sha256_file(p)) if p.exists() else dict(path=rel, exists=False)


def load_module(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def h40():
    return load_module("h40_label_grid", "scripts/3_deep_learning/h40_label_grid.py")


def h42():
    h40()                                                             # h42 는 같은 이름으로 등록된 h40 을 쓴다
    return load_module("h42_label_grid_ext", "scripts/3_deep_learning/h42_label_grid_ext.py")


# ================================================================ 1) 판정 표 읽기와 6.7 규칙
def parse_label(label: str):
    """'R1-P1|n10' → ('R1', 'P1', 10). 'R1-P0|all' → n = −1. 부가 표지('|kuok' 등)가 있으면 None."""
    parts = str(label).split("|")
    if len(parts) != 2 or "-" not in parts[0]:
        return None
    a, b = parts[0].rsplit("-", 1)
    t = parts[1].strip()
    if t == "all":
        n = N_ALL
    elif t.startswith("n"):
        try:
            n = int(t[1:])
        except ValueError:
            return None
    else:
        return None
    return a.strip(), b.strip(), int(n)


def _scope(v) -> str:
    v = str(v)
    return "MEAN4" if v in ("MEAN", "MEAN4") else v


def mean_regions(target) -> list:
    """'MEAN[Lena|x,Canada|x]' → ['Lena|x', 'Canada|x']. 형식이 다르면 빈 목록."""
    m = re.fullmatch(r"MEAN\[(.*)\]", str(target).strip())
    return [v.strip() for v in m.group(1).split(",") if v.strip()] if m else []


def _pool_text(r: dict) -> str:
    """풀 지역 수 표기. h39 는 pool 열, h42 는 n_ci_regions·n_regions_target 열이다."""
    p = r.get("pool")
    if isinstance(p, str) and p.strip() and p.strip().lower() != "nan":
        return p.strip()
    try:
        k = int(float(r.get("n_ci_regions")))
    except (TypeError, ValueError):
        return ""
    try:
        N = int(float(r.get("n_regions_target")))
    except (TypeError, ValueError):
        N = len(MAIN4_T)
    return f"지역 {k}/{N}" if k >= N else f"부분(지역 {k}/{N})"


BOOK_COLS = ["source", "A", "B", "n", "scope", "target", "verdict4", "delta", "test_id", "contrast", "pool"]


class VerdictBook:
    """판정 표의 대비 행을 (A, B, n, scope, target) 으로 모은다. 표 이름을 source 로 적는다.
    stats[표] = 읽은 행 수, 해석한 대비 행 수, 해석하지 못한 라벨 수, 주 4지역 밖 지역이 든 MEAN 행 수(scope 를 'MEAN_other' 로 바꿔 쓰지 않는다)."""

    def __init__(self, tables: dict):
        rows, self.stats = [], {}
        for src, df in tables.items():
            if df is None:
                self.stats[src] = dict(present=False)
                continue
            st = dict(present=True, n_rows=int(len(df)), n_parsed=0, n_unparsed_label=0, n_mean_main4=0, n_mean_not_main4=0)
            self.stats[src] = st
            if not len(df):
                continue
            need = {"contrast", "scope", "target", "verdict4", "delta"}
            miss = need - set(df.columns)
            if miss:
                raise ValueError(f"{src}: 필요한 열이 없다 {sorted(miss)}")
            for r in df.to_dict("records"):
                k = parse_label(r["contrast"])
                if k is None:
                    st["n_unparsed_label"] += 1
                    continue
                sc, tgt = _scope(r["scope"]), str(r["target"])
                if sc == "MEAN4":
                    regs = mean_regions(tgt)
                    if regs and not set(regs) <= set(MAIN4_T):
                        sc = "MEAN_other"; st["n_mean_not_main4"] += 1
                    else:
                        st["n_mean_main4"] += 1
                st["n_parsed"] += 1
                rows.append(dict(source=src, A=k[0], B=k[1], n=k[2], scope=sc, target=tgt, verdict4=str(r["verdict4"]),
                                 delta=float(r["delta"]) if pd.notna(r["delta"]) else np.nan, test_id=str(r.get("test_id", "")),
                                 contrast=str(r["contrast"]), pool=_pool_text(r) if sc == "MEAN4" else ""))
        self.rows = pd.DataFrame(rows, columns=BOOK_COLS)

    def find(self, A, B, n, scope="MEAN4", target=None, sources=None) -> pd.DataFrame:
        """자리 표시 행(verdict4 = '행 없음', h39·h42 의 missing_row)은 행이 없는 것으로 본다."""
        r = self.rows
        m = (r.A == A) & (r.B == B) & (r.n == int(n)) & (r.scope == scope) & (r.verdict4 != "행 없음")
        if target is not None:
            m &= r.target.str.startswith(target)
        if sources is not None:
            m &= r.source.isin(list(sources))
        return r[m]


def dominant(book: VerdictBook, A, B, n) -> dict:
    """층화 평균 우세(찾은 행이 모두 우세)."""
    f = book.find(A, B, n, "MEAN4")
    rec = f[["source", "verdict4", "delta", "pool"]].to_dict("records")
    if not len(f):
        return dict(ok=False, note="행 없음", rows=rec, delta=np.nan)
    vs = set(f.verdict4)
    ok = vs == {"우세"}
    note = "" if len(vs) == 1 else "재표집 의존(표마다 판정이 다르다)"
    d = f.delta.dropna()
    return dict(ok=bool(ok), note=note, rows=rec, delta=float(d.mean()) if len(d) else np.nan,
                delta_spread=float(d.max() - d.min()) if len(d) > 1 else 0.0)


def lena_not_inferior(book: VerdictBook, A, B, n) -> dict:
    f = book.find(A, B, n, "region", target=f"{TARGET}|")
    rec = f[["source", "verdict4", "delta"]].to_dict("records")
    if not len(f):
        return dict(ok=False, note="레나 행 없음", rows=rec)
    vs = set(f.verdict4)
    if not vs <= set(VALID4):
        return dict(ok=False, note="레나 행 판정 불가", rows=rec)
    return dict(ok="열세" not in vs, note="레나 행 열세" if "열세" in vs else "", rows=rec)


def l29_pick(book: VerdictBook, n: int, base: str) -> dict:
    """h42 L29 의 P*(n = 0, base P0) 또는 P1*(n > 0 또는 전량, base P1)와 목록 안 대체."""
    cands = L29_N0_H42 if n == 0 else L29_LN_H42
    in_list = GRID_LIST_N0 if n == 0 else GRID_LIST_LN
    rows, dom = [], []
    for m in cands:
        f = book.find(m, base, n, "MEAN4", sources=["lgx_tests"])
        v = f.verdict4.iloc[0] if len(f) else "행 없음"
        d = float(f.delta.iloc[0]) if len(f) else np.nan
        rows.append(dict(baseline=m, verdict4=v, delta=d))
        if v == "우세" and np.isfinite(d):
            dom.append((m, d))
    out = dict(n=n, base=base, rows=rows, h42_pick=None, pick=None, replaced=False, note="")
    if not dom:
        out["note"] = f"L29 가 {base} 보다 우세한 기준선을 정하지 않았다"
        return out
    h42_pick = min(dom, key=lambda v: v[1])[0]
    out["h42_pick"] = h42_pick
    if h42_pick in in_list:
        out["pick"] = h42_pick
        return out
    alt = []
    for m, d in dom:
        if m in in_list:
            li = lena_not_inferior(book, m, base, n)
            if li["ok"]:
                alt.append((m, d))
    out["replaced"] = True
    if alt:
        out["pick"] = min(alt, key=lambda v: v[1])[0]
        out["note"] = f"L29 의 {h42_pick} 는 격자에서 계산할 수 없어 목록 안 {out['pick']} 로 대체했다"
    else:
        out["note"] = f"L29 의 {h42_pick} 는 격자에서 계산할 수 없고 대체할 목록 안 기준선이 없어 {base} 를 쓴다"
    return out


def lgu_b2(df: pd.DataFrame | None) -> dict:
    if df is None or not len(df) or "test" not in df.columns:
        return dict(found=False, verdict="행 없음", transfer=False)
    f = df[df.test == "LGU-B2"]
    if not len(f):
        return dict(found=False, verdict="행 없음", transfer=False)
    v = str(f.verdict.iloc[0])
    return dict(found=True, verdict=v, transfer=(v == "전이"),
                detail={k: (None if pd.isna(f[k].iloc[0]) else f[k].iloc[0]) for k in ("delta", "ci_hi", "ci_hi_beq", "coverage_pool", "n_neg_regions",
                                                                                          "seeds_ok") if k in f.columns})


def l8_verdict(df: pd.DataFrame | None) -> dict:
    if df is None or not len(df) or "test_id" not in df.columns:
        return dict(found=False, verdict="행 없음", rejected=False)
    f = df[(df.test_id == "L8") & (df.get("item", pd.Series("", index=df.index)) == "verdict")]
    if not len(f):
        return dict(found=False, verdict="행 없음", rejected=False)
    v = str(f.verdict.iloc[0])
    return dict(found=True, verdict=v, rejected="기각" in v, stat=str(f.get("stat", pd.Series([""])).iloc[0]))


_LR_LABEL = re.compile(r"^(?P<ma>[A-Za-z0-9@_:]+)\[(?P<la>[^\]]+)\]-(?P<mb>[A-Za-z0-9@_:]+)\[(?P<lb>[^\]]+)\]"
                       r"\|n(?P<n>-?\d+|all)(?:\|lam(?P<lam>[0-9.]+))?$")


def parse_learner_label(label):
    """학습기 대비 라벨. 'R1[T]-R1[C]|n10|lam0.25' → dict(ma R1, la T, mb R1, lb C, n 10, lam 0.25). λ 표기가 없으면 lam None.
    그 밖의 부가 표지('|nondefault' 등)가 있으면 None."""
    m = _LR_LABEL.match(str(label).strip())
    if not m:
        return None
    return dict(ma=m["ma"], la=m["la"], mb=m["mb"], lb=m["lb"], n=N_ALL if m["n"] in ("all", "-1") else int(m["n"]),
                lam=float(m["lam"]) if m["lam"] else None)


def _is_cb(name: str) -> bool:
    b = str(name).rstrip("*")
    return b in CB_NAMES or b.startswith("catboost")


def learner_en(name: str) -> str:
    b = str(name).rstrip("*")
    base = b.split("_tuned")[0]
    txt = LEARNER_EN.get(base, base)
    return txt + (" (tuned)" if (str(name).endswith("*") or "_tuned" in b) else "")


def gpu_learner_notes(tables: dict) -> dict:
    """GPU 학습기가 CatBoost 보다 층화 평균 우세인 R1 대비(캡션 사실. 지도에는 쓰지 않는다, 6.7 공통).
    대상: lgx_lg_aux 의 L5, lgt_tests 의 L32, lgf_tests 의 LGF-F2, lgfn_tests 의 LGF-N3. 기준 λ(0.25) 행과 λ 표기가 없는 행만 본다.
    반환 dict(notes = [...], status = {표: 상태}). LGT 행에는 '등록 시점에 결과 존재(미열람)' 표지를 붙인다."""
    notes, status = [], {}
    for src, exp, mark in GPU_SOURCES:
        df = tables.get(src)
        if df is None:
            status[src] = "없음"
            continue
        if not len(df) or not {"contrast", "scope", "verdict4", "delta"} <= set(df.columns):
            status[src] = "열 없음 또는 빈 표"
            continue
        n_pair = 0
        for r in df.to_dict("records"):
            if _scope(r.get("scope")) != "MEAN4":
                continue
            regs = mean_regions(r.get("target", ""))
            if regs and not set(regs) <= set(MAIN4_T):
                continue
            k = parse_learner_label(r["contrast"])
            if k is None or k["ma"] != "R1" or k["mb"] != "R1" or not _is_cb(k["lb"]) or _is_cb(k["la"]):
                continue
            if k["lam"] is not None and abs(k["lam"] - LAM) > 1e-9:
                continue
            n_pair += 1
            if str(r["verdict4"]) == "우세":
                notes.append(dict(learner=k["la"], learner_en=learner_en(k["la"]), comparator=k["lb"], contrast=str(r["contrast"]), n=int(k["n"]),
                                  delta=float(r["delta"]) if pd.notna(r["delta"]) else np.nan, source=src, experiment=exp, mark=mark,
                                  test_id=str(r.get("test_id", "")), pool=_pool_text(r)))
        status[src] = f"읽음(학습기 대비 층화 평균 행 {n_pair})"
    return dict(notes=notes, status=status)


def rule_select(book: VerdictBook, b2: dict, l8: dict, gpu: dict | None = None) -> dict:
    """6.7 의 선택. 반환 dict(panels = {a, b, c, d}, 기록)."""
    rec = {}
    # (a)
    r0_dom = dominant(book, "R0", "P0", 0)
    r0_len = lena_not_inferior(book, "R0", "P0", 0)
    ps = l29_pick(book, 0, "P0")
    ps_len = lena_not_inferior(book, ps["pick"], "P0", 0) if ps["pick"] else dict(ok=False, note="P* 없음", rows=[])
    ps_dom = dominant(book, ps["pick"], "P0", 0) if ps["pick"] else dict(ok=False, note="P* 없음", rows=[], delta=np.nan)
    cand = []
    if r0_dom["ok"] and r0_len["ok"]:
        cand.append(("R0", r0_dom["delta"]))
    if ps["pick"] and ps_len["ok"]:
        cand.append((ps["pick"], ps_dom["delta"]))
    if len(cand) == 2:
        a_m = min(cand, key=lambda v: (v[1] if np.isfinite(v[1]) else np.inf))[0]
        why = f"R0 와 P*({ps['pick']}) 모두 조건을 만족해 층화 평균 점 추정이 낮은 {a_m}"
    elif cand:
        a_m = cand[0][0]
        why = "R0 − P0(n 0) 우세, 레나 행 열세 아님" if a_m == "R0" else f"L29 P* = {a_m}, 레나 행 열세 아님"
    else:
        a_m = "P0"
        why = "R0 와 P* 가 조건을 만족하지 않아 기본 방법 P0"
    rec["a"] = dict(method=a_m, default="P0", why=why, R0_minus_P0=dict(dominant=r0_dom, lena=r0_len), l29=ps, Pstar_lena=ps_len,
                    Pstar_dominant=ps_dom)

    def recal(n, default_why):
        p1s = l29_pick(book, n, "P1")
        pr = p1s["pick"] or "P1"
        c1d, c1l = dominant(book, "R1", "P1", n), lena_not_inferior(book, "R1", "P1", n)
        if p1s["pick"]:
            c2d, c2l = dominant(book, "R1", p1s["pick"], n), lena_not_inferior(book, "R1", p1s["pick"], n)
        else:
            c2d, c2l = dict(ok=True, note="P1* 없음(조건 해당 없음)", rows=[]), dict(ok=True, note="P1* 없음", rows=[])
        ok = c1d["ok"] and c1l["ok"] and c2d["ok"] and c2l["ok"]
        m = "R1" if ok else pr
        why = ("R1 − P1 우세" + (f", R1 − {p1s['pick']} 우세" if p1s["pick"] else "") + ", 레나 행 열세 아님") if ok else \
            f"R1 조건 불충족({'; '.join(x for x in (c1d['note'], c1l['note'], c2d.get('note', ''), c2l.get('note', '')) if x) or '우세 아님'}). {default_why} {pr}"
        return dict(method=m, default="P1" if n != N_ALL else "R1", Pr=pr, why=why, l29=p1s, R1_minus_P1=dict(dominant=c1d, lena=c1l),
                    R1_minus_P1star=dict(dominant=c2d, lena=c2l))
    rec["b"] = recal(N_PANEL_B, "재보정 기준")
    rec["c"] = recal(N_ALL, "재보정 기준")
    rec["c"]["L8"] = l8
    rec["c"]["caption_L8"] = bool(l8.get("rejected"))
    rec["d"] = dict(method=a_m, normalizer="nflow" if b2.get("transfer") else "const", B2=b2)
    rec["placement"] = ("main Fig 6c: (a), (d), (e); SI: (b), (c), (f)" if b2.get("transfer")
                        else "SI 1장(전 패널). 캡션: 구간 폭은 예측값에 비례한다")
    gpu = gpu if gpu is not None else dict(notes=[], status={})
    rec["gpu_notes"] = gpu["notes"]
    rec["gpu_status"] = gpu["status"]
    rec["panels"] = {k: rec[k]["method"] for k in ("a", "b", "c", "d")}
    return rec


def check_rule_inputs(book: VerdictBook, sel: dict, tables: dict, b2: dict, l8: dict) -> list:
    """본 실행 전의 입력 확인(6.7). 규칙이 요구하는 대비 행이 없으면 그 사실을 문장으로 돌려준다(빈 목록 = 통과).
    행이 없는 대비를 기본 방법으로 조용히 넘기지 않기 위해서다."""
    probs = []
    lena = f"{TARGET}|"

    def need(A, B, n, scope="MEAN4", target=None, sources=None):
        if not len(book.find(A, B, n, scope, target=target, sources=sources)):
            where = f" ({', '.join(sources)})" if sources else ""
            what = "레나 행" if scope == "region" else "층화 평균 행"
            probs.append(f"{A} − {B}(n {n}) {what} 없음{where}")
    for src in ("lgx_lg_aux", "lgx_tests", "lgw_bundle", "lgw_map_aux"):
        st = book.stats.get(src, {})
        if not st.get("present"):
            probs.append(f"{src}.csv 없음")
        elif st.get("n_parsed", 0) == 0:
            probs.append(f"{src}.csv: 해석한 대비 행 0(행 {st.get('n_rows', 0)}, 해석 못 한 라벨 {st.get('n_unparsed_label', 0)})")
    for n in (N_PANEL_B, N_ALL):                                           # (b)(c) R1 − P1: AB5·AB9 와 LG L4
        for s in ("lgw_bundle", "lgx_lg_aux"):
            need("R1", "P1", n, sources=[s])
        need("R1", "P1", n, "region", lena)
    need("R0", "P0", 0)                                                   # (a) R0 − P0(n 0)
    need("R0", "P0", 0, "region", lena)
    for n, base, cands in ((0, "P0", L29_N0_H42), (N_PANEL_B, "P1", L29_LN_H42), (N_ALL, "P1", L29_LN_H42)):
        if not any(len(book.find(m, base, n, "MEAN4", sources=["lgx_tests"])) for m in cands):
            probs.append(f"L29 기준선 − {base}(n {n}): lgx_tests 에 층화 평균 행이 하나도 없다")
    ps = sel["a"]["l29"].get("pick")
    if ps:
        need(ps, "P0", 0, "region", lena)
    for p, n in (("b", N_PANEL_B), ("c", N_ALL)):
        p1s = sel[p]["l29"].get("pick")
        if p1s:
            need("R1", p1s, n)
            need("R1", p1s, n, "region", lena)
    for s in ("lgw_bundle", "lgw_map_aux", "lgw_aux4"):
        df = tables.get(s)
        if df is None or not len(df):
            continue
        if "nboot_capped" in df.columns and df.nboot_capped.astype(str).str.strip().str.lower().isin(["true", "1"]).any():
            probs.append(f"{s}.csv: 재표집 제한 행(nboot_capped)이 있다")
        if "nboot" in df.columns and (pd.to_numeric(df.nboot, errors="coerce") < NBOOT_MIN).any():
            probs.append(f"{s}.csv: nboot < {NBOOT_MIN:,} 행이 있다")
    if not b2.get("found"):
        probs.append("LGU-B2 행 없음(lgu_b_tests.csv)")
    if not l8.get("found"):
        probs.append("L8 판정 행 없음(lg_tests.csv)")
    return probs


def _all_files():
    for grp in (RULE_FILES, OPTIONAL_FILES, GPU_FILES):
        for key, files in grp.items():
            for f in files:
                yield grp, key, f


def read_verdict_tables(dirs: dict) -> tuple[dict, dict]:
    """판정 표를 읽는다(본 실행에서만 호출). 반환 (tables, 파일 정보)."""
    tabs, info = {}, {}
    for _, key, f in _all_files():
        p = Path(dirs[key]) / f
        info[f] = file_info(p)
        tabs[f.replace(".csv", "")] = pd.read_csv(p) if p.exists() else None
    return tabs, info


def verdict_inputs_present(dirs: dict) -> dict:
    """{파일: (묶음, 있음)}. 묶음 = rule(규칙 입력), optional, gpu(학습기 사실)."""
    name = {id(RULE_FILES): "rule", id(OPTIONAL_FILES): "optional", id(GPU_FILES): "gpu"}
    return {f: (name[id(grp)], (Path(dirs[key]) / f).exists()) for grp, key, f in _all_files()}


# ================================================================ 2) 방법(적합과 예측)
@dataclass
class Rows:
    """적합이나 예측에 쓰는 행 묶음. X = x25(float32), y(없으면 NaN), s = √TDD, anc = 앵커 원값(soil, ku, ed, cci)."""
    X: np.ndarray
    s: np.ndarray
    anc: dict
    y: np.ndarray | None = None
    macro: np.ndarray | None = None

    def sub(self, m):
        m = np.asarray(m)
        return Rows(self.X[m], self.s[m], {k: v[m] for k, v in self.anc.items()}, None if self.y is None else self.y[m],
                    None if self.macro is None else self.macro[m])

    def __len__(self):
        return len(self.s)


def empty_rows(ncol: int) -> Rows:
    return Rows(np.zeros((0, ncol), np.float32), np.zeros(0), {k: np.zeros(0) for k in ANCHOR_COL}, np.zeros(0))


def ls_E(y, s) -> float:
    return float(h40().ls_E(np.asarray(y, float), np.asarray(s, float)))


def shrink(E_ls, E0, m, kappa=KAPPA) -> float:
    return float(h42().shrink(E_ls, E0, m, kappa))


@dataclass
class Fitter:
    threads: int = 2
    n_fit: int = 0
    sec: float = 0.0
    log_rows: list = field(default_factory=list)

    def cb_g(self, X, y, Xp_list, tag=""):
        """seed 0·1 CatBoost(h40.cb_fit) 잔차 예측의 평균."""
        H = h40()
        args = SimpleNamespace(cb_iters=CB_ITERS, threads=int(self.threads))
        outs = [np.zeros(len(p)) for p in Xp_list]
        for sd in SEEDS:
            t0 = time.time()
            m = H.cb_fit(args, np.asarray(X, np.float32), np.asarray(y, float), sd)
            for j, p in enumerate(Xp_list):
                if len(p):
                    outs[j] += np.asarray(m.predict(np.asarray(p, np.float32), thread_count=int(self.threads)), float) / len(SEEDS)
            dt = time.time() - t0
            self.n_fit += 1; self.sec += dt
            self.log_rows.append(dict(tag=tag, seed=sd, n_train=int(len(y)), sec=round(dt, 2)))
        return outs


def anchor_values(kind: str, src: Rows, lab: Rows, tgts: list) -> tuple:
    """h42.anchor_fill 로 앵커 b 를 채운다. 반환 (b_src, b_lab, [b_tgt…], ρ)."""
    H2 = h42()
    b_lab_t = np.concatenate([lab.anc[kind]] + [t.anc[kind] for t in tgts]) if (len(lab) or tgts) else np.zeros(0)
    s_lab_t = np.concatenate([lab.s] + [t.s for t in tgts]) if (len(lab) or tgts) else np.zeros(0)
    out, rho, _ = H2.anchor_fill(src.anc[kind], b_lab_t, np.zeros(0), src.s, s_lab_t, np.zeros(0))
    allv = out["A"]
    cut = np.cumsum([len(lab)] + [len(t) for t in tgts])[:-1]
    parts = np.split(allv, cut)
    return out["src"], parts[0], parts[1:], rho


def fit_predict(method: str, src: Rows, lab: Rows, tgts: list, F: Fitter, tag="") -> tuple[list, dict]:
    """방법 하나를 원천(src)과 레나 라벨(lab, 없으면 빈 묶음)로 적합해 대상 묶음들(tgts)을 예측한다. 반환 (예측 목록, 계수 기록)."""
    H2 = h42()
    n = len(lab)
    E0 = ls_E(src.y, src.s)
    E_ls = ls_E(lab.y, lab.s) if n else np.nan
    En = shrink(E_ls, E0, n)
    info = dict(method=method, n_lab=int(n), E0=E0, E_ls=E_ls, E_n=En)
    if method == "P0":
        return [E0 * t.s for t in tgts], info
    if method == "P1":
        return [En * t.s for t in tgts], info
    if method in ("R0", "R1"):
        Ea = E0 if (method == "R0" or n == 0) else En
        X = src.X; y = src.y - E0 * src.s
        if method == "R1" and n:
            X = np.vstack([src.X, lab.X]); y = np.concatenate([y, lab.y - En * lab.s])
        g = F.cb_g(X, y, [t.X for t in tgts], tag=f"{tag}{method}")
        info.update(anchor_E=Ea, lam=LAM, seeds=list(SEEDS))
        return [Ea * t.s + LAM * gg for t, gg in zip(tgts, g)], info
    if method == "B:s_aff":
        a, b = H2.affine_ls(src.s, src.y)
        info.update(a=a, b=b)
        return [a + b * t.s for t in tgts], info
    if method.startswith("P0@") or method.startswith("P1@") or method.endswith("_raw") or method in ("B:cci_aff", "B:ens"):
        def anc(kind):
            bs, bl, bt, rho = anchor_values(kind, src, lab, tgts)
            c0 = ls_E(src.y, bs)
            cl = ls_E(lab.y, bl) if n else np.nan
            c1 = shrink(cl, c0, n)
            return bs, bl, bt, rho, c0, c1
        if method.startswith("P0@") or method.startswith("P1@"):
            kind = method.split("@")[1]
            _, _, bt, rho, c0, c1 = anc(kind)
            c = c0 if method.startswith("P0@") else c1
            info.update(anchor=kind, rho=rho, c0=c0, c1=c1)
            return [c * b for b in bt], info
        if method.endswith("_raw"):
            kind = method.split(":")[1].replace("_raw", "")
            _, _, bt, rho, _, _ = anc(kind)
            info.update(anchor=kind, rho=rho)
            return list(bt), info
        if method == "B:cci_aff":
            _, _, bt, rho, _, _ = anc("cci")
            a, b = H2.affine_ls(src.anc["cci"], src.y)
            info.update(a=a, b=b, rho=rho)
            return [a + b * v for v in bt], info
        if method == "B:ens":
            _, _, bt_ku, _, ck0, ck1 = anc("ku")
            _, _, bt_cc, _, cc0, cc1 = anc("cci")
            if n == 0:
                info.update(c_ku=ck0, c_cci=cc0, E=E0)
                return [(E0 * t.s + ck0 * k + cc0 * c) / 3.0 for t, k, c in zip(tgts, bt_ku, bt_cc)], info
            info.update(c_ku=ck1, c_cci=cc1, E=En)
            return [(En * t.s + ck1 * k + cc1 * c) / 3.0 for t, k, c in zip(tgts, bt_ku, bt_cc)], info
    raise ValueError(f"지도에서 계산할 수 없는 방법: {method}")


# ================================================================ 3) 구간(6.6 (d))
def cal_groups(src: Rows, min_cells=MIN_CAL_CELLS) -> list:
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(src.y) & np.isfinite(src.s) & (src.y > 0) & (src.s > 0)
    reg, cnt = np.unique(src.macro[ok].astype(str), return_counts=True)
    return sorted(str(r) for r, c in zip(reg, cnt) if int(c) >= int(min_cells))


def log_score(y, yhat):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(np.log(np.asarray(y, float) / np.maximum(np.asarray(yhat, float), Y_FLOOR_CM)))


def calibrate(method: str, src: Rows, F: Fitter, sigma_cal: dict | None = None) -> dict:
    """지역 하나 제외 표본 밖 점수의 지역 등가중 0.9 분위. sigma_cal = {지역: 셀별 σ}(nflow 정규화기, 없으면 σ = 1)."""
    import polar.lgu_common as LC
    G = cal_groups(src)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(src.y) & np.isfinite(src.s) & (src.y > 0) & (src.s > 0)
    scores, groups, per = [], [], {}
    for k in G:
        te = ok & (src.macro.astype(str) == k)
        tr = src.macro.astype(str) != k
        (p,), info = fit_predict(method, src.sub(tr), empty_rows(src.X.shape[1]), [src.sub(te)], F, tag=f"cal[{k}]:")
        sc = log_score(src.y[te], p)
        if sigma_cal is not None:
            sc = sc / np.asarray(sigma_cal[k], float)
        scores.append(sc); groups.append(np.full(len(sc), k))
        per[k] = dict(n=int(te.sum()), median=float(np.nanmedian(sc)), p90=float(np.nanpercentile(sc, 90)),
                      E0_minus_k=info.get("E0"))
    s_all = np.concatenate(scores) if scores else np.zeros(0)
    g_all = np.concatenate(groups) if groups else np.zeros(0)
    q = float(LC.hier_quantiles(s_all, g_all, [Q_LEVEL])[0]) if len(s_all) else np.nan
    return dict(q=q, groups=G, groups_lgu20=cal_groups(src, MIN_CAL_CELLS_LGU), per_group=per, n_scores=int(len(s_all)),
                normalizer="nflow" if sigma_cal is not None else "const")


def interval(yhat, q, sigma=None):
    y = np.maximum(np.asarray(yhat, float), Y_FLOOR_CM)
    sg = 1.0 if sigma is None else np.asarray(sigma, float)
    lo, hi = y * np.exp(-q * sg), y * np.exp(q * sg)
    return lo, hi, hi - lo


# ================================================================ 4) 외삽 표시(6.6 (e))
def extrap_flags(Xsrc: np.ndarray, Xg: np.ndarray, cols: list, shown: np.ndarray) -> tuple[pd.DataFrame, dict]:
    lo = np.nanpercentile(Xsrc, EXTRAP_PCT[0], axis=0)
    hi = np.nanpercentile(Xsrc, EXTRAP_PCT[1], axis=0)
    fin = np.isfinite(Xg)
    out = fin & ((Xg < lo) | (Xg > hi))
    n_out = out.sum(1)
    n_miss = (~fin).sum(1)
    tot = n_out + n_miss
    freq = out[shown].mean(0) if shown.any() else np.zeros(len(cols))
    order = np.argsort(-freq)
    top = [dict(column=cols[j], frac_outside_shown=float(freq[j])) for j in order[:3]]
    df = pd.DataFrame(dict(n_extrap_out=n_out.astype(int), n_x25_missing=n_miss.astype(int), n_extrap_total=tot.astype(int),
                           extrap_cat=np.minimum(tot, 3).astype(int)))
    info = dict(pct=list(EXTRAP_PCT), columns=cols, lo=dict(zip(cols, map(float, lo))), hi=dict(zip(cols, map(float, hi))), top3=top,
                cat_counts_shown={str(k): int(v) for k, v in pd.Series(np.minimum(tot, 3)[shown]).value_counts().sort_index().items()},
                frac_outside_by_col_shown=dict(zip(cols, map(float, freq))))
    return df, info


# ================================================================ 5) 문맥(실제 자료와 합성 자료)
@dataclass
class MapCtx:
    src: Rows
    lena_all: Rows                      # 레나 F4_direct 전체(패널 (c) 라벨)
    lena_A: Rows                        # 예시 분할의 A(행 순서 = half_split_blocks 의 A_idx)
    grid: Rows
    grid_df: pd.DataFrame
    feats: list
    split: int
    split_info: dict
    draws_by_split: dict                # {split: {draw: sel(A 색인)}}, 패널 (b) 캡션의 E_n 범위
    A_by_split: dict                    # {split: Rows}
    meta: dict = field(default_factory=dict)


def rows_from_df(df: pd.DataFrame, feats: list, y_col=None, macro_col=None) -> Rows:
    anc = {k: df[c].values.astype(float) if c in df.columns else np.full(len(df), np.nan) for k, c in ANCHOR_COL.items()}
    return Rows(df[feats].values.astype(np.float32), df["e5_sqrt_tdd"].values.astype(float), anc,
                df[y_col].values.astype(float) if y_col else None, df[macro_col].values if macro_col else None)


def real_ctx(grid_path: Path, threads: int, D=None, grid_df: pd.DataFrame | None = None) -> MapCtx:
    """h40 의 자료(D, 없으면 h40.get_data)로 원천·레나 라벨·분할·추출을 만든다. D 와 grid_df 는 시험에서 합성 자료를 줄 때 쓴다."""
    H = h40()
    if D is None:
        args = H.parse_args(["--threads", str(threads), "--splits", "5"])
        D = H.get_data(args)
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx(TARGET, MODE)
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src_ok = src_idx[ok]
    feats = list(H.FEATS)
    s_df = df.iloc[src_ok]
    src = Rows(s_df[feats].values.astype(np.float32), s_df.s.values.astype(float),
               {k: s_df[c].values.astype(float) for k, c in ANCHOR_COL.items()}, s_df.y.values.astype(float), s_df.macro.values.astype(str))
    lena_all = rows_from_df(df.iloc[t_idx], feats, "y")
    info = D.split_structure(TARGET)
    valid = sorted(sp for sp, v in info.items() if v["valid"])
    if not valid:
        raise SystemExit("레나 x 에 유효 분할이 없다")
    draws, A_by = {}, {}
    for sp in valid:
        A_idx, _ = H.half_split_blocks(df, t_idx, sp)
        A_by[sp] = rows_from_df(df.iloc[A_idx], feats, "y")
        draws[sp] = {d: H.draw_cells(TARGET, MODE, sp, N_PANEL_B, d, len(A_idx)) for d in range(5)}
    sp0 = valid[0]
    g = grid_df if grid_df is not None else pd.read_csv(grid_path, dtype={"cell_id": str})
    grid = rows_from_df(g, feats)
    meta = dict(source=comp, n_src=int(len(src)), n_lena=int(len(t_idx)), valid_splits=valid, split_info={int(k): v for k, v in info.items()},
                subregion_src=getattr(D, "sub_src", ""), h40_sha=file_info(ROOT / "scripts/3_deep_learning/h40_label_grid.py"),
                h42_sha=file_info(ROOT / "scripts/3_deep_learning/h42_label_grid_ext.py"))
    return MapCtx(src, lena_all, A_by[sp0], grid, g, feats, sp0, info[sp0], draws, A_by, meta)


def synthetic_ctx(seed=0, n_grid_side=40) -> tuple[MapCtx, dict]:
    """합성 원천·레나·격자(실제 자료를 읽지 않는다). 반환 (문맥, 합성 판정 표 생성에 쓰는 정보)."""
    from polar.fidelity import SHARED_CORE
    from polar.m1_core import half_split_blocks
    from polar.fidelity import add_group_keys
    H = h40()
    feats = list(SHARED_CORE)
    rng = np.random.RandomState(seed)
    regs = dict(Alaska=(400, 64.0, -150.0, 5.2), Canada=(150, 62.0, -115.0, 4.3), Russia_W=(34, 67.0, 70.0, 3.6),
                Russia_E=(32, 68.0, 160.0, 4.8), Tibet=(22, 34.0, 93.0, 9.0))

    def make(nr, lat0, lon0, E, lena=False):
        la = lat0 + rng.rand(nr) * 1.5; lo = lon0 + rng.rand(nr) * 4.0
        s = rng.uniform(20, 42, nr)
        X = {c: rng.normal(0, 1, nr) for c in feats}
        X["e5_sqrt_tdd"] = s; X["e5_tdd"] = s ** 2; X["cci_valid"] = np.ones(nr)
        y = E * s * np.exp(rng.normal(0, 0.18, nr) + 0.05 * X["dem_slope"])
        X["cci_alt"] = np.where(rng.rand(nr) < 0.05, np.nan, y * rng.uniform(0.7, 1.3, nr))
        d = pd.DataFrame(X); d["lat"] = la; d["lon"] = lo; d["y"] = y
        d["e5_sqrt_tdd_soil"] = s * rng.uniform(0.85, 1.05, nr)
        d["p4_ku"] = np.where(rng.rand(nr) < 0.1, np.nan, y * rng.uniform(0.6, 1.5, nr))
        d["p2_edaphic"] = y * rng.uniform(0.5, 1.8, nr)
        return d
    parts = []
    for k, (nr, la, lo, E) in regs.items():
        d = make(nr, la, lo, E); d["macro"] = k; parts.append(d)
    S = pd.concat(parts, ignore_index=True)
    src = rows_from_df(S, feats, "y", "macro")
    L = make(300, 71.6, 123.5, 3.4, lena=True)
    L = add_group_keys(L)
    lena_all = rows_from_df(L, feats, "y")
    t_idx = np.arange(len(L))
    draws, A_by, info = {}, {}, {}
    for sp in (1, 2, 3):
        A_idx, B_idx = half_split_blocks(L, t_idx, sp)
        A_by[sp] = rows_from_df(L.iloc[A_idx], feats, "y")
        draws[sp] = {d: H.draw_cells(TARGET, MODE, sp, N_PANEL_B, d, len(A_idx)) for d in range(5)}
        info[sp] = dict(valid=True, n_A=int(len(A_idx)))
    # 합성 격자: 실제 격자와 같은 셀 색인 식(6B.3)으로 만든 연속 셀. n_grid_side = 0 이면 영역 전체(약 5.3만 셀), 아니면 영역 가운데의
    # n × n 조각이다. 좌표만 실제 규칙을 따르고 공변량 값은 합성이다
    cd = 0.009
    ky_all = [k for k in range(int(71.5 / cd) - 1, int(73.6 / cd) + 2) if 71.5 <= (k + 0.5) * cd <= 73.6]
    if n_grid_side:
        mid = len(ky_all) // 2
        ky_all = ky_all[mid - n_grid_side // 2: mid - n_grid_side // 2 + n_grid_side]
    rows_g = []
    for k in ky_all:
        c = np.cos(np.radians((k + 0.5) * cd))
        kx = np.arange(int(123.3 * c / cd) - 1, int(130.1 * c / cd) + 2)
        lonc = (kx + 0.5) * cd / c
        kx = kx[(lonc >= 123.3) & (lonc <= 130.1)]
        if n_grid_side:
            m0 = len(kx) // 2
            kx = kx[m0 - n_grid_side // 2: m0 - n_grid_side // 2 + n_grid_side]
        rows_g += [(k, int(x), (k + 0.5) * cd, (x + 0.5) * cd / c) for x in kx]
    gk = np.array(rows_g)
    LA, LO = gk[:, 2], gk[:, 3]
    ng = len(gk)
    G = {c: rng.normal(0, 1.1, ng) for c in feats}
    s = 26 + 8 * (LA - 71.5) / 2.1 + rng.normal(0, 1, ng)
    G["e5_sqrt_tdd"] = s; G["e5_tdd"] = s ** 2; G["cci_valid"] = np.ones(ng)
    G["cci_alt"] = 3.3 * s * rng.uniform(0.8, 1.2, ng)
    g = pd.DataFrame(G)
    g.insert(0, "cell_id", [f"{int(a)}_{int(b)}" for a, b in gk[:, :2]]); g["lat"] = LA; g["lon"] = LO
    g["ky"] = gk[:, 0].astype(int); g["kx"] = gk[:, 1].astype(int)
    g["e5_sqrt_tdd_soil"] = s * 0.95
    g["p4_ku"] = np.where(rng.rand(ng) < 0.05, np.nan, 3.0 * s)
    g["p2_edaphic"] = 3.6 * s
    g["dem_slope"] = np.where(rng.rand(ng) < 0.02, 9.0, g.dem_slope)          # 외삽 셀 일부
    # 합성 마스크(6.5 의 사유 이름): 바다(DEM 타일 없음), 수체(호수 하나와 흩어진 셀), 육지의 PFR 결측·CCI 결측 소수
    sea = (LA > 73.3) & (LO > 129.0)
    water = ~sea & ((rng.rand(ng) < 0.06) | (np.hypot((LA - 72.4) / 0.25, (LO - 126.8) / 0.9) < 1))
    land = ~sea & ~water
    u = rng.rand(ng)
    pfr_miss, cci_miss = land & (u < 0.002), land & (u >= 0.002) & (u < 0.006)
    reason = np.select([sea, water, pfr_miss, cci_miss], ["dem_tile_absent", "water", "pfr_missing", "cci_missing"], "")
    g["gray_reason"] = reason; g["gray"] = (reason != "").astype(int); g["land"] = land.astype(int)
    g.loc[(sea | water) & (rng.rand(ng) < 0.5), "cci_alt"] = np.nan
    g.loc[cci_miss, "cci_alt"] = np.nan
    grid = rows_from_df(g, feats)
    meta = dict(synthetic=True, seed=seed, regions={k: v[0] for k, v in regs.items()}, n_lena=len(L))
    return MapCtx(src, lena_all, A_by[1], grid, g, feats, 1, info[1], draws, A_by, meta), dict(feats=feats)


# ================================================================ 6) 합성 판정 표(시험 전용)
def synthetic_tables(scenario: str) -> dict:
    """시나리오별 합성 판정 표. h40·h42·h39·h49_map_contrasts·h45·h43·h47·h48 의 열 구성과 라벨 형식을 흉내 낸다(실제 결과가 아니다).
    baseline 모두 기본 방법, L8 기각 | all_ml R0·R1 우세, GPU 학습기 우세 | pstar L29 의 P*(목록 밖 P0@tddm → 목록 안 P0@soil)와 P1*(n 10, P1@ku)
    replace1 n 10 의 L29 P1* 가 목록 밖 P1@tddm 이라 목록 안 P1@soil 로 대체하고 R1 − P1@soil 은 lgw_map_aux 에서 읽는다 | b2 LGU-B2 '전이'."""
    if scenario not in SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오 {scenario}")
    M4 = "MEAN[" + ",".join(MAIN4_T) + "]"
    boot = dict(nboot=NBOOT_MIN, nboot_capped=False)

    def row(test, label, scope, target, v, d, n, **kw):
        return dict(test_id=test, contrast=label, scope=scope, target=target, verdict4=v, delta=d, n=n, **kw)

    def pair(test, label, v_mean, d_mean, v_lena, d_lena, n, **kw):
        """h42·h39 처럼 지역 행 4개와 층화 평균 행 1개."""
        rs = [row(test, label, "region", t, v_lena if t == "Lena|x" else "미결정", d_lena if t == "Lena|x" else d_mean, n, **kw) for t in MAIN4_T]
        return rs + [row(test, label, "MEAN", M4, v_mean, d_mean, n, n_ci_regions=4, n_regions_target=4, **kw)]
    ml = scenario in ("all_ml", "pstar", "replace1")
    lgx_aux, lgx_tests, bundle, mapaux = [], [], [], []
    for n, lab in ((N_PANEL_B, "n10"), (N_ALL, "n-1")):                          # LG L4(h42 lgx_lg_aux)
        lgx_aux += pair("L4", f"R1-P1|{lab}", "우세" if ml else "미결정", -1.2 if ml else -0.1, "우세" if ml else "동등", -2.0, n)
    for ab, lab, n in (("AB5", "R1-P1|n10", N_PANEL_B), ("AB9", "R1-P1|all", N_ALL)):  # h39 bundle(10,000회)
        bundle += [dict(r, ab=ab, pool="지역 4/4" if r["scope"] == "MEAN" else "", **boot)
                   for r in pair("", lab, "우세" if ml else "미결정", -1.2 if ml else -0.1, "우세" if ml else "동등", -2.0, n)]
    r0 = scenario == "all_ml"                                                    # h49_map_contrasts: R0 − P0(n 0)
    mapaux += [dict(r, **boot) for r in pair("MAP-a", "R0-P0|n0", "우세" if r0 else "미결정", -2.5 if r0 else 0.3, "동등" if r0 else "미결정", -0.3, 0)]
    for b in MAP_B:                                                              # h49_map_contrasts: R1 − b(n 10, 전량)
        for n, lab in ((N_PANEL_B, "n10"), (N_ALL, "n-1")):
            dom = ml and not (scenario == "pstar" and n == N_ALL and b != "P1@ku")
            mapaux += [dict(r, **boot) for r in pair("MAP-bc", f"R1-{b}|{lab}", "우세" if dom else "미결정", -0.9 if dom else 0.1, "미결정", -0.4, n)]
    for m in L29_N0_H42:                                                         # h42 L29(lgx_tests)
        dom = scenario == "pstar" and m in ("P0@tddm", "P0@soil")
        d = {"P0@tddm": -3.0, "P0@soil": -1.5}.get(m, 0.4) if scenario == "pstar" else 0.4
        lgx_tests += pair("L29", f"{m}-P0|n0", "우세" if dom else "미결정", d, "미결정", d, 0)
    ln_dom = {"pstar": {(N_PANEL_B, "P1@ku"): -0.8}, "replace1": {(N_PANEL_B, "P1@tddm"): -1.0, (N_PANEL_B, "P1@soil"): -0.6}}.get(scenario, {})
    for n, lab in ((N_PANEL_B, "n10"), (N_ALL, "n-1")):
        for m in L29_LN_H42:
            d = ln_dom.get((n, m))
            lgx_tests += pair("L29", f"{m}-P1|{lab}", "우세" if d is not None else "미결정", d if d is not None else 0.2, "동등", -0.2, n)
        picks = [m for (nn, m), _ in sorted(ln_dom.items(), key=lambda kv: kv[1]) if nn == n][:1]
        for m in picks:                                                          # h42 는 자기가 고른 P1* 에 대해서만 R1 − P1* 를 만든다
            lgx_tests += pair("L29", f"R1-{m}|{lab}", "우세", -0.9, "미결정", -0.4, n)
    lgx_aux += pair("L5", "R1[tabm]-R1[catboost_lo]|n10", "우세" if scenario == "all_ml" else "미결정", -0.7, "미결정", -0.5, 10)
    lgt = pair("L32", "R1[T]-R1[C]|n10|lam0.25", "우세" if scenario == "all_ml" else "동등", -0.6, "미결정", -0.2, 10)
    lgt += pair("L32", "R1[T]-R1[C]|n10|lam1.0", "우세", -0.9, "미결정", -0.2, 10)          # λ 1.0 은 캡션 사실에 쓰지 않는다
    lgf = pair("LGF-F2", "R1[I]-R1[C]|n10|lam0.25", "미결정", -0.1, "미결정", -0.1, 10)
    lgfn = pair("LGF-N3", "R1[mlp*]-R1[CB]|n10|lam0.25", "우세" if scenario == "all_ml" else "미결정", -0.5, "미결정", -0.3, 10)
    lgfn += pair("LGF-N3", "R1[mlp]-R1[CB]|n10", "미결정", -0.2, "미결정", -0.1, 10)
    lg = pd.DataFrame([dict(test_id="L8", item="verdict", scope="verdict", verdict="기각" if scenario == "baseline" else "지지", stat="")])
    lgu = pd.DataFrame([dict(test="LGU-B2", verdict="전이" if scenario == "b2" else "미결정", delta=-0.1, ci_hi=0.02, ci_hi_beq=0.05,
                             coverage_pool=0.83)])
    aux4 = pd.DataFrame([dict(row("L6", "R1@a1-P0|n10", "region", "Lena|x", "미결정", 0.2, 10), **boot)])
    return dict(lg_tests=lg, lgx_lg_aux=pd.DataFrame(lgx_aux), lgx_tests=pd.DataFrame(lgx_tests), lgw_bundle=pd.DataFrame(bundle),
                lgw_map_aux=pd.DataFrame(mapaux), lgw_aux4=aux4, lgu_b_tests=lgu, lgt_tests=pd.DataFrame(lgt), lgf_tests=pd.DataFrame(lgf),
                lgfn_tests=pd.DataFrame(lgfn))


# ================================================================ 7) 실행
def land_gray_stats(g: pd.DataFrame) -> dict:
    """6.5 의 캡션 사실: 육지 셀 수, 회색 육지 셀 수와 비율, 첫 사유별 수(격자의 land, gray, gray_reason 열)."""
    if not {"land", "gray"} <= set(g.columns):
        return {}
    land, gray = g.land.values.astype(int) == 1, g.gray.values.astype(int) == 1
    rs = g.gray_reason.fillna("").astype(str).values if "gray_reason" in g.columns else np.full(len(g), "")
    by = pd.Series(rs[land & gray]).value_counts()
    return dict(n_land=int(land.sum()), n_land_gray=int((land & gray).sum()), frac_land_gray=float((land & gray).sum() / max(int(land.sum()), 1)),
                land_by_reason_first={str(k): int(v) for k, v in by.items() if str(k)})


def check_out(path, forbid=()) -> Path:
    """출력 보호: data/processed/ 의 실험 산출 폴더(PROTECT), results/, forbid 의 폴더 안이면 거부한다."""
    p = Path(os.path.realpath(str(path)))
    for q in [PROC / d for d in PROTECT] + [ROOT / "results"] + [Path(x) for x in forbid]:
        q = Path(os.path.realpath(str(q)))
        if p == q or q in p.parents:
            raise SystemExit(f"[거부] 출력 폴더 {p} 는 보호 폴더 {q} 안이다")
    return p


def run(ctx: MapCtx, tables: dict, out_dir: Path, threads: int, tag_info: dict, sigma_file: str | None = None, strict: bool = False,
        grid_meta: dict | None = None) -> dict:
    """규칙 적용, 적합과 예측, 구간, 외삽 표시. strict 이면 규칙 입력 확인(check_rule_inputs)에 문제가 있을 때 선택을 기록하지 않고 중단한다."""
    t_start = time.time()
    book = VerdictBook({k: tables.get(k) for k in BOOK_SOURCES})
    b2 = lgu_b2(tables.get("lgu_b_tests"))
    l8 = l8_verdict(tables.get("lg_tests"))
    gpu = gpu_learner_notes(tables)
    sel = rule_select(book, b2, l8, gpu)
    probs = check_rule_inputs(book, sel, tables, b2, l8)
    sel["input_check"] = dict(strict=bool(strict), problems=probs, table_stats=book.stats)
    if strict and probs:
        log("[중단] 규칙 입력 확인 실패:\n  " + "\n  ".join(probs))
        raise SystemExit("[중단] 6.7 규칙에 필요한 판정 행이 없다. 선택을 기록하지 않았다. h39 summarize 와 h49_map_contrasts 의 산출, --lgw-dir 를 "
                         "확인한다")
    sel["gpu_missing"] = [f"{s}.csv" for s, _, _ in GPU_SOURCES if tables.get(s) is None]
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_path = out_dir / "lena_pred_v1_selection.json"
    sel_path.write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d %H:%M"), selection=sel, **tag_info), ensure_ascii=False, indent=1,
                                   default=_jsonable))
    log(f"[rule] 패널 방법 {sel['panels']} · 정규화기 {sel['d']['normalizer']} · 배치 {sel['placement']}")
    sigma = None
    if sel["d"]["normalizer"] == "nflow":
        if not sigma_file or not Path(sigma_file).exists():
            raise SystemExit("[중단] LGU-B2 가 '전이'라 패널 (d)에 nflow σ 가 필요하다. LGU 의 σ 파일(--sigma-file)을 확인하고 LGU 개정 이력에 적은 뒤 "
                             "다시 실행한다(6.9)")
        z = np.load(sigma_file, allow_pickle=True)
        sigma = dict(cal={str(k): np.asarray(v, float) for k, v in z["cal"].item().items()},
                     grid=pd.Series(np.asarray(z["grid_sigma"], float), index=np.asarray(z["grid_cell_id"]).astype(str)))
    F = Fitter(threads=threads)
    grid = ctx.grid
    shown = ctx.grid_df.gray.values == 0 if "gray" in ctx.grid_df else np.ones(len(grid), bool)
    # 라벨 묶음
    sel_b = ctx.draws_by_split[ctx.split][0]
    lab_b = ctx.lena_A.sub(sel_b)
    lab_c = ctx.lena_all
    none = empty_rows(grid.X.shape[1])
    preds, coefs = {}, {}

    def pred(key, method, lab):
        if key not in preds:
            (p,), info = fit_predict(method, ctx.src, lab, [grid], F, tag=f"grid[{key}]:")
            preds[key] = p; coefs[key] = info
        return preds[key]
    # 기본 방법(패널 기본값)과 선택 방법
    pred("P0", "P0", none)
    pred(f"P1_n{N_PANEL_B}", "P1", lab_b)
    pred(f"R1_n{N_PANEL_B}", "R1", lab_b)
    pred("P1_all", "P1", lab_c)
    pred("R1_all", "R1", lab_c)
    ma, mb, mc = sel["panels"]["a"], sel["panels"]["b"], sel["panels"]["c"]
    ka = ma if ma == "P0" else f"{ma}_n0"
    pred(ka, ma, none)
    kb = {"P1": f"P1_n{N_PANEL_B}", "R1": f"R1_n{N_PANEL_B}"}.get(mb, f"{mb}_n{N_PANEL_B}")
    pred(kb, mb, lab_b)
    kc = {"P1": "P1_all", "R1": "R1_all"}.get(mc, f"{mc}_all")
    pred(kc, mc, lab_c)
    # 구간
    cal = calibrate(ma, ctx.src, F, sigma_cal=sigma["cal"] if sigma else None)
    sg_grid = sigma["grid"].reindex(ctx.grid_df.cell_id.astype(str)).values if sigma else None
    lo, hi, wid = interval(preds[ka], cal["q"], sg_grid)
    # 외삽
    cont = [c for c in ctx.feats if c != "cci_valid"]
    Xs = ctx.src.X[:, [ctx.feats.index(c) for c in cont]].astype(float)
    Xg = grid.X[:, [ctx.feats.index(c) for c in cont]].astype(float)
    ex, ex_info = extrap_flags(Xs, Xg, cont, shown)
    # 패널 (b) 캡션: 분할·추출별 E_n 과 영역 평균 ALT(해석식, P1)
    E0 = ls_E(ctx.src.y, ctx.src.s)
    s_mean = float(np.nanmean(grid.s[shown])) if shown.any() else np.nan
    en_rows = []
    for sp, dd in ctx.draws_by_split.items():
        for d, sl in dd.items():
            A = ctx.A_by_split[sp]
            En = shrink(ls_E(A.y[sl], A.s[sl]), E0, len(sl))
            en_rows.append(dict(split=int(sp), draw=int(d), E_n=En, mean_alt_cm=En * s_mean))
    en = pd.DataFrame(en_rows)
    same = en[en.split == ctx.split]
    cap_b = dict(split=int(ctx.split), draw=0, n=N_PANEL_B, E0=E0,
                 E_n_same_split=[float(same.E_n.min()), float(same.E_n.max())], mean_alt_same_split=[float(same.mean_alt_cm.min()), float(same.mean_alt_cm.max())],
                 E_n_all=[float(en.E_n.min()), float(en.E_n.max())], mean_alt_all=[float(en.mean_alt_cm.min()), float(en.mean_alt_cm.max())],
                 n_splits=int(en.split.nunique()), note="해석식(P1 = E_n·s) 영역 평균. 패널 (b) 방법이 R1 이면 잔차 항은 포함하지 않는다")
    # 표
    g = ctx.grid_df
    o = pd.DataFrame(dict(cell_id=g.cell_id.values, lat=g.lat.values, lon=g.lon.values, ky=g.ky.values, kx=g.kx.values))
    for c in ("gray", "gray_reason", "land"):
        if c in g:
            o[c] = g[c].values
    o["pred_a"] = preds[ka]; o["pred_b"] = preds[kb]; o["pred_c"] = preds[kc]
    o["lo90_a"] = lo; o["hi90_a"] = hi; o["width90_a"] = wid
    o["diff_c_minus_a"] = o.pred_c - o.pred_a
    for k, v in preds.items():
        o[f"m_{k}"] = v
    o = pd.concat([o, ex], axis=1)
    pred_path = out_dir / "lena_pred_v1.csv.gz"
    o.to_csv(pred_path, index=False, compression=dict(method="gzip", mtime=0), float_format="%.6g")
    shown_vals = {k: _stats(o.loc[shown, c]) for k, c in (("a", "pred_a"), ("b", "pred_b"), ("c", "pred_c"), ("width", "width90_a"),
                                                              ("diff", "diff_c_minus_a"))}
    gl = land_gray_stats(g)
    if grid_meta and gl:                                                   # 격자 메타의 같은 값과 대조(다르면 격자와 메타의 판이 다르다)
        ms = grid_meta.get("masks", {}).get("summary", {})
        gl["matches_grid_meta"] = bool(ms.get("n_land") == gl["n_land"] and ms.get("n_land_gray") == gl["n_land_gray"])
    caption = dict(
        panels={p: dict(method=sel["panels"][p], why=sel[p]["why"]) for p in ("a", "b", "c")},
        interval=dict(method=ma, normalizer=cal["normalizer"], q90=cal["q"], groups=cal["groups"], note="구간 폭은 예측값에 비례한다" if cal["normalizer"] == "const" else ""),
        placement=sel["placement"], L8_rejected=sel["c"]["caption_L8"], gpu_notes=sel["gpu_notes"], gpu_missing=sel["gpu_missing"],
        gray_land=gl,
        replaced=[x for x in (sel["a"]["l29"], sel["b"]["l29"], sel["c"]["l29"]) if x.get("replaced")],
        extrap_top3=ex_info["top3"], panel_b=cap_b,
        gray=dict(n_gray=int((~shown).sum()), n_cells=int(len(o)),
                  by_reason=o.loc[~shown, "gray_reason"].value_counts().to_dict() if "gray_reason" in o else {}),
        fixed=["정적 지도(ERA5-Land 2015–2020 기후값, CCI ALT 1997–2021 평균)", "라벨은 점·지점 평균이고 지도는 1 km 격자다",
               "영구동토 마스크: CCI PFR 1997–2021 평균 < 10 % 회색", "외삽 표시는 오차 위험 지표가 아니다(FAILURE_ANALYSIS §4.3, 순위상관 −0.64)",
               "지도는 CPU 학습기(catboost_lo)와 해석식만 쓴다"])
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/h49_transfer_map.py", script_sha256=sha256_file(Path(__file__)),
                plan="docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.6–6.10 (커밋 316714c)", **tag_info,
                selection=sel, coefs=coefs, calibration=cal, extrap=ex_info, caption=caption, shown_stats=shown_vals,
                en_table=en_rows, ctx=ctx.meta, fits=dict(n=F.n_fit, sec=round(F.sec, 1), detail=F.log_rows),
                constants=dict(KAPPA=KAPPA, LAM=LAM, SEEDS=list(SEEDS), CB_ITERS=CB_ITERS, Q_LEVEL=Q_LEVEL, MIN_CAL_CELLS=MIN_CAL_CELLS,
                               Y_FLOOR_CM=Y_FLOOR_CM, EXTRAP_PCT=list(EXTRAP_PCT)),
                outputs=dict(pred=file_info(pred_path), selection=file_info(sel_path)),
                threads=threads, max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                elapsed_s=round(time.time() - t_start, 1))
    (out_dir / "lena_pred_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_jsonable))
    log(f"[done] 적합 {F.n_fit}건 {F.sec:.1f}s · q90 {cal['q']:.3f} · 산출 {pred_path}")
    return meta


def _stats(s: pd.Series) -> dict:
    v = s.values.astype(float); v = v[np.isfinite(v)]
    if not len(v):
        return {}
    return dict(n=int(len(v)), min=float(v.min()), p02=float(np.percentile(v, 2)), p50=float(np.median(v)), p98=float(np.percentile(v, 98)),
                max=float(v.max()))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H49 지도 과제 예측기(레나 x)")
    ap.add_argument("--grid", default="data/processed/map_lena/lena_grid_x25_v1.csv.gz")
    ap.add_argument("--grid-meta", default="data/processed/map_lena/lena_grid_x25_v1_meta.json")
    ap.add_argument("--out-dir", default="data/processed/map_lena")
    ap.add_argument("--lg-dir", default="data/processed/lg")
    ap.add_argument("--lgx-dir", default="data/processed/lgx")
    ap.add_argument("--lgw-dir", default="data/processed/wrapup", help="h39 summarize·h49_map_contrasts 의 산출 폴더(WRAPUP 결과 절의 실제 폴더)")
    ap.add_argument("--lgu-dir", default="data/processed/lgu")
    ap.add_argument("--lgt-dir", default="data/processed/lgt")
    ap.add_argument("--lgf-dir", default="data/processed/lgf")
    ap.add_argument("--allow-missing-gpu-tables", action="store_true",
                    help="LGT·LGF 판정 표가 없어도 본 실행을 한다(메타와 캡션에 'LGT·LGF 학습기 사실 미포함'을 적는다)")
    ap.add_argument("--sigma-file", default="", help="LGU-B2 가 '전이'일 때 nflow σ(npz: cal={지역: σ}, grid_cell_id, grid_sigma)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow-run", action="store_true", help="판정 표가 모두 나온 뒤의 본 실행을 허용한다")
    ap.add_argument("--check-inputs", action="store_true", help="판정 표의 존재만 확인한다(내용을 열지 않는다)")
    ap.add_argument("--synthetic-test", default="", help="합성 입력 시험의 출력 폴더")
    ap.add_argument("--scenario", default="baseline", choices=list(SCENARIOS), help="합성 판정 표 시나리오")
    ap.add_argument("--synthetic-grid-side", type=int, default=40, help="합성 격자 조각의 한 변 셀 수(0 = 영역 전체)")
    return ap.parse_args(argv)


def _abs(p):
    return Path(p) if os.path.isabs(str(p)) else ROOT / p


def main(argv=None):
    a = parse_args(argv)
    if a.threads > MAX_THREADS:
        raise SystemExit(f"[거부] 스레드 {a.threads} > {MAX_THREADS}(6.9)")
    try:
        if os.nice(0) < 10:
            os.nice(10 - os.nice(0))
    except OSError:
        pass
    dirs = dict(lg=_abs(a.lg_dir), lgx=_abs(a.lgx_dir), lgw=_abs(a.lgw_dir), lgu=_abs(a.lgu_dir), lgt=_abs(a.lgt_dir), lgf=_abs(a.lgf_dir))
    if a.check_inputs:
        pres = verdict_inputs_present(dirs)
        for k, (grp, v) in pres.items():
            log(f"  [{grp}] {k}: {'있음' if v else '없음'}")
        g = _abs(a.grid)
        log(f"  격자 {g.relative_to(ROOT)}: {'있음' if g.exists() else '없음'}")
        return dict(present=pres)
    if a.synthetic_test:
        out = check_out(_abs(a.synthetic_test), forbid=(_abs(a.out_dir), PROC / "map_lena"))
        ctx, _ = synthetic_ctx(n_grid_side=a.synthetic_grid_side)
        tabs = synthetic_tables(a.scenario)
        return run(ctx, tabs, out, a.threads, dict(synthetic=True, scenario=a.scenario, note="합성 입력 시험. 실제 결과가 아니다"), strict=True)
    if not a.allow_run:
        raise SystemExit("[거부] 본 실행은 LG 집계, LGX L29, h39 집계, LGU-B2 판정이 나온 뒤 --allow-run 으로만 한다(6.11). 지금은 --synthetic-test 로 시험한다")
    out = check_out(_abs(a.out_dir))
    pres = verdict_inputs_present(dirs)
    miss = [k for k, (grp, v) in pres.items() if grp == "rule" and not v]
    if miss:
        raise SystemExit(f"[거부] 규칙 입력 판정 표가 없다: {miss}")
    miss_gpu = [k for k, (grp, v) in pres.items() if grp == "gpu" and not v]
    if miss_gpu and not a.allow_missing_gpu_tables:
        raise SystemExit(f"[거부] 학습기 사실(6.7 공통)의 판정 표가 없다: {miss_gpu}. 표가 나온 뒤 실행하거나 --allow-missing-gpu-tables 로 "
                         "'LGT·LGF 학습기 사실 미포함'을 적고 실행한다")
    gm = json.loads(_abs(a.grid_meta).read_text())
    if not gm.get("qa", {}).get("passed"):
        raise SystemExit("[거부] 격자 조립 QA(6.4)를 통과하지 못했다. 원인을 고친 뒤 다시 조립한다")
    tabs, info = read_verdict_tables(dirs)
    ctx = real_ctx(_abs(a.grid), a.threads)
    return run(ctx, tabs, out, a.threads, dict(synthetic=False, grid=file_info(_abs(a.grid)), grid_meta=file_info(_abs(a.grid_meta)),
                                              verdict_inputs=info, verdict_dirs={k: str(v) for k, v in dirs.items()},
                                              gpu_tables_missing=miss_gpu, allow_missing_gpu_tables=bool(a.allow_missing_gpu_tables)),
               sigma_file=a.sigma_file or None, strict=True, grid_meta=gm)


if __name__ == "__main__":
    main()
