"""H39 · 마무리 부록(WRAPUP) 집계기. 계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 3.6절(개정 1, 커밋 316714c)의 구현.

모드(--mode)
  units       2.2 지역별 라벨 단위 표와 2.3 추출별 단위 표. LG 의 라벨 행을 h40.draw_cells·draw_blocks 로 다시 만들고, LG 사전 점검과
              스모크 CPU 조각(키 열, n_lab, n_blocks_lab 만 읽는다)과 키별로 대조한다.
  sens-unit   2.4 (S-a) 셀 평균 묶음 민감도. 해석식 P0, P1, P2 만 쓴다. 분할 1–200(LGX-N4 규약), 추출 5, n ∈ {3, 10, 40}.
  sens-year   2.4 (S-b) 러시아 W·E 의 단년 라벨 민감도. CALM 원자료(PANGAEA 972777)의 지점·연도 값을 쓴다.
  sc3         3.5 SC3w 순차 멈춤 규칙. 해석식, LG 유효 분할(1–5), 반복 20, 블록 부트스트랩.
  bias-mae    1.7 편향·MAE 분해(LGX 기준 축 lgxb 의 _cells.npz). LGX 결과 회수 뒤에만 실행한다.
  summarize   1.1 묶음 AB1–AB10, 1.4 (a) L1 전량 행·L3·L6·L7, 1.4 (b) 과거 결과 재채점·재분류, 1.6 분할 SD/SE 비, 3.2–3.4 시나리오와 SC1w·SC2w,
              4절 L43, 5절 AK1w, 9.4 δ_rel. LG·LGX 결과 회수 뒤, 부록 커밋 뒤에만 실행한다.
  xenv-gate   8.4 (ii) 교차 환경 점검의 대조. 로컬 집계 표와 Rescale 집계 표의 열별 최대 절대 차, 결측 위치 불일치 수, 문자열 불일치 수만 쓴다.
              Δ 와 판정 문구는 출력하지 않는다.

동결 규칙
  h40(scripts/3_deep_learning/h40_label_grid.py), h42(h42_label_grid_ext.py), src/polar 의 기존 모듈은 고치지 않는다. h40 과 h42 는 importlib 로
  파일 경로에서 읽어 쓴다. 입력 조각과 표는 읽기 모드로만 연다.

쓰기 보호
  --out-dir 이 data/processed/ 의 lg/, lgx/, lgt/, lgd/, lgu/, lgf/ 안이거나 results/ 안이면 거부한다. 비교는 심볼릭 링크를 푼 실제 경로
  (os.path.realpath)로 한다.

재표집과 실행 표지(부록 0.5, 3.6)
  새 CI 는 10,000회다(h4_common.boot_delta_blocks, seed = seed_of('lgw', 대상, 대비)). --allow-local 이 없으면 스레드 1, 재표집 1,000회 이하로
  낮추고 표의 nboot·nboot_capped 열에 적는다. 그 값은 판정에 쓰지 않는다. --allow-local 에서도 스레드는 4 이하로 자른다(3.6 자원 규칙).
  직접 실행하면 스레드 환경 변수(OMP, OPENBLAS, MKL, NUMEXPR)를 이 값으로 덮어쓴다.

결과 열람 규칙
  units, sens-unit, sens-year, sc3, xenv-gate 는 LG·LGX·LGT 조각의 RMSE 열과 곡선·판정 표를 읽지 않는다(units 의 대조는 사전 점검·스모크 조각의
  키 열과 n_lab, n_blocks_lab 두 열만. xenv-gate 는 두 집계 표의 차이만 계산하고 값 자체는 쓰지 않는다).
  sc3 는 LG 결과와 같은 값이거나 같은 형식인 열(P0 대비, RMSE 열)을 봉인 폴더 <out>/sealed/ 에만 쓴다(규약 12).

계획서에 없는 구현 규약
  1. 1 km 위치 색인은 부록 2.2(6B.3)의 정의다: ky = floor(lat / 0.009), φ = (ky + 0.5)·0.009°, kx = floor(lon·cos φ / 0.009).
     위치 순서는 (ky, kx) 사전식 순서다. ERA5 조합은 8열을 소수 6자리로 반올림하고 결측을 −9999 로 둔 고유 행이다.
     최근접 0.1° 격자 색인은 (round(10·lat), round(10·lon)), 0.01° 반올림 위치는 (round(lat, 2), round(lon, 2)) 다.
  2. S-a 의 분할은 h40 의 분할 구조 규칙(중복 분할 제외, 유효 분할이 하나라도 있으면 채점 블록 2개 미만 분할 제외)을 split_seed 1–200 에 적용한다.
     재보정의 주 대비는 P1(κ = 10 수축)이다. P2 는 서술 열이다. 해석 규칙은 P1 의 Δ_unit 에 적용한다.
  3. S-b 의 라벨 행 추출 seed 는 seed_of('lgw-sb-row', 대상, 분할, n, 추출)이다(부록은 연도 선택 seed 만 정했다). 단년 값으로 바꾸는 행은
     CALM 지점 하나와만 대응하고 그 지점도 그 행 하나와만 대응하며 유효 연도가 있는 행이다. 그 밖의 행은 다년 평균을 그대로 쓴다.
  4. SC3w 의 상한은 min(40, |A|)이다. 단계는 {3, 10} 가운데 상한 미만인 값과 상한이다. |A| < 4 인 대상은 뺀다.
     잭나이프의 E^(−i) 가운데 비유한 값이나 0 이하가 있으면 SE = ∞ 로 두어 멈추지 않는다. 멈춘 단계의 E_ls 가 비유한이면 E0 을 쓴다(h40 과 같다).
  5. L43 의 2단 재표집에서 채점 블록 가중은 분할마다 h4_common.boot_weights(nb, nboot, seed_of(seed, 분할))로 두 배치가 공유하고,
     추출 재표집의 난수는 배치마다 seed_of(seed, 분할, 배치)로 따로 둔다. 셀 가중과 블록 등가중은 같은 재표집 색인을 쓴다.
  6. 9.4 의 δ_rel 판정은 가중마다 한계가 다르다(셀 가중 CI 는 0.02 × 셀 가중 P0 RMSE, 블록 등가중 CI 는 0.02 × 블록 등가중 P0 RMSE).
  7. AB10 은 --lgu-dir 의 '*tests*.csv' 에서 test_id ∈ {LGU-A1, A1} 이고 두 가중 p 열(p_cell·p_beq, 또는 p_boot·p_boot_beq)이 있는 행을 찾는다.
     찾지 못하면 행 없음(p = 1)이다.
  8. 1.4 (b) 재채점 대상은 h_analysis.py 와 h21_coef_borrow.py 에 등록된 확인적 대비다: H18a, H19d(x25_A), H19p(x25_A, x25_lst), H20, H21, H22
     (irm, vrex, dann 의 resid λ 0.25 대 stefan). H20 의 예측은 h21_preds.npz 안에 있다(cci_blockE, stefan).
  9. S-a 해석 규칙(부록 2.4)은 세 규칙 모두 n ∈ {3, 10} 에만 적용하고 n = 40 은 서술로 둔다. 계획서는 첫 규칙에만 n 범위를 적었고
     둘째·셋째 규칙에는 적지 않았다. 이 범위는 첫 계산(2026-09-30 03:25) 뒤에 확정한 코드의 해석이므로 '값을 본 뒤 정한 해석'으로 표기한다.
  10. 1.4 (b) 재분류 입력은 RECLASS_SPEC 의 실제 경로와 열 대응이다. 형식은 넷이다. delta2 = 블록 부트스트랩 CI(두 가중이 모두 유한하면
     4분 판정, 한 가중만 있으면 '4분 판정 불가(CI 규약 다름)', 둘 다 없으면 '판정 불가(CI 없음)'). rowci = (분할, 반복) 행 재표집 CI 만
     있는 결과(H23, H24, H25 원판): '규약 다름, 서술'. desc = 비율·R²·상관·개수·손익분기 n 형식(H27, F7 등): '규약 다름, 서술'.
     absent = 등록 검정이 산출되지 않은 가설(H26). 목록의 파일이나 열이 없으면 건너뛰지 않고 '원자료 없음' 행을 남긴다.
     원래 재표집 횟수(nboot_original)는 원 표의 메타(nboot 또는 args.nboot)에서 읽고, 메타나 키가 없으면 '확인하지 못함'으로 적는다.
  11. 9.4 δ_rel 의 외부 표(LGX 표, LGT·LGF·LGD 표)는 식별 열(target, mode, test_id, contrast, scope), CI 끝값(ci_lo, ci_hi, ci_lo_beq,
     ci_hi_beq, LGF 의 별칭 ci_lo_b·ci_hi_b), P0 RMSE 열(rmse_p0 등, 있을 때만)만 읽는다. verdict4(δ 0.5), verdict4_d10(δ 1.0),
     verdict4_rel 은 h39 가 CI 끝값으로 다시 계산하고 원 표의 판정 열과 Δ 열은 읽지 않는다. 표에 P0 RMSE 열이 없으면 LG 조각의 P0 RMSE 를
     쓰고 p0_source 열에 적는다. 대상 이름은 그대로 또는 '대상|모드'로 대응시키고, 모드가 여럿인데 표에 모드가 없으면 대응시키지 않는다.
     대응되지 않은 행은 '판정 불가(P0 RMSE 없음)'로 남기고 수를 메타(delta_rel_inputs)에 적는다. 풀 행(MEAN[…])의 판정 불가 조건
     (h40.strat_mean)은 CI 끝값만으로 다시 계산할 수 없으므로 pool_row 열로 표시한다. 파일 이름에 smoke, precheck, _pre, interim, count 가
     있는 표는 읽지 않는다.
  12. sc3 의 lgw_sc3.csv 는 판정용 rule-cap 행만 두고 RMSE 열(rmse_A, rmse_B)을 뺀다. 병기 대비(rule-P0, cap-P0)와 RMSE 열을 포함한
     전체 표는 <out>/sealed/lgw_sc3_full.csv 에 쓴다. 상한이 |A| 인 대상(러시아 W·E 등)의 cap-P0 는 LG 의 P1 − P0 | 전량과 식·분할·채점
     셀이 같아 점 추정이 같고, 레나·캐나다·Alaska(x)의 cap-P0 는 LG 의 P1 − P0 | n = 40 과 같은 형식이다(추출만 다름, lg_equiv 열).
     rule-cap 행의 rmse_B 도 같은 이유로 LG 의 P1 RMSE 와 같거나 같은 형식이다. sealed/ 는 LG 회수 뒤에만 연다.
  13. 산출 폴더는 부록 3.6 의 data/processed/lgw/ 하나다(8.4 (ii)의 로컬 집계는 data/processed/lgw/xenv/lgx/).

실행(ROOT, nice 10)
  python3 scripts/2_evaluation/h39_scenarios.py --mode units --out-dir data/processed/lgw --threads 1
  python3 scripts/2_evaluation/h39_scenarios.py --mode sens-unit --out-dir data/processed/lgw --threads 1
  python3 scripts/2_evaluation/h39_scenarios.py --mode sens-year --out-dir data/processed/lgw --threads 1
  python3 scripts/2_evaluation/h39_scenarios.py --mode sc3 --allow-local --out-dir data/processed/lgw --threads 1
  python3 scripts/2_evaluation/h39_scenarios.py --mode xenv-gate --xenv-local data/processed/lgw/xenv/lgx --xenv-ref <Rescale 집계 폴더> --xenv-prefix lgx_precheck
  (LG·LGX 회수 뒤) python3 scripts/2_evaluation/h39_scenarios.py --mode summarize --allow-local --threads 4 --lg-shards … --lgx-shards …
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import resource
import sys
import time
import warnings
from pathlib import Path


THREADS_MAX = 4                                                                                      # 부록 3.6 자원 규칙


def _peek_threads(av):
    """numpy 를 부르기 전에 스레드 수를 정한다. 허용 표지(--allow-local)가 없으면 1, 있으면 --threads 를 1–4 로 자른 값이다."""
    if "--allow-local" not in av:
        return "1"
    v = "1"
    for i, x in enumerate(av):
        if x == "--threads" and i + 1 < len(av):
            v = av[i + 1]
        elif x.startswith("--threads="):
            v = x.split("=", 1)[1]
    try:
        return str(max(1, min(int(v), THREADS_MAX)))
    except ValueError:
        return "1"


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = _peek_threads(sys.argv)                   # 직접 실행: 기존 값이 있어도 덮어쓴다
    else:
        os.environ.setdefault(_v, _peek_threads(sys.argv))         # 모듈로 불릴 때(시험)는 부르는 쪽의 설정을 둔다
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "scripts" / "3_deep_learning"
for _p in (str(DL), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, path):
    """동결 파일을 파일 경로에서 읽어 sys.modules 에 등록한다(이미 있으면 그 모듈을 쓴다)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("h40_label_grid", DL / "h40_label_grid.py")
X = _load("h42_label_grid_ext", DL / "h42_label_grid_ext.py")

import polar.h4_common as H4                                                                         # noqa: E402
from polar.h4_common import BlockStore, load_stores, seed_of, boot_delta_blocks                      # noqa: E402
from polar.m1_core import eval_mask, half_split_blocks                                               # noqa: E402
from polar import m1_stats as MS                                                                     # noqa: E402

# ================================================================ 고정값(부록 0.5, 2–5, 9절)
MODES = ("units", "sens-unit", "sens-year", "sc3", "bias-mae", "summarize", "xenv-gate")
MAIN4 = list(H.MAIN4)
IND5 = MAIN4 + [H.ALASKA]
M3 = ["Lena", "Canada", H.ALASKA]
SUB10 = list(H.SUB_AL) + list(H.SUB_OTHER)
MACROS7 = ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]
N_GRID = [0, 3, 10, 40, 160, 320, 1000, -1]
PLACE_N = [10, 40, 160]
DRAWS = 5
KAPPA = 10.0
DELTA = 0.5
DELTA_AUX = 1.0
REL_COEF = 0.02
NBOOT = 10000
LOCAL_NBOOT_MAX = 1000
LOC_DEG = 0.009
E5_COLS = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
PROTECT = ("lg", "lgx", "lgt", "lgd", "lgu", "lgf")
TAUS = (0.05, 0.025, 0.10, 0.20)
TAU_MAIN = 0.05
SC3_REPS = 20
NA_V = ("행 없음", "판정 불가")
LABEL_TYPE = {"Alaska": "점(ABoVE, GPR 다수), CALM 지점 평균 일부", "Lena": "점(ALLena, 대부분 단년 단일 방문)",
              "Canada": "점(ABoVE), CALM 지점 평균 일부", "Russia_W": "CALM 지점 다년 평균", "Russia_E": "CALM 지점 다년 평균",
              "Russia_C": "CALM 지점 다년 평균", "Greenland": "CALM 지점 다년 평균"}
BLIND = {"SC1w": "비맹검 부분 포함(h25b S3, b1, a2·b2 의 κ = 10 수축을 보았다). SC1w-P 는 등록 시점에 결과 존재(미열람)",
         "SC2w": "비맹검 부분 포함(a2_minn, a2 Alaska_f0. qOkSo 사전 점검 값 미열람)",
         "SC3w": "비맹검 부분 포함(b2 의 κ = 10 수축 곡선, RESEARCH_FRAME A.2 의 캐나다 E 수축 결과)",
         "L43": "비맹검 부분 포함(a2_spread, a2 블록 분산 추출, h29b)", "AK1w": "비맹검 부분 포함(h25b AL-1–AL-6, h25x)",
         "S-a": "해석식 민감도(판정 없음, 해석 규칙만 결과 전 고정)", "S-b": "해석식 민감도(판정 없음, 해석 규칙만 결과 전 고정)"}


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H39 마무리 부록 집계기(WRAPUP 3.6)")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out-dir", default="data/processed/lgw")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--allow-local", action="store_true", help="등록한 재표집 횟수(10,000)와 --threads 를 허용한다")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--ref-shards", default="results/rescale_lg_smoke/data/processed/lg/shards", help="units 대조용 조각 폴더(읽기 전용)")
    ap.add_argument("--ref-tags", default="lg_precheck,lg_smoke")
    ap.add_argument("--sa-splits", type=int, default=200)
    ap.add_argument("--sa-draws", type=int, default=5)
    ap.add_argument("--sb-splits", type=int, default=200)
    ap.add_argument("--sb-draws", type=int, default=5)
    ap.add_argument("--sc3-reps", type=int, default=SC3_REPS)
    ap.add_argument("--calm", default="data/raw/calm/PANGAEA_972777_CALM_ALT_NH.tab")
    ap.add_argument("--lg-shards", default="data/processed/lg/shards")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--lg-dir", default="data/processed/lg", help="LG 집계 표(lg_curve.csv) 폴더")
    ap.add_argument("--lgx-shards", default="data/processed/lgx/shards")
    ap.add_argument("--lgx-tag", default="lgx")
    ap.add_argument("--lgx-dir", default="data/processed/lgx")
    ap.add_argument("--lgd-dir", default="")
    ap.add_argument("--lgu-dir", default="")
    ap.add_argument("--lgt-dir", default="")
    ap.add_argument("--lgf-dir", default="")
    ap.add_argument("--xenv-local", default="")
    ap.add_argument("--xenv-ref", default="")
    ap.add_argument("--xenv-prefix", default="lgx_precheck", help="쉼표 목록. 대조할 표 이름의 앞부분")
    ap.add_argument("--xenv-name", default="ii", help="산출 파일 lgw_xenv_gate_<name>.csv")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def _abs(p):
    return Path(p) if os.path.isabs(str(p)) else ROOT / str(p)


def _real(p):
    return Path(os.path.realpath(str(p)))


def check_out_dir(path):
    """쓰기 보호: data/processed/{lg,lgx,lgt,lgd,lgu,lgf} 와 results/ 안이면 거부한다. 심볼릭 링크를 푼 실제 경로로 비교한다."""
    p = _real(_abs(path))
    proc = ROOT / "data" / "processed"
    for d in PROTECT:
        for q in {_real(proc / d), Path(os.path.abspath(str(proc / d)))}:
            if p == q or q in p.parents:
                raise SystemExit(f"[거부] 출력 폴더 {p} 는 보호 폴더 {q} 안이다(부록 3.6 쓰기 보호)")
    for res in {_real(ROOT / "results"), Path(os.path.abspath(str(ROOT / "results")))}:
        if p == res or res in p.parents:
            raise SystemExit(f"[거부] 출력 폴더 {p} 는 results/ 안이다(Rescale 결과 폴더에 쓰지 않는다)")
    return p


def finalize(a):
    a.OUT = check_out_dir(a.out_dir)
    a.PROC = _abs(a.data_dir)
    a.PERMIT = bool(a.allow_local)
    a.nboot_asked = int(a.nboot)
    a.threads_asked = int(a.threads)
    if not a.PERMIT:
        a.threads = 1
        a.nboot = min(int(a.nboot), LOCAL_NBOOT_MAX)
    else:
        a.threads = max(1, min(int(a.threads), THREADS_MAX))
    a.CAPPED = int(a.nboot) < int(a.nboot_asked) or int(a.nboot) < NBOOT
    return a


# ================================================================ 공통 도우미
def ns_eq():
    """h42.stats_row·pool_rows 가 쓰는 인자 묶음(δ 0.5, 보조 1.0)."""
    return argparse.Namespace(delta_eq=DELTA, delta_eq_aux=DELTA_AUX)


def file_sha(path, n=16):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except OSError:
        return "NA"


def loc1km(lat, lon):
    """1 km 위치 색인(부록 2.2, LG 6B.3)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    ky = np.floor(lat / LOC_DEG).astype(np.int64)
    phi = (ky + 0.5) * LOC_DEG
    kx = np.floor(lon * np.cos(np.radians(phi)) / LOC_DEG).astype(np.int64)
    return ky, kx


def _inv(M):
    return np.unique(M, axis=0, return_inverse=True)[1].ravel().astype(np.int64)


class Codes:
    """행별 정수 부호: 1 km 위치, ERA5 조합, 최근접 0.1° 격자, 0.01° 반올림 위치, 블록."""

    def __init__(self, df):
        ky, kx = loc1km(df.lat.values, df.lon.values)
        self.loc = _inv(np.c_[ky, kx])
        E = np.round(df[E5_COLS].values.astype(float), 6)
        self.era5 = _inv(np.where(np.isfinite(E), E, -9999.0))
        self.grid01 = _inv(np.c_[np.round(df.lat.values * 10.0), np.round(df.lon.values * 10.0)])
        self.loc001 = _inv(np.c_[np.round(df.lat.values, 2), np.round(df.lon.values, 2)])
        self.block = pd.factorize(pd.Series(df.block.values).astype(str))[0].astype(np.int64)


def nu(v):
    return int(len(np.unique(v))) if len(v) else 0


def h40_ns(a, splits=5):
    return H.parse_args(["--data-dir", str(a.PROC), "--subregion-map", str(a.subregion_map), "--splits", str(int(splits)), "--threads", "1"])


def make_data(a, splits=5):
    return H.Data(h40_ns(a, splits))


def with_splits(D, K):
    """분할 범위만 다른 자료 객체(자료와 원천 색인은 공유한다)."""
    D2 = copy.copy(D)
    D2.args = copy.copy(D.args)
    D2.args.SPLITS = list(range(1, int(K) + 1))
    D2._split = {}
    return D2


def lg_splits(D, t):
    """h40.enumerate_units 와 같은 분할 목록: 중복 분할, 채점 셀 또는 A 셀이 없는 분할 제외. 유효 분할이 있으면 무효 분할 제외."""
    info = D.split_structure(t)
    any_valid = any(v["valid"] for v in info.values())
    out = []
    for sp, v in sorted(info.items()):
        if v["dup_of"] >= 0 or v["n_eval"] == 0 or v["n_A"] == 0:
            continue
        if not v["valid"] and any_valid:
            continue
        out.append(int(sp))
    return out


def ab_rows(D, t, sp):
    """h40.build_ctx 와 같은 A 행 순서와 채점 행(B 블록의 eval_mask 행)."""
    df = D.df
    t_idx = D.target_idx(t)
    A_idx, B_idx = half_split_blocks(df, t_idx, sp)
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    return A_idx, evB


def source_E0(D, t, mode):
    _, _, src, comp = D.source_idx(t, mode)
    return float(H.ls_E(D.df.y.values[src], D.df.s.values[src])), comp


def shrink(E_ls, E0, n, kappa=KAPPA):
    if n <= 0 or not np.isfinite(E_ls):
        return float(E0)
    return float((n * E_ls + kappa * E0) / (n + kappa))


def rmse(pred, y):
    e = np.asarray(pred, float) - np.asarray(y, float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def qstats(v, prefix=""):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return {f"{prefix}count": 0, f"{prefix}mean": np.nan, f"{prefix}p10": np.nan, f"{prefix}p50": np.nan, f"{prefix}p90": np.nan}
    return {f"{prefix}count": int(len(v)), f"{prefix}mean": float(v.mean()), f"{prefix}p10": float(np.percentile(v, 10)),
            f"{prefix}p50": float(np.percentile(v, 50)), f"{prefix}p90": float(np.percentile(v, 90))}


def update_tests(out, rows):
    """lgw_tests.csv 의 같은 test_id 행을 새 행으로 바꾼다(모드마다 따로 실행하므로 합친다)."""
    p = Path(out) / "lgw_tests.csv"
    new = pd.DataFrame(rows)
    if p.exists() and p.stat().st_size > 1:
        old = pd.read_csv(p, dtype=str, keep_default_na=False)
        old = old[~old.test_id.isin(set(new.test_id.astype(str)))]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(p, index=False)
    return p


def update_meta(a, mode, info):
    p = a.OUT / "lgw_meta.json"
    meta = {}
    if p.exists():
        try:
            meta = json.loads(p.read_text())
        except ValueError:
            meta = {}
    import scipy
    import sklearn
    info = dict(info)
    info.update(time=time.strftime("%Y-%m-%d %H:%M:%S %Z"), argv=a.ARGV, nboot=int(a.nboot), nboot_asked=int(a.nboot_asked),
                nboot_capped=bool(a.CAPPED), threads=int(a.threads), threads_asked=int(getattr(a, "threads_asked", a.threads)),
                env_threads={k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                allow_local=bool(a.PERMIT),
                code_sha_h39=file_sha(__file__), code_sha_h40=H.code_sha(), code_sha_h42=X.code_sha_x(),
                versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                              sklearn=sklearn.__version__),
                max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1), nice=os.nice(0))
    if mode not in ("sc3", "summarize"):                                    # 재표집이 없는 모드
        info.update(nboot=None, nboot_asked=None, nboot_capped=None, resampling="없음")
    meta[mode] = info
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    return p


def mark_nboot(df, a):
    if df is not None and len(df):
        df["nboot"] = int(a.nboot)
        df["nboot_capped"] = bool(a.CAPPED)
    return df


def input_hashes(a):
    fs = [a.PROC / "fidelity_base_v3.csv", a.PROC / "e5_soil_tdd_v3.csv", a.PROC / str(a.subregion_map)]
    return {str(f.relative_to(ROOT)) if str(f).startswith(str(ROOT)) else str(f): file_sha(f, 64) for f in fs}


# ================================================================ units(2.2–2.3)
def units_region_table(D, codes):
    df = D.df
    rows = []
    for t in MACROS7 + SUB10:
        try:
            idx = D.target_idx(t)
        except ValueError:
            continue
        if not len(idx):
            continue
        y = df.y.values[idx]
        L = codes.loc[idx]
        g = pd.DataFrame(dict(l=L, y=y)).groupby("l").y
        sz, var = g.size(), g.var(ddof=1)
        m = sz >= 5
        dof = float((sz[m] - 1).sum())
        sd = float(np.sqrt(((sz[m] - 1) * var[m]).sum() / dof)) if dof > 0 else np.nan
        par = D.parent_of(t)
        rows.append(dict(target=t, kind="macro" if t in MACROS7 else "subregion", parent=par, n_rows=int(len(idx)), n_loc_1km=nu(L),
                         n_era5=nu(codes.era5[idx]), n_era5_grid01=nu(codes.grid01[idx]), n_block=nu(codes.block[idx]),
                         n_loc_001=nu(codes.loc001[idx]), sd_in_1km_cm=sd, n_loc_sd=int(m.sum()), rows_per_loc=float(len(idx) / max(nu(L), 1)),
                         label_type=LABEL_TYPE.get(par, "")))
    return pd.DataFrame(rows)


class DrawBook:
    """LG 의 라벨 추출을 다시 만든다. A 행은 h40.build_ctx 와 같은 순서(half_split_blocks 의 idx[inA])다."""

    def __init__(self, D, codes):
        self.D, self.codes, self._A = D, codes, {}

    def A(self, t, sp):
        if (t, sp) not in self._A:
            self._A[(t, sp)] = ab_rows(self.D, t, sp)[0]
        return self._A[(t, sp)]

    def sel(self, t, m, sp, n, d, placement):
        A_idx = self.A(t, sp)
        nA = len(A_idx)
        if placement == "block":
            return H.draw_blocks(t, m, sp, n, d, self.D.df.block.values[A_idx])
        return H.draw_cells(t, m, sp, n, d, nA)

    def counts(self, t, m, sp, n, d, placement):
        A_idx = self.A(t, sp)
        s = self.sel(t, m, sp, n, d, placement)
        r = A_idx[s]
        return dict(nA=int(len(A_idx)), n_rows=int(len(s)), n_loc_1km=nu(self.codes.loc[r]), n_era5=nu(self.codes.era5[r]),
                    n_block=nu(self.codes.block[r]))


def units_draws(D, codes, targets=None):
    B = DrawBook(D, codes)
    targets = targets if targets is not None else H.default_targets(["i", "x"])
    det = []
    for t, m in targets:
        for sp in lg_splits(D, t):
            nA = len(B.A(t, sp))
            for placement, cells in (("cell", H.cells_of(N_GRID, DRAWS, nA)), ("block", H.cells_of(PLACE_N, DRAWS, nA))):
                for n, d in cells:
                    det.append(dict(target=t, mode=m, split=sp, n=int(n), draw=int(d), placement=placement, **B.counts(t, m, sp, n, d, placement)))
    det = pd.DataFrame(det)
    if not len(det):
        return det, det
    agg = det.groupby(["target", "mode", "n", "placement"], as_index=False).agg(
        n_splits=("split", "nunique"), n_units=("draw", "size"), nA_mean=("nA", "mean"),
        **{f"{c}_{f}": (c, f) for c in ("n_rows", "n_loc_1km", "n_era5", "n_block") for f in ("mean", "min", "max")})
    return det, agg


REF_COLS = ["target", "mode", "split", "method", "learner", "placement", "n", "draw", "n_lab", "n_blocks_lab"]


def units_check(D, codes, ref_dir, tags):
    """LG 사전 점검·스모크 CPU 조각과 키별 대조. 키 열과 n_lab, n_blocks_lab 만 읽는다(usecols)."""
    B = DrawBook(D, codes)
    rows, bad = [], []
    for tg in tags:
        for p in sorted(Path(ref_dir).glob(f"{tg}__cpu__*_runs.csv")):
            r = pd.read_csv(p, usecols=REF_COLS, dtype=dict(target=str, mode=str, method=str, learner=str, placement=str))
            keys = r[["target", "mode", "split", "n", "draw", "placement"]].drop_duplicates()
            got = {}
            for t, m, sp, n, d, pl in keys.itertuples(index=False, name=None):
                got[(t, m, int(sp), int(n), int(d), pl)] = B.counts(t, m, int(sp), int(n), int(d), pl)
            lab = np.array([got[(t, m, int(sp), int(n), int(d), pl)]["n_rows"] for t, m, sp, n, d, pl in
                            r[["target", "mode", "split", "n", "draw", "placement"]].itertuples(index=False, name=None)])
            nbk = np.array([got[(t, m, int(sp), int(n), int(d), pl)]["n_block"] for t, m, sp, n, d, pl in
                            r[["target", "mode", "split", "n", "draw", "placement"]].itertuples(index=False, name=None)])
            m_lab = r.n_lab.values.astype(int) != lab
            m_blk = r.n_blocks_lab.values.astype(int) != nbk
            if (m_lab | m_blk).any():
                bad.append(r[m_lab | m_blk].assign(n_lab_regen=lab[m_lab | m_blk], n_block_regen=nbk[m_lab | m_blk], file=p.name))
            rows.append(dict(file=p.name, tag=tg, n_rows=int(len(r)), n_keys_draw=int(len(keys)),
                             n_values=",".join(str(v) for v in sorted(r.n.unique())), placements=",".join(sorted(r.placement.unique())),
                             n_mismatch_n_lab=int(m_lab.sum()), n_mismatch_n_blocks=int(m_blk.sum()), passed=bool(not (m_lab | m_blk).any())))
    return pd.DataFrame(rows), (pd.concat(bad, ignore_index=True) if bad else pd.DataFrame())


def run_units(a):
    D = make_data(a, 5)
    codes = Codes(D.df)
    reg = units_region_table(D, codes)
    det, agg = units_draws(D, codes)
    chk, bad = units_check(D, codes, _abs(a.ref_shards), [v for v in a.ref_tags.split(",") if v])
    reg.to_csv(a.OUT / "lgw_label_units.csv", index=False)
    agg.to_csv(a.OUT / "lgw_label_units_draws.csv", index=False)
    det.to_csv(a.OUT / "lgw_label_units_draws_detail.csv", index=False)
    chk.to_csv(a.OUT / "lgw_label_units_check.csv", index=False)
    if len(bad):
        bad.to_csv(a.OUT / "lgw_label_units_check_mismatch.csv", index=False)
    n_bad = int((~chk.passed).sum()) if len(chk) else -1
    update_meta(a, "units", dict(inputs=input_hashes(a), ref_shards=str(a.ref_shards), ref_files=list(chk.file) if len(chk) else [],
                                 n_detail=int(len(det)), n_check_files=int(len(chk)), n_check_failed=n_bad,
                                 subregion_src=getattr(D, "sub_src", ""), subregion_kmeans_diff=int(getattr(D, "sub_kmeans_diff", 0))))
    print(f"[units] 지역 {len(reg)} · 추출 {len(det)} · 대조 조각 {len(chk)}(불일치 조각 {n_bad}) → {a.OUT}", flush=True)
    return dict(region=reg, draws=agg, detail=det, check=chk)


# ================================================================ sens-unit(2.4 S-a)
def _ls(y, s):
    return float(H.ls_E(np.asarray(y, float), np.asarray(s, float)))


def loc_means(codes_loc, y, s):
    """행을 1 km 위치로 묶은 평균 ALT·평균 s. 위치 순서는 부호의 오름차순((ky, kx) 사전식)이다."""
    d = pd.DataFrame(dict(l=codes_loc, y=y, s=s)).groupby("l", sort=True)
    return d.y.mean().values.astype(float), d.s.mean().values.astype(float)


def sa_region(D, codes, t, E0, n_list, draws, ref_only=False):
    """분할별 추출 평균 Δ(P1 − P0), Δ(P2 − P0)(행 조건, 셀 평균 조건, 채점 둘)."""
    df = D.df
    out = []
    for sp in lg_splits(D, t):
        A_idx, evB = ab_rows(D, t, sp)
        yA, sA = df.y.values[A_idx], df.s.values[A_idx]
        yB, sB = df.y.values[evB], df.s.values[evB]
        ybL, sbL = loc_means(codes.loc[evB], yB, sB)
        ylA, slA = loc_means(codes.loc[A_idx], yA, sA)
        nA, nL = len(A_idx), len(ylA)
        r0, r0L = rmse(E0 * sB, yB), rmse(E0 * sbL, ybL)
        for n in n_list:
            rec = dict(target=t, split=sp, n=int(n), nA=nA, nLocA=nL, n_eval=int(len(evB)), n_evalLoc=int(len(ybL)))
            for cond, ok, seed_tag, Y, S, N in (("row", nA > n, "lgw-sa-row", yA, sA, nA), ("cell", nL > n, "lgw-sa-cell", ylA, slA, nL)):
                if ref_only and cond == "cell":
                    continue
                if not ok:
                    rec.update({f"{cond}_excluded": True})
                    continue
                v = {k: [] for k in ("p1", "p2", "p1L", "p2L")}
                for d in range(draws):
                    sel = np.random.RandomState(seed_of(seed_tag, t, sp, n, d)).choice(N, int(n), replace=False)
                    El = _ls(Y[sel], S[sel]); En = shrink(El, E0, n)
                    E2 = El if np.isfinite(El) else E0
                    v["p1"].append(rmse(En * sB, yB) - r0); v["p2"].append(rmse(E2 * sB, yB) - r0)
                    v["p1L"].append(rmse(En * sbL, ybL) - r0L); v["p2L"].append(rmse(E2 * sbL, ybL) - r0L)
                rec.update({f"{cond}_excluded": False, **{f"d_{cond}_{k}": float(np.mean(vv)) for k, vv in v.items()}})
            out.append(rec)
    return pd.DataFrame(out)


def sa_summary(sp_df, ref_df):
    rows = []
    for (t, n), g in sp_df.groupby(["target", "n"]):
        base = dict(target=t, n=int(n), n_splits_total=int(len(g)),
                    n_splits_excl_row=int(g.get("row_excluded", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()),
                    n_splits_excl_cell=int(g.get("cell_excluded", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum()))
        for k in ("p1", "p2", "p1L", "p2L"):
            cr, cc = f"d_row_{k}", f"d_cell_{k}"
            if cr in g and cc in g:
                rows.append(dict(base, quantity=f"unit_{k}", **qstats((g[cc] - g[cr]).values)))
            for c_, q in ((cr, f"row_{k}"), (cc, f"cell_{k}")):
                if c_ in g:
                    rows.append(dict(base, quantity=q, **qstats(g[c_].values)))
    for (t, n), g in ref_df.groupby(["target", "n"]):
        for k in ("p1", "p2"):
            c_ = f"d_row_{k}"
            if c_ in g:
                rows.append(dict(target=t, n=int(n), n_splits_total=int(len(g)), n_splits_excl_row=int(g.row_excluded.astype(bool).sum()),
                                 n_splits_excl_cell=np.nan, quantity=f"ref_row_{k}", **qstats(g[c_].values)))
    return pd.DataFrame(rows)


def _q(summ, t, n, qty, col="p50"):
    r = summ[(summ.target == t) & (summ.n == n) & (summ.quantity == qty)]
    return float(r[col].iloc[0]) if len(r) and r["count"].iloc[0] > 0 else np.nan


def sa_verdict(summ, regions=("Lena", "Canada", "Alaska"), ns=(3, 10), ref="Russia_W"):
    """부록 2.4 (S-a) 해석 규칙(결과 전 고정). 주 대비는 P1 − P0 의 Δ_unit 이다."""
    med = {(t, n): _q(summ, t, n, "unit_p1") for t in regions for n in ns}
    lines, flags = [], []
    fin = {k: v for k, v in med.items() if np.isfinite(v)}
    if len(fin) < len(med):
        flags.append(f"값 없음 {len(med) - len(fin)}칸")
    if fin and all(abs(v) <= 0.5 for v in fin.values()) and len(fin) == len(med):
        lines.append("라벨 단위(점 대 1 km 위치 평균)는 소수 라벨 재보정의 이득을 바꾸지 않았다")
    else:
        neg = [f"{t} n = {n}, {v:+.2f} cm" for (t, n), v in fin.items() if v < -0.5]
        pos = [f"{t} n = {n}, {v:+.2f} cm" for (t, n), v in fin.items() if v > 0.5]
        if neg:
            lines.append("점 라벨 조건에서는 소수 라벨 재보정의 이득이 1 km 위치 평균 조건보다 작았다(" + "; ".join(neg)
                         + "). 배포 지침의 라벨 단위를 '1 km 위치 평균'으로 적는다")
        if pos:
            lines.append("1 km 위치 평균 조건에서는 소수 라벨 재보정의 이득이 점 라벨 조건보다 작았다(" + "; ".join(pos) + ")")
        if not neg and not pos and fin:
            lines.append("세 지역 가운데 값이 없는 칸이 있어 첫 문장의 조건(세 지역 모두)을 확인하지 못했다")
    sg = {t: np.sign(np.nanmedian([med[(t, n)] for n in ns])) for t in regions}
    if len({v for v in sg.values() if v != 0 and np.isfinite(v)}) > 1:
        flags.append("지역 사이 Δ_unit 방향이 달라 문장을 지역별로 쓴다")
    over = []
    for t in ("Lena", "Canada"):
        for n in ns:
            c10, c90 = _q(summ, t, n, "cell_p1", "p10"), _q(summ, t, n, "cell_p1", "p90")
            r10, r90 = _q(summ, t, n, "row_p1", "p10"), _q(summ, t, n, "row_p1", "p90")
            w10, w90 = _q(summ, ref, n, "ref_row_p1", "p10"), _q(summ, ref, n, "ref_row_p1", "p90")
            if not all(np.isfinite([c10, c90, r10, r90, w10, w90])):
                continue
            cell_ov = (c10 <= w90) and (w10 <= c90)
            row_ov = (r10 <= w90) and (w10 <= r90)
            if cell_ov and not row_ov:
                over.append(f"{t} n = {n}")
    if over:
        lines.append("지역 사이의 재보정 이득 차이 가운데 일부는 라벨 단위로 설명된다(" + ", ".join(over) + ")")
    stat = "; ".join(f"{t} n={n}: 중앙값 Δ_unit {v:+.2f}" for (t, n), v in med.items())
    d40 = {t: _q(summ, t, 40, "unit_p1") for t in regions}
    if any(np.isfinite(v) for v in d40.values()):
        stat += "; 서술(n = 40, 규칙 적용 밖) " + ", ".join(f"{t} {v:+.2f}" for t, v in d40.items() if np.isfinite(v))
    flags.append("해석 규칙은 세 규칙 모두 n ∈ {3, 10} 에만 적용했다(규약 9, 값을 본 뒤 정한 해석). n = 40 은 서술")
    return dict(test_id="S-a", role="해석(민감도)", scope="verdict", verdict=" ".join(s + "." for s in lines) if lines else "문장 없음",
                stat=stat, flags="; ".join(flags), blind=BLIND["S-a"])


def run_sens_unit(a):
    D5 = make_data(a, 5)
    D = with_splits(D5, a.sa_splits)
    codes = Codes(D.df)
    parts, refs, e0 = [], [], {}
    for t in ("Lena", "Canada", "Alaska"):
        E0, comp = source_E0(D, t, "x"); e0[t] = dict(E0=E0, **comp)
        parts.append(sa_region(D, codes, t, E0, (3, 10, 40), a.sa_draws))
    E0, comp = source_E0(D, "Russia_W", "x"); e0["Russia_W"] = dict(E0=E0, **comp)
    refs.append(sa_region(D, codes, "Russia_W", E0, (3, 10), a.sa_draws, ref_only=True))
    sp_df = pd.concat(parts, ignore_index=True); ref_df = pd.concat(refs, ignore_index=True)
    summ = sa_summary(sp_df, ref_df)
    v = sa_verdict(summ)
    sp_df.to_csv(a.OUT / "lgw_sens_unit_splits.csv", index=False)
    ref_df.to_csv(a.OUT / "lgw_sens_unit_ref_splits.csv", index=False)
    summ.to_csv(a.OUT / "lgw_sens_unit.csv", index=False)
    update_tests(a.OUT, [dict(v, mode="sens-unit", nboot="", nboot_capped="")])
    update_meta(a, "sens-unit", dict(inputs=input_hashes(a), E0=e0, n_splits_asked=int(a.sa_splits), draws=int(a.sa_draws),
                                     n_splits_used={t: int(sp_df[sp_df.target == t].split.nunique()) for t in sp_df.target.unique()}))
    print(f"[sens-unit] 분할 행 {len(sp_df)} · 요약 {len(summ)} → {a.OUT}", flush=True)
    return dict(splits=sp_df, ref=ref_df, summary=summ, verdict=v)


# ================================================================ sens-year(2.4 S-b)
def read_calm(path):
    """CALM 원자료: 머리말(/* … */) 뒤의 탭 구분 표. 부등호 표기, 빈 값, inactive 행을 뺀다."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    end = next(i for i, ln in enumerate(lines) if ln.strip() == "*/")
    from io import StringIO
    raw = pd.read_csv(StringIO("\n".join(lines[end + 1:])), sep="\t", dtype=str, keep_default_na=False)
    ald = raw["ALD [cm]"].astype(str).str.strip()
    ineq = ald.str.contains("[<>]", regex=True)
    inactive = raw["Sample comment"].astype(str).str.lower().str.contains("inactive")
    val = pd.to_numeric(ald.where(~ineq), errors="coerce")
    d = pd.DataFrame(dict(event=raw["Event"], lat=pd.to_numeric(raw["Latitude"], errors="coerce"), lon=pd.to_numeric(raw["Longitude"], errors="coerce"),
                          year=pd.to_numeric(raw["Date/Time"].str[:4], errors="coerce"), ald=val, ineq=ineq, inactive=inactive))
    stats = dict(n_rows=int(len(d)), n_ineq=int(ineq.sum()), n_ineq_gt=int(ald.str.contains(">").sum()), n_ineq_lt=int(ald.str.contains("<").sum()),
                 n_inactive=int(inactive.sum()), n_empty=int((ald == "").sum()))
    ok = d.ald.notna() & ~d.ineq & ~d.inactive & d.year.notna()
    stats["n_valid"] = int(ok.sum())
    return d, d[ok].copy(), stats


def calm_match(df_t, calm_all, calm_ok, tol=0.005):
    """v3 행(대상 지역)과 CALM 지점의 좌표 대응(허용 차 위도·경도 각 0.005°). 반환 = loc_id → 유효 연도 값 dict 와 집계."""
    ev = calm_all.groupby("event").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    cand = {}
    for i, (lid, la, lo) in enumerate(zip(df_t.loc_id.values, df_t.lat.values, df_t.lon.values)):
        m = (np.abs(ev.lat.values - la) <= tol + 1e-12) & (np.abs(ev.lon.values - lo) <= tol + 1e-12)
        cand[int(lid)] = list(ev.event.values[m])
    site_cells = {}
    for lid, evs in cand.items():
        for e in evs:
            site_cells.setdefault(e, []).append(lid)
    vals, qa = {}, []
    n_multi = n_none = n_noyear = 0
    for lid, evs in cand.items():
        if len(evs) == 0:
            n_none += 1; continue
        if len(evs) > 1 or len(site_cells[evs[0]]) > 1:
            n_multi += 1; continue
        g = calm_ok[calm_ok.event == evs[0]]
        if not len(g):
            n_noyear += 1; continue
        yv = g.groupby("year").ald.mean()
        vals[lid] = dict(event=evs[0], years=yv.index.values.astype(int), values=yv.values.astype(float))
        qa.append(dict(loc_id=lid, event=evs[0], n_years=int(len(yv)), v3_minus_mean=np.nan))
    agg = dict(n_cells=int(len(cand)), n_single=int(len(vals)), n_multi=int(n_multi), n_unmatched=int(n_none), n_no_valid_year=int(n_noyear))
    return vals, pd.DataFrame(qa), agg


def sb_region(D, t, E0, vals, n_list, draws):
    df = D.df
    out = []
    for sp in lg_splits(D, t):
        A_idx, evB = ab_rows(D, t, sp)
        yA, sA, lidA = df.y.values[A_idx], df.s.values[A_idx], df.loc_id.values[A_idx]
        yB, sB = df.y.values[evB], df.s.values[evB]
        nA = len(A_idx)
        for n in n_list:
            if nA <= n:
                out.append(dict(target=t, split=sp, n=int(n), nA=nA, excluded=True)); continue
            dy, rep = [], []
            for d in range(draws):
                sel = np.random.RandomState(seed_of("lgw-sb-row", t, sp, n, d)).choice(nA, int(n), replace=False)
                y1 = yA[sel].copy(); k = 0
                for j, i in enumerate(sel):
                    lid = int(lidA[i])
                    if lid in vals:
                        v = vals[lid]
                        yr = np.random.RandomState(seed_of("lgw-sb", t, sp, n, d, lid)).choice(v["years"])
                        y1[j] = float(v["values"][list(v["years"]).index(int(yr))]); k += 1
                Em = shrink(_ls(yA[sel], sA[sel]), E0, n); Es = shrink(_ls(y1, sA[sel]), E0, n)
                dy.append(rmse(Es * sB, yB) - rmse(Em * sB, yB)); rep.append(k / float(n))
            out.append(dict(target=t, split=sp, n=int(n), nA=nA, excluded=False, d_year_p1=float(np.mean(dy)), frac_replaced=float(np.mean(rep))))
    return pd.DataFrame(out)


def sb_verdict(summ, regions=("Russia_W", "Russia_E"), ns=(3, 10)):
    med = {(t, n): (float(summ[(summ.target == t) & (summ.n == n)].p50.iloc[0]) if len(summ[(summ.target == t) & (summ.n == n)]) else np.nan)
           for t in regions for n in ns}
    fin = {k: v for k, v in med.items() if np.isfinite(v)}
    lines = []
    if len(fin) == len(med) and all(abs(v) <= 0.5 for v in fin.values()):
        lines.append("단년 관측 라벨에서도 라벨 3–10개의 재보정 이득이 유지되었다")
    else:
        pos = [f"{t} n = {n}, {v:+.2f} cm" for (t, n), v in fin.items() if v > 0.5]
        neg = [f"{t} n = {n}, {v:+.2f} cm" for (t, n), v in fin.items() if v < -0.5]
        if pos:
            lines.append("단년 라벨에서는 이득이 줄었다(" + "; ".join(pos) + "). 배포 지침의 '라벨'을 '다년 평균 지점'으로 한정한다")
        if neg:
            lines.append("단년 라벨의 중앙값 Δ_year 가 −0.5 cm 미만이었다(" + "; ".join(neg) + ")")
        if len(fin) < len(med):
            lines.append(f"값이 없는 칸이 {len(med) - len(fin)}개다")
    stat = "; ".join(f"{t} n={n}: 중앙값 Δ_year {v:+.2f}" for (t, n), v in med.items())
    return dict(test_id="S-b", role="해석(민감도)", scope="verdict", verdict=" ".join(s + "." for s in lines) if lines else "문장 없음",
                stat=stat, flags="", blind=BLIND["S-b"])


def run_sens_year(a):
    D5 = make_data(a, 5)
    D = with_splits(D5, a.sb_splits)
    calm_all, calm_ok, cstats = read_calm(_abs(a.calm))
    parts, match, e0 = [], {}, {}
    for t in ("Russia_W", "Russia_E"):
        idx = D.target_idx(t)
        dt = D.df.iloc[idx]
        vals, qa, agg = calm_match(dt, calm_all, calm_ok)
        if len(qa):
            v3 = dict(zip(dt.loc_id.values.astype(int), dt.y.values.astype(float)))
            qa["v3_minus_mean"] = [v3[int(r.loc_id)] - float(np.mean(vals[int(r.loc_id)]["values"])) for r in qa.itertuples()]
        E0, comp = source_E0(D, t, "x"); e0[t] = dict(E0=E0, **comp)
        match[t] = dict(agg, qa_abs_diff_median=float(np.median(np.abs(qa.v3_minus_mean))) if len(qa) else np.nan,
                        qa_abs_diff_max=float(np.max(np.abs(qa.v3_minus_mean))) if len(qa) else np.nan,
                        qa_n_within_1cm=int((np.abs(qa.v3_minus_mean) <= 1.0).sum()) if len(qa) else 0,
                        years_median=float(np.median(qa.n_years)) if len(qa) else np.nan)
        parts.append(sb_region(D, t, E0, vals, (3, 10), a.sb_draws))
    sp_df = pd.concat(parts, ignore_index=True)
    rows = []
    for (t, n), g in sp_df.groupby(["target", "n"]):
        gg = g[~g.excluded.astype(bool)]
        rows.append(dict(target=t, n=int(n), n_splits_total=int(len(g)), n_splits_excluded=int(g.excluded.astype(bool).sum()),
                         frac_replaced_mean=float(gg.frac_replaced.mean()) if len(gg) else np.nan, **qstats(gg.d_year_p1.values if len(gg) else [])))
    summ = pd.DataFrame(rows)
    mt = pd.DataFrame([dict(target=t, **v) for t, v in match.items()])
    v = sb_verdict(summ)
    sp_df.to_csv(a.OUT / "lgw_sens_year_splits.csv", index=False)
    summ.to_csv(a.OUT / "lgw_sens_year.csv", index=False)
    mt.to_csv(a.OUT / "lgw_sens_year_match.csv", index=False)
    update_tests(a.OUT, [dict(v, mode="sens-year", nboot="", nboot_capped="")])
    update_meta(a, "sens-year", dict(inputs=dict(input_hashes(a), calm=file_sha(_abs(a.calm), 64)), calm=cstats, match=match, E0=e0,
                                     n_splits_asked=int(a.sb_splits), draws=int(a.sb_draws)))
    print(f"[sens-year] 분할 행 {len(sp_df)} · 요약 {len(summ)} → {a.OUT}", flush=True)
    return dict(splits=sp_df, summary=summ, match=mt, verdict=v)


# ================================================================ sc3(3.5 SC3w)
def jk_se(y, s):
    """선택 라벨의 잭나이프 표준오차(log E, 수축 전 최소제곱 계수)."""
    y = np.asarray(y, float); s = np.asarray(s, float); n = len(y)
    if n < 2:
        return np.inf
    E = np.array([H.ls_E(np.delete(y, i), np.delete(s, i)) for i in range(n)], float)
    if not np.all(np.isfinite(E)) or np.any(E <= 0):
        return np.inf
    le = np.log(E)
    return float(np.sqrt((n - 1) / n * np.sum((le - le.mean()) ** 2)))


def sc3_stages(nA):
    cap = int(min(40, nA))
    return [k for k in (3, 10) if k < cap] + [cap], cap


def sc3_stop(yA, sA, perm, stages, tau):
    """멈춘 단계. 마지막 단계(상한) 전의 단계에서 SE_jk ≤ τ 이면 그 단계에서 멈춘다."""
    for k in stages[:-1]:
        lab = perm[:k]
        if jk_se(yA[lab], sA[lab]) <= tau:
            return int(k)
    return int(stages[-1])


def sc3_class(ratio, mean_delta, hi, hib, any_worse):
    """SC3w 판정(부록 3.5): 지지 · 부분(정밀도 부족) · 규칙 비작동 · 기각."""
    c1 = np.isfinite(ratio) and ratio <= 0.5
    c2 = np.isfinite(mean_delta) and mean_delta <= 0.3
    c3 = np.isfinite(hi) and np.isfinite(hib) and hi < 0.5 and hib < 0.5
    c4 = not any_worse
    if c1 and c2 and c3 and c4:
        return "지지"
    if c1 and c2 and c4:
        return "부분(정밀도 부족)"
    if np.isfinite(ratio) and ratio >= 0.9:
        return "규칙 비작동"
    return "기각"


def sc3_target(D, t, mode, E0, reps, taus=TAUS):
    """대상 하나의 분할별 저장소(키 = (이름, 반복))와 멈춘 단계 기록."""
    df = D.df
    stores, recs = {}, []
    for sp in lg_splits(D, t):
        A_idx, evB = ab_rows(D, t, sp)
        nA = len(A_idx)
        if nA < 4 or not len(evB):
            continue
        yA, sA = df.y.values[A_idx], df.s.values[A_idx]
        yB, sB = df.y.values[evB], df.s.values[evB]
        stages, cap = sc3_stages(nA)
        st = BlockStore(f"{t}|{mode}", sp, df.block.values[evB])
        st.add(("P0", 0), yB, E0 * sB)
        for r in range(reps):
            perm = np.random.RandomState(seed_of("lgw-sc3", t, mode, sp, r)).permutation(nA)
            Ecap = shrink(_ls(yA[perm[:cap]], sA[perm[:cap]]), E0, cap)
            st.add(("P1cap", r), yB, Ecap * sB)
            for tau in taus:
                k = sc3_stop(yA, sA, perm, stages, tau)
                Ek = shrink(_ls(yA[perm[:k]], sA[perm[:k]]), E0, k)
                st.add((f"P1rule@{tau:g}", r), yB, Ek * sB)
                recs.append(dict(target=t, mode=mode, split=sp, rep=r, tau=float(tau), stop=k, cap=cap, nA=nA))
        stores[sp] = st
    return stores, pd.DataFrame(recs)


def _bd(stores, A, B, nboot, seed):
    return boot_delta_blocks(stores, A, B, nboot=nboot, seed=seed, rep_fn=lambda k: k[1], return_dist=True)


def sc3_contrasts(name, stores, reps, nboot, tau):
    """Δ_SC3 = rule − cap, rule − P0, cap − P0 의 지역 통계(두 가중, 같은 재표집)."""
    rule = [(f"P1rule@{tau:g}", r) for r in range(reps)]
    cap = [("P1cap", r) for r in range(reps)]
    p0 = [("P0", 0)]
    nb_union = len(set().union(*[set(st.blocks.tolist()) for st in stores.values()])) if stores else 0
    out = {}
    for lab, A, B in (("rule-cap", rule, cap), ("rule-P0", rule, p0), ("cap-P0", cap, p0)):
        r = _bd(stores, A, B, nboot, seed_of("lgw", name, f"SC3w:{lab}:tau{tau:g}"))
        ok = nb_union >= MS.MIN_BLOCKS_CI and "dist" in r
        out[lab] = dict(delta=r["delta"], delta_beq=r["delta_beq"], rmse_A=r["rmse_A"], rmse_B=r["rmse_B"], n_splits=r["n_splits"],
                        n_blocks_min=int(min(r["n_blocks"])) if r["n_blocks"] else 0, ok=bool(ok), dist=r.get("dist") if ok else None,
                        dist_beq=r.get("dist_beq") if ok else None, ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], ci_lo_beq=r["ci_lo_beq"],
                        ci_hi_beq=r["ci_hi_beq"], nb_union=nb_union)
    return out


def sc3_rows(per, recs, taus=TAUS, judge=tuple(f"{t}|x" for t in M3)):
    """지역 행과 3지역 층화 평균 행, SC3w 판정 행."""
    rows, tests = [], []
    for tau in taus:
        for lab in ("rule-cap", "rule-P0", "cap-P0"):
            for nm, d in per.items():
                s = d[tau][lab]
                rc = recs[(recs.target == nm.split("|")[0]) & (np.isclose(recs.tau, tau))]
                lab_mean = float(rc.stop.mean()) if len(rc) else np.nan
                cap = float(rc.cap.mean()) if len(rc) else np.nan
                v = X.verdict4(s["ci_lo"], s["ci_hi"], s["ci_lo_beq"], s["ci_hi_beq"], DELTA)
                fr = {f"frac_stop_{k}": float((rc.stop == k).mean()) for k in (3, 10)} if len(rc) else {}
                rows.append(dict(scope="region", role="판정" if nm in judge else "서술", target=nm, tau=tau, contrast=lab, delta=s["delta"],
                                 ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], delta_blockeq=s["delta_beq"], ci_lo_beq=s["ci_lo_beq"], ci_hi_beq=s["ci_hi_beq"],
                                 verdict4=v, worse=bool(v == "열세"), rmse_A=s["rmse_A"], rmse_B=s["rmse_B"], n_splits=s["n_splits"],
                                 n_blocks_split_min=s["n_blocks_min"], nb_union=s["nb_union"], ci_pool=bool(s["ok"]), labels_mean=lab_mean,
                                 cap_mean=cap, label_ratio=lab_mean / 40.0 if nm in judge else lab_mean / cap if cap else np.nan,
                                 frac_stop_cap=float((rc.stop == rc.cap).mean()) if len(rc) else np.nan, **fr))
            use = [nm for nm in judge if nm in per]
            pool = [nm for nm in use if per[nm][tau][lab]["dist"] is not None]
            if not use:
                continue
            dist = MS.strat([per[nm][tau][lab]["dist"] for nm in pool]) if pool else None
            distb = MS.strat([per[nm][tau][lab]["dist_beq"] for nm in pool]) if pool else None
            lo, hi = X._ci(dist); lob, hib = X._ci(distb)
            base = pool if pool else use
            dm = float(np.mean([per[nm][tau][lab]["delta"] for nm in base]))
            dmb = float(np.mean([per[nm][tau][lab]["delta_beq"] for nm in base]))
            rr = [r for r in rows if r["scope"] == "region" and r["target"] in use and r["tau"] == tau and r["contrast"] == lab]
            ratio = float(np.mean([r["label_ratio"] for r in rr])) if rr else np.nan
            bad = len(pool) < H.MIN_POOL_REGIONS or not np.isfinite(lo) or dm < lo - 1e-9 or dm > hi + 1e-9
            v = "판정 불가" if bad else X.verdict4(lo, hi, lob, hib, DELTA)
            rows.append(dict(scope="MEAN3", role="판정", target=f"MEAN[{','.join(base)}]", tau=tau, contrast=lab, delta=dm, ci_lo=lo, ci_hi=hi,
                             delta_blockeq=dmb, ci_lo_beq=lob, ci_hi_beq=hib, verdict4=v, n_ci_regions=len(pool), label_ratio=ratio,
                             labels_mean=float(np.mean([r["labels_mean"] for r in rr])) if rr else np.nan,
                             undetermined=bool(bad)))
    df = pd.DataFrame(rows)
    for tau in taus:
        m = df[(df.scope == "MEAN3") & np.isclose(df.tau, tau) & (df.contrast == "rule-cap")]
        if not len(m):
            continue
        m = m.iloc[0]
        reg = df[(df.scope == "region") & np.isclose(df.tau, tau) & (df.contrast == "rule-cap") & df.target.isin(list(judge))]
        worse = [r.target for r in reg.itertuples() if r.worse]
        cls = "판정 불가(풀 지역 부족)" if bool(m.undetermined) and not (np.isfinite(m.label_ratio) and m.label_ratio >= 0.9) else \
            sc3_class(m.label_ratio, m.delta, m.ci_hi, m.ci_hi_beq, bool(worse))
        conds = dict(i=bool(np.isfinite(m.label_ratio) and m.label_ratio <= 0.5), ii=bool(np.isfinite(m.delta) and m.delta <= 0.3),
                     iii=bool(np.isfinite(m.ci_hi) and np.isfinite(m.ci_hi_beq) and m.ci_hi < 0.5 and m.ci_hi_beq < 0.5), iv=not worse)
        if cls == "지지":
            txt = (f"순차 멈춤 규칙(τ {tau:g})은 라벨 수를 항상 40개 대비 절반 이하로 줄이면서 오차 증가를 0.3 cm 이하로 유지했고, "
                   f"오차가 커진 지역은 없었다(3지역)")
        elif cls == "규칙 비작동":
            txt = f"{'주 ' if np.isclose(tau, TAU_MAIN) else ''}τ {tau:g} 에서 멈춤 규칙은 거의 작동하지 않았다(라벨 비 {m.label_ratio:.2f}). Δ_SC3 는 서술만 한다"
        elif cls == "부분(정밀도 부족)":
            txt = f"τ {tau:g} 에서 라벨 비와 점 추정 조건은 만족했으나 3지역 평균의 CI 상한이 0.5 cm 이상이다(정밀도 부족)"
        elif cls == "기각":
            txt = f"τ {tau:g} 에서 충족하지 않은 조건: " + ", ".join(f"({k})" for k, ok in conds.items() if not ok)
        else:
            txt = f"τ {tau:g}"
        stat = (f"라벨 비 {m.label_ratio:.3f}; 3지역 평균 Δ_SC3 {m.delta:+.3f} [{m.ci_lo:+.3f}, {m.ci_hi:+.3f}], 블록 등가중 {m.delta_blockeq:+.3f} "
                f"[{m.ci_lo_beq:+.3f}, {m.ci_hi_beq:+.3f}]; 열세 지역 {worse if worse else '없음'}")
        tests.append(dict(test_id="SC3w" if np.isclose(tau, TAU_MAIN) else f"SC3w-tau{tau:g}", role="주" if np.isclose(tau, TAU_MAIN) else "서술(τ 민감도)",
                          scope="verdict", verdict=(cls if np.isclose(tau, TAU_MAIN) else f"(서술, τ {tau:g}) {cls}") + ". " + txt,
                          stat=stat + "; 조건 " + ", ".join(f"({k}) {'충족' if ok else '불충족'}" for k, ok in conds.items()),
                          flags="규칙 비작동과 기각이면 이 규칙을 배포 지침에 넣지 않는다" if cls in ("규칙 비작동", "기각") else "",
                          blind=BLIND["SC3w"]))
    return df, tests


SEALED_README = """봉인 폴더(결과 비열람)

이 폴더의 파일에는 LG·LGX 결과와 같은 값이거나 같은 형식인 수치, 또는 판정 행이 들어 있다
(h39 머리말 규약 12, 부록 WRAPUP '결과(실행 후 추가)' 절).
- lgw_sc3_full.csv: SC3w 전체 표. 상한이 |A| 인 대상(러시아 W·E 등)의 cap-P0 행은 LG 의 P1 − P0 | 전량과 같은 값이고,
  레나·캐나다·Alaska(x)의 cap-P0 행은 LG 의 P1 − P0 | n = 40 과 같은 형식이다(lg_equiv 열). rmse_A, rmse_B 열도 같은 이유로 봉인한다.
- logs/: 8.4 (ii) 교차 환경 점검에서 h42·h41 이 표준 출력에 찍은 판정 행이 든 로그.

LG·LGX 결과 회수(부록 0.3 의 기준 시각) 전에는 열지 않는다. 연 사람은 누가, 언제, 무엇을 열었는지 부록 개정 이력에 적는다.
"""


def write_sealed_readme(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "README_SEALED.txt").write_text(SEALED_README, encoding="utf-8")
    return d


def sc3_lg_equiv(contrast, cap_all_A, cap_any_A):
    """SC3w 표의 행이 LG 결과와 어떤 관계인지(규약 12). cap_all_A = 모든 분할에서 상한 = |A|."""
    if contrast == "cap-P0":
        if cap_all_A:
            return "LG 의 P1 − P0 | 전량과 같은 값(식·분할·채점 셀이 같다)"
        if not cap_any_A:
            return "LG 의 P1 − P0 | n = 40 과 같은 형식(추출만 다름)"
        return "분할마다 LG 의 P1 − P0 | 전량 또는 n = 40 과 같은 형식"
    if contrast == "rule-P0":
        return "LG 의 P1 − P0(n = 3, 10, 상한)과 같은 형식의 혼합"
    return "rmse_B 는 cap-P0 와 같은 이유로 LG 의 P1 RMSE 와 같거나 같은 형식"


def sc3_write(out, df, recs):
    """규약 12: 판정용 표(rule-cap, RMSE 열 없음)와 봉인 표(전체, lg_equiv 열)를 나눠 쓴다."""
    out = Path(out)
    capA = {t: (bool((g.cap == g.nA).all()), bool((g.cap == g.nA).any())) for t, g in recs.groupby("target")}

    def _eq(r):
        nm = str(r["target"])
        names = nm[5:-1].split(",") if nm.startswith("MEAN[") else [nm]
        f = [capA.get(x.split("|")[0], (False, False)) for x in names]
        return sc3_lg_equiv(r["contrast"], all(v[0] for v in f), any(v[1] for v in f))
    df = df.copy()
    df["lg_equiv"] = [_eq(r) for r in df.to_dict("records")]
    sealed = write_sealed_readme(out / "sealed")
    df.to_csv(sealed / "lgw_sc3_full.csv", index=False)
    main = df[df.contrast == "rule-cap"].drop(columns=["rmse_A", "rmse_B", "lg_equiv"], errors="ignore")
    main.to_csv(out / "lgw_sc3.csv", index=False)
    return df, main, capA


def run_sc3(a):
    D = make_data(a, 5)
    judge = [(t, "x") for t in M3]
    desc = [("Russia_W", "x"), ("Russia_E", "x")] + [(t, "x") for t in SUB10]
    per, recs, e0 = {}, [], {}
    for t, m in judge + desc:
        E0, comp = source_E0(D, t, m); e0[f"{t}|{m}"] = dict(E0=E0, **comp)
        stores, rc = sc3_target(D, t, m, E0, a.sc3_reps)
        if not stores:
            continue
        per[f"{t}|{m}"] = {tau: sc3_contrasts(f"{t}|{m}", stores, a.sc3_reps, a.nboot, tau) for tau in TAUS}
        recs.append(rc)
        H4.boot_weights.cache_clear()
    recs = pd.concat(recs, ignore_index=True)
    df, tests = sc3_rows(per, recs)
    mark_nboot(df, a)
    df, main, capA = sc3_write(a.OUT, df, recs)                            # 병기 대비와 RMSE 열은 봉인 표에만(규약 12)
    stop = recs.groupby(["target", "mode", "tau", "stop"], as_index=False).size().rename(columns={"size": "count"})
    stop.to_csv(a.OUT / "lgw_sc3_stops.csv", index=False)
    for r in tests:
        r.update(mode="sc3", nboot=int(a.nboot), nboot_capped=bool(a.CAPPED))
        if a.CAPPED:
            r["verdict"] = "(재표집 제한, 판정에 쓰지 않음) " + r["verdict"]
    update_tests(a.OUT, tests)
    update_meta(a, "sc3", dict(inputs=input_hashes(a), E0=e0, reps=int(a.sc3_reps), taus=list(TAUS), tau_main=TAU_MAIN,
                               splits={k: sorted(int(s) for s in recs[recs.target == k.split("|")[0]].split.unique()) for k in per},
                               sealed=dict(file="sealed/lgw_sc3_full.csv", n_rows=int(len(df)), contrasts_sealed=["rule-P0", "cap-P0"],
                                           cols_sealed=["rmse_A", "rmse_B"], cap_all_A=sorted(t for t, v in capA.items() if v[0])),
                               main=dict(file="lgw_sc3.csv", n_rows=int(len(main)), contrasts=["rule-cap"])))
    print(f"[sc3] 대상 {len(per)} · 행 {len(df)} → {a.OUT}", flush=True)
    return dict(table=df, stops=stop, tests=tests)


# ================================================================ bias-mae(1.7)
BM_METHODS = ("P0", "P1", "D0", "R0", "R1")
BM_N = (0, 10, -1)


def bias_mae_unit(path):
    """lgxb 의 _cells.npz 하나에서 (방법, n, 추출, seed) 별 편향·MAE·R²·MSE 분해."""
    rows = []
    with np.load(path, allow_pickle=False) as z:
        keys = [json.loads(k) for k in z["keys"]]
        P, E = z["P"].astype(float), z["E"].astype(float)
        y, s = z["y"].astype(float), z["s"].astype(float)
    sst = float(np.sum((y - y.mean()) ** 2))

    def put(method, n, d, seed, pred):
        e = pred - y
        mse = float(np.mean(e ** 2)); b = float(np.mean(e))
        rows.append(dict(method=method, n=int(n), draw=int(d), seed=int(seed), bias=b, mae=float(np.mean(np.abs(e))), mse=mse, bias2=b * b,
                         var_err=float(np.var(e)), r2=float(1.0 - np.sum(e ** 2) / sst) if sst > 0 else np.nan))
    done_p = set()
    for j, (m, lr, al, pl, n, d, seed, comp) in enumerate(keys):
        if str(al) != "1" or pl != "cell" or int(n) not in BM_N:
            continue
        if m == "D0" and comp == "pred" and lr == H.BASE_LEARNER:
            put("D0", n, d, seed, P[j])
        elif m in ("R0", "R1") and comp == "g" and lr == H.BASE_LEARNER and np.isfinite(E[j]):
            put(m, n, d, seed, E[j] * s + H.LAM_BASE * P[j])
            if m == "R0" and int(n) == 0 and ("P0",) not in done_p:
                put("P0", 0, 0, -1, E[j] * s); done_p.add(("P0",))
            if m == "R1" and ("P1", int(n), int(d)) not in done_p:
                put("P1", n, d, -1, E[j] * s); done_p.add(("P1", int(n), int(d)))
    return pd.DataFrame(rows)


def run_bias_mae(a):
    sh = X.find_shards_x(_abs(a.lgx_shards), f"{a.lgx_tag}b")
    rows = []
    for s_ in sh:
        if s_["mode"] != "x" or s_["target"] not in IND5 or not s_["cells"].exists():
            continue
        u = bias_mae_unit(s_["cells"])
        if len(u):
            rows.append(u.assign(target=s_["target"], mode=s_["mode"], split=s_["split"]))
    if not rows:
        print("[bias-mae] lgxb 셀 파일 없음", flush=True)
        return None
    d = pd.concat(rows, ignore_index=True)
    g = d.groupby(["target", "mode", "method", "n", "split"], as_index=False)[["bias", "mae", "mse", "bias2", "var_err", "r2"]].mean()
    out = g.groupby(["target", "mode", "method", "n"], as_index=False).agg(n_splits=("split", "nunique"), bias=("bias", "mean"), mae=("mae", "mean"),
                                                                            mse=("mse", "mean"), bias2=("bias2", "mean"), var_err=("var_err", "mean"),
                                                                            r2=("r2", "mean"))
    out["frac_bias2"] = out.bias2 / out.mse
    out.to_csv(a.OUT / "lgw_bias_mae.csv", index=False)
    update_meta(a, "bias-mae", dict(n_units=len(rows), shards=str(a.lgx_shards)))
    return out


# ================================================================ summarize: 대비 엔진
def G_lg(method, n, alpha="1", placement="cell", learner=None, lam=None):
    """LG(h40) 곡선 키. 물리식과 V1 은 학습기 none, λ 0.0. 기준 λ 는 h40.base_lam 이다."""
    if method == "P0":
        return H.P0_GRP
    if method in ("P1", "P2", "P3", "V1"):
        return (method, "none", "1", placement, int(n), 0.0)
    return (method, learner or H.BASE_LEARNER, str(alpha), placement, int(n), float(H.base_lam(method) if lam is None else lam))


def grp_rmse2(tm, g):
    """곡선 키의 (셀 가중, 블록 등가중) RMSE. 추출·seed 평균 뒤 분할 평균."""
    vc, vb = [], []
    for sp, st in tm.used.items():
        ks = tm.idx[sp].get(g)
        if not ks:
            continue
        S, C = st.matrices(ks)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore")
            vc.append(float(np.nanmean(H4._rmse_rows(S, C)))); vb.append(float(np.nanmean(H4._beq_rows(S, C))))
    return (float(np.mean(vc)) if vc else np.nan, float(np.mean(vb)) if vb else np.nan)


def region_stat(tm, gA, gB, label, nboot):
    """지역 하나의 대비. 주 분포는 h40.contrast(추출 공통 번호 제한, seed = seed_of('lgw', 대상, 대비)), 보조는 h42.boot_delta_common."""
    t2 = copy.copy(tm)
    t2.seed = seed_of("lgw", tm.name, label)
    t2.nboot = int(nboot) if tm.has_ci else 0
    r = H.contrast(t2, gA, gB, return_dist=True)
    if r is None or not np.isfinite(r["delta"]):
        return None
    ok = bool(tm.has_ci and tm.nb_union >= MS.MIN_BLOCKS_CI and "dist" in r)
    rc = X.boot_delta_common(t2, gA, gB, nboot=t2.nboot) if ok else None
    p0c, p0b = grp_rmse2(tm, H.P0_GRP)
    return dict(delta=float(r["delta"]), delta_beq=float(r["delta_beq"]), rmse_A=float(r["rmse_A"]), rmse_B=float(r["rmse_B"]),
                n_splits=int(r["n_splits"]), n_splits_expected=X.n_expected(tm), n_blocks_min=int(min(r["n_blocks"])) if r["n_blocks"] else 0,
                ok=ok, dist=r["dist"] if ok else None, dist_beq=r["dist_beq"] if ok else None, cdist=rc.get("dist") if rc else None,
                cdist_beq=rc.get("dist_beq") if rc else None, p0_cell=p0c, p0_beq=p0b, n_draws=int(r.get("n_draws", 0)))


def noninf(hi, hib, delta=DELTA):
    return bool(np.isfinite(hi) and np.isfinite(hib) and hi < delta and hib < delta)


def verdict4_2(lo, hi, lob, hib, dc, db):
    """가중마다 한계가 다른 4분 판정(9.4 의 δ_rel)."""
    v = [lo, hi, lob, hib, dc, db]
    if not all(x is not None and np.isfinite(x) for x in v):
        return "판정 불가"
    if hi < 0 and hib < 0:
        return "우세"
    if lo > 0 and lob > 0:
        return "열세"
    if max(abs(lo), abs(hi)) <= dc and max(abs(lob), abs(hib)) <= db:
        return "동등"
    return "미결정"


def rel_cols(r, p0c, p0b):
    """9.4: δ_rel = 0.02 × P0 RMSE(가중별)의 4분 판정과 한계 의존 표지."""
    dc, db = REL_COEF * p0c if np.isfinite(p0c) else np.nan, REL_COEF * p0b if np.isfinite(p0b) else np.nan
    vr = verdict4_2(r.get("ci_lo"), r.get("ci_hi"), r.get("ci_lo_beq"), r.get("ci_hi_beq"), dc, db)
    vs = [str(r.get("verdict4", "")), str(r.get("verdict4_d10", "")), vr]
    dep = "한계 의존" if (all(v not in NA_V and v != "" for v in vs) and len(set(vs)) > 1) else ""
    return dict(rmse_p0=p0c, rmse_p0_beq=p0b, delta_rel=dc, delta_rel_beq=db, verdict4_rel=vr, limit_dependence=dep)


def stat_row(s, target, extra=None):
    lo, hi = X._ci(s.get("dist")); lob, hib = X._ci(s.get("dist_beq"))
    clo, chi = X._ci(s.get("cdist")); clob, chib = X._ci(s.get("cdist_beq"))
    pc = H4.boot_p(s["dist"]) if s.get("dist") is not None else np.nan
    pb = H4.boot_p(s["dist_beq"]) if s.get("dist_beq") is not None else np.nan
    pe = [X.eq_p(s[k], DELTA) for k in ("dist", "dist_beq") if s.get(k) is not None]
    v = X.verdict4(lo, hi, lob, hib, DELTA)
    vc = X.verdict4(clo, chi, clob, chib, DELTA)
    row = dict(target=target, delta=s["delta"], ci_lo=lo, ci_hi=hi, delta_blockeq=s["delta_beq"], ci_lo_beq=lob, ci_hi_beq=hib,
               ci_lo_c=clo, ci_hi_c=chi, ci_lo_beq_c=clob, ci_hi_beq_c=chib, p_cell=pc, p_beq=pb,
               p_two=float(np.nanmax([pc, pb])) if np.isfinite([pc, pb]).any() else np.nan, p_eq=float(np.nanmax(pe)) if pe else np.nan,
               verdict4=v, verdict4_d10=X.verdict4(lo, hi, lob, hib, DELTA_AUX), verdict4_common=vc,
               ci_dependence="분할 독립 가정 의존" if (v != vc and v != "행 없음") else "", noninf=noninf(hi, hib, DELTA),
               noninf_d10=noninf(hi, hib, DELTA_AUX), worse=bool(np.isfinite(lo) and np.isfinite(lob) and lo > 0 and lob > 0),
               rmse_A=s.get("rmse_A", np.nan), rmse_B=s.get("rmse_B", np.nan))
    row.update(rel_cols(row, s.get("p0_cell", np.nan), s.get("p0_beq", np.nan)))
    if v in ("우세", "열세") and np.isfinite(s["delta"]) and abs(s["delta"]) < 0.5:
        row["small_note"] = X.SMALL_EFFECT_TXT
    row.update(extra or {})
    return row


def pool_label(k, N):
    if k < H.MIN_POOL_REGIONS:
        return "판정 불가"
    return f"지역 {k}/{N}" if k >= N else f"부분(지역 {k}/{N})"


def pool(per, names, extra=None):
    """지역 행과 층화 평균 행(h42.pool_rows 와 같은 풀 규칙). 평균 행의 분포는 '_dist', '_dist_beq' 키에 둔다."""
    have = [nm for nm in names if nm in per]
    if not have:
        return []
    rows = [stat_row(per[nm], nm, dict(extra or {}, scope="region", n_splits=per[nm]["n_splits"], n_splits_expected=per[nm]["n_splits_expected"],
                                       n_blocks_split_min=per[nm]["n_blocks_min"], ci_pool=bool(per[nm]["dist"] is not None))) for nm in have]
    pl = [nm for nm in have if per[nm]["dist"] is not None]
    use = pl if pl else have
    cd = [per[nm]["cdist"] for nm in pl if per[nm]["cdist"] is not None]
    cdb = [per[nm]["cdist_beq"] for nm in pl if per[nm]["cdist_beq"] is not None]
    m = dict(delta=float(np.mean([per[nm]["delta"] for nm in use])), delta_beq=float(np.mean([per[nm]["delta_beq"] for nm in use])),
             rmse_A=float(np.mean([per[nm]["rmse_A"] for nm in use])), rmse_B=float(np.mean([per[nm]["rmse_B"] for nm in use])),
             dist=MS.strat([per[nm]["dist"] for nm in pl]) if pl else None, dist_beq=MS.strat([per[nm]["dist_beq"] for nm in pl]) if pl else None,
             cdist=MS.strat(cd) if cd else None, cdist_beq=MS.strat(cdb) if cdb else None,
             p0_cell=float(np.nanmean([per[nm]["p0_cell"] for nm in use])), p0_beq=float(np.nanmean([per[nm]["p0_beq"] for nm in use])))
    mr = stat_row(m, f"MEAN[{','.join(use)}]", dict(extra or {}, scope="MEAN", n_ci_regions=len(pl), n_regions_target=len(names),
                                                   pool_regions=",".join(pl), pool=pool_label(len(pl), len(names)),
                                                   splits_short="; ".join(f"{nm} {per[nm]['n_splits']}/{per[nm]['n_splits_expected']}" for nm in use
                                                                          if per[nm]["n_splits"] < per[nm]["n_splits_expected"])))
    lo, hi = mr["ci_lo"], mr["ci_hi"]
    bad = len(pl) < H.MIN_POOL_REGIONS or not np.isfinite(lo) or (mr["delta"] < lo - 1e-9 or mr["delta"] > hi + 1e-9)
    if bad:
        mr.update(verdict4="판정 불가", verdict4_d10="판정 불가", verdict4_common="판정 불가", verdict4_rel="판정 불가", noninf=False, noninf_d10=False,
                  worse=False, limit_dependence="")
    mr["undetermined"] = bool(bad)
    mr["_dist"], mr["_dist_beq"] = m["dist"], m["dist_beq"]
    return rows + [mr]


def clean(rows):
    return pd.DataFrame([{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows])


def missing_row(**kw):
    return dict(kw, scope=kw.get("scope", "MEAN"), verdict4="행 없음", verdict4_d10="행 없음", noninf=False, noninf_d10=False, worse=False,
                p_two=np.nan, p_eq=np.nan, delta=np.nan)


def contrast_pool(tms, names, gA, gB, label, nboot, extra=None):
    per = {}
    for nm in names:
        tm = tms.get(nm)
        if tm is None:
            continue
        s = region_stat(tm, gA, gB, label, nboot)
        if s is not None:
            per[nm] = s
    H4.boot_weights.cache_clear()
    return pool(per, names, extra)


# ================================================================ summarize: 1.1 묶음 AB1–AB10
AB_DEF = [
    ("AB1", "L1", "D0-P0|n0", "lg", ("D0", 0), ("P0", 0), None),
    ("AB2", "L29", "B:ens-P0|n0", "lgx", ("B:ens", 0), ("P0", 0), ("tests", "L29", "B:ens-P0|n0")),
    ("AB3", "L15", "D1-D1@shuffle|n0", "lgx", ("D1", 0), ("D1@shuffle", 0), ("tests", "L15", "D1-D1@shuffle|n0")),
    ("AB4", "곡선", "P1-P0|n10", "lg", ("P1", 10), ("P0", 0), None),
    ("AB5", "L4", "R1-P1|n10", "lg", ("R1", 10), ("P1", 10), ("aux", "L4", "R1-P1|n10")),
    ("AB6", "L2", "R2-R1|n10", "lg", ("R2", 10), ("R1", 10), ("aux", "L2", "R2-R1|n10")),
    ("AB7", "L10", "R1-F1k|n10", "lgx", ("R1", 10), ("F1k", 10), ("tests", "L10", "R1-F1k|n10")),
    ("AB8", "L8", "R1-P0|all", "lg", ("R1", -1), ("P0", 0), ("aux", "L8", "R1-P0|all")),
    ("AB9", "L4", "R1-P1|all", "lg", ("R1", -1), ("P1", -1), ("aux", "L4", "R1-P1|n-1")),
]


def _g(src, m, n):
    return G_lg(m, n) if src == "lg" else X.G(m, n)


def h42_verdict(tables, ref):
    """h42 표(lgx_lg_aux.csv 또는 lgx_tests.csv)의 같은 대비 층화 평균 행의 4분 판정. 없으면 빈 문자열."""
    if ref is None or tables is None:
        return ""
    kind, tid, lab = ref
    df = tables.get(kind)
    if df is None or not len(df) or "contrast" not in df:
        return ""
    q = df[(df.test_id.astype(str) == tid) & (df.contrast.astype(str) == lab) & (df.scope.astype(str) == "MEAN")]
    return str(q.verdict4.iloc[0]) if len(q) else ""


def lgu_ab10(lgu_dir):
    """AB10 의 입력(LGU 판정 표). 두 가중 p 가 없으면 None(행 없음)."""
    if not lgu_dir:
        return None
    for p in sorted(Path(_abs(lgu_dir)).glob("*tests*.csv")):
        df = pd.read_csv(p)
        if "test_id" not in df:
            continue
        q = df[df.test_id.astype(str).isin(["LGU-A1", "A1"])]
        if "scope" in q:
            qm = q[q.scope.astype(str).isin(["MEAN", "pool", "main"])]
            q = qm if len(qm) else q
        for pc, pb in (("p_cell", "p_beq"), ("p_boot", "p_boot_beq"), ("p_boot_cell", "p_boot_beq")):
            if len(q) and pc in q and pb in q and np.isfinite(float(q[pc].iloc[0])) and np.isfinite(float(q[pb].iloc[0])):
                r = q.iloc[0]
                return dict(p_cell=float(r[pc]), p_beq=float(r[pb]), p_eq=float(r["p_eq"]) if "p_eq" in q else np.nan,
                            verdict4=str(r.get("verdict4", "")), delta=float(r["delta"]) if "delta" in q else np.nan, file=p.name)
    return None


def abstract_rule(v, hp, hpe, pool_txt):
    if v in NA_V or v == "":
        return "초록에서 뺀다(행 없음 또는 판정 불가)"
    if v in ("우세", "열세"):
        t = "(a) 방향을 쓴다" if (np.isfinite(hp) and hp < 0.05) else "(b) 초록은 '차이를 확인하지 못했다', 본문에 '보정 전 유의'"
    elif v == "동등":
        t = "(c) 한계를 명시한다('0.5 cm 안에서 같다')" if (np.isfinite(hpe) and hpe <= X.EQ_ALPHA) else "(c) 보정 전 동등. 초록은 '차이를 확인하지 못했다'"
    else:
        t = "(d) 차이를 확인하지 못했다"
    if str(pool_txt).startswith("부분"):
        t += f"; (e) 지역 수를 넣는다({pool_txt})"
    return t


def bundle_table(tms_lg, tms_lgx, nboot, h42_tables=None, ab10=None):
    names = [f"{t}|x" for t in MAIN4]
    rows, mean_rows = [], {}
    for ab, hyp, lab, src, a_, b_, ref in AB_DEF:
        tms = tms_lg if src == "lg" else tms_lgx
        gA, gB = _g(src, *a_), _g(src, *b_)
        pr = contrast_pool(tms or {}, names, gA, gB, f"{ab}|{lab}", nboot, dict(ab=ab, hypothesis=hyp, contrast=lab, source=src))
        if not pr:
            mr = missing_row(ab=ab, hypothesis=hyp, contrast=lab, source=src, target="MEAN[]")
            pr = [mr]
        mr = pr[-1]
        hv = h42_verdict(h42_tables, ref)
        mr["h42_verdict4"] = hv
        mr["resample_dependence"] = "재표집 의존" if (hv and hv not in NA_V and mr["verdict4"] not in NA_V and hv != mr["verdict4"]) else ""
        rows += pr
        mean_rows[ab] = mr
    if ab10 is None:
        mr = missing_row(ab="AB10", hypothesis="LGU-A1", contrast="구간 점수(α 0.1) 단 (iii) − B4 | n10", source="lgu", target="MEAN[레나,캐나다,Alaska(x)]",
                         note="LGU 판정 표에 두 가중 p 가 없다")
    else:
        mr = dict(ab="AB10", hypothesis="LGU-A1", contrast="구간 점수(α 0.1) 단 (iii) − B4 | n10", source=f"lgu:{ab10['file']}", scope="MEAN",
                  target="MEAN[레나,캐나다,Alaska(x)]", p_cell=ab10["p_cell"], p_beq=ab10["p_beq"], p_two=max(ab10["p_cell"], ab10["p_beq"]),
                  p_eq=ab10["p_eq"], verdict4=ab10["verdict4"] or "판정 불가", delta=ab10["delta"], pool="지역 3/3",
                  note="p 분해능 5e-4(2,000회)")
    rows.append(mr); mean_rows["AB10"] = mr
    order = [f"AB{i}" for i in range(1, 11)]

    def pv(r, k):
        v = r.get(k, np.nan)
        ok = r.get("verdict4", "") not in NA_V and v is not None and np.isfinite(v)
        return float(v) if ok else 1.0
    p = np.array([pv(mean_rows[k], "p_two") for k in order])
    pe = np.array([pv(mean_rows[k], "p_eq") for k in order])
    hp, hpe = MS.holm(p), MS.holm(pe)
    for i, k in enumerate(order):
        r = mean_rows[k]
        r.update(holm_input_p=p[i], holm_p=float(hp[i]), holm_input_p_eq=pe[i], holm_p_eq=float(hpe[i]), holm_m=len(order),
                 equiv_note="보정 전 동등" if (r.get("verdict4") == "동등" and hpe[i] > X.EQ_ALPHA) else "",
                 abstract_rule=abstract_rule(str(r.get("verdict4", "")), float(hp[i]), float(hpe[i]), r.get("pool", "")))
    return clean(rows)


# ================================================================ summarize: 1.4 (a) L1 전량·L3·L6·L7
def l6_groups(tm):
    gs = {g for gd in tm.idx.values() for g in gd}
    return sorted(g for g in gs if g[0] in H.METHODS_ALL and g[0] != "P0" and g[2] != "nested" and g[3] == "cell"
                  and g[1] in ("none", H.BASE_LEARNER) and g[4] in (10, 40, 160) and abs(float(g[5]) - H.base_lam(g[0])) < 1e-9)


def aux4_table(tms, nboot):
    rows = []

    def one(test, lab, nm, gA, gB, **kw):
        tm = tms.get(nm)
        s = region_stat(tm, gA, gB, f"{test}|{lab}", nboot) if tm is not None else None
        if s is None:
            rows.append(missing_row(test_id=test, contrast=lab, target=nm, scope="region", **kw)); return
        rows.append(stat_row(s, nm, dict(test_id=test, contrast=lab, scope="region", n_splits=s["n_splits"], n_blocks_split_min=s["n_blocks_min"], **kw)))
    for nm in ("Russia_W|x", "Russia_E|x"):
        one("L1", "D0-P0|all", nm, G_lg("D0", -1), H.P0_GRP, n=-1)
    for nm in ("Lena|x", "Canada|x", "AL-2|i", "AL-5|i"):
        for n in (10, 40):
            one("L3", f"R0@nested-R0@a1|n{n}", nm, G_lg("R0", n, alpha="nested"), G_lg("R0", n), n=n)
            one("L3", f"R0@nested-P0|n{n}", nm, G_lg("R0", n, alpha="nested"), H.P0_GRP, n=n)
    H4.boot_weights.cache_clear()
    for nm in [f"{t}|i" for t in H.SUB_AL] + [f"{t}|x" for t in MAIN4]:
        tm = tms.get(nm)
        if tm is None:
            continue
        for g in l6_groups(tm):
            one("L6", f"{g[0]}@a{g[2]}-P0|n{g[4]}", nm, g, H.P0_GRP, n=g[4], method=g[0], alpha=g[2])
        H4.boot_weights.cache_clear()
    for nm in ("Canada|x", "CA-3|i"):
        tm = tms.get(nm)
        if tm is None:
            continue
        ns = sorted({g[4] for gd in tm.idx.values() for g in gd if g[0] in ("V1", "V1r") and (g[4] >= 40 or g[4] == -1)})
        for n in ns:
            for A, B in (("V1", "P0"), ("V1r", "P0"), ("V1", "P1"), ("V1r", "P1")):
                one("L7", f"{A}-{B}|n{n}", nm, G_lg(A, n), G_lg(B, n) if B == "P1" else H.P0_GRP, n=n)
        H4.boot_weights.cache_clear()
    return clean(rows)


# ================================================================ summarize: 1.6 분할 SD/SE 비
SR_CONTRASTS = [("D0-P0|n0", ("D0", 0), ("P0", 0), True), ("R1-P1|all", ("R1", -1), ("P1", -1), True),
                ("R1-P0|all", ("R1", -1), ("P0", 0), True), ("P1-P0|all", ("P1", -1), ("P0", 0), True),
                ("P2-P0|all", ("P2", -1), ("P0", 0), True), ("P1-P0|n10", ("P1", 10), ("P0", 0), False), ("R1-P1|n10", ("R1", 10), ("P1", 10), False)]


def beq_split_sd(tm, gA, gB):
    """분할별 블록 등가중 Δ 의 표준편차(N4 저장소, 부트스트랩 없음)."""
    v = []
    for sp in tm.used:
        pk = X._pair_keys(tm, sp, gA, gB)
        if pk is None:
            continue
        st = tm.used[sp]
        SA, CA = st.matrices(pk[0]); SB, CB = st.matrices(pk[1])
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore")
            v.append(float(np.nanmean(H4._beq_rows(SA, CA)) - np.nanmean(H4._beq_rows(SB, CB))))
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def splitratio_table(tms, splitdist, nboot, n_blocks_target=None, tms_n4=None):
    rows = []
    for t in IND5:
        nm = f"{t}|x"
        tm = tms.get(nm)
        if tm is None:
            continue
        nbt = (n_blocks_target or {}).get(nm, getattr(tm, "n_blocks_target", np.nan))
        f = [st.nb / nbt for st in tm.used.values()] if np.isfinite(nbt) and nbt > 0 else []
        fbar = float(np.mean(f)) if f else np.nan
        R0 = float(np.sqrt(1.0 - fbar)) if np.isfinite(fbar) else np.nan
        for lab, a_, b_, flagged in SR_CONTRASTS:
            gA, gB = G_lg(*a_), G_lg(*b_)
            se, seb = [], []
            for sp in sorted(tm.used):
                t1 = X.sub_tm(tm, [sp])
                t1.seed = seed_of("lgw-se", nm, lab, sp); t1.nboot = int(nboot) if t1.has_ci else 0
                r = H.contrast(t1, gA, gB, return_dist=True)
                if r is not None and "dist" in r:
                    se.append(float(np.nanstd(r["dist"], ddof=1))); seb.append(float(np.nanstd(r["dist_beq"], ddof=1)))
            H4.boot_weights.cache_clear()
            sdb = np.nan; src = ""; out10 = None
            q = splitdist[(splitdist.target.astype(str) == nm) & (splitdist.contrast.astype(str) == lab)] if splitdist is not None and len(splitdist) else pd.DataFrame()
            if len(q):
                qp = q[q.primary_source.astype(str) == "True"] if "primary_source" in q else q
                qp = qp if len(qp) else q
                sdb = float(qp.delta_sd.iloc[0]); src = str(qp.source.iloc[0])
                out10 = str(qp.lg5_outside_10_90.iloc[0]) == "True" if "lg5_outside_10_90" in qp else None
            sd_beq = beq_split_sd(tms_n4[nm], gA, gB) if (tms_n4 and nm in tms_n4) else np.nan
            R = sdb / np.mean(se) if (se and np.isfinite(sdb) and np.mean(se) > 0) else np.nan
            Rb = sd_beq / np.mean(seb) if (seb and np.isfinite(sd_beq) and np.mean(seb) > 0) else np.nan
            big = bool(flagged and np.isfinite(R) and np.isfinite(R0) and R0 > 0 and R / R0 > 1.5)
            rows.append(dict(target=nm, contrast=lab, flagged_contrast=flagged, source=src, sd_between=sdb, se_within_mean=float(np.mean(se)) if se else np.nan,
                             n_splits_se=len(se), R=R, f_bar=fbar, R0=R0, R_over_R0=R / R0 if (np.isfinite(R) and R0) else np.nan,
                             flag="SD/SE 비 과대(R/R0 > 1.5)" if big else ("서술(n = 10 대비)" if not flagged else ""),
                             lg5_outside_10_90=out10, main_switch=bool(big and out10 is True),
                             sd_between_beq=sd_beq, se_within_beq_mean=float(np.mean(seb)) if seb else np.nan, R_beq=Rb))
    return pd.DataFrame(rows)


# ================================================================ summarize: 3.2–3.4 시나리오, SC1w, SC2w
STAGES = [("T0", 0, ["R0"], ["P0"]), ("T3", 3, ["P1", "R1"], ["P0"]), ("T10", 10, ["P1", "R1"], ["P0"]),
          ("T40", 40, ["P1", "R1"], ["P0", "P1"]), ("T160", 160, ["P1", "R1"], ["P0", "P1"])]


def _nanmean(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def scenario_tables(tms, nboot):
    names_all = [f"{t}|x" for t in IND5 + SUB10]
    cells, summ = [], []
    for stage, n, recipes, refs in STAGES:
        for rec in recipes:
            for ref in refs:
                if rec == ref:
                    continue
                gA = G_lg(rec, n)
                gB = H.P0_GRP if ref == "P0" else G_lg("P1", n)
                lab = f"{rec}-{ref}|n{n}"
                per = {}
                for nm in names_all:
                    tm = tms.get(nm)
                    if tm is None:
                        continue
                    s = region_stat(tm, gA, gB, f"scen|{lab}", nboot)
                    if s is not None:
                        per[nm] = s
                nv = {}
                if rec == "R1" and n > 0:
                    for nm in per:
                        s0 = region_stat(tms[nm], gA, G_lg("R1", 0), f"scen|net|{lab}", nboot)
                        if s0 is not None:
                            nv[nm] = s0
                H4.boot_weights.cache_clear()
                ex = dict(stage=stage, n=n, recipe=rec, reference=ref, contrast=lab)
                reg = []
                for nm, s in per.items():
                    r = stat_row(s, nm, dict(ex, scope="region", independent=nm.split("|")[0] in IND5, n_splits=s["n_splits"],
                                              n_blocks_split_min=s["n_blocks_min"]))
                    if rec == "P1" and ref == "P0":
                        r["net_value"] = r["delta"]; r["net_lo"], r["net_hi"] = r["ci_lo"], r["ci_hi"]
                    elif nm in nv:
                        lo, hi = X._ci(nv[nm]["dist"])
                        r["net_value"], r["net_lo"], r["net_hi"] = nv[nm]["delta"], lo, hi
                    reg.append(r)
                cells += reg
                m4 = pool({k: v for k, v in per.items() if k in [f"{t}|x" for t in MAIN4]}, [f"{t}|x" for t in MAIN4], dict(ex, pool_set="main4"))
                m3 = pool({k: v for k, v in per.items() if k in [f"{t}|x" for t in M3]}, [f"{t}|x" for t in M3], dict(ex, pool_set="3지역 보조"))
                ind = [r for r in reg if r["independent"]]
                sub = [r for r in reg if not r["independent"]]
                cnt = {v: sum(r["verdict4"] == v for r in ind) for v in ("우세", "동등", "미결정", "열세")}
                worst = max(ind, key=lambda r: r["delta"]) if ind else None
                summ.append(dict(ex, main4_delta=m4[-1]["delta"] if m4 else np.nan, main4_ci_lo=m4[-1]["ci_lo"] if m4 else np.nan,
                                 main4_ci_hi=m4[-1]["ci_hi"] if m4 else np.nan, main4_ci_lo_beq=m4[-1]["ci_lo_beq"] if m4 else np.nan,
                                 main4_ci_hi_beq=m4[-1]["ci_hi_beq"] if m4 else np.nan, main4_verdict4=m4[-1]["verdict4"] if m4 else "행 없음",
                                 main4_noninf=bool(m4[-1]["noninf"]) if m4 else False, main4_pool=m4[-1].get("pool", "") if m4 else "",
                                 m3_delta=m3[-1]["delta"] if m3 else np.nan, m3_verdict4=m3[-1]["verdict4"] if m3 else "행 없음",
                                 n_independent=len(ind), **{f"n_{k}": v for k, v in cnt.items()},
                                 n_noninf=sum(bool(r["noninf"]) for r in ind), worst_delta=worst["delta"] if worst else np.nan,
                                 worst_target=worst["target"] if worst else "", worst_ci=f"[{worst['ci_lo']:.3f}, {worst['ci_hi']:.3f}]" if worst else "",
                                 net_value_mean=_nanmean([r.get("net_value", np.nan) for r in ind]),
                                 sub_improve=sum(r["verdict4"] == "우세" for r in sub), sub_worse=sum(r["verdict4"] == "열세" for r in sub),
                                 n_sub=len(sub)))
    return clean(cells), pd.DataFrame(summ)


def sc1w_verdict(cells, recipe="R1", test_id="SC1w", delta_col="noninf"):
    """SC1w(부록 3.3). 칸: 비열등 = 두 가중 CI 상한 < +δ, 열세 = 두 가중 CI 하한 > 0, 그 밖(행 없음 포함)은 미확인."""
    want = [(f"{t}|x", n) for t in IND5 for n in (3, 10)]
    c = cells[(cells.recipe == recipe) & (cells.reference == "P0")] if len(cells) else cells
    st, pv, info = [], [], []
    for nm, n in want:
        q = c[(c.target == nm) & (c.n == n)] if len(c) else c
        if not len(q) or str(q.verdict4.iloc[0]) in NA_V:
            st.append("미확인"); pv.append(1.0); info.append((nm, n, None)); continue
        r = q.iloc[0]
        if bool(r.worse):
            st.append("열세")
        elif bool(r[delta_col]):
            st.append("비열등")
        else:
            st.append("미확인")
        pv.append(float(r.p_two) if np.isfinite(r.p_two) else 1.0); info.append((nm, n, r))
    hp = MS.holm(np.array(pv))
    k = st.count("비열등")
    if st.count("열세") >= 1:
        bad = [f"{nm.split('|')[0]} n = {n}(Δ {r.delta:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}], Holm p {hp[i]:.3g})"
               for i, ((nm, n, r), s) in enumerate(zip(info, st)) if s == "열세"]
        v = "기각"; txt = "같은 레시피는 " + "; ".join(bad) + " 에서 오차를 키웠다"
    elif k == len(want):
        cnt = {x: sum(1 for (_, _, r) in info if r is not None and r.verdict4 == x) for x in ("우세", "동등", "미결정")}
        v = "지지"; txt = (f"라벨 3–10개로 계수를 재보정하고 잔차를 더한 레시피의 오차 증가는 시험한 독립 지역 5곳의 10칸 모두에서 {'0.5' if delta_col == 'noninf' else '1.0'} cm "
                         f"안이었다(비열등. 4분 판정은 우세 {cnt['우세']}, 동등 {cnt['동등']}, 미결정 {cnt['미결정']}/10)")
    else:
        miss = [f"{nm.split('|')[0]} n = {n}" for (nm, n, _), s in zip(info, st) if s != "비열등"]
        v = f"안전성 미확인(비열등 칸 {k}/10)"
        txt = f"열세 판정 칸은 없었으나 오차 증가가 {'0.5' if delta_col == 'noninf' else '1.0'} cm 안인지는 10칸 가운데 {k}칸에서만 확인했다(미확인 칸: {', '.join(miss)})"
    return dict(test_id=test_id, role="주" if delta_col == "noninf" else "보조(δ 1.0)", scope="verdict", verdict=f"{v}. {txt}",
                stat="; ".join(f"{nm.split('|')[0]} n{n}: {s}" for (nm, n, _), s in zip(info, st)), blind=BLIND["SC1w"])


def sc2w_verdict(cells, n=40, test_id="SC2w", role="주"):
    regs = [f"{t}|x" for t in M3]
    c = cells[(cells.recipe == "R1") & (cells.reference == "P1") & (cells.n == n)] if len(cells) else cells
    vs = {}
    for nm in regs:
        q = c[c.target == nm] if len(c) else c
        vs[nm] = str(q.verdict4.iloc[0]) if len(q) else "행 없음"
    judged = [nm for nm, v in vs.items() if v not in NA_V]
    if len(judged) < 2:
        return dict(test_id=test_id, role=role, scope="verdict", verdict=f"판정 불가(판정이 나온 지역 {len(judged)}/3)",
                    stat="; ".join(f"{k}: {v}" for k, v in vs.items()), blind=BLIND["SC2w"])
    k = sum(v == "우세" for v in vs.values())
    worse = [nm.split("|")[0] for nm, v in vs.items() if v == "열세"]
    txt = (f"라벨 {n}개에서 잔차 학습은 3지역 가운데 {k}곳에서 재보정 물리식보다 오차를 줄였다" if k >= 2 else
           f"라벨 {n}개에서 잔차 학습의 추가 이득은 3지역 가운데 {k}곳에서만 확인되었다")
    if worse:
        txt += f". 지역 {', '.join(worse)} 에서는 오차를 키웠다"
    return dict(test_id=test_id, role=role, scope="verdict", verdict=("문장 A. " if k >= 2 and not worse else "문장 B. " if not worse else "") + txt,
                stat="; ".join(f"{kk}: {v}" for kk, v in vs.items()), blind=BLIND["SC2w"])


# ================================================================ summarize: 4절 L43
def _draw_R(st, keys):
    """키(추출 × seed) → 추출별(seed 평균) 행 색인 묶음."""
    by = {}
    for j, k in enumerate(keys):
        by.setdefault(int(k[5]), []).append(j)
    return [by[d] for d in sorted(by)]


def l43_region(tm, M, n, nboot, seed):
    """2단 재표집(블록 가중 공유, 추출 재표집은 배치마다 따로). 반환 dict 또는 None."""
    gb, gc = G_lg(M, n, placement="block"), G_lg(M, n, placement="cell")
    dc, db, pts, ptb, win = [], [], [], [], []
    for sp in sorted(tm.used):
        st = tm.used[sp]
        kb, kc = tm.idx[sp].get(gb), tm.idx[sp].get(gc)
        if not kb or not kc:
            continue
        W = H4.boot_weights(st.nb, nboot, seed_of(seed, sp)) if nboot > 0 else None
        res = {}
        for pl, ks in (("block", kb), ("cell", kc)):
            S, C = st.matrices(ks)
            grp = _draw_R(st, ks)
            with np.errstate(invalid="ignore", divide="ignore"):
                r_pt = np.array([np.nanmean(np.sqrt(S[g].sum(1) / C[g].sum(1))) for g in grp])
                v = np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), np.nan)
                r_ptb = np.array([np.nanmean(np.nanmean(v[g], 1)) for g in grp])
                if W is not None:
                    Rk = np.sqrt((S @ W.T) / (C @ W.T))
                    vv = np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), 0.0); mm = (C > 0).astype(float)
                    Ek = (vv @ W.T) / (mm @ W.T)
                    Rd = np.vstack([np.nanmean(Rk[g], 0) for g in grp]); Ed = np.vstack([np.nanmean(Ek[g], 0) for g in grp])
                    I = np.random.RandomState(seed_of(seed, sp, pl)).randint(0, len(grp), size=(nboot, len(grp)))
                    cols = np.arange(nboot)[:, None]
                    res[pl] = (r_pt, r_ptb, np.nanmean(Rd.T[cols, I], 1), np.nanmean(Ed.T[cols, I], 1))
                else:
                    res[pl] = (r_pt, r_ptb, None, None)
        pts.append(float(res["block"][0].mean() - res["cell"][0].mean())); ptb.append(float(res["block"][1].mean() - res["cell"][1].mean()))
        if W is not None:
            dc.append(res["block"][2] - res["cell"][2]); db.append(res["block"][3] - res["cell"][3])
        rng = np.random.RandomState(seed_of("lgw-l43-win", tm.name, M, n, sp))
        a_, b_ = res["block"][0], res["cell"][0]
        m_ = min(len(a_), len(b_))
        win.append(float(np.mean([np.mean(a_[rng.permutation(len(a_))[:m_]] - b_[rng.permutation(len(b_))[:m_]] < 0) for _ in range(1000)])))
    if not pts:
        return None
    out = dict(delta=float(np.mean(pts)), delta_beq=float(np.mean(ptb)), n_splits=len(pts), win_rate=float(np.mean(win)))
    if dc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.update(dist=np.nanmean(np.vstack(dc), 0), dist_beq=np.nanmean(np.vstack(db), 0))
    return out


def l43_tables(tms, nboot, runs_nb=None):
    main = [f"{t}|x" for t in MAIN4]
    rows, per_all = [], {}
    specs = [("P1", 10, "주"), ("P1", 40, "주"), ("R1", 40, "보조"), ("R1", 160, "보조"), ("P1", 160, "보조")]
    for M, n, role in specs:
        per2, perc = {}, {}
        for nm in main + [f"{H.ALASKA}|x"] + sorted(k for k in tms if k.split("|")[0] in SUB10):
            tm = tms.get(nm)
            if tm is None:
                continue
            r = l43_region(tm, M, n, int(nboot) if tm.has_ci else 0, seed_of("lgw-l43", nm, M, n))
            if r is None:
                continue
            ok = bool(tm.has_ci and tm.nb_union >= MS.MIN_BLOCKS_CI and r.get("dist") is not None)
            p0c, p0b = grp_rmse2(tm, H.P0_GRP)
            sc = region_stat(tm, G_lg(M, n, placement="block"), G_lg(M, n, placement="cell"), f"L43|{M}|n{n}", nboot)
            per2[nm] = dict(delta=r["delta"], delta_beq=r["delta_beq"], rmse_A=np.nan, rmse_B=np.nan, n_splits=r["n_splits"],
                            n_splits_expected=X.n_expected(tm), n_blocks_min=sc["n_blocks_min"] if sc else 0, dist=r.get("dist") if ok else None,
                            dist_beq=r.get("dist_beq") if ok else None, cdist=sc["cdist"] if sc else None, cdist_beq=sc["cdist_beq"] if sc else None,
                            p0_cell=p0c, p0_beq=p0b, win_rate=r["win_rate"])
            if sc is not None:
                perc[nm] = sc
            H4.boot_weights.cache_clear()
        ex = dict(test_id="L43", method=M, n=n, role=role)
        pm = pool({k: v for k, v in per2.items() if k in main}, main, dict(ex, ci_kind="2단 재표집"))
        pc = pool({k: v for k, v in perc.items() if k in main}, main, dict(ex, ci_kind="추출 조건부"))
        if pm and pc:
            pm[-1]["verdict4_draw_conditional"] = pc[-1]["verdict4"]
            pm[-1]["draw_dependence"] = "추출 변동 의존" if pm[-1]["verdict4"] != pc[-1]["verdict4"] else ""
            pm[-1]["win_rate_mean"] = float(np.mean([per2[k]["win_rate"] for k in per2 if k in main])) if any(k in main for k in per2) else np.nan
        rows += pm + pc
        for nm in per2:
            if nm not in main:
                rows.append(stat_row(per2[nm], nm, dict(ex, scope="region", ci_kind="2단 재표집", win_rate=per2[nm]["win_rate"],
                                                        independent=nm.split("|")[0] == H.ALASKA)))
        per_all[(M, n)] = pm[-1] if pm else None
    if runs_nb is not None and len(runs_nb):
        nb = runs_nb.groupby(["method", "placement", "n"], as_index=False).n_blocks_lab.mean()
        for r in nb.itertuples():
            rows.append(dict(test_id="L43", scope="n_blocks_lab", method=r.method, placement=r.placement, n=int(r.n), n_blocks_lab_mean=float(r.n_blocks_lab)))
    df = clean(rows)
    m10, m40 = per_all.get(("P1", 10)), per_all.get(("P1", 40))
    ps = [float(r["p_two"]) if (r is not None and r["verdict4"] not in NA_V and np.isfinite(r["p_two"])) else 1.0 for r in (m10, m40)]
    hp = MS.holm(np.array(ps))
    v = [r["verdict4"] if r is not None else "행 없음" for r in (m10, m40)]
    if v == ["우세", "우세"]:
        txt = "지지. 재보정 물리식에서 라벨을 여러 블록에 흩어 뽑으면 셀 무작위 추출보다 n = 10(주 4지역)과 n = 40(레나·캐나다 2지역)에서 오차가 작았다"
    elif "우세" in v and "열세" in v:
        txt = "배치 효과는 라벨 수에 따라 방향이 바뀌었다(" + ", ".join(f"n = {n}: {x}" for n, x in zip((10, 40), v)) + ")"
    elif "우세" in v:
        txt = "부분 지지(" + ", ".join(f"n = {n}: {x}" for n, x in zip((10, 40), v)) + "). n = 40 은 레나·캐나다 2지역이다"
    elif "열세" in v:
        txt = "기각. 블록 분산 추출이 불리했다(" + ", ".join(f"n = {n}" for n, x in zip((10, 40), v) if x == "열세") + ")"
    elif v == ["동등", "동등"]:
        txt = "배치의 차이는 0.5 cm 안이었다"
    else:
        txt = "배치 효과는 확인되지 않았다"
    test = dict(test_id="L43", role="주", scope="verdict", verdict=txt,
                stat="; ".join(f"P1 n{n}: {x} ({'' if r is None else r.get('pool', '')}, Holm p {h:.3g})" for n, x, r, h in zip((10, 40), v, (m10, m40), hp)),
                blind=BLIND["L43"])
    return df, test


# ================================================================ summarize: 5절 AK1w
def merge_modes(st_x, st_i):
    """두 모드의 같은 분할 저장소를 한 저장소로 합친다. 채점 블록 목록이나 블록별 셀 수가 다르면 None(합치지 않는다)."""
    if not (np.array_equal(st_x.blocks, st_i.blocks) and np.array_equal(st_x.ncell, st_i.ncell)):
        return None
    keys, S, C = [], [], []
    for tag, st in (("x", st_x), ("i", st_i)):
        s_, c_ = st.matrices(st.keys)
        keys += [(tag,) + tuple(k) for k in st.keys]; S.append(s_); C.append(c_)
    return BlockStore._from_arrays(st_x.target.replace("|x", "|xi"), st_x.split, st_x.blocks, st_x.ncell, keys, np.vstack(S), np.vstack(C), {})


def ak1_rprime(tm_x, tm_i, g, nboot, seed):
    """RMSE(M, x) − RMSE(P0, i) 의 두 가중 CI. 분할 하나라도 블록이 다르면 (None, '블록 불일치')."""
    merged = {}
    for sp in sorted(set(tm_x.used) & set(tm_i.used)):
        m = merge_modes(tm_x.used[sp], tm_i.used[sp])
        if m is None:
            return None, "블록 불일치"
        merged[sp] = m
    if not merged:
        return None, "공통 분할 없음"
    A = [k for st in merged.values() for k in st.keys if k[0] == "x" and tuple(k[1:4]) == tuple(g[:3]) and k[4] == g[3] and int(k[5]) == int(g[4])
         and abs(float(k[8]) - float(g[5])) < 1e-9]
    B = [k for st in merged.values() for k in st.keys if k[0] == "i" and tuple(k[1:]) == tuple(H.P0_KEY)]
    if not A or not B:
        return None, "키 없음"
    r = boot_delta_blocks(merged, list(dict.fromkeys(A)), list(dict.fromkeys(B)), nboot=nboot, seed=seed, rep_fn=lambda k: k[6], return_dist=True)
    return r, ""


def ak1_tables(curve, tms, nboot, few_blocks=("AL-1", "AL-4", "AL-6")):
    """AK1w(부록 5절). curve = LG 곡선 표(h40 집계, sig_p0). tms = LG 저장소(모드 i, x)."""
    c = curve[curve.target.isin(H.SUB_AL) & (curve.placement == "cell") & (curve.alpha.astype(str) != "nested") & (curve.method != "P0")
              & curve.learner.isin(["none", H.BASE_LEARNER]) & curve.n.isin([10, 40, 160])].copy()
    c = c[[abs(float(l) - H.base_lam(m)) < 1e-9 for m, l in zip(c.method, c["lam"])]]
    key = ["target", "method", "alpha", "n"]
    ci = c[c["mode"] == "i"].set_index(key); cx = c[c["mode"] == "x"].set_index(key)
    rows = []
    for k, ri in ci.iterrows():
        rx = cx.loc[k] if k in cx.index else None
        pi = str(ri.sig_p0) == "improve"
        px = (str(rx.sig_p0) == "improve") if rx is not None else None
        pi_c = bool(np.isfinite(ri.d_p0_hi) and ri.d_p0_hi < 0)
        px_c = bool(rx is not None and np.isfinite(rx.d_p0_hi) and rx.d_p0_hi < 0)
        rows.append(dict(zip(key, k), pass_i=pi, pass_x=px, pass_i_cell=pi_c, pass_x_cell=px_c))
    pr = pd.DataFrame(rows)
    if not len(pr):
        return pr, dict(test_id="AK1w", role="주", scope="verdict", verdict="판정 불가(곡선 행 없음)", stat="", blind=BLIND["AK1w"])
    pr["pass_rprime"] = None; pr["rprime_note"] = ""; pr["d_x_p0i"] = np.nan; pr["d_x_p0i_hi"] = np.nan; pr["d_x_p0i_beq_hi"] = np.nan
    pr["d_x_p0i_lo"] = np.nan; pr["d_x_p0i_beq_lo"] = np.nan
    pr["base_diff_p0x_p0i"] = np.nan; pr["base_diff_mx_p0i"] = np.nan
    for j, r in pr[pr.pass_i].iterrows():
        tx, ti = tms.get(f"{r.target}|x"), tms.get(f"{r.target}|i")
        if tx is None or ti is None:
            pr.at[j, "rprime_note"] = "저장소 없음"; continue
        g = G_lg(r.method, int(r.n), alpha=str(r.alpha))
        res, note = ak1_rprime(tx, ti, g, nboot, seed_of("lgw", r.target, f"AK1w|{g}"))
        H4.boot_weights.cache_clear()
        if res is None:
            pr.at[j, "rprime_note"] = note; continue
        lo, hi = res.get("ci_lo", np.nan), res.get("ci_hi", np.nan); lob, hib = res.get("ci_lo_beq", np.nan), res.get("ci_hi_beq", np.nan)
        pr.at[j, "pass_rprime"] = bool(np.isfinite(hi) and np.isfinite(hib) and hi < 0 and hib < 0)
        pr.at[j, "d_x_p0i"], pr.at[j, "d_x_p0i_hi"], pr.at[j, "d_x_p0i_beq_hi"] = res["delta"], hi, hib
        pr.at[j, "d_x_p0i_lo"], pr.at[j, "d_x_p0i_beq_lo"] = lo, lob
        pr.at[j, "base_diff_mx_p0i"] = res["delta"]
        p0x, p0i = grp_rmse2(tx, H.P0_GRP)[0], grp_rmse2(ti, H.P0_GRP)[0]
        pr.at[j, "base_diff_p0x_p0i"] = p0x - p0i

    def rate(df, col_i="pass_i", col_x="pass_x"):
        pi = df[df[col_i].astype(bool)]
        n_pairs, n_t = len(pi), pi.target.nunique()
        if n_pairs < 4 or n_t < 3:
            return np.nan, n_pairs, n_t, "판정 불가"
        r_ = float((pi[col_x] == True).mean())                          # noqa: E712  행 없음(None)은 통과로 세지 않는다
        return r_, n_pairs, n_t, ("지지" if r_ >= 0.5 else "기각")
    r, npair, nt, v = rate(pr)
    pi = pr[pr.pass_i]
    used_rp = pi[pi.pass_rprime.notna()]
    rp = float((used_rp.pass_rprime == True).mean()) if len(used_rp) else np.nan    # noqa: E712
    vp = ("지지" if rp >= 0.5 else "기각") if np.isfinite(rp) else "판정 불가"
    dep = "기준선 의존" if (v in ("지지", "기각") and vp in ("지지", "기각") and v != vp) else ""
    tgt = pi.groupby("target").apply(lambda g: float((g.pass_x == True).mean()) > 0.5) if len(pi) else pd.Series(dtype=bool)   # noqa: E712
    combo_i = pr.groupby(["method", "alpha", "n"]).apply(lambda g: g.pass_i.mean() > 0.5) if len(pr) else pd.Series(dtype=bool)
    sel = [k for k, ok in combo_i.items() if ok]
    aux3 = float(np.mean([pr[(pr.method == m) & (pr.alpha == a_) & (pr.n == n)].pass_x.eq(True).mean() for m, a_, n in sel])) if sel else np.nan
    r_s, np_s, nt_s, v_s = rate(pr[~pr.target.isin(list(few_blocks))])
    r_c, np_c, nt_c, v_c = rate(pr, "pass_i_cell", "pass_x_cell")
    if v == "지지":
        txt = (f"알래스카 하위 지역에서 지역 내 원천으로 물리식을 넘은 조합의 과반은 알래스카 밖 원천만 쓸 때도 같은 하위 지역에서 물리식을 넘었다"
               f"(r {r:.2f}, 쌍 {npair}, 대상 {nt})")
    elif v == "기각":
        txt = f"이 통과는 원천에 같은 지역 자료가 있는 조건에 의존했다(r {r:.2f}, 쌍 {npair}, 대상 {nt})"
    else:
        txt = f"판정 불가(모드 i 통과 쌍 {npair}, 대상 {nt})"
    if dep:
        txt += f". 기준선 의존(r' {rp:.2f})"
    test = dict(test_id="AK1w", role="주", scope="verdict", verdict=txt,
                stat=(f"r {r:.3f}; r' {rp:.3f}(쌍 {len(used_rp)}, 제외 {int((pi.pass_rprime.isna()).sum())}); 대상 단위 과반 {int(tgt.sum())}/{len(tgt)}; "
                      f"조합 단위 {aux3:.3f}(조합 {len(sel)}); 민감도(블록 적은 대상 제외) {v_s} r {r_s:.3f}; 셀 가중 CI 만 {v_c} r {r_c:.3f}"),
                blind=BLIND["AK1w"])
    return pr, test


# ================================================================ summarize: 1.4 (b) 재채점·재분류
# 1.4 (b) 재분류 입력(규약 10). 경로는 --data-dir 기준이다. 대비 = (이름, Δ 셀 가중, 하한, 상한, Δ 블록 등가중, 하한, 상한).
_C2 = ("A-B", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq")
_H25_IDS = ("target", "parent", "rule", "stage", "scope", "lam", "alpha", "n")
_H25_BLK = (("방법-물리식", "d_phys_blk", "d_phys_lo", "d_phys_hi", "d_phys_beq", "d_phys_beq_lo", "d_phys_beq_hi"),)
RECLASS_SPEC = [
    dict(hyp="M1 H 계열", src="m1/m1_sc_tests.csv", meta="m1/m1_sc_meta.json", fmt="delta2", ids=("H", "label", "cond", "target"),
         contrasts=(("A-B", "delta_rmse", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_blockeq", "ci_hi_blockeq"),)),
    dict(hyp="F4(b1)", src="h4/b1_summary.csv", meta="h4/b1_meta.json", fmt="delta2", ids=("target", "parent", "method", "analysis", "role", "n"),
         contrasts=(("방법-물리식", "d_phys_block", "d_phys_lo", "d_phys_hi", "d_phys_beq", "d_phys_lo_beq", "d_phys_hi_beq"),)),
    dict(hyp="F4(b1)", src="h4/b1_protocol.csv", meta="h4/b1_meta.json", fmt="desc", note="악화 대상 수·오라클 분기 격차(개수 형식)"),
    dict(hyp="F5(b2)", src="h4/b2_curve.csv", meta="h4/b2_meta.json", fmt="delta2", ids=("target", "parent", "est", "variant", "rule", "scope", "lam", "n"),
         contrasts=(("추정기-물리식", "d_phys_blk", "d_phys_lo", "d_phys_hi", "d_phys_beq", "d_phys_beq_lo", "d_phys_beq_hi"),
                    ("추정기-수축", "d_shrink_blk", "d_shrink_lo", "d_shrink_hi", "d_shrink_beq", "d_shrink_beq_lo", "d_shrink_beq_hi"))),
    dict(hyp="F5(b2)", src="h4/b2_regional.csv", meta="h4/b2_meta.json", fmt="delta2", ids=("est", "variant", "rule", "scope", "n", "lam", "vs", "region_set"),
         contrasts=(("지역 묶음", "delta", "ci_lo", "ci_hi", None, None, None),)),
    dict(hyp="F5(b2)", src="h4/b2_f5.csv", meta="h4/b2_meta.json", fmt="desc", note="F5 요약(넓은 표)"),
    dict(hyp="F6(b2)", src="h4/b2_theory_corr.csv", meta="h4/b2_meta.json", fmt="desc", note="Spearman 상관"),
    dict(hyp="F7(b3)", src="h4/b3_summary.csv", meta="h4/b3_meta.json", fmt="delta2", ids=("target", "parent", "layer", "rule", "stage", "n"),
         contrasts=(("규칙-물리식", "blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi", "blk_d_phys_beq", "blk_d_phys_beq_lo", "blk_d_phys_beq_hi"),
                    ("규칙-무작위", "blk_d_rand", "blk_d_rand_lo", "blk_d_rand_hi", "blk_d_rand_beq", "blk_d_rand_beq_lo", "blk_d_rand_beq_hi"))),
    dict(hyp="F7(b3)", src="h4/b3_region.csv", meta="h4/b3_meta.json", fmt="delta2", ids=("test", "target"), contrasts=(_C2,)),
    dict(hyp="F7(b3)", src="h4/b3_f7.csv", meta="h4/b3_meta.json", fmt="desc", note="E 추정 분산 비"),
    dict(hyp="F8(b4)", src="h4/b4_tests.csv", meta="h4/b4_meta.json", fmt="delta2", ids=("test", "target"), contrasts=(_C2,)),
    dict(hyp="F9(c1)", src="h4/c1_tests.csv", meta="h4/c1_meta.json", fmt="delta2", ids=("family", "test", "cond", "target", "level"),
         contrasts=(("A-B", "delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq"),)),
    dict(hyp="H18–H22", src="h2/h_tests_all.csv", meta="h2/h_tests_meta.json", fmt="delta2", ids=("family", "source", "cond", "test", "target"),
         contrasts=(_C2,)),
    dict(hyp="H23", src="h2/h23_tests.csv", meta="h2/h23_meta.json", fmt="rowci", ids=("target", "method", "lam", "n"),
         contrasts=(("방법-수축", "delta_vs_shrink", "ci_lo", "ci_hi", None, None, None),)),
    dict(hyp="H24", src="h2/h24_tests.csv", meta="h2/h24_meta.json", fmt="rowci", ids=("target", "strategy", "method", "lam", "n"),
         contrasts=(("전략-무작위", "delta_vs_random", "ci_lo", "ci_hi", None, None, None),)),
    dict(hyp="H25", src="h3/h25_curve.csv", meta="h3/h25_meta.json", fmt="rowci", ids=_H25_IDS,
         contrasts=(("방법-물리식", "d_phys_mean", "d_phys_lo", "d_phys_hi", None, None, None),), note="원판(행 재표집 CI). h25b 가 블록 CI 로 대체"),
    dict(hyp="H25", src="h3/h25b_curve.csv", meta="h3/h25b_meta.json", fmt="delta2", ids=_H25_IDS, contrasts=_H25_BLK),
    dict(hyp="H25", src="h3/h25x_curve.csv", meta="h3/h25x_meta.json", fmt="delta2", ids=_H25_IDS, contrasts=_H25_BLK),
    dict(hyp="H25", src="h3/h25b_breakeven.csv", meta="h3/h25b_meta.json", fmt="desc", note="손익분기 n"),
    dict(hyp="H25", src="h3/h25x_breakeven.csv", meta="h3/h25x_meta.json", fmt="desc", note="손익분기 n"),
    dict(hyp="H26", src="", meta="", fmt="absent",
         note="등록 짝지음 검정(α 10 대 α 1)이 산출되지 않았다(LABEL_BUDGET 09-26 감사). α 별 곡선은 h3/h25_curve.csv(행 재표집 CI)에만 있다"),
    dict(hyp="H27", src="h3/h27_rule.csv", meta="", fmt="desc", note="LOO R²·Spearman"),
    dict(hyp="H27", src="h3/h27b_rule.csv", meta="", fmt="desc", note="Spearman·Kendall·MW"),
    dict(hyp="H27", src="h3/h27c_rule.csv", meta="", fmt="desc", note="Spearman·Kendall·MW(09-26 감사 정정판)"),
    dict(hyp="H27", src="h3/h27x_rule.csv", meta="", fmt="desc", note="Spearman·Kendall·MW"),
    dict(hyp="H28", src="h3/h28_tests.csv", meta="h3/h28_meta.json", fmt="delta2", ids=("family", "cond", "test", "target", "role"), contrasts=(_C2,)),
    dict(hyp="H29", src="h3/h29_summary.csv", meta="", fmt="desc", note="해로운 블록 비율·상관"),
    dict(hyp="H29", src="h3/h29b_summary.csv", meta="", fmt="desc", note="해로운 블록 비율·상관(고유 블록 단위, 09-26 감사)"),
    dict(hyp="H30", src="h3/h30_deploy_tests.csv", meta="h3/h30_meta.json", fmt="delta2", ids=("family", "cond", "test", "target", "role"),
         contrasts=(_C2,)),
]
NBOOT_UNKNOWN = "확인하지 못함"


def meta_nboot(proc, rel):
    """원 표 메타의 재표집 횟수(nboot 또는 args.nboot). 없으면 ('확인하지 못함', 사유)."""
    if not rel:
        return NBOOT_UNKNOWN, "메타 지정 없음"
    p = Path(proc) / rel
    if not p.exists():
        return NBOOT_UNKNOWN, f"{rel} 없음"
    try:
        m = json.loads(p.read_text())
    except (ValueError, OSError):
        return NBOOT_UNKNOWN, f"{rel} 읽기 실패"
    for path in (("nboot",), ("args", "nboot")):
        v = m
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            try:
                return str(int(v)), f"{rel}:{'.'.join(path)}"
            except (TypeError, ValueError):
                pass
    return NBOOT_UNKNOWN, f"{rel} 에 nboot 없음"


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def reclassify(proc, spec=None):
    """1.4 (b) 재분류(규약 10). 모든 행은 '재분류(비맹검)'이고 원래 판정은 바꾸지 않는다."""
    rows = []
    for sp in (RECLASS_SPEC if spec is None else spec):
        nb, nb_src = meta_nboot(proc, sp.get("meta", ""))
        base = dict(kind="재분류(비맹검)", hyp=sp["hyp"], source=sp.get("src", ""), fmt=sp["fmt"], nboot_original=nb, nboot_source=nb_src,
                    note=sp.get("note", ""))
        if sp["fmt"] == "absent":
            rows.append(dict(base, ci_kind="없음", verdict4="원자료 없음", n_rows_source=0)); continue
        p = Path(proc) / sp["src"]
        if not p.exists():
            rows.append(dict(base, ci_kind="없음", verdict4="원자료 없음", n_rows_source=0, note=";".join(v for v in (base["note"], "파일 없음") if v)))
            continue
        df = pd.read_csv(p, low_memory=False)
        if sp["fmt"] == "desc":
            rows.append(dict(base, ci_kind="비율·상관·개수 형식", verdict4="규약 다름, 서술", n_rows_source=int(len(df)))); continue
        for lab, dc, lo_c, hi_c, dbc, lob_c, hib_c in sp["contrasts"]:
            need = [c for c in (dc, lo_c, hi_c, dbc, lob_c, hib_c) if c]
            miss = [c for c in need if c not in df]
            if miss:
                rows.append(dict(base, contrast=lab, ci_kind="없음", verdict4="원자료 없음", n_rows_source=int(len(df)),
                                 note=";".join(v for v in (base["note"], "열 없음: " + ",".join(miss)) if v)))
                continue
            ids = [c for c in sp.get("ids", ()) if c in df]
            for d in df.to_dict("records"):
                lo, hi = _fnum(d.get(lo_c)), _fnum(d.get(hi_c))
                lob, hib = (_fnum(d.get(lob_c)), _fnum(d.get(hib_c))) if lob_c else (np.nan, np.nan)
                fc, fb = bool(np.isfinite([lo, hi]).all()), bool(np.isfinite([lob, hib]).all())
                if sp["fmt"] == "rowci":
                    kind, v = "행 재표집 CI", "규약 다름, 서술"
                elif fc and fb:
                    kind, v = "두 가중", X.verdict4(lo, hi, lob, hib, DELTA)
                elif fc or fb:
                    kind, v = ("셀 가중만" if fc else "블록 등가중만"), "4분 판정 불가(CI 규약 다름)"
                else:
                    kind, v = "CI 없음", "판정 불가(CI 없음)"
                rows.append(dict(base, contrast=lab, test_id="|".join(str(d.get(c, "")) for c in ids), target=str(d.get("target", "")),
                                 cond=str(d.get("cond", "")), n_splits=d.get("n_splits", np.nan), ci_kind=kind, delta=_fnum(d.get(dc)),
                                 ci_lo=lo, ci_hi=hi, delta_blockeq=_fnum(d.get(dbc)) if dbc else np.nan, ci_lo_beq=lob, ci_hi_beq=hib,
                                 verdict4=v, n_rows_source=int(len(df))))
    return pd.DataFrame(rows)


def _rescore_stat(units, nboot, seed):
    """units = [(split, y, block, PA (S,n), PB (S,n))] → boot_delta_blocks(두 가중) 결과."""
    stores = {}
    for sp, y, blk, PA, PB in units:
        st = BlockStore("u", int(sp), np.asarray(blk).astype(str))
        for i in range(PA.shape[0]):
            st.add(("A", i), y, PA[i])
        for i in range(PB.shape[0]):
            st.add(("B", i), y, PB[i])
        stores[int(sp)] = st
    A = [k for st in stores.values() for k in st.keys if k[0] == "A"]
    B = [k for st in stores.values() for k in st.keys if k[0] == "B"]
    r = boot_delta_blocks(stores, list(dict.fromkeys(A)), list(dict.fromkeys(B)), nboot=nboot, seed=seed, rep_fn=lambda k: k[1], return_dist=True)
    nb = len(set().union(*[set(st.blocks.tolist()) for st in stores.values()]))
    return r, nb


def _rescore_pool(name, cond, per, nboot):
    """지역 행과 층화 평균 행(블록 ≥ 8 지역만 분포)."""
    out = []
    ok = {t: r for t, (r, nb) in per.items() if nb >= MS.MIN_BLOCKS_CI and "dist" in r}
    for t, (r, nb) in per.items():
        lo, hi, lob, hib = r.get("ci_lo"), r.get("ci_hi"), r.get("ci_lo_beq"), r.get("ci_hi_beq")
        out.append(dict(kind="재채점(비맹검)", test_id=name, cond=cond, target=t, delta=r["delta"], ci_lo=lo, ci_hi=hi, delta_blockeq=r["delta_beq"],
                        ci_lo_beq=lob, ci_hi_beq=hib, n_blocks=nb, verdict4=X.verdict4(lo, hi, lob, hib, DELTA) if t in ok else "판정 불가(블록 < 8)"))
    if ok:
        d = MS.strat([r["dist"] for r in ok.values()]); db = MS.strat([r["dist_beq"] for r in ok.values()])
        lo, hi = X._ci(d); lob, hib = X._ci(db)
        out.append(dict(kind="재채점(비맹검)", test_id=name, cond=cond, target=f"MEAN[{','.join(ok)}]", delta=float(np.mean([r["delta"] for r in ok.values()])),
                        ci_lo=lo, ci_hi=hi, delta_blockeq=float(np.mean([r["delta_beq"] for r in ok.values()])), ci_lo_beq=lob, ci_hi_beq=hib,
                        verdict4=X.verdict4(lo, hi, lob, hib, DELTA) if len(ok) >= H.MIN_POOL_REGIONS else "판정 불가"))
    return out


def rescore_h1819(proc, nboot, tag="h1819"):
    from glob import glob
    P, EV = {}, {}
    for f in sorted(glob(str(Path(proc) / "m1" / f"{tag}_shard*_preds.npz"))):
        with np.load(f, allow_pickle=False) as z:
            for k in z.files:
                if k.startswith("g::") or k.startswith("anchor::"):
                    P[k] = z[k]
                elif k.startswith("eval::"):
                    _, tk, fld = k.split("::"); EV.setdefault(tk, {})[fld] = z[k]
    if not P:
        return []
    DEF = dict(dset="alaska", xset="x25", anchor="stefan", pseudo="none", r=0.0, resid="catboost_lo", iw=0, lam=0.0)

    def spec(**kw):
        s = dict(DEF); s.update(kw)
        if s["pseudo"] != "none" and s["r"] == 0.0:
            s["r"] = 10.0
        return s

    def preds(tk, s):
        suf = f"|{s['dset']}|{s['xset']}|{s['anchor']}|{s['pseudo']}|{s['r']}|{s['resid']}|{s['iw']}"
        seeds = sorted({int(k.split("|")[3]) for k in P if k.startswith(f"g::{tk}|") and k.endswith(suf)})
        if s["lam"] == 0.0 and s["anchor"] != "none":
            ak = f"anchor::{tk}|{s['dset']}|{s['anchor']}"
            return np.stack([P[ak]] * max(1, len(seeds))) if ak in P else None
        if not seeds:
            return None
        G = np.stack([P[f"g::{tk}|{sd}{suf}"] for sd in seeds])
        return G if s["anchor"] == "none" else P[f"anchor::{tk}|{s['dset']}|{s['anchor']}"][None, :] + s["lam"] * G

    xs = sorted({k.split("|")[5] for k in P if k.startswith("g::")})
    have_lst = any("stefan_lst" in k for k in P)
    tests = []
    for cond, tg in (("noinfo", ["Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"]),
                     ("covonly", ["Lena", "Canada", "Russia_W", "Russia_E"])):
        if have_lst:
            tests.append(("H18a_stefan_lst_vs_stefan", cond, tg, spec(anchor="stefan_lst"), spec()))
        if "x25_A" in xs and cond == "noinfo":
            tests.append(("H19d_x25_A_vs_x25_direct", cond, tg, spec(xset="x25_A", anchor="none", lam=1.0), spec(anchor="none", lam=1.0)))
        if cond == "covonly":
            for x in ("x25_A", "x25_lst"):
                if x in xs:
                    tests.append((f"H19p_{x}_vs_x25_pseudo", cond, tg, spec(xset=x, anchor="none", pseudo="stefan", lam=1.0),
                                  spec(anchor="none", pseudo="stefan", lam=1.0)))
    rows = []
    for name, cond, tg, sA, sB in tests:
        per = {}
        for t in tg:
            units = []
            for tk in sorted([k for k in EV if k.startswith(f"{cond}|{t}|")], key=lambda x: int(x.split("|")[2])):
                if "block" not in EV[tk]:
                    continue
                PA, PB = preds(tk, sA), preds(tk, sB)
                if PA is None or PB is None:
                    continue
                S = min(PA.shape[0], PB.shape[0])
                units.append((int(tk.split("|")[2]), EV[tk]["y"], EV[tk]["block"], PA[:S], PB[:S]))
            if units:
                per[t] = _rescore_stat(units, nboot, seed_of("lgw", name, cond, t))
        rows += _rescore_pool(name, cond, per, nboot)
        H4.boot_weights.cache_clear()
    return rows


def rescore_h21(proc, nboot):
    p = Path(proc) / "h2" / "h21_preds.npz"
    if not p.exists():
        return []
    from polar.fidelity import TRANSFER_MAIN
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    rows = []
    for name, a_, b_ in (("H21_nested_vs_EAK", "nested", "E_AK"), ("H20_cci_blockE_vs_stefan", "cci_blockE", "stefan")):
        per = {}
        for t in TRANSFER_MAIN:
            ka, kb = f"pred::{t}::{a_}", f"pred::{t}::{b_}"
            if ka not in d or kb not in d or f"eval::{t}::block" not in d:
                continue
            per[t] = _rescore_stat([(0, d[f"eval::{t}::y"], d[f"eval::{t}::block"], d[ka][None, :], d[kb][None, :])], nboot, seed_of("lgw", name, t))
        rows += _rescore_pool(name, "noinfo", per, nboot)
        H4.boot_weights.cache_clear()
    return rows


def rescore_h22(proc, nboot):
    p = Path(proc) / "h2" / "h22_preds.npz"
    if not p.exists():
        return []
    rows22 = pd.read_csv(Path(proc) / "h2" / "h22_rows.csv")
    z = np.load(p, allow_pickle=True)
    EV = {}
    for k in z.files:
        if k.startswith("eval::"):
            _, tk, fld = k.split("::"); EV.setdefault(tk, {})[fld] = z[k]
    rows = []
    for cond, tg in (("noinfo", ["Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"]), ("covonly", ["Lena", "Canada", "Russia_W", "Russia_E"])):
        for m in ("irm", "vrex", "dann"):
            name = f"H22_{m}_resid_lam0.25_vs_stefan"
            per = {}
            for t in tg:
                units = []
                for tk in sorted([k for k in EV if k.startswith(f"{cond}|{t}|")], key=lambda x: int(x.split("|")[2])):
                    sp = int(tk.split("|")[2])
                    sub = rows22[(rows22.cond == cond) & (rows22.target == t) & (rows22.split == sp) & (rows22.ytype == "resid") & (rows22.method == m)]
                    if not len(sub) or "block" not in EV[tk]:
                        units = []; break
                    hp = sub.hp_nested.iloc[0]
                    seeds = sorted(sub[sub.hp == hp].seed.unique())
                    Gm = np.stack([z[f"g::{tk}|resid|{m}|{hp:g}|{sd}"] for sd in seeds])
                    anc = z[f"anchor::{tk}"]
                    units.append((sp, EV[tk]["y"], EV[tk]["block"], anc[None, :] + 0.25 * Gm, anc[None, :]))
                if units:
                    per[t] = _rescore_stat(units, nboot, seed_of("lgw", name, cond, t))
            rows += _rescore_pool(name, cond, per, nboot)
            H4.boot_weights.cache_clear()
    return rows


# ================================================================ summarize: 9.4 δ_rel(외부 표)
REL_ID_COLS = ("target", "mode", "test_id", "contrast", "scope")
REL_CI_COLS = ("ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq", "ci_lo_b", "ci_hi_b")
REL_P0_COLS = (("rmse_p0", "p0_rmse", "rmse_P0"), ("rmse_p0_beq", "p0_rmse_beq", "rmse_P0_beq"))
REL_SKIP_FILE = re.compile(r"(smoke|precheck|_pre[_.]|interim|count)", re.I)


def read_ci_table(path):
    """9.4 의 외부 표 읽기(규약 11): 식별 열, CI 끝값, P0 RMSE 열만 읽는다. 판정 열과 Δ 열은 읽지 않는다."""
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 1:
        return None
    want = set(REL_ID_COLS) | set(REL_CI_COLS) | {c for g in REL_P0_COLS for c in g}
    df = pd.read_csv(p, usecols=lambda c: c in want, low_memory=False)
    if "ci_lo_beq" not in df and "ci_lo_b" in df and "ci_hi_b" in df:
        df = df.rename(columns={"ci_lo_b": "ci_lo_beq", "ci_hi_b": "ci_hi_beq"})
    return df.drop(columns=[c for c in ("ci_lo_b", "ci_hi_b") if c in df])


def resolve_target(name, mode, p0map):
    """표의 대상 이름을 LG 조각 이름('대상|모드')에 대응시킨다. 모드가 여럿이면 표에 모드가 있을 때만 대응시킨다."""
    name = str(name).strip()
    if name in p0map:
        return name
    if mode is not None and str(mode) not in ("", "nan") and f"{name}|{mode}" in p0map:
        return f"{name}|{mode}"
    cand = [k for k in p0map if k.split("|")[0] == name]
    return cand[0] if len(cand) == 1 else None


def delta_rel_external(df, p0map, source, mark=""):
    """9.4 δ_rel(규약 11). df 는 read_ci_table 의 결과다. 세 판정(δ 0.5, δ 1.0, δ_rel)을 CI 끝값으로 다시 계산한다.
    (표, 대응되지 않은 행 수)를 돌려준다."""
    need = ("ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq", "target")
    if df is None or not len(df) or not all(c in df for c in need):
        return pd.DataFrame(), 0
    pc = next((c for c in REL_P0_COLS[0] if c in df), None)
    pb = next((c for c in REL_P0_COLS[1] if c in df), None)
    rows, unmatched = [], 0
    for r in df.to_dict("records"):
        if str(r.get("scope", "")) == "verdict":                                        # 판정 행(CI 없음)은 대상이 아니다
            continue
        t = str(r.get("target", ""))
        pool_row = t.startswith("MEAN[") and t.endswith("]")
        names = [x for x in (t[5:-1].split(",") if pool_row else [t]) if x.strip()]
        keys = [resolve_target(nm, r.get("mode"), p0map) for nm in names]
        ok = bool(keys) and all(k is not None for k in keys)
        lg_c = float(np.mean([p0map[k][0] for k in keys])) if ok else np.nan
        lg_b = float(np.mean([p0map[k][1] for k in keys])) if ok else np.nan
        tc = _fnum(r.get(pc)) if pc else np.nan
        tb = _fnum(r.get(pb)) if pb else np.nan
        p0c, p0b = (tc if np.isfinite(tc) else lg_c), (tb if np.isfinite(tb) else lg_b)
        src = lambda t_, l_: "표" if np.isfinite(t_) else ("LG 조각" if np.isfinite(l_) else "없음")     # noqa: E731
        ci = [_fnum(r.get(c)) for c in ("ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq")]
        rr = dict(source=source, test_id=r.get("test_id", ""), contrast=r.get("contrast", ""), scope=r.get("scope", ""), target=t,
                  lg_targets=";".join(k or "?" for k in keys), pool_row=bool(pool_row), ci_lo=ci[0], ci_hi=ci[1], ci_lo_beq=ci[2], ci_hi_beq=ci[3],
                  verdict4=X.verdict4(*ci, DELTA), verdict4_d10=X.verdict4(*ci, DELTA_AUX),
                  p0_source=f"셀 가중 {src(tc, lg_c)}; 블록 등가중 {src(tb, lg_b)}", mark=mark,
                  pool_note="원 표의 풀 행 판정 불가 조건은 다시 계산하지 않았다(원 표의 주 판정이 판정 불가이면 이 행을 쓰지 않는다)" if pool_row else "")
        rr.update(rel_cols(rr, p0c, p0b))
        if not (np.isfinite(p0c) and np.isfinite(p0b)):
            unmatched += 1
            rr.update(verdict4_rel="판정 불가(P0 RMSE 없음)", limit_dependence="")
        rows.append(rr)
    return pd.DataFrame(rows), unmatched


# ================================================================ summarize: 입출력
def load_tms(shard_dir, tags, D, nboot, parts=("cpu",), allow_mixed=False):
    sh = [s_ for tg in tags for s_ in X.find_shards_x(shard_dir, tg) if s_["part"] in parts]
    if not sh:
        return {}, []
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    X.check_cfg_x(argparse.Namespace(allow_mixed_cfg=allow_mixed), sh, units)
    stores = load_stores([s_["npz"] for s_ in sh])
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, nboot) for nm in sorted({k[0] for k in stores})}
    return tms, sh


def run_summarize(a, D=None):
    """집계. D 를 주면 그 자료 객체를 쓴다(시험의 합성 자료)."""
    t0 = time.time()
    D = D if D is not None else make_data(a, 5)
    tms_lg, sh_lg = load_tms(_abs(a.lg_shards), [a.lg_tag], D, a.nboot, allow_mixed=a.allow_mixed_cfg)
    lgx_tags = [f"{a.lgx_tag}{s}" for s in ("b", "1", "2", "9")]
    tms_lgx, sh_lgx = load_tms(_abs(a.lgx_shards), lgx_tags, D, a.nboot, allow_mixed=a.allow_mixed_cfg)
    tms_n4, _ = load_tms(_abs(a.lgx_shards), [f"{a.lgx_tag}n4"], D, 0, allow_mixed=a.allow_mixed_cfg)
    if not tms_lg:
        raise SystemExit(f"[summarize] LG 조각 없음: {a.lg_shards}")
    for nm, tm in tms_lg.items():
        tm.n_blocks_target = int(len(np.unique(D.df.block.values[D.target_idx(nm.split("|")[0])])))
    lgx_dir = _abs(a.lgx_dir)
    rd = (lambda f: pd.read_csv(lgx_dir / f) if (lgx_dir / f).exists() else None)
    h42_tables = dict(aux=rd("lgx_lg_aux.csv"), tests=rd("lgx_tests.csv"))
    splitdist = rd("lgx_splitdist.csv")
    out = {}
    out["bundle"] = bundle_table(tms_lg, tms_lgx, a.nboot, h42_tables, lgu_ab10(a.lgu_dir))
    out["aux4"] = aux4_table(tms_lg, a.nboot)
    out["splitratio"] = splitratio_table(tms_lg, splitdist, a.nboot, tms_n4=tms_n4)
    cells, summ = scenario_tables(tms_lg, a.nboot)
    out["scenarios"], out["scenarios_summary"] = cells, summ
    tests = [sc1w_verdict(cells), sc1w_verdict(cells, delta_col="noninf_d10", test_id="SC1w-d10"),
             sc1w_verdict(cells, recipe="P1", test_id="SC1w-P"), sc2w_verdict(cells), sc2w_verdict(cells, n=160, test_id="SC2w-n160", role="보조")]
    runs_nb = None
    fr = [pd.read_csv(s_["runs"], usecols=["target", "mode", "method", "placement", "n", "n_blocks_lab"]) for s_ in sh_lg if s_["part"] == "cpu"]
    if fr:
        runs_nb = pd.concat(fr, ignore_index=True)
        runs_nb = runs_nb[runs_nb.method.isin(["P1", "R1"]) & runs_nb.n.isin(PLACE_N)]
    out["l43"], t43 = l43_tables(tms_lg, a.nboot, runs_nb)
    tests.append(t43)
    cpath = _abs(a.lg_dir) / f"{a.lg_tag}_curve.csv"
    if cpath.exists():
        out["ak1"], tak = ak1_tables(pd.read_csv(cpath, dtype=dict(alpha=str)), tms_lg, a.nboot)
    else:
        out["ak1"], tak = pd.DataFrame(), dict(test_id="AK1w", role="주", scope="verdict", verdict="판정 불가(LG 곡선 표 없음)", stat="", blind=BLIND["AK1w"])
    tests.append(tak)
    out["rescore"] = pd.concat([reclassify(a.PROC), pd.DataFrame(rescore_h1819(a.PROC, a.nboot) + rescore_h21(a.PROC, a.nboot) + rescore_h22(a.PROC, a.nboot))],
                               ignore_index=True)
    p0map = {nm: grp_rmse2(tm, H.P0_GRP) for nm, tm in tms_lg.items()}
    ext, rel_info = [], []

    def add_rel(path, label, mark=""):
        df_ = read_ci_table(path)
        d_, un = delta_rel_external(df_, p0map, label, mark)
        rel_info.append(dict(file=str(path), source=label, n_rows_read=int(len(df_)) if df_ is not None else 0, n_rows_out=int(len(d_)),
                             n_unmatched=int(un), cols_read=sorted(df_.columns.tolist()) if df_ is not None else []))
        if len(d_):
            ext.append(d_)
    add_rel(lgx_dir / "lgx_tests.csv", "lgx_tests")
    add_rel(lgx_dir / "lgx_lg_aux.csv", "lgx_lg_aux")
    for d_, lab, mark in ((a.lgt_dir, "lgt", "등록 시점에 결과 존재(미열람)"), (a.lgf_dir, "lgf", ""), (a.lgd_dir, "lgd", "교차 환경")):
        if d_:
            for p in sorted(_abs(d_).glob("*tests*.csv")):
                if REL_SKIP_FILE.search(p.name):
                    rel_info.append(dict(file=str(p), source=f"{lab}:{p.name}", skipped="스모크·사전 점검·중간 표")); continue
                add_rel(p, f"{lab}:{p.name}", mark)
    out["delta_rel"] = pd.concat(ext, ignore_index=True) if ext else pd.DataFrame()
    names = dict(bundle="lgw_bundle.csv", aux4="lgw_aux4.csv", splitratio="lgw_splitratio.csv", scenarios="lgw_scenarios.csv",
                 scenarios_summary="lgw_scenarios_summary.csv", l43="lgw_l43.csv", ak1="lgw_ak1.csv", rescore="lgw_rescore.csv", delta_rel="lgw_delta_rel.csv")
    for k, f in names.items():
        mark_nboot(out[k], a).to_csv(a.OUT / f, index=False)
    for r in tests:
        r.update(mode="summarize", nboot=int(a.nboot), nboot_capped=bool(a.CAPPED))
        if a.CAPPED:
            r["verdict"] = "(재표집 제한, 판정에 쓰지 않음) " + r["verdict"]
    update_tests(a.OUT, tests)
    update_meta(a, "summarize", dict(inputs=input_hashes(a), lg_shards=[str(s_["npz"].name) for s_ in sh_lg],
                                     lg_shard_sha={str(s_["npz"].name): file_sha(s_["npz"]) for s_ in sh_lg},
                                     lgx_shards=[str(s_["npz"].name) for s_ in sh_lgx], delta_rel_inputs=rel_info,
                                     rescore_sources=out["rescore"].fillna({"source": "", "ci_kind": ""}).groupby(["source", "ci_kind"]).size().reset_index().values.tolist()
                                     if len(out["rescore"]) and {"source", "ci_kind"} <= set(out["rescore"].columns) else [],
                                     elapsed_s=round(time.time() - t0, 1)))
    return out


# ================================================================ xenv-gate(8.4 (ii))
TIME_COLS = frozenset({"elapsed_s", "device", "hostname", "max_rss_mb", "git_commit"})     # 실행 환경 열(명시 목록). 그 밖의 열은 모두 대조한다
TIME_TABLE = re.compile(r"(timing|_meta\.json$|_count_|tuned_select)", re.I)      # 시간 표와 실행 중에 쓰는 표(집계 산출이 아니다)


def compare_tables(local_dir, ref_dir, prefixes, tol=1e-6):
    """두 폴더의 같은 이름 표를 열별로 대조한다. 값 자체는 쓰지 않고 차이만 쓴다."""
    rows = []
    for pre in prefixes:
        for pr in sorted(Path(ref_dir).glob(f"{pre}_*.csv")):
            if TIME_TABLE.search(pr.name):
                continue
            pl = Path(local_dir) / pr.name
            base = dict(table=pr.name)
            if not pl.exists():
                rows.append(dict(base, column="(표)", kind="missing_local", passed=False)); continue
            R = pd.read_csv(pr, low_memory=False) if pr.stat().st_size > 1 else pd.DataFrame()
            L = pd.read_csv(pl, low_memory=False) if pl.stat().st_size > 1 else pd.DataFrame()
            shape_ok = (len(R) == len(L)) and (list(R.columns) == list(L.columns))
            rows.append(dict(base, column="(표)", kind="shape", n_rows_ref=len(R), n_rows_local=len(L), n_cols_ref=R.shape[1], n_cols_local=L.shape[1],
                             cols_only_ref=";".join(c for c in R.columns if c not in L.columns), cols_only_local=";".join(c for c in L.columns if c not in R.columns),
                             empty_both=bool(len(R) == 0 and len(L) == 0), passed=bool(shape_ok)))
            if len(R) != len(L):
                continue
            for c in [c for c in R.columns if c in L.columns]:
                if str(c) in TIME_COLS:
                    rows.append(dict(base, column=c, kind="skipped_time", passed=True)); continue
                a_, b_ = R[c], L[c]
                if pd.api.types.is_numeric_dtype(a_) and pd.api.types.is_numeric_dtype(b_):
                    x, y = a_.values.astype(float), b_.values.astype(float)
                    fx, fy = np.isfinite(x), np.isfinite(y)
                    nanmis = int((fx != fy).sum())
                    both = fx & fy
                    md = float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0
                    rows.append(dict(base, column=c, kind="numeric", n_rows_ref=len(R), max_abs_diff=md, n_nan_mismatch=nanmis,
                                     n_over_tol=int((np.abs(x[both] - y[both]) > tol).sum()), passed=bool(md <= tol and nanmis == 0)))
                else:
                    sa_, sb_ = a_.astype(str).where(a_.notna(), "<NA>"), b_.astype(str).where(b_.notna(), "<NA>")
                    nm_ = int((sa_.values != sb_.values).sum())
                    rows.append(dict(base, column=c, kind="string", n_rows_ref=len(R), n_str_mismatch=nm_, passed=bool(nm_ == 0)))
    return pd.DataFrame(rows)


def run_xenv(a):
    if not a.xenv_local or not a.xenv_ref:
        raise SystemExit("--xenv-local 과 --xenv-ref 가 필요하다")
    df = compare_tables(_abs(a.xenv_local), _abs(a.xenv_ref), [v for v in a.xenv_prefix.split(",") if v])
    df.to_csv(a.OUT / f"lgw_xenv_gate_{a.xenv_name}.csv", index=False)
    num = df[df.kind == "numeric"] if len(df) else df
    st = df[df.kind == "string"] if len(df) else df
    summ = dict(n_tables=int(df.table.nunique()) if len(df) else 0, n_numeric_cols=int(len(num)), n_string_cols=int(len(st)),
                max_abs_diff=float(num.max_abs_diff.max()) if len(num) else np.nan,
                n_cols_over_tol=int((~num.passed.astype(bool)).sum()) if len(num) else 0,
                n_string_mismatch=int(st.n_str_mismatch.sum()) if len(st) else 0,
                n_shape_fail=int((~df[df.kind == "shape"].passed.astype(bool)).sum()) if len(df) else 0,
                n_missing_local=int((df.kind == "missing_local").sum()) if len(df) else 0,
                tables_empty=sorted(df[(df.kind == "shape") & df.empty_both.fillna(False).astype(bool)].table.tolist())
                if len(df) and "empty_both" in df else [],
                skipped_cols=sorted(set(df[df.kind == "skipped_time"].column.astype(str))) if len(df) else [])
    summ["n_tables_empty"] = len(summ["tables_empty"])
    update_meta(a, f"xenv-gate-{a.xenv_name}", dict(local=str(a.xenv_local), ref=str(a.xenv_ref), prefix=a.xenv_prefix, summary=summ))
    print(f"[xenv-gate] {json.dumps(summ, ensure_ascii=False)}", flush=True)
    return df, summ


# ================================================================ 실행
def main(argv=None):
    a = parse_args(argv)
    a.OUT.mkdir(parents=True, exist_ok=True)
    fn = {"units": run_units, "sens-unit": run_sens_unit, "sens-year": run_sens_year, "sc3": run_sc3, "bias-mae": run_bias_mae,
          "summarize": run_summarize, "xenv-gate": run_xenv}[a.mode]
    return fn(a)


if __name__ == "__main__":
    main()
