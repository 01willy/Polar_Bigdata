"""XE_hires_covariates(격자 안 해상도 입력) 실험 하네스. 계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.5절(개정 1, T0 = git 2678100).

질문: 공개 고해상 입력(10–500 m)을 더하면 R1 의 ERA5 격자 안 RMSE 가 x25 보다 작아지고 재보정 Stefan(P1)보다도 작아지는가(WF9-a 의 재시험).

구성(h54 를 고치지 않고 불러 쓴다)
  단위 XERUnit ⊂ h54.R9Unit. 문맥은 h54.build_rctx(WF6·WF9 와 같은 지역 내 모드 r, 분할 1–25 의 구조 = h54.split_structure_ext)에
  특징 표(data/processed/xe/xe_feat_v1.csv, loc_id 키)의 열을 붙인 것이다. 추출 h40.draw_cells(대상, 'r', 분할, n, 추출), seed 0·1,
  교차검증 λ(h42.cv_folds_of 블록 5묶음, seed 0 적합), 격자 n {200, 500, 1,000, 전량}(|A| 미만), 추출 3(전량 1)은 WF9 와 같다.
  변형 x25 는 h54.R9Unit.run 을 그대로 부른다(WF9 의 방법 P1, Pk, Pc, R1, Re, D0 와 같은 키. 재현 관문). 그 밖의 변형은 같은 추출에서
  계획의 방법 P1, R1(교차검증 λ ∈ {0.25, 0.5, 1.0}, λ 0.25·0.5·1.0 고정 키도 저장), D0(catboost 기본 용량)만 적합하고 키의 방법 이름에
  '@<변형>' 을 붙인다(R1@xh, D0@xh. h54 wf1 의 R1@x34 와 같은 방식). 모든 변형 조각은 같은 저장소 이름을 쓰므로 집계에서 한 TMx 로 합쳐지고,
  변형 사이 대비는 같은 라벨 집합 대비(h40.contrast, 분할 안 채점 블록 재표집)다.
  저장소(계획 2.5 '분해 저장소'): 총 <대상>|r, 격자 안 <대상>~w|r(묶음 = (블록, 기온 √TDD 값), 2셀 이상 묶음 셀의 묶음 평균 편차),
  격자 사이 <대상>~b|r(묶음 평균), 위치 안 <대상>~l|r(묶음 = (블록, √TDD, ky = floor(lat/0.009), kx = floor(lon·cosφ/0.009),
  φ = (ky + 0.5)·0.009°), 2셀 이상 위치 묶음 셀의 위치 평균 편차), 격자 안·위치 사이 <대상>~gl|r(2셀 이상 격자 묶음 셀의 (위치 평균 − 격자 평균),
  셀 수 가중). 블록마다 SSE_총 = SSE_w + SSE_b, SSE_w = SSE_gl + SSE_l(시험 (c)).
변형(x_hires_registry.py): x25, xh0 = x25 + H0, xt2 = x25 + H0 + T2, xh = x25 + H(45열에서 90 % 규칙으로 뺀 군 제외).
  SI 서술 변형: add_T2·add_M·add_O·add_V·add_S, sg250(토양 5 km 9열 → 250 m 7열. cfvo·phh2o 제거 효과가 섞인다), xh_px(30 m 이하 제품의
  점 화소 민감도), cov_AK·cov_LE(미실행, 등록 이탈: 취득·추출 경로 없음. SI 상태 표에 적는다).
  작업 묶음: --stage r1b(R1b: x25, xh0, xt2), --stage r3(R3: x25, xh, SI 변형). 한 대비를 한 작업 안에서 닫으려고 x25 를 두 작업에서 모두 적합한다.
가설(계획 2.5. 판정은 집계에서 봉인 폴더에만 쓴다)
  XE-a(주)  격자 안 RMSE: R1(교차검증 λ, xh) − R1(교차검증 λ, x25), n 500·1,000·전량, 3대상 층화(캐나다는 전량만, '지역 k/3'), _rule3(우세)
  XE-b(주)  격자 안 RMSE: R1(교차검증 λ, xh) − P1, 같은 n·풀, _rule3(우세)
  Holm 가족 = XE-a 와 XE-b 의 n 별 양측 p 6개(m = 6, 보조 열. 판정은 보정 전 CI 의 4분 판정)
  XE-a0(보조) XE-a 의 xt2 판. XE-c(보조) 위치 안 RMSE 의 두 대비, D0(xh) − D0(x25) 격자 안, 총 RMSE R1(xh) − R1(x25), xh0 대비, 대상별 행,
            n 200(XE-a·XE-b·XE-a0 의 n 200 행은 XE-c 로 적는다).
  XE-d(서술) 군별·교체·피복 변형의 설명 비율(1 − SSE_w(M)/SSE_w(P1), 격자 안과 위치 안), 격자 사이 성분.
  XE-e(진단, 탐색) ABoVE 현장 VWC 부분 집합(알래스카·캐나다, 15 m 안에 VWC 위치가 있는 라벨 셀. 계획의 2,468·31 / 602·15 와 준비 메타에서 대조)에서
            P1 격자 안 잔차(y − ȳ_g, E1 과 무관)를 비 GPR 층별 VWC(하단 ≤ 12 cm, > 12 cm)로 설명하는 비율(블록 교차검증 K ≤ 5, 1차 회귀와
            catboost_lo). GPR VWC 판과 예측 변수 유한 셀 판은 민감도. 판정을 바꾸지 않는다(--xe-e-prep → --xe-e, 출력은 봉인 폴더).
출력 제한(계획 0.3): 스모크·세기·집계의 화면에는 RMSE·Δ·판정을 쓰지 않는다. 집계 표는 data/processed/xbatch/XE_hires_covariates/sealed/ 에만 쓴다
  (변형 묶음별 파일: <tag>_xh0_*, <tag>_xt2_*(xe_feat 최종 해시 커밋 뒤, --feat-gate-only 통과 뒤 연다), <tag>_xh_*(가설 표 <tag>_xh_hypotheses.csv),
  <tag>_si_*, xe_e_*). 스모크 조각은 실제 자료의 적합 결과이므로 봉인 폴더(<OUT>/smoke/sealed/shards)에 둔다.
산출: data/processed/xbatch/XE_hires_covariates/shards/<tag>__cpu__<대상>__r__s<분할>__<변형>_{runs.csv, blocksse.npz, unit.json}
명령(ROOT 에서)
  세기:   CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/3_deep_learning/x_hires_covariates.py --count-only --stage r1b [--feat 경로]
  스모크: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c <코어 4개> python3 scripts/3_deep_learning/x_hires_covariates.py \
              --smoke --threads 2 --workers 0 --feat <스모크 특징 표> --allow-unfinal
  본 실행: WF_RESCALE=1 python3 scripts/3_deep_learning/x_hires_covariates.py --stage r1b --workers 22 --threads 4 --resume --no-summarize
  한 조각: WF_RESCALE=1 python3 ... --stage r3 --shard Alaska:7:xh(쉼표 목록) 또는 --shard 3/8(단위 목록의 3번째 몫, 8등분)
  집계:   WF_RESCALE=1 python3 ... --stage r1b --summarize-only --workers 0 [--gate-wf9 <WF9 조각 폴더>]   (재현 관문은 기본으로 먼저 돈다)
  xt2 열람 점검: python3 ... --stage r1b --feat-gate-only --feat data/processed/xe/xe_feat_v1.csv   (해시만 읽는다. 통과해야 xt2 표를 연다)
  XE-e 준비(세기 범주, 좌표·VWC 만): CUDA_VISIBLE_DEVICES= nice -n 10 python3 ... --xe-e-prep
  XE-e:   WF_RESCALE=1 python3 ... --xe-e --threads 4     (스모크: --xe-e --smoke, 캐나다·seed 0)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import xbatch_core as XB                                                                             # noqa: E402  numpy 보다 먼저

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

H, X, W = XB.H, XB.X, XB.W
ROOT = XB.ROOT


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RG = _load("x_hires_registry", SCRIPT_DIR / "x_hires_registry.py")
from polar.fidelity import SOIL                                                                      # noqa: E402
from polar.h4_common import BlockStore                                                               # noqa: E402

# ================================================================ 고정값(계획 2.5. 바꾸면 사전 등록에서 벗어난다)
EXP_ID = "XE"
EXP_NAME = XB.EXP_NAMES[EXP_ID]
OUT_DEFAULT = XB.out_dir(EXP_ID)
FEAT_DEFAULT = ROOT / "data" / "processed" / "xe" / "xe_feat_v1.csv"
H_EXP = "wf9"                                                               # 격자·추출 규칙을 빌리는 h54 실험 이름(a.G['wf9'], draws_for)
GRID = tuple(W.WF9_GRID)                                                    # (200, 500, 1000, −1)
DRAWS = int(W.WF9_DRAWS)                                                    # 3(전량 1)
SPLITS = tuple(XB.INREGION_SPLITS)                                          # 1–25
TARGETS = tuple(RG.TARGETS)
MODE = RG.MODE
MAIN_N = (500, 1000, -1)                                                    # XE-a·XE-b 의 등록 n
AUX_N = (200, 500, 1000, -1)                                                # XE-c 는 n 200 포함
HOLM_M = 6
LO, HI = XB.LO, XB.HI
LAM_CV = XB.LAM_CV
CV_K = int(W.CV_K)                                                          # 블록 5묶음(h54 교차검증 λ 와 같다)
SOIL_COLS = [c for c in XB.FEATS if c in SOIL]
WF9_SHARDS = ROOT / "results" / "rescale_wf3" / "data" / "processed" / "wf" / "shards"
WF9_TIMING = ROOT / "results" / "rescale_wf3" / "data" / "processed" / "wf" / "wf3b_timing.csv"
LOC_DEG = 0.009                                                             # LG 6B.3 셀 색인
PARTS = ("", "~w", "~b", "~l", "~gl")
PART_WORD = {"": "총", "~w": "격자 안", "~b": "격자 사이", "~l": "위치 안", "~gl": "격자 안·위치 사이"}
WITHIN_100M = {"Alaska": 49.2, "Lena": 19.0, "Canada": 60.2}               # within_grid_inputs.md 3.3(격자 안 분산 가운데 100 m 셀 안 비율, %)
FINAL_REQUIRED_TXT = "xe_feat 최종 해시 없음(계획 2.5: 최종판 커밋 뒤 R3)"
SMOKE_MAX_THREADS = 2                                                       # 계획 1절·0.3 제한 스모크(코어 4개, 스레드 2)
SMOKE_MAX_CORES = 4
# XM(추가 등록 docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md, 커밋 ff7c97f). --stage xm 은 xw·xw_lc 만 적합하고 x25 는 XE r1b 조각을 다시 쓴다
XM_OUT = XB.XBATCH_ROOT / RG.XM_EXP_NAME
XM_FEAT_DEFAULT = XM_OUT / "inputs" / "xm_feat_v1.csv"
XM_X25_FROM = OUT_DEFAULT / "shards"
XM_X25_TAG = "xe_r1b"
XM_TAG = "xm"
XM_HOLM_M = 3                                                               # 등록 4절: XM-a 의 n 500·1,000·전량
XM_DESIGN = "결과 열람 뒤 설계, 탐색(SI)"
XM_BLIND = ("비맹검 부분 포함", "xh0 의 treecover_1km·water_occ_1km 이 WC 의 수목·수면 비율과 정보가 겹치고 xh0 격자 안 대비를 열람했다(등록 4절)")
XM_CELL_SHARE = {"Alaska": 18.8, "Lena": 47.8, "Canada": 17.7}             # within_grid_inputs.md 3.3: 격자 안 분산 가운데 1 km 셀 사이 몫(%)
XM_SENT = {   # 등록 5절 '사전 고정 해석 문장'. {inp} = 입력 이름(xw: WorldCover·Sentinel-2, xw_lc 대체: WorldCover 만)
    "superior": "1 km 셀의 {inp} 더하면 R1 의 ERA5 격자 안 오차가 {a} cm 작아졌다(격자 안 설명 비율 {x} %)",
    "superior_tail": "탐색 결과이므로 C8 의 격자 안 문장과 지도 문구는 고치지 않고 확인 시험 후보로 SI 에 적는다.",
    "eq": "피복·식생 입력을 더한 R1 의 격자 안 오차는 x25 와 0.5 cm 안에서 같았다",
    "eq_tail": "XE xh0·XE-e 와 함께 '공개 10–500 m 입력은 격자 안 오차를 줄이지 못했다'는 SI 서술을 유지한다.",
    "und": "피복·식생 입력으로 격자 안 오차가 줄어드는 것을 확인하지 못했다",
    "worse": "피복·식생 입력은 격자 안 오차를 {a} cm 늘렸다",
    "na": "판정할 수 없었다",
}
XM_INPUT_WORD = {"xw": "WorldCover 10 m 피복 비율과 Sentinel-2 20 m 여름 식생 지수를", "xw_lc": "WorldCover 10 m 피복 비율을"}   # 조사 포함
XM_S2_FALLBACK_TXT = "S2 군이 세 대상 모두에서 빠짐(등록 3절 (d): xw 를 돌리지 않고 xw_lc 를 주 대비에 쓴다)"

# 맹검 표지(1절 어휘)와 근거 문장. 계획 2.5 표의 맹검 열을 따르고, 표가 비운 칸(XE-c 의 xh 행, XE-d, XE-e)은 같은 사유 규칙으로 채운다(구현 기록 3절)
BLIND_HYP = {"XE-a": ("비맹검 부분 포함", "H0 10열이 H19 입력에 있었다(계획 2.5 XE-a)"),
             "XE-b": ("비맹검 부분 포함", "WF9-a 열람(계획 2.5 XE-b)"),
             "XE-a0": ("맹검", "T2 는 새 열(계획 2.5 XE-a0)")}
SENT = {   # 계획 2.5 '사전 고정 해석 문장'
    "aw_bw": "공개 고해상 입력(10–500 m)을 더하면 물리 기반 ML 의 ERA5 격자 안 설명 비율이 {x} % 로 늘었고 재보정 Stefan 보다 격자 안 오차가 {a} cm 작았다",
    "aw_bn": "고해상 입력은 ML 의 격자 안 오차를 {a} cm 줄였으나 재보정 Stefan(격자 안 실측 표준편차)과의 차이는 확인하지 못했다",
    "aw_bl": "고해상 입력은 ML 의 격자 안 오차를 {a} cm 줄였으나 재보정 Stefan(격자 안 실측 표준편차)보다 격자 안 오차가 여전히 컸다",
    "eq": "고해상 입력을 더한 ML 의 격자 안 오차는 x25 와 0.5 cm 안에서 같았다",
    "und": "취득 가능한 공개 고해상 입력으로 격자 안 오차가 줄어드는 것을 확인하지 못했다",
    "worse": "고해상 입력은 격자 안 오차를 {a} cm 늘렸다",
    "na": "판정할 수 없었다",
}
SI_NOTES = {   # SI 상태 표의 설명(구현 기록 2절)
    "sg250": "x25 의 토양 9열(sg_cfvo_5_15·sg_phh2o_5_15 포함)을 빼고 SoilGrids 250 m 7열을 더한다(등록 문구 '5 km 9열 대신 250 m 열'의 문자 그대로 읽기). "
             "해상도 교체 효과와 cfvo·phh2o 두 변수의 제거 효과가 섞인다",
    "xh_px": "xh 의 30 m 이하 제품 3 × 3 창 열을 점 화소 열로 바꾼 민감도",
    "cov_AK": RG.COVER_NOT_RUN, "cov_LE": RG.COVER_NOT_RUN,
}


def blind_of(hyp, variant="", contrast=""):
    """행의 맹검 표지(1절 어휘)와 근거 문장. 반환 (표지, 근거)."""
    if hyp in BLIND_HYP:
        return BLIND_HYP[hyp]
    if hyp == "XE-e":
        return "맹검", "VWC 와 P1 잔차의 관계를 계산한 적이 없다(계획 2.5 열람 상태). 등록 표의 맹검 칸은 비어 있다"
    vsP1 = str(contrast).endswith("−P1")
    if vsP1:
        return "비맹검 부분 포함", "WF9-a 열람(R1 − P1 격자 안 대비)"
    if variant in ("xh0", "xh", "xh_px"):
        return "비맹검 부분 포함", "H0 10열이 H19 입력에 있었다" + ("(계획 2.5 XE-c: xh0 은 비맹검 부분 포함)" if variant == "xh0" else "")
    if variant == "xt2":
        return "맹검", "T2 는 새 열(XE-a0 와 같은 사유)"
    return "맹검", "새 열(군별 추가·교체 변형)의 대비를 본 적이 없다. 등록 표의 맹검 칸은 비어 있다"


# ================================================================ 최종 해시의 커밋 확인
def revision_section(text) -> str:
    """계획 문서의 '## 개정 이력' 절(없으면 빈 문자열)."""
    i = str(text or "").find("## 개정 이력")
    return "" if i < 0 else str(text)[i:]


def final_committed(sha, head_text=None, file_text=None) -> dict:
    """xe_feat 최종 sha256 이 계획 문서의 개정 이력에 커밋되었는가(계획 2.5 'sha256 을 개정 이력에 커밋'). 로컬은 git HEAD 의 계획 문서,
    git 이 없는 곳(Rescale 묶음)은 묶음에 넣은 계획 문서 사본을 본다. 둘 다 없으면 거짓(실패 쪽으로 닫는다). 시험은 head_text·file_text 로 준다."""
    sha = str(sha or "")
    if len(sha) != 64:
        return dict(ok=False, how="최종 sha256 없음")
    if head_text is None and file_text is None:
        head_text = XB._git("show", f"HEAD:{XB.PLAN_DOC}")
        if not head_text:
            p = ROOT / XB.PLAN_DOC
            file_text = p.read_text() if p.exists() else None
    if head_text:
        return dict(ok=sha in revision_section(head_text), how="git HEAD 계획 문서 개정 이력 대조")
    if file_text:
        return dict(ok=sha in revision_section(file_text), how="계획 문서 사본 개정 이력 대조(git 없음, 묶음)")
    return dict(ok=False, how="계획 문서와 git 이 모두 없어 대조할 수 없다")


# ================================================================ 특징 표
class FeatTable:
    """xe_feat_v1.csv 와 메타(…_meta.json), 최종 해시(…_final.json). path = None 이면 세기 전용 빈 표(모든 군 포함으로 센다. 적합에는 쓰지 않는다)."""

    def __init__(self, path=None):
        self.path = None if path in (None, "", "none") else Path(path)
        self.meta, self.final, self.sha = {}, None, ""
        self._commit = None
        if self.path is None:
            self.df = pd.DataFrame(columns=["loc_id"] + RG.H_COLS + RG.PX_COLS).set_index("loc_id")
            return
        if not self.path.exists():
            raise FileNotFoundError(f"특징 표가 없다: {self.path}(scripts/1_data_prep/xe_point_covariates.py 로 만든다)")
        self.df = pd.read_csv(self.path).set_index("loc_id")
        mp = self.path.with_name(self.path.stem + "_meta.json")
        fp = self.path.with_name(self.path.stem + "_final.json")
        if not mp.exists():                                              # 메타가 없으면 90 % 규칙을 알 수 없다(모든 군 포함으로 돌지 않게 멈춘다)
            raise FileNotFoundError(f"특징 표 메타가 없다: {mp}(묶음에 xe_feat_v1_meta.json 과 xe_feat_v1_final.json 을 함께 넣는다)")
        self.meta = json.loads(mp.read_text())
        self.final = json.loads(fp.read_text()) if fp.exists() else None
        self.sha = RG.sha256_file(self.path)
        if self.meta.get("sha256") and self.meta["sha256"] != self.sha:
            raise RuntimeError(f"특징 표 {self.path.name} 의 sha256 이 메타와 다르다(assemble 뒤 표가 바뀌었다)")

    @property
    def is_dummy(self):
        return self.path is None

    @property
    def final_local_ok(self):
        """최종 해시 파일의 sha256 이 표와 같다(로컬 파일 조건만)."""
        return bool(self.final and self.final.get("sha256") == self.sha)

    @property
    def final_commit(self) -> dict:
        if self._commit is None:
            self._commit = final_committed(self.sha) if self.final_local_ok else dict(ok=False, how="최종 해시 파일 없음 또는 표와 다름")
        return self._commit

    @property
    def final_ok(self):
        """최종판 조건 = 최종 해시 파일이 표와 같고, 그 sha256 이 계획 개정 이력에 커밋되어 있다."""
        return bool(self.final_local_ok and self.final_commit.get("ok"))

    def decisions(self, target):
        """대상의 {군: 포함 여부}. 빈 표(세기 전용)는 None(모두 포함으로 센다). 실제 표의 메타에 group_decisions 가 없거나 대상이 빠져 있으면
        멈춘다(90 % 규칙을 모르는 채 모든 군을 넣지 않는다)."""
        if self.is_dummy:
            return None
        g = (self.meta or {}).get("group_decisions")
        if not g or target not in g:
            raise RuntimeError(f"특징 표 메타에 {target} 의 group_decisions 가 없다(90 % 규칙 결과 없이 돌지 않는다)")
        return RG.decisions_for(self.meta, target)

    def columns(self):
        return list(self.df.columns)

    def matrix(self, loc_ids, cols):
        """loc_id 순서의 열 행렬(float32). 표에 없는 loc_id 와 열은 NaN."""
        d = self.df.reindex(index=np.asarray(loc_ids)).reindex(columns=list(cols))
        return d.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    def group_sha(self, groups):
        g = (self.final or {}).get("group_cols_sha256") if self.final_local_ok else self.meta.get("group_cols_sha256", {})
        return {k: (g or {}).get(k, "") for k in groups}


class XMFeatTable(FeatTable):
    """XM 특징 표(xm_feat_v1.csv 와 xm_feat_v1_meta.json, 등록 3절). 최종 해시 파일은 쓰지 않는다(등록 3절 (e): 적합 전 sha256 을 조각과 봉인 메타에
    적는다). path = None 이면 세기 전용 빈 표."""

    def __init__(self, path=None):
        if path in (None, "", "none"):
            FeatTable.__init__(self, None)
            self.df = pd.DataFrame(columns=["loc_id"] + RG.XM_COLS).set_index("loc_id")
            return
        FeatTable.__init__(self, path)

    def s2_all_dropped(self, targets=TARGETS) -> bool:
        """S2 군이 등록 세 대상 모두에서 90 % 규칙으로 빠졌는가(등록 3절 (d): 그러면 xw 를 돌리지 않고 xw_lc 를 주 대비에 쓴다)."""
        if self.is_dummy:
            return False
        g = (self.meta or {}).get("group_decisions", {})
        return all(not bool(((g.get(t) or {}).get("S2") or {}).get("included", False)) for t in targets)

    def dropped_targets(self, group, targets=TARGETS) -> list:
        g = (self.meta or {}).get("group_decisions", {})
        return [t for t in targets if not bool(((g.get(t) or {}).get(group) or {}).get("included", False))] if not self.is_dummy else []


def is_xm(variant) -> bool:
    return variant in RG.XM_VARIANTS


def needs_final(variant) -> bool:
    """최종 해시가 커밋된 표에서만 돌리는 변형(2단계 열을 쓴다). xh0·xt2 는 1단계 표로 돈다(작업 R1b). XM 변형은 XM 특징 표로 돈다(등록 3절 (e))."""
    return variant not in ("x25", "xh0", "xt2") + RG.XM_VARIANTS


def feat_of(a, variant):
    """변형의 특징 표(XM 변형은 a.FEAT_XM, 그 밖은 a.FEAT)."""
    return a.FEAT_XM if is_xm(variant) else a.FEAT


def variant_cols(a, variant, target):
    if variant == "x25":
        return RG.variant_columns("x25", target, XB.FEATS, SOIL_COLS)
    if is_xm(variant):
        ft = a.FEAT_XM
        cols, info = RG.xm_variant_columns(variant, target, XB.FEATS, ft.decisions(target))
        if cols is not None and not ft.is_dummy:
            miss = [c for c in cols if c not in XB.FEATS and c not in ft.columns()]
            if miss:
                info["skip"] = f"XM 특징 표에 열이 없다({', '.join(miss[:4])}…)"
                return None, info
        return cols, info
    cols, info = RG.variant_columns(variant, target, XB.FEATS, SOIL_COLS, a.FEAT.decisions(target), a.FEAT.columns())
    if cols is not None:
        miss = [c for c in cols if c not in XB.FEATS and c not in a.FEAT.columns()]
        if miss and not a.FEAT.is_dummy:
            info["skip"] = f"특징 표에 열이 없다({', '.join(miss[:4])}…)"
            return None, info
    return cols, info


# ================================================================ 위치 묶음과 분해 저장소
def loc_cell(lat, lon):
    """LG 6B.3 셀 색인 (ky, kx): ky = floor(lat/0.009), φ = (ky + 0.5)·0.009°, kx = floor(lon·cosφ/0.009)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    ky = np.floor(lat / LOC_DEG).astype(np.int64)
    kx = np.floor(lon * np.cos(np.radians((ky + 0.5) * LOC_DEG)) / LOC_DEG).astype(np.int64)
    return ky, kx


def loc_groups(sB, blkB, latB, lonB):
    """격자 묶음(h54.grid_groups 와 같은 열쇠)과 그 안에 중첩된 위치 묶음. 반환 (격자 번호, 격자 2셀 이상, 위치 번호, 위치 2셀 이상)."""
    gid, multi_g = W.grid_groups(sB, blkB)
    sB = np.asarray(sB, float); blkB = np.asarray(blkB).astype(str)
    ky, kx = loc_cell(latB, lonB)
    key = [f"{b}|{v:.12g}#{y}_{x}" if np.isfinite(v) else f"{b}|nan{i}" for i, (b, v, y, x) in enumerate(zip(blkB, sB, ky, kx))]
    lid = pd.factorize(pd.Series(key))[0].astype(np.int64)
    cnt = np.bincount(lid)
    return gid, multi_g, lid, cnt[lid] >= 2


def loc_decomp_fns(gid, multi_g, lid, multi_l):
    """위치 안 within_loc(y, p) = 2셀 이상 위치 묶음 셀의 (y − ȳ_l, p − p̄_l), 격자 안·위치 사이 between_loc(y, p) = 2셀 이상 격자 묶음 셀의
    (ȳ_l − ȳ_g, p̄_l − p̄_g). 위치 묶음이 격자 묶음 안에 있으므로 블록마다 SSE_w = SSE_gl + SSE_l 이다."""
    gid = np.asarray(gid, np.int64); lid = np.asarray(lid, np.int64)
    multi_g = np.asarray(multi_g, bool); multi_l = np.asarray(multi_l, bool)
    G = int(gid.max()) + 1 if len(gid) else 0
    L = int(lid.max()) + 1 if len(lid) else 0
    cg = np.maximum(np.bincount(gid, minlength=G).astype(float), 1.0)
    cl = np.maximum(np.bincount(lid, minlength=L).astype(float), 1.0)

    def mg(v):
        return (np.bincount(gid, weights=np.asarray(v, float), minlength=G) / cg)[gid]

    def ml(v):
        return (np.bincount(lid, weights=np.asarray(v, float), minlength=L) / cl)[lid]

    def within_loc(y, p):
        return (np.asarray(y, float) - ml(y))[multi_l], (np.asarray(p, float) - ml(p))[multi_l]

    def between_loc(y, p):
        return (ml(y) - mg(y))[multi_g], (ml(p) - mg(p))[multi_g]
    return within_loc, between_loc


# ================================================================ 단위
class XERCtx(W.RCtx):
    """RCtx + 변형 행렬. XAf(xs)·XBf(xs) 는 x25 면 기본 행렬, 그 밖은 변형 행렬(x25 열 + 고해상 열 또는 토양 교체)을 준다."""

    def XAf(self, xs):
        if xs == "x25":
            return self.XA
        if xs in getattr(self, "xmats", {}):
            return self.xmats[xs][0]
        return W.RCtx.XAf(self, xs)

    def XBf(self, xs):
        if xs == "x25":
            return self.XB
        if xs in getattr(self, "xmats", {}):
            return self.xmats[xs][1]
        return W.RCtx.XBf(self, xs)


def attach_variant(c, variant, colsA, colsB):
    """문맥에 변형 행렬을 붙인다(문맥의 클래스를 XERCtx 로 바꾼다)."""
    if not isinstance(c, XERCtx):
        c.__class__ = XERCtx
    if not hasattr(c, "xmats"):
        c.xmats = {}
    XA = np.asarray(colsA, np.float32); XBm = np.asarray(colsB, np.float32)
    if XA.shape[0] != len(c.yA) or XBm.shape[0] != len(c.yB) or XA.shape[1] != XBm.shape[1]:
        raise ValueError("변형 행렬의 크기가 문맥과 맞지 않는다")
    c.xmats[variant] = (XA, XBm)
    return c


class XERUnit(W.R9Unit):
    """XE 단위(h54.R9Unit 하위 클래스). x25 는 R9Unit.run 그대로, 그 밖의 변형은 P1, R1(교차검증 λ와 고정 λ), D0(catboost)만 적합한다.
    저장소는 GridDecompMixin 의 총·~w·~b 에 위치 안 ~l 과 격자 안·위치 사이 ~gl 을 더한다."""

    def __init__(self, a, c, variant="x25", dry=False):
        if variant != "x25" and variant not in getattr(c, "xmats", {}):
            raise ValueError(f"문맥에 변형 {variant} 의 행렬이 없다(attach_variant)")
        self.xs = variant
        W.RUnit.__init__(self, a, c, "xe", variant, dry)

    def _make_xstores(self):
        out = W.GridDecompMixin._make_xstores(self)
        c = self.c
        gid, multi_g, lid, multi_l = loc_groups(c.sB, c.blkB, c.latB, c.lonB)
        t, m = self.name.split("|")
        self.notes["loc"] = dict(n_loc=int(lid.max()) + 1 if len(lid) else 0, n_multi_loc=int(np.sum(np.bincount(lid) >= 2)) if len(lid) else 0,
                                 n_multi_loc_cells=int(multi_l.sum()), n_multi_grid_cells=int(multi_g.sum()), cell_deg=LOC_DEG)
        wl, bl = loc_decomp_fns(gid, multi_g, lid, multi_l)
        meta = dict(target=t, mode=c.mode, exp=self.exp, variant=self.variant)
        blk = np.asarray(c.blkB)
        if multi_l.any():
            out.append((BlockStore(f"{t}~l|{m}", c.split, blk[multi_l], meta=dict(meta, part="within_loc")), wl))
        if multi_g.any():
            out.append((BlockStore(f"{t}~gl|{m}", c.split, blk[multi_g], meta=dict(meta, part="between_loc_within_grid")), bl))
        return out

    def run(self):
        if self.xs == "x25":
            return W.R9Unit.run(self)
        c, v = self.c, self.xs
        suf = RG.method_suffix(v)
        for n, d in W.cells_for(self.a, H_EXP, self.nA):
            sel = self.draw(n, d); nl, nb = len(sel), self.nb(sel)
            self.trace("select", sel, n=n, draw=str(d))
            E1, E2 = self.coefs(sel)
            self.physics(n, d, sel, E1, E2, which=("P1",))                  # x25 조각의 P1 과 같은 키·같은 값
            lcv, K, cf = self.lam_cv("R1", LO, v, sel, n, d, suf)
            a1A, a1B = E1 * c.sA[sel], E1 * c.sB
            for seed in self.seeds:
                (p,) = self.fit_direct(HI, "D0" + suf, sel, seed, n, d, xs=v)
                self.add("D0" + suf, HI, "cell", n, d, seed, 1.0, p, np.nan, nl, nb, flag=self.F.last_flag, nrow=nl)
                (g,) = self.fit_resid(LO, "R1" + suf, sel, a1A, seed, n, d, xs=v)
                self.emit("R1" + suf, LO, n, d, seed, a1B, g, E1, nl, nb, lcv, K, cf, self.F.last_flag, nrow=nl)
        return self


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XE_hires_covariates(계획 2.5) 하네스")
    ap.add_argument("--stage", default="r1b", choices=["r1b", "r3", "all", "xm"],
                    help="작업 묶음: r1b = x25·xh0·xt2(R1b), r3 = x25·xh·SI 변형(R3), xm = xw·xw_lc(XM 추가 등록, x25 는 XE r1b 조각 재사용)")
    ap.add_argument("--feat-xm", default=str(XM_FEAT_DEFAULT), help="XM 특징 표 경로('none' = 세기 전용 빈 표)")
    ap.add_argument("--x25-from", default=str(XM_X25_FROM), help="XM 집계가 다시 쓰는 x25 조각 폴더(XE r1b)")
    ap.add_argument("--x25-tag", default=XM_X25_TAG, help="XM 집계가 다시 쓰는 x25 조각의 tag")
    ap.add_argument("--variants", default="", help="변형 쉼표 목록(주면 --stage 의 목록 대신 쓴다)")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--grid", default=W._grid_txt(GRID))
    ap.add_argument("--splits", default="", help="분할 목록(쉼표) 또는 K(1..K). 기본 1–25")
    ap.add_argument("--seeds", type=int, default=len(XB.SEEDS))
    ap.add_argument("--draws-cap", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="프로세스 수. 0 = 풀 없이 차례로")
    ap.add_argument("--cb-iters", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--feat", default=str(FEAT_DEFAULT), help="특징 표 경로('none' = 세기 전용, 모든 군 포함으로 센다. 적합에는 쓰지 않는다)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--shard", default="", help="'대상:분할:변형' 쉼표 목록 또는 'K/N'(단위 목록의 K번째 몫, 1부터)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="작은 설정(캐나다, 분할 7, n 200·전량, 추출 1, seed 1). 로컬은 스레드 2, 코어 4개 이하")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--feat-gate-only", action="store_true", help="xt2 표 열람 전 점검: 조각의 H0·T2 군 열 해시와 커밋된 최종판 대조(해시만 읽는다)")
    ap.add_argument("--xe-e", action="store_true", help="XE-e 현장 VWC 상한 진단(봉인 폴더에 쓴다)")
    ap.add_argument("--xe-e-prep", action="store_true", help="XE-e 의 VWC 특징 표 준비(좌표·VWC 만 읽는다, 세기 범주)")
    ap.add_argument("--vwc", default=str(OUT_DEFAULT / "inputs" / "xe_vwc_v1.csv"), help="XE-e 의 셀별 VWC 표")
    ap.add_argument("--above", default=str(ROOT / "data" / "raw" / "above" / "ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv"))
    ap.add_argument("--allow-local", action="store_true")
    ap.add_argument("--allow-unfinal", action="store_true", help="최종 해시 없이 2단계 변형을 돈다(스모크·세기 전용, 결과는 판정에 쓰지 않는다)")
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    ap.add_argument("--gate-wf9", nargs="?", const=str(WF9_SHARDS), default=str(WF9_SHARDS),
                    help="집계에서 봉인 표보다 먼저 x25 저장소를 WF9 조각과 대조한다(재현 관문, 기본 켬). 'none' 이면 끄고 판정은 '판정 불가(관문 미실시)'")
    ap.add_argument("--gate-level", default="", choices=[""] + list(XB.GATE_TOL), help="기본: 스모크 local_rescale, 그 밖 elm_hematite")
    ap.add_argument("--pool-retries", type=int, default=2)
    ap.add_argument("--min-mem-gb", type=float, default=30.0, help="로컬 시작 전 가용 메모리 하한(1절, 0 = 확인 안 함)")
    ap.add_argument("--mem-wait-s", type=float, default=1800.0, help="가용 메모리가 하한 아래일 때 기다리는 최대 시간(초)")
    ap.add_argument("--max-mem-gb", type=float, default=10.0, help="로컬 프로세스 상주 메모리 상한(1절 작업 하나 10 GB, 0 = 감시 안 함)")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    return finalize(a)


def run_mode(a) -> str:
    """xbatch_core.guard 의 모드. 세기 범주(세기, XE-e 준비, xt2 열람 점검), 집계, 스모크, 본 실행(XE-e 포함)."""
    if a.count_only or a.xe_e_prep or a.feat_gate_only:
        return "count"
    if a.summarize_only:
        return "summarize"
    return "smoke" if a.smoke else "run"


def finalize(a):
    a.PERMIT = XB.run_permitted(a.ARGV)
    vs = [v for v in a.variants.split(",") if v] if a.variants else (list(RG.STAGE_VARIANTS[a.stage]) if a.stage != "all" else list(RG.VARIANTS_ALL))
    bad = [v for v in vs if v not in RG.VARIANTS_ALL + RG.XM_VARIANTS]
    if bad:
        raise SystemExit(f"알 수 없는 변형 {bad}")
    if a.stage == "xm" and any(v not in RG.XM_VARIANTS for v in vs):
        raise SystemExit("[거부] --stage xm 은 xw·xw_lc 만 적합한다(x25 는 XE r1b 조각을 다시 쓴다, 등록 2절)")
    if a.stage != "xm" and any(v in RG.XM_VARIANTS for v in vs):
        raise SystemExit("[거부] xw·xw_lc 는 --stage xm 에서만 돈다(산출 폴더와 집계가 XM 이다)")
    if a.stage == "xm" and a.out_dir == str(OUT_DEFAULT):                 # XM 산출 경로(등록 10절)
        a.out_dir = str(XM_OUT)
    if a.allow_unfinal and not (a.smoke or a.count_only):            # 본 실행·집계에서는 받지 않는다(계획 2.5: xh·SI 는 최종판 커밋 뒤 R3)
        raise SystemExit("[거부] --allow-unfinal 은 스모크(--smoke)와 세기(--count-only)에서만 받는다")
    a.VARIANTS = vs
    if a.splits:
        sp = [int(v) for v in a.splits.split(",") if v]
        a.SPLIT_LIST = sp if len(sp) > 1 or "," in a.splits else list(range(1, sp[0] + 1))
    else:
        a.SPLIT_LIST = list(SPLITS)
    a.TARGETS = [t for t in a.targets.split(",") if t]
    grid = W._n_list(a.grid)
    if a.smoke:
        a.TARGETS, a.SPLIT_LIST, grid = ["Canada"], [7], [200, -1]
        a.draws_cap = a.draws_cap or 1
        a.seeds = 1
        a.nboot = min(int(a.nboot), 500)
        if not a.variants:
            a.VARIANTS = list(RG.STAGE_VARIANTS["xm" if a.stage == "xm" else "r1b"])
        if not a.PERMIT:
            a.threads = min(int(a.threads), SMOKE_MAX_THREADS)      # 제한 스모크(스레드 2)
    if a.xe_e or a.xe_e_prep:
        a.TARGETS = [t for t in a.TARGETS if t in XE_E_TARGETS]
    a.NBOOT_ASKED = int(a.nboot)
    a.threads, a.nboot = XB.local_limits(a.threads, a.nboot, a.ARGV)
    a.TAG = (a.tag or ("xe_e" if (a.xe_e or a.xe_e_prep) else (XM_TAG if a.stage == "xm" else f"xe_{a.stage}"))) + ("_smoke" if a.smoke else "")
    a.OUT = Path(a.out_dir) if os.path.isabs(a.out_dir) else ROOT / a.out_dir
    # 봉인 폴더 = <OUT>/sealed(기본 data/processed/xbatch/XE_hires_covariates/sealed), 스모크는 <OUT>/smoke/sealed
    a.SEALED_ROOT, a.SEALED_NAME = a.OUT.parent, a.OUT.name + ("/smoke" if a.smoke else "")
    # 조각: 본 실행은 <OUT>/shards. 스모크 조각은 실제 자료의 적합 결과(RMSE·계수)를 담으므로 봉인 폴더 안(<OUT>/smoke/sealed/shards)에 둔다(0.3)
    a.SHARDS = (a.OUT / "smoke" / "sealed" / "shards") if a.smoke else (a.OUT / "shards")
    a.GATE_LEVEL = a.gate_level or ("local_rescale" if (a.smoke or a.stage == "xm") else "elm_hematite")   # XM: 로컬 x25 조각 대 WF9(등록 10절)
    a.h = XB.h54_args(exp=H_EXP, splits=a.SPLIT_LIST, grid=grid, threads=a.threads, cb_iters=a.cb_iters, seeds=a.seeds, nboot=a.nboot,
                      data_dir=a.data_dir, tag=a.TAG, draws_cap=a.draws_cap, allow_local=a.PERMIT)
    a.GRID = list(grid)
    a.X25_FROM = Path(a.x25_from) if os.path.isabs(a.x25_from) else ROOT / a.x25_from
    a.X25_TAG = str(a.x25_tag)
    fx = None if str(a.feat_xm).lower() in ("", "none") else (Path(a.feat_xm) if os.path.isabs(a.feat_xm) else ROOT / a.feat_xm)
    if a.stage == "xm":
        if fx is not None and not fx.exists():
            if a.count_only:
                fx = None
            else:
                raise SystemExit(f"XM 특징 표가 없다: {fx}(scripts/1_data_prep/xm_landcover_s2_features.py 로 만든다)")
        a.FEAT_XM = XMFeatTable(fx)
        a.FEAT = FeatTable(None)                                          # XM 은 XE 특징 표를 쓰지 않는다
        a.FEAT_NOTE = "XM: XE 특징 표 미사용"
        return a
    a.FEAT_XM = XMFeatTable(None)
    fp = None if str(a.feat).lower() in ("", "none") else (Path(a.feat) if os.path.isabs(a.feat) else ROOT / a.feat)
    a.FEAT_NOTE = ""
    if fp is not None and not fp.exists():
        if a.count_only or a.summarize_only or a.xe_e or a.xe_e_prep or set(a.VARIANTS) <= {"x25"}:
            a.FEAT_NOTE, fp = f"특징 표 없음({fp.name}): 빈 표로 둔다", None
        elif a.feat_gate_only:
            raise SystemExit(f"[거부] xt2 열람 점검에는 최종판 특징 표가 필요하다: {fp}")
        else:
            raise SystemExit(f"특징 표가 없다: {fp}(scripts/1_data_prep/xe_point_covariates.py 로 만든다)")
    a.FEAT = FeatTable(fp)
    return a


# ================================================================ 로컬 자원과 실행 전 점검
_WATCHDOG: dict = {}


def rss_mb() -> float:
    """이 프로세스의 현재 상주 메모리(MB, /proc/self/status 의 VmRSS)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def rss_watchdog(gb, poll_s=2.0) -> bool:
    """1절 로컬 자원 '작업 하나 10 GB 이하'의 강제: 상주 메모리가 상한을 넘으면 기록을 남기고 프로세스를 끝낸다(종료 코드 3, XF·XH 와 같은 방식).
    주소 공간 상한(prlimit --as)은 CatBoost 의 가상 예약에 걸려 적합 안에서 멈출 수 있으므로 MALLOC_ARENA_MAX 가 있을 때만 함께 건다."""
    if gb is None or float(gb) <= 0 or _WATCHDOG.get("on"):
        return False
    lim = float(gb) * 1024.0

    def run():
        while True:
            r = rss_mb()
            if np.isfinite(r) and r > lim:
                sys.__stderr__.write(f"[자원] 상주 메모리 {r:.0f} MB 가 상한 {lim:.0f} MB 를 넘어 끝낸다(계획 1절 로컬 자원)\n")
                sys.__stderr__.flush()
                os._exit(3)
            time.sleep(poll_s)
    threading.Thread(target=run, name="xe_rss_watchdog", daemon=True).start()
    _WATCHDOG["on"] = True
    return True


def local_resources(a) -> dict:
    """허용 표지가 없는 실행(로컬 스모크·세기·집계·XE-e 준비): 시작 전 가용 메모리 확인(하한 30 GB, 모자라면 기다린다), 상주 메모리 10 GB 감시,
    MALLOC_ARENA_MAX 가 있으면 주소 공간 상한(10 GB, XI·XJ 와 같은 조건). 허용 표지가 있으면 하지 않는다(Rescale 노드)."""
    out = dict(permitted=bool(a.PERMIT), mem_available_gb=None, watchdog=False, as_limit=False)
    if a.PERMIT:
        return out
    if float(a.min_mem_gb) > 0:
        out["mem_available_gb"] = round(float(XB.require_memory(float(a.min_mem_gb), wait_s=float(a.mem_wait_s), poll_s=60.0)), 1)
    if float(a.max_mem_gb) > 0:
        out["watchdog"] = rss_watchdog(float(a.max_mem_gb))
        if os.environ.get("MALLOC_ARENA_MAX", ""):
            out["as_limit"] = XB.limit_memory(float(a.max_mem_gb))
    return out


def check_smoke_env(a, mode):
    """제한 스모크(계획 0.3·1절: 코어 4개, 스레드 2)를 허용 표지가 없는 스모크에 강제한다. 코어 묶음이 4개를 넘으면 거부한다."""
    if mode != "smoke" or a.PERMIT:
        return
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
        if n > SMOKE_MAX_CORES:
            raise SystemExit(f"[거부] 스모크 코어 묶음 {n}개 > {SMOKE_MAX_CORES}(taskset -c <코어 4개> 로 묶는다, 계획 1절 로컬 자원)")
    if int(a.threads) > SMOKE_MAX_THREADS:
        raise SystemExit(f"[거부] 스모크 스레드 {a.threads} > {SMOKE_MAX_THREADS}")


def check_feat_for_fit(a, mode):
    """적합(본 실행·스모크)에 빈 특징 표를 쓰지 않는다. x25 밖의 변형이 있으면 거부한다(고해상 열이 모두 NaN 인 채 적합되지 않게)."""
    if mode in ("run", "smoke") and a.FEAT.is_dummy and any(v != "x25" and not is_xm(v) for v in a.VARIANTS):
        raise SystemExit("[거부] 적합에 빈 특징 표(--feat none 또는 없는 파일)를 쓸 수 없다. x25 밖의 변형에는 xe_feat 표와 메타가 필요하다")
    if mode in ("run", "smoke") and a.FEAT_XM.is_dummy and any(is_xm(v) for v in a.VARIANTS):
        raise SystemExit("[거부] XM 변형의 적합에 빈 특징 표를 쓸 수 없다(xm_feat_v1.csv 와 메타가 필요하다, 등록 3절)")


def check_requested(a, skipped, mode):
    """요청한 변형 가운데 '최종 해시 없음'으로 빠진 것이 있으면 적합 전에 거부한다(본 실행·스모크). 90 % 규칙의 군 제외는 정상 건너뜀이다."""
    if mode not in ("run", "smoke"):
        return
    miss = sorted({s_["variant"] for s_ in skipped if s_.get("status") == FINAL_REQUIRED_TXT})
    if miss:
        raise SystemExit(f"[거부] 변형 {miss} 은 xe_feat 최종 해시를 개정 이력에 커밋한 뒤에만 돈다(계획 2.5·0.3). "
                         f"최종 해시: {a.FEAT.final_commit.get('how')}")


# ================================================================ 문맥과 실행
def split_info(a, target):
    _, D = W.get_data(a.h)
    return W.split_structure_ext(D, target, a.SPLIT_LIST)


def split_keep(a, target):
    _, D = W.get_data(a.h)
    info = split_info(a, target)
    keep, skip, _ = W.split_plan(D, target, a.SPLIT_LIST, info)
    return keep, skip, info


def build_ctx(a, target, split, variant):
    """WF9 와 같은 지역 내 문맥(h54.build_rctx, 분할 1–25 구조)에 변형 열을 붙인다. A·채점 색인은 h54.build_rctx 와 같은 규칙으로 다시 구하고
    라벨·√TDD·블록이 문맥과 같은지 단언한다(특징 표의 loc_id 결합 위치 확인)."""
    info = split_info(a, target)[int(split)]
    c = W.build_rctx(a.h, target, int(split), info)
    cols, vinfo = variant_cols(a, variant, target)
    if cols is None:
        raise RuntimeError(f"{target}: 변형 {variant} 을 돌리지 않는다({vinfo['skip']})")
    if variant == "x25":
        return c, cols, vinfo
    _, D = W.get_data(a.h)
    df = D.df
    t_idx = D.target_idx(target)
    A_idx, B_idx = XB.half_split_blocks(df, t_idx, int(split))
    evB = B_idx[XB.eval_mask(df.iloc[B_idx])]
    for nm, idx, yy, ss, bb in (("A", A_idx, c.yA, c.sA, c.blkA), ("B", evB, c.yB, c.sB, c.blkB)):
        if not (len(idx) == len(yy) and np.allclose(df.y.values[idx].astype(float), yy, equal_nan=True)
                and np.allclose(df.s.values[idx].astype(float), ss, equal_nan=True) and np.array_equal(df.block.values[idx], bb)):
            raise RuntimeError(f"{target} 분할 {split}: {nm} 색인이 h54.build_rctx 와 다르다")
    base = [col for col in cols if col in XB.FEATS]
    extra = [col for col in cols if col not in XB.FEATS]
    ft = feat_of(a, variant)

    def mat(idx):
        M0 = df[base].values[idx].astype(np.float32)
        M1 = ft.matrix(df.loc_id.values[idx], extra) if extra else np.zeros((len(idx), 0), np.float32)
        return np.hstack([M0, M1])
    attach_variant(c, variant, mat(A_idx), mat(evB))
    c.meta.update(xe_cols=len(cols), xe_extra=len(extra))
    return c, cols, vinfo


def unit_cfg(a, variant, cols):
    """설정 요약. 변형마다 다른 항목(특징 표 해시, 열 수)은 data_sha 에 넣는다(공통 해시에서 빠지는 항목, h54 CFG_UNIT_KEYS 와 같다)."""
    ft = feat_of(a, variant)
    feat_sha = "" if variant == "x25" else ("dummy" if ft.is_dummy else ft.sha[:16])
    dsha = XB.data_sha(a.h) + ":" + feat_sha + ":" + str(len(cols))
    return XB.make_unit_cfg("xe", variant, dsha, grid=list(a.GRID), draws=DRAWS, draws_cap=int(a.draws_cap), seeds=list(range(int(a.seeds))),
                            cb_iters=int(a.cb_iters), splits=list(a.SPLIT_LIST), wf9_group="air", loc_deg=LOC_DEG,
                            methods_variant=["P1", "R1(λ cv, 0.25, 0.5, 1.0)", "D0(catboost)"], methods_x25="h54.R9Unit.run", plan_section=RG.PLAN_SECTION)


def run_unit(a, target, split, variant, dry=False, expected=None):
    t0 = time.time()
    c, cols, vinfo = build_ctx(a, target, split, variant)
    U = XERUnit(a.h, c, variant, dry)
    U.run()
    rows, stores, stats = U.finish()
    if dry:
        fac = len(cols) / float(len(XB.FEATS))
        return dict(target=target, split=int(split), variant=variant, n_A=len(c.yA), n_eval=len(c.yB), n_cols=len(cols), n_rows=stats["n_rows"],
                    fit_total=int(sum(stats["n_fit"].values())), est_s=round(float(sum(stats["est_detail"].values())), 2),
                    est_col_s=round(float(sum(stats["est_detail"].values())) * fac, 2), n_krige=stats["n_krige"], est_krige_s=stats["est_krige_s"],
                    _detail=stats["n_fit_detail"])
    cfg = unit_cfg(a, variant, cols)
    ft = feat_of(a, variant)
    unit = {**stats, **{k: v for k, v in c.meta.items() if not isinstance(v, (dict, list))}}
    unit.update(exp="xe", stage=a.stage, target=target, mode=MODE, parent=c.parent, split=int(split), variant=variant, elapsed_s=round(time.time() - t0, 1),
                n_fit_total=int(sum(stats["n_fit"].values())), n_A=int(len(c.yA)), n_eval=int(len(c.yB)), E0=float(c.E0), cols=list(cols),
                groups=vinfo.get("groups", []), groups_dropped=vinfo.get("dropped", []), feat_file=str(ft.path) if ft.path else "",
                feat_sha256=ft.sha, feat_final_ok=ft.final_ok, feat_final_local_ok=ft.final_local_ok,
                feat_final_commit=ft.final_commit.get("how", "") if not ft.is_dummy else "", feat_group_sha=ft.group_sha(vinfo.get("groups", [])),
                allow_unfinal=bool(a.allow_unfinal), smoke=bool(a.smoke), threads=int(a.threads),
                code_sha_registry=XB.code_sha(SCRIPT_DIR / "x_hires_registry.py"))
    if is_xm(variant):                                                    # XM 추가 등록의 표지(설정 해시에는 넣지 않는다: x25 조각과 공통 설정이 같아야 한다)
        unit.update(plan_addendum=RG.XM_PLAN_DOC, plan_addendum_commit=RG.XM_PLAN_COMMIT, same_as=vinfo.get("same_as", ""), design=XM_DESIGN)
    return XB.write_shard(a.SHARDS, a.TAG, target, MODE, split, rows, stores, cfg, unit, variant, expected, __file__)


def parse_shard(spec, units):
    """--shard: 'K/N'(1부터) 이면 단위 목록의 K번째 몫(K, K+N, …), 그 밖은 '대상:분할:변형' 목록."""
    if not spec:
        return units
    m = re.fullmatch(r"(\d+)/(\d+)", spec.strip())
    if m:
        k, n = int(m.group(1)), int(m.group(2))
        if not 1 <= k <= n:
            raise SystemExit("--shard K/N 은 1 ≤ K ≤ N 이어야 한다")
        return units[k - 1::n]
    want = set()
    for s_ in spec.split(","):
        t, sp, v = s_.split(":")
        want.add((t, int(sp), v))
    out = [u for u in units if (u[0], int(u[1]), u[2]) in want]
    miss = want - {(u[0], int(u[1]), u[2]) for u in out}
    if miss:
        raise SystemExit(f"--shard 의 단위가 단위 목록에 없다: {sorted(miss)}")
    return out


def enumerate_units(a):
    units, skipped, expected = [], [], {}
    for t in a.TARGETS:
        keep, skip, _ = split_keep(a, t)
        expected[t] = keep
        for sp, why, _v in skip:
            skipped.append(dict(target=t, split=int(sp), variant="*", status=why))
        for v in a.VARIANTS:
            if needs_final(v) and not a.FEAT.is_dummy and not a.FEAT.final_ok and not a.allow_unfinal:
                skipped.append(dict(target=t, split=-1, variant=v, status=FINAL_REQUIRED_TXT))
                continue
            if v == "xw" and a.FEAT_XM.s2_all_dropped():                   # 등록 3절 (d): xw 를 돌리지 않고 xw_lc 를 주 대비에 쓴다
                skipped.append(dict(target=t, split=-1, variant=v, status=XM_S2_FALLBACK_TXT))
                continue
            cols, info = variant_cols(a, v, t)
            if cols is None:
                skipped.append(dict(target=t, split=-1, variant=v, status=info["skip"]))
                continue
            units += [(t, int(sp), v) for sp in keep]
    return parse_shard(a.shard, units), skipped, expected


_A = None
_EXPECTED: dict = {}


def _worker_init(argv, expected):
    """프로세스 풀 워커 초기화(spawn): 인자를 다시 만들고 대상별 기대 분할을 둔다."""
    global _A, _EXPECTED
    warnings.filterwarnings("ignore")
    _A = parse_args(argv)
    _EXPECTED = dict(expected)
    if not _A.PERMIT and float(_A.max_mem_gb) > 0:                        # 로컬 워커도 작업 하나 10 GB 상한을 지킨다(1절 로컬 자원)
        rss_watchdog(float(_A.max_mem_gb))


def _worker_run(target, split, variant):
    u = run_unit(_A, target, split, variant, expected=_EXPECTED.get(target))
    return dict(target=target, split=int(split), variant=variant, status=u.get("status"), n_fit=int(u.get("n_fit_total", 0)), s=u.get("elapsed_s"))


def run_all(a, units, expected):
    global _A, _EXPECTED
    todo = []
    for t, sp, v in units:
        if a.resume:
            cols, _ = variant_cols(a, v, t)
            done, _why = XB.unit_state(a.SHARDS, a.TAG, t, MODE, sp, unit_cfg(a, v, cols), v)
            if done:
                continue
        todo.append((t, sp, v))
    XB.check_out_dir(a.SHARDS)
    a.SHARDS.mkdir(parents=True, exist_ok=True)
    print(f"[run] 단위 {len(units)} · 할 일 {len(todo)} · 워커 {a.workers} · 스레드 {a.threads} · tag {a.TAG}", flush=True)
    if int(a.workers) <= 0:
        _A, _EXPECTED = a, dict(expected)
        done, failed = XB.execute(todo, _worker_run, workers=0)
    else:
        done, failed = XB.execute(todo, _worker_run, workers=a.workers, threads=a.threads, init=_worker_init, init_args=(a.ARGV, expected),
                                  retries=a.pool_retries)
    if failed:
        fp = a.OUT / f"{a.TAG}_failed_units.json"
        fp.write_text(json.dumps([dict(unit=list(u), error=e) for u, e in failed], ensure_ascii=False, indent=1))
        print(f"[run] 실패 단위 {len(failed)} → {fp.name}", flush=True)
    return done, failed


# ================================================================ 세기(라벨 값 미사용 출력)
def count_only(a, units, skipped):
    t0 = time.time()
    rows, det = [], Counter()
    for t, sp, v in units:
        r = run_unit(a, t, sp, v, dry=True)
        for k, n in r.pop("_detail").items():
            det[f"{v}|{k}"] += n
        rows.append(r)
    df = pd.DataFrame(rows)
    XB.check_out_dir(a.OUT)
    a.OUT.mkdir(parents=True, exist_ok=True)
    if not len(df):
        print(f"[count-only] 작업 단위 없음 · 건너뜀 {len(skipped)}", flush=True)
        return df
    emp = empirical_seconds(df)
    df["est_wf9_s"] = emp
    df.to_csv(a.OUT / f"{a.TAG}_count.csv", index=False)
    pd.DataFrame(skipped).to_csv(a.OUT / f"{a.TAG}_count_skipped.csv", index=False)
    by = df.groupby("variant").agg(units=("split", "size"), targets=("target", "nunique"), cols=("n_cols", "max"), fits=("fit_total", "sum"),
                                    est_h=("est_s", "sum"), est_col_h=("est_col_s", "sum"), est_wf9_h=("est_wf9_s", "sum"))
    for c_ in ("est_h", "est_col_h", "est_wf9_h"):
        by[c_] = (by[c_] / 3600).round(3)
    print(f"[count-only] 작업 단위 {len(df)} · 건너뜀 {len(skipped)} · 세기 {time.time() - t0:.0f}s · tag {a.TAG}", flush=True)
    with pd.option_context("display.width", 200):
        print(by.to_string(), flush=True)
    tot = float(df.est_col_s.sum()) / 3600
    tw = float(np.nansum(df.est_wf9_s)) / 3600
    print(f"[count-only] 총 적합 {int(df.fit_total.sum()):,} · h54 모형 추정(열 수 비례 보정) {tot:.2f} 워커·시간 · WF9 실측 환산 {tw:.2f} 워커·시간"
          f"(4스레드). 워커 22개 약 {max(tot, tw) / 22:.2f} h", flush=True)
    if skipped:
        print(f"[count-only] 건너뜀 사유: {dict(Counter(s_['status'] for s_ in skipped))}", flush=True)
    return df


def empirical_seconds(df):
    """WF9 3차 Rescale 실측(wf3b_timing.csv 의 대상별 단위 경과 시간 평균, hematite 4스레드)으로 단위 시간을 환산한다. x25 = 실측 그대로,
    그 밖 = 실측 × (R1·R1cv·D0 시간 비율) × (열 수/25)[가정: CatBoost 시간은 열 수에 비례]. 실측 표가 없으면 NaN."""
    if not WF9_TIMING.exists():
        return np.full(len(df), np.nan)
    t = pd.read_csv(WF9_TIMING)
    t = t[t.exp == "wf9"]
    per = t.groupby(["target", "split"]).agg(s=("sec", "sum")).reset_index().groupby("target").s.mean().to_dict()
    sh = t.groupby("method").sec.sum()
    share = float(sh.reindex(["R1cv", "R1", "D0"]).fillna(0).sum() / max(sh.sum(), 1e-9))
    out = []
    for r in df.itertuples():
        base = per.get(r.target, np.nan)
        out.append(base if r.variant == "x25" else base * share * (r.n_cols / float(len(XB.FEATS))))
    return np.asarray(out, float)


# ================================================================ 집계(봉인 폴더)
def gR(v, n):
    return XB.gk("R1" + RG.method_suffix(v), n, LO, LAM_CV)


def gD(v, n):
    return XB.gk("D0" + RG.method_suffix(v), n, HI, 1.0)


def gP(n):
    return XB.gk("P1", n, "none")


def contrast_specs(variants):
    """(가설, 묶음, 변형, 대비 이름, 저장소 접미사, gA 함수, gB 함수, n 목록, 판정 역할)."""
    S = []
    if "xh" in variants:
        S += [("XE-a", "xh", "xh", "R1(λ cv, xh)−R1(λ cv, x25)", "~w", lambda n: gR("xh", n), lambda n: gR("x25", n), AUX_N, "주"),
              ("XE-b", "xh", "xh", "R1(λ cv, xh)−P1", "~w", lambda n: gR("xh", n), gP, AUX_N, "주"),
              ("XE-c", "xh", "xh", "R1(λ cv, xh)−R1(λ cv, x25)", "~l", lambda n: gR("xh", n), lambda n: gR("x25", n), AUX_N, "보조"),
              ("XE-c", "xh", "xh", "R1(λ cv, xh)−P1", "~l", lambda n: gR("xh", n), gP, AUX_N, "보조"),
              ("XE-c", "xh", "xh", "D0(xh)−D0(x25)", "~w", lambda n: gD("xh", n), lambda n: gD("x25", n), AUX_N, "보조"),
              ("XE-c", "xh", "xh", "R1(λ cv, xh)−R1(λ cv, x25)", "", lambda n: gR("xh", n), lambda n: gR("x25", n), AUX_N, "보조"),
              ("XE-d", "xh", "xh", "R1(λ cv, xh)−P1", "~b", lambda n: gR("xh", n), gP, AUX_N, "서술")]
    if "xt2" in variants:
        S += [("XE-a0", "xt2", "xt2", "R1(λ cv, xt2)−R1(λ cv, x25)", "~w", lambda n: gR("xt2", n), lambda n: gR("x25", n), AUX_N, "보조"),
              ("XE-a0", "xt2", "xt2", "R1(λ cv, xt2)−P1", "~w", lambda n: gR("xt2", n), gP, AUX_N, "서술")]
    if "xh0" in variants:
        S += [("XE-c", "xh0", "xh0", "R1(λ cv, xh0)−R1(λ cv, x25)", "~w", lambda n: gR("xh0", n), lambda n: gR("x25", n), AUX_N, "보조"),
              ("XE-c", "xh0", "xh0", "R1(λ cv, xh0)−P1", "~w", lambda n: gR("xh0", n), gP, AUX_N, "보조")]
    for v in variants:
        if v in RG.VARIANTS_SI:
            for part in ("~w", "~l"):
                S += [("XE-d", "si", v, f"R1(λ cv, {v})−R1(λ cv, x25)", part, (lambda vv: lambda n: gR(vv, n))(v), lambda n: gR("x25", n), AUX_N, "서술"),
                      ("XE-d", "si", v, f"R1(λ cv, {v})−P1", part, (lambda vv: lambda n: gR(vv, n))(v), gP, AUX_N, "서술")]
    return S


def row_label(hyp, role, n):
    """n 별 행의 가설·역할. XE-a·XE-b·XE-a0 의 등록 n 은 500·1,000·전량이고, n 200 행은 XE-c(보조)에 둔다(계획 2.5 XE-c 'n 200')."""
    if hyp in ("XE-a", "XE-b", "XE-a0") and int(n) not in MAIN_N:
        return "XE-c", "보조"
    return hyp, role


def run_contrasts(tms, variants, targets):
    rows, pooled = [], {}
    for hyp0, fam, var, name, part, fA, fB, ns, role0 in contrast_specs(variants):
        names = [f"{t}{part}|{MODE}" for t in targets]
        for n in ns:
            hyp, role = row_label(hyp0, role0, n)
            rr, _ = XB.contrast_pool(tms, names, fA(n), fB(n), label=f"{name}[{PART_WORD[part]}]|n{W.nlab(n)}", kind="same", registered=len(TARGETS))
            bl, br = blind_of(hyp, var, name)
            for r in rr:
                r.update(hypothesis=hyp, hypothesis_registered=hyp0, family=fam, variant=var, contrast=name, part=PART_WORD[part], n=int(n), role=role,
                         blind=bl, blind_reason=br, design="결과 열람 뒤 설계", main_n=bool(n in MAIN_N))
                if r.get("scope") == "MEAN":
                    pooled[(hyp, name, PART_WORD[part], int(n))] = r
            rows += rr
    return rows, pooled


def branch(rule_txt, verdicts):
    """_rule3 문구와 n 별 풀 판정 → 해석 문장 갈래(우세, 열세, 동등, 미결정, 판정 불가)."""
    t = str(rule_txt)
    if t.startswith("지지") or t.startswith("부분 지지"):
        return "우세"
    if t.startswith("판정 불가"):
        return "판정 불가"
    vv = [v for v in verdicts if v is not None and v not in XB.NA_VERDICTS]
    if any(v == "열세" for v in vv):
        return "열세"
    if vv and all(v == "동등" for v in vv):
        return "동등"
    return "미결정"


MAIN_CONTRASTS = (("XE-a", "R1(λ cv, xh)−R1(λ cv, x25)"), ("XE-b", "R1(λ cv, xh)−P1"))
N_PREF = (-1, 1000, 500)                                                    # 문장 수치를 가져오는 n 의 우선 순서(전량 > 1,000 > 500)


def pick_n(pooled, hyp, name, want):
    """문장 수치를 가져올 n: 그 대비의 풀 판정이 want 인 등록 n 가운데 가장 큰 n(전량 > 1,000 > 500). 반환 (n, 행) 또는 (None, None)."""
    for n in N_PREF:
        r = pooled.get((hyp, name, "격자 안", n))
        if r is not None and r.get("verdict4") == want:
            return n, r
    return None, None


def verdict_rows(pooled, region_rows, decomp, feat_meta, targets, gate=None):
    """XE-a, XE-b 의 _rule3 판정, Holm(m = 6, 보조 열), 사전 고정 해석 문장(계획 2.5 표와 1절 공통 규칙).
    문장 수치의 출처(구현 기록 3절): 문장의 a, Holm 보정 p('보정 전 유의'), 효과 크기 표기의 Δ 는 모두 같은 대비·같은 n 에서 가져온다.
    n 은 그 대비의 풀 판정이 갈래의 기준(우세 또는 열세)을 충족한 등록 n 가운데 가장 큰 n 이다. 첫째 경우(aw_bw)의 a 와 x 는 XE-b 의 그 n,
    둘째 경우(aw_bn·aw_bl)와 열세 경우의 a 는 XE-a 의 그 n 이다. 풀 문장 뒤에 XE-a·XE-b 지역 행의 열세(등록 n 전부)를 덧붙인다.
    gate = 재현 관문 결과(dict). 통과하지 않았으면 두 가설을 '판정 불가(관문 …)'로 쓴다(계획 1절: 관문을 넘으면 판정을 멈춘다)."""
    out, hol_lab, hol_p = [], [], []
    res = {}
    for hyp, name in MAIN_CONTRASTS:
        items = []
        for n in MAIN_N:
            r = pooled.get((hyp, name, "격자 안", n))
            items.append((f"n={W.nlab(n)}", None if r is None else r.get("verdict4")))
            hol_lab.append(f"{hyp}|n{W.nlab(n)}")
            hol_p.append(None if (r is None or r.get("undetermined")) else r.get("p_two"))
        res[hyp] = (XB.rule3(items, "우세"), items)
    ht = XB.holm_table(hol_lab, hol_p, HOLM_M)
    hp = dict(zip(ht.label, ht.p_holm))
    gate_txt = "" if gate is None or gate.get("passed") else str(gate.get("status", "관문 미실시"))
    if gate_txt:
        for hyp in res:
            res[hyp] = (f"판정 불가(관문: {gate_txt})", res[hyp][1])
    ba = branch(res["XE-a"][0], [v for _, v in res["XE-a"][1]])
    bb = branch(res["XE-b"][0], [v for _, v in res["XE-b"][1]])
    worse = [(f"{RG_NAME.get(str(r['target']).split('~')[0].split('|')[0], r['target'])}({r['hypothesis']}, n {W.nlab(r['n'])})", r["delta"], r["ci_lo"],
              r["ci_hi"]) for r in region_rows if r.get("hypothesis") in ("XE-a", "XE-b") and r.get("part") == "격자 안" and int(r.get("n", 0)) in MAIN_N
             and r.get("contrast") in dict(MAIN_CONTRASTS).values() and r.get("worse")]
    map_change = False
    src = dict(hyp="", n=None, delta=np.nan, holm=np.nan, x=np.nan, holm_other="", holm_marker=np.nan)

    def take(hyp, name, want):
        n_, r_ = pick_n(pooled, hyp, name, want)
        if r_ is None:
            return np.nan, None, np.nan
        d_ = float(r_["delta"])
        src.update(hyp=hyp, n=n_, delta=d_, holm=float(hp.get(f"{hyp}|n{W.nlab(n_)}", np.nan)))
        return abs(d_), n_, d_

    if ba == "우세":
        if bb == "우세":
            key = "aw_bw"
            # 첫째 문장은 XE-a(설명 비율 x)와 XE-b(a cm)를 함께 주장한다. 수치 a·Δ 는 XE-b 의 기준 충족 n 에서, '보정 전 유의' 표지는 두 가설의
            # (각자 기준 충족 n 의) Holm 보정 p 가운데 큰 값으로 정한다(어느 한 가설이 보정 뒤 유의하지 않으면 표지를 붙인다. 1절 규칙의 보수적 읽기)
            na_, ra_ = pick_n(pooled, "XE-a", MAIN_CONTRASTS[0][1], "우세")
            holm_a = float(hp.get(f"XE-a|n{W.nlab(na_)}", np.nan)) if ra_ is not None else np.nan
            a_, nb_, d_ = take("XE-b", MAIN_CONTRASTS[1][1], "우세")
            x = expl_ratio(decomp, "xh", nb_, "~w") if nb_ is not None else float("nan")
            src["x"] = x
            src["holm_other"] = f"XE-a|n{W.nlab(na_)}={holm_a:.4g}" if ra_ is not None else "XE-a 행 없음"
            src["holm_marker"] = float(np.nanmax([src["holm"], holm_a])) if np.isfinite([src["holm"], holm_a]).any() else np.nan
        else:
            key = "aw_bl" if bb == "열세" else "aw_bn"
            a_, _, d_ = take("XE-a", MAIN_CONTRASTS[0][1], "우세")
            x = float("nan")
            src["holm_marker"] = src["holm"]
        base = SENT[key].format(x=f"{x:.1f}" if np.isfinite(x) else "x", a=f"{a_:.2f}" if np.isfinite(a_) else "a")
        sent = XB.compose_sentence({"우세": base}, "우세", src["holm_marker"], d_, worse)
        map_change = True                                                 # 해석 표 첫째·둘째 경우에만 지도 문구를 고친다
    elif ba == "열세":
        a_, _, d_ = take("XE-a", MAIN_CONTRASTS[0][1], "열세")
        src["holm_marker"] = src["holm"]
        sent = XB.compose_sentence({"열세": SENT["worse"].format(a=f"{a_:.2f}" if np.isfinite(a_) else "a")}, "열세", src["holm"], d_, worse)
    elif ba == "동등":
        sent = XB.compose_sentence({"동등": SENT["eq"]}, "동등", worse_regions=worse)
    elif ba == "미결정":
        w100 = ", ".join(f"{RG_NAME.get(t, t)} {WITHIN_100M[t]} %" for t in TARGETS if t in WITHIN_100M)
        sent = XB.compose_sentence({"미결정": SENT["und"]}, "미결정", worse_regions=worse)
        sent += f" 100 m 셀 안 분산 비율은 {w100} 이다(within_grid_inputs.md 3.3). XE 는 남은 변동의 원인을 구별하지 않는다."
    else:
        sent = XB.compose_sentence({"판정 불가": SENT["na"]}, "판정 불가", reason=res["XE-a"][0])
    tot_small = small_total_targets(region_rows)
    tail = []
    if tot_small:
        tail.append(f"총 RMSE 차가 0.5 cm 미만인 대상: {', '.join(tot_small)}.")
    canada_o = ((feat_meta or {}).get("group_decisions", {}).get("Canada", {}).get("O", {}) or {})
    if canada_o and not canada_o.get("included", True):
        tail.append("O 군(SoilGrids 250 m, NCSCD)은 캐나다에서 90 % 규칙으로 빠졌다.")
    tail.append("지도 문구('1 km 셀로 표시한 기후 격자 단위 보정 지도')는 " + ("고친다." if map_change else "고치지 않는다."))
    tail.append("XE-e 는 판정을 바꾸지 않는다.")
    for hyp in ("XE-a", "XE-b"):
        txt, items = res[hyp]
        bl, br = blind_of(hyp)
        out.append(dict(hypothesis=hyp, scope="verdict", verdict=txt, items="; ".join(f"{k} {v}" for k, v in items), branch=ba if hyp == "XE-a" else bb,
                        holm_p=";".join(f"{k}={hp[k]:.4g}" for k in hp if k.startswith(hyp)), holm_m=HOLM_M, blind=bl, blind_reason=br,
                        design="결과 열람 뒤 설계", gate=gate_txt or "통과"))
    out.append(dict(hypothesis="XE-a·XE-b", scope="sentence", verdict="", sentence=sent + " " + " ".join(tail), branch=f"{ba}/{bb}",
                    map_change=map_change, expl_within_xh=src["x"], sentence_source_hyp=src["hyp"],
                    sentence_n=W.nlab(src["n"]) if src["n"] is not None else "", sentence_delta=src["delta"], sentence_holm_p=src["holm"],
                    sentence_holm_p_other=src["holm_other"], sentence_holm_p_marker=src["holm_marker"],
                    sentence_rule="a·Δ·Holm p 는 같은 대비·같은 n(기준 충족 등록 n 가운데 가장 큰 n, 전량 > 1,000 > 500). 첫째 문장(XE-a·XE-b 모두 우세)의 "
                                  "'보정 전 유의' 표지는 두 가설의 Holm p 가운데 큰 값으로 정한다", gate=gate_txt or "통과"))
    return pd.DataFrame(out), ht


RG_NAME = {"Alaska": "알래스카", "Lena": "레나", "Canada": "캐나다"}


def small_total_targets(region_rows):
    """총 RMSE 차 |R1(xh) − R1(x25)| < 0.5 cm 인 대상(n 전량, 지역 행, 셀 가중 점 추정. 해석 표 '공통' 행)."""
    out = []
    for r in region_rows:
        if r.get("hypothesis") == "XE-c" and r.get("contrast") == MAIN_CONTRASTS[0][1] and r.get("part") == "총" and int(r.get("n", 0)) == -1 \
                and r.get("scope") == "region" and np.isfinite(r.get("delta", np.nan)) and abs(float(r["delta"])) < 0.5:
            out.append(RG_NAME.get(str(r["target"]).split("|")[0], str(r["target"])))
    return out


def decomp_table(tms):
    """XE-d 서술 표: 대상·방법·n 별 저장소(총, ~w, ~b, ~l, ~gl)의 RMSE, 설명 비율 1 − MSE_part(M)/MSE_part(P1)(격자 안·위치 안),
    블록 합 항등식 점검(SSE_tot − SSE_w − SSE_b, SSE_w − SSE_gl − SSE_l). 판정에 쓰지 않는다."""
    rt = W.store_rmse_table({"xe": tms})
    if not len(rt):
        return pd.DataFrame()
    rt["base"] = rt.store.str.replace(r"~(w|b|l|gl)\|", "|", regex=True)
    rt["part"] = rt.store.str.extract(r"(~gl|~w|~b|~l)\|")[0].fillna("tot")
    key = ["base", "method", "learner", "placement", "n", "lam"]
    wide = rt.pivot_table(index=key, columns="part", values=["rmse", "sse_mean"], aggfunc="first")
    wide.columns = [f"{a_}_{b_}" for a_, b_ in wide.columns]
    wide = wide.reset_index()
    ref = wide[(wide.method == "P1") & (wide.learner == "none")].set_index(["base", "n"])
    rows = []
    for r in wide.to_dict("records"):
        k = (r["base"], r["n"])
        out = dict(r)
        if k in ref.index:
            p1 = ref.loc[k]
            p1 = p1.iloc[0] if isinstance(p1, pd.DataFrame) else p1
            for part in ("~w", "~l", "~gl", "tot"):
                a_, b_ = r.get(f"rmse_{part}", np.nan), p1.get(f"rmse_{part}", np.nan)
                out[f"expl{part.replace('~', '_')}"] = 1.0 - (a_ ** 2) / (b_ ** 2) if np.isfinite(a_) and np.isfinite(b_) and b_ > 0 else np.nan
        out["check_tot_w_b"] = r.get("sse_mean_tot", np.nan) - np.nan_to_num(r.get("sse_mean_~w", np.nan)) - r.get("sse_mean_~b", np.nan)
        out["check_w_gl_l"] = np.nan_to_num(r.get("sse_mean_~w", np.nan)) - np.nan_to_num(r.get("sse_mean_~gl", np.nan)) - np.nan_to_num(r.get("sse_mean_~l", np.nan))
        out["variant"] = r["method"].split("@")[1] if "@" in r["method"] else "x25"
        rows.append(out)
    return pd.DataFrame(rows)


def expl_ratio(decomp, variant, n, part="~w"):
    """설명 비율(%) = 100·(1 − MSE_part(R1 교차검증 λ, 변형)/MSE_part(P1))의 대상 평균."""
    if decomp is None or not len(decomp):
        return float("nan")
    m = "R1" + RG.method_suffix(variant)
    q = decomp[(decomp.method == m) & (decomp.learner == LO) & (decomp.n == n) & (decomp.lam == LAM_CV)]
    col = f"expl{part.replace('~', '_')}"
    if not len(q) or col not in q:
        return float("nan")
    return float(100.0 * np.nanmean(q[col].values))


def feat_gate(units, feat):
    """xe_feat 정합: 조각에 적힌 군 열 해시(작업 당시 표)가 커밋된 최종판의 군 열 해시와 같은가(R1b 의 1단계 표와 최종판의 H0·T2 비교)."""
    fin = (feat.final or {}).get("group_cols_sha256", {}) if feat and feat.final_ok else {}
    rows = []
    for u in units:
        for g, s_ in (u.get("feat_group_sha") or {}).items():
            rows.append(dict(target=u.get("target"), split=u.get("split"), variant=u.get("variant"), group=g, unit_sha=s_[:16], final_sha=fin.get(g, "")[:16],
                             same=bool(fin.get(g)) and fin.get(g) == s_, final_present=bool(fin)))
    return pd.DataFrame(rows, columns=["target", "split", "variant", "group", "unit_sha", "final_sha", "same", "final_present"])


def feat_gate_check(a):
    """xt2 표 열람 전 점검(계획 0.3 열람 순서 2: xt2 표는 xe_feat 최종판 해시를 개정 이력에 커밋한 뒤 연다). R1b 조각 unit.json 의 H0·T2 군 열 해시가
    커밋된 최종판의 군 열 해시와 모두 같아야 통과다. 라벨 값은 읽지 않고 해시만 비교한다. 반환 종료 코드(0 통과, 1 불통과)."""
    if a.FEAT.is_dummy:
        raise SystemExit("[거부] --feat-gate-only 에는 최종판 특징 표가 필요하다")
    sh = [s_ for s_ in XB.find_shards(a.SHARDS, a.TAG) if s_["variant"] in ("xh0", "xt2")]
    units = [json.loads(s_["unit"].read_text()) for s_ in sh]
    fg = feat_gate(units, a.FEAT)
    n_xt2 = sum(1 for u in units if u.get("variant") == "xt2")
    ok = bool(a.FEAT.final_ok and len(fg) and fg.same.all() and n_xt2 > 0)
    XB.check_out_dir(a.OUT)
    a.OUT.mkdir(parents=True, exist_ok=True)
    fg.to_csv(a.OUT / f"{a.TAG}_feat_gate.csv", index=False)
    summ = dict(tag=a.TAG, feat_file=str(a.FEAT.path), feat_sha256=a.FEAT.sha, final_local_ok=a.FEAT.final_local_ok, final_commit=a.FEAT.final_commit,
                n_units=len(units), n_xt2_units=n_xt2, n_rows=int(len(fg)), n_same=int(fg.same.sum()) if len(fg) else 0, xt2_open_ok=ok,
                checked=time.strftime("%Y-%m-%d %H:%M:%S"))
    (a.OUT / f"{a.TAG}_feat_gate_check.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    print(f"[feat-gate] 조각 {len(units)}(xt2 {n_xt2}) · 군 행 {len(fg)} · 같음 {summ['n_same']} · 최종판 커밋 {a.FEAT.final_commit.get('how')} "
          f"{'확인' if a.FEAT.final_ok else '없음'} · xt2 표 열람 조건 {'충족' if ok else '불충족'} → {a.TAG}_feat_gate_check.json", flush=True)
    return 0 if ok else 1


def expected_x25(units, targets) -> set:
    """재현 관문의 기대 x25 단위 (대상, 분할): 조각 unit.json 의 expected_splits 합집합. 조각이 하나도 없는 대상은 (대상, −1)로 둔다."""
    exp = set()
    seen = set()
    for u in units:
        t = str(u.get("target"))
        seen.add(t)
        for sp in (u.get("expected_splits") or []):
            exp.add((t, int(sp)))
    for t in targets:
        if t not in seen:
            exp.add((t, -1))
    return exp


def gate_coverage(mine, ref) -> dict:
    """재현 관문 포괄률. mine·ref = {(저장소, 분할): BlockStore}. x25 조각의 (저장소, 분할, 키) 가운데 WF9 에 있는 비율과, 같은 (저장소, 분할)의
    WF9 키 가운데 x25 조각에 있는 비율."""
    n_m = f_m = n_r = f_r = 0
    miss_pairs = []
    for k, st in mine.items():
        keys = list(st.keys)
        n_m += len(keys)
        if k not in ref:
            miss_pairs.append(k)
            continue
        rk = list(ref[k].keys)
        rset, mset = set(rk), set(keys)
        f_m += sum(1 for kk in keys if kk in rset)
        n_r += len(rk)
        f_r += sum(1 for kk in rk if kk in mset)
    return dict(n_keys_x25=n_m, n_keys_x25_in_wf9=f_m, coverage_x25_in_wf9=(f_m / n_m) if n_m else 0.0,
                n_keys_wf9=n_r, n_keys_wf9_in_x25=f_r, coverage_wf9_in_x25=(f_r / n_r) if n_r else 0.0, n_pairs_missing_in_wf9=len(miss_pairs))


def gate_wf9(a, units, mine_dir=None, mine_tag=None, out_tag=None):
    """재현 관문(계획 2.5): x25 의 총·격자 안·격자 사이 저장소가 WF9 조각과 같다(1절 허용 오차). 봉인 표보다 먼저 돈다.
    mine_dir·mine_tag 를 주면 그 폴더·tag 의 x25 조각을 본다(XM 은 XE r1b 의 x25 조각을 다시 쓴다). 표 이름은 <out_tag 또는 tag>_gate_wf9.csv.
    통과 조건: 공통 키 1개 이상, 실패 0, x25 키의 WF9 포괄률 1, 같은 (저장소, 분할)의 WF9 키의 x25 포괄률 1(스모크는 보지 않는다),
    기대 x25 단위(대상, 분할) 누락 없음. 표에는 SSE 차와 수만 있다. 반환 dict(봉인 메타와 가설 표에 적는다)."""
    st = dict(level=a.GATE_LEVEL, ref=str(a.gate_wf9), ran=False, passed=False, n_keys=0, n_fail=0, n_x25_units=0, n_x25_units_expected=0,
              missing_x25_units="")
    if not a.gate_wf9 or str(a.gate_wf9).lower() == "none":
        return dict(st, status="관문 미실시(--gate-wf9 none)")
    ref_sh = XB.find_shards(a.gate_wf9, "wf9")
    mine_sh = [s_ for s_ in XB.find_shards(mine_dir or a.SHARDS, mine_tag or a.TAG) if s_["variant"] == "x25" and s_["target"] in a.TARGETS]
    exp = expected_x25(units, a.TARGETS if not a.smoke else ["Canada"])
    have = {(s_["target"], int(s_["split"])) for s_ in mine_sh}
    miss = sorted(exp - have)
    st.update(n_x25_units=len(have), n_x25_units_expected=len(exp), missing_x25_units=";".join(f"{t}:{sp}" for t, sp in miss))
    if not ref_sh:
        return dict(st, status=f"관문 미실시(WF9 조각 없음: {a.gate_wf9})")
    if not mine_sh:
        return dict(st, status="관문 실패(x25 조각 없음)")
    ref_sh = [s_ for s_ in ref_sh if (s_["target"], int(s_["split"])) in have]
    ref = XB.load_stores([s_["npz"] for s_ in ref_sh]) if ref_sh else {}
    mine = XB.load_stores([s_["npz"] for s_ in mine_sh])
    mine = {k: v for k, v in mine.items() if not re.search(r"~(l|gl)\|", k[0])}
    g = XB.gate_compare(mine, ref, level=a.GATE_LEVEL)
    s_ = XB.gate_summary(g)
    cov = gate_coverage(mine, ref)
    full = cov["coverage_x25_in_wf9"] == 1.0 and (a.smoke or cov["coverage_wf9_in_x25"] == 1.0)
    passed = bool(s_["passed"] and full and not miss)
    why = [] if passed else ([f"키 실패 {s_['n_fail']}"] if s_["n_fail"] else []) + ([] if s_["n_keys"] else ["공통 키 0"]) + \
        ([] if full else [f"포괄률 x25→WF9 {cov['coverage_x25_in_wf9']:.4f}, WF9→x25 {cov['coverage_wf9_in_x25']:.4f}"]) + \
        ([f"x25 단위 누락 {len(miss)}"] if miss else [])
    XB.check_out_dir(a.OUT)
    a.OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(a.OUT / f"{out_tag or a.TAG}_gate_wf9.csv", index=False)
    st.update(ran=True, passed=passed, n_keys=s_["n_keys"], n_fail=s_["n_fail"], **cov,
              status="통과" if passed else "관문 실패(" + ", ".join(why) + ")")
    print(f"[gate] x25 대 WF9({a.GATE_LEVEL}): 공통 키 {s_['n_keys']} · 실패 키 {s_['n_fail']} · x25 키 {cov['n_keys_x25']}(WF9 에 있음 "
          f"{cov['n_keys_x25_in_wf9']}) · WF9 키 {cov['n_keys_wf9']}(x25 에 있음 {cov['n_keys_wf9_in_x25']}) · x25 단위 {len(have)}/{len(exp)} · "
          f"통과 {passed} → {out_tag or a.TAG}_gate_wf9.csv", flush=True)
    return st


def check_final_shards(a, units) -> list:
    """2단계 변형(needs_final) 조각의 특징 표 조건(계획 2.5: xh·SI 는 최종판 해시 커밋 뒤 R3). 조각의 allow_unfinal 이 거짓이고, feat_final_ok 가
    참이고, feat_sha256 이 집계 쪽 커밋된 최종판 해시와 같아야 한다. 반환 [(조각 이름, 사유)]."""
    fin = (a.FEAT.final or {}).get("sha256", "") if a.FEAT.final_ok else ""
    bad = []
    for u in units:
        v = str(u.get("variant", ""))
        if not needs_final(v):
            continue
        why = []
        if u.get("allow_unfinal"):
            why.append("allow_unfinal")
        if not u.get("feat_final_ok"):
            why.append("feat_final_ok 거짓")
        if not fin:
            why.append("집계 쪽 최종판 없음 또는 미커밋")
        elif u.get("feat_sha256") != fin:
            why.append("feat_sha256 이 최종 해시와 다름")
        if why:
            bad.append((f"{u.get('target')}:{u.get('split')}:{v}", ", ".join(why)))
    return bad


def variant_status(a, variants) -> pd.DataFrame:
    """변형 × 대상의 실행 상태(90 % 규칙 제외, 미실행 등록 이탈, 특징 표 열 없음)와 SI 설명. 라벨 값을 쓰지 않는다(SI 상태 표, 메타)."""
    rows = []
    for v in variants:
        for t in TARGETS:
            try:
                cols, info = variant_cols(a, v, t)
                st = "실행" if cols is not None else info.get("skip", "")
            except RuntimeError as e:                                   # 메타에 그 대상의 group_decisions 가 없다
                cols, info, st = None, {}, f"특징 표 메타 없음({e})"
            if needs_final(v) and not a.FEAT.final_ok and not a.FEAT.is_dummy and cols is not None:
                st = "실행 조건 미충족(" + FINAL_REQUIRED_TXT + ")"
            rows.append(dict(variant=v, target=t, status=st, groups=",".join(info.get("groups", []) or []),
                             groups_dropped=",".join(info.get("dropped", []) or []), n_cols=len(cols) if cols is not None else 0,
                             note=SI_NOTES.get(v, ""), registered_as="SI 서술(XE-d)" if v in RG.VARIANTS_SI else "주(계획 2.5)"))
    return pd.DataFrame(rows)


def summarize(a):
    if a.stage == "xm":
        return summarize_xm(a)
    t0 = time.time()
    tms, units, runs = XB.load_tms(a.SHARDS, a.TAG, a.nboot, allow_mixed=a.allow_mixed_cfg)
    if not tms:
        print(f"[summarize] 조각 없음: {a.SHARDS}/{a.TAG}__*", flush=True)
        return None
    bad = check_final_shards(a, units)
    if bad and not a.smoke:
        raise SystemExit(f"[거부] 2단계 변형 조각 {len(bad)}개가 최종판 조건을 만족하지 않는다(예: {bad[0][0]}: {bad[0][1]}). "
                         "계획 2.5 는 xh·SI 를 xe_feat 최종판 해시 커밋 뒤의 R3 에서만 돌린다")
    gate = gate_wf9(a, units)                                             # 봉인 표보다 먼저(계획 1절·0.3 열람 순서 5)
    variants = sorted({u.get("variant") for u in units})
    targets = [t for t in TARGETS if any(nm.startswith(f"{t}|") for nm in tms)]
    rows, pooled = run_contrasts(tms, variants, targets)
    region_rows = [r for r in rows if r.get("scope") == "region"]
    dt = decomp_table(tms)
    nboot_note = "" if int(a.nboot) >= XB.NBOOT else f"재표집 {int(a.nboot):,}회(등록 이탈, 등록 {XB.NBOOT:,}회)"
    vst = variant_status(a, sorted(set(variants) | (set(RG.VARIANTS_SI) if any(v in RG.VARIANTS_SI for v in variants) else set())))
    skips = {f"{r.variant}|{r.target}": r.status for r in vst.itertuples() if r.status != "실행"}

    def mark(df_):
        df_ = df_.copy()
        df_["nboot"] = int(a.nboot)
        df_["nboot_note"] = nboot_note
        df_["gate"] = gate.get("status", "")
        return df_
    tables = {}
    df = XB.clean_rows(rows)
    for fam in ("xh0", "xt2", "xh", "si"):
        q = df[df.family == fam] if len(df) else df
        if len(q):
            q = mark(q)
            q["excluded_targets"] = q.variant.map(lambda v: ";".join(f"{k.split('|')[1]}({s_})" for k, s_ in skips.items() if k.split("|")[0] == v))
            tables[f"{a.TAG}_{fam}_tests.csv"] = q
    if "xh" in variants:
        vr, ht = verdict_rows(pooled, region_rows, dt, a.FEAT.meta, targets, gate)
        vr = mark(vr)
        vr["excluded_targets"] = ";".join(f"{k.split('|')[1]}({s_})" for k, s_ in skips.items() if k.split("|")[0] == "xh")
        tables[f"{a.TAG}_xh_hypotheses.csv"] = vr
        tables[f"{a.TAG}_xh_holm.csv"] = mark(ht)
    if len(dt):                                                          # 변형 묶음별 분해 표(x25·P1 기준 행을 함께 둔다)
        for fam, vs in (("xh0", ["xh0"]), ("xt2", ["xt2"]), ("xh", ["xh"]), ("si", list(RG.VARIANTS_SI))):
            q = dt[dt.variant.isin(vs + ["x25"])]
            if len(q) and q.variant.isin(vs).any():
                q = mark(q)
                q["note"] = q.variant.map(lambda v: SI_NOTES.get(v, ""))
                tables[f"{a.TAG}_{fam}_decomp.csv"] = q
    if any(v in RG.VARIANTS_SI for v in variants):
        tables[f"{a.TAG}_si_status.csv"] = vst[vst.registered_as.str.startswith("SI")]
    fg = feat_gate(units, a.FEAT)
    meta = dict(stage="XE", plan=f"{XB.PLAN_DOC} {RG.PLAN_SECTION}", plan_commit=XB.PLAN_COMMIT, tag=a.TAG, variants=variants, targets=targets,
                n_units=len(units), n_stores=len(tms), grid=a.GRID, splits=a.SPLIT_LIST, nboot=int(a.nboot), nboot_registered=XB.NBOOT,
                nboot_note=nboot_note, holm_m=HOLM_M, main_n=list(MAIN_N), smoke=bool(a.smoke),
                unit_status=dict(Counter(str(u.get("status")) for u in units)), n_fit_total=int(sum(int(u.get("n_fit_total", 0)) for u in units)),
                unit_elapsed_s=float(sum(float(u.get("elapsed_s", 0)) for u in units)), feat_sha256=a.FEAT.sha, feat_final_ok=a.FEAT.final_ok,
                feat_final_commit=a.FEAT.final_commit if not a.FEAT.is_dummy else {}, unfinal_shards=[f"{k}({w})" for k, w in bad],
                feat_gate_rows=int(len(fg)), feat_gate_all_same=bool(len(fg) and fg.same.all()), gate_wf9=gate, variant_target_skips=skips,
                summarize_s=round(time.time() - t0, 1),
                reading_rule="xh0 표는 R2a 제출 뒤, xt2 표는 xe_feat 최종 해시 커밋 뒤(--feat-gate-only 통과, 둘 가운데 늦은 시점), xh·SI 표는 R3 회수 뒤 "
                             "재현 관문 통과 뒤에 연다(계획 0.3). 관문 결과는 gate_wf9 항목에 있다")
    tables[f"{a.TAG}_meta.json"] = meta
    XB.write_sealed(a.SEALED_NAME, tables, root=a.SEALED_ROOT)
    if len(fg):
        XB.check_out_dir(a.OUT)
        fg.to_csv(a.OUT / f"{a.TAG}_feat_gate.csv", index=False)
    print(f"[summarize] 조각 {len(units)} · 저장소 {len(tms)} · 변형 {variants} · 대비 행 {len(df)} · 관문 {gate.get('status')} · "
          f"재표집 {int(a.nboot)} · {time.time() - t0:.0f}s", flush=True)
    return dict(rows=len(df), tables=list(tables), gate=gate)


# ================================================================ XM 집계(추가 등록 docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md 4·5절)
def xm_x25_shards(a):
    """XM 이 다시 쓰는 x25 조각(XE r1b, 대상 목록 안)."""
    return [s_ for s_ in XB.find_shards(a.X25_FROM, a.X25_TAG) if s_["variant"] == "x25" and s_["target"] in a.TARGETS]


def load_tms_xm(a):
    """XM 조각(xw·xw_lc)과 XE r1b 의 x25 조각을 저장소 이름으로 합친 TMx(h4_common.load_stores 의 키 합집합, h54.make_tm). 공통 설정 해시가
    하나여야 한다(xbatch_core.check_cfg, 다르면 멈춘다). 반환 (tms, units, x25 조각 목록, XM 조각 목록)."""
    sh_xm = [s_ for s_ in XB.find_shards(a.SHARDS, a.TAG) if s_["variant"] in RG.XM_VARIANTS and s_["target"] in a.TARGETS]
    sh_25 = xm_x25_shards(a)
    if not sh_xm:
        return {}, [], sh_25, sh_xm
    units = [json.loads(s_["unit"].read_text()) for s_ in sh_25 + sh_xm]
    XB.check_cfg(units, a.allow_mixed_cfg)
    stores = XB.load_stores([s_["npz"] for s_ in sh_25 + sh_xm])
    by = {}
    for (nm, sp), st in stores.items():
        by.setdefault(nm, {})[int(sp)] = st
    tms = {nm: W.make_tm(nm, bs, W.units_for(units, nm), int(a.nboot), XB.is_point_only_name(nm)) for nm, bs in sorted(by.items())}
    return tms, units, sh_25, sh_xm


def xm_primary(variants, feat):
    """주 대비 변형: xw(조각이 있으면), S2 가 세 대상 모두에서 빠졌거나 xw 조각이 없으면 xw_lc(등록 3절 (d))."""
    if "xw" in variants and not feat.s2_all_dropped():
        return "xw"
    return "xw_lc" if "xw_lc" in variants else None


def xm_contrast_specs(primary, variants):
    """(가설, 변형, 대비 이름, 저장소 접미사, gA, gB, n 목록, 역할). XM-a 격자 안(주), XM-b 총, XM-c 격자 사이, XM-d 그 밖(등록 4절)."""
    S = []
    if primary is None:
        return S
    v = primary
    nm = f"R1(λ cv, {v})−R1(λ cv, x25)"
    fA, fB = (lambda n: gR(v, n)), (lambda n: gR("x25", n))
    S += [("XM-a", v, nm, "~w", fA, fB, AUX_N, "주"), ("XM-b", v, nm, "", fA, fB, AUX_N, "보조"), ("XM-c", v, nm, "~b", fA, fB, AUX_N, "보조"),
          ("XM-d", v, nm, "~l", fA, fB, AUX_N, "보조"), ("XM-d", v, nm, "~gl", fA, fB, AUX_N, "보조"),
          ("XM-d", v, f"R1(λ cv, {v})−P1", "~w", fA, gP, AUX_N, "보조"),
          ("XM-d", v, f"D0({v})−D0(x25)", "~w", (lambda n: gD(v, n)), (lambda n: gD("x25", n)), AUX_N, "보조")]
    for o in variants:
        if o == v or o not in RG.XM_VARIANTS:
            continue
        nmo = f"R1(λ cv, {o})−R1(λ cv, x25)"
        fo = (lambda oo: lambda n: gR(oo, n))(o)
        S += [("XM-d", o, nmo, part, fo, fB, AUX_N, "보조") for part in ("~w", "", "~b")]
    return S


def xm_row_label(hyp, n):
    """XM-a 의 등록 n 은 500·1,000·전량이고 n 200 행은 XM-d(보조)로 적는다(등록 4절)."""
    if hyp == "XM-a" and int(n) not in MAIN_N:
        return "XM-d", "보조"
    return hyp, ("주" if hyp == "XM-a" else "보조")


def xm_run_contrasts(tms, primary, variants, targets):
    rows, pooled = [], {}
    for hyp0, var, name, part, fA, fB, ns, _role in xm_contrast_specs(primary, variants):
        names = [f"{t}{part}|{MODE}" for t in targets]
        for n in ns:
            hyp, role = xm_row_label(hyp0, n)
            rr, _ = XB.contrast_pool(tms, names, fA(n), fB(n), label=f"{name}[{PART_WORD[part]}]|n{W.nlab(n)}", kind="same", registered=len(TARGETS))
            for r in rr:
                r.update(hypothesis=hyp, hypothesis_registered=hyp0, family="xm", variant=var, contrast=name, part=PART_WORD[part], n=int(n), role=role,
                         blind=XM_BLIND[0], blind_reason=XM_BLIND[1], design=XM_DESIGN, main_n=bool(n in MAIN_N), primary_variant=primary)
                if r.get("scope") == "MEAN":
                    pooled[(hyp, name, PART_WORD[part], int(n))] = r
            rows += rr
    return rows, pooled


def xm_verdict_rows(pooled, region_rows, decomp, primary, gate=None, s2_dropped=(), fallback=False):
    """XM-a 의 _rule3(우세 기준, n 500·1,000·전량), Holm(m = 3, 보조 열), 사전 고정 해석 문장(등록 5절). 문장 수치는 기준 충족 등록 n 가운데 가장 큰 n
    (전량 > 1,000 > 500)의 풀 점 추정(셀 가중), 설명 비율은 같은 n 의 대상 평균이다. 관문을 통과하지 않았으면 판정 불가로 쓴다."""
    name = f"R1(λ cv, {primary})−R1(λ cv, x25)" if primary else ""
    items, lab, pv = [], [], []
    for n in MAIN_N:
        r = pooled.get(("XM-a", name, "격자 안", n))
        items.append((f"n={W.nlab(n)}", None if r is None else r.get("verdict4")))
        lab.append(f"XM-a|n{W.nlab(n)}")
        pv.append(None if (r is None or r.get("undetermined")) else r.get("p_two"))
    rule = XB.rule3(items, "우세") if primary else "판정 불가(주 대비 변형 없음)"
    ht = XB.holm_table(lab, pv, XM_HOLM_M)
    hp = dict(zip(ht.label, ht.p_holm))
    gate_txt = "" if gate is None or gate.get("passed") else str(gate.get("status", "관문 미실시"))
    if gate_txt:
        rule = f"판정 불가(관문: {gate_txt})"
    br = branch(rule, [v for _, v in items])
    worse = [(f"{RG_NAME.get(str(r['target']).split('~')[0].split('|')[0], r['target'])}(n {W.nlab(r['n'])})", r["delta"], r["ci_lo"], r["ci_hi"])
             for r in region_rows if r.get("hypothesis") == "XM-a" and r.get("part") == "격자 안" and int(r.get("n", 0)) in MAIN_N
             and r.get("contrast") == name and r.get("worse")]
    src = dict(n=None, delta=np.nan, holm=np.nan, x=np.nan)
    inp = XM_INPUT_WORD.get(primary, "피복·식생 입력을")

    def take(want):
        n_, r_ = None, None
        for n in N_PREF:
            q = pooled.get(("XM-a", name, "격자 안", n))
            if q is not None and q.get("verdict4") == want:
                n_, r_ = n, q
                break
        if r_ is None:
            return np.nan, None, np.nan
        d_ = float(r_["delta"])
        src.update(n=n_, delta=d_, holm=float(hp.get(f"XM-a|n{W.nlab(n_)}", np.nan)))
        return abs(d_), n_, d_
    if br == "우세":
        a_, n_, d_ = take("우세")
        x = expl_ratio(decomp, primary, n_, "~w") if n_ is not None else float("nan")
        src["x"] = x
        base = XM_SENT["superior"].format(inp=inp, a=f"{a_:.2f}" if np.isfinite(a_) else "a", x=f"{x:.1f}" if np.isfinite(x) else "x")
        sent = XB.compose_sentence({"우세": base}, "우세", src["holm"], d_, worse) + " " + XM_SENT["superior_tail"]
    elif br == "열세":
        a_, n_, d_ = take("열세")
        sent = XB.compose_sentence({"열세": XM_SENT["worse"].format(a=f"{a_:.2f}" if np.isfinite(a_) else "a")}, "열세", src["holm"], d_, worse)
    elif br == "동등":
        sent = XB.compose_sentence({"동등": XM_SENT["eq"]}, "동등", worse_regions=worse) + " " + XM_SENT["eq_tail"]
    elif br == "미결정":
        share = ", ".join(f"{RG_NAME.get(t, t)} {XM_CELL_SHARE[t]} %" for t in TARGETS)
        sent = XB.compose_sentence({"미결정": XM_SENT["und"]}, "미결정", worse_regions=worse)
        sent += f" 격자 안 분산 가운데 1 km 셀 사이 몫은 {share} 다(within_grid_inputs.md 3.3)."
    else:
        sent = XB.compose_sentence({"판정 불가": XM_SENT["na"]}, "판정 불가", reason=rule)
    tail = []
    if s2_dropped:
        tail.append(f"S2 군이 90 % 규칙으로 빠진 대상: {', '.join(RG_NAME.get(t, t) for t in s2_dropped)}(그 대상의 xw 는 xw_lc 와 열이 같다).")
    tail.append(f"주 대비 변형은 {primary}" + ("(S2 군이 세 대상 모두에서 빠져 xw_lc 로 대체했다, 등록 3절 (d))." if fallback else "."))
    tail.append("XM 은 남은 변동의 원인을 구별하지 않는다.")
    out = [dict(hypothesis="XM-a", scope="verdict", verdict=rule, items="; ".join(f"{k} {v}" for k, v in items), branch=br,
                holm_p=";".join(f"{k}={hp[k]:.4g}" for k in hp), holm_m=XM_HOLM_M, blind=XM_BLIND[0], blind_reason=XM_BLIND[1], design=XM_DESIGN,
                gate=gate_txt or "통과", primary_variant=primary or ""),
           dict(hypothesis="XM-a", scope="sentence", verdict="", sentence=sent + " " + " ".join(tail), branch=br, primary_variant=primary or "",
                sentence_n=W.nlab(src["n"]) if src["n"] is not None else "", sentence_delta=src["delta"], sentence_holm_p=src["holm"],
                expl_within=src["x"], sentence_rule="a·Δ·Holm p 는 기준 충족 등록 n 가운데 가장 큰 n(전량 > 1,000 > 500)의 풀 점 추정(셀 가중), "
                                                    "x 는 같은 n 의 대상 평균 격자 안 설명 비율", gate=gate_txt or "통과", design=XM_DESIGN)]
    return pd.DataFrame(out), ht


def summarize_xm(a):
    """XM 집계(등록 4·5·10절). 봉인 폴더(data/processed/xbatch/XM_landcover_vegetation/sealed)에만 쓰고 화면에는 행 수와 해시만 쓴다.
    재현 관문(다시 쓰는 x25 조각 대 WF9, local_rescale)을 봉인 표보다 먼저 돌린다. x25 조각 파일의 sha256 을 봉인 밖 표와 메타에 적는다."""
    t0 = time.time()
    tms, units, sh_25, sh_xm = load_tms_xm(a)
    if not tms:
        print(f"[summarize-xm] XM 조각 없음: {a.SHARDS}/{a.TAG}__*", flush=True)
        return None
    u25 = [json.loads(s_["unit"].read_text()) for s_ in sh_25]
    gate = gate_wf9(a, u25, mine_dir=a.X25_FROM, mine_tag=a.X25_TAG, out_tag=a.TAG)
    reuse = pd.DataFrame([dict(target=s_["target"], split=s_["split"], file=Path(s_[k]).name, sha256=RG.sha256_file(s_[k]))
                          for s_ in sh_25 for k in ("runs", "npz", "unit")])
    XB.check_out_dir(a.OUT)
    a.OUT.mkdir(parents=True, exist_ok=True)
    reuse.to_csv(a.OUT / f"{a.TAG}_x25_reuse.csv", index=False)
    reuse_sha = RG.sha256_file(a.OUT / f"{a.TAG}_x25_reuse.csv")
    variants = sorted({u.get("variant") for u in units if u.get("variant") in RG.XM_VARIANTS})
    targets = [t for t in TARGETS if any(nm.startswith(f"{t}|") for nm in tms)]
    primary = xm_primary(variants, a.FEAT_XM)
    fallback = primary == "xw_lc"
    rows, pooled = xm_run_contrasts(tms, primary, variants, targets)
    region_rows = [r for r in rows if r.get("scope") == "region"]
    dt = decomp_table(tms)
    s2_dropped = a.FEAT_XM.dropped_targets("S2")
    nboot_note = "" if int(a.nboot) >= XB.NBOOT else f"재표집 {int(a.nboot):,}회(등록 이탈, 등록 {XB.NBOOT:,}회)"

    def mark(df_):
        df_ = df_.copy()
        df_["nboot"] = int(a.nboot)
        df_["nboot_note"] = nboot_note
        df_["gate"] = gate.get("status", "")
        return df_
    tables = {}
    df = XB.clean_rows(rows)
    if len(df):
        tables[f"{a.TAG}_tests.csv"] = mark(df)
    vr, ht = xm_verdict_rows(pooled, region_rows, dt, primary, gate, s2_dropped, fallback)
    tables[f"{a.TAG}_hypotheses.csv"] = mark(vr)
    tables[f"{a.TAG}_holm.csv"] = mark(ht)
    if len(dt):
        q = dt[dt.variant.isin(list(RG.XM_VARIANTS) + ["x25"])]
        tables[f"{a.TAG}_decomp.csv"] = mark(q)
    vst = []
    for v in RG.XM_VARIANTS:
        for t in TARGETS:
            cols, info = variant_cols(a, v, t)
            n_u = sum(1 for u in units if u.get("variant") == v and u.get("target") == t)
            vst.append(dict(variant=v, target=t, n_units=n_u, n_cols=len(cols) if cols else 0, groups=",".join(info.get("groups", [])),
                            dropped=",".join(info.get("dropped", [])), same_as=info.get("same_as", ""), skip=info.get("skip", "")))
    meta = dict(stage="XM", plan=RG.XM_PLAN_DOC, plan_commit=RG.XM_PLAN_COMMIT, base_plan=f"{XB.PLAN_DOC} {RG.PLAN_SECTION}", tag=a.TAG,
                variants=variants, primary_variant=primary, fallback_xw_lc=fallback, targets=targets, n_units_xm=len(sh_xm), n_units_x25=len(sh_25),
                n_stores=len(tms), grid=a.GRID, splits=a.SPLIT_LIST, nboot=int(a.nboot), nboot_registered=XB.NBOOT, nboot_note=nboot_note,
                holm_m=XM_HOLM_M, main_n=list(MAIN_N), design=XM_DESIGN, blind=list(XM_BLIND),
                unit_status=dict(Counter(str(u.get("status")) for u in units if u.get("variant") in RG.XM_VARIANTS)),
                n_fit_xm=int(sum(int(u.get("n_fit_total", 0)) for u in units if u.get("variant") in RG.XM_VARIANTS)),
                unit_elapsed_s_xm=float(sum(float(u.get("elapsed_s", 0)) for u in units if u.get("variant") in RG.XM_VARIANTS)),
                feat_file=str(a.FEAT_XM.path), feat_sha256=a.FEAT_XM.sha, feat_group_decisions=(a.FEAT_XM.meta or {}).get("group_decisions"),
                feat_coverage=(a.FEAT_XM.meta or {}).get("coverage"), s2_dropped_targets=s2_dropped, variant_status=vst,
                x25_from=str(a.X25_FROM), x25_tag=a.X25_TAG, x25_reuse_table=str(a.OUT / f"{a.TAG}_x25_reuse.csv"), x25_reuse_sha256=reuse_sha,
                cfg_common=sorted({str(u.get("cfg_common")) for u in units}), gate_wf9=gate, summarize_s=round(time.time() - t0, 1),
                reading_rule="봉인 표는 조정 담당이 연다. 수행자는 열지 않는다(등록 10절)")
    tables[f"{a.TAG}_meta.json"] = meta
    XB.write_sealed(a.SEALED_NAME, tables, root=a.SEALED_ROOT)
    print(f"[summarize-xm] 조각 XM {len(sh_xm)} · x25 {len(sh_25)} · 저장소 {len(tms)} · 변형 {variants} · 주 변형 {primary} · 대비 행 {len(df)} · "
          f"관문 {gate.get('status')} · 재표집 {int(a.nboot)} · {time.time() - t0:.0f}s", flush=True)
    return dict(rows=len(df), tables=list(tables), gate=gate, primary=primary)


# ================================================================ XE-e(현장 VWC 상한 진단, 계획 2.5 '진단(탐색)')
# 등록 문구: ABoVE 현장 VWC 부분 집합(알래스카 2,468셀·31블록, 캐나다 602셀·15블록, 15 m 안)에서 P1 격자 안 잔차를 비 GPR VWC 로 설명하는 비율
# (블록 교차검증, 1차 회귀와 catboost_lo). GPR VWC 는 민감도. 등록하지 않은 세부(층, 날짜 맞춤, 접기, 결측)는 구현 기록 6절의 문자 그대로 읽기다.
XE_E_TARGETS = ("Alaska", "Canada")
XE_E_RADIUS_M = 15.0
XE_E_SHALLOW_MAX_CM = 12.0                                                  # within_grid_inputs.md 5.8: 측정 하단 ≤ 12 cm 와 > 12 cm 를 따로
XE_E_CRS = "EPSG:3338"                                                      # within_grid_inputs.md 부록 A(최근접 거리)
XE_E_SETS = {"main": ("vwc_shallow", "vwc_deep"), "gpr": ("vwc_gpr",)}
XE_E_SET_WORD = {"main": "비 GPR VWC(주)", "gpr": "GPR VWC(민감도)"}
XE_E_SUBSETS = ("registered", "finite")
XE_E_SUBSET_WORD = {"registered": "등록 부분 집합(15 m 안에 VWC 위치가 있는 셀 전체. 층 없는 위치만 있는 셀은 예측 변수 결측)",
                    "finite": "민감도(그 묶음의 예측 변수가 유한한 셀만)"}
XE_E_REGISTERED = {"Alaska": (2468, 31), "Canada": (602, 15)}             # 계획 2.5 XE-e 의 (셀, 블록). 준비·실행 메타에서 대조만 한다
XE_E_LEARNERS = ("ols", LO)                                                 # 1차 회귀(단조 제약 없음), catboost_lo
XE_E_EQUIV = {"Alaska": 7.9, "Lena": 5.8, "Canada": 5.0}                    # 0.5 cm 등가 설명 비율(%), 계획 2.5 검정력 근사·within_grid_inputs.md 5.8
ABOVE_COLS = ("latitude", "longitude", "date", "ALT_instrument", "VWC_instrument", "depth_bottom", "VWC")   # ALT·ALT_err 값은 읽지 않는다
COORD_READ = ("loc_id", "lat", "lon", "region", "block")                    # v3 에서 읽는 열(라벨 열 없음)
VWC_CLASSES = ("shallow", "deep", "gpr")


def vwc_rows(raw) -> pd.DataFrame:
    """ABoVE 행 → VWC 측정 행(VWC 0–100 %, 유효 위경도)과 층 분류. 위치 열쇠 = 소수 4자리 위·경도(within_grid_inputs.md 3.4).
    층: VWC_instrument 가 GPR 이면 gpr(민감도), 그 밖(기기 미표기 포함)은 측정 하단 0 < d ≤ 12 cm 이면 shallow, d > 12 cm 이면 deep, 하단이
    결측(−9999 등)이면 층 없음(뺀다)."""
    d = raw.copy()
    lat = pd.to_numeric(d.latitude, errors="coerce"); lon = pd.to_numeric(d.longitude, errors="coerce")
    v = pd.to_numeric(d.VWC, errors="coerce")
    ok = lat.between(-90, 90) & lon.between(-180, 180) & v.between(0, 100)
    d = d[ok.values].copy()
    d["lat"], d["lon"], d["vwc"] = lat[ok].values, lon[ok].values, v[ok].values
    d["key"] = [f"{y:.4f}_{x:.4f}" for y, x in zip(np.round(d.lat.values, 4), np.round(d.lon.values, 4))]
    d["year"] = pd.to_datetime(d.date, errors="coerce").dt.year
    db = pd.to_numeric(d.depth_bottom, errors="coerce").values
    ins = d.VWC_instrument.astype("string").fillna("미표기").astype(str).values
    cls = np.where(ins == "GPR", "gpr", np.where((db > 0) & (db <= XE_E_SHALLOW_MAX_CM), "shallow", np.where(db > XE_E_SHALLOW_MAX_CM, "deep", "")))
    d["instrument"], d["cls"] = ins, cls
    return d[["key", "lat", "lon", "year", "instrument", "cls", "vwc"]]


def alt_years(raw) -> dict:
    """위치 열쇠 → ALT 측정 연도 집합(ALT_instrument 가 있는 행의 날짜. ALT 값은 읽지 않는다)."""
    d = raw[raw.ALT_instrument.notna()]
    lat = pd.to_numeric(d.latitude, errors="coerce"); lon = pd.to_numeric(d.longitude, errors="coerce")
    yr = pd.to_datetime(d.date, errors="coerce").dt.year
    ok = (lat.between(-90, 90) & lon.between(-180, 180) & yr.notna()).values
    keys = [f"{y:.4f}_{x:.4f}" for y, x in zip(np.round(lat.values[ok], 4), np.round(lon.values[ok], 4))]
    out: dict = {}
    for k, y in zip(keys, yr.values[ok].astype(int)):
        out.setdefault(k, set()).add(int(y))
    return out


def vwc_location_values(rows, years) -> pd.DataFrame:
    """위치 × 층의 VWC. 같은 위치에서 ALT 를 잰 해(같은 계절 캠페인)의 VWC 행이 있으면 그 행만, 없으면 그 위치·층의 모든 행을 쓴다.
    값 = 위치·층 안에서 기기별 평균의 평균(위치·기기별 평균). 반환 (key, lat, lon, cls, vwc, campaign(같은 해 행 사용 여부), n_rows, n_instr)."""
    r = rows[rows.cls != ""].copy()
    if not len(r):
        return pd.DataFrame(columns=["key", "lat", "lon", "cls", "vwc", "campaign", "n_rows", "n_instr"])
    r["camp"] = [bool(y == y and int(y) in years.get(k, ())) for k, y in zip(r.key.values, r.year.values)]
    has = r.groupby(["key", "cls"]).camp.transform("any").values
    r = r[(~has) | r.camp.values]
    per_ins = r.groupby(["key", "cls", "instrument"]).agg(vwc=("vwc", "mean"), n=("vwc", "size"), camp=("camp", "any")).reset_index()
    out = per_ins.groupby(["key", "cls"]).agg(vwc=("vwc", "mean"), campaign=("camp", "any"), n_rows=("n", "sum"), n_instr=("instrument", "size")).reset_index()
    ll = r.groupby("key").agg(lat=("lat", "mean"), lon=("lon", "mean"))
    return out.join(ll, on="key")[["key", "lat", "lon", "cls", "vwc", "campaign", "n_rows", "n_instr"]]


def project_xy(lat, lon):
    import pyproj
    tr = pyproj.Transformer.from_crs("EPSG:4326", XE_E_CRS, always_xy=True)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.column_stack([x, y])


def vwc_locations_any(rows) -> pd.DataFrame:
    """VWC 행이 있는 모든 위치(층 없는 행 포함, key, lat, lon). 등록 부분 집합('15 m 안에 VWC 위치가 있는 셀')의 소속을 정하는 데 쓴다."""
    if not len(rows):
        return pd.DataFrame(columns=["key", "lat", "lon"])
    return rows.groupby("key").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()


def cell_vwc(cells, locv, radius_m=XE_E_RADIUS_M, xy_fn=project_xy, loc_any=None) -> pd.DataFrame:
    """라벨 셀(loc_id, lat, lon, target, block) × 반경 안 VWC 위치. 층마다 반경 안 위치 값의 평균(유한값만)과 위치 수, 같은 해 행을 쓴 위치 비율.
    loc_any(모든 VWC 위치, vwc_locations_any)를 주면 n_loc_any = 층과 관계없이 반경 안에 있는 VWC 위치 수(등록 부분 집합의 소속 기준)."""
    from scipy.spatial import cKDTree
    out = cells[["loc_id", "target", "block"]].copy().reset_index(drop=True)
    cxy = xy_fn(cells.lat.values, cells.lon.values)
    for cls in VWC_CLASSES:
        q = locv[locv.cls == cls]
        vals = np.full(len(out), np.nan); nloc = np.zeros(len(out), int); camp = np.full(len(out), np.nan)
        if len(q):
            tree = cKDTree(xy_fn(q.lat.values, q.lon.values))
            hits = tree.query_ball_point(cxy, r=float(radius_m))
            vv, cc = q.vwc.values.astype(float), q.campaign.values.astype(float)
            for i, h in enumerate(hits):
                if h:
                    vals[i] = float(np.mean(vv[h])); nloc[i] = len(h); camp[i] = float(np.mean(cc[h]))
        out[f"vwc_{cls}"], out[f"n_loc_{cls}"], out[f"camp_{cls}"] = vals, nloc, camp
    nany = out[[f"n_loc_{c}" for c in VWC_CLASSES]].sum(axis=1).to_numpy(int)
    if loc_any is not None and len(loc_any):
        tree = cKDTree(xy_fn(loc_any.lat.values, loc_any.lon.values))
        nany = np.array([len(h) for h in tree.query_ball_point(cxy, r=float(radius_m))], int)
    out["n_loc_any"] = nany
    return out


def read_coords(path) -> pd.DataFrame:
    """v3 의 좌표·블록 열만 읽는다(라벨 열을 읽지 않는다)."""
    cols = list(COORD_READ)
    if set(cols) & set(RG.LABEL_COLS):
        raise RuntimeError("XE-e 준비가 라벨 열을 읽으려 했다")
    d = pd.read_csv(path, usecols=cols, low_memory=False)
    d["target"] = d.region.map(RG.MACRO)
    return d


def vwc_prep(a):
    """XE-e 의 셀별 VWC 표(세기 범주: 좌표·블록과 ABoVE 의 좌표·날짜·기기·깊이·VWC 만 읽는다. ALT 값과 v3 라벨은 읽지 않는다).
    산출 <OUT>/inputs/xe_vwc_v1.csv 와 메타(sha256, 규칙, 대상·층별 셀·블록 수). 화면에는 수만 쓴다."""
    t0 = time.time()
    src = Path(a.above)
    raw = pd.read_csv(src, usecols=list(ABOVE_COLS), low_memory=False)
    rows = vwc_rows(raw)
    yrs = alt_years(raw)
    locv = vwc_location_values(rows, yrs)
    loc_any = vwc_locations_any(rows)
    cells = read_coords(ROOT / a.data_dir / "fidelity_base_v3.csv")
    cells = cells[cells.target.isin(list(XE_E_TARGETS))].sort_values("loc_id").reset_index(drop=True)
    tab = cell_vwc(cells, locv, loc_any=loc_any)
    tab = tab[(tab.n_loc_any > 0).values].reset_index(drop=True)            # 등록 부분 집합 = 15 m 안에 VWC 위치가 있는 셀(층 없는 위치 포함)
    out = Path(a.vwc) if os.path.isabs(a.vwc) else ROOT / a.vwc
    XB.check_out_dir(out.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False, float_format="%.6g")
    counts = {}
    for t in XE_E_TARGETS:
        q = tab[tab.target == t]
        m_any = (q.n_loc_any > 0).values
        counts[f"{t}|any"] = dict(cells=int(m_any.sum()), blocks=int(len(np.unique(q.block.values[m_any].astype(str)))),
                                  registered=XE_E_REGISTERED.get(t))
        for s_, cols in XE_E_SETS.items():
            m = q[list(cols)].notna().any(axis=1).values
            counts[f"{t}|{s_}"] = dict(cells=int(m.sum()), blocks=int(len(np.unique(q.block.values[m].astype(str)))))
    meta = dict(file=str(out.relative_to(ROOT)) if str(out).startswith(str(ROOT)) else str(out), sha256=RG.sha256_file(out), source=str(src),
                source_sha256=RG.sha256_file(src), source_cols=list(ABOVE_COLS), coord_cols=list(COORD_READ), radius_m=XE_E_RADIUS_M, crs=XE_E_CRS,
                shallow_max_cm=XE_E_SHALLOW_MAX_CM, n_vwc_rows=int(len(rows)), n_vwc_rows_no_layer=int((rows.cls == "").sum()),
                n_locations=int(loc_any.key.nunique()) if len(loc_any) else 0, n_locations_layered=int(locv.key.nunique()) if len(locv) else 0,
                counts=counts, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                rules=dict(subset="등록 부분 집합 = 15 m 안에 VWC 위치(층 없는 행 포함)가 있는 라벨 셀(n_loc_any > 0). 설명 비율은 그 가운데 2셀 이상 격자 묶음 "
                                  "셀에서 낸다(한 셀 묶음의 격자 안 잔차는 0). 계획 2.5 의 셀·블록 수(알래스카 2,468·31, 캐나다 602·15)는 counts 의 "
                                  "<대상>|any 와 대조한다",
                           layers="GPR 기기 행 = gpr(민감도). 그 밖(기기 미표기 포함): 측정 하단 0 < d ≤ 12 cm = shallow, d > 12 cm = deep, 하단 결측은 "
                                  "층 없음(예측 변수 결측, 소속에는 센다)",
                           campaign="같은 위치(소수 4자리 위경도)에서 ALT 를 잰 해의 VWC 행이 있으면 그 행만, 없으면 모든 행",
                           aggregate="위치·층·기기 평균 → 기기 평균의 평균 → 셀 반경 15 m 안 위치 값의 평균", plan="계획 2.5 XE-e, 구현 기록 6절·검토 반영"))
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[xe-e 준비] VWC 행 {len(rows):,}(층 없음 {meta['n_vwc_rows_no_layer']:,}) · 위치 {meta['n_locations']:,}(층 있음 {meta['n_locations_layered']:,}) · "
          f"셀 {len(tab):,} → {out.name} · {time.time() - t0:.0f}s", flush=True)
    for k, v in counts.items():
        t, s_ = k.split("|")
        word = "VWC 위치 15 m 안(등록 부분 집합)" if s_ == "any" else XE_E_SET_WORD[s_]
        reg = f" · 계획 2.5 등록 {v['registered'][0]:,}·{v['registered'][1]}" if v.get("registered") else ""
        print(f"  {RG_NAME.get(t, t)} {word}: 셀 {v['cells']:,} · 블록 {v['blocks']}{reg}", flush=True)
    return meta


def block_folds(blocks, seed_parts, k_max=CV_K):
    """블록 교차검증 묶음(h42.cv_folds_of 와 같은 배정 규칙: 블록 순열을 돌려 K = min(5, 블록 수) 묶음에 배정). 블록 2개 미만이면 None."""
    b = np.asarray(blocks).astype(str)
    ub = np.unique(b)
    if len(ub) < 2:
        return None, 0
    rng = np.random.RandomState(XB.seed_of(*seed_parts))
    K = min(int(k_max), len(ub))
    fo = dict(zip(ub[rng.permutation(len(ub))], np.arange(len(ub)) % K))
    return np.array([fo[v] for v in b], int), K


def ols_fit_predict(Xtr, ytr, Xte):
    """1차 회귀(절편 포함 최소제곱, 단조 제약 없음). 결측은 학습 묶음 평균으로 채우고 결측 표시 열을 더한다(전체 부분 집합에서 결측이 있는 열만).
    반환 (예측, 계수 [절편, 열…, 표시…])."""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    mu = np.array([np.nanmean(c) if np.isfinite(c).any() else 0.0 for c in Xtr.T])
    miss = [j for j in range(Xtr.shape[1]) if (~np.isfinite(Xtr[:, j])).any() or (~np.isfinite(Xte[:, j])).any()]

    def design(M):
        F = np.where(np.isfinite(M), M, mu[None, :])
        return np.column_stack([np.ones(len(M)), F] + [(~np.isfinite(M[:, j])).astype(float) for j in miss])
    beta, *_ = np.linalg.lstsq(design(Xtr), np.asarray(ytr, float), rcond=None)
    return design(Xte) @ beta, beta


def within_grid_residual(y, s, blk):
    """P1 격자 안 잔차. 격자 묶음(블록, 기온 √TDD)에서 P1 = E1·s 는 묶음 안에서 상수이므로 (y − E1·s) − 묶음 평균 = y − ȳ_g 이고 E1(라벨 수·분할)과
    무관하다. 반환 (잔차, 2셀 이상 묶음 마스크)."""
    y = np.asarray(y, float)
    gid, multi = W.grid_groups(s, blk)
    cnt = np.maximum(np.bincount(gid).astype(float), 1.0)
    return y - (np.bincount(gid, weights=y) / cnt)[gid], multi


def xe_e_fit(e, Xm, blocks, target, set_name, seeds, cb_args, subset="registered"):
    """블록 교차검증 묶음 밖 예측으로 격자 안 잔차 e 를 설명하는 비율 1 − Σ(e − ê)²/Σe²(셀 가중). 반환 행 목록.
    subset = registered(등록 부분 집합: VWC 15 m 안 셀 전체, 예측 변수 결측 포함) 또는 finite(그 층 예측 변수가 유한한 셀, 민감도)."""
    folds, K = block_folds(blocks, ("xe-e", target, set_name, subset, "r", 0))
    base = dict(target=target, feature_set=set_name, subset=subset, subset_word=XE_E_SUBSET_WORD.get(subset, subset), n_cells=int(len(e)),
                n_cells_finite=int(np.isfinite(np.asarray(Xm, float)).any(axis=1).sum()) if len(e) else 0,
                n_blocks=int(len(np.unique(np.asarray(blocks).astype(str)))), n_folds=int(K))
    if folds is None or len(e) < 2 * max(K, 1):
        return [dict(base, learner=l_, seed="", expl_pct=np.nan, status="계산 불가(블록 2개 미만)") for l_ in XE_E_LEARNERS]
    sst = float(np.sum(e ** 2))
    rows = []
    for learner in XE_E_LEARNERS:
        for seed in (["-"] if learner == "ols" else list(seeds)):
            pred = np.full(len(e), np.nan)
            for f in range(K):
                te, tr = folds == f, folds != f
                if learner == "ols":
                    pred[te], _ = ols_fit_predict(Xm[tr], e[tr], Xm[te])
                else:
                    m = H.cb_fit(cb_args, Xm[tr].astype(np.float32), e[tr], int(seed))
                    pred[te] = m.predict(Xm[te].astype(np.float32))
            sse = float(np.sum((e - pred) ** 2))
            rows.append(dict(base, learner=learner, seed=str(seed), expl_pct=100.0 * (1.0 - sse / sst) if sst > 0 else np.nan, status="ok"))
    cb = [r for r in rows if r["learner"] == LO and r["status"] == "ok"]
    if len(cb) > 1:
        rows.append(dict(base, learner=LO, seed="mean", expl_pct=float(np.mean([r["expl_pct"] for r in cb])), status="ok"))
    _, beta = ols_fit_predict(Xm, e, Xm[:1])
    for r in rows:
        r.update({f"ols_coef_{c}": float(beta[1 + j]) for j, c in enumerate(XE_E_SETS[set_name])})
    return rows


def xe_e_run(a):
    """XE-e 진단(계획 2.5): 대상(알래스카, 캐나다)의 채점 가능 셀(eval_mask) 전체에서 격자 묶음을 만들고 P1 격자 안 잔차를 구한 뒤, 등록 부분 집합
    (15 m 안에 VWC 위치가 있는 셀, 계획의 '알래스카 2,468셀·31블록, 캐나다 602셀·15블록') 가운데 2셀 이상 묶음 셀에서 블록 교차검증으로 설명 비율을
    낸다. 주 행 = 비 GPR VWC(층별 두 열. 예측 변수가 없는 셀은 결측), 민감도 = GPR VWC, 그리고 예측 변수가 유한한 셀만 쓴 판. 결과는 봉인 폴더에만
    쓰고 화면에는 셀·블록 수만 쓴다. 판정을 바꾸지 않는다."""
    t0 = time.time()
    vp = Path(a.vwc) if os.path.isabs(a.vwc) else ROOT / a.vwc
    if not vp.exists():
        raise SystemExit(f"[거부] XE-e 의 VWC 표가 없다: {vp}(--xe-e-prep 로 만든다)")
    vmeta_p = vp.with_name(vp.stem + "_meta.json")
    vmeta = json.loads(vmeta_p.read_text()) if vmeta_p.exists() else {}
    vsha = RG.sha256_file(vp)
    if vmeta.get("sha256") and vmeta["sha256"] != vsha:
        raise RuntimeError("XE-e VWC 표의 sha256 이 메타와 다르다")
    vwc = pd.read_csv(vp).set_index("loc_id")
    _, D = W.get_data(a.h)
    df = D.df
    cb_args = SimpleNamespace(cb_iters=int(a.cb_iters), threads=int(a.threads))
    seeds = list(range(int(a.seeds)))
    rows, counts = [], {}
    for t in a.TARGETS:
        t_idx = np.asarray(D.target_idx(t))
        ev = t_idx[XB.eval_mask(df.iloc[t_idx])]
        e_all, multi = within_grid_residual(df.y.values[ev], df.s.values[ev], df.block.values[ev])
        F = vwc.reindex(df.loc_id.values[ev])
        all_cols = [c for cs in XE_E_SETS.values() for c in cs]
        if "n_loc_any" in F.columns:                                       # 등록 부분 집합 = 15 m 안에 VWC 위치가 있는 셀(층 없는 위치 포함)
            member = np.nan_to_num(F["n_loc_any"].to_numpy(dtype=float)) > 0
        else:                                                               # 옛 표(n_loc_any 없음): 층이 있는 VWC 위치가 있는 셀
            member = np.isfinite(F[all_cols].to_numpy(dtype=float)).any(axis=1)
        reg = multi & member
        blk_ev = df.block.values[ev]
        counts[f"{t}|any|member"] = dict(cells=int(member.sum()), blocks=int(len(np.unique(np.asarray(blk_ev[member]).astype(str)))),
                                         cells_multi=int(reg.sum()), blocks_multi=int(len(np.unique(np.asarray(blk_ev[reg]).astype(str)))),
                                         registered=XE_E_REGISTERED.get(t), eval_cells=int(len(ev)), multi_cells=int(multi.sum()))
        for s_, cols in XE_E_SETS.items():
            Xm = F[list(cols)].to_numpy(dtype=float)
            for sub, m in (("registered", reg), ("finite", multi & np.isfinite(Xm).any(axis=1))):
                blocks = blk_ev[m]
                counts[f"{t}|{s_}|{sub}"] = dict(cells=int(m.sum()), blocks=int(len(np.unique(np.asarray(blocks).astype(str)))),
                                                 cells_finite=int((m & np.isfinite(Xm).any(axis=1)).sum()), eval_cells=int(len(ev)), multi_cells=int(multi.sum()))
                for r in xe_e_fit(e_all[m], Xm[m], blocks, t, s_, seeds, cb_args, subset=sub):
                    bl, br = blind_of("XE-e")
                    r.update(hypothesis="XE-e", kind="진단(탐색)", feature_set_word=XE_E_SET_WORD[s_], features=",".join(cols),
                             role="주" if (s_ == "main" and sub == "registered") else "민감도", equiv_05cm_pct=XE_E_EQUIV.get(t, np.nan), blind=bl,
                             blind_reason=br, design="결과 열람 뒤 설계", note="판정을 바꾸지 않는다(계획 2.5 공통 행). 0.5 cm 등가 비율은 비교 참고값",
                             vwc_sha256=vsha[:16])
                    rows.append(r)
    meta = dict(stage="XE-e", plan=f"{XB.PLAN_DOC} {RG.PLAN_SECTION} XE-e", plan_commit=XB.PLAN_COMMIT, tag=a.TAG, targets=a.TARGETS, seeds=seeds,
                cb_iters=int(a.cb_iters), threads=int(a.threads), cv_k=CV_K, learners=list(XE_E_LEARNERS), sets={k: list(v) for k, v in XE_E_SETS.items()},
                subsets=dict(XE_E_SUBSET_WORD), main_row="feature_set main, subset registered(그 밖은 민감도)",
                subset_rule="등록 부분 집합 소속 = 15 m 안에 VWC 위치가 있는 셀(n_loc_any > 0, 층 없는 위치 포함). 설명 비율은 2셀 이상 격자 묶음 셀에서 낸다"
                            "(한 셀 묶음의 격자 안 잔차는 0). counts 의 <대상>|any|member 가 계획 2.5 의 셀·블록 수와 대조하는 값이다",
                registered_counts=dict(XE_E_REGISTERED),
                vwc_file=str(vp), vwc_sha256=vsha, vwc_meta=vmeta, counts=counts, smoke=bool(a.smoke), elapsed_s=round(time.time() - t0, 1),
                max_rss_mb=XB.max_rss_mb(), residual="P1 격자 안 잔차 = y − ȳ_g(격자 묶음 (블록, 기온 √TDD), 채점 가능 셀 전체, E1 과 무관)",
                ratio="1 − Σ(e − ê)²/Σe²(셀 가중, 블록 교차검증 묶음 밖 예측)",
                reading_rule="xh·SI 표와 같은 시점(R3 회수 뒤 재현 관문 통과 뒤)에 연다(구현 기록 6절)")
    XB.write_sealed(a.SEALED_NAME, {f"{a.TAG}_xe_e.csv": pd.DataFrame(rows), f"{a.TAG}_xe_e_meta.json": meta}, root=a.SEALED_ROOT)
    for k, v in counts.items():
        t, s_, sub = k.split("|")
        if s_ == "any":
            regtxt = f" · 계획 2.5 등록 {v['registered'][0]:,}·{v['registered'][1]}" if v.get("registered") else ""
            print(f"[xe-e] {RG_NAME.get(t, t)} VWC 위치 15 m 안 셀 {v['cells']:,} · 블록 {v['blocks']}{regtxt} → 2셀 이상 묶음 셀 {v['cells_multi']:,} · "
                  f"블록 {v['blocks_multi']} (채점 가능 {v['eval_cells']:,}, 2셀 이상 묶음 {v['multi_cells']:,})", flush=True)
            continue
        print(f"[xe-e] {RG_NAME.get(t, t)} {XE_E_SET_WORD[s_]} {'등록 부분 집합' if sub == 'registered' else '유한 셀'}: 셀 {v['cells']:,}"
              f"(예측 변수 유한 {v['cells_finite']:,}) · 블록 {v['blocks']}", flush=True)
    print(f"[xe-e] 행 {len(rows)} · {time.time() - t0:.0f}s · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
    return dict(rows=len(rows), counts=counts)


# ================================================================ 진입점
def main(argv=None):
    a = parse_args(argv)
    mode = run_mode(a)
    XB.guard(mode, a.ARGV)
    with XB.restricted_output():
        check_smoke_env(a, mode)
        res = local_resources(a)
        rep = XB.smoke_env_report()
        ftxt = (f"XM 특징 표 {'세기 전용(빈 표)' if a.FEAT_XM.is_dummy else a.FEAT_XM.path.name}(sha256 {a.FEAT_XM.sha[:16]}) · x25 재사용 "
                f"{a.X25_TAG}" if a.stage == "xm" else
                f"특징 표 {'세기 전용(빈 표)' if a.FEAT.is_dummy else a.FEAT.path.name}(최종판 {'커밋 확인' if a.FEAT.final_ok else '없음'})")
        print(f"[xe] 모드 {mode} · 단계 {a.stage} · 변형 {a.VARIANTS} · 대상 {a.TARGETS} · 분할 {len(a.SPLIT_LIST)} · 격자 {a.GRID} · 스레드 {a.threads} · "
              f"{ftxt}"
              + (f" · 가용 메모리 {res['mem_available_gb']} GB" if res.get("mem_available_gb") is not None else "")
              + (f" · 경고 {rep['warn']}" if rep["warn"] else ""), flush=True)
        if mode == "summarize" and a.NBOOT_ASKED > a.nboot:
            print(f"[local] 허용 표지가 없다: 재표집 {a.NBOOT_ASKED} → {a.nboot}(등록 {XB.NBOOT}회가 아니다. 표에 등록 이탈 열을 단다)", flush=True)
        if a.xe_e_prep:
            vwc_prep(a)
            return 0
        if a.feat_gate_only:
            return feat_gate_check(a)
        if a.xe_e:
            xe_e_run(a)
            return 0
        if a.summarize_only:
            summarize(a)
            return 0
        check_feat_for_fit(a, mode)
        units, skipped, expected = enumerate_units(a)
        if a.count_only:
            count_only(a, units, skipped)
            return 0
        check_requested(a, skipped, mode)
        if skipped:
            print(f"[xe] 건너뜀: {dict(Counter(s_['status'] for s_ in skipped))}", flush=True)
        t0 = time.time()
        done, failed = run_all(a, units, expected)
        print(f"[xe] 완료 {len(done)} · 실패 {len(failed)} · {time.time() - t0:.0f}s · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
        if not a.no_summarize:
            summarize(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
