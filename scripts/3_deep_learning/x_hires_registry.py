"""XE_hires_covariates 입력 집합 등록부(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.5절 '입력 집합' 표, 개정 1, T0 = git 2678100).

이 모듈은 numpy·pandas 만 쓴다. 특징 추출(scripts/1_data_prep/xe_point_covariates.py)과 실험 모듈(x_hires_covariates.py)이 같은 등록부를 쓴다.
열 목록, 군, 변형, 결측 규칙(90 %), 마감 시각은 등록 문서의 값이며 바꾸면 사전 등록에서 벗어난다. 등록 문서가 정하지 않은 세부(군 유한값 비율의
계산 방식, 변형 이름, 점 화소 민감도 열 이름)는 docs/research/2026-10-04/impl_notes/x_hires_covariates.md 에 적었다.

군(등록 열 45개)
  H0(10)  covariates_ext_v1.csv 의 H19 T·W 군과 treecover_1km(점 중심 1 km 이동 창)
  T2(6)   Copernicus DEM 30 m 점 TWI, 90·270 m TPI, 3 × 3 평균 경사, GSW·Hansen 3 × 3 평균
  M(5)    ArcticDEM 10 m 미지형(30–150 m)
  O(9)    SoilGrids 250 m 7층, NCSCD 2층
  V(12)   MOD13Q1 4열, WorldCover 3 × 3 창 등급 비율 7열, CAVM 등급
  S(3)    MOD10A1(대체 MOD10A2) 적설 일수·소멸일·시작일
변형(계획 2.5 '대상·범위')
  주: x25(재현용), xh0 = x25 + H0, xt2 = x25 + H0 + T2, xh = x25 + H
  SI 서술: add_<군>(x25 + 군 하나), sg250(SoilGrids 교체: x25 의 5 km 9열 대신 250 m 7열), xh_px(30 m 이하 제품의 점 화소 민감도),
           cov_AK·cov_LE(지역 피복: 알래스카 ABoVE 30 m, 레나 Lisovski 2025 10 m). 원자료의 취득·추출 경로를 만들지 않았으므로 미실행이다
           (등록 이탈, 구현 기록 6절·SI 상태 표). 열이 표에 있으면 도는 규칙만 남겨 두었다
결측 규칙(계획 2.5 '결합 규칙'): 대상 지역에서 한 군의 유한값 비율이 90 % 미만이면 그 군을 그 대상에서 뺀다. 유한값 비율은 그 군의 (행, 열) 칸 가운데
유한한 칸의 비율이다(문구 '한 군의 유한값 비율'). 나머지 결측은 CatBoost 기본 처리(Min)에 맡긴다.
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

PLAN_DOC = "docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md"
PLAN_SECTION = "2.5 XE_hires_covariates"
PLAN_COMMIT = "2678100"
KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 10, 4, 14, 22, 39, tzinfo=KST)                     # 개정 1 커밋 시각(계획 머리말)

# ================================================================ 군과 열(계획 2.5 표. 순서도 고정한다)
GROUPS = {
    "H0": ["twi", "twi_sd", "flowacc_log", "curv_plan", "curv_prof", "rel_1km", "water_occ_1km", "water_occ_5km", "water_dist_km",
           "treecover_1km"],
    "T2": ["twi_pt", "tpi_90", "tpi_270", "slope_90", "gsw_occ_pt", "treecover_pt"],
    "M": ["ad_slope_30", "ad_curv_30", "ad_tpi_50", "ad_tpi_150", "ad_rough_50"],
    "O": ["sg250_soc_0_5", "sg250_soc_5_15", "sg250_soc_15_30", "sg250_bdod_5_15", "sg250_clay_5_15", "sg250_sand_5_15", "sg250_silt_5_15",
          "ncscd_soc_0_30", "ncscd_soc_0_100"],
    "V": ["ndvi250_jja", "evi250_jja", "ndwi250_jja", "ndvi250_max", "wc_tree", "wc_shrub", "wc_grass", "wc_wetland", "wc_moss", "wc_bare",
          "wc_water", "cavm_class"],
    "S": ["scd500", "snowoff_doy500", "snowon_doy500"],
}
GROUP_SIZES = {"H0": 10, "T2": 6, "M": 5, "O": 9, "V": 12, "S": 3}
assert {k: len(v) for k, v in GROUPS.items()} == GROUP_SIZES and sum(GROUP_SIZES.values()) == 45, "등록 열 수(45)가 계획과 다르다"
STAGE1 = ("H0", "T2")                                                       # 1단계(디스크 원자료)
STAGE2 = ("M", "O", "V", "S")                                               # 2단계(취득 자료)
H_GROUPS = STAGE1 + STAGE2
H_COLS = [c for g in H_GROUPS for c in GROUPS[g]]
SG250_COLS = GROUPS["O"][:7]                                                # SoilGrids 교체 변형의 250 m 7열
WC_CODES = {"wc_tree": 10, "wc_shrub": 20, "wc_grass": 30, "wc_wetland": 90, "wc_moss": 100, "wc_bare": 60, "wc_water": 80}
# WorldCover 2021 v200 범례(data/raw/worldcover_v200/docs/WorldCover_PUM_V2.0.pdf 표 1, 추출 단계 확인): 10 수목, 20 관목, 30 초지, 40 경작지,
# 50 시가지, 60 나지·희소 식생, 70 눈·얼음, 80 영구 수면, 90 초본 습지, 95 맹그로브, 100 이끼·지의류. 0 은 자료 없음.

# 30 m 이하 제품의 3 × 3 창 열 → 점 화소 민감도 열(계획 2.5 '결합 규칙': 3 × 3 창 통계를 주 값, 점 화소 값을 민감도)
PX_MAP = {"slope_90": "slope_px", "gsw_occ_pt": "gsw_occ_px", "treecover_pt": "treecover_px",
          "wc_tree": "wc_tree_px", "wc_shrub": "wc_shrub_px", "wc_grass": "wc_grass_px", "wc_wetland": "wc_wetland_px", "wc_moss": "wc_moss_px",
          "wc_bare": "wc_bare_px", "wc_water": "wc_water_px", "ad_slope_30": "ad_slope_px", "ad_curv_30": "ad_curv_px"}
PX_COLS = list(PX_MAP.values())
PX_GROUP = {"slope_px": "T2", "gsw_occ_px": "T2", "treecover_px": "T2", "ad_slope_px": "M", "ad_curv_px": "M"}
PX_GROUP.update({PX_MAP[k]: "V" for k in WC_CODES})

# 지역 피복 변형(SI 서술). 열 이름은 등록되지 않았다. 표에 접두사 열이 있을 때만 그 대상에서 돈다. 취득·추출 경로(xe_point_covariates.py,
# xe_stage2_acquire.py)가 접두사 열을 만들지 않으므로 이번 묶음에서는 미실행(등록 이탈)이다
COVER = {"cov_AK": ("Alaska", "abv_"), "cov_LE": ("Lena", "lis_")}
COVER_NOT_RUN = "미실행(등록 이탈): 지역 피복 원자료(ABoVE 30 m, Lisovski 2025 10 m)의 취득·추출 경로를 만들지 않았다"

# ================================================================ 대상
TARGETS = ("Alaska", "Lena", "Canada")                                      # 지역 내(모드 r)
MODE = "r"
MACRO = {"ABoVE_AK": "Alaska", "United States (Alaska)": "Alaska", "ABoVE_CA": "Canada", "Canada": "Canada", "CALM_Canada": "Canada",
         "Lena_RU": "Lena"}                                                 # src/polar/fidelity.py MACRO_REGION 의 세 대상 부분
COORD_COLS = ("loc_id", "lat", "lon", "region")                             # 특징 추출이 라벨 표에서 읽는 열(시험 (e))
LABEL_COLS = ("alt_cm", "sigma_prior_cm", "right_censored", "fidelity_level", "alt_sd", "alt_n", "y", "z")   # 추출 코드가 읽으면 안 되는 열

# ================================================================ XM(추가 등록 docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md, 커밋 ff7c97f)
# 1 km 라벨 셀(LG 6B.3) 안의 WorldCover 10 m 등급 비율 11열(WC)과 Sentinel-2 20 m 여름 식생 지수 3열(S2). 특징 표는
# data/processed/xbatch/XM_landcover_vegetation/inputs/xm_feat_v1.csv(scripts/1_data_prep/xm_landcover_s2_features.py). XE 의 군·변형과 섞지 않는다.
XM_PLAN_DOC = "docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md"
XM_PLAN_COMMIT = "ff7c97f"
XM_EXP_NAME = "XM_landcover_vegetation"
XM_GROUPS = {
    "WC": ["wc1k_tree", "wc1k_shrub", "wc1k_grass", "wc1k_crop", "wc1k_built", "wc1k_bare", "wc1k_snow", "wc1k_water", "wc1k_wetland",
           "wc1k_mangrove", "wc1k_moss"],
    "S2": ["s2_ndvi_med", "s2_ndmi_med", "s2_ndvi_sd"],
}
XM_COLS = XM_GROUPS["WC"] + XM_GROUPS["S2"]
XM_VARIANT_GROUPS = {"xw": ("WC", "S2"), "xw_lc": ("WC",)}                 # 등록 2절: xw = x25 + WC + S2(주), xw_lc = x25 + WC(보조·대체)
XM_VARIANTS = tuple(XM_VARIANT_GROUPS)


def xm_variant_columns(variant, target, x25, decisions=None):
    """XM 변형·대상의 입력 열과 설명(등록 2·3절). decisions = {군: 포함 여부}(특징 메타의 90 % 규칙 결과, None 이면 모두 포함).
    반환 (열 목록 또는 None, info). xw 에서 S2 가 빠지면 열이 xw_lc 와 같고 info['same_as'] = 'xw_lc' 다(등록 3절 (c): 그대로 적합하고 표지를 단다)."""
    if variant not in XM_VARIANT_GROUPS:
        raise ValueError(f"알 수 없는 XM 변형 {variant}")
    x25 = list(x25)
    dec = {g: True for g in XM_GROUPS} if decisions is None else dict(decisions)
    gs = XM_VARIANT_GROUPS[variant]
    keep = [g for g in gs if dec.get(g, False)]
    info = dict(variant=variant, target=target, groups=keep, dropped=[g for g in gs if g not in keep], skip="", same_as="")
    if not keep:
        info["skip"] = "모든 군 제외(90 % 규칙)"
        return None, info
    if variant == "xw" and tuple(keep) == XM_VARIANT_GROUPS["xw_lc"]:
        info["same_as"] = "xw_lc"
    return x25 + [c for g in XM_GROUPS if g in keep for c in XM_GROUPS[g]], info


# ================================================================ 결측 규칙과 마감(계획 2.5)
FINITE_MIN = 0.90
DEADLINE_H = {"O": 72, "M": 72, "V_wc": 72, "V_cavm": 72, "V_mod13q1": 96, "S": 96, "final": 100}
DEADLINES = {k: T0 + timedelta(hours=h) for k, h in DEADLINE_H.items()}
GROUP_SOURCES = {"H0": ("H0",), "T2": ("T2",), "M": ("M",), "O": ("O",), "V": ("V_wc", "V_cavm", "V_mod13q1"), "S": ("S",)}

# ================================================================ 변형
VARIANTS_MAIN = ("x25", "xh0", "xt2", "xh")
VARIANTS_SI = ("add_T2", "add_M", "add_O", "add_V", "add_S", "sg250", "xh_px", "cov_AK", "cov_LE")
VARIANTS_ALL = VARIANTS_MAIN + VARIANTS_SI
# 작업 단위 묶음(계획 2.5 '작업': R1b = xh0·xt2, R3 = xh 와 SI 변형). 한 대비는 한 작업 안에서 닫으므로(1절 플랫폼) x25 를 두 작업에서 모두 다시 적합한다
STAGE_VARIANTS = {"r1b": ("x25", "xh0", "xt2"), "r3": ("x25", "xh") + VARIANTS_SI,
                  "xm": XM_VARIANTS}                                        # XM: x25 는 XE r1b 조각을 다시 쓰고 적합하지 않는다(등록 2절)
VARIANT_GROUPS = {"xh0": ("H0",), "xt2": ("H0", "T2"), "xh": H_GROUPS, "xh_px": H_GROUPS,
                  "add_T2": ("T2",), "add_M": ("M",), "add_O": ("O",), "add_V": ("V",), "add_S": ("S",), "sg250": ("O",)}


def method_suffix(variant) -> str:
    """키의 방법 이름 접미사. x25 는 WF9 와 같은 키(접미사 없음), 그 밖은 '@<변형>'(h54 wf1 의 'R1@x34' 와 같은 방식)."""
    return "" if variant == "x25" else f"@{variant}"


def cover_cols(columns, variant):
    """지역 피복 변형의 열(접두사로 찾는다). 없으면 빈 목록."""
    pre = COVER[variant][1]
    return [c for c in columns if str(c).startswith(pre)]


def variant_columns(variant, target, x25, soil_cols, decisions=None, columns=None):
    """변형·대상의 입력 열 목록과 설명. 반환 (열 목록 또는 None, info). None 이면 그 대상에서 이 변형을 돌리지 않는다(사유는 info['skip']).
    x25 = x25 열 목록(h40.FEATS), soil_cols = x25 의 토양 9열, decisions = {군: 포함 여부}(그 대상의 90 % 규칙 결과, None 이면 모두 포함),
    columns = 특징 표의 열(지역 피복 변형의 접두사 열을 찾는다)."""
    x25 = list(x25)
    dec = {g: True for g in H_GROUPS} if decisions is None else dict(decisions)
    info = dict(variant=variant, target=target, groups=[], dropped=[], skip="")
    if variant == "x25":
        return x25, info
    if variant in COVER:
        tg, _ = COVER[variant]
        cc = cover_cols(columns or [], variant)
        if target != tg:
            info["skip"] = f"대상 아님({tg} 전용)"
            return None, info
        if not cc:
            info["skip"] = COVER_NOT_RUN
            return None, info
        if not dec.get(variant, decisions is None):                         # 결정이 주어졌는데 이 변형의 항목이 없으면 빼는 쪽으로 닫는다
            info["skip"] = "군 제외(90 % 규칙 또는 결정 없음)"
            return None, info
        info["groups"] = [variant]
        return x25 + cc, info
    if variant not in VARIANT_GROUPS:
        raise ValueError(f"알 수 없는 변형 {variant}")
    gs = VARIANT_GROUPS[variant]
    keep = [g for g in gs if dec.get(g, False)]
    info["groups"], info["dropped"] = keep, [g for g in gs if g not in keep]
    if variant == "sg250":
        if not keep:
            info["skip"] = "군 제외(O, 90 % 규칙)"
            return None, info
        return [c for c in x25 if c not in soil_cols] + list(SG250_COLS), info
    if not keep:
        info["skip"] = "군 제외(90 % 규칙)" if variant.startswith("add_") else "모든 군 제외"
        return None, info
    extra = [c for g in H_GROUPS if g in keep for c in GROUPS[g]]
    if variant == "xh_px":
        extra = [PX_MAP.get(c, c) for c in extra]
    return x25 + extra, info


def group_finite_fraction(df, cols) -> float:
    """군의 유한값 비율 = (행, 열) 칸 가운데 유한한 칸의 비율. 열이 표에 없으면 그 열의 칸은 모두 결측으로 센다."""
    if len(df) == 0 or not cols:
        return float("nan")
    tot = len(df) * len(cols)
    fin = 0
    for c in cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            fin += int(np.isfinite(v).sum())
    return fin / tot


def group_decisions(feat, target_of, groups=H_GROUPS, extra=None, min_frac=FINITE_MIN) -> dict:
    """대상마다 군의 유한값 비율과 포함 여부(계획 2.5 결측 규칙). feat = loc_id 를 열로 갖는 특징 표, target_of = loc_id → 대상 이름(Series).
    extra = {이름: 열 목록}(지역 피복 등). 반환 {대상: {군: dict(finite_frac, included, reason, n_rows)}}."""
    t = feat.loc_id.map(target_of)
    out = {}
    gl = {g: GROUPS[g] for g in groups}
    gl.update(extra or {})
    for tg in TARGETS:
        sub = feat[t.values == tg]
        out[tg] = {}
        for g, cols in gl.items():
            fr = group_finite_fraction(sub, cols)
            inc = bool(np.isfinite(fr) and fr >= min_frac)
            out[tg][g] = dict(finite_frac=None if not np.isfinite(fr) else round(float(fr), 6), included=inc, n_rows=int(len(sub)),
                              reason="" if inc else ("행 없음" if not len(sub) else f"유한값 비율 {fr:.3f} < {min_frac:.2f}"))
    return out


def decisions_for(meta, target) -> dict:
    """특징 메타의 group_decisions 에서 한 대상의 {군: 포함 여부}."""
    g = (meta or {}).get("group_decisions", {}).get(target)
    if g is None:
        return None
    return {k: bool(v.get("included")) for k, v in g.items()}


# ================================================================ 해시(정준 CSV)
def canonical_csv(df, cols) -> bytes:
    """loc_id 순으로 정렬한 (loc_id + cols) 의 정준 CSV(소수 17자리 반올림 표기). 열 부분집합의 해시를 판 사이에서 비교하려고 쓴다."""
    d = df[["loc_id"] + list(cols)].sort_values("loc_id")
    buf = io.StringIO()
    d.to_csv(buf, index=False, float_format="%.17g", na_rep="nan")
    return buf.getvalue().encode()


def cols_sha(df, cols) -> str:
    """열 부분집합의 sha256(정준 CSV). 1단계 판(R1b)과 최종판(R3)의 H0·T2 열이 같은지 대조한다."""
    return hashlib.sha256(canonical_csv(df, cols)).hexdigest()


def sha256_file(path, n=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    s = h.hexdigest()
    return s[:int(n)] if n else s
