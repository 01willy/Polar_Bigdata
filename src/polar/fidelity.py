"""Source-aware multi-fidelity ALT 공통 스키마·split·누설통제 유틸 (S0).

`docs/RESEARCH_PLAN_multifidelity_2026-07-22.md` §4·§5 구현. 모델 무관(GBM·MLP·FT-T·
TabPFN·source-aware DL 등 어느 모델이든 같은 데이터·fold·누설규약을 쓰도록) 공통 계층.

핵심 원칙:
- 공변량 코어(34)는 전량 사용. 라벨 파생 7종은 입력에서 영구 제외(타깃 누설).
- fold는 0.5° 공간블록 GroupKFold / LORO(region) / leave-source-out 세 축.
- Stefan E 계수는 반드시 train fold 내부에서만 역산(fold-safe). test 라벨 미사용.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# ============================================================
# 1. 공변량 코어 (전량 사용) vs 라벨 파생 (영구 제외)
# ============================================================
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]  # 6
CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]  # 8
SOIL = ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15", "sg_cfvo_5_15",
        "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15", "sg_soc_15_30"]  # 9
INSAR = ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]  # 5
POLSAR = ["polsar_alt", "polsar_std", "polsar_valid"]  # 3
CCI = ["cci_alt", "cci_valid"]  # 2
FLAGS = ["insar_miss"]  # 1

# 공변량 코어 34 (지형6+기후8+토양9+InSAR5+PolSAR3+CCI2+flag1). input 최대 활용.
COVARIATE_CORE = TERRAIN + CLIMATE + SOIL + INSAR + POLSAR + CCI + FLAGS

# 타깃에서 파생 → 입력·SHAP에서 영구 제외(누설). test_leakage.py가 강제.
# sigma_prior_cm은 alt_sd(라벨) 파생이므로 feature 금지(S3 σ 역가중 시 per-cell 누설 방지).
LABEL_DERIVED_BANNED = ["alt_sd", "alt_min", "alt_max", "n_obs", "n_years", "year_min", "year_max",
                        "sigma_prior_cm"]

TARGET = "alt_cm"
# pooled 전이(공유 feature-only) 트랙: SAR 없는 지역용. SAR는 알래스카만 유효.
SHARED_CORE = TERRAIN + CLIMATE + SOIL + CCI  # 25 (전지역 가용)

# LORO 전이 단위 = 매크로 지역. 동일 지점의 세부 라벨(ABoVE_AK ↔ United States (Alaska))이
# train/test로 갈리는 지리 누설 방지(전체 리뷰 must_fix).
MACRO_REGION = {
    "ABoVE_AK": "Alaska", "United States (Alaska)": "Alaska",
    "ABoVE_CA": "Canada", "Canada": "Canada",
    "Lena_RU": "Lena",
    "GTNPenv_RU": "Siberia_GTNP", "GTNPenv_SJ": "Svalbard", "GTNPenv_US": "US_GTNP",
    "GTNPenv_CH": "Alps", "GTNPenv_AQ": "Antarctica", "QTP_CN": "Tibet",
}


def macro_region(df):
    return df["region"].map(lambda r: MACRO_REGION.get(r, r)).values

# ============================================================
# 2. group key (누설통제)
# ============================================================
BLOCK_DEG = 0.5  # 공간블록 크기(°). site-GKF 누설(같은 블록이 train/test 갈림) 방지.


def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """0.5° 공간블록 id 등 group key 부여. 같은 block은 반드시 같은 fold."""
    df = df.copy()
    df["block"] = (np.floor(df.lat / BLOCK_DEG).astype(int) * 100000
                   + np.floor(df.lon / BLOCK_DEG).astype(int))
    if "loc_id" not in df.columns:
        df["loc_id"] = np.arange(len(df))
    return df


# ============================================================
# 3. split (세 축) — 전부 block/region 그룹 단위, 누설 없음
# ============================================================
def spatial_block_splits(df: pd.DataFrame, n_splits: int = 6, sub_idx=None):
    """0.5° 공간블록 GroupKFold. in-domain 평가(공간 자기상관 통제)."""
    idx = np.arange(len(df)) if sub_idx is None else np.asarray(sub_idx)
    g = df["block"].values[idx]
    return [(idx[tr], idx[te]) for tr, te in GroupKFold(n_splits=n_splits).split(idx, groups=g)]


def loro_splits(df: pd.DataFrame, min_test: int = 100):
    """Leave-One-(macro)Region-Out. 매크로 지역 단위로 전이(OOD) 평가.
    동일 지점의 세부 라벨이 train/test로 갈리지 않도록 MACRO_REGION으로 묶는다."""
    macro = macro_region(df)
    out = []
    for r in pd.unique(macro):
        te = np.where(macro == r)[0]
        if len(te) < min_test:
            continue
        tr = np.where(macro != r)[0]
        out.append((r, tr, te))
    return out


def leave_source_out_splits(long_df: pd.DataFrame, source_col: str = "source_id"):
    """관측 source를 하나씩 빼고 학습(source-aware 검증). long-format 관측 테이블용."""
    out = []
    for s in pd.unique(long_df[source_col]):
        te = np.where(long_df[source_col].values == s)[0]
        tr = np.where(long_df[source_col].values != s)[0]
        out.append((s, tr, te))
    return out


# ============================================================
# 4. fold-safe Stefan E 역산 (물리 F1 소스 생성)
# ============================================================
def fit_stefan_E(alt_cm: np.ndarray, sqrt_tdd: np.ndarray) -> float:
    """ALT = E·√TDD 의 E를 최소제곱 역산. train fold 라벨만 넘겨야 함(호출자 책임)."""
    m = np.isfinite(alt_cm) & np.isfinite(sqrt_tdd) & (sqrt_tdd > 0)
    if m.sum() < 3:
        return np.nan
    # E = argmin ||alt - E·s||^2 = (s·alt)/(s·s)
    s, a = sqrt_tdd[m], alt_cm[m]
    return float((s @ a) / (s @ s))


def stefan_predict(sqrt_tdd: np.ndarray, E: float) -> np.ndarray:
    return E * sqrt_tdd


def assert_fold_safe_E(train_idx, test_idx, df) -> None:
    """test 셀이 E 역산 train에 포함되지 않았는지 assert(fold-safe 보증)."""
    inter = set(np.asarray(train_idx).tolist()) & set(np.asarray(test_idx).tolist())
    assert not inter, f"fold-safe 위반: train/test 교집합 {len(inter)}개"


# ============================================================
# 5. fidelity 체계 (F4>F3>F2>F1>F0) + source 정의
# ============================================================
FIDELITY = {
    "F4_direct": 4,    # 직접 탐침(ABoVE·Lena)
    "F3_temp": 3,      # 온도유도(shallow3d·field3d·KPDC)
    "F2_gtnp_env": 2,  # 심부 borehole 포락선
    "F1_stefan": 1,    # 물리 경험식
    "F0_remote": 0,    # 원격(InSAR/PolSAR/CCI/ReSALT)
}


def covariates_present(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """각 공변량 그룹의 유효율(%)을 region×group으로 요약(overlap 진단 보조)."""
    cols = cols or COVARIATE_CORE
    groups = {"terrain": TERRAIN, "climate": CLIMATE, "soil": SOIL,
              "insar": ["insar_alt"], "polsar": ["polsar_alt"], "cci": ["cci_alt"]}
    rows = []
    for reg, g in df.groupby("region"):
        for name, gc in groups.items():
            gc = [c for c in gc if c in df.columns]
            if gc:
                rows.append(dict(region=reg, group=name,
                                 valid_pct=round(100 * g[gc].notna().all(axis=1).mean(), 1),
                                 n=len(g)))
    return pd.DataFrame(rows)
