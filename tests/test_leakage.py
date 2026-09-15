"""S0 누설 방지 unit test 스위트.

`docs/RESEARCH_PLAN_multifidelity_2026-07-22.md` §5 구현. 하나라도 실패 시 이후 모든
실험 결과는 무효로 간주. 과거 헤드라인 붕괴 원인(site-GKF 블록 공유·특징복제·타깃 파생
누설·fold 밖 물리보정)을 코드로 차단한다.

실행: pytest tests/test_leakage.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import (COVARIATE_CORE, LABEL_DERIVED_BANNED, SHARED_CORE, TARGET,
                            INSAR, POLSAR, add_group_keys, spatial_block_splits, loro_splits,
                            leave_source_out_splits, fit_stefan_E, assert_fold_safe_E)

BASE = ROOT / "data" / "processed" / "fidelity_base.csv"
LONG = ROOT / "data" / "processed" / "fidelity_observations_long.csv"


@pytest.fixture(scope="module")
def df():
    d = pd.read_csv(BASE, low_memory=False)
    return add_group_keys(d)


# ---- 1. 타깃 파생 컬럼 누설 방지 ----
def test_label_derived_not_in_covariates():
    """alt_sd·alt_min·alt_max·n_obs·n_years·year_* 는 절대 feature가 아니어야 함."""
    for c in LABEL_DERIVED_BANNED:
        assert c not in COVARIATE_CORE, f"라벨 파생 {c}가 공변량에 포함됨(타깃 누설)"


def test_target_not_in_covariates():
    assert TARGET not in COVARIATE_CORE, "타깃 alt_cm이 feature에 포함됨"


def test_shared_core_has_no_sar():
    """pooled 전이 트랙(SHARED_CORE)에는 SAR 금지(신규지역 결측 → region 교락)."""
    for c in INSAR + POLSAR:
        assert c not in SHARED_CORE, f"공유 코어에 SAR {c} 포함(전지역 결측 교락)"


# ---- 2. 공간블록 GroupKFold 누설 방지 ----
def test_spatial_block_no_shared_block(df):
    """같은 0.5° 블록이 train·test로 갈리면 site-GKF 누설. 절대 금지."""
    for tr, te in spatial_block_splits(df):
        assert len(set(tr) & set(te)) == 0, "train/test 인덱스 겹침"
        tr_blocks = set(df["block"].values[tr])
        te_blocks = set(df["block"].values[te])
        assert tr_blocks.isdisjoint(te_blocks), "같은 블록이 train·test에 분리(누설)"


def test_spatial_block_covers_all(df):
    """모든 셀이 정확히 한 번 test에 등장(폴드 커버리지)."""
    seen = np.zeros(len(df), dtype=int)
    for _, te in spatial_block_splits(df):
        seen[te] += 1
    assert (seen == 1).all(), "일부 셀이 test에 0회 또는 2회 이상 등장"


# ---- 3. LORO 지역 전이 누설 방지 ----
def test_loro_region_disjoint(df):
    """test 매크로지역 셀이 train에 절대 없어야 함(진짜 OOD)."""
    from polar.fidelity import macro_region
    macro = macro_region(df)
    for r, tr, te in loro_splits(df):
        assert set(macro[te]) == {r}, "test에 다른 매크로지역 혼입"
        assert r not in set(macro[tr]), f"train에 test 매크로지역 {r} 누설"


# ---- 4. leave-source-out 누설 방지 ----
def test_leave_source_out_disjoint():
    long = pd.read_csv(LONG)
    for s, tr, te in leave_source_out_splits(long):
        assert set(long["source_id"].values[te]) == {s}
        assert s not in set(long["source_id"].values[tr]), f"train에 test source {s} 누설"


# ---- 5. fold-safe Stefan E 역산 ----
def test_stefan_E_fold_safe(df):
    """E는 train 라벨로만 역산, test 셀은 배제되어야 함."""
    folds = spatial_block_splits(df)
    tr, te = folds[0]
    assert_fold_safe_E(tr, te, df)  # 교집합 있으면 AssertionError
    E = fit_stefan_E(df[TARGET].values[tr], df["e5_sqrt_tdd"].values[tr])
    assert np.isfinite(E) and E > 0, "train E 역산 실패"


def test_stefan_E_rejects_test_leakage(df):
    """train에 test 인덱스를 일부러 섞으면 assert가 잡아야 함(가드 자체 검증)."""
    folds = spatial_block_splits(df)
    tr, te = folds[0]
    poisoned = np.concatenate([tr, te[:5]])  # test 5개 오염
    with pytest.raises(AssertionError):
        assert_fold_safe_E(poisoned, te, df)


# ---- 6. base 데이터 무결성 ----
def test_base_has_all_core(df):
    for c in COVARIATE_CORE:
        assert c in df.columns, f"base에 공변량 {c} 없음"


def test_base_no_banned_leak_into_core(df):
    """base 파일이 실수로 라벨파생을 feature로 승격하지 않았는지(존재는 가능, 코어엔 불가)."""
    core_in_base = [c for c in COVARIATE_CORE if c in df.columns]
    assert not (set(core_in_base) & set(LABEL_DERIVED_BANNED)), "코어와 금지목록 교집합"


def test_sigma_prior_is_banned():
    """sigma_prior_cm(alt_sd 파생)은 feature 금지 목록에 있어야 함(S3 σ 누설 방지)."""
    assert "sigma_prior_cm" in LABEL_DERIVED_BANNED


# ---- 7. macro LORO 지리 누설 (동일 지점 세부라벨 분리 방지) ----
def test_macro_loro_no_alaska_split(df):
    """ABoVE_AK와 'United States (Alaska)'가 같은 매크로(Alaska)로 묶여 train/test 안 갈림."""
    from polar.fidelity import macro_region
    macro = macro_region(df)
    for r, tr, te in loro_splits(df):
        tr_macro = set(macro[tr]); te_macro = set(macro[te])
        assert tr_macro.isdisjoint(te_macro), f"매크로 지역 {r}이 train·test에 분리(누설)"
    # US Alaska가 Alaska 매크로로 묶였는지
    assert set(macro[df.region.values == "United States (Alaska)"]) <= {"Alaska"}


# ---- 8. physics.fit_E fold-safe (실제 프로덕션 E 추정기) ----
def test_physics_fit_E_uses_train_only():
    """physics.fit_E가 mask=train만 반영, test 라벨 오염에 불변인지(실사용 경로 회귀 테스트)."""
    from polar.physics import fit_E as pfit_E
    rng = np.random.RandomState(0)
    n = 200
    sqtdd = rng.uniform(15, 45, n)
    alt = 1.5 * sqtdd + rng.normal(0, 3, n)
    mask = np.zeros(n, bool); mask[:100] = True   # train=앞100
    E_clean = pfit_E(alt, sqtdd, mask)
    alt2 = alt.copy(); alt2[100:] = 9999.0          # test 라벨 오염
    E_poisoned = pfit_E(alt2, sqtdd, mask)
    assert abs(E_clean - E_poisoned) < 1e-9, "physics.fit_E가 test 라벨에 반응(fold-safe 위반)"


# ---- 9. fold_prep fold-safe 표준화 (train 통계만) ----
def test_fold_prep_train_only_stats():
    """fold_prep의 median·표준화가 test 통계 변화에 불변인지(train-only)."""
    from polar.preprocessing import fold_prep
    rng = np.random.RandomState(1)
    Xtr = rng.normal(5, 2, (80, 4)); Xte = rng.normal(5, 2, (20, 4))
    _, Xte_a = fold_prep(Xtr.copy(), Xte.copy(), nan_native=False)
    Xte2 = Xte.copy(); Xte2 += 1000.0                # test 분포 대폭 이동
    _, Xte_b = fold_prep(Xtr.copy(), Xte2.copy(), nan_native=False)
    # 같은 train이면 test 표준화는 (원본차이만큼만) 이동, train 통계는 불변
    diff = (Xte_b - Xte_a) - (1000.0 / (Xtr.std(0) + 1e-6))
    assert np.allclose(diff, 0, atol=1e-3), "fold_prep가 test 통계를 사용(누설)"


def test_fold_prep_nan_native_passthrough():
    from polar.preprocessing import fold_prep
    Xtr = np.array([[1.0, np.nan], [2.0, 3.0]]); Xte = np.array([[np.nan, 1.0]])
    a, b = fold_prep(Xtr, Xte, nan_native=True)
    assert np.array_equal(a, Xtr, equal_nan=True) and np.array_equal(b, Xte, equal_nan=True)


# ---- 9. E3 확충 데이터(fidelity_base_v2) 무결성 (2026-09-08) ----
BASE_V2 = ROOT / "data" / "processed" / "fidelity_base_v2.csv"


@pytest.mark.skipif(not BASE_V2.exists(), reason="fidelity_base_v2.csv 미생성")
def test_v2_old_rows_unchanged():
    """v2의 앞 17,423행은 fidelity_base.csv와 값이 같아야 한다(확충이 기존 셀을 바꾸지 않음)."""
    old = pd.read_csv(BASE, low_memory=False)
    v2 = pd.read_csv(BASE_V2, low_memory=False)
    assert list(v2.columns) == list(old.columns)
    head = v2.iloc[:len(old)].reset_index(drop=True)
    pd.testing.assert_frame_equal(head, old, check_dtype=False, check_exact=False, atol=1e-9)
    assert v2.loc_id.is_unique


@pytest.mark.skipif(not BASE_V2.exists(), reason="fidelity_base_v2.csv 미생성")
def test_v2_new_regions_mapped_and_disjoint():
    """신규 CALM_* 지역은 전부 MACRO_REGION에 매핑되고 주/심부 전이 집합 중 하나에 속하며,
    LORO에서 매크로 지역이 train·test로 갈리지 않는다. 신규 셀은 기존 셀과 0.01° 이상 떨어진다."""
    from polar.fidelity import macro_region, MACRO_REGION, TRANSFER_MAIN, TRANSFER_DEEP
    v2 = add_group_keys(pd.read_csv(BASE_V2, low_memory=False))
    new = v2[v2.region.astype(str).str.startswith("CALM_")]
    assert len(new) > 0
    for r in new.region.unique():
        assert r in MACRO_REGION, f"{r} 매크로 매핑 없음"
    macro = macro_region(v2)
    new_macro = set(macro[v2.region.astype(str).str.startswith("CALM_").values])
    assert new_macro <= set(TRANSFER_MAIN + TRANSFER_DEEP), new_macro - set(TRANSFER_MAIN + TRANSFER_DEEP)
    for r, tr, te in loro_splits(v2, min_test=3):
        assert set(macro[tr]).isdisjoint(set(macro[te])), f"매크로 {r} train/test 분리"
    old = pd.read_csv(BASE, low_memory=False)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(old[["lat", "lon"]].values).query(new[["lat", "lon"]].values, p=np.inf)
    assert (d >= 0.01).all(), f"기존 셀과 0.01° 이내 신규 셀 {(d < 0.01).sum()}개"
    # 신규 셀은 SAR 결측·insar_miss=1·F4_direct
    assert new.insar_alt.isna().all() and (new.insar_miss == 1).all() and (new.source_id == "F4_direct").all()
