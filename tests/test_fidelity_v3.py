"""fidelity_base_v3 (S-A 데이터 정비, 2026-09-18) 무결성·매크로·전이 집합 검증.

실행: pytest tests/test_fidelity_v3.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import (FIDELITY, MACRO_REGION, TRANSFER_MAIN, TRANSFER_DEEP,  # noqa: E402
                            add_group_keys, loro_splits, macro_region, spatial_block_splits)

V2 = ROOT / "data" / "processed" / "fidelity_base_v2.csv"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
META = ROOT / "data" / "processed" / "fidelity_base_v3_meta.json"
SOIL = ROOT / "data" / "processed" / "e5_soil_tdd.csv"
SOIL3 = ROOT / "data" / "processed" / "e5_soil_tdd_v3.csv"
UPDATED = ["region", "source_id", "fidelity_level"]

pytestmark = pytest.mark.skipif(not V3.exists(), reason="fidelity_base_v3.csv 미생성")


@pytest.fixture(scope="module")
def v2():
    return pd.read_csv(V2, low_memory=False)


@pytest.fixture(scope="module")
def v3():
    return pd.read_csv(V3, low_memory=False)


def test_v3_fixed_columns_unchanged(v2, v3):
    """region·source_id·fidelity_level 외 모든 열은 v2와 값이 같다(loc_id·좌표·공변량·라벨 불변)."""
    assert list(v3.columns) == list(v2.columns) and len(v3) == len(v2) == 17572
    fixed = [c for c in v2.columns if c not in UPDATED]
    pd.testing.assert_frame_equal(v3[fixed], v2[fixed], check_dtype=False, check_exact=False, atol=1e-9)
    assert v3.loc_id.is_unique


def test_v3_changes_confined_to_calm_rows(v2, v3):
    """변경 행은 CALM 신규 셀(v2에서 CALM_*)에 한정: region 3행(→ CALM_Scandinavia), source_id 68행(→ F4_calm_temp)."""
    reg = v2.region != v3.region
    src = v2.source_id != v3.source_id
    assert reg.sum() == 3 and set(v3.region[reg]) == {"CALM_Scandinavia"} and set(v2.region[reg]) == {"CALM_Svalbard"}
    assert (v3.lat[reg] < 76).all()
    assert src.sum() == 68 and set(v3.source_id[src]) == {"F4_calm_temp"}
    assert v2.region.str.startswith("CALM_")[reg | src].all()
    assert FIDELITY["F4_calm_temp"] == 3
    assert (v3.fidelity_level[v3.source_id == "F4_calm_temp"] == 3).all()
    assert (v3.fidelity_level[v3.source_id == "F4_direct"] == 4).all()
    # F2 포락선(GTNPenv_SJ 등)은 손대지 않음
    assert (v3.source_id[v2.source_id == "F2_gtnp_env"] == "F2_gtnp_env").all()


def test_v3_macro_mapping_and_transfer_sets(v3):
    d = add_group_keys(v3)
    macro = macro_region(d)
    assert set(d.region.unique()) <= set(MACRO_REGION), set(d.region.unique()) - set(MACRO_REGION)
    new_macro = set(macro[d.region.str.startswith("CALM_").values])
    assert new_macro <= set(TRANSFER_MAIN + TRANSFER_DEEP)
    assert "Scandinavia" in TRANSFER_DEEP and "Scandinavia" not in TRANSFER_MAIN
    assert set(macro[d.region.values == "CALM_Scandinavia"]) == {"Scandinavia"}
    assert set(macro[d.region.values == "GTNPenv_SJ"]) == {"Svalbard"}
    assert (macro == "Scandinavia").sum() == 3
    assert (macro == "Svalbard").sum() == 4 + 10  # CALM 진짜 스발바르 4 + GTNPenv_SJ 10


def test_v3_splits_work(v3):
    d = add_group_keys(v3)
    macro = macro_region(d)
    for r, tr, te in loro_splits(d, min_test=3):
        assert set(macro[te]) == {r} and r not in set(macro[tr])
    for tr, te in spatial_block_splits(d):
        assert set(d.block.values[tr]).isdisjoint(set(d.block.values[te]))


def test_v3_main_set_probe_cells_retained(v2, v3):
    """주 집합(TRANSFER_MAIN)의 탐침 셀은 100% 유지. 캐나다만 지온 유도 CALM 2셀(C16·C17)이 F4_calm_temp로 빠진다."""
    a = add_group_keys(v2); a["macro"] = macro_region(a)
    b = add_group_keys(v3); b["macro"] = macro_region(b)
    for m in TRANSFER_MAIN:
        n2 = int(((a.macro == m) & (a.source_id == "F4_direct")).sum())
        n3 = int(((b.macro == m) & (b.source_id == "F4_direct")).sum())
        assert n3 == n2 - (2 if m == "Canada" else 0), (m, n2, n3)
    # 탐침 셀 집합 자체가 보존되는지(loc_id 기준)
    probe2 = set(a.loc_id[(a.source_id == "F4_direct") & a.macro.isin(TRANSFER_MAIN) & (b.source_id == "F4_direct")])
    probe3 = set(b.loc_id[(b.source_id == "F4_direct") & b.macro.isin(TRANSFER_MAIN)])
    assert probe2 == probe3


def test_v3_meta_consistent(v3):
    m = json.loads(META.read_text())
    assert m["n_rows"] == len(v3) and m["fixed_cols_unchanged"] is True
    assert m["fixed_cols_hash_v2"] == m["fixed_cols_hash_v3"]
    assert m["obs_method"]["n_f4_calm_temp"] == 68 and m["obs_method"]["matches_document"] is True


@pytest.mark.skipif(not SOIL3.exists(), reason="e5_soil_tdd_v3.csv 미생성")
def test_soil_tdd_v3():
    s = pd.read_csv(SOIL); s3 = pd.read_csv(SOIL3)
    assert len(s) == len(s3) and list(s.columns) == list(s3.columns)
    zero = (s.e5_tdd_soil <= 0).fillna(False)
    assert zero.sum() == 8
    assert s3.e5_tdd_soil[zero].isna().all() and s3.e5_sqrt_tdd_soil[zero].isna().all()
    assert not (s3.e5_tdd_soil <= 0).any()
    other = [c for c in s.columns if c not in ("e5_tdd_soil", "e5_sqrt_tdd_soil")]
    pd.testing.assert_frame_equal(s[other], s3[other], check_dtype=False)
    pd.testing.assert_series_equal(s.e5_tdd_soil[~zero], s3.e5_tdd_soil[~zero])
