"""S0: Source-aware multi-fidelity 통합 스키마 + overlap matrix + long-format 관측 테이블.

`docs/RESEARCH_PLAN_multifidelity_2026-07-22.md` §4 구현. 모델 무관 base 계층을 만든다.

입력:  data/processed/dl_dataset_cell_v3_soil.csv (17,423셀, 51컬럼)
       (+ shallow3d_alaska_altmatch.csv, field3d_reeval_altmatch.csv = F3 온도유도 관측)
산출:  data/processed/fidelity_base.csv            공변량 코어34 + group key + fidelity 메타 (wide, 학습 base)
       data/processed/source_overlap_matrix.csv    source쌍 셀 overlap (A5 식별성 gate)
       data/processed/fidelity_observations_long.csv  (cell × source) 관측 long (source-aware용)
       data/processed/fidelity_schema_meta.json     스키마·제외컬럼·해시

실행: python scripts/1_data_prep/build_fidelity_schema.py
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (COVARIATE_CORE, LABEL_DERIVED_BANNED, SHARED_CORE, TARGET,
                            add_group_keys, covariates_present, FIDELITY)

PROC = C.PROCESSED
CELL = PROC / "dl_dataset_cell_v3_soil.csv"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:12] if Path(path).exists() else "NA"


# ---------------- 1. base 로드 + group key + 메타 ----------------
df = pd.read_csv(CELL, low_memory=False)
df = add_group_keys(df)
print(f"[load] {len(df):,}셀 · {df.shape[1]}컬럼 · region {df.region.nunique()}종")

# 누설 방어: 라벨 파생 컬럼은 base의 feature 목록에서 제외(존재하면 표시만, 학습 입력엔 안 씀)
present_banned = [c for c in LABEL_DERIVED_BANNED if c in df.columns]
core = [c for c in COVARIATE_CORE if c in df.columns]
missing_core = [c for c in COVARIATE_CORE if c not in df.columns]
print(f"[feature] 공변량 코어 {len(core)}/{len(COVARIATE_CORE)} 사용 · 라벨파생 제외 {present_banned}")
if missing_core:
    print(f"[warn] base에 없는 코어 컬럼: {missing_core}")

# fidelity 메타: 각 셀의 direct 라벨은 F4(대부분)·F2(GTNPenv 심부유도). censor_flag 반영.
df["source_id"] = "F4_direct"
df.loc[df.region.astype(str).str.startswith("GTNPenv"), "source_id"] = "F2_gtnp_env"
df["fidelity_level"] = df["source_id"].map(FIDELITY)
# 공간지지 규모(m): 셀 라벨 100m. (탐침 point 0.3m는 원자료 레벨)
df["spatial_support_m"] = 100.0
# 관측 불확실성 prior(cm): 있으면 alt_sd, 없으면 지역 대푯값. (feature 아님, σ_s prior 초기화용)
df["sigma_prior_cm"] = pd.to_numeric(df.get("alt_sd"), errors="coerce").fillna(12.0).clip(3, 40)
# 검열: censor_flag(우측검열) + alt_max 탐침한계(150 부근) 힌트
df["right_censored"] = (pd.to_numeric(df.get("censor_flag"), errors="coerce").fillna(0) > 0).astype(int)

# base 저장(공변량 코어 + group + 메타 + 타깃). 라벨파생은 진단용으로 남기되 feature 목록엔 없음.
keep = (["loc_id", "lat", "lon", "region", "block", TARGET,
         "source_id", "fidelity_level", "spatial_support_m", "sigma_prior_cm", "right_censored"]
        + core)
base = df[[c for c in keep if c in df.columns]].copy()
base.to_csv(PROC / "fidelity_base.csv", index=False)
print(f"[save] fidelity_base.csv · {base.shape}")


# ---------------- 2. source overlap matrix (A5 식별성 gate) ----------------
# 각 셀에서 어떤 source의 ALT 관측이 유효한가 → source쌍 셀 overlap.
avail = pd.DataFrame({
    "F4_direct": df[TARGET].notna(),
    "F1_stefan": df["e5_sqrt_tdd"].notna() if "e5_sqrt_tdd" in df else False,  # 전 셀 계산 가능
    "F0_insar": df["insar_alt"].notna() if "insar_alt" in df else False,
    "F0_polsar": df["polsar_alt"].notna() if "polsar_alt" in df else False,
    "F0_cci": df["cci_alt"].notna() if "cci_alt" in df else False,
})
srcs = list(avail.columns)
n = len(df)
ov = pd.DataFrame(index=srcs, columns=srcs, dtype=float)
for a in srcs:
    for b in srcs:
        both = (avail[a] & avail[b]).sum()
        ov.loc[a, b] = round(100 * both / n, 1)
ov.to_csv(PROC / "source_overlap_matrix.csv")
print("[overlap] direct 대비 각 source 셀 overlap(%):")
for b in srcs:
    if b != "F4_direct":
        print(f"    {b:12s}: {ov.loc['F4_direct', b]:5.1f}%")

# F3 온도유도(별도 파일) paired overlap
f3_pairs = 0
for f in ["shallow3d_alaska_altmatch.csv", "field3d_reeval_altmatch.csv"]:
    p = PROC / f
    if p.exists():
        t = pd.read_csv(p)
        obs_col = [c for c in t.columns if "obs" in c.lower() and "alt" in c.lower()]
        f3_pairs += t[obs_col[0]].notna().sum() if obs_col else len(t)
print(f"    F3_temp(paired obs/pred): {f3_pairs}쌍 (셀 overlap 희소, pooled b_temp까지만 식별)")


# ---------------- 3. long-format 관측 테이블 (source-aware용) ----------------
# 각 셀의 여러 source 관측을 long으로. y_value·source·fidelity·sigma·support.
rows = []
covs = df[["loc_id", "region", "block"] + core].copy()
# F4 direct
d = df[df[TARGET].notna()]
rows.append(pd.DataFrame(dict(loc_id=d.loc_id, y_cm=d[TARGET], source_id=d.source_id,
                             fidelity=d.fidelity_level, sigma_cm=d.sigma_prior_cm,
                             support_m=100.0, censored=d.right_censored)))
# F0 remote (관측이 아닌 별도 소스로. source-aware b_s 추정 대상)
for col, sid, sup in [("insar_alt", "F0_insar", 30.0), ("polsar_alt", "F0_polsar", 30.0),
                      ("cci_alt", "F0_cci", 1000.0)]:
    if col in df.columns:
        m = df[col].notna()
        rows.append(pd.DataFrame(dict(loc_id=df.loc_id[m], y_cm=df[col][m], source_id=sid,
                                     fidelity=0, sigma_cm=31.0, support_m=sup, censored=0)))
long = pd.concat(rows, ignore_index=True)
long.to_csv(PROC / "fidelity_observations_long.csv", index=False)
print(f"[save] fidelity_observations_long.csv · {len(long):,}행 · source {long.source_id.value_counts().to_dict()}")


# ---------------- 4. 공변량 가용성 진단 (지역 트랙 결정 근거) ----------------
cp = covariates_present(df)
cp.to_csv(PROC / "covariate_availability_by_region.csv", index=False)
piv = cp.pivot_table(index="group", columns="region", values="valid_pct", aggfunc="first")
main = [r for r in ["ABoVE_AK", "Lena_RU", "ABoVE_CA"] if r in piv.columns]
print("[covariate] 주요 지역 그룹 유효율(%):")
print(piv[main].to_string())


# ---------------- 5. 메타 저장 ----------------
meta = dict(
    purpose="S0 source-aware multi-fidelity 통합 스키마 + overlap gate",
    n_cells=int(len(df)), n_covariate_core=len(core), covariate_core=core,
    shared_core=SHARED_CORE, label_derived_banned=present_banned,
    overlap_direct_pct={b: float(ov.loc["F4_direct", b]) for b in srcs if b != "F4_direct"},
    f3_paired=int(f3_pairs),
    input_hash=md5(CELL),
    note="공변량 코어34 전량 사용, 라벨파생7 영구제외. A5 식별성: Stefan·CCI만 clean overlap.",
)
(PROC / "fidelity_schema_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"[done] meta 저장. overlap gate: Stefan {ov.loc['F4_direct','F1_stefan']}% · CCI {ov.loc['F4_direct','F0_cci']}% clean")
