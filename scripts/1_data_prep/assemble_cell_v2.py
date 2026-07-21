"""다지역 통합 셀 데이터셋 v2 조립.

- 기존 data/processed/dl_dataset_cell.csv 14,348행은 그대로 유지(수정 없음).
- 신규 지역(Lena_RU 3,037 + QTP_CN 1 + GTNPenv_* 45) 행을 append.
- 신규 행 규칙
  * loc_id: 기존 max+1부터 연속 부여.
  * insar_*(insar_alt/alt_std/sub/dist/n)·polsar_alt/std = NaN 유지, polsar_valid=0.
    알래스카 중앙값으로 채우지 않는다(학습 시 fold별 대체 전제).
  * cci_alt/cci_valid는 enrich_new_regions.py에서 부착한 값.
- insar_miss 컬럼 신설
  * 기존 행: insar_dist>5.0 -> 1, 아니면 0.
    (기존 insar_*는 부착 단계에서 이미 중앙값 대체가 끝난 값이므로 원거리 여부를
     insar_dist로 사후 플래그한다.)
  * 신규 행: 1 (InSAR 미부착).
- 라벨 QC 컬럼(qc, qc_flag, borehole_id, site, censor_flag) 유지
  * 신규 행: 소스 값 그대로(GTNPenv qc=high/approx, ALLena qc_flag, QTP censor_flag=1).
  * 기존 행: qc/qc_flag/borehole_id/site=NaN(비해당), censor_flag=0.
- 무결성 검사: round(4) 좌표 전역 유일성, GTNPenv 행과 타 행 간 0.01°(체비쇼프)
  근접 없음, 기존 14,348행 값 불변.
- block 컬럼은 만들지 않는다(하네스가 lat/lon에서 계산).

산출: data/processed/dl_dataset_cell_v2.csv, dl_dataset_cell_v2_meta.json
실행: /home/anaconda3/bin/python scripts/1_data_prep/assemble_cell_v2.py  (ROOT에서)
"""
import json

import numpy as np
import pandas as pd

BASE = "data/processed/dl_dataset_cell.csv"
NEW = "data/processed/new_regions_covariates.csv"
OUT = "data/processed/dl_dataset_cell_v2.csv"
META = "data/processed/dl_dataset_cell_v2_meta.json"
INSAR_DIST_TH = 5.0

QC_COLS = ["qc", "qc_flag", "borehole_id", "site", "censor_flag"]

old = pd.read_csv(BASE)
assert len(old) == 14348, f"기존 셀 행수 변동: {len(old)}"
new = pd.read_csv(NEW)

old = old.copy()
old["insar_miss"] = (old["insar_dist"] > INSAR_DIST_TH).astype(int)
for c in ["qc", "qc_flag", "borehole_id", "site"]:
    old[c] = np.nan                       # 기존 행 비해당
old["censor_flag"] = 0                    # 기존 라벨은 비censored

rows = pd.DataFrame(index=range(len(new)), columns=old.columns)
for c in old.columns:
    if c in new.columns:
        rows[c] = new[c].values
rows["loc_id"] = np.arange(len(new)) + int(old.loc_id.max()) + 1
rows["polsar_valid"] = 0
rows["insar_miss"] = 1
# insar_*·polsar_alt/std는 NaN 유지(컬럼 대응 없음 -> 자동 NaN)
rows = rows.astype({c: old[c].dtype for c in old.columns
                    if c not in QC_COLS and old[c].dtype != object
                    and rows[c].notna().all()})

v2 = pd.concat([old, rows], ignore_index=True)
assert v2.loc_id.is_unique

# ---------- 무결성 검사 ----------
# (1) round(4) 좌표 전역 유일성 (중복 셀 방지)
key = list(zip(v2.lat.round(4), v2.lon.round(4)))
assert len(set(key)) == len(key), "round(4) 좌표 중복 존재"
# (2) GTNPenv 행은 타 행과 0.01°(체비쇼프) 이상 분리 (dedup 정책 준수)
envm = v2.region.astype(str).str.startswith("GTNPenv")
la, lo = v2.lat.values, v2.lon.values
for i in np.where(envm.values)[0]:
    near = (np.abs(la - la[i]) <= 0.01) & (np.abs(lo - lo[i]) <= 0.01)
    near[i] = False
    assert not near.any(), f"GTNPenv 행 loc_id={v2.loc_id.iloc[i]} 근접 중복"
# (3) 기존 행 값 불변
base_chk = pd.read_csv(BASE)
for c in base_chk.columns:
    a, b = v2[c].values[:len(base_chk)], base_chk[c].values
    if base_chk[c].dtype == object:
        assert (pd.Series(a).fillna("") == pd.Series(b).fillna("")).all(), c
    else:
        assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), c
print("[check] round4 유일성 / GTNPenv 근접 0 / 기존 행 값 불변: 통과")

v2.to_csv(OUT, index=False)
print(f"[saved] {OUT}: {len(old):,} + {len(rows):,} = {len(v2):,}행 x {v2.shape[1]}열")

# ---------- 메타 JSON ----------
cov_cols = [c for c in v2.columns if c.startswith(("dem_", "e5_", "cci_", "insar_", "polsar_"))]
by_region = {}
for reg, g in v2.groupby("region"):
    q1, q3 = g.alt_cm.quantile([0.25, 0.75])
    by_region[reg] = dict(
        n_rows=int(len(g)),
        alt_cm_median=round(float(g.alt_cm.median()), 1),
        alt_cm_iqr=[round(float(q1), 1), round(float(q3), 1)],
        nan_rate={c: round(float(g[c].isna().mean()), 4) for c in cov_cols
                  if g[c].isna().any()},
    )
# 알려진 한계(known issues) 대상 행 식별
ch = v2[v2.region == "GTNPenv_CH"]
lena_deep = v2[(v2.region == "Lena_RU") & (v2.alt_cm > 200)]
meta = dict(
    created="2026-07-14",
    base=BASE, base_rows=int(len(old)), new_rows=int(len(rows)), total_rows=int(len(v2)),
    insar_miss_rule=f"기존 행 insar_dist>{INSAR_DIST_TH} -> 1 (n={int(old.insar_miss.sum())}), 신규 행 1",
    note=("신규 행 insar_*/polsar_* NaN 유지(fold별 대체 전제), polsar_valid=0. "
          "기존 행 insar_*는 부착 단계 중앙값 대체 완료값. "
          "GTNPenv_AQ 1행은 ERA5 NH 커버리지 밖이라 e5_* NaN."),
    dedup_policy=("GTNPenv: 기존 데이터(round4·0.01° 체비쇼프) 대비 제외 + envelope 내부 "
                  "0.01° 클러스터 병합(좌표 평균, alt 평균, alt_sd=시추공 간 SD). "
                  "Lena_RU는 round4 위치 집계이며 0.01° 셀 내 다행 밀집 허용(기존 규약과 동일)."),
    label_semantics=dict(
        n_obs=("원시 관측 수. Lena_RU=연중 최대 산출에 쓰인 관측 합, "
               "GTNPenv=상부 3m 깊이별 월 레코드 합, QTP_CN=유효(비censored) 연도 수."),
        alt_sd=("Lena_RU·QTP_CN=연간 최대값의 연도 간 SD(단일연도 0). "
                "GTNPenv 단일공=NaN(연간 SD 미산출, 불확실성 0 아님), "
                "GTNPenv 병합행=시추공 간 ALT SD."),
        n_years=("Lena_RU=관측 연도 수(고유), GTNPenv=관측기간 연수(year_max-year_min+1), "
                 "QTP_CN=유효 연도 수."),
        qc="GTNPenv 전용: high=전 깊이 seasonal 샘플링, approx=비계절 샘플링 포함.",
        qc_flag="Lena_RU 전용: 1=그림 추출값 포함 위치(PANGAEA QF TD==1).",
        censor_flag=("1=censored 연도 제외로 라벨이 하한 성격(QTP_CN Wudaoliang: "
                     "2015-2018·2020 ALT>320cm censored 제외, 2014·2019 평균). 기존 행 0."),
        borehole_id="GTNPenv 전용: GTN-P borehole id(병합행은 ';' 연결).",
    ),
    known_issues=[
        dict(issue="alps_covariate_mismatch",
             desc=("GTNPenv_CH 2행(표고 2669-2907m): ERA5-Land 0.1° 격자의 산악 표고 "
                   "평활로 기후 공변량이 시추공 고도 대비 과온(e5_maat 평균 +1.1°C)인데 "
                   "alt_cm 중앙 272.5cm. 학습 시 off-manifold 표본 후보(2행이라 영향 제한적)."),
             loc_ids=[int(x) for x in ch.loc_id],
             e5_maat=[round(float(x), 2) for x in ch.e5_maat]),
        dict(issue="lena_deep_single_obs",
             desc=("Lena_RU alt_cm>200cm 행: 전부 2020년 단일 관측(열카르스트·수변 "
                   "심융해 가능성, 검증 불가). 위치 블록 CV 시 지역 가중 확인 필요."),
             loc_ids=[int(x) for x in lena_deep.loc_id],
             alt_cm=[round(float(x), 1) for x in lena_deep.alt_cm]),
        dict(issue="qtp_label_censoring",
             desc="QTP_CN 1행은 censor_flag=1(라벨 하한 성격). 학습·평가 시 참조.",
             loc_ids=[int(x) for x in v2.loc_id[v2.censor_flag == 1]]),
    ],
    scripts=[
        "scripts/1_data_prep/parse_allena.py",
        "scripts/1_data_prep/parse_qtec.py",
        "scripts/1_data_prep/derive_alt_gtnp_envelope.py",
        "scripts/1_data_prep/enrich_new_regions.py",
        "scripts/0_download/copernicus_dem.py",
        "scripts/1_data_prep/assemble_cell_v2.py",
    ],
    regions=by_region,
)
with open(META, "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[saved] {META}")
print(v2.region.value_counts().to_string())
print("insar_miss:", v2.insar_miss.value_counts().to_dict())
