"""LGD 점 자료 → 약 1 km 셀 집계(ext_cells_v1). 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.3·6B.4.

입력
  data/processed/ext_labels/<src_id>_points.csv (형식 정의 _schema.json 1.0)
  data/processed/fidelity_base_v3.csv (읽기만. loc_id, lat, lon, region, source_id 열)
산출
  data/processed/ext_labels/ext_cells_v1.csv        셀 표(라벨 집합 × 셀)
  data/processed/ext_labels/ext_cells_v1_locs.csv   위치 표(셀 구성 위치, 셀 id 포함)
  data/processed/ext_labels/ext_cells_v1_meta.json  규칙, 행 처리 집계, 지역별 셀·블록 수, 해시
  data/processed/ext_labels/ext_cells_v1_cens.csv   절단 행의 위치 요약(개정 13. 지점 값이 들어 있어 커밋하지 않는다)

규칙(6B.3)
  행 단위 라벨 집합
    direct         label_def = direct_eos, qc_flag = ok, 연도 ≥ 1990, 0 < alt_cm ≤ 600, right_censored ≠ 1,
                   disturbed ≠ 1, dup_of 빈 칸, 식별자 제외 목록에 없음
    temp           label_def = temp_derived 이고 위와 같은 QC. CALM 망 값을 옮겨 실은 집계 자료 행(원출처가 v3 의
                   CALM_QTP_China 셀)은 뺀다
    unknown_yamal  nsidc_ggd402_yamal 의 label_def = unknown 행. 연도 규칙만 면제(1976–1989), 나머지 QC 는 같다
    dated          label_def = direct_dated 행(서술용, v4 에 넣지 않는다). 연도·범위·검열·교란·중복 규칙은 같고
                   qc_flag 는 ok 또는 date
    unknown_other  그 밖의 label_def = unknown 행(서술용, v4 에 넣지 않는다)
  위치 = (src_id, site_id). 위치 값 = 그 위치의 연 값(다년 값 행은 한 값) 평균, 위치 좌표 = 행 좌표 평균.
  셀 색인 ky = floor(lat/0.009), kx = floor(lon·cos φ/0.009), φ = (ky + 0.5)·0.009°. 위치 좌표로 셀을 정한다.
  셀 값 = 셀 안 위치 값의 평균, 셀 좌표 = 위치 좌표의 평균. 0.5° 블록은 fidelity.add_group_keys 와 같은 식.
  v3 중복: 셀 좌표가 v3 F4_direct 행과 체비쇼프 0.01° 이내(≤)이면 새 셀에서 뺀다(dup_v3 = 1).
  v3 관계: v3 의 F2_gtnp_env, F4_calm_temp 행과 0.01° 이내이면 새 셀로 두고 v3_rel 에 적는다.
  독립성: 다른 지리 macro 의 v3 F4_direct 행까지 최소 대권 거리가 100 km 미만이면 대상에서 뺀다. v3 행의 지리 macro 는
    6B.4 의 정의로 붙인다(CALM_Greenland, Abisko S2, Kapp Linné S1 은 NAtlantic, QTP_CN 은 Tibet).
  대상(target) = direct 이고 macro ∈ {Tibet, NAtlantic, Russia_C, Russia_W, Russia_E, Canada}, dup_v3 = 0, 독립성 통과.

개정 13(검증 지적 반영, 결과 열람 전에 정한 규칙)
  v3 지리 macro: 레나 델타 상자(71.5–73.6°N, 123.3–130.1°E) 안의 v3 행은 region 과 관계없이 Lena 다(6B.4 의
    Russia_C 정의. v3 loc 17557, CALM R8 Tiksi 가 여기에 든다).
  판단 제외 행(ROW_EXCLUDE): 같은 지점의 다른 측정과 방법이 다른 값. Tavvavuoma T1 2005, T4 2006 의 설치 때 값.
  위치 범위 규칙: 1990년 이후 행 가운데 범위(> 600 cm) 밖 값이 있는 위치는 그 위치 전체를 뺀다(남은 연도의
    다년 평균이 아래로 치우치기 때문이다. qtp_fu_temp 의 TM1, TG2, AMD1, TT1, AD1).
  이름 합치기(ALIAS_MERGE): 같은 자료원에서 같은 해, 약 10 m 안의 두 번째 기록(qtp_du_gpr 의 WDL02副)을 원 위치로
    합친다. 위치 값은 연도별 평균의 평균이다(같은 해 두 값은 먼저 평균한다).
  절단 표시: 우측 절단 행은 셀 값에서 빼되(6B.3), 빼지 않았다면 들어갔을 라벨 집합과 위치를 계산해 셀에
    n_cens_removed(뺀 절단 행 수), cens_lb_max·cens_lb_mean(뺀 하한의 최댓값·평균), cens_lb_gt_cell(cens_lb_max 가
    셀 값보다 큼), alt_cm_cens_lb(절단 행을 하한값으로 넣어 다시 계산한 셀 값. 절단 행만 있는 위치는 같은 셀이
    있을 때만 더한다)를 적는다.
  절단 의심(CENS_SUSPECT): 원자료가 절단으로 표기하지 않았으나 한계 값과 같은 값. 셀 값에 넣고 n_cens_suspect 로
    표시한다. cens_affected = cens_lb_gt_cell 또는 n_cens_suspect > 0. L41 (e)는 cens_affected 셀을 뺀다. L41 (d)는 WRAPUP 7.3 (a)9 의 year_max >= 2010 변형이다.
  값 경계: nsidc_ggd353 의 '<'(상한, value_bound=upper), '~'(근사) 행 수와 상한 행을 뺀 셀 값(alt_cm_no_upper).
  라벨 세부 정의(label_subtypes): palmtag2022_pedon 은 단면의 영구동토 상한 깊이(pf_top_profile)이고 Kytalyk 은 같은
    날 동결면보다 깊다(pf_top_below_frozen_al).
  품질 표시(q_flags): 원관측 1건 셀(single_measurement), 판단 목록(LOC_QUALITY: 이상점이 섞인 측선, 값 공유 지점).
  날짜 근거(date_basis): ru_yamal_walker 측선 행의 날짜는 같은 이벤트 relevé 날짜를 빌린 값이다. L41 (b)는 이 행을
    보고서의 조사 시작일로 판정한다(보수적).
  약관 표시(lic_unverified): 'unverified' 또는 'rights reserved' 가 든 셀(build_fidelity_base_v4 의 license_attention 과 같다).
  unknown_yamal 셀의 월은 시추 월이라 obs_months 를 비우고 drill_months 에 적는다.
  v3 F4_direct 까지 대권 거리(v3_f4_km)와 1.1 km 미만 표시(v3_f4_lt1p1km).

실행(ROOT): OMP_NUM_THREADS=1 python3 scripts/1_data_prep/build_ext_cells_v1.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EXT = PROC / "ext_labels"
sys.path.insert(0, str(ROOT / "src"))
from polar.m1_ext import haversine_km  # noqa: E402

CELL_DEG = 0.009
DUP_DEG = 0.01
INDEP_KM = 100.0
TARGET_MACROS = ["Tibet", "NAtlantic", "Russia_C", "Russia_W", "Russia_E", "Canada"]
DIRECT3 = {"probe", "thaw_tube", "frost_tube"}          # L41 (a)
# 식별자 제외 목록(판단 기록). '废' 는 중국어 조사표에서 무효(作废) 표시로 쓰이는 글자다. 같은 해 같은 지점(BLH02)의
# 정상 기록이 따로 있으므로 대상 셀에서 뺀다. 점 자료는 고치지 않는다.
ID_EXCLUDE = {("qtp_du_gpr", "BLH02_废"): "식별자 접미 '废'(무효 표시로 판단, 같은 해 BLH02 기록 있음)"}
# CALM 망 값을 옮겨 실은 집계 자료의 지온 유도 행(원출처는 PANGAEA 972777 이고 v3 CALM_QTP_China 셀로 이미 있다)
CALM_COPY = {"qtp_liux_compile": "CALM network", "qtp_yang_compile": "CALM-ALT"}
LENA_BOX = (71.5, 73.6, 123.3, 130.1)          # 6B.4 의 레나 델타 영역(위도, 경도 범위)
RANGE_MAX = 600.0
# ---- 개정 13 판단 목록(결과 열람 전. 근거는 메타에 옮겨 적는다)
ROW_EXCLUDE = {
    ("swe_tavvavuoma_sannel", "T1", 2005): "설치 때 측정값(2005-09-23, 75 cm). 2010–2012년 표 값(3회 반복, 한계 100 cm)과 다른 측정이고 "
                                            "같은 지점의 이후 하한(>100)·2007년 지온(91 cm 센서 최고 0.21 °C)과 맞지 않는다",
    ("swe_tavvavuoma_sannel", "T4", 2006): "설치 때 측정값(2006-09-04, 115 cm). 표 값 측정 도구의 한계(100 cm)를 넘으므로 다른 방법으로 잰 값이다",
}
ALIAS_MERGE = {
    ("qtp_du_gpr", "WDL02副"): ("WDL02", "같은 해(2018) WDL02 기록과 약 10 m 떨어진 두 번째 기록('副' 는 부(副) 기록으로 판단). "
                                          "WDL02 위치로 합치고 2018년 값은 두 값의 평균으로 한다. 뜻은 원자료로 확인하지 못했다"),
}
CENS_SUSPECT = {
    ("calm_web_subsites", "S2_AB4", 2019): "110–146 cm 로 증가하던 값이 2019–2021년 150 cm 로 같고 2022–2023년은 결측이다. 탐침 한계 값으로 의심",
    ("calm_web_subsites", "S2_AB4", 2020): "같은 근거(2019–2021년 150 cm 연속)",
    ("calm_web_subsites", "S2_AB4", 2021): "같은 근거(2019–2021년 150 cm 연속)",
    ("grl_ilulissat_scheer", "ILU16013T", 2021): "107 cm. 이웃 H2 의 같은 해 값 107 cm 에 'Out of range' 주석이 있다",
    ("grl_ilulissat_scheer", "ERT_CALM_43", 2021): "110 cm. 이웃 ERT_CALM_40–42 의 같은 해 값 110 cm 에 'Out of range' 주석이 있다",
    ("firealt_talucci2025", "firealt_talucci2025_0122", 2019): "CG2-20C. 2019-10-04 방문 5건이 모두 150.0 cm 다. 직전 09-28 방문은 118–168 cm",
    ("swe_tavvavuoma_sannel", "T7", 2011): "반복값(74, 76, 100) 가운데 100 cm 가 측정 한계 값이다",
}
LOC_QUALITY = {
    ("ru_yamal_walker", "Laborovaya2_T18"): "outlier_transect(N=5, 최대 136, 최소 5, 평균 73.5, 표준편차 60.4 cm. 사유 미확인)",
    ("qtp_fu_temp", "QSH-2"): "value_share(QSH-3 과 20년 가운데 12년 값이 같다, 상관 0.92)",
    ("qtp_fu_temp", "QSH-3"): "value_share(QSH-2 와 20년 가운데 12년 값이 같다, 상관 0.92)",
}
SUBTYPE_SRC = {"palmtag2022_pedon": "pf_top_profile"}
SUBTYPE_SUBUNIT = {("palmtag2022_pedon", "Indigirka_Kytalyk"): "pf_top_below_frozen_al"}
# ru_yamal_walker 보고서의 지점별 조사 기간 시작일(ru_yamal_walker_meta.json 의 sampling_periods_report)
WALKER_SURVEY_START = {"Nadym": "2007-08-03", "Laborovaya": "2007-08-13", "VaskinyDachi": "2007-08-21", "Kharasavey": "2008-08-18"}
BORROWED_DATE_NOTE = "측선의 관측일은 원자료에 없다"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def in_lena_box(lat: float, lon: float) -> bool:
    return LENA_BOX[0] <= lat <= LENA_BOX[1] and LENA_BOX[2] <= lon <= LENA_BOX[3]


def geo_macro_v3(region: str, lat: float, lon: float) -> str:
    """v3 행의 지리 macro(6B.4 정의). 독립성 판정에만 쓴다. 레나 델타 상자 안의 러시아 행은 Lena 다(개정 13)."""
    if region in ("CALM_Russia_C", "GTNPenv_RU", "Lena_RU") and in_lena_box(lat, lon):
        return "Lena"
    if region in ("ABoVE_AK", "United States (Alaska)", "GTNPenv_US"):
        return "Alaska"
    if region in ("ABoVE_CA", "Canada", "CALM_Canada"):
        return "Canada"
    if region == "Lena_RU":
        return "Lena"
    if region in ("CALM_Greenland", "CALM_Scandinavia", "CALM_Svalbard", "GTNPenv_SJ"):
        return "NAtlantic"
    if region in ("QTP_CN", "CALM_QTP_China"):
        return "Tibet"
    if region.startswith("CALM_Russia_"):
        return region.replace("CALM_", "")
    if region == "GTNPenv_RU":
        return "Russia_W" if lon < 90 else ("Russia_E" if (lon >= 140 or lon < 0) else "Russia_C")
    return region


def cell_index(lat, lon):
    ky = np.floor(np.asarray(lat, float) / CELL_DEG).astype(int)
    phi = (ky + 0.5) * CELL_DEG
    kx = np.floor(np.asarray(lon, float) * np.cos(np.radians(phi)) / CELL_DEG).astype(int)
    return ky, kx


def block_of(lat, lon):
    return (np.floor(np.asarray(lat, float) / 0.5).astype(int) * 100000 + np.floor(np.asarray(lon, float) / 0.5).astype(int))


def jn(vals):
    return ";".join(sorted({str(v) for v in vals if isinstance(v, str) or (v == v and v is not None)}))


def label_sets(pts, ld, qc, yr_ok, common, calm_copy):
    sets = pd.Series("", index=pts.index, dtype=object)
    sets[(ld == "direct_eos") & (qc == "ok") & yr_ok & common] = "direct"
    sets[(ld == "temp_derived") & (qc == "ok") & yr_ok & common & ~calm_copy] = "temp"
    sets[(pts.src_id == "nsidc_ggd402_yamal") & (ld == "unknown") & qc.isin(["ok", "year_pre1990"]) & common] = "unknown_yamal"
    sets[(ld == "direct_dated") & qc.isin(["ok", "date"]) & yr_ok & common] = "dated"
    sets[(ld == "unknown") & (pts.src_id != "nsidc_ggd402_yamal") & qc.isin(["ok", "date"]) & common] = "unknown_other"
    return sets


def year_key(df):
    """연 값 키. 연도가 없는 다년 값 행은 (year_min, year_max) 로 묶는다."""
    y = df.year.map(lambda v: "" if v != v else str(int(v)))
    m = df.year_min.map(lambda v: "" if v != v else str(int(v))) + "-" + df.year_max.map(lambda v: "" if v != v else str(int(v)))
    return np.where(y != "", y, "m" + m)


def loc_value(df, col="alt_cm"):
    """위치 값 = 연도별 평균의 평균(같은 해 값이 둘 이상이면 먼저 평균한다. 개정 13 이전과 결과가 같다)."""
    if len(df) == 0:
        return np.nan
    return float(df.groupby(year_key(df))[col].mean().mean())


def main():
    t0 = time.time()
    files = sorted(glob.glob(str(EXT / "*_points.csv")))
    pts = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""]) for f in files], ignore_index=True)
    for c in ["lat", "lon", "alt_cm", "year", "month", "year_min", "year_max", "n_obs", "disturbed", "right_censored",
              "coord_prec_deg", "alt_sd_cm"]:
        pts[c] = pd.to_numeric(pts[c], errors="coerce")
    n_in = len(pts)
    src_hash = {Path(f).name: sha(Path(f)) for f in files}

    # ---- 이름 합치기(개정 13): site_id 를 원 위치로 바꾸고 원래 이름은 orig_site_id 에 남긴다
    pts["orig_site_id"] = pts.site_id
    pts["alias_of"] = ""
    for (s_, a_), (to_, _why) in ALIAS_MERGE.items():
        k = (pts.src_id == s_) & (pts.site_id == a_)
        pts.loc[k, "alias_of"] = to_
        pts.loc[k, "site_id"] = to_

    yr = pts.year.fillna(pts.year_min)
    yr_ok = (yr >= 1990) & (pts.year_max.fillna(pts.year).fillna(yr) >= 1990)
    rng_ok = (pts.alt_cm > 0) & (pts.alt_cm <= RANGE_MAX)
    cens = pts.right_censored.fillna(0) == 1
    dist = pts.disturbed.fillna(0) == 1
    dup = pts.dup_of.notna()
    qc = pts.qc_flag.fillna("")
    idex = pd.Series([(s, i) in ID_EXCLUDE for s, i in zip(pts.src_id, pts.orig_site_id)], index=pts.index)
    rowex = pd.Series([(s, i, int(y) if y == y else None) in ROW_EXCLUDE for s, i, y in zip(pts.src_id, pts.orig_site_id, pts.year)],
                      index=pts.index)
    calm_copy = pd.Series([CALM_COPY.get(s) is not None and str(o).strip() == CALM_COPY[s]
                           for s, o in zip(pts.src_id, pts.orig_source)], index=pts.index)
    # 위치 범위 규칙(개정 13): 1990년 이후 행 가운데 range 밖 값이 있는 위치는 전체를 뺀다
    over = yr_ok & (pts.alt_cm > RANGE_MAX) & pts.label_def.isin(["direct_eos", "direct_dated", "temp_derived"])
    over_locs = set(zip(pts.src_id[over], pts.site_id[over]))
    locrange = pd.Series([(s, i) in over_locs for s, i in zip(pts.src_id, pts.site_id)], index=pts.index)
    common = rng_ok & ~cens & ~dist & ~dup & ~idex & ~rowex & ~locrange
    common_nc = rng_ok & ~dist & ~dup & ~idex & ~rowex & ~locrange          # 절단 조건만 뺀 것(절단 표시 계산용)
    ld = pts.label_def
    sets = label_sets(pts, ld, qc, yr_ok, common, calm_copy)
    sets_nc = label_sets(pts, ld, qc, yr_ok, common_nc, calm_copy)
    pts["label_set"] = sets
    pts["label_set_nc"] = sets_nc

    # 행 처리 사유(첫 사유) 집계
    reason = pd.Series("", index=pts.index, dtype=object)
    for cond, why in [(sets != "", "used"), (idex, "id_exclude"), (rowex, "row_exclude_judgment"), (dup, "dup_of"),
                      (locrange & rng_ok, "location_has_range_year"), (~rng_ok, "range"), (cens, "right_censored"),
                      (dist, "disturbed"), (calm_copy & (ld == "temp_derived"), "calm_copy_in_v3"),
                      (~yr_ok & (pts.src_id != "nsidc_ggd402_yamal"), "year_pre1990"), (qc != "ok", "qc_flag")]:
        reason[(reason == "") & cond] = why
    reason[reason == ""] = "other"
    pts["row_reason"] = reason
    disp = (pts.groupby(["src_id", "label_def", "row_reason"]).size().rename("n").reset_index())

    # ---- 행 단위 표시
    pts["cens_suspect"] = [int((s, i, int(y) if y == y else None) in CENS_SUSPECT)
                           for s, i, y in zip(pts.src_id, pts.orig_site_id, pts.year)]
    nt = pts.notes.fillna("")
    pts["vb_upper"] = ((pts.src_id == "nsidc_ggd353_thawtube") & nt.str.contains("value_bound=upper")).astype(int)
    pts["vb_approx"] = ((pts.src_id == "nsidc_ggd353_thawtube") & nt.str.contains(r"approx=1")).astype(int)
    pts["label_subtype"] = [SUBTYPE_SUBUNIT.get((s, u), SUBTYPE_SRC.get(s, "")) for s, u in zip(pts.src_id, pts.subunit.fillna(""))]
    borrowed = (pts.src_id == "ru_yamal_walker") & nt.str.contains(BORROWED_DATE_NOTE)
    pts["date_basis"] = np.where(borrowed, "borrowed_releve_date", np.where(pts.date.notna(), "record", ""))
    d_eff = pts.date.fillna("").copy()
    for i in pts.index[borrowed]:
        ev = re.match(r"^([A-Za-z]+?)\d", pts.site_id[i])
        d_eff[i] = WALKER_SURVEY_START[ev.group(1)]
    pts["date_l41b"] = d_eff

    use = pts[sets != ""].copy()
    d = use.date_l41b.fillna("")
    mm = pd.to_numeric(d.str[5:7], errors="coerce")
    dd = pd.to_numeric(d.str[8:10], errors="coerce")
    use["early_single"] = ((use.eos_basis == "record_date") & ((mm < 8) | ((mm == 8) & (dd < 15)))).astype(int)
    use["rd_no_day"] = ((use.eos_basis == "record_date") & dd.isna()).astype(int)
    use["ds_stmt"] = (use.eos_basis == "dataset_statement").astype(int)
    use["m_direct3"] = use.method.isin(DIRECT3).astype(int)
    lic_l = use.license.fillna("unverified").str.lower()
    use["lic_unverified"] = (lic_l.str.contains("unverified") | lic_l.str.contains("rights reserved")).astype(int)
    use["borrowed_date"] = (use.date_basis == "borrowed_releve_date").astype(int)

    # ---- 위치
    g = use.groupby(["label_set", "src_id", "site_id"], sort=True)
    loc = g.agg(lat=("lat", "mean"), lon=("lon", "mean"), n_val=("alt_cm", "size"),
                macro=("macro", lambda s: jn(s)), subunit=("subunit", lambda s: jn(s)), country=("country", lambda s: jn(s)),
                method=("method", lambda s: jn(s)), eos_basis=("eos_basis", lambda s: jn(s)), value_kind=("value_kind", lambda s: jn(s)),
                y0=("year", "min"), y1=("year", "max"), ymin0=("year_min", "min"), ymax1=("year_max", "max"),
                license=("license", lambda s: jn(s)), early_single=("early_single", "max"), rd_no_day=("rd_no_day", "max"),
                ds_stmt=("ds_stmt", "max"), m_direct3=("m_direct3", "min"), lic_unverified=("lic_unverified", "max"),
                months=("month", lambda s: jn([str(int(v)) for v in s.dropna()])),
                n_obs_total=("n_obs", "sum"), n_cens_suspect=("cens_suspect", "sum"), n_upper_bound=("vb_upper", "sum"),
                n_approx=("vb_approx", "sum"), label_subtype=("label_subtype", lambda s: jn([x for x in s if x])),
                borrowed_date=("borrowed_date", "max"), alias_of=("alias_of", lambda s: jn([x for x in s if x])),
                orig_site_ids=("orig_site_id", lambda s: jn(s))).reset_index()
    assert not loc.macro.str.contains(";").any(), "위치 하나에 macro 둘 이상"
    val = {k: loc_value(v) for k, v in use.groupby(["label_set", "src_id", "site_id"])}
    loc["alt_loc"] = [val[(a, b, c)] for a, b, c in zip(loc.label_set, loc.src_id, loc.site_id)]
    noup = {k: loc_value(v[v.vb_upper == 0]) for k, v in use.groupby(["label_set", "src_id", "site_id"])}
    loc["alt_loc_no_upper"] = [noup[(a, b, c)] for a, b, c in zip(loc.label_set, loc.src_id, loc.site_id)]
    loc["year_first"] = loc[["y0", "ymin0"]].min(axis=1)
    loc["year_last"] = loc[["y1", "ymax1"]].max(axis=1)
    loc["ky"], loc["kx"] = cell_index(loc.lat, loc.lon)
    loc["cell_key"] = loc.ky.astype(str) + "_" + loc.kx.astype(str)
    loc["cell_uid"] = loc.label_set + "|" + loc.cell_key
    loc["loc_quality"] = [LOC_QUALITY.get((s, i), "") for s, i in zip(loc.src_id, loc.site_id)]

    # ---- 절단 행(개정 13): 절단 조건이 없었다면 들어갔을 집합과 위치
    cr = pts[cens & (sets_nc != "")].copy()
    cr["label_set"] = cr.label_set_nc
    loc_key = {(a, b, c): (la, lo, ck) for a, b, c, la, lo, ck in zip(loc.label_set, loc.src_id, loc.site_id, loc.lat, loc.lon, loc.cell_key)}
    crl = []
    for (ls, s_, i_), gg in cr.groupby(["label_set", "src_id", "site_id"]):
        if (ls, s_, i_) in loc_key:
            la, lo, ck = loc_key[(ls, s_, i_)]
            only = 0
        else:
            la, lo = float(gg.lat.mean()), float(gg.lon.mean())
            ky, kx = cell_index([la], [lo])
            ck = f"{int(ky[0])}_{int(kx[0])}"
            only = 1
        crl.append(dict(label_set=ls, src_id=s_, site_id=i_, cell_key=ck, cens_only_loc=only, n_cens=len(gg),
                        lb_max=float(gg.alt_cm.max()), lb_mean=float(gg.alt_cm.mean()),
                        years=jn([str(int(y)) for y in gg.year.dropna()]), macro=jn(gg.macro)))
    crl = pd.DataFrame(crl, columns=["label_set", "src_id", "site_id", "cell_key", "cens_only_loc", "n_cens", "lb_max",
                                     "lb_mean", "years", "macro"])
    # 하한 대입 위치 값(변형 e): 사용 행 + 절단 행(하한값)
    both = pd.concat([use.assign(_c=0), cr.assign(_c=1)], ignore_index=True)
    lbv = {k: loc_value(v) for k, v in both.groupby(["label_set", "src_id", "site_id"])}
    loc["alt_loc_cens_lb"] = [lbv[(a, b, c)] for a, b, c in zip(loc.label_set, loc.src_id, loc.site_id)]
    loc = loc.merge(crl[["label_set", "src_id", "site_id", "n_cens", "lb_max"]].rename(
        columns={"n_cens": "n_cens_removed", "lb_max": "cens_lb_max"}), on=["label_set", "src_id", "site_id"], how="left")
    loc["n_cens_removed"] = loc.n_cens_removed.fillna(0).astype(int)
    cens_only = crl[crl.cens_only_loc == 1].copy()
    cens_only["alt_loc_cens_lb"] = [lbv[(a, b, c)] for a, b, c in zip(cens_only.label_set, cens_only.src_id, cens_only.site_id)]

    # 서로 다른 자료원의 위치가 체비쇼프 0.0005°(약 50 m) 안에 있는 경우(같은 라벨 집합). 진단용 목록
    near_pairs = []
    for ls, sub in loc.groupby("label_set"):
        if sub.src_id.nunique() < 2:
            continue
        tr = cKDTree(sub[["lat", "lon"]].values)
        for i, j in sorted(tr.query_pairs(0.0005, p=np.inf)):
            a, b = sub.iloc[i], sub.iloc[j]
            if a.src_id != b.src_id:
                near_pairs.append(dict(label_set=ls, a=f"{a.src_id}:{a.site_id}", b=f"{b.src_id}:{b.site_id}",
                                       dlat=round(abs(a.lat - b.lat), 6), dlon=round(abs(a.lon - b.lon), 6),
                                       alt_a=round(a.alt_loc, 2), alt_b=round(b.alt_loc, 2),
                                       years_a=f"{a.year_first:.0f}-{a.year_last:.0f}", years_b=f"{b.year_first:.0f}-{b.year_last:.0f}"))

    # ---- 셀
    co_by = {k: v for k, v in cens_only.groupby(["label_set", "cell_key"])}
    cr_by = {k: v for k, v in crl.groupby(["label_set", "cell_key"])}

    def agg_cell(s):
        ls, ck = s.label_set.iloc[0], s.cell_key.iloc[0]
        keys = set(zip(s.src_id, s.site_id))
        vals = use.loc[use.label_set.eq(ls) & use.src_id.isin(s.src_id) & use.site_id.isin(s.site_id)]
        vals = vals[[(a, b) in keys for a, b in zip(vals.src_id, vals.site_id)]]
        alt = s.alt_loc.mean()
        c_all = cr_by.get((ls, ck))
        c_only = co_by.get((ls, ck))
        lb_vals = list(s.alt_loc_cens_lb) + (list(c_only.alt_loc_cens_lb) if c_only is not None else [])
        n_cr = int(c_all.n_cens.sum()) if c_all is not None else 0
        lbmax = float(c_all.lb_max.max()) if c_all is not None else np.nan
        lbmean = float((c_all.lb_mean * c_all.n_cens).sum() / c_all.n_cens.sum()) if c_all is not None else np.nan
        nup = s.alt_loc_no_upper.dropna()
        qf = []
        if int(s.n_obs_total.sum()) == 1 and int(s.n_val.sum()) == 1:
            qf.append("single_measurement")
        qf += [q.split("(")[0] + ":" + b for q, b in zip(s.loc_quality, s.site_id) if q]
        subt = jn([x for x in ";".join(s.label_subtype.fillna("")).split(";") if x])
        months = jn(";".join(s.months.fillna("")).split(";"))
        return pd.Series(dict(
            lat=s.lat.mean(), lon=s.lon.mean(), alt_cm=alt,
            alt_sd_loc=s.alt_loc.std(ddof=1) if len(s) > 1 else np.nan,
            alt_sd_all=vals.alt_cm.std(ddof=1) if len(vals) > 1 else np.nan,
            n_loc=len(s), n_values=int(s.n_val.sum()), n_src=s.src_id.nunique(), sources=jn(s.src_id),
            loc_by_src=";".join(f"{k}:{v}" for k, v in s.src_id.value_counts().sort_index().items()),
            macro=jn(s.macro), subunit=jn(";".join(s.subunit.fillna("")).split(";")), country=jn(";".join(s.country).split(";")),
            methods=jn(";".join(s.method).split(";")), eos_bases=jn(";".join(s.eos_basis.fillna("")).split(";")),
            value_kinds=jn(";".join(s.value_kind.fillna("")).split(";")),
            obs_months="" if ls == "unknown_yamal" else months, drill_months=months if ls == "unknown_yamal" else "",
            year_min=s.year_first.min(), year_max=s.year_last.max(),
            l41a_all_direct3=int(s.m_direct3.min()), l41b_early_single=int(s.early_single.max()), l41b_rd_no_day=int(s.rd_no_day.max()),
            l41c_dataset_statement=int(s.ds_stmt.max()), license=jn(";".join(s.license.fillna("")).split(";")),
            lic_unverified=int(s.lic_unverified.max()),
            n_cens_removed=n_cr, n_cens_loc_only=int(len(c_only)) if c_only is not None else 0,
            cens_lb_max=lbmax, cens_lb_mean=lbmean, cens_lb_gt_cell=int(bool(n_cr) and lbmax > alt),
            alt_cm_cens_lb=float(np.mean(lb_vals)), n_cens_suspect=int(s.n_cens_suspect.sum()),
            n_upper_bound=int(s.n_upper_bound.sum()), n_approx=int(s.n_approx.sum()),
            alt_cm_no_upper=float(nup.mean()) if len(nup) else np.nan,
            label_subtypes=subt, q_flags=";".join(qf), n_obs_total=int(s.n_obs_total.sum()),
            date_basis="borrowed_releve_date" if int(s.borrowed_date.max()) else "",
            aliases=jn([f"{a}<-{x}" for a, o in zip(s.site_id, s.orig_site_ids) for x in o.split(";") if x != a]),
            sites=";".join(f"{a}:{b}" for a, b in zip(s.src_id, s.site_id))))

    cells = (loc.groupby(["label_set", "cell_key", "ky", "kx"], sort=True).apply(agg_cell).reset_index())
    assert not cells.macro.str.contains(";").any(), "셀 하나에 macro 둘 이상"
    cells["cell_uid"] = cells.label_set + "|" + cells.cell_key
    cells["block"] = block_of(cells.lat, cells.lon)
    cells["cens_affected"] = ((cells.cens_lb_gt_cell == 1) | (cells.n_cens_suspect > 0)).astype(int)
    # 절단 행만 있는 위치 가운데 대응하는 셀이 없는 것(변형 e 에서도 셀을 새로 만들지 않는다)
    ck_set = set(zip(cells.label_set, cells.cell_key))
    cens_orphan = cens_only[[(a, b) not in ck_set for a, b in zip(cens_only.label_set, cens_only.cell_key)]]

    # ---- v3 대조
    v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon", "region", "source_id", "alt_cm"], low_memory=False)
    f4 = v3[v3.source_id == "F4_direct"].reset_index(drop=True)
    nf4 = v3[v3.source_id != "F4_direct"].reset_index(drop=True)
    t4 = cKDTree(f4[["lat", "lon"]].values)
    dd4, jj4 = t4.query(cells[["lat", "lon"]].values, p=np.inf)
    cells["v3_f4_cheb_deg"] = np.round(dd4, 6)
    cells["v3_f4_near_loc"] = f4.loc_id.values[jj4]
    cells["dup_v3"] = (dd4 <= DUP_DEG).astype(int)
    tn = cKDTree(nf4[["lat", "lon"]].values)
    ddn, jjn = tn.query(cells[["lat", "lon"]].values, p=np.inf)
    cells["v3_rel"] = [f"{nf4.loc_id[j]}:{nf4.source_id[j]}:{nf4.region[j]}:{dd:.4f}" if dd <= DUP_DEG else ""
                       for dd, j in zip(ddn, jjn)]
    f4["geo_macro"] = [geo_macro_v3(r, a, b) for r, a, b in zip(f4.region, f4.lat, f4.lon)]
    dmin, dmac, dloc, dkm, dkm_loc = [], [], [], [], []
    for r in cells.itertuples(index=False):
        dk_all = haversine_km(r.lat, r.lon, f4.lat.values, f4.lon.values)
        k0 = int(np.argmin(dk_all))
        dkm.append(float(dk_all[k0])); dkm_loc.append(int(f4.loc_id.values[k0]))
        o = (f4.geo_macro != r.macro).values
        dk = dk_all[o]
        k = int(np.argmin(dk))
        dmin.append(float(dk[k])); dmac.append(f4.geo_macro.values[o][k]); dloc.append(int(f4.loc_id.values[o][k]))
    cells["dist_other_macro_km"] = np.round(dmin, 1)
    cells["other_macro_nearest"] = [f"{m}:{l}" for m, l in zip(dmac, dloc)]
    cells["indep_ok"] = (cells.dist_other_macro_km >= INDEP_KM).astype(int)
    cells["v3_f4_km"] = np.round(dkm, 3)
    cells["v3_f4_km_loc"] = dkm_loc
    cells["v3_f4_lt1p1km"] = (cells.v3_f4_km < 1.1).astype(int)
    # 같은 셀에 직접 라벨 셀이 있는 보조 셀 표시
    dkeys = set(cells.loc[cells.label_set == "direct", "cell_key"])
    cells["same_cell_direct"] = [int(k in dkeys) if ls != "direct" else 0 for k, ls in zip(cells.cell_key, cells.label_set)]

    in_scope = cells.macro.isin(TARGET_MACROS)
    cells["in_v4"] = (((cells.label_set == "direct") & in_scope) | ((cells.label_set == "temp") & (cells.macro == "Tibet"))
                      | (cells.label_set == "unknown_yamal")).astype(int) * (1 - cells.dup_v3)
    cells["target"] = ((cells.label_set == "direct") & in_scope & (cells.dup_v3 == 0) & (cells.indep_ok == 1)).astype(int)
    why = []
    for r in cells.itertuples(index=False):
        w = []
        if r.label_set != "direct":
            w.append(f"label_set={r.label_set}")
        if r.macro not in TARGET_MACROS:
            w.append(f"macro={r.macro}")
        if r.dup_v3:
            w.append(f"dup_v3:{r.v3_f4_near_loc}")
        if not r.indep_ok:
            w.append(f"indep<{INDEP_KM:.0f}km:{r.other_macro_nearest}")
        why.append(";".join(w))
    cells["excl_reason"] = why

    cols = ["cell_uid", "label_set", "cell_key", "ky", "kx", "lat", "lon", "block", "macro", "subunit", "country", "alt_cm",
            "alt_sd_loc", "alt_sd_all", "n_loc", "n_values", "n_src", "sources", "loc_by_src", "methods", "eos_bases", "value_kinds",
            "obs_months", "drill_months", "year_min", "year_max", "l41a_all_direct3", "l41b_early_single", "l41b_rd_no_day",
            "l41c_dataset_statement", "n_cens_removed", "n_cens_loc_only", "cens_lb_max", "cens_lb_mean", "cens_lb_gt_cell",
            "alt_cm_cens_lb", "n_cens_suspect", "cens_affected", "n_upper_bound", "n_approx", "alt_cm_no_upper",
            "label_subtypes", "q_flags", "n_obs_total", "date_basis", "aliases",
            "v3_f4_cheb_deg", "v3_f4_near_loc", "dup_v3", "v3_rel", "v3_f4_km", "v3_f4_km_loc", "v3_f4_lt1p1km",
            "dist_other_macro_km", "other_macro_nearest", "indep_ok",
            "same_cell_direct", "in_v4", "target", "excl_reason", "license", "lic_unverified", "sites"]
    cells = cells[cols].sort_values(["label_set", "macro", "ky", "kx"]).reset_index(drop=True)
    cells.to_csv(EXT / "ext_cells_v1.csv", index=False)
    loc.drop(columns=["y0", "y1", "ymin0", "ymax1"]).to_csv(EXT / "ext_cells_v1_locs.csv", index=False)
    crl.to_csv(EXT / "ext_cells_v1_cens.csv", index=False)

    # ---- 요약
    def summ(sub):
        return dict(n_cells=int(len(sub)), n_blocks=int(sub.block.nunique()), n_loc=int(sub.n_loc.sum()),
                    alt_mean=round(float(sub.alt_cm.mean()), 1) if len(sub) else None)
    by = {}
    for (ls, mac), sub in cells.groupby(["label_set", "macro"]):
        by[f"{ls}|{mac}"] = dict(all=summ(sub), not_dup=summ(sub[sub.dup_v3 == 0]), in_v4=summ(sub[sub.in_v4 == 1]),
                                 target=summ(sub[sub.target == 1]),
                                 by_source={s: int(sub.loc_by_src.str.contains(f"{s}:").sum()) for s in sorted(set(";".join(sub.sources).split(";")))})
    tg = cells[cells.target == 1]
    cens_summary = {}
    for mac, sub in tg.groupby("macro"):
        cens_summary[mac] = dict(
            n_target=int(len(sub)), n_cells_with_cens_removed=int((sub.n_cens_removed > 0).sum()),
            n_cens_lb_gt_cell=int(sub.cens_lb_gt_cell.sum()),
            n_cens_lb_mean_gt_cell=int((sub.cens_lb_mean > sub.alt_cm).sum()),
            n_cens_suspect_cells=int((sub.n_cens_suspect > 0).sum()), n_cens_affected=int(sub.cens_affected.sum()),
            alt_mean=round(float(sub.alt_cm.mean()), 2), alt_mean_cens_lb=round(float(sub.alt_cm_cens_lb.mean()), 2),
            cells_cens_lb_gt_cell=[dict(cell=r.cell_uid, alt=round(r.alt_cm, 2), lb_max=round(r.cens_lb_max, 1),
                                        lb_mean=round(r.cens_lb_mean, 1), alt_cens_lb=round(r.alt_cm_cens_lb, 2), sources=r.sources)
                                   for r in sub[sub.cens_lb_gt_cell == 1].itertuples()],
            cells_cens_suspect=[dict(cell=r.cell_uid, alt=round(r.alt_cm, 2), n=int(r.n_cens_suspect), sources=r.sources)
                                for r in sub[sub.n_cens_suspect > 0].itertuples()])
    vb = tg[tg.n_upper_bound > 0]
    vb_check = dict(rule="nsidc_ggd353 의 '<' 행(value_bound=upper, 최소 침하로 계산한 상한)을 뺀 셀 값과 넣은 셀 값의 차",
                    n_target_cells_with_upper=int(len(vb)), n_upper_rows_in_target=int(vb.n_upper_bound.sum()),
                    n_approx_rows_in_target=int(tg.n_approx.sum()),
                    diff_cm=[dict(cell=r.cell_uid, alt=round(r.alt_cm, 2), alt_no_upper=(None if r.alt_cm_no_upper != r.alt_cm_no_upper
                                                                                         else round(r.alt_cm_no_upper, 2)))
                             for r in vb.itertuples()],
                    mean_abs_diff_cm=(round(float((vb.alt_cm - vb.alt_cm_no_upper).abs().mean()), 2) if len(vb) else None),
                    max_abs_diff_cm=(round(float((vb.alt_cm - vb.alt_cm_no_upper).abs().max()), 2) if len(vb) else None),
                    n_cells_all_upper=int(vb.alt_cm_no_upper.isna().sum()))
    borderline = cells[(cells.v3_f4_cheb_deg > 0.0095) & (cells.v3_f4_cheb_deg <= 0.0105)][["cell_uid", "v3_f4_cheb_deg", "v3_f4_near_loc"]]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = ""
    meta = dict(
        stage="LGD 6B.6 단계 2(셀 집계). 개정 13 규칙 반영", created=time.strftime("%Y-%m-%d %H:%M"), git_head=head,
        script="scripts/1_data_prep/build_ext_cells_v1.py", script_sha256=sha(Path(__file__)),
        plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.3, 6B.4, 개정 13",
        inputs=dict(points=src_hash, v3=dict(file="data/processed/fidelity_base_v3.csv", sha256=sha(PROC / "fidelity_base_v3.csv"))),
        rules=dict(cell="ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009 deg. 위치 좌표로 셀 배정",
                   location="(src_id, site_id). 위치 값 = 연도별 평균의 평균, 위치 좌표 = 행 좌표 평균",
                   cell_value="셀 안 위치 값의 평균. 셀 좌표 = 위치 좌표 평균",
                   dup_v3=f"셀 좌표와 v3 F4_direct 행의 체비쇼프 거리 <= {DUP_DEG} deg 이면 새 셀에서 뺀다",
                   v3_rel="v3 F2_gtnp_env·F4_calm_temp 행과 0.01 deg 이내면 새 셀로 두고 v3_rel 에 적는다",
                   independence=f"다른 지리 macro 의 v3 F4_direct 행까지 대권 거리 < {INDEP_KM} km 이면 대상 제외. "
                                "v3 지리 macro: CALM_Greenland·CALM_Scandinavia(Abisko S2)·CALM_Svalbard(Kapp Linné S1) = NAtlantic, "
                                f"QTP_CN = Tibet, 레나 델타 상자 {LENA_BOX} 안의 러시아 행 = Lena(개정 13)",
                   label_sets={"direct": "direct_eos, qc ok, 연도>=1990, 0<alt<=600, 비검열, 비교란, dup_of 없음, 식별자·판단 제외 목록 밖, "
                                         "범위 밖 연도가 있는 위치 제외",
                               "temp": "temp_derived, 같은 QC. CALM 망 값을 옮겨 실은 집계 자료 행 제외",
                               "unknown_yamal": "nsidc_ggd402_yamal unknown. 연도 규칙만 면제",
                               "dated": "direct_dated(서술용)", "unknown_other": "그 밖의 unknown(서술용)"},
                   id_exclude={f"{k[0]}:{k[1]}": v for k, v in ID_EXCLUDE.items()},
                   row_exclude={f"{k[0]}:{k[1]}:{k[2]}": v for k, v in ROW_EXCLUDE.items()},
                   alias_merge={f"{k[0]}:{k[1]}": dict(to=v[0], why=v[1]) for k, v in ALIAS_MERGE.items()},
                   location_range="1990년 이후 행 가운데 alt_cm > 600 인 연도가 있는 위치는 전체를 뺀다(개정 13)",
                   cens_suspect={f"{k[0]}:{k[1]}:{k[2]}": v for k, v in CENS_SUSPECT.items()},
                   cens_suspect_note="원자료 제공자 확인 전의 판단이다. 셀 값에 넣고 n_cens_suspect 로 표시한다",
                   censoring=("우측 절단 행은 셀 값에서 뺀다(6B.3). 절단 조건만 뺀 라벨 집합으로 그 행의 위치와 셀을 정해 "
                              "n_cens_removed, cens_lb_max, cens_lb_mean, cens_lb_gt_cell(cens_lb_max > alt_cm), alt_cm_cens_lb(하한 대입 셀 값)를 적는다. "
                              "절단 행만 있는 위치는 대응하는 셀이 있을 때만 alt_cm_cens_lb 에 더하고 셀을 새로 만들지 않는다. "
                              "cens_affected = cens_lb_gt_cell 또는 n_cens_suspect > 0"),
                   loc_quality={f"{k[0]}:{k[1]}": v for k, v in LOC_QUALITY.items()},
                   label_subtype=dict(src=SUBTYPE_SRC, subunit={f"{k[0]}:{k[1]}": v for k, v in SUBTYPE_SUBUNIT.items()},
                                      note="palmtag2022_pedon 의 라벨은 'Depth Top Permafrost'(단면 판정의 영구동토 상한)이다. Kytalyk 은 "
                                           "같은 날 동결면(Depth frozen AL)보다 깊다(비교 가능 17행 가운데 15행). 다른 조사지는 비교할 열이 없다"),
                   date_basis=("ru_yamal_walker 측선 행은 같은 이벤트 relevé 날짜를 빌린 값이다(borrowed_releve_date). L41 (b) 판정에는 "
                               f"보고서의 조사 시작일을 쓴다: {WALKER_SURVEY_START}"),
                   q_flags="single_measurement = 셀의 원관측이 1건(n_values 1, n_obs 합 1). 그 밖은 loc_quality 판단 목록",
                   lic_unverified="license 에 'unverified' 또는 'rights reserved' 가 있으면 1",
                   drill_months="unknown_yamal(GGD402) 셀의 월은 시추 월이다. obs_months 는 비운다",
                   v3_f4_km="셀 좌표에서 가장 가까운 v3 F4_direct 행까지 대권 거리(km). v3_f4_lt1p1km = 1.1 km 미만",
                   l41="a = 셀의 모든 위치가 probe·thaw_tube·frost_tube, b = 셀에 8월 15일 이전 record_date 단일 방문 행이 있음"
                       "(빌린 날짜는 조사 시작일로 판정), c = 셀에 dataset_statement 행이 있음, d = year_max >= 2010 셀만(WRAPUP (a)9), "
                       "e = cens_affected 셀 제외, f = alt_cm_cens_lb 를 셀 값으로 씀(개정 13)"),
        n_points_in=int(n_in), row_disposition=disp.to_dict(orient="records"),
        n_rows_dup_of=int(dup.sum()),
        n_locations={k: int(v) for k, v in loc.label_set.value_counts().items()},
        location_range_excluded=sorted(f"{a}:{b}" for a, b in over_locs),
        cens_rows_by_set={k: int(v) for k, v in cr.label_set.value_counts().items()},
        cens_only_locations_without_cell=cens_orphan[["label_set", "src_id", "site_id", "cell_key", "n_cens", "lb_max", "macro"]].to_dict(orient="records"),
        censoring_target=cens_summary, value_bound_check=vb_check,
        cells=by, near_pairs_cross_source=near_pairs, dup_borderline=borderline.to_dict(orient="records"),
        elapsed_s=round(time.time() - t0, 1))
    (EXT / "ext_cells_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[points] {n_in} 행 · 사용 {len(use)} 행 · 위치 {len(loc)} · 셀 {len(cells)} · 절단 행 {len(cr)}")
    print(cells.groupby(["label_set", "macro"]).agg(cells=("cell_uid", "size"), blocks=("block", "nunique"),
                                                   dup=("dup_v3", "sum"), in_v4=("in_v4", "sum"), target=("target", "sum"),
                                                   alt=("alt_cm", "mean")).round(1).to_string())
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("cells_")} for k, v in cens_summary.items()}, ensure_ascii=False))
    print(f"[near pairs] {len(near_pairs)} · [borderline] {len(borderline)} · {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
