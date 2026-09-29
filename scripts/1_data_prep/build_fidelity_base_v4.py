"""LGD v4 입력 표 조립. 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.3, 6B.6 단계 3.

입력(읽기만)
  data/processed/fidelity_base_v3.csv, e5_soil_tdd_v3.csv, fidelity_base_v3_meta.json
  data/processed/ext_labels/ext_cells_v1.csv(build_ext_cells_v1.py), ext_cells_v1_cov.csv(ext_cells_covariates_v1.py)
  data/raw/calm/PANGAEA_972777_CALM_ALT_NH.tab(v3 의 'Canada', 'United States (Alaska)' 행의 관측법 대조에만 쓴다)
산출
  data/processed/fidelity_base_v4.csv         v3 17,572행(region 을 바꾼 CALM_Greenland 3행 외 불변) + 새 셀. 열 45개(v3 와 같은 순서)
  data/processed/e5_soil_tdd_v4.csv           v3 토양 도일 표 17,580행 불변 + 새 셀(loc_id 키)
  data/processed/fidelity_base_v4_labels.csv  라벨 부가 표(loc_id 당 1행)
  data/processed/fidelity_base_v4_meta.json   자료원별·지역별 수, 공변량 완결 비율, 해시 대조

새 행 규약
  loc_id = 20000 + 일련번호(라벨 집합 direct, temp, unknown_yamal 순, 그 안에서 macro, ky, kx 순)
  region: macro_region(fidelity.MACRO_REGION.get(r, r))의 항등 대응을 쓴다. NAtlantic, Russia_C, Russia_W, Russia_E, Canada 는
    그 이름을 그대로 region 으로 쓴다(macro 가 같은 이름). Tibet 은 v3 의 QTP_CN, CALM_QTP_China(macro 'Tibet')와 섞이지 않도록
    'Tibet_LGD' 를 쓴다(macro 'Tibet_LGD'). v3 의 CALM_Greenland 3행은 region 을 'NAtlantic' 으로 바꾼다(6B.3).
  source_id: direct → F4_ext_direct(4), temp → F3_ext_temp(3), unknown_yamal → F2_ext_unknown(2). load_base 의 기본 필터
    (F4_direct)로 읽으면 v3 의 F4_direct 행만 나온다. 실행 표를 만드는 래퍼가 대상 행의 source_id 를 F4_direct 로 바꾼다.
  spatial_support_m = 1000(약 1 km 셀), sigma_prior_cm = 셀 안 연 값 표준편차를 3–40 으로 자른 값(없으면 12, E3 와 같다),
  right_censored = 0(검열 행은 셀 값에서 뺐다), InSAR·PolSAR 결측, polsar_valid 0, insar_miss 1(E3 와 같다).
개정 13(검증 지적 반영): 라벨 부가 표에 절단 표시(n_cens_removed, cens_lb_max, cens_lb_gt_cell, alt_cm_cens_lb,
  n_cens_suspect, cens_affected), 값 경계(n_upper_bound, alt_cm_no_upper), 라벨 세부 정의(label_subtypes), 품질 표시
  (q_flags), 날짜 근거(date_basis), 시추 월(drill_months), v3 F4_direct 까지 km 거리(v3_f4_km, v3_f4_lt1p1km),
  ERA5-Land 사용 격자의 월 적설 최솟값과 빙하 격자 표지(e5_grid_sd_min_m, e5_glacier_grid; v3 행 포함),
  격자 상자 DEM 평균과 고도 차(e5_grid_dem_mean_m, elev_minus_e5grid_m), DEM 중심 평탄 표지(dem_center_flat; v3 행 포함)를
  더한다. v3 loc 17557(레나 델타 상자 안)의 lgd_role 은 Russia_C 대상 제외로 적는다. 메타에 새 지역별 고도 범위와
  공변량별 원천 분포(1–99 백분위) 밖 비율, 개정된 래퍼 설계(원문 줄 복사, loc_id·source_id·lgd_role 선택, L42 규칙)를 적는다.
실행(ROOT): OMP_NUM_THREADS=1 python3 scripts/1_data_prep/build_fidelity_base_v4.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EXT = PROC / "ext_labels"
TAB = ROOT / "data" / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import macro_region, COVARIATE_CORE, SHARED_CORE, TERRAIN, CLIMATE, SOIL  # noqa: E402

LOC0 = 20000
REGION_OF = {"Tibet": "Tibet_LGD", "NAtlantic": "NAtlantic", "Russia_C": "Russia_C", "Russia_W": "Russia_W",
             "Russia_E": "Russia_E", "Canada": "Canada"}
SOURCE_OF = {"direct": ("F4_ext_direct", 4), "temp": ("F3_ext_temp", 3), "unknown_yamal": ("F2_ext_unknown", 2)}
SET_ORDER = {"direct": 0, "temp": 1, "unknown_yamal": 2}
LABEL_DEF_OF = {"direct": "direct_eos", "temp": "temp_derived", "unknown_yamal": "unknown"}
E5_PERIOD, CCI_PERIOD = (2015, 2020), (1997, 2021)
LENA_BOX = (71.5, 73.6, 123.3, 130.1)
GLACIER_SD_M = 1.0
EXCLUDED_V3_TARGET = {17557: "v3_excluded_lena_delta_box(Russia_C 대상 제외, 개정 13: 6B.4 지리 정의)"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_df(df: pd.DataFrame, cols=None) -> str:
    d = df if cols is None else df[cols]
    return hashlib.md5(pd.util.hash_pandas_object(d.reset_index(drop=True), index=False).values).hexdigest()[:12]


def append_rows_text(src: Path, dst: Path, rows: pd.DataFrame, region_fix=None):
    """src CSV 의 원문 줄을 그대로 쓰고(region 열 값만 region_fix 로 바꿈) rows 를 이어 쓴다."""
    lines = src.read_text(encoding="utf-8").splitlines()
    hdr = lines[0].split(",")
    assert hdr == list(rows.columns), "열 순서가 다르다"
    assert not any('"' in l for l in lines), "따옴표 칸이 있다"
    out, n_fix = [lines[0]], 0
    ir = hdr.index("region") if region_fix else None
    for l in lines[1:]:
        if region_fix:
            f = l.split(",")
            if f[ir] in region_fix:
                f[ir] = region_fix[f[ir]]; l = ",".join(f); n_fix += 1
        out.append(l)
    body = rows.to_csv(index=False, header=False, lineterminator="\n")
    dst.write_text("\n".join(out) + "\n" + body, encoding="utf-8")
    return n_fix


def gap(y0, y1, per):
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return np.nan
    if y1 < per[0]:
        return float(per[0] - y1)
    if y0 > per[1]:
        return float(y0 - per[1])
    return 0.0


def pangaea_methods():
    """PANGAEA 972777 이벤트 헤더의 Method(build_fidelity_base_v3.py 와 같은 파싱·분류)."""
    lines = TAB.read_text(encoding="utf-8", errors="replace").split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.startswith("Event(s):"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("Parameter(s):"))
    rows = []
    for l in lines[i0:i1]:
        l = l.replace("Event(s):\t", "").strip()
        parts = l.split(" * ")
        d = dict(event=parts[0].split(" ")[0])
        for p in parts[1:]:
            k, _, v = p.partition(":")
            if k in ("LATITUDE", "LONGITUDE"):
                d[k.lower()] = float(v)
            elif k == "COMMENT":
                d["comment"] = v.strip()
        m = re.search(r"Method:\s*(.*)$", d.get("comment", ""))
        s = m.group(1).strip() if m else None
        if not isinstance(s, str):
            cls = "unknown"
        else:
            t = s.lower()
            cls = ("probe" if ("mechanical probing" in t and "temperature" not in t) else
                   "thaw_tube" if "thaw tube" in t else "temperature" if "temperature" in t else "other")
        d["method_class"] = cls
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    v3_path, s3_path = PROC / "fidelity_base_v3.csv", PROC / "e5_soil_tdd_v3.csv"
    v3_sha0, s3_sha0 = sha(v3_path), sha(s3_path)
    v3 = pd.read_csv(v3_path, low_memory=False)
    s3 = pd.read_csv(s3_path)
    cols = list(v3.columns)
    assert len(cols) == 45 and len(v3) == 17572
    cells = pd.read_csv(EXT / "ext_cells_v1.csv")
    cov = pd.read_csv(EXT / "ext_cells_v1_cov.csv")
    new = cells[cells.in_v4 == 1].copy()
    new = new.merge(cov.rename(columns={"id": "cell_uid"}).drop(columns=["lat", "lon", "macro"]), on="cell_uid", how="left")
    assert len(new) == int((cells.in_v4 == 1).sum()) and new.cell_uid.is_unique
    new["_o"] = new.label_set.map(SET_ORDER)
    new = new.sort_values(["_o", "macro", "ky", "kx"]).reset_index(drop=True)
    new["loc_id"] = LOC0 + np.arange(len(new))
    assert v3.loc_id.max() < LOC0

    # ---------------------------------------------------------------- v4 새 행(45열)
    nr = pd.DataFrame(index=new.index)
    nr["loc_id"] = new.loc_id
    nr["lat"], nr["lon"] = new.lat.round(6), new.lon.round(6)
    nr["region"] = new.macro.map(REGION_OF)
    assert nr.region.notna().all()
    nr["block"] = (np.floor(nr.lat / 0.5).astype(int) * 100000 + np.floor(nr.lon / 0.5).astype(int))
    nr["alt_cm"] = new.alt_cm
    nr["source_id"] = new.label_set.map(lambda s: SOURCE_OF[s][0])
    nr["fidelity_level"] = new.label_set.map(lambda s: SOURCE_OF[s][1])
    nr["spatial_support_m"] = 1000.0
    nr["sigma_prior_cm"] = pd.to_numeric(new.alt_sd_all, errors="coerce").fillna(12.0).clip(3, 40)
    nr["right_censored"] = 0
    for c in cols:
        if c in nr.columns:
            continue
        if c in new.columns:
            nr[c] = new[c].values
        elif c in ("insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n", "polsar_alt", "polsar_std"):
            nr[c] = np.nan
        elif c == "polsar_valid":
            nr[c] = 0.0
        elif c == "insar_miss":
            nr[c] = 1
        else:
            raise KeyError(c)
    nr = nr[cols]

    # ---------------------------------------------------------------- v3 행(region 3행만 변경)
    v4a = v3.copy()
    gl = v4a.region == "CALM_Greenland"
    assert int(gl.sum()) == 3
    v4a.loc[gl, "region"] = "NAtlantic"
    v4 = pd.concat([v4a, nr], ignore_index=True)
    assert v4.loc_id.is_unique and list(v4.columns) == cols
    out_v4 = PROC / "fidelity_base_v4.csv"
    # v3 행은 원문 줄을 그대로 옮긴다(region 3칸만 바꾼다). 부동소수 문자열을 다시 쓰면 파서에 따라 1 ulp 차가 생기기 때문이다.
    append_rows_text(v3_path, out_v4, nr, region_fix={"CALM_Greenland": "NAtlantic"})

    # 해시 대조(저장 뒤 다시 읽어서)
    v4r = pd.read_csv(out_v4, low_memory=False)
    head = v4r.iloc[:len(v3)].reset_index(drop=True)
    no_region = [c for c in cols if c != "region"]
    chk = dict(
        v3_rows_hash_44cols_v3=hash_df(v3, no_region), v3_rows_hash_44cols_v4=hash_df(head, no_region),
        v3_rows_hash_45cols_v3=hash_df(v3), v3_rows_hash_45cols_v4=hash_df(head),
        region_changed=head.loc[head.region != v3.region, ["loc_id"]].assign(
            region_v3=v3.region[head.region != v3.region].values, region_v4=head.region[head.region != v3.region].values).to_dict(orient="records"),
        v4_full_hash=hash_df(v4r), new_rows_hash=hash_df(v4r.iloc[len(v3):]))
    chk["v3_rows_unchanged_except_region"] = chk["v3_rows_hash_44cols_v3"] == chk["v3_rows_hash_44cols_v4"]
    l3 = v3_path.read_text(encoding="utf-8").splitlines(); l4 = out_v4.read_text(encoding="utf-8").splitlines()
    chk["text_lines_differing_in_v3_part"] = int(sum(a != b for a, b in zip(l3, l4[:len(l3)])))
    assert chk["v3_rows_unchanged_except_region"], "v3 행의 region 외 열이 바뀌었다"
    assert len(chk["region_changed"]) == 3
    # v3 에서 region 까지 복원하면 45열 해시가 v3 와 같아야 한다
    rest = head.copy(); rest.loc[rest.loc_id.isin([r["loc_id"] for r in chk["region_changed"]]), "region"] = "CALM_Greenland"
    chk["v3_rows_hash_45cols_after_region_restore"] = hash_df(rest)
    chk["v3_restored_equals_v3"] = chk["v3_rows_hash_45cols_after_region_restore"] == chk["v3_rows_hash_45cols_v3"]
    assert chk["v3_restored_equals_v3"]
    chk["v3_meta_full_hash"] = json.loads((PROC / "fidelity_base_v3_meta.json").read_text()).get("v3_full_hash")

    # ---------------------------------------------------------------- 토양 도일 v4
    sn = pd.DataFrame(dict(loc_id=new.loc_id, lat=nr.lat, lon=nr.lon, src="lgd_ext_v1",
                           e5_tdd_soil=new.e5_tdd_soil, e5_fdd_soil=new.e5_fdd_soil, e5_sqrt_tdd_soil=new.e5_sqrt_tdd_soil,
                           e5_stl1_twarm=new.e5_stl1_twarm, e5_stl1_tcold=new.e5_stl1_tcold, soil_fallback_deg=new.soil_fallback_deg))
    zero = (sn.e5_tdd_soil <= 0).fillna(False)
    sn.loc[zero, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan          # v3 규칙과 같다
    s4 = pd.concat([s3, sn[list(s3.columns)]], ignore_index=True)
    out_s4 = PROC / "e5_soil_tdd_v4.csv"
    append_rows_text(s3_path, out_s4, sn[list(s3.columns)])
    s4r = pd.read_csv(out_s4)
    chk["soil_v3_rows_hash_v3"] = hash_df(s3)
    chk["soil_v3_rows_hash_v4"] = hash_df(s4r.iloc[:len(s3)])
    chk["soil_v3_rows_unchanged"] = chk["soil_v3_rows_hash_v3"] == chk["soil_v3_rows_hash_v4"]
    m3 = s3_path.read_text(encoding="utf-8").splitlines(); m4 = out_s4.read_text(encoding="utf-8").splitlines()
    chk["soil_text_lines_differing_in_v3_part"] = int(sum(a != b for a, b in zip(m3, m4[:len(m3)])))
    assert chk["soil_v3_rows_unchanged"]
    chk["soil_new_tdd_le0_set_nan"] = int(zero.sum())
    chk["soil_new_tdd_nan_total"] = int(sn.e5_sqrt_tdd_soil.isna().sum())      # 개정 13: cov 단계에서 이미 결측으로 둔 행 포함

    # ---------------------------------------------------------------- 라벨 부가 표
    mac4 = macro_region(v4r)
    v3m = json.loads((PROC / "fidelity_base_v3_meta.json").read_text())
    pc = pd.DataFrame(v3m["obs_method"]["per_cell"]).set_index("loc_id")
    ev = pangaea_methods()
    lab_v3 = pd.DataFrame(dict(loc_id=v3.loc_id, part="v3", region_v4=head.region, macro_v4=mac4[:len(v3)], source_id=v3.source_id))
    meth, basis = [], []
    ev_ll = ev.dropna(subset=["latitude", "longitude"])
    for r in v3.itertuples(index=False):
        if r.loc_id in pc.index:
            meth.append(pc.loc[r.loc_id, "obs_method"]); basis.append(f"v3_meta:{pc.loc[r.loc_id, 'pangaea_event']}")
        elif r.region in ("Canada", "United States (Alaska)"):
            dd = np.maximum(np.abs(ev_ll.latitude.values - r.lat), np.abs(ev_ll.longitude.values - r.lon))
            k = dd <= 0.005
            cls = sorted(set(ev_ll.method_class.values[k]))
            meth.append(cls[0] if len(cls) == 1 else ("ambiguous:" + "|".join(cls) if cls else "no_event_match"))
            basis.append("pangaea_event_coord:" + "|".join(ev_ll.event.values[k]) if k.any() else "pangaea_event_coord:none")
        else:
            meth.append(""); basis.append("")
    lab_v3["method"] = meth
    lab_v3["method_basis"] = basis
    role = []
    for r in lab_v3.itertuples(index=False):
        if r.source_id != "F4_direct":
            role.append("v3_non_f4")
        elif r.loc_id in (17520, 17569):
            role.append("v3_parent_average_of_subsites(NAtlantic 대상 제외)")
        elif r.loc_id in EXCLUDED_V3_TARGET:
            role.append(EXCLUDED_V3_TARGET[r.loc_id])
        elif r.region_v4 == "QTP_CN":
            role.append("v3_f4_temp_derived(Tibet_LGD 대상 아님)")
        elif r.macro_v4 in ("NAtlantic", "Russia_C", "Russia_W", "Russia_E", "Canada"):
            role.append(f"v3_member_of_{r.macro_v4}")
        else:
            role.append("v3_f4_other")
    lab_v3["lgd_role"] = role
    # v3 행의 ERA5-Land 사용 격자(ext_cells_covariates_v1.py --input v3all --stage e5). 다시 계산한 e5_tdd 로 격자를 확인한다
    g3 = pd.read_csv(EXT / "cov_parts" / "v3all_e5.csv", dtype={"id": str})
    g3["loc_id"] = g3.id.astype(int)
    g3 = g3.set_index("loc_id").reindex(v3.loc_id)
    lab_v3["e5_grid_lat"] = g3.e5_grid_lat.values
    lab_v3["e5_grid_lon"] = g3.e5_grid_lon.values
    lab_v3["e5_grid_sd_min_m"] = g3.e5_grid_sd_min_m.values
    lab_v3["e5_glacier_grid"] = (lab_v3.e5_grid_sd_min_m >= GLACIER_SD_M).astype(int)
    tdd_ok = np.isclose(g3.e5_tdd.values, v3.e5_tdd.values, rtol=0, atol=1e-6) | (np.isnan(g3.e5_tdd.values) & np.isnan(v3.e5_tdd.values))
    lab_v3["e5_grid_check"] = np.where(tdd_ok, "tdd_match", "tdd_mismatch")
    chk_grid_v3 = dict(n=int(len(v3)), n_tdd_match=int(tdd_ok.sum()),
                       mismatch_by_region=v3.region[~tdd_ok].value_counts().to_dict(),
                       glacier_grid_f4_direct=v3.loc[(lab_v3.e5_glacier_grid.values == 1) & (v3.source_id == "F4_direct").values,
                                                     ["loc_id", "region"]].to_dict(orient="records"))

    lab_new = pd.DataFrame(dict(
        loc_id=new.loc_id, part="new", region_v4=nr.region, macro_v4=mac4[len(v3):], source_id=nr.source_id,
        method=new.methods, method_basis="ext_labels 점 자료 method 열", lgd_role=np.where(
            new.label_set == "direct", np.where(new.target == 1, "target", "direct_not_target:" + new.excl_reason.fillna("")),
            np.where(new.label_set == "temp", "aux_temp_L39", "aux_unknown_yamal")),
        label_set=new.label_set, label_def=new.label_set.map(LABEL_DEF_OF), cell_key=new.cell_key, subunit=new.subunit,
        country=new.country, sources=new.sources, loc_by_src=new.loc_by_src, n_src=new.n_src, n_loc=new.n_loc,
        n_values=new.n_values, eos_bases=new.eos_bases, value_kinds=new.value_kinds, obs_months=new.obs_months,
        year_min=new.year_min, year_max=new.year_max, alt_sd_loc=new.alt_sd_loc, alt_sd_all=new.alt_sd_all,
        l41a_all_direct3=new.l41a_all_direct3, l41b_early_single=new.l41b_early_single, l41b_rd_no_day=new.l41b_rd_no_day,
        l41c_dataset_statement=new.l41c_dataset_statement, v3_rel=new.v3_rel, v3_f4_cheb_deg=new.v3_f4_cheb_deg,
        dist_other_macro_km=new.dist_other_macro_km, same_cell_direct=new.same_cell_direct,
        e5_fallback_deg=new.e5_fallback_deg, soil_fallback_deg=new.soil_fallback_deg, sg_windows=new.sg_windows,
        cci_n_years=new.cci_n_years, dem_tile_missing=new.dem_tile_missing, license=new.license, lic_unverified=new.lic_unverified,
        drill_months=new.drill_months, n_cens_removed=new.n_cens_removed, n_cens_loc_only=new.n_cens_loc_only,
        cens_lb_max=new.cens_lb_max, cens_lb_mean=new.cens_lb_mean, cens_lb_gt_cell=new.cens_lb_gt_cell,
        alt_cm_cens_lb=new.alt_cm_cens_lb, n_cens_suspect=new.n_cens_suspect, cens_affected=new.cens_affected,
        n_upper_bound=new.n_upper_bound, n_approx=new.n_approx, alt_cm_no_upper=new.alt_cm_no_upper,
        label_subtypes=new.label_subtypes, q_flags=new.q_flags, n_obs_total=new.n_obs_total, date_basis=new.date_basis,
        aliases=new.aliases, v3_f4_km=new.v3_f4_km, v3_f4_km_loc=new.v3_f4_km_loc, v3_f4_lt1p1km=new.v3_f4_lt1p1km,
        e5_grid_lat=new.e5_grid_lat, e5_grid_lon=new.e5_grid_lon, e5_grid_sd_min_m=new.e5_grid_sd_min_m,
        e5_grid_dem_mean_m=new.e5_grid_dem_mean_m, e5_grid_dem_cov=new.e5_grid_dem_cov,
        sites=new.sites))
    lab_new["e5_glacier_grid"] = (lab_new.e5_grid_sd_min_m >= GLACIER_SD_M).astype(int)
    lab_new["elev_minus_e5grid_m"] = (nr.dem_elev.values - new.e5_grid_dem_mean_m.values).round(2)
    lab_new["clim_gap_e5_yr"] = [gap(a, b, E5_PERIOD) for a, b in zip(new.year_min, new.year_max)]
    lab_new["clim_gap_cci_yr"] = [gap(a, b, CCI_PERIOD) for a, b in zip(new.year_min, new.year_max)]
    lab = pd.concat([lab_v3, lab_new], ignore_index=True)
    lab["dem_center_flat"] = ((v4r.dem_slope.values == 0) & (v4r.dem_elev.values == 0)).astype(int)
    assert (lab.loc_id.values == v4r.loc_id.values).all()
    lab.to_csv(PROC / "fidelity_base_v4_labels.csv", index=False)

    # ---------------------------------------------------------------- 메타
    def complete(sub):
        ev_ok = sub.alt_cm.notna() & sub.cci_alt.notna() & sub.e5_sqrt_tdd_soil.notna()
        return dict(n=int(len(sub)), terrain6=round(float(sub[TERRAIN].notna().all(axis=1).mean()), 4),
                    climate8=round(float(sub[CLIMATE].notna().all(axis=1).mean()), 4),
                    soil9=round(float(sub[SOIL].notna().all(axis=1).mean()), 4),
                    cci_valid=round(float(sub.cci_valid.mean()), 4),
                    soil_tdd=round(float(sub.e5_sqrt_tdd_soil.notna().mean()), 4),
                    x25_all=round(float(sub[SHARED_CORE].notna().all(axis=1).mean()), 4),
                    eval_ready=round(float(ev_ok.mean()), 4), n_eval_ready=int(ev_ok.sum()),
                    n_blocks=int(sub.block.nunique()), n_blocks_eval_ready=int(sub.block[ev_ok].nunique()))
    nm = nr.drop(columns=["e5_sqrt_tdd_soil"], errors="ignore").merge(
        sn[["loc_id", "e5_sqrt_tdd_soil"]], on="loc_id", how="left")
    nm["label_set"] = new.label_set.values
    nm["macro"] = new.macro.values
    comp = {f"{ls}|{m}": complete(g) for (ls, m), g in nm.groupby(["label_set", "macro"])}
    comp_target = {m: complete(g) for m, g in nm[new.target.values == 1].groupby("macro")}
    by_src = {}
    for s in sorted(set(";".join(new.sources).split(";"))):
        k = new.sources.str.split(";").map(lambda L: s in L)
        lic = sorted({x for L in new.license[k] for x in str(L).split(";") if x})
        by_src[s] = dict(n_cells=int(k.sum()), license=lic,
                         license_attention=bool(any(("unverified" in x.lower()) or ("rights reserved" in x.lower()) for x in lic)), by_set=new[k].groupby("label_set").size().to_dict(),
                         by_macro=new[k].groupby("macro").size().to_dict(),
                         n_locations=int(sum(int(x.split(":")[1]) for L in new.loc_by_src[k] for x in L.split(";") if x.split(":")[0] == s)))
    by_region = {}
    for (ls, m), g in new.groupby(["label_set", "macro"]):
        v3mac = mac4[:len(v3)]
        v3f4 = v3[(v3.source_id == "F4_direct").values & (v3mac == (m if m != "Tibet" else "__none__"))]
        by_region[f"{ls}|{m}"] = dict(new_cells=int(len(g)), new_blocks=int(g.block.nunique()), target_cells=int(g.target.sum()),
                                     target_blocks=int(g.block[g.target == 1].nunique()),
                                     v3_f4_cells_same_macro=int(len(v3f4)), v3_f4_blocks_same_macro=int(v3f4.block.nunique()),
                                     union_blocks_with_v3=int(len(set(v3f4.block) | set(g.block[g.target == 1]))) if ls == "direct" else None,
                                     alt_mean_new=round(float(g.alt_cm.mean()), 1),
                                     subunits=g.subunit.value_counts().to_dict())
    # 개정 13: 새 행의 고도 범위와 공변량별 원천(v3 F4_direct) 1–99 백분위 밖 비율(라벨은 쓰지 않는다)
    src4 = v3[v3.source_id == "F4_direct"]
    ood_cols = [c for c in SHARED_CORE if c != "cci_valid"]
    q01, q99 = src4[ood_cols].quantile(0.01), src4[ood_cols].quantile(0.99)
    nrg = nr.assign(label_set=new.label_set.values, macro=new.macro.values, target=new.target.values)
    for (ls, m), g in nrg.groupby(["label_set", "macro"]):
        k = f"{ls}|{m}"
        if k not in by_region:
            continue
        el = g.dem_elev
        by_region[k]["dem_elev_m"] = dict(min=round(float(el.min()), 1), median=round(float(el.median()), 1), max=round(float(el.max()), 1),
                                          frac_above_src_p99=round(float((el > q99["dem_elev"]).mean()), 4))
        by_region[k]["frac_outside_src_p1_p99"] = {c: round(float(((g[c] < q01[c]) | (g[c] > q99[c]))[g[c].notna()].mean()), 3)
                                                   for c in ood_cols if g[c].notna().any()}
    ood_ref = dict(source="v3 F4_direct 17,467행", p01={c: round(float(q01[c]), 4) for c in ood_cols},
                   p99={c: round(float(q99[c]), 4) for c in ood_cols})
    el_diff = lab_new.assign(macro=new.macro.values, label_set=new.label_set.values)
    elev_diff = {f"{ls}|{m}": dict(n=int(len(g)), median=round(float(g.elev_minus_e5grid_m.median()), 1),
                                   p05=round(float(g.elev_minus_e5grid_m.quantile(0.05)), 1),
                                   n_le_m300=int((g.elev_minus_e5grid_m <= -300).sum()),
                                   n_abs_gt300=int((g.elev_minus_e5grid_m.abs() > 300).sum()),
                                   cells_abs_gt300=g.loc[g.elev_minus_e5grid_m.abs() > 300, ["loc_id", "elev_minus_e5grid_m"]].to_dict(orient="records"))
                 for (ls, m), g in el_diff.groupby(["label_set", "macro"])}
    glacier_new = lab_new.loc[lab_new.e5_glacier_grid == 1, ["loc_id", "macro_v4", "lgd_role", "e5_grid_sd_min_m"]].to_dict(orient="records")
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        git_head = ""
    meth_chk = lab_v3[lab_v3.method_basis.str.startswith("pangaea_event_coord")].groupby(["region_v4", "method"]).size()
    meta = dict(
        stage="LGD 6B.6 단계 3(v4 조립)", created=time.strftime("%Y-%m-%d %H:%M"), git_head=git_head,
        script="scripts/1_data_prep/build_fidelity_base_v4.py", script_sha256=sha(Path(__file__)),
        plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.3, 6B.4, 6B.6",
        inputs=dict(v3=dict(path="data/processed/fidelity_base_v3.csv", sha256=v3_sha0),
                    soil_v3=dict(path="data/processed/e5_soil_tdd_v3.csv", sha256=s3_sha0),
                    ext_cells=dict(path="data/processed/ext_labels/ext_cells_v1.csv", sha256=sha(EXT / "ext_cells_v1.csv")),
                    ext_cov=dict(path="data/processed/ext_labels/ext_cells_v1_cov.csv", sha256=sha(EXT / "ext_cells_v1_cov.csv"))),
        v3_files_unchanged_after_run=dict(v3=sha(v3_path) == v3_sha0, soil_v3=sha(s3_path) == s3_sha0),
        outputs={p: sha(PROC / p) for p in ["fidelity_base_v4.csv", "e5_soil_tdd_v4.csv", "fidelity_base_v4_labels.csv"]},
        n_rows=dict(v3=int(len(v3)), new=int(len(nr)), total=int(len(v4)),
                    new_by_source_id=nr.source_id.value_counts().to_dict(), new_by_region=nr.region.value_counts().to_dict()),
        hash_check=chk,
        conventions=dict(
            loc_id=f"새 행 loc_id = {LOC0} + 일련번호(direct, temp, unknown_yamal 순, 그 안에서 macro, ky, kx 순)",
            region={**REGION_OF, "v3 CALM_Greenland(3행)": "NAtlantic"},
            macro_mapping="fidelity.macro_region 은 MACRO_REGION.get(region, region) 이다. 표에 없는 region 은 그 이름이 macro 가 된다. "
                          "새 region 이름을 macro 이름과 같게 두어 m1_core 를 고치지 않고 대응한다. Tibet 은 v3 의 'Tibet'(QTP_CN, "
                          "CALM_QTP_China)과 나누려고 'Tibet_LGD' 로 둔다",
            source_id={k: v[0] for k, v in SOURCE_OF.items()},
            fidelity_level={v[0]: v[1] for v in SOURCE_OF.values()},
            spatial_support_m="새 행 1000(약 1 km 셀). v3 행은 100 그대로",
            sigma_prior_cm="셀 안 연 값 표준편차(alt_sd_all)를 3–40 으로 자름, 없으면 12(E3 규약). 입력 금지 열",
            sar="insar_* · polsar_alt · polsar_std 결측, polsar_valid 0, insar_miss 1(E3 규약)",
            soil_tdd="e5_soil_tdd_v4 의 새 행은 loc_id 로 병합된다(load_base 의 첫 병합). e5_tdd_soil <= 0 이면 tdd·sqrt 결측(v3 규칙)",
            wrapper_design=[
                "실행 표 = v4 의 v3 행 전부 + 그 대상의 새 행(labels 의 lgd_role 로 고른다). 고른 새 행의 source_id 를 F4_direct 로 바꾼다. "
                "다른 새 지역 행은 넣지 않는다(6B.6 주 설정). L42 보조 설정만 다른 새 지역의 target 행을 더한다",
                "개정 13: 실행 표는 v4 의 원문 줄을 골라 옮기고 대상 새 행의 source_id 칸만 F4_direct 로 바꿔 만든다(pandas 로 읽고 다시 쓰면 "
                "v3 행의 부동소수 11개 값이 최대 2.8e-14 달라져 6B.6 재현 점검의 입력이 본 실행과 비트 단위로 같지 않다). 구현은 "
                "scripts/1_data_prep/lgd_eligibility_v1.py 의 write_run_table_text 다",
                "개정 13: 새 행은 region 이 아니라 loc_id(>= 20000), source_id, labels 의 lgd_role 로 고른다. v4 region 'Canada' 에는 v3 CALM 16행과 "
                "새 직접 라벨 행이, 'Russia_W' 에는 새 직접 라벨 행과 GGD402 보조 행(F2_ext_unknown)이 함께 있기 때문이다. 래퍼와 집계기는 "
                "고른 새 행이 모두 loc_id >= 20000 이고 source_id 가 기대값이며 lgd_role 이 기대값인지 assert 한다",
                "개정 13: Russia_C 실행 표(주 설정과 모든 변형)에서 v3 loc 17557(CALM R8 Tiksi, 레나 델타 상자 안)을 뺀다. 6B.4 는 레나 델타 영역을 "
                "Russia_C 에서 제외한다. 이 행을 대상에 두면 100 km 버퍼가 레나 셀 1,252개를 원천에서 뺀다",
                "개정 13, L42 원천 규칙: 다른 새 지역의 직접 라벨 셀을 원천에 더할 때 NAtlantic 셀을 더하면 v3 의 하위 지점 평균 행 17520(Abisko S2)과 "
                "17569(Kapp Linné S1)을 원천에서 뺀다(같은 지점의 이중 계산). v3 QTP_CN 17389(지온 유도, 좌표 소수 2자리)는 Tibet 셀을 더해도 "
                "측정과 라벨 정의가 달라 원천에 남기고 표시한다. 17557 은 v3 행으로 원천에 남는다(새 Russia_C 셀과 168 km 이상)",
                "후보 (가) 를 권한다: 대상별 디렉터리에 실행 표를 fidelity_base_v3.csv, e5_soil_tdd_v4.csv 내용을 e5_soil_tdd_v3.csv 이름으로 두고 "
                "h40 에 --data-dir, --targets <macro>:x, --tag lgd, --out-dir data/processed/lgd 를 준다. h40·m1_core 는 고치지 않는다",
                "대상 이름: Tibet_LGD, NAtlantic, Russia_C, Russia_W, Russia_E, Canada. L39 는 Tibet_LGD 에 F3_ext_temp 행을 넣은 표로 따로 돌린다",
                "하위 지역 대응표(lg_subregion_map_v1.csv)는 대상 디렉터리에 두지 않는다. Canada 확충판의 새 블록이 표에 없어 "
                "apply_subregion_map 이 중단하기 때문이다. 표가 없으면 h40 은 k-means 하위 지역을 쓰며 macro 대상 실행에는 영향이 없다",
                "h40 의 집계(--summarize-only)는 주 4지역과 MAIN_POINT(Russia_C 점 추정 고정)를 가정한다. LGD 판정은 새 집계기가 조각을 읽어 계산한다",
                "scripts/1_data_prep/lgd_eligibility_v1.py 의 run_table 이 위 실행 표를 만들고 load_base 로 읽는 경로를 이미 거쳤다(v3 참조 26행이 "
                "h40 --count-only 산출과 일치)"]),
        by_source=by_src, by_region=by_region, ood_reference=ood_ref, elev_minus_e5grid=elev_diff,
        elev_minus_e5grid_note=("ERA5-Land 사용 격자 중심 ±0.05° 상자의 Copernicus DEM 평균을 격자 고도의 대리값으로 쓰고 dem_elev 에서 뺀 값(m). "
                                "ERA5-Land 공변량에는 지점 고도 보정이 없다(v3 규약과 같다). 값은 보정하지 않는다"),
        e5_glacier_grid=dict(rule=f"사용 격자의 2015–2020년 월 적설 수당량 최솟값 >= {GLACIER_SD_M} m", new_rows=glacier_new,
                             v3_grid_check=chk_grid_v3,
                             note="육지 판정(t2m 유효)이 빙하 격자를 거르지 않는다. v3 규약을 지키려고 값은 바꾸지 않았다. NAtlantic 에는 "
                                  "빙하 격자 셀을 뺀 변형 L41 (g)를 둔다(개정 13)"),
        covariate_completeness=dict(by_set_macro=comp, target_by_macro=comp_target,
                                    note="eval_ready = alt_cm, cci_alt, e5_sqrt_tdd_soil 이 모두 유효(m1_core.eval_mask 조건)"),
        e5_fallback=nm.assign(fb=new.e5_fallback_deg.values).groupby("macro").fb.value_counts(dropna=False).rename("n").reset_index().to_dict(orient="records"),
        v3_method_check=dict(rule="v3 region 'Canada', 'United States (Alaska)' 행을 PANGAEA 972777 이벤트 좌표와 체비쇼프 0.005° 이내로 대조",
                             counts={f"{a}|{b}": int(v) for (a, b), v in meth_chk.items()}),
        elapsed_s=round(time.time() - t0, 1))
    (PROC / "fidelity_base_v4_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[v4] {v4.shape} · 새 행 {len(nr)} · v3 행 불변(region 제외) {chk['v3_rows_unchanged_except_region']} · "
          f"토양 v3 행 불변 {chk['soil_v3_rows_unchanged']} · {time.time() - t0:.1f}s")
    print(json.dumps(comp_target, ensure_ascii=False))
    print(meth_chk.to_string())


if __name__ == "__main__":
    main()
