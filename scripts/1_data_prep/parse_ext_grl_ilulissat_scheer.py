#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""일루리사트(그린란드 서부) ALT 탐침 자료를 표준 점 자료로 바꾼다.

자료   : Scheer 등 2024, PANGAEA 964306 (CC BY 4.0)
입력   : data/raw/grl_ilulissat_scheer/ILU_ALT_measurements_Vegetation_surveys_2020-2021.txt
출력   : data/processed/ext_labels/grl_ilulissat_scheer_points.csv
         data/processed/ext_labels/grl_ilulissat_scheer_meta.json
형식   : data/processed/ext_labels/_schema.json (버전 1.0), 계획서 6B.3

처리 규칙
1. 행 단위는 (지점, 연도)다. 원자료의 ALT_2020, ALT_2021 열을 연도별 행으로 편다.
   두 해 모두 값이 없는 지점은 점 자료에 넣지 않고 제외 수만 기록한다.
2. method = probe, label_def = direct_eos, eos_basis = dataset_statement.
   근거는 자료 설명서(Metadata.pdf 표 13)의 'probed at the end of the summer in 2020 and/or 2021' 이다.
   파일에 ALT 측정 날짜가 없으므로 month 와 date 는 빈 칸이다.
   FlorSurv_Date 는 식생 조사일이다. ALT 측정일과 같은지 확인하지 못했으므로 notes 에만 적는다.
3. 좌표는 파일의 LATITUDE, LONGITUDE(WGS84 십진 도) 문자열을 그대로 쓴다.
   Easting, Northing(EPSG:3182, GR96 / UTM 22N)은 검산에만 쓴다(pyproj 가 있을 때).
4. avg_CALM_* 행은 CALM 격자 절점(CALM_A1 에서 CALM_K11)을 10 m 화소로 평균한 파생값이다.
   화소 중심 ±5 m 안의 절점 평균과 대조해 일치하면 dup_of 에 절점 목록을 적는다.
   지우지 않는다. 셀 집계에서는 dup_of 가 있는 행을 빼야 절점을 두 번 세지 않는다.
5. 좌표와 값이 모두 같은 지점 쌍은 뒤의 지점에 dup_of 를 적는다.
6. 원자료 주석이 'Out of range' 이면 right_censored = 1 이다(탐침 범위 초과).
   'ALT might be superior to indicated value' 이면 값이 하한이므로 right_censored = 1 이다.
7. 범위 QC 는 0 < alt_cm <= 600, 연도 QC 는 1990년 이후다.

실행: OMP_NUM_THREADS=1 python3 scripts/1_data_prep/parse_ext_grl_ilulissat_scheer.py
의존: 표준 라이브러리. pyproj 는 좌표 검산에만 쓰고 없으면 검산을 건너뛴다.
"""
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict

# ------------------------------------------------------------------
# 경로와 상수
# ------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ID = "grl_ilulissat_scheer"
RAW_DIR = os.path.join(ROOT, "data", "raw", SRC_ID)
RAW_FILE = os.path.join(RAW_DIR, "ILU_ALT_measurements_Vegetation_surveys_2020-2021.txt")
OUT_DIR = os.path.join(ROOT, "data", "processed", "ext_labels")
POINTS_OUT = os.path.join(OUT_DIR, SRC_ID + "_points.csv")
META_OUT = os.path.join(OUT_DIR, SRC_ID + "_meta.json")
CUSP_FILE = os.path.join(ROOT, "data", "raw", "cusp_v1_1", "cusp_v1.1.csv")
CUSP_SOURCE_KEY = "Scheer_etal_2023"

ACCESSED = "2026-09-29"
DOI = "10.1594/PANGAEA.964306"
URL = "https://doi.org/10.1594/PANGAEA.964306"
FILE_URLS = OrderedDict([
    ("ILU_ALT_measurements_Vegetation_surveys_2020-2021.txt",
     "https://download.pangaea.de/dataset/964306/files/ILU_ALT_measurements_Vegetation_surveys_2020-2021.txt"),
    ("Metadata.pdf", "https://download.pangaea.de/reference/123920/attachments/Metadata.pdf"),
    ("PANGAEA_964306_description.tab", "https://doi.pangaea.de/10.1594/PANGAEA.964306?format=textfile"),
])
LICENSE = "CC-BY-4.0"
CITATION = ("Scheer, Johanna; Caduff, Rafael; How, Penelope; Marcer, Marco; Strozzi, Tazio; "
            "Bartsch, Annett; Ingeman-Nielsen, Thomas (2024): Mapping the frost susceptibility of the "
            "ground from thaw-season InSAR surface displacements and extrapolated active layer "
            "thicknesses, Ilulissat, West-Greenland [dataset]. PANGAEA, "
            "https://doi.org/10.1594/PANGAEA.964306")
RELATED_PAPER = ("Scheer, J.; Caduff, R.; How, P.; Marcer, M.; Strozzi, T.; Bartsch, A.; "
                 "Ingeman-Nielsen, T. (2023): Thaw-Season InSAR Surface Displacements and Frost "
                 "Susceptibility Mapping to Support Community-Scale Planning in Ilulissat, West "
                 "Greenland. Remote Sensing 15(13), 3310, https://doi.org/10.3390/rs15133310")

EXPECTED_COLS = [
    "Site_ID", "Loc_ID", "Easting", "Northing", "Ref_syst", "EPSG", "Soil_type",
    "ALT_2020 [cm]", "ALT_2021 [cm]", "Landform", "Periglacial_features", "Transition",
    "Drainage", "FlorSurv_Date", "FlorSurv_area [m2]", "NV [%]", "S [%]", "F [%]", "G [%]",
    "GG [%]", "GS [%]", "GTS [%]", "GR [%]", "L [%]", "B [%]", "P [%]", "Comments",
    "GT_V_name", "Fines", "LONGITUDE", "LATITUDE",
]
YEAR_COLS = OrderedDict([(2020, "ALT_2020 [cm]"), (2021, "ALT_2021 [cm]")])

# PANGAEA 자료 설명의 범위(Coverage). 좌표 QC 에 쓴다.
BBOX = dict(lat_min=69.195710, lat_max=69.275006, lon_min=-51.157112, lon_max=-50.984989)
ALT_MIN_EXCL, ALT_MAX_INCL = 0.0, 600.0
YEAR_MIN = 1990
COORD_PREC_DEG = "0.00001"   # 파일 좌표는 소수 5자리다(끝자리 0 은 생략되어 있다)
AVG_HALF_PIXEL_M = 5.0       # avg_CALM 화소는 10 m 다
AVG_TOL_CM = 0.01            # 원자료 평균은 소수 3자리로 반올림되어 있다
CELL_DEG = 0.009
BLOCK_DEG = 0.5

OUT_COLS = [
    # 필수 15
    "src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method",
    "label_def", "n_obs", "country", "macro", "citation", "license",
    # 확장 15
    "subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm",
    "coord_prec_deg", "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of",
    "qc_flag", "notes",
]

NUM_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")
CALM_NODE_RE = re.compile(r"^CALM_[A-K]([1-9]|1[01])$")
IB_RE = re.compile(r"^[0-9A-F]{2}-[0-9A-F]{3}$")
LETTER_RE = re.compile(r"^[A-X][12]?$")


# ------------------------------------------------------------------
# 보조 함수
# ------------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def site_group(sid):
    """지점 이름의 접두로 조사 묶음을 나눈다(notes 와 요약에만 쓴다)."""
    if sid.startswith("avg_CALM_"):
        return "avg_CALM"
    if sid.startswith("ERT_CALM_"):
        return "ERT_CALM"
    if CALM_NODE_RE.match(sid):
        return "CALM_grid"
    if sid.startswith("ALT_T_"):
        return "ALT_T"
    if sid.startswith("T14_"):
        return "T14"
    if sid.startswith("Q50_"):
        return "Q50"
    if sid.startswith("Drill"):
        return "Drill"
    if sid.startswith("ILU") or sid.startswith("G2018"):
        return "borehole_ILU"
    if IB_RE.match(sid):
        return "iB"
    if LETTER_RE.match(sid):
        return "letter"
    return "other"


def parse_alt(token, sid, col):
    """ALT 문자열을 (값 문자열, 실수, '>' 표기 여부)로 바꾼다. 빈 칸은 None 이다."""
    t = token.strip()
    if t == "":
        return None
    gt = False
    if t.startswith(">"):
        gt = True
        t = t[1:].strip()
    if not NUM_RE.match(t):
        raise ValueError("숫자가 아닌 ALT 값: 지점 %s, 열 %s, 값 %r" % (sid, col, token))
    return t, float(t), gt


def iso_date_ddmmyyyy(token):
    t = token.strip()
    if t == "":
        return ""
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", t)
    if not m:
        raise ValueError("FlorSurv_Date 형식이 다르다: %r" % token)
    return "%s-%s-%s" % (m.group(3), m.group(2), m.group(1))


def cell_index(lat, lon):
    """계획서 6B.3 의 약 1 km 셀 색인."""
    ky = math.floor(lat / CELL_DEG)
    phi = math.radians((ky + 0.5) * CELL_DEG)
    kx = math.floor(lon * math.cos(phi) / CELL_DEG)
    return ky, kx


def block_index(lat, lon):
    """src/polar/fidelity.py 의 add_group_keys 와 같은 0.5° 블록 번호."""
    return int(math.floor(lat / BLOCK_DEG)) * 100000 + int(math.floor(lon / BLOCK_DEG))


def git_info(path):
    info = dict(head="", parser_tracked=None)
    try:
        info["head"] = subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        rc = subprocess.call(["git", "-C", ROOT, "ls-files", "--error-unmatch", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        info["parser_tracked"] = (rc == 0)
    except Exception:
        pass
    return info


def median(vals):
    return statistics.median(vals) if vals else None


# ------------------------------------------------------------------
# 1. 원자료 읽기
# ------------------------------------------------------------------
def read_raw():
    with open(RAW_FILE, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    header = rows[0]
    if header != EXPECTED_COLS:
        raise ValueError("머리글이 예상과 다르다: %r" % header)
    recs = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) == 0 or all(x.strip() == "" for x in r):
            continue
        if len(r) != len(header):
            raise ValueError("%d행의 열 수가 %d 이다(예상 %d)" % (i, len(r), len(header)))
        d = dict(zip(header, r))
        d["_line"] = i
        recs.append(d)
    return recs


# ------------------------------------------------------------------
# 2. 지점 단위 점검
# ------------------------------------------------------------------
def check_sites(recs):
    ids = [r["Site_ID"] for r in recs]
    if len(set(ids)) != len(ids):
        raise ValueError("Site_ID 가 중복된다: %r" % [k for k, v in Counter(ids).items() if v > 1])
    if any(s.strip() == "" or s != s.strip() for s in ids):
        raise ValueError("Site_ID 에 빈 값 또는 공백이 있다")
    for r in recs:
        if r["Ref_syst"] != "GR96" or r["EPSG"] != "3182":
            raise ValueError("좌표계 표기가 다르다: %s %s %s" % (r["Site_ID"], r["Ref_syst"], r["EPSG"]))
        r["_lat"] = float(r["LATITUDE"])
        r["_lon"] = float(r["LONGITUDE"])
        r["_e"] = float(r["Easting"])
        r["_n"] = float(r["Northing"])
        r["_grp"] = site_group(r["Site_ID"])
        r["_alt"] = OrderedDict()
        for yr, col in YEAR_COLS.items():
            p = parse_alt(r[col], r["Site_ID"], col)
            if p is not None:
                r["_alt"][yr] = p
        in_box = (BBOX["lat_min"] - 1e-4 <= r["_lat"] <= BBOX["lat_max"] + 1e-4
                  and BBOX["lon_min"] - 1e-4 <= r["_lon"] <= BBOX["lon_max"] + 1e-4)
        r["_coord_ok"] = bool(in_box)
        r["_flor_iso"] = iso_date_ddmmyyyy(r["FlorSurv_Date"])


def check_utm(recs):
    """UTM 좌표를 WGS84 로 바꿔 파일의 위경도와 비교한다(검산)."""
    out = dict(done=False)
    try:
        from pyproj import Transformer
    except Exception:
        out["reason"] = "pyproj 없음"
        return out
    tr = Transformer.from_crs(3182, 4326, always_xy=True)
    dlat, dlon, dm = [], [], []
    for r in recs:
        lon, lat = tr.transform(r["_e"], r["_n"])
        dlat.append(abs(lat - r["_lat"]))
        dlon.append(abs(lon - r["_lon"]))
        dy = (lat - r["_lat"]) * 111320.0
        dx = (lon - r["_lon"]) * 111320.0 * math.cos(math.radians(r["_lat"]))
        dm.append(math.hypot(dx, dy))
    out.update(done=True, n=len(recs), max_abs_dlat_deg=round(max(dlat), 7),
               max_abs_dlon_deg=round(max(dlon), 7), max_dist_m=round(max(dm), 2),
               median_dist_m=round(median(dm), 2))
    return out


def resolve_avg_calm(recs):
    """avg_CALM 행을 CALM 격자 절점의 화소 평균으로 재현한다."""
    nodes = [r for r in recs if r["_grp"] == "CALM_grid"]
    res = {}
    used = Counter()
    for r in recs:
        if r["_grp"] != "avg_CALM":
            continue
        inside = [n for n in nodes
                  if r["_e"] - AVG_HALF_PIXEL_M <= n["_e"] < r["_e"] + AVG_HALF_PIXEL_M
                  and r["_n"] - AVG_HALF_PIXEL_M <= n["_n"] < r["_n"] + AVG_HALF_PIXEL_M]
        per_year = {}
        ok = len(inside) > 0
        for yr, (txt, val, _gt) in r["_alt"].items():
            vals = [n["_alt"][yr][1] for n in inside if yr in n["_alt"]]
            if len(vals) == 0:
                ok = False
                continue
            mean = sum(vals) / len(vals)
            sd = statistics.stdev(vals) if len(vals) >= 2 else None
            match = abs(mean - val) <= AVG_TOL_CM
            ok = ok and match
            per_year[yr] = dict(n=len(vals), mean=mean, sd=sd, match=match)
        res[r["Site_ID"]] = dict(ok=ok, node_ids=[n["Site_ID"] for n in inside], per_year=per_year)
        if ok:
            for n in inside:
                used[n["Site_ID"]] += 1
    return res, used, len(nodes)


def resolve_same_coord(recs):
    """좌표(Easting, Northing)가 같은 지점 묶음을 찾는다. avg_CALM 은 따로 다룬다."""
    by_xy = defaultdict(list)
    for r in recs:
        if r["_grp"] == "avg_CALM":
            continue
        by_xy[(r["Easting"], r["Northing"])].append(r)
    dup_of, same_coord_diff = {}, []
    for key, grp in by_xy.items():
        if len(grp) < 2:
            continue
        first = grp[0]
        sig0 = tuple((yr, v[1]) for yr, v in first["_alt"].items())
        for other in grp[1:]:
            sig = tuple((yr, v[1]) for yr, v in other["_alt"].items())
            if sig == sig0 and len(sig) > 0:
                dup_of[other["Site_ID"]] = first["Site_ID"]
            else:
                same_coord_diff.append((first["Site_ID"], other["Site_ID"]))
    return dup_of, same_coord_diff


# ------------------------------------------------------------------
# 3. 점 자료 만들기
# ------------------------------------------------------------------
def build_points(recs, avg_res, dup_exact, same_coord_diff):
    censor_values = set()
    for r in recs:
        if re.search(r"out of range", r["Comments"], flags=re.I):
            for yr, (txt, val, _gt) in r["_alt"].items():
                censor_values.add(val)
    pair_note = {}
    for a, b in same_coord_diff:
        pair_note[a] = b
        pair_note[b] = a

    points, stats = [], Counter()
    censored_rows, near_limit_rows = [], []
    for r in recs:
        sid = r["Site_ID"]
        if len(r["_alt"]) == 0:
            stats["site_no_alt_both_years"] += 1
            continue
        comment = r["Comments"].strip()
        cen_reason = ""
        if re.search(r"out of range", comment, flags=re.I):
            cen_reason = "out_of_range"
        elif re.search(r"might be superior", comment, flags=re.I):
            cen_reason = "lower_bound_unreliable"
        for yr, (txt, val, gt) in r["_alt"].items():
            flags = []
            if not (ALT_MIN_EXCL < val <= ALT_MAX_INCL):
                flags.append("range")
            if not r["_coord_ok"]:
                flags.append("coord")
            if yr < YEAR_MIN:
                flags.append("year_pre1990")
            qc = ";".join(flags) if flags else "ok"

            rc = 1 if (gt or cen_reason) else 0
            notes = ["grp=" + r["_grp"]]
            n_obs, sd_txt, dup = 1, "", ""
            if r["_grp"] == "avg_CALM":
                a = avg_res[sid]
                if a["ok"]:
                    py = a["per_year"][yr]
                    n_obs = py["n"]
                    sd_txt = "" if py["sd"] is None else "%.3f" % py["sd"]
                    dup = ";".join("%s:%s" % (SRC_ID, n) for n in a["node_ids"])
                    notes.append("CALM 격자 절점 %d개의 10 m 화소 평균(파생값, 절점 행과 중복)" % py["n"])
                else:
                    notes.append("CALM 격자 절점 평균으로 재현되지 않았다")
            elif sid in dup_exact:
                dup = "%s:%s" % (SRC_ID, dup_exact[sid])
                notes.append("좌표와 값이 %s 와 같다" % dup_exact[sid])
            if sid in pair_note:
                notes.append("좌표가 %s 와 같고 값은 다르다" % pair_note[sid])
            if rc == 1:
                why = "'>' 표기" if gt and not cen_reason else cen_reason
                notes.append("right_censored 근거=" + why)
                if len(r["_alt"]) > 1:
                    notes.append("주석이 어느 연도의 값에 해당하는지 확인하지 못했다")
                censored_rows.append(dict(site_id=sid, year=yr, alt_cm=val, reason=why))
            elif val in censor_values:
                notes.append("값이 절단 표기 행의 값과 같다. 절단 여부는 확인하지 못했다")
                near_limit_rows.append(dict(site_id=sid, year=yr, alt_cm=val))
            if comment:
                notes.append("src_comment=" + re.sub(r"\s+", " ", comment))
            if r["_flor_iso"]:
                notes.append("florsurv_date=" + r["_flor_iso"] + "(식생 조사일)")

            points.append(OrderedDict([
                ("src_id", SRC_ID), ("site_id", sid), ("site_name", sid),
                ("lat", r["LATITUDE"].strip()), ("lon", r["LONGITUDE"].strip()),
                ("year", yr), ("month", ""), ("alt_cm", txt), ("method", "probe"),
                ("label_def", "direct_eos"), ("n_obs", n_obs), ("country", "Greenland"),
                ("macro", "NAtlantic"), ("citation", CITATION), ("license", LICENSE),
                ("subunit", "Greenland"), ("date", ""), ("eos_basis", "dataset_statement"),
                ("value_kind", "annual_value"), ("year_min", ""), ("year_max", ""),
                ("alt_sd_cm", sd_txt), ("coord_prec_deg", COORD_PREC_DEG),
                ("disturbed", ""), ("disturb_type", ""), ("right_censored", rc),
                ("orig_source", ""), ("dup_of", dup), ("qc_flag", qc),
                ("notes", " | ".join(notes)),
                ("_grp", r["_grp"]), ("_latf", r["_lat"]), ("_lonf", r["_lon"]), ("_val", val),
            ]))
    return points, stats, censored_rows, near_limit_rows


def write_points(points):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(POINTS_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for p in points:
            w.writerow(p)


# ------------------------------------------------------------------
# 4. 요약과 대조
# ------------------------------------------------------------------
def dist_summary(vals):
    if not vals:
        return dict(n=0)
    return dict(n=len(vals), min=min(vals), median=round(median(vals), 3), max=max(vals),
                mean=round(sum(vals) / len(vals), 3))


def cell_preview(points, all_rows=False):
    """셀 집계 미리 보기. 대상은 direct_eos, qc ok, 절단 아님, dup_of 없음인 행이다.

    all_rows 가 참이면 점 자료의 모든 행으로 센다(탐색 단계 추정과 비교하는 용도).
    """
    if all_rows:
        elig = list(points)
    else:
        elig = [p for p in points if p["label_def"] == "direct_eos" and p["qc_flag"] == "ok"
                and p["right_censored"] == 0 and p["dup_of"] == ""]
    loc = defaultdict(list)
    xy = {}
    for p in elig:
        loc[p["site_id"]].append(p["_val"])
        xy[p["site_id"]] = (p["_latf"], p["_lonf"])
    cells = defaultdict(list)
    for sid, vals in loc.items():
        lat, lon = xy[sid]
        cells[cell_index(lat, lon)].append((lat, lon, sum(vals) / len(vals)))
    rows = []
    for (ky, kx), items in sorted(cells.items()):
        rows.append(dict(ky=ky, kx=kx, n_sites=len(items),
                         lat=round(sum(i[0] for i in items) / len(items), 5),
                         lon=round(sum(i[1] for i in items) / len(items), 5),
                         alt_cm=round(sum(i[2] for i in items) / len(items), 2)))
    n_sites = sum(r["n_sites"] for r in rows)
    top2 = sum(sorted((r["n_sites"] for r in rows), reverse=True)[:2])
    blocks = sorted(set(block_index(r["lat"], r["lon"]) for r in rows))
    out = dict(rule="ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009 deg",
               n_rows_eligible=len(elig), n_sites_eligible=n_sites, n_cells=len(rows),
               n_blocks=len(blocks), blocks=blocks,
               share_sites_top2_cells=round(top2 / n_sites, 3) if n_sites else None)
    if all_rows:
        out["selection"] = "점 자료의 모든 행(파생 중복과 절단 행 포함)"
        return out
    out["selection"] = "label_def = direct_eos, qc_flag = ok, right_censored = 0, dup_of 없음"
    out["cells"] = rows
    out["all_rows_variant"] = cell_preview(points, all_rows=True)
    out["note"] = ("v3 와의 0.01° 중복 제거와 공변량 유효성 판정은 v4 조립 단계의 일이다. "
                   "탐색 단계 추정(셀 14개)은 셀 색인 정의가 달랐다"
                   "(탐색 표의 ky 는 7696 에서 7702, 이 규칙의 ky 는 7689 에서 7695)")
    return out


def compare_cusp(points):
    out = dict(done=False, file="data/raw/cusp_v1_1/cusp_v1.1.csv", source_key=CUSP_SOURCE_KEY)
    if not os.path.exists(CUSP_FILE):
        out["reason"] = "CUSP 파일 없음"
        return out
    cusp = {}
    with open(CUSP_FILE, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["source"] != CUSP_SOURCE_KEY:
                continue
            cusp[(row["site_id"], int(row["date"][:4]))] = row
    mine = {(p["site_id"], p["year"]): p for p in points}
    both = sorted(set(cusp) & set(mine))
    dv = [abs(float(cusp[k]["thaw_depth"]) - mine[k]["_val"]) for k in both]
    dc = [max(abs(float(cusp[k]["lat"]) - mine[k]["_latf"]),
              abs(float(cusp[k]["lon"]) - mine[k]["_lonf"])) for k in both]
    out.update(done=True, n_cusp_rows=len(cusp), n_points_rows=len(mine), n_matched=len(both),
               n_only_cusp=len(set(cusp) - set(mine)), n_only_points=len(set(mine) - set(cusp)),
               n_value_diff_gt_0p01cm=sum(1 for d in dv if d > 0.01),
               max_abs_value_diff_cm=round(max(dv), 4) if dv else None,
               max_abs_coord_diff_deg=round(max(dc), 7) if dc else None,
               cusp_dates=sorted(set(r["date"] for r in cusp.values())),
               cusp_obs_limit_filled=sum(1 for r in cusp.values() if r["obs_limit"].strip() != ""),
               cusp_quality_flags=sorted(set(r["quality_flags"] for r in cusp.values())),
               note=("CUSP 의 날짜는 연도별로 한 값이다. 원자료에 ALT 측정 날짜가 없으므로 "
                     "CUSP 의 날짜를 month 에 쓰지 않았다. 자료원 사이의 중복은 계획서 6B.3 에 따라 "
                     "이 자료원(개별 저장소 표)을 우선한다"))
    return out


def main():
    recs = read_raw()
    check_sites(recs)
    utm = check_utm(recs)
    avg_res, node_used, n_nodes = resolve_avg_calm(recs)
    dup_exact, same_coord_diff = resolve_same_coord(recs)
    points, stats, censored_rows, near_limit_rows = build_points(recs, avg_res, dup_exact, same_coord_diff)
    write_points(points)

    # ---- 집계 ----
    n_sites_raw = len(recs)
    sites_pts = sorted(set(p["site_id"] for p in points))
    n_blank_cells = sum(1 for r in recs for yr in YEAR_COLS if yr not in r["_alt"])
    by_year = Counter(p["year"] for p in points)
    by_grp_sites = Counter()
    seen = set()
    for p in points:
        if p["site_id"] not in seen:
            seen.add(p["site_id"])
            by_grp_sites[p["_grp"]] += 1
    n_years_per_site = Counter(Counter(p["site_id"] for p in points).values())
    vals_all = [p["_val"] for p in points]
    prim = [p for p in points if p["dup_of"] == "" and p["right_censored"] == 0 and p["qc_flag"] == "ok"]
    vals_prim = [p["_val"] for p in prim]
    avg_ok = sum(1 for v in avg_res.values() if v["ok"])
    flor = Counter()
    for r in recs:
        if len(r["_alt"]) == 0 or r["_flor_iso"] == "":
            continue
        fy = int(r["_flor_iso"][:4])
        flor["sites_with_alt_and_florsurv_date"] += 1
        if fy in r["_alt"]:
            flor["florsurv_year_has_alt_value"] += 1
    flor_dates = sorted(set(r["_flor_iso"] for r in recs if r["_flor_iso"]))

    gi = git_info(os.path.relpath(os.path.abspath(__file__), ROOT))
    files = []
    for name, url in FILE_URLS.items():
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            files.append(dict(name=name, size_bytes=os.path.getsize(path), sha256=sha256_of(path), url=url))

    qc = OrderedDict([
        ("n_sites_raw", n_sites_raw),
        ("n_sites_points", len(sites_pts)),
        ("n_rows_points", len(points)),
        ("n_rows_by_year", {str(k): v for k, v in sorted(by_year.items())}),
        ("n_sites_by_n_years", {str(k): v for k, v in sorted(n_years_per_site.items())}),
        ("n_sites_by_group", dict(sorted(by_grp_sites.items()))),
        ("n_by_macro", dict(Counter(p["macro"] for p in points))),
        ("n_by_subunit", dict(Counter(p["subunit"] for p in points))),
        ("year_range", [min(by_year), max(by_year)]),
        ("alt_cm_all_rows", dist_summary(vals_all)),
        ("alt_cm_primary_rows", dist_summary(vals_prim)),
        ("primary_rows_definition", "dup_of 없음, right_censored = 0, qc_flag = ok"),
        ("n_rows_primary", len(prim)),
        ("n_sites_primary", len(set(p["site_id"] for p in prim))),
        ("n_rows_right_censored", sum(1 for p in points if p["right_censored"] == 1)),
        ("n_rows_dup_of", sum(1 for p in points if p["dup_of"] != "")),
        ("n_rows_qc_not_ok", sum(1 for p in points if p["qc_flag"] != "ok")),
        ("lat_range", [min(p["_latf"] for p in points), max(p["_latf"] for p in points)]),
        ("lon_range", [min(p["_lonf"] for p in points), max(p["_lonf"] for p in points)]),
    ])

    meta = OrderedDict([
        ("src_id", SRC_ID),
        ("name", "Scheer 등 2024, 일루리사트 ALT 탐침 측정 2020–2021(PANGAEA 964306)"),
        ("url", URL),
        ("doi", DOI),
        ("accessed", ACCESSED),
        ("files", files),
        ("license", LICENSE),
        ("license_basis", "PANGAEA 자료 설명의 License 항목(Creative Commons Attribution 4.0 International)"),
        ("citation", CITATION),
        ("related_paper", RELATED_PAPER),
        ("parser", "scripts/1_data_prep/parse_ext_grl_ilulissat_scheer.py"),
        ("parser_git_commit", gi["head"]),
        ("parser_git_note", "parser_git_commit 은 실행 시점의 HEAD 다. parser_tracked 가 false 이면 파서는 아직 커밋되지 않았다"),
        ("parser_tracked", gi["parser_tracked"]),
        ("parser_sha256", sha256_of(os.path.abspath(__file__))),
        ("schema_version", "1.0"),
        ("n_rows_raw", n_sites_raw),
        ("n_rows_raw_note", "원자료의 한 행은 한 지점이고 ALT 열이 두 개(2020, 2021)다. 값 칸은 %d개, 빈 칸은 %d개다"
         % (n_sites_raw * len(YEAR_COLS), n_blank_cells)),
        ("n_rows_points", len(points)),
        ("n_by_label_def", dict(Counter(p["label_def"] for p in points))),
        ("n_by_method", dict(Counter(p["method"] for p in points))),
        ("n_by_eos_basis", dict(Counter(p["eos_basis"] for p in points))),
        ("n_excluded_by_reason", OrderedDict([
            ("site_no_alt_both_years", stats["site_no_alt_both_years"]),
            ("note", "두 해 모두 ALT 값이 없는 지점이다. 라벨 값이 없어 점 자료에 행을 만들지 않았다. "
                     "값이 있는 기록은 지우지 않았다"),
        ])),
        ("coordinate_conversion", OrderedDict([
            ("used", "파일의 LATITUDE, LONGITUDE(WGS84 십진 도) 문자열을 그대로 썼다. 변환하지 않았다"),
            ("sign", "경도는 모두 음수(서경)다"),
            ("precision", "소수 5자리(끝자리 0 생략). coord_prec_deg = 0.00001"),
            ("utm_check", utm),
            ("utm_check_note", "Easting, Northing(EPSG:3182)을 WGS84 로 바꿔 파일의 위경도와 비교한 검산이다"),
        ])),
        ("label_rule", OrderedDict([
            ("method", "probe. 근거는 Metadata.pdf 표 13 의 'Active layer thicknesses (ALT) were probed'"),
            ("label_def", "direct_eos"),
            ("eos_basis", "dataset_statement. Metadata.pdf 표 13 의 'at the end of the summer in 2020 and/or 2021', "
                          "표 14 의 'at the end of the 2020(2021) thawing season'"),
            ("month", "빈 칸. 파일에 ALT 측정 날짜가 없다"),
            ("value_kind", "annual_value. 연도별 값 하나가 주어진다. 방문 횟수는 확인하지 못했다"),
            ("n_obs", "1(모름). avg_CALM 행은 평균에 들어간 절점 수"),
            ("unit", "cm(열 이름 'ALT_2020 [cm]', 'ALT_2021 [cm]'). 환산하지 않았다"),
            ("missing_code", "빈 칸. 숫자 부호(-999 등)와 '>' 표기는 없었다"),
        ])),
        ("right_censored", OrderedDict([
            ("rule", "주석 'Out of range' 는 탐침 범위 초과로 보고 1 을 주었다. 주석 'ALT might be superior to "
                     "indicated value' 는 값이 하한이라는 뜻이므로 1 을 주었다"),
            ("rows", censored_rows),
            ("same_value_without_comment", near_limit_rows),
            ("same_value_note", "절단 표기 행과 값이 같지만 주석이 없는 행이다. right_censored = 0 으로 두고 notes 에 적었다"),
        ])),
        ("derived_and_duplicate_rows", OrderedDict([
            ("avg_calm_sites", len(avg_res)),
            ("avg_calm_reproduced", avg_ok),
            ("avg_calm_rule", "화소 중심(Easting, Northing) ±5 m 안의 CALM 격자 절점 평균과 두 해 모두 0.01 cm 안에서 일치"),
            ("calm_grid_nodes", n_nodes),
            ("calm_grid_nodes_covered_once", sum(1 for v in node_used.values() if v == 1)),
            ("calm_grid_nodes_covered_more", sum(1 for v in node_used.values() if v > 1)),
            ("exact_duplicates", [dict(site_id=k, dup_of=v) for k, v in sorted(dup_exact.items())]),
            ("same_coord_different_value", [list(x) for x in same_coord_diff]),
            ("dup_of_format", "<src_id>:<site_id>. avg_CALM 행은 절점이 여러 개라 세미콜론으로 이었다"),
        ])),
        ("florsurv_date", OrderedDict([
            ("meaning", "식생 조사일이다. ALT 측정일과 같은지 확인하지 못했다. month 와 date 에 쓰지 않았다"),
            ("distinct_dates", flor_dates),
            ("sites_with_alt_and_florsurv_date", flor["sites_with_alt_and_florsurv_date"]),
            ("florsurv_year_has_alt_value", flor["florsurv_year_has_alt_value"]),
        ])),
        ("cusp_comparison", compare_cusp(points)),
        ("related_sources", [
            OrderedDict([
                ("src_id", "cusp_v1_1"), ("relation", "같은 기록의 재수록(출처 키 Scheer_etal_2023)"),
                ("priority", "이 자료원을 우선한다. CUSP 쪽 행에 dup_of 를 적는 일은 CUSP 파서의 몫이다"),
            ]),
            OrderedDict([
                ("src_id", "mythaw_pangaea"),
                ("relation", "myThaw 의 'Ilulissat CALM site'(69.219, -51.055, 2022년과 2023년)는 "
                             "이 자료의 CALM 격자와 같은 1 km 셀에 든다. 연도가 달라 중복 기록은 아니다"),
                ("basis", "탐색 단계 후보 표의 값이다. myThaw 원자료와 직접 대조하지는 않았다"),
            ]),
            OrderedDict([
                ("src_id", "v3"),
                ("relation", "v3 의 가장 가까운 F4_direct 셀은 CALM G3(Disko, 69.25, -53.5)이고 같은 macro(NAtlantic)다. "
                             "0.01° 안에 드는 v3 셀은 없다"),
            ]),
        ]),
        ("cell_preview", cell_preview(points)),
        ("qc_summary", qc),
        ("unverified_items", [
            "지점별 ALT 측정 날짜. 파일과 설명서에 없다. 계절 말 근거는 자료 설명뿐이다",
            "탐침 길이와 지점별 반복 측정 수. 소수 값(예: 90.33, 69.67)은 반복 측정 평균으로 보이나 확인하지 못했다",
            "논문 본문(Scheer 등 2023, Remote Sensing 15, 3310)의 측정 절차. mdpi.com 이 HTTP 403 을 돌려주어 읽지 못했다",
            "지점별 시설 부지 여부. 조사 지역은 일루리사트 시가지와 주변이지만 파일에 지점별 표시가 없다. disturbed 는 빈 칸이다",
            "주석 없이 값이 110 cm, 107 cm 인 행의 절단 여부",
            "좌표가 같고 값이 다른 지점 쌍(ERT_station, ILU2018-04)의 위치 관계",
            "주석 'Stony area, yearly ALT might not be reliable'(80-03B)의 값이 하한인지 여부. right_censored = 0 으로 두었다",
            "주석 'Top frozen', 'Frozen', 'Ice thickness 14 cm' 의 뜻(지표 재동결 여부)",
            "주석 'WTC and ALT_2016 measurements from drilling log information (08-2016)'(ILU16 시추공 8지점)과 "
            "'ALT_2018 from drilling log information (08-2018)'(ILU2018-03, ILU2018-04)이 있는 지점의 2020, 2021 값이 "
            "탐침 값인지 확인하지 못했다. 파일에 ALT_2016, ALT_2018 열은 없고, 설명서 표 14 는 ALT 2020, ALT 2021 을 "
            "해당 해 융해기 말의 값으로 정의한다. 탐침 값으로 두고 주석은 notes 에 남겼다",
            "주석 'Data from iB ... included here'(CALM_H5, CALM_F6, CALM_D7)가 ALT 값까지 합친 것인지 식생 자료만 "
            "합친 것인지 확인하지 못했다. 언급된 iB 지점(A5-A30, B7-ABB, E7-D3E)은 파일에 따로 없으므로 파일 안의 중복은 아니다",
        ]),
    ])
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---- 화면 요약 ----
    print("[%s] 원자료 지점 %d, 점 자료 지점 %d, 행 %d" % (SRC_ID, n_sites_raw, len(sites_pts), len(points)))
    print("  연도별 행:", dict(sorted(by_year.items())))
    print("  제외: 두 해 모두 값 없는 지점 %d" % stats["site_no_alt_both_years"])
    print("  ALT(cm) 전체 행:", dist_summary(vals_all))
    print("  ALT(cm) 주 집합:", dist_summary(vals_prim))
    print("  right_censored 행 %d, dup_of 행 %d, qc 비정상 행 %d"
          % (qc["n_rows_right_censored"], qc["n_rows_dup_of"], qc["n_rows_qc_not_ok"]))
    print("  avg_CALM 재현 %d/%d, 절점 %d" % (avg_ok, len(avg_res), n_nodes))
    print("  UTM 검산:", utm)
    cp = meta["cell_preview"]
    print("  셀 미리 보기: 셀 %d, 블록 %d, 상위 두 셀의 지점 비율 %s"
          % (cp["n_cells"], cp["n_blocks"], cp["share_sites_top2_cells"]))
    cc = meta["cusp_comparison"]
    print("  CUSP 대조:", {k: v for k, v in cc.items() if k not in ("note", "cusp_dates", "cusp_quality_flags")})
    print("  출력:", POINTS_OUT)
    print("  메타:", META_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
