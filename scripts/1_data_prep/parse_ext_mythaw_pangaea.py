"""mythaw_pangaea: T-MOSAiC myThaw 표준 규약 융해 깊이 2021-2023 을 표준 점 자료로 만든다.

근거
----
계획서 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B 절(LGD, 개정 10), 형식 정의
`data/processed/ext_labels/_schema.json`(버전 1.0).

입력
----
data/raw/mythaw_pangaea/P956039.tab (2021년판), P971586.tab (2022년판), P974461.tab (2023년판)
  PANGAEA 탭 구분 텍스트. 머리말 주석(/* ... */) 뒤에 열 이름 1행과 자료 행렬이 온다.
data/processed/fidelity_base_v3.csv (읽기 전용. 좌표, region, source_id 만 읽어 v3 와의 관계를 기록한다)

규칙
----
1. 자료 행렬에서 Parameter 가 'thaw depth' 인 행만 쓴다. 방법은 강철 탐침(method = probe)이다.
2. 방문은 (판, Site, 날짜, Protocol ID) 다. 방문 값은 값이 있는 측선 점의 평균이다. 빈 칸은 결측이다.
   0 은 원자료가 적은 값(동결 또는 적설)이므로 평균에 넣는다.
3. 적격 방문은 값이 있는 점이 max(3, 측선 점 수의 절반) 이상인 방문이다(이 파서가 더한 조건이다).
4. 연 값은 그 해 8월 1일 이후 적격 방문의 평균 가운데 최댓값이다. 같은 값이면 이른 날짜를 쓴다.
   - 그 해 적격 방문이 2회 이상이면 label_def = direct_eos, eos_basis = series_max, value_kind = annual_value.
   - 그 해 적격 방문이 1회이고 관측 월이 8-9월이면 direct_eos, eos_basis = record_date, value_kind = single_visit.
   - 그 해 적격 방문이 1회이고 관측 월이 10월 이후이면 direct_dated, eos_basis = none, qc_flag = date.
   - (개정 13, 검증 지적 반영) 최댓값 방문이 10월 이후이면 그 해 8-9월 적격 방문과 비교해 유효 점 수가 8-9월
     방문의 최대 유효 점 수와 같고 측선 점 수(n_points)가 같을 때만 쓴다(Protocol ID 는 방문마다 다른 기록 번호라
     측정 종류 비교에 쓸 수 없다). 조건을 채우지 못하면 8-9월 적격 방문 가운데
     최댓값을 쓴다(8-9월 적격 방문이 2회 이상이면 series_max, 1회면 record_date). 8-9월 적격 방문이 없으면
     direct_dated, qc_flag = date 다. 이유: 10월 이후 방문은 재동결 중이거나 측정점 구성이 달라 같은 지점 평균의
     최댓값이라는 6B.3 정의와 맞지 않을 수 있다.
5. 8월 1일 이후 적격 방문이 없는 해는 마지막 적격 방문(없으면 값이 있는 마지막 방문)을
   direct_dated, eos_basis = none, qc_flag = date 로 둔다.
6. 탐침 길이 한계를 적은 방문에서 150 cm 이상인 점과 설명이 번호로 지목한 점을 절단 점으로 센다.
   연 값을 준 방문에 절단 점이 하나라도 있으면 right_censored = 1 이다. '>' 표기는 원자료에 없어야 한다(있으면 중단).
   (개정 13) 연 값을 준 방문의 설명에 '완전 융해'(thawed completely through, completely thawed) 점이 적혀 있으면
   그 방문도 절단 방문으로 보고 right_censored = 1 로 둔다. 그 점은 값이 없어 평균에서 빠지는데, 융해 깊이가
   탐침으로 잰 동결면보다 깊다는 뜻이므로 방문 평균은 하한 쪽으로 치우친다.
7. 좌표는 지점마다 하나로 고정한다. 머리말 Event 좌표(2021년판, 2023년판)를 먼저 쓰고, 머리말에 없는 지점은
   2022년판의 지점 설명 좌표 열(Latitude descr, Longitude descr)을 쓴다. 2022년판의 측선 좌표 열은 정수 도로
   반올림되어 있어 쓰지 않고, 고정한 좌표를 반올림한 값과 같은지만 확인한다.
8. 자료 제공자가 자연 교란으로 'thaw slumps' 를 적은 지점은 disturbed = 1, disturb_type = thermo_erosion 이다.
   그 밖의 교란 기재(거리가 있는 시설, 초식, 침수)는 disturbed = 0 으로 두고 원문을 notes 에 적는다.
9. v3 의 F4_direct 셀과 체비쇼프 0.01° 이내인 지점은 dup_of 에 'v3:<loc_id>' 를 적는다. 행은 지우지 않는다.
10. 범위(0 < ALT <= 600 cm) 밖은 qc_flag 에 range, 1990년 이전은 year_pre1990 을 붙인다.

산출
----
data/processed/ext_labels/mythaw_pangaea_points.csv
data/processed/ext_labels/mythaw_pangaea_meta.json
data/raw/mythaw_pangaea/mythaw_pangaea_visits.csv (방문 단위 요약. 파서가 만든다)

실행(ROOT): OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_mythaw_pangaea.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ID = "mythaw_pangaea"
SRC_NAME = "T-MOSAiC myThaw 표준 규약 융해 깊이 2021-2023(PANGAEA 956039, 971586, 974461)"
URL = "https://doi.org/10.1594/PANGAEA.971787"
DOI = "10.1594/PANGAEA.971787"
ACCESSED = "2026-09-29"
LICENSE = "CC BY 4.0"
ORIG_SOURCE = ("myThaw app, Permafrost Thaw Action Group database (https://permafrostthaw.org/); "
               "series https://doi.org/10.1594/PANGAEA.971787; protocol Boike et al. 2021, "
               "https://doi.org/10.1139/as-2021-0007")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"

EDITIONS = [
    dict(year=2021, file="P956039.tab", doi="10.1594/PANGAEA.956039",
         doc="Pangaea_METADATA_TMOSAiC_2021_myThaw_v2.pdf",
         doc_url="https://download.pangaea.de/reference/117389/attachments/Pangaea_METADATA_TMOSAiC_2021_myThaw_v2.pdf"),
    dict(year=2022, file="P971586.tab", doi="10.1594/PANGAEA.971586",
         doc="Pangaea_METADATA_TMOSAiC_2022_myThaw.pdf",
         doc_url="https://download.pangaea.de/reference/127187/attachments/Pangaea_METADATA_TMOSAiC_2022_myThaw.pdf"),
    dict(year=2023, file="P974461.tab", doi="10.1594/PANGAEA.974461",
         doc="Pangaea_METADATA_2023_myThaw.pdf",
         doc_url="https://download.pangaea.de/reference/131401/attachments/Pangaea_METADATA_2023_myThaw.pdf"),
]
ROUNDED_EDITIONS = {2022}      # 측선 좌표 열이 정수 도로 반올림된 판

# 원자료의 Site 값 -> 지점 정보. spacing_m 은 자료 설명 문서(표 1)의 점 간격이다.
SITES = {
    "Bayelva": dict(site_id="Bayelva", country="Norway", subunit="Svalbard", spacing_m=1, calm_grid=False),
    "CNR@Bayelva": dict(site_id="CNR_at_Bayelva", country="Norway", subunit="Svalbard", spacing_m=1,
                        calm_grid=False),
    "Kevo Vaisejaeggi": dict(site_id="Kevo_Vaisejaeggi", country="Finland", subunit="Scandinavia", spacing_m=1,
                             calm_grid=False),
    "Iskoras": dict(site_id="Iskoras", country="Norway", subunit="Scandinavia", spacing_m=1, calm_grid=False),
    "Northern Border of Russel Glacier": dict(site_id="Russel_Glacier_north", country="Greenland",
                                              subunit="Greenland", spacing_m=1, calm_grid=False),
    "Ilulissat CALM site": dict(site_id="Ilulissat_CALM_site", country="Greenland", subunit="Greenland",
                                spacing_m=1, calm_grid=True),
    "Zackenberg CALM dry transect": dict(site_id="Zackenberg_dry", country="Greenland", subunit="Greenland",
                                         spacing_m=10, calm_grid=True),
    "Zackenberg CALM wet transect": dict(site_id="Zackenberg_wet", country="Greenland", subunit="Greenland",
                                         spacing_m=10, calm_grid=True),
    "Cambridge Bay": dict(site_id="CambridgeBay", country="Canada", subunit="Nunavut", spacing_m=1,
                          calm_grid=False),
    "NRC Lake Transect": dict(site_id="NRC_Lake", country="Canada", subunit="Northwest Territories",
                              spacing_m=1, calm_grid=False),
    "Siksik Creek (TVC)": dict(site_id="Siksik_Creek", country="Canada", subunit="Northwest Territories",
                               spacing_m=1, calm_grid=False),
    "Toolik Field Station": dict(site_id="Toolik_Field_Station", country="United States", subunit="Alaska",
                                 spacing_m=1, calm_grid=False),
    "Samoylov": dict(site_id="Samoylov", country="Russia", subunit="Lena Delta", spacing_m=1, calm_grid=True),
}
# 계획서와 작업 지시가 v3 와 같은 지점으로 적은 지점(확인용)
EXPECTED_V3_DUP = {"Zackenberg_dry", "Zackenberg_wet", "Samoylov", "Toolik_Field_Station", "Siksik_Creek"}
# 다른 자료원과 같은 위치(연도는 다르다). v4 조립 단계에서 위치를 합친다
CROSS_SOURCE = {"Ilulissat_CALM_site": "grl_ilulissat_scheer 의 CALM 격자 점(CALM_*, 2020, 2021년)과 같은 "
                                       "1 km 셀에 있다. 점 대응은 확인하지 못했다"}

# v3 region -> 6B.4 의 대상 지역(독립성 거리 계산용)
V3_MACRO = {
    "ABoVE_AK": "Alaska", "United States (Alaska)": "Alaska", "ABoVE_CA": "Canada", "Canada": "Canada",
    "CALM_Canada": "Canada", "Lena_RU": "Lena", "CALM_Greenland": "NAtlantic", "CALM_Svalbard": "NAtlantic",
    "CALM_Scandinavia": "NAtlantic", "CALM_Russia_W": "Russia_W", "CALM_Russia_C": "Russia_C",
    "CALM_Russia_E": "Russia_E", "QTP_CN": "Tibet", "CALM_QTP_China": "Tibet",
}

YEAR_MIN = 1990
ALT_MAX_CM = 600.0
CELL_DEG = 0.009           # 약 1 km 셀(6B.3)
DUP_DEG = 0.01             # v3 F4_direct 와의 중복 기준(체비쇼프)
LATE_MMDD = "08-01"        # 계절 말 창의 시작
MIN_VALID_FRAC = 0.5       # 적격 방문: 값이 있는 점의 비율
MIN_VALID_N = 3            # 적격 방문: 값이 있는 점의 최소 수
PROBE_LIMIT_CM = 150.0     # 자료 설명 문서의 탐침 길이(1-1.5 m)와 방문 설명의 한계 값

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method",
            "label_def", "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm",
            "coord_prec_deg", "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of",
            "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED

C_PARAM = "Parameter"
C_SITE = "Site"
C_DATE = "Date/Time"
C_PROT = "Protocol ID"
C_PID = "ID (ID of transect point)"
C_VAL = "Thaw depth [cm]"
C_ERR = "Thaw depth e [cm]"
C_CMT = "Comment (on thaw depth measurements)"
C_LAT_T = "Latitude (of transect)"
C_LON_T = "Longitude (of transect)"
C_ELEV_T = "Elevation [m a.s.l.] (of transect)"
C_LAT_S = "Latitude descr (latitude of site in deg)"
C_LON_S = "Longitude descr (longitude of site in deg)"
C_DTYPE = "Disturbance Type"
C_DDESC = "Disturbance descr"
NEEDED = [C_PARAM, C_SITE, C_DATE, C_PROT, C_PID, C_VAL, C_ERR, C_CMT, C_LAT_T, C_LON_T, C_ELEV_T,
          C_LAT_S, C_LON_S, C_DTYPE, C_DDESC]

CENS_PAT = re.compile(r"could not measure more than|maxed out|indicates the depth is more than|"
                      r"cannot be captured", re.I)
CENS_LIST_PAT = re.compile(r"Points?\s+((?:\d+\s*(?:,|and|&)?\s*)+)might feature a deeper", re.I)
THAWED_PAT = re.compile(r"thawed completely through|completely thawed", re.I)


# ---------------------------------------------------------------- 원자료 읽기
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pangaea_tab(path: Path):
    """머리말 주석 행, 열 이름, 자료 행을 돌려준다."""
    lines = path.read_text(encoding="utf-8").split("\n")
    end = next(i for i, l in enumerate(lines) if l.strip() == "*/")
    head = [l.rstrip("\r") for l in lines[:end]]
    cols = lines[end + 1].rstrip("\r").split("\t")
    rows = []
    for l in lines[end + 2:]:
        l = l.rstrip("\r")
        if not l.strip():
            continue
        r = l.split("\t")
        if len(r) != len(cols):
            raise ValueError(f"열 수 불일치: {path.name}: {len(r)} != {len(cols)}: {l[:80]}")
        rows.append(r)
    return head, cols, rows


def header_field(head, key: str) -> str:
    for l in head:
        if l.startswith(key + ":"):
            return l.split("\t", 1)[1].strip()
    return ""


def parse_events(head) -> dict:
    """머리말의 Event(s) 구역을 지점 이름(괄호 안 표기)별 사전으로 만든다. 좌표는 문자열 그대로 둔다."""
    ev, inside = {}, False
    for l in head:
        if l.startswith("Event(s):"):
            inside = True
        elif inside and not l.startswith("\t"):
            break
        if not inside:
            continue
        body = l.split("\t", 1)[1].strip()
        parts = body.split(" * ")
        eid, _, label = parts[0].partition(" ")
        label = label.strip()
        if label.startswith("(") and label.endswith(")"):
            label = label[1:-1]
        d = dict(event=eid, label=label, lat_str="", lon_str="", elev_m="")
        for p in parts[1:]:
            k, _, v = p.partition(": ")
            k, v = k.strip(), v.strip()
            if k == "LATITUDE":
                d["lat_str"] = v
            elif k == "LONGITUDE":
                d["lon_str"] = v
            elif k == "ELEVATION":
                d["elev_m"] = v.replace(" m", "")
        if label in ev:
            raise ValueError(f"머리말 Event 의 지점 이름 중복: {label}")
        ev[label] = d
    return ev


# ---------------------------------------------------------------- 값 해석
def strip_zeros(s: str) -> str:
    """'74.466000' 을 '74.466' 으로 만든다(값은 바꾸지 않는다)."""
    if "." not in s:
        return s
    s = s.rstrip("0")
    return s + "0" if s.endswith(".") else s


def n_decimals(s: str) -> int:
    s = strip_zeros(s)
    if "." not in s or s.endswith(".0"):
        return 0
    return len(s.split(".")[1])


def is_whole_unit(s: str, per_deg: float, tol: float) -> bool:
    x = abs(float(s))
    u = (x - math.floor(x)) * per_deg
    return abs(u - round(u)) <= tol


def coord_precision(lat_s: str, lon_s: str) -> float:
    """표기에서 추정한 좌표 정밀도(도). 분 단위 값이면 0.0167, 초 단위 값이면 0.0003 이다."""
    if n_decimals(lat_s) <= 4 and n_decimals(lon_s) <= 4:
        if is_whole_unit(lat_s, 60.0, 0.005) and is_whole_unit(lon_s, 60.0, 0.005):
            return 0.0167
        return float(10.0 ** (-max(n_decimals(lat_s), n_decimals(lon_s))))
    if is_whole_unit(lat_s, 3600.0, 0.005) and is_whole_unit(lon_s, 3600.0, 0.005):
        return 0.0003
    return float(10.0 ** (-max(n_decimals(lat_s), n_decimals(lon_s))))


def macro_rule(country: str, subunit: str, lat: float, lon: float) -> str:
    """6B.4 의 지리 규칙. 라벨 값을 쓰지 않는다."""
    if country == "United States":
        return "Alaska" if subunit == "Alaska" else "other"
    if country == "Canada":
        return "Canada"
    if country in ("Greenland", "Norway", "Sweden", "Finland"):
        return "NAtlantic"
    if country == "Russia":
        if lon < 0:
            return "Russia_E"
        if lon < 90:
            return "Russia_W"
        if lon < 140:
            if 71.5 <= lat <= 73.6 and 123.3 <= lon <= 130.1:
                return "Lena"
            return "Russia_C"
        return "Russia_E"
    return "other"


def cell_index(lat: float, lon: float):
    ky = math.floor(lat / CELL_DEG)
    phi = math.radians((ky + 0.5) * CELL_DEG)
    kx = math.floor(lon * math.cos(phi) / CELL_DEG)
    return int(ky), int(kx)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def parse_value(s: str, where: str):
    s = s.strip()
    if s == "":
        return np.nan
    if s[0] in "<>":
        raise ValueError(f"부등호 표기: {where}: {s}")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        raise ValueError(f"수치가 아닌 융해 깊이: {where}: {s}")
    return float(s)


def clean_comment(txt: str) -> str:
    """표 채우기로 앞 숫자가 늘어난 설명(150, 151, ... indicates)을 하나로 모은다."""
    t = re.sub(r"\s+", " ", txt.strip())
    return re.sub(r"^\d+ indicates the depth", "150 indicates the depth", t)


def censored_points(pids, vals, comments):
    """탐침 길이 한계를 적은 방문에서 절단 점의 번호를 돌려준다."""
    text = " ".join(sorted(set(comments)))
    if not CENS_PAT.search(text):
        return set()
    out = {int(p) for p, v in zip(pids, vals) if not np.isnan(v) and v >= PROBE_LIMIT_CM}
    m = CENS_LIST_PAT.search(text)
    if m:
        out |= {int(x) for x in re.findall(r"\d+", m.group(1))}
    return out


# ---------------------------------------------------------------- 방문 표
def load_editions():
    """세 판을 읽어 융해 깊이 행, 판별 머리말 정보, 행 수를 돌려준다."""
    frames, info = [], []
    for ed in EDITIONS:
        path = RAW_DIR / ed["file"]
        head, cols, rows = read_pangaea_tab(path)
        for c in NEEDED:
            if cols.count(c) != 1:
                raise ValueError(f"열 이름이 없거나 중복: {path.name}: {c}: {cols.count(c)}")
        raw = pd.DataFrame(rows, columns=cols)[NEEDED].copy()
        years = set(raw[C_DATE].str[:4])
        if years != {str(ed["year"])}:
            raise ValueError(f"판의 연도와 다른 날짜: {path.name}: {sorted(years)}")
        lic = header_field(head, "License")
        if "CC-BY-4.0" not in lic:
            raise ValueError(f"약관이 CC BY 4.0 이 아님: {path.name}: {lic}")
        unknown = sorted(set(raw[C_SITE]) - set(SITES))
        if unknown:
            raise ValueError(f"지점 표에 없는 Site: {path.name}: {unknown}")
        thaw = raw[raw[C_PARAM] == "thaw depth"].copy()
        thaw["edition"] = ed["year"]
        thaw["date"] = thaw[C_DATE].str[:10]
        if not thaw["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError(f"날짜 형식: {path.name}")
        thaw["pid"] = thaw[C_PID].astype(float).astype(int)
        thaw["v"] = [parse_value(s, f"{path.name} {p}") for s, p in zip(thaw[C_VAL], thaw[C_PROT])]
        if thaw.duplicated([C_SITE, "date", "pid"]).any():
            raise ValueError(f"(지점, 날짜, 점 번호) 중복: {path.name}")
        if (thaw.groupby([C_SITE, "date"])[C_PROT].nunique() > 1).any():
            raise ValueError(f"같은 지점과 날짜에 Protocol ID 가 둘 이상: {path.name}")
        frames.append(thaw)
        cit = header_field(head, "Citation").rstrip().rstrip(",").rstrip()
        info.append(dict(year=ed["year"], file=ed["file"], doi=ed["doi"], citation=cit, license_line=lic,
                         events=parse_events(head), n_rows=len(raw), n_cols=len(cols),
                         n_by_parameter={k: int(v) for k, v in raw[C_PARAM].value_counts().items()},
                         n_thaw=len(thaw), n_thaw_missing=int(thaw["v"].isna().sum()),
                         sites=sorted(set(raw[C_SITE])), thaw_sites=sorted(set(thaw[C_SITE]))))
    return pd.concat(frames, ignore_index=True), info


def build_visits(thaw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ed, site, date, prot), g in thaw.groupby(["edition", C_SITE, "date", C_PROT], sort=True):
        g = g.sort_values("pid")
        v = g["v"].to_numpy()
        ok = ~np.isnan(v)
        n_pts, n_valid = len(g), int(ok.sum())
        cmts = [clean_comment(c) for c in g[C_CMT] if c.strip()]
        cens = censored_points(g["pid"], v, cmts)
        over = int((v[ok] >= PROBE_LIMIT_CM).sum())
        out.append(dict(
            edition=ed, site=site, site_id=SITES[site]["site_id"], date=date, protocol_id=prot,
            n_points=n_pts, n_valid=n_valid,
            mean_cm=float(np.mean(v[ok])) if n_valid else np.nan,
            sd_cm=float(np.std(v[ok], ddof=1)) if n_valid >= 2 else np.nan,
            min_cm=float(np.min(v[ok])) if n_valid else np.nan,
            max_cm=float(np.max(v[ok])) if n_valid else np.nan,
            n_zero=int((v[ok] == 0).sum()), n_ge150=over, n_censored=len(cens),
            censored_pids=" ".join(str(p) for p in sorted(cens)),
            thawed_through=int(any(THAWED_PAT.search(c) for c in cmts)),
            eligible=int(n_valid >= max(MIN_VALID_N, math.ceil(MIN_VALID_FRAC * n_pts))),
            late=int(date[5:] >= LATE_MMDD),
            err_cm="|".join(sorted(set(x for x in g[C_ERR] if x.strip()))),
            lat_transect="|".join(sorted(set(g[C_LAT_T]))), lon_transect="|".join(sorted(set(g[C_LON_T]))),
            elev_transect="|".join(sorted(set(g[C_ELEV_T]))),
            lat_site="|".join(sorted(set(g[C_LAT_S]))), lon_site="|".join(sorted(set(g[C_LON_S]))),
            disturbance_type="|".join(sorted(set(g[C_DTYPE]))),
            disturbance_descr="|".join(sorted(set(g[C_DDESC]))),
            comment=" | ".join(sorted(set(cmts)))))
    vis = pd.DataFrame(out).sort_values(["site_id", "date"]).reset_index(drop=True)
    vis["chosen"] = 0
    return vis


# ---------------------------------------------------------------- 좌표
def fix_coordinates(vis: pd.DataFrame, info) -> dict:
    """지점마다 좌표를 하나로 고정하고 판별 좌표 열과 대조한다."""
    ev_by_site = {}
    for inf in info:
        if inf["year"] in ROUNDED_EDITIONS:
            continue
        for label, e in inf["events"].items():
            key = (strip_zeros(e["lat_str"]), strip_zeros(e["lon_str"]))
            if label in ev_by_site and ev_by_site[label]["key"] != key:
                raise ValueError(f"판 사이 머리말 Event 좌표 불일치: {label}")
            ev_by_site.setdefault(label, dict(key=key, editions=[]))["editions"].append(inf["year"])
    coords = {}
    for site, g in vis.groupby("site"):
        if site in ev_by_site:
            lat_s, lon_s = ev_by_site[site]["key"]
            src = "event_header_" + "_".join(str(y) for y in ev_by_site[site]["editions"])
        else:
            if set(g["edition"]) - ROUNDED_EDITIONS:
                raise ValueError(f"머리말 Event 에 없는 지점이 다른 판에 있음: {site}")
            la, lo = set(g["lat_site"]), set(g["lon_site"])
            if len(la) != 1 or len(lo) != 1 or "|" in next(iter(la)) or "|" in next(iter(lo)):
                raise ValueError(f"지점 설명 좌표가 하나가 아님: {site}")
            lat_s, lon_s = strip_zeros(la.pop()), strip_zeros(lo.pop())
            src = "site_descr_2022"
        lat, lon = float(lat_s), float(lon_s)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"좌표 범위 밖: {site}")
        max_dev_m = 0.0
        for _, r in g.iterrows():
            if "|" in r["lat_transect"] or "|" in r["lon_transect"]:
                raise ValueError(f"한 방문 안에서 측선 좌표가 다름: {site} {r['date']}")
            tl, to = float(r["lat_transect"]), float(r["lon_transect"])
            if r["edition"] in ROUNDED_EDITIONS:
                if tl != math.floor(lat + 0.5) or to != math.floor(lon + 0.5):
                    raise ValueError(f"반올림 좌표와 고정 좌표 불일치: {site} {r['date']}: {tl}, {to}")
            elif abs(tl - lat) > 1e-6 or abs(to - lon) > 1e-6:
                raise ValueError(f"측선 좌표 열과 머리말 Event 좌표 불일치: {site} {r['date']}: {tl}, {to}")
            sl, so = float(r["lat_site"]), float(r["lon_site"])
            max_dev_m = max(max_dev_m, float(haversine_km(lat, lon, sl, so)) * 1000.0)
        coords[site] = dict(lat_str=lat_s, lon_str=lon_s, lat=lat, lon=lon, coord_source=src,
                            coord_prec_deg=coord_precision(lat_s, lon_s),
                            site_descr_coords=sorted(set(zip(g["lat_site"], g["lon_site"]))),
                            site_descr_max_dev_m=round(max_dev_m, 1))
    return coords


# ---------------------------------------------------------------- v3 대조
def v3_relation(coords: dict) -> dict:
    """v3 표(읽기 전용)와의 관계. 새 셀 제외 기준은 F4_direct 셀과 체비쇼프 0.01° 이내다."""
    v3 = pd.read_csv(V3, usecols=["loc_id", "lat", "lon", "region", "alt_cm", "source_id"])
    f4 = v3[v3["source_id"] == "F4_direct"].reset_index(drop=True)
    f4_macro = f4["region"].map(lambda r: V3_MACRO.get(r, r))
    rel = {}
    for site, c in coords.items():
        sid = SITES[site]["site_id"]
        macro = macro_rule(SITES[site]["country"], SITES[site]["subunit"], c["lat"], c["lon"])
        d_all = np.maximum((v3["lat"] - c["lat"]).abs(), (v3["lon"] - c["lon"]).abs())
        d_f4 = np.maximum((f4["lat"] - c["lat"]).abs(), (f4["lon"] - c["lon"]).abs())
        i4 = int(d_f4.idxmin())
        near = v3[d_all <= DUP_DEG + 1e-9].assign(cheb=d_all[d_all <= DUP_DEG + 1e-9]).sort_values("cheb")
        other = f4[f4_macro != macro]
        km_other = haversine_km(c["lat"], c["lon"], other["lat"].to_numpy(), other["lon"].to_numpy())
        j = int(np.argmin(km_other))
        n_f4_near = int((d_f4 <= DUP_DEG + 1e-9).sum())
        rel[sid] = dict(
            site_id=sid, macro=macro,
            nearest_f4_direct_cheb_deg=round(float(d_f4.min()), 5),
            nearest_f4_direct_loc_id=int(f4.loc[i4, "loc_id"]),
            nearest_f4_direct_region=str(f4.loc[i4, "region"]),
            n_f4_direct_within_0p01=n_f4_near,
            dropped_by_v3_f4_rule=bool(n_f4_near > 0),
            v3_non_f4_within_0p01=[dict(loc_id=int(r.loc_id), region=str(r.region), source_id=str(r.source_id),
                                        alt_cm=round(float(r.alt_cm), 1), cheb_deg=round(float(r.cheb), 5))
                                   for r in near.itertuples() if r.source_id != "F4_direct"][:5],
            nearest_other_macro_f4_km=round(float(km_other[j]), 1),
            nearest_other_macro_f4_region=str(other.iloc[j]["region"]),
            within_100km_of_other_macro=bool(km_other[j] <= 100.0))
    return rel


# ---------------------------------------------------------------- 연 값
def select_annual(vis: pd.DataFrame, coords: dict, info, rel) -> pd.DataFrame:
    cit = {inf["year"]: inf["citation"] for inf in info}
    doi = {inf["year"]: inf["doi"] for inf in info}
    rows = []
    for (site, year), g in vis.groupby(["site", "edition"], sort=True):
        s, c = SITES[site], coords[site]
        sid = s["site_id"]
        g = g.sort_values("date")
        has_val = g[g["n_valid"] > 0]
        elig = g[g["eligible"] == 1]
        late = elig[elig["late"] == 1]
        if len(has_val) == 0:
            raise ValueError(f"값이 있는 방문이 없음: {site} {year}")
        qc = []
        oct_rule = ""
        if len(late) > 0:
            ch = late.loc[late["mean_cm"].idxmax()]          # 같은 값이면 이른 날짜(정렬 순서)
            month = int(ch["date"][5:7])
            if len(elig) >= 2:
                label_def, eos, kind = "direct_eos", "series_max", "annual_value"
            elif month in (8, 9):
                label_def, eos, kind = "direct_eos", "record_date", "single_visit"
            else:
                label_def, eos, kind = "direct_dated", "none", "single_visit"
                qc.append("date")
            # 개정 13: 10월 이후 최댓값 방문의 비교 조건(유효 점 수와 Protocol ID 가 8-9월 방문과 같다)
            if label_def == "direct_eos" and month not in (8, 9):
                w = late[late["date"].str[5:7].isin(["08", "09"])]
                if len(w) == 0:
                    label_def, eos, kind = "direct_dated", "none", "single_visit"
                    qc.append("date")
                    oct_rule = "oct_series_max_rejected=no_aug_sep_visit"
                elif int(ch["n_valid"]) == int(w["n_valid"].max()) and set(w["n_points"]) == {ch["n_points"]}:
                    oct_rule = (f"oct_series_max_kept=n_valid {int(ch['n_valid'])} = aug_sep max "
                                f"{int(w['n_valid'].max())}, same n_points {int(ch['n_points'])}")
                else:
                    old = ch
                    ch = w.loc[w["mean_cm"].idxmax()]
                    month = int(ch["date"][5:7])
                    eos, kind = ("series_max", "annual_value") if len(w) >= 2 else ("record_date", "single_visit")
                    oct_rule = (f"oct_series_max_rejected={old['date']}:mean {old['mean_cm']:.2f} cm, n_valid "
                                f"{int(old['n_valid'])}/{int(old['n_points'])} vs aug_sep max n_valid "
                                f"{int(w['n_valid'].max())}; n_points {int(old['n_points'])} vs "
                                f"{'|'.join(sorted(set(str(int(x)) for x in w['n_points'])))}")
        else:
            pool = elig if len(elig) > 0 else has_val
            ch = pool.iloc[-1]
            month = int(ch["date"][5:7])
            label_def, eos, kind = "direct_dated", "none", "single_visit"
            qc.append("date")
        vis.loc[ch.name, "chosen"] = 1
        alt = round(float(ch["mean_cm"]), 2)
        if not (0 < alt <= ALT_MAX_CM):
            qc.append("range")
        if year < YEAR_MIN:
            qc.append("year_pre1990")
        dtype_raw, ddesc_raw = ch["disturbance_type"], ch["disturbance_descr"]
        disturbed, disturb_type = 0, ""
        if dtype_raw == "natural" and re.search(r"thaw slump", ddesc_raw, re.I):
            disturbed, disturb_type = 1, "thermo_erosion"

        notes = [f"edition=PANGAEA.{doi[year].split('.')[-1]}", f"protocol_id={ch['protocol_id']}",
                 f"n_points={int(ch['n_points'])}", f"n_valid={int(ch['n_valid'])}",
                 f"n_visits={len(elig)}", f"n_visits_from_aug1={len(late)}",
                 f"thaw_visits={has_val['date'].iloc[0]}..{has_val['date'].iloc[-1]}",
                 f"spacing_m={s['spacing_m']}", f"calm_grid={'yes' if s['calm_grid'] else 'no'}"]
        if ch["elev_transect"]:
            notes.append(f"elev_m={ch['elev_transect']}")
        if ch["err_cm"]:
            notes.append(f"reading_error_cm={ch['err_cm']}")
        notes.append(f"coord_source={c['coord_source']}")
        if c["site_descr_max_dev_m"] > 0:
            notes.append(f"site_descr_dev_m={c['site_descr_max_dev_m']}")
        if len(late) > 0 and month not in (8, 9):
            w = late[late["date"].str[5:7].isin(["08", "09"])]
            if len(w) > 0:
                b = w.loc[w["mean_cm"].idxmax()]
                notes.append(f"aug_sep_max_cm={b['mean_cm']:.2f}@{b['date']}")
            else:
                notes.append("aug_sep_max_cm=none")
        early = elig[(elig["late"] == 0) & (elig["mean_cm"] > ch["mean_cm"])]
        if len(late) > 0 and len(early) > 0:
            b = early.loc[early["mean_cm"].idxmax()]
            notes.append(f"pre_aug_higher_cm={b['mean_cm']:.2f}@{b['date']}")
        skipped = has_val[(has_val["eligible"] == 0) & (has_val["late"] == 1) & (has_val.index != ch.name)]
        for _, b in skipped.iterrows():
            notes.append(f"ineligible_visit={b['date']}:mean {b['mean_cm']:.2f} cm from "
                         f"{int(b['n_valid'])}/{int(b['n_points'])} points")
        if len(elig) == 0:
            notes.append("no_eligible_visit=1")
        if oct_rule:
            notes.append(oct_rule)
        if int(ch["n_censored"]) > 0:
            notes.append(f"n_censored_points={int(ch['n_censored'])}/{int(ch['n_valid'])}"
                         f"(points {ch['censored_pids']})")
        if int(ch["n_zero"]) > 0:
            notes.append(f"n_zero_points={int(ch['n_zero'])}")
        if int(ch["thawed_through"]) == 1:
            notes.append(f"thawed_through_points_missing={int(ch['n_points']) - int(ch['n_valid'])}")
            notes.append("right_censored_basis=thawed_through(개정 13)")
        if dtype_raw and dtype_raw != "none":
            notes.append(f"provider_disturbance={dtype_raw}: {ddesc_raw}")
        if rel is not None:
            for x in rel[sid]["v3_non_f4_within_0p01"]:
                notes.append(f"v3_near={x['source_id']}:{x['loc_id']}({x['cheb_deg']} deg)")
        if sid in CROSS_SOURCE:
            notes.append(f"cross_source={CROSS_SOURCE[sid]}")
        if ch["comment"]:
            notes.append(f"comment={ch['comment'][:400]}")

        dup_of = ""
        if rel is not None and rel[sid]["dropped_by_v3_f4_rule"]:
            dup_of = f"v3:{rel[sid]['nearest_f4_direct_loc_id']}"
        rows.append(dict(
            src_id=SRC_ID, site_id=sid, site_name=site, lat=c["lat_str"], lon=c["lon_str"], year=int(year),
            month=month, alt_cm=alt, method="probe", label_def=label_def, n_obs=int(ch["n_valid"]),
            country=s["country"], macro=macro_rule(s["country"], s["subunit"], c["lat"], c["lon"]),
            citation=cit[year], license=LICENSE, subunit=s["subunit"], date=ch["date"], eos_basis=eos,
            value_kind=kind, year_min="", year_max="",
            alt_sd_cm=round(float(ch["sd_cm"]), 2) if not np.isnan(ch["sd_cm"]) else "",
            coord_prec_deg=c["coord_prec_deg"], disturbed=disturbed, disturb_type=disturb_type,
            right_censored=1 if (int(ch["n_censored"]) > 0 or int(ch["thawed_through"]) == 1) else 0,
            orig_source=ORIG_SOURCE, dup_of=dup_of,
            qc_flag=";".join(qc) if qc else "ok", notes="; ".join(notes)))
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["site_id", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------- 요약
def dist(s: pd.Series) -> dict:
    if len(s) == 0:
        return dict(n=0)
    return dict(n=int(len(s)), min=float(s.min()), median=float(s.median()), max=float(s.max()),
                mean=round(float(s.mean()), 2))


def summarize(pts: pd.DataFrame, coords: dict, rel) -> tuple:
    main = pts[(pts["label_def"] == "direct_eos") & (pts["qc_flag"] == "ok")
               & (pts["right_censored"] == 0) & (pts["disturbed"] == 0)]
    new = pts[pts["dup_of"] == ""]
    main_new = main[main["dup_of"] == ""]
    sites = []
    for site, c in coords.items():
        s = SITES[site]
        sid = s["site_id"]
        a, m = pts[pts["site_id"] == sid], main[main["site_id"] == sid]
        ky, kx = cell_index(c["lat"], c["lon"])
        d = dict(site_id=sid, site_name=site, lat=c["lat_str"], lon=c["lon_str"],
                 coord_source=c["coord_source"], coord_prec_deg=c["coord_prec_deg"],
                 site_descr_coords=[list(x) for x in c["site_descr_coords"]],
                 site_descr_max_dev_m=c["site_descr_max_dev_m"], country=s["country"], subunit=s["subunit"],
                 macro=a["macro"].iloc[0], spacing_m_doc=s["spacing_m"], calm_grid=s["calm_grid"],
                 cell_ky=ky, cell_kx=kx, years=[int(y) for y in a["year"]],
                 n_rows=int(len(a)), n_main=int(len(m)),
                 alt_by_year_cm={str(int(y)): float(v) for y, v in zip(a["year"], a["alt_cm"])},
                 alt_mean_main_cm=round(float(m["alt_cm"].mean()), 1) if len(m) else None,
                 dup_of=a["dup_of"].iloc[0], n_right_censored=int(a["right_censored"].sum()),
                 n_disturbed=int(a["disturbed"].sum()))
        if rel is not None:
            d.update({k: rel[sid][k] for k in ("nearest_f4_direct_cheb_deg", "nearest_f4_direct_loc_id",
                                               "nearest_other_macro_f4_km", "nearest_other_macro_f4_region",
                                               "within_100km_of_other_macro")})
        sites.append(d)
    sdf = pd.DataFrame(sites)
    new_sites = sdf[sdf["dup_of"] == ""]
    tgt_sites = new_sites[new_sites["n_main"] > 0]
    site_mean = main_new.groupby("site_id")["alt_cm"].mean()
    summ = dict(
        n_sites=int(pts["site_id"].nunique()), n_rows=int(len(pts)), n_rows_main=int(len(main)),
        main_set_rule="label_def = direct_eos, qc_flag = ok, right_censored = 0, disturbed = 0",
        n_sites_new=int(len(new_sites)), n_rows_new=int(len(new)), n_rows_main_new=int(len(main_new)),
        new_rule="dup_of 가 빈 지점(v3 의 F4_direct 셀과 체비쇼프 0.01° 이내가 아니다)",
        n_sites_by_macro={k: int(v) for k, v in sdf.groupby("macro").size().items()},
        n_new_sites_by_macro={k: int(v) for k, v in new_sites.groupby("macro").size().items()},
        n_target_sites_by_macro={k: int(v) for k, v in tgt_sites.groupby("macro").size().items()},
        n_target_sites_by_subunit={k: int(v) for k, v in tgt_sites.groupby("subunit").size().items()},
        n_rows_by_macro={k: int(v) for k, v in pts.groupby("macro").size().items()},
        n_rows_by_method={k: int(v) for k, v in pts.groupby("method").size().items()},
        n_rows_by_label_def={k: int(v) for k, v in pts.groupby("label_def").size().items()},
        n_rows_by_eos_basis={k: int(v) for k, v in pts.groupby("eos_basis").size().items()},
        n_rows_by_qc_flag={k: int(v) for k, v in pts.groupby("qc_flag").size().items()},
        n_rows_by_month={str(k): int(v) for k, v in pts.groupby("month").size().items()},
        n_right_censored=int(pts["right_censored"].sum()), n_disturbed=int(pts["disturbed"].sum()),
        n_dup_of_v3=int((pts["dup_of"] != "").sum()),
        year_range=[int(pts["year"].min()), int(pts["year"].max())],
        alt_cm_rows_all=dist(pts["alt_cm"]), alt_cm_rows_main=dist(main["alt_cm"]),
        alt_cm_rows_main_new=dist(main_new["alt_cm"]),
        alt_cm_new_target_site_multiyear_mean=dict(
            n=int(len(site_mean)), min=round(float(site_mean.min()), 1),
            median=round(float(site_mean.median()), 1), max=round(float(site_mean.max()), 1)),
        n_distinct_1km_cells_all=int(sdf[["cell_ky", "cell_kx"]].drop_duplicates().shape[0]),
        n_distinct_1km_cells_new=int(new_sites[["cell_ky", "cell_kx"]].drop_duplicates().shape[0]),
        n_distinct_1km_cells_new_target=int(tgt_sites[["cell_ky", "cell_kx"]].drop_duplicates().shape[0]))
    return summ, sites


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def file_entry(path: Path, source_url: str, kind: str) -> dict:
    return dict(name=str(path.relative_to(ROOT)), size_bytes=path.stat().st_size, sha256=sha256_of(path),
                source_url=source_url, kind=kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-v3", action="store_true", help="v3 표 대조를 건너뛴다(dup_of 가 빈 칸이 된다)")
    args = ap.parse_args()

    thaw, info = load_editions()
    vis = build_visits(thaw)
    coords = fix_coordinates(vis, info)
    rel = None if args.no_v3 else v3_relation(coords)
    pts = select_annual(vis, coords, info, rel)

    # 형식 확인
    assert list(pts.columns) == COLUMNS
    assert pts[REQUIRED].notna().all().all()
    assert (pts[REQUIRED].astype(str) != "").all().all()
    assert not pts.duplicated(["site_id", "year"]).any()
    assert set(pts["method"]) == {"probe"}
    assert set(pts["label_def"]) <= {"direct_eos", "direct_dated"}
    assert int(vis["chosen"].sum()) == len(pts)
    n_over_no_comment = int(((vis["n_ge150"] > 0) & (vis["n_censored"] == 0)).sum())
    dup_found = set(pts.loc[pts["dup_of"] != "", "site_id"])
    dup_check = None if rel is None else dict(expected=sorted(EXPECTED_V3_DUP), found=sorted(dup_found),
                                              identical=dup_found == EXPECTED_V3_DUP)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{SRC_ID}_points.csv"
    pts.to_csv(out_csv, index=False, encoding="utf-8")
    vis_csv = RAW_DIR / f"{SRC_ID}_visits.csv"
    vis.round({"mean_cm": 3, "sd_cm": 3}).to_csv(vis_csv, index=False, encoding="utf-8")

    # 제외 수(방문 단위와 행 단위)
    n_raw = sum(i["n_rows"] for i in info)
    n_thaw = sum(i["n_thaw"] for i in info)
    n_missing = sum(i["n_thaw_missing"] for i in info)
    not_chosen = vis[vis["chosen"] == 0]
    excluded = {
        "융해 깊이가 아닌 매개변수 행(적설, 식생, 수위, 토양, 수고)": int(n_raw - n_thaw),
        "융해 깊이 값이 빈 행(결측)": int(n_missing),
        "연 값으로 쓰지 않은 방문: 8월 1일 이전": int(((not_chosen["late"] == 0)).sum()),
        "연 값으로 쓰지 않은 방문: 8월 1일 이후 적격이나 최댓값이 아님":
            int(((not_chosen["late"] == 1) & (not_chosen["eligible"] == 1)).sum()),
        "연 값으로 쓰지 않은 방문: 8월 1일 이후이나 값이 있는 점이 부족(부적격)":
            int(((not_chosen["late"] == 1) & (not_chosen["eligible"] == 0)).sum()),
    }
    summ, sites = summarize(pts, coords, rel)
    files = []
    for ed in EDITIONS:
        files.append(file_entry(RAW_DIR / ed["file"],
                                f"https://doi.pangaea.de/{ed['doi']}?format=textfile", "내려받은 원자료"))
    for ed in EDITIONS:
        p = RAW_DIR / ed["doc"]
        if p.exists():
            files.append(file_entry(p, ed["doc_url"], "자료 설명 문서"))
    files.append(file_entry(vis_csv, "", "파서가 만든 방문 단위 요약"))

    meta = dict(
        src_id=SRC_ID, name=SRC_NAME, url=URL, doi=DOI, accessed=ACCESSED,
        editions=[dict(year=i["year"], file=i["file"], doi=i["doi"], citation=i["citation"],
                       license_line=i["license_line"], n_rows=i["n_rows"], n_cols=i["n_cols"],
                       n_by_parameter=i["n_by_parameter"], n_thaw_rows=i["n_thaw"],
                       n_thaw_missing=i["n_thaw_missing"], sites=i["sites"], thaw_sites=i["thaw_sites"],
                       header_event_sites=sorted(i["events"].keys())) for i in info],
        files=files,
        license=LICENSE,
        license_basis="세 판 모두 원자료 머리말의 License 행이 Creative Commons Attribution 4.0 International "
                      "(CC-BY-4.0) 이다. 파서가 실행할 때마다 확인한다",
        citation="; ".join(i["citation"] for i in info),
        parser=f"scripts/1_data_prep/parse_ext_{SRC_ID}.py",
        parser_sha256=sha256_of(Path(__file__).resolve()),
        parser_git_commit=git_head(),
        parser_git_note="파서 파일은 이 커밋에 들어 있지 않다(작업 트리의 새 파일). 값은 실행 시점의 HEAD 다",
        schema_version="1.0",
        n_rows_raw=int(n_raw), n_rows_raw_thaw=int(n_thaw), n_rows_raw_thaw_valid=int(n_thaw - n_missing),
        n_visits=int(len(vis)), n_visits_eligible=int(vis["eligible"].sum()),
        n_visits_from_aug1=int(vis["late"].sum()), n_rows_points=int(len(pts)),
        n_by_label_def={k: int(v) for k, v in pts.groupby("label_def").size().items()},
        n_by_method={k: int(v) for k, v in pts.groupby("method").size().items()},
        n_excluded_by_reason=excluded,
        label_rule=dict(
            late_from=f"MM-DD >= {LATE_MMDD}", min_valid_frac=MIN_VALID_FRAC, min_valid_n=MIN_VALID_N,
            probe_limit_cm=PROBE_LIMIT_CM,
            text="연 값은 8월 1일 이후 적격 방문의 지점 평균 가운데 최댓값이다. 적격 방문 조건(값이 있는 점이 "
                 "max(3, 측선 점 수의 절반) 이상)은 이 파서가 더했다. 그 해 적격 방문이 1회뿐이면 eos_basis 를 "
                 "record_date 로 적는다(8-9월). 10월 이후의 최댓값은 유효 점 수가 8-9월 방문의 최대 유효 점 수와 "
                 "같고 측선 점 수가 같을 때만 direct_eos(series_max)로 두고, 아니면 8-9월 적격 방문의 최댓값을 쓴다"
                 "(개정 13). 8-9월 방문의 최댓값은 notes 의 aug_sep_max_cm 에 적는다. 연 값을 준 방문에 '완전 융해' "
                 "점이 있으면 right_censored = 1 이다(개정 13)"),
        coordinate_conversion="변환 없음. 원자료가 WGS84 십진 도이고 서경은 음수다. 지점마다 좌표를 하나로 고정했다. "
                              "머리말 Event 좌표(2021년판, 2023년판. 자료 행렬의 측선 좌표 열과 1e-6° 안에서 같음을 "
                              "확인)를 먼저 쓰고, 머리말에 없는 Iskoras 와 Northern Border of Russel Glacier 는 "
                              "2022년판의 지점 설명 좌표 열을 썼다. 2022년판 머리말의 Event 는 2021년 이벤트 9개뿐이다. "
                              "2022년판의 측선 좌표 열은 정수 도이고 고정한 좌표의 반올림 값과 같음을 확인했다",
        coord_prec_rule="위경도가 모두 분 단위 값이면 0.0167, 초 단위 값이면 0.0003, 아니면 10^-(표기 소수 자릿수). "
                        "표기에서 추정한 값이다",
        units="Thaw depth [cm]. 단위 변환 없음",
        missing_code="빈 칸. 0 은 결측이 아니라 원자료가 적은 값이다(동결, 적설)",
        inequality_marks="'<', '>' 표기 없음(있으면 파서가 중단한다). 탐침 길이 한계는 방문 설명으로 적혀 있다",
        n_visits_ge150_without_limit_comment=n_over_no_comment,
        duplicates="(지점, 날짜, 점 번호) 중복 없음. 같은 지점과 날짜에 Protocol ID 는 하나다(파서가 확인한다)",
        disturbance_rule="자료 제공자의 Disturbance Type 이 natural 이고 설명이 thaw slumps 인 지점(Iskoras)만 "
                         "disturbed = 1, disturb_type = thermo_erosion 이다. 나머지 기재는 notes 의 "
                         "provider_disturbance 에 원문으로 적었다",
        v3_dup_check=dup_check,
        qc_summary=summ, sites=sites, v3_relation=rel,
        unverified_items=[
            "측선이 교란 지형 위에 있는지 여부. Iskoras 의 thaw slumps, NRC Lake 2022-09-16 의 침수 기재는 "
            "지점 주변(30 m x 30 m) 설명이고 측선 위 상태는 확인하지 못했다",
            "지점별 탐침 길이. 자료 설명 문서는 1-1.5 m 강철 봉으로 적는다. Bayelva 밖의 지점에서 "
            "탐침 길이에 닿은 값이 있는지는 확인하지 못했다",
            "Kevo Vaisejaeggi 의 '완전 융해' 점(2022-11-02 의 3점, 2023년의 2점)이 영구동토 소멸인지 "
            "탐침 길이 초과인지. 값이 없어 평균에서 빠졌다. 개정 13 부터 그런 방문이 연 값을 주면 right_censored = 1 이다",
            "10월 이후 최댓값 방문의 비교 조건(유효 점 수, 측선 점 수)은 결과 열람 전에 정한 파서 규칙이다. "
            "자료 제공자의 권고는 확인하지 못했다",
            "CNR@Bayelva 는 암석지라 점마다 3회 측정의 최댓값을 올렸다고 적혀 있다(2023년). 다른 연도의 "
            "측정 방식은 확인하지 못했다. 2021-08-31 방문의 3점은 제공자가 80 cm 로 바꿔 올렸다",
            "Iskoras 와 Northern Border of Russel Glacier 의 좌표는 지점 설명 좌표다. 자료 설명 문서는 "
            "지점 좌표가 측선 좌표와 다를 수 있다고 적는다. 두 좌표가 같이 있는 지점의 차이는 "
            "sites 의 site_descr_max_dev_m 에 적었다",
            "Ilulissat CALM site 의 2022년 방문은 21점 가운데 11점에만 값이 있다(설명: CALM grid 1.1-1.11). "
            "2022년과 2023년의 측선이 같은 점 배치인지는 확인하지 못했다",
            "단일 방문 연도(Siksik Creek, Iskoras, Russel Glacier, Ilulissat)의 값이 그 해 최대 융해 깊이에 "
            "얼마나 가까운지",
            "Ilulissat CALM site 와 grl_ilulissat_scheer 의 CALM 격자 점 사이의 점 대응. 두 자료는 연도가 다르고"
            "(2022-2023년과 2020-2021년) 같은 1 km 셀에 있다. 위치 합치기는 v4 조립 단계의 일이다",
            "2024년판(PANGAEA 997035). 로그인과 저자 승인이 필요해 받지 않았다",
            "ERA5-Land 육지 폴백, CCI ALT 유효 여부, 공변량 부착 결과. v4 조립 단계의 일이다",
        ])
    out_meta = OUT_DIR / f"{SRC_ID}_meta.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"원자료 행 {n_raw}, 융해 깊이 행 {n_thaw}(결측 {n_missing}), 방문 {len(vis)}, 점 자료 행 {len(pts)}")
    print(f"지점 {summ['n_sites']}(새 지점 {summ['n_sites_new']}), 주 집합 행 {summ['n_rows_main']}"
          f"(새 지점 {summ['n_rows_main_new']})")
    print("label_def:", meta["n_by_label_def"], "eos_basis:", summ["n_rows_by_eos_basis"])
    print("v3 중복 확인:", dup_check)
    print(f"150 cm 이상이나 한계 설명이 없는 방문: {n_over_no_comment}")
    print("저장:", out_csv.relative_to(ROOT), out_meta.relative_to(ROOT), vis_csv.relative_to(ROOT))


if __name__ == "__main__":
    main()
