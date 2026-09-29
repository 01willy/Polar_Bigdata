#!/usr/bin/env python3
"""ext_labels 파서: nsidc_ggd353_thawtube.

자료: Nixon, F. M. (2003, 2009년 1월 갱신), Active Layer Monitoring, Arctic and Subarctic Canada, Version 6
(NSIDC GGD353). https://doi.org/10.7265/7m84-k262
원 출처: 캐나다 지질조사소(GSC)의 매켄지 계곡 융해관 관측망(책임자 F. M. Nixon). 포트심프슨에서 보퍼트 해안의
노스헤드까지 약 60개 지점이고 그 가운데 10개가 CALM 지점(C3, C4, C5, C7, C8, C9, C11, C13, C14, C15)이다.

입력(data/raw/nsidc_ggd353_thawtube/)
  ggd353data_v6.rtf   자료 전체(지점 설명과 'Active Layer DATA' 표). 한 파일이다
  (문서) README.txt, ggd353-v006-userguide_1.pdf, landing_ggd353_v6.html, ggd353_dif_metadata.xml,
         ggd353_citation.bib
  (대조) data/raw/calm/PANGAEA_972777_CALM_ALT_NH.tab 의 CALM_C*A, CALM_C13 이벤트(융해관 계열),
         data/processed/fidelity_base_v3.csv(좌표, 지역, 자료 구분, 라벨 칸만 읽는다. 고치지 않는다),
         data/processed/ext_labels/ 의 다른 자료원 점 자료(거리만 본다)
출력
  data/processed/ext_labels/nsidc_ggd353_thawtube_points.csv   표준 점 자료(형식 1.0)
  data/processed/ext_labels/nsidc_ggd353_thawtube_meta.json    메타
  data/raw/nsidc_ggd353_thawtube/ggd353_table_cells.csv         RTF 표의 칸 단위 원문(지점 x 연도)
  data/raw/nsidc_ggd353_thawtube/ggd353_site_descriptions.csv   RTF 지점 설명의 항목별 원문

규칙(계획서 6B.2, 6B.3, _schema.json 1.0, _sources_plan.json 의 label_rule)
  1. RTF 를 제어어 단위로 읽어 표의 행(\\row)과 칸(\\cell)을 복원한다. 'Active Layer DATA:' 뒤의 두 표
     (CALM, Non-CALM)를 쓴다. 머리글은 Site, Latitude, Longitude, 1991-2007 이다(칸 20개).
  2. 한 행은 (지점, 연도)다. 값이 있는 칸만 행으로 만든다. 'N.A.', 'NA', 빈 칸은 결측이다.
  3. 값은 연 최대 활동층 두께(cm, 정수, 정밀도 ±1 cm)다. 자료 설명이 정의한 ALT 는 '융해관의 융해 깊이에서
     최대 침하 때 지표 위 관 높이를 뺀 값'이다. 파일에는 이 보정된 값 하나만 있고 보정 전 값이나 융기, 침하
     값은 없다. 따라서 선택할 것이 없고 이 값을 그대로 쓴다.
     method = thaw_tube, label_def = direct_eos, eos_basis = protocol, value_kind = annual_value, n_obs = 1.
  4. 값 앞뒤 표기(자료 설명의 표 아래 주석)
       '>' 최솟값. 융해가 관의 측정 범위 아래로 내려갔다. right_censored = 1
       '<' 최댓값. 최대 침하 값이 없어 다음 해 한여름 지표 높이를 최소 침하로 썼다. 참값은 이 값 이하다.
           notes 에 value_bound=upper 를 적는다. label_def 는 direct_eos 로 둔다
       '*' 융해관 측정이 아니라 늦은 시기(late year) 관측에 근거한 값. 관측 방법과 날짜가 없다.
           method = other, label_def = direct_dated, eos_basis = dataset_statement
       '~' 근사값(설명 없음). notes 에 approx=1
       '?' 불확실(설명 없음). notes 에 uncertain=1
  5. 92TT12(Tulita)의 1994, 1995 칸은 'Fire in', '1995' 라는 글이다(1995년 산불). 값으로 읽지 않는다.
     지점 설명의 '(site burned in 1995 fire)' 에 따라 1995년 이후 값은 disturbed = 1, disturb_type = burned.
  6. 좌표는 표의 도분초다(북위, 서경). 서경은 음수. 십진 도 = 도 + 분/60 + 초/3600, 소수 6자리.
     coord_prec_deg = 0.0003(1초). 지점 설명의 Location 좌표와 1 km 넘게 다르면 qc_flag 에 coord 를 붙인다.
  7. 지점 이력(History)에 'not visited A - B' 가 있으면 A 이상 B 미만 연도의 값에 qc_flag date 를 붙인다.
  8. CALM 10개 지점은 dup_of 를 적는다. 같은 CALM 좌표의 v3 행이 있으면 v3:<loc_id>, 없으면(C3) 같은
     지점의 탐침 격자를 담은 calm_pangaea_v4add:CALM_C3B 다. PANGAEA 972777 의 융해관 이벤트(CALM_C*A,
     CALM_C13)와 연 값을 대조해 notes 와 메타에 적는다.
  9. 범위 QC 0 < alt_cm <= 600. 1990년 이전 값은 없다(표가 1991년부터다).

실행: 스레드 1개. 입력 RTF 는 1.1 MB 이고 v3 표는 좌표 칸만 읽는다.
  OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_nsidc_ggd353_thawtube.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SRC_ID = "nsidc_ggd353_thawtube"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3_PATH = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
PANGAEA_PATH = ROOT / "data" / "raw" / "calm" / "PANGAEA_972777_CALM_ALT_NH.tab"

ACCESSED = "2026-09-29"
URL = "https://nsidc.org/data/ggd353/versions/6"
DOI = "10.7265/7m84-k262"
FTP_DIR = "ftp://sidads.colorado.edu/pub/DATASETS/fgdc/ggd353_activlayer_canada/"
FILE_URLS = {
    "ggd353data_v6.rtf": FTP_DIR + "ggd353data_v6.rtf",
    "README.txt": FTP_DIR + "README.txt",
    "ggd353-v006-userguide_1.pdf": "https://nsidc.org/sites/default/files/ggd353-v006-userguide_1.pdf",
    "landing_ggd353_v6.html": URL,
    "ggd353_dif_metadata.xml": ("http://nsidc.org/api/dataset/metadata/oai?verb=GetRecord"
                                "&metadataPrefix=dif&identifier=GGD353.006"),
    "ggd353_citation.bib": "https://doi.org/10.7265/7m84-k262 (BibTeX 형식. 받은 경로의 기록이 없어 확인하지 못했다)",
}
DOC_FILES = ["ggd353data_v6.rtf", "README.txt", "ggd353-v006-userguide_1.pdf", "landing_ggd353_v6.html",
             "ggd353_dif_metadata.xml", "ggd353_citation.bib"]
CITATION = ("Nixon, F. M. (2003). Active Layer Monitoring, Arctic and Subarctic Canada. (GGD353, Version 6). "
            "[Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. "
            "https://doi.org/10.7265/7m84-k262. Subset: 'Active Layer DATA' tables (CALM and Non-CALM), "
            "1991-2007. Date Accessed 09-29-2026.")
ORIG_SOURCE = ("Geological Survey of Canada, Mackenzie Valley active layer monitoring network (thaw tubes; "
               "PI F. M. Nixon); Nixon et al. 2003; Tarnocai et al. 2004")
LICENSE = "unverified"
COUNTRY = "Canada"
MACRO = "Canada"
SUBUNIT = "Mackenzie"
PREC_SECOND = 0.0003
COORD_DISAGREE_KM = 1.0
YEARS_EXPECTED = list(range(1991, 2008))
# 자료 누리집이 적은 범위(북 69°43', 남 61°53', 서 135°20', 동 121°36'). 점검용으로 조금 넓힌다
BBOX = dict(lat_min=61.0, lat_max=70.5, lon_min=-136.0, lon_max=-120.5)

# CALM 표의 지점 이름과 CALM 번호(표 이름은 줄임말이 섞여 있다. 지점 설명의 '이름 (Cn)' 과 대조한다)
TABLE_TO_CALM = {
    "North Head": "C3", "Taglu": "C4", "Lousy Point": "C5", "Reindeer Dpt.": "C7", "Rengleng R.": "C8",
    "Mountain R.": "C9", "Norman Wells": "C11", "Ochre River": "C13", "Willowlake R.": "C14",
    "Fort Simpson": "C15",
}
# PANGAEA 972777 에서 융해관 계열인 이벤트. C13 은 Method 표기가 없는 이벤트 하나뿐이다
PANGAEA_TT_EVENT = {"C3": "CALM_C3A", "C4": "CALM_C4A", "C5": "CALM_C5A", "C7": "CALM_C7A",
                    "C8": "CALM_C8A", "C9": "CALM_C9A", "C11": "CALM_C11A", "C13": "CALM_C13",
                    "C14": "CALM_C14A", "C15": "CALM_C15A"}
# C3 은 v3 에 행이 없고 같은 지점의 탐침 격자가 calm_pangaea_v4add 에 있다
DUP_FALLBACK = {"C3": "calm_pangaea_v4add:CALM_C3B"}

COLUMNS = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def",
           "n_obs", "country", "macro", "citation", "license",
           "subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
           "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes"]

# ---------------------------------------------------------------- RTF 읽기
# 내용이 아닌 목적지 그룹. '{\*\...}' 는 모두 건너뛴다(책갈피, 스마트 태그, 필드 명령 등)
SKIP_DEST = {
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "headerl", "headerr", "headerf", "footer",
    "footerl", "footerr", "footerf", "listtable", "listoverridetable", "rsidtbl", "generator", "themedata",
    "colorschememapping", "latentstyles", "datastore", "object", "shpinst", "shprslt", "xmlnstbl", "fldinst",
    "bkmkstart", "bkmkend", "xmlopen", "xmlclose", "xmlattrname", "xmlattrvalue", "factoidname", "nonshppict",
    "ftnsep", "ftnsepc", "aftnsep", "aftnsepc", "userprops", "docvar", "template", "background", "shp",
    "annotation", "atnid", "atnauthor", "atndate", "atnref", "revtbl", "filetbl", "pgdsctbl", "mmathPr",
}
TOK = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|([^\\{}\r\n]+)|[\r\n]+", re.S)


def rtf_events(raw: bytes):
    """RTF 를 문단('para', 글)과 표 행('row', [칸 글])의 순서 목록으로 바꾼다."""
    s = raw.decode("latin-1")
    stack, skip, uc, pending = [], False, 1, 0
    cur, cells, events = [], [], []
    in_tbl, first = False, False
    for m in TOK.finditer(s):
        word, arg, hexc, sym, brace, text = m.groups()
        if brace == "{":
            stack.append((skip, uc))
            first = True
            continue
        if brace == "}":
            skip, uc = stack.pop() if stack else (False, 1)
            first = False
            continue
        is_first, first = first, False
        if sym == "*":
            if is_first:
                skip = True
            continue
        if word is not None:
            if is_first and word in SKIP_DEST:
                skip = True
            if skip:
                continue
            if word == "uc":
                uc = int(arg or 1)
            elif word == "u":
                cur.append(chr(int(arg) % 65536))
                pending = uc
            elif word in ("par", "sect", "page", "line"):
                if in_tbl:
                    cur.append(" ")
                else:
                    events.append(("para", "".join(cur)))
                    cur = []
            elif word == "intbl":
                in_tbl = True
            elif word == "pard":
                in_tbl = False
            elif word in ("cell", "nestcell"):
                cells.append("".join(cur))
                cur = []
            elif word in ("row", "nestrow"):
                events.append(("row", cells))
                cells, cur = [], []
            elif word == "tab":
                cur.append("\t")
            elif word in ("emdash", "endash"):
                cur.append("-")
            elif word in ("lquote", "rquote"):
                cur.append("'")
            elif word in ("ldblquote", "rdblquote"):
                cur.append('"')
            continue
        if skip:
            continue
        if hexc is not None:
            if pending:
                pending -= 1
                continue
            cur.append(bytes([int(hexc, 16)]).decode("cp1252", "replace"))
            continue
        if sym is not None:
            if sym in "\\{}":
                cur.append(sym)
            elif sym == "~":
                cur.append(" ")
            elif sym == "_":
                cur.append("-")
            continue
        if text is not None:
            if pending:
                n = min(pending, len(text))
                text, pending = text[n:], pending - n
            cur.append(text)
    if cur:
        events.append(("para", "".join(cur)))
    return events


# ---------------------------------------------------------------- 보조 함수
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True)
        return out.stdout.decode().strip()
    except Exception:
        return ""


DMS = re.compile(r"^\s*(\d{1,3})\s*°\s*(\d{1,2})\s*'\s*(\d{1,2})?\s*[\"”″']*\s*$")
LOC = re.compile(r"(\d{1,3})\s*°\s*(\d{1,2})\s*'\s*(\d{1,2})\s*[\"”″]?\s*N\.?\s*(\d{1,3})\s*°\s*(\d{1,2})\s*'"
                 r"\s*(\d{1,2})\s*[\"”″]?\s*W")


def dms(text: str):
    m = DMS.match(text)
    if not m:
        raise ValueError(f"도분초 표기가 아니다: {text!r}")
    d, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if mi >= 60 or se >= 60:
        raise ValueError(f"분 또는 초가 60 이상이다: {text!r}")
    return d + mi / 60.0 + se / 3600.0, m.group(3) is not None


def km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def cell_index(lat: float, lon: float):
    ky = math.floor(lat / 0.009)
    phi = math.radians((ky + 0.5) * 0.009)
    kx = math.floor(lon * math.cos(phi) / 0.009)
    return ky, kx


def block_id(lat: float, lon: float) -> int:
    return math.floor(lat / 0.5) * 100000 + math.floor(lon / 0.5)


def stats_of(vals):
    if not vals:
        return None
    return {"n": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals),
            "mean": round(statistics.fmean(vals), 2)}


def norm_tt(text: str) -> str:
    """'90 TT 14', '91 TT  A', '91TT15', '92TT 3' 를 '90TT14', '91TTA', '91TT15', '92TT3' 으로 맞춘다."""
    return re.sub(r"\s+", "", text).upper()


VALUE = re.compile(r"^([<>~])?\s*(\d{1,3})\s*([*?])?$")
MISSING = {"N.A.", "NA", "N.A", "N/A"}


def classify(text: str):
    t = text.strip()
    if t == "":
        return ("empty", None, None, None)
    if t in MISSING:
        return ("na", None, None, None)
    m = VALUE.match(t)
    if m:
        return ("value", int(m.group(2)), m.group(1) or "", m.group(3) or "")
    return ("text", None, None, None)


# ---------------------------------------------------------------- 지점 설명
DESC_KEYS = ["History", "Responsibility", "Location", "Elevation", "Slope", "Landform", "Soil", "Vegetation",
             "Snow", "Installations"]


def site_descriptions(events):
    """'Site:' 문단부터 다음 'Site:' 또는 'Active Layer DATA:' 앞까지를 한 지점 설명으로 묶는다."""
    descs, cur, key = [], None, None
    for kind, v in events:
        if kind != "para":
            continue
        t = v.strip()
        if not t:
            continue
        if t.startswith("Active Layer DATA"):
            break
        m = re.match(r"^Site:\s*(.+)$", t)
        if m:
            cur = {"header": re.sub(r"\s+", " ", m.group(1)).strip(), "extra": []}
            descs.append(cur)
            key = None
            continue
        if cur is None:
            continue
        m = re.match(r"^(" + "|".join(DESC_KEYS) + r"):\s*(.*)$", t)
        if m:
            key = m.group(1)
            cur[key] = re.sub(r"\s+", " ", m.group(2)).strip()
            continue
        if key == "Installations" and t.startswith("-"):
            cur[key] = cur.get(key, "") + " " + re.sub(r"\s+", " ", t)
            continue
        if t.startswith("(") and t.endswith(")"):
            cur["extra"].append(t)
            continue
        key = None if not t.startswith("-") else key
    out = []
    for d in descs:
        h = d["header"]
        calm = re.search(r"\((C\d+)\)", h)
        if calm:
            name = h[:calm.start()].strip()
            tt = re.match(r"^\s*(\d{2}\s*TT\s*\w+)\s+Installed", d.get("History", ""))
            tube = norm_tt(tt.group(1)) if tt else ""
            cid = calm.group(1)
        else:
            m = re.match(r"^(\d{2})\s*TT\s*(\w+)\s+(.*)$", h)
            if not m:
                raise ValueError(f"지점 머리 표기를 읽지 못했다: {h!r}")
            tube = f"{m.group(1)}TT{m.group(2)}".upper()
            name = m.group(3).strip()
            cid = ""
        loc = LOC.search(d.get("Location", ""))
        lat = lon = None
        if loc:
            g = [int(x) for x in loc.groups()]
            lat = g[0] + g[1] / 60.0 + g[2] / 3600.0
            lon = -(g[3] + g[4] / 60.0 + g[5] / 3600.0)
        out.append({"header": h, "tube_id": tube, "calm_id": cid, "name": name, "desc_lat": lat,
                    "desc_lon": lon, **{k.lower(): d.get(k, "") for k in DESC_KEYS},
                    "extra": " ".join(d["extra"])})
    return out


# ---------------------------------------------------------------- 대조 자료
def v3_reference(path: Path):
    """v3 표에서 좌표, 지역, 자료 구분, 라벨만 읽는다(북위 55-75, 서경 145-110). 표는 고치지 않는다."""
    ref = []
    if not path.exists():
        return ref
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lat, lon = float(r["lat"]), float(r["lon"])
            if 55.0 <= lat <= 75.0 and -145.0 <= lon <= -110.0:
                ref.append({"loc_id": r["loc_id"], "lat": lat, "lon": lon, "region": r["region"],
                            "source_id": r["source_id"], "alt_cm": float(r["alt_cm"])})
    return ref


def pangaea_tt_series(path: Path):
    """PANGAEA 972777 의 융해관 계열 이벤트의 연 값(원문 문자열)과 이벤트 좌표를 읽는다."""
    want = set(PANGAEA_TT_EVENT.values())
    ser, coords = defaultdict(dict), {}
    if not path.exists():
        return ser, coords
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("*/"))
    header = lines[k + 1].split("\t")
    ix = {h: i for i, h in enumerate(header)}
    for ln in lines[k + 2:]:
        c = ln.split("\t")
        if len(c) < len(header) or c[0] not in want:
            continue
        coords[c[0]] = (float(c[ix["Latitude"]]), float(c[ix["Longitude"]]))
        y, v = c[ix["Date/Time"]].strip(), c[ix["ALD [cm]"]].strip()
        if y.isdigit() and v:
            ser[c[0]][int(y)] = v
    return ser, coords


def other_sources_nearby(out_dir: Path, pts):
    """같은 폴더의 다른 자료원 점 자료 가운데 매켄지 주변(북위 60-71, 서경 137-120) 지점과의 거리를 본다.
    자료원 사이의 중복 판정은 v4 조립에서 한다. 여기서는 거리와 체비쇼프 0.01도 쌍만 적는다."""
    res = {}
    for p in sorted(out_dir.glob("*_points.csv")):
        if p.name == f"{SRC_ID}_points.csv":
            continue
        try:
            sites = {}
            with open(p, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        lat, lon = float(r["lat"]), float(r["lon"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    if 60.0 <= lat <= 71.0 and -137.0 <= lon <= -120.0:
                        sites[r["site_id"]] = (lat, lon, r.get("site_name", ""))
        except Exception as e:  # 다른 작업이 쓰는 중인 파일일 수 있다
            res[p.name] = {"error": str(e)}
            continue
        if not sites:
            continue
        best, pairs = None, []
        for sid, (lat, lon, name) in sites.items():
            for q in pts:
                d = km(lat, lon, q["lat"], q["lon"])
                if max(abs(lat - q["lat"]), abs(lon - q["lon"])) <= 0.01:
                    pairs.append({"site_id": q["site_id"], "other_site_id": sid, "other_site_name": name,
                                  "distance_km": round(d, 3)})
                if best is None or d < best[0]:
                    best = (d, sid, name, q["site_id"])
        res[p.name] = {"n_sites_in_bbox": len(sites), "min_distance_km": round(best[0], 3),
                       "nearest_other_site_id": best[1], "nearest_other_site_name": best[2],
                       "nearest_site_id": best[3], "pairs_chebyshev_0p01": pairs}
    return res


# ---------------------------------------------------------------- 본체
def main():
    ap = argparse.ArgumentParser(description="ext_labels 파서: nsidc_ggd353_thawtube")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--v3", type=Path, default=V3_PATH)
    ap.add_argument("--pangaea", type=Path, default=PANGAEA_PATH)
    args = ap.parse_args()

    rtf_path = args.raw_dir / "ggd353data_v6.rtf"
    events = rtf_events(rtf_path.read_bytes())

    # ---- 표 찾기
    i0 = next(i for i, (k, v) in enumerate(events) if k == "para" and v.strip().startswith("Active Layer DATA"))
    section, header, table = None, None, []
    foot = []
    for k, v in events[i0 + 1:]:
        if k == "para":
            t = v.strip()
            if t in ("CALM", "Non-CALM"):
                section = t
            elif t and table:
                foot.append(t)
            continue
        cells = [re.sub(r"\s+", " ", c).strip() for c in v]
        if cells and cells[0] == "Site":
            if [c for c in cells[:3]] != ["Site", "Latitude", "Longitude"]:
                raise ValueError(f"머리글이 예상과 다르다: {cells[:3]}")
            yrs = [int(c) for c in cells[3:]]
            if yrs != YEARS_EXPECTED:
                raise ValueError(f"연도 머리글이 예상과 다르다: {yrs}")
            header = cells
            continue
        if header is None or section is None:
            raise ValueError("머리글 전에 자료 행이 있다")
        if len(cells) != len(header):
            raise ValueError(f"칸 수가 머리글과 다르다({len(cells)} != {len(header)}): {cells[:3]}")
        table.append({"section": section, "cells": cells})
    footnote = " ".join(foot)

    descs = site_descriptions(events)
    by_calm = {d["calm_id"]: d for d in descs if d["calm_id"]}
    by_tube = {}
    for d in descs:
        if not d["calm_id"]:
            if d["tube_id"] in by_tube:
                raise ValueError(f"지점 설명의 관 번호가 겹친다: {d['tube_id']}")
            by_tube[d["tube_id"]] = d

    # ---- 대조 자료
    ref = v3_reference(args.v3)
    pser, pcoords = pangaea_tt_series(args.pangaea)

    # ---- 행 만들기
    points, cell_rows, site_rows = [], [], []
    kinds = Counter()
    text_cells = []
    seen_site = set()
    for tr in table:
        c = tr["cells"]
        label = c[0]
        if tr["section"] == "CALM":
            cid = TABLE_TO_CALM[label]
            d = by_calm[cid]
            site_id = d["tube_id"]
            site_name = f"{d['name']} ({cid})"
        else:
            site_id = norm_tt(label)
            d = by_tube[site_id]
            site_name = d["header"]
            cid = ""
        if site_id in seen_site:
            raise ValueError(f"표의 지점이 겹친다: {site_id}")
        seen_site.add(site_id)
        lat, lat_sec = dms(c[1])
        lon_abs, lon_sec = dms(c[2])
        lon = -lon_abs
        if not (lat_sec and lon_sec):
            raise ValueError(f"초 자리가 없는 좌표: {label}")
        if not (BBOX["lat_min"] <= lat <= BBOX["lat_max"] and BBOX["lon_min"] <= lon <= BBOX["lon_max"]):
            raise ValueError(f"범위 밖 좌표: {label} {lat} {lon}")
        dd = km(lat, lon, d["desc_lat"], d["desc_lon"]) if d["desc_lat"] is not None else None
        coord_bad = dd is not None and dd > COORD_DISAGREE_KM
        # 방문 공백(not visited A - B)
        gap = re.search(r"not visited (\d{4})\s*-\s*(\d{4})", d.get("history", ""))
        gap = (int(gap.group(1)), int(gap.group(2))) if gap else None
        burned_from = None
        if re.search(r"burned in (\d{4}) fire", d.get("extra", "") + " " + d.get("vegetation", "")):
            burned_from = int(re.search(r"burned in (\d{4}) fire",
                                        d.get("extra", "") + " " + d.get("vegetation", "")).group(1))
        # dup_of
        dup = ""
        pev = PANGAEA_TT_EVENT.get(cid, "")
        if cid:
            if pev in pcoords:
                plat, plon = pcoords[pev]
                hits = [r for r in ref if max(abs(r["lat"] - plat), abs(r["lon"] - plon)) <= 0.00015]
                if hits:
                    dup = ";".join(f"v3:{r['loc_id']}" for r in hits)
            if not dup:
                dup = DUP_FALLBACK.get(cid, "")
            if not dup:
                raise ValueError(f"CALM 지점의 dup_of 를 정하지 못했다: {cid}")
        site_rows.append({"site_id": site_id, "site_name": site_name, "section": tr["section"], "calm_id": cid,
                          "lat": lat, "lon": lon, "desc_lat": d["desc_lat"], "desc_lon": d["desc_lon"],
                          "desc_dist_km": dd, "coord_flag": coord_bad, "dup_of": dup, "pangaea_event": pev,
                          "history": d.get("history", ""), "landform": d.get("landform", ""),
                          "soil": d.get("soil", ""), "vegetation": d.get("vegetation", ""),
                          "burned_from": burned_from, "visit_gap": gap})
        prev_text = False
        for j, y in enumerate(YEARS_EXPECTED):
            raw = c[3 + j]
            kind, val, pre, post = classify(raw)
            # 앞 칸이 글이고 이 칸이 연도 네 자리면 글의 이어짐('Fire in' '1995')으로 본다
            if prev_text and re.fullmatch(r"(19|20)\d\d", raw.strip()):
                kind, val = "text", None
            if kind == "value" and val is not None and val >= 1000:
                kind, val = "text", None
            prev_text = kind == "text"
            kinds[kind] += 1
            cell_rows.append({"section": tr["section"], "site_id": site_id, "table_label": label,
                              "lat_text": c[1], "lon_text": c[2], "year": y, "cell_text": raw, "kind": kind})
            if kind == "text":
                text_cells.append({"site_id": site_id, "year": y, "text": raw})
            if kind != "value":
                continue
            notes = [f"table={tr['section']}", f"tube_id={site_id}", f"raw={raw}"]
            if cid:
                notes.append(f"calm_id={cid}")
                notes.append(f"pangaea972777_event={pev}")
            method, label_def, eos_basis = "thaw_tube", "direct_eos", "protocol"
            rc = 0
            if pre == ">":
                rc = 1
                notes.append("value_bound=lower(thaw below tube range)")
            elif pre == "<":
                notes.append("value_bound=upper(max subsidence not available; min subsidence used)")
            elif pre == "~":
                notes.append("approx=1")
            if post == "*":
                method, label_def, eos_basis = "other", "direct_dated", "dataset_statement"
                notes.append("late_year_obs=1(value based on late year observation rather than thaw tube)")
            elif post == "?":
                notes.append("uncertain=1")
            disturbed, dtype = 0, ""
            if burned_from is not None and y >= burned_from:
                disturbed, dtype = 1, "burned"
                notes.append(f"site burned in {burned_from} fire")
            qc = []
            if not (0 < val <= 600):
                qc.append("range")
            if coord_bad:
                qc.append("coord")
                notes.append(f"desc_coord=({d['desc_lat']:.6f},{d['desc_lon']:.6f}) dist_km={dd:.2f}")
            if gap and gap[0] <= y < gap[1]:
                qc.append("date")
                notes.append(f"history: not visited {gap[0]}-{gap[1]}")
            if y < 1990:
                qc.append("year_pre1990")
            points.append({
                "src_id": SRC_ID, "site_id": site_id, "site_name": site_name,
                "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "year": y, "month": "", "alt_cm": f"{float(val):.1f}",
                "method": method, "label_def": label_def, "n_obs": 1, "country": COUNTRY, "macro": MACRO,
                "citation": CITATION, "license": LICENSE, "subunit": SUBUNIT, "date": "", "eos_basis": eos_basis,
                "value_kind": "annual_value", "year_min": "", "year_max": "", "alt_sd_cm": "",
                "coord_prec_deg": PREC_SECOND, "disturbed": disturbed, "disturb_type": dtype,
                "right_censored": rc, "orig_source": ORIG_SOURCE, "dup_of": dup,
                "qc_flag": ";".join(qc) if qc else "ok", "notes": "; ".join(notes),
                "_lat": lat, "_lon": lon, "_alt": float(val), "_pre": pre, "_post": post, "_cid": cid,
            })

    # ---- 쓰기
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.out_dir / f"{SRC_ID}_points.csv"
    with open(points_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for p in points:
            w.writerow(p)
    cells_path = args.raw_dir / "ggd353_table_cells.csv"
    with open(cells_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in cell_rows:
            w.writerow(r)
    desc_path = args.raw_dir / "ggd353_site_descriptions.csv"
    dcols = ["header", "tube_id", "calm_id", "name", "desc_lat", "desc_lon"] + [k.lower() for k in DESC_KEYS] + ["extra"]
    with open(desc_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dcols, lineterminator="\n")
        w.writeheader()
        for d in descs:
            w.writerow({k: d.get(k, "") for k in dcols})

    # ---- 요약
    alts = [p["_alt"] for p in points]
    years = [p["year"] for p in points]
    f4 = [r for r in ref if r["source_id"] == "F4_direct"]
    oth = [r for r in ref if r["source_id"] != "F4_direct"]

    def eligible(p):
        return (p["label_def"] == "direct_eos" and p["right_censored"] == 0 and p["disturbed"] == 0
                and p["qc_flag"] == "ok" and p["year"] >= 1990)

    elig = [p for p in points if eligible(p)]
    # 셀 미리 보기(v4 조립 전 참고 값). 위치별 다년 평균 -> 셀 평균
    v3_f4_blocks = set(block_id(r["lat"], r["lon"]) for r in f4)

    def preview(rows):
        by_site = defaultdict(list)
        for p in rows:
            by_site[p["site_id"]].append(p)
        site_mean = {s: (ps[0]["_lat"], ps[0]["_lon"], statistics.fmean(q["_alt"] for q in ps), len(ps))
                     for s, ps in by_site.items()}
        cells = defaultdict(list)
        for s, (la, lo, mu, n) in site_mean.items():
            cells[cell_index(la, lo)].append((s, la, lo, mu, n))
        out = []
        for ck, ss in sorted(cells.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
            clat = statistics.fmean(x[1] for x in ss)
            clon = statistics.fmean(x[2] for x in ss)
            hit_f4 = [r for r in f4 if max(abs(r["lat"] - clat), abs(r["lon"] - clon)) <= 0.01]
            hit_oth = [r for r in oth if max(abs(r["lat"] - clat), abs(r["lon"] - clon)) <= 0.01]
            calm = [x[0] for x in ss if site_rows_by(site_rows, x[0])["calm_id"]]
            out.append({"cell": list(ck), "lat": round(clat, 6), "lon": round(clon, 6),
                        "sites": [x[0] for x in ss], "calm_sites": calm,
                        "alt_cm": round(statistics.fmean(x[3] for x in ss), 2),
                        "n_site_years": sum(x[4] for x in ss),
                        "n_dup_v3_f4": len(hit_f4),
                        "dup_v3_f4_first5": [f"v3:{r['loc_id']}({r['region']})" for r in hit_f4[:5]],
                        "near_v3_other": [f"v3:{r['loc_id']}({r['region']},{r['source_id']})" for r in hit_oth],
                        "block": block_id(clat, clon)})
        new = [c for c in out if c["n_dup_v3_f4"] == 0]
        blocks = set(c["block"] for c in new)
        summ = {"n_sites": len(by_site), "n_cells_1km": len(out),
                "n_cells_dup_v3_f4_chebyshev_0p01": sum(1 for c in out if c["n_dup_v3_f4"]),
                "n_new_cells": len(new),
                "n_new_cells_with_calm_site": sum(1 for c in new if c["calm_sites"]),
                "n_blocks_0p5deg_new_cells": len(blocks),
                "n_blocks_not_in_v3_f4": len(blocks - v3_f4_blocks),
                "new_cell_alt_cm_stats": stats_of([c["alt_cm"] for c in new])}
        return out, new, blocks, summ

    crow, new_cells, blocks_new, summ_all = preview(elig)
    crow_nd, new_cells_nd, blocks_nd, summ_nd = preview([p for p in elig if not p["dup_of"]])

    # 점 단위 v3 거리
    nearest = []
    for s in site_rows:
        best = None
        for r in f4:
            dkm = km(s["lat"], s["lon"], r["lat"], r["lon"])
            if best is None or dkm < best[0]:
                best = (dkm, r)
        nearest.append((s["site_id"], best))
    within = {f"n_sites_within_{t}km_of_v3_f4": sum(1 for _, b in nearest if b and b[0] <= t) for t in (1, 5, 25)}

    # PANGAEA 융해관 계열과 대조
    calm_check = {}
    for s in site_rows:
        if not s["calm_id"]:
            continue
        ev = s["pangaea_event"]
        ours = {r["year"]: r["cell_text"] for r in cell_rows if r["site_id"] == s["site_id"] and r["kind"] == "value"}
        theirs = pser.get(ev, {})
        common = sorted(set(ours) & set(theirs))
        same = [y for y in common if ours[y].replace(" ", "") == theirs[y].replace(" ", "")]
        diff = {str(y): {"ggd353": ours[y], "pangaea": theirs[y]} for y in common if y not in same}
        v3hit = [r for r in ref if s["dup_of"].startswith("v3:") and f"v3:{r['loc_id']}" in s["dup_of"].split(";")]
        uncens = [p["_alt"] for p in points if p["site_id"] == s["site_id"] and eligible(p)]
        calm_check[s["calm_id"]] = {
            "site_id": s["site_id"], "pangaea_event": ev,
            "pangaea_event_coord": list(pcoords.get(ev, ())),
            "n_years_ggd353": len(ours), "n_years_pangaea": len(theirs), "n_common": len(common),
            "n_identical_text": len(same), "differences": diff,
            "years_only_ggd353": sorted(set(ours) - set(theirs)),
            "years_only_pangaea_1991_2007": sorted(y for y in set(theirs) - set(ours) if 1991 <= y <= 2007),
            "dup_of": s["dup_of"],
            "v3_alt_cm": [round(r["alt_cm"], 2) for r in v3hit],
            "ggd353_mean_eligible_cm": round(statistics.fmean(uncens), 2) if uncens else None,
            "ggd353_n_eligible": len(uncens),
        }

    near_src = other_sources_nearby(args.out_dir, [{"site_id": s["site_id"], "lat": s["lat"], "lon": s["lon"]}
                                                   for s in site_rows])

    files_meta = []
    for name in DOC_FILES:
        p = args.raw_dir / name
        if p.exists():
            files_meta.append({"name": name, "size_bytes": p.stat().st_size, "sha256": sha256_of(p),
                               "url": FILE_URLS[name],
                               "role": "parsed" if name == "ggd353data_v6.rtf" else "documentation"})
    derived_meta = [
        {"name": cells_path.name, "size_bytes": cells_path.stat().st_size, "sha256": sha256_of(cells_path),
         "note": "이 파서가 만든 표다. RTF 'Active Layer DATA' 표의 칸 원문(지점 55 x 연도 17 = 935칸)과 칸 분류"},
        {"name": desc_path.name, "size_bytes": desc_path.stat().st_size, "sha256": sha256_of(desc_path),
         "note": "이 파서가 만든 표다. RTF 지점 설명 56개의 항목별 원문"},
    ]

    def cnt(key):
        return dict(sorted(Counter(str(p[key]) for p in points).items()))

    flag_counts = Counter()
    for p in points:
        for fl in p["qc_flag"].split(";"):
            flag_counts[fl] += 1
    mark_counts = {"gt_lower_bound(>)": sum(1 for p in points if p["_pre"] == ">"),
                   "lt_upper_bound(<)": sum(1 for p in points if p["_pre"] == "<"),
                   "approx(~)": sum(1 for p in points if p["_pre"] == "~"),
                   "late_year_obs(*)": sum(1 for p in points if p["_post"] == "*"),
                   "uncertain(?)": sum(1 for p in points if p["_post"] == "?")}
    coord_rows = [{"site_id": s["site_id"], "site_name": s["site_name"], "table_lat": round(s["lat"], 6),
                   "table_lon": round(s["lon"], 6),
                   "desc_lat": round(s["desc_lat"], 6) if s["desc_lat"] is not None else None,
                   "desc_lon": round(s["desc_lon"], 6) if s["desc_lon"] is not None else None,
                   "dist_km": round(s["desc_dist_km"], 3) if s["desc_dist_km"] is not None else None}
                  for s in site_rows]
    coord_rows.sort(key=lambda r: -(r["dist_km"] or 0))

    n_elig_sites = len(set(p["site_id"] for p in elig))
    meta = {
        "src_id": SRC_ID,
        "name": ("NSIDC GGD353 캐나다 매켄지 계곡 융해관 관측망(Active Layer Monitoring, Arctic and Subarctic "
                 "Canada, Version 6)"),
        "schema_version": "1.0",
        "url": URL,
        "doi": DOI,
        "download_dir": FTP_DIR,
        "accessed": ACCESSED,
        "files": files_meta,
        "derived_files_in_raw_dir": derived_meta,
        "license": LICENSE,
        "license_evidence": (
            "자료 누리집과 사용 안내서는 'As a condition of using these data, you must cite the use of this data "
            "set' 라는 인용 조건을 적었다. 누리집 개요와 RTF 끝의 문구는 'This data is the property of the "
            "people of Canada and the responsibility of the Geological Survey of Canada. Please consult prior to "
            "use in publication. If published, adequate acknowledgment is expected.' 이다. 이용 허락 표시(CC 등)와 "
            "재배포 조건은 누리집, 사용 안내서, README.txt, DIF 메타 기록(Use_Constraints, Access_Constraints "
            "요소 없음)에서 찾지 못했다. 재배포 가능 여부를 확인하지 못했으므로 점 자료를 커밋하지 않는다"
            "(data/processed/ext_labels/.gitignore 에 이미 들어 있다). 논문에 쓰기 전에 GSC 담당자와 협의가 "
            "필요하다는 문구가 있다"),
        "access_note": "익명 FTP 다. 계정이 필요 없다. 2026-09-29 23:34 KST 의 FTP 폴더 목록은 README.txt(766), "
                       "ggd353data_v6.rtf(1,131,795) 두 파일이고 로컬 사본과 크기가 같다",
        "citation": CITATION,
        "citation_user_guide": ("Nixon, F. Mark 2003, updated January 2009. Active Layer Monitoring, Arctic and "
                                "Subarctic Canada, Version 6. [Indicate subset used]. Boulder, Colorado USA. NSIDC: "
                                "National Snow and Ice Data Center. doi: https://doi.org/10.7265/7m84-k262"),
        "orig_source": ORIG_SOURCE,
        "parser": "scripts/1_data_prep/parse_ext_nsidc_ggd353_thawtube.py",
        "parser_sha256": sha256_of(Path(__file__).resolve()),
        "parser_git_commit": f"미커밋(실행 시점 HEAD {git_head(ROOT)})",
        "points_file": f"data/processed/ext_labels/{SRC_ID}_points.csv",
        "points_sha256": sha256_of(points_path),
        "n_rows_raw": {"table_rows": len(table), "table_rows_calm": sum(1 for t in table if t["section"] == "CALM"),
                       "table_rows_non_calm": sum(1 for t in table if t["section"] == "Non-CALM"),
                       "table_cells": len(cell_rows), "cells_by_kind": dict(kinds),
                       "site_descriptions": len(descs)},
        "n_rows_points": len(points),
        "n_sites": len(site_rows),
        "n_sites_calm": sum(1 for s in site_rows if s["calm_id"]),
        "n_by_label_def": cnt("label_def"),
        "n_by_method": cnt("method"),
        "n_by_eos_basis": cnt("eos_basis"),
        "n_by_macro": cnt("macro"),
        "n_by_subunit": cnt("subunit"),
        "n_by_qc_flag": dict(sorted(flag_counts.items())),
        "n_by_year": cnt("year"),
        "n_right_censored": sum(1 for p in points if p["right_censored"] == 1),
        "n_disturbed": sum(1 for p in points if p["disturbed"] == 1),
        "n_value_marks": mark_counts,
        "year_range": [min(years), max(years)],
        "alt_cm_stats_all_rows": stats_of(alts),
        "alt_cm_stats_eligible_rows": stats_of([p["_alt"] for p in elig]),
        "alt_cm_stats_by_label_def": {k: stats_of([p["_alt"] for p in points if p["label_def"] == k])
                                      for k in sorted(set(p["label_def"] for p in points))},
        "n_excluded_by_reason": {
            "_설명": ("점 자료에서 지운 값은 없다. 값이 없는 칸(결측 표기, 빈 칸, 글)은 행을 만들지 않았다. "
                     "target_* 는 대상 라벨 규칙(6B.3, 6B.4)을 통과하지 못하는 행의 수이고 사유는 겹친다"),
            "cells_na_mark": kinds.get("na", 0),
            "cells_empty": kinds.get("empty", 0),
            "cells_text_not_value": kinds.get("text", 0),
            "text_cells": text_cells,
            "dropped_from_points": 0,
            "target_label_def_not_direct_eos": sum(1 for p in points if p["label_def"] != "direct_eos"),
            "target_right_censored": sum(1 for p in points if p["right_censored"] == 1),
            "target_disturbed_burned": sum(1 for p in points if p["disturbed"] == 1),
            "target_qc_coord": flag_counts.get("coord", 0),
            "target_qc_date": flag_counts.get("date", 0),
            "target_qc_range": flag_counts.get("range", 0),
            "target_year_pre1990": flag_counts.get("year_pre1990", 0),
            "n_eligible_rows": len(elig),
            "n_eligible_sites": n_elig_sites,
        },
        "label_semantics": {
            "value": "연 최대 활동층 두께(cm, 정수, ±1 cm). 표의 연도는 융해 계절의 연도다",
            "definition_in_source": ("The active layer is defined as the thaw recorded in the thaw tube, minus "
                                     "the height of the tube above the ground surface at maximum surface "
                                     "subsidence, assumed to occur about the time of maximum thaw."),
            "subsidence_choice": ("파일의 표에는 침하 보정을 마친 ALT 하나만 있다. 보정 전 융해 깊이, 융기, 침하 값은 "
                                  "이 파일에 없다(누리집은 관측 항목으로 적었으나 공개 표에는 없다). 자료 설명의 "
                                  "ALT 정의에 맞는 값이므로 그대로 썼다. 최대 침하가 없던 해는 '<' 로 표시되어 "
                                  "참값보다 크거나 같다"),
            "footnote_in_source": footnote,
            "eos": ("융해관의 표지 구슬은 융해기 동안 내려가 가을 재동결 때 최대 깊이에 갇힌다(자료 설명). 따라서 "
                    "연 최대값이고 eos_basis = protocol 이다. 판독 날짜는 파일에 없어 month 와 date 는 비웠다"),
            "late_year_obs": ("'*' 값은 융해관 측정이 아니라 늦은 시기 관측에 근거한다. 관측 방법과 날짜가 없다. "
                              "누리집 개요는 관측 항목에 'current depth of thaw' 를 적었으나 '*' 값과의 관계는 "
                              "확인하지 못했다. direct_dated 로 두었다"),
        },
        "table_notes": {
            "92TT12": ("1994, 1995 칸은 'Fire in', '1995' 라는 글이다. 값으로 읽지 않았다. 지점 설명 '(site burned "
                       "in 1995 fire)', 이력 'destroyed 1995, reestablished 1996, lost 1996'. 1995년 이후 값에 "
                       "disturbed = 1, burned"),
            "91TT18": ("이력 'not visited 1999 - 2005' 인데 2000년 칸에 89 가 있다. 2000년 값에 qc_flag date 를 "
                       "붙였다. 2005년 값(79*)은 늦은 시기 관측 표시가 있어 공백의 끝 해로 보고 date 를 붙이지 않았다"),
            "95TT1": "지점 설명(Saline River)만 있고 표에 행이 없다(이력 'Installed 1995, no record')",
            "missing_then_value": ("결측(N.A.) 다음 해의 값은 표지 구슬이 두 해 이상의 최대를 기록했을 가능성이 있다. "
                                   "자료 설명에 이 경우의 처리가 없어 표시하지 않았다"),
        },
        "coordinate_conversion": (
            "표의 위도, 경도 칸은 도분초다(예: 69°43'11\"). 십진 도 = 도 + 분/60 + 초/3600, 소수 6자리. 반구 표기는 "
            "표에 없고 지점 설명은 N., W. 이다. 경도는 서경이라 음수로 적었다. 측지계 표기는 없고 변환하지 않았다. "
            "두 칸(90TT12 경도 134°21'44, 91TT19 위도 65°39'41)은 초 기호가 빠져 있으나 초 자리로 읽었다"),
        "coordinate_source_choice": (
            "좌표는 값과 같은 행에 있는 표의 좌표를 썼다. 지점 설명(Location)의 좌표와 다른 지점이 있다. "
            f"{COORD_DISAGREE_KM} km(1 km 셀 크기)를 넘게 다르면 qc_flag 에 coord 를 붙였다. CALM 지점은 표 "
            "좌표가 PANGAEA 972777 이벤트 좌표와 같다(North Head 69.71972, -134.4619 는 표의 69°43'11\", "
            "134°27'43\" 와 같고 설명의 69°43'15\", 134°27'41\" 와 다르다)"),
        "coordinate_check": {
            "bbox_from_landing_page": "북 69.717, 남 61.883, 서 -134.95, 동 -121.6(누리집 표기)",
            "lat_range": [round(min(s["lat"] for s in site_rows), 6), round(max(s["lat"] for s in site_rows), 6)],
            "lon_range": [round(min(s["lon"] for s in site_rows), 6), round(max(s["lon"] for s in site_rows), 6)],
            "note": ("표의 남쪽 끝(92TT4, 92TT5 Manners Creek, 61°46') 과 서쪽 끝(91TT F Kendel Island, 135°20') 은 "
                     "누리집의 범위 표기(남 61.883, 서 -134.95) 밖이다. 사용 안내서의 범위는 서쪽 135°20' 로 Kendel "
                     "Island 를 포함하나 남쪽 61°53' 은 Manners Creek 을 포함하지 않는다. 지점 설명의 Manners Creek "
                     "좌표(61°46'12\", 61°45'56\")도 61°53' 보다 남쪽이다. 범위 표기는 포트심프슨 CALM 지점(61°53')을 "
                     "남쪽 끝으로 적었다. 좌표 부호와 단위의 점검은 BBOX(북위 61-70.5, 서경 136-120.5)로 했다"),
            "n_sites_coord_flag": sum(1 for s in site_rows if s["coord_flag"]),
            "table_vs_description": coord_rows,
        },
        "coord_prec_rule": "모든 표 좌표가 초 단위라 coord_prec_deg = 0.0003(1초는 0.000278도)",
        "units": "표의 값은 cm 정수다. 바꾸지 않았다",
        "calm_duplicates": {
            "_설명": ("CALM 10개 지점의 표 값을 PANGAEA 972777 의 융해관 계열 이벤트(C*A, C13)와 원문 문자열로 대조했다. "
                     "dup_of 는 같은 CALM 좌표의 v3 행(v3:<loc_id>)이고, v3 에 행이 없는 C3 은 같은 지점의 탐침 격자를 "
                     "담은 calm_pangaea_v4add:CALM_C3B 다. v3 행의 값이 융해관 이벤트에서 왔는지 탐침 격자 "
                     "이벤트에서 왔는지는 이 파서에서 확인하지 않았다"),
            "by_calm_id": calm_check,
        },
        "cell_preview": {
            "_설명": ("v4 조립 전의 참고 값이다. 대상 규칙 통과 행(direct_eos, 검열 아님, 교란 아님, qc ok, 1990년 이후)만 "
                     "썼다. '<' 값(최댓값)과 '~' 값은 포함했다. 셀 값은 위치별 다년 평균의 평균이다. "
                     "all_eligible 은 CALM 10개 지점(dup_of 있음)을 포함한 것이고 excluding_dup_of 는 뺀 것이다. "
                     "C13(92TT10)은 v3 행(63.5, -123.7, 소수 1자리 좌표)과 3.8 km 떨어져 체비쇼프 규칙으로는 빠지지 "
                     "않으나 같은 CALM 지점이다. C3(90TT13)은 calm_pangaea_v4add 의 CALM_C3B 와 같은 좌표다"),
            "cell_rule": "ky = floor(lat/0.009), kx = floor(lon*cos(phi)/0.009), phi = (ky+0.5)*0.009 도",
            "all_eligible": summ_all,
            "excluding_dup_of": summ_nd,
            "cells_all_eligible": crow,
        },
        "v3_check": {
            "bbox_compared": "55 <= lat <= 75, -145 <= lon <= -110",
            "n_ref_rows_compared": len(ref),
            "n_ref_f4_direct": len(f4),
            "ref_regions": dict(Counter(r["region"] for r in ref)),
            **within,
            "nearest_v3_f4_by_site": {sid: {"loc_id": b[1]["loc_id"], "region": b[1]["region"],
                                            "km": round(b[0], 3)} for sid, b in nearest if b},
        },
        "other_ext_sources_nearby": {
            "_설명": ("실행 시점에 같은 폴더에 있던 다른 자료원 점 자료 가운데 북위 60-71, 서경 137-120 안의 지점과의 "
                     "거리다. 자료원 사이의 중복 판정은 v4 조립에서 한다"),
            "by_file": near_src,
        },
        "field_choices": {
            "site_id": "관 번호(예: 90TT14, 91TTA). CALM 지점은 지점 설명 이력의 관 번호(C3 = 90TT13, C4 = 91TTC 등)",
            "site_name": "비CALM 지점은 지점 설명 머리('90TT14 North Head'), CALM 지점은 '이름 (Cn)'",
            "month, date": "빈 칸. 판독 날짜와 최대 융해 날짜가 파일에 없다",
            "n_obs": "1(관 하나)",
            "subunit": "Mackenzie(전 지점)",
            "disturbed": ("92TT12 의 1995년 이후만 1(burned). 91TT F Kendel Island 의 토양 설명 'under shallow "
                          "water' 는 습지의 얕은 물로 보고 수체 바닥으로 표시하지 않았다(notes 없음, 메타에만 적음)"),
        },
        "unverified_items": [
            "이용 약관과 재배포 조건. 인용 조건과 '논문에 쓰기 전에 협의' 문구만 확인했다",
            "'*' 값(늦은 시기 관측)의 관측 방법과 날짜",
            "'~', '?' 표기의 뜻. 자료 설명에 없다",
            "좌표의 측지계. 파일과 사용 안내서에 없다. 변환하지 않았다",
            "표 좌표와 지점 설명 좌표 가운데 어느 쪽이 맞는지. 1 km 넘게 다른 지점은 qc_flag coord 로 표시했다",
            "결측 다음 해의 값이 두 해 이상의 최대인지",
            "91TT18 의 2000년 값이 방문 공백(1999-2005) 안에 있는 이유",
            "관을 다시 설치한 지점(90TT1, 90TT12, 92TT12, 92TT8, 91TT16 등)의 새 관 위치가 원래 위치와 같은지",
            "91TT F Kendel Island('under shallow water')를 수체 바닥 교란으로 볼지",
            "v3 의 CALM 좌표 행(C4-C15)이 융해관 이벤트 값인지 탐침 격자 값인지",
        ],
    }
    meta_path = args.out_dir / f"{SRC_ID}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"지점 {len(site_rows)}개(CALM {meta['n_sites_calm']}), 행 {len(points)}개, "
          f"연도 {meta['year_range']}, label_def {meta['n_by_label_def']}, qc {meta['n_by_qc_flag']}")
    print(f"셀(전체) {summ_all}")
    print(f"셀(dup_of 제외) {summ_nd}")


def site_rows_by(site_rows, sid):
    for s in site_rows:
        if s["site_id"] == sid:
            return s
    raise KeyError(sid)


if __name__ == "__main__":
    main()
