"""calm_web_subsites: CALM 누리집의 지점별 원자료(Abisko S2, Kapp Linné S1 하위 지점)를 표준 점 자료로 만든다.

근거
----
계획서 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B 절(LGD, 개정 10), 형식 정의
`data/processed/ext_labels/_schema.json`(버전 1.0).

입력(data/raw/calm_web_subsites/, 출처와 받은 날짜는 같은 폴더의 SOURCE.md)
----
s02_abisko_area/S2_alt_abisko_78_23.xls  시트 'data'(1978-2023, 하위 지점 12열), 'Sheet2'(2002년 11월판, 대조용)
s02_abisko_area/site_descr.txt           하위 지점 AB1-AB11 의 좌표와 표고(Akerman 1998, CAPS 1.0)
s01_kappe_linne/s01_alt_2011.xls         시트 'data'(1972-2011, 하위 지점 8열)
s01_kappe_linne/site_descr.txt           하위 지점 AL1-AL20 의 좌표와 표고, 표본 추출 방식
s01_kappe_linne/kapp72_00.dat            1972-2000년판(쉼표 구분 텍스트, 대조용)
qc_dem/, data/raw/dem/                   Copernicus DEM GLO-30 타일(좌표 점검용, 읽기 전용)
data/raw/gtnp/sites.json                 GTN-P 시추공 좌표(좌표 점검용, 읽기 전용)
data/processed/fidelity_base_v3.csv      읽기 전용. 좌표와 source_id 만 읽어 v3 와의 관계를 기록한다

규칙
----
1. 값의 단위. 두 파일의 하위 지점 열은 m 다(값의 범위 0.2-1.5). 100 을 곱해 cm 로 둔다.
   s02_files.txt 는 cm 라고 적었으나 값과 맞지 않는다. 파서는 값의 범위로 단위를 확인한다.
2. 결측 부호. Abisko 는 문자열 'nan', 'ND', 'x' 와 빈 칸, Kapp Linné 는 'nd' 다. 값이 없으므로
   점 자료에 넣지 않고 부호별 수를 메타에 적는다. 범례의 'NO'(영구동토 소멸)는 자료 칸에 없다.
3. 좌표. 설명문의 표기 DDoMM'HH''(예: 68o20'88''N)는 도, 분, 분의 백분위다.
   십진 도 = DD + (MM + HH/100)/60. 셋째 자리에 60 이상인 값(88, 76, 80, 61, 71, 82, 91, 84)이 있어
   초가 아니다. 근거와 대조 결과는 메타의 coordinate_conversion 에 적는다.
4. Bergfors 열은 설명문에 좌표가 없어 점 자료에 넣지 않는다. 'Narkevaere' 열은 Sheet2 의
   'AB11; Narkervare' 열과 같은 계열임을 값으로 확인하고 AB11 의 좌표를 쓴다.
5. 라벨. method = probe, label_def = direct_eos, eos_basis = protocol(CALM 규약), value_kind = annual_value.
   파일에 관측 날짜가 없으므로 month 와 date 는 빈 칸이다.
6. 연도. 1990년 이전 연 값은 지우지 않고 qc_flag 에 year_pre1990 을 붙인다. 주 집합은 qc_flag = ok 다.
7. 범위. 0 < alt_cm <= 600 밖이면 qc_flag 에 range 를 붙인다.
8. 좌표 점검. 변환한 좌표의 DEM 표고가 설명문 표고와 100 m 넘게 다르거나, DEM 5 x 5 창이 평탄한
   수면(범위 0.01 m 미만)이면 그 하위 지점의 모든 행에 qc_flag = coord 를 붙인다. 값은 지우지 않는다.
9. 우측 절단. 원자료에 '>' 표기가 없어 right_censored 는 모두 0 이다. 같은 값이 3년 넘게 이어지는
   계열은 notes 에 표시하고 메타의 확인하지 못한 항목에 적는다.
10. n_obs. Kapp Linné 설명문은 1972-1973년에 10 x 10 m 안 10점, 1974-1993년에 100 x 100 m 안 25점 무작위 표본,
    1994년부터 ITEX 방식이라 적었다. 1972-1973년 행은 10, AL5 를 뺀 하위 지점의 1974-1993년 행은 25,
    그 밖은 1(측점 수 미확인)이다. Abisko 는 모두 1 이다.

산출
----
data/processed/ext_labels/calm_web_subsites_points.csv   표준 점 자료(30열)
data/processed/ext_labels/calm_web_subsites_meta.json    메타
data/raw/calm_web_subsites/derived/calm_web_subsites_raw_long.csv  원자료의 모든 칸(연도 x 하위 지점)과 처리 결과
data/raw/calm_web_subsites/derived/calm_web_subsites_site_qc.csv   하위 지점별 좌표 점검 표

실행(ROOT): OMP_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_calm_web_subsites.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import xlrd

SRC_ID = "calm_web_subsites"
SRC_NAME = "CALM 누리집 지점별 원자료(Abisko S2 하위 지점, Kapp Linné S1 하위 지점)"
URL = "https://www2.gwu.edu/~calm/data/north.htm"
BASE_URL = "https://www2.gwu.edu/~calm/data/CALM_Data/Europe/Nordic"
ACCESSED = "2026-09-29"
LICENSE = "unverified"
SCHEMA_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / SRC_ID
DERIVED = RAW / "derived"
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
GTNP_SITES = ROOT / "data" / "raw" / "gtnp" / "sites.json"
DEM_DIRS = [ROOT / "data" / "raw" / "dem", RAW / "qc_dem"]

YEAR_MIN = 1990
ALT_MAX_CM = 600.0
CELL_DEG = 0.009            # 약 1 km 셀(6B.3)
DUP_DEG = 0.01              # v3 F4_direct 와의 중복 기준(체비쇼프)
ELEV_TOL_M = 100.0          # 설명문 표고와 DEM 표고의 허용 차
WATER_RANGE_M = 0.01        # DEM 5 x 5 창의 범위가 이보다 작으면 평탄한 수면으로 본다
COORD_PREC_DEG = 0.00017    # 0.01 분 = 1/6000 도
RUN_LEN_FLAG = 3            # 같은 값이 이 횟수 이상 이어지면 notes 에 표시한다

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method",
            "label_def", "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm",
            "coord_prec_deg", "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of",
            "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED

CITATION = {
    "S2": ("Akerman, H.J. 1998. Active layer monitoring, Abisko area, Sweden. In: International Permafrost "
           "Association, Data and Information Working Group, comp. Circumpolar Active-Layer Permafrost System "
           "(CAPS), version 1.0. CD-ROM. Boulder, Colorado: NSIDC, University of Colorado at Boulder; "
           "Akerman, H.J. and Johansson, M. 2008. Thawing permafrost and thicker active layers in sub-arctic "
           "Sweden. Permafrost and Periglacial Processes 19: 279-292, https://doi.org/10.1002/ppp.626; "
           "CALM site S2 data file S2_alt_abisko_78_23.xls, " + URL + " (accessed " + ACCESSED + ")"),
    "S1": ("Akerman, H.J. 1998. Active layer monitoring, Kapp Linne', Svalbard. In: International Permafrost "
           "Association, Data and Information Working Group, comp. Circumpolar Active-Layer Permafrost System "
           "(CAPS), version 1.0. CD-ROM. Boulder, Colorado: NSIDC, University of Colorado at Boulder; "
           "CALM site S1 data file s01_alt_2011.xls, " + URL + " (accessed " + ACCESSED + ")"),
}
ORIG_SOURCE = {
    "S2": "CALM site S2 Abisko area (https://www2.gwu.edu/~calm/data/webforms/s2_f.html); "
          "data submission: Margareta Johansson, Lund University; series started by H. Jonas Akerman",
    "S1": "CALM site S1 Kapp Linne (https://www2.gwu.edu/~calm/data/webforms/s1_f.html); "
          "data submission: H. Jonas Akerman, Lund University",
}

# 파일별 정의. parent_v3_loc_id 는 같은 CALM 지점의 평균을 담은 v3 행이다(6B.3 '하위 지점과 상위 행').
SOURCES = OrderedDict(
    S2=dict(
        calm_name="Abisko area", country="Sweden", subunit="Scandinavia", macro="NAtlantic",
        dir="s02_abisko_area", xls="S2_alt_abisko_78_23.xls", descr="site_descr.txt",
        listing="s02_files.txt", sheet="data", header_row=0, first_data_row=1,
        descr_encoding="latin-1", parent_v3_loc_id=17520,
    ),
    S1=dict(
        calm_name="Kapp Linne", country="Norway", subunit="Svalbard", macro="NAtlantic",
        dir="s01_kappe_linne", xls="s01_alt_2011.xls", descr="site_descr.txt",
        listing="s01_files.txt", sheet="data", header_row=1, first_data_row=2,
        descr_encoding="latin-1", parent_v3_loc_id=17569,
    ),
)

# 하위 지점 이름(자료 파일 머리글과 설명문에서 옮겼다)
SITE_NAMES = {
    ("S2", "AB1"): "Storflaket", ("S2", "AB2"): "Kursflaket", ("S2", "AB3"): "Mellanflaket",
    ("S2", "AB4"): "Torneträsk", ("S2", "AB5"): "Heliport", ("S2", "AB6"): "Låkta Hpl.",
    ("S2", "AB7"): "Katterjokk", ("S2", "AB8"): "Rakkas I", ("S2", "AB9"): "Rakkas II",
    ("S2", "AB10"): "Rakkas III", ("S2", "AB11"): "Narkervare",
    ("S1", "AL3"): "Bog", ("S1", "AL4"): "Raised beach ridge with ice wedge polygons",
    ("S1", "AL5"): "Sorted net surface", ("S1", "AL9"): "Low deflation surface with sorted net",
    ("S1", "AL10"): "Slope with small sorted steps",
    ("S1", "AL14"): "Raised beach ridge with soil wedge polygons",
    ("S1", "AL19"): "High deflation surface", ("S1", "AL20"): "Dryas dominated deflation surface",
}
# GTN-P 시추공 이름과 대조할 하위 지점(이름의 앞부분이 같으면 같은 지점으로 본다)
GTNP_MATCH = {
    ("S2", "AB1"): "storflaket", ("S2", "AB2"): "kursflaket", ("S2", "AB7"): "katterjokk",
    ("S1", "AL3"): "kapp linne", ("S1", "AL4"): "kapp linne", ("S1", "AL5"): "kapp linne",
    ("S1", "AL9"): "kapp linne", ("S1", "AL10"): "kapp linne", ("S1", "AL14"): "kapp linne",
    ("S1", "AL19"): "kapp linne", ("S1", "AL20"): "kapp linne",
}
# GTN-P 목록에서 좌표가 어긋난 항목(이름은 Storflaket 인데 경도가 18.265 다). 대조에서 뺀다.
GTNP_SKIP_IDS = {1099}

COORD_RE = re.compile(r"(\d+)\s*o\s*(\d+)'(\d+)''\s*N\s*,\s*(\d+)\s*o\s*(\d+)'(\d+)''\s*E")


# ---------------------------------------------------------------- 도구
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def git_tracked_clean(path: Path) -> bool:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out == ""
    except Exception:
        return False


def decmin_to_deg(d: int, m: int, h: int) -> float:
    """도, 분, 분의 백분위 표기를 십진 도로 바꾼다."""
    return d + (m + h / 100.0) / 60.0


def dms_to_deg(d: int, m: int, s: int) -> float:
    return d + m / 60.0 + s / 3600.0


def haversine_km(lat0, lon0, lat1, lon1) -> float:
    r = 6371.0088
    p0, p1 = math.radians(lat0), math.radians(lat1)
    dl = math.radians(lon1 - lon0)
    a = math.sin((p1 - p0) / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cell_index(lat: float, lon: float):
    ky = math.floor(lat / CELL_DEG)
    phi = math.radians((ky + 0.5) * CELL_DEG)
    kx = math.floor(lon * math.cos(phi) / CELL_DEG)
    return int(ky), int(kx)


def block_id(lat: float, lon: float) -> int:
    """v3 의 0.5° 블록 번호(floor(lat/0.5) * 100000 + floor(lon/0.5))."""
    return int(math.floor(lat / 0.5)) * 100000 + int(math.floor(lon / 0.5))


def fnum(x, nd):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def dist_stats(vals):
    v = np.asarray(list(vals), dtype=float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), min=round(float(v.min()), 2), median=round(float(np.median(v)), 2),
                max=round(float(v.max()), 2), mean=round(float(v.mean()), 2))


# ---------------------------------------------------------------- 설명문(좌표, 표고)
def parse_descr(path: Path, encoding: str, prefix: str) -> dict:
    """설명문의 '1. Site Name;' 묶음에서 하위 지점 부호, 좌표 표기, 표고를 읽는다."""
    txt = path.read_bytes().decode(encoding)
    lines = [ln.strip() for ln in txt.splitlines()]
    out = OrderedDict()
    for i, ln in enumerate(lines):
        if not re.match(r"^1\.\s+Site Name;", ln):
            continue
        name_line = next((x for x in lines[i + 1:i + 4] if x), "")
        m = re.match(rf"^({prefix})\s*(\d+)\b", name_line)
        if not m:
            continue                      # 지역 전체 묶음(TT/GEN 등)
        code = f"{m.group(1)}{int(m.group(2))}"
        coord_raw, elev_raw = "", ""
        for j in range(i + 1, min(i + 30, len(lines))):
            if re.match(r"^3\.\s+Latitude and longitude", lines[j]):
                coord_raw = next((x for x in lines[j + 1:j + 3] if x), "")
            if re.match(r"^4\.\s+Elevation above sea level", lines[j]):
                elev_raw = next((x for x in lines[j + 1:j + 3] if x), "")
                break
        mc = COORD_RE.search(coord_raw)
        if not mc:
            raise ValueError(f"{path.name}: {code} 의 좌표 표기를 읽지 못했다: {coord_raw!r}")
        la = tuple(int(x) for x in mc.groups()[:3])
        lo = tuple(int(x) for x in mc.groups()[3:])
        me = re.search(r"(\d+)(?:\s*-\s*(\d+))?", elev_raw)
        if me:
            e0 = float(me.group(1))
            e1 = float(me.group(2)) if me.group(2) else e0
            elev = (e0 + e1) / 2.0
        else:
            elev = None
        if code in out:
            raise ValueError(f"{path.name}: {code} 가 두 번 나온다")
        out[code] = dict(coord_raw=coord_raw.rstrip(". "), lat_parts=la, lon_parts=lo,
                         elev_raw=elev_raw, elev_stated_m=elev)
    return out


# ---------------------------------------------------------------- 자료 파일
def header_code(src: str, header: str):
    """자료 열 머리글에서 하위 지점 부호를 얻는다. 하위 지점 열이 아니면 None."""
    h = " ".join(str(header).split())
    if src == "S2":
        m = re.match(r"^AB\s*(\d+)\s*;", h)
        if m:
            return f"AB{int(m.group(1))}"
        if h == "Narkevaere":
            return "AB11"
        if h == "Bergfors":
            return "Bergfors"
        return None
    m = re.match(r"^AL\s*(\d+)$", h)
    return f"AL{int(m.group(1))}" if m else None


def read_sheet(src: str, cfg: dict):
    """자료 시트의 (연도, 하위 지점) 칸을 모두 읽는다. 값은 원자료 단위 그대로다."""
    wb = xlrd.open_workbook(str(RAW / cfg["dir"] / cfg["xls"]))
    sh = wb.sheet_by_name(cfg["sheet"])
    cols = OrderedDict()
    for c in range(sh.ncols):
        cell = sh.cell(cfg["header_row"], c)
        if cell.ctype != xlrd.XL_CELL_TEXT:
            continue
        code = header_code(src, cell.value)
        if code is not None:
            if code in cols:
                raise ValueError(f"{cfg['xls']}: 머리글 {code} 가 두 번 나온다")
            cols[code] = (c, " ".join(str(cell.value).split()))
    cells = []
    for r in range(cfg["first_data_row"], sh.nrows):
        y = sh.cell(r, 0)
        if y.ctype != xlrd.XL_CELL_NUMBER:
            continue                      # 범례 행, 되풀이한 머리글 행
        year = int(round(y.value))
        if not (1900 < year < 2100):
            continue
        for code, (c, hdr) in cols.items():
            cell = sh.cell(r, c)
            if cell.ctype == xlrd.XL_CELL_NUMBER:
                val, raw, miss = float(cell.value), repr(cell.value), ""
            elif cell.ctype == xlrd.XL_CELL_EMPTY or (cell.ctype == xlrd.XL_CELL_TEXT
                                                       and not str(cell.value).strip()):
                val, raw, miss = None, "", "empty"
            elif cell.ctype == xlrd.XL_CELL_BLANK:
                val, raw, miss = None, "", "empty"
            elif cell.ctype == xlrd.XL_CELL_TEXT:
                t = str(cell.value).strip()
                if t.startswith(">") or t.startswith("<"):
                    raise ValueError(f"{cfg['xls']} {year} {code}: 부등호 표기 {t!r} 는 규칙에 없다")
                val, raw, miss = None, t, t
            else:
                raise ValueError(f"{cfg['xls']} {year} {code}: 칸 형식 {cell.ctype} 을 처리하지 못한다")
            cells.append(dict(calm_site=src, sub_code=code, header=hdr, year=year, value_raw=raw,
                              value=val, missing_code=miss, sheet_row=r + 1, sheet_col=c + 1))
    return cells, cols, wb


def compare_sheet2(wb, cells) -> dict:
    """Abisko 'Sheet2'(2002년 11월판, 소수 2자리)와 'data' 시트를 같은 연도, 같은 하위 지점에서 견준다."""
    sh = wb.sheet_by_name("Sheet2")
    cols = {}
    for c in range(sh.ncols):
        cell = sh.cell(1, c)
        if cell.ctype == xlrd.XL_CELL_TEXT:
            m = re.match(r"^AB\s*(\d+)\s*;", " ".join(cell.value.split()))
            if m:
                cols[f"AB{int(m.group(1))}"] = c
    old = {}
    for r in range(2, sh.nrows):
        y = sh.cell(r, 0)
        if y.ctype != xlrd.XL_CELL_NUMBER:
            continue
        for code, c in cols.items():
            cell = sh.cell(r, c)
            if cell.ctype == xlrd.XL_CELL_NUMBER:
                old[(code, int(round(y.value)))] = float(cell.value)
    diffs, n_cmp = [], 0
    per_code = {}
    for ce in cells:
        k = (ce["sub_code"], ce["year"])
        if ce["value"] is None or k not in old:
            continue
        n_cmp += 1
        d_cm = (ce["value"] - old[k]) * 100.0
        per_code.setdefault(ce["sub_code"], []).append(abs(d_cm))
        if abs(d_cm) > 1.0:               # 반올림(0.5 cm)보다 큰 차이
            diffs.append(dict(sub_code=ce["sub_code"], year=ce["year"],
                              data_cm=round(ce["value"] * 100, 3), sheet2_cm=round(old[k] * 100, 3),
                              diff_cm=round(d_cm, 3)))
    return dict(
        note="Sheet2 는 'UPDATE nov 2002' 판이고 소수 2자리(m)다. 점 자료는 'data' 시트를 쓴다",
        n_compared=n_cmp, n_diff_gt_1cm=len(diffs),
        n_diff_gt_1cm_1990plus=sum(1 for d in diffs if d["year"] >= YEAR_MIN),
        max_abs_diff_cm_by_sub_code={k: round(max(v), 3) for k, v in sorted(per_code.items())},
        diffs_gt_1cm=diffs,
    )


def compare_kapp_dat(cells) -> dict:
    """Kapp Linné 의 1972-2000년판 텍스트 파일과 자료 시트를 같은 연도, 같은 하위 지점에서 견준다."""
    path = RAW / "s01_kappe_linne" / "kapp72_00.dat"
    if not path.exists():
        return dict(ran=False, note="kapp72_00.dat 가 없어 대조하지 않았다")
    lines = [ln for ln in path.read_text(encoding="latin-1").splitlines() if ln.strip()]
    rd = list(csv.reader(lines))
    head = [h.strip() for h in rd[0]]
    idx = {h: i for i, h in enumerate(head) if re.match(r"^AL\d+$", h)}
    old = {}
    for r in rd[1:]:
        if not re.match(r"^\d{4}$", r[0].strip()):
            continue
        for h, i in idx.items():
            if i < len(r) and r[i].strip():
                old[(h, int(r[0]))] = float(r[i])
    n_cmp, diffs = 0, []
    for ce in cells:
        k = (ce["sub_code"], ce["year"])
        if ce["calm_site"] != "S1" or ce["value"] is None or k not in old:
            continue
        n_cmp += 1
        if abs(ce["value"] - old[k]) > 1e-9:
            diffs.append(dict(sub_code=k[0], year=k[1], xls_cm=round(ce["value"] * 100, 3),
                              dat_cm=round(old[k] * 100, 3)))
    return dict(ran=True, file="kapp72_00.dat", n_compared=n_cmp, n_diff=len(diffs), diffs=diffs,
                years=[min(y for _, y in old), max(y for _, y in old)])


# ---------------------------------------------------------------- 좌표 점검
def dem_sample(lat: float, lon: float):
    """좌표가 든 DEM 타일에서 5 x 5 창을 읽는다. 타일이 없으면 None."""
    import rasterio
    from rasterio.windows import Window
    ns = f"N{int(math.floor(lat)):02d}"
    ew = f"E{int(math.floor(lon)):03d}" if lon >= 0 else f"W{int(math.ceil(-lon)):03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM.tif"
    for d in DEM_DIRS:
        p = d / name
        if p.exists():
            with rasterio.open(p) as ds:
                r, c = ds.index(lon, lat)
                r0, c0 = max(r - 2, 0), max(c - 2, 0)
                w = ds.read(1, window=Window(c0, r0, 5, 5)).astype(float)
                center = float(w[min(r - r0, w.shape[0] - 1), min(c - c0, w.shape[1] - 1)])
                return dict(elev=center, rng=float(w.max() - w.min()), tile=name,
                            tile_dir=str(d.relative_to(ROOT)))
    return None


def gtnp_boreholes():
    sites = json.loads(GTNP_SITES.read_text())
    out = []
    for s in sites:
        for b in s.get("boreholes", []):
            if b.get("id") in GTNP_SKIP_IDS:
                continue
            out.append(dict(id=b["id"], name=b["name"], lat=float(b["latitude"]), lon=float(b["longitude"])))
    return out


def read_v3():
    rows = []
    with open(V3, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append((int(float(r["loc_id"])), float(r["lat"]), float(r["lon"]), r["region"],
                         r["source_id"], float(r["alt_cm"]) if r["alt_cm"] else float("nan")))
    return rows


def nearest_v3(lat, lon, v3, only=None):
    best = None
    for loc, la, lo, reg, sid, alt in v3:
        if only and sid != only:
            continue
        d = max(abs(la - lat), abs(lo - lon))
        if best is None or d < best[0]:
            best = (d, loc, reg, sid, haversine_km(lat, lon, la, lo))
    return best


def build_sites(skip_dem: bool):
    v3 = read_v3()
    bhs = gtnp_boreholes() if GTNP_SITES.exists() else []
    sites = OrderedDict()
    for src, cfg in SOURCES.items():
        prefix = "AB" if src == "S2" else "AL"
        descr = parse_descr(RAW / cfg["dir"] / cfg["descr"], cfg["descr_encoding"], prefix)
        for code, d in descr.items():
            lat = decmin_to_deg(*d["lat_parts"])
            lon = decmin_to_deg(*d["lon_parts"])
            dms_ok = d["lat_parts"][2] < 60 and d["lon_parts"][2] < 60
            alt_lat = dms_to_deg(*d["lat_parts"]) if dms_ok else None
            alt_lon = dms_to_deg(*d["lon_parts"]) if dms_ok else None
            ky, kx = cell_index(lat, lon)
            rec = dict(
                calm_site=src, sub_code=code, site_id=f"{src}_{code}",
                site_name=f"{cfg['calm_name']} {code} {SITE_NAMES.get((src, code), '')}".strip(),
                coord_raw=d["coord_raw"], lat=lat, lon=lon,
                third_field_ge60=int(not dms_ok), lat_if_dms=alt_lat, lon_if_dms=alt_lon,
                shift_if_dms_km=(haversine_km(lat, lon, alt_lat, alt_lon) if dms_ok else None),
                elev_raw=d["elev_raw"], elev_stated_m=d["elev_stated_m"],
                cell_ky=ky, cell_kx=kx, cell_id=f"{ky}_{kx}", block=block_id(lat, lon),
            )
            # DEM 대조
            ds = None if skip_dem else dem_sample(lat, lon)
            if not skip_dem and ds is None:
                raise FileNotFoundError(f"{rec['site_id']} 의 DEM 타일이 없다. --skip-dem-qc 로 건너뛸 수 있다")
            rec.update(elev_dem_m=(ds["elev"] if ds else None), dem_win_range_m=(ds["rng"] if ds else None),
                       dem_tile=(ds["tile"] if ds else ""), dem_tile_dir=(ds["tile_dir"] if ds else ""))
            if ds and dms_ok:
                ds2 = dem_sample(alt_lat, alt_lon)
                rec["elev_dem_if_dms_m"] = ds2["elev"] if ds2 else None
                rec["dem_win_range_if_dms_m"] = ds2["rng"] if ds2 else None
            else:
                rec["elev_dem_if_dms_m"] = None
                rec["dem_win_range_if_dms_m"] = None
            # GTN-P 시추공 대조
            key = GTNP_MATCH.get((src, code))
            rec.update(gtnp_ref_name="", gtnp_ref_id="", gtnp_ref_lat=None, gtnp_ref_lon=None,
                       gtnp_ref_dist_km=None, gtnp_ref_dist_if_dms_km=None)
            if key and bhs:
                cand = [b for b in bhs if b["name"].lower().startswith(key)]
                if cand:
                    b = min(cand, key=lambda b: haversine_km(lat, lon, b["lat"], b["lon"]))
                    rec.update(gtnp_ref_name=b["name"], gtnp_ref_id=b["id"], gtnp_ref_lat=b["lat"],
                               gtnp_ref_lon=b["lon"],
                               gtnp_ref_dist_km=haversine_km(lat, lon, b["lat"], b["lon"]))
                    if dms_ok:
                        b2 = min(cand, key=lambda b: haversine_km(alt_lat, alt_lon, b["lat"], b["lon"]))
                        rec["gtnp_ref_dist_if_dms_km"] = haversine_km(alt_lat, alt_lon, b2["lat"], b2["lon"])
            # v3 와의 관계
            n4 = nearest_v3(lat, lon, v3, only="F4_direct")
            na = nearest_v3(lat, lon, v3)
            rec.update(v3_f4_near_loc_id=n4[1], v3_f4_near_region=n4[2], v3_f4_near_cheb_deg=n4[0],
                       v3_f4_near_km=n4[4], v3_f4_dup=int(n4[0] <= DUP_DEG),
                       v3_any_near_loc_id=na[1], v3_any_near_source_id=na[3], v3_any_near_cheb_deg=na[0],
                       v3_any_near_km=na[4], parent_v3_loc_id=cfg["parent_v3_loc_id"])
            # 좌표 판정
            reasons = []
            if ds is not None and d["elev_stated_m"] is not None:
                if abs(ds["elev"] - d["elev_stated_m"]) > ELEV_TOL_M:
                    reasons.append(f"dem_elev_diff_gt_{int(ELEV_TOL_M)}m")
                if ds["rng"] < WATER_RANGE_M:
                    reasons.append("dem_flat_water_surface")
            rec["coord_flag"] = int(bool(reasons))
            rec["coord_reason"] = "|".join(reasons)
            sites[(src, code)] = rec
    return sites, v3


# ---------------------------------------------------------------- 점 자료
def run_lengths(series):
    """연도 순서의 (year, value) 에서 같은 값이 이어지는 구간을 돌려준다."""
    runs, cur = [], []
    for y, v in series:
        if cur and abs(v - cur[-1][1]) < 1e-12 and y == cur[-1][0] + 1:
            cur.append((y, v))
        else:
            if len(cur) >= RUN_LEN_FLAG:
                runs.append(cur)
            cur = [(y, v)]
    if len(cur) >= RUN_LEN_FLAG:
        runs.append(cur)
    return runs


def build(skip_dem: bool):
    sites, v3 = build_sites(skip_dem)
    all_cells, col_info, sheet2_cmp, units = [], {}, None, {}
    for src, cfg in SOURCES.items():
        cells, cols, wb = read_sheet(src, cfg)
        col_info[src] = {k: dict(sheet_col=v[0] + 1, header=v[1]) for k, v in cols.items()}
        vals = [c["value"] for c in cells if c["value"] is not None]
        med, vmax = float(np.median(vals)), float(np.max(vals))
        if not (0.05 < med < 3.0 and vmax < 6.0):
            raise ValueError(f"{cfg['xls']}: 값의 범위(중앙 {med}, 최대 {vmax})가 m 단위 가정과 맞지 않는다")
        units[src] = dict(unit_raw="m", factor_to_cm=100.0, raw_min=float(np.min(vals)), raw_median=med,
                          raw_max=vmax, basis="값의 범위(중앙값 0.05-3.0, 최대 6.0 미만)로 확인")
        if src == "S2":
            sheet2_cmp = compare_sheet2(wb, cells)
            # 'Narkevaere' 열과 Sheet2 'AB11; Narkervare' 열이 같은 계열인지 확인한다
            sh2 = wb.sheet_by_name("Sheet2")
            c11 = [c for c in range(sh2.ncols) if sh2.cell(1, c).ctype == xlrd.XL_CELL_TEXT
                   and re.match(r"^AB\s*11\s*;", " ".join(sh2.cell(1, c).value.split()))]
            if len(c11) != 1:
                raise ValueError("Sheet2 에서 AB11 열을 찾지 못했다")
            old11 = {int(round(sh2.cell(r, 0).value)): float(sh2.cell(r, c11[0]).value)
                     for r in range(2, sh2.nrows)
                     if sh2.cell(r, 0).ctype == xlrd.XL_CELL_NUMBER
                     and sh2.cell(r, c11[0]).ctype == xlrd.XL_CELL_NUMBER}
            new11 = {c["year"]: c["value"] for c in cells if c["sub_code"] == "AB11" and c["value"] is not None}
            common = sorted(set(old11) & set(new11))
            d11 = {y: abs(old11[y] - new11[y]) * 100.0 for y in common}
            n_same = sum(1 for v in d11.values() if v <= 0.5 + 1e-6)      # 소수 2자리 반올림 범위
            if not (len(common) >= 10 and n_same / len(common) >= 0.75):
                raise ValueError(f"'Narkevaere' 열과 Sheet2 AB11 열이 맞지 않는다"
                                 f"(공통 {len(common)}년 가운데 {n_same}년만 반올림 범위에서 같다)")
            sheet2_cmp["narkevaere_is_ab11"] = dict(
                n_common_years=len(common), n_same_within_rounding=n_same,
                years=[common[0], common[-1]],
                years_revised={str(y): round(v, 3) for y, v in d11.items() if v > 0.5 + 1e-6},
                note="반올림 범위를 넘는 연도는 'data' 시트에서 값이 바뀐 연도다(다른 하위 지점에도 같은 연도에 차이가 있다)")
        all_cells.extend(cells)
    kapp_cmp = compare_kapp_dat(all_cells)

    # 중복 점검 1: 같은 (하위 지점, 연도) 가 두 번 나오는가
    keys = Counter((c["calm_site"], c["sub_code"], c["year"]) for c in all_cells)
    dup_keys = [k for k, n in keys.items() if n > 1]
    if dup_keys:
        raise ValueError(f"같은 (하위 지점, 연도) 가 두 번 나온다: {dup_keys[:5]}")

    # 중복 점검 2: 같은 연도에 하위 지점 사이에서 값이 완전히 같은 경우(소수 3자리 이상 값만)
    same_year = []
    by_year = {}
    for c in all_cells:
        if c["value"] is not None:
            by_year.setdefault((c["calm_site"], c["year"]), []).append(c)
    for (src, y), lst in sorted(by_year.items()):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                fine = abs(a["value"] * 100 - round(a["value"] * 100)) > 1e-9   # cm 정수가 아닌 값
                if fine and abs(a["value"] - b["value"]) < 1e-12:
                    same_year.append(dict(calm_site=src, year=y, sub_codes=[a["sub_code"], b["sub_code"]],
                                          value_cm=round(a["value"] * 100, 4)))
    # 중복 점검 3: 한 하위 지점 안에서 다른 연도의 값이 소수 끝까지 같은 경우(소수 4자리 넘는 값만)
    same_site = []
    by_site = {}
    for c in all_cells:
        if c["value"] is not None:
            by_site.setdefault((c["calm_site"], c["sub_code"]), []).append(c)
    for (src, code), lst in by_site.items():
        seen = {}
        for c in lst:
            v = c["value"]
            long_frac = abs(v * 1e4 - round(v * 1e4)) > 1e-7
            if long_frac:
                seen.setdefault(repr(v), []).append(c["year"])
        for rv, ys in seen.items():
            if len(ys) > 1:
                same_site.append(dict(calm_site=src, sub_code=code, years=ys,
                                      value_cm=round(float(rv) * 100, 4)))
    flag_same_year = {(d["calm_site"], sc, d["year"]): [x for x in d["sub_codes"] if x != sc]
                      for d in same_year for sc in d["sub_codes"]}
    flag_same_site = {(d["calm_site"], d["sub_code"], y): [x for x in d["years"] if x != y]
                      for d in same_site for y in d["years"]}

    # 같은 값이 이어지는 구간(탐침 길이 상한일 수 있다. 원자료에는 표기가 없다)
    runs_out, flag_run = [], {}
    for (src, code), lst in by_site.items():
        ser = sorted((c["year"], c["value"]) for c in lst)
        for run in run_lengths(ser):
            runs_out.append(dict(calm_site=src, sub_code=code, years=[run[0][0], run[-1][0]],
                                 n_years=len(run), value_cm=round(run[0][1] * 100, 4)))
            for y, _ in run:
                flag_run[(src, code, y)] = (run[0][0], run[-1][0], len(run))

    # 2002년 11월판(Sheet2)과 1 cm 넘게 다른 칸
    flag_sheet2 = {("S2", d["sub_code"], d["year"]): d["sheet2_cm"] for d in sheet2_cmp["diffs_gt_1cm"]}

    points, long_rows = [], []
    excluded = Counter()
    miss_codes = Counter()
    for c in all_cells:
        src, code, year = c["calm_site"], c["sub_code"], c["year"]
        cfg = SOURCES[src]
        row = dict(calm_site=src, sub_code=code, header=c["header"], year=year, value_raw=c["value_raw"],
                   missing_code=c["missing_code"], sheet_row=c["sheet_row"], sheet_col=c["sheet_col"],
                   alt_cm="", in_points=0, status="")
        if c["value"] is None:
            miss_codes[f"{src}:{c['missing_code']}"] += 1
            excluded["값 없음(결측 부호 또는 빈 칸)"] += 1
            row["status"] = "missing"
            long_rows.append(row)
            continue
        alt_cm = round(c["value"] * 100.0, 4)
        row["alt_cm"] = repr(alt_cm)
        if (src, code) not in sites:
            excluded["좌표 없음(Bergfors 열)"] += 1
            row["status"] = "no_coordinates"
            long_rows.append(row)
            continue
        st = sites[(src, code)]
        flags = []
        if year < YEAR_MIN:
            flags.append("year_pre1990")
        if not (0.0 < alt_cm <= ALT_MAX_CM):
            flags.append("range")
        if st["coord_flag"]:
            flags.append("coord")
        qc = ";".join(flags) if flags else "ok"
        # n_obs
        if src == "S1" and year <= 1973:
            n_obs, nobs_note = 10, "n_obs_basis=site_descr(10 random points in 10x10 m, 1972-1973)"
        elif src == "S1" and code != "AL5" and 1974 <= year <= 1993:
            n_obs, nobs_note = 25, "n_obs_basis=site_descr(25 random points in 100x100 m, 1974-1993)"
        elif src == "S1":
            n_obs, nobs_note = 1, ("n_obs_basis=unknown(AL5 kept at 10x10 m)" if code == "AL5" and year <= 1993
                                   else "n_obs_basis=unknown(ITEX scheme from 1994, node count not given)"
                                   if year >= 1994 else "n_obs_basis=unknown")
        else:
            n_obs, nobs_note = 1, "n_obs_basis=unknown(CALM metadata form: Grid 100)"
        notes = [f"sub_code={code}", f"coord_raw={st['coord_raw']}", f"elev_stated_m={st['elev_raw']}",
                 f"parent_v3_loc_id={st['parent_v3_loc_id']}(CALM {src} site mean)", nobs_note,
                 f"cell={st['cell_id']}", f"block={st['block']}"]
        if st["elev_dem_m"] is not None:
            notes.append(f"elev_dem_m={st['elev_dem_m']:.1f}")
        if st["coord_flag"]:
            notes.append(f"coord_reason={st['coord_reason']}")
        if st["gtnp_ref_dist_km"] is not None:
            notes.append(f"gtnp_ref={st['gtnp_ref_name']}@{st['gtnp_ref_dist_km']:.2f}km")
        k = (src, code, year)
        if k in flag_same_year:
            notes.append("same_value_same_year_as=" + "|".join(flag_same_year[k]))
        if k in flag_same_site:
            notes.append("same_value_as_year=" + "|".join(str(y) for y in flag_same_site[k]))
        if k in flag_run:
            a, b, n = flag_run[k]
            notes.append(f"constant_run={a}-{b}(n={n}, probe limit not confirmed)")
        if k in flag_sheet2:
            notes.append(f"sheet2_nov2002_cm={flag_sheet2[k]}")
        points.append(dict(
            src_id=SRC_ID, site_id=st["site_id"], site_name=st["site_name"],
            lat=f"{st['lat']:.6f}", lon=f"{st['lon']:.6f}", year=year, month="", alt_cm=repr(alt_cm),
            method="probe", label_def="direct_eos", n_obs=n_obs, country=cfg["country"], macro=cfg["macro"],
            citation=CITATION[src], license=LICENSE, subunit=cfg["subunit"], date="",
            eos_basis="protocol", value_kind="annual_value", year_min="", year_max="", alt_sd_cm="",
            coord_prec_deg=f"{COORD_PREC_DEG:.5f}", disturbed=0, disturb_type="", right_censored=0,
            orig_source=ORIG_SOURCE[src], dup_of="", qc_flag=qc, notes="; ".join(notes),
        ))
        row.update(in_points=1, status=qc)
        long_rows.append(row)

    order = {k: i for i, k in enumerate(sites.keys())}
    points.sort(key=lambda p: (order[(p["site_id"].split("_")[0], p["site_id"].split("_")[1])], p["year"]))
    return dict(points=points, long_rows=long_rows, sites=sites, excluded=excluded, miss_codes=miss_codes,
                col_info=col_info, sheet2_cmp=sheet2_cmp, kapp_cmp=kapp_cmp, units=units, same_year=same_year,
                same_site=same_site, runs=runs_out, n_raw=len(all_cells))


# ---------------------------------------------------------------- 요약
def summarize(res) -> dict:
    pts, sites = res["points"], res["sites"]
    is_main = lambda p: (p["label_def"] == "direct_eos" and p["qc_flag"] == "ok"
                         and p["right_censored"] == 0 and p["disturbed"] == 0)
    main = [p for p in pts if is_main(p)]
    y1990 = [p for p in pts if p["year"] >= YEAR_MIN]
    per_site = []
    for (src, code), st in sites.items():
        sp = [p for p in pts if p["site_id"] == st["site_id"]]
        s90 = [p for p in sp if p["year"] >= YEAR_MIN]
        sm = [p for p in sp if is_main(p)]
        per_site.append(dict(
            site_id=st["site_id"], site_name=st["site_name"], subunit=SOURCES[src]["subunit"],
            lat=round(st["lat"], 6), lon=round(st["lon"], 6), cell_id=st["cell_id"], block=st["block"],
            coord_flag=st["coord_flag"], n_rows=len(sp), n_rows_1990plus=len(s90), n_main=len(sm),
            year_min=min((p["year"] for p in sp), default=None),
            year_max=max((p["year"] for p in sp), default=None),
            year_min_1990plus=min((p["year"] for p in s90), default=None),
            year_max_1990plus=max((p["year"] for p in s90), default=None),
            alt_mean_1990plus_cm=(round(float(np.mean([float(p["alt_cm"]) for p in s90])), 2) if s90 else None),
            alt_mean_main_cm=(round(float(np.mean([float(p["alt_cm"]) for p in sm])), 2) if sm else None),
        ))
    with_rows = [s for s in per_site if s["n_rows_1990plus"] > 0]
    main_sites = [s for s in per_site if s["n_main"] > 0]

    def cells_blocks(lst):
        out = {}
        for sub in sorted({s["subunit"] for s in lst}):
            ss = [s for s in lst if s["subunit"] == sub]
            out[sub] = dict(n_sites=len(ss), n_cells=len({s["cell_id"] for s in ss}),
                            n_blocks=len({s["block"] for s in ss}))
        out["all"] = dict(n_sites=len(lst), n_cells=len({s["cell_id"] for s in lst}),
                          n_blocks=len({s["block"] for s in lst}))
        return out

    return dict(
        n_sites=len({p["site_id"] for p in pts}),
        n_sites_in_descr=len(sites),
        n_sites_in_descr_without_data=sorted(st["site_id"] for st in sites.values()
                                             if not any(p["site_id"] == st["site_id"] for p in pts)),
        n_rows=len(pts), n_rows_1990plus=len(y1990), n_rows_main=len(main),
        main_set_rule="label_def = direct_eos, qc_flag = ok, right_censored = 0, disturbed = 0",
        n_sites_by_subunit=dict(Counter(s["subunit"] for s in per_site if s["n_rows"] > 0)),
        n_rows_by_subunit=dict(Counter(p["subunit"] for p in pts)),
        n_rows_1990plus_by_subunit=dict(Counter(p["subunit"] for p in y1990)),
        n_rows_main_by_subunit=dict(Counter(p["subunit"] for p in main)),
        n_rows_by_macro=dict(Counter(p["macro"] for p in pts)),
        n_rows_by_method=dict(Counter(p["method"] for p in pts)),
        n_rows_by_qc_flag=dict(Counter(p["qc_flag"] for p in pts)),
        n_right_censored=sum(p["right_censored"] for p in pts),
        n_disturbed=sum(p["disturbed"] for p in pts),
        year_range=[min(p["year"] for p in pts), max(p["year"] for p in pts)],
        year_range_1990plus=[min(p["year"] for p in y1990), max(p["year"] for p in y1990)],
        year_range_main=[min(p["year"] for p in main), max(p["year"] for p in main)],
        alt_cm_rows_all=dist_stats(float(p["alt_cm"]) for p in pts),
        alt_cm_rows_1990plus=dist_stats(float(p["alt_cm"]) for p in y1990),
        alt_cm_rows_main=dist_stats(float(p["alt_cm"]) for p in main),
        alt_cm_site_multiyear_mean_1990plus=dist_stats(s["alt_mean_1990plus_cm"] for s in with_rows),
        alt_cm_site_multiyear_mean_main=dist_stats(s["alt_mean_main_cm"] for s in main_sites),
        cells_1990plus=cells_blocks(with_rows),
        cells_main=cells_blocks(main_sites),
        sites_coord_flagged=[s["site_id"] for s in per_site if s["coord_flag"] and s["n_rows"] > 0],
        per_site=per_site,
    )


# ---------------------------------------------------------------- 쓰기
def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


SITE_QC_COLS = ["site_id", "calm_site", "sub_code", "site_name", "coord_raw", "lat", "lon", "third_field_ge60",
                "lat_if_dms", "lon_if_dms", "shift_if_dms_km", "elev_raw", "elev_stated_m", "elev_dem_m",
                "dem_win_range_m", "elev_dem_if_dms_m", "dem_win_range_if_dms_m", "dem_tile", "dem_tile_dir",
                "gtnp_ref_name", "gtnp_ref_id", "gtnp_ref_lat", "gtnp_ref_lon", "gtnp_ref_dist_km",
                "gtnp_ref_dist_if_dms_km", "cell_ky", "cell_kx", "cell_id", "block", "v3_f4_near_loc_id",
                "v3_f4_near_region", "v3_f4_near_cheb_deg", "v3_f4_near_km", "v3_f4_dup", "v3_any_near_loc_id",
                "v3_any_near_source_id", "v3_any_near_cheb_deg", "v3_any_near_km", "parent_v3_loc_id",
                "coord_flag", "coord_reason", "n_rows", "n_rows_1990plus", "n_main", "year_min_1990plus",
                "year_max_1990plus", "alt_mean_1990plus_cm"]
LONG_COLS = ["calm_site", "sub_code", "header", "year", "value_raw", "missing_code", "alt_cm", "in_points",
             "status", "sheet_row", "sheet_col"]


def round_site(rec: dict) -> dict:
    out = dict(rec)
    for k, nd in [("lat", 6), ("lon", 6), ("lat_if_dms", 6), ("lon_if_dms", 6), ("shift_if_dms_km", 3),
                  ("elev_dem_m", 1), ("dem_win_range_m", 2), ("elev_dem_if_dms_m", 1),
                  ("dem_win_range_if_dms_m", 2), ("gtnp_ref_dist_km", 3), ("gtnp_ref_dist_if_dms_km", 3),
                  ("v3_f4_near_cheb_deg", 4), ("v3_f4_near_km", 2), ("v3_any_near_cheb_deg", 4),
                  ("v3_any_near_km", 2)]:
        if isinstance(out.get(k), float):
            out[k] = round(out[k], nd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-dem-qc", action="store_true",
                    help="DEM 타일 없이 실행한다. 좌표 판정(qc_flag = coord)이 빠지므로 점검용으로만 쓴다")
    args = ap.parse_args()

    res = build(args.skip_dem_qc)
    pts = res["points"]
    assert list(pts[0].keys()) == COLUMNS, "열 순서가 형식 정의와 다르다"
    summ = summarize(res)

    per = {s["site_id"]: s for s in summ["per_site"]}
    site_rows = []
    for st in res["sites"].values():
        r = round_site(st)
        r.update({k: per[st["site_id"]][k] for k in ["n_rows", "n_rows_1990plus", "n_main",
                                                     "year_min_1990plus", "year_max_1990plus",
                                                     "alt_mean_1990plus_cm"]})
        site_rows.append(r)

    out_pts = OUT_DIR / f"{SRC_ID}_points.csv"
    out_meta = OUT_DIR / f"{SRC_ID}_meta.json"
    out_long = DERIVED / f"{SRC_ID}_raw_long.csv"
    out_site = DERIVED / f"{SRC_ID}_site_qc.csv"
    write_csv(out_pts, pts, COLUMNS)
    write_csv(out_long, res["long_rows"], LONG_COLS)
    write_csv(out_site, site_rows, SITE_QC_COLS)

    files = []
    for src, cfg in SOURCES.items():
        for fn in (cfg["xls"], cfg["descr"], cfg["listing"]):
            p = RAW / cfg["dir"] / fn
            files.append(dict(name=str(p.relative_to(ROOT)), size_bytes=p.stat().st_size,
                              sha256=sha256_of(p), url=f"{BASE_URL}/{cfg['dir']}/{fn}",
                              role=("자료" if fn == cfg["xls"] else "설명문(좌표, 표고)" if fn == cfg["descr"]
                                    else "파일 목록")))
    p = RAW / "s01_kappe_linne" / "kapp72_00.dat"
    if p.exists():
        files.append(dict(name=str(p.relative_to(ROOT)), size_bytes=p.stat().st_size, sha256=sha256_of(p),
                          url=f"{BASE_URL}/s01_kappe_linne/kapp72_00.dat", role="대조용(1972-2000년판)"))
    dem_used = sorted({(st["dem_tile_dir"], st["dem_tile"]) for st in res["sites"].values() if st["dem_tile"]})
    qc_inputs = []
    for d, t in dem_used:
        p = ROOT / d / t
        qc_inputs.append(dict(name=str(p.relative_to(ROOT)), size_bytes=p.stat().st_size, sha256=sha256_of(p),
                              url=f"https://copernicus-dem-30m.s3.amazonaws.com/{t[:-4]}/{t}",
                              role="좌표 점검용 DEM(Copernicus DEM GLO-30)"))
    if GTNP_SITES.exists():
        qc_inputs.append(dict(name=str(GTNP_SITES.relative_to(ROOT)), size_bytes=GTNP_SITES.stat().st_size,
                              sha256=sha256_of(GTNP_SITES), role="좌표 점검용 GTN-P 시추공 목록"))

    parser_path = Path(__file__).resolve()
    sites_meta = [round_site(st) for st in res["sites"].values()]
    n_by_label = dict(Counter(p["label_def"] for p in pts))
    n_by_method = dict(Counter(p["method"] for p in pts))
    excluded = dict(res["excluded"])

    meta = OrderedDict(
        src_id=SRC_ID, name=SRC_NAME, url=URL, doi="",
        doi_note=("자료 파일 자체의 DOI 는 없다. 관련 문헌 Akerman & Johansson 2008 의 DOI 는 10.1002/ppp.626 이다"
                  "(Crossref 에서 확인). 같은 지점의 평균 계열은 PANGAEA 972777(10.1594/PANGAEA.972777)에 있다"),
        accessed=ACCESSED, files=files, qc_input_files=qc_inputs,
        license=LICENSE,
        license_basis=("CALM 누리집의 자료 쪽(north.htm, data-links.htm, webforms/s1_f.html, s2_f.html)과 "
                       "두 설명문에서 CC 표기나 재배포 조항을 찾지 못했다. 설명문은 인용문을 지정한다"
                       "(Please cite these data as follows). 재배포 허용 여부는 확인하지 못했다. "
                       "점 자료는 커밋하지 않는다(data/processed/ext_labels/.gitignore)"),
        citation=CITATION,
        parser=str(parser_path.relative_to(ROOT)), parser_sha256=sha256_of(parser_path),
        parser_git_commit=git_head(),
        parser_git_note=("파서 파일이 이 커밋의 내용과 같다" if git_tracked_clean(parser_path)
                         else "파서 파일은 이 커밋에 들어 있지 않다(작업 트리의 새 파일 또는 수정본). 값은 실행 시점의 HEAD 다"),
        schema_version=SCHEMA_VERSION,
        outputs=dict(points=str(out_pts.relative_to(ROOT)), meta=str(out_meta.relative_to(ROOT)),
                     raw_long=str(out_long.relative_to(ROOT)), site_qc=str(out_site.relative_to(ROOT))),
        n_rows_raw=res["n_raw"],
        n_rows_raw_note="자료 시트의 (연도, 하위 지점) 칸 수. Abisko 46년 x 12열, Kapp Linné 40년 x 8열",
        n_rows_points=len(pts),
        n_by_label_def=n_by_label, n_by_method=n_by_method,
        n_excluded_by_reason=excluded,
        n_missing_by_code=dict(res["miss_codes"]),
        n_flagged_in_points=dict(
            year_pre1990=sum(1 for p in pts if "year_pre1990" in p["qc_flag"]),
            coord=sum(1 for p in pts if "coord" in p["qc_flag"]),
            range=sum(1 for p in pts if "range" in p["qc_flag"]),
            note="표시한 행은 점 자료에 남아 있다. 주 집합은 qc_flag = ok 인 행이다"),
        row_rule=("행은 (하위 지점, 연도)의 연 값이다. 1990년 이전 값은 형식 정의의 연도 규칙에 따라 지우지 않고 "
                  "qc_flag = year_pre1990 으로 남겼다. 자료원 계획의 '1990년 이후 연 값' 은 qc_flag 에 "
                  "year_pre1990 이 없는 행이다"),
        coordinate_conversion=OrderedDict(
            source="두 설명문(site_descr.txt)의 하위 지점별 '3. Latitude and longitude' 항목",
            notation="DDoMM'HH''N, DDDoMM'HH''E (예: 68o20'88''N, 18 o 58'30''E)",
            formula="십진 도 = DD + (MM + HH/100) / 60. 북위와 동경이므로 부호는 양수다",
            example="AB1: 68 + 20.88/60 = 68.348000, 18 + 58.30/60 = 18.971667",
            interpretation_basis=[
                "셋째 자리에 60 이상인 값이 있다(Abisko 88, 76, 80, Kapp Linné 61, 71, 82, 91, 84). 초라면 나올 수 없다",
                "AB1, AB2 의 변환 좌표가 GTN-P 시추공 Storflaket, Kursflaket 의 좌표와 0.3 km 안에서 맞는다(sites 의 gtnp_ref_dist_km)",
                "도분초로 읽으면 Kapp Linné AL2, AL3 은 DEM 표고 0 m 인 곳(바다)에 놓인다(sites 의 elev_dem_if_dms_m)",
                "Johansson 등 2011(Ambio 40, PMC3357866)의 시추공 좌표는 Storflaket 68°20′48-53″N 18°57′55″-58′38″E, "
                "Kursflaket 68°21′03″N 18°52′24″E 이고 변환 좌표와 0.3 km 안에서 맞는다",
            ],
            coord_prec_deg=COORD_PREC_DEG,
            coord_prec_note="표기 단위 0.01 분(약 19 m)이다. 실제 위치 정확도는 확인하지 못했다",
            alternative_shift_note="도분초로 읽었을 때의 위치 차는 sites 의 shift_if_dms_km 에 적었다(최대 약 0.7 km)",
            check_rule=(f"DEM 표고와 설명문 표고의 차가 {int(ELEV_TOL_M)} m 를 넘거나 DEM 5 x 5 창의 범위가 "
                        f"{WATER_RANGE_M} m 미만(평탄한 수면)이면 qc_flag = coord"),
            dem_check_ran=not args.skip_dem_qc,
        ),
        units=res["units"],
        units_note=("두 자료 파일의 하위 지점 열은 m 다. 100 을 곱했다. s02_files.txt 의 '(cm)' 표기는 값과 맞지 않는다. "
                    "PANGAEA 972777 의 S1, S2 연 값(cm)이 자료 파일의 지점 평균 열에 100 을 곱한 값과 맞는다"),
        missing_code=("Abisko: 문자열 'nan', 'ND', 'x', 빈 칸. Kapp Linné: 'nd'. 범례는 'NO= no permafrost left', "
                      "'X= no data' 다. 'NO' 는 자료 칸에 없다. 'ND' 의 뜻은 범례에 없다"),
        season_end_basis=("CALM 규약(eos_basis = protocol). 자료 파일에는 연도만 있고 관측 날짜가 없어 month 와 date 는 "
                          "빈 칸이다. 하위 지점별 실제 관측 시기는 확인하지 못했다"),
        columns_in_data_sheet=res["col_info"],
        sheet_comparison_abisko=res["sheet2_cmp"],
        file_comparison_kapp_linne=res["kapp_cmp"],
        duplicate_checks=dict(
            same_site_year_twice=0,
            same_value_between_sub_sites_in_a_year=res["same_year"],
            same_value_between_years_in_a_sub_site=res["same_site"],
            constant_runs=res["runs"],
            note=("값이 소수 끝까지 같은 경우를 찾았다. 지우지 않았고 해당 행의 notes 에 표시했다. "
                  "Bergfors 열과 같은 값은 Bergfors 가 점 자료에 없으므로 상대 열의 행에만 표시된다"),
        ),
        qc_summary=summ,
        sites=sites_meta,
        v3_relation=dict(
            parent_rows=[
                dict(calm_site="S2", v3_loc_id=17520, v3_region="CALM_Scandinavia", v3_source_id="F4_direct",
                     note="PANGAEA 972777 의 S2 연 값은 자료 파일 'summary' 시트의 'Mean all Valley sites' 와 맞는다"),
                dict(calm_site="S1", v3_loc_id=17569, v3_region="CALM_Svalbard", v3_source_id="F4_direct",
                     note="PANGAEA 972777 의 S1 연 값은 자료 파일의 'Average all sites' 와 맞는다(1990년 값은 74 대 100.4 로 다르다)"),
            ],
            rule=("v3 의 두 행은 하위 지점의 평균이다. 하위 지점 셀을 더하는 대상에서는 두 v3 행을 대상 셀에 넣지 않는다"
                  "(계획서 6B.3). 하위 지점 가운데 v3 F4_direct 셀과 체비쇼프 0.01° 이내인 것은 sites 의 v3_f4_dup 에 적었다"),
            n_sites_dup_with_v3_f4=sum(st["v3_f4_dup"] for st in res["sites"].values()),
        ),
        scope_note=("UNISCALM(N3)과 Ecogrid(IT1)의 지점 평균은 이 자료원에 넣지 않았다. calm_pangaea_v4add 에 있다. "
                    "Kapp Linné AL1, AL2 는 설명문에 좌표가 있으나 자료 파일에 열이 없다"),
        unverified_items=[
            "이용 약관. 재배포 허용 여부를 확인하지 못했다",
            "하위 지점 좌표와 Akerman & Johansson 2008 의 표 대조. 논문 본문을 열람하지 못했다",
            "qc_flag = coord 인 하위 지점(AB5, AB7, AB8, AB9)의 실제 좌표. 설명문의 좌표 또는 표고 가운데 어느 쪽이 틀렸는지 확인하지 못했다. "
            "AB7 은 위도의 분을 25 로 읽으면(68o25'88'') GTN-P 시추공 Katterjokk 와 0.7 km 안이고 DEM 표고가 461 m 이나 추정이므로 좌표를 고치지 않았다",
            "AB2, AB4 는 DEM 표고가 설명문 표고보다 34-40 m 낮다. AB2 는 Johansson 등 2011 의 시추공 표고 355 m 와 DEM 이 맞으므로 설명문 표고가 부정확한 것으로 보인다",
            "행별 관측 날짜와 월. 자료 파일에는 연도만 있다",
            "Abisko 하위 지점의 측점 수와 격자 크기. n_obs 는 1 로 두었다. Kapp Linné 의 1994년 이후 측점 수도 확인하지 못했다",
            "행 안 표준편차(alt_sd_cm). 자료에 없다",
            "AB4 의 2019-2021년 값이 150.0 cm 로 같다. 탐침 길이 상한인지 확인하지 못했다. right_censored 는 0 으로 두었다",
            "Abisko 결측 부호 'ND' 와 'nan' 의 뜻(영구동토 소멸인지 미관측인지). Johansson 등 2011 은 Katterjokk 의 영구동토가 사라졌다고 적었다",
            "Bergfors 하위 지점의 좌표. 설명문에 없다",
            "값이 소수 끝까지 같은 칸(duplicate_checks)이 입력 오류인지 여부",
            "Abisko 'data' 시트와 'Sheet2'(2002년 11월판)의 값 차이(sheet_comparison_abisko)의 원인",
            "CCI ALT, ERA5-Land, 토양 도일 공변량의 유효 여부. 이 단계에서는 붙이지 않았다",
        ],
    )
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # 화면 요약
    print(f"[{SRC_ID}] 원자료 칸 {res['n_raw']}, 점 자료 행 {len(pts)}, 1990년 이후 {summ['n_rows_1990plus']}, "
          f"주 집합 {summ['n_rows_main']}")
    print("  제외:", excluded)
    print("  결측 부호:", dict(res["miss_codes"]))
    print("  qc_flag:", summ["n_rows_by_qc_flag"])
    print("  하위 단위별 행(1990년 이후):", summ["n_rows_1990plus_by_subunit"])
    print("  셀과 블록(1990년 이후):", summ["cells_1990plus"])
    print("  셀과 블록(주 집합):", summ["cells_main"])
    print("  ALT(cm) 1990년 이후:", summ["alt_cm_rows_1990plus"])
    print("  ALT(cm) 주 집합:", summ["alt_cm_rows_main"])
    print("  좌표 의심 하위 지점:", summ["sites_coord_flagged"])
    for s in site_rows:
        print(f"  {s['site_id']:8s} {s['lat']:.6f} {s['lon']:.6f} 표고 {s['elev_stated_m']} DEM {s['elev_dem_m']} "
              f"창 {s['dem_win_range_m']} GTN-P {s['gtnp_ref_dist_km']} 셀 {s['cell_id']} 블록 {s['block']} "
              f"v3F4 {s['v3_f4_near_cheb_deg']} n90 {s['n_rows_1990plus']} 평균 {s['alt_mean_1990plus_cm']} "
              f"{s['coord_reason']}")
    print("  산출:", out_pts, out_meta, out_long, out_site, sep="\n    ")


if __name__ == "__main__":
    main()
