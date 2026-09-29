"""cusp_v1_1: CUSP v1.1 합성 관측표의 두 출처를 표준 점 자료로 만든다.

대상 출처는 Jorgenson_Kanevskiy_2025(Alaska Permafrost Soils Inventory and Thermokarst Monitoring
Database 2024 Update, Arctic Data Center)와 Petrone_etal_2016(서그린란드 Two Boat Lake 집수역,
PANGAEA 845258)이다.

근거
----
계획서 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B 절(LGD, 개정 10), 형식 정의
`data/processed/ext_labels/_schema.json`(버전 1.0), 자료원 계획 `_sources_plan.json` 의 cusp_v1_1 항목.

입력(모두 data/raw/cusp_v1_1/ 아래, 내려받은 곳은 SOURCE.md)
----
cusp_v1.1.csv                                   CUSP v1.1 관측표(12열, 79,389행). 주 입력
orig_adc_A27P8TG0G/tbl_Site_2024.csv            Jorgenson & Kanevskiy 2025 의 지점 표(원 저장소). 교란, 채취 방법, 지명
orig_adc_A27P8TG0G/REF_Site_Thermokarst_Types_2024.csv   열카르스트 유형 부호표
orig_adc_A27P8TG0G/REF_Site_Thermokarst_Stage_2024.csv   열카르스트 단계 부호표
repo_src/Petrone_etal_2016/GPR_profiles_with_depths_by_class.xlsx
                                                CUSP 저장소의 GPR 깊이 환산 표(원 픽, 식생 등급, 깊이)
orig_pangaea_845258/Petrone-etal_2016/Probe_transects/Probe_transects.xlsx
                                                PANGAEA 원본의 탐침 측선 표(기록 날짜 확인용)
data/processed/fidelity_base_v3.csv             읽기 전용. 좌표와 source_id 만 읽어 v3 와의 관계를 기록한다

규칙
----
1. source 가 Jorgenson_Kanevskiy_2025, Petrone_etal_2016 인 행만 쓴다. 나머지 출처는 제외 수에 센다.
2. method 는 tp → probe, gp → gpr, pit·pit_aug·aug → pit_core 다. 그 밖의 부호가 나오면 중단한다.
3. 라벨 값은 thaw_depth(cm)다. 단위 변환은 없다(CUSP 규약이 cm). 부동소수 잔차를 없애려고 소수 6자리로 맞춘다.
4. 단일 방문의 관측 월이 8월 또는 9월이면 label_def = direct_eos, 아니면 direct_dated 다.
   eos_basis 는 기록 날짜가 원자료에 있으면 record_date, CUSP 가 조사 날짜를 부여한 행(Petrone 의 GPR)은
   dataset_statement 다(Petrone 등 2016 본문: 탐침과 GPR 조사를 2011년 8월에 함께 했다).
5. thaw_depth 가 비고 pf_depth 만 있는 행(코어 기재의 영구동토 상한 깊이)은 alt_cm 에 pf_depth 를 넣고
   label_def = unknown 으로 둔다. 융해 깊이 관측이 아니므로 주 집합에 들어가지 않는다.
6. pf_observed = 0 인 행(관측 한계 안에서 영구동토 미검출)은 alt_cm 에 obs_limit 를 넣고 right_censored = 1 로 둔다.
   값은 하한이다. 영구동토가 없는 지점일 수 있으므로 notes 에 pf_observed=0 을 적는다.
7. 깊이 값이 하나도 없는 행(pf_observed = 1, thaw_depth·pf_depth 빈 칸)은 점 자료에 넣지 않고 제외 수에 센다.
8. 1990년 이전은 qc_flag 에 year_pre1990, 범위(0 < ALT <= 600 cm) 밖은 range 를 붙인다.
9. 교란 표시
   - Jorgenson & Kanevskiy: 원 지점 표의 DisturbClass, TkarstType, TKarstStage, TopoMicro 로 정한다(아래 disturb_from_aux).
     라벨 값은 쓰지 않는다. 열카르스트는 형식의 어휘에 맞춰 thermo_erosion 으로 적는다.
     지점 표와 날짜까지 맞는 기록을 찾지 못한 행은 disturbed = 0 으로 두고 notes 에 aux_site=unmatched 를 적는다.
   - Petrone 의 GPR: 5 m 칸 안 원 픽 가운데 식생 등급이 Water(호안, 포화 실트)인 픽이 하나라도 있으면
     disturbed = 1, disturb_type = water 다.
10. 행 단위
   - Jorgenson & Kanevskiy: CUSP 의 site_id(토양 단면 하나)와 연도.
   - Petrone: CUSP 의 한 행(탐침 점 하나 또는 GPR 5 m 칸 하나)이 한 위치다. site_id 는 <측선>_<측선 안 순번 3자리>다.
     순번은 CUSP 파일의 행 순서다. cusp_obs_id 를 notes 에 적어 추적한다.
11. 좌표는 CUSP 의 문자열을 그대로 쓴다(WGS84 십진 도, 서경 음수, 변환 없음).
12. macro 는 좌표 상자로 정한다(라벨 값을 쓰지 않는다). 상자 밖의 행이 나오면 중단한다.
13. CUSP 의 quality_flags 는 notes 의 cusp_flags 에 옮긴다(구분자는 '|').

산출
----
data/processed/ext_labels/cusp_v1_1_points.csv      (약관 때문에 커밋하지 않는다. 같은 폴더의 .gitignore)
data/processed/ext_labels/cusp_v1_1_meta.json
data/raw/cusp_v1_1/cusp_v1_1_selected_rows.csv       (두 출처의 CUSP 원행 전체와 대조 결과. 방문 단위 원자료)

실행(ROOT): OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/1_data_prep/parse_ext_cusp_v1_1.py
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

SRC_ID = "cusp_v1_1"
SRC_NAME = "CUSP v1.1 합성 관측표 가운데 Jorgenson & Kanevskiy 2025, Petrone 등 2016 출처 행"
URL = "https://github.com/jonschwenk/cusp/releases/tag/v1.1"
ACCESSED = "2026-09-29"

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
CSV = RAW_DIR / "cusp_v1.1.csv"
BIB = RAW_DIR / "cusp_sources_v1.1.bib"
RELEASE_INFO = RAW_DIR / "RELEASE_INFO.md"
AUX_SITE = RAW_DIR / "orig_adc_A27P8TG0G" / "tbl_Site_2024.csv"
AUX_TK = RAW_DIR / "orig_adc_A27P8TG0G" / "REF_Site_Thermokarst_Types_2024.csv"
AUX_STAGE = RAW_DIR / "orig_adc_A27P8TG0G" / "REF_Site_Thermokarst_Stage_2024.csv"
AUX_GPR = RAW_DIR / "repo_src" / "Petrone_etal_2016" / "GPR_profiles_with_depths_by_class.xlsx"
AUX_PROBE = (RAW_DIR / "orig_pangaea_845258" / "Petrone-etal_2016" / "Probe_transects"
             / "Probe_transects.xlsx")
SELECTED_OUT = RAW_DIR / f"{SRC_ID}_selected_rows.csv"
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"

EXPECTED_SHA256 = "75e2acba72087156b14b6d980c97a3cb440a52484a10f274da52736456e7b039"  # RELEASE_INFO.md
EXPECTED_ROWS = 79389

SRC_JK = "Jorgenson_Kanevskiy_2025"
SRC_PE = "Petrone_etal_2016"
TARGET_SOURCES = [SRC_JK, SRC_PE]
# 라벨 규칙이 이름을 적어 뺀 출처와 CUSP 출처 키의 대응(서지 파일의 제목으로 확인했다)
NAMED_ORIGINAL = {"CALM": "CALM", "Moore_et_al_2025": "ABoVE", "Veremeeva_etal_2025": "ALLena",
                  "Talucci_2024": "FireALT", "PERMOS_2024": "PERMOS", "Scheer_etal_2023": "Scheer"}
NAMED_UNKNOWN_METHOD = ["Smith_Burgess_2000", "Smith_Burgess_2002"]

METHOD_MAP = {"tp": "probe", "gp": "gpr", "pit": "pit_core", "pit_aug": "pit_core", "aug": "pit_core"}
EOS_MONTHS = (8, 9)
YEAR_MIN = 1990
ALT_MAX_CM = 600.0
CELL_DEG = 0.009           # 약 1 km 셀(6B.3)
DUP_DEG = 0.01             # v3 F4_direct 와의 중복 기준(체비쇼프)
BLOCK_DEG = 0.5
GPR_SPACING_M = 5.0

CITATION_CUSP = ("CUSP contributors (2026): CUSP: CommUnity near-Surface Permafrost data synthesis, version 1.1 "
                 "(released 2026-08-06). https://github.com/jonschwenk/cusp/releases/tag/v1.1 "
                 "(DOI 는 확인하지 못했다)")
ORIG = {
    SRC_JK: dict(
        citation=("Jorgenson, M. T.; Kanevskiy, M. (2025): Alaska Permafrost Soils Inventory and Thermokarst "
                  "Monitoring Database 2024 Update. Arctic Data Center, https://doi.org/10.18739/A27P8TG0G"),
        license_orig="CC0 1.0"),
    SRC_PE: dict(
        citation=("Petrone, J.; Sohlenius, G.; Johansson, E.; Lindborg, T.; Naslund, J.-O.; Stromgren, M.; "
                  "Brydsten, L. (2016): Using ground-penetrating radar, topography and classification of "
                  "vegetation to model the sediment and active layer thickness in a periglacial lake catchment, "
                  "Western Greenland [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.845258; "
                  "Earth Syst. Sci. Data 8, 663-677, https://doi.org/10.5194/essd-8-663-2016"),
        license_orig="CC BY-NC-SA 3.0"),
}
LICENSE_CUSP = "CUSP: all rights reserved (LANL notice, LICENSE.txt)"

# 좌표 상자(라벨 값을 쓰지 않는다). lat, lon 은 (최소, 최대)
REGION_BOXES = [
    dict(name="Alaska", country="United States", macro="Alaska", subunit="", lat=(54.0, 72.0),
         lon=(-170.0, -141.0), landmark=None),
    dict(name="Kangerlussuaq", country="Greenland", macro="NAtlantic", subunit="Greenland",
         lat=(66.5, 67.6), lon=(-52.5, -49.0), landmark="Kangerlussuaq"),
    dict(name="Bylot Island", country="Canada", macro="Canada", subunit="Bylot_Island",
         lat=(72.5, 73.8), lon=(-81.5, -77.5), landmark="Bylot"),
    dict(name="Ward Hunt Island", country="Canada", macro="Canada", subunit="Ward_Hunt_Island",
         lat=(82.9, 83.3), lon=(-75.5, -73.0), landmark="Ward Hunt"),
]

REQUIRED = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method",
            "label_def", "n_obs", "country", "macro", "citation", "license"]
EXTENDED = ["subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm",
            "coord_prec_deg", "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of",
            "qc_flag", "notes"]
COLUMNS = REQUIRED + EXTENDED

AUX_FIELDS = ["SoilMethod", "DisturbClass", "TkarstType", "TKarstStage", "TopoMicro", "WaterDep_cm",
              "WaterAbvBlw", "LocLandmark", "LocEcoregion"]
LAKE_CODES = {"DL", "SL", "LD", "LS", "LG"}           # 열카르스트 유형 부호 가운데 호수
TK_NONE = {"NA", "ND", "UD", ""}                      # 유형 없음, 미정, 미열화
STAGE_UNDEGRADED = {"UD", "PC", "PR"}                 # 미열화, 다각형 중심, 다각형 둔덕
STAGE_NONE = {"ND", "NA", ""}                         # 미정, 해당 없음
STAGE_LAKE = {"LKD", "LKS"}                           # 호수
DISTURB_ORDER = ["burned", "water", "thermo_erosion", "infrastructure"]


# ---------------------------------------------------------------- 공통
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def cell_index(lat: float, lon: float):
    ky = math.floor(lat / CELL_DEG)
    phi = math.radians((ky + 0.5) * CELL_DEG)
    kx = math.floor(lon * math.cos(phi) / CELL_DEG)
    return int(ky), int(kx)


def block_id(lat: float, lon: float) -> int:
    """v3 의 block 과 같은 식: floor(lat/0.5)·100000 + floor(lon/0.5)."""
    return int(math.floor(lat / BLOCK_DEG) * 100000 + math.floor(lon / BLOCK_DEG))


def region_of(lat: float, lon: float) -> dict:
    hit = [b for b in REGION_BOXES
           if b["lat"][0] <= lat <= b["lat"][1] and b["lon"][0] <= lon <= b["lon"][1]]
    if len(hit) != 1:
        raise ValueError(f"좌표 상자에 하나로 들어가지 않는 행: lat={lat}, lon={lon}, 후보 {len(hit)}개")
    return hit[0]


def qc_of(alt: float, year: int) -> str:
    f = []
    if not (0.0 < alt <= ALT_MAX_CM):
        f.append("range")
    if year < YEAR_MIN:
        f.append("year_pre1990")
    return ";".join(f) if f else "ok"


def label_of(month: int):
    return ("direct_eos", "record_date") if month in EOS_MONTHS else ("direct_dated", "none")


def fmt_num(x: float) -> str:
    """소수 6자리로 맞추고 뒤의 0 을 뗀다(부동소수 잔차 제거)."""
    s = f"{round(float(x), 6):.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def read_cusp() -> pd.DataFrame:
    got = sha256_of(CSV)
    if got != EXPECTED_SHA256:
        raise ValueError(f"cusp_v1.1.csv 의 sha256 이 RELEASE_INFO.md 와 다르다: {got}")
    d = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    exp_cols = ["cusp_obs_id", "source", "site_id", "lat", "lon", "date", "pf_observed", "thaw_depth",
                "pf_depth", "obs_limit", "method", "quality_flags"]
    if list(d.columns) != exp_cols:
        raise ValueError(f"열 구성이 다르다: {list(d.columns)}")
    if len(d) != EXPECTED_ROWS:
        raise ValueError(f"행 수가 다르다: {len(d)}")
    if d["cusp_obs_id"].duplicated().any():
        raise ValueError("cusp_obs_id 중복")
    return d


# ---------------------------------------------------------------- Jorgenson & Kanevskiy 보조 표
def read_aux_site() -> pd.DataFrame:
    o = pd.read_csv(AUX_SITE, dtype=str, keep_default_na=False, encoding="cp1252")
    o["date_iso"] = pd.to_datetime(o["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    o["sid_base"] = o["SiteIDYear"].str.replace(r"-\d{4}$", "", regex=True)
    o["thaw_num"] = pd.to_numeric(o["SoilThawDep_cm"], errors="coerce")
    for c in AUX_FIELDS:
        o[c] = o[c].str.strip()
    return o


def read_tk_codes():
    """열카르스트 유형 부호와 단계 부호(열화, 안정화, 재동결)를 돌려준다."""
    t = pd.read_csv(AUX_TK, dtype=str, keep_default_na=False)
    types = {c.strip().upper() for c in t["TkarstCode"]}
    g = pd.read_csv(AUX_STAGE, dtype=str, keep_default_na=False)
    stages = {c.strip().upper() for c in g["TK-Stage-Code"]}
    active = stages - STAGE_UNDEGRADED - STAGE_NONE - STAGE_LAKE
    return types, active


def match_aux(o: pd.DataFrame, by_id: dict, sid: str, date: str, thaw: float | None):
    """지점 표에서 (지점 식별자, 날짜)가 맞는 기록을 찾는다. 돌려주는 값은 (기록 또는 None, 대조 수준)."""
    idx = by_id.get(sid, [])
    if len(idx) == 0:
        return None, "unmatched_id"
    cand = o.loc[idx]
    cand = cand[cand["date_iso"] == date]
    if len(cand) == 0:
        return None, "unmatched_date"
    level = "id+date"
    if thaw is not None:
        c2 = cand[(cand["thaw_num"] - thaw).abs() < 0.005]
        if len(c2) > 0:
            cand, level = c2, "id+date+thaw"
    if len(cand) > 1:
        if cand[AUX_FIELDS].drop_duplicates().shape[0] > 1:
            return None, "ambiguous"
    return cand.iloc[0], level


def disturb_from_aux(rec, tk_codes: set, stage_active: set):
    """원 지점 표의 부호에서 교란 표시를 만든다. 돌려주는 값은 (disturbed, disturb_type, 모든 종류, 모르는 부호)."""
    dc = rec["DisturbClass"].lower()
    tk = rec["TkarstType"].upper()
    st = rec["TKarstStage"].upper()
    tm = rec["TopoMicro"].lower()
    kinds, unknown = set(), []
    if dc.startswith("h"):
        kinds.add("infrastructure")                      # Human(H*)
    if dc in ("nf", "nft"):
        kinds.add("burned")                              # Fire, Fire/Thermokarst
    if dc in ("ngt", "ngtm", "nft", "hct", "hfdt"):
        kinds.add("thermo_erosion")                      # Thermokarst 를 포함한 등급
    if dc == "w" or tm == "w" or tk in LAKE_CODES or st in STAGE_LAKE:
        kinds.add("water")                               # Water
    if st in stage_active:
        kinds.add("thermo_erosion")                      # 단계가 열화, 안정화, 재동결 가운데 하나로 적혔다
    elif st not in STAGE_UNDEGRADED and st not in STAGE_NONE and st not in STAGE_LAKE:
        unknown.append(f"tkarst_stage={rec['TKarstStage']}")
    if tk not in TK_NONE and tk not in LAKE_CODES:
        if tk in tk_codes:
            if st not in STAGE_UNDEGRADED:
                kinds.add("thermo_erosion")              # 열카르스트 지형이고 단계가 미열화로 적히지 않았다
        else:
            unknown.append(f"tkarst={rec['TkarstType']}")
    known_dc = dc in ("", "a", "nd", "u", "w", "n", "dc") or dc.startswith("h") or dc.startswith("n")
    if not known_dc:
        unknown.append(f"disturb_class={rec['DisturbClass']}")
    order = [k for k in DISTURB_ORDER if k in kinds]
    return (1 if order else 0), (order[0] if order else ""), order, unknown


def build_jk(d: pd.DataFrame):
    x = d[d["source"] == SRC_JK].copy().reset_index(drop=True)
    o = read_aux_site()
    tk_codes, stage_active = read_tk_codes()
    by_id: dict = {}
    for col in ("SiteID", "sid_base"):
        for i, v in o[col].items():
            if v != "":
                by_id.setdefault(v, [])
                if i not in by_id[v]:
                    by_id[v].append(i)

    bad_m = sorted(set(x["method"]) - set(METHOD_MAP))
    if bad_m:
        raise ValueError(f"대응표에 없는 method: {bad_m}")
    if not x["pf_observed"].isin(["0", "1"]).all():
        raise ValueError("pf_observed 값이 0, 1 이 아니다")

    rows, sel, excluded = [], [], {}
    stat = dict(aux_level={}, unknown_codes={}, landmark_mismatch=0, thaw_ne_pf=0)
    for r in x.itertuples(index=False):
        lat, lon = float(r.lat), float(r.lon)
        if lon >= 0:
            raise ValueError(f"동경 좌표: {r.site_id} {r.lon}")
        reg = region_of(lat, lon)
        year, month = int(r.date[:4]), int(r.date[5:7])
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.date):
            raise ValueError(f"날짜 형식: {r.date}")
        thaw = float(r.thaw_depth) if r.thaw_depth != "" else None
        pfd = float(r.pf_depth) if r.pf_depth != "" else None
        lim = float(r.obs_limit) if r.obs_limit != "" else None
        if thaw is not None and pfd is not None and abs(thaw - pfd) > 1e-9:
            stat["thaw_ne_pf"] += 1

        rec, level = match_aux(o, by_id, r.site_id, r.date, thaw)
        stat["aux_level"][level] = stat["aux_level"].get(level, 0) + 1
        notes = [f"cusp_obs_id={r.cusp_obs_id}", f"cusp_flags={r.quality_flags.replace(';', '|')}",
                 f"cusp_method={r.method}"]
        disturbed, dtype, subunit, qc_extra, site_name = 0, "", reg["subunit"], [], r.site_id
        aux_out = {c: "" for c in AUX_FIELDS}
        if rec is None:
            notes.append(f"aux_site={level}")
        else:
            aux_out = {c: rec[c] for c in AUX_FIELDS}
            disturbed, dtype, kinds, unknown = disturb_from_aux(rec, tk_codes, stage_active)
            for u in unknown:
                stat["unknown_codes"][u] = stat["unknown_codes"].get(u, 0) + 1
            notes.append(f"aux_site={level}")
            notes.append(f"soil_method={rec['SoilMethod']}")
            notes.append(f"disturb_class={rec['DisturbClass']}")
            notes.append(f"tkarst={rec['TkarstType']}/{rec['TKarstStage']}")
            notes.append(f"topo_micro={rec['TopoMicro']}")
            notes.append(f"water_cm={rec['WaterDep_cm']}({rec['WaterAbvBlw']})")
            if len(kinds) > 1:
                notes.append("disturb_all=" + "|".join(kinds))
            if rec["LocLandmark"] != "":
                notes.append(f"landmark={rec['LocLandmark']}")
                site_name = f"{rec['LocLandmark']} {r.site_id}"
            if reg["macro"] == "Alaska":
                subunit = rec["LocEcoregion"].replace(" ", "_")
            elif reg["landmark"] is not None and reg["landmark"].lower() not in rec["LocLandmark"].lower():
                stat["landmark_mismatch"] += 1
                qc_extra.append("coord")

        # 값의 종류
        if r.pf_observed == "0":
            if lim is None or thaw is not None or pfd is not None:
                raise ValueError(f"pf_observed = 0 행의 값 구성이 규약과 다르다: {r.cusp_obs_id}")
            alt, censored = lim, 1
            label_def, eos = label_of(month)
            notes.append("pf_observed=0")
            notes.append("alt_cm=obs_limit(lower bound; permafrost not detected within limit)")
        elif thaw is not None:
            alt, censored = thaw, 0
            label_def, eos = label_of(month)
        elif pfd is not None:
            alt, censored = pfd, 0
            label_def, eos = "unknown", "none"
            notes.append("alt_cm=pf_depth(cryostratigraphic permafrost top; thaw_depth blank)")
        else:
            key = "깊이 값 없음(pf_observed = 1, thaw_depth 와 pf_depth 모두 빈 칸)"
            excluded[key] = excluded.get(key, 0) + 1
            sel.append(dict(r._asdict(), points_site_id="", included=0, reason=key, aux_level=level, **aux_out))
            continue

        qc = qc_of(alt, year)
        if qc_extra:
            qc = ";".join(([] if qc == "ok" else qc.split(";")) + qc_extra)
        rows.append(dict(
            src_id=SRC_ID, site_id=r.site_id, site_name=site_name, lat=r.lat, lon=r.lon, year=year,
            month=month, alt_cm=fmt_num(alt), method=METHOD_MAP[r.method], label_def=label_def, n_obs=1,
            country=reg["country"], macro=reg["macro"], citation=CITATION_CUSP,
            license=f"{LICENSE_CUSP}; original: {ORIG[SRC_JK]['license_orig']}",
            subunit=subunit, date=r.date, eos_basis=eos, value_kind="single_visit", year_min="", year_max="",
            alt_sd_cm="", coord_prec_deg="0.01", disturbed=disturbed, disturb_type=dtype,
            right_censored=censored, orig_source=ORIG[SRC_JK]["citation"], dup_of="", qc_flag=qc,
            notes="; ".join(notes)))
        sel.append(dict(r._asdict(), points_site_id=r.site_id, included=1, reason="", aux_level=level, **aux_out))
    stat["coord_decimals_max"] = int(max(x["lat"].map(n_decimals).max(), x["lon"].map(n_decimals).max()))
    stat["aux_coord_decimals_max"] = int(max(o["LatWGS84"].map(n_decimals).max(),
                                             o["LonWGS84"].map(n_decimals).max()))
    # 원 지점 표에는 있으나 CUSP 에 없는 기록(대상 상자 안, 융해 깊이 유효, 영구동토 있음)
    extra = []
    have = set(zip(x["site_id"], x["date"]))
    for rr in o.itertuples(index=False):
        try:
            la, lo = float(rr.LatWGS84), float(rr.LonWGS84)
        except ValueError:
            continue
        if not (np.isfinite(rr.thaw_num) and 0 < rr.thaw_num < 999 and rr.SoilPFrost.lower() == "p"):
            continue
        box = [b for b in REGION_BOXES if b["lat"][0] <= la <= b["lat"][1] and b["lon"][0] <= lo <= b["lon"][1]]
        if len(box) != 1 or box[0]["macro"] == "Alaska":
            continue
        if (rr.SiteID, rr.date_iso) in have or (rr.sid_base, rr.date_iso) in have:
            continue
        extra.append(dict(SiteIDYear=rr.SiteIDYear, date=rr.date_iso, lat=la, lon=lo, thaw_cm=float(rr.thaw_num),
                          region=box[0]["name"], soil_method=rr.SoilMethod))
    stat["orig_rows_not_in_cusp_target_boxes"] = extra
    return pd.DataFrame(rows, columns=COLUMNS), sel, excluded, stat, len(x)


# ---------------------------------------------------------------- Petrone 보조 표
def rebuild_gpr_cells() -> pd.DataFrame:
    """CUSP 의 5 m 칸 집계를 다시 계산한다(cusp/data_utils.py 의 aggregate_gpr_points 와 같은 절차)."""
    from pyproj import Transformer
    xls = pd.ExcelFile(AUX_GPR)
    parts = []
    for sheet in xls.sheet_names:
        if sheet.upper().startswith("SUMMARY") or sheet.upper().startswith("VEG_"):
            continue
        df = xls.parse(sheet)
        num = int(re.findall(r"\d+", sheet)[0])
        parts.append(pd.DataFrame(dict(
            site_id=f"GPRT{num}", lat=df["Latitude"].astype(float), lon=df["Longitude"].astype(float),
            twt_ns=df["Two-way Travel Time [ns]"].astype(float), depth_cm=df["Depth_m"].astype(float) * 100.0,
            veg=df["Veg_Class_Name"].astype(str))))
    nat = pd.concat(parts, ignore_index=True)
    nat["cx"], nat["cy"] = 0, 0
    for sid, g in nat.groupby("site_id", sort=False):
        zone = int(np.clip(np.floor((g["lon"].median() + 180.0) / 6.0) + 1, 1, 60))
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + zone}", always_xy=True)
        xx, yy = tr.transform(g["lon"].tolist(), g["lat"].tolist())
        nat.loc[g.index, "cx"] = np.floor(np.asarray(xx) / GPR_SPACING_M).astype("int64")
        nat.loc[g.index, "cy"] = np.floor(np.asarray(yy) / GPR_SPACING_M).astype("int64")
    out = []
    for (sid, cx, cy), g in nat.groupby(["site_id", "cx", "cy"], sort=False):
        vc = g["veg"].value_counts()
        out.append(dict(site_id=sid, lat=g["lat"].mean(), lon=g["lon"].mean(), depth_cm=g["depth_cm"].mean(),
                        n=int(len(g)), sd=float(g["depth_cm"].std(ddof=1)) if len(g) > 1 else np.nan,
                        n_water=int((g["veg"] == "Water").sum()), veg_major=str(vc.index[0]),
                        veg_all="|".join(f"{k}:{int(v)}" for k, v in vc.items()),
                        v_m_ns=float((2.0 * g["depth_cm"] / 100.0 / g["twt_ns"]).mean())))
    return pd.DataFrame(out), len(nat)


def check_probe_dates() -> dict:
    """PANGAEA 원본의 탐침 표에 기록 날짜 열이 있는지와 값을 확인한다."""
    xls = pd.ExcelFile(AUX_PROBE)
    n, dates, cols_ok = 0, set(), True
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        cols_ok &= list(df.columns) == ["Latitude", "Longitude", "Active Layer Depth [m]", "Date"]
        n += len(df)
        dates |= set(pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"))
    return dict(n_points=int(n), n_sheets=len(xls.sheet_names), columns_as_expected=bool(cols_ok),
                dates=sorted(dates))


def build_petrone(d: pd.DataFrame):
    x = d[d["source"] == SRC_PE].copy().reset_index(drop=True)
    bad_m = sorted(set(x["method"]) - set(METHOD_MAP))
    if bad_m:
        raise ValueError(f"대응표에 없는 method: {bad_m}")
    if not (x["pf_observed"] == "1").all() or (x["thaw_depth"] == "").any() or (x["obs_limit"] != "").any():
        raise ValueError("Petrone 행의 값 구성이 예상과 다르다")
    cells, n_native = rebuild_gpr_cells()
    probe = check_probe_dates()
    stat = dict(gpr_native_picks=int(n_native), gpr_cells_rebuilt=int(len(cells)), gpr_rows_matched=0,
                gpr_rows_unmatched=0, gpr_rows_water=0, probe_workbook=probe)
    rows, sel = [], []
    seq: dict = {}
    for r in x.itertuples(index=False):
        lat, lon = float(r.lat), float(r.lon)
        reg = region_of(lat, lon)
        if reg["name"] != "Kangerlussuaq":
            raise ValueError(f"Petrone 행이 캉에를루수아크 상자 밖에 있다: {r.cusp_obs_id}")
        year, month = int(r.date[:4]), int(r.date[5:7])
        alt = float(r.thaw_depth)
        seq[r.site_id] = seq.get(r.site_id, 0) + 1
        sid = f"{r.site_id}_{seq[r.site_id]:03d}"
        notes = [f"cusp_obs_id={r.cusp_obs_id}", f"cusp_flags={r.quality_flags.replace(';', '|')}",
                 f"cusp_method={r.method}", f"transect={r.site_id}"]
        label_def, eos = label_of(month)
        n_obs, sd, disturbed, dtype, prec = 1, "", 0, "", "0.000001"
        if r.method == "gp":
            prec = "0.00005"                               # 5 m 칸 평균 좌표
            if label_def == "direct_eos":
                eos = "dataset_statement"                  # 날짜는 CUSP 가 부여(DA). 본문은 2011년 8월 조사로 적었다
            c = cells[(cells["site_id"] == r.site_id) & ((cells["lat"] - lat).abs() < 1e-9)
                      & ((cells["lon"] - lon).abs() < 1e-9) & ((cells["depth_cm"] - alt).abs() < 1e-6)]
            if len(c) == 1:
                c = c.iloc[0]
                stat["gpr_rows_matched"] += 1
                n_obs = int(c["n"])
                sd = "" if not np.isfinite(c["sd"]) else fmt_num(c["sd"])
                notes += [f"veg_class={c['veg_all']}", f"v_m_ns={round(float(c['v_m_ns']), 4)}",
                          "alt_cm=TWT*v/2 (v by vegetation class, Petrone et al. 2016 Table 2)"]
                if int(c["n_water"]) > 0:
                    disturbed, dtype = 1, "water"
                    stat["gpr_rows_water"] += 1
            else:
                stat["gpr_rows_unmatched"] += 1
                notes.append("gpr_cell=unmatched")
        else:
            if probe["dates"] != [r.date] or not probe["columns_as_expected"]:
                raise ValueError("탐침 원본 표의 날짜가 CUSP 와 다르다")
        rows.append(dict(
            src_id=SRC_ID, site_id=sid, site_name=f"Two Boat Lake {r.site_id}", lat=r.lat, lon=r.lon, year=year,
            month=month, alt_cm=fmt_num(alt), method=METHOD_MAP[r.method], label_def=label_def, n_obs=n_obs,
            country=reg["country"], macro=reg["macro"], citation=CITATION_CUSP,
            license=f"{LICENSE_CUSP}; original: {ORIG[SRC_PE]['license_orig']}",
            subunit=reg["subunit"], date=r.date, eos_basis=eos, value_kind="single_visit", year_min="",
            year_max="", alt_sd_cm=sd, coord_prec_deg=prec, disturbed=disturbed, disturb_type=dtype,
            right_censored=0, orig_source=ORIG[SRC_PE]["citation"], dup_of="", qc_flag=qc_of(alt, year),
            notes="; ".join(notes)))
        sel.append(dict(r._asdict(), points_site_id=sid, included=1, reason="", aux_level="",
                        **{c: "" for c in AUX_FIELDS}))
    return pd.DataFrame(rows, columns=COLUMNS), sel, {}, stat, len(x)


# ---------------------------------------------------------------- v3 대조와 셀
def nearest_f4(lat: np.ndarray, lon: np.ndarray):
    v3 = pd.read_csv(V3, usecols=["loc_id", "lat", "lon", "region", "source_id"])
    f4 = v3[v3["source_id"] == "F4_direct"].reset_index(drop=True)
    fla, flo = f4["lat"].to_numpy(), f4["lon"].to_numpy()
    dmin, imin = np.empty(len(lat)), np.empty(len(lat), dtype=int)
    for s in range(0, len(lat), 256):
        e = min(s + 256, len(lat))
        dd = np.maximum(np.abs(lat[s:e, None] - fla[None, :]), np.abs(lon[s:e, None] - flo[None, :]))
        imin[s:e] = dd.argmin(axis=1)
        dmin[s:e] = dd.min(axis=1)
    return dmin, f4.loc[imin, "loc_id"].to_numpy(), f4.loc[imin, "region"].to_numpy()


def main_mask(p: pd.DataFrame) -> pd.Series:
    return ((p["label_def"] == "direct_eos") & (p["qc_flag"] == "ok") & (p["right_censored"] == 0)
            & (p["disturbed"] == 0))


def cell_table(p: pd.DataFrame, use_v3: bool) -> pd.DataFrame:
    """주 집합의 1 km 셀 표. 셀 값은 위치별 다년 평균의 평균, 셀 좌표는 위치 좌표의 평균이다."""
    m = p[main_mask(p)].copy()
    m["latf"], m["lonf"], m["altf"] = m["lat"].astype(float), m["lon"].astype(float), m["alt_cm"].astype(float)
    site = m.groupby("site_id").agg(lat=("latf", "mean"), lon=("lonf", "mean"), alt=("altf", "mean"),
                                    macro=("macro", "first"), subunit=("subunit", "first"),
                                    method=("method", "first"), n_years=("year", "nunique"),
                                    year_min=("year", "min"), year_max=("year", "max")).reset_index()
    idx = [cell_index(a, b) for a, b in zip(site["lat"], site["lon"])]
    site["ky"], site["kx"] = [i[0] for i in idx], [i[1] for i in idx]
    cell = site.groupby(["macro", "ky", "kx"]).agg(
        subunit=("subunit", lambda s: "|".join(sorted(set(s)))), lat=("lat", "mean"), lon=("lon", "mean"),
        alt_cm=("alt", "mean"), n_sites=("site_id", "size"),
        methods=("method", lambda s: "|".join(f"{k}:{v}" for k, v in s.value_counts().sort_index().items())),
        year_min=("year_min", "min"), year_max=("year_max", "max")).reset_index()
    cell["block"] = [block_id(a, b) for a, b in zip(cell["lat"], cell["lon"])]
    if use_v3 and len(cell) > 0:
        dmin, loc, reg = nearest_f4(cell["lat"].to_numpy(), cell["lon"].to_numpy())
        cell["nearest_f4_cheb_deg"], cell["nearest_f4_loc_id"], cell["nearest_f4_region"] = dmin, loc, reg
        cell["dropped_by_v3_f4_rule"] = dmin <= DUP_DEG
    return cell


def dist(s: pd.Series) -> dict:
    s = s.astype(float)
    if len(s) == 0:
        return dict(n=0)
    return dict(n=int(len(s)), min=round(float(s.min()), 2), median=round(float(s.median()), 2),
                max=round(float(s.max()), 2), mean=round(float(s.mean()), 2))


def count(p: pd.DataFrame, by) -> dict:
    g = p.groupby(by).size()
    if isinstance(by, list):
        return {"|".join(str(v) for v in k): int(n) for k, n in g.items()}
    return {str(k): int(n) for k, n in g.items()}


def summarize(p: pd.DataFrame, cells: pd.DataFrame) -> dict:
    m = p[main_mask(p)]
    by_sub = {}
    for (macro, sub), g in p.groupby(["macro", "subunit"]):
        if macro == "Alaska":
            continue
        gm = g[main_mask(g)]
        day = pd.to_datetime(g["date"])
        by_sub[f"{macro}|{sub}"] = dict(
            n_rows=int(len(g)), n_sites=int(g["site_id"].nunique()), n_rows_main=int(len(gm)),
            n_by_label_def=count(g, "label_def"), n_by_method=count(g, "method"),
            n_disturbed=int(g["disturbed"].sum()), date_min=str(g["date"].min()), date_max=str(g["date"].max()),
            n_rows_aug_day_1_to_10=int(((day.dt.month == 8) & (day.dt.day <= 10)).sum()),
            alt_cm_all=dist(g["alt_cm"]), alt_cm_main=dist(gm["alt_cm"]),
            alt_cm_main_by_method={k: dist(v["alt_cm"]) for k, v in gm.groupby("method")})
    cell_sum = {}
    if len(cells) > 0:
        for macro, g in cells.groupby("macro"):
            e = dict(n_cells=int(len(g)), n_blocks=int(g["block"].nunique()), alt_cm_cells=dist(g["alt_cm"]))
            if "dropped_by_v3_f4_rule" in g:
                keep = g[~g["dropped_by_v3_f4_rule"]]
                e.update(n_cells_dropped_by_v3_f4_rule=int(g["dropped_by_v3_f4_rule"].sum()),
                         n_cells_new=int(len(keep)), n_blocks_new=int(keep["block"].nunique()),
                         nearest_f4_cheb_deg_min=round(float(g["nearest_f4_cheb_deg"].min()), 4))
            cell_sum[str(macro)] = e
    return dict(
        n_sites=int(p["site_id"].nunique()), n_rows=int(len(p)), n_rows_main=int(len(m)),
        main_set_rule="label_def = direct_eos, qc_flag = ok, right_censored = 0, disturbed = 0",
        n_sites_by_macro={str(k): int(v) for k, v in p.groupby("macro")["site_id"].nunique().items()},
        n_rows_by_macro=count(p, "macro"), n_rows_main_by_macro=count(m, "macro"),
        n_rows_by_method=count(p, "method"), n_rows_main_by_method=count(m, "method"),
        n_rows_by_label_def=count(p, "label_def"), n_rows_by_qc_flag=count(p, "qc_flag"),
        n_rows_by_macro_label_def=count(p, ["macro", "label_def"]),
        n_rows_by_month=count(p, "month"),
        n_right_censored=int(p["right_censored"].sum()), n_disturbed=int(p["disturbed"].sum()),
        n_disturbed_by_type=count(p[p["disturbed"] == 1], "disturb_type"),
        year_range=[int(p["year"].min()), int(p["year"].max())],
        year_range_main=[int(m["year"].min()), int(m["year"].max())],
        alt_cm_rows_all=dist(p["alt_cm"]), alt_cm_rows_main=dist(m["alt_cm"]),
        alt_cm_rows_main_by_macro={str(k): dist(v["alt_cm"]) for k, v in m.groupby("macro")},
        by_macro_subunit_outside_alaska=by_sub, cells_main_by_macro=cell_sum)


def other_sources_outside_alaska(d: pd.DataFrame) -> list:
    """라벨 규칙에 없는 출처 가운데 알래스카 상자 밖에 융해 깊이 값이 있는 것(참고용)."""
    x = d[~d["source"].isin(TARGET_SOURCES) & (d["thaw_depth"] != "")].copy()
    la, lo = x["lat"].astype(float), x["lon"].astype(float)
    x = x[~((la >= 54) & (la <= 72) & (lo >= -170) & (lo <= -141))]
    out = []
    for s, g in x.groupby("source"):
        la, lo = g["lat"].astype(float), g["lon"].astype(float)
        out.append(dict(source=s, n_rows=int(len(g)), lat_min=round(float(la.min()), 2),
                        lat_max=round(float(la.max()), 2), lon_min=round(float(lo.min()), 2),
                        lon_max=round(float(lo.max()), 2), methods=count(g, "method"),
                        date_min=str(g["date"].min()), date_max=str(g["date"].max()),
                        in_label_rule=("named_original" if s in NAMED_ORIGINAL else
                                       "named_unknown_method" if s in NAMED_UNKNOWN_METHOD else "not_named")))
    return out


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def file_entry(path: Path, source_url: str, note: str = "") -> dict:
    e = dict(name=str(path.relative_to(ROOT)), size_bytes=path.stat().st_size, sha256=sha256_of(path),
             source_url=source_url)
    if note:
        e["note"] = note
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-v3", action="store_true", help="v3 표 대조를 건너뛴다")
    args = ap.parse_args()

    d = read_cusp()
    n_raw = len(d)
    jk, sel_jk, exc_jk, stat_jk, n_jk = build_jk(d)
    pe, sel_pe, exc_pe, stat_pe, n_pe = build_petrone(d)
    pts = pd.concat([jk, pe], ignore_index=True)

    # 제외 수
    src_n = d["source"].value_counts()
    excluded = {}
    named = [s for s in NAMED_ORIGINAL if s in src_n.index]
    excluded["원자료원을 쓰는 출처(" + ", ".join(f"{s}={NAMED_ORIGINAL[s]}" for s in named) + ")"] = \
        int(src_n[named].sum())
    excluded["방법 불명 출처(" + ", ".join(NAMED_UNKNOWN_METHOD) + ")"] = \
        int(src_n[[s for s in NAMED_UNKNOWN_METHOD if s in src_n.index]].sum())
    rest = [s for s in src_n.index if s not in TARGET_SOURCES and s not in NAMED_ORIGINAL
            and s not in NAMED_UNKNOWN_METHOD]
    excluded[f"라벨 규칙에 없는 그 밖의 출처({len(rest)}개)"] = int(src_n[rest].sum())
    for k, v in {**exc_jk, **exc_pe}.items():
        excluded["선택 출처 안: " + k] = int(v)
    assert n_raw == len(pts) + sum(excluded.values())
    assert n_jk + n_pe == len(sel_jk) + len(sel_pe)

    # 형식 확인
    assert list(pts.columns) == COLUMNS
    assert (pts[REQUIRED].astype(str) != "").all().all()
    assert not pts.duplicated(["site_id", "year"]).any()
    assert pts["method"].isin(["probe", "thaw_tube", "frost_tube", "borehole_temp", "gpr", "pit_core",
                               "other", "unknown"]).all()
    assert pts["label_def"].isin(["direct_eos", "direct_dated", "temp_derived", "unknown"]).all()
    assert pts["lon"].astype(float).between(-180, 0).all() and pts["lat"].astype(float).between(54, 84).all()
    assert pts["disturb_type"].isin(["", "burned", "water", "thermo_erosion", "infrastructure"]).all()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{SRC_ID}_points.csv"
    pts.to_csv(out_csv, index=False, encoding="utf-8")
    pd.DataFrame(sel_jk + sel_pe).to_csv(SELECTED_OUT, index=False, encoding="utf-8")

    cells = cell_table(pts, use_v3=not args.no_v3)
    summ = summarize(pts, cells)
    site_v3 = None
    if not args.no_v3:
        s = pts.drop_duplicates("site_id")
        dmin, _, _ = nearest_f4(s["lat"].astype(float).to_numpy(), s["lon"].astype(float).to_numpy())
        s = s.assign(near=dmin <= DUP_DEG)
        site_v3 = dict(rule="위치 좌표가 v3 F4_direct 셀과 체비쇼프 0.01° 이내인 위치 수(참고용. 제외 판정은 셀 단위다)",
                       n_sites_within_0p01_by_macro={str(k): int(v) for k, v in s.groupby("macro")["near"].sum().items()},
                       n_sites_by_macro={str(k): int(v) for k, v in s.groupby("macro").size().items()})
    cells_out = cells[cells["macro"] != "Alaska"].round(6).to_dict(orient="records")

    repo = "https://raw.githubusercontent.com/jonschwenk/cusp/v1.1/"
    rel = "https://github.com/jonschwenk/cusp/releases/download/v1.1/"
    adc = "https://arcticdata.io/metacat/d1/mn/v2/object/"
    meta = dict(
        src_id=SRC_ID, name=SRC_NAME, url=URL, doi="", accessed=ACCESSED,
        doi_note="CUSP 자체의 DOI 는 확인하지 못했다. 원 출처의 DOI 는 10.18739/A27P8TG0G, 10.1594/PANGAEA.845258 이다",
        files=[
            file_entry(CSV, rel + "cusp_v1.1.csv", "주 입력. sha256 이 RELEASE_INFO.md 의 값과 같다"),
            file_entry(BIB, rel + "cusp_sources_v1.1.bib"),
            file_entry(RELEASE_INFO, rel + "RELEASE_INFO.md"),
            file_entry(AUX_SITE, adc + "urn%3Auuid%3A17a73344-cf00-468f-a51f-cf2196e3f8c2",
                       "Arctic Data Center doi:10.18739/A27P8TG0G 의 tbl_Site_2024.csv(CC0 1.0)"),
            file_entry(AUX_TK, adc + "urn%3Auuid%3A75f53252-7b60-465e-ba96-19c74ad4ddd1",
                       "같은 자료집의 열카르스트 유형 부호표"),
            file_entry(AUX_STAGE, adc + "urn%3Auuid%3Ad7f9363c-6dd5-472c-9d87-b4f2388ba5df",
                       "같은 자료집의 열카르스트 단계 부호표"),
            file_entry(AUX_GPR, repo + "data/Petrone_etal_2016/GPR/GPR_profiles_with_depths_by_class.xlsx",
                       "CUSP 저장소의 파생 표. PANGAEA 원본(GPR_profiles.xlsx)에는 왕복 주시만 있다"),
            file_entry(AUX_PROBE, "https://hs.pangaea.de/model/grasp/Petrone-etal_2016.zip",
                       "PANGAEA 845258 원본 압축 파일 안의 표(CC BY-NC-SA 3.0)"),
        ],
        license="CUSP 저장소: LICENSE.txt 가 저작권 고지(© 2026 Triad National Security, LLC. All rights reserved)이고 "
                "공개 이용 허락 조항이 없다. 원 출처: Jorgenson & Kanevskiy 2025 는 CC0 1.0, Petrone 등 2016 은 "
                "CC BY-NC-SA 3.0(PANGAEA 메타데이터)",
        license_basis="GitHub API 의 license 는 NOASSERTION. 저장소 LICENSE.txt 와 README 의 License 절을 읽었다. "
                      "원 출처 약관은 Arctic Data Center EML 의 intellectualRights 와 PANGAEA JSON-LD 의 license 값이다",
        redistribution="점 자료(cusp_v1_1_points.csv)는 커밋하지 않는다(data/processed/ext_labels/.gitignore). "
                       "이 메타 파일에는 원자료 행을 넣지 않았다(집계 수와 대상 지역의 셀 요약만 있다)",
        citation=CITATION_CUSP, citation_original=[ORIG[SRC_JK]["citation"], ORIG[SRC_PE]["citation"]],
        citation_note="CUSP 는 CUSP 와 원 출처를 함께 인용하라고 요구한다(README, CITATION.cff)",
        parser=f"scripts/1_data_prep/parse_ext_{SRC_ID}.py",
        parser_sha256=sha256_of(Path(__file__).resolve()),
        parser_git_commit=git_head(),
        parser_git_note="파서 파일은 이 커밋에 들어 있지 않다(작업 트리의 새 파일). 값은 실행 시점의 HEAD 다",
        schema_version="1.0",
        n_rows_raw=int(n_raw), n_rows_raw_selected_sources={SRC_JK: int(n_jk), SRC_PE: int(n_pe)},
        n_sources_raw=int(d["source"].nunique()),
        n_rows_points=int(len(pts)),
        n_by_label_def=count(pts, "label_def"), n_by_method=count(pts, "method"),
        n_excluded_by_reason=excluded,
        n_excluded_by_source={str(k): int(v) for k, v in src_n.items() if k not in TARGET_SOURCES},
        coordinate_conversion="변환 없음. CUSP 의 lat, lon 은 WGS84 십진 도이고 서경은 음수다. 문자열을 그대로 옮겼다. "
                              "선택한 행의 경도는 모두 음수(-165.62 에서 -50.07), 위도는 58.67 에서 83.09 다",
        coord_prec_rule="Jorgenson & Kanevskiy 행은 0.01(CUSP 와 원 지점 표 모두 소수 2자리). Petrone 탐침 행은 "
                        "0.000001(원본 표기 소수 6자리). Petrone GPR 행은 0.00005(5 m 칸 안 원 픽 좌표의 평균)",
        units="thaw_depth, pf_depth, obs_limit 는 cm(CUSP 자료 규약). 단위 변환 없음. Petrone 원본은 m 이고 CUSP 가 "
              "100 을 곱했다(탐침 0.7 m → 70 cm 확인). alt_cm 은 소수 6자리로 맞췄다",
        missing_code="CUSP 는 빈 칸. 원 지점 표의 999(융해 깊이), 9999(관측 깊이)는 CUSP 가 결측으로 처리했다",
        season_end_basis="기록 날짜의 월이 8월 또는 9월이면 direct_eos. Jorgenson & Kanevskiy 와 Petrone 탐침은 "
                         "record_date(원자료에 방문 날짜가 있다). Petrone GPR 은 dataset_statement(CUSP 가 "
                         "2011-08-18 을 부여했고 본문은 2011년 8월에 탐침과 함께 조사했다고 적었다)",
        row_unit="Jorgenson & Kanevskiy: 토양 단면(site_id)과 연도. Petrone: 탐침 점 하나 또는 GPR 5 m 칸 하나",
        disturb_rule=dict(
            jorgenson_kanevskiy="원 지점 표의 DisturbClass 가 H* 이면 infrastructure, Nf·Nft 이면 burned, "
                                "Ngt·Ngtm·Nft·Hct·Hfdt 이면 thermo_erosion, W 이거나 TopoMicro 가 W 이거나 "
                                "TkarstType 이 호수 부호(DL, SL, LD, LS, LG)이거나 TKarstStage 가 LKD, LKS 이면 "
                                "water. TkarstType 이 부호표의 열카르스트 지형이고 TKarstStage 가 UD, PC, PR 이 "
                                "아니면 thermo_erosion. TKarstStage 가 부호표의 열화, 안정화, 재동결 단계이면 "
                                "유형과 무관하게 thermo_erosion. "
                                "여러 종류면 burned, water, thermo_erosion, infrastructure 순으로 하나를 적고 "
                                "전체는 notes 의 disturb_all 에 적는다. 열카르스트를 thermo_erosion 에 넣은 것은 "
                                "형식의 어휘에 맞춘 파서의 해석이다",
            petrone="GPR 5 m 칸 안 원 픽의 식생 등급에 Water 가 하나라도 있으면 water",
            fluvial_coastal_eolian="Ng, Ngf*, Ngm*, Nge*, Ngl, Ngd 등 지형 과정 등급은 표시하지 않고 notes 에 부호만 적는다"),
        jorgenson_kanevskiy_checks=dict(
            aux_match_level=stat_jk["aux_level"], unknown_codes=stat_jk["unknown_codes"],
            landmark_mismatch=stat_jk["landmark_mismatch"], n_rows_thaw_ne_pf_depth=stat_jk["thaw_ne_pf"],
            coord_decimals_max_cusp=stat_jk["coord_decimals_max"],
            coord_decimals_max_orig_site_table=stat_jk["aux_coord_decimals_max"],
            orig_rows_not_in_cusp_target_boxes=stat_jk["orig_rows_not_in_cusp_target_boxes"]),
        petrone_checks=stat_pe,
        qc_summary=summ, cells_main_outside_alaska=cells_out, v3_site_relation=site_v3,
        other_sources_outside_alaska=other_sources_outside_alaska(d),
        unverified_items=[
            "CUSP 자료의 재배포 허용 여부. LICENSE.txt 는 저작권 고지이고 공개 이용 허락 조항이 없다. 저장소 관리자에게 "
            "확인하지 않았다",
            "CUSP 의 DOI(Zenodo 기록은 접근 제한이라 보지 못했다)",
            "Jorgenson & Kanevskiy 좌표의 실제 위치 정확도. 공개된 원 지점 표도 소수 2자리(약 1.1 km)라서 같은 "
            "조사 구역의 단면들이 같은 좌표를 갖는다. 1 km 셀 배정이 한 칸 어긋날 수 있다",
            "Jorgenson & Kanevskiy 의 융해 깊이 측정 기구. CUSP 는 method 를 pit_aug 로 고정했고 MU(방법 근사) 표시를 "
            "붙였다. 원 지점 표의 SoilMethod 는 토양 채취 방법이고 융해 깊이의 측정 기구를 따로 적지 않았다",
            "Petrone GPR 깊이 환산 표(GPR_profiles_with_depths_by_class.xlsx)를 만든 절차. PANGAEA 원본에는 왕복 "
            "주시만 있고 식생 등급과 깊이 열은 CUSP 저장소 파일에만 있다. 속도 값은 Petrone 등 2016 표 2 와 같다",
            "Petrone GPR 조사의 날짜(일). 본문은 2011년 8월이라고만 적었고 2011-08-18 은 CUSP 가 부여했다",
            "Petrone 탐침 점 수. 본문은 83점, 원본 표와 CUSP 는 76점이다. 차이의 이유는 확인하지 못했다",
            "8월 1–10일 관측(바일럿섬, 워드헌트섬)이 계절 말 값에 얼마나 가까운지. 규칙(관측 월 8–9월)으로는 "
            "direct_eos 다",
            "다른 ext_labels 자료원과의 위치 중복(dup_of 는 비워 두었다). v4 조립 단계의 일이다",
            "ERA5-Land 육지 폴백, CCI ALT 유효 여부, 공변량 부착 결과. v4 조립 단계의 일이다",
        ])
    out_meta = OUT_DIR / f"{SRC_ID}_meta.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(f"[원자료] 행 {n_raw}, 출처 {d['source'].nunique()}개, 선택 출처 행 {n_jk} + {n_pe}")
    print(f"[점 자료] {out_csv.relative_to(ROOT)}: 행 {len(pts)}, 위치 {summ['n_sites']}")
    print("[제외]", json.dumps(excluded, ensure_ascii=False))
    print("[label_def]", meta["n_by_label_def"], "[method]", meta["n_by_method"])
    print("[macro 위치]", summ["n_sites_by_macro"], "[macro 행]", summ["n_rows_by_macro"])
    print("[macro 주 집합 행]", summ["n_rows_main_by_macro"])
    print("[qc_flag]", summ["n_rows_by_qc_flag"], "[검열]", summ["n_right_censored"], "[교란]",
          summ["n_disturbed"], summ["n_disturbed_by_type"])
    print("[연도]", summ["year_range"], "[ALT 전체 행]", summ["alt_cm_rows_all"])
    print("[ALT 주 집합 행]", summ["alt_cm_rows_main"], summ["alt_cm_rows_main_by_macro"])
    print("[보조 표 대조]", stat_jk["aux_level"], "모르는 부호", stat_jk["unknown_codes"],
          "지명 불일치", stat_jk["landmark_mismatch"])
    print("[Petrone]", {k: v for k, v in stat_pe.items()})
    for k, v in summ["by_macro_subunit_outside_alaska"].items():
        print("[대상]", k, json.dumps(v, ensure_ascii=False))
    print("[셀]", json.dumps(summ["cells_main_by_macro"], ensure_ascii=False))
    if len(cells_out) > 0:
        print(pd.DataFrame(cells_out).to_string(index=False))
    print("[원 지점 표에만 있는 대상 지역 기록]", len(stat_jk["orig_rows_not_in_cusp_target_boxes"]))
    if site_v3 is not None:
        print("[v3 대조]", json.dumps(site_v3, ensure_ascii=False))


if __name__ == "__main__":
    main()
