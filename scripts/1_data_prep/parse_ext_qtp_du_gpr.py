"""ext_labels 파서: qtp_du_gpr (Du 등 2026, Zenodo 21999366, 청장고원 GPR·인력 시추 ALT 조사점).

계획서: docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B 절(LGD, 개정 10).
형식 정의: data/processed/ext_labels/_schema.json (버전 1.0).

입력
- data/raw/qtp_du_gpr/Code_Archive_QTB_ALM_v2s.zip (Zenodo 기록 21999366 의 유일한 파일, 36,297,128 바이트).
  없으면 탐색 단계 사본(data/raw/qtp_alt_open_2026-09-29/zenodo21999366/)을 복사한다.
  둘 다 없으면 내려받기 명령을 출력하고 끝낸다(이 스크립트는 네트워크를 쓰지 않는다).
- zip 안의 04_sentinel1_gee/探坑_GPR活动层水分调查点.xlsx 첫 시트.
  열: 编号(식별자), 年份(연도), 数据类型(방법), Lon, Lat, ALT/m, GPR波速(m/ns), VWC_活动层.
  zip 항목 이름은 UTF-8 표시(flag 0x800)가 없으면 cp437 → utf-8 로 되돌린다.
- data/processed/fidelity_base_v3.csv (읽기 전용, lat·lon·region·source_id·block 열만).
  v3 셀과의 거리와 새 셀 수의 예비 추정에만 쓴다. v3 는 고치지 않는다.

라벨 규칙(6B.2, 6B.3)
- ALT 값이 있는 행만 점 자료에 넣는다. 探坑(탐갱) 행과 ALT 가 빈 GPR 행은 넣지 않는다.
- method: GPR → gpr, 人工钻(인력 시추) → pit_core.
- label_def = direct_eos, eos_basis = dataset_statement.
  근거는 시트 이름(最大融化期, 최대 융해기)과 동봉 원고 초안의 문장
  "collected during 2009-2024 peak-thaw seasons (September-October)" 이다.
  점별 관측 날짜는 없다. month 와 date 는 빈 칸이다.
- alt_cm = ALT/m × 100 (십진 연산). 범위 QC 는 0 < alt_cm <= 600.
- 우측 절단 표기('>')는 원자료에 없다. right_censored = 0.
- disturbed = 0 은 '원자료에 교란 표기가 없다'는 뜻이다. 현장 교란 부재를 확인한 값이 아니다.

산출
- data/processed/ext_labels/qtp_du_gpr_points.csv  표준 점 자료(30열)
- data/processed/ext_labels/qtp_du_gpr_meta.json   메타와 품질 요약
- data/raw/qtp_du_gpr/extracted/                   대상 xlsx, LICENSE, README.md, 원고 초안
- data/raw/qtp_du_gpr/survey_points_all_rows.csv   시트 전체 264행과 제외 사유

실행(ROOT, 스레드 1개):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/1_data_prep/parse_ext_qtp_du_gpr.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_ID = "qtp_du_gpr"
RAW = ROOT / "data" / "raw" / SRC_ID
ZIP_NAME = "Code_Archive_QTB_ALM_v2s.zip"
ZIP_PATH = RAW / ZIP_NAME
LOCAL_COPY = ROOT / "data" / "raw" / "qtp_alt_open_2026-09-29" / "zenodo21999366" / ZIP_NAME
OUT_DIR = ROOT / "data" / "processed" / "ext_labels"
POINTS = OUT_DIR / f"{SRC_ID}_points.csv"
META = OUT_DIR / f"{SRC_ID}_meta.json"
RAW_DUMP = RAW / "survey_points_all_rows.csv"
EXTRACT = RAW / "extracted"
V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
SCHEMA = OUT_DIR / "_schema.json"

URL = "https://zenodo.org/records/21999366"
FILE_URL = "https://zenodo.org/api/records/21999366/files/Code_Archive_QTB_ALM_v2s.zip/content"
DOI = "10.5281/zenodo.21999366"
NCDC_DOI = "10.12072/ncdc.permafrost.db7703.2026"
ZENODO_MD5 = "6b1b19d5e0fb04df8f795d4d0fa51913"   # Zenodo API 의 파일 checksum(2026-09-29 조회)
ZENODO_SIZE = 36297128
ACCESSED = "2026-09-29"

TARGET = "Code_Archive_QTB_ALM/04_sentinel1_gee/探坑_GPR活动层水分调查点.xlsx"
EXTRA = {
    "Code_Archive_QTB_ALM/LICENSE": "LICENSE",
    "Code_Archive_QTB_ALM/README.md": "README.md",
    "Code_Archive_QTB_ALM/03_final_pipeline/Tables/Paper14_ESSD_revision_text_draft.md":
        "Paper14_ESSD_revision_text_draft.md",
    "Code_Archive_QTB_ALM/03_final_pipeline/Results/all_samples_342.csv": "all_samples_342.csv",
}
HEADER = ["编号", "年份", "数据类型", "Lon", "Lat", "ALT/m", "GPR波速", "VWC_活动层"]
METHOD_MAP = {"GPR": "gpr", "人工钻": "pit_core"}
PIT = "探坑"

LICENSE = "MIT (Zenodo record 10.5281/zenodo.21999366, archive LICENSE file)"
CITATION = (
    "Du, E., Wu, T., Dai, L., Ran, Y., Min, Y., Zhao, L., Wang, L., Yao, J., Li, Z., Feng, K., Xiao, Y., "
    "Hu, G., Zou, D., Liu, G., Xing, Z., Chen, J., Zhu, X. (2026). Code for \"A first 90 m resolution active "
    "layer moisture dataset across the Qinghai-Tibet Plateau permafrost region derived from multi-model machine "
    "learning fusion\" (version 2.0.0). Zenodo. https://doi.org/10.5281/zenodo.21999366. "
    "Field data: https://doi.org/10.12072/ncdc.permafrost.db7703.2026. "
    "Manuscript: Earth System Science Data (revision stage, DOI not confirmed)."
)
ORIG_SOURCE = "Du et al. 2026 field survey (NCDC doi:10.12072/ncdc.permafrost.db7703.2026)"

# Tibet 의 지리 범위(6B.4)와 하위 단위 규칙. 라벨 값을 쓰지 않는다.
# 하위 단위의 경계는 탐색 단계 초안(data/raw/qtp_alt_open_2026-09-29/derived_draft/newcells.py)과 같다.
TIBET_BOX = dict(lat_min=26.0, lat_max=40.0, lon_min=73.0, lon_max=105.0)
SUBUNIT_RULE = [
    ("Qilian", "치롄", dict(lat_min=37.0, lat_max=40.0, lon_min=96.0, lon_max=103.0, lon_max_incl=True)),
    ("YellowSrc_East", "황하원·동부", dict(lat_min=33.5, lat_max=36.5, lon_min=95.0, lon_max=101.0, lon_max_incl=True)),
    ("QTEC", "회랑", dict(lat_min=31.0, lat_max=36.2, lon_min=90.0, lon_max=95.0, lon_max_incl=False)),
    ("West_QTP", "서부", dict(lat_min=30.0, lat_max=38.0, lon_min=77.0, lon_max=90.0, lon_max_incl=False)),
]
SHIFT_NOTE_DEG = 0.001     # 같은 식별자의 연도 간 좌표 차이가 이 값을 넘으면 notes 에 적는다
DUP_DEG = 0.01             # v3 셀과의 체비쇼프 거리 기준(6B.3)
CELL = 0.009               # 약 1 km 셀(6B.3)


def subunit_of(lat: float, lon: float) -> str:
    for name, _, b in SUBUNIT_RULE:
        lon_ok = (b["lon_min"] <= lon <= b["lon_max"]) if b["lon_max_incl"] else (b["lon_min"] <= lon < b["lon_max"])
        if b["lat_min"] <= lat <= b["lat_max"] and lon_ok:
            return name
    return "QTP_other"


def restored_name(info: zipfile.ZipInfo) -> str:
    """zip 항목 이름 복원. UTF-8 표시가 있으면 그대로, 없으면 cp437 → utf-8."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_decimals(x: float) -> int:
    s = repr(float(x))
    if "e" in s or "E" in s:
        s = format(Decimal(s), "f")
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def fmt_num(x) -> str:
    """원자료 값을 반올림 없이 적는다(float 의 최단 표현)."""
    return repr(float(x))


def alt_cm_of(alt_m) -> Decimal:
    return Decimal(repr(float(alt_m))) * Decimal(100)


def dec_str(d: Decimal) -> str:
    s = format(d.normalize(), "f")
    return s


def cell_key(lat: float, lon: float):
    ky = math.floor(lat / CELL)
    phi = math.radians((ky + 0.5) * CELL)
    kx = math.floor(lon * math.cos(phi) / CELL)
    return ky, kx


def block_of(lat: float, lon: float) -> int:
    return int(math.floor(lat / 0.5) * 100000 + math.floor(lon / 0.5))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    EXTRACT.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 원자료 확인 ----------
    if not ZIP_PATH.exists():
        if LOCAL_COPY.exists():
            shutil.copy2(LOCAL_COPY, ZIP_PATH)
        else:
            print(f"원자료가 없다: {ZIP_PATH}\n내려받기: curl -L -o {ZIP_PATH} '{FILE_URL}'", file=sys.stderr)
            return 2
    size = ZIP_PATH.stat().st_size
    md5 = file_hash(ZIP_PATH, "md5")
    sha = file_hash(ZIP_PATH, "sha256")
    if md5 != ZENODO_MD5 or size != ZENODO_SIZE:
        print(f"zip 의 크기 또는 md5 가 Zenodo 기록과 다르다: size={size}, md5={md5}", file=sys.stderr)
        return 3

    # ---------- 2. zip 에서 대상 파일 읽기 ----------
    z = zipfile.ZipFile(ZIP_PATH)
    members = {restored_name(i): i for i in z.infolist()}
    if TARGET not in members:
        print(f"zip 안에 대상 파일이 없다: {TARGET}", file=sys.stderr)
        return 4
    xlsx_bytes = z.read(members[TARGET])
    files_meta = [dict(name=ZIP_NAME, size=size, sha256=sha, md5=md5, url=FILE_URL)]
    target_out = EXTRACT / Path(TARGET).name
    target_out.write_bytes(xlsx_bytes)
    files_meta.append(dict(name=f"extracted/{target_out.name}", size=len(xlsx_bytes),
                           sha256=hashlib.sha256(xlsx_bytes).hexdigest(), zip_member=TARGET))
    for member, out_name in EXTRA.items():
        if member in members:
            b = z.read(members[member])
            (EXTRACT / out_name).write_bytes(b)
            files_meta.append(dict(name=f"extracted/{out_name}", size=len(b),
                                   sha256=hashlib.sha256(b).hexdigest(), zip_member=member))

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), data_only=True)
    ws = wb.worksheets[0]
    sheet_name = ws.title
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c) if c is not None else "" for c in rows[0]]
    if header != HEADER:
        print(f"머리글이 예상과 다르다: {header}", file=sys.stderr)
        return 5
    data = [r for r in rows[1:] if any(c is not None for c in r)]
    n_raw = len(data)

    # 저자 학습 표(342점)에 같은 식별자·연도가 있는지 대조한다(참고 정보).
    train_keys = set()
    if (EXTRACT / "all_samples_342.csv").exists():
        with open(EXTRACT / "all_samples_342.csv", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                train_keys.add((str(r["SiteID"]).strip(), str(r["Year"]).strip()))

    # ---------- 3. 행 분류 ----------
    recs = []
    excluded = Counter()
    unexpected = []
    for i, r in enumerate(data, start=2):          # 시트의 행 번호(머리글이 1행)
        sid, year, dtype, lon, lat, alt_m, gv, vwc = r
        sid_s = str(sid).strip()
        rec = dict(sheet_row=i, orig_id=sid_s, year=year, dtype=dtype, lon=lon, lat=lat, alt_m=alt_m,
                   gpr_v=gv, vwc=vwc, used=0, exclude_reason="")
        censored = isinstance(alt_m, str) and (">" in alt_m or "＞" in alt_m)
        if alt_m is None or (isinstance(alt_m, str) and alt_m.strip() == ""):
            rec["exclude_reason"] = "pit_no_alt" if dtype == PIT else "gpr_alt_missing" if dtype == "GPR" \
                else "other_alt_missing"
        elif dtype not in METHOD_MAP:
            rec["exclude_reason"] = "method_not_in_rule"
        elif censored or not isinstance(alt_m, (int, float)):
            rec["exclude_reason"] = "alt_not_numeric"
            unexpected.append(rec)
        elif not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)) or not isinstance(year, int):
            rec["exclude_reason"] = "coord_or_year_missing"
            unexpected.append(rec)
        else:
            rec["used"] = 1
        if rec["exclude_reason"]:
            excluded[rec["exclude_reason"]] += 1
        recs.append(rec)

    used = [r for r in recs if r["used"] == 1]

    # ---------- 4. 중복과 좌표 점검 ----------
    key_count = Counter((r["orig_id"], r["year"]) for r in used)
    dup_keys = {k for k, c in key_count.items() if c > 1}
    seen_key = Counter()
    for r in used:
        k = (r["orig_id"], r["year"])
        if k in dup_keys:
            seen_key[k] += 1
            r["site_id"] = f"{r['orig_id']}__{seen_key[k]}"
        else:
            r["site_id"] = r["orig_id"]

    # 같은 연도에 좌표와 값이 모두 같은 기록(식별자만 다르다)
    exact = defaultdict(list)
    for r in used:
        exact[(r["year"], repr(float(r["lat"])), repr(float(r["lon"])), repr(float(r["alt_m"])))].append(r)
    exact_dups = [g for g in exact.values() if len(g) > 1]
    for g in exact_dups:
        first = g[0]
        for r in g[1:]:
            r["dup_of"] = f"{SRC_ID}:{first['site_id']}"
            r["dup_note"] = f"같은 연도에 좌표와 값이 {first['site_id']} 와 같다"

    # 같은 식별자의 연도 간 좌표 차이
    by_id = defaultdict(list)
    for r in used:
        by_id[r["orig_id"]].append(r)
    shifts = {}
    for sid, g in by_id.items():
        if len(g) > 1:
            la = [float(x["lat"]) for x in g]
            lo = [float(x["lon"]) for x in g]
            shifts[sid] = dict(n_years=len(g), years=[int(x["year"]) for x in g],
                               dlat=max(la) - min(la), dlon=max(lo) - min(lo))

    # ---------- 5. 표준 점 자료 ----------
    schema_cols = [c["name"] for c in json.load(open(SCHEMA, encoding="utf-8"))["columns"]]
    out_rows = []
    for r in used:
        lat, lon = float(r["lat"]), float(r["lon"])
        alt = alt_cm_of(r["alt_m"])
        flags = []
        if not (Decimal(0) < alt <= Decimal(600)):
            flags.append("range")
        if not (TIBET_BOX["lat_min"] <= lat <= TIBET_BOX["lat_max"]
                and TIBET_BOX["lon_min"] <= lon <= TIBET_BOX["lon_max"]):
            flags.append("coord")
        if int(r["year"]) < 1990:
            flags.append("year_pre1990")
        d = max(n_decimals(lat), n_decimals(lon))
        prec = format(Decimal(1).scaleb(-min(d, 5)), "f")     # 0.00001 처럼 고정 소수점으로 적는다
        notes = [f"orig_type={r['dtype']}"]
        if r["gpr_v"] is not None:
            notes.append(f"gpr_v_m_per_ns={fmt_num(r['gpr_v'])}")
        if r["vwc"] is not None:
            notes.append(f"vwc_active_layer={fmt_num(r['vwc'])}")
        notes.append("in_author_training_table=" + ("1" if (r["orig_id"], str(r["year"])) in train_keys else "0"))
        if "废" in r["orig_id"]:
            notes.append("식별자에 '废' 접미가 있다(뜻은 확인하지 못했다)")
        if "副" in r["orig_id"]:
            notes.append("식별자에 '副' 접미가 있다(뜻은 확인하지 못했다)")
        sh = shifts.get(r["orig_id"])
        if sh and max(sh["dlat"], sh["dlon"]) > SHIFT_NOTE_DEG:
            notes.append(f"같은 식별자의 연도 간 좌표 차이 dlat={sh['dlat']:.5f} dlon={sh['dlon']:.5f}")
        if r.get("dup_note"):
            notes.append(r["dup_note"])
        out_rows.append(dict(
            src_id=SRC_ID, site_id=r["site_id"], site_name=r["site_id"],
            lat=fmt_num(lat), lon=fmt_num(lon), year=int(r["year"]), month="",
            alt_cm=dec_str(alt), method=METHOD_MAP[r["dtype"]], label_def="direct_eos", n_obs=1,
            country="China", macro="Tibet", citation=CITATION, license=LICENSE,
            subunit=subunit_of(lat, lon), date="", eos_basis="dataset_statement", value_kind="single_visit",
            year_min=int(r["year"]), year_max=int(r["year"]), alt_sd_cm="", coord_prec_deg=prec,
            disturbed=0, disturb_type="", right_censored=0, orig_source=ORIG_SOURCE,
            dup_of=r.get("dup_of", ""), qc_flag=";".join(flags) if flags else "ok", notes="; ".join(notes),
        ))
    assert list(out_rows[0].keys()) == schema_cols, "열 순서가 형식 정의와 다르다"
    assert len({(o["site_id"], o["year"]) for o in out_rows}) == len(out_rows), "(site_id, year) 가 유일하지 않다"

    with open(POINTS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=schema_cols, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)

    with open(RAW_DUMP, "w", encoding="utf-8", newline="") as f:
        cols = ["sheet_row", "orig_id", "year", "dtype", "lon", "lat", "alt_m", "gpr_v", "vwc", "used",
                "exclude_reason"]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in recs:
            o = dict(r)
            for k in ("lon", "lat", "alt_m", "gpr_v", "vwc"):
                o[k] = "" if o[k] is None else (fmt_num(o[k]) if isinstance(o[k], (int, float)) else str(o[k]))
            w.writerow(o)

    # ---------- 6. 품질 요약 ----------
    df = pd.DataFrame(out_rows)
    df["lat_f"] = df.lat.astype(float)
    df["lon_f"] = df.lon.astype(float)
    df["alt_f"] = df.alt_cm.astype(float)

    def dist(s: pd.Series) -> dict:
        return dict(n=int(len(s)), min=float(s.min()), p25=float(s.quantile(0.25)), median=float(s.median()),
                    mean=round(float(s.mean()), 2), p75=float(s.quantile(0.75)), max=float(s.max()))

    n_loc4 = int(len({(round(a, 4), round(b, 4)) for a, b in zip(df.lat_f, df.lon_f)}))

    # 예비 셀 추정(6B.3 의 셀 규칙). 확정 값은 v4 조립 단계에서 정한다.
    main_df = df[(df.qc_flag == "ok") & (df.dup_of == "")].copy()
    main_df["cell"] = [cell_key(a, b) for a, b in zip(main_df.lat_f, main_df.lon_f)]
    loc = main_df.groupby(["cell", "site_id"], as_index=False).agg(lat=("lat_f", "mean"), lon=("lon_f", "mean"),
                                                                     alt=("alt_f", "mean"), n_years=("year", "nunique"))
    cells = loc.groupby("cell", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"), alt_cm=("alt", "mean"),
                                                     n_loc=("site_id", "size"))
    cells["block"] = [block_of(a, b) for a, b in zip(cells.lat, cells.lon)]
    cells["subunit"] = [subunit_of(a, b) for a, b in zip(cells.lat, cells.lon)]

    v3 = pd.read_csv(V3, usecols=["lat", "lon", "region", "source_id", "block"], low_memory=False)
    box = v3[v3.lat.between(TIBET_BOX["lat_min"] - 1, TIBET_BOX["lat_max"] + 1)
             & v3.lon.between(TIBET_BOX["lon_min"] - 1, TIBET_BOX["lon_max"] + 1)]

    def near(sub: pd.DataFrame) -> np.ndarray:
        if len(sub) == 0:
            return np.zeros(len(cells), dtype=bool)
        dl = np.abs(cells.lat.values[:, None] - sub.lat.values[None, :])
        dn = np.abs(cells.lon.values[:, None] - sub.lon.values[None, :])
        return (np.maximum(dl, dn) <= DUP_DEG).any(axis=1)

    cells["dup_v3_direct"] = near(box[box.source_id == "F4_direct"])
    cells["near_v3_other"] = near(box[box.source_id != "F4_direct"])
    new_cells = cells[~cells.dup_v3_direct]
    cell_est = dict(
        rule="6B.3 의 약 1 km 셀(ky = floor(lat/0.009), kx = floor(lon·cos φ/0.009)). 위치는 site_id 단위. "
             "qc_flag 가 ok 이고 dup_of 가 빈 행만 썼다",
        status="예비 추정이다. 확정 값은 v4 조립 단계에서 정한다",
        n_cells=int(len(cells)), n_blocks=int(cells.block.nunique()),
        n_cells_dup_v3_F4_direct=int(cells.dup_v3_direct.sum()),
        n_cells_near_v3_other_source=int(cells.near_v3_other.sum()),
        n_new_cells=int(len(new_cells)), n_new_blocks=int(new_cells.block.nunique()),
        new_cells_by_subunit={k: int(v) for k, v in new_cells.subunit.value_counts().items()},
        new_blocks_by_subunit={k: int(v) for k, v in new_cells.groupby("subunit").block.nunique().items()},
        cell_alt_cm=dist(new_cells.alt_cm) if len(new_cells) else None,
        v3_rows_in_tibet_box=[dict(region=a, source_id=b, n=int(c)) for (a, b), c in
                              box.groupby(["region", "source_id"]).size().items()],
        exploration_estimate="새 0.01° 셀 132개, 0.5° 블록 34개, v3 중복 2셀(탐색 단계, 격자 정의가 다르다)",
    )

    rep_rows = []
    for sid, g in df.groupby("site_id"):
        if g.year.nunique() > 1:
            rep_rows.append(dict(site_id=sid, years=[int(y) for y in g.year], alt_cm=[float(a) for a in g.alt_f],
                                 range_cm=float(g.alt_f.max() - g.alt_f.min())))
    rep_range = pd.Series([r["range_cm"] for r in rep_rows], dtype=float)
    repeat_summary = dict(
        n_ids=int(len(rep_rows)),
        range_cm=dict(min=float(rep_range.min()), median=float(rep_range.median()),
                      mean=round(float(rep_range.mean()), 2), max=float(rep_range.max())) if len(rep_rows) else None,
        note="같은 식별자를 2–3개 연도에 측정한 값의 최대와 최소의 차이다. 연도 간 변화와 측정 오차, 측선 위치 차이가 섞여 있다",
        rows=sorted(rep_rows, key=lambda r: -r["range_cm"]),
    )
    near_rows = []
    for _, c in cells[cells.near_v3_other].iterrows():
        dl = np.abs(box.lat.values - c.lat)
        dn = np.abs(box.lon.values - c.lon)
        j = int(np.argmin(np.maximum(dl, dn)))
        near_rows.append(dict(cell_lat=round(float(c.lat), 6), cell_lon=round(float(c.lon), 6),
                              site_ids=sorted(loc[loc.cell == c.cell].site_id.tolist()),
                              v3_region=str(box.region.values[j]), v3_source_id=str(box.source_id.values[j]),
                              v3_lat=float(box.lat.values[j]), v3_lon=float(box.lon.values[j]),
                              cheb_deg=round(float(max(dl[j], dn[j])), 5)))
    cell_est["cells_near_v3_other_source"] = near_rows
    not_in_train = sorted(df[df.notes.str.contains("in_author_training_table=0")].site_id.tolist())

    big_shift = {k: dict(n_years=v["n_years"], years=v["years"], dlat=round(v["dlat"], 6), dlon=round(v["dlon"], 6))
                 for k, v in shifts.items() if max(v["dlat"], v["dlon"]) > SHIFT_NOTE_DEG}

    # 탐갱 식별자의 날짜(YYYYMMDD-n 또는 YYMMDDnnn)를 월별로 센다(개정 13, 조사 월 근거 기록)
    pit_dates = []
    for r in recs:
        if r["dtype"] != PIT:
            continue
        sid_s = str(r["orig_id"])
        m8 = re.match(r"^(20\d{2})(\d{2})(\d{2})-", sid_s)
        m9 = re.match(r"^(\d{2})(\d{2})(\d{2})\d{3}$", sid_s)
        if m8:
            pit_dates.append((int(m8.group(1)), int(m8.group(2)), int(m8.group(3))))
        elif m9:
            pit_dates.append((2000 + int(m9.group(1)), int(m9.group(2)), int(m9.group(3))))
    n_oct = sum(1 for y, mth, d in pit_dates if mth >= 10)
    pit_month_note = (f"탐갱 식별자 가운데 날짜로 읽히는 것은 {len(pit_dates)}개이고 그 가운데 {n_oct}개가 10월이다"
                      f"({', '.join(f'{y}-{mth:02d}-{d:02d}' for y, mth, d in sorted(pit_dates) if mth >= 10)}). "
                      "조사 기간이 10월을 포함함을 뒷받침한다. GPR 점의 조사일은 여전히 없다")
    meta = dict(
        src_id=SRC_ID,
        name="Du 등 2026, 청장고원 활동층 수분 자료 코드 묶음의 GPR·인력 시추 ALT 조사점(Zenodo 21999366)",
        url=URL, doi=DOI, related_doi=[NCDC_DOI], accessed=ACCESSED,
        zenodo=dict(version="2.0.0", publication_date="2026-08-19", resource_type="software",
                    license_id="mit-license", concept_doi="10.5281/zenodo.21999365",
                    file_checksum=f"md5:{ZENODO_MD5}", checksum_match=True),
        files=files_meta,
        license=LICENSE,
        license_detail="Zenodo 기록의 약관 표기는 MIT 이고 zip 안의 LICENSE 파일도 MIT 다. MIT 는 소프트웨어 약관이며 "
                       "기록 단위로 붙어 있다. 조사점 표는 코드 묶음에 동봉된 파일이다. 현장 자료 정본(NCDC)의 약관은 "
                       "확인하지 못했다",
        citation=CITATION,
        parser="scripts/1_data_prep/parse_ext_qtp_du_gpr.py",
        parser_git_commit=f"{git_commit()} (HEAD, 파서는 아직 커밋되지 않았다)",
        schema_version="1.0",
        sheet=dict(file=TARGET, sheet_name=sheet_name, header=HEADER, n_sheets=len(wb.worksheets)),
        n_rows_raw=n_raw,
        n_rows_points=len(out_rows),
        n_sites=int(df.site_id.nunique()),
        n_unique_locations_4dec=n_loc4,
        n_raw_by_type={k: int(v) for k, v in Counter(r["dtype"] for r in recs).items()},
        n_by_label_def={k: int(v) for k, v in df.label_def.value_counts().items()},
        n_by_method={k: int(v) for k, v in df.method.value_counts().items()},
        n_by_eos_basis={k: int(v) for k, v in df.eos_basis.value_counts().items()},
        n_by_macro={k: int(v) for k, v in df.macro.value_counts().items()},
        n_by_subunit={k: int(v) for k, v in df.subunit.value_counts().items()},
        n_by_year={str(k): int(v) for k, v in df.year.value_counts().sort_index().items()},
        n_by_qc_flag={k: int(v) for k, v in df.qc_flag.value_counts().items()},
        n_right_censored=int((df.right_censored == 1).sum()),
        n_with_dup_of=int((df.dup_of != "").sum()),
        year_range=[int(df.year.min()), int(df.year.max())],
        alt_cm=dist(df.alt_f),
        alt_cm_by_method={k: dist(g.alt_f) for k, g in df.groupby("method")},
        alt_cm_by_subunit={k: dist(g.alt_f) for k, g in df.groupby("subunit")},
        lat_range=[float(df.lat_f.min()), float(df.lat_f.max())],
        lon_range=[float(df.lon_f.min()), float(df.lon_f.max())],
        n_excluded_by_reason={k: int(v) for k, v in excluded.items()},
        exclusion_reason_desc=dict(pit_no_alt="탐갱(探坑) 행이고 ALT 값이 없다",
                                   gpr_alt_missing="GPR 행이지만 ALT 값이 비어 있다(식별자 AEJ01–AEJ05, 2015년)"),
        duplicates=dict(
            id_year_duplicates=[list(map(str, k)) for k in sorted(dup_keys)],
            exact_same_year_coord_value=[[x["site_id"] for x in g] for g in exact_dups],
            handling="좌표와 값이 같은 기록은 지우지 않고 뒤의 행에 dup_of 를 적었다",
            repeated_ids_across_years=int(len(shifts)),
            repeated_ids_with_coord_shift_gt_0p001deg=big_shift,
        ),
        coordinate_conversion="변환하지 않았다. 원자료가 WGS84 로 보이는 십진 도(Lon, Lat 열)다. 측지계 표기는 원자료에 없다. "
                              "동경이므로 부호는 양수다. 값은 float 의 최단 표현으로 적었다(반올림하지 않았다)",
        coord_prec_rule="coord_prec_deg = 10^-min(d, 5), d 는 위도와 경도의 소수 자릿수 가운데 큰 값. 소수 6자리 이상은 "
                        "도분 환산에서 생긴 자릿수로 보고 0.00001 로 둔다. 측위 정확도는 확인하지 못했다",
        subunit_rule=[dict(subunit=n, label=k, box=b) for n, k, b in SUBUNIT_RULE]
                     + [dict(subunit="QTP_other", label="그 밖", box="위 네 상자 밖의 Tibet 범위")],
        macro_rule=dict(macro="Tibet", box=TIBET_BOX),
        unit_conversion="alt_cm = ALT/m × 100 (십진 연산)",
        missing_value_code="빈 셀(None). 숫자 결측 부호(-999 등)는 없다",
        eos_evidence=[
            f"시트 이름: {sheet_name}",
            "동봉 원고 초안(03_final_pipeline/Tables/Paper14_ESSD_revision_text_draft.md)의 초록: "
            "'342 field-surveyed observations collected during 2009–2024 peak-thaw seasons (September–October)'",
            "탐갱 행의 식별자가 날짜 형식이다(2009-09-14 부터 2009-10-10, 2010-09-02 부터 2010-10-15). "
            "이 행은 ALT 가 없어 점 자료에 넣지 않았다. GPR 행의 날짜 근거는 아니다",
            pit_month_note,
        ],
        october_survey_limit=("개정 13(검증 지적 반영): 조사 시기 서술(9–10월)과 탐갱 식별자의 날짜는 조사 기간이 10월을 포함함을 "
                              "뒷받침한다. 6B.3 은 단일 방문의 10월 이후 값을 direct_dated 로 두지만 GPR 점에는 날짜가 없어 이 규칙을 "
                              "점 단위로 적용할 수 없다. Tibet 대상 셀은 모두 eos_basis = dataset_statement 이고 L41 (a)와 (c)에서 "
                              "대상 셀이 0개라 민감도를 낼 수 없다. NCDC db7703 정본에서 점별 날짜를 확보하면 10월 점을 direct_dated 로 "
                              "내린다. 그 전까지 PE2 결론에 이 한계를 병기한다"),
        repeat_measurement=repeat_summary,
        rows_not_in_author_training_table=dict(
            site_ids=not_in_train,
            note="저자 학습 표(03_final_pipeline/Results/all_samples_342.csv, 활동층 수분 학습용)에 같은 식별자·연도가 없는 행이다. "
                 "ALT 값은 조사점 표에 있다. 빠진 이유는 확인하지 못했다"),
        preliminary_cell_estimate=cell_est,
        unverified_items=[
            "점별 관측 날짜와 월(원자료에 없다). 계절 말 근거는 자료 설명뿐이다",
            "조사 시기 9–10월은 v3 의 계절 말 창(8–9월)과 다르다. 10월 조사 여부를 점 단위로 가릴 수 없다",
            "현장 자료 정본(NCDC doi:10.12072/ncdc.permafrost.db7703.2026)의 내용과 약관. 이 서버에서 www.ncdc.ac.cn 접속이 되지 않는다",
            "원 논문(ESSD, 심사 중)의 DOI 와 방법 절. GPR 전파 속도의 산정 방식(공통 중간점 측정 여부)",
            "GPR 반사면이 영구동토 상한인지에 대한 점별 검증 여부",
            "행마다의 원관측 수(GPR 측선의 트레이스 수). n_obs 는 1 로 두었다",
            "한 해에 한 번 방문했는지 여부. value_kind 는 single_visit 로 두었다",
            "교란 지점 여부. 원자료에 표기가 없어 disturbed 는 0 으로 두었다",
            "식별자 접미 '废', '副' 의 뜻(BLH02_废, WDL02副, 2018년)",
            "R15 와 R16(2011년)은 좌표와 값이 같다. 중복 입력인지 좌표 오기인지 확인하지 못했다",
            "측지계(WGS84 로 가정)",
            "Zenodo 설명문의 인용 예시는 버전 1.0.0 이고 기록 메타의 버전은 2.0.0 이다",
            "T31, T47(2020년)이 저자 학습 표에 없는 이유",
        ],
        unexpected_rows=[dict(sheet_row=u["sheet_row"], orig_id=u["orig_id"], reason=u["exclude_reason"])
                         for u in unexpected],
    )
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"원자료 {n_raw}행, 점 자료 {len(out_rows)}행, 지점 {meta['n_sites']}곳")
    print("방법별:", meta["n_by_method"], "하위 단위별:", meta["n_by_subunit"])
    print("제외:", meta["n_excluded_by_reason"], "qc:", meta["n_by_qc_flag"])
    print("연도:", meta["year_range"], "ALT(cm):", meta["alt_cm"])
    print("예비 셀:", {k: cell_est[k] for k in ("n_cells", "n_blocks", "n_cells_dup_v3_F4_direct",
                                             "n_cells_near_v3_other_source", "n_new_cells", "n_new_blocks")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
