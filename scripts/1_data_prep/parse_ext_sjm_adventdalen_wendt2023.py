#!/usr/bin/env python3
"""ext_labels 파서: sjm_adventdalen_wendt2023 (Wendt 2024, Zenodo 11187360). XF_new_regions(계획 2.6 순서 2).

자료
- 스발바르 Adventdalen 코어 12지점의 지중 얼음 함량(2023년 4월 코어)과 2023년 9월 현장 융해 깊이, InSAR 변위.
- 원자료: data/raw/sjm_adventdalen_wendt2023/Adventdalen_data.xlsx(Overview 시트). 출처와 해시는 data/raw/sjm_adventdalen_wendt2023/SOURCE.md.

라벨 규칙(값 계산 전에 정했다: docs/research/2026-10-04/impl_notes/x_new_regions.md 9절, LGD 6B.3)
- 라벨 열: 'Thaw depth in-situ (cm)'(Thaw depth date 의 현장 탐침 측정, open_regions.md 4.1 이 인용한 TC 20, 1179, 2026 본문).
  'ALT (cm)' 는 현장 융해 깊이에 InSAR 지표 침하 보정을 더한 값으로 보여 6B.3 '보정값 금지'와 v3·CALM 의 보정하지 않은 탐침 규약에 맞지 않아
  쓰지 않는다. 'Surface subsidence correction to thaw depth (cm)' 도 쓰지 않는다.
- 2023년 9월 단일 방문: label_def = direct_eos, eos_basis = record_date, value_kind = single_visit(6B.3 record_date). 관측 월이 8–9월이
  아니면 direct_dated 다.
- 2023년은 이례적 고온 해다(open_regions.md 4.1). notes 에 연도 편차 표지(year_anomaly:2023_warm)를 둔다. qc_flag 는 바꾸지 않는다.
- 좌표: UTM X, UTM Y 를 WGS84 UTM 33N(EPSG:32633)으로 가정해 십진 도로 바꾼다([가정], 측지계·구역이 파일에 없다). 바꾼 좌표가
  Adventdalen 범위(78.0–78.4°N, 15.3–16.6°E) 밖이면 qc_flag coord.
- '>' 표기 값은 right_censored = 1(하한). 빈 칸은 행을 만들지 않는다. 그 밖의 문자 칸은 값으로 쓰지 않고 메타에 센다.
- 0 < alt_cm ≤ 600 범위 규칙과 1990년 이후 규칙은 _schema.json 과 같다.
- macro = NAtlantic, country = Norway, subunit = Svalbard(LGD 6B.4 지리 정의, x_new_regions.geo_macro 와 같다).

열람 규칙(계획 0.3, 2.6): 화면과 메타에 라벨 통계(평균, 분산, 최솟값·최댓값, 계수 비)를 쓰지 않는다. 행 수, 지점 수, label_def·qc_flag 별
행 수만 쓴다.

산출(계획 1절 산출 경로)
- data/processed/xbatch/XF_new_regions/ext_labels/sjm_adventdalen_wendt2023_points.csv   표준 점 자료(_schema.json 1.0)
- data/processed/xbatch/XF_new_regions/ext_labels/sjm_adventdalen_wendt2023_meta.json    메타(라벨 통계 없음)

실행(저장소 최상위, 스레드 1개): OMP_NUM_THREADS=1 nice -n 10 python3 scripts/1_data_prep/parse_ext_sjm_adventdalen_wendt2023.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ID = "sjm_adventdalen_wendt2023"
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
RAW_FILE = RAW_DIR / "Adventdalen_data.xlsx"
RAW_SHA256 = "a9876f8280461c7b675e3a231ac4cd1d9d466a24d492598b59b17f03fdb1df49"     # SOURCE.md
SCHEMA = ROOT / "data" / "processed" / "ext_labels" / "_schema.json"              # 형식 정의(읽기만)
OUT_DIR = ROOT / "data" / "processed" / "xbatch" / "XF_new_regions" / "ext_labels"
POINTS = OUT_DIR / f"{SRC_ID}_points.csv"
META = OUT_DIR / f"{SRC_ID}_meta.json"

DOI = "10.5281/zenodo.11187360"
URL = "https://doi.org/10.5281/zenodo.11187360"
ACCESSED = "2026-10-04"
LICENSE = "CC-BY-4.0"
CITATION = "Wendt, L. (2024). Ground ice contents and InSAR displacements from Adventdalen, Svalbard. Zenodo. https://doi.org/10.5281/zenodo.11187360"
SHEET = "Overview"
COL_CORE, COL_X, COL_Y = "Core", "UTM X", "UTM Y"
COL_LABEL, COL_DATE = "Thaw depth in-situ (cm)", "Thaw depth date"
COL_NOT_USED = ("ALT (cm)", "Surface subsidence correction to thaw depth (cm)")
COL_DEPOSIT, COL_GRAIN = "Sediment deposit type", "Main grain size active layer"
EPSG_UTM = 32633                                                             # WGS84 UTM 33N [가정]
BBOX = dict(lat=(78.0, 78.4), lon=(15.3, 16.6))                              # Adventdalen 범위(좌표 점검용)
EOS_MONTHS = (8, 9)
YEAR_MIN = 1990
RANGE = (0.0, 600.0)
GT = re.compile(r"^\s*>\s*(-?\d+(?:\.\d+)?)\s*$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cell_value(v):
    """칸 → (값 또는 None, 절단 여부, 종류: num, gt, blank, text)."""
    if v is None:
        return None, False, "blank"
    if isinstance(v, bool):
        return None, False, "text"
    if isinstance(v, (int, float)):
        return abs(float(v)), False, "num"
    s = str(v).strip()
    if not s:
        return None, False, "blank"
    m = GT.match(s)
    if m:
        return abs(float(m.group(1))), True, "gt"
    try:
        return abs(float(s)), False, "num"
    except ValueError:
        return None, False, "text"


def main():
    if not RAW_FILE.exists():
        sys.exit(f"원자료가 없다: {RAW_FILE}")
    got = sha256(RAW_FILE)
    if got != RAW_SHA256:
        sys.exit(f"원자료 sha256 이 SOURCE.md 기록과 다르다: {got}")
    import openpyxl
    from pyproj import Transformer
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    columns = [c["name"] for c in schema["columns"]]
    wb = openpyxl.load_workbook(RAW_FILE, read_only=True, data_only=True)
    rows = list(wb[SHEET].iter_rows(min_row=1, values_only=True))
    hdr = [str(v).strip() if v is not None else "" for v in rows[0]]
    for c in (COL_CORE, COL_X, COL_Y, COL_LABEL, COL_DATE) + COL_NOT_USED:
        assert c in hdr, (c, hdr)
    ix = {c: hdr.index(c) for c in hdr if c}
    tr = Transformer.from_crs(EPSG_UTM, 4326, always_xy=True)
    kinds = Counter()
    out, n_rows_raw = [], 0
    for r in rows[1:]:
        core = r[ix[COL_CORE]]
        if core is None or not str(core).strip():
            continue
        n_rows_raw += 1
        v, cens, kind = cell_value(r[ix[COL_LABEL]])
        kinds[kind] += 1
        if v is None:
            continue
        d = r[ix[COL_DATE]]
        if not isinstance(d, dt.datetime):
            kinds["date_not_datetime"] += 1
            continue
        lon, lat = tr.transform(float(r[ix[COL_X]]), float(r[ix[COL_Y]]))
        flags = []
        if not (BBOX["lat"][0] <= lat <= BBOX["lat"][1] and BBOX["lon"][0] <= lon <= BBOX["lon"][1]):
            flags.append("coord")
        if d.year < YEAR_MIN:
            flags.append("year_pre1990")
        if not (RANGE[0] < v <= RANGE[1]):
            flags.append("range")
        ld, eos = ("direct_eos", "record_date") if d.month in EOS_MONTHS else ("direct_dated", "none")
        dep, grain = r[ix[COL_DEPOSIT]], r[ix[COL_GRAIN]]
        notes = [f"코어 {core}", f"라벨 열 '{COL_LABEL}'(현장 탐침). 'ALT (cm)'(침하 보정 포함)는 쓰지 않는다",
                 "year_anomaly:2023_warm(2023년은 이례적 고온 해, open_regions.md 4.1)", f"퇴적 유형 {dep}, 활동층 주 입도 {grain}",
                 "좌표는 UTM X·Y 를 WGS84 UTM 33N 으로 가정해 바꿨다([가정])", "같은 지점에서 2023년 4월 코어를 채취했다(교란 표시는 하지 않았다)"]
        out.append(dict(src_id=SRC_ID, site_id=str(core).strip(), site_name=f"Adventdalen core {str(core).strip()}", lat=f"{lat:.6f}", lon=f"{lon:.6f}",
                        year=d.year, month=d.month, alt_cm=round(float(v), 4), method="probe", label_def=ld, n_obs=1, country="Norway",
                        macro="NAtlantic", citation=CITATION, license=LICENSE, subunit="Svalbard", date=d.date().isoformat(), eos_basis=eos,
                        value_kind="single_visit", year_min="", year_max="", alt_sd_cm="", coord_prec_deg=0.00001, disturbed=0, disturb_type="",
                        right_censored=int(cens), orig_source="", dup_of="", qc_flag=";".join(flags) if flags else "ok", notes=". ".join(notes)))
    assert all(set(r) == set(columns) for r in out), "열 구성이 형식 정의와 다르다"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gi = OUT_DIR / ".gitignore"
    if not gi.exists():
        gi.write_text("# XF 새 자료원 점 자료: 적격 판정 기록 전 라벨 값이 든다. 커밋하지 않는다(계획 0.3, 2.6).\n*\n!.gitignore\n")
    tmp = POINTS.with_name(POINTS.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        wr.writeheader()
        wr.writerows(out)
    os.replace(tmp, POINTS)
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                     # noqa: BLE001
        commit = ""
    meta = dict(src_id=SRC_ID, name="Wendt 2024, Adventdalen 코어 12지점 2023년 현장 융해 깊이(Zenodo 11187360)", url=URL, doi=DOI, accessed=ACCESSED,
                files=[dict(name=RAW_FILE.name, path=str(RAW_FILE.relative_to(ROOT)), size_bytes=RAW_FILE.stat().st_size, sha256=got)],
                license=LICENSE, license_statement="Zenodo API metadata.license.id = cc-by-4.0(2026-10-04, SOURCE.md)", citation=CITATION,
                related_paper="The Cryosphere 20, 1179, 2026(open_regions.md 4.1)",
                parser=str(Path(__file__).resolve().relative_to(ROOT)), parser_sha256=sha256(Path(__file__).resolve()), parser_git_commit=commit,
                parser_git_note="파서는 이 커밋의 작업 트리에서 새로 만든 파일이고 아직 커밋하지 않았다", schema_version=schema["version"],
                created=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds"),
                label_column_decision="docs/research/2026-10-04/impl_notes/x_new_regions.md 9절(값 계산 전 기록)", label_column=COL_LABEL,
                columns_not_used=list(COL_NOT_USED), n_rows_raw=n_rows_raw, n_rows_points=len(out),
                n_by_label_def=dict(Counter(r["label_def"] for r in out)), n_by_qc_flag=dict(Counter(r["qc_flag"] for r in out)),
                n_right_censored_rows=int(sum(r["right_censored"] for r in out)), cell_kinds=dict(kinds),
                months=dict(Counter(f"{r['year']}-{r['month']:02d}" for r in out)),
                coordinate_conversion=f"UTM X, UTM Y → WGS84 십진 도(pyproj, EPSG:{EPSG_UTM} 가정, always_xy)",
                macro_rule="LGD 6B.4: 노르웨이(스발바르) = NAtlantic(subunit Svalbard)",
                unverified_items=["좌표 측지계와 UTM 구역(파일에 없음, 33N 가정)", "지점별 측정 반복 수(n_obs 1 로 둠)", "탐침 길이",
                                  "코어 채취(2023년 4월)가 같은 지점의 9월 융해 깊이에 준 영향"],
                label_statistics="쓰지 않는다(계획 0.3, 2.6 열람 상태)")
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[{SRC_ID}] 원자료 코어 행 {n_rows_raw} · 점 자료 행 {len(out)} · 관측 연월 {meta['months']}", flush=True)
    print(f"  label_def {meta['n_by_label_def']} · qc_flag {meta['n_by_qc_flag']} · 절단 행 {meta['n_right_censored_rows']} · 칸 종류 {dict(kinds)}", flush=True)
    print(f"  산출 {POINTS.relative_to(ROOT)}, {META.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
