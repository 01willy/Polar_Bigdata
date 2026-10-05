#!/usr/bin/env python3
"""ext_labels 파서: swe_stordalen_crill (Crill, McCalley, Abisko Field Crew 2023, Zenodo 10420396). XF_new_regions(계획 2.6 순서 1).

자료
- Stordalen 이탄습지 자동 챔버 지점의 수동 활동층 깊이(A L)와 지하수위(W D), 2003–2017(2018년 6월 1행 포함).
- 금속 막대를 더 들어가지 않을 때까지 넣어 잰 깊이(cm, 식생 표면 기준, 음수 = 표면 아래). '>' 는 활동층이 탐침보다 깊다는 표기다.
- 원자료: data/raw/swe_stordalen_crill/Active_Layer_Water_Table_03-17.xlsx(DATA 시트, 1행 지점 이름, 3행 열 이름, 4행부터 자료).
  출처와 해시는 data/raw/swe_stordalen_crill/SOURCE.md.

라벨 규칙(값 계산 전에 정했다: docs/research/2026-10-04/impl_notes/x_new_regions.md 9절, LGD 6B.3)
- 라벨 열: 지점별 'A L' 열. 음수 표기는 절댓값을 쓴다(_schema.json units).
- 직접 라벨(qc_flag ok)로 쓰는 지점: Dry(Palsa 자동 챔버 지점 1, 3, 5)와 Palsa center. Mesic(Sphagnum), Wet(Eriophorum), Tussok(9), E, F 는
  영구동토 존재가 확인되지 않아 qc_flag = pf_uncertain 으로 점 자료에 남기고 직접 라벨에서 뺀다.
- 연 값(지점 × 연도): 8월 1일 이후 값이 있는 방문이 2회 이상이면 그 최댓값(eos_basis = series_max, direct_eos), 1회이고 8–9월이면
  record_date(direct_eos), 8월 1일 이후 방문이 10월 이후뿐이거나 없으면 direct_dated(값은 각각 그 방문들의 최댓값, 그 해 마지막 방문 값).
  FireALT 파서의 개정 13 규칙과 같다. 방문마다 값이 하나라 10월 최댓값 방문의 비교 조건(기록 수, 측정 종류)은 늘 만족한다.
- 절단: 그 해 후보 방문에 '>' 값이 하나라도 있으면 right_censored = 1 이고 값은 숫자 최댓값과 '>' 하한 가운데 큰 값이다
  (ru_kwbs_makarieva 파서 (a) 규칙). '>' 뒤에 숫자가 없는 칸과 'inf'(동결층을 찾지 못함) 칸은 하한을 알 수 없으므로 그 방문을 절단 방문으로만
  센다. 그 밖의 문자 칸('<' 표기 1칸)은 값으로 쓰지 않고 메타의 cell_kinds 에 센다. 이 표기들은 직접 라벨 지점(Dry, Palsa center)에는 없다.
- 좌표: 원파일에 없다. ICOS SE-Sto 소개의 68°21′N, 19°03′E(분 단위, coord_prec_deg 0.0167)를 모든 지점에 쓴다(한 1 km 셀).
- 1990년 이후 규칙, 0 < alt_cm ≤ 600 범위 규칙은 _schema.json 과 같다(벗어나면 qc_flag 에 year_pre1990, range).
- macro = NAtlantic, country = Sweden, subunit = Scandinavia(LGD 6B.4 지리 정의, x_new_regions.geo_macro 와 같다).

열람 규칙(계획 0.3, 2.6): 화면과 메타에 라벨 통계(평균, 분산, 최솟값·최댓값, 계수 비)를 쓰지 않는다. 행 수, 지점 수, 연도 범위,
label_def·qc_flag 별 행 수만 쓴다.

산출(계획 1절 산출 경로)
- data/processed/xbatch/XF_new_regions/ext_labels/swe_stordalen_crill_points.csv   표준 점 자료(_schema.json 1.0)
- data/processed/xbatch/XF_new_regions/ext_labels/swe_stordalen_crill_meta.json    메타(라벨 통계 없음)

실행(저장소 최상위, 스레드 1개): OMP_NUM_THREADS=1 nice -n 10 python3 scripts/1_data_prep/parse_ext_swe_stordalen_crill.py
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
SRC_ID = "swe_stordalen_crill"
RAW_DIR = ROOT / "data" / "raw" / SRC_ID
RAW_FILE = RAW_DIR / "Active_Layer_Water_Table_03-17.xlsx"
RAW_SHA256 = "4be42209334dc2821d99af67f7460bcc1c403626af805ed3b4cc87dec56229c1"     # SOURCE.md
SCHEMA = ROOT / "data" / "processed" / "ext_labels" / "_schema.json"              # 형식 정의(읽기만)
OUT_DIR = ROOT / "data" / "processed" / "xbatch" / "XF_new_regions" / "ext_labels"
POINTS = OUT_DIR / f"{SRC_ID}_points.csv"
META = OUT_DIR / f"{SRC_ID}_meta.json"

DOI = "10.5281/zenodo.10420396"
URL = "https://doi.org/10.5281/zenodo.10420396"
ACCESSED = "2026-10-04"
LICENSE = "CC-BY-4.0"
CITATION = ("Crill, P., McCalley, C., Abisko Field Crew (2023). Manual active layer and and water table depth measurements from the "
            "autochamber sites at Stordalen Mire, northern Sweden (2003-2017). Zenodo. https://doi.org/10.5281/zenodo.10420396")
LAT, LON, COORD_PREC = 68.0 + 21.0 / 60.0, 19.0 + 3.0 / 60.0, 0.0167           # ICOS SE-Sto 소개(분 단위)
SHEET = "DATA"
HEADER_ROW, NAME_ROW, FIRST_DATA_ROW = 3, 1, 4
# 열 번호(0 부터): 1행 지점 이름 → 'A L' 열. 이름과 열 이름을 실행 때 다시 확인한다
SITES = {2: ("Dry", "Dry (1,3,5)", "Palsa autochamber sites 1, 3, 5", True),
         4: ("Mesic", "Mesic (2,4,6)", "Sphagnum autochamber sites 2, 4, 6", False),
         6: ("Wet", "Wet (7,8)", "Eriophorum autochamber sites 7, 8", False),
         8: ("Tussok", "Tussok (9)", "Tussock site 9", False),
         10: ("E", "E", "site E (identity not documented)", False),
         12: ("F", "F", "site F (identity not documented)", False),
         15: ("Palsa_center", "Palsa center", "palsa centre", True)}
AL_HEADERS = {"A L", "AL"}
LATE = (8, 1)                       # 8월 1일 이후(6B.3 series_max)
EOS_MONTHS = (8, 9)
YEAR_MIN = 1990
RANGE = (0.0, 600.0)
GT = re.compile(r"^\s*>\s*(-?\d+(?:\.\d+)?)?\s*$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cell_value(v):
    """칸 → (값 또는 None, 절단 여부, 종류). 종류: num, gt(하한 있음), gt_nobound, inf(동결층 없음, 하한 없는 절단), blank, text."""
    if v is None:
        return None, False, "blank"
    if isinstance(v, bool):
        return None, False, "text"
    if isinstance(v, (int, float)):
        return abs(float(v)), False, "num"
    s = str(v).strip()
    if s == "":
        return None, False, "blank"
    if s.lower() == "inf":                                                # 동결층을 찾지 못함(탐침보다 깊음과 같은 뜻으로 본다, 하한 없음)
        return None, True, "inf"
    m = GT.match(s)
    if m:
        return (abs(float(m.group(1))), True, "gt") if m.group(1) is not None else (None, True, "gt_nobound")
    return None, False, "text"


def annual(visits):
    """한 지점·연도의 방문 [(date, value, censored)] → (연 값, 날짜, label_def, eos_basis, right_censored, n_late, 메모)."""
    visits = sorted(visits, key=lambda x: x[0])
    late = [v for v in visits if (v[0].month, v[0].day) >= LATE]
    win = [v for v in late if v[0].month in EOS_MONTHS]
    note = []
    if not late:
        cand, ld, eos = [visits[-1]], "direct_dated", "none"
        note.append("8월 1일 이후 방문 없음(그 해 마지막 방문)")
    elif not win:
        cand, ld, eos = late, "direct_dated", "none"
        note.append("8월 1일 이후 방문이 10월 이후뿐")
    elif len(late) == 1:
        cand, ld, eos = late, "direct_eos", "record_date"
    else:
        cand, ld, eos = late, "direct_eos", "series_max"
    nums = [v for v in cand if v[1] is not None and not v[2]]
    bounds = [v for v in cand if v[2] and v[1] is not None]
    cens = any(v[2] for v in cand)
    pool = nums + bounds
    if not pool:
        return None
    best = max(pool, key=lambda x: (x[1], x[0]))                         # 같으면 늦은 방문
    if cens:
        note.append(f"후보 방문 {len(cand)}회 가운데 '>' {sum(v[2] for v in cand)}회(우측 절단, 값은 하한)")
    if eos == "series_max" and best[0].month not in EOS_MONTHS:
        note.append("최댓값 방문이 10월 이후(방문마다 값 하나, 같은 측정이라 유지)")
    return best[1], best[0], ld, eos, int(cens), len(late), note


def main():
    if not RAW_FILE.exists():
        sys.exit(f"원자료가 없다: {RAW_FILE}")
    got = sha256(RAW_FILE)
    if got != RAW_SHA256:
        sys.exit(f"원자료 sha256 이 SOURCE.md 기록과 다르다: {got}")
    import openpyxl
    schema = json.load(open(SCHEMA, encoding="utf-8"))
    columns = [c["name"] for c in schema["columns"]]
    wb = openpyxl.load_workbook(RAW_FILE, read_only=True, data_only=True)
    ws = wb[SHEET]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    names, header = rows[NAME_ROW - 1], rows[HEADER_ROW - 1]
    assert str(header[0]).strip() == "Date", header[0]
    for j, (sid, nm, _, _) in SITES.items():
        assert str(names[j]).strip() == nm, (j, names[j], nm)
        assert str(header[j]).strip() in AL_HEADERS, (j, header[j])
    kinds = Counter()
    by_site = {sid: {} for sid, *_ in SITES.values()}
    n_rows_raw, n_date_bad = 0, 0
    for r in rows[FIRST_DATA_ROW - 1:]:
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in r):
            continue
        n_rows_raw += 1
        d = r[0]
        if not isinstance(d, dt.datetime):
            n_date_bad += 1
            continue
        for j, (sid, *_rest) in SITES.items():
            v, cens, kind = cell_value(r[j] if j < len(r) else None)
            kinds[(sid, kind)] += 1
            if kind in ("blank", "text"):
                continue
            by_site[sid].setdefault(d.year, []).append((d.date(), v, cens))
    out = []
    for j, (sid, nm, desc, pf_ok) in SITES.items():
        for year in sorted(by_site[sid]):
            res = annual(by_site[sid][year])
            if res is None:
                continue
            val, date, ld, eos, cens, n_late, note = res
            flags = []
            if year < YEAR_MIN:
                flags.append("year_pre1990")
            if not (RANGE[0] < val <= RANGE[1]):
                flags.append("range")
            if not pf_ok:
                flags.append("pf_uncertain")
            notes = [f"지점 {nm}: {desc}", f"그 해 방문 {len(by_site[sid][year])}회(8월 1일 이후 {n_late}회)"] + note
            if not pf_ok:
                notes.append("영구동토 존재 미확인(open_regions.md 4.1, SOURCE.md). 직접 라벨에서 뺀다")
            notes.append("좌표는 ICOS SE-Sto 소개의 분 단위 위치(지점 좌표 미확인)")
            out.append(dict(src_id=SRC_ID, site_id=sid, site_name=f"Stordalen {nm}", lat=f"{LAT:.6f}", lon=f"{LON:.6f}", year=year, month=date.month,
                            alt_cm=round(float(val), 4), method="probe", label_def=ld, n_obs=1, country="Sweden", macro="NAtlantic", citation=CITATION,
                            license=LICENSE, subunit="Scandinavia", date=date.isoformat(), eos_basis=eos,
                            value_kind="annual_value" if eos == "series_max" else "single_visit", year_min="", year_max="", alt_sd_cm="",
                            coord_prec_deg=COORD_PREC, disturbed=0, disturb_type="", right_censored=cens, orig_source="", dup_of="",
                            qc_flag=";".join(flags) if flags else "ok", notes=". ".join(notes)))
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
    years = sorted({r["year"] for r in out})
    meta = dict(src_id=SRC_ID, name="Crill 등 2023, Stordalen 자동 챔버 지점 수동 활동층 깊이 2003–2017(Zenodo 10420396)", url=URL, doi=DOI,
                accessed=ACCESSED, files=[dict(name=RAW_FILE.name, path=str(RAW_FILE.relative_to(ROOT)), size_bytes=RAW_FILE.stat().st_size, sha256=got)],
                license=LICENSE, license_statement="Zenodo API metadata.license.id = cc-by-4.0(2026-10-04, SOURCE.md)", citation=CITATION,
                parser=str(Path(__file__).resolve().relative_to(ROOT)), parser_sha256=sha256(Path(__file__).resolve()), parser_git_commit=commit,
                parser_git_note="파서는 이 커밋의 작업 트리에서 새로 만든 파일이고 아직 커밋하지 않았다", schema_version=schema["version"],
                created=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds"),
                label_column_decision="docs/research/2026-10-04/impl_notes/x_new_regions.md 9절(값 계산 전 기록)",
                n_rows_raw=n_rows_raw, n_rows_points=len(out), n_rows_date_not_datetime=n_date_bad,
                n_by_label_def=dict(Counter(r["label_def"] for r in out)), n_by_method=dict(Counter(r["method"] for r in out)),
                n_by_qc_flag=dict(Counter(r["qc_flag"] for r in out)), n_by_site=dict(Counter(r["site_id"] for r in out)),
                n_right_censored_rows=int(sum(r["right_censored"] for r in out)), year_range=[years[0], years[-1]] if years else [],
                cell_kinds={f"{s}:{k}": int(v) for (s, k), v in sorted(kinds.items())},
                n_excluded_by_reason=dict(blank_or_text_cells=int(sum(v for (s, k), v in kinds.items() if k in ("blank", "text")))),
                coordinate_conversion="원파일에 좌표 없음. ICOS SE-Sto 소개의 68°21′N, 19°03′E(분 단위)를 십진 도로 바꿔 모든 지점에 썼다",
                label_rule="impl_notes/x_new_regions.md 9절: A L 절댓값, Dry·Palsa center 만 직접 라벨(그 밖 pf_uncertain), 8월 1일 이후 series_max, "
                           "'>' 이면 right_censored",
                macro_rule="LGD 6B.4: 스웨덴 = NAtlantic(subunit Scandinavia)",
                unverified_items=["지점 좌표(파일·PDF 에 없음)", "탐침 길이('>' 하한의 뜻)", "Mesic·Wet·Tussok·E·F 지점의 영구동토 존재",
                                  "Dry 열이 지점 1, 3, 5 의 평균인지 대표 지점 하나인지", "E, F 지점의 정체"],
                label_statistics="쓰지 않는다(계획 0.3, 2.6 열람 상태)")
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[{SRC_ID}] 원자료 행 {n_rows_raw} · 점 자료 행 {len(out)} · 지점 {len(set(r['site_id'] for r in out))} · 연도 {meta['year_range']}", flush=True)
    print(f"  label_def {meta['n_by_label_def']} · qc_flag {meta['n_by_qc_flag']} · 절단 행 {meta['n_right_censored_rows']}", flush=True)
    print(f"  산출 {POINTS.relative_to(ROOT)}, {META.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
