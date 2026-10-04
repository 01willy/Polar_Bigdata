"""Table 1 v3: 대상 지역, 라벨, 자료(재구성 비교판).

명세: outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 10절(열 10.2, 행 10.3, 설명문 10.4),
지침 design/journal_grade_style_guide.md 3.1(원고 표), 6.8(Table 1), 7.2(T-01–T-05), 부록 A.

원천(읽기만 한다)
- data/processed/paper_figs/table1_rows.csv  (D README E10–E19, MANIFEST 해시 대조)
- data/processed/paper_figs/fig1_source.csv  (D README E33–E37, 알래스카 비율 share_alaska)
- data/processed/paper_figs/fig1_meta.csv    (v3 직접 라벨 셀 수)
- 대조용: data/processed/lgd/{Russia_C,Tibet}/shards/*__cpu__*_unit.json, data/processed/fidelity_base_v3.csv,
  data/processed/fidelity_base_v4.csv, data/processed/fidelity_base_v4_labels.csv, paper/claims/D_data_and_design/README.md

산출(outputs/figures/paper/v3_restructure/)
- Table1_data.tex        편집 가능한 booktabs 표(원고에 넣는 정본, 지침 R-14, 그림 명세 1.7)
- Table1.pdf, Table1.png 검토용 조판본(Table1_data.tex 를 pdflatex 로 조판, PNG 600 dpi)
- Table1_source_data.csv 표 값(원값과 표시값)
- Table1_legend.md       표 설명문(150단어 이하, 그림 명세 10.4)
- Table1_values.txt      원천 대조와 점검 기록

실행: python scripts/4_visualization/paper_v3/table1.py [--build-dir DIR]
손으로 고친 셀은 0개이다. 값과 문구를 바꾸려면 이 파일을 고친다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S  # noqa: E402

ROOT = S.ROOT
SRC_ROWS = S.PAPER_FIGS / "table1_rows.csv"
SRC_SOURCE = S.PAPER_FIGS / "fig1_source.csv"
SRC_META = S.PAPER_FIGS / "fig1_meta.csv"
MANIFEST = ROOT / "paper" / "claims" / "D_data_and_design" / "tables" / "MANIFEST.csv"
D_README = ROOT / "paper" / "claims" / "D_data_and_design" / "README.md"
LGD_DIR = ROOT / "data" / "processed" / "lgd"
LGD_UNIT_DIR = {"Russia_C_LGD": LGD_DIR / "Russia_C" / "shards", "Tibet_LGD": LGD_DIR / "Tibet" / "shards"}
LGD_MACRO = {"Russia_C_LGD": "Russia_C", "Tibet_LGD": "Tibet_LGD"}       # fidelity_base_v4_labels.macro_v4
V3_TABLE = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
V4_TABLE = ROOT / "data" / "processed" / "fidelity_base_v4.csv"
V4_LABELS = ROOT / "data" / "processed" / "fidelity_base_v4_labels.csv"
BUFFER_KM = 100.0

OUT = S.OUT
STEM = "Table1"

# ---------------------------------------------------------------- 행(그림 명세 10.3)과 이름(지침 2.5, D-14)
GROUPS = [("Main regions", ["Lena", "Canada", "Russia_W", "Russia_E"]),
          ("Reference", ["Alaska"]),
          ("New regions", ["Russia_C_LGD", "Tibet_LGD"])]

# 라벨 출처의 짧은 이름(그림 명세 10.2). 원천 label_type 의 머리말로 정한다.
LABEL_SOURCE_RULES = [("Point (ALLena", "Points (ALLena)"),
                      ("Point (ABoVE", "Points (ABoVE)"),
                      ("CALM site", "CALM site means"),
                      ("~1 km cell means", "1 km cell means")]

COLUMNS = [  # (원천 데이터 열 이름, 표 머리 줄들(LaTeX), 정렬)
    ("region", ["Region"], "l"),
    ("label_source", ["Label source"], "l"),
    ("labels", ["Labels"], "r"),
    ("locations_1km", ["1 km", "locations"], "r"),
    ("blocks_0p5deg", [r"0.5\textdegree{}", "blocks"], "r"),
    ("valid_splits", ["Splits"], "r"),
    ("source_coefficient", ["Source", r"coefficient $E_0$", r"(cm per \textsurd(\textdegree{}C~d))"], "r"),
    ("alaska_share_pct", ["Alaska share", "of source", r"cells (\%)"], "r"),
]

TITLE = "Target regions, labels and data sources."
LEGEND_BODY = [
    "Labels counts label rows, each a point observation, a multi-year CALM site mean or a mean over an "
    "approximately 1 km cell.",
    "Alaska, Canada and Central Russia also contain some CALM site means.",
    "1 km locations counts distinct 1 km cells.",
    "0.5° blocks are the units of the label and scoring split, and Splits is the number of valid block splits.",
    "$E_0$ is the Stefan coefficient fitted on source cells outside the region and its 100 km buffer.",
    "Alaska share is the percentage of source cells in Alaska.",
    "Main regions were used in earlier experiments and are retested here.",
    "Alaska is a reference target not averaged with them.",
    "New regions were added after the label tables had been fixed (licence-verified edition).",
    "Sub-regions, point-estimate-only targets and the North Atlantic group are in Supplementary Table S1.",
    "Access conditions are given under Data availability.",
]
LEGEND_MAX_WORDS = 150
TABLE_SIZE = r"\footnotesize"   # 원고 10 pt 본문에서 8 pt(v2 와 같음)
TABCOLSEP = "4pt"                # 기본 6 pt 에서 줄여 표 폭을 본문 폭 안에 둔다
GAP_AFTER_REGION = "10pt"         # 지역 이름과 라벨 출처 사이만 넓게
ROW_INDENT = "0em"               # 묶음 머리는 기울임과 위 여백으로 구분한다(들여쓰기 없음)
MAX_TABLE_WIDTH_MM = 160.0       # [판단] A4 본문 폭(여백 25 mm)


# ---------------------------------------------------------------- 도우미
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_source(label_type: str) -> str:
    for head, name in LABEL_SOURCE_RULES:
        if str(label_type).startswith(head):
            return name
    raise ValueError(f"label_type 에 규칙이 없다: {label_type!r}")


def tex_text(s: str) -> str:
    """설명문 문자열(유니코드 °, √, %)을 LaTeX 본문으로 바꾼다. $…$ 수식은 그대로 둔다."""
    parts = re.split(r"(\$[^$]+\$)", s)
    out = []
    for p in parts:
        if p.startswith("$"):
            out.append(p)
            continue
        p = p.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
        p = p.replace("°", r"\textdegree{}").replace("√", r"\textsurd{}")
        p = re.sub(r"(\d) (km|cm|mm|m)\b", r"\1~\2", p)          # 숫자와 단위를 한 줄에
        p = re.sub(r"\b(Table|Fig\.) (S?\d)", r"\1~\2", p)
        out.append(p)
    return "".join(out)


def word_count(text: str) -> int:
    return len([w for w in re.sub(r"\$([^$]+)\$", r"\1", text).split() if w.strip()])


def haversine_min_km(a_latlon: np.ndarray, b_latlon: np.ndarray) -> float:
    """a 의 각 점에서 b 까지 최소 대권 거리의 최솟값(km)."""
    a = np.deg2rad(a_latlon)[:, None, :]
    b = np.deg2rad(b_latlon)[None, :, :]
    dlat = b[..., 0] - a[..., 0]
    dlon = b[..., 1] - a[..., 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(dlon / 2) ** 2
    return float((2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).min())


# ---------------------------------------------------------------- 자료
def build_rows() -> tuple[pd.DataFrame, dict]:
    rows = pd.read_csv(SRC_ROWS)
    src = pd.read_csv(SRC_SOURCE).set_index("target")
    meta = pd.read_csv(SRC_META).iloc[0]
    rec = []
    alaska_cells = int(rows.loc[rows.target == "Alaska", "label_rows"].iloc[0])  # 알래스카 v3 직접 라벨 셀 수
    for group, targets in GROUPS:
        for t in targets:
            r = rows[rows.target == t]
            if len(r) != 1:
                raise ValueError(f"table1_rows 에 {t} 행이 {len(r)}개다")
            r = r.iloc[0]
            if t in src.index:
                share = float(src.loc[t, "share_alaska"])
                n_src = int(src.loc[t, "n_src"])
                basis = "registered value"
            else:  # 새 지역: 알래스카 셀이 모두 원천에 남는지 verify() 가 확인한다
                n_src = int(r.n_src_x)
                share = alaska_cells / n_src
                basis = "derived: Alaska cells / source cells"
            rec.append(dict(
                group=group, region=S.REGION_NAME[t], label_source=label_source(r.label_type),
                labels=int(r.label_rows), locations_1km=int(r.loc_1km), blocks_0p5deg=int(r.blocks),
                valid_splits=int(r.valid_splits_x), source_coefficient=float(r.E0_x), source_cells=n_src,
                alaska_share_fraction=share, alaska_share_pct=int(round(100 * share)), alaska_share_basis=basis,
                _target=t, _label_type=r.label_type, _n_src_x=int(r.n_src_x)))
    df = pd.DataFrame(rec)
    ctx = dict(alaska_cells=alaska_cells, n_v3_cells=int(meta.n_v3_cells))
    return df, ctx


def display_cells(r: pd.Series) -> list[str]:
    return [r.region, r.label_source, S.fmt_int(r.labels), S.fmt_int(r.locations_1km), S.fmt_int(r.blocks_0p5deg),
            S.fmt_int(r.valid_splits), S.fmt_num(r.source_coefficient, 2), S.fmt_int(r.alaska_share_pct)]


# ---------------------------------------------------------------- LaTeX
def tabular_tex(df: pd.DataFrame) -> str:
    al = [c[2] for c in COLUMNS]
    spec = "@{}" + al[0] + r"@{\hspace{" + GAP_AFTER_REGION + "}}" + "".join(al[1:]) + "@{}"
    ncol = len(COLUMNS)

    def head(lines, align):
        if len(lines) == 1:
            return lines[0]
        return r"\begin{tabular}[b]{@{}" + align + r"@{}}" + r"\\".join(lines) + r"\end{tabular}"

    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(head(c[1], c[2]) for c in COLUMNS) + r" \\", r"\midrule"]
    first = True
    for group, _ in GROUPS:
        if not first:
            lines.append(r"\addlinespace")
        first = False
        lines.append(r"\multicolumn{" + str(ncol) + r"}{@{}l}{\textit{" + group + r"}} \\")
        for _, r in df[df.group == group].iterrows():
            cells = [tex_text(c) for c in display_cells(r)]
            if ROW_INDENT not in ("0em", "0pt"):
                cells[0] = r"\hspace{" + ROW_INDENT + "}" + cells[0]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def legend_tex() -> str:
    return tex_text(" ".join(LEGEND_BODY))


def table_env_tex(df: pd.DataFrame) -> str:
    return "\n".join([
        "% Table 1. Generated by script from the source tables; edit the script, not this file. Requires booktabs.",
        r"\begin{table}[htbp]",
        TABLE_SIZE + r"\setlength{\tabcolsep}{" + TABCOLSEP + "}",
        r"\caption{" + tex_text(TITLE) + "}",
        r"\label{tab:data}",
        r"\noindent" + tabular_tex(df),
        r"\par\medskip",
        r"\noindent " + legend_tex(),
        r"\end{table}", ""])


REVIEW_PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage[labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false,skip=6pt,font=footnotesize]{caption}
%GEOMETRY%
\pagestyle{empty}
\setlength{\parindent}{0pt}
\makeatletter
\renewenvironment{table}[1][]{\def\@captype{table}\begin{minipage}[t]{\linewidth}}{\end{minipage}}
\makeatother
"""


def compile_review(df: pd.DataFrame, tex_path: Path, build: Path) -> Path:
    """Table1_data.tex 를 그대로 \\input 하는 검토용 문서를 표 폭에 맞춘 쪽으로 조판한다(2회: 측정, 조판)."""
    build.mkdir(parents=True, exist_ok=True)
    shutil.copy(tex_path, build / tex_path.name)
    tab = tabular_tex(df)
    meas = REVIEW_PREAMBLE.replace("%GEOMETRY%", r"\usepackage[a3paper,margin=10mm]{geometry}") + "\n".join([
        r"\begin{document}", TABLE_SIZE + r"\setlength{\tabcolsep}{" + TABCOLSEP + "}",
        r"\setbox0=\hbox{" + tab + "}",
        r"\typeout{TABWD=\the\wd0}",
        r"\setlength{\linewidth}{\wd0}\setlength{\textwidth}{\wd0}\setlength{\hsize}{\wd0}",
        r"\setbox1=\vbox{\input{" + tex_path.stem + "}}",
        r"\typeout{TOTHT=\the\dimexpr\ht1+\dp1\relax}",
        r"\end{document}", ""])
    (build / "measure.tex").write_text(meas)
    log = _pdflatex(build, "measure")
    wd = float(re.search(r"TABWD=([\d.]+)pt", log).group(1))
    ht = float(re.search(r"TOTHT=([\d.]+)pt", log).group(1))
    margin_mm = 5.0
    slack_pt = 14.0                  # \topskip 와 마지막 줄 깊이 여유
    pt2mm = 25.4 / 72.27
    geom = (r"\usepackage[paperwidth=%.2fmm,paperheight=%.2fmm,textwidth=%.2fpt,left=%.1fmm,top=%.1fmm,"
            r"textheight=%.2fpt,nohead,nofoot]{geometry}" % (wd * pt2mm + 2 * margin_mm, (ht + slack_pt) * pt2mm + 2 * margin_mm,
                                                              wd, margin_mm, margin_mm, ht + slack_pt))
    doc = REVIEW_PREAMBLE.replace("%GEOMETRY%", geom) + "\n".join([
        r"\begin{document}", r"\input{" + tex_path.stem + "}", r"\end{document}", ""])
    (build / "review.tex").write_text(doc)
    _pdflatex(build, "review")
    return build / "review.pdf", dict(table_width_mm=wd * pt2mm, total_height_mm=ht * pt2mm)


def _pdflatex(build: Path, name: str) -> str:
    p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"], cwd=build,
                       capture_output=True, text=True)
    log = (build / f"{name}.log").read_text(errors="replace") if (build / f"{name}.log").exists() else p.stdout
    if p.returncode != 0:
        raise RuntimeError(f"pdflatex {name} 실패:\n" + log[-3000:])
    return log


# ---------------------------------------------------------------- 대조(Table1_values.txt)
def verify(df: pd.DataFrame, ctx: dict) -> list[str]:
    out = []
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        out.append(f"[{'OK' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    # 1. 원천 해시(MANIFEST)
    man = pd.read_csv(MANIFEST)
    for p in (SRC_ROWS, SRC_SOURCE, SRC_META):
        rel = str(p.relative_to(ROOT))
        exp = man.loc[man.orig_path == rel, "sha256"]
        chk(f"sha256 {rel} = MANIFEST", len(exp) and sha256(p) == exp.iloc[0], sha256(p)[:16])

    # 2. 표 값 = 원천 값(다시 읽어 대조)
    rows = pd.read_csv(SRC_ROWS).set_index("target")
    src = pd.read_csv(SRC_SOURCE).set_index("target")
    for _, r in df.iterrows():
        t = r._target
        q = rows.loc[t]
        chk(f"{r.region}: Labels {S.fmt_int(r.labels)} = table1_rows.label_rows", r.labels == int(q.label_rows))
        chk(f"{r.region}: 1 km locations {r.locations_1km} = table1_rows.loc_1km", r.locations_1km == int(q.loc_1km))
        chk(f"{r.region}: 0.5° blocks {r.blocks_0p5deg} = table1_rows.blocks", r.blocks_0p5deg == int(q.blocks))
        chk(f"{r.region}: Splits {r.valid_splits} = table1_rows.valid_splits_x", r.valid_splits == int(q.valid_splits_x))
        chk(f"{r.region}: E0 {r.source_coefficient:.6f} = table1_rows.E0_x", abs(r.source_coefficient - float(q.E0_x)) < 1e-12)
        if t in src.index:
            chk(f"{r.region}: E0 table1_rows = fig1_source.E0", abs(float(q.E0_x) - float(src.loc[t, "E0"])) < 1e-12)
            chk(f"{r.region}: Labels = fig1_source.n_target", r.labels == int(src.loc[t, "n_target"]))
            chk(f"{r.region}: n_src_x {int(q.n_src_x)} = fig1_source.n_src", int(q.n_src_x) == int(src.loc[t, "n_src"]))
            chk(f"{r.region}: Alaska share {r.alaska_share_fraction:.4f} = fig1_source.share_alaska",
                abs(r.alaska_share_fraction - float(src.loc[t, "share_alaska"])) < 1e-12)
        chk(f"{r.region}: Label source '{r.label_source}' <- '{q.label_type}'", True)

    # 3. D README 근거 표(E10–E18, E35, E37) 값과 대조(소수 둘째 자리)
    txt = D_README.read_text()
    ev = {}
    for line in txt.splitlines():
        m = re.match(r"\| (E\d\d) \|", line)
        if m:
            ev[m.group(1)] = [c.strip() for c in line.split("|")[1:-1]]
    readme_rows = {"Alaska": "E10", "Lena": "E11", "Canada": "E12", "Russia_W": "E13", "Russia_E": "E14",
                   "Russia_C_LGD": "E17", "Tibet_LGD": "E18"}
    for _, r in df.iterrows():
        e = ev.get(readme_rows[r._target])
        vals = re.findall(r"[\d,]+", e[5]) if e else []
        nums = [int(v.replace(",", "")) for v in vals[:3]]
        chk(f"{r.region}: Labels/1 km locations/blocks = D README {readme_rows[r._target]} {e[5] if e else None}",
            nums == [r.labels, r.locations_1km, r.blocks_0p5deg])
    e35 = dict(re.findall(r"(\w+) (0\.\d\d)", ev["E35"][5]))
    e37 = dict(re.findall(r"(\w+) (\d\.\d\d)", ev["E37"][5]))
    for _, r in df.iterrows():
        t = r._target
        if t in e37:
            chk(f"{r.region}: E0 {S.fmt_num(r.source_coefficient, 2)} = D README E37 {e37[t]}",
                S.fmt_num(r.source_coefficient, 2) == e37[t])
        if t in e35:
            chk(f"{r.region}: Alaska share {r.alaska_share_fraction:.2f} = D README E35 {e35[t]}",
                f"{r.alaska_share_fraction:.2f}" == e35[t])

    # 4. 새 지역: LGD 단위 파일(분할별)과 대조, 알래스카 비율 유도 근거
    for t, d in LGD_UNIT_DIR.items():
        units = [json.loads(p.read_text()) for p in sorted(d.glob("*__cpu__*_unit.json"))]
        r = df[df._target == t].iloc[0]
        e0s = {round(u["E0"], 12) for u in units}
        nsrc = {u["n_src"] for u in units}
        nbuf = {u["n_buffer_excluded"] for u in units}
        chk(f"{r.region}: LGD 단위 파일 {len(units)}개, E0 한 값 = 표 값", len(e0s) == 1 and abs(e0s.pop() - r.source_coefficient) < 1e-9)
        chk(f"{r.region}: LGD 단위 n_src {sorted(nsrc)} = table1_rows.n_src_x {r._n_src_x}", nsrc == {r._n_src_x})
        chk(f"{r.region}: LGD 단위 n_cells, n_blocks, n_valid_splits = 표 값",
            {(u['n_cells'], u['n_blocks'], u['n_valid_splits']) for u in units} == {(r.labels, r.blocks_0p5deg, r.valid_splits)})
        out.append(f"       {r.region}: n_buffer_excluded {sorted(nbuf)}, v3 직접 라벨 셀 {ctx['n_v3_cells']} − 원천 {r._n_src_x}"
                   f" = {ctx['n_v3_cells'] - r._n_src_x}")

    # 알래스카 셀이 새 지역의 대상이나 100 km 완충 안에 없는지(그러면 원천 알래스카 셀 = 알래스카 직접 라벨 셀 전부)
    sys.path.insert(0, str(ROOT / "src"))
    from polar.fidelity import MACRO_REGION
    v3 = pd.read_csv(V3_TABLE, usecols=["loc_id", "lat", "lon", "region", "source_id", "alt_cm"], low_memory=False)
    v3 = v3[(v3.source_id == "F4_direct") & np.isfinite(v3.alt_cm.astype(float))]
    v3["macro"] = v3.region.map(lambda x: MACRO_REGION.get(x, x))
    ak = v3[v3.macro == "Alaska"]
    chk(f"v3 직접 라벨 셀 {len(v3)} = fig1_meta.n_v3_cells {ctx['n_v3_cells']}", len(v3) == ctx["n_v3_cells"])
    chk(f"v3 알래스카 직접 라벨 셀 {len(ak)} = Alaska Labels {ctx['alaska_cells']} = max fig1_source.n_alaska "
        f"{int(src.n_alaska.max())}", len(ak) == ctx["alaska_cells"] == int(src.n_alaska.max()))
    b4 = pd.read_csv(V4_TABLE, usecols=["loc_id", "lat", "lon"])
    l4 = pd.read_csv(V4_LABELS, usecols=["loc_id", "macro_v4", "lgd_role"])
    tg = b4.merge(l4, on="loc_id")
    for t, macro in LGD_MACRO.items():
        r = df[df._target == t].iloc[0]
        cells = tg[(tg.macro_v4 == macro) & (tg.lgd_role == "target")]
        dmin = haversine_min_km(ak[["lat", "lon"]].to_numpy(float), cells[["lat", "lon"]].to_numpy(float))
        chk(f"{r.region}: 알래스카 셀과 대상 셀({len(cells)}개)의 최소 거리 {dmin:.0f} km > 완충 {BUFFER_KM:.0f} km",
            dmin > BUFFER_KM and len(cells) > 0)
        chk(f"{r.region}: Alaska share = {ctx['alaska_cells']} / {r._n_src_x} = {r.alaska_share_fraction:.4f} -> "
            f"{r.alaska_share_pct} %  [이번 계산, 등록 근거 표에 없는 값]", True)

    # 5. 표시 형식
    for _, r in df.iterrows():
        cells = display_cells(r)
        chk(f"{r.region}: 표시 {cells}", not any(S.COMMA4_RE.search(c) for c in cells) and
            all(("," in c) == (c.replace(",", "").isdigit() and int(c.replace(",", "")) >= 10000) for c in cells[2:]))
    out.insert(0, f"대조 전체: {'모두 통과' if ok_all else '실패 있음'}")
    return out, ok_all


def audit(tex: str, pdf: Path, legend_text: str, df: pd.DataFrame, dims: dict) -> tuple[list[str], bool]:
    """지침 7.2 T-01–T-05, 부록 A.1 audit_v3(표 글자 대리 그림), A.2 pdf_audit, 문장 점검."""
    out, ok_all = [], True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        out.append(f"[{'OK' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    body = re.sub(r"(?m)^%.*$", "", tex)
    i0 = body.index(r"\begin{tabular}{") + len(r"\begin{tabular}{")
    depth, i = 1, i0
    while depth:
        depth += {"{": 1, "}": -1}.get(body[i], 0)
        i += 1
    colspec = body[i0:i - 1]
    pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    npages = int(re.search(r"Pages:\s+(\d+)", pages).group(1))
    chk("T-01 booktabs 세 줄(\\toprule, \\midrule, \\bottomrule)", all(k in body for k in (r"\toprule", r"\midrule", r"\bottomrule")))
    chk("T-01 이미지 아님(\\includegraphics 0)", r"\includegraphics" not in body)
    chk(f"T-01 한 쪽 이내: 조판 {npages}쪽, 표+설명문 높이 {dims['total_height_mm']:.1f} mm, 표 폭 {dims['table_width_mm']:.1f} mm",
        npages == 1 and dims["total_height_mm"] < 247.0 and dims["table_width_mm"] <= MAX_TABLE_WIDTH_MM,
        "기준[판단]: A4 본문 160 × 247 mm")
    chk(f"T-02 세로선 0(열 지정 '{colspec}')", "|" not in colspec and r"\vline" not in body)
    chk("T-02 \\hline·셀 채움·색 글자 0",
        not re.search(r"\\(hline|cellcolor|rowcolor|columncolor|color|textcolor|colorbox)\b", body))
    chk("T-03 각주 0", not re.search(r"\\(footnote|footnotemark|footnotetext|tnote|tablefootnote)\b", body))
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    codes = [m.group(0) for m in S.CODE_RE.finditer(txt)]
    extra = re.findall(r"\b(PE\d|v\d|LGD|LG[A-Z]?|Point est\.?|mode[s]? [xi])\b|_", txt)
    chk("T-04 내부 코드 0(CODE_RE + PE·v3·LGD·Point est.·밑줄)", not codes and not extra, f"{codes} {extra}")
    chk("T-05 수치는 원천에서 생성(대조는 Table1_values.txt 1–5절)", True)
    chk("T-05 범위 표기 없음(대시 범위·대괄호 0)", not re.search(r"\d\s*[\u2013-]\s*\d|\[", txt.replace("0.5°", "")))
    ta = S.text_audit(txt + "\n" + legend_text)
    chk("M-01 연결어 대시 0", not ta["dash"], str(ta["dash"]))
    chk("M-02 금지 어휘 0", not ta["banned"], str(ta["banned"]))
    chk("F-14 네 자리 쉼표 0, 가운뎃점 숫자 나열 0", not ta["comma4"] and not ta["middot_num"], f"{ta['comma4']} {ta['middot_num']}")
    chk("음수 부호 U+2212(하이픈 음수 0)", not re.search(r"(?<![\w])-\d", txt))
    n_words = word_count(legend_text)
    chk(f"설명문 단어 수 {n_words} <= {LEGEND_MAX_WORDS}(그림 명세 10.4, 지침 3.1)", n_words <= LEGEND_MAX_WORDS)
    chk("설명문 첫 문장 명사구, 'This table' 시작 0", not legend_text.lower().startswith("this "))

    # A.2 pdf_audit: 쪽 크기, 글꼴(내장, Type 3 없음). 7 pt 기준은 그림 규칙이라 표에는 크기 분포만 기록한다.
    pa = S.pdf_audit(pdf)
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout.splitlines()[2:]
    not_emb = [f for f in fonts if f.split()[-5] != "yes"] if fonts else ["pdffonts 출력 없음"]
    chk(f"A.2 pdf_audit 쪽 {pa['width_mm']} × {pa['height_mm']} mm, 글자 {pa['chars']}자, Type 3 {pa['type3']}", not pa["type3"])
    chk(f"A.2 글꼴 모두 내장({len(fonts)}종)", not not_emb, str(not_emb))
    out.append(f"       글꼴 계열 {pa['families']}")
    out.append(f"       글자 크기 분포(pt: 자수) {dict(sorted(pa['sizes'].items()))}, 5 pt 미만 {pa['lt5']}자")

    # A.1 audit_v3: 표는 matplotlib 그림이 아니므로, 표의 모든 글자열을 7 pt 글자로 둔 대리 그림에 같은 함수를 돌린다.
    # 표 칸의 수치는 표의 내용이고(R-25 는 그림 규칙), 머리·범주 칸의 수는 이름과 단위('1 km', '0.5°')이므로
    # gid "table_value"와 "category"를 수치 검사에서 뺀다. 범주 칸의 4 단어 상한은 아래에서 따로 센다.
    S.use_v3("paper")
    fig = S.fig_mm(S.W2_MM, 120.0)
    def plain(h):
        h = h.replace(r"\textdegree{}", "°").replace(r"\textsurd", "√").replace("$E_0$", "E0").replace(r"\%", "%")
        return re.sub(r"\s+", " ", h.replace("~", " ")).strip()
    strings = [("category", plain(h), "header") for c in COLUMNS for h in c[1]]
    strings += [("category", g, "group") for g, _ in GROUPS]
    for _, r in df.iterrows():
        cells = display_cells(r)
        strings += [("category", cells[0], "row"), ("category", cells[1], "row")] + [("table_value", c, "value") for c in cells[2:]]
    for i, (gid, s, _k) in enumerate(strings):
        t = fig.text(0.02 + 0.24 * (i % 4), 0.97 - 0.035 * (i // 4), s, fontsize=S.FONT_PT, va="top")
        t.set_gid(gid)
    a = S.audit_v3(fig, allowed_num_gids=("scale", "sizekey", "table_value", "category"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    chk(f"A.1 audit_v3(표 글자 대리 그림, 글자열 {len(strings)}개) fails={a['fails']}", not a["fails"],
        f"codes={a['codes']} comma4={a['comma4']} long={a['long_labels']} numbers={a['loose_numbers']}")
    cat_long = [s for g, s, k in strings if k in ("group", "row") and len([w for w in s.split() if w != "+"]) > 4]
    chk("범주 라벨(묶음 머리, 지역, 라벨 출처) 4 단어 이하(R-28). 열 머리는 축 이름처럼 변수와 단위", not cat_long, str(cat_long))
    out.insert(0, f"점검 전체: {'모두 통과' if ok_all else '실패 있음'}")
    return out, ok_all


# ---------------------------------------------------------------- 실행
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=None, help="LaTeX 조판 임시 폴더(기본: 시스템 임시 폴더)")
    args = ap.parse_args(argv)

    df, ctx = build_rows()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) 원천 데이터
    pub = df[["group", "region", "label_source", "labels", "locations_1km", "blocks_0p5deg", "valid_splits",
              "source_coefficient", "source_cells", "alaska_share_fraction", "alaska_share_pct", "alaska_share_basis"]].copy()
    disp = np.array([display_cells(r) for _, r in df.iterrows()])
    for j, name in enumerate(["labels", "locations_1km", "blocks_0p5deg", "valid_splits", "source_coefficient",
                              "alaska_share_pct"], start=2):
        pub[f"{name}_shown"] = disp[:, j]
    pub = pub.rename(columns={"source_coefficient": "source_coefficient_E0_cm_per_sqrt_degC_d"})
    pub.to_csv(OUT / f"{STEM}_source_data.csv", index=False)

    # 2) LaTeX 표(정본)
    tex = table_env_tex(df)
    tex_path = OUT / f"{STEM}_data.tex"
    tex_path.write_text(tex)

    # 3) 설명문
    legend_text = TITLE + " " + " ".join(LEGEND_BODY)
    n_words = word_count(legend_text)
    (OUT / f"{STEM}_legend.md").write_text(
        f"**Table 1. {TITLE}** {' '.join(LEGEND_BODY)}\n\n"
        f"<!-- {n_words} words (limit {LEGEND_MAX_WORDS}, FIGURE_SPEC_v3 10.4). Generated by table1.py. -->\n")

    # 4) 검토용 조판(PDF, PNG 600 dpi)
    tmp = None
    if args.build_dir:
        build = Path(args.build_dir)
    else:
        tmp = tempfile.TemporaryDirectory()
        build = Path(tmp.name)
    pdf_built, dims = compile_review(df, tex_path, build)
    shutil.copy(pdf_built, OUT / f"{STEM}.pdf")
    subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", str(OUT / f"{STEM}.pdf"), str(OUT / STEM)], check=True)

    # 5) 대조와 점검
    vlines, v_ok = verify(df, ctx)
    alines, a_ok = audit(tex, OUT / f"{STEM}.pdf", legend_text, df, dims)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    head = [
        f"Table 1 v3 값 대조와 점검 기록({now}, scripts/4_visualization/paper_v3/table1.py 생성)",
        "",
        "원천: data/processed/paper_figs/table1_rows.csv, fig1_source.csv, fig1_meta.csv(D README E10–E19, E33–E37).",
        "대조용: LGD 단위 파일(data/processed/lgd/{Russia_C,Tibet}/shards/*__cpu__*_unit.json), fidelity_base_v3.csv,",
        "fidelity_base_v4.csv, fidelity_base_v4_labels.csv, paper/claims/D_data_and_design/README.md 근거 표.",
        "",
        "표기: [OK] 통과, [FAIL] 실패. '이번 계산'은 등록 근거 표에 없는 값을 등록 수치로 유도한 것이다.",
        "",
        "== 1. 원천 대조 ==",
    ]
    notes = [
        "",
        "== 3. 남은 확인 사항 ==",
        "- 새 지역 2행의 알래스카 비율(78 %)은 이번 계산이다. 알래스카 직접 라벨 셀 13,606개(Alaska Labels = "
        "fig1_source.n_alaska 최댓값)를 LGD 단위 파일의 원천 셀 수(n_src)로 나눴다. 알래스카 셀이 대상이나 100 km "
        "완충 안에 없음을 거리로 확인했다(1절 '최소 거리' 행). D README 근거 표(E35)에 행을 더할지는 사용자 확인 사항이다.",
        "- 명세와 다르게 둔 것: (가) E0 열 머리를 'Source coefficient $E_0$ (cm per √(°C d))'로 썼다(그림 명세 10.2 는 "
        "'$E_0$ (cm per √(°C d))', 지침 6.8 과 원고 명세 7절은 'Source coefficient'). (나) 백분율 기호를 값이 아닌 열 머리 "
        "'(%)'에 두었다(지침 3.1 '단위는 열 머리 괄호에'). (다) 설명문은 그림 명세 10.4 초안의 내용을 따르되 150단어 안에 "
        "라벨 단위 문장을 더하려고 문장을 줄였다. (라) 표 글자는 8 pt(\\footnotesize, v2 와 같음), 열 간격 4 pt 로 표 폭을 "
        f"{dims['table_width_mm']:.1f} mm 로 맞췄다. 9 pt 에서는 열 간격 3 pt 에도 160.4 mm 로 160 mm 를 넘었다(2026-10-04 시험 조판).",
        "- 행 구성은 그림 명세 10.3(D-5, D-15 기본값)을 따랐다. 하위 지역, 점 추정 전용 대상(Central Russia v3 7셀, "
        "Greenland 3셀), North Atlantic 은 Supplementary Table S1 로 옮긴다. XF 행은 더하지 않았다(등록 문서 6절).",
        "- 라벨 출처 이름: ALLena 는 Veremeeva et al. 2025(PANGAEA 973813) 자료명 'ALLena: Thaw depth measurements ...',",
        "  ABoVE 는 Moore et al. 2025(ORNL DAAC 2369) 자료명 'ABoVE: Soil Moisture and Active Layer Thickness ...'와 같다",
        "  (docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md 참고문헌 표 263, 289행). CALM 은 GTN-P/CALM(Streletskiy et al. 2025).",
        "- 지역 이름 'W Russia', 'E Russia', 'Lena Delta'의 표기와 레나델타 범위는 사용자 확인 사항이다(지침 2.5, 8절 7, 원고 명세 [DECISION]).",
        "- 짧은 라벨 출처 이름은 원천 label_type 의 첫 자료원만 보인다. 알래스카·캐나다의 CALM 지점 평균 일부와 "
        "Central Russia (expanded)의 v3 CALM 6셀은 설명문 둘째 문장에 적었다.",
        "- 검토용 조판은 Latin Modern(원고 sn-jnl 의 Computer Modern 계열)으로 했다. 원고 안 최종 글꼴과 크기는 템플릿과 "
        "출판사 조판을 따른다. 7 pt 단일 크기 규칙(지침 2.2)은 그림 규칙이라 표에 적용하지 않았다.",
        "- Table1_data.tex 는 그림 명세 1.7이 정한 Table 1 정본 경로이다(지침 R-14: 편집 가능한 LaTeX 표).",
    ]
    text = "\n".join(head + vlines + ["", "== 2. 형식 점검(지침 7.2 T-01–T-05, 부록 A.1, A.2, 7.3 M-01·M-02) =="] + alines + notes) + "\n"
    (OUT / f"{STEM}_values.txt").write_text(text)
    if tmp is not None:
        tmp.cleanup()
    print(text)
    return 0 if (v_ok and a_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
