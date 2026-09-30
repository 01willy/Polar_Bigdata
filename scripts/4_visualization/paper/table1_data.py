"""Table 1 (v2). 자료와 대상. 스펙: figure_spec.json display_items 'Table 1' f3_v2.

행: 주 지역 6(모드 x), Alaska(x) 참조, 하위 지역 10(모드 i·x 한 행) = 대상 27, 그 아래 LGD 새 지역(약관 확인분 판).
열: 역할, 라벨 행 수, 1 km 위치 수, 0.5° 블록 수, 유효 분할, |A| 범위, 채점 셀(블록) 범위, E0, 라벨 유형, 약관.
출력: outputs/figures/paper/v2/Table1_data.{csv, tex, md} 와 미리보기 {pdf, svg, png}(QA 용), source_data/v2/Table1_*.csv.
자료: data/processed/paper_figs/table1_rows.csv, table1_expanded.csv, fig1_not_drawn.csv(v2_data.py). 약관 미확인 셀은 행에 넣지 않고 수만 각주.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps                                                  # noqa: E402

NAME = "Table1_data"
SPEC_ID = "paper_table1_data"
TYPE_SHORT = {"Point (ALLena, mostly single-year visits)": "Point, ALLena (single visits)",
              "Point (ABoVE, mostly GPR); some CALM site means": "Point, ABoVE (mostly GPR)",
              "Point (ABoVE); some CALM site means": "Point, ABoVE",
              "CALM site, multi-year mean": "CALM site mean"}
METHOD_EN = {"pit_core": "pit/core", "probe": "probe", "other": "other", "gpr": "GPR", "pit_core;probe": "pit/core + probe",
             "borehole_temp": "borehole T"}
ROLE_SHORT = {"P4 (confirmatory)": "P4", "Point estimate only": "Point est.", "Reference (not pooled)": "Reference"}


def _fmt_int(v) -> str:
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{int(v):,}"


def _fmt_range(s) -> str:
    if not isinstance(s, str) or not s:
        return "–"
    a, b = s.split("–")
    return f"{int(a):,}" if a == b else f"{int(a):,}–{int(b):,}"


def build_rows():
    T = V.rd("table1_rows")
    rows = []
    for r in T.itertuples():
        sub = r.group == "Sub-region"
        lgd = r.group.startswith("New region")
        if r.target == "NAtlantic":
            continue
        if sub:
            role = "Sub-region"
            e0 = f"{r.E0_x:.2f} / {r.E0_i:.2f}"
        else:
            role = ROLE_SHORT.get(r.role, r.role)
            e0 = f"{r.E0_x:.2f}"
        if lgd:
            import re
            m = re.findall(r"([a-z_;]+) (\d+)", r.label_type.split("(")[1].split(")")[0])
            ltype = "1 km cell means: " + ", ".join(f"{METHOD_EN.get(k, k)} {v}" for k, v in m)
            if "v3 CALM" in r.label_type:
                ltype += "; plus 6 v3 CALM sites"
            lic = "all verified"
        else:
            ltype = TYPE_SHORT.get(r.label_type, r.label_type)
            lic = "v3"
        scored = f"{_fmt_range(r.eval_cells_x)} ({_fmt_range(r.eval_blocks_x)})"
        rows.append({"Group": r.group, "Target": r.name + (" (x)" if r.target == "Alaska" and "(x)" not in r.name else ""),
                     "Modes": r.modes.replace(",", ", "), "Role": role, "Rows": _fmt_int(r.label_rows),
                     "1 km loc.": _fmt_int(r.loc_1km), "Blocks": _fmt_int(r.blocks), "Splits": _fmt_int(r.valid_splits_x),
                     "A cells": _fmt_range(r.A_cells_x), "Scored cells (blocks)": scored, "E0, x / i": e0, "Label type": ltype,
                     "Licence": lic})
    return T, pd.DataFrame(rows)


def footnotes(T, ex, nd) -> list:
    nat = T[T.target == "NAtlantic"].iloc[0]
    ndt = ", ".join(f"{ps.region_name(r) if r != 'NAtlantic' else 'North Atlantic'} {int(n)}" for r, n in zip(nd.region, nd.n_cells_not_drawn))
    exs = "; ".join(f"{ps.region_name(r.target)} {int(r.n_cells)} cells" + (f" ({int(r.n_cells_lic)} licence-verified)" if np.isfinite(r.n_cells_lic) else "")
                    for r in ex.itertuples())
    n27 = int(T.n_targets.sum())
    return [
        f"Targets: {n27} (six main regions and Alaska in mode x; ten sub-regions in modes i and x). P4 = Lena, Canada, Russia W and "
        "Russia E. AL, CA and LE sub-regions belong to Alaska, Canada and Lena and are not independent.",
        "Rows, label rows of the label table (n counts rows); 1 km loc., locations by the cell index ky = floor(lat/0.009), "
        "kx = floor(lon cos φ/0.009); Blocks, 0.5° blocks; Splits, valid splits of five.",
        "A cells and scored cells are ranges over valid splits (A = labelled half, scored = evaluation cells of the B half); "
        "Greenland has no valid split and is scored on invalid splits for point estimates only.",
        "E0 in cm (°C d)^−1/2, source pool of mode x (sub-regions: x / i).",
        f"LGD rows are the licence-verified edition (main verdicts). Cells with unverified licence are not listed ({ndt}). "
        f"North Atlantic ({int(nat.label_rows)} cells in the full edition, {int(nat.label_rows) - 19} licence-verified) is ineligible "
        "in the licence-verified edition and appears in the SI only.",
        f"Expanded versions used only for the L40 sensitivity: {exs}.",
    ]


def to_md(D: pd.DataFrame, notes) -> str:
    cols = [c for c in D.columns if c != "Group"]
    out = ["**Table 1. Targets, labels and data sources.**", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    g0 = None
    for _, r in D.iterrows():
        if r["Group"] != g0:
            out.append("| *" + r["Group"] + "* |" + " |" * (len(cols) - 1))
            g0 = r["Group"]
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    out.append("")
    for i, n in enumerate(notes):
        out.append(f"{chr(97 + i)}. {n}")
    return "\n".join(out) + "\n"


def _tex(s: str) -> str:
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("^−1/2", "$^{-1/2}$"),
                 ("–", "--"), ("−", "$-$"), ("°", "\\textdegree{}"), ("φ", "$\\varphi$"), ("~", "$\\sim$")):
        s = s.replace(a, b)
    return s


def to_tex(D: pd.DataFrame, notes) -> str:
    cols = [c for c in D.columns if c != "Group"]
    spec = "l" * 3 + "r" * 7 + "l" * (len(cols) - 10)
    lines = ["% Table 1 (v2). Generated by scripts/4_visualization/paper/table1_data.py. Requires booktabs.",
             "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\caption{Targets, labels and data sources.}", "\\label{tab:data}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(_tex(c) for c in cols) + " \\\\", "\\midrule"]
    g0 = None
    for _, r in D.iterrows():
        if r["Group"] != g0:
            if g0 is not None:
                lines.append("\\addlinespace")
            lines.append(f"\\multicolumn{{{len(cols)}}}{{l}}{{\\textit{{{_tex(r['Group'])}}}}} \\\\")
            g0 = r["Group"]
        lines.append(" & ".join(_tex(r[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip", "\\begin{minipage}{\\textwidth}\\scriptsize"]
    lines += [f"({chr(97 + i)}) {_tex(n)}\\\\" for i, n in enumerate(notes)]
    lines += ["\\end{minipage}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def preview(D: pd.DataFrame, notes):
    """180 mm 미리보기(QA 용). 글자 6.5 pt, 격자 배치, 라벨 유형 칸은 줄바꿈."""
    import textwrap
    ps.use_paper()
    cols = [c for c in D.columns if c not in ("Group",)]
    widths = {"Target": 25, "Modes": 8, "Role": 17, "Rows": 10, "1 km loc.": 10, "Blocks": 9, "Splits": 8,
              "A cells": 14, "Scored cells (blocks)": 23, "E0, x / i": 14, "Label type": 29, "Licence": 11}
    wrapc = {"Label type": 25, "Role": 15}
    x = [2.0]
    for c in cols[:-1]:
        x.append(x[-1] + widths[c])
    lh = 2.75
    cell = []
    for _, r in D.iterrows():
        cell.append({c: ("\n".join(textwrap.wrap(str(r[c]), wrapc[c])) if c in wrapc else str(r[c])) for c in cols})
    nl = [max(v.count("\n") + 1 for v in d.values()) for d in cell]
    groups = D.Group.nunique()
    wrapped = ["\n".join(textwrap.wrap(n.replace("^−1/2", "$^{−1/2}$"), 150)) for n in notes]
    nnote = sum(w.count("\n") + 1 for w in wrapped)
    H = 10.0 + 7.0 + sum(n * lh + 0.9 for n in nl) + groups * 3.4 + 4.0 + nnote * 2.9 + 0.5 * len(notes) + 3.0
    fig = ps.paper_figure(180.0, H)
    ax = ps.axes_mm(fig, 0, 0, 180.0, H)
    ax.set_xlim(0, 180); ax.set_ylim(0, H); ax.axis("off")
    fs = ps.FS["annot"]
    y = H - 3.0
    ax.text(2.0, y, "Table 1. Targets, labels and data sources.", ha="left", va="top", fontsize=ps.FS["base"], fontweight="bold")
    y -= 7.0
    ax.plot([2, 178], [y + 3.9, y + 3.9], color="#000000", lw=0.8)
    for c, xx in zip(cols, x):
        ax.text(xx, y + 1.2, "\n".join(textwrap.wrap(c, max(4, int((widths[c] - 1.0) / 1.35)), break_long_words=False)), ha="left",
                va="center", fontsize=fs,
                fontweight="bold", linespacing=0.95)
    y -= 2.6
    ax.plot([2, 178], [y + 0.6, y + 0.6], color="#000000", lw=0.5)
    g0 = None
    for (_, r), d, n in zip(D.iterrows(), cell, nl):
        if r.Group != g0:
            y -= 3.4
            ax.text(2.0, y + 1.7, r.Group, ha="left", va="center", fontsize=fs, fontstyle="italic")
            g0 = r.Group
        h = n * lh + 0.9
        y -= h
        for c, xx in zip(cols, x):
            ax.text(xx, y + h - 0.45, d[c], ha="left", va="top", fontsize=fs, linespacing=1.0)
    ax.plot([2, 178], [y - 0.4, y - 0.4], color="#000000", lw=0.8)
    y -= 2.5
    for i, w in enumerate(wrapped):
        ax.text(2.0, y, f"{chr(97 + i)}. {w}", ha="left", va="top", fontsize=fs, linespacing=1.1)
        y -= 2.9 * (w.count("\n") + 1) + (0.8 if "$" in w else 0.3)
    ax.set_gid("table")
    return fig


def build():
    T, D = build_rows()
    ex = V.rd("table1_expanded")
    nd = V.rd("fig1_not_drawn")
    notes = footnotes(T, ex, nd)
    V.OUT_V2.mkdir(parents=True, exist_ok=True)
    D.to_csv(V.OUT_V2 / f"{NAME}.csv", index=False)
    (V.OUT_V2 / f"{NAME}.md").write_text(to_md(D, notes))
    (V.OUT_V2 / f"{NAME}.tex").write_text(to_tex(D, notes))
    fig = preview(D, notes)
    cap = dict(
        definition=("Targets are regions or sub-regions held out from the source pool; mode x removes the parent region, mode i keeps it. "
                    "n counts label rows; 1 km locations group rows by a 0.009° cell index."),
        statistics="Descriptive; no tests.",
        panels=("Main regions, reference and sub-regions give the 27 LG targets. LGD rows are the licence-verified edition; cells with "
                "unverified licence are only counted in footnote e."),
        data=("v3 and v4 label tables, LG target table (Rescale) and LGD unit files (local); label units from h39 (units mode)."),
    )
    src = dict(rows=T, display=D, expanded=ex)
    spec_extra = dict(module="table1_data.py", panels=0, message="27 targets and LGD regions; label rows vs 1 km locations",
                      exports_table=[str((V.OUT_V2 / f"{NAME}.{k}").relative_to(V.ROOT)) for k in ("csv", "tex", "md")])
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="Targets, labels and data sources.", spec_extra=spec_extra)


if __name__ == "__main__":
    build()
