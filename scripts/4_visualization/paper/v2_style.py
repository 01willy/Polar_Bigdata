"""F3 1부(본문 Fig 2–4, v2) 공용 그림 도우미. paperstyle(ps)·_common 을 재사용하고 v2 전용 규칙만 더한다.

- 범주 n 축(0, 3, 10, 40, 160, 320, 1,000, 축 끊김, all)
- 4분 판정 기호(모양 + 채움, 회색 단계만. 색으로 구분하지 않는다)
- 저장: outputs/figures/paper/v2/<이름>.{pdf, svg, png(600 dpi)}, _qa/v2_<이름>_qa.json·_100pct.png,
  source_data/v2/<Fig>_<패널>.csv, CAPTIONS.md 의 '## v2/<이름>' 절, figure_spec.json figures[] 의 spec_id 항목(들여쓰기 2)
- 09-26 산출(outputs/figures/paper/Fig*.pdf/png, source_data/*.csv)은 건드리지 않는다.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import _common as C
from _common import ps

ROOT = C.ROOT
DERIVED = ROOT / "data" / "processed" / "paper_figs"
OUT_V2 = C.OUT / "v2"
SRC_V2 = C.SRC_DIR / "v2"
for _d in (OUT_V2, SRC_V2):
    _d.mkdir(parents=True, exist_ok=True)

N_ALL = -1
N_GRID = [0, 3, 10, 40, 160, 320, 1000]
X_ALL = 7.45                                   # 'all' 위치(1,000 = 6 과 사이에 축 끊김)
X_BREAK = 6.72
NPOS = {n: i for i, n in enumerate(N_GRID)} | {N_ALL: X_ALL}
N_TICK = {0: "0", 3: "3", 10: "10", 40: "40", 160: "160", 320: "320", 1000: "1,000", N_ALL: "all"}
MINUS = "−"

# ---------------------------------------------------------------- 방법 표기(범례·축 글자)
M_STYLE = {                                    # key → (paperstyle METHOD 키, 선종 변형, 표시 이름)
    "P1": ("refit", None, "P1: Stefan, E re-fit (κ = 10)"),
    "R0": ("residual", "dashed", "R0: P0 + residual ML"),
    "R1": ("residual", None, "R1: P1 + residual ML"),
    "R2": ("augment", None, "R2: R1 + pseudo-label rows"),
    "D0": ("direct", None, "D0: direct ML"),
}
PSTAR_STYLE = dict(color="#000000", ls=(0, (1.2, 1.4)), lw=0.9)           # P* 가로선(점선)
PSTAR_LABEL = "P*: P0 with year-matched TDD (n = 0)"
P1STAR_LABEL = "P1*: edaphic Stefan, re-fit (n = 40, 160)"


def mstyle(key: str, filled: bool = True, ms: float | None = None) -> dict:
    m, var, _ = M_STYLE[key]
    return ps.style_of(m, var, filled=filled, ms=ms)


def method_handle(key: str) -> Line2D:
    m, var, lab = M_STYLE[key]
    return ps.method_handle(m, lab, variant=var)


def pstar_handle(label: str = PSTAR_LABEL) -> Line2D:
    return Line2D([], [], label=label, **PSTAR_STYLE)


def p1star_handle(label: str = P1STAR_LABEL, method: str = "refit") -> Line2D:
    c = ps.COLOR[method]
    return Line2D([], [], ls="none", marker="s", ms=ps.MS["main"] + 0.4, mfc="white", mec=c, mew=0.9, color=c, label=label)


# ---------------------------------------------------------------- 범주 n 축
def n_axis(ax, ns=None, lim=(-0.55, X_ALL + 0.55), labels=True, xlabel: str | None = "Target labels, n", break_mark=True):
    """범주 n 축. ns = 눈금을 둘 n 목록(기본 전체). 1,000 과 all 사이에 축 끊김 표시."""
    ns = N_GRID + [N_ALL] if ns is None else list(ns)
    ax.set_xlim(*lim)
    ax.set_xticks([NPOS[n] for n in ns])
    ax.set_xticklabels([N_TICK[n] for n in ns] if labels else [])
    ax.tick_params(axis="x", length=2.0)
    if xlabel:
        ax.set_xlabel(xlabel)
    if break_mark and N_ALL in ns and lim[1] > X_BREAK:
        xbreak(ax)


def xbreak(ax, x=X_BREAK, w=0.09, h_pt=2.6):
    """x 축 위 '//' 끊김 표시(축 좌표 y = 0)."""
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    fig = ax.figure
    h = h_pt / 72 * fig.dpi / ax.bbox.height if ax.bbox.height > 0 else 0.03
    for dx in (-w / 2, w / 2):
        ln = ax.plot([x + dx - 0.05, x + dx + 0.05], [-h, h], color="#000000", lw=0.6, transform=tr, clip_on=False, zorder=6)[0]
        ln.set_gid("xbreak")
    ax.add_patch(Rectangle((x - w / 2 + 0.01, -h / 2), w - 0.02, h, transform=tr, facecolor="white", edgecolor="none", clip_on=False, zorder=5.5))


def symlog_y(ax, lim, ticks, linthresh=2.0, linscale=1.0, axis="y"):
    """Δ symlog 축(선형 ±linthresh cm). linscale < 1 이면 로그 구간(큰 |Δ|)의 눈금 간격이 넓어진다."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    (ax.set_yscale if axis == "y" else ax.set_xscale)("symlog", linthresh=linthresh, linscale=linscale)
    (ax.set_ylim if axis == "y" else ax.set_xlim)(*lim)
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(list(ticks))); a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))


def linear_y(ax, lim, ticks):
    """선형 Δ 축(풀 판처럼 |Δ| 가 작은 패널)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.set_ylim(*lim)
    ax.yaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))


def better_note(ax, text="negative = lower error than the baseline", loc=(0.99, 0.97), ha="right", va="top"):
    """'더 나음' 방향 표지(6.5 pt, 그림당 1회). 자료가 없는 모서리에 둔다."""
    t = ax.text(*loc, text, transform=ax.transAxes, ha=ha, va=va, fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    t.set_gid("better")
    return t


def unavailable_band(ax, x0, x1, text=None, y_text=0.5):
    """격자에 없는 n 구간(러시아 W·E 의 n ≥ 40)을 옅은 빗금으로."""
    r = ax.axvspan(x0, x1, facecolor="none", edgecolor="#c8c8c8", hatch="//////", lw=0, zorder=0.2)
    if text:
        ax.text((x0 + x1) / 2, y_text, text, transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                ha="center", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], rotation=0, zorder=0.3,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    return r


# ---------------------------------------------------------------- 4분 판정 기호
VERDICT_ORDER = ["superior", "equivalent", "undecided", "inferior", "not determinable"]
VERDICT_TEXT = {"superior": "Lower error", "equivalent": "Equivalent (±0.5 cm)", "undecided": "Undecided", "inferior": "Higher error",
                "not determinable": "Not determinable"}
VSYM = {
    "superior": dict(marker="v", mfc="#000000", mec="#000000", ms=4.0),
    "equivalent": dict(marker="o", mfc="white", mec="#000000", ms=3.8),
    "undecided": dict(marker="D", mfc="#bdbdbd", mec="#000000", ms=3.1),
    "inferior": dict(marker="^", mfc="white", mec="#000000", ms=4.0),
    "not determinable": dict(marker="x", mfc="none", mec="#000000", ms=3.4),
}
DAGGER = "†"


def draw_verdict(ax, x, y, verdict: str, small: bool = False, transform=None, zorder=5):
    """판정 기호 하나. small = '통계적으로 구별되나 크기는 0.5 cm 미만' → 오른쪽 위 †."""
    kw = dict(transform=transform) if transform is not None else {}
    if verdict in VSYM:
        s = VSYM[verdict]
        ax.plot([x], [y], ls="none", marker=s["marker"], ms=s["ms"], mfc=s["mfc"], mec=s["mec"], mew=0.7, color=s["mec"],
                clip_on=False, zorder=zorder, **kw)
        if small:
            t = ax.annotate(DAGGER, xy=(x, y), xycoords=transform or ax.transData, xytext=(2.6, 0.8), textcoords="offset points",
                            fontsize=ps.FS["annot"], ha="left", va="center", annotation_clip=False)
            t.set_gid("dagger")
    elif verdict == "same as P1":
        ax.text(x, y, "=P1", ha="center", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], clip_on=False, **kw)
    elif verdict == "no row":
        ax.text(x, y, "·", ha="center", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], clip_on=False, **kw)


def verdict_key(ax, x0, y, dx_list=None, include_small=True, fontsize=None):
    """판정 기호 설명 한 줄(축 좌표 x0, y 에서 시작). ax 는 보이지 않는 축이어도 된다. 반환: 끝 x."""
    fs = fontsize or ps.FS["annot"]
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x = x0
    inv = ax.transAxes.inverted()
    items = [(v, VERDICT_TEXT[v]) for v in VERDICT_ORDER]
    if include_small:
        items.append(("dagger", f"{DAGGER} |Δ| < 0.5 cm"))
    for v, lab in items:
        if v == "dagger":
            t = ax.text(x, y, lab, transform=ax.transAxes, fontsize=fs, ha="left", va="center")
        else:
            draw_verdict(ax, x + 0.012, y, v, transform=ax.transAxes)
            t = ax.text(x + 0.028, y, lab, transform=ax.transAxes, fontsize=fs, ha="left", va="center")
        bb = t.get_window_extent(rend)
        x = inv.transform((bb.x1, bb.y0))[0] + 0.03
    return x


def verdict_key_grid(fig, rect_mm, ncol=2, same_as_p1=True, col_w=None, extra=None, text_dx_mm=6.0, verdicts=True):
    """판정 기호 설명을 격자로(범례 객체가 아니라 기호 + 글자). rect_mm = (x, y, w, h).
    extra = [(짧은 글자, 설명)] 는 글자 표지(n.c., AB 번호 등)를 같은 격자에 더한다."""
    ax = ps.axes_mm(fig, *rect_mm)
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    items = ([(v, VERDICT_TEXT[v]) for v in VERDICT_ORDER] + [("dagger", f"{DAGGER}  |Δ| < 0.5 cm")]) if verdicts else []
    if same_as_p1:
        items.append(("same as P1", "P1* = P1 at this n"))
    for sym, lab in (extra or []):
        items.append((("txt", sym), lab))
    nrow = int(np.ceil(len(items) / ncol))
    cw = col_w or [1.0 / ncol] * ncol
    cx = np.concatenate([[0.0], np.cumsum(cw)[:-1]])
    mm = 1.0 / rect_mm[2]                                  # 축 좌표 1 mm
    for k, (v, lab) in enumerate(items):
        col, row = k // nrow, k % nrow
        x = cx[col] + 0.5 * mm
        y = 1 - (row + 0.5) / nrow
        if v == "dagger":
            ax.text(x, y, lab, fontsize=ps.FS["annot"], ha="left", va="center")
        elif v == "same as P1":
            ax.text(x, y, "=P1", fontsize=ps.FS["annot"], ha="left", va="center", color=ps.GREY["text2"])
            ax.text(x + 6.0 * mm, y, lab, fontsize=ps.FS["annot"], ha="left", va="center")
        elif isinstance(v, tuple):
            ax.text(x, y, v[1], fontsize=ps.FS["annot"], ha="left", va="center", color=ps.GREY["text2"])
            ax.text(x + text_dx_mm * mm, y, lab, fontsize=ps.FS["annot"], ha="left", va="center")
        else:
            draw_verdict(ax, x + 1.2 * mm, y, v)
            ax.text(x + 3.6 * mm, y, lab, fontsize=ps.FS["annot"], ha="left", va="center")
    ax.set_gid("verdict_key")
    return ax


def verdict_table(ax, rows, cols, cells, row_labels, col_labels, sub_labels=None, col_x=None, xlim=None,
                  head_title=None, sub_title=None, row_label_fs=None, tags=None):
    """기호 표. rows·cols = 키 목록, cells = {(row, col): (verdict, small)}. 데이터 좌표: 열 x = col_x(기본 0..k−1),
    행 y = m−1..0(첫 행이 위). 머리 두 줄(열 이름 = n, 아래 줄 = 풀 구성)은 첫 행 위에 글자로 둔다.
    head_title·sub_title 은 두 머리 줄의 왼쪽 설명(행 이름 열 자리)."""
    fs = ps.FS["annot"]
    m = len(rows)
    xs = list(range(len(cols))) if col_x is None else list(col_x)
    for i, r in enumerate(rows):
        yy = m - 1 - i
        for j, c in enumerate(cols):
            v = cells.get((r, c))
            if v is None:
                continue
            if v[0] == "not computed":
                ax.text(xs[j], yy, "n.c.", ha="center", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
                continue
            draw_verdict(ax, xs[j], yy, v[0], v[1])
    ax.set_yticks([m - 1 - i for i in range(m)])
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="y", length=0, pad=2, labelsize=row_label_fs or ps.FS["tick"])
    ax.set_xticks([])
    y_head, y_sub = m - 1 + 1.55, m - 1 + 0.85
    for j, lab in enumerate(col_labels):
        ax.text(xs[j], y_head, lab, ha="center", va="center", fontsize=ps.FS["tick"], clip_on=False)
    if sub_labels is not None:
        for j, lab in enumerate(sub_labels):
            ax.text(xs[j], y_sub, lab, ha="center", va="center", fontsize=fs, color=ps.GREY["text2"], clip_on=False)
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    if head_title:
        ax.text(-0.02, y_head, head_title, transform=tr, ha="right", va="center", fontsize=ps.FS["tick"], clip_on=False)
    if sub_title and sub_labels is not None:
        ax.text(-0.02, y_sub, sub_title, transform=tr, ha="right", va="center", fontsize=fs, color=ps.GREY["text2"], clip_on=False)
    ax.set_ylim(-0.6, m - 1 + 2.0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.axhline(m - 1 + 0.45, color=ps.GREY["light"], lw=0.5, zorder=0)
    for (r, c), code in (tags or {}).items():                   # AB 표지: 기호 아래 글자
        if r in rows and c in cols:
            ab_tag(ax, xs[cols.index(c)], m - 1 - rows.index(r), code)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("none")
    return xs


# ---------------------------------------------------------------- 자료 읽기
def rd(name: str) -> pd.DataFrame:
    p = DERIVED / f"{name}.csv"
    if not p.exists():
        raise C.MissingData(str(p.relative_to(ROOT)) + " (먼저 v2_data.py 실행)")
    return pd.read_csv(p, low_memory=False)


def pool_code(pool: str) -> str:
    """'지역 4/4' → 'P4', '지역 2/4' → 'L+C', '지역 1/4' → 'Lena'."""
    s = str(pool)
    return {"지역 4/4": "P4", "지역 2/4": "L+C", "지역 1/4": "Lena"}.get(s, "")


# ---------------------------------------------------------------- 저장·QA·기록
def _svg_text_ok(svg_path: Path) -> dict:
    s = svg_path.read_text(errors="ignore")
    n_text = len(re.findall(r"<text\b", s))
    n_glyph_paths = len(re.findall(r'id="(?:DejaVuSans|LiberationSans)[^"]*"', s))
    return dict(svg_text_elements=n_text, svg_glyph_defs=n_glyph_paths, svg_text_ok=n_text > 0 and n_glyph_paths == 0)


def _png_dpi(png: Path) -> float | None:
    try:
        from PIL import Image
        with Image.open(png) as im:
            d = im.info.get("dpi")
            return float(d[0]) if d else None
    except Exception:                                       # noqa: BLE001
        return None


def write_caption_v2(name: str, caption: dict, title: str | None = None) -> dict:
    return C.write_caption(f"v2/{name}", caption, title)


def update_spec_registry(entry: dict) -> None:
    """figure_spec.json figures[] 의 id 항목을 교체(없으면 추가). 원 파일 형식(들여쓰기 2)을 유지한다."""
    with C._locked(C.FIG_SPEC):
        d = json.loads(C.FIG_SPEC.read_text())
        figs = d.setdefault("figures", [])
        for i, f in enumerate(figs):
            if f.get("id") == entry["id"]:
                figs[i] = entry
                break
        else:
            figs.append(entry)
        C.FIG_SPEC.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")


def save_v2(fig, name: str, caption: dict, spec_id: str, sources: dict, title: str | None = None, spec_extra: dict | None = None,
            width_mm: float = ps.W2_MM, register: bool = True) -> dict:
    """PDF(Type 42)·SVG(글자 = 텍스트)·600 dpi PNG 저장 → _common.qa_check → 기록. 실패하면 파일명에 _FAIL, 기록 생략."""
    import matplotlib.pyplot as plt
    for old in OUT_V2.glob(f"{name}_FAIL.*"):
        old.unlink()
    paths = ps.save_figure(fig, OUT_V2 / name, formats=("pdf", "svg", "png"), expect_width_mm=width_mm)
    qa = C.qa_check(fig, paths["pdf"], caption, expect_width_mm=width_mm)
    qa.update(_svg_text_ok(Path(paths["svg"])))
    qa["png_dpi"] = _png_dpi(Path(paths["png"]))
    if not qa["svg_text_ok"]:
        qa["fails"].append("svg text not kept as text")
    if qa["png_dpi"] is not None and abs(qa["png_dpi"] - 600) > 1:
        qa["fails"].append(f"png dpi {qa['png_dpi']}")
    qa["ok"] = not qa["fails"]
    qa.update(name=name, version="v2", time=_dt.datetime.now().isoformat(timespec="seconds"))
    fig.savefig(C.QA_DIR / f"v2_{name}_100pct.png", dpi=300, facecolor="white")
    if not qa["ok"]:
        for k in ("pdf", "svg", "png"):
            p = Path(paths[k]); q = p.with_name(f"{name}_FAIL{p.suffix}")
            p.replace(q); paths[k] = str(q)
    qa["paths"] = paths
    src = []
    for k, v in (sources or {}).items():
        p = SRC_V2 / f"{name.split('_')[0]}_{k}.csv"
        v.to_csv(p, index=False)
        src.append(str(p.relative_to(ROOT)))
    qa["source_data"] = src
    (C.QA_DIR / f"v2_{name}_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1, default=C._jsonable))
    if qa["ok"] and register:
        write_caption_v2(name, caption, title)
        entry = dict(id=spec_id, version="v2", script=f"scripts/4_visualization/paper/{spec_extra.get('module', '')}" if spec_extra else "",
                     data="scripts/4_visualization/paper/v2_data.py → data/processed/paper_figs",
                     exports=[str(Path(paths[k]).relative_to(ROOT)) for k in ("pdf", "svg", "png")], source_data=src,
                     layout=dict(figsize_mm=[qa["width_mm"], qa["height_mm"]]), units="cm", medium="paper (Sci Rep 2-column 180 mm)",
                     style="polar.paperstyle + v2_style(범주 n 축, 4분 판정 기호 모양·채움)", generated=_dt.date.today().isoformat(), qa="pass",
                     qa_json=str((C.QA_DIR / f"v2_{name}_qa.json").relative_to(ROOT)))
        entry.update({k: v for k, v in (spec_extra or {}).items() if k != "module"})
        update_spec_registry(entry)
    plt.close(fig)
    status = "ok" if qa["ok"] else "FAIL " + "; ".join(qa["fails"][:4])
    print(f"[v2] {name}: {qa['width_mm']}×{qa['height_mm']} mm, min {qa['min_font_pt']} pt, {status}")
    return qa


def pdf_text(path) -> str:
    try:
        return subprocess.run(["pdftotext", "-q", str(path), "-"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


# ---------------------------------------------------------------- 2부: 초록 주 대비(AB) 표지
AB_RULE_TEXT = {"a": "direction holds after Holm correction", "b": "significant before correction only",
                "c": "equivalent (±0.5 cm) after correction", "d": "no difference established"}


def ab_tag(ax, x, y, code: str, dx_pt: float = 0.0, dy_pt: float = -7.2, ha: str = "center", va: str = "center", transform=None,
           zorder: float = 6):
    """AB 번호 글자(6.5 pt, 짙은 회색). 기호 아래(dy_pt < 0) 또는 옆에 둔다. 판정은 기호가 나타내고 이 글자는 대비 이름만 준다."""
    xy_tr = transform or ax.transData
    t = ax.annotate(code, xy=(x, y), xycoords=xy_tr, xytext=(dx_pt, dy_pt), textcoords="offset points", ha=ha, va=va,
                    fontsize=ps.FS["annot"], color=ps.GREY["text2"], annotation_clip=False, zorder=zorder)
    t.set_gid("ab_tag")
    return t


def ab_note(ab_cells: pd.DataFrame, codes) -> str:
    """AB 표지 설명 한 줄(lgw_bundle abstract_rule 코드 → 영문). 같은 규칙끼리 묶는다."""
    q = ab_cells.drop_duplicates("ab").set_index("ab")
    groups: dict = {}
    for c in sorted(codes, key=lambda x: int(str(x)[2:])):
        groups.setdefault(q.loc[c, "rule"], []).append(c)
    parts = []
    for rule in ("b", "c", "a", "d"):
        if rule in groups:
            parts.append(f"{', '.join(groups[rule])}: {AB_RULE_TEXT[rule]}")
    return "; ".join(parts)
