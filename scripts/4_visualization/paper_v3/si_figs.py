"""Supplementary Figs S7–S15 (v4 양식). 원고 SI 의 계획 캡션(paper/manuscript/en/sections/si_figs.tex)을 따른다.

그림과 자료(모든 수치는 등록 기록에서 읽는다. 새 적합은 없다)
  S7  검증 사다리(알래스카, M1 채점 방식 비교): 분할 지도 4개(seed 0), 채점 방식별 거리 분포 짝, 채점 방식별 RMSE(seed 3개 평균).
      data/processed/m1/cv_scheme_comparison.csv, cv_scheme_distances.csv, cv_scheme_comparison_meta.json.
      분할은 scripts/3_deep_learning/m1_cv_scheme_comparison.py 의 배정 함수로 다시 만들고, 저장된 거리(seed 0)와 같은지 확인한다.
  S8  라벨 0 전이: 대상별 지도(저가중 잔차, 앵커 + 잔차; WF0 재분석), 학습기 × 지역 세부(LGX L30·L11, LGT L34, LGF-F1·F5, LGF-N1, AB1).
  S9  계수 재보정: 대상별 지도(원천 계수 오차 ln(E_own/E0), 라벨 10개 재보정 이득), 토양 보정 Stefan 곡선(LGX P1@ed), 지역 대비(AB4–AB7).
  S10 방법 선택: 대상별 지도(규칙 W 가 가장 자주 고른 방법, W − 고정 레시피), LGF-N2 학습기 × 대비 격자, LGF-N4 산점도.
  S11 지역 내 이득: 채점 블록별 오차 변화 지도(레나델타, 캐나다; WF9), XE·XI·XB 봉인 표.
  S12 배치: 대상별 지도(블록 층화 − 무작위, 공변량 분산 − 무작위; XD 봉인 표), L23·L24(LGX), XD 대비.
  S14 예측 구간: 레나델타 1 km 지도(라벨 0 의 예측 ALT, 90 % 구간 폭, 원천 범위 밖 공변량 수; h49 산출 lena_pred_v1).
  S15 배포 시나리오: 시나리오 행렬, 순차 멈춤 규칙 곡선, 초록 대비 묶음(옛 v2 Fig 7 b–d 의 자료 표 data/processed/paper_figs/fig7_*).
  S13 은 그리지 않는다: 독립 지역(러시아 중부, 티베트 고원)의 격자 예측 산출이 없고, XF 새 지역 결과는 계획 8절에 기록되지 않았다
      (적격은 북대서양 v5 민감도 판뿐). 두 지역의 오차 변화는 Fig 6a·b 와 S8–S10·S12 지도에 이미 있다.

양식: design/style_tokens_v4.json(style.py). FreeSans 7 pt, 패널 문자 8 pt 굵게, 화살촉·축 밖 삼각형·글씨 상자·그림 제목 없음.
실행: OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v3/si_figs.py [--only S7,S8,...]
산출: outputs/figures/paper/v3_restructure/si/FigS<n>.{pdf,png}, FigS<n>_source_data.csv, FigS<n>_qa.json (설명문 FigS<n>_legend.md 는 따로 쓴다)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402

ROOT = S.ROOT
for _p in (ROOT / "src", ROOT / "scripts" / "2_evaluation", ROOT / "scripts" / "3_deep_learning"):
    sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

OUT = S.OUT / "si"
PROC = ROOT / "data" / "processed"
PF = S.PAPER_FIGS
W_MM = S.W2_MM                       # 170 mm
FS = S.FONT_PT                       # 7 pt
INK, INK2 = S.INK, S.INK_AUX
MINUS = S.MINUS
C = {k: v["color"] for k, v in S.METHOD.items()}
SOIL_LS = S.METHOD["soil_adjusted_stefan"]["ls"]
REGION_KEY = {"Lena": "Lena Delta", "Canada": "Canada", "Russia_W": "W Russia", "Russia_E": "E Russia", "Alaska": "Alaska",
              "Russia_C": "Central Russia", "Russia_C~lgd": "Central Russia", "Tibet_LGD": "Tibetan Plateau"}
REG4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
SOURCES: dict[str, list[str]] = {}   # 그림별 원천 파일(설명문·QA 기록용)
NODATA_TARGET = "#A0A0A0"            # 값이 없는 대상 원(지도 바탕 회색과 0 근처 흰색 모두와 구별)


# ================================================================ 공용 도우미
def fig_new(h_mm: float, w_mm: float = W_MM):
    """SI 그림(최종 크기). 본문 상한(180 mm)보다 높은 SI 그림은 A4 쪽 안(240 mm 이하)에서 허용한다."""
    if h_mm > 240.0:
        raise ValueError(h_mm)
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
    fig._mm = (w_mm, h_mm)
    return fig


def ax_mm(fig, x, y, w, h, **kw):
    return S.axes_mm(fig, x, y, w, h, **kw)


def text_mm(fig, x, y, s, **kw):
    W, H = fig._mm
    kw.setdefault("fontsize", FS)
    return fig.text(x / W, 1 - y / H, s, **kw)


def letter(fig, x, y, k):
    return S.panel_letter(fig, x, y, k)


def clean(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for sp in ax.spines.values():
        sp.set_linewidth(S.LW["axis"])
    ax.tick_params(which="both", width=S.LW["tick"], length=S.TICK_LEN_PT, pad=1.5)
    if not left:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if not bottom:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def zero(ax, axis="x", v=0.0):
    f = ax.axvline if axis == "x" else ax.axhline
    return f(v, color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], ls="-", zorder=1.5)


def band(ax, axis="x"):
    f = ax.axvspan if axis == "x" else ax.axhspan
    h = S.EQUIV_HALF_WIDTH_CM
    return f(-h, h, facecolor=S.EQUIV_BAND, edgecolor="none", lw=0, zorder=0.5)


def num(v, nd=2, plus=True):
    s = f"{float(v):+.{nd}f}" if plus else f"{float(v):.{nd}f}"
    return s.replace("-", MINUS)


def tick_fmt(nd=0):
    return FuncFormatter(lambda v, p: S.fmt_num(v, nd) if abs(v) > 1e-12 else "0")


def offset_tr(ax, dx_mm=0.0, dy_mm=0.0):
    return mtrans.offset_copy(ax.transData, fig=ax.figure, x=dx_mm / 25.4 * 72, y=dy_mm / 25.4 * 72, units="points")


def ci_h(ax, y, r, color, xlim=None, ms=None, z=3.0, mfc=None, block=True):
    """가로 점추정 + 셀 가중 95 % CI(1.6 pt) + 블록 등가중 95 % CI(0.8 pt, 0.8 mm 아래).
    r = dict(d, lo, hi, db, lob, hib). 축 밖 점추정은 축 끝에 값을 적는다(v4 토큰 symbols.out_of_range)."""
    ms = ms or S.MS["main"]
    xlim = xlim or ax.get_xlim()
    lo_x, hi_x = xlim
    d, lo, hi = (float(r.get(k, np.nan)) for k in ("d", "lo", "hi"))
    if not np.isfinite(d):
        return None
    clip = lambda a, b: (max(a, lo_x), min(b, hi_x))  # noqa: E731
    if np.isfinite(lo) and np.isfinite(hi):
        a, b = clip(lo, hi)
        if a < b:
            ax.plot([a, b], [y, y], color=color, lw=S.LW["ci_forest_cell"], solid_capstyle="butt", zorder=z, clip_on=False)
    lob, hib = float(r.get("lob", np.nan)), float(r.get("hib", np.nan))
    if block and np.isfinite(lob) and np.isfinite(hib):
        a, b = clip(lob, hib)
        if a < b:
            ax.plot([a, b], [y, y], color=color, lw=S.LW["ci_forest_block"], solid_capstyle="butt", zorder=z,
                    transform=offset_tr(ax, 0.0, -S.FOREST_BLOCK_OFFSET_MM), clip_on=False)
    if lo_x <= d <= hi_x:
        ax.plot([d], [y], ls="none", marker="o", ms=ms, mfc=mfc or color, mec=color, mew=S.LW["marker_edge_open"] if mfc else 0,
                zorder=z + 0.1, clip_on=False)
        return None
    side = hi_x if d > hi_x else lo_x
    t = ax.text(side, y, num(d, 1), ha="right" if d > hi_x else "left", va="center", color=color, fontsize=FS, zorder=z + 0.2,
                transform=offset_tr(ax, -0.4 if d > hi_x else 0.4, 0.0))
    t.set_gid("offaxis")
    return t


def ci_v(ax, x, r, color, ms=None, z=3.0, dx_mm=0.0):
    """세로 점추정 + 두 CI(블록 등가중은 0.8 mm 오른쪽)."""
    ms = ms or S.MS["main"]
    d, lo, hi = (float(r.get(k, np.nan)) for k in ("d", "lo", "hi"))
    if np.isfinite(lo):
        ax.plot([x, x], [lo, hi], color=color, lw=S.LW["ci_forest_cell"], solid_capstyle="butt", zorder=z, transform=offset_tr(ax, dx_mm, 0))
    lob, hib = float(r.get("lob", np.nan)), float(r.get("hib", np.nan))
    if np.isfinite(lob):
        ax.plot([x, x], [lob, hib], color=color, lw=S.LW["ci_forest_block"], solid_capstyle="butt", zorder=z,
                transform=offset_tr(ax, dx_mm + S.FOREST_BLOCK_OFFSET_MM, 0))
    ax.plot([x], [d], ls="none", marker="o", ms=ms, mfc=color, mec="none", zorder=z + 0.1, transform=offset_tr(ax, dx_mm, 0))


def rec(row, d="delta", lo="ci_lo", hi="ci_hi", db="delta_blockeq", lob="ci_lo_beq", hib="ci_hi_beq") -> dict:
    g = lambda k: float(row[k]) if (k in row and pd.notna(row[k])) else np.nan  # noqa: E731
    return dict(d=g(d), lo=g(lo), hi=g(hi), db=g(db), lob=g(lob), hib=g(hib))


def one(df, **flt):
    q = df
    for k, v in flt.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise SystemExit(f"한 행이 아니다({len(q)}): {flt}")
    return q.iloc[0]


def key_ci(fig, x, y, color=INK):
    """두 CI 열쇠(셀 가중 굵은 선, 블록 등가중 가는 선)와 ±0.5 cm 띠 견본. 반환 끝 x(mm)."""
    W, H = fig._mm
    ov = fig._ov
    ov.plot([x, x + 4.0], [y, y], color=color, lw=S.LW["ci_forest_cell"], solid_capstyle="butt")
    t1 = ov.text(x + 5.0, y, "Cell-weighted", ha="left", va="center", fontsize=FS)
    x2 = x + 5.0 + 1.32 * len("Cell-weighted") + 3.0
    ov.plot([x2, x2 + 4.0], [y, y], color=color, lw=S.LW["ci_forest_block"], solid_capstyle="butt")
    ov.text(x2 + 5.0, y, "Block-equal", ha="left", va="center", fontsize=FS)
    x3 = x2 + 5.0 + 1.32 * len("Block-equal") + 3.0
    ov.add_patch(Rectangle((x3, y - 1.0), 4.0, 2.0, facecolor=S.EQUIV_BAND, edgecolor="none", gid="key_swatch"))
    ov.text(x3 + 5.0, y, f"±0.5 cm", ha="left", va="center", fontsize=FS, gid="key")
    return t1


def overlay(fig):
    W, H = fig._mm
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.set_axis_off()
    ov.set_xlim(0, W)
    ov.set_ylim(H, 0)
    ov.set_zorder(20)
    fig._ov = ov
    return ov


def drawn_texts(fig) -> list:
    """실제로 그려지는 글자만: 그림 글자, 축 글자, 보이는 축의 보기 범위 안 눈금 라벨과 축 이름(축을 끈 축, 숨긴 축 제외)."""
    fig.canvas.draw()
    out = list(fig.texts)
    for a in fig.axes:
        out += list(a.texts)
        if not a.axison:
            continue
        for axis, lim in ((a.xaxis, a.get_xlim()), (a.yaxis, a.get_ylim())):
            if not axis.get_visible():
                continue
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks():
                loc = tk.get_loc()
                if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9):
                    continue
                for lab in (tk.label1, tk.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        out.append(lab)
            if axis.label.get_visible() and axis.label.get_text().strip():
                out.append(axis.label)
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def text_overlaps(fig) -> dict:
    import itertools
    rend = fig.canvas.get_renderer()
    ts = drawn_texts(fig)
    bbs = [(t.get_text().strip(), t.get_window_extent(rend)) for t in ts]
    pairs = [(a[:25], b[:25]) for (a, ba), (b, bb) in itertools.combinations(bbs, 2) if ba.overlaps(bb)]
    W, H = fig.get_size_inches() * fig.dpi
    outside = [s_[:25] for s_, b in bbs if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    return dict(n_texts=len(bbs), overlap_pairs=pairs, outside=outside)


def pdffonts(pdf: Path) -> dict:
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    rows = [ln.split() for ln in out.splitlines()[2:] if ln.strip()]
    names = sorted({r[0].split("+")[-1] for r in rows})
    return dict(fonts=names, type3=("Type 3" in out), liberation=("Liberation" in out),
                ok=(not ("Type 3" in out)) and (not ("Liberation" in out)) and all(n.startswith(("FreeSans", "Pretendard")) for n in names))


def finish(fig, stem: str, src_rows: list[dict], extra_qa: dict | None = None, max_chars=1600,
           allowed=("scale", "sizekey", "offaxis", "key", "value", "graticule", "category", "colkey")):
    """저장(PDF, 600 dpi PNG), 점검(audit_v3, 글자 겹침, pdffonts), 원천 표 저장."""
    OUT.mkdir(parents=True, exist_ok=True)
    au = S.audit_v3(fig, max_chars=max_chars, max_words=12, allowed_num_gids=allowed)
    ov = text_overlaps(fig)
    paths = S.save_fig(fig, stem, out_dir=OUT, formats=("pdf", "png"))
    pf = pdffonts(paths[0])
    pd.DataFrame(src_rows).to_csv(OUT / f"{stem}_source_data.csv", index=False)
    qa = dict(stem=stem, created=time.strftime("%Y-%m-%d %H:%M:%S"), size_mm=list(fig._mm), audit_fails=au["fails"],
              sizes=sorted(au["sizes"]), chars=au["chars"], thin_lines=au["thin_lines"][:10], codes=au["codes"][:10],
              loose_numbers=au["loose_numbers"][:20], text_overlap_pairs=ov["overlap_pairs"][:20], text_outside=ov["outside"][:10],
              pdffonts=pf, pdf_bytes=paths[0].stat().st_size, sources=SOURCES.get(stem, []), **(extra_qa or {}))
    (OUT / f"{stem}_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    plt.close(fig)
    bad = [k for k in au["fails"] if k not in ("label",)]
    print(f"[{stem}] fails={au['fails']} sizes={sorted(au['sizes'])} chars={au['chars']} overlaps={len(ov['overlap_pairs'])} "
          f"outside={ov['outside']} fonts={pf['fonts']} type3={pf['type3']} pdf={paths[0].stat().st_size/1e6:.2f} MB", flush=True)
    if ov["overlap_pairs"]:
        print("   overlaps:", ov["overlap_pairs"][:12])
    if au["loose_numbers"]:
        print("   loose numbers:", au["loose_numbers"][:12])
    if au["codes"]:
        print("   codes:", au["codes"][:12])
    return qa, bad


def src(stem, *paths):
    SOURCES.setdefault(stem, [])
    for p in paths:
        s = str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
        if s not in SOURCES[stem]:
            SOURCES[stem].append(s)


# ================================================================ S15 배포 시나리오(옛 v2 Fig 7 b–d)
S15_REG = [("Lena", "Lena Delta"), ("Canada", "Canada"), ("Russia_W", "W Russia"), ("Russia_E", "E Russia"), ("Alaska", "Alaska")]
S15_ROWS = [("T0", "R0", "P0", "0 labels", "Source anchor + residual − source Stefan"),
            ("T3", "P1", "P0", "3 labels", "Recalibrated − source Stefan"),
            ("T3", "R1", "P0", "3 labels", "Anchor + residual ML − source Stefan"),
            ("T10", "P1", "P0", "10 labels", "Recalibrated − source Stefan"),
            ("T10", "R1", "P0", "10 labels", "Anchor + residual ML − source Stefan"),
            ("T40", "P1", "P0", "40 labels", "Recalibrated − source Stefan"),
            ("T40", "R1", "P0", "40 labels", "Anchor + residual ML − source Stefan"),
            ("T40", "R1", "P1", "40 labels", "Anchor + residual ML − recalibrated Stefan"),
            ("T160", "P1", "P0", "160 labels", "Recalibrated − source Stefan"),
            ("T160", "R1", "P0", "160 labels", "Anchor + residual ML − source Stefan"),
            ("T160", "R1", "P1", "160 labels", "Anchor + residual ML − recalibrated Stefan")]
S15_COL = {"R0": C["source_anchor_residual"], "P1": C["recalibrated_stefan"], "R1": C["anchor_residual"]}
VERDICT_KO = {"우세": "lower", "열세": "higher", "동등": "equivalent", "미결정": "undecided", "판정 불가": "nd"}
VERDICT_EN = {"superior": "lower", "inferior": "higher", "equivalent": "equivalent", "undecided": "undecided"}
AB_ROWS = [("AB1", "Direct ML − source Stefan, 0 labels", "direct_ml"),
           ("AB2", "Ensemble − source Stefan, 0 labels", "stefan_cci_anchor"),
           ("AB3", "Pseudo-labels − shuffled, 0 labels", "physics_pseudo"),
           ("AB4", "Recalibrated − source Stefan, 10 labels", "recalibrated_stefan"),
           ("AB5", "Anchor + residual − recalibrated, 10 labels", "anchor_residual"),
           ("AB6", "Pseudo-labels added to residual, 10 labels", "physics_pseudo"),
           ("AB7", "Residual − physics-input structure, 10 labels", "anchor_residual"),
           ("AB8", "Anchor + residual − source Stefan, all labels", "anchor_residual"),
           ("AB9", "Anchor + residual − recalibrated, all labels", "anchor_residual")]
AB_RULE_EN = {"a": "direction holds after Holm correction", "b": "significant before correction only",
              "c": "equivalent (±0.5 cm) after correction", "d": "no difference established"}


def verdict_marker(ax, x, y, v, color, ms=4.6, z=3.0):
    """판정 기호(삼각형을 쓰지 않는다): 오차 감소 = 채운 원, 오차 증가 = 채운 사각형, 동등 = 빈 원, 미결정 = 작은 빈 마름모."""
    kw = dict(ls="none", zorder=z, clip_on=False)
    if v == "lower":
        ax.plot([x], [y], marker="o", ms=ms, mfc=color, mec=color, mew=0.0, **kw)
    elif v == "higher":
        ax.plot([x], [y], marker="s", ms=ms - 0.4, mfc=color, mec=color, mew=0.0, **kw)
    elif v == "equivalent":
        ax.plot([x], [y], marker="o", ms=ms, mfc="white", mec=color, mew=S.LW["marker_edge_open"], **kw)
    elif v == "undecided":
        ax.plot([x], [y], marker="D", ms=ms - 1.6, mfc="white", mec=INK2, mew=S.LW["marker_edge_open"], **kw)
    else:
        ax.plot([x], [y], marker="x", ms=ms - 1.2, mfc="none", mec=INK2, mew=S.LW["marker_edge_open"], **kw)


def tw(sx: str) -> float:
    """7 pt FreeSans 글자 폭 근사(mm). 열쇠 간격용."""
    return 1.32 * len(sx)


def build_S15():
    stem = "FigS15"
    B = pd.read_csv(PF / "fig7_b.csv")
    SS = pd.read_csv(PF / "fig7_b_summary.csv")
    Cq = pd.read_csv(PF / "fig7_c.csv")
    AB = pd.read_csv(PF / "fig7_d.csv")
    R = pd.read_csv(PF / "fig7_d_regions.csv")
    A10 = pd.read_csv(PF / "fig7_d_ab10.csv").iloc[0]
    src(stem, PF / "fig7_b.csv", PF / "fig7_b_summary.csv", PF / "fig7_c.csv", PF / "fig7_d.csv", PF / "fig7_d_regions.csv",
        PF / "fig7_d_ab10.csv", PF / "v2_data_meta.json")
    H = 166.0
    fig = fig_new(H)
    ov = overlay(fig)
    kax = ov
    rows = []

    # ---------------------------------------------------------------- a 시나리오 행렬(행 = 단계·대비, 열 = 지역 + 4지역 평균)
    letter(fig, 0.3, 0.3, "a")
    stage_w, name_w, col_w, row_h, head_h = 13.0, 55.0, 10.6, 4.1, 10.5
    top = head_h + 1.0
    ncol = len(S15_REG) + 1
    axa = ax_mm(fig, name_w, top, col_w * ncol, row_h * len(S15_ROWS))
    axa.set_xlim(-0.5, ncol - 0.5)
    axa.set_ylim(len(S15_ROWS) - 0.5, -0.5)
    axa.set_axis_off()
    heads = ["Lena\nDelta", "Canada", "W\nRussia", "E\nRussia", "Alaska", "Four-\nregion\nmean"]
    for j, hd in enumerate(heads):
        ov.text(name_w + col_w * (j + 0.5), head_h, hd, ha="center", va="bottom", fontsize=FS, linespacing=1.0).set_gid("category")
    short = {"Source anchor + residual − source Stefan": "Source anchor + residual − source",
             "Recalibrated − source Stefan": "Recalibrated − source",
             "Anchor + residual ML − source Stefan": "Anchor + residual − source",
             "Anchor + residual ML − recalibrated Stefan": "Anchor + residual − recalibrated"}
    prev = None
    for i, (st, rc, rf, stage_name, cname) in enumerate(S15_ROWS):
        yy = top + row_h * (i + 0.5)
        if stage_name != prev:
            ov.text(0.3, yy, stage_name, ha="left", va="center", fontsize=FS, color=INK2).set_gid("category")
            if prev is not None:
                ov.plot([0.3, name_w + col_w * ncol], [top + row_h * i] * 2, color=S.BASEMAP["graticule"], lw=S.LW["axis"], zorder=0.2)
            prev = stage_name
        col = S15_COL[rc]
        ov.text(name_w - 1.0, yy, short[cname], ha="right", va="center", fontsize=FS, color=col).set_gid("category")
        for j, (rk, rn) in enumerate(S15_REG):
            q = B[(B.stage == st) & (B.recipe == rc) & (B.reference == rf) & (B.region == rk)]
            if not len(q):
                axa.add_patch(Rectangle((j - 0.42, i - 0.40), 0.84, 0.80, facecolor="none", edgecolor=S.BASEMAP["hatch"], hatch="////",
                                        lw=0, zorder=0.5))
                rows.append(dict(panel="a", stage=st, contrast=f"{rc}-{rf}", region=rk, verdict="more labels than available"))
                continue
            r = q.iloc[0]
            v = VERDICT_EN.get(r.verdict, VERDICT_KO.get(r.verdict4, "nd"))
            verdict_marker(axa, j, i, v, col)
            if str(r.noninf).lower() == "true":
                axa.add_patch(Rectangle((j - 0.36, i - 0.40), 0.72, 0.80, facecolor="none", edgecolor=INK, lw=S.LW["axis"], zorder=4,
                                        gid="allowed_frame"))
            rows.append(dict(panel="a", stage=st, contrast=f"{rc}-{rf}", region=rk, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             delta_blockeq=r.delta_blockeq, ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq, verdict=v,
                             noninf=bool(str(r.noninf).lower() == "true"), source="data/processed/paper_figs/fig7_b.csv"))
        s_ = one(SS, stage=st, recipe=rc, reference=rf)
        v = VERDICT_EN.get(s_.verdict, VERDICT_KO.get(s_.main4_verdict4, "nd"))
        j = len(S15_REG)
        verdict_marker(axa, j, i, v, col)
        if str(s_.main4_noninf).lower() == "true":
            axa.add_patch(Rectangle((j - 0.36, i - 0.40), 0.72, 0.80, facecolor="none", edgecolor=INK, lw=S.LW["axis"], zorder=4,
                                    gid="allowed_frame"))
        pool2 = str(s_.main4_pool).startswith("부분")
        if pool2:
            ov.text(name_w + col_w * ncol + 0.2, yy, "†", ha="left", va="center", fontsize=FS, color=INK2)
        rows.append(dict(panel="a", stage=st, contrast=f"{rc}-{rf}", region="MEAN4", delta=s_.main4_delta, ci_lo=s_.main4_ci_lo,
                         ci_hi=s_.main4_ci_hi, verdict=v, noninf=bool(str(s_.main4_noninf).lower() == "true"),
                         pool=("Lena Delta and Canada" if pool2 else "four regions"), source="data/processed/paper_figs/fig7_b_summary.csv"))
    xv = name_w + col_w * len(S15_REG)
    ov.plot([xv, xv], [top, top + row_h * len(S15_ROWS)], color=S.BASEMAP["graticule"], lw=S.LW["axis"], zorder=0.2)
    # 기호 열쇠(행렬 아래 두 줄)
    yk = top + row_h * len(S15_ROWS) + 4.0
    kx = 1.8
    for lab, v in (("Lower error", "lower"), ("Equivalent", "equivalent"), ("Undecided", "undecided"), ("Higher error", "higher")):
        verdict_marker(kax, kx, yk, v, INK if v != "undecided" else INK2)
        kax.text(kx + 1.8, yk, lab, ha="left", va="center", fontsize=FS).set_gid("key")
        kx += 1.8 + tw(lab) + 3.2
    yk2 = yk + 4.4
    kx = 1.8
    kax.add_patch(Rectangle((kx - 1.2, yk2 - 1.4), 2.4, 2.8, facecolor="none", edgecolor=INK, lw=S.LW["axis"], gid="allowed_frame"))
    kax.text(kx + 1.8, yk2, "Non-inferior (margin 0.5 cm)", ha="left", va="center", fontsize=FS).set_gid("key")
    kx += 1.8 + tw("Non-inferior (margin 0.5 cm)") + 3.2
    kax.add_patch(Rectangle((kx - 1.2, yk2 - 1.2), 2.4, 2.4, facecolor="none", edgecolor=S.BASEMAP["hatch"], hatch="////", lw=0))
    kax.text(kx + 1.8, yk2, "Fewer labels available", ha="left", va="center", fontsize=FS).set_gid("key")
    kx += 1.8 + tw("Fewer labels available") + 3.2
    kax.text(kx - 0.6, yk2, "†", ha="left", va="center", fontsize=FS, color=INK2)
    kax.text(kx + 1.2, yk2, "Lena Delta and Canada", ha="left", va="center", fontsize=FS).set_gid("key")

    # ---------------------------------------------------------------- b 순차 멈춤 규칙(SC3w): 재보정 Stefan, 멈춤 − 항상 40개
    xb, yb, wb, hb = 133.0, 9.0, 35.0, 40.0
    letter(fig, 121.5, 0.3, "b")
    axb = ax_mm(fig, xb, yb, wb, hb)
    clean(axb)
    colb = C["recalibrated_stefan"]
    m = Cq[Cq.scope == "MEAN3"].sort_values("tau")
    for r in Cq[Cq.scope == "region"].itertuples():
        reg = r.region
        axb.plot([r.label_ratio], [r.delta], ls="none", marker=S.REGION_MARKER.get(reg, "o"), ms=S.MS["region_point"],
                 mfc=S.REGION_COLOR[REGION_KEY[reg]], mec="none", alpha=0.8, zorder=2.5)
        rows.append(dict(panel="b", scope="region", region=reg, tau=r.tau, label_ratio=r.label_ratio, delta=r.delta,
                         source="data/processed/paper_figs/fig7_c.csv"))
    axb.plot(m.label_ratio, m.delta, color=colb, lw=S.LW["aux"], zorder=2.6)
    for r in m.itertuples():
        axb.plot([r.label_ratio, r.label_ratio], [r.ci_lo, r.ci_hi], color=colb, lw=S.LW["ci_forest_cell"], solid_capstyle="butt", zorder=2.7)
        axb.plot([r.label_ratio], [r.delta], ls="none", marker="o", ms=S.MS["main"], mfc=colb, mec="none", zorder=3)
        rows.append(dict(panel="b", scope="MEAN3", tau=r.tau, label_ratio=r.label_ratio, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                         delta_blockeq=r.delta_blockeq, ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq, verdict4=r.verdict4,
                         source="data/processed/paper_figs/fig7_c.csv"))
    off = {0.025: (0.0, 0.6, "right", "bottom"), 0.05: (-1.0, -1.4, "right", "top"), 0.1: (-1.0, -1.4, "right", "top"),
           0.2: (1.2, 0.0, "left", "center")}
    for r in m.itertuples():
        dx, dy, ha, va = off[round(float(r.tau), 3)]
        axb.text(r.label_ratio, r.delta, f"τ = {r.tau:g}", ha=ha, va=va, fontsize=FS, color=colb,
                 transform=offset_tr(axb, dx, dy)).set_gid("value")
    for yv, ls, va in ((0.3, (0, (1.0, 1.4)), "top"), (0.5, (0, (3.0, 1.6)), "bottom")):
        axb.axhline(yv, color=S.REF_COLOR, lw=S.LW["ref"], ls=ls, zorder=1)
        axb.text(0.02, yv, f"{yv:g} cm", ha="left", va=va, fontsize=FS, color=INK2, gid="value",
                 transform=mtrans.blended_transform_factory(axb.transAxes, axb.transData))
    axb.axvline(0.5, color=S.REF_COLOR, lw=S.LW["ref"], ls=(0, (1.0, 1.4)), zorder=1)
    zero(axb, "y")
    axb.set_xlim(0, 1.06)
    axb.xaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
    axb.xaxis.set_major_formatter(FuncFormatter(lambda v, p: {0: "0", 0.5: "0.5", 1.0: "1"}[round(v, 2)]))
    axb.set_ylim(-2.4, 0.9)
    axb.yaxis.set_major_locator(FixedLocator([-2, -1, 0]))
    axb.yaxis.set_major_formatter(tick_fmt(0))
    axb.set_xlabel("Label ratio (labels used / 40)")
    axb.set_ylabel("Error change vs 40 labels (cm)")
    kyy = yb + hb + 9.0
    kxx = xb - 9.0
    for reg in ("Lena", "Canada", "Alaska"):
        kax.plot([kxx], [kyy], ls="none", marker=S.REGION_MARKER[reg], ms=S.MS["region_point"], mfc=S.REGION_COLOR[REGION_KEY[reg]],
                 mec="none", alpha=0.8)
        kax.text(kxx + 1.4, kyy, REGION_KEY[reg], ha="left", va="center", fontsize=FS).set_gid("key")
        kxx += 1.4 + tw(REGION_KEY[reg]) + 2.4
    kax.plot([xb - 10.0, xb - 6.0], [kyy + 4.2] * 2, color=colb, lw=S.LW["ci_forest_cell"])
    kax.text(xb - 5.0, kyy + 4.2, "Three-region mean", ha="left", va="center", fontsize=FS).set_gid("key")

    # ---------------------------------------------------------------- c 초록 대비 묶음(AB1–AB9) + AB10
    yc = 72.0
    letter(fig, 0.3, yc + 0.3, "c")
    name_w2, wc, rh = 66.0, 50.0, 5.2
    y0c = yc + 6.0
    axc = ax_mm(fig, name_w2, y0c, wc, rh * len(AB_ROWS))
    clean(axc, left=False)
    axc.set_xscale("symlog", linthresh=2.0, linscale=0.7)
    xl = (-18.0, 7.0)
    axc.set_xlim(*xl)
    axc.set_ylim(len(AB_ROWS) - 0.5, -0.6)
    band(axc, "x")
    zero(axc, "x")
    axc.xaxis.set_major_locator(FixedLocator([-15, -5, -2, -1, 0, 1, 2, 5]))
    axc.xaxis.set_major_formatter(FuncFormatter(lambda v, p: S.fmt_int(v) if abs(v) > 1e-9 else "0"))
    axc.xaxis.set_minor_locator(NullLocator())
    axc.set_xlabel("Error change, four-region mean (cm; log scale beyond ±2 cm)")
    xw = name_w2 + wc + 2.5
    for i, (ab, lab, mk) in enumerate(AB_ROWS):
        r = one(AB, ab=ab)
        col = C[mk]
        for g in R[R.ab == ab].itertuples():
            axc.plot([g.delta], [i + 0.30], ls="none", marker=S.REGION_MARKER.get(g.region, "o"), ms=S.MS["region_point"] - 0.6,
                     mfc=S.REGION_COLOR[REGION_KEY[g.region]], mec="none", alpha=0.8, zorder=2.2)
            rows.append(dict(panel="c", ab=ab, region=g.region, delta=g.delta, ci_lo=g.ci_lo, ci_hi=g.ci_hi, verdict4=g.verdict4,
                             source="data/processed/paper_figs/fig7_d_regions.csv"))
        ci_h(axc, i, rec(r, db="delta_beq"), col, xlim=xl)
        ov.text(name_w2 - 1.5, y0c + rh * (i + 0.5), lab, ha="right", va="center", fontsize=FS, color=col).set_gid("category")
        ov.text(xw, y0c + rh * (i + 0.5), AB_RULE_EN[str(r.rule)], ha="left", va="center", fontsize=FS).set_gid("category")
        rows.append(dict(panel="c", ab=ab, region="MEAN4", delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_blockeq=r.delta_beq,
                         ci_lo_beq=r.ci_lo_beq, ci_hi_beq=r.ci_hi_beq, verdict4=r.verdict4, holm_p=r.holm_p, rule=r.rule,
                         source="data/processed/paper_figs/fig7_d.csv"))
    ov.text(xw, y0c - 2.2, "Wording after Holm correction", ha="left", va="center", fontsize=FS, color=INK2).set_gid("category")
    y10 = y0c + rh * len(AB_ROWS) + 13.0
    ax10 = ax_mm(fig, name_w2, y10, wc, rh)
    clean(ax10, left=False)
    ax10.set_xlim(-5, 25)
    ax10.set_ylim(0.6, -0.6)
    ax10.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20, 25]))
    ax10.xaxis.set_major_formatter(tick_fmt(0))
    zero(ax10, "x")
    col10 = S.INTERVALS["Conformal"]
    ci_h(ax10, 0, dict(d=A10.delta_cm, lo=A10.ci_lo_cm, hi=A10.ci_hi_cm), col10, xlim=(-5, 25), block=False)
    ax10.set_xlabel("Interval score change, three regions (cm)")
    ov.text(name_w2 - 1.5, y10 + rh / 2, "Hierarchical − constant width, 10 labels", ha="right", va="center", fontsize=FS,
            color=col10).set_gid("category")
    ov.text(xw, y10 + rh / 2, AB_RULE_EN["d"], ha="left", va="center", fontsize=FS).set_gid("category")
    rows.append(dict(panel="c", ab="AB10", delta=A10.delta_cm, ci_lo=A10.ci_lo_cm, ci_hi=A10.ci_hi_cm, verdict4=A10.verdict4_lgu,
                     source="data/processed/paper_figs/fig7_d_ab10.csv"))
    yk3 = y10 + rh + 12.0
    key_ci(fig, 0.8, yk3)
    kxx = 66.0
    for reg in REG4:
        kax.plot([kxx], [yk3], ls="none", marker=S.REGION_MARKER[reg], ms=S.MS["region_point"] - 0.6, mfc=S.REGION_COLOR[REGION_KEY[reg]],
                 mec="none", alpha=0.8)
        kax.text(kxx + 1.4, yk3, REGION_KEY[reg], ha="left", va="center", fontsize=FS).set_gid("key")
        kxx += 1.4 + tw(REGION_KEY[reg]) + 2.4
    return finish(fig, stem, rows)


# ================================================================ S14 레나델타 예측 구간 지도(h49)
def build_S14():
    import cartopy.crs as ccrs  # noqa: F401
    import maps_alt_v3 as M3
    from cmcrameri import cm as cmc
    stem = "FigS14"
    M3.use_medium("paper")
    Md = M3.Med("paper")
    p = pd.read_csv(PROC / "map_lena" / "lena_pred_v1.csv.gz", dtype={"cell_id": str})
    meta = json.loads((PROC / "map_lena" / "lena_pred_v1_meta.json").read_text())
    src(stem, PROC / "map_lena" / "lena_pred_v1.csv.gz", PROC / "map_lena" / "lena_pred_v1_meta.json")
    reason = p.gray_reason.fillna("").astype(str).values
    gray = p.gray.values == 1
    d = dict(key="lena", cells=p, proj=M3.projection("lena"), **M3.REG["lena"])
    d["shown"] = (~gray) & np.isfinite(p.pred_a.values)
    d["water"] = gray & (reason == "water")
    d["nodata"] = gray & ~np.isin(reason, ["water", "dem_tile_absent", "pfr_missing"]) & (p.land.values == 1)
    d["p1"] = p.pred_a.values.astype(float)
    d["r1"] = p.pred_a.values.astype(float)
    d["corr"] = np.zeros(len(p))
    M3.prepare(d)
    ext = M3.square_extent(d["ext"])
    sh = d["shown"]
    n_shown, n_land = int(sh.sum()), int((p.land.values == 1).sum())
    n_land_gray = int(((p.land.values == 1) & gray).sum())
    ss = meta.get("shown_stats", {})
    H = 86.0
    fig = fig_new(H)
    pw, gap, top = 50.0, 6.5, 5.0
    xs = [8.0, 8.0 + pw + gap, 8.0 + 2 * (pw + gap)]
    # 색 범위: ALT 는 표시 셀의 2–98 백분위를 5 cm 단위로, 폭은 같은 규칙, 외삽은 0, 1, 2, ≥3 의 순서형
    a_lo, a_hi = M3.pct_range(p.pred_a.values[sh], 2.0, 98.0, 5.0)
    w_lo, w_hi = M3.pct_range(p.width90_a.values[sh], 2.0, 98.0, 5.0)
    norm_a = matplotlib.colors.Normalize(a_lo, a_hi)
    norm_w = matplotlib.colors.Normalize(w_lo, w_hi)
    cm_w = M3.lut(cmc.acton_r, 0.10, 0.92)
    cat_cols = [cmc.devon_r(v) for v in (0.18, 0.40, 0.62, 0.86)]
    cm_c = ListedColormap(cat_cols)
    norm_c = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    cat = np.clip(p.extrap_cat.values.astype(float), 0, 3)
    panels = [("a", "Source Stefan, no target labels", p.pred_a.values, M3.CM_ALT, norm_a),
              ("b", "Width of 90% interval", p.width90_a.values, cm_w, norm_w),
              ("c", "Covariates outside source range", cat, cm_c, norm_c)]
    rows = []
    for k, (lt, head, vals, cmap, norm) in enumerate(panels):
        x = xs[k]
        ax = ax_mm(fig, x, top + 4.0, pw, pw, projection=d["proj"])
        M3.base_map(ax, d, ext, Md)
        M3.add_cells(ax, d["mesh"], M3.cell_rgba(d, vals, cmap, norm))
        M3.graticule(ax, [124, 126, 128], d["ylocs"], Md, left=(k == 0), bottom=True)
        letter(fig, x - 5.0 if k else 0.3, 0.3, lt)
        text_mm(fig, x + pw / 2, top + 1.0, head, ha="center", va="top").set_gid("category")
        if k == 0:
            M3.scale_in_corner(ax, d, ext, Md, pw, km=50)
        cax = ax_mm(fig, x + 4.0, top + 4.0 + pw + 8.5, pw - 8.0, 2.5)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        if lt == "c":
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1, 2, 3])
            cb.set_ticklabels(["0", "1", "2", "≥3"])
            M3.style_cbar(cb, "Covariates outside range (count)", Md)
        else:
            lo_, hi_ = norm.vmin, norm.vmax
            ext_kind = M3.extend_of(vals[sh], lo_, hi_)
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=ext_kind, extendfrac=0.04)
            cb.set_ticks(M3.ticks_between(lo_, hi_))
            cb.set_ticklabels([S.fmt_int(v) for v in M3.ticks_between(lo_, hi_)])
            M3.style_cbar(cb, "Active-layer thickness (cm)" if lt == "a" else "90% interval width (cm)", Md)
        cax.set_label("<colorbar>")
        v = np.asarray(vals, float)[sh]
        rows.append(dict(panel=lt, quantity=head, n_cells=int(np.isfinite(v).sum()), p2=float(np.percentile(v, 2)),
                         median=float(np.median(v)), p98=float(np.percentile(v, 98)), color_lo=float(norm.vmin), color_hi=float(norm.vmax),
                         source="data/processed/map_lena/lena_pred_v1.csv.gz"))
    for c_ in range(4):
        rows.append(dict(panel="c", quantity=f"extrap_cat={c_}{'+' if c_ == 3 else ''}", n_cells=int(((cat == c_) & sh).sum()),
                         source="data/processed/map_lena/lena_pred_v1.csv.gz"))
    # 회색 열쇠(지도 아래 한 줄)
    W, Hh = fig._mm
    ov = overlay(fig)
    yk = H - 2.5
    ov.add_patch(Rectangle((8.0, yk - 1.0), 3.0, 2.0, facecolor=S.BASEMAP["nodata"], edgecolor="none", gid="key_swatch"))
    ov.text(12.0, yk, "No data (outside the permafrost mask or missing input)", ha="left", va="center", fontsize=FS).set_gid("key")
    ov.add_patch(Rectangle((90.0, yk - 1.0), 3.0, 2.0, facecolor="white", edgecolor=S.BASEMAP["coast"], lw=S.LW["axis"], gid="key_swatch"))
    ov.text(94.0, yk, "Water", ha="left", va="center", fontsize=FS).set_gid("key")
    cal = meta["caption"]["interval"]
    extra = dict(n_shown=n_shown, n_cells=int(len(p)), n_land=n_land, n_land_gray=n_land_gray, shown_stats_meta=ss,
                 interval=dict(method=cal["method"], q90=cal["q90"], groups=cal["groups"]), extrap_top3=meta["caption"].get("extrap_top3"))
    rows.append(dict(panel="all", quantity="cells shown", n_cells=n_shown, total=int(len(p)), land=n_land, land_gray=n_land_gray,
                     q90=cal["q90"], source="data/processed/map_lena/lena_pred_v1_meta.json"))
    return finish(fig, stem, rows, extra_qa=extra)


# ================================================================ S7 검증 사다리(알래스카, M1 채점 방식 비교)
S7_SCHEMES = [("random_cell", "Random cell"), ("site_0.05", "0.05° site"), ("block_0.5", "0.5° block"), ("knndm", "kNNDM")]
S7_MODELS = [("catboost_lo", "Direct ML", "direct_ml"), ("stefan", "Stefan model", "recalibrated_stefan"),
             ("stefan_ridge_l075", "Stefan + residual", "anchor_residual")]
FOLD_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]   # Paul Tol bright(범주 6개)
CV_COLOR = "#0072B2"


def s7_folds():
    """M1 의 분할(seed 0)을 원 스크립트의 배정 함수로 다시 만든다. 반환 (알래스카 셀 표, {방식: fold}, 점검 기록)."""
    from sklearn.model_selection import KFold
    from polar.m1_core import load_base, eval_mask
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        import m1_cv_scheme_comparison as M1
    finally:
        sys.argv = argv
    df = load_base(PROC)
    ak = df[df.macro.values == "Alaska"].reset_index(drop=True)
    ak = ak[eval_mask(ak)].reset_index(drop=True)
    n = len(ak)
    rad = M1.to_rad(ak.lon.values, ak.lat.values)
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    px, py = tf.transform(ak.lon.values, ak.lat.values)
    XY = np.c_[px, py] / 1000.0
    seed, K = 0, M1.K_FOLDS
    rng = np.random.RandomState(seed)
    folds = {}
    f = np.empty(n, int)
    for fi, (_, te) in enumerate(KFold(K, shuffle=True, random_state=seed).split(np.arange(n))):
        f[te] = fi
    folds["random_cell"] = f
    folds["site_0.05"] = M1.group_folds_random(M1.grid_key(ak.lat.values, ak.lon.values, 0.05), K, rng, n)
    folds["block_0.5"] = M1.group_folds_random(ak.block.values, K, rng, n)
    meta = json.loads((PROC / "m1" / "cv_scheme_comparison_meta.json").read_text())
    kinfo = [k for k in meta["knndm"] if k["seed"] == seed][0]
    q0 = int(kinfo["q_selected"])
    Xc = XY - XY.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    lab = M1.kmeans_labels(XY, q0, seed)
    folds["knndm"] = lab if q0 == K else M1.merge_clusters(lab, XY, vt[0], K, 0.5)
    D = pd.read_csv(PROC / "m1" / "cv_scheme_distances.csv")
    chk = {}
    for sch, fo in folds.items():
        d = M1.cv_nnd(fo, rad)
        st = D[(D.scheme == sch) & (D.seed == seed)].dist_km.values
        chk[sch] = dict(n=int(len(st)), max_abs_diff_km=float(np.max(np.abs(np.round(d, 4) - st))),
                        fold_sizes=np.bincount(fo, minlength=K).tolist())
        if len(st) != n or chk[sch]["max_abs_diff_km"] > 1.5e-4:
            raise SystemExit(f"S7: 다시 만든 분할의 거리가 저장값과 다르다 {sch} {chk[sch]}")
    if chk["knndm"]["fold_sizes"] != kinfo["fold_sizes"]:
        raise SystemExit(f"S7: kNNDM fold 크기 {chk['knndm']['fold_sizes']} != {kinfo['fold_sizes']}")
    chk["knndm"]["q_selected"] = q0
    return ak, folds, D, chk


def build_S7():
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    stem = "FigS7"
    ak, folds, D, chk = s7_folds()
    Cm = pd.read_csv(PROC / "m1" / "cv_scheme_comparison.csv")
    src(stem, PROC / "m1" / "cv_scheme_comparison.csv", PROC / "m1" / "cv_scheme_distances.csv",
        PROC / "m1" / "cv_scheme_comparison_meta.json", ROOT / "scripts" / "3_deep_learning" / "m1_cv_scheme_comparison.py")
    H = 134.0
    fig = fig_new(H)
    ov = overlay(fig)
    rows = []
    proj = ccrs.epsg(3338)
    pc = ccrs.PlateCarree()
    xy = proj.transform_points(pc, ak.lon.values, ak.lat.values)[:, :2]
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    mw, mh, gap = 40.0, 36.0, 3.33
    padx = 0.06 * (x1 - x0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = (x1 - x0) / 2 + padx
    half_h = half_w * mh / mw
    if (y1 - y0) / 2 * 1.08 > half_h:
        half_h = (y1 - y0) / 2 * 1.08
        half_w = half_h * mw / mh
    ext = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
    land = list(shpreader.Reader(shpreader.natural_earth(resolution="50m", category="physical", name="land")).geometries())
    cmap_f = ListedColormap(FOLD_COLORS)
    for k, (sch, head) in enumerate(S7_SCHEMES):
        x = k * (mw + gap)
        ax = ax_mm(fig, x, 5.0, mw, mh, projection=proj)
        ax.set_extent(ext, crs=proj)
        ax.add_geometries(land, crs=pc, facecolor=S.BASEMAP["land"], edgecolor=S.BASEMAP["coast"], linewidth=S.LW["coast"], zorder=0)
        ax.spines["geo"].set_linewidth(S.LW["axis"])
        ax.spines["geo"].set_edgecolor(INK)
        order = np.random.default_rng(0).permutation(len(xy))          # 그리는 순서를 섞어 한 fold 가 다른 fold 를 덮지 않게
        ax.scatter(xy[order, 0], xy[order, 1], c=folds[sch][order], cmap=cmap_f, vmin=-0.5, vmax=5.5, s=2.2, linewidths=0, transform=proj,
                   zorder=3, rasterized=True)
        letter(fig, x + 0.3, 0.3, "abcd"[k])
        text_mm(fig, x + mw / 2, 0.6, head, ha="center", va="top").set_gid("category")
        sizes = np.bincount(folds[sch], minlength=6)
        rows.append(dict(panel="abcd"[k], scheme=sch, seed=0, fold_sizes=";".join(map(str, sizes)),
                         max_abs_diff_km_vs_stored=chk[sch]["max_abs_diff_km"], source="regenerated with m1_cv_scheme_comparison.py functions; "
                         "checked against data/processed/m1/cv_scheme_distances.csv"))
        if k == 0:
            # 축척 막대 200 km(EPSG:3338 은 등적 원추, 표준 위선 55°·65° N 에서 축척 1)
            sx, sy = ext[1] - 0.05 * (ext[1] - ext[0]) - 200e3, ext[3] - 0.30 * (ext[3] - ext[2])
            ax.plot([sx, sx + 200e3], [sy, sy], color=INK, lw=S.LW["scale_bar"], solid_capstyle="butt", transform=proj, zorder=6)
            ax.text(sx + 200e3, sy + 0.035 * (ext[3] - ext[2]), "200 km", ha="right", va="bottom", fontsize=FS, transform=proj,
                    zorder=6).set_gid("scale")
    # fold 열쇠
    yk = 5.0 + mh + 3.6
    ov.text(0.3, yk, "Fold", ha="left", va="center", fontsize=FS)
    for i, colr in enumerate(FOLD_COLORS):
        xk = 7.5 + i * 7.0
        ov.add_patch(Rectangle((xk, yk - 1.0), 2.4, 2.0, facecolor=colr, edgecolor="none", gid="key_swatch"))
        ov.text(xk + 3.0, yk, str(i + 1), ha="left", va="center", fontsize=FS).set_gid("key")
    ov.text(54.0, yk, "Labelled 1 km cells, seed 0 split", ha="left", va="center", fontsize=FS, color=INK2).set_gid("key")

    # e–h 거리 분포 짝
    bins = np.linspace(-3.0, 3.2, 63)
    pred = D[D.scheme == "prediction_permafrost"].dist_km.values
    lp = np.log10(np.clip(pred, 1e-3, None))
    hp, _ = np.histogram(lp, bins=bins)
    hp = hp / hp.sum()
    ymax = 0
    hists = {}
    for sch, _ in S7_SCHEMES:
        dv = D[(D.scheme == sch) & (D.seed == 0)].dist_km.values
        h, _ = np.histogram(np.log10(np.clip(dv, 1e-3, None)), bins=bins)
        hists[sch] = h / h.sum()
        ymax = max(ymax, hists[sch].max())
    ymax = max(ymax, hp.max())
    yt = 52.0
    dw, dh = 30.0, 22.0
    for k, (sch, head) in enumerate(S7_SCHEMES):
        x = k * (mw + gap) + 8.0
        ax = ax_mm(fig, x, yt, dw, dh)
        clean(ax)
        h = hists[sch]
        ax.fill_between(bins[:-1], h, step="post", color=CV_COLOR, alpha=0.30, lw=0, zorder=1)
        ax.step(bins[:-1], h, where="post", color=CV_COLOR, lw=S.LW["aux"], zorder=2)
        ax.step(bins[:-1], hp, where="post", color=S.REF_COLOR, lw=S.LW["main"], zorder=3)
        ax.set_xlim(-3.2, 3.4)
        ax.set_ylim(0, ymax * 1.08)
        ax.xaxis.set_major_locator(FixedLocator([-3, -1, 1, 3]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: {-3: "0.001", -1: "0.1", 1: "10", 3: "1000"}[int(round(v))]))
        yt_ = [0, 0.1, 0.2] if ymax > 0.15 else [0, 0.05, 0.1]
        ax.yaxis.set_major_locator(FixedLocator([v for v in yt_ if v <= ymax * 1.08]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: S.fmt_num(v, 1) if v else "0"))
        if k == 0:
            ax.set_ylabel("Share of cells")
        else:
            ax.tick_params(axis="y", labelleft=False)
        letter(fig, k * (mw + gap) + 0.3, yt - 3.2, "efgh"[k])
        text_mm(fig, x + dw / 2, yt - 3.0, head, ha="center", va="top").set_gid("category")
        dv = D[(D.scheme == sch)].copy()
        med = dv.groupby("seed").dist_km.median()
        rows.append(dict(panel="efgh"[k], scheme=sch, quantity="test to nearest training cell (km)", seed=0, n=int(len(dv[dv.seed == 0])),
                         median_seed0=float(med.loc[0]), median_mean_3seeds=float(med.mean()),
                         source="data/processed/m1/cv_scheme_distances.csv"))
    rows.append(dict(panel="efgh", scheme="prediction_permafrost", quantity="map cell to nearest label (km)", n=int(len(pred)),
                     median_seed0=float(np.median(pred)), source="data/processed/m1/cv_scheme_distances.csv"))
    text_mm(fig, W_MM / 2, yt + dh + 6.5, "Distance (km, log scale)", ha="center", va="center")
    yk2 = yt + dh + 11.0
    ov.add_patch(Rectangle((6.0, yk2 - 1.0), 4.0, 2.0, facecolor=CV_COLOR, alpha=0.30, edgecolor="none", gid="key_swatch"))
    ov.plot([6.0, 10.0], [yk2 - 1.0, yk2 - 1.0], color=CV_COLOR, lw=S.LW["aux"])
    ov.text(11.0, yk2, "Test cell to nearest training cell", ha="left", va="center", fontsize=FS).set_gid("key")
    ov.plot([62.0, 66.0], [yk2, yk2], color=S.REF_COLOR, lw=S.LW["main"])
    ov.text(67.0, yk2, "Permafrost map cell to nearest label", ha="left", va="center", fontsize=FS).set_gid("key")

    # i 채점 방식별 RMSE(seed 3개 평균)
    yi = yk2 + 8.0
    letter(fig, 0.3, yi - 2.0, "i")
    axi = ax_mm(fig, 14.0, yi, 80.0, H - yi - 9.0)
    clean(axi)
    xs = np.arange(len(S7_SCHEMES))
    G = Cm.groupby(["scheme", "model"]).rmse_cm.agg(["mean", "min", "max"]).reset_index()
    for mod, lab, mk in S7_MODELS:
        v = [float(G[(G.scheme == sch) & (G.model == mod)]["mean"].iloc[0]) for sch, _ in S7_SCHEMES]
        lo = [float(G[(G.scheme == sch) & (G.model == mod)]["min"].iloc[0]) for sch, _ in S7_SCHEMES]
        hi = [float(G[(G.scheme == sch) & (G.model == mod)]["max"].iloc[0]) for sch, _ in S7_SCHEMES]
        axi.plot(xs, v, color=C[mk], lw=S.LW["main"], marker="o", ms=S.MS["main"], mec="none", zorder=3)
        axi.text(xs[-1], v[-1], lab, ha="left", va="center", fontsize=FS, color=C[mk], transform=offset_tr(axi, 2.2, 0.0)).set_gid("category")
        for sch, a_, l_, h_ in zip([s_[0] for s_ in S7_SCHEMES], v, lo, hi):
            rows.append(dict(panel="i", scheme=sch, model=mod, rmse_mean_3seeds=a_, rmse_min_seed=l_, rmse_max_seed=h_,
                             source="data/processed/m1/cv_scheme_comparison.csv"))
    axi.set_xlim(-0.3, len(S7_SCHEMES) - 0.7)
    axi.xaxis.set_major_locator(FixedLocator(list(xs)))
    axi.xaxis.set_major_formatter(FuncFormatter(lambda v, p: [h for _, h in S7_SCHEMES][int(round(v))]))
    axi.set_ylim(10.5, 19.0)
    axi.yaxis.set_major_locator(FixedLocator([12, 14, 16, 18]))
    axi.yaxis.set_major_formatter(tick_fmt(0))
    axi.set_ylabel("RMSE (cm)")
    return finish(fig, stem, rows, extra_qa=dict(split_check=chk))


# ================================================================ 대상별 지도(Fig 6 a·b 와 같은 틀)
_F6 = None
_PFR = None


def f6():
    global _F6
    if _F6 is None:
        import fig6 as F6
        _F6 = F6
    return _F6


def map_targets() -> pd.DataFrame:
    """지도 대상 17개(독립 지역 7 + 하위 지역 10, 모드 x)의 이름과 라벨 셀 중심(Fig 6 과 같은 정의)."""
    F6 = f6()
    Cn = F6.target_centroids()
    keep = F6.MACRO7 + F6.SUB10
    Cn = Cn.set_index("name").loc[keep].reset_index()
    return Cn


def target_maps(fig, M, panels, norm, cmap, slots, cbar=None, cbar_label="", cbar_ticks=None, cbar_fmt=None, ext_kind="neither",
                pf_key=True, rows=None, value_name="value"):
    """대상별 지도 두 장(Fig 6 a·b 와 같은 투영, 원 배치, 삽도). M: name, lat, lon, 값 열. panels: [(문자, 열, 머리)].
    값이 없는 대상은 회색 채움(자료 없음)."""
    global _PFR
    F6 = f6()
    W, H = fig._mm
    F6.W_MM, F6.H_MM = W, H
    if _PFR is None:
        _PFR = F6.pfr_classes()
    P = _PFR
    ov = fig._ov
    cm_ = cmap.with_extremes(bad=NODATA_TARGET) if hasattr(cmap, "with_extremes") else cmap
    arc = M[~M.name.isin(F6.INSET_TARGETS)]
    main = arc[~arc.name.isin(F6.AK_TARGETS)].reset_index(drop=True)
    ak = M[M.name.isin(F6.AK_TARGETS)].set_index("name").loc[F6.AK_TARGETS].reset_index()
    rho = abs(F6.PROJ.transform_point(F6.LON0, F6.LAT_MIN, F6.PC)[1])
    m_per_mm = 2 * rho / F6.MAP_D_MM
    Pm = F6.PROJ.transform_points(F6.PC, main.lon.values, main.lat.values)[:, :2]
    P_mm = (Pm + rho) / m_per_mm
    center = np.array([F6.MAP_D_MM / 2, F6.MAP_D_MM / 2])
    Q_mm, disp, cluster = F6.resolve_overlaps(P_mm, center, F6.MAP_D_MM / 2)
    GAK = F6.alaska_geometry(ak, F6.AK_INSET_W_MM, F6.AK_INSET_H_MM)
    import matplotlib.axes as maxes
    _orig_scatter = maxes.Axes.scatter

    def _scatter(self, *a, **k):                          # 값이 없는 대상도 원을 그린다(색표의 'bad' = 자료 없음 회색)
        k.setdefault("plotnonfinite", True)
        return _orig_scatter(self, *a, **k)
    maxes.Axes.scatter = _scatter
    try:
        _draw_target_panels(fig, M, panels, norm, cm_, slots, P, F6, ov, rho, main, P_mm, Q_mm, disp, ak, GAK, pf_key)
    finally:
        maxes.Axes.scatter = _orig_scatter
    if cbar is not None:
        cx, cy, cw, ch = cbar
        cax = ax_mm(fig, cx, cy, cw, ch)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=ext_kind, extendfrac=0.03)
        cax.set_label("<colorbar>")
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
            cb.set_ticklabels(cbar_fmt(cbar_ticks) if cbar_fmt else [S.fmt_num(v, 0) if abs(v) > 1e-9 else "0" for v in cbar_ticks])
        cb.outline.set_linewidth(S.LW["axis"])
        cb.outline.set_edgecolor(INK)
        cb.dividers.set_linewidth(S.LW["axis"])
        cax.tick_params(width=S.LW["tick"], length=S.TICK_LEN_PT, pad=1.5)
        cax.xaxis.set_minor_locator(NullLocator())
        cb.set_label(cbar_label, labelpad=1.5)
        return cb
    return None


def _draw_target_panels(fig, M, panels, norm, cm_, slots, P, F6, ov, rho, main, P_mm, Q_mm, disp, ak, GAK, pf_key):
    for letter_, col, head in panels:
        x, y, w, h = slots[letter_]
        rect = F6.map_rect(slots[letter_])
        S.panel_letter(fig, x + F6.EDGE_MM, y + F6.EDGE_MM, letter_)
        ov.text(rect[0] + rect[2] / 2, y + F6.EDGE_MM, head, ha="center", va="top", fontsize=FS).set_gid("category")
        ax, _ = F6.polar_axes(fig, rect)
        F6.draw_pfr(ax, F6.PROJ, rect[2], rect[3], P)
        F6.graticule(ax)
        F6.draw_targets(ax, rho, main, col, P_mm, Q_mm, disp, norm, cm_, letter_)
        trow = M[M.name == "Tibet_LGD"].iloc[0]
        irect = (x, y + h - F6.INSET_H_MM, F6.INSET_W_MM, F6.INSET_H_MM)
        F6.tibet_inset(fig, irect, trow, col, norm, cm_, P, "a" if letter_ == panels[0][0] else letter_)
        arect = (x + w - F6.AK_INSET_W_MM, y + h - F6.AK_INSET_H_MM, F6.AK_INSET_W_MM, F6.AK_INSET_H_MM)
        ax_ak, _ = F6.alaska_inset(fig, arect, ak, col, norm, cm_, P, GAK, "a" if letter_ == panels[0][0] else letter_)
        F6.zoom_frame(ax, GAK, ov, ax_ak, letter_)
        if letter_ == panels[0][0]:
            ov.text(irect[0] + F6.EDGE_MM, irect[1] + irect[3] + 0.4, "Tibetan Plateau", ha="left", va="top", fontsize=FS)
            ov.text(arect[0], arect[1] + arect[3] + 0.4, "Alaska", ha="left", va="top", fontsize=FS)
            F6.lat_labels(ax)
            F6.scale_bar(ax, F6.PROJ, F6.SCALE["lon"], F6.SCALE["lat"], F6.SCALE["km"])
            if pf_key:
                ov.text(x + F6.EDGE_MM, y + F6.HEAD_MM + 1.8, "Permafrost zone", ha="left", va="center", fontsize=FS, zorder=6)
                for k, (lab, colr) in enumerate((("Continuous", S.BASEMAP["continuous"]), ("Discontinuous", S.BASEMAP["discontinuous"]))):
                    yy = y + F6.HEAD_MM + 1.8 + (k + 1) * 3.2
                    ov.add_patch(Rectangle((x + F6.EDGE_MM, yy - 0.9), 3.0, 1.8, facecolor=colr, edgecolor="none", zorder=6, gid="key_swatch"))
                    ov.text(x + 4.1, yy, lab, ha="left", va="center", fontsize=FS, zorder=6)


def sym_ticks(vmax: float) -> list:
    """0 중심 대칭 눈금: 간격 후보 가운데 눈금 9개 이하이고 vmax 가 간격의 정수배인 가장 작은 간격."""
    for st in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        k = vmax / st
        if abs(k - round(k)) < 1e-9 and 2 * round(k) + 1 <= 7:
            return [float(v) for v in np.arange(-vmax, vmax + 1e-9, st)]
    return [-vmax, 0.0, vmax]


def sym_norm(vals, step=5.0, q=99.0):
    v = np.abs(np.asarray(vals, float))
    v = v[np.isfinite(v)]
    vmax = step * math.ceil(float(np.percentile(v, q)) / step)
    return vmax, TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def ext_of(vals, vmax):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    lo, hi = bool((v < -vmax).any()), bool((v > vmax).any())
    return {(False, False): "neither", (True, False): "min", (False, True): "max", (True, True): "both"}[(lo, hi)]


# ================================================================ 학습기 × 열 격자(S8, S10)
def grid_forest(fig, x0, y0, rows_spec, cols_spec, get, name_w=38.0, col_w=19.5, col_gap=1.6, row_h=3.0, xlabel="", out_rows=None,
                head_gap=4.2, tick_vals=None, col_colors=None, head_colors=None):
    """행 = 학습기(묶음 머리 포함), 열 = 범위(풀·지역). 열마다 자기 x 축. 반환 아래 끝 y(mm).
    rows_spec: [("head", 이름) 또는 ("row", 열쇠, 이름, 색)]. cols_spec: [(열쇠, 머리, (xmin, xmax), 눈금)]."""
    ov = fig._ov
    nrow = len(rows_spec)
    ytop = y0 + head_gap
    hgt = row_h * nrow
    ys = {}
    for i, sp in enumerate(rows_spec):
        yy = ytop + row_h * (i + 0.5)
        if sp[0] == "head":
            ov.text(x0, yy, sp[1], ha="left", va="center", fontsize=FS, color=INK2).set_gid("category")
        else:
            ov.text(x0 + name_w - 1.0, yy, sp[2], ha="right", va="center", fontsize=FS, color=sp[3]).set_gid("category")
            ys[sp[1]] = i
    for j, (ck, head, xl, xt) in enumerate(cols_spec):
        xa = x0 + name_w + j * (col_w + col_gap)
        ax = ax_mm(fig, xa, ytop, col_w, hgt)
        clean(ax, left=False)
        ax.set_xlim(*xl)
        ax.set_ylim(nrow - 0.5, -0.5)
        band(ax, "x")
        zero(ax, "x")
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_major_formatter(tick_fmt(0))
        ov.text(xa + col_w / 2, y0 + head_gap - 0.8, head, ha="center", va="bottom", fontsize=FS, linespacing=1.0,
                color=(head_colors or {}).get(ck, INK)).set_gid("category")
        for sp in rows_spec:
            if sp[0] != "row":
                continue
            r = get(sp[1], ck)
            if r is None:
                continue
            ci_h(ax, ys[sp[1]], r, (col_colors or {}).get(ck, sp[3]), xlim=xl, ms=S.MS["main"] - 0.6)
            if out_rows is not None:
                out_rows.append(dict(row=sp[2], row_key=sp[1], column=ck, **{k: r.get(k) for k in ("d", "lo", "hi", "db", "lob", "hib")},
                                     source=r.get("source", "")))
    if xlabel:
        xm = x0 + name_w + (len(cols_spec) * (col_w + col_gap) - col_gap) / 2
        text_mm(fig, xm, ytop + hgt + 6.6, xlabel, ha="center", va="center").set_gid("category")
    return ytop + hgt + 8.5


# ================================================================ S8 라벨 0 전이
S8_LEARNERS = [("head", "Direct ML"),
               ("row", "cb_main", "CatBoost, main", C["direct_ml"]),
               ("row", "cb_large", "CatBoost, larger", C["direct_ml"]),
               ("row", "cb_tuned", "CatBoost, tuned", C["direct_ml"]),
               ("row", "rf", "Random forest", C["direct_ml"]),
               ("row", "f1k", "CatBoost, physics inputs", C["physics_input"]),
               ("row", "tabpfn", "TabPFN v2", C["direct_ml"]),
               ("row", "tabicl", "TabICL v2", C["direct_ml"]),
               ("row", "tabicl_full", "TabICL v2, full context", C["direct_ml"]),
               ("row", "mlp", "MLP", C["direct_ml"]),
               ("row", "tabm", "Multi-head MLP", C["direct_ml"]),
               ("row", "ftt", "FT-Transformer, reduced", C["direct_ml"]),
               ("row", "realmlp", "RealMLP", C["direct_ml"]),
               ("head", "Low-weight residual (λ = 0.25)"),
               ("row", "r0_tabpfn", "TabPFN v2", C["source_anchor_residual"]),
               ("row", "r0_cb", "CatBoost, same context", C["source_anchor_residual"]),
               ("row", "r0_tabicl", "TabICL v2", C["source_anchor_residual"])]
S8_SRC = {"cb_large": ("lgx/lgx_tests.csv", "L30", "D0[catboost]-P0|n0"), "cb_tuned": ("lgx/lgx_tests.csv", "L30", "D0[catboost_tuned]-P0|n0"),
          "rf": ("lgx/lgx_tests.csv", "L30", "D0[rf]-P0|n0"), "f1k": ("lgx/lgx_tests.csv", "L11", "F1k-P0|n0"),
          "tabpfn": ("lgt/lgt_tests.csv", "L34", "D0[T]-P0|n0"), "tabicl": ("lgf/lgf_tests.csv", "LGF-F1", "D0[I]-P0|n0"),
          "tabicl_full": ("lgf/lgf_tests.csv", "LGF-F1", "D0@full[I]-P0|n0"), "mlp": ("lgf/lgfn_tests.csv", "LGF-N1", "D0[mlp*]-P0|n0"),
          "tabm": ("lgf/lgfn_tests.csv", "LGF-N1", "D0[tabm*]-P0|n0"), "ftt": ("lgf/lgfn_tests.csv", "LGF-N1", "D0[ftt*]-P0|n0"),
          "realmlp": ("lgf/lgfn_tests.csv", "LGF-N1", "D0[realmlp*]-P0|n0"), "r0_tabpfn": ("lgt/lgt_tests.csv", "L34", "R0[T]-P0|n0"),
          "r0_cb": ("lgt/lgt_tests.csv", "L34", "R0[C]-P0|n0"), "r0_tabicl": ("lgf/lgf_tests.csv", "LGF-F5", "R0[I]-P0|n0")}
S8_COLS = [("MEAN", "Four-region\nmean", (-3.0, 8.0), [0, 4, 8]), ("Lena|x", "Lena\nDelta", (-5.0, 10.0), [-4, 0, 4, 8]),
           ("Canada|x", "Canada", (-5.0, 10.0), [-4, 0, 4, 8]), ("Russia_W|x", "W\nRussia", (-5.0, 10.0), [-4, 0, 4, 8]),
           ("Russia_E|x", "E\nRussia", (-5.0, 18.0), [0, 8, 16]), ("Alaska|x", "Alaska", (-5.0, 40.0), [0, 20, 40])]


def s8_table(stem):
    """학습기 × 열의 대비 값(셀 가중, 블록 등가중). 주 CatBoost 는 AB1(4지역)과 LG 곡선(알래스카 모드 x, n 0)."""
    T = {}
    cache = {}
    for key, (f, tid, con) in S8_SRC.items():
        if f not in cache:
            cache[f] = pd.read_csv(PROC / f, low_memory=False)
            src(stem, PROC / f)
        d = cache[f]
        q = d[(d.test_id == tid) & (d.contrast == con)]
        for ck, *_ in S8_COLS:
            qq = q[q.scope == "MEAN"] if ck == "MEAN" else q[(q.scope == "region") & (q.target == ck)]
            if len(qq) == 1:
                r = rec(qq.iloc[0])
                r["source"] = f"data/processed/{f} (test_id={tid}, contrast={con}, {'scope=MEAN' if ck == 'MEAN' else 'target=' + ck})"
                T[(key, ck)] = r
            elif len(qq) > 1:
                raise SystemExit(f"S8: {key} {ck} {len(qq)}")
    b = pd.read_csv(PROC / "lgw" / "lgw_bundle.csv")
    src(stem, PROC / "lgw" / "lgw_bundle.csv")
    a = b[b.ab == "AB1"]
    for ck, *_ in S8_COLS:
        qq = a[a.scope == "MEAN"] if ck == "MEAN" else a[(a.scope == "region") & (a.target == ck)]
        if len(qq) == 1:
            r = rec(qq.iloc[0])
            r["source"] = f"data/processed/lgw/lgw_bundle.csv (ab=AB1, {'scope=MEAN' if ck == 'MEAN' else 'target=' + ck})"
            T[("cb_main", ck)] = r
    import wf0_reanalysis as W0
    c = W0.load()
    q = c[(c.target == "Alaska") & (c["mode"] == "x") & (c.n == 0) & (c.method == "D0")]
    if len(q) != 1:
        raise SystemExit(f"S8: LG 곡선 알래스카 D0 행 {len(q)}")
    r = q.iloc[0]
    T[("cb_main", "Alaska|x")] = dict(d=r.d_p0, lo=r.d_p0_lo, hi=r.d_p0_hi, db=r.d_p0_beq, lob=r.d_p0_beq_lo, hib=r.d_p0_beq_hi,
                                      source="results/rescale_lg/data/processed/lg/lg_curve.csv (Alaska, mode x, n 0, D0, catboost_lo, cell, alpha 1)")
    src(stem, ROOT / "results/rescale_lg/data/processed/lg/lg_curve.csv")
    return T


def build_S8():
    stem = "FigS8"
    F6 = f6()
    R30, M0 = F6.load_target_deltas()
    src(stem, ROOT / "results/rescale_lg/data/processed/lg/lg_curve.csv", PROC / "lgd" / "lgd_curve_lic.csv",
        ROOT / "scripts/2_evaluation/wf0_reanalysis.py")
    Cn = map_targets()
    M = Cn.merge(M0[["name", "R1_0.25", "R1_1.0", "sig_R1_0.25", "sig_R1_1.0", "P0"]], on="name", how="left", validate="one_to_one")
    T = s8_table(stem)
    H = 168.0
    fig = fig_new(H)
    overlay(fig)
    rows = []
    vals = np.r_[M.loc[~M.name.isin(F6.INSET_TARGETS), "R1_0.25"].values, M.loc[~M.name.isin(F6.INSET_TARGETS), "R1_1.0"].values]
    vmax, norm = sym_norm(vals)
    from cmcrameri import cm as cmc
    slots = dict(a=(0.0, 0.0, 82.0, 76.0), b=(88.0, 0.0, 82.0, 76.0))
    ek = ext_of(np.r_[M["R1_0.25"].values, M["R1_1.0"].values], vmax)
    target_maps(fig, M, [("a", "R1_0.25", "Low-weight residual (λ = 0.25)"), ("b", "R1_1.0", "Anchor + residual (λ = 1.0)")], norm, cmc.broc,
                slots, cbar=(30.0, 80.0, 110.0, 2.5), cbar_label="Error change vs source Stefan, 0 labels (cm; negative = lower error)",
                cbar_ticks=sym_ticks(vmax), ext_kind=ek, rows=None)
    for r in M.itertuples():
        for lt, col in (("a", "R1_0.25"), ("b", "R1_1.0")):
            rows.append(dict(panel=lt, target=r.name, quantity=f"{col} − P0 at n = 0 (cm)", value=getattr(r, col.replace(".", "_")) if False else
                             float(M.loc[M.name == r.name, col].iloc[0]), sig=str(M.loc[M.name == r.name, "sig_" + col].iloc[0]),
                             P0_rmse=float(M.loc[M.name == r.name, "P0"].iloc[0]), lat=r.lat, lon=r.lon,
                             source="scripts/2_evaluation/wf0_reanalysis.py load(), deltas() (lg_curve.csv, lgd_curve_lic.csv)"))
    # c 학습기 × 지역
    yc = 92.0
    letter(fig, 0.3, yc + 0.3, "c")
    get = lambda rk, ck: T.get((rk, ck))  # noqa: E731
    yend = grid_forest(fig, 0.3, yc + 1.0, S8_LEARNERS, S8_COLS, get, name_w=38.0, col_w=19.5, col_gap=2.0, row_h=3.15,
                       xlabel="Error change vs source Stefan, 0 labels (cm)", out_rows=rows)
    key_ci(fig, 40.0, yend + 2.0)
    for r in rows:
        r.setdefault("panel", "c")
    extra = dict(vmax=vmax, n_map_targets=int(len(M)), missing_map_values={c_: M.loc[M[c_].isna(), "name"].tolist() for c_ in ("R1_0.25", "R1_1.0")})
    return finish(fig, stem, rows, extra_qa=extra)


# ================================================================ S9 계수 재보정
S9_REG = [("Lena", "Lena Delta"), ("Canada", "Canada"), ("Russia_W", "W Russia"), ("Russia_E", "E Russia"), ("Alaska", "Alaska")]
S9_N = [0, 3, 10, 40, 160, 320, 1000, -1]
S9_AB = [("AB4", "Recalibrated −\nsource Stefan", (-20.0, 5.0), [-15, -10, -5, 0], "recalibrated_stefan"),
         ("AB5", "Anchor + residual\n− recalibrated", (-2.0, 2.0), [-1, 0, 1], "anchor_residual"),
         ("AB6", "Pseudo-labels\nadded to residual", (-2.0, 2.0), [-1, 0, 1], "physics_pseudo"),
         ("AB7", "Residual − physics-\ninput structure", (-20.0, 8.0), [-15, -10, -5, 0, 5], "anchor_residual")]


def build_S9():
    from cmcrameri import cm as cmc
    stem = "FigS9"
    F6 = f6()
    import wf0_reanalysis as W0
    Rw = W0.deltas(W0.load())
    src(stem, ROOT / "results/rescale_lg/data/processed/lg/lg_curve.csv", PROC / "lgd" / "lgd_curve_lic.csv",
        ROOT / "scripts/2_evaluation/wf0_reanalysis.py", ROOT / "results/rescale_lg/data/processed/lg/lg_targets.csv",
        PROC / "lgx" / "lgx_curve.csv", PROC / "lgw" / "lgw_bundle.csv")
    r10 = Rw[Rw.n == 10].copy()
    r10["name"] = r10.target.str.split("|").str[0]
    r10 = r10[r10.target.str.endswith("|x")]
    tg = pd.read_csv(ROOT / "results/rescale_lg/data/processed/lg/lg_targets.csv", low_memory=False)
    tg = tg[(tg.valid == True) & (tg["mode"] == "x")]  # noqa: E712
    lr = tg.groupby("target").agg(logE=("logE_ratio_own", "mean"), logE_sd=("logE_ratio_own", "std"), E_own=("E_own", "mean"),
                                   E0=("E0", "mean")).reset_index().rename(columns={"target": "name"})
    if (lr.logE_sd.fillna(0) > 1e-9).any():
        raise SystemExit("S9: 분할 사이 ln(E_own/E0) 가 다르다")
    Cn = map_targets()
    M = Cn.merge(lr[["name", "logE", "E_own", "E0"]], on="name", how="left").merge(r10[["name", "P1", "sig_P1", "P0"]], on="name", how="left")
    if M.P1.isna().any():
        raise SystemExit(f"S9: n 10 P1 없음 {M.loc[M.P1.isna(), 'name'].tolist()}")
    H = 214.0
    fig = fig_new(H)
    ov = overlay(fig)
    rows = []
    # a 계수 오차(부호 있는 로그 비), b 라벨 10개 재보정 이득
    vmax_a = 0.2 * math.ceil(float(np.nanmax(np.abs(M.logE.values))) / 0.2)
    norm_a = TwoSlopeNorm(vmin=-vmax_a, vcenter=0.0, vmax=vmax_a)
    slots = dict(a=(0.0, 0.0, 82.0, 76.0), b=(88.0, 0.0, 82.0, 76.0))
    target_maps(fig, M, [("a", "logE", "Source coefficient error")], norm_a, cmc.vik, slots,
                cbar=(8.0, 80.0, 66.0, 2.5), cbar_label="ln(target coefficient / source coefficient)",
                cbar_ticks=[-vmax_a, -vmax_a / 2, 0.0, vmax_a / 2, vmax_a],
                cbar_fmt=lambda t: [S.fmt_num(v, 1) if abs(v) > 1e-9 else "0" for v in t])
    arc = M[~M.name.isin(F6.INSET_TARGETS)]
    vmax_b, norm_b = sym_norm(arc.P1.values)
    target_maps(fig, M, [("b", "P1", "Recalibration with 10 labels")], norm_b, cmc.broc, slots, pf_key=False,
                cbar=(96.0, 80.0, 66.0, 2.5), cbar_label="Error change vs source Stefan (cm)",
                cbar_ticks=sym_ticks(vmax_b), ext_kind=ext_of(M.P1.values, vmax_b))
    for r in M.itertuples():
        rows.append(dict(panel="a", target=r.name, quantity="ln(E_own/E0), LG target table", value=r.logE, E_own=r.E_own, E0=r.E0,
                         lat=r.lat, lon=r.lon, source="results/rescale_lg/data/processed/lg/lg_targets.csv (valid, mode x)"))
        rows.append(dict(panel="b", target=r.name, quantity="P1 − P0 at n = 10 (cm)", value=r.P1, sig=r.sig_P1, P0_rmse=r.P0,
                         lat=r.lat, lon=r.lon, source="scripts/2_evaluation/wf0_reanalysis.py deltas(), n = 10"))
    # c 토양 보정 Stefan 곡선(재보정 Stefan 과 함께), 지역별
    cv = pd.read_csv(PROC / "lgx" / "lgx_curve.csv", low_memory=False)
    cv = cv[cv.method.isin(["P1", "P1@ed"]) & (cv["mode"] == "x") & (cv.placement == "cell") & (cv.learner == "none")]
    yc = 94.0
    letter(fig, 0.3, yc + 0.3, "c")
    pw, gap = 27.0, 5.0
    labs = {0: "0", 3: "3", 10: "10", 40: "40", 160: "160", 320: "320", 1000: "1000", -1: "All"}
    for k, (rg, rn) in enumerate(S9_REG):
        x = 10.0 + k * (pw + gap)
        ax = ax_mm(fig, x, yc + 5.0, pw, 30.0)
        clean(ax)
        q = cv[cv.target == rg]
        ns_avail = [n for n in S9_N if n in set(q.n.unique()) and n != -1]
        pos = {n: i for i, n in enumerate(ns_avail)}
        pos[-1] = len(ns_avail) - 1 + 1.4
        lo_all, hi_all = [], []
        for meth, mk, ls in (("P1", "recalibrated_stefan", "-"), ("P1@ed", "soil_adjusted_stefan", SOIL_LS)):
            qq = q[q.method == meth].set_index("n")
            ns = [n for n in ns_avail + [-1] if n in qq.index]
            xs = [pos[n] for n in ns]
            dd = [float(qq.loc[n, "d_p0"]) for n in ns]
            lo = [float(qq.loc[n, "d_p0_lo"]) for n in ns]
            hi = [float(qq.loc[n, "d_p0_hi"]) for n in ns]
            main_ = [(x_, d_) for x_, d_, n in zip(xs, dd, ns) if n != -1]
            ax.plot([a_ for a_, _ in main_], [b_ for _, b_ in main_], color=C[mk], lw=S.LW["main"], ls=ls, zorder=3)
            ax.vlines(xs, lo, hi, color=C[mk], lw=S.LW["aux"], zorder=2.5)
            ax.plot(xs, dd, ls="none", marker="o" if meth == "P1" else "s", ms=S.MS["main"] - 1.0,
                    mfc=C[mk] if meth == "P1" else "white", mec=C[mk], mew=S.LW["marker_edge_open"], zorder=3.2)
            lo_all += lo
            hi_all += hi
            for n, d_, l_, h_ in zip(ns, dd, lo, hi):
                rows.append(dict(panel="c", region=rg, method=("recalibrated Stefan" if meth == "P1" else "soil-property recalibrated Stefan"),
                                 n=n, d=d_, lo=l_, hi=h_, source="data/processed/lgx/lgx_curve.csv (mode x, placement cell, learner none)"))
        zero(ax, "y")
        ylo, yhi = min(lo_all + [0]), max(hi_all + [0])
        st = 5.0 if (yhi - ylo) > 12 else (2.0 if (yhi - ylo) > 5 else 1.0)
        ylo, yhi = st * math.floor(ylo / st), st * math.ceil(yhi / st)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(FixedLocator(list(np.arange(ylo, yhi + 1e-9, st if (yhi - ylo) / st <= 5 else 2 * st))))
        ax.yaxis.set_major_formatter(tick_fmt(0))
        ax.set_xlim(-0.6, pos[-1] + 0.6)
        tk = [pos[n] for n in ns_avail] + [pos[-1]]
        ax.xaxis.set_major_locator(FixedLocator(tk))
        tl_ = [labs[n] for n in ns_avail] + ["All"]
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p_, tk=tk, tl_=tl_: tl_[int(np.argmin(np.abs(np.array(tk) - v)))]))
        for tl in ax.get_xticklabels():
            tl.set_rotation(90)
        if k == 0:
            ax.set_ylabel("Error change (cm)")
        text_mm(fig, x + pw / 2, yc + 1.5, rn, ha="center", va="top").set_gid("category")
    text_mm(fig, W_MM / 2, yc + 5.0 + 30.0 + 9.5, "Labels, n", ha="center", va="center")
    yk = yc + 5.0 + 30.0 + 14.0
    ov.plot([10.0, 15.0], [yk, yk], color=C["recalibrated_stefan"], lw=S.LW["main"])
    ov.plot([12.5], [yk], ls="none", marker="o", ms=S.MS["main"] - 1.0, mfc=C["recalibrated_stefan"], mec=C["recalibrated_stefan"])
    ov.text(16.5, yk, "Recalibrated Stefan", ha="left", va="center", fontsize=FS, color=C["recalibrated_stefan"]).set_gid("key")
    ov.plot([50.0, 55.0], [yk, yk], color=C["soil_adjusted_stefan"], lw=S.LW["main"], ls=SOIL_LS)
    ov.plot([52.5], [yk], ls="none", marker="s", ms=S.MS["main"] - 1.0, mfc="white", mec=C["soil_adjusted_stefan"], mew=S.LW["marker_edge_open"])
    ov.text(56.5, yk, "Soil-property recalibrated Stefan", ha="left", va="center", fontsize=FS, color=C["soil_adjusted_stefan"]).set_gid("key")
    ov.plot([104.0, 104.0], [yk - 1.2, yk + 1.2], color=INK2, lw=S.LW["aux"])
    ov.text(105.5, yk, "Cell-weighted 95% CI", ha="left", va="center", fontsize=FS).set_gid("key")
    # d 지역 대비(AB4–AB7, 라벨 10개)
    b = pd.read_csv(PROC / "lgw" / "lgw_bundle.csv")
    yd = yk + 6.0
    letter(fig, 0.3, yd + 0.3, "d")
    rows_spec = [("row", rg, REGION_KEY[rg], INK) for rg in REG4] + [("row", "MEAN", "Four-region mean", INK)]
    cols_spec = [(ab, head, xl, xt) for ab, head, xl, xt, _ in S9_AB]
    col_colors = {ab: C[mk] for ab, *_, mk in S9_AB}

    def get(rk, ab):
        q = b[b.ab == ab]
        qq = q[q.scope == "MEAN"] if rk == "MEAN" else q[(q.scope == "region") & (q.target == f"{rk}|x")]
        if len(qq) != 1:
            return None
        r = rec(qq.iloc[0])
        r["source"] = f"data/processed/lgw/lgw_bundle.csv (ab={ab}, {'scope=MEAN' if rk == 'MEAN' else 'target=' + rk + '|x'})"
        return r
    yend = grid_forest(fig, 0.3, yd + 1.5, rows_spec, cols_spec, get, name_w=30.0, col_w=31.0, col_gap=3.5, row_h=4.2,
                       xlabel="Error change, 10 labels (cm)", out_rows=rows, head_gap=7.0, col_colors=col_colors, head_colors=col_colors)
    key_ci(fig, 32.0, yend + 2.0)
    for r in rows:
        r.setdefault("panel", "d")
    extra = dict(vmax_a=vmax_a, vmax_b=vmax_b, logE_missing=M.loc[M.logE.isna(), "name"].tolist())
    return finish(fig, stem, rows, extra_qa=extra)


# ================================================================ S10 방법 선택(규칙 W, 원천 교차검증 조정)
S10_CAT = [("P0", "Source Stefan", "#8A9BB0"), ("P2", "Local least-squares Stefan", "#4F8CC9"),
           ("R1@0.25", "Anchor + residual, λ = 0.25", "#EFA27A"), ("R1@1.0", "Anchor + residual, λ = 1.0", C["anchor_residual"])]
S10_NN = [("mlp", "MLP"), ("tabm", "Multi-head MLP"), ("ftt", "FT-Transformer, reduced"), ("realmlp", "RealMLP")]
S10_COLS = [("R|n0|lam0.25", "0"), ("R|n10|lam0.25", "10"), ("R|nall|lam0.25", "All"), ("D|n0|lam1", "0"), ("D|n10|lam1", "10"),
            ("D|nall|lam1", "All"), ("R|n0|lam1", "0"), ("R|n10|lam1", "10"), ("R|nall|lam1", "All")]
S10_GROUPS = [("Residual, λ = 0.25", 0, 3, "anchor_residual", (-1.5, 1.5), [-1, 0, 1]),
              ("Direct ML", 3, 6, "direct_ml", (-2.0, 5.5), [0, 2, 4]),
              ("Residual, λ = 1.0", 6, 9, "anchor_residual", (-3.0, 2.0), [-2, 0, 2])]
S10_MARK = {"mlp": "o", "tabm": "s", "ftt": "D", "realmlp": "^"}


def build_S10():
    stem = "FigS10"
    F6 = f6()
    meta = json.loads((ROOT / "results/rescale_wf/data/processed/wf/wf_meta.json").read_text())
    ch = meta["choices"]["wf4|W"]
    cv = pd.read_csv(ROOT / "results/rescale_wf/data/processed/wf/wf_curve.csv", low_memory=False)
    nt = pd.read_csv(PROC / "lgf" / "lgfn_tests.csv", low_memory=False)
    src(stem, ROOT / "results/rescale_wf/data/processed/wf/wf_meta.json", ROOT / "results/rescale_wf/data/processed/wf/wf_curve.csv",
        PROC / "lgf" / "lgfn_tests.csv")
    w4 = cv[(cv.exp == "wf4") & (cv.n == 10) & (cv["mode"] == "x")]
    key_of = {"Russia_C": "Russia_C~lgd"}
    Cn = map_targets()
    rec_rows, mode_cat, dW, shares = [], [], [], []
    cat_idx = {k: i for i, (k, *_) in enumerate(S10_CAT)}
    for r in Cn.itertuples():
        t = key_of.get(r.name, r.name)
        cc = ch.get(f"{t}|10", {})
        tot = sum(cc.values())
        if cc:
            best = max(cc.items(), key=lambda kv: (kv[1], -cat_idx.get(kv[0], 99)))
            mode_cat.append(float(cat_idx[best[0]]) if best[0] in cat_idx else np.nan)
            shares.append(best[1] / tot)
        else:
            best = (None, 0)
            mode_cat.append(np.nan)
            shares.append(np.nan)
        q = w4[w4.target == t]
        rw = q[q.method == "W"].rmse
        rr = q[(q.method == "R1") & np.isclose(q.lam.astype(float), 0.25)].rmse
        d = float(rw.iloc[0] - rr.iloc[0]) if (len(rw) == 1 and len(rr) == 1 and np.isfinite(rw.iloc[0])) else np.nan
        dW.append(d)
        rec_rows.append(dict(panel="a,b", target=r.name, wf_target=t, n=10, choices=json.dumps(cc), modal=best[0],
                             modal_share=(best[1] / tot if tot else np.nan), n_choices=tot, W_minus_R1_025=d,
                             rmse_W=(float(rw.iloc[0]) if len(rw) == 1 else np.nan), rmse_R1_025=(float(rr.iloc[0]) if len(rr) == 1 else np.nan),
                             lat=r.lat, lon=r.lon, source="results/rescale_wf/data/processed/wf/wf_meta.json choices['wf4|W']; wf_curve.csv (exp wf4, mode x, n 10)"))
    M = Cn.copy()
    M["modal"] = mode_cat
    M["dW"] = dW
    H = 200.0
    fig = fig_new(H)
    ov = overlay(fig)
    rows = list(rec_rows)
    slots = dict(a=(0.0, 0.0, 82.0, 76.0), b=(88.0, 0.0, 82.0, 76.0))
    cmap_c = ListedColormap([c for *_, c in S10_CAT])
    norm_c = BoundaryNorm(np.arange(-0.5, len(S10_CAT) + 0.5), len(S10_CAT))
    target_maps(fig, M, [("a", "modal", "Method chosen most often, 10 labels")], norm_c, cmap_c, slots)
    from cmcrameri import cm as cmc
    arc = M[~M.name.isin(F6.INSET_TARGETS)]
    vmax, norm_b = sym_norm(arc.dW.values, step=1.0)
    target_maps(fig, M, [("b", "dW", "Rule − fixed residual recipe, 10 labels")], norm_b, cmc.broc, slots, pf_key=False,
                cbar=(96.0, 80.0, 66.0, 2.5), cbar_label="Error change vs fixed recipe (cm)", cbar_ticks=sym_ticks(vmax),
                ext_kind=ext_of(M.dW.values, vmax))
    # a 범주 열쇠(지도 아래 2줄, 지도 a 폭 안)
    yk = 82.0
    for i, (k, lab, colr) in enumerate(S10_CAT):
        xk = 2.0 + (i % 2) * 38.0
        yy = yk + (i // 2) * 3.6
        ov.plot([xk], [yy], ls="none", marker="o", ms=S.MS["main"], mfc=colr, mec=INK, mew=0.6)
        ov.text(xk + 2.0, yy, lab, ha="left", va="center", fontsize=FS).set_gid("key")
    if np.isnan(M.modal.values).any():
        ov.plot([2.0], [yk + 7.2], ls="none", marker="o", ms=S.MS["main"], mfc=NODATA_TARGET, mec=INK, mew=0.6)
        ov.text(4.0, yk + 7.2, "Not evaluated at 10 labels", ha="left", va="center", fontsize=FS).set_gid("key")
    # c LGF-N2 조정판 − 기본판(학습기 × 대비)
    yc = 96.0
    letter(fig, 0.3, yc + 0.3, "c")
    T = {}
    for lk, _ in S10_NN:
        for ck, _h in S10_COLS:
            kind, nn, lam = ck.split("|")
            con = (f"R1[{lk}*]-R1[{lk}]|{nn}|{lam}" if kind == "R" else f"D0[{lk}*]-D0[{lk}]|{nn}|{lam}")
            q = nt[(nt.test_id == "LGF-N2") & (nt.scope == "MEAN") & (nt.contrast == con)]
            if len(q) != 1:
                raise SystemExit(f"S10: LGF-N2 {con} {len(q)}")
            r = rec(q.iloc[0])
            r["source"] = f"data/processed/lgf/lgfn_tests.csv (test_id=LGF-N2, scope=MEAN, contrast={con})"
            r["verdict4"] = q.iloc[0].verdict4
            T[(lk, ck)] = r
    rows_spec = [("row", lk, lab, INK) for lk, lab in S10_NN]
    name_w, col_w, col_gap = 32.0, 13.0, 1.8
    cols_spec, col_colors = [], {}
    for gname, a_, b_, mk, xl, xt in S10_GROUPS:
        for ck, hd in S10_COLS[a_:b_]:
            cols_spec.append((ck, hd, xl, xt))
            col_colors[ck] = C[mk]
    get = lambda rk, ck: T.get((rk, ck))  # noqa: E731
    yend = grid_forest(fig, 0.3, yc + 5.0, rows_spec, cols_spec, get, name_w=name_w, col_w=col_w, col_gap=col_gap, row_h=4.4,
                       xlabel="Error change, tuned − default (cm)", out_rows=rows, head_gap=4.0, col_colors=col_colors, head_colors=col_colors)
    for gname, a_, b_, mk, *_ in S10_GROUPS:
        xa = 0.3 + name_w + a_ * (col_w + col_gap)
        xb = 0.3 + name_w + b_ * (col_w + col_gap) - col_gap
        ov.plot([xa, xb], [yc + 4.4, yc + 4.4], color=C[mk], lw=S.LW["axis"])
        ov.text((xa + xb) / 2, yc + 3.8, gname, ha="center", va="bottom", fontsize=FS, color=C[mk]).set_gid("category")
    text_mm(fig, 0.3 + name_w - 1.0, yc + 8.2, "Labels", ha="right", va="center", color=INK2).set_gid("category")
    key_ci(fig, 34.0, yend + 1.5)
    # d LGF-N4 산점도
    yd = yend + 8.0
    letter(fig, 0.3, yd + 0.3, "d")
    r4 = nt[(nt.test_id == "LGF-N4") & nt.spearman.notna()].iloc[0]
    pts = pd.DataFrame(json.loads(r4.points))
    axd = ax_mm(fig, 16.0, yd + 3.0, 62.0, H - yd - 12.0)
    clean(axd)
    tie = (pts.gain.abs() < 1e-12) & (pts.delta.abs() < 1e-12)
    for lk, lab in S10_NN:
        for kind, mk in (("D", "direct_ml"), ("R", "anchor_residual")):
            q = pts[(pts.learner == lk) & (pts.kind == kind) & ~tie]
            axd.plot(q.gain, q.delta, ls="none", marker=S10_MARK[lk], ms=S.MS["main"], mfc=C[mk], mec="none", alpha=0.9, zorder=3)
    axd.plot([0], [0], ls="none", marker="o", ms=S.MS["main"] + 2.0, mfc="white", mec=INK, mew=S.LW["marker_edge_open"], zorder=4)
    axd.text(0, 0, f"{int(tie.sum())} points, default kept", ha="left", va="bottom", fontsize=FS, transform=offset_tr(axd, 1.6, 0.8)).set_gid("value")
    zero(axd, "x")
    zero(axd, "y")
    axd.set_xlim(-10.0, 1.0)
    axd.set_ylim(-1.5, 5.5)
    axd.xaxis.set_major_locator(FixedLocator([-10, -8, -6, -4, -2, 0]))
    axd.xaxis.set_major_formatter(tick_fmt(0))
    axd.yaxis.set_major_locator(FixedLocator([0, 2, 4]))
    axd.yaxis.set_major_formatter(tick_fmt(0))
    axd.set_xlabel("Source cross-validation change, tuned − default (cm)")
    axd.set_ylabel("Target error change, 0 labels (cm)")
    for r in pts.itertuples():
        rows.append(dict(panel="d", learner=r.learner, kind=r.kind, target=r.target, gain=r.gain, delta=r.delta, tie=bool(abs(r.gain) < 1e-12 and abs(r.delta) < 1e-12),
                         source="data/processed/lgf/lgfn_tests.csv (test_id=LGF-N4, points)"))
    # d 열쇠
    kx, ky = 96.0, yd + 6.0
    for i, (lk, lab) in enumerate(S10_NN):
        ov.plot([kx], [ky + i * 3.6], ls="none", marker=S10_MARK[lk], ms=S.MS["main"], mfc=INK2, mec="none")
        ov.text(kx + 2.2, ky + i * 3.6, lab, ha="left", va="center", fontsize=FS).set_gid("key")
    for i, (lab, mk) in enumerate((("Direct ML", "direct_ml"), ("Residual", "anchor_residual"))):
        ov.add_patch(Rectangle((kx - 1.0, ky + (4 + i) * 3.6 + 0.6 - 1.0), 2.0, 2.0, facecolor=C[mk], edgecolor="none", gid="key_swatch"))
        ov.text(kx + 2.2, ky + (4 + i) * 3.6 + 0.6, lab, ha="left", va="center", fontsize=FS, color=C[mk]).set_gid("key")
    extra = dict(spearman=float(r4.spearman), n_points=int(r4.n_points), n_tie=int(tie.sum()),
                 modal_categories={k: int((M.modal == i).sum()) for i, (k, *_) in enumerate(S10_CAT)}, vmax_b=vmax)
    return finish(fig, stem, rows, extra_qa=extra)


# ================================================================ S11 라벨이 많은 지역 안의 이득
WF9_SHARDS = ROOT / "results/rescale_wf3/data/processed/wf/shards"


def wf9_blocks(region: str):
    """WF9 지역 내 총 저장소(region|r)에서 교차검증 잔차 가중 잔차 ML(seed 0·1 평균)과 재보정 Stefan 의 블록별 SSE 를 분할에 걸쳐 합산한
    블록별 RMSE 차(Fig 7b 와 같은 계산)."""
    import glob
    import fig7 as F7
    files = sorted(glob.glob(str(WF9_SHARDS / f"wf9__cpu__{region}__r__s*_blocksse.npz")))
    KR = [["R1", "catboost_lo", "1", "cell", -1, 0, s_, -1.0] for s_ in (0, 1)]
    KP = ["P1", "none", "1", "cell", -1, 0, -1, 0.0]
    acc, d_split, splits = {}, [], []
    for f in files:
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            i = [j for j, u in enumerate(meta["units"]) if u["target"] == f"{region}|r"]
            if len(i) != 1:
                continue
            i = i[0]
            splits.append(int(meta["units"][i]["split"]))
            keys = [json.loads(k) for k in z[f"u{i}_keys"]]
            sse, cnt, blocks, ncell = z[f"u{i}_sse"], z[f"u{i}_cnt"], z[f"u{i}_blocks"], z[f"u{i}_ncell"]
        ir = [keys.index(k) for k in KR]
        ip = keys.index(KP)
        a = [np.sqrt(sse[j].sum() / cnt[j].sum()) for j in ir]
        b = np.sqrt(sse[ip].sum() / cnt[ip].sum())
        d_split.append(np.mean(a) - b)
        for k, blk in enumerate(blocks):
            s_ = acc.setdefault(int(blk), dict(sse_resid=0.0, sse_recal=0.0, cnt=0, n_splits_scored=0, ncell=int(ncell[k])))
            s_["sse_resid"] += 0.5 * (sse[ir[0], k] + sse[ir[1], k])
            s_["sse_recal"] += sse[ip, k]
            s_["cnt"] += int(cnt[ip, k])
            s_["n_splits_scored"] += 1
    rows = []
    for bid, s_ in sorted(acc.items()):
        lat0, lon0 = F7.block_latlon(bid)
        rm = np.sqrt(s_["sse_resid"] / s_["cnt"]) if s_["cnt"] else np.nan
        rp = np.sqrt(s_["sse_recal"] / s_["cnt"]) if s_["cnt"] else np.nan
        rows.append(dict(block=bid, lat_s=lat0, lon_w=lon0, rmse_resid=rm, rmse_recal=rp, delta=rm - rp, n_cells_scored_sum=s_["cnt"],
                         n_cells_block=s_["ncell"], n_splits_scored=s_["n_splits_scored"], masked=bool(s_["cnt"] < 10)))
    return pd.DataFrame(rows), dict(n_shards=len(files), n_splits=len(set(splits)), delta_cell_weighted_mean=float(np.mean(d_split)))


def region_obs(macro: str):
    from polar.fidelity import MACRO_REGION
    import fig7 as F7
    d = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon", "region", "block", "alt_cm", "source_id"], low_memory=False)
    d["macro"] = d.region.map(lambda r: MACRO_REGION.get(r, r))
    a = d[(d.macro == macro) & (d.source_id == "F4_direct") & np.isfinite(d.alt_cm.astype(float))].copy()
    a["loc1km"] = F7.loc1km(a.lat, a.lon)
    return a.groupby("loc1km").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()


def block_map(fig, rect, B, obs, lon0, norm, cmap, scale_km=100, xlocs=None, ylocs=None, marker_mm=None):
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from matplotlib.collections import PolyCollection
    import fig7 as F7
    pc = ccrs.PlateCarree()
    proj = ccrs.Stereographic(central_latitude=90, central_longitude=lon0, true_scale_latitude=70)
    x, y, w, h = rect
    ax = ax_mm(fig, x, y, w, h, projection=proj)
    ext = F7.map_extent(proj, B, w / h, pad=0.05)
    ax.set_extent(ext, crs=proj)
    ax.spines["geo"].set_linewidth(S.LW["axis"])
    ax.spines["geo"].set_edgecolor(INK)
    land = list(shpreader.Reader(shpreader.natural_earth(resolution="50m", category="physical", name="land")).geometries())
    ax.add_geometries(land, crs=pc, facecolor=S.BASEMAP["land"], edgecolor=S.BASEMAP["coast"], linewidth=S.LW["coast"], zorder=0.4)
    gl = ax.gridlines(crs=pc, draw_labels={"left": "y", "bottom": "x"}, linewidth=S.LW["grid_map"], color=S.BASEMAP["graticule"],
                      xlocs=FixedLocator(xlocs) if xlocs is not None else None, ylocs=FixedLocator(ylocs) if ylocs is not None else None,
                      x_inline=False, y_inline=False, zorder=0.6)
    gl.rotate_labels = False
    gl.xlabel_style = dict(size=FS)
    gl.ylabel_style = dict(size=FS)
    gl.xpadding = 2
    gl.ypadding = 2
    show = B[~B.masked]
    if marker_mm:                                   # 블록이 지도 축척에서 1 mm 보다 작으면 블록 중심의 사각 표지(크기 고정)로 그린다
        ax.scatter(show.lon_w.values + 0.25, show.lat_s.values + 0.25, s=(marker_mm / 25.4 * 72) ** 2, marker="s",
                   c=show.delta.values, cmap=cmap, norm=norm, edgecolors=INK, linewidths=0.4, transform=pc, zorder=2.5)
    else:
        polys = [F7.block_poly(proj, r.lat_s, r.lon_w) for r in show.itertuples()]
        ax.add_collection(PolyCollection(polys, facecolors=cmap(norm(show.delta.values)), edgecolors="none", linewidths=0, zorder=1.0))
    msk = B[B.masked]
    if len(msk):
        mp = [F7.block_poly(proj, r.lat_s, r.lon_w) for r in msk.itertuples()]
        mc = PolyCollection(mp, facecolors=S.BASEMAP["nodata"], edgecolors=S.BASEMAP["hatch"], linewidths=0, zorder=1.0)
        mc.set_hatch("//")
        ax.add_collection(mc)
    if obs is not None:
        ax.plot(obs.lon.values, obs.lat.values, ls="none", marker="o", ms=1.5, mfc=INK, mec="none", transform=pc, zorder=2.0)
    x0, x1, y0, y1 = ext
    sx, sy = x1 - 0.05 * (x1 - x0) - scale_km * 1e3, y0 + 0.07 * (y1 - y0)
    ax.plot([sx, sx + scale_km * 1e3], [sy, sy], color=INK, lw=S.LW["scale_bar"], solid_capstyle="butt", transform=proj, zorder=3)
    ax.text(sx + scale_km * 1e3, sy + 0.02 * (y1 - y0), f"{scale_km} km", ha="right", va="bottom", fontsize=FS, transform=proj).set_gid("scale")
    return ax


S11_XE = [("R1(λ cv, xh0)−R1(λ cv, x25)", "High-resolution − baseline inputs, residual ML"),
          ("R1(λ cv, xh0)−P1", "Residual ML with high-resolution inputs − recalibrated")]
S11_XI = [("XI-a", "EP[D0(catboost)-P1]", "Penalty, direct ML − recalibrated", "direct_ml"),
          ("XI-b", "EP[R1(λ cv)-D0(catboost)]", "Penalty, residual − direct ML", "anchor_residual"),
          ("XI-c", "R1(λ cv)-P1[W]", "Residual − recalibrated, warm blocks", "anchor_residual")]
S11_XB = [("XB-1", "MEAN", "Stacking − recalibrated, transfer"),
          ("XB-2", "MEAN", "Stacked residual − residual, transfer"),
          ("XB-3", "MEAN", "Stacked residual − residual, within"),
          ("XB-4", "Alaska|r", "Stacking − recalibrated, within Alaska")]
NLAB = {-1: "All labels", 10: "10 labels", 40: "40", 100: "100 labels", 160: "160", 200: "200 labels", 500: "500", 1000: "1000"}


def forest_block(fig, x0, y0, name_w, plot_w, groups, xl, xt, xlabel, rows_out, row_h=3.2):
    """묶음 머리 + 행(n) 포레스트. groups: [(머리, 색, [(행 이름, rec, 열린 표지 여부)])]. 반환 아래 끝 y."""
    ov = fig._ov
    items = []
    for head, col, rr in groups:
        items.append(("head", head, col))
        for lab, r, open_ in rr:
            items.append(("row", lab, col, r, open_))
    n = len(items)
    ax = ax_mm(fig, x0 + name_w, y0, plot_w, row_h * n)
    clean(ax, left=False)
    ax.set_xlim(*xl)
    ax.set_ylim(n - 0.5, -0.5)
    band(ax, "x")
    zero(ax, "x")
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(tick_fmt(1 if max(abs(v) for v in xt) < 2 else 0))
    for i, it in enumerate(items):
        yy = y0 + row_h * (i + 0.5)
        if it[0] == "head":
            ov.text(x0, yy, it[1], ha="left", va="center", fontsize=FS, color=it[2]).set_gid("category")
            continue
        _, lab, col, r, open_ = it
        ov.text(x0 + name_w - 1.0, yy, lab, ha="right", va="center", fontsize=FS, color=INK).set_gid("category")
        ci_h(ax, i, r, col, xlim=xl, ms=S.MS["main"] - 0.4, mfc=("white" if open_ else None))
        rows_out.append(dict(row=lab, group=[g for g in groups if any(rr[1] is r for rr in g[2])][0][0], **{k: r.get(k) for k in ("d", "lo", "hi", "db", "lob", "hib")},
                             source=r.get("source", "")))
    ax.set_xlabel(xlabel)
    return y0 + row_h * n + 9.0


def build_S11():
    from cmcrameri import cm as cmc
    stem = "FigS11"
    BL, il = wf9_blocks("Lena")
    BC, ic = wf9_blocks("Canada")
    oL, oC = region_obs("Lena"), region_obs("Canada")
    src(stem, WF9_SHARDS / "wf9__cpu__Lena__r__s*_blocksse.npz", WF9_SHARDS / "wf9__cpu__Canada__r__s*_blocksse.npz",
        PROC / "fidelity_base_v3.csv")
    H = 198.0
    fig = fig_new(H)
    ov = overlay(fig)
    rows = []
    show = np.r_[BL[~BL.masked].delta.values, BC[~BC.masked].delta.values]
    vmax = 5.0 * math.ceil(float(np.nanpercentile(np.abs(show), 99)) / 5.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cmc.broc
    letter(fig, 0.3, 0.3, "a")
    text_mm(fig, 8.0 + 37.0, 0.6, "Lena Delta", ha="center", va="top").set_gid("category")
    block_map(fig, (8.0, 5.0, 74.0, 50.0), BL, oL, 126.7, norm, cmap, scale_km=50, xlocs=[122, 126, 130], ylocs=[71, 72, 73])
    letter(fig, 88.3, 0.3, "b")
    text_mm(fig, 96.0 + 37.0, 0.6, "Canada", ha="center", va="top").set_gid("category")
    lonC = float(np.round(BC.lon_w.mean() + 0.25))
    block_map(fig, (96.0, 5.0, 74.0, 50.0), BC, None, lonC, norm, cmap, scale_km=200, xlocs=[-140, -130, -120, -110, -100],
              ylocs=[60, 65, 70], marker_mm=2.2)
    cax = ax_mm(fig, 40.0, 62.0, 90.0, 2.5)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend=ext_of(show, vmax), extendfrac=0.03)
    cax.set_label("<colorbar>")
    cb.set_ticks(sym_ticks(vmax))
    cb.set_ticklabels([S.fmt_num(v, 0) if abs(v) > 1e-9 else "0" for v in sym_ticks(vmax)])
    cb.outline.set_linewidth(S.LW["axis"])
    cb.dividers.set_linewidth(S.LW["axis"])
    cax.tick_params(width=S.LW["tick"], length=S.TICK_LEN_PT, pad=1.5)
    cb.set_label("Error change, residual ML − recalibrated Stefan, all labels (cm)", labelpad=1.5)
    ov.plot([8.0], [71.0], ls="none", marker="o", ms=1.5, mfc=INK, mec="none")
    ov.text(9.5, 71.0, "Label location (1 km), a", ha="left", va="center", fontsize=FS).set_gid("key")
    ov.plot([131.0], [71.0], ls="none", marker="s", ms=2.2 / 25.4 * 72, mfc="white", mec=INK, mew=0.4)
    ov.text(133.0, 71.0, "Scoring block, enlarged, b", ha="left", va="center", fontsize=FS).set_gid("key")
    for reg, B, info in (("Lena", BL, il), ("Canada", BC, ic)):
        for r in B.itertuples():
            rows.append(dict(panel="a" if reg == "Lena" else "b", region=reg, block=r.block, lat_s=r.lat_s, lon_w=r.lon_w, d=r.delta,
                             rmse_resid=r.rmse_resid, rmse_recal=r.rmse_recal, n_cells_scored_sum=r.n_cells_scored_sum,
                             n_splits_scored=r.n_splits_scored, masked=r.masked,
                             source=f"results/rescale_wf3/data/processed/wf/shards/wf9__cpu__{reg}__r__s*_blocksse.npz"))
    # c XE(격자 안 입력, R1 교차검증 λ, 3대상 층화 평균)
    xe = pd.read_csv(PROC / "xbatch/XE_hires_covariates/sealed/xe_r1b_xh0_tests.csv")
    src(stem, PROC / "xbatch/XE_hires_covariates/sealed/xe_r1b_xh0_tests.csv")
    groups = []
    for con, head in S11_XE:
        rr = []
        for n_ in (200, 500, 1000, -1):
            r = one(xe[xe.scope == "MEAN"], contrast=con, n=n_)
            g = rec(r)
            g["source"] = f"xe_r1b_xh0_tests.csv (scope MEAN, contrast {con}, n {n_})"
            rr.append((NLAB[n_], g, False))
        groups.append((head, C["anchor_residual"], rr))
    yc = 77.0
    letter(fig, 0.3, yc + 0.3, "c")
    text_mm(fig, 0.3 + 4.0, yc + 0.6, "High-resolution inputs, three regions", ha="left", va="top", color=INK2).set_gid("category")
    yend_c = forest_block(fig, 0.3, yc + 5.0, 46.0, 30.0, groups, (-1.0, 1.0), [-1, -0.5, 0, 0.5, 1],
                          "Error change (cm)", rows)
    # d XI(기후 외삽 공간 대용, 캐나다)
    xi = pd.read_csv(PROC / "xbatch/XI_climate_extrapolation_retest/sealed/xi_tests.csv", low_memory=False)
    src(stem, PROC / "xbatch/XI_climate_extrapolation_retest/sealed/xi_tests.csv")
    groups = []
    for tid, lab_pat, head, mk in S11_XI:
        rr = []
        for n_ in (100, -1):
            for var, open_ in (("warm_trim", False), ("warm", True)):
                q = xi[(xi.variant == var) & (xi.n == n_) & (xi.target == f"Canada~exp~lic~{var}|r") & xi["item"].isin(["a", "b", "c"])
                       & xi.label.astype(str).str.startswith(lab_pat + "|Canada")]          # 주 판 = XI-a·b·c, 민감도 판 = XI-d 의 a·b·c 행
                if len(q) != 1:
                    raise SystemExit(f"S11 XI {tid} {var} {n_} {len(q)}")
                g = rec(q.iloc[0])
                g["source"] = f"xi_tests.csv ({q.iloc[0].label})"
                rr.append((NLAB[n_] + (", basic edition" if var == "warm" else ""), g, open_))
        groups.append((head, C[mk], rr))
    letter(fig, 88.3, yc + 0.3, "d")
    text_mm(fig, 92.0, yc + 0.6, "Climate extrapolation, Canada", ha="left", va="top", color=INK2).set_gid("category")
    yend_d = forest_block(fig, 88.3, yc + 5.0, 40.0, 40.0, groups, (-8.0, 6.0), [-8, -4, 0, 4], "Error change (cm)", rows, row_h=3.0)
    # e XB(적층)
    xb = pd.read_csv(PROC / "xbatch/XB_multisource_stacking/sealed/xb_tests.csv", low_memory=False)
    src(stem, PROC / "xbatch/XB_multisource_stacking/sealed/xb_tests.csv")
    groups = []
    for hyp, scope, head in S11_XB:
        rr = []
        q0 = xb[xb.hyp == hyp]
        q0 = q0[q0.scope == "MEAN"] if scope == "MEAN" else q0[(q0.scope == "region") & (q0.target == scope)]
        for n_ in sorted(q0.n.unique(), key=lambda v: (v == -1, v)):
            q = q0[q0.n == n_]
            if len(q) != 1:
                raise SystemExit(f"S11 XB {hyp} {n_} {len(q)}")
            g = rec(q.iloc[0])
            g["source"] = f"xb_tests.csv (hyp {hyp}, {scope}, n {n_})"
            rr.append((NLAB[int(n_)], g, False))
        groups.append((head, C["stack"], rr))
    ye = yend_c + 1.0
    letter(fig, 0.3, ye + 0.3, "e")
    text_mm(fig, 4.3, ye + 0.6, "Label-weighted stacking", ha="left", va="top", color=INK2).set_gid("category")
    yend_e = forest_block(fig, 0.3, ye + 5.0, 44.0, 38.0, groups, (-4.0, 3.0), [-4, -2, 0, 2], "Error change (cm)", rows, row_h=2.75)
    key_ci(fig, 92.0, yend_d + 2.0)
    ov.plot([92.5], [yend_d + 7.0], ls="none", marker="o", ms=S.MS["main"] - 0.4, mfc="white", mec=INK, mew=S.LW["marker_edge_open"])
    ov.text(94.5, yend_d + 7.0, "Basic warm-block edition, d", ha="left", va="center", fontsize=FS).set_gid("key")
    extra = dict(vmax=vmax, lena=il, canada=ic, n_blocks=dict(lena=int(len(BL)), canada=int(len(BC))),
                 masked=dict(lena=int(BL.masked.sum()), canada=int(BC.masked.sum())), yend=yend_e)
    return finish(fig, stem, rows, extra_qa=extra)


# ================================================================ S12 새 라벨의 배치
S12_L23 = [("D0[S_rand5]-D0[S_block]", "Direct ML, random split − block split", "direct_ml"),
           ("[D0-P1](S_randm)-[D0-P1](S_block)", "Direct ML − recalibrated, near − block labels", "direct_ml"),
           ("[R1-P1](S_randm)-[R1-P1](S_block)", "Residual − recalibrated, near − block labels", "anchor_residual"),
           ("[R1-P1](inblk)-[R1-P1](A)|n10", "Residual − recalibrated, 10 labels in scored − other blocks", "anchor_residual")]
S12_VAR = [("S8a", "Learned policy"), ("S8b", "Farthest cell in block"), ("S8c", "Square-root allocation"),
           ("S8p", "Proportional allocation")]


def build_S12():
    from cmcrameri import cm as cmc
    stem = "FigS12"
    F6 = f6()
    xc = pd.read_csv(PROC / "xbatch/XD_placement_policy/sealed/xd_alg_contrasts.csv", low_memory=False)
    src(stem, PROC / "xbatch/XD_placement_policy/sealed/xd_alg_contrasts.csv", PF / "fig5_b.csv", PF / "fig5_c.csv",
        PROC / "lgx" / "lgx_tests.csv")
    xr = xc[(xc.method.astype(str) == "R1") & np.isclose(xc.lam.astype(float), 0.25)]
    Cn = map_targets()
    key_of = {"Russia_C": "Russia_C~lgd"}
    M = Cn.copy()
    rows = []
    for con, col in (("S2-S1", "s2"), ("S4-S1", "s4")):
        vals = []
        for r in Cn.itertuples():
            t = key_of.get(r.name, r.name) + "|x"
            q = xr[(xr.contrast == con) & (xr.scope == "region") & (xr.n == 10) & (xr.target == t)]
            v = float(q.iloc[0].delta) if len(q) == 1 else np.nan
            vals.append(v)
            rows.append(dict(panel="a" if col == "s2" else "b", target=r.name, xd_target=t, contrast=con, n=10, d=v,
                             lo=(float(q.iloc[0].ci_lo) if len(q) == 1 else np.nan), hi=(float(q.iloc[0].ci_hi) if len(q) == 1 else np.nan),
                             db=(float(q.iloc[0].delta_blockeq) if len(q) == 1 else np.nan),
                             verdict4=(q.iloc[0].verdict4 if len(q) == 1 else "not run"), lat=r.lat, lon=r.lon,
                             source="xd_alg_contrasts.csv (method R1, λ 0.25, scope region, n 10)"))
        M[col] = vals
    H = 206.0
    fig = fig_new(H)
    ov = overlay(fig)
    slots = dict(a=(0.0, 0.0, 82.0, 76.0), b=(88.0, 0.0, 82.0, 76.0))
    arc = M[~M.name.isin(F6.INSET_TARGETS)]
    vmax, norm = sym_norm(np.r_[arc.s2.values, arc.s4.values], step=1.0)
    target_maps(fig, M, [("a", "s2", "Block stratified − random cells, 10 labels"), ("b", "s4", "Covariate spread − random cells, 10 labels")],
                norm, cmc.broc, slots, cbar=(30.0, 80.0, 110.0, 2.5), cbar_label="Error change, residual ML (cm; negative = lower error)",
                cbar_ticks=sym_ticks(vmax), ext_kind=ext_of(np.r_[M.s2.values, M.s4.values], vmax))
    ov.plot([147.0], [81.3], ls="none", marker="o", ms=S.MS["main"], mfc=NODATA_TARGET, mec=INK, mew=0.6)
    ov.text(149.0, 81.3, "Not tested", ha="left", va="center", fontsize=FS).set_gid("key")
    # c L23 근접 대비(4지역 층화 평균 + 지역 점)
    b5 = pd.read_csv(PF / "fig5_b.csv")
    yc = 92.0
    letter(fig, 0.3, yc + 0.3, "c")
    text_mm(fig, 4.3, yc + 0.6, "Proximity of labels to scored cells, four-region mean", ha="left", va="top", color=INK2).set_gid("category")
    name_w, pw, rh = 78.0, 60.0, 4.6
    axc = ax_mm(fig, name_w, yc + 5.0, pw, rh * len(S12_L23))
    clean(axc, left=False)
    xl = (-8.5, 3.0)
    axc.set_xlim(*xl)
    axc.set_ylim(len(S12_L23) - 0.5, -0.6)
    band(axc, "x")
    zero(axc, "x")
    axc.xaxis.set_major_locator(FixedLocator([-8, -6, -4, -2, 0, 2]))
    axc.xaxis.set_major_formatter(tick_fmt(0))
    axc.set_xlabel("Error change, first − second label set (cm)")
    for i, (con, lab, mk) in enumerate(S12_L23):
        r = one(b5, contrast=con, kind="mean")
        for g in b5[(b5.contrast == con) & (b5.kind == "region")].itertuples():
            reg = g.target.split("|")[0]
            axc.plot([g.delta], [i + 0.30], ls="none", marker=S.REGION_MARKER.get(reg, "o"), ms=S.MS["region_point"] - 0.6,
                     mfc=S.REGION_COLOR[REGION_KEY[reg]], mec="none", alpha=0.8, zorder=2.2)
            rows.append(dict(panel="c", contrast=con, target=g.target, d=g.delta, lo=g.ci_lo, hi=g.ci_hi, verdict4=g.verdict4,
                             source="data/processed/paper_figs/fig5_b.csv (LGX L23)"))
        ci_h(axc, i, rec(r), C[mk], xlim=xl)
        ov.text(name_w - 1.5, yc + 5.0 + rh * (i + 0.5), lab, ha="right", va="center", fontsize=FS, color=C[mk]).set_gid("category")
        rows.append(dict(panel="c", contrast=con, target="MEAN4", **rec(r), verdict4=r.verdict4, source="data/processed/paper_figs/fig5_b.csv (LGX L23)"))
    kx = name_w + pw + 4.0
    for j, reg in enumerate(REG4):
        ov.plot([kx], [yc + 7.0 + j * 3.4], ls="none", marker=S.REGION_MARKER[reg], ms=S.MS["region_point"] - 0.6,
                mfc=S.REGION_COLOR[REGION_KEY[reg]], mec="none", alpha=0.8)
        ov.text(kx + 1.6, yc + 7.0 + j * 3.4, REGION_KEY[reg], ha="left", va="center", fontsize=FS).set_gid("key")
    # d L24 근 − 원 차
    c5 = pd.read_csv(PF / "fig5_c.csv")
    yd = yc + 5.0 + rh * len(S12_L23) + 12.0
    letter(fig, 0.3, yd + 0.3, "d")
    text_mm(fig, 4.3, yd + 0.6, "Nearest minus farthest third of scored cells", ha="left", va="top", color=INK2).set_gid("category")
    tg = [t for t in ["Lena|x", "Canada|x", "Alaska|x"] + sorted(set(c5.target) - {"Lena|x", "Canada|x", "Alaska|x"}) if t in set(c5.target)]
    items = [(t, n_) for t in tg for n_ in (40, 160) if len(c5[(c5.target == t) & (c5.n == n_)])]
    rh2 = 3.4
    axd = ax_mm(fig, 40.0, yd + 5.0, 40.0, rh2 * len(items))
    clean(axd, left=False)
    xl2 = (-3.5, 3.0)
    axd.set_xlim(*xl2)
    axd.set_ylim(len(items) - 0.5, -0.5)
    band(axd, "x")
    zero(axd, "x")
    axd.xaxis.set_major_locator(FixedLocator([-3, -2, -1, 0, 1, 2, 3]))
    axd.xaxis.set_major_formatter(tick_fmt(0))
    axd.set_xlabel("Near − far difference (cm)")
    for i, (t, n_) in enumerate(items):
        nm = t.split("|")[0]
        lab = (REGION_KEY.get(nm, nm) + f", {n_} labels") if "|x" in t else f"Sub-region {nm} (within), {n_}"
        ov.text(40.0 - 1.0, yd + 5.0 + rh2 * (i + 0.5), lab, ha="right", va="center", fontsize=FS).set_gid("category")
        for which, mk, dy in (("R1 − P1", "anchor_residual", -0.16), ("P1 − P0", "recalibrated_stefan", 0.18)):
            q = c5[(c5.target == t) & (c5.n == n_) & (c5.which == which)]
            if len(q) != 1:
                continue
            r = rec(q.iloc[0])
            ci_h(axd, i + dy, r, C[mk], xlim=xl2, ms=S.MS["main"] - 1.0, block=False)
            rows.append(dict(panel="d", target=t, n=n_, which=which, **r, verdict4=q.iloc[0].verdict4,
                             source="data/processed/paper_figs/fig5_c.csv (LGX L24)"))
    ky = yd + 5.0 + rh2 * len(items) + 12.0
    ov.plot([2.0, 6.0], [ky, ky], color=C["anchor_residual"], lw=S.LW["ci_forest_cell"])
    ov.text(7.0, ky, "Residual − recalibrated", ha="left", va="center", fontsize=FS, color=C["anchor_residual"]).set_gid("key")
    ov.plot([42.0, 46.0], [ky, ky], color=C["recalibrated_stefan"], lw=S.LW["ci_forest_cell"])
    ov.text(47.0, ky, "Recalibrated − source", ha="left", va="center", fontsize=FS, color=C["recalibrated_stefan"]).set_gid("key")
    # e XD 배치 정책(전이, PE1 층화 평균; 2단 CI)
    xe0 = 86.0
    letter(fig, xe0 + 0.3, yd + 0.3, "e")
    text_mm(fig, xe0 + 4.3, yd + 0.6, "Placement policies, five-region mean", ha="left", va="top", color=INK2).set_gid("category")
    T = {}
    for vk, _ in S12_VAR:
        for ref in ("S2", "S4"):
            for n_ in (10, 40):
                q = xr[(xr.contrast == f"{vk}-{ref}") & (xr.scope == "PE1") & (xr.n == n_)]
                if len(q) == 1:
                    r = rec(q.iloc[0])
                    r["source"] = f"xd_alg_contrasts.csv (R1 λ 0.25, PE1, {vk}-{ref}, n {n_})"
                    T[(vk, f"{ref}|{n_}")] = r
    rows_spec = [("row", vk, lab, C["workflow"]) for vk, lab in S12_VAR]
    cols_spec = [("S2|10", "10", (-2.0, 1.5), [-1, 0, 1]), ("S2|40", "40", (-2.0, 1.5), [-1, 0, 1]),
                 ("S4|10", "10", (-2.0, 1.5), [-1, 0, 1]), ("S4|40", "40", (-2.0, 1.5), [-1, 0, 1])]
    get = lambda rk, ck: T.get((rk, ck))  # noqa: E731
    nw_e, cw_e, cg_e = 33.0, 11.5, 1.6
    yend = grid_forest(fig, xe0 + 0.3, yd + 6.0, rows_spec, cols_spec, get, name_w=nw_e, col_w=cw_e, col_gap=cg_e, row_h=3.6,
                       xlabel="Error change (cm)", out_rows=rows, head_gap=4.4)
    for gname, a_, b_ in (("vs block stratified", 0, 2), ("vs covariate spread", 2, 4)):
        xa = xe0 + 0.3 + nw_e + a_ * (cw_e + cg_e)
        xb = xe0 + 0.3 + nw_e + b_ * (cw_e + cg_e) - cg_e
        ov.plot([xa, xb], [yd + 6.0, yd + 6.0], color=INK2, lw=S.LW["axis"])
        ov.text((xa + xb) / 2, yd + 5.4, gname, ha="center", va="bottom", fontsize=FS, color=INK2).set_gid("category")
    ov.text(xe0 + 0.3 + nw_e - 1.0, yd + 9.6, "Labels", ha="right", va="center", fontsize=FS, color=INK2).set_gid("category")
    # XD-1, XD-2
    groups = []
    rr = []
    for n_ in (10, 40):
        q = xr[(xr.contrast == "S8a-S1") & (xr.scope == "PE1") & (xr.n == n_)]
        g = rec(q.iloc[0]); g["source"] = f"xd_alg_contrasts.csv (R1 λ 0.25, PE1, S8a-S1, n {n_})"
        rr.append((NLAB[n_] if n_ == 10 else "40 labels", g, False))
    groups.append(("Learned policy − random cells", C["workflow"], rr))
    rr = []
    for n_ in (10, 40):
        q = xr[(xr.contrast == "S8w-S8a") & (xr.scope == "region") & (xr.target == "Lena|x") & (xr.n == n_)]
        g = rec(q.iloc[0]); g["source"] = f"xd_alg_contrasts.csv (R1 λ 0.25, Lena|x, S8w-S8a, n {n_})"
        rr.append((NLAB[n_] if n_ == 10 else "40 labels", g, False))
    groups.append(("Design-weighted coefficient, Lena Delta", C["workflow"], rr))
    yend2 = forest_block(fig, xe0 + 0.3, yend + 3.0, 38.0, 41.0, groups, (-4.0, 2.0), [-4, -2, 0, 2], "Error change (cm)", rows, row_h=3.2)
    key_ci(fig, xe0 + 2.0, yend2 + 1.0)
    for r in rows:
        r.setdefault("panel", "e")
    extra = dict(vmax=vmax, missing_map=M.loc[M.s2.isna(), "name"].tolist(), yend=yend2)
    return finish(fig, stem, rows, extra_qa=extra)


# ================================================================ 실행
BUILDERS = {"S7": build_S7, "S8": build_S8, "S9": build_S9, "S10": build_S10, "S11": build_S11, "S12": build_S12, "S14": build_S14, "S15": build_S15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(BUILDERS))
    a = ap.parse_args()
    S.use_v3("paper")
    S.mathtext_liberation()
    matplotlib.rcParams.update({"axes.unicode_minus": True, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5})
    t0 = time.time()
    for k in [s.strip() for s in a.only.split(",") if s.strip()]:
        t1 = time.time()
        S.use_v3("paper")
        S.mathtext_liberation()
        matplotlib.rcParams.update({"axes.unicode_minus": True, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5})
        BUILDERS[k]()
        print(f"[done] {k} {time.time() - t1:.0f}s", flush=True)
    print(f"[all] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
