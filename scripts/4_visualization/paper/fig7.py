"""Fig 7. 배포 절차와 확인적 검증(스펙 figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §2.7, §1.6, §5; 확정 결과 사실 목록).

패널
  a  증거 기반 배포 절차 결정 나무(라벨 0 / 3–10 / 수백·전량, 실선 화살표). 사전 등록 2단계 분기(3라벨 E비 진단)는
     B1 에서 기각(F4)되었으므로 권고 경로가 아니라 회색 점선 곁가지 'tested, not supported (b)' 로만 둔다.
     잎 태그 = 기대 Δ 범위 + 근거 그림 번호. 수치는 workflow_tags.json(build_workflow_tags 가 원천 CSV 에서 계산)에서만 읽는다.
  b  B1 2단계 프로토콜 대 비교군 덤벨(14 대상 평균 = 채운 마커, 최악 대상 = 세로 틱 4 pt(스펙 §1.1), n = 3 실선·n = 10 파선)
     + 악화 대상 수 막대(연회색 = 점 추정 Δ > 0, 진회색 = 블록 CI 하한 > 0). 주 판정 행 = protocol@loto.
  c  D1 외부 홀드아웃(몽골·중앙아시아) 상대 Δ(% of physics RMSE), 블록 부트스트랩 95 % CI(끝 틱 없음). 사전 등록 검정 행(Protocol)은
     굵은 행 이름으로만 표시하고 예측 방향(Δ < 0)은 캡션 F11 문장에 적는다(전 구간 빗금 띠는 정보가 없어 제거, 검토 2026-09-26).

규칙
  - 자리표 금지: 본 실행 자료가 없으면 MissingData 로 멈춘다.
  - 라벨 3–10 잎 기대 Δ 는 b 의 'Shrink + res. (S3)'(b1_summary always_shrink_resid, n = 3·10) 대상별 블록 부트스트랩 평균의
    오차 유형별 범위다. 오차 유형 = 오라클 |log(E_own/E0)|(수준 ≥ 0.20, 중간 0.15–0.20, 구조 < 0.15).
    n = 3 또는 10 에서 Δ > +1 cm 인 대상(EXC_CUT)은 범위에서 빼고 '> 1 cm 악화' 대상 수와 최대 악화 대상(CA-3)만 적는다
    (사실 목록 '수준 −1.6 ~ −11, 구조 약 0 ± 1, 일부 악화(CA-3 +4)'와 같은 정의·같은 예시. 대상별 값은 Supplementary Table 4).
  - D1 의 offset_only 와 B1 의 explore_gate_agree 는 결과 열람 후 추가한 탐색 행이다: 계열 색 유지 + 행 이름 '†'·기울임,
    7행 위아래 0.3 pt 규선(b·c 에서 서로 다른 방법).
  - c 의 게재 조건: d1_bias_check() 가 라벨 정의·단위·영구동토 존재·조사 연도를 원자료에서 재확인해
    _qa/Fig7_d1_bias_check.json 에 기록한다. 불통과면 c 를 빼고 b 를 넓힌 대체 판을 만든다.
  - D1 대상 라벨은 전부 F4_calm_temp(지온 유도 융해 깊이)이고 원천 학습 라벨과 정의가 다르다. 캡션에 교락을 명시한다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import (ps, H2, H3, H4, OUT, ROOT, TARGETS14, LEVEL_THRESH, MissingData, abslogE_map, fmt_num, fmt_ci,
                     load_recipe_table, load_b1, load_d1, load_c2_coverage, load_a2_minn, rd, save_paper)

TAGS_JSON = OUT / "workflow_tags.json"
DAG = "†"                                        # † = 결과 열람 후 추가한 탐색 행

# ---------------------------------------------------------------- 1) 태그 JSON(스펙 §2.7 스키마, 증거 기반 절차 판)
# 잎: P0 = 라벨 0, PF = 라벨 3–10, P3 = 라벨 수백·전량(Table 1 이 P3 수치를 a1_recipe_table 과 대조), PROTOCOL = 사전 등록
# 2단계 분기(F4 기각, 권고 경로 아님, 회색 점선 곁가지). 수치는 전부 아래 원천 CSV 에서 계산한다.
LEVEL_CUT, STRUCT_CUT = 0.20, 0.15                  # 원고 단일 정의: 수준 ≥ 0.20, 중간 0.15–0.20, 구조 < 0.15(오라클 |log E비|)
EXC_CUT = 1.0                                       # PF 잎: n = 3·10 중 Δ > +1 cm 인 대상은 범위 밖 '악화' 목록으로
MIN_BLOCKS = 5                                      # src/polar/h4_common.py MIN_BLOCKS_FLAG 와 같은 값(채점 블록 < 5 = 저신뢰)
TAG_SCHEMA = [
    dict(leaf_id="P0", fig_ref="Figs 6a, 6c, 6d", source_csv="data/processed/h4/c2_coverage.csv",
         filter="test=='label0' & method=='hier2_cdf' & target.startswith('MEAN6[Russia_W,Russia_E,Canada,Lena]') & scope=='all'",
         stat="coverage_ci", text="Δ 0 cm by construction; 90 % interval coverage {est} [{lo}, {hi}] · Figs 6a, 6c, 6d"),
    dict(leaf_id="PF", fig_ref="Figs 3e, 4f, 5d, 7b", source_csv="data/processed/h4/b1_summary.csv",
         filter="method=='always_shrink_resid' & n in (3, 10); class by oracle |log(E_own/E0)|; targets with Δ > +1 cm at "
                "either n listed separately", stat="d_phys_block_range_by_class",
         text="Expected Δ: level {lv_lo} to {lv_hi} cm, structure {st_lo} to {st_hi} cm\n"
              "Worse by > 1 cm: {n_exc} targets, largest {exc_top} · Figs 3e, 4f, 5d, 7b"),
    dict(leaf_id="P3", fig_ref="Figs 2b, 4e", source_csv="data/processed/h4/a1_recipe_table.csv",
         filter="H=='H13' & target=='REGION_SUMMARY_AB4' & lam==0.25", stat="delta_ci",
         text="Expected Δ {est} [{lo}, {hi}] cm, 4-region mean · Figs 2b, 4e"),
    dict(leaf_id="PROTOCOL", fig_ref="Fig. 7b", source_csv="data/processed/h4/b1_protocol.csv",
         filter="method=='protocol@{tau_sel}' & n==3", stat="mean_d_n_worse",
         text="Tested, not supported (b): mean Δ {n3} cm at n = 3, {n_worse} of {n_targets} targets worse"),
]


def _classes() -> dict:
    m = abslogE_map()
    cls = {}
    for t in TARGETS14:
        v = m.get(t, np.nan)
        cls[t] = "level" if v >= LEVEL_CUT else ("structure" if v < STRUCT_CUT else "intermediate")
    return cls


def _tag_values(leaf: str, draft: bool) -> dict:
    """잎별 수치. 자료가 없으면 MissingData."""
    if leaf == "P0":
        C2 = load_c2_coverage(draft)
        q = C2[(C2.test == "label0") & (C2.method == "hier2_cdf") & (C2.scope == "all")
               & C2.target.astype(str).str.startswith("MEAN6[Russia_W,Russia_E,Canada,Lena]")]
        if len(q) != 1:
            raise MissingData(f"c2_coverage hier2_cdf AB4 행 {len(q)}개")
        r = q.iloc[0]
        return dict(est=float(r.coverage), lo=float(r.coverage_lo), hi=float(r.coverage_hi), width_cm=float(r.width_cm),
                    ci_kind="block")
    if leaf == "PF":
        B = load_b1(draft)
        S = B["summary"]
        q = S[(S.method == "always_shrink_resid") & S.n.isin([3, 10])].copy()
        cls = _classes()
        q["cls"] = q.target.map(cls)
        if q.cls.isna().any() or len(q) != 2 * len(TARGETS14):
            raise MissingData("b1_summary always_shrink_resid 대상·n 불완전")
        tmax = q.groupby("target").blk_d_phys.max()
        exc = tmax[tmax > EXC_CUT].sort_values(ascending=False)             # n = 3·10 중 한 번이라도 +1 cm 넘게 악화
        core = q[~q.target.isin(exc.index)]
        lv, st = core[core.cls == "level"], core[core.cls == "structure"]
        nb = q.groupby("target").n_blocks_min.min() if "n_blocks_min" in q else pd.Series(dtype=float)
        low_blk = sorted(t for t, v in nb.items() if v < MIN_BLOCKS)
        ca3 = q[q.target == "CA-3"].blk_d_phys
        return dict(lv_lo=float(lv.blk_d_phys.min()), lv_hi=float(lv.blk_d_phys.max()),
                    st_lo=float(st.blk_d_phys.min()), st_hi=float(st.blk_d_phys.max()),
                    exc={t: float(v) for t, v in exc.items()}, exc_cls={t: cls[t] for t in exc.index},
                    exc_blocks={t: int(nb.get(t, -1)) for t in exc.index},
                    ca3=float(ca3.max()), ca3_all=[float(v) for v in ca3],
                    lo=float(q.blk_d_phys.min()), hi=float(q.blk_d_phys.max()),
                    level_targets=sorted(lv.target.unique()), structure_targets=sorted(st.target.unique()),
                    lv_argmax=str(lv.loc[lv.blk_d_phys.idxmax(), "target"]) + f" n={int(lv.loc[lv.blk_d_phys.idxmax(), 'n'])}",
                    st_argmax=str(st.loc[st.blk_d_phys.idxmax(), "target"]) + f" n={int(st.loc[st.blk_d_phys.idxmax(), 'n'])}",
                    low_block_targets=low_blk, min_blocks_flag=MIN_BLOCKS,
                    ci_kind="block (per target)")
    if leaf == "P3":
        R = load_recipe_table("H13", None)
        q = R[(R.target == "REGION_SUMMARY_AB4") & R.label.str.contains("λ=0.25", regex=False)]
        if len(q) != 1:
            raise MissingData(f"a1_recipe_table H13 AB4 λ0.25 행 {len(q)}개")
        r = q.iloc[0]
        M = load_a2_minn(draft)
        lm = M[(M.target == "Lena") & (M.e_treat == "E0_fixed") & (M.spread == "all_blocks") & np.isclose(M.lam, 0.25)
               & (M.stage == "resid")]
        if len(lm) != 1:
            raise MissingData("a2_minn Lena E0_fixed all_blocks λ0.25 행 없음")
        lr = lm.iloc[0]
        # 구조 오차 대상(원고 단일 정의, 오라클 |log E비| < 0.15) 중 E0 고정·전 블록 분산·λ 0.25 에서 n ≤ 40 으로 n* 에 이르지 못한 수
        # (사실 목록 F1 정정: 8개 중 6개, 캐나다·AL-5 는 n = 0 에서 이미 물리식보다 낫다)
        g = M[(M.e_treat == "E0_fixed") & (M.spread == "all_blocks") & np.isclose(M.lam, 0.25) & (M.stage == "resid")]
        struct = [t for t, c in _classes().items() if c == "structure"]
        gs = g[g.target.isin(struct)].set_index("target")
        if set(gs.index) != set(struct):
            raise MissingData(f"a2_minn 구조 대상 누락: {sorted(set(struct) - set(gs.index))}")
        ok40 = sorted(t for t in struct if (not bool(gs.loc[t, "censored"])) and float(gs.loc[t, "n_star"]) <= 40)
        return dict(est=float(r.delta), lo=float(r.ci_lo), hi=float(r.ci_hi), p_holm=float(r.p_holm), ci_kind="strat_AB4",
                    lena_nstar=None if bool(lr.censored) else int(lr.n_star), lena_nmax=int(lr.n_max),
                    n_struct=len(struct), n_struct_slow=len(struct) - len(ok40), struct_fast=ok40)
    if leaf == "PROTOCOL":
        B = load_b1(draft)
        if B["tau_sel"] is None:
            raise MissingData("b1 τ 선택 기록 없음(tau_sel_loto·protocol@loto)")
        P = B["protocol"]
        q = P[P.method == f"protocol@{B['tau_sel']}"].set_index("n")
        return dict(n3=float(q.loc[3, "mean_d"]), n10=float(q.loc[10, "mean_d"]), lo=float(q.mean_d.min()), hi=float(q.mean_d.max()),
                    n_worse=int(q.loc[3, "n_worse"]), n_worse_ci=int(q.loc[3, "n_worse_ci"]), n_targets=int(q.loc[3, "n_targets"]),
                    gap_to_oracle=float(q.loc[3, "gap_to_oracle"]), tau_sel=B["tau_sel"], verdict="F4 not supported")
    raise KeyError(leaf)


def build_workflow_tags(draft: bool = False, write: bool = True) -> list[dict]:
    """workflow_tags.json 생성. 본 실행 자료가 없으면 MissingData 를 올린다(자리표 금지)."""
    out = []
    for s in TAG_SCHEMA:
        d = {**s, "lo": None, "hi": None}
        v = _tag_values(s["leaf_id"], draft)
        d.update(v); d["pending"] = False
        nd = 2 if s["leaf_id"] in ("P0", "PROTOCOL") else 1
        fmt = {k: fmt_num(v[k], nd) for k in ("lo", "hi", "est", "n3", "n10", "lv_lo", "lv_hi", "st_lo", "st_hi") if k in v}
        if "ca3" in v:
            fmt["ca3"] = ("+" if v["ca3"] > 0 else "") + fmt_num(v["ca3"], 1)
        if "exc" in v:                                  # 사실 목록과 같은 예시(최대 악화 대상)만 이름으로, 나머지는 수로
            t0, x0 = next(iter(v["exc"].items()))
            fmt["exc_top"] = f"{t0} +{fmt_num(x0, 1)} cm"
            fmt["n_exc"] = str(len(v["exc"]))
        for k in ("n_worse", "n_targets"):
            if k in v:
                fmt[k] = str(v[k])
        if s["leaf_id"] == "PF":                        # 상한이 양수면 '+' 를 붙여 부호를 명시한다
            for k in ("lv_hi", "st_hi"):
                if v[k] > 0:
                    fmt[k] = "+" + fmt[k]
        d["render"] = s["text"].format(**fmt)
        out.append(d)
    if write:
        TAGS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n")
    return out


# ---------------------------------------------------------------- 2) 패널 a: 결정 나무(증거 기반 절차)
# 좌표는 패널 축 안 mm(가로 0–170, 세로 0–80, 좌하 원점). 노드: (종류, 중심 x, 중심 y, 폭, 높이).
# 권고 경로 = 실선 화살표 3개(라벨 0 / 3–10 / 수백·전량). 사전 등록 2단계 분기는 F4 기각이므로 회색 점선 곁가지로만 둔다.
AX_W, AX_H = 170.0, 66.0
Y_MAIN = 37.0
LX0 = 66.0                                           # 잎 왼쪽 x
LW_ = AX_W - LX0 - 0.4
NODES = {
    "N0": ("start", 11.0, Y_MAIN, 22.0, 13.0),
    "Q1": ("decision", 39.0, Y_MAIN, 24.0, 14.0),
    "P0": ("leaf", LX0 + LW_ / 2, 58.4, LW_, 12.4),
    "PF": ("leaf", LX0 + LW_ / 2, Y_MAIN, LW_, 19.4),
    "PR": ("tested", 74.0 + (AX_W - 0.4 - 74.0) / 2, 20.4, AX_W - 0.4 - 74.0, 8.6),
    "P3": ("leaf", LX0 + LW_ / 2, 6.4, LW_, 12.4),
}
LEAF_METHOD = {"P0": "phys", "PF": "residual", "P3": "residual"}
X_SIDE = 62.5                                        # 곁가지 분기 x(Q1 → PF 간선 위)


def _box(n):
    _, cx, cy, w, h = NODES[n]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _port(n, side):
    x0, y0, x1, y1 = _box(n)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return dict(l=(x0, cy), r=(x1, cy), t=(cx, y1), b=(cx, y0))[side]


# 간선: (출발, 도착, waypoint, 라벨, 라벨 위치(x, y, ha, va), 종류 'main'|'tested')
def _edges():
    q1t, q1b, q1r = _port("Q1", "t"), _port("Q1", "b"), _port("Q1", "r")
    p0l, pfl, p3l, prl = _port("P0", "l"), _port("PF", "l"), _port("P3", "l"), _port("PR", "l")
    return [
        ("N0", "Q1", [_port("N0", "r"), _port("Q1", "l")], None, None, "main"),
        ("Q1", "P0", [q1t, (q1t[0], p0l[1]), p0l], "n = 0", (q1t[0] - 1.6, (q1t[1] + p0l[1]) / 2, "right", "center"), "main"),
        ("Q1", "PF", [q1r, pfl], "n = 3–10", ((q1r[0] + X_SIDE) / 2 + 0.3, Y_MAIN + 0.8, "center", "bottom"), "main"),
        ("Q1", "P3", [q1b, (q1b[0], p3l[1]), p3l], "Hundreds or all", (q1b[0] - 1.6, (q1b[1] + p3l[1]) / 2, "right", "center"),
         "main"),
        ("Q1", "PR", [(X_SIDE, Y_MAIN), (X_SIDE, prl[1]), prl], None, None, "tested"),
    ]


def check_edges(edges) -> list:
    """간선 선분이 출발·도착 이외 노드 상자를 지나면 목록으로 돌려준다(스펙: 교차 0건). 마름모는 외접 사각형으로 검사."""
    from shapely.geometry import LineString, box
    bad = []
    for s, t, pts, *_ in edges:
        line = LineString(pts)
        for n in NODES:
            if n in (s, t):
                continue
            if line.intersects(box(*_box(n)).buffer(0.3)):
                bad.append((s, t, n))
    return bad


def _tint(hexcol: str, a: float = 0.13) -> tuple:
    from matplotlib.colors import to_rgb
    r = np.array(to_rgb(hexcol))
    return tuple(1 - a * (1 - r))


def draw_tree(ax, tags: dict):
    from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
    from matplotlib.path import Path
    ax.set_xlim(0, AX_W); ax.set_ylim(0, AX_H); ax.set_axis_off()
    ax.set_gid("diagram")
    FS_NODE, FS_SMALL = 7.0, ps.FS["annot"]
    EDGE, TESTED = "#4d4d4d", ps.GREY["mid"]
    p3 = tags["P3"]
    lena = (f"Lena n* = {p3['lena_nstar']}" if p3.get("lena_nstar") is not None else f"Lena n* > {p3['lena_nmax']}")
    body = {
        "N0": "New region\n25 covariates, TDD",
        "Q1": "Target\nlabels?",
        "P0": "Physics anchor E$_0$ + hierarchical conformal 90 % interval + AOA mask\n"
              "No direct ML (Fig. 1d); no CCI fusion (no gain, Fig. 6c)",
        "PF": "Estimate E from the labels, shrunk toward E$_0$ (κ = 10)\n"
              "Spread labels at random over many blocks; add source residual ML (λ 0.25)\n"
              "If |log(Ê$_3$/E$_0$)| > 1: offset-MLE shrinkage (exploratory, c)",
        "P3": "Pre-specified recipe: Stefan anchor + CatBoost residual ML (λ 0.25)\n"
              f"{p3['n_struct_slow']} of {p3['n_struct']} structure-error regions need hundreds of labels ({lena}, Fig. 4)",
        "PR": "Pre-registered two-stage rule: if |log(Ê$_3$/E$_0$)| ≥ τ, shrink E only; else E$_0$ + residual ML",
    }
    for n, (kind, cx, cy, w, h) in NODES.items():
        x0, y0, x1, y1 = _box(n)
        if kind == "decision":
            ax.add_patch(Polygon([(x0, cy), (cx, y1), (x1, cy), (cx, y0)], closed=True, facecolor="white", edgecolor="#000000",
                                 lw=0.6, zorder=2))
            ax.text(cx, cy, body[n], ha="center", va="center", fontsize=FS_NODE, linespacing=1.1, zorder=3)
            continue
        if kind == "start":
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=ps.GREY["land"],
                                        edgecolor="#808080", lw=0.6, zorder=2))
            ax.text(cx, cy, body[n], ha="center", va="center", fontsize=FS_NODE, linespacing=1.25, zorder=3)
            continue
        pad = 1.6
        if kind == "tested":                           # 검정했으나 지지되지 않은 곁가지: 흰 바탕, 회색 점선 테두리, 회색 글자
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor="white",
                                        edgecolor=TESTED, lw=0.6, ls=(0, (2.0, 1.4)), zorder=2))
            ax.text(x0 + pad, y1 - 1.3, body[n], ha="left", va="top", fontsize=FS_SMALL, color=ps.GREY["text2"], zorder=3)
            t = ax.text(x1 - pad, y0 + 1.2, tags["PROTOCOL"]["render"], ha="right", va="bottom", fontsize=FS_SMALL,
                        color=ps.GREY["text2"], style="italic", zorder=4)
            t.set_gid("tag")
            continue
        m = LEAF_METHOD[n]
        fc, ec = _tint(ps.COLOR[m], 0.13 if m != "phys" else 0.10), ps.COLOR[m]
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fc, edgecolor=ec,
                                    lw=0.6, zorder=2))
        ax.text(x0 + pad, y1 - 1.3, body[n], ha="left", va="top", fontsize=FS_NODE, linespacing=1.25, zorder=3)
        t = ax.text(x1 - pad, y0 + 1.2, tags[n]["render"], ha="right", va="bottom", fontsize=FS_SMALL, color="#000000", zorder=4)
        t.set_gid("tag")
    edges = _edges()
    bad = check_edges(edges)
    if bad:
        raise RuntimeError(f"Fig 7a 간선이 노드를 지난다: {bad}")
    for s, t_, pts, lab, lpos, kind in edges:
        codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
        tested = kind == "tested"
        if tested:                                     # 점선 몸통 + 실선 화살촉(점선 무늬가 화살촉 윤곽에 걸리지 않게)
            (xe, ye), (xp, yp) = pts[-1], pts[-2]
            stub = (xe - 1.6, ye) if xp < xe else (xe, ye)
            xs, ys = zip(*(pts[:-1] + [stub]))
            ax.plot(xs, ys, color=TESTED, lw=0.6, ls=(0, (2.0, 1.4)), solid_joinstyle="miter", zorder=1.5)
            pts, codes = [stub, (xe, ye)], [Path.MOVETO, Path.LINETO]
        arr = FancyArrowPatch(path=Path(pts, codes), arrowstyle="-|>,head_length=3,head_width=1.5", mutation_scale=1,
                              color=TESTED if tested else EDGE, lw=0.6, shrinkA=0, shrinkB=0.3, joinstyle="miter",
                              capstyle="butt", zorder=1.5)
        ax.add_patch(arr)
        if lab:
            x, y, ha, va = lpos
            tt = ax.text(x, y, lab, ha=ha, va=va, fontsize=FS_SMALL, color="#000000", zorder=4)
            tt.set_gid("edge_label")
    return edges


# ---------------------------------------------------------------- 3) 범례 전용 기호(평균→최악 덤벨, CI 막대)
from matplotlib.legend_handler import HandlerBase


class DumbbellHandle:
    """범례: 채운 마커(평균) → 세로 틱(최악 대상) 덤벨(스펙 §1.1 '세로 틱 4 pt = 덤벨의 최악 대상 값')."""
    def __init__(self, color, marker="o", ms=3.0, label=""):
        self.color, self.marker, self.ms, self._label = color, marker, ms, label

    def get_label(self):
        return self._label


class CIHandle:
    """범례: 끝 틱이 있는 가로 CI 막대."""
    def __init__(self, color, label=""):
        self.color, self._label = color, label

    def get_label(self):
        return self._label


class _HDumbbell(HandlerBase):
    def create_artists(self, legend, h, x0, y0, width, height, fontsize, trans):
        from matplotlib.lines import Line2D
        y = y0 + height / 2
        a, b = x0 + 0.12 * width, x0 + 0.88 * width
        ln = Line2D([a, b], [y, y], color=h.color, lw=0.8, transform=trans)
        m1 = Line2D([a], [y], ls="none", marker=h.marker, ms=h.ms, mfc=h.color, mec=h.color, mew=ps.LW["marker_edge"],
                    transform=trans)
        m2 = Line2D([b], [y], ls="none", marker="|", ms=WORST_MS, mew=WORST_MEW, color=h.color, transform=trans)
        return [ln, m1, m2]


class _HCI(HandlerBase):
    def create_artists(self, legend, h, x0, y0, width, height, fontsize, trans):
        from matplotlib.lines import Line2D
        y = y0 + height / 2
        a, b, c = x0 + 0.1 * width, x0 + 0.9 * width, x0 + 0.5 * width
        ln = Line2D([a, b], [y, y], color=h.color, lw=ps.LW["ci"], transform=trans)
        mk = Line2D([c], [y], ls="none", marker="o", ms=ps.MS["main"], mfc=h.color, mec=h.color, mew=ps.LW["marker_edge"],
                    transform=trans)
        return [ln, mk]


class BarPairHandle:
    """범례: 연회색 막대 위에 진회색 막대를 겹친 b 막대 축 기호(그림과 같은 겹침 방식)."""
    def __init__(self, label=""):
        self._label = label

    def get_label(self):
        return self._label


class _HBarPair(HandlerBase):
    def create_artists(self, legend, h, x0, y0, width, height, fontsize, trans):
        from matplotlib.patches import Rectangle
        bh = 0.62 * height
        yb = y0 + (height - bh) / 2
        a = x0 + 0.05 * width
        light = Rectangle((a, yb), 0.90 * width, bh, facecolor=BAR_LIGHT, edgecolor="none", transform=trans)
        dark = Rectangle((a, yb), 0.40 * width, bh, facecolor=BAR_DARK, edgecolor="none", transform=trans)
        return [light, dark]


HANDLER_MAP = {DumbbellHandle: _HDumbbell(), CIHandle: _HCI(), BarPairHandle: _HBarPair()}

# ---------------------------------------------------------------- 3) 패널 b·c 공통 행
ROWS = ["E$_0$ + res. (n = 0)", "E$_0$ + res.", "Shrink + res. (S3)", "Shrink only", "Re-fit only", "Protocol", None, "Oracle branch"]
ROW_METHOD = ["residual", "residual", "residual", "refit", "refit", "residual", None, "oracle_branch"]
# 7행(None)은 패널마다 다른 탐색 행이다: (행 이름, 계열 색). 탐색 표기는 두 패널 모두 '†' + 기울임, 색은 계열 색 유지.
ROW7 = {"b": (f"Explore-gate{DAG} (b only)", "residual"), "c": (f"Offset only{DAG} (c only)", "refit")}
I7 = ROWS.index(None)
B1_KEYS = ["E0_resid_src", "always_E0_resid", "always_shrink_resid", "shrink_only", "refit_only", "protocol@{tau}",
           "explore_gate_agree@0.15", "oracle_branch"]
D1_KEYS = ["resid_src_only", "e0fix_resid", "shrink_resid_S3", "shrink_only", "refit_only", "protocol", "offset_only", "oracle"]
OFF = {0: 0.0, 3: 0.18, 10: -0.18}                  # n = 3 위, n = 10 아래
NVAR = {0: "solid", 3: "solid", 10: "dashed"}
RULE = dict(color="#b0b0b0", lw=0.3, zorder=0.5)     # 7행 구획 규선(0.3 pt)
WORST_MS, WORST_MEW = 4.0, 1.0                      # b 최악 대상 세로 틱(스펙 §1.1: 4 pt). c CI 는 끝 틱 없이 선만 그린다
BAR_LIGHT, BAR_DARK = "#c8c8c8", ps.GREY["text2"]   # b 막대: 점 추정 Δ > 0 대상 수(연), 블록 CI 하한 > 0 대상 수(진)


def _row_y():
    return np.arange(len(ROWS))[::-1].astype(float)


def _set_rows(ax, labels):
    y = _row_y()
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    for tl in ax.get_yticklabels():
        if tl.get_text() == "Protocol" or tl.get_text().startswith("Protocol"):
            tl.set_fontweight("bold")
        if DAG in tl.get_text():
            tl.set_fontstyle("italic")
    return y


def frame_row7(axes):
    """7행(b·c 에서 서로 다른 탐색 행) 위아래 0.3 pt 규선. 행 간격 1 이므로 경계 = y ± 0.5."""
    y7 = _row_y()[I7]
    for ax in axes:
        for yb in (y7 - 0.5, y7 + 0.5):
            ax.axhline(yb, **RULE).set_gid("row7_rule")


def _mstyle(method):
    s = ps.style_of(method)
    ms = ps.MS["main"] + (1.6 if method == "oracle_branch" else 0)
    return dict(color=s["color"], marker=s["marker"], mfc=s["mfc"], mec=s["mec"], ms=ms)


def panel_b(ax, axr, B):
    P = B["protocol"].copy()
    tau = B["tau_sel"]
    keys = [k.format(tau=tau) for k in B1_KEYS]
    XL = (-2.0, 6.5)
    y = _set_rows(ax, [r if r is not None else ROW7["b"][0] for r in ROWS])
    rows = []
    for i, k in enumerate(keys):
        q = P[P.method == k]
        if q.empty:
            raise MissingData(f"b1_protocol.csv method {k}")
        method = ROW_METHOD[i] if ROWS[i] is not None else ROW7["b"][1]    # explore-gate 는 잔차 포함(보라), 이름에 †
        st = _mstyle(method)
        for _, r in q.iterrows():
            n = int(r.n); yy = y[i] + OFF[n]
            m_, w_ = float(r.mean_d), float(r.worst_d)
            ls = ps.VARIANT_LS[NVAR[n]]
            w_draw = min(w_, XL[1])
            ax.plot([m_, w_draw], [yy, yy], color=st["color"], ls=ls, lw=0.8, zorder=2, solid_capstyle="butt")
            if w_ > XL[1]:
                ps.offscale_marker(ax, w_, yy, XL, "h", method)
            else:
                # 최악 대상 = 세로 틱 4 pt(스펙 §1.1). 빈 마커는 전역에서 블록 등가중 채점 전용이므로 쓰지 않는다
                ax.plot([w_], [yy], ls="none", marker="|", ms=WORST_MS, mew=WORST_MEW, color=st["color"], zorder=3)
            ax.plot([m_], [yy], ls="none", marker=st["marker"], ms=st.get("ms", ps.MS["main"]), mfc=st["mfc"], mec=st["mec"],
                    mew=ps.LW["marker_edge"], zorder=4)
            # 악화 대상 수 막대(n_worse/14)
            # 악화 대상 수(점 추정 Δ > 0, 연회색) 위에 블록 CI 하한 > 0 대상 수(진회색)를 겹친다
            axr.barh(yy, float(r.n_worse), height=0.30, color=BAR_LIGHT, edgecolor="none", zorder=2)
            axr.barh(yy, float(r.n_worse_ci), height=0.30, color=BAR_DARK, edgecolor="none", zorder=2.5)
            rows.append(dict(row=ROWS[i] or ROW7["b"][0], method=k, n=n, mean_d=m_, worst_d=w_, worst_target=r.worst_target,
                             n_worse=int(r.n_worse), n_worse_ci=int(r.n_worse_ci), n_targets=int(r.n_targets), gap_to_oracle=r.gap_to_oracle,
                             offscale=bool(w_ > XL[1])))
    ax.set_xlim(*XL)
    ax.set_xticks([-2, 0, 2, 4, 6])
    ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v, p: fmt_num(v, 0)))
    ax.set_xlabel("ΔRMSE vs physics (cm)")
    ps.zero_line(ax, "x")
    # 막대 축
    nt = int(P.n_targets.max())
    axr.set_xlim(0, nt); axr.set_ylim(ax.get_ylim())
    axr.set_yticks([]); axr.spines["left"].set_visible(False)
    axr.set_xticks([0, nt]); axr.set_xticklabels(["0", str(nt)])
    axr.axvline(0, color="#000000", lw=0.5, zorder=3)
    axr.set_xlabel("Targets\nΔ > 0", labelpad=1.5, linespacing=1.1)
    return pd.DataFrame(rows)


CAP_MS, CAP_MEW = 3.4, 0.6                          # c CI 끝 틱(pt). 마커(3.0 pt)보다 조금 길게, 얇게


def _marker_halfwidth_data(ax, marker, ms):
    """마커 반폭(가로)을 x 자료 단위로. 'D' 는 대각선 반폭 = 0.707·ms, 나머지는 0.5·ms(pt)."""
    f = {"D": 0.707, "d": 0.5, "*": 0.55}.get(marker, 0.5)
    px = f * ms * ax.figure.dpi / 72.0
    x0 = ax.transData.transform((0.0, 0.0))[0]
    return abs(ax.transData.inverted().transform((x0 + px, 0.0))[0])


def _pt_to_data(ax, pt):
    """가로 길이 pt → x 자료 단위."""
    px = pt * ax.figure.dpi / 72.0
    x0 = ax.transData.transform((0.0, 0.0))[0]
    return abs(ax.transData.inverted().transform((x0 + px, 0.0))[0])


def panel_c(ax, D):
    pred = D.attrs.get("prediction") or {}
    tau_main = float(pred.get("decision_rule", {}).get("tau_main", 0.15))
    XL = (-70.0, 5.0)
    y = _set_rows(ax, [r if r is not None else ROW7["c"][0] for r in ROWS])
    ax.set_xlim(*XL)                                  # 마커 반폭 → 자료 단위 환산에 필요
    rows = []
    for i, k in enumerate(D1_KEYS):
        q = D[D.method == k]
        if k == "protocol":
            q = q[np.isclose(q.tau, tau_main)]
        if q.empty:
            raise MissingData(f"d1_summary.csv method {k}")
        method = ROW_METHOD[i] if ROWS[i] is not None else ROW7["c"][1]   # offset_only = E 조정만(파랑), 이름에 †
        st = _mstyle(method)
        for _, r in q.iterrows():
            n = int(r.n); yy = y[i] + OFF.get(n, 0.0)
            ls = ps.VARIANT_LS[NVAR.get(n, "solid")]
            lo, hi = np.clip([r.rel_lo, r.rel_hi], *XL)
            # 파선 무늬가 추정점에서 시작하도록 추정점 → 각 경계로 반 구간 두 개를 그린다(마커 옆 끊김 방지)
            stub = _pt_to_data(ax, 1.6)                 # 파선 끝이 빈칸에 떨어져 CI 가 짧아 보이지 않게 끝 1.6 pt 는 실선
            for e in (lo, hi):
                ax.plot([r.rel_d, e], [yy, yy], color=st["color"], ls=ls, lw=ps.LW["ci"], zorder=2, solid_capstyle="butt",
                        dash_capstyle="butt")
                if ls != "-" and abs(e - r.rel_d) > stub:
                    ax.plot([e - np.sign(e - r.rel_d) * stub, e], [yy, yy], color=st["color"], ls="-", lw=ps.LW["ci"], zorder=2,
                            solid_capstyle="butt")
            ax.plot([r.rel_d], [yy], ls="none", marker=st["marker"], ms=st.get("ms", ps.MS["main"]), mfc=st["mfc"], mec=st["mec"],
                    mew=ps.LW["marker_edge"], zorder=4)
            # CI 는 끝 틱 없이 선만 그린다(세로 틱 = b 의 최악 대상 전용, 스펙 §1.1). 마커 반폭 안의 경계는 캡션에 수치로.
            hw = _marker_halfwidth_data(ax, st["marker"], st.get("ms", ps.MS["main"]))
            hidden = []
            for v in (r.rel_lo, r.rel_hi):
                if not (XL[0] <= v <= XL[1]):
                    ps.offscale_marker(ax, v, yy, XL, "h", method)
                elif abs(v - r.rel_d) <= hw:
                    hidden.append(v)
            rows.append(dict(row=ROWS[i] or ROW7["c"][0], method=k, n=n, tau=r.tau, rel_d=r.rel_d, rel_lo=r.rel_lo,
                             rel_hi=r.rel_hi, d_cm=r.blk_d_phys, d_lo_cm=r.blk_d_phys_lo, d_hi_cm=r.blk_d_phys_hi,
                             rmse_phys=r.rmse_phys, split_win=r.split_win, rep_win=r.rep_win, exploratory=bool(r.exploratory),
                             ci_kind=r.ci_kind, shrink_frac=r.get("shrink_frac", np.nan),
                             ci_hidden_by_marker=bool(hidden)))
    ax.set_xlim(*XL)
    ax.set_xticks([-60, -40, -20, 0])
    ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v, p: fmt_num(v, 0)))
    ax.set_xlabel("ΔRMSE vs physics (% of physics RMSE)")
    ps.zero_line(ax, "x")
    return pd.DataFrame(rows), tau_main


# ---------------------------------------------------------------- 4) D1 편향 점검(스펙 §2.7 함정 4, c 게재 조건)
BIAS_JSON = OUT / "_qa" / "Fig7_d1_bias_check.json"
PFR_MIN_FRAC = 0.10                                  # CCI PFR < 50 % 셀 비율 허용 상한


def d1_bias_check(draft: bool = False, write: bool = True) -> dict:
    """(i) 라벨 정의 (ii) 단위 (iii) 영구동토 존재 (iv) 조사 연도를 원자료에서 재확인한다.
    통과 기준: (i) 대상 라벨이 모두 CALM ALT(여름 말 최대 융해 깊이)이고 원천 방법이 기록됨,
    (ii) 원자료 단위 [cm] 이고 대상 값이 원자료 site-year 범위 안(단위 오류면 1–6 수준),
    (iii) CCI PFR < 50 % 셀 비율 ≤ 10 %, (iv) 연도 범위 결측 없음(값은 기록만)."""
    import re
    import xarray as xr
    from polar.fidelity import macro_region
    from polar.alt_dataset import parse_pangaea_calm, CALM_TAB
    tgt = "Mongolia_CAsia"
    F = rd(ROOT / "data/processed/fidelity_base_v3.csv")
    F["macro"] = macro_region(F)
    t = F[F.macro == tgt]
    srcs = t.source_id.value_counts().to_dict()
    # (i) 라벨 정의: PANGAEA 머리말의 ALT 정의 + 해당 사이트 관측 방법
    head = CALM_TAB.read_text(errors="replace").split("*/")[0]
    ald = [ln for ln in head.splitlines() if "Active layer depth" in ln]
    defn_ok = bool(ald) and "maximum thaw depth" in ald[0]
    meth = [re.sub(r".*Method: ", "", ln).strip() for ln in head.splitlines() if re.search(r"LOCATION: (Mongolia|Kazak)", ln)]
    n_temp = sum(m.startswith("Ground temperature") for m in meth)
    i_ok = defn_ok and set(srcs) == {"F4_calm_temp"} and n_temp == len(meth) > 0
    # (ii) 단위
    raw = parse_pangaea_calm()
    rawm = raw[raw.country.str.contains("Mongolia|Kazak", na=False)]
    unit_cm = bool(ald) and "[cm]" in ald[0]
    ii_ok = unit_cm and t.alt_cm.min() >= rawm.alt_cm.min() - 1e-6 and t.alt_cm.max() <= rawm.alt_cm.max() + 1e-6 and t.alt_cm.median() > 20
    # (iii) 영구동토 존재(CCI PFR 1997–2021 평균, 최근접 격자)
    pfr = xr.open_dataset(ROOT / "data/processed/cci_pfr_mean_1997_2021.nc").pfr_mean
    pv = pfr.sel(lat=xr.DataArray(t.lat.values, dims="k"), lon=xr.DataArray(t.lon.values, dims="k"), method="nearest").values
    frac_lt50 = float(np.mean(pv < 50))
    iii_ok = frac_lt50 <= PFR_MIN_FRAC
    # (iv) 조사 연도
    C = rd(ROOT / "data/processed/alt_calm_global_cell.csv")
    c = C[C.loc_id.isin(t.loc_id)]
    iv_ok = len(c) == len(t) and c.year_min.notna().all() and c.year_max.notna().all()
    # (v) 원천 학습 라벨 구성(F4_direct = 직접 탐침, src/polar/fidelity.py FIDELITY 주석): 조사 계열별 셀 수
    s4 = F[(F.source_id == "F4_direct") & (F.macro != tgt)]
    grp = np.where(s4.region.astype(str).str.startswith("ABoVE"), "ABoVE",
                   np.where(s4.region.astype(str).str.startswith("Lena"), "Lena", "other (CALM probe grids etc.)"))
    comp = pd.Series(grp).value_counts()
    train_comp = dict(n_cells=int(len(s4)), counts={k: int(v) for k, v in comp.items()},
                      frac={k: float(v / len(s4)) for k, v in comp.items()},
                      method_basis="source_id F4_direct = direct probing (ABoVE·Lena·CALM probe grids); CALM 지온·융해관 사이트는 "
                                   "F4_calm_temp 로 분리(PANGAEA 972777 Event Method 파싱). 셀별 관측 방법 열은 없음")
    runs = rd(H4 / ("d1_smoke_runs.csv" if draft and not (H4 / "d1_runs.csv").exists() else "d1_runs.csv"))
    bias_m = float(runs[runs.method == "physics"].bias_cm.mean()) / 100.0
    out = dict(
        target=tgt, n_cells=int(len(t)), label_sources=srcs, passed=bool(i_ok and ii_ok and iii_ok and iv_ok),
        i_label_definition=dict(ok=bool(i_ok), pangaea_definition=ald[0].split("COMMENT:")[-1].strip() if ald else None,
                                site_methods_ground_temperature=f"{n_temp}/{len(meth)}",
                                note="원천 학습 라벨(F4_direct)은 직접 탐침(ABoVE·Lena 현장 조사가 대부분, v 참조), 대상은 지온 프로파일 유도. "
                                     "같은 물리량(ALT), 다른 관측 방법"),
        v_training_labels=train_comp,
        ii_units=dict(ok=bool(ii_ok), unit="cm", target_alt_cm=[float(t.alt_cm.min()), float(t.alt_cm.median()), float(t.alt_cm.max())],
                      raw_site_year_alt_cm=[float(rawm.alt_cm.min()), float(rawm.alt_cm.median()), float(rawm.alt_cm.max())]),
        iii_permafrost=dict(ok=bool(iii_ok), pfr_median=float(np.median(pv)), n_pfr_lt50=int(np.sum(pv < 50)), frac_pfr_lt50=frac_lt50,
                            threshold=PFR_MIN_FRAC),
        iv_years=dict(ok=bool(iv_ok), year_min=int(c.year_min.min()), year_max=int(c.year_max.max()),
                      cell_median_span=[float(c.year_min.median()), float(c.year_max.median())], n_years_median=float(c.n_years.median())),
        physics_bias_m=bias_m,
    )
    out["log_text"] = (
        f"D1 편향 점검(Fig 7c 게재 조건, 스펙 §2.7 함정 4): (i) 대상 {len(t)}셀 라벨 전부 F4_calm_temp, PANGAEA 정의 'ALT, 여름 말 최대 "
        f"융해 깊이', 몽골·카자흐 사이트 관측 방법 {n_temp}/{len(meth)} 지온 측정. 원천(F4_direct)과 같은 물리량, 다른 관측 방법. "
        f"(ii) 단위 cm, 대상 값 {t.alt_cm.min():.0f}–{t.alt_cm.max():.0f} cm(중앙 {t.alt_cm.median():.0f})이 원자료 site-year 범위 "
        f"{rawm.alt_cm.min():.0f}–{rawm.alt_cm.max():.0f} cm 안. (iii) CCI PFR 중앙 {np.median(pv):.0f} %, PFR < 50 % 셀 "
        f"{int(np.sum(pv < 50))}/{len(t)}. (iv) 조사 연도 {int(c.year_min.min())}–{int(c.year_max.max())}(셀 중앙 "
        f"{c.year_min.median():.0f}–{c.year_max.median():.0f}). 판정 {'통과' if out['passed'] else '불통과'}: 물리식 편향 "
        f"{bias_m:.1f} m 는 자료 오류(단위·비영구동토)가 아니라 깊은 활동층 체제와 관측 방법 차이로 본다.")
    if write:
        BIAS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n")
    return out


# ---------------------------------------------------------------- 5) 조립
DEVIATIONS = [
    "a 를 사실 목록의 증거 기반 배포 절차(라벨 0 / 3–10 / 수백·전량)로 다시 그림. 스펙 §2.7 의 S3·Q2 분기(3라벨 E비 진단)는 "
    "F4 기각으로 권고 경로에서 빼고 회색 점선 곁가지 'tested, not supported (b)' 로만 둔다",
    "잎 배지('95 % CI if ≥ 8 blocks') 제거: 잎 태그 = 기대 Δ 범위 + 근거 그림 번호. CI 종류는 캡션",
    "라벨 3–10 잎 기대 Δ = b 의 'Shrink + res. (S3)' 대상별 블록 부트스트랩 평균(n = 3·10)의 오차 유형별 범위(그림 안 자기 근거). "
    "n = 3·10 중 Δ > +1 cm 대상(4개)은 범위에서 빼고 '> 1 cm 악화' 대상 수와 최대 악화 대상(CA-3)만 적는다(사실 목록 '−1.6 ~ −11, "
    "약 0 ± 1, 일부 악화(CA-3 +4)'와 같은 정의·예시, 대상별 값은 Supplementary Table 4). 채점 블록 < 5 대상은 캡션에 명시",
    "P3 잎: '구조 오차 지역 8개 중 6개가 수백 라벨 필요'(a2_minn E0 고정·전 블록·λ 0.25 에서 계산). 캐나다·AL-5 는 n = 0 에서 이미 "
    "물리식보다 나음을 캡션에 적는다(사실 목록 F1 정정)",
    "간선 라벨 'Hundreds or all'(스펙 'n ≥ tens'): A2 에서 구조 오차 지역은 수백 라벨이 필요",
    "그림 높이 144 mm(스펙 150): a 잎 상자를 글 높이에 맞춰 줄임",
    "b x범위 [−2, 6.5](스펙 [−2, 6]): 최악 대상 마커 6.0 cm 가 축 끝에 겹치지 않게",
    "b 막대: 연회색 = 점 추정 Δ > 0 대상 수, 진회색 = 블록 CI 하한 > 0 대상 수(n_worse_ci)",
    "b 막대 축 폭 6 mm·c 라벨 열과 간격 4 mm(스펙 8 mm + 2 mm): 막대 축 제목이 c 행 이름 열과 겹치지 않게",
    "b τ_sel = 'loto'(대상별 LOTO 선택 행 protocol@loto, 주 판정)",
    "b·c 7행은 패널별 탐색 행(b explore_gate_agree@0.15, c offset_only). 0.3 pt 규선·기울임·†, 색은 계열 색",
    "기호 분리: b 최악 대상 = 세로 틱 4 pt(스펙 §1.1 그대로, 빈 마커 = 블록 등가중 전역 규칙 보존), c CI = 끝 틱 없는 선(파선 끝 "
    "1.6 pt 실선). 마커보다 좁은 CI(n = 0 행)는 수치를 캡션에",
    "c 사전 예측 방향 띠 제거(검토: 모든 값이 음수라 전 구간 빗금은 정보 없음). 사전 등록 검정 행은 굵은 행 이름 + 캡션 문장",
    "b 막대 기호를 범례에 넣음(연회색 막대 위 진회색 막대 겹침 = BarPairHandle, 'Targets Δ > 0; dark, CI above 0')",
    "b·c 7행 행 이름에 '(b only)'·'(c only)' 표기: 두 패널의 나머지 행은 같은 순서",
    "c 게재 조건(편향 점검)은 _qa/Fig7_d1_bias_check.json 에 기록",
]


def build(draft: bool = False, force_fallback: bool = False, save=None):
    ps.use_paper()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    tags_list = build_workflow_tags(draft)
    tags = {d["leaf_id"]: d for d in tags_list}
    B = load_b1(draft)
    if B["tau_sel"] is None:
        raise MissingData("b1 τ 선택 기록 없음(대체 금지)")
    D = load_d1(draft)
    chk = d1_bias_check(draft, write=save is None)
    c_main = chk["passed"] and not force_fallback     # 불통과면 c 는 보충 자료로, b 를 180 mm 로(스펙 §2.7 c 게재 조건)

    H = 72.0 + AX_H + 6.0                          # 행 1(a) 위 6 mm = 패널 문자 자리
    fig = ps.paper_figure(180, H)
    Y2, H2_ = 13.0, 44.0
    if c_main:
        L = dict(b=[0, Y2, 30, 47, H2_], bars=[80, Y2, 6, H2_], c=[90, Y2, 25, 63, H2_])
    else:
        L = dict(b=[0, Y2, 30, 132, H2_], bars=[166, Y2, 10, H2_])
    ax_b = ps.slot_mm(fig, *L["b"])
    ax_r = ps.axes_mm(fig, *L["bars"])
    ax_c = ps.slot_mm(fig, *L["c"]) if c_main else None
    ax_a = ps.slot_mm(fig, 0, 72, 5, AX_W, AX_H)

    draw_tree(ax_a, tags)
    src_b = panel_b(ax_b, ax_r, B)
    src_c, tau_main = panel_c(ax_c, D) if c_main else (None, None)
    frame_row7([a for a in (ax_b, ax_r, ax_c) if a is not None])

    grey = "#4d4d4d"
    handles = [
        ps.method_handle("residual", "Includes residual ML", line=False),
        ps.method_handle("refit", "E shrink or re-fit only", line=False),
        Line2D([], [], ls="none", marker="*", ms=ps.MS["main"] + 1.6, color="#000000", label="Oracle branch (post hoc)"),
        Line2D([], [], color=grey, lw=0.8, ls="-", label="n = 3"),
        Line2D([], [], color=grey, lw=0.8, ls=ps.VARIANT_LS["dashed"], label="n = 10"),
        DumbbellHandle(grey, "o", ps.MS["main"], label="Mean to worst target (b)"),
    ]
    handles.append(BarPairHandle(label="Targets Δ > 0; dark, CI above 0 (b)"))
    if c_main:
        handles.append(CIHandle(grey, label="95 % block CI (c)"))
    else:
        handles.append(Line2D([], [], ls="none", marker=">", ms=ps.MS["main"] + 0.6, color=grey, label="Off-scale"))
    ps.legend_below(fig, handles, (4, 62.5, 172, 7.5), ncol=4, handler_map=HANDLER_MAP)
    ps.label_panels([a for a in (ax_a, ax_b, ax_c) if a is not None], "abc")

    # ---- 캡션 수치(모두 자료에서 읽음)
    pf, p3 = tags["PF"], tags["P3"]
    P = B["protocol"]
    nt = int(P.n_targets.max())
    pr = P[(P.method == f"protocol@{B['tau_sel']}") & (P.n == 3)].iloc[0]
    s3 = P[(P.method == "always_shrink_resid") & (P.n == 3)].iloc[0]
    f4_ok = (pr.n_worse < s3.n_worse) and (pr.gap_to_oracle <= 0.5)
    f4 = (f"F4 (pre-registered: fewer worsened targets than S3, mean Δ within 0.5 cm of the oracle branch) "
          f"{'supported' if f4_ok else 'not supported'}: at n = 3, {int(pr.n_worse)} of {nt} targets worsen "
          f"({int(pr.n_worse_ci)} with block CI above 0) versus {int(s3.n_worse)} for S3; mean Δ {fmt_num(pr.mean_d, 2)} cm, "
          f"{fmt_num(pr.gap_to_oracle, 2)} cm above the oracle.")
    ob = src_b[src_b.offscale]
    off_b = ("; arrowheads, off-scale (" + ", ".join(f"{fmt_num(v, 1)}" for v in ob.worst_d) + " cm)."
             if len(ob) else "")
    hid_c, cap_c = "", ""
    if c_main:
        if src_c.ci_hidden_by_marker.any():
            hh = src_c[src_c.ci_hidden_by_marker]
            hid_c = " CI hidden by symbol: " + "; ".join(
                f"{r.row.replace('$_0$', '0').replace(' (', ', ').rstrip(')')}, {fmt_ci(r.rel_d, r.rel_lo, r.rel_hi, 1)} %" for r in hh.itertuples()) + "."
        rmse_phys = float(D.rmse_phys.iloc[0])
        lf = D.attrs["prediction"]["label_free_metrics"]
        shrink_all = bool(np.all(src_c[src_c.method == "protocol"].shrink_frac >= 1.0))
        pc = src_c[(src_c.method == "protocol") & (src_c.n == 3)].iloc[0]
        f11_ok = pc.d_hi_cm < 0
        iv = chk["iv_years"]
        trc = chk["v_training_labels"]
        tr_n = trc["n_cells"]
        tr_frac = trc["frac"].get("ABoVE", 0.0) + trc["frac"].get("Lena", 0.0)
        cap_c = (
            f" c, D1 holdout (Mongolia, Central Asia; {lf['n_cells']} cells, {lf['n_blocks']} blocks); Δ as % of "
            f"physics RMSE ({rmse_phys:.1f} cm). Target labels, ground-temperature thaw depths "
            f"({iv['year_min']}–{iv['year_max']}); training labels, direct probing (ABoVE, Lena: "
            f"{100 * tr_frac:.0f} % of {tr_n:,} cells); region and label definition confounded. "
            f"Bold row, pre-registered prediction Δ < 0. F11 {'supported' if f11_ok else 'not supported'} with this "
            f"caveat: protocol n = 3 "
            f"(τ {'0.10–0.20 identical' if shrink_all else f'{tau_main:.2f}'}), "
            f"{fmt_ci(pc.d_cm, pc.d_lo_cm, pc.d_hi_cm, 1)} cm." + hid_c)
    caption = dict(
        definition=(
            "Fig. 7 | Deployment procedure and its confirmatory tests. ΔRMSE, method minus physics-anchor RMSE (E0 from "
            "source regions) on held-out B blocks; negative is better. Error type, oracle |log(E_own/E0)|: level ≥ 0.20, "
            "structure < 0.15. E0 + res., anchor plus CatBoost residual ML (λ 0.25); Shrink, E shrunk toward E0 "
            f"(κ = 10); Re-fit, unshrunk E; S3, shrink plus residual ML; Protocol, the dashed rule in a. {DAG} Exploratory (post hoc). "
            f"Under {MIN_BLOCKS} scoring blocks: {', '.join(pf['low_block_targets'])}."),
        statistics=(
            f"b, {nt} targets, 3 splits × 20 repeats × 2 seeds, k-medoid labels (nested at n = 10); no CI on means. "
            + ("c, 95 % block-bootstrap CI (1,000 resamples of scoring blocks within split)."
               + " " if c_main else "")
            + f4),
        panels=(
            "a, Solid arrows, recommended; dashed, pre-registered branch (b). Tags, expected Δ, "
            "evidence; 3–10 tag, range of S3 target means in b (n = 3, 10) per error type, without targets worse "
            f"by > {EXC_CUT:.0f} cm. Bottom leaf: {p3['n_struct_slow']} of {p3['n_struct']} structure targets "
            f"lack n* ≤ 40 (A2, E0 fixed); {', '.join(p3['struct_fast'])} beat physics at n = 0. "
            "b, τ per target by leave-one-target-out. Filled, mean; tick, worst target"
            + off_b + " Bars, upper n = 3." + cap_c
            + ("" if c_main else " External holdout (D1) in Supplementary Fig. 7 (label checks not passed).")),
        data=(
            "Source: data/processed/h4/b1_protocol.csv, b1_summary.csv, b1_meta.json, c2_coverage.csv, a1_recipe_table.csv, a2_minn.csv"
            + (", d1_summary.csv" if c_main else "")
            + "; outputs/figures/paper/workflow_tags.json. τ per target: b1_summary.csv tau_loto, rule b1_meta.json. "
            "Per-target values: Supplementary Tables 4, 7."),
    )
    keep = ("leaf_id", "fig_ref", "source_csv", "filter", "stat", "lo", "hi", "est", "n3", "n10", "lv_lo", "lv_hi", "st_lo",
            "st_hi", "exc", "low_block_targets", "n_struct", "n_struct_slow", "struct_fast", "ca3", "n_worse", "n_worse_ci", "n_targets", "gap_to_oracle", "lena_nstar", "render", "ci_kind")
    src_a = pd.DataFrame([{k: d.get(k) for k in keep} for d in tags_list])
    lay = {"a": [5, 72, AX_W, AX_H], "b": [30, Y2, L["b"][3], H2_], "b_bars": L["bars"], "legend": [4, 62.5, 172, 7.5]}
    if c_main:
        lay["c"] = [L["c"][0] + L["c"][2], Y2, L["c"][3], H2_]
    spec = dict(
        intent="증거 기반 배포 절차(라벨 0 / 3–10 / 수백·전량)와 확인적 검증. 사전 등록 2단계 분기는 B1 에서 기각(F4), "
               "외부 홀드아웃 D1 은 라벨 정의 교락을 밝힌 채 지지(F11)",
        panels=3 if c_main else 2,
        layout_mm=lay,
        assets=["data/processed/h4/b1_protocol.csv", "data/processed/h4/b1_summary.csv", "data/processed/h4/b1_meta.json",
                "data/processed/h4/c2_coverage.csv", "data/processed/h4/a1_recipe_table.csv", "data/processed/h4/a2_minn.csv",
                "data/processed/h4/d1_summary.csv", "data/processed/h4/d1_prediction.json", "data/processed/h4/d1_runs.csv",
                "outputs/figures/paper/workflow_tags.json", "outputs/figures/paper/_qa/Fig7_d1_bias_check.json"],
        ci_kinds=dict(a="태그별(P0 coverage 블록, PF 대상별 블록 평균 범위, P3 strat_AB4)", b="none(평균·최악), 막대 진회색 = 블록 CI",
                      c="block(끝 틱)"),
        verdicts=dict(F4="not supported" if not f4_ok else "supported", F11=("supported (label-definition confound)"
                                                                             if c_main and f11_ok else "see caption")),
        placeholders=[],                               # 본 실행 자료로 렌더(자리표 없음). 이전 항목의 자리표 목록을 덮어쓴다
        tau_sel=str(B["tau_sel"]), d1_tau=tau_main,
        d1_bias_check=dict(passed=chk["passed"], c_in_main=c_main, record="outputs/figures/paper/_qa/Fig7_d1_bias_check.json"),
        checks="간선·노드 교차 검사(shapely) 통과 시에만 렌더; c 는 D1 편향 점검 통과 시에만 본문",
        deviations=DEVIATIONS,
    )
    sources = {"a": src_a, "b": src_b}
    if c_main:
        sources["c"] = src_c
    return (save or save_paper)(fig, "Fig7_workflow", caption=caption, spec=spec, draft=draft, sources=sources)
