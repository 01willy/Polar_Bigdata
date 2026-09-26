"""부록 그림 S1–S12 와 Supplementary Table ST1–ST7. 정본 스펙: figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §4.

build(draft) 가 항목별 생성 함수를 차례로 부른다. 한 항목의 자료가 없으면(MissingData) 그 항목만 '대기'로 기록하고
나머지를 계속 만든다. 상태 목록은 outputs/figures/paper/_qa/supp_status.json 과 CAPTIONS.md 의
'## Supplementary_index' 절에 남긴다.

산출
  그림  outputs/figures/paper/FigS{N}_*.pdf/.png(600 dpi, 180 mm) + CAPTIONS.md 절 + source_data/FigS{N}_<panel>.csv
  표    outputs/figures/paper/supp_tables/TableS{N}_*.csv(전체 행) + .tex(booktabs, 본문 게재용 요약) + CAPTIONS.md 절

감사 조치(audit_plan) 반영
  - H27 은 이진 달성 서술(달성 4/14, Mann-Whitney 정확 p, 완전 분리)과 Kendall τ_b 로만 보고한다. ρ −0.84 주장은 쓰지 않는다.
  - 절단 대상은 '> n_max (max tested)' 로만 쓰고 대입값(2·n_max)은 표·그림에 쓰지 않는다.
  - H24 는 '채택'이 아니라 탐색적(n = 3 한정 부분 지지)으로 표기한다. H23·H24 는 Holm 가족 밖이다.
  - h4 자료의 그림 CI 는 블록 부트스트랩만 쓴다. 행 재표집 CI 는 ST3 로 보낸다.
오차 유형 3단 분류(감사 조치 9): 구조 |log E비| < 0.15, 수준 ≥ 0.2, 그 사이 = 중간. 오라클 E_own 기반 사후 지표다.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _common import *                                          # noqa: F401,F403
from _common import (ps, ROOT, H2, H3, H4, M1, OUT, QA_DIR, PROC, MissingData, rd, load_h4, load_h25_curve,
                     load_h25_targets, h25_series, h25_allA_ref, load_subregions, save_paper,
                     write_caption, caption_problems, caption_text, abslogE_map, load_a2_minn, load_c2_coverage,
                     load_b1, load_d1, load_b4, load_m1_tests, h4_path, fmt_ci)

TAB_DIR = OUT / "supp_tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)
STRUCT_MAX, LEVEL_MIN = 0.15, 0.20
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
SUB10 = ["AL-1", "AL-2", "AL-3", "AL-4", "AL-5", "AL-6", "CA-2", "CA-3", "LE-1", "LE-2"]
STATUS: list[dict] = []


# ================================================================ 공용 도우미
def err_class(v: float) -> str:
    """감사 조치 9 의 통일 분류(오라클 E_own 기반 사후 지표)."""
    if v is None or not np.isfinite(v):
        return ""
    return "structure" if v < STRUCT_MAX else ("level" if v >= LEVEL_MIN else "intermediate")


def _symlog_y(ax, lim, ticks=(-5, -2, -1, 0, 1, 2, 5), linscale=0.45):
    """Fig 3 과 같은 symlog Δ 축(linthresh 2 cm, 선형 구간 축소)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_yscale("symlog", linthresh=2.0, linscale=linscale)
    ax.set_ylim(*lim)
    ax.yaxis.set_major_locator(FixedLocator(list(ticks))); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))


def _minus_fmt(nd=0):
    return matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, nd))


def _dodge_rows(x_mm, min_gap_mm: float = 1.5) -> np.ndarray:
    """절단(미달성) 표식 줄 배정: x 순으로 훑어 같은 줄의 앞 점과 min_gap_mm 보다 가까우면 다음 줄로 보낸다(x 값은 바꾸지 않음)."""
    x_mm = np.asarray(x_mm, float)
    rows = np.zeros(len(x_mm), int)
    last = []                                                   # 줄별 마지막 x
    for i in np.argsort(x_mm, kind="stable"):
        for r_, lx in enumerate(last):
            if x_mm[i] - lx >= min_gap_mm:
                rows[i] = r_; last[r_] = x_mm[i]; break
        else:
            rows[i] = len(last); last.append(x_mm[i])
    return rows


def _fig_text(fig, x_mm, y_mm, s, **kw):
    fx, fy = ps.mm_to_fig(fig, x_mm, y_mm)
    d = dict(fontsize=ps.FS["label"], ha="center", va="bottom")
    d.update(kw)
    return fig.text(fx, fy, s, **d)


def _record(item: str, status: str, files=None, note: str = ""):
    STATUS.append(dict(item=item, status=status, files=files or [], note=note))


def _tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                 ("{", r"\{"), ("}", r"\}"), ("−", r"$-$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("×", r"$\times$"),
                 ("λ", r"$\lambda$"), ("κ", r"$\kappa$"), ("α", r"$\alpha$"), ("τ", r"$\tau$"), ("Δ", r"$\Delta$"),
                 ("ρ", r"$\rho$"), ("|", r"$|$"), (">", r"$>$"), ("<", r"$<$"), ("†", r"$^\dagger$")):
        s = s.replace(a, b)
    return s


def write_table(name: str, full: pd.DataFrame, show: pd.DataFrame | None, caption: dict, col_fmt: str | None = None) -> dict:
    """전체 행 CSV + 게재용 LaTeX(booktabs) + CAPTIONS 절. 캡션 규칙 위반이면 CAPTIONS 는 쓰지 않는다."""
    p_csv = TAB_DIR / f"{name}.csv"
    full.to_csv(p_csv, index=False)
    show = full if show is None else show
    cols = list(show.columns)
    col_fmt = col_fmt or ("l" * min(2, len(cols)) + "r" * max(0, len(cols) - 2))
    head = " & ".join(_tex_escape(c) for c in cols) + r" \\"
    longt = len(show) > 40                                      # 긴 표는 longtable(쪽 넘김 시 머리행 반복)
    env = "longtable" if longt else "tabular"
    lines = [r"\begin{" + env + "}{" + col_fmt + "}", r"\toprule", head, r"\midrule"]
    if longt:
        lines += [r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead"]
    for _, r in show.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else _tex_escape(ps.fmt_num(float(v), 2)))
            else:
                cells.append("" if (v is None or (isinstance(v, float) and np.isnan(v))) else _tex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{" + env + "}"]
    p_tex = TAB_DIR / f"{name}.tex"
    p_tex.write_text("\n".join(lines) + "\n")
    probs = caption_problems(caption)
    _, wc = caption_text(caption)
    if not probs:
        write_caption(name, caption)
        from _common import update_figure_spec
        import datetime as _dt
        update_figure_spec(dict(id=f"paper_{name.lower()}", script="scripts/4_visualization/paper_figs.py --only supp",
                                exports=[str(p_csv.relative_to(ROOT)), str(p_tex.relative_to(ROOT))], kind="supplementary table",
                                rows_csv=int(len(full)), rows_tex=int(len(show)), generated=_dt.date.today().isoformat(), qa="pass",
                                style="booktabs LaTeX(긴 표 longtable), 음수 U+2212, 절단 '> n_max' 표기"))
    res = dict(name=name, ok=not probs, fails=probs, words=wc, paths=dict(csv=str(p_csv.relative_to(ROOT)), tex=str(p_tex.relative_to(ROOT))))
    (QA_DIR / f"{name}_qa.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"[supp] {name}: {len(full)} rows (tex {len(show)}), caption words {wc['total']}, {'ok' if not probs else probs}")
    return res


def load_h27(tag: str = "c"):
    """H27 규칙 통계·대상 표. tag 'c' = h25b 재실행 기반(블록 CI 'main' 정의 포함, 우선), 'b' = h25 기반, 'x' = 같은 상위 지역
    원천 제외(하위 지역 10). 절단 행 be_n → NaN(대입값 제거). 반환 (R, T, 파일 접두)."""
    for t in ([tag] if tag != "c" else ["c", "b"]):
        pr, pt = H3 / f"h27{t}_rule.csv", H3 / f"h27{t}_targets.csv"
        if pr.exists() and pt.exists():
            R, T = rd(pr), rd(pt)
            c = T.censored.astype(str).str.lower().isin(["true", "1"])
            T["censored"] = c
            T.loc[c, "be_n"] = np.nan
            return R, T, f"h27{t}"
    raise MissingData(f"h3/h27{tag}_rule.csv")


def fmt_censored(n, censored, n_max, rl_thresh=40) -> str:
    """'10' 또는 '> 320 (max tested)'. 대입값을 쓰지 않는다."""
    if bool(censored) or not np.isfinite(n):
        rl = np.isfinite(n_max) and n_max < rl_thresh
        return f"> {int(n_max)} (max tested{', range-limited' if rl else ''})" if np.isfinite(n_max) else "not reached"
    return f"{int(n)}"


# ================================================================ S3 하위 지역 정의 지도
MAPS_S3 = {
    "Alaska": dict(ll=(-168.0, -141.0, 58.5, 71.8), lon0=-154.0, glat=5, glon=10, bar=200, res="50m", lats=[60, 65, 70], inset="lower right"),
    "Canada": dict(ll=(-136.0, -70.0, 58.0, 79.0), lon0=-100.0, glat=5, glon=10, bar=500, res="50m", lats=[60, 70], inset="lower right"),
    "Lena": dict(ll=(122.0, 131.5, 70.6, 73.9), lon0=127.0, glat=1, glon=5, bar=100, res="10m", lats=[71, 72, 73]),
}


def _subregion_cells():
    """h25_label_budget.make_subregions 와 같은 규칙(블록 중심 k-means, random_state 0)으로 셀별 하위 지역을 재구성하고
    h3/subregions.csv 의 블록·셀 수와 일치하는지 검사한다(불일치면 ValueError)."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from sklearn.cluster import KMeans
    from polar.m1_core import load_base
    df = load_base(PROC)
    S = load_subregions()
    rows = []
    for reg, k in (("Alaska", 6), ("Canada", 3), ("Lena", 2)):
        idx = np.where(df.macro.values == reg)[0]
        bt = df.iloc[idx].groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size")).reset_index()
        X = np.c_[bt.lat.values, bt.lon.values * np.cos(np.radians(bt.lat.values))]
        import sys
        hook = sys.unraisablehook
        sys.unraisablehook = lambda *_: None                # threadpoolctl 의 라이브러리 탐색 경고(무해) 억제
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=np.sqrt(bt.n.values))
        sys.unraisablehook = hook
        lab = km.labels_
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])
        name_of = {c: f"{reg[:2].upper()}-{r + 1}" for r, c in enumerate(order)}
        bt["subregion"] = [name_of[c] for c in lab]; bt["parent"] = reg
        rows.append(bt)
    B = pd.concat(rows, ignore_index=True)
    chk = B.groupby("subregion").agg(n_blocks=("block", "size"), n_cells=("n", "sum"))
    ref = S.set_index("subregion")[["n_blocks", "n_cells"]]
    if not chk.reindex(ref.index).astype(int).equals(ref.astype(int)):
        raise ValueError(f"하위 지역 재구성 불일치:\n{chk}\n{ref}")
    return B, S


def _fit_map_ax(fig, rect_mm, cfg):
    """축 상자 비율에 맞춘 극 입체 지도(true scale 70°N). fig5 의 _map_ax 와 같은 규칙."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    proj = ccrs.NorthPolarStereo(central_longitude=cfg["lon0"], true_scale_latitude=70)
    x, y, w, h = rect_mm
    ax = ps.axes_mm(fig, x, y, w, h, projection=proj)
    lo0, lo1, la0, la1 = cfg["ll"]
    lons = np.r_[np.linspace(lo0, lo1, 60), np.full(30, lo1), np.linspace(lo1, lo0, 60), np.full(30, lo0)]
    lats = np.r_[np.full(60, la0), np.linspace(la0, la1, 30), np.full(60, la1), np.linspace(la1, la0, 30)]
    if cfg.get("fit_lonlat") is not None:                     # 자료 점 범위(+ 여백 비율)에 맞춘다
        lons, lats = (np.asarray(v, float) for v in cfg["fit_lonlat"])
    P = proj.transform_points(ccrs.PlateCarree(), lons, lats)
    x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    cx, cy, dx, dy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    mg = 1.0 + 2 * cfg.get("fit_margin", 0.0)
    dx, dy = dx * mg, dy * mg
    if dx / dy < w / h:
        dx = dy * w / h
    else:
        dy = dx * h / w
    ax.set_extent((cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2), crs=proj)
    ax.add_feature(cfeature.LAND.with_scale(cfg["res"]), facecolor=ps.GREY["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="white", edgecolor=ps.GREY["edge"], lw=0.25, zorder=0.6)
    ax.add_feature(cfeature.COASTLINE.with_scale(cfg["res"]), edgecolor=ps.GREY["edge"], lw=0.4 if cfg["res"] == "50m" else 0.3, zorder=1)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, lw=0.4, color=ps.GREY["grid"], alpha=0.6, zorder=0.5,
                 xlocs=np.arange(-180, 181, cfg["glon"]), ylocs=np.arange(40, 91, cfg["glat"]))
    ax.spines["geo"].set_linewidth(0.5)
    ax._paper_proj = proj
    sb = ps.scale_bar(ax, cfg["bar"], loc=(0.05, 0.05))
    sb.set_bbox(dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    if cfg.get("inset", "upper right"):
        ps.inset_locator(fig, ax, cfg["ll"], frac=0.24, corner=cfg.get("inset", "upper right"))
    return ax


def _lat_labels(ax, cfg):
    import cartopy.crs as ccrs
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    lo0 = cfg["ll"][0]
    for la in cfg["lats"]:
        px, py = ax._paper_proj.transform_point(lo0 + 0.08 * (cfg["ll"][1] - cfg["ll"][0]), la, ccrs.PlateCarree())
        if not (x0 <= px <= x1 and y0 + 0.12 * (y1 - y0) <= py <= y1 - 0.08 * (y1 - y0)):
            continue
        ax.text(px, py, f"{la}°N", fontsize=ps.FS["annot"], color=ps.GREY["text2"], ha="left", va="bottom", transform=ax.transData,
                zorder=4, clip_on=True, bbox=dict(boxstyle="square,pad=0.1", facecolor=ps.GREY["land"], edgecolor="none", alpha=0.9))


BLOB_R_MM = 2.6
LABEL_OFF_S3 = {   # 하위 지역 이름 위치(외곽선 중심 기준 mm 이동, 렌더 확인 후 고정)
    "AL-1": (3, 6), "AL-2": (6, -5), "AL-3": (-10, 2), "AL-4": (-8, -3), "AL-5": (2, -8), "AL-6": (8, -6),
    "CA-1": (0, 5), "CA-2": (0, 7), "CA-3": (7, -4), "LE-1": (0, 7), "LE-2": (6, -7),
}


def build_S3(draft: bool = False):
    """하위 지역 11개(Alaska 6, Canada 3, Lena 2) 정의 지도. 블록 = 면적 비례 원, 하위 지역 = 블록 중심 볼록 껍질 + 이름."""
    import cartopy.crs as ccrs
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    B, S = _subregion_cells()
    ps.use_paper()
    fig = ps.paper_figure(180, 90)
    rects = {"Alaska": (0, 14, 70, 70), "Canada": (72, 14, 62, 70), "Lena": (136, 14, 44, 70)}
    axes, src_rows = [], []
    alr = abslogE_map()
    for reg, rect in rects.items():
        cfg = MAPS_S3[reg]
        ax = _fit_map_ax(fig, rect, cfg)
        _lat_labels(ax, cfg)
        q = B[B.parent == reg]
        ps.density_circles(ax, q.lon.values, q.lat.values, q.n.values, color=ps.COLOR["refit"], alpha=0.5)
        proj = ax._paper_proj
        W_mm = rect[2]
        x0, x1 = ax.get_xlim()
        m_per_mm = (x1 - x0) / W_mm
        for sname, g in q.groupby("subregion"):
            P = proj.transform_points(ccrs.PlateCarree(), g.lon.values, g.lat.values)[:, :2]
            # 외곽선 = 블록 중심 원(반지름 R mm) 합집합을 닫힘 연산으로 매끈하게(가까운 블록끼리만 이어진다)
            R = BLOB_R_MM * m_per_mm
            blob = unary_union([Point(p).buffer(R, resolution=24) for p in P]).buffer(1.5 * R).buffer(-1.5 * R)
            parts = list(blob.geoms) if blob.geom_type == "MultiPolygon" else [blob]
            big = max(parts, key=lambda g_: g_.area)
            for poly in parts:
                if poly is not big:                           # 떨어진 조각: 0.4 R 줄인 점선 외곽(이웃 하위 지역 조각과 겹침 방지)
                    sm_ = poly.buffer(-0.4 * R)
                    if not sm_.is_empty:
                        poly = max(list(sm_.geoms), key=lambda g_: g_.area) if sm_.geom_type == "MultiPolygon" else sm_
                hx, hy = poly.exterior.xy
                ax.plot(hx, hy, color=ps.GREY["text2"], lw=0.6 if poly is big else 0.5, zorder=2.6, transform=proj,
                        ls="-" if poly is big else (0, (2.0, 1.2)))
                if poly is not big:                           # 떨어진 조각에도 이름(보통 굵기)을 달아 소속을 밝힌다
                    bxx = poly.bounds                         # 주 조각 반대쪽에 두고, 축 밖(약 8 mm 필요)이면 뒤집는다
                    right = poly.centroid.x >= big.centroid.x
                    if right and bxx[2] + 9 * m_per_mm > x1:
                        right = False
                    if not right and bxx[0] - 9 * m_per_mm < x0:
                        right = True
                    ax.text(bxx[2] + 0.8 * m_per_mm if right else bxx[0] - 0.8 * m_per_mm, (bxx[1] + bxx[3]) / 2, sname,
                            transform=proj, fontsize=ps.FS["annot"],
                            ha="left" if right else "right", va="center", zorder=6, color=ps.GREY["text2"],
                            bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=0.7))
            dx, dy = LABEL_OFF_S3.get(sname, (0, 5))
            nrm = np.hypot(dx, dy)
            ux, uy = dx / nrm, dy / nrm
            # 이름 위치: 가장 큰 조각의 중심에서 (dx, dy) 방향으로 외곽선 밖 1.5 mm, 추가 이동 없음
            cx, cy = big.centroid.x, big.centroid.y
            ray = LineString([(cx, cy), (cx + ux * 200 * m_per_mm, cy + uy * 200 * m_per_mm)])
            ip = ray.intersection(big.exterior)
            if ip.is_empty:
                bx, by = cx, cy
            else:
                pts = [ip] if ip.geom_type == "Point" else list(getattr(ip, "geoms", [ip]))
                far = max(pts, key=lambda p_: np.hypot(p_.x - cx, p_.y - cy))
                bx, by = far.x, far.y
            lx, ly = bx + ux * 1.2 * m_per_mm, by + uy * 1.2 * m_per_mm
            ha = "left" if ux > 0.38 else ("right" if ux < -0.38 else "center")
            va = "bottom" if uy > 0.38 else ("top" if uy < -0.38 else "center")
            ax.text(lx, ly, sname, transform=proj, fontsize=ps.FS["annot"], ha=ha, va=va, zorder=6, fontweight="bold",
                    color="#000000", bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.8))
            r = S[S.subregion == sname].iloc[0]
            src_rows.append(dict(panel="abc"[list(rects).index(reg)], subregion=sname, parent=reg, n_blocks=int(r.n_blocks),
                                 n_cells=int(r.n_cells), centroid_lat=float(r.lat), centroid_lon=float(r.lon),
                                 abs_logE=alr.get(sname, np.nan), error_type=err_class(alr.get(sname, np.nan)),
                                 analysed=sname != "CA-1"))
        ax.text(0.025, 0.975, reg if reg != "Lena" else "Lena Delta", transform=ax.transAxes, ha="left", va="top",
                fontsize=ps.FS["label"], zorder=6, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
        axes.append(ax)
    handles = ps.size_legend_handles((10, 100, 1000), color=ps.COLOR["refit"], alpha=0.5, fmt="{:,} cells per block")
    handles.append(Line2D([], [], color=ps.GREY["text2"], lw=0.6, label="Subregion outline (block centroids)"))
    handles.append(Line2D([], [], color=ps.GREY["text2"], lw=0.5, ls=(0, (2.0, 1.2)), label="Detached part"))
    ps.legend_below(fig, handles, (0, 0, 180, 9), ncol=5)
    ps.label_panels(axes, "abc", dy_mm=1.0)
    ncell = S.set_index("subregion").n_cells
    caption = dict(
        definition=("Subregion definition. Subregions are k-means clusters of scoring-block centroids within each parent "
                    "region (Alaska k = 6, Canada k = 3, Lena Delta k = 2; square-root cell weights; labels not used), named north to south. "
                    "Each subregion is a transfer target whose source excludes cells within 100 km."),
        statistics=(f"CA-1 ({int(ncell['CA-1'])} cells, 3 A-block labels) is shown but excluded from analysis, leaving 10 subregions "
                    "and 14 analysed targets with the 4 main regions. Cell and block counts, E_own, E0 and |log(E_own/E0)| are in "
                    "Supplementary Table 1."),
        panels=("a, Alaska. b, Canada. c, Lena Delta. Circles are scoring blocks with area proportional to cells per block; "
                "outlines are the smoothed union of 2.6 mm discs around the block centroids of one subregion; detached parts are drawn dashed and 1 mm tighter, carry the subregion name in plain type and are unrelated to any neighbouring outline they approach. Polar stereographic projection, true scale at 70°N; "
                "insets show location."),
        data="Source data: data/processed/h3/subregions.csv; clustering reproduced from data/processed/fidelity_base_v3.csv.",
    )
    spec = dict(intent="하위 지역 11개 정의 지도(Alaska·Canada·Lena), 블록 면적 비례 원 + 하위 지역 볼록 껍질 외곽선 + 이름",
                panels=3, assets=["data/processed/h3/subregions.csv", "data/processed/fidelity_base_v3.csv"],
                script="scripts/4_visualization/paper_figs.py --only supp",
                check="make_subregions 재현: 11개 하위 지역 블록·셀 수가 subregions.csv 와 일치")
    return save_paper(fig, "FigS3_subregions", caption=caption, spec=spec, draft=False,
                      sources={"abc": pd.DataFrame(src_rows)})


# ================================================================ S4 하위 지역 계단 + n = 3·10 포레스트
S4_SERIES = [
    ("S1", "kmedoid", 0.0, "refit", None, "E re-fit, k-medoid labels"),
    ("S1", "random", 0.0, "refit", "dashed", "E re-fit, random labels"),
    ("S2", "kmedoid", 1.0, "augment", None, "+ pseudo-label augmentation"),
    ("S3", "kmedoid", 0.25, "residual", None, "+ residual ML (λ = 0.25)"),
]
S4_YLIM = (-9.0, 7.0)
S4_FLIM = (-12.0, 6.0)


def build_S4(draft: bool = False):
    C = load_h25_curve()
    T = load_h25_targets()
    alr = dict(zip(T.target, T.abslogE))
    pool = dict(zip(T.target, T.n_A))
    blk = C.attrs.get("ci_kind") == "block"
    ycol = "blk_d_phys" if blk else "d_phys_mean"
    ps.use_paper()
    fig = ps.paper_figure(180, 168)
    subs = sorted(SUB10, key=lambda t: alr.get(t, np.inf))
    grid_axes, rows_grid = [], []
    W_AX, H_AX, LAB, GAP = 28.6, 27.0, 13.0, 4.8
    ys_row = [131.0, 92.0]
    for i, t in enumerate(subs):
        r_, c_ = divmod(i, 5)
        x0 = 0 if c_ == 0 else LAB + c_ * (W_AX + GAP) - GAP + 0.0
        ax = ps.slot_mm(fig, x0, ys_row[r_], LAB if c_ == 0 else GAP, W_AX, H_AX)
        ps.zero_line(ax, "y", better_text=(i == 4))
        _symlog_y(ax, S4_YLIM)
        ps.log_n_axis(ax, ticks=(3, 10, 30, 100, 300), lim=(2.5, 380))
        n_last = 0
        for st, rule, lam, m, var, lbl in S4_SERIES:
            q = h25_series(C, t, st, rule, lam, 1.0)
            if not len(q):
                continue
            yv = q[ycol].to_numpy(float); xv = q.n.to_numpy(float)
            inside = (yv >= S4_YLIM[0]) & (yv <= S4_YLIM[1])
            yc = np.clip(yv, *S4_YLIM)
            ps.plot_curve(ax, xv, yc, method=m, variant=var, band=False, label=lbl, delta=False, ms=2.4)
            for xx, vv in zip(xv[~inside], yv[~inside]):          # 범위 밖 점: 축 끝 채운 삼각형
                ps.offscale_marker(ax, vv, xx, S4_YLIM, orient="v", method=m, ms=3.0)
            if (~inside).any():                                     # 수치는 계열당 1개(범위), 첫 범위 밖 점 왼쪽 아래
                vo = yv[~inside]; xlast = xv[~inside].max()
                s_ = ps.fmt_num(vo.min(), 1) if len(vo) == 1 else f"{ps.fmt_num(vo.min(), 1)}–{ps.fmt_num(vo.max(), 1)}"
                up = vo[0] > 0
                ax.annotate(s_, xy=(xlast, S4_YLIM[1] if up else S4_YLIM[0]), xytext=(2, -4.5 if up else 4.5), textcoords="offset points",
                            ha="right", va="top" if up else "bottom", fontsize=ps.FS["annot"], color=ps.COLOR[m])
            n_last = max(n_last, int(xv.max()))
            for _, rr in q.iterrows():
                rows_grid.append(dict(panel="abcdefghij"[i], target=t, stage=st, rule=rule, lam=lam, n=int(rr.n),
                                      d_phys=float(rr[ycol]), n_runs=int(rr.n_runs), n_splits=int(rr.n_splits),
                                      ci_lo=float(rr.blk_d_phys_lo) if blk else np.nan, ci_hi=float(rr.blk_d_phys_hi) if blk else np.nan))
        ref = h25_allA_ref(C, t)
        if np.isfinite(ref):
            ps.ref_line(ax, ref, "ref_allA")
            rows_grid.append(dict(panel="abcdefghij"[i], target=t, stage="S3", rule="all", lam=0.25, n=-1, d_phys=ref))
        npool = pool.get(t, np.inf)
        if n_last and n_last < 320:
            ax.axvspan(max(npool, n_last), 380, color=ps.GREY["band"], lw=0, zorder=0.2)
        ax.text(0.0, 1.0 + 1.3 / H_AX, ps.region_label(t, alr.get(t)), transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"])
        ax.text(1.0, 1.0 + 1.3 / H_AX, err_class(alr.get(t, np.nan)), transform=ax.transAxes, ha="right", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
        if c_ == 0:
            ax.set_ylabel("ΔRMSE vs physics (cm)")
        else:
            ps.hide_ylabels(ax)
        if r_ == 0:
            ax.tick_params(axis="x", labelbottom=False)
        grid_axes.append(ax)
    _fig_text(fig, LAB + (5 * W_AX + 4 * GAP) / 2, 81.5, "Target labels n")
    # 범례(그림당 1개): 격자와 포레스트 사이
    handles = [ps.method_handle(m, lbl, var) for _, _, _, m, var, lbl in S4_SERIES]
    handles += [Line2D([], [], color=ps.COLOR["ref_allA"], ls=ps.METHOD["ref_allA"]["ls"], lw=ps.METHOD["ref_allA"]["lw"],
                       label="All A-block labels (reference)"),
                Line2D([], [], color=ps.DELTA_ZERO["color"], lw=ps.DELTA_ZERO["lw"], label="Physics anchor (Δ = 0)")]
    if blk:
        handles.append(Line2D([], [], color="#000000", lw=ps.LW["ci_forest"], label="95 % block-bootstrap CI (k, l)"))
    ps.legend_below(fig, handles, (0, 69.5, 180, 10), ncol=4)

    # k, l 포레스트: 14 대상 × {E re-fit, residual ML}, n = 3 / 10
    targets = sorted(MAIN4 + SUB10, key=lambda t: alr.get(t, np.inf))
    labels = [ps.region_label(t, alr.get(t)) for t in targets]
    rows_f, f_axes = [], []
    for k, n in enumerate((3, 10)):
        ax = ps.slot_mm(fig, 0 if k == 0 else 101, 9, 33 if k == 0 else 3, 65, 55)
        for m, st, lam, off in (("refit", "S1", 0.0, 0.18), ("residual", "S3", 0.25, -0.18)):
            est, lo, hi = [], [], []
            for t in targets:
                q = h25_series(C, t, st, "kmedoid", lam, 1.0)
                q = q[q.n == n]
                if len(q):
                    rr = q.iloc[0]
                    est.append(float(rr[ycol])); lo.append(float(rr.blk_d_phys_lo) if blk else np.nan)
                    hi.append(float(rr.blk_d_phys_hi) if blk else np.nan)
                    rows_f.append(dict(panel="kl"[k], target=t, method=m, stage=st, lam=lam, n=n, d_phys=est[-1], ci_lo=lo[-1],
                                       ci_hi=hi[-1], split_win=float(rr.split_win) if blk else np.nan,
                                       rep_win=float(rr.rep_win) if blk else np.nan, ci_kind="block" if blk else "none"))
                else:
                    est.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            ys = ps.forest(ax, labels, est, lo if blk else None, hi if blk else None, method=m, offset=off, offscale=S4_FLIM,
                           sharey_with=(f_axes[0] if k == 1 else None), better_text=(m == "refit"), ms=2.8)
        lev = [y for y, t in zip(ys, targets) if err_class(alr.get(t, np.nan)) == "level"]
        mid = [y for y, t in zip(ys, targets) if err_class(alr.get(t, np.nan)) == "intermediate"]
        # 수준 대상은 연속 행(오름차순 정렬의 아래쪽)
        ps.group_bands(ax, lev, text="level" if k == 1 else None)
        if mid and k == 1:
            tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1.01, float(np.mean(mid)), "int.", transform=tr, fontsize=ps.FS["annot"], rotation=90, ha="left", va="center",
                    color=ps.GREY["text2"], clip_on=False)
        ax.set_xticks([-10, -5, 0, 5])
        ax.xaxis.set_major_formatter(_minus_fmt())
        ax.set_xlabel(f"ΔRMSE vs physics at n = {n} (cm)")
        f_axes.append(ax)
    ps.label_panels(grid_axes + f_axes, "abcdefghijkl", dy_mm=5.0)
    src = C.attrs.get("source", "data/processed/h3/h25_curve.csv")
    stat_ci = ("k, l show 95 % block-bootstrap CIs (1,000 resamples of scoring blocks within each split, indices shared across "
               "methods, repeats and seeds; splits averaged); panels a to j show no CI." if blk else
               "No block-level CI is available; row-resampling intervals are in Supplementary Table 3.")
    caption = dict(
        definition=("Label-budget staircase for subregions. ΔRMSE is method RMSE minus physics-anchor RMSE (E0 fitted on "
                    "source cells, n = 0) on held-out B blocks, in cm; negative is better. E re-fit shrinks the target E toward "
                    "E0 (κ = 10). All A-block labels is a reference, not an upper bound. Error type uses the oracle in-region E_own "
                    "(structure < 0.15, level ≥ 0.2, intermediate between)."),
        statistics=("Curves are means over 3 splits × 20 label draws (E re-fit) or 3 splits × 5 draws × 2 seeds (augmentation, "
                    f"residual ML); duplicate splits are counted once. {stat_ci}"),
        panels=("a to j, Ten subregions ordered by |log(E_own/E0)| (in parentheses); symmetric-log y axis, linear within ±2 cm; "
                "grey shading marks n beyond the A-block label pool; filled triangles at the axis edge mark off-scale means "
                "(range of values shown). k, l, All 14 analysed targets at n = 3 and n = 10 (k-medoid labels); grey band marks level "
                "targets, int. marks intermediate targets; CIs are clipped at the axis limits and off-scale means are drawn as "
                "filled triangles. Negative ΔRMSE (down in a to j, left in k and l) is better in every panel; the level band and "
                "int. mark label the same rows in k and l."),
        data=f"Source data: {src}. Supplementary Tables 1 and 3.",
    )
    spec = dict(intent="H25 하위 지역 10 대상 계단(Fig 3a–d 문법, 띠 없음) + 14 대상 n = 3·10 Δ 포레스트(블록 CI)", panels=12,
                assets=[src, "data/processed/h3/h25_targets.csv"], ci_kinds=dict(a_j="none(곡선)", k_l="block" if blk else "none"),
                script="scripts/4_visualization/paper_figs.py --only supp")
    return save_paper(fig, "FigS4_subregion_budget", caption=caption, spec=spec, draft=False,
                      sources={"a-j": pd.DataFrame(rows_grid), "k-l": pd.DataFrame(rows_f)})


# ================================================================ S5 H18–H24 관측 설계 곡선 + 표
S5_STRAT = [   # (전략, 범례, 색, 선종, 마커)  방법 색이 아니라 규칙 구분: k-중심만 행의 방법 색, 나머지는 회색 계열
    ("random", "Random labels", "#000000", (0, (3.5, 1.8)), ""),
    ("block_rr", "Block round-robin", "#808080", "-", "s"),
    ("farthest", "Farthest point (k-center)", "#808080", "-", "^"),
    ("stefan_strat", "Physics-quantile strata", "#808080", "-", "v"),
    ("geo_strat", "Coordinate k-means strata", "#808080", "-", "P"),
    ("calm_sites", "Existing CALM sites (Alaska)", "#b0b0b0", "-", "*"),
]
S5_ROWS = [("shrink_k10", 0.0, "refit", "E re-fit (κ = 10)"), ("resid", 0.25, "residual", "Residual ML (λ = 0.25)")]
S5_COLS = ["Lena", "Canada", "Russia_W", "Russia_E", "Alaska"]


def build_S5(draft: bool = False):
    Cc = rd(H2 / "h24_curve.csv")
    T = load_h25_targets()
    pool = dict(zip(T.target, T.n_A))
    ps.use_paper()
    fig = ps.paper_figure(180, 98)
    W_AX, H_AX, LAB, GAP = 27.4, 30.0, 16.0, 5.4
    ys_row = [60.0, 22.0]
    axes, rows = [], []
    for i, (meth, lam, mkey, mlab) in enumerate(S5_ROWS):
        for j, tgt in enumerate(S5_COLS):
            x0 = 0 if j == 0 else LAB + j * (W_AX + GAP) - GAP
            ax = ps.slot_mm(fig, x0, ys_row[i], LAB if j == 0 else GAP, W_AX, H_AX)
            sub = Cc[(Cc.target == tgt) & (Cc.method == meth) & np.isclose(Cc.lam, lam) & (Cc.n > 0)]
            ps.zero_line(ax, "y", better_text=(i == 0 and j == 4))
            if i == 1 and j == 4:                                   # j: 0선 아래는 곡선과 겹치므로 우상단 빈 곳에 방향 표시
                ax.text(0.98, 0.97, "better ↓", transform=ax.transAxes, ha="right", va="top", fontsize=ps.FS["annot"],
                        color=ps.GREY["text2"])
            for st, lbl, col, ls, mk in S5_STRAT:
                q = sub[sub.strategy == st].sort_values("n")
                if not len(q):
                    continue
                ax.plot(q.n, q.d_phys_mean, color=col, ls=ls, lw=0.7, marker=mk, ms=2.4, mfc=col, mec=col, mew=0.4, zorder=2.5)
                rows += [dict(panel="abcdefghij"[i * 5 + j], target=tgt, method=meth, lam=lam, strategy=st, n=int(r.n),
                              d_phys=float(r.d_phys_mean), n_runs=int(r.n_runs), ci_kind="none") for _, r in q.iterrows()]
            q = sub[sub.strategy == "kmedoid"].sort_values("n")
            ps.plot_curve(ax, q.n, q.d_phys_mean, method=mkey, delta=False, ms=2.8, zorder=3.5)
            rows += [dict(panel="abcdefghij"[i * 5 + j], target=tgt, method=meth, lam=lam, strategy="kmedoid", n=int(r.n),
                          d_phys=float(r.d_phys_mean), n_runs=int(r.n_runs), ci_kind="none") for _, r in q.iterrows()]
            ps.log_n_axis(ax, ticks=(3, 10, 30, 100), lim=(2.5, 260))
            nmax = int(sub.n.max()) if len(sub) else 0
            if tgt in pool and nmax < 200:
                ax.axvspan(max(nmax, 2.6), 260, color=ps.GREY["band"], lw=0, zorder=0.2)
            lo_, hi_ = ax.get_ylim()
            lo_, hi_ = min(lo_, -0.3), max(hi_, 0.3)
            ax.set_ylim(lo_, hi_)
            tv = matplotlib.ticker.MaxNLocator(4).tick_values(lo_, hi_)
            tv = [v for v in tv if lo_ <= v <= hi_]                 # 보이는 눈금만 고정(범위 밖 눈금 글자 잔존 방지)
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tv))
            nd = 0 if all(abs(v - round(v)) < 1e-9 for v in tv) else 1
            ax.yaxis.set_major_formatter(_minus_fmt(nd))
            if i == 0:
                nm = ps.region_name(tgt) + (" (sandbox)" if tgt == "Alaska" else "")
                ax.text(0.0, 1.0, nm, transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"])
                ax.tick_params(axis="x", labelbottom=False)
            if j == 0:
                ax.set_ylabel(f"{mlab}\nΔRMSE vs physics (cm)", linespacing=1.1)
            axes.append(ax)
    _fig_text(fig, LAB + (5 * W_AX + 4 * GAP) / 2, 13.3, "Target labels n")
    handles = [Line2D([], [], color=ps.COLOR["refit"], lw=1.0, marker="o", ms=2.8, label="k-medoid labels, E re-fit"),
               Line2D([], [], color=ps.COLOR["residual"], lw=1.0, marker="D", ms=2.8, label="k-medoid labels, residual ML")]
    handles += [Line2D([], [], color=col, ls=ls, lw=0.7, marker=mk, ms=2.4, mfc=col, mec=col, label=lbl) for _, lbl, col, ls, mk in S5_STRAT]
    ps.legend_below(fig, handles, (0, 0, 180, 9), ncol=4)
    ps.label_panels(axes, "abcdefghij", dy_mm=3.8)
    caption = dict(
        definition=("Observation design for sparse labels (H24). ΔRMSE is method RMSE minus physics-anchor RMSE (n = 0) on "
                    "held-out B blocks, in cm; negative is better. Rules choose which n A-block cells are labelled; the "
                    "estimator is fixed within a row. Alaska is a sandbox with a five-region equal-weight E0 anchor."),
        statistics=("Means over 3 splits × 20 draws (E re-fit) or 3 splits × 10 draws × 3 seeds (residual ML); no CI is "
                    "drawn. H24 was not in the Holm family and is reported as exploratory: the pre-registered gain over "
                    "random labels held only at n = 3. Hypothesis tests H18 to H24 are listed in Supplementary Table 8."),
        panels=("a to e, E re-fit with κ = 10. f to j, Physics anchor plus residual ML. Columns share the log n axis; y axes "
                "are independent. Grey shading marks n beyond the tested range; the grey horizontal line is the physics anchor "
                "(Δ = 0). Negative ΔRMSE (down) is better in every panel."),
        data="Source data: data/processed/h2/h24_curve.csv; tests in h2/h24_tests.csv and h2/h_tests_all.csv.",
    )
    spec = dict(intent="H24 관측 설계 곡선(h2_fig07 재작성): 2 추정량 × 5 지역, 규칙 7종(k-중심만 방법 색)", panels=10,
                assets=["data/processed/h2/h24_curve.csv"], ci_kinds=dict(all="none"),
                script="scripts/4_visualization/paper_figs.py --only supp")
    res = save_paper(fig, "FigS5_obs_design", caption=caption, spec=spec, draft=False, sources={"a-j": pd.DataFrame(rows)})
    return [res, table_S5()]


def table_S5():
    """ST(S5) H18–H24 검정표. 평균 행(AB4)만. H23·H24 는 CI 미산출·Holm 가족 밖."""
    A = rd(H2 / "h_tests_all.csv")
    A = A[A.is_mean.astype(str).str.lower().isin(["true", "1"])].copy()
    A["hyp"] = A.test.str.extract(r"^(H\d+)")[0]
    A = A.assign(comparison=A.test, delta_cm=A.delta, ci_lo_cm=A.ci_lo, ci_hi_cm=A.ci_hi, ci_kind="strat_AB4",
                 in_holm_family=A.p_holm.notna(), note="")
    A["confirmatory"] = A.confirmatory.astype(str).str.lower().isin(["true", "1"])
    B = rd(H2 / "h23_tests.csv"); B = B[B.target == "MEAN[AB4]"].copy()
    B = B.assign(hyp="H23", family="sparse", cond="labels", confirmatory=False, comparison=B.method + " − shrink κ10, λ " +
                 B.lam.map(lambda v: f"{v:g}") + ", n = " + B.n.astype(str), delta_cm=B.delta_vs_shrink, ci_lo_cm=np.nan,
                 ci_hi_cm=np.nan, p_boot=np.nan, p_holm=np.nan, ci_kind="none", in_holm_family=False,
                 note="win rate " + B.win_vs_shrink.map(lambda v: f"{v:.2f}") + "; not in Holm family")
    C = rd(H2 / "h24_tests.csv"); C = C[C.target == "MEAN[AB4]"].copy()
    C = C.assign(hyp="H24", family="design", cond="labels", confirmatory=False, comparison=C.strategy + " − random, " + C.method +
                 ", n = " + C.n.astype(str), delta_cm=C.delta_vs_random, ci_lo_cm=np.nan, ci_hi_cm=np.nan, p_boot=np.nan,
                 p_holm=np.nan, ci_kind="none", in_holm_family=False,
                 note="win rate " + C.win_vs_random.map(lambda v: f"{v:.2f}") + "; exploratory (gain at n = 3 only); not in Holm family")
    cols = ["hyp", "family", "cond", "confirmatory", "comparison", "delta_cm", "ci_lo_cm", "ci_hi_cm", "ci_kind", "p_boot", "p_holm",
            "in_holm_family", "note"]
    full = pd.concat([A[cols], B[cols], C[cols]], ignore_index=True)
    show = pd.concat([A[A.confirmatory][cols],
                      B[(B.method.isin(["finetune", "maml", "shrink+finetune", "shrink+maml"])) & np.isclose(B.lam, 0.25)][cols],
                      C[(C.strategy == "kmedoid")][cols]], ignore_index=True)
    show = show.assign(**{"Δ [95 % CI] (cm)": [
        (fmt_ci(d, lo, hi, 2) if np.isfinite(lo) else ps.fmt_num(d, 2)) for d, lo, hi in zip(show.delta_cm, show.ci_lo_cm, show.ci_hi_cm)]})
    show = show[["hyp", "cond", "comparison", "Δ [95 % CI] (cm)", "p_holm", "note"]].rename(
        columns={"hyp": "Hypothesis", "cond": "Condition", "comparison": "Comparison", "p_holm": "Holm p", "note": "Note"})
    caption = dict(
        definition=("Supplementary Table 8. Label-free and sparse-label hypothesis tests H18 to H24, four-region (AB4) means. "
                    "Δ is RMSE of the first method minus the comparator, in cm; negative favours the first method."),
        statistics=("H18 to H22 CIs are stratified block-bootstrap intervals over the AB4 regions with Holm adjustment within "
                    "each pre-registered family. H23 and H24 means have no CI and are outside the Holm family; H24 is "
                    "exploratory (the pre-registered gain over random labels held only at n = 3)."),
        panels="Printed rows: confirmatory H18 to H22 tests, H23 at λ = 0.25 and H24 k-medoid rows; all rows in the CSV file.",
        data="Source: data/processed/h2/h_tests_all.csv, h23_tests.csv, h24_tests.csv.",
    )
    return write_table("TableS8_h18_h24_tests", full, show, caption, col_fmt="lllrrl")


# ================================================================ S6 필요 라벨 수 보조(H27·A2·민감도·F6)
def _h27_primary():
    """H27b 주 행(k-medoid, S3 λ 0.25, 회복률 50 % 규칙, 오라클 E) 대상 표와 규칙 통계."""
    R, T, tag = load_h27("c")
    q = T[(T.rule == "kmedoid") & (T.stage == "S3") & np.isclose(T.lam, 0.25) & (T.definition == "rec50")].copy()
    sel = lambda d: R[(R.rule == "kmedoid") & (R.stage == "S3") & np.isclose(R.lam, 0.25) & (R.definition == d) & (R.E_kind == "oracle")]  # noqa: E731
    r, rm = sel("rec50"), sel("main")
    if not len(q) or not len(r):
        raise MissingData(f"h3/{tag}_targets.csv: kmedoid S3 λ 0.25 rec50 행 없음")
    q["undefined_all"] = q.n_rec_den_nonpos >= q.n_tested            # 모든 검사 n 에서 회복률 분모 ≤ 0
    q.attrs["tag"] = tag
    return q.sort_values("abs_logE_oracle"), r.iloc[0], (rm.iloc[0] if len(rm) else None), tag


def _a2_survival(draft: bool):
    """A2 생존형 계단 자료: 주 정의 n*(E0 고정, 전 블록, 잔차 λ 0.25). Alaska_f* 샌드박스 제외."""
    M = load_a2_minn(draft)
    q = M[(M.e_treat == "E0_fixed") & (M.spread.isin(["all_blocks", "all"])) & (M.stage == "resid") & np.isclose(M.lam, 0.25)]
    if "analysis" in q:
        q = q[q.analysis.astype(str).str.lower().isin(["true", "1"])]
    q = q[~q.target.astype(str).str.startswith("Alaska_f")].drop_duplicates("target")
    if not len(q):
        raise MissingData("a2_minn: E0_fixed·all_blocks·resid λ 0.25 행 없음")
    return q, M.attrs


def _sensitivity_rows():
    """대상 14개(CA-1 제외)의 설정 차이 Δ(변형) − Δ(기본) 중앙값·사분위(k-medoid 라벨).
    α·λ: S3 잔차(기본 λ 0.25, α 1), κ: S1refit(수축 없음) − S1(κ 10). α 는 h25(재실행 h25b 는 α 1 만 검사)."""
    Cb = load_h25_curve()
    C0 = rd(H3 / "h25_curve.csv")
    tg = MAIN4 + SUB10
    out = []

    def one(C, t, st, lam, alpha):
        q = C[(C.target == t) & (C.stage == st) & (C.rule == "kmedoid") & (C.scope == "n") & np.isclose(C.lam, lam) & np.isclose(C.alpha, alpha)]
        if "n_splits" in q and len(q):
            q = q[q.n_splits == q.n_splits.max()]
        return q.set_index("n").d_phys_mean
    specs = [("α = 10 vs 1", C0, ("S3", 0.25, 10.0), ("S3", 0.25, 1.0), "residual", "dashed", "h3/h25_curve.csv"),
             ("α = 100 vs 1", C0, ("S3", 0.25, 100.0), ("S3", 0.25, 1.0), "residual", "dotted", "h3/h25_curve.csv"),
             ("λ = 0.5 vs 0.25", Cb, ("S3", 0.5, 1.0), ("S3", 0.25, 1.0), "residual", None, Cb.attrs.get("source")),
             ("no shrinkage vs κ = 10", Cb, ("S1refit", 0.0, 1.0), ("S1", 0.0, 1.0), "refit", None, Cb.attrs.get("source"))]
    for lbl, C, a, b, m, var, src in specs:
        for t in tg:
            d = (one(C, t, *a) - one(C, t, *b)).dropna()
            for n, v in d.items():
                out.append(dict(setting=lbl, method=m, variant=var, target=t, n=int(n), d_setting=float(v), source=src))
    D = pd.DataFrame(out)
    S = D.groupby(["setting", "method", "n"], dropna=False).d_setting.agg(
        median="median", q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75), n_targets="size").reset_index()
    return D, S, specs


def _theory_rows(draft: bool):
    """F6: 이론 n_theory(ε 0.05, 사전 분산) 대 경험 손익분기 n(CI 규칙, S1 k-medoid). B2 산출이 없으면 MissingData."""
    Th = load_h4("b2", "theory", draft)
    col = "n_theory_prior_eps0.05"
    if col not in Th:
        raise MissingData(f"b2_theory: {col} 열 없음")
    from _common import load_h25_breakeven
    Be = load_h25_breakeven()
    be = Be[(Be.rule == "kmedoid") & (Be.stage == "S1") & np.isclose(Be.lam, 0.0) & np.isclose(Be.alpha, 1.0)]
    cens = be.censored.astype(str).str.lower().isin(["true", "1"])
    be = be.assign(censored=cens, n_emp=np.where(cens, np.nan, be.breakeven_n_ci))
    J = Th[["target", col, "abs_logE_ratio"]].merge(be[["target", "n_emp", "censored", "n_max"]], on="target", how="inner")
    J = J.rename(columns={col: "n_theory"})
    try:
        Cr = load_h4("b2", "theory_corr", draft)
        cr = Cr[Cr.primary.astype(str).str.lower().isin(["true", "1"])]
        cr = cr.iloc[0] if len(cr) else None
    except MissingData:
        cr = None
    return J, cr, Th.attrs, Be.attrs


S6_BE_LIM = (2.2, 450)


def build_S6(draft: bool = False):
    q, rstat, rmain, h27tag = _h27_primary()
    A2, a2_attrs = _a2_survival(draft)
    D, S, specs = _sensitivity_rows()
    J, cr, th_attrs, be_attrs = _theory_rows(draft)
    smoke = bool(a2_attrs.get("is_smoke")) or bool(th_attrs.get("is_smoke"))
    if smoke and not draft:
        raise MissingData("A2·B2 본 실행 산출 없음")
    alr = abslogE_map()
    ps.use_paper()
    fig = ps.paper_figure(180, 118)
    X_LIM = (-0.02, 0.56)
    # ---------------------------------------------------------------- a H27 손익분기(달성) + 미달성 띠
    ax_a = ps.slot_mm(fig, 0, 81, 15, 70, 31)
    ax_ab = ps.slot_mm(fig, 0, 70, 15, 70, 8.5)
    ach = q[~q.censored]
    for _, r in ach.iterrows():
        ax_a.plot([r.abs_logE_oracle], [r.be_n], ls="none", **{k: v for k, v in ps.style_of("residual").items() if k not in ("ls", "lw")})
        ax_a.annotate(ps.region_name(r.target), xy=(r.abs_logE_oracle, r.be_n), xytext=(3.5, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=ps.FS["annot"])
    ax_a.set_yscale("log"); ax_a.set_ylim(*S6_BE_LIM)
    ax_a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([3, 10, 30, 100, 300]))
    ax_a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax_a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_a.set_xlim(*X_LIM); ax_a.tick_params(axis="x", labelbottom=False)
    ax_a.set_ylabel("Break-even n\n(recovery rule, H27)", linespacing=1.1)
    cz = q[q.censored]
    ax_ab.set_xlim(*X_LIM)
    # 미달성 띠: ▲ 만 찍고 n_max 는 캡션에 적는다(대상별 라벨은 0–0.13 구간에서 겹친다)
    ax_ab.set_facecolor(ps.GREY["band"]); ax_ab.set_ylim(0, 1); ax_ab.set_yticks([])
    for sp in ("left", "right", "top"):
        ax_ab.spines[sp].set_visible(False)
    xs_c = cz.abs_logE_oracle.values
    rw_c = _dodge_rows((xs_c - X_LIM[0]) * 70 / (X_LIM[1] - X_LIM[0]), 1.5)   # 1.5 mm 안쪽으로 붙은 ▲ 는 다른 줄로(최대 3줄)
    ys_c = np.array([0.2, 0.5, 0.8])[np.minimum(rw_c, 2)]
    for xi, yi, nm, und in zip(xs_c, ys_c, cz.n_max.values, cz.undefined_all.values):
        c_ = ps.CENSOR["range_limited"] if nm < 40 else ps.CENSOR["censored"]
        ax_ab.plot([xi], [yi], "^", color=c_, mec=c_, mfc="white" if und else c_, mew=0.7, ms=ps.MS["main"] + 0.4, zorder=3, clip_on=False)
    ax_ab.text(-0.012, 0.5, "not\nreached", transform=ax_ab.get_yaxis_transform(), ha="right", va="center",
               fontsize=ps.FS["annot"], color=ps.GREY["text2"], linespacing=1.0)
    ax_ab.set_xlabel("|log(E_own/E0)| (oracle)")
    ax_ab.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.1f}"))
    # ---------------------------------------------------------------- b A2 생존형 계단
    ax_b = ps.slot_mm(fig, 94, 70, 15, 54, 42)
    grid = np.array([3, 5, 10, 20, 40, 80, 160, 320], float)
    rows_b = []
    A2 = A2.assign(cls=A2.target.map(lambda t: err_class(alr.get(t, np.nan))))
    for cls, var in (("level", None), ("structure", "dashed"), ("intermediate", "dotted")):
        g = A2[A2.cls == cls]
        if not len(g):
            continue
        frac = [float(np.mean(~(g.n_star <= n) )) for n in grid]           # 아직 n* 에 이르지 못한 비율(절단 = 끝까지 남음)
        nmax = float(g.n_max.max())
        xs = grid[grid <= nmax]
        ys = np.array(frac)[:len(xs)]
        ax_b.step(np.r_[2.5, xs], np.r_[1.0, ys], where="post", color=ps.COLOR["residual"], ls=ps.VARIANT_LS.get(var, "-") if var else "-",
                  lw=1.0)
        ax_b.text(xs[-1] * 1.08, ys[-1], f"{cls} ({len(g)})", ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
        rows_b += [dict(panel="b", error_type=cls, n=int(x), frac_not_reached=float(y), n_targets=len(g)) for x, y in zip(xs, ys)]
    ps.log_n_axis(ax_b, ticks=(3, 10, 30, 100, 300), lim=(2.5, 380))
    ax_b.set_ylim(-0.03, 1.05)
    ax_b.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax_b.set_ylabel("Targets without n* (fraction)")
    ax_b.set_xlabel("Target labels n")
    ax_b.spines["bottom"].set_bounds(2.5, 380)
    # ---------------------------------------------------------------- c α·λ·κ 민감도
    ax_c = ps.slot_mm(fig, 0, 17, 15, 70, 33)
    ps.zero_line(ax_c, "y", better_text=False)
    ax_c.text(0.02, 0.03, "better ↓", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    rows_c = []
    ends = []
    for lbl, _, _, _, m, var, _ in specs:
        g = S[S.setting == lbl].sort_values("n")
        if not len(g):
            continue
        ps.plot_curve(ax_c, g.n, g["median"], method=m, variant=var, delta=False, ms=2.4)
        ends.append([float(g["median"].iloc[-1]), lbl, m, float(g.n.iloc[-1])])
        rows_c += [dict(panel="c", setting=lbl, n=int(r.n), median=float(r["median"]), q25=float(r.q25), q75=float(r.q75),
                        n_targets=int(r.n_targets)) for _, r in g.iterrows()]
    lo_c, hi_c = ax_c.get_ylim()
    tv = [v for v in matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(lo_c, hi_c) if lo_c <= v <= hi_c]
    ax_c.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tv)); ax_c.set_ylim(lo_c, hi_c)
    step = 3.9 * (hi_c - lo_c) / 33.0                              # 끝 라벨 최소 간격 3.9 mm
    ends.sort(key=lambda e: e[0])
    ypos = [e[0] for e in ends]
    for _ in range(50):                                            # 아래에서 위로 밀어 간격 확보
        for k_ in range(1, len(ypos)):
            if ypos[k_] - ypos[k_ - 1] < step:
                ypos[k_] = ypos[k_ - 1] + step
    shift = min(0.0, hi_c - step * 0.4 - ypos[-1])
    ypos = [v + shift for v in ypos]
    for (v, lbl, m, nx), yl in zip(ends, ypos):
        ax_c.text(470, yl, lbl, ha="left", va="center", fontsize=ps.FS["annot"], color=ps.COLOR[m])
        ax_c.plot([nx * 1.08, 450], [v, yl], color=ps.GREY["light"], lw=0.4, zorder=1, clip_on=False)   # 끝점 → 라벨 연결선
    ps.log_n_axis(ax_c, ticks=(3, 10, 30, 100, 300), lim=(2.5, 380))
    ax_c.set_xlim(2.5, 2600)
    ax_c.spines["bottom"].set_bounds(2.5, 380)
    fr = (np.log10(380) - np.log10(2.5)) / (np.log10(2600) - np.log10(2.5))
    for ln in ax_c.get_lines():                                    # 0선은 자료 구간(n ≤ 380)까지만
        if ln.get_gid() == "delta_zero":
            ln.set_xdata([0, fr])
    ax_c.set_ylabel("ΔRMSE vs default\nsetting (cm), median", linespacing=1.1)
    ax_c.set_xlabel("Target labels n")
    ax_c.xaxis.set_label_coords((np.log10(380) - np.log10(2.5)) / 2 / (np.log10(2600) - np.log10(2.5)), -0.17)
    ax_c.yaxis.set_major_formatter(_minus_fmt(1))
    for a_ in (ax_b, ax_c):                                        # 라벨 여백 구간(n > 380)의 보조 눈금 제거
        a_.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([v for d_ in (1, 10, 100) for v in np.arange(2, 10) * d_ if 2.5 <= v <= 380]))
    # ---------------------------------------------------------------- d F6 이론 대 경험
    ax_d = ps.slot_mm(fig, 94, 17, 15, 71, 25)
    ax_db = ps.slot_mm(fig, 94, 43, 15, 71, 7)                   # 미달성 띠(패널 a 와 같은 규칙, 줄 배정으로 겹침 방지)
    lim = (1.5, 400)
    ax_d.plot(lim, lim, color=ps.GREY["mid"], ls=":", lw=0.7, zorder=1)
    rows_d = []
    for _, r in J[~J.censored.astype(bool)].iterrows():
        ax_d.plot([float(r.n_theory)], [r.n_emp], ls="none", **{k: v for k, v in ps.style_of("refit").items() if k not in ("ls", "lw")})
    Jc = J[J.censored.astype(bool)]
    xmm = (np.log10(Jc.n_theory.clip(*lim).values) - np.log10(lim[0])) * 71 / (np.log10(lim[1]) - np.log10(lim[0]))
    rw_d = _dodge_rows(xmm, 1.3)
    nrow = max(1, int(rw_d.max()) + 1) if len(Jc) else 1
    for (_, r), rr in zip(Jc.iterrows(), rw_d):
        c_ = ps.CENSOR["range_limited"] if r.n_max < 40 else ps.CENSOR["censored"]
        ax_db.plot([float(np.clip(r.n_theory, *lim))], [(rr + 0.5) / nrow], "^", color=c_, mec=c_, mfc=c_, ms=ps.MS["main"],
                   clip_on=False, zorder=3)
    for _, r in J.iterrows():
        rows_d.append(dict(panel="d", target=r.target, n_theory=float(r.n_theory), n_empirical=r.n_emp, censored=bool(r.censored),
                           n_max=r.n_max))
    ax_db.set_xscale("log"); ax_db.set_xlim(*lim); ax_db.set_ylim(0, 1); ax_db.set_yticks([])
    ax_db.set_facecolor(ps.GREY["band"]); ax_db.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for sp in ("left", "right", "top", "bottom"):
        ax_db.spines[sp].set_visible(False)
    ax_db.text(0.985, 0.5, f"not reached ({len(Jc)})", transform=ax_db.transAxes, ha="right", va="center",
               fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    for f_ in (ax_d.set_xscale, ax_d.set_yscale):
        f_("log")
    ax_d.set_xlim(*lim); ax_d.set_ylim(*lim)
    for a_ in (ax_d.xaxis, ax_d.yaxis):
        a_.set_major_locator(matplotlib.ticker.FixedLocator([3, 10, 30, 100, 300]))
        a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
        a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_d.set_xlabel("n_theory (ε = 0.05)")
    ax_d.set_ylabel("Empirical n\n(block-CI rule, H25)", linespacing=1.1)
    # ---------------------------------------------------------------- 범례·패널 문자
    tri = lambda mfc, c, lab: Line2D([], [], ls="none", marker="^", ms=ps.MS["main"] + 0.4, color=c, mec=c, mfc=mfc, label=lab)
    handles = [ps.method_handle("residual", "Break-even reached (a)", line=False),
               tri(ps.CENSOR["censored"], ps.CENSOR["censored"], "Not reached by n_max"),
               tri("white", ps.CENSOR["censored"], "Not reached, recovery undefined (a)"),
               tri(ps.CENSOR["range_limited"], ps.CENSOR["range_limited"], "Not reached, n_max < 40"),
               ps.method_handle("refit", "Empirical n reached (d)", line=False),
               Line2D([], [], color=ps.GREY["mid"], ls=":", lw=0.7, label="1:1 line (d)")]
    ps.legend_below(fig, handles, (0, 0, 180, 8), ncol=3)
    ps.label_panels([ax_a, ax_b, ax_c, ax_db], "abcd", dy_mm=1.5)
    n_ach, n_tot = int((~q.censored).sum()), len(q)
    n_und = int(q.undefined_all[q.censored].sum())
    nm_mode = int(cz.n_max.mode().iloc[0])
    exc = cz[cz.n_max != nm_mode]
    exc_txt = "; ".join(f"{', '.join(ps.region_name(t) for t in g_.target)} {int(v)}" for v, g_ in exc.groupby("n_max"))
    nmax_txt = f"n_max = {nm_mode}" + (f" except {exc_txt}; grey, n_max < 40" if len(exc) else "")
    sep = "complete separation" if bool(rstat.separation) else "no complete separation"
    caption = dict(
        definition=("Auxiliary label-number analyses. Break-even n (H27, recovery rule) is the smallest n at which residual ML "
                    "(k-medoid labels, λ = 0.25) recovers at least 50 % of the gap between physics and the all-A-block-label "
                    "reference with win rate ≥ 0.75; n* (A2) is the pre-specified CI rule of Fig. 4. Not-reached targets are "
                    "shown as > n_max only."),
        statistics=(f"a: {n_ach} of {n_tot} targets reach break-even; reaching is associated with |log(E_own/E0)| "
                    f"(Mann-Whitney U = {rstat.mw_U:.0f}, exact p = {rstat.p_mw:.3f}, {sep}; Kendall τ_b = {ps.fmt_num(rstat.kendall_tau, 2)}, "
                    f"p = {rstat.kendall_p:.3f}, censored targets tied at the maximum rank)"
                    + (f"; under the block-CI rule {int(rmain.n_uncensored)} of {n_tot} reach it (τ_b = {ps.fmt_num(rmain.kendall_tau, 2)}, "
                       f"p = {rmain.kendall_p:.3f})" if rmain is not None else "")
                    + ". |log E| uses the oracle in-region E_own."
                    + (f" d: censored-tied Spearman ρ = {ps.fmt_num(cr.spearman, 2)} (p = {cr.p:.2f}; {int(cr.n_censored)} of "
                       f"{int(cr.n_targets)} not reached); pre-registered F6 threshold ρ ≥ 0.6." if cr is not None and not smoke
                       else "")),
        panels=(f"a, Upper axis, reached targets; band, not-reached targets ({nmax_txt}); open triangles ({n_und} targets) mark "
                "targets whose reference was worse than physics at every tested n; triangles closer than 1.5 mm are stacked in rows "
                "(x unchanged). b, Fraction of targets without n* by n "
                "(E0 fixed, all blocks), by error type (count in parentheses; level ≥ 0.2, structure < 0.15). c, Median over 14 targets of the ΔRMSE "
                "change from the default setting (α = 1, λ = 0.25, κ = 10); α from the original H25 run. d, Theory n versus "
                "empirical n*; the band above holds targets not reached by n_max (grey, n_max < 40), stacked in rows where they "
                "would overlap (x unchanged). Lena and Russia W coincide at n* = 3."),
        data=(f"Source data: data/processed/h3/{h27tag}_targets.csv, {h27tag}_rule.csv (a); {a2_attrs.get('source')} (b); h3/h25_curve.csv, "
              f"{load_h25_curve().attrs.get('source')} (c); {th_attrs.get('source')}, {be_attrs.get('source')} (d). Supplementary Table 6."),
    )
    spec = dict(intent="S6 필요 라벨 수 보조: H27 이진 달성 + A2 생존형 계단 + α·λ·κ 민감도 + F6 이론 대 경험", panels=4,
                assets=[f"data/processed/h3/{h27tag}_targets.csv", f"data/processed/h3/{h27tag}_rule.csv", a2_attrs.get("source"),
                        th_attrs.get("source")], script="scripts/4_visualization/paper_figs.py --only supp --draft",
                smoke_panels=["b", "d"] if smoke else [])
    src_a = q[["target", "abs_logE_oracle", "be_n", "censored", "n_max", "n_tested", "undefined_all"]].assign(panel="a")
    return save_paper(fig, "FigS6_label_number_aux", caption=caption, spec=spec, draft=draft or smoke,
                      sources={"a": src_a, "b": pd.DataFrame(rows_b), "c": pd.DataFrame(rows_c), "d": pd.DataFrame(rows_d)})


# ================================================================ ST3 행 재표집 CI 대 블록 CI
def _ci_pairs(draft: bool) -> pd.DataFrame:
    """출처별 (점 추정, 블록 CI, 행 재표집 CI) 행. 두 CI 가 모두 있는 행만."""
    parts = []

    def add(df, src, keys, est, blo, bhi, rlo, rhi, label):
        d = df[keys].copy()
        d["key"] = d.astype(str).agg(" | ".join, axis=1)
        d = d[["key"]].assign(source=label, file=src, est=df[est].values, blk_lo=df[blo].values, blk_hi=df[bhi].values,
                              rep_lo=df[rlo].values, rep_hi=df[rhi].values)
        parts.append(d)
    Cb = load_h25_curve()
    if Cb.attrs.get("ci_kind") == "block":
        q = Cb[Cb.scope == "n"]
        add(q, Cb.attrs["source"], ["target", "stage", "rule", "lam", "alpha", "n"], "blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi",
            "ci_rep_lo", "ci_rep_hi", "H25 (re-run)")
    for tag, label, keys in (("b1", "B1 protocol", ["target", "method", "analysis", "n"]),
                             ("b3", "B3 label design", ["target", "layer", "rule", "stage", "n"]),
                             ("d1", "D1 external hold-out", ["target", "method", "n", "tau"])):
        try:
            B = rd(h4_path(tag, "blk", draft))
        except MissingData:
            continue
        add(B, B.attrs["source"], keys, "blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi", "ci_rep_lo", "ci_rep_hi", label)
    for tag, kind, label, keys, cols in (
            ("b2", "pooling", "B2 partial pooling", ["target", "est", "variant", "rule", "lam", "n"],
             ("d_phys_blk", "d_phys_lo", "d_phys_hi", "ci_rep_phys_lo", "ci_rep_phys_hi")),
            ("b4", "summary", "B4 tabular foundation model", ["target", "method", "lam", "n"],
             ("d_phys_blk", "d_phys_lo", "d_phys_hi", "ci_rep_lo", "ci_rep_hi")),
            ("a2", "curve", "A2 confounder-free minimum n", ["target", "e_treat", "spread", "scope", "n", "lam", "stage"],
             ("d_phys_mean", "d_phys_lo", "d_phys_hi", "d_phys_lo_rep", "d_phys_hi_rep"))):
        try:
            B = load_h4(tag, kind, draft)
        except MissingData:
            continue
        if B.attrs.get("is_smoke") and not draft:
            continue
        add(B, B.attrs["source"], keys, *cols, label + (" (smoke)" if B.attrs.get("is_smoke") else ""))
    P = pd.concat(parts, ignore_index=True)
    P = P[np.isfinite(P.blk_lo) & np.isfinite(P.blk_hi) & np.isfinite(P.rep_lo) & np.isfinite(P.rep_hi)].copy()
    P["blk_width"] = P.blk_hi - P.blk_lo
    P["rep_width"] = P.rep_hi - P.rep_lo
    P["width_ratio"] = P.blk_width / P.rep_width.where(P.rep_width > 0)
    P["blk_excl0"] = (P.blk_hi < 0) | (P.blk_lo > 0)
    P["rep_excl0"] = (P.rep_hi < 0) | (P.rep_lo > 0)
    P["rep_only_excl0"] = P.rep_excl0 & ~P.blk_excl0
    return P


def table_ST3(draft: bool = False):
    P = _ci_pairs(draft)
    g = P.groupby(["source", "file"], sort=False)
    Sm = g.agg(rows=("key", "size"), median_width_ratio=("width_ratio", "median"),
               q25=("width_ratio", lambda v: v.quantile(0.25)), q75=("width_ratio", lambda v: v.quantile(0.75)),
               pct_block_excl0=("blk_excl0", "mean"), pct_rep_excl0=("rep_excl0", "mean"),
               pct_rep_only=("rep_only_excl0", "mean")).reset_index()
    for c in ("pct_block_excl0", "pct_rep_excl0", "pct_rep_only"):
        Sm[c] = 100 * Sm[c]
    show = pd.DataFrame({
        "Experiment": Sm.source, "Rows": Sm.rows.astype(int).astype(str),
        "Width ratio, block / row (median [IQR])": [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(Sm.median_width_ratio, Sm.q25, Sm.q75)],
        "Block CI excludes 0 (%)": Sm.pct_block_excl0.round(0).astype(int).astype(str),
        "Row CI excludes 0 (%)": Sm.pct_rep_excl0.round(0).astype(int).astype(str),
        "Row CI only (%)": Sm.pct_rep_only.round(0).astype(int).astype(str)})
    pending = [] if any(P.source.str.startswith("A2")) else ["A2 (full run pending)"]
    caption = dict(
        definition=("Supplementary Table 3. Row-resampling versus block-bootstrap 95 % CIs for ΔRMSE vs physics. The block CI "
                    "resamples scoring blocks within each split (1,000 draws, indices shared across methods, repeats and seeds) "
                    "and is the CI drawn in figures; the row CI resamples (split, repeat) rows and ignores scoring-block variation."),
        statistics=("Width ratio is block CI width divided by row CI width per row. Row CI only is the share of rows whose row "
                    "CI excludes 0 while the block CI does not."
                    + (f" Pending: {', '.join(pending)}." if pending else "")),
        panels="Summary by experiment; all paired rows with both intervals are in the CSV file.",
        data=("Source: data/processed/h3/h25b_curve.csv; data/processed/h4/b1_blk.csv, b2_pooling.csv, b3_blk.csv, "
              "b4_summary.csv, d1_blk.csv, a2_curve.csv."),
    )
    return write_table("TableS3_ci_row_vs_block", P, show, caption, col_fmt="lrrrrr")


# ================================================================ ST6 H27 손익분기(절단 표기) + C2 블록 등가중 커버리지
def table_ST6(draft: bool = False):
    R, T, tag = load_h27("c")
    out = []
    dfns = ["rec50", "main", "ci"] if (T.definition == "main").any() else ["rec50", "ci"]
    combos = [(r_, d_) for r_ in ("kmedoid", "random") for d_ in dfns]
    q = T[(T.stage == "S3") & np.isclose(T.lam, 0.25)]
    for t, g in q.groupby("target"):
        r0 = g.iloc[0]
        row = dict(target=t, parent=r0.parent, n_cells=int(r0.n_cells), n_blocks=int(r0.n_blocks),
                   abs_logE_oracle=float(r0.abs_logE_oracle), abs_logE_est3_pooled=float(r0.abs_logE_est3_pooled),
                   error_type=err_class(float(r0.abs_logE_oracle)))
        for rule, dfn in combos:
            h = g[(g.rule == rule) & (g.definition == dfn)]
            if len(h):
                h = h.iloc[0]
                row[f"be_{rule}_{dfn}"] = fmt_censored(h.be_n, h.censored, h.n_max)
                row[f"be_{rule}_{dfn}_value"] = h.be_n if not h.censored else np.nan
                row[f"censored_{rule}_{dfn}"] = bool(h.censored)
                row[f"n_max_{rule}_{dfn}"] = int(h.n_max)
                row[f"recovery_undefined_all_n_{rule}_{dfn}"] = bool(h.n_rec_den_nonpos >= h.n_tested) if dfn == "rec50" else False
        out.append(row)
    A = pd.DataFrame(out).sort_values("abs_logE_oracle")
    show = pd.DataFrame({"Target": [ps.region_name(t) for t in A.target], "|log E| (oracle)": A.abs_logE_oracle.map(lambda v: f"{v:.3f}"),
                         "|log E| (3 labels)": A.abs_logE_est3_pooled.map(lambda v: f"{v:.3f}"), "Type": A.error_type,
                         "k-medoid, recovery": [s + (" †" if u else "") for s, u in zip(A.be_kmedoid_rec50, A.recovery_undefined_all_n_kmedoid_rec50)],
                         "k-medoid, block CI": A[f"be_kmedoid_{dfns[1]}"],
                         "random, recovery": [s + (" †" if u else "") for s, u in zip(A.be_random_rec50, A.recovery_undefined_all_n_random_rec50)],
                         "random, block CI": A[f"be_random_{dfns[1]}"]})
    show = show.replace(regex={r" \(max tested\)": "", r"\(max tested, range-limited\)": "(range-limited)"})           # 표 본문은 '> n_max' 만(캡션에 정의)
    cap_a = dict(
        definition=("Supplementary Table 6a. H27 break-even n for residual ML (λ = 0.25) by label rule and definition. Recovery "
                    "rule: at least 50 % recovery of the physics to all-A-block-label gap with win rate ≥ 0.75. Block CI rule: "
                    "block-bootstrap 95 % CI upper bound below 0, split win rate ≥ 2/3 and repeat win rate ≥ 0.75."),
        statistics=("Not-reached targets are reported as > n_max (the largest n tested), never as imputed values; Russia E "
                    "(n_max 10) is range-limited. The row-resampling CI rule is in the CSV file only. † marks targets whose all-label reference was worse than physics at every tested n, so recovery "
                    "is undefined. |log E| (oracle) uses all target cells; 3 labels uses pooled three-label estimates."),
        panels="Rows ordered by oracle |log(E_own/E0)|; error type: structure < 0.15, level ≥ 0.2, intermediate between.",
        data=f"Source: data/processed/h3/{tag}_targets.csv (derived from the H25 re-run, h3/h25b_curve.csv).",
    )
    res_a = write_table("TableS6a_h27_breakeven", A, show, cap_a, col_fmt="lrrlllll")
    # 6b 규칙 통계
    Rb = R[R.stage.isin(["S1", "S3"])].copy().assign(target_set="all 14")
    try:
        Rx, _, tagx = load_h27("x")
        Rb = pd.concat([Rb, Rx.assign(target_set="10 subregions, same-parent source excluded")], ignore_index=True)
    except MissingData:
        tagx = None
    Rb["reached"] = Rb.n_uncensored.astype(int).astype(str) + " / " + Rb.n_targets.astype(int).astype(str)
    showb = Rb[(Rb.E_kind.isin(["oracle", "est3_pooled"]) & (Rb.target_set == "all 14") & Rb.definition.isin(["rec50", "main"]))
               | ((Rb.E_kind == "oracle") & (Rb.target_set != "all 14") & Rb.definition.isin(["rec50", "main"]))]
    showb = pd.DataFrame({"Targets": showb.target_set.map({"all 14": "14"}).fillna("10 (excl.)"), "Rule": showb.rule, "Stage": showb.stage, "Definition": showb.definition.replace({"rec50": "recovery", "main": "block CI", "ci": "row CI"}), "|log E|": showb.E_kind.replace(
        {"oracle": "oracle", "est3_pooled": "3 labels"}), "Reached": showb.reached,
        "Kendall τ_b": showb.kendall_tau.map(lambda v: ps.fmt_num(v, 2)), "p (Kendall)": showb.kendall_p.map(lambda v: f"{v:.3f}"),
        "Mann-Whitney U": showb.mw_U.map(lambda v: f"{v:.0f}"), "exact p": showb.p_mw.map(lambda v: f"{v:.3f}"),
        "Complete separation": showb.separation.map(lambda v: "yes" if str(v).lower() in ("true", "1", "1.0") else "no")})
    cap_b = dict(
        definition=("Supplementary Table 6b. Association between reaching break-even and |log(E_own/E0)| for each label rule, "
                    "stage and definition (S1, E re-fit; S3, residual ML λ = 0.25)."),
        statistics=("Kendall τ_b treats not-reached targets as tied at the maximum rank; Mann-Whitney U compares |log E| of reached "
                    "and not-reached targets (exact p). Logistic coefficients are not reported under complete separation. "
                    "Spearman correlations are in the CSV file and are not used as the primary statistic."),
        panels=("Printed rows: recovery and block CI rules; 14 targets with oracle and three-label pooled |log E|, and the "
                "10 subregions refitted with same-parent source cells excluded (excl.). Other variants are in the CSV file."),
        data=f"Source: data/processed/h3/{tag}_rule.csv" + (f", {tagx}_rule.csv." if tagx else "."),
    )
    res_b = write_table("TableS6b_h27_rule_stats", Rb, showb, cap_b, col_fmt="lllllrrrrrl")
    # 6c C2 커버리지
    Cv = load_c2_coverage(draft)
    c = Cv[Cv.test == "label0"].copy()
    c["width"] = np.where(c.is_inf, np.inf, c.width_cm)
    order = ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland", "Alaska"]
    c["o"] = c.target.map(lambda t: order.index(t) if t in order else 99)
    c = c.sort_values(["o", "method"])
    showc = c[c.method.isin(["pooled", "hier2_cdf", "hier2_blk"]) & (c.o < 99)]
    showc = pd.DataFrame({"Target": showc.target.map(ps.region_name), "Method": showc.method,
                          "Coverage, cell-weighted": showc.coverage.map(lambda v: f"{v:.2f}"),
                          "Coverage, block-equal": showc.coverage_beq.map(lambda v: f"{v:.2f}"),
                          "Width (cm)": showc.width_cm.map(lambda v: "inf" if not np.isfinite(v) else f"{v:.0f}"),
                          "Cells / blocks": showc.n_eval.astype(int).astype(str) + " / " + showc.n_blocks.astype(int).astype(str)})
    cap_c = dict(
        definition=("Supplementary Table 6c. Label-free 90 % conformal intervals (C2): leave-one-region-out coverage with "
                    "cell-weighted and block-equal scoring. pooled, single pooled quantile; hier2_cdf, hierarchical two-level "
                    "(primary); hier2_blk, block-level calibration."),
        statistics=("Block-equal coverage weights each scoring block equally. Russia C and Greenland have fewer than 8 blocks. "
                    "hier2_exact is infinite when the number of calibration regions is below 1/α and is listed only in the CSV file."),
        panels="Rows by target region and method.",
        data=f"Source: {Cv.attrs.get('source')}.",
    )
    res_c = write_table("TableS6c_c2_coverage_blockeq", c.drop(columns=["o"]), showc, cap_c, col_fmt="llrrrr")
    return [res_a, res_b, res_c]


# ================================================================ ST1 지역·대상 셀·블록 수와 E
def table_ST1(draft: bool = False):
    """(a) 주 지역: 셀·블록, E = exp(mean z) 와 95 % 블록 CI(Fig 1b 산출값), (b) 전이 대상 15: 셀·블록·A 라벨 풀·E_own·E0·원천 구성."""
    rows = []
    fb = OUT / "source_data" / "Fig1_b.csv"
    Fb = pd.read_csv(fb) if fb.exists() else None
    tb = H3 / "h25b_targets.csv"
    T = rd(tb if tb.exists() else H3 / "h25_targets.csv")
    g = T.groupby("target").agg(parent=("parent", "first"), n_cells=("n_cells", "first"), n_blocks=("n_blocks", "first"),
                                n_A_min=("n_A", "min"), n_A_max=("n_A", "max"), n_eval_min=("n_eval", "min"), n_eval_max=("n_eval", "max"),
                                E_own=("E_own", "first"), E0=("E0", "mean"),
                                frac_src_parent=("frac_src_parent", "mean") if "frac_src_parent" in T else ("E0", "size"),
                                n_valid_splits=("n_valid_splits", "first") if "n_valid_splits" in T else ("E0", "size")).reset_index()
    g["E_ratio"] = g.E_own / g.E0; g["abs_logE"] = np.abs(np.log(g.E_ratio)); g["error_type"] = g.abs_logE.map(err_class)
    g.loc[g.target == "CA-1", "error_type"] = "excluded (3 A-block labels)"
    g = g.sort_values(["parent", "target"]).assign(section="transfer target")
    full = g.copy()
    if Fb is not None:
        Fr = Fb.assign(section="region (Fig. 1b)", target=Fb.region)
        full = pd.concat([Fr, g], ignore_index=True)
    show_t = pd.DataFrame({
        "Target": g.target.map(ps.region_name), "Parent": g.parent.map(ps.region_name),
        "Cells / blocks": g.n_cells.map(lambda v: f"{int(v):,}") + " / " + g.n_blocks.astype(int).astype(str),
        "A-block cells": [f"{int(a):,}" if a == b else f"{int(a):,}–{int(b):,}" for a, b in zip(g.n_A_min, g.n_A_max)],
        "E or E_own": g.E_own.map(lambda v: f"{v:.2f}"), "E0": g.E0.map(lambda v: f"{v:.2f}"),
        "|log(E_own/E0)|": g.abs_logE.map(lambda v: f"{v:.3f}"), "Type": g.error_type,
        "Same-parent source (%)": (100 * g.frac_src_parent).map(lambda v: f"{v:.0f}") if "frac_src_parent" in T else ""})
    if Fb is not None:
        show_r = pd.DataFrame({"Target": Fb.region.map(ps.region_name), "Parent": "", "Cells / blocks":
                               Fb.n_cells.map(lambda v: f"{int(v):,}") + " / " + Fb.n_blocks.astype(int).astype(str),
                               "A-block cells": "", "E or E_own": [f"{e:.2f} [{a:.2f}, {b:.2f}]" if np.isfinite(a) else f"{e:.2f}"
                                                            for e, a, b in zip(Fb.E, Fb.E_lo, Fb.E_hi)],
                               "E0": "", "|log(E_own/E0)|": "", "Type": "", "Same-parent source (%)": ""})
        show = pd.concat([show_r, show_t], ignore_index=True)
    else:
        show = show_t
    caption = dict(
        definition=("Supplementary Table 1. Scoring cells, blocks and Stefan coefficients. Upper rows: regions of Fig. 1b with "
                    "E = exp(mean z), z = log ALT − log √TDD, and its 95 % block-bootstrap CI (regions with ≥ 8 blocks). Lower rows: "
                    "the 15 transfer targets, with E_own fitted on all target cells (oracle) and E0 fitted on source cells."),
        statistics=("A-block cells give the label pool range over three splits; E0 is the split mean. Error type: structure "
                    "|log(E_own/E0)| < 0.15, level ≥ 0.2, intermediate between; CA-1 is excluded. Same-parent source is the "
                    "share of source cells from the target's parent region."),
        panels="Cell counts follow polar.m1_core.load_base (population 17,467 cells).",
        data=f"Source: outputs/figures/paper/source_data/Fig1_b.csv; {T.attrs.get('source')}.",
    )
    return write_table("TableS1_cells_blocks_E", full, show, caption, col_fmt="llrrlllll")


# ================================================================ ST2 Holm 가족과 조정 p
def _holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def table_ST2(draft: bool = False):
    parts = []
    M = load_m1_tests(valid_only=False)
    q = M[M.p_holm.notna()]
    parts.append(pd.DataFrame(dict(figure="Fig. 1d, 2 (M1)", source="m1/m1_sc_tests.csv", family=q.H.astype(str) + " / " + q.holm_group.astype(str),
                                   test=q.H, cond=q.cond, target=q.target, delta_cm=q.delta_rmse, ci_lo=q.ci_lo, ci_hi=q.ci_hi,
                                   p_boot=q.p_boot, p_holm=q.p_holm, holm_m=q.holm_m)))
    A = rd(H2 / "h_tests_all.csv"); q = A[A.p_holm.notna()]
    parts.append(pd.DataFrame(dict(figure="Fig. 6a (H18 to H22)", source="h2/h_tests_all.csv", family=q.family + " / " + q.cond, test=q.test,
                                   cond=q.cond, target=q.target, delta_cm=q.delta, ci_lo=q.ci_lo, ci_hi=q.ci_hi, p_boot=q.p_boot,
                                   p_holm=q.p_holm, holm_m=np.nan)))
    H = rd(H3 / "h28_tests.csv"); q = H[H.is_mean.astype(str).str.lower().isin(["true", "1"]) & ~H.target.str.contains("Alaska")].copy()
    q["p_holm8"] = _holm(q.p_boot.values)
    parts.append(pd.DataFrame(dict(figure="Fig. 2d (H28)", source="h3/h28_tests.csv", family="H28, all 8 AB4 mean tests",
                                   test=q.test, cond=q.cond, target=q.target, delta_cm=q.delta, ci_lo=q.ci_lo, ci_hi=q.ci_hi,
                                   p_boot=q.p_boot, p_holm=q.p_holm8, holm_m=len(q))))
    H = rd(H3 / "h30_deploy_tests.csv"); q = H[H.p_holm.notna()]
    parts.append(pd.DataFrame(dict(figure="Fig. 6b (H30)", source="h3/h30_deploy_tests.csv", family=q.family, test=q.test, cond=q.cond,
                                   target=q.target, delta_cm=q.delta, ci_lo=q.ci_lo, ci_hi=q.ci_hi, p_boot=q.p_boot, p_holm=q.p_holm,
                                   holm_m=np.nan)))
    for tag, kind, fig_ in (("c1", "tests", "Fig. 6c (C1)"), ("b4", "tests", "Supplementary Fig. 7 (B4)")):
        try:
            B = load_h4(tag, kind, draft)
        except MissingData:
            continue
        q = B[B.p_holm.notna()]
        parts.append(pd.DataFrame(dict(figure=fig_, source=B.attrs["source"].replace("data/processed/", ""),
                                       family=(q.family if "family" in q else q.test_name) , test=q.test, cond=q.get("cond", ""),
                                       target=q.target, delta_cm=q.delta, ci_lo=q.ci_lo, ci_hi=q.ci_hi, p_boot=q.p_boot,
                                       p_holm=q.p_holm, holm_m=np.nan)))
    full = pd.concat(parts, ignore_index=True)
    show = full[~full.target.astype(str).isin(["Russia_C", "Greenland"])].copy()
    show = pd.DataFrame({"Figure": show.figure, "Family": show.family, "Test": show.test, "Target": show.target.astype(str).str.replace(
        "Lena,Canada,Russia_W,Russia_E", "AB4").str.replace("REGION_SUMMARY_", "mean "),
        "Δ [95 % CI] (cm)": [fmt_ci(d, a, b, 2) if np.isfinite(a) else ps.fmt_num(d, 2) for d, a, b in zip(show.delta_cm, show.ci_lo, show.ci_hi)],
        "p": show.p_boot.map(lambda v: f"{v:.3f}"), "Holm p": show.p_holm.map(lambda v: f"{v:.3f}")})
    caption = dict(
        definition=("Supplementary Table 2. Holm families and adjusted p values for the confirmatory tests referenced in the "
                    "figures. p is the two-sided paired bootstrap p value; Holm p is adjusted within the listed family."),
        statistics=("H28 is adjusted over all eight AB4 mean tests (nested selection, exploratory and reference contrasts): the "
                    "pre-specified λ = 0.25 recipe remains significant (Holm p 0.032) while λ = 0.5 does not (0.112); the registered "
                    "H28 nested-selection hypothesis is rejected. H23 and H24 were not part of any Holm family."),
        panels="All rows with a Holm p value; region-level rows are in the CSV file together with the family sizes.",
        data="Source: data/processed/m1/m1_sc_tests.csv, h2/h_tests_all.csv, h3/h28_tests.csv, h3/h30_deploy_tests.csv, h4/c1_tests.csv, h4/b4_tests.csv.",
    )
    return write_table("TableS2_holm_families", full, show, caption, col_fmt="lllllrr")


# ================================================================ ST4 B1 프로토콜 전 방법·τ + Fig 2d 두 채점
def table_ST4(draft: bool = False):
    P = load_h4("b1", "protocol", draft)
    show = pd.DataFrame({"Method": P.method, "Analysis": P.analysis.str.replace("_", " "), "n": P.n.astype(int).astype(str),
                         "Worse (point)": P.n_worse.astype(int).astype(str) + " / " + P.n_targets.astype(int).astype(str),
                         "Worse (block CI)": P.n_worse_ci.astype(int).astype(str), "Better (block CI)": P.n_better_ci.astype(int).astype(str),
                         "Mean Δ (cm)": P.mean_d.map(lambda v: ps.fmt_num(v, 2)), "Worst Δ (cm)": P.worst_d.map(lambda v: ps.fmt_num(v, 2)),
                         "Worst target": P.worst_target.map(ps.region_name),
                         "Gap to oracle branch (cm)": P.gap_to_oracle.map(lambda v: ps.fmt_num(v, 2) if np.isfinite(v) else "")})
    cap = dict(
        definition=("Supplementary Table 4a. Two-stage protocol (B1): all methods and gating thresholds τ over 14 targets. Δ is "
                    "RMSE minus physics-anchor RMSE (cm). Worse counts targets with Δ > 0 (point) or with the 95 % block-bootstrap CI "
                    "above 0; Better counts CIs below 0. Gap to oracle branch is the mean Δ difference to the post hoc best branch."),
        statistics=("protocol@loto selects τ by leave-one-target-out; its selected τ counts are in the CSV file. Explore rows were "
                    "added after results were viewed and are outside the pre-registered analysis."),
        panels="Rows by method and n (3 or 10 target labels).",
        data=f"Source: {P.attrs.get('source')}; block CIs from h4/b1_blk.csv.",
    )
    ra = write_table("TableS4a_b1_protocol", P, show, cap, col_fmt="llrrrrrrlr")
    H = rd(H3 / "h28_tests.csv")
    q = H[H.is_mean.astype(str).str.lower().isin(["true", "1"])].copy()
    showb = pd.DataFrame({"Test": q.test, "Target": q.target.str.replace("Lena,Canada,Russia_W,Russia_E", "AB4"), "Role": q.role,
                          "Cell-weighted Δ [95 % CI]": [fmt_ci(d, a, b, 2) for d, a, b in zip(q.delta, q.ci_lo, q.ci_hi)],
                          "Block-equal Δ [95 % CI]": [fmt_ci(d, a, b, 2) for d, a, b in zip(q.delta_blockeq, q.ci_lo_beq, q.ci_hi_beq)]})
    capb = dict(
        definition=("Supplementary Table 4b. Recipe selection (H28, Fig. 2d) under two scorings: cell-weighted RMSE and "
                    "block-equal RMSE (each scoring block weighted equally). Δ is RMSE minus physics-anchor RMSE (cm)."),
        statistics="CIs are stratified block-bootstrap intervals over the AB4 regions (1,000 resamples).",
        panels="Rows: registered nested selection, exploratory variants, pre-specified references and the Alaska fold check.",
        data="Source: data/processed/h3/h28_tests.csv.",
    )
    rb = write_table("TableS4b_h28_two_scorings", q, showb, capb, col_fmt="lllrr")
    return [ra, rb]


# ================================================================ ST7 D1 외부 홀드아웃 절대 RMSE·편향
def table_ST7(draft: bool = False):
    D = load_d1(draft)
    show = pd.DataFrame({"Method": D.method + np.where(D.exploratory, " (exploratory)", ""), "n": D.n.astype(int).astype(str),
                         "τ": D.tau.map(lambda v: "" if v < 0 else f"{v:.2f}"),
                         "RMSE (cm)": D.rmse_mean.map(lambda v: f"{v:.1f}"), "Physics RMSE (cm)": D.rmse_phys.map(lambda v: f"{v:.1f}"),
                         "Bias (cm)": D.bias_mean.map(lambda v: ps.fmt_num(v, 1)),
                         "Δ [95 % CI] (cm)": [fmt_ci(d, a, b, 1) for d, a, b in zip(D.blk_d_phys, D.blk_d_phys_lo, D.blk_d_phys_hi)],
                         "Δ (% of physics)": D.rel_d.map(lambda v: ps.fmt_num(v, 1))})
    cap = dict(
        definition=("Supplementary Table 7. External hold-out D1 (Mongolia and Central Asia, 46 labelled cells). Δ is RMSE minus "
                    "physics-anchor RMSE (cm); bias is mean prediction minus label. refit_allA and shrink_allA use all A-block "
                    "labels (reference, not an upper bound)."),
        statistics=("95 % block-bootstrap CIs over 3 splits. All D1 labels are thaw depths derived from ground-temperature profiles "
                    "(fidelity 3), unlike the probe-based source labels, so the physics bias mixes regional and label-definition "
                    "differences. offset_only was added after results were viewed (exploratory, outside F11)."),
        panels="Rows by method, n target labels and gating threshold τ (protocol only).",
        data=f"Source: {D.attrs.get('source')}; block CIs from h4/d1_blk.csv.",
    )
    return write_table("TableS7_d1_external", D, show, cap, col_fmt="lrrrrrrr")


# ================================================================ S11 소프트웨어·시드·실행 시간(표)
def table_S11(draft: bool = False):
    import glob
    import importlib
    rows = []
    for f in sorted(glob.glob(str(PROC / "*" / "*_meta.json"))):
        if "_smoke" in f:
            continue
        try:
            m = json.loads(open(f).read())
        except Exception:                                   # noqa: BLE001
            continue
        el = m.get("elapsed_s")
        try:
            el = float(el)
        except (TypeError, ValueError):
            el = np.nan
        seeds = m.get("seeds", m.get("n_seed", ""))
        rows.append(dict(file=str(Path(f).relative_to(ROOT)), stage=str(m.get("stage", ""))[:60], git_commit=m.get("git_commit", ""),
                         seeds=str(seeds)[:40], splits=str(m.get("splits", ""))[:20], reps=m.get("reps", ""), nboot=m.get("nboot", ""),
                         elapsed_h=el / 3600 if np.isfinite(el) else np.nan))
    Mt = pd.DataFrame(rows)
    Mt = Mt[Mt.file.str.contains("/h[234]/|/m1/")]
    vers = []
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "catboost", "torch", "matplotlib", "cartopy", "shapely", "pyproj", "cmcrameri"):
        try:
            mod = importlib.import_module(pkg)
            vers.append(dict(package=pkg, version=getattr(mod, "__version__", "")))
        except Exception:                                   # noqa: BLE001
            vers.append(dict(package=pkg, version="not installed"))
    import platform
    vers.append(dict(package="python", version=platform.python_version()))
    V = pd.DataFrame(vers)
    capv = dict(definition="Supplementary Table 10a. Software versions used for the analyses and figures.",
                statistics="Versions are read from the analysis environment at figure build time.",
                panels="One row per package.", data="Source: runtime environment of scripts/4_visualization/paper_figs.py.")
    ra = write_table("TableS10a_software", V, V, capv, col_fmt="ll")
    showm = Mt[Mt.elapsed_h.notna()].assign(elapsed=lambda d: d.elapsed_h.map(lambda v: f"{v:.2f}"))[["file", "git_commit", "seeds", "splits", "reps", "nboot", "elapsed"]]
    showm = showm.rename(columns={"file": "Run metadata", "git_commit": "Commit", "seeds": "Seeds", "splits": "Splits", "reps": "Repeats",
                                  "nboot": "Bootstrap", "elapsed": "Run time (h)"})
    showm["Run metadata"] = showm["Run metadata"].str.replace("data/processed/", "", regex=False)
    capm = dict(definition="Supplementary Table 10b. Experiment runs: seeds, splits, repeats, bootstrap resamples and wall-clock run time.",
                statistics="Run times are single-node wall-clock hours on the shared server (CPU, 4 to 6 threads per process).",
                panels="Runs with recorded run time; all metadata files are listed in the CSV file.",
                data="Source: data/processed/*/*_meta.json.")
    rb = write_table("TableS10b_runs", Mt, showm, capm, col_fmt="llllrrr")
    return [ra, rb]


# ================================================================ S7 B4 표형 파운데이션 모델
S7_N = [(3, "solid"), (10, "dashed"), (40, "dotted")]
S7_LIM = (-3.0, 3.0)


def build_S7(draft: bool = False):
    Sm = load_b4("summary", draft)
    Tt = load_b4("tests", draft)
    alr = abslogE_map()
    tg = sorted(Sm.target.unique(), key=lambda t: alr.get(t, np.inf))
    labels = [ps.region_label(t, alr.get(t)) for t in tg] + ["Mean over targets"]
    ps.use_paper()
    fig = ps.paper_figure(180, 96)
    ax_a = ps.slot_mm(fig, 0, 17, 33, 62, 70)
    ax_b = ps.slot_mm(fig, 107, 17, 3, 62, 70)
    rows = []
    q = Sm[np.isclose(Sm.lam, 0.25) & (Sm.method == "tfm")]
    for k, (n, var) in enumerate(S7_N):
        off = 0.26 * (1 - k)
        e, lo, hi = [], [], []
        for t in tg:
            r = q[(q.target == t) & (q.n == n)]
            if len(r):
                r = r.iloc[0]; e.append(r.d_cb_blk); lo.append(r.d_cb_lo); hi.append(r.d_cb_hi)
                rows.append(dict(panel="a", target=t, n=n, d_tfm_minus_cb=r.d_cb_blk, ci_lo=r.d_cb_lo, ci_hi=r.d_cb_hi, ci_kind="block"))
            else:
                e.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        m = Tt[(Tt.test_name == "tfm_vs_cb") & (Tt.n == n) & np.isclose(Tt.lam, 0.25) & (Tt.scope != "region")
               & (Tt.target.str.count(",") >= 9)]
        if len(m):
            m = m.iloc[0]; e.append(m.delta); lo.append(m.ci_lo); hi.append(m.ci_hi)
            rows.append(dict(panel="a", target=m.target, n=n, d_tfm_minus_cb=m.delta, ci_lo=m.ci_lo, ci_hi=m.ci_hi, ci_kind="strat",
                             p_holm=m.p_holm))
        else:
            e.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        ps.forest(ax_a, labels, e, lo, hi, method="tfm", offset=off, ls=var, offscale=S7_LIM, ms=2.6, better_text=(k == 0),
                  lw=0.9)
    ax_a.set_xlabel("ΔRMSE, TFM − CatBoost residual (cm)")
    ys = np.arange(len(labels))[::-1]
    ax_a.axhline(ys[-1] + 0.5, color="#d9d9d9", lw=0.4)
    # b: n = 10 Δ vs physics, TFM 대 CatBoost 잔차
    for m_, key, off in (("tfm", "tfm", 0.18), ("cb", "residual", -0.18)):
        r = Sm[np.isclose(Sm.lam, 0.25) & (Sm.method == m_) & (Sm.n == 10)].set_index("target")
        e = [r.d_phys_blk.get(t, np.nan) for t in tg] + [np.nan]
        lo = [r.d_phys_lo.get(t, np.nan) for t in tg] + [np.nan]
        hi = [r.d_phys_hi.get(t, np.nan) for t in tg] + [np.nan]
        ps.forest(ax_b, labels, e, lo, hi, method=key, offset=off, offscale=(-12.0, 6.0), sharey_with=ax_a, ms=2.6, lw=0.9,
                  better_text=(m_ == "tfm"))
        rows += [dict(panel="b", target=t, method=m_, n=10, d_phys=a, ci_lo=b, ci_hi=c, ci_kind="block") for t, a, b, c in zip(tg, e, lo, hi)]
    ax_b.set_xlabel("ΔRMSE vs physics at n = 10 (cm)")
    ax_b.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-10, -5, 0, 5])); ax_b.xaxis.set_major_formatter(_minus_fmt())
    ax_a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-3, -2, -1, 0, 1, 2, 3])); ax_a.xaxis.set_major_formatter(_minus_fmt())
    ax_b.axhline(ys[-1] + 0.5, color="#d9d9d9", lw=0.4)
    handles = [Line2D([], [], color=ps.COLOR["tfm"], ls=ps.VARIANT_LS[v], lw=0.9, marker="s", ms=2.6, label=f"TFM − CatBoost, n = {n} (a)")
               for n, v in S7_N]
    handles += [ps.method_handle("tfm", "TFM residual (b)"), ps.method_handle("residual", "CatBoost residual (b)")]
    ps.legend_below(fig, handles, (0, 0, 180, 8), ncol=5)
    ps.label_panels([ax_a, ax_b], "ab")
    caption = dict(
        definition=("Tabular foundation model as residual learner (B4). TFM replaces CatBoost as the residual model on the physics "
                    "anchor (λ = 0.25, k-medoid labels). ΔRMSE in cm; negative favours TFM in a and the method in b."),
        statistics=("Target rows: 95 % block-bootstrap CIs (1,000 resamples, indices shared across methods). Mean row: stratified "
                    "block bootstrap over the targets tested at that n (13 targets at n = 3 and 10; 11 at n = 40). Holm-adjusted p "
                    "values in Supplementary Table 2."),
        panels=(f"a, TFM minus CatBoost residual at n = 3, 10 and 40 (line style). b, Both residual learners against the physics "
                f"anchor at n = 10. Rows ordered by |log(E_own/E0)|; values outside the axis are drawn as filled triangles at the edge."),
        data=f"Source data: {Sm.attrs.get('source')}, {Tt.attrs.get('source')}.",
    )
    spec = dict(intent="B4 TFM 대 CatBoost 잔차 포레스트(n = 3·10·40) + n = 10 물리식 대비", panels=2,
                assets=[Sm.attrs.get("source"), Tt.attrs.get("source")], script="scripts/4_visualization/paper_figs.py --only supp")
    return save_paper(fig, "FigS7_tfm", caption=caption, spec=spec, draft=draft and bool(Sm.attrs.get("is_smoke")),
                      sources={"ab": pd.DataFrame(rows)})


# ================================================================ S8 CCI 편향 전파
def build_S8(draft: bool = False):
    Cc = load_h4("c1", "cells", draft)
    base = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon"])
    Cc = Cc.merge(base, on="loc_id", how="left")
    Cc = Cc[np.isfinite(Cc.cci_alt) & np.isfinite(Cc.y)].copy()
    Cc["cci_err"] = Cc.cci_alt - Cc.y
    Cc["comb_err"] = Cc.pred_stefan_cci - Cc.y
    Cc["phys_err"] = Cc.pred_stefan - Cc.y
    ps.use_paper()
    fig = ps.paper_figure(180, 122)
    rng = np.random.default_rng(0)
    order = ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]
    TOP = 78.0                                                   # 상단 행(a–c) 축 하단 y(mm); 지도(d)는 그 아래 96 × 62 mm, 자료 범위에 맞춤
    # a CCI ALT 대 관측
    ax_a = ps.slot_mm(fig, 0, TOP, 14, 38, 38)
    for t in order:
        g = Cc[Cc.target == t]
        if len(g) > 400:
            g = g.iloc[rng.choice(len(g), 400, replace=False)]
        mk, sc = ps.region_marker(t)
        ax_a.plot(g.y, g.cci_alt, ls="none", marker=mk, ms=ps.MS["point"] * sc * 1.2, color=ps.COLOR["cci"], alpha=0.45, mew=0)
    lim = (5, 400)
    ax_a.plot(lim, lim, color=ps.GREY["mid"], ls=":", lw=0.7)
    for f_ in (ax_a.set_xscale, ax_a.set_yscale):
        f_("log")
    ax_a.set_xlim(*lim); ax_a.set_ylim(*lim)
    for a_ in (ax_a.xaxis, ax_a.yaxis):
        a_.set_major_locator(matplotlib.ticker.FixedLocator([10, 30, 100, 300]))
        a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
        a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_a.set_xlabel("Observed ALT (cm)"); ax_a.set_ylabel("CCI ALT (cm)")
    # b 편향 전파: CCI 오차 대 결합 앵커 오차(구간 중앙값)
    ax_b = ps.slot_mm(fig, 62, TOP, 14, 38, 38)
    bins = np.array([-150, -80, -50, -30, -15, -5, 5, 15, 30, 50, 80, 150])
    rows_b = []
    for key, col, lbl in (("phys_err", ps.COLOR["phys"], "physics anchor"), ("comb_err", ps.COLOR["cci"], "CCI-combined anchor")):
        idx = np.digitize(Cc.cci_err, bins)
        med = Cc.groupby(idx).agg(x=("cci_err", "median"), y=(key, "median"), n=(key, "size"))
        med = med[(med.n >= 20)]
        ax_b.plot(med.x, med.y, color=col, lw=1.0, marker="v" if key == "comb_err" else "o", ms=2.6, mfc=col, mec=col)
        rows_b += [dict(panel="b", series=key, cci_err_median=r.x, err_median=r.y, n_cells=int(r.n)) for _, r in med.iterrows()]
    ax_b.axhline(0, **ps.DELTA_ZERO); ax_b.axvline(0, color=ps.GREY["light"], lw=0.5, zorder=0.5)
    ax_b.plot([-150, 150], [-150, 150], color=ps.GREY["mid"], ls=":", lw=0.7)
    ax_b.set_xlim(-120, 120); ax_b.set_ylim(-60, 60)
    ax_b.set_xlabel("CCI error (cm)"); ax_b.set_ylabel("Anchor error (cm)")
    ax_b.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-100, -50, 0, 50, 100]))
    ax_b.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60]))
    ax_b.xaxis.set_major_formatter(_minus_fmt()); ax_b.yaxis.set_major_formatter(_minus_fmt())
    # c 불확도 가중 결합의 원천 분산 민감도
    ax_c = ps.slot_mm(fig, 124, TOP, 14, 38, 38)
    g = Cc[np.isfinite(Cc.pred_cci_uncw) & np.isfinite(Cc.pred_cci_uncw_srcvar)]
    gg = g.iloc[rng.choice(len(g), min(len(g), 1500), replace=False)]
    ax_c.plot(gg.pred_cci_uncw, gg.pred_cci_uncw_srcvar, ls="none", marker="o", ms=ps.MS["point"], color=ps.COLOR["cci"], alpha=0.35, mew=0)
    lim_c = (float(np.nanpercentile(g[["pred_cci_uncw", "pred_cci_uncw_srcvar"]].values, 0.5)) * 0.9,
             float(np.nanpercentile(g[["pred_cci_uncw", "pred_cci_uncw_srcvar"]].values, 99.5)) * 1.05)
    ax_c.plot(lim_c, lim_c, color=ps.GREY["mid"], ls=":", lw=0.7)
    ax_c.set_xlim(*lim_c); ax_c.set_ylim(*lim_c)
    tv = [v for v in matplotlib.ticker.MaxNLocator(5).tick_values(*lim_c) if lim_c[0] <= v <= lim_c[1]]
    ax_c.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tv)); ax_c.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tv))
    ax_c.set_xlabel("Uncertainty-weighted (cm)"); ax_c.set_ylabel("With source variance (cm)")
    d_rmse = lambda c: float(np.sqrt(np.mean((g[c] - g.y) ** 2)))    # noqa: E731
    # d 블록 평균 CCI 편향 지도(범북극)
    B = Cc.groupby(["target", "block"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), bias=("cci_err", "mean"), n=("y", "size")).reset_index()
    ax_d = _fit_map_ax(fig, (0, 3, 96, 62), dict(ll=(-180.0, 180.0, 55.0, 88.0), lon0=-45.0, glat=10, glon=30, bar=1000,
                                                  res="50m", inset=None, fit_lonlat=(B.lon.values, B.lat.values), fit_margin=0.05))
    norm = ps.shared_diverging_norm([B.bias.values], step_cm=10)
    sc = ps.density_circles(ax_d, B.lon.values, B.lat.values, B.n.values, c=B.bias.values, norm=norm, cmap_=ps.cmap("diverging"), smax=80)
    cax = ps.axes_mm(fig, 99, 11, 2.5, 46)
    cb = fig.colorbar(sc, cax=cax)
    vmx = norm.vmax
    cb.set_ticks([v for v in (-100, -50, 0, 50, 100) if -vmx <= v <= vmx])
    cb.set_label("Block-mean CCI error (cm)"); cb.ax.yaxis.set_major_formatter(_minus_fmt()); cb.outline.set_linewidth(0.4)
    ax_d.set_gid("map")
    handles = [Line2D([], [], ls="none", marker=ps.region_marker(t)[0], ms=3.2, color=ps.COLOR["cci"], label=ps.region_name(t)) for t in order]
    handles += [Line2D([], [], color=ps.COLOR["phys"], lw=1.0, marker="o", ms=2.6, label="Physics anchor error (b)"),
                Line2D([], [], color=ps.COLOR["cci"], lw=1.0, marker="v", ms=2.6, label="CCI-combined anchor error (b)")]
    ps.legend_below(fig, handles, (116, 3, 64, 62), ncol=1)
    ps.label_panels([ax_a, ax_b, ax_c, ax_d], "abcd")
    caption = dict(
        definition=("ESA CCI Permafrost ALT as a label-free input (C1). CCI error is CCI ALT minus the observed label; anchor error "
                    "is prediction minus label for the physics anchor (E0 × √TDD) and for the CCI-combined anchor. Cells of the six "
                    "label-free targets."),
        statistics=(f"b shows medians within CCI-error bins holding ≥ 20 cells. In c the uncertainty-weighted combination has RMSE "
                    f"{d_rmse('pred_cci_uncw'):.1f} cm with CCI uncertainty only and {d_rmse('pred_cci_uncw_srcvar'):.1f} cm when "
                    "source-region variance is added."),
        panels=("a, Observed versus CCI ALT (log axes; up to 400 cells per target shown), 1:1 dotted. b, Propagation of CCI error "
                "into the combined anchor; dotted 1:1 is full propagation. c, Sensitivity of the uncertainty-weighted combination "
                "to source variance (1,500 cells shown). d, Block-mean CCI error, circle area proportional to cells; polar "
                "stereographic, true scale at 70°N, extent fitted to the labelled blocks."),
        data=f"Source data: {Cc.attrs.get('source')}; coordinates from data/processed/fidelity_base_v3.csv.",
    )
    spec = dict(intent="C1 CCI 편향 전파: 산점(1:1)·오차 전파·원천 분산 민감도·블록 편향 지도", panels=4,
                assets=[Cc.attrs.get("source"), "data/processed/fidelity_base_v3.csv"], script="scripts/4_visualization/paper_figs.py --only supp")
    return save_paper(fig, "FigS8_cci_bias", caption=caption, spec=spec, draft=draft and bool(Cc.attrs.get("is_smoke")),
                      sources={"b": pd.DataFrame(rows_b), "d": B})


# ================================================================ S9 라벨 연도 민감도
S9_M = [("stefan", "phys", "o", "Physics anchor (E fitted)"), ("stefan_ridge", "refit", "s", "Physics + ridge covariate offset"),
        ("catboost_lo", "direct", "x", "Direct ML (CatBoost)")]


def build_S9(draft: bool = False):
    L = rd(M1 / "label_year_sensitivity.csv")
    Tn = L[L.table == "delta_trainset"].copy()
    Sb = L[(L.table == "delta_subset") & (L.region.isin(["Lena", "Canada", "MAIN6"])) & (L.train_set == "alaska_all")].copy()
    ps.use_paper()
    fig = ps.paper_figure(180, 84)
    subs = [("overlap-all", "overlap"), ("overlap-all", "inside"), ("overlap-all", "before"),
            ("before-all", "overlap"), ("before-all", "inside"), ("before-all", "before")]
    lab_a = [f"{'2015–2020' if ts == 'overlap-all' else 'pre-2015'} labels; test {sb_}" for ts, sb_ in subs]
    ax_a = ps.slot_mm(fig, 0, 18, 40, 44, 56)
    lim = (-3.0, 5.0)
    rows = []
    for k, (m_, key, mk, lbl) in enumerate(S9_M):
        e, lo, hi = [], [], []
        for ts, sb_ in subs:
            r = Tn[(Tn.method == m_) & (Tn.train_set == ts) & (Tn.subset == sb_)]
            r = r.iloc[0] if len(r) else None
            e.append(r.d_rmse if r is not None else np.nan); lo.append(r.d_rmse_lo if r is not None else np.nan)
            hi.append(r.d_rmse_hi if r is not None else np.nan)
            rows.append(dict(panel="a", method=m_, train_set=ts, subset=sb_, d_rmse=e[-1], ci_lo=lo[-1], ci_hi=hi[-1]))
        ya = ps.forest(ax_a, lab_a, e, lo, hi, method=key, marker=mk, offset=0.26 * (1 - k), offscale=lim, ms=2.8, lw=0.9,
                       better_text=(k == 0))
        for yv_, v_ in zip(ya + 0.26 * (1 - k), e):                 # 범위 밖 점 추정값은 축 끝 삼각형 옆에 수치로(Fig. 3e·S4 관례)
            if np.isfinite(v_) and not (lim[0] <= v_ <= lim[1]):
                up = v_ > lim[1]
                ax_a.annotate(ps.fmt_num(v_, 1), xy=(lim[1] if up else lim[0], yv_), xytext=(-3 if up else 3, 3.2),
                              textcoords="offset points", ha="right" if up else "left", va="bottom", fontsize=ps.FS["annot"],
                              color=ps.COLOR[key], annotation_clip=False)
    ax_a.set_xlabel("ΔRMSE, subset-trained − all-trained (cm)")
    ys = np.arange(len(subs))[::-1]
    ax_a.axhline(ys[2] - 0.5, color="#d9d9d9", lw=0.4)
    # b 전이: 라벨 연도 부분집합 간 RMSE 차이(서로 다른 셀)
    cons = ["overlap-before", "inside-before", "inside-overlap"]
    regs = ["Lena", "Canada", "MAIN6"]
    lab_b = [f"{'Main regions' if r_ == 'MAIN6' else ps.region_name(r_)}: {c_.replace('-', ' − ')}" for r_ in regs for c_ in cons]
    ax_b = ps.slot_mm(fig, 90, 18, 37, 46, 56)
    for k, (m_, key, mk) in enumerate((("stefan", "phys", "o"), ("stefan_cci", "cci", "v"))):
        e, lo, hi = [], [], []
        for r_ in regs:
            for c_ in cons:
                r = Sb[(Sb.region == r_) & (Sb.subset == c_) & (Sb.method == m_)]
                r = r.iloc[0] if len(r) else None
                e.append(r.d_rmse if r is not None else np.nan); lo.append(r.d_rmse_lo if r is not None else np.nan)
                hi.append(r.d_rmse_hi if r is not None else np.nan)
                rows.append(dict(panel="b", method=m_, region=r_, contrast=c_, d_rmse=e[-1], ci_lo=lo[-1], ci_hi=hi[-1]))
        ps.forest(ax_b, lab_b, e, lo, hi, method=key, marker=mk, offset=0.18 * (1 - 2 * k), offscale=(-10.0, 20.0), ms=2.8, lw=0.9,
                  better_text=False)
    ax_b.set_xlabel("ΔRMSE between label-year subsets (cm)")
    ax_b.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-10, 0, 10, 20])); ax_b.xaxis.set_major_formatter(_minus_fmt())
    ax_a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-2, 0, 2, 4])); ax_a.xaxis.set_major_formatter(_minus_fmt())
    for j in (3, 6):
        ax_b.axhline(len(lab_b) - j - 0.5, color="#d9d9d9", lw=0.4)
    handles = [Line2D([], [], color=ps.COLOR[k_], ls="none", marker=mk, ms=2.8, mfc=ps.COLOR[k_], mec=ps.COLOR[k_], label=l_)
               for _, k_, mk, l_ in S9_M]
    handles.append(Line2D([], [], color=ps.COLOR["cci"], ls="none", marker="v", ms=2.8, label="CCI-combined anchor (b)"))
    ps.legend_below(fig, handles, (0, 0, 180, 8), ncol=4)
    ps.label_panels([ax_a, ax_b], "ab", dx_mm=0)
    caption = dict(
        definition=("Label-year sensitivity (D3). ERA5-Land covariates are 2015–2020 climatologies while labels span 1993–2023. "
                    "a, Alaska in-domain: models trained on labels from 2015–2020 (overlap) or before 2015, minus the same model "
                    "trained on all labels, scored on the same test cells. b, Transfer from Alaska: RMSE difference between label-year "
                    "subsets of the target cells."),
        statistics=("95 % CIs from 400 block-bootstrap resamples (paired on cells in a; unpaired subsets in b). Six-fold block "
                    "cross-validation, three seeds for CatBoost."),
        panels=("a, Rows: training subset and test subset; point estimates outside −3 to 5 cm are drawn as filled triangles at the axis "
                "edge with the value printed beside them; CIs are clipped at the axis limits. b, Rows: region and subset contrast (inside, labels within 2015–2020; before, before 2015; main regions, "
                "cell-pooled Lena, Canada and Russia); CIs are clipped at the axis limits."),
        data="Source data: data/processed/m1/label_year_sensitivity.csv (tables delta_trainset and delta_subset).",
    )
    spec = dict(intent="D3 라벨 연도 민감도: 학습 연도 부분집합 짝지은 Δ(알래스카) + 전이 부분집합 차이", panels=2,
                assets=["data/processed/m1/label_year_sensitivity.csv"], script="scripts/4_visualization/paper_figs.py --only supp")
    return save_paper(fig, "FigS9_label_year", caption=caption, spec=spec, draft=False, sources={"ab": pd.DataFrame(rows)})


# ================================================================ S1 ALT 대 √TDD 산점 + 공변량 표
S1_REG = ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]
COV_DESC = {
    "dem_elev": ("Terrain", "Elevation", "m", "Copernicus DEM, 30 m"), "dem_slope": ("Terrain", "Slope", "°", "Copernicus DEM, 30 m"),
    "dem_aspect_sin": ("Terrain", "Aspect, sine", "–", "Copernicus DEM, 30 m"), "dem_aspect_cos": ("Terrain", "Aspect, cosine", "–", "Copernicus DEM, 30 m"),
    "dem_tpi": ("Terrain", "Topographic position index", "m", "Copernicus DEM, 30 m"), "dem_rough": ("Terrain", "Roughness", "m", "Copernicus DEM, 30 m"),
    "e5_maat": ("Climate", "Mean annual air temperature", "°C", "ERA5-Land, 2015–2020"), "e5_tdd": ("Climate", "Thawing degree-days", "°C d", "ERA5-Land, 2015–2020"),
    "e5_fdd": ("Climate", "Freezing degree-days", "°C d", "ERA5-Land, 2015–2020"), "e5_sqrt_tdd": ("Climate", "Square root of TDD", "(°C d)^0.5", "ERA5-Land, 2015–2020"),
    "e5_twarm": ("Climate", "Warmest-month temperature", "°C", "ERA5-Land, 2015–2020"), "e5_tcold": ("Climate", "Coldest-month temperature", "°C", "ERA5-Land, 2015–2020"),
    "e5_stl1": ("Climate", "Top-layer soil temperature", "°C", "ERA5-Land, 2015–2020"), "e5_swe": ("Climate", "Snow water equivalent", "m", "ERA5-Land, 2015–2020"),
    "sg_clay_5_15": ("Soil", "Clay, 5–15 cm", "g/kg", "SoilGrids 250 m"), "sg_sand_5_15": ("Soil", "Sand, 5–15 cm", "g/kg", "SoilGrids 250 m"),
    "sg_silt_5_15": ("Soil", "Silt, 5–15 cm", "g/kg", "SoilGrids 250 m"), "sg_bdod_5_15": ("Soil", "Bulk density, 5–15 cm", "cg/cm³", "SoilGrids 250 m"),
    "sg_cfvo_5_15": ("Soil", "Coarse fragments, 5–15 cm", "cm³/dm³", "SoilGrids 250 m"), "sg_phh2o_5_15": ("Soil", "pH (H2O), 5–15 cm", "pH × 10", "SoilGrids 250 m"),
    "sg_soc_0_5": ("Soil", "Soil organic carbon, 0–5 cm", "dg/kg", "SoilGrids 250 m"), "sg_soc_5_15": ("Soil", "Soil organic carbon, 5–15 cm", "dg/kg", "SoilGrids 250 m"),
    "sg_soc_15_30": ("Soil", "Soil organic carbon, 15–30 cm", "dg/kg", "SoilGrids 250 m"),
    "cci_alt": ("Remote sensing", "ESA CCI Permafrost ALT", "cm", "CCI Permafrost v4, 1 km"), "cci_valid": ("Remote sensing", "CCI ALT available", "0/1", "CCI Permafrost v4"),
}


def build_S1(draft: bool = False):
    from polar.m1_core import load_base, INPUT_SETS
    df = load_base(PROC)
    df = df[np.isfinite(df.e5_sqrt_tdd) & (df.e5_sqrt_tdd > 0) & np.isfinite(df.alt_cm) & (df.alt_cm > 0)]
    ps.use_paper()
    fig = ps.paper_figure(180, 92)
    rng = np.random.default_rng(0)
    W, H, LAB, GAP = 30.2, 30.0, 13.0, 3.3
    axes, rows = [], []
    for i, reg in enumerate(S1_REG + ["pooled"]):
        r_, c_ = divmod(i, 4)
        x0 = 0 if c_ == 0 else LAB + c_ * (W + GAP + 7.0) - GAP - 7.0 + 0.0
        ax = ps.slot_mm(fig, x0 if c_ == 0 else x0, [52.0, 13.0][r_], LAB if c_ == 0 else GAP + 7.0, W, H)
        g = df if reg == "pooled" else df[df.macro == reg]
        x, y = g.e5_sqrt_tdd.values, g.alt_cm.values
        if len(g) > 800:
            k = rng.choice(len(g), 800, replace=False)
            xs, ys_ = x[k], y[k]
        else:
            xs, ys_ = x, y
        ax.plot(xs, ys_, ls="none", marker="o", ms=ps.MS["point"], color=ps.GREY["mid"], alpha=ps.POINT_ALPHA if len(g) > 30 else 0.8, mew=0)
        E_lin = float(np.sum(x * y) / np.sum(x * x))                       # 원점 통과 최소제곱(선형 공간)
        E_log = float(np.exp(np.mean(np.log(y) - np.log(x))))             # 로그 공간(E = exp(mean z))
        xx = np.linspace(0, 60, 50)
        ax.plot(xx, E_lin * xx, color=ps.COLOR["refit"], lw=1.0)
        ax.plot(xx, E_log * xx, color=ps.COLOR["refit"], lw=1.0, ls=ps.VARIANT_LS["dashed"])
        ax.set_xlim(0, 60); ax.set_ylim(0, 250)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 20, 40, 60]))
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 100, 200]))
        nm = "All regions" if reg == "pooled" else ps.region_name(reg)
        ax.text(0.03, 0.97, f"{nm}\nn = {len(g):,}", transform=ax.transAxes, ha="left", va="top", fontsize=ps.FS["annot"], linespacing=1.1)
        # 규칙: 적합선이 우상단 글자 띠(y > 0.8·250 cm, x > 36)에 들어오면 E 값을 우하단 고정 위치로 옮기고 흰 바탕을 둔다
        steep = max(E_lin, E_log) * 60 > 0.8 * 250
        ax.text(0.97, 0.03 if steep else 0.97, f"E {E_lin:.2f} / {E_log:.2f}", transform=ax.transAxes, ha="right",
                va="bottom" if steep else "top", fontsize=ps.FS["annot"], color=ps.COLOR["refit"], zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        n_off = int(np.sum(y > 250))
        if n_off:
            ax.text(0.97, 0.85, f"{n_off} above 250 cm", transform=ax.transAxes, ha="right", va="top", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
        if c_ == 0:
            ax.set_ylabel("ALT (cm)")
        if r_ == 1:
            ax.set_xlabel("√TDD (√(°C d))")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        rows.append(dict(region=reg, n_cells=len(g), E_linear_origin_ls=E_lin, E_log_space=E_log, n_above_250=n_off))
        axes.append(ax)
    handles = [Line2D([], [], color=ps.COLOR["refit"], lw=1.0, label="Least squares through origin (linear space)"),
               Line2D([], [], color=ps.COLOR["refit"], lw=1.0, ls=ps.VARIANT_LS["dashed"], label="E = exp(mean z) (log space)"),
               Line2D([], [], ls="none", marker="o", ms=2.4, color=ps.GREY["mid"], label="Label cell (up to 800 shown)")]
    ps.legend_below(fig, handles, (0, 0, 180, 5), ncol=3)
    ps.label_panels(axes, "abcdefgh", dy_mm=1.2)
    caption = dict(
        definition=("Stefan relation by region. ALT is the probe-measured active-layer thickness label; √TDD is the square root "
                    "of the ERA5-Land thawing degree-day sum (2015–2020). The slope E of ALT = E·√TDD is fitted two ways: least "
                    "squares through the origin in linear space and E = exp(mean z), z = log ALT − log √TDD (Fig. 1b)."),
        statistics=("Numbers give E for the two fits (linear / log). Fits use all label cells of the region; points are a random "
                    "subsample."),
        panels=("a to g, Regions in Table 1 order; h, all regions pooled. Axes are clipped at 250 cm; the count of "
                "cells above is shown. The 25 covariates are listed in Supplementary Table 9."),
        data="Source data: data/processed/fidelity_base_v3.csv via polar.m1_core.load_base (17,467 cells).",
    )
    spec = dict(intent="S1: 지역별 ALT 대 √TDD 산점과 두 E 적합(선형 원점 통과 최소제곱·로그 공간) + 공변량 25 표", panels=8,
                assets=["data/processed/fidelity_base_v3.csv"], script="scripts/4_visualization/paper_figs.py --only supp")
    res = save_paper(fig, "FigS1_stefan_scatter", caption=caption, spec=spec, draft=False, sources={"a-h": pd.DataFrame(rows)})
    feats = INPUT_SETS["x25"]
    C = pd.DataFrame([dict(variable=f, group=COV_DESC.get(f, ("", "", "", ""))[0], description=COV_DESC.get(f, ("", f, "", ""))[1],
                           unit=COV_DESC.get(f, ("", "", "", ""))[2], source=COV_DESC.get(f, ("", "", "", ""))[3],
                           coverage_pct=float(100 * np.isfinite(pd.to_numeric(df[f], errors="coerce")).mean()) if f in df else np.nan)
                      for f in feats])
    showc = C.assign(coverage=C.coverage_pct.map(lambda v: f"{v:.1f}"))[["group", "variable", "description", "unit", "source", "coverage"]]
    showc.columns = ["Group", "Variable", "Description", "Unit", "Source", "Non-missing (%)"]
    capc = dict(
        definition=("Supplementary Table 9. The 25 covariates available in every region (input set x25), used by all "
                    "ML models. Label-derived quantities (observation counts, years, spread) are excluded from inputs."),
        statistics="Non-missing share over the 17,467 label cells after load_base.",
        panels="Rows by covariate group.",
        data="Source: polar.m1_core.INPUT_SETS, data/processed/fidelity_base_v3.csv, fidelity_base_v3_meta.json.",
    )
    rc = write_table("TableS9_covariates", C, showc, capc, col_fmt="lllllr")
    return [res, rc]


# ================================================================ S12 A2 전체 대상 행렬 + ST5 A2 n*
S12_ET = [("E0_fixed", "E0 fixed"), ("shrink_k10", "E shrunk (κ = 10)"), ("offset_mle", "log E offset (MLE)")]


def _f1_sentence(Cv: pd.DataFrame, q: pd.DataFrame, alr: dict) -> str:
    """F1 판정 문장(E0 고정, 라벨 전 블록 분산). 오차 유형은 err_class(오라클 |log E|, 0.15·0.20) 단일 정의 = Table S6a error_type."""
    fx = lambda v, nd=2: ps.fmt_num(v, nd)                                  # noqa: E731
    g = q[q.e_treat == "E0_fixed"]
    tg_all = sorted(g.target.unique(), key=lambda t: alr.get(t, np.inf))
    cls = {t: err_class(alr.get(t, np.nan)) for t in tg_all}
    cnt = {c: sum(v == c for v in cls.values()) for c in ("structure", "intermediate", "level")}
    struct = [t for t in tg_all if cls[t] == "structure"]
    nstar = {}
    for t in struct:
        gp = g[(g.target == t) & g["pass"].astype(bool)]
        nstar[t] = float(gp.n.min()) if len(gp) else np.nan
    nmax = {t: float(g[g.target == t].n.max()) for t in struct}
    fail = [t for t in struct if not (nstar[t] <= 40)]
    early = [t for t in struct if nstar[t] <= 40]
    nm = lambda ts: ", ".join(ps.region_name(t) for t in ts)                 # noqa: E731
    late = [t for t in fail if np.isfinite(nstar[t])]
    never = [t for t in fail if not np.isfinite(nstar[t])]
    parts = [f"F1 (pre-registered) is supported: with E0 fixed, {len(fail)} of {len(struct)} structure targets "
             f"({nm(fail)}) do not reach n* at n ≤ 40"]
    for t in late:
        r = g[(g.target == t) & (g.n == nstar[t])].iloc[0]
        parts.append(f"; {ps.region_name(t)} reaches it at n = {int(nstar[t])} (ΔRMSE {fx(r.blk_d_phys)} cm "
                     f"[{fx(r.blk_d_phys_lo)}, {fx(r.blk_d_phys_hi)}])")
    if never:
        mx = sorted({int(nmax[t]) for t in never})
        parts.append(f", the others not by n_max ({' or '.join(str(v) for v in mx)})")
    parts.append(".")
    stat = "".join(parts)
    parts = []
    if early:
        z = Cv[(Cv.scope == "n0") & (Cv.e_treat == "E0_fixed") & (Cv.stage == "resid") & np.isclose(Cv.lam, 0.25)]
        z0 = {t: z[z.target == t].iloc[0] for t in early if (z.target == t).any()}
        n0_txt = ", ".join(f"{ps.region_name(t)} {fx(r.blk_d_phys)} cm [{fx(r.blk_d_phys_lo)}, {fx(r.blk_d_phys_hi)}]" for t, r in z0.items())
        ex = early[0]
        ge = g[g.target == ex].sort_values("n")
        parts.append(f" {' and '.join(ps.region_name(t) for t in early)} meet the rule at n = {int(min(nstar[t] for t in early))}, "
                     f"the smallest n tested, because the source residual already beats physics at n = 0 ({n0_txt}); target labels leave the curve flat "
                     f"({ps.region_name(ex)} {fx(ge.blk_d_phys.iloc[0])} cm at n = {int(ge.n.iloc[0])}, {fx(ge.blk_d_phys.iloc[-1])} cm "
                     f"at n = {int(ge.n.iloc[-1])}).")
    parts.append(f" Error types as in Supplementary Table 6a: {cnt['structure']} structure, {cnt['intermediate']} intermediate, "
                 f"{cnt['level']} level (oracle |log(E_own/E0)| < 0.15, 0.15 to 0.20, ≥ 0.20).")
    return stat, "".join(parts).strip()


def build_S12(draft: bool = False):
    from _common import load_a2_curve, minn_pass
    Cv = load_a2_curve(draft)
    smoke = bool(Cv.attrs.get("is_smoke"))
    q = Cv[(Cv.scope == "n") & (Cv.spread == "all_blocks") & (Cv.stage == "resid") & np.isclose(Cv.lam, 0.25) & (Cv.n > 0)].copy()
    q = q[~q.target.astype(str).str.startswith("Alaska_f") & (q.target != "CA-1")]
    if not len(q):
        raise MissingData("a2_curve: all_blocks·resid λ 0.25 행 없음")
    q["pass"] = minn_pass(q)
    alr = abslogE_map()
    tg = sorted(q.target.unique(), key=lambda t: alr.get(t, np.inf))
    ns = sorted(q.n.unique())
    norm = ps.shared_diverging_norm([q.blk_d_phys.values], step_cm=5)
    cm_ = ps.cmap("diverging")
    ps.use_paper()
    h_ax = max(20.0, 3.4 * len(tg))
    fig = ps.paper_figure(180, h_ax + 37)
    axes, rows = [], []
    LAB, W, GAP = 22.0, 44.0, 4.0
    for k, (et, lbl) in enumerate(S12_ET):
        ax = ps.slot_mm(fig, 0 if k == 0 else LAB + k * (W + GAP) - GAP, 27, LAB if k == 0 else GAP, W, h_ax)
        g = q[q.e_treat == et]
        M = np.full((len(tg), len(ns)), np.nan); Pm = np.zeros_like(M, bool)
        for _, r in g.iterrows():
            i, j = tg.index(r.target), ns.index(r.n)
            M[i, j] = r.blk_d_phys; Pm[i, j] = bool(r["pass"])
            rows.append(dict(panel="abc"[k], e_treat=et, target=r.target, n=int(r.n), blk_d_phys=r.blk_d_phys, ci_hi=r.blk_d_phys_hi,
                             split_win=r.split_win, rep_win=r.rep_win, minn_pass=bool(r["pass"])))
        ax.imshow(np.ma.masked_invalid(M), cmap=cm_, norm=norm, aspect="auto", interpolation="nearest",
                  extent=(-0.5, len(ns) - 0.5, len(tg) - 0.5, -0.5))
        miss = np.isnan(M)
        for i, j in zip(*np.where(miss)):
            ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=ps.GREY["missing"], edgecolor=ps.GREY["edge"],
                                                      hatch="///", lw=0, zorder=1))
        for i in range(len(tg)):                                    # 세 조건 충족 칸 테두리(계단선)
            for j in range(len(ns)):
                if Pm[i, j]:
                    ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="#000000", lw=0.8, zorder=3))
        ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"{int(v)}" for v in ns])
        ax.set_yticks(range(len(tg)))
        if k == 0:
            ax.set_yticklabels([ps.region_label(t, alr.get(t)) for t in tg])
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.0, 1.0, lbl, transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"])
        ax.set_xlabel("Target labels n")
        axes.append(ax)
    cax = ps.axes_mm(fig, 30, 9.0, 60, 2.2)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm_)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("ΔRMSE vs physics (cm)", labelpad=1); cb.outline.set_linewidth(0.4)
    vm = norm.vmax
    stp = 5 if vm <= 20 else 10
    cb.set_ticks(list(np.arange(-vm, vm + 1e-9, stp))); cb.ax.xaxis.set_major_formatter(_minus_fmt())
    handles = [Patch(facecolor="none", edgecolor="#000000", lw=0.8, label="All three n* conditions met"),
               Patch(facecolor=ps.GREY["missing"], edgecolor=ps.GREY["edge"], hatch="///", lw=0, label="Not tested")]
    ps.legend_below(fig, handles, (100, 3, 80, 12), ncol=1)
    ps.label_panels(axes, "abc", dy_mm=4.0)
    f1_stat, f1_pan = _f1_sentence(Cv, q, alr)
    caption = dict(
        definition=("Confounder-free minimum label number (A2), all targets. Cells show ΔRMSE of residual ML (λ = 0.25, labels "
                    "spread over all blocks) minus the physics anchor, in cm, for three E treatments. n* is the smallest n at which "
                    "the block-bootstrap 95 % CI upper bound of ΔRMSE is below 0, at least two of three splits improve, and at least "
                    "75 % of repeats improve (pre-specified)."),
        statistics=("Colour scale is shared across panels, centred at 0 (broc). Outlined cells meet all three conditions; n* is the "
                    "leftmost outlined cell per row. " + f1_stat),
        panels="a to c, E0 fixed, E shrunk toward E0 (κ = 10) and log E offset (MLE). Rows ordered by |log(E_own/E0)|. " + f1_pan,
        data=f"Source data: {Cv.attrs.get('source')} with block CIs from {Cv.attrs.get('blk_source', 'h4/a2_blk.csv')}. Supplementary Table 5.",
    )
    spec = dict(intent="A2 전 대상 행렬(E 처리 3종, broc 0 중심 공통 범위, 세 조건 충족 칸 테두리)", panels=3,
                assets=[Cv.attrs.get("source")], script="scripts/4_visualization/paper_figs.py --only supp")
    res = save_paper(fig, "FigS12_a2_matrix", caption=caption, spec=spec, draft=draft or smoke, sources={"abc": pd.DataFrame(rows)})
    return [res, table_ST5(draft)]


def table_ST5(draft: bool = False):
    M = load_a2_minn(draft)
    smoke = bool(M.attrs.get("is_smoke"))
    M = M[~M.target.astype(str).str.startswith("Alaska_f") & (M.target != "CA-1")].copy()
    if "analysis" in M:
        M = M[M.analysis.astype(str).str.lower().isin(["true", "1"])]
    M["n_star_text"] = [fmt_censored(a, b, c) for a, b, c in zip(M.n_star, M.censored, M.n_max)]
    alr = abslogE_map()
    M["abs_logE"] = M.target.map(alr)
    M = M.sort_values(["e_treat", "spread", "stage", "lam", "abs_logE"])
    show = M[(M.stage == "resid") & np.isclose(M.lam, 0.25)]
    piv = show.pivot_table(index="target", columns=["e_treat", "spread"], values="n_star_text", aggfunc="first")
    colmap = [(("E0_fixed", "all_blocks"), "E0 fixed"), (("shrink_k10", "all_blocks"), "E shrunk (κ = 10)"),
              (("offset_mle", "all_blocks"), "log E offset"), (("E_own_fixed", "all_blocks"), "E_own fixed (oracle)"),
              (("E0_fixed", "concentrated"), "E0 fixed, concentrated"), (("E0_fixed", "spread5"), "E0 fixed, 5 blocks")]
    piv = pd.DataFrame({lbl: piv[c] for c, lbl in colmap if c in piv.columns}, index=piv.index)
    piv = piv.replace(regex={r" \(max tested\)": "", r"\(max tested, range-limited\)": "(range-limited)"})
    piv = piv.reindex(sorted(piv.index, key=lambda t: alr.get(t, np.inf))).reset_index()
    piv.insert(1, "|log E|", piv.target.map(lambda t: f"{alr.get(t, np.nan):.3f}"))
    piv["target"] = piv.target.map(ps.region_name)
    piv = piv.rename(columns={"target": "Target"})
    cap = dict(
        definition=("Supplementary Table 5. Minimum label number n* (A2) for residual ML (λ = 0.25) by E treatment and label "
                    "placement (all blocks, concentrated, spread over five blocks). n* uses the pre-specified three-condition rule of Fig. 4."),
        statistics=("Not-reached targets are reported as > n_max, the largest n tested; range-limited marks n_max < 40. No "
                    "values are imputed."),
        panels=("Printed columns: labels over all blocks for four E treatments (E_own fixed uses the oracle in-region E), and "
                "E0 fixed with concentrated or five-block placement. Rows ordered by |log(E_own/E0)|; other combinations, stages and "
                "λ = 0.5 are in the CSV file."),
        data=f"Source: {M.attrs.get('source')}.",
    )
    name = "TableS5_a2_nstar" + ("_draft" if (draft or smoke) else "")
    return write_table(name, M, piv, cap, col_fmt="lr" + "r" * (piv.shape[1] - 2))


# ================================================================ S2 채점 규약 도식
def _canvas(fig, x, y, w, h):
    """mm 좌표 그대로 그리는 빈 축(도식용)."""
    ax = ps.axes_mm(fig, x, y, w, h)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_axis_off()
    return ax


def _box(ax, x, y, w, h, fc="white", ec="#000000", lw=0.6, hatch=None, text=None, fs=None, tc="#000000", z=2):
    ax.add_patch(matplotlib.patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, hatch=hatch, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs or ps.FS["annot"], color=tc, zorder=z + 1)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#4d4d4d", mutation_scale=6,
                                                                  shrinkA=0, shrinkB=0), zorder=4)


def build_S2(draft: bool = False):
    ps.use_paper()
    fig = ps.paper_figure(180, 74)
    A_FC = "#dfe7f0"                                              # A 블록(라벨 풀) 채움: refit 파랑의 옅은 톤 + 해칭
    # ---------------------------------------------------------------- a A/B 블록 분할
    ax = _canvas(fig, 0, 6, 56, 62)
    ax.text(0, 60, "Target region, 0.5° blocks", ha="left", va="top", fontsize=ps.FS["annot"])
    rng = np.random.default_rng(3)
    layout = np.array([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1]])
    for i in range(4):
        for j in range(5):
            x0, y0 = 2 + j * 10, 47 - i * 10
            isA = layout[i, j] == 1
            _box(ax, x0, y0, 9, 9, fc=A_FC if isA else "white", hatch="////" if isA else None, ec="#4d4d4d", lw=0.5)
            k = rng.integers(3, 8)
            px, py = x0 + 1 + 7 * rng.random(k), y0 + 1 + 7 * rng.random(k)
            ax.plot(px, py, ls="none", marker="o", ms=1.4, color="#000000" if not isA else ps.COLOR["refit"], zorder=3)
    for j, (lbl, fc, h) in enumerate((("A blocks: label pool (n labels drawn)", A_FC, "////"), ("B blocks: held-out scoring", "white", None))):
        _box(ax, 2, 7.5 - j * 5.2, 4, 3.2, fc=fc, hatch=h, ec="#4d4d4d", lw=0.5)
        ax.text(7.5, 9.1 - j * 5.2, lbl, ha="left", va="center", fontsize=ps.FS["annot"])
    # ---------------------------------------------------------------- b 블록 부트스트랩(분할 안, 방법 공통 인덱스)
    bx = _canvas(fig, 62, 6, 64, 62)
    bx.text(0, 60, "Within one split", ha="left", va="top", fontsize=ps.FS["annot"])
    for j in range(6):
        _box(bx, 2 + j * 7.5, 46, 6.5, 6.5, text=f"B{j + 1}", ec="#4d4d4d", lw=0.5)
    draw = [3, 1, 1, 5, 6, 2]
    bx.text(26.5, 41.8, "resample with replacement", ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    for j, d in enumerate(draw):
        _box(bx, 2 + j * 7.5, 31, 6.5, 6.5, text=f"B{d}", ec="#4d4d4d", lw=0.5, fc="#f2f2f2")
    _arrow(bx, 24.5, 44.5, 24.5, 39.0)
    for r_, (key, lbl) in enumerate((("residual", "Method"), ("phys", "Physics"))):
        y0 = 20 - r_ * 8.5
        _box(bx, 2, y0, 44.5, 6.0, fc="white", ec=ps.COLOR[key], lw=0.9)
        bx.text(4, y0 + 3, lbl, ha="left", va="center", fontsize=ps.FS["annot"], color=ps.COLOR[key])
        bx.text(45, y0 + 3, "√(ΣSSE / Σcells)", ha="right", va="center", fontsize=ps.FS["annot"])
    _arrow(bx, 24.5, 30.0, 24.5, 26.8)
    bx.text(49, 16.5, "same\nindices", ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], linespacing=1.0)
    _box(bx, 2, 0.5, 44.5, 6.0, fc="#f2f2f2", ec="#4d4d4d", lw=0.5, text="Δ* = mean over repeats and seeds")
    _arrow(bx, 24.5, 11.0, 24.5, 7.2)
    bx.text(49, 3.5, "× 1,000", ha="left", va="center", fontsize=ps.FS["annot"])
    # ---------------------------------------------------------------- c 결합: 분할·지역
    cx = _canvas(fig, 130, 6, 50, 62)
    cx.text(0, 60, "Combination", ha="left", va="top", fontsize=ps.FS["annot"])
    for j in range(3):
        _box(cx, 1 + j * 16, 46, 14.5, 6.5, text=f"split {j + 1}", ec="#4d4d4d", lw=0.5)
        _arrow(cx, 8.25 + j * 16, 45.5, 24.5, 38.8)
    _box(cx, 1, 32, 47, 6.5, fc="#f2f2f2", ec="#4d4d4d", lw=0.5, text="average draw b across splits")
    cx.text(1, 27.5, "Region mean (AB4)", ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    for j, r_ in enumerate(("Lena", "Canada", "Russia W", "Russia E")):
        _box(cx, 1 + j * 12, 17.5, 11, 6.5, text=r_, ec="#4d4d4d", lw=0.5, fs=ps.FS["annot"])
    _arrow(cx, 24.5, 17.0, 24.5, 10.0)
    cx.text(26.0, 13.5, "resampled\nwithin region", ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], linespacing=1.0)
    _box(cx, 1, 3, 47, 6.5, fc="#f2f2f2", ec="#4d4d4d", lw=0.5, text="stratified bootstrap of mean")
    ps.label_panels([ax, bx, cx], "abc", dy_mm=-1.5)
    for a_ in (ax, bx, cx):
        a_._slot_mm = (a_._slot_mm[0], a_._slot_mm[1], a_._slot_mm[2], a_._slot_mm[3])
    handles = [Line2D([], [], ls="none", marker="o", ms=2.0, color=ps.COLOR["refit"], label="Label cell in A block"),
               Line2D([], [], ls="none", marker="o", ms=2.0, color="#000000", label="Scoring cell in B block")]
    ps.legend_below(fig, handles, (0, 0, 180, 6), ncol=2)
    caption = dict(
        definition=("Scoring protocol. Each target's 0.5° blocks are split into A blocks, the pool from which n labels are drawn, "
                    "and B blocks, on which every method and the physics anchor are scored. Three random A/B splits are used; "
                    "sources exclude the target and a 100 km buffer."),
        statistics=("Block bootstrap: within a split, B blocks are resampled with replacement 1,000 times; the same indices are "
                    "applied to both methods and to all repeats and seeds; RMSE is √(ΣSSE/Σcells) per draw. Draw b is averaged "
                    "across splits to give the CI of ΔRMSE. Regional means use a stratified block bootstrap."),
        panels=("a, A/B blocks. b, Paired block resampling in one split. c, Combination over splits and over the four main "
                "regions (AB4). Cell-weighted scoring pools cells; block-equal scoring weights each block equally "
                "(Supplementary Table 4b)."),
        data="Implementation: src/polar/h4_common.py (boot_delta_blocks), src/polar/m1_stats.py (strat).",
    )
    spec = dict(intent="채점 규약 도식: A/B 블록, 분할 안 짝지은 블록 부트스트랩, 분할·지역 결합", panels=3, assets=[],
                script="scripts/4_visualization/paper_figs.py --only supp")
    return save_paper(fig, "FigS2_scoring_protocol", caption=caption, spec=spec, draft=False)


class NotPerformed(Exception):
    """계획했으나 수행하지 않기로 확정한 항목(종결 상태). '대기(MissingData)'와 구분한다."""


S10_NOTE = ("Supplementary Fig. 10 slot (C4, not performed). The planned weak-label multi-fidelity analysis, which would use "
            "InSAR seasonal subsidence as low-fidelity ALT labels, was not performed because of time constraints; no figure, "
            "table or result is reported. It is treated as a limitation on literature grounds: InSAR-derived ALT proxies depend on "
            "ground ice and soil moisture and have been unstable outside Alaska (negative correlation on the Qinghai-Tibet Plateau, "
            "Chang 2024; R² 0.03 in Svalbard, Wendt 2026; see also Sadeghi Chorsi 2024), and no validated InSAR-ALT grid exists "
            "outside Alaska. The slot is kept so that Supplementary items S11 and S12 retain their original numbers (no renumbering).")


def build_S10(draft: bool = False):
    """S10 약라벨 다중 충실도(C4): 시간 제약으로 수행하지 않음(확정). 색인에 종결 상태로 기록하고 번호 자리 설명 절을 쓴다."""
    p = PROC / "h4" / "c4_summary.csv"
    if not p.exists():
        write_caption("FigS10_c4_not_performed", S10_NOTE)
        raise NotPerformed("C4 InSAR multi-fidelity: not performed (time constraint); limitation described from the literature "
                           "(Chang 2024; Wendt 2026)")
    raise NotImplementedError("C4 산출 형식 확정 후 작성")


# ================================================================ 값 대조(스펙 §5.1 6단계)
def value_checks() -> list:
    """그림 Source Data 에 찍힌 값을 원천 CSV 에서 다시 읽어 대조. 결과를 _qa/supp_values.txt 에 쓴다."""
    SD = OUT / "source_data"
    out = []

    def chk(name, a, b, tol=1e-9):
        ok = (np.isnan(a) and np.isnan(b)) or abs(float(a) - float(b)) <= tol
        out.append(f"{'OK ' if ok else 'BAD'} {name}: figure {a!r} vs source {b!r}")
    try:
        F = pd.read_csv(SD / "FigS4_k-l.csv"); C = load_h25_curve()
        for t, m, st, lam, n in (("Russia_W", "residual", "S3", 0.25, 3), ("AL-3", "refit", "S1", 0.0, 10), ("Canada", "residual", "S3", 0.25, 10)):
            a = F[(F.target == t) & (F.method == m) & (F.n == n)].d_phys.iloc[0]
            q = h25_series(C, t, st, "kmedoid", lam, 1.0); b = q[q.n == n].blk_d_phys.iloc[0]
            chk(f"S4 k/l {t} {m} n={n}", a, b)
    except Exception as e:                                          # noqa: BLE001
        out.append(f"ERR S4: {e}")
    try:
        F = pd.read_csv(SD / "FigS6_a.csv"); _, T, tag = load_h27("c")
        T = T[(T.rule == "kmedoid") & (T.stage == "S3") & np.isclose(T.lam, 0.25) & (T.definition == "rec50")]
        for t in ("AL-1", "CA-2", "AL-3", "Russia_W"):
            chk(f"S6a {t} be_n", F[F.target == t].be_n.iloc[0], T[T.target == t].be_n.iloc[0])
    except Exception as e:                                          # noqa: BLE001
        out.append(f"ERR S6: {e}")
    try:
        F = pd.read_csv(SD / "FigS7_ab.csv"); B = load_b4("summary")
        for t, n in (("AL-2", 3), ("Lena", 10), ("CA-2", 40)):
            a = F[(F.panel == "a") & (F.target == t) & (F.n == n)].d_tfm_minus_cb.iloc[0]
            b = B[(B.target == t) & (B.method == "tfm") & (B.n == n) & np.isclose(B.lam, 0.25)].d_cb_blk.iloc[0]
            chk(f"S7a {t} n={n}", a, b)
    except Exception as e:                                          # noqa: BLE001
        out.append(f"ERR S7: {e}")
    try:
        F = pd.read_csv(SD / "FigS5_a-j.csv"); H = rd(H2 / "h24_curve.csv")
        for t, m, st, n in (("Lena", "shrink_k10", "kmedoid", 3), ("Alaska", "resid", "random", 200), ("Russia_W", "shrink_k10", "geo_strat", 10)):
            a = F[(F.target == t) & (F.method == m) & (F.strategy == st) & (F.n == n)].d_phys.iloc[0]
            b = H[(H.target == t) & (H.method == m) & (H.strategy == st) & (H.n == n)].d_phys_mean.iloc[0]
            chk(f"S5 {t} {m} {st} n={n}", a, b)
    except Exception as e:                                          # noqa: BLE001
        out.append(f"ERR S5: {e}")
    try:
        F = pd.read_csv(SD / "FigS3_abc.csv"); S = load_subregions().set_index("subregion")
        for t in ("AL-1", "CA-3", "LE-2"):
            chk(f"S3 {t} n_cells", F[F.subregion == t].n_cells.iloc[0], S.loc[t, "n_cells"])
    except Exception as e:                                          # noqa: BLE001
        out.append(f"ERR S3: {e}")
    try:
        F = pd.read_csv(SD / "FigS12_abc.csv"); A = rd(H4 / "a2_curve.csv")
        for t, et, n in (("AL-3", "E0_fixed", 10), ("Russia_W", "shrink_k10", 3), ("Canada", "offset_mle", 40)):
            a = F[(F.target == t) & (F.e_treat == et) & (F.n == n)].blk_d_phys.iloc[0]
            b = A[(A.target == t) & (A.e_treat == et) & (A.n == n) & (A.spread == "all_blocks") & (A.stage == "resid") & np.isclose(A.lam, 0.25)].d_phys_mean.iloc[0]
            chk(f"S12 {t} {et} n={n}", a, b)
    except Exception as e:                                          # noqa: BLE001
        out.append(f"ERR S12: {e}")
    (QA_DIR / "supp_values.txt").write_text("\n".join(out) + "\n")
    print(f"[supp] value checks: {sum(l.startswith('OK') for l in out)} OK, {sum(l.startswith(('BAD', 'ERR')) for l in out)} not OK")
    return out


# ================================================================ 진입점
ITEMS = []          # (항목, 함수, 종류). 아래에서 채운다


def build(draft: bool = False):
    import os
    STATUS.clear()
    out = []
    only = [v.strip() for v in os.environ.get("SUPP_ONLY", "").split(",") if v.strip()]   # 개발용 부분 실행(S3,ST6 등)
    for item, fn, kind in ITEMS:
        if only and item not in only:
            continue
        try:
            res = fn(draft)
            res = res if isinstance(res, list) else [res]
            for r in res:
                if isinstance(r, dict):
                    out.append(r)
            ok = all(r.get("ok", True) for r in res if isinstance(r, dict))
            files = [r.get("name") for r in res if isinstance(r, dict)]
            _record(item, ("done" if ok else "qa_fail") + ("_draft" if draft and any(str(f).endswith("_draft") for f in files) else ""),
                    files, "; ".join(sum([r.get("fails") or [] for r in res if isinstance(r, dict)], []))[:300])
        except NotPerformed as e:
            _record(item, "not_run", note=str(e))
            print(f"[supp] {item}: 수행하지 않음({e})")
        except MissingData as e:
            _record(item, "pending", note=f"missing data: {e}")
            print(f"[supp] {item}: 대기(자료 없음: {e})")
        except NotImplementedError as e:
            _record(item, "pending", note=str(e))
            print(f"[supp] {item}: 대기({e})")
    _write_index(draft)
    if not only:
        value_checks()
    return out


def _write_index(draft: bool):
    tag = "_draft" if draft else ""
    (QA_DIR / f"supp_status{tag}.json").write_text(json.dumps(STATUS, ensure_ascii=False, indent=1))
    lines = []
    def _k(e):
        m = re.match(r"(S|ST)(\d+)", e["item"])
        return (0 if m.group(1) == "S" else 1, int(m.group(2))) if m else (2, 0)
    for s in sorted(STATUS, key=_k):
        st = {"pending": "pending (대기)", "not_run": "not performed"}.get(s["status"], s["status"])
        f = ", ".join(s["files"]) if s["files"] else "none"
        if s["item"] == "S10" and s["status"] == "not_run":
            f = "none (note in FigS10_c4_not_performed)"
        if s["item"] == "S11":
            f += " (no figure; software and run metadata only)"
        note = re.sub(r"\s*\([^)]*[가-힣][^)]*\)", "", s["note"] or "")          # 색인에는 영문 사유만
        lines.append(f"- {s['item']}: {st}; files: {f}" + (f"; note: {note}" if note and s["status"] in ("pending", "not_run") else ""))
    pre = ("Status of Supplementary Figures S1–S12 and Tables ST1–ST10. Numbering is fixed and was not renumbered: S10 (C4, "
           "InSAR weak-label multi-fidelity) was not performed because of time constraints and keeps its slot as a limitation note, "
           "so S11 and S12 keep their original numbers. S11 has no figure; it is the software and run metadata in Supplementary "
           "Tables 10a and 10b. S12 is paired with Supplementary Table 5.")
    write_caption(f"Supplementary_index{tag}", pre + "\n\n" + "\n".join(lines))
    _order_supp_sections()


def _order_supp_sections():
    """CAPTIONS.md 에서 부록 절(FigS·TableS·Supplementary_index)만 본문 절 뒤로 모아 번호 순으로 정렬한다(본문 절 순서·내용 불변)."""
    from _common import CAPTIONS, _locked
    with _locked(CAPTIONS):
        txt = CAPTIONS.read_text()
        m = re.search(r"^## ", txt, re.M)
        if not m:
            return
        head, body = txt[:m.start()], txt[m.start():]
        secs = re.findall(r"^## .*?(?=^## |\Z)", body, re.S | re.M)
        is_s = lambda t: re.match(r"## (FigS|TableS|Supplementary_index)", t) is not None      # noqa: E731

        def key(t):
            nm = t.split("\n", 1)[0][3:]
            mm = re.match(r"(FigS|TableS)(\d+)([a-z]?)", nm)
            if nm.startswith("Supplementary_index"):
                return (0, 0, "", nm)
            return (1 if mm.group(1) == "FigS" else 2, int(mm.group(2)), mm.group(3), nm) if mm else (3, 0, "", nm)
        main = [t for t in secs if not is_s(t)]
        sup = sorted([t for t in secs if is_s(t)], key=key)
        CAPTIONS.write_text(head + "".join(x.rstrip("\n") + "\n\n" for x in main + sup).rstrip("\n") + "\n")


ITEMS += [("S1", build_S1, "fig"), ("S2", build_S2, "fig"), ("S3", build_S3, "fig"), ("S4", build_S4, "fig"), ("S5", build_S5, "fig"), ("S6", build_S6, "fig"), ("ST3", table_ST3, "table"), ("ST6", table_ST6, "table"),
          ("ST1", table_ST1, "table"), ("ST2", table_ST2, "table"), ("ST4", table_ST4, "table"), ("ST7", table_ST7, "table"),
          ("S11", table_S11, "table"), ("S7", build_S7, "fig"), ("S8", build_S8, "fig"), ("S9", build_S9, "fig"),
          ("S12+ST5", build_S12, "fig"), ("S10", build_S10, "fig")]
