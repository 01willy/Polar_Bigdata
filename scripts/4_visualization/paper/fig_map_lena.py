"""지도 과제(MAP) 그림: 레나 x 1 km 격자의 ALT 지도, 90 % 구간 폭, 외삽 표시, 차이. 계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.6, 6.8, 6.10.

입력은 scripts/2_evaluation/h49_transfer_map.py 의 산출(lena_pred_v1.csv.gz, lena_pred_v1_meta.json)이다. 예측은 LG·LGU 판정 뒤에만 만들어지므로
그 전에는 합성 입력으로만 시험한다(--synthetic-test: h49 의 합성 산출, --demo-grid: 실제 격자의 좌표와 마스크에 합성 값을 얹은 배치 시험).
합성 시험의 그림에는 'SYNTHETIC TEST' 표지를 넣고 출력 폴더를 따로 둔다.

그림 규칙(6.8)
  투영      등적 방위 투영(LAEA, 중심 72.5°N, 126.7°E). 경위도 눈금(위도 1°, 경도 2°), 축척 막대(50 km), 단위 cm
  패널      (a) 라벨 0, (b) 라벨 10개, (c) 라벨 전량의 ALT(cm). 세 패널은 같은 색 범위(표시 셀 값의 2–98 백분위를 10 cm 단위로 넓힌 범위)
            (d) 패널 (a) 방법의 90 % 구간 폭(cm), (e) 원천 범위 밖 공변량 수(0, 1, 2, 3 이상, 결측 열 포함), (f) (c) − (a)(cm, 0 중심)
  색        ALT = cmc.oslo_r(지각 균등 순차형), 구간 폭 = cmc.acton(별도 순차형), 차이 = cmc.broc(0 중심 발산형, TwoSlopeNorm),
            외삽 = cmc.devon_r 에서 뽑은 순서형 4색(davos_r 의 중간 회녹색이 마스크 회색과 가까워 바꿨다). 무지개·jet 를 쓰지 않는다. 회색 = 마스크(6.5: 영구동토, 수체, 바다·입력 결측)
  배치      LGU-B2 가 '전이'이면 본문 Fig 6c 용 3패널과 SI 6패널을 모두 만든다. 본문 3패널은 SI 의 (a), (d), (e)를 a, b, c 로 다시 붙이고
            캡션에 대응(본문 b = SI d, 본문 c = SI e)을 적는다. 그 밖은 SI 6패널 한 장이고 캡션에 '구간 폭은 예측값에 비례한다'를 쓴다
            (h49 메타의 selection.placement). QA 는 그림의 패널 문자와 캡션 패널 문장의 문자가 같은지도 본다(panel_letters_ok)
  내보내기  PDF, SVG, 600 dpi PNG. 글꼴은 polar.paperstyle(Liberation Sans, Type 42). QA 는 paper/_common.qa_check(폭 180 mm, 최소 6.5 pt,
            글자 겹침, 잘림, 축 제목 금지, 범례 1개 이하, PDF 글꼴, 유니코드 마이너스, 캡션 규칙)
  캡션      영문(논문 규칙). 6.8 의 필수 항목(정적 지도, 라벨 지지와 1 km 격자의 차이, 영구동토 마스크 정의, 외삽 표시는 오차 위험 지표가
            아니라는 점, 방법과 선택 규칙)을 h49 메타의 caption 사실에서 만든다. 6.5 의 회색 육지 셀 비율과 첫 사유별 수(caption.gray_land,
            없으면 격자 메타 masks.summary), 6.6 (b)의 분할 간 영역 평균 ALT 범위, 6.7 의 GPU 학습기 사실(LGT 는 '등록 시점에 결과 존재,
            미열람' 표지)과 LGT·LGF 표가 빠졌다는 사실도 적는다

출력 보호: --synthetic-test·--demo-grid 는 data/processed/ 의 실험 산출 폴더, results/, data/processed/map_lena, outputs/maps/transfer_lena
  안에 쓰지 않는다. 스크립트로 실행하면 OMP·OpenBLAS·MKL·NUMEXPR 스레드를 1 로 둔다.

실행(ROOT, CPU 1스레드, nice 10)
  합성 시험:   nice -n 10 python3 scripts/4_visualization/paper/fig_map_lena.py --synthetic-test <폴더>
  배치 시험:   nice -n 10 python3 scripts/4_visualization/paper/fig_map_lena.py --demo-grid <폴더>
  본 그림:     nice -n 10 python3 scripts/4_visualization/paper/fig_map_lena.py            (h49 본 실행 뒤)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = "1"                                       # 스크립트 실행: 1 스레드로 명시한다
    else:
        os.environ.setdefault(_v, "1")

import argparse                                                                                         # noqa: E402
import importlib.util                                                                                   # noqa: E402
import json                                                                                             # noqa: E402
import math                                                                                             # noqa: E402
import re                                                                                               # noqa: E402
import sys                                                                                              # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from polar import paperstyle as ps                                                                      # noqa: E402

import matplotlib                                                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize                     # noqa: E402

CENTER = dict(lat=72.5, lon=126.7)
BOX = dict(lat0=71.5, lat1=73.6, lon0=123.3, lon1=130.1)
CELL_DEG = 0.009
RASTER_M = 500.0                                                  # 표시 래스터 화소(m). 셀(약 1 km)보다 작게 둔다
SCALE_KM = 50
MASK_RGBA = (0.616, 0.631, 0.651, 1.0)                           # #9da1a6 마스크 회색(oslo_r 의 밝은 회청색 끝과 명도로 구별한다)
OUT_DEFAULT = ROOT / "outputs" / "maps" / "transfer_lena" / "fig_map_lena"
PROTECT = ("lg", "lgx", "lgt", "lgd", "lgu", "lgf", "ext_labels")
LAYOUT_PANELS = dict(all6=("a", "b", "c", "d", "e", "f"), main3=("a", "d", "e"))   # 그리는 SI 패널
LAYOUT_LETTERS = dict(all6="abcdef", main3="abc")                                  # 그림에 붙이는 문자(본문 3패널은 a, b, c)
MASK_LEGEND = "Masked (PFR < 10% or missing, water, sea, missing input)"
REASON_EN = dict(dem_tile_absent="no DEM tile", water="water", pfr_missing="PFR missing", pfr_lt10="PFR below 10%", cci_missing="CCI ALT missing",
                 soil_tdd_missing="soil thaw index missing", s_missing="thaw index missing")
PANEL_TXT = {"a": "No labels", "b": "10 labels", "c": "All labels", "d": "90% interval width, panel a",
             "e": "Covariates outside source range", "f": "Difference c − a"}
METHOD_EN = {"P0": "P0 (source Stefan coefficient)", "P1": "P1 (shrunk coefficient)", "R0": "R0 (P0 anchor + residual CatBoost)",
             "R1": "R1 (shrunk anchor + residual CatBoost)"}


def load_module(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def method_en(m: str) -> str:
    if m in METHOD_EN:
        return METHOD_EN[m]
    if m.startswith("P0@") or m.startswith("P1@"):
        return f"{m} (anchor {m.split('@')[1]})"
    return m


# ================================================================ 래스터화
def laea():
    import cartopy.crs as ccrs
    return ccrs.LambertAzimuthalEqualArea(central_longitude=CENTER["lon"], central_latitude=CENTER["lat"])


def extent_xy(proj):
    import cartopy.crs as ccrs
    lo = np.r_[np.linspace(BOX["lon0"], BOX["lon1"], 60), np.full(30, BOX["lon1"]), np.linspace(BOX["lon1"], BOX["lon0"], 60), np.full(30, BOX["lon0"])]
    la = np.r_[np.full(60, BOX["lat0"]), np.linspace(BOX["lat0"], BOX["lat1"], 30), np.full(60, BOX["lat1"]), np.linspace(BOX["lat1"], BOX["lat0"], 30)]
    p = proj.transform_points(ccrs.PlateCarree(), lo, la)
    pad = 2000.0
    return float(p[:, 0].min() - pad), float(p[:, 0].max() + pad), float(p[:, 1].min() - pad), float(p[:, 1].max() + pad)


class Raster:
    """LAEA 정칙 래스터의 화소 → 1 km 셀(ky, kx) 대응. 셀 색인은 build_ext_cells_v1.cell_index 와 같은 식이다."""

    def __init__(self, cells: pd.DataFrame, proj, res=RASTER_M):
        import cartopy.crs as ccrs
        self.ext = extent_xy(proj)
        x0, x1, y0, y1 = self.ext
        self.nx, self.ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        xs = x0 + (np.arange(self.nx) + 0.5) * res
        ys = y0 + (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        ll = ccrs.PlateCarree().transform_points(proj, X.ravel(), Y.ravel())
        lon, lat = ll[:, 0], ll[:, 1]
        ky = np.floor(lat / CELL_DEG).astype(np.int64)
        kx = np.floor(lon * np.cos(np.radians((ky + 0.5) * CELL_DEG)) / CELL_DEG).astype(np.int64)
        key = ky * 1_000_000 + kx
        ckey = cells.ky.values.astype(np.int64) * 1_000_000 + cells.kx.values.astype(np.int64)
        idx = pd.Index(ckey).get_indexer(key)
        self.idx = idx.reshape(self.ny, self.nx)                  # −1 = 격자 밖

    def values(self, v):
        v = np.asarray(v, float)
        out = np.full(self.idx.shape, np.nan)
        m = self.idx >= 0
        out[m] = v[self.idx[m]]
        return out

    def mask(self, gray):
        g = np.asarray(gray, bool)
        out = np.zeros(self.idx.shape, bool)
        m = self.idx >= 0
        out[m] = g[self.idx[m]]
        return out


# ================================================================ 축과 장식
def map_axes(fig, rect, proj, ext, left_labels, bottom_labels):
    import cartopy.crs as ccrs
    ax = ps.axes_mm(fig, *rect, projection=proj)
    ax.set_extent(ext, crs=proj)
    ax.spines["geo"].set_linewidth(0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, lw=0.35, color=ps.GREY["grid"], alpha=0.8, zorder=3,
                      xlocs=[124, 126, 128, 130], ylocs=[72, 73], x_inline=False, y_inline=False)
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = bool(left_labels); gl.bottom_labels = bool(bottom_labels)
    gl.xlabel_style = dict(size=ps.FS["tick"]); gl.ylabel_style = dict(size=ps.FS["tick"])
    gl.rotate_labels = False
    return ax


def draw_field(ax, R: Raster, arr, gray_mask, cmap, norm):
    x0, x1, y0, y1 = R.ext
    ax.imshow(np.ma.masked_invalid(np.where(gray_mask, np.nan, arr)), origin="lower", extent=(x0, x1, y0, y1), transform=ax.projection,
              cmap=cmap, norm=norm, interpolation="nearest", zorder=1, rasterized=True)
    rgba = np.zeros(gray_mask.shape + (4,))
    rgba[gray_mask] = MASK_RGBA
    ax.imshow(rgba, origin="lower", extent=(x0, x1, y0, y1), transform=ax.projection, interpolation="nearest", zorder=1.1, rasterized=True)


def scale_bar(ax, km=SCALE_KM):
    import matplotlib.patheffects as pe
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xs = x0 + 0.06 * (x1 - x0); ys = y0 + 0.06 * (y1 - y0)
    ax.plot([xs, xs + km * 1000], [ys, ys], color="#000000", lw=1.2, solid_capstyle="butt", transform=ax.transData, zorder=5,
            path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
    t = ax.text(xs + km * 500, ys + 0.02 * (y1 - y0), f"{km} km", ha="center", va="bottom", fontsize=ps.FS["annot"], transform=ax.transData, zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    t.set_gid("scalebar")


def trunc(cmap, a, b, n=256):
    """색표의 [a, b] 구간만 쓴다(양 끝의 흰색·검은색을 피한다)."""
    return ListedColormap(cmap(np.linspace(a, b, n)), name=f"{cmap.name}_{a:g}_{b:g}")


def cond_label(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top", fontsize=ps.FS["annot"], zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def hcbar(fig, rect, mappable, label, ticks=None, ticklabels=None, extend="neither"):
    cax = ps.axes_mm(fig, *rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, fontsize=ps.FS["label"], labelpad=1.5)
    cb.ax.tick_params(labelsize=ps.FS["tick"], length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:                                                          # 범위 밖 눈금 글자(보이지 않아도 잘림 검사에 걸린다)를 뺀다
        lo, hi = mappable.norm.vmin, mappable.norm.vmax
        eps = 1e-9 * max(abs(lo), abs(hi), 1.0)
        cb.set_ticks([t for t in cb.get_ticks() if lo - eps <= t <= hi + eps])
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    return cb


def _round_range(vals, step=10.0, lo_q=2, hi_q=98):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, step
    lo = step * math.floor(np.percentile(v, lo_q) / step)
    hi = step * math.ceil(np.percentile(v, hi_q) / step)
    return max(lo, 0.0), max(hi, lo + step)


# ================================================================ 그림
def build_figure(pred: pd.DataFrame, meta: dict, layout="all6", synthetic_label=""):
    from cmcrameri import cm as cmc
    ps.use_paper()
    proj = laea()
    R = Raster(pred, proj)
    gray = R.mask(pred.gray.values == 1) if "gray" in pred else np.zeros(R.idx.shape, bool)
    shown = (pred.gray.values == 0) if "gray" in pred else np.ones(len(pred), bool)
    ext = R.ext
    sel = meta.get("selection", {}).get("panels", {})
    alt_lo, alt_hi = _round_range(np.r_[pred.pred_a[shown], pred.pred_b[shown], pred.pred_c[shown]])
    alt_norm = Normalize(alt_lo, alt_hi)
    w_lo, w_hi = _round_range(pred.width90_a[shown], step=5.0, lo_q=2, hi_q=98)   # 폭의 공간 변화가 보이도록 2–98 백분위(09-30 시각 검토)
    w_norm = Normalize(w_lo, w_hi)
    d_norm = ps.shared_diverging_norm([pred.diff_c_minus_a[shown].values], step_cm=5.0)
    cat_cols = [cmc.devon_r(v) for v in (0.15, 0.4, 0.62, 0.88)]     # davos_r 의 중간 색(회녹색)은 마스크 회색과 가까워 devon_r 을 쓴다
    cat_cmap = ListedColormap(cat_cols)
    cat_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    alt_cmap, w_cmap = trunc(cmc.oslo_r, 0.12, 0.90), trunc(cmc.acton, 0.05, 0.95)      # design/brand_tokens.json 의 절단 범위
    fields = dict(a=(pred.pred_a, alt_cmap, alt_norm), b=(pred.pred_b, alt_cmap, alt_norm), c=(pred.pred_c, alt_cmap, alt_norm),
                  d=(pred.width90_a, w_cmap, w_norm), e=(pred.extrap_cat, cat_cmap, cat_norm), f=(pred.diff_c_minus_a, cmc.broc, d_norm))
    panels = list(LAYOUT_PANELS[layout])
    W = ps.W2_MM
    S = 50.0                                                       # 지도 한 변(mm)
    xcol = [11.0, 67.0, 123.0]
    if layout == "all6":
        H = 162.0
        pos = dict(a=(xcol[0], 100.0), b=(xcol[1], 100.0), c=(xcol[2], 100.0), d=(xcol[0], 20.0), e=(xcol[1], 20.0), f=(xcol[2], 20.0))
    else:
        H = 80.0
        pos = dict(a=(xcol[0], 20.0), d=(xcol[1], 20.0), e=(xcol[2], 20.0))
    fig = ps.paper_figure(W, H)
    axes = {}
    for p in panels:
        x, y = pos[p]
        left = x == xcol[0]
        bottom = True
        ax = map_axes(fig, (x, y, S, S), proj, ext, left, bottom)
        arr, cmap, norm = fields[p]
        draw_field(ax, R, R.values(arr), gray, cmap, norm)
        axes[p] = ax
    ps.label_panels([axes[p] for p in panels], letters=LAYOUT_LETTERS[layout], dx_mm=-9.0, dy_mm=1.0)
    for p in panels:
        txt = {"a": f"{PANEL_TXT['a']} · {sel.get('a', '')}", "b": f"{PANEL_TXT['b']} · {sel.get('b', '')}", "c": f"{PANEL_TXT['c']} · {sel.get('c', '')}",
               "d": f"90% interval · {sel.get('a', '')}", "e": "Outside source range", "f": "Difference c − a"}[p]
        cond_label(axes[p], txt)
    scale_bar(axes["a"])
    # 색막대
    sm = {k: plt.cm.ScalarMappable(norm=fields[k][2], cmap=fields[k][1]) for k in fields}
    cb_y = lambda p: pos[p][1] - 13.0                              # noqa: E731
    if layout == "all6":
        cx = (xcol[0] + xcol[2] + S) / 2.0                         # 세 지도의 가운데
        hcbar(fig, (cx - 50.0, cb_y("a"), 100.0, 2.2), sm["a"], "Active layer thickness (cm)", extend="both")
    else:
        hcbar(fig, (pos["a"][0] + 5.0, cb_y("a"), S - 10.0, 2.2), sm["a"], "ALT (cm)", extend="both")
    hcbar(fig, (pos["d"][0] + 5.0, cb_y("d"), S - 10.0, 2.2), sm["d"], "90% interval width (cm)", extend="both")
    hcbar(fig, (pos["e"][0] + 5.0, cb_y("e"), S - 10.0, 2.2), sm["e"], "Covariates outside range (count)", ticks=[0, 1, 2, 3],
          ticklabels=["0", "1", "2", "≥3"])
    if "f" in panels:
        hcbar(fig, (pos["f"][0] + 5.0, cb_y("f"), S - 10.0, 2.2), sm["f"], "Difference c − a (cm)", extend="both")
    # 마스크 범례(그림 전체에 하나)
    from matplotlib.patches import Patch
    fx, fy = ps.mm_to_fig(fig, W - 2.0, H - 1.0)
    fig.legend(handles=[Patch(facecolor=MASK_RGBA, edgecolor="none", label=MASK_LEGEND)], loc="upper right",
               bbox_to_anchor=(fx, fy), fontsize=ps.FS["legend"], frameon=False)
    if synthetic_label:
        fx, fy = ps.mm_to_fig(fig, 2.0, H - 1.5)                  # 왼쪽 위(오른쪽 위의 마스크 범례와 겹치지 않게)
        t = fig.text(fx, fy, synthetic_label, ha="left", va="top", fontsize=9, color="#b5651d", fontweight="bold")
        t.set_gid("synthetic_label")
    return fig, dict(alt_range=[alt_lo, alt_hi], width_range=[w_lo, w_hi], diff_vmax=float(d_norm.vmax), raster=[R.ny, R.nx])


# ================================================================ 캡션(영문)
def fmt(v, nd=1):
    return ps.fmt_num(float(v), nd) if v is not None and np.isfinite(float(v)) else "NA"


def build_caption(meta: dict, layout: str, grid_meta: dict | None) -> dict:
    """영문 캡션(블록 예산: definition 80, statistics 80, panels 150, data 40 단어). 6.8 의 필수 항목을 모두 넣는다."""
    cap = meta.get("caption", {})
    sel = meta.get("selection", {})
    pan = cap.get("panels", {})
    itv = cap.get("interval", {})
    pb = cap.get("panel_b", {})
    gr = cap.get("gray", {})
    top = ", ".join(t["column"] for t in cap.get("extrap_top3", []))
    rng = lambda k, nd=2: (f"{fmt(pb.get(k, [np.nan])[0], nd)}\u2013{fmt(pb.get(k, [np.nan, np.nan])[-1], nd)}")   # noqa: E731
    definition = ("Static 1 km maps of active layer thickness (ALT) for the Lena delta treated as a new region (source excludes Lena and a "
                  "100 km buffer), from ERA5-Land 2015\u20132020 climatology and the CCI ALT 1997\u20132021 mean. Labels are point or site-mean "
                  "observations; map values are 1 km cell predictions. Grey: CCI PFR 1997\u20132021 mean below 10% or missing, surface water "
                  "occurrence of 50% or more, sea, or missing input.")
    stats = ("Methods were fixed by pre-registered rules before mapping: a, R0 or the best grid-computable physics baseline if superior to P0 "
             "and not inferior in Lena, otherwise P0; b and c, R1 if superior to P1 (and to P1*) and not inferior in Lena, otherwise the "
             "recalibration reference. "
             f"90% interval: hierarchical conformal quantile of leave-one-region-out log errors (q = {fmt(itv.get('q90'), 2)}, "
             f"{len(itv.get('groups', []))} source regions)" + ("; width is proportional to the prediction." if itv.get("normalizer") == "const"
                                                                else "; nflow normaliser."))
    notes = []
    if sel.get("c", {}).get("caption_L8"):
        notes.append("L8 was rejected (R1 did not beat P0 with all labels).")
    gn = gpu_note(sel.get("gpu_notes", []))
    if gn:
        notes.append(gn)
    miss = cap.get("gpu_missing") or sel.get("gpu_missing") or []
    if miss:
        notes.append("LGT/LGF learner comparisons not included.")
    for r in cap.get("replaced", []):
        notes.append(f"{r.get('h42_pick')} is undefined on a static grid; {r.get('pick') or r.get('base')} was used.")
    ext = (f"covariates outside the source 0.5\u201399.5 percentile range, missing included (most often {top}); not an error-risk indicator.")
    alt_all = f" ({rng('mean_alt_all', 0)} cm)" if pb.get("mean_alt_all") else ""          # 6.6 (b): 분할 간 영역 평균 ALT 범위
    if layout == "main3":
        panels = (f"a, {method_en(pan.get('a', {}).get('method', ''))}. b, 90% interval width of a. c, {ext} "
                  "b and c are Supplementary panels d and e. " + " ".join(notes))
    else:
        panels = (f"a, {method_en(pan.get('a', {}).get('method', ''))}. b, {method_en(pan.get('b', {}).get('method', ''))}, one draw of 10 labels "
                  f"(split {pb.get('split', 'NA')}); recalibrated coefficient {rng('E_n_same_split')} over 5 draws (area-mean ALT "
                  f"{rng('mean_alt_same_split', 0)} cm), {rng('E_n_all')} over {pb.get('n_splits', 'NA')} splits{alt_all}. c, "
                  f"{method_en(pan.get('c', {}).get('method', ''))}, all Lena labels. d, 90% interval width of a. e, {ext} f, c minus a. "
                  + " ".join(notes))
    ng, nc = gr.get("n_gray", 0), gr.get("n_cells", 0)
    data = f"{nc - ng:,} of {nc:,} cells shown. {land_sentence(cap, grid_meta)}LAEA projection centred at 72.5\u00b0N, 126.7\u00b0E."
    return dict(definition=definition, statistics=stats, panels=panels.strip(), data=data)


def gpu_note(notes: list) -> str:
    """GPU 학습기 사실(6.7 공통) 한 문장. 학습기마다 가장 낮은 \u0394 의 행을 쓴다. LGT 행은 '등록 시점에 결과 존재, 미열람' 표지를 붙인다."""
    if not notes:
        return ""
    best = {}
    for g in notes:
        k = (g.get("learner_en") or g.get("learner"), g.get("experiment", ""))
        d = g.get("delta")
        if k not in best or (d is not None and np.isfinite(float(d)) and float(d) < float(best[k].get("delta") or np.inf)):
            best[k] = g
    parts = []
    for (name, exp), g in best.items():
        mark = "; result existed at registration, unviewed" if g.get("mark") else ""
        parts.append(f"{name} ({exp}, n = {'all' if int(g['n']) == -1 else g['n']}, \u0394 {fmt(g.get('delta'), 2)} cm{mark})")
    return "Ahead of CatBoost but not mapped: " + ", ".join(parts) + "."


def land_sentence(cap: dict, grid_meta: dict | None) -> str:
    """6.5: 회색 육지 셀의 비율과 첫 사유별 수. h49 메타의 caption.gray_land, 없으면 격자 메타의 masks.summary."""
    gl = cap.get("gray_land") or {}
    if not gl and grid_meta:
        ms = grid_meta.get("masks", {}).get("summary", {})
        if "n_land" in ms:
            gl = dict(n_land=ms["n_land"], n_land_gray=ms["n_land_gray"], frac_land_gray=ms["frac_land_gray"],
                      land_by_reason_first=ms.get("land_by_reason_first", {}))
    if not gl or "n_land" not in gl:
        return ""
    rs = ", ".join(f"{REASON_EN.get(k, k)} {int(v):,}" for k, v in gl.get("land_by_reason_first", {}).items() if int(v) > 0)
    pct = 100.0 * float(gl.get("frac_land_gray", 0.0))
    return (f"Of {int(gl['n_land']):,} land cells, {int(gl['n_land_gray']):,} ({pct:.1f}%) are grey"
            + (f" ({rs})" if rs else "") + ". ")


def caption_letters(caption: dict) -> str:
    """캡션 panels 블록에서 패널 문장을 여는 문자('a, ' 형식)를 순서대로 모은다."""
    return "".join(re.findall(r"(?:^|\. )([a-h]), ", caption.get("panels", "")))


def panel_letters_check(fig, caption: dict, layout: str) -> dict:
    """그림의 패널 문자(gid panel_label)와 캡션 패널 문장의 문자가 같은지(6.8 시각 QA 보강)."""
    got = "".join(t.get_text() for t in fig.texts if t.get_gid() == "panel_label")
    cap = caption_letters(caption)
    return dict(panel_letters_fig=got, panel_letters_caption=cap, panel_letters_ok=bool(got == LAYOUT_LETTERS[layout] == cap))


# ================================================================ 합성 입력
def demo_pred_on_grid(grid_csv: Path, seed=0) -> tuple[pd.DataFrame, dict]:
    """실제 격자의 좌표와 마스크 열만 쓰고 값은 합성한다(배치 시험. 공변량과 모형을 쓰지 않는다)."""
    g = pd.read_csv(grid_csv, usecols=["cell_id", "ky", "kx", "lat", "lon", "gray", "gray_reason", "land"], dtype={"cell_id": str})
    rng = np.random.RandomState(seed)
    la, lo = g.lat.values, g.lon.values
    base = 45 + 25 * np.sin((lo - 123.3) / 6.8 * np.pi) * np.cos((la - 71.5) / 2.1 * np.pi) + rng.normal(0, 3, len(g))
    p = pd.DataFrame(dict(cell_id=g.cell_id, ky=g.ky, kx=g.kx, lat=la, lon=lo, gray=g.gray, gray_reason=g.gray_reason.fillna("")))
    p["pred_a"] = base; p["pred_b"] = base * 0.92 + 3; p["pred_c"] = base * 0.85 + 6 + 4 * np.sin(la * 7)
    p["width90_a"] = p.pred_a * (np.exp(0.6) - np.exp(-0.6))
    p["diff_c_minus_a"] = p.pred_c - p.pred_a
    p["extrap_cat"] = np.minimum(rng.poisson(0.4, len(g)), 3)
    meta = dict(synthetic=True, selection=dict(panels=dict(a="P0", b="P1", c="R1", d="P0"), placement="SI", c=dict(caption_L8=False), gpu_notes=[]),
                caption=dict(panels=dict(a=dict(method="P0"), b=dict(method="P1"), c=dict(method="R1")),
                             interval=dict(q90=0.6, groups=["A", "B", "C", "D"], normalizer="const"),
                             panel_b=dict(split=1, E_n_same_split=[3.0, 3.4], mean_alt_same_split=[80, 91], n_splits=5, E_n_all=[2.9, 3.6]),
                             gray=dict(n_gray=int(g.gray.sum()), n_cells=int(len(g))), extrap_top3=[dict(column="dem_slope"), dict(column="e5_swe"),
                                                                                                  dict(column="sg_soc_0_5")], replaced=[]))
    meta["caption"]["panel_b"]["mean_alt_all"] = [76, 95]
    return p, meta


# ================================================================ 저장과 QA
def save_and_qa(fig, stem: Path, caption: dict, qa_width=ps.W2_MM, layout="all6") -> dict:
    C = load_module("paper_common", "scripts/4_visualization/paper/_common.py")
    paths = ps.save_figure(fig, stem, formats=("pdf", "svg", "png"), dpi=600, expect_width_mm=qa_width)
    qa = C.qa_check(fig, paths["pdf"], caption, expect_width_mm=qa_width, ignore_overlap_gids=("inset",))
    qa.update(panel_letters_check(fig, caption, layout))
    if not qa["panel_letters_ok"]:
        qa["ok"] = False
        qa.setdefault("fails", []).append(f"panel letters: figure '{qa['panel_letters_fig']}' vs caption '{qa['panel_letters_caption']}'")
    txt, wc = C.caption_text(caption)
    cap_path = stem.with_name(stem.name + "_caption.md")
    cap_path.write_text(f"# {stem.name}\n\n{txt}\n\n<!-- words: {wc} -->\n")
    qa.update(paths=paths, caption=str(cap_path))
    stem.with_name(stem.name + "_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1, default=str))
    plt.close(fig)
    return qa


def render(pred, meta, out_stem: Path, grid_meta=None, synthetic_label=""):
    res = {}
    lay = ["all6"]
    if meta.get("selection", {}).get("d", {}).get("normalizer") == "nflow":
        lay.append("main3")                                        # LGU-B2 '전이': 본문 3패널도 만든다(6.8)
    for L in lay:
        fig, info = build_figure(pred, meta, L, synthetic_label)
        cap = build_caption(meta, L, grid_meta)
        stem = out_stem if L == "all6" else out_stem.with_name(out_stem.name + "_main3")
        qa = save_and_qa(fig, stem, cap, layout=L)
        qa.update(info)
        res[L] = qa
        print(f"[fig] {stem.name}: {qa['width_mm']}×{qa['height_mm']} mm, min {qa['min_font_pt']} pt, "
              f"{'ok' if qa['ok'] else 'FAIL ' + '; '.join(qa['fails'][:4])}", flush=True)
    return res


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="지도 과제 그림(레나 x)")
    ap.add_argument("--pred", default="data/processed/map_lena/lena_pred_v1.csv.gz")
    ap.add_argument("--pred-meta", default="data/processed/map_lena/lena_pred_v1_meta.json")
    ap.add_argument("--grid", default="data/processed/map_lena/lena_grid_x25_v1.csv.gz")
    ap.add_argument("--grid-meta", default="data/processed/map_lena/lena_grid_x25_v1_meta.json")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--synthetic-test", default="", help="h49 합성 산출로 그림을 만든다(출력 폴더)")
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--demo-grid", default="", help="실제 격자 좌표·마스크 + 합성 값(출력 폴더)")
    return ap.parse_args(argv)


def _abs(p):
    return Path(p) if os.path.isabs(str(p)) else ROOT / p


def main(argv=None):
    a = parse_args(argv)
    try:
        if os.nice(0) < 10:
            os.nice(10 - os.nice(0))
    except OSError:
        pass
    t0 = time.time()
    forbid = [ROOT / "data" / "processed" / d for d in PROTECT] + [ROOT / "results", ROOT / "data" / "processed" / "map_lena", OUT_DEFAULT.parent,
                                                                  _abs(a.out).parent]
    for arg in (a.synthetic_test, a.demo_grid):
        if arg:
            p = Path(os.path.realpath(str(_abs(arg))))
            for q in forbid:
                q = Path(os.path.realpath(str(q)))
                if p == q or q in p.parents:
                    raise SystemExit(f"[거부] 합성 시험 출력 폴더 {p} 는 보호 폴더 {q} 안이다")
    if a.synthetic_test:
        out = Path(a.synthetic_test); out.mkdir(parents=True, exist_ok=True)
        h49 = load_module("h49_transfer_map", "scripts/2_evaluation/h49_transfer_map.py")
        h49.main(["--synthetic-test", str(out), "--scenario", a.scenario, "--threads", "1", "--synthetic-grid-side", "0"])
        pred = pd.read_csv(out / "lena_pred_v1.csv.gz", dtype={"cell_id": str})
        meta = json.loads((out / "lena_pred_v1_meta.json").read_text())
        return render(pred, meta, out / "fig_map_lena_synthetic", synthetic_label="SYNTHETIC TEST (not results)")
    if a.demo_grid:
        out = Path(a.demo_grid); out.mkdir(parents=True, exist_ok=True)
        pred, meta = demo_pred_on_grid(_abs(a.grid))
        gm = json.loads(_abs(a.grid_meta).read_text()) if _abs(a.grid_meta).exists() else None
        return render(pred, meta, out / "fig_map_lena_demo_grid", gm, synthetic_label="SYNTHETIC VALUES ON REAL GRID (layout test)")
    pm = _abs(a.pred_meta)
    if not pm.exists():
        raise SystemExit("[거부] h49 본 실행 산출이 없다. 예측은 LG·LGU 판정 뒤에 만든다(6.11)")
    meta = json.loads(pm.read_text())
    if meta.get("synthetic"):
        raise SystemExit("[거부] 기본 경로의 예측 메타가 합성 시험 산출이다")
    pred = pd.read_csv(_abs(a.pred), dtype={"cell_id": str})
    gm = json.loads(_abs(a.grid_meta).read_text())
    res = render(pred, meta, _abs(a.out), gm)
    print(f"[done] {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
