"""G3 전체 연구 흐름 (덱 전용, 12 × 4.65 in = 3600 × 1395 px, 300 dpi, 영문 FreeSans).

v10 개정(2026-10-16, 사용자 지시): 그림 안 글자는 모두 영어(논문 어휘). 셰브런 사이에 보이는 틈(가로 0.12 in = 사선 수직 ≈ 0.10 in),
단계 종류별 옅은 채움 + 같은 색상의 조금 진한 테두리 + 잉크 글자(자료 = 옅은 청회색, 기후 입력 = 옅은 복숭아색, 물리 = #DCEBDD/#86AE8A,
잔차 학습 = #F7E9AE/#C9AE5B(M01 Stefan · Gradient boosting 상자와 같다), 검증 = 옅은 라벤더, 지도·제품 = 옅은 청색; 색상 원천은 STAGE 주석).
셰브런 높이 0.62 in(두 줄 이름). 색 막대 단위 글줄은 막대 위 왼쪽(열 폭 안). 저장 전 한글 글자를 찾으면 멈춘다.

v9 개정(2026-10-15, 설계 지침 v9 항 6·7·8; v8 명세를 대체)
- 셰브런: 흰 바탕 + 진한 외곽선(#4D4D4D) + 잉크 글자 14 pt(연녹색 채움·흐린 글자 폐기). 번호 없이 단계 이름만:
  셀·연도 자료 → 관측 연도 기후 입력 → 물리 기준(Stefan) → 잔차 학습 → 다섯 가지 검증 분할 → 지도·제품 비교. '강제력' 표현 없음.
- 열마다 두 층: 위 = 주 썸네일(범북극 지도 · 알래스카 TDD 2014 · 레나 ALT 대 √TDD 산점 · 보정량 2014 · 분할 다섯 줄 목록 · ALT 2014 지도),
  아래 = 보조 요소(연도별 셀-연도 막대 · 연도별 TDD 막대 · Stefan 식 블록 · 잔차 ML 트리 블록 · (목록 이어짐) · 같은 해 Wei 2026 지도).
  v8 의 분할별 RMSE 막대와 제품 RMSE 가로 막대(가로 혼잡)는 뺐다. 둘째 줄 채움 글줄('해마다 다른 …', '학습된 …')도 없다.
- 회전한 y축 제목 없이 단위 글줄을 차트 왼쪽 위에 둔다(열 사이 가로 혼잡 해소). 아래 글줄은 색 열쇠가 필요한 열(지역 · 분할)만.
- 축척 막대(200 km)는 지도 틀 안 왼쪽 아래(베링해), 글자는 막대 위 왼맞춤·흰 테. 제곱근은 mathtext.
- 자료: labels_cell_year_v5.csv, forcing_year.csv, maps/Alaska/{pred_2014, mean_2011_2024}.csv.gz, site_series_year_obs.csv.gz, maps/Lena/meta.json,
  Wei 2026 2014 음영 격자 재투영 캐시(outputs/figures/paper/v7/_cache/f8_alaska_wei_2014_s1.npz; v8_f8_product_maps.product_px 와 같은 값, 없으면 그 함수로 만든다).
- 색: 지역 S.REGION_HEX, 방법 S.METHOD_HEX, 제품 S.PRODUCT_HEX(토큰), 학습·시험 = 공정 블록 테두리색, 지도 MV.CM_ALT · MV.CM_DIFF · TDD lajolla_r.
- 산출: deck/assets/v8/G3_flow.png(RGB 3600 × 1395) + G3_flow.pdf + G3_flow_spec.json.
- 실행(CPU 만): OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_g3_flow.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402  (canvas, T, RT, chevron, block, hline, vline, sub_axes, qa_and_save, setup_fonts)
import polar.style_v5 as S  # noqa: E402  (REGION_HEX, METHOD_HEX, PRODUCT_HEX)
import polar.testbed_v5 as TB  # noqa: E402  (겹 배정: _balanced_group_folds, folds_year)
import maps_alt_v3 as MV  # noqa: E402  (CM_ALT, CM_DIFF, lut, pct_range, sym_max, extend_of, ne_layers)
import v7_f6_maps as F6  # noqa: E402  (음영 드레이프: cell_table, cell_values, Base, render, frame, coast, graticule, scalebar)
import v7_f1_study_area as F1  # noqa: E402  (pfr_image, BOX_A, WATER, COAST)
import cmcrameri.cm as cmc  # noqa: E402

M.OUT = ROOT / "deck" / "assets" / "v8"
PROC = ROOT / "data" / "processed" / "v5"
RUN = PROC / "runs" / "core_v6"
MAPS = PROC / "maps"
STEM = "G3_flow"

# ================================================================ 배치(인치)
W, H = 12.0, 4.65
X0, X1 = 0.10, 11.90
TIP, GAP = 0.17, 0.12                                    # 셰브런 홈 깊이 · 셰브런 사이 가로 틈(사선 수직 간격 ≈ 0.12 × 0.31/√(0.17² + 0.31²) ≈ 0.105 in)
CW_CH = (X1 - X0 + 5 * (TIP - GAP)) / 6                  # 셰브런 폭(여섯이 홈에 맞물려 X0–X1 을 채운다)
PITCH = CW_CH - TIP + GAP                                # 열 간격
CX = [X0 + i * PITCH + CW_CH / 2 for i in range(6)]      # 열 가운데
CHEV_H = 0.62
CHEV_Y = H - 0.08 - CHEV_H                               # 셰브런 아래 끝
Y_TOP = CHEV_Y - 0.16                                    # 열 내용 위 끝
MW = 1.60                                                # 알래스카 지도 폭(②·④·⑥ 위)
CB_H = 0.085                                             # 가로 색 막대 높이
CB_X, CB_W = -0.88, 0.62                                 # 색 막대 왼쪽(열 가운데 기준)·폭. 단위 글줄은 오른쪽(열 안에 든다)
B_TOP, B_BOT = 1.62, 0.56                                # 아래 층 띠(보조 요소)
Y_LAB = 0.25                                             # 아래 글줄(색 열쇠가 필요한 열만)
XL_YR = (2008.6, 2024.8)                                 # 연도 축
BLK_H = 0.64                                             # 물리·학습 블록 높이(학습 블록은 트리 위 · 글자 아래 두 층)
FS_T, FS_K, FS_CH = 13.0, 12.0, 14.0                     # 한글 라벨 · 눈금 · 셰브런(pt)
R1 = S.METHOD_HEX["R1"]; P1 = S.METHOD_HEX["P1"]
CM_TDD = MV.lut(cmc.lajolla_r, 0.12, 0.80)                                  # v7 F3 CM_TDD(lajolla_r) 계열, 검은 끝을 잘라 지도가 어둡지 않게(낮음 = 밝음)
NODATA = S.NEUTRAL["no_data"]
REG_KO = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}       # 그림 지역 이름(영문, S.REGION_EN 과 같다)
SCHEMES = [("random", "Random"), ("block", "Spatial block"), ("year", "Year hold-out"), ("block_year", "Block × year"), ("region", "Region hold-out")]
STEPS = ["Cell-year data", "Year-specific\nclimate input", "Physics baseline\n(Stefan)", "Residual learning", "Validation\n(five splits)", "Maps and\nproduct comparison"]


def _tint(hex_, a):
    """hex_ 를 흰색과 a : 1 − a 로 섞는다(옅은 채움·테두리)."""
    from matplotlib.colors import to_hex, to_rgb
    c = np.array(to_rgb(hex_)); return to_hex(a * c + (1 - a) * np.ones(3))


# 단계 종류별 (채움, 테두리). 색상 원천: 자료 = 방법 P0 청회색 #8A9BB0, 기후 입력 = TDD 색표(lajolla_r) 중간, 물리·학습 = 토큰 process 채움 + method_figs 테두리,
# 검증 = 제품 Aalto 라벤더 #7D6B9E, 지도·제품 = ALT 색표(oslo_r) 중간. 채움은 흰색과 섞은 옅은 색, 테두리는 같은 색상의 조금 진한 색
STAGE = [(_tint(S.METHOD_HEX["P0"], 0.24), _tint(S.METHOD_HEX["P0"], 0.85)),
         (_tint(CM_TDD(0.40), 0.20), _tint(CM_TDD(0.40), 0.72)),
         (M.GRN, M.GRN_EDGE),
         (M.YEL, M.YEL_EDGE),
         (_tint(S.PRODUCT_HEX["Aalto"], 0.18), _tint(S.PRODUCT_HEX["Aalto"], 0.66)),
         (_tint(MV.CM_ALT(0.55), 0.18), _tint(MV.CM_ALT(0.55), 0.62))]
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
BOX = dict(F1.BOX_A); BOX["Canada"] = (-140.0, F1.BOX_A["Canada"][1], F1.BOX_A["Canada"][2], F1.BOX_A["Canada"][3])   # ① 캐나다 상자 서쪽 변 −141 → −140: 알래스카 상자와 공유하던 변을 띄운다(1° ≈ 3 px)
LENA_ZOOM = (128.85, 130.30, 71.47, 71.92)                                   # ⑤ 레나 남동부 확대(블록 4개 · 셀-연도 ≈ 52 %), 종횡비 ≈ 1, 시험 블록 아래 변이 보이게
TRAIN_HEX, TEST_HEX, EXCL_HEX = M.YEL_EDGE, M.GRN_EDGE, NODATA                # ⑤ 학습(학습 노랑 테두리색) · 시험(평가 초록 테두리색) · 제외(중립)
BLK_GRID = (129.0, 129.5, 130.0)                                             # ⑤ 0.5° 블록 경계(확대 범위 안)
WEI_CACHE = F6.CACHE / "f8_alaska_wei_2014_s1.npz"                            # ⑥ Wei 2026 2014 음영 격자(step 1) 재투영 캐시(v8_f8_product_maps.product_px)
SPEC: dict = {"stem": STEM, "size_in": [W, H], "layout": {}, "sources": {}, "choices": {}}


# ================================================================ 축 양식
def src(key, path):
    SPEC["sources"][key] = str(Path(path).relative_to(ROOT))



def minus(s):
    return s.replace("-", "−")



def chart_axes(fig, x, y, w, h):
    ax = M.sub_axes(fig, x, y, w, h)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(0.8); ax.spines[k].set_edgecolor(M.ZERO)
    ax.tick_params(labelsize=FS_K, length=3, width=0.8, pad=2, colors=M.INK)
    return ax



def hcbar(fig, ax, x, y, w, cmap, norm, ticks, label, extend="neither"):  # noqa: D401
    """가로 색 막대 + 막대 위 왼쪽 단위 라벨(13 pt, 열 폭 안). 눈금 12 pt. (x, y) = 막대 왼쪽 아래."""
    cax = M.sub_axes(fig, x, y, w, CB_H)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks, extend=extend, extendfrac=0.08)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(M.ZERO)
    cax.tick_params(labelsize=FS_K, length=3, width=0.8, pad=2, colors=M.INK)
    M.T(ax, x - 0.02, y + CB_H + 0.05, label, size=FS_T, ha="left", va="bottom")
    return cax



def swatch_row(ax, cx, y, items, size=FS_T, color=M.AUX, r=0.045, gap=0.06, sep=0.18):
    """색 열쇠 한 줄(원 견본 + 낱말), cx 가운데 정렬. items = [(hex, 낱말), ...]."""
    fig = ax.figure; rd = fig.canvas.get_renderer(); ws = []
    for _, s in items:
        t = ax.text(0, 0, s, fontsize=size); ws.append(t.get_window_extent(rd).width / fig.dpi); t.remove()
    total = sum(2 * r + gap + w for w in ws) + sep * (len(items) - 1)
    x = cx - total / 2
    for (c, s), w in zip(items, ws):
        ax.add_patch(Circle((x + r, y), r, facecolor=c, edgecolor="none", zorder=6))
        M.T(ax, x + 2 * r + gap, y, s, size=size, ha="left", color=color)
        x += 2 * r + gap + w + sep
    return total



# ================================================================ 자료
def load_labels():
    L = pd.read_csv(PROC / "labels_cell_year_v5.csv"); src("labels", PROC / "labels_cell_year_v5.csv")
    return L[L.in_forcing == 1].copy()



class Alaska:
    """알래스카 1 km 지도 자료(2014 연도 지도: TDD · 보정량 R1 − P1 · ALT R1)와 음영 격자(v7 F6 Base, 250 m × 2)."""

    def __init__(self):
        t = time.time()
        self.key = "alaska"
        self.cells, lut, ky0, kx0 = F6.cell_table(self.key)
        self.base = F6.Base(self.key, self.cells, lut, ky0, kx0, step=2, tag="full")
        self.tf = self.base.tf; self.lim = F6.region_limits(self.key, self.tf)
        self.asp = (self.lim[3] - self.lim[2]) / (self.lim[1] - self.lim[0])
        f = MAPS / "Alaska" / "pred_2014.csv.gz"; src("map_2014", f)
        p = pd.read_csv(f, dtype={"cell_id": str}, usecols=["cell_id", "e5_tdd", "p1", "r1"]).set_index("cell_id")
        m = pd.read_csv(MAPS / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"]).set_index("cell_id")
        src("map_mean_aoa", MAPS / "Alaska" / "mean_2011_2024.csv.gz")
        self.aoa = np.nan_to_num(F6.cell_values(self.key, self.cells, m.aoa_in.astype(float)), nan=0.0)
        self.tdd = F6.cell_values(self.key, self.cells, p.e5_tdd)
        self.corr = F6.cell_values(self.key, self.cells, p.r1 - p.p1)
        self.alt = F6.cell_values(self.key, self.cells, p.r1)
        fin = lambda v: v[np.isfinite(v)]  # noqa: E731
        self.tdd_rng = MV.pct_range(fin(self.tdd), step=100.0)
        self.corr_max = MV.sym_max(fin(self.corr), q=98.0, step=5.0)
        self.alt_rng = MV.pct_range(fin(self.alt), step=5.0)
        SPEC["choices"]["alaska_2014"] = dict(n_cells_drawn=int(np.isfinite(self.alt).sum()), tdd_range=self.tdd_rng, corr_sym_max=self.corr_max,
                                              alt_range=self.alt_rng, hillshade="map_alaska/hillshade_v1/shade_250m.tif, step 2", aoa_fade="corr·alt 만(F6 와 같다)",
                                              tdd_cmap="lajolla_r trimmed 0.12–0.80 (deck thumbnail)", elapsed_s=round(time.time() - t, 1))

    def frame_axes(self, fig, x, y, w, rgb, M6):
        h = w * self.asp
        axm = M.sub_axes(fig, x, y, w, h)
        F6.frame(axm, rgb, self.base.extent, self.lim, M6); F6.coast(axm, self.key, self.tf, M6)
        F6.graticule(axm, self.key, self.tf, self.lim, M6)
        return axm, h

    def draw(self, fig, x, y, w, kind, M6, scale=True):
        hatch = self.base.hatch_px(w * 25.4, self.lim, M6.hatch_mm)
        if kind == "tdd":
            rgb = F6.render(self.base, self.tdd, CM_TDD, Normalize(*self.tdd_rng), feather=2.0, hatch=hatch)
        elif kind == "corr":
            v = self.corr_max
            rgb = F6.render(self.base, self.corr, MV.CM_DIFF, TwoSlopeNorm(0.0, -v, v), aoa=self.aoa, feather=2.0, aoa_sigma=3.0, hatch=hatch)
        else:
            rgb = F6.render(self.base, self.alt, MV.CM_ALT, Normalize(*self.alt_rng), aoa=self.aoa, feather=2.0, aoa_sigma=3.0, hatch=hatch)
        axm, h = self.frame_axes(fig, x, y, w, rgb, M6)
        if scale:
            scalebar_in(axm, self.lim, 200, M6)
        return axm, h

    def draw_wei(self, fig, x, y, w, M6):
        """Wei 2026 의 2014 ALT 를 같은 음영 격자·같은 색표·범위로 드레이프. 우리 지도가 그려지는 화소(같은 영역)만 칠한다."""
        if WEI_CACHE.exists():
            v1 = np.load(WEI_CACHE)["v"]; how = f"cache {WEI_CACHE.relative_to(ROOT)}"
        else:
            import v8_f8_product_maps as F8
            c1, lut1, ky1, kx1 = F6.cell_table(self.key)
            b1 = F6.Base(self.key, c1, lut1, ky1, kx1, step=1, tag="full")
            v1 = F8.product_px(b1, "wei", 2014); how = "v8_f8_product_maps.product_px (step 1)"
        hh, ww = self.base.idx.shape
        v = np.nanmean(v1[: 2 * hh, : 2 * ww].reshape(hh, 2, ww, 2), axis=(1, 3))    # step 1 → step 2 (2 × 2 평균)
        idx = self.base.idx; has = idx >= 0
        ours = np.zeros(idx.shape, bool); ours[has] = np.isfinite(self.alt[idx[has]])
        drawn = ours & self.base.valid & np.isfinite(v)
        water = self.base.water_cell | (~self.base.land & ~drawn)
        rgb = np.empty(idx.shape + (3,), np.float32); rgb[:] = F6.LAND; rgb[water] = F6.WATER
        col = MV.CM_ALT(Normalize(*self.alt_rng)(np.where(drawn, v, np.nanmean(v[drawn]))))[..., :3].astype(np.float32)
        rgb[drawn] = col[drawn]
        sh = np.where(drawn & self.base.valid, self.base.shade, self.base.shade_fill); sh = np.where(water, 1.0, sh)
        rgb = (np.clip(rgb * ((1 - F6.W_SHADE) + F6.W_SHADE * sh)[..., None], 0, 1) * 255).astype(np.uint8)
        SPEC["choices"]["wei_2014_thumb"] = dict(source=how, resample="2 × 2 nanmean to step 2", mask="pixels drawn in our 2014 map",
                                                 n_px=int(drawn.sum()), colour_range=self.alt_rng, cmap="MV.CM_ALT (same as ours)")
        return self.frame_axes(fig, x, y, w, rgb, M6)


def scalebar_in(axm, lim, km, M6, fx=0.06, fy=0.07):
    """지도 틀 안 왼쪽 아래 축척 막대(투영 좌표 m): 막대 + 양 끝 눈금, 글자는 막대 위에 왼맞춤(틀 밖으로 나가지 않는다), 흰 테."""
    x0, x1, y0, y1 = lim; L = km * 1000.0; tick = 0.016 * (y1 - y0)
    xs = x0 + fx * (x1 - x0); ys = y0 + fy * (y1 - y0)
    halo = [pe.withStroke(linewidth=M6.halo + 0.4, foreground="white")]
    axm.plot([xs, xs + L], [ys, ys], color="black", lw=M6.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    for xx in (xs, xs + L):
        axm.plot([xx, xx], [ys, ys + tick], color="black", lw=0.8 * M6.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    t = axm.text(xs, ys + 1.6 * tick, f"{km} km", ha="left", va="bottom", fontsize=M6.fs_tick, zorder=9, path_effects=M6.halo_fx)
    SCALEBARS.append((axm, t))
    return t


SCALEBARS: list = []


def map_label(axm, s, color, size=FS_T):
    """지도 틀 안 왼쪽 위 이름(흰 테). 알래스카 지도의 왼쪽 위는 척치해(물)."""
    return axm.text(0.04, 0.95, s, transform=axm.transAxes, ha="left", va="top", fontsize=size, color=color, zorder=9,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def unit_label(ax, a, s, size=FS_T):
    """작은 차트의 단위 글줄: 축 왼쪽 위(눈금 글자 열 왼쪽 끝에 맞춤), 회전 없음."""
    fig = a.figure; r = fig.canvas.get_renderer(); fig.canvas.draw()
    x_left = min(t.get_window_extent(r).x0 for t in a.get_yticklabels() if t.get_text()) / fig.dpi
    p_ = a.get_position()
    return M.T(ax, x_left, (p_.y1 * H) + 0.13, s, size=size, color=M.AUX, ha="left")


def tdd_by_year(L):
    """알래스카 라벨 셀(in_forcing)의 연 TDD 평균(2010–2024), forcing_year.csv e5_tdd."""
    f = PROC / "forcing_year.csv"; src("forcing", f)
    F = pd.read_csv(f, usecols=["loc_id", "year", "e5_tdd"])
    ids = L[L.region == "Alaska"].loc_id.unique()
    s = F[F.loc_id.isin(ids)].groupby("year").e5_tdd.mean().reindex(range(2010, 2025))
    SPEC["choices"]["alaska_tdd_by_year"] = dict(n_cells=int(len(ids)), mean_tdd={int(k): round(float(v), 1) for k, v in s.items()})
    return s



def lena_stefan():
    """레나 관측 셀-연도의 ALT 대 √TDD(관측일까지), 지도 산출에 쓴 재보정 계수 E(maps/Lena/meta.json)."""
    f = PROC / "site_series" / "site_series_year_obs.csv.gz"; src("site_series", f)
    s = pd.read_csv(f); q = s[(s.region == "Lena") & (s.observed == 1) & s.tdd_obs.notna() & s.y.notna()]
    meta = json.loads((MAPS / "Lena" / "meta.json").read_text()); src("lena_E", MAPS / "Lena" / "meta.json")
    x = np.sqrt(q.tdd_obs.values); y = q.y.values; E = float(meta["E"])
    E_ls = float((x @ y) / (x @ x))
    SPEC["choices"]["lena_stefan"] = dict(n_cell_years=int(len(q)), n_cells=int(q.loc_id.nunique()), E_map=round(E, 4), E_origin_ls=round(E_ls, 4),
                                          sqrt_tdd_range=[round(float(x.min()), 2), round(float(x.max()), 2)], alt_max=float(y.max()),
                                          n_alt_above_150=int((y > 150).sum()))
    return x, y, E



def lena_split_example(L):
    """v7 F3 c 와 같은 겹 배정(testbed_v5: 무작위 = 행 순열 % 5, 블록 = 균형 그룹 겹, 연도 = 시험 행 ≥ 20 인 해, 블록 × 연도, 지역)에서
    예시 하나(겹 1 = f 0, 연도 = 레나 최다 관측 해)를 골라 셀-연도를 세 범주로 나눈다: 2 = 시험, 1 = 학습, 0 = 제외(블록 × 연도만 생긴다)."""
    q = L[L.region == "Lena"].reset_index(drop=True)
    f_rand = np.random.RandomState(0).permutation(len(q)) % 5
    f_blk = TB._balanced_group_folds(q.block.values, 5, 0)
    y_scored = sorted(int(f.name[1:]) for f in TB.folds_year(q))
    yr = int(q.year.value_counts().idxmax()); assert yr in y_scored, (yr, y_scored)
    yv = q.year.values
    cat = {"random": np.where(f_rand == 0, 2, 1), "block": np.where(f_blk == 0, 2, 1), "year": np.where(yv == yr, 2, 1),
           "block_year": np.where((f_blk == 0) & (yv == yr), 2, np.where((f_blk != 0) & (yv != yr), 1, 0)), "region": np.full(len(q), 2)}
    bq = q.assign(f=f_blk).groupby("block").agg(lon=("lon", "min"), lat=("lat", "min"), f=("f", "first"))
    bq["x"] = np.floor(bq.lon / 0.5) * 0.5; bq["y"] = np.floor(bq.lat / 0.5) * 0.5
    test_blocks = bq[bq.f == 0]
    lo0, lo1, la0, la1 = LENA_ZOOM
    inz = (q.lon >= lo0) & (q.lon <= lo1) & (q.lat >= la0) & (q.lat <= la1)
    SPEC["choices"]["lena_splits"] = dict(
        n_cell_years=int(len(q)), n_cells=int(q.loc_id.nunique()), n_blocks=int(q.block.nunique()), years_scored=y_scored, seed=0,
        example=dict(fold="f0 (겹 1)", year=yr, test_blocks=[str(b) for b in test_blocks.index]),
        zoom=dict(extent=list(LENA_ZOOM), n_cell_years=int(inz.sum()), n_cells=int(q.loc_id[inz].nunique()), blocks=sorted(str(b) for b in q.block[inz].unique())),
        counts_in_zoom={s: {"test": int((c[inz] == 2).sum()), "train": int((c[inz] == 1).sum()), "excluded": int((c[inz] == 0).sum())} for s, c in cat.items()},
        encoding="dot = cell-year (test drawn last); train = YEL_EDGE, test = GRN_EDGE (process block edge colours), excluded = neutral no_data; "
                 "0.5° block grid hairlines; test block outline in test colour for block / block × year; region hold-out = all Lena cells test",
        colours=dict(train=TRAIN_HEX, test=TEST_HEX, excluded=EXCL_HEX), note="fold palette (Okabe–Ito 5) of v7 F3 c is not used here: two-tone train/test is legible at thumbnail scale")
    return q, cat, test_blocks, yr



def tree_glyph(ax, cx, ytop, w=0.26, h=0.30, color=M.INK):
    """결정 트리 글리프(깊이 2): 마디 원 + 곧은 가지."""
    r = 0.018; dy = h / 2
    nodes = {(0, 0): (cx, ytop - r)}
    for k in (-1, 1):
        nodes[(1, k)] = (cx + k * w * 0.27, ytop - r - dy)
        for j in (-1, 1):
            nodes[(2, 2 * k + j)] = (cx + k * w * 0.27 + j * w * 0.13, ytop - r - 2 * dy)
    for (lv, k), (px, py) in nodes.items():
        if lv == 0:
            continue
        parent = nodes[(lv - 1, 0 if lv == 1 else (1 if k > 0 else -1))]
        ax.plot([parent[0], px], [parent[1], py], color=color, lw=0.8, solid_capstyle="round", zorder=4)
    for (lv, k), (px, py) in nodes.items():
        ax.add_patch(Circle((px, py), r, facecolor="white" if lv < 2 else color, edgecolor=color, lw=0.8, zorder=5))



# ================================================================ 열별 그리기
def col1_data(fig, ax, L):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.path as mpath
    i = 0; D = 1.74
    proj = ccrs.NorthPolarStereo(central_longitude=-100); PC = ccrs.PlateCarree()
    src("pfr", ROOT / "data" / "processed" / "cci_pfr_mean_1997_2021.nc")
    axm = M.sub_axes(fig, CX[i] - D / 2, Y_TOP - D, D, D, projection=proj); axm.set_extent([-180, 180, 52, 90], PC)
    th = np.linspace(0, 2 * np.pi, 240)
    axm.set_boundary(mpath.Path(np.c_[0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)]), transform=axm.transAxes)
    axm.set_facecolor("white")
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor=M.LAND, edgecolor="none", zorder=0).set_rasterized(True)
    M.draw_pfr_slide(axm, proj, D, D)                      # 영구동토 2단계(연속·불연속), 토큰 map_base.slide 색(덱 범북극 지도 양식)
    axm.coastlines("50m", linewidth=0.45, color=M.COAST, zorder=2)
    axm.gridlines(crs=PC, draw_labels=False, linewidth=0.35, color=M.GRATICULE, xlocs=range(-180, 180, 60), ylocs=[60, 70, 80], zorder=1)
    cells = L.drop_duplicates("loc_id")
    for r in ("Alaska", "Lena", "Canada"):
        lo0, lo1, la0, la1 = BOX[r]
        lons = np.r_[np.linspace(lo0, lo1, 40), np.full(40, lo1), np.linspace(lo1, lo0, 40), np.full(40, lo0)]
        lats = np.r_[np.full(40, la0), np.linspace(la0, la1, 40), np.full(40, la1), np.linspace(la1, la0, 40)]
        axm.plot(lons, lats, transform=PC, color=S.REGION_HEX[r], lw=1.5, zorder=5, path_effects=HALO)
        c = cells[cells.region == r]
        axm.scatter(c.lon, c.lat, s=1.0, c=M.INK, lw=0, alpha=0.7, transform=PC, zorder=4, rasterized=True)
    axm.spines["geo"].set_linewidth(0.7); axm.spines["geo"].set_edgecolor("#777777")
    # 연도별 셀-연도 수(지역 누적 막대), 단위 글줄은 왼쪽 위
    cnt = L.groupby(["year", "region"]).size().unstack(fill_value=0).reindex(range(2010, 2025), fill_value=0)
    SPEC["choices"]["cell_years_by_year"] = {r: {int(y): int(v) for y, v in cnt[r].items()} for r in ("Alaska", "Lena", "Canada")}
    a = chart_axes(fig, CX[i] - 0.62, B_BOT + 0.22, 1.50, B_TOP - 0.30 - (B_BOT + 0.22))
    bot = np.zeros(len(cnt))
    for r in ("Alaska", "Lena", "Canada"):
        v = cnt[r].values / 1000.0
        a.bar(cnt.index, v, bottom=bot, width=0.75, color=S.REGION_HEX[r], lw=0); bot = bot + v
    a.set_xlim(*XL_YR); a.set_xticks([2010, 2015, 2020]); a.set_ylim(0, 8.5); a.set_yticks([0, 4, 8])
    u = unit_label(ax, a, "Cell-years (10³)")
    (arts, tot) = M.RT(ax, X0, Y_LAB, ["Alaska", " · ", "Lena Delta", " · ", "Canada (NW)"], size=FS_T, ha="left",
                       color=[S.REGION_HEX["Alaska"], M.AUX, S.REGION_HEX["Lena"], M.AUX, S.REGION_HEX["Canada"]])   # 지역 열쇠(그림 왼쪽 아래, 왼맞춤)
    SPEC["layout"]["col1"] = dict(polar_D_in=D, region_line_w_in=round(float(tot), 3))
    return [axm, a]


def col2_climate(fig, ax, A, L, M6):
    i = 1
    x = CX[i] - MW / 2
    axm, h = A.draw(fig, x, Y_TOP - MW * A.asp, MW, "tdd", M6)
    yb = Y_TOP - h
    lo, hi = A.tdd_rng
    ticks = [v for v in (500, 1500) if lo <= v <= hi]
    hcbar(fig, ax, CX[i] - MW / 2 + 0.06, yb - 0.32 - CB_H, 0.74, CM_TDD, Normalize(lo, hi), ticks, "TDD (°C·d)", extend=MV.extend_of(A.tdd[np.isfinite(A.tdd)], lo, hi))
    map_label(axm, "2014", M.INK)
    s = tdd_by_year(L)
    a = chart_axes(fig, CX[i] - 0.62, B_BOT + 0.22, 1.50, B_TOP - 0.30 - (B_BOT + 0.22))
    cols = [CM_TDD(0.92) if y == 2014 else CM_TDD(0.45) for y in s.index]
    a.bar(s.index, s.values, width=0.75, color=cols, lw=0)
    top2 = float(s.max()) * 1.55                                              # 모든 막대 위에 '2014' 글줄 한 줄(축 위 단위 글줄과 떨어지게)
    a.set_xlim(*XL_YR); a.set_xticks([2010, 2015, 2020]); a.set_ylim(0, top2); a.set_yticks([0, 1000])   # 낮은 축: 눈금 둘
    a.text(2014, float(s.max()) * 1.04, "2014", ha="center", va="bottom", fontsize=FS_K, color=M.INK)      # 짙은 막대(2014) 위, 이웃 막대보다 높게
    u = unit_label(ax, a, "Annual TDD (°C·d)")
    SPEC["layout"]["col2"] = dict(map_wh_in=[MW, round(float(h), 3)])
    return [axm, a]


def col3_physics(fig, ax):
    i = 2
    x, y, E = lena_stefan()
    a = chart_axes(fig, CX[i] - 0.44, 2.42, 1.26, Y_TOP - 0.26 - 2.42)
    YMAX = 150.0; hi = y > YMAX
    a.scatter(x[~hi], y[~hi], s=5.0, c=M.INK, alpha=0.30, lw=0, zorder=3, rasterized=True)
    a.scatter(x[hi], np.full(int(hi.sum()), YMAX), marker="^", s=18, facecolor="white", edgecolor=M.INK, lw=0.7, clip_on=False, zorder=5)
    SPEC["choices"]["lena_stefan"]["clipped"] = dict(n=int(hi.sum()), marker="open up-triangle on the top spine at y = 150 cm", y_max_data=float(y.max()))
    xm = float(np.ceil(x.max())) + 1.0; x0 = float(np.floor(x.min())) - 1.0
    a.plot([0, xm], [0, E * xm], color=P1, lw=2.0, solid_capstyle="butt", zorder=4)
    a.set_xlim(x0, xm); a.set_ylim(0, YMAX); a.set_xticks([20, 25, 30]); a.set_yticks([0, 50, 100, 150])
    a.set_xlabel(r"$\sqrt{\mathrm{TDD}}$ ($\sqrt{\mathrm{°C·d}}$)", fontsize=FS_K, labelpad=2); a.set_ylabel("ALT (cm)", fontsize=FS_T, labelpad=3)   # 12 pt: 열 폭 안
    p_ = a.get_position()
    t_reg = M.T(ax, p_.x0 * W, p_.y1 * H + 0.13, "Lena Delta", size=FS_T, color=S.REGION_HEX["Lena"], ha="left")   # 축 위(점과 겹치지 않게)
    bw = 1.62
    M.block(ax, CX[i] - bw / 2, (B_TOP + B_BOT) / 2 - BLK_H / 2 + 0.05, bw, BLK_H, r"$\mathrm{ALT} = E\sqrt{\mathrm{TDD}}$", fill=M.GRN, size=FS_CH)
    return [a]


def col4_residual(fig, ax, A, M6):
    i = 3
    x = CX[i] - MW / 2
    axm, h = A.draw(fig, x, Y_TOP - MW * A.asp, MW, "corr", M6)
    yb = Y_TOP - h; v = A.corr_max
    tk = [-v, 0, v] if v <= 10 else [-10, 0, 10]
    hcbar(fig, ax, CX[i] - MW / 2 + 0.06, yb - 0.32 - CB_H, 0.74, MV.CM_DIFF, TwoSlopeNorm(0.0, -v, v), tk, "Correction (cm)",
          extend=MV.extend_of(A.corr[np.isfinite(A.corr)], -v, v))
    map_label(axm, "2014", M.INK)
    bw = 1.62; by = (B_TOP + B_BOT) / 2 - BLK_H / 2 + 0.05; bx = CX[i] - bw / 2
    M.block(ax, bx, by, bw, BLK_H, None, fill=M.YEL)                          # 학습 블록(노랑): 위 = 트리 앙상블 글리프, 아래 = '잔차 ML'
    for k in range(3):
        tree_glyph(ax, CX[i] + (k - 1) * 0.34, by + BLK_H - 0.06, w=0.26, h=0.22)
    M.T(ax, CX[i], by + 0.15, "Residual ML", size=FS_T, ha="center")
    SPEC["layout"]["col4"] = dict(map_wh_in=[MW, round(float(h), 3)])
    return [axm]


def col5_splits(fig, ax, L):
    """다섯 줄 목록: 줄마다 [레나 남동부 확대 지도(학습·시험·제외 셀)] [분할 이름]. 위에서 아래로 무작위 → 지역 홀드아웃."""
    i = 4
    q, cat, tb, yr = lena_split_example(L)
    ne = MV.ne_layers("lena"); ext = LENA_ZOOM; asp = 1.0 / np.cos(np.deg2rad((ext[2] + ext[3]) / 2))
    ytop = Y_TOP; ybot = B_BOT - 0.02
    rp = (ytop - ybot) / 5.0; mh = rp - 0.09
    mw = mh * (ext[1] - ext[0]) / ((ext[3] - ext[2]) * asp)
    w_names = max(M.RT(ax, -10, -10, [ko], size=FS_T)[1] for _, ko in SCHEMES)
    for t in list(ax.texts):
        if t.get_position()[0] < -5:
            t.remove()
    total = mw + 0.14 + w_names
    xl = CX[i] - total / 2
    rng = np.random.RandomState(1)
    axes = []
    for k, (sch, ko) in enumerate(SCHEMES):
        y0 = ytop - k * rp - mh
        a = M.sub_axes(fig, xl, y0, mw, mh); axes.append(a)
        for gg in ne["land"]:
            for part in getattr(gg, "geoms", [gg]):
                xs, ys = part.exterior.xy; a.fill(xs, ys, facecolor=M.LAND, edgecolor="none", lw=0, zorder=1)
        for gg in ne["coastline"]:
            for part in getattr(gg, "geoms", [gg]):
                xs, ys = part.xy; a.plot(xs, ys, color=M.COAST, lw=0.4, zorder=2)
        for gl in BLK_GRID:                                                                  # 0.5° 블록 경계(헤어라인)
            a.plot([gl, gl], [ext[2], ext[3]], color="#B0B0B0", lw=0.5, zorder=2.5)
        a.plot([ext[0], ext[1]], [71.5, 71.5], color="#B0B0B0", lw=0.5, zorder=2.5)
        if sch in ("block", "block_year"):                                                   # 시험 블록 테두리
            for _, bb in tb.iterrows():
                a.add_patch(Rectangle((bb.x, bb.y), 0.5, 0.5, facecolor="none", edgecolor=TEST_HEX, lw=1.2, zorder=3.5))
        c = cat[sch]
        if sch == "random":                                                                  # 무작위: 섞은 순서로 한 번에(범주가 덮지 않게)
            o = rng.permutation(len(q)); cols = np.where(c[o] == 2, TEST_HEX, TRAIN_HEX)
            a.scatter(q.lon.values[o], q.lat.values[o], s=3.5, c=cols, lw=0, zorder=4, rasterized=True)
        else:                                                                                # 제외 → 학습 → 시험 순으로 겹쳐 그린다
            for cv, hx, z in ((0, EXCL_HEX, 3.6), (1, TRAIN_HEX, 3.8), (2, TEST_HEX, 4.0)):
                m = c == cv
                if m.any():
                    a.scatter(q.lon.values[m], q.lat.values[m], s=3.5, color=hx, lw=0, zorder=z, rasterized=True)
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_aspect(asp); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_edgecolor(M.HAIR); sp.set_linewidth(0.6)
        M.T(ax, xl + mw + 0.14, y0 + mh / 2, ko, size=FS_T, ha="left")
    key_w = swatch_row(ax, CX[i], Y_LAB, [(TRAIN_HEX, "Train"), (TEST_HEX, "Test"), (EXCL_HEX, "Excluded")])
    SPEC["layout"]["col5"] = dict(map_wh_in=[round(mw, 3), round(mh, 3)], row_pitch_in=round(rp, 3), names_w_in=round(float(w_names), 3), key_w_in=round(float(key_w), 3))
    return axes


def col6_maps_products(fig, ax, A, M6):
    i = 5
    x = CX[i] - MW / 2
    axm, h = A.draw(fig, x, Y_TOP - MW * A.asp, MW, "alt", M6)
    yb = Y_TOP - h; lo, hi = A.alt_rng
    hcbar(fig, ax, CX[i] - MW / 2 + 0.06, yb - 0.32 - CB_H, 0.74, MV.CM_ALT, Normalize(lo, hi), [v for v in (40, 60, 80) if lo <= v <= hi], "ALT 2014 (cm)",
          extend=MV.extend_of(A.alt[np.isfinite(A.alt)], lo, hi))
    map_label(axm, "Physics +\nresidual ML", R1)
    # 아래 층: 같은 해 Wei 2026(같은 음영 격자·색표·범위·영역)
    wh = B_TOP - (B_BOT - 0.10); ww = wh / A.asp
    axw, hw = A.draw_wei(fig, CX[i] - ww / 2, B_BOT - 0.10, ww, M6)
    map_label(axw, "Wei 2026", S.PRODUCT_HEX["wei"])
    SPEC["layout"]["col6"] = dict(map_wh_in=[MW, round(float(h), 3)], wei_wh_in=[round(float(ww), 3), round(float(wh), 3)])
    return [axm, axw]


# ================================================================ 점검
def tick_label_artists(fig):
    ids = set()
    for a in fig.axes:
        if hasattr(a, "projection") or not a.axison:
            continue
        for t in a.get_xticklabels(minor=False) + a.get_yticklabels(minor=False):
            ids.add(id(t))
    return ids



def no_hangul(fig):
    """그림 안 모든 글자 객체(눈금 글자 포함)에 한글이 있으면 멈춘다(v10: 그림 글자는 영어만)."""
    import re
    bad = [t.get_text() for t in fig.findobj(Text) if re.search("[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", t.get_text())]
    if bad:
        raise RuntimeError(f"한글 글자 남음: {bad[:8]}")


def korean(s):
    return any("가" <= ch <= "힣" for ch in s)



# ================================================================ 그림
def g3():
    t0 = time.time()
    M.setup_fonts()
    fig, ax = M.canvas(W, H)
    L = load_labels()
    # ---- 셰브런(흰 바탕 + 진한 외곽선 + 잉크 글자). 글자 폭이 셰브런 안쪽 폭에 드는지 잰다
    rr = fig.canvas.get_renderer(); fits = []
    for i, s_ in enumerate(STEPS):
        M.chevron(ax, X0 + i * PITCH, CHEV_Y, CW_CH, CHEV_H, STAGE[i][0], s_, first=(i == 0), tip=TIP, size=FS_T, edge=STAGE[i][1], tcolor=M.INK)
    fig.canvas.draw()
    yc = CHEV_Y + CHEV_H / 2; hh = CHEV_H / 2
    for t in ax.texts:
        if t.get_text() in STEPS:
            i = STEPS.index(t.get_text()); x = X0 + i * PITCH
            b = t.get_window_extent(rr); x0, x1, y0, y1 = b.x0 / fig.dpi, b.x1 / fig.dpi, b.y0 / fig.dpi, b.y1 / fig.dpi
            # 셰브런 왼쪽 홈 경계는 가운데 높이에서 가장 안쪽(x + TIP), 오른쪽 뾰족 경계는 글자 상자의 위·아래 끝에서 가장 안쪽
            d_near = 0.0 if y0 <= yc <= y1 else min(abs(y0 - yc), abs(y1 - yc)); d_far = max(abs(y0 - yc), abs(y1 - yc))
            left_lim = x if i == 0 else x + TIP * (1 - d_near / hh)
            right_lim = x + CW_CH - TIP * min(d_far / hh, 1.0)
            fits.append([t.get_text().replace("\n", " / "), round(x0 - left_lim, 3), round(right_lim - x1, 3)])   # [이름, 왼쪽 여유, 오른쪽 여유] 인치
    SPEC["layout"]["chevrons"] = dict(x0=X0, pitch=round(PITCH, 4), w=round(CW_CH, 4), y=CHEV_Y, h=CHEV_H, tip=TIP, gap=GAP, cx=[round(c, 3) for c in CX],
                                      style="stage tint fill + darker same-hue edge, ink text 14 pt", stage_colours=[list(c) for c in STAGE],
                                      perpendicular_gap_in=round(GAP * (CHEV_H / 2) / float(np.hypot(TIP, CHEV_H / 2)), 3), text_fit=fits)
    # ---- 열
    A = Alaska(); M6 = F6.Med("deck")
    col1_data(fig, ax, L)
    col2_climate(fig, ax, A, L, M6)
    col3_physics(fig, ax)
    col4_residual(fig, ax, A, M6)
    col5_splits(fig, ax, L)
    col6_maps_products(fig, ax, A, M6)
    # ---- 점검: 글자 크기(눈금 12 pt 허용, 한글 13 pt 이상, 그 밖 12 pt 이상), 축척 글자가 지도 틀 안인지, 셰브런 글자 폭
    fig.canvas.draw()
    bad = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not t.get_visible() or not s.strip():
            continue
        fs = float(t.get_fontsize())
        if korean(s) and fs < FS_T - 1e-6:
            bad.append([s[:30], fs, "korean<13"])
        elif fs < FS_K - 1e-6:
            bad.append([s[:30], fs, "<12"])
    sb_in = []
    for axm, t in SCALEBARS:
        a_ = axm.get_window_extent(rr); b_ = t.get_window_extent(rr)
        sb_in.append(bool(b_.x0 >= a_.x0 + 2 and b_.x1 <= a_.x1 - 2 and b_.y0 >= a_.y0 + 2 and b_.y1 <= a_.y1 - 2))
    chev_ok = all(lm >= 0.05 and rm >= 0.05 for _, lm, rm in fits)   # 글자 상자와 셰브런 경계(홈·뾰족 사선) 사이 여유 ≥ 0.05 in
    no_hangul(fig)
    rec = M.qa_and_save(fig, STEM, allow_overlap=())
    from PIL import Image
    png_s = M.OUT / f"{STEM}_slide.png"; pdf_s = M.OUT / f"{STEM}_slide.pdf"
    png = M.OUT / f"{STEM}.png"; pdf = M.OUT / f"{STEM}.pdf"
    with Image.open(png_s) as im:
        if im.mode != "RGB":
            bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None); bg.save(png, dpi=(M.DPI, M.DPI))
        else:
            im.save(png, dpi=(M.DPI, M.DPI))
        px = list(im.size)
    png_s.unlink(); os.replace(pdf_s, pdf)
    SPEC["qa"] = {k: rec[k] for k in ("min_font_pt", "texts", "outside", "overlaps", "small", "text_line", "text_axes", "text_patch")}
    SPEC["qa"]["png_px"] = px; SPEC["qa"]["png_px_ok"] = bool(px == [int(W * M.DPI), int(H * M.DPI)])
    SPEC["qa"]["font_size_violations"] = bad
    SPEC["qa"]["scalebar_text_inside_map"] = sb_in
    SPEC["qa"]["chevron_text_fits"] = chev_ok
    SPEC["qa"]["bare_sqrt_glyph_texts"] = [t.get_text()[:40] for t in fig.findobj(Text) if "√" in t.get_text()]
    SPEC["qa"]["elapsed_s"] = round(time.time() - t0, 1)
    (M.OUT / f"{STEM}_spec.json").write_text(json.dumps(SPEC, ensure_ascii=False, indent=1))
    print("글자 크기 점검:", "OK" if not bad else f"CHECK {bad}", "· 축척 글자 지도 안:", sb_in, "· 셰브런 글자 폭:", "OK" if chev_ok else f"CHECK {fits}",
          "· 글리프 √:", SPEC["qa"]["bare_sqrt_glyph_texts"] or "없음", "· 화소", px, f"· {time.time() - t0:.0f} s", flush=True)
    print("출력:", png, pdf, M.OUT / f"{STEM}_spec.json")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", action="store_true", help="호환용. 이 그림은 슬라이드판 하나만 만든다")
    ap.add_argument("--fig", default="G3", help="호환용")
    ap.parse_args()
    M.OUT.mkdir(parents=True, exist_ok=True)
    g3()
