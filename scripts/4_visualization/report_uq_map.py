"""신뢰 범위 지도 — 보정된 90% 예측구간 폭 + 적용가능 영역 표기.

의도(단일 메시지): 알래스카 ALT 예측지도의 신뢰 범위를 공간적으로 제시한다.
  (a) 지역 내(0.5° 공간블록 교차검증) 보정된 90% 예측구간 폭 (cm). 값이 클수록 신뢰 폭이 넓다.
      지역 간 전이 시 적용가능 영역 밖으로 판정된 셀은 해칭으로 덮어 외삽 구간을 명시한다.
  (b) 학습 분포와의 환경 차이 (무차원). 적용가능 영역 판정의 연속 근거.
  (c) 비유사도 구간별 지역 간 전이 오차와 90% 구간 커버리지. 비유사도 표기의 타당성 근거.

입력(모두 기저장 셀별 값. 모델·통계 재계산 없음):
  data/processed/s11_conformal_oof.csv     lat/lon/width90_cqr (알래스카 13,606 셀)
  data/processed/alt_cell_best_oof.csv     lat/lon/di/aoa_inside
  data/processed/alt_aoa_cell_transfer.csv 비유사도 구간별 RMSE·커버리지
  data/processed/s11_conformal_meta.json   보정 커버리지 헤드라인(콘솔 출력 검증용)

출력: outputs/maps/alt_uncertainty_aoa_map.png (300 dpi) + .pdf

실행: PYTHONPATH=src python3 scripts/4_visualization/report_uq_map.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C  # noqa: E402
from polar.plotstyle import use_polar  # noqa: E402
from polar.geomap import (make_ax, mask_ocean, to_proj, map_projection,  # noqa: E402
                          projected_aspect, add_scalebar, add_inset_locator)

plt = use_polar()
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from cmcrameri import cm as cmc  # noqa: E402

PROC = Path(C.PROCESSED)
OUT = Path(C.ROOT) / "outputs/maps"
OUT.mkdir(parents=True, exist_ok=True)
NAME = "alt_uncertainty_aoa_map"

# 관측 셀 분포(위도 61.2–71.3, 경도 -166.0–-141.1)에 맞춘 알래스카 창.
# 표준 프리셋(-170~-138, 59~72)은 자료 없는 여백이 넓어 패널 종횡비가 눕는다.
AK = ("알래스카", -168.0, -139.0, 59.5, 72.0)

# 저채도 냉색 팔레트(전 그림 공통 규약)
SLATE = "#8296a8"           # 막대·중립 강조 (저채도 청회)
HATCH_C = "#4a4a4a"
NA_EDGE, NA_FILL = "#9a9a9a", "#e6e8ea"   # 판정 없음 셀 표기

# 지도 컬러맵: 부호 없는 양수 스칼라 2종 → 단조 순차형. 양끝(순백·근흑) 절단으로
# 과포화를 막고, 저단은 육지 배경(#efece6, 웜그레이)과 구분되도록 냉색 틴트에서 시작한다.
CMAP_WIDTH = LinearSegmentedColormap.from_list(          # 청 → 남색 (cmcrameri oslo)
    "uq_width", cmc.oslo_r(np.linspace(0.18, 0.88, 256)))
CMAP_DI = LinearSegmentedColormap.from_list(             # 연청록 → 청록 (규약 팔레트)
    "uq_di", ["#cfdfe2", "#a9c2c7", "#84a5ad", "#628792", "#436a77", "#2b4d5c"])
for _c in (CMAP_WIDTH, CMAP_DI):
    _c.set_bad("#e9ecef")

GRIDSIZE = 30
# 인쇄 1:1. 아래 pt 값이 곧 인쇄 pt 이며 하한 6.5 pt 를 지킨다.
FS_AX, FS_TK = 7.0, 6.6
FS_CB, FS_CBT, FS_GEO = 6.8, 6.5, 6.5
FS_LEG, FS_NOTE, FS_LAB = 6.6, 6.5, 7.6


# ------------------------------------------------------------------ 자료 적재
def load_cells() -> pd.DataFrame:
    """셀별 보정 구간 폭과 비유사도·적용가능 영역 판정을 좌표로 결합한다.

    두 CSV는 동일한 알래스카 셀 집합(13,606)을 담되 좌표 문자열 정밀도가 달라
    소수 6자리 포맷 키로 1:1 결합한다(결합 전제를 assert로 검증).
    """
    w = pd.read_csv(PROC / "s11_conformal_oof.csv",
                    usecols=["lat", "lon", "region", "alt_cm", "width90_cqr"])
    b = pd.read_csv(PROC / "alt_cell_best_oof.csv",
                    usecols=["lat", "lon", "region", "di", "aoa_inside"])
    b = b[b.region.isin(w.region.unique())].copy()

    def key(d):
        return [f"{la:.6f}_{lo:.6f}" for la, lo in zip(d.lat, d.lon)]

    w["_k"], b["_k"] = key(w), key(b)
    assert w._k.is_unique and b._k.is_unique, "좌표 키 중복 — 1:1 결합 정의 불가"
    assert set(w._k) == set(b._k), "두 셀 집합 불일치 — 결합 전제 위반"

    df = w.merge(b[["_k", "di", "aoa_inside"]], on="_k", how="left")
    assert len(df) == len(w)
    return df.drop(columns="_k")


# ------------------------------------------------------------------ 지도 헬퍼
def hexfield(ax, lon, lat, vals, extent_xy, reduce=np.median, **kw):
    """경위도 관측을 육각 셀 통계로 집계(보간 없음). 패널 간 격자를 일치시키기 위해
    투영좌표 extent를 명시적으로 고정한다."""
    x, y = to_proj(ax, np.asarray(lon), np.asarray(lat))
    return ax.hexbin(x, y, C=np.asarray(vals), reduce_C_function=reduce,
                     gridsize=GRIDSIZE, mincnt=1, extent=extent_xy,
                     linewidths=0.15, edgecolors="face", rasterized=True, **kw)


def hex_extent(ax, lon, lat):
    """hexbin extent(투영좌표). 두 지도 패널의 육각 격자를 동일하게 맞추는 기준."""
    x, y = to_proj(ax, np.asarray(lon), np.asarray(lat))
    return (x.min(), x.max(), y.min(), y.max())


def hex_overlay(ax, hb_ref, select, hatch="///", color=HATCH_C, lw=0.35,
                facecolor="none", zorder=2.6):
    """hexbin과 동일한 육각형 위에 해칭·채움을 얹는다.

    hb_ref : 같은 격자로 집계한 hexbin(위치 기준. 그리기용 아니면 호출부에서 제거).
    select : hb_ref의 셀 순서에 대응하는 불리언 마스크.
    """
    poly = hb_ref.get_paths()[0].vertices             # 데이터 단위 육각형(중심 기준)
    off = np.asarray(hb_ref.get_offsets())            # 데이터 단위 셀 중심
    verts = [poly + off[i] for i in np.where(select)[0]]
    if not verts:
        return None
    pc = PolyCollection(verts, facecolors=facecolor, edgecolors=color,
                        linewidths=lw, hatch=hatch, zorder=zorder)
    pc.set_transform(ax.transData)
    ax.add_collection(pc)
    return pc


def _offkeys(hb):
    """육각 셀 중심 좌표를 1 m 단위 정수 키로. 두 hexbin의 셀 대응을 잡는다."""
    off = np.asarray(hb.get_offsets())
    return [(int(round(p[0])), int(round(p[1]))) for p in off]


def style_colorbar(cb, label):
    cb.set_label(label, fontsize=FS_CB)
    cb.ax.tick_params(labelsize=FS_CBT, length=2.5, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#b0b0b0")
    cb.solids.set_rasterized(False)     # PDF 인쇄 banding 방지(벡터 패스 유지)
    cb.solids.set_edgecolor("face")
    return cb


def geo_labels(ax, size=FS_GEO, left=True):
    """위경도 눈금 레이블 크기·회전 정리(cartopy 0.23: Gridliner는 ax.artists에 있음).

    left=False 면 위도 라벨을 두지 않는다. 두 지도 패널이 같은 범위이므로 오른쪽
    패널의 위도 라벨은 정보를 더하지 않고 왼쪽 패널 컬러바 이름과 겹친다.
    """
    from cartopy.mpl.gridliner import Gridliner
    gls = [a for a in ax.artists if isinstance(a, Gridliner)]
    if getattr(ax, "_gl", None) is not None and ax._gl not in gls:
        gls.append(ax._gl)
    for gl in gls:
        gl.xlabel_style = {"size": size, "color": "0.35"}
        gl.ylabel_style = {"size": size, "color": "0.35"}
        gl.rotate_labels = False
        gl.left_labels = left
        gl.right_labels = gl.top_labels = False


# ------------------------------------------------------------------ 레이아웃
# 지도 종횡비를 먼저 구해 축 사각형을 인치 단위로 직접 배치한다(패널 높이·정렬 고정).
# main.tex 삽입 폭 0.78 textwidth = 137.3 mm 와 같은 figsize 로 만들고 tight 크롭을
# 쓰지 않으므로 축소 배율이 1.0 이다. 여백은 위 글자 크기에 맞춰 다시 잡았다.
FIG_W = 137.3 / 25.4
M_L, M_R, M_T, M_B = 0.34, 0.07, 0.17, 0.34      # 바깥 여백 (in)
CB_GAP, CB_W, CB_LAB = 0.04, 0.075, 0.30         # 컬러바 블록 (in)
COL_GAP, ROW_GAP = 0.30, 0.60                    # 패널 간격 (in)
CHART_H = 1.02                                   # (c) 높이 (in)
LEG_DROP = 0.19                                  # 지도 하단 ~ 공통 범례 (in)

_ASP = projected_aspect(AK)                      # 투영 후 지도 가로/세로
_CB_BLOCK = CB_GAP + CB_W + CB_LAB
_MAP_W = (FIG_W - M_L - M_R - 2 * _CB_BLOCK - COL_GAP) / 2.0
_MAP_H = _MAP_W / _ASP
FIG_H = M_T + _MAP_H + ROW_GAP + CHART_H + M_B


def _rects():
    """(a)(b)(c) 축과 컬러바 축의 figure-fraction 사각형."""
    map_w, map_h, cb_block, chart_h = _MAP_W, _MAP_H, _CB_BLOCK, CHART_H
    y_map = M_B + chart_h + ROW_GAP
    xa, xb = M_L, M_L + map_w + cb_block + COL_GAP

    def fr(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    return {
        "a": fr(xa, y_map, map_w, map_h),
        "ca": fr(xa + map_w + CB_GAP, y_map, CB_W, map_h),
        "b": fr(xb, y_map, map_w, map_h),
        "cb": fr(xb + map_w + CB_GAP, y_map, CB_W, map_h),
        "c": fr(xa, M_B, xb + map_w - xa, chart_h),
        # 지도 공통 범례: 두 패널 가로 중앙, 경도 눈금 레이블 아래 한 줄
        "leg_x": (xa + (xb + map_w - xa) / 2.0) / FIG_W,
        "leg_y": (y_map - LEG_DROP) / FIG_H,
    }


# ------------------------------------------------------------------ 본 그림
def build(df: pd.DataFrame, dec: pd.DataFrame):
    R = _rects()
    proj = map_projection(AK)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axA = fig.add_axes(R["a"], projection=proj)
    axB = fig.add_axes(R["b"], projection=proj)
    axC = fig.add_axes(R["c"])
    caxA, caxB = fig.add_axes(R["ca"]), fig.add_axes(R["cb"])
    for ax in (axA, axB):
        make_ax(AK, ax=ax, fig=fig)
        geo_labels(ax, left=ax is axA)

    ext = hex_extent(axA, df.lon.values, df.lat.values)
    # 모든 레이어가 같은 extent·gridsize를 써서 두 지도 패널의 육각 셀이 1:1 대응한다.
    ok = df.di.notna() & df.aoa_inside.notna()      # 적용가능 영역 판정이 있는 셀

    # ---- (a) 보정된 90% 예측구간 폭 + 적용가능 영역 표기 ----
    w = df.width90_cqr.values
    hbA = hexfield(axA, df.lon, df.lat, w, ext, cmap=CMAP_WIDTH,
                   vmin=np.nanpercentile(w, 2), vmax=np.nanpercentile(w, 98))
    mask_ocean(axA)
    # p2–p98 절단 → 양끝 삼각형으로 절단 사실을 표기(과포화 방지 + 정직한 색축)
    style_colorbar(fig.colorbar(hbA, cax=caxA, extend="both", extendfrac=0.028),
                   "보정된 90% 예측구간 폭 (cm)")

    # 판정 레이어: 셀 내 '밖' 비율이 과반이면 사선 해칭, 판정 자료가 없는 셀은 회색 윤곽
    hb_ok = hexfield(axA, df.lon[ok], df.lat[ok], 1.0 - df.aoa_inside[ok].values,
                     ext, reduce=np.mean)
    hb_na = hexfield(axA, df.lon[~ok], df.lat[~ok], np.ones((~ok).sum()), ext)
    for h in (hb_ok, hb_na):
        h.set_visible(False)
    keys_ok = set(_offkeys(hb_ok))
    sel_na = np.array([k not in keys_ok for k in _offkeys(hb_na)])
    hex_overlay(axA, hb_ok, np.asarray(hb_ok.get_array(), float) > 0.5, hatch="///")
    hex_overlay(axA, hb_na, sel_na, hatch=None, color=NA_EDGE, lw=0.55)

    add_scalebar(axA, fontsize=FS_GEO)
    add_inset_locator(fig, axA, AK, size=0.22)

    # ---- (b) 환경 비유사도 지수 ----
    d = df.di[ok].values
    hbB = hexfield(axB, df.lon[ok], df.lat[ok], d, ext, cmap=CMAP_DI,
                   vmin=0.0, vmax=np.nanpercentile(d, 98))
    hex_overlay(axB, hb_na, sel_na, hatch=None, color=NA_EDGE, lw=0.55,
                facecolor=NA_FILL, zorder=1.5)
    mask_ocean(axB)
    style_colorbar(fig.colorbar(hbB, cax=caxB, extend="max", extendfrac=0.028),
                   "학습 분포와의 환경 차이 (무차원)")
    for h in (hb_ok, hb_na):
        h.remove()

    # ---- 지도 공통 범례(두 패널 아래 한 줄) ----
    fig.legend(handles=[
        Patch(facecolor="none", edgecolor=HATCH_C, hatch="///", linewidth=0.35,
              label="지역 간 전이 시 적용가능 영역 밖"),
        Patch(facecolor=NA_FILL, edgecolor=NA_EDGE, linewidth=0.55,
              label="적용가능 영역 판정 없음")],
        loc="upper center", bbox_to_anchor=(R["leg_x"], R["leg_y"]), ncol=2,
        frameon=False, fontsize=FS_LEG, handlelength=1.4, handletextpad=0.45,
        columnspacing=1.4)

    # ---- (c) 비유사도 구간별 전이 오차 (커버리지는 막대 위 값 라벨) ----
    x = np.arange(len(dec))
    rmse, cov = dec.rmse_cm.values, dec.coverage_pct.values
    axC.bar(x, rmse, width=0.48, color=SLATE, lw=0, zorder=2)
    for xi, r, c in zip(x, rmse, cov):
        axC.text(xi, r + 0.9, f"{c:.0f}%", ha="center", va="bottom",
                 fontsize=FS_NOTE, color="#5a5a5a")
    axC.text(0.012, 0.99, "막대 위 숫자 = 90% 예측구간 커버리지 (%), 목표 90%\n"
             "전이 평가 대상: 알래스카·캐나다 전 지역 셀 (지도 범위 밖 포함)",
             transform=axC.transAxes, ha="left", va="top", fontsize=FS_NOTE,
             color="#6a6a6a", linespacing=1.4)
    axC.set_ylabel("지역 간 전이 RMSE (cm)", fontsize=FS_AX)
    axC.set_xlabel("학습 분포와의 환경 차이 (구간 평균, 무차원)", fontsize=FS_AX)
    axC.set_xticks(x)
    axC.set_xticklabels([f"{v:.1f}" if v < 100 else f"{v:.0f}" for v in dec.di_mean])
    axC.set_xlim(-0.60, len(dec) - 0.40)
    axC.set_ylim(0, float(rmse.max()) * 1.50)
    axC.tick_params(labelsize=FS_TK, length=3, width=0.6)
    axC.grid(False)                       # rcParams 기본 격자(x축 포함) 해제 후 y만
    axC.grid(axis="y", alpha=0.25, lw=0.5)
    axC.set_axisbelow(True)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    for ax, lab in ((axA, "(a)"), (axB, "(b)"), (axC, "(c)")):
        ax.set_title(lab, loc="left", fontsize=FS_LAB, fontweight="bold", pad=2.5)
    return fig, (axA, axB)


def main():
    df = load_cells()
    dec = pd.read_csv(PROC / "alt_aoa_cell_transfer.csv").sort_values("di_decile")
    h = json.load(open(PROC / "s11_conformal_meta.json"))["headline"]

    n_out = int((df.aoa_inside == 0).sum())
    n_di = int(df.aoa_inside.notna().sum())
    print(f"셀 {len(df):,} · 구간 폭 중앙값 {np.median(df.width90_cqr):.1f} cm "
          f"(p5–p95 {np.percentile(df.width90_cqr, 5):.1f}–"
          f"{np.percentile(df.width90_cqr, 95):.1f})")
    print(f"보정 90% 구간 커버리지 {h['cov90_cqr']*100:.1f}% "
          f"[{h['cov90_cqr_ci'][0]*100:.1f}, {h['cov90_cqr_ci'][1]*100:.1f}] · "
          f"평균 폭 {h['width90_cqr_cm']:.1f} cm")
    print(f"적용가능 영역 밖 셀 {n_out:,}/{n_di:,} ({100*n_out/n_di:.1f}%)")

    fig, maps = build(df, dec)
    fig.canvas.draw()   # 종횡비 적용 결과 확인(계산 사각형과 실제 배치 일치 검증)
    for tag, ax in zip("ab", maps):
        p = ax.get_position()
        print(f"  패널({tag}) 배치 h={p.height:.4f} w={p.width:.4f}")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{NAME}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches=None)
    plt.close(fig)
    print("저장:", OUT / f"{NAME}.png", "·", OUT / f"{NAME}.pdf")


if __name__ == "__main__":
    main()
