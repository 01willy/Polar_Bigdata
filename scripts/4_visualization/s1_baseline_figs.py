"""S1 시각화: 실제 지도 배경 위 관측·예측·잔차 + Taylor diagram + 모델 비교.

`docs/RESEARCH_PLAN_...` §11.5. 냉색 규약, cartopy 지도 배경(위경도 지역 식별),
표<그래프<지도 우선. 여러 모델을 나란히(한 모델 단정 금지 시각화).

실행: python scripts/4_visualization/s1_baseline_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP, tnorm
from polar.geomap import (make_ax, scatter_map, hexbin_map, add_colorbar, add_scalebar,
                          add_inset_locator, add_zoom_inset, mask_ocean, ALASKA)

use_polar()
# PDF 텍스트는 TrueType(Type 42)으로 임베드 — 논문 편집기에서 글자가 벡터로 유지된다.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
PROC = C.PROCESSED
OUT = C.FIGURES / "s1_baseline"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name, dpi=300):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"[fig] {OUT.name}/{name}.png+pdf  ({w:.2f}x{h:.2f} in, dpi {dpi})")


# ---------------- 논문 게재용 공통 스타일 상수 ----------------
# 저채도·중명도 냉색 4단(진청→연청회). 채도 높은 원색과 붉은 계열을 쓰지 않는다.
NAVY, TEAL, SLATE, MIST = "#2f4b6e", "#4a7c8c", "#8296a8", "#b8c4d0"
BAR_BASE = SLATE            # 단일 계열 막대
REG_COLORS = [MIST, TEAL, NAVY]  # 지역 3분류 기준 3단(가까운 지역=옅게, 먼 지역=진하게)
LBL = "#444444"             # 값 라벨(작고 연한 회색)
TXT = "#333333"             # 보조 텍스트
GRIDC = "#c8c8c8"
# 게재 폭 정합: 보고서 본문(outputs/report/main.tex)에서 두 그림은 figure* 안의 subfigure로
# 각각 0.50\textwidth·0.47\textwidth에 배치된다. 판면 폭이 176 mm이므로 실제 게재 폭은
# 88 mm(3.46 in)·83 mm(3.26 in)이다. 종전에는 9.6 in으로 렌더해 축소율 0.36이 걸렸고,
# 렌더 시 9~10 pt였던 글자가 인쇄물에서 3 pt대로 줄어 판독이 불가능했다.
# 따라서 게재 폭과 같은 크기로 렌더하고(축소율 1.0) 글자 크기를 인쇄 기준 pt로 직접 지정한다.
FIGW_BAR = 3.46   # 0.50 x 176 mm
FIGW_MAP = 3.26   # 0.47 x 176 mm
# 인쇄 기준 폰트 위계(pt): 패널라벨 > 축라벨 > 눈금 > 범례·값라벨 > 각주
FS_PANEL, FS_PANEL_TXT = 7.2, 6.6
FS_AXLAB, FS_TICK, FS_SMALL, FS_NOTE = 6.6, 6.0, 5.6, 5.2


def panel_label(ax, letter, text=None, dy=6, gap=22, fs=None, fs_txt=None):
    """패널 식별자 (a)와 설명을 축 좌상단에 배치. 그림 내부 제목 대신 사용."""
    fs = FS_PANEL if fs is None else fs
    fs_txt = FS_PANEL_TXT if fs_txt is None else fs_txt
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=fs, fontweight="bold", color="#222222")
    if text:
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(gap, dy),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=fs_txt, color=TXT)


def panel_note(ax, text, dy=3.5, loc="left", fs=None, color=None):
    """패널 식별자 없이 설명만 축 위쪽에 배치. 이 그림이 보고서에서 subfigure (a)로
    들어가므로 그림 내부에 다시 (a)·(b)를 쓰면 라벨이 이중이 된다."""
    x, ha = (0.0, "left") if loc == "left" else (1.0, "right")
    ax.annotate(text, xy=(x, 1), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", ha=ha, va="bottom",
                fontsize=FS_PANEL_TXT if fs is None else fs,
                color=TXT if color is None else color)


def clean_axes(ax, axis="y", labelsize=None):
    """좌·하 spine만, 옅은 격자(막대 방향에 수직), 작은 눈금."""
    ax.grid(False)
    ax.grid(True, axis=axis, color=GRIDC, lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a"); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(length=2.0, width=0.6,
                   labelsize=FS_TICK if labelsize is None else labelsize, colors=LBL)


# 모델 표기: 내부 소문자 키 → 문헌 표기. 지역 표기: 내부 코드 → 평이한 지명.
MODEL_LABEL = {"mlp": "MLP", "tabm": "TabM", "catboost": "CatBoost", "xgboost": "XGBoost",
               "lightgbm": "LightGBM", "histgbm": "HistGBM", "ftt": "FT-Transformer"}
REGION_LABEL = {"ABoVE_AK": "알래스카", "ABoVE_CA": "캐나다", "Lena_RU": "레나델타"}


def mname(m):
    return MODEL_LABEL.get(m, m)


def cool_cats(n):
    """모델 범주용 냉색 이산 팔레트(붉은계열 회피). davos_r 저-고 구간을 균등 샘플."""
    from cmcrameri import cm as _cmc
    import numpy as _np
    base = _cmc.davos_r
    return [base(t) for t in _np.linspace(0.12, 0.86, max(n, 1))]


def _only():
    """--only NAME 지정 시 그 그림만 재렌더. 미지정이면 None."""
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


oof = pd.read_csv(PROC / "s1_baseline_oof.csv")
res = pd.read_csv(PROC / "s1_baseline_results.csv")
pred_cols = [c for c in oof.columns if c.startswith("pred_")]
models = [c.replace("pred_", "") for c in pred_cols]

# 지역 내 대표값의 정의: 보고서 본문 표(모델별 ALT 예측)와 같은 **3-seed 앙상블 OOF** 기준.
# s1_baseline_oof.csv의 pred_* 열은 이미 난수 3회 예측의 평균이며, 이 열로 계산한 RMSE가
# 표의 값(MLP 14.37 등)과 일치한다. 종전에는 s1_baseline_results.csv의 seed별 RMSE를 평균해
# (14.66 등) 같은 양이 표와 그림에서 다른 값으로 보였다. 앙상블 평균이 seed 노이즈를 줄이므로
# 앙상블 RMSE는 seed별 RMSE의 평균보다 낮다. 두 값은 정의가 다르다.
seed_mean = res[res.cv == "spatial_block_AK"].groupby("model").rmse_cm.mean()


def ens_rmse(m):
    """3-seed 앙상블 예측의 지역 내 OOF RMSE (cm)."""
    d = oof[["alt_cm", f"pred_{m}"]].dropna()
    return float(np.sqrt(((d[f"pred_{m}"] - d["alt_cm"]) ** 2).mean()))


rank = pd.Series({m: ens_rmse(m) for m in seed_mean.index
                  if f"pred_{m}" in oof.columns}).sort_values()
# 발산 모델(RMSE>30)은 시각화에서 제외(별도 안정화 대상). 예측 열이 없는 모델은 seed 평균으로 판정.
diverged = ([m for m in rank.index if rank[m] > 30]
            + [m for m in seed_mean.index if m not in rank.index and seed_mean[m] > 30])
ordered = [m for m in rank.index if rank[m] <= 30]
if diverged:
    print(f"[warn] 발산 모델 시각화 제외: {diverged} (별도 안정화 필요)")
print(f"[load] {len(oof):,} OOF · 모델 {ordered}")
print("[chk] 앙상블 OOF RMSE(cm): "
      + " · ".join(f"{mname(m)} {rank[m]:.2f}" for m in ordered))

ONLY = _only()  # None이면 전체 재렌더, 이름 지정 시 해당 그림만.


def want(name):
    """--only 지정 시 해당 그림만 렌더할지 여부."""
    return ONLY is None or ONLY == name

# 색 범위는 관측 분위수(p2=23.2, p98=91.1 cm)를 5 cm 단위로 정리해 잡는다. 종전 20~110 cm는
# 상단 20 cm가 비어 대부분의 점이 옅은 구간에 몰렸다. 절단분은 컬러바 삼각형(extend)으로 표기.
vmin, vmax = 25, 90


# ---------------- 1. 관측 + 상위 3모델 예측 지도 (전 셀 산점 + 밀집 사이트 확대 inset) ----------------
# 사용자 지적: 13,606 셀이 실재하나 ABoVE 250m 격자 사이트라 전 지도에서는 작은 클러스터로 뭉쳐 보인다.
#   (배로 71.25/-156.5 3,978셀, 육쿤델타 61.5/-163.0 1,669셀, 북사면 69.0/-149.0 1,594셀 등 83개
#    0.5도 지역에 초집중). 이는 버그가 아니라 데이터 구조이자 연구의 공간 대표성 한계다.
# 대응 (1) 최대 밀집지 배로를 확대 inset으로 붙여 250m 격자 개별 셀 수천 점이 촘촘히 보이게 하고,
#        본 지도에 확대 위치 사각형을 표시한다. (2) 겹침 구간에서 개별 점이 드러나도록 마커를
#        작게(s=4)·alpha 낮춰(0.35) 밀도가 짙어지게 한다. 예측 패널도 동일 방식으로 정합 유지.
# 지적: 1x4 배열은 각 패널 폭이 좁아 가로로 넓은 알래스카(경도폭 32°>위도폭 13°)를 세로로
# 찌그러뜨린다. 2x2 배열로 각 패널을 넓게 배치해 알래스카 실제 종횡비와 점 밀도를 보존한다.
from polar.geomap import _proj
from cartopy.mpl.gridliner import Gridliner
proj = _proj(ALASKA)
n_cell = len(oof)
top = ordered[:3]
# 배로 확대 범위(최대 밀집지): 실측 데이터 extent(lon -156.642~-156.538, lat 71.246~71.324)에 소폭 여백.
BARROW = (-156.652, -156.528, 71.242, 71.328)  # (lon0, lon1, lat0, lat1)
bmask = (oof.lat.between(BARROW[2], BARROW[3])) & (oof.lon.between(BARROW[0], BARROW[1]))
n_barrow = int(bmask.sum())
# 종횡비: 알래스카 extent는 Albers 등적 투영에서 x 약 1,470 km · y 약 1,440 km로 거의 1:1이다.
# 따라서 2x2 배치의 그림 전체도 정사각에 가깝게 잡아야 패널 사이 빈 여백이 생기지 않는다
# (종전 11.4x8.6 in에서는 가로 여백이 과다해 패널이 작아 보이고 행 사이 라벨이 겹쳤다).
# 괄호 안 RMSE도 보고서 표와 같은 3-seed 앙상블 OOF 기준(소수 둘째 자리까지 표기해 표와 일치).
# 패널 식별 문자는 두지 않는다. 보고서에서 이 그림 자체가 하위그림 (b)로 들어가므로
# 내부에도 (a)~(d)가 있으면 라벨이 이중이 된다. 설명만으로 패널이 구분된다.
panels = [("", "관측 ALT", oof.alt_cm)] + [
    ("", f"{mname(m)} ({rank[m]:.2f} cm)", oof[f"pred_{m}"])
    for m in top]
# 색축 절단 비율: 색 범위 밖으로 나가 컬러바 삼각형에 뭉개지는 표본의 비율. 리뷰 지적대로
# 그림 안에서 정량 표기해야 색 범위 선택이 결과 해석을 왜곡하지 않음을 확인할 수 있다.
clip_lo = 100 * float((oof.alt_cm < vmin).mean())
clip_hi = 100 * float((oof.alt_cm > vmax).mean())
pred_clip = max(100 * float(((oof[f"pred_{m}"] < vmin) | (oof[f"pred_{m}"] > vmax)).mean())
                for m in top)
if want("alaska_obs_vs_pred_maps"):
    # 높이는 패널 종횡비(Albers 알래스카 ≈ 1.27:1)에 맞춰 잡는다. 여유 높이가 남으면
    # cartopy가 축을 세로 중앙에 정렬해 행 사이에 빈 띠가 생긴다.
    fig = plt.figure(figsize=(FIGW_MAP, 2.62))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.845, bottom=0.165, top=0.930,
                          hspace=0.19, wspace=0.07)
    sc = None
    for i, (letter, title, vals) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection=proj)
        make_ax(ALASKA, ax=ax, fig=fig)
        # 경위도 라벨은 바깥 테두리(좌열·하행)에만 둔다. 네 패널이 같은 범위·같은 투영이므로
        # 중복 라벨은 정보를 더하지 않고 행 사이 간격만 벌린다. 단 폭 실크기 렌더이므로
        # 눈금은 3개씩으로 줄이고 라벨은 인쇄 기준 5.2 pt로 둔다.
        for gl in [a for a in ax.artists if isinstance(a, Gridliner)]:
            gl.rotate_labels = False
            gl.bottom_labels = i >= 2
            gl.left_labels = i % 2 == 0
            gl.xlocator = mticker.FixedLocator([-160, -150, -140])
            gl.ylocator = mticker.FixedLocator([60, 65, 70])
            gl.xlabel_style = gl.ylabel_style = {"size": FS_NOTE, "color": "0.35"}
        panel_label(ax, letter, title, dy=2.0, gap=11, fs=6.6, fs_txt=6.0)
        sc = scatter_map(ax, oof.lon, oof.lat, vals, cmap=CMAP.alt, vmin=vmin, vmax=vmax,
                         s=1.6, edge=False)
        sc.set_alpha(0.5)
        sc.set_rasterized(True)  # 점 6.8만 개는 래스터로, 글자·축은 벡터로 유지(PDF 경량화)
        # 밀집 사이트 확대: 250m 격자 개별 셀이 촘촘히 드러나게(작은 마커·본지도 위치 표시).
        # 확대 범위 종횡비(약 4.4 km x 9.5 km)에 맞춰 세로로 긴 상자를 잡는다.
        # 확대창 소제목은 단 폭에서 상자보다 넓어져 넣지 않고, 각주 한 줄로 설명한다.
        axz = add_zoom_inset(fig, ax, ALASKA, BARROW, loc=(0.745, 0.06, 0.19, 0.46),
                             edgecolor=SLATE, lw=0.45, ms=3.2)
        if axz is not None:
            scz = scatter_map(axz, oof.lon[bmask], oof.lat[bmask], vals[bmask],
                              cmap=CMAP.alt, vmin=vmin, vmax=vmax, s=2.2, edge=False)
            scz.set_alpha(0.6)
            scz.set_rasterized(True)
        if i == 0:
            add_inset_locator(fig, ax, ALASKA, size=0.30)
            add_scalebar(ax, loc="lower left", fontsize=FS_NOTE)
    cax = fig.add_axes([0.866, 0.21, 0.021, 0.56])
    cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.set_label("ALT (cm)", fontsize=FS_AXLAB, color=TXT, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.8, width=0.5, colors=LBL, pad=1.5)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor("#9a9a9a")
    cb.solids.set_alpha(1.0)
    # 각주: 색축 절단 비율(리뷰 지적)·확대창 표본 수·괄호 수치의 정의. 그림만으로 확인 가능해야 한다.
    fig.text(0.012, 0.068,
             f"색 범위 {vmin}–{vmax} cm. 관측의 {clip_lo:.1f}%가 하한 미만, "
             f"{clip_hi:.1f}%가 상한 초과(컬러바 삼각형).",
             fontsize=FS_NOTE, color="#666666", ha="left", va="bottom")
    pred_note = "예측 3종은 절단 없음" if pred_clip < 0.05 else f"예측 절단 {pred_clip:.1f}%"
    fig.text(0.012, 0.010,
             f"{pred_note}. 확대창 = 배로 250 m 격자 {n_barrow:,}셀. 괄호 안은 지역 내 RMSE.",
             fontsize=FS_NOTE, color="#666666", ha="left", va="bottom")
    save(fig, "alaska_obs_vs_pred_maps", dpi=600)

# figs-only 분기: alaska_obs_vs_pred_maps만 재렌더(다른 그림 미변경).
if "--only-maps" in sys.argv:
    print("[done] alaska_obs_vs_pred_maps만 재렌더")
    sys.exit(0)


# ---------------- 2. 모델별 잔차 지도 (hexbin, broc 0중심) ----------------
if want("alaska_residual_maps"):
    n = len(ordered)
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(3.4 * ncol, 3.4 * nrow))
    for i, m in enumerate(ordered):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection=proj)
        make_ax(ALASKA, ax=ax, fig=fig, title=f"{m}  (bias {(oof[f'pred_{m}']-oof.alt_cm).mean():+.1f})")
        resid = oof[f"pred_{m}"] - oof.alt_cm
        hb = hexbin_map(ax, oof.lon, oof.lat, resid, gridsize=38, cmap=CMAP.diff, norm=tnorm(-40, 40, 0))
        mask_ocean(ax)
    cb = fig.colorbar(hb, ax=fig.axes, fraction=0.012, pad=0.02)
    cb.set_label("예측 - 관측 (cm)", fontsize=9)
    fig.suptitle("S1 모델별 잔차 지도 (hexbin 중앙값; 파랑=과소, 갈색=과대)", fontsize=12, y=1.0)
    save(fig, "alaska_residual_maps")


# ---------------- 3. Taylor diagram (in-domain) ----------------
if want("taylor_indomain"):
    obs = oof.alt_cm.values
    sd_obs = obs.std()
    fig = plt.figure(figsize=(6.4, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    catcol = cool_cats(len(ordered))
    for i, m in enumerate(ordered):
        p = oof[f"pred_{m}"].values
        r = np.corrcoef(obs, p)[0, 1]
        sd = p.std()
        theta = np.arccos(np.clip(r, -1, 1))
        ax.plot(theta, sd, "o", ms=10, color=catcol[i], markeredgecolor="0.3",
                markeredgewidth=0.5, label=f"{m} (r={r:.2f})")
    ax.plot(0, sd_obs, "k*", ms=16, label="관측")
    # RMSE 반원 등고선(관측 기준)
    for rms in [8, 12, 16, 20]:
        th = np.linspace(0, np.pi / 2, 100)
        xr = sd_obs + rms * np.cos(th + np.pi)  # 근사
    ax.set_rlabel_position(90)
    ax.set_xlabel("표준편차 (cm)")
    ax.set_title("Taylor 다이어그램: 알래스카 in-domain\n(관측 별표에 가까울수록 우수)", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=8)
    save(fig, "taylor_indomain")


# ---------------- 4. 모델 비교 막대 (in-domain RMSE + LORO) ----------------
if want("model_comparison_bars"):
    # 배치: 단 폭(88 mm) 실크기 렌더이므로 1x2 가로 배열은 패널당 44 mm가 되어 모델 7종의
    # 이름과 21개 막대를 담을 수 없다. 2행 1열로 바꾸고 두 패널 모두 가로 방향으로 두어
    # 모델명을 회전 없이 읽게 한다. 두 패널의 모델 순서는 지역 내 RMSE 오름차순으로 공유한다.
    # 지역 내 대표값 = 3-seed 앙상블 OOF RMSE(rank). 난수별 RMSE는 산포 표기에만 쓴다.
    ind = pd.DataFrame({"ens": [rank[m] for m in ordered]}, index=ordered)
    sd_g = res[res.cv == "spatial_block_AK"].groupby("model").rmse_cm
    ind["lo"] = sd_g.min().reindex(ordered)
    ind["hi"] = sd_g.max().reindex(ordered)
    ind["size"] = sd_g.size().reindex(ordered)
    g = res[res.cv == "LORO"].groupby(["model", "region"]).rmse_cm
    lo_m = g.mean().unstack().reindex(ordered)
    lo_lo = g.min().unstack().reindex(ordered)
    lo_hi = g.max().unstack().reindex(ordered)
    lo_n = g.size().unstack().reindex(ordered)
    cols = [c for c in ["ABoVE_AK", "ABoVE_CA", "Lena_RU"] if c in lo_m.columns]
    cols += [c for c in lo_m.columns if c not in cols]
    y = np.arange(len(ind))
    names = [mname(m) for m in ind.index]

    # 좌우 여백: 좌측은 모델명(최장 "FT-Transformer", 6.0 pt) 폭까지만 남기고, 우측은 값
    # 라벨이 잘리지 않는 최소치만 남긴다(보고서에서 지도 그림과 나란히 놓이므로 축소율 정합).
    # 지역 간 전이 패널은 매크로 지역 병합 이전 산출(알래스카 일부 셀이 학습에 포함)이라
    # 본문 전이 표와 정합하지 않는다. 전이 비교는 별도 그림·표가 담당하므로 여기서는
    # 지역 내 비교만 싣는다(DRAW_TRANSFER=False).
    DRAW_TRANSFER = False
    nrow = 2 if DRAW_TRANSFER else 1
    fig = plt.figure(figsize=(FIGW_BAR, 3.18 if DRAW_TRANSFER else 2.02))
    gs = fig.add_gridspec(nrow, 1, left=0.178, right=0.998,
                          bottom=0.175 if DRAW_TRANSFER else 0.285,
                          top=0.912, hspace=0.60)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0]) if DRAW_TRANSFER else None
    for ax in ([ax0, ax1] if DRAW_TRANSFER else [ax0]):
        clean_axes(ax, axis="x")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=FS_TICK)
        ax.tick_params(axis="y", pad=1.5)
        ax.set_ylim(len(y) - 0.5, -0.5)          # 성능 상위 모델이 위
        ax.set_xlabel("RMSE (cm)", fontsize=FS_AXLAB, color=TXT, labelpad=1.5)

    # 위 패널 · 알래스카 지역 내: 막대 = 앙상블 RMSE, 가는 선 = 난수별 RMSE 범위(최소–최대).
    # 앙상블 RMSE는 난수별 RMSE 범위 밖(대개 아래)에 놓일 수 있으므로 오차막대가 아니라
    # 별도의 산포 표식으로 그린다. 오차막대로 그리면 막대값의 불확도로 오독된다.
    ax0.barh(y, ind["ens"], height=0.6, color=BAR_BASE, edgecolor="none", zorder=2)
    lo0, hi0 = ind["lo"].values, ind["hi"].values
    ax0.hlines(y, lo0, hi0, color="#5f5f5f", lw=0.6, zorder=4)
    ax0.scatter(np.concatenate([lo0, hi0]), np.tile(y, 2), marker="|", s=7,
                color="#5f5f5f", linewidths=0.6, zorder=4)
    # 값 라벨이 축 밖으로 나가지 않을 만큼만 우측 여유를 둔다(1.14: 라벨 폭 약 7% + 간격 1.4%).
    xmax0 = float(np.nanmax(np.maximum(ind["ens"].values, hi0))) * 1.14
    for yi, v, h in zip(y, ind["ens"].values, hi0):
        ax0.text(max(v, h) + xmax0 * 0.014, yi, f"{v:.2f}", va="center", ha="left",
                 fontsize=FS_SMALL, color=LBL)
    ax0.set_xlim(0, xmax0)
    # 관측 대표성 하한: 본문은 단일 탐침 정밀도(약 ±3 cm)와 셀 내 자연 공간변동을 결합한
    # 10–12 cm를 쓴다. 종전의 단일 14 cm 파선은 본문과 불일치했다.
    # 모든 막대가 이 구간을 지나므로 음영을 막대 뒤에 두면 막대 사이 틈에서만 보인다.
    # 옅은 음영을 막대 위에 얹고 경계에 파선을 넣어 구간이 한눈에 읽히게 한다.
    ax0.axvspan(10, 12, color="#33556f", alpha=0.16, lw=0, zorder=3)
    for xb in (10, 12):
        ax0.axvline(xb, color="#5b7286", ls=(0, (3, 2)), lw=0.55, alpha=0.85, zorder=3.5)
    # 본문 표는 물리식 단독과 평균 예측을 참조로 함께 싣는다. 그림에 없으면 표와
    # 대조할 때 두 기준이 사라지므로, 세로 기준선으로 같은 두 값을 표시한다.
    _ref = []
    _p = pd.read_csv(Path(PROC) / "s2_physics_results.csv")
    _s = _p[(_p.cv == "spatial_block_AK") & (_p.model == "p1_stefan")].rmse_cm
    if len(_s):
        _ref.append((float(_s.iloc[0]), "물리식 단독(Stefan)", "#1f4e79", (0, (5, 2))))
    _yobs = oof.alt_cm.values.astype(float)
    _ref.append((float(np.sqrt(np.mean((_yobs - _yobs.mean()) ** 2))),
                 "평균 예측(참조 기준선)", "#8a94a0", (0, (1.6, 1.8))))
    _refnote = []
    for _v, _lab, _c, _ls in _ref:
        ax0.axvline(_v, color=_c, ls=_ls, lw=0.9, zorder=3.6)
        _refnote.append(f"{_lab} {_v:.2f} cm")
    panel_note(ax0, "알래스카 지역 내")
    panel_note(ax0, "음영 = 지점 관측 대표성 하한 10–12 cm", loc="right",
               fs=FS_SMALL, color="#6f6f6f")

    if DRAW_TRANSFER:
        # 아래 패널 · 지역 간 전이: 평가 지역 3분류를 같은 행에 점으로 겹쳐 배치(막대 21개는
        #     단 폭에서 판독 불가). 색과 표식 모양을 함께 써서 색 단독 부호화를 피한다.
        #     전이에는 앙상블 예측 파일이 없으므로 난수별 RMSE의 평균을 점으로, 범위를 선으로 쓴다.
        MARKS = ["o", "s", "^"]
        DOTC = ["#9aabbb", TEAL, NAVY]   # 흰 배경에서 가장 옅은 단계는 조금 진하게
        off = 0.26
        for j, c in enumerate(cols):
            m_, l_, h_ = lo_m[c].values, lo_lo[c].values, lo_hi[c].values
            err = np.vstack([np.nan_to_num(m_ - l_), np.nan_to_num(h_ - m_)])
            ax1.errorbar(m_, y + (j - (len(cols) - 1) / 2) * off, xerr=err,
                         fmt=MARKS[j % len(MARKS)], ms=2.6, mew=0, ls="none",
                         color=DOTC[j % len(DOTC)], ecolor=DOTC[j % len(DOTC)], elinewidth=0.6,
                         capsize=1.3, capthick=0.6, label=REGION_LABEL.get(c, c))
        ax1.set_xlim(0, float(np.nanmax(lo_hi.values)) * 1.08)
        panel_note(ax1, "지역 간 전이")
        leg = ax1.legend(fontsize=FS_SMALL, frameon=False, loc="lower right",
                         bbox_to_anchor=(1.005, 0.99), ncol=3, handlelength=0.9,
                         handletextpad=0.25, columnspacing=0.9, borderaxespad=0.0)
        for t in leg.get_texts():
            t.set_color(TXT)

    # 각주: 대표값의 정의(앙상블 RMSE)와 산포 표식의 정의를 그림 안에서 밝힌다.
    # 난수 1회 모델 안내는 전이 패널에만 해당하므로 그 패널을 그릴 때만 붙인다.
    n_rep = int(np.nanmax(ind["size"].values))
    det = [mname(m) for m in ordered if int(lo_n.loc[m].max()) == 1]
    note2 = bool(det) and DRAW_TRANSFER
    fig.text(0.006, 0.048 if note2 else 0.004,
             (f"지역 내 = 난수 {n_rep}회 예측을 평균한 앙상블의 RMSE, "
              f"전이 = 난수별 RMSE의 평균." if DRAW_TRANSFER else
              f"막대 = 난수 {n_rep}회 예측을 평균한 앙상블의 RMSE, "
              f"가는 선 = 난수별 RMSE의 범위.\n"
              f"세로선 = " + ", ".join(_refnote) + "."),
             fontsize=FS_NOTE, color="#666666", ha="left", va="bottom")
    if note2:
        fig.text(0.006, 0.004,
                 f"가는 선 = 난수별 RMSE 범위. 전이에서 난수 1회 모델"
                 f"({'·'.join(det)})은 범위가 없다.",
                 fontsize=FS_NOTE, color="#666666", ha="left", va="bottom")
    save(fig, "model_comparison_bars", dpi=600)

print("[done] S1 시각화 완료")
