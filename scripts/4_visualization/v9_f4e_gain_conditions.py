"""v9 F4e 잔차 결합 이득의 조건(설계 지침 v9 2.3 항 18): 공변량별 지역 안 3분위에서 ΔRMSE = RMSE(물리 + 잔차 ML) − RMSE(Stefan 재보정).

자료: data/processed/v5/runs/core_v6/preds 의 블록 CV OOF 예측(주 설정: scheme block, forcing year_obs, featset F8TS, R1 catboost λ 0.5,
      P1 '-' λ −1), 씨앗 평균. 공변량은 polar.testbed_v5.load_v5 의 같은 cell_year_id 행에서 읽는다(OOF 파일의 loc_id·year 와 일치 확인).
공변량(행): TDD(관측 연도 전 계절 융해 도일, y_e5_tdd; conditions_tdd.csv 와 같은 정의) · FDD(전겨울 10–4월 동결 도일) · SWE(수문년 평균 적설 수당량)
      · SOC 0–5 cm(SoilGrids) · 점토 5–15 cm(SoilGrids) · 경사 · 고도(DEM) · NDVI(MODIS 여름 NDVI 다년 평균, 채움값 −3000 은 결측).
3분위: 지역 안 셀-연도의 33.3·66.7 백분위(np.digitize, conditions_tdd.csv 와 같은 규칙). 공변량 결측 행은 그 행에서만 빠진다.
구간: polar.testbed_v5.boot_delta(블록 재추출, 2,000회, seed 0) 의 2.5–97.5 백분위. 표시 조건: 셀-연도 ≥ 30 그리고 블록 ≥ 5(아니면 n/a).
출력: outputs/figures/paper/v7/source_data/F4e_gain_conditions.csv(+ .json 정의), 덱 PNG outputs/figures/paper/v7/deck/F4_results_core__deck_R03e.png
      (12.0 × 4.65 in, 300 dpi, RGB). 열지도: 행 = 공변량(기후 · 토양 · 지형 · 식생 묶음 사이 간격), 열 = 지역별 low · mid · high,
      색 = broc 0 중심 대칭(토큰 colormaps.difference_diverging), 구간이 0 을 제외한 칸은 진한 외곽선, 값은 칸 안에 적는다.
실행: cd /home/willy010313/Polar_Bigdata && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v9_f4e_gain_conditions.py
      [--recompute]   (CPU 만, 학습 없음. 기존 OOF 예측 집계)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

PROC = ROOT / "data" / "processed" / "v5"; PREDS = PROC / "runs" / "core_v6" / "preds"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; DECK_OUT = OUT / "deck"; SRC = OUT / "source_data"
CSV = SRC / "F4e_gain_conditions.csv"
REG = ["Alaska", "Lena", "Canada"]
NBOOT, SEED, MIN_N, MIN_BLOCKS = 2000, 0, 30, 5
Q = (33.3, 66.7)                                   # conditions_tdd.csv(v5_aggregate.conditions)와 같은 3분위 경계
TERC = ["low", "mid", "high"]
# 키, 열, 그림 라벨, 묶음, 정의(CSV·JSON)
COVS = [("tdd", "y_e5_tdd", "TDD", "climate", "thaw degree-days of the observation year, full season (ERA5-Land, degC d)"),
        ("fdd", "y_e5_fdd", "FDD", "climate", "freezing degree-days of the preceding winter, Oct-Apr (ERA5-Land, degC d)"),
        ("swe", "y_e5_swe", "Snow (SWE)", "climate", "mean snow water equivalent of the hydrological year, Oct-Sep (ERA5-Land, m)"),
        ("soc", "sg_soc_0_5", "SOC 0–5 cm", "soil", "SoilGrids soil organic carbon 0-5 cm (value as stored in static_v5)"),
        ("clay", "sg_clay_5_15", "Clay", "soil", "SoilGrids clay 5-15 cm (value as stored in static_v5)"),
        ("slope", "dem_slope", "Slope", "terrain", "DEM slope (deg)"),
        ("elev", "dem_elev", "Elevation", "terrain", "DEM elevation (m)"),
        ("ndvi", "c_ndvi_summer", "NDVI", "vegetation", "MODIS summer NDVI, multi-year mean per cell (fill value -3000 treated as missing)")]
PRED = "{r}__block__year_obs__F8TS__{m}__{l}__lam{lam}.csv.gz"


# ---------------------------------------------------------------- 분석
def oof(r, m, learner, lam):
    d = pd.read_csv(PREDS / PRED.format(r=r, m=m, l=learner, lam=lam), usecols=["cell_year_id", "loc_id", "year", "block", "y", "pred"])
    return d.groupby("cell_year_id").agg(loc_id=("loc_id", "first"), year=("year", "first"), block=("block", "first"), y=("y", "first"), pred=("pred", "mean"))


def compute():
    import polar.testbed_v5 as T
    base = T.load_v5(PROC, products=False).set_index("cell_year_id")
    base.loc[base.c_ndvi_summer < -1, "c_ndvi_summer"] = np.nan
    rows = []
    for r in REG:
        a = oof(r, "R1", "catboost", "0.5"); b = oof(r, "P1", "-", "-1")
        d = a.join(b[["pred"]].rename(columns={"pred": "pred_p1"}), how="inner").join(base[["loc_id", "year"] + [c for _, c, *_ in COVS]], rsuffix="_b")
        assert (d.loc_id == d.loc_id_b).all() and (d.year == d.year_b).all(), "cell_year_id 불일치"
        y, p1, r1, bl = d.y.values, d.pred_p1.values, d.pred.values, d.block.values
        bd = T.boot_delta(y, r1, p1, bl, NBOOT, SEED)
        rows.append(dict(region=r, covariate="all", label="all cell-years", group="", tercile=-1, tercile_name="all", x_lo=np.nan, x_hi=np.nan,
                         n=bd["n"], n_blocks=bd["n_blocks"], rmse_stefan=T.rmse(y, p1), rmse_r1=T.rmse(y, r1), delta=bd["delta"], lo=bd["lo"], hi=bd["hi"]))
        for key, col, lab, grp, _ in COVS:
            x = d[col].values.astype(float); ok = np.isfinite(x)
            q = np.nanpercentile(x[ok], Q); bins = np.where(ok, np.digitize(x, q), -1)
            for k in range(3):
                m = bins == k; nb = int(len(np.unique(bl[m]))) if m.any() else 0
                row = dict(region=r, covariate=key, label=lab, group=grp, tercile=k, tercile_name=TERC[k], x_lo=float(np.min(x[m])) if m.any() else np.nan,
                           x_hi=float(np.max(x[m])) if m.any() else np.nan, n=int(m.sum()), n_blocks=nb)
                if m.sum() >= MIN_N and nb >= 2:
                    bd = T.boot_delta(y[m], r1[m], p1[m], bl[m], NBOOT, SEED)
                    row.update(rmse_stefan=T.rmse(y[m], p1[m]), rmse_r1=T.rmse(y[m], r1[m]), delta=bd["delta"], lo=bd["lo"], hi=bd["hi"])
                rows.append(row)
    out = pd.DataFrame(rows)
    out["shown"] = (out.n >= MIN_N) & (out.n_blocks >= MIN_BLOCKS) & out.delta.notna()
    out["excludes_0"] = out.shown & ((out.hi < 0) | (out.lo > 0))
    SRC.mkdir(parents=True, exist_ok=True); out.to_csv(CSV, index=False, float_format="%.4f")
    meta = dict(script="scripts/4_visualization/v9_f4e_gain_conditions.py", csv=str(CSV.relative_to(ROOT)),
                quantity="delta = RMSE(Physics + residual ML) - RMSE(Stefan, recalibrated) in cm; negative = residual ML better",
                predictions="data/processed/v5/runs/core_v6/preds: block CV out-of-fold, forcing year_obs, featset F8TS; R1 catboost lam 0.5 (3 seeds averaged), P1",
                terciles=f"within-region percentiles {Q} of each covariate over labelled cell-years (np.digitize; ties go to the upper tercile)",
                interval=f"polar.testbed_v5.boot_delta, block resampling, {NBOOT} resamples, seed {SEED}, 2.5-97.5 percentiles",
                shown_rule=f"n >= {MIN_N} cell-years and >= {MIN_BLOCKS} spatial blocks in the tercile", covariates={k: dict(column=c, label=l, group=g, definition=t) for k, c, l, g, t in COVS})
    CSV.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return out


def load_table(recompute=False):
    return compute() if (recompute or not CSV.exists()) else pd.read_csv(CSV)


def summary(t):
    """주 패턴 숫자: 지역별 전체 Δ, 0 을 제외한 칸, Stefan RMSE 와 Δ 의 상관."""
    print("region-wide:", "; ".join(f"{r} {q.delta:+.2f} [{q.lo:+.2f}, {q.hi:+.2f}]" for r, q in t[t.covariate == "all"].set_index("region").iterrows()))
    s = t[(t.covariate != "all")]
    for r in REG:
        q = s[s.region == r]; sh = q[q.shown]; ex = sh[sh.excludes_0]
        print(f"{r}: shown {len(sh)}/{len(q)}, interval excludes 0 in {len(ex)} (negative {int((ex.delta < 0).sum())}), "
              f"Δ range {sh.delta.min():+.2f} … {sh.delta.max():+.2f}, r(Δ, Stefan RMSE) = {np.corrcoef(sh.delta, sh.rmse_stefan)[0, 1]:+.2f}")
        for _, z in ex.sort_values("delta").iterrows():
            print(f"    {z.label:11s} {z.tercile_name:4s} Δ {z.delta:+.2f} [{z.lo:+.2f}, {z.hi:+.2f}]  n {z.n:5d}  blocks {z.n_blocks:2d}  Stefan RMSE {z.rmse_stefan:5.1f}")
        na = q[~q.shown]
        if len(na):
            print("    n/a:", ", ".join(f"{z.label} {z.tercile_name} (n {z.n}, blocks {z.n_blocks})" for _, z in na.iterrows()))


# ---------------------------------------------------------------- 그림(덱)
def cmap_diff():
    from cmcrameri import cm as cmc       # 토큰 colormaps.difference_diverging: 남색–흰–올리브 0 중심(지도 차이 색표 broc 와 같다)
    return cmc.broc


def fmt_delta(v):
    s = f"{abs(v):.1f}"
    return ("−" if v < 0 and s != "0.0" else "+" if v > 0 and s != "0.0" else "") + s


HATCH = dict(face="white", line=S.NEUTRAL["coast"], pattern="////")   # 자료 부족 칸: 흰 바탕 + 옅은 회색 빗금(R03m)


def hatch_rect(ax, x, y, w, h, zorder=1.5):
    """자료 부족 칸(또는 열쇠 견본): 흰 바탕에 옅은 회색 빗금. 빗금 색은 edgecolor, 테두리 선은 그리지 않는다(lw 0)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=HATCH["face"], edgecolor=HATCH["line"], hatch=HATCH["pattern"], lw=0, zorder=zorder))


def draw_map(fig, t, x0, y0, w_cell, h_cell, gap_grp, gap_reg, fs_val, fs_tick, fs_reg, lw_sig, vmax, na_style="text", fs_row=None, tercile_header=None):
    """열지도를 그림 좌표(in)에 그린다. 행 = COVS(묶음 사이 gap_grp), 열 = 지역 × 3분위(지역 사이 gap_reg). 반환: (축 목록, norm, 높이).
    na_style: 'text'(R03e: 회색 칸 + 'n/a') 또는 'hatch'(R03m: 흰 바탕 옅은 회색 빗금, 글자 없음). fs_row: 행 이름 크기(기본 fs_tick + 1).
    tercile_header: 3분위 이름 줄 왼쪽(행 이름 열)에 둘 글자(없으면 생략)."""
    fw, fh = fig.get_size_inches(); cm = cmap_diff(); norm = TwoSlopeNorm(0.0, -vmax, vmax)
    groups = [g for _, _, _, g, _ in COVS]; ys, yy = [], y0
    for i in range(len(COVS) - 1, -1, -1):              # 아래에서 위로 행 y(in)
        ys.append(yy); yy += h_cell + (gap_grp if i > 0 and groups[i] != groups[i - 1] else 0.0)
    ys = ys[::-1]; H = yy - y0
    axes = []
    for j, r in enumerate(REG):
        xa = x0 + j * (3 * w_cell + gap_reg)
        ax = fig.add_axes([xa / fw, y0 / fh, 3 * w_cell / fw, H / fh]); axes.append(ax)
        ax.set_xlim(0, 3 * w_cell); ax.set_ylim(0, H); ax.axis("off")
        q = t[t.region == r]
        for i, (key, _, lab, _, _) in enumerate(COVS):
            yb = ys[i] - y0
            for k in range(3):
                z = q[(q.covariate == key) & (q.tercile == k)].iloc[0]; xb = k * w_cell
                if bool(z.shown):
                    v = float(z.delta); c = cm(norm(v))
                    ax.add_patch(Rectangle((xb, yb), w_cell, h_cell, facecolor=c, edgecolor="white", lw=1.2, zorder=1))
                    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                    ax.text(xb + w_cell / 2, yb + h_cell / 2, fmt_delta(v), ha="center", va="center", fontsize=fs_val,
                            color="white" if lum < 0.45 else S.NEUTRAL["text_slide"], zorder=3)
                    if bool(z.excludes_0):
                        ins = lw_sig / 72 / 2 + 0.012
                        ax.add_patch(Rectangle((xb + ins, yb + ins), w_cell - 2 * ins, h_cell - 2 * ins, facecolor="none", edgecolor=S.NEUTRAL["text_slide"],
                                               lw=lw_sig, zorder=2, joinstyle="miter"))
                elif na_style == "hatch":
                    e = 1.2 / 72 / 2                        # 이웃 칸 사이 흰 선(1.2 pt)과 같은 여백
                    hatch_rect(ax, xb + e, yb + e, w_cell - 2 * e, h_cell - 2 * e)
                else:
                    ax.add_patch(Rectangle((xb, yb), w_cell, h_cell, facecolor=S.NEUTRAL["equiv_band"], edgecolor="white", lw=1.2, zorder=1))
                    ax.text(xb + w_cell / 2, yb + h_cell / 2, "n/a", ha="center", va="center", fontsize=fs_val, color=S.NEUTRAL["text_aux"], zorder=3)
        for k in range(3):                               # 3분위 이름(블록 아래)
            ax.text((k + 0.5) * w_cell, -0.10, TERC[k], ha="center", va="top", fontsize=fs_tick, color=S.NEUTRAL["text_slide"], clip_on=False)
        ax.text(1.5 * w_cell, H + 0.10, S.REGION_EN[r], ha="center", va="bottom", fontsize=fs_reg, color=S.REGION_HEX[r], clip_on=False)
    ax0 = axes[0]; fs_row = fs_tick + 1 if fs_row is None else fs_row
    for i, (_, _, lab, _, _) in enumerate(COVS):         # 행 이름(왼쪽)
        ax0.text(-0.12, ys[i] - y0 + h_cell / 2, lab, ha="right", va="center", fontsize=fs_row, color=S.NEUTRAL["text_slide"], clip_on=False)
    if tercile_header:
        ax0.text(-0.12, -0.10, tercile_header, ha="right", va="top", fontsize=fs_tick, color=S.NEUTRAL["text_aux"], clip_on=False)
    return axes, norm, H


def text_w_in(fig, s, fs):
    """글자 폭(in) 측정."""
    t = fig.text(0, 0, s, fontsize=fs); fig.canvas.draw(); w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi; t.remove()
    return w


def deck(t):
    """R03e: 12.0 × 4.65 in. 열지도(왼쪽 행 이름, 지역 블록 셋) + 오른쪽 색 막대, 그 아래 외곽선 열쇠(칸 견본 + 두 줄 글자)."""
    S.setup(scale=1.69)
    Fs, Ls = S.F["slide"], S.TOK["lines"]["slide"]
    plt.rcParams.update({"font.size": Fs["direct_label_pt"], "axes.labelsize": Fs["axis_label_pt"], "xtick.labelsize": Fs["tick_pt"], "ytick.labelsize": Fs["tick_pt"],
                         "axes.linewidth": Ls["axis_pt"], "xtick.major.width": Ls["tick_pt"], "ytick.major.width": Ls["tick_pt"], "xtick.major.size": Ls["tick_len_pt"],
                         "ytick.major.size": Ls["tick_len_pt"]})
    W, Hd = 12.0, 4.65; fs_row = Fs["tick_pt"] + 1
    fig = S.figure(W * 25.4, Hd * 25.4); fig.set_size_inches(W + 1e-9, Hd + 1e-9)
    sh = t[t.shown & (t.covariate != "all")]; vmax = float(np.ceil(np.nanmax(np.abs(sh.delta.values))))
    lab_w = max(text_w_in(fig, lab, fs_row) for _, _, lab, _, _ in COVS)
    x0 = 0.10 + lab_w + 0.14                                      # 행 이름 오른쪽 끝 = 열지도 왼쪽 − 0.12 in
    y0, gap_reg, gap_grp, right = 0.80, 0.40, 0.10, 1.76           # right: 색 막대·열쇠 몫(in)
    w_cell = (W - x0 - 2 * gap_reg - right) / 9
    h_cell = (Hd - y0 - 0.46 - 3 * gap_grp) / len(COVS)
    axes, norm, H = draw_map(fig, t, x0, y0, w_cell, h_cell, gap_grp, gap_reg, fs_val=fs_row, fs_tick=Fs["tick_pt"], fs_reg=Fs["axis_label_pt"],
                             lw_sig=2.4, vmax=vmax)
    xr = x0 + 9 * w_cell + 2 * gap_reg                            # 열지도 오른쪽 끝(in)
    fig.text((x0 + (xr - x0) / 2) / W, 0.06 / Hd, "Tercile within region", ha="center", va="bottom", fontsize=Fs["axis_label_pt"])
    cbx, cbw = xr + 0.44, 0.16; cbh = 0.60 * H; cby = y0 + H - cbh   # 색 막대: 열지도 위쪽 60 %
    cax = fig.add_axes([cbx / W, cby / Hd, cbw / W, cbh / Hd])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_diff()); cb = fig.colorbar(sm, cax=cax)
    st = 3 if vmax >= 6 else 2; tk = np.arange(-np.floor(vmax / st) * st, vmax + 1e-9, st)
    cb.set_ticks(tk); cb.set_ticklabels([("−" if v < 0 else "+" if v > 0 else "") + f"{abs(v):.0f}" for v in tk])
    cb.ax.tick_params(labelsize=Fs["tick_pt"], length=Ls["tick_len_pt"], width=Ls["tick_pt"]); cb.ax.minorticks_off()
    cb.outline.set_linewidth(Ls["axis_pt"]); cb.set_label("ΔRMSE (cm)", fontsize=Fs["axis_label_pt"], labelpad=6)
    kw, kh = 0.30, 0.24; ky = y0 + h_cell / 2 - kh / 2           # 열쇠: 맨 아래 행 높이에 칸 견본 + 오른쪽 두 줄 글자
    kax = fig.add_axes([cbx / W, ky / Hd, kw / W, kh / Hd]); kax.set_xlim(0, kw); kax.set_ylim(0, kh); kax.axis("off")
    kax.add_patch(Rectangle((0.02, 0.02), kw - 0.04, kh - 0.04, facecolor=cmap_diff()(norm(-0.45 * vmax)), edgecolor=S.NEUTRAL["text_slide"], lw=2.4))
    kax.text(kw + 0.08, kh / 2, "95% CI\nexcludes 0", ha="left", va="center", fontsize=Fs["tick_pt"], color=S.NEUTRAL["text_slide"], linespacing=1.05,
             clip_on=False)
    print(f"R03e layout: row-label width {lab_w:.2f} in, x0 {x0:.2f}, cell {w_cell:.2f} × {h_cell:.2f} in, map right {xr:.2f} in, vmax {vmax:.0f}")
    return save(fig, "R03e", Hd)


def save(fig, slide, h_in):
    from PIL import Image
    DECK_OUT.mkdir(parents=True, exist_ok=True); p = DECK_OUT / f"F4_results_core__deck_{slide}.png"
    fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer()); fw, fh = fig.get_size_inches()
    fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    with Image.open(p) as im:
        rgb = im.convert("RGB")
    rgb.save(p, dpi=(300, 300)); want = (int(round(12.0 * 300)), int(round(h_in * 300)))
    print(f"{slide}: {rgb.size[0]}x{rgb.size[1]} px (want {want[0]}x{want[1]}) | ink margins in: left {bb.x0:.2f} right {fw - bb.x1:.2f} "
          f"bottom {bb.y0:.2f} top {fh - bb.y1:.2f}")
    assert rgb.size == want, (rgb.size, want)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--recompute", action="store_true", help="OOF 예측에서 표를 다시 계산한다"); a = ap.parse_args()
    t = load_table(a.recompute); summary(t)
    print(deck(t)); print(CSV)
