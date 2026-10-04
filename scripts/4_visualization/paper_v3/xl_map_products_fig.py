"""XL 그림(개정 2, 2026-10-05 검토 반영): 우리 1 km 지도와 ALT 제품의 비교(알래스카, 레나 델타), 표시 해상도 시연. 판정 없음(등록 XL).

입력  data/processed/map_alaska/alaska_pred_v1.csv.gz, data/processed/map_lena/lena_pred_v1.csv.gz(maps_alt_v3.load_region),
      data/processed/xbatch/XL_map_products/xl_{alaska,lena}_products_v1.csv.gz, xl_summary_v1.csv(xl_map_products_v1.py)
그림 1  XL_<Region>_product_differences(두 행)
  위 행: 잔차 ML ALT, CCI v5, Wei 2026, Aalto 2018, (알래스카) Yi-Kimball. 한 순차 색표(oslo_r)와 한 범위(표시 셀의 모든 위 행 값을 합친
         2–98 백분위, 5 cm 단위 바깥 반올림). 제품 지도 아래 회색 글: 영역 평균(cm)과 우리 지도와의 공간 상관 r(xl_summary_v1.csv).
  아래 행: Stefan − 잔차 ML(자체 대칭 범위, 지도 그림 b 와 같은 크기의 범위), 제품 − 잔차 ML(제품 차 패널 |차| 합동 98 백분위의 대칭 범위).
         제목의 빼기는 U+2212 다. 제품 값이 없는 표시 셀은 #B8BEC6.
그림 2  XL_display_resolution(1 × 4, 같은 크기 패널): 알래스카 50 × 50 km 확대(1 km 셀, ERA5-Land 0.1° 셀 평균, 흰 선 = 0.1° 셀 경계),
  레나 델타 전체(같은 두 판). 원 1 km 격자를 셀 사각형 그대로 그린다(maps_alt_v3.build_mesh). 지역마다 색 막대 하나(두 판 합동 1–99 백분위, 2 cm 단위).
논문판 170 mm, Liberation Sans 7 pt(그림 1 은 짧은 패널 제목, 그림 2 는 패널 문자만). 슬라이드판 12.0 in 폭 이하, 5.2 in 높이 이하, Pretendard.
산출  outputs/figures/paper/v3_restructure/maps/XL_*.{pdf,png}, XL_figures_source_values.json, deck/assets/paper_report/maps/XL_*_slide.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm                                                   # noqa: E402
from matplotlib.lines import Line2D                                                                     # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402

XL = ROOT / "data" / "processed" / "xbatch" / "XL_map_products"
MINUS = MV.MINUS
PROD_NAME = {"cci5": "CCI v5", "wei": "Wei 2026", "aalto": "Aalto 2018", "yk": "Yi-Kimball"}


def load(key):
    d = MV.prepare(MV.load_region(key))
    v = pd.read_csv(XL / f"xl_{key}_products_v1.csv.gz", dtype={"cell_id": str})
    assert (v.cell_id.values == d["cells"].cell_id.values).all()
    d["prods"] = [p for p in ("cci5", "wei", "aalto", "yk") if p in v.columns]
    sh = d["shown"]
    d["pv"] = {p: v[p].values.astype(float) for p in d["prods"]}
    d["diff"] = {p: np.where(sh, d["pv"][p] - d["r1"], np.nan) for p in d["prods"]}
    d["diff_stefan"] = np.where(sh, d["p1"] - d["r1"], np.nan)
    pooled = np.concatenate([d["r1"][sh]] + [d["pv"][p][sh] for p in d["prods"]])
    d["rng_top"] = MV.pct_range(pooled, 2.0, 98.0, 5.0)
    d["vmax_prod"] = MV.sym_max(np.concatenate([d["diff"][p][sh] for p in d["prods"]]), 98.0, 5.0)
    d["vmax_stefan"] = MV.sym_max(d["diff_stefan"][sh], 99.0, 5.0)
    summ = pd.read_csv(XL / "xl_summary_v1.csv")
    d["summ"] = summ[summ.region == key].set_index("item")
    return d


def rgba_prod(d, values, cmap, norm):
    """표시 셀 가운데 값이 없는 셀은 #B8BEC6(제품 결측). 나머지는 maps_alt_v3.cell_rgba 와 같다."""
    rgba = MV.cell_rgba(d, values, cmap, norm)
    miss = d["shown"] & ~np.isfinite(values)
    rgba[miss] = MV.to_rgba(MV.NODATA)
    return rgba


def stat_text(d, item):
    s = d["summ"]
    m = float(s.loc[item, "mean"])
    if item == "ours":
        return f"mean {S.fmt_num(m, 1)} cm"
    return f"mean {S.fmt_num(m, 1)} cm, r {S.fmt_num(float(s.loc[item, 'pearson_r']), 2)}"


def scale_bar_free_or_below(fig, ax, d, ext, M, panel_w_mm, below_xy_mm, km):
    """빈 모서리가 있으면 그 안에, 없으면 below_xy_mm(그림 mm 좌표, 좌표 글이 없는 띠)에 둔다."""
    w, h = MV.scale_box_size(d, ext, km, M, panel_w_mm)
    pts = MV.occupancy_points(d)
    k, box, counts = MV.pick_corner(pts, ext, w, h, order=("bl", "tl", "tr", "br"))
    if counts[k] == 0:
        MV.scale_bar(ax, d["proj"], box, km, M)
        return dict(place=k)
    L_mm = MV.bar_length(d["proj"], (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, km) / (ext[1] - ext[0]) * panel_w_mm
    MV.scale_bar_below(fig, below_xy_mm[0], below_xy_mm[1], L_mm, km, M)
    return dict(place="below", length_mm=round(L_mm, 2))


# ================================================================ 그림 1: 제품 비교
def fig_products(d, medium):
    M = MV.Med(medium)
    MV.use_medium(medium)
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    top = [("ours", "Residual ML")] + [(p, PROD_NAME[p]) for p in d["prods"]]
    bot = [("stefan", "Stefan")] + [(p, PROD_NAME[p]) for p in d["prods"]]
    nc = len(top)
    lo, hi = d["rng_top"]
    norm_top = Normalize(lo, hi)
    vp, vs = d["vmax_prod"], d["vmax_stefan"]
    norm_p = TwoSlopeNorm(0.0, -vp, vp)
    norm_s = TwoSlopeNorm(0.0, -vs, vs)
    sh = d["shown"]
    rec = dict(top_range=[lo, hi], prod_diff_max=vp, stefan_diff_max=vs)
    if M.paper:
        W, LM, GAP, RM = 170.0, 8.0, 1.5 if nc == 5 else 2.0, 0.5
        w = (W - LM - (nc - 1) * GAP - RM) / nc
        h = w * asp
        T, STAT, CBB, CBT, LON = 3.2, 3.4, 9.8, 12.6, 4.2           # CBT: 위 행 색 막대 칸(라벨이 아래 행 제목과 겹치지 않게)
        H = T + h + STAT + CBT + T + h + LON + CBB + 0.5
        fig = S.fig_mm(W, H)
        xs = [LM + j * (w + GAP) for j in range(nc)]
        y1 = T
        y2 = T + h + STAT + CBT + T
    else:
        W, LM, GAP, RM = MV.SLIDE_W_MM, 16.5, 2.5, 1.0
        CBS, CBR = 30.0, 29.0                                          # Stefan 세로 막대 칸, 오른쪽 세로 막대 칸
        T1, T2, STAT, BOT = 8.0, 14.5, 6.0, 9.0
        w = (W - LM - CBS - CBR - (nc - 2) * GAP - RM) / nc
        h = w * asp
        hmax = (MV.SLIDE_H_MAX_MM - T1 - T2 - STAT - BOT - 1.0) / 2
        if h > hmax:
            h = hmax
            w = h / asp
        used = LM + nc * w + CBS + CBR + (nc - 2) * GAP + RM
        off = (W - used) / 2
        H = T1 + h + STAT + T2 + h + BOT
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        xs = [off + LM] + [off + LM + w + CBS + (j - 1) * (w + GAP) for j in range(1, nc)]
        y1 = T1
        y2 = T1 + h + STAT + T2
    Wf, Hf = fig.get_size_inches() * 25.4
    axes = {}
    letters = iter("abcdefghij")
    # 위 행
    for j, (item, name) in enumerate(top):
        ax = S.axes_mm(fig, xs[j], y1, w, h, projection=proj)
        MV.base_map(ax, d, ext, M)
        vals = d["r1"] if item == "ours" else d["pv"][item]
        MV.add_cells(ax, d["mesh"], rgba_prod(d, vals, MV.CM_ALT, norm_top) if item != "ours" else MV.cell_rgba(d, vals, MV.CM_ALT, norm_top))
        MV.graticule(ax, d["xlocs"], d["ylocs"], M, left=(j == 0))
        MV.letter(fig, xs[j], y1 - (0.5 if M.paper else 1.2), next(letters), M, name)
        t = fig.text((xs[j] + w / 2) / Wf, 1 - (y1 + h + (0.6 if M.paper else 1.0)) / Hf, stat_text(d, item), ha="center", va="top",
                     fontsize=M.fs_small if M.paper else 12.0, color=MV.GRAY_TXT)
        t.set_gid("stats")
        axes[("t", j)] = ax
    # 아래 행
    for j, (item, name) in enumerate(bot):
        ax = S.axes_mm(fig, xs[j], y2, w, h, projection=proj)
        MV.base_map(ax, d, ext, M)
        if item == "stefan":
            MV.add_cells(ax, d["mesh"], MV.cell_rgba(d, d["diff_stefan"], MV.CM_DIFF, norm_s))
        else:
            MV.add_cells(ax, d["mesh"], rgba_prod(d, d["diff"][item], MV.CM_DIFF, norm_p))
        MV.graticule(ax, d["xlocs"], d["ylocs"], M, left=(j == 0), bottom=(j == 0 and M.paper))
        if M.paper:
            MV.letter(fig, xs[j], y2 - 0.5, next(letters), M, f"{name} {MINUS} residual ML")
        else:                                                          # 두 줄 제목: 문자는 첫 줄에 맞춘다
            MV.letter(fig, xs[j], y2 - 1.2, next(letters) + "\n ", M, f"{name}\n{MINUS} residual ML")
        axes[("b", j)] = ax
    # 색 막대
    ext_top = MV.extend_of(np.concatenate([d["r1"][sh]] + [d["pv"][p][sh] for p in d["prods"]]), lo, hi)
    ext_p = MV.extend_of(np.concatenate([d["diff"][p][sh] for p in d["prods"]]), -vp, vp)
    ext_s = MV.extend_of(d["diff_stefan"][sh], -vs, vs)
    if M.paper:
        span = xs[-1] + w - xs[0]
        cax = S.axes_mm(fig, xs[0] + 0.2 * span, y1 + h + STAT + 1.0, 0.6 * span, M.cbar)
        MV.colorbar(fig, cax, MV.CM_ALT, norm_top, "ALT (cm)", MV.ticks_between(lo, hi), M, ext_top)
        ycb = y2 + h + LON + 0.6
        cax = S.axes_mm(fig, xs[0] + 0.08 * w, ycb, 0.84 * w, M.cbar)
        MV.colorbar(fig, cax, MV.CM_DIFF, norm_s, "ALT difference (cm)", MV.ticks_between(-vs, vs), M, ext_s)
        span2 = xs[-1] + w - xs[1]
        cax = S.axes_mm(fig, xs[1] + 0.2 * span2, ycb, 0.6 * span2, M.cbar)
        MV.colorbar(fig, cax, MV.CM_DIFF, norm_p, "ALT difference (cm)", MV.ticks_between(-vp, vp), M, ext_p)
        rec["scale_bar"] = scale_bar_free_or_below(fig, axes[("t", 0)], d, ext, M, w, (xs[-1], y2 + h + 2.4), 500 if d["key"] == "alaska" else 50)
    else:
        xr = xs[-1] + w + 2.0
        Ms = MV.Med("slide")
        Ms.fs_lab = 14.0                                               # 세로 막대 라벨이 막대 길이(0.9 h) 안에 들도록
        cax = S.axes_mm(fig, xr, y1 + 0.05 * h, M.cbar, 0.9 * h)
        MV.colorbar(fig, cax, MV.CM_ALT, norm_top, "ALT (cm)", MV.ticks_between(lo, hi), Ms, ext_top, orientation="vertical")
        cax = S.axes_mm(fig, xr, y2 + 0.05 * h, M.cbar, 0.9 * h)
        MV.colorbar(fig, cax, MV.CM_DIFF, norm_p, "Difference (cm)", MV.ticks_between(-vp, vp), Ms, ext_p, orientation="vertical")
        cax = S.axes_mm(fig, xs[0] + w + 2.0, y2 + 0.05 * h, M.cbar, 0.9 * h)
        MV.colorbar(fig, cax, MV.CM_DIFF, norm_s, "Difference (cm)", MV.ticks_between(-vs, vs), Ms, ext_s, orientation="vertical")
        rec["scale_bar"] = scale_bar_free_or_below(fig, axes[("t", 0)], d, ext, M, w, (xs[1], y2 + h + 4.5), 500 if d["key"] == "alaska" else 50)
        lg = S.axes_mm(fig, xs[-2], y2 + h + 0.8, 2 * w, BOT - 1.2)
        lg.set_axis_off()
        MV.nodata_label_legend(lg, M, "center right", dots=False, small=True)
        lg.get_legend().get_texts()[0].set_text("No value")
    if M.paper:
        lg = S.axes_mm(fig, xs[-1] - 0.6 * w, y1 + h + STAT + 1.0, 1.6 * w, 6.0)
        lg.set_axis_off()
        MV.nodata_label_legend(lg, M, "upper right", dots=False)
        lg.get_legend().get_texts()[0].set_text("No value")
    rec["size_mm"] = [round(Wf, 1), round(Hf, 1)]
    return fig, rec


# ================================================================ 그림 2: 표시 해상도
def e5_mean(d):
    c = d["cells"]
    iy = np.floor((c.lat.values + 0.05) / 0.1).astype(np.int64)
    ix = np.floor((c.lon.values + 0.05) / 0.1).astype(np.int64)
    k = pd.Series(iy).astype(str) + "_" + pd.Series(ix).astype(str)
    return pd.Series(np.where(d["shown"], d["r1"], np.nan)).groupby(k.values).transform("mean").values


def e5_lines(ax, ext, proj, M):
    """ERA5-Land 0.1° 셀 경계(격자점 ± 0.05°), 흰 선."""
    x0, x1, y0, y1 = ext
    ll = MV.PC.transform_points(proj, np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1]))
    lo0, lo1 = ll[:, 0].min() - 0.3, ll[:, 0].max() + 0.3
    la0, la1 = ll[:, 1].min() - 0.2, ll[:, 1].max() + 0.2
    for lo in np.arange(math.floor(lo0 * 10) / 10 - 0.05, lo1 + 0.1, 0.1):
        lat = np.linspace(la0, la1, 30)
        ax.plot(np.full_like(lat, lo), lat, color="#ffffff", lw=M.lw, transform=MV.PC, zorder=3.2)
    for la in np.arange(math.floor(la0 * 10) / 10 - 0.05, la1 + 0.1, 0.1):
        lon = np.linspace(lo0, lo1, 60)
        ax.plot(lon, np.full_like(lon, la), color="#ffffff", lw=M.lw, transform=MV.PC, zorder=3.2)


def fig_resolution(da, dl, medium):
    M = MV.Med(medium)
    MV.use_medium(medium)
    lat_c, lon_c, n_lab, blk, cover = MV.zoom_center(da)
    xc, yc = da["proj"].transform_point(lon_c, lat_c, MV.PC)
    hz = da["zoom_half_m"]
    zext = [xc - hz, xc + hz, yc - hz, yc + hz]
    lext = MV.square_extent(dl["ext"])
    # 알래스카 확대: 상자 안(여유 2 km) 셀만으로 메시
    m = (np.abs(da["xy"][:, 0] - xc) <= hz + 2000) & (np.abs(da["xy"][:, 1] - yc) <= hz + 2000)
    idx = np.where(m)[0]
    c = da["cells"]
    mesh_z = MV.build_mesh(da["proj"], c.ky.values[idx], c.kx.values[idx])
    e5a, e5l = e5_mean(da), e5_mean(dl)
    vz = np.r_[da["r1"][idx][da["shown"][idx]], e5a[idx][da["shown"][idx]]]
    rng_a = MV.pct_range(vz, 1, 99, 2.0)
    vl = np.r_[dl["r1"][dl["shown"]], e5l[dl["shown"]]]
    rng_l = MV.pct_range(vl, 1, 99, 2.0)
    if M.paper:
        W, LM, GAP, RG, RM = 170.0, 0.5, 2.0, 9.0, 0.5
        w = (W - LM - 2 * GAP - RG - RM) / 4
        T, BAND, CBB = 3.2, 4.4, 9.8
        H = T + w + BAND + CBB + 0.5
        fig = S.fig_mm(W, H)
    else:
        W, LM, GAP, RG, RM = MV.SLIDE_W_MM, 1.0, 4.0, 16.0, 1.0
        w = (W - LM - 2 * GAP - RG - RM) / 4
        T, BAND, CBB = 9.0, 8.0, 21.5
        H = T + w + BAND + CBB + 1.5
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    xs = [LM, LM + w + GAP, LM + 2 * w + GAP + RG, LM + 3 * w + 2 * GAP + RG]
    titles = ["Alaska, 1 km", "Alaska, 0.1° mean", "Lena Delta, 1 km", "Lena Delta, 0.1° mean"]
    for j in range(4):
        if j < 2:
            d, ext, mesh, sel = da, zext, mesh_z, idx
            vals = da["r1"][idx] if j == 0 else e5a[idx]
            norm = Normalize(*rng_a)
        else:
            d, ext, mesh, sel = dl, lext, dl["mesh"], None
            vals = dl["r1"] if j == 2 else e5l
            norm = Normalize(*rng_l)
        ax = S.axes_mm(fig, xs[j], T, w, w, projection=d["proj"])
        MV.base_map(ax, d, ext, M)
        MV.add_cells(ax, mesh, MV.cell_rgba(d, vals, MV.CM_ALT, norm, idx=sel))
        if j < 2:
            e5_lines(ax, ext, d["proj"], M)
        else:
            MV.graticule(ax, d["xlocs"] if M.paper else d["xlocs"][::2], d["ylocs"], M, left=(j == 2), bottom=(j == 2))
        MV.letter(fig, xs[j], T - (0.5 if M.paper else 1.2), "abcd"[j], M, None if M.paper else titles[j])
        if j == 2:
            MV.scale_bar_below(fig, xs[3] + 0.4 * w, T + w + (2.4 if M.paper else 4.2), MV.bar_length(dl["proj"], (lext[0] + lext[1]) / 2,
                               (lext[2] + lext[3]) / 2, 50) / (lext[1] - lext[0]) * w, 50, M)
    L_mm = MV.bar_length(da["proj"], xc, yc, 10) / (2 * hz) * w
    MV.scale_bar_below(fig, xs[0], T + w + (2.4 if M.paper else 4.2), L_mm, 10, M)
    ycb = T + w + BAND + 0.6
    for (x0, rng, d, vals_ext) in ((xs[0], rng_a, da, vz), (xs[2], rng_l, dl, vl)):
        span = 2 * w + GAP
        cax = S.axes_mm(fig, x0 + 0.2 * span, ycb, 0.6 * span, M.cbar)
        MV.colorbar(fig, cax, MV.CM_ALT, Normalize(*rng), "ALT (cm)", MV.ticks_between(*rng), M, MV.extend_of(vals_ext, *rng))
    W_, H_ = fig.get_size_inches() * 25.4
    rec = dict(zoom=dict(lat=lat_c, lon=lon_c, block=blk, n_labels_block=n_lab, shown_cover=cover, half_km=hz / 1000.0, n_cells=int(len(idx))),
               ranges=dict(alaska=rng_a, lena=rng_l), size_mm=[round(W_, 1), round(H_, 1)])
    return fig, rec


def corr_1km_e5(d):
    sh = d["shown"]
    a = np.where(sh, d["r1"], np.nan)
    m = e5_mean(d)
    ok = np.isfinite(a) & np.isfinite(m)
    r = float(np.corrcoef(a[ok], m[ok])[0, 1])
    return dict(r=r, r2=r * r, within_sd=float(np.std(a[ok] - m[ok])), total_sd=float(np.std(a[ok])))


def main():
    out = MV.OUT_PAPER
    rec = {}
    ds = {}
    for key in ("alaska", "lena"):
        d = load(key)
        ds[key] = d
        stem = f"XL_{'Alaska' if key == 'alaska' else 'Lena'}_product_differences"
        rec[key] = dict(resolution=corr_1km_e5(d))
        for medium in ("paper", "slide"):
            fig, r = fig_products(d, medium)
            rec[key][medium] = r
            if medium == "paper":
                S.save_fig(fig, stem, out, formats=("pdf", "png"))
                au = S.audit_v3(fig, allowed_num_gids=("scale", "sizekey", "stats"))
                print(f"[audit] {stem}: sizes {sorted(au['sizes'])} chars {au['chars']} thin {au['thin_lines'][:3]} long {au['long_labels'][:3]} "
                      f"numbers {au['loose_numbers'][:3]}", flush=True)
            else:
                fig.savefig(MV.OUT_SLIDE / f"{stem}_slide.png", dpi=300)
            plt.close(fig)
        print(f"[xl-fig] {key} {json.dumps(rec[key], ensure_ascii=False)}", flush=True)
    for medium in ("paper", "slide"):
        fig, r = fig_resolution(ds["alaska"], ds["lena"], medium)
        rec.setdefault("display_resolution", {})[medium] = r
        if medium == "paper":
            S.save_fig(fig, "XL_display_resolution", out, formats=("pdf", "png"))
            au = S.audit_v3(fig)
            print(f"[audit] XL_display_resolution: sizes {sorted(au['sizes'])} chars {au['chars']} thin {au['thin_lines'][:3]}", flush=True)
        else:
            fig.savefig(MV.OUT_SLIDE / "XL_display_resolution_slide.png", dpi=300)
        plt.close(fig)
    print(f"[xl-fig] display {json.dumps(rec['display_resolution'], ensure_ascii=False)}", flush=True)
    (out / "XL_figures_source_values.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
