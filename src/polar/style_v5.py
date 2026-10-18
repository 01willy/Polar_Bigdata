"""v5 그림 양식: design/style_tokens_v4.json 을 읽어 matplotlib rc·색·글꼴을 맞춘다(논문 판).
규칙(토큰 symbols): 화살촉 주석·축 밖 삼각형·그림 안 상자 설명·그림 제목·굵은 축 라벨 금지. 패널 문자만 8 pt 굵게.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[2]
TOK = json.loads((ROOT / "design" / "style_tokens_v4.json").read_text())
MM = 1 / 25.4
W1, W2 = 88.0, 180.0                      # Nature 1단·2단 폭(mm)
C = TOK["color"]
METHOD_HEX = {k: v["hex"] for k, v in C["methods"].items()}
METHOD_EN = {k: v["en"] for k, v in C["methods"].items()}
METHOD_LS = {k: {"solid": "-", "dashed": (0, (3.5, 1.8)), "dotted": ":", "dashdot": "-."}[v["line"]] for k, v in C["methods"].items()}
METHOD_HEX.update({"P1a": "#2E7BBF", "Ku": "#56B4E9", "S": C["methods"]["Stack"]["hex"]})
METHOD_EN.update({"P1a": "Affine Stefan", "Ku": "Kudryavtsev", "S": "Label-weighted stack", "D0": "Direct ML", "R1": "Anchor + residual ML"})
METHOD_LS.update({"P1a": "-", "Ku": "-", "S": "-"})
PRODUCT_HEX = {k: v["hex"] for k, v in C["products"].items()}
PRODUCT_HEX.update({"cci4y": PRODUCT_HEX["CCI"], "cci5y": "#B8975A", "wei": PRODUCT_HEX["Wei"], "yk": PRODUCT_HEX["YK"], "aalto": PRODUCT_HEX["Aalto"],
                    "liu": PRODUCT_HEX.get("Liu", "#C9506A")})     # 토큰 color.products.Liu 를 따른다(2026-10-14: #C9506A 장미색, 파선)
PRODUCT_EN = {"cci4y": "ESA CCI v4", "cci5y": "ESA CCI v5", "wei": "Wei 2026", "yk": "Yi–Kimball 2019", "aalto": "Aalto 2018", "liu": "Liu 2024"}
REGION_HEX = {"Alaska": C["regions"]["Alaska"], "Lena": C["regions"]["Lena Delta"], "Canada": C["regions"]["Canada"]}
REGION_EN = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}
STATUS = C["status"]
NEUTRAL = C["neutral"]
F = TOK["font"]; L = TOK["lines"]["paper"]
import os as _os
PALETTE = _os.environ.get("POLAR_PALETTE", "deck")
REGION_MARKER = {"Alaska": "o", "Lena": "s", "Canada": "^"}
if PALETTE in ("paper", "paper11"):          # 논문 색: paper = v10 회색 위주(color.paper_v10), paper11 = v11 차분한 색(color.paper_v11). 딕셔너리를 제자리에서 바꾼다
    _PP = C["paper_v10" if PALETTE == "paper" else "paper_v11"]
    METHOD_HEX.update(_PP["methods"])
    PRODUCT_HEX.update({k: _PP["products"][v] for k, v in (("CCI", "CCI"), ("CCI5", "CCI5"), ("Wei", "Wei"), ("Aalto", "Aalto"), ("YK", "YK"), ("Liu", "Liu"),
                                                            ("cci4y", "CCI"), ("cci5y", "CCI5"), ("wei", "Wei"), ("yk", "YK"), ("aalto", "Aalto"), ("liu", "Liu"))})
    REGION_HEX.update({"Alaska": _PP["regions"]["Alaska"], "Lena": _PP["regions"]["Lena Delta"], "Canada": _PP["regions"]["Canada"]})
    NEUTRAL = dict(NEUTRAL); NEUTRAL["zero_line"] = _PP["zero_line"]
SCHEME_EN = {"random": "Random split", "block": "Spatial block", "year": "Year hold-out", "block_year": "Block × year", "region": "Region hold-out"}
FORCING_EN = {"year_obs": "Year-specific climate, to observation day", "year": "Year-specific climate, full season",
              "clim": "Mean climate 2011–2024"}      # 2026-10-15: 'forcing'(강제력) 표현 폐기(사용자 지적). 그림 라벨은 climate input


SCALE = 1.0            # 슬라이드판은 1.7(글자 7 → 12 pt, 선·마커 함께 확대). setup(scale=) 로 바꾼다
OUT_SUB = ""


def setup(paper=True, scale=1.0):
    global SCALE, OUT_SUB
    SCALE = scale; OUT_SUB = "" if scale == 1.0 else "slide"
    for f in (F["latin_regular_file"], F["latin_bold_file"]):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    t = F["paper"]["text_pt"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [F["latin"], "DejaVu Sans"], "font.size": t,
        "axes.labelsize": t, "xtick.labelsize": t - 0.5, "ytick.labelsize": t - 0.5,
        "legend.fontsize": t - 0.5, "axes.linewidth": L["axis_pt"], "xtick.major.width": L["tick_pt"],
        "ytick.major.width": L["tick_pt"], "xtick.major.size": L["tick_len_pt"], "ytick.major.size": L["tick_len_pt"],
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": L["data_pt"], "lines.markersize": L["marker_pt"], "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.titlesize": t, "axes.titleweight": "normal", "legend.frameon": False, "figure.dpi": 150,
        "axes.unicode_minus": False, "mathtext.default": "regular",
    })
    if PALETTE == "paper11":                 # v11: 글자 · 축을 순흑 대신 짙은 회색으로, 눈금 · 축선을 가늘게
        _P11 = C["paper_v11"]
        plt.rcParams.update({"text.color": _P11["text"], "axes.labelcolor": _P11["text"], "xtick.color": _P11["axis"], "ytick.color": _P11["axis"],
                             "xtick.labelcolor": _P11["text"], "ytick.labelcolor": _P11["text"], "axes.edgecolor": _P11["axis"], "grid.color": _P11["grid"]})


def figure(w_mm=W2, h_mm=100.0):
    """그림 크기는 mm 로 받고, 슬라이드판은 SCALE 배 키운 뒤 저장 시 dpi 를 그만큼 낮춰 글자가 SCALE 배 커진 PNG 를 만든다."""
    PANELS.clear()
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


PANELS: dict = {}      # 패널 문자 → 축 목록(슬라이드판 패널 크롭용). 그림마다 figure() 에서 비운다


def panel_letter(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=F["paper"]["panel_letter_pt"], fontweight="bold", va="bottom", ha="left")
    PANELS.setdefault(s, []).append(ax)


def register(letter, *axes):
    """패널 문자에 축(컬러바 등)을 추가로 묶는다."""
    PANELS.setdefault(letter, []).extend(axes)


def _crop_panels(fig, png, stem, sd, w_in):
    """슬라이드판: 패널 문자별 꽉 찬 상자(축·라벨·범례 포함)를 잘라 sd/<stem>__<letter>.png 로 저장(300 dpi 메타)."""
    from PIL import Image
    r = fig.canvas.get_renderer(); dpi = 3600.0 / w_in
    from matplotlib.transforms import Bbox
    fi = fig.get_tightbbox(r); full = Bbox.from_extents(fi.x0 * fig.dpi, fi.y0 * fig.dpi, fi.x1 * fig.dpi, fi.y1 * fig.dpi); pad = 0.02 * fig.dpi
    out = {}
    with Image.open(png) as im:
        Wpx, Hpx = im.size
        sx = Wpx / (full.width + 2 * pad); sy = Hpx / (full.height + 2 * pad)   # 저장 px / 그림 px
        for letter, axes in PANELS.items():
            bbs = []
            for a in axes:
                if not a.get_visible():
                    continue
                # 축 틀 + 눈금·축 라벨 + 범례 + 축 안 글자. (이미지·선의 잘리지 않은 범위는 넣지 않는다: 이웃 패널이 섞여 들어온다)
                bbs.append(a.get_window_extent(r))
                for axis in (a.xaxis, a.yaxis):
                    b = axis.get_tightbbox(r)
                    if b is not None:
                        bbs.append(b)
                leg = a.get_legend()
                if leg is not None:
                    bbs.append(leg.get_window_extent(r))
                for t in a.texts:
                    bbs.append(t.get_window_extent(r))
                for tt in (a.title, getattr(a, "_left_title", None), getattr(a, "_right_title", None)):
                    if tt is not None and tt.get_text():
                        bbs.append(tt.get_window_extent(r))
                gls = list(getattr(a, "_gridliners", [])) + [g for g in a.artists if type(g).__name__ == "Gridliner"]
                for gl in gls:   # cartopy 격자 라벨(표준 축이 아님)
                    arts = list(getattr(gl, "xlabel_artists", [])) + list(getattr(gl, "ylabel_artists", []))
                    arts += [getattr(lb, "artist", None) for lb in getattr(gl, "_labels", [])]
                    for t in arts:
                        if t is not None and t.get_visible() and t.get_text():
                            bbs.append(t.get_window_extent(r))
            bbs = [b for b in bbs if b is not None and b.width > 0]
            if not bbs:
                continue
            x0 = min(b.x0 for b in bbs); x1 = max(b.x1 for b in bbs); y0 = min(b.y0 for b in bbs); y1 = max(b.y1 for b in bbs)
            m = 0.03 * fig.dpi
            L = int(max(0, (x0 - full.x0 - m + pad) * sx)); R = int(min(Wpx, (x1 - full.x0 + m + pad) * sx))
            Tt = int(max(0, (full.y1 - y1 - m + pad) * sy)); B = int(min(Hpx, (full.y1 - y0 + m + pad) * sy))
            p = sd / f"{stem}__{letter}.png"
            im.crop((L, Tt, R, B)).save(p, dpi=(300, 300)); out[letter] = p
    return out


def export(fig, stem, outdir=ROOT / "outputs" / "figures" / "paper" / "v5", dpi=600):
    if _os.environ.get("POLAR_FIG_OUTDIR"):          # 보충 그림 재출력 등에서 출력 폴더를 바꾼다(덱 산출물을 덮어쓰지 않게)
        outdir = Path(_os.environ["POLAR_FIG_OUTDIR"])
    """논문판: pdf·svg·png(600 dpi). 슬라이드판(SCALE ≠ 1): outdir/slide/<stem>.png 하나. 그림 폭을 12 in(배치 폭)으로 두고 300 dpi 메타데이터로
    저장해 슬라이드에 12 in 으로 놓으면 배율 1.0 이고 글자는 7 pt × (12/7.09) ≈ 12 pt 가 된다."""
    outdir = Path(outdir)
    if SCALE != 1.0:
        from PIL import Image
        sd = outdir / "slide"; sd.mkdir(parents=True, exist_ok=True)
        p = sd / f"{stem}.png"
        w_in = fig.get_figwidth()
        fig.savefig(p, dpi=3600.0 / w_in, bbox_inches="tight", pad_inches=0.02)
        with Image.open(p) as im:
            im.save(p, dpi=(300, 300))
        crops = _crop_panels(fig, p, stem, sd, w_in) if PANELS else {}
        plt.close(fig)
        return [p] + list(crops.values())
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "svg", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi if ext == "png" else None, bbox_inches="tight", pad_inches=0.02)
        out.append(p)
    plt.close(fig)
    return out
