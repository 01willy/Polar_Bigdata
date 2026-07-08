"""공용 matplotlib 스타일 — 한글 폰트 + 극지/빙권 표준 컬러맵 + 논문급 프리셋.

모든 시각화 스크립트는 이 모듈만 import 하면 일관된 스타일을 얻는다.

    from polar.plotstyle import use_polar, CMAP
    plt = use_polar()
    ax.pcolormesh(x, y, alt, cmap=CMAP.alt, vmin=20, vmax=110)

컬러맵 규약 (Crameri 2020, Nature Comms — 지각·빙권 논문 표준. 지각균일·색맹안전):
- alt   : 활성층 두께 등 '깊이/두께' 순차형. 한대 색감(옅음→짙은 청색, 두꺼울수록 짙음).
- temp  : 지중온도 발산형. 0°C 중심(TwoSlopeNorm 권장) — 청=동결, 적=융해.
- err   : 오차/불확실성 순차형. 붉은 계열 회피(옅음→짙은 자주).
- diff  : 차이/개선 발산형. 0 중심.
- count : 밀도/개수 순차형. 한대 색감.
- terr  : 고도/지형 순차형(중립 대지색).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from cmcrameri import cm as _cmc

_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf",
]


def _trunc(cmap, lo=0.0, hi=1.0, n=256, name=None):
    """컬러맵 양끝(순백/순흑) 잘라 가독성 개선."""
    import numpy as np
    cols = cmap(np.linspace(lo, hi, n))
    return LinearSegmentedColormap.from_list(name or f"{cmap.name}_t", cols)


class _CMAP:
    """프로젝트 표준 컬러맵 묶음(지연 생성)."""
    # 활성층 두께/깊이: oslo 역방향(연청→짙은청). 저단(near-white)·고단(near-black)을 넉넉히
    # 잘라 (a)해양/결측 회색과의 혼동, (b)고값 포화(near-black)를 동시에 방지.
    alt = _trunc(_cmc.oslo_r, 0.12, 0.90, name="polar_alt")
    # 지중온도 발산: vik (청=한랭 ↔ 적=온난), 0°C 중심 권장
    temp = _cmc.vik
    # 오차/불확실성: acton (옅음→자주). 붉은 계열 아님, 온도와 구분됨
    err = _trunc(_cmc.acton, 0.05, 0.95, name="polar_err")
    # 차이/개선 발산: broc (청↔갈), 온도맵과 시각 분리
    diff = _cmc.broc
    # 밀도/개수: davos 역방향(옅음→짙은 청록), 한대 색감
    count = _trunc(_cmc.davos_r, 0.06, 0.97, name="polar_count")
    # 고도/지형: 중립 대지색
    terr = _cmc.bukavu if hasattr(_cmc, "bukavu") else _cmc.grayC


CMAP = _CMAP()
BAD = "#e9ecef"        # NaN/해양/결측 = 옅은 회색(중립) — 데이터 오독 방지
FROZEN = "#2166ac"     # 동결(강조 텍스트/등고선)
THAWED = "#b2182b"     # 융해

# 모든 표준 컬러맵의 결측(NaN)을 중립 회색으로 — 빙권 논문 관례
for _c in (CMAP.alt, CMAP.temp, CMAP.err, CMAP.diff, CMAP.count, CMAP.terr):
    _c.set_bad(BAD)


def tnorm(vmin, vmax, vcenter=0.0):
    """온도/차이 발산맵용 0중심 노름 — 중립색(흰색)이 0°C(동결/융해 경계)에 정렬."""
    from matplotlib.colors import TwoSlopeNorm
    vmin = min(vmin, vcenter - 1e-6)
    vmax = max(vmax, vcenter + 1e-6)
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def use_polar():
    """한글 폰트 + 논문급 rcParams 적용 후 pyplot 반환."""
    for path in _CANDIDATES:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 260,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 9.5,
        "image.cmap": "cmc.batlow",
    })
    return plt


def use_korean():
    """하위호환 별칭 — 신규 코드는 use_polar() 사용."""
    return use_polar()


def lon_formatter():
    """음수 경도를 서경(°W)으로 표기하는 FuncFormatter (지도 x축 표기 오류 방지)."""
    from matplotlib.ticker import FuncFormatter
    def _f(x, _pos):
        if x < 0:
            return f"{abs(x):.0f}°W"
        return f"{x:.0f}°E"
    return FuncFormatter(_f)


def lat_formatter():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda y, _p: f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}")


def despine(ax):
    """상/우 spine 제거 (지도/그래프 공통)."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def style_geo(ax, title=None, xlabel="경도 (°E)", ylabel="위도 (°N)"):
    """지도축 공통 마무리: 라벨·격자·타이틀."""
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.35, lw=0.4, color="#bbbbbb")
    ax.tick_params(length=3)
    return ax


def add_cbar(fig, mappable, ax, label, **kw):
    """일관된 컬러바(테두리·여백·라벨)."""
    kw.setdefault("shrink", 0.85)
    kw.setdefault("pad", 0.02)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.set_label(label, fontsize=10)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#444444")
    cb.ax.tick_params(labelsize=9, length=2.5)
    return cb
