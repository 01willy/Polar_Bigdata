"""색각 시뮬레이션·대비 검사(논문 그림 QA, 스펙 §1.1·§5.1 단계 4).

Machado 2009 severity 1.0 deuteranopia·protanopia 행렬로 선형 RGB 를 변환한 뒤 CIEDE2000 ΔE00 을 계산한다.
판정 기준: 같은 패널 안 색 쌍의 시뮬레이션 ΔE00 ≥ 12, 선·마커의 흰 바탕 WCAG 대비 ≥ 3:1.
    from polar.cvd import check_colors
    rep = check_colors({"direct": "#6b7280", "residual": "#9a7bc9"})   # {'ok': bool, 'pairs': [...], 'contrast': {...}}
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

DEU = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])
PRO = np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]])
MIN_DE00 = 12.0
MIN_CONTRAST = 3.0


def hex2rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _delin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb, kind: str = "deut") -> np.ndarray:
    """sRGB(0–1) → 색각 이상 시뮬레이션 sRGB. kind ∈ {'deut', 'prot'}."""
    M = DEU if kind == "deut" else PRO
    return _delin(M @ _lin(np.asarray(rgb, float)))


def lab(rgb) -> np.ndarray:
    l = _lin(np.asarray(rgb, float))
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M @ l / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def de00(l1, l2) -> float:
    """CIEDE2000 색차."""
    L1, a1, b1 = l1; L2, a2, b2 = l2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2); Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360; h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p; dh = h2p - h1p
    if C1p * C2p == 0:
        dh = 0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    T = 1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp)) + 0.32 * np.cos(np.radians(3 * hbp + 6)) \
        - 0.20 * np.cos(np.radians(4 * hbp - 63))
    dth = 30 * np.exp(-((hbp - 275) / 25) ** 2); RC = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2); SC = 1 + 0.045 * Cbp; SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dth)) * RC
    return float(np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)))


def luminance(h: str) -> float:
    l = _lin(hex2rgb(h))
    return float(0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2])


def contrast(h: str, bg: str = "#ffffff") -> float:
    """WCAG 대비비."""
    a, b = luminance(h), luminance(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def lightness(h: str) -> float:
    return float(lab(hex2rgb(h))[0])


def pair_de(c1: str, c2: str) -> dict:
    """정상 시각·deut·prot ΔE00 과 최소값."""
    r1, r2 = hex2rgb(c1), hex2rgb(c2)
    d = dict(normal=de00(lab(r1), lab(r2)), deut=de00(lab(simulate(r1, "deut")), lab(simulate(r2, "deut"))),
             prot=de00(lab(simulate(r1, "prot")), lab(simulate(r2, "prot"))))
    d["min_cvd"] = min(d["deut"], d["prot"])
    return d


def check_colors(colors: dict, min_de: float = MIN_DE00, min_contrast: float = MIN_CONTRAST, bg: str = "#ffffff",
                 skip_pairs: set | None = None) -> dict:
    """한 패널(또는 그림)에서 함께 쓰는 색 {이름: hex} 의 모든 쌍 시뮬레이션 ΔE00 과 흰 바탕 대비를 검사한다.
    같은 hex 쌍(예: 검정 참조 두 종)은 선종·마커로 구분하므로 건너뛴다. skip_pairs = {('a','b'), …} 추가 제외."""
    skip_pairs = {tuple(sorted(p)) for p in (skip_pairs or set())}
    pairs, fails = [], []
    for (k1, c1), (k2, c2) in combinations(colors.items(), 2):
        if c1.lower() == c2.lower() or tuple(sorted((k1, k2))) in skip_pairs:
            continue
        d = pair_de(c1, c2); d.update(a=k1, b=k2); pairs.append(d)
        if d["min_cvd"] < min_de:
            fails.append(f"ΔE00({k1},{k2}) = {d['min_cvd']:.1f} < {min_de}")
    con = {k: contrast(c, bg) for k, c in colors.items()}
    fails += [f"contrast({k}) = {v:.2f} < {min_contrast}" for k, v in con.items() if v < min_contrast]
    return dict(ok=not fails, fails=fails, pairs=pairs, contrast=con, min_pair=min((p["min_cvd"] for p in pairs), default=np.nan))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from polar.paperstyle import METHOD
    cols = {k: v["color"] for k, v in METHOD.items()}
    rep = check_colors(cols)
    for k, c in cols.items():
        print(f"{k:14s} {c}  L* {lightness(c):5.1f}  contrast {rep['contrast'][k]:5.2f}")
    for p in sorted(rep["pairs"], key=lambda d: d["min_cvd"])[:8]:
        print(f"  {p['a']:>8s}–{p['b']:<8s} ΔE00 cvd min {p['min_cvd']:5.1f} (normal {p['normal']:5.1f})")
    print("ok" if rep["ok"] else rep["fails"])
