"""영구동토 ALT 물리·반경험식 앙상블 (S2) — Stefan 5종.

`docs/RESEARCH_PLAN_...` §11.3. 워크플로 정밀조사(수식·계수·단위 실측검증) + 적대검증 반영.
멤버: p1 Stefan 기본 / p2 Stefan edaphic / p3 TTOP-Stefan(+동토마스크) / p4 Kudryavtsev / p5 λ보정(Sr변형).

⚠️ 단위 규약(적대검증 반영):
- SoilGrids 컬럼은 이미 물리단위(clay/sand/silt %, bdod g/cm³, soc g/kg). 이중 배율 적용 금지.
- Stefan I_t = n_t·TDD·86400 (도일→도초). TTOP는 °C·day 유지 후 /365.
- k_dry식은 ρ_d[kg/m³] 전제 → bdod×1000. assert로 강제.
- soc 질량분율 = soc/1000 (dg/kg 가정 ÷10000은 오류).

주의: 5종 멤버간 상관 0.93~1.00(전부 Stefan 축). 실질 다양성은 수준 오프셋·TTOP 마스크·Ku 눈 성분.
phys_std는 절대 불확실성이 아니라 상대 지표.
"""
from __future__ import annotations
import numpy as np

# --- 물리 상수 (verified: Clayton&Schaefer 2021, Johansen 1975) ---
LF = 334000.0        # 융해잠열 J/kg
RHO_W = 1000.0       # 물밀도 kg/m³
SDAY = 86400.0       # s/day
KW, KQ, KO = 0.57, 7.7, 2.0   # 물·석영·기타광물 열전도도 W/mK
KICE = 2.2           # 얼음 열전도도
RHO_MIN = 2650.0     # 광물입자밀도 kg/m³
CW_VOL = 4.19e6      # 물 체적열용량 J/m³/K
C_SOIL = 800.0       # 광물 비열 J/kg/K (석영~730-800)


def _fallback(a, ref=None):
    """NaN을 중앙값으로 대치(train 분포 기준은 호출자 책임)."""
    a = np.asarray(a, float).copy()
    m = ~np.isfinite(a)
    if m.any():
        a[m] = np.nanmedian(a) if ref is None else np.nanmedian(ref)
    return a


def load_physics_inputs(df):
    """SoilGrids·기후에서 pedotransfer로 물성 산출. dict 반환. (이미 물리단위, 이중변환 없음)"""
    tdd = np.clip(_fallback(df["e5_tdd"].values), 0, None)
    fdd = np.clip(_fallback(df["e5_fdd"].values), 0, None)   # 양수 저장
    sqtdd = np.clip(_fallback(df["e5_sqrt_tdd"].values), 0, None)
    maat = _fallback(df["e5_maat"].values)
    twarm = _fallback(df["e5_twarm"].values)
    tcold = _fallback(df["e5_tcold"].values)
    swe = np.clip(_fallback(df["e5_swe"].values), 0, 0.20)   # 빙하 이상치 clip

    rho_d = _fallback(df["sg_bdod_5_15"].values) * 1000.0    # g/cm³ → kg/m³
    assert np.all((rho_d >= 150) & (rho_d <= 2000)), "bdod 단위 오류(ρ_d 범위 이탈) — ×1000 확인"
    sand_f = np.clip(_fallback(df["sg_sand_5_15"].values) / 100.0, 0, 1)
    cfvo_f = np.clip(_fallback(df["sg_cfvo_5_15"].values) / 100.0, 0, 0.6)
    soc_frac = np.clip(_fallback(df["sg_soc_5_15"].values) / 1000.0, 0, 0.6)  # g/kg → 질량분율

    f_om = np.clip(1.724 * soc_frac, 0, 0.9)                 # 유기물 분율(Van Bemmelen)
    rho_s = 1.0 / ((1 - f_om) / RHO_MIN + f_om / 1300.0)     # 혼합 입자밀도
    phi = np.clip(1.0 - rho_d / rho_s, 0.05, 0.92)           # 공극률
    theta_sat = phi * (1 - cfvo_f)                           # 포화 체적함수비
    return dict(tdd=tdd, fdd=fdd, sqtdd=sqtdd, maat=maat, twarm=twarm, tcold=tcold,
                swe=swe, rho_d=rho_d, sand_f=sand_f, f_om=f_om, phi=phi, theta_sat=theta_sat)


def johansen_k(rho_d, sand_f, phi, f_om, Sr, frozen=False):
    """Johansen 1975 열전도도 W/mK (해빙/동결)."""
    q = np.clip(sand_f, 0, 1)
    k_o = np.where(q > 0.2, KO, 3.0)
    k_s = KQ ** q * k_o ** (1 - q)
    kw_or_ice = KICE if frozen else KW
    k_org = 0.7 if frozen else 0.55
    k_sat_min = k_s ** (1 - phi) * kw_or_ice ** phi
    k_sat = (1 - f_om) * k_sat_min + f_om * k_org
    k_dry = (0.135 * rho_d + 64.7) / (2700.0 - 0.947 * rho_d)
    if frozen:
        Ke = np.clip(Sr, 0, 1)
    else:
        c = np.where(sand_f >= 0.5, 0.7, 1.0)
        Ke = np.clip(1.0 + c * np.log10(np.clip(Sr, 1e-3, 1.0)), 0, 1)
    return np.clip((k_sat - k_dry) * Ke + k_dry, 0.05, 4.0)


def fit_E(alt_cm, sqrt_tdd, mask=None):
    """fold-safe Stefan E 역산 = median(ALT/√TDD). mask=train 인덱스 불리언."""
    a, s = np.asarray(alt_cm, float), np.asarray(sqrt_tdd, float)
    m = np.isfinite(a) & np.isfinite(s) & (s > 0)
    if mask is not None:
        m &= np.asarray(mask, bool)
    return float(np.median(a[m] / s[m])) if m.sum() >= 3 else np.nan


def _stefan_full(tdd, k_t, theta_w, n_t):
    I_t = n_t * tdd * SDAY
    L_vol = LF * RHO_W * np.clip(theta_w, 1e-2, 0.92)
    return 100.0 * np.sqrt(np.clip(2.0 * k_t * I_t / L_vol, 0, None))


def _kudryavtsev(maat, twarm, tcold, swe, k_t, k_f, C_t, theta_w):
    Ta = maat
    Aa = np.clip((twarm - tcold) / 2.0, 0.5, None)
    hsn = swe * RHO_W / 250.0                        # SWE[m]→눈깊이[m], ρ_sn=250
    Ksn = 3.233 * 0.25 ** 2 - 1.01 * 0.25 + 0.138
    Dsn = Ksn / (250.0 * 2090.0)
    dTsn = Aa * (1 - np.exp(-hsn * np.sqrt(np.pi / (3.1536e7 * Dsn))))
    Tgs = Ta + dTsn
    Ags = np.clip(Aa - dTsn * 2 / np.pi, 0.5, None)
    r = np.clip(Tgs / Ags, -0.999, 0.999)
    num = 0.5 * Tgs * (k_f + k_t) + (Ags * (k_t - k_f) / np.pi) * (r * np.arcsin(r) + np.sqrt(1 - r * r))
    Tps = num / np.where(num <= 0, k_f, k_t)
    Lv = 3.34e8 * np.clip(theta_w, 1e-2, 0.92)
    aT = np.abs(Tps)
    half = Lv / (2 * C_t)
    Aps = np.clip((Ags - aT) / np.log(np.clip((Ags + half) / (aT + half), 1.0001, None)) - half, 1e-3, None)
    b = np.sqrt(k_t * 3.1536e7 * C_t / np.pi)
    s = np.sqrt(k_t * 3.1536e7 / (np.pi * C_t))
    Zc = 2 * (Ags - aT) * b / (2 * Aps * C_t + Lv)
    denom = (2 * Aps * C_t + Lv) * Zc + (2 * Aps * C_t + Lv) * s
    Zal = (2 * (Ags - aT) * b + ((2 * Aps * C_t + Lv) * Zc * Lv * s) / np.where(denom == 0, np.nan, denom)) / (2 * Aps * C_t + Lv)
    alt = np.where((Tps < 0) & (Ags > aT), Zal * 100.0, np.nan)
    return alt, Tps


def physics_ensemble(df, E):
    """물리 5종 ALT(cm) + phys_mean/std + TTOP·perm_mask·Tps. E는 fold-safe 역산값."""
    x = load_physics_inputs(df)
    rho_d, sand_f, phi, f_om = x["rho_d"], x["sand_f"], x["phi"], x["f_om"]
    theta_sat, tdd, fdd, swe = x["theta_sat"], x["tdd"], x["fdd"], x["swe"]

    # 열물성 (해빙/동결)
    theta_09 = np.clip(theta_sat * 0.9, 1e-2, 0.92)
    theta_06 = np.clip(theta_sat * 0.6, 1e-2, 0.92)
    k_t = johansen_k(rho_d, sand_f, phi, f_om, Sr=0.9)
    k_t6 = johansen_k(rho_d, sand_f, phi, f_om, Sr=0.6)
    k_f = johansen_k(rho_d, sand_f, phi, f_om, Sr=0.9, frozen=True)
    C_t = rho_d * C_SOIL + CW_VOL * theta_09

    # p1 기본 Stefan (정확도 담당, E 라벨의존)
    p1 = E * x["sqtdd"]
    # p2 edaphic 포화근접
    p2 = _stefan_full(tdd, k_t, theta_09, n_t=0.9)
    # p3 TTOP → 동토마스크, ALT는 동토셀만
    r_k = k_t / np.clip(k_f, 0.05, None)
    ttop = (0.9 * r_k * tdd - 0.5 * fdd) / 365.0
    perm_mask = (ttop < 0).astype(float)
    p3 = np.where(ttop < 0, p2, np.nan)
    # p4 Kudryavtsev (눈 절연·오프셋, p1과 비상관 성분 최대)
    p4, Tps = _kudryavtsev(x["maat"], x["twarm"], x["tcold"], swe, k_t, k_f, C_t, theta_09)
    # p5 λ보정 + 불포화(Sr=0.6) 변형
    p5_raw = _stefan_full(tdd, k_t6, theta_06, n_t=0.9)
    Ste = np.clip(C_t * np.clip(x["twarm"], 0, None) / (LF * RHO_W * theta_06), 0, 2)
    lam = 1 - 0.16 * Ste + 0.038 * Ste ** 2
    p5 = p5_raw * lam

    P = np.vstack([p1, p2, p3, p4, p5])
    P = np.clip(P, 5.0, 300.0)
    phys_mean = np.nanmean(P, axis=0)
    phys_std = np.nanstd(P, axis=0)
    return dict(p1_stefan=P[0], p2_edaphic=P[1], p3_ttop=P[2], p4_kudryavtsev=P[3],
                p5_lambda=P[4], phys_mean=phys_mean, phys_std=phys_std,
                ttop=ttop, perm_mask=perm_mask, tps=Tps)


PHYSICS_MEMBERS = ["p1_stefan", "p2_edaphic", "p3_ttop", "p4_kudryavtsev", "p5_lambda"]
