"""S7: KPDC Council 일별 지중온도 프로파일 파싱 + 0°C 등온선 ALT 유도.

대상: kpdc/council/soil_temp/daily_profile_ID/ (README 참조)
- 배치 A(개별 파일 ID01-14·ID21-24): 2023-09-01 ~ 2024-08-25, 8층(10-80cm)·16층(10-160cm).
- 배치 B(통합 워크북 ID01_Day_soil_2024-2025.xlsx): 2024-09-07 ~ 2025-09-21, 시트별 상이.
- 두 배치는 2024-09 재설치로 심부 온도가 불연속(동일 ID라도 위치·매핑 상이 가능)
  → site_key = "{ID}_{A|B}"로 분리, 배치 간 병합 금지.

깊이 메타: 파일 내 헤더(10-160cm)가 유일한 근거. 겨울 프로파일 단조성 검정(1-3월,
표층이 심부보다 한랭해야 함)으로 배치 A는 헤더가 깊이 역순임을 확인(README 의심 확증)
→ 실깊이 = (min+max) - 헤더깊이 로 교정. 교정은 물리 추론이므로 결과에
depth_reversed 플래그로 전파한다. 판정 모호(|rho|<0.3)한 ID10-14(채널-깊이 인터리브
의심, README)는 ALT 유도에서 제외한다.

QC:
- 채널 단위: 정확히 0.0 비율>20% = 사망 채널 마스크. 이웃 보간 잔차 중앙값>2.5°C =
  배선 오류 채널 마스크(예: 배치 B ID21 110cm).
- 기간 단위: 겨울(11-3월) 최심부 >+3°C 또는 9월 10cm <-1.5°C 인 달은 비지중 기록
  (설치 전 로거)으로 판정, 해당 월 이전 전체를 절단(예: 배치 B ID01의 2024-09~2025-01).

ALT 유도(derive_alt_gtnp_envelope.py 패턴):
- 일별 융해깊이 = 표층 연결 0°C 등온선 깊이(위 양수·아래 비양수 첫 교차의 선형보간).
- 연 ALT = 융해기(5-10월) 일별 융해깊이의 최댓값. 전층 양수인 날 존재 시
  ALT > 최심 센서 깊이(우측 검열, right_censored=1, alt_cm은 하한).
- 불확실성: 최대일 교차 구간 [alt_lo, alt_hi] = 센서 간격 전파(검열 시 alt_hi=inf).

산출: data/processed/s7_council_daily_temp.csv(QC 후 long),
      data/processed/s7_council_alt_derived.csv(지점×시즌),
      data/processed/s7_parse_meta.json(QC·교정 기록).
실행: /home/anaconda3/bin/python scripts/1_data_prep/s7_parse_kpdc_council.py  (ROOT에서)
      SMOKE=1 이면 ID21만 파싱해 /tmp/s7_smoke/에 저장.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T0 = time.time()
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "kpdc/council/soil_temp/daily_profile_ID"
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUTDIR = Path("/tmp/s7_smoke") if SMOKE else ROOT / "data/processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

# README 근거 상수
EXCLUDE_A = {"ID10", "ID11", "ID12", "ID13", "ID14"}  # 채널-깊이 인터리브 의심(README) + 단조성 검정 모호
SEASON = (5, 10)          # 융해기 월 범위
PEAK = ("-07-15", "-09-10")  # 피크 커버리지 판정 구간

qc_log = {"channel_masked": {}, "period_truncated": {}, "profile_excluded": {},
          "orientation": {}}


def _winter_rho(sub, depths, w0, w1):
    """겨울 창(w0~w1) 평균 프로파일과 깊이의 스피어만 상관. 양수=심부가 온난(정상)."""
    win = sub.loc[w0:w1].mean()
    ok = np.isfinite(win.values)
    if ok.sum() < 4:
        return np.nan
    return float(spearmanr(np.asarray(depths)[ok], win.values[ok]).statistic)


def _qc_channels(sub, key):
    """사망(상수 0)·배선오류(이웃 잔차) 채널 마스크. sub 컬럼=깊이 float 오름차순."""
    masked = []
    for c in sub.columns:
        z = float((sub[c] == 0).mean())
        if z > 0.2:
            masked.append((float(c), f"dead(zeros {z:.0%})"))
    # 사망 채널을 먼저 제거해야 이웃 보간 잔차가 오염되지 않는다
    sub = sub.drop(columns=[d for d, _ in masked])
    # 배선오류: 최악 채널부터 반복 마스크(오류 채널이 이웃 잔차를 오염시키는 것을 방지)
    for _ in range(4):
        cols = list(sub.columns)
        resids = {}
        for i in range(1, len(cols) - 1):
            interp = 0.5 * (sub[cols[i - 1]] + sub[cols[i + 1]])
            r = (sub[cols[i]] - interp).abs().median()
            if np.isfinite(r):
                resids[cols[i]] = float(r)
        if not resids or max(resids.values()) <= 2.5:
            break
        worst = max(resids, key=resids.get)
        masked.append((float(worst), f"wiring(med|resid|={resids[worst]:.1f}C)"))
        sub = sub.drop(columns=[worst])
    for d, why in masked:
        qc_log["channel_masked"].setdefault(key, []).append(f"{d:.0f}cm: {why}")
    return sub


def _qc_period(sub, key):
    """비지중 기록 월 탐지 후 그 이전 전체 절단."""
    if len(sub) == 0:
        return sub
    mon = sub.resample("MS").median()
    bad = []
    deep = mon.columns.max()
    for ts, row in mon.iterrows():
        if ts.month in (11, 12, 1, 2, 3) and row.get(deep, np.nan) > 3.0:
            bad.append(ts)
        if ts.month == 9 and row.get(mon.columns.min(), np.nan) < -1.5:
            bad.append(ts)
    if bad:
        cut = max(bad) + pd.offsets.MonthEnd(1)
        qc_log["period_truncated"][key] = f"~{cut.date()} 절단(비지중 의심 월 {len(bad)}개)"
        sub = sub.loc[sub.index > cut]
    return sub


def load_deploy_A():
    """배치 A 개별 파일 → {site_key: 깊이교정·QC 후 DataFrame}."""
    out = {}
    for f in sorted(RAW.glob("ID*_Day_SoilTemperature.xlsx")):
        pid = f.name[:4]
        if SMOKE and pid != "ID21":
            continue
        if pid in EXCLUDE_A:
            qc_log["profile_excluded"][f"{pid}_A"] = "채널-깊이 인터리브 의심(README)·단조성 모호"
            continue
        df = pd.read_excel(f, header=0)
        depths = [float(c) for c in df.columns[1:]]
        df.columns = ["DATE"] + depths
        df["dt"] = pd.to_datetime(df["DATE"].astype(str).str.strip("[]"), errors="coerce")
        sub = (df.dropna(subset=["dt"]).set_index("dt")[depths]
               .apply(pd.to_numeric, errors="coerce").sort_index())
        sub = sub[~sub.index.duplicated()]
        rho = _winter_rho(sub, depths, "2023-12-15", "2024-02-10")
        key = f"{pid}_A"
        if not np.isfinite(rho) or abs(rho) < 0.3:
            qc_log["profile_excluded"][key] = f"겨울 단조성 모호(rho={rho:.2f})"
            continue
        reversed_ = rho < 0
        if reversed_:
            lo, hi = min(depths), max(depths)
            sub.columns = [lo + hi - d for d in sub.columns]
            sub = sub[sorted(sub.columns)]
        qc_log["orientation"][key] = f"rho={rho:.2f} -> {'reversed' if reversed_ else 'as-labeled'}"
        sub = _qc_channels(sub, key)
        sub = _qc_period(sub, key)
        if len(sub) == 0:
            qc_log["profile_excluded"][key] = "비지중 기록 절단 후 잔여 없음"
            continue
        # 최종 물리성 게이트: 겨울 최심부가 여전히 +3°C 초과이면 비지중 기록으로 제외
        deep_win = sub.loc["2023-11-01":"2024-03-31", sub.columns.max()].median()
        if np.isfinite(deep_win) and deep_win > 3.0:
            qc_log["profile_excluded"][key] = f"겨울 최심부 {deep_win:.1f}C(비지중 기록)"
            continue
        out[key] = (sub, reversed_)
    return out


def load_deploy_B():
    """배치 B 통합 워크북 → {site_key: QC 후 DataFrame}. 깊이순서 기준본(README)."""
    out = {}
    xl = pd.ExcelFile(RAW / "ID01_Day_soil_2024-2025.xlsx")
    for sh in xl.sheet_names:
        pid = sh.replace("_Day", "")
        if SMOKE and pid != "ID21":
            continue
        df = xl.parse(sh, header=0)
        df["dt"] = pd.to_datetime(df["New date"], errors="coerce")
        dcols = [c for c in df.columns if str(c).endswith("cm")]
        depths = [float(str(c)[:-2]) for c in dcols]
        sub = (df.dropna(subset=["dt"]).set_index("dt")[dcols]
               .apply(pd.to_numeric, errors="coerce").sort_index())
        sub = sub[~sub.index.duplicated()]
        sub.columns = depths
        sub = sub[sorted(sub.columns)]
        rho = _winter_rho(sub, sorted(depths), "2025-02-15", "2025-03-31")
        key = f"{pid}_B"
        if not np.isfinite(rho) or abs(rho) < 0.3:
            qc_log["profile_excluded"][key] = f"겨울 단조성 모호(rho={rho:.2f}), 채널-깊이 매핑 신뢰 불가"
            continue
        if rho < 0:
            qc_log["profile_excluded"][key] = f"겨울 단조성 역전(rho={rho:.2f}), 워크북 기준본 가정 위배"
            continue
        qc_log["orientation"][key] = f"rho={rho:.2f} -> as-labeled(기준본)"
        sub = _qc_channels(sub, key)
        sub = _qc_period(sub, key)
        if len(sub) == 0:
            qc_log["profile_excluded"][key] = "비지중 기록 절단 후 잔여 없음"
            continue
        out[key] = (sub, False)
    return out


def thaw_depth_daily(depths, temps):
    """표층 연결 0°C 등온선 깊이. 반환 (깊이cm, 검열여부, lo, hi). 표층 동결이면 (0,False,0,0)."""
    m = np.isfinite(temps)
    d, t = np.asarray(depths)[m], np.asarray(temps)[m]
    if len(d) < 2:
        return np.nan, False, np.nan, np.nan
    if t[0] <= 0:
        return 0.0, False, 0.0, float(d[0])
    for i in range(len(d) - 1):
        if t[i] > 0 and t[i + 1] <= 0:
            z = d[i] + (d[i + 1] - d[i]) * t[i] / (t[i] - t[i + 1])
            return float(z), False, float(d[i]), float(d[i + 1])
    return float(d[-1]), True, float(d[-1]), np.inf   # 전층 양수 = 하한 검열


def season_alt(sub, year):
    """한 시즌(5-10월)의 ALT(연중 최대 융해깊이)와 검열·커버리지."""
    w = sub.loc[f"{year}-{SEASON[0]:02d}-01":f"{year}-{SEASON[1]:02d}-31"]
    if len(w) == 0:
        return None
    depths = list(w.columns)
    rows = [thaw_depth_daily(depths, r.values) for _, r in w.iterrows()]
    td = pd.DataFrame(rows, columns=["z", "cens", "lo", "hi"], index=w.index).dropna(subset=["z"])
    if len(td) == 0:
        return None
    censored = bool(td["cens"].any())
    if censored:
        alt, lo, hi = float(w.columns[-1]), float(w.columns[-1]), np.inf
        peak_day = str(td[td["cens"]].index[0].date())
    else:
        i = td["z"].idxmax()
        alt, lo, hi = float(td.loc[i, "z"]), float(td.loc[i, "lo"]), float(td.loc[i, "hi"])
        peak_day = str(i.date())
    # 커버리지: 피크 구간(7/15-9/10) 관측일 비율
    pk = td.loc[f"{year}{PEAK[0]}":f"{year}{PEAK[1]}"]
    peak_days = int(len(pk))
    if peak_days >= 40 and td.index.max() >= pd.Timestamp(f"{year}-09-05"):
        cov = "full"
    elif len(td) and td.index.min() > pd.Timestamp(f"{year}-08-01"):
        cov = "peak_tail"     # 피크 직후 꼬리만 관측(근사 최대)
    elif len(td) and td.index.max() < pd.Timestamp(f"{year}-09-10"):
        cov = "partial_end"   # 피크 전 종료(과소 추정 가능)
    else:
        cov = "partial"
    return dict(year=int(year), alt_cm=round(alt, 1), right_censored=int(censored),
                alt_lo=round(lo, 1), alt_hi=(np.inf if censored else round(hi, 1)),
                peak_day=peak_day, n_days=int(len(td)), peak_days=peak_days,
                coverage=cov, obs_start=str(td.index.min().date()),
                obs_end=str(td.index.max().date()))


def main():
    profiles = {}
    profiles.update(load_deploy_A())
    profiles.update(load_deploy_B())
    print(f"[parse] 프로파일 {len(profiles)}개 채택, 제외 {len(qc_log['profile_excluded'])}개")

    long_rows, alt_rows = [], []
    for key, (sub, reversed_) in sorted(profiles.items()):
        pid, dep = key.split("_")
        d_max = float(sub.columns.max())
        for dt, r in sub.iterrows():
            for d, v in r.items():
                if np.isfinite(v):
                    long_rows.append((key, pid, dep, str(dt.date()), float(d), round(float(v), 3)))
        for year in sorted(set(sub.index.year)):
            res = season_alt(sub, year)
            if res is None:
                continue
            res.update(site_key=key, profile=pid, deployment=dep,
                       depth_max_cm=d_max, n_layers=len(sub.columns),
                       depth_reversed=int(reversed_))
            alt_rows.append(res)

    long_df = pd.DataFrame(long_rows, columns=["site_key", "profile", "deployment",
                                               "date", "depth_cm", "temp_c"])
    alt_df = pd.DataFrame(alt_rows)
    cols = ["site_key", "profile", "deployment", "year", "alt_cm", "right_censored",
            "alt_lo", "alt_hi", "coverage", "peak_day", "n_days", "peak_days",
            "depth_max_cm", "n_layers", "depth_reversed", "obs_start", "obs_end"]
    alt_df = alt_df[cols].sort_values(["deployment", "profile", "year"])

    long_df.to_csv(OUTDIR / "s7_council_daily_temp.csv", index=False)
    alt_df.to_csv(OUTDIR / "s7_council_alt_derived.csv", index=False)
    meta = dict(
        purpose="S7 KPDC Council 지중온도 프로파일 파싱 + 0C 등온선 ALT 유도",
        source_files=[str(p.relative_to(ROOT)) for p in sorted(RAW.glob("*.xlsx"))],
        council_coords={"lat": 64.85, "lon": -163.70, "source": "kpdc/README.md(파일 내 좌표 부재, KPDC 메타 근사)"},
        depth_provenance=("깊이=파일 내 헤더(cm). 배치 A는 겨울 단조성 검정으로 역순 확증 후 "
                          "(min+max)-d 교정(물리 추론, depth_reversed=1로 전파). 배치 B=워크북 기준본."),
        deployment_break="2024-09 재설치로 배치 A/B 심부 온도 불연속. site_key 분리, 병합 금지.",
        season_window=f"{SEASON[0]}-{SEASON[1]}월", qc=qc_log,
        n_profiles=len(profiles), n_alt_rows=len(alt_df),
        n_profiles_by_deployment={
            "A": int(sum(1 for k in profiles if k.endswith("_A"))),
            "B": int(sum(1 for k in profiles if k.endswith("_B"))),
        },
        n_profiles_note=("QC 통과 프로파일은 site_key 기준 %d종이다(배치 A %d, 배치 B %d). "
                         "배치 A/B는 재설치로 분리하므로 동일 ID라도 별개 프로파일로 센다."
                         % (len(profiles),
                            sum(1 for k in profiles if k.endswith("_A")),
                            sum(1 for k in profiles if k.endswith("_B")))),
        n_censored=int(alt_df.right_censored.sum()),
        smoke=SMOKE, runtime_s=round(time.time() - T0, 1),
    )
    with open(OUTDIR / "s7_parse_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(alt_df.to_string(index=False))
    print(f"[done] {OUTDIR} runtime {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
