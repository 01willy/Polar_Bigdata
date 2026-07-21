"""GTN-P 시추공 연최대 온도 envelope → ALT 근사 라벨.

방법
- data/processed/ground_temp_gtnp_global.csv의 깊이별 t_max(연최대 월기후 온도) 사용.
- 사이트 조건: 상부 3m 내 깊이 ≥3개, 상부 양수/하부 음수 0°C 교차 존재.
- 상부 3m 내 envelope에서 가장 깊은 +→- 교차를 깊이 선형보간 = ALT 근사(cm).
  (3m 초과 심부의 교차는 talik·열적 offset 영향으로 ALT로 보기 어려워 배제)
- qc: sampling이 전 깊이 'seasonal'이면 'high', 아니면 'approx'.
- n_obs = 상부 3m 깊이별 원시 월 레코드 수 합(ALLena 규약과 동일한 '원시 관측 합').
- alt_sd: 단일 시추공 행은 NaN(envelope 방식은 연간 SD 미산출, 0으로 기록하지 않음),
  내부 병합 행은 시추공 간 ALT SD. alt_min/alt_max도 병합 행은 시추공 간 min/max.
- 내부 근접 병합: envelope 시추공끼리 0.01° 이내(체비쇼프)면 union-find 클러스터로
  묶어 1행으로 병합(좌표 평균, alt 평균, 기존 데이터 대비 dedup 정책과 대칭).
- 중복 제거: dl_dataset_cell.csv + alt_allena_cell.csv + alt_qtec_cell.csv 좌표와
  round(4) 일치 또는 0.01° 이내(체비쇼프) 근접이면 제외.
- region = 'GTNPenv_' + ISO2 국가 축약.

산출: data/processed/alt_gtnp_envelope_cell.csv
실행: /home/anaconda3/bin/python scripts/1_data_prep/derive_alt_gtnp_envelope.py  (ROOT에서)
"""
import numpy as np
import pandas as pd

SRC = "data/processed/ground_temp_gtnp_global.csv"
OUT = "data/processed/alt_gtnp_envelope_cell.csv"
ISO2 = {"Russia": "RU", "United States": "US", "Svalbard": "SJ", "Canada": "CA",
        "Sweden": "SE", "Switzerland": "CH", "Austria": "AT", "Antarctica": "AQ"}
TOP_M = 3.0

gt = pd.read_csv(SRC)
rows = []
for bid, g in gt.groupby("borehole_id"):
    g = g.dropna(subset=["depth", "t_max"]).sort_values("depth")
    top = g[g.depth <= TOP_M]
    if len(top) < 3:
        continue
    d, t = top.depth.values, top.t_max.values
    alt = None
    for i in range(len(d) - 1):
        if t[i] > 0 and t[i + 1] <= 0:
            alt = d[i] + (d[i + 1] - d[i]) * t[i] / (t[i] - t[i + 1])
    if alt is None:
        continue
    qc = "high" if (g.sampling == "seasonal").all() else "approx"
    rows.append(dict(
        lat=round(float(g.lat.iloc[0]), 4), lon=round(float(g.lon.iloc[0]), 4),
        region="GTNPenv_" + ISO2.get(g.country_name.iloc[0], "XX"),
        alt_cm=round(alt * 100.0, 1), alt_sd=np.nan,
        alt_min=round(alt * 100.0, 1), alt_max=round(alt * 100.0, 1),
        n_obs=int(top.n_obs.sum()),
        year_min=int(g.year_min.min()), year_max=int(g.year_max.max()),
        n_years=int(g.year_max.max() - g.year_min.min() + 1),
        qc=qc, borehole_id=str(int(bid)), site=str(g.site.iloc[0]).strip()))
env = pd.DataFrame(rows)
print(f"[envelope] 교차 존재 borehole {len(env)}개 "
      f"(qc high {int((env.qc == 'high').sum())} / approx {int((env.qc == 'approx').sum())})")
print(f"  alt_cm 5/50/95% = {np.percentile(env.alt_cm, [5, 50, 95]).round(0)}")

# ---------- envelope 내부 근접 병합 (0.01° 체비쇼프, 기존 dedup 정책과 대칭) ----------
la, lo = env.lat.values, env.lon.values
parent = np.arange(len(env))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return int(i)

for i in range(len(env)):
    for j in range(i + 1, len(env)):
        if max(abs(la[i] - la[j]), abs(lo[i] - lo[j])) <= 0.01:
            parent[find(i)] = find(j)
comp = {}
for i in range(len(env)):
    comp.setdefault(find(i), []).append(i)

merged, n_multi = [], 0
for idxs in sorted(comp.values(), key=min):
    g = env.iloc[idxs]
    if len(g) == 1:
        merged.append(g.iloc[0].to_dict())
        continue
    n_multi += 1
    assert g.region.nunique() == 1, f"클러스터 내 region 불일치: {g.region.tolist()}"
    merged.append(dict(
        lat=round(float(g.lat.mean()), 4), lon=round(float(g.lon.mean()), 4),
        region=g.region.iloc[0],
        alt_cm=round(float(g.alt_cm.mean()), 1),
        alt_sd=round(float(g.alt_cm.std(ddof=1)), 1),   # 시추공 간 SD
        alt_min=float(g.alt_cm.min()), alt_max=float(g.alt_cm.max()),
        n_obs=int(g.n_obs.sum()),
        year_min=int(g.year_min.min()), year_max=int(g.year_max.max()),
        n_years=int(g.year_max.max() - g.year_min.min() + 1),
        qc="high" if (g.qc == "high").all() else "approx",
        borehole_id=";".join(g.borehole_id.astype(str)),
        site=g.site.iloc[0]))
    print(f"[merge] {g.site.iloc[0]} {len(g)}공 병합: alt {sorted(g.alt_cm.tolist())} "
          f"-> {merged[-1]['alt_cm']}±{merged[-1]['alt_sd']}cm")
env = pd.DataFrame(merged)[env.columns.tolist()].reset_index(drop=True)
la, lo = env.lat.values, env.lon.values
rem = [(i, j) for i in range(len(env)) for j in range(i + 1, len(env))
       if max(abs(la[i] - la[j]), abs(lo[i] - lo[j])) <= 0.01]
assert not rem, f"병합 후 잔여 내부 근접쌍: {rem}"
print(f"[merge] 내부 근접 클러스터 {n_multi}개 병합 -> {len(env)}행 (잔여 근접쌍 0)")

# ---------- 기존 ALT 위치와 중복 제거 ----------
ex = pd.concat([
    pd.read_csv("data/processed/dl_dataset_cell.csv")[["lat", "lon"]],
    pd.read_csv("data/processed/alt_allena_cell.csv")[["lat", "lon"]],
    pd.read_csv("data/processed/alt_qtec_cell.csv")[["lat", "lon"]],
], ignore_index=True)
exkey = set(zip(ex.lat.round(4), ex.lon.round(4)))
ela, elo = ex.lat.values, ex.lon.values

keep = []
for _, r in env.iterrows():
    if (round(r.lat, 4), round(r.lon, 4)) in exkey:
        keep.append(False)
        continue
    near = (np.abs(ela - r.lat) <= 0.01) & (np.abs(elo - r.lon) <= 0.01)
    keep.append(not bool(near.any()))
env = env[np.array(keep)].reset_index(drop=True)
print(f"[dedup] 기존 ALT 위치(round4·0.01°)와 중복 {int((~np.array(keep)).sum())}개 제거 -> {len(env)}개")
print(env.region.value_counts().to_string())

env.to_csv(OUT, index=False)
print(f"[saved] {OUT}: {len(env)}행")
