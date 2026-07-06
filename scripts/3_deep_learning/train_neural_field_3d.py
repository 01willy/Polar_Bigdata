"""B1: 조건부 3D 신경장 파일럿 — T(공변량, 깊이) 연속장 학습.
 - 데이터: ground_temp_all.csv (260 사이트, 0-30m, 이상치 필터)
 - 모델: 신경장 MLP(SiLU) vs GBM vs 최근접시추공 프로파일 vs IDW-3
 - 검증: (1) 사이트분리 GroupKFold(5)  (2) 지역분리 전이(북미/알프스/러시아 홀드아웃)
산출: data/processed/b1_neural_field_results.csv, b1_fold_predictions.csv,
      outputs/models/b1_neural_field.pt (전체 학습본, 3D 시각화용)
"""
import os, calendar
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# ---------------- 데이터 + ERA5 공변량 ----------------
g = pd.read_csv("data/processed/ground_temp_all.csv")
g = g[(g.depth_m > 0) & (g.depth_m <= 30) & (g.temp_c > -25) & (g.temp_c < 25)].reset_index(drop=True)

ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
elat, elon = clim["latitude"].values, clim["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
t = clim["t2m"].values - 273.15
stl = clim["stl1"].values - 273.15
sdp = clim["sd"].values
tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
E5 = dict(e5_maat=np.nanmean(t, 0), e5_tdd=tdd, e5_fdd=fdd, e5_sqrt_tdd=np.sqrt(tdd),
          e5_twarm=np.nanmax(t, 0), e5_tcold=np.nanmin(t, 0),
          e5_stl1=np.nanmean(stl, 0), e5_swe=np.nanmean(sdp, 0))
iy = np.clip(np.searchsorted(-elat, -g.lat.values), 0, len(elat) - 1)
ix = np.clip(np.searchsorted(elon, g.lon.values), 0, len(elon) - 1)
for k, gr in E5.items():
    g[k] = gr[iy, ix].astype(np.float32)
g = g.dropna(subset=["e5_maat"]).reset_index(drop=True)
g["logd"] = np.log1p(g.depth_m)
FEAT = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe",
        "depth_m", "logd"]
print(f"학습점 {len(g):,} / 사이트 {g.site.nunique()} (ERA5 부착 후)")

mu = g[FEAT].mean().values.astype(np.float32); sd = g[FEAT].std().values.astype(np.float32) + 1e-6
X = ((g[FEAT].values - mu) / sd).astype(np.float32)
y = g.temp_c.values.astype(np.float32)
ymu, ysd = float(y.mean()), float(y.std()) + 1e-6


class NF(nn.Module):
    """조건부 신경장: 공변량+깊이 → 온도. SiLU로 매끄러운 수직 프로파일."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def train_nf(Xtr, ytr, Xv, yv, epochs=400, bs=1024, lr=1e-3):
    net = NF(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss()
    Xt = torch.tensor(Xtr); yt = torch.tensor((ytr - ymu) / ysd)
    Xvt = torch.tensor(Xv).to(dev); yvz = (yv - ymu) / ysd
    best, best_state, pat = 1e9, None, 0
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean((net(Xvt).cpu().numpy() - yvz) ** 2))
        if v < best - 1e-4:
            best, best_state, pat = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 20: break
    net.load_state_dict(best_state)
    return net


def predict_nf(net, Xte):
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte).to(dev)).cpu().numpy() * ysd + ymu


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def profile_baselines(gtr, gte):
    """최근접시추공 & IDW-3: 훈련 사이트 프로파일을 깊이 보간해 예측."""
    sites = gtr.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    prof = {s: (d.depth_m.values, d.temp_c.values)
            for s, d in gtr.sort_values("depth_m").groupby("site")}
    p_near = np.full(len(gte), np.nan); p_idw = np.full(len(gte), np.nan)
    te_sites = gte.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first"))
    for s, r in te_sites.iterrows():
        dist = haversine(r.lat, r.lon, sites.lat.values, sites.lon.values)
        order = np.argsort(dist)[:3]
        m = (gte.site == s).values
        dq = gte.depth_m.values[m]
        preds3, w3 = [], []
        for j, oi in enumerate(order):
            dd, tt = prof[sites.site.values[oi]]
            pv = np.interp(dq, dd, tt)
            if j == 0:
                p_near[m] = pv
            preds3.append(pv); w3.append(1.0 / max(dist[oi], 0.5))
        w = np.array(w3); w /= w.sum()
        p_idw[m] = np.sum(np.array(preds3) * w[:, None], axis=0)
    return p_near, p_idw


def evaluate(split_iter, tag):
    errs = {k: [] for k in ["nf", "gbm", "near", "idw"]}
    rows = []
    for f, (tr, te) in enumerate(split_iter):
        vs = np.random.rand(len(tr)) < 0.12
        net = train_nf(X[tr][~vs], y[tr][~vs], X[tr][vs], y[tr][vs])
        pnf = predict_nf(net, X[te])
        gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                            random_state=0).fit(X[tr], y[tr])
        pg = gbm.predict(X[te])
        pn, pi = profile_baselines(g.iloc[tr], g.iloc[te])
        for k, p in [("nf", pnf), ("gbm", pg), ("near", pn), ("idw", pi)]:
            errs[k].append(p - y[te])
        rows.append(pd.DataFrame(dict(split=tag, fold=f, site=g.site.values[te],
                                      lat=g.lat.values[te], lon=g.lon.values[te],
                                      depth_m=g.depth_m.values[te], temp_c=y[te],
                                      pred_nf=pnf, pred_gbm=pg, pred_near=pn, pred_idw=pi)))
        print(f"  [{tag}] fold{f}: NF {np.sqrt(np.mean(errs['nf'][-1]**2)):5.2f}  "
              f"GBM {np.sqrt(np.mean(errs['gbm'][-1]**2)):5.2f}  "
              f"최근접 {np.sqrt(np.mean(errs['near'][-1]**2)):5.2f}  "
              f"IDW3 {np.sqrt(np.mean(errs['idw'][-1]**2)):5.2f}  (n={len(te)})")
    return errs, rows

# (1) 사이트분리 5-fold
print("\n== 사이트분리 CV ==")
gkf = GroupKFold(5)
errs_site, rows_site = evaluate(gkf.split(X, groups=g.site.values), "site")

# (2) 지역분리 전이
print("\n== 지역분리 전이 ==")
REG = dict(
    north_america=(g.lon.between(-170, -100) & g.lat.between(50, 75)),
    alps=(g.lon.between(4, 17) & g.lat.between(43, 48.5)),
    russia_asia=(g.lon.between(30, 180) & g.lat.between(45, 80)),
)
splits = []
for name, m in REG.items():
    te = np.where(m.values)[0]; tr = np.where(~m.values)[0]
    if len(te) > 30: splits.append((tr, te))
errs_loro, rows_loro = evaluate(iter(splits), "loro")

def rmse(x): x = np.concatenate(x); return round(float(np.sqrt(np.mean(x[np.isfinite(x)] ** 2))), 2)
res = pd.DataFrame([
    dict(split="사이트분리", model="신경장 NF", rmse=rmse(errs_site["nf"])),
    dict(split="사이트분리", model="GBM", rmse=rmse(errs_site["gbm"])),
    dict(split="사이트분리", model="최근접 시추공", rmse=rmse(errs_site["near"])),
    dict(split="사이트분리", model="IDW-3", rmse=rmse(errs_site["idw"])),
    dict(split="지역전이", model="신경장 NF", rmse=rmse(errs_loro["nf"])),
    dict(split="지역전이", model="GBM", rmse=rmse(errs_loro["gbm"])),
    dict(split="지역전이", model="최근접 시추공", rmse=rmse(errs_loro["near"])),
    dict(split="지역전이", model="IDW-3", rmse=rmse(errs_loro["idw"])),
])
res.to_csv("data/processed/b1_neural_field_results.csv", index=False)
pd.concat(rows_site + rows_loro, ignore_index=True).to_csv(
    "data/processed/b1_fold_predictions.csv", index=False)

# 전체 학습본(3D 시각화용)
vs = np.random.rand(len(X)) < 0.1
net = train_nf(X[~vs], y[~vs], X[vs], y[vs])
os.makedirs("outputs/models", exist_ok=True)
torch.save(dict(state={k: t.cpu() for k, t in net.state_dict().items()},
                mu=mu, sd=sd, ymu=ymu, ysd=ysd, feat=FEAT),
           "outputs/models/b1_neural_field.pt")

print("\n=== B1 3D 신경장 파일럿 (지중온도 RMSE, °C) ===")
print(res.to_string(index=False))
print("→ outputs/models/b1_neural_field.pt 저장")
