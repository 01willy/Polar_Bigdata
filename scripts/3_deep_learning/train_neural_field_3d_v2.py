"""B1b: 신경장 개선 1회 시도 — Fourier 깊이 인코딩 + 3-시드 앙상블 + 코사인 LR.
게이트: 사이트분리/지역전이에서 IDW-3·GBM을 이겨야 채택, 아니면 킬스위치(GBM 조건장으로 3D 산출).
산출: data/processed/b1b_results.csv, outputs/models/b1b_neural_field.pt
"""
import os, calendar
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# ---------------- 데이터(B1과 동일) ----------------
g = pd.read_csv("data/processed/ground_temp_all.csv")
g = g[(g.depth_m > 0) & (g.depth_m <= 30) & (g.temp_c > -25) & (g.temp_c < 25)].reset_index(drop=True)
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
elat, elon = clim["latitude"].values, clim["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
t = clim["t2m"].values - 273.15; stl = clim["stl1"].values - 273.15; sdp = clim["sd"].values
tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
E5 = dict(e5_maat=np.nanmean(t, 0), e5_tdd=tdd, e5_fdd=fdd, e5_sqrt_tdd=np.sqrt(tdd),
          e5_twarm=np.nanmax(t, 0), e5_tcold=np.nanmin(t, 0),
          e5_stl1=np.nanmean(stl, 0), e5_swe=np.nanmean(sdp, 0))
iy = np.clip(np.searchsorted(-elat, -g.lat.values), 0, len(elat) - 1)
ix = np.clip(np.searchsorted(elon, g.lon.values), 0, len(elon) - 1)
for k, gr in E5.items():
    g[k] = gr[iy, ix].astype(np.float32)
g = g.dropna(subset=["e5_maat"]).reset_index(drop=True)
E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]

# Fourier 깊이 인코딩: d/30 ∈ [0,1] → [sin,cos](2^k π d̃), k=0..4
dn = (g.depth_m.values / 30.0).astype(np.float32)
FF = []
for k in range(5):
    FF += [np.sin(2 ** k * np.pi * dn), np.cos(2 ** k * np.pi * dn)]
FFn = [f"ff{i}" for i in range(len(FF))]
for n, v in zip(FFn, FF):
    g[n] = v
g["logd"] = np.log1p(g.depth_m)
FEAT = E5F + ["depth_m", "logd"] + FFn
mu = g[FEAT].mean().values.astype(np.float32); sd = g[FEAT].std().values.astype(np.float32) + 1e-6
X = ((g[FEAT].values - mu) / sd).astype(np.float32)
y = g.temp_c.values.astype(np.float32)
ymu, ysd = float(y.mean()), float(y.std()) + 1e-6
print(f"학습점 {len(g):,} / 피처 {len(FEAT)}")


class NF(nn.Module):
    def __init__(self, d, w=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, w), nn.SiLU(),
            nn.Linear(w, w), nn.SiLU(),
            nn.Linear(w, w // 2), nn.SiLU(),
            nn.Linear(w // 2, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def train_one(Xtr, ytr, Xv, yv, seed, epochs=600):
    torch.manual_seed(seed)
    net = NF(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    lossf = nn.SmoothL1Loss()
    Xt = torch.tensor(Xtr); yt = torch.tensor((ytr - ymu) / ysd)
    Xvt = torch.tensor(Xv).to(dev); yvz = (yv - ymu) / ysd
    best, best_state, pat = 1e9, None, 0
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 1024):
            b = idx[k:k + 1024]
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean((net(Xvt).cpu().numpy() - yvz) ** 2))
        if v < best - 1e-4:
            best, best_state, pat = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 50: break
    net.load_state_dict(best_state)
    return net


def predict_ens(nets, Xte):
    ps = []
    for net in nets:
        net.eval()
        with torch.no_grad():
            ps.append(net(torch.tensor(Xte).to(dev)).cpu().numpy() * ysd + ymu)
    return np.mean(ps, 0)


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def profile_baselines(gtr, gte):
    sites = gtr.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    prof = {s: (d.depth_m.values, d.temp_c.values)
            for s, d in gtr.sort_values("depth_m").groupby("site")}
    p_idw = np.full(len(gte), np.nan)
    te_sites = gte.groupby("site").agg(lat=("lat", "first"), lon=("lon", "first"))
    for s, r in te_sites.iterrows():
        dist = haversine(r.lat, r.lon, sites.lat.values, sites.lon.values)
        order = np.argsort(dist)[:3]
        m = (gte.site == s).values
        dq = gte.depth_m.values[m]
        preds3, w3 = [], []
        for oi in order:
            dd, tt = prof[sites.site.values[oi]]
            preds3.append(np.interp(dq, dd, tt)); w3.append(1.0 / max(dist[oi], 0.5))
        w = np.array(w3); w /= w.sum()
        p_idw[m] = np.sum(np.array(preds3) * w[:, None], axis=0)
    return p_idw


def evaluate(split_iter, tag):
    errs = {k: [] for k in ["nf", "gbm", "idw"]}
    for f, (tr, te) in enumerate(split_iter):
        rng = np.random.RandomState(f)
        vs = rng.rand(len(tr)) < 0.12
        nets = [train_one(X[tr][~vs], y[tr][~vs], X[tr][vs], y[tr][vs], seed=s) for s in range(3)]
        pnf = predict_ens(nets, X[te])
        gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                            random_state=0).fit(X[tr], y[tr])
        pg = gbm.predict(X[te])
        pi = profile_baselines(g.iloc[tr], g.iloc[te])
        for k, p in [("nf", pnf), ("gbm", pg), ("idw", pi)]:
            errs[k].append(p - y[te])
        print(f"  [{tag}] fold{f}: NF(3앙상블+FF) {np.sqrt(np.mean(errs['nf'][-1]**2)):5.2f}  "
              f"GBM {np.sqrt(np.mean(errs['gbm'][-1]**2)):5.2f}  "
              f"IDW3 {np.sqrt(np.mean(errs['idw'][-1]**2)):5.2f}  (n={len(te)})")
    return errs

print("\n== 사이트분리 CV ==")
np.random.seed(0)
errs_site = evaluate(GroupKFold(5).split(X, groups=g.site.values), "site")

print("\n== 지역분리 전이 ==")
REG = dict(
    north_america=(g.lon.between(-170, -100) & g.lat.between(50, 75)),
    alps=(g.lon.between(4, 17) & g.lat.between(43, 48.5)),
    russia_asia=(g.lon.between(30, 180) & g.lat.between(45, 80)),
)
splits = [(np.where(~m.values)[0], np.where(m.values)[0]) for m in REG.values()
          if m.sum() > 30]
errs_loro = evaluate(iter(splits), "loro")

def rmse(x): x = np.concatenate(x); return round(float(np.sqrt(np.mean(x[np.isfinite(x)] ** 2))), 2)
res = pd.DataFrame([
    dict(split="사이트분리", model="NF v2 (FF+3앙상블)", rmse=rmse(errs_site["nf"])),
    dict(split="사이트분리", model="GBM", rmse=rmse(errs_site["gbm"])),
    dict(split="사이트분리", model="IDW-3", rmse=rmse(errs_site["idw"])),
    dict(split="지역전이", model="NF v2 (FF+3앙상블)", rmse=rmse(errs_loro["nf"])),
    dict(split="지역전이", model="GBM", rmse=rmse(errs_loro["gbm"])),
    dict(split="지역전이", model="IDW-3", rmse=rmse(errs_loro["idw"])),
])
res.to_csv("data/processed/b1b_results.csv", index=False)

# 전체 학습본 저장(시각화용 — 게이트 결과와 무관하게 비교용 보존)
rng = np.random.RandomState(0)
vs = rng.rand(len(X)) < 0.1
nets = [train_one(X[~vs], y[~vs], X[vs], y[vs], seed=s) for s in range(3)]
os.makedirs("outputs/models", exist_ok=True)
torch.save(dict(states=[{k: t.cpu() for k, t in n.state_dict().items()} for n in nets],
                mu=mu, sd=sd, ymu=ymu, ysd=ysd, feat=FEAT),
           "outputs/models/b1b_neural_field.pt")
print("\n=== B1b (지중온도 RMSE, °C) — B1: NF 2.17/2.05, GBM 1.33/1.43, IDW 1.30/1.69 ===")
print(res.to_string(index=False))
