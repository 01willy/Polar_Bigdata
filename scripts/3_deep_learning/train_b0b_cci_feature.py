"""B0b: CCI를 '입력 피처'로 넣어 물리제품 오차보정 학습 + DL/GBM 앙상블.
 - cci_cm(다년평균)·cci_valid 2개 피처 추가 → 16 피처
 - 사전학습(weak 4M) → fold별 미세조정, GBM(16피처), 앙상블 (DL+GBM)/2
산출: data/processed/b0b_results.csv, b0b_fold_predictions.csv
"""
import os, glob, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

BASE = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
FEAT = BASE + ["cci_cm", "cci_valid"]

# ---------------- CCI 다년평균 격자(북미) ----------------
import xarray as xr
ssum = cnt = None
for f in sorted(glob.glob("data/raw/cci_alt/*.nc")):
    try:
        c = xr.open_dataset(f)
        sub = c["ALT"].isel(time=0).sel(lat=slice(50, 75), lon=slice(-175, -95))
        a = sub.values.astype(np.float32)
        if ssum is None:
            ssum = np.zeros_like(a); cnt = np.zeros_like(a)
            CLA, CLO = sub.lat.values, sub.lon.values
        m = np.isfinite(a) & (a > 0) & (a < 10)
        ssum[m] += a[m]; cnt[m] += 1
        c.close()
    except Exception as e:
        print("CCI skip:", os.path.basename(f), str(e)[:50])
CCI_A = np.where(cnt > 0, ssum / np.maximum(cnt, 1) * 100.0, np.nan)
print(f"CCI 격자 {CCI_A.shape}, 중앙값 {np.nanmedian(CCI_A):.0f}cm")

def attach_cci(df):
    iy = np.clip(np.searchsorted(CLA, df.lat.values), 0, len(CLA) - 1)
    ix = np.clip(np.searchsorted(CLO, df.lon.values), 0, len(CLO) - 1)
    v = CCI_A[iy, ix].astype(np.float32)
    ok = np.isfinite(v)
    med = np.nanmedian(v)
    df["cci_cm"] = np.where(ok, v, med)
    df["cci_valid"] = ok.astype(np.float32)
    return ok.mean()

obs = pd.read_csv("data/processed/dl_dataset.csv").reset_index(drop=True)
pre = pd.read_parquet("data/processed/pretrain_weaklabels.parquet")
pre = pre.dropna(subset=BASE + ["y"]).reset_index(drop=True)
print(f"CCI 부착: 실측 {attach_cci(obs):.1%}, weak {attach_cci(pre):.1%} 유효")

mu = pre[FEAT].mean().values.astype(np.float32); sd = pre[FEAT].std().values.astype(np.float32) + 1e-6
def Z(df): return ((df[FEAT].values - mu) / sd).astype(np.float32)
Xpre, ypre = Z(pre), pre["y"].values.astype(np.float32)
ymu, ysd = ypre.mean(), ypre.std() + 1e-6
block = (np.floor(obs.lat / 0.5).astype(int).astype(str) + "_" + np.floor(obs.lon / 0.5).astype(int).astype(str))
Xobs = Z(obs); yobs = obs["y"].values.astype(np.float32); alt = obs["alt_cm"].values


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def run_epochs(net, opt, X, y, epochs, bs=4096, val=None):
    lossf = nn.SmoothL1Loss()
    Xt = torch.tensor(X); yt = torch.tensor((y - ymu) / ysd)
    best, best_state, pat = 1e9, None, 0
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        if val is not None:
            net.eval()
            with torch.no_grad():
                vp = net(torch.tensor(val[0]).to(dev)).cpu().numpy()
            v = np.mean((vp - (val[1] - ymu) / ysd) ** 2)
            if v < best - 1e-4:
                best, best_state, pat = v, {k: t.cpu().clone() for k, t in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 5: break
    if best_state: net.load_state_dict(best_state)
    return net

print("사전학습(16피처) 시작...")
t0 = time.time()
net0 = MLP(len(FEAT)).to(dev)
opt0 = torch.optim.Adam(net0.parameters(), lr=1e-3, weight_decay=1e-5)
vsel = np.random.rand(len(Xpre)) < 0.05
net0 = run_epochs(net0, opt0, Xpre[~vsel], ypre[~vsel], epochs=15, bs=8192, val=(Xpre[vsel], ypre[vsel]))
pre_state = {k: t.cpu().clone() for k, t in net0.state_dict().items()}
os.makedirs("outputs/models", exist_ok=True)
torch.save(dict(state=pre_state, mu=mu, sd=sd, ymu=float(ymu), ysd=float(ysd), feat=FEAT),
           "outputs/models/b0b_mlp_pretrained.pt")
print(f"사전학습 완료 {time.time()-t0:.0f}s")

gkf = GroupKFold(4)
E = {k: [] for k in ["dl", "gbm", "ens", "cci"]}
pred_rows = []
for f, (tr, te) in enumerate(gkf.split(obs, groups=block)):
    net = MLP(len(FEAT)).to(dev); net.load_state_dict(pre_state)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
    vs = np.random.rand(len(tr)) < 0.1
    net = run_epochs(net, opt, Xobs[tr][~vs], yobs[tr][~vs], epochs=60, bs=1024,
                     val=(Xobs[tr][vs], yobs[tr][vs]))
    net.eval()
    with torch.no_grad():
        pdl = net(torch.tensor(Xobs[te]).to(dev)).cpu().numpy() * ysd + ymu
    pdl = np.expm1(np.clip(pdl, np.log1p(1), np.log1p(600)))
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xobs[tr], yobs[tr])
    pg = np.expm1(np.clip(g.predict(Xobs[te]), np.log1p(1), np.log1p(600)))
    pens = 0.5 * pdl + 0.5 * pg
    pcci = obs.cci_cm.values[te].copy(); pcci[obs.cci_valid.values[te] < 1] = np.nan
    for k, p in [("dl", pdl), ("gbm", pg), ("ens", pens)]:
        E[k].append(p - alt[te])
    m = np.isfinite(pcci); E["cci"].append(pcci[m] - alt[te][m])
    pred_rows.append(pd.DataFrame(dict(fold=f, lat=obs.lat.values[te], lon=obs.lon.values[te],
                                       alt_cm=alt[te], pred_dl=pdl, pred_gbm=pg, pred_ens=pens, pred_cci=pcci)))
    print(f"  fold{f}: DL {np.sqrt(np.mean(E['dl'][-1]**2)):5.1f}  GBM {np.sqrt(np.mean(E['gbm'][-1]**2)):5.1f}"
          f"  ENS {np.sqrt(np.mean(E['ens'][-1]**2)):5.1f}")

def rmse(x): x = np.concatenate(x); return round(float(np.sqrt(np.mean(x ** 2))), 1)
res = pd.DataFrame([dict(model="DL+CCI피처 (사전학습+미세조정)", rmse=rmse(E["dl"])),
                    dict(model="GBM+CCI피처", rmse=rmse(E["gbm"])),
                    dict(model="앙상블 (DL+GBM)/2", rmse=rmse(E["ens"])),
                    dict(model="ESA CCI 그대로", rmse=rmse(E["cci"]))])
res.to_csv("data/processed/b0b_results.csv", index=False)
pd.concat(pred_rows, ignore_index=True).to_csv("data/processed/b0b_fold_predictions.csv", index=False)
print("\n=== B0b 공간블록 CV (cm) — CCI를 피처로 ===")
print(res.to_string(index=False))
print("(참고: B0 기존 — DL 18.1 / GBM 17.7 / CCI 20.8)")
