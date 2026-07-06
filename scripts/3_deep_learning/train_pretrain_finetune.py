"""B0: 사전학습→미세조정 DL로 2D ALT 예측. DL이 GBM/CCI를 이기는지 공간블록 CV로 검증.
 - 사전학습: InSAR weak label(수백만) + (있으면)CCI 샘플로 MLP 학습
 - 미세조정: 실측(dl_dataset) fold의 train으로만
 - 비교: 같은 fold에서 GBM(스칼라), CCI 최근접, [DL 미세조정]
GPU: CUDA_VISIBLE_DEVICES로 지정(0-3 중 유휴 1장). 산출: data/processed/b0_pretrain_results.csv
"""
import os, sys, time, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

FEAT = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]

obs = pd.read_csv("data/processed/dl_dataset.csv").reset_index(drop=True)
pre = pd.read_parquet("data/processed/pretrain_weaklabels.parquet")
pre = pre.dropna(subset=FEAT + ["y"]).reset_index(drop=True)
print(f"실측 {len(obs):,} / 사전학습(weak) {len(pre):,}")

# 표준화(사전학습셋 기준)
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
            if v < best - 1e-4: best, best_state, pat = v, {k: t.cpu().clone() for k, t in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 5: break
    if best_state: net.load_state_dict(best_state)
    return net

# 사전학습 1회(공통) — weak label 전량
print("사전학습 시작...")
t0 = time.time()
net0 = MLP(len(FEAT)).to(dev)
opt0 = torch.optim.Adam(net0.parameters(), lr=1e-3, weight_decay=1e-5)
vsel = np.random.rand(len(Xpre)) < 0.05
net0 = run_epochs(net0, opt0, Xpre[~vsel], ypre[~vsel], epochs=15, bs=8192, val=(Xpre[vsel], ypre[vsel]))
pre_state = {k: t.cpu().clone() for k, t in net0.state_dict().items()}
os.makedirs("outputs/models", exist_ok=True)
torch.save(dict(state=pre_state, mu=mu, sd=sd, ymu=float(ymu), ysd=float(ysd), feat=FEAT),
           "outputs/models/b0_mlp_pretrained.pt")
print(f"사전학습 완료 {time.time()-t0:.0f}s → outputs/models/b0_mlp_pretrained.pt")

# CCI baseline: 다년 평균 격자(단위 m→cm, 북미 서브셋으로 메모리 절약)
CCI_A, CCI_LA, CCI_LO = None, None, None
cci_files = sorted(glob.glob("data/raw/cci_alt/*.nc"))
if cci_files:
    import xarray as xr
    ssum = cnt = None
    used = 0
    for f in cci_files:
        try:
            c = xr.open_dataset(f)
            sub = c["ALT"].isel(time=0).sel(lat=slice(50, 75), lon=slice(-175, -95))
            a = sub.values.astype(np.float32)
            if ssum is None:
                ssum = np.zeros_like(a); cnt = np.zeros_like(a)
                CCI_LA, CCI_LO = sub.lat.values, sub.lon.values
            m = np.isfinite(a) & (a > 0) & (a < 10)
            ssum[m] += a[m]; cnt[m] += 1
            used += 1; c.close()
        except Exception as e:
            print("CCI skip:", os.path.basename(f), str(e)[:60])
    if used:
        CCI_A = np.where(cnt > 0, ssum / np.maximum(cnt, 1) * 100.0, np.nan)  # m→cm
        print(f"CCI baseline: {used}개 연도 평균, 격자 {CCI_A.shape}, "
              f"유효셀 중앙값 {np.nanmedian(CCI_A):.0f} cm")

def cci_lookup(lat, lon):
    if CCI_A is None: return None
    iy = np.clip(np.searchsorted(CCI_LA, lat), 0, len(CCI_LA) - 1)   # lat 오름차순
    ix = np.clip(np.searchsorted(CCI_LO, lon), 0, len(CCI_LO) - 1)
    return CCI_A[iy, ix].astype(float)

# 공간블록 CV
gkf = GroupKFold(4)
rows_dl, rows_gbm, rows_cci, pred_rows = [], [], [], []
for f, (tr, te) in enumerate(gkf.split(obs, groups=block)):
    # DL 미세조정
    net = MLP(len(FEAT)).to(dev); net.load_state_dict(pre_state)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
    vs = np.random.rand(len(tr)) < 0.1
    net = run_epochs(net, opt, Xobs[tr][~vs], yobs[tr][~vs], epochs=60, bs=1024,
                     val=(Xobs[tr][vs], yobs[tr][vs]))
    net.eval()
    with torch.no_grad():
        pdl = net(torch.tensor(Xobs[te]).to(dev)).cpu().numpy() * ysd + ymu
    pdl = np.expm1(np.clip(pdl, np.log1p(1), np.log1p(600)))
    rows_dl.append(pdl - alt[te])
    # GBM
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xobs[tr], yobs[tr])
    pg = np.expm1(np.clip(g.predict(Xobs[te]), np.log1p(1), np.log1p(600)))
    rows_gbm.append(pg - alt[te])
    # CCI
    pc = np.full(len(te), np.nan)
    if CCI_A is not None:
        pc = cci_lookup(obs.lat.values[te], obs.lon.values[te])
        m = np.isfinite(pc)
        rows_cci.append(pc[m] - alt[te][m])
    pred_rows.append(pd.DataFrame(dict(fold=f, lat=obs.lat.values[te], lon=obs.lon.values[te],
                                       alt_cm=alt[te], pred_dl=pdl, pred_gbm=pg, pred_cci=pc)))
    rdl = np.sqrt(np.mean(rows_dl[-1]**2)); rg = np.sqrt(np.mean(rows_gbm[-1]**2))
    print(f"  fold{f}: DL(사전학습+미세조정) {rdl:5.1f}  GBM {rg:5.1f}  (n={len(te)})")

def rmse(x): x = np.concatenate(x); return float(np.sqrt(np.mean(x**2)))
res = [dict(model="DL 사전학습+미세조정", rmse=round(rmse(rows_dl), 1)),
       dict(model="GBM (스칼라)", rmse=round(rmse(rows_gbm), 1))]
if rows_cci and sum(len(x) for x in rows_cci) > 0:
    res.append(dict(model="ESA CCI 최근접", rmse=round(rmse(rows_cci), 1)))
pd.DataFrame(res).to_csv("data/processed/b0_pretrain_results.csv", index=False)
pd.concat(pred_rows, ignore_index=True).to_csv("data/processed/b0_fold_predictions.csv", index=False)
print("\n=== B0 공간블록 CV (북미 ALT, cm) ===")
print(pd.DataFrame(res).to_string(index=False))
