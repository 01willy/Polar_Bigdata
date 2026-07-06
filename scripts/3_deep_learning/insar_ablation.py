"""정확도 지렛대 검증: InSAR(ReSALT 침하/ALT)를 학습에 넣으면 17cm 벽을 깨는가?
병목=지역내 변동, 그 직접 물리신호=InSAR 계절침하. 학습위치 92.8%가 ReSALT 5km내.
- 각 학습위치에 최근접 ReSALT 통계 부착: insar_alt(제품 base), insar_alt_std(sub-grid변동),
  insar_sub(침하), insar_dist(거리), insar_n(밀도), insar_miss(결측플래그).
- 동일 6-fold 공간블록 CV로 BASE(14) vs +InSAR(20) 비교. 모델: GBM, Diffusion, 앙상블.
- framing(a) 잔차보정 baseline도: 'insar_alt 최근접값 그대로' 오차.
산출: data/processed/insar_ablation_results.csv, dl_dataset_insar.csv
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = (np.log1p(1), np.log1p(600))
BASE = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
INSAR = ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n", "insar_miss"]

obs = pd.read_csv("data/processed/dl_dataset.csv").dropna(subset=BASE + ["alt_cm"]).reset_index(drop=True)

# ---------------- InSAR 최근접 통계 부착 ----------------
wl = pd.read_parquet("data/processed/resalt_weaklabels.parquet")
tree = BallTree(np.radians(np.c_[wl.lat.values, wl.lon.values]), metric="haversine")
loc = obs.groupby("loc_id").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
dist, idx = tree.query(np.radians(np.c_[loc.lat.values, loc.lon.values]), k=64)
dist_km = dist * 6371.0
altn = wl.alt_cm.values[idx]; subn = wl.sub_cm.values[idx]
within = dist_km <= 5.0
def stat(vals, fn):
    out = np.full(len(loc), np.nan)
    for i in range(len(loc)):
        m = within[i]
        if m.any(): out[i] = fn(vals[i][m])
    return out
loc["insar_alt"] = stat(altn, np.nanmean)
loc["insar_alt_std"] = stat(altn, lambda v: np.nanstd(v) if len(v) > 1 else 0.0)
loc["insar_sub"] = stat(subn, np.nanmean)
loc["insar_dist"] = dist_km[:, 0]
loc["insar_n"] = within.sum(1).astype(float)
loc["insar_miss"] = (~within.any(1)).astype(float)
# 결측(5km내 없음)은 중앙값 대체 + 플래그 유지
for c in ["insar_alt", "insar_alt_std", "insar_sub"]:
    loc[c] = loc[c].fillna(loc[c].median())
obs = obs.merge(loc[["loc_id"] + INSAR], on="loc_id", how="left")
obs.to_csv("data/processed/dl_dataset_insar.csv", index=False)
print(f"InSAR 부착 완료: 결측(5km내 無) {obs.insar_miss.mean()*100:.1f}%, "
      f"insar_alt 중앙 {obs.insar_alt.median():.0f}cm vs 실측중앙 {obs.alt_cm.median():.0f}cm")
print(f"insar_alt vs 실측 상관: r={np.corrcoef(obs.insar_alt, obs.alt_cm)[0,1]:.3f}")

block = (np.floor(obs.lat / 0.5).astype(int).astype(str) + "_" + np.floor(obs.lon / 0.5).astype(int).astype(str))
alt = obs.alt_cm.values.astype(np.float32); ylog = np.log1p(alt).astype(np.float32)


class CondNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.xemb = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128))
        self.net = nn.Sequential(nn.Linear(130, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, yt, t, x):
        return self.net(torch.cat([yt.unsqueeze(-1), t.unsqueeze(-1), self.xemb(x)], -1)).squeeze(-1)


def diffusion_fit_predict(Xtr, ytr, Xte, ymu, ysd, epochs=60, T=100, S=48):
    betas = torch.linspace(1e-4, 0.02, T, device=dev); acp = torch.cumprod(1 - betas, 0)
    net = CondNet(Xtr.shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(Xtr); yz = torch.tensor((ytr - ymu) / ysd)
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 8192):
            b = idx[k:k + 8192]; xb = Xt[b].to(dev); y0 = yz[b].to(dev)
            ti = torch.randint(0, T, (len(b),), device=dev); a = acp[ti]
            eps = torch.randn_like(y0); ytt = torch.sqrt(a) * y0 + torch.sqrt(1 - a) * eps
            opt.zero_grad(); ((net(ytt, ti.float() / T, xb) - eps) ** 2).mean().backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(Xte), 8192):
            xb = torch.tensor(Xte[k:k + 8192]).to(dev)
            xb_e = xb.unsqueeze(0).expand(S, -1, -1).reshape(S * len(xb), -1)
            y = torch.randn(S * len(xb), device=dev)
            for ti in reversed(range(T)):
                a = acp[ti]; b_ = betas[ti]; tt = torch.full((S * len(xb),), ti / T, device=dev)
                eps = net(y, tt, xb_e); mean = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
                y = mean + (torch.sqrt(b_) * torch.randn_like(y) if ti > 0 else 0)
            out.append((y.reshape(S, len(xb)).cpu().numpy() * ysd + ymu).mean(0))
    return np.concatenate(out)


def run(feats, tag):
    X = obs[feats].values.astype(np.float32)
    gkf = GroupKFold(6)
    eg, ed, en, eb = [], [], [], []
    for f, (tr, te) in enumerate(gkf.split(X, groups=block)):
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ytr = ylog[tr]; ymu, ysd = ytr.mean(), ytr.std() + 1e-6
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Ztr, ytr)
        pg = np.expm1(np.clip(g.predict(Zte), *CLIP))
        pd_ = np.expm1(np.clip(diffusion_fit_predict(Ztr, ytr, Zte, ymu, ysd), *CLIP))
        pe = 0.5 * pg + 0.5 * pd_
        eg.append(pg - alt[te]); ed.append(pd_ - alt[te]); en.append(pe - alt[te])
        # framing(a) baseline: InSAR ALT 그대로(있으면)
        if "insar_alt" in feats:
            eb.append(obs.insar_alt.values[te] - alt[te])
        print(f"  [{tag}] fold{f}: GBM {np.sqrt(np.mean(eg[-1]**2)):.1f}  Diff {np.sqrt(np.mean(ed[-1]**2)):.1f}  ENS {np.sqrt(np.mean(en[-1]**2)):.1f}")
    R = lambda e: round(float(np.sqrt(np.mean(np.concatenate(e)**2))), 2)
    rows = [dict(featureset=tag, model="GBM", rmse_cm=R(eg)),
            dict(featureset=tag, model="Diffusion", rmse_cm=R(ed)),
            dict(featureset=tag, model="앙상블(GBM+Diff)", rmse_cm=R(en))]
    if eb:
        rows.append(dict(featureset=tag, model="InSAR제품 그대로(base)", rmse_cm=R(eb)))
    return rows

t0 = time.time()
res = run(BASE, "BASE(14 공변량)") + run(BASE + INSAR, "+InSAR(20)")
out = pd.DataFrame(res)
out.to_csv("data/processed/insar_ablation_results.csv", index=False)
print(f"\n=== InSAR 절제실험 (공간블록 6-fold CV, cm) / {time.time()-t0:.0f}s ===")
print(out.to_string(index=False))
print("\n(참조: 기존 GBM 17.24 / Diffusion 17.09 / 앙상블 16.95)")
