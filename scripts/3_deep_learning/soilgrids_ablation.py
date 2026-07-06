"""새 공변량(SoilGrids 토양) 정확도 절제: BASE(14) vs +토양(20). 재프레이밍 아닌 진짜 정보 추가.
토양 유기탄소·용적밀도·점토·모래 = ALT 단열/수분보유의 직접 물리 인자, 현재 완전 결측.
동일 6-fold 공간블록 CV, GBM+Diffusion. 17cm를 낮추는가?
산출: data/processed/soilgrids_ablation_results.csv, dl_dataset_soil.csv
"""
import os, glob
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = (np.log1p(1), np.log1p(600))
BASE = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"

obs = pd.read_csv("data/processed/dl_dataset.csv").dropna(subset=BASE + ["alt_cm"]).reset_index(drop=True)
loc = obs.groupby("loc_id").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
tr = Transformer.from_crs("EPSG:4326", IGH, always_xy=True)
ix_x, ix_y = tr.transform(loc.lon.values, loc.lat.values)

SOIL = []
for tif in sorted(glob.glob("data/raw/soilgrids/*.tif")):
    name = "sg_" + os.path.basename(tif).replace(".tif", "").replace("_mean", "").replace("-", "_")
    with rasterio.open(tif) as src:
        band = src.read(1).astype(float); nod = src.nodata
        fc, fr = (~src.transform) * (np.asarray(ix_x), np.asarray(ix_y))
        r = np.clip(np.floor(fr).astype(int), 0, src.height - 1)
        c = np.clip(np.floor(fc).astype(int), 0, src.width - 1)
        v = band[r, c]
        if nod is not None:
            v[v == nod] = np.nan
    loc[name] = v; SOIL.append(name)
    print(f"{name}: 결측 {np.isnan(v).mean()*100:.1f}%, 중앙 {np.nanmedian(v):.0f}")

for c in SOIL:
    loc[c] = loc[c].fillna(loc[c].median())
obs = obs.merge(loc[["loc_id"] + SOIL], on="loc_id", how="left")
obs.to_csv("data/processed/dl_dataset_soil.csv", index=False)
print(f"토양 {len(SOIL)}층 부착 완료")

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


def diff_fp(Xtr, ytr, Xte, ymu, ysd, epochs=60, T=100, S=48):
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
            xe = xb.unsqueeze(0).expand(S, -1, -1).reshape(S * len(xb), -1)
            y = torch.randn(S * len(xb), device=dev)
            for ti in reversed(range(T)):
                a = acp[ti]; b_ = betas[ti]; tt = torch.full((S * len(xb),), ti / T, device=dev)
                eps = net(y, tt, xe); mean = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
                y = mean + (torch.sqrt(b_) * torch.randn_like(y) if ti > 0 else 0)
            out.append((y.reshape(S, len(xb)).cpu().numpy() * ysd + ymu).mean(0))
    return np.concatenate(out)


def run(feats, tag):
    X = obs[feats].values.astype(np.float32)
    eg, ed, en = [], [], []
    for f, (trn, te) in enumerate(GroupKFold(6).split(X, groups=block)):
        mu = X[trn].mean(0); sd = X[trn].std(0) + 1e-6
        Ztr, Zte = (X[trn] - mu) / sd, (X[te] - mu) / sd
        yt = ylog[trn]; ymu, ysd = yt.mean(), yt.std() + 1e-6
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Ztr, yt)
        pg = np.expm1(np.clip(g.predict(Zte), *CLIP))
        pdd = np.expm1(np.clip(diff_fp(Ztr, yt, Zte, ymu, ysd), *CLIP))
        eg.append(pg - alt[te]); ed.append(pdd - alt[te]); en.append(0.5 * pg + 0.5 * pdd - alt[te])
    R = lambda e: round(float(np.sqrt(np.mean(np.concatenate(e) ** 2))), 2)
    print(f"  [{tag}] GBM {R(eg)}  Diff {R(ed)}  ENS {R(en)}")
    return [dict(featureset=tag, model="GBM", rmse_cm=R(eg)),
            dict(featureset=tag, model="Diffusion", rmse_cm=R(ed)),
            dict(featureset=tag, model="앙상블(GBM+Diff)", rmse_cm=R(en))]


res = run(BASE, "BASE(14)") + run(BASE + SOIL, f"+토양({14+len(SOIL)})")
out = pd.DataFrame(res)
out.to_csv("data/processed/soilgrids_ablation_results.csv", index=False)
print("\n=== SoilGrids 절제실험 (공간블록 6-fold CV, cm) ===")
print(out.to_string(index=False))
print("(참조: 기존 GBM 17.24 / 앙상블 16.95)")
