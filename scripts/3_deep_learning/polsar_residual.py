"""A(강한 base 제품 잔차보정): ABoVE PolSAR ALT 30m(문헌 RMSE 11-12cm)을 base로,
우리 DL이 잔차를 보정하면 17cm를 깨는가? 동일 공간블록 6-fold CV.
 - 학습점에 PolSAR ALT(최근접+5px 창 통계, 2014/15/17 평균) 부착.
 - 비교: ①PolSAR 그대로(base) ②GBM/Diff(+PolSAR 특징) ③잔차보정(PolSAR + DL(obs-PolSAR)).
 - PolSAR 커버리지(북부 알래스카) 있는 점만 대상 — framing(a)가 유효한 범위로 정직히 한정.
산출: data/processed/polsar_residual_results.csv, dl_dataset_polsar.csv
"""
import os, glob, time
import numpy as np
import pandas as pd
import rasterio
import pyproj
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = (np.log1p(1), np.log1p(600))
BASE = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]

obs = pd.read_csv("data/processed/dl_dataset.csv").dropna(subset=BASE + ["alt_cm"]).reset_index(drop=True)
loc = obs.groupby("loc_id").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()

# ---------------- PolSAR ALT 부착(최근접 + 5px 창 평균/표준편차) ----------------
tifs = sorted(glob.glob("data/raw/polsar_alt/*.tif"))
print("PolSAR COGs:", [os.path.basename(t) for t in tifs])
per_year_val, per_year_std = [], []
for tif in tifs:
    with rasterio.open(tif) as src:
        tr = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tr.transform(loc.lon.values, loc.lat.values)
        fc, fr = (~src.transform) * (np.asarray(xs), np.asarray(ys))
        rows = np.floor(fr).astype(int); cols = np.floor(fc).astype(int)
        H, W = src.height, src.width
        nod = src.nodata
        band = src.read(1)
        val = np.full(len(loc), np.nan); std = np.full(len(loc), np.nan)
        inb = (rows >= 2) & (rows < H - 2) & (cols >= 2) & (cols < W - 2)
        for i in np.where(inb)[0]:
            win = band[rows[i] - 2:rows[i] + 3, cols[i] - 2:cols[i] + 3].astype(float)
            if nod is not None:
                win[win == nod] = np.nan
            win = win * 100.0                     # PolSAR 원자료 m → cm
            win[(win <= 1) | (win > 600)] = np.nan
            if np.isfinite(win).sum() >= 3:
                val[i] = np.nanmean(win); std[i] = np.nanstd(win)
    per_year_val.append(val); per_year_std.append(std)
    print(f"  {os.path.basename(tif)}: 유효 {np.isfinite(val).sum()}/{len(loc)}")

loc["polsar_alt"] = np.nanmean(per_year_val, 0)         # 연도 평균(단위 cm 가정 — 검증)
loc["polsar_std"] = np.nanmean(per_year_std, 0)
loc["polsar_valid"] = np.isfinite(loc.polsar_alt).astype(float)
obs = obs.merge(loc[["loc_id", "polsar_alt", "polsar_std", "polsar_valid"]], on="loc_id", how="left")
obs.to_csv("data/processed/dl_dataset_polsar.csv", index=False)

cov = obs.polsar_valid.mean()
sub = obs[obs.polsar_valid == 1].reset_index(drop=True)
print(f"\nPolSAR 커버리지: {cov*100:.1f}% ({len(sub):,} 점)")
print(f"PolSAR ALT 중앙 {sub.polsar_alt.median():.0f}cm vs 실측 {sub.alt_cm.median():.0f}cm | "
      f"corr r={np.corrcoef(sub.polsar_alt, sub.alt_cm)[0,1]:.3f} | "
      f"PolSAR 단독 RMSE {np.sqrt(np.mean((sub.polsar_alt-sub.alt_cm)**2)):.1f}cm, bias {np.mean(sub.polsar_alt-sub.alt_cm):+.1f}cm")

block = (np.floor(sub.lat / 0.5).astype(int).astype(str) + "_" + np.floor(sub.lon / 0.5).astype(int).astype(str))
alt = sub.alt_cm.values.astype(np.float32)
PSF = ["polsar_alt", "polsar_std"]


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


gkf = GroupKFold(6)
E = {k: [] for k in ["base", "gbm14", "gbm_ps", "diff_ps", "resid_gbm", "resid_diff", "ens"]}
for f, (tr, te) in enumerate(gkf.split(sub, groups=block)):
    at = alt[te]
    # ① PolSAR 그대로
    E["base"].append(sub.polsar_alt.values[te] - at)
    # 표준화(train)
    def Z(cols):
        X = sub[cols].values.astype(np.float32); mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        return (X - mu) / sd
    ylog = np.log1p(alt); ymu, ysd = ylog[tr].mean(), ylog[tr].std() + 1e-6
    # ② GBM 14 (참조)
    Z14 = Z(BASE)
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Z14[tr], ylog[tr])
    E["gbm14"].append(np.expm1(np.clip(g.predict(Z14[te]), *CLIP)) - at)
    # ③ +PolSAR 특징
    Zp = Z(BASE + PSF)
    gp = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Zp[tr], ylog[tr])
    pgp = np.expm1(np.clip(gp.predict(Zp[te]), *CLIP)); E["gbm_ps"].append(pgp - at)
    pdp = np.expm1(np.clip(diff_fp(Zp[tr], ylog[tr], Zp[te], ymu, ysd), *CLIP)); E["diff_ps"].append(pdp - at)
    E["ens"].append(0.5 * pgp + 0.5 * pdp - at)
    # ④ 잔차보정: DL이 (obs-polsar)를 14공변량으로 예측 → polsar + resid
    resid = (alt - sub.polsar_alt.values).astype(np.float32)   # cm
    rmu, rsd = resid[tr].mean(), resid[tr].std() + 1e-6
    gr = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Z14[tr], resid[tr])
    E["resid_gbm"].append((sub.polsar_alt.values[te] + gr.predict(Z14[te])) - at)
    # diffusion 잔차(표준화 target)
    dr = diff_fp(Z14[tr], resid[tr], Z14[te], rmu, rsd, epochs=60)  # 여기선 log 아님, cm 잔차 직접
    E["resid_diff"].append((sub.polsar_alt.values[te] + dr) - at)
    R = lambda e: np.sqrt(np.mean(e ** 2))
    print(f"fold{f}: base {R(E['base'][-1]):.1f} gbm14 {R(E['gbm14'][-1]):.1f} +PS {R(E['gbm_ps'][-1]):.1f} "
          f"residGBM {R(E['resid_gbm'][-1]):.1f} residDiff {R(E['resid_diff'][-1]):.1f} (n={len(te)})")

R = lambda e: round(float(np.sqrt(np.mean(np.concatenate(e) ** 2))), 2)
res = pd.DataFrame([
    dict(approach="PolSAR 그대로(base)", rmse_cm=R(E["base"])),
    dict(approach="GBM 14공변량(참조)", rmse_cm=R(E["gbm14"])),
    dict(approach="GBM +PolSAR특징", rmse_cm=R(E["gbm_ps"])),
    dict(approach="Diffusion +PolSAR특징", rmse_cm=R(E["diff_ps"])),
    dict(approach="앙상블 +PolSAR", rmse_cm=R(E["ens"])),
    dict(approach="잔차보정 GBM(PolSAR+ΔGBM)", rmse_cm=R(E["resid_gbm"])),
    dict(approach="잔차보정 Diffusion(PolSAR+ΔDiff)", rmse_cm=R(E["resid_diff"])),
]).sort_values("rmse_cm").reset_index(drop=True)
res.to_csv("data/processed/polsar_residual_results.csv", index=False)
print(f"\n=== A. PolSAR 잔차보정 (북부알래스카 {len(sub):,}점, 공간블록 6-fold CV, cm) ===")
print(res.to_string(index=False))
print("(참조: 전역 GBM 17.24 / 앙상블 16.95. PolSAR 커버 영역만이라 참조 gbm14와 직접비교)")
