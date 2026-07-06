"""모델 토너먼트: 현 데이터(dl_dataset 14공변량)로 최신 DL 모델군을 GBM/앙상블과 공정 비교.
동일 공간블록 CV(GroupKFold 0.5°, B0와 동일 fold) · 동일 지표(cm RMSE) · 생성모델은 UQ(90% 커버리지/샤프니스)도.
누설 방지: fold별 train으로만 표준화·학습. 각 모델은 (Xtr_z, ytr_log, Xte_z) → 예측(log). 변환/채점은 하네스가 담당.

모델: GBM(기준) · MLP · FT-Transformer · TabM(멀티헤드 앙상블) · Flow matching · Diffusion(DDPM)
      + 앙상블(GBM+FT-T). TabPFN은 미설치 → 생략(로그 표기).
산출: data/processed/model_tournament_results.csv, model_tournament_predictions.csv
SMOKE=1 이면 1 fold·소표본·소epoch로 빠른 정합성 점검.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

SMOKE = os.environ.get("SMOKE", "0") == "1"
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| SMOKE:", SMOKE)

FEAT = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
CLIP = (np.log1p(1), np.log1p(600))

df = pd.read_csv("data/processed/dl_dataset.csv").dropna(subset=FEAT + ["alt_cm"]).reset_index(drop=True)
if SMOKE:
    df = df.sample(20000, random_state=0).reset_index(drop=True)
block = (np.floor(df.lat / 0.5).astype(int).astype(str) + "_" + np.floor(df.lon / 0.5).astype(int).astype(str))
alt = df.alt_cm.values.astype(np.float32)
ylog = np.log1p(alt).astype(np.float32)
Xraw = df[FEAT].values.astype(np.float32)
print(f"데이터 {len(df):,} 점 / 블록 {block.nunique():,}")


def to_cm(pred_log):
    return np.expm1(np.clip(pred_log, *CLIP))

# ---------------- 공통 학습 유틸 ----------------
def epochs_fit(net, Xtr, ytr, Xva, yva, epochs, bs=8192, lr=1e-3, wd=1e-5, pat=6, lossf=None):
    """표준화된 X, (train기준 표준화된) y로 조기종료 학습."""
    lossf = lossf or nn.SmoothL1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    Xv = torch.tensor(Xva).to(dev)
    best, bs_state, p = 1e9, None, 0
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean((net(Xv).cpu().numpy() - yva) ** 2))
        if v < best - 1e-4:
            best, bs_state, p = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
        else:
            p += 1
            if p >= pat: break
    if bs_state: net.load_state_dict(bs_state)
    return net


# ================= 모델 정의 =================
class MLP(nn.Module):
    def __init__(self, d, out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out))
    def forward(self, x): return self.net(x).squeeze(-1) if self.net[-1].out_features == 1 else self.net(x)


class FTTransformer(nn.Module):
    """수치 피처 토크나이저 + CLS + 트랜스포머 인코더(FT-Transformer, Gorishniy 2021)."""
    def __init__(self, n_feat, d=64, heads=8, blocks=3, ff=128, drop=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_feat, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, blocks)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x):
        tok = x.unsqueeze(-1) * self.W + self.b          # (B, F, d)
        z = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
        z = self.enc(z)
        return self.head(z[:, 0]).squeeze(-1)


class TabM(nn.Module):
    """TabM 정신(효율적 딥앙상블): 공유 트렁크 + k개 독립 헤드 → 평균/표준편차(UQ)."""
    def __init__(self, d, k=8):
        super().__init__()
        self.k = k
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
                                   nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
                                    for _ in range(k)])
    def forward(self, x):
        h = self.trunk(x)
        outs = torch.stack([head(h).squeeze(-1) for head in self.heads], 0)  # (k, B)
        return outs.mean(0)          # 학습 시 평균으로 손실
    def all_heads(self, x):
        h = self.trunk(x)
        return torch.stack([head(h).squeeze(-1) for head in self.heads], 0)   # (k, B)


class CondNet(nn.Module):
    """생성모델용 조건부 신경망: (y_t, t, x) → 스칼라(속도 또는 eps)."""
    def __init__(self, d):
        super().__init__()
        self.xemb = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128))
        self.net = nn.Sequential(nn.Linear(128 + 2, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, yt, t, x):
        h = torch.cat([yt.unsqueeze(-1), t.unsqueeze(-1), self.xemb(x)], -1)
        return self.net(h).squeeze(-1)


# ================= fit_predict 래퍼(각자 log공간 예측 반환) =================
def fit_gbm(Xtr, ytr, Xte):
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    return dict(pred=g.predict(Xte))


def _val_split(n, frac=0.1, seed=0):
    v = np.random.RandomState(seed).rand(n) < frac
    return ~v, v


def fit_torch_point(model_ctor, Xtr, ytr, Xte, ymu, ysd, epochs, lr=1e-3, seeds=2):
    """seed 앙상블(평균)로 학습 분산 축소 — 순위 안정화."""
    yz = (ytr - ymu) / ysd
    preds = []
    for s in range(1 if SMOKE else seeds):
        torch.manual_seed(s)
        tr, va = _val_split(len(Xtr), seed=s)
        net = model_ctor().to(dev)
        net = epochs_fit(net, Xtr[tr], yz[tr], Xtr[va], yz[va], epochs, lr=lr)
        net.eval()
        with torch.no_grad():
            p = []
            for k in range(0, len(Xte), 65536):
                p.append(net(torch.tensor(Xte[k:k + 65536]).to(dev)).cpu().numpy())
        preds.append(np.concatenate(p))
    return dict(pred=np.mean(preds, 0) * ysd + ymu)


def fit_tabm(Xtr, ytr, Xte, ymu, ysd, epochs):
    tr, va = _val_split(len(Xtr))
    yz = (ytr - ymu) / ysd
    net = TabM(Xtr.shape[1]).to(dev)
    net = epochs_fit(net, Xtr[tr], yz[tr], Xtr[va], yz[va], epochs)
    net.eval()
    with torch.no_grad():
        heads = []
        for k in range(0, len(Xte), 65536):
            heads.append(net.all_heads(torch.tensor(Xte[k:k + 65536]).to(dev)).cpu().numpy())
    H = np.concatenate(heads, 1) * ysd + ymu       # (k, n) log공간
    return dict(pred=H.mean(0), samples=H)


def fit_flow(Xtr, ytr, Xte, ymu, ysd, epochs, S=64):
    """조건부 rectified flow matching(y|x, 1D). ODE 적분으로 샘플."""
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    net = CondNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(Xtr); yt = torch.tensor(yz)
    bs = 8192
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]
            xb = Xt[b].to(dev); y1 = yt[b].to(dev)
            y0 = torch.randn_like(y1)
            t = torch.rand(len(b), device=dev)
            ytt = (1 - t) * y0 + t * y1
            u = y1 - y0
            opt.zero_grad(); loss = ((net(ytt, t, xb) - u) ** 2).mean(); loss.backward(); opt.step()
    net.eval()
    steps = 20
    outs = []
    with torch.no_grad():
        for k in range(0, len(Xte), 16384):
            xb = torch.tensor(Xte[k:k + 16384]).to(dev)
            y = torch.randn(S, len(xb), device=dev)
            xb_e = xb.unsqueeze(0).expand(S, -1, -1).reshape(S * len(xb), -1)
            for i in range(steps):
                t = torch.full((S * len(xb),), i / steps, device=dev)
                v = net(y.reshape(-1), t, xb_e).reshape(S, len(xb))
                y = y + v / steps
            outs.append(y.cpu().numpy())
    samp = np.concatenate(outs, 1) * ysd + ymu     # (S, n)
    return dict(pred=samp.mean(0), samples=samp)


def fit_diffusion(Xtr, ytr, Xte, ymu, ysd, epochs, S=64, T=100):
    """조건부 DDPM(y|x, 1D). eps 예측·역확산 샘플."""
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    acp = torch.cumprod(1 - betas, 0)              # ᾱ_t
    net = CondNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(Xtr); yt = torch.tensor(yz); bs = 8192
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]
            xb = Xt[b].to(dev); y0 = yt[b].to(dev)
            ti = torch.randint(0, T, (len(b),), device=dev)
            a = acp[ti]
            eps = torch.randn_like(y0)
            ytt = torch.sqrt(a) * y0 + torch.sqrt(1 - a) * eps
            opt.zero_grad(); loss = ((net(ytt, ti.float() / T, xb) - eps) ** 2).mean()
            loss.backward(); opt.step()
    net.eval()
    outs = []
    with torch.no_grad():
        for k in range(0, len(Xte), 8192):
            xb = torch.tensor(Xte[k:k + 8192]).to(dev)
            xb_e = xb.unsqueeze(0).expand(S, -1, -1).reshape(S * len(xb), -1)
            y = torch.randn(S * len(xb), device=dev)
            for ti in reversed(range(T)):
                a = acp[ti]; b_ = betas[ti]
                tt = torch.full((S * len(xb),), ti / T, device=dev)
                eps = net(y, tt, xb_e)
                mean = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
                y = mean + (torch.sqrt(b_) * torch.randn_like(y) if ti > 0 else 0)
            outs.append(y.reshape(S, len(xb)).cpu().numpy())
    samp = np.concatenate(outs, 1) * ysd + ymu
    return dict(pred=samp.mean(0), samples=samp)


# ================= 토너먼트 실행 =================
EP = dict(mlp=6, ftt=6, tabm=6, flow=8, diff=6) if SMOKE else dict(mlp=60, ftt=45, tabm=60, flow=80, diff=70)
gkf = GroupKFold(2 if SMOKE else 6)   # fold 6 → 순위 안정성↑
res = {m: [] for m in ["GBM", "MLP", "FT-Transformer", "TabM", "Flow matching", "Diffusion", "앙상블(GBM+FT-T)"]}
uq = {m: dict(cov=[], sharp=[]) for m in ["TabM", "Flow matching", "Diffusion"]}
predrows = []

t_all = time.time()
for f, (tr, te) in enumerate(gkf.split(Xraw, groups=block)):
    mu = Xraw[tr].mean(0); sd = Xraw[tr].std(0) + 1e-6
    Ztr, Zte = (Xraw[tr] - mu) / sd, (Xraw[te] - mu) / sd
    ytr = ylog[tr]; ymu, ysd = ytr.mean(), ytr.std() + 1e-6
    at = alt[te]

    out = {}
    out["GBM"] = fit_gbm(Ztr, ytr, Zte)
    out["MLP"] = fit_torch_point(lambda: MLP(len(FEAT)), Ztr, ytr, Zte, ymu, ysd, EP["mlp"])
    out["FT-Transformer"] = fit_torch_point(lambda: FTTransformer(len(FEAT)), Ztr, ytr, Zte, ymu, ysd, EP["ftt"], lr=5e-4)
    out["TabM"] = fit_tabm(Ztr, ytr, Zte, ymu, ysd, EP["tabm"])
    out["Flow matching"] = fit_flow(Ztr, ytr, Zte, ymu, ysd, EP["flow"])
    out["Diffusion"] = fit_diffusion(Ztr, ytr, Zte, ymu, ysd, EP["diff"])
    out["앙상블(GBM+FT-T)"] = dict(pred=0.5 * out["GBM"]["pred"] + 0.5 * out["FT-Transformer"]["pred"])

    row = dict(fold=f, lat=df.lat.values[te], lon=df.lon.values[te], alt_cm=at)
    for m, o in out.items():
        pc = to_cm(o["pred"])
        res[m].append(pc - at)
        row[f"pred_{m}"] = pc
        if "samples" in o and m in uq:
            sc = to_cm(o["samples"])                      # (S/k, n) cm
            lo, hi = np.percentile(sc, [5, 95], axis=0)
            uq[m]["cov"].append(((at >= lo) & (at <= hi)).mean())
            uq[m]["sharp"].append((hi - lo).mean())
    predrows.append(pd.DataFrame(row))
    rmse_f = {m: np.sqrt(np.mean(res[m][-1] ** 2)) for m in res}
    print(f"fold{f}: " + "  ".join(f"{m.split('(')[0][:8]} {rmse_f[m]:4.1f}" for m in res) + f"  (n={len(te)})")

def agg(x):
    x = np.concatenate(x); return float(np.sqrt(np.mean(x ** 2))), float(np.mean(np.abs(x)))

mean_rmse = np.sqrt(np.mean((alt - alt.mean()) ** 2))
rows = []
for m in res:
    r, a = agg(res[m])
    d = dict(model=m, rmse_cm=round(r, 2), mae_cm=round(a, 2),
             skill_vs_mean=round(100 * (1 - r / mean_rmse), 1))
    if m in uq and uq[m]["cov"]:
        d["cov90_%"] = round(100 * np.mean(uq[m]["cov"]), 1)
        d["sharp90_cm"] = round(np.mean(uq[m]["sharp"]), 1)
    rows.append(d)
resdf = pd.DataFrame(rows).sort_values("rmse_cm").reset_index(drop=True)
resdf.to_csv("data/processed/model_tournament_results.csv", index=False)
pd.concat(predrows, ignore_index=True).to_csv("data/processed/model_tournament_predictions.csv", index=False)
# fold별 RMSE 행렬(모델×fold) — 변동성 시각화용
perfold = pd.DataFrame({m: [float(np.sqrt(np.mean(res[m][f] ** 2))) for f in range(len(res[m]))] for m in res})
perfold.index.name = "fold"
perfold.round(2).to_csv("data/processed/model_tournament_perfold.csv")
print(f"\n=== 모델 토너먼트 (공간블록 CV, cm) — 평균예측기 RMSE={mean_rmse:.1f} / {time.time()-t_all:.0f}s ===")
print(resdf.to_string(index=False))
print("\n※ TabPFN-v2는 미설치로 생략(별도 설치 필요). 생성모델은 cov90=90% 근처가 이상적(보정된 UQ).")
