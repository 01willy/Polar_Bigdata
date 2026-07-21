"""P1 — 전 공변량·전 지역 통합 셀 단위 6모델 재비교 (2026-07-14 계획).

데이터: dl_dataset_cell_v2.csv (알래스카 14,348 + Lena_RU + QTP_CN + GTNPenv_* 셀).
       위치 동등가중(loc당 1행), 타깃 log1p(alt_cm), 채점 polar.eval_metrics.all_metrics.
피처:  FULL = TERRAIN6 + CLIMATE8 + INSAR5 + POLSAR3 + CCI2 (+insar_miss) = 25.
       결측 모달리티(신규 지역 InSAR/PolSAR 전면 결측): GBM은 NaN 네이티브,
       torch 모델은 fold별 train 중앙값 대체 + 결측 플래그 유지(잔여 NaN→0).
평가:  (1) 공간블록 6-fold(0.5°) + LORO(region, test>=100) × 6모델+앙상블
       (2) GBM 공변량 비교: M3(기후+지형 14) vs M4(+InSAR) vs FULL(25)
       (3) 통합 vs 알래스카특화: within-Alaska 공간블록 / Lena_RU 전이, GBM FULL
출력:  data/processed/unified_tournament_results.csv        (cv×model 총괄)
       data/processed/unified_tournament_perregion.csv      (LORO region×model)
       data/processed/unified_tournament_predictions.csv    (OOF 예측, 지도용)
       data/processed/unified_feature_comparison.csv        (공변량 비교)
       data/processed/unified_vs_alaska_results.csv         (통합/특화 비교)
       data/processed/unified_tournament_meta.json
실행:  cd ROOT && CUDA_VISIBLE_DEVICES=6 /home/anaconda3/bin/python \
       scripts/3_deep_learning/unified_tournament_cell.py   (SMOKE=1 정합성 점검)
"""
import os, sys, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn
from polar.eval_metrics import all_metrics

SMOKE = os.environ.get("SMOKE", "0") == "1"
CELL = os.environ.get("CELL", "data/processed/dl_dataset_cell_v2.csv")
PROC = "data/processed"
CLIP = (np.log1p(1.0), np.log1p(600.0))
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} | SMOKE: {SMOKE} | cell: {CELL}")

df = pd.read_csv(CELL)
if "insar_miss" not in df.columns:  # 구버전 셀 CSV 폴백(스모크 용)
    df["insar_miss"] = (df["insar_dist"] > 5.0).astype(float)
df = df.dropna(subset=["alt_cm", "lat", "lon"]).reset_index(drop=True)

TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
INSAR = ["insar_alt", "insar_alt_std", "insar_sub", "insar_dist", "insar_n"]
POLSAR = ["polsar_alt", "polsar_std", "polsar_valid"]
CCI = ["cci_alt", "cci_valid"]
FLAGS = ["insar_miss"]
FULL = TERRAIN + CLIMATE + INSAR + POLSAR + CCI + FLAGS
M3F = CLIMATE + TERRAIN
M4F = CLIMATE + TERRAIN + INSAR + FLAGS

# 기후+지형은 전지구 커버라 필수(결측 행 제거), 나머지는 NaN 허용
df = df.dropna(subset=M3F).reset_index(drop=True)
y_cm = df["alt_cm"].values.astype(np.float32)
ylog = np.log1p(y_cm).astype(np.float32)
df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int))
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))
ALASKA = {"ABoVE_AK", "ABoVE_CA", "United States (Alaska)", "Canada"}
print(f"[load] {len(df):,}셀 | region: {df.region.value_counts().to_dict()}")

if SMOKE and len(df) > 6000:
    keep = np.sort(np.random.default_rng(0).choice(len(df), 6000, replace=False))
    # 신규 지역은 전량 유지(전이 fold 보존), 알래스카만 표본
    new_idx = np.where(~df.region.isin(ALASKA))[0]
    keep = np.unique(np.concatenate([keep, new_idx]))
    df = df.iloc[keep].reset_index(drop=True)
    y_cm = df["alt_cm"].values.astype(np.float32); ylog = np.log1p(y_cm).astype(np.float32)
    print(f"[smoke] {len(df):,}셀로 축소")


# ---------------- 결측 처리(모델별 입력 분리) ----------------
# GBM도 torch와 동일한 train중앙값 대체 입력을 쓴다(공정성 + 결측 라우팅 아티팩트 방지).
# NaN 네이티브 GBM은 다지역 LORO에서 "InSAR 결측=GTNPenv 고ALT" 라우팅을 학습해
# 레나 전이가 파탄(82cm)남 — Part 3 진단에서 두 입력을 병기해 문서화한다.
def prep_fold(feats, tr, te, gbm_nan_native=False):
    X = df[feats].values.astype(np.float32)
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)      # train 전면 결측 피처 → 0
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-6
    Z = ((Xi - mu) / sd).astype(np.float32)
    Xg = X if gbm_nan_native else Xi
    return Xg[tr], Xg[te], Z[tr], Z[te]


# ---------------- 공통 학습 유틸(model_tournament.py 이식) ----------------
def epochs_fit(net, Xtr, ytr, Xva, yva, epochs, bs=8192, lr=1e-3, wd=1e-5, pat=6, lossf=None):
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


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class FTTransformer(nn.Module):
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
        tok = x.unsqueeze(-1) * self.W + self.b
        z = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
        z = self.enc(z)
        return self.head(z[:, 0]).squeeze(-1)


class TabM(nn.Module):
    def __init__(self, d, k=8):
        super().__init__()
        self.k = k
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
                                   nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
                                    for _ in range(k)])
    def forward(self, x):
        h = self.trunk(x)
        return torch.stack([head(h).squeeze(-1) for head in self.heads], 0).mean(0)
    def all_heads(self, x):
        h = self.trunk(x)
        return torch.stack([head(h).squeeze(-1) for head in self.heads], 0)


class CondNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.xemb = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128))
        self.net = nn.Sequential(nn.Linear(128 + 2, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, yt, t, x):
        h = torch.cat([yt.unsqueeze(-1), t.unsqueeze(-1), self.xemb(x)], -1)
        return self.net(h).squeeze(-1)


def fit_gbm(Xtr, ytr, Xte):
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                      l2_regularization=1.0, early_stopping=True,
                                      random_state=0).fit(Xtr, ytr)
    return dict(pred=g.predict(Xte))


def _val_split(n, frac=0.1, seed=0):
    v = np.random.RandomState(seed).rand(n) < frac
    return ~v, v


def fit_torch_point(model_ctor, Xtr, ytr, Xte, ymu, ysd, epochs, lr=1e-3, seeds=2):
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
    H = np.concatenate(heads, 1) * ysd + ymu
    return dict(pred=H.mean(0), samples=H)


def fit_flow(Xtr, ytr, Xte, ymu, ysd, epochs, S=64):
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    net = CondNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(Xtr); yt = torch.tensor(yz); bs = 8192
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
    steps = 20; outs = []
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
    samp = np.concatenate(outs, 1) * ysd + ymu
    return dict(pred=samp.mean(0), samples=samp)


def fit_diffusion(Xtr, ytr, Xte, ymu, ysd, epochs, S=64, T=100):
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    betas = torch.linspace(1e-4, 0.02, T, device=dev)
    acp = torch.cumprod(1 - betas, 0)
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


# ---------------- fold 정의 ----------------
def spatial_splits(sub_idx=None):
    idx = np.arange(len(df)) if sub_idx is None else sub_idx
    g = df.block.values[idx]
    return [(idx[tr], idx[te]) for tr, te in GroupKFold(n_splits=6).split(idx, groups=g)]

def loro_splits(min_test=100):
    reg = df.region.values; out = []
    for r in pd.unique(reg):
        te = np.where(reg == r)[0]; tr = np.where(reg != r)[0]
        if len(te) >= min_test:
            out.append((r, tr, te))
    return out


MODELS = ["GBM", "MLP", "FT-Transformer", "TabM", "Flow", "Diffusion", "앙상블(GBM+FT-T)"]
EP = dict(mlp=6, ftt=6, tabm=6, flow=8, diff=6) if SMOKE else dict(mlp=60, ftt=45, tabm=60, flow=80, diff=70)


def run_models(tr, te):
    """한 fold에서 7모델 예측(cm)과 UQ 샘플 반환."""
    Xg_tr, Xg_te, Ztr, Zte = prep_fold(FULL, tr, te)
    ytr = ylog[tr]; ymu, ysd = ytr.mean(), ytr.std() + 1e-6
    out = {}
    out["GBM"] = fit_gbm(Xg_tr, ytr, Xg_te)
    out["MLP"] = fit_torch_point(lambda: MLP(len(FULL)), Ztr, ytr, Zte, ymu, ysd, EP["mlp"])
    out["FT-Transformer"] = fit_torch_point(lambda: FTTransformer(len(FULL)), Ztr, ytr, Zte, ymu, ysd, EP["ftt"], lr=5e-4)
    out["TabM"] = fit_tabm(Ztr, ytr, Zte, ymu, ysd, EP["tabm"])
    out["Flow"] = fit_flow(Ztr, ytr, Zte, ymu, ysd, EP["flow"])
    out["Diffusion"] = fit_diffusion(Ztr, ytr, Zte, ymu, ysd, EP["diff"])
    out["앙상블(GBM+FT-T)"] = dict(pred=0.5 * out["GBM"]["pred"] + 0.5 * out["FT-Transformer"]["pred"])
    return out


# ================= Part 1: 통합 6모델+앙상블 × (공간블록, LORO) =================
t_all = time.time()
pred_store = {}   # (cv, model) -> OOF cm
uqrows = []
for cv_name, folds in [("spatial_block", [(None, tr, te) for tr, te in spatial_splits()]),
                       ("LORO", loro_splits())]:
    for m in MODELS:
        pred_store[(cv_name, m)] = np.full(len(df), np.nan)
    for fname, tr, te in folds:
        t0 = time.time()
        out = run_models(tr, te)
        for m, o in out.items():
            pc = to_cm(o["pred"])
            pred_store[(cv_name, m)][te] = pc
            if "samples" in o:
                sc = to_cm(o["samples"])
                lo, hi = np.percentile(sc, [5, 95], axis=0)
                uqrows.append(dict(cv_type=cv_name, fold=str(fname), model=m,
                                   cov90=float(((y_cm[te] >= lo) & (y_cm[te] <= hi)).mean()),
                                   width90=float((hi - lo).mean()), n=len(te)))
        rmse_f = {m: float(np.sqrt(np.nanmean((to_cm(out[m]['pred']) - y_cm[te]) ** 2))) for m in MODELS}
        print(f"[{cv_name}|{fname}] " + " ".join(f"{m.split('(')[0][:7]}={rmse_f[m]:.1f}" for m in MODELS)
              + f" (n={len(te)}, {time.time()-t0:.0f}s)", flush=True)

rows = []
for (cv_name, m), p in pred_store.items():
    mask = np.isfinite(p)
    d = all_metrics(y_cm[mask], p[mask]); d.update(dict(cv_type=cv_name, model=m))
    uq_m = [u for u in uqrows if u["model"] == m and u["cv_type"] == cv_name]
    if uq_m:
        wsum = sum(u["n"] for u in uq_m)
        d["cov90_pct"] = round(100 * sum(u["cov90"] * u["n"] for u in uq_m) / wsum, 1)
        d["width90_cm"] = round(sum(u["width90"] * u["n"] for u in uq_m) / wsum, 1)
    rows.append(d)
res = pd.DataFrame(rows)
cols = ["cv_type", "model", "n", "rmse_cm", "mae_cm", "bias_cm", "r2", "target_sd_cm",
        "skill_over_mean", "cov90_pct", "width90_cm"]
res = res[[c for c in cols if c in res.columns]].sort_values(["cv_type", "rmse_cm"])
res.to_csv(f"{PROC}/unified_tournament_results.csv", index=False)
print(res.to_string(index=False))

# LORO region×model 상세
prows = []
for r, tr, te in loro_splits():
    for m in MODELS:
        p = pred_store[("LORO", m)][te]
        mask = np.isfinite(p)
        if mask.sum() == 0: continue
        d = all_metrics(y_cm[te][mask], p[mask]); d.update(dict(region=r, model=m))
        prows.append(d)
pd.DataFrame(prows).to_csv(f"{PROC}/unified_tournament_perregion.csv", index=False)

# OOF 예측 저장(지도용)
pdf = df[["loc_id", "lat", "lon", "region", "alt_cm"]].copy() if "loc_id" in df.columns \
    else df[["lat", "lon", "region", "alt_cm"]].copy()
for m in MODELS:
    pdf[f"pred_sb_{m}"] = pred_store[("spatial_block", m)]
    pdf[f"pred_loro_{m}"] = pred_store[("LORO", m)]
pdf.to_csv(f"{PROC}/unified_tournament_predictions.csv", index=False)

# ================= Part 2: GBM 공변량 비교 =================
print("\n=== Part 2: GBM 공변량 비교 (M3 / M4 / FULL) ===")
frows = []
for cv_name, folds in [("spatial_block", [(None, tr, te) for tr, te in spatial_splits()]),
                       ("LORO", loro_splits())]:
    for fset_name, feats in [("M3 기후+지형(14)", M3F), ("M4 +InSAR(20)", M4F), ("FULL 전공변량(25)", FULL)]:
        o = np.full(len(df), np.nan)
        for fname, tr, te in folds:
            Xg_tr, Xg_te, _, _ = prep_fold(feats, tr, te)
            o[te] = to_cm(fit_gbm(Xg_tr, ylog[tr], Xg_te)["pred"])
        mask = np.isfinite(o)
        d = all_metrics(y_cm[mask], o[mask]); d.update(dict(cv_type=cv_name, feature_set=fset_name, nfeat=len(feats)))
        frows.append(d)
        print(f"  [{cv_name}] {fset_name:20s} rmse={d['rmse_cm']:.2f} skill={d['skill_over_mean']*100:.1f}%")
pd.DataFrame(frows).to_csv(f"{PROC}/unified_feature_comparison.csv", index=False)

# ================= Part 3: 통합 vs 알래스카특화 (GBM FULL) =================
print("\n=== Part 3: 통합 vs 알래스카특화 ===")
ak_idx = np.where(df.region.isin(ALASKA))[0]
new_idx = np.where(~df.region.isin(ALASKA))[0]
vrows = []
# (a) within-Alaska: 알래스카 공간블록 6-fold, train에 신규지역 포함 여부 비교
ak_folds = spatial_splits(ak_idx)
for variant, extra in [("알래스카특화", None), ("통합(+신규지역)", new_idx)]:
    o = np.full(len(df), np.nan)
    for tr, te in ak_folds:
        tr2 = tr if extra is None else np.concatenate([tr, extra])
        Xg_tr, Xg_te, _, _ = prep_fold(FULL, tr2, te)
        o[te] = to_cm(fit_gbm(Xg_tr, ylog[tr2], Xg_te)["pred"])
    mask = np.isfinite(o)
    d = all_metrics(y_cm[mask], o[mask]); d.update(dict(test="Alaska(공간블록)", train=variant))
    vrows.append(d)
    print(f"  test=Alaska  train={variant:14s} rmse={d['rmse_cm']:.2f} skill={d['skill_over_mean']*100:.1f}%")
# (b) Lena_RU 전이: train=알래스카만 vs 전체(레나 제외) × 결측 입력 방식 진단
lena = np.where(df.region == "Lena_RU")[0]
if len(lena) >= 100:
    for variant, tr in [("알래스카특화", ak_idx), ("통합(레나 제외 전체)", np.where(df.region != "Lena_RU")[0])]:
        for nanmode, tag in [(False, "중앙값대체"), (True, "NaN네이티브")]:
            Xg_tr, Xg_te, _, _ = prep_fold(FULL, tr, lena, gbm_nan_native=nanmode)
            p = to_cm(fit_gbm(Xg_tr, ylog[tr], Xg_te)["pred"])
            d = all_metrics(y_cm[lena], p)
            d.update(dict(test="Lena_RU(전이)", train=f"{variant}/{tag}"))
            vrows.append(d)
            print(f"  test=Lena_RU train={variant:14s}/{tag:7s} rmse={d['rmse_cm']:.2f} skill={d['skill_over_mean']*100:.1f}%")
pd.DataFrame(vrows).to_csv(f"{PROC}/unified_vs_alaska_results.csv", index=False)

meta = dict(cell=CELL, n_cells=int(len(df)), features=FULL, n_feat=len(FULL), smoke=SMOKE,
            regions={k: int(v) for k, v in df.region.value_counts().items()},
            loro_regions=[r for r, _, _ in loro_splits()],
            epochs=EP, elapsed_s=round(time.time() - t_all, 1))
with open(f"{PROC}/unified_tournament_meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"\n[done] {time.time()-t_all:.0f}s | 산출 5종 CSV + meta 저장")
