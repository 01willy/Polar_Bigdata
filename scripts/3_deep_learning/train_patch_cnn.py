"""공간 DL: 패치 CNN (DEM 패치 이미지 + ERA5/지형 스칼라 하이브리드) 으로 ALT 예측.
공간블록 CV로 GBM(스칼라)과 비교 → 공간맥락(패치)이 도움되는지 정량화. GPU 6 사용.
산출: data/processed/dl_cnn_results.csv, outputs/models/patchcnn_fold*.pt, 예측 CSV
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")   # GPU 6
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

# ---------- 데이터 ----------
df = pd.read_csv("data/processed/dl_dataset.csv").reset_index(drop=True)
patches = np.load("data/processed/dl_patches.npy")        # [n_loc, W, W] float16 (relief)
SCAL = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
W = patches.shape[1]
y = df["y"].to_numpy(np.float32)
alt = df["alt_cm"].to_numpy(np.float32)
locid = df["loc_id"].to_numpy(int)
# 공간블록(~0.5°≈50km) 그룹
block = (np.floor(df.lat / 0.5).astype(int).astype(str) + "_"
         + np.floor(df.lon / 0.5).astype(int).astype(str))
print(f"학습셋 {len(df):,}점, {df.loc_id.nunique():,}위치, 블록 {block.nunique()}개")


class PatchCNN(nn.Module):
    def __init__(self, n_scalar):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp_s = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, patch, scal):
        return self.head(torch.cat([self.cnn(patch), self.mlp_s(scal)], 1)).squeeze(1)


def train_fold(tr, te):
    # 스칼라 표준화(train fit)
    Xs = df[SCAL].to_numpy(np.float32)
    mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
    Xs = (Xs - mu) / sd
    pscale = float(np.abs(patches[locid[tr]]).mean() + 1e-6)   # 패치 스케일
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6

    def batches(idx, bs, shuffle):
        idx = idx.copy()
        if shuffle:
            np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            p = torch.tensor(patches[locid[b]].astype(np.float32) / pscale).unsqueeze(1).to(dev)
            s = torch.tensor(Xs[b]).to(dev)
            t = torch.tensor(((y[b] - ymu) / ysd).astype(np.float32)).to(dev)
            yield p, s, t

    net = PatchCNN(len(SCAL)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss()
    # train/val 분할(train 내 90/10)
    vsel = np.random.rand(len(tr)) < 0.1
    trn, val = tr[~vsel], tr[vsel]
    best, best_state, patience = 1e9, None, 0
    for ep in range(40):
        net.train()
        for p, s, t in batches(trn, 512, True):
            opt.zero_grad(); loss = lossf(net(p, s), t); loss.backward(); opt.step()
        net.eval(); ve = []
        with torch.no_grad():
            for p, s, t in batches(val, 1024, False):
                ve.append(((net(p, s) - t) ** 2).cpu().numpy())
        v = float(np.concatenate(ve).mean())
        if v < best - 1e-4:
            best, best_state, patience = v, {k: x.cpu().clone() for k, x in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 6:
                break
    net.load_state_dict(best_state); net.eval()
    preds = []
    with torch.no_grad():
        for p, s, t in batches(te, 1024, False):
            preds.append(net(p, s).cpu().numpy())
    pred_y = np.concatenate(preds) * ysd + ymu
    pred_cm = np.expm1(np.clip(pred_y, np.log1p(1), np.log1p(600)))
    return pred_cm, net, (mu, sd, pscale, ymu, ysd)


# ---------- 공간블록 CV: CNN vs GBM ----------
gkf = GroupKFold(4)
cnn_err, gbm_err = [], []
for f, (tr, te) in enumerate(gkf.split(df, groups=block)):
    t0 = time.time()
    pred_cnn, net, _ = train_fold(tr, te)
    cnn_err.append(pred_cnn - alt[te])
    # GBM(스칼라만)
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
    g.fit(df[SCAL].to_numpy()[tr], y[tr])
    pg = np.expm1(np.clip(g.predict(df[SCAL].to_numpy()[te]), np.log1p(1), np.log1p(600)))
    gbm_err.append(pg - alt[te])
    torch.save(net.state_dict(), f"outputs/models/patchcnn_fold{f}.pt")
    rc = np.sqrt(np.mean(cnn_err[-1] ** 2)); rg = np.sqrt(np.mean(gbm_err[-1] ** 2))
    print(f"  fold{f}: CNN RMSE={rc:5.1f}  GBM RMSE={rg:5.1f}  (n={len(te)}, {time.time()-t0:.0f}s)")

ce = np.concatenate(cnn_err); ge = np.concatenate(gbm_err)
res = pd.DataFrame([
    dict(model="패치 CNN (DEM패치+ERA5/지형)", rmse=round(float(np.sqrt(np.mean(ce**2))), 1),
         mae=round(float(np.mean(np.abs(ce))), 1)),
    dict(model="GBM (스칼라만)", rmse=round(float(np.sqrt(np.mean(ge**2))), 1),
         mae=round(float(np.mean(np.abs(ge))), 1)),
])
res.to_csv("data/processed/dl_cnn_results.csv", index=False)
print("\n=== 공간블록 CV (북미 ALT, cm) ===")
print(res.to_string(index=False))
