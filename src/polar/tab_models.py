"""여러 표형식 회귀 모델 통합 인터페이스 — S1/S6/S11 공용.

사용자 지침: 한 모델로 단정하지 말고 여러 DL·부스팅을 병렬로 비교, 더 나은 게 있으면 채택.
모델군: GBM 3종(LightGBM·XGBoost·CatBoost) + HistGBM + RealMLP + torch(MLP·FT-T·TabM) + TabPFN.

인터페이스:
    from polar.tab_models import available_models, fit_predict, NAN_NATIVE
    out = fit_predict("lightgbm", Xtr, ytr, Xte, seed=0)   # dict(pred=..., [samples=...])

전처리 책임: NAN_NATIVE 모델은 X에 NaN 허용, 그 외는 호출자가 fold-safe median 대체 후 전달.
"""
from __future__ import annotations
import numpy as np

# CUDA_VISIBLE_DEVICES 존중을 위해 import 시 torch.cuda를 절대 초기화하지 않음(lazy).
# (모듈 레벨에서 is_available()을 부르면 그 시점에 보이는 전체 GPU로 CUDA context가 잡혀,
#  이후 설정한 CUDA_VISIBLE_DEVICES가 무시되어 엉뚱한 GPU를 쓰게 됨 — 실제 사고 재발 방지.)
DEV = None


def _dev():
    global DEV
    if DEV is None:
        import torch
        DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
    return DEV


def set_device(d):
    global DEV
    DEV = d


# NaN을 native 처리하는 모델(결측 대체 불필요). 그 외는 imputed X 필요.
NAN_NATIVE = {"lightgbm", "xgboost", "catboost", "histgbm"}
_TORCH = {"mlp", "ftt", "tabm"}


# ============================================================
# torch 모델 (unified_tournament_cell.py 재사용)
# ============================================================
def _torch_mods():
    import torch, torch.nn as nn

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
            return self.head(self.enc(z)[:, 0]).squeeze(-1)

    class TabM(nn.Module):
        def __init__(self, d, k=8):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
                                       nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128))
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
                                        for _ in range(k)])
        def _h(self, x): return self.trunk(x)
        def forward(self, x):
            h = self._h(x)
            return torch.stack([hd(h).squeeze(-1) for hd in self.heads], 0).mean(0)
        def all_heads(self, x):
            h = self._h(x)
            return torch.stack([hd(h).squeeze(-1) for hd in self.heads], 0)

    return MLP, FTTransformer, TabM


def _epochs_fit(net, Xtr, ytr, Xva, yva, epochs, bs=8192, lr=1e-3, wd=1e-5, pat=6):
    import torch, torch.nn as nn
    lossf = nn.SmoothL1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); Xv = torch.tensor(Xva).to(_dev())
    best, state, p = 1e9, None, 0
    for _ in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]; xb, yb = Xt[b].to(_dev()), yt[b].to(_dev())
            opt.zero_grad(); nn.SmoothL1Loss()(net(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)  # 발산 방지(TabM full 61cm 사고)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean((net(Xv).cpu().numpy() - yva) ** 2))
        if v < best - 1e-4:
            best, state, p = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
        else:
            p += 1
            if p >= pat: break
    if state: net.load_state_dict(state)
    return net


def _fit_torch(name, Xtr, ytr, Xte, seed, epochs):
    import torch
    MLP, FTT, TabM = _torch_mods()
    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed); va = rng.rand(len(Xtr)) < 0.1; tr = ~va
    ctor = {"mlp": lambda: MLP(Xtr.shape[1]),
            "ftt": lambda: FTT(Xtr.shape[1]),
            "tabm": lambda: TabM(Xtr.shape[1])}[name]
    lr = 5e-4 if name == "ftt" else 1e-3
    net = ctor().to(_dev())
    net = _epochs_fit(net, Xtr[tr], yz[tr], Xtr[va], yz[va], epochs, lr=lr)
    net.eval()
    with torch.no_grad():
        if name == "tabm":
            H = np.concatenate([net.all_heads(torch.tensor(Xte[k:k + 65536]).to(_dev())).cpu().numpy()
                                for k in range(0, len(Xte), 65536)], 1) * ysd + ymu
            return dict(pred=H.mean(0), samples=H)
        p = np.concatenate([net(torch.tensor(Xte[k:k + 65536]).to(_dev())).cpu().numpy()
                            for k in range(0, len(Xte), 65536)])
    return dict(pred=p * ysd + ymu)


# ============================================================
# GBM / RealMLP / TabPFN
# ============================================================
def _fit_gbm(name, Xtr, ytr, Xte, seed):
    if name == "histgbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                           l2_regularization=1.0, early_stopping=True, random_state=seed)
    elif name == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, n_jobs=4, verbose=-1)
    elif name == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             tree_method="hist", random_state=seed, n_jobs=4)
    elif name == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                              random_seed=seed, verbose=0, allow_writing_files=False)
    m.fit(Xtr, ytr)
    return dict(pred=np.asarray(m.predict(Xte)))


def _fit_realmlp(Xtr, ytr, Xte, seed):
    from pytabkit import RealMLP_TD_Regressor
    m = RealMLP_TD_Regressor(random_state=seed, device=_dev(), n_threads=4)
    m.fit(Xtr, ytr)
    return dict(pred=np.asarray(m.predict(Xte)))


def _fit_tabpfn(Xtr, ytr, Xte, seed):
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(device=_dev(), random_state=seed)
    m.fit(Xtr, ytr)
    try:
        q = m.predict(Xte, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
        pred = np.asarray(q[1]); samples = np.stack([np.asarray(x) for x in q])  # (3, n)
        return dict(pred=pred, quantiles=samples)
    except Exception:
        return dict(pred=np.asarray(m.predict(Xte)))


# ============================================================
# 통합 디스패치
# ============================================================
def available_models():
    """설치된 것만 반환(사용자 지침: 더 나은/가능한 모델은 모두 시도)."""
    out = ["histgbm", "mlp", "ftt", "tabm"]  # 항상 가능
    for name, mod in [("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost"),
                      ("realmlp", "pytabkit"), ("tabpfn", "tabpfn")]:
        try:
            __import__(mod); out.append(name)
        except Exception:
            pass
    return out


def fit_predict(name, Xtr, ytr, Xte, seed=0, epochs=120):
    """모델 이름으로 학습·예측. 반환 dict(pred=np.array, [samples/quantiles])."""
    if name in _TORCH:
        return _fit_torch(name, Xtr, ytr, Xte, seed, epochs)
    if name in NAN_NATIVE:
        return _fit_gbm(name, Xtr, ytr, Xte, seed)
    if name == "realmlp":
        return _fit_realmlp(Xtr, ytr, Xte, seed)
    if name == "tabpfn":
        return _fit_tabpfn(Xtr, ytr, Xte, seed)
    raise ValueError(f"unknown model {name}")
