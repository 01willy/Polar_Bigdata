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
# 조건부 생성 모델 3종 (1D 목표 y|x) — 마스터 계획 2026-09-16 §2 G축, H15(UQ)
#   cfm  : 조건부 rectified flow matching (model_tournament.py fit_flow 이식, 조기 종료 추가)
#   ddpm : 조건부 DDPM(eps 예측, T=100)
#   nflow: 조건부 정규화 플로 — 1D 목표에 대한 rational-quadratic spline(Durkan 2019) 2단 합성.
#          z~N(0,1) → y = f_x(z). 학습은 정확한 NLL(역변환 + log|det|).
# 공통: 입력 X는 fold_prep 표준화본, y는 train 평균·SD로 z-score. 표본 S개에서 중앙값(pred)·평균(pred_mean)·
#       분위(q05·q50·q95)를 반환. 점 예측은 사전 등록대로 중앙값.
# ============================================================
_GEN = {"cfm", "ddpm", "nflow"}
_GEN_S = 64          # 셀당 표본 수
_RQS_K = 8           # spline 구간 수
_RQS_B = 5.0         # 꼬리 경계(표준화 y 기준 ±5 SD 밖은 항등)
_RQS_MIN_W = 1e-3
_RQS_MIN_H = 1e-3
_RQS_MIN_D = 1e-3


def _cond_net():
    import torch, torch.nn as nn

    class CondNet(nn.Module):
        """(y_t, t, x) → 스칼라(속도 또는 eps)."""
        def __init__(self, d):
            super().__init__()
            self.xemb = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128))
            self.net = nn.Sequential(nn.Linear(128 + 2, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU(),
                                     nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        def forward(self, yt, t, x):
            h = torch.cat([yt.unsqueeze(-1), t.unsqueeze(-1), self.xemb(x)], -1)
            return self.net(h).squeeze(-1)
    return CondNet


def _rq_spline(inputs, uw, uh, ud, inverse):
    """Unconstrained rational-quadratic spline (1D, 배치). inputs (n,), uw/uh (n,K), ud (n,K-1).
    반환 (outputs, logabsdet). 꼬리(|x|>B)는 항등. nflows 구현을 1D로 축약."""
    import torch
    import torch.nn.functional as F
    B, K = _RQS_B, uw.shape[1]
    out = inputs.clone(); lad = torch.zeros_like(inputs)
    inside = (inputs >= -B) & (inputs <= B)
    if not inside.any():
        return out, lad
    x = inputs[inside]; uw, uh, ud = uw[inside], uh[inside], ud[inside]
    # 경계 도함수 1(항등 꼬리와 연속)
    ud = F.pad(ud, (1, 1), value=float(np.log(np.e - 1)))
    w = F.softmax(uw, -1); w = _RQS_MIN_W + (1 - _RQS_MIN_W * K) * w
    h = F.softmax(uh, -1); h = _RQS_MIN_H + (1 - _RQS_MIN_H * K) * h
    d = _RQS_MIN_D + F.softplus(ud)
    cw = F.pad(torch.cumsum(w, -1), (1, 0), value=0.0) * 2 * B - B; cw[:, 0] = -B; cw[:, -1] = B
    ch = F.pad(torch.cumsum(h, -1), (1, 0), value=0.0) * 2 * B - B; ch[:, 0] = -B; ch[:, -1] = B
    widths = cw[:, 1:] - cw[:, :-1]; heights = ch[:, 1:] - ch[:, :-1]
    delta = heights / widths
    bins = cw if not inverse else ch
    idx = (torch.searchsorted(bins.contiguous(), x[:, None].contiguous(), right=True) - 1).clamp(0, K - 1)[:, 0]
    g = lambda a: a.gather(-1, idx[:, None])[:, 0]
    xk, wk, yk, hk, dk, dk1, delk = g(cw[:, :-1]), g(widths), g(ch[:, :-1]), g(heights), g(d[:, :-1]), g(d[:, 1:]), g(delta)
    if inverse:
        a = (x - yk) * (dk + dk1 - 2 * delk) + hk * (delk - dk)
        b = hk * dk - (x - yk) * (dk + dk1 - 2 * delk)
        c = -delk * (x - yk)
        disc = (b ** 2 - 4 * a * c).clamp_min(0)
        root = (2 * c) / (-b - torch.sqrt(disc))
        y = root * wk + xk
        tr = root * (1 - root)
        den = delk + (dk + dk1 - 2 * delk) * tr
        deriv = delk ** 2 * (dk1 * root ** 2 + 2 * delk * tr + dk * (1 - root) ** 2) / den ** 2
        out[inside] = y; lad[inside] = -torch.log(deriv)
    else:
        th = (x - xk) / wk
        tr = th * (1 - th)
        den = delk + (dk + dk1 - 2 * delk) * tr
        y = yk + hk * (delk * th ** 2 + dk * tr) / den
        deriv = delk ** 2 * (dk1 * th ** 2 + 2 * delk * tr + dk * (1 - th) ** 2) / den ** 2
        out[inside] = y; lad[inside] = torch.log(deriv)
    return out, lad


def _nflow_mods():
    import torch, torch.nn as nn

    class CondSplineFlow(nn.Module):
        """x 조건부 1D 플로: RQ spline 2단 + 마지막 아핀. 각 단은 x 임베딩에서 파라미터 생성."""
        def __init__(self, d, n_layers=2, K=_RQS_K):
            super().__init__()
            self.K = K
            self.emb = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
            self.heads = nn.ModuleList([nn.Linear(128, 3 * K - 1) for _ in range(n_layers)])
            self.aff = nn.Linear(128, 2)
            for hd in self.heads:
                nn.init.zeros_(hd.weight); nn.init.zeros_(hd.bias)   # 항등 초기화
            nn.init.zeros_(self.aff.weight); nn.init.zeros_(self.aff.bias)

        def _params(self, x):
            e = self.emb(x)
            ps = []
            for hd in self.heads:
                p = hd(e)
                ps.append((p[:, :self.K], p[:, self.K:2 * self.K], p[:, 2 * self.K:]))
            a = self.aff(e)
            return ps, a[:, 0], a[:, 1]

        def log_prob(self, y, x):
            """y (n,) 표준화 목표. z = f^{-1}(y), log p(y|x) = log N(z) + log|dz/dy|."""
            ps, mu, ls = self._params(x)
            ls = ls.clamp(-5, 5)
            z = (y - mu) * torch.exp(-ls); lad = -ls
            for (uw, uh, ud) in reversed(ps):
                z, l = _rq_spline(z, uw, uh, ud, inverse=True); lad = lad + l
            return -0.5 * z ** 2 - 0.5 * float(np.log(2 * np.pi)) + lad

        def sample(self, x, S):
            ps, mu, ls = self._params(x)
            ls = ls.clamp(-5, 5)
            n = len(x)
            z = torch.randn(S, n, device=x.device)
            outs = []
            for s in range(S):
                zz = z[s]
                for (uw, uh, ud) in ps:
                    zz, _ = _rq_spline(zz, uw, uh, ud, inverse=False)
                outs.append(mu + torch.exp(ls) * zz)
            return torch.stack(outs, 0)
    return CondSplineFlow


def _fit_generative(name, Xtr, ytr, Xte, seed, epochs, S=_GEN_S):
    import torch
    dev = _dev()
    torch.manual_seed(seed)
    Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    rng = np.random.RandomState(seed); va = rng.rand(len(Xtr)) < 0.1; tr = ~va
    Xt, yt = torch.tensor(Xtr[tr]), torch.tensor(yz[tr])
    Xv, yv = torch.tensor(Xtr[va]).to(dev), torch.tensor(yz[va]).to(dev)
    bs, pat = 8192, 8
    if name == "nflow":
        net = _nflow_mods()(Xtr.shape[1]).to(dev)
        lr = 1e-3
    else:
        net = _cond_net()(Xtr.shape[1]).to(dev)
        lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    T = 200                                            # ᾱ_T ≈ 0.007 (β 1e-4→0.05): 순수 잡음까지 도달(T=100·0.02는 ᾱ_T≈0.37 잔존)
    if name == "ddpm":
        betas = torch.linspace(1e-4, 0.05, T, device=dev); acp = torch.cumprod(1 - betas, 0)

    def loss_fn(xb, yb):
        if name == "cfm":
            y0 = torch.randn_like(yb); t = torch.rand(len(yb), device=dev)
            ytt = (1 - t) * y0 + t * yb
            return ((net(ytt, t, xb) - (yb - y0)) ** 2).mean()
        if name == "ddpm":
            ti = torch.randint(0, T, (len(yb),), device=dev); a = acp[ti]
            eps = torch.randn_like(yb)
            ytt = torch.sqrt(a) * yb + torch.sqrt(1 - a) * eps
            return ((net(ytt, ti.float() / T, xb) - eps) ** 2).mean()
        return -net.log_prob(yb, xb).mean()

    best, state, p = 1e9, None, 0
    pat, min_ep = 12, 15
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]; xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad(); loss = loss_fn(xb, yb)
            if not torch.isfinite(loss):
                break
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():                          # 검증 손실: 고정 잡음 3회 평균(몬테카를로 잡음 억제)
            vs = []
            for rep in range(3):
                torch.manual_seed(seed + 1000 + rep)
                vs.append(np.mean([float(loss_fn(Xv[k:k + bs], yv[k:k + bs])) for k in range(0, len(Xv), bs)]))
            v = float(np.mean(vs))
        if np.isfinite(v) and v < best - 1e-4:
            best, state, p = v, {k2: t_.cpu().clone() for k2, t_ in net.state_dict().items()}, 0
        else:
            p += 1
            if p >= pat and ep >= min_ep:
                break
    if state:
        net.load_state_dict(state)
    net.eval()
    outs = []
    with torch.no_grad():
        torch.manual_seed(seed + 2000)
        for k in range(0, len(Xte), 4096):
            xb = torch.tensor(Xte[k:k + 4096]).to(dev); n = len(xb)
            if name == "nflow":
                y = net.sample(xb, S)
            else:
                xb_e = xb.unsqueeze(0).expand(S, -1, -1).reshape(S * n, -1)
                y = torch.randn(S * n, device=dev)
                if name == "cfm":
                    steps = 20
                    for i in range(steps):
                        t = torch.full((S * n,), i / steps, device=dev)
                        y = y + net(y, t, xb_e) / steps
                else:
                    for ti in reversed(range(T)):
                        a = acp[ti]; b_ = betas[ti]
                        tt = torch.full((S * n,), ti / T, device=dev)
                        eps = net(y, tt, xb_e)
                        y = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
                        if ti > 0:
                            y = y + torch.sqrt(b_) * torch.randn_like(y)
                y = y.reshape(S, n)
            outs.append(y.cpu().numpy())
    samp = np.concatenate(outs, 1) * ysd + ymu                       # (S, n)
    q = np.quantile(samp, [0.05, 0.5, 0.95], axis=0)
    return dict(pred=q[1], pred_mean=samp.mean(0), samples=samp, quantiles=q, val_loss=best)


# ============================================================
# 통합 디스패치
# ============================================================
def available_models():
    """설치된 것만 반환(사용자 지침: 더 나은/가능한 모델은 모두 시도).
    tabpfn은 import 가능해도 가중치 다운로드에 TABPFN_TOKEN(라이선스 승인)이 필요하므로 토큰이 없으면 제외."""
    import os
    out = ["histgbm", "mlp", "ftt", "tabm", "cfm", "ddpm", "nflow"]  # 항상 가능(torch)
    for name, mod in [("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost"),
                      ("realmlp", "pytabkit"), ("tabpfn", "tabpfn")]:
        try:
            __import__(mod)
            if name == "tabpfn" and not os.environ.get("TABPFN_TOKEN"):
                continue
            out.append(name)
        except Exception:
            pass
    return out


def fit_predict(name, Xtr, ytr, Xte, seed=0, epochs=120):
    """모델 이름으로 학습·예측. 반환 dict(pred=np.array, [pred_mean/samples/quantiles])."""
    if name in _TORCH:
        return _fit_torch(name, Xtr, ytr, Xte, seed, epochs)
    if name in _GEN:
        return _fit_generative(name, Xtr, ytr, Xte, seed, epochs)
    if name in NAN_NATIVE:
        return _fit_gbm(name, Xtr, ytr, Xte, seed)
    if name == "realmlp":
        return _fit_realmlp(Xtr, ytr, Xte, seed)
    if name == "tabpfn":
        return _fit_tabpfn(Xtr, ytr, Xte, seed)
    raise ValueError(f"unknown model {name}")
