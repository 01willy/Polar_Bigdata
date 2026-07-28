"""Source-aware multi-fidelity 관측모델 (S6, A5 구현) — 신규 모듈.

관측모델: y_s = z(x) + b_s + ε_s,  ε_s ~ N(0, σ_s(x)²).
- 공유 MLP 인코더(trunk)가 잠재 참값 z(x)를 회귀한다(tab_models.MLP trunk 패턴 준용).
- source별 스칼라 bias b_s: direct(s=0)는 0 고정(수준 앵커), 나머지는 학습.
- source별 log σ_s(x) 헤드(이분산): Gaussian NLL이 소스 신뢰도를 자동 가중한다.
- 예측 2모드: (i) z(x) 단독, (ii) test 셀 보조관측(Stefan·CCI)과 정밀도가중 융합.

기존 공유 모듈은 수정하지 않는다(신규 추가만). torch는 lazy import
(CUDA_VISIBLE_DEVICES를 호출 스크립트가 먼저 설정하는 규약 준수).
"""
from __future__ import annotations
import numpy as np

LOGSIG_MIN, LOGSIG_MAX = -4.0, 3.0  # 표준화 단위 log σ clamp(발산·붕괴 방지)


def _dev():
    from polar.tab_models import _dev as td
    return td()


def build_source_aware_net(d_in: int, n_sources: int):
    """공유 trunk + z 헤드 + source별 log σ 헤드 + source별 스칼라 b(직접소스 0 고정)."""
    import torch
    import torch.nn as nn

    class SourceAwareNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(d_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128))
            self.z_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.logsig_head = nn.Linear(128, n_sources)
            self.b_free = nn.Parameter(torch.zeros(n_sources - 1))

        def forward(self, x):
            h = self.trunk(x)
            z = self.z_head(h).squeeze(-1)
            logsig = self.logsig_head(h).clamp(LOGSIG_MIN, LOGSIG_MAX)
            return z, logsig

        def biases(self):
            zero = torch.zeros(1, device=self.b_free.device, dtype=self.b_free.dtype)
            return torch.cat([zero, self.b_free])

    return SourceAwareNet()


def fit_source_aware(X, y_cm, sid, seed=0, epochs=150, bs=8192, lr=1e-3, wd=1e-5,
                     pat=8, device=None):
    """source-aware Gaussian NLL 학습.

    Args:
        X: (n, d) 표준화 완료 float32(호출자가 fold-safe 전처리 책임)
        y_cm: (n,) 관측값(cm, 원단위). direct·pseudo·CCI 혼합
        sid: (n,) 소스 인덱스. 0=direct(b=0 고정), 1..=보조소스
        seed, epochs, bs, lr, wd, pat: 학습 하이퍼파라미터(tab_models._epochs_fit 준용)
    표준화: direct 관측의 mean/std로 전 소스 y를 z-score(fold-safe, train 관측만 입력됨).
    early stopping: direct 관측 10% val의 z 예측 MSE(정확도 기준. σ 보정은 cov90으로 별도 평가).
    Returns:
        dict(net, ymu, ysd)
    """
    import torch
    dev = device or _dev()
    sid = np.asarray(sid)
    n_sources = int(sid.max()) + 1
    ymu = float(np.mean(y_cm[sid == 0]))
    ysd = float(np.std(y_cm[sid == 0]) + 1e-6)
    yz = ((np.asarray(y_cm, float) - ymu) / ysd).astype(np.float32)
    X = np.asarray(X, np.float32)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    didx = np.where(sid == 0)[0]
    va_d = didx[rng.rand(len(didx)) < 0.1]          # direct 10% val
    tr = np.ones(len(X), bool)
    tr[va_d] = False

    net = build_source_aware_net(X.shape[1], n_sources).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X[tr]); yt = torch.tensor(yz[tr])
    st = torch.tensor(sid[tr].astype(np.int64))
    Xv = torch.tensor(X[va_d]).to(dev); yv = yz[va_d]
    best, state, p = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            bidx = idx[k:k + bs]
            if len(bidx) < 2:                        # BatchNorm 최소 배치 가드
                continue
            xb = Xt[bidx].to(dev); yb = yt[bidx].to(dev); sb = st[bidx].to(dev)
            z, ls = net(xb)
            lsb = ls.gather(1, sb[:, None]).squeeze(1)
            mu = z + net.biases()[sb]
            loss = (0.5 * ((yb - mu) / lsb.exp()) ** 2 + lsb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            zv, _ = net(Xv)
            v = float(np.mean((zv.cpu().numpy() - yv) ** 2))
        if v < best - 1e-4:
            best, state, p = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
        else:
            p += 1
            if p >= pat:
                break
    if state:
        net.load_state_dict(state)
    net.eval()
    return dict(net=net, ymu=ymu, ysd=ysd)


def predict_source_aware(fit, X, device=None, chunk=65536):
    """z(x)·σ_s(x)·b_s를 cm 단위로 반환.

    Returns:
        dict(z=(n,), sig=(n, S), b_cm=(S,))  — sig[:, 0]=direct 잡음 σ(x)
    """
    import torch
    dev = device or _dev()
    net, ymu, ysd = fit["net"], fit["ymu"], fit["ysd"]
    zs, sgs = [], []
    with torch.no_grad():
        for k in range(0, len(X), chunk):
            xb = torch.tensor(np.asarray(X[k:k + chunk], np.float32)).to(dev)
            z, ls = net(xb)
            zs.append(z.cpu().numpy())
            sgs.append(np.exp(ls.cpu().numpy()))
    z = np.concatenate(zs) * ysd + ymu
    sig = np.concatenate(sgs, 0) * ysd
    b_cm = net.biases().detach().cpu().numpy() * ysd
    return dict(z=z, sig=sig, b_cm=b_cm)


def fuse_posterior(z, sig_direct, obs):
    """test 셀 보조관측과의 정밀도가중 융합(가우시안 conjugate 결합).

    prior N(z, σ_d²) × Π_s N(y_s - b_s; z, σ_s²) → posterior N(mean, var).
    Args:
        z, sig_direct: (n,) 인코더 예측·직접소스 σ(x)
        obs: [(y_s, b_s, sig_s, valid_mask), ...]  y_s·sig_s (n,), b_s 스칼라(cm)
    주의: σ_d는 직접관측 잡음이지 z(x) 예측분산이 아니므로 OOD에서 과신 가능(정직 보고 대상).
    Returns:
        (post_mean, post_std)
    """
    prec = 1.0 / np.clip(sig_direct, 1e-3, None) ** 2
    num = z * prec
    for y_s, b_s, sig_s, m in obs:
        p = np.where(m, 1.0 / np.clip(sig_s, 1e-3, None) ** 2, 0.0)
        num = num + np.where(m, np.nan_to_num(y_s - b_s), 0.0) * p
        prec = prec + p
    return num / prec, 1.0 / np.sqrt(prec)
