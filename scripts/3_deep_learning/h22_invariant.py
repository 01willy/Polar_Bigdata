"""H22 불변 관계 학습 — 학습 목표 함수만 바꾼다(ERM → IRMv1 · V-REx · GroupDRO · DANN · 지역 임베딩 · 안정 특징 부분집합). GPU 1장.

설계 docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md §5. 채점 규약 개정 09-21.

고정: 입력 x25(SHARED_CORE, fold-safe 중앙값 대체·표준화), 모델 = M1 MLP(256-128-64, BN·Dropout 0.1, SmoothL1, Adam 1e-3, 조기 종료 pat 6),
      앵커 = 알래스카 E(Stefan), 예측 = 앵커 + λ·g(잔차) 또는 직접 g(λ=1), seed 3.
환경: 알래스카 0.5° 블록 6-fold(AK0–AK5) ∪ {레나·캐나다·러시아 W·E} − {대상}. 러시아 C·그린란드는 학습 환경에서 제외(셀 수).
      대상 라벨은 절대 미사용. DANN 은 공변량만 조건에서 대상 A블록 공변량(라벨 없음)을 도메인 판별기에 사용.
초모수(IRM λ, V-REx β, DANN λ_d)는 학습 환경 leave-one-region-environment-out 으로 중첩 선택(λ=0.25 잔차 RMSE 지역 등가중, seed 0).
수준 id: 22-E0 erm · 22-E1 irm · 22-E2 vrex · 22-E3 groupdro · 22-E4 dann · 22-E5 regemb · 22-S stable(CatBoost, 환경 간 안정 특징)

산출 data/processed/h2/h22_<tag>_rows.csv(구성×λ RMSE) · h22_<tag>_preds.npz · h22_<tag>_meta.json
실행: GPU=5 python3 scripts/3_deep_learning/h22_invariant.py --targets Lena,Canada [--smoke]
"""
from __future__ import annotations
import os
GPU = os.environ.get("GPU", "5")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, TERRAIN, CLIMATE, TRANSFER_MAIN, spatial_block_splits                 # noqa: E402
from polar.m1_core import INPUT_SETS, LAM_GRID, load_base, eval_mask, half_split_blocks, fit_coefs          # noqa: E402
from polar.preprocessing import fold_prep                                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="Lena,Canada,Russia_W,Russia_E,Russia_C,Greenland")
ap.add_argument("--methods", default="erm,irm,vrex,groupdro,dann,regemb,stable")
ap.add_argument("--ytypes", default="resid,direct")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--splits", type=int, default=3, help="공변량만(DANN·B 채점) 분할 수")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--be", type=int, default=512, help="환경당 배치 크기")
ap.add_argument("--tag", default="h22")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
PROC = ROOT / "data" / "processed"
OUT = PROC / "h2"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
X14 = list(TERRAIN + CLIMATE)
TARGETS = args.targets.split(","); METHODS = args.methods.split(","); YTYPES = args.ytypes.split(",")
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
EPOCHS = args.epochs if not args.smoke else 8
HP = {"irm": [1.0, 10.0, 100.0], "vrex": [1.0, 10.0, 100.0], "dann": [0.1, 1.0], "erm": [0.0], "groupdro": [0.01], "regemb": [0.0]}
if args.smoke:
    HP = {k: v[:1] for k, v in HP.items()}
REGION_ENVS = ["Lena", "Canada", "Russia_W", "Russia_E"]
t_start = time.time()

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)
AK_IDX = np.where(DF.macro.values == "Alaska")[0]
E_AK = float(fit_coefs(DF.iloc[AK_IDX])["E"])
ENV_ID = np.full(len(DF), "", object)
for fi, (_, te) in enumerate(spatial_block_splits(DF, n_splits=6, sub_idx=AK_IDX)):
    ENV_ID[te] = f"AK{fi}"
for r in REGION_ENVS:
    ENV_ID[DF.macro.values == r] = r
print(f"[data] {len(DF):,}셀 · E_AK {E_AK:.4f} · 환경 {pd.Series(ENV_ID[ENV_ID != '']).value_counts().to_dict()} · GPU {GPU} ({DEV})", flush=True)


# ---------------------------------------------------------------- 모델
class Net(nn.Module):
    def __init__(self, d, n_emb=0, emb_dim=4):
        super().__init__()
        self.emb = nn.Embedding(n_emb, emb_dim) if n_emb else None
        din = d + (emb_dim if n_emb else 0)
        self.trunk = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
                                   nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
                                   nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)
        self.dom = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def feat(self, x, e=None):
        if self.emb is not None:
            x = torch.cat([x, self.emb(e)], 1)
        return self.trunk(x)

    def forward(self, x, e=None):
        return self.head(self.feat(x, e)).squeeze(-1)


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def irm_penalty(loss_fn, pred, y):
    w = torch.tensor(1.0, device=pred.device, requires_grad=True)
    l = loss_fn(pred * w, y)
    g = torch.autograd.grad(l, [w], create_graph=True)[0]
    return (g ** 2).sum()


def train_net(method, hp, envs, seed, X_unl=None, emb_of_env=None, epochs=EPOCHS, be=args.be):
    """envs: list of (X, y) 표준화 배열(환경 순서 고정). X_unl: DANN 대상 도메인 공변량(표준화). 반환 net, ymu, ysd."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    n_env = len(envs)
    yall = np.concatenate([y for _, y in envs]); ymu, ysd = float(yall.mean()), float(yall.std() + 1e-6)
    tr_sets, va_X, va_y = [], [], []
    for X, y in envs:
        va = rng.rand(len(X)) < 0.1
        tr_sets.append((torch.tensor(X[~va], dtype=torch.float32), torch.tensor(((y[~va] - ymu) / ysd), dtype=torch.float32)))
        va_X.append(X[va]); va_y.append((y[va] - ymu) / ysd)
    Xv = torch.tensor(np.concatenate(va_X), dtype=torch.float32).to(DEV); yv = np.concatenate(va_y)
    net = Net(envs[0][0].shape[1], n_emb=n_env if method == "regemb" else 0).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss(); bce = nn.BCEWithLogitsLoss()
    q = torch.ones(n_env, device=DEV) / n_env
    Xu = torch.tensor(X_unl, dtype=torch.float32) if X_unl is not None else None
    steps = max(1, int(sum(len(t[0]) for t in tr_sets) / (be * n_env)))
    best, state, pat = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        warm = ep < epochs // 2
        for _ in range(steps):
            losses, pens = [], []
            for ei, (Xt, yt) in enumerate(tr_sets):
                b = torch.randint(0, len(Xt), (min(be, len(Xt)),))
                xb, yb = Xt[b].to(DEV), yt[b].to(DEV)
                eb = torch.full((len(b),), ei, dtype=torch.long, device=DEV) if method == "regemb" else None
                if method == "dann":
                    f = net.feat(xb); pred = net.head(f).squeeze(-1)
                    bu = torch.randint(0, len(Xu), (min(be, len(Xu)),)); fu = net.feat(Xu[bu].to(DEV))
                    lam_d = hp * (2.0 / (1.0 + np.exp(-10 * ep / max(epochs, 1))) - 1.0)
                    dl = bce(net.dom(GradRev.apply(torch.cat([f, fu]), lam_d)).squeeze(-1),
                             torch.cat([torch.zeros(len(f)), torch.ones(len(fu))]).to(DEV))
                    losses.append(lossf(pred, yb) + dl / n_env)
                    continue
                pred = net(xb, eb)
                l = lossf(pred, yb); losses.append(l)
                if method == "irm":
                    pens.append(irm_penalty(lossf, pred, yb))
            L = torch.stack(losses)
            if method == "irm":
                lam = 1.0 if warm else hp
                loss = L.mean() + lam * torch.stack(pens).mean()
                if lam > 1:
                    loss = loss / lam
            elif method == "vrex":
                loss = L.mean() + (1.0 if warm else hp) * L.var(unbiased=False)
            elif method == "groupdro":
                with torch.no_grad():
                    q = q * torch.exp(hp * L.detach()); q = q / q.sum()
                loss = (q * L).sum()
            else:
                loss = L.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():
            ev = torch.zeros(len(Xv), dtype=torch.long, device=DEV) if method == "regemb" else None
            v = float(np.mean((net(Xv, ev).cpu().numpy() - yv) ** 2))
        if v < best - 1e-4:
            best, state, pat = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 6 and not (method in ("irm", "vrex") and warm):
                break
    if state:
        net.load_state_dict(state)
    net.eval()
    return net, ymu, ysd


def predict(net, X, ymu, ysd, env_idx=None):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(DEV)
        e = torch.full((len(X),), int(env_idx), dtype=torch.long, device=DEV) if env_idx is not None else None
        return net(Xt, e).cpu().numpy() * ysd + ymu


def stable_features(envs_raw, feats, seed=0, top=12, frac=0.6):
    """환경별 CatBoost(잔차) 중요도에서 상위 top 안에 frac 이상 환경에서 드는 특징만(안정 특징)."""
    from catboost import CatBoostRegressor
    ranks = []
    for X, y in envs_raw:
        if len(X) < 30:
            continue
        m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                              allow_writing_files=False, thread_count=4).fit(X, y)
        imp = m.get_feature_importance(); ranks.append(np.argsort(-imp)[:top])
    cnt = np.zeros(len(feats))
    for r in ranks:
        cnt[r] += 1
    keep = np.where(cnt >= frac * len(ranks))[0]
    return keep if len(keep) >= 3 else np.argsort(-cnt)[:top]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# ---------------------------------------------------------------- 실행
rows, PRED, EVAL, META = [], {}, {}, {}
for t in TARGETS:
    t_idx = np.where(DF.macro.values == t)[0]
    if len(t_idx) < 6:
        continue
    env_names = [f"AK{i}" for i in range(6)] + [r for r in REGION_ENVS if r != t]
    tr_idx = np.where(np.isin(ENV_ID, env_names))[0]
    # 조건별 평가 집합: noinfo 전체 / covonly B(분할별) ; DANN 은 A 공변량
    conds = {"noinfo": [(0, t_idx[eval_mask(DF.iloc[t_idx])], None)]}
    if t in REGION_ENVS:
        conds["covonly"] = []
        for sp in SPLITS:
            A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
            conds["covonly"].append((sp, B_idx[eval_mask(DF.iloc[B_idx])], A_idx))
    for cond, lst in conds.items():
        for sp, ev, A_idx in lst:
            key = f"{cond}|{t}|{sp}"
            EVAL[key] = dict(y=DF.y.values[ev], block=DF.block.values[ev], loc_id=DF.loc_id.values[ev])
            PRED[f"anchor::{key}"] = E_AK * DF.s.values[ev]
    Xraw_tr = DF.iloc[tr_idx][FEATS].values.astype(np.float32)
    for ytype in YTYPES:
        y_tr = DF.y.values[tr_idx] - (E_AK * DF.s.values[tr_idx] if ytype == "resid" else 0.0)
        ok = np.isfinite(y_tr)
        for method in METHODS:
            if method == "dann" and t not in REGION_ENVS:
                continue
            for cond, lst in conds.items():
                if method == "dann" and cond != "covonly":
                    continue
                if method != "dann" and cond == "covonly":
                    continue                                   # 비DANN 은 noinfo 모델을 B 셀에 재채점(아래)
                for sp, ev, A_idx in lst:
                    key = f"{cond}|{t}|{sp}"
                    Xtr_all, Xte = fold_prep(Xraw_tr, DF.iloc[ev][FEATS].values.astype(np.float32), nan_native=False)
                    Xunl = None
                    if method == "dann":
                        _, Xunl = fold_prep(Xraw_tr, DF.iloc[A_idx][FEATS].values.astype(np.float32), nan_native=False)
                    env_lab = ENV_ID[tr_idx]
                    envs = [(Xtr_all[(env_lab == e) & ok], y_tr[(env_lab == e) & ok]) for e in env_names]
                    envs = [(X, y) for X, y in zip([e[0] for e in envs], [e[1] for e in envs]) if len(X) >= 20]
                    env_used = [e for e in env_names if ((env_lab == e) & ok).sum() >= 20]
                    if method == "stable":
                        envs_raw = [(Xraw_tr[(env_lab == e) & ok], y_tr[(env_lab == e) & ok]) for e in env_used]
                        keep = stable_features(envs_raw, FEATS)
                        from polar.m1_core import fit_model
                        for seed in SEEDS:
                            g = fit_model("catboost_lo", Xraw_tr[ok][:, keep], y_tr[ok], DF.iloc[ev][FEATS].values.astype(np.float32)[:, keep], seed)["pred"]
                            pk = f"{key}|{ytype}|stable|0|{seed}"; PRED[f"g::{pk}"] = g.astype(np.float32)
                            for lam in (LAM_GRID if ytype == "resid" else [1.0]):
                                p = PRED[f"anchor::{key}"] + lam * g if ytype == "resid" else g
                                rows.append(dict(cond=cond, target=t, split=sp, ytype=ytype, method="stable", hp=0.0, hp_nested=0.0, seed=seed, lam=lam,
                                                 rmse=rmse(EVAL[key]["y"], p), n=len(ev), n_feat=int(len(keep)), feats=",".join(FEATS[i] for i in keep)))
                            for sp2, ev2, _ in conds.get("covonly", []):
                                key2 = f"covonly|{t}|{sp2}"; sel = np.isin(EVAL[key]["loc_id"], EVAL[key2]["loc_id"])
                                PRED[f"g::{key2}|{ytype}|stable|0|{seed}"] = g[sel].astype(np.float32)
                                for lam in (LAM_GRID if ytype == "resid" else [1.0]):
                                    p = PRED[f"anchor::{key2}"] + lam * g[sel] if ytype == "resid" else g[sel]
                                    rows.append(dict(cond="covonly", target=t, split=sp2, ytype=ytype, method="stable", hp=0.0, hp_nested=0.0, seed=seed, lam=lam,
                                                     rmse=rmse(EVAL[key2]["y"], p), n=int(sel.sum()), n_feat=int(len(keep)), feats=""))
                        META[f"{t}|{ytype}|stable_feats"] = [FEATS[i] for i in keep]
                        continue
                    # 중첩 초모수 선택(지역 환경 LOEO, seed 0, λ=.25 잔차 또는 직접)
                    hps = HP[method]; hp_best = hps[0]; lodo = {}
                    if len(hps) > 1:
                        held = [e for e in env_used if e in REGION_ENVS]
                        for hp in hps:
                            sc = []
                            for h in held:
                                inner = [(X, y) for X, y, e in zip([e[0] for e in envs], [e[1] for e in envs], env_used) if e != h]
                                hX, hy = envs[env_used.index(h)]
                                Xu = hX if method == "dann" else None
                                net, ymu, ysd = train_net(method, hp, inner, 0, X_unl=Xu, epochs=EPOCHS)
                                g = predict(net, hX, ymu, ysd)
                                anc = E_AK * DF.s.values[tr_idx][(env_lab == h) & ok] if ytype == "resid" else 0.0
                                yy = DF.y.values[tr_idx][(env_lab == h) & ok]
                                sc.append(rmse(yy, anc + (0.25 if ytype == "resid" else 1.0) * g))
                            lodo[hp] = float(np.mean(sc)) if sc else np.inf
                        hp_best = min(hps, key=lambda h: lodo[h])
                        META[f"{key}|{ytype}|{method}|lodo"] = lodo
                    for hp in hps:                             # 전 초모수 기록(탐색적) + 중첩 선택 표시
                        for seed in SEEDS:
                            tf = time.time()
                            net, ymu, ysd = train_net(method, hp, envs, seed, X_unl=Xunl, epochs=EPOCHS)
                            if method == "regemb":               # 새 지역: 공변량 중앙값이 가장 가까운 지역 환경의 임베딩
                                med = {e: np.nanmedian(Xtr_all[(env_lab == e) & ok], 0) for e in env_used}
                                mt = np.nanmedian(Xte, 0)
                                near = min(env_used, key=lambda e: np.nansum((med[e] - mt) ** 2))
                                g = predict(net, Xte, ymu, ysd, env_idx=env_used.index(near))
                                META[f"{key}|{ytype}|regemb_near"] = near
                            else:
                                g = predict(net, Xte, ymu, ysd)
                            pk = f"{key}|{ytype}|{method}|{hp:g}|{seed}"; PRED[f"g::{pk}"] = g.astype(np.float32)
                            for lam in (LAM_GRID if ytype == "resid" else [1.0]):
                                p = PRED[f"anchor::{key}"] + lam * g if ytype == "resid" else g
                                rows.append(dict(cond=cond, target=t, split=sp, ytype=ytype, method=method, hp=hp, hp_nested=hp_best, seed=seed, lam=lam,
                                                 rmse=rmse(EVAL[key]["y"], p), n=len(ev), n_feat=len(FEATS), feats="", fit_s=round(time.time() - tf, 1)))
                            # 비DANN noinfo 모델 → 공변량만 B 셀 재채점(같은 예측의 부분집합)
                            if cond == "noinfo" and "covonly" in conds:
                                for sp2, ev2, _ in conds["covonly"]:
                                    key2 = f"covonly|{t}|{sp2}"
                                    sel = np.isin(EVAL[key]["loc_id"], EVAL[key2]["loc_id"])
                                    pk2 = f"{key2}|{ytype}|{method}|{hp:g}|{seed}"
                                    PRED[f"g::{pk2}"] = g[sel].astype(np.float32)
                                    for lam in (LAM_GRID if ytype == "resid" else [1.0]):
                                        p = PRED[f"anchor::{key2}"] + lam * g[sel] if ytype == "resid" else g[sel]
                                        rows.append(dict(cond="covonly", target=t, split=sp2, ytype=ytype, method=method, hp=hp, hp_nested=hp_best, seed=seed,
                                                         lam=lam, rmse=rmse(EVAL[key2]["y"], p), n=int(sel.sum()), n_feat=len(FEATS), feats="", fit_s=np.nan))
            print(f"  [{t}|{ytype}|{method}] 행 {len(rows)} · {time.time()-t_start:.0f}s", flush=True)

tag = args.tag + ("_smoke" if args.smoke else "")
pd.DataFrame(rows).to_csv(OUT / f"{tag}_rows.csv", index=False)
np.savez_compressed(OUT / f"{tag}_preds.npz", **{k: v for k, v in PRED.items()},
                    **{f"eval::{k}::{f}": np.asarray(v[f]) for k, v in EVAL.items() for f in ("y", "block", "loc_id")})
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:                                                                                           # noqa: BLE001
    commit = "NA"
(OUT / f"{tag}_meta.json").write_text(json.dumps(dict(stage="H22", design="docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md", E_AK=E_AK, targets=TARGETS,
    methods=METHODS, ytypes=YTYPES, hp=HP, seeds=SEEDS, splits=SPLITS, epochs=EPOCHS, be=args.be, gpu=GPU, meta=META, git_commit=commit,
    elapsed_s=round(time.time() - t_start, 1)), ensure_ascii=False, indent=1, default=str))
r = pd.DataFrame(rows)
if len(r):
    q = r[(r.hp == r.hp_nested)].groupby(["cond", "target", "ytype", "method", "lam"]).rmse.mean().reset_index()
    print(q[q.lam.isin([0.25, 1.0])].pivot_table(index=["cond", "target", "ytype", "lam"], columns="method", values="rmse").round(2).to_string())
print(f"saved {tag}_* · 행 {len(rows)} · {time.time()-t_start:.0f}s")
