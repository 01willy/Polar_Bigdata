"""H23 메타학습(FOMAML) 희소 라벨 적응 — 라벨 n ∈ {3,5,10}개로 적응하는 초기화가 계수 수축 적합보다 나은가. GPU 1장.

설계 docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md §5.3. S-F(희소 라벨 곡선) 규약 계승: 대상 A블록에서 n개 라벨 추출(반복 10) → B 채점, 분할 3.

고정: 입력 x25, 앵커 = 알래스카 E(Stefan), 잔차 목표. 모델 = MLP 256-128-64(ReLU, Dropout 0.1; BN 없음 — 지지집합 3행 적응 안정성).
환경(메타 과제) = 알래스카 6 fold ∪ {레나·캐나다·러시아 W·E} − {대상}. 대상 라벨은 적응(support)에만 n개.
수준
  maml       FOMAML 초기화(에피소드: 환경 하나에서 support n·query 128, 내부 5 step lr 0.01) → 대상 n개로 10 step 적응
  finetune   ERM(환경 풀링) 사전학습 초기화 → 같은 적응(대조: 메타학습 대 단순 사전학습)
  erm0       ERM 사전학습, 적응 없음(n=0 기준)
  shrink_k10 / refit  계수 적응(S-F 규약, 같은 라벨 추출로 짝지음)
  shrink+maml  E 수축 앵커 위에 MAML 잔차(잔차는 E_sh 기준)
예측 = 앵커 + λ·g, λ ∈ {0.25, 0.5, 1}.
산출 data/processed/h2/h23_<tag>_{runs,summary,tests}.csv, h23_<tag>_meta.json
실행: GPU=6 python3 scripts/3_deep_learning/h23_maml.py [--smoke]
"""
from __future__ import annotations
import os
GPU = os.environ.get("GPU", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.func import functional_call
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, spatial_block_splits                                                  # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks, fit_coefs                 # noqa: E402
from polar.preprocessing import fold_prep                                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="Lena,Canada,Russia_W,Russia_E")
ap.add_argument("--n-grid", default="3,5,10")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--episodes", type=int, default=3000)
ap.add_argument("--inner-steps", type=int, default=5)
ap.add_argument("--adapt-steps", type=int, default=10)
ap.add_argument("--inner-lr", type=float, default=0.01)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--tag", default="h23")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
PROC = ROOT / "data" / "processed"
OUT = PROC / "h2"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
TARGETS = args.targets.split(",")
N_GRID = [int(v) for v in args.n_grid.split(",")]
REPS = args.reps if not args.smoke else 2
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
EPISODES = args.episodes if not args.smoke else 100
LAMS = [0.25, 0.5, 1.0]
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


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


lossf = nn.SmoothL1Loss()


def inner_adapt(net, params, Xs, ys, steps, lr):
    """support 로 파라미터 dict 를 SGD 적응(1차 근사: 그래프 분리)."""
    p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    for _ in range(steps):
        l = lossf(functional_call(net, p, (Xs,)), ys)
        g = torch.autograd.grad(l, list(p.values()))
        p = {k: (v - lr * gi).detach().requires_grad_(True) for (k, v), gi in zip(p.items(), g)}
    return p


def meta_train(envs, seed, episodes, n_support_choices=(3, 5, 10), n_query=128):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net = MLP(envs[0][0].shape[1]).to(DEV); net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xe = [torch.tensor(X, dtype=torch.float32).to(DEV) for X, _ in envs]; ye = [torch.tensor(y, dtype=torch.float32).to(DEV) for _, y in envs]
    for ep in range(episodes):
        opt.zero_grad()
        for _ in range(4):                                      # 메타 배치 4 과제
            e = rng.randint(len(envs)); n = int(rng.choice(n_support_choices))
            idx = rng.permutation(len(Xe[e]))
            S, Q = idx[:n], idx[n:n + n_query]
            params = dict(net.named_parameters())
            pa = inner_adapt(net, params, Xe[e][S], ye[e][S], args.inner_steps, args.inner_lr)
            lq = lossf(functional_call(net, pa, (Xe[e][Q],)), ye[e][Q])
            g = torch.autograd.grad(lq, list(pa.values()))
            for (k, v), gi in zip(params.items(), g):             # 1차 MAML: query 기울기를 원 파라미터에 누적
                v.grad = gi.detach() / 4 if v.grad is None else v.grad + gi.detach() / 4
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    return net


def erm_train(envs, seed, epochs=100, bs=1024):
    torch.manual_seed(seed)
    X = torch.tensor(np.concatenate([e[0] for e in envs]), dtype=torch.float32).to(DEV)
    y = torch.tensor(np.concatenate([e[1] for e in envs]), dtype=torch.float32).to(DEV)
    net = MLP(X.shape[1]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    rng = np.random.RandomState(seed); va = torch.tensor(rng.rand(len(X)) < 0.1).to(DEV)
    best, state, pat = 1e9, None, 0
    for ep in range(epochs if not args.smoke else 5):
        net.train(); idx = torch.randperm(int((~va).sum()), device=DEV); Xt, yt = X[~va], y[~va]
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]; opt.zero_grad(); lossf(net(Xt[b]), yt[b]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(X[va]) - y[va]) ** 2).mean())
        if v < best - 1e-4:
            best, state, pat = v, copy.deepcopy(net.state_dict()), 0
        else:
            pat += 1
            if pat >= 6:
                break
    net.load_state_dict(state); net.eval()
    return net


def adapt_predict(net, Xs, ys, Xq, steps, lr):
    net.eval()
    params = dict(net.named_parameters())
    pa = inner_adapt(net, params, Xs, ys, steps, lr) if len(Xs) else params
    with torch.no_grad():
        return functional_call(net, pa, (Xq,)).cpu().numpy()


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2; s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


rows, meta = [], {}
for t in TARGETS:
    env_names = [f"AK{i}" for i in range(6)] + [r for r in REGION_ENVS if r != t]
    tr_idx = np.where(np.isin(ENV_ID, env_names))[0]
    t_idx = np.where(DF.macro.values == t)[0]
    Xraw_tr = DF.iloc[tr_idx][FEATS].values.astype(np.float32)
    r_tr = DF.y.values[tr_idx] - E_AK * DF.s.values[tr_idx]; ok = np.isfinite(r_tr)
    rmu, rsd = float(r_tr[ok].mean()), float(r_tr[ok].std() + 1e-6)
    Xtr0, _ = fold_prep(Xraw_tr, Xraw_tr[:1], nan_native=False)
    env_lab = ENV_ID[tr_idx]
    envs = [(Xtr0[(env_lab == e) & ok], ((r_tr - rmu) / rsd)[(env_lab == e) & ok]) for e in env_names if ((env_lab == e) & ok).sum() >= 20]
    nets = {}
    for seed in SEEDS:                                   # 대상당 1회(환경은 분할과 무관; 표준화 통계도 학습 행만 사용)
        tf = time.time()
        nets[("maml", seed)] = meta_train(envs, seed, EPISODES)
        nets[("finetune", seed)] = erm_train(envs, seed)
        meta[f"{t}|{seed}|train_s"] = round(time.time() - tf, 1)
        print(f"  [{t}|seed{seed}] 메타학습·ERM {time.time()-tf:.0f}s", flush=True)
    for sp in SPLITS:
        A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
        evB = B_idx[eval_mask(DF.iloc[B_idx])]
        A, B = DF.iloc[A_idx], DF.iloc[evB]
        Xtr, XB = fold_prep(Xraw_tr, B[FEATS].values.astype(np.float32), nan_native=False)
        _, XA = fold_prep(Xraw_tr, A[FEATS].values.astype(np.float32), nan_native=False)
        yA, sA, yB, sB = A.y.values, A.s.values, B.y.values, B.s.values
        _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
        XBt = torch.tensor(XB, dtype=torch.float32).to(DEV); XAt = torch.tensor(XA, dtype=torch.float32).to(DEV)
        anc_B = E_AK * sB
        base = dict(target=t, split=sp, n_A=len(A), n_eval=len(evB))

        def add(method, n, rep, seed, lam, pred, E_used):
            rows.append(dict(**base, method=method, n=int(n), rep=int(rep), seed=int(seed), lam=float(lam), rmse_cm=rmse(yB, pred),
                             rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)), E_used=float(E_used)))

        add("physics", 0, -1, -1, 0.0, anc_B, E_AK)
        for seed in SEEDS:
            g0 = adapt_predict(nets[("finetune", seed)], XBt[:0], None, XBt, 0, 0.0) * rsd + rmu
            for lam in LAMS:
                add("erm0", 0, -1, seed, lam, anc_B + lam * g0, E_AK)
            g0m = adapt_predict(nets[("maml", seed)], XBt[:0], None, XBt, 0, 0.0) * rsd + rmu
            for lam in LAMS:
                add("maml0", 0, -1, seed, lam, anc_B + lam * g0m, E_AK)
        for n in N_GRID:
            if n >= len(A):
                continue
            for rep in range(REPS):
                sel = np.random.RandomState(1_000_003 * sp + 1_009 * n + rep).choice(len(A), n, replace=False)   # S-F 와 같은 추출식
                y_n, s_n = yA[sel], sA[sel]
                E_ls = ls_E(y_n, s_n); E_sh = (n * E_ls + args.kappa * E_AK) / (n + args.kappa)
                add("refit", n, rep, -1, 0.0, E_ls * sB, E_ls); add("shrink_k10", n, rep, -1, 0.0, E_sh * sB, E_sh)
                Xs = XAt[sel]
                ys_fix = torch.tensor(((y_n - E_AK * s_n) - rmu) / rsd, dtype=torch.float32).to(DEV)
                ys_sh = torch.tensor(((y_n - E_sh * s_n) - rmu) / rsd, dtype=torch.float32).to(DEV)
                for seed in SEEDS:
                    for init in ("maml", "finetune"):
                        g = adapt_predict(nets[(init, seed)], Xs, ys_fix, XBt, args.adapt_steps, args.inner_lr) * rsd + rmu
                        g2 = adapt_predict(nets[(init, seed)], Xs, ys_sh, XBt, args.adapt_steps, args.inner_lr) * rsd + rmu
                        for lam in LAMS:
                            add(init, n, rep, seed, lam, anc_B + lam * g, E_AK)
                            add(f"shrink+{init}", n, rep, seed, lam, E_sh * sB + lam * g2, E_sh)
        print(f"  [{t}|{sp}] 행 {len(rows)} · {time.time()-t_start:.0f}s", flush=True)

runs = pd.DataFrame(rows)
tag = args.tag + ("_smoke" if args.smoke else "")
runs.to_csv(OUT / f"{tag}_runs.csv", index=False)
# 집계 + 짝지은 Δ(maml − shrink_k10 등), (분할, 반복, seed) 짝지음 부트스트랩
g = runs[runs.method != "physics"].groupby(["target", "split", "method", "lam", "n", "rep"], as_index=False).agg(rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"))
phys = runs[runs.method == "physics"].set_index(["target", "split"]).rmse_cm
g["phys"] = [phys.loc[(a, b)] for a, b in zip(g.target, g.split)]
summ = g.groupby(["target", "method", "lam", "n"], as_index=False).agg(rmse_mean=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), rmse_beq=("rmse_beq_cm", "mean"),
                                                                     win_phys=("rmse_cm", lambda x: np.nan), n_runs=("rmse_cm", "size"), phys=("phys", "mean"))
summ["d_phys"] = summ.rmse_mean - summ.phys
summ.to_csv(OUT / f"{tag}_summary.csv", index=False)
tests = []
ref = g[(g.method == "shrink_k10")].set_index(["target", "split", "n", "rep"]).rmse_cm
for (t, m, lam, n), sub in g[g.method.isin(["maml", "finetune", "shrink+maml", "shrink+finetune", "refit"])].groupby(["target", "method", "lam", "n"]):
    keys = list(zip(sub.target, sub.split, sub.n, sub.rep)); okk = [k in ref.index for k in keys]
    if sum(okk) < 3:
        continue
    d = sub.rmse_cm.values[okk] - ref.loc[[k for k, o in zip(keys, okk) if o]].values
    rng = np.random.RandomState(1); bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(1000)])
    tests.append(dict(target=t, method=m, lam=lam, n=n, n_pairs=len(d), delta_vs_shrink=float(d.mean()), ci_lo=float(np.percentile(bs, 2.5)),
                      ci_hi=float(np.percentile(bs, 97.5)), win_vs_shrink=float((d < 0).mean())))
tests = pd.DataFrame(tests)
if len(tests):
    ab = tests.groupby(["method", "lam", "n"], as_index=False).agg(delta_vs_shrink=("delta_vs_shrink", "mean"), neg_k=("delta_vs_shrink", lambda x: int((x < 0).sum())),
                                                                 n_regions=("target", "size"), win_vs_shrink=("win_vs_shrink", "mean"))
    ab["target"] = "MEAN[AB4]"; tests = pd.concat([tests, ab], ignore_index=True)
tests.to_csv(OUT / f"{tag}_tests.csv", index=False)
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:                                                                                           # noqa: BLE001
    commit = "NA"
(OUT / f"{tag}_meta.json").write_text(json.dumps(dict(stage="H23", design="docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md", E_AK=E_AK, targets=TARGETS, n_grid=N_GRID,
    reps=REPS, seeds=SEEDS, splits=SPLITS, episodes=EPISODES, inner_steps=args.inner_steps, adapt_steps=args.adapt_steps, inner_lr=args.inner_lr, kappa=args.kappa,
    lams=LAMS, gpu=GPU, meta=meta, git_commit=commit, elapsed_s=round(time.time() - t_start, 1)), ensure_ascii=False, indent=1, default=str))
if len(tests):
    print(tests[tests.target == "MEAN[AB4]"].to_string())
print(f"saved {tag}_* · 행 {len(runs)} · {time.time()-t_start:.0f}s")
