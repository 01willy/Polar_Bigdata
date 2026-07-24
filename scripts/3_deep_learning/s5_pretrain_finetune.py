"""S5: dense Stefan pseudo 사전학습 → 실측 finetune (A3·§9.4 게이트식).

`RESEARCH_PLAN_multifidelity` S5. 핵심 질문: 물리(Stefan) 유도장을 dense 격자에서 사전학습한
신경망이, 실측 finetune 후 from-scratch 대비 전이(LORO)를 개선하는가.
(S3에서 물리 pseudo "혼합 증강"의 순가치는 작았다. S5는 혼합이 아니라 표현 사전학습 경로.)

설계:
 - dense 격자 = pretrain_weaklabels.parquet 4.04M(알래스카·서부캐나다, 지형6+기후8).
   Stefan pseudo y = E_train·√TDD_grid. E는 LORO train 라벨만으로 역산(fold-safe).
 - 격자에서 test 셀과 같은 0.5° 블록은 제거(거리버퍼). 라벨 누설 없음(pseudo는 공변량 함수).
   단 Alaska fold는 test 지역 공변량 분포를 사전학습에 노출(transductive) → 결과 별도 표기.
 - 입력 14(지형+기후) 고정: 격자에 SAR·토양·CCI 없음. from-scratch 대조군도 동일 14로 공정 비교.
 - 모델: mlp·ftt(전이 최선). pretrain 15ep(서브샘플 500k) → finetune(early stop). 3 seed.
 - 평가: LORO(매크로) + in-domain AK 공간블록. 게이트: LORO에서 scratch 대비 개선/동률.

산출: data/processed/s5_pretrain_{results,meta}.{csv,json}
실행: GPU=6 python scripts/3_deep_learning/s5_pretrain_finetune.py   (SMOKE=1 빠른점검)
"""
import os
GPU = os.environ.get("GPU", "4")
assert GPU in {"2", "3", "4", "5", "6", "7", "8", "9"}, f"GPU는 2-9만 허용(요청 {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (TERRAIN, CLIMATE, TARGET, add_group_keys, spatial_block_splits,
                            loro_splits, macro_region, fit_stefan_E, assert_fold_safe_E,
                            BLOCK_DEG)
from polar.eval_metrics import all_metrics
from polar.tab_models import _torch_mods, set_device
import torch
set_device("cuda:0" if torch.cuda.is_available() else "cpu")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"[GPU guard] CUDA_VISIBLE_DEVICES={GPU} → uuid …{str(torch.cuda.get_device_properties(0).uuid)[-8:]}")

SMOKE = os.environ.get("SMOKE", "0") == "1"
PROC = C.PROCESSED
FEAT = TERRAIN + CLIMATE  # 14 (격자 가용 공변량)
MODELS = ["mlp", "ftt"] if not SMOKE else ["mlp"]
SEEDS = [0, 1, 2] if not SMOKE else [0]
N_GRID = 500_000 if not SMOKE else 50_000
PRE_EPOCHS = 15 if not SMOKE else 2
FT_EPOCHS = 120 if not SMOKE else 10
ALASKA = ["ABoVE_AK", "United States (Alaska)"]

df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
df["macro"] = macro_region(df)
y = df[TARGET].values.astype(float)
sqtdd = df["e5_sqrt_tdd"].values.astype(float)

grid = pd.read_parquet(PROC / "pretrain_weaklabels.parquet", columns=FEAT + ["lat", "lon"])
grid = grid.dropna(subset=FEAT).reset_index(drop=True)
rng0 = np.random.RandomState(42)
grid = grid.iloc[rng0.choice(len(grid), min(N_GRID, len(grid)), replace=False)].reset_index(drop=True)
grid["block"] = (np.floor(grid.lat / BLOCK_DEG).astype(int) * 100000
                 + np.floor(grid.lon / BLOCK_DEG).astype(int))
print(f"[S5] 실측 {len(df)} · 격자 서브샘플 {len(grid)} · feat {len(FEAT)}")

MLP, FTT, _ = _torch_mods()


def make_net(name, d):
    return {"mlp": lambda: MLP(d), "ftt": lambda: FTT(d)}[name]().to(DEV)


def epochs_fit(net, Xtr, ytr, Xva, yva, epochs, lr, bs=8192, pat=6):
    import torch.nn as nn
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); Xv = torch.tensor(Xva).to(DEV)
    best, state, p = 1e9, None, 0
    for _ in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = idx[k:k + bs]; xb, yb = Xt[b].to(DEV), yt[b].to(DEV)
            opt.zero_grad(); nn.SmoothL1Loss()(net(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.mean((net(Xv).cpu().numpy() - yva) ** 2))
        if v < best - 1e-4:
            best, state, p = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
        else:
            p += 1
            if p >= pat:
                break
    if state:
        net.load_state_dict(state)
    return net


def predict(net, X, bs=65536):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.tensor(X[k:k + bs]).to(DEV)).cpu().numpy()
                               for k in range(0, len(X), bs)])


def run_variant(name, seed, Xtr, ytr, Xte, grid_X, grid_y, variant, ft_lr=3e-4):
    """scratch: 실측만 학습. pretrain: 격자 pseudo 사전학습 → 실측 finetune.
    표준화·y-스케일은 train(+격자) 통계만 사용(fold-safe)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    # 입력 표준화: pretrain이면 격자 통계(광역 안정), scratch면 실측 train 통계
    ref = grid_X if variant == "pretrain" else Xtr
    med = np.nanmedian(ref, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    mu = np.nanmean(np.where(np.isnan(ref), med, ref), axis=0)
    sd = np.nanstd(np.where(np.isnan(ref), med, ref), axis=0) + 1e-6

    def Z(X):
        X = np.where(np.isnan(X), med, X)
        return ((X - mu) / sd).astype(np.float32)

    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
    net = make_net(name, Xtr.shape[1])
    lr0 = 5e-4 if name == "ftt" else 1e-3
    if variant == "pretrain":
        gy = ((grid_y - ymu) / ysd).astype(np.float32)
        va = rng.rand(len(grid_X)) < 0.05
        net = epochs_fit(net, Z(grid_X[~va]), gy[~va], Z(grid_X[va]), gy[va],
                         PRE_EPOCHS, lr=lr0, pat=3)
        lr_ft = ft_lr
    else:
        lr_ft = lr0
    yz = ((ytr - ymu) / ysd).astype(np.float32)
    va = rng.rand(len(Xtr)) < 0.1
    net = epochs_fit(net, Z(Xtr[va == False]), yz[va == False], Z(Xtr[va]), yz[va],
                     FT_EPOCHS, lr=lr_ft)
    return predict(net, Z(Xte)) * ysd + ymu


rows = []
X_all = df[FEAT].values.astype(np.float32)
Xg_all = grid[FEAT].values.astype(np.float32)

# ============ Part A: LORO 전이 ============
loro = loro_splits(df, min_test=100)
print(f"[Part A] LORO: {[r for r, _, _ in loro]}")
for r, tr, te in loro:
    assert_fold_safe_E(tr, te, df)
    E = fit_stefan_E(y[tr], sqtdd[tr])
    # 거리버퍼: test 셀과 같은 0.5° 블록의 격자점 제거
    te_blocks = set(df["block"].values[te].tolist())
    keep = ~grid["block"].isin(te_blocks).values
    gX = Xg_all[keep]
    g_sq = np.sqrt(np.maximum(grid["e5_tdd"].values[keep], 0.0)) if "e5_tdd" in FEAT else None
    g_sq = np.sqrt(np.maximum(gX[:, FEAT.index("e5_tdd")], 0.0))
    gy = E * g_sq
    ok = np.isfinite(gy)
    transductive = r in ("Alaska", "Canada")  # 격자가 test 지역 공변량 포함(라벨 누설은 없음)
    print(f"  [{r}] E={E:.2f} 격자 {ok.sum()}점(버퍼 후) transductive={transductive}")
    for model in MODELS:
        for variant in ["scratch", "pretrain"]:
            for seed in SEEDS:
                pred = run_variant(model, seed, X_all[tr], y[tr], X_all[te],
                                   gX[ok], gy[ok], variant)
                m = all_metrics(y[te], pred)
                rows.append(dict(part="A_loro", cv="LORO", region=r, model=model,
                                 variant=variant, seed=seed, transductive=transductive, **m))
        sc = np.mean([x["rmse_cm"] for x in rows if x["region"] == r and x["model"] == model
                      and x["variant"] == "scratch"])
        pt = np.mean([x["rmse_cm"] for x in rows if x["region"] == r and x["model"] == model
                      and x["variant"] == "pretrain"])
        print(f"    {model}: scratch {sc:.2f} → pretrain {pt:.2f} (Δ {sc-pt:+.2f})")

res_a = pd.DataFrame([x for x in rows if x["part"] == "A_loro"])
for (model, variant), g in res_a.groupby(["model", "variant"]):
    per_reg = g.groupby("region").rmse_cm.mean()
    rows.append(dict(part="A_loro", cv="LORO_gate", region="UNWEIGHTED_MEAN", model=model,
                     variant=variant, rmse_cm=float(per_reg.mean()), n=len(per_reg)))

# ============ Part B: in-domain AK 공간블록 ============
print("[Part B] in-domain AK 공간블록")
ak = df[df.region.isin(ALASKA)].reset_index(drop=True)
folds_ak = spatial_block_splits(ak, n_splits=6)
yak, sqak = ak[TARGET].values.astype(float), ak["e5_sqrt_tdd"].values.astype(float)
Xak = ak[FEAT].values.astype(np.float32)
for model in MODELS:
    for variant in ["scratch", "pretrain"]:
        for seed in SEEDS:
            oof = np.full(len(ak), np.nan)
            for tr, te in folds_ak:
                assert_fold_safe_E(tr, te, ak)
                E = fit_stefan_E(yak[tr], sqak[tr])
                te_blocks = set(ak["block"].values[te].tolist())
                keep = ~grid["block"].isin(te_blocks).values
                gX = Xg_all[keep]
                g_sq = np.sqrt(np.maximum(gX[:, FEAT.index("e5_tdd")], 0.0))
                gy = E * g_sq
                ok = np.isfinite(gy)
                oof[te] = run_variant(model, seed, Xak[tr], yak[tr], Xak[te],
                                      gX[ok], gy[ok], variant)
            m = all_metrics(yak, oof)
            rows.append(dict(part="B_indomain", cv="spatial_block_AK", region="Alaska",
                             model=model, variant=variant, seed=seed, **m))
    sc = np.mean([x["rmse_cm"] for x in rows if x["part"] == "B_indomain"
                  and x["model"] == model and x["variant"] == "scratch"])
    pt = np.mean([x["rmse_cm"] for x in rows if x["part"] == "B_indomain"
                  and x["model"] == model and x["variant"] == "pretrain"])
    print(f"  {model}: scratch {sc:.2f} → pretrain {pt:.2f} (Δ {sc-pt:+.2f})")

# ============ 저장 ============
res = pd.DataFrame(rows)
res.to_csv(PROC / "s5_pretrain_results.csv", index=False)
gate = res[res.cv == "LORO_gate"]
import subprocess
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(C.ROOT)).decode().strip()
except Exception:
    commit = "NA"
meta = dict(purpose="S5 dense Stefan pseudo 사전학습→실측 finetune", feat=FEAT, n_grid=int(len(grid)),
            models=MODELS, seeds=SEEDS, pre_epochs=PRE_EPOCHS, smoke=SMOKE,
            gate_table={f"{m}_{v}": float(g.rmse_cm.iloc[0])
                        for (m, v), g in gate.groupby(["model", "variant"])},
            gpu=GPU, git_commit=commit,
            note="입력 14(지형+기후) 고정 공정비교. Alaska·Canada fold는 transductive 표기. "
                 "pseudo=E_train·√TDD(fold-safe E). 격자 test-블록 버퍼 제거.")
(PROC / "s5_pretrain_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print("\n[done] LORO 게이트(비가중평균):")
for (m, v), g in gate.groupby(["model", "variant"]):
    print(f"  {m:5s} {v:8s} {g.rmse_cm.iloc[0]:.2f}")
