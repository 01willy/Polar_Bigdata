"""S8: mixture-of-physics experts 진단 (A6·RQ3).

`RESEARCH_PLAN_multifidelity` S8. 헤드라인이 아니라 진단이 목적이다:
"어느 물리식이 어느 환경에서 선택되는가". S2에서 물리 5종은 Stefan(p1)만 정확하고
(LORO 게이트 22.24cm), 나머지는 상방편향(+32~38cm)·상관 0.89~1.00이므로 mixture가
단일 Stefan을 못 이길 가능성이 크다. 못 이기면 그대로 negative로 보고한다.

설계:
 - 전문가 = 물리 5종(p1 Stefan·p2 edaphic·p3 TTOP·p4 Kudryavtsev·p5 λ). p1의 E는
   train fold 내부에서만 역산(fold-safe). p2~p5는 E 무관 결정론.
 - 게이트 = 공변량(지형6+기후8)에서 softmax 가중을 내는 소형 분류기 2종:
   (a) logit 다항로지스틱(결정론, seed 1), (b) mlp 2층 MLP(확률적, 3 seed).
   학습 라벨 = train fold에서 |y-p_k| 최소 전문가(argmin). fold-safe 표준화(fold_prep).
 - 결측 전문가(p3 비동토 18.5%·p4 11.2%)는 셀별로 가중치 재정규화.
 - 평가: in-domain 알래스카 공간블록 6-fold + LORO(매크로 Alaska·Lena·Canada,
   비가중평균 게이트). 비교군 = 최고 단일 물리(Stefan)·균등평균·oracle(셀별 최적
   전문가, test 라벨 사용 → 모델 아님·게이팅 헤드룸 진단 전용).
 - 진단: (i) 셀별 지배 전문가(OOF), (ii) 게이트 가중치-|오차| 스피어만 상관,
   (iii) 환경변수 오분위 구간별 전문가 선택 분포, (iv) 게이트 top-1 vs oracle 정확도.

GPU: 불필요(sklearn CPU). 타 사용자 GPU 오점유 방지를 위해 CUDA 노출을 차단한다.

산출: data/processed/s8_mixture_{results.csv,oof.csv,meta.json}
실행: python scripts/3_deep_learning/s8_mixture_physics.py   (SMOKE=1 로 빠른 점검)
"""
import os
# 게이트는 sklearn(CPU)로 충분하다. torch를 쓰지 않으며 CUDA 노출 자체를 차단한다.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import (TERRAIN, CLIMATE, TARGET, add_group_keys,
                            spatial_block_splits, loro_splits, macro_region,
                            assert_fold_safe_E)
from polar.physics import physics_ensemble, fit_E, PHYSICS_MEMBERS
from polar.preprocessing import fold_prep as prep
from polar.eval_metrics import all_metrics

T0 = time.time()
SMOKE = os.environ.get("SMOKE", "0") == "1"
PROC = C.PROCESSED
ALASKA = ["ABoVE_AK", "United States (Alaska)"]
GATE_FEATS = TERRAIN + CLIMATE          # 14, shift-robust(전지역 가용·결측 최소)
K = len(PHYSICS_MEMBERS)                # 전문가 5
SEEDS_MLP = [0] if SMOKE else [0, 1, 2]
MLP_ITER = 120 if SMOKE else 500
NBOOT = 50 if SMOKE else 300
ENV_VARS = {  # 진단용 환경변수(구간별 선택 분포): 컬럼 → (표기, 단위)
    "e5_maat": ("연평균기온 MAAT", "°C"), "e5_tdd": ("융해도일 TDD", "°C·day"),
    "e5_swe": ("적설수당량 SWE", "m"), "dem_elev": ("고도", "m"),
    "dem_slope": ("경사", "°"), "sg_soc_5_15": ("토양유기탄소 SOC 5-15cm", "g/kg"),
}

df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
df["macro"] = macro_region(df)
y = df[TARGET].values.astype(float)
sqtdd = df["e5_sqrt_tdd"].values.astype(float)
Xgate_all = df[GATE_FEATS].values.astype(np.float32)

# p2~p5는 E 무관 결정론 → 전 셀 1회 계산(공변량만 사용, S2 규약과 동일).
# p1은 라벨 의존(E) → fold별로 col 0을 E_train·√TDD로 교체한다.
phys0 = physics_ensemble(df, E=1.0)
P_FREE = np.column_stack([phys0[m] for m in PHYSICS_MEMBERS]).astype(float)
s2_gate = json.loads((PROC / "s2_physics_meta.json").read_text())["p1_stefan_LORO_gate"]
print(f"[S8] n={len(df)} · 전문가 {K}종 · 게이트 feats {len(GATE_FEATS)} · SMOKE={SMOKE}"
      f" · S2 참조(p1 LORO 게이트)={s2_gate:.2f}cm")


def expert_matrix(idx, E):
    """(len(idx), 5) 전문가 예측. col0 = fold-safe E·√TDD(S2와 동일하게 클립 없음)."""
    P = P_FREE[idx].copy()
    P[:, 0] = E * sqtdd[idx]
    return P


def best_labels(yv, P):
    """셀별 최적 전문가 argmin|y-p_k| (결측 전문가 제외). valid 마스크 병반환."""
    err = np.abs(yv[:, None] - P)
    err = np.where(np.isfinite(err), err, np.inf)
    lab = np.argmin(err, axis=1)
    valid = np.isfinite(yv) & np.isfinite(err.min(axis=1))
    return lab, valid


def fit_gate(kind, Xtr, lab, Xte, seed=0):
    """게이트 학습 → test 소프트맥스 가중 (n_te, 5). 미출현 클래스는 0."""
    Xtr2, Xte2 = prep(Xtr, Xte, nan_native=False)
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=3000, C=1.0)
    else:
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-3,
                          max_iter=MLP_ITER, random_state=seed)
    m.fit(Xtr2, lab)
    W = np.zeros((len(Xte2), K))
    W[:, np.asarray(m.classes_, int)] = m.predict_proba(Xte2)
    return W


def mix_predict(W, P):
    """결측 전문가 재정규화 가중합. 반환 (pred, 재정규화 가중 Wn)."""
    mask = np.isfinite(P)
    Wm = np.where(mask, W, 0.0)
    s = Wm.sum(1, keepdims=True)
    unif = mask / np.clip(mask.sum(1, keepdims=True), 1, None)   # fallback=유한 균등
    Wn = np.where(s > 1e-8, Wm / np.clip(s, 1e-8, None), unif)
    pred = (Wn * np.where(mask, P, 0.0)).sum(1)
    pred[mask.sum(1) == 0] = np.nan
    return pred, Wn


def uniform_predict(P):
    return mix_predict(np.full(P.shape, 1.0 / K), P)[0]


def oracle_predict(yv, P):
    """test 라벨로 셀별 최적 전문가 선택(진단 하한, 모델 아님)."""
    lab, valid = best_labels(yv, P)
    pred = P[np.arange(len(P)), lab]
    pred[~valid] = np.nan
    return pred, lab


def entropy_norm(Wn):
    """유한 전문가 수로 정규화한 게이트 엔트로피(0=확신, 1=균등)."""
    Wp = np.clip(Wn, 1e-12, 1.0)
    H = -(Wn * np.log(Wp)).sum(1)
    nf = (Wn > 1e-9).sum(1)
    return np.where(nf > 1, H / np.log(np.clip(nf, 2, None)), 0.0)


rng = np.random.RandomState(42)


def block_bootstrap_delta(yv, base, alt, blocks, nboot=NBOOT):
    """블록 리샘플 ΔRMSE(base−alt) CI. 양수 = alt(mixture 등)가 Stefan 대비 개선."""
    ublk = np.unique(blocks)
    deltas = []
    for _ in range(nboot):
        samp = np.concatenate([np.where(blocks == k)[0]
                               for k in rng.choice(ublk, len(ublk), replace=True)])
        rb = np.sqrt(np.nanmean((yv[samp] - base[samp]) ** 2))
        ra = np.sqrt(np.nanmean((yv[samp] - alt[samp]) ** 2))
        deltas.append(rb - ra)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


rows = []

# ============ Part A: in-domain 알래스카 공간블록 6-fold ============
# 인덱스 정합: ak는 fidelity_base의 알래스카 부분집합. 원본 행 위치(src_row)를 보존해
# P_FREE·Xgate_all을 조회한다.
ak = df[df.region.isin(ALASKA)].reset_index().rename(columns={"index": "src_row"})
if SMOKE:
    ak = ak.sample(n=min(3000, len(ak)), random_state=0).reset_index(drop=True)
yak = ak[TARGET].values.astype(float)
sq_ak = ak["e5_sqrt_tdd"].values.astype(float)
src = ak["src_row"].values
folds_ak = spatial_block_splits(ak, n_splits=6)
print(f"[Part A] in-domain 알래스카 {len(ak):,}셀 · 공간블록 6-fold")

oof = {m: np.full(len(ak), np.nan) for m in
       ["stefan", "uniform", "oracle", "mix_logit"] + [f"mix_mlp_s{s}" for s in SEEDS_MLP]}
Wn_oof_ak = np.full((len(ak), K), np.nan)   # logit 재정규화 가중(지도용)
for tr, te in folds_ak:
    assert_fold_safe_E(tr, te, ak)
    E = fit_E(yak[tr], sq_ak[tr])
    Ptr, Pte = expert_matrix(src[tr], E), expert_matrix(src[te], E)
    oof["stefan"][te] = Pte[:, 0]
    oof["uniform"][te] = uniform_predict(Pte)
    oof["oracle"][te] = oracle_predict(yak[te], Pte)[0]
    lab, valid = best_labels(yak[tr], Ptr)
    Xtr, Xte = Xgate_all[src[tr]], Xgate_all[src[te]]
    W = fit_gate("logit", Xtr[valid], lab[valid], Xte)
    oof["mix_logit"][te], Wn_oof_ak[te] = mix_predict(W, Pte)
    for s in SEEDS_MLP:
        W = fit_gate("mlp", Xtr[valid], lab[valid], Xte, seed=s)
        oof[f"mix_mlp_s{s}"][te] = mix_predict(W, Pte)[0]

for name, pred in oof.items():
    m = all_metrics(yak, pred)
    model = name.split("_s")[0] if name.startswith("mix_mlp") else name
    seed = int(name.split("_s")[1]) if name.startswith("mix_mlp") else 0
    rows.append(dict(part="A_indomain", cv="spatial_block_AK", model=model, seed=seed, **m))
mlp_mean = np.nanmean([oof[f"mix_mlp_s{s}"] for s in SEEDS_MLP], axis=0)
rows.append(dict(part="A_indomain", cv="spatial_block_AK", model="mix_mlp_seedmean", seed=-1,
                 **all_metrics(yak, mlp_mean)))
d, lo, hi = block_bootstrap_delta(yak, oof["stefan"], oof["mix_logit"], ak["block"].values)
rows.append(dict(part="C_boot", cv="spatial_block_AK", region="Alaska_indomain",
                 model="mix_logit", delta_rmse=d, ci_lo=lo, ci_hi=hi))
ind_tbl = {r["model"]: r["rmse_cm"] for r in rows if r["part"] == "A_indomain"}
print("  in-domain RMSE: " + " ".join(f"{k}={v:.2f}" for k, v in ind_tbl.items()))
print(f"  부트스트랩 Δ(stefan−mix_logit) {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")

# ============ Part B: LORO 전이 (게이트) ============
loro = loro_splits(df, min_test=100)
loro_regions = [r for r, _, _ in loro]
print(f"[Part B] LORO 지역: {loro_regions}")
loro_store = {}   # region → dict(te, preds, Wn_logit, oracle_lab, gate_lab_argmax)
for r, tr, te in loro:
    assert_fold_safe_E(tr, te, df)
    E = fit_E(y[tr], sqtdd[tr])
    Ptr, Pte = expert_matrix(tr, E), expert_matrix(te, E)
    preds = {"stefan": Pte[:, 0], "uniform": uniform_predict(Pte)}
    preds["oracle"], ora_lab = oracle_predict(y[te], Pte)
    lab, valid = best_labels(y[tr], Ptr)
    # 게이트 학습 라벨 분포(train, 진단)
    sh = np.bincount(lab[valid], minlength=K) / valid.sum()
    rows.append(dict(part="F_gatediag", cv="LORO", region=r, metric="train_best_share",
                     **{PHYSICS_MEMBERS[k]: float(sh[k]) for k in range(K)}, n=int(valid.sum())))
    W = fit_gate("logit", Xgate_all[tr][valid], lab[valid], Xgate_all[te])
    preds["mix_logit"], Wn = mix_predict(W, Pte)
    for s in SEEDS_MLP:
        Wm = fit_gate("mlp", Xgate_all[tr][valid], lab[valid], Xgate_all[te], seed=s)
        preds[f"mix_mlp_s{s}"] = mix_predict(Wm, Pte)[0]
    preds["mix_mlp_seedmean"] = np.nanmean([preds[f"mix_mlp_s{s}"] for s in SEEDS_MLP], axis=0)
    for name, p in preds.items():
        if name == "mix_mlp_seedmean":
            model, seed = "mix_mlp_seedmean", -1
        elif name.startswith("mix_mlp_s"):
            model, seed = "mix_mlp", int(name.rsplit("_s", 1)[1])
        else:
            model, seed = name, 0
        rows.append(dict(part="B_loro", cv="LORO", region=r, model=model, seed=seed,
                         **all_metrics(y[te], p)))
    # oracle-best 분포·게이트 top-1 정확도(진단)
    ok = np.isfinite(preds["oracle"])
    acc = float(np.mean(np.argmax(Wn, 1)[ok] == ora_lab[ok]))
    sh_o = np.bincount(ora_lab[ok], minlength=K) / ok.sum()
    rows.append(dict(part="F_gatediag", cv="LORO", region=r, metric="oracle_best_share",
                     **{PHYSICS_MEMBERS[k]: float(sh_o[k]) for k in range(K)}, n=int(ok.sum())))
    rows.append(dict(part="F_gatediag", cv="LORO", region=r, metric="gate_top1_acc",
                     value=acc, n=int(ok.sum())))
    loro_store[r] = dict(te=te, preds=preds, Wn=Wn, ora_lab=ora_lab, Pte=Pte)
    print(f"  [{r}] stefan {all_metrics(y[te], preds['stefan'])['rmse_cm']:.2f} · "
          f"uniform {all_metrics(y[te], preds['uniform'])['rmse_cm']:.2f} · "
          f"mix_logit {all_metrics(y[te], preds['mix_logit'])['rmse_cm']:.2f} · "
          f"oracle {all_metrics(y[te], preds['oracle'])['rmse_cm']:.2f} · top1 {acc:.2f}")

# 게이트 집계(비가중평균): model×seed 지역평균 → seed 평균
res_b = pd.DataFrame([x for x in rows if x["part"] == "B_loro"])
for model, g in res_b.groupby("model"):
    per_reg = g[g.seed != -1].groupby("region").rmse_cm.mean() if model == "mix_mlp" \
        else g.groupby("region").rmse_cm.mean()
    rows.append(dict(part="B_loro", cv="LORO_gate", region="UNWEIGHTED_MEAN", model=model,
                     seed=-1, rmse_cm=float(per_reg.mean()), n=len(per_reg)))
gate_tbl = {x["model"]: x["rmse_cm"] for x in rows if x.get("cv") == "LORO_gate"}
print("[게이트] LORO 비가중평균: " + " ".join(f"{k}={v:.2f}" for k, v in gate_tbl.items()))

# ============ Part C: 블록부트스트랩 CI (지역별, vs Stefan) ============
print("[Part C] 블록부트스트랩 ΔRMSE(stefan−model) 95% CI")
for r, _, te in loro:
    blocks = df["block"].values[te]
    pr = loro_store[r]["preds"]
    for model in ["mix_logit", "mix_mlp_seedmean", "uniform"]:
        d, lo, hi = block_bootstrap_delta(y[te], pr["stefan"], pr[model], blocks)
        sig = "개선유의" if lo > 0 else ("악화유의" if hi < 0 else "CI 0 포함")
        rows.append(dict(part="C_boot", cv="LORO", region=r, model=model,
                         delta_rmse=d, ci_lo=lo, ci_hi=hi))
        print(f"  [{r}] {model:16s} Δ {d:+.2f} [{lo:+.2f}, {hi:+.2f}] {sig}")

# ============ Part D: 게이트 가중치-|오차| 상관 (LORO OOF, logit) ============
print("[Part D] 가중치-|오차| 스피어만 상관")
te_all = np.concatenate([loro_store[r]["te"] for r in loro_regions])
Wn_all = np.vstack([loro_store[r]["Wn"] for r in loro_regions])
mix_all = np.concatenate([loro_store[r]["preds"]["mix_logit"] for r in loro_regions])
P_all = np.vstack([loro_store[r]["Pte"] for r in loro_regions])
reg_all = np.concatenate([[r] * len(loro_store[r]["te"]) for r in loro_regions])
err_mix = np.abs(y[te_all] - mix_all)
wmax = Wn_all.max(1)
ent = entropy_norm(Wn_all)
for scope in ["POOLED"] + loro_regions:
    m = np.ones(len(te_all), bool) if scope == "POOLED" else (reg_all == scope)
    ok = m & np.isfinite(err_mix)
    rho_w, _ = spearmanr(wmax[ok], err_mix[ok])
    rho_h, _ = spearmanr(ent[ok], err_mix[ok])
    rows.append(dict(part="D_corr", cv="LORO", region=scope, metric="wmax_vs_abserr",
                     spearman=float(rho_w), n=int(ok.sum())))
    rows.append(dict(part="D_corr", cv="LORO", region=scope, metric="entropy_vs_abserr",
                     spearman=float(rho_h), n=int(ok.sum())))
    if scope == "POOLED":
        print(f"  POOLED corr(w_max,|err|)={rho_w:+.3f} corr(entropy,|err|)={rho_h:+.3f}")
for k, mem in enumerate(PHYSICS_MEMBERS):
    err_k = np.abs(y[te_all] - P_all[:, k])
    ok = np.isfinite(err_k)
    if np.unique(Wn_all[ok, k]).size < 2:      # 가중치 상수(예: 전 셀 0) → 상관 정의 불가
        rho = np.nan
    else:
        rho, _ = spearmanr(Wn_all[ok, k], err_k[ok])
    rows.append(dict(part="D_corr", cv="LORO", region="POOLED", metric="w_k_vs_abserr_k",
                     expert=mem, spearman=float(rho), n=int(ok.sum())))
    print(f"  corr(w_{mem}, |err_{mem}|) = {rho:+.3f}")

# ============ Part E: 환경변수 구간별 선택 분포 (LORO OOF, logit) ============
print("[Part E] 환경 구간별 전문가 선택 분포(오분위)")
dom_all = np.argmax(Wn_all, 1)
for var, (label, unit) in ENV_VARS.items():
    v = df[var].values[te_all].astype(float)
    ok = np.isfinite(v)
    edges = np.unique(np.nanquantile(v[ok], [0, .2, .4, .6, .8, 1.0]))
    binid = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = ok & (binid == b)
        if m.sum() < 20:
            continue
        wmean = Wn_all[m].mean(0)
        dsh = np.bincount(dom_all[m], minlength=K) / m.sum()
        rows.append(dict(part="E_envbins", cv="LORO", var=var, var_label=label, unit=unit,
                         bin=b, bin_lo=float(edges[b]), bin_hi=float(edges[b + 1]),
                         n=int(m.sum()),
                         **{f"w_{PHYSICS_MEMBERS[k]}": float(wmean[k]) for k in range(K)},
                         **{f"dom_{PHYSICS_MEMBERS[k]}": float(dsh[k]) for k in range(K)}))
    lo_w = [x for x in rows if x["part"] == "E_envbins" and x["var"] == var]
    print(f"  {var}: bins {len(lo_w)} · w_p1(low→high) "
          + "→".join(f"{b['w_p1_stefan']:.2f}" for b in lo_w))

# ============ 저장 ============
res = pd.DataFrame(rows)
res.to_csv(PROC / "s8_mixture_results.csv", index=False)

# OOF (지도·진단 그림용): LORO + in-domain AK
oof_loro = df.iloc[te_all][["loc_id", "lat", "lon", "region", "macro", "block", TARGET]].copy()
oof_loro["eval"] = "loro"
oof_loro["stefan_pred"] = np.concatenate([loro_store[r]["preds"]["stefan"] for r in loro_regions])
oof_loro["uniform_pred"] = np.concatenate([loro_store[r]["preds"]["uniform"] for r in loro_regions])
oof_loro["mix_logit_pred"] = mix_all
oof_loro["mix_mlp_pred"] = np.concatenate([loro_store[r]["preds"]["mix_mlp_seedmean"] for r in loro_regions])
oof_loro["oracle_pred"] = np.concatenate([loro_store[r]["preds"]["oracle"] for r in loro_regions])
for k, mem in enumerate(PHYSICS_MEMBERS):
    oof_loro[f"w_{mem}"] = Wn_all[:, k]
oof_loro["dominant"] = [PHYSICS_MEMBERS[k] for k in dom_all]
oof_loro["oracle_best"] = [PHYSICS_MEMBERS[k] if np.isfinite(p) else ""
                           for k, p in zip(np.concatenate([loro_store[r]["ora_lab"] for r in loro_regions]),
                                           oof_loro["oracle_pred"])]
oof_ak = ak[["loc_id", "lat", "lon", "region", "block", TARGET]].copy()
oof_ak["macro"] = "Alaska"
oof_ak["eval"] = "indomain_ak"
oof_ak["stefan_pred"] = oof["stefan"]
oof_ak["uniform_pred"] = oof["uniform"]
oof_ak["mix_logit_pred"] = oof["mix_logit"]
oof_ak["mix_mlp_pred"] = mlp_mean
oof_ak["oracle_pred"] = oof["oracle"]
for k, mem in enumerate(PHYSICS_MEMBERS):
    oof_ak[f"w_{mem}"] = Wn_oof_ak[:, k]
oof_ak["dominant"] = [PHYSICS_MEMBERS[k] if np.isfinite(Wn_oof_ak[i]).all() else ""
                      for i, k in enumerate(np.nanargmax(np.where(np.isfinite(Wn_oof_ak), Wn_oof_ak, -1), 1))]
oof_ak["oracle_best"] = ""
pd.concat([oof_loro, oof_ak], ignore_index=True).to_csv(PROC / "s8_mixture_oof.csv", index=False)

import subprocess
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(C.ROOT)).decode().strip()
except Exception:
    commit = "NA"
verdict = "negative" if gate_tbl["mix_logit"] > gate_tbl["stefan"] - 1e-9 else "positive"
meta = dict(
    purpose="S8 mixture-of-physics experts 진단(어느 물리가 어느 환경서 선택되는가)",
    experts=PHYSICS_MEMBERS, gate_feats=GATE_FEATS, gate_models=["logit", "mlp(32,32)x3seed"],
    seeds_mlp=SEEDS_MLP, nboot=NBOOT, smoke=SMOKE,
    protocol="E fold-safe(physics.fit_E, train만)·게이트 라벨=train argmin|y-p_k|·결측 전문가 재정규화",
    loro_regions=loro_regions, s2_reference_gate_cm=float(s2_gate),
    loro_gate_cm={k: float(v) for k, v in gate_tbl.items()},
    indomain_ak_cm={k: float(v) for k, v in ind_tbl.items()},
    verdict=verdict,
    gpu="미사용(sklearn CPU, CUDA_VISIBLE_DEVICES='')", git_commit=commit,
    runtime_s=round(time.time() - T0, 1),
    note="oracle은 test 라벨 사용 진단 하한(모델 아님). p2~p5 결정론 예측·공변량 중앙값 대치는 "
         "S2 규약 승계(라벨 미사용). 진단 가중치는 logit 게이트(결정론) 기준.")
(PROC / "s8_mixture_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"\n[done] verdict={verdict} · LORO 게이트 stefan {gate_tbl['stefan']:.2f} vs "
      f"mix_logit {gate_tbl['mix_logit']:.2f} vs uniform {gate_tbl['uniform']:.2f} "
      f"(oracle {gate_tbl['oracle']:.2f}) · {time.time()-T0:.0f}s")
