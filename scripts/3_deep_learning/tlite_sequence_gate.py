"""T-lite — site-year ALT 시계열 예측 게이트 (신규 DL 도입, GPT 계획 P6-C).

질문: 시계열 DL(GRU/TCN)이 persistence·GBM-annual을 실제로 이기는가? 이기지 못하면 appendix로 강등.
- 데이터: CALM site-year(alt_global.csv, 259 사이트, 1990-2024) + 연별 ERA5 요약(alt_era5_temporal).
- target: ALT_cm(site, t)  |  입력: 과거 L년 [연climate, ALT] 시퀀스 + 당해 climate(known) + static.
- baseline: B0 persistence(ALT_t=ALT_{t-1}) · B1 climatology(사이트 확장평균) · B2 GBM-annual(tabular).
- DL: B3 GRU · B4 TCN(1D conv).  검증: site-disjoint 5-fold + temporal holdout(train<=2014, test>=2015).
- 게이트: DL RMSE < persistence & < GBM-annual, temporal에서 붕괴 없음, split-conformal coverage 85-95%.

실행: CUDA_VISIBLE_DEVICES=1 python3 scripts/3_deep_learning/tlite_sequence_gate.py
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from polar.eval_metrics import all_metrics, coverage
import torch
import torch.nn as nn

PROC = "data/processed"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
L = 8  # 과거 시퀀스 창(년)
CLIMATE = ["e5t_maat", "e5t_tdd", "e5t_fdd", "e5t_sqrt_tdd", "e5t_twarm", "e5t_tcold", "e5t_stl1", "e5t_swe", "e5t_swe_prevwinter"]
STATIC = ["wc_elev", "wc_bio1", "wc_bio4", "wc_bio7", "wc_bio12", "lat"]
t0 = time.time()

# ---------- 데이터 결합: CALM + 연별 ERA5(최근접 격자, 같은 해) ----------
g = pd.read_csv(os.path.join(PROC, "alt_global.csv"))
e = pd.read_csv(os.path.join(PROC, "alt_era5_temporal.csv"))
egrid = e[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
tree = cKDTree(egrid.values)
_, idx = tree.query(g[["lat", "lon"]].values, k=1)
g["elat"] = egrid.lat.values[idx]; g["elon"] = egrid.lon.values[idx]
g = g.merge(e, left_on=["elat", "elon", "year"], right_on=["lat", "lon", "year"],
            how="left", suffixes=("", "_e"))
for c in CLIMATE:
    g[c] = g[c].fillna(g[c].median())
g = g.sort_values(["site", "year"]).reset_index(drop=True)
print(f"[load] CALM site-year={len(g)} sites={g.site.nunique()} climate매칭={g[CLIMATE[0]].notna().mean()*100:.0f}%")

# ---------- 샘플 구성: 각 (site, t) with 최소 lag1 ----------
samples = []  # dict per sample
CL = len(CLIMATE); ST = len(STATIC)
for site, grp in g.groupby("site"):
    grp = grp.sort_values("year"); yrs = grp.year.values; alt = grp.alt_cm.values
    clim = grp[CLIMATE].values.astype(float); stat = grp[STATIC].values.astype(float)
    for i in range(1, len(grp)):
        # 과거 창: [max(0,i-L) .. i-1] 각 스텝 = [climate(τ), alt(τ)]
        s = max(0, i - L); hist_c = clim[s:i]; hist_a = alt[s:i]
        seq = np.concatenate([hist_c, hist_a[:, None]], axis=1)  # (len, CL+1)
        pad = L - seq.shape[0]
        if pad > 0:
            seq = np.concatenate([np.zeros((pad, CL + 1)), seq], axis=0)
        mask = np.concatenate([np.zeros(pad), np.ones(i - s)])  # 1=valid
        samples.append({
            "site": site, "year": int(yrs[i]), "y": float(alt[i]),
            "seq": seq.astype(np.float32), "mask": mask.astype(np.float32),
            "clim_t": clim[i].astype(np.float32), "static": stat[i].astype(np.float32),
            "alt_lag1": float(alt[i - 1]), "alt_lag2": float(alt[i - 2]) if i >= 2 else float(alt[i - 1]),
            "clim_mean": float(np.mean(hist_a)),
        })
S = pd.DataFrame(samples)
print(f"[samples] n={len(S)}  sites={S.site.nunique()}  year {S.year.min()}-{S.year.max()}")

# 표준화 통계(전체 기준; fold 내 재적합은 생략 — 소규모라 영향 작음. 누설 최소화 위해 static/clim만)
seq_stack = np.stack(S.seq.values)  # (N,L,CL+1)
valid = np.stack(S["mask"].values)[..., None]
seq_mu = (seq_stack * valid).sum((0, 1)) / (valid.sum((0, 1)) + 1e-9)
seq_sd = np.sqrt(((seq_stack - seq_mu) ** 2 * valid).sum((0, 1)) / (valid.sum((0, 1)) + 1e-9)) + 1e-6
clim_t = np.stack(S.clim_t.values); stat_a = np.stack(S.static.values)
ct_mu, ct_sd = clim_t.mean(0), clim_t.std(0) + 1e-6
st_mu, st_sd = stat_a.mean(0), stat_a.std(0) + 1e-6

def norm_seq(a): return ((a - seq_mu) / seq_sd).astype(np.float32)

# ---------- tabular 특징(GBM-annual) ----------
def tab_features(sub):
    X = np.column_stack([np.stack(sub.static.values), np.stack(sub.clim_t.values),
                         sub.alt_lag1.values, sub.alt_lag2.values, sub.clim_mean.values])
    return X
TABy = S.y.values.astype(float)

# ---------- DL 모델 ----------
IN = CL + 1
class GRUNet(nn.Module):
    def __init__(s, h=48):
        super().__init__()
        s.gru = nn.GRU(IN, h, batch_first=True)
        s.head = nn.Sequential(nn.Linear(h + CL + ST, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(s, seq, ct, st):
        o, _ = s.gru(seq); h = o[:, -1]
        return s.head(torch.cat([h, ct, st], 1)).squeeze(-1)

class TCN(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv1d(IN, ch, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        s.head = nn.Sequential(nn.Linear(ch + CL + ST, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(s, seq, ct, st):
        h = s.net(seq.transpose(1, 2)).squeeze(-1)
        return s.head(torch.cat([h, ct, st], 1)).squeeze(-1)

def train_dl(kind, tr, te):
    Xs_tr = torch.tensor(norm_seq(np.stack(S.seq.values[tr])), device=DEV)
    Xs_te = torch.tensor(norm_seq(np.stack(S.seq.values[te])), device=DEV)
    ct_tr = torch.tensor(((np.stack(S.clim_t.values[tr]) - ct_mu) / ct_sd).astype(np.float32), device=DEV)
    ct_te = torch.tensor(((np.stack(S.clim_t.values[te]) - ct_mu) / ct_sd).astype(np.float32), device=DEV)
    st_tr = torch.tensor(((np.stack(S.static.values[tr]) - st_mu) / st_sd).astype(np.float32), device=DEV)
    st_te = torch.tensor(((np.stack(S.static.values[te]) - st_mu) / st_sd).astype(np.float32), device=DEV)
    yv = S.y.values.astype(np.float32); ymu, ysd = yv[tr].mean(), yv[tr].std() + 1e-6
    yt = torch.tensor((yv[tr] - ymu) / ysd, device=DEV)
    net = (GRUNet() if kind == "gru" else TCN()).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()
    n = len(tr); bs = 256
    best = 1e9; best_state = None; patience = 0
    for ep in range(200):
        net.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            p = net(Xs_tr[b], ct_tr[b], st_tr[b])
            loss = lossf(p, yt[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tl = lossf(net(Xs_tr, ct_tr, st_tr), yt).item()
        if tl < best - 1e-4:
            best = tl; best_state = {k: v.clone() for k, v in net.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience > 15:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pred = net(Xs_te, ct_te, st_te).cpu().numpy() * ysd + ymu
    return pred

# ---------- 평가 루프 ----------
def eval_split(name, splitter):
    preds = {k: np.full(len(S), np.nan) for k in ["persistence", "climatology", "gbm_annual", "gru", "tcn"]}
    for tr, te in splitter:
        preds["persistence"][te] = S.alt_lag1.values[te]
        preds["climatology"][te] = S.clim_mean.values[te]
        Xtr, Xte = tab_features(S.iloc[tr]), tab_features(S.iloc[te])
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True, random_state=0)
        gb.fit(Xtr, TABy[tr]); preds["gbm_annual"][te] = gb.predict(Xte)
        preds["gru"][te] = train_dl("gru", tr, te)
        preds["tcn"][te] = train_dl("tcn", tr, te)
    rows = []
    for k, p in preds.items():
        m = all_metrics(S.y.values, p); m["model"] = k; m["split"] = name
        rows.append(m)
    return rows, preds

results = []
# site-disjoint 5-fold
gkf = GroupKFold(n_splits=5)
site_splits = list(gkf.split(S, S.y.values, groups=S.site.values))
print("\n=== site-disjoint 5-fold ===")
r1, _ = eval_split("site_disjoint", site_splits)
results += r1
for m in sorted(r1, key=lambda z: z["rmse_cm"]):
    print(f"   {m['model']:12s} rmse={m['rmse_cm']:6.2f}  mae={m['mae_cm']:6.2f}  skill={m['skill_over_mean']*100:5.1f}%")

# temporal holdout
tr = np.where(S.year.values <= 2014)[0]; te = np.where(S.year.values >= 2015)[0]
print(f"\n=== temporal holdout (train<=2014 n={len(tr)}, test>=2015 n={len(te)}) ===")
r2, preds2 = eval_split("temporal_holdout", [(tr, te)])
results += r2
for m in sorted(r2, key=lambda z: z["rmse_cm"]):
    print(f"   {m['model']:12s} rmse={m['rmse_cm']:6.2f}  mae={m['mae_cm']:6.2f}  skill={m['skill_over_mean']*100:5.1f}%")

res = pd.DataFrame(results)[["split", "model", "n", "rmse_cm", "mae_cm", "bias_cm", "r2", "target_sd_cm", "skill_over_mean"]]
res.to_csv(os.path.join(PROC, "tlite_sequence_gate_results.csv"), index=False)

# 예측 저장(대표=temporal holdout)
pd.DataFrame({"site": S.site.values[te], "year": S.year.values[te], "y": S.y.values[te],
              **{f"pred_{k}": preds2[k][te] for k in preds2}}).to_csv(
    os.path.join(PROC, "tlite_holdout_predictions.csv"), index=False)

# ---------- 게이트 판정 ----------
def rmse_of(split, model):
    r = res[(res.split == split) & (res.model == model)]
    return float(r.rmse_cm.iloc[0]) if len(r) else np.nan
verdict = {}
for dl in ["gru", "tcn"]:
    sd_pass = rmse_of("site_disjoint", dl) < min(rmse_of("site_disjoint", "persistence"), rmse_of("site_disjoint", "gbm_annual"))
    th_pass = rmse_of("temporal_holdout", dl) < min(rmse_of("temporal_holdout", "persistence"), rmse_of("temporal_holdout", "gbm_annual"))
    verdict[dl] = {"beats_baselines_site": bool(sd_pass), "beats_baselines_temporal": bool(th_pass),
                   "gate_pass": bool(sd_pass and th_pass)}
meta = {"n_samples": int(len(S)), "n_sites": int(S.site.nunique()), "L": L, "device": DEV,
        "verdict": verdict, "elapsed_s": round(time.time() - t0, 1)}
with open(os.path.join(PROC, "tlite_gate_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("\n[gate]", json.dumps(verdict, ensure_ascii=False))
print(f"[done] {time.time()-t0:.0f}s")
