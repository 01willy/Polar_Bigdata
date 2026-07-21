"""트랙 1 — 알래스카 내부 라벨 증강 × 다중 DL.

목표
----
실측 ALT(알래스카 약 13,606셀)에 물리유도·지중온도장유도 유사라벨(pseudo)을 더해
훈련하면 알래스카 내부 ALT 예측이 개선되는지, 어떤 모델이 증강으로 이득을 보는지
정직 판정한다. 전이는 범위 밖(알래스카 내부 한정).

회의적 원칙(최상위)
-------------------
- 평가는 반드시 실측 ALT held-out으로만 한다. 유사라벨(물리유도·지중온도유도)은
  train 블록에만 넣고 test 블록에는 절대 넣지 않는다.
- 공간블록 6-fold CV. block = floor(lat/0.5)*100000 + floor(lon/0.5).
- pseudo는 test 블록과 0.5°블록을 공유하면 fold마다 제거(좌표·블록 누설 차단).
- Stefan E는 해당 fold의 실측 train만으로 적합(누설 방지, fold마다 재적합).
- RMSE 옆 R²·skill_over_mean 병기. 개선이 유사라벨 누설·평균회귀 착시인지 진단.

라벨 소스(훈련용, test는 실측만)
--------------------------------
1. 실측 ALT(앵커): dl_dataset_cell_v3_soil.csv region∈{ABoVE_AK, United States (Alaska)}.
   held-out 실측은 이 앵커만. 공변량 = 기후8(e5_*)·지형6(dem_*)·토양9(sg_*).
2. 물리유도 ALT(pseudo): 알래스카 ERA5 격자(기후 climatology)에서 파생한 신규 내부
   좌표(앵커 격자 +0.05° 오프셋, 육지 내부)에 지형·토양을 최근접 앵커로 부착하고
   Stefan(a + E·sqrt(TDD)) 적용. E는 실측 train fold에서만 적합. train 블록에만 추가.
3. 지중온도장유도 ALT(pseudo): ground_temp_gtnp_global.csv 알래스카 시추공 t_max
   포락선의 0°C 통과 깊이(선형보간)로 ALT 유도(cm). 0-400cm 클립. 최근접 앵커 격자에
   부착. Phase 1서 field-ALT 정합 r≈0.16이었으므로 신뢰 낮을 수 있음 → 별도 소스로
   효과 분리 측정.

실험 설계(스윕)
---------------
- 소스 조합: real_only / real+phys / real+borehole / real+phys+borehole.
- 물리유도 비율 P_ratio ∈ {0, 0.5, 1, 2} × (실측 train 수).
- 유사라벨 sample_weight ∈ {0.3, 1.0}(실측 우위 유지 옵션).
- 물리식 변형: Stefan 상수E / Stefan 층화E(swe·maat) 두 종.
- 모델: GBM(baseline) + MLP + FT-Transformer + TabM(전부 GPU, torch).
- 채점: 공간블록 6-fold, held-out 실측 ALT만. RMSE·R²·skill. SOTA 대역 14-18cm 표기.

산출
----
  data/processed/aug_within_alaska_results.csv  (모델×소스조합 요약, real_only 기준)
  data/processed/aug_within_alaska_sweep.csv    (전 스윕 셀)
  data/processed/aug_within_alaska_meta.json    (E값·pseudo 통계·부트스트랩 CI)
  outputs/figures/12_aug_alaska/*.png|pdf       (히트맵·비율곡선·예측지도·분포)

실행:
  CUDA_VISIBLE_DEVICES=8 /home/anaconda3/bin/python scripts/3_deep_learning/aug_within_alaska.py
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn

from polar.eval_metrics import all_metrics

PROC = "data/processed"
CELL = os.path.join(PROC, "dl_dataset_cell_v3_soil.csv")
GT = os.path.join(PROC, "ground_temp_gtnp_global.csv")
ERA5_CLIM = "/tmp/ak_era5_clim.parquet"  # 사전 캐시(없으면 재생성)
CLIP = (np.log1p(1.0), np.log1p(600.0))
AK_REGIONS = ["ABoVE_AK", "United States (Alaska)"]

CLIMATE = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
TERRAIN = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
SOIL = ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15", "sg_cfvo_5_15",
        "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15", "sg_soc_15_30"]
FEATS = CLIMATE + TERRAIN + SOIL  # 23 공변량, 알래스카 완전(결측 없음 확인됨)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

t0 = time.time()


# ============================================================
# 데이터 로드
# ============================================================
def load_anchor():
    df = pd.read_csv(CELL, low_memory=False)
    ak = df[df["region"].isin(AK_REGIONS)].copy().reset_index(drop=True)
    for c in FEATS + ["alt_cm", "lat", "lon"]:
        ak[c] = pd.to_numeric(ak[c], errors="coerce")
    ak = ak.dropna(subset=FEATS + ["alt_cm"]).reset_index(drop=True)
    ak["block"] = (np.floor(ak.lat / 0.5).astype(int) * 100000
                   + np.floor(ak.lon / 0.5).astype(int))
    return ak


def build_era5_clim():
    """ERA5 temporal에서 알래스카 기후 climatology(연평균) 격자 생성/캐시."""
    if os.path.exists(ERA5_CLIM):
        return pd.read_parquet(ERA5_CLIM)
    src = os.path.join(PROC, "alt_era5_temporal.csv")
    cols = ["e5t_maat", "e5t_tdd", "e5t_fdd", "e5t_sqrt_tdd", "e5t_twarm",
            "e5t_tcold", "e5t_stl1", "e5t_swe"]
    rows = []
    for ch in pd.read_csv(src, chunksize=500000):
        m = ((ch.lat >= 60) & (ch.lat <= 72.5) & (ch.lon >= -167) & (ch.lon <= -140))
        sub = ch[m]
        if len(sub):
            rows.append(sub[["lat", "lon"] + cols])
    a = pd.concat(rows)
    clim = a.groupby([a.lat.round(4), a.lon.round(4)])[cols].mean().reset_index()
    clim.columns = ["lat", "lon"] + cols
    clim.to_parquet(ERA5_CLIM)
    return clim


# ============================================================
# 물리유도 pseudo 풀(신규 내부 좌표)
# ============================================================
def build_phys_pool(anchor):
    """앵커 ERA5 격자 +0.05° 오프셋 신규 내부 좌표에 지형·토양을 최근접 앵커로 부착.
    e5t_* climatology를 e5_* 로 매핑. Stefan 입력 sqrt(TDD) 보유. 라벨(ALT)은 fold별 E로
    나중에 부여."""
    clim = build_era5_clim()
    off = clim.copy()
    off["lat"] = off["lat"] + 0.05
    off["lon"] = off["lon"] + 0.05
    # 육지 내부만: 원 era5 격자와 <0.09° 근접
    tree_c = cKDTree(clim[["lat", "lon"]].values)
    d, _ = tree_c.query(off[["lat", "lon"]].values, k=1)
    off = off[d < 0.09].reset_index(drop=True)
    # e5t_* -> e5_*
    ren = {f"e5t_{k}": f"e5_{k}" for k in
           ["maat", "tdd", "fdd", "sqrt_tdd", "twarm", "tcold", "stl1", "swe"]}
    off = off.rename(columns=ren)
    # 지형·토양: 최근접 앵커
    tree_a = cKDTree(anchor[["lat", "lon"]].values)
    _, idx = tree_a.query(off[["lat", "lon"]].values, k=1)
    for c in TERRAIN + SOIL:
        off[c] = anchor[c].values[idx]
    off["block"] = (np.floor(off.lat / 0.5).astype(int) * 100000
                    + np.floor(off.lon / 0.5).astype(int))
    off["sqrt_tdd"] = np.sqrt(np.clip(off["e5_tdd"].values, 0, None))
    return off.reset_index(drop=True)


# ============================================================
# 지중온도장유도 pseudo(시추공 t_max 0°C 통과 깊이)
# ============================================================
def build_borehole_pool(anchor):
    """알래스카 시추공 t_max 포락선의 0°C 통과 깊이(선형보간, m→cm)로 ALT 유도.
    최근접 앵커 격자에 부착(공변량은 앵커 것 사용). 0-400cm 클립."""
    gt = pd.read_csv(GT)
    ak = gt[(gt.lat >= 60) & (gt.lat <= 72.5) & (gt.lon >= -167) & (gt.lon <= -140)].copy()
    recs = []
    for bid, g in ak.groupby("borehole_id"):
        g = g.sort_values("depth")
        dep = g["depth"].values.astype(float)  # m
        tmax = g["t_max"].values.astype(float)
        if len(dep) < 2:
            continue
        # 0°C 통과(양->음) 첫 지점 선형보간
        alt_m = None
        for i in range(len(dep) - 1):
            t1, t2 = tmax[i], tmax[i + 1]
            if t1 > 0 and t2 <= 0:
                frac = t1 / (t1 - t2)
                alt_m = dep[i] + frac * (dep[i + 1] - dep[i])
                break
        if alt_m is None:
            # 전부 양(활동층이 최심 관측보다 깊음) 또는 전부 음(영구동토가 표층). 스킵.
            continue
        alt_cm = float(np.clip(alt_m * 100.0, 0, 400))
        recs.append({"lat": g.lat.iloc[0], "lon": g.lon.iloc[0],
                     "borehole_id": bid, "alt_cm": alt_cm})
    bh = pd.DataFrame(recs)
    if len(bh) == 0:
        return bh
    tree_a = cKDTree(anchor[["lat", "lon"]].values)
    _, idx = tree_a.query(bh[["lat", "lon"]].values, k=1)
    for c in FEATS:
        bh[c] = anchor[c].values[idx]
    bh["block"] = (np.floor(bh.lat / 0.5).astype(int) * 100000
                   + np.floor(bh.lon / 0.5).astype(int))
    return bh.reset_index(drop=True)


# ============================================================
# Stefan E 적합(fold 실측 train만)
# ============================================================
def fit_E(y, s):
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    denom = float((s * s).sum())
    E0 = float((s * y).sum() / denom) if denom > 0 else 0.0
    r0 = float(np.sqrt(np.mean((y - E0 * s) ** 2)))
    A = np.c_[np.ones_like(s), s]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a1, E1 = float(coef[0]), float(coef[1])
    r1 = float(np.sqrt(np.mean((y - (a1 + E1 * s)) ** 2)))
    if r1 < r0:
        return {"mode": "intercept", "E": E1, "a": a1}
    return {"mode": "origin", "E": E0, "a": 0.0}


def fit_E_strat(y, s, swe, maat):
    """ALT = (g0 + g1·swe + g2·maat)·sqrt(TDD). 절편 없는 선형회귀."""
    m = np.isfinite(y) & np.isfinite(s)
    ym, sm, swem, maatm = y[m], s[m], swe[m], maat[m]
    A = np.c_[sm, swem * sm, maatm * sm]
    coef, *_ = np.linalg.lstsq(A, ym, rcond=None)
    g0, g1, g2 = map(float, coef)
    return {"g0": g0, "g1": g1, "g2": g2}


def stefan_const(fit, s):
    return fit["a"] + fit["E"] * s


def stefan_strat(fit, s, swe, maat):
    return (fit["g0"] + fit["g1"] * swe + fit["g2"] * maat) * s


# ============================================================
# torch 모델들 (MLP / FT-Transformer / TabM)
# ============================================================
class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=4, p=0.1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FTTransformer(nn.Module):
    """수치 특징 토큰화 + Transformer 인코더(간이 FT-Transformer)."""
    def __init__(self, d_in, d_token=48, n_layers=3, n_heads=4, p=0.1):
        super().__init__()
        self.d_in = d_in
        self.d_token = d_token
        # 특징별 선형 토큰화(weight·bias)
        self.w = nn.Parameter(torch.randn(d_in, d_token) * 0.02)
        self.b = nn.Parameter(torch.zeros(d_in, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        enc = nn.TransformerEncoderLayer(d_token, n_heads, d_token * 2, p,
                                         batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x):
        # x: (B, d_in) -> tokens (B, d_in, d_token)
        tok = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = torch.cat([cls, tok], dim=1)
        h = self.tr(h)
        return self.head(h[:, 0]).squeeze(-1)


class TabM(nn.Module):
    """TabM: k개 병렬 MLP 헤드 앙상블(효율적 배깅). 예측=평균."""
    def __init__(self, d_in, hidden=256, depth=3, k=8, p=0.1):
        super().__init__()
        self.k = k
        self.shared = nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden),
                                    nn.ReLU(), nn.Dropout(p))
        self.heads = nn.ModuleList()
        for _ in range(k):
            layers, d = [], hidden
            for _ in range(depth - 1):
                layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p)]
                d = hidden
            layers += [nn.Linear(d, 1)]
            self.heads.append(nn.Sequential(*layers))

    def forward(self, x):
        h = self.shared(x)
        outs = [head(h).squeeze(-1) for head in self.heads]
        return torch.stack(outs, 0).mean(0)


def train_torch(model_cls, Xtr, ytr, wtr, Xte, mu, sd, ymu, ysd,
                epochs=90, bs=2048, lr=2e-3):
    """표준화된 X, log1p·표준화된 y로 가중회귀 학습. test 예측 cm 반환."""
    model = model_cls(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    wt = torch.tensor(wtr, dtype=torch.float32, device=DEVICE)
    n = Xt.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = (wt[idx] * (pred - yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        Xe = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
        pe = model(Xe).cpu().numpy()
    ylog = pe * ysd + ymu
    return np.expm1(np.clip(ylog, *CLIP))


TORCH_MODELS = {"MLP": MLP, "FTTransformer": FTTransformer, "TabM": TabM}


def gbm_fit_predict(Xtr, ytr_log, wtr, Xte):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_leaf_nodes=63, l2_regularization=1.0,
                                      early_stopping=True, random_state=SEED)
    m.fit(Xtr, ytr_log, sample_weight=wtr)
    return np.expm1(np.clip(m.predict(Xte), *CLIP))


# ============================================================
# 스윕 실행
# ============================================================
def make_pseudo_labels_for_fold(anchor_tr, phys_pool, bh_pool, stefan_variant):
    """fold 실측 train으로 E 적합 후 pseudo ALT 부여. (phys_df, bh_df) 반환(라벨 포함)."""
    y = anchor_tr["alt_cm"].values
    s = np.sqrt(np.clip(anchor_tr["e5_tdd"].values, 0, None))
    phys = phys_pool.copy()
    sp = phys["sqrt_tdd"].values
    if stefan_variant == "const":
        fit = fit_E(y, s)
        phys["alt_cm"] = np.clip(stefan_const(fit, sp), 0, 400)
        efit = {"variant": "const", **fit}
    else:
        swe = anchor_tr["e5_swe"].values
        maat = anchor_tr["e5_maat"].values
        fit = fit_E_strat(y, s, swe, maat)
        phys["alt_cm"] = np.clip(
            stefan_strat(fit, sp, phys["e5_swe"].values, phys["e5_maat"].values), 0, 400)
        efit = {"variant": "strat", **fit}
    return phys, bh_pool, efit


def run_fold(model_name, cfg, tr_idx, te_idx, anchor, phys_pool, bh_pool):
    """단일 fold: 실측 train + pseudo(train블록만) 학습 → 실측 test 예측.
    반환 (yhat_cm[te], diag)."""
    a_tr = anchor.iloc[tr_idx].reset_index(drop=True)
    a_te = anchor.iloc[te_idx].reset_index(drop=True)
    test_blocks = set(a_te["block"].unique())

    # pseudo 라벨 부여(fold train 실측으로 E 적합)
    phys, bh, efit = make_pseudo_labels_for_fold(a_tr, phys_pool, bh_pool, cfg["stefan"])
    # 누설 차단: test 0.5°블록과 겹치는 pseudo 제거
    phys = phys[~phys["block"].isin(test_blocks)]
    bh = bh[~bh["block"].isin(test_blocks)] if len(bh) else bh

    # 소스 조합 + 물리 비율
    frames = [a_tr[FEATS + ["alt_cm"]].assign(w=1.0)]
    n_real = len(a_tr)
    if cfg["use_phys"] and cfg["p_ratio"] > 0 and len(phys):
        n_take = int(round(cfg["p_ratio"] * n_real))
        take = phys.sample(n=min(n_take, len(phys)), random_state=SEED)
        frames.append(take[FEATS + ["alt_cm"]].assign(w=cfg["pw"]))
    if cfg["use_bh"] and len(bh):
        frames.append(bh[FEATS + ["alt_cm"]].assign(w=cfg["pw"]))

    train = pd.concat(frames, ignore_index=True)
    Xtr = train[FEATS].values.astype(np.float32)
    ytr_cm = train["alt_cm"].values.astype(float)
    wtr = train["w"].values.astype(np.float32)
    ytr_log = np.log1p(np.clip(ytr_cm, 0, None))
    Xte = a_te[FEATS].values.astype(np.float32)

    diag = {"n_train": len(train), "n_real": n_real,
            "n_phys": int((train["w"].values != 1.0).sum() if cfg["use_phys"] else 0),
            "efit": efit}

    if model_name == "GBM":
        yhat = gbm_fit_predict(Xtr, ytr_log, wtr, Xte)
    else:
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        ymu = ytr_log.mean(); ysd = ytr_log.std() + 1e-8
        ytr_s = (ytr_log - ymu) / ysd
        yhat = train_torch(TORCH_MODELS[model_name], Xtr_s, ytr_s, wtr,
                           Xte_s, mu, sd, ymu, ysd)
    return yhat, diag


def main():
    print(f"[env] device={DEVICE}")
    anchor = load_anchor()
    print(f"[load] anchor AK cells={len(anchor)}  feats={len(FEATS)}")
    phys_pool = build_phys_pool(anchor)
    print(f"[phys] pseudo pool={len(phys_pool)} 신규 좌표")
    bh_pool = build_borehole_pool(anchor)
    print(f"[borehole] pseudo={len(bh_pool)}  ALT범위="
          f"{bh_pool['alt_cm'].min():.1f}-{bh_pool['alt_cm'].max():.1f}cm"
          if len(bh_pool) else "[borehole] pseudo=0")

    # 공간블록 6-fold
    gkf = GroupKFold(n_splits=6)
    splits = list(gkf.split(anchor, anchor["alt_cm"], groups=anchor["block"].values))

    # 스윕 정의
    models = ["GBM", "MLP", "FTTransformer", "TabM"]

    # 전체 스윕(GBM: CPU라 저렴, 물리 비율·가중·변형 전수)
    full_configs = []
    full_configs.append({"tag": "real_only", "use_phys": False, "use_bh": False,
                         "p_ratio": 0, "pw": 1.0, "stefan": "const"})
    for pr in [0.5, 1.0, 2.0]:
        for pw in [0.3, 1.0]:
            for sv in ["const", "strat"]:
                full_configs.append({"tag": f"real+phys_r{pr}_w{pw}_{sv}",
                                     "use_phys": True, "use_bh": False,
                                     "p_ratio": pr, "pw": pw, "stefan": sv})
    full_configs.append({"tag": "real+borehole", "use_phys": False, "use_bh": True,
                         "p_ratio": 0, "pw": 0.3, "stefan": "const"})
    full_configs.append({"tag": "real+phys+borehole", "use_phys": True, "use_bh": True,
                         "p_ratio": 1.0, "pw": 0.3, "stefan": "strat"})

    # 축소 스윕(torch: GPU 비용 큼, 비율-성능 곡선+가중·소스 핵심만).
    # 물리변형은 const 고정(strat과 차이는 GBM 전수 스윕에서 확인), 비율 곡선 유지.
    dl_configs = []
    dl_configs.append({"tag": "real_only", "use_phys": False, "use_bh": False,
                       "p_ratio": 0, "pw": 1.0, "stefan": "const"})
    for pr in [0.5, 1.0, 2.0]:          # 물리 비율-성능 곡선(가중 1.0)
        dl_configs.append({"tag": f"real+phys_r{pr}_w1.0_const",
                           "use_phys": True, "use_bh": False,
                           "p_ratio": pr, "pw": 1.0, "stefan": "const"})
    dl_configs.append({"tag": "real+phys_r1.0_w0.3_const",  # 가중 낮춤 대조
                       "use_phys": True, "use_bh": False,
                       "p_ratio": 1.0, "pw": 0.3, "stefan": "const"})
    dl_configs.append({"tag": "real+borehole", "use_phys": False, "use_bh": True,
                       "p_ratio": 0, "pw": 0.3, "stefan": "const"})
    dl_configs.append({"tag": "real+phys+borehole", "use_phys": True, "use_bh": True,
                       "p_ratio": 1.0, "pw": 0.3, "stefan": "strat"})

    def configs_for(model_name):
        return full_configs if model_name == "GBM" else dl_configs

    sweep_rows = []
    oof_store = {}  # (model,tag) -> full-length yhat array
    n = len(anchor)

    for model_name in models:
        for cfg in configs_for(model_name):
            yhat_full = np.full(n, np.nan)
            fdiag = []
            for fi, (tr, te) in enumerate(splits):
                yhat, diag = run_fold(model_name, cfg, tr, te, anchor,
                                      phys_pool, bh_pool)
                yhat_full[te] = yhat
                fdiag.append(diag)
            y = anchor["alt_cm"].values
            met = all_metrics(y, yhat_full)
            row = {"model": model_name, "config": cfg["tag"],
                   "use_phys": cfg["use_phys"], "use_bh": cfg["use_bh"],
                   "p_ratio": cfg["p_ratio"], "pseudo_w": cfg["pw"],
                   "stefan": cfg["stefan"],
                   "rmse_cm": met["rmse_cm"], "mae_cm": met["mae_cm"],
                   "bias_cm": met["bias_cm"], "r2": met["r2"],
                   "skill_over_mean": met["skill_over_mean"],
                   "pred_std_cm": round(float(np.nanstd(yhat_full)), 3),
                   "n_train_last": fdiag[-1]["n_train"],
                   "n_phys_last": fdiag[-1]["n_phys"]}
            sweep_rows.append(row)
            oof_store[(model_name, cfg["tag"])] = yhat_full
            print(f"[{model_name:13s}] {cfg['tag']:30s} "
                  f"RMSE={met['rmse_cm']:6.2f} R2={met['r2']:.3f} "
                  f"skill={met['skill_over_mean']:.3f} predSD={row['pred_std_cm']:.1f}")

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(PROC, "aug_within_alaska_sweep.csv"), index=False)

    # 요약 results: 모델별 real_only vs 최적 증강
    res_rows = []
    y = anchor["alt_cm"].values
    for model_name in models:
        sub = sweep[sweep.model == model_name]
        base = sub[sub.config == "real_only"].iloc[0]
        aug = sub[sub.config != "real_only"]
        best = aug.loc[aug.rmse_cm.idxmin()]
        # 부트스트랩 CI: real_only vs best 증강 RMSE 차(음수=증강개선)
        yb = oof_store[(model_name, "real_only")]
        ya = oof_store[(model_name, best.config)]
        m = np.isfinite(yb) & np.isfinite(ya) & np.isfinite(y)
        diffs = []
        B = 2000
        idx_all = np.where(m)[0]
        for _ in range(B):
            bi = rng.choice(idx_all, size=idx_all.size, replace=True)
            rb = np.sqrt(np.mean((y[bi] - yb[bi]) ** 2))
            ra = np.sqrt(np.mean((y[bi] - ya[bi]) ** 2))
            diffs.append(ra - rb)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        res_rows.append({
            "model": model_name,
            "real_only_rmse_cm": base.rmse_cm, "real_only_r2": base.r2,
            "real_only_skill": base.skill_over_mean,
            "best_aug_config": best.config, "best_aug_rmse_cm": best.rmse_cm,
            "best_aug_r2": best.r2, "best_aug_skill": best.skill_over_mean,
            "delta_rmse_cm": round(float(best.rmse_cm - base.rmse_cm), 3),
            "delta_rmse_ci_lo": round(float(lo), 3),
            "delta_rmse_ci_hi": round(float(hi), 3),
            "aug_helps_sig": bool(hi < 0),  # 95% CI 전체가 음수면 유의 개선
        })
    results = pd.DataFrame(res_rows)
    results.to_csv(os.path.join(PROC, "aug_within_alaska_results.csv"), index=False)

    # OOF 저장(최적 모델 예측지도용)
    best_overall = sweep.loc[sweep.rmse_cm.idxmin()]
    oof_df = anchor[["lat", "lon", "alt_cm"]].copy()
    oof_df["yhat_real_only"] = oof_store[(best_overall.model, "real_only")]
    oof_df["yhat_best"] = oof_store[(best_overall.model, best_overall.config)]
    oof_df.to_csv(os.path.join(PROC, "aug_within_alaska_oof.csv"), index=False)

    # pseudo 라벨 분포(정합성): 전체 앵커로 E 적합한 대표 pseudo
    phys_rep, bh_rep, efit_rep = make_pseudo_labels_for_fold(
        anchor, phys_pool, bh_pool, "const")

    meta = {
        "n_anchor": int(len(anchor)),
        "n_phys_pool": int(len(phys_pool)),
        "n_borehole_pool": int(len(bh_pool)),
        "feats": FEATS,
        "sota_band_cm": [14, 18],
        "efit_const_full": efit_rep,
        "phys_alt_stats": {
            "mean": round(float(phys_rep["alt_cm"].mean()), 2),
            "std": round(float(phys_rep["alt_cm"].std()), 2),
            "min": round(float(phys_rep["alt_cm"].min()), 2),
            "max": round(float(phys_rep["alt_cm"].max()), 2)},
        "borehole_alt_stats": ({
            "mean": round(float(bh_rep["alt_cm"].mean()), 2),
            "std": round(float(bh_rep["alt_cm"].std()), 2),
            "n": int(len(bh_rep))} if len(bh_rep) else {}),
        "real_alt_stats": {
            "mean": round(float(anchor["alt_cm"].mean()), 2),
            "std": round(float(anchor["alt_cm"].std()), 2),
            "min": round(float(anchor["alt_cm"].min()), 2),
            "max": round(float(anchor["alt_cm"].max()), 2)},
        "best_overall": {"model": best_overall.model,
                         "config": best_overall.config,
                         "rmse_cm": float(best_overall.rmse_cm)},
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(PROC, "aug_within_alaska_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # pseudo 대표 라벨 저장(그림용)
    phys_rep[["lat", "lon", "alt_cm"]].to_parquet("/tmp/aug_phys_rep.parquet")
    if len(bh_rep):
        bh_rep[["lat", "lon", "alt_cm"]].to_parquet("/tmp/aug_bh_rep.parquet")

    print("\n[results]")
    print(results.to_string(index=False))
    print(f"\n[done] {meta['runtime_min']} min  best={best_overall.model}/{best_overall.config} "
          f"RMSE={best_overall.rmse_cm:.2f}cm")


if __name__ == "__main__":
    main()
