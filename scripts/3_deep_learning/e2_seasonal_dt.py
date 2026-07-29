"""E2: 계절 내 융해 진행 D(t) 예측. "timelapse에 따라 ALT 계산"의 물리적 재정의.

문제 정의(연도간 외삽과 대조):
- 연도간 최대 ALT 외삽은 시간신호가 약하다(S9 연 anomaly corr 0.06, 예측불가).
- 반면 계절 내 융해 진행 D(t)=표층 연결 0°C 등온선 깊이의 봄→가을 궤적은
  누적 융해도일(TDD)의 물리로 예측 가능하다(Stefan: D=E·√cumTDD). 본 스크립트는
  KPDC 콘슬 일별 다층 지중온도(s7_council_daily_temp.csv)로 이를 실증한다.

라벨 D(t):
- 각 (site_key, date)의 표층 연결 0°C 등온선 깊이(위 양수·아래 비양수 첫 교차 선형보간,
  s7 유도규칙과 동일). 표층(10cm) 동결이면 D=0. 전층 양수(융해면이 최심센서 아래)이면
  검열 → 모델링에서 제외(하한만 알려짐). 융해기 5-10월.

입력(누설통제):
- 물리 forcing = 10cm 지온에서 파생한 누적 TDD(융해개시일 이후 양의 지온 누적). √cumTDD.
- GRU 입력 시퀀스 = 과거 lag일의 [10cm지온, 일TDD증분, √누적TDD, day-of-year sin/cos].
  표준화 통계는 전부 train fold 내부에서만 산출(fold-safe).

비교 4모델(동일 fold):
  (a) Stefan  : D=E·√cumTDD, E는 train fold 내부 최소제곱(fold-safe).
  (b) GRU     : 과거 lag 시퀀스→당일 D(t). 소형·3seed·fold내부 표준화.
  (c) persist : 전일 D(t)(지속모델).
  (d) static  : 당일 √cumTDD→D 선형회귀(train fold 계수).

검증 2축:
  (i)  profile leave-out : site_key 단위 leave-one-out(공간 일반화).
  (ii) 계절 전반→후반    : 각 프로파일 내 앞 60%→뒤 40% 시간분할(시간 외삽).
부수: EOS(융해 종료·최대깊이 도달 시점) 예측 정확도(관측 peak day vs 예측 peak day).

주의: 콘슬=알래스카 in-domain(전이 아님). 데이터 2023-2025 2년뿐 → 표본 한계 명시.

산출: data/processed/e2_seasonal_dt_results.csv · e2_seasonal_dt_meta.json
      outputs/figures/e2_seasonal_dt/*.png
실행: GPU=7 python scripts/3_deep_learning/e2_seasonal_dt.py   (SMOKE=1 로 빠른 점검)
"""
import os
import sys
# FIGS_ONLY=1 또는 --figs-only: 기존 CSV(e2_seasonal_dt_{results,eos,summary}.csv)만 읽어
#   그림을 재생성한다. torch·GPU 없이 CPU에서 실행되며 모델 재학습·수치 재계산은 하지 않는다.
FIGS_ONLY = os.environ.get("FIGS_ONLY", "0") == "1" or "--figs-only" in sys.argv
if not FIGS_ONLY:
    # ★ GPU 고정은 어떤 import(특히 torch)보다 먼저. 2026-07-27 오후 지시: 물리 7번만 허용.
    GPU = os.environ.get("GPU", "7")
    assert GPU in {"7"}, f"오늘 허용 GPU는 7번뿐(요청: {GPU})"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
else:
    GPU = "cpu"

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SMOKE = os.environ.get("SMOKE", "0") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2]
NBOOT = 500 if SMOKE else 5000
LAG = 14              # GRU 과거 시퀀스 길이(일)
MIN_DT_DAYS = 40      # 모델링 채택 최소 D(t) 관측일
MIN_PEAK = 12.0       # 실제 융해 진행이 있는 시즌만(peak D)
# 예측 지평(일). D(t)는 일간 강한 자기상관 → 전일 지속(nowcast)은 사소하게 완벽.
# 의미 있는 비교는 H일 앞 예측: persistence=D(t-H), 물리·GRU·정적은 forcing으로 D(t) 예측.
# H=0(nowcast, persistence 사소 우위)와 H>0(forecast, 물리·GRU 가치) 둘 다 산출한다.
H_FCST = int(os.environ.get("H_FCST", "7"))
SEASON = (5, 10)
# rmse_pct 고정 참조 분모(minor #5): 모델링 12시즌의 평균 관측 최대깊이 peak D(≈93.5cm).
# 모델·split·support 무관하게 동일 분모로 % 비교(모델별 표본 peak에 분모가 흔들리지 않게).
# 값은 build_dt_panel/select_seasons로부터 산출되며 재실행 때마다 meta에 기록·검증한다.
REF_PEAK_D = 93.46
RNG = np.random.default_rng(20260727)
DPI = 90 if SMOKE else 300
PROC = ROOT / "data/processed"
FIGD = ROOT / "outputs/figures/e2_seasonal_dt"
FIGD.mkdir(parents=True, exist_ok=True)

if not FIGS_ONLY:
    import torch
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        uid = str(torch.cuda.get_device_properties(0).uuid)[-8:]
        # 물리 7번 UUID 꼬리(nvidia-smi 대조): GPU-dbbe3766-…-515a931e473e → e473e
        assert "e473e" in uid, f"논리0이 물리 GPU 7이 아님(uuid …{uid}). 실행 중단"
        print(f"[GPU guard] CUDA_VISIBLE_DEVICES={GPU} → 논리0 = 물리 GPU 7 (uuid …{uid})")
else:
    torch = None
    DEV = "cpu"


# ============================================================
# 1. D(t) 라벨 유도 (표층 연결 0°C 등온선 깊이, s7 규칙)
# ============================================================
def thaw_depth(depths, temps):
    """융해면 깊이 = 표층 연결(surface-connected) 첫 0°C 교차(선형보간).

    표층부터 아래로 내려가며 첫 +→비양수 교차 깊이. 표층 연결을 요구하므로 심부에 고립된
    양수 포켓(센서 아티팩트·탈릭)이 만드는 스파이크를 배제한다(예: ID23_B 5월 140cm 고립
    +0.2°C는 무시하고 표층 연결 융해면 ~70cm를 취함). 표층(첫 센서) 비양수→D=0(봄 동결 전
    또는 가을 표층 재동결). 전층 양수(최심센서까지)→검열(융해면 센서 아래, 하한만 가용).

    주의: 가을 표층 재동결 시 D가 0으로 붕괴해 실제 심부 융해면과 괴리한다. 따라서 예측 과제는
    deepening(봄 개시→피크) 구간을 물리 유효구간으로 별도 집계한다(phase 열)."""
    m = np.isfinite(temps)
    d, t = np.asarray(depths)[m], np.asarray(temps)[m]
    if len(d) < 2:
        return np.nan, False
    o = np.argsort(d)
    d, t = d[o], t[o]
    if t[0] <= 0:
        return 0.0, False           # 표층 비양수 = 표층 동결(융해 개시 전/후 재동결)
    for i in range(len(d) - 1):
        if t[i] > 0 and t[i + 1] <= 0:
            z = d[i] + (d[i + 1] - d[i]) * t[i] / (t[i] - t[i + 1])
            return float(z), False
    return np.nan, True             # 표층부터 최심까지 양수 = 검열(융해면 센서 아래)


def build_dt_panel():
    """(site_key, year, date)별 D(t) + 물리 forcing 파생 패널.

    forcing: 10cm 지온으로부터 융해개시(첫 양의 10cm) 이후 누적 TDD, √cumTDD, 일TDD증분.
    검열일(전층 양수)은 D 결측으로 제외. season 5-10월.
    """
    daily = pd.read_csv(PROC / "s7_council_daily_temp.csv", parse_dates=["date"])
    rows = []
    for key, g in daily.groupby("site_key"):
        piv = g.pivot_table(index="date", columns="depth_cm", values="temp_c").sort_index()
        depths = np.array(sorted(piv.columns))
        if 10.0 not in depths:
            continue
        surf = piv[10.0]
        dvals, cens = [], []
        for _, r in piv.iterrows():
            z, c = thaw_depth(depths, r[depths].values)
            dvals.append(z)
            cens.append(c)
        pan = pd.DataFrame({"D": dvals, "censored": cens, "surf10": surf.values},
                           index=piv.index)
        pan["year"] = pan.index.year
        for yr, sub in pan.groupby("year"):
            sub = sub.sort_index()
            pos = sub[sub["surf10"] > 0]
            if len(pos) < 20:
                continue
            onset = pos.index.min()
            s = sub.loc[onset:].copy()
            tdd_inc = np.clip(s["surf10"].values, 0, None)
            s["tdd_inc"] = tdd_inc
            s["cumTDD"] = np.cumsum(tdd_inc)
            s["sqrtTDD"] = np.sqrt(s["cumTDD"])
            doy = s.index.dayofyear.values
            s["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
            s["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
            # 융해기 창으로 제한
            s = s[(s.index.month >= SEASON[0]) & (s.index.month <= SEASON[1])]
            s = s.reset_index().rename(columns={"index": "date"})
            s["site_key"] = key
            s["season_year"] = int(yr)
            s["onset"] = onset
            rows.append(s)
    panel = pd.concat(rows, ignore_index=True)
    panel["depth_max_cm"] = panel.groupby("site_key")["surf10"].transform(lambda x: np.nan)
    # season_id = site_key×year (모델링·fold 단위)
    panel["season_id"] = panel["site_key"] + "|" + panel["season_year"].astype(str)
    # phase: 봄→최대깊이=deepening, 이후=plateau/refreeze. Stefan(단조 √TDD)의 유효 구간 구분.
    panel["phase"] = "unknown"
    for sid, g in panel.groupby("season_id"):
        gv = g[g["D"].notna() & (~g["censored"])].sort_values("date")
        if len(gv) < 5:
            continue
        peak_date = gv.loc[gv["D"].idxmax(), "date"]
        panel.loc[g.index, "phase"] = np.where(
            panel.loc[g.index, "date"] <= peak_date, "deepening", "plateau")
    return panel


# ============================================================
# 2. 모델링 대상 시즌 선별 + fold-safe Stefan E
# ============================================================
def select_seasons(panel):
    """D(t) 관측 충분·실제 융해진행 있는 시즌만 채택(검열일 제외한 D 관측 기준)."""
    obs = panel[panel["D"].notna() & (~panel["censored"])]
    keep = []
    for sid, g in obs.groupby("season_id"):
        if len(g) >= MIN_DT_DAYS and g["D"].max() >= MIN_PEAK:
            keep.append(sid)
    return keep


def fit_E_leastsq(sqrt_tdd, D):
    """D=E·√cumTDD 최소제곱 E. train fold 관측만 넘겨야 함(fold-safe)."""
    m = np.isfinite(sqrt_tdd) & np.isfinite(D) & (sqrt_tdd > 0)
    if m.sum() < 3:
        return np.nan
    s, a = sqrt_tdd[m], D[m]
    return float((s @ a) / (s @ s))


# ============================================================
# 3. 소형 GRU (과거 lag 시퀀스 → 당일 D)
# ============================================================
if not FIGS_ONLY:
    class TinyGRU(torch.nn.Module):
        def __init__(self, n_in, hidden=24):
            super().__init__()
            self.gru = torch.nn.GRU(n_in, hidden, batch_first=True)
            self.head = torch.nn.Sequential(torch.nn.Linear(hidden, 16),
                                            torch.nn.ReLU(), torch.nn.Linear(16, 1))

        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o[:, -1, :]).squeeze(-1)


SEQ_FEATS = ["surf10", "tdd_inc", "sqrtTDD", "doy_sin", "doy_cos"]


def make_sequences(df_season, lag=LAG):
    """한 시즌 시계열에서 (과거 lag일 특징 시퀀스, 당일 D) 표본 생성.

    df_season: 날짜순 정렬된 한 시즌 패널(결측 D 포함 가능, 타깃은 비검열 D만).
    반환: X(n,lag,f), y(n,), day_idx(n,). day_idx는 해당 표본의 시즌 내 행 인덱스.
    """
    df_season = df_season.sort_values("date").reset_index(drop=True)
    feats = df_season[SEQ_FEATS].values.astype(np.float32)
    D = df_season["D"].values.astype(np.float32)
    cens = df_season["censored"].values.astype(bool)
    Xs, ys, idxs = [], [], []
    for i in range(lag, len(df_season)):
        if not np.isfinite(D[i]) or cens[i]:
            continue
        win = feats[i - lag:i + 1]  # 과거 lag + 당일
        if not np.isfinite(win).all():
            continue
        Xs.append(win)
        ys.append(D[i])
        idxs.append(i)
    if not Xs:
        return (np.empty((0, lag + 1, len(SEQ_FEATS)), np.float32),
                np.empty((0,), np.float32), np.empty((0,), int))
    return np.stack(Xs), np.array(ys, np.float32), np.array(idxs, int)


def train_gru(Xtr, ytr, mu, sd, ymu, ysd, seed, epochs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = (40 if SMOKE else 160) if epochs is None else epochs
    Xn = (Xtr - mu) / sd
    yn = (ytr - ymu) / ysd
    xt = torch.tensor(Xn, dtype=torch.float32, device=DEV)
    yt = torch.tensor(yn, dtype=torch.float32, device=DEV)
    model = TinyGRU(Xtr.shape[-1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = torch.nn.SmoothL1Loss()
    n = len(xt)
    bs = min(128, n)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for j in range(0, n, bs):
            bi = perm[j:j + bs]
            opt.zero_grad()
            out = model(xt[bi])
            loss = lossf(out, yt[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    model.eval()
    return model


def gru_predict(model, X, mu, sd, ymu, ysd):
    if len(X) == 0:
        return np.empty((0,), np.float32)
    with torch.no_grad():
        Xn = (X - mu) / sd
        out = model(torch.tensor(Xn, dtype=torch.float32, device=DEV)).cpu().numpy()
    return out * ysd + ymu


# ============================================================
# 4. 평가 하네스 (두 검증축)
# ============================================================
from polar.eval_metrics import all_metrics


def eval_split(panel, seasons, mode):
    """mode='profile'(site_key leave-out) 또는 'temporal'(시즌 내 앞→뒤).

    반환: (long OOF DataFrame, EOS 예측 리스트).
    각 예측행: season_id, site_key, season_year, date, model, D_obs, D_pred, fold.
    """
    oof, eos = [], []
    obs = panel[panel["season_id"].isin(seasons)].copy()
    # site_key→macro region(전부 Alaska, 참고 열)
    obs["region"] = "Alaska"

    if mode == "profile":
        sites = sorted(obs["site_key"].unique())
        folds = [(s, [x for x in seasons if not x.startswith(s + "|")],
                  [x for x in seasons if x.startswith(s + "|")]) for s in sites]
        folds = [(f, tr, te) for f, tr, te in folds if te]
    else:  # temporal: 각 시즌 자기 앞부분으로 학습, 뒷부분 예측. E는 pooled 앞부분.
        folds = [("temporal", None, seasons)]

    for fold_name, tr_ids, te_ids in folds:
        # ---- train/test 패널 구성 ----
        if mode == "profile":
            tr = obs[obs["season_id"].isin(tr_ids)]
            te = obs[obs["season_id"].isin(te_ids)]
            # fold-safe Stefan E (train 비검열 관측만)
            trc = tr[tr["D"].notna() & (~tr["censored"])]
            E = fit_E_leastsq(trc["sqrtTDD"].values, trc["D"].values)
            # static 선형회귀(당일 √TDD→D, 절편 포함) train 계수
            A = np.c_[np.ones(len(trc)), trc["sqrtTDD"].values]
            beta = np.linalg.lstsq(A, trc["D"].values, rcond=None)[0]
            # GRU 학습표본 = train 시즌 전체
            tr_seqs = [make_sequences(obs[obs.season_id == s]) for s in tr_ids]
        else:
            # temporal: 각 시즌을 앞60%(train)·뒤40%(test)로. E·beta는 pooled 앞부분.
            te = obs.copy()
            front_mask = np.zeros(len(obs), bool)
            for sid, g in obs.groupby("season_id"):
                gi = g.sort_values("date").index
                cut = int(np.ceil(0.6 * len(gi)))
                front_mask[obs.index.get_indexer(gi[:cut])] = True
            obs["_front"] = front_mask
            trc = obs[obs["_front"] & obs["D"].notna() & (~obs["censored"])]
            E = fit_E_leastsq(trc["sqrtTDD"].values, trc["D"].values)
            A = np.c_[np.ones(len(trc)), trc["sqrtTDD"].values]
            beta = np.linalg.lstsq(A, trc["D"].values, rcond=None)[0]
            tr_seqs = []
            for s in te_ids:
                gs = obs[obs.season_id == s].sort_values("date").reset_index(drop=True)
                cut = int(np.ceil(0.6 * len(gs)))
                tr_seqs.append(make_sequences(gs.iloc[:cut]))

        Xtr = np.concatenate([x for x, _, _ in tr_seqs if len(x)], axis=0) if tr_seqs else np.empty((0,))
        ytr = np.concatenate([y for _, y, _ in tr_seqs if len(y)], axis=0) if tr_seqs else np.empty((0,))
        gru_ready = len(Xtr) >= 30
        if gru_ready:
            mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
            sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
            ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
            models = [train_gru(Xtr, ytr, mu, sd, ymu, ysd, s) for s in SEEDS]

        # ---- 예측: 각 test 시즌 ----
        for sid in te_ids:
            gs = obs[obs.season_id == sid].sort_values("date").reset_index(drop=True)
            if mode == "temporal":
                cut = int(np.ceil(0.6 * len(gs)))
                test_rows = gs.iloc[cut:]
            else:
                test_rows = gs
            valid = test_rows[test_rows["D"].notna() & (~test_rows["censored"])]
            if len(valid) == 0:
                continue
            site = gs["site_key"].iloc[0]
            syr = int(gs["season_year"].iloc[0])
            # (a) Stefan(pooled E): 전 사이트 공통 E. 사이트 열특성 무시 → 스케일 불일치.
            stef = E * valid["sqrtTDD"].values
            # (a') Stefan(site E): 해당 시즌 초기 관측으로 E 보정(fold-safe 캘리브레이션 구간).
            #   temporal=앞60%, profile=시즌 앞 min(40%,cut) 관측으로 E 역산 후 나머지 예측.
            gsc = gs[gs["D"].notna() & (~gs["censored"])].sort_values("date")
            if mode == "temporal":
                calib = gsc.iloc[:int(np.ceil(0.6 * len(gs)))]
            else:
                calib = gsc.iloc[:max(8, int(0.4 * len(gsc)))]
            E_site = fit_E_leastsq(calib["sqrtTDD"].values, calib["D"].values)
            stef_site = (E_site * valid["sqrtTDD"].values if np.isfinite(E_site)
                         else np.full(len(valid), np.nan))
            # (c) persistence: H일 전 D(forecast baseline). 물리·GRU와 동일 forecast 과제로 통일.
            #     H일 전 결측이면 직전 유효값(ffill). 참조로 전일(H=1 근사 nowcast)도 산출.
            gs2 = gs.set_index("date")
            Dser = gs2["D"].where(~gs2["censored"]).ffill()
            past_dates = valid["date"] - pd.Timedelta(days=H_FCST)
            persist = Dser.reindex(past_dates, method="ffill").values
            persist_now = Dser.shift(1).reindex(valid["date"]).values  # 전일(nowcast 참조)
            # (d) static regression
            stat = beta[0] + beta[1] * valid["sqrtTDD"].values
            # (b) GRU
            if gru_ready:
                Xte, yte, idxte = make_sequences(gs)
                pred_map = {}
                if len(Xte):
                    preds = np.mean([gru_predict(m, Xte, mu, sd, ymu, ysd) for m in models], axis=0)
                    dts = gs["date"].values[idxte]
                    pred_map = dict(zip(pd.to_datetime(dts), preds))
                gru = np.array([pred_map.get(pd.Timestamp(t), np.nan) for t in valid["date"]])
            else:
                gru = np.full(len(valid), np.nan)

            ph = valid["phase"].values if "phase" in valid else np.full(len(valid), "unknown")
            for mname, pv in [("stefan", stef), ("stefan_site", stef_site),
                              ("persistence", persist), ("persistence_now", persist_now),
                              ("static", stat), ("gru", gru)]:
                for dt, do, dp, phz in zip(valid["date"].values, valid["D"].values, pv, ph):
                    oof.append(dict(season_id=sid, site_key=site, season_year=syr,
                                    date=pd.Timestamp(dt), model=mname, fold=fold_name,
                                    D_obs=float(do), D_pred=(float(dp) if np.isfinite(dp) else np.nan),
                                    phase=str(phz), split=mode))
            # ---- EOS(최대깊이 도달일) 예측 ----
            obs_peak_day = valid.loc[valid["D"].idxmax(), "date"]
            obs_peak_D = float(valid["D"].max())
            # 각 모델의 test-창 내 peak 예측일
            for mname, pv in [("stefan", stef), ("stefan_site", stef_site),
                              ("static", stat), ("gru", gru)]:
                if np.isfinite(pv).sum() >= 3:
                    pk = valid["date"].values[np.nanargmax(pv)]
                    eos.append(dict(season_id=sid, site_key=site, season_year=syr, model=mname,
                                    split=mode, obs_peak_day=pd.Timestamp(obs_peak_day),
                                    pred_peak_day=pd.Timestamp(pk), obs_peak_D=obs_peak_D,
                                    pred_peak_D=float(np.nanmax(pv))))
    return pd.DataFrame(oof), pd.DataFrame(eos)


def block_boot_rmse(g, n=NBOOT):
    """시즌 단위 블록 부트스트랩 RMSE 95% CI(누설통제: 시즌 재표집)."""
    sids = g["season_id"].unique()
    if len(sids) < 2:
        return (np.nan, np.nan)
    by = {s: g[g.season_id == s] for s in sids}
    out = []
    for _ in range(n):
        samp = pd.concat([by[s] for s in RNG.choice(sids, len(sids), replace=True)])
        d = samp.dropna(subset=["D_pred"])
        if len(d):
            out.append(np.sqrt(np.mean((d["D_obs"] - d["D_pred"]) ** 2)))
    if not out:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# ============================================================
# 4b. 정정 평가 하네스 (검증 지적 major #2·#3·#4 반영)
#   - eval_split_v2: fold 단위=profile(물리 위치), stefan_site calib 구간 scoring 제외.
#   - common_support_summary: 전 모델 공통 예측행(교집합)으로 restrict한 RMSE(분모 통일).
#   - EOS: 단조 모델은 창끝 고정으로 내부 peak 예측 불가 → GRU vs persistence(비단조)만 산출.
#   기존 eval_split 등은 무수정. 신규 함수만 추가한다.
# ============================================================
def _profile_of(season_id):
    """season_id='ID02_A|2025' → 물리 위치 profile 'ID02'(deployment 접미 제거)."""
    return season_id.split("|")[0].split("_")[0]


def eval_split_v2(panel, seasons, mode, fold_unit="profile"):
    """정정판 평가. eval_split과 예측 로직 동일하되 다음 두 정정 반영.

    (major #2) mode='profile'에서 fold 단위를 site_key(profile+deployment)가 아니라
      profile(물리 위치)로. 같은 지점의 A/B 재설치가 같은 fold에 묶여 test로 나가므로
      물리 위치 누설을 제거한다. te = 해당 profile로 시작하는 season_id 전부.
    (major #4) stefan_site 캘리브레이션 구간은 in-sample이므로 scoring에서 제외
      (calib_used 플래그로 표시, out-of-sample만 stefan_site 예측을 채운다).

    반환: (long OOF DataFrame, EOS DataFrame). 열은 eval_split과 동일 + calib_used.
    """
    oof, eos = [], []
    obs = panel[panel["season_id"].isin(seasons)].copy()
    obs["region"] = "Alaska"

    if mode == "profile":
        profiles = sorted({_profile_of(s) for s in seasons})
        folds = []
        for p in profiles:
            te_ids = [s for s in seasons if _profile_of(s) == p]
            tr_ids = [s for s in seasons if _profile_of(s) != p]
            if te_ids and tr_ids:
                folds.append((p, tr_ids, te_ids))
    else:
        folds = [("temporal", None, seasons)]

    for fold_name, tr_ids, te_ids in folds:
        if mode == "profile":
            tr = obs[obs["season_id"].isin(tr_ids)]
            trc = tr[tr["D"].notna() & (~tr["censored"])]
            E = fit_E_leastsq(trc["sqrtTDD"].values, trc["D"].values)
            A = np.c_[np.ones(len(trc)), trc["sqrtTDD"].values]
            beta = np.linalg.lstsq(A, trc["D"].values, rcond=None)[0]
            tr_seqs = [make_sequences(obs[obs.season_id == s]) for s in tr_ids]
        else:
            front_mask = np.zeros(len(obs), bool)
            for sid, g in obs.groupby("season_id"):
                gi = g.sort_values("date").index
                cut = int(np.ceil(0.6 * len(gi)))
                front_mask[obs.index.get_indexer(gi[:cut])] = True
            obs["_front"] = front_mask
            trc = obs[obs["_front"] & obs["D"].notna() & (~obs["censored"])]
            E = fit_E_leastsq(trc["sqrtTDD"].values, trc["D"].values)
            A = np.c_[np.ones(len(trc)), trc["sqrtTDD"].values]
            beta = np.linalg.lstsq(A, trc["D"].values, rcond=None)[0]
            tr_seqs = []
            for s in te_ids:
                gs = obs[obs.season_id == s].sort_values("date").reset_index(drop=True)
                cut = int(np.ceil(0.6 * len(gs)))
                tr_seqs.append(make_sequences(gs.iloc[:cut]))

        Xtr = np.concatenate([x for x, _, _ in tr_seqs if len(x)], axis=0) if tr_seqs else np.empty((0,))
        ytr = np.concatenate([y for _, y, _ in tr_seqs if len(y)], axis=0) if tr_seqs else np.empty((0,))
        gru_ready = len(Xtr) >= 30
        if gru_ready:
            mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
            sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
            ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-6)
            models = [train_gru(Xtr, ytr, mu, sd, ymu, ysd, s) for s in SEEDS]

        for sid in te_ids:
            gs = obs[obs.season_id == sid].sort_values("date").reset_index(drop=True)
            if mode == "temporal":
                cut = int(np.ceil(0.6 * len(gs)))
                test_rows = gs.iloc[cut:]
            else:
                test_rows = gs
            valid = test_rows[test_rows["D"].notna() & (~test_rows["censored"])]
            if len(valid) == 0:
                continue
            site = gs["site_key"].iloc[0]
            syr = int(gs["season_year"].iloc[0])
            stef = E * valid["sqrtTDD"].values
            # stefan_site: 캘리브레이션 구간 관측으로 E_site 역산. calib 구간은 in-sample.
            gsc = gs[gs["D"].notna() & (~gs["censored"])].sort_values("date")
            if mode == "temporal":
                calib = gsc.iloc[:int(np.ceil(0.6 * len(gs)))]
            else:
                calib = gsc.iloc[:max(8, int(0.4 * len(gsc)))]
            calib_dates = set(pd.to_datetime(calib["date"]).values)
            E_site = fit_E_leastsq(calib["sqrtTDD"].values, calib["D"].values)
            # (major #4) calib 구간은 scoring 제외 → 그 행 stefan_site 예측을 NaN으로.
            in_calib = np.array([pd.Timestamp(t).to_datetime64() in calib_dates
                                 for t in valid["date"]])
            if np.isfinite(E_site):
                stef_site = E_site * valid["sqrtTDD"].values
                stef_site = np.where(in_calib, np.nan, stef_site)
            else:
                stef_site = np.full(len(valid), np.nan)
            gs2 = gs.set_index("date")
            Dser = gs2["D"].where(~gs2["censored"]).ffill()
            past_dates = valid["date"] - pd.Timedelta(days=H_FCST)
            persist = Dser.reindex(past_dates, method="ffill").values
            persist_now = Dser.shift(1).reindex(valid["date"]).values
            stat = beta[0] + beta[1] * valid["sqrtTDD"].values
            if gru_ready:
                Xte, yte, idxte = make_sequences(gs)
                pred_map = {}
                if len(Xte):
                    preds = np.mean([gru_predict(m, Xte, mu, sd, ymu, ysd) for m in models], axis=0)
                    dts = gs["date"].values[idxte]
                    pred_map = dict(zip(pd.to_datetime(dts), preds))
                gru = np.array([pred_map.get(pd.Timestamp(t), np.nan) for t in valid["date"]])
            else:
                gru = np.full(len(valid), np.nan)

            ph = valid["phase"].values if "phase" in valid else np.full(len(valid), "unknown")
            for mname, pv in [("stefan", stef), ("stefan_site", stef_site),
                              ("persistence", persist), ("persistence_now", persist_now),
                              ("static", stat), ("gru", gru)]:
                for dt, do, dp, phz, ic in zip(valid["date"].values, valid["D"].values,
                                               pv, ph, in_calib):
                    oof.append(dict(season_id=sid, site_key=site, season_year=syr,
                                    date=pd.Timestamp(dt), model=mname, fold=fold_name,
                                    D_obs=float(do), D_pred=(float(dp) if np.isfinite(dp) else np.nan),
                                    phase=str(phz), split=mode,
                                    calib_used=bool(ic and mname == "stefan_site")))
            # ---- EOS(최대깊이 도달일): 내부 peak 예측 (major #3) ----
            #   단조 모델(stefan/stefan_site/static)은 √cumTDD 단조증가라 argmax(pred)가
            #   항상 창 마지막 날 → 내부 peak 예측 불가. 따라서 EOS 비교는 GRU·persistence
            #   (비단조)만. persistence의 peak=관측을 H일 지연 반영(비단조).
            obs_peak_day = valid.loc[valid["D"].idxmax(), "date"]
            obs_peak_D = float(valid["D"].max())
            for mname, pv in [("gru", gru), ("persistence", persist)]:
                if np.isfinite(pv).sum() >= 3:
                    pk = valid["date"].values[np.nanargmax(pv)]
                    eos.append(dict(season_id=sid, site_key=site, season_year=syr, model=mname,
                                    split=mode, obs_peak_day=pd.Timestamp(obs_peak_day),
                                    pred_peak_day=pd.Timestamp(pk), obs_peak_D=obs_peak_D,
                                    pred_peak_D=float(np.nanmax(pv))))
    df_oof = pd.DataFrame(oof)
    # calib 구간 stefan_site 행은 D_pred가 이미 NaN → dropna 시 자동 out-of-sample만 scoring.
    return df_oof, pd.DataFrame(eos)


def common_support_rows(oof):
    """전 모델 공통 예측행(교집합) 마스크를 (season_id,date) 키로 산출.

    각 (split) 안에서, 비교 대상 모든 모델이 non-NaN D_pred를 가진 (season_id,date)만
    공통 support로 남긴다(major #1: 분모 통일). stefan_site의 calib 제외행은 NaN이므로
    자연히 공통 support에서 빠진다. persistence_now(nowcast 참조)는 forecast 비교 대상이
    아니므로 공통 support 정의에서 제외하되 결과 표에는 그 공통행으로 재계산해 병기한다.
    반환: oof에 support 열(True=공통행) 추가한 복사본.
    """
    out = oof.copy()
    out["_support"] = False
    compare_models = ["stefan", "stefan_site", "persistence", "static", "gru"]
    for split, gsp in oof.groupby("split"):
        present = [m for m in compare_models if m in gsp["model"].unique()]
        keysets = []
        for m in present:
            d = gsp[(gsp.model == m)].dropna(subset=["D_pred"])
            keysets.append(set(zip(d["season_id"], pd.to_datetime(d["date"]))))
        if not keysets:
            continue
        common = set.intersection(*keysets)
        idx = out[out.split == split].index
        keys = list(zip(out.loc[idx, "season_id"], pd.to_datetime(out.loc[idx, "date"])))
        out.loc[idx, "_support"] = [k in common for k in keys]
    return out


# ============================================================
# 5. 실행
# ============================================================
def main():
    panel = build_dt_panel()
    seasons = select_seasons(panel)
    print(f"[panel] {len(panel)}행, 모델링 시즌 {len(seasons)}개: {seasons}")

    # 실측 참조 분모 검증(minor #5): 실제 시즌 peak 평균이 REF_PEAK_D와 근접한지 기록.
    obs_all = panel[panel.season_id.isin(seasons) & panel.D.notna() & (~panel.censored)]
    ref_peak_actual = float(obs_all.groupby("season_id")["D"].max().mean())

    allrows, alleos, summ = [], [], []
    for mode in ["profile", "temporal"]:
        # (major #2·#4) fold 단위=profile(물리 위치) + stefan_site calib 구간 scoring 제외.
        oof, eos = eval_split_v2(panel, seasons, mode, fold_unit="profile")
        oof = common_support_rows(oof)   # (major #1) 공통 예측행 마스크
        allrows.append(oof)
        alleos.append(eos)
        # phase×support 4조합: (all·deepening) × (per_model 분모 · common 공통행 분모).
        for phase_sel in ["all", "deepening"]:
            base = oof if phase_sel == "all" else oof[oof.phase == "deepening"]
            for support in ["per_model", "common"]:
                src = base if support == "per_model" else base[base["_support"]]
                for mname, g in src.groupby("model"):
                    d = g.dropna(subset=["D_pred"])
                    if len(d) == 0:
                        continue
                    m = all_metrics(d["D_obs"].values, d["D_pred"].values)
                    lo, hi = block_boot_rmse(d)
                    mean_D = float(d["D_obs"].mean())
                    summ.append(dict(
                        split=mode, phase=phase_sel, support=support, model=mname, **m,
                        rmse_pct=round(100 * m["rmse_cm"] / REF_PEAK_D, 1),          # 고정 참조 분모
                        rmse_over_mean_pct=round(100 * m["rmse_cm"] / mean_D, 1),    # rmse/mean 병기
                        mean_D_cm=round(mean_D, 1),
                        rmse_ci_lo=round(lo, 2), rmse_ci_hi=round(hi, 2)))
    res = pd.concat(allrows, ignore_index=True)
    eos_all = pd.concat(alleos, ignore_index=True)
    summ = pd.DataFrame(summ)
    res.to_csv(PROC / "e2_seasonal_dt_results.csv", index=False)
    eos_all.to_csv(PROC / "e2_seasonal_dt_eos.csv", index=False)
    summ.to_csv(PROC / "e2_seasonal_dt_summary.csv", index=False)

    # EOS 요약(일 오차): 단조 모델 제외, GRU vs persistence(비단조)만 (major #3).
    eos_all["day_err"] = (eos_all["pred_peak_day"] - eos_all["obs_peak_day"]).dt.days
    eos_summ = (eos_all.groupby(["split", "model"])["day_err"]
                .agg(mae_days=lambda x: np.mean(np.abs(x)),
                     med_days=lambda x: np.median(np.abs(x)), n="size").reset_index())

    cols = ["split", "phase", "support", "model", "n", "rmse_cm", "rmse_pct",
            "rmse_over_mean_pct", "r2", "skill_over_mean", "rmse_ci_lo", "rmse_ci_hi"]
    print(f"[ref] REF_PEAK_D={REF_PEAK_D} vs 실측 시즌peak평균 {ref_peak_actual:.2f}")
    print("\n=== D(t) RMSE (전 시즌·공통 support, 헤드라인 비교) ===")
    print(summ[(summ.phase == "all") & (summ.support == "common")][cols].to_string(index=False))
    print("\n=== D(t) RMSE (deepening·공통 support) ===")
    print(summ[(summ.phase == "deepening") & (summ.support == "common")][cols].to_string(index=False))
    print("\n=== D(t) RMSE (per-model 분모, 참고) ===")
    print(summ[(summ.phase == "all") & (summ.support == "per_model")][cols].to_string(index=False))
    print("\n=== EOS(내부 peak day) GRU vs persistence만 ===")
    print(eos_summ.to_string(index=False))

    # ---------- meta ----------
    meta = dict(
        purpose="E2 계절 내 융해 진행 D(t) 예측. timelapse 기반 ALT의 물리적 재정의(KPDC 콘슬 일별)",
        framing=("콘슬=알래스카 in-domain(전이 아님). 핵심 대조=연도간 최대 ALT 외삽은 예측불가"
                 "(S9 anomaly corr 0.06)이나 계절 내 D(t)는 관측 가능한 구조가 있다. "
                 "단 지온 유래 √cumTDD·2년 표본·단순 Stefan 조건에서 물리가 지속·GRU를 유의 초과하지는 "
                 "못한다(부분 성과). 물리는 deepening 구간 계절형태 포착에 유효. 과대해석 금지."),
        label=("D(t)=표층 연결 0°C 등온선 깊이(위 양수·아래 비양수 첫 교차 선형보간, s7 규칙). "
               "표층(10cm) 동결→0, 전층 양수→검열(제외). 융해기 5-10월."),
        forcing=("10cm 지온에서 파생한 융해개시 이후 누적 TDD·√cumTDD·일TDD증분. "
                 "GRU는 과거 %d일 시퀀스 [10cm지온·일TDD·√cumTDD·doy sin/cos]. "
                 "표준화·E·회귀계수 전부 train fold 내부(fold-safe)." % LAG),
        models=dict(
            stefan="pooled Stefan D=E·√cumTDD, E=train fold 전 사이트 공통 최소제곱(fold-safe). 사이트 열특성 무시.",
            stefan_site=("site Stefan D=E_site·√cumTDD, E_site=해당 시즌 초기 캘리브레이션 구간 관측으로 역산"
                         "(temporal=앞60%·profile=시즌 앞≥8일). 사이트 E가 알려진 물리 상한."),
            gru="소형 GRU(hidden24), 과거 lag→당일 D, 3seed 평균, fold내부 표준화",
            persistence=("%d일 전 비검열 D(ffill). forecast 지평 H=%d에서 물리·GRU와 동일 과제로 비교. "
                         "D(t)는 일간 자기상관 강해 nowcast(전일)는 사소하게 우수하나, forecast(H일)는 "
                         "계절 동역학을 못 담아 열화 → 물리·GRU의 가치가 드러남." % (H_FCST, H_FCST)),
            persistence_now="전일 비검열 D(H≈1, nowcast 참조). 사소 우위이나 forecast 아님(비교 기준선).",
            static="당일 √cumTDD→D 선형회귀(train 계수)"),
        validation=dict(
            profile=("profile(물리 위치) leave-one-out(공간 일반화). fold 단위=profile로 같은 지점 "
                     "A/B 재설치가 함께 test(major #2, 물리 위치 누설 제거)."),
            temporal="각 시즌 앞60%→뒤40% 시간분할(시간 외삽), E·회귀 pooled 앞부분"),
        n_modeling_seasons=len(seasons),
        modeling_seasons=seasons,
        n_sites=int(panel[panel.season_id.isin(seasons)].site_key.nunique()),
        season_years=sorted(int(y) for y in panel[panel.season_id.isin(seasons)].season_year.unique()),
        summary=summ.to_dict("records"),
        eos_day_error=eos_summ.to_dict("records"),
        rmse_pct_def=("rmse_pct 분모 = 모델링 12시즌 평균 관측 최대깊이 REF_PEAK_D=%.2fcm(고정 참조, "
                      "모델·split·support 무관 동일). rmse_over_mean_pct = rmse/해당 표본 평균 D 병기." % REF_PEAK_D),
        ref_peak_d_cm=REF_PEAK_D,
        ref_peak_d_actual_cm=round(ref_peak_actual, 2),
        common_support_note=("헤드라인 비교는 support='common'(전 비교모델 공통 예측행 교집합)으로 분모 통일. "
                             "per_model 행은 각 모델 non-NaN 예측행(표본·n 상이)으로 참고용. 각 행에 n 병기."),
        eos_note=("EOS(내부 peak day) 비교는 GRU vs persistence(비단조)만. stefan/stefan_site/static은 "
                  "√cumTDD 단조증가라 argmax(pred)가 항상 창 마지막 날 → 내부 peak 예측 불가(창끝 고정). "
                  "따라서 단조 모델은 EOS 비교에서 제외하고 'peak=창끝 고정'으로 명시."),
        key_finding=_describe_finding(summ, eos_summ),
        caveats=[
            "콘슬 프로파일 좌표 미상(사이트 대표점). 알래스카 in-domain 점검증, 전이 근거 아님.",
            "2023-2025 2년·모델링 시즌 %d개(대부분 2025 배치B 심부 16층). 표본 소.",
            ("profile leave-out fold 단위를 profile(물리 위치)로 변경(major #2). 같은 지점 A/B 재설치가 "
             "같은 fold로 묶여 물리 위치 누설 제거. 다만 profile당 시즌 수가 적어(대부분 1~2) fold별 "
             "test 표본이 작다 → CI 넓음, 해석 보수화."),
            "deepening 층화는 peak일을 사후 정의하므로 look-ahead 성격(minor #6). 배포 대표지표는 phase=all.",
            "D(t)는 센서 간격(10cm) 이산화로 얕은 층 계단 잡음(RMSE%%의 일부는 이산화).",
            "배치 A 2023은 피크꼬리·검열 다수 → 모델링 채택 시즌에서 대부분 제외.",
            "겨울/재동결 구간은 D(t) 정의상 표층 동결=0 또는 검열, 융해기 창(5-10월)만 사용.",
        ],
        seeds=SEEDS, lag=LAG, h_fcst_days=H_FCST, min_dt_days=MIN_DT_DAYS, min_peak=MIN_PEAK,
        nboot=NBOOT, smoke=SMOKE, device=str(DEV), gpu=GPU,
        runtime_s=round(time.time() - T0, 1),
    )
    with open(PROC / "e2_seasonal_dt_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    # ---------- 그림 ----------
    make_figures(panel, seasons, res, eos_all)
    print(f"\n[done] figures -> {FIGD}, runtime {time.time() - T0:.1f}s")


def _describe_finding(summ, eos_summ=None):
    """실제 수치로 결론 문장 생성(과대해석 금지). 헤드라인=profile·common support 기준.

    정정 반영: 모델별 분모 불일치 제거 → common support로 통일(major #1). fold 단위=profile
    (major #2). EOS는 GRU vs persistence만(major #3, 단조 모델은 내부 peak 예측 불가).
    """
    def get(split, model, phase="all", support="common", col="rmse_cm"):
        s = summ[(summ.split == split) & (summ.model == model)
                 & (summ.phase == phase) & (summ.support == support)]
        return float(s[col].iloc[0]) if len(s) else np.nan
    # 헤드라인: 전 모델 공통 예측행(common support) 기준(분모 통일).
    st_all = get("profile", "stefan_site", "all", "common")
    st_deep = get("profile", "stefan_site", "deepening", "common")
    pooled_deep = get("profile", "stefan", "deepening", "common")
    pers = get("profile", "persistence", "all", "common")
    persn = get("profile", "persistence_now", "all", "per_model")  # nowcast 참조(공통 support 제외군)
    gru = get("profile", "gru", "all", "common")
    # EOS: GRU vs persistence 내부 peak MAE(일).
    eos_txt = ""
    if eos_summ is not None and len(eos_summ):
        ep = eos_summ[eos_summ.split == "profile"].set_index("model")
        g_mae = float(ep.loc["gru", "mae_days"]) if "gru" in ep.index else np.nan
        p_mae = float(ep.loc["persistence", "mae_days"]) if "persistence" in ep.index else np.nan
        rel = "대등~열세" if not (np.isfinite(g_mae) and np.isfinite(p_mae) and g_mae < p_mae) else "우위"
        eos_txt = ("EOS(내부 최대깊이 도달일)는 GRU만 내부 peak를 예측할 수 있으며(MAE {:.0f}일), "
                   "단조 물리(Stefan·정적)는 argmax가 창끝에 고정돼 내부 peak를 예측할 수 없다. "
                   "다만 GRU의 일 오차는 persistence(H일 지연 baseline, MAE {:.0f}일) 대비 {}다"
                   "(내부 peak 예측 가능성 자체가 차별점, 정확도 우위는 아님). ".format(g_mae, p_mae, rel))
    return (
        "계절 내 D(t)는 연도간 최대 ALT 외삽(예측불가, S9 anomaly corr 0.06)과 달리 계절 궤적이 "
        "관측 가능한 구조를 가진다. 전 모델 공통 예측행(common support, 분모 통일) 기준으로, "
        "√cumTDD 단조 Stefan은 봄→피크 deepening 구간에서 site E RMSE {:.0f}cm(pooled E {:.0f}cm)로 "
        "계절 형태를 포착하나, 전 시즌으로 넓히면 가을 재동결 비단조성으로 열화한다(profile all {:.0f}cm). "
        "D(t)는 일간 자기상관이 강해 전일 지속(nowcast {:.0f}cm)이 사소하게 최강이고, "
        "{}일 forecast 지속({:.0f}cm)·GRU({:.0f}cm) 대비 물리는 대등~열세다. "
        "{}"
        "결론: 계절 내 예측가능성은 실증되나(외삽과 대조), 지온 forcing·2년 소표본·pooled 물리 조건에서 "
        "물리·GRU가 persistence를 유의 초과하지는 못한다(부분 성과). EOS는 GRU만 내부 peak 예측 가능."
        .format(st_deep, pooled_deep, st_all, persn, H_FCST, pers, gru, eos_txt))


# ============================================================
# 6. 시각화
# ============================================================
def make_figures(panel, seasons, res, eos_all):
    """논문 게재용 그림 3종. 데이터·수치 로직은 건드리지 않고 표현만 담당한다.

    표기 규범: 그림 내부에 내부 단계코드·영문 약칭을 남기지 않는다(RMSE·Stefan·GRU·TDD 등
    표준 학술 용어는 유지). 저채도 냉색 4단계 + 회색조 보조, 값 라벨은 소형 회색.
    """
    import matplotlib
    matplotlib.use("Agg")
    from polar.plotstyle import use_polar
    import matplotlib.pyplot as plt
    plt = use_polar()
    plt.rcParams.update({
        "axes.titleweight": "normal",
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.color": "#c9d1d9",
        "axes.edgecolor": "#8a8f96",
        "axes.linewidth": 0.7,
    })
    # 저채도 냉색 팔레트(진청→연청회) + 무채색 보조.
    C_DEEP, C_TEAL, C_SLATE, C_PALE = "#2f4b6e", "#4a7c8c", "#8296a8", "#b8c4d0"
    C_GRAY, C_LAB, C_FAINT = "#6e6e6e", "#444444", "#7a8590"

    def _clean(ax, grid="both"):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.7)
            ax.spines[s].set_color("#8a8f96")
        ax.grid(axis=grid, alpha=0.25, lw=0.5, color="#c9d1d9")
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.7, colors=C_LAB, labelsize=9.5)

    def _panel_tag(ax, i, text):
        """(a) 패널 식별자만 굵게, 설명은 보통 굵기."""
        ax.set_title(f"$\\bf{{({chr(97 + i)})}}$   {text}", loc="left",
                     fontsize=10.5, color="#333333", pad=6)

    def _site_label(season_id):
        """'ID23_B|2025' → 'ID23 지점 · 2025년'(내부 배치 접미 비노출)."""
        site, yr = season_id.split("|")
        return f"{site.split('_')[0]} 지점 · {yr}년"

    obs = panel[panel.season_id.isin(seasons)]

    # ------------------------------------------------------------------
    # (i) 대표 프로파일의 융해면 깊이 D(t): 관측 vs Stefan vs GRU
    #     x축은 4패널 공통(융해기 day-of-year). 프로파일마다 자료 길이가 다른 것은 렌더
    #     문제가 아니라 로거 기록 종료 시점 차이라는 자료 특성이다(예: ID02_B는 2025-06-17
    #     종료). 공통 축 위에서 기록 종료 이후를 무채색으로 비워 그 차이가 읽히게 하되,
    #     기록이 융해기 후반까지 남지 않은 시즌은 대표 패널에서 제외해 패널 간 정보밀도가
    #     크게 어긋나지 않게 한다(선정 기준은 그림 하단 주석에 명시).
    # ------------------------------------------------------------------
    prof = res[res.split == "profile"]

    def _doy(dates):
        return pd.to_datetime(pd.Series(np.asarray(dates))).dt.dayofyear.values

    LAST_DOY_MIN = 205        # 대표 패널 후보: 관측이 최소 7월 하순까지 남은 시즌
    b2025 = obs[(obs.season_year == 2025) & obs.site_key.str.endswith("_B")]
    bv = b2025[b2025.D.notna() & (~b2025.censored)]
    cand = bv.groupby("season_id").agg(peak=("D", "max"), last=("date", "max"))
    cand["last_doy"] = pd.to_datetime(cand["last"]).dt.dayofyear
    ok = cand[cand["last_doy"] >= LAST_DOY_MIN].sort_values("peak", ascending=False)
    if len(ok) >= 4:
        # 최대 융해깊이 범위를 고르게 대표하도록 목표 peak를 등간격으로 두고 최근접 선택.
        targets = np.linspace(ok["peak"].max(), ok["peak"].min(), 4)
        rep_ids = []
        for t in targets:
            for sid in (ok["peak"] - t).abs().sort_values().index:
                if sid not in rep_ids:
                    rep_ids.append(sid)
                    break
    else:
        rep_ids = list(obs.groupby("season_id")["D"].max().sort_values(ascending=False).index[:4])
    rep_ids = list(dict.fromkeys(rep_ids))[:4]

    MON_DOY = [(121, "5월"), (152, "6월"), (182, "7월"),
               (213, "8월"), (244, "9월"), (274, "10월")]

    _rep_last = max(int(_doy(obs[obs.season_id == s]["date"]).max()) for s in rep_ids)
    XLO, XHI = 118, _rep_last + 9
    DMAX = 160.0     # 최심센서 160 cm. Stefan 과대추정 곡선은 이 아래에서 축 밖으로 나간다.
    YLO, YHI = DMAX + 16, -6

    def _gapped(dates, vals, maxgap=2):
        """관측 결측·검열로 날짜가 끊긴 구간에 NaN을 넣어 선을 잇지 않는다."""
        x = _doy(dates).astype(float)
        y = np.asarray(vals, dtype=float)
        if len(x) < 2:
            return x, y
        xo, yo = [x[0]], [y[0]]
        for i in range(1, len(x)):
            if x[i] - x[i - 1] > maxgap:
                xo.append(np.nan)
                yo.append(np.nan)
            xo.append(x[i])
            yo.append(y[i])
        return np.array(xo), np.array(yo)

    MODEL_STYLE = [("stefan", C_SLATE, (0, (5, 2.2)), 1.3, "Stefan(공통 계수)"),
                   ("stefan_site", C_TEAL, (0, (6, 1.6, 1.4, 1.6)), 1.3, "Stefan(지점 계수)"),
                   ("gru", C_GRAY, (0, (1.4, 1.8)), 1.4, "GRU")]

    # 하단에 범례 1줄 + 주석 2줄을 겹치지 않게 두기 위해 여백을 명시적으로 잡는다.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.072, right=0.985, top=0.955, bottom=0.205,
                        hspace=0.22, wspace=0.055)
    for pi, (ax, sid) in enumerate(zip(axes.ravel(), rep_ids)):
        g = obs[obs.season_id == sid].sort_values("date")
        gv = g[g.D.notna() & (~g.censored)]
        ox, oy = _gapped(gv["date"], gv["D"])
        ax.plot(ox, oy, "-", color=C_DEEP, lw=1.4, label="관측", zorder=4)
        ax.scatter(_doy(gv["date"]), gv["D"], s=6, color=C_DEEP, lw=0, zorder=5)
        if len(gv):
            pk = int(_doy([gv.loc[gv["D"].idxmax(), "date"]])[0])
            # 기준선은 무채색 실선(범례 혼동 방지), 범례 기호는 축 상단 ▼ 표식으로 구분.
            ax.axvline(pk, color="#c6ccd2", ls="-", lw=0.8, zorder=1)
            ax.plot([pk], [1.0], marker="v", ms=5, ls="none", color=C_SLATE, mec="none",
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=6,
                    label=("관측 최대깊이 도달일" if pi == 0 else "_nolegend_"))
        for mname, col, ls, lw, lab in MODEL_STYLE:
            pm = prof[(prof.season_id == sid) & (prof.model == mname)].sort_values("date")
            pm = pm.dropna(subset=["D_pred"])
            if len(pm):
                px, py = _gapped(pm["date"], pm["D_pred"].clip(upper=DMAX + 26))
                ax.plot(px, py, ls=ls, color=col, lw=lw,
                        label=(lab if pi == 0 else "_nolegend_"), zorder=3)
            elif mname == "stefan_site":
                ax.text(0.03, 0.06, "Stefan(지점 계수) 미표시: 보정 구간 표본 부족",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=8.5, color=C_FAINT)
        # 로거 기록 종료 이후(관측 없음) 구간을 옅은 무채색으로 비워 표시
        rec_end = int(_doy([g["date"].max()])[0])
        if rec_end < XHI - 8:
            ax.axvspan(rec_end, XHI, color="#9aa3ab", alpha=0.09, lw=0, zorder=0)
            mmdd = pd.Timestamp(g["date"].max())
            # 라벨은 패널 아래쪽(곡선이 오지 않는 영역)에 두고, 회색 구간이 넓으면 그 안에
            # 좌측 정렬, 좁으면 경계 바로 왼쪽에 우측 정렬해 잘리지 않게 한다.
            wide = (XHI - rec_end) >= 38
            ax.text(rec_end + (3 if wide else -3), 0.035,
                    f"관측 종료 {mmdd.month}/{mmdd.day}",
                    transform=ax.get_xaxis_transform(), fontsize=8.5, color=C_FAINT,
                    ha=("left" if wide else "right"), va="bottom")
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(YLO, YHI)               # y 반전(깊이 아래로)
        ax.set_yticks([0, 50, 100, 150])
        ax.set_xticks([d for d, _ in MON_DOY if d <= XHI - 4])
        ax.set_xticklabels([m for d, m in MON_DOY if d <= XHI - 4])
        _clean(ax)
        _panel_tag(ax, pi, _site_label(sid))
    for ax in axes[:, 0]:
        ax.set_ylabel("융해면 깊이 D(t) (cm)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.125), ncol=5,
               frameon=False, fontsize=9.5, handlelength=2.6, columnspacing=1.8)
    fig.text(0.5, 0.075,
             "회색 구간은 로거 기록이 종료돼 관측이 없는 기간이며, 선이 끊긴 곳은 결측·검열로 "
             "D(t)를 정의할 수 없는 날이다. 봄·가을에 D(t)가 0으로 떨어지는 날은\n"
             "10 cm 지온 일평균이 0 °C 이하인 표층 재동결일이며 보간 아티팩트가 아니다. "
             "대표 프로파일 4개는 관측이 7월 하순 이후까지 남은 시즌 가운데\n"
             "최대 융해깊이 범위를 고르게 대표하도록 선정했다. 전체 12개 시즌 집계는 "
             "RMSE 비교 그림에 제시한다.",
             ha="center", va="top", fontsize=8.2, color=C_FAINT, linespacing=1.7)
    fig.savefig(FIGD / "e2_dt_curves.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGD / "e2_dt_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # (ii) 모델별 D(t) 예측 RMSE (지점 분리 검증 · 시간 분할 검증)
    #      값 선택 로직은 기존과 동일(공통 구간 기준, 전일 지속만 참조 표본).
    # ------------------------------------------------------------------
    summ = pd.read_csv(PROC / "e2_seasonal_dt_summary.csv")
    summ_c = summ[(summ.phase == "all") & (summ.support == "common")]
    summ_pm = summ[(summ.phase == "all") & (summ.support == "per_model")]
    fig, ax = plt.subplots(figsize=(11.0, 7.0), constrained_layout=True)
    order = ["persistence_now", "persistence", "static", "stefan", "stefan_site", "gru"]
    labmap = {"persistence_now": "지속\n(전일 값)",
              "persistence": f"지속\n({H_FCST}일 전 값)",
              "static": "선형회귀\n(당일 √TDD)",
              "stefan": "Stefan\n(공통 계수)",
              "stefan_site": "Stefan\n(지점 계수)",
              "gru": f"GRU\n(과거 {LAG}일 시퀀스)"}
    colmap = {"profile": C_DEEP, "temporal": C_SLATE}
    labsplit = {"profile": "지점 분리 검증", "temporal": "시간 분할 검증(앞→뒤)"}
    w = 0.38
    xs = np.arange(len(order))
    top = 0.0
    for k, split in enumerate(["profile", "temporal"]):
        sc = summ_c[summ_c.split == split].set_index("model")
        spm = summ_pm[summ_pm.split == split].set_index("model")

        def _row(m, col):
            if m == "persistence_now":
                return spm.loc[m, col] if m in spm.index else np.nan
            return sc.loc[m, col] if m in sc.index else np.nan

        vals = [_row(m, "rmse_cm") for m in order]
        los = [_row(m, "rmse_ci_lo") for m in order]
        his = [_row(m, "rmse_ci_hi") for m in order]
        ns = [_row(m, "n") for m in order]
        pcts = [_row(m, "rmse_pct") for m in order]
        yerr = np.array([[v - l for v, l in zip(vals, los)],
                         [h - v for v, h in zip(vals, his)]])
        yerr = np.where(np.isfinite(yerr), yerr, 0)
        ax.bar(xs + (k - 0.5) * w, vals, w, yerr=yerr, capsize=2.5,
               color=colmap[split], lw=0, label=labsplit[split], zorder=2,
               error_kw=dict(ecolor="#5a636c", elinewidth=0.9, capthick=0.9))
        for x, v, m, nn, hi, pc in zip(xs + (k - 0.5) * w, vals, order, ns, his, pcts):
            if not np.isfinite(v):
                continue
            ytop = hi if np.isfinite(hi) else v
            top = max(top, ytop)
            # RMSE(cm)를 주 라벨로, 참조 깊이 대비 비율과 표본 수는 한 줄 보조 라벨로 병기.
            # 이중축(cm/%)은 두 눈금의 숫자가 비슷해 오독을 유발하므로 사용하지 않는다.
            ax.annotate(f"{v:.1f}", (x, ytop), ha="center", va="bottom",
                        fontsize=9, color=C_LAB, xytext=(0, 3),
                        textcoords="offset points")
            sub = []
            if np.isfinite(pc):
                sub.append(f"{pc:.0f}%")
            if np.isfinite(nn):
                sub.append(f"n {int(nn)}")
            if sub:
                ax.annotate(" · ".join(sub), (x, ytop), ha="center", va="bottom",
                            fontsize=7.5, color=C_FAINT, xytext=(0, 14),
                            textcoords="offset points")
    ax.set_xticks(xs)
    ax.set_xticklabels([labmap[m] for m in order], fontsize=9.5)
    ax.set_ylabel("D(t) 예측 RMSE (cm)")
    ax.set_ylim(0, top * 1.30)
    _clean(ax, grid="y")
    ax.legend(loc="upper left", fontsize=9.5, frameon=False, ncol=1,
              bbox_to_anchor=(0.0, 1.0))
    ax.text(0.0, -0.155,
            f"오차막대: 시즌 단위 블록 부트스트랩 95% 신뢰구간({NBOOT:,}회).  "
            f"막대 위 작은 글씨의 % = 관측 최대깊이 평균 {REF_PEAK_D:.1f} cm 대비 RMSE 비율.\n"
            "n = 평가 표본 일수(전일 값 지속만 공통 구간 밖 참조 표본).",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=C_FAINT,
            linespacing=1.6)
    fig.savefig(FIGD / "e2_rmse_bars.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGD / "e2_rmse_bars.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # (iii) 최대깊이 도달일 산점: GRU vs 지속 기준선(단조 물리는 내부 정점 예측 불가)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 6.6), constrained_layout=True)
    ee = eos_all[eos_all.split == "profile"].copy()
    ee["obs_doy"] = ee["obs_peak_day"].dt.dayofyear
    ee["pred_doy"] = ee["pred_peak_day"].dt.dayofyear
    mk = {"gru": ("s", C_DEEP, "GRU"),
          "persistence": ("D", C_SLATE, f"지속({H_FCST}일 전 값)")}
    # 표식이 축 경계에서 잘려 다른 기호로 오인되지 않도록 자료 범위 바깥까지 여유를 둔다.
    _v = pd.concat([ee["obs_doy"], ee["pred_doy"]]).astype(float)
    lo = float(np.floor((_v.min() - 12) / 10) * 10)
    hi = float(np.ceil((_v.max() + 12) / 10) * 10)
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color="#9aa3ab", lw=0.9,
            label="1:1", zorder=1)
    for mname, (mkr, col, lab) in mk.items():
        sub = ee[ee.model == mname]
        if len(sub):
            mae = np.mean(np.abs(sub["pred_doy"] - sub["obs_doy"]))
            nseas = sub["season_id"].nunique()
            ax.scatter(sub["obs_doy"], sub["pred_doy"], marker=mkr, s=42, color=col,
                       edgecolor="white", lw=0.4, alpha=0.9, zorder=3, clip_on=False,
                       label=f"{lab} · 평균절대오차 {mae:.0f}일 (n {nseas})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("관측 최대깊이 도달일 (연중 일수)")
    ax.set_ylabel("예측 최대깊이 도달일 (연중 일수)")
    ax.set_aspect("equal")
    _clean(ax)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    ax.text(0.0, -0.115,
            "n = 지점-연도 수(지점 분리 검증).  Stefan·선형회귀는 예측이 누적 융해도일에 대해 "
            "단조 증가해\n구간 내부 정점을 예측할 수 없으므로 비교에서 제외했다.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=C_FAINT,
            linespacing=1.6)
    fig.savefig(FIGD / "e2_eos_scatter.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGD / "e2_eos_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def figs_only():
    """CPU 전용 그림 재생성: 기존 CSV만 읽어 make_figures 호출(모델 재학습 없음).

    build_dt_panel·select_seasons는 순수 pandas/numpy(일별 지온 CSV → D(t) 패널)로
    torch·GPU가 필요없다. e2_seasonal_dt_results.csv·e2_seasonal_dt_eos.csv를 그대로
    사용하며 수치·요약 CSV·meta는 재계산하지 않는다."""
    panel = build_dt_panel()
    seasons = select_seasons(panel)
    res = pd.read_csv(PROC / "e2_seasonal_dt_results.csv", parse_dates=["date"])
    eos_all = pd.read_csv(PROC / "e2_seasonal_dt_eos.csv",
                          parse_dates=["obs_peak_day", "pred_peak_day"])
    make_figures(panel, seasons, res, eos_all)
    print(f"[figs-only] figures -> {FIGD}, runtime {time.time() - T0:.1f}s")


if __name__ == "__main__":
    if FIGS_ONLY:
        figs_only()
    else:
        main()
