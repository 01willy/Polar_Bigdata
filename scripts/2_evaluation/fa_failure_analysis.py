"""FA 실패 원인 분석: 라벨이 늘수록 악화하는 대상(캐나다·CA-3)과 원천 잔차가 해로운 대상(AL-2·AL-4·AL-6·레나 n=0).

계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md(V1 사전 점검, L7), 큰 틀 docs/RESEARCH_FRAME_2026-09-29.md.
성격: 탐색 분석. 기존 자료와 h25b 결과만 쓴다. 새 학습은 CatBoost(catboost_lo 설정) 소수 회와 ridge 뿐이다(CPU, 스레드 2).
대상·원천·분할 규약은 h25b 와 같다(하위 지역 k-means, LORO 원천 + 100 km 버퍼, 상위 지역 포함, 분할 0–2, 채점 = B 의 eval_mask 셀).

부(part)
  0 gate   : 하위 지역·원천·E0·|A|·채점 셀 수를 h3/subregions.csv·h25b_targets.csv 와 대조한다.
  1 block  : 블록 E 분포, 블록 E 대 공변량 순위상관, A/B 구성 차이, 라벨 출처(region·기관·기기·관측월) 구성.
  2 hetero : 지역 안 E 이질성 지표 14대상 + E 수축(S1 random, n = 40·160) 악화 크기와의 순위상관.
  3 v1     : 공변량 의존 계수 모형 log(y/s) ~ 표준화 x25 ridge(원천 ∪ 대상 A 라벨). 캐나다·CA-3, 분할 3, n ∈ {0, 40, 160, 전량}.
  4 resid  : 원천 잔차(E0 앵커, CatBoost) 예측의 편향·정렬과 대상의 공변량 위치(DI, AOA 밖 비율).
  5 splits : 무작위 블록 분할 200회(half_split_blocks seed 1–200)에서 A 전량 재적합 E 의 B 오차 변화 분포(해석적, 학습 없음).
             h25b 의 분할 3회와 LG 계획의 분할 seed 1–5 가 이 분포의 어디에 있는지 기록한다.
산출: data/processed/h4/fa_*.csv, fa_meta.json
실행: python3 scripts/2_evaluation/fa_failure_analysis.py [--parts 0,1,2,3,4]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, spatial_block_splits                          # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks    # noqa: E402
from polar.m1_ext import haversine_km                                            # noqa: E402
from polar.h4_common import BlockStore, boot_delta_blocks                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="0,1,5,2,3,4")
ap.add_argument("--n-rand-splits", type=int, default=200)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--cache", default="", help="원천 색인 캐시 npz 경로(없으면 계산 후 저장)")
args = ap.parse_args()
PARTS = set(int(v) for v in args.parts.split(","))
T0 = time.time()

PROC = ROOT / "data" / "processed"; H3 = PROC / "h3"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
SPLITS = [0, 1, 2]
KAPPA, BUFFER_KM, K_SUB = 10.0, 100.0, "Alaska:6,Canada:3,Lena:2"
FOCUS = ["Canada", "CA-3", "AL-2", "AL-4", "AL-6", "Lena", "LE-1", "LE-2", "AL-5"]
COV_SHOW = ["e5_tdd", "e5_maat", "sg_soc_0_5", "cci_alt", "e5_swe", "dem_elev", "sg_bdod_5_15"]

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)


# ---------------------------------------------------------------- 공통: 하위 지역·원천(h25_label_budget.py 와 같은 정의)
def make_subregions(df):
    from sklearn.cluster import KMeans
    sub = np.array(["" for _ in range(len(df))], dtype=object); rows = []
    for spec in K_SUB.split(","):
        reg, k = spec.split(":"); k = int(k)
        idx = np.where(df.macro.values == reg)[0]
        bt = df.iloc[idx].groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size")).reset_index()
        X = np.c_[bt.lat.values, bt.lon.values * np.cos(np.radians(bt.lat.values))]
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=np.sqrt(bt.n.values))
        lab = km.labels_
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])
        name_of = {c: f"{reg[:2].upper()}-{r + 1}" for r, c in enumerate(order)}
        b2s = dict(zip(bt.block.values, [name_of[c] for c in lab]))
        for i in idx:
            sub[i] = b2s[df.block.values[i]]
        for c in range(k):
            m = lab == c
            rows.append(dict(subregion=name_of[c], parent=reg, n_blocks=int(m.sum()), n_cells=int(bt.n.values[m].sum())))
    return sub, pd.DataFrame(rows)


DF["sub"], SUBS = make_subregions(DF)
SUB_NAMES = sorted(SUBS.subregion.tolist())
TARGETS = [t for t in MAIN4 + SUB_NAMES if t != "CA-1"]                          # 분석 14대상
T_H25B = pd.read_csv(H3 / "h25b_targets.csv")
CURVE = pd.read_csv(H3 / "h25b_curve.csv")


def target_idx(t):
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def parent_of(t):
    return t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])


def source_idx(t):
    t_idx = target_idx(t)
    src = np.where(np.isfinite(DF.y.values))[0]
    src = src[~np.isin(src, t_idx)]
    la, lo = DF.lat.values, DF.lon.values
    keep = np.ones(len(src), bool); tl, tn = la[t_idx], lo[t_idx]
    for j, i in enumerate(src):
        if abs(la[i] - tl).min() < 1.0:
            if haversine_km(la[i], lo[i], tl, tn).min() < BUFFER_KM:
                keep[j] = False
    return src[keep]


def load_sources():
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        return {t: z[t] for t in TARGETS}
    out = {t: source_idx(t) for t in TARGETS}
    if cache is not None:
        np.savez_compressed(cache, **out)
    return out


SRC = load_sources()


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


E0 = {t: ls_E(DF.y.values[SRC[t]], DF.s.values[SRC[t]]) for t in TARGETS}
VALID = {(r.target, int(r.split)): bool(r.valid) for r in T_H25B.itertuples()}


def splits_of(t, valid_only=True):
    out = []
    for sp in SPLITS:
        if valid_only and not VALID.get((t, sp), False):
            continue
        A_idx, B_idx = half_split_blocks(DF, target_idx(t), sp)
        evB = B_idx[eval_mask(DF.iloc[B_idx])]
        out.append((sp, A_idx, evB))
    return out


def sp_rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    r = spearmanr(a[m], b[m])
    return float(r.correlation), float(r.pvalue), int(m.sum())


# ---------------------------------------------------------------- 라벨 출처 메타(ABoVE 원자료: 기기·관측월·기관)
def attach_label_meta():
    raw = pd.read_csv(ROOT / "data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv",
                      usecols=["team_name", "survey_technique", "latitude", "longitude", "date", "ALT_instrument", "ALT"], low_memory=False)
    raw = raw[(raw.ALT != -9999) & raw.ALT.notna() & (raw.ALT > 0) & (raw.ALT < 300) & (raw.latitude != -9999)].copy()
    dt = pd.to_datetime(raw.date, errors="coerce")
    raw["month"], raw["year"] = dt.dt.month, dt.dt.year
    raw["klat"], raw["klon"] = raw.latitude.round(4), raw.longitude.round(4)
    raw["is_gpr"] = (raw.ALT_instrument == "GPR").astype(float)
    raw["is_probe"] = (raw.ALT_instrument == "Probe").astype(float)
    raw["early"] = (raw.month <= 7).astype(float)
    raw["team_name"] = raw.team_name.astype(str).str.strip("'")
    cm = raw.groupby(["klat", "klon"], as_index=False).agg(
        n_pts=("ALT", "size"), frac_gpr=("is_gpr", "mean"), frac_probe=("is_probe", "mean"), frac_early=("early", "mean"),
        month_mean=("month", "mean"), year_mean=("year", "mean"), team=("team_name", lambda v: v.mode().iloc[0]))
    DF["klat"], DF["klon"] = DF.lat.round(4), DF.lon.round(4)
    d = DF.merge(cm, on=["klat", "klon"], how="left")
    assert len(d) == len(DF)
    above = DF.region.isin(["ABoVE_AK", "ABoVE_CA"]).values
    instr = np.where(~above, np.where(DF.region.values == "Lena_RU", "탐침(ALLena)", "탐침(CALM 격자)"),
             np.where(d.frac_gpr.values >= 0.5, "GPR", np.where(d.frac_probe.values >= 0.5, "탐침(ABoVE)", "기기 미기재(ABoVE)")))
    season = np.where(~above, "계절 최대(원자료 정의)", np.where(d.frac_early.values >= 0.5, "6–7월", "8–9월"))
    DF["instr"], DF["season"] = instr, season
    DF["team"] = np.where(above, d.team.fillna("미상").values, DF.region.values)
    DF["n_pts"], DF["year_mean"] = d.n_pts.values, d.year_mean.values
    DF["above_matched"] = np.where(above, d.n_pts.notna().values, True)


attach_label_meta()
META = dict(stage="FA 실패 원인 분석(탐색)", plan=["docs/EXPERIMENT_PLAN_LG_2026-09-29.md", "docs/RESEARCH_FRAME_2026-09-29.md"],
            targets=TARGETS, splits=SPLITS, kappa=KAPPA, buffer_km=BUFFER_KM, k_sub=K_SUB, parts=sorted(PARTS),
            above_cells_unmatched=int((~DF.above_matched).sum()))


# ================================================================ 0) 재현 대조
def part0():
    ref_sub = pd.read_csv(H3 / "subregions.csv").set_index("subregion")
    rows = []
    for t in TARGETS:
        r = T_H25B[T_H25B.target == t]
        for sp in SPLITS:
            A_idx, B_idx = half_split_blocks(DF, target_idx(t), sp)
            evB = B_idx[eval_mask(DF.iloc[B_idx])]
            q = r[r.split == sp].iloc[0]
            rows.append(dict(target=t, split=sp, n_src=len(SRC[t]), n_src_ref=int(q.n_src), E0=E0[t], E0_ref=float(q.E0),
                             n_A=len(A_idx), n_A_ref=int(q.n_A), n_eval=len(evB), n_eval_ref=int(q.n_eval),
                             n_cells=len(target_idx(t)), n_cells_ref=int(ref_sub.loc[t, "n_cells"]) if t in ref_sub.index else int(q.n_cells)))
    g = pd.DataFrame(rows)
    g["ok"] = (g.n_src == g.n_src_ref) & (np.abs(g.E0 - g.E0_ref) < 1e-9) & (g.n_A == g.n_A_ref) & (g.n_eval == g.n_eval_ref) & (g.n_cells == g.n_cells_ref)
    g.to_csv(OUT / "fa_gate.csv", index=False)
    META["gate_ok"] = bool(g.ok.all()); META["gate_n_fail"] = int((~g.ok).sum())
    print(f"[0] 재현 대조: {int(g.ok.sum())}/{len(g)} 일치", flush=True)
    if not g.ok.all():
        print(g[~g.ok].to_string())


# ================================================================ 1) 블록 E·A/B 구성·출처
def block_table(t):
    d = DF.iloc[target_idx(t)]
    rows = []
    for b, sub in d.groupby("block"):
        y, s = sub.y.values, sub.s.values
        z = np.log(y / s)
        rows.append(dict(target=t, block=int(b), n=len(sub), lat=float(sub.lat.mean()), lon=float(sub.lon.mean()), E_blk=ls_E(y, s),
                         z_mean=float(np.mean(z)), z_var=float(np.var(z, ddof=1)) if len(sub) > 1 else np.nan, y_mean=float(y.mean()),
                         instr=sub.instr.mode().iloc[0], season=sub.season.mode().iloc[0], team=sub.team.mode().iloc[0],
                         frac_early=float((sub.season == "6–7월").mean()), frac_gpr=float((sub.instr == "GPR").mean()),
                         **{f: float(np.nanmean(sub[f].values.astype(float))) if np.isfinite(sub[f].values.astype(float)).any() else np.nan for f in FEATS}))
    bt = pd.DataFrame(rows)
    bt["E0"] = E0[t]; bt["log_ratio_E0"] = np.log(bt.E_blk / E0[t])
    return bt


def tau_moment(bt, min_cells=3):
    """블록 평균 z = log(y/s) 의 블록 간 SD(표본 잡음 보정, 적률). 셀 min_cells 이상 블록만."""
    b = bt[(bt.n >= min_cells) & np.isfinite(bt.z_var)]
    if len(b) < 3:
        return np.nan
    sig2 = float(np.sum((b.n - 1) * b.z_var) / np.sum(b.n - 1))
    v = float(np.var(b.z_mean, ddof=1)) - sig2 * float(np.mean(1.0 / b.n))
    return float(np.sqrt(max(v, 0.0)))


def part1():
    bts, summ, cors, abrows, prov, camp = [], [], [], [], [], []
    for t in TARGETS:
        bt = block_table(t); bts.append(bt)
        d = DF.iloc[target_idx(t)]
        b5 = bt[bt.n >= 5]
        lz = np.log(b5.E_blk.values)
        wsd = float(np.sqrt(np.average((lz - np.average(lz, weights=b5.n)) ** 2, weights=b5.n))) if len(b5) >= 2 else np.nan
        E_own = ls_E(d.y.values, d.s.values)
        summ.append(dict(target=t, parent=parent_of(t), n_cells=len(d), n_blocks=len(bt), n_blocks5=len(b5), E0=E0[t], E_own=E_own,
                         log_Eown_E0=float(np.log(E_own / E0[t])), E_blk_min=float(b5.E_blk.min()) if len(b5) else np.nan,
                         E_blk_med=float(b5.E_blk.median()) if len(b5) else np.nan, E_blk_max=float(b5.E_blk.max()) if len(b5) else np.nan,
                         sd_logE_blk=float(np.std(lz, ddof=1)) if len(b5) >= 2 else np.nan, sd_logE_blk_w=wsd, tau_logE=tau_moment(bt),
                         top_block_share=float(bt.n.max() / bt.n.sum()), cv_blk_sqrt_tdd=float(b5.e5_sqrt_tdd.std() / b5.e5_sqrt_tdd.mean()) if len(b5) >= 2 else np.nan,
                         frac_gpr=float((d.instr == "GPR").mean()), frac_probe=float(d.instr.str.startswith("탐침").mean()),
                         frac_instr_na=float((d.instr == "기기 미기재(ABoVE)").mean()), frac_early=float((d.season == "6–7월").mean())))
        if len(b5) >= 6:
            for f in FEATS + ["lat", "frac_early", "frac_gpr"]:
                rho, p, n = sp_rho(b5[f].values, b5.E_blk.values)
                cors.append(dict(target=t, covariate=f, rho=rho, p=p, n_blocks=n))
        # A/B 구성
        for sp, A_idx, evB in splits_of(t, valid_only=False):
            A, B = DF.iloc[A_idx], DF.iloc[evB]
            EA, EB = ls_E(A.y.values, A.s.values), ls_E(B.y.values, B.s.values)
            Ek = (len(A) * EA + KAPPA * E0[t]) / (len(A) + KAPPA)
            abrows.append(dict(target=t, split=sp, valid=VALID.get((t, sp), False), n_A=len(A), n_B=len(B), nb_A=int(A.block.nunique()), nb_B=int(B.block.nunique()),
                               E0=E0[t], E_A=EA, E_B=EB, E_kappa_allA=Ek, log_EA_EB=float(np.log(EA / EB)), log_E0_EB=float(np.log(E0[t] / EB)),
                               rmse_B_E0=rmse(B.y.values, E0[t] * B.s.values), rmse_B_EA=rmse(B.y.values, EA * B.s.values),
                               rmse_B_Ekappa=rmse(B.y.values, Ek * B.s.values), rmse_B_EB=rmse(B.y.values, EB * B.s.values),
                               lat_A=float(A.lat.mean()), lat_B=float(B.lat.mean()), tdd_A=float(A.e5_tdd.mean()), tdd_B=float(B.e5_tdd.mean()),
                               soc_A=float(np.nanmean(A.sg_soc_0_5)), soc_B=float(np.nanmean(B.sg_soc_0_5)), y_A=float(A.y.mean()), y_B=float(B.y.mean()),
                               early_A=float((A.season == "6–7월").mean()), early_B=float((B.season == "6–7월").mean()),
                               gpr_A=float((A.instr == "GPR").mean()), gpr_B=float((B.instr == "GPR").mean()),
                               north66_A=float((A.lat >= 66).mean()), north66_B=float((B.lat >= 66).mean())))
            for side, part in (("A", A), ("B", B)):                       # 분할별 관측 기관 구성
                for tm, sub in part.groupby("team"):
                    camp.append(dict(target=t, split=sp, valid=VALID.get((t, sp), False), side=side, team=tm, n_cells=len(sub), share=len(sub) / len(part),
                                     n_blocks=int(sub.block.nunique()), E=ls_E(sub.y.values, sub.s.values)))
        # 출처 구성
        for (rg, tm, ins, se), sub in d.groupby(["region", "team", "instr", "season"]):
            Eg = ls_E(sub.y.values, sub.s.values)
            prov.append(dict(target=t, region=rg, team=tm, instr=ins, season=se, n_cells=len(sub), share=len(sub) / len(d), n_blocks=int(sub.block.nunique()),
                             lat=float(sub.lat.mean()), lon=float(sub.lon.mean()), year_mean=float(np.nanmean(sub.year_mean)) if sub.year_mean.notna().any() else np.nan,
                             y_mean=float(sub.y.mean()), sqrt_tdd_mean=float(sub.s.mean()), E=Eg, log_E_E0=float(np.log(Eg / E0[t])),
                             source_id=",".join(sorted(sub.source_id.unique())), spatial_support_m=float(sub.spatial_support_m.iloc[0]),
                             fidelity_level=int(sub.fidelity_level.iloc[0])))
    BT = pd.concat(bts, ignore_index=True)
    keep = ["target", "block", "n", "lat", "lon", "E_blk", "E0", "log_ratio_E0", "z_mean", "z_var", "y_mean", "instr", "season", "team", "frac_early", "frac_gpr"] + FEATS
    BT[keep].to_csv(OUT / "fa_block_E.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "fa_blockE_summary.csv", index=False)
    pd.DataFrame(cors).to_csv(OUT / "fa_blockE_covcorr.csv", index=False)
    pd.DataFrame(abrows).to_csv(OUT / "fa_split_AB.csv", index=False)
    pd.DataFrame(camp).to_csv(OUT / "fa_split_AB_campaign.csv", index=False)
    pd.DataFrame(prov).sort_values(["target", "n_cells"], ascending=[True, False]).to_csv(OUT / "fa_label_provenance.csv", index=False)
    print(f"[1] 블록 {len(BT)} · 출처 행 {len(prov)} · {time.time() - T0:.0f}s", flush=True)


# ================================================================ 5) 분할 민감도(해석적)
def part5():
    rows, summ = [], []
    for t in TARGETS:
        ti = target_idx(t); seen = set()
        for seed in range(1, args.n_rand_splits + 1):
            A_idx, B_idx = half_split_blocks(DF, ti, seed)
            evB = B_idx[eval_mask(DF.iloc[B_idx])]
            key = frozenset(DF.block.values[A_idx])
            dup = key in seen; seen.add(key)
            A, B = DF.iloc[A_idx], DF.iloc[evB]
            if B.block.nunique() < 2 or len(A) < 3:
                continue
            EA, EB = ls_E(A.y.values, A.s.values), ls_E(B.y.values, B.s.values)
            Ek = (len(A) * EA + KAPPA * E0[t]) / (len(A) + KAPPA)
            p0 = rmse(B.y.values, E0[t] * B.s.values)
            rows.append(dict(target=t, seed=seed, dup=dup, n_A=len(A), n_B=len(B), nb_A=int(A.block.nunique()), nb_B=int(B.block.nunique()),
                             E_A=EA, E_B=EB, log_EA_EB=float(np.log(EA / EB)), log_E0_EB=float(np.log(E0[t] / EB)), rmse_B_E0=p0,
                             d_refit=rmse(B.y.values, EA * B.s.values) - p0, d_kappa=rmse(B.y.values, Ek * B.s.values) - p0,
                             d_oracle=rmse(B.y.values, EB * B.s.values) - p0))
    R = pd.DataFrame(rows); R.to_csv(OUT / "fa_split_sens_runs.csv", index=False)
    ab = pd.read_csv(OUT / "fa_split_AB.csv")
    for t, g in R.groupby("target"):
        u = g[~g.dup]
        a = ab[(ab.target == t) & ab.valid]
        d3 = float((a.rmse_B_Ekappa - a.rmse_B_E0).mean())
        lg = g[g.seed <= 5]; lg = lg[~lg.dup]
        summ.append(dict(target=t, n_splits=len(g), n_unique=len(u), d_mean=float(u.d_kappa.mean()), d_median=float(u.d_kappa.median()),
                         d_q05=float(u.d_kappa.quantile(.05)), d_q95=float(u.d_kappa.quantile(.95)), frac_worse=float((u.d_kappa > 0).mean()),
                         sd_log_EA_EB=float(u.log_EA_EB.std()), mean_abs_log_EA_EB=float(u.log_EA_EB.abs().mean()),
                         d_h25b_3splits=d3, pct_rank_h25b=float((u.d_kappa < d3).mean()),
                         d_lg_seeds1to5=float(lg.d_kappa.mean()) if len(lg) else np.nan, n_lg=len(lg), d_oracle_mean=float(u.d_oracle.mean())))
    pd.DataFrame(summ).to_csv(OUT / "fa_split_sens.csv", index=False)
    print(f"[5] 분할 민감도 {len(R)}행 · {time.time() - T0:.0f}s", flush=True)


# ================================================================ 2) 이질성 지표 대 E 수축 악화
def part2():
    summ = pd.read_csv(OUT / "fa_blockE_summary.csv"); ab = pd.read_csv(OUT / "fa_split_AB.csv")
    bt = pd.read_csv(OUT / "fa_block_E.csv")
    hhi = bt.groupby("target").n.apply(lambda v: float(np.sum((v / v.sum()) ** 2))).rename("hhi_blocks").reset_index()
    summ = summ.merge(hhi, on="target", how="left")
    summ["n_blocks_eff"] = 1.0 / summ.hhi_blocks
    summ["het_se"] = summ.sd_logE_blk_w * np.sqrt(2.0 * summ.hhi_blocks)             # 절반 분할 사이 log E 차이의 근사 표준오차(블록 이질성 기인)
    summ["het_minus_level"] = summ.het_se - np.abs(summ.log_Eown_E0)
    if (OUT / "fa_split_sens.csv").exists():
        ss = pd.read_csv(OUT / "fa_split_sens.csv")[["target", "d_mean", "frac_worse", "sd_log_EA_EB"]].rename(
            columns={"d_mean": "d_rand200_mean", "frac_worse": "rand200_frac_worse", "sd_log_EA_EB": "rand200_sd_log_EA_EB"})
        summ = summ.merge(ss, on="target", how="left")
    abv = ab[ab.valid]
    g = abv.groupby("target").agg(abs_log_EA_EB=("log_EA_EB", lambda v: float(np.mean(np.abs(v)))), log_EA_EB=("log_EA_EB", "mean"),
                                  abs_log_E0_EB=("log_E0_EB", lambda v: float(np.mean(np.abs(v)))), n_valid=("split", "size"),
                                  nb_B_min=("nb_B", "min"),
                                  d_refit_allA=("rmse_B_EA", "mean"), phys=("rmse_B_E0", "mean")).reset_index()
    abv = abv.assign(excess=np.abs(abv.log_EA_EB) - np.abs(abv.log_E0_EB))
    g = g.merge(abv.groupby("target").excess.mean().rename("excess_mismatch").reset_index(), on="target")
    g["d_refit_allA"] = g.d_refit_allA - g.phys
    s1 = CURVE[(CURVE.stage == "S1") & (CURVE.rule == "random") & (CURVE.scope == "n") & CURVE.n.isin([10, 40, 160])]
    pv = s1.pivot(index="target", columns="n", values="d_phys_blk").rename(columns=lambda n: f"d_S1_n{n}").reset_index()
    lo = s1.pivot(index="target", columns="n", values="d_phys_lo").rename(columns=lambda n: f"lo_S1_n{n}").reset_index()
    hi = s1.pivot(index="target", columns="n", values="d_phys_hi").rename(columns=lambda n: f"hi_S1_n{n}").reset_index()
    al = CURVE[(CURVE.stage == "S1") & (CURVE.rule == "all")][["target", "d_phys_blk", "phys_blk"]].rename(columns={"d_phys_blk": "d_S1_all"})
    h = summ.merge(g, on="target", how="left").merge(pv, on="target", how="left").merge(lo, on="target", how="left").merge(hi, on="target", how="left").merge(al, on="target", how="left")
    h["abs_log_Eown_E0"] = np.abs(h.log_Eown_E0)
    h.to_csv(OUT / "fa_hetero_index.csv", index=False)
    idx_cols = ["sd_logE_blk", "sd_logE_blk_w", "tau_logE", "n_blocks", "n_blocks5", "n_blocks_eff", "top_block_share", "cv_blk_sqrt_tdd", "het_se",
                "het_minus_level", "abs_log_EA_EB", "abs_log_Eown_E0", "excess_mismatch", "rand200_sd_log_EA_EB", "frac_early", "frac_gpr"]
    rows = []
    for out_col in ["d_S1_n40", "d_S1_n160", "d_S1_all", "d_rand200_mean"]:
        for c in idx_cols:
            for tag, sub in (("전체", h), ("채점 블록 ≥ 5", h[h.nb_B_min >= 5])):
                rho, p, n = sp_rho(sub[c].values, sub[out_col].values)
                rows.append(dict(outcome=out_col, index=c, subset=tag, rho=rho, p=p, n_targets=n))
    pd.DataFrame(rows).to_csv(OUT / "fa_hetero_corr.csv", index=False)
    print(f"[2] 이질성 지표 {len(h)}대상 · {time.time() - T0:.0f}s", flush=True)


# ================================================================ 3) 공변량 의존 계수 모형(V1) 사전 점검
def prep_X(Xtr, Xte, w=None):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    w = np.ones(len(Xtr)) if w is None else w
    mu = np.average(Xtr, 0, weights=w); sd = np.sqrt(np.average((Xtr - mu) ** 2, 0, weights=w))
    const = sd < 1e-8                                              # 학습 집합에서 상수인 열은 0 으로 둔다(채점 셀에서 값이 달라도 외삽하지 않음)
    sd = np.where(const, 1.0, sd)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ztr[:, const] = 0.0; Zte[:, const] = 0.0
    return Ztr, Zte


def ridge_logE(Xtr, ztr, Xte, w=None, alpha=10.0):
    from sklearn.linear_model import Ridge
    Ztr, Zte = prep_X(Xtr, Xte, w)
    m = Ridge(alpha=alpha).fit(Ztr, ztr, sample_weight=w)
    res = ztr - m.predict(Ztr)
    ww = np.ones(len(ztr)) if w is None else w
    smear = float(np.average(np.exp(res), weights=ww))
    return m.predict(Zte), smear, m


def cb_fit(Xtr, ytr, seed, w=None):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                          allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr, sample_weight=w)
    return m


def part3():
    V1_T = ["Canada", "CA-3"]; NG = [40, 160]; REPS = 5
    rows, stores, coefs, n_cb = [], {}, [], 0
    runs_ref = pd.read_csv(H3 / "h25b_runs.csv")
    for t in V1_T:
        src = DF.iloc[SRC[t]]
        Xs = src[FEATS].values.astype(float); ys, ss = src.y.values, src.s.values; zs = np.log(ys / ss)
        stores[t] = {}
        for sp, A_idx, evB in splits_of(t):
            A, B = DF.iloc[A_idx], DF.iloc[evB]
            XA, yA, sA = A[FEATS].values.astype(float), A.y.values, A.s.values; zA = np.log(yA / sA)
            XB, yB, sB = B[FEATS].values.astype(float), B.y.values, B.s.values
            st = BlockStore(t, sp, B.block.values); stores[t][sp] = st
            nA = len(A)

            def add(method, n, rep, pred, **kw):
                st.add((method, int(n), int(rep)), yB, pred)
                rows.append(dict(target=t, split=sp, method=method, n=int(n), rep=int(rep), n_A=nA, n_B=len(B), rmse_cm=rmse(yB, pred),
                                 bias_cm=float(np.mean(pred - yB)), **kw))

            add("P0", 0, -1, E0[t] * sB, E_used=E0[t])
            add("oracle_EB", 0, -1, ls_E(yB, sB) * sB, E_used=ls_E(yB, sB))
            zh, sm, m0 = ridge_logE(Xs, zs, XB)
            add("V1", 0, -1, np.exp(zh) * sB); add("V1_smear", 0, -1, sm * np.exp(zh) * sB)
            lo, hi = np.percentile(zs, [2, 98]); add("V1_clip", 0, -1, np.exp(np.clip(zh, lo, hi)) * sB)
            cb0 = cb_fit(Xs.astype(np.float32), zs, 0); n_cb += 1
            add("V1_cb", 0, -1, np.exp(cb0.predict(XB.astype(np.float32))) * sB)
            draws = [(-1, -1, np.arange(nA))]
            for n in NG:
                if n < nA:
                    for rep in range(REPS):                       # h25b random 규칙과 같은 추출 seed(rule 색인 0)
                        rng = np.random.RandomState(7_919 * sp + 1_009 * n + rep)
                        draws.append((n, rep, rng.choice(nA, n, replace=False)))
            for n, rep, sel in draws:
                m_ = len(sel); E_ls = ls_E(yA[sel], sA[sel]); E_n = (m_ * E_ls + KAPPA * E0[t]) / (m_ + KAPPA)
                add("P1", n, rep, E_n * sB, E_used=E_n); add("P2", n, rep, E_ls * sB, E_used=E_ls)
                Xtr = np.vstack([Xs, XA[sel]]); ztr = np.concatenate([zs, zA[sel]])
                for a_t, name in ((1.0, "V1"), (10.0, "V1_a10"), (100.0, "V1_a100")):
                    w = np.concatenate([np.ones(len(zs)), np.full(m_, a_t)])
                    zh, sm, mm = ridge_logE(Xtr, ztr, XB, w=w)
                    add(name, n, rep, np.exp(zh) * sB)
                    if name == "V1":
                        add("V1_smear", n, rep, sm * np.exp(zh) * sB)
                        lo, hi = np.percentile(ztr, [2, 98]); add("V1_clip", n, rep, np.exp(np.clip(zh, lo, hi)) * sB)
                        if n == -1:
                            for f, c in zip(FEATS, mm.coef_):
                                coefs.append(dict(target=t, split=sp, covariate=f, coef_std=float(c)))
                for ra, name in ((1.0, "V1_r1"), (100.0, "V1_r100")):
                    zh, _, _ = ridge_logE(Xtr, ztr, XB, alpha=ra); add(name, n, rep, np.exp(zh) * sB)
                zh, sm, _ = ridge_logE(XA[sel], zA[sel], XB)
                add("V1_tgt", n, rep, np.exp(zh) * sB)
                lo, hi = np.percentile(zA[sel], [2, 98]); add("V1_tgt_clip", n, rep, np.exp(np.clip(zh, lo, hi)) * sB)
                if n == -1 or (n == 160 and rep == 0):            # CatBoost 계수 모형은 전량과 n = 160 첫 추출만(적합 수 제한)
                    cb = cb_fit(Xtr.astype(np.float32), ztr, 0); n_cb += 1
                    add("V1_cb", n, rep, np.exp(cb.predict(XB.astype(np.float32))) * sB)
            # h25b S1 random 재현 대조(앞 5회)
            ref = runs_ref[(runs_ref.target == t) & (runs_ref.split == sp) & (runs_ref.stage == "S1") & (runs_ref.rule == "random") & (runs_ref.rep < REPS)]
            mine = pd.DataFrame([r for r in rows if r["target"] == t and r["split"] == sp and r["method"] == "P1" and r["n"] > 0])
            if len(mine):
                mm_ = mine.merge(ref[["n", "rep", "rmse_cm"]], on=["n", "rep"], suffixes=("", "_ref"))
                META.setdefault("v1_p1_repro_maxdiff", {})[f"{t}|{sp}"] = float(np.abs(mm_.rmse_cm - mm_.rmse_cm_ref).max())
    R = pd.DataFrame(rows); R.to_csv(OUT / "fa_v1_runs.csv", index=False)
    pd.DataFrame(coefs).to_csv(OUT / "fa_v1_coef.csv", index=False)
    out = []
    for t in V1_T:
        for (m, n), _ in R[R.target == t].groupby(["method", "n"]):
            if m == "P0":
                continue
            for base in ("P0", "P1"):
                if base == "P1" and (n == 0 or m == "P1"):
                    continue
                kB = (lambda k: k[0] == "P0") if base == "P0" else (lambda k, n=n: k[0] == "P1" and k[1] == n)
                b = boot_delta_blocks(stores[t], lambda k, m=m, n=n: k[0] == m and k[1] == n, kB, nboot=args.nboot, seed=0, rep_fn=lambda k: k[2])
                out.append(dict(target=t, method=m, n=n, baseline=base, rmse=b["rmse_A"], rmse_base=b["rmse_B"], delta=b["delta"], ci_lo=b["ci_lo"], ci_hi=b["ci_hi"],
                                delta_beq=b["delta_beq"], ci_lo_beq=b["ci_lo_beq"], ci_hi_beq=b["ci_hi_beq"], split_win=b["split_win"], rep_win=b["rep_win"],
                                n_splits=b["n_splits"], n_blocks_min=min(b["n_blocks"]), delta_split=json.dumps({str(k): round(v, 3) for k, v in b["delta_split"].items()})))
    pd.DataFrame(out).to_csv(OUT / "fa_v1_summary.csv", index=False)
    META["v1_n_catboost_fits"] = n_cb
    print(f"[3] V1 행 {len(R)} · CatBoost 적합 {n_cb} · {time.time() - T0:.0f}s", flush=True)


# ================================================================ 4) 원천 잔차 편향과 공변량 위치
def di_vs_source(t):
    """Meyer & Pebesma(2021) DI(가중 없음). 원천 통계로 표준화, 정규화 상수 = 원천 6-fold 공간 블록 CV 최근접 거리 평균,
    임계 = 원천 CV DI 의 Q3 + 1.5·IQR. a2_shift_diagnostics 와 같은 정의이고 학습 집합만 대상별 LORO 원천으로 바꿨다."""
    from sklearn.neighbors import NearestNeighbors
    src = SRC[t]; ti = target_idx(t)
    Xs = DF.iloc[src][FEATS].values.astype(float); Xt = DF.iloc[ti][FEATS].values.astype(float)
    Zs, Zt = prep_X(Xs, Xt)
    pos = {g: i for i, g in enumerate(src)}; d_cv = np.full(len(src), np.nan)
    for tr, te in spatial_block_splits(DF, n_splits=6, sub_idx=src):
        nn = NearestNeighbors(n_neighbors=1).fit(Zs[[pos[i] for i in tr]])
        d, _ = nn.kneighbors(Zs[[pos[i] for i in te]]); d_cv[[pos[i] for i in te]] = d[:, 0]
    dbar = float(np.nanmean(d_cv)); di_tr = d_cv / dbar
    q75, q25 = np.nanpercentile(di_tr, [75, 25]); thr = float(q75 + 1.5 * (q75 - q25))
    d, _ = NearestNeighbors(n_neighbors=1).fit(Zs).kneighbors(Zt)
    return d[:, 0] / dbar, thr


def part4():
    LAM = 0.25; SEEDS = [0, 1]
    rows, strata, shap_rows, cells, nn_rows, n_cb = [], [], [], [], [], 0
    for t in TARGETS:
        src = DF.iloc[SRC[t]]; ti = target_idx(t); d = DF.iloc[ti]
        Xs = src[FEATS].values.astype(np.float32); rs = src.y.values - E0[t] * src.s.values
        Xt = d[FEATS].values.astype(np.float32)
        g_seed, sh = [], []
        for seed in SEEDS:
            m = cb_fit(Xs, rs, seed); n_cb += 1
            g_seed.append(np.asarray(m.predict(Xt), float))
            from catboost import Pool
            sv = m.get_feature_importance(Pool(Xt), type="ShapValues", thread_count=args.threads); sh.append(sv[:, :-1].mean(0)); base_val = float(sv[0, -1])
        g = np.mean(g_seed, 0); sh = np.mean(sh, 0)
        di, thr = di_vs_source(t)
        ev = eval_mask(d)
        y, s = d.y.values, d.s.values; r = y - E0[t] * s
        out_aoa = di > thr
        for j in np.argsort(-np.abs(sh))[:5]:
            xt, xs = Xt[:, j].astype(float), Xs[:, j].astype(float)
            lo, hi = np.nanpercentile(xt, [5, 95]); inr = (xs >= lo) & (xs <= hi)     # 대상 5–95 % 범위 안에 든 원천 셀(지지)
            shap_rows.append(dict(target=t, covariate=FEATS[j], mean_shap_cm=float(sh[j]), base_value_cm=base_val,
                                  smd_vs_source=float((np.nanmean(xt) - np.nanmean(xs)) / (np.nanstd(xs) + 1e-9)),
                                  tgt_q05=float(lo), tgt_q95=float(hi), src_min=float(np.nanmin(xs)), src_max=float(np.nanmax(xs)),
                                  n_src_in_tgt_range=int(inr.sum()), frac_src_in_tgt_range=float(inr.mean()),
                                  src_resid_in_range_cm=float(rs[inr].mean()) if inr.any() else np.nan,
                                  frac_tgt_out_of_src_range=float(np.mean((xt < np.nanmin(xs)) | (xt > np.nanmax(xs))))))
        # 분할 B 기준(유효 분할) Δ: h25b S3 n=0 과 같은 채점 셀
        dsp = []
        loc = {gi: k for k, gi in enumerate(ti)}
        for sp, A_idx, evB in splits_of(t):
            k = np.array([loc[i] for i in evB])
            for gs in g_seed:
                dsp.append((sp, rmse(y[k], E0[t] * s[k] + LAM * gs[k]) - rmse(y[k], E0[t] * s[k])))
        dsp = pd.DataFrame(dsp, columns=["split", "d"]).groupby("split").d.mean()
        ref = CURVE[(CURVE.target == t) & (CURVE.stage == "S3") & (CURVE.n == 0) & (CURVE.lam == LAM)]
        re_, ge = r[ev], g[ev]
        bl = pd.DataFrame(dict(b=d.block.values[ev], r=re_, g=ge)).groupby("b").mean()
        rho_c = float(np.corrcoef(re_, ge)[0, 1]); rho_b = float(np.corrcoef(bl.r, bl.g)[0, 1]) if len(bl) >= 3 else np.nan
        rows.append(dict(target=t, parent=parent_of(t), n_eval=int(ev.sum()), n_blocks=int(d.block.nunique()), E0=E0[t], frac_src_parent=float((src.macro == parent_of(t)).mean()),
                         phys_bias_cm=float(np.mean(-re_)), resid_true_mean_cm=float(re_.mean()), resid_pred_mean_cm=float(ge.mean()),
                         bias_after_lam25_cm=float(np.mean(LAM * ge - re_)), resid_pred_sd_cm=float(ge.std()), resid_true_sd_cm=float(re_.std()),
                         corr_cell=rho_c, corr_block=rho_b, lam_opt=float((re_ @ ge) / (ge @ ge)),
                         # MSE 변화(cm²) = 수준 항 + 양상 항. 수준 = λ²·ḡ² − 2λ·ḡ·r̄, 양상 = λ²·Var(g) − 2λ·Cov(g, r)
                         dmse_level=float(LAM ** 2 * ge.mean() ** 2 - 2 * LAM * ge.mean() * re_.mean()),
                         dmse_pattern=float(LAM ** 2 * ge.var() - 2 * LAM * np.mean((ge - ge.mean()) * (re_ - re_.mean()))),
                         sign_agree=bool(np.sign(re_.mean()) == np.sign(ge.mean())),
                         rmse_phys_all=rmse(y[ev], E0[t] * s[ev]), d_lam25_all=rmse(y[ev], E0[t] * s[ev] + LAM * ge) - rmse(y[ev], E0[t] * s[ev]),
                         d_lam25_splitB=float(dsp.mean()), d_h25b_S3_n0=float(ref.d_phys_blk.iloc[0]) if len(ref) else np.nan,
                         lo_h25b=float(ref.d_phys_lo.iloc[0]) if len(ref) else np.nan, hi_h25b=float(ref.d_phys_hi.iloc[0]) if len(ref) else np.nan,
                         di_median=float(np.median(di[ev])), di_q3=float(np.percentile(di[ev], 75)), di_thr=thr, out_aoa=float(out_aoa[ev].mean()),
                         frac_gpr=float((d.instr.values[ev] == "GPR").mean()), frac_early=float((d.season.values[ev] == "6–7월").mean())))
        for name, msk in (("AOA 안", ev & ~out_aoa), ("AOA 밖", ev & out_aoa)):
            if msk.sum() >= 5:
                strata.append(dict(target=t, stratum=name, n=int(msk.sum()), resid_true_mean_cm=float(r[msk].mean()), resid_pred_mean_cm=float(g[msk].mean()),
                                   rmse_phys=rmse(y[msk], E0[t] * s[msk]), d_lam25=rmse(y[msk], E0[t] * s[msk] + LAM * g[msk]) - rmse(y[msk], E0[t] * s[msk]),
                                   lam_opt=float((r[msk] @ g[msk]) / (g[msk] @ g[msk]))))
        for ins, sub in pd.DataFrame(dict(ins=d.instr.values[ev], se=d.season.values[ev], r=re_, g=ge, y=y[ev], s=s[ev])).groupby(["ins", "se"]):
            if len(sub) >= 5:
                strata.append(dict(target=t, stratum=f"{ins[0]} · {ins[1]}", n=len(sub), resid_true_mean_cm=float(sub.r.mean()), resid_pred_mean_cm=float(sub.g.mean()),
                                   rmse_phys=rmse(sub.y.values, E0[t] * sub.s.values),
                                   d_lam25=rmse(sub.y.values, E0[t] * sub.s.values + LAM * sub.g.values) - rmse(sub.y.values, E0[t] * sub.s.values),
                                   lam_opt=float((sub.r.values @ sub.g.values) / (sub.g.values @ sub.g.values))))
        # 최근접 원천 셀의 소속(하위 지역 또는 macro)과 그 셀의 원천 잔차: 잔차가 어느 원천 집단에서 넘어오는지
        from sklearn.neighbors import NearestNeighbors
        Zs, Zt = prep_X(src[FEATS].values.astype(float), d[FEATS].values.astype(float))
        dist, nn_i = NearestNeighbors(n_neighbors=1).fit(Zs).kneighbors(Zt[ev])
        grp = np.where(src["sub"].values != "", src["sub"].values, src.macro.values)[nn_i[:, 0]]
        q = pd.DataFrame(dict(g=grp, team=src.team.values[nn_i[:, 0]], r_src=rs[nn_i[:, 0]], dist=dist[:, 0], r_true=re_, g_pred=ge))
        for gname, sub in q.groupby("g"):
            nn_rows.append(dict(target=t, nn_group=gname, share=len(sub) / len(q), n=len(sub), team_mode=sub.team.mode().iloc[0], src_resid_mean_cm=float(sub.r_src.mean()),
                                resid_pred_mean_cm=float(sub.g_pred.mean()), resid_true_mean_cm=float(sub.r_true.mean()), nn_dist_mean=float(sub.dist.mean())))
        cells.append(pd.DataFrame(dict(loc_id=d.loc_id.values, target=t, block=d.block.values, eval=ev, di=di, out_aoa=out_aoa, resid_true=r, resid_pred=g)))
    R = pd.DataFrame(rows); R.to_csv(OUT / "fa_src_resid.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "fa_src_resid_strata.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(OUT / "fa_src_resid_shap.csv", index=False)
    pd.DataFrame(nn_rows).sort_values(["target", "share"], ascending=[True, False]).to_csv(OUT / "fa_src_resid_nn.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(OUT / "fa_src_resid_cells.csv", index=False)
    cr = []
    for oc in ["d_h25b_S3_n0", "d_lam25_all"]:
        for c in ["out_aoa", "di_median", "lam_opt", "corr_cell", "corr_block", "resid_pred_mean_cm", "phys_bias_cm", "frac_src_parent", "frac_gpr"]:
            rho, p, n = sp_rho(R[c].values, R[oc].values); cr.append(dict(outcome=oc, index=c, rho=rho, p=p, n_targets=n))
        R["bias_gap"] = np.abs(R.resid_pred_mean_cm - R.resid_true_mean_cm) - np.abs(R.resid_true_mean_cm)
        rho, p, n = sp_rho(R["bias_gap"].values, R[oc].values); cr.append(dict(outcome=oc, index="|g평균 − r평균| − |r평균|", rho=rho, p=p, n_targets=n))
    pd.DataFrame(cr).to_csv(OUT / "fa_src_resid_corr.csv", index=False)
    # 기존 알래스카 기준 AOA(m1/a2_shift_cells.csv)와 대조: 레나·캐나다
    sc = pd.read_csv(PROC / "m1" / "a2_shift_cells.csv")
    mine = pd.concat(cells, ignore_index=True)
    cmp_ = mine[mine.target.isin(["Lena", "Canada"])].merge(sc[["loc_id", "di_x25", "in_aoa_x25"]], on="loc_id", how="inner")
    META["aoa_crosscheck"] = {t: dict(n=int(len(g_)), out_aoa_loro=float(g_.out_aoa.mean()), out_aoa_alaska_ref=float(1 - g_.in_aoa_x25.mean()),
                                      rho_di=float(spearmanr(g_.di, g_.di_x25).correlation)) for t, g_ in cmp_.groupby("target")}
    META["resid_n_catboost_fits"] = n_cb
    print(f"[4] 원천 잔차 {len(R)}대상 · CatBoost 적합 {n_cb} · {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    for k, fn in ((0, part0), (1, part1), (5, part5), (2, part2), (3, part3), (4, part4)):
        if k in PARTS:
            fn()
    META["elapsed_s"] = round(time.time() - T0, 1)
    mp = OUT / "fa_meta.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    old.update(META)
    mp.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=float))
    print(f"saved fa_* · {time.time() - T0:.0f}s")
