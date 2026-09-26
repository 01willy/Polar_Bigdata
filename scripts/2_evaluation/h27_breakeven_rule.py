"""H27 필요 라벨 수 규칙: 대상별 손익분기 n(h25 계열 결과)과 E비 |log(E/E0)| 의 관계를 절단 반영 통계로 재검정한다. CPU.

계획 docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md §3 H27, 감사 반영 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §3.
입력 data/processed/h3/<src>_{curve,breakeven,targets,runs}.csv (--src-tag, 기본 h25; h25b 재실행 결과도 같은 형식이면 사용 가능).

손익분기 정의(curve 에서 재계산, n 은 모든 분할에 있는 격자 값만 사용)
  rec50 : 회복률 ≥ 0.5 이고 반복 승률 ≥ 0.75 인 최소 n. 회복률 = (물리식 − 방법)/(물리식 − 라벨 전량), 분모 ≤ 0 이면 NaN(h4_common.recovery).
          h25 원 집계는 |분모| > 1e-6 이면 음의 분모도 허용했으므로 이 점이 다를 수 있다.
  ci    : (분할, 반복) 행 부트스트랩 CI 상한 < 0 인 최소 n(보조 정의). 행 부트스트랩 열은 h25 형식이면 d_phys_hi,
          h25b 형식(d_phys_hi_rep 열이 있음)이면 d_phys_hi_rep 이다.
  main  : 사전 고정 주 정의. 블록 부트스트랩 CI 상한 < 0 · 분할 승률 ≥ 2/3 · 반복 승률 ≥ 0.75 인 최소 n(h4_common.min_n).
          curve 에 블록 열(d_phys_hi_rep 이 있는 h25b 형식의 d_phys_hi, 또는 ci_hi)과 split_win·rep_win 이 모두 있을 때만 계산한다.
  옛 집계 병기(be_n_old): h25 형식 breakeven 은 rec50←breakeven_n_rec50, ci←breakeven_n_ci.
          h25b 형식(breakeven_n_ci_rep 열이 있음)은 rec50←breakeven_n_rec50, ci←breakeven_n_ci_rep, main←breakeven_n_ci.
규칙 kmedoid·random, 단계 S1(λ=0)·S3(λ=.25), α=1(--stages 로 변경).
E비 종류(모두 kmedoid n=3, S1refit 단계의 E_used(3-라벨 재적합 E)/E0 에서 만든다. oracle 만 예외)
  oracle      : E_own(대상 전 셀 최소제곱) / E0(원천 최소제곱).
  est3_pooled : 3-라벨 추정의 (분할, 반복) 60회 중앙값. 서로 다른 A 집합(최대 약 180 셀, 다른 분할의 채점 셀 포함)의
                정보를 모은 값이므로, 한 번의 배포에서 얻는 3-라벨 추정보다 잡음이 작다. 상한 참고값으로만 쓴다.
  est3_split  : 분할별 반복 중앙값(대상당 3개). 분할 번호 s 를 대상 사이에 짝지어 통계를 s 마다 계산하고,
                분할 사이 중앙값(최소·최대 병기)을 보고한다.
  est3_draw   : 배포 가능한 변형. (분할, 반복) 한 번의 3-라벨 추정값을 그대로 쓴다. 추출 번호 (s, r) 를 대상 사이에 짝지어
                추출마다 통계를 계산하고, 추출 분포의 중앙값·2.5–97.5 % 분위와 p < 0.05 비율을 보고한다.
통계(절단 대상 = 격자 안 미달성, 최대 순위 동률; h4_common)
  Spearman ρ·Kendall τ-b(|log E비| 대 손익분기 n), 달성 여부 ~ |log E비| 의 Mann-Whitney(동률 없고 n ≤ 40 이면 정확)와 로지스틱(Wald),
  미절단 대상 수. 참고로 옛 방식(미달성 = 2·n_max 대입) Spearman 도 병기한다.
산출 data/processed/h3/<out>_rule.csv(규칙 × 단계 × 정의 × E비 종류), <out>_targets.csv(대상 × 규칙 × 단계 × 정의).
  <out> 기본값: src=h25 이면 h27b, 아니면 h27b_<src>.
--legacy: 옛 ridge LOO 분석(h27_rule.csv, h27_loo_S*.csv)을 다시 만든다.
실행: python3 scripts/2_evaluation/h27_breakeven_rule.py [--src-tag h25b] [--smoke]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message=".*constant.*")                  # 소표본 추출에서 x 가 상수인 경우(ρ = NaN 으로 처리)
from polar.h4_common import (recovery, min_n, spearman_censored, kendall_censored, achieved_test,       # noqa: E402
                             REP_WIN_MIN)

ap = argparse.ArgumentParser()
ap.add_argument("--src-tag", default="h25")
ap.add_argument("--src-dir", default=str(ROOT / "data" / "processed" / "h3"))
ap.add_argument("--out-tag", default="")
ap.add_argument("--stages", default="S1:0,S3:0.25", help="단계:λ 쉼표 목록")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--rules", default="kmedoid,random")
ap.add_argument("--legacy", action="store_true")
ap.add_argument("--smoke", action="store_true", help="kmedoid·S3 만, 산출 파일명에 _smoke")
args = ap.parse_args()

SRC = Path(args.src_dir); OUT = ROOT / "data" / "processed" / "h3"; OUT.mkdir(exist_ok=True)
TAG = args.out_tag or ("h27b" if args.src_tag == "h25" else f"h27b_{args.src_tag}")
if args.smoke:
    TAG += "_smoke"
STAGES = [(s.split(":")[0], float(s.split(":")[1])) for s in args.stages.split(",")]
RULES = args.rules.split(",")
if args.smoke:
    STAGES = [st for st in STAGES if st[0] == "S3"] or STAGES[:1]; RULES = RULES[:1]


# ---------------------------------------------------------------- 대상 지표·E비
def target_table():
    tg = pd.read_csv(SRC / f"{args.src_tag}_targets.csv")
    T = tg.groupby("target").agg(parent=("parent", "first"), n_cells=("n_cells", "first"), n_blocks=("n_blocks", "first"),
                                 E_own=("E_own", "first"), E0=("E0", "mean")).reset_index()
    T["E_ratio_oracle"] = T.E_own / T.E0
    runs = pd.read_csv(SRC / f"{args.src_tag}_runs.csv", usecols=["target", "split", "rule", "stage", "scope", "n", "rep", "E_used", "E0"])
    q = runs[(runs.rule == "kmedoid") & (runs.stage == "S1refit") & (runs.scope == "n") & (runs.n == 3)]
    q = q.assign(r=q.E_used / q.E0)
    e3 = q.groupby("target").agg(E3_med=("E_used", "median"), E_ratio_est3_pooled=("r", "median"), n_est3=("r", "size"),
                                 sd_logE_est3_draw=("r", lambda v: float(np.std(np.log(v), ddof=1))))
    T = T.merge(e3.reset_index(), on="target", how="left")
    T["abs_logE_oracle"] = np.abs(np.log(T.E_ratio_oracle)); T["abs_logE_est3_pooled"] = np.abs(np.log(T.E_ratio_est3_pooled))
    # 배포 가능한 변형: 추출(분할, 반복)별 값과 분할별 반복 중앙값
    D = q.groupby(["target", "split", "rep"]).r.median().reset_index()                 # 반복당 seed 1개라 중앙값 = 그 값
    D["abs_logE"] = np.abs(np.log(D.r))
    Sp = q.groupby(["target", "split"]).r.median().reset_index(); Sp["abs_logE"] = np.abs(np.log(Sp.r))
    return T, dict(est3_draw=D.pivot(index="target", columns=["split", "rep"], values="abs_logE"),
                   est3_split=Sp.pivot(index="target", columns="split", values="abs_logE"))


# ---------------------------------------------------------------- 손익분기 재계산
def breakeven_table():
    c = pd.read_csv(SRC / f"{args.src_tag}_curve.csv")
    allA = c[c.scope == "allA"].groupby(["target", "stage", "lam", "alpha"]).rmse_mean.mean()
    # 열 이름 판정: h25b 형식이면 행 부트스트랩 = d_phys_hi_rep, 블록 부트스트랩 = d_phys_hi. h25 형식이면 d_phys_hi 가 행 부트스트랩.
    if "d_phys_hi_rep" in c.columns:
        rep_col, blk_col = "d_phys_hi_rep", "d_phys_hi"
    else:
        rep_col, blk_col = "d_phys_hi", ("ci_hi" if "ci_hi" in c.columns else None)
    has_block = blk_col is not None and all(k in c.columns for k in ("split_win", "rep_win"))
    print(f"[cols] 행 부트스트랩={rep_col} · 블록 부트스트랩={blk_col if has_block else '없음(main 생략)'}")
    rows = []
    for rule in RULES:
        for stage, lam in STAGES:
            cs = c[(c.scope == "n") & (c.rule == rule) & (c.stage == stage) & np.isclose(c.lam, lam) & np.isclose(c.alpha, args.alpha)]
            for t, sub in cs.groupby("target"):
                sub = sub[sub.n_splits == sub.n_splits.max()].sort_values("n").copy()
                key = (t, stage if stage != "S1refit" else "S1", lam, args.alpha)
                full = allA.loc[key] if key in allA.index else np.nan
                phys = sub.rmse_mean - sub.d_phys_mean
                sub["rec_new"] = [recovery(p, m, full) for p, m in zip(phys, sub.rmse_mean)]
                n_max = int(sub.n.max())
                defs = {"rec50": sub[(sub.rec_new >= 0.5) & (sub.win_rate >= REP_WIN_MIN)],
                        "ci": sub[sub[rep_col] < 0]}
                for d, ok in defs.items():
                    rows.append(dict(target=t, rule=rule, stage=stage, lam=lam, definition=d, be_n=float(ok.n.iloc[0]) if len(ok) else np.nan,
                                     censored=not len(ok), n_max=n_max, n_tested=int(len(sub)),
                                     n_rec_den_nonpos=int(np.sum(~np.isfinite(sub.rec_new))) if d == "rec50" else 0))
                if has_block:
                    r = min_n(sub, n_col="n", ci_hi_col=blk_col, split_win_col="split_win", rep_win_col="rep_win")
                    rows.append(dict(target=t, rule=rule, stage=stage, lam=lam, definition="main", be_n=r["n_star"], censored=bool(r["censored"]),
                                     n_max=r["n_max"], n_tested=r["n_tested"], n_rec_den_nonpos=0))
    B = pd.DataFrame(rows)
    # 옛 집계(h25 breakeven 파일) 병기
    fb = SRC / f"{args.src_tag}_breakeven.csv"
    if fb.exists():
        be = pd.read_csv(fb)
        be = be[np.isclose(be.alpha, args.alpha)]
        old = []
        cmap = ((("rec50", "breakeven_n_rec50"), ("ci", "breakeven_n_ci_rep"), ("main", "breakeven_n_ci"))
                if "breakeven_n_ci_rep" in be.columns else (("rec50", "breakeven_n_rec50"), ("ci", "breakeven_n_ci")))
        for _, r in be.iterrows():
            for d, col in cmap:
                old.append(dict(target=r.target, rule=r.rule, stage=r.stage, lam=float(r.lam), definition=d, be_n_old=float(r[col]) if r[col] > 0 else np.nan))
        B = B.merge(pd.DataFrame(old), on=["target", "rule", "stage", "lam", "definition"], how="left")
    return B


def rule_stats(M, ekind, x=None):
    x = (M[f"abs_logE_{ekind}"].values if x is None else np.asarray(x)).astype(float)
    y = M.be_n.values.astype(float); cen = M.censored.values.astype(bool)
    rho, p_rho, n_rho = spearman_censored(x, y, censored_y=cen)
    tau, p_tau, _ = kendall_censored(x, y, censored_y=cen)
    a = achieved_test(~cen, x)
    y_old = np.where(cen, 2.0 * M.n_max.values, y)                          # 옛 방식: 미달성 = 2·n_max
    ok = np.isfinite(x)
    rho_old, p_old = spearmanr(x[ok], y_old[ok]) if ok.sum() >= 3 else (np.nan, np.nan)
    return dict(n_targets=int(n_rho), n_uncensored=int((~cen[ok]).sum()), n_censored=int(cen[ok].sum()),
                spearman_rho=rho, spearman_p=p_rho, kendall_tau=tau, kendall_p=p_tau,
                mw_U=a["U"], p_mw=a["p_mw"], mw_method=a["mw_method"], median_absE_achieved=a["median_achieved"], median_absE_censored=a["median_not"],
                logit_coef=a["logit_coef"], logit_se=a["logit_se"], logit_p=a["logit_p"], separation=a["separation"],
                spearman_rho_oldstyle=float(rho_old), spearman_p_oldstyle=float(p_old))


def rule_stats_dist(M, W, ekind):
    """추출(또는 분할)별 E비 행렬 W(대상 × 추출)로 통계를 추출마다 계산하고 분포를 요약한다. 대표값은 중앙값이다."""
    W = W.reindex(M.target.values)
    per = [rule_stats(M, ekind, x=W[c].values) for c in W.columns]
    P = pd.DataFrame(per)
    out = rule_stats(M, ekind, x=W.median(axis=1).values)                  # 대상 수·절단 수 등 공통 열의 틀
    for k in ("spearman_rho", "spearman_p", "kendall_tau", "kendall_p", "p_mw", "logit_coef", "logit_p", "spearman_rho_oldstyle", "spearman_p_oldstyle",
              "mw_U", "median_absE_achieved", "median_absE_censored", "logit_se"):
        out[k] = float(np.nanmedian(P[k])) if np.isfinite(P[k].astype(float)).any() else np.nan
    out["separation"] = float(P.separation.astype(float).mean())            # 완전 분리 추출의 비율
    for k in ("spearman_rho", "kendall_tau"):
        if not np.isfinite(P[k].astype(float)).any():
            out.update({f"{k}_q025": np.nan, f"{k}_q975": np.nan, f"{k}_min": np.nan, f"{k}_max": np.nan}); continue
        out[f"{k}_q025"] = float(np.nanquantile(P[k], 0.025)); out[f"{k}_q975"] = float(np.nanquantile(P[k], 0.975))
        out[f"{k}_min"] = float(np.nanmin(P[k])); out[f"{k}_max"] = float(np.nanmax(P[k]))
    for k in ("spearman_p", "kendall_p", "p_mw"):
        pv = P[k].values.astype(float); pv = pv[np.isfinite(pv)]
        out[f"frac_sig_{k}"] = float(np.mean(pv < 0.05)) if len(pv) else np.nan       # 계산 가능한 추출 중 p < 0.05 비율
    out["n_draws"] = int(len(P))
    return out


def main():
    T, DW = target_table(); B = breakeven_table()
    M = B.merge(T, on="target", how="left")
    M.to_csv(OUT / f"{TAG}_targets.csv", index=False)
    rows = []
    for (rule, stage, lam, d), sub in M.groupby(["rule", "stage", "lam", "definition"], sort=False):
        base = dict(src=args.src_tag, rule=rule, stage=stage, lam=lam, alpha=args.alpha, definition=d)
        for ek in ("oracle", "est3_pooled"):
            rows.append(dict(base, E_kind=ek, n_draws=1, **rule_stats(sub, ek)))
        for ek in ("est3_split", "est3_draw"):
            rows.append(dict(base, E_kind=ek, **rule_stats_dist(sub, DW[ek], ek)))
    R = pd.DataFrame(rows); R.to_csv(OUT / f"{TAG}_rule.csv", index=False)
    pd.set_option("display.width", 250)
    print(R[["rule", "stage", "definition", "E_kind", "n_draws", "n_targets", "n_uncensored", "spearman_rho", "spearman_rho_q025", "spearman_rho_q975",
             "spearman_p", "frac_sig_spearman_p", "kendall_tau", "kendall_p", "p_mw", "frac_sig_p_mw", "logit_p", "separation",
             "spearman_rho_oldstyle"]].round(3).to_string(index=False))
    show = M[(M.rule == RULES[0]) & (M.stage == STAGES[-1][0])]
    cols = ["target", "definition", "be_n", "censored", "n_max"] + (["be_n_old"] if "be_n_old" in M else []) + ["n_rec_den_nonpos", "abs_logE_oracle",
                                                                                                             "abs_logE_est3_pooled", "sd_logE_est3_draw"]
    print(show[cols].round(3).to_string(index=False))
    print(f"[out] {OUT / f'{TAG}_rule.csv'} · {OUT / f'{TAG}_targets.csv'}")


# ---------------------------------------------------------------- 옛 ridge LOO 분석(--legacy)
def legacy():
    from sklearn.linear_model import Ridge
    be = pd.read_csv(SRC / f"{args.src_tag}_breakeven.csv"); tg = pd.read_csv(SRC / f"{args.src_tag}_targets.csv")
    T = tg.groupby("target").agg(parent=("parent", "first"), n_cells=("n_cells", "first"), n_blocks=("n_blocks", "first"), smd_x25=("smd_x25", "mean"),
                                 var_ratio_x25=("var_ratio_x25", "mean"), E_own=("E_own", "first"), E0=("E0", "mean"), E_block_cv=("E_block_cv", "first"),
                                 within_block_sd_ratio=("within_block_sd_ratio", "first"), y_sd=("y_sd", "first")).reset_index()
    T["E_ratio"] = T.E_own / T.E0; T["abs_logE"] = np.abs(np.log(T.E_ratio))
    rows = []
    for stage, lam in (("S1", 0.0), ("S3", 0.25)):
        b = be[(be.rule == "kmedoid") & (be.stage == stage) & (be.lam == lam) & (be.alpha == 1.0)][["target", "breakeven_n_rec50", "breakeven_n_ci", "n_max", "best_n", "best_d_phys"]]
        M = T.merge(b, on="target")
        M["be_n"] = np.where(M.breakeven_n_rec50 > 0, M.breakeven_n_rec50, 2 * M.n_max)
        M["censored"] = M.breakeven_n_rec50 <= 0
        y = np.log2(M.be_n.values.astype(float))
        for name, feats in (("labelfree", ["smd_x25", "var_ratio_x25", "n_blocks", "n_cells"]), ("labelfree+E", ["smd_x25", "var_ratio_x25", "n_blocks", "abs_logE"]), ("E_only", ["abs_logE"])):
            X = M[feats].values.astype(float); X = np.log(X + 1e-9) if name == "labelfree" else X
            pred = np.zeros(len(M))
            for i in range(len(M)):
                tr = np.arange(len(M)) != i
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = Ridge(alpha=1.0).fit((X[tr] - mu) / sd, y[tr]); pred[i] = m.predict(((X[i] - mu) / sd)[None, :])[0]
            r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
            rho, p = spearmanr(y, pred); rho_E, p_E = spearmanr(M.abs_logE, M.be_n)
            rows.append(dict(stage=stage, lam=lam, features=name, n_targets=len(M), n_censored=int(M.censored.sum()), loo_r2=float(r2), spearman_pred=float(rho), p_pred=float(p),
                             spearman_Eratio_vs_be=float(rho_E), p_Eratio=float(p_E)))
            M[f"pred_{name}"] = 2 ** pred
        M.to_csv(OUT / f"h27_loo_{stage}.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv(OUT / "h27_rule.csv", index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    legacy() if args.legacy else main()
