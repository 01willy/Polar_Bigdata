"""H4 통합 분석 — 논문 최종 계획(docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md) Track A1·C3·D4.

A1 레시피 재현 표: 라벨 있음 조건(M1 H13·H28 사전 지정) + 공변량만 조건 대조군 사다리(M1 H3) + 정보 없음 직접 ML(M1 X-model) 을 한 표로.
A1 이득 분해: 라벨 전량(allA) 에서 물리식(E0) → E 재적합(수준 성분) → E 앵커 + 잔차 ML(구조 성분). 자료 = h4/a2_curve.csv(있으면, E 처리 4수준) 아니면 h3/h25_curve.csv(allA 행).
C3 배포 산출물 표: H30 + C1(h4/c1_tests) + C2(h4/c2_coverage) 를 한 표로(있는 것만).
D4 실용 지표: 확정 비교마다 최악 지역 Δ, 악화 지역 비율, 절대 오차 중앙값(가능한 경우).
BLK 블록 CI 표준화(그림 스펙 §6 단계 2): <tag>_blocksse.npz 가 있는 h4 산출(a2·b2·b3·b4·b1·d1)의 요약 CSV 에서
    blk_d_phys, blk_d_phys_lo, blk_d_phys_hi, split_win, rep_win, ci_rep_lo, ci_rep_hi, ci_kind 표준 열을 만든다.
    요약에 블록 CI 가 이미 있으면 그대로 옮기고(blk_source='summary'), 없거나 NaN 인 행은 npz 에서
    h4_common.boot_delta_blocks 로 재집계한다(blk_source='npz'; 각 스크립트와 같은 키·seed·rep_fn 규약).
    요약 값이 있는 행 중 최대 --nverify 개는 npz 로 다시 계산해 차이(blk_verify_absdiff)를 기록한다.
    본 실행 파일이 없으면 스모크(<tag>_smoke_*)로 동작만 확인해 <tag>_smoke_blk.csv 로 저장한다.
산출 data/processed/h4/a1_recipe_table.csv, a1_ladder.csv, a1_decomp.csv, c3_deploy_table.csv, d4_practical.csv, <tag>_blk.csv
실행: python3 scripts/2_evaluation/h4_analysis.py [--blk-only] [--nverify 6]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
H2, H3, M1 = PROC / "h2", PROC / "h3", PROC / "m1"
AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]; MAIN6 = AB4[:2] + ["Russia_W", "Russia_C", "Russia_E", "Greenland"]
pd.set_option("display.width", 250)


# ---------------------------------------------------------------- BLK 블록 CI 표준화(그림용)
def _f(v):
    return float(v)


def _keys_a2(r):
    pk = [("physics", "E0_fixed", "none", "n0", 0, -1, -1, 0.0)]
    fa = lambda k: (k[0] == r.stage and k[1] == r.e_treat and k[2] == r.spread and k[3] == r.scope and k[4] == int(r.n)   # noqa: E731
                    and abs(k[7] - _f(r.lam)) < 1e-9)
    return fa, pk, ("a2boot", r.target), (lambda k: k[5])


def _keys_b2(r):
    pk = [("physics", "E", "none", "n0", 0, -1, -1, 0.0)]
    fa = lambda k: (k[0] == r.est and k[1] == r.variant and k[2] == r.rule and k[3] == r.scope and k[4] == int(r.n)       # noqa: E731
                    and abs(k[7] - _f(r.lam)) < 1e-9)
    return fa, pk, ("b2", r.target), (lambda k: k[5])


def _keys_b3(r):
    fa = lambda k: k[0] == r.rule and k[1] == r.stage and k[2] == int(r.n)                                                 # noqa: E731
    return fa, [("none", "physics", 0, -1, -1)], ("h34blk", r.target), (lambda k: k[3])


def _keys_b4(r):
    fa = lambda k: k[0] == r.method and k[1] == int(r.n) and abs(k[3] - _f(r.lam)) < 1e-9                                # noqa: E731
    return fa, (lambda k: k[0] == "physics"), ("b4blk", r.target), (lambda k: k[2])


def _keys_b1(r):
    m = str(r.method)
    if "@" in m:
        base, t = m.split("@")
        tau = _f(r.tau_loto) if t == "loto" else _f(t)
    else:
        base, tau = m, -1.0
    fa = lambda k: k[0] == base and abs(k[1] - tau) < 1e-9 and k[2] == int(r.n)                                            # noqa: E731
    return fa, (lambda k: k[0] == "physics"), 0, (lambda k: k[3])


def _keys_d1(r):
    fa = lambda k: k[0] == r.method and k[1] == int(r.n) and abs(k[4] - _f(r.tau)) < 1e-9                                 # noqa: E731
    return fa, (lambda k: k[0] == "physics"), 11, (lambda k: k[2])


# 태그 → (요약 kind, 식별 열, 원천 열(est, lo, hi, split_win, rep_win, rep_lo, rep_hi), 덧붙일 보조 블록 열, 키 함수)
BLK_SPEC = {
    "a2": ("curve", ["target", "parent", "e_treat", "spread", "scope", "n", "lam", "stage"],
           ("d_phys_mean", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "d_phys_lo_rep", "d_phys_hi_rep"),
           ["d_noresid_mean", "d_noresid_lo", "d_noresid_hi"], _keys_a2),
    "b2": ("curve", ["target", "parent", "est", "variant", "rule", "scope", "lam", "n"],
           ("d_phys_blk", "d_phys_lo", "d_phys_hi", "split_win_phys", "rep_win_phys", "ci_rep_phys_lo", "ci_rep_phys_hi"),
           ["d_shrink_blk", "d_shrink_lo", "d_shrink_hi", "split_win_shrink", "rep_win_shrink"], _keys_b2),
    "b3": ("summary", ["target", "parent", "layer", "rule", "stage", "n"],
           ("blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi", "blk_d_phys_split_win", "blk_d_phys_rep_win", "d_phys_ci_rep_lo", "d_phys_ci_rep_hi"),
           ["blk_d_rand", "blk_d_rand_lo", "blk_d_rand_hi", "blk_d_rand_split_win", "blk_d_rand_rep_win"], _keys_b3),
    "b4": ("summary", ["target", "parent", "method", "lam", "n"],
           ("d_phys_blk", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "ci_rep_lo", "ci_rep_hi"),
           ["d_cb_blk", "d_cb_lo", "d_cb_hi", "d_cbr_blk", "d_cbr_lo", "d_cbr_hi"], _keys_b4),
    "b1": ("summary", ["target", "parent", "method", "analysis", "role", "n", "tau_loto"],
           ("d_phys_block", "d_phys_lo", "d_phys_hi", "split_win", "rep_win", "d_phys_lo_rep", "d_phys_hi_rep"), [], _keys_b1),
    "d1": ("summary", ["target", "method", "n", "tau"],
           ("d_blk", "d_blk_lo", "d_blk_hi", "split_win", "rep_win", "d_phys_lo_rep", "d_phys_hi_rep"), ["rmse_phys"], _keys_d1),
}
STD = ["blk_d_phys", "blk_d_phys_lo", "blk_d_phys_hi", "split_win", "rep_win", "ci_rep_lo", "ci_rep_hi"]


def _blk_one(tag, nverify=6, nboot=1000):
    from polar.h4_common import load_stores, stores_for_target, boot_delta_blocks, seed_of
    kind, idc, src, extra, keyf = BLK_SPEC[tag]
    for smoke in ("", "_smoke"):
        ps_, pn = OUT / f"{tag}{smoke}_{kind}.csv", OUT / f"{tag}{smoke}_blocksse.npz"
        if ps_.exists():
            break
    else:
        return None
    S = pd.read_csv(ps_)
    out = S[[c for c in idc if c in S.columns]].copy()
    for std, c in zip(STD, src):
        out[std] = S[c] if c in S.columns else np.nan
    for c in extra:
        if c in S.columns:
            out[c] = S[c]
    has = out.blk_d_phys_lo.notna()
    out["blk_source"] = np.where(has, "summary", "")
    out["blk_verify_absdiff"] = np.nan
    stores = load_stores(pn) if pn.exists() else {}
    by_t = {}

    def recompute(r):
        t = str(r.target)
        if t not in by_t:
            by_t[t] = stores_for_target(stores, t)
        fa, fb, sd, rep = keyf(r)
        seed = seed_of(*sd) if isinstance(sd, tuple) else int(sd)
        return boot_delta_blocks(by_t[t], fa, fb, nboot=nboot, seed=seed, rep_fn=rep)

    n_re = n_ver = 0; maxdiff = 0.0
    if stores:
        need = list(np.where(~has.values)[0])
        ver = list(np.where(has.values)[0]); ver = ver[:: max(1, len(ver) // max(nverify, 1))][:nverify]
        for i in need + ver:
            r = S.iloc[i]
            try:
                b = recompute(r)
            except Exception as e:                                           # noqa: BLE001
                print(f"  [blk {tag}] 행 {i} 재집계 실패: {e}"); continue
            if b["n_splits"] == 0:
                continue
            vals = [b["delta"], b["ci_lo"], b["ci_hi"], b["split_win"], b["rep_win"]]
            if i in need:
                out.loc[out.index[i], STD[:5]] = vals; out.loc[out.index[i], "blk_source"] = "npz"; n_re += 1
            else:
                d = float(np.nanmax(np.abs(np.array(vals[:3]) - out.iloc[i][STD[:3]].to_numpy(float))))
                out.loc[out.index[i], "blk_verify_absdiff"] = d; maxdiff = max(maxdiff, d); n_ver += 1
    out["ci_kind"] = np.where(out.blk_d_phys_lo.notna(), "block", "none")
    out.loc[out.blk_source == "", "blk_source"] = "none"
    dst = OUT / f"{tag}{smoke}_blk.csv"
    out.to_csv(dst, index=False)
    print(f"[BLK {tag}] {ps_.name} → {dst.name}: 행 {len(out)}, 요약 블록 CI {int(has.sum())}, npz 재집계 {n_re}, "
          f"검증 {n_ver}행 최대 |차| {maxdiff:.2e} cm, npz {'있음' if stores else '없음'}")
    return dst


def blk_reaggregate(nverify=6):
    for tag in BLK_SPEC:
        try:
            _blk_one(tag, nverify)
        except Exception as e:                                               # noqa: BLE001
            print(f"[BLK {tag}] 실패: {type(e).__name__}: {e}")


if "--blk-only" in sys.argv:
    _nv = int(sys.argv[sys.argv.index("--nverify") + 1]) if "--nverify" in sys.argv else 6
    blk_reaggregate(_nv); sys.exit(0)


def rd(p):
    return pd.read_csv(p) if Path(p).exists() else None


# ---------------------------------------------------------------- A1 레시피 표 + 사다리(M1 확정 검정표 재사용, 새 부트스트랩 없음)
T = rd(M1 / "m1_sc_tests.csv")
rows = []
if T is not None:
    keep = T[T.H.isin(["H3", "H13", "H1", "H2"])]
    for _, r in keep.iterrows():
        rows.append(dict(source="m1_sc_tests", H=r.H, label=r.label, cond=r.cond, target=r.target, delta=r.delta_rmse, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                         p_holm=r.get("p_holm", np.nan), delta_blockeq=r.get("delta_blockeq", np.nan), n_cells=r.n_cells, n_blocks=r.n_blocks))
H28 = rd(H3 / "h28_tests.csv")
if H28 is not None:
    for _, r in H28.iterrows():
        rows.append(dict(source="h28_tests", H="H28", label=r.test, cond=r.cond, target=r.target, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                         p_holm=np.nan, delta_blockeq=r.delta_blockeq, n_cells=r.n_cells, n_blocks=r.n_blocks))
R = pd.DataFrame(rows); R.to_csv(OUT / "a1_recipe_table.csv", index=False)
# 사다리(공변량만, AB4 요약): H3 의 각 대조군 대 Stefan 유사라벨
if T is not None:
    lad = T[(T.H == "H3") & (T.target == "REGION_SUMMARY_AB4")][["label", "delta_rmse", "ci_lo", "ci_hi", "p_holm", "n_regions_ci"]]
    lad.to_csv(OUT / "a1_ladder.csv", index=False)
    print("[A1 사다리 AB4]"); print(lad.round(2).to_string(index=False))
    h13 = T[(T.H == "H13") & (T.target.isin(AB4 + ["REGION_SUMMARY_AB4"]))][["label", "target", "delta_rmse", "ci_lo", "ci_hi", "p_holm"]]
    print("[A1 H13 라벨 있음 잔차]"); print(h13.round(2).to_string(index=False))

# ---------------------------------------------------------------- A1 이득 분해(라벨 전량)
A2 = rd(OUT / "a2_curve.csv")
dec = []
if A2 is not None and "etreat" in A2.columns:
    # h31 산출: 열 etreat ∈ {E0_fixed, E_own_fixed, shrink_k10, offset_mle}, stage ∈ {anchor, resid}, scope allA
    q = A2[A2.scope == "allA"]
    for t, s in q.groupby("target"):
        phys = s[(s.etreat == "E0_fixed") & (s.stage == "anchor")]
        own = s[(s.etreat == "E_own_fixed") & (s.stage == "anchor")]
        r0 = s[(s.etreat == "E0_fixed") & (s.stage == "resid") & (s.lam == 0.25)]
        r1 = s[(s.etreat == "E_own_fixed") & (s.stage == "resid") & (s.lam == 0.25)]
        if len(phys) and len(own) and len(r0) and len(r1):
            dec.append(dict(target=t, source="a2", rmse_phys=float(phys.rmse_mean.iloc[0]), level=float(own.d_phys_mean.iloc[0]),
                            struct_on_E0=float(r0.d_phys_mean.iloc[0]), struct_on_Eown=float(r1.rmse_mean.iloc[0] - own.rmse_mean.iloc[0]),
                            total_Eown_resid=float(r1.d_phys_mean.iloc[0]), n_eval=int(s.n_eval.iloc[0]) if "n_eval" in s else np.nan))
else:
    C = rd(H3 / "h25_curve.csv")
    if C is not None:
        q = C[(C.scope == "allA")]
        for t, s in q.groupby("target"):
            phys = s.rmse_mean - s.d_phys_mean
            ref = s[(s.stage == "S1refit")]; sh = s[(s.stage == "S1")]; s3 = s[(s.stage == "S3") & (s.lam == 0.25) & (s.alpha == 1.0)]
            if len(ref) and len(sh) and len(s3):
                dec.append(dict(target=t, source="h25_allA", rmse_phys=float(phys.mean()), level=float(ref.d_phys_mean.mean()), level_shrunk=float(sh.d_phys_mean.mean()),
                                struct_on_Eshrunk=float(s3.rmse_mean.mean() - sh.rmse_mean.mean()), total_shrunk_resid=float(s3.d_phys_mean.mean()),
                                n_splits=int(s.n_splits.max()) if "n_splits" in s else np.nan))
D = pd.DataFrame(dec)
if len(D):
    Tg = rd(H3 / "h25_targets.csv")
    if Tg is not None:
        er = (Tg.groupby("target").E_own.first() / Tg.groupby("target").E0.mean())
        D["abslogE"] = D.target.map(lambda t: abs(np.log(er.get(t, np.nan))))
    D = D.sort_values("abslogE", ascending=False)
    D.to_csv(OUT / "a1_decomp.csv", index=False); print("[A1 분해(라벨 전량)]"); print(D.round(2).to_string(index=False))

# ---------------------------------------------------------------- C3 배포 산출물 표
dep = []
H30 = rd(H3 / "h30_deploy_tests.csv")
if H30 is not None:
    for _, r in H30[H30.target.astype(str).str.startswith("MEAN")].iterrows():
        dep.append(dict(source="h30", test=r.test, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_blockeq=r.delta_blockeq))
C1 = rd(OUT / "c1_tests.csv")
if C1 is not None and "test" in C1:
    for _, r in C1[C1.target.astype(str).str.startswith("MEAN")].iterrows():
        dep.append(dict(source="c1", test=r.test, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_blockeq=r.get("delta_blockeq", np.nan)))
C2 = rd(OUT / "c2_coverage.csv")
if C2 is not None:
    for _, r in C2.iterrows():
        dep.append(dict(source="c2", test=f"coverage90:{r.get('method', '')}:{r.get('target', '')}", delta=r.get("coverage", np.nan), ci_lo=np.nan, ci_hi=np.nan, delta_blockeq=r.get("width_cm", np.nan)))
DP = pd.DataFrame(dep); DP.to_csv(OUT / "c3_deploy_table.csv", index=False)
if len(DP):
    print("[C3 배포 표]"); print(DP.round(2).to_string(index=False))

# ---------------------------------------------------------------- D4 실용 지표(확정 비교의 지역별 최악·악화 비율)
prac = []
def practical(df, name_col, tgt_col, d_col, tag):
    for name, s in df.groupby(name_col):
        s = s[~s[tgt_col].astype(str).str.startswith("MEAN") & ~s[tgt_col].astype(str).str.startswith("REGION")]
        if len(s) == 0:
            continue
        prac.append(dict(source=tag, test=name, n_regions=len(s), worst_delta=float(s[d_col].max()), worst_region=str(s.loc[s[d_col].idxmax(), tgt_col]),
                         frac_worse=float((s[d_col] > 0).mean()), best_delta=float(s[d_col].min())))
if T is not None:
    practical(T[T.H.isin(["H3", "H13"])], "label", "target", "delta_rmse", "m1")
if H28 is not None:
    practical(H28, "test", "target", "delta", "h28")
if H30 is not None:
    practical(H30, "test", "target", "delta", "h30")
if C1 is not None and "test" in C1:
    practical(C1, "test", "target", "delta", "c1")
PR = pd.DataFrame(prac); PR.to_csv(OUT / "d4_practical.csv", index=False)
if len(PR):
    print("[D4 실용 지표]"); print(PR.round(2).to_string(index=False))
# ---------------------------------------------------------------- BLK 블록 CI 표준화
blk_reaggregate(int(sys.argv[sys.argv.index("--nverify") + 1]) if "--nverify" in sys.argv else 6)
print("done")
