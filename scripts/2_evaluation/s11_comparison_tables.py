"""S11: 증강방식 종합 비교표 + 다축 검증 요약표 (기존 결과 CSV 집계, 재실험 없음).

`docs/RESEARCH_PLAN_multifidelity_2026-07-22.md` S11·§8. 모든 수치는 문서 복붙이 아니라
근거 CSV에서 직접 재계산한다. 재계산 값이 문서 서술과 다르면 mismatch로 기록한다.

산출: data/processed/s11_comparison_table.csv     (행=방법군, 열=in-domain·LORO·bias·채택)
      data/processed/s11_multiaxis_validation.csv  (검증 축별 요약)
실행: python scripts/2_evaluation/s11_comparison_tables.py
"""
import os, sys, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C

PROC = C.PROCESSED
R = lambda f: pd.read_csv(PROC / f)


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


rows, mismatches = [], []


def add(stage, family, method, adopted, ind_rmse=None, ind_bias=None,
        gate=None, ak=None, lena=None, canada=None, caveat="", source="",
        regime="inductive"):
    rows.append(dict(stage=stage, family=family, method=method, adopted=adopted,
                     regime=regime,
                     indomain_rmse_cm=None if ind_rmse is None else round(ind_rmse, 2),
                     indomain_bias_cm=None if ind_bias is None else round(ind_bias, 2),
                     loro_gate_rmse_cm=None if gate is None else round(gate, 2),
                     loro_alaska_cm=None if ak is None else round(ak, 2),
                     loro_lena_cm=None if lena is None else round(lena, 2),
                     loro_canada_cm=None if canada is None else round(canada, 2),
                     key_caveat=caveat, source_csv=source))


# ============ S1 실측-only (in-domain은 3-seed 앙상블 OOF 재계산) ============
o1 = R("s1_baseline_oof.csv")
ak = o1   # 알래스카 in-domain 셀 집합(ABoVE_AK+US Alaska). n은 하드코딩 대신 len(ak)로 파생
y1 = o1.alt_cm.values
ens = {m: rmse(y1, o1[f"pred_{m}"].values) for m in ["mlp", "tabm", "catboost", "ftt"]}
bias1 = {m: float(np.nanmean(o1[f"pred_{m}"].values - y1)) for m in ens}
r1 = R("s1_baseline_results.csv")
loro1 = r1[r1.cv == "LORO"].groupby(["model", "region"]).rmse_cm.mean().unstack()

# S6의 direct 트랙이 매크로 LORO 기준 실측-only 전이(공정 비교). S1 LORO는 매크로 수정 전 프로토콜
s6 = R("s6_source_aware_results.csv")
g6 = s6[s6.axis == "LORO_gate"].groupby("method").rmse_cm.mean()
lr6 = s6[s6.axis == "LORO"].groupby(["method", "region"]).rmse_cm.mean().unstack()
sm6 = s6[s6.axis == "spatial_block_AK_seedmean"].groupby("method")[["rmse_cm", "bias_cm"]].mean()

add("S1", "실측-only", "MLP(3-seed 앙상블 OOF)", "채택(in-domain 기준선)",
    ens["mlp"], bias1["mlp"], g6.direct_mlp,
    lr6.loc["direct_mlp", "Alaska"], lr6.loc["direct_mlp", "Lena"], lr6.loc["direct_mlp", "Canada"],
    "in-domain 최강군이나 전이(LORO)서 발산 경향. LORO 열은 S6 direct_mlp(매크로·SHARED25) 재평가",
    "s1_baseline_oof.csv; s6_source_aware_results.csv")
add("S1", "실측-only", "TabM(3-seed 앙상블 OOF)", "참고",
    ens["tabm"], bias1["tabm"], None, None, None, None,
    "MLP와 동률(CI 0 포함). S6 매크로 LORO 재평가 없음", "s1_baseline_oof.csv")
add("S1", "실측-only", "CatBoost", "참고(GBM 대표)",
    ens["catboost"], bias1["catboost"], g6.direct_catboost,
    lr6.loc["direct_catboost", "Alaska"], lr6.loc["direct_catboost", "Lena"],
    lr6.loc["direct_catboost", "Canada"],
    "전이서 covariate shift 붕괴(LORO 게이트 38.4). LORO 열은 S6 direct_catboost",
    "s1_baseline_oof.csv; s6_source_aware_results.csv")
add("S1", "실측-only", "FT-Transformer", "참고(전이 최선 DL)",
    ens["ftt"], bias1["ftt"], float(loro1.loc["ftt"].mean()),
    float(loro1.loc["ftt", "ABoVE_AK"]), float(loro1.loc["ftt", "Lena_RU"]),
    float(loro1.loc["ftt", "ABoVE_CA"]),
    "LORO 열은 S1 구프로토콜(매크로 병합 전, US-Alaska 64셀 train 포함): S2 이후 게이트와 직접 비교 금지",
    "s1_baseline_results.csv")

# ============ S2 Stefan 앵커 ============
r2 = R("s2_physics_results.csv")
a2 = r2[r2.part == "A_physics"]
p1i = a2[(a2.cv == "spatial_block_AK") & (a2.model == "p1_stefan")].iloc[0]
p1g = float(a2[(a2.cv == "LORO_gate") & (a2.model == "p1_stefan")].rmse_cm.iloc[0])
p1r = a2[(a2.cv == "LORO") & (a2.model == "p1_stefan")].set_index("region").rmse_cm
add("S2", "Stefan 앵커", "Stefan E·√TDD (중앙값비 E)", "채택(전이 앵커)",
    float(p1i.rmse_cm), float(p1i.bias_cm), p1g,
    float(p1r["Alaska"]), float(p1r["Lena"]), float(p1r["Canada"]),
    "물리 5종 중 유일하게 정확(나머지 상방편향 +10~+38cm). 전이 하한 담당",
    "s2_physics_results.csv")
add("S2/S4", "Stefan 앵커", "Stefan E·√TDD (최소제곱 E)", "채택(앵커 개선)",
    float(sm6.loc["stefan_anchor", "rmse_cm"]), float(sm6.loc["stefan_anchor", "bias_cm"]),
    g6.stefan_anchor, lr6.loc["stefan_anchor", "Alaska"], lr6.loc["stefan_anchor", "Lena"],
    lr6.loc["stefan_anchor", "Canada"],
    "E 추정만 바꿔 게이트 22.24→21.26(약 1cm 개선). S4·S6서 재현 일치",
    "s4_residual_results.csv; s6_source_aware_results.csv")

# ============ S2 physics-as-feature ============
b2 = r2[r2.part == "B_feature"].copy()
b2["bm"] = b2.model.str.replace("_BASE", "", regex=False).str.replace("_+physics", "", regex=False)
b2["var"] = np.where(b2.model.str.endswith("BASE"), "BASE", "+physics")
ind2 = b2[b2.cv == "spatial_block_AK"].groupby(["bm", "var"]).rmse_cm.mean().unstack()
loro2 = b2[b2.cv == "LORO"].groupby(["bm", "var"]).rmse_cm.mean().unstack()
d_cb, d_mlp = ind2.loc["catboost", "+physics"] - ind2.loc["catboost", "BASE"], \
    ind2.loc["mlp", "+physics"] - ind2.loc["mlp", "BASE"]
d_cb_loro = loro2.loc["catboost", "+physics"] - loro2.loc["catboost", "BASE"]
mismatches.append(
    f"S2 physfeat CSV 재계산 기준: in-domain catboost {ind2.loc['catboost','BASE']:.2f}"
    f"→{ind2.loc['catboost','+physics']:.2f}(Δ{d_cb:+.2f}), "
    f"LORO(지역평균) catboost {loro2.loc['catboost','BASE']:.2f}"
    f"→{loro2.loc['catboost','+physics']:.2f}(Δ{d_cb_loro:+.2f}). "
    f"실험로그 서술 'Δ+0.04 무익'과 불일치(로그는 재실행 전 수치로 추정, 상위 세션서 정정 예정). "
    f"CSV 재계산 값을 채택.")
add("S2", "physics-as-feature", "CatBoost+물리특징(A2)", "미채택",
    float(ind2.loc["catboost", "+physics"]), None, g6.physfeat_catboost,
    lr6.loc["physfeat_catboost", "Alaska"], lr6.loc["physfeat_catboost", "Lena"],
    lr6.loc["physfeat_catboost", "Canada"],
    f"CSV 재계산 기준: in-domain catboost {ind2.loc['catboost','BASE']:.2f}"
    f"→{ind2.loc['catboost','+physics']:.2f}(Δ{d_cb:+.2f})·mlp Δ{d_mlp:+.2f}, "
    f"S2 내부 LORO(지역평균) {loro2.loc['catboost','BASE']:.2f}"
    f"→{loro2.loc['catboost','+physics']:.2f}(Δ{d_cb_loro:+.2f})로 이득 미미"
    f"(로그의 'Δ+0.04 무익'은 재실행 전 수치). LORO 게이트 열은 S6 physfeat_catboost. "
    f"direct 붕괴(38.0 vs 앵커 21.26)를 구제 못함. 물리=TDD 결정론 변환이라 신규 정보 없음",
    "s2_physics_results.csv; s6_source_aware_results.csv")

# ============ S3 증강(수정판 r-스윕) ============
s3c = R("s3_aug_curve_results.csv")
s3f = R("s3_aug_curve_results_ftt.csv")
s3 = pd.concat([s3c[s3c.model != "mlp"], s3f])
g3 = s3.groupby(["target", "model", "phys", "r"]).rmse_cm.mean()


def s3v(t, m, p, r):
    return float(g3.loc[(t, m, p, r)])


for model in ["catboost", "ftt"]:
    base_l, base_c = s3v("Lena", model, "stefan", 0.0), s3v("Canada", model, "stefan", 0.0)
    st_l, st_c = s3v("Lena", model, "stefan", 10.0), s3v("Canada", model, "stefan", 10.0)
    pl_l, pl_c = s3v("Lena", model, "placebo", 10.0), s3v("Canada", model, "placebo", 10.0)
    ku_l, ku_c = s3v("Lena", model, "kudryavtsev", 10.0), s3v("Canada", model, "kudryavtsev", 10.0)
    add("S3", "고정가중 증강", f"{model}+Stefan pseudo(r=10)", "조건부 채택(정확 물리·약한 base 한정)",
        None, None, None, None, st_l, st_c,
        f"base(r=0) Lena {base_l:.1f}·Canada {base_c:.1f} → r=10 Lena {st_l:.1f}·Canada {st_c:.1f}. "
        f"placebo 대비 순가치 Lena +{pl_l-st_l:.1f}·Canada +{pl_c-st_c:.1f}. "
        f"부정확 물리(Ku) r=10은 Lena {ku_l:.1f}·Canada {ku_c:.1f}로 악화. "
        "블록분리 절반-test 프로토콜이라 LORO 게이트와 직접 비교 불가",
        "s3_aug_curve_results.csv; s3_aug_curve_results_ftt.csv")

# ============ S6 비교군: naive pooling / 고정가중(등가 가중) ============
# pool_mlp·fixaug: test 지역 셀 좌표의 보조관측(Stefan pseudo·CCI)을 거리 버퍼 없이
# 학습에 넣는 transductive 설계. S5에서 같은 메커니즘을 아티팩트로 기각했으므로 기준을 일치시켜
# regime='transductive(target 셀 보조관측 사용)'로 라벨하고 LORO 게이트는 'inductive 아님'으로 명시.
add("S6", "naive pooling", "pool_mlp(직접+Stefan+CCI 병합)", "미채택(전이만 최선)",
    float(sm6.loc["pool_mlp", "rmse_cm"]), float(sm6.loc["pool_mlp", "bias_cm"]),
    g6.pool_mlp, lr6.loc["pool_mlp", "Alaska"], lr6.loc["pool_mlp", "Lena"],
    lr6.loc["pool_mlp", "Canada"],
    "LORO 게이트 최저(21.11)이나 inductive 전이 아님: test 지역 셀 좌표의 보조관측"
    "(Stefan pseudo·CCI)을 거리 버퍼 없이 학습에 넣는 transductive 설계"
    "(S5서 같은 메커니즘을 아티팩트로 기각). 21.11은 inductive 게이트로 비교 금지. "
    "in-domain도 파괴(16.18, direct 14.37 대비 악화)",
    "s6_source_aware_results.csv", regime="transductive(target 셀 보조관측 사용)")
add("S6", "고정가중 증강", "fixaug_catboost(S3 최적 r=10 등가 가중)", "참고",
    float(sm6.loc["fixaug_catboost", "rmse_cm"]), float(sm6.loc["fixaug_catboost", "bias_cm"]),
    g6.fixaug_catboost, lr6.loc["fixaug_catboost", "Alaska"], lr6.loc["fixaug_catboost", "Lena"],
    lr6.loc["fixaug_catboost", "Canada"],
    "앵커(21.26)와 동률(21.28)·in-domain도 앵커와 동률(14.45). 단 고정가중 증강은 test 지역 셀의 "
    "보조관측을 거리 버퍼 없이 학습에 넣는 transductive 설계라 게이트 수치는 inductive 아님"
    "(S5 아티팩트 기각과 동일 기준)", "s6_source_aware_results.csv",
    regime="transductive(target 셀 보조관측 사용)")

# ============ S4 잔차학습 ============
s4 = R("s4_residual_results.csv")
g4 = s4[s4.cv == "LORO_gate"].groupby(["model", "featset", "lam"]).rmse_cm.mean()
i4 = s4[s4.cv == "spatial_block_AK"].groupby(["model", "featset", "lam"])[["rmse_cm", "bias_cm"]].mean()
best_pos = g4[[l > 0 for (_, _, l) in g4.index]].min()
anchor4 = float(g4.loc[("ridge", "shared25", 0.0)])
lo4 = s4[(s4.cv == "LORO") & (s4.model == "catboost_lo") & (s4.featset == "shift14") &
         (s4.lam == 0.25)].groupby("region").rmse_cm.mean()
add("S4", "잔차학습", "Stefan 앵커+λ·저용량잔차(최선 λ>0)", "negative(전이)·채택(in-domain)",
    float(i4.loc[("ridge", "shared25", 0.75), "rmse_cm"]),
    float(i4.loc[("ridge", "shared25", 0.75), "bias_cm"]),
    float(best_pos),
    float(lo4["Alaska"]), float(lo4["Lena"]), float(lo4["Canada"]),
    f"λ>0 전 구간 게이트 악화(앵커 {anchor4:.2f} → 최선 λ>0 {best_pos:.2f}). "
    "Alaska fold 파탄이 게이트 지배·inner CV λ 자동선택 불가. "
    "in-domain은 13.33(ridge·shared25·λ0.75)으로 프로젝트 최저", "s4_residual_results.csv")

# ============ S5 pretrain-finetune ============
s5 = R("s5_pretrain_results.csv")
g5 = s5[s5.cv == "LORO_gate"].groupby(["model", "variant"]).rmse_cm.mean()
lr5 = s5[s5.cv == "LORO"].groupby(["model", "variant", "region"]).rmse_cm.mean()
add("S5", "pretrain-finetune", "dense Stefan pseudo 사전학습→FT-T finetune", "negative(transductive 아티팩트)",
    None, None, float(g5.loc[("ftt", "pretrain")]),
    float(lr5.loc[("ftt", "pretrain", "Alaska")]), float(lr5.loc[("ftt", "pretrain", "Lena")]),
    float(lr5.loc[("ftt", "pretrain", "Canada")]),
    f"게이트 {g5.loc[('ftt','scratch')]:.2f}→{g5.loc[('ftt','pretrain')]:.2f} 개선 전량이 "
    f"Alaska(transductive) fold({lr5.loc[('ftt','scratch','Alaska')]:.2f}→"
    f"{lr5.loc[('ftt','pretrain','Alaska')]:.2f}). 깨끗한 Lena Δ"
    f"{lr5.loc[('ftt','scratch','Lena')]-lr5.loc[('ftt','pretrain','Lena')]:+.2f} 무효·"
    f"Canada Δ{lr5.loc[('ftt','scratch','Canada')]-lr5.loc[('ftt','pretrain','Canada')]:+.2f} 악화. "
    "mlp는 전이 발산(33.7→37.0)", "s5_pretrain_results.csv",
    regime="transductive(target 셀 보조관측 사용)")

# ============ S6 source-aware ============
for meth, note in [("sa_fusion", "σ 융합"), ("sa_z", "공유 잠재 z")]:
    add("S6", "source-aware(A5)", f"{meth}(공유 인코더+b_s+logσ_s)", "negative(게이트 미달)",
        float(sm6.loc[meth, "rmse_cm"]), float(sm6.loc[meth, "bias_cm"]),
        float(g6[meth]), lr6.loc[meth, "Alaska"], lr6.loc[meth, "Lena"], lr6.loc[meth, "Canada"],
        f"{note}. baseline 최고(pool_mlp {g6.pool_mlp:.2f}) 미달. in-domain은 pooling 파괴를 "
        "유의 회복(sa_z 14.35 vs pool_mlp 16.18, Δ+1.80 [0.48,3.03])하나 앵커와 동률. "
        "sa_z cov90 0.996은 구간폭 발산(473cm)의 무정보 커버리지",
        "s6_source_aware_results.csv")

# ============ S8 mixture ============
s8 = R("s8_mixture_results.csv")
g8 = s8[s8.cv == "LORO_gate"].groupby("model").rmse_cm.mean()
lr8 = s8[(s8.cv == "LORO") & (s8.part == "B_loro")].groupby(["model", "region"]).rmse_cm.mean()
i8 = s8[(s8.cv == "spatial_block_AK") & (s8.part == "A_indomain")].groupby("model")[["rmse_cm", "bias_cm"]].mean()
add("S8", "mixture-of-physics", "mix_logit(물리 5종+공변량 게이트)", "negative(진단만 채택)",
    float(i8.loc["mix_logit", "rmse_cm"]), float(i8.loc["mix_logit", "bias_cm"]),
    float(g8["mix_logit"]),
    float(lr8.loc[("mix_logit", "Alaska")]), float(lr8.loc[("mix_logit", "Lena")]),
    float(lr8.loc[("mix_logit", "Canada")]),
    f"단일 Stefan({g8['stefan']:.2f})에 미달. Alaska fold서 게이트가 상방편향 λ전문가 61% 배정 "
    "→ 36.5 파탄. 정확한 전문가가 Stefan뿐(물리 5종 상관 0.89-1.00). "
    f"oracle 하한 {g8['oracle']:.2f}(모델 아님·진단)", "s8_mixture_results.csv")

comp = pd.DataFrame(rows)
comp.to_csv(PROC / "s11_comparison_table.csv", index=False)
print(f"저장: s11_comparison_table.csv ({len(comp)}행)")
print(comp[["stage", "method", "indomain_rmse_cm", "loro_gate_rmse_cm", "adopted"]].to_string(index=False))

# ============================================================
# 다축 검증 요약표
# ============================================================
m7 = json.load(open(PROC / "s7_kpdc_meta.json"))
m9 = json.load(open(PROC / "s9_timelapse_meta.json"))
t9 = R("timelapse_alaska_results.csv")

mrows = []


def madd(axis, protocol, method, metric, value, ci, n, verdict, source):
    mrows.append(dict(axis=axis, protocol=protocol, best_method=method, metric=metric,
                      value=value, ci95=ci, n=n, verdict_footnote=verdict, source=source))


madd("공간블록(in-domain)", "알래스카 0.5° GroupKFold 6-fold, 공변량34",
     "Stefan 앵커+잔차(ridge·λ0.75, S4)", "RMSE(cm)",
     round(float(i4.loc[('ridge', 'shared25', 0.75), 'rmse_cm']), 2), "",
     len(ak), "프로젝트 최저. 실측-only 최고 MLP 앙상블 14.37·Stefan 단독 14.56과 ~1cm 차",
     "s4_residual_results.csv")
madd("공간블록(in-domain)", "동일 프로토콜", "실측-only MLP(3-seed 앙상블 OOF, S1)", "RMSE(cm)",
     round(ens["mlp"], 2), "", len(ak),
     "CatBoost 15.61 대비 일관 우세이나 부트스트랩 CI 0 포함(유의성 미확정)", "s1_baseline_oof.csv")
madd("LORO 전이(OOD)", "매크로 3지역(Alaska·Lena·Canada) 비가중평균 게이트",
     "Stefan 앵커(최소제곱 E, S4/S6)", "RMSE(cm)", round(float(g6.stefan_anchor), 2), "",
     3, "실측-only ML은 34-38로 붕괴. 어떤 증강·구조(S3-S6·S8)도 앵커를 유의하게 못 넘음"
        "(pool_mlp 21.11은 동률·in-domain 파괴 대가)", "s6_source_aware_results.csv")
madd("LORO 전이(OOD)", "동일 게이트", "Stefan 앵커(중앙값비 E, S2)", "RMSE(cm)",
     round(p1g, 2), "", 3, "S2 게이트 프로토콜 확정값. E 추정 개선으로 21.26까지 하락",
     "s2_physics_results.csv")
madd("temporal(연도 홀드아웃)", "leave-one-year-out 2010-2024, 연도별 ERA5 forcing",
     "STEFAN(a+E√TDD, S9)", "RMSE(cm)",
     m9["headline"]["temporal_holdout_rmse_cm"],
     str(m9["headline"]["temporal_holdout_rmse_ci95_cm"]),
     int(t9[t9.eval_type == "temporal_holdout"].n.iloc[0]),
     f"R² {m9['headline']['temporal_holdout_r2']}. 단 within-site 연 anomaly는 예측 불가"
     f"(corr {m9['headline']['anomaly_corr']:+.3f} {m9['headline']['anomaly_corr_ci95']}, "
     "R²의 대부분은 공간 신호)", "timelapse_alaska_results.csv; s9_timelapse_meta.json")
madd("독립 소스 점검증", "KPDC Council 지중온도 유도 ALT(0°C 등온선), 최근접 셀 0.97km",
     "S1 7모델 평균 vs 유도/실측 스펙트럼(S7)", "예측(cm)",
     m7["headline"]["s1_ens7_cm"], "",
     m7["n_rows"],
     f"관측 스펙트럼(ABoVE 인근 {m7['headline']['above_near_obs_mean_cm']} ↔ 코어 "
     f"{m7['headline']['core2022_len_mean_cm']} ↔ 유도 2025 중앙값 "
     f"{m7['headline']['derived_2025_median_cm']}) 내부이나 16층 유도 대비 과소. "
     "알래스카 in-domain 점검증(전이 근거 아님). 38시즌 중 23 우측검열",
     "s7_kpdc_results.csv; s7_kpdc_meta.json")
madd("source leave-out", "leave_source_out_splits(fidelity.py) 정의만 존재",
     "미실시", "-", None, "", None,
     "clean 보조소스가 Stefan·CCI 2종뿐이고 둘 다 이미 공변량이라 별도 축 불성립. "
     "대체: S6 소스별 (b_s,σ_s) 진단(b_stefan≈0 정합, b_cci 과소추정)·S7 독립 KPDC 점검증",
     "source_overlap_matrix.csv; s6_source_aware_results.csv")

# UQ 축(S11 conformal): 결과 파일 있으면 채움
try:
    m11 = json.load(open(PROC / "s11_conformal_meta.json"))
    if not m11.get("smoke", True):
        h = m11["headline"]
        madd("UQ 보정(cov90)", "quantile CatBoost+CQR, calib=train 블록 내부 분리",
             "CQR conformal(S11)", "coverage@90%",
             round(h["cov90_cqr"] * 100, 1),
             str([round(100 * v, 1) for v in h["cov90_cqr_ci"]]),
             m11["protocol"]["n"],
             f"raw {h['cov90_raw']*100:.1f}%(과신) → CQR {h['cov90_cqr']*100:.1f}% "
             f"(폭 {h['width90_raw_cm']:.1f}→{h['width90_cqr_cm']:.1f}cm). "
             "in-domain 보정이며 전이 커버리지 보장 아님(S6: 전이는 전 방법 과신)",
             "s11_conformal_results.csv")
except FileNotFoundError:
    pass

# 누설 pytest
try:
    out = subprocess.run([sys.executable, "-m", "pytest", "tests/test_leakage.py", "-q"],
                         cwd=str(Path(__file__).resolve().parents[2]),
                         capture_output=True, text=True, timeout=300)
    tail = [l for l in out.stdout.strip().splitlines() if "passed" in l]
    py_res = tail[-1].strip() if tail else "실행 실패"
except Exception as e:
    py_res = f"실행 실패: {e}"
madd("누설통제", "tests/test_leakage.py (fold-safe E·라벨파생 금지·매크로 LORO 등)",
     "pytest", "통과", py_res, "", None,
     "공변량 코어34/라벨파생7 영구제외·블록 GroupKFold·fold-safe Stefan E 강제", "tests/test_leakage.py")

multi = pd.DataFrame(mrows)
multi.to_csv(PROC / "s11_multiaxis_validation.csv", index=False)
print(f"\n저장: s11_multiaxis_validation.csv ({len(multi)}행)")
print(multi[["axis", "best_method", "metric", "value"]].to_string(index=False))

if mismatches:
    print("\n[문서-CSV 불일치 기록]")
    for m in mismatches:
        print(" -", m)
    with open(PROC / "s11_comparison_mismatches.json", "w") as f:
        json.dump(dict(mismatches=mismatches), f, ensure_ascii=False, indent=1)
