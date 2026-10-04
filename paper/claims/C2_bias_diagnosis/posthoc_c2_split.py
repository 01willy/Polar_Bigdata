"""C2 사후 점검(2026-10-04): P0 대비 이득을 재보정 몫과 재보정을 넘어선 ML 몫으로 나눈 순위상관.

판정이 아니다. 블록 재표집 CI 없이 대상 단위 Spearman(해석적 p)만 계산한다. 대상은 서로 독립이 아니다
(하위 지역 i·x 모드 중복, 알래스카·캐나다·레나 하위 지역 중첩). 등록 분석은 XA_c2_gain_decomposition 이 대신한다.

입력은 같은 폴더의 tables/ 사본만 쓴다(원본을 읽거나 고치지 않는다).
  tables/wf0_misspec.csv  (WF0, 라벨 전량, 30대상)
  tables/wf_tests.csv     (WF4-c 진단값·이득, WF4-b 의 W − P1·R1(0.25) − P1, 라벨 전량)
  tables/lg_targets.csv   (LG 대상별 E_own, E0, logE_ratio_own, E_block_cv; part == 'cpu' 첫 행)
산출: tables/derived_c2_wf0_split.csv, tables/derived_c2_wf4c_split.csv, tables/derived_c2_posthoc_rho.csv
실행: OMP_NUM_THREADS=1 python paper/claims/C2_bias_diagnosis/posthoc_c2_split.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

T = Path(__file__).resolve().parent / "tables"
rows = []


def rho(tag, a, b, x, y, note=""):
    z = pd.DataFrame({"a": x, "b": y}).dropna()
    s = spearmanr(z.a, z.b)
    rows.append(dict(block=tag, x=a, y=b, n_targets=len(z), rho=float(s.correlation), p_two_sided=float(s.pvalue), note=note))


# A. WF0(라벨 전량, 30대상). ml_vs_p1 = 최선 ML Δ − P1 Δ(둘 다 P0 대비) = RMSE(최선 ML) − RMSE(P1)
w0 = pd.read_csv(T / "wf0_misspec.csv")
w0["ml_gain_total"] = -w0["ml_best_delta"]          # P0 − 최선 ML
w0["ml_beyond_p1"] = -w0["ml_vs_p1"]                # P1 − 최선 ML
tg = pd.read_csv(T / "lg_targets.csv")
tg = tg[tg.part == "cpu"].groupby(["target", "mode"], as_index=False).agg(
    E0=("E0", "first"), E_own=("E_own", "first"), logE_ratio_own=("logE_ratio_own", "first"), E_block_cv=("E_block_cv", "first"))
tg["key"] = tg.target + "|" + tg["mode"]
w0 = w0.merge(tg[["key", "E0", "E_own", "logE_ratio_own", "E_block_cv"]], left_on="target", right_on="key", how="left").drop(columns="key")
w0.loc[w0.src != "LG", ["E0", "E_own", "logE_ratio_own", "E_block_cv"]] = float("nan")  # LGD 대상은 lg_targets 와 자료판이 다르다
w0["abs_logE_ratio_own"] = w0.logE_ratio_own.abs()
rho("WF0_30", "recal_gain(P0-P1)", "ml_gain_total(P0-bestML)", w0.recal_gain, w0.ml_gain_total, "wf0_meta 0.66 재현")
rho("WF0_30", "recal_gain(P0-P1)", "ml_beyond_p1(P1-bestML)", w0.recal_gain, w0.ml_beyond_p1, "QA_FINAL_REVIEW Q1 C2 점검 -0.08 재현")
rho("WF0_30", "|recal_gain|", "ml_beyond_p1(P1-bestML)", w0.recal_gain.abs(), w0.ml_beyond_p1, "계수 오차를 크기로 정의")
rho("WF0_30", "P0_rmse", "ml_beyond_p1(P1-bestML)", w0.P0, w0.ml_beyond_p1, "총 오차 수준(계수 오차가 아님)")
lg = w0[w0.src == "LG"]
rho("WF0_LG25", "|log(E_own/E0)|", "recal_gain(P0-P1)", lg.abs_logE_ratio_own, lg.recal_gain, "직접 계수 오차, LG 대상만")
rho("WF0_LG25", "|log(E_own/E0)|", "ml_beyond_p1(P1-bestML)", lg.abs_logE_ratio_own, lg.ml_beyond_p1, "직접 계수 오차, LG 대상만")
rho("WF0_LG25", "E_block_cv", "ml_beyond_p1(P1-bestML)", lg.E_block_cv, lg.ml_beyond_p1, "블록 간 E 변동(이질성)")
w0.to_csv(T / "derived_c2_wf0_split.csv", index=False)

# B. WF4-c(라벨 10개 진단값, 라벨 전량 이득, 28대상)
d = pd.read_csv(T / "wf_tests.csv")
c = d[(d.test_id == "WF4-c") & (d.scope == "region")][["target", "diag_abs", "diag_signed", "n_diag", "gain_w", "gain_r1"]]
b = d[(d.test_id == "WF4-b") & (d.scope == "region") & (d.n == -1)]
w = b[b.contrast == "W-P1|n전량"][["target", "delta", "delta_blockeq", "verdict4"]].rename(
    columns={"delta": "W_minus_P1", "delta_blockeq": "W_minus_P1_beq", "verdict4": "verdict4_W_P1"})
r = b[b.contrast == "R1(0.25)-P1|n전량"][["target", "delta", "verdict4"]].rename(
    columns={"delta": "R1_minus_P1", "verdict4": "verdict4_R1_P1"})
m = c.merge(w, on="target", how="left").merge(r, on="target", how="left")
m["recal_share"] = m.gain_w + m.W_minus_P1          # (P0 − W) + (W − P1) = P0 − P1
m["ml_beyond_W"] = -m.W_minus_P1                    # P1 − W
m["ml_beyond_R1"] = -m.R1_minus_P1                  # P1 − R1(0.25)
m["check_recal_via_r1"] = m.gain_r1 + m.R1_minus_P1  # 같은 P0 − P1 이어야 한다
for tag, z in (("WF4c_28", m), ("WF4c_27_noTibet", m[m.target != "Tibet_LGD|x"])):
    rho(tag, "|bias10|", "gain_w(P0-W)", z.diag_abs, z.gain_w, "WF4-c ρ 0.38 재현" if tag == "WF4c_28" else "")
    rho(tag, "|bias10|", "recal_share(P0-P1)", z.diag_abs, z.recal_share)
    rho(tag, "|bias10|", "ml_beyond_W(P1-W)", z.diag_abs, z.ml_beyond_W)
    rho(tag, "|bias10|", "gain_r1(P0-R1_0.25)", z.diag_abs, z.gain_r1, "WF4-c rho_gain_r1 0.47 재현" if tag == "WF4c_28" else "")
    rho(tag, "|bias10|", "ml_beyond_R1(P1-R1_0.25)", z.diag_abs, z.ml_beyond_R1)
m.to_csv(T / "derived_c2_wf4c_split.csv", index=False)

out = pd.DataFrame(rows)
out.to_csv(T / "derived_c2_posthoc_rho.csv", index=False)
print(out.round(4).to_string(index=False))
print("check max|recal_share - check_recal_via_r1| =", float((m.recal_share - m.check_recal_via_r1).abs().max()))
