"""보고서 그림: 전 접근법 LORO 전이 성능 종합(가로 막대).

산출: outputs/figures/02_evaluation/transfer_loro_summary.{png,pdf}

계열 그룹핑
- 지구통계: IDW, OK, RK                     ← e1_kriging_results.csv (LORO_gate, seed 평균)
- 순수 기계학습: CatBoost, MLP, FT-Transformer ← s11_comparison_table.csv / s1_baseline_results.csv
- 물리: Stefan 앵커(최소제곱 E)              ← s11_comparison_table.csv
- 구조 정교화: 잔차학습, source-aware, mixture ← s11_comparison_table.csv

모든 수치는 CSV에서 읽는다. 하드코딩 검증(assert)은 2026-07-28 기준 CSV 값 출처 주석과
함께 아래 VERIFY 블록에 있다. 그림 내부 제목·주장 문장 없음(캡션이 대신한다).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/willy010313/Polar_Bigdata"
sys.path.insert(0, os.path.join(ROOT, "src"))
from polar.plotstyle import use_polar, despine  # noqa: E402

plt = use_polar()
# 저널 제출 규격: Type 3(비트맵 힌팅) 금지 → TrueType(Type 42) 임베드로 강제.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PROC = os.path.join(ROOT, "data", "processed")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "02_evaluation")
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------- 데이터 적재
def load_values():
    """CSV 3종에서 (방법, LORO 게이트 RMSE, in-domain RMSE)를 읽는다."""
    e1 = pd.read_csv(os.path.join(PROC, "e1_kriging_results.csv"))
    s11 = pd.read_csv(os.path.join(PROC, "s11_comparison_table.csv"))
    s1 = pd.read_csv(os.path.join(PROC, "s1_baseline_results.csv"))

    # -- 지구통계: e1 LORO_gate(지역 비가중평균, UNWEIGHTED_MEAN 행) seed 평균
    gate = e1[e1.cv == "LORO_gate"].groupby("method")["rmse_cm"].mean()
    # in-domain = spatial_block_AK seed 평균
    indom_e1 = e1[e1.cv == "spatial_block_AK"].groupby("method")["rmse_cm"].mean()

    # -- FT-Transformer: s1_baseline_results.csv LORO에서
    #    seed별(지역 비가중평균) → seed 평균 재계산
    ftt = s1[(s1.cv == "LORO") & (s1.model == "ftt")]
    ftt_loro = ftt.groupby("seed")["rmse_cm"].mean().mean()

    # -- s11 종합표에서 방법별 행 추출
    def s11_row(pred):
        rows = s11[s11.method.map(pred)]
        assert len(rows) == 1, rows
        return rows.iloc[0]

    r_mlp = s11_row(lambda m: m.startswith("MLP(3-seed"))
    r_cat = s11_row(lambda m: m == "CatBoost")
    r_ftt = s11_row(lambda m: m == "FT-Transformer")
    r_stef = s11_row(lambda m: "최소제곱 E" in m)
    r_res = s11_row(lambda m: "저용량잔차" in m)
    r_saz = s11_row(lambda m: m.startswith("sa_z"))
    r_mix = s11_row(lambda m: m.startswith("mix_logit"))

    # (표시명, LORO 게이트 RMSE, in-domain RMSE)
    data = {
        "지구통계": [
            ("IDW", gate["IDW"], indom_e1["IDW"]),
            ("RK", gate["RK"], indom_e1["RK"]),
            ("OK", gate["OK"], indom_e1["OK"]),
        ],
        "순수 기계학습": [
            ("CatBoost", r_cat.loro_gate_rmse_cm, r_cat.indomain_rmse_cm),
            ("MLP", r_mlp.loro_gate_rmse_cm, r_mlp.indomain_rmse_cm),
        ],
        "물리": [
            ("Stefan 앵커", r_stef.loro_gate_rmse_cm, r_stef.indomain_rmse_cm),
        ],
        "구조 정교화": [
            ("mixture-of-physics", r_mix.loro_gate_rmse_cm, r_mix.indomain_rmse_cm),
            ("source-aware", r_saz.loro_gate_rmse_cm, r_saz.indomain_rmse_cm),
            ("잔차학습", r_res.loro_gate_rmse_cm, r_res.indomain_rmse_cm),
        ],
    }

    # -------- VERIFY: 2026-07-28 CSV 기준 기대값(출처 주석). 데이터 변경 시 갱신.
    chk = {name: v for grp in data.values() for name, v, _ in grp}
    expect = {
        "IDW": 50.92,                # e1 LORO_gate IDW seed0 50.924
        "OK": 29.40,                 # e1 LORO_gate OK seed0 29.4017
        "RK": 36.30,                 # e1 LORO_gate RK seed 0/1/2 = 38.299/32.057/38.551 평균
        "CatBoost": 38.42,           # s11 S1 CatBoost loro_gate_rmse_cm
        "MLP": 34.24,                # s11 S1 MLP(3-seed 앙상블 OOF) loro_gate_rmse_cm
        "Stefan 앵커": 21.26,        # s11 S2/S4 Stefan E·√TDD(최소제곱 E) loro_gate_rmse_cm
        "mixture-of-physics": 29.22, # s11 S8 mix_logit loro_gate_rmse_cm
        "source-aware": 21.84,       # s11 S6 sa_z loro_gate_rmse_cm
        "잔차학습": 21.83,           # s11 S4 잔차학습 loro_gate_rmse_cm
    }
    for k, v in expect.items():
        assert abs(chk[k] - v) < 0.05, f"{k}: csv={chk[k]:.3f} != expect {v}"
    return data


# ---------------------------------------------------------------- 그리기
def draw(data):
    stefan_ref = [v for n, v, _ in data["물리"] if n == "Stefan 앵커"][0]

    # 계열 색(냉색 규약: 빨강·주황 금지, 미세 구분)
    fam_color = {
        "지구통계": "#8aa2b8",       # 청회색
        "순수 기계학습": "#5b8fa8",  # 청록빛 청색
        "물리": "#1f4e79",           # 진청(앵커 강조)
        "구조 정교화": "#7f86b8",    # 청자색
    }
    header_c = "#243b53"

    # y 배치: 그룹 헤더 1행 + 방법 1행씩, 그룹 사이 0.45 간격
    rows = []  # (y, kind, label, loro, indom, color)
    y = 0.0
    for fam, members in data.items():
        rows.append((y, "header", fam, None, None, None))
        y += 1.0
        for name, loro, indom in members:
            rows.append((y, "bar", name, loro, indom, fam_color[fam]))
            y += 1.0
        y += 0.45

    # 인쇄폭 설계: tight bbox 최종폭 약 15 cm 타깃. 최소 폰트 11.5 pt(소스)로
    # 8 cm 단일컬럼 축소 시에도 11.5 × 8/15 ≈ 6.1 pt ≥ 6 pt 판독선 유지.
    fig, ax = plt.subplots(figsize=(4.6, 4.9))

    bar_rows = [r for r in rows if r[1] == "bar"]
    ys = [r[0] for r in bar_rows]
    vals = [r[3] for r in bar_rows]
    cols = [r[5] for r in bar_rows]
    ax.barh(ys, vals, height=0.66, color=cols, edgecolor="#3a4a5a",
            linewidth=0.5, zorder=3)

    # Stefan 기준선(점선)
    ax.axvline(stefan_ref, ls=(0, (5, 3)), lw=1.3, color="#1f4e79", zorder=2)

    # 값 라벨 + in-domain 보조 마커(흰 원)
    for _y, _k, _n, loro, indom, _c in bar_rows:
        ax.text(loro + 0.7, _y, f"{loro:.2f}", va="center", ha="left",
                fontsize=11.5, color="#333333", zorder=4)
        ax.plot(indom, _y, marker="o", ms=7, mfc="white", mec="#3a4a5a",
                mew=1.1, ls="none", zorder=5)

    # y 틱: 좌측 정렬(헤더 굵게, 방법명 들여쓰기)
    ax.set_yticks([r[0] for r in rows])
    labels = [(r[2] if r[1] == "header" else "   " + r[2]) for r in rows]
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=142)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_ha("left")
        if r[1] == "header":
            tick.set_fontweight("bold")
            tick.set_color(header_c)
            tick.set_fontsize(12.5)
        else:
            tick.set_fontsize(12)

    ax.invert_yaxis()
    ax.set_ylim(y - 0.9, -0.7)
    ax.set_xlim(0, 60)
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_xlabel("LORO 전이 RMSE (cm, 3개 지역 비가중평균)", fontsize=13)
    ax.grid(axis="x", color="#cccccc", lw=0.5, alpha=0.5)
    ax.grid(axis="y", visible=False)
    despine(ax)
    ax.tick_params(axis="x", labelsize=11.5)

    # 범례
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls=(0, (5, 3)), lw=1.3, color="#1f4e79",
               label=f"Stefan 앵커 기준선 ({stefan_ref:.2f} cm)"),
        Line2D([], [], marker="o", ms=7, mfc="white", mec="#3a4a5a", mew=1.1,
               ls="none", label="in-domain RMSE (공간블록 CV, cm)"),
    ]
    # 범례는 축 아래 바깥 좌측(축 내부는 하단 막대·값 라벨과 겹침)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.13),
              fontsize=12, borderpad=0.7, frameon=False, handletextpad=0.6,
              borderaxespad=0.0)


    for ext in ("png", "pdf"):
        path = os.path.join(OUTDIR, f"transfer_loro_summary.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("saved:", path)
    plt.close(fig)


if __name__ == "__main__":
    draw(load_values())
