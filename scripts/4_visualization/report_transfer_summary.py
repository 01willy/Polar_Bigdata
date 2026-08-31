"""보고서 그림: 방법 계열별 지역 간 전이 성능 비교(가로 막대).

산출: outputs/figures/02_evaluation/transfer_loro_summary.{png,pdf}

계열 그룹핑(그림 표시명 ← 원자료)
- 지구통계: 역거리가중, 회귀 크리깅, 보통 크리깅   ← e1_kriging_results.csv (LORO_gate, seed 평균)
- 순수 기계학습: CatBoost, 다층 퍼셉트론            ← s11_comparison_table.csv
- 물리: Stefan 물리식(최소제곱 E)                   ← s11_comparison_table.csv
- 구조 정교화: 혼합 모델, 다중충실도, 물리 결합      ← s11_comparison_table.csv / s4_residual_results.csv

수록 범위: 대상 지역 정보를 학습에 쓰지 않는(inductive) 현 게이트 프로토콜 결과만 담는다.
s11_comparison_table.csv 의 다음 항목은 비교 기준이 달라 제외한다.
- FT-Transformer(전이 22.46): S1 구프로토콜(매크로 병합 전, 알래스카 64셀 train 포함)
- pool_mlp(21.11)·fixaug_catboost(21.28)·sa_fusion(21.53): 대상 지역 셀의 보조 관측을
  거리 버퍼 없이 학습에 넣는 transductive 설계(sa_fusion은 sa_z 와 동일 계열의 변형)
따라서 캡션은 "전 접근법"이라 단정하지 않는다.

모든 수치는 CSV에서 읽는다. 하드코딩 검증(assert)은 2026-07-29 기준 CSV 값 출처 주석과
함께 아래 VERIFY 블록에 있다. 그림 내부 제목·주장 문장 없음(캡션이 대신한다).
그림에는 내부 약어(LORO·OOF·in-domain)와 원어 병기(source-aware·mixture-of-physics)를
쓰지 않는다(폭 제약. 원어는 캡션이 담당한다).

인쇄 1:1 규약: main.tex 는 이 그림을 subfigure 폭 0.47\\textwidth(=82.72 mm)에 넣는다.
figsize 폭을 그 값과 같게 잡고 bbox_inches 크롭을 끄므로 축소 배율이 1.0이 되어,
아래 FS_* 에 적은 pt 가 그대로 인쇄 pt 가 된다(최소 6.8 pt).
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
# 인쇄폭을 figsize 로 확정하므로 tight 크롭을 끈다(크롭은 폭을 바꿔 축소 배율을 깨뜨린다).
plt.rcParams["savefig.bbox"] = None

# 계열 헤더용 굵은 한글 자소. 기본 등록 폰트는 정체(regular)뿐이라 별도 등록한다.
_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
if os.path.exists(_BOLD):
    import matplotlib.font_manager as _fm
    _fm.fontManager.addfont(_BOLD)

PROC = os.path.join(ROOT, "data", "processed")
OUTDIR = os.path.join(ROOT, "outputs", "figures", "02_evaluation")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- 인쇄 규격
TEXTWIDTH_MM = 176.0                       # a4 · geometry left/right 17mm
FIG_W = 0.47 * TEXTWIDTH_MM / 25.4         # 3.257 in = 82.72 mm (subfigure 폭)
FIG_H = 2.60                               # in = 66.0 mm. 그림 7(a)와 높이를 맞춰 플로트 총높이 억제
AX_BOTTOM = 0.63                           # in — x 틱·축 라벨·2행 범례 영역
AX_TOP_GAP = 0.05                          # in
LAB_X = 0.02                               # in — y 라벨 블록 좌측 여백
LAB_GAP = 0.055                            # in — y 라벨과 축 사이 간격
RIGHT_GAP = 0.09                           # in. 중앙정렬 2자리 x 틱 라벨 반폭(4.2 pt) 확보

# 글자 크기(단일 관리). 축소 배율 1.0 이므로 아래 pt = 인쇄 pt. 하한 6.5 pt.
FS_METHOD = 7.0    # 방법명 y 틱
FS_HEADER = 7.2    # 계열 헤더 y 틱
FS_VALUE = 6.8     # 막대 끝 값 라벨
FS_AXIS = 7.2      # x축 라벨
FS_TICK = 6.8      # x 틱
FS_LEGEND = 6.8    # 범례(2행)


# ---------------------------------------------------------------- 데이터 적재
def load_values():
    """CSV 4종에서 (방법, 지역 간 전이 RMSE, 지역 내 RMSE)를 읽는다."""
    e1 = pd.read_csv(os.path.join(PROC, "e1_kriging_results.csv"))
    s11 = pd.read_csv(os.path.join(PROC, "s11_comparison_table.csv"))
    s4 = pd.read_csv(os.path.join(PROC, "s4_residual_results.csv"))

    # -- 지구통계: e1 LORO_gate(지역 비가중평균, UNWEIGHTED_MEAN 행) seed 평균
    gate = e1[e1.cv == "LORO_gate"].groupby("method")["rmse_cm"].mean()
    # 지역 내 = spatial_block_AK seed 평균
    indom_e1 = e1[e1.cv == "spatial_block_AK"].groupby("method")["rmse_cm"].mean()

    # -- s11 종합표에서 방법별 행 추출
    def s11_row(pred):
        rows = s11[s11.method.map(pred)]
        assert len(rows) == 1, rows
        return rows.iloc[0]

    r_mlp = s11_row(lambda m: m.startswith("MLP(3-seed"))
    r_cat = s11_row(lambda m: m == "CatBoost")
    r_stef = s11_row(lambda m: "최소제곱 E" in m)
    r_saz = s11_row(lambda m: m.startswith("sa_z"))
    r_mix = s11_row(lambda m: m.startswith("mix_logit"))

    # -- 물리 결합(전이 최적): s11 종합표의 잔차학습 행은 두 검증축의 값이 서로 다른
    #    구성에서 온다(전이 21.83 = catboost_lo·shift14·λ0.25, 지역 내 13.33 =
    #    ridge·shared25·λ0.75). 이 그림은 한 구성의 두 축을 짝지어야 하므로 s4 원표에서
    #    전이 최적 구성 하나만 골라 직접 읽는다(지역 내 13.92, 표 tab:transfer와 일치).
    def s4_pick(cv, region):
        rows = s4[(s4.model == "catboost_lo") & (s4.featset == "shift14")
                  & np.isclose(s4.lam, 0.25) & (s4.cv == cv) & (s4.region == region)]
        assert len(rows) >= 1, f"s4 행 없음: cv={cv}, region={region}"
        return float(rows["rmse_cm"].mean())   # seed 평균(LORO_gate는 단일 행)

    res_loro = s4_pick("LORO_gate", "UNWEIGHTED_MEAN")
    res_indom = s4_pick("spatial_block_AK", "Alaska")

    # (표시명, 지역 간 전이 RMSE, 지역 내 RMSE)
    data = {
        "지구통계": [
            ("역거리가중", gate["IDW"], indom_e1["IDW"]),
            ("회귀 크리깅", gate["RK"], indom_e1["RK"]),
            ("보통 크리깅", gate["OK"], indom_e1["OK"]),
        ],
        "순수 기계학습": [
            ("CatBoost", r_cat.loro_gate_rmse_cm, r_cat.indomain_rmse_cm),
            ("다층 퍼셉트론", r_mlp.loro_gate_rmse_cm, r_mlp.indomain_rmse_cm),
        ],
        "물리": [
            ("Stefan 물리식", r_stef.loro_gate_rmse_cm, r_stef.indomain_rmse_cm),
        ],
        "구조 정교화": [
            ("혼합 모델", r_mix.loro_gate_rmse_cm, r_mix.indomain_rmse_cm),
            ("다중충실도", r_saz.loro_gate_rmse_cm, r_saz.indomain_rmse_cm),
            ("물리 잔차 결합", res_loro, res_indom),
        ],
    }

    # -------- VERIFY: 2026-07-29 CSV 기준 기대값(전이, 지역 내). 데이터 변경 시 갱신.
    #          별표(*)는 보고서 표 tab:transfer 게재값과 대조되는 항목.
    chk = {name: (v, d) for grp in data.values() for name, v, d in grp}
    expect = {
        # 표시명:                      (전이,  지역 내)
        "역거리가중": (50.92, 15.80),   # e1 IDW seed0
        "보통 크리깅": (29.40, 15.77),  # e1 OK seed0                         *
        "회귀 크리깅": (36.30, 17.23),  # e1 RK seed 0/1/2 평균
        "CatBoost": (38.42, 15.61),     # s11 S1 CatBoost                     *
        "다층 퍼셉트론": (34.24, 14.37),  # s11 S1 MLP(3-seed 앙상블 OOF)     *
        "Stefan 물리식": (21.26, 14.46),  # s11 S2/S4 Stefan E·√TDD(최소제곱 E) *
        "혼합 모델": (29.22, 14.97),      # s11 S8 mix_logit (mixture-of-physics)
        "다중충실도": (21.84, 14.35),     # s11 S6 sa_z (source-aware)
        # s4 catboost_lo·shift14·λ0.25: LORO_gate 21.832 / spatial_block_AK
        # seed 0,1,2 = 13.902/13.953/13.903 → 13.919
        "물리 잔차 결합": (21.83, 13.92),      # 전이 최적 구성                    *
    }
    for k, (v, d) in expect.items():
        assert abs(chk[k][0] - v) < 0.05, f"{k} 전이: csv={chk[k][0]:.3f} != {v}"
        assert abs(chk[k][1] - d) < 0.05, f"{k} 지역 내: csv={chk[k][1]:.3f} != {d}"
    return data


# ---------------------------------------------------------------- 그리기
def draw(data):
    stefan_ref = [v for n, v, _ in data["물리"] if n == "Stefan 물리식"][0]

    # 계열 색(냉색 규약: 빨강·주황 금지, 미세 구분)
    fam_color = {
        "지구통계": "#8aa2b8",       # 청회색
        "순수 기계학습": "#5b8fa8",  # 청록빛 청색
        "물리": "#1f4e79",           # 진청(기준선 강조)
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

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0.30, AX_BOTTOM / FIG_H, 0.65,
                       1.0 - (AX_BOTTOM + AX_TOP_GAP) / FIG_H])   # 폭은 뒤에서 확정

    bar_rows = [r for r in rows if r[1] == "bar"]
    ys = [r[0] for r in bar_rows]
    vals = [r[3] for r in bar_rows]
    cols = [r[5] for r in bar_rows]
    ax.barh(ys, vals, height=0.62, color=cols, edgecolor="#3a4a5a",
            linewidth=0.35, zorder=3)

    # Stefan 기준선(점선)
    ax.axvline(stefan_ref, ls=(0, (4, 2.4)), lw=0.9, color="#1f4e79", zorder=2)

    # 값 라벨 + 지역 내 보조 마커(흰 원)
    for _y, _k, _n, loro, indom, _c in bar_rows:
        ax.text(loro + 0.9, _y, f"{loro:.2f}", va="center", ha="left",
                fontsize=FS_VALUE, color="#333333", zorder=4)
        ax.plot(indom, _y, marker="o", ms=3.4, mfc="white", mec="#3a4a5a",
                mew=0.6, ls="none", zorder=5)

    # y 틱: 좌측 정렬(헤더 굵게, 방법명 들여쓰기)
    ax.set_yticks([r[0] for r in rows])
    labels = [(r[2] if r[1] == "header" else "  " + r[2]) for r in rows]
    ax.set_yticklabels(labels)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_ha("left")
        if r[1] == "header":
            tick.set_fontweight("bold")
            tick.set_color(header_c)
            tick.set_fontsize(FS_HEADER)
        else:
            tick.set_fontsize(FS_METHOD)

    ax.invert_yaxis()
    ax.set_ylim(y - 0.9, -0.7)
    ax.set_xlim(0, 60)
    ax.set_xticks(np.arange(0, 61, 20))
    ax.set_xlabel("지역 간 전이 RMSE (cm)", fontsize=FS_AXIS, labelpad=2.0)
    ax.grid(axis="x", color="#cccccc", lw=0.4, alpha=0.6)
    ax.grid(axis="y", visible=False)
    despine(ax)
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2.0, pad=1.8)

    # y 라벨 블록 폭을 실측해 축 좌측 경계를 확정한다(라벨이 막대 영역을 침범하지 않게).
    left_in = _label_block(fig, ax) + LAB_X + LAB_GAP
    ax.set_position([left_in / FIG_W, AX_BOTTOM / FIG_H,
                     (FIG_W - left_in - RIGHT_GAP) / FIG_W,
                     1.0 - (AX_BOTTOM + AX_TOP_GAP) / FIG_H])
    # 틱 pad = 축 좌측에서 라벨 시작점까지의 거리(pt)
    ax.tick_params(axis="y", length=0, pad=(left_in - LAB_X) * 72.0)

    # 범례(2행): 폭이 좁아 한 행에 두 항목을 넣을 수 없다.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls=(0, (4, 2.4)), lw=0.9, color="#1f4e79",
               label="Stefan 물리식 기준선"),
        Line2D([], [], marker="o", ms=3.4, mfc="white", mec="#3a4a5a", mew=0.6,
               ls="none", label="지역 내 RMSE"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=1,
               bbox_to_anchor=(LAB_X / FIG_W, 0.035 / FIG_H),
               bbox_transform=fig.transFigure, fontsize=FS_LEGEND,
               frameon=False, handlelength=2.0, handletextpad=0.5,
               labelspacing=0.45, borderpad=0.0, borderaxespad=0.0)

    _report(fig, ax, left_in)
    for ext in ("png", "pdf"):
        path = os.path.join(OUTDIR, f"transfer_loro_summary.{ext}")
        fig.savefig(path, dpi=300, bbox_inches=None)
        print("saved:", path)
    plt.close(fig)


def _label_block(fig, ax):
    """y 틱 라벨 실측 최대폭(inch)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer=rend).width for t in ax.get_yticklabels()]
    return max(widths) / fig.dpi


def _report(fig, ax, left_in):
    """인쇄 검수: 폭·최소 글자 pt·라벨 침범 여부를 표준출력에 남긴다."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_x0 = ax.get_position().x0 * FIG_W
    over = [t.get_text() for t in ax.get_yticklabels()
            if t.get_window_extent(renderer=rend).x1 / fig.dpi > ax_x0]
    sizes = [FS_METHOD, FS_HEADER, FS_VALUE, FS_AXIS, FS_TICK, FS_LEGEND]
    print(f"figsize {FIG_W*25.4:.2f} x {FIG_H*25.4:.2f} mm (축소 배율 1.0)")
    print(f"y 라벨 블록 {left_in*25.4:.2f} mm · 축 좌측 {ax_x0*25.4:.2f} mm · "
          f"막대 영역 침범 {over if over else '없음'}")
    print(f"최소 글자 {min(sizes):.1f} pt (값 라벨·x 틱·범례)")


if __name__ == "__main__":
    draw(load_values())
