"""Fig 5 v3: 새 라벨의 위치(결과 절 R6, 주장 C6).

근거 문서
- 그림 명세: outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 7절(배치, 패널, 설명문 초안), 14절 D-12(능동 선정 지도)
- 양식: design/journal_grade_style_guide.md 2절(그림), 2.5·2.11(관측 원·지도), 6.5절(Fig 5), 부록 A(audit_v3)
- 수치: paper/claims/C6_label_placement/README.md 2절(근거 표)와 그 원천 CSV

구성(명세 7.2: 지도 4열 40 × 40 mm, e 0–105, f 113–170)
- a–d: 캐나다 지역 내 시험(분할 1, 라벨 100개, 첫 추출)의 전략별 선정 결과 지도. 셀 무작위, 블록 층화, 공변량 최대 최소 거리,
  예측 분산 능동 선정. 선정 셀은 실험 하네스의 추출 함수(h40.draw_cells, h40.draw_blocks, h54.kcenter_order, h54.ve_variance 로
  구성한 S6 순차 선정)를 그대로 불러 다시 만들고, WF2 조각의 P1 행(계수, 라벨 블록 수, RMSE)과 대조한다. 라벨 값은 능동 선정의
  잔차 모형 적합(원 실험과 같다)과 대조에만 쓴다. D-12 의 능동 선정 재적합은 작업 지시(Fig 5a–d)에 따라 로컬 CPU 2스레드로 했다.
  후보 1 km 셀 355개는 17개 라벨 블록 안에 모여 있어 40 mm 지도(약 3700 km)에서 한 블록이 0.3 mm 안팎으로 겹친다. 그래서 선정 지점은
  명세 1.4 의 3.0 pt 점 대신 지침 2.5·2.11 의 관측 원(중심 거리 40 km 미만 블록을 한 지점으로 묶고 면적 ∝ 선정 라벨 수, 테두리 1.0 pt,
  채움 검정 alpha 0.35)으로 그린다. 후보 셀은 명세대로 #bdbdbd 2.0 pt 점이다. 지역 경계(블록 합집합 윤곽)는 블록 경도 폭 0.26 mm 가
  지침 2.11 의 1.0 mm 조건에 못 미쳐 그리지 않는다(같은 절의 '점만 그린다' 규칙).
- e: 지역 내 예산별 Δ(전략 − 셀 무작위), 재보정 앵커 + 잔차(λ 0.25). 캐나다, 레나델타, 알래스카. 점 + 두 가중 CI + 띠(명세 1.2).
- f: 전이 조건(라벨 10개, 40개)의 Δ(전략 − 셀 무작위).

자료 처리(prepare)와 그리기(draw)를 나눈다. 실행: OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v3/fig5.py
산출: outputs/figures/paper/v3_restructure/Fig5.{pdf,png}, Fig5_source_data.csv, Fig5_values.txt
(설명문 Fig5_legend.md 는 사람이 쓰고 이 스크립트가 단어 수와 금지 표현만 점검한다)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import style as S  # noqa: E402  (공용 양식, 다른 그림 모듈과 함께 쓴다. 고치지 않는다)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import offset_copy, blended_transform_factory  # noqa: E402

ROOT = S.ROOT
OUT = S.OUT
STEM = "Fig5"

# ---------------------------------------------------------------- 원천(등록 경로)
WF_TESTS = ROOT / "results/rescale_wf/data/processed/wf/wf_tests.csv"          # WF2(지역 내)
WF2B_TESTS = ROOT / "results/rescale_wf2/data/processed/wf/wf2b_tests.csv"     # WF8(전이)
SHARDS = ROOT / "results/rescale_wf/data/processed/wf/shards"                  # WF2 조각(P1 행 대조)
C6 = ROOT / "paper/claims/C6_label_placement"                                  # 근거 묶음(사본 해시, 근거 표)
PROC = ROOT / "data/processed"
H54 = ROOT / "scripts/3_deep_learning/h54_workflow.py"

# ---------------------------------------------------------------- 설계값(명세 7.3)
MAP_TARGET, MAP_MODE, MAP_SPLIT, MAP_N, MAP_DRAW = "Canada", "r", 1, 100, 0     # 첫 추출 = 추출 번호 0
ZONE_KM = 120.0            # 중심 거리 120 km(약 1.3 mm, 지도 1 mm = 93 km) 미만인 라벨 블록은 한 지점으로 합친다(다음 쌍 178 km)
CIRCLE_MM_PER_SQRT = 0.55  # 원 지름(mm) = 0.55 × √(선정 라벨 수). 면적 ∝ 라벨 수(지침 2.3, 2.5)
SIZE_KEY = (1, 10, 50)
CB_THREADS = 2             # 능동 선정 재현의 CatBoost 스레드(작업 지시 OMP_NUM_THREADS=2; 원 실험 4스레드와 결과가 같음을 대조한다)

E_REGIONS = [("Canada|r", "Canada"), ("Lena|r", "Lena Delta"), ("Alaska|r", "Alaska")]
F_ROWS = [("head", "Ten labels"), ("Lena|x", 10), ("Canada|x", 10), ("Russia_W|x", 10), ("Russia_E|x", 10), ("Alaska|x", 10),
          ("head", "Forty labels"), ("Lena|x", 40), ("Canada|x", 40), ("Alaska|x", 40)]
STRAT = {   # 그림 짧은 이름과 선 모양(지침 6.5: 검정은 배치 전략 역할 하나, 방법 색 없음)
    "S1": dict(name="Random", color=S.INK, dashes=None),
    "S2": dict(name="Block stratified", color=S.INK, dashes=None),
    "S4": dict(name="Covariate spread", color=S.INK, dashes=(3.0, 1.5)),
    "S6": dict(name="Variance-based active", color=S.INK_AUX, dashes=(1.2, 1.4)),
}
MAP_STRATS = ["S1", "S2", "S4", "S6"]
E_STRATS = ["S2", "S4", "S6"]
F_STRATS = ["S2", "S4"]
CONTRAST_RE = re.compile(r"^(S\d)-S1\|R1\(0\.25\)\|n(\d+)$")

# ---------------------------------------------------------------- 배치(mm, 그림 왼쪽 위 원점; 명세 7.2)
L = dict(W=170.0, map_w=40.0, map_x=(0.0, 43.333, 86.667, 130.0), map_y=4.0, map_pad=0.04,
         letter2_y=49.0, ax_top=56.0, ax_h=55.0, bottom=10.0,
         e_x0=11.5, e_x1=105.0, e_gap=3.0, e_xlim_canada=(12.0, 330.0), e_xlim=(12.0, 750.0), e_ylim=(-5.2, 3.8),
         f_slot=113.0, f_x0=131.0, f_x1=170.0, f_xlim=(-3.8, 3.8),
         e_dx=1.5, f_dy=0.9, bar_off=S.FOREST_BLOCK_OFFSET_MM, inset_mm=10.0)


# ================================================================ 자료 처리
def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha1_12(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:12]


def check_sources(log: list) -> None:
    """원천 CSV 가 C6 근거 묶음의 사본 해시(MANIFEST)와 같은지 확인한다."""
    man = pd.read_csv(C6 / "tables/MANIFEST.csv")
    for p in (WF_TESTS, WF2B_TESTS):
        rel = str(p.relative_to(ROOT))
        exp = man.loc[man.orig_path == rel, "sha256"]
        got = sha256(p)
        ok = len(exp) == 1 and exp.iloc[0] == got
        log.append(f"[source] {rel} sha256 {got[:16]}… = C6 MANIFEST: {ok}")
        if not ok:
            raise SystemExit(f"원천 해시가 C6 MANIFEST 와 다르다: {rel}")


def budget_table() -> pd.DataFrame:
    """e: WF2-a, R1(λ 0.25) 의 S2·S4·S6 − S1. S6 의 n = 20 은 정의상 0 이라 뺀다(명세 7.3)."""
    w = pd.read_csv(WF_TESTS)
    rows = []
    for tgt, name in E_REGIONS:
        q = w[(w.exp == "wf2") & (w.test_id == "WF2-a") & (w.target == tgt)]
        for _, r in q.iterrows():
            m = CONTRAST_RE.match(str(r.contrast))
            if not m or m.group(1) not in E_STRATS:
                continue
            s, n = m.group(1), int(m.group(2))
            if s == "S6" and n == 20:
                assert r.delta == 0 and r.ci_lo == 0 and r.ci_hi == 0
                continue
            rows.append(dict(panel="e", element="contrast", region=name, mode="within-region", strategy=STRAT[s]["name"], code=s, n=n,
                             delta_cell=r.delta, ci_lo_cell=r.ci_lo, ci_hi_cell=r.ci_hi, delta_block=r.delta_blockeq,
                             ci_lo_block=r.ci_lo_beq, ci_hi_block=r.ci_hi_beq, verdict4=r.verdict4, verdict4_common=r.verdict4_common,
                             ci_dependence=r.ci_dependence, role=r.role, design_note=r.design_note, confirmatory=r.confirmatory,
                             nboot=r.nboot, n_splits=r.n_splits, source=str(WF_TESTS.relative_to(ROOT)),
                             row_filter=f"exp=wf2, test_id=WF2-a, target={tgt}, contrast={r.contrast}"))
    d = pd.DataFrame(rows).sort_values(["region", "code", "n"]).reset_index(drop=True)
    exp_n = {("Canada", "S2"): [20, 50, 100, 200], ("Canada", "S4"): [20, 50, 100, 200], ("Canada", "S6"): [50, 100, 200]}
    for (reg, s), g in d.groupby(["region", "code"]):
        want = exp_n.get((reg, s), [20, 50, 100, 200, 500] if s != "S6" else [50, 100, 200, 500])
        assert sorted(g.n) == want, (reg, s, sorted(g.n))
    return d


def transfer_table() -> pd.DataFrame:
    """f: WF8 서술 행(test_id == 'WF8')의 S2·S4 − S1, R1(λ 0.25), n 10·40. WF8-a·WF8-b 주 행과 값이 같은지 확인한다."""
    w = pd.read_csv(WF2B_TESTS)
    q = w[(w.exp == "wf8") & (w.test_id == "WF8")]
    rows = []
    for _, r in q.iterrows():
        m = CONTRAST_RE.match(str(r.contrast))
        if not m or m.group(1) not in F_STRATS:
            continue
        s, n = m.group(1), int(m.group(2))
        main = w[(w.exp == "wf8") & w.test_id.isin(["WF8-a", "WF8-b"]) & (w.target == r.target) & (w.contrast == r.contrast)]
        for _, mr in main.iterrows():
            for c in ("delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq"):
                assert abs(mr[c] - r[c]) < 1e-12, (r.target, r.contrast, c)
        reg = r.target.split("|")[0]
        rows.append(dict(panel="f", element="contrast", region=S.REGION_NAME[reg], mode="transfer", strategy=STRAT[s]["name"], code=s, n=n,
                         target=r.target, delta_cell=r.delta, ci_lo_cell=r.ci_lo, ci_hi_cell=r.ci_hi, delta_block=r.delta_blockeq,
                         ci_lo_block=r.ci_lo_beq, ci_hi_block=r.ci_hi_beq, verdict4=r.verdict4, verdict4_common=r.verdict4_common,
                         ci_dependence=r.ci_dependence, role=r.role, design_note=r.design_note, confirmatory=r.confirmatory,
                         nboot=r.nboot, n_splits=r.n_splits, main_rows=",".join(sorted(set(main.test_id))),
                         source=str(WF2B_TESTS.relative_to(ROOT)), row_filter=f"exp=wf8, test_id=WF8, target={r.target}, contrast={r.contrast}"))
    e = pd.DataFrame(rows)
    assert len(e) == 16, len(e)
    return e


def _harness():
    """실험 하네스(h54, h40, h42)를 읽기 전용으로 불러온다(파일을 고치지 않는다). h54 가 h40·h42 를 sys.modules 에 등록한다."""
    for p in (str(ROOT / "src"), str(H54.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "h54_workflow" not in sys.modules:
        spec = importlib.util.spec_from_file_location("h54_workflow", H54)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["h54_workflow"] = mod
        spec.loader.exec_module(mod)
    W = sys.modules["h54_workflow"]
    return W, sys.modules["h40_label_grid"], sys.modules["h42_label_grid_ext"]


def _zones(lat, lon, blk, hav):
    """라벨 블록 중심(셀 평균 좌표)을 ZONE_KM 미만 단일 연결로 묶는다. 라벨 값을 쓰지 않는다."""
    g = pd.DataFrame(dict(block=blk, lat=lat, lon=lon)).groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                                                              n_cand=("lat", "size")).reset_index()
    k = len(g)
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    dmin_out = np.inf
    for i in range(k):
        d = hav(g.lat.values[i], g.lon.values[i], g.lat.values, g.lon.values)
        for j in range(i + 1, k):
            if d[j] < ZONE_KM:
                parent[find(i)] = find(j)
            else:
                dmin_out = min(dmin_out, d[j])
    roots = [find(i) for i in range(k)]
    order = {r: z for z, r in enumerate(sorted(set(roots), key=lambda r: (-g.lat.values[r], g.lon.values[r])))}
    g["zone"] = [order[r] for r in roots]
    return g, dmin_out


def _s6_sets(W, H, X, XA, yA, sA, nA, E0, budgets, d, cb_iters):
    """h54 의 S6 순차 능동 선정(s6_sets·s6_var·coefs)을 같은 함수와 seed 로 다시 만든다. 20개 무작위(S1 n 20 추출)에서 시작해
    다음 예산 경계까지 최대 20개씩, 현재 라벨로 다시 구한 E_n 의 잔차 모형(catboost_lo + posterior_sampling)의 가상 앙상블 분산이 큰 셀."""
    def coefs(idx):
        if len(idx) == 0:
            return float(E0)
        E_ls = H.ls_E(yA[idx], sA[idx])
        return float(E0) if not np.isfinite(E_ls) else X.shrink(E_ls, E0, len(idx), W.KAPPA)
    bset = sorted(budgets)
    start = min(W.S6_START, bset[0])
    cur = list(H.draw_cells(MAP_TARGET, MAP_MODE, MAP_SPLIT, start, d, nA))
    out = {start: np.sort(np.array(cur, int))} if start in bset else {}
    n_fit = 0
    while len(cur) < bset[-1]:
        nxt = min(b for b in bset if b > len(cur))
        k = min(W.S6_BATCH, nxt - len(cur))
        ca = np.sort(np.array(cur, int))
        cand = np.setdiff1d(np.arange(nA), ca)
        E_n = coefs(ca)
        v = W.ve_variance(XA[ca], yA[ca] - E_n * sA[ca], XA[cand], W.seed_of("wf-s6", MAP_TARGET, MAP_SPLIT, d, len(cur)), cb_iters, CB_THREADS)
        n_fit += 1
        pick = cand[W.rank_desc(v, W.seed_of("wf-s6t", MAP_TARGET, MAP_SPLIT, d, len(cur)))[:k]]
        cur += [int(v_) for v_ in pick]
        if len(cur) in bset:
            out[len(cur)] = np.sort(np.array(cur, int))
    return out, n_fit


def selection_tables(log: list):
    """a–d: 캐나다 분할 1, 라벨 100개, 첫 추출의 S1·S2·S4·S6 선정 셀을 하네스 함수로 다시 만들고 WF2 조각과 대조한다."""
    W, H, X = _harness()
    from polar.m1_core import load_base, half_split_blocks, eval_mask
    from polar.fidelity import TARGET
    from polar.m1_ext import haversine_km
    df = load_base(PROC)
    df["y"] = df[TARGET].values.astype(float)
    df["s"] = df.e5_sqrt_tdd.values.astype(float)
    units = {s: json.loads((SHARDS / f"wf2__cpu__{MAP_TARGET}__{MAP_MODE}__s{MAP_SPLIT}__{s}_unit.json").read_text()) for s in MAP_STRATS}
    unit = units["S1"]
    dsha = f"{sha1_12(PROC / 'fidelity_base_v3.csv')}:{sha1_12(PROC / 'e5_soil_tdd_v3.csv')}:{sha1_12(PROC / 'lg_subregion_map_v1.csv')}"
    log.append(f"[selection] data sha {dsha} = WF2 unit cfg data_sha {unit['cfg']['data_sha']}: {dsha == unit['cfg']['data_sha']}")
    assert dsha == unit["cfg"]["data_sha"]
    t_idx = np.where(df.macro.values == MAP_TARGET)[0]
    A_idx, B_idx = half_split_blocks(df, t_idx, MAP_SPLIT)
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    XA = df[W.FEATS].values[A_idx].astype(np.float32)
    yA, sA, blkA = df.y.values[A_idx], df.s.values[A_idx], df.block.values[A_idx]
    yB, sB = df.y.values[evB], df.s.values[evB]
    nA = len(A_idx)
    chk = dict(n_A=nA, nb_A=len(np.unique(blkA)), n_eval=len(evB), nb_eval=len(np.unique(df.block.values[evB])), n_cells=len(t_idx))
    for s, u in units.items():
        for k, v in chk.items():
            assert u[k] == v, (s, k, u[k], v)
        assert abs(float(u["E0"]) - float(unit["E0"])) < 1e-12 and u["cfg"]["cb_iters"] == unit["cfg"]["cb_iters"]
    log.append(f"[selection] Canada split {MAP_SPLIT}: candidates {nA} cells in {chk['nb_A']} label blocks, scoring {chk['n_eval']} cells in "
               f"{chk['nb_eval']} blocks, region {chk['n_cells']} cells = unit json (S1, S2, S4, S6)")
    budgets = [b for b in unit["cfg"]["grid"] if 0 < b < nA]
    E0 = float(unit["E0"])
    cb_iters = int(unit["cfg"]["cb_iters"])
    n_draws = int(unit["cfg"]["draws"][0])
    Z = W.standardize(XA, W.standardize_fit(XA))
    sets, n_fit6 = {}, 0
    for d in range(n_draws):
        sets[("S1", d)] = {MAP_N: H.draw_cells(MAP_TARGET, MAP_MODE, MAP_SPLIT, MAP_N, d, nA)}
        sets[("S2", d)] = {MAP_N: H.draw_blocks(MAP_TARGET, MAP_MODE, MAP_SPLIT, MAP_N, d, blkA)}
        sets[("S4", d)] = {MAP_N: np.sort(W.kcenter_order(Z, max(budgets), W.seed_of("wf-s4", MAP_TARGET, MAP_SPLIT, d))[:MAP_N])}
        sets[("S6", d)], nf = _s6_sets(W, H, X, XA, yA, sA, nA, E0, budgets, d, cb_iters)
        n_fit6 += nf
    n_ok, n_all = 0, 0
    for (s, d), byn in sets.items():
        runs = pd.read_csv(SHARDS / f"wf2__cpu__{MAP_TARGET}__{MAP_MODE}__s{MAP_SPLIT}__{s}_runs.csv")
        for n, sel in byn.items():
            rr = runs[(runs.method == "P1") & (runs.n == n) & (runs.draw == d)]
            E1 = X.shrink(H.ls_E(yA[sel], sA[sel]), E0, len(sel), W.KAPPA)
            nb = len(np.unique(blkA[sel]))
            rmse = float(np.sqrt(np.nanmean((E1 * sB - yB) ** 2)))
            ok = (len(rr) == 1 and len(sel) == n and abs(rr.E_used.iloc[0] - E1) < 1e-9 and int(rr.n_blocks_lab.iloc[0]) == nb
                  and abs(rr.rmse_cm.iloc[0] - rmse) < 1e-6)
            n_ok += ok; n_all += 1
            if (d == MAP_DRAW and n == MAP_N) or not ok:
                log.append(f"[selection] {s} draw {d} n {n}: E {E1:.6f} vs {float(rr.E_used.iloc[0]):.6f}, label blocks {nb} vs "
                           f"{int(rr.n_blocks_lab.iloc[0])}, RMSE {rmse:.4f} vs {float(rr.rmse_cm.iloc[0]):.4f} cm -> {'match' if ok else 'MISMATCH'}")
    log.append(f"[selection] {n_ok}/{n_all} reproduced label sets (S1, S2, S4 at n {MAP_N}; S6 at n {budgets}; {n_draws} draws each) match the WF2 "
               f"shard P1 rows (coefficient, label-block count, RMSE). S6 used {n_fit6} CatBoost virtual-ensemble fits "
               f"(iterations {cb_iters}, {CB_THREADS} threads; the original run used {units['S6']['threads']}). "
               f"Label values entered only the S6 residual-model fits (as in the original run) and this check.")
    assert n_ok == n_all
    latA, lonA = df.lat.values[A_idx], df.lon.values[A_idx]
    zones, dmin_out = _zones(latA, lonA, blkA, haversine_km)
    log.append(f"[selection] sites: {zones.zone.nunique()} groups from {len(zones)} label blocks (pooling block centres < {ZONE_KM:.0f} km; "
               f"closest unpooled block pair {dmin_out:.1f} km)")
    zmap = dict(zip(zones.block, zones.zone))
    cells = pd.DataFrame(dict(loc_id=df.loc_id.values[A_idx], lat=latA, lon=lonA, block=blkA, zone=[zmap[b] for b in blkA]))
    cell_rows, zone_rows = [], []
    zc = cells.groupby("zone").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_cand=("lat", "size"), n_blocks=("block", "nunique"))
    for s in MAP_STRATS:
        sel = sets[(s, MAP_DRAW)][MAP_N]
        flag = np.zeros(nA, int); flag[sel] = 1
        for i in range(nA):
            cell_rows.append(dict(panel="a-d", element="candidate_cell", region="Canada", mode="within-region", strategy=STRAT[s]["name"],
                                  code=s, n=MAP_N, split=MAP_SPLIT, draw=MAP_DRAW + 1, loc_id=int(cells.loc_id[i]), block=int(cells.block[i]),
                                  zone=int(cells.zone[i]), lat=cells.lat[i], lon=cells.lon[i], selected=int(flag[i])))
        cnt = cells.assign(sel=flag).groupby("zone").sel.sum()
        for z, r in zc.iterrows():
            zone_rows.append(dict(panel="a-d", element="site_circle", region="Canada", mode="within-region", strategy=STRAT[s]["name"], code=s,
                                  n=MAP_N, split=MAP_SPLIT, draw=MAP_DRAW + 1, zone=int(z), lat=r.lat, lon=r.lon, n_candidates=int(r.n_cand),
                                  n_blocks=int(r.n_blocks), n_selected=int(cnt.get(z, 0))))
        log.append(f"[selection] {STRAT[s]['name']}: {len(sel)} labels in {len(np.unique(blkA[sel]))} of {chk['nb_A']} label blocks, "
                   f"{int((cnt > 0).sum())} of {len(zc)} sites; per site {dict((int(k), int(v)) for k, v in cnt.items())}")
    extent_cells = pd.DataFrame(dict(lat=df.lat.values[t_idx], lon=df.lon.values[t_idx]))
    return pd.DataFrame(cell_rows), pd.DataFrame(zone_rows), extent_cells


def readme_crosscheck(d: pd.DataFrame, e: pd.DataFrame, log: list) -> None:
    """그림 값(원천 CSV)을 C6 README 2절 근거 표(소수 둘째 자리)와 대조한다(F-13)."""
    txt = (C6 / "README.md").read_text()
    rx = re.compile(r"^\| ([A-G]\d+) \| (W2?) \| .*?target=([\w\-]+)\\\|(\w), contrast=`(S\d)-S1\\\|R1\(0\.25\)\\\|n(\d+)`.*?\| "
                    r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\] \| (-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", re.M)
    plotted = {}
    for _, r in d.iterrows():
        tgt = dict((v, k) for k, v in E_REGIONS)[r.region].split("|")[0]
        plotted[(tgt, "r", r.code, int(r.n))] = r
    for _, r in e.iterrows():
        plotted[(r.target.split("|")[0], "x", r.code, int(r.n))] = r
    n_cmp, bad, seen = 0, [], set()
    for m in rx.finditer(txt):
        rid, _, tgt, mode, s, n = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), int(m.group(6))
        key = (tgt, mode, s, n)
        if key not in plotted or key in seen:
            continue
        seen.add(key)
        r = plotted[key]
        vals = [float(v) for v in m.groups()[6:]]
        mine = [r.delta_cell, r.ci_lo_cell, r.ci_hi_cell, r.delta_block, r.ci_lo_block, r.ci_hi_block]
        n_cmp += 1
        if any(abs(round(a, 2) - b) > 0.0051 for a, b in zip(mine, vals)):
            bad.append(rid)
    log.append(f"[F-13] {n_cmp} plotted contrasts found in C6 README section 2 tables; mismatches: {bad if bad else 'none'}; "
               f"plotted contrasts not tabulated in README: {len(plotted) - len(seen)} (taken directly from the source CSV)")
    assert not bad


def prepare(log: list | None = None):
    log = [] if log is None else log
    check_sources(log)
    d, e = budget_table(), transfer_table()
    readme_crosscheck(d, e, log)
    cells, zones, extent_cells = selection_tables(log)
    return dict(e=d, f=e, cells=cells, zones=zones, extent_cells=extent_cells, log=log)


# ================================================================ 그리기
def _mm_tr(ax, fig, dx_mm=0.0, dy_mm=0.0):
    return offset_copy(ax.transData, fig=fig, x=dx_mm / 25.4, y=dy_mm / 25.4, units="inches")


def _line(ax, x, y, tr, color, lw, dashes=None, cap="round", z=3, **kw):
    ln, = ax.plot(x, y, color=color, lw=lw, transform=tr, solid_capstyle=cap, dash_capstyle="butt", zorder=z, **kw)
    if dashes:
        ln.set_dashes(dashes)
    return ln


def _ref(ax, orient):
    """±0.5 cm 동등 띠와 0선(기준 = 셀 무작위)."""
    if orient == "h":
        ax.axhspan(-S.EQUIV_HALF_WIDTH_CM, S.EQUIV_HALF_WIDTH_CM, color=S.EQUIV_BAND, lw=0, zorder=0)
        ax.axhline(0, color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], zorder=1)
    else:
        ax.axvspan(-S.EQUIV_HALF_WIDTH_CM, S.EQUIV_HALF_WIDTH_CM, color=S.EQUIV_BAND, lw=0, zorder=0)
        ax.axvline(0, color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], zorder=1)


def _ci_pair(ax, fig, pos, r, color, dashes, off_mm, orient):
    """점(셀 가중 점추정) + 셀 가중 CI(2.0 pt) + 블록 등가중 CI(1.0 pt, 0.8 mm 오른쪽 또는 아래)."""
    lw_c, lw_b = S.LW["ci_forest_cell"], S.LW["ci_forest_block"]
    if orient == "v":   # 세로 막대(e): pos = n, 값은 y
        t1, t2 = _mm_tr(ax, fig, off_mm), _mm_tr(ax, fig, off_mm + L["bar_off"])
        _line(ax, [pos, pos], [r.ci_lo_cell, r.ci_hi_cell], t1, color, lw_c, dashes, z=3)
        _line(ax, [pos, pos], [r.ci_lo_block, r.ci_hi_block], t2, color, lw_b, dashes, z=3)
        ax.plot([pos], [r.delta_cell], "o", ms=S.MS["main"], color=color, mew=0, transform=t1, zorder=4)
    else:               # 가로 막대(f): pos = 행, 값은 x
        t1, t2 = _mm_tr(ax, fig, 0, off_mm), _mm_tr(ax, fig, 0, off_mm - L["bar_off"])
        _line(ax, [r.ci_lo_cell, r.ci_hi_cell], [pos, pos], t1, color, lw_c, dashes, z=3)
        _line(ax, [r.ci_lo_block, r.ci_hi_block], [pos, pos], t2, color, lw_b, dashes, z=3)
        ax.plot([r.delta_cell], [pos], "o", ms=S.MS["main"], color=color, mew=0, transform=t1, zorder=4)


def _ftext(fig, x_mm, y_mm, s, **kw):
    W, H = fig.get_size_inches() * 25.4
    return fig.text(x_mm / W, 1 - y_mm / H, s, fontsize=S.FONT_PT, **kw)


def _map_geoms():
    import cartopy.io.shapereader as shpreader
    land = list(shpreader.Reader(shpreader.natural_earth("50m", "physical", "land")).geometries())
    lakes = list(shpreader.Reader(shpreader.natural_earth("50m", "physical", "lakes")).geometries())
    land110 = list(shpreader.Reader(shpreader.natural_earth("110m", "physical", "land")).geometries())
    return land, lakes, land110


def _circle_pts(n):
    """선정 라벨 n 개의 원 지름(pt). 면적 ∝ n."""
    return CIRCLE_MM_PER_SQRT * np.sqrt(np.asarray(n, float)) / 25.4 * 72


def draw_maps(fig, T, proj, ext, geoms):
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    land, lakes, _ = geoms
    cells, zones = T["cells"], T["zones"]
    axes = []
    mm_to_m = (ext[1] - ext[0]) / L["map_w"]
    for j, s in enumerate(MAP_STRATS):
        x0 = L["map_x"][j]
        ax = S.axes_mm(fig, x0, L["map_y"], L["map_w"], L["map_w"], projection=proj)
        ax.set_extent(ext, crs=proj)
        ax.spines["geo"].set_visible(False)
        ax.add_geometries(land, pc, facecolor=S.BASEMAP["land"], edgecolor="none", zorder=0).set_rasterized(True)       # 밀집 지도 층: 600 dpi 래스터
        ax.add_geometries(lakes, pc, facecolor=S.BASEMAP["sea"], edgecolor="none", zorder=0.1).set_rasterized(True)
        lon_s = np.linspace(-180, 0, 361)
        for la in (60, 70, 80):        # 경위선: 위도 3개, 경도 5개(지침 2.3 상한)
            ax.plot(lon_s, np.full_like(lon_s, la), color=S.BASEMAP["graticule"], lw=S.LW["grid_map"], transform=pc, zorder=0.5)
        lat_s = np.linspace(40, 89.5, 200)
        for lo in (-140, -120, -100, -80, -60):
            ax.plot(np.full_like(lat_s, lo), lat_s, color=S.BASEMAP["graticule"], lw=S.LW["grid_map"], transform=pc, zorder=0.5)
        c = cells[cells.code == s]
        ax.plot(c.lon.values, c.lat.values, "o", ms=2.0, color=S.BASEMAP["candidate"], mew=0, transform=pc, zorder=2)[0].set_rasterized(True)
        z = zones[(zones.code == s) & (zones.n_selected > 0)].sort_values("n_selected", ascending=False)
        ax.scatter(z.lon.values, z.lat.values, s=_circle_pts(z.n_selected.values) ** 2, facecolor=(0, 0, 0, 0.35), edgecolor=S.INK,
                   linewidth=S.LW["marker_edge_open"], transform=pc, zorder=3)
        _ftext(fig, x0 + L["map_w"] / 2, L["map_y"] - 0.7, STRAT[s]["name"], ha="center", va="bottom").set_gid("category")
        S.panel_letter(fig, x0, 0.0, "abcd"[j])
        axes.append(ax)
    return axes, mm_to_m


def _edge_crossing(proj, lat, ext, side):
    """위도선이 지도 왼쪽(또는 오른쪽) 가장자리와 만나는 투영 y. 없으면 None."""
    import cartopy.crs as ccrs
    x0, x1, y0, y1 = ext
    lo = np.linspace(-180, 0, 20001)
    p = proj.transform_points(ccrs.PlateCarree(), lo, np.full_like(lo, float(lat)))
    xe = x0 if side == "left" else x1
    sgn = np.sign(p[:, 0] - xe)
    for j in np.where(np.diff(sgn) != 0)[0]:
        y = p[j, 1]
        if y0 < y < y1:
            return float(y)
    return None


def map_furniture(fig, ax, proj, ext, mm_to_m, geoms):
    """a 에만: 축척 막대(70° N 에서 참), 위도 라벨, 위치 삽도, 후보 셀·원 크기 열쇠(2줄). 반환: 기록용 정보."""
    x0, x1, y0, y1 = ext
    info = {}
    # 축척 막대 500 km(투영 단위 = 70° N 에서 m), 왼쪽 아래
    bx, by = x0 + 2.5 * mm_to_m, y0 + 2.8 * mm_to_m
    ax.plot([bx, bx + 500e3], [by, by], color=S.INK, lw=S.LW["scale_bar"], solid_capstyle="butt", transform=proj, zorder=5)
    ax.text(bx + 250e3, by + 0.8 * mm_to_m, "500 km", ha="center", va="bottom", fontsize=S.FONT_PT, transform=proj, zorder=5).set_gid("scale")
    info["scale_bar_mm"] = 500e3 / mm_to_m
    # 위도 라벨: 위도선이 지도 가장자리를 지나는 곳(자료 없음). 70° N 은 왼쪽 위, 60° N 은 오른쪽(래브라도); 80° N 은 가장자리를 지나지 않는다
    info["lat_labels"] = []
    for la, side in ((70, "left"), (60, "right")):
        y = _edge_crossing(proj, la, ext, side)
        if y is not None:
            xx = x0 + 0.8 * mm_to_m if side == "left" else x1 - 0.8 * mm_to_m
            ax.text(xx, y + 0.6 * mm_to_m, f"{la}° N", ha="left" if side == "left" else "right", va="bottom", fontsize=S.FONT_PT,
                    transform=proj, zorder=5).set_gid("scale")
            info["lat_labels"].append(la)
    # 위치 삽도(범북극 윤곽과 대상 사각형), 오른쪽 위
    iw = L["inset_mm"]
    axi = S.axes_mm(fig, L["map_x"][0] + L["map_w"] - iw - 0.5, L["map_y"] + 0.5, iw, iw, projection=proj)
    axi.set_extent([-4.7e6, 4.7e6, -4.7e6, 4.7e6], crs=proj)
    import cartopy.crs as ccrs
    axi.add_geometries(geoms[2], crs=ccrs.PlateCarree(), facecolor="#d9d9d9", edgecolor="none", linewidth=0, zorder=0.4).set_rasterized(True)
    axi.set_facecolor("white")
    axi.spines["geo"].set_linewidth(1.0); axi.spines["geo"].set_edgecolor(S.INK_AUX)
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=axi.transData, facecolor="none", edgecolor=S.INK, lw=1.0, zorder=3)
    rect.set_gid("allowed_frame")
    axi.add_patch(rect)
    # 열쇠(오른쪽 아래): 1줄 후보 셀, 2줄 원 크기(값은 크기 열쇠 값, 명세 1.6)
    kx = x1 - 19.5 * mm_to_m
    ky1, ky2 = y0 + 9.0 * mm_to_m, y0 + 2.8 * mm_to_m
    ax.plot([kx + 0.5 * mm_to_m], [ky1], "o", ms=2.0, color=S.BASEMAP["candidate"], mew=0, transform=proj, zorder=5)
    ax.text(kx + 1.6 * mm_to_m, ky1, "Candidate cells", ha="left", va="center", fontsize=S.FONT_PT, transform=proj, zorder=5)
    cx = kx
    for v in SIZE_KEY:
        dmm = CIRCLE_MM_PER_SQRT * np.sqrt(v)
        cx += max(dmm, 1.0) / 2 * mm_to_m
        cy = ky2 + dmm / 2 * mm_to_m          # 아래끝 정렬
        ax.scatter([cx], [cy], s=_circle_pts(v) ** 2, facecolor=(0, 0, 0, 0.35), edgecolor=S.INK, linewidth=S.LW["marker_edge_open"],
                   transform=proj, zorder=5)
        cx += max(dmm, 1.0) / 2 * mm_to_m + 0.5 * mm_to_m
        t = ax.text(cx, ky2, str(v), ha="left", va="bottom", fontsize=S.FONT_PT, transform=proj, zorder=5)
        t.set_gid("sizekey")
        cx += (1.0 + 1.25 * len(str(v))) * mm_to_m
    return info


def _logx(ax, xlim, ticks):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([S.fmt_int(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())


def draw_budget(fig, T, top):
    d = T["e"]
    dec = [np.log10(L["e_xlim_canada"][1] / L["e_xlim_canada"][0])] + [np.log10(L["e_xlim"][1] / L["e_xlim"][0])] * 2
    mm_per_dec = (L["e_x1"] - L["e_x0"] - 2 * L["e_gap"]) / sum(dec)
    x, axes = L["e_x0"], []
    shift = -(L["bar_off"] + 0.18 - 0.62) / 2       # 묶음(점 왼쪽 끝부터 가는 막대 오른쪽 끝까지)의 가운데를 n 에 둔다
    offs = {s: (k - 1) * L["e_dx"] + shift for k, s in enumerate(E_STRATS)}
    for k, (tgt, name) in enumerate(E_REGIONS):
        w = dec[k] * mm_per_dec
        ax = S.axes_mm(fig, x, top, w, L["ax_h"])
        xl = L["e_xlim_canada"] if k == 0 else L["e_xlim"]
        _logx(ax, xl, [20, 50, 100, 200] if k == 0 else [20, 50, 100, 200, 500])
        ax.set_ylim(*L["e_ylim"])
        ax.yaxis.set_major_locator(FixedLocator([-4, -2, 0, 2]))
        ax.yaxis.set_major_formatter(FixedFormatter([S.fmt_num(v, 0) for v in (-4, -2, 0, 2)]))
        ax.yaxis.set_minor_locator(NullLocator())
        if k > 0:
            ax.tick_params(axis="y", labelleft=False)
        _ref(ax, "h")
        q = d[d.region == name]
        for s in E_STRATS:
            g = q[q.code == s].sort_values("n")
            st = STRAT[s]
            _line(ax, g.n.values, g.delta_cell.values, _mm_tr(ax, fig, offs[s]), st["color"], S.LW["aux"], st["dashes"], cap="butt", z=2)
            for _, r in g.iterrows():
                _ci_pair(ax, fig, r.n, r, st["color"], None, offs[s], "v")
        ax.text(0.5, 1.0, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=S.FONT_PT).set_gid("category")
        if k == 0:
            ax.set_ylabel("Error change vs random cells (cm)", labelpad=2.0)
        if k == 1:
            ax.set_xlabel("Labels, $n$", labelpad=1.5)
        axes.append(ax)
        x += w + L["e_gap"]
    return axes, offs


def strategy_key(fig, ax):
    """레나델타 하위 축의 빈 아래쪽(자료 없음, y < −1.6)에 전략 3개의 선 견본과 이름. 위아래 순서 = 지도 열 순서 b, c, d.
    (명세 7.3 은 첫 하위 축 선 끝 직접 라벨이나, 캐나다의 세 선 끝이 0.75 cm 안에 모여 라벨이 겹치므로 이 자리에 둔다.)"""
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for i, s in enumerate(E_STRATS):
        y = -2.55 - 0.68 * i
        st = STRAT[s]
        x0 = 0.05
        _line(ax, [x0, x0 + 0.14], [y, y], tr, st["color"], S.LW["aux"], st["dashes"], cap="butt", z=5)
        ax.plot([x0 + 0.07], [y], "o", ms=S.MS["main"], color=st["color"], mew=0, transform=tr, zorder=6)
        ax.text(x0 + 0.18, y, st["name"], transform=tr, ha="left", va="center", fontsize=S.FONT_PT, zorder=6)


def ci_key(fig, ax):
    """알래스카 하위 축의 빈 아래쪽(자료 없음, y < −2.5)에 두 가중 막대와 동등 띠의 열쇠(그림당 한 번, 명세 1.2)."""
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    W, H = fig.get_size_inches() * 25.4
    bb = ax.get_position()
    ax_w_mm, ax_h_mm = bb.width * W, bb.height * H
    cm_per_mm = (L["e_ylim"][1] - L["e_ylim"][0]) / ax_h_mm
    xk = 0.10
    half = 1.2 * cm_per_mm
    rows = [(-3.25, "cell"), (-3.9, "block"), (-4.55, "band")]
    for y, kind in rows:
        if kind == "cell":
            _line(ax, [xk, xk], [y - half, y + half], tr, S.INK, S.LW["ci_forest_cell"], None, cap="round", z=5)
            lab = "Cell-weighted"
        elif kind == "block":
            _line(ax, [xk, xk], [y - half, y + half], tr, S.INK, S.LW["ci_forest_block"], None, cap="round", z=5)
            lab = "Block-equal"
        else:
            wfrac = 2.4 / ax_w_mm
            ax.add_patch(Rectangle((xk - wfrac / 2, y - half), wfrac, 2 * half, transform=tr, facecolor=S.EQUIV_BAND, edgecolor="none",
                                   zorder=5))
            lab = "±0.5 cm"
        t = ax.text(xk + 2.2 / ax_w_mm, y, lab, transform=tr, ha="left", va="center", fontsize=S.FONT_PT, zorder=6)
        if kind == "band":
            t.set_gid("sizekey")    # 열쇠 값(명세 1.2 의 '±0.5 cm' 견본). 관찰 수치가 아니다


def direction_marker(fig, ax):
    """방향 표지 그림당 1개(R-11): e 의 y 축 바깥 왼쪽 열, 아래쪽 직선 화살표."""
    W, H = fig.get_size_inches() * 25.4
    bb = ax.get_position()
    bot = (1 - bb.y0) * H
    xm = 1.2
    _ftext(fig, xm, bot - 21.0, "Lower error", rotation=90, ha="center", va="center", color=S.INK_AUX)
    a = FancyArrowPatch((xm / W, 1 - (bot - 13.6) / H), (xm / W, 1 - (bot - 7.0) / H), transform=fig.transFigure, arrowstyle="-|>",
                        mutation_scale=5.5, lw=1.0, color=S.INK_AUX, shrinkA=0, shrinkB=0)
    fig.add_artist(a)


def draw_transfer(fig, T, top):
    e = T["f"]
    ax = S.axes_mm(fig, L["f_x0"], top, L["f_x1"] - L["f_x0"], L["ax_h"])
    ax.set_xlim(*L["f_xlim"])
    ax.xaxis.set_major_locator(FixedLocator([-2, 0, 2]))
    ax.xaxis.set_major_formatter(FixedFormatter([S.fmt_num(v, 0) for v in (-2, 0, 2)]))
    ax.xaxis.set_minor_locator(NullLocator())
    nrow = len(F_ROWS)
    ax.set_ylim(nrow - 0.5, -0.5)
    ticks, labels = [], []
    _ref(ax, "v")
    heads = []
    for i, (key, val) in enumerate(F_ROWS):
        if key == "head":
            heads.append((i, val))
            continue
        ticks.append(i); labels.append(S.REGION_NAME[key.split("|")[0]])
        for k, s in enumerate(F_STRATS):
            r = e[(e.target == key) & (e.n == val) & (e.code == s)].iloc[0]
            off = (L["f_dy"] if k == 0 else -L["f_dy"]) + (L["bar_off"] + 0.18 - 0.62) / 2
            _ci_pair(ax, fig, i, r, STRAT[s]["color"], STRAT[s]["dashes"], off, "h")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.spines["left"].set_visible(False)
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for i, h in heads:
        ax.text(L["f_slot"] / L["W"], i, h, transform=tr, ha="left", va="center", fontsize=S.FONT_PT).set_gid("category")
    ax.set_xlabel("Error change vs random cells (cm)", labelpad=1.5)
    return ax


def draw(T, medium="paper"):
    """medium: 'paper'(170 mm, 7 pt). 'slide' 는 같은 자료·색·범위로 글자만 슬라이드 크기로 바꾼 시험판이다(배치 검토 전)."""
    S.use_v3(medium)
    matplotlib.rcParams["lines.scale_dashes"] = False      # 선 모양 길이를 pt 로 고정(선 굵기와 무관)
    import cartopy.crs as ccrs
    ec = T["extent_cells"]
    lon0 = float(np.round((ec.lon.min() + ec.lon.max()) / 2))
    proj = ccrs.Stereographic(central_latitude=90, central_longitude=lon0, true_scale_latitude=70)
    P = proj.transform_points(ccrs.PlateCarree(), ec.lon.values, ec.lat.values)
    x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    side = max((x1 - x0), (y1 - y0)) * (1 + 2 * L["map_pad"])      # 정사각 범위(40 × 40 mm, 명세 7.2)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ext = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)
    fig_h = L["ax_top"] + L["ax_h"] + L["bottom"]
    fig = S.fig_mm(L["W"], fig_h)
    if medium == "slide":
        fig.set_size_inches(12.0, 12.0 * fig_h / L["W"])
    geoms = _map_geoms()
    maxes, mm_to_m = draw_maps(fig, T, proj, ext, geoms)
    minfo = map_furniture(fig, maxes[0], proj, ext, mm_to_m, geoms)
    eaxes, offs = draw_budget(fig, T, L["ax_top"])
    strategy_key(fig, eaxes[1])
    ci_key(fig, eaxes[2])
    direction_marker(fig, eaxes[0])
    fax = draw_transfer(fig, T, L["ax_top"])
    S.panel_letter(fig, 0.0, L["letter2_y"], "e")
    S.panel_letter(fig, L["f_slot"], L["letter2_y"], "f")
    meta = dict(fig_w=L["W"], fig_h=round(fig_h, 2), map_w=L["map_w"], lon0=lon0, extent_m=[round(v) for v in ext],
                km_per_mm=round(mm_to_m / 1e3, 2), scale_bar_mm=round(minfo["scale_bar_mm"], 2), lat_labels=minfo["lat_labels"],
                e_offsets_mm={k: round(v, 3) for k, v in offs.items()}, ax_top=L["ax_top"])
    return fig, dict(maps=maxes, e=eaxes, f=fax), meta


# ================================================================ 내보내기와 점검
def panel_areas(fig, axs):
    W, H = fig.get_size_inches() * 25.4
    out = {}
    for k, a in zip("abcd", axs["maps"]):
        bb = a.get_position(); out[k] = bb.width * W * bb.height * H
    bb0, bb1 = axs["e"][0].get_position(), axs["e"][-1].get_position()
    out["e"] = (bb1.x1 - bb0.x0) * W * bb0.height * H
    bb = axs["f"].get_position(); out["f"] = (bb.x1 - L["f_slot"] / W) * W * bb.height * H
    return out


def legend_check(log: list) -> None:
    p = OUT / f"{STEM}_legend.md"
    if not p.exists():
        log.append("[legend] Fig5_legend.md 없음")
        return
    body = "\n".join(ln for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#") and not ln.startswith("<!--"))
    body = re.sub(r"\*\*", "", body)
    words = len(body.split())
    ta = S.text_audit(body)
    colour = re.findall(r"\b(grey|gray|black|white|blue|red|green|open|filled|thick|thin|dashed|dotted|solid)\b", body, re.I)
    first = body.split(". ")[0]
    log.append(f"[legend] words {words} (limit 350); first sentence words {len(first.split())}; dash {ta['dash']}; banned {ta['banned']}; "
               f"middot-number {ta['middot_num']}; comma4 {ta['comma4']}; internal codes {ta['codes']}; colour/shape words {colour}; "
               f"'This figure' start: {body.lstrip().lower().startswith('this figure')}")
    for must in ("Natural Earth", "Cartopy", "stereographic", "10,000"):
        log.append(f"[legend] contains '{must}': {must in body}")


def plotted_values(T, log: list) -> None:
    """그림의 모든 수치(e·f 의 점추정과 두 CI, a–d 의 지점별 선정 수)를 원천 경로·행 필터와 함께 적는다."""
    log.append("[plotted] panel e, f: delta_cell [ci_lo_cell, ci_hi_cell] / delta_block [ci_lo_block, ci_hi_block] (cm) | source | row filter")
    for key in ("e", "f"):
        for _, r in T[key].sort_values(["region", "code", "n"]).iterrows():
            log.append(f"  {key} {r.region:12s} {r.strategy:22s} n {int(r.n):>3}: {r.delta_cell:+.4f} [{r.ci_lo_cell:+.4f}, {r.ci_hi_cell:+.4f}] / "
                       f"{r.delta_block:+.4f} [{r.ci_lo_block:+.4f}, {r.ci_hi_block:+.4f}] | {r.source} | {r.row_filter}")
    log.append("[plotted] panels a-d: labels per site (circle area) | source: harness reproduction (see [selection]); site = label blocks pooled "
               f"< {ZONE_KM:.0f} km; filter: Canada, within-region, split {MAP_SPLIT}, n {MAP_N}, draw {MAP_DRAW + 1}")
    z = T["zones"]
    for s in MAP_STRATS:
        q = z[(z.code == s)].sort_values("zone")
        log.append(f"  {STRAT[s]['name']:22s}: " + ", ".join(f"site {int(r.zone)} ({r.lat:.2f}° N, {abs(r.lon):.2f}° W; {int(r.n_candidates)} cand.): "
                                                        f"{int(r.n_selected)}" for _, r in q.iterrows()))


def main():
    log = ["Fig 5 v3 value and QA log (generated by scripts/4_visualization/paper_v3/fig5.py)", ""]
    T = prepare(log)
    fig, axs, meta = draw(T, "paper")
    au = S.audit_v3(fig)
    log.append(f"[layout] {json.dumps(meta)}")
    area = panel_areas(fig, axs)
    tot = meta["fig_w"] * meta["fig_h"]
    main_share = sum(area[k] for k in "abcde") / tot
    log.append(f"[F-17] panel areas mm2 {dict((k, round(v)) for k, v in area.items())}; main panels a-e {main_share:.0%} of figure; "
               f"largest panel {max(area, key=area.get)}")
    log.append(f"[audit_v3] fails={au['fails']} sizes={sorted(au['sizes'])} chars={au['chars']} thin={au['thin_lines']} "
               f"titles={au['titles']} boxed={au['boxed_text']} codes={au['codes']} comma4={au['comma4']} long_labels={au['long_labels']} "
               f"loose_numbers={au['loose_numbers']}")
    paths = S.save_fig(fig, STEM, OUT, formats=("pdf", "png"))
    pa = S.pdf_audit(paths[0])
    log.append(f"[pdf_audit] {json.dumps(pa, ensure_ascii=False)}")
    import subprocess
    fonts_txt = subprocess.run(["pdffonts", str(paths[0])], capture_output=True, text=True).stdout
    log.append("[pdffonts]\n" + "\n".join("  " + ln for ln in fonts_txt.strip().splitlines()))
    from PIL import Image
    with Image.open(paths[1]) as im:
        log.append(f"[png] size_px {im.size}, dpi {tuple(round(v) for v in im.info.get('dpi', (0, 0)))}")
    src = pd.concat([T["cells"], T["zones"], T["e"], T["f"]], ignore_index=True, sort=False)
    cols = ["panel", "element", "region", "mode", "strategy", "code", "n", "split", "draw", "loc_id", "block", "zone", "lat", "lon", "selected",
            "n_candidates", "n_blocks", "n_selected", "target", "delta_cell", "ci_lo_cell", "ci_hi_cell", "delta_block", "ci_lo_block",
            "ci_hi_block", "verdict4", "verdict4_common", "ci_dependence", "role", "design_note", "confirmatory", "nboot", "n_splits",
            "main_rows", "source", "row_filter"]
    src = src[[c for c in cols if c in src.columns]]
    src.to_csv(OUT / f"{STEM}_source_data.csv", index=False)
    log.append(f"[source_data] {len(src)} rows -> {STEM}_source_data.csv")
    dep = pd.concat([T["e"], T["f"]])
    dep = dep[dep.ci_dependence.notna()]
    for _, r in dep.iterrows():
        log.append(f"[ci_dependence] panel {r.panel} {r.region} {r.strategy} n {int(r.n)}: verdict {r.verdict4} -> common-resampling "
                   f"{r.verdict4_common} ({r.ci_dependence})")
    plotted_values(T, log)
    legend_check(log)
    (OUT / f"{STEM}_values.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return au


if __name__ == "__main__":
    main()
