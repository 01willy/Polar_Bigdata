"""LGD 적격 표(계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.4, 6B.6 단계 4, 개정 13). 학습은 하지 않는다.

대상마다 실행 표를 만든다(래퍼 설계 (가): v4 의 v3 행과 그 대상의 새 행. 새 행의 source_id 를 F4_direct 로 바꾼다).
그 표를 m1_core.load_base(파일 인자만 바꿈)로 읽고 h40 의 규칙(Data.source_idx, Data.split_structure)을 그대로 따라
  대상 셀 수, 0.5° 블록 수, 채점 가능 셀(eval_mask), 100 km 버퍼 뒤 원천 셀 수, 분할 1–5 의 구조(중복, 채점 블록 수, 유효),
  사용 분할의 채점 블록 합집합
을 계산한다. 적격 = 유효 분할 ≥ 1 이고 채점 블록 합집합 ≥ 8(MIN_BLOCKS_CI). 레짐 = 대상 셀 ALT 평균 150 cm 기준.
v3 참조 행(v3 표 그대로)은 h40 --count-only 산출(data/processed/lg/lg_count_cpu.csv)과 대조해 구현을 확인한다.

개정 13(검증 지적 반영)
  - 실행 표는 v4 의 원문 줄을 골라 옮기고 대상 새 행의 source_id 칸만 바꾼다(write_run_table_text). pandas 로 다시 쓰면
    v3 행의 부동소수 값이 1 ulp 수준으로 달라지기 때문이다. v3 행만 고르면 v3 원문 줄과 같은지 확인한다(region 을 바꾼 3줄 제외).
  - 새 행은 loc_id(>= 20000), source_id, labels 의 lgd_role 로 고르고 assert 한다(region 이름으로 고르지 않는다).
  - Russia_C 의 모든 실행 표에서 v3 loc 17557(레나 델타 상자 안)을 뺀다(6B.4).
  - 다른 지리 macro 의 v3 F4_direct 까지 최소 거리를 대상의 v3 구성 셀까지 포함해 계산한다(min_dist_other_macro_all_km).
  - 분할별 채점 블록 수와 최대 블록 셀 비중(max_block_share), 사용 분할의 최소 채점 블록 수와 최대 비중, 소수 블록 표시
    (최소 채점 블록 < 5), 대상 셀의 위치 간 거리 단일 연결 묶음 수(0.25, 0.5, 1 km), 자료원별 셀·블록 구성을 적는다.
  - 변형: L41 (d) year_max >= 2010 셀만(WRAPUP 7.3 (a)9 채택), (e) 절단 영향 셀 제외, (f) 절단 하한 대입, (g) 빙하 격자 셀 제외,
    (h) 최대 단일 조사 제외(점 추정만, WRAPUP (a)9), L40 Russia_E 확충판의 Kytalyk 제외. 셀을 빼는 변형에서 v3 구성 셀은 (g) 에서만
    뺀다(빙하 격자 표지가 v3 행에도 있다).
  - load_base 의 물리 앙상블 결측 대체 중앙값이 실행 표 전체로 계산되므로 v3 행의 p4_ku, p2_edaphic 가 v3 표와 달라질 수 있다.
    그 최대 차를 적는다(max_abs_diff_p4_ku_v3rows).

산출: data/processed/lgd_eligibility_v1.csv, lgd_eligibility_v1_splits.csv, lgd_eligibility_v1_meta.json
실행(ROOT): OMP_NUM_THREADS=1 python3 scripts/1_data_prep/lgd_eligibility_v1.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EXT = PROC / "ext_labels"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "1_data_prep"))
from polar.m1_core import load_base, eval_mask, half_split_blocks  # noqa: E402
from polar.m1_ext import haversine_km  # noqa: E402
from polar.m1_stats import MIN_BLOCKS_CI  # noqa: E402
from build_ext_cells_v1 import geo_macro_v3  # noqa: E402

SPLITS = [1, 2, 3, 4, 5]
BUFFER_KM = 100.0
DEEP_CM = 150.0
FEW_BLOCKS = 5
LOC0 = 20000
DROP_V3 = {"Russia_C": [17557]}               # 개정 13: 6B.4 레나 델타 영역 제외
GEO_OF_TARGET = {"Tibet_LGD": "Tibet"}
LARGEST_SURVEY = {"NAtlantic": "grl_ilulissat_scheer", "Russia_C": "ru_mamontovklyk_grosse"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_run_table_text(v4_path: Path, out: Path, keep_new, drop_v3=(), alt_override=None, expect_roles=None, lab=None):
    """v4 원문 줄에서 실행 표를 만든다. v3 행(loc_id < 20000)은 drop_v3 를 빼고 그대로, 새 행은 keep_new 만 남기고
    source_id 칸을 F4_direct 로 바꾼다. alt_override(loc_id → 값)는 새 행의 alt_cm 칸만 바꾼다(L41 e)."""
    keep_new, drop_v3 = set(int(x) for x in keep_new), set(int(x) for x in drop_v3)
    if lab is not None and expect_roles is not None:
        r = lab.set_index("loc_id").loc[sorted(keep_new)]
        assert (r.index >= LOC0).all(), "새 행 선택에 v3 loc_id 가 섞였다"
        assert r.lgd_role.isin(expect_roles).all(), f"lgd_role 이 기대값 밖: {sorted(set(r.lgd_role) - set(expect_roles))}"
        assert r.source_id.isin(["F4_ext_direct", "F3_ext_temp"]).all(), "source_id 가 기대값 밖"
    lines = v4_path.read_text(encoding="utf-8").splitlines()
    hdr = lines[0].split(",")
    i_id, i_src, i_alt = hdr.index("loc_id"), hdr.index("source_id"), hdr.index("alt_cm")
    outl, n_new, n_drop = [lines[0]], 0, 0
    for ln in lines[1:]:
        f = ln.split(",")
        lid = int(f[i_id])
        if lid < LOC0:
            if lid in drop_v3:
                n_drop += 1
                continue
            outl.append(ln)
        elif lid in keep_new:
            f[i_src] = "F4_direct"
            if alt_override is not None and lid in alt_override:
                f[i_alt] = repr(float(alt_override[lid]))
            outl.append(",".join(f)); n_new += 1
    assert n_new == len(keep_new), f"새 행 {n_new} != {len(keep_new)}"
    assert n_drop == len(drop_v3), f"뺀 v3 행 {n_drop} != {len(drop_v3)}"
    out.write_text("\n".join(outl) + "\n", encoding="utf-8")
    return dict(n_new=n_new, n_drop_v3=n_drop)


def structure(df, target):
    """h40 Data.source_idx(모드 x, macro 대상) 와 Data.split_structure 를 같은 규칙으로 계산."""
    y = df.alt_cm.values.astype(float)
    t_idx = np.where(df.macro.values == target)[0]
    if len(t_idx) == 0:                                            # 대상 셀이 없는 변형(예: Tibet 의 L41 a)
        return dict(n_cells=0, n_blocks=0, n_eval_cells=0, n_eval_blocks=0, alt_mean=np.nan, n_src=np.nan,
                    n_buffer_excluded=np.nan, n_unique_splits=0, n_valid_splits=0, nb_union=0,
                    min_nb_eval_used=np.nan, max_block_share_used=np.nan, few_blocks=False), {}, t_idx
    src = np.where(np.isfinite(y))[0]
    src = src[~np.isin(src, t_idx)]
    n0 = len(src)
    la, lo = df.lat.values, df.lon.values
    tl, tn = la[t_idx], lo[t_idx]
    keep = np.ones(len(src), bool)
    for j, i in enumerate(src):
        if np.abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < BUFFER_KM:
            keep[j] = False
    src = src[keep]
    allb = frozenset(df.block.values[t_idx])
    seen, info = [], {}
    for sp in SPLITS:
        A_idx, B_idx = half_split_blocks(df, t_idx, sp)
        a = frozenset(df.block.values[A_idx])
        evB = B_idx[eval_mask(df.iloc[B_idx])]
        dup = next((q for q, aq in seen if aq == a), -1)
        mir = next((q for q, aq in seen if aq == allb - a), -1)
        nb = int(df.iloc[evB].block.nunique())
        vc = pd.Series(df.block.values[evB]).value_counts()
        info[sp] = dict(split=sp, dup_of=int(dup), mirror_of=int(mir), n_A=int(len(A_idx)), nb_A=len(a), n_eval=int(len(evB)),
                        nb_eval=nb, valid=bool(dup < 0 and nb >= 2), max_block=int(vc.index[0]) if len(vc) else -1,
                        max_block_cells=int(vc.iloc[0]) if len(vc) else 0,
                        max_block_share=round(float(vc.iloc[0] / len(evB)), 4) if len(evB) else np.nan,
                        eval_blocks=sorted(set(df.block.values[evB].tolist())))
        seen.append((sp, a))
    uniq = [v for v in info.values() if v["dup_of"] < 0]
    valid = [v for v in uniq if v["valid"]]
    used = valid if valid else uniq
    union = set().union(*[set(v["eval_blocks"]) for v in used]) if used else set()
    em = eval_mask(df.iloc[t_idx])
    tdf = df.iloc[t_idx]
    mn = min(v["nb_eval"] for v in used) if used else np.nan
    return dict(n_cells=int(len(t_idx)), n_blocks=int(len(allb)), n_eval_cells=int(em.sum()),
                n_eval_blocks=int(tdf.block.values[em].size and len(set(tdf.block.values[em]))),
                alt_mean=round(float(np.nanmean(y[t_idx])), 1) if len(t_idx) else np.nan,
                n_src=int(len(src)), n_buffer_excluded=int(n0 - len(src)),
                n_unique_splits=len(uniq), n_valid_splits=len(valid), nb_union=len(union),
                min_nb_eval_used=mn, max_block_share_used=max(v["max_block_share"] for v in used) if used else np.nan,
                few_blocks=bool(used and mn < FEW_BLOCKS)), info, t_idx


def clusters(df, t_idx, locs, thresholds=(0.25, 0.5, 1.0)):
    """대상 셀을 위치 사이 대권 거리로 단일 연결해 묶음 수를 센다. 새 셀은 ext_cells_v1_locs 의 위치, v3 셀은 셀 좌표."""
    pts, owner = [], []
    for k, i in enumerate(t_idx):
        lid = int(df.loc_id.values[i])
        ll = locs.get(lid)
        if ll is None:
            pts.append((df.lat.values[i], df.lon.values[i])); owner.append(k)
        else:
            for a, b in ll:
                pts.append((a, b)); owner.append(k)
    P = np.array(pts)
    owner = np.array(owner)
    out = {}
    for th in thresholds:
        parent = list(range(len(t_idx)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        n_pairs = set()
        for j in range(len(P)):
            if j + 1 >= len(P):
                break
            d = haversine_km(P[j, 0], P[j, 1], P[j + 1:, 0], P[j + 1:, 1])
            for q in np.where(d <= th)[0]:
                a, b = owner[j], owner[j + 1 + q]
                if a != b:
                    n_pairs.add((min(a, b), max(a, b)))
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
        out[f"n_clusters_{th}km"] = len({find(x) for x in range(len(t_idx))})
        out[f"n_linked_pairs_{th}km"] = len(n_pairs)
    return out


def main():
    t0 = time.time()
    v4_path = PROC / "fidelity_base_v4.csv"
    v4 = pd.read_csv(v4_path, low_memory=False)
    lab = pd.read_csv(PROC / "fidelity_base_v4_labels.csv", low_memory=False)
    locs_t = pd.read_csv(EXT / "ext_cells_v1_locs.csv", low_memory=False)
    new = lab[lab.part == "new"].copy()
    new["cell_uid"] = new.label_set + "|" + new.cell_key
    soil4 = PROC / "e5_soil_tdd_v4.csv"
    tgt = new[new.lgd_role == "target"]
    v3lab = lab[lab.part == "v3"].set_index("loc_id")
    # 새 셀의 위치 좌표(묶음 계산)
    lc = locs_t.groupby("cell_uid")[["lat", "lon"]].apply(lambda g: list(zip(g.lat, g.lon))).to_dict()
    loc_of = {int(r.loc_id): lc.get(r.cell_uid) for r in new.itertuples()}
    # v3 지리 macro(독립성 거리)
    v3part = v4[v4.loc_id < LOC0]
    v3_geo = pd.Series([geo_macro_v3(r, a, b) for r, a, b in zip(v3part.region, v3part.lat, v3part.lon)], index=v3part.loc_id.values)

    def ids(macro, extra=None):
        m = (tgt.macro_v4 == macro).values
        if extra is not None:
            m &= extra.reindex(tgt.index).fillna(False).values.astype(bool)
        return tgt.loc_id.values[m]

    specs = []
    for t in ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]:
        specs.append(dict(spec=f"v3ref_{t}", target=t, kind="v3_reference", table="v3", keep=None))
    specs.append(dict(spec="Tibet", target="Tibet_LGD", kind="new_region", table="v4", keep=ids("Tibet_LGD")))
    specs.append(dict(spec="NAtlantic", target="NAtlantic", kind="new_region", table="v4", keep=ids("NAtlantic")))
    specs.append(dict(spec="Russia_C", target="Russia_C", kind="new_region(v3 셀 확충)", table="v4", keep=ids("Russia_C")))
    for t in ["Russia_W", "Russia_E", "Canada"]:
        specs.append(dict(spec=f"{t}_expanded", target=t, kind="expanded_L40", table="v4", keep=ids(t)))
    kyt = tgt.label_subtypes.fillna("").str.contains("pf_top_below_frozen_al")
    specs.append(dict(spec="Russia_E_expanded_noKytalyk", target="Russia_E", kind="expanded_L40_variant", table="v4",
                      keep=ids("Russia_E", ~kyt)))
    aux = new[(new.lgd_role == "aux_temp_L39")].loc_id.values
    specs.append(dict(spec="Tibet_L39_temp", target="Tibet_LGD", kind="aux_L39", table="v4", keep=aux, roles=["aux_temp_L39"]))
    a_ok = tgt.l41a_all_direct3 == 1
    b_ok = tgt.l41b_early_single == 0
    c_ok = tgt.l41c_dataset_statement == 0
    d_ok = tgt.year_max >= 2010
    e_ok = tgt.cens_affected == 0
    g_ok = tgt.e5_glacier_grid == 0
    for t, mac in [("Tibet", "Tibet_LGD"), ("NAtlantic", "NAtlantic"), ("Russia_C", "Russia_C")]:
        for v, ok in [("a", a_ok), ("b", b_ok), ("c", c_ok), ("d", d_ok), ("e", e_ok)]:
            specs.append(dict(spec=f"{t}_L41{v}", target=mac, kind=f"variant_L41{v}", table="v4", keep=ids(mac, ok)))
        k = ids(mac)
        alt_f = tgt.set_index("loc_id").loc[k, "alt_cm_cens_lb"].to_dict()
        specs.append(dict(spec=f"{t}_L41f", target=mac, kind="variant_L41f", table="v4", keep=k, alt=alt_f))
        gl_v3 = [int(i) for i in v3lab.index[(v3lab.macro_v4 == mac) & (v3lab.e5_glacier_grid == 1) & (v3lab.source_id == "F4_direct")]]
        specs.append(dict(spec=f"{t}_L41g", target=mac, kind="variant_L41g", table="v4", keep=ids(mac, g_ok), drop_extra=gl_v3))
        if t in LARGEST_SURVEY:
            h_ok = ~tgt.sources.fillna("").str.split(";").map(lambda L: LARGEST_SURVEY[t] in L)
            specs.append(dict(spec=f"{t}_L41h", target=mac, kind="variant_L41h", table="v4", keep=ids(mac, h_ok),
                              note="점 추정만(WRAPUP (a)9)"))
        else:
            specs.append(dict(spec=f"{t}_L41h", target=mac, kind="variant_L41h", table="v4", keep=np.array([], int),
                              note="대상 셀 전부가 한 자료원(qtp_du_gpr)이라 변형 불가"))

    base_v3 = load_base(PROC)
    p4_v3 = base_v3.set_index("loc_id")[["p4_ku", "p2_edaphic"]]
    rows, srows = [], []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 재현 확인: v3 행만 고른 실행 표가 v4 의 v3 원문 줄과 같다(region 을 바꾼 3줄은 v3 원문과 다르다)
        pchk = td / "v3only.csv"
        write_run_table_text(v4_path, pchk, keep_new=[])
        l_run = pchk.read_text(encoding="utf-8").splitlines()
        l_v3 = (PROC / "fidelity_base_v3.csv").read_text(encoding="utf-8").splitlines()
        text_check = dict(n_lines_run=len(l_run), n_lines_v3=len(l_v3),
                          n_lines_differ_from_v3=int(sum(a != b for a, b in zip(l_run, l_v3))),
                          note="다른 줄은 region 을 CALM_Greenland 에서 NAtlantic 으로 바꾼 3줄이어야 한다")
        for s in specs:
            if s["table"] == "v3":
                df = base_v3
                wr = {}
            else:
                p = td / "run.csv"
                drop = DROP_V3.get(s["target"], []) + s.get("drop_extra", [])
                wr = write_run_table_text(v4_path, p, s["keep"], drop_v3=drop, alt_override=s.get("alt"),
                                          expect_roles=s.get("roles", ["target"]), lab=lab)
                df = load_base(PROC, base=str(p), soil=str(soil4))
            st, info, t_idx = structure(df, s["target"])
            if s["table"] == "v4":
                k = new[new.loc_id.isin(s["keep"])]
                st.update(n_new_cells=int(len(k)), n_v3_cells=int(st["n_cells"] - len(k)),
                          n_new_blocks=int(v4[v4.loc_id.isin(s["keep"])].block.nunique()),
                          min_dist_other_macro_km=float(k.dist_other_macro_km.min()) if len(k) else np.nan,
                          n_drop_v3=wr.get("n_drop_v3", 0), drop_v3=";".join(str(x) for x in (DROP_V3.get(s["target"], []) + s.get("drop_extra", []))))
                # v3 구성 셀까지 포함한 독립성 거리(6B.4, 지리 macro)
                geo_t = GEO_OF_TARGET.get(s["target"], s["target"])
                f4 = df[df.loc_id < LOC0]
                other = f4[v3_geo.reindex(f4.loc_id.values).values != geo_t]
                if len(t_idx):
                    dmins = [float(haversine_km(df.lat.values[i], df.lon.values[i], other.lat.values, other.lon.values).min()) for i in t_idx]
                    j = int(np.argmin(dmins))
                    st.update(min_dist_other_macro_all_km=round(dmins[j], 1), min_dist_other_macro_all_loc=int(df.loc_id.values[t_idx[j]]))
                    st.update(**clusters(df, t_idx, loc_of))
                    comp = {}
                    for i in t_idx:
                        lid = int(df.loc_id.values[i])
                        key = "v3" if lid < LOC0 else str(new.set_index("loc_id").loc[lid, "sources"])
                        comp.setdefault(key, []).append(int(df.block.values[i]))
                    st["composition"] = json.dumps({kk: dict(cells=len(vv), blocks=len(set(vv))) for kk, vv in sorted(comp.items())},
                                                   ensure_ascii=False)
                # v3 행의 p4_ku, p2_edaphic 차(결측 대체 중앙값이 실행 표 전체로 계산되는 영향)
                pv = df.set_index("loc_id")[["p4_ku", "p2_edaphic"]]
                com = pv.index.intersection(p4_v3.index)
                dif = (pv.loc[com] - p4_v3.loc[com]).abs()
                st.update(max_abs_diff_p4_ku_v3rows=float(np.nanmax(dif.p4_ku.values)) if len(com) else np.nan,
                          max_abs_diff_p2_edaphic_v3rows=float(np.nanmax(dif.p2_edaphic.values)) if len(com) else np.nan)
            regime = ("shallow" if st["alt_mean"] < DEEP_CM else "deep") if st["n_cells"] else ""
            elig = bool(st["n_valid_splits"] >= 1 and st["nb_union"] >= MIN_BLOCKS_CI)
            if s["kind"].startswith("new_region"):
                role = ("PE1+PE2" if regime == "shallow" else "PE2") if elig else "point_only"
            elif s["kind"].startswith("expanded_L40"):
                role = "L40 민감도(풀에는 v3 셀의 본 실행 값)" + ("" if elig else "; 확충판 부적격")
            elif s["kind"] == "aux_L39":
                role = "L39 보조" + ("" if elig else "; 점 추정만")
            elif s["kind"].startswith("variant_L41"):
                role = "L41 변형" + ("" if elig else "; 변형 불가(부적격)")
            else:
                role = "참조"
            if s.get("note"):
                role += "; " + s["note"]
            rows.append(dict(spec=s["spec"], target=s["target"], kind=s["kind"], **st, regime=regime,
                             eligible=elig, role=role))
            for sp, v in info.items():
                srows.append(dict(spec=s["spec"], target=s["target"], **{k: v[k] for k in v if k != "eval_blocks"}))
            print(f"  {s['spec']:30s} 셀 {st['n_cells']:5d} 블록 {st['n_blocks']:3d} 채점 {st['n_eval_cells']:5d} "
                  f"유효 분할 {st['n_valid_splits']} 합집합 {st['nb_union']:3d} 최소 채점 블록 {st['min_nb_eval_used']} "
                  f"최대 비중 {st['max_block_share_used']} 원천 {st['n_src']} 평균 {st['alt_mean']} → {role}", flush=True)
    el = pd.DataFrame(rows)
    sp = pd.DataFrame(srows)
    el.to_csv(PROC / "lgd_eligibility_v1.csv", index=False)
    sp.to_csv(PROC / "lgd_eligibility_v1_splits.csv", index=False)

    # h40 --count-only(v3) 대조
    ref = pd.read_csv(PROC / "lg" / "lg_count_cpu.csv")
    ref = ref[(ref["mode"] == "x") & (ref.learner.isna())][["target", "split", "n_A", "n_eval", "nb_eval", "n_src", "valid"]].drop_duplicates()
    mine = sp[sp.spec.str.startswith("v3ref_")].merge(el[["spec", "n_src"]], on="spec")
    cmp = mine.merge(ref, on=["target", "split"], suffixes=("", "_h40"))
    diff = {c: int((cmp[c] != cmp[c + "_h40"]).sum()) for c in ["n_A", "n_eval", "nb_eval", "n_src", "valid"]}
    meta = dict(
        stage="LGD 6B.6 단계 4(적격 표). 개정 13", created=time.strftime("%Y-%m-%d %H:%M"),
        script="scripts/1_data_prep/lgd_eligibility_v1.py", script_sha256=sha(Path(__file__)),
        inputs={p: sha(PROC / p) for p in ["fidelity_base_v4.csv", "e5_soil_tdd_v4.csv", "fidelity_base_v4_labels.csv"]},
        rules=dict(splits=SPLITS, buffer_km=BUFFER_KM, min_blocks_union=MIN_BLOCKS_CI, deep_cm=DEEP_CM, few_blocks=FEW_BLOCKS,
                   valid="중복 아님 · 채점 블록 ≥ 2(h40)", union="유효 분할(없으면 고유 분할)의 B 채점 블록 합집합(h40 TMx.nb_union)",
                   source="finite y 인 F4_direct 행(실행 표 기준) − 대상 셀 − 대상 셀 100 km 버퍼(h40 source_idx, 모드 x)",
                   run_table="v4 원문 줄에서 v3 행 전부(대상별 drop_v3 제외) + 그 대상의 새 행(source_id 를 F4_direct 로 바꿈). "
                             "다른 새 지역 행은 넣지 않는다(6B.6 주 설정)",
                   drop_v3={k: v for k, v in DROP_V3.items()},
                   l41="a = 셀의 모든 위치가 탐침·융해관·동결관, b = 8월 15일 이전 record_date 단일 방문 행이 있는 셀 제외(빌린 날짜는 조사 "
                       "시작일로 판정), c = dataset_statement 행이 있는 셀 제외, d = year_max >= 2010 셀만(WRAPUP (a)9, v3 구성 셀은 남김), "
                       "e = 절단 영향 셀(cens_affected) 제외, f = 절단 행을 하한값으로 넣은 셀 값(alt_cm_cens_lb), g = ERA5-Land 빙하 격자 셀"
                       "(월 적설 최솟값 >= 1 m) 제외(v3 구성 셀 포함), h = 새 셀이 가장 많은 단일 조사의 셀 제외(점 추정만) "
                       f"{LARGEST_SURVEY}",
                   l40_variant="Russia_E 확충판에서 Kytalyk(palmtag, pf_top_below_frozen_al) 셀 제외",
                   max_block_share="분할의 B 채점 셀 가운데 가장 큰 블록의 셀 비중. 사용 분할의 최댓값을 적는다",
                   clusters="대상 셀을 구성 위치(새 셀) 또는 셀 좌표(v3 셀) 사이 대권 거리로 단일 연결한 묶음 수",
                   min_dist_other_macro_all_km="대상의 모든 셀(v3 구성 셀 포함)에서 다른 지리 macro 의 v3 F4_direct 까지 최소 거리"),
        run_table_text_check=text_check,
        h40_count_only_check=dict(n_compared=int(len(cmp)), n_mismatch=diff, reference="data/processed/lg/lg_count_cpu.csv(모드 x, 방법 축 행)"),
        outputs={p: sha(PROC / p) for p in ["lgd_eligibility_v1.csv", "lgd_eligibility_v1_splits.csv"]},
        elapsed_s=round(time.time() - t0, 1))
    (PROC / "lgd_eligibility_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[check h40 count-only] 비교 {len(cmp)} 행 · 불일치 {diff} · 원문 줄 점검 {text_check} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
