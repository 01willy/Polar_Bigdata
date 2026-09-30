"""scripts/4_visualization/paper/n1_target_summary.py 단위 시험(합성 곡선 표만 쓴다. 실제 lg_curve·lgt·lgf·lgfn 표를 열지 않는다).

(a) 4분 판정 경계: δ 0.5 cm 경계 포함(동등), 한 가중만 CI 상한 < 0 이면 우세가 아니다, 비유한 CI 는 판정 불가, δ 1.0 보조 열.
    우세·열세가 h40._sig 의 improve·worse 와 같고, 상수가 h42 기본값과 같다.
(b) 평균·중앙값(점 추정), 점 추정 증가 수, 최대 증가와 그 대상(동률이면 묶음 순서의 앞 대상).
(c) 구성 고정: 묶음 대상 목록은 n 과 무관하고, 행이 없는 대상은 n_missing 으로, point_only 대상은 n_point_only 로 따로 센다.
(d) 행 선택: α·배치·기준 λ·학습기 필터, P1 의 d_p1 제외, 보조 표는 학습기별로 센다, 같은 대상 중복 행은 오류.
(e) h40.build_curve 로 만든 합성 lg_curve(형식 동일)에서 센 값이 표의 CI 로 다시 센 값과 같다. 개수의 합이 맞다.
(f) 같은 입력이면 산출 CSV 가 바이트 단위로 같다. nboot 는 곡선 표 메타에서 읽고, 없으면 기본값과 'default' 를 적는다.
(g) 명령행 --synthetic: 화면에 결과 값(소수)이 나오지 않는다.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_n1_target_summary.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


N1 = _load("n1_target_summary", "scripts/4_visualization/paper/n1_target_summary.py")
X, H = N1.X, N1.H
NAN = float("nan")


def row(target, mode="x", method="D0", n=0, learner=None, lam=None, alpha="1", placement="cell", point_only=False, p0=None, p1=None, **kw):
    """곡선 표 한 행. p0·p1 = (d, lo, hi, d_beq, beq_lo, beq_hi)."""
    learner = learner if learner is not None else ("none" if method.startswith("P") else H.BASE_LEARNER)
    lam = lam if lam is not None else H.base_lam(method)
    r = dict(target=target, mode=mode, method=method, learner=learner, alpha=alpha, placement=placement, n=n, lam=lam, point_only=point_only)
    for k, v in (("p0", p0), ("p1", p1)):
        v = v if v is not None else (NAN,) * 6
        r.update({f"d_{k}": v[0], f"d_{k}_lo": v[1], f"d_{k}_hi": v[2], f"d_{k}_beq": v[3], f"d_{k}_beq_lo": v[4], f"d_{k}_beq_hi": v[5]})
    r.update(kw)
    return r


def ci(d, lo, hi, blo=None, bhi=None):
    return (d, lo, hi, d, lo if blo is None else blo, hi if bhi is None else bhi)


def summary(rows, main=True):
    df = pd.DataFrame(rows)
    out = N1.frame(N1.summarize(df, "lg", main=main), ["lg"])
    return out


def pick(s, group, method="D0", n=0, kind="d_p0", learner=None):
    q = s[(s.group == group) & (s.method == method) & (s.n == n) & (s.kind == kind)]
    if learner is not None:
        q = q[q.learner == learner]
    assert len(q) == 1, (group, method, n, kind, learner, len(q))
    return q.iloc[0]


# ---------------------------------------------------------------- (a)
def test_verdict_boundaries_match_h42_h40():
    a = X.parse_args([])
    assert N1.DELTA_EQ == a.delta_eq == 0.5 and N1.DELTA_EQ_AUX == a.delta_eq_aux == 1.0
    cases = {
        "Lena": (ci(-1.5, -2.0, -1.0, -2.0, -0.1), "우세", "우세"),
        "Canada": (ci(0.0, -0.5, 0.5), "동등", "동등"),                           # 경계 포함
        "Russia_W": (ci(0.0, -0.5, 0.5000001), "미결정", "동등"),
        "Russia_E": (ci(1.0, 0.1, 2.0, 0.2, 3.0), "열세", "열세"),
        "Alaska": (ci(-0.5, -1.0, -0.1, -1.0, 0.1), "미결정", "동등"),             # 셀 가중만 상한 < 0: 우세 아님
    }
    rows = [row(t, p0=v[0]) for t, v in cases.items()]
    rows.append(row("AL-1", mode="i", p0=(0.3, NAN, 0.6, 0.3, -0.1, 0.6)))      # 비유한 끝값
    s = summary(rows)
    r = pick(s, "indep")
    assert (r.n_superior, r.n_equivalent, r.n_undecided, r.n_inferior, r.n_na) == (1, 1, 2, 1, 0)
    assert (r.n_superior_d10, r.n_equivalent_d10, r.n_undecided_d10, r.n_inferior_d10) == (1, 3, 0, 1)
    assert r.superior_targets == "Lena|x" and r.inferior_targets == "Russia_E|x"
    p4 = pick(s, "P4")
    assert (p4.n_superior, p4.n_equivalent, p4.n_undecided, p4.n_inferior) == (1, 1, 1, 1)
    si = pick(s, "sub_i")
    assert si.n_na == 1 and si.n_rows == 1 and si.n_missing == 9
    for t, (v, e5, e10) in cases.items():
        assert X.verdict4(v[1], v[2], v[4], v[5], 0.5) == e5 and X.verdict4(v[1], v[2], v[4], v[5], 1.0) == e10
        sg = H._sig(v[1], v[2], v[4], v[5])
        assert (e5 == "우세") == (sg == "improve") and (e5 == "열세") == (sg == "worse")


# ---------------------------------------------------------------- (b)
def test_mean_median_max_increase():
    ds = {"Lena": 1.0, "Canada": -2.0, "Russia_W": 3.0, "Russia_E": 3.0, "Alaska": 0.5}
    rows = [row(t, p0=ci(d, d - 1, d + 1)) for t, d in ds.items()]
    s = summary(rows)
    r = pick(s, "indep")
    assert r.mean_d == pytest.approx(np.mean(list(ds.values()))) and r.median_d == pytest.approx(1.0)
    assert r.n_increase_point == 4 and r.n_finite == 5
    assert r.max_increase == pytest.approx(3.0) and r.max_increase_target == "Russia_W|x"      # 동률은 묶음 순서의 앞 대상
    p4 = pick(s, "P4")
    assert p4.mean_d == pytest.approx(np.mean([1.0, -2.0, 3.0, 3.0])) and p4.median_d == pytest.approx(2.0)
    neg = summary([row("Lena", p0=ci(-1.0, -2, 0)), row("Canada", p0=ci(-3.0, -4, -2))])
    q = pick(neg, "P4")
    assert q.max_increase == pytest.approx(-1.0) and q.max_increase_target == "Lena|x" and q.n_increase_point == 0


# ---------------------------------------------------------------- (c)
def test_composition_missing_and_point_only():
    rows = [row(t, n=40, p0=ci(0.2, -0.1, 0.5)) for t in ("Lena", "Canada", "Alaska")]
    rows += [row("AL-2", n=40, point_only=True, p0=(0.4, NAN, NAN, 0.4, NAN, NAN)), row("AL-1", n=40, p0=ci(0.1, -0.2, 0.4))]
    rows += [row("Russia_C", n=40, point_only=True, p0=(5.0, NAN, NAN, 5.0, NAN, NAN)), row("Greenland", n=40, point_only=True, p0=(9.0,) * 6)]
    s = summary(rows)
    r = pick(s, "indep", n=40)
    assert (r.n_group, r.n_rows, r.n_missing, r.n_point_only) == (5, 3, 2, 0)
    assert r.missing_targets == "Russia_W|x,Russia_E|x" and not bool(r.composition_complete)
    assert r.group_targets == ",".join(N1.GROUPS["indep"])
    sx = pick(s, "sub_x", n=40)
    assert (sx.n_group, sx.n_rows, sx.n_point_only, sx.n_missing) == (10, 1, 1, 8)
    assert sx.point_only_targets == "AL-2|x" and sx.max_increase_target == "AL-1|x"
    assert sx.n_na == 0                                                       # point_only 행은 판정 불가로도 세지 않는다
    for g in ("indep", "P4", "sub_x"):                                       # 러시아 C·그린란드는 어느 묶음에도 없다
        assert "Russia_C" not in pick(s, g, n=40).group_targets and "Greenland" not in pick(s, g, n=40).group_targets


# ---------------------------------------------------------------- (d)
def test_row_selection_and_aux_learners():
    good = ci(-1.0, -2.0, -0.5)
    rows = [row("Lena", method="R1", n=10, p0=good, p1=ci(0.3, 0.1, 0.5)),
            row("Lena", method="R1", n=10, alpha="10", p0=ci(9.0, 8, 10)),
            row("Lena", method="R1", n=10, placement="block", p0=ci(9.0, 8, 10)),
            row("Lena", method="R1", n=10, lam=0.5, p0=ci(9.0, 8, 10)),
            row("Lena", method="R1", n=10, learner="mlp", p0=ci(9.0, 8, 10)),
            row("Lena", method="R2", n=10, p0=ci(9.0, 8, 10)),
            row("Lena", method="P1", n=10, p0=ci(0.5, 0.1, 0.9), p1=(0.0,) * 6)]
    s = summary(rows)
    assert set(s.method) == {"R1", "P1"} and set(s.learner) == {H.BASE_LEARNER, "none"}
    r = pick(s, "P4", method="R1", n=10)
    assert r.mean_d == pytest.approx(-1.0) and r.n_superior == 1
    assert pick(s, "P4", method="R1", n=10, kind="d_p1").n_inferior == 1
    assert not ((s.method == "P1") & (s.kind == "d_p1")).any()
    aux = summary([row("Lena", learner="mlp_tuned", p0=ci(2.0, 1.0, 3.0)), row("Lena", learner="mlp", p0=ci(-1.0, -2.0, -0.5)),
                   row("Canada", learner="mlp_tuned", p0=ci(0.1, -0.2, 0.3))], main=False)
    assert set(aux.learner) == {"mlp", "mlp_tuned"}
    t = pick(aux, "P4", learner="mlp_tuned")
    assert (t.n_rows, t.n_inferior, t.n_equivalent) == (2, 1, 1) and t.contrast == "D0[mlp_tuned]-P0"
    with pytest.raises(ValueError):
        summary([row("Lena", p0=good), row("Lena", p0=good)])


# ---------------------------------------------------------------- (e)(f)
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("n1")
    curve, aux = N1.synthetic_curve(tmp / "in", nboot=100)
    res = N1.run(curve, tmp / "out", aux=[aux], synthetic=True)
    return dict(tmp=tmp, curve=curve, aux=aux, res=res, df=res["df"], meta=json.loads(Path(res["meta"]).read_text()))


def test_counts_from_build_curve_format(synth):
    cur = pd.read_csv(synth["curve"], dtype=dict(alpha=str))
    assert {"d_p0_lo", "d_p0_hi", "d_p0_beq_lo", "d_p0_beq_hi", "d_p1_lo", "point_only", "n_splits_valid"} <= set(cur.columns)
    df = synth["df"]
    lg = df[df.source == "lg"]
    assert len(lg) > 50
    v4 = ["n_superior", "n_equivalent", "n_undecided", "n_inferior", "n_na"]
    assert (lg[v4].sum(1) == lg.n_rows).all() and (lg.n_rows + lg.n_missing + lg.n_point_only == lg.n_group).all()
    sel = N1.select_rows(cur, main=True)
    for r in lg.sample(25, random_state=0).itertuples():
        k = r.kind[2:]
        q = sel[(sel.method == r.method) & (sel.learner == r.learner) & (sel.n == r.n) & sel["name"].isin(N1.GROUPS[r.group]) & ~sel.point_only]
        v = [X.verdict4(a, b, c, d, 0.5) for a, b, c, d in q[[f"d_{k}_lo", f"d_{k}_hi", f"d_{k}_beq_lo", f"d_{k}_beq_hi"]].itertuples(index=False)]
        assert r.n_inferior == v.count("열세") and r.n_superior == v.count("우세") and r.n_rows == len(q)
        d = q[f"d_{k}"].astype(float)
        if d.notna().any():
            assert r.mean_d == pytest.approx(d.mean()) and r.median_d == pytest.approx(d.median()) and r.max_increase == pytest.approx(d.max())
    ind = lg[(lg.group == "indep") & (lg.method == "D0") & (lg.kind == "d_p0")]
    assert sorted(ind.n.unique()) == [-1, 0, 3, 10, 40, 160]
    i40 = ind[ind.n == 40].iloc[0]
    assert i40.missing_targets == "Russia_W|x,Russia_E|x"                   # 러시아 W·E 는 |A| < 40
    assert (lg.nboot == 100).all() and (lg.nboot_source == "meta").all()
    fn = df[df.source == "lgfn"]
    assert set(fn.learner) == {"mlp_tuned", "none"}


def test_deterministic_and_meta(synth, tmp_path):
    res2 = N1.run(synth["curve"], tmp_path / "again", aux=[synth["aux"]], synthetic=True)
    assert Path(synth["res"]["csv"]).read_bytes() == Path(res2["csv"]).read_bytes()
    m = synth["meta"]
    assert [i["source"] for i in m["inputs"]] == ["lg", "lgfn"] and all(len(i["sha256"]) == 64 for i in m["inputs"])
    assert m["delta_eq"] == 0.5 and m["delta_eq_aux"] == 1.0 and set(m["code_sha256"]) >= {"module", "h40", "h42"}
    other = tmp_path / "x_curve.csv"                                           # 메타가 없는 표
    pd.read_csv(synth["curve"]).to_csv(other, index=False)
    r3 = N1.run(other, tmp_path / "nometa", curve_nboot=1000)
    assert (r3["df"].nboot == 1000).all() and (r3["df"].nboot_source == "default").all()


# ---------------------------------------------------------------- (g)
def test_cli_synthetic_prints_no_values(tmp_path, capsys):
    res = N1.main(["--synthetic", "--out", str(tmp_path), "--synthetic-nboot", "50"])
    out = capsys.readouterr().out
    assert Path(res["csv"]).name == "n1_target_summary_synthetic.csv" and Path(res["meta"]).exists()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines and all(ln.startswith("[n1_target_summary]") for ln in lines)
    assert not re.search(r"\d\.\d", out)
    with pytest.raises(SystemExit):
        N1.parse_args([])
