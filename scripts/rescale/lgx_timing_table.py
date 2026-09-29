#!/usr/bin/env python3
"""LGX(H41·H42) 적합 시간 표와 본 실행 환산. scripts/rescale/lg_timing_table.py 의 LGX 판이다(그 파일은 본 실행 기록이라 고치지 않는다).

입력
  <lgx-dir>/shards/*_unit.json            h42 조각의 완료 표지(분할 4·5 축은 h40 이 쓴 표지). n_fit_detail·sec_detail·rows_detail·est_detail,
                                          키 = "적합 축|학습기|방법"
  <lgx-dir>/ladder/shards/*_unit.json     h41 조각의 완료 표지(같은 형식, scheme·rep·fold)
  <lg-dir>/shards/{lg,lgrm}_precheck__gpu__*_unit.json   LG 사전 점검(qOkSo)의 gpu 조각 표지. 분할 4·5 축(g45)의 실측으로 쓴다
                                          (h42 의 사전 점검은 이 축을 다시 재지 않는다. 같은 적합을 LG 사전 점검이 이미 쟀다)
  <lgx-dir>/<count-tag>_count_{cpu,gpu}_detail.csv   h42 본 실행 범위의 --count-only 결과(축, 학습기, 방법, n_fit, est_h)
  <lgx-dir>/ladder/<ladder-count-tag>_count.csv      h41 본 실행 범위의 --count-only 결과(단위별 est_s)

산출(<out-dir>)
  <prefix>_timing_by_learner.csv    (하네스, tag, 부분, 축, 학습기, 방법)별 적합 수, 시간, 학습 행 수, 실행 epoch
  <prefix>_full_projection.csv      (하네스, 부분, 축 또는 단, 학습기)별 본 실행 누적 시간 환산. 적합 수 표가 없는 하네스는 뺀다

환산식
  보정 계수 c = Σ(실측 시간 × epoch 환산) / Σ(추정 시간). 추정 시간은 조각의 est_detail(하네스의 기준 시간표)이다.
  epoch 환산 = 본 실행 epoch / 실측 epoch(신경망 학습기만). 조기 종료와 고정 비용을 무시하므로 상한에 가깝다.
  h42: (축, 학습기, 방법)의 실측이 있으면 그 c, 없으면 (축, 학습기), 그다음 학습기 전체의 c 를 쓴다. 본 실행 누적 시간 = c × 적합 수 표의 est_h.
  h41: 단(scheme)별 c 를 쓰고 실측이 없는 단은 전체 c 를 쓴다. 본 실행 누적 시간 = c × 적합 수 표의 단별 est_s 합.
  사전 점검 조각(tag 가 _precheck 로 끝남)의 실측이 있으면 그것을 쓰고, 없으면 스모크 실측을 쓴다(source 열에 적는다).
  벽시계 = 누적 시간 / 워커 수. 동시 실행의 감속은 반영하지 않는다.

한계
  실측은 대상 하나(캐나다 x) 또는 스모크 대상 3개의 값이다. 4코어 노드(iolite-1)에서 프로세스 1개로 잰 값이라 64코어 동시 실행의 감속은 들어 있지 않다.

실행: python3 scripts/rescale/lgx_timing_table.py --lgx-dir data/processed/lgx --out-dir . --prefix lgx_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TORCH = ("mlp", "realmlp", "ftt", "tabm", "cfm", "ddpm", "nflow")
NO_EPOCH_AXES = ("mlp_cont",)               # 이어 학습 축은 ft_epochs 를 쓴다(스모크에서도 본 실행과 같다)
TAG_AXIS = {"b": "base", "1": "x1", "2": "x2", "9": "x9", "n3": "n3", "p": "prox", "5": "x5", "9f": "x9f", "n4": "n4", "n4a": "n4a",
            "g": "g45", "nn": "nn"}                  # tag 마디 → h42 축(r0 은 부분으로 가른다: cpu = ridge, gpu = r0)
LG_PRECHECK_TAGS = ("lg_precheck", "lgrm_precheck")   # LG 사전 점검의 조각 tag(RealMLP 는 epoch 를 줄여 따로 쟀다)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LGX 적합 시간 표와 본 실행 환산")
    ap.add_argument("--lgx-dir", default="data/processed/lgx")
    ap.add_argument("--lg-dir", default="data/processed/lg", help="LG 사전 점검의 gpu 조각 표지가 있는 디렉터리(없으면 건너뛴다)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prefix", default="lgx_smoke")
    ap.add_argument("--tag", default="lgx", help="h42 조각 tag 의 앞 마디")
    ap.add_argument("--count-tag", default="lgx", help="h42 적합 수 표의 tag(<tag>_count_<부분>_detail.csv)")
    ap.add_argument("--ladder-count-tag", default="lgv", help="h41 적합 수 표의 tag(<tag>_count.csv)")
    ap.add_argument("--epochs-full", type=int, default=100)
    ap.add_argument("--realmlp-epochs-full", type=int, default=256)
    ap.add_argument("--gpu-workers", type=int, default=12, help="본 실행 GPU 워커 수(GPU 4장 × 3)")
    ap.add_argument("--cpu-workers", type=int, default=14, help="본 실행 h42 CPU 워커 수")
    ap.add_argument("--h41-workers", type=int, default=2, help="본 실행 h41 워커 수")
    return ap.parse_args(argv)


def axis_of(u, tag_head):
    """조각의 h42 축. h42 가 쓴 표지에는 axis 가 있다. 분할 4·5 축(h40 이 쓴 표지)은 tag 마디로 가린다."""
    ax = str(u.get("axis", "") or "")
    if ax:
        return ax
    tag = str(u.get("tag", ""))
    for suf in ("_smoke", "_precheck"):
        if tag.endswith(suf):
            tag = tag[:-len(suf)]
    node = tag[len(tag_head):] if tag.startswith(tag_head) else tag
    if node == "r0":
        return "ridge" if str(u.get("part", "")) == "cpu" else "r0"
    return TAG_AXIS.get(node, node or "unknown")


def epochs_of(cfg, learner, fit_axis, full):
    """(실측 epoch, 본 실행 epoch). 해당 없음은 (nan, nan)."""
    if learner not in TORCH or fit_axis in NO_EPOCH_AXES:
        return np.nan, np.nan
    if learner == "realmlp":
        return float(cfg.get("realmlp_epochs", np.nan)), float(full["realmlp"])
    return float(cfg.get("epochs", np.nan)), float(full["torch"])


def read_units(shards: Path, harness, tag_head, full, lg_precheck=False):
    """조각 표지를 읽는다. lg_precheck = True 이면 LG 사전 점검의 gpu 조각만 읽어 분할 4·5 축(g45)의 실측으로 둔다."""
    rows, bad = [], []
    if not shards.exists():
        return pd.DataFrame(), bad
    for p in sorted(shards.glob("*_unit.json")):
        try:
            u = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            bad.append(f"{p.name}: {e!r}"); continue
        det = u.get("n_fit_detail") or {}
        if not det:
            continue                                                       # 해석식만 있는 조각(n4a 등)
        if lg_precheck and not (str(u.get("tag", "")) in LG_PRECHECK_TAGS and str(u.get("part", "")) == "gpu"):
            continue
        cfg = u.get("cfg") or {}
        ax = "g45" if lg_precheck else (axis_of(u, tag_head) if harness == "h42" else str(u.get("scheme", "")))
        for k, nf in det.items():
            fa, lr, m = (k.split("|") + ["", ""])[:3]
            er, ef = epochs_of(cfg, lr, fa, full)
            rows.append(dict(harness=harness, tag=str(u.get("tag", "")), part=str(u.get("part", "cpu") or "cpu"), axis=ax, fit_axis=fa, learner=lr,
                             method=m, unit=p.name[:-len("_unit.json")], n_fit=int(nf), sec=float((u.get("sec_detail") or {}).get(k, 0.0)),
                             rows=float((u.get("rows_detail") or {}).get(k, 0.0)), est_s=float((u.get("est_detail") or {}).get(k, np.nan)),
                             epochs_run=er, epochs_full=ef, device=str(u.get("device", "")), threads=u.get("threads", ""),
                             status=str(u.get("status", "")), unit_elapsed_s=float(u.get("elapsed_s", np.nan))))
    return pd.DataFrame(rows), bad


def by_learner(d: pd.DataFrame):
    g = d.groupby(["harness", "tag", "part", "axis", "learner", "method"], as_index=False, sort=True).agg(
        n_units=("unit", "nunique"), n_fit=("n_fit", "sum"), sec=("sec", "sum"), rows=("rows", "sum"), est_s=("est_s", "sum"),
        epochs_run=("epochs_run", "max"), epochs_full=("epochs_full", "max"), device=("device", lambda s: ",".join(sorted(set(s)))),
        threads=("threads", "max"))
    g["sec_per_fit"] = g.sec / g.n_fit.clip(lower=1)
    g["rows_per_fit"] = g.rows / g.n_fit.clip(lower=1)
    g["calib"] = np.where(g.est_s > 0, g.sec / g.est_s.where(g.est_s > 0), np.nan)
    g["epoch_scale"] = (g.epochs_full / g.epochs_run).where(g.epochs_run > 0)
    g["sec_per_fit_full_epochs"] = g.sec_per_fit * g.epoch_scale.fillna(1.0)
    cols = ["harness", "tag", "part", "axis", "learner", "method", "n_units", "n_fit", "sec", "sec_per_fit", "rows_per_fit", "est_s", "calib",
            "epochs_run", "epochs_full", "epoch_scale", "sec_per_fit_full_epochs", "device", "threads"]
    return g[cols]


def calib_of(d: pd.DataFrame):
    """(보정 계수, 실측 적합 수). 실측 시간에 epoch 환산을 곱한 뒤 추정 시간으로 나눈다."""
    d = d[(d.est_s > 0) & (d.n_fit > 0)]
    if not len(d):
        return np.nan, 0
    sc = (d.epochs_full / d.epochs_run).where(d.epochs_run > 0).fillna(1.0)
    return float((d.sec * sc).sum() / d.est_s.sum()), int(d.n_fit.sum())


def pick_source(d: pd.DataFrame):
    """사전 점검 실측이 있으면 그것, 없으면 스모크 실측."""
    if not len(d):
        return d, "없음"
    pre = d[d.tag.astype(str).str.endswith("_precheck")]
    if len(pre):
        return pre, "사전 점검"
    return d, "스모크(신경망은 epoch 환산, 상한)"


def project_h42(d: pd.DataFrame, lgx_dir: Path, a):
    out = []
    for part, workers in (("cpu", a.cpu_workers), ("gpu", a.gpu_workers)):
        f = lgx_dir / f"{a.count_tag}_count_{part}_detail.csv"
        if not f.exists():
            continue
        cnt = pd.read_csv(f, keep_default_na=False, na_values=["nan", "NaN"])
        dp = d[(d.harness == "h42") & (d.part == part)] if len(d) else d
        for (ax, lr), g in cnt.groupby(["axis", "learner"], sort=False):
            proj, used, n_meas = 0.0, set(), 0
            for _, r in g.iterrows():
                cands = [("축·학습기·방법", dp[(dp.axis == ax) & (dp.learner == lr) & (dp.method == r.method)]),
                         ("축·학습기", dp[(dp.axis == ax) & (dp.learner == lr)]), ("학습기", dp[dp.learner == lr])] if len(dp) else []
                c, lvl, src_name = np.nan, "없음", "없음"
                for name, sub in cands:
                    src, sn = pick_source(sub)
                    cc, nn = calib_of(src)
                    if np.isfinite(cc):
                        c, lvl, src_name = cc, name, sn
                        n_meas += nn if name == "축·학습기·방법" else 0
                        break
                used.add(f"{lvl}/{src_name}")
                proj = proj + c * float(r.est_h) if np.isfinite(c) else np.nan
            out.append(dict(harness="h42", part=part, axis=ax, learner=lr, n_fit_full=int(g.n_fit.sum()), est_h_full=float(g.est_h.sum()),
                            proj_h_full=proj, calib_mean=proj / float(g.est_h.sum()) if np.isfinite(proj) and g.est_h.sum() > 0 else np.nan,
                            n_fit_measured=int(n_meas), source="; ".join(sorted(used)), workers=int(workers),
                            proj_wall_h=proj / max(int(workers), 1) if np.isfinite(proj) else np.nan))
    return pd.DataFrame(out)


def project_h41(d: pd.DataFrame, lgx_dir: Path, a):
    f = lgx_dir / "ladder" / f"{a.ladder_count_tag}_count.csv"
    if not f.exists():
        return pd.DataFrame()
    cnt = pd.read_csv(f)
    dv = d[d.harness == "h41"] if len(d) else d
    c_all, _ = calib_of(dv) if len(dv) else (np.nan, 0)
    out = []
    for sc, g in cnt.groupby("scheme", sort=False):
        sub = dv[dv.axis == sc] if len(dv) else dv
        c, n_meas = calib_of(sub) if len(sub) else (np.nan, 0)
        src = "같은 단의 스모크" if np.isfinite(c) else ("전체 스모크" if np.isfinite(c_all) else "없음")
        c = c if np.isfinite(c) else c_all
        est_h = float(g.est_s.sum()) / 3600.0
        proj = c * est_h if np.isfinite(c) else np.nan
        out.append(dict(harness="h41", part="cpu", axis=sc, learner="(전체)", n_fit_full=int(g.fit_total.sum()) if "fit_total" in g else -1,
                        est_h_full=est_h, proj_h_full=proj, calib_mean=c, n_fit_measured=int(n_meas), source=src, workers=int(a.h41_workers),
                        proj_wall_h=proj / max(int(a.h41_workers), 1) if np.isfinite(proj) else np.nan))
    return pd.DataFrame(out)


def main(argv=None):
    a = parse_args(argv)
    lgx_dir, out_dir = Path(a.lgx_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = dict(torch=a.epochs_full, realmlp=a.realmlp_epochs_full)
    d1, bad1 = read_units(lgx_dir / "shards", "h42", a.tag, full)
    d2, bad2 = read_units(lgx_dir / "ladder" / "shards", "h41", a.tag, full)
    d3, bad3 = read_units(Path(a.lg_dir) / "shards", "h42", a.tag, full, lg_precheck=True)
    if len(d3):
        print(f"[timing] LG 사전 점검의 gpu 조각 {d3.unit.nunique()}개를 분할 4·5 축(g45)의 실측으로 읽었다", flush=True)
    for b in bad1 + bad2 + bad3:
        print(f"[timing] 제외: {b}", flush=True)
    frames = [x for x in (d1, d2, d3) if len(x)]
    if not frames:
        print(f"[timing] 적합 기록이 있는 조각이 없다: {lgx_dir}", flush=True)
        return 1
    d = pd.concat(frames, ignore_index=True)
    tab = by_learner(d)
    f1 = out_dir / f"{a.prefix}_timing_by_learner.csv"
    tab.to_csv(f1, index=False)
    lt = d.assign(sec_full=d.sec * (d.epochs_full / d.epochs_run).where(d.epochs_run > 0).fillna(1.0)).groupby(
        ["harness", "tag", "part", "axis", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), sec=("sec", "sum"),
                                                                           sec_full_epochs=("sec_full", "sum"), est_s=("est_s", "sum"),
                                                                           epochs_run=("epochs_run", "max"))
    lt["sec_per_fit"] = lt.sec / lt.n_fit.clip(lower=1)
    lt["calib"] = (lt.sec_full_epochs / lt.est_s.where(lt.est_s > 0))
    print(f"[timing] 조각 {d.unit.nunique()} · 행 {len(tab)} → {f1}", flush=True)
    print(lt.round(3).to_string(index=False), flush=True)
    pj = [x for x in (project_h42(d, lgx_dir, a), project_h41(d, lgx_dir, a)) if len(x)]
    if not pj:
        print(f"[timing] 적합 수 표({a.count_tag}_count_<부분>_detail.csv, ladder/{a.ladder_count_tag}_count.csv)가 없어 본 실행 환산을 생략한다", flush=True)
        return 0
    pj = pd.concat(pj, ignore_index=True)
    f2 = out_dir / f"{a.prefix}_full_projection.csv"
    pj.to_csv(f2, index=False)
    print(f"[timing] 본 실행 환산 → {f2} (누적 시간 h, 워커 수로 나눈 벽시계 h)", flush=True)
    print(pj.round(3).to_string(index=False), flush=True)
    for (hn, part), g in pj.groupby(["harness", "part"], sort=False):
        ok = g[np.isfinite(g.proj_h_full)]
        miss = sorted(set(zip(g.axis, g.learner)) - set(zip(ok.axis, ok.learner)))
        print(f"[timing] {hn} {part}: 환산 누적 {ok.proj_h_full.sum():.1f} h · 벽시계 {ok.proj_wall_h.sum():.1f} h(워커 {int(g.workers.iloc[0])}개) · "
              f"실측 없는 (축, 학습기) {miss}", flush=True)
    rm = pj[(pj.harness == "h42") & (pj.part == "gpu")]
    if len(rm):
        a_ = rm[rm.learner != "realmlp"].proj_h_full.sum(); b_ = rm[rm.learner == "realmlp"].proj_h_full.sum()
        print(f"[timing] h42 gpu: RealMLP 제외 {a_:.1f} GPU-h · RealMLP {b_:.1f} GPU-h(환산이 없는 항목은 0 으로 더했다)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
