#!/usr/bin/env python3
"""LG(H40) 적합 시간 표. 조각의 unit.json 에서 학습기별 적합 시간을 모으고, 적합 수 표가 있으면 본 실행 시간을 환산한다.

입력
  <lg-dir>/shards/*_unit.json         h40 조각의 완료 표지(n_fit_detail·sec_detail·rows_detail·est_detail, 키 = "축|학습기|방법")
  <lg-dir>/lg_count_{cpu,gpu}_learners.csv   본 실행 범위의 --count-only 결과(있을 때만 환산표를 만든다)

산출(<out-dir>)
  <prefix>_timing_by_learner.csv      (tag, part, learner, axis, method)별 적합 수·시간·학습 행 수·실행 epoch
  <prefix>_full_projection.csv        (part, learner)별 본 실행 누적 시간 환산. 적합 수 표가 없으면 만들지 않는다

환산식
  보정 계수 c = 실측 시간 합 / h40 추정 시간 합(est_detail, BASE_FIT 기준). 유사라벨 행을 넣는 방법(D1·R2)과 그 밖을 따로 구한다.
  epoch 환산 = 본 실행 epoch / 실측 epoch(신경망 학습기만, 이어 학습 축 제외). 조기 종료와 고정 비용을 무시하므로 상한에 가깝다.
  본 실행 누적 시간 = c × epoch 환산 × 적합 수 표의 est_h.
  사전 점검(tag 가 _precheck 로 끝남) 실측이 있으면 그것을 쓰고, 없으면 스모크 실측을 쓴다(출처 열에 적는다).

실행: python3 scripts/rescale/lg_timing_table.py --lg-dir data/processed/lg --out-dir . --prefix lg_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PSEUDO = ("D1", "R2")                       # 유사라벨 행을 학습 행렬에 넣는 방법(h40 count_only 의 n_fit_pseudo 와 같은 묶음)
TORCH = ("mlp", "realmlp", "ftt", "tabm", "cfm", "ddpm", "nflow")
NO_EPOCH_AXES = ("mlp_cont",)               # 이어 학습 축은 ft_epochs 를 쓴다(스모크에서도 본 실행과 같다)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LG 적합 시간 표와 본 실행 환산")
    ap.add_argument("--lg-dir", default="data/processed/lg")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prefix", default="lg_smoke")
    ap.add_argument("--count-tag", default="lg", help="적합 수 표의 tag(<tag>_count_<part>_learners.csv)")
    ap.add_argument("--epochs-full", type=int, default=100)
    ap.add_argument("--realmlp-epochs-full", type=int, default=256)
    ap.add_argument("--gpu-workers", type=int, default=12, help="본 실행 GPU 워커 수(GPU 4장 × 3)")
    ap.add_argument("--cpu-workers", type=int, default=14, help="본 실행 CPU 워커 수((코어 수 − 8) / 4)")
    return ap.parse_args(argv)


def epochs_of(cfg, learner, axis, full):
    """(실측 epoch, 본 실행 epoch). 해당 없음은 (nan, nan)."""
    if learner not in TORCH or axis in NO_EPOCH_AXES:
        return np.nan, np.nan
    if learner == "realmlp":
        return float(cfg.get("realmlp_epochs", np.nan)), float(full["realmlp"])
    return float(cfg.get("epochs", np.nan)), float(full["torch"])


def read_units(lg_dir: Path, full):
    rows, bad = [], []
    for p in sorted((lg_dir / "shards").glob("*_unit.json")):
        try:
            u = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            bad.append(f"{p.name}: {e!r}"); continue
        det = u.get("n_fit_detail") or {}
        if not det:                                                       # 개정 전 형식의 조각(키별 기록 없음)
            bad.append(f"{p.name}: n_fit_detail 없음"); continue
        cfg = u.get("cfg") or {}
        for k, nf in det.items():
            ax, lr, m = (k.split("|") + ["", ""])[:3]
            er, ef = epochs_of(cfg, lr, ax, full)
            rows.append(dict(tag=u.get("tag", ""), part=u.get("part", ""), target=u.get("target", ""), mode=u.get("mode", ""),
                             split=u.get("split", -1), unit=p.name[:-len("_unit.json")], axis=ax, learner=lr, method=m, n_fit=int(nf),
                             sec=float((u.get("sec_detail") or {}).get(k, 0.0)), rows=float((u.get("rows_detail") or {}).get(k, 0.0)),
                             est_s=float((u.get("est_detail") or {}).get(k, np.nan)), epochs_run=er, epochs_full=ef,
                             device=str(u.get("device", "")), threads=u.get("threads", ""), status=str(u.get("status", "")),
                             unit_elapsed_s=float(u.get("elapsed_s", np.nan))))
    return pd.DataFrame(rows), bad


def by_learner(d: pd.DataFrame):
    g = d.groupby(["tag", "part", "learner", "axis", "method"], as_index=False, sort=True).agg(
        n_units=("unit", "nunique"), n_fit=("n_fit", "sum"), sec=("sec", "sum"), rows=("rows", "sum"), est_s=("est_s", "sum"),
        epochs_run=("epochs_run", "max"), epochs_full=("epochs_full", "max"), device=("device", lambda s: ",".join(sorted(set(s)))),
        threads=("threads", "max"))
    g["sec_per_fit"] = g.sec / g.n_fit.clip(lower=1)
    g["rows_per_fit"] = g.rows / g.n_fit.clip(lower=1)
    g["calib"] = np.where(g.est_s > 0, g.sec / g.est_s.where(g.est_s > 0), np.nan)
    g["epoch_scale"] = (g.epochs_full / g.epochs_run).where(g.epochs_run > 0)
    g["sec_per_fit_full_epochs"] = g.sec_per_fit * g.epoch_scale.fillna(1.0)
    cols = ["tag", "part", "learner", "axis", "method", "n_units", "n_fit", "sec", "sec_per_fit", "rows_per_fit", "est_s", "calib",
            "epochs_run", "epochs_full", "epoch_scale", "sec_per_fit_full_epochs", "device", "threads"]
    return g[cols].drop(columns=["rows"], errors="ignore")


def _calib(d: pd.DataFrame):
    """(보정 계수, epoch 환산 가중 평균). 실측 시간에 epoch 환산을 먼저 곱한 뒤 추정 시간으로 나눈다."""
    d = d[(d.est_s > 0) & (d.n_fit > 0)]
    if not len(d):
        return np.nan, 0
    sc = (d.epochs_full / d.epochs_run).where(d.epochs_run > 0).fillna(1.0)
    return float((d.sec * sc).sum() / d.est_s.sum()), int(d.n_fit.sum())


def projection(d: pd.DataFrame, lg_dir: Path, a):
    out = []
    for part, workers in (("cpu", a.cpu_workers), ("gpu", a.gpu_workers)):
        f = lg_dir / f"{a.count_tag}_count_{part}_learners.csv"
        if not f.exists():
            continue
        cnt = pd.read_csv(f)
        dp = d[d.part == part]
        for _, r in cnt.iterrows():
            dl = dp[dp.learner == r.learner]
            pre = dl[dl.tag.astype(str).str.endswith("_precheck")]
            src = pre if len(pre) else dl
            src_name = "없음" if not len(src) else ("사전 점검" if len(pre) else "스모크(epoch 환산, 상한)")
            c_all, n_all = _calib(src)
            c_ps, n_ps = _calib(src[src.method.isin(PSEUDO)])
            c_np, n_np = _calib(src[~src.method.isin(PSEUDO)])
            est_ps = float(r.get("est_h_pseudo", 0.0) or 0.0); est_np = float(r.est_h) - est_ps
            use_ps = c_ps if np.isfinite(c_ps) else c_all
            use_np = c_np if np.isfinite(c_np) else c_all
            proj = use_np * est_np + use_ps * est_ps if np.isfinite(c_all) else np.nan
            out.append(dict(part=part, learner=r.learner, source=src_name, n_fit_measured=n_all, n_fit_measured_pseudo=n_ps,
                            calib_all=c_all, calib_pseudo=c_ps, calib_other=c_np, n_fit_full=int(r.n_fit), est_h_full=float(r.est_h),
                            est_h_full_pseudo=est_ps, proj_h_full=proj, proj_h_full_pseudo=use_ps * est_ps if np.isfinite(use_ps) else np.nan,
                            workers=int(workers), proj_wall_h=proj / max(int(workers), 1) if np.isfinite(proj) else np.nan))
    return pd.DataFrame(out)


def main(argv=None):
    a = parse_args(argv)
    lg_dir, out_dir = Path(a.lg_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d, bad = read_units(lg_dir, dict(torch=a.epochs_full, realmlp=a.realmlp_epochs_full))
    for b in bad:
        print(f"[timing] 제외: {b}", flush=True)
    if not len(d):
        print(f"[timing] 키별 기록이 있는 조각이 없다: {lg_dir / 'shards'}", flush=True)
        return 1
    tab = by_learner(d)
    f1 = out_dir / f"{a.prefix}_timing_by_learner.csv"
    tab.to_csv(f1, index=False)
    lt = d.assign(sec_full=d.sec * (d.epochs_full / d.epochs_run).where(d.epochs_run > 0).fillna(1.0)).groupby(
        ["tag", "part", "learner"], as_index=False).agg(n_fit=("n_fit", "sum"), sec=("sec", "sum"), sec_full_epochs=("sec_full", "sum"),
                                                        epochs_run=("epochs_run", "max"))
    lt["sec_per_fit"] = lt.sec / lt.n_fit.clip(lower=1)
    print(f"[timing] 조각 {d.unit.nunique()} · 행 {len(tab)} → {f1}", flush=True)
    print(lt.round(3).to_string(index=False), flush=True)
    pj = projection(d, lg_dir, a)
    if len(pj):
        f2 = out_dir / f"{a.prefix}_full_projection.csv"
        pj.to_csv(f2, index=False)
        print(f"[timing] 본 실행 환산 → {f2} (누적 시간 h, 워커 수로 나눈 벽시계 h)", flush=True)
        print(pj[["part", "learner", "source", "n_fit_measured", "calib_all", "n_fit_full", "est_h_full", "proj_h_full", "workers",
                  "proj_wall_h"]].round(3).to_string(index=False), flush=True)
        for part, g in pj.groupby("part"):
            ok = g[np.isfinite(g.proj_h_full)]
            print(f"[timing] 부분 {part}: 환산 누적 {ok.proj_h_full.sum():.1f} h · 벽시계 {ok.proj_wall_h.sum():.1f} h"
                  f"(워커 {int(g.workers.iloc[0])}개) · 실측 없는 학습기 {sorted(set(g.learner) - set(ok.learner))}", flush=True)
    else:
        print(f"[timing] 적합 수 표({a.count_tag}_count_<part>_learners.csv)가 없어 본 실행 환산을 생략한다", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
