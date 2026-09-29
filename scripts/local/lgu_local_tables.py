"""LGU 로컬 실행 보조 표. scripts/local/run_lgu_local.sh 가 부른다.

표준 라이브러리만 쓴다(numpy, pandas, torch 를 적재하지 않으므로 CPU 스레드 규칙에 걸리지 않는다).
계획: docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 8.1절(로컬 실행 규칙)과 2.9절(사전 점검 뒤 확정 항목).

결과 비열람 규칙(계획서 6절)
  이 파일은 점수, 대비, 판정, 곡선 표를 읽지 않는다. 읽는 것은 다음뿐이다.
  - 조각 unit.json 의 status 값(개수만 센다)
  - 시간표 lgu_a_<접미사>_timing.csv 의 boot 행(sec, note), 적합 수 표 lgu_a_count.csv(n_eval, n_nd)
  - 정밀도 표 lgu_a_precheck_precision.csv 의 test, half_width_rel, precheck_rule_10pct 열(반폭 비율만. 계획서 2.9절)
  - 구적 점검 표 lgu_a_precheck_quadrature.csv(합성 자료)
  - 실험 B fit 조각 runs.csv 의 sec, n_pred, failed, fail_reason, flag, device 열과 unit.json 의 status, error, cfg.s_gen
  - 실행 중 스레드 등록부 data/processed/lgu/.running/*.json

하위 명령
  status     단계 기록(TSV)과 조각의 unit.json status 개수로 상태 표(lgu_status.csv)를 쓴다. 같은 mode·단계의 이전 행은 새 행으로 바꾼다.
  heartbeat  조각 완료 수 한 줄(진행 줄용).
  registry   실행 중 등록부의 살아 있는 항목과 스레드 합(죽은 항목은 세지 않는다. 파일은 지우지 않는다).
  decide     계획서 2.9절의 확정 규칙을 적용해 lgu_<단계>_decisions.csv 를 쓴다(--stage precheck 또는 smoke).
             사전 점검(precheck)이 계획서가 정한 측정이다. 스모크(smoke) 확정표는 사전 점검을 하지 못했을 때의 대체값이다(계획서 개정 2).
  params     --full 이 쓸 값을 'NBOOT=<값> SGEN=<값> SRC=<출처>' 한 줄로 출력한다. 항목마다 사전 점검 확정표, 스모크 확정표,
             계획서 기본값의 순서로 숫자 결정이 있는 첫 표를 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

STEP_KINDS = {                      # 단계 → (실험, 조각 종류). None 은 그 실험 태그의 모든 조각
    "a_cpu": ("A", ("emu", "cpu")), "a_gate": ("A", ("gate",)), "a_flow": ("A", ("flow",)),
    "b_fit": ("B", ("fit",)), "b_score": ("B", ("score",)), "c_diag": ("C", None),
}
TAG_BASE = {"A": "lgua", "B": "lgub", "C": "lguc"}
STATUS_COLS = ["mode", "run_id", "step", "harness", "device", "threads", "workers", "status", "exit_code", "started", "ended",
               "wall_s", "units_ok", "units_partial", "units_failed", "units_other", "log", "command"]
DECIDE_COLS = ["item", "default", "measured", "rule", "decision", "source", "note"]

# 계획서 2.9절의 확정 규칙 상수
CFM_SEC_MAX = 120.0                 # cfm 적합 1건의 표본 추출 시간 상한(s)
BOOT_UNIT_MAX_S = 1800.0            # 2단 재표집의 작업 단위당 시간 상한(s)
QUAD_MAX = 0.002                    # 구적 격자: 닫힌 형태와 격자 해의 분위 최대 절대 차 상한
STAGE_COMBOS_SPLIT1 = {             # 분할 1 의 (n, 추출) 조합 수(h44 finalize 의 설정): n = 0 은 추출 1개
    "precheck": 1 + 2 * 2,          # n {0, 10, 40}, 추출 2
    "smoke": 1 + 3 * 1,             # n {0, 3, 10, 40}, 추출 1
}
DEFAULTS = dict(nboot=2000, s_gen=512, h_grid=0.005)


def sfx_of(mode: str) -> str:
    return {"smoke": "_smoke", "precheck": "_precheck"}.get(mode, "")


def tag_of(exp: str, mode: str) -> str:
    return TAG_BASE[exp] + sfx_of(mode)


def _read_status(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except (OSError, ValueError):
        return "unreadable"


def unit_counts(shards: Path, tag: str, kinds=None) -> dict:
    """shards/<tag>__<종류>__..._unit.json 의 status 개수. kinds 가 None 이면 그 태그의 모든 unit.json."""
    out = dict(ok=0, partial=0, failed=0, other=0)
    if not shards.is_dir():
        return out
    for p in shards.glob(f"{tag}__*_unit.json"):
        rest = p.name[len(tag) + 2:]
        kind = rest.split("__", 1)[0]
        if kinds is not None and kind not in kinds:
            continue
        st = _read_status(p)
        out[st if st in ("ok", "partial", "failed") else "other"] += 1
    return out


def _atomic_csv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_csv(path: Path) -> list:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- status
def cmd_status(a) -> int:
    shards = Path(a.shards)
    rows_new = []
    for r in _read_csv(Path(a.tsv)) if a.tsv.endswith(".csv") else _read_tsv(Path(a.tsv)):
        st = r.get("step", "")
        exp_kinds = STEP_KINDS.get(st)
        cnt = unit_counts(shards, tag_of(exp_kinds[0], a.mode), exp_kinds[1]) if exp_kinds else {}
        rows_new.append(dict(r, mode=a.mode, run_id=a.run_id, units_ok=cnt.get("ok", ""), units_partial=cnt.get("partial", ""),
                             units_failed=cnt.get("failed", ""), units_other=cnt.get("other", "")))
    out = Path(a.out)
    steps_new = {r.get("step") for r in rows_new}
    keep = [r for r in _read_csv(out) if not (r.get("mode") == a.mode and r.get("step") in steps_new)]   # 같은 모드·단계의 이전 행만 바꾼다
    _atomic_csv(out, STATUS_COLS, keep + rows_new)
    return 0


def _read_tsv(path: Path) -> list:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh, delimiter="\t"))
    except OSError:
        return []


# ---------------------------------------------------------------- heartbeat
def cmd_heartbeat(a) -> int:
    shards = Path(a.shards)
    parts = []
    for exp, kinds in (("A", ("emu", "cpu", "gate", "flow")), ("B", ("fit", "score")), ("C", None)):
        tag = tag_of(exp, a.mode)
        if kinds is None:
            c = unit_counts(shards, tag)
            parts.append(f"C 완료 {c['ok'] + c['partial']} 실패 {c['failed']}")
            continue
        segs = []
        for k in kinds:
            c = unit_counts(shards, tag, (k,))
            segs.append(f"{k} {c['ok'] + c['partial'] + c['other']}" + (f"(실패 {c['failed']})" if c["failed"] else ""))
        parts.append(f"{exp} " + " ".join(segs))
    print(" · ".join(parts))
    return 0


# ---------------------------------------------------------------- registry
def _pid_alive(pid: int, start: str = "") -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except (OSError, ValueError):
        return False
    if start:
        try:
            txt = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
            return txt.rpartition(")")[2].split()[19] == str(start)
        except (OSError, IndexError):
            return True
    return True


def cmd_registry(a) -> int:
    d = Path(a.dir)
    live, tot = [], 0
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _pid_alive(int(r.get("pid", -1)), str(r.get("start", ""))):
            live.append(r)
            tot += int(r.get("threads", 0))
    print(f"THREADS={tot} N={len(live)} " + ";".join(f"{r.get('tag')}:pid{r.get('pid')}:thr{r.get('threads')}" for r in live))
    return 0


# ---------------------------------------------------------------- decide(사전 점검 뒤 계획서 2.9절)
def _decide_cfm(out_dir: Path, stage: str) -> dict:
    shards = out_dir / "shards"
    secs, preds, mem_err, n_rows, devs = [], [], False, 0, set()
    s_gen = set()
    for p in sorted(glob.glob(str(shards / f"lgub_{stage}__fit__*__x__cfm_runs.csv"))):
        for r in _read_csv(Path(p)):
            n_rows += 1
            s = _f(r.get("sec"))
            if s == s:
                secs.append(s); preds.append(int(_f(r.get("n_pred"), 0)))
            devs.add(str(r.get("device", "")))
            if "out of memory" in str(r.get("fail_reason", "")).lower():
                mem_err = True
    for p in sorted(glob.glob(str(shards / f"lgub_{stage}__fit__*__x__cfm_unit.json"))):
        try:
            u = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s_gen.add(str((u.get("cfg") or {}).get("s_gen", "")))
        if u.get("status") == "failed" and "memory" in str(u.get("error", "")).lower():
            mem_err = True
    rule = "적합 1건 추출 시간 > 120 s 또는 메모리 오류 → 256(측정값 sec 는 적합 시간을 포함하므로 보수적이다). GPU, 셀당 표본 512 의 측정만 쓴다"
    src = f"shards/lgub_{stage}__fit__*__cfm_runs.csv(sec, device), _unit.json(cfg.s_gen)"
    if not secs and not mem_err:
        return dict(item="cfm_s_gen", default=DEFAULTS["s_gen"], measured="없음", rule=rule, decision=f"미정({stage} 조각 없음)", source=src, note="")
    i = max(range(len(secs)), key=lambda j: secs[j]) if secs else -1
    mx = secs[i] if secs else float("nan")
    meas = (f"적합+추출 최대 {mx:.1f} s(n_pred {preds[i] if secs else 'NA'}, 적합 {n_rows}건, 장치 {','.join(sorted(devs))}, "
            f"셀당 표본 {','.join(sorted(s_gen))}), 메모리 오류 {'있음' if mem_err else '없음'}")
    gpu = bool(devs) and all(d.startswith("cuda") for d in devs)
    if not gpu or s_gen != {str(DEFAULTS["s_gen"])}:
        return dict(item="cfm_s_gen", default=DEFAULTS["s_gen"], measured=meas, rule=rule,
                    decision="미정(GPU·셀당 표본 512 측정이 아니다)", source=src, note="CPU 측정은 규칙에 쓰지 않는다")
    dec = 256 if (mem_err or (mx == mx and mx > CFM_SEC_MAX)) else 512
    return dict(item="cfm_s_gen", default=DEFAULTS["s_gen"], measured=meas, rule=rule, decision=str(dec), source=src, note="")


def _decide_nboot(out_dir: Path, stage: str) -> dict:
    tim = _read_csv(out_dir / f"lgu_a_{stage}_timing.csv")
    rows = [r for r in tim if r.get("kind") == "boot" and r.get("target") == "Canada"]
    cnt = {r.get("target"): r for r in _read_csv(out_dir / "lgu_a_count.csv")}
    rule = ("캐나다 x 분할 1 의 2단 CI 시간을 본 실행의 알래스카 작업 단위(대상 하나의 2단 재표집: 분할 5, (n, 추출) 조합 n_nd, 재표집 2,000)로 "
            "선형 환산: t × (2,000 / 측정 재표집 수) × (알래스카 채점 셀 수 / 캐나다 채점 셀 수) × (알래스카 n_nd / 측정 조합 수). 1,800 s 초과 → 1,000")
    src = f"lgu_a_{stage}_timing.csv(kind boot, sec, note), lgu_a_count.csv(n_eval, n_nd)"
    t = _f(rows[0].get("sec")) if rows else float("nan")
    if t != t:
        return dict(item="nboot_2stage", default=DEFAULTS["nboot"], measured="없음", rule=rule, decision=f"미정({stage} 시간표 없음)", source=src, note="")
    m = re.search(r"재표집 (\d+)", str(rows[0].get("note", "")))
    nb_meas = int(m.group(1)) if m else DEFAULTS["nboot"]
    ca, ak = cnt.get("Canada"), cnt.get("Alaska")
    if not ca or not ak:
        return dict(item="nboot_2stage", default=DEFAULTS["nboot"], measured=f"캐나다 {t:.1f} s(재표집 {nb_meas})", rule=rule,
                    decision="미정(lgu_a_count.csv 없음. h44 --count-only 를 먼저 실행한다)", source=src, note="")
    r_nb = DEFAULTS["nboot"] / max(nb_meas, 1)
    r_cell = _f(ak.get("n_eval")) / max(_f(ca.get("n_eval")), 1.0)
    r_combo = _f(ak.get("n_nd")) / STAGE_COMBOS_SPLIT1[stage]
    proj = t * r_nb * r_cell * r_combo
    dec = 1000 if proj > BOOT_UNIT_MAX_S else 2000
    return dict(item="nboot_2stage", default=DEFAULTS["nboot"],
                measured=(f"캐나다 분할 1: {t:.1f} s(재표집 {nb_meas}, 조합 {STAGE_COMBOS_SPLIT1[stage]}) · 재표집 비 {r_nb:.0f} · 셀 비 {r_cell:.2f} · "
                          f"조합 비 {r_combo:.2f} · 알래스카 작업 단위 환산 {proj:.0f} s"),
                rule=rule, decision=str(dec), source=src,
                note="선형 환산은 고정 비용까지 늘리므로 보수적이다. 하네스의 --nboot 하나가 실험 A·B 의 1단·2단 재표집에 함께 쓰인다")


def _decide_precision(out_dir: Path, stage: str) -> dict:
    rows = [r for r in _read_csv(out_dir / "lgu_a_precheck_precision.csv") if r.get("test") == "LGU-A1"] if stage == "precheck" else []
    if not rows:
        return dict(item="a1_equivalence", default="주 한계 5 %", measured="없음", rule="LGU-A1 반폭 비율 > 10 % → 본 실행 전 동등 판정 검정 불가 등록",
                    decision="미정(사전 점검 정밀도 표 없음)", source="lgu_a_precheck_precision.csv",
                    note="스모크는 정밀도 표를 만들지 않는다(대상에 레나·알래스카가 없다)")
    flags = [str(r.get("precheck_rule_10pct", "")) for r in rows]
    rel = [_f(r.get("half_width_rel")) for r in rows]
    rel_ok = [v for v in rel if v == v]
    meas = f"행 {len(rows)} · 반폭 비율 최대 {max(rel_ok):.3f}" if rel_ok else f"행 {len(rows)} · 반폭 비율 없음"
    if any(f.startswith("본 실행 전") for f in flags):
        dec = "등록(LGU-A1 동등 판정 검정 불가)"
    elif flags and all(f == "등록 없음" for f in flags):
        dec = "등록 없음(동등 판정 가능)"
    else:
        dec = "판정 불가(정밀도 표에 비유한 행)"
    return dict(item="a1_equivalence", default="주 한계 5 %", measured=meas, rule="LGU-A1 반폭 비율 > 10 % → 본 실행 전 동등 판정 검정 불가 등록",
                decision=dec, source="lgu_a_precheck_precision.csv(half_width_rel, precheck_rule_10pct)",
                note="본 실행의 판정은 2.7절 정밀도 규칙(주 한계 5 %)을 그대로 쓴다")


def _decide_quad(out_dir: Path, stage: str) -> dict:
    rows = {r.get("check"): r for r in _read_csv(out_dir / "lgu_a_precheck_quadrature.csv")} if stage == "precheck" else {}
    cf = rows.get("closed_form_max_abs_diff")
    if not cf:
        return dict(item="quad_grid", default=DEFAULTS["h_grid"], measured="없음", rule="분위 최대 절대 차 > 0.002 → 0.0025",
                    decision="미정(구적 점검 표 없음)", source="lgu_a_precheck_quadrature.csv",
                    note="단위 시험 test_h44_ladder.py::test_c 가 같은 계산(quadrature_check)을 한다(계획서 개정 2)")
    v = _f(cf.get("value"))
    sbc = "; ".join(f"{k} {_f(r.get('value')):.4f}({r.get('threshold')}, ok={r.get('ok')})" for k, r in rows.items() if k != "closed_form_max_abs_diff")
    return dict(item="quad_grid", default=DEFAULTS["h_grid"], measured=f"닫힌 형태 대비 최대 절대 차 {v:.2e}", rule="분위 최대 절대 차 > 0.002 → 0.0025",
                decision=str(0.0025 if v > QUAD_MAX else 0.005), source="lgu_a_precheck_quadrature.csv", note=sbc)


def _decide_cbq(out_dir: Path, stage: str) -> dict:
    flags = []
    for p in sorted(glob.glob(str(out_dir / "shards" / f"lgub_{stage}__fit__*__x__cbq_runs.csv"))):
        flags += [str(r.get("flag", "")) for r in _read_csv(Path(p))]
    if not flags:
        return dict(item="cbq_loss", default="MultiQuantile", measured="없음", rule="동작하지 않으면 분위별 Quantile 손실(적합 수 5배)",
                    decision=f"미정({stage} 조각 없음)", source=f"lgub_{stage} fit cbq", note="")
    multi = sum("multiquantile" in f for f in flags)
    dec = "MultiQuantile" if multi == len(flags) else "분위별 Quantile(일부 또는 전부)"
    return dict(item="cbq_loss", default="MultiQuantile", measured=f"적합 {len(flags)}건 중 MultiQuantile {multi}건",
                rule="동작하지 않으면 분위별 Quantile 손실(적합 수 5배)", decision=dec, source=f"shards/lgub_{stage}__fit__*__cbq_runs.csv(flag)", note="")


def cmd_decide(a) -> int:
    out_dir, stage = Path(a.out_dir), a.stage
    rows = [_decide_cfm(out_dir, stage), _decide_nboot(out_dir, stage), _decide_precision(out_dir, stage), _decide_quad(out_dir, stage),
            _decide_cbq(out_dir, stage),
            dict(item="wrapper_repro", default="최대 절대 차 1e-5 이하", measured="단위 시험 test_lgu_common.py::test_k", rule="1e-5 초과면 구현을 고친다",
                 decision="단위 시험 결과를 따른다", source="pytest", note="이 표의 작업에서는 재지 않는다")]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for r in rows:
        r["note"] = (r.get("note", "") + f" [{stage} 확정표 작성 {stamp}]").strip()
    path = out_dir / f"lgu_{stage}_decisions.csv"
    _atomic_csv(path, DECIDE_COLS, rows)
    for r in rows:
        print(f"[decide] {r['item']}: {r['decision']} · 측정 {r['measured']}")
    print(f"[decide] → {path}")
    return 0


def cmd_params(a) -> int:
    tabs = [(st, {r.get("item"): r for r in _read_csv(Path(a.out_dir) / f"lgu_{st}_decisions.csv")}) for st in ("precheck", "smoke")]
    out, src = {}, []
    for item, key in (("nboot_2stage", "nboot"), ("cfm_s_gen", "s_gen")):
        val, where = DEFAULTS[key], "default"
        for st, rows in tabs:
            d = str(rows.get(item, {}).get("decision", ""))
            if d.isdigit():
                val, where = int(d), st
                break
        out[key] = val
        src.append(f"{key}:{where}")
    print(f"NBOOT={out['nboot']} SGEN={out['s_gen']} SRC={','.join(src)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LGU 로컬 실행 보조 표(run_lgu_local.sh)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("--tsv", required=True); s.add_argument("--out", required=True)
    s.add_argument("--shards", required=True); s.add_argument("--mode", required=True); s.add_argument("--run-id", required=True)
    h = sub.add_parser("heartbeat"); h.add_argument("--shards", required=True); h.add_argument("--mode", required=True)
    r = sub.add_parser("registry"); r.add_argument("--dir", required=True)
    d = sub.add_parser("decide"); d.add_argument("--out-dir", required=True); d.add_argument("--stage", choices=["precheck", "smoke"], required=True)
    p = sub.add_parser("params"); p.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return dict(status=cmd_status, heartbeat=cmd_heartbeat, registry=cmd_registry, decide=cmd_decide, params=cmd_params)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
