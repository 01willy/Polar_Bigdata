#!/bin/bash
# WF(H54) Rescale 작업. 계획 docs/EXPERIMENT_PLAN_WF_2026-10-01.md §4.
# 설정: configs/rescale/wf_smoke.yaml(WF_MODE=smoke, 사전 점검), configs/rescale/wf_full.yaml(WF_MODE=full, 본 실행).
# CPU 전용 노드(elm, AMD EPYC Milan 96코어)에서 돈다. 로컬 공유 서버에서는 실행하지 않는다(저장소(.git) 안에서는 거부한다).
#
# WF_MODE=smoke (단계마다 종료 코드를 wf_smoke_status.csv 에 남긴다. 필수 = 1 인 단계가 실패하면 종료 코드 1)
#   1 env, inputs, deps   환경 출력, 입력 확인, 필수 패키지 설치(lg_common.sh 의 lg_install_deps), 선택 패키지 pytest·pykrige
#   2 smoke               WF_RESCALE=1 h54 --smoke --workers 4 --threads 4 (실험 4종의 모든 경로, 대상 축소, 분할 1)
#   3 pytest              LG_RUN_HEAVY=1 python3 -m pytest -q tests/test_h54_workflow.py
#                         (WF_RESCALE·LG_RESCALE 는 지우고 실행한다. 시험 (i)가 허용 표지 없는 본 실행의 거부를 확인한다)
#   4 count               h54 --count-only (본 실행 범위의 적합 수와 추정 시간. 환산의 분모)
#   5 precheck            WF_RESCALE=1 h54 --precheck --workers 7 --threads 4 (실험마다 본 실행 크기의 대표 단위, 분할 1)
#   6 project             사전 점검 단위의 실측 시간과 같은 단위의 추정 시간의 비로 본 실행 누적 시간을 환산한다(wf_smoke_projection.csv)
#   결과 묶음 results_wf_smoke.tar.gz 는 단계마다 다시 쓴다(벽시계 상한에 걸려도 앞 단계 결과가 남는다)
# WF_MODE=full
#   1 env, inputs, deps
#   2 이전 결과 묶음(results_wf_prev.tar.gz 또는 results_wf.tar.gz)이 입력에 있으면 풀어 조각을 복원한다(--resume 이 건너뛴다)
#   3 WF_RESCALE=1 h54 --workers W --threads T --resume --no-summarize (시간 한도 = 예산 − 집계 예비, timeout 이 워커까지 끝낸다)
#   4 5분마다 진행 줄, 30분마다 results_wf.tar.gz 갱신(임시 파일에 쓴 뒤 교체)
#   5 WF_RESCALE=1 h54 --summarize-only --workers W --threads T (재표집 10,000회), 마지막 묶음, 정리
#
# 환경 변수(설정 파일의 command 에서 준다)
#   WF_MODE             smoke | full. 기본 full
#   WF_BUDGET_MIN       스크립트 시작부터 쓸 시간(분). 기본 smoke 50, full 210
#   WF_SUM_RESERVE_MIN  집계와 묶음에 남길 시간(분, full). 기본 50
#   WF_THREADS          워커당 스레드. 기본 4         WF_WORKERS   워커 수. 기본 (코어 수 − 8) / 스레드
#   WF_EXPS             실험 목록. 기본 wf1,wf2,wf3,wf4
#   WF_KEEP_TREE        1 이면 끝난 뒤 푼 코드와 입력을 지우지 않는다. 기본 0
# 종료 코드: 0 = 모든 단계 통과, 1 = 필수 단계 실패 또는 시간 한도로 중단, 2 = 선택 단계만 실패

main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then                                                    # 끝에 푼 코드와 입력을 지우므로 저장소 안에서는 실행하지 않는다
    echo "[wf] 저장소(.git)가 있는 디렉터리다. 이 스크립트는 Rescale 작업 디렉터리에서만 실행한다"; exit 1
  fi
  if [ -f wf_payload.tar.gz ] && [ ! -f data/processed/fidelity_base_v3.csv ]; then
    tar xzf wf_payload.tar.gz || { echo "[wf] 묶음 해제 실패"; exit 1; }
  fi
  # shellcheck source=scripts/rescale/lg_common.sh
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env

  local MODE T0 BUDGET_S SUM_RESERVE PACK_EVERY PROG_EVERY NCORE THREADS WORKERS EXPS H54 WFD LOGD STATUS OUTB
  MODE="${WF_MODE:-full}"
  T0=$(date +%s)
  if [ "$MODE" = "smoke" ]; then BUDGET_S=$(( ${WF_BUDGET_MIN:-50} * 60 )); else BUDGET_S=$(( ${WF_BUDGET_MIN:-210} * 60 )); fi
  SUM_RESERVE=$(( ${WF_SUM_RESERVE_MIN:-50} * 60 ))
  PACK_EVERY=$(( ${WF_PACK_MIN:-30} * 60 ))
  PROG_EVERY=$(( ${WF_PROGRESS_MIN:-5} * 60 ))
  NCORE=$(lg_ncores)
  THREADS="${WF_THREADS:-4}"
  WORKERS="${WF_WORKERS:-$(( (NCORE - 8) / THREADS ))}"
  [ "$WORKERS" -ge 1 ] || WORKERS=1
  EXPS="${WF_EXPS:-wf1,wf2,wf3,wf4}"
  H54=scripts/3_deep_learning/h54_workflow.py
  WFD=data/processed/wf
  if [ "$MODE" = "smoke" ]; then LOGD=logs_wf_smoke; STATUS=wf_smoke_status.csv; OUTB=results_wf_smoke.tar.gz
  else LOGD=logs_wf; STATUS=wf_status.csv; OUTB=results_wf.tar.gz; fi
  mkdir -p "$LOGD" "$WFD"
  echo "stage,required,rc,sec,note" > "$STATUS"
  [ -f wf_payload_info.txt ] && sed 's/^/[payload] /' wf_payload_info.txt

  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
  pack() { lg_pack "$OUTB" "$WFD" "$LOGD" "$STATUS" wf_smoke_projection.csv wf_payload_info.txt; }
  note() { echo "$1,$2,$3,$(( $(date +%s) - T0 )),$4" >> "$STATUS"; }
  # run_stage <이름> <필수 0|1> <제한 시간 s> <명령...>
  run_stage() {
    local name="$1" req="$2" lim="$3"; shift 3
    local t0 rc msg=""
    t0=$(date +%s)
    if [ "$lim" -lt 30 ]; then
      note "$name" "$req" NA "남은 시간 부족(${lim} s)"; lg_log "[stage] ${name} 생략: 남은 시간 부족"; return 0
    fi
    lg_log "[stage] ${name} 시작(제한 ${lim} s): $*"
    timeout -k 30 "$lim" "$@" > "$LOGD/${name}.log" 2>&1; rc=$?
    [ "$rc" -eq 124 ] && msg="시간 제한 ${lim} s 로 중단"
    echo "${name},${req},${rc},$(( $(date +%s) - t0 )),${msg}" >> "$STATUS"
    grep -a -E "\[FAIL\]|\[warn\]|\[pool\]|Traceback|^\[(data|plan|done|count-only|summarize)\]|passed|failed" "$LOGD/${name}.log" | head -n 40 \
      | sed "s/^/  [${name}] /"
    tail -n 6 "$LOGD/${name}.log" | sed "s/^/  [${name}] /"
    lg_log "[stage] ${name} 끝: 종료 코드 ${rc} · $(( $(date +%s) - t0 )) s · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    pack
    return 0
  }
  finish() {
    pack
    lg_log "[wf] 단계별 상태"
    sed 's/^/  /' "$STATUS"
    if [ "${WF_KEEP_TREE:-0}" != "1" ]; then                             # 푼 코드와 입력은 출력이 아니다(출력 동기화 절약)
      rm -rf src scripts tests .pydeps .lg_pip_constraints.txt wf_payload.tar.gz results_wf_prev.tar.gz \
             data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/e5_soil_tdd_v4.csv \
             data/processed/lg_subregion_map_v1.csv data/processed/lgd
      if [ -f "$OUTB" ] && gzip -t "$OUTB" 2>/dev/null \
         && [ "$(tar tzf "$OUTB" | grep -c '_unit\.json$')" -eq "$(find "$WFD/shards" -name '*_unit.json' 2>/dev/null | wc -l)" ]; then
        rm -rf "$WFD/shards"                                              # 조각은 묶음에 있다. 집계 표(wf_*.csv)는 남긴다
      else
        lg_log "[wf] 묶음 확인 실패 또는 조각 없음. 조각 디렉터리를 남긴다"
      fi
    fi
    local bad
    bad=$(awk -F, 'NR > 1 && $2 == 1 && $3 != 0 {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then lg_log "[wf] 실패한 필수 단계: $bad"; exit 1; fi
    bad=$(awk -F, 'NR > 1 && $2 == 0 && $3 != 0 && $3 != "NA" {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then lg_log "[wf] 필수 단계는 통과했다. 실패한 선택 단계: $bad"; exit 2; fi
    lg_log "[wf] 모든 단계 통과 · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    exit 0
  }

  # ---------------------------------------------------------------- 1 환경·입력·의존성
  lg_env_report 2>&1 | tee "$LOGD/env.log"
  local f miss=0
  for f in "$H54" scripts/3_deep_learning/h40_label_grid.py scripts/3_deep_learning/h42_label_grid_ext.py src/polar/m1_core.py \
           src/polar/h4_common.py src/polar/m1_stats.py data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv \
           data/processed/lg_subregion_map_v1.csv tests/test_h54_workflow.py; do
    [ -f "$f" ] || { lg_log "[input] 없음: $f"; miss=1; }
  done
  for f in Tibet NAtlantic_lic Russia_C; do                               # LGD 새 지역 실행 표(없으면 wf4 가 그 대상을 건너뛰고 기록한다)
    [ -f "data/processed/lgd/run_tables/$f/fidelity_base_v3.csv" ] && [ -f "data/processed/lgd/run_tables/$f/e5_soil_tdd_v3.csv" ] \
      || lg_log "[input] 경고: LGD 실행 표 $f 가 없다(wf4 대상에서 빠진다)"
  done
  note inputs 1 "$miss" ""
  [ "$miss" -eq 0 ] || finish
  if lg_install_deps 2>&1 | tee "$LOGD/deps.log"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    note deps 1 1 "필수 패키지 설치 실패"; finish
  fi
  note deps 1 0 ""
  [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
  lg_log "[wf] 모드 ${MODE} · 코어 ${NCORE} · 워커 ${WORKERS} × 스레드 ${THREADS} · 실험 ${EXPS} · 시간 한도 $(( BUDGET_S / 60 )) min"

  if [ "$MODE" = "smoke" ]; then
    lg_install_opt pytest "pytest" 2>&1 | tee -a "$LOGD/deps.log"
    lg_install_opt pykrige "pykrige==1.7.3" "pykrige" 2>&1 | tee -a "$LOGD/deps.log"   # 시험 (f)의 대조 기준(없으면 그 시험만 건너뛴다)
    [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
    local lim
    lim=$(( $(left) - 900 )); [ "$lim" -gt 900 ] && lim=900
    run_stage smoke 1 "$lim" env WF_RESCALE=1 python3 -u "$H54" --smoke --exps "$EXPS" --workers 4 --threads "$THREADS"
    lim=$(( $(left) - 600 )); [ "$lim" -gt 900 ] && lim=900
    run_stage pytest 1 "$lim" env -u WF_RESCALE -u LG_RESCALE LG_RUN_HEAVY=1 OMP_NUM_THREADS="$THREADS" python3 -m pytest -q -p no:cacheprovider \
      tests/test_h54_workflow.py
    lim=$(( $(left) - 600 )); [ "$lim" -gt 600 ] && lim=600
    run_stage count 1 "$lim" env -u WF_RESCALE -u LG_RESCALE python3 -u "$H54" --count-only --exps "$EXPS" --threads 1
    lim=$(( $(left) - 240 )); [ "$lim" -gt 1800 ] && lim=1800
    run_stage precheck 0 "$lim" env WF_RESCALE=1 python3 -u "$H54" --precheck --exps "$EXPS" --workers 7 --threads "$THREADS" --nboot 1000
    run_stage project 0 120 python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd
d = Path("data/processed/wf")
rows = []
cnt = pd.read_csv(d / "wf_count.csv") if (d / "wf_count.csv").exists() else None
for p in sorted((d / "shards").glob("wf*_precheck__*_unit.json")):
    u = json.loads(p.read_text())
    est = float(sum((u.get("est_detail") or {}).values())) + float(u.get("est_krige_s", 0.0)) + float(u.get("est_kmeans_s", 0.0))
    rows.append(dict(exp=u["exp"], target=u["target"], mode=u["mode"], split=u["split"], variant=u.get("variant", ""), elapsed_s=u["elapsed_s"],
                     est_s=est, ratio=u["elapsed_s"] / est if est > 0 else float("nan"), n_fit=u.get("n_fit_total", 0)))
pr = pd.DataFrame(rows)
if len(pr) and cnt is not None:
    r_exp = pr.groupby("exp").apply(lambda g: g.elapsed_s.sum() / max(g.est_s.sum(), 1e-9)).to_dict()
    r_all = float(pr.elapsed_s.sum() / max(pr.est_s.sum(), 1e-9))
    proj = cnt.groupby("exp").est_total_s.sum().rename("est_s").reset_index()
    proj["ratio"] = proj.exp.map(r_exp).fillna(r_all)
    proj["proj_cpu_h"] = proj.est_s * proj.ratio / 3600
    out = pd.concat([pr.assign(kind="precheck_unit"), proj.assign(kind="projection")], ignore_index=True)
    out.to_csv("wf_smoke_projection.csv", index=False)
    print(pr.to_string(index=False))
    print(proj.to_string(index=False))
    print(f"[project] 실측/추정 비(전체) {r_all:.2f} · 환산 누적 {proj.proj_cpu_h.sum():.1f} CPU-h(워커 1개, 4스레드)")
else:
    print("[project] 사전 점검 조각 또는 적합 수 표가 없다")
PY
    finish
  fi

  # ---------------------------------------------------------------- full: 복원 → 실행 → 집계
  local prev
  for prev in results_wf_prev.tar.gz results_wf.tar.gz; do
    if [ -f "$prev" ]; then
      if tar xzf "$prev" "$WFD" 2>/dev/null; then
        lg_log "[wf] 이전 결과 복원: $prev · 조각 $(find "$WFD/shards" -name '*_unit.json' 2>/dev/null | wc -l)개"
      else
        lg_log "[wf] 경고: $prev 를 풀지 못했다. 처음부터 실행한다"
      fi
      break
    fi
  done
  local RUN_S PID RC=0 RC_S=0 now last_prog last_pack
  RUN_S=$(( $(left) - SUM_RESERVE ))
  [ "$RUN_S" -gt 60 ] || { note run 1 1 "시간 한도가 집계 예비 시간보다 짧다"; finish; }
  timeout -k 60 "$RUN_S" env WF_RESCALE=1 python3 -u "$H54" --exps "$EXPS" --workers "$WORKERS" --threads "$THREADS" --resume --no-summarize \
    > "$LOGD/wf_run.log" 2>&1 &
  PID=$!
  lg_log "[wf] 실행 시작(pid $PID, 제한 ${RUN_S} s) → $LOGD/wf_run.log"
  last_prog=$(date +%s); last_pack=$last_prog
  while kill -0 "$PID" 2>/dev/null; do
    sleep 20
    now=$(date +%s)
    if [ $(( now - last_prog )) -ge "$PROG_EVERY" ]; then
      lg_log "[progress] 경과 $(( ( now - T0 ) / 60 )) min · 조각 $(find "$WFD/shards" -name '*_unit.json' 2>/dev/null | wc -l)개 · " \
             "$(grep -ao '완료 [0-9]*/[0-9]*' "$LOGD/wf_run.log" | tail -1) · 실패 $(grep -ac '\[FAIL\]' "$LOGD/wf_run.log") · " \
             "부하 $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
      last_prog=$now
    fi
    if [ $(( now - last_pack )) -ge "$PACK_EVERY" ]; then pack; last_pack=$(date +%s); fi
  done
  wait "$PID"; RC=$?
  note run 1 "$RC" "$([ "$RC" -eq 124 ] && echo '시간 한도로 중단' || echo '')"
  grep -a -E "\[FAIL\]|\[warn\]|\[pool\]|Traceback" "$LOGD/wf_run.log" | head -n 40 | sed 's/^/  [run] /'
  grep -a -E "^\[(data|plan|done)\]" "$LOGD/wf_run.log" | sed 's/^/  [run] /'
  pack
  local lim
  lim=$(( $(left) - 300 )); [ "$lim" -gt 300 ] || lim=300
  lg_log "[wf] 집계 시작(제한 ${lim} s)"
  timeout -k 30 "$lim" env WF_RESCALE=1 python3 -u "$H54" --summarize-only --exps "$EXPS" --workers "$WORKERS" --threads "$THREADS" \
    > "$LOGD/wf_summarize.log" 2>&1
  RC_S=$?
  note summarize 1 "$RC_S" "$([ "$RC_S" -eq 124 ] && echo '시간 제한으로 중단' || echo '')"
  tail -n 40 "$LOGD/wf_summarize.log" | sed 's/^/  [summarize] /'
  finish
}

main "$@"
