#!/bin/bash
# xbatch(XA–XJ) Rescale 실행 스크립트. 계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 1·4·5절.
# 작업 종류는 XBATCH_JOB 으로 고른다.
#   A  : XD-alg(xd_r, xd_t, xd_learn) + S9-GBM 학습 + XI + XJ + XB(xb_r, xb_t) + XH  (계획 R0·R1a·R1b 의 내용을 한 노드에서 차례로)
#   C  : XC + XC-r(부록 XC-0 커밋 뒤)
#   D  : XD-learn 시험 적합(S9* 동결 커밋 뒤)
#   E  : XE·XG 셀 단위·XB cci5y·XF 등 나머지(입력 준비 뒤)
# 단계마다 시간 제한·종료 코드를 상태 표에 적고, 한 실험의 실패가 다른 실험을 막지 않는다.
# 판정 표는 각 모듈이 봉인 폴더(data/processed/xbatch/<이름>/sealed/)에 쓴다. 화면 출력에서 판정·차이·RMSE 줄은 걸러 낸다.
# 이 스크립트는 저장소 밖(Rescale 작업 디렉터리)에서만 돈다.

main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then echo "[xb] 저장소(.git) 안에서는 실행하지 않는다"; exit 1; fi
  if [ -f xbatch_payload.tar.gz ] && [ ! -f data/processed/fidelity_base_v3.csv ]; then
    tar xzf xbatch_payload.tar.gz || { echo "[xb] 묶음 해제 실패"; exit 1; }
  fi
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env

  local JOB T0 BUDGET_S NCORE THREADS WORKERS LOGD STATUS OUTB FILT
  JOB="${XBATCH_JOB:-A}"
  T0=$(date +%s)
  BUDGET_S=$(( ${XBATCH_BUDGET_MIN:-170} * 60 ))
  NCORE=$(lg_ncores)
  THREADS="${XBATCH_THREADS:-4}"
  WORKERS="${XBATCH_WORKERS:-$(( (NCORE - 8) / THREADS ))}"
  [ "$WORKERS" -ge 1 ] || WORKERS=1
  LOGD="logs_xbatch_${JOB}"; STATUS="xbatch_${JOB}_status.csv"; OUTB="results_xbatch_${JOB}.tar.gz"
  FILT='판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'
  mkdir -p "$LOGD" data/processed/xbatch
  echo "stage,required,rc,sec,note" > "$STATUS"
  [ -f xbatch_payload_info.txt ] && grep -E '^(created|git_)' xbatch_payload_info.txt | sed 's/^/[payload] /'

  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
  pack() {
    tar czf "${OUTB}.part" data/processed/xbatch "$LOGD" "$STATUS" xbatch_payload_info.txt 2>/dev/null && mv -f "${OUTB}.part" "$OUTB"
  }
  run_stage() {
    local name="$1" req="$2" lim="$3"; shift 3
    local t0 rc msg=""
    t0=$(date +%s)
    local rem; rem=$(left)
    [ "$lim" -gt "$(( rem - 300 ))" ] && lim=$(( rem - 300 ))
    if [ "$lim" -lt 60 ]; then
      echo "${name},${req},NA,0,남은 시간 부족" >> "$STATUS"; lg_log "[stage] ${name} 생략: 남은 시간 부족"; return 0
    fi
    lg_log "[stage] ${name} 시작(제한 ${lim} s)"
    timeout -k 30 "$lim" "$@" 2>&1 | grep --line-buffered -v -E "$FILT" > "$LOGD/${name}.log"; rc=${PIPESTATUS[0]}
    [ "$rc" -eq 124 ] && msg="시간 제한 ${lim} s"
    echo "${name},${req},${rc},$(( $(date +%s) - t0 )),${msg}" >> "$STATUS"
    tail -n 4 "$LOGD/${name}.log" | sed "s/^/  [${name}] /"
    lg_log "[stage] ${name} 끝: 종료 코드 ${rc} · $(( $(date +%s) - t0 )) s · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    pack
  }
  finish() {
    pack
    lg_log "[xb] 단계별 상태"; sed 's/^/  /' "$STATUS"
    local bad
    bad=$(awk -F, 'NR > 1 && $2 == 1 && $3 != 0 {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then lg_log "[xb] 실패한 필수 단계: $bad"; exit 1; fi
    bad=$(awk -F, 'NR > 1 && $2 == 0 && $3 != 0 && $3 != "NA" {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then lg_log "[xb] 실패한 선택 단계: $bad"; exit 2; fi
    lg_log "[xb] 모든 단계 통과 · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"; exit 0
  }

  lg_env_report 2>&1 | tee "$LOGD/env.log"
  # 패키지 판 고정(계획 1절): 제약 파일의 판으로 설치하고 판을 기록한다. torch 는 바꾸지 않는다.
  local tv0 tv1
  tv0=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 없음)
  if ! python3 -m pip install -q -c scripts/rescale/xbatch_constraints.txt catboost scikit-learn pandas scipy numpy pytest > "$LOGD/deps.log" 2>&1; then
    python3 -m pip install -q --user -c scripts/rescale/xbatch_constraints.txt catboost scikit-learn pandas scipy numpy pytest >> "$LOGD/deps.log" 2>&1 \
      || { echo "deps,1,1,0,pip 설치 실패" >> "$STATUS"; finish; }
  fi
  tv1=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 없음)
  [ "$tv0" = "$tv1" ] || lg_log "[deps] 경고: torch 판이 바뀌었다($tv0 → $tv1). xbatch 는 torch 를 쓰지 않는다"
  if python3 scripts/3_deep_learning/xbatch_core.py --deps-check >> "$LOGD/deps.log" 2>&1; then
    echo "deps,1,0,0," >> "$STATUS"
  else
    echo "deps,1,1,0,판 확인 실패" >> "$STATUS"; tail -n 20 "$LOGD/deps.log"; finish
  fi
  tail -n 6 "$LOGD/deps.log" | sed 's/^/  [deps] /'
  python3 scripts/3_deep_learning/xbatch_core.py --self-check > "$LOGD/selfcheck.log" 2>&1
  echo "selfcheck,1,$?,0," >> "$STATUS"
  lg_log "[xb] 작업 ${JOB} · 코어 ${NCORE} · 워커 ${WORKERS} × 스레드 ${THREADS} · 시간 한도 $(( BUDGET_S / 60 )) min"

  # 단위 시험(로컬에서 전부 통과한 묶음. 여기서는 환경 확인용, 선택 단계)
  run_stage pytest_core 0 900 env -u WF_RESCALE OMP_NUM_THREADS=2 python3 -m pytest -q -p no:cacheprovider tests/test_xbatch_core.py

  local R="env WF_RESCALE=1 OMP_NUM_THREADS=${THREADS} python3 -u"
  local W="--workers ${WORKERS} --threads ${THREADS}"
  case "$JOB" in
    A)
      run_stage xd_run      0 3000 $R scripts/3_deep_learning/x_placement_policy.py --exp xd_r,xd_t,xd_learn $W --no-summarize
      run_stage xd_gbm      0 1500 $R scripts/3_deep_learning/x_placement_policy.py --train-gbm $W
      run_stage xd_sum      0 1800 $R scripts/3_deep_learning/x_placement_policy.py --summarize-only $W
      run_stage xi_run      0 1500 $R scripts/3_deep_learning/x_climate_extrapolation_retest.py $W
      run_stage xj_run      0 900  $R scripts/3_deep_learning/x_tempderived_aux_labels.py $W
      run_stage xb_run      0 3000 $R scripts/3_deep_learning/x_multisource_stacking.py --exp xb_r,xb_t $W
      run_stage xh_run      0 2400 $R scripts/3_deep_learning/x_validation_ladder.py $W
      ;;
    C)
      run_stage xc_gate     0 1200 $R scripts/3_deep_learning/x_workflow_end_to_end.py --gate $W
      run_stage xc_run      1 7200 $R scripts/3_deep_learning/x_workflow_end_to_end.py --part xc,xcr $W ${XBATCH_XC_ARGS:-}
      ;;
    D)
      run_stage xd_test     1 3000 $R scripts/3_deep_learning/x_placement_policy.py --exp xd_test $W ${XBATCH_XD_ARGS:-}
      ;;
    E)
      eval "${XBATCH_E_STAGES:-true}"
      ;;
    *) lg_log "[xb] 알 수 없는 작업 ${JOB}"; echo "job,1,1,0,unknown" >> "$STATUS" ;;
  esac
  finish
}
main "$@"
