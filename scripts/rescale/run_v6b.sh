#!/bin/bash
# v6b Rescale 실행(2026-10-14): MODIS 2010–2024 입력(AppEEARS 재요청 완료)으로 공변량 사다리의 MODIS·전체 묶음을 관측일 정합 강제력으로 다시 돈다.
# V6_JOB=modis(elm 96코어). 저장소 밖(Rescale 작업 디렉터리)에서만 돈다. 산출 results_v6_modis.tar.gz
main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then echo "[v6b] 저장소 안에서는 실행하지 않는다"; exit 1; fi
  [ -f v6b_payload.tar.gz ] && [ ! -f data/processed/v5/labels_cell_year_v5.csv ] && { tar xzf v6b_payload.tar.gz || exit 1; }
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env; lg_env_report; lg_install_deps || exit 1
  local JOB="${V6_JOB:-modis}" T0=$(date +%s) BUDGET_S=$(( ${V6_BUDGET_MIN:-80} * 60 )) NCORE=$(lg_ncores) LOGD STATUS OUTB
  LOGD="logs_v6_${JOB}"; STATUS="v6_${JOB}_status.csv"; OUTB="results_v6_${JOB}.tar.gz"
  mkdir -p "$LOGD" data/processed/v5/runs; echo "stage,rc,sec" > "$STATUS"
  H="python3 scripts/3_deep_learning/v5_compare.py --no-products --proc data/processed/v5 --out data/processed/v5/runs"
  A="python3 scripts/2_evaluation/v5_aggregate.py"
  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
  pack() { tar czf "${OUTB}.part" data/processed/v5/runs "$LOGD" "$STATUS" 2>/dev/null && mv -f "${OUTB}.part" "$OUTB"; }
  stage() { local name="$1"; shift; local t0=$(date +%s) rem=$(left); [ "$rem" -lt 300 ] && { echo "${name},NA,0" >> "$STATUS"; return 0; }
            lg_log "[stage] $name"; timeout -k 30 "$rem" "$@" > "$LOGD/${name}.log" 2>&1; local rc=$?; echo "${name},${rc},$(( $(date +%s) - t0 ))" >> "$STATUS"
            tail -n 2 "$LOGD/${name}.log" | sed "s/^/  [$name] /"; pack; }
  local TH=$(( NCORE / 3 )); [ "$TH" -gt 24 ] && TH=24
  # MODIS 2010–2024 묶음: 기준 F8TS 와 함께(같은 분할·씨앗에서 짝지은 차이), 블록·연도 분할, 관측일 정합
  for R in Alaska Lena Canada; do
    $H --tag modis_v6 --regions $R --schemes block,year --forcings year_obs --featsets F8TS,F8TS_M,F8TS_ALL --methods P1,D0,R1 --learners catboost --lams 0.5 --seeds 0,1,2 --threads $TH > "$LOGD/modis_$R.log" 2>&1 &
  done; wait; echo "modis,0,$(( $(date +%s) - T0 ))" >> "$STATUS"; pack
  stage agg_modis $A --run data/processed/v5/runs/modis_v6 --nboot 2000 --refs P1
  pack; lg_log "[v6b] 끝 $(( ( $(date +%s) - T0 ) / 60 )) min"; sed 's/^/  /' "$STATUS"
}
main "$@"
