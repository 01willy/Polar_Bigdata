#!/bin/bash
# v6c Rescale 실행(2026-10-15): 라벨 수 사다리 LL(지역 안)·TL(새 지역 전이 + 대상 라벨 n개). 관측일 정합 강제력, CatBoost. 산출 results_v6_ladder.tar.gz
main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then echo "[v6c] 저장소 안에서는 실행하지 않는다"; exit 1; fi
  [ -f v6c_payload.tar.gz ] && [ ! -f data/processed/v5/labels_cell_year_v5.csv ] && { tar xzf v6c_payload.tar.gz || exit 1; }
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env; lg_env_report; lg_install_deps || exit 1
  local JOB="${V6_JOB:-ladder}" T0=$(date +%s) BUDGET_S=$(( ${V6_BUDGET_MIN:-200} * 60 )) NCORE=$(lg_ncores) LOGD STATUS OUTB
  LOGD="logs_v6_${JOB}"; STATUS="v6_${JOB}_status.csv"; OUTB="results_v6_${JOB}.tar.gz"
  mkdir -p "$LOGD" data/processed/v5/runs; echo "stage,rc,sec" > "$STATUS"
  pack() { tar czf "${OUTB}.part" data/processed/v5/runs "$LOGD" "$STATUS" 2>/dev/null && mv -f "${OUTB}.part" "$OUTB"; }
  local TH=$(( NCORE / 3 )); [ "$TH" -gt 24 ] && TH=24
  for R in Alaska Lena Canada; do
    python3 scripts/3_deep_learning/v6_label_ladder.py --tag ladder_v6 --exp LL,TL --regions $R --threads $TH > "$LOGD/ladder_$R.log" 2>&1 &
  done; wait; echo "ladder,0,$(( $(date +%s) - T0 ))" >> "$STATUS"; pack
  python3 scripts/2_evaluation/v6_label_ladder_agg.py --run data/processed/v5/runs/ladder_v6 --nboot 1000 > "$LOGD/agg.log" 2>&1; echo "agg,$?,$(( $(date +%s) - T0 ))" >> "$STATUS"
  pack; lg_log "[v6c] 끝 $(( ( $(date +%s) - T0 ) / 60 )) min"; sed 's/^/  /' "$STATUS"
}
main "$@"
