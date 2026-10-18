#!/bin/bash
# v9b Rescale 실행(2026-10-16): 외부 지역 소수 라벨 적응(EXTFT). 소지역 묶음 4개를 병렬로 돌리고 집계. 산출 results_v9b_extft.tar.gz
main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then echo "[v9b] 저장소 안에서는 실행하지 않는다"; exit 1; fi
  [ -f v9b_payload.tar.gz ] && [ ! -f data/processed/v5/labels_cell_year_v5.csv ] && { tar xzf v9b_payload.tar.gz || exit 1; }
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env; lg_env_report; lg_install_deps || exit 1
  local T0=$(date +%s) NCORE=$(lg_ncores) LOGD="logs_v9b_extft" STATUS="v9b_extft_status.csv" OUTB="results_v9b_extft.tar.gz"
  mkdir -p "$LOGD" data/processed/v5/runs/extra_v9; echo "stage,rc,sec" > "$STATUS"
  pack() { tar czf "${OUTB}.part" data/processed/v5/runs/extra_v9 "$LOGD" "$STATUS" 2>/dev/null && mv -f "${OUTB}.part" "$OUTB"; }
  local TH=$(( NCORE / 4 )); [ "$TH" -gt 24 ] && TH=24; [ "$TH" -lt 4 ] && TH=4
  python3 scripts/3_deep_learning/v9_extra.py --tag extra_v9 --exp EXTFT --subregions "NE Siberia" --threads $TH > "$LOGD/ft_nes.log" 2>&1 &
  python3 scripts/3_deep_learning/v9_extra.py --tag extra_v9 --exp EXTFT --subregions "W Siberia" --threads $TH > "$LOGD/ft_ws.log" 2>&1 &
  python3 scripts/3_deep_learning/v9_extra.py --tag extra_v9 --exp EXTFT --subregions "C Siberia,European Russia" --threads $TH > "$LOGD/ft_cs_er.log" 2>&1 &
  python3 scripts/3_deep_learning/v9_extra.py --tag extra_v9 --exp EXTFT --subregions "Chukotka,Svalbard,Kamchatka,Greenland" --threads $TH > "$LOGD/ft_rest.log" 2>&1 &
  wait; echo "fit,0,$(( $(date +%s) - T0 ))" >> "$STATUS"; pack
  python3 scripts/2_evaluation/v9_extft_agg.py --run data/processed/v5/runs/extra_v9 > "$LOGD/agg.log" 2>&1; echo "agg,$?,$(( $(date +%s) - T0 ))" >> "$STATUS"
  pack; lg_log "[v9b] 끝 $(( ( $(date +%s) - T0 ) / 60 )) min"; sed 's/^/  /' "$STATUS"
}
main "$@"
