#!/bin/bash
# v9 Rescale 실행(2026-10-15): 제품 기반 잔차 결합(PA) · 외부 지역 무라벨 시험(EXT) · 잔차 학습기 SHAP. 관측일 기후 입력, CatBoost. 산출 results_v9_extra.tar.gz
main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then echo "[v9] 저장소 안에서는 실행하지 않는다"; exit 1; fi
  [ -f v9_payload.tar.gz ] && [ ! -f data/processed/v5/labels_cell_year_v5.csv ] && { tar xzf v9_payload.tar.gz || exit 1; }
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env; lg_env_report; lg_install_deps || exit 1
  local T0=$(date +%s) NCORE=$(lg_ncores) LOGD="logs_v9_extra" STATUS="v9_extra_status.csv" OUTB="results_v9_extra.tar.gz"
  mkdir -p "$LOGD" data/processed/v5/runs; echo "stage,rc,sec" > "$STATUS"
  pack() { tar czf "${OUTB}.part" data/processed/v5/runs/extra_v9 "$LOGD" "$STATUS" 2>/dev/null && mv -f "${OUTB}.part" "$OUTB"; }
  local TH=$(( NCORE / 4 )); [ "$TH" -gt 24 ] && TH=24; [ "$TH" -lt 4 ] && TH=4
  for R in Alaska Lena Canada; do
    python3 scripts/3_deep_learning/v9_extra.py --tag extra_v9 --exp PA --regions $R --threads $TH > "$LOGD/pa_$R.log" 2>&1 &
  done
  python3 scripts/3_deep_learning/v9_extra.py --tag extra_v9 --exp EXT,SHAP --threads $TH > "$LOGD/ext_shap.log" 2>&1 &
  wait; echo "fit,0,$(( $(date +%s) - T0 ))" >> "$STATUS"; pack
  python3 scripts/2_evaluation/v9_extra_agg.py --run data/processed/v5/runs/extra_v9 > "$LOGD/agg.log" 2>&1; echo "agg,$?,$(( $(date +%s) - T0 ))" >> "$STATUS"
  pack; lg_log "[v9] 끝 $(( ( $(date +%s) - T0 ) / 60 )) min"; sed 's/^/  /' "$STATUS"
}
main "$@"
