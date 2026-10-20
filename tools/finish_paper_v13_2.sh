#!/usr/bin/env bash
# finish_paper_v13_2.sh (2026-10-20): after the Canada 1 km maps exist, render the three-region figures (Fig 6, Fig 7, S8, S9), regenerate the
# macro files (v10, v6, v13), build both manuscripts (main + SI), run the compliance check and package the v13 deliverable.
# Run: nohup bash tools/finish_paper_v13_2.sh > logs/fig_v13/final_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg
log() { echo "[finish] $(date +%H:%M) $*"; }
# ---- wait for the Canada map prediction chain
until grep -q "\[maps\] canada EXIT" logs/canada_map/map_predict_chain.log 2>/dev/null; do sleep 60; done
if ! grep -q "\[maps\] canada EXIT 0" logs/canada_map/map_predict_chain.log; then log "Canada maps failed: $(grep 'EXIT' logs/canada_map/map_predict_chain.log)"; exit 1; fi
log "Canada maps ready: $(ls data/processed/v5/maps/Canada | wc -l) files"
# ---- figures (sequential: each loads one region's rasters at a time)
for s in fig6_product_maps fig7_maps si_s08_map_detail si_s09_correction_maps; do
  log "render $s"; nice -n 10 python3 scripts/4_visualization/paper_v13/$s.py > logs/fig_v13/${s}_final.log 2>&1; rc=$?
  grep "QA\|overlap\|outside\|color:" logs/fig_v13/${s}_final.log | head -5; log "$s EXIT $rc"
  [ $rc -ne 0 ] && { tail -5 logs/fig_v13/${s}_final.log; exit 1; }
done
# ---- macros
log "numbers"; nice -n 10 python3 scripts/2_evaluation/v10_numbers.py > logs/fig_v13/v10_numbers_final.log 2>&1 || { log "v10_numbers failed"; tail -5 logs/fig_v13/v10_numbers_final.log; exit 1; }
nice -n 10 python3 scripts/2_evaluation/v6_numbers.py > logs/fig_v13/v6_numbers_final.log 2>&1 || { log "v6_numbers failed"; exit 1; }
cp paper/manuscript/en_v6/numbers.tex paper/manuscript/shared_v10/numbers_v6.tex
nice -n 10 python3 scripts/2_evaluation/v13_numbers.py > logs/fig_v13/v13_numbers_final.log 2>&1 || { log "v13_numbers failed"; tail -5 logs/fig_v13/v13_numbers_final.log; exit 1; }
grep -c "newcommand\|providecommand" paper/manuscript/shared_v13/numbers_v13.tex
# ---- manuscripts
for l in en ko; do log "build $l"; bash paper/manuscript/${l}_v13/build.sh main si; done
log "undefined macros (en main):"; grep -A4 "Undefined control sequence" paper/manuscript/en_v13/build/main.log | grep -o "\\\\n[A-Za-z]*" | sort | uniq -c | head -20
log "compliance"; (cd paper/manuscript/en_v13 && nice -n 10 python3 compliance.py > ../../../logs/fig_v13/compliance_final.log 2>&1); tail -3 logs/fig_v13/compliance_final.log
log "package"; bash tools/package_paper_v13.sh > logs/fig_v13/package_final.log 2>&1; tail -12 logs/fig_v13/package_final.log
log "DONE"
