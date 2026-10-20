#!/usr/bin/env bash
# finish_paper_v14.sh (2026-10-20): wait for the v14 figure renders and the 1 September site series, then regenerate the v14 macros, re-render the
# two supplementary figures that depend on them (S6 band panel, S10 site series), build both manuscripts (main + SI), run the compliance check and
# package the v14 deliverable. Run: nohup bash tools/finish_paper_v14.sh > logs/fig_v14/finish_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg
log() { echo "[finish14] $(date +%H:%M) $*"; }
until grep -q "^EXIT" logs/fig_v14/fig6.log 2>/dev/null && grep -q "^EXIT" logs/fig_v14/fig7b.log 2>/dev/null && grep -q "DONE" logs/fig_v14/others_chain.log 2>/dev/null && grep -q "^EXIT" logs/fig_v14/site_series_d244.log 2>/dev/null; do sleep 60; done
log "inputs ready: fig6 $(grep ^EXIT logs/fig_v14/fig6.log) fig7 $(grep ^EXIT logs/fig_v14/fig7b.log) site-series $(grep ^EXIT logs/fig_v14/site_series_d244.log)"
grep -h "ok=\|problems" logs/fig_v14/fig6.log logs/fig_v14/fig7b.log | cut -c1-160
log "numbers"; nice -n 10 python3 scripts/2_evaluation/v14_numbers.py > logs/fig_v14/v14_numbers.log 2>&1 || { log "v14_numbers failed"; tail -5 logs/fig_v14/v14_numbers.log; exit 1; }
grep -c "command" paper/manuscript/shared_v14/numbers_v14.tex
for s in si_s06_training_sites si_s10_site_series; do log "render $s"; nice -n 10 python3 scripts/4_visualization/paper_v14/$s.py > logs/fig_v14/${s}_final.log 2>&1; log "$s EXIT $? $(grep -h "ok=\|problems" logs/fig_v14/${s}_final.log | tail -1 | cut -c1-120)"; done
for l in en ko; do log "build $l"; bash paper/manuscript/${l}_v14/build.sh main si; done
log "undefined macros (en main):"; grep -A4 "Undefined control sequence" paper/manuscript/en_v14/build/main.log | grep -o "\\\\n[A-Za-z]*" | sort | uniq -c | head -20
log "compliance"; (cd paper/manuscript/en_v14 && nice -n 10 python3 compliance.py > ../../../logs/fig_v14/compliance.log 2>&1); grep "본문(\|표시 항목\|첫 제출\|그림 설명" paper/manuscript/en_v14/COMPLIANCE_v14.md
log "package"; bash tools/package_paper_v14.sh > logs/fig_v14/package.log 2>&1; tail -12 logs/fig_v14/package.log
log "DONE"
