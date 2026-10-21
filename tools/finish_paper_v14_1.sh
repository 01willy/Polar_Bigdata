#!/usr/bin/env bash
# finish_paper_v14_1.sh (2026-10-21): macros -> both manuscripts (main + SI) -> compliance -> deliverables sync (compressed copy included).
# Run after the figure renders are done: nohup bash tools/finish_paper_v14_1.sh > logs/fig_v14/finish_chain_v14_1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg
log() { echo "[finish14.1] $(date +%H:%M) $*"; }
log "numbers"; nice -n 10 python3 scripts/2_evaluation/v14_numbers.py > logs/fig_v14/v14_numbers.log 2>&1 || { log "v14_numbers failed"; exit 1; }
nice -n 10 python3 scripts/2_evaluation/v14_cellyear_rescore.py > logs/fig_v14/cellyear_rescore.log 2>&1 || { log "cellyear rescore failed"; exit 1; }
log "duplicate macro definitions (should be none):"; grep -ho "providecommand{\\\\n[A-Za-z]*}" paper/manuscript/shared_v14/numbers_v14.tex | sort | uniq -d | head; grep -ho "newcommand{\\\\n[A-Za-z]*}" paper/manuscript/shared_v14/numbers_v14_cellyear.tex | sort | uniq -d | head
for l in en ko; do log "build $l"; bash paper/manuscript/${l}_v14/build.sh main si; done
log "undefined macros (en main / si):"; grep -h -A4 "Undefined control sequence" paper/manuscript/en_v14/build/main.log paper/manuscript/en_v14/build/si.log 2>/dev/null | grep -o "\\\\n[A-Za-z]*" | sort | uniq -c | head
log "first SI errors (en):"; grep -m 3 "^[^:]*:[0-9]*: " paper/manuscript/en_v14/build/si.log || true
log "compliance"; (cd paper/manuscript/en_v14 && nice -n 10 python3 compliance.py > ../../../logs/fig_v14/compliance.log 2>&1); grep "본문(\|표시 항목\|첫 제출\|그림 설명\|초록" paper/manuscript/en_v14/COMPLIANCE_v14.md
log "sync deliverables"; VER=v14 bash tools/sync_deliverables.sh > logs/fig_v14/sync_deliverables.log 2>&1; tail -3 logs/fig_v14/sync_deliverables.log
ls -la deliverables/paper | awk '{print $5, $9}' | grep -i "pdf"
log "DONE"
