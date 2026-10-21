#!/usr/bin/env bash
# finish_paper_v15.sh (2026-10-21): macros (v15 + cell-year + extra) -> both manuscripts (main + SI) -> compliance -> deliverables sync.
# Run after the figure renders: nohup bash tools/finish_paper_v15.sh > logs/fig_v15/finish_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg
mkdir -p logs/fig_v15
log() { echo "[finish15] $(date +%H:%M) $*"; }
log "numbers"; nice -n 10 python3 scripts/2_evaluation/v15_numbers.py > logs/fig_v15/numbers.log 2>&1 || { log "v15_numbers failed"; tail -3 logs/fig_v15/numbers.log; exit 1; }
nice -n 10 python3 scripts/2_evaluation/v15_cellyear_rescore.py > logs/fig_v15/cellyear_rescore.log 2>&1 || { log "cellyear rescore failed"; exit 1; }
nice -n 10 python3 scripts/2_evaluation/v15_extra_numbers.py > logs/fig_v15/extra_numbers.log 2>&1 || { log "extra numbers failed"; exit 1; }
log "duplicate macro definitions (should be none):"
cat paper/manuscript/shared_v10/numbers_v6.tex paper/manuscript/shared_v10/numbers_v9.tex paper/manuscript/shared_v10/numbers_v10.tex paper/manuscript/shared_v11/numbers_v11.tex paper/manuscript/shared_v15/numbers_v15_cellyear.tex paper/manuscript/shared_v15/numbers_v15_extra.tex | grep -o "newcommand{\\\\n[A-Za-z]*}" | sort | uniq -d | head
grep -ho "providecommand{\\\\n[A-Za-z]*}" paper/manuscript/shared_v15/numbers_v15.tex | sort | uniq -d | head
for l in en ko; do log "build $l"; bash paper/manuscript/${l}_v15/build.sh main si; done
log "undefined macros (en main / si):"; grep -h -A4 "Undefined control sequence" paper/manuscript/en_v15/build/main.log paper/manuscript/en_v15/build/si.log 2>/dev/null | grep -o "\\\\n[A-Za-z]*" | sort | uniq -c | head
log "first errors (en main / si):"; grep -m 3 "^[^:]*:[0-9]*: " paper/manuscript/en_v15/build/main.log paper/manuscript/en_v15/build/si.log || true
log "sync deliverables (builds the compressed copy)"; VER=v15 bash tools/sync_deliverables.sh > logs/fig_v15/sync_deliverables.log 2>&1; tail -3 logs/fig_v15/sync_deliverables.log
log "compliance"; (cd paper/manuscript/en_v15 && nice -n 10 python3 compliance.py > ../../../logs/fig_v15/compliance.log 2>&1); grep "본문(\|표시 항목\|첫 제출\|그림 설명\|초록" paper/manuscript/en_v15/COMPLIANCE_v15.md; VER=v15 bash tools/sync_deliverables.sh --no-build > /dev/null 2>&1
ls -la deliverables/paper | awk '{print $5, $9}' | grep -i "pdf"
log "DONE"
