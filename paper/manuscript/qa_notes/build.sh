#!/usr/bin/env bash
# build.sh: compile the Korean Q&A notes (xelatex, two passes) -> qa_notes_ko.pdf
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
mkdir -p build
for i in 1 2; do nice -n 10 xelatex -interaction=nonstopmode -file-line-error -output-directory=build qa_notes_ko.tex > build/pass$i.txt 2>&1; done
cp build/qa_notes_ko.pdf ./qa_notes_ko.pdf 2>/dev/null
echo "qa_notes_ko: errors $(grep -c '^\(! \|[^:]*:[0-9]*: \)' build/qa_notes_ko.log), pages $(pdfinfo qa_notes_ko.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
