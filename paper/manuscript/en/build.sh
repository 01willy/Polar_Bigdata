#!/usr/bin/env bash
# build.sh : compile main.pdf and si_main.pdf (Springer Nature template, sn-nature.bst).
# Usage: ./build.sh [main|si_main ...]   (default: both). ENGINE=pdflatex ./build.sh to try pdflatex.
# Rules: OMP_NUM_THREADS=2, nice -n 10. Logs: build/<doc>.log, build/<doc>.blg; PDFs copied to ./.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
HERE=$(pwd)
TPL="$HERE/../template/sn_template/sn-article-template"
export TEXINPUTS=".:$TPL//:"
export BSTINPUTS=".:$TPL/bst//:"
export BIBINPUTS="$HERE:"
ENGINE=${ENGINE:-xelatex}
DOCS=${*:-main si_main}

python3 tools/merge_bib.py > /dev/null || { echo "merge_bib failed (see build/merge_bib.log)"; exit 1; }
mkdir -p build
for doc in $DOCS; do
  run() { nice -n 10 "$ENGINE" -interaction=nonstopmode -file-line-error -output-directory=build "$doc.tex" > "build/$doc.pass$1.txt" 2>&1; }
  run 1
  if grep -q '\\citation' "build/$doc.aux" 2>/dev/null; then
    (cd build && nice -n 10 bibtex "$doc" > "$doc.bibtex.txt" 2>&1)
  fi
  run 2
  run 3
  if [ -f "build/$doc.pdf" ]; then cp "build/$doc.pdf" "./$doc.pdf"; fi
  errs=$(grep -c '^\(! \|[^:]*:[0-9]*: \)' "build/$doc.log" 2>/dev/null || true)
  warns=$(grep -c 'Warning' "build/$doc.log" 2>/dev/null || true)
  pages=$(pdfinfo "./$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "$doc: engine $ENGINE, errors $errs, warning lines $warns, pages ${pages:-none}"
done
