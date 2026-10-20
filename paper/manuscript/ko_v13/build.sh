#!/usr/bin/env bash
# build.sh (v11): compile main.pdf and si.pdf with the Springer Nature template (xelatex + bibtex sn-nature.bst), three passes.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
HERE=$(pwd); TPL="$HERE/../template/sn_template/sn-article-template"
export TEXINPUTS=".:$TPL//:" BSTINPUTS=".:$TPL/bst//:" BIBINPUTS="$HERE:"
DOCS=${*:-main si}
mkdir -p build
for doc in $DOCS; do
  run() { nice -n 10 xelatex -interaction=nonstopmode -file-line-error -output-directory=build "$doc.tex" > "build/$doc.pass$1.txt" 2>&1; }
  run 1
  if grep -q '\\citation' "build/$doc.aux" 2>/dev/null; then (cd build && BIBINPUTS="$HERE:" nice -n 10 bibtex "$doc" > "$doc.bibtex.txt" 2>&1); fi
  run 2; run 3
  [ -f "build/$doc.pdf" ] && cp "build/$doc.pdf" "./$doc.pdf"
  errs=$(grep -c '^\(! \|[^:]*:[0-9]*: \)' "build/$doc.log" 2>/dev/null || true)
  undef=$(grep -c 'Undefined control sequence\|Citation .* undefined\|Reference .* undefined' "build/$doc.log" 2>/dev/null || true)
  pages=$(pdfinfo "./$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  echo "$doc: errors $errs, undefined $undef, pages ${pages:-none}"
done
