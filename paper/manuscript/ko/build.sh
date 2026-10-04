#!/usr/bin/env bash
# build.sh : 국문판 ko/main.pdf 컴파일(xelatex, bibtex sn-nature.bst, 3회). 영문판 references.bib 를 쓴다.
# 규칙: OMP_NUM_THREADS=2, nice -n 10. 로그 build/main.log, build/main.blg.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
HERE=$(pwd)
TPL="$HERE/../template/sn_template/sn-article-template"
export BSTINPUTS=".:$TPL/bst//:"
export BIBINPUTS="$HERE/../en:"
( cd ../en && python3 tools/merge_bib.py > /dev/null ) || { echo "merge_bib failed"; exit 1; }
mkdir -p build
run() { nice -n 10 xelatex -interaction=nonstopmode -file-line-error -output-directory=build main.tex > "build/main.pass$1.txt" 2>&1; }
run 1
(cd build && nice -n 10 bibtex main > main.bibtex.txt 2>&1)
run 2
run 3
[ -f build/main.pdf ] && cp build/main.pdf ./main.pdf
errs=$(grep -c '^\(! \|[^:]*:[0-9]*: \)' build/main.log 2>/dev/null || true)
warns=$(grep -c 'Warning' build/main.log 2>/dev/null || true)
pages=$(pdfinfo ./main.pdf 2>/dev/null | awk '/^Pages/{print $2}')
echo "ko/main: errors $errs, warning lines $warns, pages ${pages:-none}"
