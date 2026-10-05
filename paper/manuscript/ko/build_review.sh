#!/usr/bin/env bash
# build_review.sh : 국문 검토판 main_review.pdf(검토 메모 첫 쪽, 그림과 표 1 은 처음 인용한 곳 가까이). 제출판은 build.sh.
# 규칙: OMP_NUM_THREADS=2, nice -n 10.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
HERE=$(pwd)
TPL="$HERE/../template/sn_template/sn-article-template"
export BSTINPUTS=".:$TPL/bst//:"
export BIBINPUTS="$HERE/../en:"
( cd ../en && python3 tools/merge_bib.py > /dev/null ) || { echo "merge_bib failed"; exit 1; }
python3 ../en/tools/make_review.py ko || exit 1
mkdir -p build/rv
run() { nice -n 10 xelatex -interaction=nonstopmode -file-line-error -output-directory=build/rv main_review.tex > "build/rv/pass$1.txt" 2>&1; }
run 1
(cd build/rv && nice -n 10 bibtex main_review > bibtex.txt 2>&1)
run 2
run 3
[ -f build/rv/main_review.pdf ] && cp build/rv/main_review.pdf ./main_review.pdf
errs=$(grep -c '^\(! \|[^:]*:[0-9]*: \)' build/rv/main_review.log 2>/dev/null || true)
echo "ko/main_review: errors $errs, pages $(pdfinfo ./main_review.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
