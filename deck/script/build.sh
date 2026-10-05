#!/usr/bin/env bash
# build.sh : 발표 대본 PDF 컴파일(xelatex 2회, nice -n 10). 출력 deck/render/발표대본_연구결과보고_2026-10-06.pdf
# 로그 build/presentation_script.log. 점검: pdffonts(글꼴 내장), Overfull 상자 수, 쪽 수.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p build
run() { nice -n 10 xelatex -interaction=nonstopmode -file-line-error -output-directory=build presentation_script.tex > "build/pass$1.txt" 2>&1; }
run 1
run 2
OUT="../render/발표대본_연구결과보고_2026-10-06.pdf"
if [ -f build/presentation_script.pdf ]; then cp build/presentation_script.pdf "$OUT"; fi
errs=$(grep -cE '^(! |./[^:]*:[0-9]+: )' build/presentation_script.log 2>/dev/null || true)
over=$(grep -c 'Overfull \\hbox' build/presentation_script.log 2>/dev/null || true)
miss=$(grep -c 'Missing character' build/presentation_script.log 2>/dev/null || true)
pages=$(pdfinfo "$OUT" 2>/dev/null | awk '/^Pages/{print $2}')
echo "presentation_script: errors $errs, overfull hbox $over, missing chars $miss, pages ${pages:-none}"
echo "fonts:"; pdffonts "$OUT" 2>/dev/null | awk 'NR>2{print "  "$0}' | sed -n '1,12p'
