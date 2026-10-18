#!/bin/bash
# ingest_v6.sh: Rescale v6 결과 반영 파이프라인(2026-10-14). 결과 묶음을 풀고 수치 매크로·그림·지도·덱·보고서·원고를 차례로 다시 만든다.
# 사용: bash scripts/rescale/ingest_v6.sh [cpu|gpu|modis ...] (기본: 세 묶음 가운데 내려받은 것 전부)
#   단계: fetch(있으면 건너뜀) → tar 복원(runs/*, site_series) → numbers → F3·F4·F5(+slide) → 지도(year_obs, 로컬 CPU 8스레드) → F6(+slide) → 덱 → 보고서 → 원고 → 논리 점검
# 로컬 계산은 지도 예측(추론)과 그림뿐이며 nice 10, 스레드 8 이하로 돈다.
set -uo pipefail; cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4
JOBS=${*:-"cpu gpu modis"}
declare -A ID=([cpu]=CJsgU [gpu]=RkmTeb [modis]=CECkU)
declare -A DIR=([cpu]=results/rescale_v6_cpu [gpu]=results/rescale_v6_gpu [modis]=results/rescale_v6_modis)
declare -A TAR=([cpu]=results_v6_cpu.tar.gz [gpu]=results_v6_gpu.tar.gz [modis]=results_v6_modis.tar.gz)
log() { echo "[ingest $(date +%H:%M)] $*"; }
got=""
for j in $JOBS; do
  d=${DIR[$j]}; t=${TAR[$j]}
  if [ ! -f "$d/$t" ]; then
    st=$(python3 tools/rescale_client.py status "${ID[$j]}" 2>/dev/null | head -1 | awk '{print $NF}')
    if echo "$st" | grep -qi completed; then log "fetch $j (${ID[$j]})"; python3 tools/rescale_client.py fetch "${ID[$j]}" --dest "$d" > "$d.fetch.log" 2>&1 || { log "fetch 실패 $j"; continue; }; else log "$j 는 아직 $st"; continue; fi
  fi
  f=$(find "$d" -name "$t" | head -1); [ -z "$f" ] && { log "$j 묶음 없음"; continue; }
  log "복원 $f"; tar xzf "$f" -C . data/processed/v5/runs data/processed/v5/site_series 2>/dev/null || tar xzf "$f" -C . data/processed/v5/runs
  got="$got $j"
done
[ -z "$got" ] && { log "반영할 결과 없음"; exit 0; }
if echo "$got" | grep -q cpu; then
  C=data/processed/v5/runs/core_v6
  log "제품 비교·연도별·공통 셀(로컬 채점)"
  [ -f $C/products_eval.csv ] || nice -n 10 python3 scripts/2_evaluation/v5_products_eval.py --run $C --forcing year_obs --nboot 2000 > $C/products_eval.log 2>&1
  [ -f $C/products_eval_mask25.csv ] || nice -n 10 python3 scripts/2_evaluation/v5_products_eval.py --run $C --forcing year_obs --nboot 2000 --mask-km 25 --suffix _mask25 > $C/products_eval_mask25.log 2>&1
  [ -f $C/products_eval_mask5.csv ] || nice -n 10 python3 scripts/2_evaluation/v5_products_eval.py --run $C --forcing year_obs --nboot 1000 --mask-km 5 --suffix _mask5 > $C/products_eval_mask5.log 2>&1
  [ -d $C/yearwise_year_obs_F8TS ] || nice -n 10 python3 scripts/2_evaluation/v5_yearwise.py --run $C --forcing year_obs > $C/yearwise.log 2>&1
  [ -e $C/yearwise ] || ln -sfn yearwise_year_obs_F8TS $C/yearwise
  [ -f $C/common_cells_eval.csv ] || nice -n 10 python3 scripts/2_evaluation/v5_common_cells.py --run core_v6 --forcing year_obs --ens none > $C/common_cells.log 2>&1
fi
log "수치 매크로"; python3 scripts/2_evaluation/v6_numbers.py || exit 1
if echo "$got" | grep -q cpu; then
  log "논리 점검"; python3 scripts/2_evaluation/v5_audit_logic.py --run data/processed/v5/runs/core_v6 --forcing year_obs > data/processed/v5/audit_logic_core_v6.log 2>&1 || log "논리 점검 스크립트 인자 확인 필요"
  log "그림 F3 F4 F5"; nice -n 10 python3 scripts/4_visualization/v6_figs.py --fig 3 --fig 4 --fig 5 2>&1 | grep "^\[F\|Error"
  nice -n 10 python3 scripts/4_visualization/v6_figs.py --slide --fig 3 --fig 4 --fig 5 2>&1 | grep "^\[F\|Error"
  for R in Alaska Lena; do
    log "지도 $R (관측일 정합 학습)"; nice -n 10 python3 scripts/3_deep_learning/v5_map_predict.py --region $R --forcing year_obs --run data/processed/v5/runs/core_v6 --threads 8 > data/processed/v5/maps/map_${R}_v6.log 2>&1 || log "지도 $R 실패"
  done
  log "그림 F6"; nice -n 10 python3 scripts/4_visualization/v6_maps.py 2>&1 | grep "^\[F6\]\|Error"; nice -n 10 python3 scripts/4_visualization/v6_maps.py --slide 2>&1 | grep "^\[F6\]\|Error"
  python3 scripts/2_evaluation/v6_numbers.py
fi
if echo "$got" | grep -q "gpu\|modis"; then
  log "그림 F4(학습기·사다리)"; nice -n 10 python3 scripts/4_visualization/v6_figs.py --fig 4 2>&1 | grep "^\[F\|Error"; nice -n 10 python3 scripts/4_visualization/v6_figs.py --slide --fig 4 2>&1 | grep "^\[F\|Error"
fi
log "덱"; OMP_NUM_THREADS=2 nice -n 10 python3 deck/build_v6_report.py 2>&1 | grep "audit\] 지적\|\[pdf\]"
log "보고서"; (cd outputs/report_v6 && for i in 1 2; do nice -n 10 xelatex -interaction=nonstopmode -halt-on-error report_v6.tex > build.log 2>&1 || true; done; grep -c "^!" build.log; pdfinfo report_v6.pdf | grep Pages)
log "원고"; (cd paper/manuscript/en_v6 && OMP_NUM_THREADS=2 ./build.sh main 2>&1 | tail -1)
log "끝:$got"
