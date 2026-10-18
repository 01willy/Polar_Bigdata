#!/bin/bash
# v6 Rescale 실행(2026-10-14). 관측일 정합(year_obs)을 주 강제력으로 핵심 비교·공변량 사다리·확인 실행·지점 시계열·집계를 한 노드에서 돈다.
# 작업 종류 V6_JOB: cpu(CatBoost·물리식, elm 96코어) | gpu(신경망 학습기, A100). 저장소 밖(Rescale 작업 디렉터리)에서만 돈다.
main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then echo "[v6] 저장소 안에서는 실행하지 않는다"; exit 1; fi
  [ -f v6_payload.tar.gz ] && [ ! -f data/processed/v5/labels_cell_year_v5.csv ] && { tar xzf v6_payload.tar.gz || exit 1; }
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env; lg_env_report; lg_install_deps || exit 1
  local JOB="${V6_JOB:-cpu}" T0=$(date +%s) BUDGET_S=$(( ${V6_BUDGET_MIN:-170} * 60 )) NCORE=$(lg_ncores) LOGD STATUS OUTB
  LOGD="logs_v6_${JOB}"; STATUS="v6_${JOB}_status.csv"; OUTB="results_v6_${JOB}.tar.gz"
  mkdir -p "$LOGD" data/processed/v5/runs; echo "stage,rc,sec" > "$STATUS"
  H="python3 scripts/3_deep_learning/v5_compare.py --no-products --proc data/processed/v5 --out data/processed/v5/runs"
  A="python3 scripts/2_evaluation/v5_aggregate.py"
  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
  pack() { tar czf "${OUTB}.part" data/processed/v5/runs data/processed/v5/site_series "$LOGD" "$STATUS" 2>/dev/null && mv -f "${OUTB}.part" "$OUTB"; }
  stage() { local name="$1"; shift; local t0=$(date +%s) rem=$(left); [ "$rem" -lt 300 ] && { echo "${name},NA,0" >> "$STATUS"; return 0; }
            lg_log "[stage] $name"; timeout -k 30 "$rem" "$@" > "$LOGD/${name}.log" 2>&1; local rc=$?; echo "${name},${rc},$(( $(date +%s) - t0 ))" >> "$STATUS"
            tail -n 2 "$LOGD/${name}.log" | sed "s/^/  [$name] /"; pack; }
  if [ "$JOB" = "cpu" ]; then
    local TH=$(( NCORE / 3 )); [ "$TH" -gt 24 ] && TH=24
    # 1. 핵심 비교: 세 지역 병렬(강제력 3판 × 분할 4단 × 방법 10 × λ 3 × 씨앗 3)
    for R in Alaska Lena Canada; do
      $H --tag core_v6 --regions $R --schemes block,year,random,block_year --forcings year_obs,year,clim --featsets F8TS --methods P0,P1,P1a,Ku,Ed,PE,D0,F1,D1,R1 --learners catboost --lams 0.25,0.5,1 --seeds 0,1,2 --threads $TH > "$LOGD/core_$R.log" 2>&1 &
    done; wait; echo "core,0,$(( $(date +%s) - T0 ))" >> "$STATUS"; pack
    stage core_region $H --tag core_v6 --schemes region --forcings year_obs,year,clim --featsets F8TS --methods P0,P1,P1a,Ku,Ed,PE,D0,F1,D1,R1 --learners catboost --lams 0.25,0.5,1 --seeds 0,1,2 --threads $NCORE
    # 2. 공변량 사다리(관측일 정합)
    for R in Alaska Lena Canada; do
      $H --tag featsets_v6 --regions $R --schemes block,year --forcings year_obs --featsets F8,F8T,F8TS,F8TS_L,F8TS_M,F8TS_X,F8TS_ALL,F8TS_SAR --methods P1,D0,R1 --learners catboost --lams 0.5 --seeds 0,1,2 --threads $TH > "$LOGD/feat_$R.log" 2>&1 &
    done; wait; echo "featsets,0,$(( $(date +%s) - T0 ))" >> "$STATUS"; pack
    # 3. 확인 실행: 분할 씨앗 1, 블록 0.25°·1.0°, 랜덤 포레스트
    stage confirm $H --tag confirm_v6 --fold-seed 1 --regions Alaska,Lena,Canada --schemes block --forcings year_obs --featsets F8TS --methods P1,D0,R1 --learners catboost --lams 0.5 --seeds 0,1,2 --threads $NCORE
    stage blk025 $H --tag blk025_v6 --block-deg 0.25 --regions Alaska,Lena,Canada --schemes block --forcings year_obs --featsets F8TS --methods P1,D0,R1 --learners catboost --lams 0.5 --seeds 0,1,2 --threads $NCORE
    stage blk100 $H --tag blk100_v6 --block-deg 1.0 --regions Alaska,Lena,Canada --schemes block --forcings year_obs --featsets F8TS --methods P1,D0,R1 --learners catboost --lams 0.5 --seeds 0,1,2 --threads $NCORE
    stage rf $H --tag rf_v6 --regions Alaska,Lena,Canada --schemes block,year --forcings year_obs --featsets F8TS --methods D0,R1 --learners rf --lams 0.5 --seeds 0,1,2 --threads $NCORE
    # 4. 지점 전 연도 시계열(관측일 정합·연도 정합)
    stage site_obs python3 scripts/2_evaluation/v5_site_series.py --forcing year_obs --threads $NCORE
    stage site_year python3 scripts/2_evaluation/v5_site_series.py --forcing year --threads $NCORE
    # 5. 집계(부트스트랩 2,000)
    stage agg_core $A --run data/processed/v5/runs/core_v6 --nboot 2000 --conditions --forcing-pairs year_obs:clim,year_obs:year,year:clim
    stage agg_feat $A --run data/processed/v5/runs/featsets_v6 --nboot 1000 --refs P1
    for t in confirm_v6 blk025_v6 blk100_v6; do stage agg_$t $A --run data/processed/v5/runs/$t --nboot 1000 --refs P1; done
    stage agg_rf $A --run data/processed/v5/runs/rf_v6 --ref-run data/processed/v5/runs/core_v6 --nboot 1000 --refs P1
  else
    local GPUS; GPUS=$(lg_gpu_list); local i=0
    for R in Alaska Lena Canada; do
      g=$(echo "$GPUS" | cut -d, -f$(( i % 4 + 1 ))); i=$(( i + 1 ))
      $H --tag nn_v6 --regions $R --schemes block,year --forcings year_obs --featsets F8TS,F8TS_M --methods D0,R1 --learners mlp,tabm,ftt --lams 0.5 --seeds 0,1,2 --threads 4 --epochs 100 --gpu "$g" > "$LOGD/nn_$R.log" 2>&1 &
    done; wait; echo "nn,0,$(( $(date +%s) - T0 ))" >> "$STATUS"; pack
    stage agg_nn $A --run data/processed/v5/runs/nn_v6 --nboot 1000 --refs P1
  fi
  pack; lg_log "[v6] 끝 $(( ( $(date +%s) - T0 ) / 60 )) min"; sed 's/^/  /' "$STATUS"
}
main "$@"
