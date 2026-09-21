#!/usr/bin/env bash
# M1 S-C 주효과 스크린 — GPU당 1 프로세스 순차 큐(공유 서버 CPU 보호). 로그 logs/m1/<tag>_s<shard>.log
# 사용: bash scripts/3_deep_learning/m1_run_queue.sh   (GPU 5–9 고정, 2026-09-21 사용자 지정)
set -u
cd "$(dirname "$0")/../.."
mkdir -p logs/m1
SPLITS=3; SEEDS=3
run () {  # run <gpu> <tag> <shard> <axis> <conds> <extra...>
  local g=$1 tag=$2 sh=$3 axis=$4 conds=$5; shift 5
  echo "[$(date +%H:%M)] start $tag shard $sh on GPU $g" >> logs/m1/queue.log
  GPU=$g python3 -u scripts/3_deep_learning/m1_master_factorial.py --axis "$axis" --conds "$conds" \
     --splits "$SPLITS" --seeds "$SEEDS" --min-cells 3 --tag "$tag" --shard "$sh" "$@" > "logs/m1/${tag}_s${sh}.log" 2>&1
  echo "[$(date +%H:%M)] done  $tag shard $sh on GPU $g (exit $?)" >> logs/m1/queue.log
}
# 레인 5개(GPU별 순차). 무거운 모델 축 shard를 서로 다른 레인에 분산.
( run 5 m1_model 0 model covonly,noinfo,labels --alaska --nshard 3 --exclude-models realmlp --splits 1
  run 5 m1_iw    0 iw    covonly,noinfo --splits 1 ) &
( run 6 m1_model 1 model covonly,noinfo,labels --alaska --nshard 3 --exclude-models realmlp --splits 1
  run 6 m1_anchor 0 anchor covonly,noinfo,labels,deploy --alaska --nshard 2
  run 6 m1_dset  0 dset  covonly,noinfo ) &
( run 7 m1_model 2 model covonly,noinfo,labels --alaska --nshard 3 --exclude-models realmlp --splits 1
  run 7 m1_anchor 1 anchor covonly,noinfo,labels,deploy --alaska --nshard 2
  run 7 m1_xset  0 xset  covonly,noinfo,labels --alaska ) &
( run 8 m1_pseudo 0 pseudo covonly,deploy --nshard 2
  run 8 m1_deep  0 anchor noinfo,deploy --targets Svalbard,Scandinavia,Alps,Mongolia_CAsia,Tibet --sources F4_direct,F4_calm_temp --splits 1 ) &
( run 9 m1_pseudo 1 pseudo covonly,deploy --nshard 2
  run 9 m1_realmlp 0 model covonly,noinfo,labels --alaska --only-models realmlp --splits 1 --seeds 1 ) &
wait
echo "[$(date +%H:%M)] ALL DONE" >> logs/m1/queue.log
