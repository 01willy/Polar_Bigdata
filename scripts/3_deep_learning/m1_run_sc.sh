#!/usr/bin/env bash
# M1 S-C 주효과 스크린 런처 — 축별 스크린을 GPU에 분배해 병렬 실행. 로그는 logs/m1/.
# 사용: bash scripts/3_deep_learning/m1_run_sc.sh "5 6 7 8 9" [SPLITS] [SEEDS]  (2026-09-21 사용자 지정 GPU 5–9)
# 각 축은 shard로 나뉘어 GPU 목록에 라운드로빈 배정된다. 완료 후 m1_analysis.py --tags ... 로 병합.
set -u
cd "$(dirname "$0")/../.."
GPUS=(${1:-"5 6 7 8 9"})
SPLITS=${2:-3}
SEEDS=${3:-3}
mkdir -p logs/m1
NG=${#GPUS[@]}
i=0
launch () {  # launch <tag> <axis|--configs f> <conds> <extra...>
  local tag=$1; local axis=$2; local conds=$3; shift 3
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  if [[ "$axis" == *.json ]]; then A="--configs $axis"; else A="--axis $axis"; fi
  echo "[launch] $tag on GPU $g ($A, conds=$conds, $*)"
  GPU=$g nohup python3 -u scripts/3_deep_learning/m1_master_factorial.py $A --conds "$conds" \
      --splits "$SPLITS" --seeds "$SEEDS" --min-cells 3 --tag "$tag" "$@" > "logs/m1/${tag}.log" 2>&1 &
}
# 축별 스크린 (기본 대상 = TRANSFER_MAIN 6지역; 알래스카 6-fold는 --alaska)
launch m1_anchor  anchor  covonly,noinfo,labels,deploy --alaska --nshard 2 --shard 0
launch m1_anchor  anchor  covonly,noinfo,labels,deploy --alaska --nshard 2 --shard 1
launch m1_pseudo  pseudo  covonly,deploy               --nshard 2 --shard 0
launch m1_pseudo  pseudo  covonly,deploy               --nshard 2 --shard 1
# 모델 축은 신경망 적합 비용(≈25 s) 때문에 분할 반복 없이(E1 규약 split 0) 돌린다. RealMLP는 CPU 병목(143 s/적합)이라 seed 1.
launch m1_model   model   covonly,noinfo,labels        --alaska --nshard 3 --shard 0 --exclude-models realmlp --splits 1
launch m1_model   model   covonly,noinfo,labels        --alaska --nshard 3 --shard 1 --exclude-models realmlp --splits 1
launch m1_model   model   covonly,noinfo,labels        --alaska --nshard 3 --shard 2 --exclude-models realmlp --splits 1
launch m1_realmlp model   covonly,noinfo,labels        --alaska --only-models realmlp --splits 1 --seeds 1
launch m1_dset    dset    covonly,noinfo
launch m1_xset    xset    covonly,noinfo,labels        --alaska
launch m1_iw      iw      covonly,noinfo               --splits 1
# 심부 레짐(지온·시추공 유도 라벨 포함, 정보 없음·배포형만, 앵커 축): 전이 한계 정량화용 별도 절
launch m1_deep    anchor  noinfo,deploy --targets Svalbard,Scandinavia,Alps,Mongolia_CAsia,Tibet --sources F4_direct,F4_calm_temp --splits 1
wait
echo "[done] all shards finished"
