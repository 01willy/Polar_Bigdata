#!/bin/bash
# H18·H19 ML 축 실행(설계 §1.2·§2.3): M1 하네스 --ext --axis ext, 조건 covonly·noinfo, 분할 3, seed 3, GPU 2장 샤딩.
# 평가 마스크는 전 수준 공통(--eval-require LST,T,W,V,H). 실행(ROOT): bash scripts/3_deep_learning/h1819_run.sh 5 7
cd "$(dirname "$0")/../.."
G0=${1:-5}; G1=${2:-7}
COMMON="--ext covariates_ext_v1.csv --eval-require LST,T,W,V,H --axis ext --conds covonly,noinfo --splits 3 --seeds 3 --tag h1819"
GPU=$G0 nohup python3 scripts/3_deep_learning/m1_master_factorial.py $COMMON --shard 0 --nshard 2 > logs/h1819_shard0.log 2>&1 &
GPU=$G1 nohup python3 scripts/3_deep_learning/m1_master_factorial.py $COMMON --shard 1 --nshard 2 > logs/h1819_shard1.log 2>&1 &
echo "started shards on GPU $G0,$G1"
