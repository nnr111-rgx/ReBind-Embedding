#!/usr/bin/env bash
set -euo pipefail

PRETRAINED_REBIND="${1:?Usage: bash run_finetune_pipeline.sh /path/to/rebind_best.pt}"

mkdir -p data logs runs/4ary_stage1 runs/4ary_stage2 runs/4ary_stage3 runs/4ary_eval

python scripts/01_generate_cc_marker_dataset.py \
  --output data/cc_marker_ids_4ary.h5 \
  --train-samples 120000 \
  --val-samples 12000 \
  --test-samples 12000 \
  --seed 202607 \
  2>&1 | tee logs/01_generate_dataset.log

python scripts/02_finetune_embedding.py \
  --data data/cc_marker_ids_4ary.h5 \
  --stage stage1 \
  --pretrained-rebind "$PRETRAINED_REBIND" \
  --save-dir runs/4ary_stage1 \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --lambda-bit 0.35 \
  --workers 4 \
  --seed 0 \
  2>&1 | tee logs/02_stage1.log

python scripts/02_finetune_embedding.py \
  --data data/cc_marker_ids_4ary.h5 \
  --stage stage2 \
  --init-stage-checkpoint runs/4ary_stage1/best.pt \
  --save-dir runs/4ary_stage2 \
  --epochs 100 \
  --batch-size 128 \
  --lr 5e-5 \
  --lambda-bit 0.35 \
  --workers 4 \
  --seed 0 \
  2>&1 | tee logs/03_stage2.log

python scripts/02_finetune_embedding.py \
  --data data/cc_marker_ids_4ary.h5 \
  --stage stage3 \
  --init-stage-checkpoint runs/4ary_stage2/best.pt \
  --save-dir runs/4ary_stage3 \
  --epochs 100 \
  --batch-size 128 \
  --lr 3e-5 \
  --lambda-bit 0.35 \
  --workers 4 \
  --seed 0 \
  2>&1 | tee logs/04_stage3.log

python scripts/03_eval_embedding.py \
  --data data/cc_marker_ids_4ary.h5 \
  --checkpoint runs/4ary_stage3/best.pt \
  --output-dir runs/4ary_eval \
  --n-test 12000 \
  --batch-size 256 \
  --seed 9001 \
  2>&1 | tee logs/05_eval.log
