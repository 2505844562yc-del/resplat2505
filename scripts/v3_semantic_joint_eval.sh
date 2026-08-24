#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v3_semantic_joint_eval.sh [steps] [test_len] [overfit|multiscene]
STEPS="${1:-20}"
TEST_LEN="${2:-5}"
SCOPE="${3:-multiscene}"
TRAIN_OUT="outputs/v3_semantic_joint/${SCOPE}_${STEPS}steps"
EVAL_OUT="${TRAIN_OUT}_eval_${TEST_LEN}samples"
CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "checkpoint missing under ${TRAIN_OUT}/checkpoints" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=v3_semantic_joint_dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.test_chunk_interval=1 \
  dataset.test_len="${TEST_LEN}" \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.image_shape='[256,448]' \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${EVAL_OUT}"
