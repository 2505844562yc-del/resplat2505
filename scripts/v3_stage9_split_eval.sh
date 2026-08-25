#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v3_stage9_split_eval.sh <stage7c|identity|split_only|joint> [steps] [test_len]
MODE="${1:-identity}"
STEPS="${2:-20}"
TEST_LEN="${3:-5}"
STAGE7C_CKPT="outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt"

USE_SPLIT=true
USE_ADAPTER=false
case "${MODE}" in
  stage7c)
    CHECKPOINT="${STAGE7C_CKPT}"
    EVAL_OUT="outputs/v3_stage9_fixed_split/stage7c_eval_${TEST_LEN}samples"
    USE_SPLIT=false
    ;;
  identity)
    CHECKPOINT="${STAGE7C_CKPT}"
    EVAL_OUT="outputs/v3_stage9_fixed_split/identity_eval_${TEST_LEN}samples"
    ;;
  split_only|joint)
    TRAIN_OUT="outputs/v3_stage9_fixed_split/${MODE}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    EVAL_OUT="${TRAIN_OUT}_eval_${TEST_LEN}samples"
    if [[ "${MODE}" == "joint" ]]; then
      USE_ADAPTER=true
    fi
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint missing for ${MODE}: ${CHECKPOINT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m src.main \
  +experiment=v3_semantic_fixed_split_dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.test_chunk_interval=1 \
  dataset.test_len="${TEST_LEN}" \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  model.encoder.use_semantic_fixed_candidate_split="${USE_SPLIT}" \
  model.encoder.use_semantic_updater_adapter="${USE_ADAPTER}" \
  model.encoder.semantic_split_train_only=false \
  model.encoder.semantic_updater_adapter_train_only=false \
  model.encoder.semantic_updater_unfreeze_last_block=false \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${EVAL_OUT}"
