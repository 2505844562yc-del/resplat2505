#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v4_situation_a_eval.sh <baseline|identity|trained> [steps]
MODE="${1:-identity}"
STEPS="${2:-20}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${V4_SCENE:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
INDEX="${V4_HELDOUT_INDEX:-assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json}"
RUN_NAME="${V4_RUN_NAME:-a1_adapter_only}"
OUTPUT_ROOT="${V4_OUTPUT_ROOT:-outputs/v4_situation_a}"
START_CKPT="${V4_START_CKPT:-outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt}"
ENABLE_A=true

case "${MODE}" in
  baseline)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/baseline_heldout"
    ENABLE_A=false
    ;;
  identity)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/identity_heldout"
    ;;
  trained)
    TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps_heldout"
    ;;
  *)
    echo "mode must be baseline, identity, or trained" >&2
    exit 2
    ;;
esac

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint is missing for ${MODE}: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${INDEX}" ]]; then
  echo "held-out index is missing: ${INDEX}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=v4_situation_a_residual_injection_dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene="${SCENE}" \
  dataset.test_len=1 \
  dataset.test_times_per_scene=1 \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path="${INDEX}" \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  model.encoder.use_semantic_depth_residual_injection="${ENABLE_A}" \
  model.encoder.semantic_depth_train_only=false \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${OUT}"
