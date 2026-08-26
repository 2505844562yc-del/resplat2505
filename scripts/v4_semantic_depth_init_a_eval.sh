#!/usr/bin/env bash
set -euo pipefail

# Evaluate the zero-initialized V3 checkpoint or a trained Situation A adapter.
# Usage: bash scripts/v4_semantic_depth_init_a_eval.sh <identity|trained> [steps]
MODE="${1:-identity}"
STEPS="${2:-20}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${V4_SCENE:-dl3dv_970a5c674c27b504d592d0a70c496d0e35ab0dc76802fb6e1bf336a4c1fe150a}"
INDEX="${V4_HELDOUT_INDEX:-assets/dl3dv_evaluation/v3_stage9_medium_single_scene_heldout.json}"
EXPERIMENT="${V4_EXPERIMENT:-v4_semantic_depth_init_a_dl3dv}"
RUN_NAME="${V4_RUN_NAME:-adapter_only}"
START_CKPT="outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt"

case "${MODE}" in
  identity)
    CHECKPOINT="${START_CKPT}"
    OUT="outputs/v4_semantic_depth_init_a/${RUN_NAME}/identity_heldout"
    ;;
  trained)
    TRAIN_OUT="outputs/v4_semantic_depth_init_a/${RUN_NAME}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    OUT="outputs/v4_semantic_depth_init_a/${RUN_NAME}/${STEPS}steps_heldout"
    ;;
  *)
    echo "mode must be identity or trained" >&2
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
  +experiment="${EXPERIMENT}" mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene="${SCENE}" \
  dataset.test_len=1 \
  dataset.test_times_per_scene=1 \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path="${INDEX}" \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  model.encoder.use_semantic_depth_logit_adapter=true \
  model.encoder.semantic_depth_train_only=false \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${OUT}"
