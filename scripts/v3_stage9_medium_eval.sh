#!/usr/bin/env bash
set -euo pipefail

# Evaluate Stage 7C or a Stage 7-9 checkpoint on disjoint held-out views.
# Usage: bash scripts/v3_stage9_medium_eval.sh
#   <stage7c|identity|joint_no_split|joint> [step] [train_steps]
MODE="${1:-stage7c}"
STEP="${2:-500}"
TRAIN_STEPS="${3:-500}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="dl3dv_970a5c674c27b504d592d0a70c496d0e35ab0dc76802fb6e1bf336a4c1fe150a"
INDEX="assets/dl3dv_evaluation/v3_stage9_medium_single_scene_heldout.json"
STAGE7C_CKPT="outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt"

case "${MODE}" in
  stage7c)
    CHECKPOINT="${STAGE7C_CKPT}"
    USE_SPLIT=false
    USE_ADAPTER=false
    EVAL_OUT="outputs/v3_stage9_medium/stage7c_heldout"
    ;;
  identity)
    CHECKPOINT="${STAGE7C_CKPT}"
    USE_SPLIT=true
    USE_ADAPTER=true
    EVAL_OUT="outputs/v3_stage9_medium/identity_heldout"
    ;;
  joint_no_split|joint)
    TRAIN_OUT="outputs/v3_stage9_medium/single_scene_joint/${TRAIN_STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name "*step_${STEP}.ckpt" -print -quit)"
    if [[ "${MODE}" == "joint" ]]; then
      USE_SPLIT=true
    else
      USE_SPLIT=false
    fi
    USE_ADAPTER=true
    EVAL_OUT="outputs/v3_stage9_medium/${MODE}_step${STEP}_heldout"
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
if [[ ! -f "${INDEX}" ]]; then
  echo "held-out evaluation index is missing: ${INDEX}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=v3_semantic_fixed_split_dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene="${SCENE}" \
  dataset.test_len=1 \
  dataset.test_times_per_scene=1 \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path="${INDEX}" \
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
