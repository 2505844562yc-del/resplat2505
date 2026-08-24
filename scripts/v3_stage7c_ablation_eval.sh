#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v3_stage7c_ablation_eval.sh <baseline|semantic_z|z_uncertainty|z_uncertainty_support|full_joint> [steps] [test_len]
VARIANT="${1:?variant is required}"
STEPS="${2:-500}"
TEST_LEN="${3:-20}"
OFFICIAL_CKPT="pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth"

if [[ "${VARIANT}" == "baseline" ]]; then
  CHECKPOINT="${OFFICIAL_CKPT}"
  EVAL_OUT="outputs/v3_stage7c/baseline/eval_${TEST_LEN}samples"
  EXPERIMENT="dl3dv"
  SEMANTIC_OVERRIDES=(
    model.encoder.num_refine=1
    model.encoder.use_semantic_gaussian_features=false
    model.encoder.use_semantic_state_refinement=false
    model.encoder.use_semantic_residual_feedback=false
    model.encoder.use_semantic_uncertainty_refinement=false
    model.encoder.use_semantic_support_refinement=false
    model.encoder.use_semantic_ray_depth_head=false
    model.encoder.semantic_joint_train_new_heads=false
  )
else
  TRAIN_OUT="outputs/v3_stage7c/${VARIANT}/${STEPS}steps"
  CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
  EVAL_OUT="${TRAIN_OUT}_eval_${TEST_LEN}samples"
  EXPERIMENT="v3_semantic_joint_dl3dv"
  USE_UNCERTAINTY=false
  USE_SUPPORT=false
  USE_RAY_DEPTH=false
  case "${VARIANT}" in
    semantic_z)
      ;;
    z_uncertainty)
      USE_UNCERTAINTY=true
      ;;
    z_uncertainty_support)
      USE_UNCERTAINTY=true
      USE_SUPPORT=true
      ;;
    full_joint)
      USE_UNCERTAINTY=true
      USE_SUPPORT=true
      USE_RAY_DEPTH=true
      ;;
    *)
      echo "unknown variant: ${VARIANT}" >&2
      exit 2
      ;;
  esac
  SEMANTIC_OVERRIDES=(
    "model.encoder.use_semantic_uncertainty_refinement=${USE_UNCERTAINTY}"
    "model.encoder.use_semantic_support_refinement=${USE_SUPPORT}"
    "model.encoder.use_semantic_ray_depth_head=${USE_RAY_DEPTH}"
  )
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint missing for ${VARIANT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment="${EXPERIMENT}" mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.test_chunk_interval=1 \
  dataset.test_len="${TEST_LEN}" \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  "${SEMANTIC_OVERRIDES[@]}" \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${EVAL_OUT}"
