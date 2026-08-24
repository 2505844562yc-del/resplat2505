#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v3_semantic_support_eval.sh [steps] [test_len] [opacity_gain] [scale_gain]
STEPS="${1:-20}"
TEST_LEN="${2:-5}"
OPACITY_GAIN="${3:-0.1}"
SCALE_GAIN="${4:-0.05}"
OPACITY_TAG="${OPACITY_GAIN//./p}"
SCALE_TAG="${SCALE_GAIN//./p}"
TRAIN_OUT="outputs/v3_semantic_support/${STEPS}steps"
CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
EVAL_OUT="outputs/v3_semantic_support/${STEPS}steps_eval_${TEST_LEN}samples_op_${OPACITY_TAG}_sc_${SCALE_TAG}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "checkpoint missing under ${TRAIN_OUT}/checkpoints" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.test_chunk_interval=1 \
  dataset.test_len="${TEST_LEN}" \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  model.encoder.num_refine=1 \
  model.encoder.use_semantic_gaussian_features=true \
  model.encoder.semantic_feature_dim=16 \
  model.encoder.semantic_feature_projection_trainable=false \
  model.encoder.use_semantic_state_refinement=true \
  model.encoder.use_semantic_residual_feedback=true \
  model.encoder.semantic_residual_alignment=raster_vjp \
  model.encoder.semantic_residual_alpha_floor=0.1 \
  model.encoder.semantic_vjp_direct_gain=0.3 \
  model.encoder.use_semantic_uncertainty_refinement=true \
  model.encoder.semantic_uncertainty_hidden_channels=128 \
  model.encoder.semantic_uncertainty_residual_gain=0.25 \
  model.encoder.semantic_uncertainty_need_floor=0.25 \
  model.encoder.semantic_uncertainty_reliability_floor=0.05 \
  model.encoder.semantic_uncertainty_support_relative_scale=4.0 \
  model.encoder.semantic_uncertainty_loss_weight=0.0 \
  model.encoder.use_semantic_support_refinement=true \
  model.encoder.semantic_support_hidden_channels=256 \
  model.encoder.semantic_support_opacity_gain="${OPACITY_GAIN}" \
  model.encoder.semantic_support_scale_gain="${SCALE_GAIN}" \
  model.encoder.semantic_support_regularization_weight=0.0 \
  model.encoder.use_semantic_geometry_vjp=false \
  model.encoder.use_semantic_ray_depth_head=false \
  model.encoder.semantic_feature_loss_weight=0.0 \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${EVAL_OUT}"
