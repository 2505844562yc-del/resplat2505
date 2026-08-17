#!/usr/bin/env bash
set -euo pipefail

# Evaluate deterministic raster-VJP semantic correction without training.
# Usage: bash scripts/v3_semantic_vjp_direct_eval.sh [gain] [test_len]
GAIN="${1:-0.3}"
TEST_LEN="${2:-5}"
GAIN_TAG="${GAIN//./p}"
EVAL_OUT="outputs/v3_semantic_vjp_direct/gain_${GAIN_TAG}_${TEST_LEN}samples"

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
  model.encoder.semantic_vjp_direct_gain="${GAIN}" \
  model.encoder.semantic_state_residual_gain=0.1 \
  model.encoder.semantic_feature_loss_weight=0.1 \
  checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${EVAL_OUT}"
