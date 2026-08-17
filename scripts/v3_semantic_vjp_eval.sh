#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v3_semantic_vjp_eval.sh [steps] [test_len]
STEPS="${1:-20}"
TEST_LEN="${2:-5}"
TRAIN_OUT="outputs/v3_semantic_vjp/${STEPS}steps"
EVAL_OUT="outputs/v3_semantic_vjp/${STEPS}steps_eval"
CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"

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
  model.encoder.semantic_state_residual_gain=0.1 \
  model.encoder.semantic_feature_loss_weight=0.1 \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${EVAL_OUT}"
