#!/usr/bin/env bash
set -euo pipefail

# Evaluate a V2 short-ablation checkpoint and report both initial and refined metrics.
# Usage: bash scripts/v2_eval_ablation.sh <baseline|v1|init|init_v1> [steps] [test_len]

MODE="${1:-init_v1}"
STEPS="${2:-20}"
TEST_LEN="${3:-5}"
TRAIN_OUT="outputs/v2_ablation/${MODE}_${STEPS}steps"
EVAL_OUT="outputs/v2_ablation/${MODE}_${STEPS}steps_eval"

case "${MODE}" in
  baseline)
    USE_SEMANTIC_INIT=false
    FEEDBACK=false
    ;;
  v1)
    USE_SEMANTIC_INIT=false
    FEEDBACK=true
    ;;
  init)
    USE_SEMANTIC_INIT=true
    FEEDBACK=false
    ;;
  init_v1)
    USE_SEMANTIC_INIT=true
    FEEDBACK=true
    ;;
  *)
    echo "mode must be baseline, v1, init, or init_v1" >&2
    exit 2
    ;;
esac

CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "checkpoint missing under ${TRAIN_OUT}/checkpoints" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.test_chunk_interval=1 \
  dataset.test_len="${TEST_LEN}" \
  dataset.load_boundaries=true \
  dataset.boundary_roots='[datasets/dl3dv_boundaries_sam2_small]' \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  model.encoder.num_refine=1 \
  model.encoder.use_semantic_gaussian_init="${USE_SEMANTIC_INIT}" \
  model.encoder.semantic_init_hidden_channels=32 \
  model.encoder.semantic_init_gate_bias=-2.0 \
  model.encoder.semantic_init_proximity_radius=4 \
  model.encoder.semantic_init_depth_gain=0.1 \
  model.encoder.semantic_init_scale_gain=0.5 \
  model.encoder.use_semantic_boundary_feedback="${FEEDBACK}" \
  model.encoder.semantic_boundary_feedback_scale=0.35 \
  model.encoder.semantic_boundary_feature_mode=residual_alignment \
  model.encoder.semantic_boundary_alignment_radius=2 \
  model.encoder.semantic_boundary_alignment_sigma=1.0 \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${EVAL_OUT}"
