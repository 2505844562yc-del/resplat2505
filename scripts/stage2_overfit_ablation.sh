#!/usr/bin/env bash
set -euo pipefail

# Controlled single-scene ablation for Stage 2.
# Usage: bash scripts/stage2_overfit_ablation.sh <variant> [steps] [boundary_weight] [feedback_scale] [feature_mode] [alignment_radius] [alignment_sigma] [use_multiview_consensus] [consensus_blend] [gate_boundary_loss]

VARIANT="${1:-joint}"
STEPS="${2:-20}"
BOUNDARY_WEIGHT="${3:-0.05}"
FEEDBACK_SCALE="${4:-1.0}"
FEATURE_MODE="${5:-residual}"
ALIGNMENT_RADIUS="${6:-4}"
ALIGNMENT_SIGMA="${7:-2.0}"
USE_MULTIVIEW_CONSENSUS="${8:-false}"
CONSENSUS_BLEND="${9:-0.5}"
GATE_BOUNDARY_LOSS="${10:-false}"
DISPLACEMENT_DIAGNOSTIC_PATH="${11:-null}"
USE_PARAMETER_ROUTING="${12:-false}"
ROUTE_APPEARANCE="${13:-true}"
USE_SELECTIVE_REFINEMENT="${14:-false}"
USE_SEMANTIC_INIT="${15:-false}"
SCENE="032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"

case "${VARIANT}" in
  baseline)
    LOSSES='[mse]'
    LOAD_BOUNDARIES=false
    FEEDBACK=false
    ;;
  loss_only)
    LOSSES='[mse,boundary]'
    LOAD_BOUNDARIES=true
    FEEDBACK=false
    ;;
  feedback_only)
    LOSSES='[mse]'
    LOAD_BOUNDARIES=true
    FEEDBACK=true
    ;;
  joint)
    LOSSES='[mse,boundary]'
    LOAD_BOUNDARIES=true
    FEEDBACK=true
    ;;
  *)
    echo "variant must be baseline, loss_only, feedback_only, or joint" >&2
    exit 2
    ;;
esac

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi

if [[ "${FEATURE_MODE}" != residual && "${FEATURE_MODE}" != residual_gradient \
      && "${FEATURE_MODE}" != residual_alignment \
      && "${FEATURE_MODE}" != residual_alignment_consensus \
      && "${FEATURE_MODE}" != residual_alignment_consensus_gated \
      && "${FEATURE_MODE}" != residual_alignment_consensus_dual \
      && "${FEATURE_MODE}" != residual_alignment_consensus_dual_warmup \
      && "${FEATURE_MODE}" != residual_alignment_displacement ]]; then
  echo "unsupported feature_mode: ${FEATURE_MODE}" >&2
  exit 2
fi

for value in "${USE_MULTIVIEW_CONSENSUS}" "${GATE_BOUNDARY_LOSS}" "${USE_PARAMETER_ROUTING}" "${ROUTE_APPEARANCE}" "${USE_SELECTIVE_REFINEMENT}" "${USE_SEMANTIC_INIT}"; do
  if [[ "${value}" != true && "${value}" != false ]]; then
    echo "multi-view flags must be true or false" >&2
    exit 2
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
SAFE_BW="${BOUNDARY_WEIGHT//./p}"
SAFE_FS="${FEEDBACK_SCALE//./p}"
SAFE_AR="${ALIGNMENT_RADIUS//./p}"
SAFE_AS="${ALIGNMENT_SIGMA//./p}"
OUT="outputs/stage2_ablation/${VARIANT}_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}"
if [[ "${FEATURE_MODE}" == residual_gradient ]]; then
  OUT="outputs/stage3_directional/${FEATURE_MODE}_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}"
elif [[ "${FEATURE_MODE}" == residual_alignment ]]; then
  OUT="outputs/stage4_alignment/${FEATURE_MODE}_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}_ar${SAFE_AR}_as${SAFE_AS}"
elif [[ "${FEATURE_MODE}" == residual_alignment_consensus \
        || "${FEATURE_MODE}" == residual_alignment_consensus_gated \
        || "${FEATURE_MODE}" == residual_alignment_consensus_dual \
        || "${FEATURE_MODE}" == residual_alignment_consensus_dual_warmup ]]; then
  OUT="outputs/stage7_dual_stream/${FEATURE_MODE}_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}_ar${SAFE_AR}_as${SAFE_AS}"
elif [[ "${FEATURE_MODE}" == residual_alignment_displacement ]]; then
  OUT="outputs/stage8_displacement/${FEATURE_MODE}_focused_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}_ar${SAFE_AR}_as${SAFE_AS}"
fi
if [[ "${USE_PARAMETER_ROUTING}" == true ]]; then
  ROUTE_NAME="geometry_appearance"
  if [[ "${ROUTE_APPEARANCE}" == false ]]; then
    ROUTE_NAME="geometry_only"
  fi
  OUT="outputs/stage9_parameter_routing/${FEATURE_MODE}_${ROUTE_NAME}_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}_ar${SAFE_AR}_as${SAFE_AS}"
fi
if [[ "${USE_SELECTIVE_REFINEMENT}" == true ]]; then
  OUT="outputs/stage10_selective_refinement/${FEATURE_MODE}_focused_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}_ar${SAFE_AR}_as${SAFE_AS}"
fi
if [[ "${USE_SEMANTIC_INIT}" == true ]]; then
  OUT="outputs/v2_semantic_init/${VARIANT}_${FEATURE_MODE}_geometry_grad_${STEPS}steps_bw${SAFE_BW}_fs${SAFE_FS}_ar${SAFE_AR}_as${SAFE_AS}"
fi
if [[ "${USE_MULTIVIEW_CONSENSUS}" == true ]]; then
  SAFE_BLEND="${CONSENSUS_BLEND//./p}"
  OUT="outputs/stage6_consensus/normalized_feedback_${STEPS}steps_blend${SAFE_BLEND}"
fi
EXTRA_OVERRIDES=()
if [[ "${VARIANT}" == loss_only || "${VARIANT}" == joint ]]; then
  EXTRA_OVERRIDES+=("loss.boundary.weight=${BOUNDARY_WEIGHT}")
fi
if [[ "${USE_MULTIVIEW_CONSENSUS}" == true ]]; then
  EXTRA_OVERRIDES+=(
    "model.encoder.use_multiview_boundary_consensus=true"
    "model.encoder.multiview_boundary_radius=2"
    "model.encoder.multiview_boundary_depth_relative_tolerance=0.05"
    "model.encoder.multiview_boundary_min_support_views=2.0"
    "model.encoder.multiview_boundary_blend=${CONSENSUS_BLEND}"
    "loss.boundary.use_multiview_consensus=${GATE_BOUNDARY_LOSS}"
  )
fi
if [[ "${FEATURE_MODE}" == residual_alignment_displacement ]]; then
  EXTRA_OVERRIDES+=(
    "model.encoder.multiview_boundary_displacement_radius=4"
    "model.encoder.multiview_boundary_displacement_source_radius=4"
    "model.encoder.multiview_boundary_displacement_diagnostic_path=${DISPLACEMENT_DIAGNOSTIC_PATH}"
  )
fi
if [[ "${USE_PARAMETER_ROUTING}" == true ]]; then
  EXTRA_OVERRIDES+=(
    "model.encoder.use_semantic_parameter_routing=true"
    "model.encoder.semantic_parameter_route_hidden_channels=64"
    "model.encoder.semantic_parameter_route_gain=0.5"
    "model.encoder.semantic_parameter_route_appearance=${ROUTE_APPEARANCE}"
  )
fi
if [[ "${USE_SELECTIVE_REFINEMENT}" == true ]]; then
  EXTRA_OVERRIDES+=(
    "model.encoder.use_semantic_selective_refinement=true"
    "model.encoder.semantic_selective_hidden_channels=64"
    "model.encoder.semantic_selective_gate_bias=-2.0"
    "model.encoder.semantic_selective_gain=0.5"
    "model.encoder.semantic_selective_radius=2"
  )
fi
if [[ "${USE_SEMANTIC_INIT}" == true ]]; then
  EXTRA_OVERRIDES+=(
    "model.encoder.use_semantic_gaussian_init=true"
    "model.encoder.semantic_init_hidden_channels=32"
    "model.encoder.semantic_init_gate_bias=-2.0"
    "model.encoder.semantic_init_proximity_radius=4"
    "model.encoder.semantic_init_depth_gain=0.1"
    "model.encoder.semantic_init_scale_gain=0.5"
  )
fi

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
  "loss=${LOSSES}" \
  wandb.mode=disabled \
  output_dir="${OUT}" \
  trainer.max_steps="${STEPS}" \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=null \
  data_loader.train.batch_size=1 \
  data_loader.train.num_workers=0 \
  data_loader.train.persistent_workers=false \
  data_loader.val.num_workers=0 \
  data_loader.val.persistent_workers=false \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene="${SCENE}" \
  dataset.overfit_max_views=32 \
  dataset.load_boundaries="${LOAD_BOUNDARIES}" \
  dataset.boundary_roots='[datasets/dl3dv_boundaries_sam2_small]' \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.num_target_views=2 \
  dataset.view_sampler.min_distance_between_context_views=24 \
  dataset.view_sampler.max_distance_between_context_views=45 \
  dataset.image_shape='[256,448]' \
  dataset.pose_align_middle_view=true \
  model.encoder.num_refine=1 \
  model.encoder.train_min_refine=1 \
  model.encoder.train_max_refine=1 \
  model.encoder.use_semantic_boundary_feedback="${FEEDBACK}" \
  model.encoder.semantic_boundary_feedback_scale="${FEEDBACK_SCALE}" \
  model.encoder.semantic_boundary_feature_mode="${FEATURE_MODE}" \
  model.encoder.semantic_boundary_alignment_radius="${ALIGNMENT_RADIUS}" \
  model.encoder.semantic_boundary_alignment_sigma="${ALIGNMENT_SIGMA}" \
  "${EXTRA_OVERRIDES[@]}" \
  train.depth_smooth_loss_weight=0. \
  train.print_log_every_n_steps=1 \
  optimizer.lr=1e-4 \
  optimizer.lr_monodepth=0. \
  checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1
