#!/usr/bin/env bash
set -euo pipefail

# Controlled single-scene ablation for Stage 2.
# Usage: bash scripts/stage2_overfit_ablation.sh <variant> [steps]

VARIANT="${1:-joint}"
STEPS="${2:-20}"
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
OUT="outputs/stage2_ablation/${VARIANT}_${STEPS}steps"

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
  train.depth_smooth_loss_weight=0. \
  train.print_log_every_n_steps=1 \
  optimizer.lr=1e-4 \
  optimizer.lr_monodepth=0. \
  checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1
