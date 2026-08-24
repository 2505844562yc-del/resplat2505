#!/usr/bin/env bash
set -euo pipefail

# Fair Stage-7C training from the same official ReSplat checkpoint.
# Usage: bash scripts/v3_stage7c_ablation_train.sh <semantic_z|z_uncertainty|z_uncertainty_support|full_joint> [steps]
VARIANT="${1:?variant is required}"
STEPS="${2:-500}"
OFFICIAL_CKPT="pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth"
OUT="outputs/v3_stage7c/${VARIANT}/${STEPS}steps"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${OFFICIAL_CKPT}" ]]; then
  echo "official ReSplat checkpoint is missing: ${OFFICIAL_CKPT}" >&2
  exit 1
fi

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

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=v3_semantic_joint_dl3dv \
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
  dataset.overfit_to_scene=null \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.num_target_views=2 \
  dataset.view_sampler.min_distance_between_context_views=24 \
  dataset.view_sampler.max_distance_between_context_views=45 \
  dataset.image_shape='[256,448]' \
  model.encoder.use_semantic_uncertainty_refinement="${USE_UNCERTAINTY}" \
  model.encoder.use_semantic_support_refinement="${USE_SUPPORT}" \
  model.encoder.use_semantic_ray_depth_head="${USE_RAY_DEPTH}" \
  train.print_log_every_n_steps=25 \
  optimizer.lr=5e-5 \
  optimizer.lr_monodepth=0.0 \
  checkpointing.pretrained_model="${OFFICIAL_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1
