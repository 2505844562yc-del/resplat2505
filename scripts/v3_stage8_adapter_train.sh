#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v3_stage8_adapter_train.sh <adapter_only|adapter_last_block|joint> [steps]
MODE="${1:-adapter_only}"
STEPS="${2:-50}"
START_CKPT="outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt"
OUT="outputs/v3_stage8_updater_adapter/${MODE}/${STEPS}steps"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
  echo "Stage-7C full-joint checkpoint is missing: ${START_CKPT}" >&2
  exit 1
fi

case "${MODE}" in
  adapter_only)
    ADAPTER_ONLY=true
    UNFREEZE_LAST_BLOCK=false
    LR=5e-5
    ;;
  adapter_last_block)
    ADAPTER_ONLY=true
    UNFREEZE_LAST_BLOCK=true
    LR=1e-5
    ;;
  joint)
    ADAPTER_ONLY=false
    UNFREEZE_LAST_BLOCK=false
    LR=5e-5
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES=0 python -m src.main \
  +experiment=v3_semantic_updater_adapter_dl3dv \
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
  model.encoder.semantic_updater_adapter_train_only="${ADAPTER_ONLY}" \
  model.encoder.semantic_updater_unfreeze_last_block="${UNFREEZE_LAST_BLOCK}" \
  train.print_log_every_n_steps=10 \
  optimizer.lr="${LR}" \
  optimizer.lr_monodepth=0.0 \
  checkpointing.pretrained_model="${START_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1
