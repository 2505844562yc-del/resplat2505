#!/usr/bin/env bash
set -euo pipefail

# Short adapter-only training for Situation A.
# Usage: bash scripts/v4_semantic_depth_init_a_train.sh [steps]
STEPS="${1:-20}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${V4_SCENE:-dl3dv_970a5c674c27b504d592d0a70c496d0e35ab0dc76802fb6e1bf336a4c1fe150a}"
LR="${V4_LR:-5e-5}"
EXPERIMENT="${V4_EXPERIMENT:-v4_semantic_depth_init_a_dl3dv}"
RUN_NAME="${V4_RUN_NAME:-adapter_only}"
OUTPUT_ROOT="${V4_OUTPUT_ROOT:-outputs/v4_semantic_depth_init_a}"
START_CKPT="outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt"
OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
  echo "V3 checkpoint is missing: ${START_CKPT}" >&2
  exit 1
fi
if [[ -d "${OUT}/checkpoints" ]] \
  && find "${OUT}/checkpoints" -name '*.ckpt' -print -quit | grep -q .; then
  echo "output already contains a checkpoint: ${OUT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment="${EXPERIMENT}" \
  wandb.mode=disabled \
  output_dir="${OUT}" \
  trainer.max_steps="${STEPS}" \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=null \
  train.eval_model_every_n_val=0 \
  data_loader.train.batch_size=1 \
  data_loader.train.num_workers=0 \
  data_loader.train.persistent_workers=false \
  data_loader.val.num_workers=0 \
  data_loader.val.persistent_workers=false \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene="${SCENE}" \
  dataset.overfit_max_views=120 \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.num_target_views=2 \
  dataset.view_sampler.min_distance_between_context_views=24 \
  dataset.view_sampler.max_distance_between_context_views=45 \
  dataset.image_shape='[256,448]' \
  train.print_log_every_n_steps=1 \
  optimizer.lr="${LR}" \
  optimizer.lr_monodepth=0.0 \
  checkpointing.pretrained_model="${START_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1
