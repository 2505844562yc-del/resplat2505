#!/usr/bin/env bash
set -euo pipefail

# Strict screenshot Situation A: F_depth' = F_c + gamma * A(F_s).
# Usage: bash scripts/v4_situation_a_train.sh [steps]
STEPS="${1:-20}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${V4_SCENE:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
LR="${V4_LR:-1e-4}"
UNFREEZE_DEPTH_TAIL="${V4_UNFREEZE_DEPTH_TAIL:-false}"
RUN_NAME="${V4_RUN_NAME:-a1_adapter_only}"
OUTPUT_ROOT="${V4_OUTPUT_ROOT:-outputs/v4_situation_a}"
START_CKPT="${V4_START_CKPT:-outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt}"
OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"

if [[ "${UNFREEZE_DEPTH_TAIL}" == "true" ]]; then
  DEFAULT_DEPTH_TAIL_LR="1e-5"
elif [[ "${UNFREEZE_DEPTH_TAIL}" == "false" ]]; then
  DEFAULT_DEPTH_TAIL_LR="0.0"
else
  echo "V4_UNFREEZE_DEPTH_TAIL must be true or false" >&2
  exit 2
fi
DEPTH_TAIL_LR="${V4_DEPTH_TAIL_LR:-${DEFAULT_DEPTH_TAIL_LR}}"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
  echo "pretrained checkpoint is missing: ${START_CKPT}" >&2
  exit 1
fi
if [[ -d "${OUT}/checkpoints" ]] \
  && find "${OUT}/checkpoints" -name '*.ckpt' -print -quit | grep -q .; then
  echo "output already contains a checkpoint: ${OUT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=v4_situation_a_residual_injection_dl3dv \
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
  model.encoder.semantic_depth_unfreeze_last_layers="${UNFREEZE_DEPTH_TAIL}" \
  optimizer.lr="${LR}" \
  optimizer.lr_monodepth="${DEPTH_TAIL_LR}" \
  checkpointing.pretrained_model="${START_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1
