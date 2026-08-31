#!/usr/bin/env bash
set -euo pipefail

# Situation B1 medium-run validation on all DL3DV training scenes.
# Boundary sidecars are intentionally disabled: the local cache contains only
# held-out test scenes. Semantic feature supervision remains enabled.
# Usage: bash scripts/v4_situation_b_multiscene_train.sh [steps]
STEPS="${1:-1000}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
LR="${V4_LR:-1e-4}"
RUN_NAME="${V4_RUN_NAME:-b1_all_train_scenes}"
OUTPUT_ROOT="${V4_OUTPUT_ROOT:-outputs/v4_situation_b_multiscene}"
START_CKPT="${V4_START_CKPT:-outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt}"
OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"

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

mkdir -p "${OUT}"
STARTED_AT="$(date +%s)"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=v4_situation_b_direct_concat_dl3dv \
  wandb.mode=disabled \
  output_dir="${OUT}" \
  trainer.max_steps="${STEPS}" \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=null \
  train.eval_model_every_n_val=0 \
  data_loader.train.batch_size=1 \
  data_loader.train.num_workers=2 \
  data_loader.train.persistent_workers=true \
  data_loader.val.num_workers=0 \
  data_loader.val.persistent_workers=false \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene=null \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.num_target_views=2 \
  dataset.view_sampler.min_distance_between_context_views=24 \
  dataset.view_sampler.max_distance_between_context_views=45 \
  dataset.image_shape='[256,448]' \
  dataset.load_boundaries=false \
  model.encoder.semantic_depth_boundary_weight=0.0 \
  train.print_log_every_n_steps=20 \
  model.encoder.semantic_depth_unfreeze_last_layers=false \
  optimizer.lr="${LR}" \
  optimizer.lr_monodepth=0.0 \
  checkpointing.pretrained_model="${START_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1

FINISHED_AT="$(date +%s)"
echo "elapsed_seconds=$((FINISHED_AT - STARTED_AT))"
echo "checkpoint_dir=${OUT}/checkpoints"
