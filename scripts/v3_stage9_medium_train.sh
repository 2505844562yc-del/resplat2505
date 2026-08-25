#!/usr/bin/env bash
set -euo pipefail

# Single-scene Stage 7-9 joint adaptation with a disjoint held-out view range.
# Usage: bash scripts/v3_stage9_medium_train.sh [steps] [checkpoint_interval]
STEPS="${1:-500}"
CHECKPOINT_INTERVAL="${2:-100}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${V3_SCENE:-dl3dv_970a5c674c27b504d592d0a70c496d0e35ab0dc76802fb6e1bf336a4c1fe150a}"
RUN_NAME="${V3_RUN_NAME:-single_scene_joint}"
TRAIN_MAX_VIEWS="${V3_TRAIN_MAX_VIEWS:-120}"
START_CKPT="outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt"
OUT="outputs/v3_stage9_medium/${RUN_NAME}/${STEPS}steps"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if ! [[ "${CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "checkpoint_interval must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
  echo "Stage-7C checkpoint is missing: ${START_CKPT}" >&2
  exit 1
fi
if [[ -d "${OUT}/checkpoints" ]] && find "${OUT}/checkpoints" -name '*.ckpt' -print -quit | grep -q .; then
  echo "output already contains checkpoints: ${OUT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=v3_semantic_fixed_split_dl3dv \
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
  dataset.overfit_max_views="${TRAIN_MAX_VIEWS}" \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.num_target_views=2 \
  dataset.view_sampler.min_distance_between_context_views=24 \
  dataset.view_sampler.max_distance_between_context_views=45 \
  dataset.image_shape='[256,448]' \
  model.encoder.use_semantic_updater_adapter=true \
  model.encoder.semantic_updater_unfreeze_last_block=true \
  model.encoder.use_semantic_fixed_candidate_split=true \
  model.encoder.semantic_split_train_only=false \
  train.print_log_every_n_steps=10 \
  optimizer.lr=1e-5 \
  optimizer.lr_monodepth=0.0 \
  checkpointing.pretrained_model="${START_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${CHECKPOINT_INTERVAL}" \
  checkpointing.save_top_k=5
