#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/v3_semantic_joint_train.sh [steps] [overfit|multiscene]
STEPS="${1:-20}"
SCOPE="${2:-multiscene}"
SCENE="032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"
OUT="outputs/v3_semantic_joint/${SCOPE}_${STEPS}steps"
STAGE6_CKPT="$(find outputs/v3_semantic_priority_depth/50steps/checkpoints -maxdepth 1 -name '*.ckpt' -print -quit)"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if [[ "${SCOPE}" != "overfit" && "${SCOPE}" != "multiscene" ]]; then
  echo "scope must be overfit or multiscene" >&2
  exit 2
fi
if [[ -z "${STAGE6_CKPT}" ]]; then
  echo "Stage-6 priority-depth checkpoint is missing" >&2
  exit 1
fi

DATASET_ARGS=(dataset.overfit_to_scene=null)
if [[ "${SCOPE}" == "overfit" ]]; then
  DATASET_ARGS=(
    "dataset.overfit_to_scene=${SCENE}"
    dataset.overfit_max_views=32
  )
fi

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
  "${DATASET_ARGS[@]}" \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.num_target_views=2 \
  dataset.view_sampler.min_distance_between_context_views=24 \
  dataset.view_sampler.max_distance_between_context_views=45 \
  dataset.image_shape='[256,448]' \
  train.print_log_every_n_steps=1 \
  optimizer.lr=5e-5 \
  optimizer.lr_monodepth=0.0 \
  checkpointing.pretrained_model="${STAGE6_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1
