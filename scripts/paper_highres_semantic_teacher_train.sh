#!/usr/bin/env bash
set -euo pipefail

# Continue the clean paper mainline with source-resolution DINO supervision.
# The default is a compact multi-scene adaptation from the validated 1000-step
# checkpoint; no original ReSplat parameter is unfrozen.
STEPS="${1:-300}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${PAPER_SCENE:-null}"
LR="${PAPER_LR:-5e-5}"
RUN_NAME="${PAPER_RUN_NAME:-highres_teacher}"
OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-outputs/paper_highres_semantic_teacher}"
START_CKPT="${PAPER_START_CKPT:-outputs/paper_clean_semantic_refinement/clean_joint_multiscene/1000steps/checkpoints/epoch_2-step_1000.ckpt}"
OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
  echo "starting checkpoint is missing: ${START_CKPT}" >&2
  exit 1
fi
if [[ -d "${OUT}/checkpoints" ]] \
  && find "${OUT}/checkpoints" -name '*.ckpt' -print -quit | grep -q .; then
  echo "output already contains a checkpoint: ${OUT}" >&2
  exit 1
fi

LOAD_BOUNDARIES=true
BOUNDARY_WEIGHT=4.0
OVERFIT_MAX_VIEWS=120
if [[ "${SCENE}" == "null" ]]; then
  LOAD_BOUNDARIES=false
  BOUNDARY_WEIGHT=0.0
  OVERFIT_MAX_VIEWS=null
fi

mkdir -p "${OUT}"
STARTED_AT="$(date +%s)"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=paper_highres_semantic_teacher_dl3dv \
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
  dataset.overfit_to_scene="${SCENE}" \
  dataset.overfit_max_views="${OVERFIT_MAX_VIEWS}" \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.num_target_views=2 \
  dataset.view_sampler.min_distance_between_context_views=24 \
  dataset.view_sampler.max_distance_between_context_views=45 \
  dataset.image_shape='[256,448]' \
  dataset.load_boundaries="${LOAD_BOUNDARIES}" \
  model.encoder.semantic_depth_boundary_weight="${BOUNDARY_WEIGHT}" \
  train.print_log_every_n_steps=10 \
  optimizer.lr="${LR}" \
  optimizer.lr_monodepth=0.0 \
  checkpointing.pretrained_model="${START_CKPT}" \
  checkpointing.no_strict_load=true \
  checkpointing.every_n_train_steps="${STEPS}" \
  checkpointing.save_top_k=1

FINISHED_AT="$(date +%s)"
echo "elapsed_seconds=$((FINISHED_AT - STARTED_AT))"
echo "checkpoint_dir=${OUT}/checkpoints"
