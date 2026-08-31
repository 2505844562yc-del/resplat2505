#!/usr/bin/env bash
set -euo pipefail

# Evaluate Situation B on the fixed eight held-out DL3DV scenes for which the
# local SAM2 boundary cache is complete.
# Usage: bash scripts/v4_situation_b_multiscene_eval.sh <baseline|identity|trained> [steps]
MODE="${1:-baseline}"
STEPS="${2:-1000}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
INDEX="${V4_HELDOUT_INDEX:-assets/dl3dv_evaluation/v4_situation_b_8scene_heldout.json}"
RUN_NAME="${V4_RUN_NAME:-b1_all_train_scenes}"
OUTPUT_ROOT="${V4_OUTPUT_ROOT:-outputs/v4_situation_b_multiscene}"
START_CKPT="${V4_START_CKPT:-outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt}"
ENABLE_B=true

case "${MODE}" in
  baseline)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/eval8/baseline"
    ENABLE_B=false
    ;;
  identity)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/eval8/identity"
    ;;
  trained)
    TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    OUT="${OUTPUT_ROOT}/eval8/trained_${STEPS}steps"
    ;;
  *)
    echo "mode must be baseline, identity, or trained" >&2
    exit 2
    ;;
esac

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint is missing for ${MODE}: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${INDEX}" ]]; then
  echo "held-out index is missing: ${INDEX}" >&2
  exit 1
fi
if [[ ! -d datasets/dl3dv_boundaries_sam2_small/test ]]; then
  echo "held-out boundary cache is missing" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=v4_situation_b_direct_concat_dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene=null \
  dataset.test_len=-1 \
  dataset.test_times_per_scene=1 \
  dataset.test_chunk_interval=1 \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path="${INDEX}" \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  data_loader.test.num_workers=0 \
  data_loader.test.persistent_workers=false \
  model.encoder.use_semantic_depth_direct_concat="${ENABLE_B}" \
  model.encoder.use_semantic_gaussian_features="${ENABLE_B}" \
  model.encoder.semantic_depth_train_only=false \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  wandb.mode=disabled \
  output_dir="${OUT}"

echo "metrics=${OUT}/metrics/scores_all_avg.json"
