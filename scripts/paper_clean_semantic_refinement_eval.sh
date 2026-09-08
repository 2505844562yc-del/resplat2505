#!/usr/bin/env bash
set -euo pipefail

# Fixed-scene comparison and optional group-meeting demo export.
# Usage: bash scripts/paper_clean_semantic_refinement_eval.sh \
#          <baseline|identity|trained> [steps]
MODE="${1:-identity}"
STEPS="${2:-50}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${PAPER_SCENE:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
INDEX="${PAPER_HELDOUT_INDEX:-assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json}"
RUN_NAME="${PAPER_RUN_NAME:-clean_joint}"
OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-outputs/paper_clean_semantic_refinement}"
START_CKPT="${PAPER_START_CKPT:-outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt}"
SAVE_DEMO="${PAPER_SAVE_DEMO:-true}"

ENABLE_MAINLINE=true
case "${MODE}" in
  baseline)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/baseline_heldout"
    ENABLE_MAINLINE=false
    ;;
  identity)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/identity_heldout"
    ;;
  trained)
    TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps_heldout"
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

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=paper_clean_semantic_refinement_dl3dv mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.overfit_to_scene="${SCENE}" \
  dataset.test_len=1 \
  dataset.test_times_per_scene=1 \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path="${INDEX}" \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  data_loader.test.num_workers=0 \
  data_loader.test.persistent_workers=false \
  model.encoder.use_semantic_depth_direct_concat="${ENABLE_MAINLINE}" \
  model.encoder.use_semantic_gaussian_features="${ENABLE_MAINLINE}" \
  model.encoder.use_semantic_residual_feedback="${ENABLE_MAINLINE}" \
  model.encoder.use_semantic_residual_adapter="${ENABLE_MAINLINE}" \
  model.encoder.semantic_joint_train_new_heads="${ENABLE_MAINLINE}" \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  test.save_image="${SAVE_DEMO}" \
  test.save_gt_image="${SAVE_DEMO}" \
  test.save_input_images="${SAVE_DEMO}" \
  test.save_initial_image="${SAVE_DEMO}" \
  test.save_semantic="$([[ "${ENABLE_MAINLINE}" == true ]] && echo "${SAVE_DEMO}" || echo false)" \
  test.save_video="${SAVE_DEMO}" \
  wandb.mode=disabled \
  output_dir="${OUT}"

echo "metrics=${OUT}/metrics/scores_all_avg.json"
if [[ "${SAVE_DEMO}" == true ]]; then
  echo "demo_images=${OUT}/images/${SCENE}"
  echo "demo_video=${OUT}/videos"
fi
