#!/usr/bin/env bash
set -euo pipefail

# Evaluate the clean mainline and its component ablations on the same fixed
# eight held-out DL3DV scenes used by the Situation-B validation.
# Usage: bash scripts/paper_clean_semantic_refinement_multiscene_eval.sh \
#          <baseline|identity|trained|b_only|adapter_only> [steps]
MODE="${1:-trained}"
STEPS="${2:-1000}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
INDEX="${PAPER_HELDOUT_INDEX:-assets/dl3dv_evaluation/v4_situation_b_8scene_heldout.json}"
RUN_NAME="${PAPER_RUN_NAME:-clean_joint_multiscene}"
OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-outputs/paper_clean_semantic_refinement}"
START_CKPT="${PAPER_START_CKPT:-outputs/v3_stage7c/full_joint/500steps/checkpoints/epoch_1-step_500.ckpt}"
SAVE_DEMO="${PAPER_SAVE_DEMO:-false}"

ENABLE_B=true
ENABLE_ADAPTER=true
case "${MODE}" in
  baseline)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/eval8_baseline"
    ENABLE_B=false
    ENABLE_ADAPTER=false
    ;;
  identity)
    CHECKPOINT="${START_CKPT}"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/eval8_identity"
    ;;
  trained)
    TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/eval8_trained_${STEPS}steps"
    ;;
  b_only)
    TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/eval8_b_only_${STEPS}steps"
    ENABLE_ADAPTER=false
    ;;
  adapter_only)
    TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"
    CHECKPOINT="$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
    OUT="${OUTPUT_ROOT}/${RUN_NAME}/eval8_adapter_only_${STEPS}steps"
    ENABLE_B=false
    ;;
  *)
    echo "mode must be baseline, identity, trained, b_only, or adapter_only" >&2
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
  +experiment=paper_clean_semantic_refinement_dl3dv mode=test \
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
  model.encoder.use_semantic_gaussian_features="$([[ "${ENABLE_B}" == true || "${ENABLE_ADAPTER}" == true ]] && echo true || echo false)" \
  model.encoder.use_semantic_residual_feedback="${ENABLE_ADAPTER}" \
  model.encoder.use_semantic_residual_adapter="${ENABLE_ADAPTER}" \
  model.encoder.semantic_joint_train_new_heads="${ENABLE_ADAPTER}" \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  test.save_image="${SAVE_DEMO}" \
  test.save_gt_image="${SAVE_DEMO}" \
  test.save_input_images="${SAVE_DEMO}" \
  test.save_initial_image="${SAVE_DEMO}" \
  test.save_semantic="$([[ "${ENABLE_B}" == true || "${ENABLE_ADAPTER}" == true ]] && echo "${SAVE_DEMO}" || echo false)" \
  wandb.mode=disabled \
  output_dir="${OUT}"

echo "metrics=${OUT}/metrics/scores_all_avg.json"
