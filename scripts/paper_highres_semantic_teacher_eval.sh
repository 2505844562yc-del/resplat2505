#!/usr/bin/env bash
set -euo pipefail

# Export a held-out RGB/depth/semantic panel for the high-resolution teacher.
STEPS="${1:-300}"
PYTHON="${PYTHON:-/root/miniconda3/envs/resplat/bin/python}"
SCENE="${PAPER_SCENE:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
INDEX="${PAPER_HELDOUT_INDEX:-assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json}"
RUN_NAME="${PAPER_RUN_NAME:-highres_teacher}"
OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-outputs/paper_highres_semantic_teacher}"
TRAIN_OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps"
CHECKPOINT="${PAPER_START_CKPT:-$(find "${TRAIN_OUT}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)}"
OUT="${OUTPUT_ROOT}/${RUN_NAME}/${STEPS}steps_heldout"

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint is missing: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${INDEX}" ]]; then
  echo "held-out index is missing: ${INDEX}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m src.main \
  +experiment=paper_highres_semantic_teacher_dl3dv mode=test \
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
  checkpointing.pretrained_model="${CHECKPOINT}" \
  checkpointing.no_strict_load=true \
  test.save_image=true \
  test.save_gt_image=true \
  test.save_input_images=true \
  test.save_initial_image=true \
  test.save_semantic=true \
  test.save_video=false \
  wandb.mode=disabled \
  output_dir="${OUT}"

echo "metrics=${OUT}/metrics/scores_all_avg.json"
echo "demo_images=${OUT}/images/${SCENE}"
