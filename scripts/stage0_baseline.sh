#!/usr/bin/env bash
set -euo pipefail

# Reproducible Stage-0 evaluation for the official 8-view ReSplat base model.
# Usage: bash scripts/stage0_baseline.sh <num_refine:0-4> [smoke|full]

REFINE="${1:-0}"
SCOPE="${2:-smoke}"

case "${REFINE}" in
  0|1|2|3|4) ;;
  *) echo "num_refine must be one of 0,1,2,3,4" >&2; exit 2 ;;
esac

case "${SCOPE}" in
  smoke) TEST_LEN=1 ;;
  full) TEST_LEN=-1 ;;
  *) echo "scope must be smoke or full" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth"
EXPECTED_SHA="1934a04c1884d31ae81992e5f07694ae06f9565be7694d9bcd073f90c25722be"
ACTUAL_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "Checkpoint SHA-256 mismatch: ${ACTUAL_SHA}" >&2
  exit 3
fi

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "No NVIDIA GPU is visible. Start the 4090D instance before running Stage 0B." >&2
  exit 4
fi

OUT="outputs/stage0/official_256x448_8v/${SCOPE}/refine_${REFINE}"
mkdir -p "${OUT}"

git rev-parse HEAD > "${OUT}/git_commit.txt"
sha256sum "${CHECKPOINT}" > "${OUT}/checkpoint_sha256.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "${OUT}/gpu.txt"

time CUDA_VISIBLE_DEVICES=0 \
python -m src.main +experiment=dl3dv \
  mode=test \
  dataset.roots='[datasets/dl3dv_480p]' \
  dataset.test_chunk_interval=1 \
  dataset.test_len="${TEST_LEN}" \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.num_context_views=8 \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.pose_align_middle_view=true \
  dataset.image_shape='[256,448]' \
  model.encoder.num_refine="${REFINE}" \
  checkpointing.pretrained_model="${CHECKPOINT}" \
  wandb.mode=disabled \
  output_dir="${OUT}" \
  2>&1 | tee "${OUT}/run.log"

