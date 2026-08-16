#!/usr/bin/env bash
set -euo pipefail

# Train coherent displacement feedback on the same three Stage 5 scenes and
# evaluate every checkpoint on the same deterministic five-scene test.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STEPS="${1:-200}"
SCENES=(
  "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"
  "073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561"
  "14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49"
)

if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi

run_train() {
  local scene="$1"
  local tag="${scene:0:8}"
  local out="outputs/stage8_minival/train/${tag}/displacement_${STEPS}steps"
  if [[ -f "${out}/metrics/train_complete.txt" ]]; then
    echo "skip completed training: ${tag}"
    return
  fi
  mkdir -p "${out}/metrics"
  bash <(sed \
    -e "s|^SCENE=.*|SCENE=\"${scene}\"|" \
    -e "s|^REPO_ROOT=.*|REPO_ROOT=\"${REPO_ROOT}\"|" \
    -e "s|^  OUT=\"outputs/stage8_displacement/.*|  OUT=\"${out}\"|" \
    scripts/stage2_overfit_ablation.sh) \
    joint "${STEPS}" 0.01 0.35 residual_alignment_displacement \
    2 1.0 false 0.5 false null
  date --iso-8601=seconds > "${out}/metrics/train_complete.txt"
}

run_eval() {
  local scene="$1"
  local tag="${scene:0:8}"
  local train_out="outputs/stage8_minival/train/${tag}/displacement_${STEPS}steps"
  local eval_out="outputs/stage8_minival/eval/${tag}/displacement_${STEPS}steps"
  local checkpoint
  if [[ -f "${eval_out}/metrics/scores_all_avg.json" ]]; then
    echo "skip completed evaluation: ${tag}"
    return
  fi
  checkpoint="$(find "${train_out}/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
  if [[ -z "${checkpoint}" ]]; then
    echo "checkpoint missing: ${train_out}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv mode=test \
    dataset.roots='[datasets/dl3dv_480p]' \
    dataset.test_chunk_interval=1 dataset.test_len=5 \
    dataset.load_boundaries=true \
    dataset.boundary_roots='[datasets/dl3dv_boundaries_sam2_small]' \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
    dataset.pose_align_middle_view=true dataset.image_shape='[256,448]' \
    model.encoder.num_refine=1 \
    model.encoder.use_semantic_boundary_feedback=true \
    model.encoder.semantic_boundary_feedback_scale=0.35 \
    model.encoder.semantic_boundary_feature_mode=residual_alignment_displacement \
    model.encoder.semantic_boundary_alignment_radius=2 \
    model.encoder.semantic_boundary_alignment_sigma=1.0 \
    checkpointing.pretrained_model="${checkpoint}" \
    checkpointing.no_strict_load=true wandb.mode=disabled \
    output_dir="${eval_out}"
}

for scene in "${SCENES[@]}"; do
  run_train "${scene}"
  run_eval "${scene}"
done

echo "Stage 8 minimal multi-scene validation complete"
