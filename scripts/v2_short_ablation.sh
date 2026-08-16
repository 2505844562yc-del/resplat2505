#!/usr/bin/env bash
set -euo pipefail

# Four-way V2 comparison on the same scene and training schedule.
# Usage: bash scripts/v2_short_ablation.sh <baseline|v1|init|init_v1> [steps]

MODE="${1:-init_v1}"
STEPS="${2:-20}"

case "${MODE}" in
  baseline)
    VARIANT=baseline
    USE_SEMANTIC_INIT=false
    ;;
  v1)
    VARIANT=joint
    USE_SEMANTIC_INIT=false
    ;;
  init)
    VARIANT=init_only
    USE_SEMANTIC_INIT=true
    ;;
  init_v1)
    VARIANT=init_v1
    USE_SEMANTIC_INIT=true
    ;;
  *)
    echo "mode must be baseline, v1, init, or init_v1" >&2
    exit 2
    ;;
esac

bash scripts/stage2_overfit_ablation.sh \
  "${VARIANT}" "${STEPS}" \
  0.01 0.35 residual_alignment 2 1.0 \
  false 0.5 false null false true false \
  "${USE_SEMANTIC_INIT}" "${MODE}"
