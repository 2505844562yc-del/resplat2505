#!/usr/bin/env bash
# Evaluate fine-tuned model on DL3DV test set
# Usage: bash scripts/eval_finetuned.sh [CHECKPOINT_PATH]
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate resplat
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd /root/autodl-tmp/resplat

CKPT=${1:-checkpoints/finetune-v1/checkpoints/last.ckpt}

echo "=== Evaluating: $CKPT ==="

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv     mode=test     dataset.roots=[datasets/dl3dv_480p]     dataset/view_sampler=evaluation     dataset.view_sampler.num_context_views=8     dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json     dataset.image_shape=[256,448]     model.encoder.num_refine=4     checkpointing.pretrained_model=${CKPT}     checkpointing.no_strict_load=true
