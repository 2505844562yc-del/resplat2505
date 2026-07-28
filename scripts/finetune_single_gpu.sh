#!/usr/bin/env bash
# ReSplat single-GPU fine-tuning script (Phase 1)
# Freezes: depth_predictor (106M) + Point Transformer (97M) + ResNet18 (0.7M)
# Trains: update_module + gaussian_regressor + update_error_attn + update_head (~20M)
# Expected VRAM: ~17GB on RTX 4090 24GB
#
# Usage: bash scripts/finetune_single_gpu.sh [DATASET_PATH] [OUTPUT_DIR] [MAX_STEPS]
# Example: bash scripts/finetune_single_gpu.sh datasets/dl3dv_480p checkpoints/finetune-v1 30000

set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate resplat
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd /root/autodl-tmp/resplat

DATASET=${1:-datasets/dl3dv_480p}
OUTPUT=${2:-checkpoints/finetune-v1}
STEPS=${3:-30000}

echo "========================================="
echo "ReSplat Single-GPU Fine-tuning"
echo "  Dataset: $DATASET"
echo "  Output:  $OUTPUT"
echo "  Steps:   $STEPS"
echo "  Frozen:  depth_predictor + Point Transformer"
echo "  Trainable: ~20M (refinement module only)"
echo "========================================="

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv     mode=train         dataset.roots=[${DATASET}]     dataset.image_shape=[256,448]     dataset.ori_image_shape=[270,480]     dataset.view_sampler.num_context_views=8     dataset.view_sampler.num_target_views=6     dataset.view_sampler.min_distance_between_context_views=24     dataset.view_sampler.max_distance_between_context_views=45     dataset.view_sampler.initial_min_distance_between_context_views=20     dataset.view_sampler.initial_max_distance_between_context_views=30         trainer.max_steps=${STEPS}     trainer.num_nodes=1     data_loader.train.batch_size=1     data_loader.train.num_workers=4         model.encoder.num_refine=4     model.encoder.train_min_refine=1     model.encoder.train_max_refine=4     model.encoder.recurrent_use_checkpointing=true     model.encoder.use_semantic=true     model.encoder.use_semantic_gate=true     model.encoder.use_amp=true         checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth     checkpointing.no_strict_load=true     checkpointing.every_n_train_steps=1000     checkpointing.save_top_k=5         optimizer.lr=1e-4     optimizer.lr_monodepth=0.     optimizer.lr_depth=0.     optimizer.warm_up_steps=500         wandb.mode=disabled     train.no_log_video=true     train.no_log_projections=true         output_dir=${OUTPUT}
