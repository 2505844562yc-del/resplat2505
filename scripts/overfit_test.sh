#!/usr/bin/env bash
# Quick overfit test on 1 scene to verify training pipeline
# Usage: bash scripts/overfit_test.sh [DATASET_PATH] [SCENE_NAME]
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate resplat
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd /root/autodl-tmp/resplat

DATASET=${1:-datasets/dl3dv_480p}
SCENE=${2:-}
STEPS=3000

if [ -z "$SCENE" ]; then
    # Get first scene from index.json
    SCENE=$(python -c "import json; d=json.load(open('${DATASET}/train/index.json')); print(list(d.keys())[0])")
    echo "Auto-selected scene: $SCENE"
fi

echo "========================================="
echo "Overfit Test: 1 scene, 3000 steps"
echo "  Scene:   $SCENE"
echo "========================================="

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv     mode=train     dataset.roots=[${DATASET}]     dataset.overfit_to_scene=${SCENE}     dataset.image_shape=[256,448]     dataset.ori_image_shape=[270,480]     dataset.view_sampler.num_context_views=8     dataset.view_sampler.num_target_views=6     trainer.max_steps=${STEPS}     trainer.num_nodes=1     data_loader.train.batch_size=1     model.encoder.num_refine=4     model.encoder.recurrent_use_checkpointing=true     model.encoder.use_semantic=true     model.encoder.use_semantic_gate=true     model.encoder.use_amp=true     checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth     checkpointing.no_strict_load=true     optimizer.lr=1e-4     optimizer.lr_monodepth=0.     optimizer.lr_depth=0.     optimizer.warm_up_steps=100     wandb.mode=disabled     train.no_log_video=true     train.no_log_projections=true     output_dir=checkpoints/overfit-test
