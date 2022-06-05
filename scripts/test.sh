#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+10))
NAME='pifu_ov'

# Network configuration

BATCH_SIZE=1
MLP_DIM='257 1024 512 256 128 1'
MLP_DIM_COLOR='513 1024 512 256 128 3'

# Reconstruction resolution
# NOTE: one can change here to reconstruct mesh in a different resolution.
VOL_RES=256

CHECKPOINTS_NETG_PATH='./checkpoints/net_G'
CHECKPOINTS_NETC_PATH='./checkpoints/net_C'

TEST_FOLDER_PATH='./input_images'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./apps/eval.py \
    --name ${NAME} \
    --batch_size ${BATCH_SIZE} \
    --mlp_dim ${MLP_DIM} \
    --mlp_dim_color ${MLP_DIM_COLOR} \
    --num_stack 4 \
    --num_hourglass 2 \
    --resolution ${VOL_RES} \
    --hg_down 'ave_pool' \
    --norm 'group' \
    --norm_color 'group' \
    --test_folder_path ${TEST_FOLDER_PATH} \
    --load_netG_checkpoint_path ${CHECKPOINTS_NETG_PATH} \
    --load_netC_checkpoint_path ${CHECKPOINTS_NETC_PATH}


python3 .\apps\eval.py  --name 'pifu_ov' --batch_size 1 --mlp_dim '257 1024 512 256 128 1' \
    --mlp_dim_color '513 1024 512 256 128 3' \
    --mlp_dim_color '513 1024 512 256 128 3' \
    --num_hourglass 2 --resolution 256 --hg_down 'ave_pool' \
    --norm 'group' --norm_color 'group' --test_folder_path .\input_images  
    --load_netG_checkpoint_path .\checkpoints\net_G \
    --load_netC_checkpoint_path .\checkpoints\net_C
python .\apps\eval.py --name 'pifu_ov' --batch_size 1 --mlp_dim 257 1024 512 256 128 1 --mlp_dim_color 513 1024 512 256 128 3 --mlp_dim_color 513 1024 512 256 128 3 --num_hourglass 2 --resolution 256 --hg_down 'ave_pool' --norm 'group' --norm_color 'group' --test_folder_path .\input_images\