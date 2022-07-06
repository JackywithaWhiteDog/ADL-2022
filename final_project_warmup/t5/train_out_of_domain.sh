#!bin/bash

export CUDA_VISIBLE_DEVICES="5"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# 3.512, 54.755
# python3.8 $SCRIPT_DIR/train.py \
#     --dataset_root ./data \
#     --domain out_of_domain \
#     --model_name_or_path ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 20 \
#     --output_dir ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75

# 5.268, 41.321
# python3.8 $SCRIPT_DIR/train.py \
#     --dataset_root ./data \
#     --domain out_of_domain \
#     --model_name_or_path ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 2 \
#     --output_dir ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75

# 5.788, 40.550 *
python3.8 $SCRIPT_DIR/train.py \
    --dataset_root ./data \
    --domain out_of_domain \
    --model_name_or_path ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
    --output_dir ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75

# python3.8 $SCRIPT_DIR/train.py \
#     --do_train \
#     --dataset_root ./data \
#     --domain out_of_domain \
#     --model_name_or_path t5-small \
#     --learning_rate 5e-3 \
#     --weight_decay 1e-1 \
#     --train_bsize 32 \
#     --output_dir ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75
