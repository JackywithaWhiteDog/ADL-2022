#!bin/bash

export CUDA_VISIBLE_DEVICES="5"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# 2.737, 59.639
# python3.8 $SCRIPT_DIR/train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 20 \
#     --output_dir ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# 3.136, 53.583
# python3.8 $SCRIPT_DIR/train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 10 \
#     --output_dir ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# 3.703, 49.667
# python3.8 $SCRIPT_DIR/train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 5 \
#     --output_dir ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# 4.237, 47.304 *
python3.8 $SCRIPT_DIR/train.py \
    --dataset_root ./data \
    --domain in_domain \
    --model_name_or_path ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
    --do_sample \
    --top_k 2 \
    --output_dir ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# python3.8 $SCRIPT_DIR/train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --num_beams 5 \
#     --output_dir ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# 4.177, 49.864
# python3.8 $SCRIPT_DIR/train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --output_dir ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# python3.8 $SCRIPT_DIR/train.py \
#     --do_train \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path t5-small \
#     --learning_rate 5e-3 \
#     --weight_decay 1e-1 \
#     --train_bsize 32 \
#     --output_dir ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75
