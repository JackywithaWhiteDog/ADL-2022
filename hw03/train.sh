#!bin/bash

export CUDA_VISIBLE_DEVICES="7"

python ./main.py \
    --do_train \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/new
