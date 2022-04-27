#!bin/bash

export CUDA_VISIBLE_DEVICES="3"

python train.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/mt5_small_1e-3_8

python train.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/mt5_small_1e-3_16

python train.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/mt5_small_1e-3_32

python train.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/mt5_small_1e-3_4

python train.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/mt5_small_1e-3_2

python train.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --text_column maintext \
    --summary_column title \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --source_prefix "summarize: " \
    --output_dir ./ckpt/mt5_small_1e-3_1
