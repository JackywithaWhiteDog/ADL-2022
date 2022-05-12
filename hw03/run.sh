#!bin/bash

# export CUDA_VISIBLE_DEVICES="7"

# "${1}": path to the input file.
# "${2}": path to the output file.

python3.8 ./predict.py \
    --model_name_or_path ./ckpt/mt5 \
    --max_source_length 256 \
    --max_target_length 64 \
    --num_beams 5 \
    --top_k 0 \
    --top_p 1 \
    --temperature 1 \
    --text_column maintext \
    --per_device_eval_batch_size 64 \
    --source_prefix "summarize: " \
    --test_file "${1}" \
    --pred_file "${2}"

# python3.8 ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --num_beams 5 \
#     --top_k 0 \
#     --top_p 1 \
#     --temperature 1 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file "${1}" \
#     --pred_file "${2}"

# python3.8 ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_rl \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --num_beams 1 \
#     --top_k 0 \
#     --top_p 1 \
#     --temperature 1 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file "${1}" \
#     --pred_file "${2}"

# python ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_small_5e-4_8 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --do_sample \
#     --num_beams 1 \
#     --top_k 0 \
#     --top_p 0.2 \
#     --temperature 0.2 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file ./data/public.jsonl \
#     --pred_file ./pred/public_p02_t02.jsonl

# python ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_small_5e-4_8 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --do_sample \
#     --num_beams 1 \
#     --top_k 5 \
#     --top_p 1 \
#     --temperature 0.2 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file ./data/public.jsonl \
#     --pred_file ./pred/public_k05_t02.jsonl

# python ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_small_5e-4_8 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --do_sample \
#     --num_beams 1 \
#     --top_k 0 \
#     --top_p 0.2 \
#     --temperature 1 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file ./data/public.jsonl \
#     --pred_file ./pred/public_p02.jsonl

# python ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_small_5e-4_8 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --do_sample \
#     --num_beams 1 \
#     --top_k 2 \
#     --top_p 1 \
#     --temperature 1 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file ./data/public.jsonl \
#     --pred_file ./pred/public_k02.jsonl

# python ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_small_5e-4_8 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --num_beams 5 \
#     --top_k 0 \
#     --top_p 1 \
#     --temperature 1 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file ./data/public.jsonl \
#     --pred_file ./pred/public_beam05.jsonl

# python ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_small_5e-4_8 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --do_sample \
#     --num_beams 1 \
#     --top_k 0 \
#     --top_p 1 \
#     --temperature 1 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file ./data/public.jsonl \
#     --pred_file ./pred/public_sample.jsonl

# python ./main.py \
#     --do_predict \
#     --model_name_or_path ./ckpt/mt5_small_5e-4_8 \
#     --max_source_length 256 \
#     --max_target_length 64 \
#     --num_beams 1 \
#     --top_k 0 \
#     --top_p 1 \
#     --temperature 1 \
#     --text_column maintext \
#     --summary_column title \
#     --per_device_eval_batch_size 64 \
#     --source_prefix "summarize: " \
#     --test_file ./data/public.jsonl \
#     --pred_file ./pred/public_greedy.jsonl
