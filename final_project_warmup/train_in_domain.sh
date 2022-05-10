#!bin/bash

export CUDA_VISIBLE_DEVICES="7"

# {'bleu': 6.12241281607659} 59.63908767700195
python3.8 ./train.py \
    --dataset_root ./data \
    --domain in_domain \
    --model_name_or_path ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
    --do_sample \
    --top_k 20 \
    --output_dir ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75
python3.8 ./perplexity.py \
    ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75/generated_dialogues.txt

# {'bleu': 6.14132818406183} 53.58314514160156
# python3.8 ./train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 10 \
#     --output_dir ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# {'bleu': 6.203102841662676} 49.66691589355469
# python3.8 ./train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 5 \
#     --output_dir ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# {'bleu': 6.144379208236427} 47.30373764038086
# python3.8 ./train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --do_sample \
#     --top_k 2 \
#     --output_dir ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# {'bleu': 3.3808855105613413} 49.86359786987305
# python3.8 ./train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
#     --output_dir ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# {'bleu': 3.3162845356325743} 50.84589385986328
# python3.8 ./train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path t5-small \
#     --learning_rate 5e-3 \
#     --weight_decay 1e-1 \
#     --train_bsize 32 \
#     --output_dir ./ckpt/in_domain/wd1e-1_lr5e-3_batch32_source50_target75

# {'bleu': 3.7732790685864552} 47.3178825378418
# python3.8 ./train.py \
#     --dataset_root ./data \
#     --domain in_domain \
#     --model_name_or_path t5-small \
#     --learning_rate 5e-3 \
#     --train_bsize 32 \
#     --output_dir ./ckpt/in_domain/wd2_lr5e-3_batch32_target60