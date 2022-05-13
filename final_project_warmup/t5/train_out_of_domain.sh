#!bin/bash

export CUDA_VISIBLE_DEVICES="7"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# {'bleu': 6.510484395182762} 48.718788146972656
# python3.8 $SCRIPT_DIR/train.py \
#     --do_train \
#     --dataset_root ./data \
#     --domain out_of_domain \
#     --model_name_or_path t5-small \
#     --learning_rate 5e-3 \
#     --weight_decay 1e-1 \
#     --train_bsize 32 \
#     --do_sample \
#     --top_k 10 \
#     --output_dir ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75

# {'bleu': 6.315407462150533} 54.75518035888672
python3.8 $SCRIPT_DIR/train.py \
    --dataset_root ./data \
    --domain out_of_domain \
    --model_name_or_path ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75 \
    --do_sample \
    --top_k 20 \
    --output_dir ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75

python3.8 $SCRIPT_DIR/perplexity.py \
    ./ckpt/t5/out_of_domain/wd1e-1_lr5e-3_batch32_source50_target75/generated_dialogues.txt
