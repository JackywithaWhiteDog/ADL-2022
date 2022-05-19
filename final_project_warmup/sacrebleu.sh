#!bin/bash

DATA_TYPE=out_of_domain
PRE_TRAINED=gpt2-small

# in_domain_anlg (10, wd=0, lr-5e-5): 5.756, 54.152
# in_domain_anlg (20, wd=0, lr-5e-5, max): 6.128, 51.185
# in_domain_anlg (40, wd=0, lr-5e-5): 6.093, 54.179

# in_domain_anlg (20, wd=0, lr-5e-5, top-k=20): 4.101, 49.857 *
# in_domain_anlg (20, wd=0, lr-5e-5, top-p=.2): 5.056, 53.836

# in_domain_anlg (20, wd=5e-5, lr-5e-5, max): 6.150, 51.843

# in_domain_anlg (20, wd=0, lr-1e-4): 5.956, 51.919

# out_of_domain_anlg (20, wd=0, lr-5e-5): 5.846, 51.224
# out_of_domain_anlg (20, wd=0, lr-5e-5, top-k=5): 4.268, 49.780 *
# out_of_domain_anlg (20, wd=0, lr-5e-5, top-k=10): 3.459, 49.624
# out_of_domain_anlg (20, wd=0, lr-5e-5, top-k=20): 3.477, 52.593

echo "SacreBLEU: $(
    sacrebleu \
        ./data/${DATA_TYPE}/test/reference.txt \
        -i ./models/${DATA_TYPE}/grf-${DATA_TYPE}_${PRE_TRAINED}_eval/result_ep:test.txt \
        -b -m bleu -w 3 --lowercase
)"

# echo "SacreBLEU: $(
#     sacrebleu \
#         ./data/in_domain/test/reference.txt \
#         -i ./ckpt/t5/in_domain/wd1e-1_lr5e-3_batch32_source50_target75/generated_predictions.txt \
#         -b -m bleu -w 3 --lowercase
# )"
