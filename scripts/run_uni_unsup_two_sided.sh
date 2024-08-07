#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

export CUDA_VISIBLE_DEVICES="3"

declare -a layers=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
for layer in "${layers[@]}"
do 
    python train.py \
        --transform_layer $layer \
        --higher_transform_p 1.0 \
        --model_name_or_path bert-base-uncased \
        --train_file data/wiki1m_for_simcse.txt \
        --output_dir "/mnt2/brg/simcse-data/S2_L${layer}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 64 \
        --learning_rate 3e-5 \
        --max_seq_length 32 \
        --evaluation_strategy steps \
        --metric_for_best_model stsb_spearman \
        --load_best_model_at_end \
        --eval_steps 125 \
        --pooler_type cls \
        --mlp_only_train \
        --overwrite_output_dir \
        --temp 0.05 \
        --do_train \
        --do_eval \
        --fp16 \
        "$@"
    wait
done