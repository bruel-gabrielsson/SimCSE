#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

# lr 1e-5, 3e-5, 5e-5
# batch 64, 128, 256, 512
# epochs 1, 2
# higher_transform_p 0.5 0.25
# higher_dropout_p 0.5 0.25

# "layer$((${LAYER}+1))" \

#export CUDA_VISIBLE_DEVICES="5"

#declare -a layers=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)

declare -a batch_sizes=(128) ####
declare -a learning_rates=(5e-5)
# 12 4 3 13 2 1 0
declare -a layers=(7 8 9 10 6 5 11)
declare -a devices=(0) 
for learning_rate in "${learning_rates[@]}"
do 
    for batch_size in "${batch_sizes[@]}"
    do 
        device_index=-1
        for _ in "${devices[@]}"
        do
            device_index=$((device_index + 1))

            #learning_rate=${learning_rates[$device_index]}
            layer=${layers[$device_index]}
            device=$((devices[device_index]))
            
            output_dir="/skunk-pod-storage-brg-40mit-2eedu-pvc/DATA/simcse-data/HYPER/T_MLM_LONGS25/TMLMLONGS_L${layer}_b$((2*${batch_size}))_lr${learning_rate}"
            echo "device ${device} batch_size $((2*${batch_size})) output_dir ${output_dir}"
            CUDA_VISIBLE_DEVICES="0,1" python train.py \
                --transform_layer $layer \
                --higher_transform_p 0.5 \
                --higher_dropout_p 0.25 \
                --transform_trainable \
                --do_mlm \
                --attention_probs_dropout_prob 0.1 \
                --hidden_dropout_prob 0.1 \
                --model_name_or_path bert-base-uncased \
                --train_file data/wiki1m_for_simcse.txt \
                --output_dir $output_dir \
                --num_train_epochs 3 \
                --per_device_train_batch_size $batch_size \
                --learning_rate $learning_rate \
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
                "$@" \
                &
        done
        wait
        wait
    done
done