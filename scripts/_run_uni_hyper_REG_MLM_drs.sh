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
declare -a layers=(0 1 2 3 4 5 6)
declare -a batch_sizes=(64 128 256) # 512) ####
declare -a learning_rates=(1e-5 3e-5 5e-5) # 1e-5 3e-5) #  5e-5)
declare -a devices=(0 1 2) 
declare -a dropout_rates=(0.5 0.25 0.125) #  0.05)
for layer in "${layers[@]}"
do 
    for batch_size in "${batch_sizes[@]}"
    do 
        for dropout_rate in "${dropout_rates[@]}"
            echo $dropout_rate
            device_index=-1
            for _ in "${devices[@]}"
            do
                device_index=$((device_index + 1))

                learning_rate=${learning_rates[$device_index]}
                device=$((devices[device_index]))
                
                #output_dir="/mnt2/brg/simcse-data/HYPER/REG_MLM/REGMLM_L${layer}_b${batch_size}_lr${learning_rate}"
                output_dir="/mnt2/brg/simcse-data/HYPER/Jan13_025REG_MLM/REGMLM_L${layer}_b${batch_size}_lr${learning_rate}_dr${dropout_rate}"
                echo "device ${device} batch_size ${batch_size} output_dir ${output_dir}"
                CUDA_VISIBLE_DEVICES="${device}" python train.py \
                    --transform_layer $layer \
                    --higher_transform_p 0.5 \
                    --higher_dropout_p $dropout_rate \
                    --do_mlm \
                    --attention_probs_dropout_prob 0.25 \
                    --hidden_dropout_prob 0.25 \
                    --model_name_or_path bert-base-uncased \
                    --train_file data/wiki1m_for_simcse.txt \
                    --output_dir $output_dir \
                    --num_train_epochs 1 \
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
done
