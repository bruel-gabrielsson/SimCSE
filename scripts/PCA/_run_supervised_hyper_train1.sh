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

declare -a layers=(12 13)
#declare -a layers=(3) # 0 1 2 3 4 5 6)
declare -a batch_sizes=(64) # 128 256 512) ####
declare -a learning_rates=(0.0001) # 1e-5 3e-5) #  5e-5)
declare -a devices=(3 4 5) #  1 2) 
declare -a seeds=(1 2 3) # 2 3) # 2) # 42
declare -a dropout_rates=(0) # 0.5 0.25 0.125) #  0.05)


# for seed in "${seeds[@]}"
# do
for layer in "${layers[@]}"
do 
    for batch_size in "${batch_sizes[@]}"
    do 
        device_index=-1
        for _ in "${devices[@]}"
        do
            device_index=$((device_index + 1))

            learning_rate=0.0001 # 3e-5 # ${learning_rates[$device_index]}
            seed=${seeds[$device_index]}
            device=$((devices[device_index]))
            dropout_rate=0 # ${dropout_rates[$device_index]}
            size=1

            #output_dir="/mnt2/brg/simcse-data/HYPER/REG_MLM/REGMLM_L${layer}_b${batch_size}_lr${learning_rate}"
            #output_dir="/skunk-pod-storage-brg-40mit-2eedu-pvc/DATA/simcse-data/HYPER/REG_MLMO_ODA/REGMLMO_L${layer}_b${batch_size}_lr${learning_rate}"
            #output_dir="/skunk-pod-storage-brg-40mit-2eedu-pvc/DATA/simcse-data/HYPER/SUPER_REG_NOV23_S${seed}/SUPREG_${layer}_b${batch_size}_lr${learning_rate}_s${seed}"
            
            output_dir="/mnt2/brg/simcse-data/HYPER/SUPER_PCA${size}_TRAIN_DEC25_S${seed}/SUPREG_L${layer}_dr${dropout_rate}_b${batch_size}_lr${learning_rate}_s${seed}"
            echo "device ${device} batch_size ${batch_size} output_dir ${output_dir}"
            # 
            # --transform_layer $layer \
            # --higher_transform_p 0.5 \
            # --higher_dropout_p 0.5 \
            # 
            CUDA_VISIBLE_DEVICES="${device}" python train.py \
                --transform_layer $layer \
                --higher_transform_p 0.5 \
                --PCA_size $size \
                --transform_trainable \
                --attention_probs_dropout_prob 0.1 \
                --hidden_dropout_prob 0.1 \
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
                --skip_contrastive_loss \
                --overwrite_output_dir \
                --temp 0.05 \
                --do_train \
                --do_train_supervised \
                --do_eval \
                --seed $seed \
                --fp16 \
                "$@" \
                &
        done
        wait
        wait
    done
done
# done
