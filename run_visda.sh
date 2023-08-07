#!/bin/bash

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set paths and parameters
train_dataset="/cis/net/r32/data/visda17/validation"
val_dataset="/cis/net/r32/data/visda17/test"
dataset_frac=0.1
model_path="checkpoints/synth_pretrain_visda17.pth"
# model_path=""
model_type="resnet50"
batch_size=32

# Define lam values
lam_values=(0.0 0.1 0.25 0.5)

# Run the training script in parallel for each lam value and GPU
for i in "${!lam_values[@]}"
do
    lam=${lam_values[i]}
    gpu=$((i % 4))
    experiment_name="visda_fromsynth_0.1frac_${lam}"
    echo "Running training with lam=$lam on GPU $gpu, experiment name: $experiment_name"
    CUDA_VISIBLE_DEVICES=$gpu python train_sequential.py \
        --train_dataset "$train_dataset" \
        --val_dataset "$val_dataset" \
        --dataset_frac $dataset_frac \
        --model_path "$model_path" \
        --experiment_name "$experiment_name" \
        --model_type "$model_type" \
        --lam $lam \
        --lr 0.01 \
        --cropped \
        --batch_size $batch_size &
done

# Wait for all training processes to finish
wait