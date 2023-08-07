CUDA_VISIBLE_DEVICES=1 python train_sequential.py \
    --taps_disabled \
    --train_dataset /export/r32/data/visda17/validation \
    --val_dataset /export/r32/data/visda17/test \
    --dataset_frac 0.1 \
    --experiment_name standard_ft_0.1_frac \
    --model_type resnet50 \
    --lr 0.01 \
    --cropped \
    --batch_size 32