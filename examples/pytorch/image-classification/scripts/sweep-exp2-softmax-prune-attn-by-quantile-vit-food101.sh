#!/usr/bin/env bash

export WANDB_MODE=disabled

MODEL_ID=nateraw/vit-base-food101

quantile=0.95
CUDA_VISIBLE_DEVICES=0 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.9
CUDA_VISIBLE_DEVICES=1 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.8
CUDA_VISIBLE_DEVICES=2 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.7
CUDA_VISIBLE_DEVICES=3 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.6
CUDA_VISIBLE_DEVICES=4 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir 

quantile=0.5
CUDA_VISIBLE_DEVICES=0 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.4
CUDA_VISIBLE_DEVICES=1 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.3
CUDA_VISIBLE_DEVICES=2 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.2
CUDA_VISIBLE_DEVICES=3 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.1
CUDA_VISIBLE_DEVICES=4 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.0
CUDA_VISIBLE_DEVICES=5 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --softmax_exp2 \
    --prune_attn_by_quantile $quantile \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-exp2-softmax-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &