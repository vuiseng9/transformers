#!/usr/bin/env bash

export WANDB_MODE=disabled

MODEL_ID=nateraw/vit-base-food101

lambda=-5
CUDA_VISIBLE_DEVICES=0 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-10
CUDA_VISIBLE_DEVICES=1 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-15
CUDA_VISIBLE_DEVICES=2 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-20
CUDA_VISIBLE_DEVICES=3 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-25
CUDA_VISIBLE_DEVICES=4 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir 

lambda=-30
CUDA_VISIBLE_DEVICES=0 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-35
CUDA_VISIBLE_DEVICES=1 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-40
CUDA_VISIBLE_DEVICES=2 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-45
CUDA_VISIBLE_DEVICES=3 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=-50
CUDA_VISIBLE_DEVICES=4 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &

lambda=0.0
CUDA_VISIBLE_DEVICES=5 python run_image_classification.py \
    --model_name_or_path $MODEL_ID \
    --analyze_sparsity --attn_sparsity_only \
    --sparsemax --sparsemax_lambda ${lambda} \
    --ignore_mismatched_sizes --remove_unused_columns False \
    --dataset_name food101 \
    --do_eval --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --output_dir ./food101-tuned-vit-b-sparsemax-lambda-${lambda}-noft \
    --overwrite_output_dir &