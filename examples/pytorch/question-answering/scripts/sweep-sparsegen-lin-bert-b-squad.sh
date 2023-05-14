#!/usr/bin/env bash

lambda=-7.0
CUDA_VISIBLE_DEVICES=0 python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --sparsegen_lin --sparsegen_lambda $lambda \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-tuned-bert-b-subs-sparsegen_lin_${lambda}-noft \
    --overwrite_output_dir &

lambda=-8.0
CUDA_VISIBLE_DEVICES=1 python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --sparsegen_lin --sparsegen_lambda $lambda \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-tuned-bert-b-subs-sparsegen_lin_${lambda}-noft \
    --overwrite_output_dir &

lambda=-9.0
CUDA_VISIBLE_DEVICES=2 python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --sparsegen_lin --sparsegen_lambda $lambda \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-tuned-bert-b-subs-sparsegen_lin_${lambda}-noft \
    --overwrite_output_dir &

lambda=-10.0
CUDA_VISIBLE_DEVICES=3 python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --sparsegen_lin --sparsegen_lambda $lambda \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-tuned-bert-b-subs-sparsegen_lin_${lambda}-noft \
    --overwrite_output_dir &

# lambda=-5.0
# CUDA_VISIBLE_DEVICES=4 python run_qa.py \
#     --model_name_or_path vuiseng9/bert-base-uncased-squad \
#     --sparsegen_lin --sparsegen_lambda $lambda \
#     --analyze_sparsity --attn_sparsity_only \
#     --dataset_name squad \
#     --do_eval --per_device_eval_batch_size 1 \
#     --max_seq_length 384 \
#     --pad_to_max_length False \
#     --doc_stride 128 \
#     --output_dir ./squad-tuned-bert-b-subs-sparsegen_lin_${lambda}-noft \
#     --overwrite_output_dir &

# lambda=-6.0
# CUDA_VISIBLE_DEVICES=5 python run_qa.py \
#     --model_name_or_path vuiseng9/bert-base-uncased-squad \
#     --sparsegen_lin --sparsegen_lambda $lambda \
#     --analyze_sparsity --attn_sparsity_only \
#     --dataset_name squad \
#     --do_eval --per_device_eval_batch_size 1 \
#     --max_seq_length 384 \
#     --pad_to_max_length False \
#     --doc_stride 128 \
#     --output_dir ./squad-tuned-bert-b-subs-sparsegen_lin_${lambda}-noft \
#     --overwrite_output_dir &