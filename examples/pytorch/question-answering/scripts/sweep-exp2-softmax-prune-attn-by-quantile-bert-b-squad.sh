#!/usr/bin/env bash

quantile=0.95
CUDA_VISIBLE_DEVICES=0 python run_qa.py \
    --model_name_or_path /data5/vchua/run/apr23-optimum/sparse-attn-act/exp2-softmax-ft-squad-bert-b-2eph-lr3e-5 \
    --prune_attn_by_quantile $quantile \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-exp2-softmax-tuned-bert-b-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.3
CUDA_VISIBLE_DEVICES=1 python run_qa.py \
    --model_name_or_path /data5/vchua/run/apr23-optimum/sparse-attn-act/exp2-softmax-ft-squad-bert-b-2eph-lr3e-5 \
    --prune_attn_by_quantile $quantile \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-exp2-softmax-tuned-bert-b-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.2
CUDA_VISIBLE_DEVICES=2 python run_qa.py \
    --model_name_or_path /data5/vchua/run/apr23-optimum/sparse-attn-act/exp2-softmax-ft-squad-bert-b-2eph-lr3e-5 \
    --prune_attn_by_quantile $quantile \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-exp2-softmax-tuned-bert-b-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

quantile=0.1
CUDA_VISIBLE_DEVICES=3 python run_qa.py \
    --model_name_or_path /data5/vchua/run/apr23-optimum/sparse-attn-act/exp2-softmax-ft-squad-bert-b-2eph-lr3e-5 \
    --prune_attn_by_quantile $quantile \
    --analyze_sparsity --attn_sparsity_only \
    --dataset_name squad \
    --do_eval --per_device_eval_batch_size 1 \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --output_dir ./squad-exp2-softmax-tuned-bert-b-prune-attn-by-quantile-${quantile}-noft \
    --overwrite_output_dir &

# quantile=0.5
# CUDA_VISIBLE_DEVICES=4 python run_qa.py \
#     --model_name_or_path /data5/vchua/run/apr23-optimum/sparse-attn-act/exp2-softmax-ft-squad-bert-b-2eph-lr3e-5 \
#     --prune_attn_by_quantile $quantile \
#     --analyze_sparsity --attn_sparsity_only \
#     --dataset_name squad \
#     --do_eval --per_device_eval_batch_size 1 \
#     --max_seq_length 384 \
#     --pad_to_max_length False \
#     --doc_stride 128 \
#     --output_dir ./squad-exp2-softmax-tuned-bert-b-prune-attn-by-quantile-${quantile}-noft \
#     --overwrite_output_dir &

# quantile=0.4
# CUDA_VISIBLE_DEVICES=5 python run_qa.py \
#     --model_name_or_path /data5/vchua/run/apr23-optimum/sparse-attn-act/exp2-softmax-ft-squad-bert-b-2eph-lr3e-5 \
#     --prune_attn_by_quantile $quantile \
#     --analyze_sparsity --attn_sparsity_only \
#     --dataset_name squad \
#     --do_eval --per_device_eval_batch_size 1 \
#     --max_seq_length 384 \
#     --pad_to_max_length False \
#     --doc_stride 128 \
#     --output_dir ./squad-exp2-softmax-tuned-bert-b-prune-attn-by-quantile-${quantile}-noft \
#     --overwrite_output_dir &