#!/usr/bin/env bash

# Important:
# 1. Always evaluate with batch size of 1 and --pad_to_max_length False 
#    to avoid the implication of padding and attention mask.
#    BS>1 will have sequences equalized in to max length in the batch
# 2. Required and Verified env - vuiseng9/transformers (v4.30.2-ibert-alt-softmax) - commit id: acc6acff5

HFTX_ROOT=/data5/vchua/dev/ibert-alt-softmax/transformers
EVAL_OUTDIR_ROOT=/data5/vchua/run/ibert-alt-softmax/eval-hfhub

export CUDA_VISIBLE_DEVICES=0

# IBERT MRPC unquantized
model_id=vuiseng9/baseline-ft-mrpc-IRoberta-b-unquantized
cd $HFTX_ROOT/examples/pytorch/text-classification
python run_glue.py \
    --model_name_or_path $model_id \
    --task_name MRPC \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --pad_to_max_length False \
    --output_dir $EVAL_OUTDIR_ROOT/eval-$(echo $model_id | tr / -) \
    --overwrite_output_dir

# IBERT MRPC 8bit
model_id=vuiseng9/baseline-ft-mrpc-IRoberta-b-8bit
cd $HFTX_ROOT/examples/pytorch/text-classification
python run_glue.py \
    --model_name_or_path $model_id \
    --task_name MRPC \
    --ibert_quant_mode \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --pad_to_max_length False \
    --output_dir $EVAL_OUTDIR_ROOT/eval-$(echo $model_id | tr / -) \
    --overwrite_output_dir

# IBERT MRPC 8bit (--hweff_logexp_softmax)
model_id=vuiseng9/baseline-ft-mrpc-IRoberta-b-8bit
cd $HFTX_ROOT/examples/pytorch/text-classification
python run_glue.py \
    --model_name_or_path $model_id \
    --task_name MRPC \
    --ibert_quant_mode \
    --hweff_logexp_softmax \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --pad_to_max_length False \
    --output_dir $EVAL_OUTDIR_ROOT/eval-hweff_logexp_softmax-$(echo $model_id | tr / -) \
    --overwrite_output_dir
